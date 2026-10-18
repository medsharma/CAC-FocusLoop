"""Tests for the realistic SYNTHETIC generator, the extractor port and the augmentations."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/synth"))
sys.path.insert(0, str(ROOT / "ml/common"))

import augment as A  # noqa: E402
import live_extractor as X  # noqa: E402
import synthlib as S  # noqa: E402

ZONE = (0.0, 0.0, 0.44, 0.31)


def _inputs(seconds, hz=30, blinks=(), face_off=(), jaw=None, pitch=None):
    n = int(seconds * hz)
    t = np.arange(n) / hz
    face = np.ones(n, bool)
    for a, b in face_off:
        face[(t >= a) & (t < b)] = False
    shapes = {k: np.zeros(n) for k in X.FACE_KEYS}
    for a, d, v in blinks:
        m = (t >= a) & (t < a + d)
        shapes["eyeBlinkLeft"][m] = v; shapes["eyeBlinkRight"][m] = v
    if jaw is not None:
        shapes["jawOpen"] = jaw(t)
    pose = {k: np.zeros(n) for k in X.POSE_KEYS}
    if pitch is not None:
        pose["headPitch"] = pitch(t)
    mt = t[::3]
    return t, face, shapes, pose, mt, np.zeros(len(mt), bool), np.zeros(len(mt))


def _frame(F, k, t):
    return F[k][int(round(t * 10))]


def test_blink_counters_warm_up_and_decay_like_the_extractor():
    t, face, shapes, pose, mt, pk, acc = _inputs(130, blinks=[(5.0, 0.2, 0.9), (20.0, 0.7, 0.95)])
    F = X.extract(t, face, shapes, pose, ZONE, mt, pk, acc, [])
    # counted when the blink ENDS, rate divided by elapsed time during the first 60 s
    assert _frame(F, "blinkRate", 5.1) == 0
    assert _frame(F, "blinkRate", 6.0) == pytest.approx(1 / 6.1 * 60, rel=1e-3)
    assert _frame(F, "longBlinkCount", 21.0) == 1 and _frame(F, "blinkRate", 30.0) == pytest.approx(2 / 30.1 * 60, rel=1e-3)
    assert _frame(F, "blinkMeanDuration", 30.0) == pytest.approx((0.2 + 0.7) / 2, abs=0.05)
    # leaves the trailing 60 s window 60 s after its ONSET
    assert _frame(F, "blinkRate", 64.9) == pytest.approx(2.0)   # window 4.9..64.9 still holds both
    assert _frame(F, "blinkRate", 65.1) == pytest.approx(1.0)
    assert _frame(F, "longBlinkCount", 80.1) == 0 and _frame(F, "blinkRate", 81.0) == 0
    # PERCLOS: closed frames over tracked frames
    assert _frame(F, "perclos", 21.0) == pytest.approx(9 / 211, abs=0.01)


def test_face_loss_semantics():
    t, face, shapes, pose, mt, pk, acc = _inputs(20, face_off=[(10.0, 12.0)])
    pose["gazeYaw"][:] = 0.8          # outside the zone the whole time
    F = X.extract(t, face, shapes, pose, ZONE, mt, pk, acc, [])
    dev = _frame(F, "gazeDeviation", 9.9)
    assert dev > 0
    assert _frame(F, "faceDetected", 11.0) == 0 and _frame(F, "inWorkZone", 11.0) == 0
    assert _frame(F, "gazeDeviation", 11.0) == pytest.approx(dev)       # held
    assert _frame(F, "gazeYaw", 11.0) == 0 and _frame(F, "headMotion", 11.0) == 0
    # dwell keeps growing through the face loss
    assert _frame(F, "offZoneDwell", 11.0) == pytest.approx(_frame(F, "offZoneDwell", 9.9) + 1.1, abs=0.05)
    # perclos denominator excludes face-less frames: still 0 closed / tracked
    assert _frame(F, "perclos", 11.5) == 0


def test_yawns_need_1_5_s_and_nods_need_recovery():
    jaw = lambda t: np.where((t >= 5) & (t < 7), 0.8, 0.0) + np.where((t >= 15) & (t < 16), 0.8, 0.0)
    pitch = lambda t: -0.4 * (((t >= 30) & (t < 30.8)).astype(float)) - 0.4 * ((t >= 40).astype(float))
    t, face, shapes, pose, mt, pk, acc = _inputs(60, jaw=jaw, pitch=pitch)
    F = X.extract(t, face, shapes, pose, ZONE, mt, pk, acc, [])
    assert _frame(F, "yawnCount", 6.4) == 0 and _frame(F, "yawnCount", 6.6) == 1
    assert _frame(F, "yawnActive", 15.5) == 1 and _frame(F, "yawnCount", 20) == 1   # laugh: open < 1.5 s
    assert _frame(F, "nodCount", 31.0) == 1
    assert _frame(F, "nodCount", 59.0) == 1        # head stays down from 40 s: not a nod


def test_heart_rate_staleness_and_baseline():
    t, face, shapes, pose, mt, pk, acc = _inputs(20)
    hr = [(10.0, 70.0, 0.8, None), (11.0, 72.0, 0.9, 65.0)]
    F = X.extract(t, face, shapes, pose, ZONE, mt, pk, acc, hr)
    assert _frame(F, "heartRate", 9.9) == 0
    assert _frame(F, "heartRate", 10.5) == 70 and _frame(F, "heartRateDelta", 10.5) == 0   # no baseline yet
    assert _frame(F, "heartRateDelta", 12.0) == pytest.approx(7.0)
    assert _frame(F, "heartRate", 16.0) == 72 and _frame(F, "heartRate", 16.2) == 0        # > 5 s old


def test_live_generator_session_properties():
    subj = S.LiveSubject.sample(3, np.random.default_rng([0, 3]))
    t, lab, true, F, info = S.synth_session_live(900, np.random.default_rng(1), subj)
    assert set(F) == set(S.FEATS) and all(len(v) == len(t) == 9000 for v in F.values())
    assert all(np.isfinite(v).all() for v in F.values())
    assert 0.5 < info["focused_frac"] < 0.97
    assert info["label_noise_frac"] < 0.2
    assert set(np.unique(true)) <= set(S.STATES)
    for k in ("eyeBlinkLeft", "jawOpen", "perclos", "browTension", "heartRateQuality"):
        assert F[k].min() >= 0 and F[k].max() <= 1
    # face-less frames carry no face values
    nf = F["faceDetected"] == 0
    assert np.all(F["jawOpen"][nf] == 0) and np.all(F["inWorkZone"][nf] == 0)


def test_gradual_onsets_and_labels():
    subj = S.LiveSubject.sample(5, np.random.default_rng([0, 5]))
    eps = [S.Episode(60, 120, "drowsy", ramp_in=10, ramp_out=6)]
    t, lab, true, F, _ = S.render_session(180, np.random.default_rng(2), subj, eps, label_noise=0)
    assert (true[(t >= 60) & (t < 120)] == "drowsy").all() and (true[t < 60] == "focused").all()
    assert (lab == true).all()
    i = S._intensity(np.array([57.0, 60.0, 65.0, 125.0]), eps[0])
    assert 0 < i[0] < 0.5 and i[1] == pytest.approx(0.5) and 0.5 < i[2] <= 1 and i[3] == 0


def test_augment_keeps_extractor_semantics():
    rng = np.random.default_rng(0)
    subj = S.LiveSubject.sample(1, np.random.default_rng([0, 1]))
    _, _, _, F, _ = S.synth_session_live(120, np.random.default_rng(3), subj)
    M = np.stack([F[k] for k in S.FEATS], 1).astype(np.float32)
    Xb = np.stack([M[i - 130:i] for i in range(200, 1200, 40)])
    std = M.std(0) + 1e-3
    Y = A.augment(Xb, 100, std, rng)
    assert Y.shape == (len(Xb), 100, 42) and np.isfinite(Y).all()
    nf = Y[..., 0] < 0.5
    assert np.all(Y[..., 1:25][nf] == 0)                      # no face -> no blendshapes / pose
    assert set(np.unique(Y[..., A.BOOL])) <= {0.0, 1.0}
    assert Y[..., A.UNIT].min() >= 0 and Y[..., A.UNIT].max() <= 1
    assert np.array_equal(A.augment(Xb, 100, std, rng, strength=0.0), Xb[:, -100:])
