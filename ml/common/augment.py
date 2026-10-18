"""Training-time augmentations for raw FeatureFrame windows (shared by ml/state and ml/forecast).

All ops work on RAW feature values (schema order) and respect the extractor's semantics, so an
augmented window still looks like something AttentionFeatureExtractor could emit:
  * time warp      - resample the window at 0.8-1.25x speed (end frame anchored, extra history used);
  * offset / scale - per window, per feature (calibration offsets, per-person baselines, gain errors),
                     only on frames where the feature is defined (face present / HR available);
  * Gaussian noise - per frame, on the same masks;
  * face dropout   - 0.2-3 s spans with no face: face fields 0, inWorkZone 0, gazeDeviation held,
                     offZoneDwell keeps growing, headMotion / browTension / yawnActive 0;
  * HR dropout     - spans (or the whole window) with heartRate / quality / delta = 0, or no baseline
                     (delta = 0 while HR is present).
Numpy, vectorised over the batch except for the span bookkeeping. Deterministic given `rng`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATS = [f["name"] for f in json.loads((ROOT / "ml/schema/feature_schema.json").read_text())["features"]]
I = {k: i for i, k in enumerate(FEATS)}
HZ = 10

BOOL = [I[k] for k in ("faceDetected", "inWorkZone", "yawnActive", "phonePickup")]
BLEND = [I[k] for k in FEATS[1:20]]                       # ARKit blendshapes (0-1, face only)
POSE = [I[k] for k in ("gazeYaw", "gazePitch", "headYaw", "headPitch", "headRoll")]
FACE_CONT = BLEND + POSE + [I["headMotion"], I["browTension"]]
COUNTERS = [I[k] for k in ("blinkRate", "blinkMeanDuration", "longBlinkCount", "perclos", "yawnCount", "nodCount")]
UNIT = BLEND + [I[k] for k in ("perclos", "browTension", "heartRateQuality")]
NONNEG = COUNTERS + [I[k] for k in ("gazeDeviation", "offZoneDwell", "motionMagnitude", "headMotion", "heartRate")]
FACE_ZERO = BLEND + POSE + [I[k] for k in ("faceDetected", "inWorkZone", "yawnActive", "headMotion", "browTension")]
HR3 = [I["heartRate"], I["heartRateQuality"], I["heartRateDelta"]]


def time_warp(X: np.ndarray, out_len: int, rng, p: float = 0.5, lo: float = 0.8, hi: float = 1.25):
    """X (B, Tctx, F) with Tctx >= out_len -> (B, out_len, F). The last frame stays the last frame;
    with probability p the time axis is stretched by r ~ logU(lo, hi) (+/-10% smooth modulation),
    reading older frames from the extra history (clamped at the first frame = front padding)."""
    B, T, Fn = X.shape
    tau = np.arange(out_len)[None, :]
    back = (out_len - 1 - tau).astype(float)                 # frames before the end, (1, out_len)
    r = np.where(rng.random(B) < p, np.exp(rng.uniform(np.log(lo), np.log(hi), B)), 1.0)[:, None]
    ph = rng.uniform(0, 2 * np.pi, B)[:, None]
    mod = 1 + 0.1 * (r != 1) * np.sin(2 * np.pi * back / out_len + ph)
    d = back * r * mod
    d = np.maximum.accumulate(d[:, ::-1], axis=1)[:, ::-1]   # keep it monotone
    src = np.clip(T - 1 - d, 0, T - 1)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (src - i0)[..., None]
    b = np.arange(B)[:, None]
    Y = X[b, i0] * (1 - w) + X[b, i1] * w
    near = np.where(w[..., 0] < 0.5, i0, i1)
    Y[..., BOOL] = X[b, near][..., BOOL]
    # face-less frames must stay exactly face-less (no blending of zeros with values)
    nf = Y[..., I["faceDetected"]] < 0.5
    Y[..., FACE_ZERO] = np.where(nf[..., None], 0.0, Y[..., FACE_ZERO])
    return Y.astype(np.float32)


def offset_scale(X, std, rng, p: float = 0.8):
    """Per-window, per-feature gain and offset, applied only where the feature is defined."""
    B, T, Fn = X.shape
    on = (rng.random(B) < p)[:, None, None]
    face = X[..., I["faceDetected"]][..., None] > 0.5
    hr = X[..., I["heartRate"]][..., None] > 0
    Y = X.copy()
    g = 1 + rng.normal(0, 0.08, (B, 1, Fn)); o = rng.normal(0, 0.25, (B, 1, Fn)) * std[None, None, :]
    fc = np.zeros(Fn, bool); fc[FACE_CONT] = True
    Y = np.where(on & face & fc, Y * g + o, Y)
    pose = np.zeros(Fn, bool); pose[POSE] = True
    Y = np.where(on & face & pose, Y + rng.normal(0, 0.05, (B, 1, Fn)), Y)        # calibration offsets
    cnt = np.zeros(Fn, bool); cnt[COUNTERS] = True
    Y = np.where(on & cnt, Y * (1 + rng.normal(0, 0.12, (B, 1, Fn))), Y)
    mot = np.zeros(Fn, bool); mot[[I["motionMagnitude"], I["headMotion"]]] = True
    Y = np.where(on & mot, Y * np.exp(rng.normal(0, 0.3, (B, 1, Fn))), Y)
    hb = rng.normal(0, 4, (B, 1)); be = rng.normal(0, 3, (B, 1))
    m = (on[..., 0] & hr[..., 0])
    Y[..., I["heartRate"]] = np.where(m, Y[..., I["heartRate"]] + hb, Y[..., I["heartRate"]])
    dl = Y[..., I["heartRateDelta"]]
    Y[..., I["heartRateDelta"]] = np.where(m & (dl != 0), dl + hb + be, dl)
    Y[..., I["heartRateQuality"]] = np.where(m, Y[..., I["heartRateQuality"]] * (1 + rng.normal(0, 0.1, (B, 1))),
                                            Y[..., I["heartRateQuality"]])
    return Y


def gaussian_noise(X, std, rng, p: float = 0.5, scale: float = 0.03):
    B, T, Fn = X.shape
    on = (rng.random(B) < p)[:, None, None]
    face = X[..., I["faceDetected"]][..., None] > 0.5
    fc = np.zeros(Fn, bool); fc[FACE_CONT] = True
    return np.where(on & face & fc, X + rng.normal(0, 1, X.shape) * std[None, None, :] * scale, X)


def face_dropout(X, rng, p: float = 0.3, min_len: int = 2, max_len: int = 30):
    """0.2-3 s face-loss spans, applied the way the extractor reports them."""
    Y = X.copy()
    T = X.shape[1]
    for b in np.nonzero(rng.random(len(X)) < p)[0]:
        for _ in range(rng.integers(1, 3)):
            L = int(rng.integers(min_len, max_len + 1))
            a = int(rng.integers(0, max(1, T - L + 1)))
            e = min(T, a + L)
            y = Y[b]
            prev_dev = y[a - 1, I["gazeDeviation"]] if a > 0 else y[a, I["gazeDeviation"]]
            prev_dw = y[a - 1, I["offZoneDwell"]] if a > 0 else y[a, I["offZoneDwell"]]
            y[a:e, FACE_ZERO] = 0.0
            y[a:e, I["gazeDeviation"]] = prev_dev
            dw = prev_dw + 0.1 * np.arange(1, e - a + 1)
            y[a:e, I["offZoneDwell"]] = dw
            # dwell keeps counting after the span while the user is still outside the zone
            j = e
            carry = dw[-1] + 0.1
            while j < T and y[j, I["offZoneDwell"]] > 0:
                y[j, I["offZoneDwell"]] = carry; carry += 0.1; j += 1
    return Y


def hr_dropout(X, rng, p: float = 0.25, p_all: float = 0.08, p_nobase: float = 0.08):
    Y = X.copy()
    T = X.shape[1]
    r = rng.random(len(X))
    for b in np.nonzero(r < p)[0]:
        L = int(rng.integers(10, max(11, T)))
        a = int(rng.integers(0, max(1, T - L + 1)))
        Y[b, a:a + L][:, HR3] = 0.0
    for b in np.nonzero((r >= p) & (r < p + p_all))[0]:   # no HR in the whole window
        Y[b][:, HR3] = 0.0
    nb = np.nonzero(rng.random(len(X)) < p_nobase)[0]
    Y[nb, :, I["heartRateDelta"]] = 0.0
    return Y


def clip_ranges(X):
    X[..., UNIT] = np.clip(X[..., UNIT], 0, 1)
    X[..., NONNEG] = np.maximum(X[..., NONNEG], 0)
    return X


def augment(Xctx: np.ndarray, out_len: int, std: np.ndarray, rng, strength: float = 1.0) -> np.ndarray:
    """Full pipeline: (B, Tctx, 42) raw with extra history -> (B, out_len, 42) augmented raw."""
    X = time_warp(Xctx, out_len, rng, p=0.5 * strength)
    X = offset_scale(X, std, rng, p=0.8 * strength)
    X = gaussian_noise(X, std, rng, p=0.5 * strength)
    X = face_dropout(X, rng, p=0.3 * strength)
    X = hr_dropout(X, rng, p=0.25 * strength, p_all=0.08 * strength, p_nobase=0.08 * strength)
    return clip_ranges(X).astype(np.float32)
