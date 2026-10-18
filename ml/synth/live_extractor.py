"""Python port of Modules/Perception/AttentionFeatureExtractor.swift (+ PerceptionEventDetectors.swift).

The realistic SYNTHETIC generator (synthlib.render_session) simulates camera-rate face samples,
Core Motion samples and 1 Hz rPPG readings, then runs them through this port so every 10 Hz
FeatureFrame column is computed the way the app computes it live:

  * blinks: hysteresis 0.5 / 0.3 on mean(eyeBlinkLeft, eyeBlinkRight) at camera rate; a blink is
    added when it ENDS (offset) and leaves the trailing 60 s window 60 s after its ONSET; an
    in-progress blink is abandoned when the face is lost. Lids resting above 0.3 merge blinks.
  * blinkRate = blinks in window / min(elapsed, 60 s) * 60, elapsed = t - t0 + 0.1 (warm-up).
  * perclos = closed (mean closure > 0.8) / tracked 10 Hz frames in the trailing 60 s (frames with
    no face are not in the denominator).
  * yawn: jawOpen > 0.6 sustained >= 1.5 s, event at confirmation; yawnActive = jaw > 0.6 now.
  * nod: pitch drop >= 0.25 below a slow (tau 4 s) baseline, recovery within 2 s; event at recovery.
  * no face (or a sample older than 0.5 s): face fields 0, inWorkZone 0, gazeDeviation HOLDS its
    last value, offZoneDwell keeps growing, headMotion / browTension / yawnActive 0.
  * headMotion = EMA (tau 0.1 s) of |dyaw/dt| + |dpitch/dt| between consecutive tracked samples.
  * browTension = mean of mean(browDownL, browDownR) over the trailing 2 s of tracked samples.
  * motionMagnitude = EMA (tau 0.3 s) of user acceleration; phonePickup = detector state.
  * heart rate: the latest rPPG reading if it is <= 5 s old and bpm > 0; delta needs a baseline.

Everything this produces from simulated input is SYNTHETIC.
"""
from __future__ import annotations

import math

import numpy as np

FACE_KEYS = ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
             "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
             "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawOpen", "browDownLeft",
             "browDownRight", "browInnerUp", "mouthClose"]
POSE_KEYS = ["gazeYaw", "gazePitch", "headYaw", "headPitch", "headRoll"]
HEAD_WEIGHT = 0.3


def deviation(yaw, pitch, zone):
    """WorkZoneMath.deviation on arrays; zone = (centerYaw, centerPitch, halfWidthYaw, halfWidthPitch)."""
    cy, cp, hy, hp = zone
    dy = np.maximum(0, np.abs(yaw - cy) - hy)
    dp = np.maximum(0, np.abs(pitch - cp) - hp)
    return np.sqrt(dy * dy + dp * dp)


def _blinks(t, closure, face):
    """PerceptionBlinkDetector over camera samples. Returns (onsets, offsets) arrays."""
    on, off = [], []
    onset = None
    for ti, c, f in zip(t.tolist(), closure.tolist(), face.tolist()):
        if not f:
            onset = None
            continue
        if onset is not None:
            if c < 0.3:
                on.append(onset); off.append(ti); onset = None
        elif c > 0.5:
            onset = ti
    return np.array(on), np.array(off)


def _yawns(t, jaw, face):
    """PerceptionYawnDetector. Returns (event times, isOpen per sample)."""
    ev, is_open = [], np.zeros(len(t), bool)
    since, counted = None, False
    for i, (ti, j, f) in enumerate(zip(t.tolist(), jaw.tolist(), face.tolist())):
        if not f:
            since, counted = None, False
            continue
        if j > 0.6:
            if since is None:
                since, counted = ti, False
            if not counted and ti - since >= 1.5 - 1e-9:
                counted = True
                ev.append(ti)
        else:
            since, counted = None, False
        is_open[i] = since is not None
    return np.array(ev), is_open


def _nods(t, pitch, face):
    """PerceptionNodDetector (fed tracked samples only; never reset on face loss, as in Swift)."""
    ev = []
    base = last_t = dip = None
    deep = expired = False
    for ti, p, f in zip(t.tolist(), pitch.tolist(), face.tolist()):
        if not f:
            continue
        if base is None or last_t is None:
            base, last_t = p, ti
            continue
        dt = max(0.0, ti - last_t)
        last_t = ti
        below = p < base - 0.1
        if dip is None:
            if below:
                dip, deep, expired = ti, p <= base - 0.25, False
            else:
                base += (1 - math.exp(-dt / 4.0)) * (p - base)
            continue
        if p <= base - 0.25:
            deep = True
        if ti - dip > 2.0:
            expired = True
        if not below:
            if deep and not expired:
                ev.append(ti)
            dip, deep, expired = None, False, False
            continue
        if expired:
            base += (1 - math.exp(-dt / 4.0)) * (p - base)
    return np.array(ev)


def _head_motion(t, yaw, pitch, face):
    """Smoothed head angular speed per camera sample (state value, held through face loss)."""
    out = np.zeros(len(t))
    hm, prev = 0.0, None
    for i, (ti, y, p, f) in enumerate(zip(t.tolist(), yaw.tolist(), pitch.tolist(), face.tolist())):
        if not f:
            prev = None
        else:
            if prev is not None and ti > prev[0]:
                dt = ti - prev[0]
                speed = (abs(y - prev[1]) + abs(p - prev[2])) / dt
                hm += (1 - math.exp(-dt / 0.1)) * (speed - hm)
            prev = (ti, y, p)
        out[i] = hm
    return out


def _window_count(ev_add, ev_key, frame_t, w=60.0):
    """Number of events added at ev_add <= t whose key (onset / event time) >= t - w."""
    if len(ev_add) == 0:
        return np.zeros(len(frame_t))
    ev_add, ev_key = np.asarray(ev_add), np.asarray(ev_key)
    out = np.zeros(len(frame_t))
    # events sorted by key; count key in [t-w, t] and added <= t (add >= key always)
    order = np.argsort(ev_key)
    k, a = ev_key[order], ev_add[order]
    lo = np.searchsorted(k, frame_t - w - 1e-9, side="left")
    hi = np.searchsorted(k, frame_t + 1e-9, side="right")
    for i, (l, h) in enumerate(zip(lo, hi)):
        if h > l:
            out[i] = np.count_nonzero(a[l:h] <= frame_t[i] + 1e-9)
    return out


def extract(cam_t, face, shapes, pose, zone, motion_t, picked, accel, hr, frame_hz=10, n_frames=None):
    """Run the extractor over simulated inputs.

    cam_t (n,) camera sample times; face (n,) bool; shapes {FACE_KEYS: (n,)}; pose {POSE_KEYS: (n,)}
    (gaze in camera space, i.e. head + eye, as WorkZoneMath.gazeDirection returns).
    motion_t (m,), picked (m,) bool, accel (m,) g. hr: list of (t, bpm, quality, baseline or None).
    Returns {feature: (n_frames,)} at frame_hz, frames at k / frame_hz.
    """
    cam_t = np.asarray(cam_t, float)
    n_frames = n_frames or int(round(cam_t[-1] * frame_hz)) + 1
    ft = np.arange(n_frames) / frame_hz
    # latest camera sample at or before each frame tick
    ci = np.clip(np.searchsorted(cam_t, ft + 1e-9, side="right") - 1, 0, len(cam_t) - 1)
    fresh = face[ci] & (ft - cam_t[ci] <= 0.5 + 1e-9)
    F = {}
    F["faceDetected"] = fresh.astype(float)
    for k in FACE_KEYS:
        F[k] = np.where(fresh, shapes[k][ci], 0.0)
    for k in POSE_KEYS:
        F[k] = np.where(fresh, pose[k][ci], 0.0)

    # work zone, deviation hold and dwell (frame loop: dwell / hold are stateful)
    comb_y = pose["gazeYaw"][ci] + HEAD_WEIGHT * pose["headYaw"][ci]
    comb_p = pose["gazePitch"][ci] + HEAD_WEIGHT * pose["headPitch"][ci]
    dev_raw = deviation(comb_y, comb_p, zone)
    dev = np.zeros(n_frames); dwell = np.zeros(n_frames); inside = np.zeros(n_frames)
    last_dev, d = 0.0, 0.0
    dt = 1.0 / frame_hz
    for i in range(n_frames):
        if fresh[i]:
            last_dev = float(dev_raw[i])
            ins = last_dev == 0
            inside[i] = 1.0 if ins else 0.0
            d = 0.0 if ins else d + dt
        else:
            d += dt
        dev[i] = last_dev; dwell[i] = d
    F["gazeDeviation"], F["inWorkZone"], F["offZoneDwell"] = dev, inside, dwell

    # event detectors at camera rate
    closure = (shapes["eyeBlinkLeft"] + shapes["eyeBlinkRight"]) / 2
    b_on, b_off = _blinks(cam_t, closure, face)
    y_ev, y_open = _yawns(cam_t, shapes["jawOpen"], face)
    n_ev = _nods(cam_t, pose["headPitch"], face)
    hm = _head_motion(cam_t, pose["headYaw"], pose["headPitch"], face)

    elapsed = np.minimum(60.0, ft - ft[0] + 1.0 / frame_hz)
    bc = _window_count(b_off, b_on, ft)
    F["blinkRate"] = bc / elapsed * 60
    if len(b_on):
        dur = b_off - b_on
        sum_d = np.zeros(n_frames); n_long = np.zeros(n_frames)
        order = np.argsort(b_on)
        on_s, off_s, d_s = b_on[order], b_off[order], dur[order]
        lo = np.searchsorted(on_s, ft - 60 - 1e-9, side="left")
        hi = np.searchsorted(on_s, ft + 1e-9, side="right")
        for i, (l, h) in enumerate(zip(lo, hi)):
            if h > l:
                m = off_s[l:h] <= ft[i] + 1e-9
                sum_d[i] = d_s[l:h][m].sum(); n_long[i] = np.count_nonzero(d_s[l:h][m] > 0.5)
        F["blinkMeanDuration"] = np.where(bc > 0, sum_d / np.maximum(bc, 1), 0.0)
        F["longBlinkCount"] = n_long
    else:
        F["blinkMeanDuration"] = np.zeros(n_frames); F["longBlinkCount"] = np.zeros(n_frames)

    # PERCLOS over tracked frames, trailing 60 s (frames with t >= now - 60 + 1e-6)
    closed = (fresh & (closure[ci] > 0.8)).astype(float)
    w = int(round(60 * frame_hz))
    cc = np.concatenate([[0], np.cumsum(closed)]); cf = np.concatenate([[0], np.cumsum(fresh.astype(float))])
    j = np.arange(n_frames); a = np.maximum(0, j + 1 - w)
    num, den = cc[j + 1] - cc[a], cf[j + 1] - cf[a]
    F["perclos"] = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    F["yawnCount"] = _window_count(y_ev, y_ev, ft)
    F["nodCount"] = _window_count(n_ev, n_ev, ft)
    F["yawnActive"] = np.where(fresh, y_open[ci], False).astype(float)
    F["headMotion"] = np.where(fresh, hm[ci], 0.0)

    # brow tension: trailing 2 s mean over tracked samples (relative to the latest tracked sample)
    brow = (shapes["browDownLeft"] + shapes["browDownRight"]) / 2
    tt = cam_t[face]; bb = brow[face]
    cb = np.concatenate([[0], np.cumsum(bb)])
    hi = np.searchsorted(tt, cam_t[ci] + 1e-9, side="right")
    lo = np.searchsorted(tt, cam_t[ci] - 2.0 - 1e-9, side="left")
    F["browTension"] = np.where(fresh & (hi > lo), (cb[hi] - cb[lo]) / np.maximum(hi - lo, 1), 0.0)

    # motion: EMA of acceleration (tau 0.3 s), pickup state of the latest sample
    motion_t = np.asarray(motion_t, float)
    mm = np.zeros(len(motion_t)); v = None; lt = None
    for i, (ti, x) in enumerate(zip(motion_t.tolist(), np.asarray(accel).tolist())):
        if v is None:
            v = x
        elif ti > lt:
            v += (1 - math.exp(-(ti - lt) / 0.3)) * (x - v)
        lt = ti; mm[i] = v
    mi = np.searchsorted(motion_t, ft + 1e-9, side="right") - 1
    has = mi >= 0
    F["phonePickup"] = np.where(has, np.asarray(picked)[np.maximum(mi, 0)], 0).astype(float)
    F["motionMagnitude"] = np.where(has, mm[np.maximum(mi, 0)], 0.0)

    # heart rate pass-through
    F["heartRate"] = np.zeros(n_frames); F["heartRateQuality"] = np.zeros(n_frames)
    F["heartRateDelta"] = np.zeros(n_frames)
    if hr:
        ht = np.array([h[0] for h in hr]); hb = np.array([h[1] for h in hr]); hq = np.array([h[2] for h in hr])
        hbase = np.array([h[3] if h[3] is not None else 0.0 for h in hr])
        k = np.searchsorted(ht, ft + 1e-9, side="right") - 1
        ok = (k >= 0)
        kk = np.maximum(k, 0)
        ok &= (ft - ht[kk] <= 5.0 + 1e-9) & (hb[kk] > 0)
        F["heartRate"] = np.where(ok, hb[kk], 0.0)
        F["heartRateQuality"] = np.where(ok, hq[kk], 0.0)
        F["heartRateDelta"] = np.where(ok & (hbase[kk] > 0), hb[kk] - hbase[kk], 0.0)
    return F
