#!/usr/bin/env python3
"""SYNTHETIC replay stream generator.

Writes Resources/Replay/synthetic_session.replay.csv: a scripted 180 s, 10 Hz FeatureFrame stream
(columns exactly = ml/schema/feature_schema.json order, plus `label` and `lapse` ground truth),
aligned to the MOCK sample_lecture.brain.json valleys (12-35, 69-90, 140-160 s) so that the
Simulator demo shows content valleys -> rising risk -> drift -> nudge -> refocus.

Everything here is SYNTHETIC. It is for development, Simulator demos and tests only and must
never be reported as a result.

Usage: ml/.venv/bin/python ml/synth/make_replay.py [--seconds 180] [--seed 3] [--out PATH]
"""
import argparse, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATS = [f["name"] for f in json.loads((ROOT / "ml/schema/feature_schema.json").read_text())["features"]]
HZ = 10

# (start_s, end_s, state). Anything not covered is focused.
SCRIPT = [
    (28, 41, "distracted"),     # gaze drifts off during/after valley 12-35
    (80, 89, "distracted"),     # phone pickup inside valley 69-90
    (108, 127, "drowsy"),       # long blinks, yawn, nod
    (146, 163, "stressed"),     # HR up, brow tension, in valley 140-160
]
PICKUP = (80, 87)
YAWNS = [113.0, 121.5]
NODS = [117.0, 124.0]
# "pre-lapse" ramps: subtle signs 8 s before each lapse so a forecaster has something to learn
PRE = 8.0


def state_at(t):
    for a, b, s in SCRIPT:
        if a <= t < b:
            return s
    return "focused"


def pre_lapse(t):
    """0..1 ramp in the PRE seconds before a scripted lapse."""
    for a, _, _ in SCRIPT:
        if a - PRE <= t < a:
            return (t - (a - PRE)) / PRE
    return 0.0


def main(seconds=180, seed=3, out=None):
    rng = np.random.default_rng(seed)
    n = seconds * HZ
    t = np.arange(n) / HZ
    st = np.array([state_at(x) for x in t])
    pre = np.array([pre_lapse(x) for x in t])
    F = {k: np.zeros(n) for k in FEATS}
    F["faceDetected"][:] = 1

    # --- blinks: Poisson events; rate and duration depend on state ---
    closure = np.zeros(n)
    blink_starts, blink_durs = [], []
    i = 0
    while i < n:
        s = st[i]
        rate = {"focused": 14, "distracted": 22, "drowsy": 26, "stressed": 24}[s] / 60.0
        rate *= 1 + 0.6 * pre[i]
        gap = rng.exponential(1 / rate)
        i += max(2, int(gap * HZ))
        if i >= n:
            break
        dur = rng.uniform(0.6, 1.4) if st[i] == "drowsy" and rng.random() < 0.45 else rng.uniform(0.15, 0.3)
        k = max(1, int(round(dur * HZ)))
        closure[i:i + k] = rng.uniform(0.85, 1.0)
        blink_starts.append(i); blink_durs.append(dur)
    drowsy = st == "drowsy"
    closure = np.maximum(closure, drowsy * np.clip(rng.normal(0.45, 0.1, n), 0, 0.75))  # heavy lids
    F["eyeBlinkLeft"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)
    F["eyeBlinkRight"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)

    # --- gaze / head ---
    off = st == "distracted"
    smooth = lambda x, w: np.convolve(x, np.ones(w) / w, mode="same")
    gyaw = smooth(rng.normal(0, 0.05, n), 15) + off * 0.75 + pre * 0.12 * np.sin(t * 1.3)
    gpitch = smooth(rng.normal(0, 0.04, n), 15) - off * 0.25
    pick = (t >= PICKUP[0]) & (t < PICKUP[1])
    gpitch = gpitch - pick * 0.45
    hyaw = 0.5 * gyaw + smooth(rng.normal(0, 0.02, n), 9)
    hpitch = smooth(rng.normal(0, 0.02, n), 9) - pick * 0.3
    for nt in NODS:  # nod: pitch drop then recovery
        m = (t >= nt) & (t < nt + 1.6)
        hpitch[m] -= 0.35 * np.sin(np.pi * (t[m] - nt) / 1.6)
    F["gazeYaw"], F["gazePitch"] = gyaw, gpitch
    F["headYaw"], F["headPitch"] = hyaw, hpitch
    F["headRoll"] = smooth(rng.normal(0, 0.02, n), 9)
    F["eyeLookOutLeft"] = np.clip(gyaw * 1.2, 0, 1); F["eyeLookInRight"] = np.clip(gyaw * 1.2, 0, 1)
    F["eyeLookInLeft"] = np.clip(-gyaw * 1.2, 0, 1); F["eyeLookOutRight"] = np.clip(-gyaw * 1.2, 0, 1)
    F["eyeLookUpLeft"] = F["eyeLookUpRight"] = np.clip(gpitch * 1.5, 0, 1)
    F["eyeLookDownLeft"] = F["eyeLookDownRight"] = np.clip(-gpitch * 1.5, 0, 1)
    comb_yaw, comb_pitch = gyaw + 0.3 * hyaw, gpitch + 0.3 * hpitch
    dev = np.sqrt(np.maximum(0, np.abs(comb_yaw) - 0.44) ** 2 + np.maximum(0, np.abs(comb_pitch) - 0.31) ** 2)
    F["gazeDeviation"] = dev
    F["inWorkZone"] = (dev == 0).astype(float)
    dwell = np.zeros(n)
    for j in range(1, n):
        dwell[j] = 0 if dev[j] == 0 else dwell[j - 1] + 1 / HZ
    F["offZoneDwell"] = dwell

    # --- other blendshapes ---
    stressed = st == "stressed"
    F["eyeSquintLeft"] = F["eyeSquintRight"] = np.clip(0.1 + stressed * 0.25 + rng.normal(0, 0.03, n), 0, 1)
    F["eyeWideLeft"] = F["eyeWideRight"] = np.clip(0.05 + rng.normal(0, 0.02, n), 0, 1)
    jaw = np.clip(0.03 + rng.normal(0, 0.015, n), 0, 1)
    for yt in YAWNS:
        m = (t >= yt) & (t < yt + 3.0)
        jaw[m] = np.maximum(jaw[m], 0.85 * np.sin(np.pi * (t[m] - yt) / 3.0))
    F["jawOpen"] = jaw
    F["mouthClose"] = np.clip(0.2 - jaw * 0.2, 0, 1)
    brow = np.clip(0.1 + stressed * 0.45 + pre * 0.05 + smooth(rng.normal(0, 0.05, n), 9), 0, 1)
    F["browDownLeft"], F["browDownRight"] = brow, np.clip(brow + rng.normal(0, 0.02, n), 0, 1)
    F["browInnerUp"] = np.clip(0.1 + stressed * 0.2 + rng.normal(0, 0.03, n), 0, 1)
    F["browTension"] = smooth((F["browDownLeft"] + F["browDownRight"]) / 2, 20)

    # --- rolling 60 s window metrics ---
    w = 60 * HZ
    starts = np.zeros(n); starts[blink_starts] = 1
    longs = np.zeros(n); longs[[s for s, d in zip(blink_starts, blink_durs) if d > 0.5]] = 1
    durs = np.zeros(n); durs[blink_starts] = blink_durs
    cs = lambda x: np.concatenate([[0], np.cumsum(x)])
    win = lambda x: (lambda c: np.array([c[j + 1] - c[max(0, j + 1 - w)] for j in range(n)]))(cs(x))
    elapsed = np.minimum(np.arange(1, n + 1), w) / HZ
    bc = win(starts)
    F["blinkRate"] = bc / elapsed * 60
    F["blinkMeanDuration"] = np.where(bc > 0, win(durs) / np.maximum(bc, 1), 0)
    F["longBlinkCount"] = win(longs)
    F["perclos"] = win((closure > 0.8).astype(float)) / np.minimum(np.arange(1, n + 1), w)
    yawn_ev = np.zeros(n); nod_ev = np.zeros(n)
    for yt in YAWNS: yawn_ev[int((yt + 1.5) * HZ)] = 1
    for nt in NODS: nod_ev[int((nt + 1.6) * HZ)] = 1
    F["yawnCount"], F["nodCount"] = win(yawn_ev), win(nod_ev)
    F["yawnActive"] = (jaw > 0.6).astype(float)

    # --- motion ---
    F["phonePickup"] = pick.astype(float)
    F["motionMagnitude"] = np.clip(np.abs(rng.normal(0.01, 0.005, n)) + pick * np.abs(rng.normal(0.15, 0.05, n)), 0, 2)
    F["headMotion"] = np.abs(np.gradient(hyaw) * HZ) + np.abs(np.gradient(hpitch) * HZ)

    # --- heart rate (as rPPG would report it, 1 Hz updates held) ---
    base = 68.0
    hr = base + 2 * np.sin(t / 23) + stressed * 17 * np.clip((t - 146) / 6, 0, 1) + pre * 2
    hr_held = np.repeat(hr[::HZ], HZ)[:n] + np.repeat(rng.normal(0, 0.8, seconds), HZ)[:n]
    hr_held[: 10 * HZ] = 0  # 10 s buffer before first estimate
    q = np.clip(0.8 - F["headMotion"] * 0.8 + rng.normal(0, 0.05, n), 0.05, 0.98)
    q[: 10 * HZ] = 0
    F["heartRate"], F["heartRateQuality"] = hr_held, q
    F["heartRateDelta"] = np.where(hr_held > 0, hr_held - base, 0)

    lapse = (st != "focused").astype(int)
    out = pathlib.Path(out or ROOT / "Resources/Replay/synthetic_session.replay.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# SYNTHETIC replay stream generated by ml/synth/make_replay.py (seed=%d). Not real data.\n" % seed)
        fh.write(",".join(["t", "label", "lapse"] + FEATS) + "\n")
        for j in range(n):
            fh.write(",".join([f"{t[j]:.1f}", st[j], str(lapse[j])] + [f"{F[k][j]:.4f}" for k in FEATS]) + "\n")
    print(f"wrote {out} ({seconds}s, {n} frames, SYNTHETIC)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.seconds, a.seed, a.out)
