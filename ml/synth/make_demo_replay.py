#!/usr/bin/env python3
"""SYNTHETIC scripted-demo replay stream generator.

Writes Resources/Replay/demo_session.replay.csv: a choreographed 10 Hz FeatureFrame stream that runs
for the full length of the `leaf-photosynthesis-01` lecture ("Travel deep inside a leaf"), so the
app's one-tap scripted demo (Settings -> Demo -> "Run scripted demo (SYNTHETIC)", or the
`-scriptedDemo` launch argument) can play it in lock-step with the lecture video.

Beats (derived from the lecture's predicted valleys and recall questions in its brain JSON and in
lectures.json, so they move with the content):
  1. focused start
  2. pre-lapse ramp just before the first predicted valley (gaze wobble inside the work zone, more
     blinks, slight brow and heart-rate rise): the forecaster's risk climbs while the student stays
     focused, so a pre-emptive glow fires (no recall question is unlocked yet)
  3. a phone pickup (distracted), then a refocus a few seconds later ("Back in X.X s")
  4. a second pre-lapse ramp before the first valley after a recall question unlocks, so the
     pre-emptive nudge can be the recall question
  5. a yawn plus nods and heavy lids (drowsy) -> stretch break
  6. a heart-rate rise plus brow tension (stressed) -> paced breathing, with the heart rate falling
     back afterwards
  7. a focused ending

Same column format as make_replay.py (t, label, lapse, then ml/schema/feature_schema.json order);
make_replay.py itself is not changed. `label` is the scripted ground truth; `lapse` = label != focused.

Everything here is SYNTHETIC. It exists only to demonstrate the app (Demo Director, DEMONSTRATION
ONLY) and to test that the engine reacts to each beat. It must never be reported as a result.

Usage: ml/.venv/bin/python ml/synth/make_demo_replay.py [--lecture leaf-photosynthesis-01] [--seed 11] [--out PATH]
"""
import argparse
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATS = [f["name"] for f in json.loads((ROOT / "ml/schema/feature_schema.json").read_text())["features"]]
HZ = 10

# Work-zone half-extents used by the app's default zone (rad), see WorkZone.default / make_replay.py.
ZONE_YAW, ZONE_PITCH = 0.44, 0.31


def load_lecture(lecture_id):
    """Duration, valleys and recall-question unlock times for a bundled lecture."""
    lec_dir = ROOT / "Resources/Lectures"
    brain = json.loads((lec_dir / f"{lecture_id}.brain.json").read_text())
    hz = float(brain.get("hz", 1))
    duration = len(brain["predicted_engagement"]) / hz
    valleys = sorted((float(v["start"]), float(v["end"])) for v in brain["valleys"])
    recall_after = []
    catalog = json.loads((lec_dir / "lectures.json").read_text())
    for lec in catalog:
        if lec["id"] == lecture_id:
            # The video can run a little past the last 1 Hz brain sample; cover all of it.
            duration = max(duration, float(lec.get("durationSeconds", duration)))
            recall_after = sorted(float(q["after"]) for q in lec.get("recallQuestions", []))
    return duration, valleys, recall_after, bool(brain.get("mock", True))


def plan_beats(duration, valleys, recall_after):
    """Session times of every beat (lecture time == session time in the scripted demo)."""
    if not valleys:
        raise SystemExit("lecture has no predicted valleys; the scripted demo needs at least one")
    v1 = valleys[0]
    beats = {}
    # 2. Pre-lapse ramp: the 13 s before the first valley, gently holding through it (student stays focused).
    beats["pre1"] = (max(8.0, v1[0] - 13.0), v1[1])
    # 3. Phone pickup soon after that valley; refocus after ~7 s.
    pick_start = v1[1] + 10.0
    beats["pickup"] = (pick_start, pick_start + 7.0)
    # 4. Recall-question ramp: before the first valley that starts after a question unlocks.
    unlock = recall_after[0] if recall_after else None
    v2 = next((v for v in valleys if unlock is not None and v[0] - 10.0 >= unlock and v[0] > pick_start + 20), None)
    if v2 is not None:
        beats["pre2"] = (max(unlock, v2[0] - 12.0), v2[1])
    after = (v2[1] if v2 is not None else pick_start + 20.0) + 8.0
    # 5. Drowsy: heavy lids, two yawns, two nods.
    beats["drowsy"] = (after, after + 16.0)
    # 6. Stressed: heart rate up for ~12 s, brow tension; then the HR falls back while breathing.
    s0 = beats["drowsy"][1] + 8.0
    beats["stressed"] = (s0, s0 + 19.0)
    last = beats["stressed"][1]
    if last + 15.0 > duration:
        raise SystemExit(f"beats need {last + 15:.0f} s but the lecture is only {duration:.0f} s")
    return beats


def main(lecture_id="leaf-photosynthesis-01", seed=11, out=None):
    rng = np.random.default_rng(seed)
    duration, valleys, recall_after, mock = load_lecture(lecture_id)
    beats = plan_beats(duration, valleys, recall_after)
    n = int(math.ceil(duration * HZ))
    t = np.arange(n) / HZ

    def span(name):
        a, b = beats.get(name, (-1.0, -1.0))
        return (t >= a) & (t < b)

    def ramp(name, rise=6.0, fall=2.0):
        """0..1 envelope over a beat: rises over `rise` s, holds, falls over the last `fall` s."""
        a, b = beats.get(name, (-1.0, -1.0))
        up = np.clip((t - a) / rise, 0, 1)
        down = np.clip((b - t) / fall, 0, 1)
        return np.where((t >= a) & (t < b), np.minimum(up, down), 0.0)

    # Ground-truth labels.
    st = np.full(n, "focused", dtype=object)
    st[span("pickup")] = "distracted"
    st[span("drowsy")] = "drowsy"
    st[span("stressed")] = "stressed"
    st = st.astype(str)
    pre = np.maximum(ramp("pre1"), ramp("pre2"))   # student still focused: label stays "focused"

    F = {k: np.zeros(n) for k in FEATS}
    F["faceDetected"][:] = 1

    # --- blinks: Poisson events; rate and duration depend on the beat ---
    closure = np.zeros(n)
    blink_starts, blink_durs = [], []
    i = 0
    while i < n:
        rate = {"focused": 14, "distracted": 20, "drowsy": 24, "stressed": 22}[st[i]] / 60.0
        rate *= 1 + 0.6 * pre[i]
        i += max(2, int(rng.exponential(1 / rate) * HZ))
        if i >= n:
            break
        long_blink = st[i] == "drowsy" and rng.random() < 0.55
        dur = rng.uniform(0.7, 1.4) if long_blink else rng.uniform(0.15, 0.28)
        k = max(1, int(round(dur * HZ)))
        closure[i:i + k] = rng.uniform(0.88, 1.0)
        blink_starts.append(i)
        blink_durs.append(dur)
    drowsy = st == "drowsy"
    lids = np.clip(rng.normal(0.48, 0.06, n), 0.3, 0.7) * ramp("drowsy", rise=2.0, fall=1.0)
    closure = np.maximum(closure, lids)
    F["eyeBlinkLeft"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)
    F["eyeBlinkRight"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)

    # --- gaze / head ---
    smooth = lambda x, w: np.convolve(x, np.ones(w) / w, mode="same")
    pick = span("pickup")
    pick_env = ramp("pickup", rise=0.8, fall=0.6)
    # Pre-lapse: restless gaze that stays inside the work zone (combined yaw well under 0.44 rad).
    wobble = pre * (0.11 * np.sin(2 * np.pi * t / 2.3) + 0.045 * np.sin(2 * np.pi * t / 0.9 + 1.0))
    gyaw = smooth(rng.normal(0, 0.035, n), 15) + wobble
    gpitch = smooth(rng.normal(0, 0.03, n), 15) - pick_env * 0.75 + pre * 0.05 * np.sin(2 * np.pi * t / 3.1)
    hyaw = 0.5 * gyaw + smooth(rng.normal(0, 0.015, n), 9)
    hpitch = smooth(rng.normal(0, 0.015, n), 9) - pick_env * 0.35
    da, db = beats["drowsy"]
    nods = [da + 4.5, da + 11.0]
    yawns = [da + 1.0, da + 8.0]
    hpitch = hpitch - 0.06 * ramp("drowsy", rise=2.0, fall=1.0)
    for nt in nods:                            # slow nod: pitch drops, then recovers
        m = (t >= nt) & (t < nt + 1.8)
        hpitch[m] -= 0.4 * np.sin(np.pi * (t[m] - nt) / 1.8)
    F["gazeYaw"], F["gazePitch"] = gyaw, gpitch
    F["headYaw"], F["headPitch"] = hyaw, hpitch
    F["headRoll"] = smooth(rng.normal(0, 0.015, n), 9)
    F["eyeLookOutLeft"] = np.clip(gyaw * 1.2, 0, 1); F["eyeLookInRight"] = np.clip(gyaw * 1.2, 0, 1)
    F["eyeLookInLeft"] = np.clip(-gyaw * 1.2, 0, 1); F["eyeLookOutRight"] = np.clip(-gyaw * 1.2, 0, 1)
    F["eyeLookUpLeft"] = F["eyeLookUpRight"] = np.clip(gpitch * 1.5, 0, 1)
    F["eyeLookDownLeft"] = F["eyeLookDownRight"] = np.clip(-gpitch * 1.5, 0, 1)
    cy, cp = gyaw + 0.3 * hyaw, gpitch + 0.3 * hpitch
    dev = np.sqrt(np.maximum(0, np.abs(cy) - ZONE_YAW) ** 2 + np.maximum(0, np.abs(cp) - ZONE_PITCH) ** 2)
    F["gazeDeviation"] = dev
    F["inWorkZone"] = (dev == 0).astype(float)
    dwell = np.zeros(n)
    for j in range(1, n):
        dwell[j] = 0 if dev[j] == 0 else dwell[j - 1] + 1 / HZ
    F["offZoneDwell"] = dwell

    # --- other blendshapes ---
    stress_env = ramp("stressed", rise=3.0, fall=2.5)
    F["eyeSquintLeft"] = np.clip(0.1 + stress_env * 0.3 + rng.normal(0, 0.025, n), 0, 1)
    F["eyeSquintRight"] = np.clip(F["eyeSquintLeft"] + rng.normal(0, 0.015, n), 0, 1)
    F["eyeWideLeft"] = F["eyeWideRight"] = np.clip(0.05 + rng.normal(0, 0.02, n), 0, 1)
    jaw = np.clip(0.03 + rng.normal(0, 0.012, n), 0, 1)
    for yt in yawns:                           # big slow yawn, jaw wide for ~3 s
        m = (t >= yt) & (t < yt + 3.2)
        jaw[m] = np.maximum(jaw[m], 0.9 * np.sin(np.pi * (t[m] - yt) / 3.2))
    F["jawOpen"] = jaw
    F["mouthClose"] = np.clip(0.2 - jaw * 0.2, 0, 1)
    brow = np.clip(0.1 + stress_env * 0.5 + pre * 0.06 + smooth(rng.normal(0, 0.04, n), 9), 0, 1)
    F["browDownLeft"], F["browDownRight"] = brow, np.clip(brow + rng.normal(0, 0.015, n), 0, 1)
    F["browInnerUp"] = np.clip(0.1 + stress_env * 0.2 + rng.normal(0, 0.025, n), 0, 1)
    F["browTension"] = smooth((F["browDownLeft"] + F["browDownRight"]) / 2, 20)

    # --- rolling 60 s window metrics (as the app's feature extractor reports them) ---
    w = 60 * HZ
    starts = np.zeros(n); starts[blink_starts] = 1
    longs = np.zeros(n); longs[[s for s, d in zip(blink_starts, blink_durs) if d > 0.5]] = 1
    durs = np.zeros(n); durs[blink_starts] = blink_durs

    def win(x):
        c = np.concatenate([[0], np.cumsum(x)])
        j = np.arange(n)
        return c[j + 1] - c[np.maximum(0, j + 1 - w)]

    elapsed = np.minimum(np.arange(1, n + 1), w) / HZ
    bc = win(starts)
    F["blinkRate"] = bc / elapsed * 60
    F["blinkMeanDuration"] = np.where(bc > 0, win(durs) / np.maximum(bc, 1), 0)
    F["longBlinkCount"] = win(longs)
    F["perclos"] = win((closure > 0.8).astype(float)) / np.minimum(np.arange(1, n + 1), w)
    yawn_ev = np.zeros(n); nod_ev = np.zeros(n)
    for yt in yawns: yawn_ev[min(n - 1, int((yt + 1.6) * HZ))] = 1
    for nt in nods: nod_ev[min(n - 1, int((nt + 1.8) * HZ))] = 1
    F["yawnCount"], F["nodCount"] = win(yawn_ev), win(nod_ev)
    F["yawnActive"] = (jaw > 0.6).astype(float)

    # --- motion ---
    F["phonePickup"] = pick.astype(float)
    F["motionMagnitude"] = np.clip(np.abs(rng.normal(0.01, 0.004, n)) + pick * np.abs(rng.normal(0.16, 0.04, n)), 0, 2)
    F["headMotion"] = smooth(np.abs(np.gradient(hyaw) * HZ) + np.abs(np.gradient(hpitch) * HZ), 3)

    # --- heart rate (as rPPG would report it: 1 Hz updates held, first estimate after 10 s) ---
    base = 68.0
    sa, sb = beats["stressed"]
    rise = np.clip((t - sa) / 6.0, 0, 1)                   # +18 bpm over 6 s
    fall = np.clip((t - (sa + 12.0)) / 14.0, 0, 1)         # then back down over 14 s (during breathing)
    stress_hr = 18.0 * rise * (1 - fall) * (t >= sa)
    hr = base + 1.5 * np.sin(t / 23) + stress_hr + pre * 3.0
    hr_held = np.repeat(hr[::HZ], HZ)[:n] + np.repeat(rng.normal(0, 0.6, int(math.ceil(n / HZ))), HZ)[:n]
    q = np.clip(0.82 - F["headMotion"] * 0.5 + rng.normal(0, 0.03, n), 0.2, 0.97)
    q = np.where(pick, np.minimum(q, 0.25), q)             # phone in hand: weak camera signal
    hr_held[: 10 * HZ] = 0
    q[: 10 * HZ] = 0
    F["heartRate"], F["heartRateQuality"] = hr_held, q
    F["heartRateDelta"] = np.where(hr_held > 0, hr_held - base, 0)

    lapse = (st != "focused").astype(int)
    out = pathlib.Path(out or ROOT / "Resources/Replay/demo_session.replay.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# SYNTHETIC scripted demo replay generated by ml/synth/make_demo_replay.py "
                 f"(lecture={lecture_id}, seed={seed}). Not real data. DEMONSTRATION ONLY.\n")
        fh.write(f"# Lecture brain timeline mock={str(mock).lower()}; duration {duration:.1f} s; "
                 f"valleys {', '.join(f'{a:g}-{b:g}' for a, b in valleys)}\n")
        for name, (a, b) in beats.items():
            fh.write(f"# beat {name}: {a:.1f}-{b:.1f} s\n")
        fh.write(",".join(["t", "label", "lapse"] + FEATS) + "\n")
        for j in range(n):
            fh.write(",".join([f"{t[j]:.1f}", st[j], str(lapse[j])] + [f"{F[k][j]:.4f}" for k in FEATS]) + "\n")
    print(f"wrote {out} ({duration:.1f} s, {n} frames, SYNTHETIC)")
    for name, (a, b) in beats.items():
        print(f"  {name:8s} {a:6.1f} - {b:6.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture", default="leaf-photosynthesis-01")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.lecture, a.seed, a.out)
