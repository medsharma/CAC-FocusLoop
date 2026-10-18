"""Shared SYNTHETIC FeatureFrame signal generator (multi-subject).

Generalises the signal ideas in make_replay.py (which is left unchanged) to arbitrary state scripts
and per-subject baselines (blink rate, heart rate, gaze noise, lighting / rPPG quality, face dropouts).

Everything produced here is SYNTHETIC. It exists to exercise the training / export pipeline and must
never be reported as a real result.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "ml/schema/feature_schema.json").read_text())
FEATS = [f["name"] for f in SCHEMA["features"]]
HZ = int(SCHEMA["rate_hz"])
STATES = ["focused", "distracted", "drowsy", "stressed"]
PRE = 8.0  # seconds of subtle "pre-lapse" ramp before each lapse (for forecasters)


@dataclass
class SubjectParams:
    subject: int
    blink_base: float      # blinks/min while focused
    blink_mult: dict       # per-state multiplier of blink_base
    hr_base: float         # resting bpm
    hr_stress: float       # bpm increase when stressed
    baseline_err: float    # error of the app's personal resting-HR baseline (bpm)
    gaze_noise: float      # rad, per-frame gaze jitter (pre-smoothing)
    gaze_offset: float     # rad, calibration offset of the work zone
    quality: float         # 0..1, lighting / skin / camera quality multiplier for rPPG
    dropout_rate: float    # per-second probability a face-tracking dropout starts
    brow_base: float
    brow_stress: float
    squint_base: float
    lid_drowsy: float      # heavy-lid closure level while drowsy
    glance_rate: float     # brief off-zone glances per minute while focused (hard negatives)

    @staticmethod
    def sample(subject: int, rng: np.random.Generator) -> "SubjectParams":
        base = float(np.clip(rng.normal(15, 4), 6, 28))
        return SubjectParams(
            subject=subject,
            blink_base=base,
            blink_mult={"focused": 1.0,
                        "distracted": float(rng.uniform(1.2, 1.7)),
                        "drowsy": float(rng.uniform(1.3, 2.0)),
                        "stressed": float(rng.uniform(1.2, 1.8))},
            hr_base=float(np.clip(rng.normal(70, 8), 50, 95)),
            hr_stress=float(rng.uniform(7, 22)),
            baseline_err=float(rng.normal(0, 2.5)),
            gaze_noise=float(rng.uniform(0.03, 0.09)),
            gaze_offset=float(rng.normal(0, 0.08)),
            quality=float(rng.uniform(0.35, 1.0)),
            dropout_rate=float(rng.uniform(0.0, 0.02)),
            brow_base=float(rng.uniform(0.05, 0.25)),
            brow_stress=float(rng.uniform(0.15, 0.5)),
            squint_base=float(rng.uniform(0.05, 0.2)),
            lid_drowsy=float(rng.uniform(0.3, 0.55)),
            glance_rate=float(rng.uniform(0.3, 2.0)),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def random_script(seconds: float, rng: np.random.Generator,
                  p=(0.0, 0.4, 0.3, 0.3)) -> list[tuple[float, float, str]]:
    """Random (start, end, state) lapse segments; uncovered time is focused."""
    script, t = [], float(rng.uniform(15, 40))
    while True:
        s = str(rng.choice(STATES, p=p))
        dur = {"distracted": rng.uniform(6, 35), "drowsy": rng.uniform(15, 60),
               "stressed": rng.uniform(12, 50)}[s]
        if t + dur > seconds - 5:
            break
        script.append((t, t + float(dur), s))
        t += float(dur) + float(rng.uniform(20, 90))
    return script


def _smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def synth_session(seconds: int, rng: np.random.Generator, subj: SubjectParams,
                  script: list[tuple[float, float, str]] | None = None):
    """Return (t, states[str], F{feature: array}) for one SYNTHETIC session."""
    n = int(seconds * HZ)
    t = np.arange(n) / HZ
    script = random_script(seconds, rng) if script is None else script
    st = np.full(n, "focused", dtype=object)
    pre = np.zeros(n)
    for a, b, s in script:
        st[(t >= a) & (t < b)] = s
        m = (t >= a - PRE) & (t < a)
        pre[m] = np.maximum(pre[m], (t[m] - (a - PRE)) / PRE)
    st = st.astype(str)
    F = {k: np.zeros(n) for k in FEATS}

    # --- discrete events within segments ---
    pick = np.zeros(n, bool)
    off = np.zeros(n)               # gaze-off magnitude (yaw) during distraction
    off_pitch = np.zeros(n)
    yawns, nods = [], []
    for a, b, s in script:
        if s == "distracted":
            m = (t >= a) & (t < b)
            if rng.random() < 0.45:  # phone pickup style
                pa = a + rng.uniform(0, 0.3) * (b - a)
                pb = min(b, pa + rng.uniform(4, 15))
                pick |= (t >= pa) & (t < pb)
                off_pitch[(t >= pa) & (t < pb)] = -rng.uniform(0.3, 0.55)
            # gaze away (left/right/down), possibly intermittently returning
            direction = rng.choice([-1.0, 1.0])
            mag = rng.uniform(0.5, 1.0)
            wobble = 0.5 + 0.5 * (np.sin(2 * np.pi * t / rng.uniform(4, 12) + rng.uniform(0, 6)) > -0.6)
            off[m] = direction * mag * wobble[m]
            off_pitch[m] += -rng.uniform(0, 0.3)
        elif s == "drowsy":
            for _ in range(rng.integers(0, 4)):
                yawns.append(float(rng.uniform(a, max(a, b - 3))))
            for _ in range(rng.integers(0, 4)):
                nods.append(float(rng.uniform(a, max(a, b - 2))))
    # brief glances while focused (hard negatives), shorter than the work-zone dwell of a lapse
    glance = np.zeros(n)
    for g in rng.uniform(0, seconds, rng.poisson(subj.glance_rate * seconds / 60)):
        m = (t >= g) & (t < g + rng.uniform(0.5, 2.0))
        glance[m] = rng.choice([-1, 1]) * rng.uniform(0.4, 0.8)
    glance[st != "focused"] = 0

    # --- blinks: Poisson events; rate and duration depend on state ---
    closure = np.zeros(n)
    blink_starts, blink_durs = [], []
    i = 0
    while i < n:
        rate = subj.blink_base * subj.blink_mult[st[i]] / 60.0 * (1 + 0.6 * pre[i])
        i += max(2, int(rng.exponential(1 / rate) * HZ))
        if i >= n:
            break
        long_p = 0.45 if st[i] == "drowsy" else 0.03
        dur = rng.uniform(0.6, 1.4) if rng.random() < long_p else rng.uniform(0.12, 0.32)
        k = max(1, int(round(dur * HZ)))
        closure[i:i + k] = rng.uniform(0.85, 1.0)
        blink_starts.append(i); blink_durs.append(dur)
    drowsy = st == "drowsy"
    closure = np.maximum(closure, drowsy * np.clip(rng.normal(subj.lid_drowsy, 0.1, n), 0, 0.78))
    F["eyeBlinkLeft"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)
    F["eyeBlinkRight"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)

    # --- gaze / head ---
    gn = subj.gaze_noise
    gyaw = subj.gaze_offset + _smooth(rng.normal(0, gn, n), 15) + off + glance + pre * 0.12 * np.sin(t * 1.3)
    gpitch = _smooth(rng.normal(0, gn * 0.8, n), 15) + off_pitch
    hyaw = 0.5 * gyaw + _smooth(rng.normal(0, gn * 0.4, n), 9)
    hpitch = _smooth(rng.normal(0, gn * 0.4, n), 9) - pick * 0.3 + drowsy * rng.uniform(-0.12, 0)
    for nt in nods:
        m = (t >= nt) & (t < nt + 1.6)
        hpitch[m] -= rng.uniform(0.28, 0.45) * np.sin(np.pi * (t[m] - nt) / 1.6)
    F["gazeYaw"], F["gazePitch"], F["headYaw"], F["headPitch"] = gyaw, gpitch, hyaw, hpitch
    F["headRoll"] = _smooth(rng.normal(0, gn * 0.4, n), 9)
    F["eyeLookOutLeft"] = np.clip(gyaw * 1.2, 0, 1); F["eyeLookInRight"] = np.clip(gyaw * 1.2, 0, 1)
    F["eyeLookInLeft"] = np.clip(-gyaw * 1.2, 0, 1); F["eyeLookOutRight"] = np.clip(-gyaw * 1.2, 0, 1)
    F["eyeLookUpLeft"] = F["eyeLookUpRight"] = np.clip(gpitch * 1.5, 0, 1)
    F["eyeLookDownLeft"] = np.clip(-gpitch * 1.5, 0, 1)
    F["eyeLookDownRight"] = np.clip(-gpitch * 1.5 + rng.normal(0, 0.01, n), 0, 1)
    cy, cp = gyaw + 0.3 * hyaw, gpitch + 0.3 * hpitch
    dev = np.sqrt(np.maximum(0, np.abs(cy) - 0.44) ** 2 + np.maximum(0, np.abs(cp) - 0.31) ** 2)

    # --- other blendshapes ---
    stressed = st == "stressed"
    F["eyeSquintLeft"] = np.clip(subj.squint_base + stressed * rng.uniform(0.1, 0.3) + rng.normal(0, 0.03, n), 0, 1)
    F["eyeSquintRight"] = np.clip(F["eyeSquintLeft"] + rng.normal(0, 0.02, n), 0, 1)
    F["eyeWideLeft"] = F["eyeWideRight"] = np.clip(0.05 + rng.normal(0, 0.02, n), 0, 1)
    jaw = np.clip(0.03 + rng.normal(0, 0.015, n), 0, 1)
    talk = rng.uniform(0, seconds, rng.poisson(seconds / 120))  # occasional non-yawn mouth movement
    for tt in talk:
        m = (t >= tt) & (t < tt + rng.uniform(0.5, 2))
        jaw[m] = np.maximum(jaw[m], rng.uniform(0.15, 0.4))
    for yt in yawns:
        m = (t >= yt) & (t < yt + 3.0)
        jaw[m] = np.maximum(jaw[m], rng.uniform(0.7, 0.95) * np.sin(np.pi * (t[m] - yt) / 3.0))
    F["jawOpen"] = jaw
    F["mouthClose"] = np.clip(0.2 - jaw * 0.2, 0, 1)
    brow = np.clip(subj.brow_base + stressed * subj.brow_stress + pre * 0.05
                   + _smooth(rng.normal(0, 0.06, n), 9), 0, 1)
    F["browDownLeft"], F["browDownRight"] = brow, np.clip(brow + rng.normal(0, 0.02, n), 0, 1)
    F["browInnerUp"] = np.clip(0.1 + stressed * rng.uniform(0.05, 0.25) + rng.normal(0, 0.03, n), 0, 1)

    # --- motion ---
    F["phonePickup"] = pick.astype(float)
    F["motionMagnitude"] = np.clip(np.abs(rng.normal(0.01, 0.005, n)) + pick * np.abs(rng.normal(0.15, 0.05, n)), 0, 2)
    F["headMotion"] = np.abs(np.gradient(hyaw) * HZ) + np.abs(np.gradient(hpitch) * HZ)

    # --- heart rate (as rPPG would report it, 1 Hz updates held) ---
    secs = int(np.ceil(seconds))
    stress_ramp = np.zeros(n)
    for a, b, s in script:
        if s == "stressed":
            m = (t >= a) & (t < b)
            stress_ramp[m] = np.clip((t[m] - a) / 6, 0, 1)
    hr = subj.hr_base + 2 * np.sin(t / 23 + rng.uniform(0, 6)) + subj.hr_stress * stress_ramp + pre * 2
    hr_noise_sd = 0.8 + 6 * (1 - subj.quality)
    hr_held = np.repeat(hr[::HZ], HZ)[:n] + np.repeat(rng.normal(0, hr_noise_sd, secs), HZ)[:n]
    q = np.clip(subj.quality * (0.85 - F["headMotion"] * 0.8) + rng.normal(0, 0.05, n), 0.02, 0.98)
    unavailable = np.repeat(rng.random(secs) < 0.25 * (1 - subj.quality), HZ)[:n]
    unavailable[: 10 * HZ] = True  # 10 s buffer before first estimate
    hr_held[unavailable] = 0; q[unavailable] = 0
    F["heartRate"], F["heartRateQuality"] = hr_held, q
    baseline = subj.hr_base + subj.baseline_err
    F["heartRateDelta"] = np.where(hr_held > 0, hr_held - baseline, 0)

    # --- face-tracking dropouts (missing_value = 0 for face-derived per-frame features) ---
    face = np.ones(n, bool)
    for d in np.nonzero(rng.random(secs) < subj.dropout_rate)[0]:
        face[d * HZ: d * HZ + int(rng.uniform(0.5, 3) * HZ)] = False
    F["faceDetected"] = face.astype(float)
    per_frame_face = [k for k in FEATS if k.startswith(("eye", "jaw", "brow", "mouth", "gaze", "head"))
                      and k not in ("headMotion",)]
    for k in per_frame_face:
        F[k] = np.where(face, F[k], 0.0)
    dev = np.where(face, dev, 0.0)
    closure = np.where(face, closure, 0.0)

    F["gazeDeviation"] = dev
    F["inWorkZone"] = ((dev == 0) & face).astype(float)
    dwell = np.zeros(n)
    for j in range(1, n):
        dwell[j] = 0 if dev[j] == 0 else dwell[j - 1] + 1 / HZ
    F["offZoneDwell"] = dwell
    F["browTension"] = _smooth((F["browDownLeft"] + F["browDownRight"]) / 2, 20)

    # --- rolling 60 s window metrics ---
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
    for yt in yawns:
        yawn_ev[min(n - 1, int((yt + 1.5) * HZ))] = 1
    for nt in nods:
        nod_ev[min(n - 1, int((nt + 1.6) * HZ))] = 1
    F["yawnCount"], F["nodCount"] = win(yawn_ev), win(nod_ev)
    F["yawnActive"] = (F["jawOpen"] > 0.6).astype(float)
    return t, st, F


def write_csv(path, t, st, F, comment: str):
    """Write in the app's export format: t,label,lapse + feature columns (schema order)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lapse = (st != "focused").astype(int)
    with open(path, "w") as fh:
        for line in comment.splitlines():
            fh.write("# " + line + "\n")
        fh.write(",".join(["t", "label", "lapse"] + FEATS) + "\n")
        cols = np.stack([F[k] for k in FEATS], 1)
        for j in range(len(t)):
            fh.write(f"{t[j]:.1f},{st[j]},{lapse[j]}," + ",".join(f"{v:.4f}" for v in cols[j]) + "\n")


# =====================================================================================================
# Realistic SYNTHETIC generator ("live" generator, Phase 4)
#
# The legacy generator above writes 10 Hz features directly. The generator below instead simulates
# what the sensors deliver (camera-rate ARKit face samples, Core Motion samples, 1 Hz rPPG readings)
# and runs them through `live_extractor.extract`, a line-by-line port of the app's
# AttentionFeatureExtractor, so the trailing-60 s counters warm up, decay, merge blinks and ignore
# face-less frames exactly as they do live. Everything here is still SYNTHETIC: the behaviour model
# (how often people glance away, what drowsiness looks like, how lapses start) is our assumption,
# not measured data.
# =====================================================================================================

import sys as _sys  # noqa: E402

_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from live_extractor import extract as _extract, _head_motion  # noqa: E402

CAM_HZ = 30          # simulated ARKit rate (the device runs at 60 Hz; 30 Hz keeps generation fast)
LAPSES = ["distracted", "drowsy", "stressed"]


def _lp(x, tau, hz):
    """One-pole causal low-pass with time constant tau (s), started in steady state."""
    from scipy.signal import lfilter
    x = np.asarray(x, float)
    if tau <= 0 or len(x) == 0:
        return x.copy()
    a = 1 - np.exp(-1.0 / (tau * hz))
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=[x[0] * (1 - a)])
    return y


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _smooth_noise(rng, n, hz, tau, sd):
    """Zero-mean low-passed Gaussian noise scaled to standard deviation `sd`."""
    y = _lp(rng.normal(0, 1, n), tau, hz)
    y = y - y.mean()
    s = y.std()
    return y / s * sd if s > 0 else y


def _calibrate(center, corners):
    """WorkZoneMath.calibrate on fixation medians (margin 0.05 rad + 15 %, clamped 0.08-0.9)."""
    cy, cp = center
    hy = max(abs(c[0] - cy) for c in corners) * 1.15 + 0.05
    hp = max(abs(c[1] - cp) for c in corners) * 1.15 + 0.05
    return (float(cy), float(cp), float(np.clip(hy, 0.08, 0.9)), float(np.clip(hp, 0.08, 0.9)))


@dataclass
class LiveSubject:
    """Per-subject baselines, calibration offsets and behaviour rates (all SYNTHETIC assumptions)."""
    subject: int
    screen: tuple          # screen centre (yaw, pitch) in camera space + half extents (yaw, pitch)
    posture: tuple         # resting head (yaw, pitch) relative to the camera
    posture_drift: float   # sd (rad) of slow posture drift (the calibration goes stale)
    head_gain: float       # fraction of a gaze shift the head follows
    head_tau: float        # s, head lag behind the eyes
    notes: tuple | None    # (yaw, pitch) of paper notes when the calibration included them
    zone: tuple            # calibrated work zone (centerYaw, centerPitch, halfWidthYaw, halfWidthPitch)
    eye_gain: float        # eyeLook blendshape per rad of eye-in-head angle
    reading: float         # 0 = watching video, 1 = reading text (line sweeps)
    blink_base: float      # blinks/min while focused
    blink_peak: float      # blink closure peak (some people never close fully: < 0.8)
    lid_rest: float        # resting eyeBlink value (eye shape, glasses)
    lid_asym: float        # left - right resting offset
    lid_drowsy: float      # extra lid closure when fully drowsy
    long_p_drowsy: float   # P(long blink) when fully drowsy
    blink_mult: dict       # blink-rate multiplier per lapse state at full intensity
    yawn_rate: float       # per min when fully drowsy
    nod_rate: float        # per min when fully drowsy
    jaw_rest: float
    mouth_rest: float
    brow_base: float
    brow_focus: float      # concentration frown while focused (hard negative for "stressed")
    brow_stress: float
    inner_base: float
    squint_base: float
    squint_stress: float
    squint_bright: float   # squint in bright light (hard negative)
    wide_base: float
    bs_noise: float        # blendshape noise sd
    gaze_noise: float      # lookAtPoint gaze noise sd (rad)
    head_jitter: float     # ARKit head-pose jitter sd (rad)
    hr_rest: float
    hr_stress: float
    hr_err: float          # bpm sd of good rPPG readings
    skin_q: float          # rPPG quality factor (skin tone x camera x distance)
    baseline_known_p: float  # P(a resting-HR baseline is already persisted when a session starts)
    baseline_err: float    # bpm sd of that persisted baseline
    glance_rate: float     # brief off-zone glances per min while focused
    notes_rate: float      # note-reading bouts per min (only with notes)
    pick_brief_rate: float # brief (< 4 s) phone touches per min while focused
    talk_rate: float       # focused talking (reading aloud, murmuring) bouts per min
    burst_rate: float      # head-motion bursts (stretching, posture) per min
    dropout_rate: float    # face-tracking dropouts per min in good light
    desk_vib: float        # Core Motion acceleration noise (typing on the desk), g

    @staticmethod
    def sample(subject: int, rng: np.random.Generator) -> "LiveSubject":
        u = rng.uniform
        scy, scp = u(-0.15, 0.15), u(-0.05, 0.22)          # camera usually below the screen
        ey, ep = u(0.2, 0.38), u(0.12, 0.24)
        k_face = u(0.3, 0.8)                               # how squarely the head faces the screen
        posture = (k_face * scy + rng.normal(0, 0.05), k_face * scp + rng.normal(0, 0.06))
        hg, eg = u(0.2, 0.55), u(1.8, 2.8)
        notes = (scy + u(-0.12, 0.12), scp - ep - u(0.12, 0.28)) if rng.random() < 0.45 else None

        def comb(g):  # combined direction while fixating g during calibration
            head = (posture[0] + 0.3 * hg * (g[0] - scy), posture[1] + 0.3 * hg * (g[1] - scp))
            return (g[0] + 0.3 * head[0] + rng.normal(0, 0.015), g[1] + 0.3 * head[1] + rng.normal(0, 0.015))
        bottom = notes[1] - 0.04 if notes else scp - ep
        corners = [comb((scy - ey, scp + ep)), comb((scy + ey, scp + ep)),
                   comb((scy - ey, bottom)), comb((scy + ey, bottom))]
        zone = _calibrate(comb((scy, scp)), corners)
        return LiveSubject(
            subject=subject, screen=(scy, scp, ey, ep), posture=posture, posture_drift=u(0.01, 0.07),
            head_gain=hg, head_tau=u(0.25, 0.7), notes=notes, zone=zone, eye_gain=eg, reading=u(0, 1),
            blink_base=float(np.clip(rng.normal(15, 5), 5, 30)),
            blink_peak=u(0.72, 0.85) if rng.random() < 0.15 else u(0.85, 1.0),
            lid_rest=u(0.0, 0.15), lid_asym=float(rng.normal(0, 0.04)), lid_drowsy=u(0.08, 0.4),
            long_p_drowsy=u(0.15, 0.5),
            blink_mult={"distracted": u(1.0, 1.6), "drowsy": u(1.1, 2.0), "stressed": u(1.1, 1.8)},
            yawn_rate=u(0.2, 2.0), nod_rate=u(0.1, 2.5),
            jaw_rest=u(0.0, 0.08), mouth_rest=u(0.05, 0.3), brow_base=u(0.02, 0.22),
            brow_focus=u(0.0, 0.18) if rng.random() < 0.5 else 0.0, brow_stress=u(0.03, 0.45),
            inner_base=u(0.02, 0.2), squint_base=u(0.02, 0.22), squint_stress=u(0.0, 0.25),
            squint_bright=u(0.0, 0.12), wide_base=u(0.0, 0.1), bs_noise=u(0.008, 0.035),
            gaze_noise=u(0.015, 0.06), head_jitter=u(0.0002, 0.0007),
            hr_rest=float(np.clip(rng.normal(70, 9), 48, 98)), hr_stress=u(4, 22), hr_err=u(0.8, 3.0),
            skin_q=u(0.55, 1.05), baseline_known_p=u(0.3, 0.9), baseline_err=u(1.0, 4.0),
            glance_rate=u(0.3, 3.0), notes_rate=u(0.1, 1.2) if notes else 0.0, pick_brief_rate=u(0.0, 0.3),
            talk_rate=u(0.0, 0.4), burst_rate=u(0.1, 1.5), dropout_rate=u(0.1, 2.5), desk_vib=u(0.003, 0.02),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=float)


@dataclass
class Episode:
    """One lapse. Its intensity ramps in over `ramp_in` s centred on `onset` and out over `ramp_out`
    centred on `end`, so the true label switches exactly where the intensity crosses 0.5."""
    onset: float
    end: float
    state: str
    ramp_in: float = 2.0
    ramp_out: float = 2.0
    kind: str = ""                 # distracted: gaze | phone | talk | away
    secondary: str | None = None   # ambiguous mixture: a second state's signals at `mix` strength
    mix: float = 0.0
    pre_len: float = 0.0           # optional precursor ("pre-lapse") ramp before the ramp-in
    pre_amp: float = 0.0
    label: str | None = None       # label written to the CSV (None = state); label noise lives here


def _intensity(t, e: Episode):
    a = _smoothstep((t - (e.onset - e.ramp_in / 2)) / max(e.ramp_in, 1e-3))
    b = _smoothstep(((e.end + e.ramp_out / 2) - t) / max(e.ramp_out, 1e-3))
    return np.minimum(a, b)


def _distracted_kind(rng):
    return str(rng.choice(["gaze", "phone", "talk", "away"], p=[0.5, 0.27, 0.15, 0.08]))


def realistic_script(seconds: float, rng: np.random.Generator, focus_frac: float | None = None,
                     p=(0.45, 0.32, 0.23), start_min: float = 15.0) -> list[Episode]:
    """Random lapses for one session; mostly focused (focus_frac 0.62-0.9 of the time)."""
    focus_frac = float(rng.uniform(0.62, 0.9)) if focus_frac is None else focus_frac
    durs = {"distracted": (6, 60), "drowsy": (20, 150), "stressed": (15, 100)}
    mean_d = 0.45 * 22 + 0.32 * 60 + 0.23 * 42
    gap_mean = mean_d * focus_frac / (1 - focus_frac)
    eps, t = [], float(rng.uniform(start_min, start_min + gap_mean))
    while True:
        fat = min(1.0, t / 1800)
        pp = np.array([p[0], p[1] * (0.6 + 0.8 * fat), p[2]]); pp /= pp.sum()
        s = str(rng.choice(LAPSES, p=pp))
        lo, hi = durs[s]
        dur = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if t + dur > seconds - 3:
            break
        rin = {"distracted": rng.uniform(0.4, 3), "drowsy": rng.uniform(3, 15), "stressed": rng.uniform(3, 10)}[s]
        rout = {"distracted": rng.uniform(0.4, 3), "drowsy": rng.uniform(2, 10), "stressed": rng.uniform(4, 15)}[s]
        e = Episode(t, t + dur, s, float(rin), float(rout))
        if s == "distracted":
            e.kind = _distracted_kind(rng)
        if rng.random() < 0.15:  # ambiguous mixture, e.g. drowsy AND looking away
            e.secondary = str(rng.choice([x for x in LAPSES if x != s]))
            e.mix = float(rng.uniform(0.3, 0.85))
            if e.secondary == "distracted":
                e.kind = str(rng.choice(["gaze", "phone"]))
        if rng.random() < 0.6:
            e.pre_len, e.pre_amp = float(rng.uniform(6, 15)), float(rng.uniform(0.4, 1.0))
        eps.append(e)
        t = e.end + max(8.0, float(rng.exponential(gap_mean)))
    return eps


def add_label_noise(eps: list[Episode], seconds: float, rng: np.random.Generator, rate: float = 0.05):
    """Mislabel about `rate` of the frames the way self-reports and annotations go wrong: blurred
    boundaries, a lapse called the wrong state, a missed lapse, a lapse reported with no visible signs.
    Sets each episode's `.label`; returns (label-only intervals [(a, b, label)], boundary jitter (k, 2))."""
    for e in eps:
        r = rng.random()
        e.label = (str(rng.choice([x for x in LAPSES if x != e.state])) if r < 0.05
                   else "focused" if r < 0.08 else e.state)
    lapse_s = sum(e.end - e.onset for e in eps)
    budget = rate * seconds - 0.08 * lapse_s - 1.6 * len(eps)
    extra, tries = [], 0
    while budget > 3 and tries < 300:
        tries += 1
        d = float(rng.uniform(5, 20))
        a = float(rng.uniform(20, max(21.0, seconds - d - 1)))
        if any(a < e.end + e.ramp_out + 3 and a + d > e.onset - e.ramp_in - e.pre_len - 3 for e in eps) or \
                any(a < x[1] + 3 and a + d > x[0] - 3 for x in extra):
            continue
        extra.append((a, a + d, str(rng.choice(LAPSES))))
        budget -= d
    return extra, rng.normal(0, 1.0, (len(eps), 2))


def _events(rng, seconds, rate_per_min, dur_lo, dur_hi, weight=None, grid=None, log=True):
    """Poisson events [(start, duration)]; `weight` (0..1, sampled on `grid`) thins the rate."""
    n = rng.poisson(max(0.0, rate_per_min) * seconds / 60)
    st = np.sort(rng.uniform(0, seconds, n))
    if weight is not None and n:
        st = st[rng.random(n) < np.interp(st, grid, weight)]
    d = (np.exp(rng.uniform(np.log(dur_lo), np.log(dur_hi), len(st))) if log
         else rng.uniform(dur_lo, dur_hi, len(st)))
    return list(zip(st.tolist(), d.tolist()))


def _bump(t, a, d):
    """Half-sine 0..1..0 over [a, a + d]."""
    u = (t - a) / max(d, 1e-3)
    out = np.zeros_like(t)
    m = (u >= 0) & (u <= 1)
    out[m] = np.sin(np.pi * u[m])
    return out


def _plateau(t, a, d, edge=0.15):
    """Plateau over [a, a + d] with smooth `edge`-second flanks (gaze shifts are saccades)."""
    return _smoothstep((t - a) / edge) * _smoothstep((a + d - t) / edge)


def render_session(seconds: float, rng: np.random.Generator, subj: LiveSubject, episodes: list[Episode],
                   label_noise: float = 0.05, extra_labels=None):
    """Simulate the sensors for one session and extract 10 Hz FeatureFrames the way the app does.

    Returns (t, labels, true_labels, F, info): labels = with label noise (what a person would have
    reported), true_labels = the generator's ground truth, F = {feature: (n,)}, info = diagnostics.
    """
    R = CAM_HZ
    nc = int(round(seconds * R))
    tc = np.arange(nc) / R
    u = rng.uniform
    scy, scp, ey, ep = subj.screen
    cy, cpz, hy, hpz = subj.zone
    grid = tc[::R]

    # ---------------- lighting: piecewise level with smooth transitions ----------------
    light = np.full(nc, u(0.35, 1.0))
    light_steps = []
    for _ in range(rng.integers(0, 4)):
        a, d, lv = u(0, seconds), u(0.5, 4), u(0.25, 1.0)
        m = tc >= a
        light[m] = light[m] + (lv - light[m]) * _smoothstep((tc[m] - a) / d)
        light_steps.append(a)
    light = np.clip(light, 0.2, 1.0)
    dim = 1 - light

    # ---------------- state intensities (gradual onsets / offsets, mixtures, precursors) -------------
    D, Z, S, P = (np.zeros(nc) for _ in range(4))
    kinds = {k: np.zeros(nc) for k in ("gaze", "phone", "talk", "away")}
    for e in episodes:
        I = _intensity(tc, e)
        tgt = {"distracted": D, "drowsy": Z, "stressed": S}
        np.maximum(tgt[e.state], I, out=tgt[e.state])
        if e.secondary:
            np.maximum(tgt[e.secondary], e.mix * I, out=tgt[e.secondary])
        if "distracted" in (e.state, e.secondary):
            k = e.kind or "gaze"
            np.maximum(kinds[k], I * (1 if e.state == "distracted" else e.mix), out=kinds[k])
        if e.pre_len > 0:
            a0 = e.onset - e.ramp_in / 2 - e.pre_len
            m = (tc >= a0) & (tc < e.onset)
            P[m] = np.maximum(P[m], e.pre_amp * np.clip((tc[m] - a0) / e.pre_len, 0, 1))
    focusedness = 1 - np.maximum.reduce([D, Z, S])
    fatigue = tc / 1800.0

    # ---------------- gaze target: reading / watching around the screen ----------------
    speed = np.clip(1 + 0.3 * _smooth_noise(rng, nc, R, 3, 1), 0.3, 2) * (1 - 0.5 * Z)
    phase = np.cumsum(speed) / R / u(2.0, 5.0)
    sweep_y = (2 * (phase % 1) - 1) * 0.6 * ey
    sweep_p = (1 - 2 * ((phase / u(8, 20)) % 1)) * 0.55 * ep
    walk_y = _smooth_noise(rng, nc, R, 1.5, 0.35 * ey)
    walk_p = _smooth_noise(rng, nc, R, 1.5, 0.35 * ep)
    rd = subj.reading
    track = 1 - 0.7 * P                       # pre-lapse: the eyes stop tracking the text
    gy = scy + track * (rd * sweep_y + (1 - rd) * walk_y)
    gp = scp + track * (rd * sweep_p + (1 - rd) * walk_p) - 0.05 * Z
    jumps = np.repeat(rng.uniform(-0.6, 0.6, nc // 12 + 1) * ey, 12)[:nc]
    gy = gy + S * 0.5 * ((scy + jumps) - gy)          # stressed: rapid scanning saccades

    off_w = np.zeros(nc); off_y = np.zeros(nc); off_p = np.zeros(nc)
    head_extra = np.zeros(nc)

    def put(w, ty, tp):
        m = w > off_w
        off_w[m] = w[m]
        off_y[m] = ty[m] if np.ndim(ty) else ty
        off_p[m] = tp[m] if np.ndim(tp) else tp

    def outside_point():
        if rng.random() < 0.7:
            return cy + rng.choice([-1, 1]) * (hy + u(0.08, 0.7)), cpz + u(-0.4, 0.25)
        return cy + u(-0.6, 0.6), cpz + rng.choice([-1, 1]) * (hpz + u(0.08, 0.45))

    # brief glances off the zone while focused (hard negatives; more frequent before a lapse)
    gw = np.clip(focusedness[::R] * (1 + 4 * P[::R]) / 5, 0, 1)
    for a, d in _events(rng, seconds, subj.glance_rate * 5, 0.3, 2.5, gw, grid):
        oy, op = outside_point()
        put(_plateau(tc, a, d, 0.08), oy, op)
    # reading paper notes (inside the extended zone for subjects who calibrated with their notes)
    if subj.notes:
        for a, d in _events(rng, seconds, subj.notes_rate, 3, 25, focusedness[::R], grid):
            ny, npt = subj.notes
            put(_plateau(tc, a, d, 0.2), ny + 0.4 * ey * np.sin(2 * np.pi * tc / u(2, 4)),
                npt + _smooth_noise(rng, nc, R, 2, 0.03))
    else:  # a quick look down at a desk paper, outside the zone
        for a, d in _events(rng, seconds, 0.15, 1, 4, focusedness[::R], grid):
            put(_plateau(tc, a, d, 0.2), scy + u(-0.2, 0.2), cpz - hpz - u(0.1, 0.35))

    # distraction: look-away bouts from a two-state process driven by the intensity
    away_drive = np.clip(kinds["gaze"] + kinds["talk"], 0, 1)
    if away_drive.any():
        a10 = away_drive[::3]
        st, cur = np.zeros(len(a10)), False
        for i, x in enumerate(a10.tolist()):
            if x <= 0.02:
                cur = False
            elif not cur and rng.random() < (0.02 + 1.2 * x * x) / 10:
                cur = True
            elif cur and rng.random() < (0.05 + 1.5 * (1 - x) ** 2) / 10:
                cur = False
            st[i] = 1.0 if cur else 0.0
        w = _lp(np.repeat(st, 3)[:nc], 0.08, R)
        side_y, side_p = np.zeros(nc), np.zeros(nc)
        for e in episodes:
            if "distracted" in (e.state, e.secondary) and e.kind in ("gaze", "talk", ""):
                m = (tc >= e.onset - e.ramp_in - 3) & (tc < e.end + e.ramp_out + 3)
                oy, op = outside_point()
                if e.kind == "talk":
                    oy = cy + np.sign(oy - cy) * (hy + u(0.25, 0.75))
                    head_extra[m] = np.maximum(head_extra[m], 0.45 * kinds["talk"][m])
                side_y[m], side_p[m] = oy, op
        put(w, side_y + _smooth_noise(rng, nc, R, 2.0, 0.08), side_p + _smooth_noise(rng, nc, R, 2.0, 0.05))

    # phone: sustained pickups while phone-distracted, brief touches while focused
    pickups = []
    for e in episodes:
        if e.kind == "phone" and "distracted" in (e.state, e.secondary):
            a = e.onset - e.ramp_in / 2 + u(0, max(0.1, e.ramp_in))
            b = min(e.end + e.ramp_out / 2, a + max(5.0, (e.end - e.onset) * u(0.7, 1.0)))
            pickups.append((a, b - a))
    pickups += _events(rng, seconds, subj.pick_brief_rate, 1.0, 3.5, focusedness[::R], grid, log=False)
    pick = np.zeros(nc, bool)
    jump_y, jump_p, shake = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    face_lost = np.zeros(nc, bool)
    accel = np.abs(rng.normal(0, subj.desk_vib, nc)) + 0.5 * subj.desk_vib * np.abs(_smooth_noise(rng, nc, R, 0.5, 1))
    for a, d in pickups:
        w = _plateau(tc, a, d, 0.3)
        pick |= (tc >= a) & (tc < a + d)
        put(w, scy + u(-0.15, 0.15) + _smooth_noise(rng, nc, R, 1, 0.04), scp - ep - u(0.25, 0.55))
        jump_y += w * u(-0.15, 0.15); jump_p += w * u(-0.25, -0.05)   # the camera itself moved
        shake = np.maximum(shake, w)
        for x in (a, a + d):
            accel += u(0.25, 0.9) * _bump(tc, x - 0.15, u(0.2, 0.5))
            if rng.random() < 0.4:
                face_lost |= (tc >= x) & (tc < x + u(0.3, 1.5))
        accel += w * np.abs(rng.normal(0, u(0.01, 0.05), nc))
    for a, _ in _events(rng, seconds, 0.03, 0.2, 0.3):   # desk knock: spike trips the pickup detector
        accel += u(0.36, 0.6) * _bump(tc, a, 0.2)
    # PerceptionPickupDetector: picked up on tilt or a spike > 0.35 g, released after 1 s of calm
    picked = pick | (accel > 0.35)
    picked = np.convolve(picked.astype(float), np.ones(R), mode="full")[:nc] > 0

    # ---------------- head pose ----------------
    tgt_y = (1 - off_w) * gy + off_w * off_y
    tgt_p = (1 - off_w) * gp + off_w * off_p
    # the head follows big gaze shifts (off the screen) but barely follows reading saccades
    gain = subj.head_gain + head_extra
    fy = gain * off_w * (off_y - scy) + 0.15 * subj.head_gain * (1 - off_w) * (gy - scy)
    fp = gain * off_w * (off_p - scp) + 0.15 * subj.head_gain * (1 - off_w) * (gp - scp)
    hyaw = subj.posture[0] + _smooth_noise(rng, nc, R, 90, subj.posture_drift) + _lp(fy, subj.head_tau, R)
    hpitch = subj.posture[1] + _smooth_noise(rng, nc, R, 90, subj.posture_drift) + _lp(fp, subj.head_tau, R) \
        - Z * u(0.0, 0.15)
    hyaw += _smooth_noise(rng, nc, R, 2.5, u(0.004, 0.015)) + jump_y
    hpitch += _smooth_noise(rng, nc, R, 2.5, u(0.004, 0.015)) + jump_p
    hyaw += _smooth_noise(rng, nc, R, 0.5, 0.015) * np.clip(S + P, 0, 1)       # fidgeting
    bursts = [(a, d, 0.3) for a, d in _events(rng, seconds, subj.burst_rate, 0.5, 3.0)]
    bursts += [(a, d, 0.1) for a, d in _events(rng, seconds, 0.5, 0.5, 2.0, np.clip(S + 0.5 * P, 0, 1)[::R], grid)]
    for a, d, amp in bursts:                                                    # stretch / posture / fidget
        b = _bump(tc, a, d)
        ay, ap = u(-1, 1) * (0.4 if amp > 0.2 else 0.25), u(-amp, 0.7 * amp)
        hyaw += ay * b; hpitch += ap * b
        tgt_y = tgt_y + 0.6 * ay * b; tgt_p = tgt_p + 0.6 * ap * b
        if rng.random() < 0.25:
            face_lost |= (tc >= a + d * 0.3) & (tc < a + d * 0.3 + u(0.2, 1.5))
    hyaw += shake * _smooth_noise(rng, nc, R, 0.05, u(0.01, 0.03))             # hand-held camera
    hpitch += shake * _smooth_noise(rng, nc, R, 0.05, u(0.01, 0.03))
    for a, _ in _events(rng, seconds, subj.nod_rate, 1, 1, Z[::R], grid):     # drowsy nods
        drop, td, th, tr = u(0.26, 0.5), u(0.3, 0.8), u(0.0, 0.5), u(0.3, 0.8)
        if rng.random() < 0.2:
            th = u(1.5, 4)    # slow head drop: too long for the nod detector
        hpitch -= drop * np.interp(tc, [a, a + td, a + td + th, a + td + th + tr], [0, 1, 1, 0], left=0, right=0)
    yawns = _events(rng, seconds, subj.yawn_rate, 2.0, 6.0, Z[::R], grid, log=False)
    yawns += _events(rng, seconds, 0.05, 2.0, 5.0, log=False)                  # rare yawns while alert
    hroll = 0.2 * subj.posture[0] + _smooth_noise(rng, nc, R, 2, 0.03)
    jit = subj.head_jitter * (1 + dim)
    hyaw_m = hyaw + rng.normal(0, 1, nc) * jit
    hpitch_m = hpitch + rng.normal(0, 1, nc) * jit
    hroll_m = hroll + rng.normal(0, 1, nc) * jit

    # ---------------- gaze (camera space = head + eye) ----------------
    gn = subj.gaze_noise * (1 + dim)
    gyaw = tgt_y + _lp(rng.normal(0, 1, nc), 0.03, R) * gn + _smooth_noise(rng, nc, R, 5, 0.02)
    gpitch = tgt_p + _lp(rng.normal(0, 1, nc), 0.03, R) * gn * 0.8 + _smooth_noise(rng, nc, R, 5, 0.02)
    gyaw += P * (u(0.08, 0.25) * _smooth_noise(rng, nc, R, 1.0, 1.0) + u(-0.15, 0.15))  # pre-lapse wander
    gpitch += P * (u(0.03, 0.12) * _smooth_noise(rng, nc, R, 1.0, 1.0) + u(-0.08, 0.04))

    # ---------------- face presence ----------------
    for a, d in _events(rng, seconds, subj.dropout_rate * (1 + 3 * float(dim.mean())), 0.2, 3.0):
        face_lost |= (tc >= a) & (tc < a + d)
    for a in light_steps:
        if rng.random() < 0.5:
            face_lost |= (tc >= a) & (tc < a + u(0.2, 1.0))
    for e in episodes:
        if e.kind == "away" and "distracted" in (e.state, e.secondary):
            face_lost |= (tc >= e.onset + u(0, 1.5)) & (tc < e.end - u(0, 1.5))
    face_lost |= (np.abs(hyaw) > 0.85) | (hpitch < -0.75) | (hpitch > 0.6)

    # ---------------- blinks and lids ----------------
    rate = subj.blink_base / 60 * (1 + D * (subj.blink_mult["distracted"] - 1)) \
        * (1 + Z * (subj.blink_mult["drowsy"] - 1)) * (1 + S * (subj.blink_mult["stressed"] - 1)) \
        * (1 + 1.0 * P) * (1 + 0.2 * fatigue) * (0.8 + 0.2 * (1 - subj.reading * focusedness))
    cum = np.cumsum(rate) / R
    marks = np.cumsum(rng.exponential(1, int(cum[-1] * 1.3) + 20))
    starts = np.searchsorted(cum, marks[marks < cum[-1]])
    closure = np.zeros(nc)
    long_p = 0.015 + Z * subj.long_p_drowsy
    last_end = -10
    for i in starts.tolist():
        if i <= last_end + 3:
            continue
        d = u(0.5, 1.6) if rng.random() < long_p[i] else \
            float(np.clip(np.exp(rng.normal(np.log(0.17), 0.3)) * (1 + 0.8 * Z[i]), 0.08, 0.5))
        k = max(2, int(round(d * R)))
        prof = np.sin(np.pi * np.linspace(0, 1, k + 2)[1:-1]) ** 0.5
        pk = min(1.0, subj.blink_peak + rng.normal(0, 0.04))
        j = min(nc, i + k)
        closure[i:j] = np.maximum(closure[i:j], pk * prof[: j - i])
        last_end = i + k
    for a, d in _events(rng, seconds, 1.2, 1.2, 4.0, Z[::R] ** 2, grid):        # microsleeps
        closure = np.maximum(closure, 0.95 * _plateau(tc, a, d, 0.2))
    lid = subj.lid_rest + Z * subj.lid_drowsy * (1 + 0.35 * np.sin(2 * np.pi * tc / u(10, 40) + u(0, 6))) + 0.05 * dim
    closure = np.maximum(closure, lid)
    for a, d in yawns:                                                          # eyes narrow in a yawn
        closure = np.maximum(closure, lid + 0.35 * _bump(tc, a, d))
    bsn = subj.bs_noise * (1 + 1.5 * dim)
    blinkL = np.clip(closure + subj.lid_asym / 2 + rng.normal(0, 1, nc) * bsn, 0, 1)
    blinkR = np.clip(closure - subj.lid_asym / 2 + rng.normal(0, 1, nc) * bsn, 0, 1)

    # ---------------- mouth: talking, laughing, yawning ----------------
    jaw = subj.jaw_rest + np.abs(_smooth_noise(rng, nc, R, 0.3, 0.01))
    talk_env = np.zeros(nc)
    for a, d in _events(rng, seconds, subj.talk_rate, 3, 15, focusedness[::R], grid):
        talk_env = np.maximum(talk_env, u(0.08, 0.3) * _plateau(tc, a, d, 0.3))
    talk_env = np.maximum(talk_env, kinds["talk"] * u(0.15, 0.45))
    if talk_env.any():
        syl = 0.5 * (1 + np.sin(2 * np.pi * np.cumsum(u(3, 5) + _smooth_noise(rng, nc, R, 0.3, 1)) / R))
        words = (_smooth_noise(rng, nc, R, 0.4, 1) > -0.3).astype(float)
        jaw = jaw + talk_env * syl * words
        for a, d in _events(rng, seconds, 1.0, 0.3, 1.0, (talk_env > 0)[::R].astype(float), grid):
            jaw = np.maximum(jaw, u(0.62, 0.8) * _bump(tc, a, d))               # laugh: open, not a yawn
    for a, d in yawns:
        jaw = np.maximum(jaw, u(0.7, 1.0) * _bump(tc, a, d) ** 0.7)
        if rng.random() < 0.12:                                                 # hand over the mouth
            face_lost |= (tc >= a + 0.3 * d) & (tc < a + 0.8 * d)
    face = ~face_lost
    jaw = np.clip(jaw + rng.normal(0, 1, nc) * bsn * 0.5 - 0.03 * S, 0, 1)
    mouth = np.clip(subj.mouth_rest - 0.2 * jaw + 0.05 * S + rng.normal(0, 1, nc) * bsn, 0, 1)

    # ---------------- brows / squint / wide ----------------
    reading_focus = focusedness * (1 - off_w)
    brow = subj.brow_base + subj.brow_focus * reading_focus * np.clip(0.6 + 0.4 * _smooth_noise(rng, nc, R, 20, 1), 0, 1.5) \
        + S * subj.brow_stress + P * u(0.0, 0.1) + _smooth_noise(rng, nc, R, 1.0, 0.04)
    browL = np.clip(brow + rng.normal(0, 1, nc) * bsn, 0, 1)
    browR = np.clip(brow + u(-0.04, 0.04) + rng.normal(0, 1, nc) * bsn, 0, 1)
    inner = np.clip(subj.inner_base + S * u(0.0, 0.2) + Z * u(0.0, 0.15) + _smooth_noise(rng, nc, R, 1, 0.03)
                    + rng.normal(0, 1, nc) * bsn, 0, 1)
    bright = np.clip((light - 0.75) / 0.25, 0, 1)
    sq = subj.squint_base + S * subj.squint_stress + bright * subj.squint_bright + _smooth_noise(rng, nc, R, 1, 0.03)
    squintL = np.clip(sq + rng.normal(0, 1, nc) * bsn, 0, 1)
    squintR = np.clip(sq + u(-0.03, 0.03) + rng.normal(0, 1, nc) * bsn, 0, 1)
    wide = np.clip(subj.wide_base + S * u(0, 0.15) + rng.normal(0, 1, nc) * bsn, 0, 1)

    # ---------------- eyeLook blendshapes from eye-in-head angles ----------------
    eye_y, eye_p = gyaw - hyaw_m, gpitch - hpitch_m
    eg = subj.eye_gain
    out_l, in_l = np.clip(eye_y * eg, 0, 1), np.clip(-eye_y * eg, 0, 1)

    def nz(x, s=0.5):
        return np.clip(x + rng.normal(0, 1, nc) * bsn * s, 0, 1)
    shapes = {
        "eyeBlinkLeft": blinkL, "eyeBlinkRight": blinkR,
        "eyeLookInLeft": nz(in_l), "eyeLookInRight": nz(out_l), "eyeLookOutLeft": nz(out_l),
        "eyeLookOutRight": nz(in_l), "eyeLookUpLeft": np.clip(eye_p * eg * 1.2, 0, 1),
        "eyeLookUpRight": nz(np.clip(eye_p * eg * 1.2, 0, 1)),
        "eyeLookDownLeft": np.clip(-eye_p * eg * 1.2 + 0.3 * closure, 0, 1),
        "eyeLookDownRight": nz(np.clip(-eye_p * eg * 1.2 + 0.3 * closure, 0, 1)),
        "eyeSquintLeft": squintL, "eyeSquintRight": squintR, "eyeWideLeft": wide, "eyeWideRight": nz(wide),
        "jawOpen": jaw, "browDownLeft": browL, "browDownRight": browR, "browInnerUp": inner, "mouthClose": mouth,
    }
    pose = {"gazeYaw": gyaw, "gazePitch": gpitch, "headYaw": hyaw_m, "headPitch": hpitch_m, "headRoll": hroll_m}

    # ---------------- rPPG readings at 1 Hz, as POSHeartRateEstimator reports them ----------------
    hm = _head_motion(tc, hyaw_m, hpitch_m, face)
    hr_true = subj.hr_rest + u(1, 3) * np.sin(2 * np.pi * tc / u(60, 200) + u(0, 6)) \
        + subj.hr_stress * _lp(S, u(6, 14), R) + P * u(1, 5) - Z * u(0, 5) + 2.0 * (talk_env > 0) \
        + 4.0 * _lp(shake, 5, R) + 3 * fatigue * u(-1, 1)
    secs = int(seconds)
    q_swing = np.zeros(secs + 2)
    for i in range(1, len(q_swing)):
        q_swing[i] = 0.75 * q_swing[i - 1] + rng.normal(0, 0.1)
    baseline = subj.hr_rest + rng.normal(0, subj.baseline_err) if rng.random() < subj.baseline_known_p else None
    edges = np.diff(np.concatenate([[0], face_lost.astype(int), [0]]))
    rs, re_ = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    resets = np.array([tc[min(e_, nc - 1)] for s_, e_ in zip(rs, re_) if (e_ - s_) / R > 0.5])
    c_hi = np.concatenate([[0], np.cumsum(hm > 0.6)])
    c_calm = np.concatenate([[0], np.cumsum(hm > 0.2)])
    c_talk = np.concatenate([[0], np.cumsum(talk_env > 0)])
    c_hr = np.concatenate([[0], np.cumsum(hr_true)])
    c_lt = np.concatenate([[0], np.cumsum(light)])
    step_t = np.array(light_steps)
    hr, warm, last_good = [], [], None
    for s in range(1, secs + 1):
        k = min(nc, s * R)
        if face_lost[k - 1]:
            continue
        r = resets[resets <= s] if len(resets) else resets
        start = float(r[-1]) if len(r) else 0.0
        if s - start < 10:
            continue                                   # 10 s analysis buffer not full yet
        k0 = max(0, k - 10 * R)
        mean_hr = (c_hr[k] - c_hr[k0]) / (k - k0)
        mot = (c_hi[k] - c_hi[k0]) / (k - k0)
        talk = (c_talk[k] - c_talk[k0]) / (k - k0)
        lt = (c_lt[k] - c_lt[k0]) / (k - k0)
        q = subj.skin_q * (0.4 + 0.6 * lt) * (1 - 0.6 * min(1.0, 5 * mot)) * (1 - 0.3 * talk) + q_swing[s]
        if len(step_t) and np.any((step_t <= s) & (step_t > s - 10)):
            q *= 0.5                                   # a lighting change inside the window
        q = float(np.clip(q, 0, 1))
        if q >= 0.5:
            bpm = mean_hr + rng.normal(0, subj.hr_err * (1.5 - q))
        elif rng.random() < 0.2 + 1.2 * (0.5 - q):     # wrong spectral peak (noise / harmonic)
            bpm = float(rng.choice([u(42, 180), 2 * mean_hr, mean_hr * u(0.6, 0.8)]))
        else:
            bpm = mean_hr + rng.normal(0, 3 + 12 * (0.5 - q))
        bpm = float(np.clip(bpm, 42, 180))
        if c_hi[k] - c_hi[max(0, k - 2 * R)] > 0:     # motion-gated: hold the last good value
            hr.append((float(s), last_good[0] if last_good else bpm, last_good[1] * 0.5 if last_good else 0.0, baseline))
            continue
        if q >= 0.5 and c_calm[k] - c_calm[k0] == 0:   # good, calm for the whole window
            if baseline is not None:
                baseline = baseline + 0.02 * (bpm - baseline)
            else:
                warm.append(bpm)
                if len(warm) >= 15:
                    baseline = float(np.mean(warm))
        if q >= 0.5:
            last_good = (bpm, q)
        hr.append((float(s), bpm, q, baseline))

    # ---------------- extraction (Core Motion sampled at 10 Hz) ----------------
    mi = np.arange(0, nc, R // 10)
    n = int(round(seconds * HZ))
    F = _extract(tc, face, shapes, pose, subj.zone, tc[mi], picked[mi], accel[mi], hr, frame_hz=HZ, n_frames=n)

    # ---------------- labels ----------------
    t = np.arange(n) / HZ
    true = np.full(n, "focused", dtype=object)
    noisy = np.full(n, "focused", dtype=object)
    if label_noise > 0:
        extra, jitter = add_label_noise(episodes, seconds, rng, label_noise)
    else:
        extra, jitter = [], np.zeros((len(episodes), 2))
    for e in episodes:
        true[(t >= e.onset) & (t < e.end)] = e.state
    for e, (j0, j1) in zip(episodes, jitter):
        noisy[(t >= e.onset + j0) & (t < e.end + j1)] = (e.label or e.state) if label_noise > 0 else e.state
    for a, b, lab in list(extra_labels or []) + extra:
        noisy[(t >= a) & (t < b)] = lab
    info = {"label_noise_frac": float(np.mean(noisy != true)), "focused_frac": float(np.mean(true == "focused")),
            "face_frac": float(F["faceDetected"].mean()), "hr_frac": float((F["heartRate"] > 0).mean())}
    return t, noisy.astype(str), true.astype(str), F, info


def synth_session_live(seconds: float, rng: np.random.Generator, subj: LiveSubject,
                       episodes: list[Episode] | None = None, label_noise: float = 0.05):
    """One realistic SYNTHETIC session (random script unless `episodes` is given)."""
    eps = realistic_script(seconds, rng) if episodes is None else episodes
    return render_session(seconds, rng, subj, eps, label_noise=label_noise)


def write_csv_live(path, t, labels, true_labels, F, comment: str):
    """App export format (t,label,lapse + 42 features) plus a trailing `true_label` column (generator
    ground truth before label noise). Loaders that don't know `true_label` ignore it."""
    import pandas as pd
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"t": np.round(t, 1), "label": labels, "lapse": (labels != "focused").astype(int)})
    for k in FEATS:
        df[k] = F[k]
    df["true_label"] = true_labels
    with open(path, "w") as fh:
        for line in comment.splitlines():
            fh.write("# " + line + "\n")
        df.to_csv(fh, index=False, float_format="%.4f")
