#!/usr/bin/env python3
"""SYNTHETIC training corpus for the LapseForecaster, coupled to MOCK brain-content timelines.

Everything this script writes is SYNTHETIC (sensor streams, SART trials, experience-sampling
answers) or MOCK (brain timelines). It exists so the training / eval / ablation / export pipeline
runs end to end before real data exists. Numbers computed from it must be labelled SYNTHETIC and
must never be reported as results.

Design (ideas reused from ml/synth/make_replay.py and ml/brain/make_mock_timeline.py):
  * N_LECTURES MOCK lectures generated with the make_mock_timeline.py recipe, different seeds,
    written to out/lectures/<id>.brain.json (mock: true).
  * N_SUBJECTS synthetic subjects with their own baselines (blink rate, HR, gaze bias) and a
    "content susceptibility" k_c.
  * Per subject: 1 SART session (no lecture -> neutral content) + LECTURE_SESSIONS lecture sessions.
  * Lapse onsets are drawn from a per-second hazard while focused:
        logit h(t) = b0 + subj + k_c * (0.55 * driftRisk(t) + 1.3 * valleyNear(t)) + 0.6 * fatigue(t)
    valleyNear = 1 inside a predicted valley or when one starts within 8 s. So content raises the
    lapse probability, and the with/without-content ablation is meaningful ON SYNTHETIC DATA ONLY.
  * Physiological precursors: with p=0.7 a lapse is preceded by a 4-12 s "pre-lapse" ramp of
    random amplitude (blinks, gaze jitter + wander, brow, HR). The rest have no precursor, so
    sensors alone cannot predict every lapse.
  * Session CSVs carry `label`/`lapse` (lecture sessions) and `playhead`; SART sessions carry no
    labels so their positives come only from the SART CSV (commission errors, RT-CV spikes).

Phase 4 default (`--generator live`): the same hazard / content coupling, but each session is rendered
by the realistic generator (ml/synth/synthlib.render_session: simulated camera-rate face samples,
Core Motion and rPPG readings run through a port of the app's AttentionFeatureExtractor) with
per-subject baselines and calibration, dropouts, lighting changes, hard negatives, gradual onsets,
mixtures and ~5% label noise; 10 min lectures, longer lapses and a lower hazard, so sessions are
mostly focused. Precursors (70% of lapses, 6-15 s): the eyes stop tracking the text, gaze wanders,
glances and fidgeting increase, blink rate rises, brows tighten, HR creeps up. These precursors are
an ASSUMPTION of the generator, not a measured effect. `--generator legacy` = the previous corpus.

Writes out/manifest.csv (paths relative to it) (the same manifest format train.py expects for real data).

Usage: ml/.venv/bin/python ml/forecast/synth_forecast.py [--subjects 40] [--seed 11]
           [--generator live|legacy] [--lecture-seconds 600] [--label-noise 0.05]
(Phase 4 models were trained with --subjects 144.)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "ml/synth"))
import synthlib as S  # noqa: E402
FEATS = [f["name"] for f in json.loads((ROOT / "ml/schema/feature_schema.json").read_text())["features"]]
HZ = 10
NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


# ------------------------------------------------------------------ MOCK lectures

def make_mock_lecture(lecture_id: str, seconds: int, seed: int) -> dict:
    """Same recipe as ml/brain/make_mock_timeline.py (MOCK), parameterised by id/length/seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); k /= k.sum()
    nets = {}
    for i, key in enumerate(NETS):
        x = np.convolve(rng.standard_normal(seconds + 24), k, mode="same")[12:-12]
        x += 0.8 * np.sin(2 * np.pi * t / (40 + 9 * i) + i + rng.uniform(0, 6))
        nets[key] = ((x - x.mean()) / x.std()).round(3).tolist()
    att = (np.array(nets["DorsAttn"]) + np.array(nets["SalVentAttn"]) + np.array(nets["Vis"])) / 3
    eng = (1 / (1 + np.exp(-att * 1.5))).round(3)
    drift = (np.array(nets["Default"]) - np.array(nets["DorsAttn"])).round(3)
    valleys, inside, start = [], False, 0
    for s, e in enumerate(eng):
        if e < 0.35 and not inside:
            start, inside = s, True
        if (e >= 0.35 or s == seconds - 1) and inside:
            inside = False
            if s - start >= 6:
                valleys.append({"start": start, "end": s})
    return {"lecture_id": lecture_id, "hz": 1, "mock": True,
            "note": "MOCK synthetic data - not produced by the brain model (ml/forecast/synth_forecast.py)",
            "networks": nets, "predicted_engagement": eng.tolist(),
            "predicted_drift_risk": drift.tolist(), "valleys": valleys}


def valley_near(doc: dict, seconds: int, lead: float = 8.0) -> np.ndarray:
    v = np.zeros(seconds)
    for va in doc.get("valleys", []):
        a, b = va["start"], va["end"]
        v[max(0, int(a - lead)):min(seconds, int(b))] = 1
    return v


# ------------------------------------------------------------------ lapse script

def draw_script(seconds: int, rng, subj: dict, drift: np.ndarray | None, near: np.ndarray | None):
    """Returns list of (onset, end, state, ramp_len, ramp_amp)."""
    episodes, s, refractory = [], 0, 20
    while s < seconds:
        if s < refractory:
            s += 1; continue
        fatigue = s / 600.0
        z = subj["b0"] + 0.6 * fatigue
        if drift is not None:
            z += subj["k_c"] * (0.55 * drift[s] + 1.3 * near[s])
        h = 1 / (1 + np.exp(-z))
        if rng.random() < h:
            r = rng.random()
            p_drowsy = 0.2 + 0.3 * min(1, fatigue)
            state = "drowsy" if r < p_drowsy else ("stressed" if r < p_drowsy + 0.18 else "distracted")
            dur = int(rng.integers(6, 17))
            has_ramp = rng.random() < 0.7
            ramp_len = float(rng.uniform(4, 12)) if has_ramp else 0.0
            ramp_amp = float(rng.uniform(0.5, 1.0)) if has_ramp else 0.0
            episodes.append((s, min(seconds, s + dur), state, ramp_len, ramp_amp))
            refractory = s + dur + int(rng.integers(5, 12))
            s = refractory
        else:
            s += 1
    return episodes


# ------------------------------------------------------------------ frames

def synth_frames(seconds: int, episodes, subj: dict, rng) -> tuple[dict, np.ndarray]:
    n = seconds * HZ
    t = np.arange(n) / HZ
    st = np.array(["focused"] * n, dtype=object)
    pre = np.zeros(n)
    pick = np.zeros(n, bool)
    yawns, nods = [], []
    stress_hr = np.zeros(n)
    for (a, b, s, rl, ra) in episodes:
        st[a * HZ:b * HZ] = s
        if rl > 0:
            m = (t >= a - rl) & (t < a)
            pre[m] = np.maximum(pre[m], ra * (t[m] - (a - rl)) / rl)
        if s == "distracted" and rng.random() < 0.4:
            pa = a + rng.uniform(0, 2); pick |= (t >= pa) & (t < min(b, pa + rng.uniform(3, 7)))
        if s == "drowsy":
            if rng.random() < 0.6: yawns.append(a + rng.uniform(0.5, max(1, b - a - 3)))
            if rng.random() < 0.5: nods.append(a + rng.uniform(0.5, max(1, b - a - 2)))
        if s == "stressed":
            amp = rng.uniform(10, 20)
            m = (t >= a) & (t < b)
            stress_hr[m] = amp * np.clip((t[m] - a) / 6, 0, 1)
    fatigue = t / 600.0
    F = {k: np.zeros(n) for k in FEATS}
    F["faceDetected"][:] = 1
    # brief tracking dropouts
    for _ in range(rng.integers(0, 3)):
        a = rng.integers(0, n - 20); F["faceDetected"][a:a + rng.integers(3, 15)] = 0

    closure = np.zeros(n)
    blink_starts, blink_durs = [], []
    i = 0
    while i < n:
        s = st[i]
        rate = {"focused": subj["blink"], "distracted": subj["blink"] * 1.5, "drowsy": subj["blink"] * 1.8,
                "stressed": subj["blink"] * 1.6}[s] / 60.0
        rate *= (1 + 0.9 * pre[i]) * (1 + 0.25 * fatigue[i])
        i += max(2, int(rng.exponential(1 / rate) * HZ))
        if i >= n:
            break
        dur = rng.uniform(0.6, 1.4) if st[i] == "drowsy" and rng.random() < 0.45 else rng.uniform(0.15, 0.3 + 0.2 * pre[i])
        k = max(1, int(round(dur * HZ)))
        closure[i:i + k] = rng.uniform(0.85, 1.0)
        blink_starts.append(i); blink_durs.append(dur)
    drowsy = st == "drowsy"
    closure = np.maximum(closure, drowsy * np.clip(rng.normal(0.45, 0.1, n), 0, 0.75))
    F["eyeBlinkLeft"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)
    F["eyeBlinkRight"] = np.clip(closure + rng.normal(0, 0.02, n), 0, 1)

    off = st == "distracted"
    smooth = lambda x, w: np.convolve(x, np.ones(w) / w, mode="same")
    side = np.ones(n)  # off-zone direction, fixed per episode below
    for (a, b, s, rl, _) in episodes:
        side[max(0, int((a - rl) * HZ)):b * HZ] = rng.choice([-1, 1])
    jitter = smooth(rng.normal(0, 1, n), 5)
    gyaw = (subj["gaze_bias"] + smooth(rng.normal(0, 0.05, n), 15) + off * 0.75 * side + pre * 0.18 * jitter
            + pre * 0.22 * side)  # gaze starts wandering toward where it will go
    gpitch = smooth(rng.normal(0, 0.04, n), 15) - off * 0.2 - pre * 0.06 * np.abs(jitter)
    gpitch = gpitch - pick * 0.45
    hyaw = 0.5 * gyaw + smooth(rng.normal(0, 0.02, n), 9)
    hpitch = smooth(rng.normal(0, 0.02, n), 9) - pick * 0.3 - 0.02 * fatigue
    for nt in nods:
        m = (t >= nt) & (t < nt + 1.6)
        hpitch[m] -= 0.35 * np.sin(np.pi * (t[m] - nt) / 1.6)
    F["gazeYaw"], F["gazePitch"], F["headYaw"], F["headPitch"] = gyaw, gpitch, hyaw, hpitch
    F["headRoll"] = smooth(rng.normal(0, 0.02, n), 9)
    F["eyeLookOutLeft"] = np.clip(gyaw * 1.2, 0, 1); F["eyeLookInRight"] = np.clip(gyaw * 1.2, 0, 1)
    F["eyeLookInLeft"] = np.clip(-gyaw * 1.2, 0, 1); F["eyeLookOutRight"] = np.clip(-gyaw * 1.2, 0, 1)
    F["eyeLookUpLeft"] = F["eyeLookUpRight"] = np.clip(gpitch * 1.5, 0, 1)
    F["eyeLookDownLeft"] = F["eyeLookDownRight"] = np.clip(-gpitch * 1.5, 0, 1)
    cy, cp = gyaw + 0.3 * hyaw, gpitch + 0.3 * hpitch
    dev = np.sqrt(np.maximum(0, np.abs(cy) - 0.44) ** 2 + np.maximum(0, np.abs(cp) - 0.31) ** 2)
    F["gazeDeviation"] = dev
    F["inWorkZone"] = (dev == 0).astype(float)
    dwell = np.zeros(n)
    for j in range(1, n):
        dwell[j] = 0 if dev[j] == 0 else dwell[j - 1] + 1 / HZ
    F["offZoneDwell"] = dwell

    stressed = st == "stressed"
    F["eyeSquintLeft"] = F["eyeSquintRight"] = np.clip(0.1 + stressed * 0.25 + rng.normal(0, 0.03, n), 0, 1)
    F["eyeWideLeft"] = F["eyeWideRight"] = np.clip(0.05 + rng.normal(0, 0.02, n), 0, 1)
    jaw = np.clip(0.03 + rng.normal(0, 0.015, n), 0, 1)
    for yt in yawns:
        m = (t >= yt) & (t < yt + 3.0)
        jaw[m] = np.maximum(jaw[m], 0.85 * np.sin(np.pi * (t[m] - yt) / 3.0))
    F["jawOpen"] = jaw
    F["mouthClose"] = np.clip(0.2 - jaw * 0.2, 0, 1)
    brow = np.clip(subj["brow"] + stressed * 0.45 + pre * 0.12 + smooth(rng.normal(0, 0.05, n), 9), 0, 1)
    F["browDownLeft"], F["browDownRight"] = brow, np.clip(brow + rng.normal(0, 0.02, n), 0, 1)
    F["browInnerUp"] = np.clip(0.1 + stressed * 0.2 + rng.normal(0, 0.03, n), 0, 1)
    F["browTension"] = smooth((F["browDownLeft"] + F["browDownRight"]) / 2, 20)

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
    ye = np.zeros(n); ne = np.zeros(n)
    for yt in yawns: ye[min(n - 1, int((yt + 1.5) * HZ))] = 1
    for nt in nods: ne[min(n - 1, int((nt + 1.6) * HZ))] = 1
    F["yawnCount"], F["nodCount"] = win(ye), win(ne)
    F["yawnActive"] = (jaw > 0.6).astype(float)

    F["phonePickup"] = pick.astype(float)
    F["motionMagnitude"] = np.clip(np.abs(rng.normal(0.01, 0.005, n)) + pick * np.abs(rng.normal(0.15, 0.05, n)), 0, 2)
    F["headMotion"] = np.abs(np.gradient(hyaw) * HZ) + np.abs(np.gradient(hpitch) * HZ)

    base = subj["hr"]
    hr = base + 2 * np.sin(t / 23 + subj["hr_phase"]) + stress_hr + pre * 4
    hr_held = np.repeat(hr[::HZ], HZ)[:n] + np.repeat(rng.normal(0, 0.8, seconds), HZ)[:n]
    hr_held[:10 * HZ] = 0
    q = np.clip(0.8 - F["headMotion"] * 0.8 + rng.normal(0, 0.05, n), 0.05, 0.98)
    q[:10 * HZ] = 0
    F["heartRate"], F["heartRateQuality"] = hr_held, q
    F["heartRateDelta"] = np.where(hr_held > 0, hr_held - base, 0)
    ff = F["faceDetected"] == 0
    for k in FEATS:
        if k in ("faceDetected", "phonePickup", "motionMagnitude", "blinkRate", "blinkMeanDuration",
                 "longBlinkCount", "perclos", "yawnCount", "nodCount", "heartRate", "heartRateQuality",
                 "heartRateDelta", "offZoneDwell"):
            continue
        F[k] = np.where(ff, 0, F[k])
    return F, st


def write_session(path, t, st, F, playhead, with_labels: bool, seed: int):
    with open(path, "w") as fh:
        fh.write(f"# SYNTHETIC forecaster session generated by ml/forecast/synth_forecast.py (seed={seed}). Not real data.\n")
        cols = ["t", "playhead"] + (["label", "lapse"] if with_labels else []) + FEATS
        fh.write(",".join(cols) + "\n")
        for j in range(len(t)):
            row = [f"{t[j]:.1f}", f"{playhead[j]:.1f}" if playhead is not None else ""]
            if with_labels:
                row += [st[j], str(int(st[j] != "focused"))]
            row += [f"{F[k][j]:.4f}" for k in FEATS]
            fh.write(",".join(row) + "\n")


def synth_sart(seconds: int, st: np.ndarray, pre: np.ndarray, rng, subj: dict):
    rows, ts = [], 0.5
    while ts < seconds - 1.2:
        j = int(ts * HZ)
        lapsed = st[j] != "focused"
        digit = int(rng.integers(1, 10))
        if digit == 3:
            responded = rng.random() < (0.7 if lapsed else 0.08 + 0.15 * pre[j])
        else:
            responded = rng.random() > (0.12 if lapsed else 0.01)
        rt = ""
        if responded:
            sd = 0.13 if lapsed else subj["rt_sd"] * (1 + 1.2 * pre[j])
            rt = f"{max(0.12, rng.normal(0.34 if lapsed else subj['rt'], sd)):.3f}"
        rows.append((ts, digit, int(responded), rt))
        ts += 1.15
    return rows


# ------------------------------------------------------------------ realistic generator (default)

def draw_episodes(seconds: int, rng, subj: dict, drift: np.ndarray | None, near: np.ndarray | None):
    """Hazard-driven lapse onsets (same content coupling as draw_script) rendered as synthlib Episodes:
    gradual ramp-in/out, a distraction kind, occasional ambiguous mixtures, and a precursor ramp for
    ~70% of lapses. Longer episodes and a lower hazard than the legacy script, so sessions are mostly
    focused."""
    eps, s, refractory = [], 0, 20
    while s < seconds - 5:
        if s < refractory:
            s += 1
            continue
        fatigue = s / 900.0
        z = subj["b0"] + 0.6 * fatigue
        if drift is not None:
            z += subj["k_c"] * (0.55 * drift[min(s, len(drift) - 1)] + 1.3 * near[min(s, len(near) - 1)])
        if rng.random() < 1 / (1 + np.exp(-z)):
            r = rng.random()
            p_drowsy = 0.2 + 0.3 * min(1, fatigue)
            state = "drowsy" if r < p_drowsy else ("stressed" if r < p_drowsy + 0.18 else "distracted")
            lo, hi = {"distracted": (6, 40), "drowsy": (15, 90), "stressed": (12, 60)}[state]
            dur = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            e = S.Episode(float(s), float(min(seconds - 1, s + dur)), state,
                          ramp_in=float({"distracted": rng.uniform(0.4, 3), "drowsy": rng.uniform(3, 12),
                                         "stressed": rng.uniform(3, 8)}[state]),
                          ramp_out=float(rng.uniform(1, 6)))
            if state == "distracted":
                e.kind = S._distracted_kind(rng)
            if rng.random() < 0.15:
                e.secondary = str(rng.choice([x for x in S.LAPSES if x != state]))
                e.mix = float(rng.uniform(0.3, 0.85))
                if e.secondary == "distracted":
                    e.kind = str(rng.choice(["gaze", "phone"]))
            if rng.random() < 0.7:
                e.pre_len, e.pre_amp = float(rng.uniform(6, 15)), float(rng.uniform(0.5, 1.0))
            eps.append(e)
            refractory = int(e.end + e.ramp_out + rng.integers(5, 15))
            s = refractory
        else:
            s += 1
    return eps


def render(seconds: int, eps, live_subj, rng, label_noise: float):
    """Realistic sensors -> extractor port. Returns (F, true states, noisy labels, precursor ramp)."""
    t, noisy, true, F, _ = S.render_session(seconds, rng, live_subj, eps, label_noise=label_noise)
    pre = np.zeros(len(t))
    for e in eps:
        if e.pre_len > 0:
            a0 = e.onset - e.ramp_in / 2 - e.pre_len
            m = (t >= a0) & (t < e.onset)
            pre[m] = np.maximum(pre[m], e.pre_amp * np.clip((t[m] - a0) / e.pre_len, 0, 1))
    return F, true, noisy, pre


def main(subjects=40, seed=11, n_lectures=8, lecture_s=None, sart_s=180, lecture_sessions=None, out=None,
         generator="live", label_noise=0.05):
    """`generator="live"` (default): realistic sensors + extractor port (synthlib.render_session), 10 min
    lectures, 2 lecture sessions per subject, ~5% label noise. `"legacy"`: the previous generator."""
    legacy = generator == "legacy"
    lecture_s = lecture_s or (300 if legacy else 600)
    lecture_sessions = lecture_sessions or (3 if legacy else 2)
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out or HERE / "out")
    lec_dir, ses_dir = out / "lectures", out / "synthetic"
    lec_dir.mkdir(parents=True, exist_ok=True); ses_dir.mkdir(parents=True, exist_ok=True)
    lectures = []
    for k in range(n_lectures):
        lid = f"mock_lecture_{k:02d}"
        doc = make_mock_lecture(lid, lecture_s, seed=1000 + 37 * k + seed)
        p = lec_dir / f"{lid}.brain.json"
        json.dump(doc, open(p, "w"))
        lectures.append((lid, p, doc))
    manifest = ["subject,session_id,session_csv,sart_csv,es_csv,lecture_json,playhead_offset,synthetic"]
    n_lapses = 0
    for si in range(subjects):
        subj = {"b0": rng.normal(-4.6 if legacy else -5.3, 0.25), "k_c": rng.uniform(0.6, 1.4),
                "blink": rng.uniform(10, 20),
                "hr": rng.uniform(60, 82), "hr_phase": rng.uniform(0, 6), "gaze_bias": rng.normal(0, 0.06),
                "brow": rng.uniform(0.05, 0.18), "rt": rng.uniform(0.36, 0.48), "rt_sd": rng.uniform(0.035, 0.06)}
        live = None if legacy else S.LiveSubject.sample(si, np.random.default_rng([seed, si, 7]))
        sid = f"S{si:02d}"
        # SART session (no lecture -> neutral content; positives come from SART CSV only)
        n = sart_s * HZ; t = np.arange(n) / HZ
        if legacy:
            eps = draw_script(sart_s, rng, subj, None, None)
            F, st = synth_frames(sart_s, eps, subj, rng)
            pre = np.zeros(n)
            for (a, b, s, rl, ra) in eps:
                if rl > 0:
                    m = (t >= a - rl) & (t < a); pre[m] = np.maximum(pre[m], ra * (t[m] - (a - rl)) / rl)
        else:
            eps = draw_episodes(sart_s, rng, subj, None, None)
            F, st, _, pre = render(sart_s, eps, live, rng, 0.0)
        name = f"{sid}_sart"
        write_session(ses_dir / f"{name}.csv", t, st, F, None, False, seed)
        with open(ses_dir / f"{name}.sart.csv", "w") as fh:
            fh.write("# SYNTHETIC SART trials (ml/forecast/synth_forecast.py). Not real data.\n")
            fh.write("t,digit,responded,rt\n")
            for r in synth_sart(sart_s, st, pre, rng, subj):
                fh.write(f"{r[0]:.3f},{r[1]},{r[2]},{r[3]}\n")
        manifest.append(f"{sid},{name},synthetic/{name}.csv,synthetic/{name}.sart.csv,,,0,1")
        n_lapses += len(eps)
        # Lecture sessions
        for li in rng.choice(n_lectures, size=lecture_sessions, replace=False):
            lid, lp, doc = lectures[li]
            drift = np.array(doc["predicted_drift_risk"]); near = valley_near(doc, lecture_s)
            n = lecture_s * HZ; t = np.arange(n) / HZ
            if legacy:
                eps = draw_script(lecture_s, rng, subj, drift, near)
                F, st = synth_frames(lecture_s, eps, subj, rng)
                lab = st
            else:
                eps = draw_episodes(lecture_s, rng, subj, drift, near)
                F, st, lab, _ = render(lecture_s, eps, live, rng, label_noise)
            name = f"{sid}_{lid}"
            write_session(ses_dir / f"{name}.csv", t, lab, F, t, True, seed)
            # experience sampling every 45-90 s (denser than the app's 8 min, SYNTHETIC)
            with open(ses_dir / f"{name}.es.csv", "w") as fh:
                fh.write("# SYNTHETIC experience-sampling answers. Not real data.\nt,focused,reason\n")
                ts = rng.uniform(40, 90)
                while ts < lecture_s - 5:
                    lapsed = st[int(ts * HZ)] != "focused"
                    focused = (not lapsed) if rng.random() > 0.1 else lapsed
                    reason = "" if focused else {"drowsy": "tired", "stressed": "stressed"}.get(st[int(ts * HZ)], "distracted")
                    fh.write(f"{ts:.1f},{int(focused)},{reason}\n")
                    ts += rng.uniform(45, 90)
            manifest.append(f"{sid},{name},synthetic/{name}.csv,,synthetic/{name}.es.csv,lectures/{lid}.brain.json,0,1")
            n_lapses += len(eps)
    (out / "manifest.csv").write_text("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest) - 1} SYNTHETIC sessions ({subjects} subjects, {n_lapses} lapse episodes, "
          f"{generator} generator) + {n_lectures} MOCK lectures to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    ap.add_argument("--generator", choices=["live", "legacy"], default="live")
    ap.add_argument("--lecture-seconds", type=int, default=None, help="default 600 (live) / 300 (legacy)")
    ap.add_argument("--label-noise", type=float, default=0.05)
    a = ap.parse_args()
    main(a.subjects, a.seed, out=a.out, generator=a.generator, lecture_s=a.lecture_seconds,
         label_noise=a.label_noise)
