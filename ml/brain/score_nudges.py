#!/usr/bin/env python3
"""Step 5: score each nudge arm with the brain model and write weak Beta priors.

    python ml/brain/score_nudges.py            # -> Resources/Lectures/nudge_priors.json

Each NudgeArm (except noNudge) is rendered as a short synthetic clip (portrait 180x320, 10 fps, 12 s):
a dim, static "study screen" for 2 s, the nudge for 6 s, then 4 s of study screen again. Visual arms are
drawn with numpy/Pillow (glow = edge brightness ramp, light pulse = full-screen flashes, cards = rendered
text, stretch = countdown, breathing = expanding/contracting circle); sound arms are synthesised tones
(chime = two soft bell tones; haptic = a quiet 170 Hz buzz, a crude audio proxy for a vibration, since the
model has no touch input). A no-nudge baseline clip is rendered the same way.

Every clip goes through run_brain_model() and to_networks(z=False) (lag-shifted, 1 Hz, NOT z-scored, so
clips stay comparable). Per arm, relative to the baseline clip, averaged over the nudge window:
    capture = d(DorsAttn) + d(SalVentAttn)     predicted attention capture
    arousal = max(0, d(Limbic))                predicted limbic/arousal load (penalised)
    score   = capture - 0.5 * arousal
Scores are squashed robustly, p = 0.5 + 0.15 * tanh(score / (2 * median |score|)) (so p is in 0.35..0.65),
with prior strength alpha + beta = 4 (weak; the in-app trial data dominates quickly).
noNudge gets a neutral Beta(1, 1). The experimental lightPulse arm is scored but always gets a neutral
Beta(2, 2), so no prior implies that it works.

Output: {"mock": <bool>, "note": "...", "priors": [{"arm", "alpha", "beta"}, ...], "scores": {...}}.
With the MOCK model, "mock" is true: these priors are MOCK and carry no evidence about real nudges.
The app decodes it with NudgePriorsLoader (Modules/BrainContent), which also accepts a bare array.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_common import LECTURES_DIR, NETWORKS, WORK_DIR  # noqa: E402
from run_brain_model import run_brain_model  # noqa: E402
from to_networks import to_networks  # noqa: E402

# Must match NudgeArm raw values in Modules/Contracts/Types.swift.
ARMS = ["ambientGlow", "haptic", "chime", "putMeDown", "recapCard", "stretchBreak", "lightPulse",
        "pacedBreathing", "preemptiveGlow", "recallQuestion"]
CONTROL = "noNudge"
# NudgeArm.isExperimental: scored and reported, but always given a neutral Beta(2, 2) prior so nothing
# (not even a MOCK score) suggests it works before the user's own trial data does.
EXPERIMENTAL = {"lightPulse"}

W, H, FPS, SR = 180, 320, 10, 16000
CLIP_S, ON_S, OFF_S = 12.0, 2.0, 8.0
PRIOR_STRENGTH, PRIOR_SPREAD = 4.0, 0.15

CARD_TEXT = {
    "putMeDown": ["Put me down", "", "Back to your", "lecture?"],
    "recapCard": ["Recap", "", "Light makes the", "silica conduct,", "leaving a charged", "image."],
    "recallQuestion": ["Quick question", "", "What forms around", "a black hole?", "", "A) a disk", "B) a star",
                       "C) a comet"],
}


# ----------------------------------------------------------------------------- rendering

def _study_screen() -> np.ndarray:
    f = np.full((H, W, 3), 0.10, np.float32)
    f[60:200, 15:165] = 0.18                    # a dim "lecture" panel
    f[215:225, 15:120] = 0.25                   # a caption line
    return f


def _text_card(lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([14, 70, W - 14, H - 70], radius=14, fill=235)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    y = 88
    for ln in lines:
        d.text((26, y), ln, fill=20, font=font)
        y += 19
    return np.asarray(img, np.float32)[..., None].repeat(3, 2) / 255.0


def _edge_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]).astype(np.float32)
    return np.exp(-dist / 18.0)[..., None]


def render_frames(arm: str | None) -> np.ndarray:
    n = int(CLIP_S * FPS)
    base = _study_screen()
    frames = np.repeat(base[None], n, 0)
    if arm is None or arm in ("haptic", "chime"):
        return frames
    t = np.arange(n) / FPS
    on = (t >= ON_S) & (t < OFF_S)
    phase = np.clip((t - ON_S) / (OFF_S - ON_S), 0, 1)
    if arm in ("ambientGlow", "preemptiveGlow"):
        color = np.array([1.0, 0.7, 0.3] if arm == "ambientGlow" else [0.4, 0.8, 1.0], np.float32)
        peak = 0.6 if arm == "ambientGlow" else 0.3
        level = np.where(on, peak * np.sin(np.pi * phase), 0)              # smooth ramp up and down
        frames = frames + level[:, None, None, None] * _edge_mask()[None] * color
    elif arm == "lightPulse":
        level = np.where(on, 0.85 * (np.sin(2 * np.pi * 1.0 * (t - ON_S)) > 0), 0)   # 1 Hz cool-white flashes
        frames = frames + level[:, None, None, None] * np.array([0.85, 0.95, 1.0], np.float32)
    elif arm in CARD_TEXT:
        card = _text_card(CARD_TEXT[arm])
        alpha = np.clip(np.minimum((t - ON_S) / 0.3, (OFF_S - t) / 0.3), 0, 1)
        frames = frames * (1 - alpha[:, None, None, None]) + card[None] * alpha[:, None, None, None]
    elif arm == "stretchBreak":
        for i in np.nonzero(on)[0]:
            card = _text_card(["Stand up and", "stretch", "", f"{60 - int(t[i] - ON_S)} s"])
            frames[i] = 0.5 * card + 0.5 * np.array([0.1, 0.5, 0.5], np.float32)
    elif arm == "pacedBreathing":
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx - W / 2, yy - H / 2)
        for i in np.nonzero(on)[0]:
            radius = 30 + 40 * (0.5 - 0.5 * np.cos(2 * np.pi * (t[i] - ON_S) / 10.0))   # 6 breaths/min
            frames[i] = np.where((r < radius)[..., None], np.array([0.35, 0.55, 0.75], np.float32), frames[i])
    else:
        raise ValueError(arm)
    return np.clip(frames, 0, 1)


def render_audio(arm: str | None) -> np.ndarray:
    t = np.arange(int(CLIP_S * SR)) / SR
    rng = np.random.default_rng(0)
    x = rng.standard_normal(len(t)).astype(np.float32) * 10 ** (-55 / 20)    # quiet room tone
    if arm == "chime":
        for s0 in (ON_S, ON_S + 2.5):
            m = t >= s0
            tt = t[m] - s0
            x[m] += (0.25 * np.exp(-tt / 0.6) * (np.sin(2 * np.pi * 880 * tt) + 0.5 * np.sin(2 * np.pi * 1320 * tt)))
    elif arm == "haptic":
        for s0 in (ON_S, ON_S + 0.4, ON_S + 0.8):
            m = (t >= s0) & (t < s0 + 0.2)
            x[m] += 0.06 * np.sin(2 * np.pi * 170 * (t[m] - s0))
    return np.clip(x, -1, 1)


def write_clip(arm: str | None, out_dir: Path) -> tuple[Path, Path]:
    from scipy.io import wavfile

    name = f"{arm or 'baseline'}_MOCK"
    video, audio = out_dir / f"{name}.mp4", out_dir / f"{name}.wav"
    frames = (render_frames(arm) * 255).round().astype(np.uint8)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video)], input=frames.tobytes(), check=True)
    wavfile.write(str(audio), SR, (render_audio(arm) * 32767).astype(np.int16))
    return video, audio


# ----------------------------------------------------------------------------- scoring

def clip_networks(arm: str | None, out_dir: Path) -> tuple[np.ndarray, bool]:
    video, audio = write_clip(arm, out_dir)
    pred = run_brain_model(video, audio, [], duration=CLIP_S, seed=0)
    res = to_networks(pred, duration=CLIP_S, z=False)
    return res["networks"], bool(pred.get("mock"))


def priors_from_scores(scores: dict[str, float]) -> list[dict]:
    scored = [abs(scores[a]) for a in ARMS if a not in EXPERIMENTAL]
    scale = max(1e-9, 2 * float(np.median(scored)))
    out = []
    for arm in ARMS:
        p = 0.5 if arm in EXPERIMENTAL else 0.5 + PRIOR_SPREAD * float(np.tanh(scores[arm] / scale))
        out.append({"arm": arm, "alpha": round(PRIOR_STRENGTH * p, 3), "beta": round(PRIOR_STRENGTH * (1 - p), 3)})
    out.append({"arm": CONTROL, "alpha": 1.0, "beta": 1.0})
    return out


def score_all(out_dir: Path) -> tuple[dict, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, mock = clip_networks(None, out_dir)
    win = slice(int(ON_S), int(OFF_S) + 2)                     # nudge window (+2 s hemodynamic tail)
    i_da, i_sva, i_lim = (NETWORKS.index(k) for k in ("DorsAttn", "SalVentAttn", "Limbic"))
    table = {}
    for arm in ARMS:
        nets, m = clip_networks(arm, out_dir)
        mock = mock or m
        d = (nets[win] - base[win]).mean(0)
        capture, arousal = float(d[i_da] + d[i_sva]), float(max(0.0, d[i_lim]))
        table[arm] = {"capture": round(capture, 4), "arousal": round(arousal, 4),
                      "score": round(capture - 0.5 * arousal, 4)}
    return table, mock


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(LECTURES_DIR / "nudge_priors.json"))
    ap.add_argument("--clips", default=str(WORK_DIR / "nudge_clips_MOCK"))
    a = ap.parse_args(argv)
    table, mock = score_all(Path(a.clips))
    priors = priors_from_scores({k: v["score"] for k, v in table.items()})
    doc = {"mock": mock,
           "note": ("MOCK: scored with the MOCK feature-driven encoder, not the brain model; these priors carry no "
                    "evidence about real nudges. " if mock else "") +
                   "Weak Beta priors (alpha+beta = 4; noNudge = Beta(1,1)) from predicted attention capture "
                   "(DorsAttn+SalVentAttn) minus 0.5 x limbic/arousal, relative to a no-nudge clip. "
                   "Experimental arm lightPulse is pinned to a neutral Beta(2,2). "
                   "Written by ml/brain/score_nudges.py.",
           "priors": priors,
           "scores": table}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}{' (MOCK)' if mock else ''}")
    for p in priors:
        s = table.get(p["arm"], {})
        print(f"  {p['arm']:15s} Beta({p['alpha']:.2f}, {p['beta']:.2f})  {s}")


if __name__ == "__main__":
    main()
