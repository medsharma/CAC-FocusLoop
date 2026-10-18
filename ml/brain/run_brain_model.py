#!/usr/bin/env python3
"""Step 2: the single adapter between FocusLoop's offline pipeline and a brain-encoding model.

    run_brain_model(video_path, audio_path, transcript) -> predictions (dict)

The real Axonal Labs model (video + audio + text -> predicted fMRI response, "average brain") is
pre-existing work and is plugged in at the TODO(human) block below. Until then this file ships a
**MOCK** implementation: plausible fsaverage5 vertex time series driven by simple, content-dependent
stimulus features (frame-difference motion, luminance change, visual detail, audio RMS loudness and
onsets, transcript speech/word rate, time since the last scene cut), convolved with a canonical HRF.
It is NOT a brain model. Everything it produces carries "mock": True and intermediate files written
from it have MOCK in their names.

Return shape (what to_networks.py accepts), a dict with:
    "space": "fsaverage5" | "fsaverage" | "mni_volume"
    "data":  np.ndarray (n_timepoints, n_vertices) for the surface spaces
             - fsaverage5: n_vertices = 20,484 (LH 0-10241 then RH), FreeSurfer vertex order
             - fsaverage:  n_vertices = 327,684 (LH 0-163841 then RH); fsaverage5 = first 10,242 per hemi
             (vertices x time is also accepted; orientation is detected from the vertex count)
    "img":   for "mni_volume" only: a 4-D NIfTI (nibabel image or path) in MNI152 space, x/y/z/time
    "tr":    seconds between timepoints (float)
    "mock":  bool
    "t0":    optional, stimulus time (s) of timepoint 0 (default 0)

Usage (prints a summary; writes ml/data/brain/<id>/predictions_MOCK.npz):
    python ml/brain/run_brain_model.py ml/data/brain/<id>/stimulus.json
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_common import N_FS5, NETWORKS, load_network_labels  # noqa: E402

MOCK_TR = 1.5          # deliberately not 1 s, so the 1 Hz resampling in to_networks.py is exercised
FEATURE_DT = 0.5       # feature grid (s)


# ======================================================================= adapter

def run_brain_model(video_path: str | Path | None, audio_path: str | Path | None, transcript: list[dict] | None,
                    *, duration: float | None = None, seed: int | None = None) -> dict:
    """Predict the (average-brain) response to one stimulus. See module docstring for the return shape."""
    # ------------------------------------------------------------------------------------------
    # TODO(human): plug in the Axonal Labs model call here.
    #
    #   preds = axonal.predict(video=video_path, audio=audio_path, text=transcript)
    #   return {"space": "fsaverage5",          # or "fsaverage" / "mni_volume"
    #           "data": preds.vertices,          # (n_timepoints, 20484) or (n_timepoints, 327684)
    #           # "img": preds.nifti,            # 4-D MNI NIfTI instead of "data" when volumetric
    #           "tr": preds.tr,                  # seconds per timepoint
    #           "mock": False}
    #
    # `transcript` is [{"start": s, "end": s, "text": str}] from prepare_stimulus.py (may be empty).
    # Predictions should be raw (not lag-shifted); to_networks.py shifts back by the hemodynamic lag.
    # ------------------------------------------------------------------------------------------
    return mock_brain_model(video_path, audio_path, transcript, duration=duration, seed=seed)


# ======================================================================= MOCK model

def _decode_gray(video_path: str | Path, dt: float, w: int = 64, h: int = 36) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH. Install it with: brew install ffmpeg")
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-vf",
                          f"fps={1 / dt},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0


def _read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    sr, x = wavfile.read(str(path))
    scale = float(np.iinfo(x.dtype).max) + 1 if np.issubdtype(x.dtype, np.integer) else 1.0
    x = x.astype(np.float32) / scale
    if x.ndim > 1:
        x = x.mean(1)
    return x, int(sr)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, np.float32)
    out[:min(n, len(a))] = a[:n]
    if 0 < len(a) < n:
        out[len(a):] = a[-1]
    return out


def stimulus_features(video_path, audio_path, transcript, duration=None, dt=FEATURE_DT) -> dict[str, np.ndarray]:
    """Simple content features on a `dt` grid, each roughly in 0..1 (fixed scales, not per-stimulus)."""
    frames = _decode_gray(video_path, dt) if video_path else np.zeros((0, 36, 64), np.float32)
    audio, sr = _read_wav(audio_path) if audio_path else (np.zeros(0, np.float32), 16000)
    if duration is None:
        duration = max(len(frames) * dt, len(audio) / sr)
    n = max(1, int(np.ceil(duration / dt)))

    if len(frames):
        lum = frames.mean((1, 2))
        motion = np.r_[0, np.abs(np.diff(frames, axis=0)).mean((1, 2))]
        gy, gx = np.gradient(frames, axis=(1, 2))
        edge = np.hypot(gx, gy).mean((1, 2))
    else:
        lum = motion = edge = np.zeros(1, np.float32)
    lum, motion, edge = _fit(lum, n), _fit(motion, n), np.clip(_fit(edge, n) * 8, 0, 1)
    dlum = np.r_[0, np.abs(np.diff(lum))]
    cut = (motion > np.maximum(0.08, 3 * np.median(motion) + 1e-6)).astype(np.float32)

    hop = int(sr * dt)
    nb = len(audio) // hop if hop else 0
    rms = np.sqrt((audio[:nb * hop].reshape(nb, hop) ** 2).mean(1) + 1e-12) if nb else np.zeros(1)
    loud = np.clip((20 * np.log10(rms + 1e-9) + 50) / 40, 0, 1)        # -50 dBFS -> 0, -10 dBFS -> 1
    loud = _fit(loud, n)
    onset = np.clip(np.r_[0, np.diff(loud)], 0, None) * 3

    speech, words = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for seg in transcript or []:
        a, b = int(seg["start"] / dt), max(int(seg["start"] / dt) + 1, int(np.ceil(seg["end"] / dt)))
        a, b = max(0, min(n, a)), max(0, min(n, b))
        if b > a:
            speech[a:b] = 1
            words[a:b] += len(seg["text"].split()) / max(dt, seg["end"] - seg["start"]) / 3.0
    words = np.clip(words, 0, 1.5)

    since_cut = np.zeros(n, np.float32)
    acc = 0.0
    for i in range(n):
        acc = 0.0 if cut[i] else acc + dt
        since_cut[i] = acc
    boredom = 1 - np.exp(-since_cut / 20.0)

    return {"lum": lum, "dlum": dlum, "motion": motion, "edge": edge, "cut": cut, "loud": loud,
            "onset": onset, "speech": speech, "words": words, "boredom": boredom.astype(np.float32)}


def mock_network_drive(f: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 7) network drives in NETWORKS order. Hand-written MOCK weights, loosely inspired by the
    usual functional roles (visual motion -> Vis/DorsAttn, sudden change -> SalVentAttn, sound ->
    SomMot incl. auditory cortex, language/detail -> Cont, unchanging content -> Default)."""
    d = np.stack([
        4.0 * f["motion"] + 1.0 * f["edge"] + 3.0 * f["dlum"],                                   # Vis
        1.0 * f["loud"] + 0.5 * f["speech"],                                                     # SomMot
        3.0 * f["motion"] + 0.8 * f["edge"] + 0.8 * f["cut"] + 0.4 * f["words"],                 # DorsAttn
        2.5 * f["onset"] + 4.0 * f["dlum"] + 1.0 * f["cut"],                                     # SalVentAttn
        0.6 * f["loud"] * (1 - f["speech"]) + 2.5 * f["dlum"] * f["lum"] + 0.3 * f["onset"],     # Limbic
        0.8 * f["words"] + 0.6 * f["edge"],                                                      # Cont
        0.9 * f["boredom"] - 2.0 * f["motion"] - 0.5 * f["onset"] - 0.3 * f["words"],            # Default
    ], 1)
    return d.astype(np.float32)


def canonical_hrf(tr: float, length: float = 32.0) -> np.ndarray:
    """SPM-style double-gamma HRF sampled every `tr` seconds (peak ~5 s), unit sum."""
    from scipy.stats import gamma
    t = np.arange(0, length, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return (h / h.sum()).astype(np.float32)


def _seed_for(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def mock_brain_model(video_path, audio_path, transcript, *, duration=None, seed=None, tr=MOCK_TR) -> dict:
    """MOCK: content-feature-driven fsaverage5 vertex time series. Not a brain model."""
    f = stimulus_features(video_path, audio_path, transcript, duration)
    drive = mock_network_drive(f)                                   # (n_fine, 7) on FEATURE_DT grid
    n_fine = drive.shape[0]
    dur = n_fine * FEATURE_DT
    n_tr = max(1, int(np.ceil(dur / tr)))
    # average the fine grid into TR bins
    edges = np.minimum((np.arange(n_tr + 1) * tr / FEATURE_DT).round().astype(int), n_fine)
    binned = np.stack([drive[a:max(b, a + 1)].mean(0) if a < n_fine else drive[-1]
                       for a, b in zip(edges[:-1], edges[1:])])
    hrf = canonical_hrf(tr)
    bold = np.stack([np.convolve(binned[:, k], hrf)[:n_tr] for k in range(7)], 1)

    rng = np.random.default_rng(seed if seed is not None else _seed_for(Path(str(video_path)).name, n_fine))
    ar = np.zeros_like(bold)
    e = rng.standard_normal(bold.shape).astype(np.float32) * 0.02
    for i in range(1, n_tr):                                        # slow AR(1) "physiological" noise
        ar[i] = 0.8 * ar[i - 1] + e[i]
    bold = bold + ar

    labels = load_network_labels("annot")
    vr = np.random.default_rng(1234)                                # fixed vertex gains across stimuli
    gain = vr.uniform(0.6, 1.3, N_FS5).astype(np.float32)
    data = np.zeros((n_tr, N_FS5), np.float32)
    cortex = labels > 0
    data[:, cortex] = bold[:, labels[cortex] - 1] * gain[cortex] + 0.15 * bold.mean(1, keepdims=True)
    data += rng.standard_normal(data.shape).astype(np.float32) * 0.02
    return {"space": "fsaverage5", "data": data, "tr": tr, "t0": 0.0, "mock": True,
            "model": "MOCK feature-driven encoder (run_brain_model.mock_brain_model); not a brain model",
            "features": {k: v.tolist() for k, v in f.items()}, "feature_dt": FEATURE_DT}


def save_predictions(pred: dict, path: str | Path) -> Path:
    """Intermediate cache. Mock outputs must have MOCK in the file name."""
    path = Path(path)
    if pred.get("mock") and "MOCK" not in path.name:
        path = path.with_name(path.stem + "_MOCK" + path.suffix)
    np.savez_compressed(path, data=np.asarray(pred["data"], np.float16), tr=pred["tr"], space=pred["space"],
                        mock=bool(pred.get("mock")))
    return path


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    m = json.loads(Path(argv[0]).read_text())
    tr = json.loads(Path(m["transcript"]).read_text())
    pred = run_brain_model(m["video"], m["audio_wav"], tr, duration=m["duration"])
    out = save_predictions(pred, Path(argv[0]).with_name("predictions_MOCK.npz" if pred["mock"] else "predictions.npz"))
    print(f"{m['lecture_id']}: space={pred['space']} data={np.shape(pred['data'])} tr={pred['tr']} "
          f"mock={pred['mock']} -> {out}")


if __name__ == "__main__":
    main()
