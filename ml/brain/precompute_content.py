#!/usr/bin/env python3
"""Step 4: one command per lecture (or --all) -> Resources/Lectures/<id>.brain.json (+ optional <id>.vertices.f16).

    python ml/brain/precompute_content.py black-holes-01
    python ml/brain/precompute_content.py --all
    python ml/brain/precompute_content.py --video some.mp4 --id my-lecture     # ad-hoc, not in lectures.json

Pipeline: prepare_stimulus -> run_brain_model -> to_networks -> content summary.
  networks              7 x N, 1 Hz, lag-shifted, z-scored within the lecture (BrainTimeline contract)
  predicted_engagement  sigmoid(1.5 * mean(z_DorsAttn, z_SalVentAttn, z_Vis)), 0..1
  predicted_drift_risk  z_Default - z_DorsAttn
  valleys               runs with engagement < --threshold (default 0.35) lasting >= 6 s
                        (gaps <= 2 s inside a run are bridged first); end is exclusive
Output matches ml/schema/brain_schema.json. With the MOCK model, "mock": true and a MOCK note are set.

<id>.vertices.f16 (optional per-vertex playback for BrainCortexView(vertexActivity:)):
  raw little-endian float16, no header, row-major [second][vertex], N x 20,484, values 0..1
  (per-vertex z-score over the lecture, then clip(z / 2.5, 0, 1), matching BrainTimeline.activations).
  Only written when smaller than --max-f16-mb (default 8 MB, i.e. lectures up to ~195 s); otherwise skipped
  and any stale file removed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_common import LECTURES_DIR, N_FS5, NETWORKS, WORK_DIR, load_catalog  # noqa: E402
from prepare_stimulus import prepare  # noqa: E402
from run_brain_model import run_brain_model, save_predictions  # noqa: E402
from to_networks import DEFAULT_LAG, to_networks, zscore  # noqa: E402

ENGAGEMENT_GAIN = 1.5
VALLEY_THRESHOLD = 0.35
VALLEY_MIN_SECONDS = 6
VALLEY_MERGE_GAP = 2


def engagement_from(nets_z: np.ndarray) -> np.ndarray:
    i = [NETWORKS.index(k) for k in ("DorsAttn", "SalVentAttn", "Vis")]
    return 1.0 / (1.0 + np.exp(-ENGAGEMENT_GAIN * nets_z[:, i].mean(1)))


def drift_from(nets_z: np.ndarray) -> np.ndarray:
    return nets_z[:, NETWORKS.index("Default")] - nets_z[:, NETWORKS.index("DorsAttn")]


def find_valleys(eng: np.ndarray, threshold: float = VALLEY_THRESHOLD, min_len: int = VALLEY_MIN_SECONDS,
                 merge_gap: int = VALLEY_MERGE_GAP, hz: float = 1.0) -> list[dict]:
    low = np.asarray(eng) < threshold
    runs, start = [], None
    for i, b in enumerate(low):
        if b and start is None:
            start = i
        if not b and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(low)])
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [{"start": round(a / hz, 3), "end": round(b / hz, 3)} for a, b in merged if (b - a) / hz >= min_len]


def vertices_to_f16(vertices: np.ndarray) -> bytes:
    act = np.clip(zscore(vertices) / 2.5, 0, 1)
    return act.astype("<f2").tobytes()


def build_doc(lecture_id: str, nets_z: np.ndarray, mock: bool, note: str | None = None,
              threshold: float = VALLEY_THRESHOLD) -> dict:
    eng = engagement_from(nets_z)
    doc = {"lecture_id": lecture_id, "hz": 1, "mock": bool(mock)}
    if note:
        doc["note"] = note
    doc["networks"] = {k: np.round(nets_z[:, i], 3).tolist() for i, k in enumerate(NETWORKS)}
    doc["predicted_engagement"] = np.round(eng, 3).tolist()
    doc["predicted_drift_risk"] = np.round(drift_from(nets_z), 3).tolist()
    doc["valleys"] = find_valleys(np.round(eng, 3), threshold)
    return doc


def precompute(lecture_id: str, video: str | Path, out_dir: str | Path = LECTURES_DIR, lag: float = DEFAULT_LAG,
               threshold: float = VALLEY_THRESHOLD, write_vertices: bool = True, max_f16_mb: float = 8.0,
               reuse_stimulus: bool = False) -> dict:
    out_dir = Path(out_dir)
    man_path = WORK_DIR / lecture_id / "stimulus.json"
    if reuse_stimulus and man_path.exists():
        man = json.loads(man_path.read_text())
    else:
        man = prepare(video, lecture_id)
    transcript = json.loads(Path(man["transcript"]).read_text())
    pred = run_brain_model(man["video"], man["audio_wav"], transcript, duration=man["duration"])
    mock = bool(pred.get("mock"))
    save_predictions(pred, WORK_DIR / lecture_id / ("predictions_MOCK.npz" if mock else "predictions.npz"))

    res = to_networks(pred, lag=lag, duration=man["duration"], return_vertices=write_vertices)
    nets = res["networks"]
    n = nets.shape[0]
    f16_path = out_dir / f"{lecture_id}.vertices.f16"
    f16_note = ""
    if write_vertices:
        size = n * N_FS5 * 2
        if size <= max_f16_mb * 1_000_000:
            f16_path.write_bytes(vertices_to_f16(res["vertices"]))
            f16_note = f" vertices.f16 written ({size / 1e6:.1f} MB)."
        else:
            f16_path.unlink(missing_ok=True)
            f16_note = f" vertices.f16 skipped ({size / 1e6:.1f} MB > {max_f16_mb:g} MB)."
    note = (("MOCK: produced by the MOCK feature-driven encoder in ml/brain/run_brain_model.py, not by the brain "
             "model. " if mock else "") +
            f"Predicted brain response to this content (average brain). 1 Hz, shifted back {lag:g} s for "
            f"hemodynamic lag, z-scored within the lecture. Transcript: {man['transcript_source']}.{f16_note}")
    doc = build_doc(lecture_id, nets, mock, note, threshold)
    (out_dir / f"{lecture_id}.brain.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {out_dir / f'{lecture_id}.brain.json'}: {n}s, {len(doc['valleys'])} valleys"
          f"{' (MOCK)' if mock else ''}.{f16_note}")
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="lecture ids from Resources/Lectures/lectures.json")
    ap.add_argument("--all", action="store_true", help="every lecture in lectures.json that has a videoFile")
    ap.add_argument("--video", help="ad-hoc video path (use with --id)")
    ap.add_argument("--id")
    ap.add_argument("--out", default=str(LECTURES_DIR))
    ap.add_argument("--lag", type=float, default=DEFAULT_LAG, help="hemodynamic lag in seconds (default 5)")
    ap.add_argument("--threshold", type=float, default=VALLEY_THRESHOLD)
    ap.add_argument("--no-vertices", action="store_true")
    ap.add_argument("--max-f16-mb", type=float, default=8.0)
    ap.add_argument("--reuse-stimulus", action="store_true", help="reuse ml/data/brain/<id>/ from a previous run")
    a = ap.parse_args(argv)

    jobs: list[tuple[str, Path]] = []
    if a.video:
        jobs.append((a.id or Path(a.video).stem, Path(a.video)))
    catalog = {x["id"]: x for x in load_catalog()}
    wanted = list(catalog) if a.all else a.ids
    for lid in wanted:
        info = catalog.get(lid)
        if info is None:
            raise SystemExit(f"{lid}: not in lectures.json")
        if not info.get("videoFile"):
            print(f"skip {lid}: no videoFile (timer-driven lecture; its brain.json is maintained separately)")
            continue
        jobs.append((lid, LECTURES_DIR / info["videoFile"]))
    if not jobs:
        ap.error("give lecture ids, --all, or --video/--id")
    for lid, video in jobs:
        precompute(lid, video, a.out, a.lag, a.threshold, not a.no_vertices, a.max_f16_mb, a.reuse_stimulus)


if __name__ == "__main__":
    main()
