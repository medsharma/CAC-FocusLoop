"""Shared constants and helpers for the offline brain-content pipeline (Mac only, never in the app).

Network order everywhere = BrainNetwork.allCases: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default.
Surface = fsaverage5: vertex i == fsaverage5 vertex i (LH 0-10241, RH 10242-20483), same as BrainMesh.bin.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LECTURES_DIR = REPO / "Resources" / "Lectures"
BRAIN_DIR = REPO / "Resources" / "Brain"
MESH_BIN = BRAIN_DIR / "BrainMesh.bin"
TRANSCRIPTS_DIR = HERE / "transcripts"
# Intermediate outputs (frames, wav, MOCK predictions). ml/data/ is gitignored.
WORK_DIR = REPO / "ml" / "data" / "brain"

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
N_FS5_HEMI = 10_242
N_FS5 = 2 * N_FS5_HEMI           # 20,484
N_FS7_HEMI = 163_842             # full fsaverage (ico7)
N_FS7 = 2 * N_FS7_HEMI


def load_mesh_networks(path: str | os.PathLike = MESH_BIN) -> np.ndarray:
    """`network` array from BrainMesh.bin (uint8, 0 = medial wall, 1..7 = NETWORKS[k-1])."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"FLBM":
        raise ValueError(f"{path}: bad magic")
    _, n, _ = struct.unpack("<III", raw[4:16])
    off = 16 + n * 12 + n * 12 + n  # positions, normals, ao
    return np.frombuffer(raw, np.uint8, count=n, offset=off).copy()


def load_annot_networks(annot_dir: str | os.PathLike = HERE) -> np.ndarray:
    """Network labels from the bundled Schaefer-400/7 .annot files (fsaverage5), same coding as BrainMesh.bin."""
    import nibabel as nib

    out = []
    for h in ("lh", "rh"):
        lab, _, names = nib.freesurfer.read_annot(os.path.join(annot_dir, f"{h}.Schaefer400_7.annot"))
        names = [x.decode() if isinstance(x, bytes) else x for x in names]
        lab = lab.clip(0)
        net = np.zeros(len(lab), np.uint8)
        for li, nm in enumerate(names):
            if li == 0:
                continue  # medial wall / unknown
            net[lab == li] = NETWORKS.index(nm.split("_")[2]) + 1
        out.append(net)
    return np.concatenate(out)


def load_network_labels(source: str = "annot") -> np.ndarray:
    """0/1..7 per fsaverage5 vertex. source = "annot" (default) or "mesh"."""
    return load_annot_networks() if source == "annot" else load_mesh_networks()


def labels_agree() -> tuple[bool, int]:
    """(agree, n_mismatched_vertices) between the .annot files and BrainMesh.bin."""
    a, m = load_annot_networks(), load_mesh_networks()
    if a.shape != m.shape:
        return False, max(len(a), len(m))
    bad = int((a != m).sum())
    return bad == 0, bad


def load_catalog(path: str | os.PathLike = LECTURES_DIR / "lectures.json") -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def find_transcript(lecture_id: str, video_path: str | os.PathLike | None = None) -> Path | None:
    """Sidecar transcript: ml/brain/transcripts/<id>.{vtt,srt}, else <video>.{vtt,srt} next to the video."""
    cands = [TRANSCRIPTS_DIR / f"{lecture_id}.{e}" for e in ("vtt", "srt")]
    if video_path is not None:
        v = Path(video_path)
        cands += [v.with_suffix(".vtt"), v.with_suffix(".srt")]
    return next((c for c in cands if c.exists()), None)
