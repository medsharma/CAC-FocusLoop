#!/usr/bin/env python3
"""Step 3: model predictions -> fsaverage5 vertices -> 7 network time series (1 Hz, lag-shifted, z-scored).

Handles the two output spaces in the spec:
  * volumetric MNI (4-D NIfTI) -> nilearn.surface.vol_to_surf onto the fsaverage5 pial surface
    (nilearn's bundled fsaverage5; nothing is downloaded)
  * full fsaverage (327,684 vertices) -> first 10,242 vertices per hemisphere (icosahedral meshes are nested)
  * fsaverage5 (20,484 vertices) is used as is.
Vertices are averaged per network with the bundled Schaefer-400/7 .annot labels (identical to the
`network` array in Resources/Brain/BrainMesh.bin; see brain_common.labels_agree()). No other atlas.

Then: shift back by the hemodynamic lag (default 5 s), resample to 1 Hz, z-score within the lecture.
Network order = Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_common import N_FS5, N_FS5_HEMI, N_FS7, N_FS7_HEMI, NETWORKS, load_network_labels  # noqa: E402

DEFAULT_LAG = 5.0


def _vol_to_fs5(img) -> np.ndarray:
    import nibabel as nib
    from nilearn import datasets, surface

    if isinstance(img, (str, Path)):
        img = nib.load(str(img))
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")                 # bundled with nilearn, offline
    left = surface.vol_to_surf(img, fs5.pial_left)                    # (10242, T) or (10242,)
    right = surface.vol_to_surf(img, fs5.pial_right)
    left, right = np.atleast_2d(left.T).T, np.atleast_2d(right.T).T
    return np.concatenate([left, right], 0).T.astype(np.float32)      # (T, 20484)


def to_fsaverage5(pred: dict) -> np.ndarray:
    """(n_timepoints, 20484) float32 from any supported prediction dict (see run_brain_model.py)."""
    space = pred["space"]
    if space == "mni_volume":
        return _vol_to_fs5(pred["img"])
    data = np.asarray(pred["data"], np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] in (N_FS5, N_FS7) and data.shape[1] not in (N_FS5, N_FS7):
        data = data.T                                                 # vertices x time -> time x vertices
    nv = data.shape[1]
    if space == "fsaverage" or nv == N_FS7:
        if nv != N_FS7:
            raise ValueError(f"fsaverage expects {N_FS7} vertices, got {nv}")
        return np.ascontiguousarray(np.concatenate([data[:, :N_FS5_HEMI],
                                                    data[:, N_FS7_HEMI:N_FS7_HEMI + N_FS5_HEMI]], 1))
    if space == "fsaverage5":
        if nv != N_FS5:
            raise ValueError(f"fsaverage5 expects {N_FS5} vertices, got {nv}")
        return data
    raise ValueError(f"unknown space {space!r}")


def network_means(vertex_ts: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """(T, 7) mean over the vertices of each network (medial wall excluded, NaNs ignored)."""
    labels = load_network_labels("annot") if labels is None else labels
    out = np.zeros((vertex_ts.shape[0], 7), np.float32)
    for k in range(7):
        m = labels == k + 1
        out[:, k] = np.nanmean(vertex_ts[:, m], 1, dtype=np.float64) if m.any() else 0.0
    return out


def lag_resample(x: np.ndarray, tr: float, lag: float = DEFAULT_LAG, hz: float = 1.0,
                 n_out: int | None = None, t0: float = 0.0) -> np.ndarray:
    """Sample i of `x` (T, ...) is the response at t0 + i*tr, which reflects the stimulus at t0 + i*tr - lag.
    Returns stimulus-aligned values at t = 0, 1/hz, 2/hz ... (linear interpolation, edges held)."""
    x = np.asarray(x, np.float32)
    T = x.shape[0]
    if n_out is None:
        n_out = max(1, int(np.floor(((T - 1) * tr + t0 - lag) * hz)) + 1)
    t = np.arange(n_out) / hz
    pos = np.clip((t + lag - t0) / tr, 0, T - 1)                     # fractional source index
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0).astype(np.float32).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    mu = x.mean(axis, keepdims=True)
    sd = x.std(axis, keepdims=True)
    return np.where(sd > 1e-8, (x - mu) / np.where(sd > 1e-8, sd, 1), 0).astype(np.float32)


def to_networks(pred: dict, lag: float = DEFAULT_LAG, hz: float = 1.0, duration: float | None = None,
                z: bool = True, labels: np.ndarray | None = None, return_vertices: bool = False) -> dict:
    """Full step 3. Returns {"networks": (N, 7) [z-scored unless z=False], "vertices": (N, 20484) | None,
    "hz": hz, "lag": lag, "keys": NETWORKS}. N = floor(duration * hz) when duration is given."""
    v = to_fsaverage5(pred)
    nets = network_means(v, labels)
    n_out = int(np.floor(duration * hz)) if duration is not None else None
    tr, t0 = float(pred["tr"]), float(pred.get("t0", 0.0))
    nets = lag_resample(nets, tr, lag, hz, n_out, t0)
    verts = lag_resample(v, tr, lag, hz, n_out, t0) if return_vertices else None
    return {"networks": zscore(nets) if z else nets, "vertices": verts, "hz": hz, "lag": lag, "keys": NETWORKS}


if __name__ == "__main__":
    import json
    from run_brain_model import run_brain_model

    m = json.loads(Path(sys.argv[1]).read_text())
    p = run_brain_model(m["video"], m["audio_wav"], json.loads(Path(m["transcript"]).read_text()),
                        duration=m["duration"])
    r = to_networks(p, duration=m["duration"])
    print(m["lecture_id"], "networks", r["networks"].shape, "mock" if p.get("mock") else "")
    for k, name in enumerate(NETWORKS):
        print(f"  {name:12s} min {r['networks'][:, k].min():+.2f} max {r['networks'][:, k].max():+.2f}")
