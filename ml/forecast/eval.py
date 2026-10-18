#!/usr/bin/env python3
"""Evaluate trained LapseForecaster checkpoints on the held-out TEST subjects.

Metrics (definitions in metrics.py):
  * AUROC over test samples (1 Hz, full 15 s horizon observed, not already in a lapse).
  * Precision (and recall) at a fixed 10% alert rate: the threshold is the 90th percentile of the
    test predictions, so exactly 10% of scored seconds alert. Chance precision = positive rate.
  * Lead time: median seconds between the first alert in the 15 s before each true lapse onset and
    the onset, plus the detection rate. Onsets = all positive events (label_shift.py rules).
    Predictions for lead time are made every second from t = 20 s, including in-lapse seconds.

Usage: ml/.venv/bin/python ml/forecast/eval.py [--models ml/forecast/out/models] [--manifest ...]
Results computed on the SYNTHETIC corpus are SYNTHETIC and are labelled so in every output.
"""
from __future__ import annotations

import argparse

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import heuristic as H  # noqa: E402
import label_shift as L  # noqa: E402
import metrics as M  # noqa: E402
from train import DEFAULT_MANIFEST, DEFAULT_OUT, load_net, predict_at  # noqa: E402

ALERT_RATE = 0.10


class TestGrid:
    """1 Hz prediction grid over test sessions + which grid points are scored samples."""

    def __init__(self, sessions, subjects):
        self.sessions = [i for i, s in enumerate(sessions) if s.subject in subjects]
        si, t, scored, y = [], [], [], []
        for i in self.sessions:
            s = sessions[i]
            g = L.sample_times(s.t_end - 0.1, full_horizon=False)
            yy = np.full(len(g), -1)
            keep = np.zeros(len(g), bool)
            pos = {round(x, 3): k for k, x in enumerate(g)}
            for k, (ts, yv, ex) in enumerate(zip(s.samples, s.y, s.exclude)):
                j = pos.get(round(ts, 3))
                if j is not None and not ex:
                    keep[j] = True; yy[j] = yv
            si.append(np.full(len(g), i)); t.append(g); scored.append(keep); y.append(yy)
        self.si, self.t = np.concatenate(si), np.concatenate(t)
        self.scored, self.y = np.concatenate(scored), np.concatenate(y)



def _repo_rel(p) -> str:
    """Path relative to the repo root when inside it, so reports don't record local user paths."""
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[2]
    q = _pl.Path(p).resolve()
    try:
        return str(q.relative_to(root))
    except ValueError:
        return _pl.Path(p).name

def score(sessions, grid: TestGrid, p: np.ndarray, alert_rate=ALERT_RATE) -> dict:
    ys, ps = grid.y[grid.scored], p[grid.scored]
    prec, rec, thr, realised = M.precision_at_alert_rate(ys, ps, alert_rate)
    leads = []
    for i in grid.sessions:
        m = grid.si == i
        leads.append(M.lead_times(grid.t[m], p[m], L.all_events(sessions[i].events), thr, L.HORIZON_S))
    leads = np.concatenate(leads) if leads else np.zeros(0)
    out = {"auroc": M.auroc(ys, ps), "precision_at_alert_rate": prec, "recall_at_alert_rate": rec,
           "alert_rate": alert_rate, "realised_alert_rate": realised, "threshold": thr,
           "positive_rate": float(ys.mean()), "n_samples": int(len(ys))}
    out.update(M.lead_summary(leads))
    out["_leads"] = leads
    return out


def coreml_predict(pkg, store, grid, uses_content, bs=1):
    """Predictions of a Core ML package (e.g. the previously shipped model) on the test grid."""
    import coremltools as ct
    ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = np.zeros(len(grid.t))
    for a in range(0, len(grid.t), 512):
        X, C = store.gather(grid.si[a:a + 512], grid.t[a:a + 512])
        for i in range(len(X)):
            feed = {"features": X[i:i + 1]}
            if uses_content:
                feed["content"] = C[i:i + 1]
            out[a + i] = float(np.asarray(ml.predict(feed)["lapse_probability"]).reshape(-1)[0])
    return out


def evaluate(models_dir=DEFAULT_OUT, manifest=DEFAULT_MANIFEST, include_heuristic=True, previous_dir=None):
    models_dir = pathlib.Path(models_dir)
    ckpts = sorted(models_dir.glob("forecaster_*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints in {models_dir}; run train.py first")
    sessions = D.load_manifest(manifest)
    synthetic = any(s.synthetic for s in sessions)
    store = D.FlatStore(sessions)
    results, preds, split = {}, {}, None
    for cp in ckpts:
        ck = torch.load(cp, weights_only=False)
        split = split or ck["split"]
        assert ck["split"] == split, "checkpoints were trained on different splits"
        grid = TestGrid(sessions, set(split["test"]))
        net = load_net(ck)
        p = predict_at(net, store, grid.si, grid.t)
        name = "with content" if ck["use_content"] else "without content"
        results[name] = score(sessions, grid, p); preds[name] = (grid, p)
        results[name]["val_auroc"] = ck["val_auroc"]
    if previous_dir:
        grid = TestGrid(sessions, set(split["test"]))
        for uc, pkg in [(True, "LapseForecaster"), (False, "LapseForecasterNoContent")]:
            p = coreml_predict(pathlib.Path(previous_dir) / f"{pkg}.mlpackage", store, grid, uc)
            name = f"previous model {'with' if uc else 'without'} content"
            results[name] = score(sessions, grid, p); preds[name] = (grid, p)
    if include_heuristic:
        grid = TestGrid(sessions, set(split["test"]))
        for uc, name in [(True, "heuristic with content"), (False, "heuristic without content")]:
            ps = []
            for a in range(0, len(grid.t), 4096):
                X, C = store.gather(grid.si[a:a + 4096], grid.t[a:a + 4096])
                ps.append(H.probability(X, C if uc else None))
            p = np.concatenate(ps)
            results[name] = score(sessions, grid, p); preds[name] = (grid, p)
    meta = {"synthetic": synthetic, "manifest": _repo_rel(manifest), "split": split,
            "test_subjects": len(split["test"]), "alert_rate": ALERT_RATE,
            "test_onsets": int(next(iter(results.values()))["onsets"])}
    return results, preds, meta


def strip(results):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(DEFAULT_OUT))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    a = ap.parse_args()
    res, _, meta = evaluate(a.models, a.manifest)
    tag = "SYNTHETIC " if meta["synthetic"] else ""
    print(f"{tag}test set: {meta['test_subjects']} subjects, {meta['test_onsets']} onsets")
    for k, v in strip(res).items():
        print(f"  {tag}{k:28s} AUROC={v['auroc']:.3f}  P@{int(ALERT_RATE * 100)}%={v['precision_at_alert_rate']:.3f} "
              f"(chance {v['positive_rate']:.3f})  median lead={v['median_lead_s']:.1f}s  detected={v['detection_rate']:.2f}")


if __name__ == "__main__":
    main()
