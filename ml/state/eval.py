#!/usr/bin/env python3
"""Evaluate a trained attention-state checkpoint on its held-out test split and on the bundled replay.

    ml/.venv/bin/python ml/state/eval.py --data <dir-of-csvs> [--ckpt ml/state/runs/<TAG>/model.pt]
        [--synthetic] [--reports ml/reports] [--split test] [--replay Resources/Replay/synthetic_session.replay.csv]
        [--baseline-mlpackage OLD.mlpackage]

Writes ml/reports/state_<TAG>_confusion.png, state_<TAG>_reliability.png, state_<TAG>_metrics.json and
(with a replay) state_<TAG>_replay_confusion.png + state_<TAG>_model_vs_heuristic.png.

Top-level metrics = the model on the test split against the generator's ground truth (`true_label`
column) when the CSVs have one, else against `label`. Also reported: the same against the noisy
`label` column, the Python port of HeuristicAttentionClassifier (heuristic.py) on the same windows,
the app's as-deployed blend (heuristic override when the face is missing, averaging below 0.45
confidence), the replay CSV, and optionally a previous model (.mlpackage) for before/after.
<TAG> is SYNTHETIC whenever the data or checkpoint is synthetic; it is also in every plot title.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import heuristic as H  # noqa: E402
from data import CLASSES, ROOT, WindowStore, load_csv, load_dir  # noqa: E402
from model import load_checkpoint  # noqa: E402
from train import resolve_split, strided_subset  # noqa: E402

REPLAY = ROOT / "Resources/Replay/synthetic_session.replay.csv"


def _repo_rel(p) -> str:
    """Path relative to the repo root when inside it, so reports don't record local user paths."""
    q = pathlib.Path(p).resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return pathlib.Path(p).name


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Top-label reliability bins and expected calibration error."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "confidence": None, "accuracy": None})
            continue
        c, acc = float(conf[m].mean()), float(correct[m].mean())
        ece += m.mean() * abs(acc - c)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "confidence": c, "accuracy": acc})
    return bins, float(ece)


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y, pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = s > 0
    bins, ece = reliability(probs, y)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "macro_f1": float(f[present].mean()) if present.any() else None,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": cm.tolist()},
        "ece": ece,
        "reliability_bins": bins,
    }


def short(m: dict) -> dict:
    return {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "ece": m["ece"],
            "f1": {c: v["f1"] for c, v in m["per_class"].items()}}


def model_probs(model, X, bs=1024) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])))[0].numpy()
                               for i in range(0, len(X), bs)]) if len(X) else np.zeros((0, 4))


def mlpackage_probs(path, X) -> np.ndarray:
    import coremltools as ct
    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    return np.concatenate([np.asarray(ml.predict({"window": x[None].astype(np.float32)})["probabilities"]).reshape(1, -1)
                           for x in X])


def plot_confusion(metrics: dict, tag: str, path: pathlib.Path, title: str):
    cm = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(4), CLASSES, rotation=30); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"[{tag}] {title}\nmacro-F1 {metrics['macro_f1']:.3f}, acc {metrics['accuracy']:.3f}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_reliability(metrics: dict, tag: str, path: pathlib.Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4.8))
    b = [x for x in metrics["reliability_bins"] if x["count"]]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "o-", label="model")
    for x in b:
        ax.annotate(str(x["count"]), (x["confidence"], x["accuracy"]), fontsize=6, xytext=(3, -8),
                    textcoords="offset points")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (top-class probability)"); ax.set_ylabel("accuracy")
    ax.set_title(f"[{tag}] {title}\nECE {metrics['ece']:.3f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_comparison(rows: dict, tag: str, path: pathlib.Path):
    """Macro-F1 of each classifier on held-out SYNTHETIC subjects and on the replay."""
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6), facecolor="#fcfcfb")
    for ax, (key, title) in zip(axes, [("heldout", "Held-out SYNTHETIC subjects (true labels)"),
                                       ("replay", "Bundled SYNTHETIC replay (180 s)")]):
        vals = [rows[n].get(key, {}).get("macro_f1") or 0 for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=["#2a78d6" if "model" in n else "#9a9a95" for n in names], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.set_xlim(0, 1.08); ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("macro-F1")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"[{tag}] Attention-state classifiers, macro-F1  |  SYNTHETIC DATA, NOT A RESULT",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--reports", default=str(ROOT / "ml/reports"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--replay", default=str(REPLAY), help="replay CSV to score too ('' to skip)")
    ap.add_argument("--baseline-mlpackage", help="previous model to compare against (before/after)")
    a = ap.parse_args(argv)

    sessions = load_dir(a.data)
    data_synth = a.synthetic or any(s.synthetic for s in sessions)
    ckpt = pathlib.Path(a.ckpt or ROOT / "ml/state/runs" / ("SYNTHETIC" if data_synth else "REAL") / "model.pt")
    model, ck = load_checkpoint(ckpt)
    synthetic = data_synth or bool(ck.get("synthetic"))
    tag = "SYNTHETIC" if synthetic else "REAL"
    stride = ck.get("stride", 10)

    store = WindowStore(sessions)
    split = resolve_split(store.groups, ck["split"])
    idx = np.arange(len(store.y)) if a.split == "all" else split[a.split]
    idx = strided_subset(store, idx, stride)
    if len(idx) == 0:
        raise SystemExit(f"empty {a.split} split")
    X = store.gather(store.ends[idx])
    y_true, y_noisy = store.y_true[idx], store.y[idx]
    has_true = bool((y_true != y_noisy).any())
    P = model_probs(model, X)
    Hp, _ = H.predict(X)
    Bp = H.blended(P, X)
    m = evaluate(P, y_true)
    m.update({"tag": tag, "SYNTHETIC": synthetic, "split": a.split, "split_kind": ck["split"]["kind"],
              "labels": "true_label (generator ground truth)" if has_true else "label",
              "test_subjects": ck["split"].get("test"), "checkpoint": _repo_rel(ckpt),
              "temperature": ck.get("temperature"), "class_bias": ck.get("class_bias"),
              "n_params": ck.get("n_params"), "window_stride_frames": stride,
              "note": ("SYNTHETIC data: these numbers only show the pipeline works; they are NOT real results."
                       if synthetic else "Real exported data.")})
    m["noisy_labels"] = short(evaluate(P, y_noisy))
    m["label_noise_rate_test"] = float((y_true != y_noisy).mean())
    comp = {"model (new)": {"heldout": short(m)},
            "as deployed (blend)": {"heldout": short(evaluate(Bp, y_true))},
            "heuristic": {"heldout": short(evaluate(Hp, y_true))}}
    base_P = None
    if a.baseline_mlpackage:
        base_P = mlpackage_probs(a.baseline_mlpackage, X)
        comp["previous model"] = {"heldout": short(evaluate(base_P, y_true))}
        comp["previous as deployed"] = {"heldout": short(evaluate(H.blended(base_P, X), y_true))}

    reports = pathlib.Path(a.reports)
    reports.mkdir(parents=True, exist_ok=True)
    plot_confusion(m, tag, reports / f"state_{tag}_confusion.png", f"Attention state confusion ({a.split}, held-out subjects)")
    plot_reliability(m, tag, reports / f"state_{tag}_reliability.png", f"Reliability ({a.split}, held-out subjects)")

    if a.replay and pathlib.Path(a.replay).exists():
        rs = load_csv(a.replay)
        rst = WindowStore(rs)
        k = rst.strided(10)
        Xr, yr = rst.gather(rst.ends[k]), rst.y[k]
        Pr = model_probs(model, Xr)
        Hr, fm = H.predict(Xr)
        rm = evaluate(Pr, yr)
        m["replay"] = {"file": _repo_rel(a.replay), "model": short(rm),
                       "as_deployed": short(evaluate(H.blended(Pr, Xr), yr)),
                       "heuristic": short(evaluate(Hr, yr)), "face_missing_windows": int(fm.sum()),
                       "confusion_matrix": rm["confusion_matrix"]}
        comp["model (new)"]["replay"] = m["replay"]["model"]
        comp["as deployed (blend)"]["replay"] = m["replay"]["as_deployed"]
        comp["heuristic"]["replay"] = m["replay"]["heuristic"]
        if a.baseline_mlpackage:
            bp = mlpackage_probs(a.baseline_mlpackage, Xr)
            m["replay"]["previous_model"] = short(evaluate(bp, yr))
            comp["previous model"]["replay"] = m["replay"]["previous_model"]
            comp["previous as deployed"]["replay"] = short(evaluate(H.blended(bp, Xr), yr))
        plot_confusion(rm, tag, reports / f"state_{tag}_replay_confusion.png", "Attention state confusion (bundled replay)")
    m["comparison"] = comp
    plot_comparison(comp, tag, reports / f"state_{tag}_model_vs_heuristic.png")
    pj = reports / f"state_{tag}_metrics.json"
    pj.write_text(json.dumps(m, indent=1))
    print(f"[{tag}] {a.split} ({m['labels']}): n={m['n']} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} ECE={m['ece']:.4f}")
    for c, v in m["per_class"].items():
        print(f"  {c:10s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    for name, v in comp.items():
        print(f"  {name:22s} " + "  ".join(f"{k}: acc={x['accuracy']:.3f} mF1={x['macro_f1']:.3f} ECE={x['ece']:.3f}"
                                           for k, x in v.items()))
    print(f"wrote reports to {reports}")
    return m


if __name__ == "__main__":
    main()
