#!/usr/bin/env python3
"""Train the attention-state model.

    ml/.venv/bin/python ml/state/train.py --data <dir-of-csvs> [--synthetic] [--out ml/state/runs/<tag>]

Works for SYNTHETIC data (ml/synth/generate_dataset.py) and for the app's REAL label exports
(t,label,lapse + 42 features; rows with an empty label are skipped as targets).

* Split: subject-wise train/val/test (falls back to a chronological split with a 10 s gap when there
  are < 3 subjects).
* Samples: every labelled frame (from the 100th of a segment on) is a possible window end; each epoch
  draws `--epoch-size` of them at random. Target = label of the window's last frame (the CSV `label`
  column, i.e. WITH label noise, as real labels would be).
* Augmentation (ml/common/augment.py; off with --no-augment): time warp, per-window per-feature
  offset/scale, Gaussian noise, face-dropout spans, HR dropout.
* Loss: cross-entropy with inverse-sqrt-frequency class weights and label smoothing 0.05.
* Early stopping on validation macro-F1; then post-hoc calibration on the validation split: a
  temperature T and a per-class bias b (p = softmax(logits / T + b)), both baked into the model.
Writes <out>/model.pt (weights + normalisation stats + split) and <out>/norm.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import augment as A  # noqa: E402
from data import CLASSES, FEATS, HZ, ROOT, WINDOW, WindowStore, load_dir, subject_split, temporal_split  # noqa: E402
from model import AttentionStateNet, n_params, save_checkpoint  # noqa: E402

WARP_CONTEXT = 30  # extra history frames for time-warp augmentation


def _repo_rel(p) -> str:
    """Path relative to the repo root when inside it, so checkpoints/reports don't record local user paths."""
    q = pathlib.Path(p).resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return pathlib.Path(p).name


def split_indices(groups: np.ndarray, seed: int, gap: int = WINDOW):
    """Indices into the window arrays for train/val/test. `gap` = windows dropped at each boundary of
    the chronological fallback so overlapping windows don't leak across splits."""
    sp = subject_split(list(groups), seed)
    if sp is not None:
        return {k: np.nonzero(np.isin(groups, v))[0] for k, v in sp.items()}, {"kind": "subject", **sp}
    idx = temporal_split(len(groups))
    idx = {"train": idx["train"][:-gap] if len(idx["train"]) > gap else idx["train"],
           "val": idx["val"][:-gap] if len(idx["val"]) > gap else idx["val"],
           "test": idx["test"]}
    return idx, {"kind": "temporal", "n": int(len(groups)), "gap": gap}


def resolve_split(groups: np.ndarray, info: dict):
    """Recreate split indices from a checkpoint's split info (used by eval.py)."""
    if info["kind"] == "subject":
        return {k: np.nonzero(np.isin(groups, info[k]))[0] for k in ("train", "val", "test")}
    idx = temporal_split(len(groups))
    g = info.get("gap", WINDOW)
    return {"train": idx["train"][:-g], "val": idx["val"][:-g], "test": idx["test"]}


def strided_subset(store: WindowStore, idx: np.ndarray, stride: int) -> np.ndarray:
    """The members of `idx` that are classic sliding-window ends (stride in frames)."""
    return np.intersect1d(idx, store.strided(stride))


def predict_logits(model, X, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model.logits(torch.from_numpy(np.ascontiguousarray(X[i:i + bs]))))
    return torch.cat(out) if out else torch.zeros(0, len(CLASSES))


def fit_calibration(logits: torch.Tensor, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Temperature + per-class bias minimising validation NLL."""
    if len(y) == 0:
        return 1.0, np.zeros(len(CLASSES))
    log_t = torch.zeros(1, requires_grad=True)
    b = torch.zeros(len(CLASSES), requires_grad=True)
    yt = torch.from_numpy(y)
    opt = torch.optim.LBFGS([log_t, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp() + b, yt) + 1e-3 * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    T = float(np.clip(log_t.exp().item(), 0.25, 10.0))
    return T, (b.detach().numpy() - b.detach().numpy().mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--synthetic", action="store_true", help="label all outputs SYNTHETIC (auto if any CSV says so)")
    ap.add_argument("--out")
    ap.add_argument("--extra-train-data", help="CSV dir used ONLY for training (never val/test), e.g. a second "
                    "SYNTHETIC generator for diversity")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=60000, help="random windows per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stride", type=int, default=10, help="validation / eval window stride in frames (10 = 1 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--threads", type=int, default=6, help="torch CPU threads")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sessions = load_dir(a.data)
    synthetic = a.synthetic or any(s.synthetic for s in sessions)
    tag = "SYNTHETIC" if synthetic else "REAL"
    out = pathlib.Path(a.out or ROOT / "ml/state/runs" / tag)
    out.mkdir(parents=True, exist_ok=True)

    n_primary = len(sessions)
    if a.extra_train_data:
        extra = load_dir(a.extra_train_data)
        for s in extra:
            s.subject = "extra:" + s.subject
        synthetic = synthetic or any(s.synthetic for s in extra)
        sessions = sessions + extra
    store = WindowStore(sessions)
    if len(store.y) == 0:
        raise SystemExit("no labelled windows found (labels must be focused/distracted/drowsy/stressed)")
    is_extra = np.char.startswith(store.groups.astype(str), "extra:")
    idx, split_info = split_indices(store.groups[~is_extra], a.seed)
    prim = np.nonzero(~is_extra)[0]
    idx = {k: prim[v] for k, v in idx.items()}
    tr = np.concatenate([idx["train"], np.nonzero(is_extra)[0]])
    if a.extra_train_data:
        split_info["extra_train_data"] = _repo_rel(a.extra_train_data)
        split_info["extra_train_sessions"] = len(sessions) - n_primary
    va = strided_subset(store, idx["val"], a.stride)
    ytr = store.y[tr]
    print(f"[{tag}] {len(sessions)} sessions, {len(set(store.groups))} subjects, {len(store.y)} window ends; "
          f"split={split_info['kind']} train={len(tr)} val={len(va)} (stride {a.stride}) test={len(idx['test'])}")

    counts = np.bincount(ytr, minlength=len(CLASSES))
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0)
    w = w / w[counts > 0].mean()
    print("train class counts", dict(zip(CLASSES, counts.tolist())), "weights", np.round(w, 3).tolist())

    model = AttentionStateNet(width=a.width, layers=a.layers)
    frames = store.F[store.ends[tr]]
    model.set_norm(frames.mean(0), frames.std(0))
    std = model.std.numpy()
    print(f"params: {n_params(model)}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    steps = a.epochs * int(np.ceil(min(a.epoch_size, len(tr)) / a.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)
    Xva, yva = store.gather(store.ends[va]), store.y[va]

    best, best_state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        sel = rng.choice(tr, size=min(a.epoch_size, len(tr)), replace=False)
        tl, t0 = 0.0, time.time()
        for i in range(0, len(sel), a.batch):
            b = sel[i:i + a.batch]
            if a.no_augment:
                xb = store.gather(store.ends[b])
            else:
                xb = A.augment(store.gather(store.ends[b], WINDOW + WARP_CONTEXT), WINDOW, std, rng)
            loss = lossf(model.logits(torch.from_numpy(xb)), torch.from_numpy(store.y[b]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(b)
        if len(yva):
            pv = predict_logits(model, Xva).argmax(1).numpy()
            f1 = f1_score(yva, pv, average="macro", labels=np.unique(yva))
        else:
            f1 = -tl
        print(f"epoch {ep + 1:2d} loss {tl / len(sel):.4f} val macroF1 {f1:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if f1 > best:
            best, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break
    model.load_state_dict(best_state)
    T, bias = fit_calibration(predict_logits(model, Xva), yva)
    model.temperature.fill_(T)
    model.class_bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    print(f"best val macroF1 {best:.4f}; temperature {T:.3f}; class bias {np.round(bias, 3).tolist()}")

    norm = {"features": FEATS, "mean": model.mean.tolist(), "std": model.std.tolist()}
    (out / "norm.json").write_text(json.dumps(norm, indent=1))
    save_checkpoint(out / "model.pt", model, {
        "tag": tag, "synthetic": synthetic, "data": _repo_rel(a.data),
        "split": split_info, "stride": a.stride, "classes": CLASSES, "features": FEATS,
        "val_macro_f1": best, "temperature": T, "class_bias": bias.tolist(), "class_weights": w.tolist(),
        "train_class_counts": counts.tolist(), "n_params": n_params(model), "augment": not a.no_augment,
        "seed": a.seed})
    print(f"saved {out / 'model.pt'} [{tag}]")
    return out


if __name__ == "__main__":
    main()
