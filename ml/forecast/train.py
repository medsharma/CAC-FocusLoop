#!/usr/bin/env python3
"""Train the LapseForecaster (with and/or without brain-content features). One command:

    ml/.venv/bin/python ml/forecast/train.py                       # SYNTHETIC corpus, both variants
    ml/.venv/bin/python ml/forecast/train.py --manifest ml/data/forecast/manifest.csv   # real data

Inputs per sample: the previous 20 s of raw FeatureFrames (200 x 42, schema order) and the previous
20 s of ContentFeatures (20 x 11, `ContentFeatures.featureNames` order). Target: a positive event in
the next 15 s (rules in label_shift.py). Loss: plain BCE (default; keeps probabilities meaningful),
or focal loss / BCE with pos_weight (`--loss focal|weighted`). Training windows are augmented with
ml/common/augment.py (time warp, offset/scale, noise, face and HR dropout; `--no-augment` to disable).
After early stopping on validation AUROC, Platt scaling fitted on the validation split is folded into
the last layer, and the default alert threshold is the 90th percentile of the calibrated validation
predictions (10% alert rate). Split: subject-wise train/val/test (seeded); the split is stored in
the checkpoint and reused by eval.py / ablation.py / export_coreml.py.

Real-data manifest (CSV, paths relative to the manifest):
    subject,session_id,session_csv,sart_csv,es_csv,lecture_json,playhead_offset,synthetic
See ml/forecast/README.md for each file's columns.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import augment as A  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402
from model import LapseNet, focal_loss  # noqa: E402

DEFAULT_MANIFEST = HERE / "out/manifest.csv"
DEFAULT_OUT = HERE / "out/models"
ALERT_RATE = 0.10
WARP_CONTEXT = 50  # extra history frames for time-warp augmentation (<= FlatStore.EXTRA)


class SampleIndex:
    """Training/eval samples of a subject set: arrays (session index, time, target)."""

    def __init__(self, sessions, subjects, include_excluded=False):
        si, t, y = [], [], []
        for i, s in enumerate(sessions):
            if s.subject not in subjects:
                continue
            keep = np.ones(len(s.samples), bool) if include_excluded else ~s.exclude
            si.append(np.full(keep.sum(), i)); t.append(s.samples[keep]); y.append(s.y[keep])
        self.si = np.concatenate(si) if si else np.zeros(0, int)
        self.t = np.concatenate(t) if t else np.zeros(0)
        self.y = np.concatenate(y).astype(np.float32) if y else np.zeros(0, np.float32)

    def __len__(self):
        return len(self.t)



def _repo_rel(p) -> str:
    """Path relative to the repo root when inside it, so reports don't record local user paths."""
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[2]
    q = _pl.Path(p).resolve()
    try:
        return str(q.relative_to(root))
    except ValueError:
        return _pl.Path(p).name

def gather(store, ix, sel, aug=None):
    """Batch tensors; `aug` = (feature std, rng) applies ml/common/augment.py (time warp needs history)."""
    if aug is None:
        X, C = store.gather(ix.si[sel], ix.t[sel])
    else:
        X, C = store.gather(ix.si[sel], ix.t[sel], extra=WARP_CONTEXT)
        X = A.augment(X, D.WIN, aug[0], aug[1])
    return torch.from_numpy(X), torch.from_numpy(C), torch.from_numpy(ix.y[sel])


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * logit + b) on validation predictions (minimises NLL)."""
    z = torch.from_numpy(np.asarray(logits, np.float32)); yt = torch.from_numpy(np.asarray(y, np.float32))
    a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.item()), float(b.item())


@torch.no_grad()
def predict_at(net, store, si, t, bs=2048):
    net.eval()
    out = []
    for a in range(0, len(t), bs):
        X, C = store.gather(si[a:a + bs], t[a:a + bs])
        X, C = torch.from_numpy(X), torch.from_numpy(C)
        out.append(net(X, C if net.use_content else None).squeeze(1).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def predict(net, store, ix):
    return predict_at(net, store, ix.si, ix.t)


def norm_stats(sessions, subjects):
    F = np.concatenate([s.frames for s in sessions if s.subject in subjects])
    Cs = [s.content for s in sessions if s.subject in subjects and s.has_content]
    C = np.concatenate(Cs) if Cs else np.tile(D.NONE_CONTENT, (2, 1))
    return F.mean(0), F.std(0), C.mean(0), C.std(0)


def train_variant(sessions, store, split, use_content, args, synthetic):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr = SampleIndex(sessions, set(split["train"]))
    va = SampleIndex(sessions, set(split["val"]))
    pos = float(tr.y.mean())
    fm, fs, cm, cs = norm_stats(sessions, set(split["train"]))
    net = LapseNet(use_content, fm, fs, cm, cs, width=args.width)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    alpha = 1 - pos  # weight positives by the inverse prevalence
    pw = torch.tensor((1 - pos) / max(pos, 1e-3))
    yva = va.y
    best = (-1.0, None, 0)
    name = "content" if use_content else "no_content"
    print(f"[{name}] train samples={len(tr)} (pos {pos:.3f}) val={len(va)} loss={args.loss}")
    rng = np.random.default_rng(args.seed)
    aug = None if args.no_augment else (fs.astype(np.float32) + 1e-3, rng)
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        order = rng.permutation(len(tr))
        for a in range(0, len(order), args.batch):
            X, C, Y = gather(store, tr, order[a:a + args.batch], aug)
            lg = net.logits(X, C if use_content else None).squeeze(1)
            if args.loss == "focal":
                loss = focal_loss(lg, Y, alpha)
            elif args.loss == "weighted":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, Y, pos_weight=pw)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, Y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(Y)
        pva = predict(net, store, va)
        auc = M.auroc(yva, pva)
        print(f"[{name}] epoch {ep + 1:2d} loss={tot / len(tr):.4f} val_auroc={auc:.3f} ({time.time() - t0:.1f}s)",
              flush=True)
        if auc > best[0]:
            best = (auc, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        elif ep + 1 - best[2] >= args.patience:
            print(f"[{name}] early stop (best epoch {best[2]})"); break
    net.load_state_dict(best[1])
    # Platt calibration on validation, folded into the last layer so the exported model emits it
    pva = predict(net, store, va)
    lg = np.log(np.clip(pva, 1e-6, 1 - 1e-6) / np.clip(1 - pva, 1e-6, 1))
    pa, pb = fit_platt(lg, yva) if args.calibrate else (1.0, 0.0)
    with torch.no_grad():
        last = net.head[-1]
        last.weight.mul_(pa); last.bias.mul_(pa).add_(pb)
    pva = predict(net, store, va)
    thr = M.alert_threshold(pva, ALERT_RATE)
    print(f"[{name}] best epoch {best[2]} val AUROC {best[0]:.3f}; Platt a={pa:.3f} b={pb:.3f}; "
          f"alert threshold (90th pct of val) {thr:.3f}; val p quantiles "
          f"{np.round(np.quantile(pva, [0.1, 0.5, 0.9, 0.99]), 3).tolist()}")
    ckpt = {"state_dict": net.state_dict(), "use_content": use_content, "split": split,
            "norm": {"f_mean": fm, "f_std": fs, "c_mean": cm, "c_std": cs},
            "val_auroc": best[0], "best_epoch": best[2], "alert_rate": ALERT_RATE, "alert_threshold": thr,
            "train_pos_rate": pos, "loss": args.loss, "seed": args.seed, "width": args.width, "manifest": _repo_rel(args.manifest),
            "synthetic": synthetic, "feature_names": D.L.FEATS, "content_names": D.CONTENT_NAMES,
            "augment": not args.no_augment, "platt": [pa, pb]}
    return net, ckpt


def load_net(ckpt) -> LapseNet:
    n = ckpt["norm"]
    net = LapseNet(ckpt["use_content"], n["f_mean"], n["f_std"], n["c_mean"], n["c_std"], width=ckpt.get("width", 48))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-content", action="store_true", help="train only the features-only variant")
    g.add_argument("--content-only-variant", action="store_true", help="train only the with-content variant")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--loss", choices=["bce", "focal", "weighted"], default="bce",
                    help="bce (default) keeps probabilities calibrated; focal / weighted re-weight positives")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false", help="skip Platt scaling on val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--extra-manifest", help="manifest of sessions used ONLY for training (never val/test)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    if not pathlib.Path(args.manifest).exists():
        sys.exit(f"manifest not found: {args.manifest}\n(run ml/forecast/synth_forecast.py for the SYNTHETIC corpus)")
    sessions = D.load_manifest(args.manifest)
    synthetic = any(s.synthetic for s in sessions)
    split = D.split_subjects(sessions, args.split_seed)
    print(f"{len(sessions)} sessions, {len({s.subject for s in sessions})} subjects{' (SYNTHETIC)' if synthetic else ''}; "
          f"split train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])} subjects")
    if args.extra_manifest:  # train-only sessions (never val/test), e.g. a second SYNTHETIC generator
        extra = D.load_manifest(args.extra_manifest)
        for s in extra:
            s.subject = "extra:" + s.subject
        synthetic = synthetic or any(s.synthetic for s in extra)
        sessions = sessions + extra
        split["train"] = split["train"] + sorted({s.subject for s in extra})
        split["extra_manifest"] = _repo_rel(args.extra_manifest)
        print(f"+ {len(extra)} train-only sessions from {split['extra_manifest']}")
    store = D.FlatStore(sessions)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    variants = [False] if args.no_content else [True] if args.content_only_variant else [True, False]
    tag = "_SYNTHETIC" if synthetic else ""
    for uc in variants:
        _, ckpt = train_variant(sessions, store, split, uc, args, synthetic)
        p = out / f"forecaster_{'content' if uc else 'no_content'}{tag}.pt"
        torch.save(ckpt, p)
        print(f"saved {p} (val AUROC {ckpt['val_auroc']:.3f}{', SYNTHETIC' if synthetic else ''})")
    (out / f"split{tag}.json").write_text(json.dumps(split, indent=1))


if __name__ == "__main__":
    main()
