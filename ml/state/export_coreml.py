#!/usr/bin/env python3
"""Export a trained attention-state checkpoint to Core ML (ML Program, float16).

    ml/.venv/bin/python ml/state/export_coreml.py [--ckpt ml/state/runs/<TAG>/model.pt]
        [--out Resources/Models/AttentionStateModel.mlpackage] [--metrics ml/reports/state_<TAG>_metrics.json]

I/O contract (normalisation and calibration temperature are baked in; pass RAW FeatureFrame.vector values):
  input  "window"        float32 (1, 100, 42)  oldest frame first, columns in feature_schema.json order
  output "probabilities" float32 (1, 4)        softmax in AttentionState.allCases order
                                               (focused, distracted, drowsy, stressed)
  output "embedding"     float32 (1, E)        penultimate-layer embedding for the personal head
Also writes <out>.json metadata next to the package, and verifies coremltools predict vs torch (macOS).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import CLASSES, FEATS, HZ, ROOT, WINDOW  # noqa: E402
from model import load_checkpoint  # noqa: E402


def convert(model, out: pathlib.Path, synthetic: bool, description: str = ""):
    import coremltools as ct
    model.eval()
    example = torch.randn(1, WINDOW, len(FEATS))
    traced = torch.jit.trace(model, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="window", shape=(1, WINDOW, len(FEATS)), dtype=np.float32)],
        outputs=[ct.TensorType(name="probabilities", dtype=np.float32),
                 ct.TensorType(name="embedding", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    label = "SYNTHETIC-trained" if synthetic else "trained on real exported labels"
    ml.short_description = (f"FocusLoop attention-state classifier ({label}). "
                            "10 s window of raw 10 Hz FeatureFrames -> 4-state probabilities + embedding. "
                            "Wellness use only; not a medical device. " + description)
    ml.author = "FocusLoop"
    ml.version = "1"
    ml.input_description["window"] = ("(1,100,42) raw FeatureFrame.vector values, oldest first, "
                                      "feature_schema.json order")
    ml.output_description["probabilities"] = "(1,4) softmax: focused, distracted, drowsy, stressed"
    ml.output_description["embedding"] = "(1,E) penultimate embedding for the on-device personal head"
    ml.user_defined_metadata["training_data"] = "SYNTHETIC" if synthetic else "REAL"
    ml.user_defined_metadata["classes"] = ",".join(CLASSES)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    return ml


def parity(model, mlpath: pathlib.Path, X: np.ndarray):
    """Max abs diff between torch (fp32) and Core ML (fp16 compute) outputs. macOS only."""
    if platform.system() != "Darwin":
        return None
    import coremltools as ct
    ml = ct.models.MLModel(str(mlpath), compute_units=ct.ComputeUnit.CPU_ONLY)
    dp, de, agree = 0.0, 0.0, 0
    for x in X:
        with torch.no_grad():
            p, e = model(torch.from_numpy(x[None]))
        o = ml.predict({"window": x[None].astype(np.float32)})
        dp = max(dp, float(np.abs(o["probabilities"] - p.numpy()).max()))
        de = max(de, float(np.abs(o["embedding"] - e.numpy()).max()))
        agree += int(o["probabilities"].argmax() == p.numpy().argmax())
    return {"n": int(len(X)), "max_abs_diff_probabilities": dp, "max_abs_diff_embedding": de,
            "argmax_agreement": agree / max(1, len(X)), "compute_units": "CPU_ONLY",
            "output_shapes": {k: list(np.shape(v)) for k, v in o.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "ml/state/runs/SYNTHETIC/model.pt"))
    ap.add_argument("--out", default=str(ROOT / "Resources/Models/AttentionStateModel.mlpackage"))
    ap.add_argument("--metrics", help="eval metrics JSON to embed (default: ml/reports/state_<TAG>_metrics.json)")
    ap.add_argument("--data", help="optional CSV dir: use real windows for the parity check")
    ap.add_argument("--n-check", type=int, default=64)
    a = ap.parse_args(argv)

    model, ck = load_checkpoint(a.ckpt)
    synthetic = bool(ck.get("synthetic", True))
    tag = "SYNTHETIC" if synthetic else "REAL"
    out = pathlib.Path(a.out)
    convert(model, out, synthetic)

    rng = np.random.default_rng(0)
    if a.data:
        from data import load_dir, make_windows
        Xw, _, _ = make_windows(load_dir(a.data), stride=50)
        X = Xw[rng.choice(len(Xw), min(a.n_check, len(Xw)), replace=False)]
    else:  # raw-scale random windows around the training distribution
        X = (model.mean.numpy() + model.std.numpy() * rng.normal(size=(a.n_check, WINDOW, len(FEATS)))).astype(np.float32)
    par = parity(model, out, X)

    mp = pathlib.Path(a.metrics or ROOT / f"ml/reports/state_{tag}_metrics.json")
    metrics = json.loads(mp.read_text()) if mp.exists() else None
    if metrics:
        metrics = {k: metrics[k] for k in ("n", "accuracy", "macro_f1", "ece", "per_class", "split", "split_kind",
                                            "labels", "label_noise_rate_test", "noisy_labels", "confusion_matrix",
                                            "replay", "comparison") if k in metrics}
        metrics.get("replay", {}).pop("confusion_matrix", None)
    cfg = ck["config"]
    meta = {
        "model": "AttentionStateModel",
        "training_data": tag,
        "SYNTHETIC": synthetic,
        "note": ("Trained on SYNTHETIC data from ml/synth/generate_dataset.py (live generator: simulated "
                 "sensors run through a port of AttentionFeatureExtractor, ~5% label noise). Metrics below are "
                 "SYNTHETIC and are not real results. Retrain on real exports with ml/state/retrain.sh."
                 if synthetic else "Trained on real exported labels."),
        "architecture": "dilated 1D-CNN (%d conv layers, width %d, dilations 1..%d) + mean/max/last-%d-step "
                        "pooling + linear embedding" % (cfg.get("layers", 3), cfg["width"],
                                                        2 ** (cfg.get("layers", 3) - 1), cfg.get("tail", 0)),
        "training": {"augmentation": ck.get("augment", False) and
                     "time warp, per-window offset/scale, Gaussian noise, face-dropout spans, HR dropout",
                     "split": "subject-wise", "seed": ck.get("seed"), "val_macro_f1_noisy_labels": ck.get("val_macro_f1"),
                     "class_weights": ck.get("class_weights")},
        "calibration": {"temperature": ck.get("temperature"), "class_bias": ck.get("class_bias"),
                        "formula": "softmax(logits / temperature + class_bias), fitted on validation subjects"},
        "n_params": ck.get("n_params"),
        "rate_hz": HZ,
        "window_frames": WINDOW,
        "window_seconds": WINDOW // HZ,
        "feature_count": len(FEATS),
        "feature_names": FEATS,
        "classes": CLASSES,
        "embedding_dim": ck["config"]["embed"],
        "io": {
            "inputs": {"window": {"dtype": "float32", "shape": [1, WINDOW, len(FEATS)],
                                  "layout": "[batch, time (oldest first), feature (schema order)]",
                                  "values": "RAW FeatureFrame.vector (normalisation baked in)"}},
            "outputs": {"probabilities": {"dtype": "float32", "shape": [1, len(CLASSES)], "order": CLASSES},
                        "embedding": {"dtype": "float32", "shape": [1, ck["config"]["embed"]]}},
        },
        "short_history_padding": "if fewer than 100 frames are available, left-pad by repeating the oldest frame",
        "normalisation": {"mean": model.mean.tolist(), "std": model.std.tolist(), "clip": 8.0,
                          "baked_into_model": True},
        "temperature": ck.get("temperature"),
        "precision": "float16 (ML Program, iOS17)",
        "metrics": metrics,
        "parity_check": par,
    }
    mj = out.with_suffix(".json")
    mj.write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} and {mj} [{tag}]")
    print("parity:", json.dumps(par))
    return meta


if __name__ == "__main__":
    main()
