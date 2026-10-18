#!/usr/bin/env python3
"""Export trained LapseForecaster checkpoints to Core ML (.mlpackage, mlprogram, float16 compute).

    ml/.venv/bin/python ml/forecast/export_coreml.py [--models ml/forecast/out/models]

Writes Resources/Models/LapseForecaster.mlpackage           inputs "features" (1,200,42), "content" (1,20,11)
       Resources/Models/LapseForecasterNoContent.mlpackage  input  "features" (1,200,42)
both with output "lapse_probability" (1,1), raw (un-normalised) inputs, plus a JSON metadata file each
(<name>.json: feature order, alert threshold for a 10% alert rate, SYNTHETIC flag, parity check).
Parity: coremltools predict vs PyTorch on held-out windows (max abs diff must be < 0.02).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import coremltools as ct
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import train as T  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "Resources/Models"
NAMES = {True: "LapseForecaster", False: "LapseForecasterNoContent"}


class Prob(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, features, content=None):
        return self.net(features, content)


def export(ckpt_path, sessions, store, n_parity=64):
    ck = torch.load(ckpt_path, weights_only=False)
    uc = ck["use_content"]
    net = T.load_net(ck)
    wrap = Prob(net).eval()
    test = T.SampleIndex(sessions, set(ck["split"]["test"]))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(test), size=min(n_parity, len(test)), replace=False)
    X, C = store.gather(test.si[sel], test.t[sel])
    ex = (torch.from_numpy(X[:1]), torch.from_numpy(C[:1])) if uc else (torch.from_numpy(X[:1]),)
    traced = torch.jit.trace(wrap, ex)
    inputs = [ct.TensorType(name="features", shape=(1, D.WIN, D.N_FEAT), dtype=np.float32)]
    if uc:
        inputs.append(ct.TensorType(name="content", shape=(1, D.CWIN, D.N_CONTENT), dtype=np.float32))
    ml = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="lapse_probability", dtype=np.float32)],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS17)
    syn = bool(ck["synthetic"])
    name = NAMES[uc]
    ml.short_description = (("SYNTHETIC-TRAINED (not validated on real data). " if syn else "") +
                            f"FocusLoop {name}: P(lapse within 15 s) from 20 s of FeatureFrames"
                            + (" + brain-content features (predicted, average brain)." if uc else "."))
    ml.author = "FocusLoop ml/forecast"
    ml.version = "1"
    ml.input_description["features"] = "Raw FeatureFrame values, 200 x 42 (10 Hz, 20 s, schema order), oldest first"
    if uc:
        ml.input_description["content"] = "Raw ContentFeatures.vector, 20 x 11 (1 Hz), oldest first"
    ml.output_description["lapse_probability"] = "P(lapse begins within the next 15 s)"
    ml.user_defined_metadata.update({"synthetic": str(syn).lower(), "alert_threshold": f"{ck['alert_threshold']:.4f}",
                                     "val_auroc": f"{ck['val_auroc']:.4f}"})
    dst = OUT / f"{name}.mlpackage"
    if dst.exists():
        shutil.rmtree(dst)
    ml.save(str(dst))
    # parity
    with torch.no_grad():
        ref = net(torch.from_numpy(X), torch.from_numpy(C) if uc else None).squeeze(1).numpy()
    got = []
    loaded = ct.models.MLModel(str(dst))
    for i in range(len(X)):
        feed = {"features": X[i:i + 1]}
        if uc:
            feed["content"] = C[i:i + 1]
        got.append(float(np.asarray(loaded.predict(feed)["lapse_probability"]).reshape(-1)[0]))
    diff = float(np.max(np.abs(np.array(got) - ref)))
    meta = {"name": name, "synthetic": syn, "uses_content": uc,
            "inputs": {"features": [1, D.WIN, D.N_FEAT]} | ({"content": [1, D.CWIN, D.N_CONTENT]} if uc else {}),
            "output": {"lapse_probability": [1, 1]}, "feature_names": D.L.FEATS,
            "content_names": D.CONTENT_NAMES if uc else [], "window_s": D.L.WINDOW_S, "horizon_s": D.L.HORIZON_S,
            "alert_rate": ck["alert_rate"], "alert_threshold": round(float(ck["alert_threshold"]), 4),
            "val_auroc": round(float(ck["val_auroc"]), 4), "train_pos_rate": round(float(ck["train_pos_rate"]), 4),
            "loss": ck["loss"], "augment": ck.get("augment", False), "platt_scaling": ck.get("platt"),
            "seed": ck.get("seed"), "parity_max_abs_diff": round(diff, 5), "parity_n": len(X),
            "note": ("SYNTHETIC: trained on ml/forecast/synth_forecast.py data (live generator: simulated sensors "
                     "through a port of AttentionFeatureExtractor) + MOCK content. Not a result." if syn
                     else "Trained on real data listed in " + ck["manifest"])}
    tag = "_SYNTHETIC" if syn else ""
    abl = ROOT / f"ml/reports/forecast_ablation{tag}.json"
    if abl.exists():
        res = json.loads(abl.read_text())
        row = res["results"].get("with content" if uc else "without content")
        if row:
            meta["test_metrics" + ("_SYNTHETIC" if syn else "")] = {
                k: {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]} for k, v in row.items()}
            meta["test_metrics_seeds"] = res["meta"].get("seeds")
    tr = ROOT / "ml/reports/forecast_replay_trace_SYNTHETIC.json"
    if syn and tr.exists():
        s = json.loads(tr.read_text())["summary"].get("with content" if uc else "without content")
        if s:
            meta["replay_trace_SYNTHETIC"] = {"baseline_median_focused": s["baseline_median_focused"],
                                              "onsets": [{k: r[k] for k in ("onset_s", "p_minus10", "p_minus5",
                                                                            "mean_last8", "lead_s")} for r in s["onsets"]]}
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name}: saved {dst.relative_to(ROOT)}; parity max |coreml - torch| = {diff:.5f} over {len(X)} windows"
          f"{' (SYNTHETIC model)' if syn else ''}")
    assert diff < 0.02, f"Core ML parity failed for {name}: {diff}"
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(T.DEFAULT_OUT))
    ap.add_argument("--manifest", default=None, help="defaults to the manifest the checkpoints were trained on")
    a = ap.parse_args()
    ckpts = sorted(pathlib.Path(a.models).glob("forecaster_*.pt"))
    if not ckpts:
        sys.exit("no checkpoints; run train.py first")
    manifest = a.manifest or torch.load(ckpts[0], weights_only=False)["manifest"]
    sessions = D.load_manifest(manifest)
    store = D.FlatStore(sessions)
    OUT.mkdir(parents=True, exist_ok=True)
    for cp in ckpts:
        export(cp, sessions, store)


if __name__ == "__main__":
    main()
