#!/usr/bin/env python3
"""Lapse-risk trace on the bundled SYNTHETIC replay (what the Simulator demo shows).

    ml/.venv/bin/python ml/forecast/replay_trace.py [--models ml/forecast/out/models]
        [--mlpackage-dir Resources/Models] [--tag new] [--reports ml/reports]

Feeds Resources/Replay/synthetic_session.replay.csv (SYNTHETIC, 180 s) through the forecaster exactly
as SessionPipeline does in Study Mode with the MOCK sample lecture: every second t, the last 20 s of
frames (front-padded with the oldest frame) and the content features of playheads t-19..t
(playhead = session time) from Resources/Brain/sample_lecture.brain.json (MOCK).
Scores PyTorch checkpoints (default) or Core ML packages (--mlpackage-dir, e.g. to trace the previous
models). Writes ml/reports/forecast_replay_trace_SYNTHETIC.{png,json,md}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import label_shift as L  # noqa: E402

ROOT = HERE.parents[1]
REPLAY = ROOT / "Resources/Replay/synthetic_session.replay.csv"
LECTURE = ROOT / "Resources/Brain/sample_lecture.brain.json"
REPORTS = ROOT / "ml/reports"


def replay_inputs():
    df = L.load_session(REPLAY)
    frames = df[L.FEATS].to_numpy(np.float32)
    doc = json.loads(LECTURE.read_text())
    t_end = float(df["t"].iloc[-1]) + 0.1
    seconds = int(np.ceil(t_end)) + 1
    content = np.stack([D.content_features(doc, float(s)) for s in range(seconds)])
    sess = D.Session("replay", "replay", frames, content, True, {}, L.lapse_mask(df), True, t_end)
    onsets = L.label_onsets(df["t"].to_numpy(float), L.lapse_mask(df))
    lapses = []
    m = L.lapse_mask(df); t = df["t"].to_numpy(float)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        lapses.append((float(t[a]), float(t[b - 1]) + 0.1, str(df["label"].iloc[a])))
    return sess, doc, onsets, lapses


def trace_torch(ckpt_path, store, ts):
    import train as T
    ck = torch.load(ckpt_path, weights_only=False)
    net = T.load_net(ck)
    p = T.predict_at(net, store, np.zeros(len(ts), int), ts)
    return p, float(ck["alert_threshold"]), bool(ck["use_content"])


def trace_coreml(pkg, store, ts, uses_content):
    import coremltools as ct
    ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    X, C = store.gather(np.zeros(len(ts), int), ts)
    out = []
    for i in range(len(ts)):
        feed = {"features": X[i:i + 1]}
        if uses_content:
            feed["content"] = C[i:i + 1]
        out.append(float(np.asarray(ml.predict(feed)["lapse_probability"]).reshape(-1)[0]))
    meta = json.loads(pathlib.Path(str(pkg).replace(".mlpackage", ".json")).read_text())
    return np.array(out), float(meta["alert_threshold"]), uses_content


def summarize(ts, p, thr, onsets, lapses, horizon=15.0):
    in_lapse = np.zeros(len(ts), bool)
    for a, b, _ in lapses:
        in_lapse |= (ts >= a) & (ts < b)
    pre = np.zeros(len(ts), bool)
    for o in onsets:
        pre |= (ts >= o - horizon) & (ts < o)
    calm = ~in_lapse & ~pre & (ts >= 20)
    base = float(np.median(p[calm])) if calm.any() else float("nan")
    rows = []
    for o in onsets:
        def at(x):
            return float(p[np.argmin(np.abs(ts - x))])
        w = (ts >= o - horizon) & (ts < o)
        alert = w & (p >= thr)
        rows.append({"onset_s": float(o), "p_minus15": at(o - 15), "p_minus10": at(o - 10), "p_minus5": at(o - 5),
                     "p_minus1": at(o - 1), "max_before": float(p[w].max()) if w.any() else float("nan"),
                     "mean_last8": float(p[(ts >= o - 8) & (ts < o)].mean()),
                     "lead_s": float(o - ts[alert][0]) if alert.any() else None,
                     "rises_above_baseline": bool(p[(ts >= o - 8) & (ts < o)].mean() > base)})
    calm_alert_rate = float((p[calm] >= thr).mean()) if calm.any() else float("nan")
    return {"threshold": thr, "baseline_median_focused": base, "focused_alert_rate": calm_alert_rate,
            "alert_rate_all": float((p >= thr).mean()), "onsets": rows}


def plot(ts, traces, onsets, lapses, doc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"distracted": "#eb6834", "drowsy": "#7a5cd6", "stressed": "#d63a5c"}
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for v in doc.get("valleys", []):
        ax.axvspan(v["start"], v["end"], color="#9a9a95", alpha=0.12, lw=0)
    for a, b, s in lapses:
        ax.axvspan(a, b, color=colors.get(s, "#999"), alpha=0.18, lw=0)
        ax.text(a + 0.5, 0.97, s, fontsize=8, color=colors.get(s, "#333"), va="top")
    for o in onsets:
        ax.axvline(o, color="#333", lw=0.8, ls=":")
    style = {"with content": ("#2a78d6", "-"), "without content": ("#1baf7a", "-"),
             "previous with content": ("#2a78d6", "--"), "previous without content": ("#1baf7a", "--")}
    for name, (p, thr) in traces.items():
        c, ls = style.get(name, ("#555", "-"))
        ax.plot(ts, p, ls, color=c, lw=1.6 if ls == "-" else 1.0, label=f"{name} (alert >= {thr:.2f})")
        if ls == "-":
            ax.axhline(thr, color=c, lw=0.8, ls="-.", alpha=0.6)
    ax.set_xlim(0, ts[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("session / playhead time (s)"); ax.set_ylabel("P(lapse begins within 15 s)")
    ax.set_title("[SYNTHETIC] LapseForecaster risk on the bundled replay + MOCK sample_lecture content  |  "
                 "SYNTHETIC DATA, NOT A RESULT\ngrey = MOCK predicted valleys, coloured = scripted lapses, "
                 "dotted = lapse onsets, dash-dot = alert thresholds", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.88))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(HERE / "out/models"))
    ap.add_argument("--mlpackage-dir", help="trace Core ML packages from this dir instead of checkpoints")
    ap.add_argument("--previous-dir", help="also trace previous Core ML packages from this dir (dashed)")
    ap.add_argument("--reports", default=str(REPORTS))
    a = ap.parse_args(argv)
    sess, doc, onsets, lapses = replay_inputs()
    store = D.FlatStore([sess])
    ts = np.arange(1.0, np.floor(sess.t_end - 0.1) + 1)
    traces, summary = {}, {}
    if a.mlpackage_dir:
        for uc, name in [(True, "LapseForecaster"), (False, "LapseForecasterNoContent")]:
            p, thr, _ = trace_coreml(pathlib.Path(a.mlpackage_dir) / f"{name}.mlpackage", store, ts, uc)
            traces["with content" if uc else "without content"] = (p, thr)
    else:
        for cp in sorted(pathlib.Path(a.models).glob("forecaster_*.pt")):
            p, thr, uc = trace_torch(cp, store, ts)
            traces["with content" if uc else "without content"] = (p, thr)
    if a.previous_dir:
        for uc, name in [(True, "LapseForecaster"), (False, "LapseForecasterNoContent")]:
            p, thr, _ = trace_coreml(pathlib.Path(a.previous_dir) / f"{name}.mlpackage", store, ts, uc)
            traces[f"previous {'with' if uc else 'without'} content"] = (p, thr)
    for name, (p, thr) in traces.items():
        summary[name] = summarize(ts, p, thr, onsets, lapses)
    rep = pathlib.Path(a.reports); rep.mkdir(parents=True, exist_ok=True)
    plot(ts, traces, onsets, lapses, doc, rep / "forecast_replay_trace_SYNTHETIC.png")
    out = {"SYNTHETIC": True, "note": "SYNTHETIC replay + MOCK content. Shows the demo behaviour, not a result.",
           "replay": str(REPLAY.relative_to(ROOT)), "content": str(LECTURE.relative_to(ROOT)),
           "onsets_s": [float(o) for o in onsets], "lapses": lapses,
           "trace": {k: [round(float(x), 4) for x in v[0]] for k, v in traces.items()}, "t": ts.tolist(),
           "summary": summary}
    (rep / "forecast_replay_trace_SYNTHETIC.json").write_text(json.dumps(out, indent=1))
    lines = ["# SYNTHETIC LapseForecaster risk trace on the bundled replay", "",
             "**SYNTHETIC replay + MOCK content. Not a result.** `Resources/Replay/synthetic_session.replay.csv` "
             "fed second by second with MOCK `sample_lecture` content (playhead = session time), as in the "
             "Simulator demo. The replay's pre-lapse cues are scripted by `ml/synth/make_replay.py` from the "
             "same kind of assumptions the training generator makes, so this checks the demo, not real-world "
             "forecasting.", ""]
    for name, s in summary.items():
        lines += [f"## {name} (alert threshold {s['threshold']:.3f})", "",
                  f"Median risk while focused and not in the 15 s before a lapse: {s['baseline_median_focused']:.3f}; "
                  f"alerts on {s['focused_alert_rate']:.0%} of those seconds.", "",
                  "| onset (s) | p @ -15 s | p @ -10 s | p @ -5 s | p @ -1 s | mean last 8 s | max before | lead (s) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in s["onsets"]:
            lead = f"{r['lead_s']:.0f}" if r["lead_s"] is not None else "no alert"
            lines.append(f"| {r['onset_s']:.0f} | {r['p_minus15']:.3f} | {r['p_minus10']:.3f} | {r['p_minus5']:.3f} | "
                         f"{r['p_minus1']:.3f} | {r['mean_last8']:.3f} | {r['max_before']:.3f} | {lead} |")
        lines.append("")
    (rep / "forecast_replay_trace_SYNTHETIC.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
