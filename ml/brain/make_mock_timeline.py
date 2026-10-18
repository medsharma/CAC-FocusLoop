#!/usr/bin/env python3
"""Write a clearly-labelled MOCK <lecture>.brain.json so the app and preview run before the
real Axonal Labs model is plugged into precompute_content.py. Values are synthetic."""
import json, sys
import numpy as np

def main(out="Resources/Brain/sample_lecture.brain.json", seconds=180, seed=7):
    rng = np.random.default_rng(seed)
    keys = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    t = np.arange(seconds)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); k /= k.sum()
    nets = {}
    for i, key in enumerate(keys):
        x = np.convolve(rng.standard_normal(seconds + 24), k, mode="same")[12:-12]
        x += 0.8 * np.sin(2 * np.pi * t / (40 + 9 * i) + i)
        nets[key] = ((x - x.mean()) / x.std()).round(3).tolist()
    att = (np.array(nets["DorsAttn"]) + np.array(nets["SalVentAttn"]) + np.array(nets["Vis"])) / 3
    eng = (1 / (1 + np.exp(-att * 1.5))).round(3)
    drift = (np.array(nets["Default"]) - np.array(nets["DorsAttn"])).round(3)
    valleys, inside = [], False
    for s, e in enumerate(eng):
        if e < 0.35 and not inside: start, inside = s, True
        if (e >= 0.35 or s == seconds - 1) and inside:
            inside = False
            if s - start >= 6: valleys.append({"start": start, "end": s})
    doc = {"lecture_id": "sample_lecture", "hz": 1, "mock": True,
           "note": "MOCK synthetic data - not produced by the brain model",
           "networks": nets, "predicted_engagement": eng.tolist(),
           "predicted_drift_risk": drift.tolist(), "valleys": valleys}
    json.dump(doc, open(out, "w"))
    print(f"wrote {out}: {seconds}s, {len(valleys)} valleys (MOCK)")

if __name__ == "__main__":
    main(*sys.argv[1:2])
