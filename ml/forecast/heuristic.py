"""Python mirror of Modules/Forecasting/HeuristicLapseForecaster.swift (default weights), vectorised over
windows, so the no-model fallback can be scored with the same metrics as the networks."""
from __future__ import annotations

import numpy as np

import label_shift as L

IDX = {k: i for i, k in enumerate(L.FEATS)}
W = dict(bias=-2.2, blinkRateTrend=0.35, gazeDispersion=0.9, offZoneDwell=0.25, perclosSlope=0.5,
         heartRateDelta=0.3, driftRisk=0.45, valleyProximity=1.0)


def signals(X: np.ndarray, C: np.ndarray | None) -> dict:
    """X: (B,200,42) raw frames; C: (B,20,11) raw content or None."""
    head, tail, last10 = X[:, :50], X[:, -50:], X[:, -100:]
    s = {}
    s["blinkRateTrend"] = tail[..., IDX["blinkRate"]].mean(1) - head[..., IDX["blinkRate"]].mean(1)
    s["perclosSlope"] = tail[..., IDX["perclos"]].mean(1) - head[..., IDX["perclos"]].mean(1)
    tracked = last10[..., IDX["faceDetected"]] > 0.5
    def var(v):
        n = tracked.sum(1)
        m = np.where(n > 0, (v * tracked).sum(1) / np.maximum(n, 1), 0)
        return np.where(n > 1, (((v - m[:, None]) ** 2) * tracked).sum(1) / np.maximum(n, 1), 0)
    s["gazeDispersion"] = np.sqrt(var(last10[..., IDX["gazeYaw"]]) + var(last10[..., IDX["gazePitch"]]))
    s["offZoneDwell"] = X[:, -1, IDX["offZoneDwell"]]
    good = (tail[..., IDX["heartRateQuality"]] > 0.3) & (tail[..., IDX["heartRate"]] > 0)
    n = good.sum(1)
    s["heartRateDelta"] = np.where(n > 0, (tail[..., IDX["heartRateDelta"]] * good).sum(1) / np.maximum(n, 1), 0)
    if C is not None:
        last = C[:, -1]
        s["driftRisk"] = last[:, 8]
        stv = np.clip(last[:, 9] * 60, 0, 60)
        s["valleyProximity"] = np.where(last[:, 10] > 0.5, 1.0, np.maximum(0, 1 - stv / 15))
    return s


def probability(X, C=None, w=W) -> np.ndarray:
    s = signals(X, C)
    z = np.full(len(X), w["bias"], float)
    z += w["blinkRateTrend"] * np.maximum(0, s["blinkRateTrend"]) / 5
    z += w["gazeDispersion"] * s["gazeDispersion"] / 0.1
    z += w["offZoneDwell"] * np.clip(s["offZoneDwell"], 0, 10)
    z += w["perclosSlope"] * np.maximum(0, s["perclosSlope"]) / 0.05
    z += w["heartRateDelta"] * np.maximum(0, s["heartRateDelta"]) / 5
    if C is not None:
        z += w["driftRisk"] * s["driftRisk"]
        z += w["valleyProximity"] * np.clip(s["valleyProximity"], 0, 1)
    return 1 / (1 + np.exp(-z))
