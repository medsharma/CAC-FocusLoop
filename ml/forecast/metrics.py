"""Forecaster metrics: AUROC, precision at a fixed alert rate, and lead time. Unit-tested in
ml/tests/test_forecast_labels.py.

Lead time (exact rule): for each true lapse onset o, look at the 1 Hz predictions at times
t in [o - horizon, o). An "alert" is p(t) >= threshold. The first alert in that window counts;
lead = o - t_first (so an alert already active at o - 15 gives lead 15 s). Onsets with no alert in
the window are misses (no lead time). Report the median lead over detected onsets and the
detection rate = detected / onsets. Onsets whose look-back window starts before the first
available prediction are skipped (not enough history to score them).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, p) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def alert_threshold(p, rate: float) -> float:
    """Threshold that alerts on `rate` of samples (the (1-rate) quantile)."""
    return float(np.quantile(np.asarray(p, float), 1 - rate))


def precision_at_alert_rate(y, p, rate: float = 0.10, threshold: float | None = None):
    """Returns (precision, recall, threshold, realised alert rate)."""
    y = np.asarray(y); p = np.asarray(p, float)
    thr = alert_threshold(p, rate) if threshold is None else threshold
    a = p >= thr
    prec = float(y[a].mean()) if a.any() else float("nan")
    rec = float(a[y == 1].mean()) if (y == 1).any() else float("nan")
    return prec, rec, thr, float(a.mean())


def lead_times(pred_t, pred_p, onsets, threshold: float, horizon: float = 15.0) -> np.ndarray:
    """Lead time (s) per scorable onset; NaN where the onset was missed."""
    pred_t = np.asarray(pred_t, float); pred_p = np.asarray(pred_p, float)
    order = np.argsort(pred_t); pred_t, pred_p = pred_t[order], pred_p[order]
    out = []
    if pred_t.size == 0:
        return np.zeros(0)
    for o in np.asarray(onsets, float):
        if o - horizon < pred_t[0] - 1e-9:
            continue
        m = (pred_t >= o - horizon - 1e-9) & (pred_t < o - 1e-9) & (pred_p >= threshold)
        out.append(o - pred_t[m][0] if m.any() else np.nan)
    return np.array(out, float)


def lead_summary(leads: np.ndarray) -> dict:
    det = leads[~np.isnan(leads)]
    return {"onsets": int(leads.size), "detected": int(det.size),
            "detection_rate": float(det.size / leads.size) if leads.size else float("nan"),
            "median_lead_s": float(np.median(det)) if det.size else float("nan")}
