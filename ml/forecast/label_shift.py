#!/usr/bin/env python3
"""Forecaster label construction ("label shifting") for FocusLoop's LapseForecaster.

The forecaster answers: P(a lapse begins within the next 15 s). Every rule below is
deterministic and unit-tested in ml/tests/test_forecast_labels.py.

Inputs (per session; see ml/forecast/README.md for the file formats)
  * session CSV  : 10 Hz FeatureFrame rows. Required: `t` + the 42 schema features.
                   Optional `label` (focused/distracted/drowsy/stressed) and/or `lapse` (0/1).
  * SART CSV     : one row per trial: `t` (stimulus onset, session seconds), `digit` (1-9),
                   `responded` (0/1), `rt` (seconds, empty if no response).
  * ES CSV       : experience-sampling answers: `t`, `focused` (0/1), `reason` (optional).

Positive EVENTS (each is a point in time, seconds since session start)
  P1  SART commission error: a trial with digit == 3 and responded == 1. Event at the trial onset.
  P2  SART RT-variability spike: over GO trials that have an RT, rolling coefficient of variation
      CV_i = std(rt[i-3..i]) / mean(rt[i-3..i]) (population std, 4-trial window ending at trial i).
      A spike is the RISING EDGE where CV_i > thr_user and CV_{i-1} <= thr_user (or i is the first
      full window). Event at the onset of trial i. thr_user = the user's 80th percentile of CV
      over ALL of that user's SART trials (all their SART sessions pooled).
  P3  Experience sampling "not focused" (focused == 0). Event at the answer time.
  P4  Session-CSV label onsets: each transition into a non-focused state, i.e. frame j where
      lapse_j == 1 and lapse_{j-1} == 0 (lapse := label != focused when `label` exists, else the
      `lapse` column). A non-focused first frame counts as an onset at that frame's time.

TARGET at a sample time t: y(t) = 1 iff any event e satisfies t < e <= t + 15.

SAMPLES: one per second, t in {20, 21, ...} with t <= T_end - 15 (a full 20 s input window and a
fully observed 15 s horizon). Samples where the session label says the user is ALREADY in a
non-focused episode at t are excluded (weight 0): the forecaster predicts onsets, and those
seconds are handled by the reactive (state classifier) path. Excluded samples are still scored at
inference time for the lead-time metric.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "ml/schema/feature_schema.json").read_text())
FEATS = [f["name"] for f in SCHEMA["features"]]
HZ = int(SCHEMA["rate_hz"])
WINDOW_S = int(SCHEMA["window_forecast_s"])   # 20
HORIZON_S = float(SCHEMA["forecast_horizon_s"])  # 15
CV_WINDOW = 4
CV_PERCENTILE = 80.0


# ---------------------------------------------------------------- loaders

def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def read_sart(path) -> pd.DataFrame:
    """Load a SART trial CSV in either format and normalise to columns t, digit, responded, rt (s).

    Accepts the app's export (`FocusLoop_sart_trials.csv`:
    trial,onset_s,digit,is_nogo,tapped,rt_ms,commission,omission) or the native t,digit,responded,rt.
    """
    df = read_csv(path)
    if "onset_s" in df.columns:
        rt = pd.to_numeric(df.get("rt_ms"), errors="coerce") / 1000.0
        df = pd.DataFrame({"t": df["onset_s"].astype(float), "digit": df["digit"].astype(int),
                           "responded": df["tapped"].astype(int), "rt": rt})
    return df


def load_session(path) -> pd.DataFrame:
    df = read_csv(path)
    missing = [c for c in ["t"] + FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return df


def lapse_mask(df: pd.DataFrame) -> np.ndarray | None:
    """Per-frame bool: user is in a non-focused episode. None if the CSV has no labels."""
    if "label" in df.columns:
        lab = df["label"].astype(str).str.strip().str.lower()
        return (lab != "focused").to_numpy() & lab.notna().to_numpy() & (lab != "nan").to_numpy()
    if "lapse" in df.columns:
        return df["lapse"].fillna(0).astype(int).to_numpy() == 1
    return None


# ---------------------------------------------------------------- event rules

def label_onsets(t: np.ndarray, lapse: np.ndarray) -> np.ndarray:
    """P4: times of transitions focused -> non-focused."""
    lapse = np.asarray(lapse, bool)
    if lapse.size == 0:
        return np.zeros(0)
    prev = np.concatenate([[False], lapse[:-1]])
    return np.asarray(t)[lapse & ~prev].astype(float)


def commission_events(sart: pd.DataFrame) -> np.ndarray:
    """P1: onsets of no-go (digit 3) trials that got a response."""
    m = (sart["digit"].astype(int) == 3) & (sart["responded"].astype(int) == 1)
    return sart.loc[m, "t"].to_numpy(float)


def go_rt_cv(sart: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rolling 4-trial CV of RT over go trials with a response. Returns (onset times, cv)."""
    go = sart[(sart["digit"].astype(int) != 3) & (sart["responded"].astype(int) == 1)]
    go = go[pd.to_numeric(go["rt"], errors="coerce").notna()]
    t = go["t"].to_numpy(float)
    rt = pd.to_numeric(go["rt"]).to_numpy(float)
    if len(rt) < CV_WINDOW:
        return np.zeros(0), np.zeros(0)
    cv = np.array([np.std(rt[i - CV_WINDOW + 1:i + 1]) / max(1e-6, np.mean(rt[i - CV_WINDOW + 1:i + 1]))
                   for i in range(CV_WINDOW - 1, len(rt))])
    return t[CV_WINDOW - 1:], cv


def user_cv_threshold(sarts: list[pd.DataFrame]) -> float | None:
    """The user's 80th percentile of rolling RT CV, pooled over all their SART sessions."""
    cvs = [go_rt_cv(s)[1] for s in sarts]
    cvs = np.concatenate(cvs) if cvs else np.zeros(0)
    return float(np.percentile(cvs, CV_PERCENTILE)) if cvs.size else None


def rt_spike_events(sart: pd.DataFrame, threshold: float) -> np.ndarray:
    """P2: rising edges of CV > threshold."""
    t, cv = go_rt_cv(sart)
    if cv.size == 0:
        return np.zeros(0)
    above = cv > threshold
    prev = np.concatenate([[False], above[:-1]])
    return t[above & ~prev]


def es_events(es: pd.DataFrame) -> np.ndarray:
    """P3: experience-sampling "not focused" answers."""
    return es.loc[es["focused"].astype(int) == 0, "t"].to_numpy(float)


def session_events(session: pd.DataFrame, sart: pd.DataFrame | None = None, es: pd.DataFrame | None = None,
                   cv_threshold: float | None = None) -> dict[str, np.ndarray]:
    """All positive events of one session, by rule."""
    ev = {"label_onset": np.zeros(0), "commission": np.zeros(0), "rt_spike": np.zeros(0), "es_not_focused": np.zeros(0)}
    m = lapse_mask(session)
    if m is not None:
        ev["label_onset"] = label_onsets(session["t"].to_numpy(float), m)
    if sart is not None and len(sart):
        ev["commission"] = commission_events(sart)
        if cv_threshold is not None:
            ev["rt_spike"] = rt_spike_events(sart, cv_threshold)
    if es is not None and len(es):
        ev["es_not_focused"] = es_events(es)
    return ev


def all_events(ev: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.concatenate([np.asarray(v, float) for v in ev.values()]) if ev else np.zeros(0)
    return np.unique(arr)


# ---------------------------------------------------------------- targets

def shift_targets(sample_t: np.ndarray, events: np.ndarray, horizon: float = HORIZON_S) -> np.ndarray:
    """y(t) = 1 iff any event e with t < e <= t + horizon."""
    sample_t = np.asarray(sample_t, float)
    ev = np.sort(np.asarray(events, float))
    if ev.size == 0:
        return np.zeros(len(sample_t), np.int64)
    # first event strictly after t, check it is <= t + horizon
    k = np.searchsorted(ev, sample_t, side="right")
    nxt = np.where(k < ev.size, ev[np.minimum(k, ev.size - 1)], np.inf)
    return (nxt <= sample_t + horizon + 1e-9).astype(np.int64)


def sample_times(t_end: float, window_s: float = WINDOW_S, horizon: float = HORIZON_S, step: float = 1.0,
                 full_horizon: bool = True) -> np.ndarray:
    last = t_end - horizon if full_horizon else t_end
    if last < window_s:
        return np.zeros(0)
    return np.arange(window_s, last + 1e-9, step)


def in_lapse_at(session: pd.DataFrame, ts: np.ndarray) -> np.ndarray:
    m = lapse_mask(session)
    if m is None:
        return np.zeros(len(ts), bool)
    idx = np.clip(np.round(np.asarray(ts) * HZ).astype(int), 0, len(m) - 1)
    return m[idx]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print the forecaster events/targets for one session.")
    ap.add_argument("session_csv"); ap.add_argument("--sart"); ap.add_argument("--es")
    a = ap.parse_args()
    s = load_session(a.session_csv)
    sart = read_sart(a.sart) if a.sart else None
    es = read_csv(a.es) if a.es else None
    thr = user_cv_threshold([sart]) if sart is not None else None
    ev = session_events(s, sart, es, thr)
    ts = sample_times(float(s["t"].iloc[-1]))
    y = shift_targets(ts, all_events(ev))
    for k, v in ev.items():
        print(f"{k:15s} {len(v):4d} events")
    print(f"samples={len(ts)} positive_rate={y.mean() if len(y) else 0:.3f}")
