"""Python port of Modules/Inference/HeuristicAttentionClassifier.swift (default Config).

Used to compare the trained model against the rule-based classifier with the same metrics, on the
same windows. Mirrors `analyze(window:)` + `probabilities(for:)` line by line (float64 here, Float32 in
Swift; differences are at the 1e-6 level). Window = (100, 42) raw FeatureFrame rows, oldest first.

Also mirrors the face-missing rule BlendedAttentionClassifier applies before the model runs
(`faceMissing`), so `blended_override()` tells which windows the app never sends to the model.
"""
from __future__ import annotations

import numpy as np

from data import FEATS

I = {k: i for i, k in enumerate(FEATS)}


class Cfg:
    recentFrames = 30
    gazeFrames = 20
    browFrames = 10
    longBlinkLookback = 50
    minFaceFraction = 0.5
    missingFaceDistractedEvidence = 0.75
    dwellFullSeconds = 3.0
    closureOnset = 0.1
    closureSpan = 0.25
    longBlinkClosure = 0.8
    longBlinkFrames = 6
    longBlinkWeight = 0.25
    yawnJawOpen = 0.6
    yawnWeight = 0.5
    nodDrop = 0.2
    nodRecovery = 0.15
    nodWeight = 0.5
    browOnset = 0.15
    browSpan = 0.3
    squintOnset = 0.15
    squintSpan = 0.2
    squintWeight = 0.7
    heartRateQualityMin = 0.4
    minHeartRateFrames = 5
    heartRateDeltaOnset = 6.0
    heartRateDeltaSpan = 8.0
    stressBrowWeight = 0.55
    stressHeartRateWeight = 0.55
    logitGain = 4.0
    logitOffset = 1.0
    focusedLogit = 1.0
    maxEvidence = 1.5


def _clamp01(x):
    return min(max(x, 0.0), 1.0)


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def _median(x):
    return float(np.median(x)) if len(x) else 0.0


def _run_count(xs, thr, min_len):
    runs = run = 0
    for x in xs:
        if x > thr:
            run += 1
        else:
            if run >= min_len:
                runs += 1
            run = 0
    if run >= min_len:
        runs += 1
    return runs


def evidence(w: np.ndarray, c=Cfg) -> dict:
    """Evidence scores for one (T, 42) window (oldest first)."""
    face = w[:, I["faceDetected"]] > 0.5
    recent = w[-c.recentFrames:]
    gaze_span = w[-c.gazeFrames:]
    recent_face = _mean(recent[:, I["faceDetected"]] > 0.5)
    face_missing = recent_face < c.minFaceFraction

    gface = gaze_span[gaze_span[:, I["faceDetected"]] > 0.5]
    wf = w[face]
    distracted = 0.0
    if len(gface) >= 5:
        off_recent = 1 - _mean(gface[:, I["inWorkZone"]])
        dwell = float(gface[-1, I["offZoneDwell"]])
        distracted = max(off_recent, _clamp01(dwell / c.dwellFullSeconds))
    pickup = _mean(gaze_span[:, I["phonePickup"]] > 0.5)
    distracted = max(distracted, pickup)
    if face_missing:
        distracted = max(distracted, c.missingFaceDistractedEvidence)

    drowsy = stressed = 0.0
    if len(wf) >= 10:
        closure = np.minimum(wf[:, I["eyeBlinkLeft"]], wf[:, I["eyeBlinkRight"]])
        closure_median = _median(closure[-c.recentFrames:])
        long_blinks = _run_count(closure[-c.longBlinkLookback:], c.longBlinkClosure, c.longBlinkFrames)
        rf = wf[-c.recentFrames:]
        yawn = 1.0 if rf[:, I["jawOpen"]].max() > c.yawnJawOpen else 0.0
        pitch = wf[:, I["headPitch"]]
        pmin = pitch[-c.recentFrames:].min()
        pmed = _median(pitch)
        nod = 1.0 if (pmed - pmin > c.nodDrop and pitch[-1] > pmin + c.nodRecovery) else 0.0
        e_cl = _clamp01((closure_median - c.closureOnset) / c.closureSpan)
        drowsy = min(c.maxEvidence, e_cl + c.longBlinkWeight * long_blinks + c.yawnWeight * yawn + c.nodWeight * nod)
        bs = wf[-c.browFrames:]
        brow = _mean(bs[:, I["browTension"]])
        squint = _mean((bs[:, I["eyeSquintLeft"]] + bs[:, I["eyeSquintRight"]]) / 2)
        e_brow = max(_clamp01((brow - c.browOnset) / c.browSpan),
                     c.squintWeight * _clamp01((squint - c.squintOnset) / c.squintSpan))
        stressed = c.stressBrowWeight * e_brow
    good = gaze_span[(gaze_span[:, I["heartRate"]] > 0) & (gaze_span[:, I["heartRateQuality"]] >= c.heartRateQualityMin)]
    if len(good) >= c.minHeartRateFrames:
        hr_delta = _mean(good[:, I["heartRateDelta"]])
        if len(wf) >= 10:
            stressed += c.stressHeartRateWeight * _clamp01((hr_delta - c.heartRateDeltaOnset) / c.heartRateDeltaSpan)
    stressed = min(stressed, c.maxEvidence)
    return {"distracted": distracted, "drowsy": drowsy, "stressed": stressed, "face_missing": face_missing}


def probabilities(e: dict, c=Cfg) -> np.ndarray:
    def logit(x):
        return c.logitGain * min(max(x, 0.0), c.maxEvidence) - c.logitOffset
    z = np.array([c.focusedLogit, logit(e["distracted"]), logit(e["drowsy"]), logit(e["stressed"])])
    z = np.exp(z - z.max())
    return z / z.sum()


def predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X (N,T,42) -> (probabilities (N,4), face_missing (N,) bool)."""
    P, fm = np.zeros((len(X), 4)), np.zeros(len(X), bool)
    for i, w in enumerate(X):
        e = evidence(w)
        P[i] = probabilities(e)
        fm[i] = e["face_missing"]
    return P, fm


def blended(model_probs: np.ndarray, X: np.ndarray, confidence_threshold: float = 0.45) -> np.ndarray:
    """What BlendedAttentionClassifier returns without a personal head: heuristic when the face is
    missing, else the model, averaged with the heuristic when the model's max prob < 0.45."""
    H, fm = predict(X)
    out = model_probs.copy()
    low = model_probs.max(1) < confidence_threshold
    out[low] = (model_probs[low] + H[low]) / 2
    out[fm] = H[fm]
    return out
