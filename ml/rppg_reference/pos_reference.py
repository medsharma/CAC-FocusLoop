"""Independent numpy/scipy reference of the FocusLoop rPPG pipeline, used only to cross-check the
Swift implementation (Modules/RPPG/POSHeartRateEstimator.swift). Runs on the Mac, never in the app.

POS is reimplemented from the paper (Wang, den Brinker, Stuijk & de Haan 2017, "Algorithmic
Principles of Remote PPG", IEEE TBME 64(7), Algorithm 1); SNR follows de Haan & Jeanne 2013.
Deliberately written differently from the Swift code (batch POS with vectorised numpy, scipy
Butterworth SOS + sosfiltfilt, numpy rfft) so agreement is meaningful.

All signals produced by `generate` are SYNTHETIC.

Usage:  ml/.venv/bin/python ml/rppg_reference/pos_reference.py      # prints a SYNTHETIC BPM table
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "synthetic_params.json").read_text())
MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------- synthetic signal
class SplitMix64:
    """Bit-identical to RPPGSynth.SplitMix64 in the Swift tests."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return (self.next() >> 11) * 2.0**-53

    def normal(self) -> float:
        u1 = 1.0 - self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate(bpm: float, duration: float | None = None, pulse_amplitude: float | None = None,
             noise_sigma: float | None = None, seed: int | None = None, p: dict = PARAMS):
    """SYNTHETIC skin-colour trace. Returns (t [N], rgb [N, 3]) exactly as the Swift generator."""
    duration = p["duration"] if duration is None else duration
    amp = p["pulseRelAmplitude"] if pulse_amplitude is None else pulse_amplitude
    sigma = p["noiseSigma"] if noise_sigma is None else noise_sigma
    rng = SplitMix64(p["seedBase"] + int(bpm) if seed is None else seed)
    f = bpm / 60.0
    n = int(duration * p["fsNominal"])
    base, w = p["baseRGB"], p["pulseChannelWeights"]
    h, phi = p["harmonicRatio"], p["harmonicPhase"]
    t = np.empty(n)
    rgb = np.empty((n, 3))
    for k in range(n):
        tk = k / p["fsNominal"] + p["jitter"] * (2 * rng.uniform() - 1)
        pulse = (math.sin(2 * math.pi * f * tk) + h * math.sin(4 * math.pi * f * tk + phi)) / (1 + h)
        illum = 1 + p["driftAmplitude"] * math.sin(2 * math.pi * p["driftFrequency"] * tk) + p["driftSlope"] * tk
        t[k] = tk
        for i in range(3):
            rgb[k, i] = illum * base[i] * (1 + amp * w[i] * pulse) + sigma * rng.normal()
    rgb = rgb.astype(np.float32).astype(np.float64)  # Swift feeds Float32 samples
    return t, rgb


# ----------------------------------------------------------------------------- pipeline
@dataclass
class Estimate:
    bpm: float
    snr_db: float
    quality: float


FS = 30.0
POS_WINDOW_S = 1.6
ANALYSIS_S = 10.0
BAND = (0.7, 3.0)
SNR_HALF_WIDTH = 0.1
NFFT = 2048
SNR_Q0, SNR_Q1 = -2.0, 4.0


def resample_uniform(t: np.ndarray, rgb: np.ndarray, fs: float = FS):
    grid = np.arange(t[0], t[-1] + 1e-12, 1.0 / fs)
    return grid, np.stack([np.interp(grid, t, rgb[:, i]) for i in range(3)], axis=1)


def pos(rgb: np.ndarray, fs: float = FS, window_s: float = POS_WINDOW_S) -> np.ndarray:
    """Plane-Orthogonal-to-Skin, overlap-added over sliding windows of length l."""
    n = rgb.shape[0]
    l = int(math.ceil(window_s * fs))
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(n)
    for start in range(0, n - l + 1):
        c = rgb[start:start + l]
        cn = c / c.mean(axis=0)                 # temporal normalisation
        s = cn @ proj.T                         # (l, 2) projection onto the plane orthogonal to skin
        s1, s2 = s[:, 0], s[:, 1]
        sd2 = s2.std()
        hwin = s1 + (s1.std() / sd2 if sd2 > 0 else 0.0) * s2
        out[start:start + l] += hwin - hwin.mean()
    return out


def spectral_estimate(h: np.ndarray, fs: float = FS) -> tuple[Estimate, np.ndarray]:
    x = signal.detrend(h, type="linear")
    sos = signal.butter(2, BAND[0], btype="highpass", fs=fs, output="sos")
    sos = np.vstack([sos, signal.butter(2, BAND[1], btype="lowpass", fs=fs, output="sos")])
    x = signal.sosfiltfilt(sos, x, padtype="odd", padlen=min(int(1.5 * fs), len(x) - 1))
    win = np.hanning(len(x) + 2)[1:-1]          # periodic-free Hann, endpoints excluded
    spec = np.abs(np.fft.rfft(x * win, n=NFFT)) ** 2
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    band = np.where((freqs >= BAND[0]) & (freqs <= BAND[1]))[0]
    k0 = band[np.argmax(spec[band])]
    offset = 0.0
    if band[0] < k0 < band[-1]:
        a, b, c = np.log(spec[k0 - 1:k0 + 2])
        d = a - 2 * b + c
        offset = float(np.clip(0.5 * (a - c) / d, -0.5, 0.5)) if d != 0 else 0.0
    f0 = (k0 + offset) * fs / NFFT
    fb = freqs[band]
    sig_mask = (np.abs(fb - f0) <= SNR_HALF_WIDTH) | (np.abs(fb - 2 * f0) <= SNR_HALF_WIDTH)
    sig, noise = spec[band][sig_mask].sum(), spec[band][~sig_mask].sum()
    snr = 10 * math.log10(sig / noise)
    q = float(np.clip((snr - SNR_Q0) / (SNR_Q1 - SNR_Q0), 0, 1))
    return Estimate(bpm=f0 * 60, snr_db=snr, quality=q), x


def estimate_last_window(t: np.ndarray, rgb: np.ndarray) -> Estimate:
    """BPM over the final 10 s of the trace (what the Swift estimator reports at the end)."""
    _, u = resample_uniform(t, rgb)
    h = pos(u)
    n = int(round(ANALYSIS_S * FS))
    return spectral_estimate(h[-n:])[0]


def main() -> None:
    print("SYNTHETIC - POS reference on generated signals (not real data)")
    print(f"{'true_bpm':>8} {'est_bpm':>8} {'err':>6} {'snr_db':>7} {'quality':>7}")
    for bpm in PARAMS["bpms"]:
        e = estimate_last_window(*generate(bpm))
        print(f"{bpm:8.0f} {e.bpm:8.2f} {e.bpm - bpm:6.2f} {e.snr_db:7.2f} {e.quality:7.3f}")
    for seed in range(1, 6):
        e = estimate_last_window(*generate(72, pulse_amplitude=0.0, seed=seed))
        print(f"noise seed={seed}: est={e.bpm:.1f} snr_db={e.snr_db:.2f} quality={e.quality:.3f}")


if __name__ == "__main__":
    main()
