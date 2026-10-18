"""Cross-check of the Swift rPPG pipeline against the independent numpy reference. SYNTHETIC data only.

Run:  ml/.venv/bin/python -m pytest ml/rppg_reference -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pos_reference import PARAMS, SplitMix64, estimate_last_window, generate, pos, resample_uniform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SWIFT_SYNTH = REPO / "Tests" / "RPPGTests" / "RPPGSyntheticSignal.swift"
SWIFT_RESULTS = HERE / "swift_results_SYNTHETIC.json"


def test_params_match_swift():
    """The Swift generator's `static let` literals must equal synthetic_params.json."""
    src = SWIFT_SYNTH.read_text()
    for key, value in PARAMS.items():
        if key in ("label", "description"):
            continue
        m = re.search(rf"static let {key}(?::[^=]+)? = (.+)", src)
        assert m, f"{key} missing from {SWIFT_SYNTH.name}"
        swift_val = json.loads(m.group(1).strip())
        assert swift_val == value, f"{key}: swift {swift_val} != json {value}"


def test_splitmix_known_vector():
    # splitmix64(seed=0) first output, from the reference C implementation.
    assert SplitMix64(0).next() == 0xE220A8397B1DCDAF


@pytest.mark.parametrize("bpm", PARAMS["bpms"])
def test_reference_recovers_bpm(bpm):
    est = estimate_last_window(*generate(bpm))
    assert abs(est.bpm - bpm) <= 3.0
    assert est.quality > 0.7


def test_reference_noise_is_low_quality():
    q = [estimate_last_window(*generate(72, pulse_amplitude=0.0, seed=s)).quality for s in range(1, 6)]
    assert np.mean(q) < 0.3


def test_pos_cancels_common_illumination():
    """A pure multiplicative intensity change (same for R, G, B) must vanish after POS."""
    t = np.arange(0, 20, 1 / 30)
    illum = 1 + 0.2 * np.sin(2 * np.pi * 1.2 * t)
    rgb = np.outer(illum, [0.42, 0.28, 0.22])
    h = pos(resample_uniform(t, rgb)[1])
    assert np.max(np.abs(h)) < 1e-9


def test_agrees_with_swift_results():
    """Swift's final estimates (copied from the XCTest log, see README in the JSON) vs this reference."""
    if not SWIFT_RESULTS.exists():
        pytest.skip("run the Swift tests and export swift_results_SYNTHETIC.json first")
    data = json.loads(SWIFT_RESULTS.read_text())
    assert data["label"] == "SYNTHETIC"
    for row in data["results"]:
        ref = estimate_last_window(*generate(row["bpm"]))
        assert abs(ref.bpm - row["est"]) <= 1.5, (row, ref)
        assert abs(ref.snr_db - row["snr_db"]) <= 1.5, (row, ref)
