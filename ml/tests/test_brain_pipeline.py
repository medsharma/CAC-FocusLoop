"""Tests for the offline brain-content pipeline (ml/brain/*) and its bundled outputs (Resources/Lectures/*).

Run: ml/.venv/bin/python -m pytest ml/tests/test_brain_pipeline.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml" / "brain"))

import brain_common as bc  # noqa: E402
from precompute_content import build_doc, engagement_from, find_valleys, vertices_to_f16  # noqa: E402
from prepare_stimulus import parse_subtitles  # noqa: E402
from run_brain_model import canonical_hrf, mock_network_drive  # noqa: E402
from to_networks import lag_resample, network_means, to_fsaverage5, to_networks  # noqa: E402

SCHEMA = json.loads((REPO / "ml" / "schema" / "brain_schema.json").read_text())
LECTURES = REPO / "Resources" / "Lectures"
BRAIN_FILES = sorted(LECTURES.glob("*.brain.json")) + [REPO / "Resources" / "Brain" / "sample_lecture.brain.json"]
NUDGE_ARMS = ["ambientGlow", "haptic", "chime", "putMeDown", "recapCard", "stretchBreak", "lightPulse",
              "pacedBreathing", "preemptiveGlow", "recallQuestion", "noNudge"]


@pytest.fixture(scope="module")
def labels() -> np.ndarray:
    return bc.load_network_labels("annot")


# ------------------------------------------------------------------ bundled brain.json files

def test_there_are_lecture_timelines():
    assert len(BRAIN_FILES) >= 3


@pytest.mark.parametrize("path", BRAIN_FILES, ids=lambda p: p.name)
def test_brain_json_matches_schema(path: Path):
    import jsonschema

    doc = json.loads(path.read_text())
    jsonschema.validate(doc, SCHEMA)
    assert list(doc["networks"].keys()) == bc.NETWORKS, "network order must be BrainNetwork.allCases"
    n = len(doc["predicted_engagement"])
    assert all(len(v) == n for v in doc["networks"].values())
    assert len(doc["predicted_drift_risk"]) == n
    assert doc["hz"] == 1
    for k, v in doc["networks"].items():  # z-scored within the lecture
        assert abs(np.mean(v)) < 0.05 and abs(np.std(v) - 1) < 0.05, k


@pytest.mark.parametrize("path", BRAIN_FILES, ids=lambda p: p.name)
def test_valleys_are_at_least_6s_and_ordered(path: Path):
    doc = json.loads(path.read_text())
    n = len(doc["predicted_engagement"])
    prev_end = -1.0
    for v in doc.get("valleys", []):
        assert v["end"] - v["start"] >= 6
        assert 0 <= v["start"] < v["end"] <= n
        assert v["start"] >= prev_end
        prev_end = v["end"]


@pytest.mark.parametrize("path", sorted(LECTURES.glob("*.brain.json")), ids=lambda p: p.name)
def test_lecture_timelines_are_mock_and_consistent(path: Path):
    doc = json.loads(path.read_text())
    assert doc["mock"] is True and "MOCK" in doc["note"], "MOCK model output must be labelled"
    z = np.array([doc["networks"][k] for k in bc.NETWORKS]).T
    np.testing.assert_allclose(doc["predicted_engagement"], np.round(engagement_from(z), 3), atol=2e-3)
    np.testing.assert_allclose(doc["predicted_drift_risk"],
                               np.array(doc["networks"]["Default"]) - np.array(doc["networks"]["DorsAttn"]),
                               atol=2e-3)


# ------------------------------------------------------------------ catalog, priors, vertices

def test_lectures_catalog():
    cat = json.loads((LECTURES / "lectures.json").read_text())
    ids = [x["id"] for x in cat]
    assert "sample_lecture" in ids and len(set(ids)) == len(ids)
    keys = {"id", "title", "subject", "source", "sourceURL", "license", "attribution", "videoFile",
            "durationSeconds", "recaps", "recallQuestions"}
    for lec in cat:
        assert keys <= set(lec), lec["id"]
        if lec["id"] == "sample_lecture":
            assert lec["videoFile"] is None and lec["durationSeconds"] == 180
            assert (REPO / "Resources" / "Brain" / "sample_lecture.brain.json").exists()
        else:
            assert (LECTURES / lec["videoFile"]).exists(), lec["videoFile"]
            assert (LECTURES / lec["videoFile"]).stat().st_size <= 40 * 1000 * 1000
            doc = json.loads((LECTURES / f"{lec['id']}.brain.json").read_text())
            assert len(doc["predicted_engagement"]) == math.floor(lec["durationSeconds"])
            assert 3 <= len(lec["recaps"]) <= 6
            assert lec["sourceURL"].startswith("https://") and lec["attribution"]
        assert 2 <= len(lec["recallQuestions"]) <= 4
        for r in lec["recaps"]:
            assert 0 <= r["start"] < r["end"] <= lec["durationSeconds"] + 1e-6 and r["text"]
        for q in lec["recallQuestions"]:
            assert 0 <= q["correctIndex"] < len(q["choices"]) and q["after"] <= lec["durationSeconds"]


def test_bundle_size_budget():
    total = sum(p.stat().st_size for p in LECTURES.iterdir() if p.is_file())
    assert total <= 120 * 1000 * 1000


def test_nudge_priors():
    doc = json.loads((LECTURES / "nudge_priors.json").read_text())
    assert doc["mock"] is True
    arms = {p["arm"]: p for p in doc["priors"]}
    assert set(arms) == set(NUDGE_ARMS)
    for p in doc["priors"]:
        assert p["alpha"] > 0 and p["beta"] > 0 and p["alpha"] + p["beta"] <= 6
    assert (arms["noNudge"]["alpha"], arms["noNudge"]["beta"]) == (1, 1)
    assert arms["lightPulse"]["alpha"] == arms["lightPulse"]["beta"], "experimental arm stays neutral"


@pytest.mark.parametrize("path", sorted(LECTURES.glob("*.vertices.f16")), ids=lambda p: p.name)
def test_vertices_f16(path: Path):
    lid = path.name.removesuffix(".vertices.f16")
    n = len(json.loads((LECTURES / f"{lid}.brain.json").read_text())["predicted_engagement"])
    a = np.fromfile(path, "<f2")
    assert a.size == n * bc.N_FS5 and path.stat().st_size <= 8_000_000
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1


# ------------------------------------------------------------------ atlas

def test_annot_labels_match_brainmesh():
    a, m = bc.load_annot_networks(), bc.load_mesh_networks()
    assert a.shape == m.shape == (bc.N_FS5,)
    np.testing.assert_array_equal(a, m)
    assert set(np.unique(a)) == set(range(8))


# ------------------------------------------------------------------ to_networks on synthetic input

def _synthetic(labels: np.ndarray, T: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Vertex data whose network-k vertices equal known[t, k]; medial wall is junk."""
    rng = np.random.default_rng(0)
    known = rng.standard_normal((T, 7)).astype(np.float32) + np.arange(7)
    data = np.full((T, bc.N_FS5), 1e3, np.float32)
    for k in range(7):
        data[:, labels == k + 1] = known[:, [k]]
    return data, known


def test_network_means_recover_known_values(labels):
    data, known = _synthetic(labels)
    np.testing.assert_allclose(network_means(data, labels), known, atol=1e-5)


def test_to_fsaverage5_handles_orientation_and_full_fsaverage(labels):
    data, known = _synthetic(labels, 8)
    np.testing.assert_array_equal(to_fsaverage5({"space": "fsaverage5", "data": data.T, "tr": 1}), data)
    full = np.full((8, bc.N_FS7), np.nan, np.float32)               # junk outside the nested fsaverage5 part
    full[:, :bc.N_FS5_HEMI] = data[:, :bc.N_FS5_HEMI]
    full[:, bc.N_FS7_HEMI:bc.N_FS7_HEMI + bc.N_FS5_HEMI] = data[:, bc.N_FS5_HEMI:]
    fs5 = to_fsaverage5({"space": "fsaverage", "data": full, "tr": 1})
    np.testing.assert_array_equal(fs5, data)
    np.testing.assert_allclose(network_means(fs5, labels), known, atol=1e-5)


def test_lag_shift_and_resample():
    tr, lag = 1.5, 5.0
    x = np.zeros((40, 1), np.float32)
    x[10] = 1.0                                                   # response at 15 s -> stimulus at 10 s
    y = lag_resample(x, tr, lag, hz=1.0, n_out=50)[:, 0]
    assert int(np.argmax(y)) == 10 and y[10] == pytest.approx(1.0)
    ramp = (np.arange(40) * tr)[:, None].astype(np.float32)      # value == response time
    np.testing.assert_allclose(lag_resample(ramp, tr, lag, n_out=30)[:, 0], np.arange(30) + lag, atol=1e-4)


def test_to_networks_end_to_end_zscore(labels):
    data, known = _synthetic(labels, 60)
    res = to_networks({"space": "fsaverage5", "data": data, "tr": 1.0}, lag=5, duration=50)
    assert res["networks"].shape == (50, 7) and res["keys"] == bc.NETWORKS
    expect = known[5:55]
    expect = (expect - expect.mean(0)) / expect.std(0)
    np.testing.assert_allclose(res["networks"], expect, atol=1e-4)
    raw = to_networks({"space": "fsaverage5", "data": data, "tr": 1.0}, lag=5, duration=50, z=False)
    np.testing.assert_allclose(raw["networks"], known[5:55], atol=1e-4)


def test_mni_volume_projects_onto_fsaverage5(labels):
    import nibabel as nib

    affine = np.array([[4, 0, 0, -90], [0, 4, 0, -126], [0, 0, 4, -72], [0, 0, 0, 1]], float)
    vol = np.ones((46, 55, 46, 3), np.float32) * np.array([1.0, 2.0, 3.0], np.float32)
    img = nib.Nifti1Image(vol, affine)
    fs5 = to_fsaverage5({"space": "mni_volume", "img": img, "tr": 2.0})
    assert fs5.shape == (3, bc.N_FS5)
    nets = network_means(fs5, labels)
    np.testing.assert_allclose(nets, np.array([[1.0], [2.0], [3.0]]).repeat(7, 1), atol=1e-4)


# ------------------------------------------------------------------ content summary + helpers

def test_find_valleys_min_length_and_merge():
    e = np.full(40, 0.8)
    e[2:6] = 0.1                    # 4 s: too short
    e[10:14] = 0.1
    e[15:19] = 0.1                  # 4 + 1 gap + 4 -> merged 9 s
    e[30:40] = 0.2                  # runs to the end
    assert find_valleys(e) == [{"start": 10.0, "end": 19.0}, {"start": 30.0, "end": 40.0}]


def test_build_doc_schema_and_f16(labels):
    import jsonschema

    rng = np.random.default_rng(1)
    z = rng.standard_normal((30, 7)).astype(np.float32)
    doc = build_doc("unit", z, mock=True, note="MOCK unit test")
    jsonschema.validate(doc, SCHEMA)
    blob = vertices_to_f16(rng.standard_normal((30, bc.N_FS5)).astype(np.float32))
    a = np.frombuffer(blob, "<f2")
    assert a.size == 30 * bc.N_FS5 and a.min() >= 0 and a.max() <= 1


def test_parse_subtitles_sidecars():
    for p in sorted(bc.TRANSCRIPTS_DIR.glob("*.vtt")):
        cues = parse_subtitles(p)
        assert cues and all(c["end"] >= c["start"] and c["text"] for c in cues), p.name


def test_mock_model_is_content_dependent():
    n = 20
    still = {k: np.zeros(n, np.float32) for k in ("lum", "dlum", "motion", "edge", "cut", "loud", "onset",
                                                  "speech", "words")}
    still["boredom"] = np.linspace(0, 1, n).astype(np.float32)
    moving = dict(still, motion=np.full(n, 0.1, np.float32), cut=np.ones(n, np.float32),
                  boredom=np.zeros(n, np.float32))
    a, b = mock_network_drive(still).mean(0), mock_network_drive(moving).mean(0)
    i = {k: bc.NETWORKS.index(k) for k in bc.NETWORKS}
    assert b[i["DorsAttn"]] > a[i["DorsAttn"]] and b[i["Vis"]] > a[i["Vis"]]
    assert b[i["Default"]] < a[i["Default"]]
    h = canonical_hrf(1.0)
    assert 4 <= int(np.argmax(h)) <= 6 and h.sum() == pytest.approx(1.0)
