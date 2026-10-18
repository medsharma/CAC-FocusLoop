"""Tests for the attention-state pipeline (ml/state). All data here is SYNTHETIC."""
import json
import pathlib
import platform
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/state"))
sys.path.insert(0, str(ROOT / "ml/synth"))

import data as sdata  # noqa: E402
from model import EMBED_DIM, AttentionStateNet, n_params, load_checkpoint  # noqa: E402

FEATS = sdata.FEATS


# ---------- model ----------

def test_model_shapes_and_size():
    m = AttentionStateNet().eval()
    assert n_params(m) < 100_000
    assert 16 <= EMBED_DIM <= 32
    p, e = m(torch.randn(3, 100, 42))
    assert p.shape == (3, 4) and e.shape == (3, EMBED_DIM)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-5)


def test_normalisation_is_baked_in():
    m = AttentionStateNet().eval()
    mean, std = np.arange(42, dtype=np.float32), np.full(42, 3.0, np.float32)
    m.set_norm(mean, std)
    x = torch.randn(2, 100, 42)
    m2 = AttentionStateNet().eval()
    m2.load_state_dict(m.state_dict())
    m2.set_norm(np.zeros(42), np.ones(42))
    p1, _ = m(x * 3 + torch.from_numpy(mean))
    p2, _ = m2(x)
    assert torch.allclose(p1, p2, atol=1e-5)


# ---------- CSV loader ----------

def _write_app_csv(path, rows, labels, extra_cols=None, comment=None):
    cols = ["t", "label", "lapse"] + FEATS + list(extra_cols or {})
    with open(path, "w") as fh:
        if comment:
            fh.write(f"# {comment}\n")
        fh.write(",".join(cols) + "\n")
        for i, (r, lab) in enumerate(zip(rows, labels)):
            vals = [f"{r[0]:.1f}", lab, "0" if lab in ("focused", "") else "1"] + [f"{v:.4f}" for v in r[1:]]
            vals += [str(v[i]) for v in (extra_cols or {}).values()]
            fh.write(",".join(vals) + "\n")


def test_loader_handles_app_export(tmp_path):
    n = 250
    rng = np.random.default_rng(0)
    rows = np.concatenate([np.arange(n)[:, None] / 10, rng.random((n, 42))], 1)
    labels = ["focused"] * 100 + [""] * 50 + ["Drowsy"] * 100  # empty = unlabeled, case-insensitive
    _write_app_csv(tmp_path / "export.csv", rows, labels)
    (s,) = sdata.load_csv(tmp_path / "export.csv")
    assert not s.synthetic and s.X.shape == (n, 42) and s.X.dtype == np.float32
    assert (s.y[:100] == 0).all() and (s.y[100:150] == -1).all() and (s.y[150:] == 2).all()
    X, y, g = sdata.make_windows([s], stride=10)
    assert X.shape[1:] == (100, 42)
    assert set(y.tolist()) <= {0, 2}  # unlabeled ends are skipped
    np.testing.assert_allclose(X[0], s.X[:100])


def test_loader_segments_on_gaps_and_session_column(tmp_path):
    # two labelled windows of 6 s each from different sessions, with a time reset in between
    n = 60
    t = np.concatenate([np.arange(n), np.arange(n)]) / 10
    rows = np.concatenate([t[:, None], np.random.default_rng(1).random((2 * n, 42))], 1)
    labels = ["stressed"] * n + ["distracted"] * n
    _write_app_csv(tmp_path / "e.csv", rows, labels, extra_cols={"sessionID": ["a"] * n + ["b"] * n})
    (s,) = sdata.load_csv(tmp_path / "e.csv")
    assert len(np.unique(s.seg)) == 2
    X, y, _ = sdata.make_windows([s], stride=10)
    assert len(y) == 2 and sorted(y.tolist()) == [1, 3]
    # short windows are left-padded by repeating the first frame
    np.testing.assert_allclose(X[0, :40], np.repeat(X[0, 40:41], 40, 0))


def test_loader_rejects_missing_columns(tmp_path):
    (tmp_path / "bad.csv").write_text("t,label,lapse,faceDetected\n0.0,focused,0,1\n")
    with pytest.raises(ValueError):
        sdata.load_csv(tmp_path / "bad.csv")


def test_replay_fixture_loads_as_synthetic():
    (s,) = sdata.load_csv(ROOT / "Resources/Replay/synthetic_session.replay.csv")
    assert s.synthetic and s.X.shape == (1800, 42) and (s.y >= 0).all()


# ---------- tiny end-to-end: generate -> train -> eval -> export ----------

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    import generate_dataset
    import train
    d = tmp_path_factory.mktemp("synth")
    generate_dataset.main(subjects=4, sessions=1, seconds=150, seed=5, out=d / "csv")
    run = d / "run"
    train.main(["--data", str(d / "csv"), "--out", str(run), "--epochs", "2", "--stride", "20"])
    return d, run


def test_train_and_eval_tiny(tiny_run):
    import eval as seval
    d, run = tiny_run
    model, ck = load_checkpoint(run / "model.pt")
    assert ck["synthetic"] and ck["split"]["kind"] == "subject"
    assert json.loads((run / "norm.json").read_text())["features"] == FEATS
    m = seval.main(["--data", str(d / "csv"), "--ckpt", str(run / "model.pt"), "--reports", str(d / "rep")])
    assert 0 <= m["ece"] <= 1 and m["n"] > 0
    for f in ["state_SYNTHETIC_confusion.png", "state_SYNTHETIC_reliability.png", "state_SYNTHETIC_metrics.json"]:
        assert (d / "rep" / f).exists()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML predict needs macOS")
def test_export_parity(tiny_run):
    import export_coreml
    d, run = tiny_run
    out = d / "AttentionStateModel.mlpackage"
    meta = export_coreml.main(["--ckpt", str(run / "model.pt"), "--out", str(out),
                               "--data", str(d / "csv"), "--n-check", "16"])
    assert out.exists() and out.with_suffix(".json").exists()
    assert meta["training_data"] == "SYNTHETIC"
    par = meta["parity_check"]
    assert par["output_shapes"] == {"probabilities": [1, 4], "embedding": [1, EMBED_DIM]}
    assert par["max_abs_diff_probabilities"] < 2e-2
    assert par["argmax_agreement"] >= 0.9


def test_shipped_model_metadata():
    mj = ROOT / "Resources/Models/AttentionStateModel.json"
    if not mj.exists():
        pytest.skip("model not exported yet")
    meta = json.loads(mj.read_text())
    assert meta["feature_names"] == FEATS and meta["classes"] == sdata.CLASSES
    assert meta["io"]["inputs"]["window"]["shape"] == [1, 100, 42]
    assert meta["training_data"] in ("SYNTHETIC", "REAL")


def test_loader_segments_on_app_window_id(tmp_path):
    # the app's label export (CSVLabelSink) names its grouping columns session_id / window_id
    n = 50
    t = np.arange(2 * n) / 10  # contiguous in time: only window_id separates the two windows
    rows = np.concatenate([t[:, None], np.random.default_rng(2).random((2 * n, 42))], 1)
    labels = ["focused"] * n + ["drowsy"] * n
    _write_app_csv(tmp_path / "w.csv", rows, labels,
                   extra_cols={"session_id": ["s"] * (2 * n), "source": ["experienceSample"] * (2 * n),
                               "window_id": ["w1"] * n + ["w2"] * n})
    (s,) = sdata.load_csv(tmp_path / "w.csv")
    assert len(np.unique(s.seg)) == 2


def _replay_windows():
    (s,) = sdata.load_csv(ROOT / "Resources/Replay/synthetic_session.replay.csv")
    return sdata.make_windows([s], stride=10)


def test_heuristic_port_matches_documented_replay_accuracy():
    import heuristic
    X, y, _ = _replay_windows()
    P, face_missing = heuristic.predict(X)
    assert not face_missing.any()
    assert (P.argmax(1) == y).mean() == pytest.approx(0.977, abs=0.01)   # recorded value: 0.978


def test_true_label_column_is_loaded(tmp_path):
    n = 120
    rows = np.concatenate([np.arange(n)[:, None] / 10, np.random.default_rng(3).random((n, 42))], 1)
    _write_app_csv(tmp_path / "s.csv", rows, ["drowsy"] * n, extra_cols={"true_label": ["focused"] * n},
                   comment="SYNTHETIC")
    (s,) = sdata.load_csv(tmp_path / "s.csv")
    assert (s.y == 2).all() and (s.y_true == 0).all()
    _, y, _ = sdata.make_windows([s], target="true")
    assert (y == 0).all()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML predict needs macOS")
def test_shipped_model_on_replay():
    import coremltools as ct
    pkg = ROOT / "Resources/Models/AttentionStateModel.mlpackage"
    if not pkg.exists():
        pytest.skip("model not exported yet")
    X, y, _ = _replay_windows()
    ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    P = np.concatenate([np.asarray(ml.predict({"window": x[None]})["probabilities"]).reshape(1, -1) for x in X])
    assert (P.argmax(1) == y).mean() >= 0.95
