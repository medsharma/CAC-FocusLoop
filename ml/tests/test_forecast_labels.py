"""Hand-built checks for forecaster label shifting and metrics (ml/forecast)."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "forecast"))
import data as D  # noqa: E402
import label_shift as L  # noqa: E402
import metrics as M  # noqa: E402


# ---------------------------------------------------------------- targets

def test_shift_window_is_open_closed():
    ts = np.array([0, 5, 10, 15, 20, 25], float)
    y = L.shift_targets(ts, [20.0], horizon=15)
    # event at 20 is positive for t in [5, 20): (t, t+15] contains 20
    assert y.tolist() == [0, 1, 1, 1, 0, 0]


def test_shift_multiple_events_and_empty():
    ts = np.arange(0, 60, 1.0)
    y = L.shift_targets(ts, [30.0, 50.0])
    assert y[14] == 0 and y[15] == 1 and y[29] == 1 and y[30] == 0 and y[35] == 1 and y[49] == 1 and y[50] == 0
    assert L.shift_targets(ts, []).sum() == 0


def test_sample_times():
    assert L.sample_times(60.0).tolist() == list(range(20, 46))
    assert L.sample_times(30.0).size == 0
    assert L.sample_times(30.0, full_horizon=False).tolist() == list(range(20, 31))


# ---------------------------------------------------------------- event rules

def test_label_onsets():
    t = np.arange(10) / 10
    lab = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0], bool)
    assert L.label_onsets(t, lab).tolist() == pytest.approx([0.2, 0.5])
    assert L.label_onsets(t, np.ones(10, bool)).tolist() == [0.0]


def test_lapse_mask_from_label_or_lapse_column():
    df = pd.DataFrame({"t": [0, .1, .2], "label": ["focused", "drowsy", "focused"]})
    assert L.lapse_mask(df).tolist() == [False, True, False]
    df = pd.DataFrame({"t": [0, .1, .2], "lapse": [0, 1, 1]})
    assert L.lapse_mask(df).tolist() == [False, True, True]
    assert L.lapse_mask(pd.DataFrame({"t": [0]})) is None


def test_commission_errors():
    sart = pd.DataFrame({"t": [1, 2, 3, 4], "digit": [3, 3, 5, 3], "responded": [1, 0, 1, 1], "rt": [.3, "", .4, .2]})
    assert L.commission_events(sart).tolist() == [1, 4]


def test_rt_cv_threshold_and_rising_edges():
    # go trials: stable RTs, then a burst of variable RTs, then stable again
    rts = [0.40, 0.41, 0.40, 0.41, 0.40, 0.41, 0.20, 0.60, 0.25, 0.55, 0.40, 0.41, 0.40, 0.41, 0.40, 0.41]
    sart = pd.DataFrame({"t": np.arange(len(rts)) * 1.15, "digit": 5, "responded": 1, "rt": rts})
    t, cv = L.go_rt_cv(sart)
    assert len(cv) == len(rts) - 3
    assert cv[0] == pytest.approx(np.std(rts[:4]) / np.mean(rts[:4]))
    thr = L.user_cv_threshold([sart])
    assert thr == pytest.approx(np.percentile(cv, 80))
    ev = L.rt_spike_events(sart, thr)
    # exactly one spike episode -> one rising edge, at a trial inside the variable burst
    assert len(ev) == 1
    assert 6 * 1.15 <= ev[0] <= 10 * 1.15
    # no-go trials and missing RTs are ignored
    sart2 = pd.concat([sart, pd.DataFrame({"t": [100.0], "digit": [3], "responded": [1], "rt": [0.1]})])
    assert len(L.go_rt_cv(sart2)[1]) == len(cv)


def test_es_events_and_session_union():
    es = pd.DataFrame({"t": [30.0, 60.0, 90.0], "focused": [1, 0, 0], "reason": ["", "tired", "distracted"]})
    assert L.es_events(es).tolist() == [60.0, 90.0]
    n = 1000
    df = pd.DataFrame({"t": np.arange(n) / 10, "label": ["focused"] * 400 + ["distracted"] * 100 + ["focused"] * 500})
    ev = L.session_events(df, None, es, None)
    assert ev["label_onset"].tolist() == [40.0]
    assert L.all_events(ev).tolist() == [40.0, 60.0, 90.0]
    ts = L.sample_times(100.0)
    excl = L.in_lapse_at(df, ts)
    assert excl[ts.tolist().index(45.0)] and not excl[ts.tolist().index(39.0)]


# ---------------------------------------------------------------- metrics

def test_lead_time_hand_built():
    t = np.arange(0, 100, 1.0)
    p = np.zeros_like(t)
    p[40:46] = 0.9      # alert from 40 s; onset at 50 -> lead 10
    p[78:80] = 0.9      # alert from 78 s; onset at 80 -> lead 2
    p[60] = 0.9         # alert at 60; onset at 99 has no alert in [84, 99) -> miss
    leads = M.lead_times(t, p, [50.0, 80.0, 99.0], threshold=0.5)
    assert leads[0] == pytest.approx(10) and leads[1] == pytest.approx(2) and np.isnan(leads[2])
    s = M.lead_summary(leads)
    assert s["onsets"] == 3 and s["detected"] == 2 and s["median_lead_s"] == pytest.approx(6)
    assert s["detection_rate"] == pytest.approx(2 / 3)


def test_lead_time_alert_already_on_and_alert_at_onset_not_counted():
    t = np.arange(0, 60, 1.0)
    p = np.full_like(t, 0.9)
    assert M.lead_times(t, p, [40.0], 0.5)[0] == pytest.approx(15)   # first scorable alert is at o-15
    p = np.zeros_like(t); p[40] = 0.9
    assert np.isnan(M.lead_times(t, p, [40.0], 0.5)[0])               # alert AT onset is not warning
    # onset too early to have a full look-back window is skipped
    assert M.lead_times(np.arange(20, 60, 1.0), np.zeros(40), [25.0], 0.5).size == 0


def test_precision_at_alert_rate():
    p = np.linspace(0, 1, 100, endpoint=False)
    y = (p >= 0.95).astype(int)
    prec, rec, thr, realised = M.precision_at_alert_rate(y, p, 0.10)
    assert realised == pytest.approx(0.10, abs=0.011)
    assert prec == pytest.approx(0.5, abs=0.06) and rec == pytest.approx(1.0)
    assert M.auroc(y, p) == pytest.approx(1.0)


# ---------------------------------------------------------------- windows / content mirror

def _session(n_frames=400, seconds=41):
    frames = np.arange(n_frames * 42, dtype=np.float32).reshape(n_frames, 42)
    content = np.arange(seconds * 11, dtype=np.float32).reshape(seconds, 11)
    return D.Session("S", "x", frames, content, True, {}, None, True, n_frames / 10)


def test_window_padding_and_flatstore_equivalence():
    s = _session()
    fw, cw = D.window(s, 5.0)
    assert fw.shape == (200, 42) and cw.shape == (20, 11)
    assert (fw[:150] == s.frames[0]).all() and (fw[-1] == s.frames[50]).all()
    assert (cw[-1] == s.content[5]).all() and (cw[0] == s.content[0]).all()
    store = D.FlatStore([s])
    for t in [0.0, 5.0, 19.9, 20.0, 33.3, 39.9]:
        X, C = store.gather([0], [t])
        a, b = D.window(s, t)
        assert np.array_equal(X[0], a) and np.array_equal(C[0], b), t


def test_content_features_mirror_swift():
    doc = {"hz": 1, "networks": {k: [float(i)] * 100 for i, k in enumerate(D.NETS)},
           "predicted_engagement": [0.5] * 100, "predicted_drift_risk": list(np.linspace(0, 1, 100)),
           "valleys": [{"start": 30, "end": 40}]}
    v = D.content_features(doc, 10)
    assert v[:7].tolist() == list(range(7)) and v[9] == pytest.approx(20 / 60) and v[10] == 0
    assert D.content_features(doc, 35)[9] == 0 and D.content_features(doc, 35)[10] == 1
    assert D.content_features(doc, 50)[9] == 1.0          # no valley ahead -> 60 s -> 1.0
    assert D.content_features(None, 3).tolist() == D.NONE_CONTENT.tolist()


def test_read_sart_accepts_app_export(tmp_path):
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "forecast"))
    L = importlib.import_module("label_shift")
    p = tmp_path / "FocusLoop_sart_trials.csv"
    p.write_text("# FocusLoop SART export (REAL device data)\n"
                 "trial,onset_s,digit,is_nogo,tapped,rt_ms,commission,omission\n"
                 "0,1.0,5,0,1,350,0,0\n1,2.15,3,1,1,300,1,0\n2,3.3,7,0,0,,0,1\n")
    df = L.read_sart(p)
    assert list(df.columns) == ["t", "digit", "responded", "rt"]
    assert df["rt"].iloc[0] == 0.35 and df["responded"].iloc[2] == 0
    assert list(L.commission_events(df)) == [2.15]


def test_manifest_accepts_app_feature_stream_export(tmp_path):
    """The app's Settings > Export feature streams writes exactly this layout (FeatureStreamStore.swift):
    manifest.csv + stream-*.csv (`t,playhead,<42 features>`, '#' header) + optional .es.csv + brain JSON."""
    import json as _json
    feats = L.FEATS
    n = 400  # 40 s at 10 Hz
    rng = np.random.default_rng(0)
    head = "# FocusLoop feature stream v1 (SYNTHETIC replay)\n" + ",".join(["t", "playhead"] + feats) + "\n"
    body = "".join(f"{i / 10:.3f},{i / 10 + 2:.2f}," + ",".join(f"{v:.4f}" for v in rng.random(len(feats))) + "\n"
                   for i in range(n))
    (tmp_path / "stream-1-lecture_x-SYNTHETIC.csv").write_text(head + body)
    (tmp_path / "stream-1-lecture_x-SYNTHETIC.es.csv").write_text("t,focused,reason\n30.000,0,tired\n")
    doc = {"lecture_id": "x", "hz": 1, "mock": True,
           "networks": {k: [0.0] * 60 for k in ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]},
           "predicted_engagement": [0.5] * 60, "predicted_drift_risk": [0.0] * 60, "valleys": [{"start": 30, "end": 40}]}
    (tmp_path / "x.brain.json").write_text(_json.dumps(doc))
    (tmp_path / "manifest.csv").write_text(
        "subject,session_id,session_csv,sart_csv,es_csv,lecture_json,playhead_offset,synthetic\n"
        "me,stream-1-lecture_x-SYNTHETIC,stream-1-lecture_x-SYNTHETIC.csv,,stream-1-lecture_x-SYNTHETIC.es.csv,"
        "x.brain.json,0,1\n")
    (s,) = D.load_manifest(tmp_path / "manifest.csv")
    assert s.synthetic and s.has_content and s.frames.shape == (n, len(feats))
    assert s.y.sum() > 0  # the ES "not focused" answer at 30 s yields positive targets before it
    assert s.content[0, 10] == 0 and s.content[30, 10] == 1  # playhead column drives content (t + 2 s)


def test_flatstore_extra_history_ends_with_the_plain_window():
    s = _session()
    store = D.FlatStore([s])
    for t in [0.0, 5.0, 20.0, 39.9]:
        X, _ = store.gather([0], [t])
        Xe, _ = store.gather([0], [t], extra=50)
        assert Xe.shape == (1, D.WIN + 50, 42) and np.array_equal(Xe[0, 50:], X[0])


def test_shipped_forecaster_metadata_is_sane():
    import json
    root = pathlib.Path(__file__).resolve().parents[2]
    for name, uses in [("LapseForecaster", True), ("LapseForecasterNoContent", False)]:
        p = root / f"Resources/Models/{name}.json"
        if not p.exists():
            pytest.skip("models not exported yet")
        meta = json.loads(p.read_text())
        assert meta["uses_content"] == uses and meta["inputs"]["features"] == [1, 200, 42]
        assert 0.1 < meta["alert_threshold"] < 0.9          # a usable (non-degenerate) alert threshold
        assert meta["parity_max_abs_diff"] < 0.02
        assert not meta["synthetic"] or "SYNTHETIC" in meta["note"]
