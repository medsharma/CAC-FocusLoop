"""Dataset assembly for the LapseForecaster: sessions -> (20 s feature window, 20 s content window, target).

Content features mirror `BrainContentTimeline.features(at:)` + `ContentFeatures.vector` in Swift exactly:
7 z-scored networks (BrainNetwork order), engagement, driftRisk, min(60, secondsToValley)/60, inValley.
Sessions without a lecture use `ContentFeatures.none` (networks 0, engagement 0.5, drift 0, 1.0, 0).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import label_shift as L

NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
CONTENT_NAMES = NETS + ["engagement", "driftRisk", "secondsToValleyNorm", "inValley"]
N_FEAT, N_CONTENT = len(L.FEATS), len(CONTENT_NAMES)
WIN = L.WINDOW_S * L.HZ  # 200 frames
CWIN = L.WINDOW_S        # 20 content steps (1 Hz)
NONE_CONTENT = np.array([0.0] * 7 + [0.5, 0.0, 1.0, 0.0], np.float32)


def _sample(series, x: float) -> float:
    s = series
    if len(s) == 0:
        return 0.0
    c = max(0.0, min(len(s) - 1.0, x))
    i = int(c); j = min(i + 1, len(s) - 1); f = c - i
    return s[i] * (1 - f) + s[j] * f


def content_features(doc: dict | None, playhead: float) -> np.ndarray:
    """Python mirror of BrainContentTimeline.features(at:).vector."""
    if doc is None:
        return NONE_CONTENT.copy()
    hz = doc.get("hz", 1)
    x = playhead * hz
    nets = [_sample(doc["networks"][k], x) for k in NETS]
    eng = _sample(doc["predicted_engagement"], x)
    drift = _sample(doc["predicted_drift_risk"], x)
    valleys = doc.get("valleys") or []
    inside = any(v["start"] <= playhead < v["end"] for v in valleys)
    nxt = next((v["start"] - playhead for v in valleys if v["start"] >= playhead), 60.0)
    stv = 0.0 if inside else min(60.0, nxt)
    return np.array(nets + [eng, drift, min(60.0, max(0.0, stv)) / 60.0, 1.0 if inside else 0.0], np.float32)


@dataclass
class Session:
    subject: str
    session_id: str
    frames: np.ndarray          # (n, 42) float32
    content: np.ndarray         # (seconds, 11) float32, content at session second s
    has_content: bool
    events: dict
    lapse: np.ndarray | None    # per-frame bool
    synthetic: bool
    t_end: float
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0))   # sample times (s)
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    exclude: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def load_manifest(path) -> list[Session]:
    """Manifest CSV columns: subject, session_id, session_csv, sart_csv, es_csv, lecture_json,
    playhead_offset, synthetic. Paths are relative to the manifest file. Empty = not available."""
    path = pathlib.Path(path)
    base = path.parent
    man = pd.read_csv(path, comment="#", dtype=str).fillna("")
    rows = man.to_dict("records")
    # per-user CV threshold over all that user's SART sessions
    sarts_by_user: dict[str, list] = {}
    for r in rows:
        if r.get("sart_csv"):
            sarts_by_user.setdefault(r["subject"], []).append(L.read_sart(base / r["sart_csv"]))
    thr = {u: L.user_cv_threshold(s) for u, s in sarts_by_user.items()}
    lectures: dict[str, dict] = {}
    out = []
    for r in rows:
        df = L.load_session(base / r["session_csv"])
        sart = L.read_sart(base / r["sart_csv"]) if r.get("sart_csv") else None
        es = L.read_csv(base / r["es_csv"]) if r.get("es_csv") else None
        ev = L.session_events(df, sart, es, thr.get(r["subject"]))
        t = df["t"].to_numpy(float)
        frames = df[L.FEATS].to_numpy(np.float32)
        t_end = float(t[-1]) + 1.0 / L.HZ
        seconds = int(np.ceil(t_end)) + 1
        doc = None
        if r.get("lecture_json"):
            lp = base / r["lecture_json"]
            doc = lectures.setdefault(str(lp), json.loads(lp.read_text()))
        offset = float(r.get("playhead_offset") or 0)
        if doc is not None and "playhead" in df.columns and df["playhead"].notna().any():
            ph = df["playhead"].to_numpy(float)[::L.HZ]
            ph = np.concatenate([ph, ph[-1] + np.arange(1, seconds - len(ph) + 1)])[:seconds] if len(ph) < seconds else ph[:seconds]
        else:
            ph = np.arange(seconds) + offset
        content = np.stack([content_features(doc, p) for p in ph]) if doc is not None else np.tile(NONE_CONTENT, (seconds, 1))
        s = Session(r["subject"], r["session_id"], frames, content, doc is not None, ev, L.lapse_mask(df),
                    r.get("synthetic", "0") in ("1", "true", "True"), t_end)
        s.samples = L.sample_times(t_end)
        s.y = L.shift_targets(s.samples, L.all_events(ev))
        s.exclude = L.in_lapse_at(df, s.samples)
        out.append(s)
    return out


def window(sess: Session, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Frames covering (t-20, t] (200 rows, oldest first, front-padded by repeating the first frame)
    and content for seconds t-19..t (20 rows)."""
    end = int(round(t * L.HZ))  # frame index of time t (inclusive)
    a = end - WIN + 1
    if a >= 0:
        fw = sess.frames[a:end + 1]
    else:
        fw = np.concatenate([np.repeat(sess.frames[:1], -a, 0), sess.frames[:end + 1]])
    if len(fw) < WIN:  # t beyond the end
        fw = np.concatenate([fw, np.repeat(fw[-1:], WIN - len(fw), 0)])
    secs = np.clip(np.arange(int(round(t)) - CWIN + 1, int(round(t)) + 1), 0, len(sess.content) - 1)
    return fw, sess.content[secs]


def split_subjects(sessions: list[Session], seed: int = 0, frac=(0.65, 0.175, 0.175)) -> dict[str, list[str]]:
    subs = sorted({s.subject for s in sessions})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = max(1, int(round(frac[0] * n))); n_va = max(1, int(round(frac[1] * n))) if n >= 3 else 0
    return {"train": subs[:n_tr], "val": subs[n_tr:n_tr + n_va], "test": subs[n_tr + n_va:] or subs[-1:]}


def build_arrays(sessions: list[Session], include_excluded: bool = False):
    """Materialise (X, C, y, session_idx, t) for a list of sessions."""
    X, C, Y, SI, T = [], [], [], [], []
    for si, s in enumerate(sessions):
        for t, y, ex in zip(s.samples, s.y, s.exclude):
            if ex and not include_excluded:
                continue
            fw, cw = window(s, t)
            X.append(fw); C.append(cw); Y.append(y); SI.append(si); T.append(t)
    if not X:
        return (np.zeros((0, WIN, N_FEAT), np.float32), np.zeros((0, CWIN, N_CONTENT), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    return np.stack(X).astype(np.float32), np.stack(C).astype(np.float32), np.array(Y), np.array(SI), np.array(T)



class FlatStore:
    """All sessions' frames/content concatenated (each front-padded like `window`) so batches of
    windows are gathered with one fancy-index instead of a Python loop. Equivalent to `window()`."""

    EXTRA = 60  # extra history frames available to gather(extra=...) (time-warp augmentation)

    def __init__(self, sessions: list[Session]):
        fs, cs, fo, co, nf, nc = [], [], [], [], [], []
        pf = pc = 0
        for s in sessions:
            f = np.concatenate([np.repeat(s.frames[:1], WIN - 1 + self.EXTRA, 0), s.frames])
            c = np.concatenate([np.repeat(s.content[:1], CWIN - 1, 0), s.content])
            fs.append(f); cs.append(c); fo.append(pf); co.append(pc)
            nf.append(len(s.frames)); nc.append(len(s.content))
            pf += len(f); pc += len(c)
        self.F = np.concatenate(fs).astype(np.float32)
        self.C = np.concatenate(cs).astype(np.float32)
        self.fo, self.co = np.array(fo), np.array(co)
        self.nf, self.nc = np.array(nf), np.array(nc)
        self._af, self._ac = np.arange(WIN), np.arange(CWIN)

    def gather(self, si, t, extra: int = 0):
        """(X (B, WIN + extra, 42), C (B, CWIN, 11)) for sessions `si` at times `t`; `extra` older
        frames (<= EXTRA) come first, for time-warp augmentation."""
        si = np.asarray(si); t = np.asarray(t, float)
        end = np.clip(np.round(t * L.HZ).astype(int), 0, self.nf[si] - 1)
        af = np.arange(self.EXTRA - extra, self.EXTRA + WIN)
        X = self.F[(self.fo[si] + end)[:, None] + af[None, :]]
        cs = np.clip(np.round(t).astype(int), 0, self.nc[si] - 1)
        C = self.C[(self.co[si] + cs)[:, None] + self._ac[None, :]]
        return X, C
