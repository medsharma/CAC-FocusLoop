"""CSV loading and windowing for the attention-state model.

Reads the app's label export format (and the SYNTHETIC generator output, which uses the same format):
optional '#' comment lines, then a header with columns `t,label,lapse` + the 42 FeatureFrame columns
(schema order). Extra columns are ignored, except optional grouping columns (`subject`, `subjectID`,
`sessionID`, `session`, `windowID`, `windowId`, and the app export's `session_id` / `window_id`) which start a new contiguous segment when they change.
Rows with an empty / unknown label are kept as model *input* context but never used as a target.
An optional `true_label` column (written by the SYNTHETIC live generator: ground truth before label
noise) is loaded into `Session.y_true` and can be used as the evaluation target (`target="true"`).
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "ml/schema/feature_schema.json").read_text())
FEATS = [f["name"] for f in SCHEMA["features"]]
N_FEAT = len(FEATS)
HZ = int(SCHEMA["rate_hz"])
WINDOW = int(SCHEMA["window_state_s"]) * HZ  # 100 frames
CLASSES = ["focused", "distracted", "drowsy", "stressed"]  # == AttentionState.allCases
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
GROUP_COLS = ("subject", "subjectID", "sessionID", "session", "windowID", "windowId",
              "session_id", "window_id")  # the last two are the app's label export columns
SUBJECT_COLS = ("subject", "subjectID")


@dataclass
class Session:
    path: pathlib.Path
    subject: str
    synthetic: bool
    X: np.ndarray            # (n, 42) float32 raw features
    y: np.ndarray            # (n,) int64 class index, -1 = unlabeled
    seg: np.ndarray          # (n,) int segment id (contiguous runs at ~10 Hz)
    meta: dict = field(default_factory=dict)
    y_true: np.ndarray | None = None   # generator ground truth (SYNTHETIC `true_label` column), if any


def _comments(path: pathlib.Path) -> list[str]:
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            out.append(line[1:].strip())
    return out


def load_csv(path) -> list[Session]:
    """Load one exported CSV. Returns one Session per subject found in the file."""
    path = pathlib.Path(path)
    comments = _comments(path)
    synthetic = any("SYNTHETIC" in c.upper() for c in comments)
    df = pd.read_csv(path, comment="#", skip_blank_lines=True, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ["t", "label"] + FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing[:6]}{'...' if len(missing) > 6 else ''}")
    X = df[FEATS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    X = np.nan_to_num(X, nan=float(SCHEMA["missing_value"]), posinf=0.0, neginf=0.0)
    labels = df["label"].fillna("").astype(str).str.strip().str.lower()
    y = labels.map(lambda s: CLS_IDX.get(s, -1)).to_numpy(np.int64)
    y_true = None
    if "true_label" in df.columns:
        tl = df["true_label"].fillna("").astype(str).str.strip().str.lower()
        y_true = tl.map(lambda s: CLS_IDX.get(s, -1)).to_numpy(np.int64)
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(np.float64)

    # contiguous segments: break on time gaps / reversals or a change of any grouping column
    dt = np.diff(t, prepend=np.nan)
    brk = ~((dt > 0) & (dt <= 2.5 / HZ))
    for c in GROUP_COLS:
        if c in df.columns:
            v = df[c].astype(str).to_numpy()
            brk |= np.concatenate([[True], v[1:] != v[:-1]])
    seg = np.cumsum(brk) - 1

    m = re.search(r"subject=(\S+)", " ".join(comments)) or re.search(r"subj(\d+)", path.stem)
    default_subject = m.group(1) if m else path.stem
    subj_col = next((c for c in SUBJECT_COLS if c in df.columns), None)
    if subj_col is None:
        return [Session(path, str(default_subject), synthetic, X, y, seg, y_true=y_true)]
    out = []
    for s in pd.unique(df[subj_col].astype(str)):
        idx = (df[subj_col].astype(str) == s).to_numpy()
        out.append(Session(path, s, synthetic, X[idx], y[idx], seg[idx],
                           y_true=None if y_true is None else y_true[idx]))
    return out


def load_dir(data_dir) -> list[Session]:
    data_dir = pathlib.Path(data_dir)
    files = sorted(p for p in data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no .csv files under {data_dir}")
    sessions = []
    for f in files:
        sessions.extend(load_csv(f))
    return sessions


class WindowStore:
    """All contiguous segments concatenated, each front-padded by repeating its first frame, so any
    window (and extra history for time-warp augmentation) is one fancy-index away. Keeps memory at
    the size of the frames instead of materialising every overlapping window.

    `ends` are flat row indices of window END frames: every labelled frame from the 100th frame of a
    segment on (segments shorter than WINDOW contribute only their last frame, left-padded, as the app's
    short labelled windows do). `y`/`groups` are the end frames' targets and subjects.
    """

    def __init__(self, sessions: list[Session], min_len: int = 30, target: str = "label", pad: int = 2 * WINDOW):
        F, ends, ys, yts, gs, strided, seglen = [], [], [], [], [], [], []
        off = 0
        self.pad = pad
        for s in sessions:
            yv = s.y_true if (target == "true" and s.y_true is not None) else s.y
            yt = s.y_true if s.y_true is not None else s.y
            for sid in np.unique(s.seg):
                idx = np.nonzero(s.seg == sid)[0]
                L = len(idx)
                if L < min_len:
                    continue
                F.append(np.concatenate([np.repeat(s.X[idx[:1]], pad, 0), s.X[idx]]))
                yseg, ytseg = yv[idx], yt[idx]
                first = min(L, WINDOW)
                for e in range(first, L + 1):
                    if yseg[e - 1] < 0:
                        continue
                    ends.append(off + pad + e - 1); ys.append(yseg[e - 1]); gs.append(s.subject)
                    yts.append(ytseg[e - 1] if ytseg[e - 1] >= 0 else yseg[e - 1])
                    strided.append(e); seglen.append(L)
                off += pad + L
        self.F = np.concatenate(F).astype(np.float32) if F else np.zeros((0, N_FEAT), np.float32)
        self.ends = np.array(ends, np.int64)
        self.y = np.array(ys, np.int64)
        self.y_true = np.array(yts, np.int64)   # generator ground truth when available, else = y
        self.groups = np.array(gs)
        self._e = np.array(strided, np.int64)   # 1-based end position inside its segment
        self._L = np.array(seglen, np.int64)

    def gather(self, ends, length: int = WINDOW) -> np.ndarray:
        ends = np.asarray(ends, np.int64)
        return self.F[ends[:, None] - length + 1 + np.arange(length)[None, :]]

    def strided(self, stride: int = HZ) -> np.ndarray:
        """Indices (into ends/y/groups) of the classic sliding windows: end positions WINDOW,
        WINDOW + stride, ... plus each segment's last frame."""
        e = self._e
        keep = (((e - WINDOW) % stride == 0) & (e >= WINDOW)) | (e == self._L)
        return np.nonzero(keep)[0]


def make_windows(sessions: list[Session], stride: int = HZ, min_len: int = 30, target: str = "label"):
    """Sliding 100-frame windows inside contiguous segments; target = label of the window's last frame.

    Segments shorter than WINDOW (but >= min_len frames, e.g. a short labelled window from the app)
    are left-padded by repeating their first frame. Returns X (N,100,42), y (N,), groups (N,) subject.
    `target="true"` uses the SYNTHETIC generator's `true_label` column when present.
    """
    st = WindowStore(sessions, min_len=min_len, target=target)
    if len(st.y) == 0:
        return np.zeros((0, WINDOW, N_FEAT), np.float32), np.zeros(0, np.int64), np.array([], dtype=str)
    k = st.strided(stride)
    return st.gather(st.ends[k]), st.y[k], st.groups[k]


def subject_split(subjects: list[str], seed: int = 0, frac=(0.7, 0.15, 0.15)):
    """Assign subjects to train/val/test. With fewer than 3 subjects, returns None (caller falls back)."""
    subs = sorted(set(subjects))
    if len(subs) < 3:
        return None
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(frac[1] * len(subs))))
    n_test = max(1, int(round(frac[2] * len(subs))))
    return {"train": subs[: len(subs) - n_val - n_test],
            "val": subs[len(subs) - n_val - n_test: len(subs) - n_test],
            "test": subs[len(subs) - n_test:]}


def temporal_split(n: int, frac=(0.7, 0.15, 0.15)):
    """Fallback for single-subject (e.g. one user's real data): contiguous chronological blocks."""
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    idx = np.arange(n)
    return {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}
