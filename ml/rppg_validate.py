"""Compare FocusLoop camera rPPG heart rate against Apple Watch heart rate.

Inputs: two CSVs with columns `timestamp,bpm` (timestamp = Unix seconds or ISO-8601). Both series
are aligned on a common 1 Hz grid over their overlapping span (linear interpolation, no
extrapolation; grid points more than --max-gap seconds from a real sample on either side are
dropped). Outputs to ml/reports/: a scatter plot, a Bland-Altman plot (mean bias, +-1.96 SD limits
of agreement), and a JSON summary with N, MAE, RMSE, bias, LoA and Pearson r.

Honesty guard: if either input mentions SYNTHETIC (file name, a comment line, or a `label`/`source`
column), every output is labelled SYNTHETIC and the script refuses --real-label. `--demo-synthetic`
writes clearly labelled SYNTHETIC inputs and runs the comparison on them to prove the script works;
its plots are titled "SYNTHETIC DEMO - NOT REAL DATA".

Usage:
  ml/.venv/bin/python ml/rppg_validate.py --rppg focusloop_hr.csv --watch watch_hr.csv
  ml/.venv/bin/python ml/rppg_validate.py --demo-synthetic
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ML = Path(__file__).resolve().parent
REPORTS = ML / "reports"
DEMO_DIR = ML / "synth" / "out"
SYNTH_TITLE = "SYNTHETIC DEMO - NOT REAL DATA"



def _repo_rel(p) -> str:
    """Path relative to the repo root when inside it, so reports don't record local user paths."""
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[1]
    q = _pl.Path(p).resolve()
    try:
        return str(q.relative_to(root))
    except ValueError:
        return _pl.Path(p).name

def is_synthetic(path: Path) -> bool:
    if "synthetic" in path.name.lower():
        return True
    text = path.read_text(errors="replace")
    return "SYNTHETIC" in text.upper()


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"timestamp", "bpm"} <= set(df.columns):
        raise SystemExit(f"{path}: needs columns timestamp,bpm (got {list(df.columns)})")
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        secs = ts.astype(float)
    else:
        secs = pd.to_datetime(ts, utc=True).astype("int64") / 1e9
    s = pd.Series(df["bpm"].astype(float).values, index=secs.values).dropna()
    s = s[(s > 20) & (s < 250)]
    return s[~s.index.duplicated()].sort_index()


def align_1hz(a: pd.Series, b: pd.Series, max_gap: float) -> pd.DataFrame:
    start, end = max(a.index.min(), b.index.min()), min(a.index.max(), b.index.max())
    if not end > start:
        raise SystemExit("The two recordings do not overlap in time.")
    grid = np.arange(math.ceil(start), math.floor(end) + 1, 1.0)

    def interp(s: pd.Series) -> np.ndarray:
        x = s.index.values.astype(float)
        y = np.interp(grid, x, s.values)
        idx = np.searchsorted(x, grid)
        prev_gap = grid - x[np.clip(idx - 1, 0, len(x) - 1)]
        next_gap = x[np.clip(idx, 0, len(x) - 1)] - grid
        exact = np.isin(grid, x)
        ok = exact | ((prev_gap <= max_gap) & (next_gap <= max_gap) & (prev_gap >= 0) & (next_gap >= 0))
        y[~ok] = np.nan
        return y

    out = pd.DataFrame({"t": grid, "rppg": interp(a), "watch": interp(b)}).dropna()
    if len(out) < 3:
        raise SystemExit("Fewer than 3 aligned 1 Hz points; check timestamps / --max-gap.")
    return out


def metrics(df: pd.DataFrame) -> dict:
    d = df["rppg"].values - df["watch"].values
    sd = float(np.std(d, ddof=1))
    bias = float(np.mean(d))
    return {
        "n_points_1hz": int(len(df)),
        "mae_bpm": float(np.mean(np.abs(d))),
        "rmse_bpm": float(np.sqrt(np.mean(d ** 2))),
        "bias_bpm": bias,
        "sd_diff_bpm": sd,
        "loa_low_bpm": bias - 1.96 * sd,
        "loa_high_bpm": bias + 1.96 * sd,
        "pearson_r": float(np.corrcoef(df["rppg"], df["watch"])[0, 1]),
    }


def plot(df: pd.DataFrame, m: dict, title_tag: str, prefix: str) -> list[Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    paths = []
    lo = float(min(df["rppg"].min(), df["watch"].min())) - 5
    hi = float(max(df["rppg"].max(), df["watch"].max())) + 5

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["watch"], df["rppg"], s=10, alpha=0.6, color="#3a6ea5")
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", label="identity")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Apple Watch HR (BPM)"); ax.set_ylabel("FocusLoop rPPG HR (BPM)")
    ax.set_title(f"{title_tag}\nrPPG vs Apple Watch (N={m['n_points_1hz']}, MAE {m['mae_bpm']:.1f}, "
                 f"RMSE {m['rmse_bpm']:.1f}, r {m['pearson_r']:.2f})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = REPORTS / f"{prefix}_scatter.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    mean = (df["rppg"] + df["watch"]) / 2
    diff = df["rppg"] - df["watch"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(mean, diff, s=10, alpha=0.6, color="#3a6ea5")
    for y, lab, ls in [(m["bias_bpm"], f"bias {m['bias_bpm']:+.1f}", "-"),
                       (m["loa_high_bpm"], f"+1.96 SD {m['loa_high_bpm']:+.1f}", "--"),
                       (m["loa_low_bpm"], f"-1.96 SD {m['loa_low_bpm']:+.1f}", "--")]:
        ax.axhline(y, color="#c0504d" if ls == "-" else "#888888", ls=ls, lw=1)
        ax.text(ax.get_xlim()[1], y, f" {lab}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Mean of rPPG and Watch (BPM)"); ax.set_ylabel("rPPG - Watch (BPM)")
    ax.set_title(f"{title_tag}\nBland-Altman: rPPG vs Apple Watch", fontsize=9)
    fig.tight_layout()
    p = REPORTS / f"{prefix}_bland_altman.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    paths.append(p)
    return paths


def make_demo_inputs(seed: int = 7) -> tuple[Path, Path]:
    """Writes SYNTHETIC rPPG and 'watch' CSVs (labelled in file name, header comment and column)."""
    rng = np.random.default_rng(seed)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    t0 = 1_700_000_000.0
    t = np.arange(0, 600, 1.0)
    true_hr = 68 + 6 * np.sin(2 * np.pi * t / 240) + np.cumsum(rng.normal(0, 0.15, t.size))
    watch_t = np.sort(rng.choice(t, size=120, replace=False)) + rng.uniform(0, 0.9, 120)
    watch = np.interp(watch_t, t, true_hr) + rng.normal(0, 1.0, watch_t.size)
    rppg = true_hr + 1.0 + rng.normal(0, 2.5, t.size)
    rppg_p, watch_p = DEMO_DIR / "rppg_hr_SYNTHETIC.csv", DEMO_DIR / "watch_hr_SYNTHETIC.csv"
    for path, ts, bpm in [(rppg_p, t0 + t, rppg), (watch_p, t0 + watch_t, watch)]:
        with path.open("w") as f:
            f.write("# SYNTHETIC demo data generated by ml/rppg_validate.py --demo-synthetic. NOT REAL DATA.\n")
            f.write("timestamp,bpm,label\n")
            for a, b in zip(ts, bpm):
                f.write(f"{a:.3f},{b:.2f},SYNTHETIC\n")
    return rppg_p, watch_p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rppg", "--focusloop", dest="rppg", type=Path,
                    help="FocusLoop camera heart-rate CSV (timestamp,bpm): Session report > Heart rate (CSV)")
    ap.add_argument("--watch", type=Path, help="Apple Watch HR CSV (timestamp,bpm)")
    ap.add_argument("--max-gap", type=float, default=10.0,
                    help="max seconds to a real sample for 1 Hz interpolation (Watch samples ~every 5 s)")
    ap.add_argument("--demo-synthetic", action="store_true", help="generate SYNTHETIC inputs and run on them")
    ap.add_argument("--real-label", action="store_true",
                    help="assert the inputs are real recordings (refused if anything says SYNTHETIC)")
    ap.add_argument("--prefix", default=None, help="output file prefix (default rppg_validation[_SYNTHETIC])")
    args = ap.parse_args(argv)

    if args.demo_synthetic:
        if args.real_label:
            print("Refusing: --demo-synthetic output can never be labelled real.", file=sys.stderr)
            return 2
        args.rppg, args.watch = make_demo_inputs()
    if not (args.rppg and args.watch):
        ap.error("--rppg and --watch are required (or use --demo-synthetic)")

    synthetic = args.demo_synthetic or is_synthetic(args.rppg) or is_synthetic(args.watch)
    if synthetic and args.real_label:
        print("Refusing: an input file is marked SYNTHETIC; results cannot be labelled real.", file=sys.stderr)
        return 2

    df = align_1hz(load_series(args.rppg), load_series(args.watch), args.max_gap)
    m = metrics(df)
    if args.demo_synthetic:
        tag = SYNTH_TITLE
    elif synthetic:
        tag = "SYNTHETIC - NOT REAL DATA"
    else:
        tag = "FocusLoop rPPG validation"
    prefix = args.prefix or ("rppg_validation_SYNTHETIC_DEMO" if args.demo_synthetic
                             else "rppg_validation_SYNTHETIC" if synthetic else "rppg_validation")
    if synthetic and "SYNTHETIC" not in prefix.upper():
        prefix += "_SYNTHETIC"
    paths = plot(df, m, tag, prefix)
    summary = {"label": "SYNTHETIC" if synthetic else "REAL", "title": tag,
               "inputs": {"rppg": _repo_rel(args.rppg), "watch": _repo_rel(args.watch)}, **m}
    sp = REPORTS / f"{prefix}_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    paths.append(sp)
    print(f"[{summary['label']}] N={m['n_points_1hz']}  MAE={m['mae_bpm']:.2f}  RMSE={m['rmse_bpm']:.2f}  "
          f"bias={m['bias_bpm']:+.2f}  LoA=[{m['loa_low_bpm']:+.2f}, {m['loa_high_bpm']:+.2f}]  "
          f"r={m['pearson_r']:.3f}")
    for p in paths:
        print("wrote", p.relative_to(ML.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
