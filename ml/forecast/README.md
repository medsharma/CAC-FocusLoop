# LapseForecaster pipeline (`ml/forecast/`)

Predicts P(a lapse begins within the next 15 s) from the previous 20 s of 10 Hz `FeatureFrame`s plus,
during lectures, the previous 20 s of brain-content features (predicted brain response to content,
average brain). Label rules: `label_shift.py` docstring. Metric definitions: `metrics.py` docstring.

## One-command steps

```
PY=/path/to/ml/.venv/bin/python
$PY ml/forecast/synth_forecast.py          # SYNTHETIC corpus + MOCK lectures -> ml/forecast/out/ (gitignored)
$PY ml/forecast/train.py                   # both variants; --no-content / --content-only-variant for one
$PY ml/forecast/eval.py                    # AUROC, precision @ 10% alert rate, lead time (test subjects)
$PY ml/forecast/ablation.py                # with vs without content -> ml/reports/forecast_ablation[_SYNTHETIC].{md,json,png}
$PY ml/forecast/replay_trace.py            # risk trace on the bundled replay -> ml/reports/forecast_replay_trace_SYNTHETIC.*
$PY ml/forecast/export_coreml.py           # -> Resources/Models/LapseForecaster{,NoContent}.mlpackage + .json
$PY -m pytest ml/tests/test_forecast_labels.py
```

Phase 4 (shipped SYNTHETIC models) used:

```
$PY ml/forecast/synth_forecast.py --subjects 144                                    # realistic generator
$PY ml/forecast/synth_forecast.py --generator legacy --subjects 40 --seed 12 --out ml/forecast/out/legacy
$PY ml/forecast/ablation.py --seeds 0 1 2 --extra-manifest ml/forecast/out/legacy/manifest.csv \
    --previous-dir <dir with the pre-Phase-4 LapseForecaster*.mlpackage>
$PY ml/forecast/replay_trace.py --models ml/forecast/out/models/seed0
$PY ml/forecast/export_coreml.py --models ml/forecast/out/models/seed0
```

`--extra-manifest` sessions are train-only (never validation/test).

For real data, pass `--manifest ml/data/forecast/manifest.csv` to `train.py`, `eval.py` and `ablation.py`
(`ml/data/` is gitignored). Outputs are then named without `_SYNTHETIC`.

## Real-data inputs

**Manifest** (CSV, paths relative to the manifest file; empty cell = not available):

| column | meaning |
|---|---|
| `subject` | participant id; the train/val/test split is by subject |
| `session_id` | unique name |
| `session_csv` | 10 Hz feature CSV (below) |
| `sart_csv` | SART trials recorded during this session (SART calibration sessions), else empty |
| `es_csv` | experience-sampling answers for this session, else empty |
| `lecture_json` | `<lecture>.brain.json` that was playing (Study Mode), else empty (free study / SART) |
| `playhead_offset` | seconds added to `t` to get the lecture playhead if the CSV has no `playhead` column |
| `synthetic` | `1` for synthetic data, `0` for real |

**Session CSV**: one row per 10 Hz frame, `#` comment lines allowed. Required: `t` (s since session
start) and the 42 features named exactly as in `ml/schema/feature_schema.json`. Optional: `label`
(`focused`/`distracted`/`drowsy`/`stressed`) or `lapse` (0/1) for labelled windows, and `playhead`
(lecture seconds). This is the app's labelled-window CSV export with those column names.

**SART CSV**: `t,digit,responded,rt` — stimulus onset in session seconds, digit 1–9 (3 = no-go),
1 if the user tapped, RT in seconds (empty when no tap).

**Experience-sampling CSV**: `t,focused,reason` — answer time, 1/0, optional `tired`/`distracted`/`stressed`.

**Content timeline**: the bundled `<lecture>.brain.json` (`ml/schema/brain_schema.json`). Content
features are computed exactly like `BrainContentTimeline.features(at:)` in Swift (`data.content_features`).

## Notes

- Normalisation (train-set mean/std per input) is baked into the network, so the Core ML models
  take raw values. Short windows are front-padded by repeating the oldest frame (same in Swift).
- Everything produced from the synthetic corpus is SYNTHETIC and labelled so in filenames, plot titles,
  JSON and model metadata. Do not report those numbers as results.
