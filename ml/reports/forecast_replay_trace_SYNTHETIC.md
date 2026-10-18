# SYNTHETIC LapseForecaster risk trace on the bundled replay

**SYNTHETIC replay + MOCK content. Not a result.** `Resources/Replay/synthetic_session.replay.csv` fed second by second with MOCK `sample_lecture` content (playhead = session time), as in the Simulator demo. The replay's pre-lapse cues are scripted by `ml/synth/make_replay.py` from the same kind of assumptions the training generator makes, so this checks the demo, not real-world forecasting.

## with content (alert threshold 0.588)

Median risk while focused and not in the 15 s before a lapse: 0.159; alerts on 0% of those seconds.

| onset (s) | p @ -15 s | p @ -10 s | p @ -5 s | p @ -1 s | mean last 8 s | max before | lead (s) |
|---|---|---|---|---|---|---|---|
| 28 | 0.635 | 0.573 | 0.528 | 0.502 | 0.514 | 0.642 | 15 |
| 80 | 0.444 | 0.530 | 0.514 | 0.554 | 0.525 | 0.554 | no alert |
| 108 | 0.139 | 0.118 | 0.164 | 0.228 | 0.176 | 0.228 | no alert |
| 146 | 0.471 | 0.544 | 0.606 | 0.700 | 0.637 | 0.700 | 7 |

## without content (alert threshold 0.410)

Median risk while focused and not in the 15 s before a lapse: 0.243; alerts on 4% of those seconds.

| onset (s) | p @ -15 s | p @ -10 s | p @ -5 s | p @ -1 s | mean last 8 s | max before | lead (s) |
|---|---|---|---|---|---|---|---|
| 28 | 0.420 | 0.383 | 0.392 | 0.515 | 0.413 | 0.515 | 15 |
| 80 | 0.373 | 0.405 | 0.392 | 0.488 | 0.423 | 0.488 | 4 |
| 108 | 0.237 | 0.196 | 0.251 | 0.465 | 0.289 | 0.465 | 1 |
| 146 | 0.188 | 0.221 | 0.217 | 0.510 | 0.305 | 0.510 | 2 |

## previous with content (alert threshold 0.604)

Median risk while focused and not in the 15 s before a lapse: 0.560; alerts on 45% of those seconds.

| onset (s) | p @ -15 s | p @ -10 s | p @ -5 s | p @ -1 s | mean last 8 s | max before | lead (s) |
|---|---|---|---|---|---|---|---|
| 28 | 0.601 | 0.530 | 0.534 | 0.512 | 0.519 | 0.601 | no alert |
| 80 | 0.480 | 0.533 | 0.551 | 0.633 | 0.574 | 0.633 | 2 |
| 108 | 0.537 | 0.440 | 0.370 | 0.526 | 0.428 | 0.537 | no alert |
| 146 | 0.725 | 0.668 | 0.599 | 0.634 | 0.606 | 0.737 | 15 |

## previous without content (alert threshold 0.535)

Median risk while focused and not in the 15 s before a lapse: 0.505; alerts on 37% of those seconds.

| onset (s) | p @ -15 s | p @ -10 s | p @ -5 s | p @ -1 s | mean last 8 s | max before | lead (s) |
|---|---|---|---|---|---|---|---|
| 28 | 0.518 | 0.475 | 0.482 | 0.523 | 0.489 | 0.523 | no alert |
| 80 | 0.486 | 0.514 | 0.504 | 0.573 | 0.525 | 0.573 | 3 |
| 108 | 0.499 | 0.498 | 0.461 | 0.599 | 0.505 | 0.599 | 2 |
| 146 | 0.514 | 0.505 | 0.514 | 0.571 | 0.532 | 0.571 | 9 |

