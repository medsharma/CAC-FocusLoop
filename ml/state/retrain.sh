#!/usr/bin/env bash
# One command: train -> eval -> Core ML export for the attention-state model.
#   ml/state/retrain.sh ml/data/            # your REAL exported CSVs (from the app's label export)
#   ml/state/retrain.sh ml/synth/out --synthetic
# Outputs: ml/state/runs/<TAG>/model.pt, ml/reports/state_<TAG>_*,
#          Resources/Models/AttentionStateModel.{mlpackage,json}
# <TAG> = SYNTHETIC if any CSV carries a "# SYNTHETIC" header (or --synthetic is passed), else REAL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/ml/.venv/bin/python}"
DATA="${1:?usage: ml/state/retrain.sh <dir-of-csvs> [--synthetic] [extra train.py args]}"
shift
SYN=()
EXTRA=()
for a in "$@"; do
  if [[ "$a" == "--synthetic" ]]; then SYN=(--synthetic); else EXTRA+=("$a"); fi
done
if [[ ${#SYN[@]} -eq 0 ]] && grep -rqs --include='*.csv' -m1 "^# SYNTHETIC" "$DATA"; then
  SYN=(--synthetic)
fi
if [[ ${#SYN[@]} -gt 0 ]]; then TAG=SYNTHETIC; else TAG=REAL; fi
RUN="$ROOT/ml/state/runs/$TAG"
echo "== [$TAG] train on $DATA"
"$PY" "$ROOT/ml/state/train.py" --data "$DATA" --out "$RUN" ${SYN[@]+"${SYN[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
echo "== [$TAG] eval"
"$PY" "$ROOT/ml/state/eval.py" --data "$DATA" --ckpt "$RUN/model.pt" ${SYN[@]+"${SYN[@]}"}
echo "== [$TAG] export Core ML"
"$PY" "$ROOT/ml/state/export_coreml.py" --ckpt "$RUN/model.pt" --data "$DATA" \
  --metrics "$ROOT/ml/reports/state_${TAG}_metrics.json"
echo "== done [$TAG]: $ROOT/Resources/Models/AttentionStateModel.mlpackage"
