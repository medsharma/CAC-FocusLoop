#!/usr/bin/env bash
# Runs every Python test suite in its own process. ml/state and ml/forecast both have modules named
# data.py / model.py that they put on sys.path, so they must not share one interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-ml/.venv/bin/python}"
status=0
for t in ml/tests/test_*.py ml/rppg_reference/test_*.py; do
  echo "== $t"
  "$PY" -m pytest -q "$t" || status=1
done
exit $status
