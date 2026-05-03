#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -m src.dag_disruption --config configs/junyi.yaml
"$PYTHON_BIN" -m src.dag_disruption --config configs/assist2012.yaml
"$PYTHON_BIN" -m src.dag_disruption --config configs/xes3g5m.yaml
