#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
CONFIGS=(
  "configs/junyi.yaml"
  "configs/assist2012.yaml"
  "configs/xes3g5m.yaml"
)
PROCESSED=(
  "data/processed/junyi.parquet"
  "data/processed/assist2012.parquet"
  "data/processed/xes3g5m.parquet"
)

for IDX in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$IDX]}"
  PROCESSED_PATH="${PROCESSED[$IDX]}"
  echo "==> Running full P0 pipeline for ${CFG}"
  if [[ -f "$PROCESSED_PATH" && "${FORCE_PREPROCESS:-0}" != "1" ]]; then
    echo "    Skipping preprocess; found ${PROCESSED_PATH}"
  else
    "$PYTHON_BIN" -m src.preprocess --config "$CFG"
  fi
  "$PYTHON_BIN" -m src.split_checker --config "$CFG"
  "$PYTHON_BIN" -m src.graph_builder --config "$CFG"
  "$PYTHON_BIN" -m src.dag_audit --config "$CFG"
  "$PYTHON_BIN" -m src.dag_disruption --config "$CFG"
  "$PYTHON_BIN" -m src.baseline_runner --config "$CFG"
  "$PYTHON_BIN" -m src.cold_start_report --config "$CFG"
done

echo "==> Generating paper artefacts"
"$PYTHON_BIN" scripts/generate_paper_artifacts.py
"$PYTHON_BIN" -m src.report_generator --out results/reports/

echo "==> Done. Main report: results/reports/p0_diagnostic_report.md"
