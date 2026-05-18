#!/usr/bin/env bash
# Full P0 pipeline on Linux servers (same stages as run_all_datasets_full.ps1).
#
# Usage:
#   chmod +x scripts/run_all_datasets_full.sh
#   ./scripts/run_all_datasets_full.sh --server              # ~32GB RAM friendly thread caps
#   ./scripts/run_all_datasets_full.sh --force-full --server # rebuild all parquet from raw
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_all_datasets_full.sh --server
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
FORCE_FULL=0
SERVER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-full) FORCE_FULL=1 ;;
    --server) SERVER=1 ;;
    -h|--help)
      echo "Usage: $0 [--force-full] [--server]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "$FORCE_FULL" -eq 1 ]]; then
  export FORCE_PREPROCESS=1
  echo "[run_all_datasets_full] FORCE_PREPROCESS=1"
fi

if [[ "$SERVER" -eq 1 ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  echo "[run_all_datasets_full] Server profile: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"
fi

echo "[run_all_datasets_full] PYTHON=$PYTHON FORCE_PREPROCESS=${FORCE_PREPROCESS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

run_step() {
  "$PYTHON" "$@"
}

DATASETS=(
  "configs/junyi.yaml:data/processed/junyi.parquet"
  "configs/assist2012.yaml:data/processed/assist2012.parquet"
  "configs/xes3g5m.yaml:data/processed/xes3g5m.parquet"
  "configs/synthetic_c2.yaml:data/processed/synthetic_c2.parquet"
  "configs/synthetic_c5.yaml:data/processed/synthetic_c5.parquet"
)

for entry in "${DATASETS[@]}"; do
  CFG="${entry%%:*}"
  PROC="${entry##*:}"
  echo "==> Running full P0 pipeline for $CFG"
  if [[ -f "$PROC" && "${FORCE_PREPROCESS:-}" != "1" ]]; then
    echo "    Skipping preprocess; found $PROC"
  else
    run_step -m src.preprocess --config "$CFG"
  fi
  run_step -m src.split_checker --config "$CFG"
  run_step -m src.graph_builder --config "$CFG"
  run_step -m src.export_full_log_graph --config "$CFG"
  run_step -m src.dag_audit --config "$CFG"
  run_step -m src.dag_disruption --config "$CFG"
  run_step -m src.baseline_runner --config "$CFG"
  run_step -m src.cold_start_report --config "$CFG"
done

echo "==> Generating paper artefacts"
run_step scripts/generate_paper_artifacts.py
run_step -m src.report_generator --out results/reports/

echo "==> Done. Main report: results/reports/p0_diagnostic_report.md"
echo "    Optional Junyi GT CV: python scripts/run_gt_cross_validation_junyi.py"
