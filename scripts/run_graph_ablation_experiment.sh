#!/usr/bin/env bash
# Ablation: train-only graphs vs full-log graphs + baselines (GKT/GIKT).
# Usage (from repo root):
#   chmod +x scripts/run_graph_ablation_experiment.sh
#   ./scripts/run_graph_ablation_experiment.sh
#
# Env:
#   PYTHON=python3   override interpreter
#   SERVER_PROFILE=1 set OMP/MKL/OpenBLAS threads to 8 if unset
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

if [[ "${SERVER_PROFILE:-}" == "1" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
  echo "[graph_ablation] SERVER_PROFILE threads=$OMP_NUM_THREADS"
fi

run() {
  echo ">> $PY $*"
  "$PY" "$@"
}

CONFIGS=(
  "configs/junyi.yaml"
  "configs/assist2012.yaml"
  "configs/xes3g5m.yaml"
)

# Ước lượng tiến trình theo bước (graph_builder / export / baseline × 3 dataset + artefacts).
step_total=0
for cfg in "${CONFIGS[@]}"; do
  step_total=$((step_total + 3))
done
step_total=$((step_total + 1))
step_i=0

for cfg in "${CONFIGS[@]}"; do
  step_i=$((step_i + 1))
  pct=$((100 * (step_i - 1) / step_total))
  echo ""
  echo "========== [$step_i/$step_total ~${pct}%] graph_builder: $cfg =========="
  run -m src.graph_builder --config "$cfg"
  step_i=$((step_i + 1))
  pct=$((100 * (step_i - 1) / step_total))
  echo "========== [$step_i/$step_total ~${pct}%] export_full_log_graph: $cfg =========="
  run -m src.export_full_log_graph --config "$cfg"
  step_i=$((step_i + 1))
  pct=$((100 * (step_i - 1) / step_total))
  echo "========== [$step_i/$step_total ~${pct}%] baseline_runner: $cfg =========="
  if [[ "$cfg" == *"junyi"* ]]; then
    echo "[junyi] baseline with --skip-cold-start"
    run -m src.baseline_runner --config "$cfg" --skip-cold-start
  else
    run -m src.baseline_runner --config "$cfg"
  fi
done

step_i=$((step_i + 1))
pct=$((100 * (step_i - 1) / step_total))
echo ""
echo "========== [$step_i/$step_total ~${pct}%] generate_paper_artifacts =========="
run scripts/generate_paper_artifacts.py

echo ""
echo "[OK] See results/tables/graph_ablation_summary.csv and graph_ablation.tex"
