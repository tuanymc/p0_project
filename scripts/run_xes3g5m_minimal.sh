#!/usr/bin/env bash
set -euo pipefail
CFG=configs/xes3g5m.yaml
python -m src.preprocess          --config "$CFG"
python -m src.split_checker       --config "$CFG"
python -m src.graph_builder       --config "$CFG"
python -m src.dag_audit           --config "$CFG"
python -m src.dag_disruption      --config "$CFG"
python -m src.baseline_runner     --config "$CFG"
python -m src.cold_start_report   --config "$CFG"
python -m src.report_generator    --out results/reports/
