# P0 Diagnostic Report

## assist2012_dag_disruption.csv
Source: `results\tables\assist2012_dag_disruption.csv`

## baseline_results.csv
Source: `results\tables\baseline_results.csv`

## cold_start_metrics.csv
Source: `results\tables\cold_start_metrics.csv`

## dag_audit_summary.csv
Source: `results\tables\dag_audit_summary.csv`

## dag_disruption.csv
Source: `results\tables\dag_disruption.csv`

## dataset_stats.csv
Source: `results\tables\dataset_stats.csv`

## graph_stats.csv
Source: `results\tables\graph_stats.csv`

## junyi_dag_disruption.csv
Source: `results\tables\junyi_dag_disruption.csv`

## xes3g5m_dag_disruption.csv
Source: `results\tables\xes3g5m_dag_disruption.csv`

## assist2012_dag_report.md
# DAG Audit Report: assist2012

- source_edges: `data\processed\assist2012\fold_0\e_pre_train_only.csv`
- nodes: 139
- edges: 261
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True


## cold_start_report.md
# Cold-Start KC Diagnostic

- assist2012 / akt: mean AUC=0.709, ACC=0.710, NLL=0.605, n=816,951
- assist2012 / bkt: mean AUC=0.662, ACC=0.662, NLL=0.629, n=816,951
- assist2012 / dkt: mean AUC=0.709, ACC=0.699, NLL=0.612, n=816,951
- assist2012 / gikt: mean AUC=0.705, ACC=0.708, NLL=0.600, n=816,951
- assist2012 / gkt: mean AUC=0.658, ACC=0.653, NLL=0.629, n=816,951
- assist2012 / simplekt: mean AUC=0.709, ACC=0.716, NLL=0.589, n=816,951
- junyi / akt: mean AUC=0.697, ACC=0.659, NLL=0.606, n=4,837,977
- junyi / bkt: mean AUC=0.650, ACC=0.655, NLL=0.615, n=4,837,977
- junyi / dkt: mean AUC=0.694, ACC=0.639, NLL=0.612, n=4,837,977
- junyi / gikt: mean AUC=0.696, ACC=0.650, NLL=0.603, n=4,837,977
- junyi / gkt: mean AUC=0.646, ACC=0.637, NLL=0.625, n=4,837,977
- junyi / simplekt: mean AUC=0.700, ACC=0.672, NLL=0.594, n=4,837,977
- xes3g5m / akt: mean AUC=0.745, ACC=0.773, NLL=0.501, n=1,922,838
- xes3g5m / bkt: mean AUC=0.698, ACC=0.774, NLL=0.491, n=1,922,838
- xes3g5m / dkt: mean AUC=0.745, ACC=0.757, NLL=0.500, n=1,922,838
- xes3g5m / gikt: mean AUC=0.745, ACC=0.768, NLL=0.492, n=1,922,838
- xes3g5m / gkt: mean AUC=0.693, ACC=0.763, NLL=0.508, n=1,922,838
- xes3g5m / simplekt: mean AUC=0.748, ACC=0.775, NLL=0.482, n=1,922,838


## junyi_dag_report.md
# DAG Audit Report: junyi

- source_edges: `data\processed\junyi\fold_0\e_pre_train_only.csv`
- nodes: 1249
- edges: 2320
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True


## paper_artifact_index.md
# Paper Artefact Index

## Tables
- `results/tables/assist2012_dag_disruption.csv`
- `results/tables/baseline_results.csv`
- `results/tables/baseline_results.tex`
- `results/tables/cold_start_metrics.csv`
- `results/tables/cold_start_metrics.tex`
- `results/tables/dag_audit_summary.csv`
- `results/tables/dag_audit_summary.tex`
- `results/tables/dag_disruption.csv`
- `results/tables/dataset_stats.csv`
- `results/tables/dataset_stats.tex`
- `results/tables/ddr_summary.tex`
- `results/tables/graph_stats.csv`
- `results/tables/junyi_dag_disruption.csv`
- `results/tables/leakage_audit.tex`
- `results/tables/xes3g5m_dag_disruption.csv`

## Figures
- `results/figures/fig_ddr_assist2012.pdf`
- `results/figures/fig_ddr_junyi.pdf`
- `results/figures/fig_ddr_xes3g5m.pdf`

## Reports
- `results/reports/assist2012_dag_pruning_log.csv`
- `results/reports/assist2012_dag_report.md`
- `results/reports/cold_start_report.md`
- `results/reports/junyi_dag_pruning_log.csv`
- `results/reports/junyi_dag_report.md`
- `results/reports/p0_diagnostic_report.md`
- `results/reports/xes3g5m_dag_pruning_log.csv`
- `results/reports/xes3g5m_dag_report.md`


## xes3g5m_dag_report.md
# DAG Audit Report: xes3g5m

- source_edges: `data\processed\xes3g5m\fold_0\e_pre_train_only.csv`
- nodes: 455
- edges: 701
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

