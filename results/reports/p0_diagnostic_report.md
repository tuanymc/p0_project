# P0 Diagnostic Report

## assist2012_dag_disruption.csv
Source: `results\tables\assist2012_dag_disruption.csv`

## assist2012_dag_disruption_summary.csv
Source: `results\tables\assist2012_dag_disruption_summary.csv`

## baseline_fold_results.csv
Source: `results\tables\baseline_fold_results.csv`

## baseline_results.csv
Source: `results\tables\baseline_results.csv`

## cold_start_metrics.csv
Source: `results\tables\cold_start_metrics.csv`

## dag_audit_summary.csv
Source: `results\tables\dag_audit_summary.csv`

## dag_disruption.csv
Source: `results\tables\dag_disruption.csv`

## dag_disruption_summary.csv
Source: `results\tables\dag_disruption_summary.csv`

## dataset_stats.csv
Source: `results\tables\dataset_stats.csv`

## graph_stats.csv
Source: `results\tables\graph_stats.csv`

## junyi_dag_disruption.csv
Source: `results\tables\junyi_dag_disruption.csv`

## junyi_dag_disruption_summary.csv
Source: `results\tables\junyi_dag_disruption_summary.csv`

## xes3g5m_dag_disruption.csv
Source: `results\tables\xes3g5m_dag_disruption.csv`

## xes3g5m_dag_disruption_summary.csv
Source: `results\tables\xes3g5m_dag_disruption_summary.csv`

## assist2012_dag_report.md
# DAG Audit Report: assist2012

## Fold 0
- source_edges: `data\processed\assist2012\fold_0\e_pre_train_only.csv`
- nodes: 139
- edges: 261
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\assist2012\fold_1\e_pre_train_only.csv`
- nodes: 141
- edges: 264
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\assist2012\fold_2\e_pre_train_only.csv`
- nodes: 140
- edges: 258
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True


## cold_start_report.md
# Cold-Start KC Diagnostic

- assist2012 / akt: mean AUC=0.734, ACC=0.678, NLL=0.605, n=2,425,092
- assist2012 / bkt: mean AUC=0.694, ACC=0.667, NLL=0.616, n=2,425,092
- assist2012 / dkt: mean AUC=0.733, ACC=0.669, NLL=0.616, n=2,425,092
- assist2012 / gikt: mean AUC=0.733, ACC=0.675, NLL=0.605, n=2,425,092
- assist2012 / gkt: mean AUC=0.688, ACC=0.645, NLL=0.631, n=2,425,092
- assist2012 / simplekt: mean AUC=0.737, ACC=0.683, NLL=0.591, n=2,425,092
- junyi / akt: mean AUC=0.643, ACC=0.746, NLL=0.556, n=23,438,642
- junyi / bkt: mean AUC=0.643, ACC=0.744, NLL=0.531, n=23,438,642
- junyi / dkt: mean AUC=0.643, ACC=0.631, NLL=0.574, n=23,438,642
- junyi / gikt: mean AUC=0.641, ACC=0.640, NLL=0.565, n=23,438,642
- junyi / gkt: mean AUC=0.639, ACC=0.631, NLL=0.595, n=23,438,642
- junyi / simplekt: mean AUC=0.643, ACC=0.744, NLL=0.531, n=23,438,642
- xes3g5m / akt: mean AUC=0.747, ACC=0.765, NLL=0.484, n=5,765,996
- xes3g5m / bkt: mean AUC=0.713, ACC=0.782, NLL=0.469, n=5,765,996
- xes3g5m / dkt: mean AUC=0.748, ACC=0.754, NLL=0.485, n=5,765,996
- xes3g5m / gikt: mean AUC=0.747, ACC=0.766, NLL=0.476, n=5,765,996
- xes3g5m / gkt: mean AUC=0.708, ACC=0.762, NLL=0.496, n=5,765,996
- xes3g5m / simplekt: mean AUC=0.750, ACC=0.783, NLL=0.460, n=5,765,996


## junyi_dag_report.md
# DAG Audit Report: junyi

## Fold 0
- source_edges: `data\processed\junyi\fold_0\e_pre_train_only.csv`
- nodes: 573
- edges: 1139
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\junyi\fold_1\e_pre_train_only.csv`
- nodes: 578
- edges: 1147
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\junyi\fold_2\e_pre_train_only.csv`
- nodes: 576
- edges: 1130
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True


## paper_artifact_index.md
# Paper Artefact Index

## Tables
- `results/tables/assist2012_dag_disruption.csv`
- `results/tables/assist2012_dag_disruption_summary.csv`
- `results/tables/baseline_fold_results.csv`
- `results/tables/baseline_results.csv`
- `results/tables/baseline_results.tex`
- `results/tables/cold_start_by_stratum.tex`
- `results/tables/cold_start_metrics.csv`
- `results/tables/cold_start_metrics.tex`
- `results/tables/dag_audit_summary.csv`
- `results/tables/dag_audit_summary.tex`
- `results/tables/dag_disruption.csv`
- `results/tables/dag_disruption_summary.csv`
- `results/tables/dataset_stats.csv`
- `results/tables/dataset_stats.tex`
- `results/tables/ddr_summary.tex`
- `results/tables/graph_stats.csv`
- `results/tables/junyi_dag_disruption.csv`
- `results/tables/junyi_dag_disruption_summary.csv`
- `results/tables/leakage_audit.tex`
- `results/tables/xes3g5m_dag_disruption.csv`
- `results/tables/xes3g5m_dag_disruption_summary.csv`

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

## Fold 0
- source_edges: `data\processed\xes3g5m\fold_0\e_pre_train_only.csv`
- nodes: 455
- edges: 701
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\xes3g5m\fold_1\e_pre_train_only.csv`
- nodes: 453
- edges: 700
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\xes3g5m\fold_2\e_pre_train_only.csv`
- nodes: 455
- edges: 707
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

