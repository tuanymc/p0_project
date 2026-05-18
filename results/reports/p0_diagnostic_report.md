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

## graph_ablation_summary.csv
Source: `results\tables\graph_ablation_summary.csv`

## graph_stats.csv
Source: `results\tables\graph_stats.csv`

## junyi_dag_disruption.csv
Source: `results\tables\junyi_dag_disruption.csv`

## junyi_dag_disruption_summary.csv
Source: `results\tables\junyi_dag_disruption_summary.csv`

## leakage_metrics.csv
Source: `results\tables\leakage_metrics.csv`

## synthetic_c2_dag_disruption.csv
Source: `results\tables\synthetic_c2_dag_disruption.csv`

## synthetic_c2_dag_disruption_summary.csv
Source: `results\tables\synthetic_c2_dag_disruption_summary.csv`

## synthetic_c5_dag_disruption.csv
Source: `results\tables\synthetic_c5_dag_disruption.csv`

## synthetic_c5_dag_disruption_summary.csv
Source: `results\tables\synthetic_c5_dag_disruption_summary.csv`

## xes3g5m_dag_disruption.csv
Source: `results\tables\xes3g5m_dag_disruption.csv`

## xes3g5m_dag_disruption_summary.csv
Source: `results\tables\xes3g5m_dag_disruption_summary.csv`

## assist2012_dag_report.md
# DAG Audit Report: assist2012

## Fold 0
- source_edges: `data\processed\assist2012\fold_0\e_pre_train_only.csv`
- nodes: 108
- edges: 239
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\assist2012\fold_1\e_pre_train_only.csv`
- nodes: 109
- edges: 241
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\assist2012\fold_2\e_pre_train_only.csv`
- nodes: 106
- edges: 236
- cycles_before: 100
- cycles_after: 0
- topo_sort_passed: True


## cold_start_report.md
# Cold-Start KC Diagnostic

- assist2012 / akt: mean AUC=0.734, ACC=0.678, NLL=0.605, n=2,425,092
- assist2012 / bkt: mean AUC=0.694, ACC=0.667, NLL=0.616, n=2,425,092
- assist2012 / dgekt: mean AUC=0.735, ACC=0.671, NLL=0.610, n=2,425,092
- assist2012 / dkt: mean AUC=0.733, ACC=0.669, NLL=0.616, n=2,425,092
- assist2012 / dygkt: mean AUC=0.716, ACC=0.612, NLL=0.638, n=2,425,092
- assist2012 / gikt: mean AUC=0.736, ACC=0.674, NLL=0.607, n=2,425,092
- assist2012 / gkt: mean AUC=0.692, ACC=0.638, NLL=0.634, n=2,425,092
- assist2012 / simplekt: mean AUC=0.737, ACC=0.683, NLL=0.591, n=2,425,092
- assist2012 / skt: mean AUC=0.721, ACC=0.643, NLL=0.626, n=2,425,092
- junyi / akt: mean AUC=0.643, ACC=0.746, NLL=0.556, n=23,438,642
- junyi / bkt: mean AUC=0.643, ACC=0.744, NLL=0.531, n=23,438,642
- junyi / dkt: mean AUC=0.643, ACC=0.631, NLL=0.574, n=23,438,642
- junyi / gikt: mean AUC=0.641, ACC=0.640, NLL=0.565, n=23,438,642
- junyi / gkt: mean AUC=0.639, ACC=0.631, NLL=0.595, n=23,438,642
- junyi / simplekt: mean AUC=0.643, ACC=0.744, NLL=0.531, n=23,438,642
- synthetic_c2 / akt: mean AUC=0.619, ACC=0.687, NLL=0.608, n=180,000
- synthetic_c2 / bkt: mean AUC=0.554, ACC=0.687, NLL=0.616, n=180,000
- synthetic_c2 / dgekt: mean AUC=0.635, ACC=0.687, NLL=0.611, n=180,000
- synthetic_c2 / dkt: mean AUC=0.619, ACC=0.687, NLL=0.610, n=180,000
- synthetic_c2 / dygkt: mean AUC=0.632, ACC=0.687, NLL=0.619, n=180,000
- synthetic_c2 / gikt: mean AUC=0.635, ACC=0.687, NLL=0.609, n=180,000
- synthetic_c2 / gkt: mean AUC=0.554, ACC=0.687, NLL=0.621, n=180,000
- synthetic_c2 / simplekt: mean AUC=0.626, ACC=0.687, NLL=0.604, n=180,000
- synthetic_c2 / skt: mean AUC=0.633, ACC=0.687, NLL=0.618, n=180,000
- synthetic_c5 / akt: mean AUC=0.622, ACC=0.611, NLL=0.656, n=180,000
- synthetic_c5 / bkt: mean AUC=0.525, ACC=0.611, NLL=0.667, n=180,000
- synthetic_c5 / dgekt: mean AUC=0.633, ACC=0.611, NLL=0.656, n=180,000
- synthetic_c5 / dkt: mean AUC=0.628, ACC=0.611, NLL=0.658, n=180,000
- synthetic_c5 / dygkt: mean AUC=0.622, ACC=0.611, NLL=0.666, n=180,000
- synthetic_c5 / gikt: mean AUC=0.633, ACC=0.611, NLL=0.655, n=180,000
- synthetic_c5 / gkt: mean AUC=0.524, ACC=0.611, NLL=0.668, n=180,000
- synthetic_c5 / simplekt: mean AUC=0.630, ACC=0.611, NLL=0.651, n=180,000
- synthetic_c5 / skt: mean AUC=0.629, ACC=0.611, NLL=0.664, n=180,000
- xes3g5m / akt: mean AUC=0.747, ACC=0.765, NLL=0.484, n=5,765,996
- xes3g5m / bkt: mean AUC=0.713, ACC=0.782, NLL=0.469, n=5,765,996
- xes3g5m / dgekt: mean AUC=0.745, ACC=0.766, NLL=0.479, n=5,765,996
- xes3g5m / dkt: mean AUC=0.748, ACC=0.754, NLL=0.485, n=5,765,996
- xes3g5m / dygkt: mean AUC=0.728, ACC=0.753, NLL=0.503, n=5,765,996
- xes3g5m / gikt: mean AUC=0.747, ACC=0.766, NLL=0.476, n=5,765,996
- xes3g5m / gkt: mean AUC=0.708, ACC=0.762, NLL=0.496, n=5,765,996
- xes3g5m / simplekt: mean AUC=0.750, ACC=0.783, NLL=0.460, n=5,765,996
- xes3g5m / skt: mean AUC=0.731, ACC=0.762, NLL=0.490, n=5,765,996


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
- edges: 1129
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
- `results/tables/graph_ablation.tex`
- `results/tables/graph_ablation_summary.csv`
- `results/tables/graph_stats.csv`
- `results/tables/junyi_dag_disruption.csv`
- `results/tables/junyi_dag_disruption_summary.csv`
- `results/tables/leakage_audit.tex`
- `results/tables/leakage_metrics.csv`
- `results/tables/leakage_metrics.tex`
- `results/tables/synthetic_c2_dag_disruption.csv`
- `results/tables/synthetic_c2_dag_disruption_summary.csv`
- `results/tables/synthetic_c5_dag_disruption.csv`
- `results/tables/synthetic_c5_dag_disruption_summary.csv`
- `results/tables/xes3g5m_dag_disruption.csv`
- `results/tables/xes3g5m_dag_disruption_summary.csv`

## Figures
- `results/figures/fig_ddr_assist2012.pdf`
- `results/figures/fig_ddr_junyi.pdf`
- `results/figures/fig_ddr_synthetic_c2.pdf`
- `results/figures/fig_ddr_synthetic_c5.pdf`
- `results/figures/fig_ddr_xes3g5m.pdf`

## Reports
- `results/reports/assist2012_dag_pruning_log.csv`
- `results/reports/assist2012_dag_report.md`
- `results/reports/cold_start_report.md`
- `results/reports/junyi_dag_pruning_log.csv`
- `results/reports/junyi_dag_report.md`
- `results/reports/p0_diagnostic_report.md`
- `results/reports/synthetic_c2_dag_pruning_log.csv`
- `results/reports/synthetic_c2_dag_report.md`
- `results/reports/synthetic_c5_dag_report.md`
- `results/reports/xes3g5m_dag_pruning_log.csv`
- `results/reports/xes3g5m_dag_report.md`


## synthetic_c2_dag_report.md
# DAG Audit Report: synthetic_c2

## Fold 0
- source_edges: `data\processed\synthetic_c2\fold_0\e_pre_train_only.csv`
- nodes: 2
- edges: 1
- cycles_before: 1
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\synthetic_c2\fold_1\e_pre_train_only.csv`
- nodes: 2
- edges: 1
- cycles_before: 1
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\synthetic_c2\fold_2\e_pre_train_only.csv`
- nodes: 2
- edges: 1
- cycles_before: 1
- cycles_after: 0
- topo_sort_passed: True


## synthetic_c5_dag_report.md
# DAG Audit Report: synthetic_c5

## Fold 0
- source_edges: `data\processed\synthetic_c5\fold_0\e_pre_train_only.csv`
- nodes: 3
- edges: 2
- cycles_before: 0
- cycles_after: 0
- topo_sort_passed: True

## Fold 1
- source_edges: `data\processed\synthetic_c5\fold_1\e_pre_train_only.csv`
- nodes: 3
- edges: 2
- cycles_before: 0
- cycles_after: 0
- topo_sort_passed: True

## Fold 2
- source_edges: `data\processed\synthetic_c5\fold_2\e_pre_train_only.csv`
- nodes: 3
- edges: 2
- cycles_before: 0
- cycles_after: 0
- topo_sort_passed: True


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

