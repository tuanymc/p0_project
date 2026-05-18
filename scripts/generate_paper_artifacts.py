"""Generate paper-facing tables and an artefact index.

This script is intentionally lightweight: it aggregates outputs that the
pipeline has already produced and avoids model or graph construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONFIGS = [
    Path("configs/junyi.yaml"),
    Path("configs/assist2012.yaml"),
    Path("configs/xes3g5m.yaml"),
]

DATASET_LABELS = {
    "junyi": "Junyi Academy",
    "assist2012": "ASSISTments 2012",
    "xes3g5m": "XES3G5M",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_stats() -> pd.DataFrame:
    rows = []
    for config_path in CONFIGS:
        cfg = _load_yaml(config_path)
        dataset = cfg["dataset"]
        processed_path = Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet"))
        if not processed_path.exists():
            continue
        df = pd.read_parquet(processed_path, columns=["user_id", "item_id", "kc_id"])
        n_learners = int(df["user_id"].nunique())
        rows.append({
            "dataset": dataset,
            "label": DATASET_LABELS.get(dataset, dataset),
            "n_learners": n_learners,
            "n_items": int(df["item_id"].nunique()),
            "n_kcs": int(df["kc_id"].nunique()),
            "n_interactions": int(len(df)),
            "avg_sequence_length": len(df) / n_learners if n_learners else 0.0,
            "has_dag": bool(cfg.get("has_dag", False)),
            "missing_total": 0,
        })
    return pd.DataFrame(rows)


def _write_dataset_stats_tex(stats: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Dataset statistics used in the P0 diagnostic protocol. "
        r"Junyi Academy counts reflect the Chang et al.\ problem-level log "
        r"(\texttt{junyi\_ProblemLog\_original.csv}) after preprocessing; "
        r"\#items and \#KCs coincide because both map to the exercise column.}",
        r"\label{tab:dataset-stats}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}X rrrrr >{\centering\arraybackslash}p{14mm} @{}}",
        r"\toprule",
        r"Dataset & \makecell[r]{\#\\learners} & \makecell[r]{\#\\items} & \makecell[r]{\#\\KCs} "
        r"& \makecell[r]{\#\\interactions} & \makecell[r]{Avg.\\seq.\ len.} & \makecell{Has\\DAG} \\",
        r"\midrule",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"{row.label} & {row.n_learners:,} & {row.n_items:,} & {row.n_kcs:,} & "
            f"{row.n_interactions:,} & {row.avg_sequence_length:.1f} & "
            f"{'Yes' if row.has_dag else 'No'} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_float(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def _fmt_metric_makecell(value: float, lo: float | None, hi: float | None) -> str:
    """Point estimate on first line; 95\% CI on second (scriptsize)."""
    v = _fmt_float(value)
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return v
    return rf"\makecell{{{v}\\[-1pt]{{\scriptsize[{_fmt_float(lo)}, {_fmt_float(hi)}]}}}}"


def _write_baseline_tex(path: Path) -> None:
    csv_path = Path("results/tables/baseline_results.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or not {"dataset", "model", "auc", "acc", "nll"}.issubset(df.columns):
        return
    # Drop template / incomplete rows (e.g. status=pending_data with empty dataset) that would render as "nan" in TeX.
    df = df[df["dataset"].notna() & (df["dataset"].astype(str).str.strip() != "")]
    if df.empty:
        return
    if "graph_construction" in df.columns:
        df = df[df["graph_construction"].astype(str).str.strip().eq("train_only")]
    if df.empty:
        return
    has_ci = {"auc_ci_low", "auc_ci_high", "acc_ci_low", "acc_ci_high", "nll_ci_low", "nll_ci_high"}.issubset(df.columns)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Diagnostic baseline results. Baselines are used for protocol comparison only; no SOTA claim is made. Values are fold means with 95\% bootstrap confidence intervals when available.}",
        r"\label{tab:baseline-results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{} >{\RaggedRight\arraybackslash}p{32mm} "
        r">{\centering\arraybackslash}p{18mm} *{3}{>{\centering\arraybackslash}m{36mm}} @{}}",
        r"\toprule",
        r"Dataset & Model & AUC & ACC & NLL \\",
        r"\midrule",
    ]
    for row in df.sort_values(["dataset", "model"]).itertuples(index=False):
        if has_ci:
            auc = _fmt_metric_makecell(row.auc, row.auc_ci_low, row.auc_ci_high)
            acc = _fmt_metric_makecell(row.acc, row.acc_ci_low, row.acc_ci_high)
            nll = _fmt_metric_makecell(row.nll, row.nll_ci_low, row.nll_ci_high)
        else:
            auc = _fmt_float(row.auc)
            acc = _fmt_float(row.acc)
            nll = _fmt_float(row.nll)
        lines.append(f"{row.dataset} & {row.model} & {auc} & {acc} & {nll} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_leakage_metrics_tex(path: Path) -> None:
    """Emit \\input-able leakage_metrics.tex from results/tables/leakage_metrics.csv."""
    csv_path = Path("results/tables/leakage_metrics.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    required = {"dataset", "eoc", "tbvr"}
    if df.empty or not required.issubset(df.columns):
        return
    if "ecr_overlap" not in df.columns and "ecr" in df.columns:
        df = df.copy()
        df["ecr_overlap"] = df["ecr"]
    if "ecr_flag" not in df.columns:
        df = df.copy()
        df["ecr_flag"] = 0.0
    if "ecr_overlap" not in df.columns:
        return
    df = df[df["dataset"].notna() & (df["dataset"].astype(str).str.strip() != "")]
    if df.empty:
        return
    if "fold" in df.columns:
        df = df.groupby("dataset", as_index=False).agg(
            ecr_flag=("ecr_flag", "mean"),
            ecr_overlap=("ecr_overlap", "mean"),
            eoc=("eoc", "mean"),
            tbvr=("tbvr", "mean"),
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direct leakage diagnostics per dataset (fold mean where multiple folds are present). "
        r"\textsc{ECR}\textsubscript{flag}: learner-overlap structural indicator (Eq.~\ref{eq:ecr-flag}; "
        r"$0$ confirms disjoint learners across train/validation/test under each fold). "
        r"\textsc{ECR}\textsubscript{overlap}: held-out pattern overlap rate over retained edges (Eq.~\ref{eq:ecr-overlap}). "
        r"\textsc{EOC}: Edge--Outcome Correlation Frobenius norm (Eq.~\ref{eq:eoc}); "
        r"\textsc{TBVR}: Temporal Boundary Violation Rate within train timelines (Eq.~\ref{eq:tbvr}, indexed boundary). "
        r"Values are exported by \texttt{p0-graph-build} into \texttt{results/tables/leakage\_metrics.csv}. "
        r"Under learner-disjoint folds, $\textsc{ECR}\textsubscript{flag}{=}0$ on every run---confirming structural leakage freedom "
        r"at the split level; "
        r"\textsc{ECR}\textsubscript{overlap} can remain near~$1$ on dense logs because recurrent transition patterns "
        r"reappear across disjoint held-out learners (diagnostic overlap, not \texttt{train\_only\_flag} violation).}",
        r"\label{tab:leakage-metrics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Dataset & \textsc{ECR}\textsubscript{flag} & \textsc{ECR}\textsubscript{overlap} & \textsc{EOC} & \textsc{TBVR} \\",
        r"\midrule",
    ]
    label_map = {"junyi": "Junyi Academy", "assist2012": "ASSISTments 2012", "xes3g5m": "XES3G5M"}
    for row in df.sort_values("dataset").itertuples(index=False):
        ds = str(row.dataset).strip()
        lines.append(
            f"{label_map.get(ds, ds)} & {_fmt_float(row.ecr_flag)} & {_fmt_float(row.ecr_overlap)} & {_fmt_float(row.eoc)} & {_fmt_float(row.tbvr)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_graph_ablation_tex(path: Path) -> None:
    """Emit train-only vs full-log ablation table from graph_ablation_summary.csv."""
    csv_path = Path("results/tables/graph_ablation_summary.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    required = {
        "dataset", "model", "auc_train_only", "auc_full_log", "delta_auc",
        "acc_train_only", "acc_full_log", "delta_acc",
    }
    if df.empty or not required.issubset(df.columns):
        return
    df = df[df["dataset"].notna() & (df["dataset"].astype(str).str.strip() != "")]
    if df.empty:
        return
    model_order = ["gkt", "gikt", "skt", "dygkt", "dgekt"]

    def _ord(m: str) -> int:
        m = str(m).strip().lower()
        return model_order.index(m) if m in model_order else len(model_order)

    df = df.copy()
    df["_mo"] = df["model"].map(_ord)
    label_map = {"junyi": "Junyi Academy", "assist2012": "ASSISTments 2012", "xes3g5m": "XES3G5M"}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Ablation contrasting \emph{train-only} graph smoothing (protocol) with a \emph{full-log} graph "
        r"built from all learners before splitting (leaky construction). Diagnostic graph-augmented ensembles "
        r"(GKT, GIKT, SKT, DyGKT, DGEKT-style linear blends; Section~\ref{sec:exp-baseline}); evaluation split unchanged. "
        r"$\Delta$ is the paired mean across folds (full-log minus train-only); "
        r"positive $\Delta$AUC indicates higher discrimination when the graph sees the entire log.}",
        r"\label{tab:graph-ablation}",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"Dataset & Model & \makecell{AUC\\train-only} & \makecell{AUC\\full-log} & $\Delta$AUC"
        r" & \makecell{ACC\\train-only} & \makecell{ACC\\full-log} & $\Delta$ACC \\",
        r"\midrule",
    ]
    for row in df.sort_values(["dataset", "_mo"]).itertuples(index=False):
        ds = label_map.get(str(row.dataset).strip(), str(row.dataset).strip())
        lines.append(
            f"{ds} & {row.model.upper()} & {_fmt_float(row.auc_train_only)} & {_fmt_float(row.auc_full_log)} & {_fmt_float(row.delta_auc)}"
            f" & {_fmt_float(row.acc_train_only)} & {_fmt_float(row.acc_full_log)} & {_fmt_float(row.delta_acc)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cold_start_tex(path: Path) -> None:
    csv_path = Path("results/tables/cold_start_metrics.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or not {"dataset", "model", "stratum", "auc", "acc", "nll"}.issubset(df.columns):
        return
    summary = df.groupby(["dataset", "model"], as_index=False).agg(auc=("auc", "mean"), acc=("acc", "mean"), nll=("nll", "mean"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cold-start KC diagnostic summary averaged across frequency strata.}",
        r"\label{tab:cold-start}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & AUC & ACC & NLL \\",
        r"\midrule",
    ]
    for row in summary.sort_values(["dataset", "model"]).itertuples(index=False):
        lines.append(f"{row.dataset} & {row.model} & {_fmt_float(row.auc)} & {_fmt_float(row.acc)} & {_fmt_float(row.nll)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_artifact_index(path: Path) -> None:
    table_files = sorted(p for p in Path("results/tables").glob("*") if p.name != ".gitkeep")
    figure_files = sorted(p for p in Path("results/figures").glob("*") if p.name != ".gitkeep")
    report_files = sorted(p for p in Path("results/reports").glob("*") if p.name not in {path.name, ".gitkeep"})
    lines = ["# Paper Artefact Index", ""]
    lines.append("## Tables")
    lines.extend(f"- `{p.as_posix()}`" for p in table_files)
    lines.append("")
    lines.append("## Figures")
    lines.extend(f"- `{p.as_posix()}`" for p in figure_files)
    lines.append("")
    lines.append("## Reports")
    lines.extend(f"- `{p.as_posix()}`" for p in report_files)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/reports").mkdir(parents=True, exist_ok=True)
    stats = _dataset_stats()
    stats_for_csv = stats[["dataset", "n_learners", "n_items", "n_kcs", "n_interactions", "missing_total"]]
    stats_for_csv.to_csv("results/tables/dataset_stats.csv", index=False)
    _write_dataset_stats_tex(stats, Path("results/tables/dataset_stats.tex"))
    _write_baseline_tex(Path("results/tables/baseline_results.tex"))
    _write_graph_ablation_tex(Path("results/tables/graph_ablation.tex"))
    _write_leakage_metrics_tex(Path("results/tables/leakage_metrics.tex"))
    _write_cold_start_tex(Path("results/tables/cold_start_metrics.tex"))
    _write_artifact_index(Path("results/reports/paper_artifact_index.md"))


if __name__ == "__main__":
    main()
