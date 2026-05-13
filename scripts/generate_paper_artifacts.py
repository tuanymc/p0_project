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
        r"\caption{Dataset statistics used in the P0 diagnostic protocol. "
        r"Junyi Academy counts reflect the Chang et al.\ problem-level log "
        r"(\texttt{junyi\_ProblemLog\_original.csv}) after preprocessing; "
        r"\#items and \#KCs coincide because both map to the exercise column.}",
        r"\label{tab:dataset-stats}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Dataset & \#learners & \#items & \#KCs & \#interactions & Avg. seq. length & Has DAG \\",
        r"\midrule",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"{row.label} & {row.n_learners:,} & {row.n_items:,} & {row.n_kcs:,} & "
            f"{row.n_interactions:,} & {row.avg_sequence_length:.1f} & "
            f"{'Yes' if row.has_dag else 'No'} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_float(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


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
    has_ci = {"auc_ci_low", "auc_ci_high", "acc_ci_low", "acc_ci_high", "nll_ci_low", "nll_ci_high"}.issubset(df.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostic baseline results. Baselines are used for protocol comparison only; no SOTA claim is made. Values are fold means with 95\% bootstrap confidence intervals when available.}",
        r"\label{tab:baseline-results}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & AUC & ACC & NLL \\",
        r"\midrule",
    ]
    for row in df.sort_values(["dataset", "model"]).itertuples(index=False):
        if has_ci:
            auc = f"{_fmt_float(row.auc)} [{_fmt_float(row.auc_ci_low)}, {_fmt_float(row.auc_ci_high)}]"
            acc = f"{_fmt_float(row.acc)} [{_fmt_float(row.acc_ci_low)}, {_fmt_float(row.acc_ci_high)}]"
            nll = f"{_fmt_float(row.nll)} [{_fmt_float(row.nll_ci_low)}, {_fmt_float(row.nll_ci_high)}]"
        else:
            auc = _fmt_float(row.auc)
            acc = _fmt_float(row.acc)
            nll = _fmt_float(row.nll)
        lines.append(f"{row.dataset} & {row.model} & {auc} & {acc} & {nll} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
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
    _write_cold_start_tex(Path("results/tables/cold_start_metrics.tex"))
    _write_artifact_index(Path("results/reports/paper_artifact_index.md"))


if __name__ == "__main__":
    main()
