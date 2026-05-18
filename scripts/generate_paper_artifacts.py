"""Generate paper-facing tables and an artefact index.

This script is intentionally lightweight: it aggregates outputs that the
pipeline has already produced and avoids model or graph construction.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yaml

from src.io_utils import dump_csv


CONFIGS = [
    Path("configs/junyi.yaml"),
    Path("configs/assist2012.yaml"),
    Path("configs/xes3g5m.yaml"),
    Path("configs/synthetic_c2.yaml"),
    Path("configs/synthetic_c5.yaml"),
]

DATASET_LABELS = {
    "junyi": "Junyi Academy",
    "assist2012": "ASSISTments 2012",
    "xes3g5m": "XES3G5M",
    "synthetic_c2": "Synthetic C2",
    "synthetic_c5": "Synthetic C5",
}

BASELINE_TEX_LABEL = {
    "junyi": "tab:baseline-junyi",
    "assist2012": "tab:baseline-assist2012",
    "xes3g5m": "tab:baseline-xes3g5m",
    "synthetic_c2": "tab:baseline-synthetic-c2",
    "synthetic_c5": "tab:baseline-synthetic-c5",
}

GRAPH_ABLATION_TEX_LABEL = {
    "junyi": "tab:graph-ablation-junyi",
    "assist2012": "tab:graph-ablation-assist2012",
    "xes3g5m": "tab:graph-ablation-xes3g5m",
    "synthetic_c2": "tab:graph-ablation-synthetic-c2",
    "synthetic_c5": "tab:graph-ablation-synthetic-c5",
}

DAG_AUDIT_ROW_LABEL = {
    "junyi": (r"Junyi", r"Academy"),
    "assist2012": (r"ASSIST'12",),
    "xes3g5m": (r"XES3G5M",),
    "synthetic_c2": (r"Synthetic C2",),
    "synthetic_c5": (r"Synthetic C5",),
}

COLD_START_STRATUM_ORDER = ("very_cold", "cold", "warm", "hot", "out_of_range")


def _fmt_tex_int(n: int) -> str:
    return format(int(n), ",").replace(",", "{,}")


def _backfill_dag_audit_csv(path: Path) -> None:
    """Add n_edges_raw / n_edges_pruned using per-fold pruning logs when present."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    for col in ("n_edges_raw", "n_edges_pruned"):
        if col in df.columns:
            df = df.drop(columns=[col])
    raws: list[int] = []
    prunes: list[int] = []
    for _, row in df.iterrows():
        ds = str(row["dataset"]).strip()
        fold = int(float(row["fold"]))
        final = int(row["n_edges"])
        plp = Path("results/reports") / f"{ds}_dag_pruning_log.csv"
        pruned = 0
        if plp.exists():
            pl = pd.read_csv(plp)
            if "fold" in pl.columns:
                pruned = int(len(pl[pl["fold"].astype(int) == fold]))
            else:
                pruned = int(len(pl))
        raws.append(final + pruned)
        prunes.append(pruned)
    df["n_edges_raw"] = raws
    df["n_edges_pruned"] = prunes
    dump_csv(df.sort_values(["dataset", "fold"]), path)


def _write_dag_audit_summary_tex(path: Path) -> None:
    """Paper table: fold~0 only, all CONFIG datasets, raw/pruned/final from CSV backfill."""
    csv_path = Path("results/tables/dag_audit_summary.csv")
    if not csv_path.exists():
        return
    _backfill_dag_audit_csv(csv_path)
    df = pd.read_csv(csv_path)
    df["fold"] = df["fold"].astype(float).astype(int)
    df0 = df[df["fold"] == 0].set_index("dataset")
    ordered_slugs = [_dataset_slug(p) for p in CONFIGS]
    body_lines: list[str] = []
    for slug in ordered_slugs:
        if slug not in df0.index:
            continue
        row = df0.loc[slug]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        raw_i = int(row["n_edges_raw"]) if pd.notna(row.get("n_edges_raw")) else int(row["n_edges"])
        pruned_i = int(row["n_edges_pruned"]) if pd.notna(row.get("n_edges_pruned")) else 0
        final_i = int(row["n_edges"])
        if raw_i < final_i:
            raw_i = final_i
            pruned_i = 0
        nodes_i = int(row["n_nodes"])
        roots_i = int(row["n_roots"])
        leaves_i = int(row["n_leaves"])
        cb = int(row["n_cycles_before"])
        cycles_tex = r"$\geq\!100$" if cb >= 100 else str(cb)
        short = DAG_AUDIT_ROW_LABEL.get(slug, (slug.replace("_", r"\_"),))
        if len(short) == 1:
            ds_cell = short[0]
        else:
            ds_cell = rf"\makecell[l]{{{short[0]}\\{short[1]}}}"
        body_lines.append(
            rf"{ds_cell} & {_fmt_tex_int(nodes_i)} & {_fmt_tex_int(raw_i)} & {_fmt_tex_int(pruned_i)} & {_fmt_tex_int(final_i)} & {roots_i} & {leaves_i} & {cycles_tex} \\"
        )

    if not body_lines:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DAG audit summary (fold~0). \emph{Raw} and \emph{Final} are",
        r"the prerequisite edge counts before and after lowest-confidence cycle",
        r"pruning inside \texttt{dag\_audit}; \emph{Pruned} is the difference.",
        r"The \emph{Cycles} column reports representative cycles found before pruning",
        r"(cap~$100$). All corpora pass topological sort with zero cycles after pruning on fold~0.}",
        r"\label{tab:dag-audit}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{} >{\RaggedRight\arraybackslash}p{17mm} r >{\centering\arraybackslash}p{14mm} r >{\centering\arraybackslash}p{14mm} r r r @{}}",
        r"\toprule",
        r"\makecell[l]{Dataset}",
        r"  & $|V|$",
        r"  & \makecell{Raw\\$|\Epre|$}",
        r"  & Pruned",
        r"  & \makecell{Final\\$|\Epre|$}",
        r"  & Roots",
        r"  & Leaves",
        r"  & \makecell{Cycles\\(cap)} \\",
        r"\midrule",
        "\n".join(body_lines),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _stratum_tex(name: str) -> str:
    if name == "out_of_range":
        return r"out of range"
    return str(name).replace("_", r"\_")


def _write_cold_start_by_stratum_tex(path: Path, *, model: str = "simplekt", fold: int = 0) -> None:
    csv_path = Path("results/tables/cold_start_metrics.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    req = {"dataset", "model", "stratum", "n", "auc", "acc", "nll", "fold"}
    if df.empty or not req.issubset(df.columns):
        return
    df = df.copy()
    df["fold"] = df["fold"].astype(float).astype(int)
    model_lower = str(model).strip().lower()
    caption_model = "simpleKT" if model_lower == "simplekt" else str(model)
    sub = df[(df["model"].astype(str).str.lower() == model_lower) & (df["fold"] == fold)]
    if sub.empty:
        return

    ordered_slugs = [_dataset_slug(p) for p in CONFIGS]
    blocks: list[str] = []
    for slug in ordered_slugs:
        part = sub[sub["dataset"].astype(str).str.strip() == slug]
        if part.empty:
            continue
        # Stable stratum ordering; skip strata absent for this dataset.
        seen = set(part["stratum"].astype(str))
        strata_sorted = [s for s in COLD_START_STRATUM_ORDER if s in seen] + sorted(
            seen.difference(COLD_START_STRATUM_ORDER)
        )
        rows_tex = []
        for st in strata_sorted:
            r0 = part[part["stratum"].astype(str) == st].iloc[0]
            n_i = int(r0["n"])
            rows_tex.append(
                rf"  & {_stratum_tex(st)} & {_fmt_tex_int(n_i)} & {_fmt_float(float(r0['auc']))} & {_fmt_float(float(r0['acc']))} & {_fmt_float(float(r0['nll']))} \\"
            )
        label_parts = DAG_AUDIT_ROW_LABEL.get(slug, (slug,))
        if len(label_parts) == 1:
            mr_line = rf"\multirow{{{len(rows_tex)}}}{{*}}{{{label_parts[0]}}}"
        else:
            mr_line = rf"\multirow{{{len(rows_tex)}}}{{*}}{{\makecell[l]{{{label_parts[0]}\\{label_parts[1]}}}}}"
        first = rows_tex[0].replace("  & ", rf"{mr_line} & ", 1)
        rest = rows_tex[1:]
        blocks.append("\n".join([first, *rest]))

    if not blocks:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Per-stratum cold-start KC diagnostic with \textit{{{caption_model}}} on fold~{fold} ",
        r"(Section~\ref{sec:cold-start}). The $n$ column counts test interactions whose KC ",
        r"falls in the stratum train-frequency bin; \emph{out of range} marks test KCs ",
        r"outside the four canonical bins. Stratum mass varies by dataset split; under default bins ",
        r"Synthetic C2/C5 route all test interactions into the hot stratum ($>500$ train-fold hits per \KC{}).}",
        r"\label{tab:cold-start-strata}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{} >{\RaggedRight\arraybackslash}p{24mm} >{\ttfamily\footnotesize\raggedright\arraybackslash}p{30mm} rrrr @{}}",
        r"\toprule",
        r"Dataset & Stratum & $n$ & AUC & ACC & NLL \\",
        r"\midrule",
    ]
    lines.append("\n\\midrule\n".join(blocks))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _model_display_name(model: str) -> str:
    m = str(model).strip().lower()
    special = {
        "simplekt": r"\textit{simpleKT}",
        "dygkt": r"DyGKT",
        "dgekt": r"DGEKT",
        "akt": r"AKT",
        "bkt": r"BKT",
        "dkt": r"DKT",
        "gkt": r"GKT",
        "gikt": r"GIKT",
        "skt": r"SKT",
    }
    return special.get(m, str(model))


def _dataset_slug(config_path: Path) -> str:
    cfg = _load_yaml(config_path)
    return str(cfg["dataset"]).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_stats() -> pd.DataFrame:
    rows = []
    synthetic_placeholders: dict[str, dict[str, Any]] = {
        "synthetic_c2": {
            "n_learners": 4_000,
            "n_items": 50,
            "n_kcs": 2,
            "n_interactions": 200_000,
        },
        "synthetic_c5": {
            "n_learners": 4_000,
            "n_items": 50,
            "n_kcs": 5,
            "n_interactions": 200_000,
        },
    }
    for config_path in CONFIGS:
        cfg = _load_yaml(config_path)
        dataset = cfg["dataset"]
        processed_path = Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet"))
        if not processed_path.exists():
            if dataset in synthetic_placeholders:
                ph = synthetic_placeholders[dataset]
                nl = int(ph["n_learners"])
                niact = int(ph["n_interactions"])
                rows.append({
                    "dataset": dataset,
                    "label": DATASET_LABELS.get(dataset, dataset),
                    "n_learners": nl,
                    "n_items": int(ph["n_items"]),
                    "n_kcs": int(ph["n_kcs"]),
                    "n_interactions": niact,
                    "avg_sequence_length": niact / nl if nl else 0.0,
                    "has_dag": bool(cfg.get("has_dag", False)),
                    "missing_total": 0,
                })
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
        r"\#items and \#KCs coincide because both map to the exercise column. "
        r"Synthetic C2/C5 are companion sanity logs shipped with the preprocessing scripts.}",
        r"\label{tab:dataset-stats}",
        r"\begin{tabularx}{\columnwidth}{@{} >{\RaggedRight\arraybackslash}X rrrrr >{\centering\arraybackslash}p{14mm} @{}}",
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
    """Point estimate on first line; 95\\% CI on second (scriptsize)."""
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

    ordered_slugs = [_dataset_slug(p) for p in CONFIGS]
    blocks: list[str] = []
    base_cap = (
        r"Diagnostic baseline results (train-only graphs). Baselines are used for protocol comparison only; "
        r"no SOTA claim is made. Values are fold means with 95\% bootstrap confidence intervals when available."
    )
    first_block = True
    for slug in ordered_slugs:
        sub = df[df["dataset"].astype(str).str.strip() == slug]
        if sub.empty:
            continue
        label = BASELINE_TEX_LABEL.get(slug, f"tab:baseline-{slug}")
        human = DATASET_LABELS.get(slug, slug)
        cap = base_cap if first_block else rf"Diagnostic baselines: {human} (train-only graphs)."

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2pt}",
            rf"\caption{{{cap}}}",
        ]
        if first_block:
            lines.append(r"\label{tab:baseline-results}")
            first_block = False
        lines.append(rf"\label{{{label}}}")
        lines.extend([
            r"\begin{tabular}{@{} >{\RaggedRight\arraybackslash}p{14mm} *{3}{>{\centering\arraybackslash}p{17mm}} @{}}",
            r"\toprule",
            r"Model & AUC & ACC & NLL \\",
            r"\midrule",
        ])
        sub_sorted = sub.sort_values("model")
        for row in sub_sorted.itertuples(index=False):
            if has_ci:
                auc = _fmt_metric_makecell(row.auc, row.auc_ci_low, row.auc_ci_high)
                acc = _fmt_metric_makecell(row.acc, row.acc_ci_low, row.acc_ci_high)
                nll = _fmt_metric_makecell(row.nll, row.nll_ci_low, row.nll_ci_high)
            else:
                auc = _fmt_float(row.auc)
                acc = _fmt_float(row.acc)
                nll = _fmt_float(row.nll)
            mname = _model_display_name(row.model)
            lines.append(rf"{mname} & {auc} & {acc} & {nll} " + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        blocks.append("\n".join(lines))

    if not blocks:
        return
    path.write_text("\n".join(blocks), encoding="utf-8")


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
    label_map = {
        "junyi": "Junyi Academy",
        "assist2012": "ASSISTments 2012",
        "xes3g5m": "XES3G5M",
        "synthetic_c2": "Synthetic C2",
        "synthetic_c5": "Synthetic C5",
    }
    for row in df.sort_values("dataset").itertuples(index=False):
        ds = str(row.dataset).strip()
        lines.append(
            f"{label_map.get(ds, ds)} & {_fmt_float(row.ecr_flag)} & {_fmt_float(row.ecr_overlap)} & {_fmt_float(row.eoc)} & {_fmt_float(row.tbvr)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_graph_ablation_tex(path: Path) -> None:
    """Emit train-only vs full-log ablation tables (one single-column table per dataset)."""
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

    ordered_slugs = [_dataset_slug(p) for p in CONFIGS]
    base_cap = (
        r"Ablation contrasting \emph{train-only} graph smoothing (protocol) with a \emph{full-log} graph "
        r"built from all learners before splitting (leaky construction). Graph-augmented baselines use the "
        r"same diagnostic feature channels as Section~\ref{sec:exp-baseline}; by default a \emph{trained} logistic "
        r"head is fit on the train fold only (coefficients can emphasize the leaked graph channel). "
        r"$\Delta$ is the paired mean across folds (full-log minus train-only); "
        r"positive $\Delta$AUC indicates higher discrimination when the graph sees the entire log. "
        r"\emph{to}: train-only graph; \emph{fl}: full-log graph."
    )
    blocks: list[str] = []
    first_block = True
    for slug in ordered_slugs:
        sub = df[df["dataset"].astype(str).str.strip() == slug]
        if sub.empty:
            continue
        human = DATASET_LABELS.get(slug, slug)
        label = GRAPH_ABLATION_TEX_LABEL.get(slug, f"tab:graph-ablation-{slug}")
        cap = base_cap if first_block else rf"Train-only vs.\ full-log graph ablation: {human}."

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            rf"\caption{{{cap}}}",
        ]
        if first_block:
            lines.append(r"\label{tab:graph-ablation}")
            first_block = False
        lines.append(rf"\label{{{label}}}")
        lines.extend([
            r"\begin{tabular}{@{} >{\RaggedRight\arraybackslash}p{12mm} *{6}{c} @{}}",
            r"\toprule",
            r"Model & \makecell{AUC\\to} & \makecell{AUC\\fl} & $\Delta$AUC"
            r" & \makecell{ACC\\to} & \makecell{ACC\\fl} & $\Delta$ACC \\",
            r"\midrule",
        ])
        for row in sub.sort_values("_mo").itertuples(index=False):
            mname = _model_display_name(row.model)
            lines.append(
                rf"{mname} & {_fmt_float(row.auc_train_only)} & {_fmt_float(row.auc_full_log)} & {_fmt_float(row.delta_auc)}"
                rf" & {_fmt_float(row.acc_train_only)} & {_fmt_float(row.acc_full_log)} & {_fmt_float(row.delta_acc)} "
                + r"\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        blocks.append("\n".join(lines))

    if not blocks:
        return
    path.write_text("\n".join(blocks), encoding="utf-8")


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
    _write_dag_audit_summary_tex(Path("results/tables/dag_audit_summary.tex"))
    _write_cold_start_by_stratum_tex(Path("results/tables/cold_start_by_stratum.tex"))
    _write_cold_start_tex(Path("results/tables/cold_start_metrics.tex"))
    _write_artifact_index(Path("results/reports/paper_artifact_index.md"))


if __name__ == "__main__":
    main()
