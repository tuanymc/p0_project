#!/usr/bin/env python3
"""Run GT cross-validation: inferred train-only E_pre vs Junyi expert annotation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gt_cross_validation import (
    DEFAULT_K_LIST,
    align_kc_ids,
    compute_overlap_metrics,
    diagnose_disagreement,
    disagreement_dict_to_dataframe,
    load_expert_dag,
    precision_recall_sweep,
)
from src.io_utils import load_interactions, load_yaml
from src.split_checker import learner_based_split

logger = logging.getLogger(__name__)


def _parse_k_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return list(DEFAULT_K_LIST)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _write_tex_table(sweep: pd.DataFrame, n_expert_edges: int, path: Path, dataset_label: str = "Junyi Academy") -> None:
    """Camera-ready LaTeX snippet (requires \\usepackage{booktabs})."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_tex = []
    want_k = {200, 500, 800, 1200, 1800, 2500, 3500, 5000, n_expert_edges}
    subset = sweep[sweep["top_k"].isin(want_k)].copy()
    subset = subset.sort_values("top_k")
    for _, r in subset.iterrows():
        tk = int(r["top_k"])
        if tk == n_expert_edges:
            label = rf"\makecell[l]{{$\lvert E_\mathrm{{expert}}\rvert={n_expert_edges}$}}"
        elif tk == 5000:
            label = r"5000 (cap)"
        else:
            label = str(tk)
        rows_tex.append(
            f"{label} & {r['edge_precision']:.3f} & {r['edge_recall']:.3f} & "
            f"{r['direction_agreement']:.3f} & {r['reachability_f1']:.3f} " + r"\\"
        )
    body = "\n".join(rows_tex)
    content = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        rf"\caption{{GT cross-validation: agreement between train-only inferred "
        rf"$E_{{\text{{pre}}}}$ and expert prerequisite DAG on {dataset_label} "
        rf"($|E|={n_expert_edges}$ directed edges).}}" + "\n"
        r"\label{tab:gt-validation}" + "\n"
        r"\footnotesize" + "\n"
        r"\setlength{\tabcolsep}{2pt}" + "\n"
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}X cccc @{}}" + "\n"
        r"\toprule" + "\n"
        r"\makecell[l]{$K$ (top inferred)} & \makecell{Edge\\prec.} & \makecell{Edge\\rec.} & "
        r"\makecell{Dir.\\agr.} & \makecell{Rch.\\F1} \\" + "\n"
        r"\midrule" + "\n"
        f"{body}\n"
        r"\bottomrule" + "\n"
        r"\end{tabularx}" + "\n"
        r"\end{table}" + "\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Junyi GT cross-validation")
    parser.add_argument("--expert-dag", type=Path, default=Path("data/raw/junyi/junyi_dag.csv"))
    parser.add_argument("--inferred-dag", type=Path, default=Path("data/processed/junyi/fold_0/e_pre_train_only.csv"))
    parser.add_argument("--kc-mapping", type=Path, default=Path("data/processed/junyi/kc_name_to_id.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gt_validation/junyi"))
    parser.add_argument("--config", type=Path, default=Path("configs/junyi.yaml"))
    parser.add_argument("--k-list", type=str, default="", help="Comma-separated K values; default uses built-in list.")
    parser.add_argument("--prereq-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    k_list = sorted(set(_parse_k_list(args.k_list)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.config))

    mapping_path = Path(args.kc_mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"KC mapping not found: {mapping_path}")
    with mapping_path.open(encoding="utf-8") as f:
        kc_name_to_id = {str(k): int(v) for k, v in json.load(f).items()}

    expert_loaded = load_expert_dag(
        Path(args.expert_dag),
        schema="junyi_name",
        kc_name_to_id=kc_name_to_id,
        prereq_threshold=float(args.prereq_threshold),
    )
    expert_loaded = expert_loaded.drop_duplicates(subset=["src_kc", "dst_kc"], keep="first")
    parquet_path = Path(cfg.get("processed_path", f"data/processed/{cfg['dataset']}.parquet"))
    df = load_interactions(parquet_path)
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    split_seed = int(cfg.get("split", {}).get("seed", args.seed))
    splits = learner_based_split(df, ratios, seed=split_seed)
    train_kc_ids = set(int(x) for x in splits["train"]["kc_id"].unique())

    aligned, alignment_report = align_kc_ids(expert_loaded.copy(), train_kc_ids=train_kc_ids)
    alignment_report["split_seed"] = split_seed
    alignment_report["prereq_threshold"] = float(args.prereq_threshold)
    alignment_report["n_expert_edges_loaded"] = len(expert_loaded)
    (out_dir / "alignment_report.json").write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    rate = float(alignment_report.get("alignment_rate", 0.0))
    if rate < 0.8:
        logger.warning("Train-fold alignment_rate=%s < 0.80; continuing.", rate)

    expert_matched = aligned.loc[aligned["status"] == "matched", ["src_kc", "dst_kc", "confidence_score"]].copy()
    n_expert = len(expert_matched)

    inferred = pd.read_csv(Path(args.inferred_dag))
    for col in ("src_kc", "dst_kc"):
        inferred[col] = inferred[col].astype(np.int64)

    extra_k = [k for k in (n_expert, 5000) if k not in k_list]
    sweep_k = sorted(set(k_list + extra_k))
    sweep = precision_recall_sweep(inferred, expert_matched, sweep_k)
    sweep.to_csv(out_dir / "overlap_metrics_at_K.csv", index=False)

    summary = {
        "K_equal_|expert|": compute_overlap_metrics(inferred, expert_matched, max(n_expert, 0)),
        "K_5000": compute_overlap_metrics(inferred, expert_matched, 5000),
        "n_expert_edges_matched": n_expert,
        "n_inferred_edges_available": len(inferred),
    }
    (out_dir / "overlap_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diag_k = max(n_expert, 1)
    diag = diagnose_disagreement(
        inferred,
        expert_matched,
        top_k=diag_k,
        n_examples=20,
        kc_name_to_id=kc_name_to_id,
    )
    disagreement_dict_to_dataframe(diag).to_csv(out_dir / "disagreement_examples.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["edge_recall"], sweep["edge_precision"], marker="o")
    for _, row in sweep.iterrows():
        ax.annotate(str(int(row["top_k"])), (row["edge_recall"], row["edge_precision"]), fontsize=7)
    ax.set_xlabel("Edge recall")
    ax.set_ylabel("Edge precision")
    ax.set_title("GT CV: precision vs recall over top-K truncation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pr_curve.pdf")
    plt.close()

    _write_tex_table(sweep, n_expert, out_dir / "gt_validation_table.tex")

    logger.info("Wrote GT validation artefacts under %s", out_dir)


if __name__ == "__main__":
    main()
