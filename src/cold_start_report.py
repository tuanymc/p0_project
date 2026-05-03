"""Cold-start KC diagnostic reporting for the P0 protocol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.io_utils import dump_csv

logger = logging.getLogger(__name__)


def bin_kcs_by_frequency(
    train: pd.DataFrame,
    bins: tuple = ((-1, 19), (20, 99), (100, 499), (500, 10_000_000)),
) -> pd.DataFrame:
    """Assign KCs to frequency strata from train interactions."""
    logger.info("Binning KCs train_shape=%s bins=%s", train.shape, bins)
    counts = train.groupby("kc_id").size().rename("n_train_interactions").reset_index()
    labels = ["very_cold", "cold", "warm", "hot"][: len(bins)]
    def label_count(n: int) -> str:
        for label, (lo, hi) in zip(labels, bins):
            if lo < n <= hi:
                return label
        return "out_of_range"
    counts["stratum"] = counts["n_train_interactions"].map(label_count)
    logger.info("KC strata shape=%s", counts.shape)
    return counts


def per_stratum_metrics(
    predictions: pd.DataFrame,
    strata: pd.DataFrame,
    metrics: Sequence[str] = ("auc", "acc", "nll"),
) -> pd.DataFrame:
    """Compute diagnostic prediction metrics per KC frequency stratum."""
    logger.info("Computing per-stratum metrics predictions_shape=%s strata_shape=%s metrics=%s", predictions.shape, strata.shape, metrics)
    if any("user" in str(s).lower() or "item" in str(s).lower() for s in strata.get("stratum", [])):
        raise ValueError("Cold-start report is KC-only, not user/item.")
    required = {"kc_id", "y_true", "y_prob"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    merged = predictions.merge(strata[["kc_id", "stratum"]], on="kc_id", how="inner")
    rows = []
    for stratum, part in merged.groupby("stratum"):
        row = {"stratum": stratum, "n": len(part)}
        if "auc" in metrics:
            row["auc"] = roc_auc_score(part["y_true"], part["y_prob"]) if part["y_true"].nunique() > 1 else np.nan
        if "acc" in metrics:
            row["acc"] = accuracy_score(part["y_true"], part["y_prob"] >= 0.5)
        if "nll" in metrics:
            row["nll"] = log_loss(part["y_true"], part["y_prob"], labels=[0, 1])
        rows.append(row)
    result = pd.DataFrame(rows)
    logger.info("Per-stratum metrics shape=%s", result.shape)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Write cold-start KC diagnostic report")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    out = Path("results/reports/cold_start_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path("results/tables/cold_start_metrics.csv")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.DataFrame(columns=["dataset", "model", "stratum", "n", "auc", "acc", "nll"])
        dump_csv(metrics, metrics_path)
    if metrics.empty:
        out.write_text(
            "# Cold-Start KC Diagnostic\n\n"
            "No diagnostic prediction metrics are available yet. Run `src.baseline_runner` first.\n",
            encoding="utf-8",
        )
        return
    summary = metrics.groupby(["dataset", "model"], as_index=False).agg(
        auc=("auc", "mean"),
        acc=("acc", "mean"),
        nll=("nll", "mean"),
        n=("n", "sum"),
    )
    lines = ["# Cold-Start KC Diagnostic", ""]
    for row in summary.sort_values(["dataset", "model"]).itertuples(index=False):
        lines.append(
            f"- {row.dataset} / {row.model}: mean AUC={row.auc:.3f}, "
            f"ACC={row.acc:.3f}, NLL={row.nll:.3f}, n={int(row.n):,}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
