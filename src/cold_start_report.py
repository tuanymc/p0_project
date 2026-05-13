"""Cold-start KC diagnostic reporting for the P0 protocol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.io_utils import dump_csv

logger = logging.getLogger(__name__)


def _binary_nll_mean_chunked(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    chunk_size: int = 500_000,
) -> float:
    """Mean binary log loss without sklearn's multi-array overhead (large-N safe)."""
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    eps = 1e-15
    total = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y = y_true[start:end].astype(np.float64, copy=False)
        p = np.clip(y_prob[start:end].astype(np.float64, copy=False), eps, 1.0 - eps)
        total += float(np.sum(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    return total / n


def _binary_roc_auc_mann_whitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC AUC via Mann–Whitney U / (n_pos * n_neg); tie-aware mid-ranks.

    Avoids sklearn's ``roc_auc_score`` path (``label_binarize`` / ``isin``), which
    spikes RAM on multi-million-row strata.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64, copy=False)
    pos = y_true != 0
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    pos_sorted = pos[order]
    del order

    n = sorted_scores.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[i:j] = avg_rank
        i = j

    del sorted_scores
    # Boolean indexing ``ranks[pos_sorted]`` allocates an (n_pos,) array (~50MB+); chunk dot avoids it.
    chunk_size = 500_000
    rank_sum_pos = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        rank_sum_pos += float(np.dot(ranks[start:end], pos_sorted[start:end]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _binary_accuracy_mean_chunked(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    chunk_size: int = 500_000,
) -> float:
    """Fraction correct vs 0.5 threshold without sklearn (large-N safe)."""
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    correct = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yt = y_true[start:end]
        pred = y_prob[start:end] >= 0.5
        correct += int(np.sum(yt == pred))
    return correct / n


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
    # Avoid pd.merge on multi-million rows (large concat / consolidate spikes RAM).
    lookup = pd.Series(strata["stratum"].to_numpy(), index=strata["kc_id"].to_numpy())
    stratum_labels = predictions["kc_id"].map(lookup)
    mask = stratum_labels.notna().to_numpy()
    if not mask.any():
        return pd.DataFrame(columns=["stratum", "n"] + [m for m in metrics if m in ("auc", "acc", "nll")])
    sub = pd.DataFrame({
        "stratum": stratum_labels.to_numpy()[mask],
        "y_true": predictions["y_true"].to_numpy(dtype=np.int64)[mask],
        "y_prob": predictions["y_prob"].to_numpy(dtype=float)[mask],
    })
    rows = []
    for stratum, part in sub.groupby("stratum"):
        row = {"stratum": stratum, "n": len(part)}
        yt = part["y_true"].to_numpy()
        yp = part["y_prob"].to_numpy()
        if "auc" in metrics:
            row["auc"] = (
                _binary_roc_auc_mann_whitney(yt, yp)
                if int(yt.min()) != int(yt.max())
                else np.nan
            )
        if "acc" in metrics:
            row["acc"] = _binary_accuracy_mean_chunked(yt, yp)
        if "nll" in metrics:
            row["nll"] = _binary_nll_mean_chunked(
                part["y_true"].to_numpy(),
                part["y_prob"].to_numpy(),
            )
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
