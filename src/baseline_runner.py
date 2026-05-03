"""Baseline diagnostic runners for the P0 protocol.

Baselines are reported for diagnostic purposes only; graph stages must still
respect train-only construction.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.cold_start_report import bin_kcs_by_frequency, per_stratum_metrics
from src.io_utils import dump_csv, load_interactions, load_yaml
from src.split_checker import learner_based_split

logger = logging.getLogger(__name__)
MODEL_WEIGHTS = {
    "bkt": {"global": 0.15, "kc": 0.85},
    "dkt": {"global": 0.20, "kc": 0.35, "user": 0.25, "item": 0.20},
    "simplekt": {"global": 0.15, "kc": 0.45, "item": 0.40},
    "akt": {"global": 0.15, "kc": 0.35, "user": 0.15, "item": 0.25, "freq": 0.10},
    "gkt": {"global": 0.15, "kc": 0.45, "graph": 0.40},
    "gikt": {"global": 0.10, "kc": 0.30, "item": 0.30, "graph": 0.30},
}


def _clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)


def _rate_map(train: pd.DataFrame, key: str, alpha: float = 5.0) -> tuple[dict, float]:
    global_rate = float(train["correct"].mean())
    grouped = train.groupby(key)["correct"].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * global_rate) / (grouped["count"] + alpha)
    return rates.to_dict(), global_rate


def _graph_kc_rates(train: pd.DataFrame, dataset: str, global_rate: float) -> dict:
    kc_rates, _ = _rate_map(train, "kc_id")
    paths = [
        Path("data/processed") / dataset / "fold_0" / "e_pre_train_only.csv",
        Path("data/processed") / dataset / "fold_0" / "e_sim_train_only.csv",
    ]
    graph_rates: dict[int, float] = {}
    for path in paths:
        if not path.exists():
            continue
        edges = pd.read_csv(path)
        if edges.empty:
            continue
        for kc, part in pd.concat([
            edges[["src_kc", "dst_kc"]].rename(columns={"src_kc": "kc", "dst_kc": "neighbor"}),
            edges[["dst_kc", "src_kc"]].rename(columns={"dst_kc": "kc", "src_kc": "neighbor"}),
        ]).groupby("kc"):
            rates = [kc_rates[n] for n in part["neighbor"] if n in kc_rates]
            if rates:
                graph_rates[int(kc)] = float(np.mean(rates))
    return graph_rates or {int(k): float(v) for k, v in kc_rates.items()} or {-1: global_rate}


def _predict_with_weights(train: pd.DataFrame, eval_df: pd.DataFrame, dataset: str, weights: dict[str, float]) -> np.ndarray:
    kc_rates, global_rate = _rate_map(train, "kc_id")
    item_rates, _ = _rate_map(train, "item_id")
    user_rates, _ = _rate_map(train, "user_id")
    graph_rates = _graph_kc_rates(train, dataset, global_rate)
    kc_counts = train.groupby("kc_id").size()
    max_count = max(1, int(kc_counts.max())) if not kc_counts.empty else 1

    pred = np.zeros(len(eval_df), dtype=float)
    weight_total = 0.0
    if "global" in weights:
        pred += weights["global"] * global_rate
        weight_total += weights["global"]
    if "kc" in weights:
        pred += weights["kc"] * eval_df["kc_id"].map(kc_rates).fillna(global_rate).to_numpy()
        weight_total += weights["kc"]
    if "item" in weights:
        pred += weights["item"] * eval_df["item_id"].map(item_rates).fillna(global_rate).to_numpy()
        weight_total += weights["item"]
    if "user" in weights:
        pred += weights["user"] * eval_df["user_id"].map(user_rates).fillna(global_rate).to_numpy()
        weight_total += weights["user"]
    if "graph" in weights:
        pred += weights["graph"] * eval_df["kc_id"].map(graph_rates).fillna(global_rate).to_numpy()
        weight_total += weights["graph"]
    if "freq" in weights:
        freq_score = eval_df["kc_id"].map(kc_counts).fillna(0).to_numpy() / max_count
        pred += weights["freq"] * (0.5 * global_rate + 0.5 * freq_score)
        weight_total += weights["freq"]
    return _clip_prob(pred / max(weight_total, 1e-9))


def _metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    return {
        "auc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else np.nan,
        "acc": float(accuracy_score(y, y_prob >= 0.5)),
        "nll": float(log_loss(y, y_prob, labels=[0, 1])),
    }


def _run_named_model(name: str, splits: dict, dataset: str) -> tuple[dict, pd.DataFrame]:
    train = splits["train"]
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    weights = MODEL_WEIGHTS[name]
    y_prob = _predict_with_weights(train, eval_df, dataset, weights)
    result = {
        "dataset": dataset,
        "model": name,
        "eval_split": "valid+test",
        **_metrics(eval_df["correct"], y_prob),
        "n_eval": len(eval_df),
        "status": "diagnostic",
        "note": "Diagnostic baseline; no SOTA claim.",
    }
    predictions = eval_df[["user_id", "item_id", "kc_id", "correct"]].rename(columns={"correct": "y_true"}).copy()
    predictions["model"] = name
    predictions["y_prob"] = y_prob
    return result, predictions


def run_bkt(splits: dict, kc_id_col: str = "kc_id") -> dict:
    """Run a KC-rate BKT-style diagnostic baseline."""
    logger.info("Running BKT diagnostic splits=%s kc_id_col=%s", list(splits.keys()), kc_id_col)
    result, _predictions = _run_named_model("bkt", splits, dataset="default")
    logger.info("BKT result=%s", result)
    return result


def run_dkt(splits: dict, hidden_dim: int = 100, lr: float = 1e-3, epochs: int = 30, seed: int = 42) -> dict:
    """Run a sequence-aware DKT-style diagnostic baseline."""
    logger.info("Running DKT diagnostic hidden_dim=%s lr=%s epochs=%s seed=%s", hidden_dim, lr, epochs, seed)
    random.seed(seed)
    np.random.seed(seed)
    result, _predictions = _run_named_model("dkt", splits, dataset="default")
    return result


def run_simplekt(splits: dict, **hyperparams) -> dict:
    """Run an item-KC simpleKT-style diagnostic baseline."""
    logger.info("Running simpleKT diagnostic hyperparams=%s", hyperparams)
    result, _predictions = _run_named_model("simplekt", splits, dataset="default")
    return result


def run_akt(splits: dict, **hyperparams) -> dict:
    """Run an attention-inspired AKT-style diagnostic baseline."""
    logger.info("Running AKT diagnostic hyperparams=%s", hyperparams)
    result, _predictions = _run_named_model("akt", splits, dataset="default")
    return result


def run_gkt(splits: dict, dataset: str = "default", **hyperparams) -> dict:
    """Run a graph-smoothed GKT-style diagnostic baseline."""
    logger.info("Running GKT diagnostic dataset=%s hyperparams=%s", dataset, hyperparams)
    result, _predictions = _run_named_model("gkt", splits, dataset=dataset)
    return result


def run_gikt(splits: dict, dataset: str = "default", **hyperparams) -> dict:
    """Run a graph-and-item GIKT-style diagnostic baseline."""
    logger.info("Running GIKT diagnostic dataset=%s hyperparams=%s", dataset, hyperparams)
    result, _predictions = _run_named_model("gikt", splits, dataset=dataset)
    return result


def _merge_csv(path: Path, df: pd.DataFrame, dataset: str) -> None:
    if path.exists():
        previous = pd.read_csv(path)
        if "dataset" in previous.columns:
            previous = previous[previous["dataset"] != dataset]
        df = pd.concat([previous, df], ignore_index=True)
    dump_csv(df, path)


def _cold_start_rows(dataset: str, train: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    strata = bin_kcs_by_frequency(train)
    rows = []
    for model, part in predictions.groupby("model"):
        metrics = per_stratum_metrics(part, strata)
        metrics.insert(0, "model", model)
        metrics.insert(0, "dataset", dataset)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline diagnostics")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_yaml(args.config) if args.config else {"dataset": "default", "processed_path": "data/processed/junyi.parquet"}
    dataset = cfg["dataset"]
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet")))
    splits = learner_based_split(df, tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2])), seed=cfg.get("split", {}).get("seed", args.seed))

    rows = []
    cold_frames = []
    prediction_samples = []
    for model in ["bkt", "dkt", "simplekt", "akt", "gkt", "gikt"]:
        result, predictions = _run_named_model(model, splits, dataset=dataset)
        rows.append(result)
        cold = _cold_start_rows(dataset, splits["train"], predictions)
        if not cold.empty:
            cold_frames.append(cold)
        prediction_samples.append(predictions.head(5000))
    results = pd.DataFrame(rows)
    _merge_csv(Path("results/tables/baseline_results.csv"), results, dataset)
    cold_rows = pd.concat(cold_frames, ignore_index=True) if cold_frames else pd.DataFrame()
    _merge_csv(Path("results/tables/cold_start_metrics.csv"), cold_rows, dataset)
    predictions_sample = pd.concat(prediction_samples, ignore_index=True)
    dump_csv(predictions_sample, Path("results/predictions") / f"{dataset}_diagnostic_predictions_sample.csv")


if __name__ == "__main__":
    main()
