"""Baseline diagnostic runners for the P0 protocol.

Baselines are reported for diagnostic purposes only; graph stages must still
respect train-only construction.

GKT/GIKT/SKT/DyGKT/DGEKT here are *linear diagnostic ensembles* (global / KC / item /
user / graph weights), not trained PyTorch checkpoints. Names follow the cited model
families but instantiate stylised blends for reproducible protocol diagnostics only.
The ``graph`` term averages neighbour KC correctness rates along exported edges. For ``graph_construction="train_only"``, neighbours are
scored using **train-fold** outcomes only; for ``"full_log"``, edge lists come from
``full_log/e_pre.csv`` (etc.) and neighbour scores use **pooled** correctness over the
entire preprocessed interaction table so the ablation reflects both topology and
statistics from the full log, not topology alone.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.cold_start_report import bin_kcs_by_frequency, per_stratum_metrics
from src.io_utils import dump_csv, load_interactions, load_yaml
from src.split_checker import fold_seeds, learner_based_folds

logger = logging.getLogger(__name__)
MODEL_WEIGHTS = {
    "bkt": {"global": 0.15, "kc": 0.85},
    "dkt": {"global": 0.20, "kc": 0.35, "user": 0.25, "item": 0.20},
    "simplekt": {"global": 0.15, "kc": 0.45, "item": 0.40},
    "akt": {"global": 0.15, "kc": 0.35, "user": 0.15, "item": 0.25, "freq": 0.10},
    "gkt": {"global": 0.15, "kc": 0.45, "graph": 0.40},
    "gikt": {"global": 0.10, "kc": 0.30, "item": 0.30, "graph": 0.30},
    # Stylised graph-KT diagnostics (weights sum to 1; not official checkpoints).
    "skt": {"global": 0.10, "kc": 0.38, "item": 0.10, "graph": 0.42},  # prereq+similarity proxy via graph+KC
    "dygkt": {"global": 0.10, "kc": 0.33, "user": 0.22, "item": 0.05, "graph": 0.30},  # dynamic: user + graph
    "dgekt": {"global": 0.08, "kc": 0.30, "item": 0.27, "graph": 0.35},  # dual-view: item + graph
}


def _clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)


def _rate_map(train: pd.DataFrame, key: str, alpha: float = 5.0) -> tuple[dict, float]:
    global_rate = float(train["correct"].mean())
    grouped = train.groupby(key)["correct"].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * global_rate) / (grouped["count"] + alpha)
    return rates.to_dict(), global_rate


def _full_log_graph_paths(dataset: str) -> list[Path]:
    return [
        Path("data/processed") / dataset / "full_log" / "e_pre.csv",
        Path("data/processed") / dataset / "full_log" / "e_sim.csv",
    ]


def _graph_kc_rates(
    train: pd.DataFrame,
    dataset: str,
    global_rate: float,
    fold: int = 0,
    *,
    graph_construction: str = "train_only",
    neighbor_stats_df: pd.DataFrame | None = None,
) -> dict:
    """Neighbour correctness means for graph smoothing along exported edges.

    Train-only mode uses train-fold outcomes for neighbours and fold-specific edge CSVs.
    Full-log mode uses ``full_log`` edge CSVs and, when ``neighbor_stats_df`` is the full
    interaction table, pooled KC correctness (deliberate leakage in the graph branch only).
    """
    kc_rates_train, _ = _rate_map(train, "kc_id")
    if graph_construction == "full_log":
        paths = _full_log_graph_paths(dataset)
        stats_df = neighbor_stats_df if neighbor_stats_df is not None else train
        if neighbor_stats_df is None:
            logger.warning(
                "full_log graph smoothing without pooled interactions; falling back to train "
                "(graph ablation Δ will be topology-only and often near zero)"
            )
    else:
        paths = [
            Path("data/processed") / dataset / f"fold_{fold}" / "e_pre_train_only.csv",
            Path("data/processed") / dataset / f"fold_{fold}" / "e_sim_train_only.csv",
        ]
        stats_df = train

    kc_rates_neighbors, _ = _rate_map(stats_df, "kc_id")
    graph_rates: dict[int, float] = {}
    edge_counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        edges = pd.read_csv(path)
        edge_counts[path.name] = len(edges)
        if edges.empty:
            continue
        for kc, part in pd.concat([
            edges[["src_kc", "dst_kc"]].rename(columns={"src_kc": "kc", "dst_kc": "neighbor"}),
            edges[["dst_kc", "src_kc"]].rename(columns={"dst_kc": "kc", "src_kc": "neighbor"}),
        ]).groupby("kc"):
            rates = [kc_rates_neighbors[n] for n in part["neighbor"] if n in kc_rates_neighbors]
            if rates:
                graph_rates[int(kc)] = float(np.mean(rates))
    if edge_counts:
        logger.info(
            "Graph KC smoothing dataset=%s fold=%s construction=%s edge_counts=%s neighbor_rows=%s",
            dataset,
            fold,
            graph_construction,
            edge_counts,
            len(stats_df),
        )
    return graph_rates or {int(k): float(v) for k, v in kc_rates_train.items()} or {-1: global_rate}


def _predict_with_weights(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    dataset: str,
    weights: dict[str, float],
    fold: int = 0,
    *,
    graph_construction: str = "train_only",
    full_interactions: pd.DataFrame | None = None,
) -> np.ndarray:
    kc_rates, global_rate = _rate_map(train, "kc_id")
    item_rates, _ = _rate_map(train, "item_id")
    user_rates, _ = _rate_map(train, "user_id")
    neighbor_df = full_interactions if graph_construction == "full_log" else train
    graph_rates = _graph_kc_rates(
        train,
        dataset,
        global_rate,
        fold=fold,
        graph_construction=graph_construction,
        neighbor_stats_df=neighbor_df,
    )
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


def _mean_binary_nll(y: np.ndarray, p: np.ndarray, chunk: int = 1_000_000) -> float:
    """Mean binary log-loss in chunks to limit peak RAM on large val+test folds."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    n = len(y)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        yt, pt = y[sl], p[sl]
        total += float(np.sum(yt * np.log(pt) + (1.0 - yt) * np.log(1.0 - pt)))
    return float(-total / n)


def _metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(y_prob, dtype=np.float64)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "acc": float(accuracy_score(y, p >= 0.5)),
        "nll": _mean_binary_nll(y, p),
    }


def _run_named_model(
    name: str,
    splits: dict,
    dataset: str,
    fold: int = 0,
    split_seed: int | None = None,
    *,
    graph_construction: str = "train_only",
    prediction_cap: int | None = None,
    full_interactions: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    train = splits["train"]
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    weights = MODEL_WEIGHTS[name]
    y_prob = _predict_with_weights(
        train,
        eval_df,
        dataset,
        weights,
        fold=fold,
        graph_construction=graph_construction,
        full_interactions=full_interactions,
    )
    result = {
        "dataset": dataset,
        "fold": fold,
        "split_seed": split_seed,
        "model": name,
        "graph_construction": graph_construction,
        "eval_split": "valid+test",
        **_metrics(eval_df["correct"], y_prob),
        "n_eval": len(eval_df),
        "status": "diagnostic",
        "note": "Diagnostic baseline; no SOTA claim.",
    }
    cap = len(eval_df) if prediction_cap is None else min(int(prediction_cap), len(eval_df))
    tail = eval_df.iloc[:cap]
    predictions = tail[["user_id", "item_id", "kc_id", "correct"]].rename(columns={"correct": "y_true"}).copy()
    predictions["fold"] = fold
    predictions["model"] = name
    predictions["y_prob"] = y_prob[:cap]
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
        if "graph_construction" not in previous.columns and path.name in {"baseline_fold_results.csv", "baseline_results.csv"}:
            previous["graph_construction"] = "train_only"
        df = pd.concat([previous, df], ignore_index=True)
    dump_csv(df, path)


def _cold_start_rows(
    dataset: str,
    fold: int,
    split_seed: int | None,
    train: pd.DataFrame,
    predictions: pd.DataFrame,
    strata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    strata = strata if strata is not None else bin_kcs_by_frequency(train)
    rows = []
    for model, part in predictions.groupby("model"):
        metrics = per_stratum_metrics(part, strata)
        metrics.insert(0, "model", model)
        metrics.insert(0, "split_seed", split_seed)
        metrics.insert(0, "fold", fold)
        metrics.insert(0, "dataset", dataset)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _enabled_models(cfg: dict) -> list[str]:
    configured = cfg.get("baselines")
    if not configured:
        return list(MODEL_WEIGHTS)
    models = []
    for item in configured:
        name = item.get("name")
        if item.get("enabled", True) and name in MODEL_WEIGHTS:
            models.append(name)
    return models


def _bootstrap_mean_ci(values: pd.Series, seed: int, n_bootstrap: int = 1000) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _summarize_fold_results(fold_results: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    df = fold_results.copy()
    if "graph_construction" not in df.columns:
        df["graph_construction"] = "train_only"
    df["graph_construction"] = df["graph_construction"].fillna("train_only")
    for (dataset, model, graph_construction), part in df.groupby(["dataset", "model", "graph_construction"]):
        row = {
            "dataset": dataset,
            "model": model,
            "graph_construction": graph_construction,
            "eval_split": "valid+test",
            "n_folds": int(part["fold"].nunique()),
            "n_eval": int(part["n_eval"].sum()),
            "status": "diagnostic",
            "note": "Diagnostic baseline; mean over folds with bootstrap CI.",
        }
        for metric in ["auc", "acc", "nll"]:
            row[metric] = float(part[metric].mean())
            ci_low, ci_high = _bootstrap_mean_ci(part[metric], seed=seed, n_bootstrap=n_bootstrap)
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_graph_ablation(fold_results: pd.DataFrame) -> pd.DataFrame:
    r"""One row per (dataset, model) with paired train-only vs.\ full-log means."""
    df = fold_results.copy()
    if "graph_construction" not in df.columns:
        return pd.DataFrame()
    df["graph_construction"] = df["graph_construction"].fillna("train_only")
    paired_models: list[str] = []
    for model, part in df.groupby("model"):
        modes = set(part["graph_construction"].astype(str))
        if "train_only" in modes and "full_log" in modes:
            paired_models.append(str(model))
    if not paired_models:
        return pd.DataFrame()
    subset = df[df["model"].isin(paired_models)]
    rows: list[dict[str, float | str]] = []
    for (ds, model), part in subset.groupby(["dataset", "model"]):
        t = part[part["graph_construction"] == "train_only"].sort_values("fold")
        f = part[part["graph_construction"] == "full_log"].sort_values("fold")
        if len(t) != len(f) or t.empty:
            continue
        rows.append({
            "dataset": ds,
            "model": model,
            "auc_train_only": float(t["auc"].mean()),
            "auc_full_log": float(f["auc"].mean()),
            "delta_auc": float((f["auc"].to_numpy() - t["auc"].to_numpy()).mean()),
            "acc_train_only": float(t["acc"].mean()),
            "acc_full_log": float(f["acc"].mean()),
            "delta_acc": float((f["acc"].to_numpy() - t["acc"].to_numpy()).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline diagnostics")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--skip-cold-start",
        action="store_true",
        help="Skip per-stratum cold-start metrics (saves RAM on very large val+test folds).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_yaml(args.config) if args.config else {"dataset": "default", "processed_path": "data/processed/junyi.parquet"}
    dataset = cfg["dataset"]
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet")))
    split_cfg = cfg.get("split", {})
    ratios = tuple(split_cfg.get("ratios", [0.7, 0.1, 0.2]))
    n_bootstrap = int(cfg.get("evaluation", {}).get("n_bootstrap", cfg.get("baselines_n_bootstrap", 1000)))
    models = _enabled_models(cfg)

    graph_ablation_cfg = cfg.get("graph_ablation", {})
    ablation_enabled = graph_ablation_cfg.get("enabled", True)
    full_log_ready = _full_log_graph_paths(dataset)[0].exists()
    ablation_models = [m for m in graph_ablation_cfg.get("models", ["gkt", "gikt", "skt", "dygkt", "dgekt"]) if m in MODEL_WEIGHTS]
    pred_cap = 5000 if args.skip_cold_start else None

    fold_seed_list = fold_seeds(split_cfg, default_seed=args.seed)
    n_plan_folds = len(fold_seed_list)
    full_log_models = [m for m in ablation_models if m in models] if ablation_enabled and full_log_ready else []
    total_runs = n_plan_folds * len(models) + n_plan_folds * len(full_log_models)
    run_idx = 0

    rows = []
    cold_frames = []
    prediction_samples = []
    for fold, split_seed, splits in learner_based_folds(df, ratios, split_cfg, default_seed=args.seed):
        strata_once = bin_kcs_by_frequency(splits["train"])
        for model in models:
            run_idx += 1
            logger.info(
                "Baseline progress [%s] %d/%d (~%.0f%%) fold=%s model=%s graph_construction=train_only",
                dataset,
                run_idx,
                total_runs,
                100.0 * (run_idx - 1) / max(total_runs, 1),
                fold,
                model,
            )
            result, predictions = _run_named_model(
                model,
                splits,
                dataset=dataset,
                fold=fold,
                split_seed=split_seed,
                graph_construction="train_only",
                prediction_cap=pred_cap,
                full_interactions=df,
            )
            rows.append(result)
            if not args.skip_cold_start:
                cold = _cold_start_rows(dataset, fold, split_seed, splits["train"], predictions, strata=strata_once)
                if not cold.empty:
                    cold_frames.append(cold)
            prediction_samples.append(predictions.head(5000))
        if ablation_enabled and full_log_ready:
            for model in ablation_models:
                if model not in models:
                    continue
                run_idx += 1
                logger.info(
                    "Baseline progress [%s] %d/%d (~%.0f%%) fold=%s model=%s graph_construction=full_log",
                    dataset,
                    run_idx,
                    total_runs,
                    100.0 * (run_idx - 1) / max(total_runs, 1),
                    fold,
                    model,
                )
                result, _predictions = _run_named_model(
                    model,
                    splits,
                    dataset=dataset,
                    fold=fold,
                    split_seed=split_seed,
                    graph_construction="full_log",
                    prediction_cap=pred_cap,
                    full_interactions=df,
                )
                rows.append(result)
        elif ablation_enabled and not full_log_ready and fold == 0:
            logger.warning(
                "Graph ablation skipped for %s: missing %s (run: python -m src.export_full_log_graph --config configs/%s.yaml)",
                dataset,
                _full_log_graph_paths(dataset)[0],
                dataset,
            )
    fold_results = pd.DataFrame(rows)
    results = _summarize_fold_results(fold_results, seed=args.seed, n_bootstrap=n_bootstrap)
    _merge_csv(Path("results/tables/baseline_fold_results.csv"), fold_results, dataset)
    _merge_csv(Path("results/tables/baseline_results.csv"), results, dataset)
    ab_summary = _summarize_graph_ablation(fold_results)
    if not ab_summary.empty:
        _merge_csv(Path("results/tables/graph_ablation_summary.csv"), ab_summary, dataset)
    if not args.skip_cold_start:
        cold_rows = pd.concat(cold_frames, ignore_index=True) if cold_frames else pd.DataFrame()
        _merge_csv(Path("results/tables/cold_start_metrics.csv"), cold_rows, dataset)
    else:
        logger.info("Cold-start CSV merge skipped (--skip-cold-start)")
    predictions_sample = pd.concat(prediction_samples, ignore_index=True)
    dump_csv(predictions_sample, Path("results/predictions") / f"{dataset}_diagnostic_predictions_sample.csv")


if __name__ == "__main__":
    main()
