"""Train-only graph construction utilities for the P0 diagnostic protocol.

Every public graph builder accepts train-fold data only and should record edge
provenance for downstream leakage audit reports.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.io_utils import append_audit_row, dump_csv, load_interactions, load_yaml
from src.split_checker import learner_based_split

logger = logging.getLogger(__name__)


def _assert_train_only(train: pd.DataFrame) -> None:
    if "fold" in train.columns:
        assert train["fold"].nunique(dropna=False) == 1, "Q-matrix must be built per fold."
    if "split" in train.columns:
        values = set(train["split"].dropna().astype(str).str.lower().unique())
        assert values <= {"train"}, "Graph construction must receive train rows only."


def build_q_matrix_from_train(train: pd.DataFrame) -> pd.DataFrame:
    """Build an item-KC table from train interactions only."""
    logger.info("Building Q-matrix from train shape=%s", train.shape)
    _assert_train_only(train)
    q_train = train[["item_id", "kc_id"]].drop_duplicates().sort_values(["item_id", "kc_id"]).reset_index(drop=True)
    logger.info("Built Q-matrix shape=%s", q_train.shape)
    return q_train


def infer_similarity_edges_from_train(
    train: pd.DataFrame,
    q_train: pd.DataFrame,
    method: Literal["jaccard", "pmi"] = "jaccard",
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Infer KC similarity edges from train-fold co-occurrence only."""
    logger.info("Inferring similarity edges train_shape=%s q_shape=%s method=%s threshold=%s", train.shape, q_train.shape, method, threshold)
    _assert_train_only(train)
    if method not in {"jaccard", "pmi"}:
        raise ValueError("method must be 'jaccard' or 'pmi'")
    item_sets = q_train.groupby("kc_id")["item_id"].apply(lambda s: set(s.astype(int))).to_dict()
    n_items = max(1, q_train["item_id"].nunique())
    edges: list[dict[str, object]] = []
    for src, dst in itertools.permutations(sorted(item_sets), 2):
        inter = len(item_sets[src] & item_sets[dst])
        if inter == 0:
            continue
        if method == "jaccard":
            union = len(item_sets[src] | item_sets[dst])
            weight = inter / union if union else 0.0
        else:
            p_xy = inter / n_items
            p_x = len(item_sets[src]) / n_items
            p_y = len(item_sets[dst]) / n_items
            weight = float(np.log(p_xy / (p_x * p_y))) if p_x and p_y and p_xy else 0.0
        if weight >= threshold:
            edges.append({"src_kc": src, "dst_kc": dst, "weight": weight, "source": f"train_{method}"})
    result = pd.DataFrame(edges, columns=["src_kc", "dst_kc", "weight", "source"])
    logger.info("Inferred similarity edges shape=%s", result.shape)
    return result


def infer_prerequisites_from_train(train: pd.DataFrame, q_train: pd.DataFrame) -> pd.DataFrame:
    """Infer prerequisite edges from train-only temporal KC transitions."""
    logger.info("Inferring prerequisite edges train_shape=%s q_shape=%s", train.shape, q_train.shape)
    _assert_train_only(train)
    ordered = train.sort_values(["user_id", "timestamp"])
    transitions = []
    for _user_id, part in ordered.groupby("user_id", sort=False):
        kcs = part["kc_id"].to_numpy()
        if len(kcs) < 2:
            continue
        src = kcs[:-1]
        dst = kcs[1:]
        mask = src != dst
        if mask.any():
            transitions.append(pd.DataFrame({"src_kc": src[mask], "dst_kc": dst[mask]}))
    if not transitions:
        return pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
    counts = pd.concat(transitions, ignore_index=True).value_counts(["src_kc", "dst_kc"]).rename("support").reset_index()
    min_support = max(2, int(np.ceil(counts["support"].quantile(0.75)))) if len(counts) > 10 else 1
    edges = counts[counts["support"] >= min_support].copy()
    if edges.empty:
        edges = counts.nlargest(min(200, len(counts)), "support").copy()
    max_support = max(1, edges["support"].max())
    edges["weight"] = edges["support"] / max_support
    edges["source"] = "train_temporal_precedence"
    result = edges[["src_kc", "dst_kc", "weight", "source"]].sort_values(["src_kc", "dst_kc"]).reset_index(drop=True)
    logger.info("Inferred prerequisite edges shape=%s min_support=%s", result.shape, min_support)
    return result


def load_ground_truth_dag(dataset_name: str) -> pd.DataFrame:
    """Load an independent prerequisite DAG for datasets that provide one."""
    logger.info("Loading ground-truth DAG for dataset=%s", dataset_name)
    path = Path("data/raw") / dataset_name / f"{dataset_name}_dag.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    logger.info("Loaded ground-truth DAG shape=%s", df.shape)
    return df


def dataset_has_independent_prerequisite_graph(dataset_name: str) -> bool:
    """Report whether the named dataset is expected to provide a DAG."""
    logger.info("Checking independent DAG availability for %s", dataset_name)
    has_dag = dataset_name.lower() in {"junyi", "xes3g5m"}
    logger.info("Independent DAG availability for %s: %s", dataset_name, has_dag)
    return has_dag


def _audit_edges(edges: pd.DataFrame, dataset: str, fold: int, edge_type: str) -> None:
    for row in edges.itertuples(index=False):
        append_audit_row({
            "dataset": dataset,
            "fold": fold,
            "edge_type": edge_type,
            "src_kc": getattr(row, "src_kc"),
            "dst_kc": getattr(row, "dst_kc"),
            "source_fold": "train",
            "train_only_flag": True,
        })


def _merge_stats_row(path: Path, row: dict) -> None:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty and "dataset" in current.columns:
        current = current[current["dataset"] != row["dataset"]]
    dump_csv(pd.concat([current, pd.DataFrame([row])], ignore_index=True), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only KC graph edges")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet")))
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    train = learner_based_split(df, ratios, seed=cfg.get("split", {}).get("seed", args.seed))["train"]
    train["split"] = "train"
    train["fold"] = 0
    q_train = build_q_matrix_from_train(train)
    pre = infer_prerequisites_from_train(train, q_train)
    sim = infer_similarity_edges_from_train(
        train,
        q_train,
        method=cfg.get("graph", {}).get("e_sim_method", "jaccard"),
        threshold=float(cfg.get("graph", {}).get("e_sim_threshold", 0.1)),
    )
    out_dir = Path("data/processed") / dataset / "fold_0"
    dump_csv(pre, out_dir / "e_pre_train_only.csv")
    dump_csv(sim, out_dir / "e_sim_train_only.csv")
    dump_csv(pre, out_dir / "edges_train_only.csv")
    _audit_edges(pre, dataset, 0, "E_pre")
    _audit_edges(sim, dataset, 0, "E_sim")
    _merge_stats_row(Path("results/tables/graph_stats.csv"), {
        "dataset": dataset,
        "fold": 0,
        "n_prerequisite_edges": len(pre),
        "n_similarity_edges": len(sim),
        "n_kcs": train["kc_id"].nunique(),
    })
    logger.info("Graph build report written")


if __name__ == "__main__":
    main()
