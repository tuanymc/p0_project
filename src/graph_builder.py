"""Train-only graph construction utilities for the P0 diagnostic protocol.

Every public graph builder accepts train-fold data only and should record edge
provenance for downstream leakage audit reports.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.io_utils import AUDIT_COLUMNS, dump_csv, load_interactions, load_yaml
from src.leakage_metrics import compute_leakage_row, merge_leakage_metrics_csv
from src.split_checker import learner_based_folds

logger = logging.getLogger(__name__)


def _assert_train_only(train: pd.DataFrame) -> None:
    if "fold" in train.columns:
        assert train["fold"].nunique(dropna=False) == 1, "Q-matrix must be built per fold."
    if "split" in train.columns:
        values = set(train["split"].dropna().astype(str).str.lower().unique())
        assert values <= {"train"}, "Graph construction must receive train rows only."


def build_q_matrix_from_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build an item--KC table from any canonical interaction frame."""
    logger.info("Building Q-matrix from interactions shape=%s", interactions.shape)
    q = interactions[["item_id", "kc_id"]].drop_duplicates().sort_values(["item_id", "kc_id"]).reset_index(drop=True)
    logger.info("Built Q-matrix shape=%s", q.shape)
    return q


def build_q_matrix_from_train(train: pd.DataFrame) -> pd.DataFrame:
    """Build an item-KC table from train interactions only."""
    logger.info("Building Q-matrix from train shape=%s", train.shape)
    _assert_train_only(train)
    return build_q_matrix_from_interactions(train)


def infer_similarity_edges_from_q_matrix(
    q_matrix: pd.DataFrame,
    method: Literal["jaccard", "pmi"] = "jaccard",
    threshold: float = 0.1,
    *,
    source_prefix: str = "train",
) -> pd.DataFrame:
    """Infer KC similarity edges from an item--KC co-occurrence table."""
    logger.info(
        "Inferring similarity edges q_shape=%s method=%s threshold=%s source_prefix=%s",
        q_matrix.shape,
        method,
        threshold,
        source_prefix,
    )
    if method not in {"jaccard", "pmi"}:
        raise ValueError("method must be 'jaccard' or 'pmi'")
    item_sets = q_matrix.groupby("kc_id")["item_id"].apply(lambda s: set(s.astype(int))).to_dict()
    n_items = max(1, q_matrix["item_id"].nunique())
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
            edges.append({"src_kc": src, "dst_kc": dst, "weight": weight, "source": f"{source_prefix}_{method}"})
    result = pd.DataFrame(edges, columns=["src_kc", "dst_kc", "weight", "source"])
    logger.info("Inferred similarity edges shape=%s", result.shape)
    return result


def infer_similarity_edges_from_train(
    train: pd.DataFrame,
    q_train: pd.DataFrame,
    method: Literal["jaccard", "pmi"] = "jaccard",
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Infer KC similarity edges from train-fold co-occurrence only."""
    logger.info("Inferring similarity edges train_shape=%s q_shape=%s method=%s threshold=%s", train.shape, q_train.shape, method, threshold)
    _assert_train_only(train)
    return infer_similarity_edges_from_q_matrix(q_train, method=method, threshold=threshold, source_prefix="train")


def infer_prerequisites_from_interactions(
    interactions: pd.DataFrame,
    q_matrix: pd.DataFrame,
    max_edges: int = 5000,
    top_k_per_node: int = 10,
    support_quantile: float = 0.90,
    *,
    source_tag: str = "temporal_precedence",
) -> pd.DataFrame:
    """Infer prerequisite edges from temporal KC transitions on ``interactions``."""
    logger.info(
        "Inferring prerequisite edges interactions_shape=%s q_shape=%s source=%s",
        interactions.shape,
        q_matrix.shape,
        source_tag,
    )
    transitions = []
    for _user_id, part in interactions.groupby("user_id", sort=False):
        ordered_part = part.sort_values("timestamp")
        kcs = ordered_part["kc_id"].to_numpy()
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
    min_support = max(2, int(np.ceil(counts["support"].quantile(support_quantile)))) if len(counts) > 10 else 1
    edges = counts[counts["support"] >= min_support].copy()
    if edges.empty:
        edges = counts.nlargest(min(200, len(counts)), "support").copy()
    edges = edges.sort_values(["src_kc", "support"], ascending=[True, False])
    edges = edges.groupby("src_kc", group_keys=False).head(top_k_per_node)
    edges = edges.nlargest(min(max_edges, len(edges)), "support").copy()
    max_support = max(1, edges["support"].max())
    edges["weight"] = edges["support"] / max_support
    edges["source"] = source_tag
    result = edges[["src_kc", "dst_kc", "weight", "source"]].sort_values(["src_kc", "dst_kc"]).reset_index(drop=True)
    logger.info("Inferred prerequisite edges shape=%s min_support=%s", result.shape, min_support)
    return result


def infer_prerequisites_from_train(
    train: pd.DataFrame,
    q_train: pd.DataFrame,
    max_edges: int = 5000,
    top_k_per_node: int = 10,
    support_quantile: float = 0.90,
) -> pd.DataFrame:
    """Infer prerequisite edges from train-only temporal KC transitions."""
    logger.info("Inferring prerequisite edges train_shape=%s q_shape=%s", train.shape, q_train.shape)
    _assert_train_only(train)
    return infer_prerequisites_from_interactions(
        train,
        q_train,
        max_edges=max_edges,
        top_k_per_node=top_k_per_node,
        support_quantile=support_quantile,
        source_tag="train_temporal_precedence",
    )


def evaluate_inferred_against_ground_truth(
    inferred: pd.DataFrame,
    expert: pd.DataFrame,
    top_k_list: Sequence[int],
) -> pd.DataFrame:
    """Score inferred directed KC edges against an expert graph at several @k cuts.

    *Top-k* rows are taken from ``inferred`` by descending ``support`` (ties keep
    first occurrence). For that subset:
    - *directed* hit: ``(src_kc, dst_kc)`` equals an expert row (direction matters).
    - *undirected* hit: the unordered pair appears in the expert (either orientation).

    Directed precision is ``directed_hits / undirected_hits`` (0 if no undirected
    overlap). Directed recall is ``directed_hits / n_expert_edges``.
    """
    if "support" not in inferred.columns:
        raise ValueError("inferred must include a 'support' column for @k evaluation")

    expert_direct = set(
        zip(expert["src_kc"].astype(int), expert["dst_kc"].astype(int), strict=True)
    )
    expert_undirected = {
        frozenset(pair)
        for pair in zip(expert["src_kc"].astype(int), expert["dst_kc"].astype(int), strict=True)
    }
    n_expert = len(expert)

    rows: list[dict[str, float | int]] = []
    for k in top_k_list:
        k_int = int(k)
        if k_int <= 0:
            subset = inferred.iloc[0:0]
        else:
            subset = inferred.nlargest(k_int, "support", keep="first")

        directed_hits = 0
        undirected_hits = 0
        for r in subset.itertuples(index=False):
            s = int(getattr(r, "src_kc"))
            d = int(getattr(r, "dst_kc"))
            if (s, d) in expert_direct:
                directed_hits += 1
            if frozenset((s, d)) in expert_undirected:
                undirected_hits += 1

        directed_precision = (directed_hits / undirected_hits) if undirected_hits else 0.0
        directed_recall = (directed_hits / n_expert) if n_expert else 0.0
        rows.append({
            "k": k_int,
            "directed_hits": directed_hits,
            "undirected_hits": undirected_hits,
            "directed_precision": float(directed_precision),
            "directed_recall": float(directed_recall),
        })

    return pd.DataFrame(rows)


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
    if edges.empty:
        return
    path = Path("logs/leakage_audit_log.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    rows = []
    for row in edges.itertuples(index=False):
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "edge_type": edge_type,
            "src_kc": getattr(row, "src_kc"),
            "dst_kc": getattr(row, "dst_kc"),
            "source_fold": "train",
            "train_only_flag": True,
        })
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    logger.info("Appended %s audit rows for %s/%s", len(rows), dataset, edge_type)


def _merge_stats_rows(path: Path, rows: list[dict], dataset: str) -> None:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty and "dataset" in current.columns:
        current = current[current["dataset"] != dataset]
    dump_csv(pd.concat([current, pd.DataFrame(rows)], ignore_index=True), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only KC graph edges")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if hasattr(pd.options.mode, "copy_on_write"):
        pd.options.mode.copy_on_write = True

    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet")))
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    graph_cfg = cfg.get("graph", {})
    stats_rows = []
    leakage_rows: list[dict[str, float | int | str]] = []
    train_ratio = float(ratios[0]) if ratios else 0.7
    for fold, split_seed, splits in learner_based_folds(df, ratios, cfg.get("split", {}), default_seed=args.seed):
        train = splits["train"].copy()
        train["split"] = "train"
        train["fold"] = fold
        q_train = build_q_matrix_from_train(train)
        pre = infer_prerequisites_from_train(
            train,
            q_train,
            max_edges=int(graph_cfg.get("e_pre_max_edges", 5000)),
            top_k_per_node=int(graph_cfg.get("e_pre_top_k_per_node", 10)),
            support_quantile=float(graph_cfg.get("e_pre_support_quantile", 0.90)),
        )
        sim = infer_similarity_edges_from_train(
            train,
            q_train,
            method=graph_cfg.get("e_sim_method", "jaccard"),
            threshold=float(graph_cfg.get("e_sim_threshold", 0.1)),
        )
        out_dir = Path("data/processed") / dataset / f"fold_{fold}"
        dump_csv(pre, out_dir / "e_pre_train_only.csv")
        dump_csv(sim, out_dir / "e_sim_train_only.csv")
        dump_csv(pre, out_dir / "edges_train_only.csv")
        _audit_edges(pre, dataset, fold, "E_pre")
        _audit_edges(sim, dataset, fold, "E_sim")
        leakage_rows.append(
            compute_leakage_row(
                dataset=dataset,
                fold=fold,
                splits=splits,
                pre_df=pre,
                sim_df=sim,
                q_train=q_train,
                train_ratio=train_ratio,
            )
        )
        stats_rows.append({
            "dataset": dataset,
            "fold": fold,
            "split_seed": split_seed,
            "n_prerequisite_edges": len(pre),
            "n_similarity_edges": len(sim),
            "n_kcs": train["kc_id"].nunique(),
        })
        del splits, train, q_train, pre, sim
        gc.collect()
    _merge_stats_rows(Path("results/tables/graph_stats.csv"), stats_rows, dataset)
    merge_leakage_metrics_csv(leakage_rows, dataset)
    logger.info("Graph build report written")


if __name__ == "__main__":
    main()
