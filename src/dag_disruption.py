"""DAG disruption probes for train-only prerequisite edge sets."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io_utils import dump_csv, load_yaml

logger = logging.getLogger(__name__)


def _rng(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def apply_node_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Drop edges incident to a sampled set of nodes."""
    logger.info("Applying node_drop edges_shape=%s p=%s seed=%s", edges.shape, p, seed)
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    nodes = np.array(sorted(set(edges["src_kc"]) | set(edges["dst_kc"])))
    drop_nodes = set(nodes[rng.random(len(nodes)) < p])
    result = edges[~edges["src_kc"].isin(drop_nodes) & ~edges["dst_kc"].isin(drop_nodes)].reset_index(drop=True)
    logger.info("node_drop result_shape=%s", result.shape)
    return result


def apply_edge_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Drop a sampled set of edges."""
    logger.info("Applying edge_drop edges_shape=%s p=%s seed=%s", edges.shape, p, seed)
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    keep = rng.random(len(edges)) >= p
    result = edges.loc[keep].reset_index(drop=True)
    logger.info("edge_drop result_shape=%s", result.shape)
    return result


def apply_attribute_mask(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Mask non-structural edge attributes while preserving DAG endpoints."""
    logger.info("Applying attr_mask edges_shape=%s p=%s seed=%s", edges.shape, p, seed)
    result = edges.copy()
    if result.empty or p <= 0:
        return result
    rng = _rng(seed)
    mask = rng.random(len(result)) < p
    for col in [c for c in result.columns if c not in {"src_kc", "dst_kc"}]:
        if pd.api.types.is_numeric_dtype(result[col]):
            result.loc[mask, col] = np.nan
        else:
            result.loc[mask, col] = "masked"
    logger.info("attr_mask result_shape=%s", result.shape)
    return result


def apply_subgraph_sampling(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    """Keep a random-walk-grown node-induced subgraph."""
    logger.info("Applying subgraph edges_shape=%s p=%s seed=%s", edges.shape, p, seed)
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    nodes = np.array(sorted(set(edges["src_kc"]) | set(edges["dst_kc"])))
    target_size = min(len(nodes), max(1, int(np.ceil((1.0 - p) * len(nodes)))))

    adjacency = {node: set() for node in nodes}
    for src, dst in edges[["src_kc", "dst_kc"]].itertuples(index=False, name=None):
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    start = rng.choice(nodes)
    keep_nodes = {start}
    frontier = [start]
    while len(keep_nodes) < target_size:
        if not frontier:
            remaining = np.array([node for node in nodes if node not in keep_nodes])
            if len(remaining) == 0:
                break
            start = rng.choice(remaining)
            keep_nodes.add(start)
            frontier.append(start)
            continue
        current = rng.choice(np.array(frontier))
        candidates = np.array(sorted(adjacency[current] - keep_nodes))
        if len(candidates) == 0:
            frontier.remove(current)
            continue
        nxt = rng.choice(candidates)
        keep_nodes.add(nxt)
        frontier.append(nxt)

    result = edges[edges["src_kc"].isin(keep_nodes) & edges["dst_kc"].isin(keep_nodes)].reset_index(drop=True)
    logger.info("subgraph result_shape=%s", result.shape)
    return result


def compute_dag_disruption_rate(original: pd.DataFrame, augmented: pd.DataFrame) -> float:
    """Compute DDR = |E_pre lost or reversed| / |E_pre|."""
    logger.info("Computing DDR original_shape=%s augmented_shape=%s", original.shape, augmented.shape)
    original_edges = set(original[["src_kc", "dst_kc"]].itertuples(index=False, name=None))
    if not original_edges:
        return 0.0
    augmented_edges = set(augmented[["src_kc", "dst_kc"]].itertuples(index=False, name=None))
    lost = original_edges - augmented_edges
    reversed_edges = {(src, dst) for src, dst in original_edges if (dst, src) in augmented_edges and (src, dst) not in augmented_edges}
    ddr = len(lost | reversed_edges) / len(original_edges)
    logger.info("DDR=%s", ddr)
    return ddr


def sweep_ddr(
    edges: pd.DataFrame,
    augmentations: Sequence[str] = ("node_drop", "edge_drop", "attr_mask", "subgraph"),
    ps: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    seeds: Sequence[int] = (42,),
) -> pd.DataFrame:
    """Run DDR probes for configured augmentations."""
    logger.info("Sweeping DDR edges_shape=%s augmentations=%s ps=%s seeds=%s", edges.shape, augmentations, ps, seeds)
    fns = {
        "node_drop": apply_node_drop,
        "edge_drop": apply_edge_drop,
        "attr_mask": apply_attribute_mask,
        "subgraph": apply_subgraph_sampling,
    }
    rows = []
    for aug in augmentations:
        if aug not in fns:
            raise ValueError(f"Unknown augmentation: {aug}")
        for p in ps:
            for seed in seeds:
                augmented = fns[aug](edges, float(p), int(seed))
                rows.append({"augmentation": aug, "p": float(p), "seed": int(seed), "ddr": compute_dag_disruption_rate(edges, augmented)})
    result = pd.DataFrame(rows)
    logger.info("DDR sweep shape=%s", result.shape)
    return result


def _resolve_run_config(config: Path | None, edges: Path | None, seed: int) -> tuple[str, Path, Sequence[str], Sequence[float], Sequence[int]]:
    if config is None:
        return "default", edges or Path("data/processed/junyi/fold_0/e_pre_train_only.csv"), ("node_drop", "edge_drop", "attr_mask", "subgraph"), (0.05, 0.10, 0.20, 0.30), (seed,)
    cfg = load_yaml(config)
    dataset = cfg["dataset"]
    augmentation = cfg.get("augmentation", {})
    return (
        dataset,
        edges or Path("data/processed") / dataset / "fold_0" / "e_pre_train_only.csv",
        tuple(augmentation.get("methods", ["node_drop", "edge_drop", "attr_mask", "subgraph"])),
        tuple(float(p) for p in augmentation.get("ps", [0.05, 0.10, 0.20, 0.30])),
        tuple(int(s) for s in augmentation.get("seeds", [seed])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAG disruption probes")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    dataset, edges_path, augmentations, ps, seeds = _resolve_run_config(args.config, args.edges, args.seed)
    edges = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])
    result = sweep_ddr(edges, augmentations=augmentations, ps=ps, seeds=seeds)
    result.insert(0, "dataset", dataset)
    dataset_table = Path("results/tables") / f"{dataset}_dag_disruption.csv"
    dump_csv(result, dataset_table)
    combined_table = Path("results/tables/dag_disruption.csv")
    combined = result
    if combined_table.exists():
        previous = pd.read_csv(combined_table)
        previous = previous[previous["dataset"] != dataset]
        combined = pd.concat([previous, result], ignore_index=True)
    dump_csv(combined.sort_values(["dataset", "augmentation", "p", "seed"]), combined_table)
    fig_path = Path("results/figures") / f"fig_ddr_{dataset}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        for aug, part in result.groupby("augmentation"):
            plt.plot(part["p"], part["ddr"], marker="o", label=aug)
        plt.xlabel("p")
        plt.ylabel("DDR")
        plt.title(f"DDR sweep: {dataset}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    main()
