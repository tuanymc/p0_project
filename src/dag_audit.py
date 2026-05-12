"""DAG audit utilities for prerequisite graphs built from train-fold evidence."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from src.io_utils import dump_csv, load_yaml
from src.split_checker import fold_seeds

logger = logging.getLogger(__name__)


@dataclass
class DAGReport:
    n_nodes: int
    n_edges: int
    n_roots: int
    n_leaves: int
    n_cycles_before: int
    n_cycles_after: int
    topo_sort_passed: bool
    pruning_log: list[dict]


def _to_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    if edges.empty:
        return graph
    graph.add_edges_from(edges[["src_kc", "dst_kc"]].itertuples(index=False, name=None))
    return graph


def topological_sort(edges: pd.DataFrame) -> list[int]:
    """Return a topological order for an acyclic edge set."""
    logger.info("Topological sort for edges shape=%s", edges.shape)
    order = list(nx.topological_sort(_to_graph(edges)))
    logger.info("Topological sort length=%s", len(order))
    return order


def _find_one_cycle_edges(edges: pd.DataFrame) -> list[tuple[int, int]]:
    graph = _to_graph(edges)
    try:
        cycle = nx.find_cycle(graph, orientation="original")
    except nx.NetworkXNoCycle:
        return []
    return [(src, dst) for src, dst, _orientation in cycle]


def detect_cycles(edges: pd.DataFrame, max_cycles: int = 100) -> list[list[int]]:
    """Detect representative directed cycles without enumerating all cycles.

    Enumerating all simple cycles can be exponential on dense prerequisite
    graphs, so this diagnostic returns up to ``max_cycles`` representative
    cycles found by iterative pruning.
    """
    logger.info("Detecting representative cycles for edges shape=%s max_cycles=%s", edges.shape, max_cycles)
    working = edges.copy().reset_index(drop=True)
    cycles: list[list[int]] = []
    while len(cycles) < max_cycles:
        cycle_edges = _find_one_cycle_edges(working)
        if not cycle_edges:
            break
        cycles.append([src for src, _dst in cycle_edges])
        src, dst = cycle_edges[0]
        drop_idx = working[(working["src_kc"] == src) & (working["dst_kc"] == dst)].index
        if len(drop_idx) == 0:
            break
        working = working.drop(index=drop_idx[0]).reset_index(drop=True)
    logger.info("Detected %s representative cycles", len(cycles))
    return cycles


def prune_cycles(edges: pd.DataFrame, confidence_col: str = "weight") -> tuple[pd.DataFrame, list[dict]]:
    """Remove the lowest-confidence edge from each detected cycle."""
    logger.info("Pruning cycles edges_shape=%s confidence_col=%s", edges.shape, confidence_col)
    pruned = edges.copy().reset_index(drop=True)
    if confidence_col not in pruned.columns:
        pruned[confidence_col] = 1.0
    log: list[dict] = []
    while True:
        cycle_edges = _find_one_cycle_edges(pruned)
        if not cycle_edges:
            break
        candidates = pruned[pruned.apply(lambda r: (r["src_kc"], r["dst_kc"]) in cycle_edges, axis=1)].copy()
        idx = candidates[confidence_col].astype(float).idxmin()
        removed = pruned.loc[idx].to_dict()
        removed["reason"] = "cycle_prune_lowest_confidence"
        log.append(removed)
        pruned = pruned.drop(index=idx).reset_index(drop=True)
    logger.info("Pruned edge count=%s removed=%s", len(pruned), len(log))
    return pruned, log


def audit_dag(edges: pd.DataFrame) -> DAGReport:
    """Audit DAG shape and cycle pruning outcome."""
    logger.info("Auditing DAG edges_shape=%s", edges.shape)
    cycles_before = detect_cycles(edges)
    pruned, pruning_log = prune_cycles(edges)
    cycles_after = detect_cycles(pruned)
    graph = _to_graph(pruned)
    try:
        topological_sort(pruned)
        topo_passed = True
    except nx.NetworkXUnfeasible:
        topo_passed = False
    report = DAGReport(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_roots=sum(1 for node in graph.nodes if graph.in_degree(node) == 0),
        n_leaves=sum(1 for node in graph.nodes if graph.out_degree(node) == 0),
        n_cycles_before=len(cycles_before),
        n_cycles_after=len(cycles_after),
        topo_sort_passed=topo_passed,
        pruning_log=pruning_log,
    )
    logger.info("DAG report=%s", report)
    return report


def _resolve_dataset_folds_and_edges(config: Path | None, edges: Path | None, seed: int) -> tuple[str, list[tuple[int, Path]]]:
    if config is None:
        return "default", [(0, edges or Path("data/processed/junyi/fold_0/e_pre_train_only.csv"))]
    cfg = load_yaml(config)
    dataset = cfg["dataset"]
    if edges is not None:
        return dataset, [(0, edges)]
    folds = [(fold, Path("data/processed") / dataset / f"fold_{fold}" / "e_pre_train_only.csv") for fold, _ in enumerate(fold_seeds(cfg.get("split", {}), seed))]
    return dataset, folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prerequisite DAG edges")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    dataset, fold_edges = _resolve_dataset_folds_and_edges(args.config, args.edges, args.seed)
    out = Path("results/reports") / f"{dataset}_dag_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# DAG Audit Report: {dataset}", ""]
    rows = []
    pruning_logs = []
    for fold, edges_path in fold_edges:
        edges = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])
        report = audit_dag(edges)
        lines.extend([
            f"## Fold {fold}",
            f"- source_edges: `{edges_path}`",
            f"- nodes: {report.n_nodes}",
            f"- edges: {report.n_edges}",
            f"- cycles_before: {report.n_cycles_before}",
            f"- cycles_after: {report.n_cycles_after}",
            f"- topo_sort_passed: {report.topo_sort_passed}",
            "",
        ])
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "n_roots": report.n_roots,
            "n_leaves": report.n_leaves,
            "n_cycles_before": report.n_cycles_before,
            "n_cycles_after": report.n_cycles_after,
            "topo_sort_passed": report.topo_sort_passed,
        })
        for item in report.pruning_log:
            item = dict(item)
            item["dataset"] = dataset
            item["fold"] = fold
            pruning_logs.append(item)
    out.write_text("\n".join(lines), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary_path = Path("results/tables/dag_audit_summary.csv")
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous[previous["dataset"] != dataset]
        summary = pd.concat([previous, summary], ignore_index=True)
    dump_csv(summary.sort_values(["dataset", "fold"]), summary_path)
    if pruning_logs:
        dump_csv(pd.DataFrame(pruning_logs), Path("results/reports") / f"{dataset}_dag_pruning_log.csv")


if __name__ == "__main__":
    main()
