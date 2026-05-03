"""DAG audit utilities for prerequisite graphs built from train-fold evidence."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from src.io_utils import dump_csv, load_yaml

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


def detect_cycles(edges: pd.DataFrame) -> list[list[int]]:
    """Detect directed cycles in the edge set."""
    logger.info("Detecting cycles for edges shape=%s", edges.shape)
    cycles = [list(cycle) for cycle in nx.simple_cycles(_to_graph(edges))]
    logger.info("Detected %s cycles", len(cycles))
    return cycles


def prune_cycles(edges: pd.DataFrame, confidence_col: str = "weight") -> tuple[pd.DataFrame, list[dict]]:
    """Remove the lowest-confidence edge from each detected cycle."""
    logger.info("Pruning cycles edges_shape=%s confidence_col=%s", edges.shape, confidence_col)
    pruned = edges.copy().reset_index(drop=True)
    if confidence_col not in pruned.columns:
        pruned[confidence_col] = 1.0
    log: list[dict] = []
    while True:
        cycles = detect_cycles(pruned)
        if not cycles:
            break
        cycle = cycles[0]
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
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


def _resolve_dataset_and_edges(config: Path | None, edges: Path | None) -> tuple[str, Path]:
    if config is None:
        return "default", edges or Path("data/processed/junyi/fold_0/e_pre_train_only.csv")
    cfg = load_yaml(config)
    dataset = cfg["dataset"]
    return dataset, edges or Path("data/processed") / dataset / "fold_0" / "e_pre_train_only.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prerequisite DAG edges")
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    dataset, edges_path = _resolve_dataset_and_edges(args.config, args.edges)
    edges = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])
    report = audit_dag(edges)
    out = Path("results/reports") / f"{dataset}_dag_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# DAG Audit Report: {dataset}\n\n"
        f"- source_edges: `{edges_path}`\n"
        f"- nodes: {report.n_nodes}\n"
        f"- edges: {report.n_edges}\n"
        f"- cycles_before: {report.n_cycles_before}\n"
        f"- cycles_after: {report.n_cycles_after}\n"
        f"- topo_sort_passed: {report.topo_sort_passed}\n",
        encoding="utf-8",
    )
    summary = pd.DataFrame([{
        "dataset": dataset,
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
        "n_roots": report.n_roots,
        "n_leaves": report.n_leaves,
        "n_cycles_before": report.n_cycles_before,
        "n_cycles_after": report.n_cycles_after,
        "topo_sort_passed": report.topo_sort_passed,
    }])
    summary_path = Path("results/tables/dag_audit_summary.csv")
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous[previous["dataset"] != dataset]
        summary = pd.concat([previous, summary], ignore_index=True)
    dump_csv(summary.sort_values("dataset"), summary_path)
    if report.pruning_log:
        dump_csv(pd.DataFrame(report.pruning_log), Path("results/reports") / f"{dataset}_dag_pruning_log.csv")


if __name__ == "__main__":
    main()
