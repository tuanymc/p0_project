"""Export graphs inferred from the full interaction log (leaky ablation upper bound).

Writes ``e_pre.csv`` and ``e_sim.csv`` under ``data/processed/<dataset>/full_log/``.
These artefacts feed ``baseline_runner`` when ``graph_construction='full_log'`` for graph-augmented diagnostics listed in ``graph_ablation.models``.

This path deliberately ignores train/validation/test boundaries and must **not** be used
for leakage-controlled reporting except as an explicit ablation contrast.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.graph_builder import (
    build_q_matrix_from_interactions,
    infer_prerequisites_from_interactions,
    infer_similarity_edges_from_q_matrix,
)
from src.io_utils import dump_csv, load_interactions, load_yaml

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full-log prerequisite/similarity graphs for ablation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    processed = Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet"))
    df = load_interactions(processed)
    graph_cfg = cfg.get("graph", {})
    q = build_q_matrix_from_interactions(df)
    pre = infer_prerequisites_from_interactions(
        df,
        q,
        max_edges=int(graph_cfg.get("e_pre_max_edges", 5000)),
        top_k_per_node=int(graph_cfg.get("e_pre_top_k_per_node", 10)),
        support_quantile=float(graph_cfg.get("e_pre_support_quantile", 0.90)),
        source_tag="full_log_temporal_precedence",
    )
    sim = infer_similarity_edges_from_q_matrix(
        q,
        method=graph_cfg.get("e_sim_method", "jaccard"),
        threshold=float(graph_cfg.get("e_sim_threshold", 0.1)),
        source_prefix="full_log",
    )
    out_dir = Path("data/processed") / dataset / "full_log"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(pre, out_dir / "e_pre.csv")
    dump_csv(sim, out_dir / "e_sim.csv")
    logger.info("Wrote full-log graphs to %s (|E_pre|=%s, |E_sim|=%s)", out_dir, len(pre), len(sim))


if __name__ == "__main__":
    main()
