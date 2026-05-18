"""Protocol KC graphs → row-normalised dense matrices for pyKT GKT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def edges_to_gkt_matrix(num_c: int, edge_paths: list[Path], kc_index_map: dict[int, int]) -> np.ndarray:
    """Union directed edges from CSVs ``src_kc,dst_kc``; symmetrise counts then row-normalise."""
    adj = np.zeros((num_c, num_c), dtype=np.float64)
    for p in edge_paths:
        if not p.exists():
            continue
        edges = pd.read_csv(p)
        if edges.empty or not {"src_kc", "dst_kc"}.issubset(edges.columns):
            continue
        for _, row in edges.iterrows():
            a = kc_index_map.get(int(row["src_kc"]))
            b = kc_index_map.get(int(row["dst_kc"]))
            if a is None or b is None or a < 0 or b < 0 or a >= num_c or b >= num_c:
                continue
            adj[a, b] += 1.0
            adj[b, a] += 1.0
    np.fill_diagonal(adj, 0.0)
    rowsum = adj.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    adj = adj / rowsum
    return adj.astype(np.float32)


def write_gkt_graph_npz(out_path: Path, matrix: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, matrix=matrix)
