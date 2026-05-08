import pandas as pd

from src.dag_disruption import (
    apply_attribute_mask,
    apply_edge_drop,
    apply_node_drop,
    apply_subgraph_sampling,
    compute_dag_disruption_rate,
)


def _path_graph(n_nodes: int) -> pd.DataFrame:
    return pd.DataFrame({
        "src_kc": list(range(n_nodes - 1)),
        "dst_kc": list(range(1, n_nodes)),
        "weight": [1.0] * (n_nodes - 1),
    })


def _edge_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return set(edges[["src_kc", "dst_kc"]].itertuples(index=False, name=None))


def _node_set(edges: pd.DataFrame) -> set[int]:
    return set(edges["src_kc"]) | set(edges["dst_kc"])


def test_ddr_zero_for_identity_augmentation():
    edges = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "weight": [1.0, 1.0]})
    augmented = apply_edge_drop(edges, p=0.0, seed=42)
    assert compute_dag_disruption_rate(edges, augmented) == 0.0


def test_ddr_one_when_all_edges_are_dropped():
    edges = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "weight": [1.0, 1.0]})
    augmented = edges.iloc[0:0].copy()
    assert compute_dag_disruption_rate(edges, augmented) == 1.0


def test_attribute_mask_preserves_ddr_zero():
    edges = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "weight": [1.0, 1.0]})
    augmented = apply_attribute_mask(edges, p=1.0, seed=42)
    assert compute_dag_disruption_rate(edges, augmented) == 0.0


def test_reversed_edge_counts_as_disruption():
    original = pd.DataFrame({"src_kc": [1], "dst_kc": [2], "weight": [1.0]})
    augmented = pd.DataFrame({"src_kc": [2], "dst_kc": [1], "weight": [1.0]})
    assert compute_dag_disruption_rate(original, augmented) == 1.0


def test_subgraph_sampling_keeps_contiguous_path_subgraph():
    edges = _path_graph(10)
    augmented = apply_subgraph_sampling(edges, p=0.4, seed=42)

    kept_nodes = sorted(_node_set(augmented))
    assert kept_nodes == list(range(kept_nodes[0], kept_nodes[-1] + 1))


def test_subgraph_sampling_hits_target_size_on_path_graph():
    edges = _path_graph(10)
    augmented = apply_subgraph_sampling(edges, p=0.4, seed=42)

    assert len(_node_set(augmented)) == 6


def test_subgraph_sampling_is_seed_deterministic():
    edges = _path_graph(12)

    first = apply_subgraph_sampling(edges, p=0.5, seed=17)
    second = apply_subgraph_sampling(edges, p=0.5, seed=17)

    pd.testing.assert_frame_equal(first, second)


def test_subgraph_sampling_is_distinct_from_node_drop():
    edges = _path_graph(12)

    subgraph = apply_subgraph_sampling(edges, p=0.5, seed=42)
    node_drop = apply_node_drop(edges, p=0.5, seed=42)

    assert _edge_set(subgraph) != _edge_set(node_drop)
