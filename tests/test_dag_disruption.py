import pandas as pd

from src.dag_disruption import apply_attribute_mask, apply_edge_drop, compute_dag_disruption_rate


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
