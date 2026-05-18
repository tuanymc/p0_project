"""Smoke tests for trained logistic leakage head used in graph ablation."""

import numpy as np
import pandas as pd

from src.baseline_runner import (
    _diagnostic_feature_arrays,
    _feature_keys_for_model,
    _fit_leakage_head_streaming,
    _predict_leakage_head,
)


def _tiny_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "user_id": rng.integers(0, 5, size=200),
            "item_id": rng.integers(0, 8, size=200),
            "kc_id": rng.integers(0, 6, size=200),
            "correct": rng.integers(0, 2, size=200),
        }
    )
    ev = pd.DataFrame(
        {
            "user_id": rng.integers(0, 5, size=40),
            "item_id": rng.integers(0, 8, size=40),
            "kc_id": rng.integers(0, 6, size=40),
            "correct": rng.integers(0, 2, size=40),
        }
    )
    return train, ev


def test_fit_predict_shapes_match_gkt_channels() -> None:
    train, ev = _tiny_split()
    keys = _feature_keys_for_model("gkt")
    assert keys == ("global", "kc", "graph")

    gr = {int(k): float(v) for k, v in zip(range(6), np.linspace(0.2, 0.8, 6))}
    w, b = _fit_leakage_head_streaming(
        train,
        "synthetic_unit",
        fold=0,
        graph_construction="train_only",
        full_interactions=None,
        keys=keys,
        graph_rates=gr,
        epochs=5,
        batch_size=32,
        lr=0.5,
        l2=1e-3,
        seed=123,
    )
    assert w.shape == (len(keys),)
    probs = _predict_leakage_head(
        train,
        ev,
        "synthetic_unit",
        fold=0,
        graph_construction="train_only",
        full_interactions=None,
        keys=keys,
        graph_rates=gr,
        w=w,
        b=b,
    )
    assert probs.shape == (len(ev),)
    assert np.all((probs > 0) & (probs < 1))


def test_feature_arrays_align_keys_subset() -> None:
    train, ev = _tiny_split()
    arrays = _diagnostic_feature_arrays(
        train,
        ev.iloc[:5],
        "synthetic_unit",
        fold=0,
        graph_construction="train_only",
        full_interactions=None,
        graph_rates={0: 0.4, 1: 0.6},
    )
    keys_dy = _feature_keys_for_model("dygkt")
    X = np.column_stack([arrays[k] for k in keys_dy])
    assert X.shape == (5, len(keys_dy))
