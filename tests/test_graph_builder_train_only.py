import pandas as pd
import pytest

from src.graph_builder import evaluate_inferred_against_ground_truth, infer_similarity_edges_from_train


def test_similarity_rejects_multiple_fold_values():
    train = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 11],
        "kc_id": [100, 101],
        "timestamp": [1, 2],
        "correct": [1, 0],
        "fold": [0, 1],
    })
    q_train = train[["item_id", "kc_id"]]
    with pytest.raises(AssertionError):
        infer_similarity_edges_from_train(train, q_train)


def test_ground_truth_cv_reports_directed_and_undirected_hits():
    inferred = pd.DataFrame({
        "src_kc": [1, 3, 5],
        "dst_kc": [2, 4, 6],
        "support": [10, 9, 8],
        "weight": [1.0, 0.9, 0.8],
    })
    expert = pd.DataFrame({
        "src_kc": [1, 4, 7],
        "dst_kc": [2, 3, 8],
    })

    result = evaluate_inferred_against_ground_truth(inferred, expert, [2])
    row = result.iloc[0]

    assert row["directed_hits"] == 1
    assert row["undirected_hits"] == 2
    assert row["directed_precision"] == 0.5
    assert row["directed_recall"] == pytest.approx(1 / 3)
