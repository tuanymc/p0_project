import pandas as pd
import pytest

from src.graph_builder import infer_similarity_edges_from_train


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
