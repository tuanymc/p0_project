import pandas as pd

from src.preprocess import _normalise_xes_sequence_chunk


def test_xes_sequence_chunk_expands_and_filters_padding():
    raw = pd.DataFrame({
        "uid": ["u1"],
        "questions": ["10,11,-1"],
        "concepts": ["100,101,-1"],
        "responses": ["1,0,-1"],
        "timestamps": ["1600000000000,1600000001000,-1"],
        "selectmasks": ["1,1,-1"],
    })

    result = _normalise_xes_sequence_chunk(raw)

    assert list(result.columns) == ["user_id", "item_id", "kc_id", "timestamp", "correct"]
    assert len(result) == 2
    assert result["timestamp"].tolist() == [1600000000, 1600000001]
    assert result["correct"].tolist() == [1, 0]
