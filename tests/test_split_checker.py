import pandas as pd
import pytest

from src.split_checker import assert_no_user_overlap, assert_temporal_ordering


def test_assert_no_user_overlap_raises_on_overlap():
    splits = {
        "train": pd.DataFrame({"user_id": [1], "timestamp": [1]}),
        "valid": pd.DataFrame({"user_id": [1], "timestamp": [2]}),
        "test": pd.DataFrame({"user_id": [2], "timestamp": [3]}),
    }
    with pytest.raises(AssertionError):
        assert_no_user_overlap(splits)


def test_temporal_ordering_enforced():
    splits = {
        "train": pd.DataFrame({"user_id": [1], "timestamp": [5]}),
        "valid": pd.DataFrame({"user_id": [1], "timestamp": [4]}),
        "test": pd.DataFrame({"user_id": [2], "timestamp": [10]}),
    }
    with pytest.raises(AssertionError):
        assert_temporal_ordering(splits)
