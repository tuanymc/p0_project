import pandas as pd
import pytest

from src.split_checker import assert_no_user_overlap, assert_temporal_ordering, fold_seeds, learner_based_folds


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


def test_fold_seeds_respects_n_folds_and_explicit_seeds():
    assert fold_seeds({"seed": 42, "n_folds": 3}) == [42, 43, 44]
    assert fold_seeds({"seeds": [7, 11]}) == [7, 11]


def test_learner_based_folds_tags_each_split_with_fold():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "item_id": range(10),
        "kc_id": range(10),
        "timestamp": range(10),
        "correct": [0, 1] * 5,
    })

    folds = learner_based_folds(df, (0.6, 0.2, 0.2), {"seed": 42, "n_folds": 2})

    assert [fold for fold, _seed, _splits in folds] == [0, 1]
    for fold, _seed, splits in folds:
        assert all(part["fold"].eq(fold).all() for part in splits.values())
