import pandas as pd
import pytest

from src.leakage_metrics import compute_ecr, compute_ecr_flag, compute_eoc, compute_leakage_row, compute_tbvr


def test_ecr_one_when_held_out_repeats_transition():
    q_train = pd.DataFrame({"item_id": [1, 2], "kc_id": [10, 20]})
    pre_df = pd.DataFrame({"src_kc": [10], "dst_kc": [20], "weight": [1.0], "source": ["train"]})
    sim_df = pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
    held_df = pd.DataFrame({
        "user_id": [99, 99],
        "item_id": [3, 4],
        "kc_id": [10, 20],
        "timestamp": [1, 2],
        "correct": [1, 0],
    })
    assert compute_ecr(pre_df, sim_df, held_df, q_train) == pytest.approx(1.0)


def test_ecr_zero_when_no_held_evidence():
    q_train = pd.DataFrame({"item_id": [1], "kc_id": [10]})
    pre_df = pd.DataFrame({"src_kc": [10], "dst_kc": [20], "weight": [1.0], "source": ["train"]})
    sim_df = pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
    held_df = pd.DataFrame({
        "user_id": [99],
        "item_id": [5],
        "kc_id": [99],
        "timestamp": [1],
        "correct": [1],
    })
    assert compute_ecr(pre_df, sim_df, held_df, q_train) == pytest.approx(0.0)


def test_ecr_flag_zero_when_disjoint_learners():
    splits = {
        "train": pd.DataFrame({"user_id": [1]}),
        "valid": pd.DataFrame({"user_id": [2]}),
        "test": pd.DataFrame({"user_id": [3]}),
    }
    assert compute_ecr_flag(splits) == pytest.approx(0.0)


def test_ecr_flag_one_when_learners_overlap():
    splits = {
        "train": pd.DataFrame({"user_id": [1, 2]}),
        "valid": pd.DataFrame({"user_id": [2]}),
        "test": pd.DataFrame({"user_id": [99]}),
    }
    assert compute_ecr_flag(splits) == pytest.approx(1.0)


def test_tbvr_positive_when_future_tail_supports_retained_edge():
    train_df = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_id": [1, 2, 3, 4],
        "kc_id": [10, 20, 10, 20],
        "timestamp": [1, 2, 3, 4],
        "correct": [1, 1, 1, 1],
    })
    pre_df = pd.DataFrame({"src_kc": [10], "dst_kc": [20], "weight": [1.0], "source": ["train"]})
    tbvr = compute_tbvr(train_df, pre_df, train_ratio=0.5)
    assert tbvr == pytest.approx(1.0)


def test_eoc_frobenius_two_when_weights_align_with_test_means():
    pre_df = pd.DataFrame({
        "src_kc": [10, 12],
        "dst_kc": [11, 13],
        "weight": [0.0, 1.0],
        "source": ["train", "train"],
    })
    sim_df = pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
    test_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [1, 2, 3, 4],
        "kc_id": [10, 11, 12, 13],
        "timestamp": [1, 2, 3, 4],
        "correct": [0, 0, 1, 1],
    })
    assert compute_eoc(pre_df, sim_df, test_df) == pytest.approx(2.0)


def test_compute_leakage_row_shapes_splits():
    train_df = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [1, 2],
        "kc_id": [10, 20],
        "timestamp": [1, 2],
        "correct": [1, 1],
    })
    splits = {
        "train": train_df,
        "valid": train_df.iloc[:0].copy(),
        "test": pd.DataFrame({
            "user_id": [2, 2],
            "item_id": [3, 4],
            "kc_id": [10, 20],
            "timestamp": [1, 2],
            "correct": [0, 1],
        }),
    }
    pre_df = pd.DataFrame({"src_kc": [10], "dst_kc": [20], "weight": [1.0], "source": ["t"]})
    sim_df = pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
    q_train = pd.DataFrame({"item_id": [1, 2], "kc_id": [10, 20]})
    row = compute_leakage_row(
        dataset="toy",
        fold=0,
        splits=splits,
        pre_df=pre_df,
        sim_df=sim_df,
        q_train=q_train,
        train_ratio=0.5,
    )
    assert row["dataset"] == "toy"
    assert row["fold"] == 0
    assert row["ecr_flag"] == pytest.approx(0.0)
    assert "ecr_overlap" in row and "eoc" in row and "tbvr" in row
