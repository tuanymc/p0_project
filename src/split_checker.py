"""Split and audit learner boundaries for the P0 protocol.

This module prepares folds; graph construction must use only the returned train
fold for each graph.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import dump_csv, load_interactions, load_yaml

logger = logging.getLogger(__name__)


def learner_based_split(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split by learner while preserving temporal order inside each learner."""
    logger.info("Creating learner split shape=%s ratios=%s seed=%s", df.shape, ratios, seed)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    users = np.array(sorted(df["user_id"].unique()))
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    split_users = {
        "train": set(users[:n_train]),
        "valid": set(users[n_train:n_train + n_valid]),
        "test": set(users[n_train + n_valid:]),
    }
    splits: dict[str, pd.DataFrame] = {}
    for name, user_set in split_users.items():
        part = df[df["user_id"].isin(user_set)].copy()
        part = part.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        part["split"] = name
        splits[name] = part
    logger.info("Created split sizes=%s", {k: len(v) for k, v in splits.items()})
    return splits


def assert_no_user_overlap(splits: dict) -> None:
    """Raise if any learner appears in more than one split."""
    logger.info("Checking user overlap for splits=%s", list(splits.keys()))
    seen: dict[object, str] = {}
    for split_name, df in splits.items():
        for user_id in set(df["user_id"].unique()):
            if user_id in seen:
                raise AssertionError(f"User {user_id} appears in {seen[user_id]} and {split_name}")
            seen[user_id] = split_name
    logger.info("No user overlap detected")


def assert_temporal_ordering(splits: dict) -> None:
    """Assert chronological boundaries for temporal protocols."""
    logger.info("Checking temporal ordering for splits=%s", list(splits.keys()))
    order = ["train", "valid", "test"]
    for left, right in zip(order, order[1:]):
        if left not in splits or right not in splits:
            continue
        common_users = set(splits[left]["user_id"].unique()) & set(splits[right]["user_id"].unique())
        for user_id in common_users:
            left_max = splits[left].loc[splits[left]["user_id"] == user_id, "timestamp"].max()
            right_min = splits[right].loc[splits[right]["user_id"] == user_id, "timestamp"].min()
            if left_max > right_min:
                raise AssertionError(f"Temporal boundary violated for user {user_id}: {left}>{right}")
    logger.info("Temporal ordering accepted")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and audit KT splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    _seed_all(args.seed)

    cfg = load_yaml(args.config)
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{cfg['dataset']}.parquet")))
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    splits = learner_based_split(df, ratios, seed=cfg.get("split", {}).get("seed", args.seed))
    assert_no_user_overlap(splits)
    assert_temporal_ordering(splits)
    stats = pd.DataFrame([{
        "dataset": cfg["dataset"],
        "split": name,
        "n_learners": part["user_id"].nunique(),
        "n_interactions": len(part),
    } for name, part in splits.items()])
    dump_csv(stats, Path("results/tables/dataset_stats.csv"))
    print("[OK] No learner overlap between train / valid / test.")
    print("[OK] Temporal order respected for all users.")


if __name__ == "__main__":
    main()
