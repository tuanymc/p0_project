"""I/O helpers for the P0 diagnostic protocol.

All graph-producing stages must pass train-fold data only and record edge
provenance in the leakage audit log.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
CANONICAL_COLUMNS = ["user_id", "item_id", "kc_id", "timestamp", "correct"]
AUDIT_COLUMNS = ["dataset", "fold", "edge_type", "src_kc", "dst_kc", "source_fold", "train_only_flag"]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    logger.info("Loading YAML from %s", path)
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.info("Loaded YAML keys: %s", sorted(data.keys()))
    return data


def load_interactions(path: Path) -> pd.DataFrame:
    """Load canonical interactions with stable dtypes."""
    logger.info("Loading interactions from %s", path)
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    df = df[CANONICAL_COLUMNS].copy()
    for col in ["user_id", "item_id", "kc_id", "timestamp"]:
        df[col] = df[col].astype("int64")
    df["correct"] = df["correct"].astype("int64")
    if not set(df["correct"].dropna().unique()).issubset({0, 1}):
        raise ValueError("correct must contain only 0/1 values")
    logger.info("Loaded interactions shape=%s", df.shape)
    return df


def load_q_matrix(path: Path) -> pd.DataFrame:
    """Load a Q-matrix from CSV or parquet."""
    logger.info("Loading Q-matrix from %s", path)
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    logger.info("Loaded Q-matrix shape=%s", df.shape)
    return df


def dump_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating parent folders."""
    logger.info("Writing CSV shape=%s to %s", df.shape, path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote CSV to %s", path)


def append_audit_row(row: dict[str, Any], path: Path = Path("logs/leakage_audit_log.csv")) -> None:
    """Append an edge provenance row to the audit log."""
    logger.info("Appending audit row to %s: %s", path, row)
    if row.get("train_only_flag") is not True:
        raise AssertionError("Audit rows must have train_only_flag=True")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in AUDIT_COLUMNS})
    logger.info("Appended audit row")
