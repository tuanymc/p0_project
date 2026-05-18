"""Preprocess raw KT logs into the canonical train-only protocol schema.

This module does not build graphs; downstream graph stages must consume train
folds only.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.io_utils import CANONICAL_COLUMNS, dump_csv, load_yaml

logger = logging.getLogger(__name__)
XES_SEQUENCE_COLUMNS = ["uid", "questions", "concepts", "responses", "timestamps"]


def _coalesce_mapped_columns(df_raw: pd.DataFrame, source_columns: str | list[str]) -> pd.Series:
    columns = [source_columns] if isinstance(source_columns, str) else source_columns
    missing = [col for col in columns if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing mapped columns: {missing}")
    series = df_raw[columns[0]].astype("string")
    for col in columns[1:]:
        fallback = df_raw[col].astype("string")
        missing_mask = series.isna() | series.str.strip().eq("")
        series = series.mask(missing_mask, fallback)
    return series


def _stable_int64(value: object) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def _encode_stable(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    if (values == "").any():
        raise ValueError("ID columns must not contain missing values after filtering")
    return values.map(_stable_int64).astype("int64")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype("int64")
    # CSV chunks sometimes ingest epoch-ish ints as strings (e.g. Junyi time_done in microseconds).
    numeric_try = pd.to_numeric(series, errors="coerce")
    if numeric_try.notna().all():
        return numeric_try.astype("int64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        bad_examples = series[parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse timestamp examples: {bad_examples}")
    return (parsed.astype("int64") // 1_000_000_000).astype("int64")


def _parse_correct(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")
    text = series.astype("string").str.strip().str.lower()
    bool_map = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    parsed = text.map(bool_map)
    if parsed.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        parsed = parsed.fillna(numeric.where(numeric.isin([0, 1])).astype("float"))
    if parsed.isna().any():
        bad_examples = series[parsed.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(f"correct must be binary after filtering; examples: {bad_examples}")
    return parsed.astype("int64")


def _split_sequence(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",")]


def _normalise_xes_timestamp_ms(value: str) -> int:
    timestamp_ms = int(float(value))
    return timestamp_ms // 1000


def _normalise_xes_sequence_chunk(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Expand XES3G5M sequence rows into canonical interaction rows."""
    logger.info("Normalising XES sequence chunk shape=%s", df_raw.shape)
    rows: list[dict[str, int]] = []
    for record in df_raw.itertuples(index=False):
        uid = getattr(record, "uid")
        questions = _split_sequence(getattr(record, "questions"))
        concepts = _split_sequence(getattr(record, "concepts"))
        responses = _split_sequence(getattr(record, "responses"))
        timestamps = _split_sequence(getattr(record, "timestamps"))
        selectmasks = _split_sequence(getattr(record, "selectmasks")) if hasattr(record, "selectmasks") else []
        n = min(len(questions), len(concepts), len(responses), len(timestamps))
        for i in range(n):
            if selectmasks and i < len(selectmasks) and selectmasks[i] != "1":
                continue
            if questions[i] == "-1" or concepts[i] == "-1" or responses[i] == "-1" or timestamps[i] == "-1":
                continue
            if responses[i] not in {"0", "1"}:
                continue
            rows.append({
                "user_id": _stable_int64(uid),
                "item_id": _stable_int64(questions[i]),
                "kc_id": _stable_int64(concepts[i]),
                "timestamp": _normalise_xes_timestamp_ms(timestamps[i]),
                "correct": int(responses[i]),
            })
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if not df.empty:
        df = df.astype({col: "int64" for col in CANONICAL_COLUMNS})
        df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    logger.info("Normalised XES sequence chunk shape=%s", df.shape)
    return df


def normalise_schema(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename raw columns to the canonical KT schema and enforce KT dtypes."""
    logger.info("Normalising schema with mapping=%s", mapping)
    missing = set(CANONICAL_COLUMNS) - set(mapping)
    if missing:
        raise ValueError(f"Missing canonical mappings: {sorted(missing)}")
    df = pd.DataFrame(index=df_raw.index)
    for col in ["user_id", "item_id", "kc_id"]:
        raw = _coalesce_mapped_columns(df_raw, mapping[col])
        df[col] = _encode_stable(raw)
    df["timestamp"] = _parse_timestamp(_coalesce_mapped_columns(df_raw, mapping["timestamp"]))
    df["correct"] = _parse_correct(_coalesce_mapped_columns(df_raw, mapping["correct"]))
    df = df[CANONICAL_COLUMNS].sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    logger.info("Normalised schema shape=%s", df.shape)
    return df


def report_missing(df: pd.DataFrame) -> dict[str, int]:
    """Report missing values per column."""
    logger.info("Reporting missing values for shape=%s", df.shape)
    report = {col: int(count) for col, count in df.isna().sum().items()}
    logger.info("Missing report=%s", report)
    return report


def _mapped_raw_columns(mapping: dict) -> list[str]:
    columns: list[str] = []
    for raw in mapping.values():
        columns.extend(raw if isinstance(raw, list) else [raw])
    return list(dict.fromkeys(columns))


def _apply_raw_filters(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mode = cfg.get("preprocess", {}).get("correct_mode")
    correct_col = cfg.get("schema_mapping", {}).get("correct")
    if mode == "binary_only" and isinstance(correct_col, str) and correct_col in df_raw.columns:
        numeric = pd.to_numeric(df_raw[correct_col], errors="coerce")
        df_raw = df_raw[numeric.isin([0, 1])].copy()
    kc_mapping = cfg.get("schema_mapping", {}).get("kc_id")
    kc_cols = kc_mapping if isinstance(kc_mapping, list) else [kc_mapping]
    kc_cols = [col for col in kc_cols if col in df_raw.columns]
    if kc_cols:
        has_kc = pd.Series(False, index=df_raw.index)
        for col in kc_cols:
            has_kc = has_kc | df_raw[col].notna()
        df_raw = df_raw[has_kc].copy()
    return df_raw


def _interaction_path(cfg: dict) -> Path:
    if cfg.get("raw_interactions_file"):
        return Path(cfg["raw_interactions_file"])
    raw_path = Path(cfg["raw_path"])
    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_path}")
    return csv_files[0]


def _load_and_normalise_interactions(cfg: dict, limit_rows: int | None = None) -> pd.DataFrame:
    if cfg.get("raw_format") == "xes_sequence":
        return _load_and_normalise_xes_sequences(cfg, limit_rows=limit_rows)
    if cfg.get("raw_format") == "synthetic":
        return _load_and_normalise_synthetic(cfg, limit_rows=limit_rows)
    path = _interaction_path(cfg)
    mapping = cfg.get("schema_mapping", {})
    usecols = _mapped_raw_columns(mapping)
    chunks = []
    remaining = limit_rows
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = _apply_raw_filters(chunk, cfg)
        if remaining is not None:
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        if not chunk.empty:
            chunks.append(normalise_schema(chunk, mapping))
        if remaining is not None and remaining <= 0:
            break
    if not chunks:
        raise ValueError(f"No usable interactions found in {path}")
    return pd.concat(chunks, ignore_index=True).sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def _load_and_normalise_synthetic(cfg: dict, limit_rows: int | None = None) -> pd.DataFrame:
    raw_csv_path = Path(cfg["raw_interactions_file"])
    raw_info_path = Path(cfg["raw_info_file"])
    
    logger.info("Loading synthetic dataset from csv: %s, info: %s", raw_csv_path, raw_info_path)
    
    import numpy as np
    
    # Read the info file to get concept mappings for items
    info_df = pd.read_csv(raw_info_path, sep="\t", header=None)
    item_to_kc = info_df[2].astype(int).tolist()
    
    df_raw = pd.read_csv(raw_csv_path, header=None)
    if limit_rows is not None:
        df_raw = df_raw.head(limit_rows)
        
    n_users = len(df_raw)
    n_items = df_raw.shape[1]
    
    logger.info("Synthetic raw data size: %d users, %d items", n_users, n_items)
    
    user_ids, item_ids = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    
    user_ids_flat = user_ids.ravel()
    item_ids_flat = item_ids.ravel()
    correct_flat = df_raw.to_numpy().ravel()
    
    item_to_kc_arr = np.array(item_to_kc)
    kc_ids_flat = item_to_kc_arr[item_ids_flat]
    
    timestamps_flat = item_ids_flat
    
    df_canonical = pd.DataFrame({
        "user_id": pd.Series(user_ids_flat).astype(str),
        "item_id": pd.Series(item_ids_flat).astype(str),
        "kc_id": pd.Series(kc_ids_flat).astype(str),
        "timestamp": pd.Series(timestamps_flat).astype("int64"),
        "correct": pd.Series(correct_flat).astype("int64")
    })
    
    df_canonical["user_id"] = _encode_stable(df_canonical["user_id"])
    df_canonical["item_id"] = _encode_stable(df_canonical["item_id"])
    df_canonical["kc_id"] = _encode_stable(df_canonical["kc_id"])
    
    df_canonical = df_canonical.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    return df_canonical


def _load_and_normalise_xes_sequences(cfg: dict, limit_rows: int | None = None) -> pd.DataFrame:
    paths = [Path(path) for path in cfg.get("raw_sequence_files", [])]
    if not paths:
        raise ValueError("raw_sequence_files must be configured for raw_format=xes_sequence")
    chunks = []
    remaining = limit_rows
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        usecols = [col for col in XES_SEQUENCE_COLUMNS + ["selectmasks"] if col in pd.read_csv(path, nrows=0).columns]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000):
            df = _normalise_xes_sequence_chunk(chunk)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= len(df)
            if not df.empty:
                chunks.append(df)
            if remaining is not None and remaining <= 0:
                break
        if remaining is not None and remaining <= 0:
            break
    if not chunks:
        raise ValueError("No usable XES3G5M interactions found")
    return pd.concat(chunks, ignore_index=True).sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def _dataset_stats_row(dataset: str, df: pd.DataFrame, missing: dict[str, int]) -> dict[str, int | str]:
    return {
        "dataset": dataset,
        "n_learners": df["user_id"].nunique(),
        "n_items": df["item_id"].nunique(),
        "n_kcs": df["kc_id"].nunique(),
        "n_interactions": len(df),
        "missing_total": sum(missing.values()),
    }


def _write_dataset_stats(row: dict[str, int | str], is_sample: bool) -> None:
    if is_sample:
        dump_csv(pd.DataFrame([row]), Path(f"results/tables/{row['dataset']}_sample_stats.csv"))
        return
    path = Path("results/tables/dataset_stats.csv")
    if path.exists():
        stats = pd.read_csv(path)
        stats = stats[stats["dataset"] != row["dataset"]]
        stats = pd.concat([stats, pd.DataFrame([row])], ignore_index=True)
    else:
        stats = pd.DataFrame([row])
    stats = stats.sort_values("dataset").reset_index(drop=True)
    dump_csv(stats, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a KT dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    cfg = load_yaml(args.config)
    out = args.out or Path(cfg.get("processed_path", f"data/processed/{cfg['dataset']}.parquet"))
    df = _load_and_normalise_interactions(cfg, limit_rows=args.limit_rows)
    missing = report_missing(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _write_dataset_stats(_dataset_stats_row(cfg["dataset"], df, missing), is_sample=args.limit_rows is not None)
    logger.info("Preprocess report written for %s", cfg["dataset"])


if __name__ == "__main__":
    main()
