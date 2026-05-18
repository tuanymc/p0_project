"""Export P0 parquet splits to pyKT sequence CSV + dense ID maps (train-only vocab)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_dense_maps(train_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Question (item) and KC ids remapped to contiguous indices from **train only**."""
    kcs = sorted(int(x) for x in train_df["kc_id"].unique())
    items = sorted(int(x) for x in train_df["item_id"].unique())
    return ({k: i for i, k in enumerate(items)}, {k: i for i, k in enumerate(kcs)})


def _build_rows(df: pd.DataFrame, fold_val: int, q_map: dict[int, int], c_map: dict[int, int], max_seq_len: int) -> pd.DataFrame:
    rows = []
    for uid, grp in df.groupby("user_id", sort=False):
        grp = grp.sort_values("timestamp")
        seq: list[tuple[int, int, int]] = []
        for _, r in grp.iterrows():
            qi = q_map.get(int(r["item_id"]), -1)
            ci = c_map.get(int(r["kc_id"]), -1)
            if qi < 0 or ci < 0:
                continue
            seq.append((qi, ci, int(r["correct"])))
        if len(seq) < 2:
            continue
        seq = seq[-max_seq_len:]
        L = len(seq)
        pad_n = max_seq_len - L
        questions = [str(x[0]) for x in seq] + ["-1"] * pad_n
        concepts = [str(x[1]) for x in seq] + ["-1"] * pad_n
        responses = [str(float(x[2])) for x in seq] + ["0"] * pad_n
        smasks = ["1"] * L + ["0"] * pad_n
        timestamps = [str(t) for t in range(L)] + ["-1"] * pad_n
        rows.append(
            {
                "fold": fold_val,
                "uid": int(uid),
                "questions": ",".join(questions),
                "concepts": ",".join(concepts),
                "responses": ",".join(responses),
                "selectmasks": ",".join(smasks),
                "timestamps": ",".join(timestamps),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_pykt_csvs(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    q_map: dict[int, int],
    c_map: dict[int, int],
    out_dir: Path,
    max_seq_len: int,
) -> tuple[int, int]:
    """Write ``train_valid_sequences.csv`` (fold 0=train, 1=valid) and ``test_sequences.csv`` (fold -1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = pd.concat(
        [_build_rows(train_df, 0, q_map, c_map, max_seq_len), _build_rows(valid_df, 1, q_map, c_map, max_seq_len)],
        ignore_index=True,
    )
    te = _build_rows(test_df, -1, q_map, c_map, max_seq_len)
    tv.to_csv(out_dir / "train_valid_sequences.csv", index=False)
    te.to_csv(out_dir / "test_sequences.csv", index=False)
    num_q = max(q_map.values(), default=-1) + 1
    num_c = max(c_map.values(), default=-1) + 1
    return int(num_q), int(num_c)
