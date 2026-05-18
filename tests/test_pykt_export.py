import pandas as pd

from src.pykt_export import build_dense_maps, dataframe_to_pykt_csvs


def test_pykt_export_writes_nonempty(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 11, 10, 12],
            "kc_id": [5, 6, 5, 7],
            "timestamp": [1, 2, 1, 2],
            "correct": [1, 0, 1, 1],
        }
    )
    valid = train.iloc[:2].copy()
    test = train.iloc[2:].copy()
    qm, cm = build_dense_maps(train)
    nq, nc = dataframe_to_pykt_csvs(
        train_df=train,
        valid_df=valid,
        test_df=test,
        q_map=qm,
        c_map=cm,
        out_dir=tmp_path,
        max_seq_len=50,
    )
    assert nq >= 1 and nc >= 1
    assert (tmp_path / "train_valid_sequences.csv").exists()
    assert (tmp_path / "test_sequences.csv").exists()
