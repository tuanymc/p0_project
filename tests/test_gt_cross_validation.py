"""Tests for GT cross-validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.gt_cross_validation import (
    align_kc_ids,
    compute_overlap_metrics,
    diagnose_disagreement,
    load_expert_dag,
    precision_recall_sweep,
)


def test_align_kc_ids_perfect_match():
    expert = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "confidence_score": [1.0, 1.0]})
    train_kcs = {1, 2, 3}
    aligned, report = align_kc_ids(expert, train_kc_ids=train_kcs)
    assert report["matched"] == 2
    assert report["alignment_rate"] == 1.0
    assert set(aligned["status"]) == {"matched"}


def test_align_kc_ids_partial():
    expert = pd.DataFrame({"src_kc": [1, 99], "dst_kc": [2, 3], "confidence_score": [1.0, 1.0]})
    train_kcs = {1, 2, 3}
    aligned, report = align_kc_ids(expert, train_kc_ids=train_kcs)
    matched = aligned[aligned["status"] == "matched"]
    assert len(matched) == 1
    assert report["missing_src"] >= 1 or report["missing_dst"] >= 1


def test_compute_overlap_identical():
    edges = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "support": [5, 4], "weight": [1.0, 0.8]})
    m = compute_overlap_metrics(edges, edges, top_k=10)
    assert m["edge_precision"] == pytest.approx(1.0)
    assert m["edge_recall"] == pytest.approx(1.0)


def test_compute_overlap_disjoint():
    inf = pd.DataFrame({"src_kc": [1], "dst_kc": [2], "support": [10], "weight": [1.0]})
    exp = pd.DataFrame({"src_kc": [9], "dst_kc": [8], "confidence_score": [1.0]})
    m = compute_overlap_metrics(inf, exp, top_k=10)
    assert m["edge_precision"] == 0.0
    assert m["edge_recall"] == 0.0


def test_direction_agreement_opposite():
    inf = pd.DataFrame({"src_kc": [2], "dst_kc": [1], "support": [10], "weight": [1.0]})
    exp = pd.DataFrame({"src_kc": [1], "dst_kc": [2], "confidence_score": [1.0]})
    m = compute_overlap_metrics(inf, exp, top_k=10)
    assert m["direction_agreement"] == pytest.approx(0.0)


def test_precision_recall_sweep_monotonic():
    inf = pd.DataFrame({
        "src_kc": [1, 2, 3, 4],
        "dst_kc": [2, 3, 4, 5],
        "support": [40, 30, 20, 10],
        "weight": [1.0, 0.9, 0.8, 0.7],
    })
    exp = pd.DataFrame({"src_kc": [1, 99], "dst_kc": [2, 100], "confidence_score": [1.0, 1.0]})
    sweep = precision_recall_sweep(inf, exp, [1, 2, 3, 4])
    recalls = sweep["edge_recall"].tolist()
    assert all(recalls[i] <= recalls[i + 1] + 1e-9 for i in range(len(recalls) - 1))


def test_reachability_transitive():
    exp = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 3], "confidence_score": [1.0, 1.0]})
    inf = pd.DataFrame({"src_kc": [1], "dst_kc": [3], "support": [10], "weight": [1.0]})
    m = compute_overlap_metrics(inf, exp, top_k=10)
    assert m["reachability_precision"] == pytest.approx(1.0)
    assert m["reachability_recall"] > 0


def test_load_expert_dag_junyi_name(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text(
        "Exercise_A,Exercise_B,Similarity_avg,Prerequisite_avg\n"
        "a,b,1.0,5.0\n"
        "x,y,1.0,2.0\n",
        encoding="utf-8",
    )
    mapping = {"a": 10, "b": 20, "x": 30, "y": 40}
    df = load_expert_dag(path, schema="junyi_name", kc_name_to_id=mapping, prereq_threshold=0.5)
    assert len(df) == 2
    assert set(df["src_kc"].tolist()) == {10, 30}


def test_diagnose_buckets():
    inf = pd.DataFrame({"src_kc": [1, 2], "dst_kc": [2, 9], "support": [10, 9], "weight": [1.0, 0.9]})
    exp = pd.DataFrame({"src_kc": [1], "dst_kc": [2], "confidence_score": [5.0]})
    d = diagnose_disagreement(inf, exp, top_k=10, n_examples=5)
    assert len(d["inferred_only"]) >= 1
