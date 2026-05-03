import pandas as pd

from src.dag_audit import audit_dag


def test_audit_prunes_one_cycle():
    edges = pd.DataFrame({
        "src_kc": [1, 2, 3, 3],
        "dst_kc": [2, 3, 1, 4],
        "weight": [0.9, 0.8, 0.1, 0.7],
    })
    report = audit_dag(edges)
    assert report.n_cycles_before == 1
    assert report.n_cycles_after == 0
    assert report.topo_sort_passed is True
