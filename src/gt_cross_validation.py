"""Ground-truth cross-validation helpers for comparing inferred E_pre to expert DAGs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_K_LIST = [200, 500, 800, 1200, 1800, 2500, 3500, 5000]


def load_expert_dag(
    path: Path,
    schema: Literal["auto", "integer", "junyi_name"] = "auto",
    kc_name_to_id: dict[str, int] | None = None,
    prereq_threshold: float = 0.5,
) -> pd.DataFrame:
    """Load expert prerequisite DAG.

    Supports integer CSV (src_kc, dst_kc[, confidence_score]) or Junyi Chang-style
    name columns with Prerequisite_avg filtering.

    Returns canonical columns: src_kc, dst_kc, confidence_score (float).
    Unmapped Junyi rows are dropped with a warning.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    cols = set(raw.columns.str.strip() if hasattr(raw.columns, "str") else raw.columns)

    detected = schema
    if schema == "auto":
        if {"Exercise_A", "Exercise_B"}.issubset(cols) and "Prerequisite_avg" in cols:
            detected = "junyi_name"
        elif "src_kc" in raw.columns and "dst_kc" in raw.columns:
            detected = "integer"
        else:
            raise ValueError(f"Cannot auto-detect expert DAG schema; columns={list(raw.columns)}")

    if detected == "integer":
        df = raw.rename(columns={c: c.strip() for c in raw.columns}).copy()
        score_col = None
        for cand in ("confidence_score", "confidence", "weight", "Prerequisite_avg"):
            if cand in df.columns:
                score_col = cand
                break
        if score_col:
            conf = pd.to_numeric(df[score_col], errors="coerce").fillna(1.0)
        else:
            conf = pd.Series(1.0, index=df.index)
        out = pd.DataFrame({
            "src_kc": pd.to_numeric(df["src_kc"], errors="coerce").astype("Int64"),
            "dst_kc": pd.to_numeric(df["dst_kc"], errors="coerce").astype("Int64"),
            "confidence_score": conf,
        })
        before = len(out)
        out = out.dropna(subset=["src_kc", "dst_kc"])
        out["src_kc"] = out["src_kc"].astype(np.int64)
        out["dst_kc"] = out["dst_kc"].astype(np.int64)
        if len(out) < before:
            logger.warning("Dropped %s integer rows with NaN endpoints", before - len(out))
        return out.reset_index(drop=True)

    if detected == "junyi_name":
        if kc_name_to_id is None:
            raise ValueError("kc_name_to_id required for junyi_name schema")
        df = raw.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Prerequisite_avg"] >= float(prereq_threshold)].copy()
        name_a = df["Exercise_A"].astype(str).str.strip()
        name_b = df["Exercise_B"].astype(str).str.strip()
        src = name_a.map(kc_name_to_id)
        dst = name_b.map(kc_name_to_id)
        bad = int(src.isna().sum() + dst.isna().sum())
        if bad:
            logger.warning("Junyi expert DAG: %s endpoint mappings missing after name lookup", bad)
        tmp = pd.DataFrame({"src_kc": src, "dst_kc": dst, "confidence_score": df["Prerequisite_avg"].astype(float)})
        tmp = tmp.dropna(subset=["src_kc", "dst_kc"])
        tmp["src_kc"] = tmp["src_kc"].astype(np.int64)
        tmp["dst_kc"] = tmp["dst_kc"].astype(np.int64)
        return tmp.reset_index(drop=True)

    raise ValueError(f"Unknown schema: {schema}")


def align_kc_ids(
    expert_dag: pd.DataFrame,
    kc_name_to_id: dict[str, int] | None = None,
    train_kc_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach alignment status; optionally restrict expert edges to train KCs.

    If ``expert_dag`` already has integer ``src_kc``/``dst_kc``, name mapping is a no-op.
    Raw Junyi rows (Exercise_A/B) can be passed with ``kc_name_to_id`` to produce IDs.

    Returns aligned frame with columns src_kc, dst_kc, confidence_score, name_src,
    name_dst (optional), status; and an alignment report dict.
    """
    expert_dag = expert_dag.copy()

    if {"Exercise_A", "Exercise_B"}.issubset(expert_dag.columns):
        if kc_name_to_id is None:
            raise ValueError("kc_name_to_id required when expert_dag has Exercise_A/B")
        name_src = expert_dag["Exercise_A"].astype(str).str.strip()
        name_dst = expert_dag["Exercise_B"].astype(str).str.strip()
        src = name_src.map(kc_name_to_id)
        dst = name_dst.map(kc_name_to_id)
        score = (
            expert_dag["Prerequisite_avg"].astype(float)
            if "Prerequisite_avg" in expert_dag.columns
            else pd.Series(1.0, index=expert_dag.index)
        )
        aligned = pd.DataFrame({
            "name_src": name_src,
            "name_dst": name_dst,
            "src_kc": src,
            "dst_kc": dst,
            "confidence_score": score,
        })
    else:
        aligned = expert_dag.copy()
        if "confidence_score" not in aligned.columns:
            aligned["confidence_score"] = 1.0
        aligned["name_src"] = ""
        aligned["name_dst"] = ""

    missing_src = int(aligned["src_kc"].isna().sum())
    missing_dst = int(aligned["dst_kc"].isna().sum())
    both_mapped = aligned.dropna(subset=["src_kc", "dst_kc"])
    missing_both_rows = len(aligned) - len(both_mapped)

    aligned_nm = aligned.dropna(subset=["src_kc", "dst_kc"]).copy()
    aligned_nm["src_kc"] = aligned_nm["src_kc"].astype(np.int64)
    aligned_nm["dst_kc"] = aligned_nm["dst_kc"].astype(np.int64)

    if train_kc_ids is None:
        aligned_nm["status"] = "matched"
        report = {
            "matched": len(aligned_nm),
            "missing_src": missing_src,
            "missing_dst": missing_dst,
            "missing_both": missing_both_rows,
            "alignment_rate": float(len(aligned_nm) / len(aligned)) if len(aligned) else 0.0,
            "train_filtered": False,
        }
        return aligned_nm.reset_index(drop=True), report

    def _status(row: pd.Series) -> str:
        s_ok = int(row["src_kc"]) in train_kc_ids
        d_ok = int(row["dst_kc"]) in train_kc_ids
        if s_ok and d_ok:
            return "matched"
        if not s_ok and not d_ok:
            return "missing_both"
        if not s_ok:
            return "missing_src"
        return "missing_dst"

    aligned_nm["status"] = aligned_nm.apply(_status, axis=1)
    n_matched = int((aligned_nm["status"] == "matched").sum())
    report = {
        "matched": n_matched,
        "missing_src": int((aligned_nm["status"] == "missing_src").sum()),
        "missing_dst": int((aligned_nm["status"] == "missing_dst").sum()),
        "missing_both": int((aligned_nm["status"] == "missing_both").sum()),
        "alignment_rate": float(n_matched / len(aligned_nm)) if len(aligned_nm) else 0.0,
        "train_filtered": True,
        "name_mapping_missing_src": missing_src,
        "name_mapping_missing_dst": missing_dst,
    }
    return aligned_nm.reset_index(drop=True), report


def _inferred_sort_columns(df: pd.DataFrame) -> tuple[list[str], list[bool]]:
    if "support" in df.columns:
        return ["support", "src_kc", "dst_kc"], [False, True, True]
    if "weight" in df.columns:
        return ["weight", "src_kc", "dst_kc"], [False, True, True]
    return ["src_kc", "dst_kc"], [True, True]


def _truncate_inferred(inferred_edges: pd.DataFrame, top_k: int) -> pd.DataFrame:
    inf = inferred_edges.copy()
    if inf.empty or top_k <= 0:
        return inf.iloc[0:0].copy()
    cols, asc = _inferred_sort_columns(inf)
    for c in cols:
        if c not in inf.columns:
            raise ValueError(f"Inferred edges missing column {c!r} for sorting")
    ranked = inf.sort_values(cols, ascending=asc).reset_index(drop=True)
    return ranked.head(int(top_k)).copy()


def _directed_edge_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    if edges.empty:
        return set()
    s = edges[["src_kc", "dst_kc"]].astype(np.int64)
    return set(zip(s["src_kc"].tolist(), s["dst_kc"].tolist()))


def _direction_agreement(exp: set[tuple[int, int]], inf: set[tuple[int, int]]) -> float:
    pairs: set[frozenset[int]] = set()
    for u, v in exp:
        pairs.add(frozenset((int(u), int(v))))
    for u, v in inf:
        pairs.add(frozenset((int(u), int(v))))
    denom_pairs: list[frozenset[int]] = []
    for pr in pairs:
        u, v = tuple(pr)
        exp_e = {(u, v) for u, v in ((u, v), (v, u)) if (u, v) in exp}
        inf_e = {(u, v) for u, v in ((u, v), (v, u)) if (u, v) in inf}
        if exp_e and inf_e:
            denom_pairs.append(pr)
    if not denom_pairs:
        return 0.0
    agree = 0
    for pr in denom_pairs:
        u, v = tuple(pr)
        exp_on = {(a, b) for a, b in ((u, v), (v, u)) if (a, b) in exp}
        inf_on = {(a, b) for a, b in ((u, v), (v, u)) if (a, b) in inf}
        if exp_on == inf_on:
            agree += 1
    return agree / len(denom_pairs)


def _reachable_pairs(edges: pd.DataFrame) -> set[tuple[int, int]]:
    if edges.empty:
        return set()
    g = nx.DiGraph()
    for row in edges.itertuples(index=False):
        g.add_edge(int(row.src_kc), int(row.dst_kc))
    pairs: set[tuple[int, int]] = set()
    for u in g.nodes():
        for v in nx.descendants(g, u):
            pairs.add((int(u), int(v)))
    return pairs


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _f1(p: float, r: float) -> float:
    if p <= 0 and r <= 0:
        return 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_overlap_metrics(
    inferred_edges: pd.DataFrame,
    expert_edges: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    """Compare truncated inferred edges to expert directed edges."""
    inf_trunc = _truncate_inferred(inferred_edges, top_k)
    exp_set = _directed_edge_set(expert_edges)
    inf_set = _directed_edge_set(inf_trunc)

    tp_edge = len(exp_set & inf_set)
    prec_e = _safe_div(tp_edge, len(inf_set))
    rec_e = _safe_div(tp_edge, len(exp_set))
    f1_e = _f1(prec_e, rec_e)

    dir_ag = _direction_agreement(exp_set, inf_set)

    exp_nodes = set()
    for u, v in exp_set:
        exp_nodes.add(u)
        exp_nodes.add(v)
    inf_nodes = set()
    for u, v in inf_set:
        inf_nodes.add(u)
        inf_nodes.add(v)
    uni = exp_nodes | inf_nodes
    inter = exp_nodes & inf_nodes
    jacc = _safe_div(len(inter), len(uni)) if uni else 0.0

    r_exp = _reachable_pairs(expert_edges)
    r_inf = _reachable_pairs(inf_trunc)
    tp_r = len(r_exp & r_inf)
    prec_r = _safe_div(tp_r, len(r_inf))
    rec_r = _safe_div(tp_r, len(r_exp))
    f1_r = _f1(prec_r, rec_r)

    return {
        "top_k": int(top_k),
        "n_inferred_truncated": len(inf_set),
        "n_expert_edges": len(exp_set),
        "edge_precision": prec_e,
        "edge_recall": rec_e,
        "edge_f1": f1_e,
        "direction_agreement": dir_ag,
        "reachability_precision": prec_r,
        "reachability_recall": rec_r,
        "reachability_f1": f1_r,
        "node_jaccard": jacc,
    }


def precision_recall_sweep(
    inferred_edges: pd.DataFrame,
    expert_edges: pd.DataFrame,
    k_list: list[int],
) -> pd.DataFrame:
    """One overlap-metrics row per K."""
    rows = []
    for k in k_list:
        rows.append(compute_overlap_metrics(inferred_edges, expert_edges, int(k)))
    return pd.DataFrame(rows)


def diagnose_disagreement(
    inferred_edges: pd.DataFrame,
    expert_edges: pd.DataFrame,
    top_k: int = 800,
    n_examples: int = 20,
    id_to_name: dict[int, str] | None = None,
    kc_name_to_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Qualitative disagreement buckets for paper discussion."""
    inf_trunc = _truncate_inferred(inferred_edges, top_k)
    if not inf_trunc.empty:
        inf_trunc = inf_trunc.copy()
        inf_trunc["src_kc"] = inf_trunc["src_kc"].astype("int64")
        inf_trunc["dst_kc"] = inf_trunc["dst_kc"].astype("int64")
    exp_df = expert_edges.copy()
    if not exp_df.empty:
        exp_df["src_kc"] = exp_df["src_kc"].astype("int64")
        exp_df["dst_kc"] = exp_df["dst_kc"].astype("int64")

    exp_set = _directed_edge_set(exp_df)
    inf_set = _directed_edge_set(inf_trunc)

    name_lookup: dict[int, str] = {}
    if id_to_name:
        name_lookup.update({int(k): str(v) for k, v in id_to_name.items()})
    if kc_name_to_id:
        for kn, vid in kc_name_to_id.items():
            name_lookup[int(vid)] = str(kn)

    def _nid(val: object) -> int:
        return int(np.asarray(val, dtype=np.int64).item())

    def _name_pair(u: object, v: object) -> tuple[str, str]:
        ui, vi = _nid(u), _nid(v)
        return name_lookup.get(ui, str(ui)), name_lookup.get(vi, str(vi))

    def _score_series(row: pd.Series) -> float:
        if "support" in row.index and pd.notna(row["support"]):
            return float(row["support"])
        if "weight" in row.index and pd.notna(row["weight"]):
            return float(row["weight"])
        return 0.0

    inferred_only: list[tuple[str, str, float]] = []
    cand = inf_trunc.copy()
    cols, asc = _inferred_sort_columns(cand)
    cand = cand.sort_values(cols, ascending=asc)
    for row in cand.itertuples(index=False, name="Inf"):
        u = _nid(row.src_kc)
        v = _nid(row.dst_kc)
        if (u, v) not in exp_set:
            sup = getattr(row, "support", np.nan)
            wgt = getattr(row, "weight", np.nan)
            row_series = pd.Series({"support": sup, "weight": wgt})
            inferred_only.append((*_name_pair(u, v), _score_series(row_series)))
        if len(inferred_only) >= n_examples:
            break

    expert_only: list[tuple[str, str, float]] = []
    exp_sorted = exp_df
    if not exp_sorted.empty and "confidence_score" in exp_sorted.columns:
        exp_sorted = exp_sorted.sort_values("confidence_score", ascending=False)
    for row in exp_sorted.itertuples(index=False, name="Exp"):
        u = _nid(row.src_kc)
        v = _nid(row.dst_kc)
        if (u, v) not in inf_set:
            conf = float(row.confidence_score) if hasattr(row, "confidence_score") else 1.0
            expert_only.append((*_name_pair(u, v), conf))
        if len(expert_only) >= n_examples:
            break

    wrong_direction: list[tuple[str, str, float]] = []
    seen: set[frozenset[int]] = set()
    for u, v in inf_set:
        if (v, u) in exp_set and (u, v) not in exp_set:
            pr = frozenset((u, v))
            if pr in seen:
                continue
            seen.add(pr)
            sub = inf_trunc.loc[(inf_trunc["src_kc"] == u) & (inf_trunc["dst_kc"] == v)]
            if len(sub) > 0:
                r0 = sub.iloc[0]
                scr = float(r0["support"]) if "support" in sub.columns else float(r0["weight"])
            else:
                scr = 0.0
            wrong_direction.append((*_name_pair(u, v), scr))
        if len(wrong_direction) >= n_examples:
            break

    return {
        "inferred_only": inferred_only[:n_examples],
        "expert_only": expert_only[:n_examples],
        "wrong_direction": wrong_direction[:n_examples],
    }


def disagreement_dict_to_dataframe(diag: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for bucket, triples in diag.items():
        for i, t in enumerate(triples):
            rows.append({
                "category": bucket,
                "rank": i + 1,
                "src_name": t[0],
                "dst_name": t[1],
                "score": t[2],
            })
    return pd.DataFrame(rows)
