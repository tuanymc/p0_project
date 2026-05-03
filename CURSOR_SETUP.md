# CURSOR SETUP PROMPT — P0 Project Scaffold

> **How to use this file**
> 1. Open Cursor in an empty workspace.
> 2. Open Cursor Composer / Agent (`Cmd/Ctrl + I`, choose **Agent** mode).
> 3. Copy the entire section below (everything between the `===== BEGIN PROMPT =====`
>    and `===== END PROMPT =====` markers) into the agent input.
> 4. Send. Cursor will scaffold the full project.
> 5. **After scaffolding, read every stub file.** Pay special attention to
>    anything that touches `train/valid/test` boundaries. If Cursor has even
>    once let test data into edge inference, reject the diff and prompt again
>    with the leakage rule highlighted.
>
> **What this prompt does NOT do.** It does not run any experiments and does
> not download any data. It only creates files and stubs. Real implementation
> of the algorithms (Jaccard / PMI similarity, cycle pruning, augmentation
> probes, baseline models) is the NCS's responsibility, with Cursor acting
> as a coding assistant rather than the author.

---

```
===== BEGIN PROMPT =====

You are an expert ML/data engineer helping me scaffold a research codebase
for a Knowledge Tracing (KT) protocol paper. Follow these instructions
EXACTLY. Do NOT add features I did not ask for. Do NOT delete this prompt
from your context.

# 1. PROJECT GOAL

This repository implements a *leakage-controlled* protocol for constructing
and auditing concept graphs in Knowledge Tracing (KT) experiments. It is the
companion code for a workshop/short paper titled "Leakage-Controlled Concept
Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing".

The repo is a PROTOCOL paper artefact, not a model paper. Therefore:
  - Reproducibility, audit logs, and config files matter MORE than accuracy.
  - We do NOT claim SOTA on any benchmark.
  - We do NOT propose a new KT model.
  - Every figure/table in the paper must be reproducible from a single
    bash script in `scripts/`.

# 2. NON-NEGOTIABLE RULES

  R1. TRAIN-ONLY GRAPH CONSTRUCTION.
      Any function that builds, infers, or modifies a graph (E_pre or E_sim)
      must accept ONLY the train fold as input. It must NEVER read the
      validation or test fold. Add a runtime assertion to enforce this.

  R2. FOLD-SPECIFIC GRAPHS.
      Each cross-validation fold has its own graph G_f, built from that
      fold's train data only. Do not cache a single global graph and reuse
      it across folds.

  R3. EDGE PROVENANCE LOG.
      Every edge (in both E_pre and E_sim) must be appended to
      `logs/leakage_audit_log.csv` with columns:
      `dataset, fold, edge_type, src_kc, dst_kc, source_fold, train_only_flag`.
      `train_only_flag` must be `True` for every row -- this is the audit
      contract.

  R4. NO CLAIMS IN CODE.
      Function docstrings and comments must use phrasings like
      "diagnostic", "audit", "report". Do NOT use "improves", "achieves
      state of the art", "first", "novel".

  R5. DETERMINISTIC RUNS.
      Every script that uses randomness must accept a `--seed` flag and seed
      Python `random`, NumPy, and PyTorch (if used). Default seed = 42.

# 3. DIRECTORY STRUCTURE TO CREATE

Create exactly this structure (do not add extra folders, do not omit any):

    p0-kc-graph-protocol/
    ├── README.md                # Will be replaced by the README I provide
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    ├── configs/
    │   ├── junyi.yaml
    │   ├── assist2012.yaml
    │   └── xes3g5m.yaml
    ├── data/
    │   ├── raw/                 # gitignored; populated by the user
    │   └── processed/           # gitignored; produced by preprocess.py
    ├── src/
    │   ├── __init__.py
    │   ├── io_utils.py
    │   ├── preprocess.py
    │   ├── split_checker.py
    │   ├── graph_builder.py
    │   ├── dag_audit.py
    │   ├── dag_disruption.py
    │   ├── cold_start_report.py
    │   ├── baseline_runner.py
    │   └── report_generator.py
    ├── scripts/
    │   ├── run_junyi_minimal.sh
    │   ├── run_assist_minimal.sh
    │   └── make_all_figures.sh
    ├── tests/
    │   ├── test_split_checker.py
    │   ├── test_graph_builder_train_only.py
    │   ├── test_dag_audit.py
    │   └── test_dag_disruption.py
    ├── results/
    │   ├── tables/.gitkeep
    │   ├── figures/.gitkeep
    │   └── reports/.gitkeep
    ├── logs/
    │   ├── literature_search_log.csv
    │   ├── experiment_log.csv
    │   └── leakage_audit_log.csv
    └── paper/
        └── refs.bib             # symlink target for Overleaf project

# 4. DEPENDENCY LIST (`requirements.txt`)

Pin minor versions, not patch versions:

    numpy>=1.24,<2.0
    pandas>=2.0,<3.0
    scipy>=1.10,<2.0
    networkx>=3.1,<4.0
    pyyaml>=6.0
    tqdm>=4.65
    scikit-learn>=1.3,<2.0
    matplotlib>=3.7,<4.0
    seaborn>=0.13,<1.0
    pyarrow>=14.0
    torch>=2.1,<3.0          # optional: only required for DKT/AKT/simpleKT
    pytest>=7.4
    hypothesis>=6.0          # for property-based test on DAG audit

`pyproject.toml` should declare a `p0-kc-graph-protocol` package with
console scripts:
    - p0-preprocess     -> src.preprocess:main
    - p0-split-check    -> src.split_checker:main
    - p0-graph-build    -> src.graph_builder:main
    - p0-dag-audit      -> src.dag_audit:main
    - p0-dag-disrupt    -> src.dag_disruption:main
    - p0-cold-start     -> src.cold_start_report:main
    - p0-baseline       -> src.baseline_runner:main

# 5. STUB SPECIFICATIONS

For EACH file below, generate a stub that contains:
  - A module docstring describing purpose and the train-only contract.
  - The exact function signatures listed.
  - `argparse`-based `main()` for CLI use (where indicated).
  - Type hints on all public functions.
  - `NotImplementedError` for the algorithm body, plus a TODO comment that
    quotes the relevant Survey Roadmap section number.
  - Logging through `logging.getLogger(__name__)`. Every public function
    must `logger.info` its inputs and outputs at INFO level.

## 5.1 `src/io_utils.py`
- `load_yaml(path: Path) -> dict`
- `load_interactions(path: Path) -> pd.DataFrame`
  Returns a DataFrame with columns:
  `user_id, item_id, kc_id, timestamp, correct`. Dtypes:
  user/item/kc as int64, timestamp as int64 (epoch seconds), correct in {0,1}.
- `load_q_matrix(path: Path) -> pd.DataFrame`
- `dump_csv(df: pd.DataFrame, path: Path) -> None`
- `append_audit_row(row: dict, path: Path = Path("logs/leakage_audit_log.csv")) -> None`

## 5.2 `src/preprocess.py`
- `normalise_schema(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame`
- `report_missing(df: pd.DataFrame) -> dict`
- `main()` with CLI:
  `python -m src.preprocess --config configs/junyi.yaml --out data/processed/junyi.parquet`

## 5.3 `src/split_checker.py`
- `learner_based_split(df: pd.DataFrame, ratios: tuple[float,float,float],
                        seed: int = 42) -> dict[str, pd.DataFrame]`
  Returns a dict with keys `train`, `valid`, `test`. Splits by `user_id`,
  preserving temporal order within each user.
- `assert_no_user_overlap(splits: dict) -> None`
  Raises `AssertionError` if any user appears in more than one split.
- `assert_temporal_ordering(splits: dict) -> None`
  For each user that appears in multiple splits (only allowed in temporal
  protocol), assert max(train.t) <= min(valid.t).
- `main()` produces `results/tables/dataset_stats.csv` and prints a summary.

## 5.4 `src/graph_builder.py`
- `build_q_matrix_from_train(train: pd.DataFrame) -> pd.DataFrame`
  IMPORTANT: This function must accept ONLY the train DataFrame. Add a
  runtime check: `assert "fold" not in train.columns or
  train["fold"].nunique() == 1, "Q-matrix must be built per fold."`
- `infer_similarity_edges_from_train(train: pd.DataFrame, q_train: pd.DataFrame,
       method: Literal["jaccard","pmi"] = "jaccard",
       threshold: float = 0.1) -> pd.DataFrame`
  Returns a DataFrame with columns: `src_kc, dst_kc, weight, source`.
- `infer_prerequisites_from_train(train: pd.DataFrame, q_train: pd.DataFrame
       ) -> pd.DataFrame`
  Stub the algorithm with `NotImplementedError("See Survey Roadmap section
  6.1; recommend co-occurrence + temporal precedence rule on TRAIN ONLY.")`.
- `load_ground_truth_dag(dataset_name: str) -> pd.DataFrame`
- `dataset_has_independent_prerequisite_graph(dataset_name: str) -> bool`
- `main()` writes `results/tables/graph_stats.csv` and per-fold edge files
  under `data/processed/<dataset>/fold_<k>/edges_train_only.csv`.

## 5.5 `src/dag_audit.py`
- `topological_sort(edges: pd.DataFrame) -> list[int]`
- `detect_cycles(edges: pd.DataFrame) -> list[list[int]]`
- `prune_cycles(edges: pd.DataFrame, confidence_col: str = "weight"
       ) -> tuple[pd.DataFrame, list[dict]]`
  Returns the pruned edge set AND a log of removed edges with reason.
- `audit_dag(edges: pd.DataFrame) -> DAGReport`
  Define `DAGReport` as a `dataclass` with fields:
  `n_nodes, n_edges, n_roots, n_leaves, n_cycles_before, n_cycles_after,
  topo_sort_passed, pruning_log`.
- `main()` writes `results/reports/dag_report.md`.

## 5.6 `src/dag_disruption.py`
- `apply_node_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame`
- `apply_edge_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame`
- `apply_attribute_mask(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame`
- `apply_subgraph_sampling(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame`
- `compute_dag_disruption_rate(original: pd.DataFrame,
        augmented: pd.DataFrame) -> float`
  Implement DDR as defined in Survey Roadmap section 6.2:
        DDR = |E_pre lost or reversed| / |E_pre|
- `sweep_ddr(edges: pd.DataFrame,
             augmentations: list[str] = ("node_drop","edge_drop","attr_mask","subgraph"),
             ps: list[float] = (0.05, 0.10, 0.20, 0.30),
             seeds: list[int] = (42,)) -> pd.DataFrame`
- `main()` writes `results/tables/dag_disruption.csv` and
  `results/figures/fig_ddr.pdf`.

## 5.7 `src/cold_start_report.py`
- `bin_kcs_by_frequency(train: pd.DataFrame,
       bins: tuple = ((-1, 19), (20, 99), (100, 499), (500, 10_000_000))
       ) -> pd.DataFrame`
- `per_stratum_metrics(predictions: pd.DataFrame,
       strata: pd.DataFrame,
       metrics: list[str] = ("auc","acc","nll")) -> pd.DataFrame`
- `main()` writes `results/reports/cold_start_report.md`.
  Add a guard: if any of the stratum names contains the substring
  "user" or "item", raise. We are reporting cold-start KC, never
  cold-start user/item.

## 5.8 `src/baseline_runner.py`
- `run_bkt(splits: dict, kc_id_col: str = "kc_id") -> dict`
- `run_dkt(splits: dict, hidden_dim: int = 100, lr: float = 1e-3,
           epochs: int = 30, seed: int = 42) -> dict`
- `run_simplekt(splits: dict, **hyperparams) -> dict`     # may NotImplementedError
- `run_akt(splits: dict, **hyperparams) -> dict`          # may NotImplementedError
- `main()` writes `results/tables/baseline_results.csv`.
  Caption to add when consumed by LaTeX: "Baselines reported for diagnostic
  purposes only; we do not claim SOTA."

## 5.9 `src/report_generator.py`
- `generate_p0_diagnostic_report(out_dir: Path) -> Path`
  Aggregates all CSVs and figures into `results/reports/p0_diagnostic_report.md`.

# 6. CONFIG FILES

`configs/junyi.yaml` minimum:

    dataset: junyi
    raw_path: data/raw/junyi/
    has_dag: true
    split:
      type: learner_temporal
      ratios: [0.7, 0.1, 0.2]
      seed: 42
    graph:
      e_sim_method: jaccard           # or pmi
      e_sim_threshold: 0.1
    augmentation:
      methods: [node_drop, edge_drop, attr_mask, subgraph]
      ps: [0.05, 0.10, 0.20, 0.30]
      seeds: [42, 17, 1234]
    cold_start:
      bins: [[-1,19],[20,99],[100,499],[500,10000000]]
    baselines: [bkt, dkt]

`configs/assist2012.yaml`: same shape, `has_dag: false`.
`configs/xes3g5m.yaml`: same shape, `has_dag: true`.

# 7. SHELL SCRIPTS

`scripts/run_junyi_minimal.sh` must run the full pipeline end-to-end:

    #!/usr/bin/env bash
    set -euo pipefail
    CFG=configs/junyi.yaml
    python -m src.preprocess          --config "$CFG"
    python -m src.split_checker       --config "$CFG"
    python -m src.graph_builder       --config "$CFG"
    python -m src.dag_audit           --config "$CFG"
    python -m src.dag_disruption      --config "$CFG"
    python -m src.cold_start_report   --config "$CFG"
    python -m src.baseline_runner     --config "$CFG"
    python -m src.report_generator    --out results/reports/

`scripts/run_assist_minimal.sh` mirrors the above for `assist2012.yaml`.
`scripts/make_all_figures.sh` regenerates every PDF/PNG in `results/figures/`.

All scripts must `chmod +x` themselves in the message you send back, and use
`set -euo pipefail` at the top.

# 8. TESTS

Each test file should contain at least one passing test (the rest can be
parametrised stubs marked `pytest.mark.xfail` until algorithms are filled in).

Hard requirements:

  - `tests/test_split_checker.py` -- test that `assert_no_user_overlap` raises
    on overlapping users; test that temporal ordering is enforced.
  - `tests/test_graph_builder_train_only.py` -- test that
    `infer_similarity_edges_from_train` rejects a DataFrame whose `fold`
    column has more than one value (proxy for accidental cross-fold leakage).
  - `tests/test_dag_audit.py` -- test on a hand-built DAG with one cycle:
    `n_cycles_before == 1`, `n_cycles_after == 0`, topological sort passes.
  - `tests/test_dag_disruption.py` -- test that DDR == 0 when the
    augmentation is identity (p = 0).

# 9. LOG FILES (initial content)

`logs/leakage_audit_log.csv` -- header only:
    dataset,fold,edge_type,src_kc,dst_kc,source_fold,train_only_flag

`logs/literature_search_log.csv` -- header only:
    date,source,query,result,decision,reason

`logs/experiment_log.csv` -- header only:
    date,run_id,config,git_sha,seed,duration_sec,notes

# 10. .gitignore

Include at minimum: `data/raw/`, `data/processed/`, `__pycache__/`, `.venv/`,
`*.egg-info/`, `results/figures/*.pdf`, `results/figures/*.png`,
`*.ipynb_checkpoints`.

# 11. WHEN YOU ARE DONE

End your response with:

  - A tree view of the project (use the `tree` command output style).
  - A bullet list of every file you created.
  - A short paragraph reminding me that I should now (a) read each stub,
    (b) verify the train-only contract, and (c) replace stub algorithms
    with real implementations one script at a time.

Do NOT run any of the scripts. Do NOT touch the network. Do NOT install
packages. Just create the files.

===== END PROMPT =====
```

---

## After Cursor finishes scaffolding

Open these files first and review them in this order:

1. `src/graph_builder.py` — confirm every public function in this module
   raises if given anything other than train-fold data. If Cursor did not add
   the assertion, **add it manually** before continuing.
2. `tests/test_graph_builder_train_only.py` — run `pytest -k train_only`. The
   test must already pass against the stub (it tests the assertion, not the
   algorithm).
3. `scripts/run_junyi_minimal.sh` — try `bash -n scripts/run_junyi_minimal.sh`
   to syntax-check the script. The pipeline cannot run yet (no data + stubs
   raise `NotImplementedError`) but the script structure must be correct.

Once the scaffold is verified, fill in algorithms script by script in the
order suggested by the 12-week roadmap (Week 5 → `graph_builder`,
Week 6 → `dag_audit`, Week 7 → augmentations, Week 8 → DDR, Week 9 →
cold-start, Week 10 → baselines, Week 11 → report aggregation).

## Prompts to keep handy when iterating with Cursor

* "**Show me every place where `valid` or `test` data is read by code that
  builds a graph.** If any exist, refactor so the function only sees train
  data."
* "**Add a regression test:** create a tiny synthetic KT dataset (3 users, 5
  KCs, 12 interactions), build a fold-specific graph, and assert that the
  edge set computed from fold 0 ≠ edge set computed from fold 1."
* "**Audit `logs/leakage_audit_log.csv` after a full run:** every row must
  have `train_only_flag = True`. Fail loudly if not."
* "**Replace any phrasing in docstrings that uses 'state of the art',
  'first', 'novel', or 'improves' with 'diagnostic' / 'audit' / 'report'.**"
