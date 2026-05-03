# P0: Leakage-Controlled KC Graph Construction & Cold-Start Diagnostic Protocol

Companion code repository for the workshop paper *"Leakage-Controlled Concept
Graph Construction and Cold-Start Diagnostic Protocol for Knowledge
Tracing"* (P0 of the GraphKT-ITS thesis).

> **What this repo is.** A protocol/audit pipeline for building and checking
> multi-relational concept graphs from KT logs without data leakage, plus a
> cold-start KC diagnostic and a DAG Disruption Rate metric.
>
> **What this repo is NOT.** A new KT model. There is no claim of SOTA, no
> claim about real-world learning outcomes, and no joint SSL training.

---

## Table of contents

1. [Quick start (minimal pipeline on Junyi)](#1-quick-start)
2. [Environment setup](#2-environment-setup)
3. [Data download and preparation](#3-data-download-and-preparation)
4. [Running the full pipeline](#4-running-the-full-pipeline)
5. [Per-stage commands](#5-per-stage-commands)
6. [Outputs and where they live](#6-outputs)
7. [Reproducing every figure and table](#7-reproducing-every-figure-and-table)
8. [Troubleshooting](#8-troubleshooting)
9. [Project structure](#9-project-structure)
10. [Citation, licence, and contact](#10-citation)

---

## 1. Quick start

After completing [§2 Environment setup](#2-environment-setup) and
[§3 Data download](#3-data-download-and-preparation):

```bash
bash scripts/run_junyi_minimal.sh
```

Expected runtime: **~5–15 minutes** on a laptop CPU (longer if `torch`
baselines are enabled).

Expected console tail:

```
[ split_checker     ] OK   - no learner overlap; temporal ordering verified
[ graph_builder     ] OK   - 3 folds; edges_train_only.csv written
[ dag_audit         ] OK   - cycles_before=0, topo_sort=passed
[ dag_disruption    ] OK   - DDR sweep written: 4 augmentations × 4 ps
[ cold_start_report ] OK   - 4 strata; report_md emitted
[ baseline_runner   ] OK   - BKT/DKT diagnostic table written
[ report_generator  ] OK   - results/reports/p0_diagnostic_report.md
```

A short summary report appears at
`results/reports/p0_diagnostic_report.md`.

---

## 2. Environment setup

Tested on **Python 3.10 / 3.11**, Linux and macOS. Windows works under WSL2.

### 2.1 Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2.2 Install dependencies

```bash
pip install -r requirements.txt
pip install -e .                   # registers the p0-* console scripts
```

### 2.3 Verify the install

```bash
pytest -q
```

You should see the train-only contract tests **pass** and the algorithm
tests marked **xfail** (expected — the algorithms are not yet implemented).

### 2.4 GPU (optional)

`torch` is needed only for DKT / simpleKT / AKT baselines. CPU works for
the full diagnostic pipeline. To use a GPU, install the matching CUDA build
of PyTorch *after* the requirements install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Data download and preparation

We use **public KT benchmarks only**. Each dataset has its own licence;
read and respect it before downloading.

| Dataset | Required for P0 | Has prerequisite DAG? | License notes |
|---|---|---|---|
| Junyi Academy | yes | yes (ground truth) | Check the Junyi release / PSLC DataShop entry. |
| ASSISTments 2012 | yes | no (must infer) | Check the ASSISTments release. |
| XES3G5M | recommended | yes | Verify dataset card before camera-ready. |
| EdNet KT1 | not required | partial | Optional only. |

### 3.1 Folder layout

Place raw files under `data/raw/<dataset>/`:

```
data/raw/
├── junyi/
│   ├── junyi_log.csv             # exact filenames depend on release
│   └── junyi_dag.csv             # ground-truth prerequisite DAG
├── assist2012/
│   └── 2012-2013-data-with-predictions-4-final.csv
└── xes3g5m/
    └── ...
```

These folders are **gitignored**.

### 3.2 Preprocess

```bash
python -m src.preprocess --config configs/junyi.yaml \
                         --out    data/processed/junyi.parquet
```

This normalises columns to
`user_id, item_id, kc_id, timestamp, correct` and writes a parquet file
under `data/processed/`. Run the same command for the other configs.

### 3.3 Sanity check

```bash
python -m src.split_checker --config configs/junyi.yaml
```

The script must print:

```
[OK] No learner overlap between train / valid / test.
[OK] Temporal order respected for all users.
```

Anything else means stop and fix before proceeding.

---

## 4. Running the full pipeline

Once data is in place:

```bash
bash scripts/run_junyi_minimal.sh
bash scripts/run_assist_minimal.sh   # optional but recommended
```

You can also run all figure regeneration at once:

```bash
bash scripts/make_all_figures.sh
```

---

## 5. Per-stage commands

If you prefer to run stages one at a time (recommended while iterating):

```bash
# 1. Preprocess
python -m src.preprocess          --config configs/junyi.yaml

# 2. Build splits + sanity check
python -m src.split_checker       --config configs/junyi.yaml

# 3. Train-only graph construction
python -m src.graph_builder       --config configs/junyi.yaml

# 4. DAG audit (topological sort, cycle pruning)
python -m src.dag_audit           --config configs/junyi.yaml

# 5. DAG disruption sweep
python -m src.dag_disruption      --config configs/junyi.yaml

# 6. Cold-start KC diagnostic
python -m src.cold_start_report   --config configs/junyi.yaml

# 7. Baseline diagnostic (BKT, DKT, optional simpleKT/AKT)
python -m src.baseline_runner     --config configs/junyi.yaml

# 8. Aggregate report
python -m src.report_generator    --out results/reports/
```

Every command takes a `--seed` flag (default: 42) and a `--log-level` flag
(`INFO` by default).

---

## 6. Outputs

| File | Stage | Description |
|---|---|---|
| `results/tables/dataset_stats.csv` | preprocess | #learners, #items, #KCs, #interactions, avg seq length |
| `results/tables/graph_stats.csv` | graph_builder | per-fold edge counts, degree stats, root/leaf counts |
| `results/tables/dag_disruption.csv` | dag_disruption | DDR per augmentation × p × seed |
| `results/tables/baseline_results.csv` | baseline_runner | BKT/DKT diagnostic AUC/ACC/NLL |
| `results/figures/fig_ddr.pdf` | dag_disruption | line chart DDR vs p (4 augmentations) |
| `results/figures/fig_cold_start.pdf` | cold_start_report | per-stratum AUC bar chart |
| `results/reports/dag_report.md` | dag_audit | full DAG audit log |
| `results/reports/cold_start_report.md` | cold_start_report | cold-start KC diagnostic |
| `results/reports/p0_diagnostic_report.md` | report_generator | aggregated single-page summary |
| `logs/leakage_audit_log.csv` | every stage | edge provenance log; one row per edge |
| `logs/experiment_log.csv` | every stage | one row per script run |

---

## 7. Reproducing every figure and table

Each figure/table in the paper is produced by one script with one config.
The mapping is:

| Paper artefact | Script | Config |
|---|---|---|
| Table: Dataset statistics | `src/preprocess.py` | all configs |
| Table: Leakage audit | `src/split_checker.py` + log aggregator | all configs |
| Table: Graph audit | `src/graph_builder.py` + `src/dag_audit.py` | all configs |
| Figure: DDR vs p | `src/dag_disruption.py` | `junyi.yaml` (XES3G5M optional) |
| Figure / Table: Cold-start KC diagnostic | `src/cold_start_report.py` | all configs |
| Table: Baseline diagnostic | `src/baseline_runner.py` | `junyi.yaml`, `assist2012.yaml` |

Run `bash scripts/make_all_figures.sh` to refresh everything in one go.

---

## 8. Troubleshooting

**Import error after `pip install -e .`** — ensure you ran the install from
the repo root, and that the active interpreter matches your venv.

**Pipeline complains about a fold containing test users** — your raw data
or split config is wrong. Run `python -m src.split_checker --config <cfg>
--verbose` and inspect the report. Do not silence the assertion.

**`leakage_audit_log.csv` has rows with `train_only_flag = False`** —
**stop**. This is a hard fail. Identify the offending stage, fix it, and
re-run from preprocess.

**Reproduction gap > 5% from the original baseline paper** — that is
acceptable for a P0 diagnostic. Document the gap in
`results/reports/p0_diagnostic_report.md` under "Reproduction notes" and
do not tune hyperparameters to match.

**`fig_ddr.pdf` looks flat (DDR ~ 0 for all augmentations)** — verify the
DDR formula in `src/dag_disruption.py` matches the paper definition (see
Survey Roadmap §6.2). A flat curve is a valid empirical finding; report it
honestly in the manuscript and limitations section.

---

## 9. Project structure

```
p0-kc-graph-protocol/
├── README.md                ← this file
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── junyi.yaml
│   ├── assist2012.yaml
│   └── xes3g5m.yaml
├── data/
│   ├── raw/                 ← gitignored; place datasets here
│   └── processed/           ← gitignored; produced by preprocess.py
├── src/
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
│   ├── tables/
│   ├── figures/
│   └── reports/
├── logs/
│   ├── leakage_audit_log.csv
│   ├── literature_search_log.csv
│   └── experiment_log.csv
└── paper/
    └── refs.bib
```

---

## 10. Citation

If you use this code, please cite the P0 paper once it is released. A
placeholder BibTeX entry is provided in `paper/refs.bib` and will be
updated upon publication.

**Licence.** Code: MIT (or institution-specific — confirm with supervisor
before public release). Data: each dataset retains its original licence.

**Contact.** Dao Minh Tuan — `tuan.ymc@gmail.com`. PhD candidate,
GraphKT-ITS thesis. Issues and pull requests welcome via the repository.
