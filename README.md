# P0: Leakage-Controlled KC Graph Construction & Cold-Start Diagnostic Protocol

Companion code repository for the resource paper *Leakage-Controlled Concept
Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing*
(LLNC-style manuscript under `paper/main.tex`, thesis GraphKT-ITS track).

> **What this repo is.** A protocol and audit pipeline for building and
> checking multi-relational concept graphs from KT logs under train-only,
> fold-aware discipline; cold-start KC stratification; a DAG Disruption Rate
> (**DDR**) probe over generic graph augmentations; and optional **ground-truth
> cross-validation** on Junyi (expert prerequisite DAG vs train-only inferred
> edges).
>
> **What this repo is NOT.** A new KT baseline aimed at SOTA. No claim about
> real-world learning outcomes or joint self-supervised graph pretraining.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Environment setup](#2-environment-setup)
3. [Data download and preparation](#3-data-download-and-preparation)
4. [Running experiments](#4-running-experiments) ([step-by-step](#40-step-by-step-experiment-guide))
5. [Per-stage commands](#5-per-stage-commands) ([graph_builder API](#51-graph_builder-python-api))
6. [Outputs and where they live](#6-outputs-and-where-they-live)
7. [Paper artefacts and LaTeX paths](#7-paper-artefacts-and-latex-paths)
8. [Troubleshooting](#8-troubleshooting)
9. [Project structure](#9-project-structure)
10. [Citation, licence, and contact](#10-citation-licence-and-contact)

---

## 1. Quick start

After [§2 Environment setup](#2-environment-setup) and
[§3 Data](#3-data-download-and-preparation), run a **single-dataset** smoke
pipeline (Junyi):

```bash
bash scripts/run_junyi_minimal.sh
```

On **Windows** (PowerShell, repo root):

```powershell
$env:PYTHON = "python"   # optional
bash scripts/run_junyi_minimal.sh
```

If you do not have Git Bash/WSL, run the same stages by hand (see
[§5](#5-per-stage-commands)); or use `scripts/run_all_datasets_full.ps1` for
everything including paper tables (see [§4](#4-running-experiments)). For a
**numbered walkthrough** of setup and experiment tracks, start at
[§4.0](#40-step-by-step-experiment-guide).

**Runtime.** Roughly tens of minutes on a laptop CPU for Junyi end-to-end if
deep baselines (`torch`) are enabled; longer for full three-dataset runs and
multi-fold baselines. Junyi preprocess + graph stages are memory-heavy; prefer
`--server` / `-ServerProfile` on ~32 GB RAM hosts (see full pipeline scripts).

**Sanity.** `split_checker` should report no learner leakage and temporal
ordering OK. `dag_audit` may report `cycles_before` hitting the **representative
cycle cap (100)** on dense graphs; the pruning loop still runs until the graph
is acyclic (see `paper/main.tex` / `src/dag_audit.py`).

**Reports.** `results/reports/p0_diagnostic_report.md` aggregates available
CSVs and markdown reports.

---

## 2. Environment setup

Tested with **Python 3.10 / 3.11** (3.12+ often works). Linux, macOS, and
**Windows** (PowerShell + native Python) are supported; WSL2 remains optional.

### 2.1 Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2.2 Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

**Optional pyKT backend.** If you plan to run `baseline_runner` with real PyTorch / `pykt-toolkit` training (`evaluation.baseline_backend: pykt` in YAML or `--baseline-backend pykt`), clone **with submodules** or run:

```bash
git submodule update --init --recursive
```

Then install extras:

```bash
pip install -e ".[pykt]"
```

(`pykt-toolkit` is pinned via `third_party/pykt-toolkit`; see `third_party/README.md`.)

### 2.3 Tests

From the repository root:

```bash
pytest -q
```

### 2.4 GPU (optional)

Structural stages (`preprocess` → `graph_builder` → `dag_*`) use NumPy/pandas only.

PyTorch / CUDA matters only when you install **`pip install -e ".[pykt]"`** and run
`baseline_runner` with **`evaluation.baseline_backend: pykt`** (or `--baseline-backend pykt`).
Install a CUDA build that matches your driver when appropriate:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Data download and preparation

Public KT benchmarks only; respect each dataset licence.

**Optional bundle.** A project-maintained Google Drive folder with the expected
subfolders (`raw/`, `processed/`, `graphs/`, plus a short `README.md` inside the
drive) is here: [P0 data (Google Drive)](https://drive.google.com/drive/folders/1eQNSTV0pVDeB79Mx--vnPml_YPYwVzDP?usp=sharing).
Download what you need and place files under your local `data/` tree as in
[§3.1](#31-layout); licences of the underlying benchmarks still apply.

| Dataset | Role in P0 | External prerequisite DAG | Notes |
|--------|------------|----------------------------|--------|
| Junyi Academy | Core | Yes (`junyi_dag.csv`) | Expert DAG used only in optional GT CV; train-only prerequisite edges `E_pre` are inferred like other benchmarks. |
| ASSISTments 2012 | Core | No | Q-matrix only; `E_pre` inferred from train transitions. |
| XES3G5M | Core | Metadata only | KC IDs from hierarchy labels; `E_pre` inferred from train transitions. |

### 3.1 Layout

Place raw files under `data/raw/<dataset>/` (directories are gitignored).
Example for Junyi (filenames must match `configs/junyi.yaml`):

```
data/raw/junyi/
├── junyi_ProblemLog_original.csv   # interaction log (Chang et al. style)
├── junyi_dag.csv                   # expert prerequisite annotation (GT CV)
└── junyi_Exercise_table.csv       # used when building KC name→id mapping for GT CV
```

ASSISTments and XES3G5M paths are defined in `configs/assist2012.yaml` and
`configs/xes3g5m.yaml`.

### 3.2 Preprocess

Output path defaults from `processed_path` in each YAML. Override optional:

```bash
python -m src.preprocess --config configs/junyi.yaml
# Optional explicit output:
python -m src.preprocess --config configs/junyi.yaml --out data/processed/junyi.parquet
```

Canonical columns:
`user_id, item_id, kc_id, timestamp, correct` → parquet under `data/processed/`.

### 3.3 Split check

```bash
python -m src.split_checker --config configs/junyi.yaml
```

Stop and fix any learner overlap or temporal violations before continuing.

---

## 4. Running experiments

Use this section to reproduce paper-facing numbers and optional ablations.
Always run commands from the **repository root** with the venv that has
`pip install -e .` applied.

### 4.0 Step-by-step experiment guide

Follow **A → B** once per machine; then choose **one track** under **C**. Commands below use `configs/junyi.yaml` as an example; swap for `configs/assist2012.yaml` or `configs/xes3g5m.yaml` when working on other benchmarks.

#### A. First-time setup

1. **Clone** this repository. For the optional **pyKT** neural backend, fetch the pinned submodule:
   ```bash
   git submodule update --init --recursive
   ```
2. **Create and activate** a virtual environment ([§2.1](#21-virtual-environment)).
3. **Install core dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
4. **(Optional)** Install PyTorch + submodule-backed **`pykt-toolkit`** for `--baseline-backend pykt` / `evaluation.baseline_backend: pykt`:
   ```bash
   pip install -e ".[pykt]"
   ```
   Default YAML in this repo keeps **`diagnostic`** ensembles unless you change `evaluation.baseline_backend` or pass `--baseline-backend pykt` to `baseline_runner`.
5. **Sanity check:** `pytest -q` ([§2.3](#23-tests)).

#### B. Data preparation (per benchmark)

6. **Place raw files** under `data/raw/<dataset>/` as required by each YAML ([§3.1](#31-layout)).
7. **Preprocess** to canonical parquet (repeat for each dataset you need):
   ```bash
   python -m src.preprocess --config configs/junyi.yaml
   ```
8. **Split audit** — fix any failures before graphs or baselines:
   ```bash
   python -m src.split_checker --config configs/junyi.yaml
   ```

#### C. Choose an experiment track

**Track 1 — Smoke / one dataset (Junyi)**  
9. After **A** and **B** for Junyi, run the bundled minimal pipeline:
   ```bash
   bash scripts/run_junyi_minimal.sh
   ```
   On **Windows** without Git Bash:
   ```powershell
   $env:PYTHON = "python"   # optional
   bash scripts/run_junyi_minimal.sh
   ```
   If Bash is unavailable, run the same stages manually in [§5](#5-per-stage-commands).

**Track 2 — Full protocol (all three benchmarks, paper-scale)**  
10. Ensure raw inputs exist for **Junyi, ASSISTments 2012, and XES3G5M**.  
11. Run the orchestrator (RAM/thread notes in [§4.2](#42-full-pipeline-all-benchmarks)):
    ```bash
    chmod +x scripts/run_all_datasets_full.sh
    ./scripts/run_all_datasets_full.sh --server
    ```
    ```powershell
    .\scripts\run_all_datasets_full.ps1 -ServerProfile
    ```
12. **Post-process artefacts:** tables TeX + aggregated report:
    ```bash
    python scripts/generate_paper_artifacts.py
    python -m src.report_generator --out results/reports/
    ```

**Track 3 — Graph ablation H1 (train-only vs full-log diagnostics)**  
13. Requires parquet for each dataset you include (preprocess or Track 2 preprocess stage).  
14. Run (details and flags: [`scripts/GRAPH_ABLATION_EXPERIMENT.md`](scripts/GRAPH_ABLATION_EXPERIMENT.md)):
    ```bash
    chmod +x scripts/run_graph_ablation_experiment.sh
    SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh
    ```
    ```powershell
    .\scripts\run_graph_ablation_experiment.ps1 -ServerProfile
    ```
15. Refresh paper snippets: `python scripts/generate_paper_artifacts.py`.

**Track 4 — Junyi ground-truth cross-validation**  
16. Requires preprocess + graph stages so `data/processed/junyi/kc_name_to_id.json` exists.  
17. Run:
    ```bash
    python scripts/run_gt_cross_validation_junyi.py
    ```

**Track 5 — Regenerate figures / TeX only**  
18. DDR line figures (after `dag_disruption` CSVs exist): `bash scripts/make_all_figures.sh`.  
19. Paper `\input{...}` tables from existing CSVs: `python scripts/generate_paper_artifacts.py`.

For **manual stage-by-stage** control on a single config (debugging), use the ordered CLI list in [§5](#5-per-stage-commands).

### 4.1 Experiment map

| Goal | Command | Main artefacts |
|------|---------|------------------|
| **Full protocol** (preprocess → graphs → DDR → baselines → cold-start → TeX) | `scripts/run_all_datasets_full.sh` or `scripts/run_all_datasets_full.ps1` | `results/tables/*.csv`/`.tex`, `results/figures/fig_ddr_*.pdf`, `results/reports/p0_diagnostic_report.md` |
| **Smoke / one dataset** | `scripts/run_*_minimal.sh` | Same layout under `data/processed/<dataset>/` and `results/` for that config |
| **Graph ablation** (train-only vs full-log graphs, graph-augmented diagnostics) | `scripts/run_graph_ablation_experiment.sh` or `scripts/run_graph_ablation_experiment.ps1` | `results/tables/graph_ablation_summary.csv`, `graph_ablation.tex`; see `scripts/GRAPH_ABLATION_EXPERIMENT.md` |
| **Junyi GT vs expert DAG** | `python scripts/run_gt_cross_validation_junyi.py` | `results/gt_validation/junyi/*` |
| **Paper tables only** (CSVs already produced) | `python scripts/generate_paper_artifacts.py` | Regenerates `\input{results/tables/...}` snippets |

**Leakage diagnostics** (`ECR_flag`, `ECR_overlap`, `EOC`, `TBVR`) are written to
`results/tables/leakage_metrics.csv` when you run **`python -m src.graph_builder`**
(or console **`p0-graph-build`** after editable install). Fold means are typeset
via `generate_paper_artifacts.py` → `results/tables/leakage_metrics.tex`.

### 4.2 Full pipeline (all benchmarks)

End-to-end on **Junyi, ASSISTments 2012, and XES3G5M**, **per dataset**, in order:

1. `preprocess` (skipped if the dataset parquet exists unless forced)
2. `split_checker`
3. `graph_builder` (fold-wise train-only graphs + leakage row merge)
4. `export_full_log_graph` (full-log prerequisite/similarity exports for ablations / `graph_construction` contrasts)
5. `dag_audit`
6. `dag_disruption`
7. `baseline_runner` — on **Junyi**, `run_all_datasets_full.ps1` appends **`--skip-cold-start`** to limit RAM (cold-start strata skipped; AUC/ACC/NLL still run on val+test). The Bash script calls the baseline **without** that flag; if Junyi exhausts memory on Linux/macOS, rerun step 7 manually with `--skip-cold-start` for `configs/junyi.yaml` only.
8. `cold_start_report`

Then globally: **`scripts/generate_paper_artifacts.py`** and
**`python -m src.report_generator --out results/reports/`**.

**Linux / macOS (Git Bash):**

```bash
chmod +x scripts/run_all_datasets_full.sh
./scripts/run_all_datasets_full.sh              # skip preprocess if parquet exists
./scripts/run_all_datasets_full.sh --force-full # rebuild all parquet from raw
./scripts/run_all_datasets_full.sh --server     # BLAS/thread caps for ~32 GB RAM
```

**Windows (PowerShell, repo root):**

```powershell
.\scripts\run_all_datasets_full.ps1
.\scripts\run_all_datasets_full.ps1 -ForceFull      # set FORCE_PREPROCESS=1
.\scripts\run_all_datasets_full.ps1 -ServerProfile   # thread caps + PYTHONHASHSEED
```

Minimal per-dataset scripts (no cross-dataset paper aggregation):

- `scripts/run_junyi_minimal.sh`
- `scripts/run_assist_minimal.sh`
- `scripts/run_xes3g5m_minimal.sh`

### 4.3 Graph ablation (train-only vs full-log, H1)

After parquet exists (from preprocess or the full pipeline), run:

```bash
chmod +x scripts/run_graph_ablation_experiment.sh
SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh
```

```powershell
.\scripts\run_graph_ablation_experiment.ps1 -ServerProfile
```

Requires `graph_ablation.enabled: true` and a `models` list (e.g.\ GKT, GIKT, SKT, DyGKT, DGEKT) in each
`configs/*.yaml`. Full prerequisites, flags (`-SkipGraphBuild`, Junyi RAM notes),
and output filenames are documented in **`scripts/GRAPH_ABLATION_EXPERIMENT.md`**.

### 4.4 DDR figures only

Regenerate the three DDR line PDFs under `results/figures/`:

```bash
bash scripts/make_all_figures.sh
```

### 4.5 Junyi ground-truth cross-validation

Train-only inferred `e_pre_train_only.csv` vs expert DAG; requires preprocess/graph so that
`data/processed/junyi/kc_name_to_id.json` exists:

```bash
python scripts/run_gt_cross_validation_junyi.py
```

For lightweight **directed / undirected overlap at @K** on your own edge
`DataFrame`s (with a `support` column on the inferred side), use
`evaluate_inferred_against_ground_truth` in `src/graph_builder.py` — see
[§5.1](#51-graph_builder-python-api).

Outputs: `results/gt_validation/junyi/` (`overlap_metrics_at_K.csv`,
`fig_pr_curve.pdf`, `gt_validation_table.tex`, etc.).

---

## 5. Per-stage commands

Typical order for one config (e.g. `configs/junyi.yaml`):

```bash
python -m src.preprocess          --config configs/junyi.yaml
python -m src.split_checker       --config configs/junyi.yaml
python -m src.graph_builder       --config configs/junyi.yaml
python -m src.export_full_log_graph --config configs/junyi.yaml
python -m src.dag_audit           --config configs/junyi.yaml
python -m src.dag_disruption      --config configs/junyi.yaml
python -m src.baseline_runner     --config configs/junyi.yaml   # add --skip-cold-start on Junyi if RAM-limited
python -m src.cold_start_report   --config configs/junyi.yaml
```

Paper-facing tables from already-produced CSVs (dataset stats, baseline TeX,
cold-start summary TeX, artefact index):

```bash
python scripts/generate_paper_artifacts.py
python -m src.report_generator --out results/reports/
```

Most modules accept `--seed` (default `42`) and `--log-level` (`INFO` default).
After `pip install -e .`, optional CLI aliases such as `p0-graph-build` and
`p0-baseline` are defined in `pyproject.toml` and mirror the same flags as
`python -m src.graph_builder` / `src.baseline_runner`.

Fold-specific graph exports live under
`data/processed/<dataset>/fold_<k>/` (e.g. `e_pre_train_only.csv`,
`e_sim_train_only.csv`). Full-log ablation graphs (when exported) live under
`data/processed/<dataset>/full_log/`.

### 5.1 `graph_builder` Python API

Besides the CLI (`python -m src.graph_builder`), `src/graph_builder.py` exposes
train-only helpers for scripts and experiments:

| Function | Role |
|----------|------|
| `build_q_matrix_from_train` | Item–KC table from train interactions only (enforces train-only / single-fold discipline). |
| `infer_prerequisites_from_train` | Directed prerequisite candidates from temporal KC transitions (`support`, `weight`, …). |
| `infer_similarity_edges_from_train` | KC–KC similarity edges (Jaccard or PMI). Fails if `fold` column mixes multiple fold ids. |
| `load_ground_truth_dag` / `dataset_has_independent_prerequisite_graph` | Optional expert DAG loading and dataset capability checks (Junyi / XES-style layouts). |

**Ground-truth overlap at @K** — `evaluate_inferred_against_ground_truth(inferred, expert, top_k_list)` compares a directed inferred edge table to an expert edge list (columns `src_kc`, `dst_kc`). Requirements and behaviour:

- **`inferred`** must include a numeric **`support`** column (same role as in prerequisite outputs). For each `K` in `top_k_list`, the top-*K* rows by descending `support` are evaluated.
- **Directed hit:** same ordered pair as an expert row.
- **Undirected hit:** the unordered pair matches an expert edge in either direction (so an inferred `(a,b)` can match an expert `(b,a)`).
- **Returns** a `DataFrame` with one row per `K`: `k`, `directed_hits`, `undirected_hits`, `directed_precision` (= directed / undirected hits in that top-*K* set, or `0.0` if none), and `directed_recall` (= directed hits / number of expert rows).

See `tests/test_graph_builder_train_only.py` for examples.

---

## 6. Outputs and where they live

| Location | Produced by | Role |
|----------|-------------|------|
| `data/processed/*.parquet` | `preprocess` | Canonical interaction tables |
| `data/processed/<ds>/fold_*/e_pre_train_only.csv` | `graph_builder` | Train-only prerequisite candidates |
| `data/processed/<ds>/fold_*/e_sim_train_only.csv` | `graph_builder` | Train-only similarity edges |
| `results/tables/dataset_stats.csv` (+ `.tex`) | `generate_paper_artifacts.py` | Dataset scale summary |
| `results/tables/graph_stats.csv` | `graph_builder` | Per-fold edge / KC counts |
| `results/tables/leakage_metrics.csv` (+ `.tex`) | `graph_builder` + `generate_paper_artifacts.py` | Fold-wise leakage diagnostics (`ECR_flag`, `ECR_overlap`, …); TeX is fold-mean summary |
| `results/tables/graph_ablation_summary.csv` (+ `.tex`) | `baseline_runner` with `graph_ablation` + artefacts script | Train-only vs full-log graph-augmented diagnostics (models from YAML) |
| `results/tables/dag_audit_summary.csv` (+ `.tex`) | `dag_audit` + `generate_paper_artifacts.py` | Fold-wise DAG audit (`dag_audit` writes CSV; artefacts backfill `n_edges_raw` / `n_edges_pruned` from pruning logs and emit IEEE TeX) |
| `results/reports/<dataset>_dag_report.md` | `dag_audit` | Human-readable audit |
| `results/reports/<dataset>_dag_pruning_log.csv` | `dag_audit` | Pruned edges trail |
| `results/tables/dag_disruption.csv` | `dag_disruption` | Raw DDR rows (fold × aug × p × seed) |
| `results/tables/dag_disruption_summary.csv` | `dag_disruption` | Means/CIs used in paper DDR table |
| `results/figures/fig_ddr_<dataset>.pdf` | `dag_disruption` | DDR vs `p` line chart per dataset |
| `results/tables/baseline_results.csv` (+ `.tex`) | `baseline_runner` + `generate_paper_artifacts` | Multi-fold means + bootstrap CIs when enabled |
| `results/tables/cold_start_metrics.csv` (+ `.tex`) | `cold_start_report` + artefacts script | Stratum summaries |
| `results/tables/cold_start_by_stratum.tex` | `generate_paper_artifacts.py` | Paper table from `cold_start_metrics.csv` (default: fold~0 \textit{simpleKT}) |
| `results/reports/cold_start_report.md` | `cold_start_report` | Narrative cold-start report |
| `results/reports/paper_artifact_index.md` | `generate_paper_artifacts.py` | Index of tables/figures/reports |
| `results/reports/p0_diagnostic_report.md` | `report_generator` | Aggregated diagnostic markdown |
| `results/gt_validation/junyi/*` | `run_gt_cross_validation_junyi.py` | GT overlap metrics, PR curve, TeX snippet |
| `logs/leakage_audit_log.csv` | Multiple stages | Edge provenance audit trail |
| `logs/experiment_log.csv` | Run hooks | Run ledger (if configured) |

---

## 7. Paper artefacts and LaTeX paths

- **Manuscript:** `paper/main.tex`, bibliography `paper/refs.bib`.
- **Inputs pulled from `results/`:** `\input{results/tables/dataset_stats.tex}`,
  `leakage_metrics.tex`, `dag_audit_summary.tex`, `cold_start_by_stratum.tex`,
  `baseline_results.tex`, `graph_ablation.tex` (when generated), and GT material under `results/gt_validation/junyi/` (see `main.tex`).
- **Build tip:** compile LaTeX with the **repository root** as the working
  directory so paths such as `results/tables/...` and `results/figures/...`
  resolve. Figures `fig_ddr_*.pdf` appear after running `dag_disruption` (or
  `make_all_figures.sh`); `fig_pr_curve.pdf` for GT lives under
  `results/gt_validation/junyi/` after running the GT script.

Regeneration recipe aligned with the README scripts:

1. `./scripts/run_all_datasets_full.sh --server` (or the `.ps1` equivalent).  
2. Optional H1 graph ablation: `SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh` or `.\scripts\run_graph_ablation_experiment.ps1 -ServerProfile`, then `python scripts/generate_paper_artifacts.py`.  
3. `python scripts/run_gt_cross_validation_junyi.py` (optional).  
4. `latexmk` / `pdflatex` from repo root on `paper/main.tex`.

---

## 8. Troubleshooting

**Import errors after `pip install -e .`** — Run commands from the repo root;
ensure the active interpreter is the venv you installed into.

**Split checker failures** — Inspect `python -m src.split_checker --config …`
(do not bypass assertions). Raw CSV paths and schema mapping in YAML must
match your download.

**`leakage_audit_log.csv` contains non-train-only edges** — Treat as a hard
protocol violation: fix the offending stage and rerun from a clean state.

**DDR plots missing** — Run `python -m src.dag_disruption --config configs/<ds>.yaml`
for each dataset, or `bash scripts/make_all_figures.sh`.

**Junyi GT CV missing mapping** — Ensure preprocess/graph ran so
`data/processed/junyi/kc_name_to_id.json` exists; expert CSV paths match
`run_gt_cross_validation_junyi.py` defaults.

**Baseline RAM / time** — Reduce folds in YAML (`split.n_folds`) or disable
heavy models in `configs/*.yaml` under `baselines:` while debugging structure-only stages.

**`baseline_backend=pykt` import errors** — Run `git submodule update --init --recursive`
and `pip install -e ".[pykt]"` from the repo root (see [§2.2](#22-dependencies)).

**Junyi baseline RAM (Linux/macOS full pipeline)** — `scripts/run_all_datasets_full.sh` invokes `baseline_runner` without `--skip-cold-start`; Windows `run_all_datasets_full.ps1` adds `--skip-cold-start` for Junyi.
If you hit OOM on Bash/WSL, rerun only that stage:
`python -m src.baseline_runner --config configs/junyi.yaml --skip-cold-start`.

---

## 9. Project structure

```
p0_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── junyi.yaml
│   ├── assist2012.yaml
│   └── xes3g5m.yaml
├── paper/
│   ├── main.tex
│   └── refs.bib
├── data/
│   ├── raw/              # gitignored — place benchmarks here
│   └── processed/        # gitignored — parquet + fold exports
├── src/
│   ├── preprocess.py
│   ├── split_checker.py
│   ├── graph_builder.py
│   ├── dag_audit.py
│   ├── dag_disruption.py
│   ├── cold_start_report.py
│   ├── export_full_log_graph.py
│   ├── baseline_runner.py
│   ├── gt_cross_validation.py
│   ├── report_generator.py
│   └── io_utils.py
├── third_party/
│   ├── README.md
│   └── pykt-toolkit/          # git submodule (optional pyKT backend)
├── scripts/
│   ├── run_junyi_minimal.sh
│   ├── run_assist_minimal.sh
│   ├── run_xes3g5m_minimal.sh
│   ├── run_all_datasets_full.sh
│   ├── run_all_datasets_full.ps1
│   ├── run_graph_ablation_experiment.sh
│   ├── run_graph_ablation_experiment.ps1
│   ├── GRAPH_ABLATION_EXPERIMENT.md
│   ├── make_all_figures.sh
│   ├── generate_paper_artifacts.py
│   ├── run_gt_cross_validation_junyi.py
│   └── …
├── tests/
├── results/
│   ├── tables/
│   ├── figures/
│   ├── reports/
│   └── gt_validation/junyi/   # after GT script
└── logs/
```

---

## 10. Citation, licence, and contact

If you use this code or protocol, please cite the P0 paper once it is public.
BibTeX placeholders live in `paper/refs.bib` and will be updated on publication.

**Licence.** Confirm code licence (e.g. MIT) with your institution before a
public release. Dataset licences remain with their respective publishers.

**Contact.** Dao Minh Tuan — `tuan.ymc@gmail.com`. Issues and improvements
welcome via the repository host (e.g.
https://github.com/tuanymc/p0_project.git).
