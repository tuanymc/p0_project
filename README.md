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
4. [Running the full pipeline (all benchmarks)](#4-running-the-full-pipeline-all-benchmarks)
5. [Per-stage commands](#5-per-stage-commands)
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
everything including paper tables (see [§4](#4-running-the-full-pipeline-all-benchmarks)).

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

### 2.3 Tests

From the repository root:

```bash
pytest -q
```

### 2.4 GPU (optional)

`torch` is required for DKT / simpleKT / AKT / GKT / GIKT style baselines.
CPU is acceptable for structural stages only. Install a CUDA build of
PyTorch that matches your stack if needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Data download and preparation

Public KT benchmarks only; respect each dataset licence.

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

## 4. Running the full pipeline (all benchmarks)

End-to-end on **Junyi, ASSISTments 2012, and XES3G5M**: preprocess (unless
parquet already exists), split check, graph build, DAG audit, DDR sweep,
baseline runner, cold-start report, **`scripts/generate_paper_artifacts.py`**,
and aggregated markdown report.

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

Regenerate **DDR line plots only** (three PDFs under `results/figures/`):

```bash
bash scripts/make_all_figures.sh
```

**Optional — Junyi ground-truth cross-validation** (train-only inferred
`e_pre_train_only.csv` vs expert DAG; requires preprocess/graph so that
`data/processed/junyi/kc_name_to_id.json` exists):

```bash
python scripts/run_gt_cross_validation_junyi.py
```

Outputs: `results/gt_validation/junyi/` (`overlap_metrics_at_K.csv`,
`fig_pr_curve.pdf`, `gt_validation_table.tex`, etc.).

---

## 5. Per-stage commands

Typical order for one config (e.g. `configs/junyi.yaml`):

```bash
python -m src.preprocess          --config configs/junyi.yaml
python -m src.split_checker       --config configs/junyi.yaml
python -m src.graph_builder       --config configs/junyi.yaml
python -m src.dag_audit           --config configs/junyi.yaml
python -m src.dag_disruption      --config configs/junyi.yaml
python -m src.baseline_runner     --config configs/junyi.yaml
python -m src.cold_start_report   --config configs/junyi.yaml
```

Paper-facing tables from already-produced CSVs (dataset stats, baseline TeX,
cold-start summary TeX, artefact index):

```bash
python scripts/generate_paper_artifacts.py
python -m src.report_generator --out results/reports/
```

Most modules accept `--seed` (default `42`) and `--log-level` (`INFO` default).

Fold-specific graph exports live under
`data/processed/<dataset>/fold_<k>/` (e.g. `e_pre_train_only.csv`,
`e_sim_train_only.csv`).

---

## 6. Outputs and where they live

| Location | Produced by | Role |
|----------|-------------|------|
| `data/processed/*.parquet` | `preprocess` | Canonical interaction tables |
| `data/processed/<ds>/fold_*/e_pre_train_only.csv` | `graph_builder` | Train-only prerequisite candidates |
| `data/processed/<ds>/fold_*/e_sim_train_only.csv` | `graph_builder` | Train-only similarity edges |
| `results/tables/dataset_stats.csv` (+ `.tex`) | `generate_paper_artifacts.py` | Dataset scale summary |
| `results/tables/graph_stats.csv` | `graph_builder` | Per-fold edge / KC counts |
| `results/tables/dag_audit_summary.csv` (+ `.tex`) | Manual / pipeline notes | Fold-wise DAG audit rows (`dag_audit` writes CSV) |
| `results/reports/<dataset>_dag_report.md` | `dag_audit` | Human-readable audit |
| `results/reports/<dataset>_dag_pruning_log.csv` | `dag_audit` | Pruned edges trail |
| `results/tables/dag_disruption.csv` | `dag_disruption` | Raw DDR rows (fold × aug × p × seed) |
| `results/tables/dag_disruption_summary.csv` | `dag_disruption` | Means/CIs used in paper DDR table |
| `results/figures/fig_ddr_<dataset>.pdf` | `dag_disruption` | DDR vs `p` line chart per dataset |
| `results/tables/baseline_results.csv` (+ `.tex`) | `baseline_runner` + `generate_paper_artifacts` | Multi-fold means + bootstrap CIs when enabled |
| `results/tables/cold_start_metrics.csv` (+ `.tex`) | `cold_start_report` + artefacts script | Stratum summaries |
| `results/tables/cold_start_by_stratum.tex` | From cold-start pipeline / artefacts | Per-stratum table for paper |
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
  `dag_audit_summary.tex`, `cold_start_by_stratum.tex`, `baseline_results.tex`,
  and GT material under `results/gt_validation/junyi/` (see `main.tex`).
- **Build tip:** compile LaTeX with the **repository root** as the working
  directory so paths such as `results/tables/...` and `results/figures/...`
  resolve. Figures `fig_ddr_*.pdf` appear after running `dag_disruption` (or
  `make_all_figures.sh`); `fig_pr_curve.pdf` for GT lives under
  `results/gt_validation/junyi/` after running the GT script.

Regeneration recipe aligned with the README scripts:

1. `./scripts/run_all_datasets_full.sh --server` (or the `.ps1` equivalent).  
2. `python scripts/run_gt_cross_validation_junyi.py` (optional).  
3. `latexmk` / `pdflatex` from repo root on `paper/main.tex`.

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
│   ├── baseline_runner.py
│   ├── gt_cross_validation.py
│   ├── report_generator.py
│   └── io_utils.py
├── scripts/
│   ├── run_junyi_minimal.sh
│   ├── run_assist_minimal.sh
│   ├── run_xes3g5m_minimal.sh
│   ├── run_all_datasets_full.sh
│   ├── run_all_datasets_full.ps1
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
