<#
.SYNOPSIS
    Full P0 pipeline: preprocess (optional skip), split, graphs, DAG probes, baselines, cold-start merge, paper artefacts.

.PARAMETER ForceFull
    Sets FORCE_PREPROCESS=1 so every dataset rebuilds parquet from raw CSV (use after schema/raw updates).

.PARAMETER ServerProfile
    Sensible defaults for a ~32GB RAM host: caps BLAS/OpenMP threads so pandas/sklearn do not oversubscribe.
    Does not configure CUDA; set CUDA_VISIBLE_DEVICES yourself if a stage uses GPU.

.EXAMPLE
    .\scripts\run_all_datasets_full.ps1 -ForceFull -ServerProfile

.EXAMPLE
    $env:CUDA_VISIBLE_DEVICES = "0"
    .\scripts\run_all_datasets_full.ps1 -ServerProfile

.NOTES
    Junyi full preprocess + graph + baseline is RAM-heavy; 32GB + `--ServerProfile` is appropriate.
    GPU (e.g. RTX 3090) helps only where PyTorch/deep baselines are actually used; current diagnostic
    runners are mostly CPU (numpy/sklearn). Keep CUDA env for future or optional torch workloads.
#>

param(
    [switch]$ForceFull,
    [switch]$ServerProfile
)

$ErrorActionPreference = "Stop"

if ($ForceFull) {
    $env:FORCE_PREPROCESS = "1"
    Write-Host "[run_all_datasets_full] ForceFull: FORCE_PREPROCESS=1 (rebuild parquet from raw)."
}

if ($ServerProfile) {
    $threadVars = @(
        @{ Name = "OMP_NUM_THREADS"; Default = "8" },
        @{ Name = "MKL_NUM_THREADS"; Default = "8" },
        @{ Name = "OPENBLAS_NUM_THREADS"; Default = "8" },
        @{ Name = "NUMEXPR_NUM_THREADS"; Default = "8" }
    )
    foreach ($tv in $threadVars) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($tv.Name, "Process"))) {
            Set-Item -Path "env:$($tv.Name)" -Value $tv.Default
        }
    }
    if ([string]::IsNullOrEmpty($env:PYTHONHASHSEED)) {
        $env:PYTHONHASHSEED = "0"
    }
    Write-Host "[run_all_datasets_full] ServerProfile: BLAS/thread caps (override by setting env vars before launch)."
    Write-Host "    OMP_NUM_THREADS=$($env:OMP_NUM_THREADS) MKL_NUM_THREADS=$($env:MKL_NUM_THREADS) OPENBLAS_NUM_THREADS=$($env:OPENBLAS_NUM_THREADS)"
}

if ($env:PYTHON) {
    $Python = $env:PYTHON
}
else {
    $Python = "python"
}

Write-Host "[run_all_datasets_full] Using interpreter: $Python"
Write-Host "[run_all_datasets_full] FORCE_PREPROCESS=$($env:FORCE_PREPROCESS) CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"

function Invoke-P0Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $($Arguments -join ' ')"
    }
}

$Datasets = @(
    @{
        Config = "configs/junyi.yaml"
        Processed = "data/processed/junyi.parquet"
    },
    @{
        Config = "configs/assist2012.yaml"
        Processed = "data/processed/assist2012.parquet"
    },
    @{
        Config = "configs/xes3g5m.yaml"
        Processed = "data/processed/xes3g5m.parquet"
    }
)

foreach ($Dataset in $Datasets) {
    $Cfg = $Dataset.Config
    $Processed = $Dataset.Processed
    Write-Host "==> Running full P0 pipeline for $Cfg"
    if ((Test-Path $Processed) -and ($env:FORCE_PREPROCESS -ne "1")) {
        Write-Host "    Skipping preprocess; found $Processed"
    }
    else {
        Invoke-P0Step @("-m", "src.preprocess", "--config", $Cfg)
    }
    Invoke-P0Step @("-m", "src.split_checker", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.graph_builder", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.dag_audit", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.dag_disruption", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.baseline_runner", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.cold_start_report", "--config", $Cfg)
}

Write-Host "==> Generating paper artefacts"
Invoke-P0Step @("scripts/generate_paper_artifacts.py")
Invoke-P0Step @("-m", "src.report_generator", "--out", "results/reports/")

Write-Host "==> Done. Main report: results/reports/p0_diagnostic_report.md"
Write-Host "    Optional Junyi GT CV: python scripts/run_gt_cross_validation_junyi.py (needs kc_name_to_id.json)"
