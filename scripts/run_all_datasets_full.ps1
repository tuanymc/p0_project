$ErrorActionPreference = "Stop"

if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $Python = "python"
}

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
    } else {
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
