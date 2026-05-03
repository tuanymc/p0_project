param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("junyi", "assist2012", "xes3g5m")]
    [string] $Dataset,

    [Parameter(Mandatory = $true)]
    [ValidateSet("preprocess", "split", "graph", "dag", "ddr", "baseline", "cold", "paper", "report")]
    [string] $Stage,

    [switch] $ForcePreprocess
)

$ErrorActionPreference = "Stop"

if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $Python = "python"
}

$Configs = @{
    junyi = "configs/junyi.yaml"
    assist2012 = "configs/assist2012.yaml"
    xes3g5m = "configs/xes3g5m.yaml"
}

$Processed = @{
    junyi = "data/processed/junyi.parquet"
    assist2012 = "data/processed/assist2012.parquet"
    xes3g5m = "data/processed/xes3g5m.parquet"
}

function Invoke-P0Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    Write-Host "==> $Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $($Arguments -join ' ')"
    }
}

$Cfg = $Configs[$Dataset]
$ProcessedPath = $Processed[$Dataset]

Write-Host "Dataset: $Dataset"
Write-Host "Stage:   $Stage"

switch ($Stage) {
    "preprocess" {
        if ((Test-Path $ProcessedPath) -and (-not $ForcePreprocess)) {
            Write-Host "Skipping preprocess; found $ProcessedPath. Use -ForcePreprocess to rebuild."
        } else {
            Invoke-P0Step @("-m", "src.preprocess", "--config", $Cfg)
        }
    }
    "split" {
        Invoke-P0Step @("-m", "src.split_checker", "--config", $Cfg)
    }
    "graph" {
        Invoke-P0Step @("-m", "src.graph_builder", "--config", $Cfg)
    }
    "dag" {
        Invoke-P0Step @("-m", "src.dag_audit", "--config", $Cfg)
    }
    "ddr" {
        Invoke-P0Step @("-m", "src.dag_disruption", "--config", $Cfg)
    }
    "baseline" {
        Invoke-P0Step @("-m", "src.baseline_runner", "--config", $Cfg)
    }
    "cold" {
        Invoke-P0Step @("-m", "src.cold_start_report", "--config", $Cfg)
    }
    "paper" {
        Invoke-P0Step @("scripts/generate_paper_artifacts.py")
    }
    "report" {
        Invoke-P0Step @("-m", "src.report_generator", "--out", "results/reports/")
    }
}

Write-Host "==> Done: $Dataset / $Stage"
