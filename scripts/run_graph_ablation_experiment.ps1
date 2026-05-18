<#
.SYNOPSIS
    Pipeline ablation: đồ thị train-only (theo fold) vs đồ thị full-log + baseline GKT/GIKT,
    sinh results/tables/graph_ablation_summary.csv và graph_ablation.tex.

.PARAMETER ServerProfile
    Giới hạn thread BLAS/OpenMP (khuyến nghị máy ~16-32 GB RAM).

.PARAMETER SkipGraphBuild
    Bỏ qua bước p0-graph-build nếu đã có data/processed/<ds>/fold_*/.

.PARAMETER SkipFullLogExport
    Bỏ qua export full-log nếu đã có data/processed/<ds>/full_log/e_pre.csv.

.PARAMETER SkipBaseline
    Chỉ export đồ thị / không chạy baseline.

.EXAMPLE
    cd <repo-root>
    .\scripts\run_graph_ablation_experiment.ps1 -ServerProfile

.NOTES
    Junyi: baseline gọi --skip-cold-start để tránh MemoryError khi tính cold-start trên ~8M dòng val+test.
    Cần cài package: pip install -e .
    Cần parquet: data/processed/junyi.parquet, assist2012.parquet, xes3g5m.parquet (preprocess trước nếu thiếu).

    Tiến trình: thanh Write-Progress + [bước/tổng] và % - không có % chi tiết *bên trong* baseline_runner
    (PyTorch); muốn log chi tiết hơn thêm --log-level DEBUG khi gọi baseline_runner trong mã nguồn.
#>

param(
    [switch]$ServerProfile,
    [switch]$SkipGraphBuild,
    [switch]$SkipFullLogExport,
    [switch]$SkipBaseline
)

$ErrorActionPreference = "Stop"

if ($ServerProfile) {
    foreach ($name in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name, "Process"))) {
            Set-Item -Path "env:$name" -Value "8"
        }
    }
    Write-Host "[graph_ablation] ServerProfile: OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
}

if ($env:PYTHON) { $Python = $env:PYTHON } else { $Python = "python" }

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    Write-Host ">> $Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $Python $($Arguments -join ' ')"
    }
}

$Datasets = @(
    @{ Config = "configs/junyi.yaml"; Key = "junyi" }
    @{ Config = "configs/assist2012.yaml"; Key = "assist2012" }
    @{ Config = "configs/xes3g5m.yaml"; Key = "xes3g5m" }
)

# Hàng đợi các bước [i/N] và % (baseline_runner là bước lâu, không có % chi tiết bên trong model).
$Steps = [System.Collections.Generic.List[hashtable]]::new()

foreach ($ds in $Datasets) {
    $cfg = $ds.Config
    $key = $ds.Key

    if (-not $SkipGraphBuild) {
        $Steps.Add(@{
                Dataset = $key
                Label   = "graph_builder"
                Args    = @("-m", "src.graph_builder", "--config", $cfg)
            })
    }
    if (-not $SkipFullLogExport) {
        $Steps.Add(@{
                Dataset = $key
                Label   = "export_full_log_graph"
                Args    = @("-m", "src.export_full_log_graph", "--config", $cfg)
            })
    }
    if (-not $SkipBaseline) {
        $baselineArgs = @("-m", "src.baseline_runner", "--config", $cfg)
        if ($key -eq "junyi") {
            $baselineArgs = $baselineArgs + "--skip-cold-start"
        }
        $Steps.Add(@{
                Dataset = $key
                Label   = "baseline_runner"
                Args    = $baselineArgs
            })
    }
}

$Steps.Add(@{
        Dataset = "(all)"
        Label   = "generate_paper_artifacts"
        Args    = @("scripts/generate_paper_artifacts.py")
    })

$total = $Steps.Count
if ($total -eq 0) {
    Write-Host "[graph_ablation] Không có bước nào (kiểm tra Skip*)."
    exit 0
}

Write-Host ""
Write-Host "[graph_ablation] Tổng $total bước (ước lượng tiến trình theo bước, không theo epoch trong model)."
Write-Host ""

for ($idx = 0; $idx -lt $total; $idx++) {
    $step = $Steps[$idx]
    $stepNum = $idx + 1
    $pctBefore = [math]::Min(99, [int](100 * $idx / $total))
    $statusLine = "[$stepNum/$total] (~$pctBefore%) $($step.Dataset) - $($step.Label)"

    Write-Progress `
        -Activity "Graph ablation experiment" `
        -Status $statusLine `
        -CurrentOperation "$Python $($step.Args -join ' ')" `
        -PercentComplete $pctBefore

    Write-Host "---------- $statusLine ----------"

    if ($step.Dataset -eq "junyi" -and $step.Label -eq "baseline_runner") {
        Write-Host "[junyi] baseline với --skip-cold-start (tránh OOM cold-start)"
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Step -Arguments $step.Args
    $sw.Stop()

    $pctAfter = [int](100 * $stepNum / $total)
    Write-Host "[xong] $($step.Label) - $($sw.Elapsed.ToString('hh\:mm\:ss\.fff')) (hoàn thành ~$pctAfter% các bước)"
    Write-Host ""
}

Write-Progress -Activity "Graph ablation experiment" -Completed

Write-Host "[OK] Ablation artefacts:"
Write-Host "  - results/tables/graph_ablation_summary.csv"
Write-Host "  - results/tables/baseline_results.csv (tất cả graph_construction; bảng paper lọc train_only)"
Write-Host "  - results/tables/graph_ablation.tex"
Write-Host ""
Write-Host "LaTeX: \input{results/tables/graph_ablation.tex}"
