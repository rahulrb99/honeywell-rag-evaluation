Param(
    [switch]$Full,
    [switch]$All
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Refreshing dashboard..." -ForegroundColor Cyan

if ($All) {
    Write-Host "Running full pipeline (ingest -> evals -> dashboard)..." -ForegroundColor Yellow
    python -m src.run_all
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Open: outputs/dashboard/failure_dashboard.html" -ForegroundColor Green
    exit 0
}

if ($Full) {
    Write-Host "Running full eval refresh (retrieval metrics + generation-only + dashboard)..." -ForegroundColor Yellow
    python -m src.run_eval_retrieval_only
    python -m src.run_eval_generation_only
}

python -m src.build_eval_artifacts
python -m src.build_failure_dashboard

Write-Host "Done." -ForegroundColor Green
Write-Host "Open: outputs/dashboard/failure_dashboard.html" -ForegroundColor Green
