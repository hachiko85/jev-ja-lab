param(
    [Parameter(Mandatory = $true)]
    [int]$BenchmarkProcessId
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
Wait-Process -Id $BenchmarkProcessId
& uv run python scripts/build_model_dashboard.py
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard refresh failed with exit code $LASTEXITCODE."
}
