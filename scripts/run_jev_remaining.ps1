param(
    [Parameter(Mandatory = $true)]
    [int]$NoulProcessId
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Wait-Process -Id $NoulProcessId
$statusPath = 'results\eval-jev-latest\eval_noul.json'
if (-not (Test-Path -LiteralPath $statusPath)) {
    throw 'Noul did not produce eval_noul.json; remaining tasks were not started.'
}
$status = Get-Content -Raw -Encoding UTF8 -LiteralPath $statusPath | ConvertFrom-Json
if ($status.status -ne 'complete') {
    throw "Noul status is $($status.status); remaining tasks were not started."
}

Get-Content -LiteralPath '.env' | ForEach-Object {
    if ($_ -match '^\s*([^#][A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        $value = $matches[2].Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($matches[1], $value, 'Process')
    }
}

& uv run python scripts/run_jev_benchmark.py `
    --phase production `
    --tasks choice score summary `
    --parallelism 4
if ($LASTEXITCODE -ne 0) {
    throw "Jev remaining benchmark failed with exit code $LASTEXITCODE."
}

& uv run python scripts/build_model_dashboard.py
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard refresh failed with exit code $LASTEXITCODE."
}
