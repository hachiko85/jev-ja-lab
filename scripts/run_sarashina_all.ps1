$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Get-Content -LiteralPath '.env' | ForEach-Object {
    if ($_ -match '^\s*([^#][A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        $value = $matches[2].Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($matches[1], $value, 'Process')
    }
}

& .venv\Scripts\python.exe scripts/run_sarashina_benchmark.py
if ($LASTEXITCODE -ne 0) {
    throw "Sarashina benchmark failed with exit code $LASTEXITCODE."
}

& .venv\Scripts\python.exe scripts/build_model_dashboard.py
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard refresh failed with exit code $LASTEXITCODE."
}
