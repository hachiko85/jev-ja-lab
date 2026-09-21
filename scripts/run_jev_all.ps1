$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Get-Content -LiteralPath '.env' | ForEach-Object {
    if ($_ -match '^\s*([^#][A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        $value = $matches[2].Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($matches[1], $value, 'Process')
    }
}

& uv run python scripts/run_jev_benchmark.py `
    --phase production `
    --tasks noul choice score summary `
    --parallelism 4
if ($LASTEXITCODE -ne 0) {
    throw "Jev benchmark failed with exit code $LASTEXITCODE."
}

$source = 'results\eval-jev-latest\jev-latest'
$destination = 'results\eval-20260920\jev-latest'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -Path "$source\*" -Destination $destination -Recurse -Force
& uv run python scripts/build_model_dashboard.py
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard refresh failed with exit code $LASTEXITCODE."
}
