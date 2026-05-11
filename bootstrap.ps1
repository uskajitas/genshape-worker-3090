# Bootstrap the worker shell venv. Idempotent — safe to re-run.
#
# Usage (from this directory):
#   .\bootstrap.ps1
#
# What it does:
#   1. Sanity-check that Python 3.11 is on PATH (or via py launcher).
#   2. Create .venv if it doesn't already exist.
#   3. Activate it and pip install -r requirements.txt.
#   4. Print a quick status summary.
#
# What it does NOT do:
#   - Set up runner-specific venvs (those have their own bootstrap notes
#     in each runner's requirements.txt because torch+cuda installs vary).
#   - Download model weights (gated on HF licenses; requires HF_TOKEN).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Locating Python 3.11 ==="
$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $ver = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pyCmd = "py -3.11"
        Write-Host "  found: $ver via py launcher"
    }
}
if (-not $pyCmd) {
    $pyExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (Test-Path $pyExe) {
        $pyCmd = "`"$pyExe`""
        Write-Host "  found: $pyExe (direct)"
    }
}
if (-not $pyCmd) {
    Write-Error "Python 3.11 not found. Install via python.org or winget then re-run."
    exit 1
}

Write-Host ""
Write-Host "=== Creating worker shell venv ==="
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$pyCmd -m venv .venv"
    Write-Host "  created .venv"
} else {
    Write-Host "  .venv already exists; reusing"
}

Write-Host ""
Write-Host "=== Installing worker shell deps ==="
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== Status ==="
& ".\.venv\Scripts\python.exe" --version
& ".\.venv\Scripts\python.exe" -c "import requests, boto3, dotenv; print('worker deps OK')"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Copy .env.example to .env and fill in WORKER_AUTH_TOKEN + R2 + HF_TOKEN."
Write-Host "  2. Bootstrap each runner under runners/<model>/ (see SETUP.md)."
Write-Host "  3. Run: .\.venv\Scripts\Activate.ps1; python worker.py"
