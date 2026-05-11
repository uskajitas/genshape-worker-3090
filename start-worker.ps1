# Worker launcher — invoked by start-worker.vbs at user login (via Task
# Scheduler). Activates the worker shell venv and runs worker.py.
# Logs go to logs\worker.log; rotated when over 10 MB.

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$log  = Join-Path $repo "logs\worker.log"
$logsDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

# Naive log rotation — rename current to .1 if too big, keep one previous.
if (Test-Path $log) {
    $size = (Get-Item $log).Length
    if ($size -gt 10MB) {
        $rotated = Join-Path $logsDir "worker.log.1"
        if (Test-Path $rotated) { Remove-Item $rotated -Force }
        Move-Item $log $rotated
    }
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
$worker = Join-Path $repo "worker.py"

# Append a session header so logs are easy to scan after a restart.
"" | Out-File -FilePath $log -Append -Encoding utf8
"=== worker started $(Get-Date -Format 's') ===" | Out-File -FilePath $log -Append -Encoding utf8

# Run forever ON CRASH ONLY. Exit code 0 = user clicked "Quit worker" in
# the tray menu; respect that and exit. Anything else = crash, restart
# after 5s.
while ($true) {
    & $python $worker *>&1 | Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "=== worker exited code=$code at $(Get-Date -Format 's') ===" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -eq 0) {
        "=== clean shutdown via tray quit; not restarting ===" | Out-File -FilePath $log -Append -Encoding utf8
        break
    }
    Start-Sleep -Seconds 5
}
