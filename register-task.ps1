# Registers a per-user scheduled task that runs start-worker.vbs at logon.
# No admin / UAC needed (per-user task scope).
#
# Usage:
#   .\register-task.ps1
#
# After running, verify in Task Scheduler GUI under "Task Scheduler Library".
# Task name: GenShape3D Worker (3090)
# To unregister: Unregister-ScheduledTask -TaskName "GenShape3D Worker (3090)" -Confirm:$false

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$vbs  = Join-Path $repo "start-worker.vbs"

if (-not (Test-Path $vbs)) {
    Write-Error "start-worker.vbs not found at $vbs"
    exit 1
}

$action  = New-ScheduledTaskAction  -Execute "wscript.exe" -Argument "`"$vbs`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 9999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# -RunLevel Limited keeps it user-scope (no UAC). Per-user task = no admin needed.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Multi-model image-to-3D worker dispatcher"

Register-ScheduledTask -TaskName "GenShape3D Worker (3090)" -InputObject $task -Force | Out-Null

Write-Host "Task registered."
Write-Host "  Name : GenShape3D Worker (3090)"
Write-Host "  Trigger: At logon of $env:USERNAME"
Write-Host "  Action: wscript.exe `"$vbs`""
Write-Host ""
Write-Host "It will fire automatically next time you log in."
Write-Host "To start it right now without rebooting: Start-ScheduledTask -TaskName 'GenShape3D Worker (3090)'"
Write-Host "To stop:  Stop-ScheduledTask  -TaskName 'GenShape3D Worker (3090)' && Get-Process python | Where-Object { `$_.Path -like '*genshape-worker-3090*' } | Stop-Process"
