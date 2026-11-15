<#
.SYNOPSIS
    PRINT (never execute) the commands that resume a named run at log on.

.DESCRIPTION
    Task 14b deliberately does not register anything.  This script only builds
    the text of a Windows Scheduled Task and the equivalent systemd user unit /
    cron @reboot line.  The author reviews and runs them himself.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/register_autoresume.ps1 `
      -Name pilot-oh-e2b-s0 -RunDir C:\Users\<user>\ML\gemmaloop-artifacts\runs\pilot-oh-e2b-s0 `
      -Python C:\Users\<user>\ML\GemmaLoop-wt\t14b\.venv\Scripts\python.exe `
      -Arguments "scripts/train.py --arm O-H --size e2b --data ... --tokens 40000000 --seed 0 --out ..."
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Python = "",
    [string]$Arguments = "",
    [string]$Repo = "",
    [string]$Artifacts = ""
)

$ErrorActionPreference = "Stop"

if (-not $Repo) {
    $Repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
}
if (-not $Python) {
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
}
if (-not $Artifacts) {
    if ($env:GEMMALOOP_ARTIFACTS) {
        $Artifacts = $env:GEMMALOOP_ARTIFACTS
    } else {
        $Artifacts = Join-Path $HOME "ML\gemmaloop-artifacts"
    }
}
if (-not $Arguments) {
    $Arguments = "scripts\train.py --out `"$RunDir`""
}
# This is the auto-resume task: it must ask the trainer to continue an
# existing checkpoint, never to start a fresh run over it.
if ($Arguments -notmatch '--resume') {
    $Arguments = "$Arguments --resume auto"
}

$safe = [regex]::Replace($Name, "[^A-Za-z0-9._-]", "-")
$taskName = "GemmaLoop\$safe"
$orchestrator = Join-Path $Repo "scripts\orchestrate.py"
$logDir = Join-Path $Artifacts "runs\detached"
$orchLog = Join-Path $logDir "$safe-orchestrator.log"

# The whole command is executed by the Python orchestrator.  It handles exit 75
# with --resume auto; the explicit 'cd /d' makes a logon task independent of the
# system working directory (the default is not the repo).
$inner = "set GEMMALOOP_ARTIFACTS=$Artifacts && cd /d `"$Repo`" && `"$Python`" `"$orchestrator`" --out `"$RunDir`" --name `"$safe`" -- `"$Python`" $Arguments >> `"$orchLog`" 2>&1"
$taskAction = "/TR `"cmd.exe /d /s /c `"`"$inner`"`"`""

$systemdName = "gemmaloop-$safe"
$systemdOrchestrator = "$Repo/scripts/orchestrate.py"
$systemdUnit = @"
[Unit]
Description=GemmaLoop auto-resume: $safe

[Service]
Type=oneshot
WorkingDirectory=$Repo
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONUTF8=1
Environment=GEMMALOOP_ARTIFACTS=$Artifacts
ExecStart="$Python" "$systemdOrchestrator" --out "$RunDir" --name "$safe" -- "$Python" $Arguments
RemainAfterExit=no

[Install]
WantedBy=default.target
"@

$posixRepo = $Repo -replace '\\', '/'
$posixRun = $RunDir -replace '\\', '/'
$posixArtifacts = $Artifacts -replace '\\', '/'
$posixLog = ($logDir -replace '\\', '/') + "/$safe-orchestrator.log"

Write-Output "=== Nothing below was executed. Review, then run the lines you want. ==="
Write-Output ""
Write-Output "--- Windows Scheduled Task (run this in an elevated prompt) ---"
Write-Output "schtasks /Create /TN `"$taskName`" /SC ONLOGON /RL LIMITED $taskAction /F"
Write-Output ""
Write-Output "--- Windows alternative with an explicit working directory and no time limit ---"
Write-Output "`$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/d /s /c `"$inner`"' -WorkingDirectory '$Repo'"
Write-Output "`$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable"
Write-Output "Register-ScheduledTask -TaskName '$taskName' -Action `$action -Trigger (New-ScheduledTaskTrigger -AtLogOn) -Settings `$settings -Force"
Write-Output ""
Write-Output "--- Windows: inspect / remove later ---"
Write-Output "schtasks /Query /TN `"$taskName`" /V /FO LIST"
Write-Output "schtasks /Delete /TN `"$taskName`" /F"
Write-Output ""
Write-Output "--- Linux systemd user unit (copy to ~/.config/systemd/user/$systemdName.service) ---"
Write-Output $systemdUnit
Write-Output "systemctl --user daemon-reload"
Write-Output "systemctl --user enable --now $systemdName.service"
Write-Output ""
Write-Output "--- Linux cron @reboot alternative ---"
Write-Output "(crontab -l 2>/dev/null; echo '@reboot cd $posixRepo && PYTHONUNBUFFERED=1 PYTHONUTF8=1 GEMMALOOP_ARTIFACTS=$posixArtifacts $Python $posixRepo/scripts/orchestrate.py --out $posixRun --name $safe -- $Python $Arguments >> $posixLog 2>&1') | crontab -"
Write-Output ""
Write-Output "The task is safe to leave registered: once a run writes DONE.json,"
Write-Output "user's GPU context and mapped drives exist.  schtasks creates a default"
Write-Output "72 h execution limit; use the Register-ScheduledTask form above (or the"
Write-Output "Task Scheduler GUI) to set the limit to zero."
Write-Output "A completed run is a no-op (exit 0, no launch); --rerun-complete is explicit."
