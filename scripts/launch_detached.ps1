<#
.SYNOPSIS
    Launch one GPU python command fully detached from the calling application.

.DESCRIPTION
    PowerShell 5.1 launcher for the training-free campaign.  It:

    * refuses to start when nvidia-smi lists a python*.exe compute process or
      when <artifacts>/gpu.lock is held (exit 75).  Desktop GPU users such as
      explorer.exe, claude.exe, browsers and NVIDIA overlays are logged and
      ignored: WDDM lists every app that touches the GPU, so counting them would
      always refuse;
    * creates <artifacts>/runs/detached/ if needed;
    * redirects stdout and stderr to one timestamped log file;
    * sets PYTHONUNBUFFERED=1 and PYTHONUTF8=1;
    * starts the command hidden through cmd.exe so it survives the caller;
    * writes the started process id to a sibling .pid file and returns at once.

    ``-DryRun`` parses and validates the command, prints the log/pid/command as
    JSON and exits without starting anything.  ``-SkipChecks`` is only for tests
    and dry runs; never use it on the GPU host when another job may be running.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
      -Name corpus -Python "C:\path\to\.venv\Scripts\python.exe" `
      -Arguments "-m scripts.make_p1lite_corpus --target-verified 120"

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
      -Name stageA -Arguments "-m scripts.trainfree_campaign --stage A" -DryRun

.EXAMPLE
    # Detached training run through the retry orchestrator (Task 14b):
    powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
      -Name pilot-oh-e2b-s0 `
      -Arguments "-m scripts.orchestrate --out C:\Users\<user>\ML\gemmaloop-artifacts\runs\pilot-oh-e2b-s0 -- .venv\Scripts\python.exe scripts\train.py --arm O-H --size e2b --data ... --tokens 40000000 --seed 0 --out ... --require-tag prereg-v1"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Python = "",
    [string]$Arguments = "",
    [string]$Artifacts = "",
    [string]$WorkingDirectory = "",
    [string]$NvidiaSmiOutputFile = "",
    [switch]$DryRun,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$BusyExitCode = 75

function Resolve-ArtifactsRoot {
    param([string]$Override)
    if ($Override) {
        $root = $Override
    } elseif ($env:GEMMALOOP_ARTIFACTS) {
        $root = $env:GEMMALOOP_ARTIFACTS
    } else {
        $root = Join-Path $HOME "ML\gemmaloop-artifacts"
    }
    if ($root.ToLowerInvariant().Contains("onedrive")) {
        throw "Refusing to use ${root}: artifacts must not live inside OneDrive."
    }
    return $root
}

function Safe-Name {
    param([string]$Value)
    $safe = [regex]::Replace($Value, "[^A-Za-z0-9._-]", "-").Trim("-")
    if (-not $safe) { return "job" }
    return $safe
}

function Quote-Cmd {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Test-GpuLockHeld {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $stream.Close()
        $stream.Dispose()
        return $false
    } catch {
        return $true
    }
}

function Get-NvidiaApps {
    param([string]$FakeOutputFile)
    if ($FakeOutputFile) {
        $output = Get-Content -LiteralPath $FakeOutputFile -ErrorAction Stop
        return @($output)
    }
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $command) { return @() }
    try {
        $output = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi query failed with exit code $LASTEXITCODE"
        }
        return @($output)
    } catch {
        throw "nvidia-smi check failed: $_"
    }
}

function Split-NvidiaApps {
    param([string[]]$Lines)
    $busy = @()
    $ignored = @()
    foreach ($line in $Lines) {
        if (-not $line -or $line.Trim() -eq "") { continue }
        $parts = $line -split '\s*,\s*', 2
        $pidText = $parts[0].Trim()
        $image = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $name = [System.IO.Path]::GetFileName($image)
        $entry = "{0}, {1}" -f $pidText, $image
        if ($name -match '^(?i)python[^\/]*\.exe$') {
            $busy += $entry
        } else {
            $ignored += $entry
        }
    }
    return [pscustomobject]@{ Busy = $busy; Ignored = $ignored }
}

$root = Resolve-ArtifactsRoot -Override $Artifacts
$runDir = Join-Path $root "runs\detached"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$safeName = Safe-Name -Value $Name
$logPath = Join-Path $runDir ("{0}-{1}.log" -f $stamp, $safeName)
$pidPath = Join-Path $runDir ("{0}-{1}.pid" -f $stamp, $safeName)
$lockPath = if ($env:GEMMALOOP_GPU_LOCK_PATH) {
    $env:GEMMALOOP_GPU_LOCK_PATH
} else {
    Join-Path $root "gpu.lock"
}
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $WorkingDirectory) {
    $WorkingDirectory = $repoRoot
}
if (-not $Python) {
    $Python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

$blockers = @()
$ignoredApps = @()
if (-not $SkipChecks) {
    if (Test-GpuLockHeld -Path $lockPath) {
        $blockers += "GPU lock is held: $lockPath"
    }
    try {
        $apps = Split-NvidiaApps -Lines (Get-NvidiaApps -FakeOutputFile $NvidiaSmiOutputFile)
        $ignoredApps = @($apps.Ignored)
        if ($apps.Busy.Count -gt 0) {
            $blockers += "nvidia-smi reports python compute process(es): " + ($apps.Busy -join "; ")
        }
    } catch {
        $blockers += "nvidia-smi check failed: $_"
    }
}

if ($blockers.Count -gt 0) {
    [Console]::Error.WriteLine("refusing to launch: " + ($blockers -join "; "))
    exit $BusyExitCode
}

if (-not (Test-Path -LiteralPath $runDir)) {
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
}

$cmdLine = "{0} {1} >> {2} 2>&1" -f (Quote-Cmd -Value $Python), $Arguments, (Quote-Cmd -Value $logPath)

if ($DryRun) {
    $report = [ordered]@{
        dry_run    = $true
        name       = $Name
        python     = $Python
        arguments  = $Arguments
        artifacts  = $root
        working_directory = $WorkingDirectory
        log        = $logPath
        pid_file   = $pidPath
        lock       = $lockPath
        ignored_desktop_gpu_processes = $ignoredApps
        command    = $cmdLine
    }
    $report | ConvertTo-Json -Depth 4
    exit 0
}

if ($ignoredApps.Count -gt 0) {
    Add-Content -LiteralPath $logPath -Value ("[launcher] ignored desktop GPU process(es): " + ($ignoredApps -join "; "))
}

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:GEMMALOOP_ARTIFACTS = $root

$process = Start-Process -FilePath $env:ComSpec `
    -ArgumentList @("/d", "/s", "/c", $cmdLine) `
    -WorkingDirectory $WorkingDirectory `
    -WindowStyle Hidden `
    -PassThru

[System.IO.File]::WriteAllText($pidPath, [string]$process.Id)
Write-Output ("started pid={0} log={1} pidfile={2}" -f $process.Id, $logPath, $pidPath)
exit 0
