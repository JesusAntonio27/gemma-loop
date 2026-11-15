<#
.SYNOPSIS
Run the Phase 1 E2B plan (data -> training -> evaluation -> criteria).

.DESCRIPTION
Reentrant Windows PowerShell 5.1 driver.  The ordered step list, every command
and the "already exists" decisions come from scripts/phase1_plan.py.  Steps run
strictly one after another (one GPU job at a time); the script stops on the
first failed command and logs to <artifacts>\runs\phase1_e2b.log.

Progress charts and the three-column table are regenerated after every
evaluation step, even when those files already existed before this invocation.

.PARAMETER DryRun
Print every step with "would run" / "skip (exists)" and its exact command, and
execute nothing.

.PARAMETER Prereg
Pre-registration YAML (default: prereg/prereg.yaml).

.PARAMETER Artifacts
Artifacts root (default: $env:GEMMALOOP_ARTIFACTS or the gemmaloop default).

.EXAMPLE
powershell -NoProfile -File scripts/phase1_e2b.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$Prereg = "prereg/prereg.yaml",
    [string]$Artifacts = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    # ------------------------------------------------------------------
    # Build the plan (CPU only; no model is loaded).
    # ------------------------------------------------------------------
    $planArgs = @(
        "run", "python", "scripts/phase1_plan.py",
        "--prereg", $Prereg,
        "--size", "e2b"
    )
    if ($Artifacts -ne "") {
        $planArgs += @("--artifacts", $Artifacts)
    }

    $planLines = & uv @planArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "phase1_e2b: FAILED to build the step plan (exit $LASTEXITCODE)"
        exit 1
    }
    try {
        $Plan = (($planLines | Out-String) | ConvertFrom-Json)
    }
    catch {
        Write-Host "phase1_e2b: FAILED to parse the step plan: $($_.Exception.Message)"
        exit 1
    }

    $Steps = @($Plan.steps)
    $LogPath = Join-Path $Plan.artifacts "runs\phase1_e2b.log"
    $LogDir = Split-Path -Parent $LogPath
    $ReportDir = Join-Path $Plan.artifacts "report"

    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
        New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
    }

    function Write-Log([string]$Message) {
        if (-not $DryRun) {
            Add-Content -Path $LogPath -Value $Message -Encoding UTF8
        }
        Write-Host $Message
    }

    Write-Log ("phase1_e2b: {0} steps; artifacts={1}; prereg={2}" -f `
        $Steps.Count, $Plan.artifacts, $Plan.prereg)
    Write-Log ("phase1_e2b: required data tokens = {0}" -f $Plan.required_data_tokens)

    $Ran = @{}
    $AnyEvalRan = $false
    $AnyTrainRan = $false
    $WouldRun = 0
    $Skipped = 0

    foreach ($Step in $Steps) {
        $Forced = $false
        if ($Step.force_if_any_eval -and $AnyEvalRan) {
            $Forced = $true
        }
        if ($Step.force_if_any_train -and $AnyTrainRan) {
            $Forced = $true
        }
        if ($Step.trigger -and $Ran.ContainsKey([string]$Step.trigger)) {
            $Forced = $true
        }
        $Skip = ([bool]$Step.skip) -and (-not $Forced)

        if ($Skip) {
            $Skipped++
            Write-Log ("skip (exists): [{0}] {1}" -f $Step.id, $Step.title)
            if ($Step.skip_detail) {
                Write-Log ("  reason: {0}" -f $Step.skip_detail)
            }
            Write-Log ("  command: {0}" -f $Step.display)
            continue
        }

        $WouldRun++
        if ($DryRun) {
            Write-Log ("would run: [{0}] {1}" -f $Step.id, $Step.title)
            Write-Log ("  command: {0}" -f $Step.display)
            if ($Step.output_file) {
                Write-Log ("  capture stdout to: {0}" -f $Step.output_file)
            }
            # Simulate triggers so a chart/table step after a would-run
            # evaluation is reported as "would run", not "skip (exists)".
            $Ran[[string]$Step.id] = $true
            if ($Step.kind -eq "eval") {
                $AnyEvalRan = $true
            }
            if ($Step.kind -eq "train") {
                $AnyTrainRan = $true
            }
            continue
        }

        Write-Log ""
        Write-Log ("RUN: [{0}] {1}" -f $Step.id, $Step.title)
        Write-Log ("  command: {0}" -f $Step.display)

        $Argv = @($Step.argv)
        $Exe = [string]$Argv[0]
        if ($Argv.Count -gt 1) {
            $Rest = @($Argv[1..($Argv.Count - 1)])
        }
        else {
            $Rest = @()
        }

        $PreviousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($Step.output_file) {
                $OutputFile = [string]$Step.output_file
                $OutputDir = Split-Path -Parent $OutputFile
                if ($OutputDir -ne "") {
                    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
                }
                if (Test-Path $OutputFile) {
                    Remove-Item -Force $OutputFile
                }
                & $Exe @Rest 2>&1 |
                    Tee-Object -FilePath $LogPath -Append |
                    Tee-Object -FilePath $OutputFile
            }
            else {
                & $Exe @Rest 2>&1 |
                    Tee-Object -FilePath $LogPath -Append
            }
            $ExitCode = $LASTEXITCODE
        }
        catch {
            $ExitCode = 1
            Write-Host ("  error: {0}" -f $_.Exception.Message)
        }
        finally {
            $ErrorActionPreference = $PreviousErrorAction
        }

        if ($null -eq $ExitCode) {
            $ExitCode = 1
        }
        if ($ExitCode -ne 0) {
            Write-Log ("phase1_e2b: FAILED at [{0}] {1} (exit {2})" -f `
                $Step.id, $Step.title, $ExitCode)
            Write-Log ("  command: {0}" -f $Step.display)
            Write-Log ("  log: {0}" -f $LogPath)
            if ($Step.output_file) {
                # Never leave a partial marker behind.  A criteria report that
                # already contains S1-S5 is a legitimate "criterion failed"
                # result (check_criteria exits 1 then); an aborted run is not.
                $OutputFile = [string]$Step.output_file
                if (Test-Path $OutputFile) {
                    $KeepOutput = $false
                    if ($Step.kind -eq "criteria") {
                        $KeepOutput = [bool](Select-String -Path $OutputFile -Pattern '^S[1-5]' -Quiet)
                    }
                    if (-not $KeepOutput) {
                        Remove-Item -Force $OutputFile
                    }
                }
            }
            exit 1
        }

        $Ran[[string]$Step.id] = $true
        if ($Step.kind -eq "eval") {
            $AnyEvalRan = $true
        }
        if ($Step.kind -eq "train") {
            $AnyTrainRan = $true
        }
        Write-Log ("DONE: [{0}]" -f $Step.id)
    }

    if ($DryRun) {
        Write-Log ""
        Write-Log ("dry run: {0} steps, {1} would run, {2} skip (exists)" -f `
            $Steps.Count, $WouldRun, $Skipped)
    }
    else {
        Write-Log ""
        Write-Log ("phase1_e2b: all {0} steps completed (log: {1})" -f `
            $Steps.Count, $LogPath)
    }
}
finally {
    Pop-Location
}
