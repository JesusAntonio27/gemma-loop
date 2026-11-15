#Requires -Version 5.1
<#
.SYNOPSIS
    Install LiveCodeBench outside the repository for official scoring.

.DESCRIPTION
    Clones https://github.com/LiveCodeBench/LiveCodeBench into
    C:\Users\<user>\ML\LiveCodeBench, creates a Python 3.11 virtual environment
    named .venv-lcb there, installs the package in editable mode and checks
    that lcb_runner.runner.custom_evaluator is available.

    The main Gemma-Loop environment stays untouched: LiveCodeBench pins its own
    dependencies, so it lives in a separate repository and venv.
#>
param(
    [string]$RepoDir = "C:\Users\<user>\ML\LiveCodeBench",
    [string]$RepoUrl = "https://github.com/LiveCodeBench/LiveCodeBench.git"
)

$ErrorActionPreference = "Stop"

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Assert-Command "git"
Assert-Command "uv"

if (-not (Test-Path $RepoDir)) {
    Write-Host "Cloning LiveCodeBench into $RepoDir ..."
    git clone --depth 1 $RepoUrl $RepoDir
}
elseif (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    throw "$RepoDir exists but is not a git clone. Remove it or pass -RepoDir."
}
else {
    Write-Host "Reusing existing LiveCodeBench checkout at $RepoDir"
}

Push-Location $RepoDir
try {
    $Python = Join-Path $RepoDir ".venv-lcb\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        Write-Host "Creating .venv-lcb with Python 3.11 ..."
        uv venv --python 3.11 .venv-lcb
    }
    else {
        Write-Host "Reusing existing .venv-lcb"
    }

    # LiveCodeBench declares vllm as a dependency even for the custom
    # evaluator.  vllm pulls tilelang, which needs a C/C++ toolchain and does
    # not build on a stock Windows box.  Install the package editable without
    # dependencies and then add only what custom_evaluator actually imports
    # (torch, datasets, numpy, tqdm, pebble, attrs, anthropic).
    Write-Host "Installing LiveCodeBench (editable) into .venv-lcb ..."
    uv pip install --python $Python -e . --no-deps

    Write-Host "Installing the custom_evaluator runtime dependencies (without vllm) ..."
    uv pip install --python $Python torch --index-url https://download.pytorch.org/whl/cpu
    # anthropic 0.49.0 is the version in LiveCodeBench's own lock file; newer
    # releases removed HUMAN_PROMPT/AI_PROMPT, which the prompt module imports.
    uv pip install --python $Python "datasets>=3.2.0,<4" numpy tqdm pebble attrs "anthropic==0.49.0"

    Write-Host "Checking custom_evaluator ..."
    & $Python -m lcb_runner.runner.custom_evaluator --help
    if ($LASTEXITCODE -ne 0) {
        throw "custom_evaluator --help failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "LiveCodeBench is ready. Use scripts/setup_lcb.ps1 again to refresh."
