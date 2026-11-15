#Requires -Version 5.1
<#
.SYNOPSIS
    Install the official MATH-V evaluation code outside the repository.

.DESCRIPTION
    Clones https://github.com/mathllm/MATH-V into C:\Users\<user>\ML\MATH-V and
    creates an eval-only Python 3.11 virtual environment named .venv-mathv with
    the dependencies used by evaluation/evaluate.py and evaluation/utils.py,
    including latex2sympy2 for the official symbolic exact-match.

    The main Gemma-Loop environment stays untouched; no project dependency is
    added.
#>
param(
    [string]$RepoDir = "C:\Users\<user>\ML\MATH-V",
    [string]$RepoUrl = "https://github.com/mathllm/MATH-V.git"
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
    Write-Host "Cloning MATH-V into $RepoDir ..."
    git clone --depth 1 $RepoUrl $RepoDir
}
elseif (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    throw "$RepoDir exists but is not a git clone. Remove it or pass -RepoDir."
}
else {
    Write-Host "Reusing existing MATH-V checkout at $RepoDir"
}

Push-Location $RepoDir
try {
    $Python = Join-Path $RepoDir ".venv-mathv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        Write-Host "Creating .venv-mathv with Python 3.11 ..."
        uv venv --python 3.11 .venv-mathv
    }
    else {
        Write-Host "Reusing existing .venv-mathv"
    }

    Write-Host "Installing MATH-V evaluation requirements ..."
    uv pip install --python $Python latex2sympy2 sympy numpy fire tqdm

    Write-Host "Checking the official evaluation imports ..."
    & $Python -c "import sys; sys.path.insert(0, 'evaluation'); import utils, fire; print('MATH-V evaluation imports OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "MATH-V evaluation imports failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "MATH-V official scorer is ready."
