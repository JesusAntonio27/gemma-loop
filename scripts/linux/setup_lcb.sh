#!/usr/bin/env bash
# Install LiveCodeBench outside the repository for official scoring.
#
# Linux equivalent of scripts/setup_lcb.ps1.  The Gemma-Loop environment stays
# untouched; LiveCodeBench pins its own dependencies, so it lives in a separate
# checkout and venv.
#
# Environment variables read by the evaluation code:
#   LCB_REPO         checkout directory (default: $HOME/ML/LiveCodeBench)
#   LCB_PYTHON       interpreter in the eval venv
#                    (default: $LCB_REPO/.venv-lcb/bin/python)
# Optional:
#   LCB_REPO_URL     clone URL (default: LiveCodeBench GitHub)

set -euo pipefail

REPO_URL="${LCB_REPO_URL:-https://github.com/LiveCodeBench/LiveCodeBench.git}"
REPO_DIR="${LCB_REPO:-$HOME/ML/LiveCodeBench}"
VENV_DIR="$REPO_DIR/.venv-lcb"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 \
        || die "required command '$1' was not found on PATH"
}

require_command git
require_command uv

if [[ ! -e "$REPO_DIR" ]]; then
    echo "Cloning LiveCodeBench into $REPO_DIR ..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
elif [[ ! -d "$REPO_DIR/.git" ]]; then
    die "$REPO_DIR exists but is not a git clone. Remove it or set LCB_REPO."
else
    echo "Reusing existing LiveCodeBench checkout at $REPO_DIR"
fi

if [[ -n "${LCB_PYTHON:-}" ]]; then
    PYTHON="$LCB_PYTHON"
    [[ -x "$PYTHON" ]] || die "LCB_PYTHON=$PYTHON is not executable"
else
    PYTHON="$VENV_DIR/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "Creating .venv-lcb with Python 3.11 ..."
        uv venv --python 3.11 "$VENV_DIR"
    else
        echo "Reusing existing .venv-lcb"
    fi
fi

cd "$REPO_DIR"

# LiveCodeBench declares vllm as a dependency even for the custom evaluator.
# vllm pulls tilelang, which needs a C/C++ toolchain and a large CUDA build;
# install the package editable without dependencies and add only what
# custom_evaluator actually imports (torch, datasets, numpy, tqdm, pebble,
# attrs, anthropic).
echo "Installing LiveCodeBench (editable) into the eval venv ..."
uv pip install --python "$PYTHON" -e . --no-deps

echo "Installing the custom_evaluator runtime dependencies (without vllm) ..."
uv pip install --python "$PYTHON" torch \
    --index-url https://download.pytorch.org/whl/cpu
# anthropic 0.49.0 is the version in LiveCodeBench's own lock file; newer
# releases removed HUMAN_PROMPT/AI_PROMPT, which the prompt module imports.
uv pip install --python "$PYTHON" \
    "datasets>=3.2.0,<4" numpy tqdm pebble attrs "anthropic==0.49.0"

echo "Checking custom_evaluator ..."
"$PYTHON" -m lcb_runner.runner.custom_evaluator --help

cat <<EOF
LiveCodeBench is ready. Export these (or pass them in the job environment):
  export LCB_REPO="$REPO_DIR"
  export LCB_PYTHON="$PYTHON"
EOF
