#!/usr/bin/env bash
# Install the official MATH-V evaluation code outside the repository.
#
# Linux equivalent of scripts/setup_mathv.ps1.  Creates an eval-only Python 3.11
# venv with the dependencies used by evaluation/evaluate.py and
# evaluation/utils.py, including latex2sympy2 for the official symbolic
# exact-match.  No Gemma-Loop dependency is added.
#
# Environment variables read by the evaluation code:
#   MATHV_REPO       checkout directory (default: $HOME/ML/MATH-V)
#   MATHV_PYTHON     interpreter in the eval venv
#                    (default: $MATHV_REPO/.venv-mathv/bin/python)
# Optional:
#   MATHV_REPO_URL   clone URL (default: MATH-V GitHub)

set -euo pipefail

REPO_URL="${MATHV_REPO_URL:-https://github.com/mathllm/MATH-V.git}"
REPO_DIR="${MATHV_REPO:-$HOME/ML/MATH-V}"
VENV_DIR="$REPO_DIR/.venv-mathv"

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
    echo "Cloning MATH-V into $REPO_DIR ..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
elif [[ ! -d "$REPO_DIR/.git" ]]; then
    die "$REPO_DIR exists but is not a git clone. Remove it or set MATHV_REPO."
else
    echo "Reusing existing MATH-V checkout at $REPO_DIR"
fi

if [[ -n "${MATHV_PYTHON:-}" ]]; then
    PYTHON="$MATHV_PYTHON"
    [[ -x "$PYTHON" ]] || die "MATHV_PYTHON=$PYTHON is not executable"
else
    PYTHON="$VENV_DIR/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "Creating .venv-mathv with Python 3.11 ..."
        uv venv --python 3.11 "$VENV_DIR"
    else
        echo "Reusing existing .venv-mathv"
    fi
fi

cd "$REPO_DIR"

echo "Installing MATH-V evaluation requirements ..."
uv pip install --python "$PYTHON" latex2sympy2 sympy numpy fire tqdm

echo "Checking the official evaluation imports ..."
"$PYTHON" -c \
    "import sys; sys.path.insert(0, 'evaluation'); import utils, fire; print('MATH-V evaluation imports OK')"

cat <<EOF
MATH-V official scorer is ready. Export these (or pass them in the job environment):
  export MATHV_REPO="$REPO_DIR"
  export MATHV_PYTHON="$PYTHON"
EOF
