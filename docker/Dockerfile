# syntax=docker/dockerfile:1.7
#
# CUDA 12.8 runtime image matching the torch==2.11.0+cu128 wheels in uv.lock.
# Host requirements: NVIDIA driver >= R570 (the CUDA 12.8 runtime) and an
# Ampere-or-newer GPU (compute capability >= 8.0) for bf16; T4 (7.5) and Volta
# are not supported by the cu128 build.
#
# Build from the repository root so the build context contains the project:
#   docker build -f docker/Dockerfile -t gemmaloop:r1 .
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/cache/hf \
    GEMMALOOP_ARTIFACTS=/artifacts \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/bin/python3.12 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# Ubuntu 24.04 ships Python 3.12, the version pinned by pyproject.toml.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# uv is pinned; see https://github.com/astral-sh/uv/releases.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

WORKDIR /app

# Resolve the locked environment first so dependency changes alone invalidate
# this layer, not every source edit.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# The full repository, then install the editable project into the same venv.
COPY . .
RUN uv sync --frozen

# Non-root runtime user.  Bind-mounted /cache and /artifacts must be writable by
# this UID (or pass --user "$(id -u):$(id -g)" when running the container).
RUN useradd --create-home --uid 1000 --shell /bin/bash gemma \
    && mkdir -p /cache/hf /artifacts \
    && chown -R gemma:gemma /app /cache /artifacts

USER gemma

# Entrypoint is the project interpreter: `docker run <image> scripts/train.py`
# runs a script, `docker run <image> -m pytest ...` runs a module.
ENTRYPOINT ["/app/.venv/bin/python"]
# Default command is the GPU gate: a container run without --gpus all fails
# instead of silently reporting "no GPU". CPU checks override this with -m pytest.
CMD ["scripts/check_env.py", "--require-gpu"]
