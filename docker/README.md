# Gemma-Loop Linux container (CUDA 12.8)

This image matches the committed `uv.lock`: Python 3.12, `torch==2.11.0+cu128`,
`torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`, and Linux-only
`nvidia-*-cu12` / `triton` wheels. It is CPU-safe for the test suite and adds no
GPU requirement to build.

## Host requirements

- Linux x86_64 host with an NVIDIA driver **R570 or newer** (the CUDA 12.8
  runtime userland needs it; minor-version compatibility starts at R525).
- Ampere or newer GPU, compute capability **>= 8.0**. `torch 2.11+cu128` dropped
  Volta, and T4 (7.5) has no bf16, so T4/V100 are not usable targets.
- Docker with BuildKit and the NVIDIA Container Toolkit (`--gpus all`).
- About 15-20 GB of disk for the image plus the dependency layers.

`scripts/check_env.py` validates the driver, compute capability, VRAM, cache and
disk once a container runs. The default container command runs it with
`--require-gpu`, so a container started without `--gpus all` fails instead of
reporting a false green `no GPU`.

## Build

Build from the repository root so the context contains the project (the
`docker/.dockerignore` list is mirrored in the root `.dockerignore`, the file
Docker actually reads from the context root):

```bash
docker build -f docker/Dockerfile -t gemmaloop:r1 .
```

The build pins `uv 0.11.16` via the `ghcr.io/astral-sh/uv` image and runs
`uv sync --frozen`; it never edits `pyproject.toml` or `uv.lock`.

## Run with a GPU

The container runs as UID 1000 (`gemma`) and expects `/cache` (Hugging Face
cache; `HF_HOME=/cache/hf`) and `/artifacts` (`GEMMALOOP_ARTIFACTS`) to be
writable. Either make the host directories writable by UID 1000, or pass your
own UID/GID:

```bash
mkdir -p "$HOME/gemmaloop-cache" "$HOME/ML/gemmaloop-artifacts"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  -v "$HOME/ML/gemmaloop-artifacts:/artifacts" \
  gemmaloop:r1 scripts/check_env.py --require-gpu
```

`check_env` also requires at least 8 GiB of total VRAM by default; use
`--min-vram-gb` to change the floor and `--min-free-gb` for the artifact disk
threshold. For a CPU-only smoke without a GPU, run it without `--require-gpu`
(it then reports `no GPU` and validates everything else).

The entrypoint is `/app/.venv/bin/python`, so the first non-option argument is
the script and later arguments reach it:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  -v "$HOME/ML/gemmaloop-artifacts:/artifacts" \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  gemmaloop:r1 scripts/train.py --help
```

Populate the cache once (or mount a pre-built cache bundle), then use the
offline switches for reproducibility:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  --entrypoint hf gemmaloop:r1 download google/gemma-4-E2B-it

# Optional until an E4B arm runs:
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  --entrypoint hf gemmaloop:r1 download google/gemma-4-E4B-it
```

`check_env` requires the E2B snapshot and reports E4B as optional. A snapshot
only counts as complete when it has `config.json` plus at least one weight file
(or a sharded-weight index). Set `GEMMALOOP_MODEL_REVISION=<sha>` to require an
exact cached revision instead of `refs/main`/any complete snapshot.

### Secrets: pass the HF token at run time only

The image never contains a token. Both `.dockerignore` files exclude `.env*`,
`.netrc`, `.aws/`, `.huggingface/`, `*token*`, `*.pem`, `*.key` and credentials,
so nothing local is baked in by `COPY . .`. Gated datasets such as GPQA need a
token only when a run fetches them:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  -v "$HOME/ML/gemmaloop-artifacts:/artifacts" \
  -e HF_TOKEN="$HF_TOKEN" \
  gemmaloop:r1 scripts/train.py --help

# or keep the token in an untracked env file (excluded by .env*):
docker run --rm --gpus all ... --env-file .env.hf gemmaloop:r1 ...
```

Do **not** add `ENV HF_TOKEN=...` to the Dockerfile, bake an `.env` into the
image, or commit a token. The cache bundle should be populated separately (see
above), so runs can stay `HF_HUB_OFFLINE=1`.

An interactive shell (all arguments after `--` reach `bash`):

```bash
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  -v "$HOME/ML/gemmaloop-artifacts:/artifacts" \
  --entrypoint bash gemmaloop:r1
```

## CPU test suite in the container

No GPU, cache or artifact mount is needed. The default CMD is the GPU gate, so
override it with `-m pytest`:

```bash
docker run --rm gemmaloop:r1 -m pytest -m "not gpu and not slow" -q
```

This is the command that validates the lock file on Linux: the image build
already ran `uv sync --frozen`, and pytest exercises the CPU-only project code.

## Apptainer for SLURM clusters

Convert the Docker image on a host that has it (no GPU required for the build):

```bash
# From the local Docker daemon
apptainer build gemmaloop-r1.sif docker-daemon://gemmaloop:r1

# Rootless / no shared Docker daemon: export a tar, then build from it
docker save gemmaloop:r1 -o gemmaloop-r1.tar
apptainer build gemmaloop-r1.sif docker-archive://gemmaloop-r1.tar
# add --fakeroot if the build account cannot use user namespaces
```

Run on a GPU node (`--nv` exposes the host driver; the node still needs R570+):

```bash
apptainer exec --nv --pwd /app \
  --bind "$HOME/gemmaloop-cache:/cache,$HOME/ML/gemmaloop-artifacts:/artifacts" \
  gemmaloop-r1.sif /app/.venv/bin/python scripts/check_env.py --require-gpu
```

For a batch job, use the same `exec` line inside the job script after
`#SBATCH --gres=gpu:1`; put `/cache` and `/artifacts` on a shared filesystem so
checkpoints and the HF cache survive preemption. The SIF is read-only; write
only through those bind mounts. CPU tests on a login node:

```bash
apptainer exec --pwd /app gemmaloop-r1.sif \
  /app/.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

## Validation checklist once Docker is available

```bash
docker build -f docker/Dockerfile -t gemmaloop:r1 .
docker run --rm gemmaloop:r1 -m pytest -m "not gpu and not slow" -q
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$HOME/gemmaloop-cache:/cache" \
  -v "$HOME/ML/gemmaloop-artifacts:/artifacts" \
  gemmaloop:r1 scripts/check_env.py --require-gpu
```
