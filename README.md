# Gemma-Loop

Research code that converts **Gemma 4 E2B/E4B into a looped-layer model**: a block of
decoder layers with shared weights that runs N times per token, in the spirit of recent
looped-transformer work. The model is then trained with LoRA and evaluated under a
**pre-registered, reproducible protocol** sized for a 12 GB consumer GPU plus a small
cloud budget.

The goal is a verifiable record, not a leaderboard number: the analysis plan, endpoints
and pilot rules were frozen in git (`prereg-v1`) before any trained arm was evaluated, and
every later change is a versioned, disclosed deviation. Loops cost more compute per token,
so quality is reported both per token and per FLOP.

## What is in the repository

| Area | Contents |
|---|---|
| `src/gemmaloop/model/` | Loop surgery on Gemma 4: configurable shared block (`layer_plan`), the looped forward pass, a zero-init norm-matched bridge between iterations, loop-only LoRA targets, FLOP accounting, pinned model snapshots. With N=1 the looped model reproduces the original bit-exactly. |
| `src/gemmaloop/train/` | Stage 1 "heal" training: KL to a same-tokenizer teacher plus a frozen-parent KL, loop-count sampling, budgets, guards, heal monitoring, checkpoints and resume for an unstable desktop. Stage 2 (Dr. GRPO with programmatic verifiers) is specified in the pre-registration. |
| `src/gemmaloop/eval/`, `judge/`, `pilot/`, `report/` | Benchmark runner with calibrated NLL and statistics, an LLM-judge client with rubric and order-swap agreement, pilot arm selection and analysis, figures. |
| `src/gemmaloop/serve/` | Local generation server with tool-call parsing, used to run the model inside an agent harness. |
| `src/gemmaloop/stage0/`, `teacher/`, `data/` | Data pipeline (sources, decontamination, mixes, teacher top-k), teacher qualification and runtime freeze, cloud runs on Modal with spend tracking. |
| `src/gemmaloop/prereg.py`, `prereg/` | The machine-checked pre-registration: frozen fields, deviation log, pilot shortfall amendment, hash-bound records. |
| `docker/` | Linux CUDA 12.8 container for cloud runs (Apptainer/SLURM notes included). |
| `tests/` | About 2,000 CPU-safe tests: architecture invariants, cache equivalence, budgets, pre-registration integrity. |
| `docs/` | Design spec and implementation plan (Spanish), literature notes, task reports, validation receipts. |

## Status

Pre-registration `prereg-v1` is tagged and its data addendum is frozen. Pilot and
exploratory training runs on E2B are in progress; **no confirmatory result is claimed
in this repository yet**.

---


Research code for turning Gemma 4 E2B/E4B into a looped-layer model, training it
with LoRA and evaluating it. Gemma-Loop is based on Gemma 4 (Apache 2.0).

## Environment

- Local development on Windows 11; cloud runs use the Linux CUDA 12.8 container
  in `docker/` (`docker/README.md`) on an Ampere-or-newer GPU with driver R570+.
- Python 3.12 managed by `uv`.
- CUDA 12.8 wheels of PyTorch (`torch==2.11.0`, `torchvision==0.26.0`,
  `torchaudio==2.11.0`) installed from the `pytorch-cu128` index declared in
  `pyproject.toml`; `uv.lock` is committed and resolves on Linux x86_64.
- Heavy artifacts (models, checkpoints, data, raw results) live outside the repo,
  in `~/ML/gemmaloop-artifacts` by default, or wherever `GEMMALOOP_ARTIFACTS`
  points. Paths containing `OneDrive` are rejected.

## Setup

```powershell
uv sync --python 3.12
uv run python scripts/check_env.py
uv run pytest -v
```

`scripts/check_env.py` validates the platform, Python 3.12, the torch/CUDA
build, the visible GPU (compute capability >= 8.0 for bf16), the NVIDIA driver
(R570+ for cu128), VRAM (>= 8 GiB by default), free artifact disk, the HF offline
switches and the pinned Gemma 4 snapshot in the HF cache. It is CPU-safe
(reports `no GPU`) and exits non-zero on any failure. Add `--require-gpu` on a
GPU host or in a container so a missing `--gpus all` / an old driver fails
instead of passing. See `docker/README.md` for the container build, GPU flags,
mounts, secret handling and Apptainer/SLURM conversion.
