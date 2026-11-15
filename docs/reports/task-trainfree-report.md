# T-free report — training-free loop evaluation campaign

Status: complete on CPU; no GPU job was run from this branch. No `prereg/` files,
`pyproject.toml` or `uv.lock` were modified. The campaign is **training-free,
exploratory, and measured before preregistration**; it is explicitly not an
input to block selection or to any prereg decision.

**Stage B is BLOCKED until the author decides E3.** The runner refuses
`--stage B`, `--stage B2` and `--stage all` unless `--allow-stage-b` is passed;
the controller command list below stops after stage A.

## Commits

- `0eb9ebf` feat(scripts): add GPU job lock, heartbeat and detached launcher
- `ed61795` feat(scripts): extend pilot probe to multi-N calibrated metrics
- `230bac0` feat(scripts): add P1-lite verified corpus builder
- `16094c7` feat(scripts): add resumable training-free campaign runner
- `f9d36b3` feat(scripts): add training-free campaign report
- `4e645aa` merge: task/arch (fix rounds 1-2) into task/trainfree
- fix round 1: `fix(trainfree): chunked KL/NLL, full-trace corpus, allocator cap,
  heartbeat, lock, detached commands`

## What was implemented

### 1. GPU safety, allocator cap and detached launch

- `src/gemmaloop/gpulock.py`: process-lifetime one-byte lock on
  `<artifacts>/gpu.lock` via `msvcrt.locking` on Windows and `fcntl.flock`
  elsewhere. `require_gpu_lock()` exits with code **75** when busy. Called first
  in `scripts/run_eval.py`, `scripts/pilot_probe.py`,
  `scripts/make_p1lite_corpus.py` and `scripts/trainfree_campaign.py`. There is
  no inherited-lock bypass: the campaign checks the lock, releases it, and lets
  every child own its own lock, so a parent crash cannot leave an unlocked GPU
  child.
- Allocator cap: every GPU entry point calls the existing
  `gemmaloop.train.perf.cap_cuda_allocator(0.90, 0.5)` immediately after taking
  the lock and before loading the model (`make_p1lite_corpus`, `pilot_probe`,
  `run_eval`, `trainfree_campaign`). The cap is recorded in the corpus manifest,
  pilot JSON, `RunResult.metadata["vram_cap"]`, the campaign plan and the
  heartbeat.
- `src/gemmaloop/heartbeat.py`: atomic JSON (`ts`, `step`/`items_done`,
  `items_total`, `tok_per_s`, `torch_reserved_gib`, `ram_available_gib`, plus
  phase/job labels). `run_eval` updates it per generation batch; `pilot_probe`
  updates it per geometry row, per KL row and per variant, so stage A is never
  silent for hours. Every writer also runs a background 30 s writer.
  `run_bench` keeps its `on_batch` callback.
- `scripts/launch_detached.ps1` (PowerShell 5.1): lists `nvidia-smi
  --query-compute-apps`, refuses with exit 75 only when a listed image name
  matches `python*.exe` (case-insensitive) or the GPU lock is held, and logs the
  ignored desktop GPU apps (explorer, Claude, Edge WebView, NVIDIA Overlay, …).
  It creates `<artifacts>/runs/detached/`, sets `PYTHONUNBUFFERED=1`,
  `PYTHONUTF8=1` and exports `GEMMALOOP_ARTIFACTS` to the child, starts the
  command hidden through `cmd.exe` with stdout+stderr appended to one
  timestamped log, writes a pid file and returns immediately. It honours
  `GEMMALOOP_GPU_LOCK_PATH`, defaults `-Python` to this worktree's
  `.venv\Scripts\python.exe`, and defaults `-WorkingDirectory` to the repository
  root. `-DryRun` performs every check and prints the exact command as JSON
  without starting anything.

### 2. P1-lite corpus — full completed assistant turn

`scripts/make_p1lite_corpus.py` streams DAPO-Math-17k
(`BytedTsinghua-SIA/DAPO-Math-17k`, revision
`65877096c24ffa7abc4e4fa5edb95cf3413a5674`, Apache-2.0), keeps problems with an
exact integer answer, rejects matches against the repository `EvalIndex` built
from **all** registered eval sets, and samples the original E2B once per
candidate (thinking on, T=1.0, top_p=0.95, top_k=64,
`max_new_tokens=8192`).

`GenOut.text` excludes the thinking channel, so the builder reconstructs the
**exact completed assistant turn** from `output.reasoning` + `output.text`
through the chat template (`render_completed_turn`), verifies the last
non-thinking `\boxed{}` answer, and stores prompt + full response (thinking
channel + answer + turn close) with `input_ids`/`labels_start` covering the
assistant tokens. Stage A therefore scores the same sequence the model actually
generated, reasoning included. The manifest records source revision, ids,
corpus/raw SHA-256, token-length distributions, tokenizer/template hashes,
selection/decontamination status and the allocator cap. Raw samples include the
reasoning trace and carry a model/sampling/source fingerprint so a stale resume
is refused.

### 3. Campaign runner

`scripts/trainfree_campaign.py` is a resumable, one-GPU-job-at-a-time plan
runner. The JSON plan lists every config with `pending/running/done/failed`,
is written atomically after every job status change, and records result paths.

- **Stage A**: blocks `25-29`, `16-18`, `15-29` × bridges `none` (R, pure
  repeat) and `damped` (R-damped) × `N ∈ {1,2,3,4}`. It runs
  `scripts/pilot_probe.py` once per N (one model load for six configs) and
  reports per-layer cos/RMS, `KL(N=1||N)`, calibrated NLL (T fitted per
  config/N on half the traces, scored on the other half), raw NLL and top-1.
  The KL path is chunked over tokens (256) and vocabulary, never full
  `[T, V]`; the calibrated-NLL helper uses a 512-token chunk. The stage-A output
  carries the corpus SHA-256, so changing the corpus invalidates resume.
- **Stage B / B2** (blocked until E3): same design as before — O plus R /
  R-damped at N=2 on LiveCodeBench post-cut (182) and GPQA Diamond (198), and
  N=3/4 on the block with the lowest calibrated NLL at N=2 (min over bridges;
  ties by lower start then lower end). They run only with `--allow-stage-b`.
- Eval partials are left to `gemmaloop.eval.runner`, so an interrupted stage B
  job resumes completed `(item, sample)` pairs.

### 4. Report

`scripts/trainfree_report.py` writes, under
`<artifacts>/report/trainfree/`: `trainfree_table.md` / `.csv` (accuracy, Wilson
95% CI over items, paired item bootstrap vs O with 10,000 resamples and seed 0,
mean generated tokens, truncation rate), `accuracy_<bench>.png` (accuracy vs N
per block, O as a horizontal reference with its Wilson band),
`calibrated_nll.png` (calibrated NLL vs N per block) and
`trainfree_summary.md`, labelled **“Training-free, exploratory, measured before
preregistration.”** Chart code is in this new script; `progress_charts.py` was
not edited.

## Exact commands for the controller (in order)

Every GPU step is launched through `scripts/launch_detached.ps1`, one at a time.
The launcher defaults `-Python` to this worktree's `.venv\Scripts\python.exe`
and `-WorkingDirectory` to the repository root, so no placeholder paths are
needed. Artifacts default to `$GEMMALOOP_ARTIFACTS` (or
`~/ML/gemmaloop-artifacts`). Poll the printed log/heartbeat before launching the
next step.

```powershell
# 0) Verify the GPU host before the first long job.
uv run python scripts/check_env.py --require-gpu

# 1) Build the verified P1-lite corpus (GPU; eval index on CPU).
powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
  -Name trainfree-corpus `
  -Arguments "-m scripts.make_p1lite_corpus --target-verified 120 --candidate-limit 360 --batch 8"

# Wait for the corpus job to finish; inspect
#   <artifacts>\runs\detached\corpus.json
# and <artifacts>\data\p1lite\manifest.json.

# 2) Stage A only: P1-lite sweep (blocks x bridges x N).
powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
  -Name trainfree-stageA `
  -Arguments "-m scripts.trainfree_campaign --stage A"

# Wait for stage A; inspect
#   <artifacts>\runs\detached\campaign.json and child-A_n*.json
#   <artifacts>\results\trainfree\stageA\n*.json

# 3) Report tables, charts and the one-page summary (CPU).
uv run python scripts/trainfree_report.py
```

**Stage B / B2 are intentionally absent.** Once the author decides E3, the same
launcher may run them with `--allow-stage-b` (one at a time):

```powershell
# ONLY after the E3 decision:
powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
  -Name trainfree-stageB `
  -Arguments "-m scripts.trainfree_campaign --stage B --allow-stage-b"
```

A plan-only dry run (no GPU, no lock) is:

```powershell
uv run python scripts/trainfree_campaign.py --plan-only --stage B2
```

The launcher prints `pid=... log=<artifacts>\runs\detached\<timestamp>-<name>.log
pidfile=...` and returns. For the corpus job poll
`<artifacts>\runs\detached\corpus.json`; for stage A poll
`<artifacts>\runs\detached\campaign.json` and its per-job child heartbeat
`<artifacts>\runs\detached\child-A_n<N>.json`.

## Estimated GPU hours

Planning estimates, not measurements. Assumptions: E2B bf16,
`transformers.generate`, aggregate throughput ≈200–300 tok/s at N=1 and roughly
half at N=2 (project reference 04; confirm with `bench_throughput.py`). Average
completion lengths: DAPO corpus 2.5k, LCB 1k, GPQA 3–5k.

| Step | Work | Estimate |
| --- | --- | --- |
| P1-lite corpus | 360 candidates × 2.5k tokens ≈ 0.9M generated tokens; plus CPU eval-index build | **1.5–2.5 h** |
| Stage A | 24 (block, bridge, N) configs; forward-only, 120 traces + temperature fits | **3–6 h** |
| Report | CPU | **< 5 min** |
| **Enabled total (corpus + A + report)** | | **≈ 5–9 h** |
| Stage B (blocked) | 7 model configs × (182 LCB + 198 GPQA) ≈ 7–8M generated tokens | **10–18 h** |
| Stage B2 (blocked) | 4 model configs × (182 LCB + 198 GPQA) ≈ 4–5M tokens at N=3/4 | **8–14 h** |

The corpus estimate assumes the original model answers at least 120/360
candidates correctly; if not, raise `--candidate-limit` and add ~0.4 h per 100
extra candidates.

## Tests

Command (final tree):

```powershell
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
```

Result:

```text
766 passed, 22 deselected, 15 warnings in 143.02s (0:02:23)
```

The suite includes new peak-memory scaling tests for the chunked KL path and the
calibrated-NLL lean head, a real-shaped `GenOut(reasoning, text)` corpus
round-trip test, allocator-cap recording, background heartbeat refresh, campaign
stage-B refusal and per-job plan persistence.

New coverage: lock acquire/refuse across two processes and release on process
exit; launcher dry-run parsing and refusal when the lock is held; heartbeat
atomicity/fields, background refresh and the per-batch callback; corpus
numeric/decontamination filtering, AIME-style verification, full reasoning-trace
round-trip, verified-only output, manifest hashes/distributions and resume
fingerprint; campaign plan counts/order, stage-A resume and corpus fingerprint,
stage-B result keying/resume, stage-B refusal, B2 rule and per-job plan
persistence; report Wilson/paired-bootstrap maths, tables, charts and summary on
synthetic results.

## Design decisions and ambiguities

- **B2 rule across bridges.** “The block with the lowest calibrated NLL at N=2”
  does not name a bridge, so the implementation takes each block's minimum over
  `none`/`damped`, then the minimum over blocks; ties go to the lower start then
  lower end. This is recorded verbatim in the plan's `b2_rule`.
- **O in the charts.** O is the original model with `n_loops = 0` in result
  files; the report plots it at N=1 as a horizontal reference and keeps it out
  of the looped series.
- **Stage B sampling.** The derived eval config preserves the registered
  sampling (`T=1.0`, `top_p=0.95`, `top_k=64`, thinking on, registered
  `max_new_tokens`) and only forces `samples: 1` and the 182/198 item limits.
  Its tag is `trainfree-exploratory`, so results cannot be mistaken for a
  `prereg-v1` run.
- **Stage A model loads.** Stage A runs `pilot_probe` once per N rather than
  once per (block, bridge, N), reducing model loads from 24 to 4; the plan still
  lists all 24 configs and marks them from the same output file.
- **Lock ownership.** The campaign checks the lock before starting, then
  releases it and lets each child acquire it. This trades a tiny launch race
  (the child exits 75 and the plan marks the job failed) for the guarantee that
  a parent crash cannot orphan an unlocked GPU child.

## Concerns / follow-ups

- **No GPU execution.** Per the brief, the corpus and stage A were not run
  against the real checkpoint. The CPU paths, fake-generator campaign, launcher
  dry run and synthetic-report paths are tested; the controller should still run
  a small `--candidate-limit` corpus smoke before the long run.
- **Methodology review file absent.** `methodology-review-A-architecture-loops.md`
  was not present in this worktree, so the training-free literature note used is
  the repository's `docs/superpowers/references/gemma-loop/02-looped-transformers.md`
  (“Training-Free Looped Transformers (2026)”: damped Euler/RK, mid-depth
  window, naive-repeat collapse). The campaign design (damped vs pure repeat,
  mid-depth blocks, N=1..4 curve) follows that note.
- **Stage A calibration cost.** `fit_model_temperature` rescans the vocabulary
  in 512-token/16k-vocabulary tiles for each golden-section step. It is bounded
  and memory-lean, but it is the dominant stage-A cost; the 3–6 h estimate should
  be checked on the first N job.
- **Corpus decontamination is CPU- and network-heavy.** `gather_eval_texts`
  streams every registered eval set (including the LiveCodeBench release) and
  fails closed unless `--allow-missing-eval` is passed. The manifest records
  each benchmark's load status.
- **LiveCodeBench grading** (only relevant after E3) still depends on the
  separate `.venv-lcb` environment; if it is missing, `run_eval` will fail the
  LCB bench rather than silently scoring compiled code in-process.
- **Remaining cheap Minors left documented:** the launcher's real-start path is
  covered by quoting/dry-run tests but not by a process-survival assertion; the
  nvidia-smi check still fails open when the executable is absent; the legacy
  `kl_n1_n2` JSON alias is kept for backward compatibility; the summary's “best”
  row is descriptive by raw accuracy (exploratory only).
