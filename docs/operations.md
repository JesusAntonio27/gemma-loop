# Running a long Gemma-Loop job

This page is the operational contract for a multi-day training or eval run on
the local Windows PC (and the Linux/cloud equivalent).  The design goal: a
BSOD, driver reset, reboot or accidental kill may lose at most about 30 minutes
of work and can never silently corrupt a run.

## 1. Before launching

```powershell
# clean, tagged worktree, frozen dependencies
git status --short          # must be empty
uv sync --frozen
$env:GEMMALOOP_ARTIFACTS = "C:\Users\<user>\ML\gemmaloop-artifacts"

# no other GPU job
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

`scripts/train.py` refuses to start on a dirty worktree unless `--allow-dirty`
(smoke runs only), and `--require-tag <tag>` refuses unless `HEAD` is exactly
that tag's commit.  `--require-tag` is incompatible with `--allow-dirty`, and a
`--prereg` run requires `--require-tag`.  It also checks that the imported
`gemmaloop` package resolves inside this worktree's `src/`.  It then acquires
`src/gemmaloop/gpulock.py` before any GPU work; if another process holds the
lock it exits **75** without touching the GPU.

## 2. Launch detached

Use the orchestrator, wrapped by the detached launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
  -Name pilot-oh-e2b-s0 `
  -Arguments "-m scripts.orchestrate --out C:\Users\<user>\ML\gemmaloop-artifacts\runs\pilot-oh-e2b-s0 -- .venv\Scripts\python.exe scripts\train.py --arm O-H --size e2b --data C:\Users\<user>\ML\gemmaloop-artifacts\data\heal --tokens 40000000 --seed 0 --out C:\Users\<user>\ML\gemmaloop-artifacts\runs\pilot-oh-e2b-s0 --prereg prereg\prereg.yaml --require-tag prereg-v1 --save-every-minutes 30 --resume auto"
```

The launcher returns immediately and writes the pid and a timestamped log under
`<artifacts>/runs/detached/`.  Killing the editor, the terminal or the Claude
session does not kill the job.

Linux/cloud (same orchestrator, no PowerShell):

```bash
export GEMMALOOP_ARTIFACTS=/data/gemmaloop-artifacts PYTHONUNBUFFERED=1 PYTHONUTF8=1
nohup .venv/bin/python scripts/orchestrate.py --out "$RUN" --name pilot-oh \
  -- .venv/bin/python scripts/train.py --arm O-H ... --resume auto > "$RUN.log" 2>&1 &
```

## 3. What happens on a crash or reboot

* A checkpoint is written to `<run>/checkpoint-<step>.tmp`, every file is
  flushed, `COMPLETE.json` is written last with the SHA-256 of every file, and
  only then is the directory renamed to `<run>/checkpoint-<step>`.  A directory
  without a valid `COMPLETE.json` (or with a hash mismatch) is never resumed.
* Every file in the staging directory is fsynced, and the parent directory is
  fsynced where the OS allows it.
* The last **2** complete checkpoints are kept (plus the final adapter/state in
  the run root).  Stale `*.tmp` directories from a crash are logged and deleted
  at the next start.
* A checkpoint is also saved every `--save-every-minutes` (default 30) of
  wall-clock, at an optimizer-step boundary.  Final adapter, metrics and state
  files are written through `.tmp` + `os.replace`.
* A successful run writes `<run>/DONE.json` atomically after `run.json`.  Any
  later relaunch (train.py or the orchestrator) sees it and exits 0 with
  "already complete", without loading the GPU, re-training or touching the
  adapter/`run.json`.  `--rerun-complete` is the explicit override.  The
  printed logon task is therefore safe to leave registered.
* CUDA/driver errors (`CUDA error`, `CUBLAS_STATUS`, `cuDNN`), OOM at the
  allocator cap and killed children are retryable: train.py **never writes a
  checkpoint after an exception** (counters/RNG may be mid-step); it exits
  **75** and the retry resumes from the last COMPLETE checkpoint.  The orchestrator retries any non-zero exit except 2 while
  `tokens_seen` progresses, and stops after five consecutive no-progress
  failures.
* The orchestrator sees exit **75** and reruns the command with
  `--resume auto`: it picks the newest COMPLETE checkpoint, restores model /
  LoRA / bridge / optimizer (all parameter groups, including the bridge and
  loop_ple groups), the token-based cosine schedule position, `tokens_seen`, the
  exact next dataset row, the Python/NumPy/torch CPU/CUDA RNG states and the
  throughput baseline.  Protection starts immediately after a resume (the
  persisted baseline is used from local step ~20).
  The resumed run is bit-identical to an uninterrupted run on CPU for the
  registered single-epoch recipe.
* `--resume auto` with no checkpoint yet is a normal first start (logged), so
  the orchestrator can retry a lock-busy first launch safely.
* A reboot relaunch also resumes: the orchestrator adds `--resume auto` on its
  first attempt whenever the run directory already has a valid checkpoint
  (`--fresh` disables that).  A bare `train.py --out <existing run>` without
  `--resume` is refused with exit 2 instead of silently overwriting it.
* Every checkpoint stores a run identity hash (run config + data manifest SHA +
  code git SHA + seed).  `--resume auto` only considers matching checkpoints.
  If only `run_identity.json`/`status.json` exist (a crash before the first
  checkpoint) and the identity matches, `--resume auto` starts fresh (logged).
  If checkpoint directories exist but none is valid, train.py exits **2** with
  the per-checkpoint reasons; a fresh start requires `--fresh`, which moves old
  checkpoints/run root files under `<run>/attic/`.
* The orchestrator passes `--fresh` only on the first attempt; retries drop it
  and use `--resume auto`.
* Both the GPU lock and a per-run `<run>/.run.lock` are acquired before any
  cleanup, archive or resume work, so a duplicate launch cannot touch a live
  run.
* On Windows the trainer holds `SetThreadExecutionState(ES_CONTINUOUS |
  ES_SYSTEM_REQUIRED)` for its lifetime (no-op elsewhere, no system setting
  changed); a forced Windows Update restart is still outside its control.
* `--resume <explicit path>` is preserved by the orchestrator until the child
  publishes a newer checkpoint.  Resuming an older checkpoint archives the
  newer branch so auto-resume cannot jump back to it.
* `--resume auto` refuses (exit **2**) if the run configuration, data manifest
  SHA, code git SHA or prereg SHA differ from `COMPLETE.json`, or if the
  checkpoint lacks the optimizer, scheduler, RNG, N-sampler or data-order
  state.  The explicit `--allow-resume-mismatch` is for smoke runs only.
* `--resume` together with `--allow-repeat` is refused with exit **2**: the
  multi-epoch path is not bit-exactly resumed yet.  The registered recipe is
  single-epoch.
* Status/heartbeat writes are best-effort: a file briefly held open by a poller
  or antivirus is logged and retried on the next step, and it can never stop
  training or the guard's checkpoint.

## 4. Exit codes

| code | meaning | orchestrator action |
|------|---------|---------------------|
| 0 | run complete | done |
| 2 | refused: resume identity mismatch, missing required checkpoint state, checkpoints present but none valid (unless `--fresh`), `--allow-repeat` + `--resume`, dirty worktree, `--require-tag` mismatch, missing/invalid checkpoint, fresh start over existing checkpoints | stop |
| 75 | retryable: GPU lock busy, throughput collapse (spill), available RAM < 2 GiB, CUDA/driver error, OOM at the cap, killed child, retryable checkpoint I/O | wait, retry with `--resume auto` while progress continues; progress is counted only from COMPLETE checkpoints, stop after 5 consecutive no-progress failures; lock waits use a separate 24 h budget |

Guard thresholds (recorded in `status.json`, `run.json` and every
`COMPLETE.json`):

* throughput: rolling median over `--throughput-window-steps` (10) optimizer
  steps below `--throughput-drop-ratio` (0.5) of the post-warm-up baseline for
  `--throughput-failures` (3) consecutive windows; the baseline is persisted
  in every checkpoint/status and restored on resume;
* OOM: an over-cap step raises `torch.OutOfMemoryError`, mapped to exit 75
  (there is no separate reserved-memory check);
* RAM: available system memory below `--min-ram-gib` (2.0).

When a guard fires the trainer writes a checkpoint and exits 75; it does **not**
write `run.json`, so a partial run can never be mistaken for a finished one.
If the guard fires on the final budgeted step, the run is treated as complete
(final artifacts + `run.json`) instead of retrying, so a resumed run cannot take
an extra step beyond `max_steps`/the token budget.  A resumed checkpoint whose
restored step/tokens are already at the bound is finalised without another
optimizer step.

## 5. Where to look

| file | contents |
|------|----------|
| `<run>/DONE.json` | atomic completion marker: run identity hash, final step/tokens, UTC time; relaunches are no-ops |
| `<run>/status.json` | phase, step, `tokens_seen`, tok/s, last complete checkpoint, exit reason/code, start/update UTC timestamps, frozen identity, thresholds, throughput baseline |
| `<run>/heartbeat.json` | small liveness file (same counters + reserved VRAM + available RAM) |
| `<run>/run_identity.json` | git SHA / dirty flag / data manifest SHA / prereg SHA / identity hash / smoke flag, written at start |
| `<run>/.run.lock` | process-lifetime per-run lock (duplicate launch protection) |
| `<run>/orchestrator.jsonl` | one JSON line per launch, exit, wait and stop |
| `<run>/checkpoint-<step>/COMPLETE.json` | checkpoint manifest with per-file SHA-256 |
| `<run>/run.json` | written only after a complete run; includes identity hash, thresholds and resume provenance |
| `<run>/attic/` | `--fresh` / rewind quarantines old checkpoints and run root files here |
| `<artifacts>/runs/detached/*.log` | detached launcher stdout/stderr |

Every log line is timestamped; the orchestrator sets `PYTHONUNBUFFERED=1` and
`PYTHONUTF8=1` for the child so a detached log is readable while it is written.

## 6. Resume by hand

```powershell
.venv\Scripts\python.exe scripts\train.py ... --resume auto
```

or point at a specific checkpoint with `--resume <run>\checkpoint-<step>`.

## 7. Auto-resume after reboot

`scripts/register_autoresume.ps1` **prints** the exact `schtasks` command (and a
systemd user unit / cron `@reboot` equivalent).  It never registers anything.
Review the printed command and run it yourself.  It is safe to leave the task
registered: a completed run has `DONE.json` and a relaunch is a no-op.  The
printed Windows action begins with `set GEMMALOOP_ARTIFACTS=... && cd /d
"<repo>"`, uses an absolute Python/orchestrator path and passes `--resume auto`;
the systemd unit sets `WorkingDirectory` and the cron line `cd`s to the repo.
The alternative `Register-ScheduledTask` form printed by the script sets
`-WorkingDirectory` and an execution limit of zero; the plain `schtasks` form
inherits the default 72 h limit, so use the former (or the Task Scheduler GUI)
for a multi-day run.  A BSOD reboots to the lock screen: nothing resumes until
the author logs on (ONLOGON).

## 8. Notes and limits

* The orchestrator does not hold the GPU lock; it only probes that it is free.
  The child owns the lock, so a crashed parent cannot leave an unlocked child.
  If the lock is held it waits/retries with the same backoff schedule instead of
  exiting immediately.
* Eval scripts keep their own retry behaviour; `scripts/orchestrate.py` is
  command-agnostic (`train.py`, `run_eval.py`, a campaign stage, …).
* GPU pytest is opt-in (`--run-gpu`); plain `uv run pytest` never opens a CUDA
  context.
