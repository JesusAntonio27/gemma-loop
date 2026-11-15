# Task S0 — Stage 0 data pipeline (implementation report)

Status: **DONE_WITH_CONCERNS** (code/tests complete; no GPU, Modal, Hub download or
real DeepSeek request was executed in this environment).

Authoritative inputs read from the `prereg1` worktree:

- `prereg/PREREGISTRATION.md` §4.1 Stage 0, §5 Data, §10 Compute, §11 Pins
- `prereg/prereg.yaml` `data`, `splits`, `stage2`, `pins`, `compute`
- `twostage-vs-current-design.md` (Stage 0, DeepSeek pricing/terms)
- `progress.md` Modal Ruling (Starter, stock vLLM v0.30.0, $30 usage / $0 spend,
  delete volumes after use because storage keeps billing)
- `audit-eval-stats.md` (decontamination coverage)
- `data-length-audit.md` (chat source, 8192 length rules)

Nothing in `pyproject.toml` or `uv.lock` was changed, and no training code was
edited. The paid entry point is the local script with `uv run --with
modal==1.6.3 python scripts/modal_stage0.py ...`; the `modal run` CLI is not
used because it creates the app and builds the image before the budget check.

## What was built

| Item | Module / script | Tests |
|---|---|---|
| Pool builder, pinned sources, normalised schema, EvalIndex decontamination, deterministic splits, manifest hashes | `src/gemmaloop/stage0/sources.py`, `evals.py`, `pool.py`; `scripts/build_stage0_pool.py` | `tests/stage0/test_pool.py` |
| Persistent CPU grader service, LCB-compatible code tests, AIME math normalisation, Docker sandbox option + flagged local fallback | `src/gemmaloop/stage0/grader.py` | `tests/stage0/test_grader.py` |
| Official DeepSeek hidden-test writer, env-only key, dry-run/cost, hard cap, off-peak, reference-solution filtering | `src/gemmaloop/stage0/deepseek.py`; `scripts/write_stage0_hidden_tests.py` | `tests/stage0/test_deepseek.py` |
| Modal sampling job (stock vLLM, pinned snapshot, sharded Volume + resume, download, volume delete, cost planner, dry-run) | `src/gemmaloop/stage0/modal_plan.py`; `scripts/modal_stage0.py` | `tests/stage0/test_modal_plan.py` |
| Post-processing: sample filter, verification, pass rates, Stage 2 prompt selection, Stage 1 heal rows, pilot/NLL/long-trace corpora, manifest update | `src/gemmaloop/stage0/postprocess.py`; `scripts/postprocess_stage0.py` | `tests/stage0/test_postprocess.py` |

### Registered behaviour encoded

- **Sources and pins.** `taco` `BAAI/TACO@d593ed0…`, `apps`
  `codeparrot/apps@21e74dd…`, `codecontests`
  `deepmind/code_contests@802411c…`, `omr`
  `nvidia/OpenMathReasoning@d3d0866…`, `dapo`
  `Bytedance-SIA/DAPO-Math-17k@6587709…` (repository
  `BytedTsinghua-SIA/DAPO-Math-17k`), `deepscaler`
  `agentica-org/DeepScaleR-Preview-Dataset@b6ae8c6…`, `chat`
  `HuggingFaceH4/ultrachat_200k@8049631…` (the existing decontaminated chat
  source). “Synthetic” code was not added.
- **Schema.** `problem_id`, `source`, `kind`, `prompt`, `tests`/`answer`,
  `metadata`, `bucket`, `split`, `pilot_subsplit`.
- **Decontamination.** The repo `EvalIndex` (13-gram + normalised Indel
  `>= 0.75`) against LiveCodeBench v6, GPQA Diamond, AIME 2026, MMLU-Pro,
  MMMU-Pro, MATH-Vision, SpatialEval, CV-Bench 2D and the creative briefs.
  The prompt, reference solution and exact answer are checked; verdicts are
  counted in the manifest.
- **Splits.** `bucket = int(sha256(problem_id), 16) % 100`: 0-49 train,
  50-59 pilot dev, 60-69 NLL/KL, 70-79 long traces, 80-99 held-out; pilot
  50-53 P3, 54-56 selection, 57-58 T-fit, 59 P1.
- **Grader timeout.** `6 s` per test, per the registered Stage 2 value.
- **DeepSeek.** Official base URL only (`https://api.deepseek.com`); any OpenCode
  host is refused; key from `DEEPSEEK_API_KEY` only; model alias
  `deepseek-chat` records registered model `DeepSeek-V4.1-Flash-1017`; prices
  `$0.15` input / `$0.60` output per M tokens, 2× peak (weekdays 01-04 and
  06-10 UTC); hard cap checked against the pre-flight estimate; off-peak wait;
  failing-reference tests dropped; one output shared by both arms.
- **Modal.** `google/gemma-4-E2B-it@4adf157…`, `vllm==0.30.0`, T=1.0,
  top_p=0.95, top_k=64, thinking on, max_new_tokens=32768, 4 samples/prompt,
  shards of 100 problems, skip-existing resume, local download, cleanup deletes
  the Volume. Plan cap `--max-usd` default 12 and hard registered limit $30.
- **Post-processing.** `finish_reason=stop`, closed thought channel, non-empty
  answer, rendered length <= 8192, reasoning share >= 75 % for code+math;
  Stage 2 prompts from train-bucket code/math with 1-3 correct and median
  length <= 6,000; Stage 1 rows from the train bucket; pilot verified traces
  from 50-59; NLL/KL verified traces from 60-69; long traces from 70-79.

## Exact commands, in order

Windows shell; activate the project environment with `uv run` (no global
installs). Run from `C:\Users\<user>\ML\GemmaLoop-wt\stage0`.

### 1. Pool build + decontamination + deterministic splits

Decontamination is part of this command (the manifest records the EvalIndex
size, targets and per-source drop counts). Use `--eval-limit 64` only for a
smoke run.

```powershell
uv run python scripts/build_stage0_pool.py --output-dir artifacts/stage0/pool
```

Outputs: `problems.jsonl`, `split_train.jsonl`, `split_pilot_dev.jsonl`,
`split_nll_kl.jsonl`, `split_long_traces.jsonl`, `split_heldout_test.jsonl`,
the four pilot sub-range files, and `manifest.json` with counts and SHA-256 of
every artefact.

### 2. DeepSeek hidden-test cost dry run (no request, no key needed)

```powershell
uv run python scripts/write_stage0_hidden_tests.py --pool-dir artifacts/stage0/pool --dry-run --max-usd 12
```

Real run (author-approved budget and key; official endpoint only):

```powershell
$env:DEEPSEEK_API_KEY = "<author's key>"
uv run python scripts/write_stage0_hidden_tests.py --pool-dir artifacts/stage0/pool --output artifacts/stage0/hidden_tests.jsonl --max-usd 12 --off-peak --sandbox docker
```

### 3. Modal plan / cost dry run (no Modal contact)

```powershell
uv run python scripts/modal_stage0.py --dry-run --gpu A100-80GB --max-usd 12
```

### 4. Modal launch (stock vLLM, pinned snapshot)

Every launch needs the author's approval; `--yes` is required or the job exits
after printing the plan.

```powershell
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode sample --pool-dir artifacts/stage0/pool --gpu A100-80GB --max-usd 12 --yes
```

Shards land in the `gemmaloop-stage0` Volume under `/data/shards/`; a rerun
skips shards already present.

### 5. Download results

```powershell
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode download --output-dir artifacts/stage0/samples
```

### 6. Delete the Volume (mandatory — storage bills past the usage limit)

```powershell
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode cleanup --output-dir artifacts/stage0/samples
```

The cleanup command calls Modal's volume-delete API. If that API is unavailable
in the installed client it exits non-zero and tells the operator to run
`uvx modal volume rm gemmaloop-stage0 --yes`; the volume must not be left
behind.

### 7. Post-processing (local CPU)

```powershell
uv run python scripts/postprocess_stage0.py --pool-dir artifacts/stage0/pool --samples artifacts/stage0/samples --hidden-tests artifacts/stage0/hidden_tests.jsonl --output-dir artifacts/stage0/postprocess --sandbox docker
```

Code verification uses the Docker sandbox (`--network none`) by default. The
local path is available only with both `--sandbox local --allow-unsandboxed`,
and is explicitly **non-sandboxed**. Outputs:
`sample_grades.jsonl`, `stage1_heal.jsonl`, `stage2_prompts.jsonl`,
`pilot_verified.jsonl`, `nll_kl_verified.jsonl`, `long_traces.jsonl`,
`postprocess_manifest.json`, and the `postprocess` section plus artefact hashes
in `artifacts/stage0/pool/manifest.json`.

## Estimated Modal cost per GPU type

Fix-round cost model: 5,000 prompts x 4 samples = 20,000 generations, about
64M generated + 10M prompt tokens, shard size 500 (10 sequential containers),
180 s startup/download per container, 8 CPU cores, 32 GiB memory, x1.3 safety
margin. All values are all-in USD (GPU + CPU + memory + startup), not GPU-only.

| GPU | tok/s (plan) | GPU-hours | GPU$ | CPU$ | MEM$ | all-in USD |
|---|---:|---:|---:|---:|---:|---:|
| B300 | 22,000 | 1.57 | 11.18 | 0.59 | 0.40 | 15.83 |
| B200 | 20,000 | 1.68 | 10.51 | 0.63 | 0.43 | 15.05 |
| H200 | 17,000 | 1.89 | 8.58 | 0.71 | 0.48 | 12.71 |
| H100 | 15,000 | 2.08 | 8.20 | 0.78 | 0.53 | 12.37 |
| RTX PRO 6000 | 12,000 | 2.47 | 7.48 | 0.93 | 0.63 | 11.76 |
| **A100-80GB (default)** | **10,000** | **2.86** | **7.16** | **1.08** | **0.73** | **11.66** |
| A100-40GB | 8,000 | 3.45 | 7.26 | 1.30 | 0.88 | 12.27 |
| L40S | 7,000 | 3.88 | 7.56 | 1.46 | 0.99 | 13.01 |
| A10 | 4,500 | 5.75 | 6.33 | 2.16 | 1.47 | 12.95 |
| L4 | 3,500 | 7.25 | 5.80 | 2.73 | 1.86 | 13.50 |

A100-80GB is the default and its all-in plan is below the default $12 cap;
H200/H100/A100-40GB/L40S/A10/L4/B200/B300 exceed $12 in this planning model
and are refused by `--max-usd 12` (use the registered $30 limit only with the
author's explicit approval). These remain planning estimates until the
throughput probe; `max_containers=1` and no retries bound the real run.

## Tests

Command run (CPU only, CUDA hidden by `tests/conftest.py`):

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
uv run pytest -m "not gpu and not slow" -q
```

Result after fix round 1: **775 passed, 22 deselected, 14 warnings in
196.63s**. The new Stage 0 tests are 45 of those (`tests/stage0/`: pool 6,
grader 10, DeepSeek 9, Modal plan 6, Modal job 9, post-processing 5). No test calls the real DeepSeek API, Docker, Modal
or the network; all use fake clients/loaders/fixtures.

## Concerns / notes for the controller

1. **No cloud or Hub execution here.** The pinned datasets, the Modal vLLM run,
   the real DeepSeek writer and the Docker sandbox were not executed. Their
   offline paths (planning, command construction, fakes, resume logic, hashes)
   are tested; the first real run still needs the author's Modal token,
   approval, and a throughput probe.
2. **Source revisions were not listed in the prereg.** The code pins the
   immutable Hub commits that were current when this task was written (table
   above) and records them in every manifest. If the controller wants different
   snapshots, change the constants before the data manifest is frozen.
3. **`math-verify` is not installed** (it is not a project dependency).
   `grade_math` uses the AIME normaliser from `eval/benchmarks/aime.py` and
   automatically uses `math_verify` if it ever appears. This is documented in
   the grader module. The registered `pins.math_verify` still needs to be
   filled in `prereg-v1-data`.
4. **CodeContests “pre-2023-05”** is recorded as the dataset's date window; the
   dataset has no per-row date field to re-filter, so no extra date filter is
   invented.
5. **Post-processing length filter** uses the full rendered length when a
   tokenizer is supplied and the stored rendered/prompt+completion counts
   otherwise. Stage 2 “median length ≤ 6k” is the median of kept sample
   rendered lengths, matching the registered prompt-selection rule.
6. **Manifest SHA.** `manifest.json` carries the per-run artefact hashes and a
   `manifest_sha256`. This is not yet the prereg `pins.manifest_sha256`; that
   value is frozen in the post-tag `prereg-v1-data` addendum after the real
   build, as registered.
7. **Docker sandbox** is CPU-only, `--network none`, read-only root, tmpfs
   `/tmp`, memory/PID/CPU limits, non-root, `no-new-privileges`. The local path
   is explicitly `sandboxed=False` and `sandbox="local-fallback"` if Docker is
   requested but unavailable.
8. **DeepSeek key hygiene.** Only `DEEPSEEK_API_KEY` is read; there is no file
   argument, no key in logs or exceptions, and OpenCode endpoints are refused
   before client construction. The real API was never called in tests.
9. **Modal cleanup.** The volume-delete path uses Modal's volume CLI callback
   because the Volume object API has no `delete` method in modal 1.6.3. If a
   future Modal version changes that, the cleanup command fails closed and
   prints the exact `modal volume rm` command rather than leaving storage
   billing.


## Fix round 1 (review items C1-C6, I1-I12, cheap Minors)

Status after fixes: all Critical and Important review items addressed; no real
Modal, Hub, DeepSeek or Docker call was made in this round either.

### Money-safety changes (Modal)

- `max_containers=1`, sequential `.remote()` calls, **no automatic retries**
  (resume is per shard), per-call `timeout=7200 s` (2 h).
- All-in cost model: GPU + 8 CPU cores ($0.047/core-h) + 32 GiB memory
  ($0.008/GiB-h) + 180 s startup/container, multiplied by a x1.3 safety margin.
  `format_plan` prints every component.
- Budget is checked from the plan **before any Modal API object is touched**;
  no Volume is created for refused plans, downloads or cleanup.
- Runtime guard tracks observed wall time per shard and stops before
  `spent + next_estimate > --max-usd`; partial shards remain durable for resume.
- Model weights download to `/tmp/gemmaloop-hf-cache` inside the container
  (ephemeral), never to the results Volume.
- A failed `volume.commit()` now raises and fails the shard.
- Download uses volume-relative `shards/...` paths, iterates `read_file` chunks,
  requires every expected shard (count derived from the pool or
  `--expected-shards`), checks non-empty data, writes every local file, and
  stores a `download_receipt.json` with SHA-256 per shard.
- Cleanup refuses to delete the Volume unless the receipt matches the expected
  count and every local shard hash; it then uses
  `modal.Volume.objects.delete(...)`.
- `max_model_len = max(prompt tokens in shard) + 32768`; the effective value and
  `max_prompt_tokens` are stored per sample. `SamplingParams` uses
  `skip_special_tokens=False`; channel parsing is tested against the repo's real
  Gemma 4 marker fixtures (`tests/fixtures/gemma_outputs/reasoning.txt`).
- The script inserts the repo `src/` on `sys.path` before importing
  `gemmaloop`, and the Modal image uses `add_local_python_source("gemmaloop")`,
  so the local `uv run --with modal==1.6.3 python scripts/modal_stage0.py ...` flow works from the repo root.
- New fake-Modal tests: `tests/stage0/test_modal_job.py` (concurrency=1, no
  retries, budget refusal before contact, runtime guard stop, download
  verification, cleanup refusal, shard resume/commit failure, LLM/SamplingParams
  arguments, channel parsing).

### Other fixes

- **DeepSeek cap stop is graceful**: a mid-run cap hit sets
  `stopped_reason="spending_cap"` and returns every record already paid for;
  the CLI writes them and records the stop reason. The pre-flight over-cap
  refusal remains hard.
- **Grader secret scrubbing and sandbox default**: candidate processes get a
  sanitized environment (`PATH`, locale, temp only) and never inherit
  `DEEPSEEK_API_KEY`/HF/Modal tokens. Code execution defaults to the Docker
  sandbox (`--network none`); the local path requires explicit
  `--sandbox local --allow-unsandboxed` (and never falls back silently). Docker
  per-test timeout is exactly 6 s; the image is pre-pulled outside the timer.
  Windows timeout kills the process tree.
- **Creative briefs** are always named as a decontamination target; a missing
  file is recorded as `missing` in the manifest and the production build fails
  closed unless `--allow-missing-briefs` is passed for a smoke run.
- **Tokenizer reasoning share**: post-processing passes the real tokenizer
  counter and records `reasoning_share_method`.
- **Stage 2 selection** now requires exactly 4 generated samples per problem,
  counts correctness over those 4, uses the median over all 4 rendered lengths,
  and records `n_generated`/`n_kept`. `long_traces` applies the 4096-token floor
  and pilot traces carry `pilot_subsplit`.
- **Manifest integrity**: post-processing recomputes `manifest_sha256` after
  adding its section and records `previous_manifest_sha256`.
- **Pool builder** exits non-zero when a mandatory source fails or is short
  (`--allow-short-sources` is smoke-only), records dataset revisions, tokenizer
  revision and an eval-index SHA.

### Dataset revisions for the prereg-v1-data addendum

- `taco`: `BAAI/TACO` config `ALL` split `train`, revision
  `d593ed0a2becbbc952230bb89be09189bf1056dc`
- `apps`: `codeparrot/apps` split `train`, revision
  `21e74ddf8de1a21436da12e3e653065c5213e9d1`
- `codecontests`: `deepmind/code_contests` split `train`, revision
  `802411c3010cb00d1b05bad57ca77365a3c699d6`, date window `pre_2023_05`
- `omr`: `nvidia/OpenMathReasoning` config `default` split `cot`, revision
  `d3d08664755704f422af97d43a7ff0ded4bd95df`
- `dapo`: repository `BytedTsinghua-SIA/DAPO-Math-17k` split `train`,
  revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`
- `deepscaler`: `agentica-org/DeepScaleR-Preview-Dataset` split `train`,
  revision `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`
- `chat`: `HuggingFaceH4/ultrachat_200k` config `default` split `train_sft`,
  revision `8049631c405ae6576f93f445c6b8166f76f5505a`
- model snapshot: `google/gemma-4-E2B-it`
  `4adf15748de9e1a0097b6274cee9c15836c41c37`; tokenizer revision same.

The builder reads `DATASET_REVISIONS` from
`src/gemmaloop/stage0/sources.py`; the manifest records it.

### Commands delta

Use `--sandbox docker` (default) for the DeepSeek writer and post-processing;
add `--allow-unsandboxed` only if Docker is unavailable and the operator accepts
the non-sandboxed fallback. Download then verify before cleanup:

```powershell
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode download --pool-dir artifacts/stage0/pool --output-dir artifacts/stage0/samples
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode cleanup --output-dir artifacts/stage0/samples --output-dir artifacts/stage0/samples
```

Cleanup fails with exit code 2 if the download receipt is missing or any shard
count/SHA-256 does not match.


## Fix round 2 (review r2 items I1-I7, cheap Minors)

Status: all 7 Important money/test items fixed; no real Modal, DeepSeek, HF or
Docker call was made in this round either.

### Money safety

- **Worst-case pre-launch guard (I1).** Before every container the script
  requires
  `cumulative_month_usd + runtime_cost_usd(gpu, 2h + startup, all-in, x1.3) <= min(max_usd, 30)`.
  `max_usd` is validated in code and values above the registered $30 ceiling are
  refused. A failed/timed-out shard is settled at the worst case, never at zero.
- **Persistent monthly ledger (I2).** `src/gemmaloop/stage0/spend.py` writes an
  append-only JSON ledger under the artifacts dir, by default
  `<artifacts_dir>/stage0/modal_spend_ledger.json` (normally
  `C:/Users/<user>/ML/gemmaloop-artifacts/stage0/modal_spend_ledger.json`).
  Do not pass a repo-relative `--ledger`; that would start a fresh guard.
  Entries are timestamped reservations and actuals, written atomically with
  `.tmp` + `os.replace`. The guard uses the current calendar month's cumulative
  actual spend plus outstanding reservations, so crashes/resumes cannot reset
  the counter. A reservation is written before `.remote()`.
- **Download/cleanup are not budget-gated (I3).** `download` and `cleanup` skip
  the plan/budget check entirely; they validate only the receipt and arguments.
  Retrieval no longer depends on `--max-usd` or a matching `--shard-size`.
- **Partial-run receipts and cleanup (I4).** Launch writes
  `launch_manifest.json` with the planned shard count. Download verifies every
  shard present, writes `download_receipt.json` with
  `expected_shards/present_shards/missing/partial/stopped_reason` and per-shard
  SHA-256, and succeeds for a partial run. Cleanup accepts a partial receipt only
  with explicit `--allow-partial`; every refusal prints the exact manual
  `uvx modal volume rm gemmaloop-stage0 --yes` command and the reason.
- **DeepSeek dry-run default (I5).** The writer is dry-run unless `--live` is
  passed; `--live` still requires `DEEPSEEK_API_KEY` (env only) and the
  pre-flight/running spend caps. `--live` and `--dry-run` are mutually exclusive.
- **Production commit path (I6).** New tests call `sample_shard` without the
  test-only `_volume` key, using a fake `modal.Volume.from_name` inside the
  container; a missing volume and a failed commit both fail loudly. Commit
  failure also removes the uncommitted shard and its `.committed` marker.
- **Discriminating guard test (I7).** `test_worst_case_guard_stops_sequentially`
  counts `.remote()` calls, parses the summary, and asserts exactly one call,
  `stopped_reason == "spending_cap"`, cumulative spend <= cap and
  `cumulative + worst_case > cap`. Removing the guard makes it fail.

### Cheap Minors fixed

- Sanitized grader env no longer passes `HOME`/`USERPROFILE`.
- `parse_completion` strips trailing `<turn|>`/`<eos>`/`<end_of_turn>`.
- Shard skip requires a committed marker; commit failure deletes both files.
- Launch manifest persists the shard plan; download uses it.
- Stale sandbox comments updated.
- Briefs gate checks the realised status: missing and corrupt briefs fail the
  production build, and an `--extra-eval-file` can satisfy the target.
- Non-cap DeepSeek aborts now stop gracefully and keep paid records.
- `shard_size`, `avg_generated_tokens`, `samples_per_prompt` and `max_usd` are
  validated before any Modal contact.

Not fixed (listed): Docker per-test timer still includes `docker run` startup
(pre-pulled image only), and the `prereg-v1-data` addendum itself remains a
controller governance step (the seven revisions are listed above and in the
manifest).

### Commands delta for the ledger / partial flow

```powershell
uv run python scripts/write_stage0_hidden_tests.py --pool-dir artifacts/stage0/pool --max-usd 12          # dry run (default)
uv run python scripts/write_stage0_hidden_tests.py --live --pool-dir artifacts/stage0/pool --max-usd 12 --off-peak --sandbox docker
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode sample --pool-dir artifacts/stage0/pool --max-usd 12 --yes
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode download --pool-dir artifacts/stage0/pool --output-dir artifacts/stage0/samples
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode cleanup --output-dir artifacts/stage0/samples --output-dir artifacts/stage0/samples --allow-partial   # only for a verified partial run
```

Test result after fix round 2: **788 passed, 22 deselected** (58 Stage 0 tests).


## Fix round 3 (review r3 items I1-I4, cheap money Minors)

- **I1 failed Volume commit cleanup.** `sample_shard` wraps Volume handle
  resolution (`modal` missing, `Volume.from_name`) and `commit()` in one cleanup
  block: on any failure it removes both `shard_XXXXX.jsonl` and
  `shard_XXXXX.jsonl.committed` before re-raising. A reused container therefore
  regenerates the shard instead of skipping it. Tests assert both files are gone
  for a missing Volume and for a failed commit.
- **I2 process-lifetime spend lock.** `spend.py` implements the
  `msvcrt`/`fcntl` byte-range lock used by the GPU lock, as
  `<artifacts_dir>/stage0/spend.lock`, acquired by Modal sample launches and by
  the DeepSeek `--live` writer **before reading the ledger** and held until
  process exit. A second spender exits 75 with a clear message. Ledger writes
  remain atomic and now `fsync` the temp file and directory. Tests spawn two
  processes: the second `try_acquire` returns None and a child
  `require_spend_lock` exits 75; two sequential process-ledger writers keep both
  reservations.
- **I3 discriminating DeepSeek CLI test.** The test calls `main([])` with
  monkeypatched `HiddenTestWriter`/`GraderService`: the default path builds no
  grader and no client (`dry_run=True`), while `--live` without
  `DEEPSEEK_API_KEY` raises `MissingAPIKey`.
- **I4 report/commands.** Modal commands consistently use
  `uv run --with modal==1.6.3 python scripts/modal_stage0.py ...`; the sample command no longer
  passes a repo-relative `--ledger`. The default ledger path is documented as
  `<artifacts_dir>/stage0/modal_spend_ledger.json` and asserted by a test.
- Cheap money Minors fixed: `HiddenTestWriter` library default is now
  `dry_run=True`; `--max-usd 0` is accepted for download/cleanup; trailing
  structural tokens are stripped after whitespace; the briefs gate identifies
  the actual briefs file (not any extra eval file) and is tested; commit-failure
  and missing-volume tests assert the marker is removed; the guard test pins the
  exact worst-case formula; a non-cap DeepSeek abort test keeps paid records.
- Still listed, not fixed: Docker per-test timer includes container startup; a
  stale local receipt cannot detect a newer Volume shard set; the
  `prereg-v1-data` addendum remains a controller step; `--max-usd 12` can stop
  a 10-shard run early, so resume with the registered $30 ceiling only with
  explicit approval.

Test result after fix round 3: **795 passed, 22 deselected** (65 Stage 0 tests).


## Fix round 4 (native audit)

All 8 Important native-audit money items and the cheap Minors M1-M10 are fixed;
no Modal/DeepSeek/HF/Docker contact was made in this round.

### Exact approved-launch commands

The paid entry point is the **local Python script** with the pinned Modal client;
the `modal run` CLI is never used (it creates the app and builds the image before
the budget check). Dry-run does not import modal.

```powershell
# 0. offline plan (no Modal import/contact)
uv run python scripts/modal_stage0.py --dry-run --pool-dir artifacts/stage0/pool --gpu A100-80GB --max-usd 12

# 1. throughput probe = the first shard of the real run (same run id)
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode sample   --pool-dir artifacts/stage0/pool --gpu H100 --timeout-seconds 1800   --max-shards 1 --max-usd 12 --init-ledger --yes   --frozen-manifest-sha <pool/manifest.json manifest_sha256>

# 2. full Stage 0 sample after the probe sets the shard timeout
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode sample   --pool-dir artifacts/stage0/pool --gpu A100-80GB --timeout-seconds 1800   --max-usd 12 --yes   --frozen-manifest-sha <pool/manifest.json manifest_sha256>

# 3. download and verify, then 4. delete the Volume
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode download   --output-dir artifacts/stage0/samples
uv run --with modal==1.6.3 python scripts/modal_stage0.py --mode cleanup   --output-dir artifacts/stage0/samples
```

The spend ledger default is `<artifacts_dir>/stage0/modal_spend_ledger.json`;
use `--init-ledger` only for the first run. The process-lifetime
`<artifacts_dir>/stage0/spend.lock` makes a second spender exit 75.

### Worst-case $ per call (default timeout 7200 s, startup 600 s, scaledown 10 s, cpu 8, memory 48 GiB, x1.3, +$0.25 image)

| GPU | Worst case/call |
|---|---:|
| H100 | $13.53 |
| RTX PRO 6000 | $10.94 |
| A100-80GB | $9.44 |
| L40S | $7.89 |
| L4 | $4.65 |

With the post-probe `--timeout-seconds 1800`, each bound scales down (for
A100-80GB it is about $4.9) so `--max-usd 12` can finish the 10-shard plan.

### Native-audit fixes

- **I1 local entry/Dry-run.** No `modal run`; `main()` imports Modal only for
  non-plan modes. Dry-run is offline and tested by making `_import_modal` raise.
  Lock, frozen-pool check and the first worst-case reservation happen before
  `with app.run()`. The reported "no Modal contact" claim now refers only to the
  local script.
- **I2 delete command/cleanup.** Every printed fallback is
  `uvx --from modal==1.6.3 modal volume delete gemmaloop-stage0 --yes`.
  Cleanup re-lists the Volume, refuses if any listed shard is not in the local
  receipt/size-verified copy, allows an empty Volume, and sample prints the
  reminder + exact command on every exit path.
- **I3 declared resources.** Function and `with_options` declare
  `cpu=(8,8)`, `memory=(49152,49152)`, `startup_timeout=600`,
  `scaledown_window=10`; worst case is `(timeout + startup + scaledown) x
  all-in x1.3 + image build`. `--timeout-seconds` is the probe knob.
- **I4 DeepSeek ledger/append/resume.** The same monthly `SpendLedger` is used
  with `provider="deepseek"`; each request is reserved before the call and
  settled after. Records append atomically to `<output>.partial.jsonl` after
  each paid problem; a resume skips IDs already present.
- **I5 run identity.** Shards live under
  `shards/<run_id>/`, where `run_id = sha256(pool manifest sha, shard size,
  samples/prompt, model revision, sampling)`. The launch manifest records it and
  mismatched/probe shards cannot be reused.
- **I6 frozen pool gate.** Paid sample refuses a missing manifest, a
  `problems.jsonl` hash mismatch, a non-recomputing `manifest_sha256`, missing
  decontamination, bad targets, or any smoke limit flag (`eval_limit`,
  `limit_rows`, `allow_missing_*`, `allow_short_sources`, `allow_duplicates`).
  `--frozen-manifest-sha` / `GEMMALOOP_STAGE0_FROZEN_MANIFEST_SHA` pins the
  frozen SHA when available.
- **I7 mutation tests.** Tests now fail if the spend lock, `--yes` gate,
  pre-Volume guard, `max_containers` in `with_options`, or pre-reservation
  ordering is removed.
- **I8 engine once per container.** `_cached_engine` reuses one `LLM` for the
  same model/context in a container; the context is a single payload value so a
  reused container does not rebuild vLLM. Startup/remote failure settles the
  worst case and stops (no retry).
- Cheap Minors: seeds (`LLM(seed=0)` and per-problem `SamplingParams(seed=...)`),
  pinned image versions, `trust_remote_code=False`,
  `limit_mm_per_prompt={"image":0,"audio":0}`, image-build overhead, peak-priced
  DeepSeek guard and `max_retries=0`, missing-ledger refusal (`--init-ledger`),
  plan warning when the pool is missing, and download requiring `.committed`
  markers. Still not fixed: the Docker per-test timer includes container
  startup; the `prereg-v1-data` addendum remains a controller step.

Test result after fix round 4: **802 passed, 22 deselected** (72 Stage 0 tests).


## Fix round 5

- **C1 in-container commit.** `sample_shard` now calls `_import_modal()` when no
  test `_volume` is supplied, so the container's commit path works even though
  `main()` never runs there. A fresh-module test injects a fake `modal` package
  before import, calls `sample_shard` without patching the module global, and
  asserts one commit and the kept shard.
- **C2 integer Modal settings.** `--timeout-seconds`, `--startup-timeout` and
  `--scaledown-window` are argparse `int`; the function and `with_options` pass
  `int(...)`. A strict fake App rejects non-int values; the real Modal client is
  not needed for this test.
- **I-A resumed partial count.** The launch manifest records the total
  `planned_shards` plus `already_committed` and `remaining_shards`; download uses
  the original total, so a resumed run still reports `partial: true` and lists
  every missing shard. A 4-shard/2-present test covers it.
- **I-B cleanup listing fail-closed.** Only a `NotFoundError` (no Volume/shards)
  counts as empty. Any other listing error refuses cleanup (rc 2) and prints
  `DELETE_COMMAND`; tests cover both the error and not-found paths.
- **I-C discriminating tests.** Added tests that fail if the cleanup spend lock,
  DeepSeek monthly-ledger starting total, run-id Volume namespacing, pool sha in
  the run id, the Volume reminder on exit, or the missing-ledger refusal is
  removed.
- Cheap fixes: `--run-id` is wired into download; non-partial receipts check the
  expected shard count again; download error paths print the manual delete
  command; an `app.run()` startup failure settles only the image-build overhead;
  and the probe command uses `--max-shards 1` on the real pool (same run id).
- Still not fixed: DeepSeek stores no status line for paid-but-unparseable
  problems (they can be re-requested on resume); `engine_max_model_len` is fixed
  at 131072; the frozen-manifest SHA is optional until the addendum exists;
  Docker per-test timer includes container startup.

Test result after fix round 5: **811 passed, 22 deselected** (81 Stage 0 tests).


## Fix round 6

- **T4 removed.** T4 lacks bf16 (the job forces `dtype="bfloat16"`), so it is
  gone from `MODAL_GPU_IDS` and both cost tables. A test asserts
  `--gpu T4` is rejected and T4 is absent from `GPU_HOURLY_USD` /
  `GPU_TOKENS_PER_SEC`.
- **Pinning tests.** Added tests that fail if `run_mode` writes
  `planned_shards = len(payloads)` instead of the total (M18), if
  `with_options` receives a float timeout, if the non-partial receipt count
  check is removed (M20), or if `--max-shards` is ignored (M21).
- **Approved-launch commands** now include
  `--frozen-manifest-sha <pool/manifest.json manifest_sha256>` on both paid
  commands.
- **Probe worst case per GPU at `--timeout-seconds 1800`** (startup 600 s,
  scaledown 10 s, cpu 8, memory 48 GiB, x1.3, +$0.25 image):

  | GPU | Probe worst case/call |
  |---|---:|
  | H100 | $4.35 |
  | RTX PRO 6000 | $3.55 |
  | A100-80GB | $3.09 |
  | L40S | $2.61 |
  | L4 | $1.61 |

  If the probe shard times out, that is a lower bound on throughput; the
  worst-case settlement remains bounded by the table above.

Test result after fix round 6: **815 passed, 22 deselected** (85 Stage 0 tests).
