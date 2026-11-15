# T-arch report — configurable loop block, gch bridge, loop-only adapters, calibrated NLL, pilot probe

Status: complete. No `prereg/` files were modified.

## Commits

- `2175f89` feat(model): accept an explicit KV-consumer loop block
- `57ab4c9` feat(model): add zero-init norm-matched per-channel gch bridge
- `bfd593c` feat(model): add loop_only adapter policy and configurable LoRA target sets
- `dfc011f` feat(eval): add temperature-calibrated NLL with lean chunked head
- `add41e6` feat(scripts): add CPU/GPU pilot probe for block and bridge sweep
- `c1c103b` docs(arch): document layer_scalar and residual-stream geometry
- `d7d0cbd` test(model): cover config-driven block, static cache and result metadata
- `653b2fb` test(train): verify gch parameter joins the bridge optimizer group

## What was implemented

1. **Configurable block.** `LayerPlan` / `make_layer_plan(config, block=(start, end))`
   accepts an inclusive block. Validation requires `first_shared <= start <= end <
   num_layers - 1`; producers are still rejected by `_validate_plan`, and an empty
   coda is rejected. `LoopConfig.block` is authoritative: `apply_loops` rebuilds the
   plan from it if the supplied plan disagrees, so a bare
   `apply_loops(model, default_plan, LoopConfig(block=...))` does the right thing.
   Defaults are unchanged (E2B legacy 15-29). FLOPs, `train/budget.py`, progress
   charts, the eval fingerprint, `RunResult` and `run.json` now carry/use the actual
   block. Static (preallocated producer-only) and dynamic cache generation were
   tested on a non-default block.

2. **`gch` bridge.** `x = h_pre + g * (rms(h_pre) * RMSNorm(h_prev))`, RMS over the
   last dimension per token, fp32 internally, cast once. `loop_bridge_g` is fp32,
   zero-initialised, saved in `loop_state.safetensors`, and grouped with the bridge
   LR group (`--bridge-lr`, weight decay 0). `g=0` gives exact `N=2 == N=1` while
   `d(loss)/dg != 0`. Legacy `scalar` / `none` / `damped` untouched.

3. **Adapter policy.** `LoopConfig.adapter_policy` is `"always"` or `"loop_only"`
   (`None` = legacy always). LoRA modules consult a per-forward `ContextVar`
   installed through decoder-layer pre-hooks; the pass index travels as a layer
   kwarg, so gradient-checkpointing recomputation replays the same pass. Under
   `"loop_only"`: loop-block adapters active on passes >= 2, coda adapters active
   when N >= 2, prelude adapters inactive; `L(N=1)` is bit-identical even with
   non-zero adapters and `g`. PEFT `Linear`/`Embedding` forward is wrapped once
   process-wide and only disables adapters for the duration of one call (no weight
   copies, no merging). `lora_target_layers` selects `"language"` (default) or
   `"loop_block_coda"` (regex over the configured plan's loop+coda). Policy and
   target set are saved in `loop_config.json`, inherited from an adapter, rejected
   on explicit mismatch, and recorded in `RunResult` / `run.json`.

4. **Calibrated NLL.** New `gemmaloop.eval.calibrated_nll`: per-token NLL from
   synthetic logits, bounded golden-section temperature fit, and a model path that
   walks PEFT/CausalLM/ConditionalGeneration wrappers, sets/restores N explicitly,
   and scores through a chunked lean head (token+vocab chunks; no full logits).
   Reports raw NLL, NLL at the fitted T, top-1 accuracy, and all per-position
   buckets (`<=1k`, `1-2k`, `2-4k`, `4-8k`, plus `>8k`). Temperature is applied
   after Gemma's final logit softcap. Works for looped and unlooped models at a
   given N.

5. **`scripts/pilot_probe.py`.** Forward sweep over blocks x bridges: per-layer
   `cos(out,in)`, RMS in/out/ratio, untrained `KL(N=1||N=2)`, calibrated Delta NLL
   (T fitted per N arm on alternating calibration rows, reported on the test rows),
   FLOPs share per pass, and the full `layer_scalar` vector. Writes JSON + markdown.
   `--tiny` is the CPU fp32 smoke path; the default path uses `load_model` and is
   intended for the GPU later. The GPU path was **not** run.

6. **Docs.** Added "layer_scalar and residual-stream geometry" to
   `docs/superpowers/references/gemma-loop/01-gemma4-architecture.md` with the E2B
   scalar table, per-layer cosine ranges, `cos(h14,h29)=-0.01`,
   `RMS(h29)/RMS(h14)=1.5-1.8`, block input/output cosines, and the citation to
   `methodology-review-A-architecture-loops.md` (Q1/Q2).

## Tests

Command (run for the final tree):

```
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
```

Result:

```
706 passed, 22 deselected, 14 warnings in 131.34s (0:02:11)
```

New coverage: block validation (consumer-only, non-empty coda), N=1 bit-identity
for every consumer block x every bridge (including non-unit `layer_scalar` and
non-zero g), `loop_only` N=1 identity after random non-zero LoRA/g, gch init
identity + non-zero gradient, checkpoint-recompute gradient equality, static and
dynamic cached vs uncached generation on a non-default block, PEFT save/load,
calibrated NLL on synthetic logits and on tiny looped/unlooped models, per-bucket
aggregation, the lean chunked head vs full logits, FLOPs per block, RunResult /
fingerprint metadata, the bridge optimizer group, and a CPU pilot-probe smoke run.

Pilot smoke command (also covered by `tests/test_pilot_probe.py`):

```
CUDA_VISIBLE_DEVICES=-1 uv run python scripts/pilot_probe.py --tiny \
  --rows <jsonl> --blocks 5-7,6-10 --bridges none,scalar,gch,damped \
  --n-loops 2 --max-rows 4 --out-json <json> --out-md <md>
```

## Design ambiguities resolved

- **Where the block lives.** `make_layer_plan` is the geometry builder, but
  `LoopConfig.block` is the request that `apply_loops` enforces. Adapter
  `loop_config.json` stores the block and wins by default; a conflicting explicit
  eval block is an error rather than a silent substitution.
- **Prelude with a late block.** A block starting at layer 16 makes the prelude
  0-15 (producers plus consumer 15). `_validate_plan` now checks
  `prelude == range(0, loop.start)` and `coda == range(loop.stop, num_layers)`
  instead of forcing the prelude to end at `first_shared`.
- **Policy inheritance.** `None` means "legacy/unset"; `save_loop_state` writes the
  concrete `"always"`/`"language"`, `load_model` inherits from the checkpoint and
  rejects an explicit conflicting CLI value. `run.json`/`RunResult` record the
  resolved values.
- **LoRA pass context.** A layer kwarg (`_gemmaloop_pass`, `_gemmaloop_n_loops`)
  is popped by a `with_kwargs` pre-hook and feeds a `ContextVar`; this survives
  checkpoint replay because the kwarg is captured by the checkpointed partial. The
  PEFT patch is process-wide but inert outside a looped forward.
- **gch eps / RMS.** Per-token RMS over the last dimension with `eps=1e-6`, no
  learned RMSNorm weight, fp32 arithmetic and one cast back to the residual dtype.
- **Calibration position buckets.** Target positions are 1-based token indices;
  a token belongs to the first bucket with `position < bound` (so 1024 is in
  `1-2k` and 4096 in `4-8k`). `>8k` is included so long contexts are never
  silently dropped.
- **Temperature on model logits.** T is applied after the final logit softcap
  (`softmax(softcap(hW^T)/T)`), mirroring the synthetic-logit helper.

## Concerns / follow-ups

- The pilot probe's real-checkpoint/GPU path was not executed (per brief). Only the
  `--tiny` CPU smoke path and its unit test were run.
- The PEFT policy patch targets `peft.tuners.lora.layer.Linear` and `Embedding`
  (the modules used by this project's Linear LoRA). Other PEFT variants such as
  `ParamWrapper`/DoRA are not routed through it; merging adapters is also
  incompatible with `loop_only` (merged weights cannot be switched per pass).
- `RunResult` duplicate detection still keys on
  `(variant, size, n_loops, bench, seed)` and does not include the block. Two runs
  with different blocks at the same N would be treated as duplicates; the
  fingerprint itself does include the block, so partial files cannot cross.
- Temperature fitting rescans the vocabulary in chunks per golden-section step on
  a cached calibration split. This keeps peak memory lean but is intentionally
  bounded and should be run on a small calibration split.

## Fix round 1

Review: `.superpowers/sdd/2026-11-03-gemma-loop/task-arch-review.md`. All three
Important items fixed, plus the cheap Minors.

1. **Merge guard for `loop_only`.** `assert_adapter_merge_allowed()` reads the
   adapter's `loop_config.json` and raises a clear `ValueError` before any
   `merge_and_unload`. It is called by `load.py::_apply_adapters` for every
   adapter that would be merged and by `scripts/train.py::_merge_init_adapter`
   for every entry of an `--init-adapter` chain. Tests cover both paths.
2. **Block-aware result identity and pooling.** `RunResult.geometry_key()` /
   `result_identity()` include block, bridge, adapter policy and LoRA target
   set. Duplicate detection in `prereg.py`, `progress_charts.py` and
   `three_column.py` uses the full identity; `check_criteria` rejects mixed
   geometries for one selector; progress charts, before/after bars and the
   summary table group by geometry instead of pooling blocks; FLOPs charts build
   a plan per geometry and use each run's own block. Three-column accepts
   geometry filters and refuses ambiguous pools. Tests cover duplicate keys,
   mixed-geometry rejection and per-block FLOPs x-values.
3. **Block-aware FLOPs budgets.** `scripts/train.py` resolves `--block`
   independently of the arm; O-H-ext rebuilds the plan from that block before
   `flops_matched_tokens` (and records it in `run.json`).
   `scripts/phase1_plan.py` accepts `loop_block`/`--block` for `ext_tokens`.
   Tests pin 25-29 and 16-18 budgets (and the E2B N2/N1 ratios).

Minors fixed: pilot FLOPs share includes the model-level constant; `--tiny`
uses tiny valid default blocks; the probe KL path is vocabulary-chunked (no full
logits); `run_bench` resolves adapter geometry before writing the fingerprint;
calibrated-NLL positions/`position_ids` are derived from `attention_mask`; the
checkpointing test asserts it is actually active; extra tests cover
heldout/calibrated/static-cache `loop_only`, multi-block pilot sweeps,
`_resolve_loop_config` block/policy inheritance and conflicts.

Remaining (not fixed): the PEFT policy patch still covers only
`Linear`/`Embedding` variants (not `ParamWrapper`/DoRA); `RunResult` duplicate
detection now includes geometry but the summary rows still use a compact
"legacy"/geometry label rather than a dedicated block column.

## Fix round 2

Review: `.superpowers/sdd/2026-11-03-gemma-loop/task-arch-review-r2.md`.
Both Important items fixed, plus the cheap Minors and the new prereg ruling (3).

1. **Block/geometry in every generated command.** `phase1_plan.build_plan` accepts
   `loop_block`, `bridge`, `adapter_policy` and `lora_target_layers` (CLI
   `--block/--bridge/--adapter-policy/--lora-target-layers`). Every generated
   `scripts/train.py` argv now carries `--block`, `--bridge`, `--adapter-policy`
   and `--lora-target-layers`; every eval argv carries block/policy/target and
   the configured bridge for L/L-plevec; the audit argv carries `--block`.
   The plan test builds a real E2B plan for 25-29 and asserts the full argv.
2. **Dtype-safe chunked logits.** `pilot_probe._kl_from_hidden` casts its hidden
   states to fp32 and explicitly casts the LM-head weight tile to the same dtype
   before the matmul. `calibrated_nll._chunked_nll_top1` now casts both the hidden
   chunk and the weight tile to fp32 explicitly. Tests cover fp32 hidden + bf16
   weight (the old crash), a tiny bf16 model on CPU, and a bf16 `score_model` run.
3. **`loop_block` LoRA target set (prereg ruling 3).** Added `"loop_block"`:
   only the repeated block's `q/k/v/o` (where present) and `gate/up/down`
   modules. With `loop_only`, those adapters activate on passes >= 2; unlooped
   O-H installs no policy hooks and therefore runs the same modules on its single
   pass. Parameter-count tests pin the per-block counts and all LoRA parameters to
   loop layers; N=1 bit-identity is tested with non-zero adapters/g. The `gch`
   bridge is documented and tested to act only on loop passes >= 2 and never on
   the coda (hook test asserts coda input == last loop output exactly).

Cheap Minors: O/O-H `run.json` now stores `loop_block: null` (only O-H-ext gets
the budget block); the before/after title recognises geometry-labelled L bars;
`_missing_lines` checks N coverage per geometry; the phase1 argv test and
bf16 probe tests close the gaps from the review. Full CPU suite: 728 passed.

## Fix round 3

Review: `.superpowers/sdd/2026-11-03-gemma-loop/task-arch-review-r3.md`.
Important item fixed, plus the cheap Minors.

- **Unlooped arms honour `--lora-target-layers`.** `scripts/train.py` now resolves
  the target set for every arm (`resolve_lora_target_layers`): looped arms use
  their `LoopConfig`, unlooped O-H/O-H-ext fall back to the CLI value. The
  resolved layer list (`resolve_lora_layers`) is computed from the configured
  block plan, so `--block 25-29 --lora-target-layers loop_block` on O-H selects
  layers 25-29, exactly the L arm's set. The same resolved target/layers are
  passed to `default_lora_config` and written to `run.json` for every arm
  (`lora_target_layers`, `lora_layers`); O/O-H keep `loop_block: null`.
- **Tests.** A 35-layer tiny model checks that O-H with `loop_block` 25-29 and L
  with 25-29 have identical LoRA parameter names/count and all are in 25-29;
  the run.json metadata test pins the target set/layer list for an unlooped arm;
  the plan argv test covers the generated commands.
- **Cheap Minors.** The generated audit argv now carries the full geometry
  (block/bridge/policy/targets), matching the executed plan. Looped `run.json`
  keeps the resolved plan block; null is reserved for unlooped O/O-H.

Full CPU suite: 730 passed.
