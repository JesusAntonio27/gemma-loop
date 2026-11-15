# prereg-v1 draft report

Status: **DONE_WITH_CONCERNS** (draft only; no tag created).

## Scope

Rewrote the pre-registration from the old single-SFT design to the two-stage
hybrid (Stage 1 heal + Stage 2 Dr. GRPO RLVR) on E2B, seed 0, with Part 2 as
exploratory rules. No training/eval code was changed beyond the pydantic schema
in `src/gemmaloop/prereg.py`. The git tag `prereg-v1` was **not** created.

## Files changed

- `prereg/prereg.yaml` — rewritten machine-readable prereg.
- `prereg/PREREGISTRATION.md` — rewritten English human-readable prereg, with
  changelog at the top.
- `src/gemmaloop/prereg.py` — added optional two-stage schema sections,
  validators and `MdeRow`; legacy S1-S5 schema retained for compatibility.
- `tests/test_prereg.py` — added schema/committed-YAML tests and one seed-guard
  test adjustment for the seed-0-only prereg.
- `tests/report/test_three_column.py` — one synthetic test now also replaces
  `rollout_n: 2` when it changes `primary_n` to 3, because the new schema
  requires `part1.rollout_n == stage2.rollout_n == primary_n`.

## Tests

Command:

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
```

Result:

```text
641 passed, 22 deselected, 14 warnings in 127.51s (0:02:07)
```

Targeted prereg/report suite:

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/test_prereg.py tests/test_phase1_plan.py tests/report/test_three_column.py tests/report/test_figures.py -q
88 passed, 14 warnings in 16.21s; final tests/test_prereg.py run: 40 passed in 6.12s
```

The new tests cover loading the committed YAML, the two-stage sections, the
architecture/N-distribution invariants, the E1-E4 sequence, the pilot numeric
thresholds, the MDE table, and rejection of inconsistent new sections.

## Choices made by the drafter

Every underspecified value was made concrete. These are also in
`prereg/prereg.yaml` under `drafter_choices` and in section 11 of
`PREREGISTRATION.md`.

1. **N distribution `[0.0, 0.90, 0.10, 0.0]` (N in {2,3}).** Adapters are
   N >= 2-only, so N=1 batches are wasted; N=4 is extrapolation only. This is
   the brief's "N in {2,3}" branch.
2. **Stage 1 `beta = 0.1`, direction `KL(parent || student)`.** Follows the
   cited 2608.18171 direction; the brief fixes beta but not its value.
3. **Stage 1 top-k = 64 and float16 logit storage.** Concrete offline storage
   and approximation level.
4. **Stage 1 budget default 12M, hard maximum 16M, dose-response points
   2/4/8/12/16M; pilot P2 short heals 3M.** Reconciles the design doc's 12M
   default with the brief's "pilot dose-response within a pre-set maximum" and
   the brief's ~3M pilot heals.
5. **LoRA rank 16, alpha 32, LoRA LR 2e-4; bridge LR 3e-3; loop_ple LR 1e-3;
   legacy scalar `w0 = 0.99`.** From the existing rulings and review A; values
   were not all fixed by the brief.
6. **Stage 2 G=8, 16 prompts/step, 100 steps, 8192 cap, LoRA LR 2e-5, bridge LR
   1e-3, `beta_KL = 0.0`, TIS cap 2.0.** From the two-stage design doc; the
   brief says "per the design doc" without repeating every number.
7. **S2 primary uses full `release_v6` = 1,055 items x 1 sample.** The brief
   says ">= 728"; 1,055 is the full release and gives the best registered MDE.
   The 182 post-cut items are the pre-specified subgroup.
8. **E1 TOST margins +/-0.01 nats and +/-3 pp, 90 % CI, compared with O-H.**
   The brief asks to state margins; these are the review-B/red-team values.
9. **Bootstrap 10,000 iterations, seed 0; item bootstrap at seed 0; crossed
   pigeonhole bootstrap when more than one seed exists.** Review B's structure,
   made concrete.
10. **Budget grid `B in {1,2,4,8,16,32}k`, AUC over log2 B; 1.52x for 15-29 and
    the measured ratio for the chosen block.** The brief gives the grid and the
    1.52x example; the chosen-block fallback is explicit.
11. **Pilot P2 uses a pre-listed 6-arm matrix: three blocks with the zero-init
    bridge and E4B; 15-29 with the legacy scalar and E4B; 25-29 with the
    zero-init bridge and 12B QAT q4_0; O-H.** Six 3M-token heals fit the 6-8
    local GPU-hour envelope. Selection: largest `delta_loop` with 95 % CI
    excluding 0 and `L(N=2) - O-H >= -0.02` nats; ties within 0.003 nats go to
    lower FLOPs, zero-init bridge, then smaller block. The brief asks for a
    written rule; the numeric thresholds come from the red-team pilot rule and
    review A.
12. **Pilot P3 thresholds: dev pass@1 gain >= 2 SE, truncation < 10 %, no
    hacking flags; otherwise RFT on verified self-traces.** The brief names the
    RFT fallback; the numeric gate comes from the two-stage design doc.
13. **Pilot data: 200 P1 rows, 3M P2 tokens, 400 dev rows, 1,000 held-out test
    rows, 200 long traces, 500 temperature-fit rows.** Concrete split sizes.
14. **Stage 0 pool: 2,000 code, 2,000 math, 1,000 chat prompts, 4 samples each,
    about 64M generated tokens.** From the two-stage design doc.
15. **Decontamination: 13-gram plus Indel >= 0.75 against every eval and the
    creative briefs; manifest SHA frozen before training.** From the brief and
    the data audit.
16. **`max_length = 8192`, filter mode; OT3 requires a closed thought channel
    (`<channel|>`) and a non-empty answer; reasoning share >= 75 %.** From the
    data-length audit and training-perf report.
17. **Compute tiers local / single cloud GPU / TPU, availability only, recorded
    as a covariate; O-H and L of a seed share tier and engine.** From the brief
    and progress ruling 316.
18. **R and R-damped are diagnostic only, scored on LCB seed 0.** The brief says
    "diagnostic only" without a bench; LCB seed 0 is the cheapest registered
    comparison.
19. **AIME and spatial are O-only; GPQA, MMLU-Pro and vision are
    secondary/estimation only.** The brief lists them this way but not the
    sample counts; the old registered sample counts are retained.
20. **O-H-RLext is optional, seed 0 only, only if S2 is positive, 1.5x Stage 2
    steps on fresh prompts.** From the two-stage design doc.
21. **Part 1 is E2B seed 0 only; E4B and extra seeds deferred; extra seeds run
    regardless of the seed-0 result.** From the brief.
22. **Google comparison is a "reproduction check" (forest plot), never
    "validation"; `lcb_google` is harness-only.** From review B.
23. **Creative evaluation is Part 2/exploratory; S3 is not a Part 1 confirmatory
    endpoint.** From review B and the red-team audit.
24. **Fixed sequence E1 -> E2 -> E3 -> E4 at alpha 0.05 each, stopping at the
    first failure; N=3/4 exploratory results are Holm-corrected.** From review B
    and the brief.
25. **Legacy `train_tokens.e4b = 0`.** E4B is deferred; the field is kept only
    because the old trainer/planner schema reads it. This is a deliberate
    placeholder, not a planned E4B run.

## Conflicts between sources resolved

The brief is authoritative over reports, and AUTHOR lines over Ruling lines
over reports. The main conflicts were:

1. **Old single-SFT prereg vs two-stage hybrid.** The old `prereg.yaml` and
   `PREREGISTRATION.md` described one long SFT with S1-S5; the brief and
   AUTHOR/Ruling 2026-11-04 replace it with the two-stage hybrid. The old file
   is rewritten; legacy S1-S5 fields remain only because `check_criteria` reads
   them.
2. **Stage 1 objective.** `twostage-vs-current-design.md` recommended self-KL
   from the original model. AUTHOR ruling 357/361 supersedes it with forward KL
   from E4B or 12B QAT plus beta * KL to the parent. The latter is encoded.
3. **Stage 1 budget.** The design doc fixed 12M; the brief says the pilot
   dose-response fixes it within a pre-set maximum. Encoded as default 12M,
   maximum 16M, dose points 2/4/8/12/16M.
4. **Pilot heal size.** The design doc's pilot used 4M tokens; the brief says
   ~3M. Encoded 3M for P2; the separate dose-response sets the main budget.
5. **P3 fallback.** The design doc listed OPD then RFT; the brief names RFT on
   verified self-traces. Encoded RFT.
6. **S2 endpoint.** The old prereg used 182 post-cut items x 4 samples; review B
   and the brief require the full release (>= 728) x 1 sample with the post-cut
   subgroup. Encoded full 1,055 x 1.
7. **S2 token clause.** The old S2 had a "same accuracy with fewer tokens"
   branch; review B says drop it and use budget curves. The new E2 is a pure
   superiority test and E4 is the budget-curve AUC.
8. **S1 rule.** The old S1 required a paired CI including 0 on every bench;
   review B/red-team require TOST. Encoded E1 as a unit test plus TOST on O-H
   with +/-0.01 nats and +/-3 pp.
9. **Bootstrap.** The old design used an item bootstrap over seed-averaged
   scores; review B requires a crossed pigeonhole bootstrap when > 1 seed.
   Encoded both, with the crossed version conditional on > 1 seed.
10. **Training data.** The old design used R1/QwQ teacher traces; review B and
    the two-stage design require on-policy O samples. Encoded Stage 0 sampling
    of O and Stage 2 RLVR on verified prompts.
11. **Seeds.** The old prereg had E2B seeds [0,1,2]; the brief says seed 0 only.
    Encoded seed 0, with the extra-seed policy declared now.
12. **Size.** The old prereg had E4B seed 0; the brief says E2B only. E4B is
    deferred; `train_tokens.e4b = 0` and empty E4B seed lists.
13. **O-H-ext and L-plevec.** The old design had both; the brief removes them
    and allows optional O-H-RLext only if S2 is positive. Encoded accordingly.
14. **Creative S3.** The old prereg made S3 confirmatory; review B/red-team
    demote it. Creative is Part 2/exploratory only.
15. **Google comparison wording.** The old text said "validated"; review B
    requires "reproduction check". Reworded everywhere.
16. **FLOPs ratio.** Review A measured 1.43x for 15-29, while the design doc
    uses 1.52x. The legacy 1.52x is kept only as the 15-29 default; the chosen
    block uses the measured ratio from `gemmaloop.model.flops`.
17. **Pilot P1 block list.** Review A's P1 includes 15-19; the brief's candidate
    set is {25-29, 16-18, 15-29}. 15-19 is kept in P1 as descriptive only, not
    as a selectable candidate.
18. **AIME/spatial/vision scope.** The old prereg treated them as part of the
    main matrix; the brief makes AIME and spatial O-only and GPQA/MMLU-Pro/
    vision secondary. Encoded in `endpoints.o_only` and `endpoints.secondary`.

## Concerns / implementation gaps (not fixed, by brief)

1. **Eval code still implements the legacy sets.** `src/gemmaloop/eval/runner.py`
   maps `lcb` to `post_cut=True` (182 items) and `lcb_google` to the full
   release. The new prereg registers `lcb` as the full 1,055-item release with
   a post-cut subgroup. A later eval task must honor `benches.lcb.limit` and
   store the subgroup; this draft did not change eval code, per the brief.
2. **`check_criteria` still computes legacy S1-S5.** The new confirmatory family
   E1-E4 is registered in YAML/markdown but not implemented in
   `check_criteria`. A later task must implement the fixed sequence, E1 unit
   test/TOST, E3 and E4. Tests here cover schema validation only.
3. **Stage 1/Stage 2 trainers are not implemented in this draft.** The prereg
   specifies them; code tasks follow after the tag.
4. **`train_tokens.e4b = 0`** is a deliberate deferred placeholder. The legacy
   `phase1_plan` would reject an E4B run with an empty seed list; that is
   intentional because Part 1 is E2B only.
5. **The tag was not created** and no training or trained-arm evaluation was
   run.

## Review checklist

- Confirm the 25 drafter choices, especially the N distribution, the Stage 1
  budget maximum and the pilot thresholds.
- Confirm that using the full 1,055-item LCB (rather than the 728-item minimum)
  is the intended cost/power trade-off.
- Confirm that E1's TOST is intentionally trivial for O-H because O-H is a
  no-op by construction, and that this is acceptable as an implementation-drift
  guard.
- Confirm the implementation gaps above are scheduled before the pilot.

---

# Revision round 1 (native review `prereg-v1-draft-review.md`)

Status: **DONE_WITH_CONCERNS** (draft only; no tag). This section supersedes the
"Concerns / implementation gaps" and "Review checklist" above wherever they
conflict: in particular, O-H is no longer a no-op, E1 is no longer a gated TOST,
`train_tokens` is retired, and E3 is the total effect versus O.

## What changed per review item

| Item | Resolution |
|---|---|
| C1 | O-H runs the real Stage 1 with the same teacher, data and 12M budget. Removed the no-op claim, the KL < 1e-4 check and "Stage 1 charged to L only"; both arms pay Stage 1. |
| C2 | The pilot gate is scored on O's **verified-correct traces of held-out problems**, T is fitted per (arm, N) on a fit split, and the bootstrap is by problem cluster. Raw O samples are no longer the gate; the KL-gap fraction is secondary. |
| C3 | E1 is a **precondition with no alpha**: bit-exact L(N=1) ≡ O on every engine plus descriptive heal estimates. The gated sequence is E2 → E3 → E4, each requiring a 95 % CI lower bound > 0. |
| C4 | E3 is relabelled "L(N=2) vs O — total effect vs the original"; the "cleanest loop contrast" and "no training-seed variance" claims are removed. |
| I1 | Selection/STOP is deterministic: lower bound > 0, signed `delta_arm = calNLL(O-H) - calNLL(L,N=2) >= -0.02` (point estimate), STOP when the eligible set is empty, problem-cluster bootstrap, T per (arm, N), KL(O‖L N=1) = 0 by construction. |
| I2 | P2 has 7 arms and adds O-H+12B; the teacher is chosen by the mean gain of L and O-H, with an E4B tie-break within 0.003 nats. |
| I3 | Stage 1 budget is fixed at 12M; the loss is logged every 2M tokens descriptively, with no dose-response decision. |
| I4 | P3 uses 200 held-out math problems x 4, a 4k cap on prompts with O median <= 2.5k, paired SE >= 2, truncation < 10 %; the RFT fallback recipe is registered now. |
| I5 | One deterministic split table by `sha256(problem_id) % 100`, disjoint by problem id and disjoint from Stage 1/2 data. |
| I6 | Extra seeds [1, 2] are registered; the confirmatory verdict is seed 0 only; the crossed bootstrap is robustness only. |
| I7 | The sequence rule is consistently "95 % CI lower bound > 0". |
| I8 | E4 registers r = 1.428/1.143/1.084, B = 1024·2^j (j=0..5), total-token correctness at B, trapezoid AUC / 5, and the 32,768 cap for O-H. |
| I9 | Compute registers a local HF-refill engine and a cloud vLLM engine, an honest hours table per stage, and an evaluation priority order. |
| I10 | Recipe gaps are filled: optimizer, tokens/step, schedule, clip, loss tokens, top-k tail, live parent KL, teacher runtime, Stage 2 num_iterations/normaliser/grader timeout, LoRA target = loop block only. |
| I11 | `criteria` and `train_tokens` are rejected by the schema; `check_criteria` refuses S1-S5; `phase1_plan` refuses the legacy plan; bench roles are explicit with no default to confirmatory. |
| I12 | Data provenance is named: TACO/APPS/CodeContests pre-2023-05, OMR/DAPO-17k/DeepScaleR, the existing decontaminated chat source; "synthetic" is dropped; the Stage-0 sample filter replaces the OT3 rule; the < 1,600-prompt rule is registered. |
| I13 | Pins are registered: E2B snapshot, template SHA, teacher revisions and tokenizer hashes, lcb_runner, math-verify, manifest SHA; the addendum procedure is written; the freeze point is the tag, before any trained-arm evaluation. |
| Minor | Fixed the dropped "calibrated"; NLL is secondary; legacy S1/S2/S5 names removed; Holm family specified; R-damped w = 0.99 and its item set specified; DiD has post-Stage-1 LCB runs; `lcb_google` is a virtual view of O's full 1,055 run; Part 2 dev composite and milestone cap defined; automatic checks labelled as the controller's recommendation; multi-GPU tier added; an MDE recomputation test was added; "S2 positive" became "E2 passes"; decision-log export added. |

## Required additions delivered

- **Retired legacy fields.** `criteria` and `train_tokens` are absent from
  `Prereg` and rejected by `extra="forbid"`. `check_criteria` raises a clear
  retirement error. `scripts/phase1_plan.py` and `scripts/check_criteria.py`
  refuse the legacy path. `tests/test_prereg.py` and
  `tests/test_phase1_plan.py` were rewritten for the retirement.
- **Bench roles.** `BenchSpec.role` is required and accepts only
  `confirmatory`, `secondary`, `o_only` or `harness`; there is no default.
  `lcb_google` is a virtual harness view (`generated: false`,
  `source_bench: lcb`).
- **MDE recomputation test.** `test_mde_table_recomputes_from_stats_mde`
  recomputes every row with `gemmaloop.eval.stats.mde`.
- **Honest compute section.** The YAML/MD hours table uses the review's I9
  numbers: Stage 0 local 50-70 h refill (unmeasured) / 220-300 h padded, cloud
  ~$7-11; pilot 35-40 local h; Stage 1 ~10 h for L + O-H; Stage 2 L 83-124 h /
  O-H 64-96 h local, ~4 H100-h per arm-seed in the cloud; evaluation 85-105
  local h; total local seed 0 330-460 h.
- **Evaluation priority order.** E2/E3 on full LCB first, then E4 re-scoring,
  then R/R-damped, then secondary, then O-only.
- **Pins.** E2B snapshot `4adf157...`, template SHA, teacher revisions and
  tokenizer hashes, `lcb_runner`, `math-verify` and the manifest SHA are
  registered; every null is filled in the prereg-v1 addendum before the tag.
- **Decision-log export.** `docs/prereg-v1-decision-log.md` is written at the
  tag.
- **Pending author section.** The two pending items (E3 naming; Stage 0 compute
  tier) are at the top of `PREREGISTRATION.md` and in `pending_author` in the
  YAML, with the recommended defaults encoded.

## Tests

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
598 passed, 22 deselected, 14 warnings in 143.51s (0:02:23)
```

Targeted suites after the rewrite:

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/test_prereg.py tests/test_phase1_plan.py tests/train/test_train_e2e.py tests/report/test_three_column.py -q
36 passed, 14 warnings in 101.81s
```

## Remaining unresolved / implementation tasks

1. **E1-E4 analysis code is not implemented.** `check_criteria` is retired but
   the E1 precondition, E2-E4 fixed sequence, lower-bound rule, crossed
   bootstrap and E4 re-scoring/AUC must be implemented and tagged `analysis-v1`
   before any trained-arm evaluation (review implementation task 2).
2. **Eval runner still maps `lcb` to the 182 post-cut set.** The prereg
   registers the full 1,055-item release with a post-cut subgroup; the runner
   change is part of `analysis-v1` (review implementation task 1).
3. **Stage 1/Stage 2 trainers and the teacher-logit pipeline are not
   implemented in this draft.** They are code tasks after the tag.
4. **Pins are partly null.** Template SHA, teacher revisions/tokenizer hashes,
   `lcb_runner`, `math-verify` and the manifest SHA must be filled in the
   prereg-v1 addendum before the tag.
5. **The tag was not created** and no training or trained-arm evaluation was
   run.

---

# Revision round 2 (loop ladder and training-free disclosure)

Status: **DONE_WITH_CONCERNS** (draft only; no tag). The two new rulings at the
end of `progress.md` are encoded in the YAML and the MD.

## What changed

### 1. Loop ladder (descriptive/estimation only)

- Added `loop_ladder` to `prereg.yaml` and a new section 3.4 to
  `PREREGISTRATION.md`.
- The ladder is **descriptive/estimation only**, runs **after the tag**, and is
  **never used for pilot or selection**.
- Points: O (N=1); R (pure repeat) and R-damped on the **chosen block** at
  N=2,3,4; L after Stage 1 at N=2; L final at N=2,3,4; O-H after each stage as
  the control.
- R is the `none` bridge (`h_prev`). R-damped is the registered Euler rule
  `x <- (1 - 1/N) * x + (1/N) * block_out`, with weights 0.5 / 0.333 / 0.25 at
  N=2 / 3 / 4; the legacy scalar `w = 0.99` is a different bridge and is not
  used by R-damped.
- Items: LCB post-cut 182 x 1 + GPQA 198 x 1 for every ladder point; L final
  N=2 also gets the full confirmatory suite.
- `gch` at init with `loop_only` is O exactly, so the untrained looping
  measurement uses R/R-damped, not gch.
- Budget: +20-22 local GPU-hours, added to the compute hours table; the total
  local seed-0 envelope is now 350-482 h.
- The ladder supplies the post-Stage-1 measurements for the
  difference-in-differences; the DiD budget is registered here.
- Chart: accuracy vs N with O, R, R-damped, L-heal and L-final lines plus O-H
  references; `progress_charts` `acc_vs_n` extended with stage labels.

### 2. Training-free exploratory campaign before the tag (disclosure)

- Added `training_free_campaign` to `prereg.yaml` and section 0.1 to
  `PREREGISTRATION.md`.
- P1-lite: calibrated-NLL sweep per block x {R, R-damped} x N=1..4 on a
  **120-problem verified-trace corpus** (~1 h).
- Accuracy: O and R/R-damped at N=2 on the three candidate blocks (7 configs)
  on LCB post-cut 182 x 1 + GPQA 198 x 1 (~21-25 h), then N=3,4 on the block
  with the lowest calibrated NLL (~7 h).
- The results are **exploratory**, reported in full, and are **not inputs to
  block selection**.
- The candidate set {25-29, 16-18, 15-29} x bridges and the pilot
  selection/STOP rules are **frozen as of the controller's ruling time**,
  regardless of these results.
- The P1-lite corpus is **disjoint from every registered split**; it is drawn
  from a source decontaminated against all evals and excluded from Stage 0's
  pool.

### 3. Restored test coverage

- `tests/test_prereg.py` now has schema-level tests for every E1-E4 endpoint
  field: precondition fields, sequence fields and margins, primary, heal
  estimates, bootstrap settings, fixed-sequence rule, Holm, single-seed
  statement, Google comparison, secondary/o_only and priority order.
- Added the MDE recomputation test (kept from round 1) and new tests for the
  loop ladder (points, R-damped rule/weights, items, full-suite point, gch note,
  budget, chart, DiD) and the training-free disclosure (before-tag, exploratory,
  full reporting, no selection input, frozen candidates, 120-problem corpus,
  N=2/N=3-4 plan, corpus disjointness, budget, chart).
- Added validators for the new sections and tests that reject bad values.

## Tests

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
602 passed, 22 deselected, 14 warnings in 138.69s (0:02:18)
```

## Remaining unresolved (unchanged)

- E1-E4 analysis code and the eval runner's lcb 1,055 mapping remain the
  `analysis-v1` implementation task.
- Some pins are still null pending the prereg-v1 addendum.
- The tag was not created.

---

# Revision round 3 (re-review items (1)-(9), Stage 0 record, pins)

Status: **DONE_WITH_CONCERNS** (draft only; no tag). The native re-review items
(1)-(9) and the Stage 0 cloud-tier record are encoded in the YAML and MD.

## What changed

### Re-review items (1)-(9)

1. **Single pilot selection rule.** Teacher first by mean calibrated-NLL gain
   over O of `(25-29+gch+T)` and `(O-H+T)`, with an E4B tie-break within 0.003
   nats; then block and bridge among the E4B arms, each compared with the E4B
   O-H. Encoded as `teacher_selection_rule`, `teacher_gain_basis`,
   `block_bridge_rule` and `block_bridge_teacher`.
2. **One engine per stage.** Stage 0 = Modal stock vLLM v0.30.0 (data only);
   Stage 2 rollouts and all evaluation = the current local padded adaptive-batch
   HF engine (`generate.py`, static cache) for O, O-H and L; the refill engine
   is not registered for evaluation; N=1 bit-exactness is checked against stock
   HF.
3. **LoRA/bridge scope.** LoRA targets loop-block layers only
   (`target_option: loop_block`); the bridge acts only on loop passes >= 2, not
   on the coda.
4. **R-damped defined once.** Euler `damped_update` with weights 1/N
   (0.5/0.333/0.25 at N=2/3/4); the legacy scalar `w=0.99` is a separate bridge
   and is no longer mentioned in the R-damped definition.
5. **Loss/teacher numerics.** Stage 1 and RFT loss on assistant tokens only
   (thinking + answer); teachers loaded with bitsandbytes nf4 (E4B regular
   checkpoint, 12B QAT-unquantized); numerics declared.
6. **Pins filled now.** E2B template SHA, E4B/12B revisions, three tokenizer
   hashes, and `lcb_runner` commit are filled from the local cache / HF API /
   LCB checkout. `math-verify` is not a dependency and is pinned in
   `prereg-v1-data`. Only the data manifest SHA remains null.
7. **Pilot sub-buckets.** The pilot bucket 50-59 is split into disjoint P3
   (50-53), selection (54-56), T-fit (57-58) and P1 (59) sub-ranges; P3 RL
   prompts come from the Stage 0 train bucket 0-49.
8. **Disclosure.** Cites frozen-candidate commit
   `d34af811920e8394f8ce7b197818ddc4585e84e3` and timestamp
   `2026-11-04T22:19:00-07:00`; campaign results are reused only for the ladder
   with identical engine/harness, never for E3/DiD; Stage B accuracy is not
   launched until the author decides E3 (Stage A may run).
9. **Ladder/analysis.** Ladder budget recomputed with the reuse rule (+20-22 h
   without reuse, +11-13 h with reuse; total 341-473 with reuse, 350-482 worst
   case). Analysis-v1 obligations are registered: delete dead S1-S5 code,
   re-cover helper tests, duplicate key with block/bridge/policy, implement
   E1-E4 and the lcb 1,055 mapping.

### Stage 0 cloud tier record

- Provider: **Modal**; engine: stock **vLLM v0.30.0** (data only, no plugin).
- Limits: **$30 usage limit / $0 spend limit**.
- Fallback: finish locally on the registered HF engine; the **engine is
  recorded per row** and the split assignment is unaffected.
- Pending author section now contains only E3; the Stage 0 spend is approved.

### Pins

Filled now (CPU):

- E2B snapshot `4adf15748de9e1a0097b6274cee9c15836c41c37`.
- E2B chat-template SHA-256
  `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`.
- E4B revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`.
- 12B QAT q4_0 unquantized revision
  `b6ed86275a6a5735884e208bfed95b445a684ca2`.
- Tokenizer SHA-256: E2B `ab26b2...`, E4B `91d9cf...`, 12B `cc8d3a...`.
- `lcb_runner` commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- `math-verify`: not a dependency; pinned in `prereg-v1-data`.

Still null: **data manifest SHA-256** (`pins.manifest_sha256` and
`data.manifest_sha256`), to be filled in the post-tag `prereg-v1-data`
addendum before the pilot.

## Tests

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
603 passed, 22 deselected, 14 warnings in 124.81s (0:02:04)
```

New coverage: pins and data-addendum fields, Stage 0 provider/limits/engine
record, LoRA target option, bridge scope, assistant-token loss, teacher
numerics, pilot sub-buckets, P3 prompt bucket, the two-step selection rule and
the analysis-v1 obligations.

## Remaining unresolved

- Data manifest SHA-256 (post-tag `prereg-v1-data` addendum).
- `analysis-v1` implementation: E1-E4 code, eval runner lcb 1,055 mapping,
  dead S1-S5 deletion, helper-test re-coverage and the block/bridge/policy
  duplicate key.
- The tag was not created.

---

# Revision round 4 (selection sync, freeze commit, conditional engine, hours)

Status: **DONE_WITH_CONCERNS** (draft only; no tag).

## What changed

1. **Selection text sync.** `PREREGISTRATION.md` §3.3 now reproduces the YAML
   two-step rule exactly: teacher first by mean calibrated-NLL gain over O of
   `(25-29+gch+T)` and `(O-H+T)` (E4B tie-break within 0.003 nats); then block
   and bridge among the E4B arms, each compared with the E4B O-H, with the YAML
   tie-breaks.
2. **Freeze statement.** The candidate set and selection/STOP rules are frozen
   at commit `84c5da0f1778cb7446d20bed0cd428c5799ea1b1`, timestamp
   `2026-11-04T22:39:04-07:00`. No campaign output existed at that time
   (`no_output_at_freeze: true`).
3. **Conditional evaluation engine.** The refill engine (task/t5c, pinned commit
   `a81282e`) is used iff, before `analysis-v1`, it passes the 5c fixes
   (per-request seed convention identical to padded, per-request persistence)
   and the registered gate: greedy token-exact vs padded on **32 prompts at N=1
   and N=2**, and identical per-request sampled tokens on **16 prompts**.
   Otherwise the padded adaptive-batch engine is used. The same engine is used
   for O, O-H and L wherever compared. Encoded in `compute.local_engine`,
   `compute.eval_engine`, `compute.eval_engine_gate`,
   `compute.refill_engine_branch` and `compute.refill_engine_commit`.
4. **Hours re-forecast for both engines.**
   - Stage 0 is on **Modal** (not local), stock vLLM v0.30.0, $30 usage / $0
     spend; the fallback is the **registered evaluation engine** (no refill
     citation in the Stage 0 fallback row), with the engine recorded per row and
     the split assignment unaffected.
   - Padded engine: evaluation 273-474 h (60-75M tokens at 44-61 tok/s with the
     32k cache), Stage 2 L 83-124 / O-H 64-96 h, total **500-779 h**.
   - Refill engine (estimate, replaced by the gate measurement): evaluation
     48-83 h, Stage 2 L 36-74 / O-H 27-58 h, total **191-300 h**.
   - Ladder: **≈35 h without reuse** (less if campaign stage B is reused).
   - The compute table now has explicit `padded_hours` and `refill_hours` per
     row.

## Tests

```text
CUDA_VISIBLE_DEVICES=-1 uv run pytest -m "not gpu and not slow" -q
604 passed, 22 deselected, 14 warnings in 123.31s (0:02:03)
```

New coverage: freeze commit/timestamp/no-output, conditional engine fields and
gate, per-row padded/refill hours, Stage 0 Modal row, and the ladder ≈35 h
budget.

## Remaining unresolved

- Data manifest SHA-256 (post-tag `prereg-v1-data` addendum).
- `analysis-v1` implementation and the engine gate itself.
- The tag was not created.

---

# Revision round 8b (Stage-A-exposed pre-tag technical amendment)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8b-report.md`. Stage A was exposed, so every item is a pre-tag
technical amendment and **none uses a trained-arm outcome**.

## What changed

1. **Heal monitor (R8B-01).** The MONITOR corpus is the 64 train-bucket
   (0-49) problems with the largest `sha256(seed:problem_id)` keys, one sample
   each, excluded from every training subset. Monitor teacher shards use the
   arm's frozen runtime and `max_length`; the WARN/ABORT rules are registered,
   ABORT exits 76 terminal, resume needs `--ack-abort`, a probe exception is a
   WARN row and two consecutive probe failures are an ABORT. `reserved_ids.json`
   is a problem-id map that never treats its own monitor key as an overlap.
2. **P1 count (R8B-02).** P1 is 50 bucket-59 problems x 4 samples = 200 rows;
   `pilot.p1.rows` counts completions, not problems.
3. **Pilot stages (R8B-03).** Teacher stage on 25-29+gch and O-H[25-29];
   block/bridge stage on the E4B arms against the block-matched E4B O-H; the
   chosen recipe is then trained with the selected teacher.
4. **Math grading (R8B-04).** Correct iff math-verify 0.9.0 (internal timeouts
   off, 10 s outer subprocess timeout) OR the AIME/integer normaliser;
   `decided_by` recorded. Reason: Windows multiprocessing timeout and the
   `1,234` disagreement.
5. **Part-1 pool (R8B-05).** 5,000 problems (2,000 code, 1,000 chat, 2,000
   math), no creative briefs, manifest sha256 `70053b90...`; APPS parquet pin,
   TACO parquet files and CodeContests Python-only sources.
6. **Engine (R8B-06).** Refill track closed; padded HF engine registered for
   all evaluation and Stage 2; Stage-0-lite first shard 100 requests in ~2,600
   s, ~195 tok/s, 52/100 kept, ~17-20 h.
7. **Teacher corpus (R8B-07).** Rows from O's Stage 0 local samples (train
   bucket, verified, 4/8/4/4, 20-row screen then 100 in seeded hash order),
   disjoint only from pilot evaluation/T-fit/MONITOR/confirmatory; the 120-row
   P1-lite corpus is the fallback (85 rows kept).
8. **Code verification (R8B-08).** Verified pilot targets require the Docker
   sandbox before any pilot evaluation; the unsandboxed local mode is never
   used for model-generated code; without Docker they are math-only.
9. **External APIs (R8B-09).** DeepSeek official API, deepseek-flash, off-peak
   with margins, paid pilot <= 20, waste stop >= 3 of the last 20, one
   fingerprint, phase caps ($5 phase 1; ~$40-50 project; $71 total), author
   approval per phase; NVIDIA NIM trial unusable; OpenCode endpoints never used
   for training data or judgments.
10. **Modal (R8B-10).** Probe history and the required money guards recorded;
    Modal is optional for Part 1 because local sampling is the registered path.
11. **Pins (R8B-11).** Merged commits `217b939`, `2eef68b`, `dfea02a`,
    `c840e9b`; explicit placeholder pins for `task/healpilot`,
    `task/p1corpus` and `fix/modal-imports`, filled in round 8c before the tag.

## Verification

`CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/test_prereg.py -q`:
**48 passed**. Full-suite result is in the round-8b report. The tag was not
created.

---

# Revision round 8c (final pre-tag reconciliation)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8c-report.md`. Stage A was exposed, so every item is a pre-tag
technical amendment and **none uses a trained-arm outcome**.

## What changed

1. **Padded sole operative engine (R8C-01).** §2.3/§4.3 now register the
   padded adaptive-batch HF engine unconditionally for Stage 2 and all
   evaluation; `engine_gate`/`part_b` are INACTIVE and have no execution
   authority. The smoke record is corrected: four attempts, v6 developed and
   rejected in review (`8f36aca`), repaired smoke never run, v6 unmerged. The
   operative padded source is pinned by blob hash at `54cab07`.
2. **Realized corpus and shortfall (R8C-02).** Hash counts are nominal; the
   realized counts are frozen (train 2,488 / 2,424 after MONITOR; P1 58 → 50;
   T-fit 86; selection 152; NLL fit/test 90/396). P1 is 200 attempts with a
   deterministic truncation and retained-row rule; fit/selection IDs are
   math-only until Docker code verification, then frozen; fail-closed minima
   and no-borrowing rules are registered; the dedicated D1 60-69 post-tag
   sampling backend/cap/schedule is named and its IDs are a `prereg-v1-data`
   obligation.
3. **Monitor abort/resume and missing arms (R8C-03).** ABORT writes no
   checkpoint; `--ack-abort` is required on every resume path (without it a
   registered resume over an abort record exits 2; with it, resume from the
   last COMPLETE checkpoint or archive the partial run dir and restart from
   zero); technical repair is separated from a recipe-changing new version;
   probe cadence/N/calibration are fixed and the exact token points plus the
   generation-sanity length exception are deferred to 8d; aborted/incomplete
   arms are reported, never replaced.
4. **Temperature fit and qualification timing (R8C-04).** One model-level
   estimator pinned by commit/blob, bounds (0.5, 4.0), `max_iter=40`,
   `tol=5e-3`, with explicit failure behaviour; qualification is post-tag,
   fail-closed, before the pilot.
5. **Claims and interpretability (R8C-05).** O-H gives matched LoRA capacity,
   with L's bridge/package itemized separately; D4's winner is
   exploratory/selected/optimistic; the X1 interpretability notes are in §12;
   the P1-lite disjointness, smoke exceptions, math-verify pin, 9-arm statement
   and "no later changes" text are corrected.
6. **Verifier items (R8C-06).** Hinge WARN rows; ABORT wording; 2,500
   pre/post-carve accounting; 140-row fallback mirrored; DeepSeek alias/served
   version/fingerprint; pilot 20 vs writer hard cap 50; whole-document tag gate.
7. **Pins (R8C-07).** p1corpus `8b4c020…` and modal-imports `1830a36…` filled;
   DeepSeek final `106e0e9…` recorded; healpilot remains in review and
   modal-image stays a placeholder even though its CUDA-image fix merged at
   `6d3bb65` (pin deferred to 8d).
8. **Tag gate and schema (R8C-08).** `assert_tag_ready` scans the whole
   document; only the `revision_round8c` top-level key is added and every new
   nested key is listed in the round report.

## Verification

`CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/test_prereg.py -q` and the full
non-GPU suite are recorded in `prereg-v1-round8c-report.md`. No tag was
created.

---

# Revision round 8d (pre-tag code, pin and domain alignment)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8d-report.md`. Stage A was exposed, so every item is a pre-tag
technical amendment and **none uses a trained-arm outcome**. This round also
merges `feat/gemma-loop-impl` (`e3f63f3`) and takes its single prereg schema.

## What changed

1. **Probe schedule and generation sanity (R8D-01).** The monitor now states
   the implemented schedule: interval probes at the first optimizer-step
   boundary with `tokens_seen - last_probe_tokens >= 500,000` plus a train-end
   probe when tokens advanced. At the 3M dose and ~26k-token step the probes
   are 0.52, 1.04, 1.56, 2.08 and 2.60M tokens, plus ~3.00M at train end. For
   N ≤ 4, generation-sanity WARNs on repetition loops, or on a missing answer
   / open thinking channel only when `finish_reason != "length"`; a
   length-truncated trace is not a failure.
2. **Round-8c history (R8D-02).** The R8C-03 record and the round-8c choice
   now name the deferral without the literal placeholder token.
3. **Pins and probe (R8D-03).** `task/healpilot` = `e3f63f3…`,
   `modal_image` = `6d3bb65…`, `task_modallite` = `9c4cfd2…` (the code the
   lite run used). The successful fifth probe is recorded (2,000 samples,
   11,725,158 tokens, 1,802.6 s, $2.069; month $6.482 / $30).
4. **Stage-0-lite backend (R8D-04).** The lite run samples on Modal vLLM as a
   single backend (author approval 2026-11-05 ~21:00, throughput-driven, no
   trained-arm outcome; local partial shards inspected only and not used).
   Run id `b605af7a625f93c7e3e08a02`, run cap $6, month ceiling $30.
5. **Pilot domain (R8D-05).** The fit/selection domain stays **math-only**
   because the code samples are not code-verified. Docker is available
   (2026-11-05 22:45) for later-stage verification only; it never applies
   retroactively to the pilot ranking.

## Verification

- `uv run pytest tests/test_prereg.py -q`: **93 passed**.
- Prereg-related pilot tests (`test_arms_selection`, `test_run_pilot`,
  `test_run_pilot_cli`, `test_pilot_mode`): **54 passed**.
- `uv run python scripts/check_prereg_loads.py prereg/prereg.yaml`: exit 0.
- `... --tag-mode`: exit 0.
- Full non-GPU suite: **1674 passed, 24 deselected** (388.39 s).

No tag was created.

---

# Revision round 8e (pilot shortfall amendment)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8e-report.md`. Stage A was exposed; the panel's conditional
rule was fixed **before the realized counts**, so this is a prospective
conditional amendment, not a registration made before observing any data. No
item uses a trained-arm outcome.

## What changed

1. **Pilot verified domain (R8E-01).** The pilot domain is **math plus code
   passing the frozen Docker verifier**; chat never counts. Use only the
   original selected IDs (P1's 50, the 86 originally sampled T-fit problems
   and the 150 originally sampled selection problems). Confirmatory D1 stays
   math-only under its separate contract.
2. **Verifier freeze (R8E-02).** The existing source tests with an ordered
   per-problem manifest, frozen extraction/wrapper/comparison/grading code,
   dependency lock and the content digest
   `python:3.12-slim@sha256:4840d514...`; CPU-only, network none, read-only,
   64 MiB `/tmp`, 512 MiB memory, 64 PIDs, 1 CPU, user `65534:65534`,
   no-new-privileges, 6 s per test, `sandboxed=true`, no unsandboxed fallback.
3. **P1 descriptive-only (R8E-03).** P1 is descriptive for every count and
   **never gates P2**; below 20 it is small-sample descriptive and at zero it
   is unevaluable. The historical 50×4 = 200 means attempts, never 200
   verified rows.
4. **Conditional four-attempt extension (R8E-04).** If `nT0 < 20` or
   `nS0 < 30`, add exactly four attempts at indices 4,5,6,7 for every original
   P2 math/code problem, once. Request ID `p + ":" + str(j)` and seed
   `int.from_bytes(sha256(UTF8("0:"+request_id)).digest()[:8], "little",
   signed=False) % (2**63-1)`. No P1 or chat top-up, no early stop, no second
   batch; parent-linked extension only.
5. **Final adequacy and rejected alternatives (R8E-05/06).** P2 runs only if
   final T-fit ≥ 20 **and** selection ≥ 30 distinct verified problems;
   otherwise the pilot is unevaluable and P2 does not run. D (lower minimums)
   and E (relaxed reasoning filter/prefix scoring) are rejected.
6. **Staged freeze (R8E-07).** `prereg/pilot-shortfall-freeze.json` is
   **REQUIRED before any pilot scoring or P2 training**; it is filled by the
   data addendum after the official count. The immutable rule and verifier
   specification may be sealed before it, but no scoring, P2 training, ranking
   or student-based data revision is allowed while the addendum is pending.
7. **Counts and pin (R8E-08).** Realized math-only counts (P1 13/27, T-fit
   16/33, selection 23/45) and the preview math+code union (16/27/40 from
   `codeverify/summary.json`) are recorded as facts, not decision inputs; the
   official count is re-run with the frozen verifier. `task_modallite_commit`
   is corrected to `4c260eb…`, the seed fix used by the successful run;
   `9c4cfd2…` is retained as the pre-fix failure (seed overflow, $0.758
   booked).

## Verification

- `uv run pytest tests/test_prereg.py -q`: **94 passed**.
- `uv run pytest tests/pilot -q`: **64 passed**.
- `uv run python scripts/check_prereg_loads.py prereg/prereg.yaml`: exit 0.
- `... --tag-mode`: exit 0.

No tag was created.

---

# Revision round 8g (lifecycle evidence, terminal infrastructure disposition and v3 resume)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8g-report.md`. The X1 ruling's lifecycle amendment is appended
verbatim to `prereg/pilot-shortfall-amendment.md`, and its operative fields are
schema-backed in `pilot.verifier_v3.lifecycle_evidence`,
`terminal_infrastructure`, `attempt_ledger` and
`consumer_audit.dispositions`.

## What changed

1. **Threat model and fail-closed rule.** The lifecycle design is approved for
   the frozen original-model completions only. It does not claim isolation from
   arbitrary verifier-aware malicious code; same-UID interference remains a
   disclosed residual risk. Validation establishes fail-closed treatment of
   detected corruption, not candidate-proof isolation; correct output, genuine
   child exit, timely completion, complete lifecycle validation and cleanup
   remain independent pass requirements.
2. **Lifecycle evidence.** A frozen bootstrap performs verifier setup before
   the execution-entry record, reports execution entry through a private
   child-to-wrapper pipe before candidate-controlled execution, and leaves no
   key in candidate-accessible state. The candidate receives neither the key
   nor the evidence descriptor. Keyed authentication uses actual HMAC-SHA-256
   with a fresh per-attempt key and unambiguous framing and identities.
3. **State-specific validation.** Every conclusive Docker outcome (pass, wrong
   answer, candidate error or outer-deadline failure) requires a complete,
   valid, ordered and attempt-bound record appropriate to that outcome before
   acceptance at the grader, counting classifier or count-loading boundary.
4. **Outer-deadline liveness.** The six-second outer budget over the whole
   `docker run` is unchanged and never restarted. An outer-deadline failure
   requires established execution entry before the deadline and trusted
   positive evidence that the identified child had not exited at the deadline;
   missing exit records, delayed `waitpid` observations, finish-record
   timestamps alone, and clock/observation uncertainty spanning the deadline
   are insufficient.
5. **Terminal infrastructure disposition.** After the single identical retry
   also remains infrastructure/unknown, the test is terminal
   infrastructure-unverified and the completion is not verified, retaining the
   infrastructure reason. The old "count stays blocked" rule is marked
   SUPERSEDED HISTORY. A count may finish as
   `complete_with_terminal_infrastructure`; pending retries, incomplete
   request/test inventories, invalid attempt history, freeze or configuration
   mismatches and unverified termination/cleanup remain blockers, and missing
   infrastructure accounting is never zero.
6. **Resume.** The official v3 run resumes under its existing run identity and
   immutable freeze through a durable append-only attempt ledger. Attempt-start
   is committed before dispatch and evidence/result immediately after
   completion; an interrupted attempt consumes the attempt, an interrupted
   retry exhausts the allowance, recorded conclusive results are never regraded,
   and resume is continuation of the same freeze, not a second freeze.
7. **Pool-test consumer dispositions (rows 1-16).** Registered as a pre-seal
   record from `audit-pool-test-consumers.md`: active and affected consumers
   (grader, shortfall verifier and v1/v2 counts, `corpus_v2` membership) are
   superseded by v3; active unaffected consumers (`sample_grades.jsonl`,
   heal/training corpora, P1-lite, pilot NLL scoring, LiveCodeBench) are
   retained; future affected consumers (postprocess verified corpora and Stage
   2 prompts, local `--verify-code`, DeepSeek existing-test prompt context,
   hidden-test merging) are execution-blocked; future unaffected DeepSeek
   parsing/reference-validation paths remain under the writer-wide block, and
   no RL/reward consumer is active. The pool is preserved byte-for-byte.

## Verification

- `uv run python scripts/check_prereg_loads.py prereg/prereg.yaml`: exit 0.
- `... --tag-mode`: exit 0.
- `uv run pytest tests/test_prereg.py -q`: see the round-8g report.
- No pending-pin or alignment placeholder in the registered documents.

No tag was created.

---

# Revision round 8i (void v3 verifier implementation failure and single replacement v3r)

Status: **DONE** (draft only; no tag). Full round report:
`prereg-v1-round8i-report.md`. The X1 ruling's **V3 VERIFIER IMPLEMENTATION
FAILURE, VOID DISPOSITION AND SINGLE REPLACEMENT ADDENDUM** is appended
verbatim to `prereg/pilot-shortfall-amendment.md`, and its operative fields are
schema-backed in `pilot.verifier_v3.void_v3`, `replacement_v3r` and
`author_access_declaration`.

## What changed

1. **Void v3 fact.** Recorded the official v3 run created at
   `2026-11-06T20:23:38.655567+00:00` from commit
   `b5036ac4d7cd28d354aa55b065671df0444d9bdf`, its embedded freeze manifest hash
   `add7f6f49c17d41873be422bb280c1a8a8b315d83b0cc3f13091e815f34ed184`, and the
   ruling's four file-byte SHA-256s for `v3_registry.json`,
   `verifier_v3/verifier_freeze.json`, `count_v3/count_manifest.json` and
   `count_v3/attempt_ledger.jsonl`. The four hashes are distinct from embedded
   self-hashes and are recorded as facts, not an operative count.
2. **Void disposition.** The run is designated
   `VOID_VERIFIER_IMPLEMENTATION_FAILURE`: the production count adapter passed
   `container_names` and `attempts` to `grade_code`, whose signature accepted
   neither, so argument binding failed before pilot candidate dispatch. The
   failed count's 1,144 requests (468 code, 226 kept code), 4,640 scheduled
   request/test pairs, 9,280 attempt starts and 9,280 infrastructure-error
   results are disclosed. Recovery records report synthetic no-op containers;
   the finding is no pilot candidate execution through the failed production
   path, not "no Docker activity". Infrastructure records remain the historical
   record and are neither erased nor relabelled as candidate failures.
3. **Non-operative extension.** The void count's `extension_required=true`
   (math-only T-fit 16 and selection 23) is non-operative: it authorizes no
   generation, and the branch is recomputed only from the accepted v3r count.
4. **Scoped supersession.** The preceding "freeze v3 once",
   same-implementation continuation/resume and terminal-infrastructure
   provisions are superseded **solely** for the identified void run and the
   single authorized v3r replacement; all other conditions remain operative.
5. **Single replacement v3r.** Exactly one replacement is authorized after all
   prerequisites pass, as a new disclosed freeze, not a resume under v3. The
   prerequisites are the identity-forwarding repair (through the real
   `grade_code` and `DockerSandbox` interfaces, matching ledger/lifecycle
   evidence), a production-path integration regression with the real functions
   and no permissive fake graders, a mandatory pre-official real-Docker dry run
   on synthetic non-pilot fixtures with a preserved receipt, X1 (or recorded
   X1-designated reviewer) explicit GO for the exact implementation, a single
   v3r freeze with every configuration/implementation delta disclosed, and a
   full recount of all 1,144 requests including all 468 code requests over the
   complete frozen test sets.
6. **Continuity and retries.** A fixed-location continuity record links the
   preserved crashed v3 registry, void receipt, ruling, replacement review and
   a distinct `v3r_registry.json`; v3r is claimed atomically and durably before
   official output or dispatch. Guards reject the void identity,
   alternate-output-directory bypasses, missing/inconsistent continuity
   evidence and any further fresh replacement; void artifacts remain readable
   for audit but are rejected as operative count, corpus, index or readiness
   inputs. Old attempts stay consumed within the void v3 history; each v3r test
   starts at attempt 1 with only the previously registered single
   identical-configuration infrastructure retry. There is no automatic v3r2 or
   further reset.
7. **Author declaration.** Paragraph 13's dated author access declaration is
   registered as a field with status `pending author confirmation`; its content
   is blank and is not invented.

## Verification

- `uv run python scripts/check_prereg_loads.py prereg/prereg.yaml`: exit 0.
- `... --tag-mode`: exit 0.
- `uv run pytest tests/test_prereg.py -q`: see the round-8i report.
- No pending-pin or alignment placeholder in the registered documents.

No tag was created.
