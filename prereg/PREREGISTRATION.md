# Gemma-Loop pre-registration (prereg-v1, DRAFT, revision round 8e)

Status: **draft for `prereg-v1`**. This document and `prereg/prereg.yaml` are
written to be frozen by the git tag `prereg-v1` **before the pilot and before
any training or trained-arm evaluation**. The tag is not created by this draft.

Freezing rule: after the tag, no confirmatory field changes. Any correction is a
new version (`prereg-v1.1`, `prereg-v1.2`, ...) with a written reason, and the
paper reports the deviation. The pilot rule and the endpoint order are part of
the frozen text. The public decision log is exported to
`docs/prereg-v1-decision-log.md` at the tag.

The confirmatory design is the **two-stage hybrid** on E2B: Stage 1 heal
(forward KL from a same-tokenizer teacher plus a parent KL), then Stage 2
Dr. GRPO RLVR with programmatic verifiers. O-H runs a **real** Stage 1 like L;
it is not a no-op. Part 2 is exploratory iteration and is covered only by its
rules.

---

## 0. Pre-tag decisions and their disposition (round 8: all closed)

**Rounds 8b, 8c, 8d and 8e are pre-tag technical amendments made after Stage A
was exposed.** Stage A was sealed before the author saw its results and was
later opened by auditors and reviewer X1 (section 0.1); none of that makes
already-viewed exploratory results prospective. Every round-8b/8c/8d/8e change
is therefore technical (registration accuracy, execution safety, provenance,
or a panel-ruled prospective pilot amendment) and **uses no trained-arm
outcome**. The eleven round-8b items are numbered `R8B-01` to `R8B-11` in
`revision_round8b`; the eight round-8c items are `R8C-01` to `R8C-08` in
`revision_round8c`; the five round-8d items are `R8D-01` to `R8D-05` in
`revision_round8d`; the eight round-8e items are `R8E-01` to `R8E-08` in
`revision_round8e` and in the changelog below. Round 8e is the panel-ruled
**pilot shortfall amendment**: its conditional rule was fixed before the
realized counts, so it is a prospective conditional amendment, not a
registration made before observing any data. Its governing text is
`prereg/pilot-shortfall-amendment.md` and section 3.3.

The following items are closed before the tag and are also listed in
`prereg.yaml` under `pending_author`. The author delegated the seal decision to
the reviewer panel; the X1 re-check decides the seal.

1. **E3 naming — closed by delegation.** E3 is
   **"Total effect versus O: training + loops, L(N=4) − O"**, the reviewer
   panel's recommended wording. Because L(N=1) is O bit-exactly,
   E3 = L(N=4) − O = [L − O-H] + [O-H − O]; it is the total
   training-plus-loop effect versus the original, not a clean loop contrast.
   E2 is the loop contrast. E3 is kept.
2. **Final OK on the panel-delegated decisions — closed by delegation.**
   Pilot rule B, primary N=4, the Stage 2 mix 3/3/8/1/1 (fallback 3/3/10) and
   the intelligence-per-byte metric are adopted as reviewed; the author's
   final OK is no longer a blocking item.
3. **Bench-addendum scope — closed.** Restricted to compute and memory
   numbers plus **deviation D-7b**, the Stage 2 N contingency (N=6 **or**
   N=8 → N=4, mix 3/3/10, if either does not fit with T-dial-rl). It changes
   no endpoint, selection or analysis field; its one recipe change
   (`rollout_n_groups`) is disclosed as D-7b (§9). The one-row teacher smoke
   and the gate step-time smoke are superseded in round 8 by the registered
   teacher-runtime qualification and the engine gate v6 record.
4. **Stage A addendum — closed.** The analysis rules are frozen at the
   round-7a commit (`analysis_rules_frozen_at_commit` in the yaml; the first
   freeze `f8ca082` was superseded by the disclosed round-7 rule changes,
   §0.1), before Stage A completed. At completion the controller recorded
   sha256/mtime/size of every output in **`prereg/stage-a-addendum.md`**
   (commit **`38b5bd2`**, 2026-11-05T05:56:26-07:00) **before** any result
   was opened. The charts (`d1fde78`, 06:01:44) and the results were then
   shown to the author **in chat at ≈06:02** (SendUserFile). **Later auditors
   did open Stage A after the seal and before the tag** (the round-7a/7b
   native verification at ≈08:01, and reviewer X1 as instructed); the correct
   record is in §0.1 and below. The earlier "no later change has been
   requested" sentence is **retired**: round 8b and then round 8c are later
   disclosed pre-tag technical amendments, and neither uses a trained-arm
   outcome. The seal was made before full-result presentation; it was not a
   blanket absence of prior access (the earlier N=1 and 4-row exploratory smoke
   exceptions are recorded in §0.1).
5. **E4B header and ρ — closed.** The E4B text-resident denominator is
   computed from the pinned `4ecc2ba` safetensors header with
   `huggingface_hub.get_safetensors_metadata`: **14.003 GiB text-resident**
   (8.753 GiB GPU-side text + 5.250 GiB CPU PLE; 7,518,069,034 text params;
   the whole checkpoint is 14.894 GiB). **ρ = 1.62** is frozen.
6. **Pre-tag runs — closed with the corrected record.** Four diagnostic smoke
   attempts ran; the last is a recorded SMOKE-FAIL with its sha256. The one
   bounded repair then developed **gate v6** on `fix/gate-v6` (head
   `8f36aca`), which native review **rejected** (C1–C3); the **repaired smoke
   was never run**, v6 remained unmerged, and padded is the registered engine.
   The batch-1 latency benchmark may move post-tag as descriptive only.
   Teacher qualification runs **after the tag and before the pilot**,
   fail-closed: no pilot if the registered representation is incompatible,
   unavailable or failing. A registered fallback that itself passes is not a
   new representation amendment; any other representation change requires a
   disclosed new version.
7. **Teacher-model download — closed.** The author approved downloading
   `google/gemma-4-E4B-it` @4ecc2ba (16.0 GB) and
   `google/gemma-4-12B-it-qat-q4_0-unquantized` @b6ed862 (24.0 GB) on
   **2026-11-05 at ≈10:55**.
8. **D-7b, D-7c and recipe binding — acknowledged by delegation.** The
   disclosed changes stand (`rows_per_optimizer_step` 16, the bound LRs,
   clip and warmup, the bitsandbytes 8-bit AdamW resolution and the
   2,048-token Stage 2 prompt cap).

**Stage 0 cloud tier (approved).** Stage 0 uses **Modal**, stock **vLLM
v0.30.0** (data only, no plugin), with a **$30 usage limit and a $0 spend
limit** so nothing beyond the free credits is charged. The fallback is to finish
locally on the registered HF engine; the **engine is recorded per row** and the
split assignment is unaffected.

## 0.1 Pre-registration disclosure: training-free exploratory campaign before the tag

A **training-free exploratory campaign** is being run **before the tag**. It is
disclosed here in full so that it cannot become a hidden forking path. Its
results are **exploratory**, will be **reported in full**, and are **not inputs
to block selection**.

- **P1-lite.** A calibrated-NLL sweep per block x {R, R-damped} x N=1..4 on a
  **120-problem verified-trace corpus** (about 1 h).
- **N=2 accuracy.** O and R/R-damped at N=2 on the three candidate blocks
  (7 configs) on **LCB post-cut 182 x 1 + GPQA 198 x 1** (about 21-25 h).
- **N=3,4.** Then N=3,4 on the one block with the **lowest calibrated NLL**
  (about 7 h).
- **Frozen candidates.** The candidate set {25-29, 16-18, 15-29} x bridges is
  **frozen at commit `84c5da0f1778cb7446d20bed0cd428c5799ea1b1`**, timestamp
  `2026-11-04T22:39:04-07:00`, regardless of these results. **No campaign
  output existed at that time.** Benchmark accuracy of R/R-damped is never an
  input to block selection.
- **Dial requirement (AUTHOR 2026-11-05 ~02:00).** The loop count becomes a
  dial. This changed the **training objective of every L arm uniformly**
  (`dial_objective`); the candidate set and bridges are unchanged.
- **Selection-rule deviation.** The selection/STOP rule frozen at `84c5da0`
  (selection at N=2) is superseded by the dial-aware rule B, corrected in
  round 6: the frozen eligibility E is the STOP/eligible set; E is ranked by
  `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]`; the adjacent-regression
  check (≤ 0.005 nats for N=1,2,3 and cNLL(8) ≤ cNLL(1), point estimates) is a
  **filter** that selects the filtered argmax if any arm passes, otherwise the
  unfiltered argmax over E. This is a **disclosed deviation**.
- **Primary-N and Stage-2 deviations.** Changing the primary from N=2 to N=4
  and adopting the Stage 2 mix 3/3/8/1/1 are also disclosed recipe changes.
  They were **decided at the PANEL OUTCOME at about 02:46** on 2026-11-05,
  before `n2.json` was written at **02:58:11** and before `n3.json` existed;
  only their **encoding** came in the round-5 commit at **03:09:09**. At the
  round-6 snapshot (`2026-11-05T03:42:38`) only the N=2 result existed; N=3
  was still running. No N ≥ 3 output was read.
- **What had been seen at the decision time (audit).** Controller audit from
  the OpenCode run logs 02:28–03:22 on 2026-11-05:
  - the native evaluator read only `n1.json` (cNLL 0.1774, T 0.887, top-1
    0.935; identical across blocks by construction — no loop information);
  - the round-5 drafter saw that note via `dial-panel-native.md`;
  - the t5c fix-round-4 implementer listed `runs/trainfree` without reading
    file contents;
  - no other agent touched Stage A outputs;
  - the controller saw only the 4-row smoke (25-29, N=2: ΔNLL R **+0.133**,
    R-damped **+0.022**) and progress heartbeats.
  No N ≥ 2 Stage A output existed before 02:58:11 and none was read by anyone
  before these rules. DS-1 and DS-2 had repository read access, but the logs
  show no access to `results/trainfree`. **Stage A remains exploratory and is
  never an input to selection.**
- **No registered rule uses Stage A**; a registered rule never reads it.
- **Deviation D-7b (pre-tag, after Stage A was shown to the author).** The
  round-7b Stage 2 N contingency changes the registered training-recipe field
  `rollout_n_groups`: if the **T-dial-rl memory bench** shows that an N=6
  **or** an N=8 step does not fit the allocator cap for the chosen block, both
  groups become N=4 (mix 3/3/10). This is a **disclosed recipe deviation**,
  not an execution-only amendment. The trigger is **measured memory**
  (`perf/dial_bench.json`, sha256 `ab91b834…`); **Stage A content is not an
  input**. It was made after Stage A was shown to the author (≈06:02, in
  chat; the author had not replied). The full record, the fixed T-dial-rl
  memory-bench configuration and the "fits" test are in §9 and the changelog;
  `analysis_rules_frozen_at_commit` stays **`8c6d78d`** because no selection
  or analysis rule changes.
- **Deviation D-7c (pre-tag, after Stage A was shown to the author).** Three
  round-7d changes are disclosed as **D-7c** (§9): the
  `rows_per_optimizer_step` text (registered since round 1 as 8,192 tokens per
  step; round 7c corrected it to 16 rows per step), the **8-bit AdamW
  (`adamw_8bit`)** declaration and the **new 2,048-token Stage 2 prompt-cap
  selection rule** (Stage 2 rows ≤ 10,240 tokens; the T-dial-rl bench uses
  10,240-token rows; the realised exclusions and their interaction with the
  1,600-prompt cycling rule go in `prereg-v1-data`). They were made after
  Stage A was shown in chat at **≈06:02** (SendUserFile; Stage A sealed in
  **`38b5bd2`** at 05:56:26 before any result was opened), involve **no Stage A
  content**, and leave `analysis_rules_frozen_at_commit` at **`8c6d78d`**.
- **Rule changes after the first freeze (disclosed, round 7a).** `f8ca082`
  (2026-11-05T04:03:25-07:00) was the first rules freeze. The controller's
  round-7 Ruling, logged in the decision log at about 04:16 −07:00 (the
  session-transcript time: it was written right after the merge `a360908`,
  commit time `2026-11-05T04:11:58-07:00`, whose CPU suite ended at 04:15;
  before `n3.json` existed) and drafted in `1657c99` (05:40:11), then changed three
  selection-rule fields, all from the native review of round 6 (item R1), not
  from data: (a) the teacher step measures L's gain at the primary **N=4**
  (was N=2); (b) `delta_arm` in the eligibility set and in the ranking uses the
  **block-matched O-H** (the O-H with the same block layers) instead of a
  single O-H[25-29]; (c) two pilot arms `O-H[16-18]` and `O-H[15-29]` were
  added. Stage A outputs `n1` (01:34), `n2` (02:58) and `n3` (04:25:29)
  existed on disk during these edits. The agent logs show only directory
  listings and hashing of `n2`/`n3`; **nobody opened `n2` or `n3`**. Only
  `n1.json` (N=1, identical across blocks by construction, no loop
  information) was read, by one panel evaluator. The rules are re-frozen at
  the round-7a commit, before Stage A completed.
- **Stage A visibility and sealing.** The analysis rules are frozen at
  the **round-7a commit** (`analysis_rules_frozen_at_commit`), **before Stage A
  completed** (first freeze `f8ca082` superseded; see the disclosure above).
  Stage A completed at **05:56:08**; the controller committed
  **`prereg/stage-a-addendum.md`** (**`38b5bd2`**, 05:56:26) with sha256, mtime
   and size of **every** output **before any result was opened**. The charts
   (`d1fde78`, 06:01:44) and the results were then shown to the author **in
   chat at ≈06:02** (SendUserFile); the author had not replied at the time of
   this draft. **Provenance correction (round 8):** the earlier blanket
   statement that no agent other than the presenter opened the outputs before
   the tag is false and is retired. The presenter showed the results at ≈06:02;
   subsequent auditors, including the round-7a/7b native verification at
   ≈08:01, opened and compared the sealed JSON/tables/summary **after the seal
   and before the tag**, and reviewer X1 read the results after the seal as
   instructed. Hashes demonstrate artifact integrity, not who knew what. That
   post-seal access does not make already-viewed exploratory results
   prospective; the main trained endpoints remain unobserved. Any change after
   the author sees Stage A is a **disclosed deviation** in the changelog,
   citing those hashes. The table below is the **pre-completion snapshot**
   (`2026-11-05T03:42:38-07:00`) and is not the visibility seal;
   `campaign.json` and `child-A_n*.json` are mutable status files that were
   re-hashed in the addendum.
- **Freeze sequence (round 8, replaces the "frozen at the tag" shorthand).**
   The freeze sequence is (1) **rules** — analysis and selection rules frozen
   at the `prereg-v1` tag; the analysis rules were already frozen at
   `8c6d78d`, the historical pre-Stage-A snapshot; (2) **data** — the realised
   manifest SHA, trace IDs, verifier outcomes, fit/test IDs and counts are
   frozen in `prereg-v1-data` after the tag and **before the pilot**;
   (3) **pilot selection** — the selection rule is frozen at the tag and the
   selected configuration is frozen **before main training**; (4)
   **implementation** — the resolved recipe/binding and `analysis-v1` are
   frozen **before any trained-arm evaluation**; (5) **main runs** — main
   training and evaluation start only after steps 1–4. The final `prereg-v1`
   commit is the authoritative complete specification; no commit or tag makes
   already-viewed exploratory results prospective.
- **Reuse rule.** Campaign results are reused for the **loop ladder only**, and
  only with identical engine/harness; they are **never** reused for E3 or the
  difference-in-differences.
- **Stage B gate.** Stage B (accuracy on confirmatory items) is **not launched
  before the `prereg-v1` tag**; it runs only after the tag, regardless of the
  author's E3 decision. Stage A (the calibrated-NLL sweep on the P1-lite
  corpus) does not touch confirmatory items and may run.
- **Disjoint corpus (corrected in round 8c).** The **original Stage A**
  P1-lite corpus predates the canonical-ID audit and is **not** claimed to be
  disjoint from every Stage 0 split; the later canonical-ID audit found
  protected-set overlaps. The **current teacher-qualification use** filters
  the registered protected sets (pilot evaluation, T-fit, MONITOR and
  confirmatory), and only that filtered corpus is claimed disjoint. Do not
  transfer the blanket-disjointness claim back to the original Stage A corpus.
- **Reporting.** Accuracy-vs-N per block plus a table with Wilson/paired CIs.

### Stage A sealing snapshot (path, mtime, size, sha256)

The full machine-readable list is in `training_free_campaign.stage_a_sealing`
in `prereg.yaml`; the files below are every file that existed at the snapshot.

| path | mtime | bytes | sha256 |
|---|---|---:|---|
| `runs/trainfree/plan.json` | 2026-11-05T02:58:12 | 12011 | `406d24bba8079c239f27f03aba98f1d5ea967dc11f8486aced7ab19abad154cd` |
| `results/trainfree/stageA/n1.json` | 2026-11-05T01:34:20 | 11292 | `c1b23b20ccc45f74feca87c0f91dc789daf5155b8186e9f329fd4e111f3108d8` |
| `results/trainfree/stageA/n1.md` | 2026-11-05T01:34:20 | 2869 | `607a5544f995f92afec76c3ef9a4084a0e892b0764cc50514169670d89c715f1` |
| `results/trainfree/stageA/n2.json` | 2026-11-05T02:58:11 | 11671 | `d76cdc50aa0c9b10b12dc8b0e3b84fca327787db2f60a38b0138abd872980cd6` |
| `results/trainfree/stageA/n2.md` | 2026-11-05T02:58:11 | 2869 | `cbc88f0f73d96cec1739dde3f72cb8e25252239a4c10ab2b8d2b97c662336775` |
| `runs/detached/20261104-232739-289-trainfree-corpus.log` | 2026-11-04T23:28:00 | 2388 | `90c87a73e6d890c5ba0a817717a212ae1b79a64ba1f4ae06b2efb9c27502a7d4` |
| `runs/detached/20261104-232739-289-trainfree-corpus.pid` | 2026-11-04T23:27:39 | 5 | `d0c2949eb693ef90f67e66b1752cc26b57d336933de0f2949c149556be857e2c` |
| `runs/detached/20261104-232848-261-trainfree-corpus.log` | 2026-11-04T23:30:23 | 12927 | `f053f28b4e56d18de4c933b8b30b649cfe5626292f8e3a25e7c52f634c7ed2c9` |
| `runs/detached/20261104-232848-261-trainfree-corpus.pid` | 2026-11-04T23:28:48 | 5 | `bfc0723645e5ef55c2a44fa8ddb04d07a18aaebb6123905ec33efa37d56b277a` |
| `runs/detached/20261104-234240-526-trainfree-corpus.log` | 2026-11-05T01:14:45 | 11060 | `501363c06212f6524ab820335b9e15ceacf93d80ae7d3df2122847806496029c` |
| `runs/detached/20261104-234240-526-trainfree-corpus.pid` | 2026-11-04T23:42:40 | 5 | `245cd6e5a3e8ef67ab2d81e9efc795af0b836777cb1ba1cd0a2932a54459ab9d` |
| `runs/detached/20261105-011530-002-trainfree-stageA.log` | 2026-11-05T01:17:42 | 52818 | `fc51b14aa0ca2edba5fefdc36b5da53d5879c2132b2298e618bdc25a79c6825b` |
| `runs/detached/20261105-011530-002-trainfree-stageA.pid` | 2026-11-05T01:15:30 | 5 | `a5772fa5b702c5428f0236138e2890280a2bde64334450b1770ab3bd5d38e5a1` |
| `runs/detached/20261105-012317-797-trainfree-stageA.log` | 2026-11-05T02:58:41 | 30731 | `27f717e830bbd134b0a2f325bf74b2e7e91ba7492c9c021ec0c672adaa46ec0f` |
| `runs/detached/20261105-012317-797-trainfree-stageA.pid` | 2026-11-05T01:23:17 | 5 | `aad043becd9ff750cd2e045531f70b607deb96588da134ce718e7eb67d47141a` |
| `runs/detached/campaign.json` | 2026-11-05T03:42:37 | 474 | `8f5c97b8c72e433ba40a743d8da6ce00bc711108a7a95a147e355867621dcb96` |
| `runs/detached/child-A_n1.json` | 2026-11-05T01:34:20 | 429 | `abedfd8a585916754e196519870a62de71df12c33335062f37d5f64f7505d880` |
| `runs/detached/child-A_n2.json` | 2026-11-05T02:58:11 | 429 | `e88fa32fef608a548007a901e7a08f2c8fad89b4176393d31eb04355ea26d5b2` |
| `runs/detached/child-A_n3.json` | 2026-11-05T03:42:30 | 432 | `00935e2681b116e5dfa67c769b78d07c9d952a91ba8873bd09e7f10bb69a68ba` |
| `runs/detached/child-A_n4.json` | 2026-11-05T01:17:30 | 404 | `abf5e9f522930005173eaa0ddb448ab957d77d91c3c2b05c3b084c3036882c62` |
| `runs/detached/corpus.json` | 2026-11-05T01:14:45 | 488 | `762692a405eac65cc7bf4764a3a86e476798f656af4342eae2f4aca102c4cf8e` |
| `runs/detached/pilot.json` | 2026-11-05T01:22:55 | 429 | `74e6b6f32953ce7efc2d8639d72e3406f4bedc59fbcc72b8f296f653d6c81074` |


The campaign is descriptive; it is not part of the confirmatory family.

---

## Changelog

### Revision round 8 (external review panel)

**One dated pre-tag amendment answering the external Codex panel** (X1 plan and
prereg, X2 engine gate, X3 heal teachers). It is made with Stage A and the
engine smoke visible. It does **not** imply that a new commit or a git tag makes
already-viewed exploratory results prospective. The historical pre-Stage-A
rules snapshot stays **`8c6d78d`**; the final `prereg-v1` commit is the
**authoritative complete specification**. The changes are split as X1 asked:

- **Claim clarification.**
  - Added the explicit claims paragraph (§1, §6, §12 and the YAML endpoint
    descriptions): **D1** = average calibrated-likelihood gain/non-inferiority
    on O-verifiable math traces, not a general "usable reasoning dial";
    **E2** = the complete looped training/inference package vs a matched-data,
    matched-layer adapter control; **D2** = the extra-depth effect; **D4** =
    teacher-forced likelihood evidence only. A D1 pass alone never yields the
    "usable reasoning dial" or "every extra loop helps" headline. The
    registered numerical endpoints and α allocation are unchanged.
  - Retired the "both teachers are four-bit" limitation: E4B is bf16 with CPU
    PLE (int8 fallback); the 12B is the QAT-unquantized checkpoint loaded with
    bitsandbytes NF4, not automatically the vendor's Q4_0 execution or score.
  - Intelligence per byte: fixed-byte paired effects are the main
    presentation; Google's gap fractions φ and the byte-equivalent multiplier
    m are illustrative only.
- **Analysis precision.**
  - Completed the NLL analysis contract (§6.3 and YAML
    `endpoints.nll_analysis`): fixed assistant thinking+answer mask;
    token-weighted NLL with problem-level resampling; positive-oriented
    `T_gain`/`T_adj`; `p = (1 + #T* <= 0)/10001` with conservative ties and no
    interpolation; temperatures fitted only on bucket 60-61 and held fixed in
    the primary bootstrap with inference conditional on them; an empty or
    unusable fit/score corpus makes D1 unevaluable with no rejection and no α
    transfer; fit/test trace IDs and counts frozen in the data addendum before
    any trained-arm scoring.
  - Registered the engine gate v6 replacement eligibility rule and records
    (X2): frozen padded-only calibration `[0,2,4,6]/[1,3,5,7]` and validation
    `[0,4,1,5]/[2,6,3,7]`; chosen-token regret `D` with
    `τ_g = max(0.25, max calibration D + 0.0625)` and a 0.50
    inconclusive ceiling; recorded decision traces with mandatory
    reconstruction as a hard fail; stable-support and changed-support
    certificates with raw `≤ 2·calib + 0.05` (cap 0.50) and TV
    `≤ 2·calib + 0.005` (cap 0.10); matched pooled denominators, per-key
    margins and the 3/8 floor; ≥ 32 shared steps per request; exactly one
    repair cycle.
  - Registered the pilot "why" report as descriptive diagnostics that never
    enter selection: teacher qualification error, teacher entropy/confidence,
    initial vs final student–teacher KL and gap closed, gains by span and
    position, per-N curves N=1..8, cost per token.
- **Provenance correction.**
  - Corrected the Stage A viewer statement: the presenter showed the sealed
    results at ≈06:02, and **later auditors (round-7a/7b native verification
    at ≈08:01) and reviewer X1 opened Stage A after the seal and before the
    tag**. Hashes prove artifact integrity, not who knew what.
  - Replaced the claim that the chosen architecture/data SHA freeze at the
    initial tag with the explicit sequence **rules → data → pilot selection →
    implementation/analysis-v1 → main runs** (§0.1, §9, §11).
  - ~~The engine pin is the `fix/gate-v6` commit the repaired smoke ran on.~~
    **Corrected in round 8c:** the repaired smoke **never ran**, v6 was
    rejected in review and remained unmerged; there is no repaired-smoke pin,
    and the padded engine is the sole operative engine. `eed7fa7` remains
    historical gate-v5 provenance.
  - Closed the pending list: E3 = "Total effect versus O: training + loops,
    L(N=4) − O" (author-delegated, panel-recommended); D-7b and D-7c
    acknowledged by delegation; teacher downloads approved 2026-11-05 at
    ≈10:55; the E4B header/ρ item and the pre-tag runs closed.
- **Execution contingency.**
  - Registered the Stage 2 feasibility branch (§4.3): O-H and L at the
    mandatory N=2/3/4 paths at the **actual 10,240-token maximum**, plus
    N=6/8, including actual rollout storage and optimizer state; D-7b's
    fallback must itself fit; **if a mandatory path does not fit, stop before
    either main Stage 2 arm and issue a new disclosed version** — never shorten
    one arm, discard long rows, substitute an N mix, raise the cap or retry a
    permanent OOM indefinitely.
  - Registered the teacher runtimes and ordered qualification rule (X3):
    E4B bf16 with CPU PLE primary and int8 fallback; 12B NF4 primary and the
    official QAT Q4_0 via llama.cpp as the registered fallback; the section-7
    corpus, references, metrics, acceptance table, the single 20 %-margin
    expansion to 100 rows, the ordered fallback, one frozen runtime per teacher
    shared by its L and O-H, manifest identity, and "best qualified resident
    runtime per teacher" as the comparison.
  - Recorded the **X3-C1 execution repair**: freeze teacher parameters, run the
    entire precompute including the head under `inference_mode`, detach stored
    tensors and test no `grad_fn`; a shard whose manifest does not bind the
    head/inference-mode policy fails closed.
  - Registered the **LOCAL Stage-0-lite sampler** as a Stage 0 compute-tier
    option for the pilot data (padded HF engine, same sampling, cap =
    `max_length` minus prompt length), recorded as a covariate; the full
    Stage 0 for main runs remains Modal-or-local by availability. Round 8c
    retires the earlier "identical accepted set" wording: a capped run
    preserves an accepted prefix on the same backend, but cross-backend parity
    is not established by seeds and rests on shared frozen artifacts plus
    recorded backend provenance.
- **New eligibility design.**
  - Replaced the 0.25 canonical top-two near-tie rule with the X2/X1 gate v6
    eligibility design (see `engine_gate.replacement_rule`). This is a **new
    prospective eligibility design**, not an amendment of the old smoke
    margins: the old artifact is retained as SMOKE-FAIL and cannot authorize
    any exactness-margin relaxation.
  - The **current refill candidate is NOT eligible**; the repaired smoke must
    pass the frozen v6 rule, the full counted gate and Part B. Failure or an
    inconclusive result selects the **padded** engine for this preregistration,
    with no further rework. **Round 8c closes the refill track: this round-8
    instruction is historical and none of that work is run.**

### Revision round 8b (Stage-A-exposed pre-tag technical amendment)

**One dated pre-tag technical amendment**, made after Stage A was exposed
(section 0.1). **No item uses a trained-arm outcome.** The only measured
numbers are execution measurements (Stage-0-lite padded throughput and Modal
booked spend), hashes and commit provenance. Each item says what changes, where, why
and its outcome basis; the YAML mirrors the same ids.

- **R8B-01 — heal monitor corpus, rules and reserved-id contract.** The
  MONITOR corpus is the **64 train-bucket (0-49) problems with the largest
  `sha256(seed:problem_id)` keys**, one sample each, excluded from every
  training subset and training file. The pilot precomputes monitor teacher
  shards with the same frozen runtime and `max_length` as the arm and passes
  them as `--monitor-teacher-data`; the probe fit half uses the MONITOR
  teacher, never the training collator. Registered ABORT rules: non-finite
  loss/gradient/logit; exit-1 bit-exactness failure; teacher-data integrity
  failure (> 2e-3 renormalised sums, wrong shapes/ids); parent KL above
  1.0 nat/token for 3 consecutive probes. WARN rules are the brief's defaults
  (loss spike, gradient norm, teacher-KL stall, hinge activity, overfitting
  gap, generation sanity, bridge saturation, memory and throughput); the hinge
  rule is a **row** rule (more than 50 % of rows after 1M tokens, corrected in
  round 8c). An ABORT **writes no checkpoint**, leaves the last **COMPLETE**
  checkpoint intact, writes `alerts.json` and **exits 76 terminal**; resume
  needs `--ack-abort` on every registered resume path; a probe exception is a
  WARN row and **two consecutive probe failures are an ABORT**.
  `reserved_ids.json` is a
  **problem-id** map (`monitor`, `pilot_eval`, `pilot_fit`, `heal_train`,
  `confirmatory`, extra keys allowed); the monitor proves
  `monitor ∩ (every other key ∪ training rows) = ∅` and never treats its own
  key as an overlap. Monitoring is execution safety only and never feeds
  selection.
- **R8B-02 — P1 count.** P1 is **50 bucket-59 problems × 4 samples = 200
  rows**; `pilot.p1.rows` counts sampled completions, not problems, and
  `data.splits.pilot_p1_problems` stays 50.
- **R8B-03 — pilot matrix stages.** The 11-arm matrix is registered as two
  stages: **teacher stage** on 25-29 + gch at L's N=4 and O-H[25-29] at N=1
  for E4B and 12B; **block/bridge stage** on the E4B L arms against the
  block-matched E4B O-H under rule B. The chosen recipe is then trained with
  the selected teacher, so a 12B teacher win carries through without making
  12B a block/bridge candidate.
- **R8B-04 — math grading.** Correct iff **math-verify 0.9.0** (internal
  timeouts disabled) run in a grading subprocess with a **10 s outer
  timeout**, **OR** the registered AIME/integer normalisation says correct;
  `decided_by` records `math_verify`, `normaliser`, `both` or `timeout`.
  Reason: math-verify's own timeout uses multiprocessing spawn, which fails
  on Windows, and it disagrees with the normaliser on `\boxed{1,234}` versus
  `1234`; the OR rule prevents a definitive `False` from overriding a
  normaliser match.
- **R8B-05 — Part-1 pool mode and sources.** The realised pool is Part-1 mode
  **without creative briefs**: 5,000 problems (2,000 code, 1,000 chat, 2,000
  math), manifest sha256
  `70053b90fa631fda1e5e0fa77a932e30c4bab80e81ef136f2330adf0daaef84e`.
  When Part 2 briefs are written they are decontaminated against the frozen
  pool in the reverse direction. Source fixes: **APPS** parquet at
  `refs/convert/parquet` `0f10e424e13e1c2a69f851e153097b71b6734a1f`; **TACO**
  `ALL/train-*.parquet` at pinned `d593ed0a`; **CodeContests** `pre_2023_05`
  Python-only reference solutions.
- **R8B-06 — engine.** The **refill track is closed for prereg-v1**; the
  padded adaptive-batch HF engine is registered for **all evaluation and
  Stage 2**. Measured Stage-0-lite padded rate: first committed shard **100
  requests in ≈2,600 s, ≈195 tok/s, 52/100 kept**, giving **≈17-20 h** total.
  The engine gate v6 and Part B records remain inactive historical provenance;
  round 8c rewrites §2.3/§4.3 to register this unconditionally.
- **R8B-07 — teacher runtimes and qualification corpus.** Runtimes as merged:
  E4B bf16 with int8 fallback; 12B NF4 with the Q4_0/llama.cpp fallback;
  qualification per X3 section 7. Corpus amendment: rows from **O's Stage 0
  local samples** (train bucket, verified, stratified **4/8/4/4**, 20-row
  screen then expansion to **100 in seeded hash order**), disjoint only from
  the registered protected sets (**pilot evaluation, T-fit, MONITOR,
  confirmatory**); train-pool rows are allowed. The 120-row P1-lite corpus is
  the fallback and keeps **85 rows** after protected-set filtering, so a
  shortfall fails closed.
- **R8B-08 — code-verification dependency.** Verified pilot targets (P1,
  T-fit, selection) need **sandboxed code grading**. Code samples are
  verified in the **Docker sandbox before any pilot evaluation**; the
  unsandboxed local mode is **never** used for model-generated code. The
  round-8b "math-only until post-processing verifies the code" fallback is
  **HISTORICAL ONLY and superseded by round 8e**: the operative rule requires
  the frozen verifier, with no math-only fallback and no sample silently
  relabelled verified.
- **R8B-09 — external APIs.** DeepSeek only via the **official API**, model
  **deepseek-flash** (served `DeepSeek-V4.1-Flash-1017`), **off-peak only**
  (peak `01:00-04:00` and `06:00-10:00` UTC Mon-Fri, **15 min before / 5 min
  after** margins), **paid pilot ≤ 20 items** before any full run, **waste
  stop ≥ 3 of the last 20**, **one registered fingerprint**, per-phase budgets
  and caps (**phase 1 hidden tests cap $5; whole project expected ≈$40-50;
  caps sum $71**), each phase approved by the author. **NVIDIA NIM trial is
  not usable** for anything entering training, judgments, hidden tests or the
  dossier (terms s1.2/1.4, 4.2, 4.4, 4.12). **OpenCode endpoints are never
  used for training data or judgments.**
- **R8B-10 — Modal.** Probe history: attempt 1 `InvalidError` (image order);
  attempt 2 `ModuleNotFoundError rapidfuzz`, six failed startups, stopped by
  hand, **$3.087 booked conservatively**. The money guards now required before
  any launch are registered (explicit approval and cap, frozen manifest, CPU
  preflight, image-completeness test, fail-fast stop with `app.stop()`, bounded
  startup/heartbeat and per-call deadlines, conservative booking). **Modal is
  optional for Part 1**; local sampling is the registered path.
- **R8B-11 — pins.** Already merged: **`217b939`** teacher runtime,
  **`2eef68b`** DeepSeek guards, **`dfea02a`** sources, **`c840e9b`** Modal
  image order. The pins for `task/p1corpus` and `fix/modal-imports` were
  filled in round 8c; round 8d fills the last two (`task/healpilot` and
  `modal_image`) and adds the Modal lite code pin. The tag gate refuses a tag
  while any registered placeholder remains; no placeholder survives round 8d.

### Revision round 8c (pre-tag technical reconciliation and seal-readiness amendment)

**One dated pre-tag technical amendment**, made after Stage A was exposed.
**No item uses a trained-arm outcome**: the inputs are frozen pool split
counts, commit/source provenance and execution measurements already recorded.
The eight items are `R8C-01` to `R8C-08` in the YAML `revision_round8c` block.
The numeric confirmatory endpoints and α allocation, and the pilot ranking
formula, remain fixed. R8B-01/04/08 did change the reserved training data
(the MONITOR carve-out), grading/eligibility (math OR normaliser; Docker code
verification) and stopping/domain contingencies; those effects are stated
rather than denied.

- **R8C-01 — padded is the sole operative Part 1 engine.** §2.3 and §4.3 now
  register the padded adaptive-batch HF engine unconditionally for Stage 2 and
  every evaluation, with no conditional refill authorization. `engine_gate`
  and `part_b` are marked **INACTIVE** and retain no execution authority; no
  post-tag engine addendum is required. The corrected record: **four** old
  smoke attempts; the one bounded repair developed **gate v6** on
  `fix/gate-v6` (head `8f36aca`), which review **rejected** (C1–C3); the
  **repaired smoke never ran** and v6 remained unmerged. The operative padded
  source is pinned by path and blob hash at `54cab07` (`serve/generate.py`,
  `serve/cache.py`, `eval/runner.py`).
- **R8C-02 — realized corpus and shortfall rules frozen.** Hash split counts
  are **nominal**. Realized counts from the frozen pool split files: train
  **2,488** (2,424 after the 64 MONITOR ids; 994 math), P1 **58**, T-fit
  **86** (29 math), selection **152** (62 math), NLL fit **90** (40 math) and
  NLL test **396** (162 math). P1 is **200 attempts = 50 problems × 4
  samples**, not 200 retained rows: the deterministic truncation takes the
  first 50 distinct bucket-59 problems in ascending
  `sha256(0:request_id)` order, ties by `request_id`, keeping all four samples.
  The selected set is 23 math / 19 code / 8 chat. A row is retained only if it
  passes the Stage 0 filter and its answer verifies; chat never counts as a
  verified target. The round-8c/8d pilot fit/selection domain was **math-only**
  (recorded as dated history only): that clause is **superseded by round 8e**,
  which adopts math plus code passing the frozen Docker verifier (chat never)
  and the single conditional four-attempt extension. Fail-closed minima:
  20 P1 as a descriptive label only (never a P2 gate), 20 T-fit, 30 selection,
  32 MONITOR rows, D1 fit 20 / score 40. No borrowing, regeneration or extra
  epoch; all 64 MONITOR IDs stay reserved even when traces fail. Zero
  RL-prompt and insufficient Stage 1/RFT token branches are registered. The
  dedicated D1 60-69 targets are an explicit post-tag, pre-pilot
  `prereg-v1-data` obligation: the registered Stage 0 data sampler (Modal
  vLLM when approved, otherwise the local padded engine) generates them, four
  samples per problem, `max_new_tokens = 32768` and the 8192 rendered-length
  filter, with trace IDs/outcomes/counts frozen before any trained-arm score.
- **R8C-03 — monitor abort/resume and missing-arm policy.** An ABORT writes
  **no** checkpoint and preserves the last COMPLETE one. `--ack-abort` is
  required on **every** registered resume path; without it a registered
  resume over an abort record exits 2. With the acknowledgement, a COMPLETE
  checkpoint is resumed; if none exists, the partial run directory is
  archived and the same registered run restarts from zero. Either path keeps
  unchanged data, dose, thresholds, seeds and recipe. A technical repair that
  restores the registered implementation is distinguished from a
  recipe-changing fix, which requires a new disclosed version. Probe
  configuration is fixed: 500k-token interval and train end; NLL N = 1,2,3,4;
  generation N = 1,2,4; 32 deterministic MONITOR fit/score rows; temperature
  candidates 0.5–4.0. The exact token points and due-count boundaries, and the
  generation-sanity length-truncation exception, were deferred to round 8d
  instead of being hand-computed from the implementation. An
  aborted or incomplete
  teacher/arm cell is reported as such; it is **not** replaced by an informal
  experiment, shorter dose, earlier checkpoint or survivor selection, and a
  WARN alone never changes training or selection.
- **R8C-04 — temperature-fit contract and qualification timing.** The
  temperature estimator is bound to `gemmaloop.eval.calibrated_nll`
  `fit_model_temperature` / `calibrated_nll_report` at commit `54cab07` (blob
  `f51de454…`): bounds **(0.5, 4.0)**, bounded golden-section search,
  `max_iter = 40`, `tol = 5e-3`, fp32 mean-NLL objective. The tensor helper's
  different defaults (60 / 1e-3) are **not** the registered path. The same
  estimator is used in pilot and confirmatory scoring. An empty labelled fit
  set, non-finite objective, malformed denominator or unusable fit makes that
  comparison unevaluable; a boundary fit is reported, not widened. Teacher
  qualification runs **after the tag and before the pilot**, fail-closed; a
  registered fallback that itself passes is not a new amendment, while any
  other representation change needs a disclosed version.
- **R8C-05 — claims and interpretability reconciled.** O-H is the matched-data,
  matched-runtime control with the **same rank-16 LoRA on the same block
  layers**: that is matched **LoRA capacity**, not exact total-parameter or
  training-FLOP equality. L's **1,536-parameter recurrent bridge**, multi-exit
  supervision and loop package are itemized separately; the O-H–L contrast is
  not the pure causal effect of recurrence. D4's test-selected winner and
  interval are labelled **exploratory/selected/optimistic** and require
  fresh-data validation before any Part 2 serving claim. The X1
  interpretability notes are in §12: the teacher comparison is about
  **deployable representations, not size**; the "why" report is
  **descriptive, not causal**; the pilot domain is **O-solvable math
  (verified)**; main **lineage/dose** is explicit; and **shared seeds do not
  establish backend identity**. The P1-lite blanket-disjointness claim, the
  earlier N=1/4-row smoke exceptions, the stale math-verify pin, the 9-arm
  statement, the "no later changes" text and the superseded drafter-choice
  engine/parameter wording are corrected.
- **R8C-06 — round-8b verifier items.** Hinge WARN is **rows**, not tokens;
  ABORT wording is corrected; `train_problems 2,500` is the **nominal
  pre-carve** target with 2,488 realized and 2,424 after the MONITOR ids plus
  the shortfall rule; the **≥140-row** P1-lite extension is mirrored here;
  `stage2.deepseek.model` is the alias **`deepseek-flash`** with
  `served_version = DeepSeek-V4.1-Flash-1017` and the registered fingerprint
  record. The registered paid pilot is **20 items**; the writer's
  `PILOT_MAX_ITEMS = 50` is recorded as a **code hard upper bound**, not a
  paid-run size. `assert_tag_ready` now scans the whole document.
- **R8C-07 — pins.** Full 40-character ids: `task/p1corpus` merged at
  **`8b4c020b8da5112e7b6e8b574dc5118c1d6510ba`** (branch head
  `1e789963f1e313a90686060a223276eef0a0b4ec`), `fix/modal-imports` merged at
  **`1830a361522ec2ef67d5b307e4e4b49b97d83aad`** (branch head
  `c9d0cdee3a365039ac796df64cb063358e48bf58`), and the final DeepSeek guard
  merge **`106e0e9fb2ff8f63a356f6f3be657967ad653183`** (branch head
  `37583c197e4ac76d20031b0969935e92c49a165c`). Exactly two placeholders
  remain: `task_healpilot_commit` (still in review) and `modal_image_commit`.
  The CUDA-image fix merged at `6d3bb65c7a007bc67adf1ddf1a9536272b2163c5`,
  but its pin is deliberately deferred to round 8d, so the placeholder stays.
- **R8C-08 — tag gate and schema surface.** `assert_tag_ready` recursively
  scans every string in the loaded document for the placeholder token and
  lists every location; a non-pins placeholder test is added. Round 8c adds
  no new top-level YAML key other than `revision_round8c`; every new nested key
  is listed in `prereg-v1-round8c-report.md` for the strict pydantic model.

### Revision round 8d (pre-tag code, pin and domain alignment)

**One dated pre-tag technical amendment**, made after Stage A was exposed.
**No item uses a trained-arm outcome**: the inputs are the committed
implementation, commit/provenance records and execution measurements already
recorded. The five items are `R8D-01` to `R8D-05` in the YAML
`revision_round8d` block. The numeric confirmatory endpoints, the α allocation
and the pilot ranking formula remain fixed.

- **R8D-01 — probe schedule and generation-sanity rule aligned with the
  code.** Interval probes fire at the first optimizer-step boundary where
  `tokens_seen - last_probe_tokens >= 500,000`, with `last_probe_tokens`
  advancing after every probe; a train-end probe fires when `tokens_seen >
  last_probe_tokens`. At the registered 3M-token pilot dose and the
  ~26k-token optimizer step the achieved probes are **0.52M, 1.04M, 1.56M,
  2.08M and 2.60M** tokens (the 20th, 40th, 60th, 80th and 100th step), plus
  a train-end probe at **~3.00M** tokens. For `N <= 4`, generation-sanity
  WARNs on repetition loops, or on a missing answer or an open thinking
  channel **only when the generation stopped before the 512-token cap**
  (`finish_reason != "length"`); a length-truncated trace is expected and is
  not a sanity failure, while repetition loops warn regardless of
  `finish_reason`.
- **R8D-02 — round-8c history reworded.** The R8C-03 change record and the
  round-8c choice that described the monitor deferral now name the deferral
  without quoting the literal placeholder token. No 8c decision changes.
- **R8D-03 — final pins and the Modal probe success (pin corrected by round
  8e).** `task/healpilot` is
  **`e3f63f352a32ccb3bccb013a781536ee6a537511`**; `modal_image` is
  **`6d3bb65c7a007bc67adf1ddf1a9536272b2163c5`**; and the Modal lite code pin
  is **`4c260ebe1810a0f95cb47d774bd8eed894e9cfda`** (`task_modallite_commit`,
  the seed-fixed commit the successful run actually used). Round 8e records
  the pre-fix `9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab` as the version that
  failed with a seed overflow and booked **$0.758**. The successful fifth
  probe produced **2,000 samples / 11,725,158 generated tokens in 1,802.6 s**,
  booking **$2.069**; the month's booked total is **$6.482 of the $30
  ceiling**.
- **R8D-04 — Stage-0-lite on Modal vLLM as one backend.** Under the author's
  approval of **2026-11-05 ~21:00**, the lite sampling runs on Modal vLLM as
  a **single backend** (throughput-driven, ~6,500 generated tok/s vs ~195
  tok/s locally; no trained-arm outcome). The local partial shards were only
  inspected for counts, acceptance and throughput and are **not used** or
  mixed in. Run id **`b605af7a625f93c7e3e08a02`**, run cap **$6**, monthly
  ceiling **$30**.
- **R8D-05 — pilot domain was math-only in round 8d (HISTORICAL;
  superseded by round 8e).** Round 8d recorded that the pilot fit/selection
  samples were **not code-verified** and the domain was **math-only**. That
  clause is **not operative**: round 8e adopts math plus code passing the
  frozen Docker verifier (chat never), with the single conditional
  four-attempt extension and the final T-fit ≥ 20 **and** selection ≥ 30
  distinct-problem check. That round-8d clause said code verification applied
  to later stages only and **never retroactively** to the pilot ranking; it is
  retained only as dated history. **Docker is now available** (2026-11-05
  22:45; Docker Desktop server 29.3.1, 20 CPUs, 15.5 GiB); the frozen verifier
  and staged freeze now govern code eligibility.

### Revision round 8e (pilot shortfall amendment)

**One dated prospective conditional amendment**, adopted by the panel after
Stage A was exposed and **before prereg-v1 and before any pilot scoring or
trained-arm training**. It uses realized original-model sampling/verification
counts but **no trained-arm outcome or pilot ranking**; the conditional rule
was fixed by the panel before the realized counts, so this is not a claim that
no original-model data were observed. The governing text is registered
verbatim below and in `prereg/pilot-shortfall-amendment.md`; the YAML operative
fields in `pilot.code_verification`, `pilot.shortfall`, `pilot.freeze`,
`pilot.p1` and `pilot.p2` are bound to the same requirements. The items are
`R8E-01`, `R8E-02`, `R8E-03`, `R8E-04`, `R8E-05`, `R8E-06`, `R8E-07` and
`R8E-08` in `revision_round8e`.

PILOT SHORTFALL AMENDMENT, adopted before prereg-v1 and before any pilot scoring or trained-arm training. The amendment is informed by realized original-model sampling and verification counts: P1 13 distinct verified math problems/27 rows, T-fit 16/33, and selection 23/45, from the completed single-backend Modal/vLLM Stage-0-lite run. No trained-arm outcome or pilot ranking is used. This amendment changes pilot domain/eligibility, the conditional maximum attempt count and the P1 continuation gate; it does not change confirmatory numerical endpoints, alpha allocation, or the P2 teacher/block selection formulas.

1. The pilot's verified domain is math plus code passing the frozen Docker verifier. Chat never counts as a verified target. Use only the original selected problem IDs: P1's selected 50 problems, the 86 originally sampled T-fit problems, and the 150 originally sampled selection problems, with their original split membership. These are selected input sets, not retained verified counts. Do not expand to the unused bucket members or to nominal target sizes. Freeze their complete problem/request ID lists before applying this amendment. Math keeps the registered math-verify/normaliser OR rule and timeouts. The Stage 0 filter remains finish_reason=stop, closed thought channel, nonempty answer, rendered prompt-plus-completion length <=8192 tokens, and reasoning_share >=0.75 for math/code. All these conditions and a passing verifier are required for a verified trace.

2. Code verification uses the existing source tests stored for each problem in the frozen Stage 0 pool, with a manifest of the exact ordered test bytes and their problem associations. No newly generated DeepSeek tests, changed expected answers, selectively removed tests or sample-specific test subsets are added under this amendment. Freeze the extraction, functional/stdin wrapper, output comparison and grading implementation, its dependency versions, and the exact Docker image digest. Use CPU-only Docker, network none, read-only root and candidate mount, a 64 MiB /tmp tmpfs, 512 MiB memory and memory-swap limits, 64 PIDs, 1 CPU, user 65534:65534 and no-new-privileges. The registered timeout is 6 seconds per test using the pinned grader's timeout implementation. A code completion is verified only if the nonempty full frozen test set passes, with sandboxed=true. Empty/invalid/missing tests, unsupported execution, candidate timeout or failure, and grader failure do not count as a pass. Infrastructure errors are recorded separately from wrong answers. They may be resolved only by rerunning the identical input bytes under the identical frozen configuration, never by changing limits or selecting a favorable rerun. No unsandboxed fallback is allowed. The currently running code count is usable only if its tested artifacts/configuration match this freeze; otherwise recompute all affected code grades from the same frozen original samples before pilot scoring.

3. Complete code grading and per-request accounting for the original attempts before deciding the extension branch. Count a problem once if at least one of its original samples passes the unchanged filter and verifier. Let nT0 and nS0 be the resulting distinct verified math-plus-code counts in T-fit and selection. P1 is descriptive for every count and does not gate P2. With fewer than 20 verified P1 problems label its estimates small-sample descriptive; with zero, label P1 unevaluable. No P1 metric or diagnostic affects the extension, teacher choice, arm matrix, architecture selection, stopping, or any confirmatory claim.

4. If nT0>=20 AND nS0>=30, use only the original attempts for all pilot corpora and make no extension. Otherwise generate exactly four additional original-model attempts, indices 4,5,6,7, for every math/code problem in each of the original T-fit and selection input sets, irrespective of previous success and irrespective of which split is deficient. Generate no extra P1 attempts and no chat attempts. The additional request count is exactly four times the number of math/code problems in the union of those two original input sets. Commit the full conditional request/seed/configuration plan before generating any of its completions. Do not stop early on reaching a verified-count threshold, add replacement problems, or launch another count-triggered batch.

5. Extra attempts use the same original-model revision, tokenizer/chat template, Modal/vLLM backend/version, numerical precision, pinned sampling image and sampling parameters as the completed lite run. Retain temperature=1.0, top_p=0.95, top_k=64, thinking enabled, and per-request max_new_tokens=8192-rendered_prompt_tokens, with the same rendered-length eligibility rule. For problem_id p and index j the request_id is p+":"+str(j); seed is int.from_bytes(sha256(UTF8("0:"+request_id)).digest()[:8], "little", signed=False) modulo (2**63-1), as in the corrected lite_request_seed implementation. Record every actual seed and request. Keep the original run immutable and write the extension to a separately identified, parent-linked artifact directory/run manifest. A technical resume may complete missing request IDs with the identical plan; never regenerate an already committed completion to choose a different one. All project spend ceilings and guards continue to apply. An infeasible or incomplete required extension leaves P2 pending/not executable, not authorized to use a partial rescue or lower minimums.

6. If the extension branch is taken, finish and verify all planned requests, then form each P2 corpus from the union of eligible original and additional traces. Keep every trace passing the fixed conditions; do not choose a best trace by model/teacher loss or replace an original success with a preferred new one. Order traces by ascending sha256(UTF8("0:"+request_id)), ties by request_id. Count distinct canonical problem IDs, never rows or tokens. P1 uses only its eligible original attempts. Freeze final request/problem IDs, grades, source/domain labels, row counts, scored token counts, rendered tensors/masks and content hashes before any P1 scoring or P2 training. Preserve fit/selection separation and every other reserved-split exclusion. Every compared arm and loop count uses the same applicable frozen rows and mask; an unscoreable required cell is handled by the registered missing-cell/failure policy, not by deleting rows or comparing survivors.

7. P2 can run only if final T-fit>=20 AND final selection>=30 distinct verified problems and all other registered prerequisites hold. If either minimum remains unmet, the pilot is unevaluable for insufficient corpus and P2 does not run under this registration; report the counts and reason. This is not evidence for a mechanism null. There is no automatic second batch, minimum reduction, filter relaxation, domain substitution, bucket borrowing or default teacher/architecture. Any further proposed design change requires a separately disclosed version before affected results, and preserves the original registration and failed adequacy result.

8. Temperature fitting uses the existing registered pooled token-weighted objective, optimizer, bounds, failure policy and separate T-fit split, per arm/checkpoint/N as registered. Selection keeps the existing pooled cNLL estimator, problem-cluster bootstrap, teacher-stage mean-gain statistic and 0.003-nat tie rule, block-matched O-H controls, rule-B eligibility, ranking, monotonicity and tie rules, fixed dose, and complete-cell requirements. All traces from a problem move together in paired resampling. Do not tune domain weights, fit temperatures separately by domain, alter selection statistics, or add favorable interactions under this amendment. Report per-domain problem/row/token composition and descriptive domain-specific results without using them for selection. Multiple traces/tokens are not independent problems. The selected pilot result remains selection-only; confirmatory D1 retains its separate math-only corpus and contract.

9. Freeze this rule and the verifier/test-set specification before the prereg-v1 seal. The immutable rule may precede the completed count/data addendum, as permitted by the staged freeze, but no pilot scoring, P2 training, ranking or student-based data revision is allowed while that addendum is pending. Freeze the final corpus and readiness receipt before the first P1 score and before the first P2 training, therefore before any trained-arm score. Publish the original shortfall, original and final manifests, conditional branch, complete rejection/verification accounting and any technical failures. Absence of trained-arm outcome use is an explicit access/provenance declaration, not a claim that original-model correctness data were unseen.

PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM, adopted 2026-11-06 before prereg-v1, before pilot scoring and before trained-arm training. This addendum is informed by original-model verification records and supersedes conflicting verifier-interpretation and recomputation provisions of the preceding amendment only as stated below. All sampling filters, original problem/request sets, split exclusions, P2 minimums, selection formulas and confirmatory endpoints remain unchanged.

1. A source-conversion defect serialized list-valued APPS/TACO stdin input and expected-output elements as JSON text, which the verifier then treated as literal stdin/stdout text. In the frozen v2 pilot manifest, 30 of 117 code problems have list-looking stdin fields: P1 4, T-fit 11 and selection 15. Twenty-four have kept samples; none has a verified sample. There are 56 kept samples on those problems, including 31 failures with the JSON-list integer-parsing signature. These observations establish the stored-representation problem, but do not by themselves authenticate every original upstream element type.

2. A new v3 interpretation manifest shall bind each relevant original source record to its already pinned dataset revision, record locator, problem identity and ordered test index. It shall record original input/output values and types, separate representation flags for input and output, original stored-test hashes and derived execution-text hashes. Replaying the old conversion must reproduce the frozen ordered test representation. Decoding shall never be inferred from a leading bracket or successful JSON parsing alone.

3. For stdin tests, an authenticated original flat list of strings is decoded once and converted by newline joining its elements, preserving contents, order and empty elements, without adding a terminal newline or stripping stdin. An authenticated original string remains unchanged, including a literal string containing JSON. Expected-output lists receive the same newline conversion. Output acceptance remains equality after the existing normalizer: newline joining the right-stripped lines of text.strip().splitlines(). No additional numeric tolerance, sorting, token-set comparison or alternative-answer rule is introduced. Functional-test semantics remain unchanged.

4. The original pool and all v1/v2 artifacts remain immutable. No source test is added, replaced, reordered or selectively removed. A problem with any required test whose interpretation is unauthenticated, ambiguous, malformed or unsupported is excluded in its entirety from verified-target inclusion, with its complete test inventory and exclusion reason retained. Exclusions are fixed from source/provenance evidence before v3 grading and cannot depend on candidate success.

5. V3 shall freeze the corrected interpretation, executing implementation, dependency versions, Docker content digest, sandbox settings, timeout and infrastructure-accounting policy before its recount. It shall inventory all 468 original code requests and retain accounting for all 1,144 selected requests. Every eligible kept original code sample shall be recomputed against its complete valid frozen test set, irrespective of previous success or failure. Math decisions and sample filters remain unchanged. The completed v3 result alone determines final code membership; successes are not unioned across verifier versions.

6. V2 is accepted as a justified wholesale recomputation of the inadequately accounted v1 count. The request taco:5bc221f9dec75484cb706b33:2 failed in v1 without retained failure detail and passed all 40 tests in v2, with maximum recorded test duration 5.8592191 seconds. This changed selection from 87 to 88 rows while leaving P1/T-fit/selection distinct counts at 16/27/40. The cause of its v1 failure is unknown. Its final membership follows the official v3 result without a special historical veto, favorable-version selection or request-specific retry.

7. The registered timeout remains the existing six-second outer timeout supervising the whole docker run process, including startup. It is not restarted after container or candidate startup. Trusted lifecycle records shall distinguish infrastructure failures from confirmed candidate execution still running at the outer deadline. The latter is a registered outer-deadline failure, without asserting six seconds of candidate execution. Startup/daemon failures, failures after candidate completion and unresolved attribution are infrastructure/unknown. Candidate stdout or stderr alone does not establish lifecycle attribution. No post-deadline result counts, and termination/cleanup must be verified. V2 contains one timed-out request, taco:e25eee8cc5e8629bbbeb6875:0, with 40 timed-out tests and insufficient lifecycle evidence to establish their cause.

8. Within v3, each infrastructure-affected test may receive one retry after documented infrastructure recovery, using identical candidate, test, execution and verifier bytes and identical frozen configuration. Preserve both attempts and use the first conclusive result. Conclusive passes and candidate failures are not retried. A retry that remains infrastructure/unknown leaves the count blocked. Missing infrastructure accounting cannot be treated as zero.

9. Count acceptance requires explicit complete status, exact expected-request coverage and applicable per-test execution/infrastructure accounting. The freeze and count receipt shall bind the actual resolved executing modules and entry points, including shortfall.py and decoding/classification code, rather than merely hashing a named checkout. Corpus/index acceptance shall verify manifest self-hashes and declared file hashes, derive identities and counts from actual files, and establish exact correspondence among official verified requests, original completion contents, raw rows and rendered rows/masks.

10. Before the seal, publish a source/provenance audit and disposition for every use of affected pool tests, including Stage 0 grading, DeepSeek existing-test context and parsing/reference-validation/merge paths, training/reward consumers and evaluation consumers. Active consumers must use authenticated representations. Future or unused consumers may remain explicitly execution-blocked pending the same correction and artifact freeze. Separate unaffected evaluation harnesses shall be identified with supporting inspection or fixture checks. Preserve and disclose any affected historical artifacts. Newly generated DeepSeek tests remain excluded from this pilot corpus.

11. Preserve and disclose v1, v2 and v3 verifier/count/corpus versions, file-byte and canonical/self hashes with their definitions, implementation changes, original-type evidence, exclusions, retry history, every changed request and resulting row/problem/token counts. Recompute the extension decision from the complete official v3 count. If T-fit is at least 20 and selection at least 30 distinct verified problems, make no extension; otherwise the preceding fixed conditional-extension rule continues to apply, with execution blocked until its implementation and prerequisites are complete.

12. The v3 verifier freeze, complete recount, consumer audit/disposition and count/corpus/index guard repairs are prerequisites to both this data addendum and the prereg-v1 seal. Freeze the final common corpus and readiness receipt before any pilot score or P2 training. Declare original-model verification access explicitly; absence of pilot scoring and trained-arm outcome use remains an author access/provenance declaration. Every other previously required seal and execution condition remains operative.

LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION AND V3 RESUME AMENDMENT. Adopted before the official v3 freeze and recount. This amendment explicitly supersedes conflicting lifecycle, blocked-count and continuation provisions of the preceding addendum and X1 verifier-defects ruling only as specified here. It is informed by original-model verification and verifier review. All other registered conditions remain operative.

1. The lifecycle design is approved for the frozen original-model completions to the pilot problems. It does not claim isolation from arbitrary verifier-aware malicious code. User 65534:65534 and the other registered sandbox restrictions remain unchanged. Same-UID interference remains a disclosed residual risk; validation establishes fail-closed treatment of detected corruption, not immunity to every possible compromise.

2. A frozen bootstrap shall report execution entry through a private child-to-wrapper pipe before candidate-controlled execution. The candidate shall receive neither the authentication key nor the evidence descriptor. Freeze the execution-entry boundary and account for unsuccessful or uncertain handoff as infrastructure/unknown. The wrapper shall bind child identity, execution entry, child-status observations and exit status to the attempt and forward lifecycle evidence separately from candidate stdout/stderr. Keyed authentication shall use HMAC-SHA-256 with a fresh per-attempt key and unambiguous authenticated framing and identities.

3. The timeout remains six seconds over the whole docker run, including startup, without restart. An outer-deadline failure requires established execution entry before the deadline and trusted evidence that the identified child had not exited at the deadline. Absence of an exit record, delayed waitpid observation or a finish-record timestamp alone is insufficient. Clock and observation uncertainty spanning the deadline produces infrastructure/unknown. Startup/daemon failure, failed or uncertain handoff, and timeout or transport/teardown failure after candidate completion are infrastructure/unknown. Post-deadline evidence may establish attribution and cleanup but cannot make a candidate answer acceptable.

4. Every conclusive Docker outcome requires complete, valid, ordered and attempt-bound lifecycle evidence appropriate to that outcome. This requirement applies before accepting passes, wrong answers, candidate errors and outer-deadline failures. Invalid or missing lifecycle evidence yields infrastructure/unknown regardless of output or exit code. A pass additionally requires the unchanged output comparison, genuine successful child exit, timely successful completion and verified termination/cleanup.

5. Each infrastructure-affected test receives at most one identical-configuration retry after persisted recovery evidence appropriate to its failure. Preserve both attempts; the first conclusive result controls. Conclusive outcomes are never retried. If the retry remains infrastructure/unknown, record terminal infrastructure-unverified and exclude the affected completion from verified-target inclusion, retaining the infrastructure reason. This is not a candidate failure finding. Retain the complete test inventory and finish accounting for all other scheduled tests.

6. A count with complete coverage and valid attempt accounting may finish as complete_with_terminal_infrastructure when its remaining infrastructure cases have received the terminal disposition above. Report their test/request identities, reasons and split/domain counts. Compute verified problem counts and the unchanged extension decision from the resulting verified completions. Pending retries, incomplete inventories, invalid attempt history, freeze mismatches and unverified termination/cleanup remain blockers. Missing infrastructure accounting is never zero.

7. The single official v3 run may resume under its existing run identity and immutable freeze using a durable append-only attempt ledger. Commit attempt-start before dispatch and evidence/result immediately after completion. An interrupted attempt lacking a valid durable completion is infrastructure/unknown and consumes that attempt. A retry interrupted in this way exhausts the retry allowance. Never regrade a recorded conclusive outcome. Resume only after validating frozen inputs, executing implementation, dependencies, image digest and configuration. Resume later artifact stages from validated completed artifacts without repeating grading. Resume is continuation of the same freeze, not authorization for another freeze or favorable recount.

V3 VERIFIER IMPLEMENTATION FAILURE, VOID DISPOSITION AND SINGLE REPLACEMENT ADDENDUM. This is a disclosed post-incident amendment adopted after the failed official v3 launch and before any replacement official execution. It supersedes the preceding freeze-once, same-implementation continuation and terminal-infrastructure provisions solely for the specifically identified void run and replacement authorized below. All other registered conditions remain operative.

1. The official v3 run created at 2026-11-06T20:23:38.655567+00:00 from commit b5036ac4d7cd28d354aa55b065671df0444d9bdf is designated VOID_VERIFIER_IMPLEMENTATION_FAILURE. Its embedded freeze manifest hash is add7f6f49c17d41873be422bb280c1a8a8b315d83b0cc3f13091e815f34ed184. The production count adapter supplied container_names and attempts to grade_code, whose signature accepted neither. Argument binding failed before that function executed or dispatched pilot candidate code.

2. The failed count accounts for 1,144 selected requests, including 468 code requests and 226 kept code requests. Its ledger records 4,640 scheduled request/test pairs, each with attempts 1 and 2: 9,280 attempt starts and 9,280 infrastructure-error results. These results contain no pilot candidate execution, output, exit or timing outcome. Synthetic infrastructure-recovery containers are separately disclosed. The void determination rests on the verified pre-dispatch implementation failure, not merely on absent lifecycle evidence.

3. Preserve all v1, v2 and failed-v3 artifacts, registries, manifests, ledgers, logs, source snapshots and validation/review records. Do not rewrite the failed count as completed, erase attempts, or relabel infrastructure errors as candidate failures. Publish file-byte hashes and embedded self-hashes with their definitions. Preserve the original v3 registry bytes and record the void disposition in a separate immutable, hash-bound receipt.

4. The void count supplies neither operative terminal exclusions nor final code membership or verified-problem counts. Its extension_required field does not authorize generation. The existing conditional extension plan remains preserved; execute no extension on the basis of the void count. Recompute the branch only from the accepted replacement count.

5. Exactly one replacement official run, identified as v3r, is authorized after all prerequisites in this addendum pass. It is a new freeze with an expressly disclosed implementation change, not a resume under the v3 freeze. Preserve the original candidate/completion bytes, request identities and splits, filters, authenticated interpretation, ordered tests and derived execution bytes, math decisions, comparison rules, six-second outer timeout, sandbox restrictions, image digest, dependency versions and concurrency. No favorable-version union, selective recount or candidate-specific repair is permitted. Any further semantic or environment change requires separate disclosed authorization.

6. Repair the adapter so the count’s container identities and attempt tokens are accepted and forwarded through the real grade_code and DockerSandbox interfaces and agree with durable ledger and lifecycle evidence. Silently dropping these identities is prohibited. Verifier interface/programming errors must block further official dispatch rather than be treated as evidence that a candidate failed. Such blocking does not automatically authorize another void or replacement.

7. Before official v3r execution, add an integration regression through the default production count adapter with the real grade_code and DockerSandbox.run_tests functions and signatures. CPU fixtures may control external process I/O but may not replace these grading functions with permissive fake graders. Test identity propagation, durable accounting and independent loading. Retain and run relevant existing lifecycle, deadline, cleanup, retry and resume regressions.

8. A pre-official real-Docker dry run of the exact production CLI/orchestration and default grading path is mandatory. Use only hand-authored synthetic, non-pilot candidates with separate fixture inputs, registry and outputs. Cover stdin and functional tests, known passing and failing outcomes, and agreement among container names, attempt tokens, lifecycle records, serialized count evidence and independent count loading. Preserve commands, expectations, results, executing-code hashes, environment/image bindings and a synthetic-only, nonofficial validation receipt. Direct sandbox-only tests do not satisfy this requirement.

9. Before v3r is claimed, X1 or an independent reviewer explicitly designated by X1 shall review the concrete repair, integration tests, real-Docker receipt, continuity guards and operative registration, then issue explicit GO for the exact implementation. Implementer self-certification and the previous r13 GO are insufficient. Reconcile Markdown, YAML, schema, decision records and execution/readiness guards before launch. Changes after validation or approval require renewed checks and approval appropriate to those changes.

10. Maintain a fixed-location continuity record linking the preserved crashed v3 registry, void receipt, this ruling, replacement review and distinct v3r registry. Claim v3r atomically and durably before official output or dispatch. Guards shall reject execution under the void identity, alternate-output-directory bypasses, missing or inconsistent continuity evidence, and any additional fresh replacement. Void artifacts remain readable for audit but are rejected as operative count, corpus, index or readiness inputs.

11. The old attempts remain preserved and consumed within the void v3 history. They do not consume v3r’s allowance. Each v3r test starts at attempt 1 and may receive only the previously registered single identical-configuration infrastructure retry after appropriate persisted recovery evidence. The first conclusive result controls; conclusive outcomes are not retried. Existing terminal-infrastructure, cleanup, inventory and attempt-integrity rules apply within v3r. Resume of v3r requires its same identity, immutable freeze and executing implementation. This amendment authorizes no automatic v3r2 or further reset.

12. V3r shall retain complete accounting for all 1,144 selected requests, including all 468 code requests, and recompute every eligible kept code completion against its complete frozen test set. Preserve unchanged math decisions and filters. The accepted v3r result alone determines final code membership and the unchanged conditional extension decision. Publish exclusions, retries, infrastructure dispositions, changes from previous versions, and final row/problem/token counts. All standing corpus/index validation, consumer-audit, data-addendum, readiness and seal prerequisites remain required.

13. Disclose this integration defect, failed official launch, review and test coverage gap, synthetic recovery activity, void decision and replacement authorization. Publish a dated author declaration of access to historical original-model verification, any pilot candidate execution outside the preserved records, and any pilot scoring or trained-arm training/outcome access. Replacement authorization depends on no undisclosed affected execution or outcome use; contrary evidence requires renewed X1 disposition before proceeding.

The staged freeze is explicit: `prereg/pilot-shortfall-freeze.json` is
**REQUIRED before any pilot scoring or P2 training**. It is filled by the data
addendum after the official count; the immutable rule and verifier/test-set
specification may be sealed before it, as the panel's ruling permits. The
math-only pilot clauses from rounds 8c/8d remain only as dated history and are
not operative beside this amendment. Confirmatory D1 retains its separate
math-only corpus and contract.

### Revision round 8f (verifier interpretation and recomputation addendum)

The X1 ruling's **PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM**
is registered verbatim in `prereg/pilot-shortfall-amendment.md` and is
reproduced above. It is operative in addition to the round-8e amendment, and
the typed fields in `pilot.verifier_v3` carry the same requirements:
`interpretation` (v3 interpretation manifest, per-test input/output
representation flags from authenticated original types, the whole-problem
exclusion rule), `freeze` (v3 freeze contents, full recount of the 468 original
code requests and accounting for all 1,144 selected requests, v3-only
membership), `v2_recomputation` (v2 accepted as a wholesale recomputation of
v1), `timeout_lifecycle` (the unchanged six-second outer timeout, lifecycle
attribution classes and the single precommitted identical-configuration
infrastructure retry), `acceptance` (count/freeze/index acceptance, r2
Important 1-3), `consumer_audit` (the pool-test consumer audit and disposition
as a seal prerequisite), `hashes` (v1/v2/v3 preservation and disclosure) and
`prerequisite_order`.

**Recorded file-byte SHA-256 facts (v1 | v2).** `verifier_freeze.json`:
`3118b6b0af149eda243d416bdbf98370437cd923a36c193b895c74bd5c8234e1` |
`e53a7269d13988cfc090696ed0436c29bfa1c17fba5a443cc834917aa71dd39d`.
`count_manifest.json`:
`17198e44be40690bc1e13b85d6c41bde8c181358390e10fd3246cf8dec89df67` |
`ad1cd242931271fcfa1e88242730b87ecc5a4a2d217fce4b4420568901402c93`.
`per_request.jsonl`:
`de799a68be62e1aaec701a6afd79bee618584b1d0cc21606e9df9c6da4d33730` |
`81818b0229455b7b02b229e1f0f8fc0d12dbf27f7927733518479ecaa973683d`.
The common v1/v2 test-manifest file hash is
`3f9d56317ae8a27cab59aca8f6a401fa3254edd56fc44dce13d5a7ac7f5e7e2e`.
These are file-byte hashes, distinct from embedded canonical/self hashes.

**Superseded history, not operative.** The earlier clause stating that "the
currently running code count is usable only if its tested
artifacts/configuration match this freeze" is superseded by this addendum; v3
must be frozen and fully recounted before both the data addendum and the seal,
and the preview count is not a decision input. Any statement treating v1 or v2
as final is likewise superseded: v1 is an immutable prior version with an
inadequately accounted count, v2 is accepted only as a wholesale recomputation
of v1, and **v3 alone determines final code membership**. The request
`taco:5bc221f9dec75484cb706b33:2` is recorded as a fact: it failed in v1
without retained detail, passed all 40 tests in v2 (maximum recorded test
duration 5.8592191 seconds), changed selection from 87 to 88 rows, and left the
distinct counts at P1 16, T-fit 27 and selection 40; the cause of its v1
failure is unknown. Its final membership follows the official v3 result.

**Prerequisite ordering.** The v3 verifier freeze, complete v3 recount,
pool-test consumer audit/disposition and count/corpus/index guard repairs are
prerequisites to both the data addendum and the prereg-v1 seal; the final
common corpus and readiness receipt are frozen before any pilot score or P2
training. `prereg/pilot-shortfall-freeze.json` remains a pending skeleton and
now carries the corresponding v3 sections.

### Revision round 8g (lifecycle evidence, terminal infrastructure disposition and v3 resume)

The X1 ruling's **LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION
AND V3 RESUME AMENDMENT** is registered verbatim in
`prereg/pilot-shortfall-amendment.md` and reproduced above. It supersedes the
conflicting lifecycle, blocked-count and continuation provisions of the
preceding addendum and X1 verifier-defects ruling only as specified there; all
other registered conditions remain operative.

**Threat model.** The lifecycle design is approved only for the frozen
original-model completions to the pilot problems. It does not claim isolation
from arbitrary verifier-aware malicious code; user `65534:65534` and the other
sandbox restrictions remain unchanged. Same-UID interference remains a
disclosed residual risk. Validation establishes **fail-closed treatment of
detected corruption**, not immunity to every possible compromise; correct
output, genuine child exit, timely completion, complete lifecycle validation
and cleanup remain independent pass requirements. This is not candidate-proof
isolation.

**Lifecycle evidence.** A frozen bootstrap performs verifier setup before the
execution-entry record, reports execution entry through a private
child-to-wrapper pipe before candidate-controlled execution, closes the
evidence descriptor before candidate execution and leaves no key in
candidate-accessible arguments, environment, files, globals or live runner
frames. The candidate receives neither the key nor the descriptor. `exec`
handoff is distinguished from failed or incomplete handoff. The wrapper binds
child identity, execution entry, child-status observations and exit status to
the attempt and forwards lifecycle evidence separately from candidate
stdout/stderr. Keyed authentication uses actual **HMAC-SHA-256** with a fresh
per-attempt key and unambiguous authenticated framing and identities.

**State-specific validation.** Every conclusive Docker outcome — pass, wrong
answer, candidate error or outer-deadline failure — requires a complete, valid,
ordered and attempt-bound record appropriate to that outcome. Validation
precedes acceptance at the grader, counting classifier and count-loading
boundary. Missing, malformed, unauthenticated, duplicated, inconsistent or
improperly ordered evidence yields infrastructure/unknown regardless of stdout
or exit code.

**Outer deadline.** The timeout remains six seconds over the whole `docker run`,
including startup, without restart. An outer-deadline failure requires
established execution entry before the deadline and trusted positive evidence
that the identified child had not exited at the deadline. Absence of an exit
record, a delayed `waitpid` observation or a finish-record timestamp alone is
insufficient; clock and observation uncertainty spanning the deadline produces
infrastructure/unknown. Startup/daemon failure, failed or uncertain handoff,
and timeout or transport/teardown failure after candidate completion are
infrastructure/unknown. Post-deadline evidence may establish attribution and
cleanup but cannot make a candidate answer acceptable.

**Terminal infrastructure disposition.** Each infrastructure-affected test
receives at most one identical-configuration retry after persisted recovery
evidence. Both attempts are preserved and the first conclusive result
controls; conclusive outcomes are never retried. If the retry remains
infrastructure/unknown, the test is recorded **terminal
infrastructure-unverified** and the affected completion **not verified**,
retaining its infrastructure reason. It is not relabelled as wrong answer,
candidate error or timeout, and another completion for the same problem may
still qualify through its own complete passing test set. This prospectively
replaces the previous "the count stays blocked" rule, which is retained only
as **SUPERSEDED HISTORY**. The extension decision is computed from the
resulting verified counts; no extra retry, replacement problem or special
rescue follows.

**Count status.** A count with complete coverage and valid attempt accounting
may finish as `complete_with_terminal_infrastructure`. It reports the
terminal test/request identities, reasons and split/domain counts, and computes
verified problem counts and the unchanged extension decision from the resulting
verified completions. Pending retries, incomplete request/test inventories,
invalid attempt history, freeze or configuration mismatches and unverified
termination/cleanup remain blockers. Missing infrastructure accounting is never
zero.

**Resume.** The single official v3 run may resume under its existing run
identity and immutable freeze using a **durable append-only attempt ledger**.
Attempt-start is committed before dispatch and evidence/result immediately after
completion. An interrupted attempt lacking a valid durable completion is
infrastructure/unknown and consumes that attempt; an interrupted retry exhausts
the allowance and creates no third attempt. A durably recorded conclusive result
is never regraded. Resume requires validation of frozen inputs, executing
implementation, dependencies, image digest and configuration; later artifact
stages resume from validated completed artifacts without repeating grading.
Resume is continuation of the same freeze, not a second freeze or a favorable
recount.

**Pool-test consumer dispositions (pre-seal).** The audit
`C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/2026-11-03-gemma-loop/audit-pool-test-consumers.md`
summary-table rows 1–16 are registered in
`pilot.verifier_v3.consumer_audit.dispositions`. Active and affected consumers
are superseded by v3: the grader (row 4), the pilot shortfall verifier and its
v1/v2 counts (row 6), and `corpus_v2` membership (row 7). Active but unaffected
consumers may be retained: `modal_lite/sample_grades.jsonl` (row 5; code
verification was disabled and `kept` is filter-only), heal/training corpora
(row 8), P1-lite (row 9), pilot NLL scoring runtime (row 14) and the
LiveCodeBench official evaluator (row 15). Future affected consumers are under
an explicit registered execution block until they use the authenticated
representation: `stage0/postprocess.py` verified corpora and Stage 2 prompts
(row 2), local `--verify-code` sampling (row 3), the DeepSeek existing-test
prompt context (row 10) and hidden-test merging (row 13). Future unaffected
DeepSeek parsing/reference-validation paths remain under the writer-wide block
(rows 11–12), and no RL/reward consumer is active (row 16). The original pool is
preserved byte-for-byte (row 1); a corrected representation must be a new
hash-bound artifact. The block is lifted only after the authenticated
representation is bound and generated tests remain excluded from the pilot
shortfall corpus.

**Tests and gate.** `tests/test_prereg.py` covers the new fields, both
addendum texts verbatim, and schema rejection of unknown/missing/wrong values.
`scripts/check_prereg_loads.py prereg/prereg.yaml` and its `--tag-mode` both
exit 0, and no pending-pin or alignment placeholder is present in the
registered documents.

### Revision round 8i (void v3 verifier implementation failure and single replacement v3r)

The X1 ruling's **V3 VERIFIER IMPLEMENTATION FAILURE, VOID DISPOSITION AND
SINGLE REPLACEMENT ADDENDUM** is registered verbatim in
`prereg/pilot-shortfall-amendment.md` and is reproduced above. It is operative
in addition to the round-8e/8f/8g amendments, and the typed fields in
`pilot.verifier_v3.void_v3`, `pilot.verifier_v3.replacement_v3r` and
`pilot.verifier_v3.author_access_declaration` carry the same requirements.

**The void v3 fact.** The official v3 run created at
`2026-11-06T20:23:38.655567+00:00` from commit
`b5036ac4d7cd28d354aa55b065671df0444d9bdf` is designated
`VOID_VERIFIER_IMPLEMENTATION_FAILURE`. Its embedded freeze manifest hash is
`add7f6f49c17d41873be422bb280c1a8a8b315d83b0cc3f13091e815f34ed184`. The
production count adapter passed `container_names` and `attempts` to
`grade_code`, whose signature accepted neither; argument binding failed before
that function executed or dispatched pilot candidate code. The failed count
accounts for 1,144 selected requests, including 468 code requests and 226 kept
code requests, and its ledger records 4,640 scheduled request/test pairs, each
with attempts 1 and 2: 9,280 attempt starts and 9,280 infrastructure-error
results. Those results contain no pilot candidate execution, output, exit or
timing outcome; synthetic infrastructure-recovery containers are separately
disclosed. The void determination rests on the verified pre-dispatch
implementation failure, not merely on absent lifecycle evidence.

**File-byte identity facts.** These are file-byte SHA-256s, distinct from
embedded self-hashes:

| Artifact | SHA-256 |
|---|---|
| `v3_registry.json` | `5e103b898ec9aac6a5b862cda4d607761c5bda815b701366625ab45823ce8a12` |
| `verifier_v3/verifier_freeze.json` | `3eed1f0be765e609e13f3f91460f3a15f1ff032cec1d1373c294477f914e1ea0` |
| `count_v3/count_manifest.json` | `5b0cd11b9b612914de865eb06b6419bab3376f01eb07380ed6ffd2134e854ea2` |
| `count_v3/attempt_ledger.jsonl` | `7d0e9fb47c6e73218f274cb8305590861bbbf763f33f1d9b43d09ad5455fced1` |

**Disposition and preservation.** The whole failed count is void. Its
infrastructure records remain the historical record of a failed official
launch and are neither erased nor relabelled as candidate failures. All v1, v2
and failed-v3 artifacts, registries, manifests, ledgers, logs, source snapshots
and validation/review records are preserved; `v3_registry.json` is preserved
byte-for-byte in its recorded crashed state, and the void disposition is
recorded in a separate fixed-location, immutable, hash-bound receipt.

**Non-operative extension.** The void count's `extension_required=true`, with
math-only distinct counts of T-fit 16 and selection 23, is **not** an operative
extension decision or authorization from an accepted count. No extension is
executed on the basis of the void count; the branch is recomputed only from the
accepted replacement count.

**Scoped supersession.** This addendum supersedes the preceding "freeze v3
once", same-implementation continuation/resume and terminal-infrastructure
provisions **solely** for the specifically identified void v3 run and the
single authorized v3r replacement. All other registered conditions remain
operative.

**Single authorized replacement, v3r.** Exactly one replacement official run,
identified as **v3r**, is authorized only after every prerequisite passes. It
is a new freeze with an expressly disclosed implementation change, not a
resume under the v3 freeze; no seal is granted and the registration is not
approval to launch v3r now. Its prerequisites are: (1) repair identity
forwarding, not just keyword acceptance, so container names and attempt tokens
reach the real sandbox through the real `grade_code` and `DockerSandbox`
interfaces and match durable ledger/lifecycle evidence; (2) add a
production-path integration regression through the default count adapter with
the real `grade_code` and `DockerSandbox.run_tests` functions and signatures,
without permissive fake graders; (3) perform a mandatory pre-official
real-Docker dry run of the exact production CLI/orchestration and default
grading path with synthetic non-pilot candidates, separate fixtures, stdin and
functional tests, passing and failing cases, identity propagation and a
preserved nonofficial receipt; (4) obtain X1 or a recorded X1-designated
independent review with explicit GO for the exact implementation; (5) freeze
v3r once with the same original candidates, requests, filters, authenticated
interpretation, ordered tests, math decisions, sandbox restrictions, image
digest, dependencies, concurrency and six-second timeout, disclosing every
implementation/configuration delta; and (6) recount the complete original
population of all 1,144 requests, including all 468 code requests, over the
complete frozen test sets, with v3r alone determining final code membership and
the extension branch. A fixed-location continuity record links the preserved
crashed v3 registry, void receipt, ruling, replacement review and a distinct
`v3r_registry.json`; v3r is claimed atomically and durably before official
output or dispatch, and guards reject the void identity, alternate-directory
bypasses, missing or inconsistent continuity evidence and any further fresh
replacement. The old attempts remain consumed within the void v3 history and
do not consume v3r's allowance: each v3r test starts at attempt 1 with only the
previously registered single identical-configuration infrastructure retry.
There is **no automatic v3r2** or further reset.

**Author access declaration.** Paragraph 13 requires a dated author
declaration of access to historical original-model verification, any pilot
candidate execution outside the preserved records, and any pilot scoring or
trained-arm training/outcome access. The registration records this field with
status `pending author confirmation`; its content is deliberately blank and is
not invented here.

**Tests and gate.** `tests/test_prereg.py` covers the new fields, the verbatim
addendum text, the void facts and hashes, the replacement prerequisites and
scoped supersession, the pending author declaration, and schema rejection of
unknown/missing/wrong values. `scripts/check_prereg_loads.py prereg/prereg.yaml`
and its `--tag-mode` both exit 0, and no pending-pin or alignment placeholder is
present in the registered documents.

### Revision round 8h (data addendum)

The data addendum registers everything required before the prereg-v1 seal
**except the v3r count results**, which the single official v3r run fills in a
later small step.  Stage A was exposed before this round; no trained-arm
outcome informed it.  The typed copy is `revision_round8h` in `prereg.yaml`.

**Teacher runtime outcomes.**  The realized qualification results are facts,
not decision knobs.  E4B bf16 failed the allocator-headroom check
(**0.1748 < 0.25 GiB**; `qualification/e4b-bf16/qualification.json`, SHA-256
`a28f19331d941fcd3f73219d1d92844ec71c1bd65e0d0901506fe76931acfa06`).  E4B
int8 **PASSED**: mean full KL **0.00438**, p95 **0.0231**, top-1 **0.9821**,
confident top-1 **0.99870**, allocator headroom **3.19 GiB**
(`qualification/e4b-int8-r2/qualification.json`, SHA-256
`4a4668248b27c0614260cc53350a6245d66c4db4147a0959797f7b2aa0a9dc75`).  12B
NF4 **failed fidelity after expansion** (mean KL **0.0239**, upper-95
**0.0264**, p95 **0.118**, confident top-1 **0.9889**;
`qualification/12b-nf4/qualification.json`, SHA-256
`1d58b8342be2a44a0d7927e5c8186949b26af13967c49cd17f38fac4a128512f`).  The
12B official QAT Q4_0 GGUF executed through **llama.cpp b12169** (commit
`3ed7284f2d9e22b2d1a7d05c1416c00c5a9f0bc7`; GGUF SHA-256
`93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`; ctypes
implementation branch **task/gguf12b** commit **5557d6c**) passed every overall
check (mean KL **0.00672**, p95 **0.0328**, top-1 **0.9780**, confident top-1
**0.99759**, headroom **2.35 GiB**;
`qualification/12b-q4_0-gguf/qualification.json`, SHA-256
`dee39752d1cdbb641baa2bf1abdc646e3a0da8b84848481d242bbd6787705ea7`), but it
failed the registered per-stratum rule on the short stratum (mean KL
**0.0130 > 0.010**, p95 **0.0628 > 0.050**).  The short stratum is
**~1.6-2x worse for every runtime** (NF4 0.044 vs 0.023; GGUF 0.0130 vs
0.0066; E4B int8 0.0053 vs 0.0034), so this is inherent to short chat rows,
not a GGUF-specific defect.  Under the registered fallback order both 12B
runtimes fail, so **12B has no qualified resident runtime and is EXCLUDED**;
the threshold was not relaxed and no representation was substituted.  The
author decided on **2026-11-06** not to add any exploratory 12B arm.  The
teacher step therefore has a **single qualified teacher, E4B int8**.  The
frozen runtime record is
`C:/Users/<user>/ML/gemmaloop-artifacts/qualification/teacher_runtime_freeze.json`,
SHA-256 `d98574294230c8afc1c7b900a7906ad7736e53c667aab5891f51db35f7a5ecc9`.

**GPU probe-memory smoke.**  The healpilot audit requires a real training-shape
memory check.  It **PASSED**: the N=8+generation phase reserved **6.79 GiB**
with **3.49 GiB** allocator headroom and produced **24/24** generation rows
(`C:/Users/<user>/ML/gemmaloop-artifacts/smoke/probe_memory_auditfix1_r2`;
`probe_memory.json` SHA-256
`94215907201be1439b7435bca554afe2dc6653d5749d9316c84cfec0a72b935d`).  The
disclosed implementation repairs are: probe inputs on the text-embedding
device; probe micro-batch **1** with token-weighted aggregation; exit-1
bit-exactness on the registered fixed **4** rows; single-pass NLL temperatures;
generation with the offloaded towers temporarily on the compute device; and the
frozen teacher runtime enforced by the harness.  The probe inputs are
synthetic; no pilot, MONITOR, T-fit, selection, confirmatory or evaluation
manifest is read.

**Pool-test consumer dispositions.**  The audit dispositions are already
registered in **round 8g** under
`pilot.verifier_v3.consumer_audit.dispositions`; this round references them and
does not re-register or change them.  The audit file is
`C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/2026-11-03-gemma-loop/audit-pool-test-consumers.md`,
SHA-256
`f861612ac6634e86063ef5c8112f987364f3ac0c5cef5a568552ba0c4dcd7be6`, and its
summary-table rows 1-16 remain the registered disposition map.

**Author declaration (X1 void ruling, paragraph 13).**  The author confirmed the
declaration on **2026-11-06**; the file is
`C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/2026-11-03-gemma-loop/author-declaration-2026-11-06.md`,
SHA-256
`aaae7fdd33dd5fff63a620c943e24d0fc27a9318d23f87e75405599f863c16af`.  The
round-8i field `pilot.verifier_v3.author_access_declaration` is set to
`confirmed` and carries the English rendering verbatim:

> 1. The author has seen aggregate original-model verification counts (the Docker preview; v1/v2 counts: T-fit 27, selection 40, P1 16; Stage-0-lite math counts). Per-request records were inspected by the orchestrator and the reviewers.
> 2. There was no pilot candidate execution outside the preserved records (the codeverify preview, v1, v2 and the void v3 run).
> 3. No pilot scoring has been performed, no looped pilot arm has been trained, and no trained-arm outcome has been seen.
> 4. Outside the pilot there were: the untrained Stage A evaluations (R and R-damped), the E4B teacher precompute, and a single optimizer step on 128 synthetic tokens for the probe-memory smoke, which used no pilot data and produced no evaluation outcome.

**Part 2 exploratory scaling curve (pre-specified).**  Before any pilot result,
the author fixed the exploratory scaling plan: train the **best pilot dial arm
and its block-matched O-H** at **12M tokens (the registered point), 100M and
300M tokens** with a warmup-stable-decay schedule whose **100M point is a decay
branch of the 300M run**.  Evaluation is the **same as the confirmatory part**.
The whole curve is reported as **exploratory**.  The data source and budget are
decided later and disclosed; this is a plan, **not a money commitment**.

**v3r section: complete from the single official run.**  The typed block
`revision_round8h.data_addendum.v3r` has status **`complete`** and an empty
`remaining_fields`.  The single official v3r run (the authorized replacement
for the void v3 verifier-implementation failure) executed once from commit
`2d44dff4802c4dad6c71068de35605f82aeb30d4` with
`CUDA_VISIBLE_DEVICES='' uv run --offline python scripts/freeze_recount_v3r.py`.
Its registry is `complete` and was not resumed (`attempt_base` 1,
`resume_count` 0).  The production loader `load_verified_count` accepts the
count with status `complete`.

**File-byte and embedded hashes.**  The YAML hash fields record the embedded
canonical self-hash where the artifact defines one; the ledger and corpus files
record file-byte SHA-256.  Both identities are published here:

| Artifact | File-byte SHA-256 | Embedded self-hash |
|---|---|---|
| `v3r_registry.json` | `a8deb6dcd585ccddb958f3c5612ac53d0fadb92c414f52b4e0f2ca645931cb9e` | `registry_sha256` `8e2926c090244398b68d80f34e6b767fdbcd30d621cd294367f4172628fd4c00` |
| `verifier_v3r/verifier_freeze.json` | `f6a28bb2b2dde28acecb42c3140b000a11180a14731b2c27658c97dd6a9c00a6` | `manifest_sha256` `446c0e7ecca29acb6cd2b6c7d7b2acb493ef090249483af3066f232196904533` |
| `count_v3r/count_manifest.json` | `199083684bc0f9baa25ce29cefbcc27cbccd6ceebc3bb0642279add711e6510f` | `manifest_sha256` `791159330e455206ea2807dd296a546b05d8d20877426ba990388b7046767d87` |
| `count_v3r/attempt_ledger.jsonl` | `05b20931d25ade162d577003322dccbdcf4757ae0e020f0001d628daf70b7bc5` | (file-byte identity) |
| `corpus_v3r/manifest.json` | `86d1c8f96b3fa47b3ca446797deab4d640f6b29f5de8117be3a52cca4fd7a109` | `manifest_sha256` `ffbeb71a590c7f976e2ffdd6a8bbeb65505212c4a4d6d937de980661429af7e4` |
| `pilot-shortfall-freeze-v3r.json` | `3d214968988b250e122d298967bcbeba1856b28cc5ec8453e1d3c67db585e891` | `index_sha256` `4a5b44347cebdd3933b35ac8481d1f4ed4f630028c8545efb4945161dcd15643` |

The per-file corpus hashes recorded in the YAML are: `pilot_p1.jsonl`
`a0ccbed394025c4be9f7e03a5b3eca6f0795a6e64dfc0520626f7d7251b48dc3`,
`pilot_p1.rendered.jsonl`
`95d7fcdb9672af9dade3d9590ea270987fe9ffeb29a0c56743d8f848e25076cd`,
`pilot_selection.jsonl`
`72315be23e148b1b0dba0ea63160c61f73855876d493b987158b0605aed76e1b`,
`pilot_selection.rendered.jsonl`
`07db56755ad1beb9e44d83bb900014238ec2368cbabc7bb5bab19aba7e312f7d`,
`pilot_tfit.jsonl`
`69d64dfdde8397933db0639ff46b98d12996a8aa6e14cf157118f7a5d71969b5` and
`pilot_tfit.rendered.jsonl`
`8f700c66080b13da547c8662c92fad9fa9badb878676c1ea2cc78798e7430631`.
The count manifest binds the count summary (`summary.json`, file-byte
`9148ab9a6f1c5519deb606e3114ab079891ff16c9e801d16456f19c76b42bf0a`), the
per-request rows, the grade log and the durable attempt ledger.

The addendum's `hashes.freeze_index_sha256` records the embedded
`index_sha256` of the index emitted by the accepted official v3r run
(`v3r/pilot-shortfall-freeze-v3r.json`).  The completed addendum index at
`prereg/pilot-shortfall-freeze.json` is a byte-verified rebuild from the same
v3r outputs and the final registration bytes; it binds the final T-fit and
selection rendered files (byte-identical to the v3r corpus files above), its
count status is `complete`, its `run_version` is `v3r`, and it passes
`check_freeze_readiness`.  Its own embedded `index_sha256` and file-byte hash are
reported in the task report.  Recording the rebuild's hash inside the
registration that the rebuild binds would be circular; the accepted-run index
is the stable identity.

**Distinct verified problems by domain.**  T-fit: math 16, code 16, total 32.
Selection: math 23, code 28, total 51.  P1 (descriptive, never gates): math 13,
code 6, total 19.  The count accounting is complete: 1,144 selected requests
(468 code requests, 226 kept code requests), 226 kept code requests graded,
121 passed, 167 answer failures, 1 timeout request, 0 infrastructure failures,
0 terminal infrastructure cases and 0 interpretation exclusions.  No
interpretation exclusion was needed: all 117 code problems were authenticated
from their pinned source records and the old converter replay agreed.

**Extension decision.**  The registered rule is: if final T-fit >= 20 AND final
selection >= 30 distinct verified problems, use only the original attempts and
add zero requests; otherwise add exactly four attempts at indices 4-7 for every
math/code problem in both original P2 input sets, for one batch only.  The
accepted v3r counts meet both floors (32 and 51), so `extension_required` is
`false`, `additional_requests` is `0`, and the decision is
`no_extension_original_only`.  The decision was recomputed from the v3r counts
with the production `decide_extension` rule and agrees with the count summary.
The void v3 count's `extension_required=true` (T-fit 16, selection 23) remains
non-operative.

**Terminal infrastructure, retries and timeout.**  The v3r count status is
`complete`, not `complete_with_terminal_infrastructure`: the terminal inventory
is empty (0 tests, 0 requests, 0 by domain, 0 by split) and every scheduled
test finished with one conclusive outcome.  There was exactly one retry: test
index 7 of request `taco:e25eee8cc5e8629bbbeb6875:0` (pilot selection, code)
received its single identical-configuration infrastructure retry after persisted
startup-recovery evidence; the first attempt was infrastructure/unresolved and
the retry was a conclusive host-route timeout (client hard-kill, `docker rm -f`
and `docker kill -s KILL` all returned within the frozen 0.25 s bound; host exit
after the deadline).  The first conclusive result controlled and no conclusive
outcome was regraded.  The request's remaining 39 tests were also conclusive
timeouts with no retry.  No test or request retained a terminal
infrastructure-unverified disposition, and no cleanup remained unverified.

**Version deltas.**  v1 to v2: corpus rows 183 -> 184 (+1), distinct problems
83 -> 83, scored-token denominator 906,978 -> 913,624 (+6,646), assistant scored
positions 861,725 -> 867,948; one changed request,
`taco:5bc221f9dec75484cb706b33:2`, flipped fail to pass (code samples passed
78 -> 79).  v2 to void v3: v3 produced no corpus or index (void, non-operative;
its count is blocked-infrastructure with 226 infrastructure failures and zero
candidate executions), so corpus rows/tokens go to zero; its void count
recognized 52 distinct math-only problems and no code, and 79 v2-verified code
requests left the verified set while the math set was unchanged.  v3 to v3r:
corpus rows 0 -> 226 (+226), distinct problems 52 -> 102 (+50), scored-token
denominator 0 -> 1,132,822 (+1,132,822), assistant scored positions 0 ->
1,061,067; all 121 v3-verified code requests were added and the 105
v3-verified math requests were unchanged.

**Corpus row/problem/token counts.**  Totals: 226 rows, 102 distinct problems,
1,132,822 rendered input tokens and 1,061,067 assistant scored tokens.  P1: 38
rows, 19 problems, 178,596 input tokens, 169,794 scored tokens.  T-fit: 78 rows,
32 problems, 375,644 input tokens, 352,868 scored tokens.  Selection: 110 rows,
51 problems, 578,582 input tokens, 538,405 scored tokens.  By domain (rows /
distinct problems / input tokens / scored tokens): P1 code 11 / 6 / 64,464 /
58,858 and math 27 / 13 / 114,132 / 110,936; T-fit code 45 / 16 / 198,163 /
179,345 and math 33 / 16 / 177,481 / 173,523; selection code 65 / 28 / 338,687
/ 303,483 and math 45 / 23 / 239,895 / 234,922.  `input_tokens` is the rendered
input-token count (`sum(len(input_ids))`, equal to the manifest's rendered
scored-token denominator); `scored_tokens` is the assistant-label mask count.

**Void v3 and paragraph 12-13 disclosure.**  The void v3 run, its
`VOID_VERIFIER_IMPLEMENTATION_FAILURE` determination, its four file-byte
artifact hashes, the pre-dispatch `grade_code`/adapter defect, the 1,144-request
/ 9,280-attempt accounting, the synthetic recovery containers, the preservation
and the author declaration are registered in round 8i
(`pilot.verifier_v3.void_v3`, `pilot.verifier_v3.replacement_v3r` and
`pilot.verifier_v3.author_access_declaration`) and were not rewritten by this
round.  This round adds the accepted v3r result as a fact: one replacement run,
launched only after the recorded X1 GO at `2d44dff`, completed once, not
resumed, with the exclusions, retry, infrastructure and terminal
dispositions, version deltas and final row/problem/token counts above.

**Tests and gate.**  `tests/test_prereg.py` covers the typed teacher, probe,
consumer, author and scaling blocks, the verbatim declaration, the
`awaiting_v3r` emptiness rules (on a synthetic awaiting copy), schema rejection
of unknown/missing/wrong values, and both completed tag states: plain mode and
`--tag-mode` both exit **0** for the committed document.  No tag placeholder
token is present in the registered round-8h text.

### Revision round 8j (pilot training hardware plan)

This is a **pre-seal operational amendment decided by the author on
2026-11-07, before any pilot training or scoring**.  Stage A was exposed before
this round; no trained-arm outcome informed it.  The typed copy is
`revision_round8j` in `prereg.yaml`.

**Provenance and resource basis.**  The hardware plan is informed by
**resource evidence measured on synthetic data before the decision**, never by
pilot data, a trained pilot arm, a pilot score or any pilot outcome.  The
earlier round-8j sentence saying that no hardware measurement was used is
corrected; `hardware_plan.provenance` now registers the basis explicitly:

- **trainsmoke** (synthetic-data GPU rehearsal; ledger `progress.md`, 20:52):
  the 25-29 pilot train path ran ~**580 tok/s** over 229,616 tokens with a
  **9.39 GiB** reserved peak, checkpoints/resume and the end probe PASS.
- **blocksmoke** (`.superpowers/sdd/2026-11-03-gemma-loop/task-blocksmoke-report.md`):
  the block 15-29 probe PASSed at **6.85 GiB**, but training ran at only
  **167.6 tok/s / 9.99 GiB** reserved (0.28 GiB headroom) and **step 2 stalled**
  for more than 14 minutes.
- **block1529** (`.superpowers/sdd/2026-11-03-gemma-loop/task-block1529-report.md`):
  block 15-29 sits on a **hard VRAM-capacity cliff on the 12 GB card**; no
  allowed allocator setting was stable at the registered configuration.
- **Modal price check** (ledger `progress.md`, 00:40): **H100 USD 3.95/h** and
  **A100-80GB USD 2.50/h**.
- **Credit-based cost estimate** (ledger `progress.md`, 01:00): author credits
  **USD 26.40**, internal cap **USD 24**.

All five items are synthetic-data, price or credit evidence.  None is a trained
pilot outcome, pilot score or pilot data point.  The typed block records the
correction date, the corrected prior claim and each report path.

**Platforms and identical execution.**  Pilot training children may run on the
**local RTX 5070 12 GB** or on **Modal NVIDIA H100 SXM5 80GB** containers, one
arm per container.  Every arm executes the **identical code commit,
configuration, hyperparameters, effective batch, frozen E4B int8 teacher
shards** (computed locally once), **heal corpus** and **MONITOR rows**.
**Teacher precompute, evaluation, readiness, temperature fitting and selection
run locally for every arm.**

**Block-group assignment.**  Hardware is assigned by **block group, never by
individual arm**:

| Group | Block | Arms | Platform |
|---|---|---|---|
| A | 15-29 | `L-15-29-gch-e4b`, `L-15-29-scalar-e4b`, `OH-15-29-e4b` | Modal |
| B | 16-18 | `L-16-18-gch-e4b`, `OH-16-18-e4b` | local by default; moves to Modal only if the remaining Modal credits cover it before it has started anywhere |
| C | 25-29 | `L-25-29-gch-e4b`, `OH-25-29-e4b` (teacher-stage pair) | local, together with D |
| D | 25-29 | `L-25-29-gch-e4b-single-exit-0901`, `L-25-29-gch-e4b-single-exit-n4` (single-exit ablations) | local, together with C |

Every L arm and its block-matched O-H control therefore share one hardware
platform.  Assignment depends only on **cost/credits and availability**, is
decided before the group starts, and never on any observed result.

**Run identity and disclosure.**  Each arm's run identity records the
**hardware, image, library versions and allocator cap**.  The report discloses
the platform of every arm.  Cross-block selection compares **within-platform
L-minus-O-H gains**.  This is disclosed as a limitation: **bf16 kernels differ
across GPUs**, so numbers produced on the RTX 5070 and on the H100 are not
bit-identical even under the same code and configuration.

**Money guard.**  Modal spend for pilot training is capped **below** the
author's remaining free credits (**USD 26.40 at decision time; internal cap
USD 24**) with the existing **monthly ledger** and **shared spend lock**.  The
author's Modal workspace limit is the **hard stop**.  If a Modal arm cannot
finish within the cap, its **whole group is rerun locally from scratch** (never
mixed) and that is disclosed.

**Unavailable-Modal fallback.**  If Modal is unavailable, **all groups run
locally** (the registered default).

**Decision log.**  The round records five dated author decisions: (1) the two
permitted platforms and the identical-execution rules; (2) block-group
assignment, never per arm, with B's conditional move; (3) shared L/O-H
platforms, run-identity fields, per-arm platform disclosure, within-platform
cross-block comparison and the bf16-kernel limitation; (4) the USD 24 internal
cap against USD 26.40 remaining credits, the existing ledger and spend lock, the
workspace hard stop and the whole-group local rerun rule; and (5) the all-local
fallback when Modal is unavailable.  A sixth dated entry records the provenance
correction: the plan was informed by the synthetic-data rehearsals, the Modal
price check and the credit-based cost estimate above, never by pilot data or a
trained pilot outcome.

**Tests and gate.**  `tests/test_prereg.py` covers the typed hardware plan,
the synthetic-only provenance block and its five cited evidence items, the
exact platform/group/arm assignments, the cost-only and never-observed
assignment rule, the shared L/O-H platform rule, the run-identity fields, the
money guard, the Modal-unavailable fallback and schema rejection of unknown,
missing or wrong values.  `check_prereg_loads.py` passes in both plain and
`--tag-mode` modes for the committed document.

### Revision round 7d (prompt cap, 8-bit AdamW, D-7c)

- **Stage 2 prompt cap (R7c-1).** Stage 2 registers a **2,048-token prompt
  cap** as a **new selection rule**, chosen for memory before any Stage 0 or
  Stage 2 prompt selection and not a correction to a recipe already in use.
  Prompts above it are **excluded from the Stage 2 pool at selection time**,
  identically for both arms. A Stage 2 training row is therefore at most
  **2,048 + 8,192 = 10,240 tokens**. The T-dial-rl memory bench (and its
  **"fits"** test) now uses **10,240-token rows**, i.e. the longest real
  Stage 2 row, instead of the round-7c 8,192-token row that was shorter than
  a real row. The **`prereg-v1-data` addendum** records how many prompts the
  cap excludes and how it interacts with the **1,600-prompt cycling rule**.
- **8-bit AdamW (R7c-2).** The registered optimizer is **bitsandbytes 8-bit
  AdamW (`adamw_8bit`) on GPU for every arm and stage that trains**. The
  `optimizer` yaml field keeps the legacy **`AdamW`** spelling, which the
  loader resolves to `adamw_8bit` (CPU-only dev runs use `adamw_torch`), so
  the field stays compatible; the resolution is stated in
  `stage1.recipe_binding` and in §4.2. **`run.json` records the resolved
  optimizer**; the code change is on branch **`fix/optim-record`**.
- **Disclosed deviation D-7c.** A numbered deviation records the three
  round-7d changes made after Stage A was shown to the author in chat at
  **≈06:02** (SendUserFile; the author had not replied): (1) the
  `rows_per_optimizer_step` text registered since **round 1** (`19a1fd9`;
  still present at `d716ec0`) said "8,192 tokens per optimizer step"; **round
  7c** (`2f36330`, 08:28:48) corrected it to **16 rows per optimizer step**,
  as the trainer does, ≈26k supervised tokens at the mean row length; (2) the
  **8-bit AdamW** declaration; (3) the **new Stage 2 prompt-cap selection
  rule** (and its `prereg-v1-data` accounting). **No Stage A content is
  involved**; Stage A was sealed in **`38b5bd2`** (05:56:26) before any result
  was opened, and **`analysis_rules_frozen_at_commit` stays `8c6d78d`**
  because no analysis rule changes; item 3 is the only new selection rule,
  disclosed here.
- **Minor.** The hours-assumption pointer is moved from `mde_notes` to the
  compute notes (`compute.local_seed0_total_hours` now points to the hours
  rows in the compute section).

### Revision round 7c (D-7b deviation, recipe binding, minors)

- **Deviation D-7b (pre-tag, after Stage A was shown to the author).**
  Round 7b replaced the round-6 N=8 → N=6 Stage 2 contingency with: if the
  **T-dial-rl memory bench** shows that an N=6 **or** an N=8 step does not fit
  the allocator cap for the chosen block, both groups become N=4 (mix
  **3/3/10**). This changes the registered training-recipe field
  `rollout_n_groups`, so it is a **disclosed recipe deviation**, not an
  execution-only amendment. The trigger is **measured memory**
  (`perf/dial_bench.json`, sha256
  `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1`, 13
  records); **Stage A content is not an input**. It was made after Stage A was
  shown to the author (≈06:02, in chat; the author had not replied). Stage A
  was sealed by `prereg/stage-a-addendum.md` (commit `38b5bd2`, 05:56:26)
  before any result was opened. `analysis_rules_frozen_at_commit` stays
  `8c6d78d` (the freeze covers analysis and selection rules; D-7b changes a
  recipe field).
- **The T-dial-rl memory bench is fixed now.** It runs before any Stage 2 run
  on the three candidate blocks (25-29, 16-18, 15-29), at N=6 and N=8, with a
  row of **8,192 tokens in total including the prompt** (corrected in round 7d
  to **10,240 tokens**, the 2,048-token prompt cap plus the 8,192 completion
  cap; see **D-7c**), the last-**k=3**
  window, **prelude and pass 1 under `no_grad`**, the **chunked head** and the
  **registered gradient accumulation (16)**. **"Fits"** = one full optimizer
  step completes with **no OOM** and **peak reserved memory below the
  allocator cap** (10.28 GiB on the bench device). The bench and its sha256
  are committed in a pre-Stage-2 addendum. D5 still reports N=6/8 as trained
  in Stage 1 only; the "trained" label is qualified if the fallback fires.
- **Stage 1 recipe binding (pre-tag clarification, no data involved).** Stage
  1 is registered as **per-device batch 1 row × gradient accumulation 16 rows
  per optimizer step** (`rows_per_optimizer_step: 16`; the measured tokens per
  optimizer step are recorded in `run.json`; the token-cosine schedule is
  unchanged). LoRA LR **2e-4**, bridge LR **3e-3**, loop_ple LR **1e-3**,
  gradient clip **1.0** and **3 % warmup** are bound from `prereg.yaml` under
  `--prereg`; a CLI override exits 2. `rows_per_optimizer_step` replaces the
  inconsistent `tokens_per_optimizer_step` (the schema still accepts the old
  key for compatibility); the code branch `fix/prereg-binding` adds loader
  support for both. This is a **pre-tag recipe clarification with no data
  involved**.
- **Minors.** The ≈04:16 Ruling time is sourced to the session transcript
  (right after merge `a360908`, commit time `2026-11-05T04:11:58-07:00`, CPU
  suite ended 04:15). "No longer an estimate" is reworded: the Stage 1 row is
  an estimate from one timed optimizer step. The teacher top-k precompute is
  budgeted at **4-16 h** from the measured E2B throughput (assumptions in
  §10). The inherited 15-29 totals are recomputed to **≈650-1,000 padded**
  before the precompute row, and the stale drafter-choice totals (choices 31
  and 45) are marked superseded. Pending item 4 (Stage A addendum) is closed;
  the open pre-tag runs and the teacher-model download are added to the
  pending list.

### Revision round 7a (freeze correction, controller)

- Native verification of round 7 (R7-1): round 7 changed selection-rule
  fields after the `f8ca082` freeze. The freeze moves to the round-7a commit
  (still before Stage A completed); the three changes, their timing and the
  Stage A file exposure are disclosed in §0.1; "unchanged" eligibility is
  reworded (block-matched O-H).

### Revision round 7b (bench addendum, Stage 2 contingency)

- **Technical changes without data on any endpoint.** This round merges
  `feat/gemma-loop-impl` (`99150b6`: dial training, teacher heal and bench) and
  adds only compute/memory measurements and one conditional recipe
  contingency (disclosed as **D-7b** in round 7c). It adds **no endpoint
  data**, no endpoint, no selection rule and no analysis rule.
- **Bench addendum (pre-tag).** T-dial-bench is measured: 25-29 601.9/569.2
  tok/s at T=4/T=8 (peak reserved 8.475/8.750 GiB); 16-18 491.8/482.7
  (8.795/8.795); 15-29 430.5/375.7 (9.273/9.523); O-H heal 1,085.2 tok/s at
  7.732 GiB with 150 parameters receiving gradients; the Stage 2 proxy OOMs at
  N=6/N=8 for every block (peak reserved 9.912-10.258 GiB against the 10.28
  GiB cap). Source: `C:\Users\<user>\ML\gemmaloop-artifacts\perf\dial_bench.json`,
  sha256 `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1`
  (6,188 bytes, 13 records). The implied Stage 1 hours replace the 12-16 h
  estimate with **8.6 / 10.0 / 10.8-12.0 h** by block (L 5.5/6.9/7.7-8.9 plus
  O-H 3.1) and the total ranges are recomputed (§10); the pilot row keeps its
  43-52 h envelope (each 3M-token heal is ≈0.8-2.2 h).
- **Extended Stage 2 contingency (disclosed deviation D-7b).** If the
  **T-dial-rl** trainer's own memory bench (shipped and run before any Stage 2
  run) shows that an N=6 **or** an N=8 step does not fit the allocator cap for
  the chosen block, **both become N=4**, i.e. the mix **3/3/10** over N=2/3/4.
  The round-7b proxy OOM does not by itself trigger the fallback because it is
  not the trainer. This changes the registered recipe field
  `rollout_n_groups`; it changes no endpoint, selection or analysis rule.
  Round 7c therefore records it as **deviation D-7b** (§9): the trigger is
  measured memory, Stage A content is not an input, and the change was made
  after Stage A was shown to the author. `analysis_rules_frozen_at_commit`
  remains **`8c6d78d`**.

### Revision round 7 (native verifications of round 6 and of engine fix round 6)

Source: "Verification of round 6" in `prereg-v1-round5-review.md` (R1–R3 and
minors), "Verification of fix round 6" in `native-review-t5c.md` (R6-1 and
minors), and the "Ruling (prereg round 7 + engine fix round 5)" and "Ruling
(engine gate v5)" lines of `progress.md`.

| Change | Detail |
|---|---|
| Engine pin (R3/R6-1) | Pinned the evaluation-engine candidate **`eed7fa7`** (engine fix round 6 / gate v5), merged into feat at **`a7c0633`**. Refill eligibility = the registered gate **and** the re-prefill smoke step **and** Part B ≥ 1.3×, run **after the tag and before any trained-arm evaluation**; otherwise the padded engine. `34dd46e` is relabelled the earlier fix-round-4 commit (merged at `2db1868`). **Never pin or bisect to `efd0145`** (broken alone: it deleted about 640 lines that `fc1ffe9` restored). |
| Hours (R6-1/R1/m3/m7) | One consistent set: gate **4.5–6.0 h incl. model loads**, re-prefill smoke **10–20 min**, Part B **5–10 h (central 5–9)**, engine-registration row **9.7–16.3 h**. Pilot **43–52 h** (11 arms: the ninth arm plus the two added O-H arms, +1.8 h). Evaluation charges the N=4 secondary-suite uplift explicitly (**+15–25 padded / +3–5 refill** for 25-29). T-dial-bench **≤ 1.0 h** (T=4/T=8 and the N=6/N=8 RL steps on all three blocks) and the pre-tag row **≤ 1.5 h**. Totals: 25-29 **601–932 / 236–374**, 16-18 **594–920 / 232–367**, 15-29 **627–970 / 247–397**. |
| Pilot O-H arms (R1) | Added `O-H[16-18] + E4B` and `O-H[15-29] + E4B` short heals; the existing controls are renamed `O-H[25-29]`. The matrix is **11 arms** (nine selection/control plus two descriptive anytime-tax arms). Each L arm is compared with the O-H that has the **same block layers**, which matches **LoRA capacity**, not total parameters; L's bridge and multi-exit package are itemized separately (round 8c correction). The teacher step measures **L's gain at the primary N=4**. All pilot quantities are **selection-only**. |
| H1/H0 (R1, Ruling) | H1 is the frozen eligibility gate: `E` non-empty. The selected arm's `Δ̄_arm` is reported with its 95 % CI as a **descriptive** loop-vs-heal estimate and does **not** gate. H0 is the mechanism null (`E` empty). The **confirmatory claims are E2–E4 and D1 only**. |
| Stage A visibility (R2) | The analysis rules are frozen at the round-7a commit (first freeze `f8ca082`, 04:03:25, superseded by the disclosed round-7 rule changes, §0.1), before Stage A completed. At completion the controller hashes every output in a **post-completion addendum committed before the author sees the results**; the time shown and who saw it are logged; any later change is a disclosed deviation. Removed the contradictory "post-tag addendum before the tag" text and labelled `campaign.json` / `child-A_n*.json` as mutable status files. |
| Minors (both verifications) | N=8 → N=6 contingency is per candidate block ("if that block is chosen"). D1 corpus: the bucket 62-69 NLL sweep is separate from the Stage 0 training pool (0-49), math verified by math-verify, code excluded (no hidden-test harness registered for this corpus), chat excluded. Fixed-bytes effects restricted to registered runs (no GPQA/MMLU-Pro at N=2); ρ provenance distinguishes the pinned E2B header from the E4B architecture-notes value; the `pp_per_GiB` scalar is model-level with E4B annotation-only. Pending author list rebuilt (E3 once, bench scope, Stage A addendum, pre-tag R_E4B). The `--allow-stage-b` guard is registered. Part B records the hub-access/artifact rule and the schema rejects an empty 128-prompt sha. `min_exact_fraction` is set to 0 (the registered v4 per-key/pooled rule governs); the smoke `amendment_permitted` and `resolve_run` duplicate-key obligations are registered for the registered run. |

### Revision round 6 (native review of round 5)

Source: the "Audit of stage A exposure" and "Ruling (prereg round 6)" lines at
the end of `progress.md`, `prereg-v1-round5-review.md` (every Critical,
Important and cheap Minor), `prereg-v1-round5-report.md`, `dial-design.md` and
the three panel files.

| Change | Detail |
|---|---|
| Rule B (C1/I1) | STOP/eligible set = the frozen `84c5da0` eligibility. Within it, rank by `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]` (same teacher); adjacency (≤ 0.005 for N=1,2,3) and cNLL(8) ≤ cNLL(1) are a **filter**, with the unfiltered `Δ̄_arm` ranking over E as the fallback. Ties → lower FLOPs, zero-init bridge, smaller block. STOP iff E is empty. Pilot O-H gets the same rank and block-layer LoRA on its single pass (matched LoRA capacity; the bridge/multi-exit package is itemized separately, round 8c). H1/H0 rewritten at N=4. |
| Disclosure and sealing (I2) | Added the exposure audit (who read what and when), the n1/n2 timestamps, the primary-N and Stage-2 recipe changes, and the Stage A sealing table (path, mtime, size, sha256). Round 6 sent later files to the post-tag data addendum; **round 7 replaces that with the pre-visibility post-completion addendum** (§0.1). Stage B (confirmatory items) runs only after the tag. |
| Graphical procedure (I3) | Weights `0.025 × (E2 0.8, E3 0, E4 0, D1 0.2)`, transitions `E2→E3=E3→E4=E4→D1=D1→E2=1`, Bretz et al. 2009 Algorithm 1, one-sided percentile-bootstrap p-values (10,000, seed 0), D1 p = max of its four components; E_{k+1} receives α only after E_k is rejected; final decisions after termination. |
| E4 cap (I4) | The grid is defined on O-H's budget: O-H at B = 1024·2^j, L at floor(B/r_4); both stay ≤ 32,768 and both come from saved completions. r_4 now uses the four-decimal code values 2.2824/1.4275/1.2515. |
| Stage 2 at N>4 (I5) | The policy-gradient backward uses the same last-k=3 window for every N (passes ≤ N−3 under no_grad). T-dial-bench adds the N=6/N=8 RL step. Contingency: for each candidate block whose measured N=8 step exceeds the 10.28 GiB cap, that block's N=8 groups become N=6 if that block is chosen; settled in the pre-tag addendum. |
| Hours (I6) | Stage 2 L recomputed with code `E[r]` for 3/3/8/1/1: 1.4007/1.2358/2.2022 (25-29/16-18/15-29), 1.226/1.140/1.543× the N=2 mix; rows give padded 102-153/95-141/128-191 and refill 45-91/41-84/56-114. Evaluation adds L-heal N=3/N=4 (+8-11 padded / +1.5-2 refill). Totals for all three blocks, plus a pre-tag benchmark row and a post-tag engine row. **Round 7 supersedes the round-6 pre-tag/engine/total numbers; the final set is in the round-7 row above and §10.** Stage 2 FLOPs limitation corrected to 1.24-2.20× O-H per token. The evaluation priority order is also the drop order. |
| Engine sync (I7) | Merged feat `2db1868` and pinned the then-current fix-round commit `34dd46e` (round 7 relabels it fix round 4 and pins the candidate `eed7fa7`). Rebuilt the gate from feat: noise-floor control, re-prefill smoke step, mixed-history waves, round-1024, long prompts, combinations at N ∈ {1,2,3,4,6,8} for every block × bridge, sampled parity at N=4 and every Stage 2 N, throughput at N=1 and N=4. Part B: loops 4, evaluation batch, 360 DAPO candidates, requests ≥ 3× slots. Gate, smoke and Part B run **after the tag** and before any trained-arm evaluation. |
| D6 (I8) | L unmerged is byte-minimal; "L merged" is a sensitivity (R + W_block); O-H merged is primary. Primary claim = (Δ_tot(4), Δ_loop(4)) in pp plus ΔcNLL in nats/token at ΔR≈0, with φ_loop/φ_tot and m beside them; no per-GiB difference. Prelude/block/coda bytes itemised; accuracy reference points restricted to registered runs; Google points are annotations; GiB throughout; q4_0 cross-precision sensitivity dropped; R_E4B to be recomputed from the `4ecc2ba` header before the tag. |
| Contradictions and ladder (I9) | "N fixed per arm" → per engine batch; five per-N batches; H1/H0 at N=4; L-heal N=3/N=4 on the ladder items (DiD at the primary N); second anytime-tax arm at N=4; four named Part 2 milestones with the adaptive-decode milestone counted. |
| Pending author (I10) | Completed: final OK on the panel decisions, restricted bench-addendum scope, feat merge and engine fixes, Stage A sealing, E3 naming. §9 states that the pre-tag deviations from `84c5da0` appear in the paper's deviations table. |
| Cheap minors | Code r constants; Holm at one-sided 0.025; one pilot tie-break list; D4 "no winner" → fixed N=4, ε for N_eff=4, ΔcNLL sign, per-exit temperature; temperature keyed by checkpoint; D1 corpus provenance; pilot count mismatches; KV formula summed over consumer layers and latency re-run rule; 15-29 bytes-read limitation; L-final-N2 LCB view; schema `rollout_n_groups`, derived `full_suite_at`, `fallback_rft.loop_probs` check; changelog wording. |

### Revision round 5 (loop dial, intelligence per byte, panel decisions)

Source: the AUTHOR line of 2026-11-05 ~02:00 and the PANEL OUTCOME line in
`progress.md` override `dial-design.md` where they differ; the remaining text
applies `dial-design.md` §10 and §2–§7, and the wording of the adopted
objections in `dial-panel-ds1.md`, `dial-panel-ds2.md` and
`dial-panel-native.md`.

| Change | Detail |
|---|---|
| Turn dial | Every L arm is trained with the anytime multi-exit objective (`dial_objective`): T=4 at 0.70, T∈{5,6,7,8} at 0.075 each, supervised exits = the last k=3 trainable passes; the coda + LM head run after every pass in the window; exit 1 = O without grad; uniform exit mean plus a stop-gradient monotone hinge with λ=1; teacher KL plus 0.1·KL(O‖exit) at every exit; one shared LoRA and one shared g for every pass ≥2; prelude and pass 1 run without autograd. |
| Primary N | `primary_n = 4` for E2, E3 and E4. N=2 and N=3 become Holm-corrected exploratory points. L(N=4) on the full LCB is the primary run (native vote 4 = B); L(N=2) on the full LCB is the D2 baseline (panel vote 2 = A). |
| r_4 | E4 uses r_4 = 1 + 3 × the one-extra-pass fraction of N=1 matmuls with the LM head included: 2.29 (15-29, 43 %), 1.429 (25-29, 14.3 %), 1.252 (16-18, 8.4 %). The design's analytic 2.282 for 15-29 comes from the unrounded 0.427 fraction; the registered value is the 43 % arithmetic. **Superseded by the round-6 code values.** |
| Dial endpoints | New confirmatory D1 (IUT: cNLL(1) − cNLL(4) > 0 and adjacent regressions < 0.005 nats/token for N=1,2,3) and estimation D2–D5, computed on L final with L-heal descriptive. D2 is L(N=4) − L(N=2) on the full LCB with a paired CI and a −3 pp non-inferiority margin reported but not gated. |
| Multiplicity | Graphical α (Bretz et al. 2009), one-sided 0.025 total: E2 starts at 0.02 (96 % CI), D1 at 0.005 (99 % CI); D1→E2 and E4→D1 weight 1; E2→E3→E4 is the fixed sequence. MDE at 1,055 items is 5.75 pp if D1 fails (5.57 otherwise). |
| Pilot rule | The dial-aware rule B of `dial-design.md` §11.1 replaces selection at N=2: rank by mean gain over N=2..4, prefer adjacent regressions ≤ 0.005 and cNLL(8) ≤ cNLL(1), fall back to the frozen 84c5da0 eligibility, STOP only if both are empty. This is a disclosed deviation from the frozen rule; see §0.1. **Superseded by the round-6 rule B.** |
| Pilot arms | The 7 registered arms use the dial objective; an 8th descriptive arm (25-29 + g + E4B + single-exit legacy at N 2/3, 0.9/0.1) measures the anytime tax and is never eligible. P1 extends R/R-damped to N=1..8. Pilot cost 41–50 GPU-h. **Superseded in round 7: 11 arms and 43–52 GPU-h.** |
| Stage 2 N mix | 16 groups/step: N=2:3, N=3:3, N=4:8, N=6:1, N=8:1; all G rollouts of a group share N; slot→N by sha256(seed:step:slot); TIS and the training forward use the group's N; engine batches by N. O-H stays at N=1. |
| Intelligence per byte | Registered `resident_bytes` and `intelligence_per_byte` (D6 byte frontier, estimation only): itemised resident bytes at bf16 (prelude/block/coda, CPU-side PLE, LoRA, bridge; merged and unmerged), Δ at fixed bytes, φ = share of the E2B→E4B gap closed with Google's same-protocol numbers as a reproduction check, byte-equivalent multiplier m = ρ^φ, analytic bytes read per token, batch-1 latency ≤ 0.5 GPU-h. Limitation: no cross-precision per-byte claim; the 4-bit frontier is Part 2. |
| D4 | Adaptive rules are reported with 95 % problem-cluster CIs; the winner is fixed now by rule (largest mean ΔcNLL at matched analytic FLOPs with CI lower bound > 0), enters the per-byte table and becomes the Part 2 `auto` serve map. Adaptive-decode accuracy is a named Part 2 milestone. |
| Splits and FLOPs | NLL/KL bucket 60-69 is sub-allocated into a T-fit/threshold bucket 60-61 (100) and the scored test bucket 62-69 (400). `flops` registers per-pass fractions and r_N for N=1..16 for the three blocks. |
| Hours | Round-5 estimates with N=4 primary: pilot 41–50; Stage 1 ~12–16; Stage 2 L padded 91–136 / refill 39–81; evaluation padded 309–524 / refill 57–95; totals padded 552–858 / refill 212–336 (25-29 case; other blocks per `dial-design.md` §7.1). **Superseded by rounds 6–7; the final set is in the round-7 row and §10.** |
| Part 2 | Replaced "optional adaptive loop depth" with the detached gate, gate-only RL with a compute penalty, adaptive per-row decode, the exit-1 entropy gate, effort levels low/medium/high/max/auto, span-subsampled exits, a warm-up adapter/clamped embedding, and the precision-matched 4-bit byte frontier. |

### Revision round 4 (selection sync, freeze commit, conditional engine, hours)

| Change | Detail |
|---|---|
| Selection text sync | §3.3 now reproduces the YAML two-step rule exactly: teacher first by mean calibrated-NLL gain over O of (25-29+gch+T) and (O-H+T), then block/bridge among the E4B arms versus the E4B O-H, with the YAML tie-breaks. |
| Freeze statement | Candidate set and selection/STOP rules are frozen at commit `84c5da0f1778cb7446d20bed0cd428c5799ea1b1`, timestamp `2026-11-04T22:39:04-07:00`; no campaign output existed at that time. |
| Conditional evaluation engine | Refill engine (task/t5c, commit `a81282e`) iff the 5c fixes and the gate pass before `analysis-v1`; otherwise padded adaptive-batch. Gate: greedy token-exact vs padded on 32 prompts at N=1 and N=2; identical per-request sampled tokens on 16 prompts. Same engine for O, O-H and L. |
| Hours | Re-forecast for both engines: padded 500-779 h and refill 191-300 h for seed 0; Stage 0 on Modal (not local); Stage 0 fallback = registered engine; ladder ≈35 h without reuse. |

### Revision round 3 (re-review items (1)-(9) and Stage 0 record)

| Ruling | Change |
|---|---|
| (1) Pilot selection | One deterministic two-step rule: teacher first by mean calibrated-NLL gain over O of (25-29+gch+T) and (O-H+T), with an E4B tie-break within 0.003 nats; then block and bridge among the E4B arms, each compared with the E4B O-H. |
| (2) Engines | Stage 0 = Modal stock vLLM v0.30.0 (data only); Stage 2 rollouts and all evaluation = the registered evaluation engine (refill if the gate passes, otherwise padded adaptive-batch) for O, O-H and L; same engine per comparison; N=1 bit-exactness vs stock HF. |
| (3) LoRA/bridge | LoRA targets are loop-block layers only (`loop_block` option); the bridge acts only on loop passes >= 2, not on the coda. |
| (4) R-damped | R-damped is the Euler `damped_update` with weights 1/N (0.5/0.333/0.25 at N=2/3/4) everywhere; the legacy scalar `w=0.99` is a separate bridge. |
| (5) Loss/teacher numerics | Stage 1 and RFT loss on assistant tokens only (thinking + answer); teachers loaded with bitsandbytes nf4 (E4B regular checkpoint, 12B QAT-unquantized); numerics declared. |
| (6) Pins | All CPU-computable pins are filled now (E2B template, teacher revisions, tokenizer hashes, lcb_runner); math-verify is a dependency from round 8b (0.9.0 pinned in §5/§11); only the manifest SHA remains null, in the post-tag `prereg-v1-data` addendum. |
| (7) Pilot buckets | The pilot bucket 50-59 is sub-allocated into disjoint P3 (50-53), selection (54-56), T-fit (57-58) and P1 (59) sub-ranges; P3 RL prompts are drawn from the Stage 0 train bucket 0-49. |
| (8) Disclosure | Cites the frozen-candidate commit `84c5da0` and timestamp `2026-11-04T22:39:04-07:00`; campaign results are reused only for the ladder with identical engine/harness, never for E3/DiD; Stage B accuracy is not launched until the author decides E3 (Stage A may run). |
| (9) Ladder/analysis | Ladder budget recomputed with the reuse rule (+20-22 h without reuse, +11-13 h with reuse); analysis-v1 obligations are registered (delete dead S1-S5 code, re-cover helper tests, duplicate key with block/bridge/policy). |
| Stage 0 record | Modal, stock vLLM v0.30.0, $30 usage limit / $0 spend limit, local HF-engine fallback with the engine recorded per row and split assignment unaffected. |

### Revision round 2 (loop ladder and training-free disclosure)

| Ruling | Change |
|---|---|
| Loop ladder | Registered the descriptive/estimation-only O → R/R-damped → L-heal → L-final ladder on the chosen block at N=2,3,4, with O-H after each stage, LCB post-cut 182 x 1 + GPQA 198 x 1 per point, the full confirmatory suite at L final N=2, the R-damped rule and weights, the gch-at-init note, the +20-22 local GPU-hour budget and the accuracy-vs-N chart with stage lines. It also supplies the post-Stage-1 measurements for the difference-in-differences. |
| Training-free campaign | Disclosed the pre-tag exploratory campaign: P1-lite calibrated-NLL sweep on a 120-problem verified-trace corpus; O vs R/R-damped at N=2 on the three candidate blocks on LCB post-cut 182 + GPQA 198; then N=3,4 on the block with the lowest calibrated NLL. Its results are exploratory, reported in full, and are not inputs to block selection; the candidate set and pilot selection/STOP rules are frozen as of the controller's ruling time; the P1-lite corpus is disjoint from every registered split. **(Historical; round 8c retires this blanket claim: the original Stage A corpus is not claimed disjoint from every Stage 0 split, while the filtered teacher-qualification corpus is — see §0.1.)** |
| Test coverage | Restored schema-level tests for the MDE table and every E1-E4 endpoint field, plus the loop-ladder and training-free disclosure sections. |

### Revision round 1 (native review `prereg-v1-draft-review.md`)

| Review item | Change |
|---|---|
| C1 | O-H now runs the real Stage 1 (same teacher, data and 12M budget); the "no-op" claim, the KL < 1e-4 check and "Stage 1 charged to L only" are removed. |
| C2 | The pilot gate is scored on O's **verified-correct traces of held-out problems**, with T fitted per (arm, N) on a fit split and a problem-cluster bootstrap; raw O samples are no longer the gate. |
| C3 | E1 is a **precondition with no alpha** (bit-exact L(N=1) ≡ O on every engine) plus descriptive heal estimates; the gated sequence is **E2 → E3 → E4**, each requiring a 95 % CI lower bound > 0. |
| C4 | E3 is relabelled "L(N=2) vs O — total effect vs the original"; it is no longer called a clean loop contrast. |
| I1 | Selection/STOP is deterministic: lower bound > 0, signed `delta_arm`, point estimate, STOP when the eligible set is empty, problem-cluster bootstrap, T per (arm, N), KL(O‖L N=1) = 0 by construction. |
| I2 | P2 has 7 arms and adds O-H+12B; the teacher is chosen by the mean gain of L and O-H, with an E4B tie-break within 0.003 nats. |
| I3 | Stage 1 budget is **fixed at 12M**; the loss is logged every 2M tokens descriptively, with no dose-response decision. |
| I4 | P3 uses 200 held-out math problems x 4, a 4k cap on prompts with O median <= 2.5k, paired SE >= 2 and truncation < 10 %; the RFT fallback recipe is registered now. |
| I5 | One deterministic split table by `sha256(problem_id) % 100`, disjoint by problem id and disjoint from Stage 1 and Stage 2 data. |
| I6 | Extra seeds [1, 2] are registered; the confirmatory verdict is seed 0 only and the crossed bootstrap is robustness only. |
| I7 | The sequence rule is consistently "95 % CI lower bound > 0", matching the MDE table. |
| I8 | E4 registers r = 1.428/1.143/1.084, B = 1024·2^j (j=0..5), total-token correctness at B, trapezoid AUC / 5, and the 32,768 cap for O-H. |
| I9 | Compute now registers a local HF-refill engine and a cloud vLLM engine, an honest hours table per stage, and an evaluation priority order. |
| I10 | Recipe gaps are filled: optimizer, tokens/step, schedule, clip, loss tokens, top-k tail, live parent KL, teacher runtime, Stage 2 num_iterations/normaliser/grader timeout, LoRA target = loop block only. |
| I11 | Legacy fields are retired: `criteria` and `train_tokens` are rejected by the schema; `check_criteria` refuses S1-S5; `phase1_plan` refuses the legacy plan. Bench roles are explicit with no default to confirmatory. |
| I12 | Data provenance is named: TACO/APPS/CodeContests pre-2023-05, OMR/DAPO-17k/DeepScaleR, the existing decontaminated chat source; "synthetic" is dropped; the Stage-0 sample filter replaces the OT3 rule; the < 1,600-prompt rule is registered. |
| I13 | Pins are registered: E2B snapshot, template SHA, teacher revisions and tokenizer hashes, lcb_runner, math-verify, manifest SHA; the addendum procedure is written; the freeze point is the tag, before any trained-arm evaluation. |
| Minor | Fixed the dropped "calibrated"; NLL is secondary; legacy S1/S2/S5 names removed; Holm family specified; R-damped w = 0.99 and its item set specified; DiD has post-Stage-1 LCB runs; `lcb_google` is a virtual view of O's full 1,055 run; Part 2 dev composite and milestone cap defined; automatic checks labelled as the controller's recommendation; multi-GPU tier added; an MDE recomputation test was added; "S2 positive" became "E2 passes"; decision-log export added. |

### Previous draft (from the old single-SFT prereg)

| Change | Source report / ruling |
|---|---|
| Replaced the single-SFT design with the two-stage hybrid. | `twostage-vs-current-design.md`; AUTHOR 2026-11-04 |
| Part 1 arms O, O-H, L; R/R-damped diagnostic; removed SFT mix arms, L-plevec, O-H-ext; optional O-H-RLext only if E2 passes. | `methodology-review-B-training-eval.md`; `twostage-vs-current-design.md` |
| Architecture chosen by the pilot from {25-29, 16-18, 15-29} and {zero-init per-channel g, legacy scalar}; adapters N >= 2. | `methodology-review-A-architecture-loops.md` |
| Training N distribution {2,3} at 0.9/0.1. | `methodology-review-A-architecture-loops.md` |
| Stage 1 teacher E4B or 12B QAT 4-bit; offline top-k logits + parent KL; glitch token masked. | progress AUTHOR 2026-11-04 21:10; review A |
| S2 primary on full LCB v6 1,055 x 1 with post-cut subgroup. | review B; `audit-eval-stats.md` |
| Statistics: TOST removed from the gate, problem-cluster bootstrap, fixed sequence, Holm, MDE table, single-seed statement. | review B; `audit-eval-stats.md` |
| Numeric pilot rule. | review A; `audit-redteam-design.md`; two-stage design |
| Stage 0 sampling, held-out splits, decontamination, manifest SHA, 8192 filter. | two-stage design; `data-length-audit.md`; `training-perf-report.md` |
| Compute tiers and Part 2 rules. | progress Rulings 316/359 |
| Google comparison is a reproduction check; `lcb_google` is harness-only. | review B |
| Creative S3 demoted to Part 2 / exploratory. | review B; red-team audit |

---

## 1. Hypotheses

- **H1 (pilot selection gate; not confirmatory).** The pilot's eligible set E
  (the frozen `84c5da0` rule: `delta_loop = calibrated NLL(L,N=1) − calibrated
  NLL(L,N=2) > 0` with a 95 % problem-cluster lower bound and
  `delta_arm ≥ −0.02`) is non-empty. H1 is exactly "**E is non-empty**". The
  selected arm's loop-vs-heal estimate,
  `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]` (same teacher, matched
  rank-16 LoRA capacity per block; L's 1,536-parameter bridge and multi-exit
  package are itemized separately), is reported with its 95 % CI as a **descriptive**
  quantity; it does **not** gate. All pilot quantities are selection-only;
  **the confirmatory claims are E2–E4 and D1 only**. N=2 remains the frozen
  pilot eligibility point; the teacher step is measured at the primary N=4. If
  E is empty, the study publishes the mechanism null and stops Part 1 training.
- **H2 (confirmatory primary, E2).** `L(N=4) - O-H > 0` on the full
  LiveCodeBench `release_v6` (1,055 items, one sample per seed), with a 96 % CI
  lower bound > 0 (one-sided 0.02; 95 % after D1 is rejected). O-H is the
  data-and-recipe control; L carries the loop.
- **H3 (Total effect versus O: training + loops, L(N=4) - O; E3).**
  `L(N=4) - O > 0` on the same items, with the same registered lower bound > 0.
  Because L(N=1) is O bit-exactly, this is the total training-plus-loop effect,
  not a clean loop contrast. The author delegated the naming to the reviewer
  panel, which recommended this wording; E3 is kept.
- **H4 (compute efficiency, E4).** The accuracy-versus-budget curve of L at
  N=4 is better than the FLOPs-matched curve of O-H: AUC over
  `B = 1024 * 2^j`, j = 0..5, for **O-H at B and L at floor(B / r_4)**, with
  r_4 = 2.2824 (15-29), 1.4275 (25-29) or 1.2515 (16-18). Both arms stay under
  the 32,768-token cap.
- **H5 (dial, D1).** After both stages, L's calibrated NLL improves from N=1 to
  N=4 and does not worsen by more than **0.005 nats/token** at any adjacent step
  N→N+1. This is the confirmatory dial endpoint (intersection-union test).
- **H0 (mechanism null; not confirmatory).** If the pilot's eligible set E is
  empty, the paper reports that the retrofit did not learn to use the loop
  under this budget and stops Part 1 training. H0 is the complement of the
  pilot selection gate H1; it is not a confirmatory hypothesis. The
  monotonicity filter is not part of H0 and cannot make H0 true. The
  confirmatory family remains E2–E4 and D1.
- **Secondary, estimation only.** NLL on three corpora (teacher traces, O's
  verified traces, answer segment only), GPQA, MMLU-Pro, vision, generated
  lengths, truncation, FLOPs per problem and difference-in-differences across
  stages (using the post-Stage-1 LCB runs). These are reported with intervals,
  never as pass/fail claims.

**Explicit claims (round 8; X1 item 2).** D1 establishes **average
calibrated-likelihood gain and non-inferiority on O-verifiable math traces
only**. It is a calibrated-likelihood consistency result, not a general
"usable reasoning dial", not per-token monotonicity, and not a guarantee for
hard problems O cannot solve, code, arbitrary N or generated continuations.
**E2 compares the complete looped training/inference package** (multi-exit
supervision and hinge, the extra bridge, different RL depth strata and
additional inference computation) **with a matched-data, matched-layer adapter
control**; it is not the pure causal effect of recurrence with every training
and compute factor held fixed. **D2 measures the extra-depth effect** of N=4
over N=2 (estimation only); a flat tail or a non-significant difference is
reported as a plateau or an unresolved effect, never as proof that every extra
loop helps. **D4 provides teacher-forced likelihood evidence only** and cannot
supply adaptive-decode accuracy; adaptive-decode accuracy remains a named
Part 2 milestone. A D1 pass alone must not produce the headline "a usable
reasoning dial" or "every extra loop helps". **O-H establishes matched LoRA
capacity, not exact total-parameter or training-FLOP equality**; L's recurrent
bridge, multi-exit supervision and loop package are itemized separately.
**D4's test-selected winner and interval are exploratory/selected/optimistic**
and require fresh-data validation before any Part 2 serving claim. The
registered numerical endpoints and α allocation are unchanged.

The pilot and its "why" report compare **deployable teacher representations**
(runtime, quantisation and fidelity under the registered qualification rule),
not model size alone; each teacher's fidelity result is reported beside the
student gains. The "why" material is **descriptive, not a causal identification
result**. The verification domain is **O-solvable verified math plus code
passing the frozen Docker verifier; chat never counts**. It does not establish
the best teacher for chat, hard unsolved math, arbitrary code or
generated-answer accuracy; code verification establishes passing the frozen
tests, not formal correctness. Confirmatory D1 retains its separate math-only
corpus and contract. Main training **restarts from O** for both arms; the 3M
pilot heals
are selection-only, and rendered input tokens, scored assistant tokens and
repeated multi-exit targets are separate denominators (3M + 12M is never
described as 12M). **Shared seeds do not establish cross-backend identity**;
local HF and Modal/vLLM parity rests on the same frozen artifacts and recorded
backend provenance. The paper never claims "better reasoning in general".

---

## 2. Design

### 2.1 Part 1: confirmatory

- **Size and seeds.** E2B. Seed 0 is the only confirmatory seed. Seeds [1, 2]
  are registered now as robustness-only runs, in that order, and never replace
  the seed-0 verdict. They run only if compute arrives and run regardless of the
  seed-0 result. E4B is deferred.
- **Arms.**
  - **O**: original instruction-tuned E2B, no loops, no adapter.
  - **O-H**: same two-stage recipe with no loop block. **O-H runs a real
    Stage 1** with the same teacher, data and 12M budget; its rank-16 LoRA
    targets the same block layers as the matching L arm, so **LoRA capacity is
    matched**. It carries no recurrent bridge and no multi-exit loop package.
    Stage 2 uses the identical prompts, order, G, steps, cap, sampling, LR and
    seeds. O-H is the matched-data, matched-LoRA-capacity control; L's bridge
    (1,536 parameters), multi-exit supervision and loop machinery are itemized
    separately, so L − O-H is not the pure causal effect of recurrence.
  - **L**: pilot-chosen loop block and bridge, the same Stage 1 and Stage 2,
    trained with the dial objective so every N in 2..8 is usable. In Stage 2 N
    is stratified per prompt group, not fixed per arm.
- **Primary N.** N=4 for E2, E3 and E4; N=2 and N=3 are exploratory and
  Holm-corrected. L(N=4) on the full LCB is the primary run; L(N=2) on the
  same 1,055 items is the D2 paired baseline.
- **Diagnostic only.** R (untrained loops, no bridge, no adapter) and
  R-damped (legacy scalar bridge, w = 0.99) are scored on the 182 post-cut
  items x 1 and never enter the confirmatory family.
- **Removed.** The old SFT mix arms, L-plevec and O-H-ext. O-H-RLext is
  optional, seed 0 only, only if **E2 passes**: 1.5x Stage 2 steps on fresh
  prompts; it is not confirmatory.
- **Checkpoints.** Final checkpoint only. Evaluation timepoints are after
  Stage 1 (precondition E1 and descriptive heal estimates) and after Stage 2
  (E2-E4). The pilot never scores a checkpoint on a benchmark.

### 2.2 Part 2: exploratory iteration

Part 2 starts from L-s0 (and O-H-s0 where cheap) and is development, never part
of Part 1 claims. It uses versioned checkpoints with lineage and the techniques
in section 11. Decisions are made only on dev sets; held-out test benchmarks are
frozen and scored only at the named milestones; every iteration is logged with
data, technique and results including failures; a judge is never the sole
evaluator of anything it trained.

### 2.3 Compute tiers

Registered tiers: local, single cloud GPU, multi-GPU and TPU. The tier is chosen
only by availability, fixed before any result and recorded as a covariate. O-H
and L of a seed always share the tier and engine. Stage 0 uses Modal stock
vLLM v0.30.0 (data only) for the main runs, with a local fallback. **Round 8
Stage-0-lite option:** the pilot data may come from the **LOCAL Stage-0-lite
sampler** on the padded HF engine, using the same registered sampling
(temperature 1.0, top_p 0.95, top_k 64, thinking on, four samples per prompt)
with the per-request generation cap set to `max_length` minus the rendered
prompt length. Under the per-request seeds `sha256(seed:request_id)`, a capped
run preserves the accepted prefix of a longer run on the **same backend and
random stream**. Cross-backend identity with Modal/vLLM is **not claimed from
seeds**: fairness rests on the same actual frozen artifacts shared by every arm
plus a recorded backend/engine/sampling provenance per row. The tier is
recorded as a covariate and is pilot data only. The **full Stage 0 for the
main runs remains Modal-or-local by availability** (Modal when the author's
token/approval is available, otherwise the registered local padded engine),
fixed before any result and recorded per row.

**Stage 2 rollouts and all evaluation use the padded adaptive-batch HF engine
unconditionally (round 8c).** It is the **sole operative Part 1 engine**, same
for O, O-H and L; no refill run is registered. The `engine_gate` and `part_b`
blocks are **INACTIVE historical records** with no execution authority, and no
post-tag engine addendum is required. The corrected smoke record is: **four**
pre-tag attempts (two launcher/GPU-only crashes, one OOM and one completed
SMOKE-FAIL, the last not eligible); the one bounded repair developed **gate
v6** on `fix/gate-v6` (head `8f36aca`), which native review **rejected**
(C1–C3); the **repaired smoke was never run** and v6 remained **unmerged**.
`eed7fa7` is historical gate-v5 provenance. The operative padded source is
pinned by path and blob hash at `54cab07` (`serve/generate.py`,
`serve/cache.py`, `eval/runner.py`). Never pin or bisect to `efd0145` (broken
alone).

---

## 3. Architecture candidates and the pilot rule

### 3.1 Candidate set

- **Loop block** in {25-29, 16-18, 15-29}. 15-29 is the legacy arm; 25-29 and
  16-18 are high-cosine consumer periods. Producer blocks are exploratory only.
- **Bridge** in {zero-init norm-matched per-channel `g`, legacy scalar}. The
  primary candidate is

  ```text
  x2 = h_pre + g * (rms(h_pre) * RMSNorm(h_prev)),   g in R^1536, g init 0
  ```

  so N=2 is exactly N=1 at init and the gradient is non-zero from step 0. The
  legacy scalar has `w0 = 0.99`.
- **Adapters only for N >= 2, shared.** One LoRA and one bridge `g` are shared
  by all passes >= 2 (time-invariant); pass 1 = O; there are no pass-indexed
  adapters and no pass embedding (`ple_mode: shared`; `loop_vector` has 4 rows
  and `set_n_loops` refuses N > 4). The prelude and pass 1 run without autograd.
  LoRA targets all linear layers of the chosen loop block only (`loop_block`
  option); the bridge is active only on loop passes >= 2, not on the coda.
  **In the pilot there is one O-H control per candidate block-layer set**
  (25-29, 16-18, 15-29); each applies the same rank-16 LoRA to that block's
  layers on its single pass, so each L arm and its matching O-H have matched
  rank-16 LoRA capacity; L's 1,536-parameter bridge and multi-exit/loop
  package are itemized separately and are not claimed matched. The **final O-H** uses the pilot-chosen L block's
  layers. O-H's LoRA is merged (0 resident bytes) in its deployed form, while
  L's is counted unmerged. **L(N=1) is O bit-exactly** on every engine used.
- **Training horizon.** `loop_probs = [0, 0, 0, 0.70, 0.075, 0.075, 0.075,
  0.075]`: T=4 at 70 %, T in {5,6,7,8} at 7.5 % each. Supervised exits are the
  last k=3 trainable passes (the BPTT window). N=2..8 are trained (dense on
  2–4); the registered effort levels are N=1..4; N=12 and 16 are extrapolation,
  NLL only. N=1 batches are not trained because adapters are N >= 2-only.
- **Dial objective.** Every exit in the window emits a coda + LM-head output;
  its loss is teacher forward-KL (top-64 + tail) plus 0.1·KL(O‖exit), the exits
  are averaged uniformly, and a stop-gradient monotone hinge with λ=1
  penalises a later exit that is worse than its predecessor. Excluded: learned
  exit weights, successive-exit KL, fixed-point penalties, index-increasing
  weights, pass-indexed adapters. Logged every step: per-exit KL_teacher and
  KL_O, ‖g‖, ‖Δh_t‖/‖h_{t−1}‖, the period-2 ratio and hinge activity. O-H is
  unchanged (single exit at N=1).
- **Fixed KV.** Consumers read the prelude cache; no per-pass KV is added in
  Part 1.
- **FLOPs.** An extra pass costs 42.75 % of N=1 matmuls for 15-29, 14.25 % for
  one period (25-29) and 8.38 % for 16-18 with the LM head included. E4 and D3
  use the code values `r_4 = 1 + 3 × that` = **2.2824/1.4275/1.2515**; the old
  rounded 2.29/1.429/1.252 are superseded. `flops` registers per-pass fractions
  and r_N for N=1..16; useful
  analytic per-token FLOPs including exit-check costs are the registered
  measure, with executed FLOPs and wall clock reported and attention-score
  FLOPs as a sensitivity. D3 remains a secondary FLOPs frontier.

### 3.2 Layer-scalar geometry (documented, not a bug)

Gemma 4's `layer_scalar` multiplies the whole residual stream, so the stream is
leaky and is rewritten at several layers. For E2B the scalars are L0 0.018,
L13 0.088, L14 0.029, L15 0.254, L16-24 0.43-0.66, L25-33 0.70-0.87, L34 0.167;
the product over 15-29 is 4.3e-4. Measured on three samples:
`cos(h14, h29) = -0.01`, `RMS(h29)/RMS(h14) = 1.5-1.8`, and block
input/output cosines are about 0.00 for 15-29, 0.66-0.73 for 16-18 and
0.75-0.86 for 25-29. "A loop re-applies a refinement to the residual" is
therefore false for 15-29. This is why the block is chosen by the pilot and why
15-29 is the legacy arm. E4B scalars are not read in Part 1.

### 3.3 Pilot (runs only after the tag)

The pilot uses held-out metrics only and **never** computes benchmark accuracy
on a pilot checkpoint. Its round-7 local cost estimate is **43-52 GPU-hours**
(P1 + P2 + P3), including the dial objective, the two descriptive anytime-tax
arms, the 11-arm matrix (the ninth arm and the two added O-H arms, +1.8 h),
multi-N sweeps and P1 to N=8. T-dial-bench is charged in the pre-tag row. All
pilot quantities are **selection-only**; the confirmatory claims are E2–E4 and
D1.

**Pilot "why" report (round 8; descriptive diagnostics).** The pilot harness
writes a plain-language report explaining **why** a teacher/recipe won, not only
the ranking. Its registered metrics are: **teacher qualification error**
(per-teacher approximation metrics against the bf16 reference); **teacher
entropy/confidence**; **initial vs final student–teacher KL and the fraction of
the teacher KL gap closed**; **gains by span (thinking vs answer) and by
position**; **per-N curves for N=1..8**; and **cost per token** (analytic and
measured). These diagnostics **never enter selection, eligibility, STOP or any
confirmatory endpoint**; they are descriptive only.

- **P1: forward sweep (about 0.3 GPU-h).** On **50 bucket-59 problems × 4
  samples = 200 attempts** (round 8b: `rows` counts sampled completions, not
  problems; `pilot_p1_problems` stays 50), measure per-layer cosine and RMS,
  untrained KL and temperature-calibrated delta NLL for blocks 15-29, 15-19,
  16-18 and 25-29. R/R-damped are extended to N = 1..8 descriptively. P1 is
  descriptive input to P2, not a selection rule by itself.
  **Realized P1 (round 8c).** Bucket 59 actually holds **58** problems (26
  math, 21 code, 11 chat). The deterministic truncation scans the bucket's
  four requests per problem in ascending `sha256(0:request_id)` order, ties
  by `request_id`, takes the first **50 distinct problems** and keeps all four
  samples: 23 math, 19 code, 8 chat. These are **attempts, not retained
  rows**. A row is retained only if it passes the Stage 0 filter and its
  answer verifies; chat never counts as a verified target. Rejected attempts
  are reported, never replaced. **Round 8e:** P1's verified domain is **math
  plus code passing the frozen Docker verifier**; chat never counts. P1 is
  **descriptive for every count and never gates P2**: with fewer than 20
  verified problems its estimates are labelled small-sample descriptive, with
  zero it is unevaluable, and no P1 metric or diagnostic affects the extension,
  teacher choice, arm matrix, architecture selection, stopping or any
  confirmatory claim. The historical 50×4 = 200 describes original attempts,
  never 200 verified rows. No regeneration or borrowing is allowed.
- **P2: short heals (3M tokens each).** A pre-listed **11-arm** matrix; the L
  arms are trained with the dial objective:
  - 25-29 + zero-init g + E4B
  - 16-18 + zero-init g + E4B
  - 15-29 + zero-init g + E4B
  - 15-29 + legacy scalar + E4B
  - 25-29 + zero-init g + 12B QAT q4_0
  - O-H[25-29] + E4B
  - O-H[25-29] + 12B QAT q4_0
  - O-H[16-18] + E4B
  - O-H[15-29] + E4B
  - 25-29 + zero-init g + E4B + single-exit legacy (N 2/3 at 0.9/0.1):
    **descriptive anytime-tax reference; never eligible.**
  - 25-29 + zero-init g + E4B + single-exit legacy (N=4 fixed):
    **descriptive anytime-tax reference at the primary N; never eligible.**

  The two O-H arms for 16-18 and 15-29 are the added short heals of round 7;
  each L arm is compared with the O-H whose LoRA targets the same block layers,
  so **rank-16 LoRA capacity is matched** per block; L's 1,536-parameter
  bridge and multi-exit/loop package are itemized separately and are not
  claimed to be capacity-matched.

  **The 11-arm matrix is read as two registered stages (round 8b).**
  *Teacher stage:* for each teacher in {E4B, 12B QAT q4_0}, run
  `25-29 + gch` at L's primary N=4 and `O-H[25-29]` at N=1; the teacher with
  the larger mean calibrated-NLL gain over O across those two measurements
  wins, with an E4B tie-break within 0.003 nats. *Block/bridge stage:* after
  the teacher is fixed, compare the **E4B L arms** (25-29 + gch, 16-18 + gch,
  15-29 + gch, 15-29 + legacy scalar) with rule B, each against the E4B O-H
  whose LoRA targets the same block layers. The 12B L arm is the teacher stage
  only and is **not** a block/bridge candidate; the two anytime-tax arms are
  descriptive and never eligible. The chosen recipe is then trained with the
  teacher selected in the teacher stage, so a 12B win carries through to
  Stage 1/2 and the final O-H.

  **Realized pilot corpus and domain (round 8e; counts from rounds 8c/8e).**
  The T-fit split (57-58) holds **86** problems (29 math, 38 code, 19 chat) and
  the selection split (54-56) holds **152** (62 math, 60 code, 30 chat); these
  are input counts, not retained verified traces. The pilot's verified domain
  is **math plus code passing the frozen Docker verifier**; chat never counts.
  Use only the original selected input sets: the 86 originally sampled T-fit
  problems and the 150 originally sampled selection problems, with their
  original split membership; do not expand to unused bucket members or nominal
  target sizes. Their complete problem/request ID lists are frozen before this
  amendment, and the final corpora are the distinct verified math-plus-code
  problems in ascending `sha256(0:request_id)` order within each split, frozen
  with the staged freeze and `prereg-v1-data` before any pilot scoring.
  **Counts are facts, not the decision input:** original math-only counts are
  P1 13/27 rows, T-fit 16/33, selection 23/45; the preview math+code union is
  16/27/40 from `codeverify/summary.json` with Docker image
  `python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54`.
  The official count is re-run from the same frozen original samples with the
  frozen verifier. The panel fixed the conditional rule before the counts.

  **Conditional extension and final adequacy (round 8e).** Complete the
  original per-request accounting first. If `nT0 >= 20` AND `nS0 >= 30`
  distinct verified math-plus-code problems, use only the original attempts
  and make no extension. Otherwise generate **exactly four additional
  original-model attempts, indices 4, 5, 6 and 7**, for every math/code
  problem in each original T-fit and selection input set, irrespective of
  previous success and irrespective of which split is deficient; generate no
  extra P1 or chat attempts. The additional request count is exactly four times
  the math/code problems in the union of those two input sets. For `problem_id
  p` and index `j`, `request_id = p + ":" + str(j)` and `seed =
  int.from_bytes(sha256(UTF8("0:"+request_id)).digest()[:8], "little",
  signed=False) modulo (2**63-1)`, as in the corrected `lite_request_seed`
  implementation. Commit the full conditional request/seed/configuration plan
  before generating any completion; keep the original run immutable and write
  the extension to a separately identified, parent-linked artifact. A
  technical resume may complete missing request IDs with the identical plan;
  never regenerate an already committed completion or stop early on a
  threshold. P2 runs only if **final T-fit ≥ 20 AND final selection ≥ 30
  distinct verified problems** and all other registered prerequisites hold;
  otherwise the pilot is **unevaluable for insufficient corpus and P2 does not
  run**, with the counts and reason reported. This is not evidence for a
  mechanism null. There is **no automatic second batch, minimum reduction,
  filter relaxation, domain substitution, bucket borrowing or default
  teacher/architecture**; D and E are rejected, and neither Stage 0 filter is
  relaxed. All four teacher cells must complete before ranking; an aborted or
  incomplete cell makes the comparison pending/unevaluable, never a licence to
  select survivors.

  Endpoints: `delta_loop = calibrated NLL(L,N=1) - calibrated NLL(L,N=2)`,
  `delta_arm = calibrated NLL(O-H[block]) - calibrated NLL(L,N=2)` using the
  O-H with the same block layers (matched rank-16 LoRA capacity per block;
  L's bridge and multi-exit package are itemized separately),
  `KL(O || L N=1) = 0 by construction`, and the ranking statistic
  `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) - cNLL(L,N)]` at the same teacher
  and data. Every arm also reports cNLL at N = 1..4, 6, 8, monotonicity,
  `delta_loop`, `delta_arm`, FLOPs and bytes read per token at N=4. The corpus
  is **O's verified-correct traces on held-out problems** (teacher-neutral),
  not O's raw samples. Temperature is fitted **per (arm, checkpoint, N)** on a
  separate fit split; raw NLL is never used. The bootstrap is by **problem
  cluster**.
  - **Selection (two-step, exactly as in the YAML).**
    1. **Teacher first.** Choose the teacher with the larger mean
       calibrated-NLL gain over O of `(25-29+gch+T)` **measured at L's primary
       N=4** and `(O-H[25-29]+T)` at N=1; E4B wins ties within 0.003 nats.
    2. **Then block and bridge by rule B.**
       - **Step 1 — eligible set.** `E` = the frozen `84c5da0` eligibility
         criteria, except that `delta_arm` uses the block-matched O-H (round 7,
         disclosed in §0.1): E4B L arms with a `delta_loop` 95 % problem-cluster lower
         bound > 0 and a `delta_arm` point estimate ≥ −0.02. **STOP iff E is
         empty.**
       - **Step 2 — rank.** Rank the arms in E by
         `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]`.
       - **Step 3 — monotonicity filter.** On point estimates,
         `cNLL(N+1) − cNLL(N) ≤ 0.005` nats/token for N=1,2,3 and
         `cNLL(8) ≤ cNLL(1)`. If at least one arm in E passes, select the
         argmax of `Δ̄_arm` among the passing arms; otherwise select the
         argmax of `Δ̄_arm` over all of E (unfiltered).
       - **Ties** within 0.003 nats of the maximum: lower FLOPs ratio, then
         the zero-init per-channel bridge, then the smaller block.

    The chosen recipe is then trained with the selected teacher, and the final
    O-H uses that same teacher. The old cNLL(1)-based dial gain is reported
    descriptively only; it is not the ranking statistic.
  - **STOP.** STOP iff the frozen eligible set E is empty; then publish the
    mechanism null and stop Part 1 training. The monotonicity filter cannot
    make STOP fire.
- **Secondary.** Report the fraction of the teacher KL gap closed at N=2.
- **P3: short RL on O only.** 30 steps, 4k cap, 200 held-out math problems x 4
  samples, on prompts whose O median is <= 2.5k. Keep Stage 2 if dev pass@1
  gains at least 2 **paired** standard errors and truncation stays below 10 %.
  Exact-answer math has no meaningful hacking flags; a code audit is separate.
  Otherwise use the registered RFT fallback.
- **Stage 1 budget.** Fixed at **12M tokens**; the loss is logged every 2M
  tokens **descriptively only**, with no dose-response decision.

### 3.4 Loop ladder (descriptive/estimation only, after the tag)

The ladder is **descriptive and estimation only**. It runs **after the tag**,
is **never used for pilot or selection**, and is not part of the confirmatory
family. It is the accuracy-vs-N picture with stage lines and the source of the
post-Stage-1 measurements needed by the difference-in-differences.

- **Chosen block.** The ladder uses the pilot-chosen block. Because `gch` at
  init with `loop_only` is **O exactly**, the untrained looped measurement uses
  the R and R-damped bridges, not `gch`.
- **R (pure repeat).** The bridge is `none`: each loop pass re-enters the block
  from `h_prev`.
- **R-damped.** The registered damped rule is the Euler step
  `x <- (1 - 1/N) * x + (1/N) * block_out` (`damped_update`), with `N` the total
  loop count. The registered weights are `1/N`: **0.5 at N=2, 0.333 at N=3 and
  0.25 at N=4**.
- **Points.** O (N=1 reference); R and R-damped on the chosen block at
  N=2,3,4; L after Stage 1 (healed) at **N=2, N=3 and N=4** (D1/D5 metrics at
  L-heal are descriptive); L final at N=2,3,4. O-H after each stage is the
  control (N=1). The L-heal N=3/N=4 points make the difference-in-differences
  measurable at the primary N.
- **Items.** LCB post-cut **182 x 1** + GPQA **198 x 1** for every ladder
  point. **L final N=4 gets the full confirmatory suite** (the primary run);
  its LCB part is the E2/D2 primary run. L final N=2 also runs the full LCB as
  the D2 paired baseline; its 182-item LCB part is a view of that full run, not
  a new generation.
- **Budget.** About **≈37 local GPU-hours without reuse** (registered
  estimate; the N=4 LCB part is reused from D2), plus the L-heal N=3/N=4
  points charged in the evaluation row (**+8–11 padded / +1.5–2 refill** for
  25-29), or **less than 37 h** if campaign stage B is reused with identical
  engine/harness. Campaign results are never reused for E3 or DiD.
- **Chart.** Accuracy vs N with O, R, R-damped, L-heal and L-final lines plus
  O-H references; `progress_charts` `acc_vs_n` is extended with stage labels.
- **DiD.** The post-Stage-1 LCB and GPQA points are the measurements the
  difference-in-differences uses; the primary DiD is at N=4, with N=2
  descriptive. Their budget is in the evaluation row.

---

## 4. Training stages

### 4.1 Stage 0: on-policy sampling of O

O samples the decontaminated pool with the registered evaluation sampling
(temperature 1.0, top_p 0.95, top_k 64, thinking on, max_new_tokens 32768),
four samples per prompt. The pool is about 2,000 code problems with hidden
tests, 2,000 math problems with exact answers and 1,000 chat prompts, for about
64M generated tokens. A sample is kept only if `finish_reason = stop`, the
thought channel is closed, the final answer is non-empty and the rendered length
is <= 8192. The reasoning share of code + math tokens must stay >= 75 %.

**Compute tier (round 8).** The **pilot data may come from the LOCAL
Stage-0-lite sampler** on the padded HF engine: same sampling, per-request
generation cap = `max_length` minus the rendered prompt length. On the same
backend and random stream a capped run preserves an accepted prefix; cross-
backend identity is not claimed from the seeds, and fairness rests on shared
frozen artifacts plus recorded backend provenance. It samples O on bucket 0-49
until at least 3M accepted tokens exist, plus the
held-out verified traces and fit split the pilot needs, and the tier is recorded
as a covariate. The **full Stage 0 for the main runs remains Modal-or-local by
availability** (Modal stock vLLM v0.30.0 when approved, otherwise the registered
local padded engine); the engine is recorded per row and the split assignment is
unaffected. Stage-0-lite is not used for confirmatory evaluation.

### 4.2 Stage 1: heal (L and O-H)

- **Teacher.** Candidates are `google/gemma-4-E4B-it` and the
  `google/gemma-4-12B-it` Google QAT q4_0 checkpoint. The pilot chooses by the
  mean gain of L and O-H, with an E4B tie-break within 0.003 nats. The teacher
  shares the tokenizer except for `<glitch_pruned_249788>`, which is masked.
- **Registered teacher runtimes and qualification rule (round 8; X3).**
  **E4B: bf16 with CPU PLE primary** (text-only, vision tower off GPU, tied
  embedding/head counted once; about 8.65 GiB text weights, verify the
  8,192-token forward against the ~10.28 GiB cap), **int8 fallback**
  (bitsandbytes LLM.int8 with bf16 embedding/head and CPU PLE). **12B:
  NF4 primary** from the QAT-unquantized checkpoint @`b6ed862` (bitsandbytes
  NF4 double quant, bf16 head, ~7.1 GiB); the **official Google QAT Q4_0 GGUF
  executed through llama.cpp** is the registered fallback (f16 K/V, full GPU
  layer offload, sliding-aware cache, bounded all-position exporter; the
  exporter is implemented only if NF4 fails qualification). The ordered
  fallback is E4B bf16 → E4B int8; 12B NF4 → 12B Q4_0/llama.cpp. Qualification
  runs **after the tag and before the pilot**, fail-closed: if no resident
  runtime meets the acceptance rule, **stop before the pilot** and never weaken
  validation or spill dense weights into system RAM. A registered fallback
  that itself passes is not a new representation amendment; any other
  representation change requires a disclosed new version.
- **Qualification corpus and references (round 8b amendment).** Rows are drawn
  from **O's Stage 0 local samples** on the train bucket (verified rows,
  stratified **4 short / 8 common / 4 medium / 4 long = 4/8/4/4**, a 20-row
  screen, then one expansion to 100 rows in seeded
  `sha256(seed:request_id)` hash order), never by teacher losses. They are
  disjoint only from the registered **protected sets** — **pilot evaluation,
  T-fit, MONITOR and confirmatory**; train-pool rows are allowed. The 120-row
  P1-lite corpus is the registered fallback, but protected-set filtering keeps
  only **85 rows**, so it cannot reach the 100-row expansion and a shortfall
  fails closed (exit 2); the 20-row screen is never shrunk. The fallback
  extension is **the 120-row corpus extended on GPU by hash-ordered sampling to
  at least 140 verified rows before the 20-row screen is taken** (mirrored in
  round 8c). A synthetic
  8,192-token memory smoke is added for allocation/causal-boundary checks only. The
  reference is the **same checkpoint in bf16** with full context and the same
  head policy, mask, softcap and temperature; for 12B use CPU bf16 or controlled
  sequential bf16 GPU offload for this small job only. A regular 12B model, or
  four-bit weights dequantized back to bf16, is not the bf16 reference.
- **Qualification metrics and acceptance table.** Full forward KL
  `KL(p_bf16 || q_runtime)` (and reverse KL if full vectors are available),
  common-support KL on the union of each distribution's top-64 plus one
  residual bucket, top-1 agreement overall and on confident reference tokens,
  top-64 overlap, absolute tail-mass error, entropy and teacher-forced
  `log p(O's actual next token)` by thinking/answer span; fp64 accumulation,
  token-weighted with a row-cluster bootstrap. Pass thresholds: **mean full KL
  ≤ 0.010 nat**, **95 % upper ≤ 0.015**, **p95 per-token KL ≤ 0.050**,
  **top-1 ≥ 95 % overall / ≥ 99 % confident**, **tail error mean ≤ 0.002 and
  p95 ≤ 0.010**, all tensors/IDs/masks valid, and long-row allocation below
  the cap with **≥ 0.25 GiB headroom**. If the 20-row result is within 20 % of
  a boundary or has a length/span-specific failure, **expand once to 100
  predetermined rows** under the same thresholds; do not sample until it
  passes.
- **Fairness and freeze.** The teacher comparison is the **best qualified
  resident runtime per teacher** under the same acceptance rule, exact O
  samples, tokenizer/mask, 8,192 context, temperature, target positions and
  storage format. **One frozen runtime per teacher is shared by its L and
  O-H**; never alternate NF4/Q4_0 within a teacher corpus, switch backend after
  observing pilot gains, or train O-H on another runtime's outputs. The
  manifest identity includes method/settings, compute dtype, skipped modules,
  head projection/accumulation policy, attention implementation, software
  versions, chunk sizes and any KV dtype; for GGUF the full file hash,
  upstream revision and conversion/build identities are included. Reusing a
  shard after a semantic change fails closed.
- **X3-C1 execution repair (round 8).** Freeze every loaded teacher parameter,
  wrap the **entire** production precompute including the head in
  `torch.inference_mode()` (or `no_grad`), detach stored probability/tail
  tensors explicitly, and test that outputs have no `grad_fn` and no saved
  autograd tensors. An unquantized head outside the no-grad scope can retain
  fp32 `[8192, 262144]` graphs (~8 GiB) and defeat either quantisation. A shard
  whose manifest does not bind the inference-mode/head policy fails closed.
- **Four-bit limitation (updated).** The 12B teacher is the **QAT-unquantized
  checkpoint loaded with bitsandbytes NF4**, the registered evaluated
  representation; it is not automatically the vendor's Q4_0 execution or its
  reported score. E4B is evaluated in **bf16** (int8 fallback), so the older
  "both teachers are four-bit" statement is retired. The top-64 float16 logits
  plus tail bucket approximate the full distribution, and the capacity gap may
  hurt. The glitch token is masked.
- **Objective: anytime multi-exit (`dial_objective`).** Each step runs T block
  passes, with T=4 at 0.70 and T in {5,6,7,8} at 0.075 each. The coda + LM head
  run after every pass in the last k=3 trainable-pass window; exits are averaged
  uniformly, and a stop-gradient monotone hinge with λ=1 penalises a later
  exit worse than its predecessor. Each exit gets forward KL from the teacher
  (top-k = 64 in float16 plus one tail bucket) plus `beta * KL(O || exit)` with
  `beta = 0.1`, so the parent anchor is applied at **every** exit. The parent
  KL is computed live, full-vocab, with adapters off at N=1. Passes 2..T−3 run
  under `no_grad` and h is detached at the window start (at T=4 the window is
  {2,3,4}, i.e. full BPTT because pass 1 is frozen). The multi-exit KL kernel
  computes the parent LSE once per step and the parent tile per exit.
- **Data.** O's own samples from the Stage 0 train bucket, at most 8192 tokens
  per row. O-H receives the same teacher and rows.
- **Budget and recipe.** Fixed 12M tokens. **Bitsandbytes 8-bit AdamW
  (`adamw_8bit`) on GPU for every arm and stage that trains**; the yaml field
  keeps the legacy `AdamW` spelling, which the loader resolves to `adamw_8bit`
  (CPU-only dev runs use `adamw_torch`). **`run.json` records the resolved
  optimizer** (code change on branch `fix/optim-record`). **Per-device batch 1
  row × gradient accumulation 16 rows per optimizer step**
  (`rows_per_optimizer_step: 16`); the measured tokens per optimizer step are
  recorded in `run.json` (≈26k at the audited mean row length of ≈1,650
  tokens); the token-cosine schedule is unchanged (token-based, 3 % warmup,
  ending exactly at 12M).
  **LoRA LR 2e-4, bridge LR 3e-3, loop_ple LR 1e-3, gradient clip 1.0.** All
  of these values are bound from `prereg.yaml` under `--prereg`; a CLI
   override exits 2. Loss on assistant tokens only (thinking + answer).
   Teachers use the **registered runtimes above**: E4B in bf16 with CPU PLE
   (int8 fallback), 12B from Google's QAT-unquantized checkpoint loaded with
   bitsandbytes NF4 (Q4_0/llama.cpp fallback). Numerics are declared in
   `run.json`. This is a **pre-tag recipe clarification with no data
   involved**.
- **O-H is a real Stage 1 run.** O-H's rank-16 LoRA on the same block layers
  trains; the Stage 1 cost is charged to both arms. O-H carries no recurrent
  bridge and no multi-exit/loop package, so it is a matched-data,
  matched-LoRA-capacity control, not the pure causal isolation of recurrence.
  L's 1,536-parameter bridge and package effects are itemized separately.

**Heal monitoring (round 8b; execution safety only).** The pilot precomputes a
dedicated **MONITOR** corpus: the **64 train-bucket (0–49) problems with the
largest `sha256(seed:problem_id)` keys**, one sample each, excluded from every
training subset and training file (the train target takes the smallest keys).
The pilot also precomputes monitor teacher shards with the **same frozen
teacher runtime, manifest identity and `max_length`** as the arm and passes them
as `--monitor-teacher-data`; the probe fit half uses the MONITOR teacher, never
the training collator. Monitoring observes and never changes a run except
through the registered rules. **ABORT** (**writes no checkpoint**; the last
COMPLETE checkpoint stays intact; writes `alerts.json`; **exit 76 terminal**):
any non-finite loss/gradient/logit; exit-1 bit-exactness failure; teacher-data
integrity failure (> 2e-3 renormalised sums, wrong shapes/ids); parent KL above
1.0 nat/token for three consecutive probes. **WARN**: loss spike > 3× rolling
median for 20 steps; gradient norm > 10× rolling median; teacher KL stall (< 5 %
over 1M tokens); hinge active on > 50 % of **rows** after 1M (a row rule, not
tokens); monitor/train teacher-KL gap growing for three probes; a
generation-sanity probe (N ≤ 4) with repetition loops, or with a missing
answer or an open thinking channel **only when the generation stopped before
the 512-token completion cap** (`finish_reason != "length"`); a
length-truncated trace is expected and is not a sanity failure, while
repetition loops warn regardless of `finish_reason`; bridge
`g` saturation (> 10 % of channels at |g| > 0.5) or all-zero after 1M; memory
within 0.3 GiB of the cap; tokens/s below 50 % of the post-warm-up median. A
**probe exception is a WARN row**; **two consecutive probe failures are an
ABORT**. A WARN alone never changes training, selection, eligibility or resume.
**Resume** needs an explicit `--ack-abort` (recorded in `run.json` and copied to
`orchestrator.jsonl` and `status.json`) on **every registered resume path**;
a registered resume without `--ack-abort <id>` over a run with an abort
record exits 2. With the acknowledgement, if a COMPLETE checkpoint exists
resume from the last COMPLETE checkpoint; otherwise archive the partial run
directory and restart from zero. Either path keeps the same registered run
with unchanged data, dose, thresholds, seeds and recipe; it never authorizes
changing dose/N mix/thresholds/source rows/seeds or rerunning until a
convenient checkpoint. **Technical repair versus new version:** a
technical repair may restore the registered implementation (crash, stale
monitor state, row-rendering or resume bug) without changing data, dose,
thresholds, seeds, model, estimator or selection rule, and is disclosed before
resume; any recipe-changing fix is a new disclosed prereg version.
**Probe configuration (fixed):** an interval probe at the first optimizer-step
boundary where `tokens_seen - last_probe_tokens >= 500,000`, with
`last_probe_tokens` advancing after every probe; a train-end probe when
`tokens_seen > last_probe_tokens` (tokens advanced since the last probe);
never at synthetic positions. At the registered 3M-token pilot dose and the
~26k-token optimizer step the achieved interval probes are **0.52M, 1.04M,
1.56M, 2.08M and 2.60M tokens** (the 20th, 40th, 60th, 80th and 100th step),
plus one train-end probe at **~3.00M tokens** when the incomplete next 500k
window has advanced the count. 32 deterministic MONITOR
fit/score rows; NLL N = 1,2,3,4; generation N = 1,2,4; temperature candidates
0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0 fitted on one half and scored on the other.
All 64 MONITOR IDs stay reserved;
**at least 32 usable rows** are required, else training stops (fail-closed) and
the set is never refilled after observing student metrics. An aborted or
incomplete teacher/arm cell is reported as such and is not replaced by an
informal experiment, a shorter dose, an earlier checkpoint or survivor
selection; a missing cell makes the four-cell teacher statistic
pending/unevaluable. `reserved_ids.json` is a **problem-id** split map
(`monitor`, `pilot_eval`, `pilot_fit`, `heal_train`, `confirmatory`, extra keys
allowed); the monitor proves `monitor ∩ (every other key ∪ training rows) = ∅`
and never treats its own key as an overlap. Nothing the monitor computes is a
selection input, an eligibility input, STOP, or a confirmatory endpoint.

### 4.3 Stage 2: Dr. GRPO RLVR (L and O-H)

- **Algorithm.** Dr. GRPO (no std normalisation, constant length normaliser
  8192) in TRL 1.17.1 `GRPOTrainer`, `num_iterations = 1`, custom rollout
  function around the registered engine.
- **Initialisation and lineage.** Main Stage 1 restarts from **O** for both L
  and O-H; the 3M pilot heals are selection-only and their weights are never
  continued. Stage 2 starts from the final main Stage 1 checkpoint of the same
  arm. The registered 12M Stage 1 dose counts rendered input tokens; rendered
  input tokens, scored assistant tokens and repeated multi-exit targets are
  reported separately, and 3M pilot + 12M main is never described as 12M.
- **Scale.** 16 prompts per step, G = 8, 100 steps: 1,600 prompts and 12,800
  rollouts per arm-seed. Completion cap 8192; overlong rollouts are masked from
  the loss but counted.
- **Sampling.** Temperature 1.0, top_p 1.0, top_k 0. Same prompts, order, G,
  steps, cap, sampling, LR and seeds for O-H and L.
- **N (stratified per prompt group).** Of the 16 groups per step, 3 run at
  N=2, 3 at N=3, 8 at N=4, 1 at N=6 and 1 at N=8. Slot→N is assigned by a
  `sha256(seed:step:slot)` permutation so prompt order and difficulty do not
  line up with N; all G=8 rollouts of a group share N, so advantages stay
  within-N and the baseline never rewards the choice of N. The training forward
  and truncated importance sampling use the group's N; the engine generates in
  **five** per-N batches. For every N the policy-gradient backward uses the same
  **last-k=3 window** as Stage 1 (passes ≤ N−3 under `no_grad`, hidden state
  detached at the window start); full BPTT is not registered. N=1 groups are
  excluded (pass 1 is frozen). O-H stays at N=1. Dev-split cNLL at
  N ∈ {2,4,8} is logged every 25 steps, descriptively and with no stopping
  rule.
- **N contingency (deviation D-7b; pre-tag, after Stage A was shown to the
  author).** The round-7b T-dial-bench Stage 2 proxy OOMs at N=6 and N=8 for
  every block, but that proxy is less memory-efficient than the dial windowed
  forward and is **not** the Stage 2 trainer. **T-dial-rl ships and runs its
  own memory bench before any Stage 2 run**; its configuration is fixed in
  §9 (three candidate blocks; N=6 and N=8; **10,240 tokens per row**, the
  2,048-token prompt cap plus the 8,192 completion cap; last-k=3 window;
  prelude and pass 1 under `no_grad`; chunked head;
  registered gradient accumulation 16). If that bench shows that an N=6 **or**
  an N=8 step does not fit the allocator cap for the chosen block, **both
  groups become N=4, giving the mix 3/3/10 over N=2/3/4** for that block. This
  changes the registered recipe field `rollout_n_groups`; it is disclosed as
  **deviation D-7b** and changes no selection or analysis rule, no endpoint;
  `analysis_rules_frozen_at_commit` stays `8c6d78d`. The trigger is measured
  memory; Stage A content is not an input. The Stage 2 FLOPs and hours for the
  fallback mix are recomputed from the same registered `E[r]` code before the
  run.
- **Feasibility branch (round 8; X1 item 4).** Before either main Stage 2 arm,
  T-dial-rl's own memory bench runs on the **real trainer** at the actual
  **10,240-token maximum row** (2,048-token prompt cap + 8,192-token completion
  cap). It covers **O-H at N=1 and L at the mandatory N=2, 3 and 4 paths**, as
  well as N=6 and N=8, and includes the **actual rollout storage** (prompts,
  completions, per-token old/engine logprobs, masks, rewards, all G=8 rollouts
  and the per-N batch layout) and the **actual two-moment 8-bit AdamW optimizer
  state** allocated by a complete update. It uses checkpointing, the registered
  last-k=3 backward, the chunked head and the actual update batching, with one
  warm-up and one complete update. The command, resolved configuration, code
  hash and result are published before either arm. **D-7b's 3/3/10 fallback
  must itself fit** the same bench. If a **mandatory path** (O-H N=1 or L
  N=2/3/4) does not fit, **STOP before either main Stage 2 arm and issue a new
  disclosed prereg version**; do not shorten one arm, silently discard long
  rows, substitute another N mix, raise the cap, or treat a permanent OOM as a
  crash to retry indefinitely. Repeated-step stability is an execution
  preflight after the one-step "fits" trigger.
- **Reward.** 1 if every hidden test passes in the sandbox, or the exact answer
  matches via math-verify; otherwise 0. No format or length reward.
- **Verifiers.** Programmatic only: unit tests for code and exact answers for
  math, in a CPU-only sandbox with a **6 s** per-test timeout. Hidden tests are
  written offline by DeepSeek V4.1 Flash once and shared by both arms; tests the
  reference solution fails are dropped.
- **Prompts.** Picked once from O's Stage 0 pass rates (1-3 of 4 correct) with
  a **2,048-token prompt cap**: prompts above it are **excluded from the Stage 2
  pool at selection time, identically for both arms**, so the longest training
  row is **2,048 + 8,192 = 10,240 tokens**. The pool is decontaminated. If
  between 1 and 1,599 prompts qualify, cycle the qualifying set in the same
  order for both arms. If **zero** prompts qualify, do not borrow from the
  train/NLL/held-out splits, do not relax the pass-rate or length cap and do
  not add epochs: stop before either main Stage 2 arm and report Stage 2 not
  run. The RFT fallback is used only if the P3 keep rule already selected it;
  otherwise a Stage 2 run needs a new disclosed version.
- **Optimisation.** The same registered **bitsandbytes 8-bit AdamW
  (`adamw_8bit`) on GPU** as Stage 1 (the yaml field keeps the legacy `AdamW`
  spelling; `run.json` records the resolved optimizer). LoRA LR 2e-5, bridge LR
  1e-3, `beta_KL = 0.0`, truncated importance sampling capped at 2.0 against
  engine logprobs.
- **Engine.** The **padded adaptive-batch HF engine**, unconditionally: it is
  the sole operative Part 1 engine for Stage 2 and all evaluation, same for
  O-H and L. `engine_gate` and `part_b` are INACTIVE history with no execution
  authority; no refill run is registered and no post-tag engine addendum is
  required. **N is fixed per engine batch** (the five per-N batches), not per
  arm; N=1 bit-exactness is checked against stock HF.
- **DeepSeek V4.1 Flash (official API only).** `stage2.deepseek.model` is the
  alias **`deepseek-flash`** (served **`DeepSeek-V4.1-Flash-1017`**, with the
  registered fingerprint record `artifacts/stage0/deepseek_fingerprint.json`).
  Used only for (1) writing hidden tests for weakly tested problems, shared by
  both arms, (2) auditing rollouts for reward hacking (reported, never a
  reward), and (3) the creative judge in Part 2. Never a reward or target in
  Part 1, never a training-data source in Part 1. OpenCode-hosted endpoints are
  never used for anything that enters training or results.
- **RFT fallback (registered now).** Supervised fine-tuning on O's
  verified-correct self-traces from the Stage 0 train bucket, 12M tokens,
  **bitsandbytes 8-bit AdamW (`adamw_8bit`; the yaml field keeps the legacy
  `AdamW` spelling)** as in Stage 1, LoRA LR 2e-4, bridge LR 3e-3, the same
  token-cosine schedule and dial
  horizon, multi-exit cross-entropy at the window exits with uniform weights and
  the same hinge, no teacher KL, same evaluation. If the frozen verified rows
  contain fewer than 12M rendered tokens, cycle the same frozen row order to
  exactly 12M and record the epochs; zero usable rows stops before RFT and
  requires a new version, never a borrowed split or an unregistered epoch.

---

## 5. Data

- **Stage 0 pool (Part-1 mode, round 8b).** About 2,000 code problems with
  hidden tests (TACO, APPS, CodeContests pre-2023-05), about 2,000 math
  problems with exact answers (OMR, DAPO-17k, DeepScaleR) and about 1,000 chat
  prompts (the existing decontaminated chat source): **5,000 problems**,
  manifest sha256
  `70053b90fa631fda1e5e0fa77a932e30c4bab80e81ef136f2330adf0daaef84e`.
  The pool is built in the explicit **Part-1 mode without creative briefs**;
  when the Part 2 briefs are written they are decontaminated against the
  frozen pool in the reverse direction. "Synthetic" code is dropped. O samples
  four completions per prompt with the registered sampling, about 64M tokens.
  Source fixes (§R8B-05): **APPS** parquet at `refs/convert/parquet`
  `0f10e424e13e1c2a69f851e153097b71b6734a1f`; **TACO**
  `ALL/train-*.parquet` at pinned `d593ed0a`; **CodeContests** `pre_2023_05`
  Python-only reference solutions.
- **Decontamination.** 13-gram overlap plus Indel >= 0.75 against every
  registered eval (LCB `release_v6`, GPQA Diamond, AIME 2026, MMLU-Pro,
  MMMU-Pro, MATH-Vision, SpatialEval, CV-Bench 2D). Part 1 does not require
  the Part 2 creative briefs; those briefs are checked against the frozen pool
  later, in the reverse direction.
- **Deterministic splits.** One table, assigned once by
  `bucket = int(sha256(problem_id), 16) % 100`, seed 0. The listed counts are
  **nominal hash expectations**; the realized counts in the frozen pool split
  files are:
  - 0-49: Stage 1/2 train pool, nominal 2,500; realized **2,488** rows
    (1,023 math). The 64 MONITOR ids are inside this bucket; after excluding
    them **2,424** usable train problems remain (994 math, 924 code, 506
    chat). `train_problems: 2500` is the **nominal pre-carve** target.
  - 50-59: pilot dev/selection, nominal 500; realized **506**. Sub-allocated
    P3 50-53 realized **210**, selection 54-56 realized **152** (62 math),
    T-fit 57-58 realized **86** (29 math), P1 59 realized **58** (26 math,
    21 code, 11 chat; deterministically truncated to 50: 23 math, 19 code,
    8 chat).
  - 60-69: NLL/KL, nominal 500; realized **486**, sub-allocated into 60-61
    realized **90** (40 math) for the per-(arm, N) temperature fit and D4
    thresholds, and 62-69 realized **396** (162 math) as the scored NLL test
    corpus.
  - 70-79: long traces, nominal 500; realized **507**.
  - 80-99: confirmatory held-out test, nominal 1,000; realized **1,013**.
  These are input counts, not retained verified traces. Buckets are disjoint
  by problem id and disjoint from Stage 1 and Stage 2 data.
- **Shortfall rule (round 8c).** Realized shortfalls never authorize borrowing
  another split, regenerating until favorable, relaxing a filter/verifier, or
  an unregistered extra epoch. Stage 1/RFT and Stage 2 cycle a non-empty frozen
  row/prompt set to the registered dose; zero usable rows/prompts stops and
  requires a new disclosed version. The minima are 20 verified P1 problems, 20
  T-fit, 30 selection, 32 usable MONITOR rows, and 20/40 for D1 fit/score.
- **Length handling.** `max_length = 8192` for **Stage 1, the pilot heals and
  RFT**, filter mode. Stage 2 rows are bounded by `prompt_cap +
  completion_cap = 10,240` tokens and are **not filtered by `max_length`**.
  Stage 0 keeps a sample only under the rule in section 4.1; the reasoning share
  of code + math tokens must stay >= 75 %.
- **Manifest.** `manifest_sha256` is null in this draft; the realised SHA is
  frozen in the `prereg-v1-data` addendum before training. The addendum also
  records the Stage 2 prompt-cap accounting: how many prompts the 2,048-token
  cap excludes and its interaction with the 1,600-prompt cycling rule. Any
  change is a new prereg version and is reported.

**Math grading (round 8b).** A math answer is correct iff **math-verify 0.9.0**
says correct **or** the registered AIME/integer normalisation says correct.
math-verify runs in a grading subprocess with its **internal timeouts
disabled** and a **10 s outer timeout**; if that timeout fires, the normaliser
still votes. `decided_by` records `math_verify`, `normaliser`, `both` or
`timeout`. Why: math-verify's own timeout uses multiprocessing spawn, which
fails on Windows, and it disagrees with the normaliser on `\boxed{1,234}`
versus `1234`; the OR rule prevents a definitive `False` from overriding a
normaliser match.

**Code verification (round 8e; frozen Docker verifier).** Code samples are
verified only with the **frozen Docker verifier**: CPU-only Docker,
`--network none`, read-only root and candidate mount, a 64 MiB `/tmp` tmpfs,
512 MiB memory and memory-swap limits, 64 PIDs, 1 CPU, user `65534:65534`,
`no-new-privileges` and the pinned 6-second per-test timeout. The existing
source tests stored for each problem in the frozen Stage 0 pool are used with
an ordered per-problem test manifest and SHA-256s; no newly generated tests,
changed expected answers, selectively removed tests or sample-specific subsets
are allowed. The extraction, functional/stdin wrapper, output comparison,
grading implementation and dependency versions are frozen, and the image is
pinned by content digest
`python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54`.
A completion is verified only if the nonempty full frozen test set passes with
`sandboxed=true`; invalid/no-test/unsupported/timeout/failure and grader
failure do not pass, infrastructure errors are recorded separately and rerun
only on identical input bytes and configuration, and no unsandboxed fallback is
allowed. The pilot's verified domain is **math plus code passing this frozen
verifier; chat never counts**. The official count is re-run from the same
frozen original samples with the frozen verifier before pilot scoring. The
verifier/test-set specification and the staged-freeze index
`prereg/pilot-shortfall-freeze.json` are **REQUIRED before any pilot scoring or
P2 training**; the rule may be sealed before the completed count/data
addendum, but no scoring or P2 training is allowed while that addendum is
pending. Confirmatory D1 retains its separate **math-only** corpus and
contract.

---

## 6. Endpoints and statistics

### 6.1 Precondition E1 (no alpha, no gate)

L(N=1) is bit-identical to O on **every engine used** (HF trainer and the
evaluation engine). Heal effects on O-H and L are reported as **estimates** with
intervals (NLL margin +/-0.01 nats, LCB margin +/-3 pp, 90 % CI), but E1 is not a
TOST gate. The old "CI includes 0" rule and the gated TOST are retired.

### 6.2 Fixed sequence (confirmatory family)

The primary N is **4**. The graphical procedure is fully specified (Bretz et
al. 2009, Algorithm 1):

- **Weights.** `w = 0.025 × (E2: 0.8, E3: 0, E4: 0, D1: 0.2)`, i.e. E2 starts
  at 0.02 and D1 at 0.005.
- **Transitions.** `E2→E3 = E3→E4 = E4→D1 = D1→E2 = 1`.
- **p-values.** One-sided percentile-bootstrap p-values (10,000 iterations,
  seed 0, paired problem clusters); D1's p-value is the maximum of its four
  component p-values.
- **Recycling.** `E_{k+1}` receives α only after `E_k` is rejected; there is no
  "stop at the first failure" that discards recyclable α.
- **Order.** (1) Test D1 at 0.005. (2) Run the E sequence at 0.025 if D1 was
  rejected, otherwise at 0.02. (3) If E2–E4 are all rejected and D1 was not,
  test D1 at 0.025.
- **Decisions.** Final decisions are taken after the algorithm terminates; the
  result does not depend on the order.

E1 is a precondition with no α; D1 is a separate IUT in the graphical family
(§6.2b). **The confirmatory claims are E2–E4 and D1 only.** The pilot selection
gate H1 and its complement H0 are not confirmatory endpoints and are not in the
α graph; the pilot's `Δ̄_arm` is reported descriptively.

1. **E2, loop effect (primary).** `L(N=4) - O-H` on the full LiveCodeBench
   `release_v6`: 1,055 items, one sample per seed. The 182 post-cutoff items
   (`contest_date >= 2025-01-01`) are a pre-specified subgroup reported
   separately. Primary interval: paired problem-cluster bootstrap. L(N=2) also
   runs the full 1,055 items as the D2 paired baseline.
2. **E3, Total effect versus O: training + loops, L(N=4) - O.** `L(N=4) - O`
   on the same items. Because L(N=1) is O bit-exactly, this is the total
   training-plus-loop effect versus the original; E2 is the loop contrast. The
   author delegated the naming to the reviewer panel, which recommended this
   wording; E3 is kept.
3. **E4, budget-curve AUC.** The grid is defined on O-H's budget: O-H at
   `B = 1024 * 2^j`, j = 0..5, and L at **`floor(B / r_4)`**, with
   r_4 = 2.2824 (15-29), 1.4275 (25-29) or 1.2515 (16-18). A completion counts
   as correct at a budget only if it is solved with total tokens ≤ that budget.
   Both arms stay ≤ 32,768 and both are re-scored from saved completions. AUC
   is the trapezoid over j divided by 5.

### 6.2b Dial family (D1, confirmatory)

D1 is an intersection-union test (Berger 1982): reject only if all four
component tests reject at α_D; no within-D1 adjustment is needed.

- (a) **Gain:** `cNLL(1) - cNLL(4) > 0`.
- (b) **No adjacent regression:** `cNLL(N+1) - cNLL(N) < m = 0.005`
  nats/token for N = 1, 2, 3. The margin is half the registered heal margin.

The corpus is **O's dedicated sampling on the scored NLL test bucket 62-69**
(400 problems), separate from the Stage 0 training pool (bucket **0-49**): four
samples per problem with the registered sampling, kept only if the sample
passes the Stage 0 filter and its answer is verified — **math via math-verify**.
Code problems are excluded because no hidden-test harness is registered for
this corpus; chat is excluded. The **lowest-index verified-correct trace per
problem** is scored, so the usable corpus is the verified math subset of those
problems. The realized buckets are **60-61 = 90 problems (40 math)** and
**62-69 = 396 problems (162 math)**; these are input counts before
verification and length exclusions. The dedicated 60-69 fit/test targets are
an explicit post-tag, pre-pilot `prereg-v1-data` obligation: the registered
Stage 0 data sampler (Modal stock vLLM v0.30.0 when approved/available,
otherwise the local padded adaptive-batch HF engine; the data-generation
engine is recorded per row and is not an evaluation engine) generates them
with the registered sampling, four samples per problem, `max_new_tokens =
32768` and the Stage 0 rendered-length filter of 8192. Their trace IDs,
verifier outcomes and counts are frozen in the data addendum before any
trained-arm score; the Stage 0 training-pool sampler does not cover these
buckets. A usable fit corpus requires at least 20
verified distinct problems and the score corpus at least 40; below that D1 is
unevaluable with no α transfer.
Temperature is fitted per **(arm, checkpoint, N)** on bucket **60-61** with the
single registered estimator in §6.3; D1 and D5 run at both L-heal and L-final,
so the checkpoint is part of the key. The bootstrap is paired problem-cluster with 10,000 iterations and
one-sided percentile p-values; the checkpoint is **L final** (L-heal
descriptive). D1 starts at α_D = 0.005; after the whole E sequence passes it is
tested at 0.025. **Claim (round 8):** D1 establishes average
calibrated-likelihood gain and non-inferiority **on O-verifiable math traces
only**; it is not a general "usable reasoning dial", not per-token
monotonicity, and not a guarantee for hard problems O cannot solve, code,
arbitrary N or generated continuations. A D1 pass alone must not produce the
headline "a usable reasoning dial" or "every extra loop helps".

### 6.2c Dial estimation endpoints (D2–D5, no gate)

- **D2.** `L(N=4) - L(N=2)` on the full LCB 1,055 x 1, paired 95 % CI and the
  `-3 pp` non-inferiority margin reported but not gated. This is the
  **extra-depth effect** of N=4 over N=2; a flat tail or a non-significant
  difference is reported as a plateau or an unresolved effect, never as proof
  that every extra loop helps. L(N=1) needs no run (it is O's run). The paired
  SE estimate is 1.2–1.4 pp with 15–20 % discordant items, so a few-pp
  superiority claim is out of reach. The N=4 run is the confirmatory E2 run and
  the N=2 run is the paired baseline.
- **D3.** Effort levels and the FLOPs frontier: accuracy at N = 1, 2, 4 on the
  full LCB and N = 3 on the ladder items; the budget surface `acc_L(N, B)`
  from saved completions (B = 1k..32k); the FLOPs-matched Pareto frontier
  `max_N acc_L(N, F/r_N)` versus `acc_O-H(F)`, as an AUC difference with CI.
  This is the loops-vs-thinking-tokens comparison and a **secondary FLOPs
  axis**, never the intelligence-per-byte metric.
- **D4.** Adaptive rules, exact and teacher-forced: the hidden-change rule
  (ε fitted for mean N_eff = 2, 3 **and 4**) and the exit-1 entropy gate
  (τ fitted to match the analytic FLOPs of fixed N=2, check cost included),
  each versus fixed N at matched analytic FLOPs, with a 95 % problem-cluster CI
  and the oracle bound. **Define ΔcNLL = cNLL(fixed N) − cNLL(policy)**; for
  each token use the per-(arm, checkpoint, exit-N) temperature of the exit that
  produced it. **The winning policy is chosen by a rule fixed now:** the
  largest mean ΔcNLL at matched analytic FLOPs whose 95 % CI lower bound is
  > 0 (ties: lower mean bytes read per token, then hidden-change). If no policy
  qualifies, `auto` = fixed N=4 and the table reads "no adaptive winner".
  Because the winner is chosen on the same scored data, its selected ΔcNLL and
  interval are **exploratory/selected/optimistic**, not an unbiased
  selected-policy interval; it enters the per-byte table as a selected estimate
  and becomes only a **provisional** Part 2 `auto` serve level, requiring
  **fresh-data validation before any serving claim**.
  **D4 provides teacher-forced likelihood evidence only**; it cannot turn the
  same saved completions into the policy's generated answers, so
  **adaptive-decode accuracy is unmeasured in Part 1** and remains a named
  Part 2 milestone.
- **D5.** cNLL at N = 6, 8 (trained) and N = 12, 16 (untrained); collapse flag
  if `cNLL(N) > cNLL(1) + 0.05`; ‖Δh_t‖ curves and the period-2 ratio.

### 6.2d Intelligence per resident byte (D6, estimation only)

The author's objective is intelligence per resident byte. The metric is
registered in `resident_bytes` and `intelligence_per_byte` and never gates
E1–E4.

- **Denominator R.** Every tensor that must be resident, GPU or CPU, to
  produce the scored output, at the dtype loaded in the evaluation process:
  text decoder **prelude 0..s−1, block s..e and coda e+1..34** weights
  (including layer scalars and final norm), tied embedding/LM head, the
  CPU-side PLE table, the per-layer projection, LoRA A/B (unmerged for L
  because pass 1 is adapter-free; O-H merged = 0 bytes primary), bridge g, and
  `loop_ple` if present. Expected bf16 bytes per candidate block
  (prelude/block/coda): 25-29 = 2.446/0.659/0.659 GB, 16-18 =
  1.257/0.387/2.120 GB, 15-29 = 1.128/1.977/0.659 GB. KV, activations and the
  CUDA context are excluded; KV bytes are reported separately at 8k and 32k.
  Vision is counted only for vision benchmarks, audio never. R is measured as
  `Σ numel × dtype_bytes` from the loaded state dict plus the CPU-side PLE
  table and written to `run.json`; all units are **GiB**. bf16 primary:
  O-E2B and L(N=1) = **8.657 GiB** (GPU-side 4.282 + PLE 4.375); L adds +7.9 /
  4.7 / 23.7 MB of LoRA (25-29 / 16-18 / 15-29). Invariants:
  `B(L, N=k) = B(L, N=1)` for k = 2..8 and
  `B(L, N=1) = B(O) + B(LoRA) + B(g)`.
- **Merged forms.** L cannot be merged without breaking pass 1 ≡ O: a merged L
  needs both W and W+BA resident, `R + W_block` (+0.614 GiB for 25-29, +0.360
  for 16-18, +1.841 for 15-29). **L unmerged is the byte-minimal deployable
  form; "L merged" is a sensitivity.** O-H merged = R(O) with the LoRA at 0
  bytes (primary); O-H unmerged is a sensitivity.
- **Numerator.** LCB `release_v6` 1,055 x 1 pass@1 (pp) with a problem-cluster
  bootstrap CI; calibrated NLL (nats/token) on bucket 62-69 with one
  temperature per (arm, checkpoint, N) fitted on 60-61, on O's verified-correct
  traces.
  The two axes are never merged. The derived scalar is
  `pp_per_GiB = pass@1 / R`, shown only next to both components; no nats/GiB is
  registered because a signed numerator over bytes is uninterpretable.
- **Fixed-bytes effects.** `Δ_loop(N) = L(N) − O-H` and
  `Δ_tot(N) = L(N) − O` on LCB full at N = 1, 2, 4 (N = 3 where run), plus GPQA
  and MMLU-Pro **only where registered**: at N=1 for O and at N=4 for L. There
  is **no GPQA or MMLU-Pro run at N=2**, so no N=2 secondary-bench effect is
  reported. **The main presentation is the pair of fixed-byte paired effects
  (Δ_tot(4), Δ_loop(4)) in pp plus ΔcNLL(4) in nats/token, at ΔR ≈ 0.** There
  is no per-GiB division of the differences; the absolute `pp_per_GiB` scalar
  is a separate model-level description. Because L and O-H differ by < 0.3 % in
  R, these are paired differences at fixed bytes.
- **Anchoring (illustrative only).** `φ_loop(N) = Δ_loop(N) / G` and
  `φ_tot(N) = Δ_tot(N) / G`, where G is Google's E4B − E2B gap (LCB 8.0,
  GPQA 15.2, MMLU-Pro 9.4 pp), a constant labelled "reproduction-check level".
  Their multipliers are `m_loop = ρ^φ_loop` and `m_tot = ρ^φ_tot`, with
  `ρ = R_E4B/R_E2B = 1.62` **computed from both pinned safetensors headers**:
  `R_E2B = 8.657 GiB` (E2B `4adf157`) and `R_E4B = 14.003 GiB` text-resident
  (E4B `4ecc2ba`, `huggingface_hub.get_safetensors_metadata`: 8.753 GiB
  GPU-side text + 5.250 GiB CPU PLE; 7,518,069,034 text params; whole
  checkpoint 14.894 GiB; sensitivities 2.04 R_GPU and 1.56 R_ckpt). **φ and m
  are illustrative only**, never a headline proof of intelligence density or a
  larger-model-equivalence claim. Interpreting φ as a transferable gap share
  assumes comparable benchmarks/protocols and treats uncertain published
  scores as constants; m adds a scaling-law assumption from two model sizes.
  **Google's same-protocol numbers appear only as a labelled annotation, never
  as a point on the bytes axis**, and the `pp_per_GiB` scalar is model-level
  only (E4B is an annotation). The fixed-byte paired effects carry the Part 1
  result.
- **Secondary.** Analytic `B_read(N, c) = W_base + (N−1)·(W_block + W_LoRA) +
  KV_base(c) + (N−1)·KV_block(c)`, with W_base 4.282 GiB, W_block 0.614 /
  0.360 / 1.841 GiB and **KV_block(c) summed over the block's full and sliding
  consumer layers**: `n_full·2 KiB·c + n_slide·1 KiB·min(c, 512)`, with
  (n_full, n_slide) = (1,4)/(0,3)/(3,12) for 25-29/16-18/15-29; r_bytes for
  N = 1..8, 12, 16 and c ∈ {1k, 8k, 32k}. Also a measured batch-1 latency
  micro-benchmark on the RTX 5070 (≤ 0.5 GPU-h, run before the tag on the
  padded engine for all three blocks; no refill re-run exists),
  descriptive only.
- **Reference points.** O-E2B N=1; O-H N=1 after each stage; L-heal at N=2, 3
  and 4 on the ladder items; L-final at N=2 (LCB full), N=3 (182 post-cut) and
  N=4 (LCB full); the D4 winner's mean-N policy on the same completions; E4B
  only as a labelled annotation — no local bf16 accuracy run in Part 1 (≈8.8
  GiB weights + ≈1.9 GiB KV at 32k exceeds the 10.28 GiB cap) and q4_0 is a
  different model.
- **Limitation.** Part 1 makes **no cross-precision per-byte claim**: every
  byte count is at the evaluated precision (bf16), and the q4_0 E2B
  sensitivity is dropped. R_E4B is **computed from the safetensors header of
  `teacher_e4b_revision` `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` with
  `huggingface_hub.get_safetensors_metadata`** (14.003 GiB text-resident,
  8.753 GiB GPU-side text + 5.250 GiB CPU PLE; whole checkpoint 14.894 GiB),
  and **ρ = 1.62 is frozen**. The precision-matched 4-bit frontier is Part 2
  (§8). D6 is estimation only, priority after E4. Google's gap fractions φ and
  the byte-equivalent multiplier m are illustrative only.

### 6.3 Statistics

**Calibrated-NLL analysis contract (round 8; X1 Critical issue 2).** D1's
estimand is fully executable:

- **Supervised-token mask.** Score assistant **thinking plus answer** tokens
  with a fixed mask; the same retained problems and the same mask are used at
  every exit and checkpoint (L-heal and L-final).
- **Token versus problem weighting.** Primary: **token-weighted NLL with
  problem-level resampling**,
  `C_N = sum_i S_iN / sum_i n_i`, where `S_iN` is the summed calibrated token
  loss and `n_i` the scored token count for the single retained trace of
  problem `i`. Token counts travel with their problem under resampling;
  independent-token resampling is forbidden. An equal-problem averaging
  alternative is declared but not used; if used it would have to be declared
  before the tag because the two estimands can rank arms differently.
- **Statistic.** Positive-oriented `T_gain = C_1 − C_4` and
  `T_adj,N = 0.005 − (C_(N+1) − C_N)` for N = 1, 2, 3. Resample paired problems
  and recompute each ratio statistic keeping the token counts with each
  problem. D1 uses the maximum of the four component p-values.
- **Exact p-value construction and boundary convention.** One-sided
  percentile-bootstrap
  `p = (1 + number of bootstrap T* <= 0) / 10001`, with conservative tie
  treatment. Equality at α is non-rejection. Percentile interpolation is not
  used (the empirical bootstrap distribution is used directly). This is an
  approximate bootstrap test, not an exact finite-sample guarantee.
- **Fitted temperatures (round 8c: one bound estimator).** Fit only on the
  independent fit split (bucket 60-61) with
  `gemmaloop.eval.calibrated_nll.fit_model_temperature` (equivalently the model
  path of `calibrated_nll_report`), the **model-level** path, **not** the
  tensor helper with its different defaults. Algorithm: one scalar temperature
  per (arm, checkpoint, N) minimising fp32 mean NLL by a bounded golden-section
  search. Bounds **(0.5, 4.0)**, `max_iter = 40`, `tol = 5e-3`. Source pin:
  commit `54cab07` + blob `f51de454…` of
  `src/gemmaloop/eval/calibrated_nll.py` (`pins.temperature_fit_algorithm`);
  round 8c changes no code. The **same estimator and defaults** are used in
  pilot and confirmatory scoring. The primary bootstrap **holds the fitted
  temperatures fixed** and inference is conditional on them. A separate
  fit-split-resampling sensitivity is optional; the test loss never chooses a
  calibration fit. Failure behaviour: an empty labelled fit set raises before a
  fit; a non-finite objective, malformed/mismatched denominator or unusable fit
  makes that comparison unevaluable; a boundary fit is reported, never widened,
  and no alternative estimator or post-result tuning is allowed.
- **Unevaluable-corpus rule.** Freeze trace IDs, verifier outcomes, lengths,
  actual distinct-problem counts and exclusions in the data addendum before
  any trained-arm scoring. A usable fit corpus needs at least 20 verified
  distinct problems and the score corpus at least 40; an empty or
  below-minimum corpus produces **D1 unevaluable: no rejection, no α transfer
  from D1**, and no post-result replenishment or substitution. The realized
  fit/test buckets are 90/396 (40/162 math), but "400 problems" is the nominal
  whole bucket, not the effective D1 sample; the usable verified math subset
  is expected to
  be about 80–110 problems before length exclusions, and **D1 power is
  unestablished**.
- **Frozen fit/test IDs and denominators.** Realised fit/test trace IDs and
  counts are frozen in `prereg-v1-data` **before any trained-arm output is
  examined**. Every compared arm and exit reports its actual distinct-problem
  count and scored-token sum; a missing or mismatched denominator makes that
  comparison unevaluable rather than silently dropping the row.

- **Bootstrap.** 10,000 iterations, seed 0, unit = **problem cluster**.
  Confirmatory verdict on seed 0 only. With seeds [1, 2], the crossed
  (pigeonhole) bootstrap over problems x seed pairs is a **robustness** check,
  never the primary.
- **Multiplicity.** Graphical approach (Bretz et al. 2009, Algorithm 1),
  one-sided α = 0.025 in total: weights `0.025 × (E2 0.8, E3 0, E4 0, D1 0.2)`
  (E2 0.02, D1 0.005); transitions `E2→E3 = E3→E4 = E4→D1 = D1→E2 = 1`.
  One-sided percentile-bootstrap p-values (10,000, seed 0); D1 p = max of its
  four components. `E_{k+1}` receives α only after `E_k` is rejected; final
  decisions after the algorithm terminates. Operational order: D1 at 0.005;
  E at 0.025 if D1 was rejected, else 0.02; D1 at 0.025 only if E2–E4 all
  reject and D1 did not.
- **Holm.** N=2 (full LCB) and N=3 (182 post-cut items) versus O-H are
  exploratory and Holm-corrected as one family on LCB **at one-sided 0.025**
  with paired problem-cluster bootstrap p-values, **outside the confirmatory
  FWER**; monotonicity in N is never assumed. N=4 is the confirmatory primary
  and is not in the Holm family.
- **MDE.** Section 7 uses `gemmaloop.eval.stats.mde` with alpha = 0.05
  bilateral (one-sided 0.02 for the E2-if-D1-fails row), power = 0.80,
  omega2 = 1/12, sigma2 = 1/6.
- **Single-seed limitation.** The confirmatory verdict is computed once, on seed
  0; item-level intervals do not include training-seed variance. Every
  confirmatory claim says so. Seeds 1 and 2 are robustness-only.
- **Common random numbers.** Per-request seeds are `sha256(seed, request_id)`
  and the `request_id` never contains the arm name.
- **Google comparison.** A **reproduction check**, never "validation": a forest
  plot of `mine - Google` with intervals and a table of protocol differences.
  `lcb_google` is a virtual harness view of O's full 1,055-item LCB run; no
  separate 182-item generation is run. It never enters E2-E4 and does not gate
  L versus O-H.
- **Forking paths.** The pilot rule, prompts and parsers are frozen at the tag,
  before any trained-arm evaluation. The data manifest SHA is frozen in the
  addendum. The realised filter is declared.

### 6.4 Evaluation priority order

1. E2/E3: O, O-H and L(N=4) on the full LCB 1,055 x 1 (L(N=2) on the same
   items is the D2 paired baseline).
2. D1/D4/D5: multi-exit NLL sweep N=1..16 on buckets 60-69 (L-heal and
   L-final).
3. E4: re-score saved completions at O-H's `B = 1k..32k` and L's
   `floor(B/r_4)`.
4. D2: L(N=4) − L(N=2) on the full LCB 1,055 x 1.
5. D6: byte frontier (resident bytes, φ_loop/φ_tot, m, bytes read per token,
   batch-1 latency ≤ 0.5 GPU-h).
6. L-heal N=3/N=4 on the ladder items (the primary-N DiD).
7. R and R-damped on the 182 post-cut items x 1.
8. Secondary: GPQA, MMLU-Pro, vision (at N=4 for L).
9. O-only: AIME and spatial.

**Drop order.** If the padded engine is registered, this order is also the
drop order: drop from the bottom while preserving E2/E3, E4, D1–D2 and the D6
denominator.

---

## 7. MDE table

Computed from `gemmaloop.eval.stats.mde` with the registered `n` and `k`
(samples per item), alpha = 0.05 bilateral, power = 0.80, omega2 = 1/12,
sigma2 = 1/6. Units are percentage points.

| Benchmark | n (items) | k | MDE (pp) |
|---|---:|---:|---:|
| LCB `release_v6` full | 1,055 | 1 | 5.57 |
| LCB full, E2 if D1 fails (one-sided 0.02) | 1,055 | 1 | 5.75 |
| LCB `release_v6` minimum 728 | 728 | 1 | 6.70 |
| LCB post-cutoff subgroup | 182 | 1 | 13.40 |
| `lcb_google` (virtual full view) | 1,055 | 1 | 5.57 |
| GPQA Diamond | 198 | 4 | 8.13 |
| MMLU-Pro | 2,000 | 1 | 4.04 |
| MMMU-Pro Standard(10) | 300 | 1 | 10.44 |
| MMMU-Pro Vision | 300 | 1 | 10.44 |
| MATH-Vision | 300 | 1 | 10.44 |
| AIME 2026 (O only) | 30 | 8 | 18.08 |
| SpatialEval VQA (O only) | 4,635 | 1 | 2.66 |

The primary LCB endpoint has a conservative MDE of 5.57 pp at 1,055 items and
6.70 pp at the 728-item minimum. If D1 fails, E2 is tested at one-sided 0.02
(two-sided 0.04) and its MDE rises to **5.75 pp** at 1,055 items. N does not
enter the MDE (n and k do), so moving the primary from N=2 to N=4 leaves the
item-level MDE unchanged. D2's paired accuracy SE is an estimate, not an MDE:
`sqrt((0.15–0.20)/1055) = 1.2–1.4 pp`; the −3 pp non-inferiority margin is
reported but not gated. The post-cutoff subgroup is descriptive only. AIME
cannot detect the effects of interest and is scored on O only.

---

## 8. Part 2: exploratory iteration (rules only)

Part 2 starts from L-s0 (and O-H-s0 where cheap). It uses versioned checkpoints
with lineage and may use:

- sequence-level (black-box) distillation from DeepSeek V4.1 Flash via the
  **official API**, verification-filtered where a checker exists,
  length-controlled, and mixed with the model's own verified traces;
- RL judged by DeepSeek with rubric templates, plus automatic checks where they
  exist (unit tests, exact answers; the controller's recommendation adopted in
  Ruling 359);
- the author's blind hand ratings of creative outputs at every checkpoint;
- a detached learned exit gate (Ouro stage-II labels; a 1536→1 probe on stored
  features from the D4 forward, LM frozen);
- gate-only RL with a compute penalty (LM frozen; clean credit assignment);
- adaptive decoding with per-row N in the refill engine (masked or gathered
  active rows; the KV cache is untouched);
- an exit-1 entropy gate and the D4 winning adaptive policy as the `auto`
  serve level;
- effort-level serving **low/medium/high/max/auto** (low = N=1 = O exactly,
  medium = N=2, high = N=4, max = N=8, auto = the D4 winner), a new control
  axis beside `enable_thinking`;
- span-subsampled exits (coda and head on a random 2,048-token span; legal
  because consumers are token-independent);
- a warm-up adapter (adapter A on pass 2, adapter B shared from pass 3) or a
  clamped pass embedding;
- a **precision-matched 4-bit byte frontier**: O-E2B and L-E2B with 4-bit block
  weights (LoRA bf16) versus E4B 4-bit (QAT or nf4) at matched resident bytes,
  including whether the dial survives quantization (the block's quantization
  error is re-applied on every pass);
- agentic RL in OpenCode;
- checkpoint and LoRA merging.

**Named Part 2 milestone: adaptive-decode accuracy.** Before any adaptive
result is claimed, the registered adaptive policies are measured end-to-end
(accuracy, calibrated NLL, analytic and executed bytes read per token,
latency) against fixed N at matched analytic FLOPs.

Rules:

- Decisions are made only on dev sets. The dev composite is the mean dev pass@1
  on code and math; creative ratings are reported separately and are not folded
  into the composite.
- Held-out test benchmarks are frozen and scored only at the **four named
  milestones**: end of the Part 1 seed-0 pair, the first exploratory checkpoint
  that improves the dev composite, the **adaptive-decode accuracy** milestone
  (which does score held-out benchmarks), and the final Part 2 checkpoint. No
  other held-out scoring is allowed.
- Every iteration is logged with data, technique and results, including
  failures.
- Results are reported as development, never as Part 1 confirmatory claims.
- A judge is never the sole evaluator of anything it trained.
- OpenCode-hosted endpoints are never used for data or results. OpenCode Zen
  free usage is only for plumbing or dry-run tests whose outputs are discarded.

DeepSeek Open Platform terms section 4.2(3) allow training other models
(distillation) and section 4.2(2) assigns output rights to the user (effective
2026-04-29).

---

## 9. Deviations policy

Any change after the `prereg-v1` tag is a new version (`prereg-v1.1`, ...) with
a written reason in the changelog. Silent changes are not allowed. The pilot
rule and the endpoint order are frozen before the pilot. **The freeze sequence
is explicit (round 8):** (1) **rules** — analysis and selection rules are
frozen at the tag; the analysis rules were already frozen at `8c6d78d`, the
historical pre-Stage-A snapshot; (2) **data** — the realised manifest SHA,
trace IDs, verifier outcomes and fit/test IDs are frozen in `prereg-v1-data`
after the tag and **before the pilot**; (3) **pilot selection** — the selection
rule is frozen at the tag and the selected configuration is frozen **before
main training**; (4) **implementation** — the resolved recipe/binding and
`analysis-v1` are frozen **before any trained-arm evaluation**; (5) **main
runs** — main training and evaluation start only after steps 1–4. Prompts,
parsers and the chosen architecture are therefore not all "frozen at the tag";
the earlier shorthand is retired. The public decision log is exported to
`docs/prereg-v1-decision-log.md` at the tag. The paper reports every deviation.
**Round 8b is a pre-tag technical amendment made after Stage A was exposed; it
uses no trained-arm outcome and changes no selection, endpoint or analysis
rule.** It corrects registration details (monitor safety, counts, stages,
grading, pool/source provenance, engine closure, qualification corpus, code
verification, API terms and pins), so it is not a new analysis version and does
not make already-viewed exploratory results prospective.

**Pre-tag deviations from the `84c5da0` freeze.** The selection rule (rule B
with the corrected eligible set and `Δ̄_arm` ranking) and the primary N change
(N=2 → N=4) were adopted after the freeze. Both are disclosed in §0.1 and in
the paper's deviations table, with the exact Stage A exposure audit and the
timestamp at which each decision was made.

**Deviation D-7b (pre-tag, after Stage A was shown to the author).** Round 7b
(`d716ec0`, 08:00:11) replaced the round-6 N=8 → N=6 Stage 2 contingency with:
if the **T-dial-rl memory bench** shows that an N=6 **or** an N=8 step does
not fit the allocator cap for the chosen block, both groups become N=4 (mix
**3/3/10** over N=2/3/4). This changes the registered training-recipe field
`rollout_n_groups`, so it is a **disclosed recipe deviation**, not an
execution-only amendment.

- **The change.** Stage 2 contingency v2 replaces the round-6 N=8 → N=6 rule;
  the base mix `{2:3, 3:3, 4:8, 6:1, 8:1}` is unchanged unless the bench
  triggers the fallback, in which case both groups become N=4.
- **Trigger and timing.** The trigger is **measured memory**:
  `perf/dial_bench.json`, sha256
  `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1` (13
  records), shows the Stage 2 proxy OOMs at N=6/N=8 for every block while the
  windowed dial step at T=8 fits. It was decided after Stage A was shown to the
  author: Stage A was sealed by `prereg/stage-a-addendum.md` (commit
  `38b5bd2`, 05:56:26) and the charts/results were shown in chat at **≈06:02**
  (SendUserFile); the author had not replied at the time of this draft.
  **Stage A content is not an input.**
- **What the T-dial-rl bench must register before any Stage 2 run.** The bench
  and its sha256 are committed in a pre-Stage-2 addendum. Its configuration is
  fixed now: the three candidate blocks (25-29, 16-18, 15-29); N=6 and N=8; a
  row of **10,240 tokens**, the 2,048-token Stage 2 prompt cap plus the 8,192
  completion cap (the round-7c text used 8,192 tokens including the prompt;
  round 7d corrects it to the longest real row, **D-7c**); the last-**k=3**
  window; **prelude and pass 1 under `no_grad`**; the **chunked head**; the
  **registered gradient accumulation (16)**.
- **The "fits" test.** One full optimizer step completes with **no OOM** and
  **peak reserved memory below the allocator cap** (10.28 GiB on the bench
  device).
- **Effect on the analysis.** None on endpoints or analysis. D5 still reports
  N=6/8 as trained in Stage 1 only, and the "trained" label is qualified if
  the fallback fires.

**Deviation D-7c (pre-tag, after Stage A was shown to the author).** A numbered
disclosed deviation records three round-7d changes, all made after the Stage A
results were shown to the author in chat at **≈06:02** (SendUserFile; the
author had not replied). Stage A had been sealed in
`prereg/stage-a-addendum.md` (**`38b5bd2`**, 05:56:26) before any result was
opened; **Stage A content is not an input**, and
`analysis_rules_frozen_at_commit` stays **`8c6d78d`** because no analysis rule
changes; item 3 is the only new selection rule, disclosed here.

1. **`rows_per_optimizer_step` correction.** The text registered since **round
   1** (`19a1fd9`; still present at `d716ec0`) said "8,192 tokens per optimizer
   step"; **round 7c** (`2f36330`, 08:28:48) corrected it to the trainer
   recipe: **per-device batch 1 row × gradient accumulation 16 rows per
   optimizer step** (`rows_per_optimizer_step: 16`, ≈26k supervised tokens at
   the audited mean row length).
2. **8-bit AdamW declaration.** The registered optimizer is **bitsandbytes
   8-bit AdamW (`adamw_8bit`) on GPU for every arm and stage that trains**.
   The yaml field keeps the legacy `AdamW` spelling, which the loader resolves
   to `adamw_8bit` (CPU-only dev runs use `adamw_torch`), and **`run.json`
   records the resolved optimizer**; the code change is on branch
   **`fix/optim-record`**.
3. **Stage 2 prompt cap — a new selection rule.** Stage 2 registers a
   **2,048-token prompt cap**; prompts above it are **excluded from the Stage 2
   pool at selection time, identically for both arms**. This is a **new
   selection constraint**, not a correction to a recipe already in use: it was
   chosen for memory before any Stage 0 or Stage 2 prompt selection and applies
   equally to both arms. A Stage 2 row is therefore at most
   **2,048 + 8,192 = 10,240 tokens**, and the T-dial-rl memory bench and its
   **"fits"** test use **10,240-token rows**. The **`prereg-v1-data` addendum**
   records how many prompts the cap excludes and how it interacts with the
   **1,600-prompt cycling rule**.

**D-7b and D-7c acknowledgements (round 8).** Both deviations are
**acknowledged** through the author's delegation to the reviewer panel. A
substantive acknowledgement accepts disclosed changes; it is not evidence that
nobody could have been influenced by results. D-7b openly changes the training
recipe after Stage A exposure, cites measured memory and the sealing record,
and its trigger is outcome-independent; the mandatory N=4/O-H feasibility
boundary is added separately in §4.3 and D-7b is not retracted. D-7c aligns
the registered recipe with intended execution and correctly identifies the
2,048-token prompt cap as a new selection constraint applied to both arms; the
`prereg-v1-data` addendum reports the excluded count, source composition and
resulting repetitions.

**Stage A visibility.** Stage A completed at 05:56:08 and was sealed by
`prereg/stage-a-addendum.md` (commit `38b5bd2`, 05:56:26) with every output's
sha256/mtime/size **before** any result was opened. The charts (`d1fde78`,
06:01:44) and results were shown to the author **in chat at ≈06:02**
(SendUserFile); the author had not replied at the time of this draft. The
analysis rules were frozen at the round-7a commit (§0.1) **before**
completion. **Later auditors and reviewer X1 did open the sealed outputs after
the seal and before the tag** (round-7a/7b native verification at ≈08:01);
§0.1 and the YAML `post_seal_access_record` carry the corrected record. Hashes
demonstrate artifact integrity, not who knew what. Any change requested after
the author sees Stage A is a disclosed deviation in this section's terms,
citing those hashes. Stage B remains blocked until the tag.

---

## 10. Compute and feasibility

The round-7c seed-0 local envelope depends on the engine and the block:
25-29 **padded 602-941 h / refill 237-383 h**; 16-18 **padded 596-930 /
refill 234-377**; 15-29 **padded 654-1,016 / refill 259-416**. The 15-29
figure is the recomputed base (~650-1,000, charging its larger r_4 through the
D2 increment) plus the teacher precompute row. The ladder is **≈37 h without
reuse**, plus the L-heal N=3/N=4 points charged in evaluation.
The Stage 1 row now uses the measured T-dial-bench throughput, but it is still
an **estimate from one timed optimizer step**, not a full-run measurement: it
narrows the training component to **8.6 h (25-29), 10.0 h (16-18) and
10.8-12.0 h (15-29)**. Round 7c adds the **4-16 h teacher top-k precompute**
row and recomputes the totals from those substitutions.
**Stage-0-lite is the registered Part 1 path (round 8b):** the pilot data comes
from the **LOCAL Stage-0-lite sampler** on the padded HF engine (same sampling,
cap = `max_length` minus prompt length; a same-backend capped run preserves an
accepted prefix, while cross-backend identity is not claimed), recorded as a
covariate, and **Modal is optional for
Part 1**. The first committed shard measured **100 requests in ≈2,600 s, ≈195
tok/s, 52/100 kept**; with about 1,250 non-train plus about 1,150 train
requests this is **≈17-20 h**. Full Stage 0 for the main runs remains
**Modal-or-local by availability** (Modal stock vLLM v0.30.0 when approved,
otherwise the registered local padded engine), with the engine recorded per row
and the split assignment unaffected.

**Modal probe history and guards (round 8b).** Probe attempt 1 failed at launch
before any container ran (`InvalidError`: the image called `.env` after
`add_local_python_source`; reservation $3.09, actual $0.25). Attempt 2 cycled
**six failed startups over 32 minutes** with `ModuleNotFoundError: rapidfuzz`
(the remote import closure reached `gemmaloop.data.decontam`) and was **stopped
by hand**; the ledger booked the full reservation **$3.087 conservatively**
(month total $3.34 of the $30 usage limit). Required before any Modal launch:
explicit author approval for that launch and its cap, the frozen pool manifest,
a **CPU-only preflight** of the exact remote import closure before any GPU
function, the image-completeness AST test, **fail-fast** stop on the first
startup/preflight/remote failure with `app.stop()`, bounded startup/heartbeat
and per-call deadlines, and conservative booking of the full reservation on
every failure path.

**Timing (corrected, round 8).** T-dial-bench is **complete**: **≈1.0 GPU-h**
measured (3,560 s of summed record wall time; 13 records) on the padded engine
before the tag. **The pre-tag diagnostic smoke and its one bounded repair cycle
already ran before the tag**, so the earlier blanket claim that these events
all follow the tag is retired: there were **four attempts** — a launcher/path argparse crash, a
GPU-only `_model_device` `NameError`, an OOM in `recompute_engine_logits`, and a
**completed SMOKE-FAIL** whose artifact is
`engine_gate/re_prefill_smoke.json`, sha256
**`fa117eff3be7a7ed46a5574073e7b17effef60abe0915f067e2b647ccaca046c`**
(generated 2026-11-05T17:50:46.854164+00:00, `counted=false`). The current
refill candidate is **NOT eligible**. The 11:00–11:07 stream diagnostic
(`engine_gate/stream_diag.json`) shows no sampling-stream-bug signature (O
totals 12/16/14 and L 21/16/24 for padded/refill/control; the anomalous
`lcb-04` shows refill 3/2 vs 0) and is recorded as supporting evidence, while
the sampled exactness deficit remains a real unresolved failure. The one
bounded repair then developed **gate v6** on `fix/gate-v6` (commits `9caad55`,
`a46a691`, `e4d5eab`, `78d93fa`, `8f36aca`; head `8f36aca`); native review
**rejected** it (C1–C3: greedy device mismatch, missing greedy Gumbel scores,
the boundary certificate cannot pass a real top-p flip, plus trace/partition
defects). The **repaired smoke was never run**, v6 remained unmerged, and no
repair cycle continues. **Round 8c closes the refill track for `prereg-v1`:**
the counted full gate v6, the counted re-prefill smoke and Part B are **not
run**, `engine_gate`/`part_b` are **INACTIVE** history with no execution
authority, and the **padded adaptive-batch HF engine is the sole operative
engine for all evaluation and Stage 2**. The batch-1 latency benchmark
(≤ 0.5 GPU-h) may move post-tag as descriptive only. The bench-addendum scope is compute and memory numbers plus
one conditional recipe contingency, disclosed as **deviation D-7b** (§9).

### Bench addendum (pre-tag, round 7b)

Source artifact: `C:\Users\<user>\ML\gemmaloop-artifacts\perf\dial_bench.json`,
**sha256 `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1`**
(6,188 bytes, 13 records). Configuration: E2B dial objective at 8192 tokens,
last-k=3 window, gradient accumulation 16, gradient checkpointing, one warm-up
plus one timed optimizer step per record, synthetic top-k teacher. The device
cap is 11.914 GiB total VRAM and a 10.28 GiB allocator budget. The record's
`fits` flag is the measured outcome; no endpoint data is involved.

| block | T | tok/s | peak reserved GiB | peak allocated GiB | fits |
|---|---:|---:|---:|---:|---|
| 25-29 | 4 | 601.9 | 8.475 | 8.169 | yes |
| 25-29 | 8 | 569.2 | 8.750 | 8.171 | yes |
| 16-18 | 4 | 491.8 | 8.795 | 8.351 | yes |
| 16-18 | 8 | 482.7 | 8.795 | 8.351 | yes |
| 15-29 | 4 | 430.5 | 9.273 | 8.948 | yes |
| 15-29 | 8 | 375.7 | 9.523 | 8.948 | yes |
| O-H heal (T=1) | 1 | 1,085.2 | 7.732 | 7.541 | yes (150 params with gradients) |

The **16-18** block is slower than 25-29 because the 16-layer coda runs once per
supervised exit. The **Stage 2 proxy** (not the T-dial-rl trainer) **OOMs at
both N=6 and N=8 for every block**, with peak reserved 9.912-10.258 GiB against
the 10.28 GiB cap, even though the dial step at T=8 fits; the proxy's
policy-gradient path is less memory-efficient than the dial windowed forward.

**Implied Stage 1 hours at 12M tokens** (one timed optimizer step, so an
estimate, not a full-run measurement): L **25-29 ≈5.5 h, 16-18 ≈6.9 h, 15-29
≈7.7-8.9 h**, O-H **≈3.1 h**, and **≈0.8-2.2 h per 3M-token pilot heal**.
With O-H charged to both arms the Stage 1 training row becomes **8.6 h
(25-29), 10.0 h (16-18), 10.8-12.0 h (15-29)**, replacing the round-7 12-16 h
estimate. Round 7c adds the **4-16 h teacher precompute** row (assumptions in
§10) and recomputes the totals from both substitutions.

**Stage 2 contingency (deviation D-7b).** Because the proxy is not the
trainer, the measured proxy OOM does **not** by itself trigger a fallback.
**T-dial-rl ships and runs its own memory bench before any Stage 2 run**; if
that bench shows that an N=6 **or** an N=8 step does not fit the allocator cap
for the chosen block, **both become N=4**, giving the mix **3/3/10** over
N=2/3/4. This changes the registered recipe field `rollout_n_groups`; it is a
**disclosed deviation (D-7b, §9)** and changes no selection or analysis rule,
no endpoint, and `analysis_rules_frozen_at_commit` stays `8c6d78d`. The
trigger is measured memory, Stage A content is not an input, and the change was
made after Stage A was shown to the author.

**Round 8c closes the engine branch: the padded adaptive-batch HF engine is
the sole operative engine for all evaluation and Stage 2**, and the same engine
is used for O, O-H and L wherever they are compared. No refill run is
registered and `engine_gate`/`part_b` are INACTIVE. For the historical record
only, the branch would have used the **`fix/gate-v6`** repaired-smoke candidate
if, after the tag, it had passed the 5c fixes, the registered
**gate v6** — greedy chosen-token regret and sampled decision-trace
reconstruction against the frozen padded-only calibration
`[0,2,4,6]/[1,3,5,7]` and validation `[0,4,1,5]/[2,6,3,7]` partitions,
`D ≤ τ_g` with `τ_g = max(0.25, max calibration D + 0.0625)` and a 0.50
inconclusive ceiling, stable/changed-support certificates, matched pooled
denominators, a 3/8 floor and ≥ 32 shared steps, N ∈ {1,2,3,4,6,8}, sampled
parity at N=4 and every Stage 2 N, long prompts and mixed-history waves — a
counted re-prefill smoke, and Part B ≥ 1.3× on the 360 unfiltered DAPO
candidates. That counted work is **not run** for this preregistration. Never
pin or bisect to `efd0145`; it is broken alone (it deleted about 640 lines
that `fc1ffe9` restored).

**Assumptions (stated).** Stage 2 L hours apply the code `E[r]` for the primary
3/3/8/1/1 mix (1.4007 / 1.2358 / 2.2022 for 25-29 / 16-18 / 15-29, i.e.
1.226× / 1.140× / 1.543× the N=2-only mix) to the round-4 L base (83-124 h
padded, 36-74 h refill); low per-N batch efficiency at N=6/8 makes FLOPs
scaling a lower bound. If the **T-dial-rl memory bench** triggers **deviation
D-7b**, the mix becomes 3/3/10 and the Stage 2 rows are recomputed from the
same registered `E[r]` code before the run; no endpoint or analysis field
changes. The new **teacher top-k precompute** row assumes a forward-only nf4
pass over 18M tokens (12M Stage 1 rows + 2×3M pilot heals), anchored on the
measured E2B dial step (601.9 tok/s at T=4, four forwards plus backward and
optimizer), with a single forward at ~1/4 the per-token cost and parameter
count scaling (E4B ≈2×, 12B ≈6× E2B), plus ~25 % shard I/O: **4-16 h** for
the chosen teacher, to be measured by the one-row teacher smoke. Evaluation
adds the L-heal N=3/N=4 points (+8-11 h padded / +1.5-2 h refill for 25-29)
and, explicitly, the **+15-25 h padded / +3-5 h refill N=4 secondary-suite
uplift for the 25-29 case** (no longer assumed inside the rounded range).
Per-block totals use the block's r_4 through the D2 increment (design §7.1);
the non-D2 base is unchanged; the 15-29 total is recomputed to its
~650-1,000 base plus the precompute row. The Stage 1 row is an estimate from
one timed optimizer step (not a full-run measurement); the other rows remain
estimates. The totals are the sum of the table rows and are estimates, not
full-run measurements.

**External API rules (round 8b).** DeepSeek is used **only through the
official API**, model **`deepseek-flash`** (served
`DeepSeek-V4.1-Flash-1017`), and **off-peak only**: peak is
`01:00-04:00` and `06:00-10:00` UTC Monday-Friday, with a **15-minute margin
before** and a **5-minute margin after** each window, enforced in code with no
override. Every phase starts with a **paid pilot of 20 items** and needs the
author's explicit approval; a continuation is refused when the projected cost
exceeds that phase's cap. The writer's code-level `PILOT_MAX_ITEMS = 50` is a
hard upper bound, not a registered paid-run size, and never authorizes more
than the 20-item pilot. A run stops when **≥ 3 of the last 20** paid
responses are waste. There is **one registered `system_fingerprint`** for the
project; every response must match the exact served model and that fingerprint.
Budgets: **phase 1 hidden tests cap $5**; the whole project is expected at
**≈$40-50**, and the phase caps sum to **$71**. The **NVIDIA NIM trial is not
usable** for anything that enters training, judgments, hidden tests or the
published dossier (terms s1.2/1.4, 4.2, 4.4, 4.12); only discarded plumbing
tests are allowed. **OpenCode endpoints are never used for training data or
judgments.**

| Stage | Padded engine (local h) | Refill engine (local h) | Cloud / note |
|---|---:|---:|---|
| Stage 0 sampling | 0 (Modal) | 0 (Modal) | Modal, stock vLLM v0.30.0, ~$7-11; $30 usage cap / $0 spend limit; fallback = registered engine; engine recorded per row; Modal optional for Part 1 |
| Stage-0-lite local sampling (pilot data) | ~17-20 | ~17-20 (not used; refill closed) | Registered Part 1 path; measured first shard 100 requests in ≈2,600 s, ≈195 tok/s, 52/100 kept; not added again to the totals below |
| Pilot (P1 + P2 + P3) | 43-52 | 43-52 | Round-7 estimate; dial objective, two anytime-tax arms, the 11-arm matrix (ninth arm + two O-H arms, +1.8 h), multi-N sweeps, P1 to N=8; engine affects P3 only; T-dial-bench is in the pre-tag row |
| Stage 1 heal (L + O-H) | ~8.6-12.0 | ~8.6-12.0 | Measured implied: L 5.5 / 6.9 / 7.7-8.9 h plus O-H 3.1 h, so 8.6 / 10.0 / 10.8-12.0 h by block; 12M tokens from the T-dial-bench throughput; engine-independent; an estimate from one timed optimizer step |
| Teacher top-k precompute (one teacher) | 4-16 | 4-16 | Forward-only nf4 pass over 18M tokens (12M Stage 1 rows + 2×3M pilot heals). Assumptions: anchored on the measured E2B dial step (601.9 tok/s at T=4, four forwards plus backward and optimizer); a single forward is ~1/4 the per-token cost and parameter count scales it (E4B ≈2×, 12B ≈6× E2B), giving ≈1,200-1,800 tok/s (E4B, ≈3-4 h) or ≈400-600 tok/s (12B, ≈8-13 h), plus ≈25 % shard I/O: ≈4-6 h (E4B) or ≈10-16 h (12B). The row budgets the chosen teacher at 4-16 h; the one-row teacher smoke measures the rate |
| Stage 2 RLVR L, 25-29 | 102-153 | 45-91 | E[r] 1.4007 for the primary 3/3/8/1/1 mix, 1.226× the N=2 mix; N=6/8 batch efficiency is a lower bound; the 3/3/10 fallback mix recomputes E[r] before the run |
| Stage 2 RLVR L, 16-18 | 95-141 | 41-84 | E[r] 1.2358, 1.140× |
| Stage 2 RLVR L, 15-29 | 128-191 | 56-114 | E[r] 2.2022, 1.543× |
| Stage 2 RLVR O-H | 64-96 | 27-58 | Unchanged control |
| Evaluation (seed 0, 25-29 case) | 332-560 | 61-102 | D2 L(N=4) + N=2 baseline, 60-75M tokens at 44-61 tok/s, 32k cache; includes L-heal N=3/N=4 (+8-11 padded / +1.5-2 refill) and the N=4 secondary uplift (+15-25 padded / +3-5 refill) for 25-29 |
| Loop ladder (descriptive) | ≈37 without reuse | ≈37 without reuse | After the tag; L-final-N4 LCB part = E2/D2; L-final-N2 182-item part is a view of the full N=2 run; L-heal N=3/N=4 charged in evaluation |
| **Pre-tag benchmarks** | **≤ 1.5** | **≤ 1.5** | T-dial-bench measured **≈1.0** (13 records; artifact and sha256 in §11) + one-row teacher smoke, gate step-time smoke and batch-1 latency (≤ 0.5) on the padded engine |
| **Engine registration (after tag)** | **9.7-16.3** | **9.7-16.3** | Historical estimate only: round 8b closes the refill track, so this row is not run for `prereg-v1`; the padded engine is registered |
| **Total seed 0, 25-29** | **602-941** | **237-383** | Sum of the rows after the measured Stage 1 substitution and the 4-16 h teacher precompute; 16-18 596-930 / 234-377; 15-29 654-1,016 / 259-416 (its ~650-1,000 base plus the precompute) |

N=1 bit-exactness is checked against stock HF; O, O-H and L share the engine
within a comparison.

## 11. Pins and freezes

Every pin computable on CPU is filled now. The only remaining null is the data
manifest SHA, which is filled in the post-tag **`prereg-v1-data`** addendum,
committed before the pilot. No trained-arm result is produced before that
addendum is committed.

- E2B snapshot: `4adf15748de9e1a0097b6274cee9c15836c41c37`.
- E2B chat-template SHA-256:
  `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`.
- Teacher revisions:
  - E4B: `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`.
  - 12B QAT q4_0 unquantized: `b6ed86275a6a5735884e208bfed95b445a684ca2`.
- **E4B header and ρ (closed, round 8).** Computed from the pinned E4B
  `4ecc2ba` safetensors header with
  `huggingface_hub.get_safetensors_metadata` (no weight download): **14.003
  GiB text-resident** = 8.753 GiB GPU-side text + 5.250 GiB CPU PLE,
  7,518,069,034 text params; the whole checkpoint is 14.894 GiB. With
  `R_E2B = 8.657 GiB` from the pinned E2B `4adf157` header, **ρ = 1.62** is
  frozen (R_GPU 2.04, R_ckpt 1.56 sensitivities).
- **Teacher runtimes (round 8; X3).** E4B bf16 with CPU PLE primary and int8
  fallback; 12B NF4 primary from the QAT-unquantized checkpoint and the
  official QAT Q4_0 GGUF through llama.cpp as the registered fallback. The
  ordered qualification rule, acceptance table, single 20 %-margin expansion
  to 100 rows, manifest identity and one-frozen-runtime-per-teacher rule are
  in §4.2 and `prereg.yaml` `stage1`. The qualified runtime and its manifest
  identity are recorded in the **`prereg-v1-teacher-runtime`** addendum before
  the pilot and before any trained-arm evaluation. X3-C1 is a registered
  execution repair (inference_mode over the whole precompute including the
  head, frozen teacher parameters, detached outputs, no grad_fn test).
  **Round 8b corpus amendment:** qualification rows come from O's Stage 0
  local samples (train bucket, verified, 4/8/4/4, 20-row screen then expansion
  to 100 in seeded hash order), disjoint only from pilot evaluation, T-fit,
  MONITOR and confirmatory; the 120-row P1-lite corpus is the fallback and
  keeps 85 rows after protected-set filtering, extended on GPU by hash-ordered
  sampling to at least **140** verified rows before the 20-row screen is taken.
  Qualification runs after the tag and before the pilot, fail-closed.
- **Merged commits (full 40-character ids, round 8c).**
  `217b9390081d9c838193aa06665c67a3b1a6809d` teacher runtime;
  `2eef68b361f0ce7160ff30eff580ee921a5f61cf` DeepSeek guards;
  `106e0e9fb2ff8f63a356f6f3be657967ad653183` final DeepSeek guard merge
  (branch head `37583c197e4ac76d20031b0969935e92c49a165c`);
  `dfea02a617248157e8dba85e4c8d954b20272900` sources;
  `c840e9b44deccccdf99cd13934d1380017e8efe4` Modal image order;
  `8b4c020b8da5112e7b6e8b574dc5118c1d6510ba` p1corpus merge (branch head
  `1e789963f1e313a90686060a223276eef0a0b4ec`);
  `1830a361522ec2ef67d5b307e4e4b49b97d83aad` modal-imports merge (branch
  head `c9d0cdee3a365039ac796df64cb063358e48bf58`);
  `e3f63f352a32ccb3bccb013a781536ee6a537511` task/healpilot merge (heal
  monitor, pilot harness, abort-resume and the single prereg schema with the
  tag gate); `6d3bb65c7a007bc67adf1ddf1a9536272b2163c5` fix/modal-imports
  rounds 6-7 merge (CUDA 12.8 devel image with build-essential and ninja plus
  the CPU JIT-toolchain preflight).
- **Round-8d pins (all filled; Modal lite pin corrected by round 8e).**
  `task/p1corpus` is `8b4c020b8da5112e7b6e8b574dc5118c1d6510ba`;
  `fix/modal-imports` is `1830a361522ec2ef67d5b307e4e4b49b97d83aad`;
  `task_healpilot_commit` is `e3f63f352a32ccb3bccb013a781536ee6a537511`;
  `modal_image_commit` is `6d3bb65c7a007bc67adf1ddf1a9536272b2163c5`; and
  `task_modallite_commit` is
  `4c260ebe1810a0f95cb47d774bd8eed894e9cfda`, the seed-fixed commit the
  successful Modal lite run actually used. The pre-fix
  `9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab` is historical (seed overflow,
  $0.758 booked). No placeholder remains.
  **A tag request fails while any registered placeholder remains**
  (`gemmaloop.prereg.assert_tag_ready`, which scans the whole document, not
  only `pins`); after round 8e it passes.
- **Pilot shortfall freeze (round 8e).**
  `prereg/pilot-shortfall-freeze.json` is **REQUIRED before any pilot scoring
  or P2 training**. It is filled by the data addendum after the official
  count; the immutable rule and verifier/test-set specification may be sealed
  before it, as the panel's staged freeze permits. While the index or the data
  addendum is pending, no pilot scoring, P2 training, ranking or
  student-based data revision is allowed.
- **Operative padded engine pin.** The padded adaptive-batch HF engine at the
  final commit: `serve/generate.py` (`generate_batch`) blob
  `282c66b5cd07db1060303927587139fec2ff2d3c`, `serve/cache.py` blob
  `c8c192c7e6e10851947da0ca1e957d1593ce365c`, `eval/runner.py` blob
  `11c427e329e22076ec6feaa8f18fcebd2438358e`, verified at `54cab07`; round 8c
  changes no code.
- **Temperature-fit pin.** `fit_model_temperature` /
  `calibrated_nll_report` at commit `54cab07`, blob
  `f51de454546b89018cf4012921f06f6331403c57`; bounds (0.5, 4.0),
  `max_iter = 40`, `tol = 5e-3`; the same estimator is used in pilot and
  confirmatory scoring.
- Tokenizer SHA-256:
  - E2B: `ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2`.
  - E4B: `91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc`.
  - 12B QAT q4_0 unquantized:
    `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
- `lcb_runner` commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- `math-verify`: **0.9.0 (antlr4 4.13.2)**, a dependency as of round 8b; it
  runs with internal timeouts disabled under a 10 s outer subprocess timeout,
  OR the AIME/integer normaliser (section 5). The grader records `decided_by`.
- Data manifest SHA-256: **null**, filled in `prereg-v1-data`.
- **T-dial-bench (measured, round 7b).** Complete at **≈1.0 GPU-h** under
  gpu.lock before the tag: 13 records, artifact
  `C:\Users\<user>\ML\gemmaloop-artifacts\perf\dial_bench.json`, **sha256
  `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1`**. The
  per-block and per-T tok/s and peak GiB, the O-H heal step and the Stage 2
  proxy OOM are in the **bench addendum (§10)**. The same pre-tag row still
  carries the **one-row E4B/12B teacher smoke**, the **gate step-time smoke**
  and the batch-1 latency benchmark (≤ 0.5 GPU-h; no refill re-run exists). All
  are recorded in the **pre-tag pins addendum**, whose scope
  is compute and memory numbers plus one conditional recipe contingency,
  disclosed as **deviation D-7b**; it changes no endpoint, selection or
  analysis field. If **T-dial-rl's own memory bench** (shipped before any
  Stage 2 run; configuration fixed in §9) shows that an N=6 **or** an N=8 step
  does not fit the allocator cap for the chosen block, **both become N=4,
  giving the mix 3/3/10**, if that block is chosen. The proxy OOM does not by
  itself trigger it; the contingency is settled before the run, never after,
  and `analysis_rules_frozen_at_commit` stays `8c6d78d`.
- **Pre-tag smoke record (corrected, round 8c).** The four attempts are
  registered in §10 and `engine_gate.pre_tag_smoke_attempts`; the completed
  artifact is `engine_gate/re_prefill_smoke.json`, sha256
  `fa117eff3be7a7ed46a5574073e7b17effef60abe0915f067e2b647ccaca046c`. The
  refill candidate is **not eligible**. The one bounded repair developed gate
  v6 on `fix/gate-v6` (head `8f36aca`), which review rejected (C1–C3); the
  **repaired smoke never ran** and v6 remained unmerged. The 11:00–11:07
  stream diagnostic is supporting evidence only, while the sampled exactness
  deficit remains a real failure.
- **Engine addendum — INACTIVE.** No counted engine-gate v6, re-prefill-smoke
  or Part B results exist and none will be produced for `prereg-v1`;
  `engine_gate`/`part_b` have no execution authority. The padded engine is
  registered and pinned above.
- **Pinned engine commits — none for refill.** There is no repaired-smoke
  commit: v6 was developed and rejected without a run, and `fix/gate-v6`
  remained unmerged. `eed7fa7` is historical gate-v5 provenance; merged feat
  commit = **`a7c06331a4a381c20bf8af5a01f0f39fc57e71b6`**; `34dd46e` is the
  earlier fix-round-4 commit merged at `2db1868`. **Never pin or bisect to
  `efd0145`**.
- **Freeze point and sequence.** The tag precedes any trained-arm evaluation;
  the exact sequence is rules (tag; analysis rules already at `8c6d78d`) →
  data (`prereg-v1-data`, after the tag and before the pilot) → pilot
  selection (selected configuration before main training) →
  implementation/`analysis-v1` → main runs. The final `prereg-v1` commit is
  the authoritative complete specification.

### Analysis-v1 obligations (before any trained-arm evaluation)

- Delete the dead S1-S5 code paths.
- Re-cover the helper unit tests lost with the legacy removal (stats wiring,
  MDE, bootstrap/Holm helpers).
- The duplicate result key includes block, bridge and policy from T-arch.
- Implement E1-E4 and the eval runner lcb 1,055 mapping; tag `analysis-v1`
  before any trained-arm evaluation.
- Implement the D1 intersection-union test and the graphical α procedure
  exactly as registered: weights `0.025 × (E2 0.8, E3 0, E4 0, D1 0.2)`,
  transitions `E2→E3=E3→E4=E4→D1=D1→E2=1`, Bretz et al. 2009 Algorithm 1,
  one-sided percentile-bootstrap p-values (10,000, seed 0), D1 p = max of its
  four components; `E_{k+1}` receives α only after `E_k` is rejected.
- Implement D2–D5 (D2 L(N=4)−L(N=2) on the full LCB with a paired CI; D3 FLOPs
  frontier on the code r_N; D4 adaptive policies with problem-cluster CIs,
  ε for N_eff = 2, 3 and 4, ΔcNLL = cNLL(fixed) − cNLL(policy), per-(arm,
  checkpoint, exit-N) temperature, and the pre-fixed winner rule with "no
  adaptive winner" → fixed N=4; D5 extrapolation and collapse flags).
- Implement the E4 grid on O-H's budget (O-H at B = 1024·2^j, L at
  floor(B/r_4), both ≤ 32,768, both re-scored from saved completions).
- Implement the multi-exit NLL sweep N=1..16 with per-token NLL, ‖Δh_t‖ and
  exit-1 entropy, and exactness tests: exit t equals the fixed-N=t forward
  (fp32 and bf16), exit 1 is O bit-exactly, and per-token independence.
- Implement the resident-bytes script (`Σ numel × dtype_bytes` over the loaded
  text path plus the CPU-side PLE table, written to `run.json`) and the D6
  byte-frontier analysis; test `B(L,N=k)=B(L,N=1)` for k=2..8 and
  `B(L,N=1)=B(O)+B(LoRA)+B(g)`.
- Register the D4 winner as the Part 2 `auto` serve level before any
  adaptive-decode claim.
- Before the registered engine run: remove the duplicate keys in `resolve_run`,
  keep `min_exact_fraction` at the legacy `0.0` (the v4 per-key/pooled rule
  governs), and make the re-prefill smoke report compute
  `amendment_permitted`.

---

## 12. Limitations

- **Single confirmatory seed.** Seed 0 only; seeds 1 and 2 are robustness-only.
  Item-level intervals do not include training-seed variance.
- **Small budgets.** Stage 1 is fixed at 12M tokens and Stage 2 is 100 RL steps.
  Published loop retrofits used 0.48B-120B tokens; this study is 1-3 orders of
  magnitude smaller and may return a null. The pilot gate and the mechanism null
  make that null interpretable.
- **Layer-scalar geometry.** The residual stream is leaky and 15-29 has an
  orthogonal entry; 25-29 and 16-18 are high-cosine periods. E4B scalars are not
  read in Part 1.
- **Teacher numerics (round 8).** E4B is evaluated in bf16 with CPU PLE (int8
  fallback), not four-bit. The 12B candidate is Google's QAT-unquantized
  checkpoint loaded with bitsandbytes NF4 — the registered evaluated
  representation, which is not automatically the vendor's Q4_0 execution or its
  reported score; the official QAT Q4_0 GGUF through llama.cpp is the
  registered fallback. The top-k (k = 64) float16 logits plus tail bucket
  approximate the full distribution, and the capacity gap may hurt. The glitch
  token is masked.
- **Per-token deliberation over fixed memory.** Later passes refine the same
  token over a fixed prelude memory; deeper states do not propagate to later
  tokens. The dial is per-token deliberation, a complement to thinking length.
- **Anytime tax at N=2 and N=4.** Sharing capacity over exits can tax the
  deployed point; it is measured by two descriptive pilot arms (single-exit
  legacy at N 2/3 and at N=4) and reported, not assumed away.
- **Training FLOPs are not matched.** Stage 1 is ≈2.4× O-H per token
  (multi-exit). Stage 2 against O-H is **1.24–2.20× per token** for the adopted
  3/3/8/1/1 mix (E[r] 1.4007/1.2358/2.2022 versus O-H's 1.0, before the
  O-H/L sequence-length difference); training compute is declared, not
  equalised. The round-5 "≈1.09–1.22×" was the designer's N-mix increase over
  N=2, not L versus O-H, and is superseded.
- **Trained depth.** The dial is trained to N=8; N=12 and 16 are extrapolated
  and registered as estimation, not as claims.
- **Adaptive control is not confirmatory in Part 1.** D4 is exact but
  teacher-forced; adaptive-decode accuracy is a named Part 2 milestone. D4's
  winning policy is **selected on the same scored data**, so its selected
  ΔcNLL and interval are **exploratory/selected/optimistic**, not an unbiased
  selected-policy interval; the Part 2 `auto` entry is provisional and needs
  fresh-data validation before any serving claim.
- **Pilot interpretability limits (round 8c; X1 §4).**
  - **Deployable representations, not size.** The teacher comparison compares
    the qualified E4B bf16/int8 and 12B NF4/Q4_0 representations. They differ
    in approximation error, architecture, training and calibration;
    qualification controls, but does not eliminate, representation error.
    Each teacher's fidelity results are reported beside the student gains, and
    a winner is not attributed solely to 12B's size or E4B's easier targets.
  - **"Why" is descriptive, not causal.** Entropy, gap closed, span/position
    gains and per-N curves do not identify which factor caused a teacher to
    win. A KL measured on a fixed 64-row training subsample is labelled
    **in-sample**, with identical rows/positions and each teacher's own
    initial gap; gap-closed fractions have different denominators, so absolute
    initial/final KL is reported and near-zero denominators are guarded. N=1
    is structurally O for L, so its zero learning change is not failed healing.
  - **O-solvable verified math and code domain.** The fit/selection pilot ranks
    teachers for reproducing verified math traces plus code passing the frozen
    Docker verifier; chat never counts. It does not establish the best teacher
    for chat, hard unsolved math, arbitrary code, generated-answer accuracy or
    every deployment task. Code verification establishes passing the frozen
    tests, not formal correctness. Confirmatory D1 remains math-only.
  - **Architecture × teacher interactions are untested outside 25–29.** The
    E4B-only block search followed by a 12B-trained final recipe is frozen and
    legitimate, but the final teacher × block combination may never have been
    piloted; this is reported, and missing cells are never added selectively
    after seeing results.
  - **Lineage and dose.** Main Stage 1 restarts from O for both L and O-H; the
    3M pilot heals are selection-only and never continued. Rendered input
    tokens, scored assistant tokens and repeated multi-exit targets are
    separate cost denominators; no 3M+12M exposure is described as 12M.
  - **Shared seeds do not establish backend identity.** Local HF and full
    Modal/vLLM parity is supported only by the same actual frozen artifacts
    shared by every arm and recorded backend/sampling provenance; a capped run
    preserves an accepted prefix only on the same backend/random stream.
  - **A pilot STOP is a recipe/budget gate.** It means this recipe/budget did
    not meet its continuation criterion, not a universal scientific
    mechanism-null proof.
- **Per-byte precision and bytes read.** Part 1 makes no cross-precision
  per-byte claim; every byte count is at bf16. The 4-bit frontier is Part 2.
  At N=4, 15-29 reads **2.28×** its N=1 bytes, more than E4B's 2.05× (bytes
  parity is at N ≈ 3.4); 25-29 and 16-18 read less. L cannot be merged without
  breaking pass 1 ≡ O, so a merged L costs `R + W_block` and is a sensitivity.
  **Merged bytes.** L unmerged is the byte-minimal deployable form; "L merged"
  is a sensitivity, and O-H merged is the primary byte form.
- **Fixed KV.** No per-pass KV and no cross-token propagation of newly computed
  features.
- **O-H runs a real Stage 1.** O-H is not a no-op; the Stage 1 cost is charged to
  both arms. The E2 comparison is the complete looped package versus a
  matched-data control with the **same rank-16 LoRA on the same block layers**
  (matched LoRA capacity); O-H has no recurrent bridge and no multi-exit loop
  package, and L's bridge/package effects are itemized separately. E2 is not
  the pure causal effect of recurrence with every training and compute factor
  held fixed.
- **E3 is a total effect.** L(N=4) − O includes the heal and RL changes, not only
  the loop; E2 is the loop contrast.
- **Benchmark power.** The full LCB MDE is 5.57 pp; the post-cut subgroup and
  AIME are not powered for arm comparisons. The Google comparison is a
  reproduction check, not validation.
- **RL saturation and length confounds.** E2B-it is already RL-post-trained; RL
  may saturate. A cap can reward brevity and L costs more FLOPs per token;
  budget curves, truncation logs and length reports are the mitigations.
- **Engines and compute.** The looped vLLM plugin does not exist yet and local
  HF generation is slow; the round-7c local seed-0 envelope is 25-29 padded
  602-941 h / refill 237-383 h, 16-18 596-930 / 234-377 and 15-29 654-1,016 /
  259-416 (including the ≈37 h loop ladder, the 4-16 h teacher precompute and
  the pre/post-tag engine work,
  with the gate 4.5-6.0 h incl. model loads, the smoke 10-20 min and Part B
  5-10 h). T-dial-bench is measured; the latency benchmark, the engine gate,
  the re-prefill smoke, Part B and the T-dial-rl memory bench remain estimates
  or checks. The tier is a covariate, not a randomisation.
- **Part 2 judge circularity.** DeepSeek may judge data it helped create;
  automatic checks and the author's blind hand ratings are the guards. Part 2 is
  development, not a confirmatory claim.

---

## 13. Choices made by the drafter (revision rounds 7–8c)

These are the concrete values chosen where the sources were underspecified.
They are also in `prereg.yaml` under `drafter_choices`.

1. Horizon `loop_probs = [0, 0, 0, 0.70, 0.075, 0.075, 0.075, 0.075]` (T=4 at
   70 %, T in {5,6,7,8} at 7.5 % each); N=1 batches are not trained because
   adapters are N ≥ 2-only. N=2..8 are trained, dense on 2–4.
2. Stage 1 `beta = 0.1` with `KL(parent || student)` at **every** exit; parent
   logits are computed live, full-vocab, with adapters off at N=1.
3. Stage 1 top-k = 64 in float16 plus one tail bucket (teacher mass outside the
   top-k versus the student's) so the KL is a proper divergence.
4. Stage 1 budget is fixed at 12M tokens; the loss is logged every 2M tokens
   descriptively, with no dose-response decision.
5. LoRA rank 16, alpha 32, LoRA LR 2e-4; bridge LR 3e-3; loop_ple LR 1e-3;
   legacy scalar `w0 = 0.99`. **Bitsandbytes 8-bit AdamW (`adamw_8bit`) on GPU
   for every arm and stage that trains** (the yaml field keeps the legacy
   `AdamW` spelling; `run.json` records the resolved optimizer), **per-device
   batch 1 row × gradient
   accumulation 16 rows per optimizer step** (`rows_per_optimizer_step = 16`;
   the measured tokens per step are recorded in `run.json`), gradient clip
   1.0, token-cosine with 3 % warmup; LoRA targets the loop block only; one
   LoRA and one g shared by every pass ≥ 2. Under `--prereg` every one of
   these values is read from `prereg.yaml` and a CLI override exits 2 (round
   7c recipe binding).
6. Stage 2 uses G=8, 16 prompts/step, 100 steps, a **2,048-token prompt cap**
   (prompts above it are excluded from the pool at selection time, identically
   for both arms) and an 8192 completion cap (row ≤ 10,240 tokens), the same
   **bitsandbytes 8-bit AdamW (`adamw_8bit`)** on GPU, LoRA LR 2e-5,
   bridge LR 1e-3, `beta_KL = 0.0`, TIS cap 2.0, `num_iterations = 1`, Dr. GRPO
   normaliser 8192, per-test timeout 6 s; the N mix is the structured
   `rollout_n_groups {2:3, 3:3, 4:8, 6:1, 8:1}` over N=2/3/4/6/8 (validated to
   sum to 16).
7. E2/E3/E4 primary uses **N=4** on the full `release_v6` (1,055 items x 1
   sample); the post-cut subgroup is reported separately and L(N=2) runs the
   same 1,055 items as the D2 paired baseline.
8. E1 is a precondition with no alpha: bit-exact L(N=1) on every engine plus
   descriptive heal estimates. The fixed sequence is E2 → E3 → E4.
9. The confirmatory verdict is computed once, on seed 0, with a problem-cluster
   bootstrap; seeds [1, 2] are robustness only and the crossed bootstrap is a
   robustness check.
10. E4 uses the code `r_4 = 2.2824/1.4275/1.2515` for 15-29/25-29/16-18 on
    O-H's grid: O-H at B = 1024·2^j (j=0..5), L at floor(B/r_4), both from
    saved completions and both ≤ 32,768, trapezoid AUC over j / 5.
11. Pilot P2 uses the **11-arm** matrix above (nine selection/control arms plus
    two descriptive single-exit anytime-tax arms, N 2/3 and N=4), scores on O's
    verified-correct traces, and applies one deterministic two-step rule:
    teacher first by mean gain over O of (25-29+gch+T) and (O-H+T), then
    block/bridge by **rule B**. E = the frozen `84c5da0` eligibility is the
    STOP/eligible set; arms in E are ranked by
    `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]` (same teacher;
    matched rank-16 LoRA capacity per block, round 8c; L's bridge/package
    itemized separately); the adjacent-regression check is a filter (filtered
    argmax if any arm passes, else unfiltered argmax over E).
12. Pilot P3 uses 200 held-out math problems x 4 from the named P3 bucket, a 4k
    cap on prompts with O median <= 2.5k, paired SE >= 2, truncation < 10 %;
    P3 RL prompts are drawn from the Stage 0 train bucket 0-49; the RFT fallback
    recipe is registered above.
13. One split table assigns problems by `sha256(problem_id) % 100` into disjoint
    train / pilot-dev / NLL-KL / long-trace / held-out buckets; the pilot bucket
    50-59 is sub-allocated into P3 (50-53), selection (54-56), T-fit (57-58) and
    P1 (59) sub-ranges.
14. The NLL/KL bucket 60-69 is sub-allocated: 60-61 (100 problems) fits one
    temperature per (arm, checkpoint, N) and the D4 thresholds for every post-training NLL
    endpoint; 62-69 (400 problems) is the scored NLL test corpus.
15. Stage 0 pool: 2,000 code, 2,000 math and 1,000 chat prompts, 4 samples each,
    about 64M tokens. Chat source is the existing decontaminated chat source;
    "synthetic" code is dropped; if fewer than 1,600 prompts qualify, cycle the
    qualifying set in the same order.
16. Data decontamination is 13-gram plus Indel >= 0.75 against every registered
    eval and the creative briefs; the manifest SHA is frozen in the addendum.
17. Stage-0 samples are kept only if `finish_reason = stop`, the thought channel
    is closed, the answer is non-empty and length <= 8192; the reasoning share of
    code + math tokens is >= 75 %.
18. Compute tiers are local, single cloud GPU, multi-GPU and TPU, chosen by
    availability only and recorded as a covariate. Stage 0 is Modal stock vLLM
    v0.30.0 data-only ($30 usage cap / $0 spend limit), with a local HF-engine
    fallback and the engine recorded per row. Stage 2 rollouts and all
    evaluation use the padded adaptive-batch HF engine for O, O-H and L
    (round 8c: the refill track is closed and the earlier conditional wording
    is historical).
19. R and R-damped are diagnostic only, scored on the 182 post-cut items x 1;
    R-damped uses the Euler `damped_update` with weights 1/N (0.5/0.333/0.25 at
    N=2/3/4). The legacy scalar bridge `w=0.99` is a separate bridge.
20. Bench roles are explicit in the schema: confirmatory, secondary, o_only and
    harness; there is no default to confirmatory.
21. The O-H-RLext trigger is "E2 passes", seed 0 only, 1.5x Stage 2 steps on
    fresh prompts; it is not confirmatory.
22. Part 1 is E2B; seeds [1, 2] are registered as robustness-only, in that
    order; E4B is deferred.
23. The Google comparison is a "reproduction check" on O's full 1,055-item LCB
    run; the separate 182-item generation is dropped (`lcb_google` is a virtual
    view of the same run).
24. Creative evaluation is Part 2/exploratory; S3 is not a Part 1 confirmatory
    endpoint.
25. The graphical alpha procedure (Bretz et al. 2009) replaces the uniform 95 %
    sequence: E2 0.02 and D1 0.005 initially, D1→E2 recycled at 0.025,
    E2→E3→E4 fixed sequence, E4→D1 recycled at 0.025.
26. D1 uses an intersection-union test with gain `cNLL(1) − cNLL(4) > 0` and
    adjacent regressions `< m = 0.005` nats/token for N = 1, 2, 3; the margin is
    half the registered heal margin.
27. The BPTT window is k = 3 trainable passes; the monotone hinge weight is
    `lambda_m = 1.0` and covers the (t, t−1) pairs in the window plus exit 1.
28. The legacy `criteria` S1-S5, `train_tokens` and `phase1_plan` path are
    retired: the schema rejects them and the legacy analysis/planner refuse to
    run.
29. Intelligence per resident byte is registered as a reported metric (D6,
    estimation only) with accuracy and cNLL on separate axes, the denominator
    itemised and measured at bf16, φ and m anchored to Google's
    reproduction-check gaps, bytes read per token and batch-1 latency secondary;
    no cross-precision per-byte claim in Part 1.
30. D4 reports every adaptive rule with a problem-cluster CI; the winning policy
    is fixed by rule now (largest mean ΔcNLL at matched analytic FLOPs with CI
    lower bound > 0) and becomes the per-byte-table policy and the Part 2 `auto`
    serve level; adaptive-decode accuracy is a named Part 2 milestone.
31. Stage 1, Stage 2, evaluation, pre-tag and post-tag engine hours were
    round-6 estimates. Stage 2 L uses code `E[r]` for 3/3/8/1/1
    (1.4007/1.2358/2.2022), applied to the round-4 L base; evaluation adds the
    L-heal N=3/N=4 points (+8–11 padded / +1.5–2 refill for 25-29).
    **Superseded by round 7 (choice 45) and then by round 7b/7c (choice 51).
    The final round-7c totals are 25-29 602–941 / 237–383, 16-18 596–930 /
    234–377, 15-29 654–1,016 / 259–416, with the measured Stage 1 throughput
    and the 4–16 h teacher precompute; gate 4.5–6.0 h incl. loads, smoke
    10–20 min, Part B 5–10 h and engine row 9.7–16.3 h.**
32. Round 6 corrected rule B: STOP is the frozen `84c5da0` eligible set E; the
    ranking statistic is `Δ̄_arm = mean_{N=2..4}[cNLL(O-H[block]) − cNLL(L,N)]`; the
    adjacency check is a filter, not a preference. The old cNLL(1)-based dial
    gain is descriptive only.
33. Round 6 gave pilot O-H the same rank-16 LoRA on the same block layers on
    its single pass, so `delta_arm` compares **matched LoRA capacity** (round
    8c correction: not exact total parameters; L's bridge and multi-exit
    package are itemized separately).
34. Round 6 fully specified the graphical procedure (weights 0.025 × (E2 0.8,
    E3 0, E4 0, D1 0.2), transitions 1, Algorithm 1, one-sided percentile
    p-values, D1 max, α only after a rejection, decisions after termination).
35. Round 6 moved Stage 2 to the same last-k=3 backward window for every N and
    added the N=6/N=8 RL step to T-dial-bench; the N=8 → N=6 contingency is
    settled in the pre-tag addendum.
36. Round 6 added L-heal N=3/N=4 on the ladder items (primary-N DiD) and a
    second descriptive anytime-tax arm at N=4.
37. Round 6 rebuilt the engine section from feat `2db1868`: gate and Part B
    run after the tag and before any trained-arm evaluation; gate coverage is
    N ∈ {1,2,3,4,6,8}, sampled parity at N=4 and every Stage 2 N, throughput at
    N=1 and N=4, every block × bridge; Part B uses loops 4 on the 360 DAPO
    candidates at the evaluation batch with requests ≥ 3× slots.
38. Round 6 corrected D6 (merged bytes, `(Δ_tot, Δ_loop)` claim, φ_loop/φ_tot,
    prelude/block/coda itemisation, GiB, annotation-only Google points, header
    R_E4B, dropped q4_0 sensitivity).
39. Round 6 sealed Stage A (path, mtime, size, sha256); later files were sent
    to the post-tag data addendum. **Round 7 replaces that with the
    pre-visibility post-completion addendum (choice 44).** Stage B runs only
    after the tag.
40. Round 6 recomputed the MDE table with `gemmaloop.eval.stats.mde` (values
    unchanged; 5.75 pp if D1 fails) and the Stage 2 hours with code `E[r]`.
41. Round 7 pins the engine candidate `eed7fa7` (engine fix round 6 / gate v5,
    merged into feat at `a7c0633`); `34dd46e` is the earlier fix-round-4 commit
    merged at `2db1868`. Refill is eligible only if the registered gate, the
    re-prefill smoke step and Part B (≥ 1.3×) all pass after the tag and before
    any trained-arm evaluation; otherwise padded. Round 8c closes the refill
    track: the padded engine is the sole operative engine. **Never pin or
    bisect to `efd0145`.**
42. Round 7 adds `O-H[16-18] + E4B` and `O-H[15-29] + E4B` short heals and
    renames the existing controls `O-H[25-29]`; the matrix is 11 arms (nine
    selection/control plus two descriptive anytime-tax arms). Each L arm is
    compared with the O-H that has the same block layers (matched rank-16
    LoRA capacity per block; the bridge and multi-exit package are itemized
    separately, round 8c), and the teacher step measures L's gain at the
    primary N=4. Pilot quantities are selection-only.
43. Round 7 rewrites H1/H0: H1 is "E non-empty" (the frozen selection gate) and
    the selected arm's `Δ̄_arm` is reported with its 95 % CI as a descriptive
    loop-vs-heal estimate; it does not gate. **The confirmatory claims are
    E2–E4 and D1 only.**
44. Round 7 encodes Stage A visibility: the analysis rules are frozen at
    the round-7a commit (first freeze `f8ca082` superseded, §0.1) before
    completion; at completion
    the controller records sha256/mtime/size of every output in a
    post-completion addendum committed **before** the author sees the results;
    any later change is a disclosed deviation. `campaign.json` and
    `child-A_n*.json` are labelled mutable status files.
45. Round 7 fixes one consistent hours set: gate 4.5–6.0 h incl. model loads,
    smoke 10–20 min, Part B 5–10 h (central 5–9), engine row 9.7–16.3 h; pilot
    43–52 h (11 arms); evaluation charges the N=4 secondary uplift
    (+15–25 padded / +3–5 refill for 25-29); totals 25-29 601–932 / 236–374,
    16-18 594–920 / 232–367, 15-29 627–970 / 247–397; T-dial-bench ≤ 1.0 h and
    the pre-tag row ≤ 1.5 h. **Superseded by round 7b/7c (choice 51): the final
    totals are 25-29 602–941 / 237–383, 16-18 596–930 / 234–377, 15-29
    654–1,016 / 259–416, including the 4–16 h teacher precompute.**
46. Round 7 sets the bench-addendum scope (T-dial-bench at T=4/T=8 and the
    N=6/N=8 RL steps on all three blocks, the one-row E4B/12B teacher smoke and
    the gate step-time smoke) and makes the N=8 → N=6 contingency per candidate
    block ("if that block is chosen").
47. Round 7 fixes the D1 corpus: O's dedicated bucket 62-69 NLL sweep is
    separate from the Stage 0 training pool (0-49); math is verified by
    math-verify, code is excluded (no hidden-test harness registered for this
    corpus), chat is excluded.
48. Round 7 restricts the fixed-bytes effects to registered runs (LCB full at
    N=1,2,4; GPQA and MMLU-Pro at N=1 for O and N=4 for L; none at N=2),
    clarifies that ρ uses the pinned E2B header and the E4B architecture-notes
    value until the pre-tag header recomputation, and makes `pp_per_GiB`
    model-level with E4B annotation-only.
49. Round 7 requires a non-empty 64-hex `chosen_prompts_sha256` in registered
    Part B, records the hub-access/artifact rule (materialise the 128 prompts
    once with that sha and skip the EvalIndex later), registers the
    `--allow-stage-b` guard, and records the engine-smoke `amendment_permitted`
    and `resolve_run` duplicate-key obligations for the registered run.
50. Round 7b records the measured T-dial-bench (per block and T: tok/s and peak
    reserved GiB; O-H heal; Stage 2 proxy OOM at N=6/N=8) from
    `C:\Users\<user>\ML\gemmaloop-artifacts\perf\dial_bench.json`, sha256
    `ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1`, with
    the implied Stage 1 hours (L 5.5/6.9/7.7–8.9 plus O-H 3.1 h), and extends
    the Stage 2 contingency to N=6 **or** N=8 → N=4 (mix 3/3/10) if either does
    not fit with T-dial-rl, which ships its own memory bench before any Stage 2
    run. These add no endpoint data and no selection or analysis rule change;
    `analysis_rules_frozen_at_commit` stays `8c6d78d`. Round 7c reclassifies
    the contingency as **deviation D-7b** because it changes the registered
    recipe field `rollout_n_groups`.
51. Round 7c records **deviation D-7b**, the Stage 1 recipe binding and the
    text minors. D-7b: the Stage 2 N=6/N=8 → N=4 (mix 3/3/10) fallback is
    triggered by the T-dial-rl memory bench on measured memory, not by Stage A
    content, and was decided after Stage A was shown to the author (≈06:02, in
    chat; no reply at drafting). The T-dial-rl bench configuration is fixed
    now: the three candidate blocks, N=6 and N=8, 10,240 tokens per row (the
    2,048-token Stage 2 prompt cap plus the 8,192 completion cap), last-k=3
    window, prelude and pass 1 under `no_grad`,
    chunked head and the registered gradient accumulation (16); "fits" means
    one full optimizer step with no OOM and peak reserved memory below the
    allocator cap. Stage 1 is registered as per-device batch 1 row × gradient
    accumulation 16 rows per optimizer step (`rows_per_optimizer_step = 16`;
    measured tokens per step in `run.json`), LoRA LR 2e-4, bridge LR 3e-3,
    loop_ple LR 1e-3, clip 1.0 and 3 % warmup, all bound under `--prereg`
    with CLI overrides exiting 2. The teacher precompute (4–16 h) is budgeted
    from the measured E2B throughput, and the 15-29 totals are recomputed to
    ≈650-1,000 before that row. Pending item 4 is closed; the open pre-tag
    runs and the teacher download are added to the pending list.
52. Round 7d records **deviation D-7c**: (1) the `rows_per_optimizer_step`
    text registered since **round 1** (`19a1fd9`) said "8,192 tokens per
    optimizer step"; **round 7c** (`2f36330`, 08:28:48) corrected it to **16
    rows per optimizer step** (`rows_per_optimizer_step = 16`, ≈26k supervised
    tokens at the mean row length), as the trainer does; (2) the registered
    optimizer is **bitsandbytes 8-bit AdamW (`adamw_8bit`) on GPU for every arm
    and stage that trains**, the yaml field keeps the legacy `AdamW` spelling
    resolved by the loader, and `run.json` records the resolved optimizer (code
    change on branch `fix/optim-record`); (3) Stage 2 registers a **new
    2,048-token prompt-cap selection rule**, applied at selection time
    identically for both arms, so Stage 2 rows are at most 2,048 + 8,192 =
    **10,240 tokens** and the T-dial-rl memory bench and "fits" test use
    10,240-token rows; the `prereg-v1-data` addendum records the excluded count
    and its interaction with the 1,600-prompt cycling rule.     All three were made
    after Stage A was shown to the author in chat at ≈06:02 (SendUserFile;
    Stage A sealed in `38b5bd2` at 05:56:26 before any result was opened), no
    Stage A content was involved, and `analysis_rules_frozen_at_commit` stays
    `8c6d78d` (item 3 is the only new selection rule). The hours-assumption
    pointer moves from `mde_notes` to the compute notes.

**Round 8 choices (the dated pre-tag amendment "Revision round 8 (external
review panel)").** These are recorded in `prereg.yaml` under
`revision_round8.choices`, split into claim clarification, analysis precision,
provenance correction, execution contingency and new eligibility design as X1
asked.

53. Recorded the four pre-tag smoke attempts exactly as they happened
    (argparse/path launcher crash, GPU-only `_model_device` `NameError`, OOM in
    `recompute_engine_logits`, completed SMOKE-FAIL) and stated that the
    current refill candidate is **not eligible**.
54. Recorded the 11:00–11:07 stream diagnostic as supporting evidence that no
    refill sampling-stream bug exists, while keeping the sampled exactness
    deficit as a real unresolved failure.
55. Set the engine pin to the `fix/gate-v6` commit that the repaired smoke ran
    on, recorded in a pins addendum; `eed7fa7` stays historical gate-v5
    provenance. **Round 8c correction:** the repaired smoke never ran and
    v6 remained unmerged, so this pin is retired; the padded source is pinned
    instead.
56. Registered the gate v6 calibration `[0,2,4,6]/[1,3,5,7]` and validation
    `[0,4,1,5]/[2,6,3,7]` partitions, chosen-token regret `D`, τ_g, the 0.50
    inconclusive ceiling, mandatory decision-trace reconstruction, the
    stable/changed-support certificates, raw/TV envelopes and caps, matched
    denominators, the 3/8 floor, ≥ 32 shared steps, one repair cycle and
    padded-on-failure.
57. Chose token-weighted NLL with problem-level resampling as the primary D1
    estimand (equal-problem alternative declared, not used); registered
    `p = (1 + #T* <= 0)/10001` with conservative ties and no interpolation;
    temperatures fitted on 60-61 and held fixed; an empty/unusable corpus makes
    D1 unevaluable with no rejection and no α transfer; fit/test IDs frozen in
    the data addendum.
58. Set the Stage 2 mandatory paths to O-H N=1 and L N=2/3/4 at 10,240
    tokens, with N=6/8 optional under D-7b; the fallback must itself fit; a
    mandatory-path failure stops before either arm and issues a new version.
59. Chose E4B bf16 with CPU PLE as primary and int8 as its fallback; chose 12B
    NF4 as primary and the official QAT Q4_0 GGUF through llama.cpp as its
    registered fallback, with the exporter implemented only if NF4 fails.
60. Registered "best qualified resident runtime per teacher" as the teacher
    comparison and froze one runtime per teacher for its L and O-H.
61. Computed the E4B text-resident denominator from the pinned `4ecc2ba`
    safetensors header with `huggingface_hub.get_safetensors_metadata`
    (14.003 GiB = 8.753 GPU-side text + 5.250 CPU PLE; whole checkpoint
    14.894 GiB) and froze ρ = 1.62.
62. Demoted Google's gap fractions φ and the byte-equivalent multiplier m to
    illustrations; fixed-byte paired effects remain the main presentation.
63. Added the LOCAL Stage-0-lite sampler as a pilot-data compute-tier option
    (padded HF engine, same sampling, cap = `max_length` minus prompt length);
    the full Stage 0 for main runs remains Modal-or-local by availability.
    Round 8c replaces the unsupported cross-backend "identical accepted set"
    claim with shared frozen artifacts and recorded backend provenance.
64. Registered the pilot "why" report as descriptive diagnostics that never
    enter selection.
65. Registered all round-8 changes as one dated pre-tag amendment split into
    claim clarification, analysis precision, provenance correction, execution
    contingency and new eligibility design; kept `8c6d78d` as the historical
    pre-Stage-A rules snapshot; the final `prereg-v1` commit is the
    authoritative complete specification.

**Round 8b choices (the dated pre-tag technical amendment "Revision round
8b").** These are also recorded in `prereg.yaml` under
`revision_round8b.choices`. Stage A was exposed, so none of these items uses a
trained-arm outcome.

66. Registered all round-8b items as one dated pre-tag technical amendment and
    stated explicitly that Stage A was exposed and that no item uses a
    trained-arm outcome.
67. Kept the MONITOR set at 64 train-bucket problems chosen by the largest
    `sha256(seed:problem_id)` keys, one sample each, rather than carving a
    monitor split out of the training rows.
68. Registered rows as sampled completions for P1 (200 rows = 50 problems × 4
    samples) instead of changing the split problem count.
69. Labelled the pilot stages explicitly (teacher stage, then block/bridge
    stage on the E4B arms) and carried a 12B win through to final training
    without treating 12B as a block/bridge candidate.
70. Used the OR math rule because both the Windows multiprocessing failure and
    the `1,234` disagreement are real; `decided_by` is recorded so the verifier
    is auditable.
71. Registered the Part-1 pool mode as an explicit builder mode without
    creative briefs, with the Part 2 briefs decontaminated against the frozen
    pool later.
72. Closed the refill track outright instead of keeping a conditional branch
    the project will not exercise.
73. Preferred Stage 0 local samples for teacher qualification and kept the
    85-row P1-lite extension as the fail-closed fallback.
74. Registered Docker as a hard dependency of the verified code targets;
    round 8e supersedes the historical math-only fallback with the frozen
    verifier and no unsandboxed run for model-generated code.
75. Registered the DeepSeek/NVIDIA/OpenCode rules with per-phase caps and
    explicit approval, and kept Modal optional for Part 1.
76. Left three explicit placeholder pins instead of inventing commit hashes
    and added a tag gate that refuses to tag while any remains.

**Round 8c choices (the final pre-tag technical amendment).** These are also
recorded in `prereg.yaml` under `revision_round8c.choices`. Stage A was
exposed, so none uses a trained-arm outcome.

77. Made padded the sole operative engine in both documents and marked
    `engine_gate`/`part_b` INACTIVE instead of deleting the history.
78. Recorded the realized split counts from the frozen pool files and kept the
    nominal numbers explicitly nominal.
79. Treated P1 as 200 attempts with a deterministic truncation and a fixed
    retained-row rule, not 200 verified rows.
80. HISTORICAL ONLY (superseded by round 8e): froze the pilot domain as
    math-only until Docker code verification, with code eligibility locked
    after any preliminary math-only ranking.
81. Chose fail-closed minima (20 P1, 20 T-fit, 30 selection, 32 MONITOR,
    20/40 D1 fit/score) and no borrowing, regeneration or extra epoch.
82. Required `--ack-abort` on every registered resume path and limited it to
    resuming the same run from the last COMPLETE checkpoint.
83. Bound the model-level temperature-fit path by commit/blob and failure
    behaviour; excluded the tensor-helper defaults.
84. Replaced O-H isolation/exact parameter equality with matched LoRA capacity
    plus separately itemized L bridge/package effects, and labelled D4's
    test-selected winner exploratory/selected/optimistic.
85. Filled p1corpus and modal-imports with full 40-character ids and left
    healpilot plus modal-image as the only placeholders.

**Round 8d choices (the final pre-tag technical amendment).** These are also
recorded in `prereg.yaml` under `revision_round8d.choices`. Stage A was
exposed, so none uses a trained-arm outcome.

86. Stated the probe schedule exactly as implemented (first step boundary at
    or above each 500k increment plus train end at the registered 3M dose)
    instead of hand-computed fixed token counts.
87. Added the `finish_reason != "length"` exception to the generation-sanity
    rule while keeping repetition loops unconditional; a length-truncated
    trace is expected.
88. Reworded the two historical round-8c mentions without the literal
    placeholder token so the whole-document tag gate can pass.
89. Filled the last pins with full 40-character ids (healpilot `e3f63f3`,
    modal image `6d3bb65`) and initially pinned the Modal lite code
    `9c4cfd2`; round 8e corrects it to `4c260eb`, the seed-fixed commit the
    successful run used.
90. Moved Stage-0-lite to Modal vLLM as one backend under the $6 run cap
    within the $30 monthly ceiling; discarded the local partial shards from
    this run.
91. HISTORICAL ONLY (superseded by round 8e): kept the pilot fit/selection
    domain math-only; Docker availability (2026-11-05 22:45) was recorded for
    later-stage code verification only.

**Round 8e choices (the panel-ruled pilot shortfall amendment).** These are
also recorded in `prereg.yaml` under `revision_round8e.choices`. Stage A was
exposed, and the conditional rule was fixed by the panel before the realized
counts; none uses a trained-arm outcome or pilot ranking.

92. Adopted A: the pilot verified domain is math plus code passing the frozen
    Docker verifier; chat never counts. Confirmatory D1 remains math-only.
93. Adopted C unconditionally: P1 is descriptive for every count and never
    gates P2; the historical 50×4 = 200 means attempts, never 200 verified
    rows.
94. Adopted B only at the fixed count trigger: exactly four additional
    attempts, indices 4–7, for every original P2 math/code problem, with the
    corrected request-id/seed formula.
95. Kept the original final adequacy floors: P2 runs only at final T-fit ≥ 20
    AND selection ≥ 30 distinct verified problems; otherwise the pilot is
    unevaluable and P2 does not run.
96. Rejected D and E: no lower minimums, relaxed reasoning-share filter,
    manufactured answers, prefix scoring, second batch, borrowed bucket or
    domain substitution.
97. Registered `prereg/pilot-shortfall-freeze.json` as REQUIRED before any
    pilot scoring or P2 training, with the staged freeze stated explicitly.
98. Recorded the math-only and preview union counts as facts, not decision
    inputs; the official count is re-run with the frozen verifier.
99. Corrected `task_modallite_commit` from the pre-fix `9c4cfd2` to the
    seed-fixed `4c260eb` used by the successful run, retaining `9c4cfd2` and
    its $0.758 failure as history.
