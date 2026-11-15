"""Pre-registration schema and the pre-registered success criteria (S1-S5).

The file ``prereg/prereg.yaml`` is committed and tagged ``prereg-v1`` before any
final run.  This module loads it, validates the invariant fields and turns a set
of :class:`~gemmaloop.eval.results.RunResult` files into explicit verdicts.

Statistical rules (spec section 5.6, reference 04, Know-how section 5):

* Criteria are computed **per model size**.  A result set that mixes E2B and E4B
  is rejected unless the caller names the size explicitly.
* Arms may have several seeds.  Every item's score is the mean correctness over
  all seeds and samples of that arm; only matching item ids are paired.  The
  primary interval is the **paired percentile bootstrap** over item ids (10,000
  resamples, seed 0 by default), and per-seed intervals are reported as
  robustness.
* ``paired_diff``/``paired_bootstrap`` collapse to a zero-width interval when
  there is one item or every paired difference is identical.  A single item is
  therefore *not* evidence: ``check_criteria`` returns ``passed=None`` with
  ``"insufficient data"`` unless there are at least two paired items.
* N = 3 and N = 4 are exploratory.  Their p-values are corrected with Holm's
  step-down procedure and their verdicts say so; monotonicity in N is never
  assumed (reference 02, Trampa 2).
* Harness validity follows the pre-registered rule: the original model is valid
  for a benchmark when ``|mine - google| <= tolerance`` **or** Google's figure
  lies inside the Wilson 95 % interval of ``mine``.  The Wilson interval is
  computed over **items** (samples are averaged per item first), never over
  generated samples.  When that interval is wider than the tolerance the detail
  is flagged as a low-power check (this happens in GPQA and AIME).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gemmaloop.eval.results import RunResult
from gemmaloop.eval.stats import (
    holm,
    mde,
    paired_bootstrap,
    paired_diff,
    wilson,
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

AUTHOR_DECLARATION_ENGLISH = chr(10).join(
    (
        "1. The author has seen aggregate original-model verification counts "
        "(the Docker preview; v1/v2 counts: T-fit 27, selection 40, P1 16; "
        "Stage-0-lite math counts). Per-request records were inspected by the "
        "orchestrator and the reviewers.",
        "2. There was no pilot candidate execution outside the preserved "
        "records (the codeverify preview, v1, v2 and the void v3 run).",
        "3. No pilot scoring has been performed, no looped pilot arm has been "
        "trained, and no trained-arm outcome has been seen.",
        "4. Outside the pilot there were: the untrained Stage A evaluations "
        "(R and R-damped), the E4B teacher precompute, and a single optimizer "
        "step on 128 synthetic tokens for the probe-memory smoke, which used "
        "no pilot data and produced no evaluation outcome.",
    )
)
from gemmaloop.serve.generate import Sampling


# ---------------------------------------------------------------------------
# YAML schema
# ---------------------------------------------------------------------------


class BenchSpec(BaseModel):
    """Evaluation size and sampling for one benchmark."""

    model_config = ConfigDict(extra="forbid")

    limit: int
    samples: int = 1
    seed: int = 0
    # Vision benchmarks compare against Google at 1120 soft tokens.
    image_soft_tokens: int = 280
    exploratory: bool = False
    # Required on purpose: no benchmark may silently default to "confirmatory".
    # confirmatory = fixed-sequence E2-E4; secondary = estimation only;
    # o_only = base model only; harness = Google reproduction check only.
    role: Literal["confirmatory", "secondary", "o_only", "harness"]
    # False for a virtual harness view computed from another bench's run
    # (lcb_google is O's full lcb run, not a separate generation).
    generated: bool = True
    source_bench: str | None = None


class Criteria(BaseModel):
    """Pre-registered decision rules and statistics settings."""

    model_config = ConfigDict(extra="forbid")

    s1_tolerance_pts: float = 3.0
    s2_bench: str = "lcb"
    s3_min_win_rate_lower: float = 0.5
    s4_min_gain_pts: float = 0.0
    bootstrap_iters: int = 10_000
    bootstrap_seed: int = 0
    exploratory_ns: list[int] = Field(default_factory=lambda: [3, 4])


class CreativeSpec(BaseModel):
    """Frozen creative-evaluation inputs (Task 19 fills the hashes)."""

    model_config = ConfigDict(extra="forbid")

    n_briefs: int = 150
    briefs_sha256: str | None = None
    judge_model: str = "deepseek-flash"
    judge_version: str = "DeepSeek-V4.1-Flash-1017"
    second_judge_model: str | None = None
    second_judge_revision: str | None = None
    human_pilot: int = 40
    human_main: int = 120


class TrainingSpec(BaseModel):
    """Training length settings used by the data build and the trainer.

    The values are placeholders until Part B fills them from the calibration /
    performance results.
    """

    model_config = ConfigDict(extra="forbid")

    max_length: int = 1024
    length_mode: str = "filter"

    @model_validator(mode="after")
    def _validate_length(self) -> "TrainingSpec":
        if self.max_length <= 0:
            raise ValueError("training.max_length must be positive")
        if self.length_mode not in {"truncate", "filter"}:
            raise ValueError(
                "training.length_mode must be 'truncate' or 'filter', got "
                f"{self.length_mode!r}"
            )
        return self


class Phase2Spec(BaseModel):
    """Fields frozen only in ``prereg-v2`` (Task 22), before collecting data."""

    model_config = ConfigDict(extra="forbid")

    eval_tasks_sha256: str | None = None
    max_length: int | None = None
    usd_cap: float | None = None
    s4_min_gain_pts: float = 0.0


# ---------------------------------------------------------------------------
# Two-stage hybrid design (prereg-v1)
#
# These sections are optional so that the historical single-SFT fixture used by
# the unit tests (and by ``prereg-v2`` addenda) still validates.  The committed
# ``prereg/prereg.yaml`` fills every section; the validators below reject
# internally inconsistent values whenever a section is present.
# ---------------------------------------------------------------------------


class ArmSpec(BaseModel):
    """One confirmatory or diagnostic arm."""

    model_config = ConfigDict(extra="forbid")

    role: str
    description: str


class Part1Spec(BaseModel):
    """Confirmatory Part 1: E2B, seed 0, fixed two-stage recipe."""

    model_config = ConfigDict(extra="forbid")

    size: str = "E2B"
    seeds: list[int] = Field(default_factory=lambda: [0])
    confirmatory_seed: int = 0
    extra_seeds: list[int] = Field(default_factory=list)
    extra_seeds_role: str = (
        "robustness only; extra seeds never replace the seed-0 verdict"
    )
    extra_seeds_policy: str = ""
    arms: dict[str, ArmSpec] = Field(default_factory=dict)
    diagnostic_arms: list[str] = Field(default_factory=list)
    removed_arms: list[str] = Field(default_factory=list)
    optional_arm: str | None = None
    optional_arm_condition: str = ""
    final_checkpoint_only: bool = True
    # Kept optional for older fixtures; the registered Stage 2 policy is
    # ``rollout_n_policy`` (dial-aware, stratified per prompt group).
    rollout_n: int | None = None
    rollout_n_policy: str = ""
    # Structured Stage 2 mix: N -> prompt groups per step.  The prose policy
    # above is mirrored here so the sums-to-16 rule is machine-checked.
    rollout_n_groups: dict[int, int] = Field(
        default_factory=lambda: {2: 3, 3: 3, 4: 8, 6: 1, 8: 1}
    )


class LoRASpec(BaseModel):
    """LoRA and optimizer settings shared by both training stages."""

    model_config = ConfigDict(extra="forbid")

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target: str = ""
    target_option: str = "loop_block"
    lr: float = 2e-4
    bridge_lr: float = 3e-3
    loop_ple_lr: float = 1e-3


class BridgeSpec(BaseModel):
    """Loop bridge definition and the legacy scalar fallback."""

    model_config = ConfigDict(extra="forbid")

    primary: str = "zero_init_norm_matched_per_channel_g"
    formula: str = ""
    init: str = "g = 0"
    params: int = 1536
    lr: float = 3e-3
    legacy_scalar_w0: float = 0.99
    scope: str = ""


class ArchitectureSpec(BaseModel):
    """Architecture candidates and the N-adaptation rule chosen by the pilot."""

    model_config = ConfigDict(extra="forbid")

    loop_block_candidates: list[str] = Field(default_factory=list)
    bridge_candidates: list[str] = Field(default_factory=list)
    adapters_active_from_n: int = 2
    n_distribution: list[float] = Field(default_factory=list)
    n_distribution_note: str = ""
    adapter_sharing: str = ""
    ple_mode: str = "shared"
    ple_mode_note: str = ""
    frozen_prefix: str = ""
    layer_scalar_documented: bool = True
    kv_policy: str = ""
    lora: LoRASpec = Field(default_factory=LoRASpec)
    bridge: BridgeSpec = Field(default_factory=BridgeSpec)


class DialObjectiveSpec(BaseModel):
    """Anytime multi-exit training objective for every L arm.

    Every pass in the truncated-BPTT window emits an exit (coda + LM head); the
    exits are averaged uniformly and a stop-gradient monotone hinge penalises a
    later exit that is worse than its predecessor.  The prelude and pass 1 are
    frozen and run without autograd.
    """

    model_config = ConfigDict(extra="forbid")

    exits: str = ""
    window_trainable_passes: int = 3
    horizon: list[float] = Field(default_factory=list)
    exit_loss: str = ""
    exit_weights: str = ""
    monotone_hinge: dict[str, Any] = Field(default_factory=dict)
    excluded: list[str] = Field(default_factory=list)
    logging: list[str] = Field(default_factory=list)
    oh: str = "unchanged single exit at N=1"
    bptt: str = ""
    exit_coverage: str = ""


class FlopsSpec(BaseModel):
    """Analytic per-pass cost fractions and the registered FLOPs ratios r_N."""

    model_config = ConfigDict(extra="forbid")

    n1_total_gflop: float = 4.559
    per_pass_fraction: dict[str, float] = Field(default_factory=dict)
    r_n: dict[str, dict[str, float]] = Field(default_factory=dict)
    r4_rule: str = ""
    accounting: str = ""


class ResidentBytesSpec(BaseModel):
    """Measured resident-byte denominator for the intelligence-per-byte metric."""

    model_config = ConfigDict(extra="forbid")

    role: str = "estimation only"
    denominator: str = ""
    unit: str = "GiB"
    components: list[str] = Field(default_factory=list)
    excluded: str = ""
    vision_audio: str = ""
    dtypes: str = ""
    hosts: str = ""
    merged: str = ""
    formula: str = ""
    expected_bf16: dict[str, str] = Field(default_factory=dict)
    lo_ratio: str = ""
    invariants: str = ""


class ByteFrontierSpec(BaseModel):
    """D6: the registered byte frontier (estimation only, no gate)."""

    model_config = ConfigDict(extra="forbid")

    id: str = "D6"
    name: str = "byte frontier"
    status: str = "estimation only, no gate"
    test: str = ""
    priority: str = "after E4"
    cost_gpu_hours: str = "<= 0.5"


class IntelligencePerByteSpec(BaseModel):
    """The registered intelligence-per-resident-byte metric.

    Accuracy and calibrated NLL stay on two axes and are never divided into a
    single number; the derived scalar is pass@1 per GiB, shown only beside both
    components.  Everything here is estimation only and never gates E1-E4.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = "estimation only; never gates E1-E4"
    denominator: str = "resident_bytes"
    quality: dict[str, str] = Field(default_factory=dict)
    derived: dict[str, str] = Field(default_factory=dict)
    fixed_bytes_effects: dict[str, str] = Field(default_factory=dict)
    anchoring: dict[str, str] = Field(default_factory=dict)
    secondary: dict[str, str] = Field(default_factory=dict)
    reference_points: list[str] = Field(default_factory=list)
    table: str = ""
    charts: list[str] = Field(default_factory=list)
    limitations: str = ""
    d6: ByteFrontierSpec = Field(default_factory=ByteFrontierSpec)


class TeacherSpec(BaseModel):
    """Stage 1 teacher, objective and offline top-k logit storage."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[str] = Field(default_factory=list)
    selection: str = ""
    objective: str = ""
    beta_parent_kl: float = 0.1
    parent_kl_direction: str = "KL(parent || student)"
    parent_kl_live: bool = True
    parent_kl_at_every_exit: bool = True
    top_k: int = 64
    top_k_tail: str = ""
    teacher_temperature: float = 1.0
    logits_dtype: str = "float16"
    logits_precomputed_offline: bool = True
    vocab_mask_token: str = "<glitch_pruned_249788>"
    same_teacher_for_oh: bool = True


class Stage1Spec(BaseModel):
    """Stage 1 heal: forward KL from a same-tokenizer teacher plus parent KL."""

    model_config = ConfigDict(extra="forbid")

    teacher: TeacherSpec = Field(default_factory=TeacherSpec)
    oh_runs_stage1: bool = True
    oh_stage1_note: str = ""
    data: str = ""
    budget_tokens: int = 12_000_000
    dose_logging_interval: int = 2_000_000
    dose_logging_role: str = "descriptive only; no decision use"
    # Legacy spelling kept so the yaml field stays loadable.  The registered
    # optimizer is bitsandbytes 8-bit AdamW (adamw_8bit) on GPU for every arm
    # and stage that trains; run.json records the resolved optimizer.
    optimizer: str = "AdamW"
    # Round 7c: the trainer uses per-device batch 1 row x gradient
    # accumulation 16 rows per optimizer step.  ``rows_per_optimizer_step``
    # replaces ``tokens_per_optimizer_step``; the latter is kept only so older
    # fixtures still validate (the trainer does not read it).
    rows_per_optimizer_step: int = 16
    tokens_per_optimizer_step: int = 8192
    gradient_clip: float = 1.0
    schedule: str = "token-cosine with 3% warmup, ends at the 12M budget"
    recipe_binding: str = ""
    loss_tokens: str = "all non-pad tokens"
    teacher_runtime: str = "E4B nf4 or bf16; 12B QAT q4_0 int4"
    numerics: str = ""
    description: str = ""
    # Round 8 (X3): registered per-teacher runtimes, the ordered
    # qualification rule, the manifest identity and the fairness rule.
    runtimes: dict[str, str] = Field(default_factory=dict)
    runtime_qualification: str = ""
    qualification_corpus: str = ""
    # Round 8b: the fail-closed fallback order for the teacher qualification
    # corpus (Stage 0 local samples first, P1-lite confined to the boundary
    # checks only).
    qualification_fallback: str = ""
    qualification_references: str = ""
    qualification_metrics: str = ""
    acceptance_table: list[str] = Field(default_factory=list)
    expansion_rule: str = ""
    fallback_order: list[str] = Field(default_factory=list)
    manifest_identity: str = ""
    comparison_rule: str = ""
    four_bit_limitation: str = ""
    execution_repair: str = ""
    freeze_rule: str = ""
    # Round 8b/8c: restart provenance, token accounting and the dose-shortfall
    # contingency.
    initialization: str = ""
    token_accounting: str = ""
    dose_shortfall_rule: str = ""

    @property
    def warmup_fraction(self) -> float:
        """The warmup fraction registered in the schedule text (3 % -> 0.03)."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*warmup", self.schedule or "")
        return float(match.group(1)) / 100.0 if match else 0.03


class DeepSeekSpec(BaseModel):
    """DeepSeek V4.1 Flash uses that are allowed (and forbidden) in Part 1."""

    model_config = ConfigDict(extra="forbid")

    model: str = "DeepSeek-V4.1-Flash-1017"
    served_version: str = ""
    fingerprint_record: str = ""
    api: str = "official DeepSeek Open Platform API only"
    uses: list[str] = Field(default_factory=list)
    never: list[str] = Field(default_factory=list)
    opencode_hosted: str = "never for anything that enters training or results"
    terms: str = ""


class RftSpec(BaseModel):
    """Registered P3 fallback: rejection-sampled fine-tuning on self-traces."""

    model_config = ConfigDict(extra="forbid")

    data: str = "O's verified-correct self-traces from Stage 0"
    tokens: int = 12_000_000
    # Legacy spelling; the registered resolution is adamw_8bit on GPU (see
    # ``Stage1Spec.optimizer``).
    optimizer: str = "AdamW"
    lora_lr: float = 2e-4
    bridge_lr: float = 3e-3
    schedule: str = "token-cosine with 3% warmup, ends at the 12M budget"
    loop_probs: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.70, 0.075, 0.075, 0.075, 0.075]
    )
    loss: str = "multi-exit CE at the window exits, uniform weights, same hinge"
    description: str = ""


class Stage2Spec(BaseModel):
    """Stage 2 Dr. GRPO RLVR, identical for O-H and L."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "Dr. GRPO"
    library: str = "TRL 1.17.1 GRPOTrainer"
    # Kept optional for older fixtures; the registered policy is
    # ``rollout_n_policy``.
    rollout_n: int | None = None
    rollout_n_policy: str = ""
    rollout_n_groups: dict[int, int] = Field(
        default_factory=lambda: {2: 3, 3: 3, 4: 8, 6: 1, 8: 1}
    )
    # Round 7b: execution-only contingency.  If the T-dial-rl memory bench
    # shows that the N=6/N=8 groups do not fit, they both become N=4
    # (3/3/10 over N=2/3/4).  No selection or analysis rule changes.
    rollout_n_fallback_policy: str = ""
    rollout_n_fallback_groups: dict[int, int] = Field(
        default_factory=lambda: {2: 3, 3: 3, 4: 10}
    )
    t_dial_rl_memory_bench: str = ""
    # Round 8 (X1 item 4): the Stage 2 feasibility branch.  The mandatory
    # O-H and L N=2/3/4 paths must fit at the actual 10,240-token maximum,
    # including rollout storage and optimizer state; if a mandatory path does
    # not fit, stop before either main Stage 2 arm and issue a new version.
    feasibility_bench: str = ""
    mandatory_paths: str = ""
    rollout_storage: str = ""
    optimizer_state: str = ""
    fallback_must_fit: str = ""
    mandatory_stop_rule: str = ""
    # Round 6: the Stage 2 policy-gradient backward uses the same truncated
    # window as Stage 1 for every N (passes <= N-3 run under no_grad).
    bptt_window: str = ""
    dial_monitoring: str = ""
    oh_rollout_n: int = 1
    prompts_per_step: int = 16
    group_size: int = 8
    steps: int = 100
    completion_cap: int = 8192
    # Round 7d: Stage 2 prompt cap.  Prompts above this length are excluded
    # from the Stage 2 pool at selection time, identically for both arms, so a
    # training row is at most prompt_cap + completion_cap = 10,240 tokens.
    prompt_cap: int = 2048
    # Legacy spelling; the registered resolution is adamw_8bit on GPU (see
    # ``Stage1Spec.optimizer``).
    optimizer: str = "AdamW"
    lora_lr: float = 2e-5
    bridge_lr: float = 1e-3
    beta_kl: float = 0.0
    tis_cap: float = 2.0
    num_iterations: int = 1
    dr_grpo_normalizer: int = 8192
    grader_timeout_s: int = 6
    sampling: Sampling = Field(
        default_factory=lambda: Sampling(
            temperature=1.0, top_p=1.0, top_k=0
        )
    )
    reward: str = ""
    verifiers: str = ""
    prompt_selection: str = ""
    prompt_shortfall_rule: str = ""
    initialization: str = ""
    truncation: str = ""
    same_for_both_arms: bool = True
    engine: str = ""
    deepseek: DeepSeekSpec = Field(default_factory=DeepSeekSpec)
    fallback_rft: RftSpec = Field(default_factory=RftSpec)


class HealEstimateSpec(BaseModel):
    """Descriptive heal-effect margins; E1 is not a gated TOST."""

    model_config = ConfigDict(extra="forbid")

    nll_nats: float = 0.01
    lcb_pts: float = 3.0
    ci_level: float = 0.90
    gate: bool = False


class BootstrapSpec(BaseModel):
    """Registered bootstrap settings."""

    model_config = ConfigDict(extra="forbid")

    iters: int = 10_000
    seed: int = 0
    unit: str = "problem cluster"
    confirmatory_seed: int = 0
    extra_seed_role: str = (
        "robustness only; extra seeds never replace the seed-0 verdict"
    )
    single_seed: str = "paired problem-cluster bootstrap"
    multi_seed: str = "crossed pigeonhole bootstrap over problems x seed pairs"
    sensitivity: str = ""


class EndpointSpec(BaseModel):
    """One endpoint in the fixed-sequence confirmatory family."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    test: str
    margin: str = ""
    bootstrap: str = ""


def _default_precondition() -> EndpointSpec:
    return EndpointSpec(
        id="E1",
        name="precondition: bit-exact L(N=1) and heal estimates",
        test=(
            "L(N=1) is bit-identical to O on every engine used (HF trainer and "
            "the evaluation engine); heal effects are reported as estimates."
        ),
        margin="no alpha, no gate",
    )


def _default_sequence() -> list[EndpointSpec]:
    return [
        EndpointSpec(id="E2", name="S2 loop effect", test=""),
        EndpointSpec(id="E3", name="L(N=2) vs O, total effect", test=""),
        EndpointSpec(id="E4", name="budget-curve AUC", test=""),
    ]


class DialEndpointSpec(BaseModel):
    """One endpoint of the dial family (D1 is confirmatory)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    test: str
    status: str = "confirmatory"
    corpus: str = ""
    temperature: str = ""
    bootstrap: str = ""
    checkpoint: str = ""
    margin: str = ""


class NllAnalysisSpec(BaseModel):
    """Round 8 (X1 item 3): the complete calibrated-NLL analysis contract.

    The estimand is token-weighted calibrated NLL with problem-level
    resampling; every field here is a registered analysis rule, not a runtime
    knob.  The fit/test trace IDs are frozen in the data addendum before any
    trained-arm output is examined.
    """

    model_config = ConfigDict(extra="forbid")

    supervised_token_mask: str = ""
    weighting: str = ""
    statistic: str = ""
    p_value: str = ""
    temperature: str = ""
    unevaluable_corpus: str = ""
    fit_test_freeze: str = ""
    denominators: str = ""


class EndpointsSpec(BaseModel):
    """Fixed-sequence confirmatory family and its statistics."""

    model_config = ConfigDict(extra="forbid")

    precondition: EndpointSpec = Field(default_factory=_default_precondition)
    sequence: list[EndpointSpec] = Field(default_factory=_default_sequence)
    primary: str = "E2"
    heal_estimates: HealEstimateSpec = Field(default_factory=HealEstimateSpec)
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)
    nll_analysis: NllAnalysisSpec = Field(default_factory=NllAnalysisSpec)
    fixed_sequence_rule: str = ""
    multiplicity: str = ""
    # Structured graphical procedure (Bretz et al. 2009): weights, transition
    # matrix, p-value definition and operational order.
    graphical: dict[str, Any] = Field(default_factory=dict)
    dial_family: list[DialEndpointSpec] = Field(default_factory=list)
    estimation: list[str] = Field(default_factory=list)
    holm: str = ""
    single_seed_statement: str = ""
    google_comparison: str = "reproduction check"
    secondary: list[str] = Field(default_factory=list)
    o_only: list[str] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list)
    priority_note: str = ""


class PilotP1Spec(BaseModel):
    """P1: forward-only sweep over the loop-block candidates."""

    model_config = ConfigDict(extra="forbid")

    rows: int = 200
    # Round 8b: P1 counts sampled completions (rows), not problems.
    problems: int = 50
    samples_per_problem: int = 4
    row_definition: str = ""
    truncation_rule: str = ""
    retained_row_rule: str = ""
    domain_rule: str = ""
    # Round 8e: P1 is descriptive for every count and never gates P2.
    role: str = "descriptive"
    gates_p2: bool = False
    continuation_gate: str = ""
    small_sample_threshold: int = 20
    zero_count_rule: str = ""
    min_verified_problems: int = 20
    shortfall_rule: str = ""
    blocks: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    gpu_hours: float = 0.3
    r_rdamped_n_range: str = ""
    description: str = ""


class PilotP2Spec(BaseModel):
    """P2: short heals and the architecture/teacher selection rule."""

    model_config = ConfigDict(extra="forbid")

    heal_tokens: int = 3_000_000
    candidates: str = ""
    arms: list[str] = Field(default_factory=list)
    verified_corpus: str = ""
    endpoints: list[str] = Field(default_factory=list)
    temperature_fit: str = "per (arm, N) on a separate fit split"
    bootstrap: str = "problem-cluster bootstrap"
    teacher_selection_rule: str = ""
    teacher_gain_basis: str = ""
    block_bridge_rule: str = ""
    block_bridge_teacher: str = ""
    selection_rule: str = ""
    stop_rule: str = ""
    # Rule B, corrected in round 6 (native review C1/I1).
    eligible_set: str = ""
    ranking_statistic: str = ""
    monotonicity_filter: str = ""
    fallback_rule: str = ""
    oh_lora_target: str = ""
    min_arm_vs_oh_nats: float = -0.02
    tie_nats: float = 0.003
    tie_break: list[str] = Field(default_factory=list)
    teacher_choice: str = "mean gain of L and O-H; E4B tie-break"
    # Round 8b: the two registered pilot stages and the carry-through rule.
    teacher_stage: str = ""
    block_bridge_stage: str = ""
    final_recipe_teacher_rule: str = ""
    domain_and_freeze: str = ""
    shortfall_rule: str = ""
    kl_gap_secondary: str = "fraction of the teacher KL gap closed at N=2"
    dial_rule: str = ""
    dial_metrics_reported: str = ""
    frozen_rule_deviation: str = ""
    anytime_tax_arm: str = ""
    # Round 7: every pilot quantity is a selection input only; the
    # confirmatory claims are E2-E4 and D1.
    selection_only: bool = True


class PilotP3Spec(BaseModel):
    """P3: short RL on O only, deciding Stage 2 versus the RFT fallback."""

    model_config = ConfigDict(extra="forbid")

    model: str = "O only"
    steps: int = 30
    dev: str = "200 held-out math problems x 4 samples"
    prompt_bucket: str = ""
    completion_cap: int = 4096
    prompt_median_max: int = 2500
    keep_rule: str = ""
    paired_se: float = 2.0
    max_truncation: float = 0.10
    fallback: str = "RFT on verified self-traces"
    hacking_flags: str = "not applicable to exact-answer math; code audit only"
    description: str = ""


class PilotDataSpec(BaseModel):
    """Pilot data boundaries; the held-out test split is never scored."""

    model_config = ConfigDict(extra="forbid")

    pilot_train_tokens: int = 3_000_000
    pilot_dev_problems: int = 200
    pilot_dev_samples: int = 4
    temperature_fit_problems: int = 200
    long_traces: int = 500


class PilotCodeVerificationSpec(BaseModel):
    """Round 8e: the frozen Docker code verifier for pilot targets."""

    model_config = ConfigDict(extra="forbid")

    domain: str = ""
    chat_rule: str = ""
    test_set: str = ""
    test_manifest: str = ""
    extraction: str = ""
    wrapper: str = ""
    output_comparison: str = ""
    grading_implementation: str = ""
    dependency_versions: str = ""
    docker_image: str = ""
    docker_image_digest: str = ""
    sandbox_limits: str = ""
    timeout_rule: str = ""
    pass_rule: str = ""
    fail_rule: str = ""
    infrastructure_rule: str = ""
    unsandboxed_fallback: str = ""
    current_count_rule: str = ""
    freeze_index: str = ""


class PilotShortfallSpec(BaseModel):
    """Round 8e: the prospective pilot shortfall amendment (A, B and C)."""

    model_config = ConfigDict(extra="forbid")

    rule: str = ""
    domain: str = ""
    p1_descriptive_only: str = ""
    p1_gates_p2: bool = False
    p2_domain: str = ""
    chat_never: str = ""
    extension_trigger: str = ""
    attempt_indices: list[int] = Field(default_factory=lambda: [4, 5, 6, 7])
    request_id_rule: str = ""
    seed_formula: str = ""
    request_count_rule: str = ""
    configuration: str = ""
    artifact_rule: str = ""
    immutable_original: str = ""
    resume_rule: str = ""
    early_stop_rule: str = ""
    max_extension_batches: int = 1
    final_min_tfit_problems: int = 20
    final_min_selection_problems: int = 30
    final_shortfall_rule: str = ""
    no_relaxation_rule: str = ""
    rejected_alternatives: str = ""
    unused_bucket_rule: str = ""


class PilotShortfallFreezeSpec(BaseModel):
    """Round 8e: the staged freeze index required before pilot scoring."""

    model_config = ConfigDict(extra="forbid")

    index_path: str = ""
    required: bool = True
    rule_frozen_before_data: str = ""
    staged_freeze: str = ""
    required_before: str = ""
    contents: list[str] = Field(default_factory=list)
    official_count: str = ""
    realized_counts: str = ""
    preview_counts: str = ""
    decision_input: str = ""
    pending_rule: str = ""


# ---------------------------------------------------------------------------
# Round 8f: the X1 verifier-interpretation and recomputation addendum.
#
# The addendum text is registered verbatim in
# ``prereg/pilot-shortfall-amendment.md``.  These typed blocks carry its
# operative requirements in the YAML so a sealed file cannot silently fall
# back to the superseded preview count, the v1/v2 membership or the old
# timeout attribution.
# ---------------------------------------------------------------------------


def _empty_registered_paths(value: Any, prefix: str = "") -> list[str]:
    """Return dotted paths of empty strings/lists/mappings in a model dump."""

    found: list[str] = []
    if isinstance(value, str):
        if not value.strip():
            found.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        if not value:
            found.append(prefix or "<root>")
        else:
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                found.extend(_empty_registered_paths(item, child))
    elif isinstance(value, (list, tuple)):
        if not value:
            found.append(prefix or "<root>")
        else:
            for index, item in enumerate(value):
                child = f"{prefix}[{index}]" if prefix else f"[{index}]"
                found.extend(_empty_registered_paths(item, child))
    return found


class V3InterpretationSpec(BaseModel):
    """Round 8f: the authenticated per-test representation interpretation."""

    model_config = ConfigDict(extra="forbid")

    rule: str = ""
    source_revision_binding: str = ""
    record_locator: str = ""
    problem_identity: str = ""
    ordered_test_index: str = ""
    original_values_and_types: str = ""
    input_representation_flags: str = ""
    output_representation_flags: str = ""
    original_stored_test_hashes: str = ""
    derived_execution_text_hashes: str = ""
    old_conversion_replay: str = ""
    no_inference_from_appearance: str = ""
    flat_list_input_rule: str = ""
    flat_list_output_rule: str = ""
    string_unchanged_rule: str = ""
    normalizer: str = ""
    functional_tests: str = ""
    whole_problem_exclusion: str = ""
    exclusion_before_grading: str = ""
    exclusion_inventory: str = ""
    source_test_immutability: str = ""


class V3FreezeSpec(BaseModel):
    """Round 8f: the v3 freeze and the complete original-sample recount."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    contents: list[str] = Field(default_factory=list)
    original_code_requests: int = 0
    selected_requests: int = 0
    recount_rule: str = ""
    regrade_scope: str = ""
    math_and_filters: str = ""
    membership_rule: str = ""
    no_union: str = ""
    executing_implementation: str = ""
    dependency_versions: str = ""
    docker_content_digest: str = ""
    sandbox_settings: str = ""
    timeout: str = ""
    infrastructure_accounting: str = ""


class V2RecomputationSpec(BaseModel):
    """Round 8f: v2 accepted as a wholesale recomputation of v1."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    accepted_rule: str = ""
    request_id: str = ""
    v1_record: str = ""
    v2_record: str = ""
    v2_max_test_duration_s: str = ""
    selection_change: str = ""
    distinct_counts: str = ""
    cause_unknown: str = ""
    v3_membership: str = ""
    no_veto_or_retry: str = ""


class TimeoutLifecycleSpec(BaseModel):
    """Round 8f: the unchanged six-second outer timeout and attribution."""

    model_config = ConfigDict(extra="forbid")

    outer_timeout_s: float = 0.0
    timeout_rule: str = ""
    no_restart: str = ""
    lifecycle_requirement: str = ""
    attribution_classes: dict[str, str] = Field(default_factory=dict)
    retry_count: int = 0
    retry_precommit: str = ""
    retry_configuration: str = ""
    first_conclusive: str = ""
    blocked_rule: str = ""
    cleanup_rule: str = ""
    v2_timeout_fact: str = ""
    outer_deadline_liveness: str = ""


class LifecycleEvidenceSpec(BaseModel):
    """Round 8g: X1 lifecycle evidence design and threat model."""

    model_config = ConfigDict(extra="forbid")

    threat_model: str = ""
    original_model_completions: str = ""
    residual_same_uid_risk: str = ""
    fail_closed_rule: str = ""
    not_candidate_proof_isolation: str = ""
    bootstrap: str = ""
    private_pipe: str = ""
    execution_entry_boundary: str = ""
    handoff: str = ""
    key_isolation: str = ""
    hmac_sha256: str = ""
    per_attempt_key: str = ""
    authenticated_framing: str = ""
    identity_binding: str = ""
    stdout_stderr_separation: str = ""
    state_specific_validation: str = ""
    pass_outcome: str = ""
    wrong_answer_outcome: str = ""
    candidate_error_outcome: str = ""
    outer_deadline_outcome: str = ""
    invalid_evidence_rule: str = ""
    post_deadline_rule: str = ""


class TerminalInfrastructureSpec(BaseModel):
    """Round 8g: the terminal infrastructure-unverified disposition."""

    model_config = ConfigDict(extra="forbid")

    disposition: str = ""
    completion_status: str = ""
    replace_rule: str = ""
    not_relabelled: str = ""
    other_completion_rule: str = ""
    extension_rule: str = ""
    count_status: str = ""
    count_rule: str = ""
    remaining_blockers: list[str] = Field(default_factory=list)
    pending_retry: str = ""
    missing_accounting: str = ""
    superseded_blocked_rule: str = ""


class AttemptLedgerSpec(BaseModel):
    """Round 8g: durable append-only ledger and resume-as-continuation."""

    model_config = ConfigDict(extra="forbid")

    ledger: str = ""
    durable_append_only: str = ""
    commit_start_before_dispatch: str = ""
    commit_result_after_completion: str = ""
    interrupted_attempt: str = ""
    interrupted_retry: str = ""
    never_regrade: str = ""
    resume_rule: str = ""
    not_second_freeze: str = ""
    validation_before_resume: str = ""
    artifact_stages: str = ""
    freeze_identity: str = ""


class AcceptanceGuardsSpec(BaseModel):
    """Round 8f: r2 Important 1-3 acceptance requirements."""

    model_config = ConfigDict(extra="forbid")

    count_acceptance: str = ""
    freeze_acceptance: str = ""
    corpus_index_acceptance: str = ""
    legacy_refusal: str = ""
    executing_module_binding: str = ""
    actual_bytes_derivation: str = ""


class ConsumerDispositionRowSpec(BaseModel):
    """Round 8g: one audited pool-test consumer (audit summary rows 1-16)."""

    model_config = ConfigDict(extra="forbid")

    row: int = 0
    consumer: str = ""
    status: str = ""
    effect: str = ""
    disposition_class: str = ""
    disposition: str = ""
    evidence: str = ""


class ConsumerDispositionSpec(BaseModel):
    """Round 8g: the pre-seal pool-test consumer disposition record."""

    model_config = ConfigDict(extra="forbid")

    audit_path: str = ""
    audit_head: str = ""
    adopted_rule: str = ""
    rows: list[ConsumerDispositionRowSpec] = Field(default_factory=list)
    superseded_by_v3: list[int] = Field(default_factory=list)
    preserve_pool: list[int] = Field(default_factory=list)
    active_unaffected: list[int] = Field(default_factory=list)
    future_affected_execution_block: list[int] = Field(default_factory=list)
    future_unaffected_execution_block: list[int] = Field(default_factory=list)
    unaffected_no_action: list[int] = Field(default_factory=list)
    execution_block_rule: str = ""
    lift_condition: str = ""
    preserve_disclose: str = ""

    @model_validator(mode="after")
    def _validate_dispositions(self) -> "ConsumerDispositionSpec":
        if not self.audit_path:
            return self
        expected_rows = list(range(1, 17))
        if [row.row for row in self.rows] != expected_rows:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions.rows must "
                "register summary-table rows 1-16 in order"
            )
        groups = [
            *self.preserve_pool,
            *self.superseded_by_v3,
            *self.active_unaffected,
            *self.future_affected_execution_block,
            *self.future_unaffected_execution_block,
            *self.unaffected_no_action,
        ]
        if sorted(groups) != expected_rows:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions must partition "
                "rows 1-16 exactly once across the disposition classes"
            )
        if self.preserve_pool != [1]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions.preserve_pool "
                "must be [1]"
            )
        if self.superseded_by_v3 != [4, 6, 7]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions."
                "superseded_by_v3 must be [4, 6, 7]"
            )
        if self.active_unaffected != [5, 8, 9, 14, 15]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions."
                "active_unaffected must be [5, 8, 9, 14, 15]"
            )
        if self.future_affected_execution_block != [2, 3, 10, 13]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions."
                "future_affected_execution_block must be [2, 3, 10, 13]"
            )
        if self.future_unaffected_execution_block != [11, 12]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions."
                "future_unaffected_execution_block must be [11, 12]"
            )
        if self.unaffected_no_action != [16]:
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions."
                "unaffected_no_action must be [16]"
            )
        if not self.audit_path.endswith("audit-pool-test-consumers.md"):
            raise ValueError(
                "pilot.verifier_v3.consumer_audit.dispositions.audit_path "
                "must cite the audit-pool-test-consumers.md file"
            )
        return self


class ConsumerAuditSpec(BaseModel):
    """Round 8f/8g: the pool-test consumer audit and disposition."""

    model_config = ConfigDict(extra="forbid")

    scope: list[str] = Field(default_factory=list)
    livecodebench: str = ""
    active_consumers: str = ""
    future_or_unused: str = ""
    historical_artifacts: str = ""
    new_tests_excluded: str = ""
    seal_prerequisite: str = ""
    dispositions: ConsumerDispositionSpec = Field(
        default_factory=ConsumerDispositionSpec
    )


class VersionHashDisclosureSpec(BaseModel):
    """Round 8f: v1/v2/v3 file-byte hash preservation and disclosure."""

    model_config = ConfigDict(extra="forbid")

    definition: str = ""
    v1: dict[str, str] = Field(default_factory=dict)
    v2: dict[str, str] = Field(default_factory=dict)
    common_test_manifest_sha256: str = ""
    disclosure: str = ""
    v3_preservation: str = ""


class PrerequisiteOrderSpec(BaseModel):
    """Round 8f: prerequisites ordered before the data addendum and seal."""

    model_config = ConfigDict(extra="forbid")

    order: list[str] = Field(default_factory=list)
    before_data_addendum_and_seal: str = ""


class V3VoidSpec(BaseModel):
    """Round 8i: the failed official v3 run and its void disposition."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    created_at: str = ""
    commit: str = ""
    embedded_freeze_manifest_sha256: str = ""
    file_sha256_definition: str = ""
    file_sha256: dict[str, str] = Field(default_factory=dict)
    defect: str = ""
    accounting: str = ""
    synthetic_recovery: str = ""
    basis: str = ""
    disposition: str = ""
    preservation: str = ""
    non_operative_extension: str = ""
    supersedes_scope: str = ""


class V3ReplacementSpec(BaseModel):
    """Round 8i: the single authorized v3r replacement and prerequisites."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    identity: str = ""
    authorization: str = ""
    identity_forwarding_repair: str = ""
    production_path_integration_regression: str = ""
    real_docker_dry_run: str = ""
    review_and_go: str = ""
    freeze_once: str = ""
    full_recount: str = ""
    continuity_record: str = ""
    retry_allowance: str = ""
    no_automatic_v3r2: str = ""
    preserved_unchanged: str = ""
    further_changes: str = ""


class AuthorAccessDeclarationSpec(BaseModel):
    """Round 8i/8h: the paragraph-13 dated author access declaration."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    requirement: str = ""
    # Round 8h fills the English rendering confirmed by the author on
    # 2026-11-06; it is never invented by the registration.
    content: str = ""
    date: str = ""
    declaration_path: str = ""
    declaration_sha256: str = ""
    confirmed_by: str = ""


class PilotVerifierV3Spec(BaseModel):
    """Round 8f/8g: operative v3 verifier interpretation and recomputation."""

    model_config = ConfigDict(extra="forbid")

    addendum_path: str = ""
    addendum_title: str = ""
    lifecycle_addendum_path: str = ""
    lifecycle_addendum_title: str = ""
    status: str = ""
    interpretation: V3InterpretationSpec = Field(
        default_factory=V3InterpretationSpec
    )
    freeze: V3FreezeSpec = Field(default_factory=V3FreezeSpec)
    v2_recomputation: V2RecomputationSpec = Field(
        default_factory=V2RecomputationSpec
    )
    timeout_lifecycle: TimeoutLifecycleSpec = Field(
        default_factory=TimeoutLifecycleSpec
    )
    lifecycle_evidence: LifecycleEvidenceSpec = Field(
        default_factory=LifecycleEvidenceSpec
    )
    terminal_infrastructure: TerminalInfrastructureSpec = Field(
        default_factory=TerminalInfrastructureSpec
    )
    attempt_ledger: AttemptLedgerSpec = Field(
        default_factory=AttemptLedgerSpec
    )
    acceptance: AcceptanceGuardsSpec = Field(
        default_factory=AcceptanceGuardsSpec
    )
    consumer_audit: ConsumerAuditSpec = Field(
        default_factory=ConsumerAuditSpec
    )
    hashes: VersionHashDisclosureSpec = Field(
        default_factory=VersionHashDisclosureSpec
    )
    prerequisite_order: PrerequisiteOrderSpec = Field(
        default_factory=PrerequisiteOrderSpec
    )
    # Round 8i: the void v3 fact/disposition, the single v3r replacement and
    # the still-pending dated author access declaration.
    void_v3: V3VoidSpec = Field(default_factory=V3VoidSpec)
    replacement_v3r: V3ReplacementSpec = Field(
        default_factory=V3ReplacementSpec
    )
    author_access_declaration: AuthorAccessDeclarationSpec = Field(
        default_factory=AuthorAccessDeclarationSpec
    )
    superseded_history: list[str] = Field(default_factory=list)
    no_final_before_v3: str = ""

    @model_validator(mode="after")
    def _validate_registered(self) -> "PilotVerifierV3Spec":
        if not self.status:
            return self
        dump = self.model_dump(exclude={"status"})
        # Round 8g is additive: a round-8f document (for example the previous
        # committed HEAD) has status set but no lifecycle marker.  Keep it
        # loadable, and enforce the new fields only when the 8g marker exists.
        if not self.lifecycle_addendum_title:
            for key in (
                "lifecycle_addendum_path",
                "lifecycle_addendum_title",
                "lifecycle_evidence",
                "terminal_infrastructure",
                "attempt_ledger",
            ):
                dump.pop(key, None)
            timeout_lifecycle = dump.get("timeout_lifecycle")
            if isinstance(timeout_lifecycle, dict):
                timeout_lifecycle = dict(timeout_lifecycle)
                timeout_lifecycle.pop("outer_deadline_liveness", None)
                dump["timeout_lifecycle"] = timeout_lifecycle
            consumer_audit = dump.get("consumer_audit")
            if isinstance(consumer_audit, dict):
                consumer_audit = dict(consumer_audit)
                consumer_audit.pop("dispositions", None)
                dump["consumer_audit"] = consumer_audit
        # Round 8i is additive too: a pre-8i document has no void marker and
        # remains loadable; the new fields are enforced only when it is set.
        if not self.void_v3.status:
            for key in (
                "void_v3",
                "replacement_v3r",
                "author_access_declaration",
            ):
                dump.pop(key, None)
        else:
            declaration = dump.get("author_access_declaration")
            if isinstance(declaration, dict):
                declaration = dict(declaration)
                # The pre-8h document records only the pending status and the
                # requirement; these fields appear when the author confirms.
                if declaration.get("status") != "confirmed":
                    for key in (
                        "content",
                        "date",
                        "declaration_path",
                        "declaration_sha256",
                        "confirmed_by",
                    ):
                        declaration.pop(key, None)
                dump["author_access_declaration"] = declaration
        missing = _empty_registered_paths(dump)
        if missing:
            raise ValueError(
                "pilot.verifier_v3 fields must be registered: "
                + ", ".join(missing)
            )
        if self.freeze.original_code_requests != 468:
            raise ValueError(
                "pilot.verifier_v3.freeze.original_code_requests must be 468 "
                "(the complete original code inventory)"
            )
        if self.freeze.selected_requests != 1144:
            raise ValueError(
                "pilot.verifier_v3.freeze.selected_requests must be 1144 "
                "(the complete selected-request accounting)"
            )
        if abs(float(self.timeout_lifecycle.outer_timeout_s) - 6.0) > 1e-9:
            raise ValueError(
                "pilot.verifier_v3.timeout_lifecycle.outer_timeout_s must stay "
                "the registered 6-second outer timeout"
            )
        if self.timeout_lifecycle.retry_count != 1:
            raise ValueError(
                "pilot.verifier_v3.timeout_lifecycle.retry_count must be 1 "
                "(one precommitted identical-configuration retry)"
            )
        required_classes = {
            "startup_or_daemon_failure",
            "candidate_running_at_outer_deadline",
            "failure_after_candidate_completion",
            "unresolved_attribution",
        }
        if set(self.timeout_lifecycle.attribution_classes) != required_classes:
            raise ValueError(
                "pilot.verifier_v3.timeout_lifecycle.attribution_classes must "
                f"register exactly {sorted(required_classes)}"
            )
        if self.lifecycle_addendum_title:
            if "SUPERSEDED HISTORY" not in (
                self.timeout_lifecycle.blocked_rule
            ):
                raise ValueError(
                    "pilot.verifier_v3.timeout_lifecycle.blocked_rule must "
                    "mark the old blocked-count rule as SUPERSEDED HISTORY"
                )
            if "positive liveness evidence" not in (
                self.timeout_lifecycle.outer_deadline_liveness
            ):
                raise ValueError(
                    "pilot.verifier_v3.timeout_lifecycle."
                    "outer_deadline_liveness must require positive liveness "
                    "evidence"
                )
            lifecycle = self.lifecycle_evidence
            if "HMAC-SHA-256" not in lifecycle.hmac_sha256:
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_evidence.hmac_sha256 must "
                    "register actual HMAC-SHA-256"
                )
            if "fresh per-attempt key" not in lifecycle.per_attempt_key:
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_evidence.per_attempt_key "
                    "must register a fresh per-attempt key"
                )
            if "fails closed" not in lifecycle.fail_closed_rule.lower():
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_evidence.fail_closed_rule "
                    "must state that detected corruption fails closed"
                )
            if "not candidate-proof isolation" not in (
                lifecycle.not_candidate_proof_isolation.lower()
            ):
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_evidence."
                    "not_candidate_proof_isolation must reject the isolation "
                    "claim"
                )
            if "same-UID" not in lifecycle.residual_same_uid_risk:
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_evidence."
                    "residual_same_uid_risk must disclose the residual "
                    "same-UID risk"
                )
            terminal = self.terminal_infrastructure
            if terminal.count_status != (
                "complete_with_terminal_infrastructure"
            ):
                raise ValueError(
                    "pilot.verifier_v3.terminal_infrastructure.count_status "
                    "must be complete_with_terminal_infrastructure"
                )
            required_blockers = {
                "incomplete request/test inventories",
                "invalid attempt history",
                "freeze or configuration mismatches",
                "unverified termination/cleanup",
            }
            if set(terminal.remaining_blockers) != required_blockers:
                raise ValueError(
                    "pilot.verifier_v3.terminal_infrastructure."
                    "remaining_blockers must register exactly "
                    f"{sorted(required_blockers)}"
                )
            if "continuation" not in (
                self.attempt_ledger.not_second_freeze.lower()
            ):
                raise ValueError(
                    "pilot.verifier_v3.attempt_ledger.not_second_freeze must "
                    "register resume as continuation of the same freeze"
                )
            if "append-only" not in (
                self.attempt_ledger.durable_append_only.lower()
            ):
                raise ValueError(
                    "pilot.verifier_v3.attempt_ledger.durable_append_only "
                    "must register a durable append-only ledger"
                )
        expected_v1 = {
            "verifier_freeze.json": (
                "3118b6b0af149eda243d416bdbf98370437cd923a36c193b895c74bd5c8234e1"
            ),
            "count_manifest.json": (
                "17198e44be40690bc1e13b85d6c41bde8c181358390e10fd3246cf8dec89df67"
            ),
            "per_request.jsonl": (
                "de799a68be62e1aaec701a6afd79bee618584b1d0cc21606e9df9c6da4d33730"
            ),
        }
        expected_v2 = {
            "verifier_freeze.json": (
                "e53a7269d13988cfc090696ed0436c29bfa1c17fba5a443cc834917aa71dd39d"
            ),
            "count_manifest.json": (
                "ad1cd242931271fcfa1e88242730b87ecc5a4a2d217fce4b4420568901402c93"
            ),
            "per_request.jsonl": (
                "81818b0229455b7b02b229e1f0f8fc0d12dbf27f7927733518479ecaa973683d"
            ),
        }
        if self.hashes.v1 != expected_v1 or self.hashes.v2 != expected_v2:
            raise ValueError(
                "pilot.verifier_v3.hashes.v1/v2 must record the ruling's exact "
                "file-byte SHA-256 table"
            )
        if self.hashes.common_test_manifest_sha256 != (
            "3f9d56317ae8a27cab59aca8f6a401fa3254edd56fc44dce13d5a7ac7f5e7e2e"
        ):
            raise ValueError(
                "pilot.verifier_v3.hashes.common_test_manifest_sha256 must "
                "record the common v1/v2 test-manifest file hash"
            )
        if self.v2_recomputation.request_id != (
            "taco:5bc221f9dec75484cb706b33:2"
        ):
            raise ValueError(
                "pilot.verifier_v3.v2_recomputation.request_id must record "
                "taco:5bc221f9dec75484cb706b33:2"
            )
        if len(self.prerequisite_order.order) < 4:
            raise ValueError(
                "pilot.verifier_v3.prerequisite_order.order must list the "
                "ordered prerequisites"
            )
        if self.lifecycle_addendum_title:
            if self.lifecycle_addendum_path != (
                "prereg/pilot-shortfall-amendment.md"
            ):
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_addendum_path must be "
                    "prereg/pilot-shortfall-amendment.md"
                )
            if self.lifecycle_addendum_title != (
                "LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION AND "
                "V3 RESUME AMENDMENT"
            ):
                raise ValueError(
                    "pilot.verifier_v3.lifecycle_addendum_title must register "
                    "the round-8g lifecycle amendment title"
                )
        if self.void_v3.status:
            if self.void_v3.status != "VOID_VERIFIER_IMPLEMENTATION_FAILURE":
                raise ValueError(
                    "pilot.verifier_v3.void_v3.status must be "
                    "VOID_VERIFIER_IMPLEMENTATION_FAILURE"
                )
            if self.void_v3.created_at != (
                "2026-11-06T20:23:38.655567+00:00"
            ):
                raise ValueError(
                    "pilot.verifier_v3.void_v3.created_at must record "
                    "2026-11-06T20:23:38.655567+00:00"
                )
            if self.void_v3.commit != (
                "b5036ac4d7cd28d354aa55b065671df0444d9bdf"
            ):
                raise ValueError(
                    "pilot.verifier_v3.void_v3.commit must record "
                    "b5036ac4d7cd28d354aa55b065671df0444d9bdf"
                )
            if self.void_v3.embedded_freeze_manifest_sha256 != (
                "add7f6f49c17d41873be422bb280c1a8a8b315d83b0cc3f13091e815f34ed184"
            ):
                raise ValueError(
                    "pilot.verifier_v3.void_v3.embedded_freeze_manifest_sha256 "
                    "must record the void run's embedded freeze hash"
                )
            expected_void_files = {
                "v3_registry.json": (
                    "5e103b898ec9aac6a5b862cda4d607761c5bda815b701366625ab4582"
                    "3ce8a12"
                ),
                "verifier_v3/verifier_freeze.json": (
                    "3eed1f0be765e609e13f3f91460f3a15f1ff032cec1d1373c294477f"
                    "914e1ea0"
                ),
                "count_v3/count_manifest.json": (
                    "5b0cd11b9b612914de865eb06b6419bab3376f01eb07380ed6ffd2134"
                    "e854ea2"
                ),
                "count_v3/attempt_ledger.jsonl": (
                    "7d0e9fb47c6e73218f274cb8305590861bbbf763f33f1d9b43d09ad54"
                    "55fced1"
                ),
            }
            if self.void_v3.file_sha256 != expected_void_files:
                raise ValueError(
                    "pilot.verifier_v3.void_v3.file_sha256 must record the "
                    "ruling's exact four file-byte SHA-256 values"
                )
            if "non-operative" not in (
                self.void_v3.non_operative_extension.lower()
            ):
                raise ValueError(
                    "pilot.verifier_v3.void_v3.non_operative_extension must "
                    "state that the void count's extension_required is "
                    "non-operative"
                )
            if "solely" not in self.void_v3.supersedes_scope.lower():
                raise ValueError(
                    "pilot.verifier_v3.void_v3.supersedes_scope must limit the "
                    "supersession solely to the void run and replacement"
                )
            replacement = self.replacement_v3r
            if "v3r" not in replacement.identity:
                raise ValueError(
                    "pilot.verifier_v3.replacement_v3r.identity must name v3r"
                )
            if "attempt 1" not in replacement.retry_allowance:
                raise ValueError(
                    "pilot.verifier_v3.replacement_v3r.retry_allowance must "
                    "reset each v3r test to attempt 1"
                )
            if "no automatic v3r2" not in (
                replacement.no_automatic_v3r2.lower()
            ):
                raise ValueError(
                    "pilot.verifier_v3.replacement_v3r.no_automatic_v3r2 must "
                    "state that no automatic v3r2 follows"
                )
            declaration = self.author_access_declaration
            if declaration.status == "pending author confirmation":
                if declaration.content.strip():
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration.content "
                        "must remain empty while the declaration is pending"
                    )
                if any(
                    (
                        declaration.date,
                        declaration.declaration_path,
                        declaration.declaration_sha256,
                        declaration.confirmed_by,
                    )
                ):
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration metadata "
                        "must remain empty while the declaration is pending"
                    )
            elif declaration.status == "confirmed":
                if declaration.content.strip() != AUTHOR_DECLARATION_ENGLISH:
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration.content "
                        "must register the author-confirmed English rendering "
                        "verbatim"
                    )
                if declaration.date != "2026-11-06":
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration.date must "
                        "be 2026-11-06"
                    )
                if declaration.declaration_path != (
                    r"C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/"
                    r"2026-11-03-gemma-loop/"
                    r"author-declaration-2026-11-06.md"
                ):
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration."
                        "declaration_path must cite the dated author file"
                    )
                if declaration.declaration_sha256 != (
                    "aaae7fdd33dd5fff63a620c943e24d0fc27a9318d23f87e75405599f"
                    "863c16af"
                ):
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration."
                        "declaration_sha256 must bind the dated author file"
                    )
                if "2026-11-06" not in declaration.confirmed_by:
                    raise ValueError(
                        "pilot.verifier_v3.author_access_declaration."
                        "confirmed_by must record the 2026-11-06 confirmation"
                    )
            else:
                raise ValueError(
                    "pilot.verifier_v3.author_access_declaration.status must be "
                    "pending author confirmation or confirmed"
                )
        return self


class PilotSpec(BaseModel):
    """Pre-registered pilot (runs only after the tag, never on benchmarks)."""

    model_config = ConfigDict(extra="forbid")

    runs_after_tag: bool = True
    held_out_only: bool = True
    benchmarks: bool = False
    cost_gpu_hours: str = "35-40 local GPU-hours (honest estimate)"
    # Round 8: the plain-language "why" report is descriptive only.
    why_report: str = ""
    why_role: str = "descriptive diagnostics; never selection inputs"
    why_metrics: list[str] = Field(default_factory=list)
    # Round 8e: verified pilot targets require the frozen Docker verifier.
    # The legacy string form keeps pre-8e revision fixtures loadable; the
    # committed round-8e document uses the typed block.
    code_verification: PilotCodeVerificationSpec | str = Field(
        default_factory=PilotCodeVerificationSpec
    )
    # Round 8e: prospective shortfall rule (A, B and C) and staged freeze.
    shortfall: PilotShortfallSpec = Field(default_factory=PilotShortfallSpec)
    freeze: PilotShortfallFreezeSpec = Field(
        default_factory=PilotShortfallFreezeSpec
    )
    # Round 8f: the X1 verifier-interpretation and recomputation addendum.
    verifier_v3: PilotVerifierV3Spec = Field(
        default_factory=PilotVerifierV3Spec
    )
    p1: PilotP1Spec = Field(default_factory=PilotP1Spec)
    p2: PilotP2Spec = Field(default_factory=PilotP2Spec)
    p3: PilotP3Spec = Field(default_factory=PilotP3Spec)
    data: PilotDataSpec = Field(default_factory=PilotDataSpec)


class LoopLadderPointSpec(BaseModel):
    """One point of the descriptive loop ladder."""

    model_config = ConfigDict(extra="forbid")

    id: str
    arm: str
    n: int
    stage: str
    description: str = ""


class LoopLadderSpec(BaseModel):
    """Descriptive O -> R/R-damped -> L-heal -> L-final ladder.

    It runs after the tag, is never an input to block selection or to the
    confirmatory family, and supplies the post-Stage-1 measurements needed by
    the difference-in-differences.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = "descriptive/estimation only"
    runs_after_tag: bool = True
    selection_use: str = "never used for pilot or selection"
    chosen_block: str = "pilot-chosen block"
    r_rule: str = ""
    r_damped_rule: str = ""
    r_damped_weight: str = ""
    points: list[LoopLadderPointSpec] = Field(default_factory=list)
    items: str = ""
    full_suite_at: str = "L final N=4"
    oh_control: str = ""
    untrained_bridge_note: str = ""
    dial_nll_at_heal: str = ""
    reused_from_d2: str = ""
    gpu_hours: str = ""
    gpu_hours_with_reuse: str = ""
    reuse_rule: str = ""
    chart: str = ""
    did_note: str = ""


class P1LiteSpec(BaseModel):
    """Pre-tag P1-lite calibrated-NLL sweep on the verified-trace corpus."""

    model_config = ConfigDict(extra="forbid")

    corpus: str = ""
    problems: int = 120
    sweep: str = ""
    gpu_hours: str = ""


class TrainingFreeCampaignSpec(BaseModel):
    """Disclosure of the exploratory training-free campaign run before the tag."""

    model_config = ConfigDict(extra="forbid")

    before_tag: bool = True
    status: str = "exploratory"
    reported: str = "full"
    selection_input: bool = False
    frozen_candidates: str = ""
    frozen_at_commit: str = ""
    frozen_at_timestamp: str = ""
    no_output_at_freeze: bool = True
    p1_lite: P1LiteSpec = Field(default_factory=P1LiteSpec)
    accuracy_n2: str = ""
    then_n34: str = ""
    reuse_rule: str = ""
    stage_b_gate: str = ""
    corpus_disjointness: str = ""
    budget: str = ""
    chart: str = ""
    dial_requirement_disclosure: str = ""
    rule_change_disclosure: str = ""
    seen_at_rule_change: str = ""
    exposure_audit: str = ""
    decisions_with_n2_on_disk: str = ""
    stage_a_sealing_procedure: str = ""
    stage_a_sealing: list[dict[str, Any]] = Field(default_factory=list)
    # Round 7 (R2): the analysis rules were frozen before Stage A completed;
    # the outputs are hashed at completion BEFORE the author sees them.
    analysis_rules_frozen_at_commit: str = ""
    analysis_rules_frozen_at_timestamp: str = ""
    stage_a_visibility: str = ""
    stage_a_completion_addendum: str = ""
    # Round 8 (X1 item 5): correct the pre-tag viewer statement (the presenter
    # and later auditors opened Stage A after the seal) and state the freeze
    # sequence when the actual data/pilot/implementation freezes happen.
    post_seal_access_record: str = ""
    freeze_sequence: str = ""
    mutable_status_files: list[str] = Field(default_factory=list)


class DataPoolSpec(BaseModel):
    """Stage 0 sampling pool."""

    model_config = ConfigDict(extra="forbid")

    code_count: int = 2000
    math_count: int = 2000
    chat_count: int = 1000
    samples_per_prompt: int = 4
    generated_tokens: int = 64_000_000
    # Round 8b: the Part-1 pool mode (no creative briefs) and its frozen
    # source pins.
    mode: str = ""
    part1_creative_briefs: bool = True
    part1_mode: str = ""
    manifest_sha256: str | None = None
    source_pins: dict[str, str] = Field(default_factory=dict)
    code_sources: list[str] = Field(default_factory=list)
    math_sources: list[str] = Field(default_factory=list)
    chat_sources: list[str] = Field(default_factory=list)


class DataSplitsSpec(BaseModel):
    """One deterministic split table, disjoint by problem id."""

    model_config = ConfigDict(extra="forbid")

    split_rule: str = (
        "bucket = int(sha256(problem_id), 16) % 100, seed 0; buckets are "
        "disjoint by problem id"
    )
    train_bucket: str = "0-49"
    pilot_dev_bucket: str = "50-59"
    nll_kl_bucket: str = "60-69"
    post_tfit_bucket: str = "60-61"
    nll_test_bucket: str = "62-69"
    long_traces_bucket: str = "70-79"
    heldout_test_bucket: str = "80-99"
    pilot_p3_bucket: str = "50-53"
    pilot_selection_bucket: str = "54-56"
    pilot_tfit_bucket: str = "57-58"
    pilot_p1_bucket: str = "59"
    # Held-out MONITOR corpus (round 8b): the largest sha256(seed:problem_id)
    # keys inside the train bucket; every training subset takes the smallest
    # keys and excludes these problems.
    monitor_bucket: str = "0-49"
    monitor_order_rule: str = ""
    nominal_counts_role: str = ""
    realized_counts: dict[str, int] = Field(default_factory=dict)
    train_problems_role: str = ""
    shortfall_rules: str = ""
    monitor_problems: int = 64
    monitor_samples_per_prompt: int = 1
    stage0_prompt_bucket: str = "0-49"
    train_problems: int = 2500
    pilot_dev_problems: int = 500
    pilot_p3_problems: int = 200
    pilot_selection_problems: int = 150
    pilot_tfit_problems: int = 100
    pilot_p1_problems: int = 50
    nll_kl_problems: int = 500
    post_tfit_problems: int = 100
    nll_test_problems: int = 400
    long_traces: int = 500
    heldout_test_problems: int = 1000


class DataSpec(BaseModel):
    """Data sampling, decontamination and held-out splits."""

    model_config = ConfigDict(extra="forbid")

    sampling_model: str = "O"
    sampling: Sampling = Field(
        default_factory=lambda: Sampling(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_new_tokens=32768,
        )
    )
    pool: DataPoolSpec = Field(default_factory=DataPoolSpec)
    decontamination: list[str] = Field(default_factory=list)
    decontamination_targets: list[str] = Field(default_factory=list)
    manifest_sha256: str | None = None
    manifest_addendum: str = ""
    splits: DataSplitsSpec = Field(default_factory=DataSplitsSpec)
    max_length: int = 8192
    length_mode: str = "filter"
    stage0_sample_filter: str = ""
    reasoning_share_min: float = 0.75


class ComputeHoursRow(BaseModel):
    """One honest local-versus-cloud hours row, split by engine."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    local_hours: str = ""
    padded_hours: str = ""
    refill_hours: str = ""
    cloud_hours: str = ""
    note: str = ""


class ComputeSpec(BaseModel):
    """Compute-tier rule: availability only, recorded as a covariate."""

    model_config = ConfigDict(extra="forbid")

    tiers: list[str] = Field(default_factory=list)
    selection: str = "availability only; fixed before any result"
    recorded_as_covariate: bool = True
    same_tier_per_seed_pair: bool = True
    same_engine: bool = True
    local_engine: str = (
        "registered evaluation engine: refill engine (task/t5c) if the gate "
        "passes, otherwise the padded adaptive-batch HF engine; same for O, O-H "
        "and L"
    )
    # Round 8b: the refill track is closed and the padded engine is registered.
    refill_track_status: str = ""
    registered_eval_engine: str = ""
    stage0_lite_measured: str = ""
    stage0_lite_hours: str = ""
    modal_probe_history: str = ""
    modal_money_guards: str = ""
    modal_optional_part1: str = ""
    modal_registered_path: str = ""
    eval_engine_default: str = "padded adaptive-batch HF engine"
    refill_engine_branch: str = "task/t5c"
    refill_engine_commit: str = "a81282e"
    # Round 8: the historical gate-v5 candidate provenance stays in
    # ``refill_engine_commit``; the registered pin is the repaired v6 commit.
    repaired_engine_pin: str = ""
    engine_pin_rule: str = ""
    merged_feat_commit: str = ""
    eval_engine_gate: str = ""
    engine_gate_hours: str = ""
    part_b_hours: str = ""
    cloud_engine: str = "stock vLLM for Stage 0 data only; no plugin"
    stage0_provider: str = "Modal"
    stage0_vllm_version: str = "vllm==0.30.0"
    stage0_usage_limit_usd: float = 30.0
    stage0_spend_limit_usd: float = 0.0
    stage0_default: str = "Modal, stock vLLM v0.30.0 (data only)"
    stage0_fallback: str = "finish locally on the registered HF engine"
    stage0_engine_record: str = "engine recorded per row; split assignment unaffected"
    # Round 8: the local Stage-0-lite sampler for pilot data and the rule for
    # the full main-run Stage 0 tier.
    stage0_lite: str = ""
    stage0_main_tier_rule: str = ""
    stage0_cloud_cost_usd: str = "7-11"
    local_seed0_total_hours: str = "341-473 with ladder reuse; 350-482 worst case"
    eval_engine: str = (
        "current local padded adaptive-batch HF engine (generate.py, static "
        "cache); N=1 bit-exactness vs stock HF"
    )
    hours: list[ComputeHoursRow] = Field(default_factory=list)
    evaluation_priority: list[str] = Field(default_factory=list)
    evaluation_priority_note: str = ""


class PinsSpec(BaseModel):
    """Pins and freezes registered before the tag."""

    model_config = ConfigDict(extra="forbid")

    e2b_snapshot: str = "4adf15748de9e1a0097b6274cee9c15836c41c37"
    e2b_template_sha256: str | None = None
    teacher_e4b_revision: str | None = None
    teacher_12b_qat_revision: str | None = None
    tokenizer_sha256: dict[str, str] = Field(default_factory=dict)
    lcb_runner: str | None = None
    math_verify: str | None = None
    manifest_sha256: str | None = None
    data_addendum: str = ""
    t_dial_bench: str = ""
    bench_addendum: str = ""
    bench_artifact: str = ""
    bench_artifact_sha256: str = ""
    engine_addendum: str = ""
    merged_feat_commit: str = ""
    engine_candidate_commit: str = ""
    # Round 8: eed7fa7 is historical; the pin is the repaired v6 commit.
    engine_candidate_historical: str = ""
    repaired_smoke_addendum: str = ""
    teacher_runtime_addendum: str = ""
    # Round 8b: merged commits and the explicit PIN-PENDING gate.
    merged_commits: dict[str, str] = Field(default_factory=dict)
    task_healpilot_commit: str = ""
    task_p1corpus_commit: str = ""
    fix_modal_imports_commit: str = ""
    modal_image_commit: str = ""
    task_modallite_commit: str = ""
    padded_engine_source: str = ""
    temperature_fit_algorithm: str = ""
    pin_pending_rule: str = ""
    never_pin_commits: list[str] = Field(default_factory=list)
    addendum: str = ""

    def pending_pin_fields(self) -> list[str]:
        """Return the pin fields still holding the PIN-PENDING placeholder."""
        pending: list[str] = []
        for name in self.__class__.model_fields:
            value = getattr(self, name)
            if isinstance(value, str) and value.strip() == "PIN-PENDING":
                pending.append(name)
        return pending


class Part2Spec(BaseModel):
    """Exploratory Part 2: rules only, never part of Part 1 claims."""

    model_config = ConfigDict(extra="forbid")

    starting_checkpoint: str = "L-s0"
    also: str = "O-H-s0 where cheap"
    versioned_checkpoints: bool = True
    lineage: bool = True
    techniques: list[str] = Field(default_factory=list)
    serve_map: str = ""
    rules: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    dev_composite: str = ""
    milestone_scoring_cap: str = ""
    automatic_checks_note: str = ""
    decision_log_export: str = ""
    deepseek_terms: str = ""


class MdeRow(BaseModel):
    """One pre-computed minimum detectable effect."""

    model_config = ConfigDict(extra="forbid")

    bench: str
    n: int
    k: int = 1
    mde_pts: float
    alpha: float = 0.05
    note: str = ""
class EngineGateCombination(BaseModel):
    """One registered (arm, block, bridge, policy, N set) gate combination."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    block: str | None = None
    bridge: str | None = None
    adapter_policy: str | None = None
    n_loops: list[int]


class EngineGateSpec(BaseModel):
    """Registered refill-vs-padded gate coverage and settings."""

    model_config = ConfigDict(extra="forbid")

    prompts: str = "tests/fixtures/engine_gate_prompts.json"
    size: str = "e2b"
    quant: str = "bf16"
    logprob_mean_limit: float = 0.01
    logprob_max_limit: float = 0.1
    max_new_tokens: int = 256
    round_size: int = 64
    max_batch: int = 4
    refill_budgets: list[int] = Field(
        default_factory=lambda: [128, 128, 192, 192, 256, 256, 256, 256]
    )
    seed: int = 0
    token_limit: int = 256
    near_tie_logit_gap: float = 0.25
    min_exact_fraction: float = 0.90
    sampling_n: list[int] = Field(default_factory=lambda: [2])
    throughput_n: list[int] = Field(default_factory=lambda: [1, 2])
    comparison_prompts: int = 8
    long_prompts: int = 4
    long_max_new_tokens: int = 64
    long_round_size: int = 64
    long_round_1024_tokens: int = 1200
    # Round 6: the gate runs after the tag and before any trained-arm
    # evaluation; its re-prefill smoke step is logged but never counted.
    runs_after_tag: bool = True
    noise_floor_control: str = ""
    re_prefill_smoke: str = ""
    coverage_note: str = ""
    control_batch: int = 4
    one_n1_key_per_block: bool = True
    registered_reduction: str = ""
    # Round 7: the registered hours are one consistent set.  The gate includes
    # model loads; the re-prefill smoke is timed separately and never counted
    # toward the gate.
    gate_hours: str = ""
    smoke_hours: str = ""
    step_time_fallback: str = ""
    # Round 8 (X1 item 1 / X2): the honest record of the four pre-tag smoke
    # attempts and the frozen replacement eligibility rule (engine gate v6).
    pre_tag_smoke_attempts: list[str] = Field(default_factory=list)
    # Round 8c: the historical gate is INACTIVE for prereg-v1.
    status: str = ""
    pre_tag_smoke_artifact: str = ""
    pre_tag_smoke_artifact_sha256: str = ""
    pre_tag_smoke_outcome: str = ""
    stream_diagnostic: str = ""
    replacement_rule: str = ""
    calibration_partition: str = ""
    validation_partition: str = ""
    chosen_token_regret: str = ""
    sampled_decision_traces: str = ""
    boundary_certificate: str = ""
    exactness_rule: str = ""
    shared_steps_min: int = 0
    failure_policy: str = ""
    repair_cycle: str = ""
    engine_pin_rule: str = ""
    combinations: list[EngineGateCombination] = Field(default_factory=list)


class PartBSpec(BaseModel):
    """Registered Part B throughput settings (non-confirmatory pools only)."""

    model_config = ConfigDict(extra="forbid")

    p1lite_path: str
    stage0_pool_path: str | None = None
    requests: int = 128
    max_batch: int = 8
    max_tokens: list[int] = Field(default_factory=lambda: [8192, 16384])
    round_size: int = 1024
    min_speedup: float = 1.3
    seed: int = 0
    block: str = "16-18"
    bridge: str = "gch"
    loops: int = 2
    # Round 6: the registered run uses the evaluation batch on the 360
    # unfiltered DAPO candidates, with requests >= 3x slots.
    eval_batch: int = 24
    candidates: int = 360
    candidate_pool: str = ""
    chosen_prompts_sha256: str = ""
    estimated_gpu_hours: str = ""
    # Round 7: one consistent Part B estimate and the offline/hub-access rule.
    central_gpu_hours: str = ""
    # Round 8c: Part B is registered as INACTIVE for prereg-v1.
    status: str = ""
    offline_note: str = ""

    @model_validator(mode="after")
    def _registered_pool_and_sha(self) -> "PartBSpec":
        sha = (self.chosen_prompts_sha256 or "").strip().lower()
        if len(sha) != 64 or any(
            char not in "0123456789abcdef" for char in sha
        ):
            raise ValueError(
                "part_b.chosen_prompts_sha256 must be a non-empty 64-hex "
                "digest in registered mode"
            )
        if not self.offline_note:
            raise ValueError("part_b.offline_note must be registered")
        if not self.estimated_gpu_hours:
            raise ValueError("part_b.estimated_gpu_hours must be registered")
        if not self.central_gpu_hours:
            raise ValueError("part_b.central_gpu_hours must be registered")
        return self



class RevisionRound8Spec(BaseModel):
    """Round 8: one dated pre-tag amendment answering the external panel.

    X1 asks for the changes to be split into claim clarification, analysis
    precision, provenance correction, execution contingency and new
    eligibility design.  The historical pre-Stage-A rules snapshot stays
    ``8c6d78d``; the final ``prereg-v1`` commit is the authoritative complete
    specification.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8 (external review panel)"
    categories: dict[str, list[str]] = Field(default_factory=dict)
    historical_rules_snapshot: str = ""
    authoritative_spec: str = ""
    choices: list[str] = Field(default_factory=list)


class MonitorSpec(BaseModel):
    """Round 8b early-detection contract: observe-only execution safety.

    The monitor never feeds selection; it can only abort a run through the
    registered rules.  The values the trainer reads (the abort exit code, the
    reserved-id contract and the probe-failure rule) are validated here so a
    sealed file cannot silently weaken the safety net.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = ""
    corpus: str = ""
    split_rule: str = ""
    training_exclusion: str = ""
    teacher_shards: str = ""
    abort_rules: list[str] = Field(default_factory=list)
    warn_rules: list[str] = Field(default_factory=list)
    abort_exit_code: int = 76
    abort_terminal: str = ""
    resume_rule: str = ""
    probe_failure_rule: str = ""
    reserved_ids_contract: str = ""
    report_rule: str = ""
    repair_rule: str = ""
    probe_configuration: str = ""
    minimum_rows: int = 32
    missing_arm_rule: str = ""
    warn_effect: str = ""

    @model_validator(mode="after")
    def _validate_registered_rules(self) -> "MonitorSpec":
        if not self.abort_rules:
            raise ValueError("monitor.abort_rules must be registered")
        if not self.warn_rules:
            raise ValueError("monitor.warn_rules must be registered")
        if int(self.abort_exit_code) != 76:
            raise ValueError(
                "monitor.abort_exit_code must be 76 (the terminal heal-monitor "
                f"abort), got {self.abort_exit_code!r}"
            )
        if not self.reserved_ids_contract:
            raise ValueError("monitor.reserved_ids_contract must be registered")
        if not self.probe_failure_rule:
            raise ValueError("monitor.probe_failure_rule must be registered")
        return self


class GradingSpec(BaseModel):
    """Round 8b math/code grading rules (verified targets only)."""

    model_config = ConfigDict(extra="forbid")

    math_verify_version: str = ""
    math_verify_internal_timeouts: str = ""
    math_verify_outer_timeout_s: float = 10.0
    math_rule: str = ""
    math_why: str = ""
    decided_by: str = ""
    code_sandbox: str = ""
    code_rule: str = ""
    unsandboxed_rule: str = ""
    dependency: str = ""

    @model_validator(mode="after")
    def _validate_grading_rule(self) -> "GradingSpec":
        if float(self.math_verify_outer_timeout_s) <= 0:
            raise ValueError(
                "grading.math_verify_outer_timeout_s must be positive"
            )
        if not self.math_rule:
            raise ValueError("grading.math_rule must be registered")
        decided = str(self.decided_by).lower()
        missing = [
            label
            for label in ("math_verify", "normaliser", "both", "timeout")
            if label not in decided
        ]
        if missing:
            raise ValueError(
                "grading.decided_by must record every verifier outcome; "
                f"missing {missing}"
            )
        return self


class ExternalAPIsSpec(BaseModel):
    """Round 8b external-provider terms and pre-committed spend caps."""

    model_config = ConfigDict(extra="forbid")

    deepseek_api: str = ""
    deepseek_model: str = ""
    deepseek_served_version: str = ""
    deepseek_off_peak: str = ""
    deepseek_peak_windows: str = ""
    deepseek_margins: str = ""
    deepseek_paid_pilot_max_items: int = 20
    deepseek_writer_hard_cap_items: int = 50
    deepseek_waste_stop: str = ""
    deepseek_fingerprint: str = ""
    deepseek_phase_approval: str = ""
    deepseek_budgets: dict[str, str] = Field(default_factory=dict)
    deepseek_project_expected_usd: str = ""
    deepseek_caps_sum_usd: float = 0.0
    nvidia_nim: str = ""
    nvidia_nim_terms: str = ""
    opencode_endpoints: str = ""

    @model_validator(mode="after")
    def _validate_api_caps(self) -> "ExternalAPIsSpec":
        if not str(self.deepseek_model).startswith("deepseek-flash"):
            raise ValueError(
                "external_apis.deepseek_model must be deepseek-flash (never "
                f"deepseek-v4-pro), got {self.deepseek_model!r}"
            )
        if not 0 < int(self.deepseek_paid_pilot_max_items) <= 20:
            raise ValueError(
                "external_apis.deepseek_paid_pilot_max_items must be in 1..20"
            )
        if not self.deepseek_budgets:
            # The Prereg-level gate raises the registered "every phase" error;
            # an empty map must not look like a zero-dollar sum mismatch.
            return self
        total = 0.0
        for name, text in self.deepseek_budgets.items():
            match = re.search(r"\$\s*(\d+(?:\.\d+)?)", str(text))
            if match is None:
                raise ValueError(
                    f"external_apis.deepseek_budgets[{name!r}] has no $ cap"
                )
            total += float(match.group(1))
        if abs(total - float(self.deepseek_caps_sum_usd)) > 1e-6:
            raise ValueError(
                "external_apis.deepseek_caps_sum_usd must equal the sum of the "
                f"phase caps ({total:g}), got {self.deepseek_caps_sum_usd!r}"
            )
        return self


class RevisionRound8bItemSpec(BaseModel):
    """One item of the round-8b pre-tag technical amendment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    changes: str = ""
    where: list[str] = Field(default_factory=list)
    why: str = ""
    uses_outcome: bool = False
    outcome_basis: str = ""


class RevisionRound8bSpec(BaseModel):
    """Round 8b: the Stage-A-exposed pre-tag technical amendment.

    Stage A was exposed before this round.  Every item is therefore a pre-tag
    technical amendment and none may use a trained-arm outcome.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8b (pre-tag technical amendment)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    item_ids: list[str] = Field(default_factory=list)
    items: dict[str, RevisionRound8bItemSpec] = Field(default_factory=dict)
    choices: list[str] = Field(default_factory=list)


class RevisionRound8cItemSpec(BaseModel):
    """One item of the round-8c pre-tag technical amendment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    changes: str = ""
    where: list[str] = Field(default_factory=list)
    why: str = ""
    uses_outcome: bool = False
    outcome_basis: str = ""


class RevisionRound8cSpec(BaseModel):
    """Round 8c: the final pre-tag reconciliation and seal-readiness round."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8c (pre-tag technical amendment)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    item_ids: list[str] = Field(default_factory=list)
    items: dict[str, RevisionRound8cItemSpec] = Field(default_factory=dict)
    choices: list[str] = Field(default_factory=list)


class RevisionRound8dItemSpec(BaseModel):
    """One item of the round-8d pre-tag technical amendment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    changes: str = ""
    where: list[str] = Field(default_factory=list)
    why: str = ""
    uses_outcome: bool = False
    outcome_basis: str = ""


class RevisionRound8dSpec(BaseModel):
    """Round 8d: the final pre-tag alignment of code, pins and domain."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8d (pre-tag technical amendment)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    item_ids: list[str] = Field(default_factory=list)
    items: dict[str, RevisionRound8dItemSpec] = Field(default_factory=dict)
    choices: list[str] = Field(default_factory=list)


class RevisionRound8eItemSpec(BaseModel):
    """One item of the round-8e pilot-shortfall amendment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    changes: str = ""
    where: list[str] = Field(default_factory=list)
    why: str = ""
    uses_outcome: bool = False
    outcome_basis: str = ""


class RevisionRound8eSpec(BaseModel):
    """Round 8e: the prospective pilot shortfall amendment ruled by the panel."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8e (pilot shortfall amendment)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    item_ids: list[str] = Field(default_factory=list)
    items: dict[str, RevisionRound8eItemSpec] = Field(default_factory=dict)
    choices: list[str] = Field(default_factory=list)


class TeacherQualificationMetricsSpec(BaseModel):
    """Round 8h: the qualification metrics cited for one teacher runtime."""

    model_config = ConfigDict(extra="forbid")

    mean_full_kl: float | None = None
    mean_full_kl_upper95: float | None = None
    p95_full_kl: float | None = None
    top1_agreement: float | None = None
    top1_confident: float | None = None
    allocator_headroom_gib: float | None = None
    short_mean_full_kl: float | None = None
    short_p95_full_kl: float | None = None


class TeacherRuntimeOutcomeSpec(BaseModel):
    """One registered teacher-runtime outcome and its artifact binding."""

    model_config = ConfigDict(extra="forbid")

    teacher: str = ""
    runtime: str = ""
    status: str = ""
    qualification_path: str = ""
    qualification_sha256: str = ""
    metrics: TeacherQualificationMetricsSpec = Field(
        default_factory=TeacherQualificationMetricsSpec
    )
    decision: str = ""
    failure_detail: str = ""


class TeacherRuntimeAddendumSpec(BaseModel):
    """Round 8h: the realized teacher-runtime outcomes and fallback decision."""

    model_config = ConfigDict(extra="forbid")

    e4b_bf16: TeacherRuntimeOutcomeSpec = Field(
        default_factory=TeacherRuntimeOutcomeSpec
    )
    e4b_int8: TeacherRuntimeOutcomeSpec = Field(
        default_factory=TeacherRuntimeOutcomeSpec
    )
    b12b_nf4: TeacherRuntimeOutcomeSpec = Field(
        default_factory=TeacherRuntimeOutcomeSpec
    )
    b12b_q4_0_gguf: TeacherRuntimeOutcomeSpec = Field(
        default_factory=TeacherRuntimeOutcomeSpec
    )
    selected_teacher: str = ""
    qualified_teacher_count: int = 0
    excluded_teachers: list[str] = Field(default_factory=list)
    fallback_decision: str = ""
    author_decision: str = ""
    author_decision_date: str = ""
    exploratory_12b_arm: bool = False
    runtime_freeze_path: str = ""
    runtime_freeze_sha256: str = ""
    llama_cpp_tag: str = ""
    llama_cpp_commit: str = ""
    gguf_sha256: str = ""
    implementation_branch: str = ""
    implementation_commit: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _validate_registered(self) -> "TeacherRuntimeAddendumSpec":
        expected = {
            "e4b_bf16": (
                "FAIL",
                r"C:/Users/<user>/ML/gemmaloop-artifacts/qualification/"
                r"e4b-bf16/qualification.json",
                "a28f19331d941fcd3f73219d1d92844ec71c1bd65e0d0901506fe76931acfa06",
                0.1748,
            ),
            "e4b_int8": (
                "PASS",
                r"C:/Users/<user>/ML/gemmaloop-artifacts/qualification/"
                r"e4b-int8-r2/qualification.json",
                "4a4668248b27c0614260cc53350a6245d66c4db4147a0959797f7b2aa0a9dc75",
                3.1866,
            ),
            "b12b_nf4": (
                "FAIL",
                r"C:/Users/<user>/ML/gemmaloop-artifacts/qualification/"
                r"12b-nf4/qualification.json",
                "1d58b8342be2a44a0d7927e5c8186949b26af13967c49cd17f38fac4a128512f",
                0.4522,
            ),
            "b12b_q4_0_gguf": (
                "EXCLUDED",
                r"C:/Users/<user>/ML/gemmaloop-artifacts/qualification/"
                r"12b-q4_0-gguf/qualification.json",
                "dee39752d1cdbb641baa2bf1abdc646e3a0da8b84848481d242bbd6787705ea7",
                2.3549,
            ),
        }
        for name, (status, path, digest, headroom) in expected.items():
            outcome = getattr(self, name)
            if outcome.status != status:
                raise ValueError(
                    f"revision_round8h.data_addendum.teacher_runtime.{name}."
                    f"status must be {status}"
                )
            if outcome.qualification_path != path:
                raise ValueError(
                    f"revision_round8h.data_addendum.teacher_runtime.{name}."
                    "qualification_path must cite the registered artifact"
                )
            if outcome.qualification_sha256 != digest:
                raise ValueError(
                    f"revision_round8h.data_addendum.teacher_runtime.{name}."
                    "qualification_sha256 must bind the registered artifact"
                )
            if outcome.metrics.allocator_headroom_gib is None or abs(
                float(outcome.metrics.allocator_headroom_gib) - headroom
            ) > 1e-9:
                raise ValueError(
                    f"revision_round8h.data_addendum.teacher_runtime.{name}."
                    f"metrics.allocator_headroom_gib must be {headroom}"
                )
        if self.selected_teacher != "E4B int8":
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime.selected_teacher "
                "must be E4B int8"
            )
        if int(self.qualified_teacher_count) != 1:
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "qualified_teacher_count must be 1"
            )
        if self.exploratory_12b_arm:
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "exploratory_12b_arm must be False"
            )
        excluded = " ".join(self.excluded_teachers)
        if "12B NF4" not in excluded or "12B Q4_0 GGUF" not in excluded:
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "excluded_teachers must record both 12B runtimes"
            )
        if self.runtime_freeze_sha256 != (
            "d98574294230c8afc1c7b900a7906ad7736e53c667aab5891f51db35f7a5ecc9"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "runtime_freeze_sha256 must bind the frozen runtime record"
            )
        if self.llama_cpp_tag != "b12169":
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "llama_cpp_tag must be b12169"
            )
        if self.llama_cpp_commit != (
            "3ed7284f2d9e22b2d1a7d05c1416c00c5a9f0bc7"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "llama_cpp_commit must record the llama.cpp revision"
            )
        if self.gguf_sha256 != (
            "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime.gguf_sha256 "
                "must bind the official QAT Q4_0 GGUF"
            )
        if self.implementation_branch != "task/gguf12b":
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "implementation_branch must be task/gguf12b"
            )
        if self.implementation_commit != "5557d6c":
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "implementation_commit must be 5557d6c"
            )
        if "no qualified resident runtime" not in (
            self.fallback_decision.lower()
        ):
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "fallback_decision must record the 12B exclusion"
            )
        if self.author_decision_date != "2026-11-06":
            raise ValueError(
                "revision_round8h.data_addendum.teacher_runtime."
                "author_decision_date must be 2026-11-06"
            )
        return self


class ProbeMemoryPhaseSpec(BaseModel):
    """One memory phase of the GPU probe-memory smoke."""

    model_config = ConfigDict(extra="forbid")

    phase: str = ""
    status: str = ""
    peak_reserved_gib: float | None = None
    headroom_gib: float | None = None
    generation_rows: int | None = None
    expected_generation_rows: int | None = None
    wall_s: float | None = None


class ProbeMemoryAddendumSpec(BaseModel):
    """Round 8h: the audit-required GPU probe-memory smoke receipt."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    artifact_path: str = ""
    probe_json_sha256: str = ""
    probe_md_sha256: str = ""
    training_shape: str = ""
    training: ProbeMemoryPhaseSpec = Field(
        default_factory=ProbeMemoryPhaseSpec
    )
    n1: ProbeMemoryPhaseSpec = Field(default_factory=ProbeMemoryPhaseSpec)
    n4: ProbeMemoryPhaseSpec = Field(default_factory=ProbeMemoryPhaseSpec)
    n8_generation: ProbeMemoryPhaseSpec = Field(
        default_factory=ProbeMemoryPhaseSpec
    )
    disclosed_repairs: list[str] = Field(default_factory=list)
    residual_note: str = ""

    @model_validator(mode="after")
    def _validate_registered(self) -> "ProbeMemoryAddendumSpec":
        if self.status != "PASS":
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.status must be "
                "PASS"
            )
        if self.artifact_path != (
            r"C:/Users/<user>/ML/gemmaloop-artifacts/smoke/"
            r"probe_memory_auditfix1_r2"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.artifact_path "
                "must cite the registered smoke directory"
            )
        if self.probe_json_sha256 != (
            "94215907201be1439b7435bca554afe2dc6653d5749d9316c84cfec0a72b935d"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory."
                "probe_json_sha256 must bind probe_memory.json"
            )
        if self.probe_md_sha256 != (
            "ba95ecb94c52629aca23762ed8f1f84e3eb9390f12eb8800b9d94a867770d043"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory."
                "probe_md_sha256 must bind probe_memory.md"
            )
        if "training shape" not in self.training_shape.lower():
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.training_shape "
                "must register the training shape"
            )
        n8 = self.n8_generation
        if n8.status != "PASS" or n8.generation_rows != 24:
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.n8_generation "
                "must record the PASS with 24 generation rows"
            )
        if n8.expected_generation_rows != 24:
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.n8_generation."
                "expected_generation_rows must be 24"
            )
        if n8.peak_reserved_gib is None or abs(
            float(n8.peak_reserved_gib) - 6.791
        ) > 1e-9:
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.n8_generation."
                "peak_reserved_gib must be 6.791"
            )
        if n8.headroom_gib is None or abs(
            float(n8.headroom_gib) - 3.485
        ) > 1e-9:
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory.n8_generation."
                "headroom_gib must be 3.485"
            )
        joined = " | ".join(self.disclosed_repairs).lower()
        required = (
            "text-embedding device",
            "micro-batch 1",
            "token-weighted aggregation",
            "exit-1 bit-exactness",
            "single-pass",
            "nll temperatures",
            "offloaded towers temporarily on the compute device",
            "frozen teacher runtime enforced by the harness",
        )
        if not all(item in joined for item in required):
            raise ValueError(
                "revision_round8h.data_addendum.probe_memory."
                "disclosed_repairs must register every disclosed repair"
            )
        return self


class ConsumerAuditReferenceSpec(BaseModel):
    """Round 8h: the round-8g consumer-audit dispositions, referenced."""

    model_config = ConfigDict(extra="forbid")

    registered_in: str = ""
    audit_path: str = ""
    audit_sha256: str = ""
    dispositions_path: str = ""
    summary_rows: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _validate_registered(self) -> "ConsumerAuditReferenceSpec":
        if "round 8g" not in self.registered_in.lower():
            raise ValueError(
                "revision_round8h.data_addendum.consumer_audit."
                "registered_in must reference round 8g"
            )
        if self.audit_path != (
            r"C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/"
            r"2026-11-03-gemma-loop/audit-pool-test-consumers.md"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.consumer_audit.audit_path "
                "must cite the registered audit"
            )
        if self.audit_sha256 != (
            "f861612ac6634e86063ef5c8112f987364f3ac0c5cef5a568552ba0c4dcd7be6"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.consumer_audit.audit_sha256 "
                "must bind the audit"
            )
        if self.dispositions_path != (
            "pilot.verifier_v3.consumer_audit.dispositions"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.consumer_audit."
                "dispositions_path must reference the round-8g typed block"
            )
        if "1-16" not in self.summary_rows:
            raise ValueError(
                "revision_round8h.data_addendum.consumer_audit.summary_rows "
                "must record summary-table rows 1-16"
            )
        return self


class AuthorDeclarationReferenceSpec(BaseModel):
    """Round 8h: the dated author declaration required by X1 paragraph 13."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    date: str = ""
    declaration_path: str = ""
    declaration_sha256: str = ""
    registered_field: str = ""
    confirmed_by: str = ""

    @model_validator(mode="after")
    def _validate_registered(self) -> "AuthorDeclarationReferenceSpec":
        if self.status != "confirmed":
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration.status "
                "must be confirmed"
            )
        if self.date != "2026-11-06":
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration.date must "
                "be 2026-11-06"
            )
        if self.declaration_path != (
            r"C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/"
            r"2026-11-03-gemma-loop/author-declaration-2026-11-06.md"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration."
                "declaration_path must cite the author file"
            )
        if self.declaration_sha256 != (
            "aaae7fdd33dd5fff63a620c943e24d0fc27a9318d23f87e75405599f863c16af"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration."
                "declaration_sha256 must bind the author file"
            )
        if self.registered_field != (
            "pilot.verifier_v3.author_access_declaration"
        ):
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration."
                "registered_field must be the round-8i field"
            )
        if "2026-11-06" not in self.confirmed_by:
            raise ValueError(
                "revision_round8h.data_addendum.author_declaration."
                "confirmed_by must record the confirmation date"
            )
        return self


class ExploratoryScalingCurveSpec(BaseModel):
    """Round 8h: the pre-specified Part 2 exploratory scaling-curve plan."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    part: str = ""
    basis: str = ""
    budgets_tokens: list[int] = Field(default_factory=list)
    budget_12m_status: str = ""
    schedule: str = ""
    branch_rule: str = ""
    evaluation: str = ""
    reporting: str = ""
    data_source_and_budget: str = ""
    money_commitment: bool = True
    decided_by: str = ""
    decided_on: str = ""
    before_any_pilot_result: bool = False

    @model_validator(mode="after")
    def _validate_registered(self) -> "ExploratoryScalingCurveSpec":
        if "exploratory" not in self.part.lower():
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve.part "
                "must be exploratory"
            )
        if "best pilot dial arm" not in self.basis.lower():
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "basis must name the best pilot dial arm"
            )
        if "block-matched o-h" not in self.basis.lower():
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "basis must name the block-matched O-H"
            )
        if list(self.budgets_tokens) != [12_000_000, 100_000_000, 300_000_000]:
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "budgets_tokens must be [12M, 100M, 300M]"
            )
        if "warmup-stable-decay" not in self.schedule.lower():
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "schedule must be warmup-stable-decay"
            )
        if "decay branch of the 300M run" not in self.branch_rule:
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "branch_rule must make the 100M point a decay branch of the "
                "300M run"
            )
        if "same evaluation as the confirmatory part" not in (
            self.evaluation.lower()
        ):
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "evaluation must use the confirmatory evaluation"
            )
        if "full curve" not in self.reporting.lower():
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "reporting must register the full curve"
            )
        if "decided later and disclosed" not in (
            self.data_source_and_budget.lower()
        ):
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "data_source_and_budget must defer the decision and disclose it"
            )
        if self.money_commitment:
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "money_commitment must be False"
            )
        if self.decided_by != "author":
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "decided_by must be author"
            )
        if self.decided_on != "2026-11-06":
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "decided_on must be 2026-11-06"
            )
        if not self.before_any_pilot_result:
            raise ValueError(
                "revision_round8h.data_addendum.exploratory_scaling_curve."
                "before_any_pilot_result must be True"
            )
        return self


class V3rDomainDistinctCountsSpec(BaseModel):
    """Distinct verified problems by domain for one v3r split."""

    model_config = ConfigDict(extra="forbid")

    math: int | None = None
    code: int | None = None
    total: int | None = None


class V3rDistinctCountsSpec(BaseModel):
    """The v3r T-fit, selection and P1 distinct counts by domain."""

    model_config = ConfigDict(extra="forbid")

    t_fit: V3rDomainDistinctCountsSpec = Field(
        default_factory=V3rDomainDistinctCountsSpec
    )
    selection: V3rDomainDistinctCountsSpec = Field(
        default_factory=V3rDomainDistinctCountsSpec
    )
    p1: V3rDomainDistinctCountsSpec = Field(
        default_factory=V3rDomainDistinctCountsSpec
    )


class V3rCorpusSplitCountsSpec(BaseModel):
    """Row, problem and token counts for one v3r corpus split."""

    model_config = ConfigDict(extra="forbid")

    rows: int | None = None
    problems: int | None = None
    input_tokens: int | None = None
    scored_tokens: int | None = None


class V3rCorpusCountsSpec(BaseModel):
    """The final v3r corpus row/problem/token counts."""

    model_config = ConfigDict(extra="forbid")

    rows: int | None = None
    problems: int | None = None
    input_tokens: int | None = None
    scored_tokens: int | None = None
    p1: V3rCorpusSplitCountsSpec = Field(
        default_factory=V3rCorpusSplitCountsSpec
    )
    t_fit: V3rCorpusSplitCountsSpec = Field(
        default_factory=V3rCorpusSplitCountsSpec
    )
    selection: V3rCorpusSplitCountsSpec = Field(
        default_factory=V3rCorpusSplitCountsSpec
    )


class V3rVersionDeltaSpec(BaseModel):
    """One version-to-version delta in the v1/v2/v3/v3r chain."""

    model_config = ConfigDict(extra="forbid")

    from_version: str = ""
    to_version: str = ""
    row_delta: int | None = None
    problem_delta: int | None = None
    token_delta: int | None = None
    membership_changes: int | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _validate_partial(self) -> "V3rVersionDeltaSpec":
        values = (
            self.row_delta,
            self.problem_delta,
            self.token_delta,
            self.membership_changes,
        )
        if any(value is not None for value in values):
            if not self.from_version or not self.to_version:
                raise ValueError(
                    "v3r delta must name its from_version and to_version"
                )
            if any(value is None for value in values):
                raise ValueError(
                    "v3r delta must register every delta value"
                )
        return self


class V3rDeltasSpec(BaseModel):
    """The v1/v2/v3/v3r version deltas."""

    model_config = ConfigDict(extra="forbid")

    v1_to_v2: V3rVersionDeltaSpec | None = None
    v2_to_v3: V3rVersionDeltaSpec | None = None
    v3_to_v3r: V3rVersionDeltaSpec | None = None


class V3rTerminalCaseSpec(BaseModel):
    """One terminal infrastructure-unverified v3r test case."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = ""
    test_index: int | None = None
    split: str = ""
    domain: str = ""
    attempts: int | None = None
    reason: str = ""
    completion_status: str = ""


class V3rTerminalInfrastructureSpec(BaseModel):
    """The v3r terminal-infrastructure cases and count status."""

    model_config = ConfigDict(extra="forbid")

    status: str = "awaiting_v3r"
    count_status: str | None = None
    count: int | None = None
    inventory_complete: bool = False
    cases: list[V3rTerminalCaseSpec] = Field(default_factory=list)


class V3rExtensionDecisionSpec(BaseModel):
    """The v3r-derived extension decision."""

    model_config = ConfigDict(extra="forbid")

    status: str = "awaiting_v3r"
    rule: str = ""
    required: bool | None = None
    additional_requests: int | None = None
    basis: str = ""


class V3rHashesSpec(BaseModel):
    """The v3r registry, freeze, count, corpus and freeze-index hashes."""

    model_config = ConfigDict(extra="forbid")

    registry_sha256: str | None = None
    verifier_freeze_sha256: str | None = None
    count_manifest_sha256: str | None = None
    attempt_ledger_sha256: str | None = None
    corpus_sha256: dict[str, str] = Field(default_factory=dict)
    freeze_index_sha256: str | None = None


V3R_REQUIRED_FIELDS: tuple[str, ...] = (
    "hashes.registry_sha256",
    "hashes.verifier_freeze_sha256",
    "hashes.count_manifest_sha256",
    "hashes.attempt_ledger_sha256",
    "hashes.corpus_sha256",
    "hashes.freeze_index_sha256",
    "distinct_counts.t_fit.math",
    "distinct_counts.t_fit.code",
    "distinct_counts.t_fit.total",
    "distinct_counts.selection.math",
    "distinct_counts.selection.code",
    "distinct_counts.selection.total",
    "distinct_counts.p1.math",
    "distinct_counts.p1.code",
    "distinct_counts.p1.total",
    "extension_decision.status",
    "extension_decision.required",
    "extension_decision.additional_requests",
    "extension_decision.basis",
    "terminal_infrastructure.status",
    "terminal_infrastructure.count_status",
    "terminal_infrastructure.count",
    "terminal_infrastructure.inventory_complete",
    "deltas.v1_to_v2",
    "deltas.v2_to_v3",
    "deltas.v3_to_v3r",
    "corpus_counts.rows",
    "corpus_counts.problems",
    "corpus_counts.input_tokens",
    "corpus_counts.scored_tokens",
    "corpus_counts.p1.rows",
    "corpus_counts.p1.problems",
    "corpus_counts.p1.input_tokens",
    "corpus_counts.p1.scored_tokens",
    "corpus_counts.t_fit.rows",
    "corpus_counts.t_fit.problems",
    "corpus_counts.t_fit.input_tokens",
    "corpus_counts.t_fit.scored_tokens",
    "corpus_counts.selection.rows",
    "corpus_counts.selection.problems",
    "corpus_counts.selection.input_tokens",
    "corpus_counts.selection.scored_tokens",
)


class V3rDataAddendumSpec(BaseModel):
    """Round 8h: the typed v3r result section, awaiting the official run.

    Every result field is present and typed but empty while ``status`` is
    ``awaiting_v3r``.  The single official v3r run fills them and flips the
    status to ``complete``; the tag gate refuses the tag until then.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = "awaiting_v3r"
    addendum_path: str = "prereg/pilot-shortfall-freeze.json"
    authority: str = ""
    fill_rule: str = ""
    remaining_fields: list[str] = Field(default_factory=list)
    hashes: V3rHashesSpec = Field(default_factory=V3rHashesSpec)
    distinct_counts: V3rDistinctCountsSpec = Field(
        default_factory=V3rDistinctCountsSpec
    )
    extension_decision: V3rExtensionDecisionSpec = Field(
        default_factory=V3rExtensionDecisionSpec
    )
    terminal_infrastructure: V3rTerminalInfrastructureSpec = Field(
        default_factory=V3rTerminalInfrastructureSpec
    )
    deltas: V3rDeltasSpec = Field(default_factory=V3rDeltasSpec)
    corpus_counts: V3rCorpusCountsSpec = Field(
        default_factory=V3rCorpusCountsSpec
    )

    def _filled(self, path: str) -> bool:
        value: Any = self
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return False
        if path == "hashes.corpus_sha256":
            return bool(value)
        if path == "terminal_infrastructure.status":
            return value == "complete"
        if path == "extension_decision.status":
            return value == "complete"
        if path == "terminal_infrastructure.inventory_complete":
            return value is True
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _missing(self) -> list[str]:
        return [path for path in V3R_REQUIRED_FIELDS if not self._filled(path)]

    @model_validator(mode="after")
    def _validate_status(self) -> "V3rDataAddendumSpec":
        if self.status not in {"awaiting_v3r", "complete"}:
            raise ValueError(
                "revision_round8h.data_addendum.v3r.status must be "
                "awaiting_v3r or complete"
            )
        missing = self._missing()
        if self.status == "awaiting_v3r":
            if set(self.remaining_fields) != set(V3R_REQUIRED_FIELDS):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.remaining_fields must "
                    "list exactly the unfilled v3r result fields"
                )
            filled = [path for path in V3R_REQUIRED_FIELDS if path not in missing]
            if filled:
                raise ValueError(
                    "revision_round8h.data_addendum.v3r must keep every result "
                    "field empty while awaiting_v3r; filled: "
                    + ", ".join(filled)
                )
            return self
        if missing:
            raise ValueError(
                "revision_round8h.data_addendum.v3r is complete but still "
                "missing: " + ", ".join(missing)
            )
        if self.remaining_fields:
            raise ValueError(
                "revision_round8h.data_addendum.v3r.remaining_fields must be "
                "empty when complete"
            )
        for name in (
            "registry_sha256",
            "verifier_freeze_sha256",
            "count_manifest_sha256",
            "attempt_ledger_sha256",
            "freeze_index_sha256",
        ):
            value = getattr(self.hashes, name)
            if value is not None and not _SHA256_HEX.match(value):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.hashes."
                    f"{name} must be a lowercase 64-hex SHA-256"
                )
        for name, value in self.hashes.corpus_sha256.items():
            if not _SHA256_HEX.match(value):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.hashes."
                    f"corpus_sha256[{name!r}] must be lowercase 64-hex"
                )
        for split_name in ("t_fit", "selection", "p1"):
            counts = getattr(self.distinct_counts, split_name)
            if counts.total != int(counts.math) + int(counts.code):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.distinct_counts."
                    f"{split_name}.total must equal math + code"
                )
        if self.extension_decision.required is False:
            if int(self.extension_decision.additional_requests) != 0:
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.extension_decision."
                    "additional_requests must be 0 when no extension is needed"
                )
        elif int(self.extension_decision.additional_requests) <= 0:
            raise ValueError(
                "revision_round8h.data_addendum.v3r.extension_decision."
                "additional_requests must be positive when extension is "
                "required"
            )
        if self.terminal_infrastructure.count_status not in {
            "complete",
            "complete_with_terminal_infrastructure",
        }:
            raise ValueError(
                "revision_round8h.data_addendum.v3r.terminal_infrastructure."
                "count_status must be complete or "
                "complete_with_terminal_infrastructure"
            )
        for name in ("v1_to_v2", "v2_to_v3", "v3_to_v3r"):
            delta = getattr(self.deltas, name)
            if delta is None:
                continue
            if not delta.from_version or not delta.to_version:
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.deltas."
                    f"{name} must name its versions"
                )
            if any(
                getattr(delta, field_name) is None
                for field_name in (
                    "row_delta",
                    "problem_delta",
                    "token_delta",
                    "membership_changes",
                )
            ):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.deltas."
                    f"{name} must register every delta value"
                )
        for field_name in ("rows", "problems", "input_tokens", "scored_tokens"):
            total = getattr(self.corpus_counts, field_name)
            split_sum = sum(
                getattr(getattr(self.corpus_counts, split_name), field_name)
                for split_name in ("p1", "t_fit", "selection")
            )
            if int(total) != int(split_sum):
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.corpus_counts."
                    f"{field_name} must equal the split sum"
                )
        return self


class DataAddendumSpec(BaseModel):
    """Round 8h: everything registered now except the v3r count results."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    date: str = ""
    teacher_runtime: TeacherRuntimeAddendumSpec = Field(
        default_factory=TeacherRuntimeAddendumSpec
    )
    probe_memory: ProbeMemoryAddendumSpec = Field(
        default_factory=ProbeMemoryAddendumSpec
    )
    consumer_audit: ConsumerAuditReferenceSpec = Field(
        default_factory=ConsumerAuditReferenceSpec
    )
    author_declaration: AuthorDeclarationReferenceSpec = Field(
        default_factory=AuthorDeclarationReferenceSpec
    )
    exploratory_scaling_curve: ExploratoryScalingCurveSpec = Field(
        default_factory=ExploratoryScalingCurveSpec
    )
    v3r: V3rDataAddendumSpec = Field(default_factory=V3rDataAddendumSpec)

    @model_validator(mode="after")
    def _validate_registered(self) -> "DataAddendumSpec":
        if self.status != "complete" and "awaiting_v3r" not in self.status:
            raise ValueError(
                "revision_round8h.data_addendum.status must be complete or "
                "record the awaiting_v3r state"
            )
        if self.date != "2026-11-06":
            raise ValueError(
                "revision_round8h.data_addendum.date must be 2026-11-06"
            )
        return self


class RevisionRound8hSpec(BaseModel):
    """Round 8h: the data addendum, with the v3r results still pending."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8h (data addendum)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    choices: list[str] = Field(default_factory=list)
    data_addendum: DataAddendumSpec = Field(default_factory=DataAddendumSpec)

    @model_validator(mode="after")
    def _validate_registered(self) -> "RevisionRound8hSpec":
        if self.date != "2026-11-06":
            raise ValueError("revision_round8h.date must be 2026-11-06")
        exposure = self.stage_a_exposure.lower()
        if "stage a" not in exposure or "exposed" not in exposure:
            raise ValueError(
                "revision_round8h.stage_a_exposure must state that Stage A "
                "was exposed"
            )
        outcome_use = self.outcome_use.lower()
        if "no trained-arm outcome" not in outcome_use:
            raise ValueError(
                "revision_round8h.outcome_use must state that no trained-arm "
                "outcome informed the round"
            )
        if not self.choices:
            raise ValueError("revision_round8h.choices must not be empty")
        return self


class PilotHardwarePlatformSpec(BaseModel):
    """Round 8j: one pilot-training hardware platform."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    hardware: str = ""
    memory_gb: int | None = None
    role: str = ""


class PilotHardwareGroupSpec(BaseModel):
    """Round 8j: one block group and the single platform it runs on."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    block: str = ""
    arms: list[str] = Field(default_factory=list)
    platform: str = ""
    default_platform: str = ""
    may_move_to_modal: bool = False
    move_condition: str = ""
    together_with: str = ""


class PilotHardwareEvidenceSpec(BaseModel):
    """Round 8j: one piece of resource evidence behind the hardware plan."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    kind: str = ""
    report_path: str = ""
    findings: str = ""


class PilotHardwareProvenanceSpec(BaseModel):
    """Round 8j: the synthetic-only resource evidence behind the plan.

    The hardware plan is an operational amendment informed by resource
    evidence, not by a trained pilot outcome.  Every cited measurement was
    made on synthetic data before the plan was fixed.
    """

    model_config = ConfigDict(extra="forbid")

    corrected_on: str = ""
    prior_claim_corrected: str = ""
    synthetic_data_only: bool = False
    pilot_data_used: bool = False
    pilot_score_used: bool = False
    trained_pilot_outcome_used: bool = False
    informed_plan_before_decision: bool = False
    evidence: list[PilotHardwareEvidenceSpec] = Field(default_factory=list)


class PilotHardwareMoneyGuardSpec(BaseModel):
    """Round 8j: the pilot-training Modal money guard."""

    model_config = ConfigDict(extra="forbid")

    remaining_free_credits_usd: float = 0.0
    internal_cap_usd: float = 0.0
    cap_below_remaining_credits: bool = False
    monthly_ledger: str = ""
    shared_spend_lock: str = ""
    workspace_limit_hard_stop: bool = False
    cap_rule: str = ""
    group_rerun_rule: str = ""


class PilotTrainingHardwareSpec(BaseModel):
    """Round 8j: the registered pilot-training hardware plan."""

    model_config = ConfigDict(extra="forbid")

    status: str = ""
    date: str = ""
    decided_by: str = ""
    phase: str = ""
    before_any_pilot_result: bool = False
    scope: str = ""
    platforms: list[PilotHardwarePlatformSpec] = Field(default_factory=list)
    groups: list[PilotHardwareGroupSpec] = Field(default_factory=list)
    assignment_unit: str = ""
    assignment_by_block_group: bool = False
    assignment_never_by_individual_arm: bool = False
    assignment_uses_only_cost_credits_availability: bool = False
    assignment_decided_before_group_start: bool = False
    assignment_never_uses_observed_result: bool = False
    assignment_rule: str = ""
    assignment_basis: str = ""
    shared_platform_rule: str = ""
    run_identity_fields: list[str] = Field(default_factory=list)
    run_identity_rule: str = ""
    discloses_every_arm_platform: bool = False
    report_disclosure: str = ""
    cross_block_comparison: str = ""
    limitation: str = ""
    provenance: PilotHardwareProvenanceSpec = Field(
        default_factory=PilotHardwareProvenanceSpec
    )
    money_guard: PilotHardwareMoneyGuardSpec = Field(
        default_factory=PilotHardwareMoneyGuardSpec
    )
    modal_unavailable_runs_all_local: bool = False
    modal_unavailable_fallback: str = ""
    decision_log: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_registered(self) -> "PilotTrainingHardwareSpec":
        if self.status not in {"registered", "complete"}:
            raise ValueError(
                "revision_round8j.hardware_plan.status must be registered or "
                "complete"
            )
        if self.date != "2026-11-07":
            raise ValueError(
                "revision_round8j.hardware_plan.date must be 2026-11-07"
            )
        if self.decided_by != "author":
            raise ValueError(
                "revision_round8j.hardware_plan.decided_by must be the author"
            )
        if not self.before_any_pilot_result:
            raise ValueError(
                "revision_round8j.hardware_plan.before_any_pilot_result must "
                "be true"
            )
        phase = self.phase.lower()
        if "before any pilot training" not in phase or "scoring" not in phase:
            raise ValueError(
                "revision_round8j.hardware_plan.phase must state that the "
                "plan precedes any pilot training or scoring"
            )
        for field_name in (
            "assignment_by_block_group",
            "assignment_never_by_individual_arm",
            "assignment_uses_only_cost_credits_availability",
            "assignment_decided_before_group_start",
            "assignment_never_uses_observed_result",
            "discloses_every_arm_platform",
            "modal_unavailable_runs_all_local",
        ):
            if not getattr(self, field_name):
                raise ValueError(
                    f"revision_round8j.hardware_plan.{field_name} must be true"
                )
        if self.assignment_unit != "block group":
            raise ValueError(
                "revision_round8j.hardware_plan.assignment_unit must be "
                "block group"
            )
        expected_platforms = {
            "local": ("RTX 5070 12 GB", 12),
            "modal": ("NVIDIA H100 SXM5 80GB", 80),
        }
        platforms: dict[str, PilotHardwarePlatformSpec] = {}
        for platform in self.platforms:
            if not platform.id:
                raise ValueError(
                    "revision_round8j.hardware_plan.platforms need an id"
                )
            if platform.id in platforms:
                raise ValueError(
                    "revision_round8j.hardware_plan.platforms must be unique"
                )
            platforms[platform.id] = platform
        if set(platforms) != set(expected_platforms):
            raise ValueError(
                "revision_round8j.hardware_plan.platforms must register "
                "local and modal"
            )
        for platform_id, (hardware, memory_gb) in expected_platforms.items():
            platform = platforms[platform_id]
            if platform.hardware != hardware:
                raise ValueError(
                    "revision_round8j.hardware_plan.platforms."
                    f"{platform_id}.hardware must be {hardware}"
                )
            if platform.memory_gb != memory_gb:
                raise ValueError(
                    "revision_round8j.hardware_plan.platforms."
                    f"{platform_id}.memory_gb must be {memory_gb}"
                )
            if not platform.role.strip():
                raise ValueError(
                    "revision_round8j.hardware_plan.platforms."
                    f"{platform_id}.role must be registered"
                )

        group_specs = {
            "A": (
                "15-29",
                ("L-15-29-gch-e4b", "L-15-29-scalar-e4b", "OH-15-29-e4b"),
                "modal",
            ),
            "B": (
                "16-18",
                ("L-16-18-gch-e4b", "OH-16-18-e4b"),
                "local",
            ),
            "C": (
                "25-29",
                ("L-25-29-gch-e4b", "OH-25-29-e4b"),
                "local",
            ),
            "D": (
                "25-29",
                (
                    "L-25-29-gch-e4b-single-exit-0901",
                    "L-25-29-gch-e4b-single-exit-n4",
                ),
                "local",
            ),
        }
        groups: dict[str, PilotHardwareGroupSpec] = {}
        for group in self.groups:
            if not group.id:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups need an id"
                )
            if group.id in groups:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups must be unique"
                )
            groups[group.id] = group
        if set(groups) != set(group_specs):
            raise ValueError(
                "revision_round8j.hardware_plan.groups must key exactly "
                "A, B, C and D"
            )
        seen_arms: set[str] = set()
        for group_id, (block, arms, platform_id) in group_specs.items():
            group = groups[group_id]
            if group.block != block:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups."
                    f"{group_id}.block must be {block}"
                )
            if set(group.arms) != set(arms) or len(group.arms) != len(arms):
                raise ValueError(
                    "revision_round8j.hardware_plan.groups."
                    f"{group_id}.arms must be exactly {', '.join(arms)}"
                )
            if group.platform != platform_id:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups."
                    f"{group_id}.platform must be {platform_id}"
                )
            overlap = seen_arms & set(group.arms)
            if overlap:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups must not assign an "
                    "arm twice: " + ", ".join(sorted(overlap))
                )
            seen_arms.update(group.arms)
        group_b = groups["B"]
        if not group_b.may_move_to_modal:
            raise ValueError(
                "revision_round8j.hardware_plan.groups.B.may_move_to_modal "
                "must be true"
            )
        move = group_b.move_condition.lower()
        if "credit" not in move or "started anywhere" not in move:
            raise ValueError(
                "revision_round8j.hardware_plan.groups.B.move_condition must "
                "require remaining Modal credits and an unstarted group"
            )
        for group_id in ("A", "C", "D"):
            if groups[group_id].may_move_to_modal:
                raise ValueError(
                    "revision_round8j.hardware_plan.groups."
                    f"{group_id}.may_move_to_modal must be false"
                )
        if (
            groups["C"].together_with != "D"
            or groups["D"].together_with != "C"
        ):
            raise ValueError(
                "revision_round8j.hardware_plan groups C and D must run "
                "together"
            )
        # Every L arm and its block-matched O-H control share one platform.
        block_controls = {
            "15-29": "OH-15-29-e4b",
            "16-18": "OH-16-18-e4b",
            "25-29": "OH-25-29-e4b",
        }
        arm_platform: dict[str, str] = {}
        for group in self.groups:
            for arm in group.arms:
                arm_platform[arm] = group.platform
        for arm, platform_id in arm_platform.items():
            if not arm.startswith("L-"):
                continue
            for block, control in block_controls.items():
                if f"-{block}-" in arm:
                    control_platform = arm_platform.get(control)
                    if control_platform is None or control_platform != platform_id:
                        raise ValueError(
                            "revision_round8j.hardware_plan: every L arm and "
                            "its block-matched O-H control must share one "
                            "platform: " + arm
                        )
                    break
        if list(self.run_identity_fields) != [
            "hardware",
            "image",
            "library_versions",
            "allocator_cap",
        ]:
            raise ValueError(
                "revision_round8j.hardware_plan.run_identity_fields must be "
                "hardware, image, library_versions and allocator_cap"
            )
        if "within-platform" not in self.cross_block_comparison.lower():
            raise ValueError(
                "revision_round8j.hardware_plan.cross_block_comparison must "
                "compare within-platform L-minus-O-H gains"
            )
        if "bf16 kernels differ across gpus" not in self.limitation.lower():
            raise ValueError(
                "revision_round8j.hardware_plan.limitation must disclose "
                "that bf16 kernels differ across GPUs"
            )
        provenance = self.provenance
        if provenance.corrected_on != "2026-11-07":
            raise ValueError(
                "revision_round8j.hardware_plan.provenance.corrected_on must "
                "be 2026-11-07"
            )
        prior = provenance.prior_claim_corrected.lower()
        if "no hardware measurement" not in prior or "synthetic" not in prior:
            raise ValueError(
                "revision_round8j.hardware_plan.provenance."
                "prior_claim_corrected must correct the earlier no-hardware-"
                "measurement claim with the synthetic resource evidence"
            )
        if not provenance.synthetic_data_only:
            raise ValueError(
                "revision_round8j.hardware_plan.provenance.synthetic_data_only "
                "must be true"
            )
        for field_name in (
            "pilot_data_used",
            "pilot_score_used",
            "trained_pilot_outcome_used",
        ):
            if getattr(provenance, field_name):
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance."
                    f"{field_name} must be false"
                )
        if not provenance.informed_plan_before_decision:
            raise ValueError(
                "revision_round8j.hardware_plan.provenance."
                "informed_plan_before_decision must be true"
            )
        expected_evidence = {
            "trainsmoke": (
                "synthetic-data GPU rehearsal",
                ".superpowers/sdd/2026-11-03-gemma-loop/progress.md",
            ),
            "blocksmoke": (
                "synthetic-data GPU rehearsal",
                ".superpowers/sdd/2026-11-03-gemma-loop/"
                "task-blocksmoke-report.md",
            ),
            "block1529": (
                "synthetic-data GPU rehearsal",
                ".superpowers/sdd/2026-11-03-gemma-loop/"
                "task-block1529-report.md",
            ),
            "modal_price_check": (
                "modal price check",
                ".superpowers/sdd/2026-11-03-gemma-loop/progress.md",
            ),
            "credit_cost_estimate": (
                "credit-based cost estimate",
                ".superpowers/sdd/2026-11-03-gemma-loop/progress.md",
            ),
        }
        evidence: dict[str, PilotHardwareEvidenceSpec] = {}
        for entry in provenance.evidence:
            if not entry.id:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence needs "
                    "an id"
                )
            if entry.id in evidence:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence must "
                    "be unique"
                )
            evidence[entry.id] = entry
        if set(evidence) != set(expected_evidence):
            raise ValueError(
                "revision_round8j.hardware_plan.provenance.evidence must "
                "register trainsmoke, blocksmoke, block1529, "
                "modal_price_check and credit_cost_estimate"
            )
        for evidence_id, (kind, report_path) in expected_evidence.items():
            entry = evidence[evidence_id]
            if entry.kind != kind:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    f"{evidence_id}.kind must be {kind}"
                )
            if entry.report_path != report_path:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    f"{evidence_id}.report_path must be {report_path}"
                )
            if not entry.findings.strip():
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    f"{evidence_id}.findings must be registered"
                )
        trainsmoke = evidence["trainsmoke"].findings.lower()
        if "580" not in trainsmoke or "9.39" not in trainsmoke:
            raise ValueError(
                "revision_round8j.hardware_plan.provenance.evidence.trainsmoke "
                "must record ~580 tok/s and the 9.39 GiB peak"
            )
        blocksmoke = evidence["blocksmoke"].findings.lower()
        for marker in ("167.6", "9.99", "6.85", "step 2"):
            if marker not in blocksmoke:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    "blocksmoke must record the probe PASS, 167.6 tok/s, "
                    "9.99 GiB and the step-2 stall"
                )
        block1529 = evidence["block1529"].findings.lower()
        for marker in ("12 gb", "vram", "no allowed allocator setting"):
            if marker not in block1529:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    "block1529 must record the hard VRAM cliff on the 12 GB "
                    "card and that no allowed allocator setting was stable"
                )
        price = evidence["modal_price_check"].findings.lower()
        for marker in ("h100", "3.95", "a100-80gb", "2.50"):
            if marker not in price:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    "modal_price_check must record H100 $3.95/h and "
                    "A100-80GB $2.50/h"
                )
        credits = evidence["credit_cost_estimate"].findings.lower()
        for marker in ("26.40", "24"):
            if marker not in credits:
                raise ValueError(
                    "revision_round8j.hardware_plan.provenance.evidence."
                    "credit_cost_estimate must record USD 26.40 remaining "
                    "credits and the USD 24 internal cap"
                )
        guard = self.money_guard
        if abs(float(guard.remaining_free_credits_usd) - 26.40) > 1e-9:
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard."
                "remaining_free_credits_usd must be 26.40"
            )
        if abs(float(guard.internal_cap_usd) - 24.0) > 1e-9:
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard.internal_cap_usd "
                "must be 24"
            )
        if not guard.cap_below_remaining_credits:
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard."
                "cap_below_remaining_credits must be true"
            )
        if not (
            0.0
            < float(guard.internal_cap_usd)
            < float(guard.remaining_free_credits_usd)
        ):
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard.internal_cap_usd "
                "must be below the remaining free credits"
            )
        if not guard.monthly_ledger.strip() or not guard.shared_spend_lock.strip():
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard must use the "
                "monthly ledger and the shared spend lock"
            )
        if not guard.workspace_limit_hard_stop:
            raise ValueError(
                "revision_round8j.hardware_plan.money_guard."
                "workspace_limit_hard_stop must be true"
            )
        rerun = guard.group_rerun_rule.lower()
        for phrase in (
            "whole group",
            "rerun locally",
            "from scratch",
            "never mixed",
        ):
            if phrase not in rerun:
                raise ValueError(
                    "revision_round8j.hardware_plan.money_guard."
                    "group_rerun_rule must require a whole-group local rerun "
                    "from scratch, never mixed"
                )
        fallback = self.modal_unavailable_fallback.lower()
        if "all groups run locally" not in fallback:
            raise ValueError(
                "revision_round8j.hardware_plan.modal_unavailable_fallback "
                "must run all groups locally"
            )
        if not self.decision_log:
            raise ValueError(
                "revision_round8j.hardware_plan.decision_log must not be empty"
            )
        return self


class RevisionRound8jSpec(BaseModel):
    """Round 8j: the pre-seal pilot training hardware plan."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Revision round 8j (pilot training hardware plan)"
    date: str = ""
    stage_a_exposure: str = ""
    outcome_use: str = ""
    choices: list[str] = Field(default_factory=list)
    hardware_plan: PilotTrainingHardwareSpec = Field(
        default_factory=PilotTrainingHardwareSpec
    )

    @model_validator(mode="after")
    def _validate_registered(self) -> "RevisionRound8jSpec":
        if self.date != "2026-11-07":
            raise ValueError("revision_round8j.date must be 2026-11-07")
        exposure = self.stage_a_exposure.lower()
        if "stage a" not in exposure or "exposed" not in exposure:
            raise ValueError(
                "revision_round8j.stage_a_exposure must state that Stage A "
                "was exposed"
            )
        outcome_use = self.outcome_use.lower()
        if "no trained-arm outcome" not in outcome_use:
            raise ValueError(
                "revision_round8j.outcome_use must state that no trained-arm "
                "outcome informed the round"
            )
        if "before any pilot training" not in outcome_use:
            raise ValueError(
                "revision_round8j.outcome_use must state that the plan was "
                "fixed before any pilot training or scoring"
            )
        if not self.choices:
            raise ValueError("revision_round8j.choices must not be empty")
        return self


class Prereg(BaseModel):
    """Contents of ``prereg/prereg.yaml``.

    The legacy single-SFT fields ``criteria`` (S1-S5) and ``train_tokens`` are
    deliberately absent: they are retired so that no code can silently drive the
    old plan from a new preregistration.  ``check_criteria`` refuses to run and
    ``scripts/phase1_plan.py`` refuses to build the legacy plan.
    """

    model_config = ConfigDict(extra="forbid")

    tag: str = "prereg-v1"
    version: str = "prereg-v1"
    status: str = "draft"
    primary_n: int = 2
    loop_probs: list[float] = Field(
        default_factory=lambda: [0.25, 0.40, 0.20, 0.15]
    )
    sampling: Sampling = Field(default_factory=Sampling)
    benches: dict[str, BenchSpec] = Field(default_factory=dict)
    google: dict[str, dict[str, float]] = Field(default_factory=dict)
    harness_tolerance_pts: float = 3.0
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    seeds: dict[str, list[int]] = Field(default_factory=dict)
    creative: CreativeSpec = Field(default_factory=CreativeSpec)
    engine_gate: EngineGateSpec | None = None
    part_b: PartBSpec | None = None
    phase2: Phase2Spec | None = None

    # Two-stage hybrid design (optional for backward-compatible fixtures).
    part1: Part1Spec | None = None
    architecture: ArchitectureSpec | None = None
    dial_objective: DialObjectiveSpec | None = None
    flops: FlopsSpec | None = None
    resident_bytes: ResidentBytesSpec | None = None
    intelligence_per_byte: IntelligencePerByteSpec | None = None
    stage1: Stage1Spec | None = None
    stage2: Stage2Spec | None = None
    endpoints: EndpointsSpec | None = None
    pilot: PilotSpec | None = None
    loop_ladder: LoopLadderSpec | None = None
    training_free_campaign: TrainingFreeCampaignSpec | None = None
    data: DataSpec | None = None
    compute: ComputeSpec | None = None
    # Round 8b: observe-only monitoring, grading, external APIs and the
    # Stage-A-exposed pre-tag amendment.
    monitor: MonitorSpec | None = None
    grading: GradingSpec | None = None
    external_apis: ExternalAPIsSpec | None = None
    pins: PinsSpec | None = None
    part2: Part2Spec | None = None
    mde_table: list[MdeRow] = Field(default_factory=list)
    mde_notes: str = ""
    # Round 8: the explicit claims paragraph X1 requires in MD sections 1,
    # 6 and 12 and in the YAML endpoint descriptions.
    claims: str = ""
    revision_round8: RevisionRound8Spec | None = None
    revision_round8b: RevisionRound8bSpec | None = None
    revision_round8c: RevisionRound8cSpec | None = None
    revision_round8d: RevisionRound8dSpec | None = None
    revision_round8e: RevisionRound8eSpec | None = None
    revision_round8h: RevisionRound8hSpec | None = None
    revision_round8j: RevisionRound8jSpec | None = None
    drafter_choices: list[str] = Field(default_factory=list)
    deviations: list[str] = Field(default_factory=list)
    pending_author: list[str] = Field(default_factory=list)
    analysis_v1_obligations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_invariants(self) -> "Prereg":
        if len(self.loop_probs) not in {4, 8}:
            raise ValueError(
                "loop_probs must contain four (legacy) or eight (dial) "
                "probabilities"
            )
        if abs(sum(self.loop_probs) - 1.0) > 1e-9:
            raise ValueError(
                f"loop_probs must sum to 1, got {sum(self.loop_probs)!r}"
            )
        if not 1 <= self.primary_n <= len(self.loop_probs):
            raise ValueError(
                "primary_n must be in 1..len(loop_probs) "
                f"(got {self.primary_n} with {len(self.loop_probs)} entries)"
            )

        derived_google = {"mmmu_pro_overall"}
        for size, figures in self.google.items():
            unknown = set(figures) - set(self.benches) - derived_google
            if unknown:
                raise ValueError(
                    f"google[{size!r}] references unregistered benches: {sorted(unknown)}"
                )

        for bench, spec in self.benches.items():
            if spec.role not in {
                "confirmatory",
                "secondary",
                "o_only",
                "harness",
            }:
                raise ValueError(
                    f"benches[{bench!r}].role must be explicit and registered"
                )

        if self.part1 is not None:
            if self.part1.size not in {"E2B", "E4B"}:
                raise ValueError("part1.size must be 'E2B' or 'E4B'")
            if not self.part1.seeds:
                raise ValueError("part1.seeds must not be empty")
            if self.part1.confirmatory_seed not in self.part1.seeds:
                raise ValueError(
                    "part1.confirmatory_seed must be one of part1.seeds"
                )
            if (
                self.part1.rollout_n is not None
                and self.part1.rollout_n != self.primary_n
            ):
                raise ValueError(
                    "part1.rollout_n must equal primary_n "
                    f"({self.primary_n}), got {self.part1.rollout_n}"
                )
            if not self.part1.rollout_n_policy:
                raise ValueError(
                    "part1.rollout_n_policy must be registered"
                )
            if self.part1.rollout_n_groups:
                if any(
                    n < 2 or groups <= 0
                    for n, groups in self.part1.rollout_n_groups.items()
                ):
                    raise ValueError(
                        "part1.rollout_n_groups requires N >= 2 and positive "
                        "group counts"
                    )
                if sum(self.part1.rollout_n_groups.values()) != 16:
                    raise ValueError(
                        "part1.rollout_n_groups must sum to 16 groups/step"
                    )
            extra = [seed for seed in self.part1.extra_seeds if seed in self.part1.seeds]
            if extra != self.part1.extra_seeds:
                raise ValueError(
                    "part1.extra_seeds must be a subset of part1.seeds"
                )

        if self.architecture is not None:
            allowed_blocks = {"25-29", "16-18", "15-29"}
            unknown_blocks = set(self.architecture.loop_block_candidates) - allowed_blocks
            if unknown_blocks:
                raise ValueError(
                    "architecture.loop_block_candidates has unknown entries: "
                    f"{sorted(unknown_blocks)}"
                )
            allowed_bridges = {
                "zero_init_norm_matched_per_channel_g",
                "legacy_scalar",
            }
            unknown_bridges = set(self.architecture.bridge_candidates) - allowed_bridges
            if unknown_bridges:
                raise ValueError(
                    "architecture.bridge_candidates has unknown entries: "
                    f"{sorted(unknown_bridges)}"
                )
            if self.architecture.adapters_active_from_n not in {1, 2}:
                raise ValueError(
                    "architecture.adapters_active_from_n must be 1 or 2"
                )
            if self.architecture.n_distribution:
                if len(self.architecture.n_distribution) not in {4, 8}:
                    raise ValueError(
                        "architecture.n_distribution must contain four "
                        "(legacy) or eight (dial) probabilities"
                    )
                if len(self.architecture.n_distribution) != len(
                    self.loop_probs
                ):
                    raise ValueError(
                        "architecture.n_distribution must have the same "
                        "length as loop_probs"
                    )
                if abs(sum(self.architecture.n_distribution) - 1.0) > 1e-9:
                    raise ValueError("architecture.n_distribution must sum to 1")
                if any(
                    abs(a - b) > 1e-9
                    for a, b in zip(
                        self.architecture.n_distribution, self.loop_probs
                    )
                ):
                    raise ValueError(
                        "architecture.n_distribution must equal loop_probs"
                    )
            if self.architecture.adapters_active_from_n == 2:
                if self.loop_probs[0] != 0.0:
                    raise ValueError(
                        "loop_probs[0] must be 0 when adapters are active only "
                        "for N >= 2"
                    )
            if self.architecture.ple_mode not in {"shared", "loop_vector"}:
                raise ValueError(
                    "architecture.ple_mode must be 'shared' or 'loop_vector'"
                )
            if (
                self.architecture.ple_mode == "loop_vector"
                and self.primary_n > 4
            ):
                raise ValueError(
                    "ple_mode 'loop_vector' caps N at 4; primary_n > 4 is "
                    "forbidden"
                )

        if self.dial_objective is not None:
            dial = self.dial_objective
            if dial.window_trainable_passes <= 0:
                raise ValueError(
                    "dial_objective.window_trainable_passes must be positive"
                )
            if dial.horizon:
                if len(dial.horizon) not in {4, 8}:
                    raise ValueError(
                        "dial_objective.horizon must contain four or eight "
                        "probabilities"
                    )
                if abs(sum(dial.horizon) - 1.0) > 1e-9:
                    raise ValueError("dial_objective.horizon must sum to 1")
                if any(
                    abs(a - b) > 1e-9
                    for a, b in zip(dial.horizon, self.loop_probs)
                ):
                    raise ValueError(
                        "dial_objective.horizon must equal loop_probs"
                    )
            for field_name in ("exits", "exit_loss", "exit_weights", "oh"):
                if not getattr(dial, field_name):
                    raise ValueError(
                        f"dial_objective.{field_name} must be registered"
                    )
            if not dial.monotone_hinge:
                raise ValueError(
                    "dial_objective.monotone_hinge must be registered"
                )
            if not dial.excluded or not dial.logging:
                raise ValueError(
                    "dial_objective.excluded and .logging must be registered"
                )

        if self.flops is not None:
            if not self.flops.per_pass_fraction:
                raise ValueError(
                    "flops.per_pass_fraction must be registered"
                )
            if not self.flops.r_n:
                raise ValueError("flops.r_n must be registered")
            for block, values in self.flops.r_n.items():
                if "1" not in values or "4" not in values:
                    raise ValueError(
                        f"flops.r_n[{block!r}] must register N=1 and N=4"
                    )
            if not self.flops.accounting:
                raise ValueError("flops.accounting must be registered")

        if self.resident_bytes is not None:
            rb = self.resident_bytes
            if not rb.components or not rb.formula:
                raise ValueError(
                    "resident_bytes.components and .formula must be registered"
                )
            if not rb.dtypes or not rb.hosts:
                raise ValueError(
                    "resident_bytes.dtypes and .hosts must be registered"
                )
            if not rb.invariants:
                raise ValueError(
                    "resident_bytes.invariants must be registered"
                )

        if self.intelligence_per_byte is not None:
            ipb = self.intelligence_per_byte
            if ipb.denominator != "resident_bytes":
                raise ValueError(
                    "intelligence_per_byte.denominator must be 'resident_bytes'"
                )
            for field_name in ("quality", "derived", "anchoring", "secondary"):
                if not getattr(ipb, field_name):
                    raise ValueError(
                        f"intelligence_per_byte.{field_name} must be registered"
                    )
            if ipb.d6.id != "D6":
                raise ValueError("intelligence_per_byte.d6 must be D6")
            if "estimation" not in ipb.d6.status:
                raise ValueError(
                    "intelligence_per_byte.d6.status must be estimation only"
                )
            if not ipb.limitations:
                raise ValueError(
                    "intelligence_per_byte.limitations must be registered"
                )

        if self.stage1 is not None:
            teacher = self.stage1.teacher
            if not 0.0 < teacher.beta_parent_kl <= 0.5:
                raise ValueError("stage1.teacher.beta_parent_kl must be in (0, 0.5]")
            if teacher.top_k <= 0:
                raise ValueError("stage1.teacher.top_k must be positive")
            if not self.stage1.oh_runs_stage1:
                raise ValueError(
                    "stage1.oh_runs_stage1 must be True: O-H gets the same heal"
                )
            if self.stage1.budget_tokens <= 0:
                raise ValueError("stage1.budget_tokens must be positive")
            if self.stage1.dose_logging_interval <= 0:
                raise ValueError("stage1.dose_logging_interval must be positive")
            if self.stage1.rows_per_optimizer_step <= 0:
                raise ValueError(
                    "stage1.rows_per_optimizer_step must be positive"
                )
            if self.stage1.tokens_per_optimizer_step <= 0:
                raise ValueError(
                    "stage1.tokens_per_optimizer_step must be positive"
                )
            if self.stage1.gradient_clip <= 0.0:
                raise ValueError("stage1.gradient_clip must be positive")

        if self.stage2 is not None:
            if (
                self.stage2.rollout_n is not None
                and self.stage2.rollout_n != self.primary_n
            ):
                raise ValueError(
                    "stage2.rollout_n must equal primary_n "
                    f"({self.primary_n}), got {self.stage2.rollout_n}"
                )
            if not self.stage2.rollout_n_policy:
                raise ValueError(
                    "stage2.rollout_n_policy must be registered"
                )
            if self.stage2.rollout_n_groups:
                if any(
                    n < 2 or groups <= 0
                    for n, groups in self.stage2.rollout_n_groups.items()
                ):
                    raise ValueError(
                        "stage2.rollout_n_groups requires N >= 2 and positive "
                        "group counts"
                    )
                if sum(self.stage2.rollout_n_groups.values()) != self.stage2.prompts_per_step:
                    raise ValueError(
                        "stage2.rollout_n_groups must sum to "
                        "stage2.prompts_per_step groups/step"
                    )
            if not self.stage2.bptt_window:
                raise ValueError(
                    "stage2.bptt_window must be registered"
                )
            if not self.stage2.dial_monitoring:
                raise ValueError(
                    "stage2.dial_monitoring must be registered"
                )
            if self.stage2.beta_kl < 0.0:
                raise ValueError("stage2.beta_kl must be non-negative")
            if self.stage2.prompt_cap <= 0:
                raise ValueError("stage2.prompt_cap must be positive")
            if self.stage2.tis_cap <= 0.0:
                raise ValueError("stage2.tis_cap must be positive")
            if self.stage2.num_iterations != 1:
                raise ValueError("stage2.num_iterations must be 1")
            if self.stage2.dr_grpo_normalizer <= 0:
                raise ValueError("stage2.dr_grpo_normalizer must be positive")
            if self.stage2.grader_timeout_s <= 0:
                raise ValueError("stage2.grader_timeout_s must be positive")
            if self.stage2.fallback_rft.tokens <= 0:
                raise ValueError("stage2.fallback_rft.tokens must be positive")
            if any(
                abs(a - b) > 1e-9
                for a, b in zip(self.stage2.fallback_rft.loop_probs, self.loop_probs)
            ):
                raise ValueError(
                    "stage2.fallback_rft.loop_probs must equal loop_probs"
                )

        if self.endpoints is not None:
            if self.endpoints.precondition.id != "E1":
                raise ValueError("endpoints.precondition must be E1")
            ids = [endpoint.id for endpoint in self.endpoints.sequence]
            if ids != ["E2", "E3", "E4"]:
                raise ValueError(
                    "endpoints.sequence must be exactly [E2, E3, E4], "
                    f"got {ids}"
                )
            if self.endpoints.primary not in ids:
                raise ValueError("endpoints.primary must name a registered endpoint")
            if not self.endpoints.multiplicity:
                raise ValueError(
                    "endpoints.multiplicity must be registered"
                )
            if not self.endpoints.graphical:
                raise ValueError(
                    "endpoints.graphical must be registered"
                )
            for key in ("weights", "transitions", "p_values", "operational_order"):
                if key not in self.endpoints.graphical:
                    raise ValueError(
                        f"endpoints.graphical.{key} must be registered"
                    )
            if not self.endpoints.estimation:
                raise ValueError(
                    "endpoints.estimation must register D2-D6"
                )
            dial_ids = [endpoint.id for endpoint in self.endpoints.dial_family]
            if not dial_ids or "D1" not in dial_ids:
                raise ValueError(
                    "endpoints.dial_family must register D1"
                )
            if self.endpoints.heal_estimates.nll_nats <= 0.0:
                raise ValueError("endpoints.heal_estimates.nll_nats must be positive")
            if self.endpoints.heal_estimates.lcb_pts <= 0.0:
                raise ValueError("endpoints.heal_estimates.lcb_pts must be positive")
            if not 0.0 < self.endpoints.heal_estimates.ci_level < 1.0:
                raise ValueError("endpoints.heal_estimates.ci_level must be in (0, 1)")
            if self.endpoints.heal_estimates.gate:
                raise ValueError(
                    "endpoints.heal_estimates.gate must be False: E1 is not gated"
                )
            if self.endpoints.bootstrap.iters <= 0:
                raise ValueError("endpoints.bootstrap.iters must be positive")
            if self.endpoints.bootstrap.unit != "problem cluster":
                raise ValueError(
                    "endpoints.bootstrap.unit must be 'problem cluster'"
                )

        if self.pilot is not None:
            if self.pilot.p1.rows <= 0:
                raise ValueError("pilot.p1.rows must be positive")
            if self.pilot.p2.heal_tokens <= 0:
                raise ValueError("pilot.p2.heal_tokens must be positive")
            if len(self.pilot.p2.arms) < 7:
                raise ValueError("pilot.p2.arms must list at least 7 arms")
            for field_name in (
                "dial_rule",
                "dial_metrics_reported",
                "frozen_rule_deviation",
                "eligible_set",
                "ranking_statistic",
                "monotonicity_filter",
                "fallback_rule",
                "oh_lora_target",
            ):
                if not getattr(self.pilot.p2, field_name):
                    raise ValueError(
                        f"pilot.p2.{field_name} must be registered"
                    )
            for field_name in (
                "teacher_selection_rule",
                "teacher_gain_basis",
                "block_bridge_rule",
                "block_bridge_teacher",
            ):
                if not getattr(self.pilot.p2, field_name):
                    raise ValueError(
                        f"pilot.p2.{field_name} must be registered"
                    )
            if not self.pilot.p2.selection_only:
                raise ValueError(
                    "pilot.p2.selection_only must be True: the pilot is a "
                    "selection gate, not a confirmatory claim"
                )
            if self.pilot.p2.min_arm_vs_oh_nats > 0.0:
                raise ValueError(
                    "pilot.p2.min_arm_vs_oh_nats must be <= 0 (non-inferiority)"
                )
            if self.pilot.p2.tie_nats < 0.0:
                raise ValueError("pilot.p2.tie_nats must be non-negative")
            if not 0.0 < self.pilot.p3.max_truncation < 1.0:
                raise ValueError("pilot.p3.max_truncation must be in (0, 1)")
            if self.pilot.p3.steps <= 0:
                raise ValueError("pilot.p3.steps must be positive")
            if self.pilot.p3.prompt_median_max <= 0:
                raise ValueError("pilot.p3.prompt_median_max must be positive")

            code_verification = self.pilot.code_verification
            p1 = self.pilot.p1
            # The legacy string form belongs to pre-8e revision fixtures;
            # the round-8e contract is enforced only for the typed block.
            if isinstance(code_verification, PilotCodeVerificationSpec):
                # Round 8e: frozen code-verifier contract, the prospective
                # shortfall rule and the staged freeze index.
                for field_name in (
                    "domain",
                    "chat_rule",
                    "test_set",
                    "test_manifest",
                    "extraction",
                    "wrapper",
                    "output_comparison",
                    "grading_implementation",
                    "dependency_versions",
                    "docker_image",
                    "docker_image_digest",
                    "sandbox_limits",
                    "timeout_rule",
                    "pass_rule",
                    "fail_rule",
                    "infrastructure_rule",
                    "unsandboxed_fallback",
                    "current_count_rule",
                    "freeze_index",
                ):
                    if not getattr(code_verification, field_name):
                        raise ValueError(
                            "pilot.code_verification."
                            f"{field_name} must be registered"
                        )
                if "sha256:" not in code_verification.docker_image_digest:
                    raise ValueError(
                        "pilot.code_verification.docker_image_digest must bind a "
                        "Docker content digest"
                    )
                chat_rule = code_verification.chat_rule.lower()
                if "chat" not in chat_rule or "never" not in chat_rule:
                    raise ValueError(
                        "pilot.code_verification.chat_rule must state that chat "
                        "never counts as a verified target"
                    )

                shortfall = self.pilot.shortfall
                for field_name in (
                    "rule",
                    "domain",
                    "p1_descriptive_only",
                    "p2_domain",
                    "chat_never",
                    "extension_trigger",
                    "request_id_rule",
                    "seed_formula",
                    "request_count_rule",
                    "configuration",
                    "artifact_rule",
                    "immutable_original",
                    "resume_rule",
                    "early_stop_rule",
                    "final_shortfall_rule",
                    "no_relaxation_rule",
                    "rejected_alternatives",
                    "unused_bucket_rule",
                ):
                    if not getattr(shortfall, field_name):
                        raise ValueError(
                            f"pilot.shortfall.{field_name} must be registered"
                        )
                if shortfall.p1_gates_p2:
                    raise ValueError(
                        "pilot.shortfall.p1_gates_p2 must be False: P1 is "
                        "descriptive and never gates P2"
                    )
                if list(shortfall.attempt_indices) != [4, 5, 6, 7]:
                    raise ValueError(
                        "pilot.shortfall.attempt_indices must be [4, 5, 6, 7]"
                    )
                if shortfall.max_extension_batches != 1:
                    raise ValueError(
                        "pilot.shortfall.max_extension_batches must be 1"
                    )
                if (
                    shortfall.final_min_tfit_problems < 20
                    or shortfall.final_min_selection_problems < 30
                ):
                    raise ValueError(
                        "pilot.shortfall final minima must be at least 20 T-fit "
                        "and 30 selection problems"
                    )
                p2_domain = shortfall.p2_domain.lower()
                if (
                    "math" not in p2_domain
                    or "code" not in p2_domain
                    or "chat" not in p2_domain
                    or "never" not in p2_domain
                ):
                    raise ValueError(
                        "pilot.shortfall.p2_domain must register math plus "
                        "verified code and state that chat never counts"
                    )

                freeze = self.pilot.freeze
                if not freeze.required:
                    raise ValueError(
                        "pilot.freeze.required must be True before any pilot "
                        "scoring or P2 training"
                    )
                if freeze.index_path != "prereg/pilot-shortfall-freeze.json":
                    raise ValueError(
                        "pilot.freeze.index_path must be "
                        "prereg/pilot-shortfall-freeze.json"
                    )
                for field_name in (
                    "rule_frozen_before_data",
                    "staged_freeze",
                    "required_before",
                    "official_count",
                    "realized_counts",
                    "preview_counts",
                    "decision_input",
                    "pending_rule",
                ):
                    if not getattr(freeze, field_name):
                        raise ValueError(
                            f"pilot.freeze.{field_name} must be registered"
                        )
                if not freeze.contents:
                    raise ValueError(
                        "pilot.freeze.contents must register the index contents"
                    )
                if "staged" not in freeze.staged_freeze.lower():
                    raise ValueError(
                        "pilot.freeze.staged_freeze must state the staged freeze"
                    )

                # Round 8f: the operative v3 interpretation/recomputation
                # addendum must be registered alongside the typed verifier.
                v3 = self.pilot.verifier_v3
                if not v3.status:
                    raise ValueError(
                        "pilot.verifier_v3 must register the round-8f v3 "
                        "verifier interpretation and recomputation addendum"
                    )
                if v3.addendum_path != "prereg/pilot-shortfall-amendment.md":
                    raise ValueError(
                        "pilot.verifier_v3.addendum_path must be "
                        "prereg/pilot-shortfall-amendment.md"
                    )

                if "descriptive" not in p1.role.lower():
                    raise ValueError(
                        "pilot.p1.role must be descriptive"
                    )
                if p1.gates_p2:
                    raise ValueError(
                        "pilot.p1.gates_p2 must be False: P1 never gates P2"
                    )
                if not p1.continuation_gate:
                    raise ValueError(
                        "pilot.p1.continuation_gate must be registered"
                    )
                if p1.small_sample_threshold < 1:
                    raise ValueError(
                        "pilot.p1.small_sample_threshold must be positive"
                    )
                if not p1.zero_count_rule:
                    raise ValueError("pilot.p1.zero_count_rule must be registered")

            # R2-I1: P1 counts are one sampled-completion product, and the
            # problem count is the registered split count the sampler reads.
            expected_rows = int(p1.problems) * int(p1.samples_per_problem)
            if int(p1.rows) != expected_rows:
                raise ValueError(
                    "pilot.p1.rows must equal pilot.p1.problems x "
                    "pilot.p1.samples_per_problem "
                    f"({p1.problems} x {p1.samples_per_problem} = "
                    f"{expected_rows}), got {p1.rows}"
                )
            if self.data is not None:
                split_problems = int(self.data.splits.pilot_p1_problems)
                if int(p1.problems) != split_problems:
                    raise ValueError(
                        "pilot.p1.problems must equal "
                        "data.splits.pilot_p1_problems "
                        f"({split_problems}), got {p1.problems}"
                    )
            if int(self.pilot.data.pilot_train_tokens) < int(
                self.pilot.p2.heal_tokens
            ):
                raise ValueError(
                    "pilot.data.pilot_train_tokens must be at least "
                    "pilot.p2.heal_tokens "
                    f"({self.pilot.p2.heal_tokens}), got "
                    f"{self.pilot.data.pilot_train_tokens}"
                )

        if self.loop_ladder is not None:
            if "descriptive" not in self.loop_ladder.role:
                raise ValueError("loop_ladder.role must be descriptive")
            if "never" not in self.loop_ladder.selection_use:
                raise ValueError("loop_ladder.selection_use must be 'never'")
            if not self.loop_ladder.runs_after_tag:
                raise ValueError("loop_ladder.runs_after_tag must be True")
            if not self.loop_ladder.r_damped_rule:
                raise ValueError("loop_ladder.r_damped_rule must be registered")
            if not self.loop_ladder.r_damped_weight:
                raise ValueError("loop_ladder.r_damped_weight must be registered")
            if not self.loop_ladder.points:
                raise ValueError("loop_ladder.points must not be empty")
            required_points = {
                "O",
                "R-N2",
                "R-N3",
                "R-N4",
                "R-damped-N2",
                "R-damped-N3",
                "R-damped-N4",
                "L-heal-N2",
                "L-heal-N3",
                "L-heal-N4",
                "L-final-N2",
                "L-final-N3",
                "L-final-N4",
                "O-H-heal-N1",
                "O-H-final-N1",
            }
            present = {point.id for point in self.loop_ladder.points}
            if not required_points <= present:
                raise ValueError(
                    "loop_ladder.points is missing: "
                    f"{sorted(required_points - present)}"
                )
            # Derived, not stored independently: if a fixture changes
            # primary_n, the full-suite label follows it.  The YAML still
            # carries the field so the registered value is visible.
            self.loop_ladder.full_suite_at = f"L final N={self.primary_n}"
            if not self.loop_ladder.dial_nll_at_heal:
                raise ValueError(
                    "loop_ladder.dial_nll_at_heal must be registered"
                )
            if not self.loop_ladder.gpu_hours:
                raise ValueError("loop_ladder.gpu_hours must be registered")
            if not self.loop_ladder.gpu_hours_with_reuse:
                raise ValueError(
                    "loop_ladder.gpu_hours_with_reuse must be registered"
                )
            if not self.loop_ladder.reuse_rule:
                raise ValueError("loop_ladder.reuse_rule must be registered")
            if not self.loop_ladder.untrained_bridge_note:
                raise ValueError(
                    "loop_ladder.untrained_bridge_note must be registered"
                )

        if self.training_free_campaign is not None:
            campaign = self.training_free_campaign
            if not campaign.before_tag:
                raise ValueError(
                    "training_free_campaign.before_tag must be True"
                )
            if campaign.selection_input:
                raise ValueError(
                    "training_free_campaign.selection_input must be False"
                )
            if not campaign.frozen_candidates:
                raise ValueError(
                    "training_free_campaign.frozen_candidates must be registered"
                )
            for field_name in (
                "frozen_at_commit",
                "frozen_at_timestamp",
                "reuse_rule",
                "stage_b_gate",
            ):
                if not getattr(campaign, field_name):
                    raise ValueError(
                        f"training_free_campaign.{field_name} must be registered"
                    )
            if not campaign.no_output_at_freeze:
                raise ValueError(
                    "training_free_campaign.no_output_at_freeze must be True"
                )
            if campaign.p1_lite.problems <= 0:
                raise ValueError(
                    "training_free_campaign.p1_lite.problems must be positive"
                )
            if not campaign.corpus_disjointness:
                raise ValueError(
                    "training_free_campaign.corpus_disjointness must be registered"
                )
            for field_name in (
                "dial_requirement_disclosure",
                "rule_change_disclosure",
                "seen_at_rule_change",
                "exposure_audit",
                "decisions_with_n2_on_disk",
                "stage_a_sealing_procedure",
            ):
                if not getattr(campaign, field_name):
                    raise ValueError(
                        f"training_free_campaign.{field_name} must be registered"
                    )
            if not campaign.stage_a_sealing:
                raise ValueError(
                    "training_free_campaign.stage_a_sealing must register the "
                    "sealed Stage A artifacts"
                )
            for field_name in (
                "analysis_rules_frozen_at_commit",
                "analysis_rules_frozen_at_timestamp",
                "stage_a_visibility",
                "stage_a_completion_addendum",
            ):
                if not getattr(campaign, field_name):
                    raise ValueError(
                        f"training_free_campaign.{field_name} must be registered"
                    )
            if not campaign.mutable_status_files:
                raise ValueError(
                    "training_free_campaign.mutable_status_files must list the "
                    "live status files"
                )

        if self.data is not None:
            if self.data.max_length != self.training.max_length:
                raise ValueError(
                    "data.max_length must equal training.max_length"
                )
            if self.data.length_mode != self.training.length_mode:
                raise ValueError(
                    "data.length_mode must equal training.length_mode"
                )
            if self.data.max_length <= 0:
                raise ValueError("data.max_length must be positive")
            if not 0.0 <= self.data.reasoning_share_min <= 1.0:
                raise ValueError("data.reasoning_share_min must be in [0, 1]")
            buckets = [
                self.data.splits.train_bucket,
                self.data.splits.pilot_dev_bucket,
                self.data.splits.nll_kl_bucket,
                self.data.splits.long_traces_bucket,
                self.data.splits.heldout_test_bucket,
            ]
            if len(set(buckets)) != len(buckets):
                raise ValueError("data.splits buckets must be disjoint")
            pilot_subranges = [
                self.data.splits.pilot_p3_bucket,
                self.data.splits.pilot_selection_bucket,
                self.data.splits.pilot_tfit_bucket,
                self.data.splits.pilot_p1_bucket,
            ]
            if len(set(pilot_subranges)) != len(pilot_subranges):
                raise ValueError(
                    "data.splits pilot sub-ranges must be disjoint"
                )
            nll_subbuckets = [
                self.data.splits.post_tfit_bucket,
                self.data.splits.nll_test_bucket,
            ]
            if len(set(nll_subbuckets)) != len(nll_subbuckets):
                raise ValueError(
                    "data.splits 60-69 sub-buckets must be disjoint"
                )
            if not self.data.splits.post_tfit_bucket:
                raise ValueError(
                    "data.splits.post_tfit_bucket must be registered"
                )
            if not self.data.splits.nll_test_bucket:
                raise ValueError(
                    "data.splits.nll_test_bucket must be registered"
                )

            # R2-I1: the values the code reads are cross-checked, so a sealed
            # file cannot register a split or monitor size the code ignores.
            splits = self.data.splits
            if splits.monitor_bucket != splits.train_bucket:
                raise ValueError(
                    "data.splits.monitor_bucket must equal "
                    "data.splits.train_bucket (the MONITOR set is drawn from "
                    f"the train bucket), got {splits.monitor_bucket!r} != "
                    f"{splits.train_bucket!r}"
                )
            minimum_rows = (
                int(self.monitor.minimum_rows)
                if self.monitor is not None
                else 32
            )
            if minimum_rows != 32:
                raise ValueError(
                    "monitor.minimum_rows must be 32, the number of MONITOR "
                    f"rows the probe code requires, got {minimum_rows}"
                )
            if int(splits.monitor_problems) < minimum_rows:
                raise ValueError(
                    "data.splits.monitor_problems must be at least "
                    f"monitor.minimum_rows ({minimum_rows}), got "
                    f"{splits.monitor_problems}"
                )

        if self.engine_gate is not None and not str(
            self.engine_gate.status
        ).strip().upper().startswith("INACTIVE"):
            if not self.engine_gate.combinations:
                raise ValueError("engine_gate.combinations must not be empty")
            if not self.engine_gate.runs_after_tag:
                raise ValueError(
                    "engine_gate.runs_after_tag must be True unless the block "
                    "is explicitly INACTIVE history"
                )
            if not self.engine_gate.re_prefill_smoke:
                raise ValueError(
                    "engine_gate.re_prefill_smoke must be registered"
                )
            n_values = {
                n
                for combination in self.engine_gate.combinations
                for n in combination.n_loops
            }
            if not {1, 2, 3, 4, 6, 8} <= n_values:
                raise ValueError(
                    "engine_gate combinations must cover N in {1,2,3,4,6,8}"
                )
            if not {2, 3, 4, 6, 8} <= set(self.engine_gate.sampling_n):
                raise ValueError(
                    "engine_gate.sampling_n must cover {2,3,4,6,8}"
                )
            if not {1, 4} <= set(self.engine_gate.throughput_n):
                raise ValueError("engine_gate.throughput_n must cover {1,4}")
            for field_name in (
                "registered_reduction",
                "gate_hours",
                "smoke_hours",
                "step_time_fallback",
            ):
                if not getattr(self.engine_gate, field_name):
                    raise ValueError(
                        f"engine_gate.{field_name} must be registered"
                    )

        if self.part_b is not None:
            if self.part_b.requests < 3 * self.part_b.eval_batch:
                raise ValueError(
                    "part_b.requests must be at least 3x part_b.eval_batch"
                )
            if self.part_b.candidates < self.part_b.requests:
                raise ValueError(
                    "part_b.candidates must cover the registered requests"
                )

        if self.compute is not None:
            if not self.compute.tiers:
                raise ValueError("compute.tiers must not be empty")
            if not self.compute.hours:
                raise ValueError("compute.hours must not be empty")
            if not self.compute.evaluation_priority:
                raise ValueError("compute.evaluation_priority must not be empty")
            if not self.compute.eval_engine_gate:
                raise ValueError("compute.eval_engine_gate must be registered")
            if not self.compute.refill_engine_commit:
                raise ValueError("compute.refill_engine_commit must be registered")

        if self.pins is not None:
            if not self.pins.e2b_snapshot:
                raise ValueError("pins.e2b_snapshot must be registered")
            if not self.pins.data_addendum:
                raise ValueError("pins.data_addendum must be registered")
            if not self.pins.t_dial_bench:
                raise ValueError("pins.t_dial_bench must be registered")
            if not self.pins.bench_addendum:
                raise ValueError("pins.bench_addendum must be registered")
            if not self.pins.engine_candidate_commit:
                raise ValueError(
                    "pins.engine_candidate_commit must be registered"
                )
            if "efd0145" not in self.pins.never_pin_commits:
                raise ValueError(
                    "pins.never_pin_commits must list efd0145 (broken alone)"
                )

        if self.part2 is not None and not self.part2.rules:
            raise ValueError("part2.rules must not be empty")

        if self.part1 is not None and not self.analysis_v1_obligations:
            raise ValueError(
                "analysis_v1_obligations must be registered when part1 is set"
            )

        for row in self.mde_table:
            if row.n <= 0 or row.k <= 0 or row.mde_pts <= 0.0:
                raise ValueError(
                    "mde_table rows require n > 0, k > 0 and mde_pts > 0"
                )
            if not 0.0 < row.alpha < 1.0:
                raise ValueError("mde_table rows require alpha in (0, 1)")

        # ------------------------------------------------------------------
        # Revision round 8 (external review panel)
        # ------------------------------------------------------------------
        if self.part1 is not None and not self.claims:
            raise ValueError(
                "claims must register the X1 claims paragraph (D1, E2, D2, D4)"
            )
        if self.revision_round8 is not None:
            round8 = self.revision_round8
            required_categories = {
                "claim clarification",
                "analysis precision",
                "provenance correction",
                "execution contingency",
                "new eligibility design",
            }
            if set(round8.categories) != required_categories:
                raise ValueError(
                    "revision_round8.categories must split the amendment into "
                    f"{sorted(required_categories)}"
                )
            if not all(round8.categories.values()):
                raise ValueError(
                    "revision_round8.categories must each list their changes"
                )
            if round8.historical_rules_snapshot != (
                "8c6d78dcdbe3ac4719c9e5d7a58eeff81b1f9407"
            ):
                raise ValueError(
                    "revision_round8.historical_rules_snapshot must stay "
                    "8c6d78d (the pre-Stage-A rules snapshot)"
                )
            if not round8.authoritative_spec:
                raise ValueError(
                    "revision_round8.authoritative_spec must name the final "
                    "prereg-v1 commit as authoritative"
                )
            if not round8.choices:
                raise ValueError("revision_round8.choices must not be empty")

        # ------------------------------------------------------------------
        # Revision round 8b (Stage-A-exposed pre-tag technical amendment)
        # ------------------------------------------------------------------
        if self.revision_round8b is not None:
            round8b = self.revision_round8b
            expected_ids = {f"R8B-{index:02d}" for index in range(1, 12)}
            if set(round8b.item_ids) != expected_ids:
                raise ValueError(
                    "revision_round8b.item_ids must list R8B-01..R8B-11"
                )
            if set(round8b.items) != expected_ids:
                raise ValueError(
                    "revision_round8b.items must key exactly R8B-01..R8B-11"
                )
            for item_id, item in round8b.items.items():
                if item.id != item_id:
                    raise ValueError(
                        f"revision_round8b.items[{item_id}].id must match its key"
                    )
                for field_name in (
                    "title",
                    "changes",
                    "why",
                    "outcome_basis",
                ):
                    if not getattr(item, field_name):
                        raise ValueError(
                            f"revision_round8b.items[{item_id}].{field_name} "
                            "must be registered"
                        )
                if not item.where:
                    raise ValueError(
                        f"revision_round8b.items[{item_id}].where must be "
                        "registered"
                    )
                if item.uses_outcome:
                    raise ValueError(
                        "revision_round8b items may not use a trained-arm "
                        f"outcome (item {item_id})"
                    )
            exposure = round8b.stage_a_exposure.lower()
            if "stage a" not in exposure or "exposed" not in exposure:
                raise ValueError(
                    "revision_round8b.stage_a_exposure must state that Stage A "
                    "was exposed"
                )
            outcome_use = round8b.outcome_use.lower()
            if "trained-arm outcome" not in outcome_use or not (
                "no item" in outcome_use or "no trained-arm" in outcome_use
            ):
                raise ValueError(
                    "revision_round8b.outcome_use must state that no item uses "
                    "a trained-arm outcome"
                )
            if not round8b.choices:
                raise ValueError("revision_round8b.choices must not be empty")

        # ------------------------------------------------------------------
        # Revision round 8c (final pre-tag technical amendment)
        # ------------------------------------------------------------------
        if self.revision_round8c is not None:
            round8c = self.revision_round8c
            expected_ids = {f"R8C-{index:02d}" for index in range(1, 9)}
            if set(round8c.item_ids) != expected_ids:
                raise ValueError(
                    "revision_round8c.item_ids must list R8C-01..R8C-08"
                )
            if set(round8c.items) != expected_ids:
                raise ValueError(
                    "revision_round8c.items must key exactly R8C-01..R8C-08"
                )
            for item_id, item in round8c.items.items():
                if item.id != item_id:
                    raise ValueError(
                        f"revision_round8c.items[{item_id}].id must match its key"
                    )
                for field_name in ("title", "changes", "why", "outcome_basis"):
                    if not getattr(item, field_name):
                        raise ValueError(
                            f"revision_round8c.items[{item_id}].{field_name} "
                            "must be registered"
                        )
                if not item.where:
                    raise ValueError(
                        f"revision_round8c.items[{item_id}].where must be "
                        "registered"
                    )
                if item.uses_outcome:
                    raise ValueError(
                        "revision_round8c items may not use a trained-arm "
                        f"outcome (item {item_id})"
                    )
            exposure = round8c.stage_a_exposure.lower()
            if "stage a" not in exposure or "exposed" not in exposure:
                raise ValueError(
                    "revision_round8c.stage_a_exposure must state that Stage A "
                    "was exposed"
                )
            outcome_use = round8c.outcome_use.lower()
            if "trained-arm outcome" not in outcome_use or not (
                "no item" in outcome_use or "no trained-arm" in outcome_use
            ):
                raise ValueError(
                    "revision_round8c.outcome_use must state that no item uses "
                    "a trained-arm outcome"
                )
            if not round8c.choices:
                raise ValueError("revision_round8c.choices must not be empty")

        # ------------------------------------------------------------------
        # Revision round 8d (final pre-tag code/pin/domain alignment)
        # ------------------------------------------------------------------
        if self.revision_round8d is not None:
            round8d = self.revision_round8d
            expected_ids = {f"R8D-{index:02d}" for index in range(1, 6)}
            if set(round8d.item_ids) != expected_ids:
                raise ValueError(
                    "revision_round8d.item_ids must list R8D-01..R8D-05"
                )
            if set(round8d.items) != expected_ids:
                raise ValueError(
                    "revision_round8d.items must key exactly R8D-01..R8D-05"
                )
            for item_id, item in round8d.items.items():
                if item.id != item_id:
                    raise ValueError(
                        f"revision_round8d.items[{item_id}].id must match its key"
                    )
                for field_name in ("title", "changes", "why", "outcome_basis"):
                    if not getattr(item, field_name):
                        raise ValueError(
                            f"revision_round8d.items[{item_id}].{field_name} "
                            "must be registered"
                        )
                if not item.where:
                    raise ValueError(
                        f"revision_round8d.items[{item_id}].where must be "
                        "registered"
                    )
                if item.uses_outcome:
                    raise ValueError(
                        "revision_round8d items may not use a trained-arm "
                        f"outcome (item {item_id})"
                    )
            exposure = round8d.stage_a_exposure.lower()
            if "stage a" not in exposure or "exposed" not in exposure:
                raise ValueError(
                    "revision_round8d.stage_a_exposure must state that Stage A "
                    "was exposed"
                )
            outcome_use = round8d.outcome_use.lower()
            if "trained-arm outcome" not in outcome_use or not (
                "no item" in outcome_use or "no trained-arm" in outcome_use
            ):
                raise ValueError(
                    "revision_round8d.outcome_use must state that no item uses "
                    "a trained-arm outcome"
                )
            if not round8d.choices:
                raise ValueError("revision_round8d.choices must not be empty")

        # ------------------------------------------------------------------
        # Revision round 8e (prospective pilot shortfall amendment)
        # ------------------------------------------------------------------
        if self.revision_round8e is not None:
            round8e = self.revision_round8e
            expected_ids = {f"R8E-{index:02d}" for index in range(1, 9)}
            if set(round8e.item_ids) != expected_ids:
                raise ValueError(
                    "revision_round8e.item_ids must list R8E-01..R8E-08"
                )
            if set(round8e.items) != expected_ids:
                raise ValueError(
                    "revision_round8e.items must key exactly R8E-01..R8E-08"
                )
            for item_id, item in round8e.items.items():
                if item.id != item_id:
                    raise ValueError(
                        f"revision_round8e.items[{item_id}].id must match its key"
                    )
                for field_name in ("title", "changes", "why", "outcome_basis"):
                    if not getattr(item, field_name):
                        raise ValueError(
                            f"revision_round8e.items[{item_id}].{field_name} "
                            "must be registered"
                        )
                if not item.where:
                    raise ValueError(
                        f"revision_round8e.items[{item_id}].where must be "
                        "registered"
                    )
                if item.uses_outcome:
                    raise ValueError(
                        "revision_round8e items may not use a trained-arm "
                        f"outcome (item {item_id})"
                    )
            exposure = round8e.stage_a_exposure.lower()
            if "stage a" not in exposure or "exposed" not in exposure:
                raise ValueError(
                    "revision_round8e.stage_a_exposure must state that Stage A "
                    "was exposed"
                )
            outcome_use = round8e.outcome_use.lower()
            if "trained-arm outcome" not in outcome_use or not (
                "no item" in outcome_use or "no trained-arm" in outcome_use
            ):
                raise ValueError(
                    "revision_round8e.outcome_use must state that no item uses "
                    "a trained-arm outcome"
                )
            if not round8e.choices:
                raise ValueError("revision_round8e.choices must not be empty")

        if self.monitor is not None:
            monitor = self.monitor
            for field_name in (
                "role",
                "corpus",
                "split_rule",
                "training_exclusion",
                "teacher_shards",
                "abort_terminal",
                "resume_rule",
                "probe_failure_rule",
                "reserved_ids_contract",
            ):
                if not getattr(monitor, field_name):
                    raise ValueError(
                        f"monitor.{field_name} must be registered"
                    )
            if not monitor.abort_rules or not monitor.warn_rules:
                raise ValueError(
                    "monitor.abort_rules and monitor.warn_rules must be "
                    "registered"
                )
            if monitor.abort_exit_code != 76:
                raise ValueError("monitor.abort_exit_code must be 76")

        if self.grading is not None:
            grading = self.grading
            for field_name in (
                "math_verify_version",
                "math_verify_internal_timeouts",
                "math_rule",
                "math_why",
                "decided_by",
                "code_sandbox",
                "code_rule",
                "unsandboxed_rule",
                "dependency",
            ):
                if not getattr(grading, field_name):
                    raise ValueError(
                        f"grading.{field_name} must be registered"
                    )

        if self.external_apis is not None:
            apis = self.external_apis
            for field_name in (
                "deepseek_api",
                "deepseek_model",
                "deepseek_served_version",
                "deepseek_off_peak",
                "deepseek_peak_windows",
                "deepseek_margins",
                "deepseek_waste_stop",
                "deepseek_fingerprint",
                "deepseek_phase_approval",
                "deepseek_project_expected_usd",
                "nvidia_nim",
                "nvidia_nim_terms",
                "opencode_endpoints",
            ):
                if not getattr(apis, field_name):
                    raise ValueError(
                        f"external_apis.{field_name} must be registered"
                    )
            if apis.deepseek_paid_pilot_max_items > 20:
                raise ValueError(
                    "external_apis.deepseek_paid_pilot_max_items must be <= 20"
                )
            if not apis.deepseek_budgets:
                raise ValueError(
                    "external_apis.deepseek_budgets must register every phase"
                )

        if self.teacher_round8_fields_missing():
            raise ValueError(
                "stage1.teacher must register the round-8 runtimes, "
                "qualification rule, manifest identity and fairness rule"
            )

        if self.endpoints is not None:
            nll = self.endpoints.nll_analysis
            for field_name in (
                "supervised_token_mask",
                "weighting",
                "statistic",
                "p_value",
                "temperature",
                "unevaluable_corpus",
                "fit_test_freeze",
                "denominators",
            ):
                if not getattr(nll, field_name):
                    raise ValueError(
                        f"endpoints.nll_analysis.{field_name} must be registered"
                    )

        if self.stage2 is not None:
            for field_name in (
                "feasibility_bench",
                "mandatory_paths",
                "rollout_storage",
                "optimizer_state",
                "fallback_must_fit",
                "mandatory_stop_rule",
            ):
                if not getattr(self.stage2, field_name):
                    raise ValueError(
                        f"stage2.{field_name} must be registered"
                    )

        if self.engine_gate is not None and not str(
            self.engine_gate.status
        ).strip().upper().startswith("INACTIVE"):
            gate = self.engine_gate
            if len(gate.pre_tag_smoke_attempts) != 4:
                raise ValueError(
                    "engine_gate.pre_tag_smoke_attempts must record the four "
                    "pre-tag smoke attempts"
                )
            for field_name in (
                "pre_tag_smoke_artifact",
                "pre_tag_smoke_artifact_sha256",
                "pre_tag_smoke_outcome",
                "stream_diagnostic",
                "replacement_rule",
                "calibration_partition",
                "validation_partition",
                "chosen_token_regret",
                "sampled_decision_traces",
                "boundary_certificate",
                "exactness_rule",
                "failure_policy",
                "repair_cycle",
                "engine_pin_rule",
            ):
                if not getattr(gate, field_name):
                    raise ValueError(
                        f"engine_gate.{field_name} must be registered"
                    )
            if gate.shared_steps_min < 32:
                raise ValueError(
                    "engine_gate.shared_steps_min must be at least 32"
                )
            if gate.pre_tag_smoke_artifact_sha256 != (
                "fa117eff3be7a7ed46a5574073e7b17effef60abe0915f067e2b647ccaca046c"
            ):
                raise ValueError(
                    "engine_gate.pre_tag_smoke_artifact_sha256 must record the "
                    "completed SMOKE-FAIL artifact"
                )
            if "not eligible" not in gate.pre_tag_smoke_outcome.lower():
                raise ValueError(
                    "engine_gate.pre_tag_smoke_outcome must state that the "
                    "current refill candidate is not eligible"
                )

        if self.training_free_campaign is not None:
            campaign = self.training_free_campaign
            if not campaign.post_seal_access_record:
                raise ValueError(
                    "training_free_campaign.post_seal_access_record must be "
                    "registered"
                )
            if not campaign.freeze_sequence:
                raise ValueError(
                    "training_free_campaign.freeze_sequence must be registered"
                )

        if self.compute is not None:
            if not self.compute.stage0_lite:
                raise ValueError("compute.stage0_lite must be registered")
            if not self.compute.stage0_main_tier_rule:
                raise ValueError(
                    "compute.stage0_main_tier_rule must be registered"
                )
            if not self.compute.engine_pin_rule:
                raise ValueError("compute.engine_pin_rule must be registered")

        if self.pins is not None:
            if not self.pins.engine_candidate_commit:
                raise ValueError(
                    "pins.engine_candidate_commit must be registered"
                )
            if "fix/gate-v6" not in self.pins.engine_candidate_commit:
                raise ValueError(
                    "pins.engine_candidate_commit must be the fix/gate-v6 "
                    "commit that the repaired smoke ran on"
                )
            if not self.pins.engine_candidate_historical:
                raise ValueError(
                    "pins.engine_candidate_historical must record eed7fa7"
                )
            for field_name in (
                "repaired_smoke_addendum",
                "teacher_runtime_addendum",
            ):
                if not getattr(self.pins, field_name):
                    raise ValueError(
                        f"pins.{field_name} must be registered"
                    )

        if self.pilot is not None:
            if not self.pilot.why_report or not self.pilot.why_metrics:
                raise ValueError(
                    "pilot.why_report and pilot.why_metrics must be registered"
                )

        if self.revision_round8h is not None:
            if self.pilot is None:
                raise ValueError(
                    "revision_round8h requires the registered pilot block"
                )
            declaration = self.pilot.verifier_v3.author_access_declaration
            if declaration.status != "confirmed":
                raise ValueError(
                    "revision_round8h requires the round-8i author declaration "
                    "to be confirmed"
                )
            if self.revision_round8h.data_addendum.v3r.status not in {
                "awaiting_v3r",
                "complete",
            }:
                raise ValueError(
                    "revision_round8h.data_addendum.v3r.status must be "
                    "awaiting_v3r or complete"
                )

        if self.revision_round8j is not None:
            if self.pilot is None:
                raise ValueError(
                    "revision_round8j requires the registered pilot block"
                )
            if self.revision_round8h is None:
                raise ValueError(
                    "revision_round8j requires the round-8h data addendum"
                )
            if self.revision_round8h.data_addendum.v3r.status != "complete":
                raise ValueError(
                    "revision_round8j requires the completed round-8h data "
                    "addendum v3r section"
                )

        return self

    def teacher_round8_fields_missing(self) -> bool:
        """True when a registered Stage 1 teacher lacks the round-8 fields."""
        if self.stage1 is None:
            return False
        teacher = self.stage1
        required = (
            "runtime_qualification",
            "qualification_corpus",
            "qualification_references",
            "qualification_metrics",
            "expansion_rule",
            "manifest_identity",
            "comparison_rule",
            "four_bit_limitation",
            "execution_repair",
            "freeze_rule",
        )
        if not teacher.runtimes or not teacher.acceptance_table:
            return True
        if not teacher.fallback_order:
            return True
        return any(not getattr(teacher, name) for name in required)


def load_prereg(path: str | Path) -> Prereg:
    """Load and validate a pre-registration YAML file."""

    source = Path(path)
    raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    return Prereg.model_validate(raw)


def pending_pin_fields(prereg: Prereg) -> list[str]:
    """Return ``pins`` fields still holding a PIN-PENDING placeholder."""

    if prereg.pins is None:
        return []
    return [f"pins.{name}" for name in prereg.pins.pending_pin_fields()]


PLACEHOLDERS = ("PIN-PENDING", "ALIGN-WITH-CODE")


def _placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    """Return dotted paths of every string containing a tag placeholder."""

    found: list[str] = []
    if isinstance(value, str):
        tokens = [token for token in PLACEHOLDERS if token in value]
        if tokens:
            label = prefix or "<root>"
            found.append(f"{label} ({', '.join(tokens)})")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_placeholder_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(_placeholder_paths(item, child))
    return found


def assert_tag_ready(prereg: Prereg) -> None:
    """Refuse a prereg-v1 tag request while any tag placeholder remains.

    The gate scans the WHOLE loaded document, not only ``pins`` fields, so a
    ``PIN-PENDING`` or ``ALIGN-WITH-CODE`` placeholder in any nested string
    also blocks the tag.  Call it from the tag step (and from
    ``scripts/check_prereg_loads.py --tag-mode``).
    """

    pending = _placeholder_paths(prereg.model_dump())
    if pending:
        joined = ", ".join(pending)
        raise ValueError(
            "cannot tag prereg-v1 while PIN-PENDING remains (or "
            "ALIGN-WITH-CODE is present): " + joined
        )

    if prereg.revision_round8h is not None:
        v3r = prereg.revision_round8h.data_addendum.v3r
        if v3r.status == "awaiting_v3r":
            raise ValueError(
                "cannot tag prereg-v1 while the round-8h data addendum v3r "
                "section is awaiting_v3r; fill "
                "revision_round8h.data_addendum.v3r: "
                + ", ".join(v3r.remaining_fields)
            )


def default_prereg_path() -> Path:
    """Return the repository's ``prereg/prereg.yaml`` path."""
    return Path(__file__).resolve().parents[2] / "prereg" / "prereg.yaml"


def criteria_benches(prereg: Prereg) -> list[str]:
    """Benches in the fixed-sequence confirmatory family (E2-E4)."""

    return [
        bench
        for bench, spec in prereg.benches.items()
        if spec.role == "confirmatory"
    ]


def secondary_benches(prereg: Prereg) -> list[str]:
    """Benches reported as estimation only, never as pass/fail."""

    return [
        bench
        for bench, spec in prereg.benches.items()
        if spec.role == "secondary"
    ]


def o_only_benches(prereg: Prereg) -> list[str]:
    """Benches scored on the base model only."""

    return [
        bench
        for bench, spec in prereg.benches.items()
        if spec.role == "o_only"
    ]


def harness_benches(prereg: Prereg) -> list[str]:
    """Benches kept solely for the Google reproduction check."""

    return [
        bench
        for bench, spec in prereg.benches.items()
        if spec.role == "harness"
    ]


# ---------------------------------------------------------------------------
# Summaries produced by the creative and agent tasks
# ---------------------------------------------------------------------------


class CreativeSummary(BaseModel):
    """Aggregated blind-pair judgments with the L-vs-O-H win rate."""

    model_config = ConfigDict(extra="forbid")

    win_rate: float = Field(ge=0.0, le=1.0)
    lo: float = Field(ge=0.0, le=1.0)
    hi: float = Field(ge=0.0, le=1.0)
    flip_rate: float = Field(ge=0.0, le=1.0)
    n_valid: int = Field(ge=0)
    n_invalid: int = Field(ge=0)
    n_render_error: int = Field(ge=0)
    kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    kappa_ci95: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_ordering(self) -> "CreativeSummary":
        if not self.lo <= self.win_rate <= self.hi:
            raise ValueError("CreativeSummary requires lo <= win_rate <= hi")
        if self.kappa_ci95 is not None:
            lo, hi = self.kappa_ci95
            if not -1.0 <= lo <= hi <= 1.0:
                raise ValueError("kappa_ci95 must satisfy -1 <= lo <= hi <= 1")
        return self


class AgentSummary(BaseModel):
    """OpenCode agent metrics for one variant (Task 25).

    ``per_task`` maps task ids to success in [0, 1].  It is optional because the
    Task 25 collector may not expose it yet; without it S4 cannot use the
    pre-registered paired rule and returns ``passed=None`` / "no paired data".
    """

    model_config = ConfigDict(extra="forbid")

    valid_tool_call_rate: float = Field(ge=0.0, le=1.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    success_ci95: tuple[float, float]
    mean_steps: float = Field(ge=0.0)
    tokens_per_solved_task: float = Field(ge=0.0)
    n_tasks: int = Field(ge=0)
    per_task: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> "AgentSummary":
        lo, hi = self.success_ci95
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("success_ci95 must satisfy 0 <= lo <= hi <= 1")
        if self.per_task is not None:
            for task_id, value in self.per_task.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"per_task[{task_id!r}] must be in [0, 1], got {value!r}"
                    )
        return self


class Verdict(BaseModel):
    """One pre-registered check.

    ``passed`` is ``None`` when the check is informative only (S5), when data
    are missing, or when there are too few paired items to estimate an interval.
    """

    name: str
    passed: bool | None
    detail: str


# ---------------------------------------------------------------------------
# Result selection and item-level aggregation
# ---------------------------------------------------------------------------


def _resolve_size(results: list[RunResult], size: str | None) -> str | None:
    """Return the model size to evaluate, rejecting silent mixed-size input."""
    if size is not None:
        return size.upper()
    sizes = {run.size.upper() for run in results}
    if len(sizes) > 1:
        raise ValueError(
            "results contain multiple model sizes "
            f"({sorted(sizes)}); pass size= explicitly"
        )
    return next(iter(sizes), None)


def _runs(
    results: list[RunResult],
    variant: str,
    n_loops: int,
    bench: str,
    size: str | None,
    *,
    seed: int | None = None,
) -> list[RunResult]:
    """All runs for one (size, arm, N, bench), optionally one seed."""
    selected: list[RunResult] = []
    for run in results:
        if run.variant.upper() != variant.upper():
            continue
        if run.n_loops != n_loops or run.bench != bench:
            continue
        if size is not None and run.size.upper() != size.upper():
            continue
        if seed is not None and run.seed != seed:
            continue
        selected.append(run)
    return selected


def _item_scores(
    runs: list[RunResult], item_ids: set[str] | None = None
) -> dict[str, float]:
    """Mean correctness per item over every seed and sample in ``runs``."""
    totals: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for item in run.items:
            if item_ids is not None and item.item_id not in item_ids:
                continue
            totals[item.item_id].append(1.0 if item.correct else 0.0)
    return {
        item_id: sum(values) / len(values)
        for item_id, values in totals.items()
        if values
    }


def _subgroup_item_scores(
    runs: list[RunResult], meta_key: str, value: Any
) -> dict[str, float]:
    """Per-item scores restricted to items whose metadata matches."""

    totals: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for item in run.items:
            if item.meta.get(meta_key) == value:
                totals[item.item_id].append(1.0 if item.correct else 0.0)
    return {
        item_id: sum(values) / len(values)
        for item_id, values in totals.items()
        if values
    }


@dataclass(frozen=True)
class HarnessCheck:
    """Outcome of the pre-registered harness-validity rule for one benchmark.

    ``accuracy`` is the mean per-item score (samples and seeds averaged first),
    ``n`` is the number of distinct items and ``low``/``high`` is the Wilson
    95 % interval over those items.  Individually missing pieces stay ``None``
    so callers can render a table row without inventing numbers.
    """

    passed: bool | None
    detail: str
    accuracy: float | None = None
    n: int = 0
    low: float | None = None
    high: float | None = None
    low_power: bool = False


def harness_validity_from_accuracy(
    google: float | None,
    accuracy: float | None,
    n: int,
    tolerance: float,
) -> HarnessCheck:
    """Apply the pre-registered harness-validity rule to an accuracy and ``n``.

    This is the single implementation of the rule used by
    :func:`check_criteria`, the three-column report and derived rows such as
    MMMU-Pro Overall.  The Wilson interval is computed over distinct items
    (samples are averaged per item first), never over generated samples.  The
    original model is valid when ``|mine - google| <= tolerance`` **or**
    Google's figure lies inside the Wilson interval.  An interval wider than the
    tolerance is flagged as a low-power check.
    """

    if n <= 0 or accuracy is None:
        detail = (
            "no Google figure (exploratory)"
            if google is None
            else "unchecked: no O run"
        )
        return HarnessCheck(None, detail)

    low, high = wilson(accuracy * n, n)
    if google is None:
        return HarnessCheck(
            None,
            "no Google figure (exploratory)",
            accuracy=accuracy,
            n=n,
            low=low,
            high=high,
        )

    mine = accuracy * 100.0
    low_pts, high_pts = low * 100.0, high * 100.0
    half_width = (high_pts - low_pts) / 2.0
    difference = abs(mine - google)
    passed = difference <= tolerance or low_pts <= google <= high_pts
    low_power = half_width > tolerance

    detail = (
        f"mine={mine:.1f}% vs google={google:.1f}% (n={n} items); "
        f"|diff|={difference:.1f} pp; Wilson95=[{low_pts:.1f}, "
        f"{high_pts:.1f}]"
    )
    if low_power:
        detail += "; chequeo de baja potencia"
    detail += "; validado" if passed else "; no validado"

    return HarnessCheck(
        passed=passed,
        detail=detail,
        accuracy=accuracy,
        n=n,
        low=low,
        high=high,
        low_power=low_power,
    )


def harness_validity_from_scores(
    google: float | None,
    scores: Mapping[str, float],
    tolerance: float,
) -> HarnessCheck:
    """Apply the harness rule to per-item scores (samples already averaged)."""
    accuracy = _mean(list(scores.values())) if scores else None
    return harness_validity_from_accuracy(google, accuracy, len(scores), tolerance)


def harness_validity(
    google: float | None,
    runs: Sequence[RunResult],
    tolerance: float,
) -> HarnessCheck:
    """Apply the harness-validity rule to every sample in ``runs``."""
    return harness_validity_from_scores(google, _item_scores(list(runs)), tolerance)


def _paired_scores(
    runs_a: list[RunResult], runs_b: list[RunResult]
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Pair two arms on matching item ids only."""
    scores_a = _item_scores(runs_a)
    scores_b = _item_scores(runs_b)
    common = sorted(set(scores_a) & set(scores_b))
    return (
        common,
        {item_id: scores_a[item_id] for item_id in common},
        {item_id: scores_b[item_id] for item_id in common},
    )


def _mean_tokens_per_problem(runs: list[RunResult], item_ids: list[str]) -> float:
    """Mean generated completion tokens per problem across runs/seeds."""
    if not runs or not item_ids:
        return 0.0
    wanted = set(item_ids)
    per_run: list[float] = []
    for run in runs:
        total = sum(
            item.completion_tokens for item in run.items if item.item_id in wanted
        )
        per_run.append(total / len(wanted))
    return sum(per_run) / len(per_run)


def _flops_values(
    runs: list[RunResult], flops_per_problem: Callable[[RunResult], float] | None
) -> list[float] | None:
    if flops_per_problem is None or not runs:
        return None
    return [float(flops_per_problem(run)) for run in runs]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _normal_pvalue(mean: float, se: float) -> float:
    """Two-sided normal p-value for a paired mean difference."""
    if se <= 0.0:
        return 0.0 if abs(mean) > 0.0 else 1.0
    return math.erfc(abs(mean) / (se * math.sqrt(2.0)))


def _google_value(prereg: Prereg, size: str | None, bench: str) -> float | None:
    if size is None:
        return None
    for size_key, values in prereg.google.items():
        if size_key.upper() != size.upper():
            continue
        for bench_key, value in values.items():
            if bench_key.lower() == bench.lower():
                return float(value)
    return None


def _per_seed_intervals(
    prereg: Prereg,
    l_runs: list[RunResult],
    oh_runs: list[RunResult],
) -> str:
    seeds = sorted({run.seed for run in l_runs} & {run.seed for run in oh_runs})
    parts: list[str] = []
    for seed in seeds:
        l_seed = [run for run in l_runs if run.seed == seed]
        oh_seed = [run for run in oh_runs if run.seed == seed]
        common, a, b = _paired_scores(l_seed, oh_seed)
        if len(common) < 2:
            parts.append(f"seed {seed}: insufficient")
            continue
        lo, hi = paired_bootstrap(
            a,
            b,
            iters=prereg.criteria.bootstrap_iters,
            seed=prereg.criteria.bootstrap_seed,
        )
        parts.append(f"seed {seed} [{lo * 100:+.1f}, {hi * 100:+.1f}] pp")
    return "per-seed: " + ("; ".join(parts) if parts else "n/a")


def _s2_base(
    prereg: Prereg,
    results: list[RunResult],
    n_loops: int,
    size: str | None,
    flops_per_problem: Callable[[RunResult], float] | None,
) -> tuple[Verdict, float | None, float | None]:
    """Compute the raw S2 check for one N.

    Returns ``(verdict, mean, pvalue)``.  ``mean``/``pvalue`` are ``None`` when
    there are not enough paired items to estimate anything.
    """

    name = "S2" if n_loops == prereg.primary_n else f"S2_N{n_loops}_exploratory"
    bench = prereg.criteria.s2_bench
    l_runs = _runs(results, "L", n_loops, bench, size)
    oh_runs = _runs(results, "O-H", 0, bench, size)
    if not l_runs or not oh_runs:
        return Verdict(name=name, passed=None, detail="sin datos"), None, None

    common, a, b = _paired_scores(l_runs, oh_runs)
    if len(common) < 2:
        return (
            Verdict(
                name=name,
                passed=None,
                detail="insufficient data: n<2 paired items",
            ),
            None,
            None,
        )

    lo, hi = paired_bootstrap(
        a,
        b,
        iters=prereg.criteria.bootstrap_iters,
        seed=prereg.criteria.bootstrap_seed,
    )
    mean, se, _, _ = paired_diff(a, b)

    paired_samples = sum(
        1
        for run in l_runs
        for item in run.items
        if item.item_id in set(common)
    )
    samples_per_item = max(1, round(paired_samples / len(common)))
    detectable = mde(len(common), samples_per_item) * 100.0

    l_tokens = _mean_tokens_per_problem(l_runs, common)
    oh_tokens = _mean_tokens_per_problem(oh_runs, common)
    equal_accuracy = abs(mean) <= 1e-12
    fewer_tokens = equal_accuracy and l_tokens < oh_tokens
    passes = lo > 0.0 or fewer_tokens

    l_flops = _flops_values(l_runs, flops_per_problem)
    oh_flops = _flops_values(oh_runs, flops_per_problem)
    if l_flops is None or oh_flops is None:
        flops_detail = "FLOPs: not wired"
    else:
        flops_detail = (
            f"FLOPs/problem L={_mean(l_flops):.3e} O-H={_mean(oh_flops):.3e}"
        )

    parts = [
        f"n={len(common)} paired items",
        f"mean diff={mean * 100:+.1f} pp",
        f"95% paired percentile bootstrap=[{lo * 100:+.1f}, {hi * 100:+.1f}] pp",
        f"MDE(k={samples_per_item})={detectable:.1f} pp",
        f"mean generated tokens/problem L={l_tokens:.1f} vs O-H={oh_tokens:.1f}",
        flops_detail,
        _per_seed_intervals(prereg, l_runs, oh_runs),
        "training-seed variance not modelled (paired item bootstrap)",
    ]
    if lo > 0.0:
        parts.append("CI excludes 0 in favor of L")
    elif fewer_tokens:
        parts.append("equal accuracy with fewer tokens")
    else:
        parts.append("CI includes 0")

    return (
        Verdict(name=name, passed=passes, detail="; ".join(parts)),
        mean,
        _normal_pvalue(mean, se),
    )


def _s1(prereg: Prereg, results: list[RunResult], size: str | None) -> Verdict:
    missing: list[str] = []
    parts: list[str] = []
    within_all = True

    for bench in criteria_benches(prereg):
        l_runs = _runs(results, "L", 1, bench, size)
        o_runs = _runs(results, "O", 0, bench, size)
        if not l_runs or not o_runs:
            missing.append(bench)
            continue
        common, a, b = _paired_scores(l_runs, o_runs)
        if len(common) < 2:
            missing.append(f"{bench} (n<2 paired items)")
            continue
        lo, hi = paired_bootstrap(
            a,
            b,
            iters=prereg.criteria.bootstrap_iters,
            seed=prereg.criteria.bootstrap_seed,
        )
        mean, _, _, _ = paired_diff(a, b)
        within = lo <= 0.0 <= hi
        within_all = within_all and within
        parts.append(
            f"{bench}: {mean * 100:+.1f} pp "
            f"95% CI=[{lo * 100:+.1f}, {hi * 100:+.1f}] "
            f"{'OK' if within else 'OUT'}"
        )

    if missing:
        detail = f"incomplete: missing {missing}"
        if parts:
            detail += "; " + "; ".join(parts)
        return Verdict(name="S1", passed=None, detail=detail)
    if not parts:
        return Verdict(name="S1", passed=None, detail="sin datos")
    return Verdict(
        name="S1",
        passed=within_all,
        detail="L(N=1) - O paired 95% CI includes 0 on every bench: "
        + "; ".join(parts),
    )


def _s3(prereg: Prereg, creative: dict[str, CreativeSummary] | None) -> Verdict:
    if not creative:
        return Verdict(name="S3", passed=None, detail="sin datos")

    summary = creative.get("L")
    if summary is None:
        return Verdict(
            name="S3",
            passed=None,
            detail="sin datos: no L summary (canonical key is 'L')",
        )
    if summary.n_valid == 0:
        return Verdict(name="S3", passed=None, detail="insufficient data")

    threshold = prereg.criteria.s3_min_win_rate_lower
    passed = summary.lo > threshold
    detail = (
        f"win_rate={summary.win_rate:.3f}, CI95=[{summary.lo:.3f}, "
        f"{summary.hi:.3f}], flip_rate={summary.flip_rate:.3f}, "
        f"n_valid={summary.n_valid}, n_invalid={summary.n_invalid}, "
        f"n_render_error={summary.n_render_error}"
    )
    if summary.kappa is not None:
        detail += f", kappa={summary.kappa:.3f}"
        if summary.kappa_ci95 is not None:
            detail += (
                f" [{summary.kappa_ci95[0]:.3f}, {summary.kappa_ci95[1]:.3f}]"
            )
    detail += f"; lower bound {'>' if passed else '<='} {threshold:g}"
    return Verdict(name="S3", passed=passed, detail=detail)


def _adapted_gain_reports(agent: dict[str, AgentSummary]) -> str:
    pairs = (("O+OC", "O"), ("L+OC", "L"))
    parts: list[str] = []
    for adapted, base in pairs:
        if adapted in agent and base in agent:
            gain = (
                agent[adapted].success_rate - agent[base].success_rate
            ) * 100.0
            parts.append(
                f"{adapted} vs {base}: "
                f"{agent[adapted].success_rate:.3f} vs "
                f"{agent[base].success_rate:.3f} ({gain:+.1f} pp)"
            )
        else:
            parts.append(f"{adapted} vs {base}: sin datos")
    return "; ".join(parts)


def _s4(prereg: Prereg, agent: dict[str, AgentSummary] | None) -> Verdict:
    if not agent:
        return Verdict(name="S4", passed=None, detail="sin datos")

    l_oc = agent.get("L+OC")
    oh_oc = agent.get("O-H+OC")
    if l_oc is None or oh_oc is None:
        return Verdict(name="S4", passed=None, detail="sin datos")
    if l_oc.n_tasks == 0 or oh_oc.n_tasks == 0:
        return Verdict(name="S4", passed=None, detail="insufficient data")
    if not l_oc.per_task or not oh_oc.per_task:
        return Verdict(name="S4", passed=None, detail="no paired data")

    common = sorted(set(l_oc.per_task) & set(oh_oc.per_task))
    if len(common) < 2:
        return Verdict(name="S4", passed=None, detail="insufficient data")
    a = {task_id: l_oc.per_task[task_id] for task_id in common}
    b = {task_id: oh_oc.per_task[task_id] for task_id in common}
    lo, hi = paired_bootstrap(
        a,
        b,
        iters=prereg.criteria.bootstrap_iters,
        seed=prereg.criteria.bootstrap_seed,
    )
    mean, se, _, _ = paired_diff(a, b)

    threshold = prereg.criteria.s4_min_gain_pts
    if prereg.phase2 is not None:
        threshold = prereg.phase2.s4_min_gain_pts
    passed = lo * 100.0 > threshold
    detail = (
        f"n={len(common)} paired tasks; mean success diff={mean * 100:+.1f} pp; "
        f"95% paired percentile bootstrap=[{lo * 100:+.1f}, {hi * 100:+.1f}] pp; "
        f"threshold={threshold:g} pp; "
        f"success_rate L+OC={l_oc.success_rate:.3f} vs O-H+OC="
        f"{oh_oc.success_rate:.3f}; n_tasks={l_oc.n_tasks}/{oh_oc.n_tasks}; "
        f"+OC vs non-adapted: {_adapted_gain_reports(agent)}"
    )
    return Verdict(name="S4", passed=passed, detail=detail)


def _s5(
    prereg: Prereg, results: list[RunResult], size: str | None
) -> Verdict:
    parts: list[str] = []
    for bench in criteria_benches(prereg):
        l_runs = _runs(results, "L", prereg.primary_n, bench, size)
        o_runs = _runs(results, "O", 0, bench, size)
        if not l_runs or not o_runs:
            continue
        l_scores = _item_scores(l_runs)
        o_scores = _item_scores(o_runs)
        if not l_scores or not o_scores:
            continue
        l_accuracy = _mean(list(l_scores.values()))
        o_accuracy = _mean(list(o_scores.values()))
        delta = (l_accuracy - o_accuracy) * 100.0
        parts.append(f"{bench} {delta:+.1f} pp")
    if not parts:
        return Verdict(name="S5", passed=None, detail="sin datos")
    return Verdict(
        name="S5",
        passed=None,
        detail=(
            "informativo, sin umbral (reportar regresiones, incluida la visión): "
            + "; ".join(parts)
        ),
    )


def _harness_validity(
    prereg: Prereg, results: list[RunResult], size: str | None
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    tolerance = prereg.harness_tolerance_pts

    for bench, spec in prereg.benches.items():
        google = _google_value(prereg, size, bench)
        o_runs = _runs(results, "O", 0, bench, size)
        if google is None:
            check = harness_validity(None, [], tolerance)
            if spec.role == "confirmatory" and bench == "lcb":
                detail = (
                    "no Google figure by design: lcb is the post-cut S2 set; "
                    "use lcb_google for the full-v6 harness comparison"
                )
            else:
                detail = check.detail
        else:
            check = harness_validity(google, o_runs, tolerance)
            detail = check.detail

        if bench == "lcb_google":
            subgroup = _subgroup_item_scores(o_runs, "post_cut", True)
            if subgroup:
                accuracy = _mean(list(subgroup.values()))
                detail += (
                    f"; post-cut subgroup: mine={accuracy * 100:.1f}% "
                    f"(n={len(subgroup)} items), reported separately (not "
                    "compared to the full-v6 Google figure)"
                )

        verdicts.append(
            Verdict(
                name=f"harness:{bench}",
                passed=check.passed,
                detail=detail,
            )
        )

    # MMMU-Pro Overall is registered only as a derived Google figure.  Check
    # it from the two component runs on the questions they share.
    standard_keys = [b for b in ("mmmupro", "mmmu_pro_standard") if b in prereg.benches]
    vision_keys = [b for b in ("mmmupro_vision", "mmmu_pro_vision") if b in prereg.benches]
    if standard_keys and vision_keys:
        google = _google_value(prereg, size, "mmmu_pro_overall")
        standard_scores = _item_scores(
            [
                run
                for bench in standard_keys
                for run in _runs(results, "O", 0, bench, size)
            ]
        )
        vision_scores = _item_scores(
            [
                run
                for bench in vision_keys
                for run in _runs(results, "O", 0, bench, size)
            ]
        )
        if standard_scores and vision_scores:
            shared = sorted(set(standard_scores) & set(vision_scores))
            if shared:
                overall_scores = {
                    item_id: (
                        standard_scores[item_id] + vision_scores[item_id]
                    )
                    / 2
                    for item_id in shared
                }
            else:
                accuracy = (
                    _mean(list(standard_scores.values()))
                    + _mean(list(vision_scores.values()))
                ) / 2
                overall_scores = {
                    f"overall-{index}": accuracy
                    for index in range(len(standard_scores | vision_scores))
                }
            check = harness_validity_from_scores(
                google, overall_scores, tolerance
            )
        else:
            check = harness_validity_from_accuracy(google, None, 0, tolerance)
        verdicts.append(
            Verdict(
                name="harness:mmmu_pro_overall",
                passed=check.passed,
                detail=check.detail,
            )
        )

    return verdicts


# ---------------------------------------------------------------------------
# Result-set hygiene
# ---------------------------------------------------------------------------


def filter_by_tag(results: list[RunResult], tag: str | None) -> list[RunResult]:
    """Keep only runs written under ``tag`` (``None`` disables filtering)."""

    if tag is None:
        return list(results)
    return [run for run in results if run.prereg_tag == tag]


def duplicate_result_keys(
    results: list[RunResult], size: str | None = None
) -> list[tuple]:
    """Return full result identities seen more than once.

    The identity includes the loop block, bridge, adapter policy, LoRA target
    set and generation engine: re-running the same arm with a different block
    or engine is a different experiment, not a duplicate.
    """

    counts: dict[tuple, int] = defaultdict(int)
    for run in results:
        if size is not None and run.size.upper() != size.upper():
            continue
        counts[run.result_identity()] += 1
    return sorted(key for key, count in counts.items() if count > 1)


def validate_single_geometry(
    results: list[RunResult], size: str | None = None
) -> None:
    """Reject results that would pool different loop geometries.

    Criterion evaluation selects runs by ``(arm, N, bench)``; if two blocks (or
    bridges/policies/target sets) are present for the same selector, pooling
    them would silently mix experiments.  The prereg must pick one geometry.
    """

    keys: dict[tuple, set[tuple]] = defaultdict(set)
    for run in results:
        if size is not None and run.size.upper() != size.upper():
            continue
        selector = (
            run.variant.upper(),
            run.size.upper(),
            int(run.n_loops),
            run.bench,
        )
        keys[selector].add(run.geometry_key())
    mixed = {selector: geometries for selector, geometries in keys.items() if len(geometries) > 1}
    if mixed:
        formatted = "; ".join(
            f"(variant={variant}, size={run_size}, n_loops={n_loops}, "
            f"bench={bench}) -> {sorted(map(str, geometries))}"
            for (variant, run_size, n_loops, bench), geometries in sorted(mixed.items())
        )
        raise ValueError(
            "results mix loop geometries for the same evaluation selector; "
            "select one block/bridge/policy/target set before pooling: "
            + formatted
        )


def validate_single_engine(
    results: list[RunResult], size: str | None = None
) -> None:
    """Reject comparisons that mix generation engines.

    ``O``, ``O-H`` and ``L`` are compared only when every run in the comparison
    came from the same registered evaluation engine.  A padded/refill mix is a
    harness difference, not a model difference, and must be refused rather than
    silently pooled.  The check is across arms, not only within one selector:
    ``O=padded`` versus ``L=refill`` is the confound this rule exists for.
    """

    selected = [
        run
        for run in results
        if size is None or run.size.upper() == size.upper()
    ]
    by_engine: dict[str, set[str]] = defaultdict(set)
    for run in selected:
        by_engine[run.engine_key()].add(run.variant.upper())
    if len(by_engine) > 1:
        formatted = "; ".join(
            f"{engine}: {sorted(variants)}"
            for engine, variants in sorted(by_engine.items())
        )
        raise ValueError(
            "results mix generation engines across arms; one engine is "
            "required per comparison (O, O-H and L must share it): "
            + formatted
        )


def registered_seed_gaps(
    prereg: Prereg, results: list[RunResult], size: str
) -> list[str]:
    """List seed mismatches for each arm that is present in ``results``.

    Both missing registered seeds and extra unregistered seeds are rejected for
    the two multi-seed arms.  Only O-H and L have registered training-seed
    sets.  ``O-H-ext`` and ``L-plevec`` are single runs derived from the first
    seed of those sets, so they are intentionally not held to the full list.
    """

    size_key = size.lower()
    arm_keys = {
        "O-H": f"{size_key}_oh",
        "L": f"{size_key}_l",
    }
    gaps: list[str] = []
    for variant, key in arm_keys.items():
        registered = [int(seed) for seed in prereg.seeds.get(key, [])]
        if not registered:
            continue
        arm_runs = [
            run
            for run in results
            if run.size.upper() == size.upper()
            and run.variant.upper() == variant.upper()
        ]
        if not arm_runs:
            continue
        present = {int(run.seed) for run in arm_runs}
        missing = [seed for seed in registered if seed not in present]
        if missing:
            gaps.append(f"{key} missing seed(s) {missing}")
        extra = sorted(seed for seed in present if seed not in registered)
        if extra:
            gaps.append(f"{key} has unregistered seed(s) {extra}")
    return gaps


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_criteria(
    prereg: Prereg,
    results: list[RunResult],
    creative: dict[str, CreativeSummary] | None = None,
    agent: dict[str, AgentSummary] | None = None,
    size: str | None = None,
    flops_per_problem: Callable[[RunResult], float] | None = None,
    *,
    tag: str | None = None,
    require_seeds: bool = True,
) -> list[Verdict]:
    """Return the S1-S5 verdicts plus harness-validity checks for one size.

    ``size`` must be supplied when ``results`` mixes model sizes; otherwise the
    single size present is inferred.  Criteria are never combined across sizes,
    and every run in the comparison must share one generation engine.
    ``flops_per_problem`` is injected by Part B (or anyone with a FLOPs model);
    when it is ``None`` the S2 detail says ``"FLOPs: not wired"``.

    Results are filtered by ``tag`` (defaults to ``prereg.tag``; pass ``""`` to
    disable).  Duplicate full identities (variant, size, N, bench, seed, geometry)
    raise instead of being silently pooled, and mixed geometries for one
    selector are rejected.  ``require_seeds=True`` additionally
    requires every registered O-H/L training seed for the selected size and
    lists the missing ones in the error.
    """

    raise RuntimeError(
        "S1-S5 criteria are retired in prereg-v1.  The confirmatory family is "
        "E1 (precondition) -> E2 -> E3 -> E4; implement the E1-E4 analysis and "
        "tag it analysis-v1 before any trained-arm evaluation."
    )

    selected_tag = prereg.tag if tag is None else tag
    results = filter_by_tag(results, selected_tag or None)
    resolved_size = _resolve_size(results, size)

    duplicates = duplicate_result_keys(results, resolved_size)
    if duplicates:
        formatted = ", ".join(str(key) for key in duplicates)
        raise ValueError(
            "duplicate evaluation results for the same full identity "
            f"(variant, size, n_loops, bench, seed, geometry, engine): "
            f"{formatted}"
        )
    validate_single_geometry(results, resolved_size)
    validate_single_engine(results, resolved_size)
    if require_seeds and resolved_size is not None:
        gaps = registered_seed_gaps(prereg, results, resolved_size)
        if gaps:
            raise ValueError(
                "evaluation results are missing registered training seeds: "
                + "; ".join(gaps)
            )

    verdicts: list[Verdict] = []
    verdicts.append(_s1(prereg, results, resolved_size))

    # Confirmatory N = primary_n on the pre-registered benchmark.
    primary, _, _ = _s2_base(
        prereg, results, prereg.primary_n, resolved_size, flops_per_problem
    )
    verdicts.append(primary)

    # Exploratory N values: always labelled and Holm-corrected as one family.
    exploratory = []
    for n_loops in prereg.criteria.exploratory_ns:
        if n_loops == prereg.primary_n:
            continue
        verdict, mean, pvalue = _s2_base(
            prereg, results, n_loops, resolved_size, flops_per_problem
        )
        exploratory.append((n_loops, verdict, mean, pvalue))

    estimable = [entry for entry in exploratory if entry[3] is not None]
    rejected = holm([entry[3] for entry in estimable])
    decisions = iter(rejected)
    for n_loops, verdict, mean, pvalue in exploratory:
        if pvalue is None:
            verdicts.append(verdict)
            continue
        decision = next(decisions)
        if verdict.passed is not None:
            verdict.passed = bool(decision) and (mean or 0.0) > 0.0
            verdict.detail = (
                f"exploratory; Holm-corrected (p={pvalue:.4f}, "
                f"rejected={bool(decision)}); " + verdict.detail
            )
        verdicts.append(verdict)

    verdicts.append(_s3(prereg, creative))
    verdicts.append(_s4(prereg, agent))
    verdicts.append(_s5(prereg, results, resolved_size))
    verdicts.extend(_harness_validity(prereg, results, resolved_size))

    return verdicts
