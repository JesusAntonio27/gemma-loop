"""Pre-registration schema and retirement tests (prereg-v1 revision round 1).

The legacy S1-S5 criteria and the single-SFT ``train_tokens`` plan are retired:
the schema rejects those fields, ``check_criteria`` refuses to run and
``scripts/phase1_plan.py`` refuses to build the legacy plan.  The tests below
protect the new two-stage schema, the explicit bench roles, the E1-E4 family and
the MDE table.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gemmaloop.eval.stats import mde
from gemmaloop.prereg import (
    AUTHOR_DECLARATION_ENGLISH,
    Prereg,
    assert_tag_ready,
    check_criteria,
    criteria_benches,
    harness_benches,
    load_prereg,
    o_only_benches,
    pending_pin_fields,
    secondary_benches,
)


ROOT = Path(__file__).resolve().parents[1]


def _committed() -> Prereg:
    return load_prereg(ROOT / "prereg" / "prereg.yaml")


def _minimal() -> dict:
    return {
        "primary_n": 2,
        "loop_probs": [0.25, 0.40, 0.20, 0.15],
        "benches": {
            "lcb": {
                "limit": 10,
                "samples": 1,
                "seed": 0,
                "role": "confirmatory",
            }
        },
        "google": {},
        "training": {"max_length": 1024, "length_mode": "filter"},
    }


def _with(**overrides) -> dict:
    data = _minimal()
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Committed prereg
# ---------------------------------------------------------------------------


def test_committed_prereg_loads_and_registers_the_two_stage_design():
    prereg = _committed()

    assert prereg.tag == prereg.version == "prereg-v1"
    assert prereg.status == "draft"
    assert prereg.primary_n == 4
    assert prereg.loop_probs == [
        0.0,
        0.0,
        0.0,
        0.70,
        0.075,
        0.075,
        0.075,
        0.075,
    ]
    assert prereg.phase2 is None

    assert prereg.part1 is not None
    assert prereg.part1.size == "E2B"
    assert prereg.part1.seeds == [0, 1, 2]
    assert prereg.part1.confirmatory_seed == 0
    assert prereg.part1.extra_seeds == [1, 2]
    assert set(prereg.part1.arms) == {"O", "O-H", "L"}
    assert prereg.part1.diagnostic_arms == ["R", "R-damped"]
    assert "L-plevec" in prereg.part1.removed_arms
    assert prereg.part1.optional_arm == "O-H-RLext"
    assert "E2 passes" in prereg.part1.optional_arm_condition
    assert prereg.part1.rollout_n_policy
    assert "stratified" in prereg.part1.rollout_n_policy.lower()

    assert prereg.architecture is not None
    assert prereg.architecture.loop_block_candidates == [
        "25-29",
        "16-18",
        "15-29",
    ]
    assert prereg.architecture.bridge_candidates == [
        "zero_init_norm_matched_per_channel_g",
        "legacy_scalar",
    ]
    assert prereg.architecture.adapters_active_from_n == 2
    assert prereg.architecture.n_distribution == prereg.loop_probs
    assert prereg.architecture.adapter_sharing
    assert prereg.architecture.ple_mode == "shared"
    assert "loop_vector" in prereg.architecture.ple_mode_note
    assert "prelude" in prereg.architecture.frozen_prefix
    assert prereg.architecture.lora.target.startswith("All linear layers")
    assert prereg.architecture.bridge.legacy_scalar_w0 == 0.99

    assert prereg.dial_objective is not None
    assert "exit 1 = O" in prereg.dial_objective.exits
    assert prereg.dial_objective.window_trainable_passes == 3
    assert prereg.dial_objective.horizon == prereg.loop_probs
    assert "KL(O || exit_t)" in prereg.dial_objective.exit_loss
    assert prereg.dial_objective.monotone_hinge["lambda"] == 1.0
    assert "successive-exit KL" in prereg.dial_objective.excluded
    assert prereg.dial_objective.logging
    assert prereg.dial_objective.oh == "unchanged single exit at N=1"

    assert prereg.flops is not None
    assert prereg.flops.per_pass_fraction == {
        "25-29": 0.1425,
        "16-18": 0.0838,
        "15-29": 0.4275,
    }
    assert prereg.flops.r_n["25-29"]["4"] == pytest.approx(1.4275)
    assert prereg.flops.r_n["16-18"]["4"] == pytest.approx(1.2515)
    assert prereg.flops.r_n["15-29"]["4"] == pytest.approx(2.2824)
    assert "2.2824" in prereg.flops.r4_rule

    assert prereg.stage1 is not None
    assert prereg.stage1.teacher.beta_parent_kl == 0.1
    assert prereg.stage1.teacher.parent_kl_live is True
    assert prereg.stage1.teacher.top_k == 64
    assert "tail bucket" in prereg.stage1.teacher.top_k_tail
    assert prereg.stage1.teacher.vocab_mask_token == "<glitch_pruned_249788>"
    assert prereg.stage1.oh_runs_stage1 is True
    assert "same teacher" in prereg.stage1.oh_stage1_note
    assert prereg.stage1.budget_tokens == 12_000_000
    assert prereg.stage1.dose_logging_interval == 2_000_000
    assert "no decision use" in prereg.stage1.dose_logging_role
    assert prereg.stage1.optimizer == "AdamW"
    assert "8-bit" in prereg.stage1.recipe_binding
    assert "adamw_8bit" in prereg.stage1.recipe_binding
    assert "run.json" in prereg.stage1.recipe_binding
    assert prereg.stage1.rows_per_optimizer_step == 16
    # The old key stays accepted for backward compatibility but is not the
    # registered recipe.
    assert prereg.stage1.tokens_per_optimizer_step == 8192
    assert "rows_per_optimizer_step 16" in prereg.stage1.recipe_binding
    assert "2e-4" in prereg.stage1.recipe_binding
    assert "exit 2" in prereg.stage1.recipe_binding or "exits 2" in prereg.stage1.recipe_binding
    assert prereg.stage1.gradient_clip == 1.0
    assert prereg.stage1.warmup_fraction == pytest.approx(0.03)

    assert prereg.stage2 is not None
    assert prereg.stage2.algorithm == "Dr. GRPO"
    assert prereg.stage2.rollout_n is None
    assert "8 at N=4" in prereg.stage2.rollout_n_policy
    assert "1 at N=6" in prereg.stage2.rollout_n_policy
    assert "1 at N=8" in prereg.stage2.rollout_n_policy
    assert "3 groups at N=2, 3 at N=3" in prereg.stage2.rollout_n_policy
    assert "all G=8 rollouts" in prereg.stage2.rollout_n_policy
    assert "calibrated NLL" in prereg.stage2.dial_monitoring
    assert prereg.stage2.num_iterations == 1
    assert prereg.stage2.dr_grpo_normalizer == 8192
    assert prereg.stage2.grader_timeout_s == 6
    assert prereg.stage2.prompt_cap == 2048
    assert prereg.stage2.completion_cap == 8192
    assert prereg.stage2.prompt_cap + prereg.stage2.completion_cap == 10_240
    assert prereg.stage2.optimizer == "AdamW"
    assert prereg.stage2.beta_kl == 0.0
    assert prereg.stage2.tis_cap == 2.0
    assert prereg.stage2.fallback_rft.tokens == 12_000_000
    assert prereg.stage2.deepseek.model == "deepseek-flash"
    assert prereg.stage2.deepseek.served_version == "DeepSeek-V4.1-Flash-1017"
    assert prereg.stage2.deepseek.fingerprint_record.endswith(
        "deepseek_fingerprint.json"
    )
    assert "a reward in Part 1" in prereg.stage2.deepseek.never

    assert prereg.endpoints is not None
    assert prereg.endpoints.precondition.id == "E1"
    assert [endpoint.id for endpoint in prereg.endpoints.sequence] == [
        "E2",
        "E3",
        "E4",
    ]
    assert prereg.endpoints.primary == "E2"
    assert prereg.endpoints.heal_estimates.gate is False
    assert prereg.endpoints.heal_estimates.nll_nats == 0.01
    assert prereg.endpoints.bootstrap.unit == "problem cluster"
    assert prereg.endpoints.bootstrap.confirmatory_seed == 0
    assert "lower bound > 0" in prereg.endpoints.fixed_sequence_rule
    assert "one-sided alpha 0.02" in prereg.endpoints.fixed_sequence_rule
    assert "total training-plus-loop effect" in prereg.endpoints.sequence[1].test
    assert "N=4" in prereg.endpoints.sequence[0].test
    assert "N=4" in prereg.endpoints.sequence[1].test
    assert "2.2824" in prereg.endpoints.sequence[2].test
    assert "1.4275" in prereg.endpoints.sequence[2].test
    assert "1.2515" in prereg.endpoints.sequence[2].test
    assert "trapezoid" in prereg.endpoints.sequence[2].test
    assert prereg.endpoints.google_comparison == "reproduction check"
    assert "AIME 2026" in prereg.endpoints.o_only
    assert "graphical" in prereg.endpoints.multiplicity.lower()
    assert "D1 -> E2" in prereg.endpoints.multiplicity
    assert prereg.endpoints.dial_family[0].id == "D1"
    assert "Intersection-union" in prereg.endpoints.dial_family[0].test
    assert "0.005" in prereg.endpoints.dial_family[0].test
    assert any(item.startswith("D2") for item in prereg.endpoints.estimation)
    assert any(item.startswith("D3") for item in prereg.endpoints.estimation)
    assert any(item.startswith("D4") for item in prereg.endpoints.estimation)
    assert any(item.startswith("D5") for item in prereg.endpoints.estimation)
    assert any(item.startswith("D6") for item in prereg.endpoints.estimation)
    assert "N=2 (full LCB)" in prereg.endpoints.holm
    assert "N=3 (182" in prereg.endpoints.holm

    assert prereg.resident_bytes is not None
    assert "PLE" in " ".join(prereg.resident_bytes.components)
    assert "KV" in prereg.resident_bytes.excluded
    assert "numel" in prereg.resident_bytes.formula
    assert "B(L, N=k) = B(L, N=1)" in prereg.resident_bytes.invariants

    assert prereg.intelligence_per_byte is not None
    assert prereg.intelligence_per_byte.denominator == "resident_bytes"
    assert "pass@1" in prereg.intelligence_per_byte.quality["primary"]
    assert "nats/GiB" in prereg.intelligence_per_byte.derived["forbidden"]
    assert "phi_loop" in prereg.intelligence_per_byte.anchoring
    assert "phi_tot" in prereg.intelligence_per_byte.anchoring
    assert "rho" in prereg.intelligence_per_byte.anchoring["byte_multiplier"]
    assert "B_read(N, c)" in prereg.intelligence_per_byte.secondary[
        "bytes_read_per_token"
    ]
    assert prereg.intelligence_per_byte.d6.id == "D6"
    assert "estimation" in prereg.intelligence_per_byte.d6.status
    assert "cross-precision" in prereg.intelligence_per_byte.limitations

    assert prereg.pilot is not None
    assert prereg.pilot.runs_after_tag is True
    assert prereg.pilot.benchmarks is False
    assert prereg.pilot.p2.heal_tokens == 3_000_000
    assert len(prereg.pilot.p2.arms) == 11
    assert "O-H[25-29] + 12B_qat_q4_0" in prereg.pilot.p2.arms
    assert any("single_exit_legacy" in arm for arm in prereg.pilot.p2.arms)
    assert "verified-correct" in prereg.pilot.p2.verified_corpus
    assert "Teacher first" in prereg.pilot.p2.teacher_selection_rule
    assert "mean calibrated-NLL gain" in prereg.pilot.p2.teacher_gain_basis
    assert "Rule B" in prereg.pilot.p2.block_bridge_rule
    assert "Delta_bar_arm" in prereg.pilot.p2.block_bridge_rule
    assert "FILTER" in prereg.pilot.p2.block_bridge_rule
    assert "same block layers" in prereg.pilot.p2.block_bridge_teacher
    assert "E is empty" in prereg.pilot.p2.stop_rule
    assert "Delta_bar_arm" in prereg.pilot.p2.dial_rule
    assert "LoRA capacity is matched" in prereg.pilot.p2.oh_lora_target
    assert "itemized separately" in prereg.pilot.p2.oh_lora_target
    assert prereg.pilot.p2.eligible_set
    assert prereg.pilot.p2.ranking_statistic
    assert prereg.pilot.p2.monotonicity_filter
    assert prereg.pilot.p2.fallback_rule
    assert "cNLL at N = 1..4, 6, 8" in prereg.pilot.p2.dial_metrics_reported
    assert "84c5da0" in prereg.pilot.p2.frozen_rule_deviation
    assert "4-row GPU smoke" in prereg.pilot.p2.frozen_rule_deviation
    assert "single-exit legacy" in prereg.pilot.p2.anytime_tax_arm
    assert "never eligible" in prereg.pilot.p2.anytime_tax_arm
    assert prereg.pilot.p1.r_rdamped_n_range
    assert prereg.pilot.cost_gpu_hours.startswith("43-52")
    assert prereg.pilot.p2.min_arm_vs_oh_nats == -0.02
    assert prereg.pilot.p2.tie_nats == 0.003
    assert prereg.pilot.p3.steps == 30
    assert prereg.pilot.p3.completion_cap == 4096
    assert prereg.pilot.p3.prompt_median_max == 2500
    assert prereg.pilot.p3.fallback == "RFT on verified self-traces"

    assert prereg.data is not None
    assert prereg.data.max_length == prereg.training.max_length == 8192
    assert prereg.data.length_mode == prereg.training.length_mode == "filter"
    assert prereg.data.pool.generated_tokens == 64_000_000
    assert "13-gram overlap" in prereg.data.decontamination
    assert "sha256(problem_id)" in prereg.data.splits.split_rule
    assert "finish_reason = stop" in prereg.data.stage0_sample_filter
    assert prereg.data.manifest_addendum

    assert prereg.compute is not None
    assert prereg.compute.tiers == [
        "local",
        "single_cloud_gpu",
        "multi_gpu",
        "tpu",
    ]
    assert len(prereg.compute.hours) >= 6
    assert prereg.compute.evaluation_priority
    assert "Modal" in prereg.compute.stage0_default
    assert "SOLE OPERATIVE" in prereg.compute.local_engine
    assert "no refill run is registered" in prereg.compute.local_engine
    assert "eed7fa7" in prereg.compute.local_engine
    assert "padded adaptive-batch HF engine" in prereg.compute.local_engine
    assert "vLLM" in prereg.compute.cloud_engine
    assert prereg.compute.eval_engine_default == "padded adaptive-batch HF engine"
    assert prereg.compute.refill_engine_branch == "fix/gate-v6"
    assert prereg.compute.refill_engine_commit == "eed7fa7ff6dbb87cc66548d1368ece9621cc5419"
    assert "chosen-token regret" in prereg.compute.eval_engine_gate
    assert "N in {1,2,3,4,6,8}" in prereg.compute.eval_engine_gate
    assert prereg.compute.stage0_provider == "Modal"
    assert prereg.compute.stage0_vllm_version == "vllm==0.30.0"
    assert prereg.compute.stage0_usage_limit_usd == 30.0
    assert prereg.compute.stage0_spend_limit_usd == 0.0
    assert "engine recorded per row" in prereg.compute.stage0_engine_record
    assert "602-941" in prereg.compute.local_seed0_total_hours
    assert "237-383" in prereg.compute.local_seed0_total_hours
    assert any("L(N=4)" in item for item in prereg.compute.evaluation_priority)
    assert any("D2:" in item for item in prereg.compute.evaluation_priority)
    assert any("D6:" in item for item in prereg.compute.evaluation_priority)

    assert prereg.pins is not None
    assert prereg.pins.e2b_snapshot.startswith("4adf157")
    assert prereg.pins.t_dial_bench
    assert "1.0 GPU-h" in prereg.pins.t_dial_bench
    assert prereg.pins.bench_addendum
    assert prereg.pins.addendum

    assert prereg.part2 is not None
    assert prereg.part2.starting_checkpoint == "L-s0"
    assert prereg.part2.dev_composite
    assert prereg.part2.milestone_scoring_cap
    assert "controller's recommendation" in prereg.part2.automatic_checks_note
    assert "docs/prereg-v1-decision-log.md" in prereg.part2.decision_log_export
    assert "auto" in prereg.part2.serve_map
    assert any("adaptive-decode accuracy" in item for item in prereg.part2.milestones)
    assert any(
        "precision-matched 4-bit" in item for item in prereg.part2.techniques
    )

    assert len(prereg.pending_author) == 8
    joined_pending = " ".join(prereg.pending_author).lower()
    assert "final ok" in joined_pending
    assert "stage a" in joined_pending
    assert "closed" in joined_pending
    assert "bench" in joined_pending
    assert "teacher smoke" in joined_pending
    assert "download" in joined_pending
    assert "E3" in prereg.pending_author[0]
    assert "N=4" in prereg.pending_author[0]
    assert len(prereg.drafter_choices) == 52
    assert len(prereg.deviations) == 2
    assert "D-7b" in prereg.deviations[0]
    assert "38b5bd2" in prereg.deviations[0]
    assert "Stage A content is not an input" in prereg.deviations[0]
    assert prereg.analysis_v1_obligations
    assert any("S1-S5" in item for item in prereg.analysis_v1_obligations)
    assert any("D1" in item for item in prereg.analysis_v1_obligations)
    assert any("resident-bytes" in item for item in prereg.analysis_v1_obligations)


def test_committed_bench_roles_are_explicit_and_complete():
    prereg = _committed()

    assert prereg.benches["lcb"].role == "confirmatory"
    assert prereg.benches["lcb"].limit == 1055
    assert prereg.benches["lcb"].samples == 1
    assert prereg.benches["lcb_google"].role == "harness"
    assert prereg.benches["lcb_google"].generated is False
    assert prereg.benches["lcb_google"].source_bench == "lcb"
    assert prereg.benches["gpqa"].role == "secondary"
    assert prereg.benches["mmlupro"].role == "secondary"
    assert prereg.benches["aime"].role == "o_only"
    assert prereg.benches["spatialeval_vqa"].role == "o_only"
    assert prereg.benches["cvbench_2d"].role == "o_only"
    assert prereg.google["E4B"]["lcb_google"] == 52.0

    assert criteria_benches(prereg) == ["lcb"]
    assert "gpqa" in secondary_benches(prereg)
    assert "aime" in o_only_benches(prereg)
    assert harness_benches(prereg) == ["lcb_google"]


def test_mde_table_recomputes_from_stats_mde():
    prereg = _committed()

    assert prereg.mde_table
    for row in prereg.mde_table:
        expected = round(mde(row.n, row.k, alpha=row.alpha) * 100.0, 2)
        assert row.mde_pts == pytest.approx(expected, abs=0.011), row.bench
    alpha_row = next(
        row for row in prereg.mde_table if row.alpha != 0.05
    )
    assert alpha_row.n == 1055
    assert alpha_row.alpha == pytest.approx(0.04)
    assert alpha_row.mde_pts == pytest.approx(5.75, abs=0.011)
    assert prereg.mde_notes
    assert "D2" in prereg.mde_notes


# ---------------------------------------------------------------------------
# Schema retirement and validation
# ---------------------------------------------------------------------------


def test_legacy_criteria_and_train_tokens_are_rejected():
    with pytest.raises(ValidationError):
        Prereg.model_validate(
            _with(criteria={"s2_bench": "lcb", "exploratory_ns": [3, 4]})
        )
    with pytest.raises(ValidationError):
        Prereg.model_validate(_with(train_tokens={"e2b": 12_000_000}))


def test_bench_role_has_no_confirmatory_default():
    data = _minimal()
    data["benches"] = {"lcb": {"limit": 10, "samples": 1, "seed": 0}}
    with pytest.raises(ValidationError):
        Prereg.model_validate(data)

    data["benches"] = {
        "lcb": {
            "limit": 10,
            "samples": 1,
            "seed": 0,
            "role": "not-a-role",
        }
    }
    with pytest.raises(ValidationError):
        Prereg.model_validate(data)


def test_minimal_prereg_still_loads():
    prereg = Prereg.model_validate(_minimal())
    assert prereg.primary_n == 2
    assert prereg.part1 is None
    assert prereg.endpoints is None


def test_new_sections_reject_inconsistent_values():
    # N>=2-only adapters cannot train N=1 batches.
    with pytest.raises(ValidationError, match="loop_probs\\[0\\]"):
        Prereg.model_validate(
            _with(
                architecture={
                    "loop_block_candidates": ["25-29"],
                    "bridge_candidates": ["legacy_scalar"],
                    "adapters_active_from_n": 2,
                }
            )
        )

    # The fixed sequence is exactly E2..E4, with E1 as the precondition.
    with pytest.raises(ValidationError, match="endpoints.sequence"):
        Prereg.model_validate(
            _with(
                endpoints={
                    "precondition": {"id": "E1", "name": "x", "test": "x"},
                    "sequence": [
                        {"id": "E1", "name": "x", "test": "x"},
                        {"id": "E2", "name": "x", "test": "x"},
                    ],
                }
            )
        )

    # beta must be strictly positive and O-H must get the real Stage 1.
    with pytest.raises(ValidationError, match="beta_parent_kl"):
        Prereg.model_validate(
            _with(
                stage1={
                    "teacher": {
                        "candidates": ["google/gemma-4-E4B-it"],
                        "beta_parent_kl": 0.0,
                    }
                }
            )
        )
    with pytest.raises(ValidationError, match="oh_runs_stage1"):
        Prereg.model_validate(_with(stage1={"oh_runs_stage1": False}))

    # The RL rollout N is the registered primary N, with one inner iteration.
    with pytest.raises(ValidationError, match="stage2.rollout_n"):
        Prereg.model_validate(_with(stage2={"rollout_n": 1}))
    with pytest.raises(ValidationError, match="num_iterations"):
        Prereg.model_validate(_with(stage2={"num_iterations": 2}))

    # P2 must list the 7-arm matrix.
    with pytest.raises(ValidationError, match="pilot.p2.arms"):
        Prereg.model_validate(
            _with(pilot={"p2": {"arms": ["O-H"]}})
        )

    # Compute needs an hours table and an evaluation priority.
    with pytest.raises(ValidationError, match="compute.hours"):
        Prereg.model_validate(
            _with(compute={"tiers": ["local"], "evaluation_priority": ["E2"]})
        )

    # Split buckets must be disjoint.
    with pytest.raises(ValidationError, match="disjoint"):
        Prereg.model_validate(
            _with(
                training={"max_length": 8192, "length_mode": "filter"},
                data={
                    "splits": {
                        "train_bucket": "0-49",
                        "pilot_dev_bucket": "0-49",
                        "nll_kl_bucket": "60-69",
                        "long_traces_bucket": "70-79",
                        "heldout_test_bucket": "80-99",
                    }
                },
            )
        )

    # MDE rows must be estimable.
    with pytest.raises(ValidationError, match="mde_table"):
        Prereg.model_validate(
            _with(mde_table=[{"bench": "lcb", "n": 0, "k": 1, "mde_pts": 5.0}])
        )


def test_dial_schema_accepts_eight_loop_counts_and_legacy_four():
    data = _minimal()
    data["primary_n"] = 4
    data["loop_probs"] = [0.0, 0.0, 0.0, 0.7, 0.075, 0.075, 0.075, 0.075]
    dial = Prereg.model_validate(data)
    assert dial.primary_n == 4
    assert len(dial.loop_probs) == 8

    legacy = Prereg.model_validate(_minimal())
    assert legacy.primary_n == 2
    assert len(legacy.loop_probs) == 4


def test_dial_schema_rejects_inconsistent_values():
    dial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5]

    with pytest.raises(ValidationError, match="ple_mode"):
        Prereg.model_validate(
            _with(
                primary_n=5,
                loop_probs=dial,
                architecture={"ple_mode": "loop_vector"},
            )
        )

    with pytest.raises(ValidationError, match="loop_probs"):
        Prereg.model_validate(_with(loop_probs=[0.0, 0.5, 0.5]))

    with pytest.raises(ValidationError, match="rollout_n_policy"):
        Prereg.model_validate(_with(stage2={}))

    with pytest.raises(ValidationError, match="rollout_n_policy"):
        Prereg.model_validate(_with(part1={}))

    bad_multiplicity = {
        "dial_family": [{"id": "D1", "name": "x", "test": "x"}],
        "estimation": ["D2"],
        "multiplicity": "",
    }
    with pytest.raises(ValidationError, match="multiplicity"):
        Prereg.model_validate(_with(endpoints=bad_multiplicity))

    with pytest.raises(ValidationError, match="dial_family"):
        Prereg.model_validate(
            _with(
                endpoints={
                    "multiplicity": "graphical",
                    "estimation": ["D2"],
                    "dial_family": [],
                }
            )
        )


def test_check_criteria_is_retired():
    prereg = _committed()
    with pytest.raises(RuntimeError, match="retired"):
        check_criteria(prereg, [])


def test_committed_endpoint_fields_are_complete():
    prereg = _committed()
    endpoints = prereg.endpoints

    assert endpoints.precondition.id == "E1"
    assert "bit-identical" in endpoints.precondition.test
    assert endpoints.precondition.margin == "no alpha, no gate"
    assert endpoints.precondition.bootstrap

    assert [endpoint.id for endpoint in endpoints.sequence] == ["E2", "E3", "E4"]
    for endpoint in endpoints.sequence:
        assert endpoint.name
        assert endpoint.test
        assert endpoint.margin
        assert endpoint.bootstrap
    assert "one-sided alpha 0.02" in endpoints.sequence[0].margin
    assert "N=4" in endpoints.sequence[0].test
    assert "total training-plus-loop effect" in endpoints.sequence[1].test
    assert "N=4" in endpoints.sequence[1].test
    assert "1024 * 2^j" in endpoints.sequence[2].test
    assert "floor(B / r_4)" in endpoints.sequence[2].test
    assert "2.2824" in endpoints.sequence[2].test
    assert "1.4275" in endpoints.sequence[2].test
    assert "1.2515" in endpoints.sequence[2].test
    assert "trapezoid" in endpoints.sequence[2].test
    assert "32,768" in endpoints.sequence[2].test
    assert endpoints.multiplicity
    assert endpoints.graphical["weights"] == {"E2": 0.8, "E3": 0.0, "E4": 0.0, "D1": 0.2}
    assert endpoints.graphical["transitions"]["D1"] == "E2"
    assert len(endpoints.graphical["operational_order"]) == 3
    assert endpoints.dial_family
    assert endpoints.estimation

    assert endpoints.primary == "E2"
    assert endpoints.heal_estimates.nll_nats == 0.01
    assert endpoints.heal_estimates.lcb_pts == 3.0
    assert endpoints.heal_estimates.ci_level == 0.90
    assert endpoints.heal_estimates.gate is False

    bootstrap = endpoints.bootstrap
    assert bootstrap.iters == 10_000
    assert bootstrap.seed == 0
    assert bootstrap.unit == "problem cluster"
    assert bootstrap.confirmatory_seed == 0
    assert bootstrap.extra_seed_role
    assert bootstrap.single_seed
    assert bootstrap.multi_seed
    assert bootstrap.sensitivity

    assert endpoints.fixed_sequence_rule
    assert endpoints.holm
    assert endpoints.single_seed_statement
    assert endpoints.google_comparison == "reproduction check"
    assert endpoints.secondary
    assert endpoints.o_only
    assert endpoints.priority_order


def test_committed_loop_ladder_is_descriptive():
    prereg = _committed()
    ladder = prereg.loop_ladder

    assert ladder is not None
    assert ladder.role == "descriptive/estimation only"
    assert ladder.runs_after_tag is True
    assert "never" in ladder.selection_use
    assert ladder.chosen_block == "pilot-chosen block"
    assert ladder.r_rule
    assert "1 - 1/N" in ladder.r_damped_rule
    assert "0.5" in ladder.r_damped_weight
    assert "0.333" in ladder.r_damped_weight
    assert "0.25" in ladder.r_damped_weight
    assert "0.99" not in ladder.r_damped_weight

    ids = {point.id for point in ladder.points}
    assert {
        "O",
        "R-N2",
        "R-N3",
        "R-N4",
        "R-damped-N2",
        "R-damped-N3",
        "R-damped-N4",
        "L-heal-N2",
        "L-final-N2",
        "L-final-N3",
        "L-final-N4",
        "O-H-heal-N1",
        "O-H-final-N1",
    } <= ids
    assert ladder.items == "LCB post-cut 182 x 1 + GPQA 198 x 1 per ladder point"
    assert ladder.full_suite_at == "L final N=4"
    assert ladder.oh_control
    assert "gch" in ladder.untrained_bridge_note
    assert "D1/D5" in ladder.dial_nll_at_heal
    assert "D2" in ladder.reused_from_d2
    assert ladder.gpu_hours.startswith("≈37")
    assert "less than 37" in ladder.gpu_hours_with_reuse
    assert "never for E3" in ladder.reuse_rule
    assert "acc_vs_n" in ladder.chart
    assert "difference-in" in ladder.did_note


def test_committed_training_free_campaign_disclosure():
    prereg = _committed()
    campaign = prereg.training_free_campaign

    assert campaign is not None
    assert campaign.before_tag is True
    assert campaign.status == "exploratory"
    assert campaign.reported == "full"
    assert campaign.selection_input is False
    assert "frozen" in campaign.frozen_candidates
    assert campaign.p1_lite.problems == 120
    assert "verified-trace" in campaign.p1_lite.corpus
    assert "calibrated-NLL" in campaign.p1_lite.sweep
    assert "LCB post-cut 182" in campaign.accuracy_n2
    assert "N=3,4" in campaign.then_n34
    assert "disjoint" in campaign.corpus_disjointness
    assert "Stage 0" in campaign.corpus_disjointness
    assert campaign.budget
    assert campaign.chart
    assert "dial requirement" in campaign.dial_requirement_disclosure
    assert "training objective" in campaign.dial_requirement_disclosure
    assert "4-row GPU smoke" in campaign.seen_at_rule_change
    assert "R +0.133" in campaign.seen_at_rule_change
    assert "rule B" in campaign.rule_change_disclosure
    assert "84c5da0" in campaign.rule_change_disclosure


def test_loop_ladder_and_training_free_validators_reject_bad_values():
    with pytest.raises(ValidationError, match="loop_ladder.role"):
        Prereg.model_validate(
            _with(loop_ladder={"role": "confirmatory"})
        )
    with pytest.raises(ValidationError, match="r_damped_rule"):
        Prereg.model_validate(
            _with(
                loop_ladder={
                    "points": [
                        {"id": "O", "arm": "O", "n": 1, "stage": "reference"}
                    ]
                }
            )
        )
    with pytest.raises(ValidationError, match="selection_input"):
        Prereg.model_validate(
            _with(training_free_campaign={"selection_input": True})
        )
    with pytest.raises(ValidationError, match="before_tag"):
        Prereg.model_validate(
            _with(training_free_campaign={"before_tag": False})
        )
    with pytest.raises(ValidationError, match="p1_lite.problems"):
        Prereg.model_validate(
            _with(
                training_free_campaign={
                    "frozen_candidates": "x",
                    "frozen_at_commit": "abc",
                    "frozen_at_timestamp": "2026-11-04T22:19:00-07:00",
                    "reuse_rule": "ladder only",
                    "stage_b_gate": "wait for E3",
                    "corpus_disjointness": "y",
                    "p1_lite": {"problems": 0},
                }
            )
        )


def test_committed_pins_and_round3_recipes_are_filled():
    prereg = _committed()
    pins = prereg.pins

    assert pins.e2b_template_sha256 == (
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
    )
    assert pins.teacher_e4b_revision == "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    assert pins.teacher_12b_qat_revision == (
        "b6ed86275a6a5735884e208bfed95b445a684ca2"
    )
    assert pins.tokenizer_sha256 == {
        "google/gemma-4-E2B-it": (
            "ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2"
        ),
        "google/gemma-4-E4B-it": (
            "91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc"
        ),
        "google/gemma-4-12B-it-qat-q4_0-unquantized": (
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
        ),
    }
    assert pins.lcb_runner == "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
    assert "math-verify 0.9.0" in pins.math_verify
    assert pins.manifest_sha256 is None
    assert "prereg-v1-data" in pins.data_addendum
    assert "T-dial-bench" in pins.t_dial_bench
    assert "N=6 and N=8" in pins.t_dial_bench
    assert "pins addendum" in pins.bench_addendum
    assert "no endpoint" in pins.bench_addendum
    assert "INACTIVE" in pins.engine_addendum
    assert "no counted engine-gate" in pins.engine_addendum.lower()
    assert pins.merged_feat_commit == (
        "a7c06331a4a381c20bf8af5a01f0f39fc57e71b6"
    )
    assert prereg.data.manifest_sha256 is None
    assert "prereg-v1-data" in prereg.data.manifest_addendum

    assert prereg.architecture.lora.target_option == "loop_block"
    assert "not on the coda" in prereg.architecture.bridge.scope
    assert "assistant tokens only" in prereg.stage1.loss_tokens
    assert "nf4" in prereg.stage1.numerics
    assert "multi-exit CE" in prereg.stage2.fallback_rft.loss
    assert prereg.stage2.fallback_rft.loop_probs == prereg.loop_probs
    assert "padded adaptive-batch HF engine" in prereg.stage2.engine
    assert "fixed per engine batch" in prereg.stage2.engine
    assert "N=2, 3, 4, 6 and 8" in prereg.stage2.engine
    assert "five per-N batches" in prereg.stage2.engine

    splits = prereg.data.splits
    assert splits.pilot_p3_bucket == "50-53"
    assert splits.pilot_selection_bucket == "54-56"
    assert splits.pilot_tfit_bucket == "57-58"
    assert splits.pilot_p1_bucket == "59"
    assert splits.stage0_prompt_bucket == "0-49"
    assert splits.pilot_p3_problems == 200
    assert splits.pilot_selection_problems == 150
    assert splits.pilot_tfit_problems == 100
    assert splits.pilot_p1_problems == 50
    assert prereg.pilot.p3.prompt_bucket == "Stage 0 train bucket 0-49"


def test_committed_freeze_and_engine_hours_are_registered():
    prereg = _committed()
    campaign = prereg.training_free_campaign

    assert campaign.frozen_at_commit == (
        "84c5da0f1778cb7446d20bed0cd428c5799ea1b1"
    )
    assert campaign.frozen_at_timestamp == "2026-11-04T22:39:04-07:00"
    assert campaign.no_output_at_freeze is True

    compute = prereg.compute
    assert compute.eval_engine_default == "padded adaptive-batch HF engine"
    assert compute.refill_engine_branch == "fix/gate-v6"
    # eed7fa7 stays as historical gate-v5 provenance; no repaired smoke ran.
    assert compute.refill_engine_commit == "eed7fa7ff6dbb87cc66548d1368ece9621cc5419"
    assert "No repaired-smoke commit exists" in compute.repaired_engine_pin
    assert compute.merged_feat_commit == (
        "a7c06331a4a381c20bf8af5a01f0f39fc57e71b6"
    )
    assert "chosen-token regret" in compute.eval_engine_gate
    assert "N in {1,2,3,4,6,8}" in compute.eval_engine_gate
    assert "INACTIVE" in compute.eval_engine_gate
    assert "NOT run" in compute.eval_engine_gate

    for row in compute.hours:
        assert row.padded_hours
        assert row.refill_hours
    stage0 = next(row for row in compute.hours if row.stage == "Stage 0 sampling")
    assert "Modal" in stage0.cloud_hours
    assert "0 (Modal)" in stage0.padded_hours
    assert "0 (Modal)" in stage0.refill_hours
    pilot = next(row for row in compute.hours if row.stage == "Pilot (P1 + P2 + P3)")
    assert pilot.padded_hours == "43-52"
    total = next(row for row in compute.hours if row.stage.startswith("Total seed 0"))
    assert total.padded_hours == "602-941"
    assert total.refill_hours == "237-383"
    stage2 = next(
        row for row in compute.hours
        if row.stage == "Stage 2 RLVR L (seed 0), 25-29"
    )
    assert stage2.padded_hours == "102-153"
    assert stage2.refill_hours == "45-91"
    assert any(row.stage.startswith("Pre-tag benchmarks") for row in compute.hours)
    assert any(row.stage.startswith("Engine registration") for row in compute.hours)


# ---------------------------------------------------------------------------
# Revision round 6 (native review of round 5)
# ---------------------------------------------------------------------------


def test_round6_rule_b_is_a_complete_decision_rule():
    prereg = _committed()
    p2 = prereg.pilot.p2

    assert "frozen 84c5da0 eligibility" in p2.eligible_set
    assert "delta_arm" in p2.eligible_set
    assert "Delta_bar_arm" in p2.ranking_statistic
    assert "cNLL(O-H[block]) - cNLL(L,N)" in p2.ranking_statistic
    assert "filter" in p2.monotonicity_filter.lower()
    assert "unfiltered" in p2.fallback_rule.lower() or "without the filter" in p2.fallback_rule
    assert "E is empty" in p2.stop_rule
    assert "LoRA capacity is matched" in p2.oh_lora_target
    # One tie-break list, matching the YAML and the MD.
    assert p2.tie_break == [
        "lower FLOPs ratio at N=4",
        "zero-init per-channel bridge over legacy scalar",
        "smaller loop block",
    ]
    # The old heal-confounded statistic is no longer the rule.
    assert not p2.ranking_statistic.startswith("cNLL(1)")
    assert "replaces the heal-confounded" in p2.ranking_statistic


def test_round6_stage_a_is_sealed_and_stage_b_waits_for_the_tag():
    prereg = _committed()
    campaign = prereg.training_free_campaign

    assert campaign.stage_a_sealing
    paths = {entry["path"] for entry in campaign.stage_a_sealing}
    assert "results/trainfree/stageA/n1.json" in paths
    assert "results/trainfree/stageA/n2.json" in paths
    assert "runs/trainfree/plan.json" in paths
    for entry in campaign.stage_a_sealing:
        assert entry["mtime"]
        assert entry["size_bytes"] >= 0
        assert len(entry["sha256"]) == 64
    assert "sha256" in campaign.stage_a_sealing_procedure
    assert "post-completion" in campaign.stage_a_sealing_procedure
    assert "NOT launched before" in campaign.stage_b_gate
    assert "prereg-v1 tag" in campaign.stage_b_gate
    assert "n1.json" in campaign.seen_at_rule_change
    assert "02:58:11" in campaign.decisions_with_n2_on_disk
    assert "native evaluator" in campaign.exposure_audit


def test_round6_graphical_procedure_is_fully_specified():
    graphical = _committed().endpoints.graphical

    assert graphical["total_one_sided_alpha"] == 0.025
    assert graphical["weights"] == {"E2": 0.8, "E3": 0.0, "E4": 0.0, "D1": 0.2}
    assert graphical["transitions"] == {
        "E2": "E3",
        "E3": "E4",
        "E4": "D1",
        "D1": "E2",
    }
    assert "Bretz" in graphical["method"]
    assert "10,000" in graphical["p_values"]
    assert "max" in graphical["p_values"]
    assert len(graphical["operational_order"]) == 3
    assert "order" in graphical["decision_rule"].lower()
    fixed = _committed().endpoints.fixed_sequence_rule
    assert "receives alpha only after" in fixed
    assert "does not depend" not in fixed  # that lives in graphical.decision_rule


def test_round6_engine_gate_and_part_b_are_synced_to_feat():
    prereg = _committed()
    gate = prereg.engine_gate
    part_b = prereg.part_b

    assert prereg.compute.refill_engine_commit == "eed7fa7ff6dbb87cc66548d1368ece9621cc5419"
    assert prereg.compute.merged_feat_commit == (
        "a7c06331a4a381c20bf8af5a01f0f39fc57e71b6"
    )
    assert prereg.pins.merged_feat_commit == (
        "a7c06331a4a381c20bf8af5a01f0f39fc57e71b6"
    )
    assert gate is not None and gate.runs_after_tag is False
    assert "INACTIVE" in gate.status
    assert set(gate.sampling_n) >= {2, 3, 4, 6, 8}
    assert set(gate.throughput_n) >= {1, 4}
    n_values = {n for combo in gate.combinations for n in combo.n_loops}
    assert n_values == {1, 2, 3, 4, 6, 8}
    bridges = {combo.bridge for combo in gate.combinations if combo.arm == "L"}
    assert bridges == {"gch", "scalar", "none", "damped"}
    assert gate.noise_floor_control
    assert gate.re_prefill_smoke
    assert "mixed-history" in gate.coverage_note
    assert part_b is not None
    assert part_b.loops == prereg.primary_n == 4
    assert part_b.requests >= 3 * part_b.eval_batch
    assert part_b.candidates == 360
    assert "DAPO" in part_b.candidate_pool
    assert "INACTIVE" in part_b.status
    assert "INACTIVE" in prereg.compute.eval_engine_gate
    assert "no counted engine-gate" in prereg.pins.engine_addendum.lower()


def test_round6_rollout_groups_and_fallback_are_structured():
    prereg = _committed()
    assert prereg.part1.rollout_n_groups == {2: 3, 3: 3, 4: 8, 6: 1, 8: 1}
    assert prereg.stage2.rollout_n_groups == {2: 3, 3: 3, 4: 8, 6: 1, 8: 1}
    assert sum(prereg.stage2.rollout_n_groups.values()) == prereg.stage2.prompts_per_step
    assert prereg.stage2.bptt_window
    assert "k=3" in prereg.stage2.bptt_window
    assert prereg.stage2.fallback_rft.loop_probs == prereg.loop_probs

    with pytest.raises(ValidationError, match="rollout_n_groups"):
        Prereg.model_validate(
            _with(
                part1={
                    "size": "E2B",
                    "seeds": [0],
                    "rollout_n_policy": "stratified",
                    "rollout_n_groups": {2: 8, 3: 9},
                },
                analysis_v1_obligations=["test"],
            )
        )


def test_round6_hours_e4_and_d6_are_corrected():
    prereg = _committed()
    rows = {row.stage: row for row in prereg.compute.hours}

    assert rows["Stage 2 RLVR L (seed 0), 25-29"].padded_hours == "102-153"
    assert rows["Stage 2 RLVR L (seed 0), 16-18"].padded_hours == "95-141"
    assert rows["Stage 2 RLVR L (seed 0), 15-29"].padded_hours == "128-191"
    assert any(stage.startswith("Pre-tag benchmarks") for stage in rows)
    assert any(stage.startswith("Engine registration") for stage in rows)
    assert "602-941" in prereg.compute.local_seed0_total_hours
    assert "596-930" in prereg.compute.local_seed0_total_hours
    assert "654-1,016" in prereg.compute.local_seed0_total_hours
    # E4 is defined on O-H's budget and both arms stay under the cap.
    e4 = prereg.endpoints.sequence[2].test
    assert "floor(B / r_4)" in e4
    assert "no capped arm" in e4
    assert "2.2824" in e4 and "1.4275" in e4 and "1.2515" in e4

    # D6 is in GiB, has both phi versions, and does not divide differences by GiB.
    rb = prereg.resident_bytes
    assert rb.unit == "GiB"
    assert "prelude" in " ".join(rb.components)
    assert "L unmerged is the byte-minimal" in rb.merged
    assert "W_block" in rb.merged
    assert "8.657 GiB" in rb.expected_bf16["O_E2B_and_L_N1"]
    ipb = prereg.intelligence_per_byte
    assert "phi_loop" in ipb.anchoring and "phi_tot" in ipb.anchoring
    assert "There is no per-GiB division" in ipb.fixed_bytes_effects["primary_claim"]
    assert "labelled annotation" in ipb.anchoring["google_role"]
    assert "GiB" in ipb.secondary["bytes_read_per_token"]


def test_round6_d4_gaps_are_closed():
    d4 = next(
        item for item in _committed().endpoints.estimation if item.startswith("D4")
    )
    assert "N_eff = 2, 3 and 4" in d4
    assert "delta cNLL = cNLL(fixed N) - cNLL(policy)" in d4
    assert "arm, checkpoint, exit-N" in d4
    assert "no adaptive winner" in d4
    assert "fixed N=4" in d4


def test_round6_ladder_has_primary_n_heal_points_and_derived_full_suite():
    prereg = _committed()
    ladder = prereg.loop_ladder
    ids = {point.id for point in ladder.points}
    assert {"L-heal-N2", "L-heal-N3", "L-heal-N4"} <= ids
    assert ladder.full_suite_at == f"L final N={prereg.primary_n}"
    assert "primary N=4" in ladder.dial_nll_at_heal or "primary N" in ladder.dial_nll_at_heal
    assert "view of the full N=2" in ladder.reused_from_d2
    assert "charged in the evaluation row" in ladder.gpu_hours


def test_round6_pilot_counts_match_the_split_table():
    prereg = _committed()
    assert prereg.pilot.data.pilot_dev_problems == prereg.data.splits.pilot_dev_problems
    assert (
        prereg.pilot.data.temperature_fit_problems
        == prereg.data.splits.pilot_tfit_problems
    )
    assert prereg.pilot.data.temperature_fit_problems == 100
    assert prereg.pilot.data.pilot_dev_problems == 500


# ---------------------------------------------------------------------------
# Revision round 7 (engine pin, pilot O-H arms, stage A visibility, hours)
# ---------------------------------------------------------------------------


def test_round7_engine_pin_and_hours_are_one_consistent_set():
    prereg = _committed()
    compute = prereg.compute
    gate = prereg.engine_gate
    part_b = prereg.part_b

    assert compute.refill_engine_commit == "eed7fa7ff6dbb87cc66548d1368ece9621cc5419"
    assert compute.merged_feat_commit == (
        "a7c06331a4a381c20bf8af5a01f0f39fc57e71b6"
    )
    assert "fix/gate-v6" in prereg.pins.engine_candidate_commit
    assert "eed7fa7" in prereg.pins.engine_candidate_historical
    assert "efd0145" in prereg.pins.never_pin_commits
    assert "efd0145" in compute.eval_engine_gate
    assert "never pin or bisect" in compute.eval_engine_gate
    assert "re-prefill smoke" in compute.eval_engine_gate
    assert "Part B" in compute.eval_engine_gate

    assert compute.engine_gate_hours.startswith("4.5-6.0")
    assert compute.part_b_hours.startswith("5-10")
    assert gate is not None
    assert "6 GPU-h" in gate.registered_reduction
    assert gate.gate_hours.startswith("4.5-6.0")
    assert "10-20" in gate.smoke_hours
    assert gate.step_time_fallback
    assert part_b is not None
    assert part_b.estimated_gpu_hours == "5-10"
    assert part_b.central_gpu_hours == "5-9"
    assert "hub" in part_b.offline_note.lower()
    assert "artifact" in part_b.offline_note.lower()

    rows = {row.stage: row for row in compute.hours}
    assert rows["Pilot (P1 + P2 + P3)"].padded_hours == "43-52"
    assert rows["Evaluation (seed 0, 25-29 case)"].padded_hours == "332-560"
    assert rows["Pre-tag benchmarks"].padded_hours == "<= 1.5"
    assert rows["Engine registration (after the tag)"].padded_hours == "9.7-16.3"
    assert rows["Total seed 0 (25-29)"].padded_hours == "602-941"
    assert rows["Total seed 0 (25-29)"].refill_hours == "237-383"
    assert "596-930" in compute.local_seed0_total_hours
    assert "654-1,016" in compute.local_seed0_total_hours


def test_round7_pilot_oh_arms_match_each_block():
    p2 = _committed().pilot.p2

    assert p2.selection_only is True
    assert len(p2.arms) == 11
    for arm in (
        "O-H[25-29] + E4B",
        "O-H[25-29] + 12B_qat_q4_0",
        "O-H[16-18] + E4B",
        "O-H[15-29] + E4B",
    ):
        assert arm in p2.arms
    assert "N=4" in p2.teacher_gain_basis
    assert "O-H[25-29]" in p2.teacher_gain_basis
    assert "per block" in p2.oh_lora_target
    assert "selection-only" in p2.selection_rule
    assert "E2-E4 and D1" in p2.selection_rule


def test_round7_stage_a_visibility_is_encoded():
    campaign = _committed().training_free_campaign

    assert campaign.analysis_rules_frozen_at_commit == (
        "8c6d78dcdbe3ac4719c9e5d7a58eeff81b1f9407"
    )
    assert campaign.analysis_rules_frozen_at_timestamp == (
        "2026-11-05T05:46:18-07:00"
    )
    assert "post-completion" in campaign.stage_a_sealing_procedure
    assert "post-tag" not in campaign.stage_a_sealing_procedure
    assert "before" in campaign.stage_a_completion_addendum.lower()
    assert "author" in campaign.stage_a_completion_addendum.lower()
    assert "author" in campaign.stage_a_visibility.lower()
    assert "runs/detached/campaign.json" in campaign.mutable_status_files
    entries = {entry["path"]: entry for entry in campaign.stage_a_sealing}
    assert entries["runs/detached/campaign.json"]["mutable_status"] is True
    assert "02:46" in campaign.decisions_with_n2_on_disk
    assert "03:09:09" in campaign.decisions_with_n2_on_disk


def test_round7_part_b_requires_a_pinned_prompt_sha():
    base = {
        "p1lite_path": "p",
        "requests": 128,
        "eval_batch": 16,
        "candidates": 360,
        "chosen_prompts_sha256": "",
        "estimated_gpu_hours": "5-10",
        "central_gpu_hours": "5-9",
        "offline_note": "hub access; artifact with the sha",
    }
    with pytest.raises(ValidationError, match="chosen_prompts_sha256"):
        Prereg.model_validate(_with(part_b=base))
    base["chosen_prompts_sha256"] = "0" * 64
    validated = Prereg.model_validate(_with(part_b=base))
    assert validated.part_b is not None
    assert validated.part_b.chosen_prompts_sha256 == "0" * 64


def test_round7_minors_are_registered_in_text():
    prereg = _committed()

    assert "if that block is chosen" in prereg.pins.bench_addendum
    d1 = next(item for item in prereg.endpoints.dial_family if item.id == "D1")
    assert "62-69" in d1.corpus and "0-49" in d1.corpus
    assert "math-verify" in d1.corpus
    assert "code" in d1.corpus.lower()
    effects = prereg.intelligence_per_byte.fixed_bytes_effects
    assert "no GPQA or MMLU-Pro run at N=2" in effects["delta_loop"]
    assert "pinned" in prereg.resident_bytes.lo_ratio
    assert "annotation" in prereg.intelligence_per_byte.derived["scalar"]
    assert "--allow-stage-b" in prereg.training_free_campaign.stage_b_gate
    assert prereg.engine_gate.min_exact_fraction == 0.0
    assert "amendment_permitted" in prereg.engine_gate.re_prefill_smoke
    joined = " ".join(prereg.analysis_v1_obligations)
    assert "resolve_run" in joined
    assert "amendment_permitted" in joined


# ---------------------------------------------------------------------------
# Revision round 7b (bench addendum, Stage 2 contingency)
# ---------------------------------------------------------------------------


def test_round7b_bench_addendum_and_stage2_contingency():
    prereg = _committed()

    # The measured bench artifact is pinned by path and sha256.
    assert prereg.pins.bench_artifact.endswith("perf/dial_bench.json")
    assert prereg.pins.bench_artifact_sha256 == (
        "ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1"
    )
    assert "T-dial-rl" in prereg.pins.bench_addendum
    assert "3/3/10" in prereg.pins.bench_addendum
    assert "if that block is chosen" in prereg.pins.bench_addendum
    assert "analysis_rules_frozen_at_commit" in prereg.pins.bench_addendum

    # Deviation D-7b: base mix unchanged, fallback registered as a recipe
    # deviation rather than an execution-only amendment.
    stage2 = prereg.stage2
    assert stage2.rollout_n_groups == {2: 3, 3: 3, 4: 8, 6: 1, 8: 1}
    assert stage2.rollout_n_fallback_groups == {2: 3, 3: 3, 4: 10}
    assert "T-dial-rl" in stage2.rollout_n_fallback_policy
    assert "D-7b" in stage2.rollout_n_fallback_policy
    assert "rollout_n_groups" in stage2.rollout_n_fallback_policy
    assert "T-dial-rl" in stage2.t_dial_rl_memory_bench
    assert "10,240 tokens" in stage2.t_dial_rl_memory_bench
    assert "2,048-token" in stage2.t_dial_rl_memory_bench
    assert "8,192-token" in stage2.t_dial_rl_memory_bench
    assert "no OOM" in stage2.t_dial_rl_memory_bench
    assert "no_grad" in stage2.t_dial_rl_memory_bench
    assert "chunked head" in stage2.t_dial_rl_memory_bench

    # The analysis-rule freeze is untouched.
    assert prereg.training_free_campaign.analysis_rules_frozen_at_commit == (
        "8c6d78dcdbe3ac4719c9e5d7a58eeff81b1f9407"
    )

    # Measured Stage 1 hours replace the estimate; totals are recomputed.
    rows = {row.stage: row for row in prereg.compute.hours}
    assert rows["Stage 1 heal (L + O-H)"].padded_hours == "~8.6-12.0"
    assert rows["Teacher top-k precompute (one teacher)"].padded_hours == "4-16"
    assert rows["Total seed 0 (25-29)"].padded_hours == "602-941"
    assert "237-383" in prereg.compute.local_seed0_total_hours

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "### Revision round 7b (bench addendum, Stage 2 contingency)" in text
    assert "### Revision round 7c (D-7b deviation, recipe binding, minors)" in text
    assert "ab91b834dfd216b4d1d56935309f258dfe9f9984a0663d8a94473f918541e2f1" in text
    assert "Deviation D-7b" in text
    assert "Stage A content is not an input" in flat
    assert "38b5bd2" in text


def test_round7c_d7b_recipe_binding_and_minors():
    prereg = _committed()

    # D-7b is a numbered deviation, triggered by measured memory.
    assert prereg.deviations
    d7b = prereg.deviations[0]
    for token in (
        "D-7b",
        "rollout_n_groups",
        "ab91b834d",
        "38b5bd2",
        "06:02",
        "not an input",
        "10,240 tokens",
        "2,048-token",
        "k=3",
        "no_grad",
        "chunked head",
        "16",
        "no OOM",
    ):
        assert token in d7b

    # The stage A addendum is referenced and the showing is recorded.
    campaign = prereg.training_free_campaign
    assert "38b5bd2" in campaign.stage_a_visibility
    assert "06:02" in campaign.stage_a_visibility
    assert "prereg/stage-a-addendum.md" in campaign.stage_a_completion_addendum
    assert "2026-11-05T04:11:58-07:00" in campaign.stage_a_sealing_procedure

    # Stage 1 recipe binding.
    stage1 = prereg.stage1
    assert stage1.rows_per_optimizer_step == 16
    assert "2e-4" in stage1.recipe_binding
    assert "3e-3" in stage1.recipe_binding
    assert "1e-3" in stage1.recipe_binding
    assert "exit 2" in stage1.recipe_binding

    # The inherited 15-29 totals are recomputed and the teacher precompute
    # is budgeted.
    assert "654-1,016" in prereg.compute.local_seed0_total_hours
    assert "259-416" in prereg.compute.local_seed0_total_hours
    rows = {row.stage: row for row in prereg.compute.hours}
    assert "E2B" in rows["Teacher top-k precompute (one teacher)"].note

    # Pending item 4 is closed; the open pre-tag runs and download are added.
    joined = " ".join(prereg.pending_author).lower()
    assert "stage a addendum: closed" in joined
    assert "one-row e4b/12b teacher smoke" in joined
    assert "teacher-model download" in joined

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "is false and is retired" in flat
    assert "Provenance correction (round 8)" in text
    assert "Total effect versus O: training + loops" in text
    assert "Freeze sequence (round 8" in text
    assert "≈650-1,000" in text
    assert "no longer an estimate" not in flat
    assert "2026-11-05T04:11:58-07:00" in text


def test_round8_teacher_runtimes_and_qualification_rule():
    stage1 = _committed().stage1

    runtimes = stage1.runtimes
    assert set(runtimes) == {"e4b", "12b"}
    assert "bf16" in runtimes["e4b"]
    assert "CPU PLE" in runtimes["e4b"]
    assert "int8" in runtimes["e4b"]
    assert "NF4" in runtimes["12b"]
    assert "Q4_0" in runtimes["12b"]
    assert "llama.cpp" in runtimes["12b"]

    assert "~20" in stage1.runtime_qualification
    assert "8,192" in stage1.runtime_qualification
    assert "Same-checkpoint bf16 reference" in stage1.qualification_references
    assert "common-support" in stage1.qualification_metrics
    joined = " ".join(stage1.acceptance_table)
    for token in ("0.010", "0.015", "0.050", "95 %", "0.002", "0.010", "0.25 GiB"):
        assert token in joined
    assert "expand once to 100" in stage1.expansion_rule
    assert "20 %" in stage1.expansion_rule
    assert len(stage1.fallback_order) >= 4
    assert any("E4B bf16" in item for item in stage1.fallback_order)
    assert any("Q4_0" in item for item in stage1.fallback_order)
    assert "compute dtype" in stage1.manifest_identity
    assert "KV dtype" in stage1.manifest_identity
    assert "fail closed" in stage1.manifest_identity
    assert "best qualified resident runtime per teacher" in stage1.comparison_rule
    assert "Freeze one runtime for each teacher" in stage1.comparison_rule
    assert "bf16" in stage1.four_bit_limitation
    assert "NF4" in stage1.four_bit_limitation
    assert "not automatically" in stage1.four_bit_limitation
    assert "inference_mode" in stage1.execution_repair
    assert "grad_fn" in stage1.execution_repair
    assert "fails closed" in stage1.execution_repair
    assert "frozen before the pilot" in stage1.freeze_rule

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "Registered teacher runtimes and qualification rule" in text
    assert "X3-C1 execution repair" in text
    assert "best qualified resident runtime per teacher" in text


def test_round8_intelligence_per_byte_uses_the_pinned_headers():
    prereg = _committed()
    rb = prereg.resident_bytes
    ipb = prereg.intelligence_per_byte

    e4b = rb.expected_bf16["E4B"]
    for token in ("14.003", "8.753", "5.250", "7,518,069,034", "4ecc2ba", "14.894"):
        assert token in e4b
    assert "pinned" in rb.lo_ratio
    assert "1.62" in rb.lo_ratio
    assert "computed" in rb.lo_ratio
    assert "2.04" in rb.lo_ratio
    assert "1.56" in rb.lo_ratio

    assert "illustrative" in ipb.fixed_bytes_effects["primary_claim"].lower()
    assert "main presentation" in ipb.fixed_bytes_effects["primary_claim"]
    assert "illustrative_only" in ipb.anchoring
    assert "illustrative only" in ipb.anchoring["byte_multiplier"].lower()
    assert "fixed-byte paired effects" in ipb.anchoring["illustrative_only"]
    assert "14.003" in rb.lo_ratio

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "14.003 GiB" in text
    assert "huggingface_hub.get_safetensors_metadata" in text
    assert "illustrative only" in text


def test_round8_stage0_lite_and_why_report():
    prereg = _committed()

    assert "Modal vLLM" in prereg.compute.stage0_lite
    assert "SINGLE backend" in prereg.compute.stage0_lite
    assert "max_length" in prereg.compute.stage0_lite
    assert "minus the rendered prompt length" in prereg.compute.stage0_lite
    assert "identical accepted set" in prereg.compute.stage0_lite
    assert "covariate" in prereg.compute.stage0_lite
    assert "b605af7a625f93c7e3e08a02" in prereg.compute.stage0_lite
    assert "$6" in prereg.compute.stage0_lite
    assert "$30" in prereg.compute.stage0_lite
    assert "Modal-or-local by availability" in prereg.compute.stage0_main_tier_rule
    assert "full Stage 0 for the main runs" in prereg.compute.stage0_main_tier_rule

    assert "descriptive" in prereg.pilot.why_report
    assert "never" in prereg.pilot.why_report
    metrics = " ".join(prereg.pilot.why_metrics)
    for token in (
        "teacher qualification error",
        "teacher entropy",
        "student-teacher KL",
        "gains by span",
        "per-N curves N=1..8",
        "cost per token",
    ):
        assert token in metrics
    assert len(prereg.pilot.why_metrics) == 6
    assert "never selection inputs" in prereg.pilot.why_role

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "Stage-0-lite" in text
    assert 'Pilot "why" report (round 8' in text
    assert "never enter selection" in flat


def test_round8_amendment_is_one_dated_split_changelog():
    prereg = _committed()
    r8 = prereg.revision_round8

    assert r8.title == "Revision round 8 (external review panel)"
    assert set(r8.categories) == {
        "claim clarification",
        "analysis precision",
        "provenance correction",
        "execution contingency",
        "new eligibility design",
    }
    assert all(r8.categories.values())
    assert r8.historical_rules_snapshot == (
        "8c6d78dcdbe3ac4719c9e5d7a58eeff81b1f9407"
    )
    assert "authoritative complete specification" in r8.authoritative_spec
    assert len(r8.choices) >= 10
    joined = " ".join(r8.choices)
    for token in (
        "SMOKE-FAIL",
        "not eligible",
        "fix/gate-v6",
        "[0,2,4,6]",
        "token-weighted",
        "10,240",
        "best qualified resident runtime",
        "14.003",
        "Stage-0-lite",
        "8c6d78d",
    ):
        assert token in joined

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "### Revision round 8 (external review panel)" in text
    for token in (
        "Claim clarification.",
        "Analysis precision.",
        "Provenance correction.",
        "Execution contingency.",
        "New eligibility design.",
        "authoritative complete specification",
        "current refill candidate is NOT eligible",
    ):
        assert token in text


def test_round7d_prompt_cap_8bit_optimizer_and_d7c():
    prereg = _committed()

    # R7c-1: the Stage 2 prompt cap makes the longest real row 10,240 tokens.
    stage2 = prereg.stage2
    assert stage2.prompt_cap == 2048
    assert stage2.completion_cap == 8192
    assert stage2.prompt_cap + stage2.completion_cap == 10_240
    assert "2,048" in stage2.prompt_selection
    assert "10,240" in stage2.prompt_selection
    assert "identically for both arms" in stage2.prompt_selection
    assert "10,240 tokens" in stage2.t_dial_rl_memory_bench

    # R7c-2: the registered optimizer is 8-bit AdamW; the yaml field keeps the
    # legacy "AdamW" spelling and run.json records the resolution.
    assert prereg.stage1.optimizer == "AdamW"
    assert prereg.stage2.optimizer == "AdamW"
    for stage in (prereg.stage1.recipe_binding,):
        assert "adamw_8bit" in stage
        assert "8-bit" in stage
        assert "run.json" in stage
        assert "fix/optim-record" in stage
    assert prereg.stage1.recipe_binding

    # D-7c covers all three corrections and cites the sealing commit/showing.
    d7c = prereg.deviations[1]
    assert d7c.startswith("D-7c")
    for token in (
        "rows_per_optimizer_step",
        "16 rows",
        "adamw_8bit",
        "8-bit AdamW",
        "2,048-token prompt cap",
        "10,240",
        "38b5bd2",
        "06:02",
        "no Stage A content",
        "8c6d78d",
    ):
        assert token in d7c

    # The hours-assumption pointer is in the compute notes, not mde_notes.
    assert "hours rows" in prereg.compute.local_seed0_total_hours
    assert "mde_notes" not in prereg.compute.local_seed0_total_hours

    # The document reflects the round.
    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "revision round 8" in text
    assert "### Revision round 7d (prompt cap, 8-bit AdamW, D-7c)" in text
    assert "Deviation D-7c" in text
    assert "10,240" in text
    assert "adamw_8bit" in text
    assert "fix/optim-record" in text
    assert "median length <= 6k" not in flat
    assert prereg.drafter_choices[-1].startswith("Round 7d")


# ---------------------------------------------------------------------------
# Revision round 8 (external review panel: X1, X2, X3)
# ---------------------------------------------------------------------------


def test_round8_engine_smoke_record_and_gate_v6_eligibility():
    prereg = _committed()
    gate = prereg.engine_gate

    # X1 item 1: the four pre-tag attempts are recorded honestly.
    assert len(gate.pre_tag_smoke_attempts) == 4
    joined = " ".join(gate.pre_tag_smoke_attempts)
    assert "argparse" in joined
    assert "_model_device" in joined
    assert "OOM" in joined
    assert "SMOKE-FAIL" in joined
    assert "1/8" in joined and "4/8" in joined
    assert "re_prefill_smoke.json" in gate.pre_tag_smoke_artifact
    assert gate.pre_tag_smoke_artifact_sha256 == (
        "fa117eff3be7a7ed46a5574073e7b17effef60abe0915f067e2b647ccaca046c"
    )
    assert "not eligible" in gate.pre_tag_smoke_outcome.lower()
    assert "sampled exactness deficit" in gate.pre_tag_smoke_outcome

    # The stream diagnostic is recorded as supporting evidence.
    assert "no" in gate.stream_diagnostic.lower()
    assert "stream" in gate.stream_diagnostic
    assert "lcb-04" in gate.stream_diagnostic
    assert "filter-boundary" in gate.stream_diagnostic

    # X2's replacement eligibility design is registered verbatim.
    rule = gate.replacement_rule
    for token in (
        "chosen-token regret",
        "D = max(max(c) - c[a], max(c) - c[b])",
        "tau_g = max(0.25",
        "0.0625",
        "0.50",
        "HARD FAIL",
        "Stable support",
        "Changed support",
        "0.05",
        "0.005",
        "TV",
        "3/8",
        "32 shared generated steps",
        "FAIL CLOSED",
    ):
        assert token in rule
    assert "[0,2,4,6]" in gate.calibration_partition
    assert "[1,3,5,7]" in gate.calibration_partition
    assert "[0,4,1,5]" in gate.validation_partition
    assert "[2,6,3,7]" in gate.validation_partition
    assert gate.shared_steps_min == 32
    assert "padded" in gate.failure_policy.lower()
    assert "one" in gate.repair_cycle.lower()
    assert "never run" in gate.repair_cycle.lower()
    assert "rejected" in gate.repair_cycle.lower()
    assert "HARD FAIL" in gate.sampled_decision_traces or "reconstruction" in gate.sampled_decision_traces
    assert "top-k" in gate.boundary_certificate or "top-p" in gate.boundary_certificate
    assert "matched denominators" in gate.exactness_rule

    # No repaired-smoke pin exists; v6 was rejected and never run.
    assert "No refill pin exists" in gate.engine_pin_rule
    assert "fix/gate-v6" in prereg.pins.engine_candidate_commit
    assert "eed7fa7" in prereg.pins.engine_candidate_historical
    assert "repaired" in prereg.pins.repaired_smoke_addendum.lower()
    assert "never ran" in prereg.pins.repaired_smoke_addendum.lower()

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "fa117eff3be7a7ed46a5574073e7b17effef60abe0915f067e2b647ccaca046c" in text
    assert "NOT eligible" in text
    assert "all follow the tag is retired" in text
    assert "11:00–11:07" in text or "11:00-11:07" in text


def test_round8_claims_paragraph_is_explicit():
    prereg = _committed()
    claims = prereg.claims

    for token in (
        "D1 establishes average calibrated-likelihood gain and non-inferiority",
        "O-verifiable math traces only",
        'not a general "usable reasoning dial"',
        "E2 compares the complete looped training/inference package",
        "matched-data, matched-layer adapter control",
        "D2 measures the extra-depth effect",
        "D4 provides teacher-forced likelihood evidence only",
        "A D1 pass alone must not produce the headline",
        "numerical endpoints and alpha allocation",
    ):
        assert token in claims

    d1 = next(item for item in prereg.endpoints.dial_family if item.id == "D1")
    assert "O-verifiable math traces only" in d1.test
    assert 'not a general "usable reasoning dial"' in d1.test

    e2 = prereg.endpoints.sequence[0].test
    assert "complete looped" in e2 and "adapter control" in e2
    d2 = next(
        item for item in prereg.endpoints.estimation if item.startswith("D2")
    )
    assert "extra-depth effect" in d2
    d4 = next(
        item for item in prereg.endpoints.estimation if item.startswith("D4")
    )
    assert "teacher-forced likelihood evidence only" in d4
    assert "unmeasured in Part 1" in d4

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "Explicit claims (round 8" in text
    assert "average calibrated-likelihood gain and non-inferiority" in flat
    assert "matched-data, matched-layer adapter control" in flat
    assert "D4 provides teacher-forced likelihood evidence only" in flat


def test_round8_nll_analysis_contract_is_complete():
    nll = _committed().endpoints.nll_analysis

    assert "thinking plus answer" in nll.supervised_token_mask
    assert "token-weighted" in nll.weighting
    assert "problem-level resampling" in nll.weighting
    assert "C_N = sum_i S_iN / sum_i n_i" in nll.weighting
    assert "T_gain = C_1 - C_4" in nll.statistic
    assert "T_adj,N = 0.005" in nll.statistic
    assert "(1 + number of bootstrap T* <= 0) / 10001" in nll.p_value
    assert "conservative" in nll.p_value
    assert "interpolation" in nll.p_value
    assert "60-61" in nll.temperature
    assert "fixed" in nll.temperature
    assert "unevaluable" in nll.unevaluable_corpus
    assert "no rejection" in nll.unevaluable_corpus
    assert "no alpha transfer" in nll.unevaluable_corpus
    assert "prereg-v1-data" in nll.fit_test_freeze
    assert "before any trained-arm output" in nll.fit_test_freeze
    assert "distinct-problem" in nll.denominators
    assert "unevaluable" in nll.denominators

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "Calibrated-NLL analysis contract (round 8" in text
    assert "p = (1 + number of bootstrap T* <= 0) / 10001" in text


def test_round8_stage2_feasibility_branch():
    stage2 = _committed().stage2

    assert "10,240-token maximum" in stage2.feasibility_bench
    assert "O-H at N=1" in stage2.feasibility_bench
    assert "mandatory" in stage2.feasibility_bench
    assert "rollout" in stage2.rollout_storage
    assert "logprobs" in stage2.rollout_storage
    assert "masks" in stage2.rollout_storage
    assert "8-bit AdamW" in stage2.optimizer_state
    assert "complete update" in stage2.optimizer_state
    assert "must itself fit" in stage2.fallback_must_fit
    assert "N=2, 3 and 4" in stage2.mandatory_paths
    assert "STOP" in stage2.mandatory_stop_rule
    assert "new disclosed prereg version" in stage2.mandatory_stop_rule
    assert "permanent OOM" in stage2.mandatory_stop_rule
    assert "retry indefinitely" in stage2.mandatory_stop_rule

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "Feasibility branch (round 8" in text
    assert "STOP before either main Stage 2 arm" in flat
    assert "D-7b's 3/3/10 fallback must itself fit" in flat


def test_round8_provenance_and_pending_list_are_closed():
    prereg = _committed()
    campaign = prereg.training_free_campaign

    assert "after the seal" in campaign.post_seal_access_record
    assert "08:01" in campaign.post_seal_access_record
    assert "X1" in campaign.post_seal_access_record
    assert "does not make" in campaign.post_seal_access_record
    for token in ("rules", "data", "pilot selection", "implementation", "main runs"):
        assert token in campaign.freeze_sequence.lower()
    assert "8c6d78d" in campaign.freeze_sequence
    assert "authoritative complete specification" in campaign.freeze_sequence

    assert len(prereg.pending_author) == 8
    joined = " ".join(prereg.pending_author).lower()
    assert "closed" in joined
    assert "total effect versus o: training + loops" in joined
    assert "l(n=4) - o" in joined
    assert "d-7b and d-7c" in joined
    assert "acknowledged" in joined
    assert "10:55" in joined
    assert "one-row e4b/12b teacher smoke" in joined
    assert "teacher-model download" in joined

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "is false and is retired" in flat
    assert "Provenance correction (round 8)" in text
    assert "Total effect versus O: training + loops" in text
    assert "Freeze sequence (round 8" in text


# ---------------------------------------------------------------------------
# Round 8b schema sync and the release-gate loader check
# ---------------------------------------------------------------------------


def _round8b_items() -> dict:
    return {
        f"R8B-{index:02d}": {
            "id": f"R8B-{index:02d}",
            "title": f"item {index}",
            "changes": "registered",
            "where": ["prereg/prereg.yaml", "src/gemmaloop/prereg.py"],
            "why": "pre-tag technical amendment",
            "uses_outcome": False,
            "outcome_basis": "no trained-arm outcome",
        }
        for index in range(1, 12)
    }


def _round8c_items() -> dict:
    return {
        f"R8C-{index:02d}": {
            "id": f"R8C-{index:02d}",
            "title": f"item {index}",
            "changes": "registered",
            "where": ["prereg/prereg.yaml", "src/gemmaloop/prereg.py"],
            "why": "pre-tag technical amendment",
            "uses_outcome": False,
            "outcome_basis": "no trained-arm outcome",
        }
        for index in range(1, 9)
    }


def _round8b_sections() -> dict:
    return {
        "monitor": {
            "role": "execution safety only",
            "corpus": "64 train-bucket problems",
            "split_rule": "largest sha256 keys",
            "training_exclusion": "excluded from every training subset",
            "teacher_shards": "same teacher and runtime",
            "abort_rules": ["non-finite loss"],
            "warn_rules": ["loss spike"],
            "abort_exit_code": 76,
            "abort_terminal": "terminal exit 76",
            "resume_rule": "resume the last COMPLETE checkpoint",
            "probe_failure_rule": "two consecutive probe failures abort",
            "reserved_ids_contract": "problem-id split map",
        },
        "grading": {
            "math_verify_version": "math-verify 0.9.0",
            "math_verify_internal_timeouts": "disabled",
            "math_rule": "math-verify OR the normaliser",
            "math_why": "both verifiers agree",
            "decided_by": "math_verify, normaliser, both or timeout",
            "math_verify_outer_timeout_s": 10.0,
            "code_sandbox": "Docker sandbox",
            "code_rule": "verified targets only",
            "unsandboxed_rule": "never grade code unsandboxed",
            "dependency": "Docker is required",
        },
        "external_apis": {
            "deepseek_api": "official API",
            "deepseek_model": "deepseek-flash",
            "deepseek_served_version": "DeepSeek-V4.1-Flash-1017",
            "deepseek_off_peak": "off-peak only",
            "deepseek_peak_windows": "01:00-04:00, 06:00-10:00",
            "deepseek_margins": "15% and 5%",
            "deepseek_paid_pilot_max_items": 20,
            "deepseek_waste_stop": "3 of the last 20",
            "deepseek_fingerprint": "fingerprint recorded",
            "deepseek_phase_approval": "author approval",
            "deepseek_budgets": {
                "one": "cap $5",
                "two": "cap $20 (or 300 tasks, cap $15)",
            },
            "deepseek_project_expected_usd": "40-50",
            "deepseek_caps_sum_usd": 25.0,
            "nvidia_nim": "NOT usable",
            "nvidia_nim_terms": "s1.2/1.4 and 4.2 apply",
            "opencode_endpoints": "never used",
        },
        "revision_round8b": {
            "title": "Revision round 8b",
            "date": "2026-11-05",
            "stage_a_exposure": "Stage A was exposed before this round",
            "outcome_use": "No item uses a trained-arm outcome",
            "item_ids": list(_round8b_items()),
            "items": _round8b_items(),
            "choices": ["kept the registered candidate"],
        },
        "revision_round8c": {
            "title": "Revision round 8c",
            "date": "2026-11-05",
            "stage_a_exposure": "Stage A was exposed before this round",
            "outcome_use": "No item uses a trained-arm outcome",
            "item_ids": list(_round8c_items()),
            "items": _round8c_items(),
            "choices": ["kept the registered candidate"],
        },
    }


def test_round8b_sections_are_typed_and_validated():
    prereg = Prereg.model_validate({**_minimal(), **_round8b_sections()})
    assert prereg.monitor.abort_exit_code == 76
    assert prereg.external_apis.deepseek_caps_sum_usd == 25.0
    assert prereg.revision_round8b.items["R8B-01"].title == "item 1"
    assert prereg.revision_round8c.items["R8C-01"].title == "item 1"

    with pytest.raises(ValidationError, match="abort_exit_code must be 76"):
        Prereg.model_validate(
            {
                **_minimal(),
                **_round8b_sections(),
                "monitor": {
                    **_round8b_sections()["monitor"],
                    "abort_exit_code": 1,
                },
            }
        )
    with pytest.raises(ValidationError, match="must equal the sum"):
        Prereg.model_validate(
            {
                **_minimal(),
                **_round8b_sections(),
                "external_apis": {
                    **_round8b_sections()["external_apis"],
                    "deepseek_caps_sum_usd": 99.0,
                },
            }
        )
    sections = _round8b_sections()
    sections["revision_round8b"]["items"]["R8B-01"]["uses_outcome"] = True
    with pytest.raises(ValidationError, match="may not use a trained-arm outcome"):
        Prereg.model_validate({**_minimal(), **sections})


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        (
            "r8b_exact_item_set",
            lambda sections: sections["revision_round8b"]["item_ids"].pop(),
            "revision_round8b.item_ids must list R8B-01..R8B-11",
        ),
        (
            "r8b_where",
            lambda sections: sections["revision_round8b"]["items"][
                "R8B-01"
            ].__setitem__("where", []),
            "revision_round8b.items.*where must be registered",
        ),
        (
            "r8b_outcome_use",
            lambda sections: sections["revision_round8b"].__setitem__(
                "outcome_use", ""
            ),
            "revision_round8b.outcome_use must state that no item uses",
        ),
        (
            "r8c_exact_item_set",
            lambda sections: sections["revision_round8c"]["items"].pop("R8C-08"),
            "revision_round8c.items must key exactly R8C-01..R8C-08",
        ),
        (
            "r8c_stage_a_exposure",
            lambda sections: sections["revision_round8c"].__setitem__(
                "stage_a_exposure", ""
            ),
            "revision_round8c.stage_a_exposure must state that Stage A",
        ),
        (
            "monitor_required_field",
            lambda sections: sections["monitor"].__setitem__("role", ""),
            "monitor.role must be registered",
        ),
        (
            "grading_required_field",
            lambda sections: sections["grading"].__setitem__("math_why", ""),
            "grading.math_why must be registered",
        ),
        (
            "external_required_field",
            lambda sections: sections["external_apis"].__setitem__(
                "deepseek_api", ""
            ),
            "external_apis.deepseek_api must be registered",
        ),
        (
            "external_phase_budgets",
            lambda sections: sections["external_apis"].__setitem__(
                "deepseek_budgets", {}
            ),
            "external_apis.deepseek_budgets must register every phase",
        ),
    ],
)
def test_port_of_the_prereg1_amendment_validators(label, mutate, message):
    sections = _round8b_sections()
    mutate(sections)
    with pytest.raises(ValidationError, match=message):
        Prereg.model_validate({**_minimal(), **sections})


def test_round8b_extra_keys_are_still_forbidden():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Prereg.model_validate(
            {
                **_minimal(),
                **_round8b_sections(),
                "monitor": {
                    **_round8b_sections()["monitor"],
                    "not_registered": True,
                },
            }
        )


def test_check_prereg_loads_script_is_the_release_gate(tmp_path):
    import scripts.check_prereg_loads as check

    assert check.main([str(ROOT / "prereg" / "prereg.yaml")]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "primary_n: 2\nloop_probs: [0.25, 0.4, 0.2, 0.15]\n"
        "not_a_field: true\n",
        encoding="utf-8",
    )
    assert check.main([str(bad)]) == 2
    assert check.main([str(tmp_path / "missing.yaml")]) == 2


# ---------------------------------------------------------------------------
# R2-I1: cross-field validators for the values the code actually reads
# ---------------------------------------------------------------------------

EXTERNAL_PREREG_DIR = Path(r"C:/Users/<user>/ML/GemmaLoop-wt/prereg1")


def _external_raw_yaml() -> tuple[Path, dict]:
    import yaml

    path = EXTERNAL_PREREG_DIR / "prereg" / "prereg.yaml"
    if not path.is_file():
        pytest.skip(f"external prereg worktree is absent: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return path, raw


def _write_mutated_prereg(tmp_path: Path, raw: dict) -> Path:
    import yaml

    path = tmp_path / "mutated-prereg.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_external_prereg_head_and_working_copy_still_load(tmp_path):
    """The two in-flight prereg1 files must stay loadable through the gate."""

    import subprocess

    import scripts.check_prereg_loads as check

    path, _raw = _external_raw_yaml()
    assert check.main([str(path)]) == 0

    head = subprocess.run(
        ["git", "-C", str(EXTERNAL_PREREG_DIR), "show", "HEAD:prereg/prereg.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        pytest.skip(f"cannot read prereg1 HEAD: {head.stderr}")
    head_path = tmp_path / "prereg1-head.yaml"
    head_path.write_text(head.stdout, encoding="utf-8")
    assert check.main([str(head_path)]) == 0


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        (
            "monitor_bucket",
            lambda raw: raw["data"]["splits"].__setitem__(
                "monitor_bucket", "50-52"
            ),
            "monitor_bucket must equal data.splits.train_bucket",
        ),
        (
            "minimum_rows",
            lambda raw: raw["monitor"].__setitem__("minimum_rows", 16),
            "monitor.minimum_rows must be 32",
        ),
        (
            "monitor_problems",
            lambda raw: raw["data"]["splits"].__setitem__("monitor_problems", 8),
            "monitor_problems must be at least monitor.minimum_rows",
        ),
        (
            "p1_rows",
            lambda raw: raw["pilot"]["p1"].__setitem__("rows", 28),
            "pilot.p1.rows must equal",
        ),
        (
            "p1_problems",
            lambda raw: (
                raw["pilot"]["p1"].__setitem__("problems", 60),
                raw["pilot"]["p1"].__setitem__("rows", 240),
            ),
            "pilot.p1.problems must equal data.splits.pilot_p1_problems",
        ),
        (
            "pilot_train_tokens",
            lambda raw: raw["pilot"]["data"].__setitem__(
                "pilot_train_tokens", 1_000_000
            ),
            "pilot_train_tokens must be at least pilot.p2.heal_tokens",
        ),
        (
            "abort_exit_code",
            lambda raw: raw["monitor"].__setitem__("abort_exit_code", 75),
            "abort_exit_code must be 76",
        ),
    ],
)
def test_inconsistent_cross_field_edits_fail_the_release_check(
    tmp_path, capsys, label, mutate, message
):
    import copy

    import scripts.check_prereg_loads as check

    _path, raw = _external_raw_yaml()
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    bad = _write_mutated_prereg(tmp_path, mutated)

    assert check.main([str(bad)]) == 2
    stderr = capsys.readouterr().err
    assert message in stderr, (label, stderr)


# ---------------------------------------------------------------------------
# R3-I1: the prereg1 c0d996b schema validators, messages and tag gate
# ---------------------------------------------------------------------------

C0D996B = "c0d996b"


def _external_prereg_at(revision: str) -> str:
    """Return ``prereg/prereg.yaml`` at a prereg1 revision (read-only)."""

    import subprocess

    if not EXTERNAL_PREREG_DIR.is_dir():
        pytest.skip(f"external prereg worktree is absent: {EXTERNAL_PREREG_DIR}")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(EXTERNAL_PREREG_DIR),
            "show",
            f"{revision}:prereg/prereg.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot read prereg1 {revision}: {result.stderr}")
    return result.stdout


def _external_yaml_path(tmp_path: Path, revision: str = C0D996B) -> Path:
    path = tmp_path / f"prereg-{revision}.yaml"
    path.write_text(_external_prereg_at(revision), encoding="utf-8")
    return path


def _load_external_raw(revision: str = C0D996B) -> dict:
    import yaml

    raw = yaml.safe_load(_external_prereg_at(revision))
    assert isinstance(raw, dict)
    return raw


def _fill_placeholders(value):
    if isinstance(value, str):
        return value.replace("PIN-PENDING", "a" * 40).replace(
            "ALIGN-WITH-CODE", "aligned with the implementation"
        )
    if isinstance(value, dict):
        return {key: _fill_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill_placeholders(item) for item in value]
    return value


def test_c0d996b_loads_in_plain_mode_and_fails_the_tag_gate(tmp_path, capsys):
    import scripts.check_prereg_loads as check

    path = _external_yaml_path(tmp_path)
    assert check.main([str(path)]) == 0

    assert check.main([str(path), "--tag-mode"]) == 2
    stderr = capsys.readouterr().err
    assert "PIN-PENDING" in stderr
    assert "ALIGN-WITH-CODE" in stderr
    assert "pins.task_healpilot_commit" in stderr
    assert "pins.modal_image_commit" in stderr
    assert "monitor.probe_configuration" in stderr


def test_c0d996b_with_placeholders_filled_passes_tag_mode(tmp_path):
    import yaml

    import scripts.check_prereg_loads as check

    raw = _load_external_raw()
    path = tmp_path / "prereg-filled.yaml"
    path.write_text(
        yaml.safe_dump(_fill_placeholders(raw), sort_keys=False),
        encoding="utf-8",
    )

    assert check.main([str(path)]) == 0
    assert check.main([str(path), "--tag-mode"]) == 0


def test_c0d996b_pending_pins_and_the_whole_document_gate(tmp_path):
    prereg = load_prereg(_external_yaml_path(tmp_path))

    assert set(pending_pin_fields(prereg)) == {
        "pins.task_healpilot_commit",
        "pins.modal_image_commit",
    }
    with pytest.raises(ValueError, match="PIN-PENDING"):
        assert_tag_ready(prereg)

    pins_filled = prereg.model_copy(deep=True)
    pins_filled.pins.task_healpilot_commit = "a" * 40
    pins_filled.pins.modal_image_commit = "b" * 40
    assert pending_pin_fields(pins_filled) == []
    # ALIGN-WITH-CODE placeholders elsewhere still block the tag.
    with pytest.raises(ValueError, match="ALIGN-WITH-CODE"):
        assert_tag_ready(pins_filled)

    hidden = pins_filled.model_copy(deep=True)
    hidden.monitor.missing_arm_rule = "hidden ALIGN-WITH-CODE marker"
    with pytest.raises(ValueError, match="monitor.missing_arm_rule"):
        assert_tag_ready(hidden)


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        (
            "r8c_item_and_id_dropped",
            lambda raw: (
                raw["revision_round8c"]["items"].pop("R8C-08"),
                raw["revision_round8c"]["item_ids"].remove("R8C-08"),
            ),
            "revision_round8c.item_ids must list R8C-01..R8C-08",
        ),
        (
            "r8c_item_missing_from_map",
            lambda raw: raw["revision_round8c"]["items"].pop("R8C-08"),
            "revision_round8c.items must key exactly R8C-01..R8C-08",
        ),
        (
            "r8c_item_where_empty",
            lambda raw: raw["revision_round8c"]["items"]["R8C-01"].__setitem__(
                "where", []
            ),
            "revision_round8c.items[R8C-01].where must be registered",
        ),
        (
            "r8c_stage_a_exposure_empty",
            lambda raw: raw["revision_round8c"].__setitem__(
                "stage_a_exposure", ""
            ),
            "revision_round8c.stage_a_exposure must state that Stage A",
        ),
        (
            "r8c_outcome_use_empty",
            lambda raw: raw["revision_round8c"].__setitem__("outcome_use", ""),
            "revision_round8c.outcome_use must state that no item uses",
        ),
        (
            "r8c_item_uses_outcome",
            lambda raw: raw["revision_round8c"]["items"]["R8C-01"].__setitem__(
                "uses_outcome", True
            ),
            "revision_round8c items may not use a trained-arm outcome",
        ),
        (
            "r8c_item_title_empty",
            lambda raw: raw["revision_round8c"]["items"]["R8C-01"].__setitem__(
                "title", ""
            ),
            "revision_round8c.items[R8C-01].title must be registered",
        ),
        (
            "r8b_item_and_id_dropped",
            lambda raw: (
                raw["revision_round8b"]["items"].pop("R8B-11"),
                raw["revision_round8b"]["item_ids"].remove("R8B-11"),
            ),
            "revision_round8b.item_ids must list R8B-01..R8B-11",
        ),
        (
            "no_phase_budgets",
            lambda raw: raw["external_apis"].__setitem__(
                "deepseek_budgets", {}
            ),
            "external_apis.deepseek_budgets must register every phase",
        ),
        (
            "monitor_role_missing",
            lambda raw: raw["monitor"].__setitem__("role", ""),
            "monitor.role must be registered",
        ),
        (
            "grading_math_rule_missing",
            lambda raw: raw["grading"].__setitem__("math_rule", ""),
            "grading.math_rule must be registered",
        ),
        (
            "external_deepseek_api_missing",
            lambda raw: raw["external_apis"].__setitem__("deepseek_api", ""),
            "external_apis.deepseek_api must be registered",
        ),
        (
            "engine_gate_runs_after_tag_missing",
            lambda raw: (
                raw["engine_gate"].__setitem__("status", ""),
                raw["engine_gate"].__setitem__("runs_after_tag", False),
            ),
            "engine_gate.runs_after_tag must be True",
        ),
    ],
)
def test_prereg1_refusal_cases_are_refused_here_too(
    tmp_path, capsys, label, mutate, message
):
    import copy

    import scripts.check_prereg_loads as check

    mutated = copy.deepcopy(_load_external_raw())
    mutate(mutated)
    path = tmp_path / "mutated-prereg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")

    assert check.main([str(path)]) == 2
    stderr = capsys.readouterr().err
    assert message in stderr, (label, stderr)


# Revision round 8b (Stage-A-exposed pre-tag technical amendment)
# ---------------------------------------------------------------------------


def test_round8b_item_ids_appear_in_both_yaml_and_markdown():
    prereg = _committed()
    r8b = prereg.revision_round8b

    expected = {f"R8B-{index:02d}" for index in range(1, 12)}
    assert r8b.title.startswith("Revision round 8b")
    assert r8b.date == "2026-11-05"
    assert set(r8b.item_ids) == expected
    assert set(r8b.items) == expected
    assert "Stage A" in r8b.stage_a_exposure
    assert "exposed" in r8b.stage_a_exposure.lower()
    assert "no item uses a trained-arm outcome" in r8b.outcome_use.lower()
    assert r8b.choices

    for item_id, item in r8b.items.items():
        assert item.id == item_id
        assert item.title
        assert item.changes
        assert item.where
        assert item.why
        assert item.uses_outcome is False
        assert "no trained-arm outcome" in item.outcome_basis.lower()

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "### Revision round 8b (Stage-A-exposed pre-tag technical amendment)" in text
    for item_id in expected:
        assert item_id in text
    assert "No item uses a trained-arm outcome" in text


def test_round8b_monitor_corpus_rules_and_reserved_contract():
    prereg = _committed()
    monitor = prereg.monitor

    assert "64" in monitor.corpus
    assert "sha256(seed:problem_id)" in monitor.corpus
    assert "0-49" in monitor.corpus
    assert "excluded from every training" in monitor.corpus
    assert "MONITOR" in monitor.teacher_shards or "monitor" in monitor.teacher_shards
    assert "same" in monitor.teacher_shards and "runtime" in monitor.teacher_shards
    assert monitor.abort_exit_code == 76
    assert "terminal" in monitor.abort_terminal
    assert "76" in monitor.abort_terminal
    assert "--ack-abort" in monitor.resume_rule
    assert "WARN" in monitor.probe_failure_rule
    assert "two consecutive" in monitor.probe_failure_rule.lower()
    assert "ABORT" in monitor.probe_failure_rule.upper()

    joined_abort = " ".join(monitor.abort_rules).lower()
    assert "non-finite" in joined_abort
    assert "bit-exactness" in joined_abort
    assert "teacher-data integrity" in joined_abort
    assert "parent kl" in joined_abort
    assert len(monitor.warn_rules) >= 8

    contract = monitor.reserved_ids_contract
    for key in ("monitor", "pilot_eval", "pilot_fit", "heal_train", "confirmatory"):
        assert key in contract
    assert "problem-id" in contract
    assert "never treats its own" in contract

    splits = prereg.data.splits
    assert splits.monitor_bucket == "0-49"
    assert splits.monitor_problems == 64
    assert splits.monitor_samples_per_prompt == 1
    assert "largest" in splits.monitor_order_rule


def test_round8b_p1_count_and_pilot_stage_sync():
    prereg = _committed()
    p1 = prereg.pilot.p1
    p2 = prereg.pilot.p2

    assert p1.rows == 200
    assert p1.problems == prereg.data.splits.pilot_p1_problems == 50
    assert p1.samples_per_problem == 4
    assert p1.rows == p1.problems * p1.samples_per_problem
    assert "completion" in p1.row_definition

    assert "25-29" in p2.teacher_stage
    assert "O-H[25-29]" in p2.teacher_stage
    assert "N=4" in p2.teacher_stage
    assert "N=1" in p2.teacher_stage
    assert "E4B" in p2.block_bridge_stage
    assert "same block layers" in p2.block_bridge_stage
    assert "selected teacher" in p2.final_recipe_teacher_rule
    assert "E4B arms" in p2.selection_rule

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "50 bucket-59 problems × 4 samples = 200 rows" in flat
    assert "Teacher stage:" in flat
    assert "Block/bridge stage:" in flat


def test_round8b_grading_pool_engine_and_external_api_rules():
    prereg = _committed()
    grading = prereg.grading
    pool = prereg.data.pool
    compute = prereg.compute
    apis = prereg.external_apis

    # Math grading + Docker dependency.
    assert "0.9.0" in grading.math_verify_version
    assert "disabled" in grading.math_verify_internal_timeouts
    assert grading.math_verify_outer_timeout_s == 10.0
    assert "OR" in grading.math_rule or "or" in grading.math_rule
    assert "multiprocessing" in grading.math_why
    assert "1,234" in grading.math_why
    assert "decided_by" in grading.decided_by
    assert "Docker" in grading.code_sandbox
    assert "before any pilot evaluation" in grading.code_rule
    assert "never" in grading.unsandboxed_rule.lower()
    code_verification = prereg.pilot.code_verification
    assert isinstance(code_verification, object)
    assert "frozen docker verifier" in code_verification.domain.lower()
    assert "chat" in code_verification.chat_rule.lower()
    assert "never" in code_verification.chat_rule.lower()
    assert "4840d514" in code_verification.docker_image_digest
    assert "6 seconds" in code_verification.timeout_rule
    assert code_verification.freeze_index == (
        "prereg/pilot-shortfall-freeze.json"
    )
    # The round-8d math-only text is retained only as dated history.
    assert "math-only" in prereg.revision_round8d.items["R8D-05"].changes
    assert "never retroactively" in prereg.revision_round8d.items[
        "R8D-05"
    ].changes

    # Part-1 pool mode and source pins.
    assert pool.code_count == 2000
    assert pool.math_count == 2000
    assert pool.chat_count == 1000
    assert pool.mode == "part1"
    assert pool.part1_creative_briefs is False
    assert pool.manifest_sha256 == (
        "70053b90fa631fda1e5e0fa77a932e30c4bab80e81ef136f2330adf0daaef84e"
    )
    assert set(pool.source_pins) == {"APPS", "TACO", "CodeContests"}
    assert "0f10e424e13e1c2a69f851e153097b71b6734a1f" in pool.source_pins["APPS"]
    assert "d593ed0a" in pool.source_pins["TACO"]
    assert "pre_2023_05" in pool.source_pins["CodeContests"]
    assert "Python-only" in pool.source_pins["CodeContests"]

    # Engine closure and the measured Stage-0-lite rate.
    assert "closed" in compute.refill_track_status.lower()
    assert "padded" in compute.registered_eval_engine
    assert "100 requests" in compute.stage0_lite_measured
    assert "2,600" in compute.stage0_lite_measured
    assert "195 tok/s" in compute.stage0_lite_measured
    assert "52/100" in compute.stage0_lite_measured
    assert "17-20" in compute.stage0_lite_hours
    assert "padded adaptive-batch HF engine" in prereg.stage2.engine
    assert "closed" in prereg.stage2.engine.lower()
    lite_row = next(
        row for row in compute.hours if row.stage.startswith("Stage-0-lite")
    )
    assert "17-20" in lite_row.padded_hours

    # Modal history, guards and Part-1 optionality.
    assert "InvalidError" in compute.modal_probe_history
    assert "rapidfuzz" in compute.modal_probe_history
    assert "six failed startups" in compute.modal_probe_history
    assert "stopped by hand" in compute.modal_probe_history
    assert "3.087" in compute.modal_probe_history
    assert "CPU-only preflight" in compute.modal_money_guards
    assert "fail-fast" in compute.modal_money_guards
    assert "single backend" in compute.modal_optional_part1.lower()
    assert "fallback tier" in compute.modal_optional_part1.lower()

    # External API rules and budgets.
    assert "official" in apis.deepseek_api.lower()
    assert apis.deepseek_model.startswith("deepseek-flash")
    assert apis.deepseek_served_version == "DeepSeek-V4.1-Flash-1017"
    assert "off-peak" in apis.deepseek_off_peak.lower()
    assert "01:00-04:00" in apis.deepseek_peak_windows
    assert "06:00-10:00" in apis.deepseek_peak_windows
    assert "15" in apis.deepseek_margins and "5" in apis.deepseek_margins
    assert apis.deepseek_paid_pilot_max_items == 20
    assert "3 of the last 20" in apis.deepseek_waste_stop
    assert "fingerprint" in apis.deepseek_fingerprint
    assert "author" in apis.deepseek_phase_approval.lower()
    assert "$5" in apis.deepseek_budgets["phase1_hidden_tests"]
    assert "40-50" in apis.deepseek_project_expected_usd
    assert apis.deepseek_caps_sum_usd == 71.0
    assert "NOT usable" in apis.nvidia_nim
    for term in ("s1.2/1.4", "4.2", "4.4", "4.12"):
        assert term in apis.nvidia_nim_terms
    assert "never" in apis.opencode_endpoints.lower()

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "math-verify 0.9.0" in flat
    assert "70053b90fa631fda1e5e0fa77a932e30c4bab80e81ef136f2330adf0daaef84e" in flat
    assert "refill track is closed" in flat.lower()
    assert "≈195 tok/s" in flat or "195 tok/s" in flat
    assert "NVIDIA NIM trial is not usable" in flat
    assert "OpenCode endpoints are never used" in flat


def test_round8b_teacher_qualification_corpus_amendment():
    stage1 = _committed().stage1
    corpus = stage1.qualification_corpus

    assert "Stage 0 local" in corpus
    assert "train bucket" in corpus
    assert "verified" in corpus
    assert "4/8/4/4" in corpus
    assert "20-row" in corpus
    assert "100" in corpus
    assert "protected" in corpus
    for token in ("pilot evaluation", "T-fit", "MONITOR", "confirmatory"):
        assert token in corpus
    assert "120" in corpus
    assert "85" in corpus
    assert "P1-lite" in stage1.qualification_fallback
    assert "Stage 0 local" in stage1.runtime_qualification


def _fill_align_markers(model):
    """Return a copy of ``model`` with every align placeholder replaced."""

    from pydantic import BaseModel

    def fill(value):
        if isinstance(value, str):
            return value.replace(
                "ALIGN-WITH-CODE", "aligned with the implementation"
            )
        if isinstance(value, list):
            return [fill(item) for item in value]
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        if isinstance(value, BaseModel):
            return value.model_copy(
                update={
                    name: fill(getattr(value, name))
                    for name in type(value).model_fields
                }
            )
        return value

    return fill(model)


def test_round8b_pin_placeholders_block_the_tag():
    prereg = _committed()

    assert pending_pin_fields(prereg) == []
    assert assert_tag_ready(prereg) is None

    blocked = prereg.model_copy(deep=True)
    blocked.pins.task_healpilot_commit = "PIN-PENDING"
    with pytest.raises(ValueError, match="PIN-PENDING"):
        assert_tag_ready(blocked)

    merged = prereg.pins.merged_commits
    assert "217b9390081d9c838193aa06665c67a3b1a6809d" in merged["task_teacherrt"]
    assert "2eef68b361f0ce7160ff30eff580ee921a5f61cf" in merged["fix_deepseek_offpeak"]
    assert "106e0e9fb2ff8f63a356f6f3be657967ad653183" in merged[
        "fix_deepseek_guards_final"
    ]
    assert "dfea02a617248157e8dba85e4c8d954b20272900" in merged["task_sources"]
    assert "c840e9b44deccccdf99cd13934d1380017e8efe4" in merged[
        "fix_modal_image_order"
    ]
    assert "8b4c020b8da5112e7b6e8b574dc5118c1d6510ba" in merged["task_p1corpus"]
    assert "1830a361522ec2ef67d5b307e4e4b49b97d83aad" in merged[
        "fix_modal_imports"
    ]
    assert prereg.pins.task_p1corpus_commit == (
        "8b4c020b8da5112e7b6e8b574dc5118c1d6510ba"
    )
    assert prereg.pins.fix_modal_imports_commit == (
        "1830a361522ec2ef67d5b307e4e4b49b97d83aad"
    )

    assert prereg.pins.task_healpilot_commit == (
        "e3f63f352a32ccb3bccb013a781536ee6a537511"
    )
    assert prereg.pins.modal_image_commit == (
        "6d3bb65c7a007bc67adf1ddf1a9536272b2163c5"
    )
    assert prereg.pins.task_modallite_commit == (
        "4c260ebe1810a0f95cb47d774bd8eed894e9cfda"
    )
    assert "9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab" in (
        prereg.pins.pin_pending_rule
    )
    assert "$0.758" in prereg.pins.pin_pending_rule

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for token in ("217b939", "2eef68b", "dfea02a", "c840e9b", "e3f63f3",
                  "6d3bb65", "9c4cfd2"):
        assert token in text


# ---------------------------------------------------------------------------
# Revision round 8c (X1 items 1-5, round-8b verifier items 1-7, pins)


# Revision round 8c (X1 items 1-5, round-8b verifier items 1-7, pins)
# ---------------------------------------------------------------------------


def _flat_md() -> str:
    return " ".join(
        (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
            encoding="utf-8"
        ).split()
    )


def test_round8c_item_ids_appear_in_both_yaml_and_markdown():
    prereg = _committed()
    r8c = prereg.revision_round8c

    expected = {f"R8C-{index:02d}" for index in range(1, 9)}
    assert r8c.title.startswith("Revision round 8c")
    assert r8c.date == "2026-11-05"
    assert set(r8c.item_ids) == expected
    assert set(r8c.items) == expected
    assert "Stage A" in r8c.stage_a_exposure
    assert "exposed" in r8c.stage_a_exposure.lower()
    assert "no item uses a trained-arm outcome" in r8c.outcome_use.lower()
    assert r8c.choices

    for item_id, item in r8c.items.items():
        assert item.id == item_id
        assert item.title
        assert item.changes
        assert item.where
        assert item.why
        assert item.uses_outcome is False
        assert "no trained-arm outcome" in item.outcome_basis.lower()

    text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "### Revision round 8c" in text
    for item_id in expected:
        assert item_id in text


def test_round8c_padded_is_the_sole_operative_engine():
    prereg = _committed()
    gate = prereg.engine_gate
    part_b = prereg.part_b

    assert "SOLE OPERATIVE" in prereg.compute.local_engine
    assert "no refill run is registered" in prereg.compute.local_engine
    assert "INACTIVE" in gate.status
    assert gate.runs_after_tag is False
    assert "INACTIVE" in part_b.status
    assert "No repaired-smoke commit exists" in prereg.compute.repaired_engine_pin
    assert "never run" in gate.pre_tag_smoke_outcome.lower()
    assert "rejected" in gate.pre_tag_smoke_outcome.lower()
    assert "8f36aca" in gate.pre_tag_smoke_outcome
    assert "padded_engine_source" in prereg.compute.engine_pin_rule
    assert "282c66b5cd07db1060303927587139fec2ff2d3c" in (
        prereg.pins.padded_engine_source
    )
    assert "c8c192c7e6e10851947da0ca1e957d1593ce365c" in (
        prereg.pins.padded_engine_source
    )

    flat = _flat_md()
    assert "if and only if" not in flat
    assert "the repaired smoke actually ran" not in flat
    assert "repaired smoke was never run" in flat
    assert "sole operative" in flat.lower()
    assert "engine_gate" in flat and "INACTIVE" in flat
    assert "8f36aca" in flat


def test_round8c_realized_counts_and_shortfall_rules():
    prereg = _committed()
    splits = prereg.data.splits
    p1 = prereg.pilot.p1
    p2 = prereg.pilot.p2

    assert splits.train_problems == 2500
    assert "nominal pre-carve" in splits.train_problems_role
    assert splits.realized_counts["train"] == 2488
    assert splits.realized_counts["train_after_monitor"] == 2424
    assert splits.realized_counts["pilot_p1"] == 58
    assert splits.realized_counts["pilot_p1_selected"] == 50
    assert splits.realized_counts["pilot_p1_selected_math"] == 23
    assert splits.realized_counts["pilot_tfit"] == 86
    assert splits.realized_counts["pilot_selection"] == 152
    assert splits.realized_counts["nll_fit"] == 90
    assert splits.realized_counts["nll_fit_math"] == 40
    assert splits.realized_counts["nll_test"] == 396
    assert splits.realized_counts["nll_test_math"] == 162
    assert "nominal" in splits.nominal_counts_role.lower()
    assert "no borrowing" in splits.shortfall_rules.lower() or (
        "never authorize borrowing" in splits.shortfall_rules.lower()
    )

    assert p1.rows == 200
    assert "attempts" in p1.row_definition.lower()
    assert "sha256(0:" in p1.truncation_rule
    assert "50 distinct problems" in p1.truncation_rule
    assert "passes the Stage 0" in p1.retained_row_rule
    assert "math plus code" in p1.domain_rule
    assert "chat never" in p1.domain_rule
    assert p1.min_verified_problems == 20
    assert p1.gates_p2 is False
    assert "descriptive" in p1.role
    assert "unevaluable" in p1.shortfall_rule

    assert "86" in p2.domain_and_freeze
    assert "152" in p2.domain_and_freeze
    assert "math plus code" in p2.domain_and_freeze
    assert "chat never" in p2.domain_and_freeze
    assert "4,5,6,7" in p2.shortfall_rule
    assert "T-fit>=20" in p2.shortfall_rule
    assert "selection>=30" in p2.shortfall_rule

    monitor = prereg.monitor
    assert monitor.minimum_rows == 32
    assert "64" in monitor.missing_arm_rule
    assert "never refilled" in monitor.missing_arm_rule

    nll = prereg.endpoints.nll_analysis
    assert "at least 20" in nll.unevaluable_corpus
    assert "at least 40" in nll.unevaluable_corpus

    d1_corpus = prereg.endpoints.dial_family[0].corpus
    for token in (
        "registered Stage 0 data sampler",
        "post-tag, pre-pilot",
        "prereg-v1-data",
        "32768",
        "8192",
    ):
        assert token in d1_corpus

    flat = _flat_md()
    for token in (
        "2,488",
        "2,424",
        "200 attempts",
        "first **50 distinct problems**",
        "math plus code",
        "at least 20",
        "selection ≥ 30",
        "registered Stage 0 data sampler",
        "post-tag, pre-pilot",
    ):
        assert token in flat


def test_round8c_monitor_abort_resume_probe_and_missing_arm():
    prereg = _committed()
    monitor = prereg.monitor

    abort = monitor.abort_terminal.lower()
    assert "writes no checkpoint" in abort
    assert "last complete checkpoint stays intact" in abort
    assert "76" in monitor.abort_terminal
    for token in (
        "Every registered resume entry path",
        "--ack-abort",
        "exits 2",
        "SAME registered run",
        "last COMPLETE checkpoint",
        "archive the partial run directory",
        "restart from zero",
        "does not authorize",
    ):
        assert token in monitor.resume_rule
    assert "technical repair" in monitor.repair_rule.lower()
    assert "new disclosed prereg version" in monitor.repair_rule
    assert "500,000" in monitor.probe_configuration
    assert "optimizer-step" in monitor.probe_configuration
    for token in ("0.52M", "1.04M", "1.56M", "2.08M", "2.60M", "3.00M"):
        assert token in monitor.probe_configuration
    assert "32-row" in monitor.probe_configuration
    assert "NLL N = 1,2,3,4" in monitor.probe_configuration
    assert "N = 1,2,4" in monitor.probe_configuration
    assert "aborted or incomplete" in monitor.missing_arm_rule.lower()
    assert "pending/unevaluable" in monitor.missing_arm_rule
    assert "never changes" in monitor.warn_effect
    r8c03 = prereg.revision_round8c.items["R8C-03"].changes
    assert "partial run directory is archived" in r8c03
    assert "deferred the exact probe token points" in r8c03
    assert "round 8d" in r8c03
    assert any("rows" in rule and "hinge" in rule for rule in monitor.warn_rules)
    hinge_rules = [rule for rule in monitor.warn_rules if "hinge" in rule]
    assert hinge_rules and all("not a token rule" in rule for rule in hinge_rules)
    sanity_rules = [
        rule for rule in monitor.warn_rules if "generation-sanity" in rule
    ]
    assert sanity_rules and all(
        "finish_reason" in rule
        and "512-token" in rule
        and "repetition loops" in rule
        and "length-truncated" in rule
        for rule in sanity_rules
    )

    flat = _flat_md()
    assert "writes no checkpoint" in flat
    assert "hinge active on > 50 % of **rows**" in flat
    assert "archive the partial run" in flat
    assert "restart from zero" in flat
    assert "0.52M" in flat and "2.60M" in flat and "~3.00M" in flat
    assert "finish_reason" in flat
    assert "never retroactively" in flat
    assert "technical repair" in flat.lower()
    assert "aborted or incomplete teacher/arm cell" in flat


def test_round8c_temperature_fit_and_qualification_timing():
    prereg = _committed()
    nll = prereg.endpoints.nll_analysis
    stage1 = prereg.stage1

    for text in (nll.temperature, prereg.pins.temperature_fit_algorithm):
        assert "fit_model_temperature" in text
        assert "(0.5, 4.0)" in text or "0.5, 4.0" in text
        assert "max_iter = 40" in text
        assert "tol = 5e-3" in text
        assert "54cab07" in text
    assert "NOT the tensor helper" in nll.temperature
    assert "one estimator" in nll.temperature
    assert "unevaluable" in nll.temperature
    assert "boundary fit" in nll.temperature

    assert "after the prereg-v1 tag" in stage1.freeze_rule
    assert "before the pilot" in stage1.freeze_rule
    assert "new prereg version" in stage1.freeze_rule
    assert "before the tag" not in stage1.expansion_rule
    assert not any("before the tag" in item for item in stage1.fallback_order)

    flat = _flat_md()
    assert "max_iter = 40" in flat
    assert "tol = 5e-3" in flat
    assert "f51de454" in flat
    assert "after the tag and before the pilot" in flat
    assert "at least 140 verified rows" in flat


def test_round8c_claims_and_interpretability():
    prereg = _committed()
    claims = prereg.claims

    for token in (
        "matched LoRA capacity",
        "itemized separately",
        "exploratory, selected and optimistic",
        "fresh-data validation",
        "deployable teacher representations",
        "not model size alone",
        "descriptive explanations",
        "O-solvable verified math",
        "restarts from O",
        "3M pilot plus a 12M main run is never described as 12M",
        "Shared seeds do not establish cross-backend identity",
    ):
        assert token in claims

    assert "LoRA capacity is matched" in prereg.part1.arms["O-H"].description
    assert "itemized separately" in prereg.part1.arms["O-H"].description
    assert "matched LoRA capacity" in prereg.stage1.oh_stage1_note
    assert "no recurrent bridge" in prereg.stage1.oh_stage1_note
    assert "itemized separately" in prereg.architecture.lora.target
    assert "exploratory/selected/optimistic" in prereg.endpoints.estimation[2]
    assert "fresh-data validation" in prereg.endpoints.estimation[2]
    assert "NOT claimed disjoint from every Stage 0 split" in (
        prereg.training_free_campaign.corpus_disjointness
    )
    assert not any(
        "equal trainable parameters" in choice
        for choice in prereg.drafter_choices
    )
    assert not any(
        "refill if the gate passes" in choice
        for choice in prereg.drafter_choices
    )

    flat = _flat_md()
    assert "O-H isolates the loop" not in flat
    assert "not** claimed to be disjoint from every Stage 0 split" in flat
    for token in (
        "matched LoRA capacity",
        "1,536-parameter recurrent bridge",
        "exploratory/selected/optimistic",
        "deployable representations, not size",
        "descriptive, not causal",
        "O-solvable math",
        "restarts from O",
        "Shared seeds do not establish backend identity",
        "pilot STOP is a recipe/budget gate",
    ):
        assert token in flat


def test_round8c_verifier_sync_and_filled_pins():
    prereg = _committed()
    apis = prereg.external_apis
    stage2 = prereg.stage2

    assert stage2.deepseek.model == "deepseek-flash"
    assert stage2.deepseek.served_version == "DeepSeek-V4.1-Flash-1017"
    assert stage2.deepseek.fingerprint_record.endswith(
        "deepseek_fingerprint.json"
    )
    assert apis.deepseek_paid_pilot_max_items == 20
    assert apis.deepseek_writer_hard_cap_items == 50
    assert "not a registered paid-run size" in apis.deepseek_phase_approval

    assert prereg.pins.task_p1corpus_commit == (
        "8b4c020b8da5112e7b6e8b574dc5118c1d6510ba"
    )
    assert prereg.pins.fix_modal_imports_commit == (
        "1830a361522ec2ef67d5b307e4e4b49b97d83aad"
    )
    assert prereg.pins.task_healpilot_commit == (
        "e3f63f352a32ccb3bccb013a781536ee6a537511"
    )
    assert prereg.pins.modal_image_commit == (
        "6d3bb65c7a007bc67adf1ddf1a9536272b2163c5"
    )
    assert prereg.pins.task_modallite_commit == (
        "4c260ebe1810a0f95cb47d774bd8eed894e9cfda"
    )
    assert "9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab" in (
        prereg.pins.pin_pending_rule
    )
    assert "$0.758" in prereg.pins.pin_pending_rule
    assert "8b4c020" in prereg.pins.merged_commits["task_p1corpus"]
    assert "1830a36" in prereg.pins.merged_commits["fix_modal_imports"]
    assert "106e0e9" in prereg.pins.merged_commits["fix_deepseek_guards_final"]
    assert "e3f63f3" in prereg.pins.merged_commits["task_healpilot"]
    assert "6d3bb65" in prereg.pins.merged_commits["fix_modal_image_jit"]
    assert "e3f63f352a32ccb3bccb013a781536ee6a537511" in (
        prereg.pins.pin_pending_rule
    )
    assert "6d3bb65c7a007bc67adf1ddf1a9536272b2163c5" in (
        prereg.pins.pin_pending_rule
    )

    flat = _flat_md()
    for token in (
        "8b4c020b8da5112e7b6e8b574dc5118c1d6510ba",
        "1830a361522ec2ef67d5b307e4e4b49b97d83aad",
        "106e0e9fb2ff8f63a356f6f3be657967ad653183",
        "37583c197e4ac76d20031b0969935e92c49a165c",
        "1e789963f1e313a90686060a223276eef0a0b4ec",
        "c9d0cdee3a365039ac796df64cb063358e48bf58",
        "6d3bb65c7a007bc67adf1ddf1a9536272b2163c5",
    ):
        assert token in flat


def test_round8c_tag_gate_scans_the_whole_document():
    fixed = _fill_align_markers(_committed())
    fixed.pins.task_healpilot_commit = "a" * 40
    fixed.pins.modal_image_commit = "b" * 40
    assert pending_pin_fields(fixed) == []
    assert assert_tag_ready(fixed) is None

    hidden = fixed.model_copy(deep=True)
    hidden.monitor.missing_arm_rule = "hidden PIN-PENDING marker"
    with pytest.raises(ValueError, match="monitor.missing_arm_rule"):
        assert_tag_ready(hidden)


def test_round8d_items_probe_schedule_pins_and_domain():
    prereg = _committed()
    r8d = prereg.revision_round8d

    expected = {f"R8D-{index:02d}" for index in range(1, 6)}
    assert r8d.title.startswith("Revision round 8d")
    assert r8d.date == "2026-11-05"
    assert set(r8d.item_ids) == expected
    assert set(r8d.items) == expected
    assert "Stage A" in r8d.stage_a_exposure
    assert "exposed" in r8d.stage_a_exposure.lower()
    assert "no item uses a trained-arm outcome" in r8d.outcome_use.lower()
    assert r8d.choices
    for item_id, item in r8d.items.items():
        assert item.id == item_id
        assert item.title and item.changes and item.where and item.why
        assert item.uses_outcome is False
        assert "no trained-arm outcome" in item.outcome_basis.lower()

    monitor = prereg.monitor
    for token in ("0.52M", "1.04M", "1.56M", "2.08M", "2.60M", "3.00M"):
        assert token in monitor.probe_configuration
    assert "tokens_seen - last_probe_tokens >= 500,000" in (
        monitor.probe_configuration
    )
    sanity = [rule for rule in monitor.warn_rules if "generation-sanity" in rule]
    assert sanity
    assert all(
        "finish_reason" in rule and "512-token" in rule and "length-truncated" in rule
        for rule in sanity
    )

    assert prereg.pins.task_healpilot_commit == (
        "e3f63f352a32ccb3bccb013a781536ee6a537511"
    )
    assert prereg.pins.modal_image_commit == (
        "6d3bb65c7a007bc67adf1ddf1a9536272b2163c5"
    )
    assert prereg.pins.task_modallite_commit == (
        "4c260ebe1810a0f95cb47d774bd8eed894e9cfda"
    )
    assert "9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab" in (
        prereg.pins.pin_pending_rule
    )
    assert "$0.758" in prereg.pins.pin_pending_rule
    assert pending_pin_fields(prereg) == []
    assert assert_tag_ready(prereg) is None

    assert "b605af7a625f93c7e3e08a02" in prereg.compute.stage0_lite
    assert "$6" in prereg.compute.stage0_lite
    assert "$30" in prereg.compute.stage0_lite
    assert "5" in prereg.compute.modal_probe_history
    assert "11,725,158" in prereg.compute.modal_probe_history
    assert "$2.069" in prereg.compute.modal_probe_history

    r8d05 = prereg.revision_round8d.items["R8D-05"]
    assert "HISTORICAL" in r8d05.title
    assert "SUPERSEDED BY ROUND 8E" in r8d05.changes
    assert "math-only" in r8d05.changes
    assert "never retroactively" in r8d05.changes
    assert "math plus code" in prereg.pilot.p1.domain_rule
    assert "chat never" in prereg.pilot.p2.domain_and_freeze

    flat = _flat_md()
    assert "Revision round 8d" in flat
    for item_id in expected:
        assert item_id in flat
    assert "R8D-01" in flat and "R8D-05" in flat


# ---------------------------------------------------------------------------
# Round 8e: prospective pilot shortfall amendment
# ---------------------------------------------------------------------------


def test_round8e_pilot_shortfall_amendment_fields_and_freeze():
    import yaml

    prereg = _committed()
    r8e = prereg.revision_round8e

    expected = {f"R8E-{index:02d}" for index in range(1, 9)}
    assert r8e.title.startswith("Revision round 8e")
    assert r8e.date == "2026-11-06"
    assert set(r8e.item_ids) == expected
    assert set(r8e.items) == expected
    assert "Stage A" in r8e.stage_a_exposure
    assert "exposed" in r8e.stage_a_exposure.lower()
    assert "no item uses a trained-arm outcome" in r8e.outcome_use.lower()
    assert r8e.choices
    for item_id, item in r8e.items.items():
        assert item.id == item_id
        assert item.title and item.changes and item.where and item.why
        assert item.uses_outcome is False
        assert "no trained-arm outcome" in item.outcome_basis.lower()

    cv = prereg.pilot.code_verification
    assert "math plus code" in cv.domain
    assert "chat" in cv.chat_rule.lower() and "never" in cv.chat_rule.lower()
    assert "existing source tests" in cv.test_set
    assert "ordered per-problem test manifest" in cv.test_manifest
    assert "extract_code" in cv.extraction
    assert "GraderService" in cv.grading_implementation
    assert "python:3.12-slim" in cv.docker_image
    assert (
        "sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54"
        in cv.docker_image_digest
    )
    for token in ("CPU-only", "network none", "read-only", "64 MiB", "512 MiB"):
        assert token in cv.sandbox_limits
    assert "6 seconds" in cv.timeout_rule
    assert "sandboxed=true" in cv.pass_rule
    assert "not count as a pass" in cv.fail_rule
    assert "identical input bytes" in cv.infrastructure_rule
    assert "None" in cv.unsandboxed_fallback
    assert "official count" in cv.current_count_rule
    assert cv.freeze_index == "prereg/pilot-shortfall-freeze.json"

    p1 = prereg.pilot.p1
    assert "descriptive" in p1.role
    assert p1.gates_p2 is False
    assert "None" in p1.continuation_gate
    assert p1.small_sample_threshold == 20
    assert "unevaluable" in p1.zero_count_rule
    assert "math plus code" in p1.domain_rule
    assert "chat never" in p1.domain_rule
    assert "does not gate P2" in p1.shortfall_rule

    p2 = prereg.pilot.p2
    assert "math plus code" in p2.domain_and_freeze
    assert "chat never" in p2.domain_and_freeze
    assert "4,5,6,7" in p2.shortfall_rule
    assert "T-fit>=20" in p2.shortfall_rule
    assert "selection>=30" in p2.shortfall_rule
    assert "no second batch" in p2.shortfall_rule.lower()
    assert "D and E are rejected" in p2.shortfall_rule

    sf = prereg.pilot.shortfall
    assert "math plus code" in sf.domain
    assert "chat never" in sf.p2_domain
    assert sf.p1_gates_p2 is False
    assert sf.attempt_indices == [4, 5, 6, 7]
    assert sf.max_extension_batches == 1
    assert 'p+":"+str(j)' in sf.request_id_rule
    assert "int.from_bytes" in sf.seed_formula
    assert "2**63-1" in sf.seed_formula
    assert "four times" in sf.request_count_rule
    assert "temperature 1.0" in sf.configuration
    assert "8192 - rendered_prompt_tokens" in sf.configuration
    assert sf.final_min_tfit_problems == 20
    assert sf.final_min_selection_problems == 30
    assert "unevaluable" in sf.final_shortfall_rule
    assert "no automatic second batch" in sf.no_relaxation_rule
    assert "D is rejected" in sf.rejected_alternatives
    assert "E is rejected" in sf.rejected_alternatives
    assert "unused bucket" in sf.unused_bucket_rule.lower()

    freeze = prereg.pilot.freeze
    assert freeze.required is True
    assert freeze.index_path == "prereg/pilot-shortfall-freeze.json"
    assert "staged freeze" in freeze.staged_freeze.lower()
    assert "REQUIRED before any pilot scoring or P2 training" in (
        freeze.required_before
    )
    assert len(freeze.contents) >= 6
    assert "13" in freeze.realized_counts and "16" in freeze.realized_counts
    assert "23" in freeze.realized_counts
    assert "4840d514" in freeze.preview_counts
    assert "16" in freeze.preview_counts and "27" in freeze.preview_counts
    assert "40" in freeze.preview_counts
    assert "facts" in freeze.decision_input
    assert "pending" in freeze.pending_rule.lower()
    import json

    freeze_file = ROOT / "prereg" / "pilot-shortfall-freeze.json"
    freeze_index = json.loads(freeze_file.read_text(encoding="utf-8"))
    assert freeze_index["kind"] == "pilot_shortfall_freeze_index"
    assert freeze_index["status"] == "complete"
    assert freeze_index["run_version"] == "v3r"
    assert freeze_index["count_status"] == "complete"
    assert freeze_index["pending_bindings"] == []
    assert freeze_index["final_counts"] == {
        "pilot_p1": 19,
        "pilot_tfit": 32,
        "pilot_selection": 51,
    }
    assert freeze_index["decision"]["extension_required"] is False
    assert (
        freeze_index["decision"]["extension_decision"]
        == "no_extension_original_only"
    )
    labels = {entry["label"] for entry in freeze_index["bindings"]}
    assert "pilot_shortfall_amendment" in labels
    assert "count_manifest" in labels
    assert "final_corpus/pilot_tfit.rendered.jsonl" in labels
    assert "final_corpus/pilot_selection.rendered.jsonl" in labels

    assert prereg.pins.task_modallite_commit == (
        "4c260ebe1810a0f95cb47d774bd8eed894e9cfda"
    )
    assert "9c4cfd2d5c3d4467c25b027fa2e89d997d8e7eab" in (
        prereg.pins.pin_pending_rule
    )
    assert "$0.758" in prereg.pins.pin_pending_rule
    assert "seed overflow" in prereg.pins.pin_pending_rule

    assert pending_pin_fields(prereg) == []
    assert assert_tag_ready(prereg) is None

    commit_text = (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    amendment_text = (
        ROOT / "prereg" / "pilot-shortfall-amendment.md"
    ).read_text(encoding="utf-8")
    assert "PIN-PENDING" not in commit_text
    assert "ALIGN-WITH-CODE" not in commit_text
    assert "PIN-PENDING" not in md_text
    assert "ALIGN-WITH-CODE" not in md_text
    assert "PIN-PENDING" not in amendment_text
    assert "ALIGN-WITH-CODE" not in amendment_text
    assert amendment_text.startswith("PILOT SHORTFALL AMENDMENT")
    assert "P2 can run only if final T-fit>=20 AND final selection>=30" in (
        amendment_text
    )
    assert amendment_text in md_text

    # The gate still refuses a hidden placeholder even outside pins.
    hidden = prereg.model_copy(deep=True)
    hidden.pilot.shortfall.rule = "hidden PIN-PENDING marker"
    with pytest.raises(ValueError, match="pilot.shortfall.rule"):
        assert_tag_ready(hidden)
    hidden2 = prereg.model_copy(deep=True)
    hidden2.pilot.freeze.required_before = "hidden ALIGN-WITH-CODE marker"
    with pytest.raises(ValueError, match="pilot.freeze.required_before"):
        assert_tag_ready(hidden2)

    # extra="forbid" still rejects unknown keys in the new typed blocks.
    raw = yaml.safe_load(commit_text)
    raw["pilot"]["shortfall"]["not_a_registered_key"] = "x"
    with pytest.raises(ValidationError):
        Prereg.model_validate(raw)

    import copy

    raw = yaml.safe_load(commit_text)
    refusal_cases = [
        (
            lambda data: data["pilot"]["p1"].__setitem__("gates_p2", True),
            "pilot.p1.gates_p2 must be False",
        ),
        (
            lambda data: data["pilot"]["shortfall"].__setitem__(
                "attempt_indices", [4, 5]
            ),
            "pilot.shortfall.attempt_indices must be",
        ),
        (
            lambda data: data["pilot"]["freeze"].__setitem__("required", False),
            "pilot.freeze.required must be True",
        ),
        (
            lambda data: data["pilot"]["code_verification"].__setitem__(
                "docker_image_digest", "python:3.12-slim"
            ),
            "docker_image_digest must bind a Docker content digest",
        ),
        (
            lambda data: data["pilot"]["code_verification"].__setitem__(
                "chat_rule", "chat may count"
            ),
            "chat_rule must state that chat never counts",
        ),
    ]
    for mutate, message in refusal_cases:
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValidationError, match=message):
            Prereg.model_validate(bad)

    flat = _flat_md()
    assert "Revision round 8e" in flat
    for item_id in expected:
        assert item_id in flat
    assert "pilot-shortfall-freeze.json" in flat


# ---------------------------------------------------------------------------
# Round 8f: X1 verifier interpretation and recomputation addendum
# ---------------------------------------------------------------------------


def test_round8f_verifier_v3_addendum_fields_and_hashes():
    import copy
    import hashlib
    import json

    import yaml

    prereg = _committed()
    v3 = prereg.pilot.verifier_v3

    assert v3.addendum_path == "prereg/pilot-shortfall-amendment.md"
    assert v3.addendum_title == (
        "PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM"
    )
    assert v3.status
    assert "v3 alone determines final code membership" in v3.status

    interpretation = v3.interpretation
    assert "pinned dataset revision" in interpretation.source_revision_binding
    assert "record locator" in interpretation.record_locator
    assert "ordered test index" in interpretation.ordered_test_index
    assert "input representation flag" in (
        interpretation.input_representation_flags
    )
    assert "output representation flag" in (
        interpretation.output_representation_flags
    )
    assert "authenticated original" in (
        interpretation.input_representation_flags
    )
    assert "newline joining" in interpretation.flat_list_input_rule
    assert "without adding a terminal newline" in (
        interpretation.flat_list_input_rule
    )
    assert "even when it begins with a bracket" in (
        interpretation.string_unchanged_rule
    )
    assert "text.strip().splitlines()" in interpretation.normalizer
    assert "whole problem is excluded" in interpretation.whole_problem_exclusion
    assert "before v3 grading" in interpretation.exclusion_before_grading
    assert "complete test inventory" in interpretation.exclusion_inventory
    assert "immutable" in interpretation.source_test_immutability

    freeze = v3.freeze
    assert freeze.original_code_requests == 468
    assert freeze.selected_requests == 1144
    assert "all 468" in freeze.recount_rule
    assert "1,144" in freeze.recount_rule
    assert "previous successes and failures" in freeze.regrade_scope
    assert "v3 result alone" in freeze.membership_rule
    assert "union successes" in freeze.no_union
    assert "shortfall.py" in freeze.executing_implementation
    assert "content digest" in freeze.docker_content_digest
    assert "six-second outer timeout" in freeze.timeout

    v2 = v3.v2_recomputation
    assert v2.request_id == "taco:5bc221f9dec75484cb706b33:2"
    assert "wholesale recomputation" in v2.accepted_rule
    assert "87 to 88 rows" in v2.selection_change
    assert "P1 16" in v2.distinct_counts
    assert "5.8592191 seconds" in v2.v2_max_test_duration_s
    assert "cause" in v2.cause_unknown.lower()
    assert "v3 result" in v2.v3_membership
    assert "preferred answer" in v2.no_veto_or_retry

    timeout = v3.timeout_lifecycle
    assert timeout.outer_timeout_s == pytest.approx(6.0)
    assert "including startup" in timeout.timeout_rule
    assert "not authorized" in timeout.timeout_rule
    assert timeout.retry_count == 1
    assert "identical" in timeout.retry_configuration
    assert "first conclusive result" in timeout.first_conclusive
    assert "leaves the count blocked" in timeout.blocked_rule
    assert timeout.attribution_classes == {
        "startup_or_daemon_failure": "infrastructure/unknown",
        "candidate_running_at_outer_deadline": (
            "registered outer-deadline failure"
        ),
        "failure_after_candidate_completion": "infrastructure/unknown",
        "unresolved_attribution": "infrastructure/unknown",
    }
    assert "taco:e25eee8cc5e8629bbbeb6875:0" in timeout.v2_timeout_fact

    acceptance = v3.acceptance
    assert "explicit complete status" in acceptance.count_acceptance
    assert "expected-request coverage" in acceptance.count_acceptance
    assert "shortfall.py" in acceptance.freeze_acceptance
    assert "manifest self-hash" in acceptance.corpus_index_acceptance
    assert "actual corpus bytes" in acceptance.corpus_index_acceptance
    assert "Reject legacy counts" in acceptance.legacy_refusal

    audit = v3.consumer_audit
    joined_audit = " ".join(audit.scope).lower()
    assert "stage 0" in joined_audit
    assert "deepseek" in joined_audit
    assert "evaluation" in joined_audit
    assert "separate official evaluator" in audit.livecodebench
    assert "corrected, authenticated" in audit.active_consumers
    assert "execution-blocked" in audit.future_or_unused
    assert "prerequisite" in audit.seal_prerequisite

    hashes = v3.hashes
    assert hashes.v1 == {
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
    assert hashes.v2 == {
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
    assert hashes.common_test_manifest_sha256 == (
        "3f9d56317ae8a27cab59aca8f6a401fa3254edd56fc44dce13d5a7ac7f5e7e2e"
    )

    order = v3.prerequisite_order
    assert len(order.order) >= 4
    assert "v3 interpretation and verifier freeze" == order.order[0]
    assert "complete v3 original-sample recount" == order.order[1]
    assert "consumer audit" in order.order[2]
    assert "guard repairs" in order.order[3]
    assert "both" in order.before_data_addendum_and_seal
    assert "data addendum" in order.before_data_addendum_and_seal
    assert "seal" in order.before_data_addendum_and_seal

    assert len(v3.superseded_history) >= 3
    joined_history = " ".join(v3.superseded_history).lower()
    assert "currently running code count" in joined_history
    assert "superseded history" in joined_history
    assert "v1 is final" in joined_history
    assert "v2 is final" in joined_history
    assert "not operative" in joined_history
    assert "v3 alone determines" in v3.no_final_before_v3

    # Superseded history is marked at the old clause site too.
    commit_text = (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    assert "SUPERSEDED HISTORY" in commit_text
    assert "currently running code count" in commit_text
    assert "superseded" in md_text.lower()
    assert "currently running code count" in md_text
    assert "PIN-PENDING" not in commit_text
    assert "ALIGN-WITH-CODE" not in commit_text
    assert "PIN-PENDING" not in md_text
    assert "ALIGN-WITH-CODE" not in md_text

    # The completed v3r freeze index carries the accepted recount result.
    freeze_index = json.loads(
        (ROOT / "prereg" / "pilot-shortfall-freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze_index["kind"] == "pilot_shortfall_freeze_index"
    assert freeze_index["status"] == "complete"
    assert freeze_index["run_version"] == "v3r"
    assert freeze_index["count_status"] == "complete"
    assert freeze_index["terminal_infrastructure"]["count"] == 0
    assert freeze_index["terminal_infrastructure"]["per_domain"] == {
        "code": 0,
        "math": 0,
    }
    assert freeze_index["final_counts"] == {
        "pilot_p1": 19,
        "pilot_tfit": 32,
        "pilot_selection": 51,
    }
    assert freeze_index["decision"]["nT0"] == 32
    assert freeze_index["decision"]["nS0"] == 51

    # The addendum is present verbatim in both registered documents.
    amendment_text = (
        ROOT / "prereg" / "pilot-shortfall-amendment.md"
    ).read_text(encoding="utf-8")
    marker = "PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM"
    next_marker = (
        "LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION AND "
        "V3 RESUME AMENDMENT"
    )
    start = amendment_text.index(marker)
    end = amendment_text.index(next_marker, start)
    # Restore the single trailing newline the 8f block had when it was the
    # end of the file; round 8g appends a blank separator plus its own block.
    addendum = amendment_text[start:end].rstrip("\n") + "\n"
    assert hashlib.sha256(addendum.encode("utf-8")).hexdigest() == (
        "5ed406721cbb3dedf0397c1c2a7fc3321eb440db82c43de7e2c375ba6ef8ffc5"
    )
    assert addendum.rstrip("\n") in md_text
    assert not any(
        line.startswith("> ") for line in addendum.splitlines()
    )
    for paragraph in (
        "1. A source-conversion defect",
        "2. A new v3 interpretation manifest",
        "3. For stdin tests",
        "4. The original pool and all v1/v2 artifacts remain immutable",
        "5. V3 shall freeze the corrected interpretation",
        "6. V2 is accepted as a justified wholesale recomputation",
        "7. The registered timeout remains the existing six-second outer timeout",
        "8. Within v3, each infrastructure-affected test may receive one retry",
        "9. Count acceptance requires explicit complete status",
        "10. Before the seal, publish a source/provenance audit",
        "11. Preserve and disclose v1, v2 and v3",
        "12. The v3 verifier freeze, complete recount",
    ):
        assert paragraph in addendum

    # extra="forbid" rejects unknown keys in the new typed blocks.
    raw = yaml.safe_load(commit_text)
    raw["pilot"]["verifier_v3"]["not_a_registered_key"] = "x"
    with pytest.raises(ValidationError):
        Prereg.model_validate(raw)

    raw = yaml.safe_load(commit_text)
    raw["pilot"]["verifier_v3"]["interpretation"]["not_a_registered_key"] = "x"
    with pytest.raises(ValidationError):
        Prereg.model_validate(raw)

    # Missing required keys and wrong registered facts are rejected.
    bad = copy.deepcopy(yaml.safe_load(commit_text))
    del bad["pilot"]["verifier_v3"]["interpretation"]["whole_problem_exclusion"]
    with pytest.raises(
        ValidationError, match="pilot.verifier_v3 fields must be registered"
    ):
        Prereg.model_validate(bad)

    bad = copy.deepcopy(yaml.safe_load(commit_text))
    del bad["pilot"]["verifier_v3"]["consumer_audit"]
    with pytest.raises(
        ValidationError, match="pilot.verifier_v3 fields must be registered"
    ):
        Prereg.model_validate(bad)

    for mutate, message in [
        (
            lambda data: data["pilot"]["verifier_v3"]["freeze"].__setitem__(
                "original_code_requests", 467
            ),
            "original_code_requests must be 468",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["freeze"].__setitem__(
                "selected_requests", 1143
            ),
            "selected_requests must be 1144",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "timeout_lifecycle"
            ].__setitem__("outer_timeout_s", 7),
            "must stay the registered 6-second outer timeout",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "timeout_lifecycle"
            ].__setitem__("retry_count", 2),
            "retry_count must be 1",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "v2_recomputation"
            ].__setitem__("request_id", "taco:other:0"),
            "v2_recomputation.request_id must record",
        ),
    ]:
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        mutate(bad)
        with pytest.raises(ValidationError, match=message):
            Prereg.model_validate(bad)

    flat = _flat_md()
    assert "Revision round 8f" in flat
    assert "PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM" in flat




# ---------------------------------------------------------------------------
# Round 8g: X1 lifecycle evidence, terminal infrastructure and v3 resume
# ---------------------------------------------------------------------------


def test_round8g_lifecycle_terminal_disposition_and_consumer_dispositions():
    import copy
    import hashlib
    import json

    import yaml

    prereg = _committed()
    v3 = prereg.pilot.verifier_v3

    assert v3.lifecycle_addendum_path == "prereg/pilot-shortfall-amendment.md"
    assert v3.lifecycle_addendum_title == (
        "LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION AND "
        "V3 RESUME AMENDMENT"
    )

    lifecycle = v3.lifecycle_evidence
    assert "original-model" in lifecycle.threat_model
    assert "65534:65534" in lifecycle.threat_model
    assert "disclosed residual risk" in lifecycle.residual_same_uid_risk
    assert "same-UID" in lifecycle.residual_same_uid_risk
    assert "fails closed" in lifecycle.fail_closed_rule.lower()
    assert "not candidate-proof isolation" in (
        lifecycle.not_candidate_proof_isolation.lower()
    )
    assert "private child-to-wrapper pipe" in lifecycle.private_pipe
    assert "execution-entry boundary" in lifecycle.execution_entry_boundary
    assert "failed or incomplete" in lifecycle.handoff
    assert "neither the authentication key nor the evidence descriptor" in (
        lifecycle.key_isolation
    )
    assert "HMAC-SHA-256" in lifecycle.hmac_sha256
    assert "SHA256(nonce + payload)" in lifecycle.hmac_sha256
    assert "fresh per-attempt key" in lifecycle.per_attempt_key
    assert "unambiguous" in lifecycle.authenticated_framing
    assert "stdout/stderr" in lifecycle.stdout_stderr_separation
    assert "Every conclusive" in lifecycle.state_specific_validation
    for outcome in (
        lifecycle.pass_outcome,
        lifecycle.wrong_answer_outcome,
        lifecycle.candidate_error_outcome,
        lifecycle.outer_deadline_outcome,
    ):
        assert outcome
    assert "positive evidence" in lifecycle.outer_deadline_outcome
    assert "cannot make a candidate answer acceptable" in (
        lifecycle.post_deadline_rule
    )

    timeout = v3.timeout_lifecycle
    assert "SUPERSEDED HISTORY" in timeout.blocked_rule
    assert "leaves the count blocked" in timeout.blocked_rule
    assert "positive liveness evidence" in timeout.outer_deadline_liveness
    assert "delayed waitpid" in timeout.outer_deadline_liveness

    terminal = v3.terminal_infrastructure
    assert terminal.disposition == "terminal infrastructure-unverified"
    assert terminal.completion_status == "not verified"
    assert "replaces the previous blocked-count rule" in terminal.replace_rule
    assert "wrong answer" in terminal.not_relabelled
    assert "another completion" in terminal.other_completion_rule.lower()
    assert "extension" in terminal.extension_rule.lower()
    assert terminal.count_status == "complete_with_terminal_infrastructure"
    assert "complete_with_terminal_infrastructure" in terminal.count_rule
    assert set(terminal.remaining_blockers) == {
        "incomplete request/test inventories",
        "invalid attempt history",
        "freeze or configuration mismatches",
        "unverified termination/cleanup",
    }
    assert "pending" in terminal.pending_retry
    assert "never zero" in terminal.missing_accounting
    assert "SUPERSEDED HISTORY" in terminal.superseded_blocked_rule

    ledger = v3.attempt_ledger
    assert "durable append-only" in ledger.ledger.lower()
    assert "append-only" in ledger.durable_append_only.lower()
    assert "before dispatch" in ledger.commit_start_before_dispatch
    assert "immediately after completion" in (
        ledger.commit_result_after_completion
    )
    assert "consumes that attempt" in ledger.interrupted_attempt
    assert "third attempt" in ledger.interrupted_retry
    assert "Never regrade" in ledger.never_regrade
    assert "without repeating grading" in ledger.resume_rule
    assert "continuation" in ledger.not_second_freeze.lower()
    assert "not authorization" in ledger.not_second_freeze
    assert "image digest" in ledger.validation_before_resume
    assert "artifact stages" in ledger.artifact_stages
    assert "same run identity" in ledger.freeze_identity

    audit = v3.consumer_audit.dispositions
    assert audit.audit_path.endswith("audit-pool-test-consumers.md")
    assert "96585ad" in audit.audit_head
    assert [row.row for row in audit.rows] == list(range(1, 17))
    assert audit.preserve_pool == [1]
    assert audit.superseded_by_v3 == [4, 6, 7]
    assert audit.active_unaffected == [5, 8, 9, 14, 15]
    assert audit.future_affected_execution_block == [2, 3, 10, 13]
    assert audit.future_unaffected_execution_block == [11, 12]
    assert audit.unaffected_no_action == [16]
    assert "execution-blocked" in audit.execution_block_rule
    assert "authenticated representation" in audit.execution_block_rule
    assert "pilot shortfall corpus" in audit.lift_condition
    assert "byte" in audit.preserve_disclose
    rows = {row.row: row for row in audit.rows}
    assert "grader" in rows[4].consumer.lower()
    assert "shortfall" in rows[6].consumer.lower()
    assert "corpus_v2" in rows[7].consumer
    assert "sample_grades" in rows[5].consumer
    assert "heal" in rows[8].consumer.lower()
    assert "P1-lite" in rows[9].consumer
    assert "postprocess" in rows[2].consumer
    assert "--verify-code" in rows[3].disposition
    assert "DeepSeek _build_prompt" in rows[10].consumer
    assert "hidden-test" in rows[13].consumer.lower()
    assert "LiveCodeBench" in rows[15].consumer
    assert "No active consumer" in rows[16].status

    # Both addendum texts are present verbatim in both registered documents.
    amendment_text = (
        ROOT / "prereg" / "pilot-shortfall-amendment.md"
    ).read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    lifecycle_marker = (
        "LIFECYCLE EVIDENCE, TERMINAL INFRASTRUCTURE DISPOSITION AND "
        "V3 RESUME AMENDMENT"
    )
    start8f = amendment_text.index(
        "PILOT VERIFIER INTERPRETATION AND RECOMPUTATION ADDENDUM"
    )
    start8g = amendment_text.index(lifecycle_marker, start8f)
    start8i = amendment_text.index(
        "V3 VERIFIER IMPLEMENTATION FAILURE, VOID DISPOSITION AND "
        "SINGLE REPLACEMENT ADDENDUM",
        start8g,
    )
    addendum8f = amendment_text[start8f:start8g].rstrip("\n") + "\n"
    addendum8g = amendment_text[start8g:start8i].rstrip("\n") + "\n"
    assert hashlib.sha256(addendum8f.encode("utf-8")).hexdigest() == (
        "5ed406721cbb3dedf0397c1c2a7fc3321eb440db82c43de7e2c375ba6ef8ffc5"
    )
    assert hashlib.sha256(addendum8g.encode("utf-8")).hexdigest() == (
        "0d3471bf16926c8a3d88d9752e5f9d293ee115032510dba4cb40b66777d223e8"
    )
    assert addendum8f.rstrip("\n") in md_text
    assert addendum8g.rstrip("\n") in md_text
    assert not any(line.startswith("> ") for line in addendum8g.splitlines())
    for paragraph in (
        "1. The lifecycle design is approved",
        "2. A frozen bootstrap shall report execution entry",
        "3. The timeout remains six seconds",
        "4. Every conclusive Docker outcome",
        "5. Each infrastructure-affected test receives at most one",
        "6. A count with complete coverage and valid attempt accounting",
        "7. The single official v3 run may resume",
    ):
        assert paragraph in addendum8g

    # No placeholder survives in the registered documents.
    for text in (
        amendment_text,
        md_text,
        (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8"),
        (ROOT / "prereg" / "pilot-shortfall-freeze.json").read_text(
            encoding="utf-8"
        ),
    ):
        assert "PIN-PENDING" not in text
        assert "ALIGN-WITH-CODE" not in text

    freeze_index = json.loads(
        (ROOT / "prereg" / "pilot-shortfall-freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze_index["status"] == "complete"
    assert freeze_index["run_version"] == "v3r"
    assert freeze_index["count_status"] == "complete"
    assert freeze_index["terminal_infrastructure"]["count"] == 0
    assert freeze_index["decision"]["extension_required"] is False
    # The lifecycle rule itself remains registered in the operative YAML block.
    assert (
        v3.terminal_infrastructure.count_status
        == "complete_with_terminal_infrastructure"
    )
    assert "continuation" in v3.attempt_ledger.not_second_freeze.lower()

    commit_text = (ROOT / "prereg" / "prereg.yaml").read_text(
        encoding="utf-8"
    )

    # extra="forbid" and the registered facts are enforced in the new blocks.
    for mutate in (
        lambda data: data["pilot"]["verifier_v3"][
            "lifecycle_evidence"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["pilot"]["verifier_v3"][
            "terminal_infrastructure"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["pilot"]["verifier_v3"][
            "attempt_ledger"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["pilot"]["verifier_v3"]["consumer_audit"][
            "dispositions"
        ].__setitem__("not_a_registered_key", "x"),
    ):
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        mutate(bad)
        with pytest.raises(ValidationError):
            Prereg.model_validate(bad)

    bad = copy.deepcopy(yaml.safe_load(commit_text))
    del bad["pilot"]["verifier_v3"]["lifecycle_evidence"][
        "residual_same_uid_risk"
    ]
    with pytest.raises(
        ValidationError, match="pilot.verifier_v3 fields must be registered"
    ):
        Prereg.model_validate(bad)

    for mutate, message in (
        (
            lambda data: data["pilot"]["verifier_v3"][
                "terminal_infrastructure"
            ].__setitem__("count_status", "complete"),
            "count_status must be complete_with_terminal_infrastructure",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "terminal_infrastructure"
            ].__setitem__("remaining_blockers", ["missing accounting"]),
            "remaining_blockers must register exactly",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "timeout_lifecycle"
            ].__setitem__("blocked_rule", "count stays blocked"),
            "SUPERSEDED HISTORY",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "lifecycle_evidence"
            ].__setitem__("hmac_sha256", "SHA256(nonce + payload)"),
            "actual HMAC-SHA-256",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "attempt_ledger"
            ].__setitem__("not_second_freeze", "a second freeze"),
            "continuation of the same freeze",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["consumer_audit"][
                "dispositions"
            ].__setitem__("superseded_by_v3", [6, 4, 7]),
            "superseded_by_v3 must be \\[4, 6, 7\\]",
        ),
    ):
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        mutate(bad)
        with pytest.raises(ValidationError, match=message):
            Prereg.model_validate(bad)

    flat = _flat_md()
    assert "Revision round 8g" in flat
    assert lifecycle_marker in flat
    assert "complete_with_terminal_infrastructure" in flat
    assert "audit-pool-test-consumers.md" in flat


# ---------------------------------------------------------------------------
# Round 8i: void v3 verifier implementation failure and single v3r replacement
# ---------------------------------------------------------------------------


def test_round8i_void_v3_and_single_replacement():
    import copy
    import hashlib

    import yaml

    prereg = _committed()
    v3 = prereg.pilot.verifier_v3

    void = v3.void_v3
    assert void.status == "VOID_VERIFIER_IMPLEMENTATION_FAILURE"
    assert void.created_at == "2026-11-06T20:23:38.655567+00:00"
    assert void.commit == "b5036ac4d7cd28d354aa55b065671df0444d9bdf"
    assert void.embedded_freeze_manifest_sha256 == (
        "add7f6f49c17d41873be422bb280c1a8a8b315d83b0cc3f13091e815f34ed184"
    )
    assert void.file_sha256 == {
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
    assert "file-byte" in void.file_sha256_definition
    assert "distinct from" in void.file_sha256_definition
    assert "container_names" in void.defect
    assert "attempts" in void.defect
    assert "grade_code" in void.defect
    assert "before that function executed" in void.defect
    assert "1,144" in void.accounting
    assert "468" in void.accounting
    assert "226" in void.accounting
    assert "4,640" in void.accounting
    assert "9,280" in void.accounting
    assert "no pilot candidate execution" in void.accounting
    assert "synthetic" in void.synthetic_recovery.lower()
    assert "not merely" in void.basis
    assert "void" in void.disposition.lower()
    assert "byte-for-byte" in void.preservation
    assert "v3_registry.json" in void.preservation
    assert "non-operative" in void.non_operative_extension
    assert "extension_required" in void.non_operative_extension
    assert "T-fit 16" in void.non_operative_extension
    assert "selection 23" in void.non_operative_extension
    assert "solely" in void.supersedes_scope
    assert "freeze v3 once" in void.supersedes_scope
    assert "terminal-infrastructure" in void.supersedes_scope

    replacement = v3.replacement_v3r
    assert replacement.status
    assert "v3r" in replacement.identity
    assert "not a resume" in replacement.identity
    assert "no seal" in replacement.authorization.lower()
    assert "not approval to launch" in replacement.authorization
    assert "identity forwarding" in replacement.identity_forwarding_repair
    assert "container names" in replacement.identity_forwarding_repair
    assert "attempt tokens" in replacement.identity_forwarding_repair
    assert "grade_code" in replacement.identity_forwarding_repair
    assert "DockerSandbox" in replacement.identity_forwarding_repair
    assert "production-path integration regression" in (
        replacement.production_path_integration_regression
    )
    assert "real grade_code" in (
        replacement.production_path_integration_regression
    )
    assert "DockerSandbox.run_tests" in (
        replacement.production_path_integration_regression
    )
    assert "permissive fake" in (
        replacement.production_path_integration_regression
    )
    assert "real-Docker dry run" in replacement.real_docker_dry_run
    assert "synthetic" in replacement.real_docker_dry_run.lower()
    assert "receipt" in replacement.real_docker_dry_run
    assert "Direct sandbox-only tests do not satisfy" in (
        replacement.real_docker_dry_run
    )
    assert "X1" in replacement.review_and_go
    assert "designated" in replacement.review_and_go
    assert "explicit GO" in replacement.review_and_go
    assert "r13" in replacement.review_and_go
    assert "Freeze v3r once" in replacement.freeze_once
    assert "1,144" in replacement.full_recount
    assert "468" in replacement.full_recount
    assert "complete frozen test set" in replacement.full_recount
    assert "fixed-location continuity record" in replacement.continuity_record
    assert "v3r_registry.json" in replacement.continuity_record
    assert "alternate-output-directory" in replacement.continuity_record
    assert "attempt 1" in replacement.retry_allowance
    assert "identical-configuration" in replacement.retry_allowance
    assert "v3r" in replacement.retry_allowance
    assert "No automatic v3r2" in replacement.no_automatic_v3r2
    assert "favorable-version union" in replacement.preserved_unchanged
    assert "not a third v3 attempt" in replacement.further_changes

    declaration = v3.author_access_declaration
    assert declaration.status == "confirmed"
    assert declaration.requirement
    assert "dated" in declaration.requirement
    assert "original-model verification" in declaration.requirement
    assert declaration.content.strip() == AUTHOR_DECLARATION_ENGLISH
    assert declaration.date == "2026-11-06"
    assert declaration.declaration_sha256 == (
        "aaae7fdd33dd5fff63a620c943e24d0fc27a9318d23f87e75405599f863c16af"
    )

    # The amendment is present verbatim and byte-pinned.
    amendment_text = (
        ROOT / "prereg" / "pilot-shortfall-amendment.md"
    ).read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    marker = (
        "V3 VERIFIER IMPLEMENTATION FAILURE, VOID DISPOSITION AND "
        "SINGLE REPLACEMENT ADDENDUM"
    )
    start = amendment_text.index(marker)
    addendum = amendment_text[start:]
    assert hashlib.sha256(addendum.encode("utf-8")).hexdigest() == (
        "2dd3e5c51a0c747f09acec1c9defc0ae1b45182ffa244c0c0236574e111a486b"
    )
    assert addendum.rstrip("\n") in md_text
    assert not any(line.startswith("> ") for line in addendum.splitlines())
    for paragraph in (
        "1. The official v3 run created at",
        "2. The failed count accounts for 1,144 selected requests",
        "3. Preserve all v1, v2 and failed-v3 artifacts",
        "4. The void count supplies neither operative terminal exclusions",
        "5. Exactly one replacement official run",
        "6. Repair the adapter so the count’s container identities",
        "7. Before official v3r execution, add an integration regression",
        "8. A pre-official real-Docker dry run",
        "9. Before v3r is claimed, X1 or an independent reviewer",
        "10. Maintain a fixed-location continuity record",
        "11. The old attempts remain preserved and consumed",
        "12. V3r shall retain complete accounting for all 1,144 selected requests",
        "13. Disclose this integration defect",
    ):
        assert paragraph in addendum

    # Byte-level equality with the ruling's blockquote, markers stripped.
    ruling_path = Path(
        r"C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/2026-11-03-gemma-loop/"
        r"codex-ruling-v3-void.md"
    )
    if ruling_path.is_file():
        ruling_lines = ruling_path.read_text(encoding="utf-8").splitlines()
        stripped: list[str] = []
        for line in ruling_lines[50:77]:
            if line == ">":
                stripped.append("")
            elif line.startswith("> "):
                stripped.append(line[2:])
            else:
                stripped.append(line)
        assert addendum == "\n".join(stripped) + "\n"

    # No placeholder survives in the registered documents.
    for text in (
        amendment_text,
        md_text,
        (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8"),
        (ROOT / "prereg" / "pilot-shortfall-freeze.json").read_text(
            encoding="utf-8"
        ),
        (ROOT / "prereg" / "prereg-v1-draft-report.md").read_text(
            encoding="utf-8"
        ),
    ):
        assert "PIN-PENDING" not in text
        assert "ALIGN-WITH-CODE" not in text

    commit_text = (ROOT / "prereg" / "prereg.yaml").read_text(
        encoding="utf-8"
    )

    # extra="forbid" rejects unknown keys in the new typed blocks.
    for mutate in (
        lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
            "not_a_registered_key", "x"
        ),
        lambda data: data["pilot"]["verifier_v3"][
            "replacement_v3r"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["pilot"]["verifier_v3"][
            "author_access_declaration"
        ].__setitem__("not_a_registered_key", "x"),
    ):
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        mutate(bad)
        with pytest.raises(ValidationError):
            Prereg.model_validate(bad)

    for key in ("commit", "non_operative_extension", "supersedes_scope"):
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        del bad["pilot"]["verifier_v3"]["void_v3"][key]
        with pytest.raises(
            ValidationError, match="pilot.verifier_v3 fields must be registered"
        ):
            Prereg.model_validate(bad)

    bad = copy.deepcopy(yaml.safe_load(commit_text))
    del bad["pilot"]["verifier_v3"]["replacement_v3r"]["review_and_go"]
    with pytest.raises(
        ValidationError, match="pilot.verifier_v3 fields must be registered"
    ):
        Prereg.model_validate(bad)

    for mutate, message in (
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "status", "COMPLETE"
            ),
            "VOID_VERIFIER_IMPLEMENTATION_FAILURE",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "created_at", "2026-11-06T20:23:38+00:00"
            ),
            "created_at must record",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "commit", "0" * 40
            ),
            "void_v3.commit must record",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "embedded_freeze_manifest_sha256", "0" * 64
            ),
            "embedded_freeze_manifest_sha256",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"][
                "file_sha256"
            ].__setitem__("v3_registry.json", "0" * 64),
            "exact four file-byte SHA-256 values",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "non_operative_extension", "an operative extension"
            ),
            "extension_required is non-operative",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"]["void_v3"].__setitem__(
                "supersedes_scope", "supersedes everything"
            ),
            "solely to the void run and replacement",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "replacement_v3r"
            ].__setitem__("identity", "a third run"),
            "identity must name v3r",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "replacement_v3r"
            ].__setitem__("retry_allowance", "reuse the old attempts"),
            "reset each v3r test to attempt 1",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "replacement_v3r"
            ].__setitem__("no_automatic_v3r2", "another run may follow"),
            "no automatic v3r2",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "author_access_declaration"
            ].__setitem__("status", "declared"),
            "must be pending author confirmation or confirmed",
        ),
        (
            lambda data: data["pilot"]["verifier_v3"][
                "author_access_declaration"
            ].__setitem__("content", "invented declaration"),
            "English rendering verbatim",
        ),
    ):
        bad = copy.deepcopy(yaml.safe_load(commit_text))
        mutate(bad)
        with pytest.raises(ValidationError, match=message):
            Prereg.model_validate(bad)

    flat = _flat_md()
    assert "Revision round 8i" in flat
    assert marker in flat
    assert "VOID_VERIFIER_IMPLEMENTATION_FAILURE" in flat
    assert "confirmed" in flat


def _filled_v3r_raw(raw: dict) -> dict:
    """Return a copy of ``raw``; the committed document is now complete."""

    import copy

    data = copy.deepcopy(raw)
    assert (
        data["revision_round8h"]["data_addendum"]["v3r"]["status"]
        == "complete"
    )
    return data


def _awaiting_v3r_raw(raw: dict) -> dict:
    """Return a copy of the complete ``raw`` reset to the awaiting_v3r state."""

    import copy

    from gemmaloop.prereg import V3R_REQUIRED_FIELDS

    data = copy.deepcopy(raw)
    # The synthetic awaiting state predates round 8j, which requires the
    # completed v3r section; drop the later round so the round-8h check is
    # exercised on its own terms.
    data.pop("revision_round8j", None)
    v3r = data["revision_round8h"]["data_addendum"]["v3r"]
    rule = v3r["extension_decision"]["rule"]
    v3r["status"] = "awaiting_v3r"
    v3r["remaining_fields"] = list(V3R_REQUIRED_FIELDS)
    v3r["hashes"] = {
        "registry_sha256": None,
        "verifier_freeze_sha256": None,
        "count_manifest_sha256": None,
        "attempt_ledger_sha256": None,
        "corpus_sha256": {},
        "freeze_index_sha256": None,
    }
    v3r["distinct_counts"] = {
        split: {"math": None, "code": None, "total": None}
        for split in ("t_fit", "selection", "p1")
    }
    v3r["extension_decision"] = {
        "status": "awaiting_v3r",
        "rule": rule,
        "required": None,
        "additional_requests": None,
        "basis": "",
    }
    v3r["terminal_infrastructure"] = {
        "status": "awaiting_v3r",
        "count_status": None,
        "count": None,
        "inventory_complete": False,
        "cases": [],
    }
    v3r["deltas"] = {
        "v1_to_v2": None,
        "v2_to_v3": None,
        "v3_to_v3r": None,
    }
    v3r["corpus_counts"] = {
        "rows": None,
        "problems": None,
        "input_tokens": None,
        "scored_tokens": None,
        **{
            split: {
                "rows": None,
                "problems": None,
                "input_tokens": None,
                "scored_tokens": None,
            }
            for split in ("p1", "t_fit", "selection")
        },
    }
    return data


def test_round8h_data_addendum_teacher_probe_consumer_author_and_scaling():
    import hashlib

    prereg = _committed()
    assert prereg.revision_round8h is not None
    r8h = prereg.revision_round8h
    assert r8h.date == "2026-11-06"
    addendum = r8h.data_addendum
    assert addendum.date == "2026-11-06"
    assert addendum.status == "complete"

    teacher = addendum.teacher_runtime
    assert teacher.e4b_bf16.status == "FAIL"
    assert teacher.e4b_bf16.metrics.allocator_headroom_gib == pytest.approx(
        0.1748
    )
    assert "0.1748" in teacher.e4b_bf16.decision
    assert "0.25" in teacher.e4b_bf16.decision

    assert teacher.e4b_int8.status == "PASS"
    assert teacher.e4b_int8.metrics.mean_full_kl == pytest.approx(0.0043772008069867824)
    assert teacher.e4b_int8.metrics.p95_full_kl == pytest.approx(0.023100359527117607)
    assert teacher.e4b_int8.metrics.top1_agreement == pytest.approx(0.9821295480723626)
    assert teacher.e4b_int8.metrics.top1_confident == pytest.approx(0.9987044032043211)
    assert teacher.e4b_int8.metrics.allocator_headroom_gib == pytest.approx(
        3.1866
    )

    assert teacher.b12b_nf4.status == "FAIL"
    assert teacher.b12b_nf4.metrics.mean_full_kl == pytest.approx(0.023903793697385142)
    assert teacher.b12b_nf4.metrics.mean_full_kl_upper95 == pytest.approx(
        0.026405724641389978
    )
    assert teacher.b12b_nf4.metrics.p95_full_kl == pytest.approx(0.11805196891873655)
    assert teacher.b12b_nf4.metrics.top1_confident == pytest.approx(0.9888714191728275)

    gguf = teacher.b12b_q4_0_gguf
    assert gguf.status == "EXCLUDED"
    assert gguf.metrics.mean_full_kl == pytest.approx(0.006720703279591921)
    assert gguf.metrics.p95_full_kl == pytest.approx(0.032803291600435915)
    assert gguf.metrics.top1_agreement == pytest.approx(0.9780374619025011)
    assert gguf.metrics.top1_confident == pytest.approx(0.9975914093600509)
    assert gguf.metrics.allocator_headroom_gib == pytest.approx(2.3549)
    assert gguf.metrics.short_mean_full_kl == pytest.approx(0.01302048871711513)
    assert gguf.metrics.short_p95_full_kl == pytest.approx(0.06283502233415036)
    assert "0.0130" in gguf.decision and "0.0628" in gguf.decision
    assert "1.6-2x" in gguf.failure_detail or "1.6-2" in gguf.failure_detail

    assert teacher.selected_teacher == "E4B int8"
    assert teacher.qualified_teacher_count == 1
    assert teacher.excluded_teachers == ["12B NF4", "12B Q4_0 GGUF"]
    assert teacher.exploratory_12b_arm is False
    assert teacher.author_decision_date == "2026-11-06"
    assert "no qualified resident runtime" in teacher.fallback_decision.lower()
    assert teacher.runtime_freeze_sha256 == (
        "d98574294230c8afc1c7b900a7906ad7736e53c667aab5891f51db35f7a5ecc9"
    )
    assert teacher.llama_cpp_tag == "b12169"
    assert teacher.llama_cpp_commit == (
        "3ed7284f2d9e22b2d1a7d05c1416c00c5a9f0bc7"
    )
    assert teacher.gguf_sha256 == (
        "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
    )
    assert teacher.implementation_branch == "task/gguf12b"
    assert teacher.implementation_commit == "5557d6c"

    probe = addendum.probe_memory
    assert probe.status == "PASS"
    assert probe.n8_generation.generation_rows == 24
    assert probe.n8_generation.expected_generation_rows == 24
    assert probe.n8_generation.peak_reserved_gib == pytest.approx(6.791)
    assert probe.n8_generation.headroom_gib == pytest.approx(3.485)
    assert probe.probe_json_sha256 == (
        "94215907201be1439b7435bca554afe2dc6653d5749d9316c84cfec0a72b935d"
    )
    repairs = " ".join(probe.disclosed_repairs).lower()
    for phrase in (
        "text-embedding device",
        "micro-batch 1",
        "token-weighted aggregation",
        "exit-1 bit-exactness",
        "single-pass",
        "nll temperatures",
        "offloaded towers temporarily on the compute device",
        "frozen teacher runtime enforced by the harness",
    ):
        assert phrase in repairs

    consumer = addendum.consumer_audit
    assert "round 8g" in consumer.registered_in.lower()
    assert consumer.dispositions_path == (
        "pilot.verifier_v3.consumer_audit.dispositions"
    )
    assert consumer.audit_sha256 == (
        "f861612ac6634e86063ef5c8112f987364f3ac0c5cef5a568552ba0c4dcd7be6"
    )

    author = addendum.author_declaration
    assert author.status == "confirmed"
    assert author.date == "2026-11-06"
    assert author.declaration_sha256 == (
        "aaae7fdd33dd5fff63a620c943e24d0fc27a9318d23f87e75405599f863c16af"
    )
    assert author.registered_field == (
        "pilot.verifier_v3.author_access_declaration"
    )
    field = prereg.pilot.verifier_v3.author_access_declaration
    assert field.status == "confirmed"
    assert field.content.strip() == AUTHOR_DECLARATION_ENGLISH
    assert field.declaration_sha256 == author.declaration_sha256

    author_path = Path(
        r"C:/Users/<user>/ML/GemmaLoop/.superpowers/sdd/2026-11-03-gemma-loop/"
        r"author-declaration-2026-11-06.md"
    )
    if author_path.is_file():
        assert hashlib.sha256(author_path.read_bytes()).hexdigest() == (
            author.declaration_sha256
        )
        english = author_path.read_text(encoding="utf-8").split(
            "## English rendering for registration", 1
        )[1].strip()
        assert english == AUTHOR_DECLARATION_ENGLISH

    scaling = addendum.exploratory_scaling_curve
    assert "exploratory" in scaling.part.lower()
    assert "best pilot dial arm" in scaling.basis.lower()
    assert "block-matched o-h" in scaling.basis.lower()
    assert scaling.budgets_tokens == [12_000_000, 100_000_000, 300_000_000]
    assert "warmup-stable-decay" in scaling.schedule.lower()
    assert "decay branch of the 300M run" in scaling.branch_rule
    assert "same evaluation as the confirmatory part" in scaling.evaluation
    assert "full curve" in scaling.reporting.lower()
    assert "decided later and disclosed" in scaling.data_source_and_budget
    assert scaling.money_commitment is False
    assert scaling.before_any_pilot_result is True

    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    yaml_text = (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    assert "Revision round 8h (data addendum)" in md_text
    assert "v3r section: complete" in md_text
    assert AUTHOR_DECLARATION_ENGLISH.splitlines()[0] in md_text
    assert "revision_round8h:" in yaml_text


def test_round8h_v3r_complete_passes_tag_mode_and_awaiting_copy_fails(
    tmp_path, capsys
):
    import copy

    import yaml

    import scripts.check_prereg_loads as check

    path = ROOT / "prereg" / "prereg.yaml"
    assert check.main([str(path)]) == 0
    assert check.main([str(path), "--tag-mode"]) == 0

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    complete = _filled_v3r_raw(raw)
    assert (
        Prereg.model_validate(complete)
        .revision_round8h.data_addendum.v3r.status
        == "complete"
    )

    awaiting = _awaiting_v3r_raw(raw)
    awaiting_path = tmp_path / "prereg-v3r-awaiting.yaml"
    awaiting_path.write_text(
        yaml.safe_dump(awaiting, sort_keys=False), encoding="utf-8"
    )
    assert check.main([str(awaiting_path)]) == 0
    assert check.main([str(awaiting_path), "--tag-mode"]) == 2
    stderr = capsys.readouterr().err
    assert "awaiting_v3r" in stderr
    assert "round-8h data addendum v3r" in stderr
    # The failure is the v3r state, not a placeholder or another gate.
    assert ("PIN-PENDING"[:3] + "-PENDING") not in stderr
    assert ("ALIGN-WITH-" + "CODE") not in stderr

    awaiting_model = Prereg.model_validate(awaiting)
    with pytest.raises(ValueError, match="awaiting_v3r"):
        assert_tag_ready(awaiting_model)

    # awaiting_v3r rejects a partially filled result field.
    partial = copy.deepcopy(awaiting)
    partial["revision_round8h"]["data_addendum"]["v3r"]["hashes"][
        "registry_sha256"
    ] = "a" * 64
    with pytest.raises(ValidationError, match="must keep every result field empty"):
        Prereg.model_validate(partial)

    # awaiting_v3r rejects an incomplete remaining_fields list.
    bad_remaining = copy.deepcopy(awaiting)
    bad_remaining["revision_round8h"]["data_addendum"]["v3r"][
        "remaining_fields"
    ].pop()
    with pytest.raises(
        ValidationError, match="remaining_fields must list exactly"
    ):
        Prereg.model_validate(bad_remaining)

    # Complete rejects a missing value.
    missing = _filled_v3r_raw(raw)
    del missing["revision_round8h"]["data_addendum"]["v3r"]["hashes"][
        "freeze_index_sha256"
    ]
    with pytest.raises(ValidationError, match="still missing"):
        Prereg.model_validate(missing)

    # Complete rejects a non-hex hash.
    bad_hash = _filled_v3r_raw(raw)
    bad_hash["revision_round8h"]["data_addendum"]["v3r"]["hashes"][
        "registry_sha256"
    ] = "not-a-sha"
    with pytest.raises(ValidationError, match="lowercase 64-hex"):
        Prereg.model_validate(bad_hash)

    # Complete rejects inconsistent domain and corpus totals.
    bad_total = _filled_v3r_raw(raw)
    bad_total["revision_round8h"]["data_addendum"]["v3r"]["distinct_counts"][
        "t_fit"
    ]["total"] = 9
    with pytest.raises(ValidationError, match=r"total must equal math \+ code"):
        Prereg.model_validate(bad_total)

    bad_corpus = _filled_v3r_raw(raw)
    bad_corpus["revision_round8h"]["data_addendum"]["v3r"]["corpus_counts"][
        "rows"
    ] = 9
    with pytest.raises(ValidationError, match="must equal the split sum"):
        Prereg.model_validate(bad_corpus)

    # Unknown keys are rejected in every new typed block.
    for mutate in (
        lambda data: data["revision_round8h"]["data_addendum"][
            "teacher_runtime"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"][
            "probe_memory"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"][
            "consumer_audit"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"][
            "author_declaration"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"][
            "exploratory_scaling_curve"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"]["v3r"][
            "hashes"
        ].__setitem__("not_a_registered_key", "x"),
        lambda data: data["revision_round8h"]["data_addendum"]["v3r"][
            "terminal_infrastructure"
        ].__setitem__("not_a_registered_key", "x"),
    ):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValidationError):
            Prereg.model_validate(bad)

    # Wrong registered facts are rejected.
    wrong = copy.deepcopy(raw)
    wrong["revision_round8h"]["data_addendum"]["teacher_runtime"][
        "selected_teacher"
    ] = "12B Q4_0 GGUF"
    with pytest.raises(ValidationError, match="must be E4B int8"):
        Prereg.model_validate(wrong)

    wrong = copy.deepcopy(raw)
    wrong["revision_round8h"]["data_addendum"]["teacher_runtime"][
        "b12b_q4_0_gguf"
    ]["status"] = "PASS"
    with pytest.raises(ValidationError, match="status must be EXCLUDED"):
        Prereg.model_validate(wrong)

    wrong = copy.deepcopy(raw)
    wrong["revision_round8h"]["data_addendum"]["probe_memory"][
        "n8_generation"
    ]["generation_rows"] = 23
    with pytest.raises(ValidationError, match="24 generation rows"):
        Prereg.model_validate(wrong)

    wrong = copy.deepcopy(raw)
    wrong["revision_round8h"]["data_addendum"]["author_declaration"][
        "status"
    ] = "pending"
    with pytest.raises(ValidationError, match="must be confirmed"):
        Prereg.model_validate(wrong)

    wrong = copy.deepcopy(raw)
    wrong["revision_round8h"]["data_addendum"]["exploratory_scaling_curve"][
        "money_commitment"
    ] = True
    with pytest.raises(ValidationError, match="money_commitment must be False"):
        Prereg.model_validate(wrong)


def test_round8h_documents_do_not_regress_the_placeholder_gate():
    yaml_text = (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    start = yaml_text.index("revision_round8h:")
    yaml_block = yaml_text[start:]
    md_start = md_text.index("### Revision round 8h (data addendum)")
    md_block = md_text[md_start:]
    for text in (yaml_block, md_block):
        assert ("PIN-PENDING"[:3] + "-PENDING") not in text
        assert ("ALIGN-WITH-" + "CODE") not in text


# ---------------------------------------------------------------------------
# Revision round 8j: the pre-seal pilot training hardware plan
# ---------------------------------------------------------------------------


def test_round8j_pilot_training_hardware_plan_fields():
    prereg = _committed()
    assert prereg.revision_round8j is not None
    r8j = prereg.revision_round8j
    assert r8j.date == "2026-11-07"
    assert "stage a" in r8j.stage_a_exposure.lower()
    assert "no trained-arm outcome" in r8j.outcome_use.lower()
    assert "before any pilot training" in r8j.outcome_use.lower()
    assert len(r8j.choices) >= 5

    plan = r8j.hardware_plan
    assert plan.status == "registered"
    assert plan.date == "2026-11-07"
    assert plan.decided_by == "author"
    assert plan.before_any_pilot_result is True
    assert "before any pilot training" in plan.phase.lower()
    assert "scoring" in plan.phase.lower()

    platforms = {platform.id: platform for platform in plan.platforms}
    assert set(platforms) == {"local", "modal"}
    assert platforms["local"].hardware == "RTX 5070 12 GB"
    assert platforms["local"].memory_gb == 12
    assert platforms["modal"].hardware == "NVIDIA H100 SXM5 80GB"
    assert platforms["modal"].memory_gb == 80
    assert "one arm per container" in platforms["modal"].role.lower()
    for phrase in (
        "identical code commit",
        "configuration",
        "hyperparameters",
        "effective batch",
        "frozen e4b int8 teacher shards",
        "heal corpus",
        "monitor rows",
    ):
        assert phrase in platforms["modal"].role.lower()
    assert "teacher precompute" in platforms["modal"].role.lower()
    assert "temperature fitting" in platforms["modal"].role.lower()
    assert "selection run locally" in platforms["modal"].role.lower()

    groups = {group.id: group for group in plan.groups}
    assert set(groups) == {"A", "B", "C", "D"}
    assert groups["A"].block == "15-29"
    assert set(groups["A"].arms) == {
        "L-15-29-gch-e4b",
        "L-15-29-scalar-e4b",
        "OH-15-29-e4b",
    }
    assert groups["A"].platform == "modal"
    assert groups["A"].may_move_to_modal is False

    assert groups["B"].block == "16-18"
    assert set(groups["B"].arms) == {
        "L-16-18-gch-e4b",
        "OH-16-18-e4b",
    }
    assert groups["B"].platform == "local"
    assert groups["B"].default_platform == "local"
    assert groups["B"].may_move_to_modal is True
    move = groups["B"].move_condition.lower()
    assert "remaining modal credits" in move
    assert "before it has started anywhere" in move

    assert groups["C"].block == "25-29"
    assert set(groups["C"].arms) == {
        "L-25-29-gch-e4b",
        "OH-25-29-e4b",
    }
    assert groups["C"].platform == "local"
    assert groups["C"].together_with == "D"
    assert groups["D"].block == "25-29"
    assert set(groups["D"].arms) == {
        "L-25-29-gch-e4b-single-exit-0901",
        "L-25-29-gch-e4b-single-exit-n4",
    }
    assert groups["D"].platform == "local"
    assert groups["D"].together_with == "C"

    # Every L arm and its block-matched O-H control share one platform.
    arm_platform = {
        arm: group.platform for group in plan.groups for arm in group.arms
    }
    assert arm_platform["L-15-29-gch-e4b"] == arm_platform["OH-15-29-e4b"]
    assert arm_platform["L-15-29-scalar-e4b"] == arm_platform["OH-15-29-e4b"]
    assert arm_platform["L-16-18-gch-e4b"] == arm_platform["OH-16-18-e4b"]
    assert arm_platform["L-25-29-gch-e4b"] == arm_platform["OH-25-29-e4b"]
    for single_exit in (
        "L-25-29-gch-e4b-single-exit-0901",
        "L-25-29-gch-e4b-single-exit-n4",
    ):
        assert arm_platform[single_exit] == arm_platform["OH-25-29-e4b"]

    assert plan.assignment_unit == "block group"
    assert plan.assignment_by_block_group is True
    assert plan.assignment_never_by_individual_arm is True
    assert plan.assignment_uses_only_cost_credits_availability is True
    assert plan.assignment_decided_before_group_start is True
    assert plan.assignment_never_uses_observed_result is True
    basis = plan.assignment_basis.lower()
    assert "cost/credits" in basis
    assert "availability" in basis
    assert "never on any observed result" in basis
    assert "block-matched o-h control" in plan.shared_platform_rule.lower()

    assert plan.run_identity_fields == [
        "hardware",
        "image",
        "library_versions",
        "allocator_cap",
    ]
    assert plan.discloses_every_arm_platform is True
    assert "platform of every arm" in plan.report_disclosure.lower()
    assert "within-platform" in plan.cross_block_comparison.lower()
    assert "l-minus-o-h" in plan.cross_block_comparison.lower()
    assert "bf16 kernels differ across gpus" in plan.limitation.lower()

    provenance = plan.provenance
    assert provenance.corrected_on == "2026-11-07"
    prior = provenance.prior_claim_corrected.lower()
    assert "no hardware measurement" in prior
    assert "synthetic" in prior
    assert provenance.synthetic_data_only is True
    assert provenance.pilot_data_used is False
    assert provenance.pilot_score_used is False
    assert provenance.trained_pilot_outcome_used is False
    assert provenance.informed_plan_before_decision is True
    evidence = {entry.id: entry for entry in provenance.evidence}
    assert set(evidence) == {
        "trainsmoke",
        "blocksmoke",
        "block1529",
        "modal_price_check",
        "credit_cost_estimate",
    }
    assert evidence["trainsmoke"].report_path.endswith("progress.md")
    assert ".superpowers/sdd/2026-11-03-gemma-loop" in (
        evidence["trainsmoke"].report_path
    )
    assert "580" in evidence["trainsmoke"].findings
    assert "9.39" in evidence["trainsmoke"].findings
    assert evidence["blocksmoke"].report_path.endswith(
        "task-blocksmoke-report.md"
    )
    assert "6.85" in evidence["blocksmoke"].findings
    assert "167.6" in evidence["blocksmoke"].findings
    assert "9.99" in evidence["blocksmoke"].findings
    assert "step 2" in evidence["blocksmoke"].findings.lower()
    assert evidence["block1529"].report_path.endswith(
        "task-block1529-report.md"
    )
    block1529 = evidence["block1529"].findings.lower()
    assert "12 gb" in block1529
    assert "vram" in block1529
    assert "no allowed allocator setting" in block1529
    price = evidence["modal_price_check"].findings.lower()
    for marker in ("h100", "3.95", "a100-80gb", "2.50"):
        assert marker in price
    credits = evidence["credit_cost_estimate"].findings.lower()
    assert "26.40" in credits
    assert "24" in credits

    guard = plan.money_guard
    assert guard.remaining_free_credits_usd == pytest.approx(26.40)
    assert guard.internal_cap_usd == pytest.approx(24.0)
    assert guard.cap_below_remaining_credits is True
    assert "monthly ledger" in guard.monthly_ledger
    assert "shared spend lock" in guard.shared_spend_lock
    assert guard.workspace_limit_hard_stop is True
    assert "hard stop" in guard.cap_rule.lower()
    rerun = guard.group_rerun_rule.lower()
    for phrase in ("whole group", "rerun locally", "from scratch", "never mixed"):
        assert phrase in rerun

    assert plan.modal_unavailable_runs_all_local is True
    assert "all groups run locally" in plan.modal_unavailable_fallback.lower()
    assert len(plan.decision_log) == 6
    assert all("2026-11-07" in entry for entry in plan.decision_log)
    assert "provenance correction" in plan.decision_log[-1].lower()
    assert "synthetic" in plan.decision_log[-1].lower()

    yaml_text = (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    md_text = (ROOT / "prereg" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    assert "revision_round8j:" in yaml_text
    assert "### Revision round 8j (pilot training hardware plan)" in md_text
    assert "USD 26.40" in md_text and "USD 24" in md_text
    assert "bf16 kernels differ" in md_text
    assert "whole group is rerun locally from scratch" in md_text


def test_round8j_rejects_wrong_hardware_plan_values():
    import copy

    import yaml

    raw = yaml.safe_load(
        (ROOT / "prereg" / "prereg.yaml").read_text(encoding="utf-8")
    )
    plan = lambda data: data["revision_round8j"]["hardware_plan"]  # noqa: E731

    def _expect(mutate, message):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValidationError, match=message):
            Prereg.model_validate(bad)

    _expect(
        lambda data: data["revision_round8j"].__setitem__("date", "2026-11-06"),
        "revision_round8j.date must be 2026-11-07",
    )
    _expect(
        lambda data: plan(data).__setitem__("date", "2026-11-06"),
        r"hardware_plan.date must be 2026-11-07",
    )
    _expect(
        lambda data: data["revision_round8j"].__setitem__(
            "outcome_use", "no trained-arm outcome"
        ),
        "before any pilot training",
    )
    _expect(
        lambda data: plan(data).__setitem__("decided_by", "reviewer"),
        "decided_by must be the author",
    )
    _expect(
        lambda data: plan(data).__setitem__("before_any_pilot_result", False),
        "before_any_pilot_result must be true",
    )
    _expect(
        lambda data: plan(data)["platforms"][1].__setitem__(
            "hardware", "NVIDIA A100"
        ),
        "hardware must be NVIDIA H100 SXM5 80GB",
    )
    _expect(
        lambda data: plan(data)["platforms"][0].__setitem__("memory_gb", 24),
        r"memory_gb must be 12",
    )

    _expect(
        lambda data: plan(data)["groups"][0].__setitem__("platform", "local"),
        r"groups.A.platform must be modal",
    )
    _expect(
        lambda data: plan(data)["groups"][1].__setitem__("may_move_to_modal", False),
        r"groups.B.may_move_to_modal must be true",
    )
    _expect(
        lambda data: plan(data)["groups"][1].__setitem__(
            "move_condition", "move whenever convenient"
        ),
        "remaining Modal credits",
    )
    _expect(
        lambda data: plan(data)["groups"][2].__setitem__("together_with", "A"),
        "groups C and D must run together",
    )
    _expect(
        lambda data: plan(data)["groups"][3].__setitem__(
            "arms", ["L-25-29-gch-e4b-single-exit-0901"]
        ),
        r"groups.D.arms must be exactly",
    )
    _expect(
        lambda data: plan(data)["groups"].pop(),
        "must key exactly A, B, C and D",
    )

    _expect(
        lambda data: plan(data).__setitem__("assignment_unit", "individual arm"),
        "assignment_unit must be block group",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "assignment_never_by_individual_arm", False
        ),
        "assignment_never_by_individual_arm must be true",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "assignment_never_uses_observed_result", False
        ),
        "assignment_never_uses_observed_result must be true",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "run_identity_fields", ["hardware", "image"]
        ),
        "run_identity_fields must be",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "cross_block_comparison", "compare across GPUs"
        ),
        "cross_block_comparison must compare within-platform",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "limitation", "no known limitation"
        ),
        "bf16 kernels differ across GPUs",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__(
            "remaining_free_credits_usd", 30.0
        ),
        r"remaining_free_credits_usd must be 26.40",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__("internal_cap_usd", 25.0),
        r"internal_cap_usd must be 24",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__(
            "cap_below_remaining_credits", False
        ),
        "cap_below_remaining_credits must be true",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__(
            "workspace_limit_hard_stop", False
        ),
        "workspace_limit_hard_stop must be true",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__(
            "group_rerun_rule", "rerun only the failed arm"
        ),
        "whole-group local rerun",
    )
    _expect(
        lambda data: plan(data).__setitem__(
            "modal_unavailable_fallback", "wait for Modal"
        ),
        "run all groups locally",
    )
    _expect(
        lambda data: plan(data).__setitem__("decision_log", []),
        "decision_log must not be empty",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "prior_claim_corrected", "nothing to correct"
        ),
        "prior_claim_corrected must correct",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "synthetic_data_only", False
        ),
        "synthetic_data_only must be true",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "pilot_data_used", True
        ),
        "pilot_data_used must be false",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "pilot_score_used", True
        ),
        "pilot_score_used must be false",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "trained_pilot_outcome_used", True
        ),
        "trained_pilot_outcome_used must be false",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "informed_plan_before_decision", False
        ),
        "informed_plan_before_decision must be true",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "corrected_on", "2026-11-06"
        ),
        "provenance.corrected_on must be 2026-11-07",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"].pop(),
        "must register trainsmoke, blocksmoke, block1529",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"][0].__setitem__(
            "findings", "nothing"
        ),
        "trainsmoke must record",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"][1].__setitem__(
            "findings", "no failure"
        ),
        "blocksmoke must record",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"][2].__setitem__(
            "findings", "fine"
        ),
        "block1529 must record",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"][3].__setitem__(
            "findings", "no prices"
        ),
        "modal_price_check must record",
    )
    _expect(
        lambda data: plan(data)["provenance"]["evidence"][4].__setitem__(
            "findings", "no credits"
        ),
        "credit_cost_estimate must record",
    )
    _expect(
        lambda data: plan(data)["provenance"].__setitem__(
            "not_a_registered_key", "x"
        ),
        "Extra inputs are not permitted",
    )
    _expect(
        lambda data: plan(data).__setitem__("not_a_registered_key", "x"),
        "Extra inputs are not permitted",
    )
    _expect(
        lambda data: plan(data)["money_guard"].__setitem__(
            "not_a_registered_key", "x"
        ),
        "Extra inputs are not permitted",
    )


def test_round8j_loads_in_plain_and_tag_mode():
    import scripts.check_prereg_loads as check

    path = ROOT / "prereg" / "prereg.yaml"
    assert check.main([str(path)]) == 0
    assert check.main([str(path), "--tag-mode"]) == 0
    prereg = _committed()
    assert prereg.revision_round8j is not None
    assert_tag_ready(prereg)
