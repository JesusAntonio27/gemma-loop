"""Registered pilot arm parsing and the two-step selection rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemmaloop.pilot.arms import (
    ArmSpec,
    block_stage_arms,
    parse_registered_arms,
    pilot_arms,
    teacher_stage_arms,
)
from gemmaloop.pilot.selection import (
    select_block_rule_b,
    select_teacher,
)
from gemmaloop.pilot.statistics import (
    bootstrap_ci,
    monotonic_ok,
    paired_bootstrap_ci,
)
from gemmaloop.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_PREREG = Path(
    r"C:/Users/<user>/ML/GemmaLoop-wt/prereg1/prereg/prereg.yaml"
)


def _prereg():
    return load_prereg(ROOT / "prereg" / "prereg.yaml")


def test_authoritative_external_prereg_loads_through_the_strict_model():
    """Release gate: the exact round-8b file must load through ``load_prereg``."""

    if not AUTHORITATIVE_PREREG.is_file():
        pytest.skip(f"external prereg worktree is absent: {AUTHORITATIVE_PREREG}")
    prereg = load_prereg(AUTHORITATIVE_PREREG)
    assert prereg.monitor is not None
    assert prereg.grading is not None
    assert prereg.external_apis is not None
    assert prereg.revision_round8b is not None
    assert prereg.monitor.abort_exit_code == 76
    assert prereg.external_apis.deepseek_caps_sum_usd == pytest.approx(71.0)
    assert len(prereg.revision_round8b.items) == 11
    assert len(pilot_arms(prereg)) == 11


def test_registered_arms_expand_to_the_eleven_arm_matrix():
    arms = pilot_arms(_prereg())
    assert len(arms) == 11
    ids = [arm.arm_id for arm in arms]
    assert len(set(ids)) == len(ids)
    assert "L-25-29-gch-e4b" in ids
    assert "L-25-29-gch-12b" in ids
    assert "OH-25-29-e4b" in ids
    assert "OH-25-29-12b" in ids
    # The two added O-H controls are the block-matched E4B O-H arms.
    assert "OH-16-18-e4b" in ids
    assert "OH-15-29-e4b" in ids
    # Round-7 single-exit arms: descriptive and never eligible.
    single = [arm for arm in arms if arm.single_exit]
    assert len(single) == 2
    assert all(arm.descriptive and not arm.eligible for arm in single)

    teacher_stage = teacher_stage_arms(arms)
    assert {arm.arm_id for arm in teacher_stage} == {
        "L-25-29-gch-e4b",
        "L-25-29-gch-12b",
        "OH-25-29-e4b",
        "OH-25-29-12b",
    }
    block_stage = block_stage_arms(arms)
    assert {arm.arm_id for arm in block_stage} == {
        "L-16-18-gch-e4b",
        "L-15-29-gch-e4b",
        "L-15-29-scalar-e4b",
        "OH-16-18-e4b",
        "OH-15-29-e4b",
        "L-25-29-gch-e4b-single-exit-0901",
        "L-25-29-gch-e4b-single-exit-n4",
    }
    assert {arm.arm_id for arm in teacher_stage} | {
        arm.arm_id for arm in block_stage
    } == set(ids)


def test_parser_accepts_the_authoritative_explicit_arm_strings():
    authoritative = [
        "25-29 + zero-init g + E4B",
        "16-18 + zero-init g + E4B",
        "15-29 + zero-init g + E4B",
        "15-29 + legacy scalar + E4B",
        "25-29 + zero-init g + 12B QAT q4_0",
        "O-H[25-29] + E4B",
        "O-H[25-29] + 12B QAT q4_0",
        "O-H[16-18] + E4B",
        "O-H[15-29] + E4B",
        "25-29 + zero-init g + E4B + single-exit legacy (N 2/3 at 0.9/0.1)",
        "25-29 + zero-init g + E4B + single-exit legacy (N=4 fixed)",
    ]
    arms = parse_registered_arms(authoritative)
    assert len(arms) == 11
    assert {arm.arm_id for arm in arms} == {arm.arm_id for arm in pilot_arms(_prereg())}
    explicit = {arm.arm_id: arm for arm in arms if arm.kind == "O-H"}
    assert explicit["OH-16-18-e4b"].block == (16, 18)
    assert explicit["OH-25-29-12b"].teacher == "12b"

    authoritative_yaml = Path(
        r"C:/Users/<user>/ML/GemmaLoop-wt/prereg1/prereg/prereg.yaml"
    )
    if authoritative_yaml.is_file():
        import yaml

        raw = yaml.safe_load(authoritative_yaml.read_text(encoding="utf-8"))
        registered = raw["pilot"]["p2"]["arms"]
        parsed = parse_registered_arms(registered)
        assert len(parsed) == 11
        assert {arm.arm_id for arm in parsed} == {arm.arm_id for arm in arms}


def test_run_pilot_plan_loads_both_authoritative_prereg_files():
    import scripts.run_pilot as run_pilot

    paths = [ROOT / "prereg" / "prereg.yaml"]
    if AUTHORITATIVE_PREREG.is_file():
        paths.append(AUTHORITATIVE_PREREG)
    for path in paths:
        prereg = load_prereg(path)
        assert prereg.pilot.why_role
        assert prereg.pilot.why_metrics
        plan = run_pilot.plan_stage(prereg, "all")
        assert len(plan["arms"]) == 11
        assert {arm.arm_id for arm in plan["teacher_stage"]} == {
            "L-25-29-gch-e4b",
            "L-25-29-gch-12b",
            "OH-25-29-e4b",
            "OH-25-29-12b",
        }
        required = {arm.arm_id for arm in plan["required"]}
        assert required == {
            "L-25-29-gch-e4b",
            "L-25-29-gch-12b",
            "OH-25-29-e4b",
            "OH-25-29-12b",
            "L-16-18-gch-e4b",
            "OH-16-18-e4b",
            "L-15-29-gch-e4b",
            "OH-15-29-e4b",
            "L-15-29-scalar-e4b",
        }
        assert "OH-16-18-e4b" in {arm.arm_id for arm in plan["arms"]}


def test_arm_parsing_rejects_unknown_arm_text():
    with pytest.raises(ValueError):
        parse_registered_arms(["25-29 + magic_bridge + E4B"])


def _nll(value: float) -> dict[int, float]:
    return {1: value + 0.03, 2: value + 0.02, 3: value + 0.01, 4: value, 6: value - 0.01, 8: value - 0.02}


def _arm(
    arm_id: str = "L-25-29-gch-e4b",
    *,
    kind: str = "L",
    block=(25, 29),
    bridge: str = "gch",
    teacher: str = "e4b",
    registered_teacher: str | None = None,
    nll: dict[int, float] | None = None,
    delta_loop_lower: float = 0.01,
    delta_arm: float = 0.01,
    delta_bar: float | None = None,
    flops_ratio: float = 1.1,
    eligible: bool = True,
) -> dict:
    return {
        "arm_id": arm_id,
        "kind": kind,
        "block": list(block),
        "bridge": bridge,
        "teacher": teacher,
        "registered_teacher": registered_teacher or teacher,
        "nll": nll or _nll(0.9),
        "delta_loop_lower": delta_loop_lower,
        "delta_arm": delta_arm,
        "delta_bar_arm": delta_bar,
        "flops_ratio": flops_ratio,
        "eligible": bool(eligible) and not arm_id.startswith("single"),
    }


def _teacher_arm_ids(teacher: str, *, nll4: float = 0.80, oh1: float = 0.95) -> dict:
    return {
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", teacher=teacher, nll={1: 1.0, 4: nll4, 2: 0.9, 3: 0.85, 6: 0.79, 8: 0.78}),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH", teacher=teacher, nll={1: oh1}),
    }


def test_teacher_step_takes_the_larger_mean_gain_and_e4b_breaks_ties():
    baseline = 1.0
    # 12B's mean gain = ((1.0-0.80) + (1.0-0.95))/2 = 0.125; E4B's = 0.10.
    far = select_teacher(
        _teacher_arm_ids("e4b"),
        _teacher_arm_ids("12b", nll4=0.77, oh1=0.93),
        baseline,
    )
    assert far["chosen"] == "12b"
    assert far["gains"]["12b"]["mean"] > far["gains"]["e4b"]["mean"]

    # A gap inside 0.003 nats is an E4B win, even when 12B is nominally ahead.
    close = _teacher_arm_ids("12b", nll4=0.8005, oh1=0.9505)
    tied = select_teacher(_teacher_arm_ids("e4b"), close, baseline)
    assert tied["chosen"] == "e4b"
    assert tied["tie_break"] is True


def test_rule_b_stop_when_the_eligible_set_is_empty():
    arms = {
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", delta_loop_lower=0.0),
        "L-16-18-gch-e4b": _arm("L-16-18-gch-e4b", delta_loop_lower=-0.1),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH"),
    }
    result = select_block_rule_b(arms)
    assert result["stop"] is True
    assert result["selected"] is None
    assert result["eligible"] == []


def test_rule_b_ranking_monotone_filter_and_fallback():
    arms = {
        # Eligible, monotone, Delta_bar 0.02.
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", delta_bar=0.02, nll={1: 1.00, 2: 0.99, 3: 0.985, 4: 0.98, 6: 0.975, 8: 0.97}),
        # Better Delta_bar but fails the N=1->2 adjacent-regression filter.
        "L-16-18-gch-e4b": _arm("L-16-18-gch-e4b", block=(16, 18), delta_bar=0.03, nll={1: 1.00, 2: 1.02, 3: 1.01, 4: 1.00, 6: 0.99, 8: 0.98}),
        # Not eligible: delta_loop lower bound is not positive.
        "L-15-29-gch-e4b": _arm("L-15-29-gch-e4b", block=(15, 29), delta_bar=0.05, delta_loop_lower=-0.001),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH"),
    }
    result = select_block_rule_b(arms)
    assert result["stop"] is False
    assert result["selected"] == "L-25-29-gch-e4b"
    assert set(result["eligible"]) == {"L-25-29-gch-e4b", "L-16-18-gch-e4b"}
    assert result["monotonic_pass"] == ["L-25-29-gch-e4b"]

    # If no eligible arm passes the filter, rank the unfiltered eligible set.
    arms["L-25-29-gch-e4b"]["nll"] = {1: 1.00, 2: 1.02, 3: 1.01, 4: 1.00, 6: 0.99, 8: 0.98}
    fallback = select_block_rule_b(arms)
    assert fallback["filter_applied"] is False
    assert fallback["selected"] == "L-16-18-gch-e4b"


def test_rule_b_ties_use_flops_then_bridge_then_block_size():
    # Three arms within 0.003 of the maximum: the lower-FLOPs arm wins.
    arms = {
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", delta_bar=0.0300, flops_ratio=1.2),
        "L-16-18-gch-e4b": _arm("L-16-18-gch-e4b", block=(16, 18), delta_bar=0.0295, flops_ratio=1.1),
        "L-15-29-gch-e4b": _arm("L-15-29-gch-e4b", block=(15, 29), delta_bar=0.0280, flops_ratio=1.3),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH"),
    }
    assert select_block_rule_b(arms)["selected"] == "L-16-18-gch-e4b"

    # Equal FLOPs: the zero-init bridge beats the legacy scalar.
    arms["L-25-29-gch-e4b"].update(flops_ratio=1.1, bridge="scalar")
    arms["L-16-18-gch-e4b"].update(flops_ratio=1.1, bridge="gch")
    assert select_block_rule_b(arms)["selected"] == "L-16-18-gch-e4b"

    # Equal FLOPs and bridge: the smaller loop block wins.
    arms["L-16-18-gch-e4b"]["bridge"] = "scalar"
    arms["L-25-29-gch-e4b"]["bridge"] = "scalar"
    arms["L-16-18-gch-e4b"]["flops_ratio"] = 1.1
    arms["L-25-29-gch-e4b"]["flops_ratio"] = 1.1
    assert select_block_rule_b(arms)["selected"] == "L-16-18-gch-e4b"


def test_missing_flops_ratio_orders_the_arm_last_in_a_tie():
    arms = {
        "L-25-29-gch-e4b": _arm(
            "L-25-29-gch-e4b", delta_bar=0.030, flops_ratio=None
        ),
        "L-16-18-gch-e4b": _arm(
            "L-16-18-gch-e4b", block=(16, 18), delta_bar=0.0295, flops_ratio=1.2
        ),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH"),
    }
    result = select_block_rule_b(arms)
    assert result["selected"] == "L-16-18-gch-e4b"


def test_teacher_step_accepts_teacher_specific_arm_ids():
    e4b = {
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", nll={1: 1.0, 4: 0.80, 2: 0.9, 3: 0.85, 6: 0.79, 8: 0.78}),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH", nll={1: 0.95}),
    }
    twelve = {
        "L-25-29-gch-12b": _arm("L-25-29-gch-12b", teacher="12b", nll={1: 1.0, 4: 0.70, 2: 0.9, 3: 0.85, 6: 0.79, 8: 0.78}),
        "OH-25-29-12b": _arm("OH-25-29-12b", kind="OH", teacher="12b", nll={1: 0.90}),
    }
    result = select_teacher(e4b, twelve, 1.0)
    assert result["chosen"] == "12b"
    assert result["gains"]["12b"]["mean"] > result["gains"]["e4b"]["mean"]


def test_ineligible_teacher_and_descriptive_arms_never_enter_rule_b():
    arms = {
        "L-25-29-gch-12b": _arm(
            "L-25-29-gch-12b", teacher="12b", delta_bar=0.9, delta_loop_lower=0.1
        ),
        "L-25-29-gch-e4b-single-exit-n4": _arm(
            "L-25-29-gch-e4b-single-exit-n4", delta_bar=0.9, eligible=False
        ),
        # Registered as an E4B arm but actually trained with 12B: not an E4B arm.
        "L-25-29-gch-e4b-single-exit-0901": _arm(
            "L-25-29-gch-e4b-single-exit-0901", teacher="12b", delta_bar=0.9
        ),
        "L-25-29-gch-e4b": _arm("L-25-29-gch-e4b", delta_bar=0.02),
        "OH-25-29-e4b": _arm("OH-25-29-e4b", kind="OH"),
    }
    result = select_block_rule_b(arms)
    assert result["eligible"] == ["L-25-29-gch-e4b"]
    assert result["selected"] == "L-25-29-gch-e4b"


def test_cluster_bootstrap_and_monotonicity_helpers():
    values = {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0}
    lo, hi = bootstrap_ci(values, iters=2000, seed=0)
    assert lo <= 2.5 <= hi
    paired_lo, paired_hi = paired_bootstrap_ci(values, values, iters=2000, seed=0)
    assert paired_lo == pytest.approx(0.0)
    assert paired_hi == pytest.approx(0.0)
    assert monotonic_ok({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 6: 1.0, 8: 1.0})
    assert not monotonic_ok({1: 1.0, 2: 1.01, 3: 1.0, 4: 1.0, 6: 1.0, 8: 1.0})
    assert not monotonic_ok({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 6: 1.0, 8: 1.01})
