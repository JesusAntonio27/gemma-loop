"""The registered two-step pilot selection: teacher first, then rule B.

Nothing here reads a result file directly.  Callers pass the machine-readable
per-arm summaries produced by the evaluation stage; this module applies only
the thresholds registered in ``pilot.p2``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .statistics import monotonic_ok

__all__ = [
    "RULE_B_TIE_NATS",
    "SINGLE_TEACHER_REASON",
    "TEACHER_TIE_NATS",
    "select_block_rule_b",
    "select_single_teacher",
    "select_teacher",
]

TEACHER_TIE_NATS = 0.003
RULE_B_TIE_NATS = 0.003
MIN_ARM_VS_OH_NATS = -0.02
TEACHER_KEYS = ("e4b", "12b")
SINGLE_TEACHER_REASON = (
    "the teacher comparison is not estimable: the registration excludes the "
    "other teacher (no qualified resident runtime), so there is a single "
    "qualified teacher and the recipe teacher is E4B"
)


def _nll(arm: Mapping, n: int) -> float:
    values = arm.get("nll") or {}
    for key in (n, str(n)):
        if key in values:
            return float(values[key])
    raise ValueError(f"arm {arm.get('arm_id')!r} has no N={n} calibrated NLL")


def _teacher_pair(arms: Mapping[str, Mapping], teacher: str) -> tuple[Mapping, Mapping]:
    suffixes = ("12b",) if str(teacher) == "12b" else ("e4b",)
    for suffix in suffixes:
        loop = arms.get(f"L-25-29-gch-{suffix}")
        oh = arms.get(f"OH-25-29-{suffix}")
        if loop is not None and oh is not None:
            return loop, oh
    # Compatibility with callers that key both teacher maps with the E4B ids.
    loop = arms.get("L-25-29-gch-e4b")
    oh = arms.get("OH-25-29-e4b")
    if loop is None or oh is None:
        raise ValueError(f"teacher step for {teacher!r} needs L[25-29] and O-H[25-29]")
    return loop, oh


def teacher_gain(
    loop: Mapping,
    oh: Mapping,
    baseline_nll: float,
) -> dict[str, float]:
    """Gain over O: L at the primary N=4 and O-H at N=1, then the mean."""

    l_gain = float(baseline_nll) - _nll(loop, 4)
    oh_gain = float(baseline_nll) - _nll(oh, 1)
    return {
        "l_n4": l_gain,
        "oh_n1": oh_gain,
        "mean": (l_gain + oh_gain) / 2.0,
    }


def select_teacher(
    e4b_arms: Mapping[str, Mapping],
    twelve_b_arms: Mapping[str, Mapping],
    baseline_nll: float,
    *,
    tie_nats: float = TEACHER_TIE_NATS,
) -> dict:
    """Choose the teacher by mean gain; E4B wins anything within ``tie_nats``."""

    e4b_loop, e4b_oh = _teacher_pair(e4b_arms, "e4b")
    twelve_loop, twelve_oh = _teacher_pair(twelve_b_arms, "12b")
    gains = {
        "e4b": teacher_gain(e4b_loop, e4b_oh, baseline_nll),
        "12b": teacher_gain(twelve_loop, twelve_oh, baseline_nll),
    }
    margin = gains["12b"]["mean"] - gains["e4b"]["mean"]
    if margin > float(tie_nats):
        chosen = "12b"
        tie_break = False
        reason = "12B mean gain exceeds E4B by more than the registered tie width"
    else:
        chosen = "e4b"
        tie_break = abs(margin) <= float(tie_nats)
        reason = (
            "E4B wins the registered tie-break (12B is not ahead by more than "
            f"{float(tie_nats):g} nats)"
            if margin > 0
            else "E4B has the larger mean gain"
        )
    return {
        "chosen": chosen,
        "tie_break": bool(tie_break),
        "margin_12b_minus_e4b": float(margin),
        "tie_nats": float(tie_nats),
        "baseline_nll": float(baseline_nll),
        "gains": gains,
        "reason": reason,
    }


def select_single_teacher(
    e4b_arms: Mapping[str, Mapping],
    *,
    excluded_teachers: Sequence[str] = ("12b",),
    baseline_nll: float | None = None,
) -> dict:
    """Record the registered single-teacher step without inventing a comparison.

    When the round-8h data addendum excludes every runtime of the other teacher
    there is no 12B pair to compare, so the teacher comparison is not
    estimable.  The recipe teacher is the single qualified teacher, E4B.  This
    does not change any registered selection statistic; it records the
    structural fact so the block stage can proceed with the E4B 25-29 pair.
    """

    record = {
        "chosen": "e4b",
        "recipe_teacher": "e4b",
        "estimable": False,
        "single_teacher": True,
        "qualified_teacher_count": 1,
        "excluded_teachers": sorted(str(item) for item in excluded_teachers),
        "tie_break": False,
        "reason": SINGLE_TEACHER_REASON,
    }
    if baseline_nll is not None:
        try:
            loop, oh = _teacher_pair(e4b_arms, "e4b")
        except ValueError:
            pass
        else:
            record["baseline_nll"] = float(baseline_nll)
            record["e4b_gain"] = teacher_gain(loop, oh, baseline_nll)
    return record


def _tie_key(arm: Mapping) -> tuple:
    bridge = str(arm.get("bridge", "scalar"))
    block = arm.get("block") or [0, 0]
    block_size = int(block[1]) - int(block[0]) + 1
    ratio = arm.get("flops_ratio")
    flops_ratio = float(ratio) if ratio is not None else float("inf")
    return (
        flops_ratio,
        0 if bridge == "gch" else 1,
        block_size,
        int(block[0]),
        str(arm.get("arm_id", "")),
    )


def select_block_rule_b(
    arms: Mapping[str, Mapping],
    *,
    tie_nats: float = RULE_B_TIE_NATS,
    min_delta_arm: float = MIN_ARM_VS_OH_NATS,
    monotonic_tolerance: float = 0.005,
) -> dict:
    """Apply the frozen eligibility, ranking, filter and tie-breaks.

    ``arms`` maps arm id -> a summary containing ``kind``, ``block``,
    ``bridge``, ``teacher``, ``registered_teacher``, ``eligible``, ``nll``
    (N -> point estimate), ``delta_loop_lower``, ``delta_arm``,
    ``delta_bar_arm`` and ``flops_ratio``.
    """

    eligible: list[Mapping] = []
    for arm in arms.values():
        if str(arm.get("kind")) != "L":
            continue
        if arm.get("eligible") is False:
            continue
        if str(arm.get("registered_teacher", arm.get("teacher"))) != "e4b":
            continue
        if str(arm.get("teacher")) != "e4b":
            continue
        if float(arm.get("delta_loop_lower", float("-inf"))) <= 0.0:
            continue
        delta_arm = arm.get("delta_arm")
        if delta_arm is None or float(delta_arm) < float(min_delta_arm):
            continue
        if arm.get("delta_bar_arm") is None:
            continue
        eligible.append(arm)

    eligible_ids = [str(arm["arm_id"]) for arm in eligible]
    if not eligible:
        return {
            "stop": True,
            "selected": None,
            "eligible": [],
            "ranked": [],
            "monotonic_pass": [],
            "filter_applied": False,
            "tie_candidates": [],
            "reason": "the frozen eligible set E is empty; publish the mechanism null and STOP",
        }

    def _monotonic(arm: Mapping) -> bool:
        nll = arm.get("nll") or {}
        converted = {}
        for n in (1, 2, 3, 4, 6, 8):
            if n in nll or str(n) in nll:
                converted[n] = float(nll[n] if n in nll else nll[str(n)])
        return monotonic_ok(converted, tolerance=monotonic_tolerance)

    ranked = sorted(
        eligible,
        key=lambda arm: (-float(arm.get("delta_bar_arm", 0.0)), str(arm["arm_id"])),
    )
    monotonic_pass = [str(arm["arm_id"]) for arm in ranked if _monotonic(arm)]
    pool = [arm for arm in ranked if _monotonic(arm)] if monotonic_pass else ranked
    filter_applied = bool(monotonic_pass)

    best_value = max(float(arm.get("delta_bar_arm", 0.0)) for arm in pool)
    candidates = [
        arm
        for arm in pool
        if best_value - float(arm.get("delta_bar_arm", 0.0)) <= float(tie_nats)
    ]
    candidates.sort(key=_tie_key)
    selected = candidates[0]

    return {
        "stop": False,
        "selected": str(selected["arm_id"]),
        "selected_summary": dict(selected),
        "eligible": eligible_ids,
        "ranked": [
            {"arm_id": str(arm["arm_id"]), "delta_bar_arm": float(arm.get("delta_bar_arm", 0.0))}
            for arm in ranked
        ],
        "monotonic_pass": monotonic_pass,
        "filter_applied": bool(filter_applied),
        "tie_candidates": [str(arm["arm_id"]) for arm in candidates],
        "tie_nats": float(tie_nats),
        "reason": (
            "selected the argmax of Delta_bar_arm among the monotonic passing arms"
            if filter_applied
            else "no eligible arm passed the monotonicity filter; ranked the unfiltered eligible set"
        ),
    }
