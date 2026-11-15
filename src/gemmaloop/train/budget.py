"""Token budget arithmetic for the looped arms.

``L`` trains for the same number of *data* tokens as ``O-H`` but spends more
compute per token because the block is executed several times.  To compare
quality at equal compute, ``O-H-ext`` is trained on a larger number of tokens
whose analytic FLOPs match the expected FLOPs of ``L``.

Per-layer FLOPs are **not** redefined here: :func:`per_layer_flops` is imported
from :mod:`gemmaloop.model.flops` (controller ruling for Task 13).
"""

from __future__ import annotations

from collections.abc import Sequence

from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.layer_plan import LayerPlan
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS, _validate_probs

__all__ = [
    "expected_layer_counts",
    "flops_matched_tokens",
    "per_layer_flops",
]


def expected_layer_counts(
    plan: LayerPlan, probs: Sequence[float] = DEFAULT_LOOP_PROBS
) -> list[float]:
    """Return ``E[count_i]`` for every physical layer.

    Layers inside ``plan.loop`` are executed once per block pass, so their
    expected count is ``E[N] = sum(n * p_n)``.  Prelude and coda layers are
    always executed exactly once.
    """
    values = _validate_probs(probs)
    expected_n = sum(n * p for n, p in enumerate(values, start=1))
    counts = [1.0] * plan.num_layers
    for layer in plan.loop:
        if layer < 0 or layer >= plan.num_layers:
            raise ValueError(
                f"plan.loop contains layer {layer} outside 0..{plan.num_layers - 1}"
            )
        counts[layer] = expected_n
    return counts


def flops_matched_tokens(
    base_tokens: int,
    per_layer_flops: list[int],
    plan: LayerPlan,
    probs: Sequence[float] = DEFAULT_LOOP_PROBS,
) -> int:
    """Tokens ``O-H-ext`` needs to match ``L``'s expected FLOPs.

    ``round(base_tokens * sum_i E[count_i] * f_i / sum_i f_i)``.

    This is the brief's per-layer formula.  It deliberately ignores the
    model-level ``constant_flops`` (LM head and PLE input projection) carried by
    ``LayerFlops``: including the constant would yield a slightly smaller ratio
    ``(C + W) / (C + T) < W / T`` and therefore a slightly smaller budget.  The
    omission over-provisions ``O-H-ext`` marginally; report the actual consumed
    token count (``tokens_consumed`` in ``metrics.json``) when computing FLOPs.
    """
    if base_tokens < 0:
        raise ValueError(f"base_tokens must be non-negative, got {base_tokens!r}")
    if len(per_layer_flops) < plan.num_layers:
        raise ValueError(
            f"per_layer_flops has {len(per_layer_flops)} entries, "
            f"expected at least {plan.num_layers}"
        )
    total_flops = sum(int(per_layer_flops[i]) for i in range(plan.num_layers))
    if total_flops <= 0:
        raise ValueError("per-layer FLOPs must sum to a positive value")

    counts = expected_layer_counts(plan, probs)
    weighted = sum(
        counts[i] * int(per_layer_flops[i]) for i in range(plan.num_layers)
    )
    return int(round(base_tokens * weighted / total_flops))
