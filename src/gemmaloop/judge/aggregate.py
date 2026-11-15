"""Aggregate the two order-swapped verdicts of a pair.

The judge sees the same pair twice: order 1 labels ``X`` as A, order 2 labels
``Y`` as A.  Each verdict is first mapped back to ``X``/``Y``/``tie``.  The
primary rule is the conservative MT-Bench rule: a candidate wins only when it
wins in both orders; every disagreement becomes a tie.  The secondary rule
(Arena 2026) also awards a victory when one order wins and the other ties.

Failures (``invalid`` judgments and ``render_error`` pairs) never enter the
win rate: they are counted separately by :func:`outcome_counts` and excluded
from the denominator, so a broken render can never become a win.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

OUTCOMES = ("X", "Y", "tie")
NON_SCORING = ("invalid", "render_error")
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 0

Pair = tuple[str, str]


def _check_outcome(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(
            f"unknown outcome {outcome!r}; expected one of {', '.join(OUTCOMES)}"
        )


def aggregate(order1: str, order2: str, strict: bool = True) -> tuple[str, str]:
    """Combine the two orders into ``(winner, reason)``.

    ``order1`` and ``order2`` are already mapped to ``X``/``Y``/``tie``.
    With ``strict=True`` (primary, MT-Bench) only a candidate that wins both
    orders wins; everything else is a tie.  With ``strict=False`` (secondary,
    Arena 2026) a win in one order plus a tie in the other is also a win.
    """
    _check_outcome(order1)
    _check_outcome(order2)

    if order1 == order2 and order1 in ("X", "Y"):
        return order1, "win"
    if {order1, order2} == {"X", "Y"}:
        return "tie", "split"
    if order1 == order2 == "tie":
        return "tie", ("not_both" if strict else "both_tie")
    if strict:
        return "tie", "not_both"

    winner = order1 if order1 != "tie" else order2
    return winner, "win_plus_tie"


def outcome_counts(outcomes: Iterable[str]) -> dict[str, int]:
    """Count every outcome, including the non-scoring failures.

    This is the companion of :func:`win_rate`: invalid judgments and render
    errors are reported here, never folded into the win rate.
    """
    counts = {outcome: 0 for outcome in (*OUTCOMES, *NON_SCORING)}
    for outcome in outcomes:
        if outcome not in counts:
            raise ValueError(f"unknown outcome {outcome!r}")
        counts[outcome] += 1
    return counts


def _scores(outcomes: Sequence[str]) -> np.ndarray:
    """Per-pair score: win = 1, tie = 0.5, loss = 0 (failures excluded)."""
    return np.array(
        [
            1.0 if outcome == "X" else 0.0 if outcome == "Y" else 0.5
            for outcome in outcomes
        ],
        dtype=float,
    )


def win_rate(
    outcomes: list[str],
    *,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for ``(wins + 0.5 * ties) / N``.

    ``invalid`` and ``render_error`` outcomes are dropped before the
    denominator is formed.  The 95 % interval is a percentile bootstrap over
    pairs, seeded so repeated calls are identical.  X is the candidate whose
    win rate is reported (Task 20 reports L over O-H).
    """
    if iters <= 0:
        raise ValueError("iters must be positive")

    scored = [outcome for outcome in outcomes if outcome in OUTCOMES]
    rejected = [outcome for outcome in outcomes if outcome not in OUTCOMES]
    for outcome in rejected:
        if outcome not in NON_SCORING:
            raise ValueError(f"unknown outcome {outcome!r}")
    if not scored:
        raise ValueError("win_rate needs at least one scored outcome")

    values = _scores(scored)
    n = len(values)
    point = float(values.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(iters, n))
    boot = values[indices].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def flip_rate(
    pairs: Iterable[Pair],
    *,
    include_ties: bool = False,
) -> float:
    """Fraction of pairs whose winner reversed between the two orders.

    By default a flip is ``X`` in one order and ``Y`` in the other: the judge
    changed its preference when the positions were swapped.  A tie in either
    order is not a flip (it is not a preference reversal); pass
    ``include_ties=True`` to count every disagreement instead.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("flip_rate needs at least one pair")

    flips = 0
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each pair must contain exactly two outcomes")
        first, second = pair
        _check_outcome(first)
        _check_outcome(second)
        if first == second:
            continue
        if include_ties or (first != "tie" and second != "tie"):
            flips += 1
    return flips / len(pairs)
