"""Tests for conservative two-order aggregation (Task 18)."""

from __future__ import annotations

import pytest

from gemmaloop.judge.aggregate import (
    aggregate,
    flip_rate,
    outcome_counts,
    win_rate,
)


def test_aggregate_truth_table():
    assert aggregate("X", "X") == ("X", "win")
    assert aggregate("X", "Y") == ("tie", "split")
    assert aggregate("X", "tie") == ("tie", "not_both")
    assert aggregate("X", "tie", strict=False) == ("X", "win_plus_tie")


def test_aggregate_is_symmetric_and_handles_double_ties():
    assert aggregate("Y", "Y") == ("Y", "win")
    assert aggregate("Y", "X") == ("tie", "split")
    assert aggregate("tie", "tie") == ("tie", "not_both")
    assert aggregate("tie", "tie", strict=False) == ("tie", "both_tie")
    assert aggregate("tie", "Y", strict=False) == ("Y", "win_plus_tie")
    assert aggregate("Y", "tie", strict=False) == ("Y", "win_plus_tie")


def test_aggregate_rejects_unknown_labels():
    with pytest.raises(ValueError):
        aggregate("A", "B")
    with pytest.raises(ValueError):
        aggregate("X", "Z")


def test_win_rate_counts_ties_as_half_and_reports_interval():
    point, lo, hi = win_rate(["X", "X", "Y", "tie"])

    assert point == pytest.approx(0.625)
    assert lo <= point <= hi
    assert win_rate(["X", "X", "Y", "tie"]) == (point, lo, hi)


def test_win_rate_excludes_invalid_and_render_error():
    outcomes = ["X", "invalid", "X", "render_error", "tie", "invalid"]

    point, lo, hi = win_rate(outcomes)
    counts = outcome_counts(outcomes)

    assert point == pytest.approx((2 + 0.5) / 3)
    assert lo <= point <= hi
    assert counts == {"X": 2, "Y": 0, "tie": 1, "invalid": 2, "render_error": 1}


def test_win_rate_requires_at_least_one_valid_outcome():
    with pytest.raises(ValueError):
        win_rate([])
    with pytest.raises(ValueError):
        win_rate(["invalid", "render_error"])


def test_win_rate_rejects_unknown_outcome_labels():
    with pytest.raises(ValueError):
        win_rate(["X", "nonsense"])


def test_flip_rate_counts_position_reversals_only():
    pairs = [("X", "Y"), ("X", "X"), ("tie", "X"), ("Y", "Y"), ("tie", "tie")]

    assert flip_rate(pairs) == pytest.approx(1 / 5)


def test_flip_rate_requires_pairs():
    with pytest.raises(ValueError):
        flip_rate([])


def test_flip_rate_rejects_malformed_pairs():
    with pytest.raises(ValueError):
        flip_rate([("X", "X", "Y")])  # type: ignore[list-item]
