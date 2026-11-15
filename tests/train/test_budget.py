"""FLOPs-matched token budget for the O-H-ext control (Task 13)."""

from __future__ import annotations

import pytest

from gemmaloop.model.layer_plan import LayerPlan
from gemmaloop.train.budget import (
    expected_layer_counts,
    flops_matched_tokens,
    per_layer_flops,
)


def test_flops_matched_uniform_e2b_structure():
    plan = LayerPlan(
        35, 5, (4, 9, 14, 19, 24, 29, 34), 15, range(0, 15), range(15, 30), range(30, 35)
    )
    assert flops_matched_tokens(1_000_000, [1] * 35, plan, (0.25, 0.40, 0.20, 0.15)) == 1_535_714


def test_flops_matched_nonuniform():
    plan = LayerPlan(3, 1, (0, 1, 2), 1, range(0, 1), range(1, 2), range(2, 3))
    assert flops_matched_tokens(100, [1, 2, 1], plan, (0.0, 1.0, 0.0, 0.0)) == 150


def test_expected_layer_counts():
    plan = LayerPlan(3, 1, (0, 1, 2), 1, range(0, 1), range(1, 2), range(2, 3))
    assert expected_layer_counts(plan, (0.0, 1.0, 0.0, 0.0)) == [1.0, 2.0, 1.0]
    uniform = expected_layer_counts(plan, (0.25, 0.40, 0.20, 0.15))
    assert uniform == [1.0, 2.25, 1.0]


def test_per_layer_flops_is_the_model_module_function():
    from gemmaloop.model import flops

    assert per_layer_flops is flops.per_layer_flops


def test_flops_matched_rejects_short_flop_list():
    plan = LayerPlan(3, 1, (0, 1, 2), 1, range(0, 1), range(1, 2), range(2, 3))
    with pytest.raises(ValueError):
        flops_matched_tokens(100, [1, 1], plan, (0.25, 0.40, 0.20, 0.15))
