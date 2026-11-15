"""Deterministic loop-count sampling (Task 13)."""

from __future__ import annotations

from collections import Counter

import pytest

from gemmaloop.train.loop_sampler import sample_n_loops


def test_sampler_deterministic_and_distribution():
    assert [sample_n_loops(0, s) for s in range(50)] == [
        sample_n_loops(0, s) for s in range(50)
    ]
    counts = Counter(sample_n_loops(0, s) for s in range(20000))
    for n, probability in zip((1, 2, 3, 4), (0.25, 0.40, 0.20, 0.15)):
        assert abs(counts[n] / 20000 - probability) < 0.01


def test_sampler_depends_on_seed_and_step():
    sequence = [sample_n_loops(0, s) for s in range(20)]
    assert sequence != [sample_n_loops(1, s) for s in range(20)]
    assert sample_n_loops(7, 11) == sample_n_loops(7, 11)


def test_sampler_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        sample_n_loops(0, 0, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        sample_n_loops(0, 0, (-0.1, 1.1))
