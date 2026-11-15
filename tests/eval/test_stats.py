import pytest

from gemmaloop.eval.stats import wilson, paired_diff, paired_bootstrap, holm, mde


def test_wilson_matches_reference():                 # 58.6 % de 198 → 51.6–65.2 (ref. 04)
    lo, hi = wilson(116, 198); assert round(lo, 3) == 0.516 and round(hi, 3) == 0.652


def test_paired_diff_identical_is_zero():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    assert paired_diff(a, a)[:2] == (0.0, 0.0)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [True, False, False]


def test_mde_reference_table():                       # tabla de ref. 04
    assert abs(mde(30, 1) - 0.330) < 0.005
    assert abs(mde(198, 1) - 0.129) < 0.005
    assert abs(mde(1055, 1) - 0.056) < 0.005
    assert abs(mde(30, 4) - 0.209) < 0.005


def test_wilson_reference_zero_and_full():
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and 0.8 < lo < 1.0


def test_wilson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson(0, 0)
    with pytest.raises(ValueError):
        wilson(5, 4)


def test_paired_diff_known_values():
    a = {"q1": 1.0, "q2": 0.0}
    b = {"q1": 0.0, "q2": 0.0}
    mean, se, lo, hi = paired_diff(a, b)
    assert mean == pytest.approx(0.5)
    assert se == pytest.approx(0.5)
    assert lo == pytest.approx(0.5 - 1.959964 * 0.5)
    assert hi == pytest.approx(0.5 + 1.959964 * 0.5)


def test_paired_diff_uses_common_items_only():
    a = {"q1": 1.0, "q2": 0.0, "only_a": 1.0}
    b = {"q1": 0.0, "q2": 0.0, "only_b": 1.0}
    assert paired_diff(a, b)[0] == pytest.approx(0.5)


def test_paired_diff_requires_common_items():
    with pytest.raises(ValueError):
        paired_diff({"q1": 1.0}, {"q2": 1.0})


def test_paired_diff_single_item_has_zero_se():
    assert paired_diff({"q1": 1.0}, {"q1": 0.0}) == (1.0, 0.0, 1.0, 1.0)


def test_paired_bootstrap_identical_is_zero():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    assert paired_bootstrap(a, a) == (0.0, 0.0)


def test_paired_bootstrap_is_seeded_and_paired():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.0}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    first = paired_bootstrap(a, b, iters=2000, seed=0)
    assert first == paired_bootstrap(a, b, iters=2000, seed=0)
    lo, hi = first
    assert lo <= 0.5 <= hi
    rev_lo, rev_hi = paired_bootstrap(b, a, iters=2000, seed=0)
    assert rev_lo == pytest.approx(-hi)
    assert rev_hi == pytest.approx(-lo)


def test_paired_bootstrap_uses_common_items_only():
    a = {"q1": 1.0, "q2": 0.0, "only_a": 1.0}
    b = {"q1": 0.0, "q2": 0.0, "only_b": 1.0}
    lo, hi = paired_bootstrap(a, b, iters=500, seed=1)
    assert lo <= 0.5 <= hi


def test_holm_empty_and_all_rejected():
    assert holm([]) == []
    assert holm([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]


def test_holm_stops_at_first_non_rejection():
    assert holm([0.001, 0.03, 0.04], alpha=0.05) == [True, False, False]


def test_mde_validates_inputs():
    with pytest.raises(ValueError):
        mde(0, 1)
    with pytest.raises(ValueError):
        mde(30, 0)
