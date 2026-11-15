"""Budget arithmetic and spill detection for the preregistration (Task 6)."""

from __future__ import annotations

import pytest

from gemmaloop.budget import (
    DEFAULT_SAMPLES_PER_BENCH,
    bench_rates_from_calibration,
    eval_hours,
    fit_eval_sizes,
    gen_rates_from_calibration,
    max_trainable_length,
    rate_for_completion_length,
    spill_detected,
    train_hours,
)


def test_eval_hours():
    assert abs(eval_hours(300, 5, 3000, 300.0) - 300 * 5 * 3000 / 300 / 3600) < 1e-9


def test_fit_fills_by_priority_up_to_caps():
    sizes = fit_eval_sizes(
        300.0,
        {"lcb": 3000, "gpqa": 1500},
        10.0,
        {"lcb": 182, "gpqa": 198},
        {"lcb": 1055, "gpqa": 198},
        ["lcb", "gpqa"],
        {"lcb": 1, "gpqa": 1},
    )
    assert sizes == {"lcb": 1055, "gpqa": 198}  # 3.46 M tokens <= 10.8 M


def test_fit_raises_when_minimums_exceed_budget():
    with pytest.raises(ValueError):
        fit_eval_sizes(
            300.0,
            {"lcb": 3000, "gpqa": 1500},
            0.5,
            {"lcb": 182, "gpqa": 198},
            {"lcb": 1055, "gpqa": 198},
            ["lcb", "gpqa"],
            {"lcb": 1, "gpqa": 1},
        )


def test_max_trainable_length():
    assert max_trainable_length(lambda n: n <= 6144) == 6144


def test_train_hours():
    assert abs(train_hours(40_000_000, 1500.0) - 40e6 / 1500 / 3600) < 1e-9


def test_fit_partially_fills_priority_in_order():
    # 0.47 h * 3600 s * 300 tok/s = 507,600 tokens.
    # Minimums: 100*3000 + 50*1500 + 10*8000 = 455,000.
    # lcb grows by 17 (51,000), gpqa by 1 (1,500), aime stays at its cap.
    sizes = fit_eval_sizes(
        300.0,
        {"lcb": 3000, "gpqa": 1500, "aime": 8000},
        0.47,
        {"lcb": 100, "gpqa": 50, "aime": 10},
        {"lcb": 120, "gpqa": 200, "aime": 10},
        ["lcb", "gpqa", "aime"],
        {"lcb": 1, "gpqa": 1, "aime": 1},
    )
    assert sizes == {"lcb": 117, "gpqa": 51, "aime": 10}


def test_fit_multiplies_by_registered_samples():
    # Per-completion costs, four samples each.  Minimums cost
    # (182*3000 + 198*1500)*4 = 3.372 M tokens, leaving 7.428 M.
    # lcb can grow by floor(7.428 M / (3000*4)) = 619 items, not 873.
    sizes = fit_eval_sizes(
        300.0,
        {"lcb": 3000, "gpqa": 1500},
        10.0,
        {"lcb": 182, "gpqa": 198},
        {"lcb": 1055, "gpqa": 198},
        ["lcb", "gpqa"],
        samples_per_bench={"lcb": 4, "gpqa": 4},
    )
    assert sizes == {"lcb": 801, "gpqa": 198}


def test_fit_requires_samples_for_every_bench():
    with pytest.raises(ValueError):
        fit_eval_sizes(
            300.0,
            {"lcb": 3000, "gpqa": 1500},
            10.0,
            {"lcb": 182, "gpqa": 198},
            {"lcb": 1055, "gpqa": 198},
            ["lcb", "gpqa"],
            {"lcb": 4},
        )


def test_samples_per_bench_is_a_required_argument():
    with pytest.raises(TypeError):
        fit_eval_sizes(
            300.0,
            {"lcb": 3000},
            10.0,
            {"lcb": 182},
            {"lcb": 1055},
            ["lcb"],
        )


def test_fit_rejects_priority_without_cap():
    with pytest.raises(ValueError):
        fit_eval_sizes(
            300.0,
            {"lcb": 3000},
            10.0,
            {"lcb": 182},
            {},
            ["lcb"],
            {"lcb": 1},
        )


def test_registered_samples_match_the_prereg():
    assert DEFAULT_SAMPLES_PER_BENCH == {
        "lcb": 4,
        "aime": 8,
        "gpqa": 4,
        "mmlupro": 1,
        "mmmupro": 1,
        "mathvision": 1,
        "spatialeval": 1,
        "cvbench": 1,
    }


def test_max_trainable_length_stops_at_first_failure():
    calls: list[int] = []

    def try_fn(length: int) -> bool:
        calls.append(length)
        return length <= 6144

    assert max_trainable_length(try_fn) == 6144
    assert calls == [4096, 6144, 8192]


def test_max_trainable_length_raises_without_candidate():
    with pytest.raises(ValueError):
        max_trainable_length(lambda _length: False)


def test_spill_detected_by_reserved_memory():
    total = 12 * 2**30
    assert not spill_detected([100.0, 101.0, 99.0], int(total * 0.50), total)
    assert spill_detected([100.0, 101.0, 99.0], int(total * 0.96), total)


def test_spill_detected_by_step_collapse():
    total = 12 * 2**30
    # Median is 100 tok/s; the last step at 40 tok/s is a collapse.
    assert spill_detected([100.0, 100.0, 40.0], int(total * 0.50), total)
    # A uniformly fast measurement is clean.
    assert not spill_detected([100.0, 100.0, 95.0], int(total * 0.50), total)


def test_empty_measurement_is_a_spill():
    assert spill_detected([], 0, 12 * 2**30)


def test_fit_eval_sizes_accepts_per_benchmark_rates():
    kwargs = dict(
        tokens_per_item={"fast": 1000.0, "slow": 1000.0},
        hours_per_variant=0.02,
        minimums={"fast": 0, "slow": 0},
        caps={"fast": 2, "slow": 100},
        priority=["fast", "slow"],
        samples_per_bench={"fast": 1, "slow": 1},
    )
    uniform = fit_eval_sizes(100.0, **kwargs)
    # Same total tokens but the slow benchmark is 4x more expensive per token:
    # its items must consume more of the time budget than in the uniform case.
    mapped = fit_eval_sizes({"fast": 100.0, "slow": 25.0}, **kwargs)
    assert mapped["slow"] < uniform["slow"]
    assert mapped["fast"] == uniform["fast"] == 2


# ---------------------------------------------------------------------------
# Valid-bucket selection (calibration -> per-benchmark rates)
# ---------------------------------------------------------------------------


def _gen_calibration(records: dict) -> dict:
    return {"gen_tok_per_s": {"e2b": records}}


def test_gen_rates_use_only_valid_buckets_and_the_requested_loop():
    calibration = _gen_calibration(
        {
            "n2_len2048": {
                "tok_s": 300.0,
                "invalid": False,
                "cache_new_tokens": 2048,
            },
            "n2_len4096": {
                "tok_s": None,
                "invalid": True,
                "cache_new_tokens": 4096,
            },
            "n2_len8192": {
                "tok_s": 150.0,
                "invalid": False,
                "cache_new_tokens": 8192,
            },
            "n4_len2048": {  # wrong loop count
                "tok_s": 999.0,
                "invalid": False,
                "cache_new_tokens": 2048,
            },
            "not-a-bucket": {"tok_s": 1.0},
        }
    )
    assert gen_rates_from_calibration(calibration, "e2b", 2) == {
        2048: 300.0,
        8192: 150.0,
    }


def test_gen_rates_use_a_clean_half_batch_retry():
    calibration = _gen_calibration(
        {
            "n1_len4096": {
                "tok_s": None,
                "invalid": True,
                "spill": True,
                "cache_new_tokens": 4096,
                "attempts": [
                    {"batch": 32, "tok_s": None, "spill": True, "oom": False},
                    {"batch": 16, "tok_s": 123.5, "spill": False, "oom": False},
                ],
            },
            # The retry also spilled: no usable rate for this bucket.
            "n1_len8192": {
                "tok_s": None,
                "invalid": True,
                "spill": True,
                "cache_new_tokens": 8192,
                "attempts": [
                    {"batch": 8, "tok_s": None, "spill": True, "oom": False},
                    {"batch": 4, "tok_s": None, "spill": True, "oom": True},
                ],
            },
            # The valid top-level record wins over its attempts.
            "n1_len2048": {
                "tok_s": 250.0,
                "invalid": False,
                "cache_new_tokens": 2048,
                "attempts": [{"batch": 64, "tok_s": 1.0, "spill": False}],
            },
        }
    )
    assert gen_rates_from_calibration(calibration, "e2b", 1) == {
        2048: 250.0,
        4096: 123.5,
    }


def test_gen_rates_accept_legacy_records_and_missing_inputs():
    legacy = _gen_calibration({"n1_len2048": {"tok_s": 111.0}})
    assert gen_rates_from_calibration(legacy, "e2b", 1) == {2048: 111.0}
    assert gen_rates_from_calibration({}, "e2b", 1) == {}
    assert gen_rates_from_calibration(legacy, "e4b", 1) == {}
    assert gen_rates_from_calibration(legacy, "e2b", 2) == {}


def test_rate_for_completion_length_uses_the_next_larger_valid_bucket():
    rates = {2048: 300.0, 8192: 150.0}  # 4096 was invalid and skipped
    assert rate_for_completion_length(rates, 100.0) == 300.0
    assert rate_for_completion_length(rates, 2048.0) == 300.0
    assert rate_for_completion_length(rates, 4096.0) == 150.0
    assert rate_for_completion_length(rates, 5000.0) == 150.0
    # Longer than every measured bucket: use the longest valid one.
    assert rate_for_completion_length(rates, 999_999.0) == 150.0


def test_rate_for_completion_length_requires_a_valid_bucket():
    with pytest.raises(ValueError, match="no valid generation-rate buckets"):
        rate_for_completion_length({}, 100.0)
    with pytest.raises(ValueError, match="no valid generation-rate buckets"):
        rate_for_completion_length({2048: 0.0}, 100.0)


def test_bench_rates_choose_each_benchmarks_p90_bucket():
    calibration = {
        **_gen_calibration(
            {
                "n1_len2048": {"tok_s": 300.0, "invalid": False},
                "n1_len4096": {"tok_s": None, "invalid": True},
                "n1_len8192": {"tok_s": 150.0, "invalid": False},
            }
        ),
        "tokens_per_item": {
            "aime": {"p90": 1000.0},
            "lcb": {"p90": 5000.0},
        },
    }
    assert bench_rates_from_calibration(calibration, "e2b", 1) == {
        "aime": 300.0,
        "lcb": 150.0,
    }
    with pytest.raises(ValueError, match="no p90 completion-length"):
        bench_rates_from_calibration(
            calibration, "e2b", 1, benches=["lcb", "missing"]
        )


def test_fit_raises_clearly_when_a_needed_rate_is_missing():
    kwargs = dict(
        tokens_per_item={"fast": 1000.0, "slow": 1000.0},
        hours_per_variant=0.02,
        minimums={"fast": 0, "slow": 0},
        caps={"fast": 2, "slow": 100},
        priority=["fast", "slow"],
        samples_per_bench={"fast": 1, "slow": 1},
    )
    with pytest.raises(ValueError, match="no tok_per_s entry for benchmark 'slow'"):
        fit_eval_sizes({"fast": 100.0}, **kwargs)
    with pytest.raises(ValueError, match=r"tok_per_s\['slow'\]"):
        fit_eval_sizes({"fast": 100.0, "slow": None}, **kwargs)


def test_flops_matched_budget_follows_the_configured_block():
    import json
    from pathlib import Path

    from gemmaloop.model.flops import per_layer_flops
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.train.budget import flops_matched_tokens
    from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS

    config = json.loads(
        (
            Path(__file__).resolve().parent / "fixtures" / "e2b_text_config.json"
        ).read_text(encoding="utf-8")
    )
    base = 40_000_000
    values = per_layer_flops(config)
    budgets = {}
    for block in ((15, 29), (25, 29), (16, 18)):
        plan = make_layer_plan(config, block=block)
        expected = flops_matched_tokens(
            base, values, plan, DEFAULT_LOOP_PROBS
        )
        budgets[block] = expected
        assert expected == flops_matched_tokens(
            base, values, make_layer_plan(config, block=block), DEFAULT_LOOP_PROBS
        )
    assert budgets[(16, 18)] < budgets[(25, 29)] < budgets[(15, 29)]
