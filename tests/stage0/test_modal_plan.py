"""Modal sampling planner: pure arithmetic, no Modal client needed."""

from __future__ import annotations

import pytest

from gemmaloop.stage0.modal_plan import (
    GPU_HOURLY_USD,
    GPU_TOKENS_PER_SEC,
    check_budget,
    estimate_all_gpus,
    estimate_gpu,
    per_second_price,
    sampling_plan,
)


def test_gpu_tables_cover_the_same_names():
    assert set(GPU_HOURLY_USD) == set(GPU_TOKENS_PER_SEC)
    assert per_second_price("H100") == pytest.approx(3.95 / 3600)


def test_estimate_gpu_matches_tokens_hours_dollars():
    entry = estimate_gpu(
        36_000_000,
        "H100",
        tokens_per_sec=10_000,
        overhead=0.0,
        containers=1,
        startup_seconds=0,
        safety_margin=1.0,
    )
    assert entry["gpu_hours"] == pytest.approx(1.0)
    assert entry["gpu_usd"] == pytest.approx(3.95)
    assert entry["cpu_usd"] == pytest.approx(8 * 0.047)
    assert entry["memory_usd"] == pytest.approx(48 * 0.008)
    assert entry["raw_usd"] == pytest.approx(3.95 + 8 * 0.047 + 48 * 0.008)
    assert entry["usd"] == pytest.approx(entry["raw_usd"])


def test_sampling_plan_has_every_gpu_and_registered_token_scale():
    plan = sampling_plan(
        prompts=5000,
        samples_per_prompt=4,
        avg_generated_tokens=3200,
        avg_prompt_tokens=500,
        gpu="A100-80GB",
        overhead=0.0,
    )
    assert plan["samples"] == 20_000
    assert plan["generated_tokens"] == 64_000_000
    chosen = plan["chosen"]
    assert chosen["usd"] == pytest.approx(chosen["raw_usd"] * 1.3)
    assert chosen["raw_usd"] == pytest.approx(
        chosen["gpu_usd"] + chosen["cpu_usd"] + chosen["memory_usd"]
    )
    assert set(plan["all_gpus"]) == set(GPU_HOURLY_USD)


def test_budget_refuses_plan_above_max_usd_and_hard_cap():
    estimate = estimate_gpu(100_000_000, "B300", tokens_per_sec=1_000, overhead=0.0)
    assert estimate["usd"] > 12
    with pytest.raises(ValueError, match="max-usd"):
        check_budget(estimate, max_usd=12)
    with pytest.raises(ValueError, match="max_usd"):
        check_budget(estimate, max_usd=40)
    with pytest.raises(ValueError):
        check_budget(estimate, max_usd=-1)


def test_estimate_all_gpus_is_a_table():
    table = estimate_all_gpus(20_000_000, overhead=0.10)
    assert len(table) == len(GPU_HOURLY_USD)
    for entry in table.values():
        assert entry["usd"] > 0
        assert entry["gpu_hours"] > 0


def test_modal_is_not_a_project_dependency():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert "modal" not in (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "modal" not in (root / "uv.lock").read_text(encoding="utf-8").lower()
