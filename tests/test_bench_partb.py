"""CPU tests for the Part B registration and report logic (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.bench_partb as partb


def test_partb_speedup_and_pass_threshold():
    assert partb.speedup(130.0, 100.0) == 1.3
    rows = [
        {"max_new_tokens": 8192, "engine": "padded", "speedup": 0.0},
        {"max_new_tokens": 8192, "engine": "refill", "speedup": 1.4},
        {"max_new_tokens": 16384, "engine": "padded", "speedup": 0.0},
        {"max_new_tokens": 16384, "engine": "refill", "speedup": 1.3},
    ]
    assert partb.summarize(rows, 1.3)["passed"] is True
    rows[-1]["speedup"] = 1.29
    assert partb.summarize(rows, 1.3)["passed"] is False
    assert partb.summarize([], 1.3)["passed"] is False


def test_partb_registered_settings_and_refusal():
    config = partb.load_part_b()
    assert config["requests"] >= 3 * config["max_batch"]
    assert config["max_tokens"] == [32768]
    assert config["min_speedup"] >= 1.3
    assert config["candidates"] == 360
    assert config["eval_batch"] == 16
    assert len(config["chosen_prompts_sha256"]) == 64
    args = partb.build_parser().parse_args([])
    settings = partb.resolve_settings(args, config)
    assert settings["requests"] == config["requests"]

    narrowed = partb.build_parser().parse_args(["--limit", "10"])
    with pytest.raises(SystemExit, match="--smoke"):
        partb.resolve_settings(narrowed, config)


def test_partb_selects_the_builder_candidates_only():
    rows = [
        {
            "prompt": f"Problem {index}",
            "reward_model": {"ground_truth": str(index)},
            "extra_info": {"index": index},
        }
        for index in range(30)
    ]
    conversations, stats = partb.select_prompts_from_rows(rows, 24)
    assert len(conversations) == 24
    assert stats["selected"] == 24
    assert all(
        conversation[0]["content"].startswith("Problem ")
        for conversation in conversations
    )


def test_prompt_sha256_is_deterministic():
    first = partb.prompt_sha256(["a", "b"])
    second = partb.prompt_sha256(["a", "b"])
    assert first == second
    assert first != partb.prompt_sha256(["b", "a"])


def test_partb_markdown():
    report = {
        "passed": True,
        "min_speedup": 1.3,
        "rows": [
            {
                "max_new_tokens": 8192,
                "engine": "refill",
                "requests": 128,
                "tokens": 1000,
                "elapsed_s": 10.0,
                "tok_s": 100.0,
                "speedup": 1.4,
            }
        ],
    }
    markdown = partb.render_markdown(report)
    assert "**Overall: PASS**" in markdown
    assert "1.40x" in markdown
