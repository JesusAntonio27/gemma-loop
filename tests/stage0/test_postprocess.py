"""Post-processing: filters, pass rates, heal rows and verified corpora."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gemmaloop.stage0.postprocess import (
    keep_reason,
    postprocess,
    write_postprocess,
)

REASONING = "think " * 16
CODE_OK = "```python\nGOOD\n```"
CODE_BAD = "```python\nBAD\n```"


class FakeGrader:
    def grade_code(self, code, tests):
        return SimpleNamespace(
            passed="GOOD" in code,
            to_dict=lambda: {"passed": "GOOD" in code},
        )

    def grade_math(self, prediction, answer):
        correct = answer in prediction
        return SimpleNamespace(
            correct=correct,
            to_dict=lambda: {"correct": correct},
        )


def _problem(pid: str, kind: str, split: str) -> dict:
    return {
        "problem_id": pid,
        "kind": kind,
        "split": split,
        "source": "taco" if kind == "code" else "omr",
        "prompt": "prompt " + pid,
        "tests": [],
        "answer": "42" if kind == "math" else None,
    }


def _sample(
    pid: str,
    index: int,
    *,
    good: bool = True,
    rendered: int = 1000,
    finish: str = "stop",
    closed: bool = True,
    content: str | None = None,
    reasoning: str = REASONING,
) -> dict:
    if content is None:
        content = CODE_OK if good else CODE_BAD
    return {
        "problem_id": pid,
        "sample_index": index,
        "completion": reasoning + content,
        "reasoning": reasoning,
        "content": content,
        "finish_reason": finish,
        "thought_closed": closed,
        "rendered_tokens": rendered,
        "tokens": rendered - 10,
        "prompt_tokens": 10,
    }


def test_keep_reason_filter_order():
    problem = _problem("p", "code", "train")
    assert keep_reason(_sample("p", 0), problem, token_counter=lambda text: len(text.split())) is None
    assert keep_reason(_sample("p", 0, finish="length"), problem, token_counter=lambda text: len(text.split())) == "finish_reason"
    assert keep_reason(_sample("p", 0, closed=False), problem, token_counter=lambda text: len(text.split())) == "thought_open"
    assert keep_reason(_sample("p", 0, content=""), problem, token_counter=lambda text: len(text.split())) == "empty_answer"
    assert keep_reason(_sample("p", 0, rendered=9000), problem, token_counter=lambda text: len(text.split())) == "too_long"
    low_share = _sample("p", 0, reasoning="short", content="a very long answer " * 20)
    assert keep_reason(low_share, problem, token_counter=lambda text: len(text.split())) == "reasoning_share"


def test_postprocess_builds_registered_corpora_and_prompts():
    problems = [
        _problem("p_train", "code", "train"),
        _problem("p_all_correct", "code", "train"),
        _problem("p_long_median", "code", "train"),
        _problem("p_zero", "code", "train"),
        _problem("p_pilot", "code", "pilot_dev"),
        _problem("p_nll", "math", "nll_kl"),
        _problem("p_long", "code", "long_traces"),
        _problem("p_filter", "code", "train"),
    ]
    samples = []
    # 2 of 4 correct -> Stage 2 prompt candidate.
    for index, good in enumerate([True, True, False, False]):
        samples.append(_sample("p_train", index, good=good))
    # 4 of 4 correct -> excluded by the 1-3 rule.
    for index in range(4):
        samples.append(_sample("p_all_correct", index, good=True))
    # 1 of 4 correct but median rendered length is 7000 -> excluded by length.
    for index, rendered in enumerate([7000, 7000, 7000, 1000]):
        samples.append(_sample("p_long_median", index, good=index == 0, rendered=rendered))
    # 0 correct -> excluded.
    for index in range(4):
        samples.append(_sample("p_zero", index, good=False))
    # Pilot bucket: only the two correct traces are exported.
    for index, good in enumerate([True, False, True, False]):
        samples.append(_sample("p_pilot", index, good=good))
    # NLL/KL bucket math.
    samples.append(_sample("p_nll", 0, content="\\boxed{42}", reasoning=REASONING))
    samples.append(_sample("p_nll", 1, content="\\boxed{41}", reasoning=REASONING))
    # Long traces bucket.
    samples.append(_sample("p_long", 0, rendered=6000))
    # Registered filter drops (a complete 4-sample prompt).
    samples.append(_sample("p_filter", 0, finish="length"))
    samples.append(_sample("p_filter", 1, closed=False))
    samples.append(_sample("p_filter", 2, content=""))
    samples.append(_sample("p_filter", 3, rendered=9000))

    result = postprocess(
        problems,
        samples,
        grader=FakeGrader(),
        token_counter=lambda text: len(text.split()),
    )

    assert result.manifest["counts"]["samples_input"] == len(samples)
    assert result.manifest["counts"]["samples_dropped"] == 4
    assert set(result.manifest["counts"]["dropped_by_reason"]) == {
        "finish_reason",
        "thought_open",
        "empty_answer",
        "too_long",
    }

    prompt_ids = [row["problem_id"] for row in result.stage2_prompts]
    assert prompt_ids == ["p_train"]
    assert result.stage2_prompts[0]["n_correct"] == 2

    pilot_ids = {row["problem_id"] for row in result.pilot_verified}
    assert pilot_ids == {"p_pilot"}
    assert len(result.pilot_verified) == 2

    assert {row["problem_id"] for row in result.nll_kl_verified} == {"p_nll"}
    assert len(result.nll_kl_verified) == 1

    assert {row["problem_id"] for row in result.long_traces} == {"p_long"}
    assert all(row["split"] == "train" for row in result.stage1_heal)
    assert all(row["rendered_tokens"] <= 8192 for row in result.stage1_heal)
    assert result.manifest["pass_rate_histogram"]["2"] == 2  # p_train and p_pilot


def test_write_postprocess_freezes_hashes_and_updates_pool_manifest(tmp_path: Path):
    problems = [_problem("p_train", "code", "train")]
    samples = [_sample("p_train", index, good=index < 2) for index in range(4)]
    result = postprocess(
        problems, samples, grader=FakeGrader(), token_counter=lambda text: len(text.split())
    )
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "manifest.json").write_text(
        json.dumps({"stage": "stage0_pool", "artifacts": {}}), encoding="utf-8"
    )
    manifest = write_postprocess(
        result, tmp_path / "out", pool_manifest_path=pool_dir / "manifest.json"
    )
    for name in (
        "sample_grades.jsonl",
        "stage1_heal.jsonl",
        "stage2_prompts.jsonl",
        "pilot_verified.jsonl",
        "nll_kl_verified.jsonl",
        "long_traces.jsonl",
    ):
        assert (tmp_path / "out" / name).exists()
        assert len(manifest["artifacts"][name]["sha256"]) == 64
    updated = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated["postprocess"]["counts"]["stage1_heal_rows"] == 4
    assert "postprocess/sample_grades.jsonl" in updated["artifacts"]
    import hashlib

    payload = dict(updated)
    stored = payload.pop("manifest_sha256")
    recomputed = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert stored == recomputed


def test_stage2_requires_exactly_four_generated_samples():
    problems = [_problem("p_three", "code", "train")]
    samples = [_sample("p_three", index, good=index < 2) for index in range(3)]
    result = postprocess(
        problems, samples, grader=FakeGrader(), token_counter=lambda text: len(text.split())
    )
    assert result.stage2_prompts == []
    assert result.manifest["filter"]["stage2_rule"].startswith("exactly 4")


def test_long_traces_apply_the_registered_length_floor():
    problems = [_problem("p_long", "code", "long_traces")]
    samples = [
        _sample("p_long", 0, rendered=1000),
        _sample("p_long", 1, rendered=5000),
    ]
    result = postprocess(
        problems, samples, grader=FakeGrader(), token_counter=lambda text: len(text.split())
    )
    assert len(result.long_traces) == 1
    assert result.long_traces[0]["rendered_tokens"] >= 4096
