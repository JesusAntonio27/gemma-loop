"""Stage 0 pool builder: offline tests with injected source loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.data.decontam import EvalIndex
from gemmaloop.stage0.pool import (
    PILOT_SUBRANGES,
    SPLIT_BUCKETS,
    bucket_for,
    build_pool,
    pilot_subsplit,
    primary_split,
    split_rows,
    write_pool,
)
from gemmaloop.stage0.sources import (
    convert_row,
    parse_input_output,
    source_with_loader,
)


def _code_row(question: str = "Add two numbers.", solution: str = "print(3)") -> dict:
    return {
        "question": question,
        "solutions": [solution],
        "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]}),
        "starter_code": "",
    }


def _math_row(problem: str = "What is 1 + 1?", answer: str = "2") -> dict:
    return {
        "problem": problem,
        "generated_solution": f"Therefore \\boxed{{{answer}}}.",
        "expected_answer": answer,
        "problem_source": "competition",
    }


def _chat_row(text: str = "Tell me a joke.") -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_bucket_rule_and_splits_are_disjoint():
    for problem_id in (f"taco:{index}" for index in range(200)):
        bucket = bucket_for(problem_id)
        assert 0 <= bucket <= 99
        split = primary_split(bucket)
        assert split in SPLIT_BUCKETS
        assert bucket_for(problem_id) == bucket
        sub = pilot_subsplit(bucket)
        assert (sub is not None) == (split == "pilot_dev")
    # The registered sub-ranges tile 50-59.
    covered = []
    for low, high in PILOT_SUBRANGES.values():
        covered.extend(range(low, high + 1))
    assert sorted(covered) == list(range(50, 60))


def test_parse_input_output_stdin_and_functional():
    stdin = parse_input_output(json.dumps({"inputs": ["1 2"], "outputs": ["3"]}))
    assert stdin == [{"input": "1 2", "output": "3", "type": "stdin"}]

    functional = parse_input_output(
        json.dumps({"inputs": [[1, 2]], "outputs": [3], "fn_name": "add"})
    )
    assert functional[0]["type"] == "functional"
    assert functional[0]["fn_name"] == "add"


def test_convert_rows_for_each_family():
    code = convert_row("taco", _code_row())
    assert code is not None
    assert code["kind"] == "code"
    assert code["tests"][0]["type"] == "stdin"
    assert "stdin" in code["prompt"]

    math = convert_row("omr", _math_row())
    assert math is not None
    assert math["kind"] == "math"
    assert math["answer"] == "2"
    assert "\\boxed{}" in math["prompt"]

    chat = convert_row("chat", _chat_row())
    assert chat is not None
    assert chat["prompt"] == "Tell me a joke."

    # Unusable rows fail closed.
    assert convert_row("taco", {"question": "x"}) is None
    assert convert_row("omr", {"problem": "x"}) is None


def test_build_pool_quotas_dedup_and_contamination_counts():
    contaminated = "this exact evaluation passage must never ever enter the stage zero problem pool because it is contaminated"
    eval_index = EvalIndex.build([contaminated])

    taco_rows = [
        _code_row("Add two numbers."),
        _code_row("Add two numbers."),  # duplicate
        _code_row(contaminated),  # contaminated
        _code_row("Multiply two numbers."),
    ]
    apps_rows = [_code_row("Add two numbers.")]  # duplicate across sources
    math_rows = [_math_row("What is 2 + 2?", "4"), _math_row("What is 3 + 3?", "6")]
    chat_rows = [_chat_row("Hello there.")]

    sources = [
        source_with_loader("taco", lambda: list(taco_rows), quota=2),
        source_with_loader("apps", lambda: list(apps_rows), quota=1),
        source_with_loader("omr", lambda: list(math_rows), quota=2),
        source_with_loader("chat", lambda: list(chat_rows), quota=1),
    ]
    result = build_pool(
        sources,
        eval_index=eval_index,
        quotas={"taco": 2, "apps": 1, "omr": 2, "chat": 1},
        allow_short_sources=True,
    )

    ids = [problem["problem_id"] for problem in result.problems]
    assert len(ids) == len(set(ids)) == 5
    assert result.manifest["counts"]["by_kind"] == {"chat": 1, "code": 2, "math": 2}
    assert result.manifest["dropped"]["taco"]["duplicate_prompt"] == 1
    assert any(key.startswith("contaminated_prompt_") for key in result.manifest["dropped"]["taco"])

    splits = split_rows(result.problems)
    assert sum(len(rows) for name, rows in splits.items() if name in SPLIT_BUCKETS) == 5
    for name, rows in splits.items():
        if name in SPLIT_BUCKETS:
            for row in rows:
                assert row["split"] == name


def test_write_pool_freezes_artifact_hashes(tmp_path: Path):
    sources = [
        source_with_loader("taco", lambda: [_code_row("A."), _code_row("B.")], quota=2),
        source_with_loader("omr", lambda: [_math_row("C?")], quota=1),
        source_with_loader("chat", lambda: [_chat_row("D.")], quota=1),
    ]
    result = build_pool(
        sources,
        quotas={"taco": 2, "omr": 1, "chat": 1},
    )
    manifest = write_pool(result, tmp_path)
    assert (tmp_path / "problems.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    assert manifest["manifest_sha256"]
    assert "problems.jsonl" in manifest["artifacts"]
    for entry in manifest["artifacts"].values():
        assert len(entry["sha256"]) == 64

    # Deterministic: a second build writes identical bytes and hashes.
    second = tmp_path / "second"
    result2 = build_pool(
        [
            source_with_loader("taco", lambda: [_code_row("A."), _code_row("B.")], quota=2),
            source_with_loader("omr", lambda: [_math_row("C?")], quota=1),
            source_with_loader("chat", lambda: [_chat_row("D.")], quota=1),
        ],
        quotas={"taco": 2, "omr": 1, "chat": 1},
    )
    manifest2 = write_pool(result2, second)
    assert manifest["manifest_sha256"] == manifest2["manifest_sha256"]
    assert (tmp_path / "problems.jsonl").read_bytes() == (second / "problems.jsonl").read_bytes()


def test_short_mandatory_source_is_fatal_by_default():
    sources = [source_with_loader("taco", lambda: [], quota=2)]
    try:
        build_pool(sources, quotas={"taco": 2})
    except RuntimeError as exc:
        assert "mandatory Stage 0 source" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a fatal short-source error")


def test_creative_briefs_missing_and_corrupt_are_recorded(tmp_path):
    from gemmaloop.stage0.evals import gather_eval_texts

    missing = tmp_path / "briefs_eval.jsonl"
    texts, status = gather_eval_texts(extra_eval_files=[missing], specs=[])
    assert texts == []
    assert status[f"file:{missing.name}"]["status"] == "missing"

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{not json\n", encoding="utf-8")
    texts, status = gather_eval_texts(extra_eval_files=[corrupt], specs=[])
    assert texts == []
    assert status[f"file:{corrupt.name}"]["status"] == "failed"


def test_briefs_gate_requires_the_briefs_target(tmp_path):
    import importlib.util
    from pathlib import Path

    from gemmaloop.stage0.evals import DEFAULT_BRIEFS

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_stage0_pool.py"
    spec = importlib.util.spec_from_file_location("build_pool_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    unrelated = {
        "file:other.jsonl": {"status": "ok", "path": str(tmp_path / "other.jsonl")}
    }
    with pytest.raises(RuntimeError, match="creative-brief"):
        module._briefs_gate(unrelated, False)
    module._briefs_gate(unrelated, True)

    briefs = {
        f"file:{DEFAULT_BRIEFS.name}": {
            "status": "ok",
            "path": str(DEFAULT_BRIEFS),
        }
    }
    module._briefs_gate(briefs, False)

    corrupt = {
        f"file:{DEFAULT_BRIEFS.name}": {
            "status": "failed",
            "path": str(DEFAULT_BRIEFS),
        }
    }
    with pytest.raises(RuntimeError, match="failed"):
        module._briefs_gate(corrupt, False)


def test_part1_pool_manifest_records_the_briefs_exclusion(tmp_path: Path):
    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

    sources = [
        source_with_loader("taco", lambda: [_code_row("A.")], quota=1),
        source_with_loader("omr", lambda: [_math_row("C?")], quota=1),
        source_with_loader("chat", lambda: [_chat_row("D.")], quota=1),
    ]
    result = build_pool(
        sources,
        quotas={"taco": 1, "omr": 1, "chat": 1},
        creative_briefs=PART1_CREATIVE_BRIEFS,
    )
    assert result.manifest["creative_briefs"] == PART1_CREATIVE_BRIEFS
    manifest = write_pool(result, tmp_path / "part1")
    assert manifest["creative_briefs"] == PART1_CREATIVE_BRIEFS
    assert manifest["limits"]["allow_missing_briefs"] is False


def test_build_stage0_pool_part1_mode_is_not_the_smoke_flag():
    import importlib.util
    from types import SimpleNamespace

    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_stage0_pool.py"
    spec = importlib.util.spec_from_file_location("build_pool_part1_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    include, gate_allow, note = module._briefs_config(
        SimpleNamespace(part1=True, allow_missing_briefs=False)
    )
    assert include is False
    assert gate_allow is True
    assert note == PART1_CREATIVE_BRIEFS

    include, gate_allow, note = module._briefs_config(
        SimpleNamespace(part1=False, allow_missing_briefs=False)
    )
    assert (include, gate_allow, note) == (True, False, "included")

    include, gate_allow, note = module._briefs_config(
        SimpleNamespace(part1=False, allow_missing_briefs=True)
    )
    assert (include, gate_allow, note) == (True, True, "included")
