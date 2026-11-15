"""End-to-end CPU test of the local Stage-0-lite script (fake generator)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaloop.stage0 import local
from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS
from gemmaloop.stage0.postprocess import MAX_LENGTH

from test_local_sampler import _problem, _renderer

def _load_script():
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "sample_stage0_local.py"
    spec = importlib.util.spec_from_file_location("sample_stage0_local_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_finalize_writes_heal_and_pilot_corpora_and_a_verifiable_manifest(tmp_path: Path):
    script = _load_script()
    pool = tmp_path / "pool"
    pool.mkdir()
    problems = [
        _problem("train-math", kind="math", bucket=0),
        _problem("train-chat", kind="chat", bucket=1),
        _problem("p1-math", kind="math", bucket=59),
        _problem("tfit-math", kind="math", bucket=57),
        _problem("sel-math", kind="math", bucket=54),
    ]
    for problem in problems:
        problem["prompt"] = problem["problem_id"]
        problem["split"] = (
            "train" if problem["bucket"] <= 49 else "pilot_dev"
        )
        problem["pilot_subsplit"] = {
            59: "pilot_p1",
            57: "pilot_tfit",
            54: "pilot_selection",
        }.get(problem["bucket"])
    problems_text = "".join(json.dumps(problem) + "\n" for problem in problems)
    (pool / "problems.jsonl").write_text(problems_text, encoding="utf-8")
    manifest = {
        "stage": "stage0_pool",
        "manifest_sha256": "poolsha",
        "creative_briefs": PART1_CREATIVE_BRIEFS,
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(problems_text.encode()).hexdigest(),
                "rows": len(problems),
            }
        },
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt_tokens = {problem["problem_id"]: 100 for problem in problems}
    renderer = _renderer(prompt_tokens)
    counter = lambda text: len(text.split())  # noqa: E731

    def evaluate_filter(record, problem):
        return local.evaluate_sample(
            record, problem, renderer=renderer, token_counter=counter
        )

    def verify_fn(record, problem):
        return (
            "answer" in str(record.get("content") or ""),
            {"ok": True, "decided_by": "normaliser", "timed_out": False},
        )

    def evaluate_full(record, problem):
        return local.evaluate_sample(
            record,
            problem,
            renderer=renderer,
            token_counter=counter,
            verify_fn=verify_fn,
        )

    def fake_generate(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        request_ids=None,
        max_batch=None,
    ):
        return [
            SimpleNamespace(
                text="answer",
                reasoning="t " * 10,
                token_ids=None,
                finish_reason="stop",
                completion_tokens=11,
            )
            for _ in conversations
        ]

    targets = [
        local.Target("train", 0, 49, 2, token_target=10, verified=False),
        local.Target("pilot_p1", 59, 59, 1, verified=True, rows_target=1),
        local.Target("pilot_tfit", 57, 58, 1, verified=True, rows_target=1),
        local.Target("pilot_selection", 54, 56, 1, verified=True, rows_target=1),
    ]
    output = tmp_path / "out"
    run_stats = local.run_sampling(
        problems=problems,
        targets=targets,
        output_dir=output,
        shard_size=2,
        samples_per_prompt=1,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: prompt_tokens[problem["problem_id"]],
        evaluate_fn=evaluate_filter,
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="run",
        pool_manifest_sha256=manifest["manifest_sha256"],
        model_revision="rev",
    )
    stats, payload = script.finalize(
        args=SimpleNamespace(),
        output_dir=output,
        problems=problems,
        targets=targets,
        pool_dir=pool,
        pool_manifest=manifest,
        problems_path=pool / "problems.jsonl",
        run_id="run",
        model_revision="rev",
        run_stats=run_stats,
        evaluate_filter=evaluate_filter,
        evaluate_full=evaluate_full,
        verifiers={"math": "math-verify"},
        hidden_tests={
            "path": "hidden_tests.jsonl",
            "sha256": "hidden-sha",
            "rows": 1,
            "problems_merged": 1,
        },
    )
    assert stats.train_tokens == 22  # two kept train rows x 11 assistant tokens
    assert (output / "heal" / "train.jsonl").exists()
    heal_rows = local.read_jsonl(output / "heal" / "train.jsonl")
    assert len(heal_rows) == 2 and "messages" in heal_rows[0]
    assert len(local.read_jsonl(output / "pilot" / "pilot_p1.jsonl")) == 1
    assert len(local.read_jsonl(output / "pilot" / "pilot_tfit.jsonl")) == 1
    assert len(local.read_jsonl(output / "pilot" / "pilot_selection.jsonl")) == 1
    assert (output / "request_ids.txt").read_text(encoding="utf-8").split() == [
        "p1-math:0",
        "sel-math:0",
        "tfit-math:0",
        "train-chat:0",
        "train-math:0",
    ]
    ok, reason = local.verify_manifest(output / "manifest.json")
    assert ok, reason
    assert payload["engine"] == "local-padded"
    assert payload["pool"]["creative_briefs"] == PART1_CREATIVE_BRIEFS
    assert payload["acceptance"]["kept"] == 5
    assert payload["hidden_tests"]["sha256"] == "hidden-sha"
    assert payload["selection_order"]["seed"] == 0
    assert set(payload["targets"]["train"]["sources"]) == {"omr", "chat"}

    grades = local.read_jsonl(output / "sample_grades.jsonl")
    assert len(grades) == 5
    pilot_grade = next(row for row in grades if row["problem_id"] == "p1-math")
    assert pilot_grade["decided_by"] == "normaliser"
    assert pilot_grade["verifiers"] == ["normaliser"]
    assert pilot_grade["request_id"] == "p1-math:0"
    assert pilot_grade["timing_s"] >= 0.0
    assert payload["verification"]["decided_by"] == {"normaliser": 3}
    assert payload["verification"]["timed_out"] == 0
    assert payload["files"]["sample_grades.jsonl"]["rows"] == 5
    assert any(key.startswith("shards/") for key in payload["files"])

    shard = sorted((output / "shards").glob("shard_*.jsonl"))[0]
    shard.write_text(
        shard.read_text(encoding="utf-8") + "tampered" + chr(10),
        encoding="utf-8",
    )
    ok, reason = local.verify_manifest(output / "manifest.json")
    assert not ok and "shard_" in reason


def test_dry_run_command_uses_the_part1_pool_mode():
    script = _load_script()
    command = script.build_pool_command(Path("artifacts/stage0/pool"))
    assert "build_stage0_pool.py" in command
    assert "--part1" in command


def test_run_id_and_parsing_match_the_modal_job():
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "modal_stage0_for_local_test", root / "scripts" / "modal_stage0.py"
    )
    modal = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(modal)
    assert local.compute_run_id(
        "poolsha", shard_size=25, samples_per_prompt=4, model_revision="rev"
    ) == modal.compute_run_id(
        "poolsha", shard_size=25, samples_per_prompt=4, model_revision="rev"
    )
    fixtures = root / "tests" / "fixtures" / "gemma_outputs" / "reasoning.txt"
    raw = fixtures.read_text(encoding="utf-8")
    assert local.parse_completion(raw) == modal.parse_completion(raw)
    assert modal.sampling_params_kwargs()["n"] == local.SAMPLES_PER_PROMPT


def test_verify_code_requires_the_registered_hidden_tests(tmp_path: Path):
    script = _load_script()
    missing = tmp_path / "hidden_tests.jsonl"
    assert script.resolve_hidden_tests(missing, verify_code=False) is None
    with pytest.raises(SystemExit, match="hidden-tests"):
        script.resolve_hidden_tests(missing, verify_code=True)
    missing.write_text("{}" + chr(10), encoding="utf-8")
    assert script.resolve_hidden_tests(missing, verify_code=True) == missing


def test_ensure_pool_passes_the_part1_arguments(tmp_path: Path, monkeypatch):
    script = _load_script()
    pool = tmp_path / "pool"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        script.local,
        "load_pool",
        lambda path: ([{"problem_id": "p"}], {"manifest_sha256": "poolsha"}),
    )
    loaded = script.ensure_pool(pool, build=True)
    assert loaded is not None
    command = captured["command"]
    assert "scripts.build_stage0_pool" in command
    assert "--part1" in command
    assert "--output-dir" in command


def test_finalize_persists_timeout_and_code_grader_decisions(tmp_path: Path):
    script = _load_script()
    pool = tmp_path / "pool"
    pool.mkdir()
    problems = [
        _problem("math-pilot", kind="math", bucket=59, source="omr"),
        _problem("code-pilot", kind="code", bucket=59, source="taco"),
    ]
    for problem in problems:
        problem["prompt"] = problem["problem_id"]
        problem["split"] = "pilot_dev"
        problem["pilot_subsplit"] = "pilot_p1"
    problems_text = "".join(json.dumps(problem) + chr(10) for problem in problems)
    (pool / "problems.jsonl").write_text(problems_text, encoding="utf-8")
    manifest = {
        "stage": "stage0_pool",
        "creative_briefs": PART1_CREATIVE_BRIEFS,
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(problems_text.encode()).hexdigest(),
                "rows": len(problems),
            }
        },
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt_tokens = {problem["problem_id"]: 100 for problem in problems}
    renderer = _renderer(prompt_tokens)
    counter = lambda text: len(text.split())  # noqa: E731

    def evaluate_filter(record, problem):
        return local.evaluate_sample(
            record, problem, renderer=renderer, token_counter=counter
        )

    def verify_fn(record, problem):
        if problem["kind"] == "math":
            return True, {"decided_by": "timeout", "timed_out": True}
        return True, {"passed": True}

    def evaluate_full(record, problem):
        return local.evaluate_sample(
            record,
            problem,
            renderer=renderer,
            token_counter=counter,
            verify_fn=verify_fn,
        )

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        return [
            SimpleNamespace(
                text="answer",
                reasoning="t " * 10,
                token_ids=None,
                finish_reason="stop",
                completion_tokens=11,
            )
            for _ in conversations
        ]

    targets = [
        local.Target("pilot_p1", 59, 59, 2, verified=True, rows_target=2)
    ]
    output = tmp_path / "out"
    run_stats = local.run_sampling(
        problems=problems,
        targets=targets,
        output_dir=output,
        shard_size=2,
        samples_per_prompt=1,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: prompt_tokens[problem["problem_id"]],
        evaluate_fn=evaluate_filter,
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="run",
        pool_manifest_sha256=manifest["manifest_sha256"],
        model_revision="rev",
    )
    _stats, payload = script.finalize(
        args=SimpleNamespace(),
        output_dir=output,
        problems=problems,
        targets=targets,
        pool_dir=pool,
        pool_manifest=manifest,
        problems_path=pool / "problems.jsonl",
        run_id="run",
        model_revision="rev",
        run_stats=run_stats,
        evaluate_filter=evaluate_filter,
        evaluate_full=evaluate_full,
        verifiers={"math": "math-verify", "code": "stage0-grader"},
    )
    grades = {row["problem_id"]: row for row in local.read_jsonl(output / "sample_grades.jsonl")}
    assert grades["math-pilot"]["decided_by"] == "timeout"
    assert grades["math-pilot"]["verifiers"] == ["normaliser"]
    assert grades["math-pilot"]["timed_out"] is True
    assert grades["code-pilot"]["decided_by"] == "code_grader"
    assert grades["code-pilot"]["verifiers"] == ["code_grader"]
    assert payload["verification"]["decided_by"] == {
        "code_grader": 1,
        "timeout": 1,
    }


def test_finalize_writes_the_dedicated_monitor_corpus(tmp_path: Path):
    script = _load_script()
    pool = tmp_path / "pool"
    pool.mkdir()
    problems = [_problem("monitor-math", kind="math", bucket=0)]
    problems[0]["prompt"] = problems[0]["problem_id"]
    problems[0]["split"] = "monitor"
    problems[0]["pilot_subsplit"] = None
    problems_text = "".join(json.dumps(problem) + "\n" for problem in problems)
    (pool / "problems.jsonl").write_text(problems_text, encoding="utf-8")
    manifest = {
        "stage": "stage0_pool",
        "manifest_sha256": "poolsha",
        "creative_briefs": PART1_CREATIVE_BRIEFS,
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(problems_text.encode()).hexdigest(),
                "rows": len(problems),
            }
        },
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    renderer = _renderer({problems[0]["problem_id"]: 100})
    counter = lambda text: len(text.split())  # noqa: E731

    def evaluate_filter(record, problem):
        return local.evaluate_sample(
            record, problem, renderer=renderer, token_counter=counter
        )

    def fake_generate(*args, **kwargs):
        return [
            SimpleNamespace(
                text="answer",
                reasoning="t " * 10,
                token_ids=None,
                finish_reason="stop",
                completion_tokens=11,
            )
            for _ in range(len(args[2]))
        ]

    targets = [
        local.Target(
            "monitor",
            0,
            49,
            1,
            verified=False,
            rows_target=1,
            samples_per_prompt=1,
        )
    ]
    output = tmp_path / "out"
    run_stats = local.run_sampling(
        problems=problems,
        targets=targets,
        output_dir=output,
        shard_size=2,
        samples_per_prompt=1,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: 100,
        evaluate_fn=evaluate_filter,
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="monitor-run",
        pool_manifest_sha256=manifest["manifest_sha256"],
        model_revision="rev",
    )
    stats, payload = script.finalize(
        args=SimpleNamespace(),
        output_dir=output,
        problems=problems,
        targets=targets,
        pool_dir=pool,
        pool_manifest=manifest,
        problems_path=pool / "problems.jsonl",
        run_id="monitor-run",
        model_revision="rev",
        run_stats=run_stats,
        evaluate_filter=evaluate_filter,
        evaluate_full=evaluate_filter,
        verifiers={"math": "math-verify"},
        seed=7,
    )
    monitor_rows = output / "monitor" / "monitor_rows.jsonl"
    assert monitor_rows.is_file()
    rows = local.read_jsonl(monitor_rows)
    assert len(rows) == 1
    assert rows[0]["problem_id"] == problems[0]["problem_id"]
    assert "messages" in rows[0]
    monitor_manifest = json.loads(
        (output / "monitor" / "manifest.json").read_text(encoding="utf-8")
    )
    assert monitor_manifest["problem_ids"] == [problems[0]["problem_id"]]
    assert monitor_manifest["buckets"] == [0, 49]
    assert monitor_manifest["samples_per_prompt"] == 1
    assert monitor_manifest["selection_seed"] == 7
    assert monitor_manifest["partition_seed"] == 7
    assert payload["selection_order"]["seed"] == 7
    assert payload["monitor"]["rows"] == 1
    assert payload["monitor"]["sha256"] == monitor_manifest["sha256"]
    ok, reason = local.verify_manifest(output / "manifest.json")
    assert ok, reason


def test_target_monitor_restricts_the_dry_run_plan(tmp_path: Path, capsys):
    script = _load_script()
    code = script.main(
        [
            "--dry-run",
            "--pool-dir",
            str(tmp_path / "missing-pool"),
            "--output-dir",
            str(tmp_path / "out"),
            "--target",
            "monitor",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["targets"]) == {"monitor"}
    assert payload["targets"]["monitor"]["buckets"] == [0, 49]
    assert payload["targets"]["monitor"]["samples_per_prompt"] == 1
