"""D1 corpus helpers and CLI: CPU-only tests with fake samplers/verifiers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaloop.stage0 import d1, local
from gemmaloop.stage0.constants import PART1_CREATIVE_BRIEFS
from gemmaloop.stage0.postprocess import MAX_LENGTH

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sample_d1.py"
MODAL_SCRIPT = ROOT / "scripts" / "modal_stage0.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _problem(pid: str, *, kind: str = "math", bucket: int) -> dict:
    return {
        "problem_id": pid,
        "kind": kind,
        "source": "omr" if kind == "math" else "taco",
        "prompt": f"prompt {pid}",
        "answer": "42" if kind == "math" else None,
        "tests": [],
        "split": "nll_kl" if 60 <= bucket <= 69 else "train",
        "pilot_subsplit": None,
        "bucket": bucket,
    }


def _record(
    problem: dict,
    index: int,
    *,
    answer: str = "\\boxed{42}",
    finish: str = "stop",
    engine: str = d1.D1_LOCAL_ENGINE,
) -> dict:
    reasoning = "think " * 20
    return {
        "problem_id": problem["problem_id"],
        "sample_index": index,
        "completion": reasoning + answer,
        "reasoning": reasoning,
        "content": answer,
        "finish_reason": finish,
        "thought_closed": True,
        "tokens": 50,
        "prompt_tokens": 10,
        "max_new_tokens": 500,
        "engine_cap": 500,
        "engine": engine,
        "model_revision": "rev",
    }


def _renderer(_messages) -> int:
    return 100


def _counter(text: str) -> int:
    return len(text.split())


def _verifier(correct_indices: set[int]):
    def verify(sample, problem):
        index = int(sample.get("sample_index") or 0)
        correct = index in correct_indices
        return correct, {"correct": correct, "decided_by": "math_verify"}

    return verify


def _answer_verifier(sample, problem):
    correct = "\\boxed{42}" in str(sample.get("content"))
    return correct, {"correct": correct, "decided_by": "math_verify"}


def _partition() -> d1.D1Partition:
    problems = [
        _problem("fit-a", bucket=60),
        _problem("fit-b", bucket=61),
        _problem("test-a", bucket=62),
        _problem("test-b", bucket=69),
        _problem("code-test", kind="code", bucket=62),
        _problem("chat-fit", kind="chat", bucket=60),
        _problem("math-train", bucket=10),
        _problem("math-pilot", bucket=59),
    ]
    return d1.partition_d1_problems(problems)


def test_partition_selects_only_math_in_60_69():
    partition = _partition()
    assert [row["problem_id"] for row in partition.fit] == ["fit-a", "fit-b"]
    assert [row["problem_id"] for row in partition.test] == ["test-a", "test-b"]
    assert partition.counts == {"fit": 2, "test": 2, "total": 4}
    assert partition.split_of("test-b") == "test"
    assert partition.split_of("math-train") is None


def test_targets_plan_four_requests_per_problem_in_the_existing_sampler():
    partition = _partition()
    targets = d1.d1_targets(partition)
    assert [(t.name, t.buckets, t.problems, t.samples_per_prompt) for t in targets] == [
        ("d1_fit", (60, 61), 2, 4),
        ("d1_test", (62, 69), 2, 4),
    ]
    plan = local.build_shard_plan(
        partition.problems,
        targets,
        shard_size=1,
        samples_per_prompt=4,
        seed=0,
    )
    planned = [ref.problem_id for shard in plan for ref in shard.requests]
    assert len(planned) == len(partition.problems) * 4
    for problem in partition.problems:
        assert planned.count(problem["problem_id"]) == 4
    with pytest.raises(ValueError):
        d1.d1_targets(partition, samples_per_prompt=2)


def test_build_corpus_selects_lowest_verified_index_and_counts_every_sample():
    partition = _partition()
    records = []
    # fit-a: sample 0 is dropped by the filter, samples 1 and 3 are correct.
    records += [
        _record(partition.fit[0], 0, finish="length"),
        _record(partition.fit[0], 1),
        _record(partition.fit[0], 2, answer="\\boxed{41}"),
        _record(partition.fit[0], 3),
        # fit-b: no correct sample.
        *[_record(partition.fit[1], i, answer="\\boxed{41}") for i in range(4)],
        # test-a: sample 2 is the lowest correct one; sample 3 too.
        *[_record(partition.test[0], 0, answer="\\boxed{41}")],
        *[_record(partition.test[0], 1, answer="\\boxed{41}")],
        *[_record(partition.test[0], 2)],
        *[_record(partition.test[0], 3)],
        # test-b: only sample 0 correct.
        *[_record(partition.test[1], 0)],
        *[_record(partition.test[1], i, answer="\\boxed{41}") for i in (1, 2, 3)],
    ]
    corpus = d1.build_corpus(
        records,
        partition,
        renderer=_renderer,
        token_counter=_counter,
        verify_fn=_answer_verifier,
    )
    traces = {row["problem_id"]: row for row in corpus["traces"]}
    assert traces["fit-a"]["sample_index"] == 1
    assert traces["fit-a"]["request_id"] == "fit-a:1"
    assert traces["test-a"]["sample_index"] == 2
    assert traces["test-b"]["sample_index"] == 0
    assert "fit-b" not in traces
    assert len(corpus["outcomes"]) == 16
    counts = corpus["counts"]
    assert counts["samples_input"] == 16
    assert counts["samples_verified_correct"] == 5
    assert counts["retained_traces"] == 3
    assert counts["dropped_by_reason"] == {"finish_reason": 1}
    assert counts["splits"]["fit"]["problems_verified_correct"] == 1
    assert counts["splits"]["test"]["problems_verified_correct"] == 2
    assert counts["engines"] == {d1.D1_LOCAL_ENGINE: 16}
    # Every outcome keeps the verifier decision for the addendum.
    assert {row["decided_by"] for row in corpus["outcomes"]} == {None, "math_verify"}


def test_missing_samples_and_unknown_records_are_counted_not_silent():
    partition = _partition()
    records = [_record(partition.fit[0], 0)]
    records.append({**_record(partition.fit[0], 0), "problem_id": "not-in-d1"})
    corpus = d1.build_corpus(
        records,
        partition,
        renderer=_renderer,
        token_counter=_counter,
        verify_fn=_answer_verifier,
    )
    assert corpus["counts"]["unknown_records"] == 1
    assert corpus["counts"]["splits"]["fit"]["problems_missing_samples"] == 1
    assert corpus["counts"]["retained_traces"] == 1


def _write_source_pool(tmp_path: Path) -> Path:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    rows = [
        _problem("fit-a", bucket=60),
        _problem("test-a", bucket=62),
        _problem("code-test", kind="code", bucket=62),
        _problem("chat-fit", kind="chat", bucket=60),
        _problem("math-train", bucket=10),
    ]
    sha = local.write_jsonl_atomic(pool_dir / "problems.jsonl", rows)
    from gemmaloop.stage0.evals import EVAL_SPECS

    targets = {spec.name: {"status": "ok"} for spec in EVAL_SPECS}
    manifest = {
        "stage": "stage0_pool",
        "version": 1,
        "split_rule": "bucket = int(sha256(problem_id), 16) % 100, seed 0",
        "creative_briefs": PART1_CREATIVE_BRIEFS,
        "counts": {"total": len(rows)},
        "artifacts": {"problems.jsonl": {"sha256": sha, "rows": len(rows)}},
        "decontamination": {
            "status": "built",
            "targets": targets,
        },
        "limits": {
            "eval_limit": None,
            "limit_rows": None,
            "allow_missing_eval": False,
            "allow_missing_briefs": False,
            "allow_short_sources": False,
            "allow_duplicates": False,
        },
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    local.write_json_atomic(pool_dir / "manifest.json", manifest)
    return pool_dir


def test_derived_pool_is_loadable_and_accepted_by_the_bound_modal_validator(tmp_path):
    source = _write_source_pool(tmp_path)
    d1_dir = tmp_path / "d1_pool"
    selected, manifest = d1.write_d1_pool(source, d1_dir)
    assert [row["problem_id"] for row in selected] == ["fit-a", "test-a"]
    assert manifest["derived_from"]["manifest_sha256"] == json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]

    problems, loaded = local.load_pool(d1_dir)
    assert [row["problem_id"] for row in problems] == ["fit-a", "test-a"]
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]

    modal = _load_script("modal_stage0_for_d1_test", MODAL_SCRIPT)
    validated = modal.validate_pool_manifest(d1_dir)
    assert validated["_pool_manifest_sha256"] == manifest["manifest_sha256"]


def test_corpus_files_and_addendum_round_trip_and_freeze(tmp_path):
    partition = _partition()
    records = []
    correct = set()
    for problem in partition.problems:
        for index in range(4):
            records.append(_record(problem, index))
            if index == 2:
                correct.add(2)
    corpus = d1.build_corpus(
        records,
        partition,
        renderer=_renderer,
        token_counter=_counter,
        verify_fn=lambda sample, problem: (
            int(sample["sample_index"]) == 2,
            {"correct": int(sample["sample_index"]) == 2, "decided_by": "math_verify"},
        ),
    )
    addendum = d1.build_addendum(
        pool_dir=tmp_path / "pool",
        pool_manifest={"manifest_sha256": "poolsha"},
        pool_problems_sha256="poolproblems",
        partition=partition,
        corpus=corpus,
        model_id="google/gemma-4-E2B-it",
        model_revision="rev",
        verifiers={"math": "math-verify"},
    )
    out = tmp_path / "corpus"
    payload = d1.write_corpus(out, corpus, addendum)
    assert (out / "d1_traces.jsonl").exists()
    assert (out / "d1_verifier_outcomes.jsonl").exists()
    assert payload["counts"]["retained_traces"] == 4
    assert payload["counts"]["d1_evaluable"] is False  # 2/2 < 20/40 in this fixture
    assert d1.verify_frozen_addendum(out / "d1_data_addendum.json")["manifest_sha256"] == payload["manifest_sha256"]

    script = _load_script("sample_d1_for_freeze_test", SCRIPT)
    frozen = script.freeze_addendum(out / "d1_data_addendum.json")
    assert frozen["status"] == "frozen"
    assert d1.verify_frozen_addendum(Path(frozen["addendum_path"]))["status"] == "frozen"


def test_modal_argv_forwards_the_registered_run_controls(tmp_path):
    script = _load_script("sample_d1_for_argv_test", SCRIPT)
    args = script._parse_args(
        [
            "sample-modal",
            "--pool-dir",
            str(tmp_path / "pool"),
            "--d1-pool-dir",
            str(tmp_path / "d1"),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-usd",
            "9.5",
            "--shard-size",
            "25",
            "--ledger",
            str(tmp_path / "ledger.json"),
            "--spend-lock",
            str(tmp_path / "spend.lock"),
            "--init-ledger",
            "--yes",
        ]
    )
    argv = script.modal_argv(args, "abc123")
    assert argv[:2] == ["--mode", "sample"]
    assert argv[argv.index("--samples-per-prompt") + 1] == "4"
    assert argv[argv.index("--avg-generated-tokens") + 1] == "32768"
    assert argv[argv.index("--max-usd") + 1] == "9.5"
    assert argv[argv.index("--frozen-manifest-sha") + 1] == "abc123"
    assert "--yes" in argv and "--init-ledger" in argv
    assert argv[argv.index("--ledger") + 1] == str(tmp_path / "ledger.json")
    assert argv[argv.index("--spend-lock") + 1] == str(tmp_path / "spend.lock")


def test_run_modal_sampling_builds_and_binds_the_derived_pool(tmp_path, monkeypatch):
    script = _load_script("sample_d1_for_modal_test", SCRIPT)
    source = _write_source_pool(tmp_path)
    args = script._parse_args(
        [
            "sample-modal",
            "--pool-dir",
            str(source),
            "--d1-pool-dir",
            str(tmp_path / "d1_pool"),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    calls = []

    class FakeModal:
        @staticmethod
        def main(argv):
            calls.append(list(argv))
            return 0

    assert script.run_modal_sampling(args, modal_script=FakeModal) == 0
    manifest = json.loads((tmp_path / "d1_pool" / "manifest.json").read_text(encoding="utf-8"))
    argv = calls[0]
    assert argv[:2] == ["--mode", "plan"]
    assert argv[argv.index("--frozen-manifest-sha") + 1] == manifest["manifest_sha256"]
    assert "--yes" not in argv


def test_local_sampler_integration_with_a_fake_generator(tmp_path):
    problems = [_problem("fit-a", bucket=60), _problem("test-a", bucket=62)]
    partition = d1.partition_d1_problems(problems)
    targets = d1.d1_targets(partition)
    calls: list[list[str]] = []

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        calls.append(list(request_ids or []))
        return [
            SimpleNamespace(
                text="\\boxed{42}",
                reasoning="think " * 20,
                token_ids=None,
                finish_reason="stop",
                completion_tokens=50,
            )
            for _ in conversations
        ]

    stats = local.run_sampling(
        problems=partition.problems,
        targets=targets,
        output_dir=tmp_path / "raw",
        shard_size=1,
        samples_per_prompt=4,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: 100,
        evaluate_fn=lambda record, problem: local.evaluate_sample(
            record, problem, renderer=_renderer, token_counter=_counter
        ),
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="r",
        pool_manifest_sha256="pool",
        model_revision="rev",
    )
    assert stats.records == 8
    assert sum(len(call) for call in calls) == 8
    raw = []
    for shard in sorted((tmp_path / "raw" / "shards").glob("shard_*.jsonl")):
        raw.extend(local.read_jsonl(shard))
    corpus = d1.build_corpus(
        raw,
        partition,
        renderer=_renderer,
        token_counter=_counter,
        verify_fn=_verifier({0}),
    )
    assert corpus["counts"]["samples_input"] == 8
    assert corpus["counts"]["retained_traces"] == 2
    assert all(row["sample_index"] == 0 for row in corpus["traces"])


def test_verify_corpus_cli_path_writes_traces_outcomes_and_addendum(tmp_path):
    script = _load_script("sample_d1_for_verify_test", SCRIPT)
    source = _write_source_pool(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    partition = d1.partition_d1_problems(local.load_pool(source)[0])
    rows = [
        _record(problem, index)
        for problem in partition.problems
        for index in range(4)
    ]
    local.write_jsonl_atomic(raw / "shard_00000.jsonl", rows)
    out = tmp_path / "corpus"
    payload = script.verify_corpus(
        pool_dir=source,
        raw_dir=raw,
        output_dir=out,
        renderer=_renderer,
        token_counter=_counter,
        allow_math_fallback=True,
    )
    assert payload["kind"] == "prereg_v1_data_d1"
    assert payload["counts"]["retained_traces"] == 2
    assert (out / "d1_traces.jsonl").exists()
    assert (out / "d1_verifier_outcomes.jsonl").exists()
    addendum = json.loads((out / "d1_data_addendum.json").read_text(encoding="utf-8"))
    assert addendum["selection"]["fit_trace_ids"] == ["fit-a:0"]
    assert addendum["selection"]["test_trace_ids"] == ["test-a:0"]
    assert addendum["data"]["input_counts"]["fit"] == {
        "total": 2,
        "by_kind": {"chat": 1, "math": 1},
    }
    assert addendum["data"]["input_counts"]["test"] == {
        "total": 2,
        "by_kind": {"code": 1, "math": 1},
    }
    assert d1.verify_frozen_addendum(out / "d1_data_addendum.json")["counts"]["retained_traces"] == 2
