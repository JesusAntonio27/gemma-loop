"""Local Stage-0-lite sampler: pure helpers, cap equivalence, resume, manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaloop.prereg import default_prereg_path, load_prereg
from gemmaloop.stage0 import local
from gemmaloop.stage0.postprocess import MAX_LENGTH


def _problem(
    pid: str, *, kind: str = "math", bucket: int = 0, source: str | None = None
) -> dict:
    if source is None:
        source = {"code": "taco", "math": "omr", "chat": "chat"}[kind]
    return {
        "problem_id": pid,
        "kind": kind,
        "source": source,
        "prompt": pid,
        "answer": "42",
        "tests": [],
        "split": "train" if bucket <= 49 else "pilot_dev",
        "pilot_subsplit": "pilot_p1" if bucket == 59 else None,
        "bucket": bucket,
    }


def _output(tokens: int, cap: int) -> SimpleNamespace:
    generated = min(int(tokens), int(cap))
    reasoning = (4 * generated) // 5
    content = generated - reasoning
    return SimpleNamespace(
        text="a " * content,
        reasoning="t " * reasoning,
        token_ids=None,
        finish_reason="stop" if int(tokens) <= int(cap) else "length",
        completion_tokens=generated,
    )


def _renderer(prompt_tokens: dict[str, int]):
    def render(messages):
        first = messages[0]["content"]
        assistant = messages[-1]
        return (
            prompt_tokens[first]
            + len(str(assistant.get("content") or "").split())
            + len(str(assistant.get("reasoning") or "").split())
        )

    return render


def test_request_identity_matches_the_modal_record_pair():
    pid = "taco:0123456789abcdef"
    assert local.stage0_request_id(pid, 3) == f"{pid}:3"
    assert local.parse_stage0_request_id(f"{pid}:3") == (pid, 3)
    assert local.per_problem_seed(pid) == int(
        hashlib.sha256(pid.encode("utf-8")).hexdigest()[:8], 16
    )
    # The Modal job writes exactly this pair into every record.
    modal_record = {"problem_id": pid, "sample_index": 2}
    assert (
        local.stage0_request_id(
            modal_record["problem_id"], modal_record["sample_index"]
        )
        == f"{pid}:2"
    )


def test_budget_and_batching_never_cut_a_request_before_its_budget():
    problems = [_problem(f"p{index}") for index in range(6)]
    prompt_lengths = [100, 220, 260, 700, 701, 1500]
    requests = [
        local.Stage0Request(
            problem=problem,
            sample_index=0,
            prompt_tokens=length,
            budget=local.prompt_budget(MAX_LENGTH, length),
        )
        for problem, length in zip(problems, prompt_lengths)
    ]
    calls = local.plan_request_calls(requests)
    assert calls
    for call in calls:
        assert call.max_new_tokens >= max(request.budget for request in call.requests)
        assert 1 <= call.max_batch <= 32
    at_8k = local.plan_request_calls(
        [
            local.Stage0Request(
                problem=problems[0],
                sample_index=0,
                prompt_tokens=100,
                budget=MAX_LENGTH - 100,
            )
        ]
    )
    assert at_8k[0].max_new_tokens == MAX_LENGTH - 100
    # E2B 8k bucket is 32 (measured non-spilling ceiling).
    from gemmaloop.serve.generate import max_batch_for

    assert max_batch_for(8192, "e2b") == 32


def test_per_request_cap_keeps_exactly_the_registered_filter_set():
    prompt_tokens = 500
    natural = [100, 400, 5000, 7692, 7693, 20000, 40000]
    problems = [_problem(f"p{index}") for index in range(len(natural))]
    token_counts = {problem["problem_id"]: prompt_tokens for problem in problems}

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
            _output(natural[int(request_id.rsplit(":", 1)[1])], sampling.max_new_tokens)
            for request_id in request_ids
        ]

    def run(budgets):
        requests = [
            local.Stage0Request(
                problem=problem,
                sample_index=int(problem["problem_id"][1:]),
                prompt_tokens=prompt_tokens,
                budget=budget,
            )
            for problem, budget in zip(problems, budgets)
        ]
        records = local.generate_calls(
            requests,
            generate_fn=fake_generate,
            model=None,
            processor=None,
            decode=None,
            run_id="r",
            pool_manifest_sha256="pool",
            shard_index=0,
            model_revision="rev",
        )
        kept = set()
        for record in records:
            verdict = local.evaluate_sample(
                record,
                problems[int(record["problem_id"][1:])],
                renderer=_renderer(token_counts),
                token_counter=lambda text: len(text.split()),
            )
            if verdict.kept:
                kept.add(record["request_id"])
        return kept

    local_kept = run([MAX_LENGTH - prompt_tokens] * len(natural))
    reference_kept = run([local.REGISTERED_MAX_NEW_TOKENS] * len(natural))
    assert local_kept == reference_kept
    # 7693 generated tokens would render 8193 > 8192 under the full cap too.
    assert "p3:3" in local_kept
    assert "p4:4" not in local_kept
    assert "p5:5" not in local_kept
    assert "p6:6" not in local_kept


def test_filter_applies_the_reasoning_share_rule():
    problem = _problem("p", kind="code")
    good = {
        "problem_id": "p",
        "sample_index": 0,
        "completion": "t t t t a",
        "reasoning": "t t t t",
        "content": "a",
        "finish_reason": "stop",
        "thought_closed": True,
        "tokens": 5,
        "prompt_tokens": 10,
    }
    bad = dict(good, completion="t a a a a", reasoning="t", content="a a a a")
    counter = lambda text: len(text.split())
    renderer = lambda messages: 10 + 5  # noqa: E731
    assert local.evaluate_sample(
        good, problem, renderer=renderer, token_counter=counter
    ).kept
    verdict = local.evaluate_sample(
        bad, problem, renderer=renderer, token_counter=counter
    )
    assert not verdict.kept and verdict.drop_reason == "reasoning_share"


def test_resume_skips_requests_already_persisted(tmp_path: Path):
    problems = [_problem("p0"), _problem("p1")]
    target = local.Target("train", 0, 0, 2, verified=False)
    output = tmp_path / "out"
    shard, committed, partial = local.shard_paths(output, 0)
    partial.parent.mkdir(parents=True)
    local.ensure_run_marker(
        output,
        local.build_run_marker(
            run_id="r",
            pool_manifest_sha256="pool",
            targets=[target],
            samples_per_prompt=1,
            shard_size=2,
        ),
    )
    done_record = local.build_sample_record(
        problem=problems[0],
        sample_index=0,
        request=None,
        output=_output(10, 500),
        run_id="r",
        pool_manifest_sha256="pool",
        shard_index=0,
        prompt_tokens=100,
        budget=500,
        engine_cap=500,
        model_revision="rev",
        samples_per_prompt=1,
    )
    local.append_jsonl(partial, [done_record])

    calls: list[list[str]] = []

    def fake_generate(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        request_ids=None,
        max_batch=None,
    ):
        calls.append(list(request_ids or []))
        return [_output(20, sampling.max_new_tokens) for _ in conversations]

    kwargs = dict(
        problems=problems,
        targets=[target],
        output_dir=output,
        shard_size=2,
        samples_per_prompt=1,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: 100,
        evaluate_fn=lambda record, problem: local.evaluate_sample(
            record,
            problem,
            renderer=_renderer({"p0": 100, "p1": 100}),
            token_counter=lambda text: len(text.split()),
        ),
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="r",
        pool_manifest_sha256="pool",
        model_revision="rev",
    )
    stats = local.run_sampling(**kwargs)
    assert calls == [["p1:0"]]
    assert stats.shards_committed == [0]
    assert committed.exists() and shard.exists() and not partial.exists()
    assert len(local.read_jsonl(shard)) == 2

    calls.clear()
    second = local.run_sampling(**kwargs)
    assert calls == []
    assert second.shards_committed == [0]
    assert second.records == 2


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    local.write_jsonl_atomic(output / "a.jsonl", [{"x": 1}, {"x": 2}])
    local.write_jsonl_atomic(output / "b.jsonl", [{"y": 1}])
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "problems.jsonl").write_text(
        json.dumps({"problem_id": "p"}) + "\n", encoding="utf-8"
    )
    stats = local.RunStats()
    stats.samples_generated = 3
    stats.kept = 2
    stats.train_tokens = 123
    stats.accepted_assistant_tokens = 123
    stats.add_cap(8000)
    stats.add_cap(8100)
    payload = local.build_manifest(
        output_dir=output,
        pool_dir=pool,
        pool_manifest={"manifest_sha256": "poolsha"},
        problems_path=pool / "problems.jsonl",
        run_id="run",
        stats=stats,
        targets=local.targets_from_prereg(load_prereg(default_prereg_path())),
        request_ids=["p:0", "p:1"],
        model_revision="rev",
        created="2026-11-05T00:00:00Z",
        build_command="uv run python scripts/build_stage0_pool.py --output-dir pool",
        verifiers={"math": "math-verify"},
        files={"a.jsonl": 2, "b.jsonl": 1},
    )
    path = output / "manifest.json"
    local.write_json_atomic(path, payload)
    ok, reason = local.verify_manifest(path)
    assert ok, reason
    assert payload["engine"] == local.ENGINE
    assert payload["caps"]["per_request"]["max"] == 8100
    assert payload["requests"]["count"] == 2
    assert len(payload["manifest_sha256"]) == 64

    (output / "a.jsonl").write_text('{"x": 9}\n', encoding="utf-8")
    ok, reason = local.verify_manifest(path)
    assert not ok and ("sha256" in reason or "size" in reason)


def test_targets_match_the_preregistered_buckets_and_counts():
    prereg = load_prereg(default_prereg_path())
    targets = {target.name: target for target in local.targets_from_prereg(prereg)}
    assert targets["train"].buckets == (0, 49)
    assert targets["train"].problems == 2500
    assert targets["train"].token_target == 3_000_000
    assert not targets["train"].verified
    assert targets["pilot_p1"].buckets == (59, 59)
    assert targets["pilot_p1"].problems == 50
    assert targets["pilot_p1"].rows_target == 200
    assert targets["monitor"].buckets == (0, 49)
    assert targets["monitor"].problems == 64
    assert targets["monitor"].samples_per_prompt == 1
    assert not targets["monitor"].verified
    assert targets["pilot_tfit"].buckets == (57, 58)
    assert targets["pilot_tfit"].problems == 100
    assert targets["pilot_tfit"].rows_target == 0  # problems registered, rows not
    assert targets["pilot_selection"].buckets == (54, 56)
    assert targets["pilot_selection"].problems == 150
    assert targets["pilot_selection"].rows_target == 0
    assert local.SAMPLES_PER_PROMPT == 4
    assert local.REGISTERED_MAX_LENGTH == 8192
    assert local.REGISTERED_MIN_REASONING_SHARE == 0.75


def _pool_manifest_payload(problems: bytes, *, smoke: bool = False) -> dict:
    from gemmaloop.stage0.evals import EVAL_SPECS
    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

    targets = {spec.name: {"status": "ok"} for spec in EVAL_SPECS}
    manifest = {
        "stage": "stage0_pool",
        "creative_briefs": PART1_CREATIVE_BRIEFS,
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(problems).hexdigest(),
                "rows": 1,
            }
        },
        "decontamination": {"targets": targets},
        "limits": {
            "eval_limit": 64 if smoke else None,
            "limit_rows": None,
            "allow_missing_eval": False,
            "allow_missing_briefs": False,
            "allow_short_sources": False,
            "allow_duplicates": False,
        },
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    return manifest


def _write_pool(tmp_path: Path, manifest: dict) -> tuple[Path, bytes]:
    import json as _json

    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    problems = (_json.dumps({"problem_id": "p"}) + chr(10)).encode()
    (pool / "problems.jsonl").write_bytes(problems)
    manifest = dict(manifest)
    manifest["artifacts"] = {
        "problems.jsonl": {
            "sha256": hashlib.sha256(problems).hexdigest(),
            "rows": 1,
        }
    }
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    (pool / "manifest.json").write_text(
        _json.dumps(manifest), encoding="utf-8"
    )
    return pool, problems


def test_pool_manifest_validation_rejects_a_smoke_pool(tmp_path: Path):
    pool, problems = _write_pool(
        tmp_path, _pool_manifest_payload(b"", smoke=True)
    )
    with pytest.raises(ValueError, match="smoke"):
        local.load_pool(pool)


def test_load_pool_accepts_part1_and_fails_closed_on_missing_targets(tmp_path: Path):
    valid = _pool_manifest_payload(b"")
    from gemmaloop.stage0.evals import EVAL_SPECS

    pool, _problems = _write_pool(tmp_path, valid)
    problems, manifest = local.load_pool(pool)
    assert problems and manifest["creative_briefs"]

    no_section = dict(valid)
    no_section.pop("decontamination")
    pool2, _ = _write_pool(tmp_path / "no_section", no_section)
    with pytest.raises(ValueError, match="decontamination section"):
        local.load_pool(pool2)

    missing = _pool_manifest_payload(b"")
    missing["decontamination"]["targets"] = {
        spec.name: {"status": "ok"} for spec in EVAL_SPECS[:-1]
    }
    pool3, _ = _write_pool(tmp_path / "missing", missing)
    with pytest.raises(ValueError, match="missing registered targets"):
        local.load_pool(pool3)

    no_briefs = _pool_manifest_payload(b"")
    no_briefs.pop("creative_briefs")
    pool4, _ = _write_pool(tmp_path / "no_briefs", no_briefs)
    with pytest.raises(ValueError, match="creative-briefs"):
        local.load_pool(pool4)


def test_hash_selection_is_source_neutral_and_mixed():
    sources = ("apps", "chat", "codecontests", "dapo", "omr", "taco")
    problems = [
        _problem(
            f"{source}-{index}",
            kind=(
                "code"
                if source in {"apps", "codecontests", "taco"}
                else ("chat" if source == "chat" else "math")
            ),
            bucket=0,
            source=source,
        )
        for source in sources
        for index in range(8)
    ]
    source_ordered_ids = [problem["problem_id"] for problem in problems]
    target = local.Target("train", 0, 49, 12, token_target=10, verified=False)
    selected = local.select_problems(problems, target)
    assert len(selected) == 12
    assert len({problem["source"] for problem in selected}) >= 3
    assert [problem["problem_id"] for problem in selected] != source_ordered_ids[:12]

    plan = local.build_shard_plan(
        problems, [target], shard_size=3, samples_per_prompt=4, seed=0
    )
    keys = [
        local.problem_order_key(0, ref.problem_id)
        for shard in plan
        for ref in shard.requests
    ]
    assert keys == sorted(keys)
    first_sources = {ref.problem["source"] for ref in plan[0].requests}
    assert len(first_sources) >= 2
    again = local.build_shard_plan(
        problems, [target], shard_size=3, samples_per_prompt=4, seed=0
    )
    assert [ref.request_id for shard in plan for ref in shard.requests] == [
        ref.request_id for shard in again for ref in shard.requests
    ]


def test_build_manifest_records_selection_order_and_source_mix(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    pool = output / "pool"
    pool.mkdir()
    problems_path = pool / "problems.jsonl"
    problems_path.write_text(
        json.dumps({"problem_id": "p"}) + chr(10), encoding="utf-8"
    )
    (output / "a.jsonl").write_text("{}" + chr(10), encoding="utf-8")
    stats = local.RunStats()
    stats.source_mix["train"] = {
        "omr": {"records": 3, "kept": 2, "accepted_assistant_tokens": 20},
        "taco": {"records": 1, "kept": 1, "accepted_assistant_tokens": 5},
    }
    targets = [local.Target("train", 0, 49, 2, token_target=10, verified=False)]
    payload = local.build_manifest(
        output_dir=output,
        pool_dir=pool,
        pool_manifest={"manifest_sha256": "poolsha"},
        problems_path=problems_path,
        run_id="run",
        stats=stats,
        targets=targets,
        request_ids=["p:0"],
        model_revision="rev",
        created="2026-11-05T00:00:00Z",
        build_command="build --part1",
        verifiers={"math": "math-verify"},
        files={"a.jsonl": 1},
        selection_seed=0,
    )
    assert payload["selection_order"]["seed"] == 0
    assert "sha256(seed:request_id)" in payload["selection_order"]["rule"]
    assert payload["targets"]["train"]["sources"]["omr"]["kept"] == 2
    assert payload["shortfalls"]["train_tokens"]["target"] == 10

def test_resume_refuses_a_different_run_identity(tmp_path: Path):
    problems = [_problem("p0"), _problem("p1")]
    target = local.Target("train", 0, 0, 2, verified=False)
    output = tmp_path / "out"
    shard, committed, partial = local.shard_paths(output, 0)
    partial.parent.mkdir(parents=True)
    local.ensure_run_marker(
        output,
        local.build_run_marker(
            run_id="run",
            pool_manifest_sha256="pool",
            targets=[target],
            samples_per_prompt=1,
            shard_size=2,
        ),
    )
    foreign = local.build_sample_record(
        problem=problems[0],
        sample_index=0,
        request=None,
        output=_output(10, 500),
        run_id="other-run",
        pool_manifest_sha256="pool",
        shard_index=0,
        prompt_tokens=100,
        budget=500,
        engine_cap=500,
        model_revision="rev",
        samples_per_prompt=1,
    )
    local.append_jsonl(partial, [foreign])

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        return [_output(20, sampling.max_new_tokens) for _ in conversations]

    with pytest.raises(ValueError, match="belongs to run"):
        local.run_sampling(
            problems=problems,
            targets=[target],
            output_dir=output,
            shard_size=2,
            samples_per_prompt=1,
            max_length=MAX_LENGTH,
            prompt_tokens_for=lambda problem: 100,
            evaluate_fn=lambda record, problem: local.evaluate_sample(
                record,
                problem,
                renderer=_renderer({"p0": 100, "p1": 100}),
                token_counter=lambda text: len(text.split()),
            ),
            generate_fn=fake_generate,
            model=None,
            processor=None,
            decode=None,
            run_id="run",
            pool_manifest_sha256="pool",
            model_revision="rev",
        )


def test_progress_delta_does_not_double_count_on_resume(tmp_path: Path):
    problems = [_problem("p0"), _problem("p1")]
    target = local.Target("train", 0, 0, 2, verified=False)
    output = tmp_path / "out"

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        return [_output(20, sampling.max_new_tokens) for _ in conversations]

    kwargs = dict(
        problems=problems,
        targets=[target],
        output_dir=output,
        shard_size=2,
        samples_per_prompt=1,
        max_length=MAX_LENGTH,
        prompt_tokens_for=lambda problem: 100,
        evaluate_fn=lambda record, problem: local.evaluate_sample(
            record,
            problem,
            renderer=_renderer({"p0": 100, "p1": 100}),
            token_counter=lambda text: len(text.split()),
        ),
        generate_fn=fake_generate,
        model=None,
        processor=None,
        decode=None,
        run_id="run",
        pool_manifest_sha256="pool",
        model_revision="rev",
    )
    first: list[dict] = []
    local.run_sampling(**kwargs, progress=first.append)
    total = first[-1]["generated_tokens"]
    assert total > 0
    assert sum(payload["generated_tokens_delta"] for payload in first) == total

    second: list[dict] = []
    local.run_sampling(**kwargs, progress=second.append)
    assert second
    assert all(payload["generated_tokens_delta"] == 0 for payload in second)


def test_merge_hidden_tests_appends_without_duplicates(tmp_path: Path):
    problem = {"problem_id": "p", "tests": [{"input": "1", "output": "1"}]}
    empty = {"problem_id": "q", "tests": []}
    path = tmp_path / "hidden_tests.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "p",
                "tests": [
                    {"input": "1", "output": "1"},
                    {"input": "2", "output": "2"},
                ],
            }
        )
        + chr(10)
        + json.dumps(
            {"problem_id": "q", "tests": [{"input": "3", "output": "3"}]}
        )
        + chr(10),
        encoding="utf-8",
    )
    meta = local.merge_hidden_tests([problem, empty], path)
    assert problem["tests"] == [
        {"input": "1", "output": "1"},
        {"input": "2", "output": "2"},
    ]
    assert empty["tests"] == [{"input": "3", "output": "3"}]
    assert meta["rows"] == 2
    assert meta["problems_merged"] == 2
    assert len(meta["sha256"]) == 64


def test_load_partial_discards_a_torn_trailing_line(tmp_path: Path):
    path = tmp_path / "shard_00000.jsonl.partial"
    record = {"request_id": "p:0", "problem_id": "p", "sample_index": 0}
    path.write_text(
        json.dumps(record) + chr(10) + "{torn", encoding="utf-8"
    )
    rows = local.load_partial(path)
    assert rows == [record]
    rewritten = path.read_text(encoding="utf-8")
    assert "torn" not in rewritten
    assert json.loads(rewritten) == record


def test_run_marker_refuses_a_different_selection_identity(tmp_path: Path):
    problems = [_problem("p0"), _problem("p1")]
    target = local.Target("train", 0, 0, 2, verified=False)
    output = tmp_path / "out"
    output.mkdir()
    local.write_json_atomic(
        output / "run_marker.json",
        {
            "run_id": "other-run",
            "pool_manifest_sha256": "pool",
            "selection_order": "sha256(seed:request_id) ascending over the whole bucket",
            "seed": 0,
            "samples_per_prompt": 1,
            "shard_size": 2,
        },
    )

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        return [_output(20, sampling.max_new_tokens) for _ in conversations]

    with pytest.raises(ValueError, match="different Stage 0-lite run"):
        local.run_sampling(
            problems=problems,
            targets=[target],
            output_dir=output,
            shard_size=2,
            samples_per_prompt=1,
            max_length=MAX_LENGTH,
            prompt_tokens_for=lambda problem: 100,
            evaluate_fn=lambda record, problem: local.evaluate_sample(
                record,
                problem,
                renderer=_renderer({"p0": 100, "p1": 100}),
                token_counter=lambda text: len(text.split()),
            ),
            generate_fn=fake_generate,
            model=None,
            processor=None,
            decode=None,
            run_id="run",
            pool_manifest_sha256="pool",
            model_revision="rev",
        )


def test_resume_refuses_id_shards_without_a_run_marker(tmp_path: Path):
    problems = [_problem("p0")]
    target = local.Target("train", 0, 0, 1, verified=False)
    output = tmp_path / "out"
    shard, _committed, _partial = local.shard_paths(output, 0)
    shard.parent.mkdir(parents=True)
    shard.write_text("{}" + chr(10), encoding="utf-8")

    def fake_generate(model, processor, conversations, sampling, seed=0, request_ids=None, max_batch=None):
        return [_output(20, sampling.max_new_tokens) for _ in conversations]

    with pytest.raises(local.RunIdentityError, match="no run_marker.json"):
        local.run_sampling(
            problems=problems,
            targets=[target],
            output_dir=output,
            shard_size=1,
            samples_per_prompt=1,
            max_length=MAX_LENGTH,
            prompt_tokens_for=lambda problem: 100,
            evaluate_fn=lambda record, problem: local.evaluate_sample(
                record,
                problem,
                renderer=_renderer({"p0": 100}),
                token_counter=lambda text: len(text.split()),
            ),
            generate_fn=fake_generate,
            model=None,
            processor=None,
            decode=None,
            run_id="run",
            pool_manifest_sha256="pool",
            model_revision="rev",
        )


def test_run_marker_binds_targets_prereg_and_hidden_tests(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    targets = [
        local.Target("train", 0, 49, 2500, token_target=3_000_000, verified=False),
        local.Target("pilot_p1", 59, 59, 50, verified=True, rows_target=200),
    ]
    marker = local.build_run_marker(
        run_id="run",
        pool_manifest_sha256="pool",
        targets=targets,
        samples_per_prompt=4,
        shard_size=25,
        prereg_sha256="prereg-sha",
        hidden_tests_sha256="hidden-sha",
    )
    local.ensure_run_marker(output, marker)
    assert marker["selection_order"]["version"] == local.SELECTION_ORDER_VERSION
    assert marker["targets"]["train"]["token_target"] == 3_000_000
    assert marker["targets"]["pilot_p1"]["buckets"] == [59, 59]
    assert marker["prereg_sha256"] == "prereg-sha"
    assert marker["hidden_tests_sha256"] == "hidden-sha"

    for changed in (
        {**marker, "prereg_sha256": "other-prereg"},
        {**marker, "hidden_tests_sha256": "other-hidden"},
        {**marker, "selection_order": {**marker["selection_order"], "version": "old"}},
        {
            **marker,
            "targets": {
                **marker["targets"],
                "train": {**marker["targets"]["train"], "token_target": 1_000_000},
            },
        },
    ):
        with pytest.raises(local.RunIdentityError, match="different Stage 0-lite"):
            local.ensure_run_marker(output, changed)


def test_load_pool_accepts_a_default_briefs_pool(tmp_path: Path):
    from gemmaloop.stage0.evals import DEFAULT_BRIEFS

    manifest = _pool_manifest_payload(b"")
    manifest.pop("creative_briefs")
    manifest["decontamination"]["targets"][f"file:{DEFAULT_BRIEFS.name}"] = {
        "status": "ok",
        "path": str(DEFAULT_BRIEFS),
    }
    pool, _problems = _write_pool(tmp_path, manifest)
    problems, loaded = local.load_pool(pool)
    assert problems
    assert (
        loaded["decontamination"]["targets"][f"file:{DEFAULT_BRIEFS.name}"]["status"]
        == "ok"
    )


def test_run_marker_refuses_invalid_marker_json(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "run_marker.json").write_text("{not json", encoding="utf-8")
    marker = local.build_run_marker(
        run_id="run",
        pool_manifest_sha256="pool",
        targets=[local.Target("train", 0, 49, 2500, token_target=3_000_000, verified=False)],
        samples_per_prompt=4,
        shard_size=25,
    )
    with pytest.raises(local.RunIdentityError, match="not a valid run marker"):
        local.ensure_run_marker(output, marker)


def test_monitor_ids_are_the_largest_train_keys_and_train_excludes_them():
    problems = [_problem(f"p{index:03d}", bucket=0) for index in range(100)]
    targets = [
        local.Target("train", 0, 49, 2500, verified=False),
        local.Target(
            "monitor", 0, 49, 64, verified=False, rows_target=64, samples_per_prompt=1
        ),
        local.Target("pilot_tfit", 57, 58, 100, verified=True),
    ]
    monitor_ids = local.monitor_problem_ids(problems, targets, seed=0)
    expected = {
        problem["problem_id"]
        for problem in sorted(
            problems,
            key=lambda problem: (local.problem_order_key(0, problem["problem_id"]), problem["problem_id"]),
        )[-64:]
    }
    assert monitor_ids == expected

    train = local.select_problems(
        problems, targets[0], samples_per_prompt=4, seed=0, monitor_ids=monitor_ids
    )
    assert train
    assert {problem["problem_id"] for problem in train}.isdisjoint(monitor_ids)
    assert len(train) == 36  # the remaining smallest-key train problems

    monitor = local.select_problems(
        problems, targets[1], samples_per_prompt=4, seed=0, monitor_ids=monitor_ids
    )
    assert {problem["problem_id"] for problem in monitor} == monitor_ids

    # The shard plan feeds every training subset: no train request may name a
    # monitor problem, and the monitor target is exactly 64 x 1 sample.
    plan = local.build_shard_plan(
        problems, targets, shard_size=1000, samples_per_prompt=4, seed=0
    )
    train_requests = [
        ref for shard in plan if shard.target.name == "train" for ref in shard.requests
    ]
    monitor_requests = [
        ref for shard in plan if shard.target.name == "monitor" for ref in shard.requests
    ]
    assert {ref.problem_id for ref in train_requests}.isdisjoint(monitor_ids)
    assert {ref.problem_id for ref in monitor_requests} == monitor_ids
    assert len(monitor_requests) == 64
    assert all(ref.sample_index == 0 for ref in monitor_requests)

    # The registered pilot T-fit split (57-58) is untouched by monitor logic.
    assert targets[2].buckets == (57, 58)
    assert all(0 <= local.bucket_of(problem) <= 49 for problem in monitor)
