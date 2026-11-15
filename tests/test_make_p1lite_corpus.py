"""CPU-testable pieces of the P1-lite corpus generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.make_p1lite_corpus as corpus


def _row(index: int, answer: str, problem: str | None = None) -> dict:
    text = problem or f"What is {index} + 1?"
    return {
        "data_source": "math_dapo",
        "prompt": [
            {
                "role": "user",
                "content": corpus.DAPO_PREFIX
                + text
                + corpus.DAPO_SUFFIX,
            }
        ],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"index": f"id{index}"},
    }


def _candidates(count: int = 4) -> list[dict]:
    return [
        corpus.candidate_from_row(_row(index, str(index + 1))) for index in range(count)
    ]


class FakeGenerator:
    def __init__(self, correct_ids: set[str]):
        self.correct_ids = correct_ids
        self.calls = 0

    def __call__(self, model, processor, conversations, sampling, **kwargs):
        self.calls += 1
        outputs = []
        for conversation in conversations:
            content = conversation[0]["content"]
            item_id = next(
                candidate["id"]
                for candidate in _candidates()
                if candidate["prompt"] in content
            )
            value = corpus.candidate_from_row(
                _row(int(item_id.replace("id", "")), str(int(item_id.replace("id", "")) + 1))
            )["answer"]
            text = (
                f"\\boxed{{{value}}}"
                if item_id in self.correct_ids
                else "no boxed answer"
            )
            outputs.append(
                SimpleNamespace(
                    text=text,
                    reasoning="thinking trace",
                    finish_reason="stop",
                    completion_tokens=10,
                )
            )
        return outputs


def test_extract_problem_and_numeric_filter():
    row = _row(1, "34")
    assert corpus.extract_problem(corpus.prompt_text(row)) == "What is 1 + 1?"
    candidate = corpus.candidate_from_row(row)
    assert candidate["answer"] == "34"
    assert candidate["prompt"].endswith("alone inside \\boxed{}.")
    assert corpus.candidate_from_row(_row(2, "1/2")) is None


def test_select_candidates_applies_decontamination():
    class Index:
        def check(self, text: str):
            return "exact" if "2 + 1" in text else None

    chosen, stats = corpus.select_candidates(
        [_row(0, "1"), _row(1, "2"), _row(2, "3")], 10, eval_index=Index()
    )
    assert [candidate["id"] for candidate in chosen] == ["id0", "id1"]
    assert stats["contaminated"] == 1


def test_verify_completion_uses_last_boxed_after_thinking():
    assert corpus.verify_completion("34", "<|channel>thought\\boxed{1}<channel|>\\boxed{34}")
    assert corpus.verify_completion("34", "\\boxed{34.0}") is False
    assert corpus.verify_completion("34", "\\boxed{35}") is False
    assert corpus.verify_completion("34", "Answer: 34") is False


def test_build_corpus_keeps_only_verified_traces_and_hashes(tmp_path):
    candidates = _candidates(4)
    out = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    raw = tmp_path / "samples.jsonl"
    generator = FakeGenerator({"id0", "id2"})

    result = corpus.build_corpus(
        candidates,
        generate_fn=generator,
        target_verified=2,
        out_path=out,
        manifest_path=manifest,
        raw_path=raw,
        batch=2,
        extra_manifest={"source": corpus.DATASET_ID},
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["id0", "id2"]
    assert rows[0]["problem_id"] == "id0"
    assert rows[0]["cluster_id"] == "id0"
    assert rows[0]["prompt"] and rows[0]["response"]
    assert result["n_verified"] == 2
    assert result["n_sampled"] == 4
    assert result["source_revision"] == corpus.REVISION
    assert result["corpus_sha256"] == hashlib.sha256(
        out.read_bytes()
    ).hexdigest()
    assert json.loads(manifest.read_text(encoding="utf-8"))["ids"] == ["id0", "id2"]


def test_build_corpus_resumes_without_regenerating(tmp_path):
    candidates = _candidates(4)
    out = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    raw = tmp_path / "samples.jsonl"
    first = FakeGenerator({"id0", "id2"})
    corpus.build_corpus(
        candidates,
        generate_fn=first,
        target_verified=2,
        out_path=out,
        manifest_path=manifest,
        raw_path=raw,
        batch=2,
    )
    assert first.calls == 2

    class Exploding:
        def __call__(self, *args, **kwargs):
            raise AssertionError("generator should not be called on resume")

    corpus.build_corpus(
        candidates,
        generate_fn=Exploding(),
        target_verified=2,
        out_path=out,
        manifest_path=manifest,
        raw_path=raw,
        batch=2,
    )


def test_build_corpus_refuses_when_not_enough_verified(tmp_path):
    with pytest.raises(RuntimeError, match="raise --candidate-limit"):
        corpus.build_corpus(
            _candidates(4),
            generate_fn=FakeGenerator({"id0"}),
            target_verified=3,
            out_path=tmp_path / "corpus.jsonl",
            manifest_path=tmp_path / "manifest.json",
            raw_path=tmp_path / "samples.jsonl",
            batch=4,
        )


class FakeChatTokenizer:
    """Minimal chat template that keeps the thinking channel separate."""

    chat_template = "fake-gemma-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ):
        assert not tokenize
        text = "<start>"
        for message in messages:
            if message["role"] == "user":
                text += "user:" + message["content"] + "\n"
            elif message["role"] == "assistant":
                text += "assistant:"
                if message.get("reasoning"):
                    text += (
                        "<|channel>thought\n"
                        + message["reasoning"]
                        + "<channel|>"
                    )
                text += message["content"] + "<end>"
        if add_generation_prompt:
            text += "assistant:"
        return text

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(character) for character in text])


class ReasoningGenerator:
    def __call__(self, model, processor, conversations, sampling, **kwargs):
        return [
            SimpleNamespace(
                text="\\boxed{1}",
                reasoning="I should add one.",
                finish_reason="stop",
                completion_tokens=5,
            )
        ]


def test_corpus_stores_the_full_completed_assistant_turn(tmp_path):
    candidate = corpus.candidate_from_row(_row(0, "1"))
    tokenizer = FakeChatTokenizer()
    out = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    raw = tmp_path / "samples.jsonl"

    result = corpus.build_corpus(
        [candidate],
        generate_fn=ReasoningGenerator(),
        target_verified=1,
        out_path=out,
        manifest_path=manifest,
        raw_path=raw,
        tokenizer=tokenizer,
        batch=1,
    )

    row = json.loads(out.read_text(encoding="utf-8"))
    assert "<|channel>thought\nI should add one.<channel|>" in row["response"]
    assert row["response"].endswith("\\boxed{1}<end>")
    assert row["prompt"] + row["response"] == "".join(
        chr(value) for value in row["input_ids"]
    )
    reasoning_tokens = [ord(c) for c in "I should add one."]
    assert reasoning_tokens == row["input_ids"][
        row["labels_start"] + len("<|channel>thought\n") : row["labels_start"]
        + len("<|channel>thought\n")
        + len(reasoning_tokens)
    ]
    raw_row = json.loads(raw.read_text(encoding="utf-8"))
    assert raw_row["reasoning"] == "I should add one."
    assert result["response_token_distribution"]["max"] > len("\\boxed{1}")
    assert result["chat_template_sha256"]


def test_corpus_heartbeat_path_matches_report_commands(monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("GEMMALOOP_HEARTBEAT_PATH", raising=False)
    heart = corpus.default_corpus_heartbeat_path()
    assert heart.name == "corpus.json"
    assert heart.as_posix().endswith("runs/detached/corpus.json")
    report = Path(__file__).resolve().parents[1] / "docs" / "reports" / "task-trainfree-report.md"
    text = report.read_text(encoding="utf-8")
    assert r"runs\detached\corpus.json" in text


def test_build_corpus_updates_heartbeat_and_keeps_cap(tmp_path):
    from gemmaloop.heartbeat import Heartbeat

    heartbeat = Heartbeat(tmp_path / "corpus.json", run="p1lite-corpus")
    heartbeat.update(extra={"vram_cap": {"cap_gib": 9.0}})
    result = corpus.build_corpus(
        _candidates(2),
        generate_fn=FakeGenerator({"id0"}),
        target_verified=1,
        out_path=tmp_path / "corpus.jsonl",
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "samples.jsonl",
        batch=1,
        heartbeat=heartbeat,
    )
    payload = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    assert payload["generated_tokens"] > 0
    assert payload["items_done"] >= 1
    assert payload["vram_cap"] == {"cap_gib": 9.0}
    assert result["n_verified"] == 1


def test_build_corpus_refuses_pre_fix_raw_without_meta(tmp_path):
    raw = tmp_path / "samples.jsonl"
    raw.write_text(
        json.dumps(
            {
                "id": "id0",
                "completion": "\boxed{1}",
                "finish_reason": "stop",
                "completion_tokens": 4,
                "correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="full-trace format"):
        corpus.build_corpus(
            _candidates(1),
            generate_fn=FakeGenerator(set()),
            target_verified=1,
            out_path=tmp_path / "corpus.jsonl",
            manifest_path=tmp_path / "manifest.json",
            raw_path=raw,
            batch=1,
        )


def test_build_corpus_refuses_reused_records_without_reasoning(tmp_path):
    first = corpus.build_corpus(
        _candidates(1),
        generate_fn=FakeGenerator({"id0"}),
        target_verified=1,
        out_path=tmp_path / "corpus.jsonl",
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "samples.jsonl",
        batch=1,
    )
    assert first["n_verified"] == 1
    raw = tmp_path / "samples.jsonl"
    record = json.loads(raw.read_text(encoding="utf-8"))
    record.pop("reasoning")
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reasoning field"):
        corpus.build_corpus(
            _candidates(1),
            generate_fn=FakeGenerator(set()),
            target_verified=1,
            out_path=tmp_path / "corpus.jsonl",
            manifest_path=tmp_path / "manifest.json",
            raw_path=raw,
            batch=1,
        )


def test_build_corpus_refuses_meta_without_full_trace_version(tmp_path):
    raw = tmp_path / "samples.jsonl"
    record = {
        "id": "id0",
        "completion": "\boxed{1}",
        "reasoning": "thinking",
        "finish_reason": "stop",
        "completion_tokens": 4,
        "correct": True,
    }
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
    raw.with_suffix(raw.suffix + ".meta.json").write_text(
        json.dumps({"source_revision": corpus.REVISION}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not declare"):
        corpus.build_corpus(
            _candidates(1),
            generate_fn=FakeGenerator(set()),
            target_verified=1,
            out_path=tmp_path / "corpus.jsonl",
            manifest_path=tmp_path / "manifest.json",
            raw_path=raw,
            batch=1,
        )


# ---------------------------------------------------------------------------
# Registered protected-set filtering and Stage 0 local source mode
# ---------------------------------------------------------------------------
def _dapo_row(raw_id: str, answer: str, problem: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [
            {
                "role": "user",
                "content": corpus.DAPO_PREFIX + problem + corpus.DAPO_SUFFIX,
            }
        ],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"index": raw_id},
    }


def _pool_problem(problem_id: str, kind: str, answer: str | None, bucket: int) -> dict:
    return {
        "problem_id": problem_id,
        "source": problem_id.split(":", 1)[0],
        "kind": kind,
        "answer": answer,
        "bucket": bucket,
        "prompt": f"Solve {problem_id}",
        "metadata": {},
    }


def _write_pool(tmp_path: Path) -> Path:
    """A pool whose train/protected splits exercise the registered filter."""

    pool = tmp_path / "stage0" / "pool"
    pool.mkdir(parents=True)
    rows = {
        "train_math": _pool_problem("omr:train_math", "math", "3", 10),
        "train_math_2": _pool_problem("omr:train_math_2", "math", "4", 11),
        "train_code": _pool_problem("apps:train_code", "code", None, 12),
        "protected_math": _pool_problem("omr:protected_math", "math", "5", 55),
        "heldout_math": _pool_problem("omr:heldout_math", "math", "6", 90),
    }
    (pool / "problems.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows.values()), encoding="utf-8"
    )
    (pool / "split_train.jsonl").write_text(
        "".join(
            json.dumps({**rows[name], "split": "train"}) + "\n"
            for name in ("train_math", "train_math_2", "train_code")
        ),
        encoding="utf-8",
    )
    (pool / "split_pilot_dev.jsonl").write_text(
        json.dumps({**rows["protected_math"], "split": "pilot_dev"}) + "\n",
        encoding="utf-8",
    )
    (pool / "split_heldout_test.jsonl").write_text(
        json.dumps({**rows["heldout_math"], "split": "heldout_test"}) + "\n",
        encoding="utf-8",
    )
    (pool / "manifest.json").write_text(
        json.dumps({"manifest_sha256": "70053b90" + "0" * 56}), encoding="utf-8"
    )
    return pool


def _record(
    request_id: str,
    problem_id: str,
    *,
    correct: bool,
    answer: str = "",
    finish="stop",
    closed=True,
):
    reasoning = (
        "Think it through carefully and check every step of the arithmetic "
        "before writing the final boxed answer."
    )
    content = f"\\boxed{{{answer}}}" if correct and answer else (
        "wrong" if not correct else "\\boxed{unknown}"
    )
    return {
        "completion": f"<|channel>thought\n{reasoning}<channel|>{content}<turn|>",
        "content": content,
        "reasoning": reasoning,
        "problem_id": problem_id,
        "request_id": request_id,
        "sample_index": int(request_id.rsplit(":", 1)[-1]),
        "finish_reason": finish,
        "thought_closed": closed,
        "tokens": 20,
        "prompt_tokens": 10,
        "run_id": "run-123",
        "pool_manifest_sha256": "70053b90" + "0" * 56,
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
    }


def _write_local(
    tmp_path: Path,
    *,
    pool_sha: str | None = None,
    monitor_problems: int | None = None,
    extra_records: list[dict] | None = None,
) -> Path:
    local = tmp_path / "stage0" / "local"
    shards = local / "shards"
    shards.mkdir(parents=True)
    targets = {
        "train": {
            "buckets": [0, 49],
            "problems": 2500,
            "samples_per_prompt": 4,
            "token_target": 0,
            "rows_target": 0,
            "verified": False,
        }
    }
    if monitor_problems is not None:
        targets["monitor"] = {
            "buckets": [0, 49],
            "problems": int(monitor_problems),
            "samples_per_prompt": 1,
            "token_target": 0,
            "rows_target": int(monitor_problems),
            "verified": False,
        }
    (local / "run_marker.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "pool_manifest_sha256": pool_sha or ("70053b90" + "0" * 56),
                "samples_per_prompt": 4,
                "shard_size": 25,
                "targets": targets,
                "selection_order": {
                    "rule": "sha256(seed:request_id) ascending over the whole bucket",
                    "seed": 0,
                    "version": "s0lite-selection-v1-request-hash",
                },
            }
        ),
        encoding="utf-8",
    )
    committed = [
        _record("omr:train_math:0", "omr:train_math", correct=True, answer="3"),
        _record("omr:train_math_2:0", "omr:train_math_2", correct=False, answer="4"),
        _record("omr:train_math_2:1", "omr:train_math_2", correct=True, answer="4"),
        _record("omr:protected_math:0", "omr:protected_math", correct=True, answer="5"),
        _record("omr:heldout_math:0", "omr:heldout_math", correct=True, answer="6"),
        _record("apps:train_code:0", "apps:train_code", correct=True),
        _record("omr:train_math:1", "omr:train_math", correct=False, answer="3", finish="length"),
        _record("omr:train_math_2:2", "omr:train_math_2", correct=True, answer="4", closed=False),
    ]
    committed.extend(extra_records or [])
    if pool_sha:
        for row in committed:
            row["pool_manifest_sha256"] = pool_sha
    shard = shards / "shard_00000.jsonl"
    shard.write_text(
        "".join(json.dumps(row) + "\n" for row in committed), encoding="utf-8"
    )
    shard.with_name(shard.name + ".committed").write_text(
        json.dumps({"committed": True, "shard": 0}), encoding="utf-8"
    )
    uncommitted = shards / "shard_00001.jsonl"
    uncommitted.write_text(
        json.dumps(_record("omr:train_math:3", "omr:train_math", correct=True, answer="3"))
        + "\n",
        encoding="utf-8",
    )
    return local


class FakeMathGrade:
    def __init__(self, correct: bool, answer: str):
        self.correct = correct
        self.answer = answer

    def to_dict(self):
        return {"correct": self.correct, "answer": self.answer}


def fake_math_grade(prediction, answer, **_kwargs):
    boxed = f"\\boxed{{{answer}}}"
    return FakeMathGrade(boxed in prediction, answer)


def test_load_registered_protected_uses_only_the_registered_sets(tmp_path):
    pool = _write_pool(tmp_path)
    root = pool.parent
    ids_by_source, report, info = corpus.load_registered_protected(root)

    assert set(ids_by_source) == {
        "pilot_evaluation",
        "temperature_fit",
        "monitor",
        "confirmatory",
    }
    assert "omr:protected_math" in ids_by_source["pilot_evaluation"]
    assert "omr:train_math" not in {
        value for ids in ids_by_source.values() for value in ids
    }
    assert info["filter_version"] == corpus.STAGE0_FILTER_VERSION
    assert "pilot_evaluation" in info["sets"]
    assert info["sets"]["pilot_evaluation"]["files"][0]["sha256"]


def test_select_candidates_uses_the_registered_protected_sets_only(tmp_path):
    pool = _write_pool(tmp_path)
    root = pool.parent
    _, _, _ = corpus.load_registered_protected(root)

    rows = [
        _dapo_row("train-raw", "1", "train problem"),
        _dapo_row("protected-raw", "2", "protected problem"),
    ]
    # Put the protected problem in the protected split under its canonical id.
    protected_file = root / "pool" / "split_pilot_dev.jsonl"
    protected_file.write_text(
        json.dumps(
            {
                "source": "dapo",
                "problem_id": "dapo:protected",
                "metadata": {"raw_id": "protected-raw"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ids_by_source, _, _ = corpus.load_registered_protected(root)
    protected_ids = {
        value for values in ids_by_source.values() for value in values
    }
    chosen, stats = corpus.select_candidates(
        rows, 10, protected_ids=protected_ids
    )
    assert [candidate["id"] for candidate in chosen] == ["train-raw"]
    assert stats["protected_overlap"] == 1


def test_extend_keeps_train_rows_and_drops_only_protected_rows(tmp_path):
    pool = _write_pool(tmp_path)
    root = pool.parent
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"id": "omr:train_math", "source": "omr", "answer": "1"},
        {
            "id": "omr:protected_math",
            "source": "omr",
            "answer": "2",
        },
    ]
    (source / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    # protected-raw is in split_pilot_dev; train-raw is in split_train.
    (pool / "split_train.jsonl").write_text(
        json.dumps(
            {
                "source": "dapo",
                "problem_id": "dapo:train",
                "metadata": {"raw_id": "train-raw"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = corpus.build_dry_run_plan(
        source,
        pool,
        protected_root=root,
        target_verified=10,
        candidate_limit=10,
    )
    assert plan["rows_in_source"] == 2
    assert plan["rows_kept"] == 1
    assert plan["rows_removed"] == 1
    assert plan["removed"][0]["id"] == "omr:protected_math"
    assert "pilot_evaluation" in plan["removed"][0]["reason"]


def test_build_stage0_local_corpus_keeps_complete_verified_train_rows(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path)
    out = tmp_path / "out" / "corpus.jsonl"
    manifest_path = tmp_path / "out" / "manifest.json"

    manifest = corpus.build_stage0_local_corpus(
        local,
        pool,
        protected_root=pool.parent,
        out_path=out,
        manifest_path=manifest_path,
        tokenizer=FakeTrainingTokenizer(),
        screen_rows=2,
        expand_rows=5,
        math_grade=fake_math_grade,
    )

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["id"] for row in rows]
    assert "omr:train_math:0" in ids
    assert "omr:train_math_2:1" in ids
    assert "apps:train_code:0" in ids
    assert "omr:protected_math:0" not in ids
    assert "omr:heldout_math:0" not in ids
    assert "omr:train_math:3" not in ids  # uncommitted shard
    assert "omr:train_math_2:0" not in ids  # wrong math answer
    assert rows[0]["input_ids"]
    assert rows[0]["labels"][: rows[0]["labels_start"]] == [
        -100
    ] * rows[0]["labels_start"]
    from gemmaloop.stage0.postprocess import sample_messages
    from gemmaloop.train.labels import assistant_labels

    records_by_id = {
        record["request_id"]: record
        for record in corpus._read_jsonl_by_newline(
            local / "shards" / "shard_00000.jsonl"
        )
    }
    problems, _ = corpus.load_pool_index(pool)
    for row in rows:
        record = records_by_id[row["request_id"]]
        expected = assistant_labels(
            FakeTrainingTokenizer(),
            sample_messages(record, problems[row["problem_id"]]),
            8192,
        )
        assert row["input_ids"] == expected["input_ids"]
        assert row["labels"] == expected["labels"]
        first_label = next(
            index
            for index, value in enumerate(expected["labels"])
            if value != -100
        )
        assert row["labels_start"] == first_label
        assert decode_fake_training_ids(row["input_ids"]).endswith(
            record["completion"]
        )

    assert manifest["kind"] == corpus.STAGE0_LOCAL_KIND
    assert manifest["run_id"] == "run-123"
    assert manifest["shards"][0]["sha256"]
    assert manifest["filter_version"] == corpus.STAGE0_FILTER_VERSION
    assert manifest["stage0_pool_manifest_sha256"].startswith("70053b90")
    assert manifest["protected_sets"]["pilot_evaluation"]["status"] == "present"
    assert manifest["selection"]["selected_ids"] == ids
    assert manifest["selection"]["eligible_rows"] == 3


def test_stage0_local_dry_run_reports_eligible_length_strata(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path)

    plan = corpus.stage0_local_dry_run_plan(
        local,
        pool,
        protected_root=pool.parent,
        tokenizer=None,
        screen_rows=2,
        expand_rows=5,
        math_grade=fake_math_grade,
    )
    assert plan["mode"] == "stage0_local"
    assert plan["eligible_rows"] == 3
    assert plan["eligible_by_stratum"]["short"] == 3
    assert plan["rows_selected"] == 3
    assert plan["run_id"] == "run-123"
    text = corpus.format_dry_run_plan(plan)
    assert "eligible rows by length stratum" in text
    assert "short: 3" in text


def test_load_stage0_pool_does_not_split_on_unicode_line_separators(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    row = {
        "source": "apps",
        "problem_id": "apps:u2028",
        "metadata": {"raw_id": 5},
        "prompt": "line one\u2028line two is still the same JSON string",
    }
    (pool / "problems.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (pool / "manifest.json").write_text(
        json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8"
    )

    problems, info = corpus.load_pool_index(pool)
    assert "apps:u2028" in problems
    assert info["files"]["problems.jsonl"]["rows"] == 1



def test_extend_corpus_keeps_train_rows_under_the_corrected_filter(tmp_path):
    pool = _write_pool(tmp_path)
    root = pool.parent
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"id": "omr:train_math", "source": "omr", "answer": "1"},
        {"id": "omr:protected_math", "source": "omr", "answer": "2"},
    ]
    (source / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    class NewGenerator:
        def __call__(self, model, processor, conversations, sampling, **kwargs):
            return [
                SimpleNamespace(
                    text="\\boxed{7}",
                    reasoning="thinking",
                    finish_reason="stop",
                    completion_tokens=3,
                )
            ]

    manifest = corpus.extend_corpus(
        source,
        pool,
        [_dapo_row("new-raw", "7", "new problem")],
        protected_root=root,
        generate_fn=NewGenerator(),
        target_verified=2,
        out_path=tmp_path / "out" / "corpus.jsonl",
        manifest_path=tmp_path / "out" / "manifest.json",
        raw_path=tmp_path / "out" / "samples.jsonl",
        candidate_limit=10,
        batch=1,
    )
    out = [
        json.loads(line)
        for line in (tmp_path / "out" / "corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["id"] for row in out] == ["omr:train_math", "new-raw"]
    assert manifest["kept_ids"] == ["omr:train_math"]
    assert manifest["removed"][0]["id"] == "omr:protected_math"
    assert manifest["filter_version"] == corpus.STAGE0_FILTER_VERSION
    assert manifest["protected_sets"]["pilot_evaluation"]["status"] == "present"


class FakeTrainingTokenizer:
    """Character-level tokenizer with the Gemma training turn delimiters."""

    chat_template = "fake-gemma-training-template"
    unk_token_id = -1
    SPECIAL = {
        "<|turn>": 1001,
        "<turn|>": 1002,
        "<|tool_response>": 1003,
        "<tool_response|>": 1004,
    }

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL.get(token)

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
        **kwargs,
    ):
        ids = []
        for message in messages:
            ids.append(self.SPECIAL["<|turn>"])
            role = "model" if message["role"] == "assistant" else message["role"]
            ids.extend(self.encode(role))
            ids.append(ord("\n"))
            if message["role"] == "assistant":
                if message.get("reasoning"):
                    ids.extend(
                        self.encode(
                            "<|channel>thought\n"
                            + message["reasoning"]
                            + "<channel|>"
                        )
                    )
                ids.extend(self.encode(message.get("content") or ""))
            else:
                ids.extend(self.encode(message.get("content") or ""))
            ids.append(self.SPECIAL["<turn|>"])
        if add_generation_prompt:
            ids.append(self.SPECIAL["<|turn>"])
            ids.extend(self.encode("model"))
            ids.append(ord("\n"))
        return {"input_ids": ids}


def decode_fake_training_ids(ids):
    reverse = {value: key for key, value in FakeTrainingTokenizer.SPECIAL.items()}
    return "".join(
        reverse.get(int(value), chr(int(value))) for value in ids
    )


def test_stage0_local_build_refuses_a_different_pool_manifest(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, pool_sha="f" * 64)

    with pytest.raises(corpus.CorpusBuildError, match="different pool manifest"):
        corpus.stage0_local_dry_run_plan(
            local,
            pool,
            protected_root=pool.parent,
            tokenizer=None,
            screen_rows=2,
            expand_rows=5,
            math_grade=fake_math_grade,
        )


def test_monitor_target_ids_are_derived_and_excluded(tmp_path):
    from gemmaloop.stage0.local import Target, monitor_problem_ids

    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    run_marker = json.loads((local / "run_marker.json").read_text(encoding="utf-8"))
    targets = [
        Target(
            name,
            entry["buckets"][0],
            entry["buckets"][1],
            entry["problems"],
        )
        for name, entry in run_marker["targets"].items()
    ]
    problems, _ = corpus.load_pool_index(pool)
    expected = monitor_problem_ids(list(problems.values()), targets, seed=0)
    assert len(expected) == 2

    plan = corpus.stage0_local_dry_run_plan(
        local,
        pool,
        protected_root=pool.parent,
        tokenizer=None,
        screen_rows=2,
        expand_rows=5,
        math_grade=fake_math_grade,
        require_monitor_manifest=False,
    )
    assert plan["monitor"]["derived_ids"] == 2
    assert plan["monitor"]["status"] == "missing"
    assert plan["eligible_rows"] == 1
    selected_problems = {
        request_id.rsplit(":", 1)[0] for request_id in plan["selected_ids"]
    }
    assert selected_problems <= set(problems) - expected
    screen_problems = {
        request_id.rsplit(":", 1)[0] for request_id in plan["screen_ids"]
    }
    assert screen_problems <= set(problems) - expected


def test_stage0_local_fails_closed_when_the_monitor_manifest_is_missing(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    with pytest.raises(corpus.CorpusBuildError, match="MONITOR target"):
        corpus.collect_stage0_candidates(
            local,
            pool,
            protected_root=pool.parent,
            tokenizer=None,
            screen_rows=1,
            expand_rows=1,
            math_grade=fake_math_grade,
        )


def test_extension_excludes_the_runs_monitor_ids(tmp_path):
    from gemmaloop.stage0.local import Target, monitor_problem_ids

    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    run_marker = json.loads((local / "run_marker.json").read_text(encoding="utf-8"))
    targets = [
        Target(
            name,
            entry["buckets"][0],
            entry["buckets"][1],
            entry["problems"],
        )
        for name, entry in run_marker["targets"].items()
    ]
    problems, _ = corpus.load_pool_index(pool)
    monitor_ids = monitor_problem_ids(list(problems.values()), targets, seed=0)
    monitor_dir = local / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "manifest.json").write_text(
        json.dumps({"problem_ids": sorted(monitor_ids)}), encoding="utf-8"
    )
    (monitor_dir / "monitor_rows.jsonl").write_text(
        "".join(
            json.dumps({"problem_id": pid}) + "\n" for pid in sorted(monitor_ids)
        ),
        encoding="utf-8",
    )

    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"id": pid, "source": pid.split(":", 1)[0], "answer": "1"}
        for pid in sorted(problems)
        if pid in {"omr:train_math", "omr:train_math_2", "apps:train_code"}
    ]
    (source / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    plan = corpus.build_dry_run_plan(
        source,
        pool,
        protected_root=pool.parent,
        monitor_local_dir=local,
        require_monitor_manifest=True,
        target_verified=10,
        candidate_limit=10,
    )
    assert plan["monitor"]["status"] == "present"
    assert plan["monitor"]["derived_ids"] == 2
    assert plan["rows_kept"] == 1
    assert all("monitor" in entry["reason"] for entry in plan["removed"])
    monitor_files = plan["stage0_disjointness"]["protected_sets"]["monitor"]["files"]
    assert {Path(entry["path"]).name for entry in monitor_files} >= {
        "manifest.json",
        "monitor_rows.jsonl",
    }
    assert all(entry["sha256"] for entry in monitor_files)


def test_extension_build_fails_closed_without_the_monitor_manifest(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    source = tmp_path / "source"
    source.mkdir()
    (source / "corpus.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(corpus.CorpusBuildError, match="MONITOR target"):
        corpus.extend_corpus(
            source,
            pool,
            [],
            protected_root=pool.parent,
            monitor_local_dir=local,
            generate_fn=lambda *args, **kwargs: [],
            target_verified=1,
            out_path=tmp_path / "out" / "corpus.jsonl",
            manifest_path=tmp_path / "out" / "manifest.json",
            raw_path=tmp_path / "out" / "samples.jsonl",
            candidate_limit=1,
            batch=1,
        )


def test_stage0_local_build_refuses_a_short_section7_screen(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path)
    with pytest.raises(corpus.CorpusBuildError, match="section-7 screen"):
        corpus.build_stage0_local_corpus(
            local,
            pool,
            protected_root=pool.parent,
            out_path=tmp_path / "out" / "corpus.jsonl",
            manifest_path=tmp_path / "out" / "manifest.json",
            tokenizer=FakeTrainingTokenizer(),
            screen_rows=20,
            expand_rows=100,
            math_grade=fake_math_grade,
        )


def test_read_source_corpus_tolerates_u2028_inside_a_string(tmp_path):
    pool = _write_pool(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "corpus.jsonl").write_text(
        json.dumps(
            {
                "id": "omr:train_math",
                "source": "omr",
                "prompt": "first\u2028second",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = corpus.build_dry_run_plan(
        source,
        pool,
        protected_root=pool.parent,
        target_verified=10,
        candidate_limit=10,
    )
    assert plan["rows_in_source"] == 1
    assert plan["rows_kept"] == 1


def _derived_monitor_ids(pool, local):
    from gemmaloop.stage0.local import Target, monitor_problem_ids

    run_marker = json.loads((local / "run_marker.json").read_text(encoding="utf-8"))
    targets = [
        Target(
            name,
            entry["buckets"][0],
            entry["buckets"][1],
            entry["problems"],
        )
        for name, entry in run_marker["targets"].items()
    ]
    problems, _ = corpus.load_pool_index(pool)
    return monitor_problem_ids(list(problems.values()), targets, seed=0), problems


def _train_source_corpus(tmp_path, problems):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"id": pid, "source": pid.split(":", 1)[0], "answer": "1"}
        for pid in sorted(problems)
        if pid in {"omr:train_math", "omr:train_math_2", "apps:train_code"}
    ]
    (source / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return source


def test_monitor_manifest_may_be_a_kept_subset_of_the_derived_ids(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    monitor_ids, problems = _derived_monitor_ids(pool, local)
    kept_subset = {sorted(monitor_ids)[0]}
    monitor_dir = local / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "problem_ids": sorted(kept_subset),
                "rows": 1,
                "rows_target": 2,
                "problems": 2,
            }
        ),
        encoding="utf-8",
    )
    (monitor_dir / "monitor_rows.jsonl").write_text(
        json.dumps({"problem_id": sorted(kept_subset)[0]}) + "\n",
        encoding="utf-8",
    )
    source = _train_source_corpus(tmp_path, problems)

    plan = corpus.build_dry_run_plan(
        source,
        pool,
        protected_root=pool.parent,
        monitor_local_dir=local,
        require_monitor_manifest=True,
        target_verified=10,
        candidate_limit=10,
    )
    assert plan["monitor"]["status"] == "present"
    assert plan["monitor"]["derived_ids"] == 2
    assert plan["monitor"]["manifest_ids"] == 1
    assert plan["rows_kept"] == 1
    assert len(plan["removed"]) == 2


def test_monitor_manifest_ids_outside_the_derived_set_are_refused(tmp_path):
    pool = _write_pool(tmp_path)
    local = _write_local(tmp_path, monitor_problems=2)
    monitor_ids, problems = _derived_monitor_ids(pool, local)
    monitor_dir = local / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "problem_ids": sorted(monitor_ids | {"omr:foreign_problem"}),
                "rows": 3,
                "rows_target": 2,
                "problems": 2,
            }
        ),
        encoding="utf-8",
    )
    source = _train_source_corpus(tmp_path, problems)

    with pytest.raises(corpus.CorpusBuildError, match="outside the registered"):
        corpus.build_dry_run_plan(
            source,
            pool,
            protected_root=pool.parent,
            monitor_local_dir=local,
            require_monitor_manifest=True,
            target_verified=10,
            candidate_limit=10,
        )


def test_stage0_local_drops_empty_reasoning_samples_when_thinking_is_on(tmp_path):
    pool = _write_pool(tmp_path)
    empty = _record(
        "apps:train_code:1", "apps:train_code", correct=True, answer="1"
    )
    empty["reasoning"] = ""
    empty["completion"] = "<|channel>thought\n<channel|>answer<turn|>"
    empty["content"] = "answer"
    local = _write_local(tmp_path, extra_records=[empty])

    manifest = corpus.build_stage0_local_corpus(
        local,
        pool,
        protected_root=pool.parent,
        out_path=tmp_path / "out" / "corpus.jsonl",
        manifest_path=tmp_path / "out" / "manifest.json",
        tokenizer=FakeTrainingTokenizer(),
        screen_rows=2,
        expand_rows=5,
        math_grade=fake_math_grade,
    )
    ids = manifest["ids"]
    assert "apps:train_code:1" not in ids
    assert manifest["selection"]["dropped"]["empty_reasoning"] == 1
