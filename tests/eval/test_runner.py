"""Tests for the resumable benchmark runner.

Every test uses a fake generator and, where the benchmark module is not the
subject, a fake adapter: no model is loaded and no dataset is downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gemmaloop.eval import benchmarks, runner
from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.eval.runner import read_run_config, run_bench
from gemmaloop.model.load import ModelSpec
from gemmaloop.model.looped import LoopConfig
from gemmaloop.serve.generate import GenOut, Sampling

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "prereg_runner.yaml"


@dataclass
class FakeItem:
    item_id: str
    prompt: str
    answer: str | None = None
    meta: dict = field(default_factory=dict)


class FakeAdapter:
    """Minimal adapter surface consumed by ``run_bench``."""

    def build_messages(self, item: FakeItem) -> list[dict]:
        return [{"role": "user", "content": item.prompt}]

    def score(self, item: FakeItem, completion: str) -> bool:
        return completion.strip() == (item.answer or "")


class FakeGen:
    """Counts every generator call and returns the prompt as the completion."""

    def __init__(self, fail_after: int | None = None, finish_reason: str = "stop"):
        self.calls: list[list[str]] = []
        self.fail_after = fail_after
        self.finish_reason = finish_reason

    def __call__(
        self,
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
    ):
        self.calls.append([conversation[0]["content"] for conversation in conversations])
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("simulated PC freeze")
        return [
            GenOut(
                text=conversation[0]["content"],
                reasoning="",
                tool_calls=[],
                completion_tokens=3,
                finish_reason=self.finish_reason,
                invalid_tool_call=False,
            )
            for conversation in conversations
        ]


def _spec() -> ModelSpec:
    return ModelSpec(size="e2b", quant="bf16")


def _partial(path: Path) -> Path:
    return path.with_suffix(".partial.jsonl")


def _all_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in _partial(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _records(path: Path) -> list[dict]:
    return [
        record
        for record in _all_lines(path)
        if record.get("type") != "run_fingerprint"
    ]


def _fingerprint(path: Path) -> dict:
    for record in _all_lines(path):
        if record.get("type") == "run_fingerprint":
            return record["fingerprint"]
    raise AssertionError("partial file has no fingerprint header")


# ---------------------------------------------------------------------------
# Run config reader
# ---------------------------------------------------------------------------
def test_read_run_config_fixture():
    config = read_run_config(FIXTURE)

    assert isinstance(config.sampling, Sampling)
    assert config.sampling.temperature == pytest.approx(0.7)
    assert config.sampling.top_p == pytest.approx(0.8)
    assert config.sampling.top_k == 32
    assert config.sampling.max_new_tokens == 4096
    assert config.sampling.enable_thinking is True

    assert config.tag == "prereg-v1"
    assert config.benches["aime"].limit == 3
    assert config.benches["aime"].samples == 2
    assert config.benches["aime"].seed == 7
    assert config.benches["aime"].image_soft_tokens == 280  # default
    assert config.benches["mathvision"].image_soft_tokens == 1120


def test_read_run_config_rejects_bad_image_tokens(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "sampling: {temperature: 1.0, top_p: 0.95, top_k: 64, "
        "max_new_tokens: 16, enable_thinking: true}\n"
        "benches:\n  mathvision: {limit: 1, samples: 1, seed: 0, "
        "image_soft_tokens: 999}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_run_config(path)


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------
def test_resume_skips_done_items(tmp_path):
    items = [
        FakeItem("q0", "q0", "q0"),
        FakeItem("q1", "q1", "q1"),
        FakeItem("q2", "q2", "q2"),
        FakeItem("q3", "q3", "q3"),
    ]
    sampling = Sampling()
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    first = FakeGen(fail_after=3)
    with pytest.raises(RuntimeError):
        run_bench(
            _spec(),
            "L",
            "aime",
            items,
            samples=2,
            sampling=sampling,
            out_path=out_path,
            gen_fn=first,
            adapter=adapter,
            batch=1,
        )

    assert len(_records(out_path)) == 3

    second = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=sampling,
        out_path=out_path,
        gen_fn=second,
        adapter=adapter,
        batch=1,
    )

    # Only the missing pairs were generated.
    assert sum(len(call) for call in second.calls) == len(items) * 2 - 3
    pairs = [(item.item_id, item.sample_idx) for item in result.items]
    assert len(pairs) == len(items) * 2
    assert len(set(pairs)) == len(pairs)
    assert len(_records(out_path)) == len(pairs)


def test_resume_does_not_rewrite_completed_run(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=FakeGen(),
        adapter=adapter,
    )
    before = _partial(out_path).read_bytes()

    gen = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=gen,
        adapter=adapter,
    )

    assert gen.calls == []
    assert _partial(out_path).read_bytes() == before
    assert len(result.items) == 2


def test_resume_refuses_fingerprint_mismatch(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(temperature=0.7),
        out_path=out_path,
        gen_fn=FakeGen(),
        adapter=adapter,
    )

    with pytest.raises(runner.PartialProvenanceError, match="different run configuration"):
        run_bench(
            _spec(),
            "L",
            "aime",
            items,
            samples=1,
            sampling=Sampling(temperature=1.0),
            out_path=out_path,
            gen_fn=FakeGen(),
            adapter=adapter,
        )


def test_fresh_rotates_partial_and_regenerates(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(temperature=0.7),
        out_path=out_path,
        gen_fn=FakeGen(),
        adapter=adapter,
    )
    old_header = _fingerprint(out_path)

    gen = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(temperature=1.0),
        out_path=out_path,
        gen_fn=gen,
        adapter=adapter,
        fresh=True,
    )

    assert len(gen.calls) == 1  # all pairs regenerated, not resumed
    assert result.items[0].correct is True
    new_header = _fingerprint(out_path)
    assert new_header["sampling"]["temperature"] == pytest.approx(1.0)
    assert new_header != old_header
    assert result.metadata["fingerprint"] == new_header
    backups = list(tmp_path.glob("aime.partial.jsonl.*.bak"))
    assert len(backups) == 1


def test_torn_last_partial_line_is_truncated_and_recovered(tmp_path):
    items = [FakeItem("q0", "q0", "q0"), FakeItem("q1", "q1", "q1")]
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    # First run freezes after q0, leaving a valid header and one record.
    with pytest.raises(RuntimeError):
        run_bench(
            _spec(),
            "L",
            "aime",
            items,
            samples=1,
            sampling=Sampling(),
            out_path=out_path,
            gen_fn=FakeGen(fail_after=1),
            adapter=adapter,
            batch=1,
        )

    # Simulate a freeze in the middle of writing q1's record.
    with _partial(out_path).open("a", encoding="utf-8", newline="") as handle:
        handle.write('{"item_id": "q1", "sample_idx": 0, "completion": "torn')

    first_resume = FakeGen()
    run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=first_resume,
        adapter=adapter,
        batch=1,
    )
    assert sum(len(call) for call in first_resume.calls) == 1

    second_resume = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=second_resume,
        adapter=adapter,
        batch=1,
    )
    assert second_resume.calls == []

    pairs = [(item.item_id, item.sample_idx) for item in result.items]
    assert sorted(pairs) == [("q0", 0), ("q1", 0)]
    assert len(_records(out_path)) == 2
    assert all("torn" not in record["completion"] for record in _records(out_path))


def test_finalize_rescores_instead_of_trusting_cached_correct(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    out_path = tmp_path / "aime.json"
    adapter = FakeAdapter()

    first = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=FakeGen(),
        adapter=adapter,
    )
    assert first.items[0].correct is True

    # Corrupt only the diagnostic value stored at generation time.
    lines = _partial(out_path).read_text(encoding="utf-8").splitlines()
    tampered = []
    for line in lines:
        record = json.loads(line)
        if record.get("type") != "run_fingerprint":
            record["correct"] = False
        tampered.append(json.dumps(record))
    _partial(out_path).write_text("\n".join(tampered) + "\n", encoding="utf-8")

    gen = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=out_path,
        gen_fn=gen,
        adapter=adapter,
    )
    assert gen.calls == []
    assert result.items[0].correct is True  # rescored from completion


# ---------------------------------------------------------------------------
# Truncation and scoring
# ---------------------------------------------------------------------------
def test_length_finish_flagged(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(max_new_tokens=1),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(finish_reason="length"),
        adapter=FakeAdapter(),
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item == ItemResult(
        item_id="q0",
        sample_idx=0,
        correct=False,
        completion_tokens=3,
        finish_reason="length",
    )


def test_lcb_accumulates_code_and_uses_official_grades(tmp_path, monkeypatch):
    items = [
        benchmarks.lcb.Item("q1", "q1", None, {}),
        benchmarks.lcb.Item("q2", "q2", None, {}),
    ]
    captured: dict[str, list[dict]] = {}

    def fake_evaluator(outputs, release="release_v6"):
        captured["outputs"] = outputs
        return {entry["question_id"]: [True, False] for entry in outputs}

    monkeypatch.setattr(runner.lcb, "run_official_evaluator", fake_evaluator)

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text=f"{conversation[0]['content']}\n```python\nprint(1)\n```",
                reasoning="",
                tool_calls=[],
                completion_tokens=10,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for conversation in conversations
        ]

    result = run_bench(
        _spec(),
        "L",
        "lcb",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=tmp_path / "lcb.json",
        gen_fn=gen,
    )

    outputs = {entry["question_id"]: entry["code_list"] for entry in captured["outputs"]}
    assert outputs == {
        "q1": ["print(1)", "print(1)"],
        "q2": ["print(1)", "print(1)"],
    }
    correct = {(item.item_id, item.sample_idx): item.correct for item in result.items}
    assert correct == {
        ("q1", 0): True,
        ("q1", 1): False,
        ("q2", 0): True,
        ("q2", 1): False,
    }


def test_lcb_items_store_post_cut_and_declared_timeout(tmp_path, monkeypatch):
    items = [
        benchmarks.lcb.Item("q1", "q1", None, {"post_cut": True}),
        benchmarks.lcb.Item("q2", "q2", None, {"post_cut": False}),
    ]

    def fake_evaluator(outputs, release="release_v6"):
        return {entry["question_id"]: [True] for entry in outputs}

    monkeypatch.setattr(runner.lcb, "run_official_evaluator", fake_evaluator)

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text="""```py
print(1)
```""",
                reasoning="",
                tool_calls=[],
                completion_tokens=3,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for _ in conversations
        ]

    result = run_bench(
        _spec(),
        "L",
        "lcb",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "lcb.json",
        gen_fn=gen,
    )

    assert [(item.item_id, item.meta) for item in result.items] == [
        ("q1", {"post_cut": True}),
        ("q2", {"post_cut": False}),
    ]
    assert result.metadata["lcb_timeout_s"] == 12
    assert result.metadata["lcb_release"] == "release_v6"


def test_multiple_choice_run_saves_extracted_answer_status_reasoning_and_rate(
    tmp_path,
):
    items = [
        benchmarks.gpqa.Item("q1", "p", "B", {}),
        benchmarks.gpqa.Item("q2", "p", "C", {}),
    ]
    texts = iter(["Answer: **B**", "I am not sure."])

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text=next(texts),
                reasoning=f"reasoning for {position}",
                tool_calls=[],
                completion_tokens=4,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for position, _ in enumerate(conversations)
        ]

    result = run_bench(
        _spec(),
        "L",
        "gpqa",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "gpqa.json",
        gen_fn=gen,
    )

    first, second = result.items
    assert first.extracted_answer == "B" and first.parse_status == "ok"
    assert second.extracted_answer is None and second.parse_status == "unparsed"
    assert first.reasoning == "reasoning for 0"
    assert result.metadata["parse_failures"] == 1
    assert result.metadata["parse_failure_rate"] == pytest.approx(0.5)
    assert result.metadata["parse_status_counts"] == {"ok": 1, "unparsed": 1}


def test_mathvision_uses_official_scorer_when_available(tmp_path, monkeypatch):
    items = [
        benchmarks.mathvision.Item(
            "m1",
            "what is 1?",
            "1",
            {"options": [], "subject": "algebra", "level": 1, "image": None},
        )
    ]
    captured: dict[str, list[dict]] = {}

    def official(outputs):
        captured["outputs"] = outputs
        return {entry["item_id"]: True for entry in outputs}

    monkeypatch.setattr(runner.mathvision, "run_official_scorer", official)

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text="0",  # the local scorer would mark this wrong
                reasoning="",
                tool_calls=[],
                completion_tokens=1,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for _ in conversations
        ]

    result = run_bench(
        _spec(),
        "L",
        "mathvision",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "mathvision.json",
        gen_fn=gen,
    )

    assert captured["outputs"][0]["answer"] == "1"
    assert result.items[0].correct is True
    assert result.metadata["scoring"] == "official"


def test_mathvision_missing_official_grade_raises(tmp_path, monkeypatch):
    items = [
        benchmarks.mathvision.Item(
            "m1",
            "what is 1?",
            "1",
            {"options": [], "subject": "algebra", "level": 1, "image": None},
        )
    ]

    monkeypatch.setattr(
        runner.mathvision, "run_official_scorer", lambda outputs: {}
    )

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text="1",
                reasoning="",
                tool_calls=[],
                completion_tokens=1,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for _ in conversations
        ]

    with pytest.raises(RuntimeError, match="no grade"):
        run_bench(
            _spec(),
            "L",
            "mathvision",
            items,
            samples=1,
            sampling=Sampling(),
            out_path=tmp_path / "mathvision.json",
            gen_fn=gen,
        )


def test_mathvision_missing_official_venv_marks_non_official(tmp_path, monkeypatch):
    items = [
        benchmarks.mathvision.Item(
            "m1",
            "what is 1?",
            "1",
            {"options": [], "subject": "algebra", "level": 1, "image": None},
        )
    ]

    def missing(*args, **kwargs):
        raise FileNotFoundError("no .venv-mathv")

    monkeypatch.setattr(runner.mathvision, "run_official_scorer", missing)

    def gen(model, processor, conversations, sampling, **kwargs):
        return [
            GenOut(
                text="1",
                reasoning="",
                tool_calls=[],
                completion_tokens=1,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for _ in conversations
        ]

    result = run_bench(
        _spec(),
        "L",
        "mathvision",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "mathvision.json",
        gen_fn=gen,
    )

    assert result.items[0].correct is True
    assert result.metadata["scoring"] == "non-official"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_uses_prereg_sampling(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    captured: dict[str, object] = {}

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        captured["spec"] = spec
        captured["sampling"] = sampling
        captured["samples"] = samples
        captured["fresh"] = kwargs.get("fresh")
        captured["out_path"] = out_path
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", lambda bench, limit, seed: [])
    cap = {"total_gib": 12.0, "free_at_start_gib": 11.0, "cap_gib": 9.0}
    cap_calls: list[tuple] = []

    def fake_cap(*args, **kwargs):
        cap_calls.append((args, kwargs))
        return cap

    monkeypatch.setattr(cli, "cap_cuda_allocator", fake_cap)

    cli.main(
        [
            "--variant",
            "L",
            "--size",
            "e2b",
            "--quant",
            "bf16",
            "--loops",
            "2",
            "--bridge",
            "scalar",
            "--bench",
            "aime",
            "--prereg",
            str(FIXTURE),
            "--out-dir",
            str(tmp_path),
            "--fresh",
        ]
    )

    sampling = captured["sampling"]
    assert captured["fresh"] is True
    assert "prereg-v1" in Path(captured["out_path"]).name
    assert isinstance(sampling, Sampling)
    assert sampling.temperature == pytest.approx(0.7)
    written = RunResult.load(Path(captured["out_path"]))
    assert written.metadata["vram_cap"] == cap
    # Exactly one cap, and the recorded value is the effective one.
    assert len(cap_calls) == 1
    assert cap_calls[0][0] == (0.92, 0.5)
    assert sampling.top_p == pytest.approx(0.8)
    assert sampling.top_k == 32
    assert sampling.max_new_tokens == 4096
    assert captured["samples"] == 2

    spec = captured["spec"]
    assert spec.loop is not None
    assert spec.loop.n_loops == 2
    assert spec.loop.bridge == "scalar"


def test_cli_skips_gated_gpqa(tmp_path, monkeypatch, capsys):
    import scripts.run_eval as cli

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    def fake_items(bench, limit, seed):
        if bench == "gpqa":
            raise PermissionError("401 Client Error: gated dataset")
        return []

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", fake_items)

    cli.main(
        [
            "--variant",
            "O",
            "--size",
            "e2b",
            "--bench",
            "gpqa,aime",
            "--prereg",
            str(FIXTURE),
            "--out-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert "skipped: gated dataset" in (captured.out + captured.err).lower()
    assert "summary:" in captured.out.lower()
    assert "run=aime" in captured.out.lower()
    assert "skipped=gpqa" in captured.out.lower()
    # The other benchmark still ran.
    assert any("aime" in path.name for path in tmp_path.rglob("*.json"))


def test_cli_isolates_bench_failures_and_prints_summary(
    tmp_path, monkeypatch, capsys
):
    import scripts.run_eval as cli

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        if bench == "lcb":
            raise RuntimeError("simulated grader explosion")
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", lambda bench, limit, seed: [])

    cli.main(
        [
            "--variant",
            "O",
            "--size",
            "e2b",
            "--bench",
            "aime,lcb",
            "--prereg",
            str(FIXTURE),
            "--out-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert "lcb: failed: simulated grader explosion" in captured.err
    assert "summary:" in captured.out.lower()
    assert "run=aime" in captured.out.lower()
    assert "failed=lcb" in captured.out.lower()
    # AIME still produced its result.
    assert any("aime" in path.name for path in tmp_path.rglob("*.json"))


def test_cli_output_path_changes_with_sampling(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    paths: list[Path] = []

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        paths.append(Path(out_path))
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", lambda bench, limit, seed: [])

    base = [
        "--variant",
        "O",
        "--size",
        "e2b",
        "--bench",
        "aime",
        "--out-dir",
        str(tmp_path),
    ]
    cli.main([*base, "--prereg", str(FIXTURE)])

    modified = tmp_path / "prereg_changed.yaml"
    modified.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "temperature: 0.7", "temperature: 0.9"
        ),
        encoding="utf-8",
    )
    cli.main([*base, "--prereg", str(modified)])

    assert len(paths) == 2
    assert paths[0].name != paths[1].name
    assert "prereg-v1" in paths[0].name


def test_training_seed_defaults_to_zero_without_adapters(tmp_path):
    from gemmaloop.eval.runner import training_seed_from_adapters

    assert training_seed_from_adapters(()) == 0

    result = run_bench(
        _spec(),
        "R",
        "aime",
        [FakeItem("q1", "x", "x")],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "r.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        image_soft_tokens=280,
    )
    assert result.seed == 0


def test_run_bench_records_training_seed_of_last_adapter(tmp_path):
    first = tmp_path / "adapter-first"
    second = tmp_path / "adapter-second"
    first.mkdir()
    second.mkdir()
    (first / "run.json").write_text(json.dumps({"seed": 2}), encoding="utf-8")
    (second / "run.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")

    spec = ModelSpec(
        size="e2b", quant="bf16", adapters=(str(first), str(second))
    )
    result = run_bench(
        spec,
        "L",
        "aime",
        [FakeItem("q1", "x", "x")],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "l.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        image_soft_tokens=280,
    )

    assert result.seed == 7
    assert RunResult.load(tmp_path / "l.json").seed == 7


def test_cli_exits_nonzero_when_a_benchmark_fails(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        if bench == "lcb":
            raise RuntimeError("simulated grader explosion")
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", lambda bench, limit, seed: [])

    exit_code = cli.main(
        [
            "--variant",
            "O",
            "--size",
            "e2b",
            "--bench",
            "aime,lcb",
            "--prereg",
            str(FIXTURE),
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1


def test_cli_allow_skip_only_tolerates_exploratory_skips(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    exploratory_fixture = tmp_path / "prereg_exploratory.yaml"
    exploratory_fixture.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "  gpqa:\n    limit: 2",
            "  gpqa:\n    exploratory: true\n    limit: 2",
        ),
        encoding="utf-8",
    )

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        result = RunResult(
            variant=variant,
            size=spec.size,
            n_loops=spec.loop.n_loops if spec.loop else 0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[],
        )
        result.save(out_path)
        return result

    def fake_items(bench, limit, seed):
        if bench == "gpqa":
            raise PermissionError("401 Client Error: gated dataset")
        return []

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", fake_items)

    base = [
        "--variant",
        "O",
        "--size",
        "e2b",
        "--bench",
        "gpqa,aime",
        "--prereg",
        str(exploratory_fixture),
        "--out-dir",
        str(tmp_path),
    ]
    assert cli.main(base) == 1
    assert cli.main([*base, "--allow-skip"]) == 0


def test_cli_allow_skip_does_not_forgive_a_failure(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    exploratory_fixture = tmp_path / "prereg_exploratory.yaml"
    exploratory_fixture.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "  lcb:\n    limit: 2",
            "  lcb:\n    exploratory: true\n    limit: 2",
        ),
        encoding="utf-8",
    )

    def fake_run_bench(spec, variant, bench, items, samples, sampling, out_path, **kwargs):
        raise RuntimeError("simulated grader explosion")

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli, "load_model", lambda spec: (object(), object(), None))
    monkeypatch.setattr(cli, "load_items_for_bench", lambda bench, limit, seed: [])

    exit_code = cli.main(
        [
            "--variant",
            "O",
            "--size",
            "e2b",
            "--bench",
            "lcb",
            "--prereg",
            str(exploratory_fixture),
            "--out-dir",
            str(tmp_path),
            "--allow-skip",
        ]
    )

    assert exit_code == 1


def test_training_seed_requires_run_json_on_the_evaluated_adapter(tmp_path):
    from gemmaloop.eval.runner import training_seed_from_adapters

    first = tmp_path / "first"
    last = tmp_path / "last"
    first.mkdir()
    last.mkdir()
    (first / "run.json").write_text(json.dumps({"seed": 2}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="training seed unknown"):
        training_seed_from_adapters((str(first), str(last)))


class _BatchSensitiveGen(FakeGen):
    """Fails (OOM) above a batch size, then returns one completion per item."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def __call__(
        self,
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
    ):
        self.calls.append([conversation[0]["content"] for conversation in conversations])
        if len(conversations) > self.limit:
            raise torch.cuda.OutOfMemoryError("simulated batch OOM")
        return [
            GenOut(
                text=conversation[0]["content"],
                reasoning="",
                tool_calls=[],
                completion_tokens=3,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for conversation in conversations
        ]


def test_run_bench_halves_batch_on_oom_and_keeps_finished_items(tmp_path):
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(4)]
    gen = _BatchSensitiveGen(limit=2)
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=gen,
        adapter=FakeAdapter(),
        batch=4,
    )
    assert [len(call) for call in gen.calls] == [4, 2, 2]
    assert result.metadata["batch"] == 4
    assert result.metadata["batch_after_fallback"] == 2
    assert result.metadata["fingerprint"]["batch"] == 4
    assert result.metadata["fingerprint"]["batch_policy"]
    assert len(result.items) == 4


def test_run_bench_halves_batch_on_detected_spill(tmp_path, monkeypatch):
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(3)]
    flags = iter([True, False, False])
    monkeypatch.setattr(runner, "_peak_spilled", lambda *a, **k: next(flags, False))
    gen = FakeGen()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=gen,
        adapter=FakeAdapter(),
        batch=4,
    )
    assert [len(call) for call in gen.calls] == [3, 2, 1]
    assert result.metadata["batch_after_fallback"] == 2
    assert len(result.items) == 3


def test_run_bench_default_batch_is_size_aware_and_recorded(tmp_path):
    from gemmaloop.serve.generate import max_batch_for

    items = [FakeItem("q0", "q0", "q0")]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(max_new_tokens=32768),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        batch=None,
    )
    expected = max_batch_for(32768, "e2b")
    assert result.metadata["batch"] == expected
    assert result.metadata["fingerprint"]["batch"] == expected
    assert result.metadata["fingerprint"]["batch_policy"] == "t5b-v1"


# ---------------------------------------------------------------------------
# Task 5c: generation engine selection
# ---------------------------------------------------------------------------
def test_run_bench_records_engine_and_rejects_unknown(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        batch=1,
        engine="refill",
    )
    assert result.metadata["engine"] == "refill"
    assert result.metadata["fingerprint"]["engine"] == "refill"

    with pytest.raises(ValueError):
        run_bench(
            _spec(),
            "L",
            "aime",
            items,
            samples=1,
            sampling=Sampling(),
            out_path=tmp_path / "bad.json",
            gen_fn=FakeGen(),
            adapter=FakeAdapter(),
            batch=1,
            engine="continuous",
        )


def test_run_bench_refill_engine_gets_the_whole_benchmark_queue(tmp_path, monkeypatch):
    calls: list[tuple[list[str], int | None, list[str] | None]] = []

    def fake_refill(
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
    ):
        calls.append(
            (
                [conversation[0]["content"] for conversation in conversations],
                max_batch,
                request_ids,
            )
        )
        return [
            GenOut(
                text=conversation[0]["content"],
                reasoning="",
                tool_calls=[],
                completion_tokens=2,
                finish_reason="stop",
                invalid_tool_call=False,
            )
            for conversation in conversations
        ]

    monkeypatch.setattr(runner, "generate_batch_refill", fake_refill)
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(4)]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        adapter=FakeAdapter(),
        batch=2,
        model=object(),
        processor=object(),
        engine="refill",
    )

    # One call with every item x sample (8 requests), capped at 2 slots.
    assert len(calls) == 1
    contents, max_batch, request_ids = calls[0]
    assert contents == [f"q{index}" for index in range(4)] * 2
    assert max_batch == 2
    assert request_ids == [
        f"q{index}:{sample_idx}"
        for sample_idx in range(2)
        for index in range(4)
    ]
    assert len(result.items) == 8


def _loop_spec(n_loops: int) -> ModelSpec:
    return ModelSpec(
        size="e2b", quant="bf16", loop=LoopConfig(n_loops=n_loops)
    )


def _loaded_model(n_loops: int) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(_gemmaloop_n_loops=n_loops))


def test_run_bench_records_executed_n(tmp_path):
    result = run_bench(
        _loop_spec(2),
        "L",
        "aime",
        [FakeItem("q0", "q0", "q0")],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        model=_loaded_model(2),
        processor=object(),
    )
    assert result.n_loops == 2
    assert result.metadata["executed_n_loops"] == 2


def test_run_bench_rejects_model_running_a_different_n(tmp_path):
    with pytest.raises(RuntimeError, match="requested N=2.*executes N=4"):
        run_bench(
            _loop_spec(2),
            "L",
            "aime",
            [FakeItem("q0", "q0", "q0")],
            samples=1,
            sampling=Sampling(),
            out_path=tmp_path / "aime.json",
            gen_fn=FakeGen(),
            adapter=FakeAdapter(),
            model=_loaded_model(4),
            processor=object(),
        )


# ---------------------------------------------------------------------------
# Fix round 1: production wrapper walks, CLI metadata and empty runs
# ---------------------------------------------------------------------------
def _write_loop_state(directory, *, bridge="scalar", ple_mode="shared"):
    """Create only the loop-state sidecar (no LoRA weights needed)."""
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, save_loop_state
    from tests.model.tiny import tiny_config, tiny_text_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = tiny_text_model(seed=0)
    apply_loops(
        model,
        make_layer_plan(tiny_config().to_dict()),
        LoopConfig(n_loops=1, bridge=bridge, ple_mode=ple_mode),
    )
    save_loop_state(model, directory)
    return str(directory)


def test_executed_n_loops_walks_real_peft_causal_lm():
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4ForCausalLM

    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
    from tests.model.tiny import tiny_config

    config = tiny_config()
    model = Gemma4ForCausalLM(config)
    apply_loops(
        model.model,
        make_layer_plan(config.to_dict()),
        LoopConfig(n_loops=1),
    )
    wrapped = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"]),
    )

    for n in (1, 2, 3, 4):
        set_n_loops(model.model, n)
        assert runner._executed_n_loops(wrapped) == n


def test_executed_n_loops_walks_image_text_peft_language_model():
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4VisionConfig,
    )

    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.load import language_lora_target_modules
    from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
    from tests.model.tiny import tiny_config

    config = tiny_config()
    vision_config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=64,
        max_position_embeddings=64,
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=config,
            vision_config=vision_config,
            audio_config=None,
        )
    )
    text_model = model.model.language_model
    apply_loops(
        text_model,
        make_layer_plan(config.to_dict()),
        LoopConfig(n_loops=1),
    )
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=language_lora_target_modules(),
        ),
    )

    for n in (1, 2, 3, 4):
        set_n_loops(text_model, n)
        assert runner._executed_n_loops(wrapped) == n


def test_run_eval_loop_config_inherits_adapter_bridge(tmp_path):
    import argparse
    import scripts.run_eval as cli

    adapter = _write_loop_state(tmp_path / "damped", bridge="damped")
    args = argparse.Namespace(loops=2, bridge=None, adapter=[adapter])

    loop = cli._loop_config(args)
    assert loop is not None
    assert loop.n_loops == 2
    assert loop.bridge == "damped"


def test_run_eval_loop_config_rejects_explicit_mismatch(tmp_path):
    import argparse
    import scripts.run_eval as cli

    adapter = _write_loop_state(tmp_path / "damped", bridge="damped")
    args = argparse.Namespace(loops=2, bridge="scalar", adapter=[adapter])

    with pytest.raises(ValueError, match="bridge"):
        cli._loop_config(args)


def test_run_bench_empty_items_records_preloaded_executed_n(tmp_path):
    result = run_bench(
        _loop_spec(3),
        "L",
        "aime",
        [],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "empty.json",
        gen_fn=FakeGen(),
        model=_loaded_model(3),
        processor=object(),
    )
    assert result.metadata["executed_n_loops"] == 3


def test_run_bench_records_block_bridge_policy_and_target_set(tmp_path):
    spec = ModelSpec(
        size="e2b",
        quant="bf16",
        loop=LoopConfig(
            n_loops=2,
            bridge="gch",
            block=(16, 18),
            adapter_policy="loop_only",
            lora_target_layers="loop_block_coda",
        ),
    )
    result = run_bench(
        spec,
        "L",
        "aime",
        [FakeItem("q0", "q0", "q0")],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        model=_loaded_model(2),
        processor=object(),
    )
    assert result.loop_block == (16, 18)
    assert result.bridge == "gch"
    assert result.adapter_policy == "loop_only"
    assert result.lora_target_layers == "loop_block_coda"
    assert result.metadata["fingerprint"]["loop_block"] == [16, 18]
    assert result.metadata["fingerprint"]["adapter_policy"] == "loop_only"
    loaded = RunResult.load(tmp_path / "aime.json")
    assert loaded.loop_block == (16, 18)
    assert loaded.lora_target_layers == "loop_block_coda"


def test_run_bench_resolves_adapter_geometry_before_the_fingerprint(tmp_path):
    from safetensors.torch import save_file

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file({}, adapter / "loop_state.safetensors")
    (adapter / "run.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")
    (adapter / "loop_config.json").write_text(
        json.dumps(
            {
                "n_loops_last_sampled": 1,
                "bridge": "gch",
                "bridge_init": 0.99,
                "ple_mode": "shared",
                "block": [16, 18],
                "adapter_policy": "loop_only",
                "lora_target_layers": "loop_block_coda",
                "plan": {
                    "num_layers": 35,
                    "period": 5,
                    "first_shared": 15,
                    "full_layers": [4, 9, 14, 19, 24, 29, 34],
                    "prelude": [0, 16],
                    "loop": [16, 19],
                    "coda": [19, 35],
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ModelSpec(
        size="e2b",
        quant="bf16",
        adapters=(str(adapter),),
        loop=LoopConfig(n_loops=2, bridge="gch"),
    )
    result = run_bench(
        spec,
        "L",
        "aime",
        [FakeItem("q0", "q0", "q0")],
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
        model=_loaded_model(2),
        processor=object(),
    )
    assert result.loop_block == (16, 18)
    assert result.adapter_policy == "loop_only"
    assert result.lora_target_layers == "loop_block_coda"
    assert result.metadata["fingerprint"]["loop_block"] == [16, 18]
    assert result.metadata["fingerprint"]["adapter_policy"] == "loop_only"


# ---------------------------------------------------------------------------
# Task 5d: engine parity, per-request persistence, heartbeat
# ---------------------------------------------------------------------------
def _parity_outputs(conversations: list[list[dict]]) -> list[GenOut]:
    return [
        GenOut(
            text=conversation[0]["content"],
            reasoning="",
            tool_calls=[],
            completion_tokens=2,
            finish_reason="stop",
            invalid_tool_call=False,
        )
        for conversation in conversations
    ]


@pytest.mark.parametrize("engine", ["padded", "refill"])
def test_engines_receive_the_same_seed_and_request_ids(tmp_path, monkeypatch, engine):
    """Both engines use sha256(seed:item:sample); no arm name, no seed offset."""

    calls: list[tuple[int, list[str]]] = []

    def fake(
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
        **kwargs,
    ):
        calls.append((int(seed), list(request_ids or [])))
        return _parity_outputs(conversations)

    monkeypatch.setattr(runner, "generate_batch", fake)
    monkeypatch.setattr(runner, "generate_batch_refill", fake)
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(3)]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        adapter=FakeAdapter(),
        batch=2,
        model=object(),
        processor=object(),
        engine=engine,
    )

    assert result.metadata["engine"] == engine
    assert {seed for seed, _ in calls} == {0}
    assert [
        request_id for _, request_ids in calls for request_id in request_ids
    ] == [
        f"q{index}:{sample_idx}"
        for sample_idx in range(2)
        for index in range(3)
    ]


class _PersistingRefill:
    """Calls ``on_complete`` for a prefix, then simulates a process freeze."""

    def __init__(self, kill_after: int):
        self.kill_after = kill_after
        self.request_ids: list[list[str]] = []

    def __call__(
        self,
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
        on_complete=None,
        heartbeat=None,
    ):
        pending = list(request_ids or [])
        self.request_ids.append(pending)
        if heartbeat is not None:
            heartbeat(
                {
                    "round": 1,
                    "finished": self.kill_after,
                    "active": len(pending),
                    "pending": 0,
                    "generated_tokens": self.kill_after * 2,
                }
            )
        outputs = _parity_outputs(conversations)
        for index, output in enumerate(outputs):
            if index >= self.kill_after:
                raise KeyboardInterrupt("simulated freeze")
            if on_complete is not None:
                on_complete(index, output)
        return outputs


def test_refill_persists_each_completion_and_resume_skips_them(tmp_path):
    out_path = tmp_path / "aime.json"
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(3)]

    killed = _PersistingRefill(kill_after=2)
    with pytest.raises(KeyboardInterrupt):
        run_bench(
            _spec(),
            "L",
            "aime",
            items,
            samples=2,
            sampling=Sampling(),
            out_path=out_path,
            adapter=FakeAdapter(),
            batch=3,
            model=object(),
            processor=object(),
            engine="refill",
            gen_fn=killed,
        )

    persisted = _records(out_path)
    assert {record["item_id"] for record in persisted} == {"q0", "q1"}
    heartbeat = json.loads(
        out_path.with_suffix(".heartbeat.json").read_text(encoding="utf-8")
    )
    assert heartbeat["engine"] == "refill"
    assert heartbeat["finished"] == 2

    resumed = _PersistingRefill(kill_after=99)
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=2,
        sampling=Sampling(),
        out_path=out_path,
        adapter=FakeAdapter(),
        batch=3,
        model=object(),
        processor=object(),
        engine="refill",
        gen_fn=resumed,
    )

    # Only the unfinished pairs are regenerated.
    assert resumed.request_ids == [
        ["q2:0", "q0:1", "q1:1", "q2:1"]
    ]
    assert len(result.items) == 6
    assert {record["item_id"] for record in _records(out_path)} == {
        "q0",
        "q1",
        "q2",
    }


class _OomAfterFirstRefill:
    def __init__(self):
        self.request_ids: list[list[str]] = []
        self.max_batches: list[int | None] = []

    def __call__(
        self,
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
        on_complete=None,
        heartbeat=None,
    ):
        pending = list(request_ids or [])
        self.request_ids.append(pending)
        self.max_batches.append(max_batch)
        outputs = _parity_outputs(conversations)
        if len(self.request_ids) == 1:
            if on_complete is not None:
                on_complete(0, outputs[0])
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return outputs


def test_refill_oom_halving_keeps_finished_outputs(tmp_path):
    items = [FakeItem(f"q{index}", f"q{index}", f"q{index}") for index in range(4)]
    gen = _OomAfterFirstRefill()
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        adapter=FakeAdapter(),
        batch=4,
        model=object(),
        processor=object(),
        engine="refill",
        gen_fn=gen,
    )

    # First attempt saw the whole queue; the retry excludes the completion that
    # was already persisted and runs at the halved batch.
    assert gen.request_ids[0] == ["q0:0", "q1:0", "q2:0", "q3:0"]
    assert gen.request_ids[1] == ["q1:0", "q2:0", "q3:0"]
    assert gen.max_batches == [4, 2]
    assert len(result.items) == 4
    assert len(_records(tmp_path / "aime.json")) == 4


def test_fingerprint_records_the_shared_seed_scheme_and_bumps_format(tmp_path):
    items = [FakeItem("q0", "q0", "q0")]
    result = run_bench(
        _spec(),
        "L",
        "aime",
        items,
        samples=1,
        sampling=Sampling(),
        out_path=tmp_path / "aime.json",
        gen_fn=FakeGen(),
        adapter=FakeAdapter(),
    )
    fingerprint = result.metadata["fingerprint"]
    assert fingerprint["format_version"] == 2
    assert fingerprint["seed_scheme"] == "sha256(seed:item:sample)"
