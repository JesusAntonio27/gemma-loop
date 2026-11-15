"""Heartbeat JSON is atomic and carries progress/throughput fields."""

from __future__ import annotations

import json

from gemmaloop.heartbeat import Heartbeat


def test_heartbeat_updates_and_atomic_write(tmp_path):
    path = tmp_path / "nested" / "heartbeat.json"
    heartbeat = Heartbeat(path, run="test-run")
    heartbeat.set_phase("lcb", total=10)
    heartbeat.update(
        items_done=3, tokens=300, batch=2, phase="lcb", extra={"job": "B_O"}
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run"] == "test-run"
    assert payload["phase"] == "lcb"
    assert payload["step"] == 3
    assert payload["items_done"] == 3
    assert payload["items_total"] == 10
    assert payload["generated_tokens"] == 300
    assert payload["batch"] == 2
    assert payload["job"] == "B_O"
    assert payload["tok_per_s"] > 0
    assert payload["torch_reserved_gib"] >= 0
    assert payload["ram_available_gib"] >= 0
    assert payload["ts"].endswith("+00:00")
    # No temporary file is left behind.
    assert [p.name for p in path.parent.iterdir()] == ["heartbeat.json"]


def test_heartbeat_step_override_and_token_deltas(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(path)
    heartbeat.update(tokens=100)
    heartbeat.update(tokens=50, step=7, items_total=20)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == 7
    assert payload["items_done"] == 7
    assert payload["generated_tokens"] == 150


def test_run_bench_calls_on_batch_for_each_finished_batch(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.eval.runner import run_bench
    from gemmaloop.model.load import ModelSpec
    from gemmaloop.serve.generate import GenOut, Sampling

    class Adapter:
        def build_messages(self, item):
            return [{"role": "user", "content": item.prompt}]

        def score(self, item, completion):
            return True

    class Generator:
        def __call__(self, model, processor, conversations, sampling, **kwargs):
            return [
                GenOut(
                    text="ok",
                    reasoning="",
                    tool_calls=[],
                    completion_tokens=4,
                    finish_reason="stop",
                    invalid_tool_call=False,
                )
                for _ in conversations
            ]

    items = [
        SimpleNamespace(item_id=f"q{index}", prompt="p", answer="a", meta={})
        for index in range(5)
    ]
    progress = []
    run_bench(
        ModelSpec(size="e2b", quant="bf16"),
        "O",
        "fake",
        items,
        samples=1,
        sampling=Sampling(max_new_tokens=8),
        out_path=tmp_path / "result.json",
        gen_fn=Generator(),
        batch=2,
        model=object(),
        processor=object(),
        adapter=Adapter(),
        on_batch=progress.append,
    )
    assert [call["items_done"] for call in progress] == [2, 4, 5]
    assert progress[-1]["items_total"] == 5
    assert progress[0]["tokens"] == 8


def test_background_writer_refreshes_a_long_phase(tmp_path):
    import json
    import time

    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(path, run="bg")
    heartbeat.start(0.05)
    try:
        heartbeat.update(items_done=1, items_total=10, tokens=100, phase="long")
        first = json.loads(path.read_text(encoding="utf-8"))["epoch"]
        time.sleep(0.25)
        second = json.loads(path.read_text(encoding="utf-8"))["epoch"]
        assert second > first
    finally:
        heartbeat.stop()


def test_refill_run_bench_feeds_on_batch_per_finished_request(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.eval.runner import run_bench
    from gemmaloop.model.load import ModelSpec
    from gemmaloop.serve.generate import GenOut, Sampling

    class Adapter:
        def build_messages(self, item):
            return [{"role": "user", "content": item.prompt}]

        def score(self, item, completion):
            return True

    class RefillGenerator:
        def __call__(
            self,
            model,
            processor,
            conversations,
            sampling,
            on_complete=None,
            **kwargs,
        ):
            outputs = [
                GenOut(
                    text="ok",
                    reasoning="",
                    tool_calls=[],
                    completion_tokens=5,
                    finish_reason="stop",
                    invalid_tool_call=False,
                )
                for _ in conversations
            ]
            for index, output in enumerate(outputs):
                if on_complete is not None:
                    on_complete(index, output)
            return outputs

    items = [
        SimpleNamespace(item_id=f"q{index}", prompt="p", answer="a", meta={})
        for index in range(4)
    ]
    progress = []
    run_bench(
        ModelSpec(size="e2b", quant="bf16"),
        "O",
        "fake",
        items,
        samples=1,
        sampling=Sampling(max_new_tokens=8),
        out_path=tmp_path / "result.json",
        gen_fn=RefillGenerator(),
        batch=2,
        model=object(),
        processor=object(),
        adapter=Adapter(),
        engine="refill",
        on_batch=progress.append,
    )
    assert [call["items_done"] for call in progress] == [1, 2, 3, 4]
    assert progress[-1]["items_total"] == 4
    assert progress[-1]["tokens"] == 5
