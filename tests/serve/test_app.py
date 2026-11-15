"""OpenAI-compatible FastAPI app (Task 5), driven by an injected generator."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gemmaloop.model.load import ModelSpec
from gemmaloop.serve.app import create_app
from gemmaloop.serve.generate import GenOut

IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "images" / "two_shapes.png"

TOOL_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
}


class FakeBackend:
    """Records calls; never touches a GPU, a model or a processor."""

    def __init__(self, outputs: list[GenOut] | None = None):
        self.calls: list[dict] = []
        self.loads = 0
        self.outputs = outputs

    def loader(self, spec):
        self.loads += 1
        return object(), object(), object()

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
        self.calls.append(
            {
                "conversations": conversations,
                "sampling": sampling,
                "tools": tools,
                "seed": seed,
                "image_soft_tokens": image_soft_tokens,
            }
        )
        if self.outputs is not None:
            return list(self.outputs)
        return [
            GenOut(
                text="The answer",
                reasoning="plan",
                tool_calls=[TOOL_CALL],
                completion_tokens=9,
                finish_reason="stop",
                invalid_tool_call=False,
                prompt_tokens=5,
            )
            for _ in conversations
        ]


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def fake_app_client(fake_backend):
    app = create_app(
        ModelSpec(size="e2b", quant="bf16"),
        gen_fn=fake_backend,
        loader=fake_backend.loader,
    )
    with TestClient(app) as client:
        client.backend = fake_backend
        yield client


def _request(**overrides):
    payload = {
        "model": "gemma-loop",
        "messages": [{"role": "user", "content": "hola"}],
        "max_tokens": 16,
    }
    payload.update(overrides)
    return payload


def test_non_stream_returns_tool_calls(fake_app_client):
    response = fake_app_client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    assert body["usage"]["completion_tokens"] == 9


def test_non_stream_usage_has_prompt_and_completion(fake_app_client):
    response = fake_app_client.post("/v1/chat/completions", json=_request())

    assert response.json()["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 9,
        "total_tokens": 14,
    }


def test_stream_ends_with_done(fake_app_client):
    response = fake_app_client.post(
        "/v1/chat/completions", json=_request(stream=True)
    )

    assert response.status_code == 200
    events = [event for event in response.text.split("\n\n") if event.strip()]
    assert events[-1] == "data: [DONE]"
    deltas = []
    for event in events[:-1]:
        assert event.startswith("data: ")
        deltas.append(json.loads(event[len("data: "):]))
    assert deltas[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(
        chunk["choices"][0]["delta"].get("content") == "The answer"
        for chunk in deltas
    )
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"


def test_reasoning_in_reasoning_content(fake_app_client):
    response = fake_app_client.post("/v1/chat/completions", json=_request())

    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "plan"
    assert "plan" not in message["content"]


def test_image_message_accepted(fake_app_client):
    data_url = (
        "data:image/png;base64,"
        + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    )
    payload = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ]
    )

    response = fake_app_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    captured = fake_app_client.backend.calls[0]["conversations"][0]
    parts = captured[0]["content"]
    assert parts[0]["type"] == "image"
    assert parts[0]["url"].startswith("data:image/png;base64,")


def test_list_models(fake_app_client):
    response = fake_app_client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "gemma-loop"


def test_loader_is_lazy_and_called_once(fake_app_client):
    assert fake_app_client.backend.loads == 0
    fake_app_client.post("/v1/chat/completions", json=_request())
    fake_app_client.post("/v1/chat/completions", json=_request())
    assert fake_app_client.backend.loads == 1


def test_request_sampling_is_forwarded(fake_app_client):
    fake_app_client.post(
        "/v1/chat/completions",
        json=_request(temperature=0.2, top_p=0.8, top_k=5, max_tokens=33, seed=7),
    )
    call = fake_app_client.backend.calls[0]
    assert call["sampling"].temperature == 0.2
    assert call["sampling"].top_p == 0.8
    assert call["sampling"].top_k == 5
    assert call["sampling"].max_new_tokens == 33
    assert call["seed"] == 7


def _stream_chunks(response_text: str) -> tuple[list[dict], str]:
    events = [event for event in response_text.split("\n\n") if event.strip()]
    assert events[-1] == "data: [DONE]"
    return [json.loads(event[len("data: "):]) for event in events[:-1]], events[-1]


def test_stream_usage_chunk_when_requested(fake_app_client):
    response = fake_app_client.post(
        "/v1/chat/completions",
        json=_request(stream=True, stream_options={"include_usage": True}),
    )

    assert response.status_code == 200
    chunks, last = _stream_chunks(response.text)
    assert last == "data: [DONE]"
    usage_chunks = [chunk for chunk in chunks if "usage" in chunk]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["choices"] == []
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 9,
        "total_tokens": 14,
    }


def test_stream_has_no_usage_chunk_by_default(fake_app_client):
    response = fake_app_client.post(
        "/v1/chat/completions", json=_request(stream=True)
    )

    chunks, _ = _stream_chunks(response.text)
    assert all("usage" not in chunk for chunk in chunks)


def test_invalid_tool_call_is_flagged_and_logged(caplog):
    raw = '<|tool_call>call:bash{command:<|"|>ls'
    backend = FakeBackend(
        outputs=[
            GenOut(
                text=raw,
                reasoning="",
                tool_calls=[],
                completion_tokens=4,
                finish_reason="stop",
                invalid_tool_call=True,
                prompt_tokens=3,
            )
        ]
    )
    app = create_app(
        ModelSpec(size="e2b", quant="bf16"),
        gen_fn=backend,
        loader=backend.loader,
    )

    with caplog.at_level(logging.WARNING, logger="gemmaloop.serve.app"):
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == raw
    assert message["x_invalid_tool_call"] is True
    assert any(
        "invalid tool call" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Fix round 1: CLI bridge resolution validates before loading
# ---------------------------------------------------------------------------
def _write_loop_state(directory, *, bridge="scalar"):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, save_loop_state
    from tests.model.tiny import tiny_config, tiny_text_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = tiny_text_model(seed=0)
    apply_loops(
        model,
        make_layer_plan(tiny_config().to_dict()),
        LoopConfig(n_loops=1, bridge=bridge),
    )
    save_loop_state(model, directory)
    return str(directory)


def test_build_spec_inherits_adapter_bridge(tmp_path):
    from argparse import Namespace
    from gemmaloop.serve.app import _build_spec

    adapter = _write_loop_state(tmp_path / "damped", bridge="damped")
    spec = _build_spec(
        Namespace(
            size="e2b",
            quant="bf16",
            adapter=[adapter],
            loops=2,
            bridge=None,
        )
    )
    assert spec.loop is not None
    assert spec.loop.n_loops == 2
    assert spec.loop.bridge == "damped"


def test_build_spec_rejects_explicit_mismatch(tmp_path):
    from argparse import Namespace
    from gemmaloop.serve.app import _build_spec

    adapter = _write_loop_state(tmp_path / "damped", bridge="damped")
    with pytest.raises(ValueError, match="bridge"):
        _build_spec(
            Namespace(
                size="e2b",
                quant="bf16",
                adapter=[adapter],
                loops=2,
                bridge="scalar",
            )
        )


def test_build_spec_accepts_gch_block_and_loop_only_policy():
    from argparse import Namespace
    from gemmaloop.serve.app import _build_spec

    spec = _build_spec(
        Namespace(
            size="e2b",
            quant="bf16",
            adapter=[],
            loops=2,
            bridge="gch",
            block="16-18",
            adapter_policy="loop_only",
            lora_target_layers="loop_block",
        )
    )
    assert spec.loop is not None
    assert spec.loop.n_loops == 2
    assert spec.loop.bridge == "gch"
    assert spec.loop.block == (16, 18)
    assert spec.loop.adapter_policy == "loop_only"
    assert spec.loop.lora_target_layers == "loop_block"
