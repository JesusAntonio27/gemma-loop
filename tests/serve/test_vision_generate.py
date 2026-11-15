"""GPU smoke test: vision conversations still generate through generate_batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.serve.generate import Sampling, generate_batch

IMAGE = Path(__file__).resolve().parent.parent / "fixtures" / "images" / "two_shapes.png"


@pytest.mark.gpu
@pytest.mark.slow
def test_image_conversation_uses_the_one_at_a_time_path():
    model, processor, _ = load_model(ModelSpec(size="e2b", quant="bf16"))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(IMAGE)},
                    {"type": "text", "text": "What colors are the shapes?"},
                ],
            }
        ]
    ]
    outputs = generate_batch(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=8, temperature=0.0),
        seed=0,
    )
    assert len(outputs) == 1
    assert outputs[0].completion_tokens >= 1
    assert outputs[0].prompt_tokens > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_vision_static_cache_matches_dynamic(monkeypatch):
    import gemmaloop.serve.generate as G

    model, processor, _ = load_model(ModelSpec(size="e2b", quant="bf16"))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(IMAGE)},
                    {"type": "text", "text": "What colors are the shapes?"},
                ],
            }
        ]
    ]
    sampling = Sampling(max_new_tokens=16, temperature=0.0)
    monkeypatch.setattr(G, "DYNAMIC_CACHE_MAX_NEW_TOKENS", 10**9)
    dynamic = generate_batch(model, processor, conversations, sampling, seed=0)
    monkeypatch.setattr(G, "DYNAMIC_CACHE_MAX_NEW_TOKENS", 0)
    static = generate_batch(model, processor, conversations, sampling, seed=0)
    assert (dynamic[0].text, dynamic[0].reasoning) == (
        static[0].text,
        static[0].reasoning,
    )
    assert dynamic[0].completion_tokens == static[0].completion_tokens
