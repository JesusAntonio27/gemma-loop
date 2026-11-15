"""Greedy token-exact equivalence of the new engine with the old padded path.

Acceptance test for Task 5b: for a fixed prompt set and greedy decoding, the
integrated :func:`generate_batch` must produce exactly the same tokens as the
pre-Task-5b engine (one left-padded ``model.generate`` call, dynamic cache).
"""

from __future__ import annotations

import pytest
import torch

import gemmaloop.serve.generate as G
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling, generate_batch

PROMPTS = [
    "Say hi in one word.",
    "What is 2+2? Answer briefly.",
    "Write the first ten digits of pi.",
    "Name three primary colors.",
    "Explain gravity in one sentence.",
    "What is the capital of France?",
    "List two programming languages.",
    "Why is the sky blue?",
]
MAX_NEW_TOKENS = 256


@pytest.fixture(scope="module")
def loaded_model():
    model, processor, _ = load_model(
        ModelSpec(size="e2b", quant="bf16", loop=LoopConfig(n_loops=1))
    )
    return model, processor


def _reference_ids(model, processor, conversations, sampling, seed):
    """The pre-Task-5b padded engine, byte for byte."""
    rendered = [G._render(processor, c, sampling, None) for c in conversations]
    pad = G._pad_token_id(processor)
    batch = G._move_inputs(G._left_pad(rendered, pad), G._model_device(model))
    prompt_length = batch["input_ids"].shape[1]
    kwargs = {
        "do_sample": False,
        "max_new_tokens": sampling.max_new_tokens,
        "pad_token_id": pad,
        "eos_token_id": G._eos_token_ids(model),
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(**batch, **kwargs)
    return output[:, prompt_length:].detach().cpu()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("n_loops", [1, 2])
def test_greedy_tokens_match_old_engine(loaded_model, n_loops):
    model, processor = loaded_model
    set_n_loops(model.model.language_model, n_loops)
    conversations = [[{"role": "user", "content": prompt}] for prompt in PROMPTS]
    sampling = Sampling(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)

    reference = _reference_ids(model, processor, conversations, sampling, seed=0)

    captured: list[torch.Tensor] = []
    original = G._generate

    def spy(model, inputs, sampling, seed, pad_token_id, **kwargs):
        ids = original(model, inputs, sampling, seed, pad_token_id, **kwargs)
        captured.append(ids.detach().cpu())
        return ids

    G._generate = spy
    try:
        outputs = generate_batch(
            model, processor, conversations, sampling, seed=0, max_batch=64
        )
    finally:
        G._generate = original

    assert len(outputs) == len(PROMPTS)
    assert len(captured) == 1  # 8 prompts fit in one chunk
    new_ids = captured[0]
    assert new_ids.shape == reference.shape
    assert torch.equal(new_ids, reference), "greedy tokens differ from the old engine"
    assert [output.completion_tokens for output in outputs] == [
        int((row != G._pad_token_id(processor)).sum()) for row in reference
    ]


@pytest.mark.gpu
@pytest.mark.slow
def test_static_cache_matches_dynamic_at_six_thousand_tokens(loaded_model, monkeypatch):
    """Token-exact static-cache path (the one every >4k evaluation uses)."""
    model, processor = loaded_model
    set_n_loops(model.model.language_model, 2)
    conversations = [
        [{"role": "user", "content": "Explain how a CPU pipeline works."}],
        [{"role": "user", "content": "Describe virtual memory in detail."}],
    ]
    sampling = Sampling(max_new_tokens=6000, temperature=0.0)
    reference = _reference_ids(model, processor, conversations, sampling, seed=0)

    captured: list[torch.Tensor] = []
    original = G._generate

    def spy(model, inputs, sampling, seed, pad_token_id, **kwargs):
        ids = original(model, inputs, sampling, seed, pad_token_id, **kwargs)
        captured.append(ids.detach().cpu())
        return ids

    monkeypatch.setattr(G, "DYNAMIC_CACHE_MAX_NEW_TOKENS", 0)
    G._generate = spy
    try:
        outputs = generate_batch(
            model, processor, conversations, sampling, seed=0, max_batch=2
        )
    finally:
        G._generate = original

    assert len(outputs) == 2
    assert len(captured) == 1
    assert captured[0].shape == reference.shape
    assert torch.equal(captured[0], reference)
