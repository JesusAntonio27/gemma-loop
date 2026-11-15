"""GPU equivalence of the round-based refill engine with the padded engine.

Task 5c requires greedy, per-request token-exact output against the padded
engine on a real model: E2B, twelve prompts (short and long answers, natural
lengths up to ~4k tokens) with only four slots, so requests are admitted
mid-run next to continuing long histories; N=1 and N=2.  These tests are marked
``gpu`` and ``slow``; they are not part of the CPU suite.
"""

from __future__ import annotations

import pytest
import torch

import gemmaloop.serve.generate as G
import gemmaloop.serve.refill as R
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling

PROMPTS = [
    "Say hi in one word.",
    "What is 2+2? Answer briefly.",
    "Name three primary colors.",
    "Write a detailed explanation of how a modern CPU pipeline works.",
    "Describe the memory hierarchy of a computer and why caches help.",
    "Explain virtual memory and page faults step by step.",
    "How does a filesystem journal recover after a crash?",
    "Walk through how a web request travels from a browser to a server.",
    "Explain how branch prediction and speculative execution work.",
    "Describe how a relational database keeps transactions durable.",
    "Explain the difference between processes and threads in detail.",
    "Why is the sky blue? Give a short answer.",
]
MAX_NEW_TOKENS = 4096
ROUND_SIZE = 512
SLOTS = 4


@pytest.fixture(scope="module")
def loaded_model():
    model, processor, _ = load_model(
        ModelSpec(size="e2b", quant="bf16", loop=LoopConfig(n_loops=1))
    )
    return model, processor


def _conversation(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _prompt_texts(processor, conversations, sampling):
    rendered = [G._render(processor, conversation, sampling, None) for conversation in conversations]
    return [G._prompt_text(processor, item) for item in rendered]


def _trim(ids, pad_token_id):
    ids = [int(token) for token in ids]
    while ids and ids[-1] == pad_token_id:
        ids.pop()
    return ids


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("n_loops", [1, 2])
def test_greedy_refill_matches_padded_token_exact(loaded_model, n_loops, monkeypatch):
    model, processor = loaded_model
    set_n_loops(model.model.language_model, n_loops)
    conversations = [_conversation(prompt) for prompt in PROMPTS]
    sampling = Sampling(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    pad = G._pad_token_id(processor)

    padded_rows: list[torch.Tensor] = []
    original_generate = G._generate

    def spy_generate(model, inputs, sampling, seed, pad_token_id, **kwargs):
        ids = original_generate(model, inputs, sampling, seed, pad_token_id, **kwargs)
        padded_rows.append(ids.detach().cpu())
        return ids

    monkeypatch.setattr(G, "_generate", spy_generate)
    padded = G.generate_batch(
        model, processor, conversations, sampling, seed=0, max_batch=len(PROMPTS)
    )
    monkeypatch.setattr(G, "_generate", original_generate)

    refill_tokens: dict[str, list[int]] = {}
    original_finish = R._finish

    def spy_finish(
        processor, new_ids, prompt_text, prompt_tokens, sampling, tools, **kwargs
    ):
        refill_tokens[prompt_text] = [int(token) for token in new_ids.tolist()]
        return original_finish(
            processor, new_ids, prompt_text, prompt_tokens, sampling, tools, **kwargs
        )

    monkeypatch.setattr(R, "_finish", spy_finish)
    refill = R.generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        max_batch=SLOTS,
        round_size=ROUND_SIZE,
    )
    monkeypatch.setattr(R, "_finish", original_finish)

    assert len(padded_rows) == 1
    prompt_texts = _prompt_texts(processor, conversations, sampling)
    for index, prompt_text in enumerate(prompt_texts):
        expected = _trim(padded_rows[0][index], pad)
        assert refill_tokens[prompt_text] == expected, f"prompt {index} differs"
        assert refill[index].completion_tokens == padded[index].completion_tokens
        assert refill[index].finish_reason == padded[index].finish_reason
        assert refill[index].prompt_tokens == padded[index].prompt_tokens
