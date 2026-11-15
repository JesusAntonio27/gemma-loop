"""Per-request sampling: a request's tokens must not depend on its batch."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    GenerationConfig,
)

import gemmaloop.serve.generate as G
from gemmaloop.serve.generate import Sampling, generate_batch


def _tiny_model() -> Gemma4ForConditionalGeneration:
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        num_kv_shared_layers=4,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ]
        * 2,
        sliding_window=16,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=256,
        per_layer_config={},
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=config, vision_config=None, audio_config=None)
    ).eval()
    model.generation_config = GenerationConfig(
        do_sample=True, eos_token_id=[1], pad_token_id=0
    )
    return model


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + int(token) % 26) for token in ids)


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(max_soft_tokens=280, image_seq_length=280)
        self.image_seq_length = 280

    def apply_chat_template(self, conversation, **kwargs):
        text = conversation[-1]["content"]
        ids = [2 + (ord(ch) % 200) for ch in text]
        input_ids = torch.tensor([ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }


def _conversation(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _tokens_by_request(model, processor, conversations, request_ids, max_batch):
    captured: list[torch.Tensor] = []
    original = G._generate

    def spy(*args, **kwargs):
        ids = original(*args, **kwargs)
        captured.append(ids.clone())
        return ids

    G._generate = spy
    try:
        generate_batch(
            model,
            processor,
            conversations,
            Sampling(max_new_tokens=12, temperature=1.0),
            seed=7,
            max_batch=max_batch,
            request_ids=request_ids,
        )
    finally:
        G._generate = original
    return captured


def test_same_request_gets_identical_tokens_in_different_batches():
    model = _tiny_model()
    processor = _Processor()
    requested = _conversation("alpha")
    others = [_conversation("beta"), _conversation("gamma delta")]

    alone = _tokens_by_request(model, processor, [requested], ["A"], max_batch=1)
    mixed = _tokens_by_request(
        model,
        processor,
        [others[0], requested, others[1]],
        ["B", "A", "C"],
        max_batch=3,
    )
    row = mixed[0][1]
    assert torch.equal(alone[0][0], row)


def test_gumbel_processor_rejects_missing_generators():
    processor = G._PerRequestGumbel([torch.Generator()])
    try:
        processor(torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 4))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for missing generators")


def test_greedy_ignores_request_ids():
    model = _tiny_model()
    processor = _Processor()
    conversations = [_conversation("same")]
    first = generate_batch(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=6, temperature=0.0),
        request_ids=["A"],
    )
    second = generate_batch(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=6, temperature=0.0),
        request_ids=["Z"],
    )
    assert first[0].completion_tokens == second[0].completion_tokens
    assert first[0].text == second[0].text
