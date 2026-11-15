"""Refill engine over every T-arch loop feature (CPU tiny model).

The refill engine re-prefills ``prompt + generated`` from scratch every round,
so each round must install the same per-pass adapter policy and bridge as an
uninterrupted forward.  These tests compare it with the padded engine for
configured consumer blocks, all bridge modes, N=1/N=2 and ``loop_only``
adapters with non-zero weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    GenerationConfig,
)

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.serve.generate import Sampling, generate_batch
from gemmaloop.serve.refill import generate_batch_refill
from tests.model.tiny import tiny_config


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + int(token) % 26) for token in ids)


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(
            max_soft_tokens=280, image_seq_length=280
        )
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


def _key(output):
    return (
        output.text,
        output.reasoning,
        output.completion_tokens,
        output.finish_reason,
        output.prompt_tokens,
    )


def _build_lm(block, bridge, n_loops, *, policy=None):
    config = tiny_config()
    torch.manual_seed(0)
    lm = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=config, vision_config=None, audio_config=None)
    )
    apply_loops(
        lm.model.language_model,
        make_layer_plan(config.to_dict(), block=block),
        LoopConfig(
            n_loops=1,
            bridge=bridge,
            block=block,
            adapter_policy=policy,
            lora_target_layers="loop_block" if policy else None,
        ),
    )
    set_n_loops(lm.model.language_model, n_loops)
    lm.generation_config = GenerationConfig(
        do_sample=True, eos_token_id=[1], pad_token_id=0
    )
    lm.eval()
    return lm


def _randomise_adapters(model, seed=123):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.1
                )
    return model


@pytest.mark.parametrize("block", [(5, 7), (5, 9)])
@pytest.mark.parametrize("bridge", ["scalar", "none", "damped", "gch"])
@pytest.mark.parametrize("n_loops", [1, 2])
def test_refill_matches_padded_for_every_block_and_bridge(block, bridge, n_loops):
    model = _build_lm(block, bridge, n_loops)
    processor = _Processor()
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=6, temperature=0.0)

    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2, seed=0
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=3,
        seed=0,
    )
    assert [_key(output) for output in refill] == [
        _key(output) for output in padded
    ]
    # Refill never changes the server-fixed loop count.
    assert model.model.language_model._gemmaloop_n_loops == n_loops


def test_refill_matches_padded_with_nonzero_gch_bridge_and_reprefill():
    model = _build_lm((5, 7), "gch", 2)
    with torch.no_grad():
        model.model.language_model.loop_bridge_g.uniform_(0.05, 0.2)
    processor = _Processor()
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2, seed=0
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
        seed=0,
    )
    assert [_key(output) for output in refill] == [
        _key(output) for output in padded
    ]


def test_refill_supports_loop_only_adapters_across_rounds():
    block, bridge = (5, 7), "gch"
    base = _build_lm(block, bridge, 2, policy="loop_only")
    model = _randomise_adapters(
        get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
                target_modules=[
                    "q_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    )
    processor = _Processor()
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
        _conversation("hello there"),
    ]
    sampling = Sampling(max_new_tokens=8, temperature=0.0)

    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2, seed=0
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
        seed=0,
    )

    # Non-zero loop_only adapters change the outputs relative to the unadapted
    # model, so equality is over the actual per-pass surface, not a no-op.
    unadapted = _build_lm(block, bridge, 2)
    plain = generate_batch(
        unadapted, processor, conversations, sampling, max_batch=2, seed=0
    )
    assert [_key(output) for output in refill] == [
        _key(output) for output in padded
    ]
    assert any(
        refill[index].text != plain[index].text or refill[index].completion_tokens
        != plain[index].completion_tokens
        for index in range(len(conversations))
    )
