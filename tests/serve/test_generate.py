"""Offline tests for generate_batch (fake model + fake processor, no GPU)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from gemmaloop.model.load import ModelSpec
from gemmaloop.serve.app import _State, _submit
from gemmaloop.serve.generate import GenOut, Sampling, generate_batch

# Decoding table for the fake tokenizer.  pad=0 and eos=1 must decode to their
# special-token spellings so the pad-stripping regression is visible.
VOCAB = {
    0: "<pad>",
    1: "<eos>",
    10: "h",
    11: "i",
    20: "f",
    21: "o",
    22: "u",
    23: "r",
    30: "x",
    31: "y",
    32: "z",
    50: "p",
    51: "p",
    52: "p",
    53: "p",
    54: "p",
    55: "<|turn>model\n",
}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB.get(int(token), f"<{int(token)}>") for token in ids)


class FakeProcessor:
    """Renders 2-token prompts for "short" and 5-token prompts otherwise."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(
            max_soft_tokens=280, image_seq_length=280
        )
        self.image_seq_length = 280

    def apply_chat_template(self, conversation, **kwargs):
        text = conversation[-1]["content"]
        ids = [50, 55] if text == "short" else [50, 51, 52, 53, 55]
        input_ids = torch.tensor([ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }


class FakeModel:
    """``generate`` writes the scripted rows and pads the rest of each row."""

    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.generation_config = SimpleNamespace(eos_token_id=[1])
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._parameter])

    def buffers(self):
        return iter([])

    def generate(self, **inputs):
        input_ids = inputs["input_ids"]
        max_new_tokens = inputs["max_new_tokens"]
        generated = torch.zeros(
            (input_ids.shape[0], max_new_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for index, row in enumerate(self.rows):
            row = row[:max_new_tokens]
            generated[index, : len(row)] = torch.tensor(
                row, dtype=input_ids.dtype, device=input_ids.device
            )
        return torch.cat([input_ids, generated], dim=1)


def _conversation(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_generate_batch_strips_trailing_pads_per_row():
    # Row 0 has 2 content tokens + EOS, row 1 has 4 content tokens + EOS; both
    # are padded to max_new_tokens=6 by the fake ``generate``, exactly like HF
    # generate pads finished rows.  Without stripping, the parser sees
    # "hi<eos><pad>..." and replaces the answer with "<pad><pad>...".
    model = FakeModel([[10, 11, 1], [20, 21, 22, 23, 1]])
    outs = generate_batch(
        model,
        FakeProcessor(),
        [_conversation("short"), _conversation("long")],
        Sampling(max_new_tokens=6, temperature=0.0),
    )

    assert [out.text for out in outs] == ["hi", "four"]
    assert [out.completion_tokens for out in outs] == [3, 5]
    assert [out.finish_reason for out in outs] == ["stop", "stop"]
    assert [out.prompt_tokens for out in outs] == [2, 5]


def test_generate_batch_marks_length_when_max_reached():
    model = FakeModel([[30, 31, 32]])
    outs = generate_batch(
        model,
        FakeProcessor(),
        [_conversation("short")],
        Sampling(max_new_tokens=3, temperature=0.0),
    )

    assert outs[0].text == "xyz"
    assert outs[0].completion_tokens == 3
    assert outs[0].finish_reason == "length"


def test_generate_batch_eos_on_last_token_is_stop_not_length():
    model = FakeModel([[30, 31, 1]])
    outs = generate_batch(
        model,
        FakeProcessor(),
        [_conversation("short")],
        Sampling(max_new_tokens=3, temperature=0.0),
    )

    assert outs[0].completion_tokens == 3
    assert outs[0].finish_reason == "stop"


def test_concurrent_submits_share_one_generator_call():
    calls: list[list[str]] = []

    def fake_gen(
        model,
        processor,
        conversations,
        sampling,
        tools=None,
        seed=0,
        image_soft_tokens=280,
    ):
        calls.append([conversation[-1]["content"] for conversation in conversations])
        return [
            GenOut(
                text=conversation[-1]["content"],
                reasoning="",
                tool_calls=[],
                completion_tokens=1,
                finish_reason="stop",
                invalid_tool_call=False,
                prompt_tokens=1,
            )
            for conversation in conversations
        ]

    state = _State(
        spec=ModelSpec(size="e2b", quant="bf16"),
        max_batch=8,
        wait_ms=200,
        gen_fn=fake_gen,
        loader=lambda spec: (object(), object(), object()),
    )

    async def run():
        return await asyncio.gather(
            _submit(state, _conversation("first"), Sampling(), None, 0, 280),
            _submit(state, _conversation("second"), Sampling(), None, 0, 280),
        )

    first, second = asyncio.run(run())

    assert len(calls) == 1
    assert calls[0] == ["first", "second"]
    assert (first.text, second.text) == ("first", "second")


def test_generate_batch_splits_at_max_batch_and_keeps_order():
    calls: list[int] = []

    class RecordingModel(FakeModel):
        all_rows = [[10, 11, 1], [20, 1], [30, 31, 1], [40, 41, 42, 1], [50, 1]]

        def generate(self, **inputs):
            batch = int(inputs["input_ids"].shape[0])
            start = sum(calls)
            calls.append(batch)
            self.rows = self.all_rows[start : start + batch]
            return super().generate(**inputs)

    model = RecordingModel([])
    outs = generate_batch(
        model,
        FakeProcessor(),
        [_conversation("short")] * 5,
        Sampling(max_new_tokens=6, temperature=0.0),
        max_batch=2,
    )

    assert calls == [2, 2, 1]
    assert [out.completion_tokens for out in outs] == [3, 2, 3, 4, 2]


def test_max_batch_for_is_monotone_in_generation_length():
    from gemmaloop.serve.generate import max_batch_for

    lengths = [256, 2048, 4096, 8192, 16384, 32768, 65536]
    batches = [max_batch_for(length) for length in lengths]
    assert batches == sorted(batches, reverse=True)
    assert batches[0] == 64 and batches[-1] == 8


def test_batch_tables_are_per_size_and_e4b_is_conservative():
    from gemmaloop.serve.generate import BATCH_BY_SIZE, max_batch_for

    for bucket in (2048, 4096, 8192, 16384, 32768):
        assert max_batch_for(bucket, "e4b") <= max_batch_for(bucket, "e2b")
    assert max_batch_for(32768, "e2b") == 16
    assert max_batch_for(32768, "e4b") == 4
    assert set(BATCH_BY_SIZE) == {"e2b", "e4b"}
    with pytest.raises(ValueError):
        max_batch_for(2048, "unknown")


def test_generate_batch_uses_the_size_aware_default_batch():
    from types import SimpleNamespace

    calls: list[int] = []

    class RecordingModel(FakeModel):
        def __init__(self):
            super().__init__([])
            self.config = SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(num_hidden_layers=42)
            )

        def generate(self, **inputs):
            batch = int(inputs["input_ids"].shape[0])
            calls.append(batch)
            self.rows = [[10, 11, 1] for _ in range(batch)]
            return super().generate(**inputs)

    model = RecordingModel()
    outputs = generate_batch(
        model,
        FakeProcessor(),
        [_conversation("short")] * 33,
        Sampling(max_new_tokens=2048, temperature=0.0),
    )
    # E4B's derived 2k ceiling is 32, so 33 requests split 32 + 1.
    assert calls == [32, 1]
    assert len(outputs) == 33
