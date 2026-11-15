"""CPU tests for the slot-refill engine (tiny randomly initialised model)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    GenerationConfig,
)

from gemmaloop.serve.generate import Sampling, generate_batch
from gemmaloop.serve.refill import generate_batch_refill


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


def _key(output):
    return (
        output.text,
        output.reasoning,
        output.completion_tokens,
        output.finish_reason,
        output.prompt_tokens,
    )


@pytest.fixture(scope="module")
def tiny():
    return _tiny_model(), _Processor()


def test_refill_without_slot_changes_matches_padded_greedy(tiny):
    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
    ]
    sampling = Sampling(max_new_tokens=8, temperature=0.0)
    padded = generate_batch(model, processor, conversations, sampling, seed=0)
    refill = generate_batch_refill(
        model, processor, conversations, sampling, seed=0, max_batch=3, round_size=4
    )
    assert [_key(output) for output in refill] == [_key(output) for output in padded]


def test_refill_returns_every_request_in_order_when_slots_are_reused(tiny):
    model, processor = tiny
    conversations = [_conversation(letter * (index + 1)) for index, letter in enumerate("abcdef")]
    outputs = generate_batch_refill(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=6, temperature=0.0),
        seed=0,
        max_batch=2,
        round_size=2,
    )
    assert len(outputs) == len(conversations)
    assert all(output.completion_tokens >= 1 for output in outputs)
    # Prompt token counts follow the original request order, never the slot order.
    assert [output.prompt_tokens for output in outputs] == sorted(
        output.prompt_tokens for output in outputs
    )


# ---------------------------------------------------------------------------
# Task 5c: round-based refill
# ---------------------------------------------------------------------------


def _render_prompt_texts(processor, conversations, sampling):
    import gemmaloop.serve.generate as G

    rendered = [G._render(processor, conversation, sampling, None) for conversation in conversations]
    return [G._prompt_text(processor, item) for item in rendered]


def _capture_padded(monkeypatch, model, processor, conversations, sampling, **kwargs):
    import gemmaloop.serve.generate as G

    captured: list[torch.Tensor] = []
    original = G._generate

    def spy(model, inputs, sampling, seed, pad_token_id, **extra):
        ids = original(model, inputs, sampling, seed, pad_token_id, **extra)
        captured.append(ids.detach().cpu())
        return ids

    monkeypatch.setattr(G, "_generate", spy)
    outputs = G.generate_batch(model, processor, conversations, sampling, **kwargs)
    return captured, outputs


def _capture_refill(monkeypatch, model, processor, conversations, sampling, **kwargs):
    import gemmaloop.serve.refill as R

    captured: dict[str, list[int]] = {}
    original = R._finish

    def spy(processor, new_ids, prompt_text, prompt_tokens, sampling, tools, **kwargs):
        captured[prompt_text] = [int(token) for token in new_ids.tolist()]
        return original(
            processor,
            new_ids,
            prompt_text,
            prompt_tokens,
            sampling,
            tools,
            **kwargs,
        )

    monkeypatch.setattr(R, "_finish", spy)
    outputs = R.generate_batch_refill(model, processor, conversations, sampling, **kwargs)
    # Restore the original immediately: chained spies would make two captures
    # in one test overwrite each other, comparing a run with itself.
    monkeypatch.setattr(R, "_finish", original)
    return captured, outputs


def _trim_padded(ids, pad_token_id):
    ids = [int(token) for token in ids]
    while ids and ids[-1] == pad_token_id:
        ids.pop()
    return ids


def test_round_refill_matches_padded_greedy_token_exact(tiny, monkeypatch):
    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=12, temperature=0.0)
    pad = 0

    padded_ids, padded_outputs = _capture_padded(
        monkeypatch, model, processor, conversations, sampling, max_batch=2
    )
    refill_ids, refill_outputs = _capture_refill(
        monkeypatch,
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=3,
    )

    prompt_texts = _render_prompt_texts(processor, conversations, sampling)
    row = 0
    for start in range(0, len(conversations), 2):
        chunk = conversations[start : start + 2]
        for offset in range(len(chunk)):
            index = start + offset
            expected = _trim_padded(padded_ids[row][offset], pad)
            assert refill_ids[prompt_texts[index]] == expected
            assert _key(refill_outputs[index]) == _key(padded_outputs[index])
        row += 1


def test_sampled_tokens_do_not_depend_on_round_composition(tiny, monkeypatch):
    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
        _conversation("hello there"),
    ]
    sampling = Sampling(
        max_new_tokens=10, temperature=1.0, top_p=0.95, top_k=64
    )

    first, _ = _capture_refill(
        monkeypatch, model, processor, conversations, sampling, max_batch=2, round_size=2
    )
    second, _ = _capture_refill(
        monkeypatch, model, processor, conversations, sampling, max_batch=3, round_size=7
    )

    assert set(first) == set(second)
    for prompt in first:
        assert first[prompt] == second[prompt], prompt


def test_refill_keeps_every_request_exactly_once_with_budgets(tiny):
    model, processor = tiny
    prompts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    conversations = [_conversation(prompt) for prompt in prompts]
    budgets = [1, 3, 5, 2, 4, 6]
    outputs = generate_batch_refill(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=6, temperature=0.0),
        seed=0,
        max_batch=2,
        round_size=2,
        budgets=budgets,
        stop_on_eos=False,
        request_ids=[f"item-{index}" for index in range(len(prompts))],
    )

    assert len(outputs) == len(prompts)
    assert [output.completion_tokens for output in outputs] == budgets
    assert [output.finish_reason for output in outputs] == ["length"] * len(prompts)
    assert [output.prompt_tokens for output in outputs] == [
        len(prompt) for prompt in prompts
    ]


def test_finish_reason_stop_and_usage_match_padded(tiny, monkeypatch):
    model, processor = tiny
    conversations = [_conversation("a"), _conversation("longer prompt")]
    sampling = Sampling(max_new_tokens=6, temperature=0.0)

    baseline, _ = _capture_padded(
        monkeypatch, model, processor, conversations, sampling, max_batch=2
    )
    forced_eos = int(baseline[0][0][0])
    monkeypatch.setattr(model.generation_config, "eos_token_id", [forced_eos])

    padded_ids, padded_outputs = _capture_padded(
        monkeypatch, model, processor, conversations, sampling, max_batch=2
    )
    refill_ids, refill_outputs = _capture_refill(
        monkeypatch,
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
    )

    prompt_texts = _render_prompt_texts(processor, conversations, sampling)
    assert refill_ids[prompt_texts[0]][0] == forced_eos
    for index in range(len(conversations)):
        expected = _trim_padded(padded_ids[0][index], 0)
        assert refill_ids[prompt_texts[index]] == expected
        assert _key(refill_outputs[index]) == _key(padded_outputs[index])
        if forced_eos in expected:
            assert refill_outputs[index].finish_reason == "stop"
            assert refill_outputs[index].completion_tokens == expected.index(forced_eos) + 1
        else:
            assert refill_outputs[index].finish_reason == "length"
            assert refill_outputs[index].completion_tokens == sampling.max_new_tokens


def test_admission_order_is_deterministic_by_request_id(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny
    conversations = [_conversation(letter) for letter in "abc"]
    active_ids: list[list[str]] = []
    original = R._decode_round

    def spy(
        model,
        active,
        sampling,
        step,
        pad_token_id,
        eos_ids,
        stop_on_eos,
        **kwargs,
    ):
        active_ids.append([request.request_id for request in active])
        return original(
            model,
            active,
            sampling,
            step,
            pad_token_id,
            eos_ids,
            stop_on_eos,
            **kwargs,
        )

    monkeypatch.setattr(R, "_decode_round", spy)
    outputs = R.generate_batch_refill(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=2, temperature=0.0),
        seed=0,
        max_batch=2,
        round_size=2,
        budgets=[2, 2, 2],
        stop_on_eos=False,
        request_ids=["c", "a", "b"],
    )

    assert len(outputs) == 3
    assert active_ids == [["a", "b"], ["c"]]


def test_greedy_matches_padded_with_queue_and_staggered_finishes(tiny, monkeypatch):
    """Queue > slots, different budgets: new requests join long histories."""
    model, processor = tiny
    prompts = ["a", "longer prompt", "mid prompt!", "xy"]
    budgets = [3, 8, 5, 11]
    conversations = [_conversation(prompt) for prompt in prompts]
    sampling = Sampling(max_new_tokens=max(budgets), temperature=0.0)

    refill_ids, _ = _capture_refill(
        monkeypatch,
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
        budgets=budgets,
        stop_on_eos=False,
    )

    prompt_texts = _render_prompt_texts(processor, conversations, sampling)
    for index, budget in enumerate(budgets):
        monkeypatch.setattr(model.generation_config, "min_new_tokens", budget)
        padded_ids, _ = _capture_padded(
            monkeypatch,
            model,
            processor,
            [conversations[index]],
            Sampling(max_new_tokens=budget, temperature=0.0),
            max_batch=1,
        )
        expected = _trim_padded(padded_ids[0][0], 0)
        assert refill_ids[prompt_texts[index]] == expected, prompts[index]


def test_spill_retry_restores_gumbel_generator_state(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=6, temperature=1.0, top_p=0.95, top_k=64)

    baseline, _ = _capture_refill(
        monkeypatch, model, processor, conversations, sampling, max_batch=2, round_size=2
    )

    calls = {"count": 0}
    original_spilled = R._spilled

    def forced_spill():
        calls["count"] += 1
        return calls["count"] == 1

    monkeypatch.setattr(R, "_spilled", forced_spill)
    spilled, _ = _capture_refill(
        monkeypatch, model, processor, conversations, sampling, max_batch=2, round_size=2
    )
    assert calls["count"] >= 2
    assert set(spilled) == set(baseline)
    for prompt in baseline:
        assert spilled[prompt] == baseline[prompt], prompt


def test_batch_one_oom_reraises_original_exception(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    monkeypatch.setattr(R, "_decode_round", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated OOM"):
        R.generate_batch_refill(
            model,
            processor,
            [_conversation("a")],
            Sampling(max_new_tokens=2, temperature=0.0),
            seed=0,
            max_batch=1,
            round_size=1,
        )


def test_static_round_cache_matches_padded_greedy(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny
    conversations = [_conversation("a"), _conversation("longer prompt")]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)

    monkeypatch.setattr(R, "DYNAMIC_CACHE_MAX_NEW_TOKENS", 0)
    refill_ids, refill_outputs = _capture_refill(
        monkeypatch, model, processor, conversations, sampling, max_batch=2, round_size=2
    )
    padded_ids, padded_outputs = _capture_padded(
        monkeypatch, model, processor, conversations, sampling, max_batch=2
    )

    prompt_texts = _render_prompt_texts(processor, conversations, sampling)
    for index in range(len(conversations)):
        assert refill_ids[prompt_texts[index]] == _trim_padded(
            padded_ids[0][index], 0
        )
        assert _key(refill_outputs[index]) == _key(padded_outputs[index])


# ---------------------------------------------------------------------------
# Task 5d: shared seed convention and progress callbacks
# ---------------------------------------------------------------------------
def test_same_request_seed_gives_identical_sampled_tokens_both_engines(tiny):
    """Same ``(seed, request_id)`` -> same sampled tokens under both engines.

    This is the registered convention ``sha256(seed:item:sample)``; differing
    batch/round composition must not change a request's random stream.
    """
    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("mid prompt!"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=8, temperature=1.0, top_p=0.95, top_k=64)
    request_ids = ["item-3", "item-1", "item-2", "item-0"]

    padded = generate_batch(
        model,
        processor,
        conversations,
        sampling,
        seed=7,
        max_batch=2,
        request_ids=request_ids,
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        seed=7,
        max_batch=3,
        round_size=3,
        request_ids=request_ids,
    )

    assert [_key(output) for output in refill] == [
        _key(output) for output in padded
    ]


def test_on_complete_and_heartbeat_fire_per_finished_request(tiny):
    model, processor = tiny
    prompts = ["a", "bb", "ccc", "dddd", "eeeee"]
    conversations = [_conversation(prompt) for prompt in prompts]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    finished: list[int] = []
    beats: list[dict] = []

    generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        max_batch=2,
        round_size=2,
        request_ids=[f"item-{index}" for index in range(len(prompts))],
        on_complete=lambda index, output: finished.append(index),
        heartbeat=beats.append,
    )

    assert sorted(finished) == list(range(len(prompts)))
    assert beats
    assert beats[-1]["finished"] == len(prompts)
    assert beats[-1]["active"] == 0
    assert beats[-1]["pending"] == 0


def test_media_path_persists_each_request_before_a_freeze(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny
    media = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"image prompt {index}"},
                    {"type": "image", "url": "data:image/png;base64,AAAA"},
                ],
            }
        ]
        for index in range(3)
    ]
    finished: list[int] = []
    beats: list[dict] = []
    calls = {"count": 0}

    def fake_generate(model, processor, conversations, sampling, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt("simulated freeze")
        return [
            type(
                "Output",
                (),
                {
                    "text": "ok",
                    "reasoning": "",
                    "tool_calls": [],
                    "completion_tokens": 2,
                    "finish_reason": "stop",
                    "invalid_tool_call": False,
                    "prompt_tokens": 4,
                },
            )()
        ]

    monkeypatch.setattr(R, "generate_batch", fake_generate)
    with pytest.raises(KeyboardInterrupt):
        generate_batch_refill(
            model,
            processor,
            media,
            Sampling(max_new_tokens=4, temperature=0.0),
            seed=0,
            max_batch=1,
            request_ids=["a", "b", "c"],
            on_complete=lambda index, output: finished.append(index),
            heartbeat=beats.append,
        )

    # The first media request is delivered before the freeze kills the second.
    assert finished == [0]
    assert beats and beats[-1]["finished"] == 1


def test_heartbeat_reports_rounds_rep_refills_and_late_admissions(tiny):
    model, processor = tiny
    prompts = ["a", "bb", "ccc", "dddd", "eeeee"]
    beats: list[dict] = []
    generate_batch_refill(
        model,
        processor,
        [_conversation(prompt) for prompt in prompts],
        Sampling(max_new_tokens=6, temperature=0.0),
        seed=0,
        max_batch=2,
        round_size=2,
        stop_on_eos=False,
        budgets=[2, 4, 6, 6, 6],
        request_ids=[f"item-{index}" for index in range(len(prompts))],
        heartbeat=beats.append,
    )
    last = beats[-1]
    assert last["rounds"] >= 2
    assert last["rep_refills"] > 0
    assert last["late_admissions"] > 0
    assert last["mixed_histories"] is True


def test_both_engines_record_top20_and_chosen_logprobs(tiny):
    model, processor = tiny
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2,
        record_top20=True,
    )
    refill = generate_batch_refill(
        model, processor, conversations, sampling, max_batch=2,
        round_size=2, record_top20=True,
    )

    for outputs in (padded, refill):
        for output in outputs:
            assert output.top20_logprobs is not None
            assert output.chosen_logprobs is not None
            assert len(output.chosen_logprobs) == output.completion_tokens
            assert len(output.top20_logprobs) == output.completion_tokens
            assert all(len(row) == 20 for row in output.top20_logprobs)

    for left, right in zip(padded, refill):
        assert left.chosen_logprobs == pytest.approx(right.chosen_logprobs)
        for left_row, right_row in zip(
            left.top20_logprobs, right.top20_logprobs
        ):
            assert left_row == pytest.approx(right_row)


def test_sampled_engines_record_logprobs_of_the_chosen_token(tiny):
    model, processor = tiny
    conversations = [_conversation("a"), _conversation("longer prompt")]
    sampling = Sampling(max_new_tokens=3, temperature=1.0, top_p=0.95, top_k=64)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2,
        record_top20=True, seed=5, request_ids=["a", "b"],
    )
    refill = generate_batch_refill(
        model, processor, conversations, sampling, max_batch=2,
        round_size=2, record_top20=True, seed=5, request_ids=["a", "b"],
    )
    assert [out.token_ids for out in padded] == [
        out.token_ids for out in refill
    ]
    for left, right in zip(padded, refill):
        assert left.chosen_logprobs == pytest.approx(right.chosen_logprobs)


def test_long_reprefill_is_chunked_and_matches_padded(tiny):
    model, processor = tiny
    prompt = "x" * 2200
    conversations = [_conversation(prompt)]
    sampling = Sampling(max_new_tokens=2, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=1, seed=0
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=1,
        round_size=2,
        seed=0,
    )
    assert refill[0].token_ids == padded[0].token_ids


def test_registered_queue_numbers_admit_late_requests(tiny):
    model, processor = tiny
    prompts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg", "hhhhhhhh"]
    conversations = [_conversation(prompt) for prompt in prompts]
    budgets = [128, 128, 192, 192, 256, 256, 256, 256]
    beats: list[dict] = []
    generate_batch_refill(
        model,
        processor,
        conversations,
        Sampling(max_new_tokens=256, temperature=0.0),
        seed=0,
        max_batch=4,
        round_size=64,
        budgets=budgets,
        request_ids=[f"item-{index}" for index in range(8)],
        heartbeat=beats.append,
    )
    last = beats[-1]
    assert last["rounds"] > 1
    assert last["late_admissions"] > 0
    assert last["rep_refills"] > 0
    assert last["mixed_histories"] is True


def test_chunked_prefill_multrow_left_padding_and_rounds(tiny, monkeypatch):
    import gemmaloop.serve.refill as R

    model, processor = tiny
    monkeypatch.setattr(R, "MAX_PREFILL_TOKENS", 256)
    conversations = [
        _conversation(
            "".join(chr(97 + (index + position) % 26) for position in range(length))
        )
        for index, length in enumerate((300, 600, 700, 520))
    ]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=4, seed=0
    )
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=4,
        round_size=2,
        seed=0,
    )
    assert [output.token_ids for output in refill] == [
        output.token_ids for output in padded
    ]
