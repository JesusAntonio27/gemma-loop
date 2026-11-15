"""Batched generation for Gemma 4 / Gemma-Loop.

One entry point is shared by the offline evaluator and the OpenAI-compatible
server: :func:`generate_batch`.  It renders each conversation with the model's
canonical chat template, batches text-only prompts with left padding, runs
sampling with the evaluation defaults from ``generation_config.json`` and parses
the raw output (reasoning channel, native tool calls) with
:func:`gemmaloop.serve.tool_parse.parse_gemma_output`.

Image conversations are generated one at a time: pixel tensors and
``image_position_ids`` have per-image shapes that do not left-pad like text, and
the evaluator uses batch size 1 for vision anyway.  The function still accepts
a mixed list and preserves order.

Text batches are capped by :func:`max_batch_for`, the largest batch measured to
fit in 12 GiB for the requested ``max_new_tokens`` (64 at 2k, down to 16 at
32k); larger lists are split into independent chunks.  Generations longer than
4k tokens use a preallocated producer-only KV cache (``serve/cache.py``) so
reserved VRAM stays flat instead of fragmenting; shorter ones keep the default
dynamic cache, which is marginally faster.  A round-based refill scheduler with
the same output contract lives in ``serve/refill.py`` (selected through the eval
runner's ``engine="refill"``; it becomes the registered evaluation engine only
after the Task 5d gate passes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from gemmaloop.serve.cache import build_text_cache
from gemmaloop.serve.tool_parse import (
    openai_history_to_gemma,
    parse_gemma_output,
)

# From google/gemma-4-*-it/generation_config.json (global constraint).
DEFAULT_EOS_TOKEN_IDS = (1, 106, 50)
IMAGE_SOFT_TOKEN_CHOICES = (280, 1120)

# Aggregate decode throughput grows ~linearly with the batch until the KV cache
# no longer fits.  These are the *non-spilling* ceilings measured by
# ``scripts/bench_throughput.py`` for E2B bf16 at ~516 prompt tokens: the largest
# batch whose peak reserved VRAM stays under 95 % of the 12 GiB card.  Up to
# 4,096 new tokens the default dynamic cache is used (slightly faster); longer
# runs use the preallocated producer-only cache, whose reserved VRAM is flat.
# Longer generations reserve more KV per sequence, so the batch shrinks with
# ``max_new_tokens``; requests larger than ``max_batch`` are split into
# independent chunks by :func:`_generate_text_batch`.
# E2B entries are *measured* non-spilling ceilings from
# ``scripts/bench_throughput.py`` (~516 prompt tokens, bf16, N=2): the largest
# batch whose peak reserved VRAM stays under 95 % of the 12 GiB card.
#
# E4B is nf4 on this card (bf16 does not fit) and has NOT been run yet, so its
# table is derived and unverified, conservatively from the config KV
# bytes/token with a 25 % VRAM margin:
#   producers 0..23 = 4 full-attention (2 KV heads x head_dim 512)
#                   + 20 sliding (2 KV heads x head_dim 256, capped at 512)
#   full:    4 x 2(K/V) x 2 heads x 512 x 2 B = 16,384 B/token
#   sliding: 20 x 2 x 2 x 256 x 2 B          = 40,960 B/token, only the last
#            512 tokens are kept => ~21 MB per sequence
#   non-cache footprint (nf4 weights + bf16 tied lm_head + vision) ~6 GiB
#   usable = 11.914 x 0.75 = 8.94 GiB => cache budget ~2.94 GiB
# At 32k (33,284 tokens) one sequence needs ~0.566 GiB => batch 4.
# Re-derive/verify when E4B first runs (Task 6 is RAM-gated today).
BATCH_BY_SIZE = {
    "e2b": (
        (2048, 64),
        (4096, 48),
        (8192, 32),
        (16384, 24),
        (32768, 16),
        (65536, 8),
    ),
    "e4b": (
        (2048, 32),
        (4096, 24),
        (8192, 16),
        (16384, 8),
        (32768, 4),
        (65536, 2),
    ),
}
E4B_BATCH_TABLE_STATUS = "derived_unverified"

# Version of the per-size batch policy, recorded in run fingerprints so a
# partial generated under an older table cannot be resumed under a newer one.
BATCH_POLICY_VERSION = "t5b-v1"

# Up to this generation length the default dynamic cache does not fragment
# enough to matter and is slightly faster than the preallocated one.  Longer
# runs use the preallocated producer-only cache (see ``serve/cache.py``), whose
# reserved VRAM is flat instead of growing super-linearly from ~6k tokens.
DYNAMIC_CACHE_MAX_NEW_TOKENS = 4096


def max_batch_for(max_new_tokens: int, size: str = "e2b") -> int:
    """Largest non-spilling batch for this model size and generation length.

    ``size`` selects the E2B (measured) or E4B (derived, unverified) table.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    try:
        table = BATCH_BY_SIZE[size]
    except KeyError as exc:
        raise ValueError(f"unknown model size {size!r}") from exc
    for ceiling, batch in table:
        if max_new_tokens <= ceiling:
            return batch
    return table[-1][1]


def model_size(model: Any) -> str:
    """Infer the Gemma-Loop size from a loaded model config (E2B vs E4B)."""
    text_config = _text_config(model)
    if text_config is None:
        return "e2b"
    layers = int(getattr(text_config, "num_hidden_layers", 0))
    return "e4b" if layers >= 42 else "e2b"


def _request_seed(seed: int, request_id: str) -> int:
    """64-bit seed for one request, ``sha256(f"{seed}:{request_id}")``."""
    digest = hashlib.sha256(f"{seed}:{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _gumbel_generators(
    request_ids: list[str], seed: int, device: torch.device
) -> list[torch.Generator]:
    generators = []
    for request_id in request_ids:
        generator = torch.Generator(device=device)
        generator.manual_seed(_request_seed(seed, request_id) % (2**63 - 1))
        generators.append(generator)
    return generators


class _LogprobRecorder:
    """Top-20 logprobs and the chosen-token logprob for one row."""

    def __init__(self) -> None:
        self.top20: list[tuple[float, ...]] = []
        self.chosen: list[float] = []

    def record(self, scores: torch.Tensor, chosen: int) -> None:
        logp = torch.log_softmax(scores.float(), dim=-1)
        k = min(20, int(logp.numel()))
        values = torch.topk(logp, k).values
        self.top20.append(tuple(float(value) for value in values.tolist()))
        self.chosen.append(float(logp[int(chosen)].item()))

    def result(self) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        return tuple(self.top20), tuple(self.chosen)


class _RecordGreedyLogprobs:
    """Logits processor recording greedy (argmax) step logprobs."""

    def __init__(self, recorders: list[_LogprobRecorder]) -> None:
        self.recorders = recorders

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        for row, recorder in enumerate(self.recorders):
            recorder.record(scores[row], int(scores[row].argmax()))
        return scores


class _PerRequestGumbel:
    """Gumbel-max sampling with one independent generator per row.

    The processor runs after temperature/top-k/top-p filtering and generation
    itself is greedy, so ``argmax(filtered_logits + Gumbel)`` is an exact sample
    from the filtered distribution.  Because each row draws from its own
    generator seeded by ``sha256(seed:request_id)``, a request's tokens do not
    depend on the other rows in the batch (apart from residual bf16 kernel
    differences across batch shapes).
    """

    def __init__(
        self,
        generators: list[torch.Generator],
        recorders: list[_LogprobRecorder] | None = None,
    ) -> None:
        self.generators = generators
        self.recorders = recorders

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows = scores.shape[0]
        if len(self.generators) < rows:
            raise RuntimeError(
                f"Gumbel processor has {len(self.generators)} generators for "
                f"{rows} rows"
            )
        noisy = scores.clone()
        for row in range(rows):
            uniform = torch.rand(
                scores.shape[1],
                generator=self.generators[row],
                device=scores.device,
                dtype=scores.dtype,
            )
            uniform = uniform.clamp_min(torch.finfo(scores.dtype).tiny)
            noisy[row] = scores[row] - torch.log(-torch.log(uniform))
        if self.recorders is not None:
            for row, recorder in enumerate(self.recorders):
                recorder.record(scores[row], int(noisy[row].argmax()))
        return noisy


@dataclass
class Sampling:
    """Sampling parameters, defaulting to the evaluation configuration."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 16384
    enable_thinking: bool = True


@dataclass
class GenOut:
    """One parsed completion.

    ``prompt_tokens`` counts the non-pad prompt tokens from the rendered
    attention mask; it defaults to 0 so existing keyword constructions keep
    working.
    """

    text: str
    reasoning: str
    tool_calls: list[dict]
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    invalid_tool_call: bool
    prompt_tokens: int = 0
    # Untrimmed-by-decoding generated ids (after dropping trailing pad fills).
    # ``None`` for outputs constructed by older callers or fakes; the engine
    # gate compares token ids when they are present.
    token_ids: tuple[int, ...] | None = None
    # Bug detector: top-20 logprobs and chosen-token logprob at every generated
    # step.  ``None`` unless the caller requested recording.
    top20_logprobs: tuple[tuple[float, ...], ...] | None = None
    chosen_logprobs: tuple[float, ...] | None = None


# ---------------------------------------------------------------------------
# Message normalisation (OpenAI / internal image parts)
# ---------------------------------------------------------------------------


def _data_url_of(part: dict) -> str | None:
    value = part.get("image_url", part.get("image", part.get("url")))
    if isinstance(value, dict):
        value = value.get("url")
    return value if isinstance(value, str) else None


def _normalize_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    if part.get("type") in ("image_url", "input_image"):
        url = _data_url_of(part)
        if url is None:
            return part
        return {"type": "image", "url": url}
    if part.get("type") == "image" and "url" not in part:
        url = _data_url_of(part)
        if url is not None:
            return {**part, "url": url}
    return part


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Return a conversation the Gemma 4 template can render.

    Converts OpenAI ``tool_calls`` arguments (string -> mapping) and
    ``reasoning_content`` (-> ``reasoning``), then maps OpenAI ``image_url`` /
    ``input_image`` parts to the processor's internal ``image`` part.  The
    conversion is idempotent so the server and :func:`generate_batch` can both
    call it.
    """
    converted = openai_history_to_gemma(messages)
    for message in converted:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [_normalize_part(part) for part in content]
    return converted


def _has_media(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "image",
                "image_url",
                "input_image",
                "video",
                "audio",
                "input_audio",
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Rendering / generation helpers
# ---------------------------------------------------------------------------


def _set_image_soft_tokens(processor: Any, image_soft_tokens: int) -> None:
    if image_soft_tokens not in IMAGE_SOFT_TOKEN_CHOICES:
        raise ValueError(
            f"image_soft_tokens must be one of {IMAGE_SOFT_TOKEN_CHOICES}, "
            f"got {image_soft_tokens}"
        )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "max_soft_tokens"):
            image_processor.max_soft_tokens = image_soft_tokens
        if hasattr(image_processor, "image_seq_length"):
            image_processor.image_seq_length = image_soft_tokens
    if hasattr(processor, "image_seq_length"):
        processor.image_seq_length = image_soft_tokens


def _render(
    processor: Any,
    conversation: list[dict],
    sampling: Sampling,
    tools: list[dict] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": sampling.enable_thinking,
    }
    if tools:
        kwargs["tools"] = tools
    return processor.apply_chat_template(conversation, **kwargs)


def _model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "cpu":
            return parameter.device
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            return buffer.device
    # A model whose parameters all live on the CPU (unit-test fakes, CPU-only
    # runs) must keep its inputs on the CPU even when a GPU is present.
    return torch.device("cpu")


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _prompt_text(processor: Any, inputs: dict[str, Any]) -> str:
    tokenizer = processor.tokenizer
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        input_ids = input_ids[attention_mask[0].bool()]
    return tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)


def _eos_token_ids(model: Any) -> list[int]:
    generation_config = getattr(model, "generation_config", None)
    ids = getattr(generation_config, "eos_token_id", None)
    if ids is None:
        return list(DEFAULT_EOS_TOKEN_IDS)
    if isinstance(ids, int):
        return [ids]
    return list(ids)


def _pad_token_id(processor: Any) -> int:
    """Single source of truth for the pad id used by generation and padding.

    Hugging Face ``generate`` fills finished rows of a batch with this token,
    so it must be the same id used by :func:`_left_pad` and by the trailing-pad
    stripping in :func:`_finish`.
    """
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    return 0 if pad_token_id is None else int(pad_token_id)


def _prompt_token_count(inputs: dict[str, Any]) -> int:
    """Non-pad prompt tokens, i.e. the rendered attention mask's row sum."""
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return int(inputs["input_ids"].shape[-1])
    return int(attention_mask.sum().item())



def _text_config(model: Any) -> Any | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            return None
    return config


def _cache_for(
    model: Any,
    inputs: dict[str, Any],
    sampling: Sampling,
    cache_new_tokens: int | None = None,
) -> Any | None:
    """Preallocated producer-only cache, or ``None`` for simple/fake models.

    The cache holds only the KV-producer layers and preallocates the three
    full-attention layers, which removes the allocator fragmentation that made
    a 16k-token batch reserve 25 GiB (Task 5b).  Models without a Gemma-style
    text config (unit-test fakes, non-Gemma models) fall back to the model's
    default dynamic cache.
    """
    text_config = _text_config(model)
    if text_config is None or not hasattr(text_config, "num_hidden_layers"):
        return None
    budget = int(cache_new_tokens or sampling.max_new_tokens)
    if budget <= DYNAMIC_CACHE_MAX_NEW_TOKENS:
        return None
    prompt_length = int(inputs["input_ids"].shape[1])
    try:
        return build_text_cache(text_config, prompt_length + budget)
    except (ValueError, AttributeError, IndexError) as exc:  # pragma: no cover
        import warnings

        warnings.warn(
            f"preallocated KV cache unavailable ({exc}); falling back to the "
            "dynamic cache, which may spill on long generations",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _generate(
    model: Any,
    inputs: dict[str, Any],
    sampling: Sampling,
    seed: int,
    pad_token_id: int,
    request_ids: list[str] | None = None,
    cache_new_tokens: int | None = None,
    record_top20: bool = False,
    logprob_sink: list[_LogprobRecorder] | None = None,
) -> torch.Tensor:
    prompt_length = inputs["input_ids"].shape[1]
    rows = int(inputs["input_ids"].shape[0])
    if request_ids is None:
        request_ids = [str(index) for index in range(rows)]
    if len(request_ids) != rows:
        raise ValueError(
            f"got {len(request_ids)} request ids for a batch of {rows} rows"
        )
    recorders = (
        [_LogprobRecorder() for _ in range(rows)] if record_top20 else None
    )
    kwargs: dict[str, Any] = {
        "max_new_tokens": sampling.max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": _eos_token_ids(model),
    }
    if sampling.temperature > 0:
        # Gumbel-max sampling over filtered logits with one generator per
        # request: greedy decoding plus per-row noise, independent of batch
        # composition.  Temperature/top-k/top-p run before the noise.
        device = _model_device(model)
        processors = [
            TemperatureLogitsWarper(sampling.temperature),
            TopKLogitsWarper(sampling.top_k),
            TopPLogitsWarper(sampling.top_p),
            _PerRequestGumbel(
                _gumbel_generators(request_ids, seed, device), recorders
            ),
        ]
        kwargs["do_sample"] = False
        kwargs["logits_processor"] = LogitsProcessorList(processors)
    else:
        kwargs["do_sample"] = False
        if recorders is not None:
            kwargs["logits_processor"] = LogitsProcessorList(
                [_RecordGreedyLogprobs(recorders)]
            )
    cache = _cache_for(model, inputs, sampling, cache_new_tokens=cache_new_tokens)
    if cache is not None:
        kwargs["past_key_values"] = cache

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    del cache
    if logprob_sink is not None and recorders is not None:
        logprob_sink.extend(recorders)
    return output[:, prompt_length:]


def _finish(
    processor: Any,
    new_ids: torch.Tensor,
    prompt_text: str,
    prompt_tokens: int,
    sampling: Sampling,
    tools: list[dict] | None,
    *,
    eos_token_ids: Iterable[int] | None = None,
    top20_logprobs: tuple[tuple[float, ...], ...] | None = None,
    chosen_logprobs: tuple[float, ...] | None = None,
) -> GenOut:
    tokenizer = processor.tokenizer
    pad_token_id = _pad_token_id(processor)
    ids = new_ids.tolist()

    # ``generate`` fills every row that finished before the longest one with
    # pad_token_id.  The Gemma response template does not delimit content with
    # <pad>, so decoding the fill would either append pads to the answer or --
    # after an <eos>/<turn|> close -- open a new content region and overwrite
    # it.  Drop only the trailing run: interior tokens are real output.
    end = len(ids)
    while end > 0 and ids[end - 1] == pad_token_id:
        end -= 1
    ids = ids[:end]

    completion_tokens = sum(1 for token in ids if token != pad_token_id)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    text, reasoning, tool_calls, invalid = parse_gemma_output(
        decoded, prefix=prompt_text, tools=tools
    )
    eos_ids = set(eos_token_ids if eos_token_ids is not None else DEFAULT_EOS_TOKEN_IDS)
    last_is_eos = bool(ids) and ids[-1] in eos_ids
    if top20_logprobs is not None:
        top20_logprobs = tuple(top20_logprobs[: len(ids)])
        chosen_logprobs = tuple((chosen_logprobs or ())[: len(ids)])
    return GenOut(
        text=text,
        reasoning=reasoning,
        tool_calls=tool_calls,
        completion_tokens=completion_tokens,
        finish_reason="length"
        if completion_tokens >= sampling.max_new_tokens and not last_is_eos
        else "stop",
        invalid_tool_call=invalid,
        prompt_tokens=prompt_tokens,
        token_ids=tuple(ids),
        top20_logprobs=top20_logprobs,
        chosen_logprobs=chosen_logprobs,
    )


def _generate_one(
    model: Any,
    processor: Any,
    conversation: list[dict],
    sampling: Sampling,
    tools: list[dict] | None,
    seed: int,
    request_id: str = "0",
    cache_new_tokens: int | None = None,
    record_top20: bool = False,
) -> GenOut:
    rendered = _render(processor, conversation, sampling, tools)
    prompt_text = _prompt_text(processor, rendered)
    prompt_tokens = _prompt_token_count(rendered)
    device = _model_device(model)
    inputs = _move_inputs(rendered, device)
    pad_token_id = _pad_token_id(processor)
    sink: list[_LogprobRecorder] = []
    new_ids = _generate(
        model,
        inputs,
        sampling,
        seed,
        pad_token_id,
        request_ids=[request_id],
        cache_new_tokens=cache_new_tokens,
        record_top20=record_top20,
        logprob_sink=sink,
    )[0]
    top20, chosen = sink[0].result() if sink else (None, None)
    return _finish(
        processor,
        new_ids,
        prompt_text,
        prompt_tokens,
        sampling,
        tools,
        eos_token_ids=_eos_token_ids(model),
        top20_logprobs=top20,
        chosen_logprobs=chosen,
    )


def _left_pad(
    rendered: list[dict[str, Any]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_length = max(item["input_ids"].shape[1] for item in rendered)
    input_ids = []
    attention_mask = []
    mm_token_type_ids = []
    for item in rendered:
        length = item["input_ids"].shape[1]
        pad = max_length - length
        ids = item["input_ids"][0]
        mask = item.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        else:
            mask = mask[0]
        mm = item.get("mm_token_type_ids")
        if mm is None:
            mm = torch.zeros_like(ids)
        else:
            mm = mm[0]
        input_ids.append(
            torch.cat([torch.full((pad,), pad_token_id, dtype=ids.dtype), ids])
        )
        attention_mask.append(
            torch.cat([torch.zeros((pad,), dtype=mask.dtype), mask])
        )
        mm_token_type_ids.append(
            torch.cat([torch.zeros((pad,), dtype=mm.dtype), mm])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "mm_token_type_ids": torch.stack(mm_token_type_ids),
    }


def _generate_text_batch(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    tools: list[dict] | None,
    seed: int,
    max_batch: int,
    request_ids: list[str] | None = None,
    cache_new_tokens: int | None = None,
    record_top20: bool = False,
) -> list[GenOut]:
    """Generate every conversation, chunked at ``max_batch`` sequences.

    Each chunk is a complete left-padded ``generate`` call with its own
    preallocated cache, so the peak VRAM is bounded by ``max_batch`` and the
    caller can submit more requests than fit on the card at once.  Sampling is
    per-request (Gumbel-max with a generator seeded by
    ``sha256(seed:request_id)``), so temperature>0 tokens do not depend on the
    chunk or row a request lands in; residual bf16 kernel differences across
    batch shapes can still flip a near-tie.
    """
    rendered = [
        _render(processor, conversation, sampling, tools)
        for conversation in conversations
    ]
    prompts = [_prompt_text(processor, item) for item in rendered]
    prompt_tokens = [_prompt_token_count(item) for item in rendered]
    pad_token_id = _pad_token_id(processor)
    device = _model_device(model)
    if request_ids is None:
        request_ids = [str(index) for index in range(len(conversations))]
    if len(request_ids) != len(conversations):
        raise ValueError(
            f"got {len(request_ids)} request ids for "
            f"{len(conversations)} conversations"
        )

    outputs: list[GenOut] = []
    for start in range(0, len(rendered), max_batch):
        stop = min(start + max_batch, len(rendered))
        batch = _left_pad(rendered[start:stop], pad_token_id)
        batch = _move_inputs(batch, device)
        sink: list[_LogprobRecorder] = []
        new_ids = _generate(
            model,
            batch,
            sampling,
            seed,
            pad_token_id,
            request_ids=request_ids[start:stop],
            cache_new_tokens=cache_new_tokens,
            record_top20=record_top20,
            logprob_sink=sink,
        )
        finished: list[GenOut] = []
        for index in range(stop - start):
            top20, chosen = (
                sink[index].result() if sink else (None, None)
            )
            finished.append(
                _finish(
                    processor,
                    new_ids[index],
                    prompts[start + index],
                    prompt_tokens[start + index],
                    sampling,
                    tools,
                    eos_token_ids=_eos_token_ids(model),
                    top20_logprobs=top20,
                    chosen_logprobs=chosen,
                )
            )
        outputs.extend(finished)
        del batch, new_ids
    return outputs


def generate_batch(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    tools: list[dict] | None = None,
    seed: int = 0,
    image_soft_tokens: int = 280,
    max_batch: int | None = None,
    request_ids: list[str] | None = None,
    cache_new_tokens: int | None = None,
    record_top20: bool = False,
) -> list[GenOut]:
    """Generate one :class:`GenOut` per conversation.

    ``conversations`` are OpenAI-style message lists.  Image parts may use the
    processor form ``{"type": "image", "url": ...}`` or OpenAI's
    ``{"type": "image_url", "image_url": {"url": "data:..."}}``; both are
    normalised before rendering.  With ``image_soft_tokens=1120`` the image
    processor is switched to 1120 visual tokens (the default is 280).

    ``max_batch`` caps the number of sequences in one generator call.  The
    default comes from :func:`max_batch_for` for the model size inferred from
    the config (E2B: 64/48/32/24/16) and splits larger lists into independent
    chunks.  Image conversations are still generated one at a time.

    ``request_ids`` are stable per-request identities used to seed sampling
    (``sha256(f"{seed}:{request_id}")``); they default to the list index.
    ``cache_new_tokens`` lets a caller preallocate the static cache for a
    larger generation length while only generating a short diagnostic window.
    """
    if not conversations:
        return []
    if sampling.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if request_ids is not None and len(request_ids) != len(conversations):
        raise ValueError(
            f"got {len(request_ids)} request ids for "
            f"{len(conversations)} conversations"
        )
    if request_ids is None:
        request_ids = [str(index) for index in range(len(conversations))]
    if max_batch is None:
        max_batch = max_batch_for(sampling.max_new_tokens, model_size(model))
    if max_batch < 1:
        raise ValueError(f"max_batch must be at least 1, got {max_batch}")

    normalized = [normalize_messages(conversation) for conversation in conversations]
    _set_image_soft_tokens(processor, image_soft_tokens)

    if any(_has_media(conversation) for conversation in normalized):
        return [
            _generate_one(
                model,
                processor,
                conversation,
                sampling,
                tools,
                seed,
                request_id=request_ids[index],
                cache_new_tokens=cache_new_tokens,
                record_top20=record_top20,
            )
            for index, conversation in enumerate(normalized)
        ]

    return _generate_text_batch(
        model,
        processor,
        normalized,
        sampling,
        tools,
        seed,
        max_batch,
        request_ids=request_ids,
        cache_new_tokens=cache_new_tokens,
        record_top20=record_top20,
    )
