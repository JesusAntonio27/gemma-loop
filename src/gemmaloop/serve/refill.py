"""Round-based refill generation engine.

The padded engine runs one left-padded batch per chunk: a finished sequence
keeps its slot (filled with pads) until the longest sequence in the chunk
finishes.  On natural-length reasoning workloads the completion lengths have a
long tail (Task 5c calibration: p50 5.4k, p90 10.7k, max 11.6k tokens), so most
of the card sits idle while the last few traces finish.

This module keeps the batch full.  A request queue feeds a set of slots; the
engine decodes in rounds of at most ``round_size`` tokens (default 1,024).
After every round, finished requests are removed and queued requests are
admitted, in deterministic request-id order, up to the model size's adaptive
batch (:func:`gemmaloop.serve.generate.max_batch_for`).  Continuing requests are
re-prefilled from ``prompt + generated-so-far`` instead of keeping a cache
across rounds: re-prefill is compute-bound and cheap relative to decode, and it
means no KV cache is retained (or merged, which is what made the Task 5b slot
refill spill at 13.97 GiB).

Sampling uses the per-request Gumbel-max processor from
:mod:`gemmaloop.serve.generate`.  Each request owns a generator seeded by
``sha256(seed:request_id)`` that is carried across rounds, so a request's
sampled tokens do not depend on the round size or on which other requests share
its batch.  The benchmark runner supplies the same request identity as the
padded engine, ``f"{item_id}:{sample_idx}"`` with no arm name, so both engines
hash exactly ``seed:item:sample``.  Greedy decoding is token-exact with the
padded engine (same prompt, same model), and the public contract is identical:
:func:`generate_batch_refill` returns one
:class:`~gemmaloop.serve.generate.GenOut` per conversation, in the input order,
with the same ``finish_reason``, ``completion_tokens`` and ``prompt_tokens``.
Image conversations still use the one-at-a-time padded path.

``on_complete`` delivers each finished request to the caller immediately (the
evaluation runner appends and fsyncs it to the bench partial file), and
``heartbeat`` reports round progress for a liveness file.  A freeze therefore
loses only the round in progress, and a resume skips every finished request.

Memory policy: each round chooses its cache from the round's longest input plus
``round_size``.  Short rounds use the default dynamic cache; long ones use the
preallocated producer-only cache from :mod:`gemmaloop.serve.cache` and discard
it after the round.  After every round the peak reserved memory is checked
against the Task 5b spill threshold (95 % of the card); a spill halves the batch
and retries the same round from the unchanged request state.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from gemmaloop.serve.cache import build_text_cache
from gemmaloop.serve.generate import (
    DYNAMIC_CACHE_MAX_NEW_TOKENS,
    GenOut,
    Sampling,
    _LogprobRecorder,
    _PerRequestGumbel,
    _RecordGreedyLogprobs,
    _eos_token_ids,
    _finish,
    _gumbel_generators,
    _has_media,
    _left_pad,
    _model_device,
    _move_inputs,
    _pad_token_id,
    _prompt_text,
    _prompt_token_count,
    _render,
    _set_image_soft_tokens,
    _text_config,
    generate_batch,
    max_batch_for,
    normalize_messages,
)

DEFAULT_ROUND_SIZE = 1024
SPILL_RESERVED_FRACTION = 0.95
MAX_PREFILL_TOKENS = 2048


@dataclass
class _Request:
    """One queued/generating request and its persistent sampling state."""

    index: int
    request_id: str
    prompt_text: str
    prompt_tokens: int
    prompt_ids: list[int]
    prompt_mm: list[int]
    budget: int
    generator: torch.Generator | None
    generated: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    top20: list[tuple[float, ...]] = field(default_factory=list)
    chosen: list[float] = field(default_factory=list)


def _prompt_ids(item: dict) -> tuple[list[int], list[int]]:
    """Non-pad prompt token ids and their mm token types."""
    full_ids = item["input_ids"][0]
    mask = item.get("attention_mask")
    keep = None if mask is None else mask[0].bool()
    ids = full_ids if keep is None else full_ids[keep]
    mm = item.get("mm_token_type_ids")
    if mm is None:
        mm = torch.zeros_like(full_ids)
    else:
        mm = mm[0]
    if keep is not None:
        mm = mm[keep]
    return [int(token) for token in ids.tolist()], [int(token) for token in mm.tolist()]


def _round_cache(model, input_length: int, step: int):
    """Cache for one round, or ``None`` for the default dynamic cache.

    The preallocated producer-only cache avoids the allocator fragmentation that
    makes long contexts reserve several times their live KV.  It is only worth
    its preallocation cost once the round's full context is long.
    """
    text_config = _text_config(model)
    if text_config is None or not hasattr(text_config, "num_hidden_layers"):
        return None
    total = int(input_length) + int(step)
    if total <= DYNAMIC_CACHE_MAX_NEW_TOKENS:
        return None
    try:
        return build_text_cache(text_config, total)
    except (ValueError, AttributeError, IndexError) as exc:  # pragma: no cover
        warnings.warn(
            f"preallocated KV cache unavailable ({exc}); falling back to the "
            "dynamic cache for this round",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _spilled() -> bool:
    """True when this round reserved more than 95 % of the card (Task 5b rule)."""
    if not torch.cuda.is_available():
        return False
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return bool(
        total and torch.cuda.max_memory_reserved() > SPILL_RESERVED_FRACTION * total
    )


def _decode_round(
    model,
    active: list[_Request],
    sampling: Sampling,
    step: int,
    pad_token_id: int,
    eos_ids: set[int],
    stop_on_eos: bool,
    logprob_sink: list[_LogprobRecorder] | None = None,
) -> torch.Tensor:
    """Re-prefill ``prompt + generated`` for every active request and decode."""
    device = _model_device(model)
    rows = []
    for request in active:
        ids = request.prompt_ids + request.generated
        mm = request.prompt_mm + [0] * len(request.generated)
        rows.append(
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([mm], dtype=torch.long),
            }
        )
    batch = _move_inputs(_left_pad(rows, pad_token_id), device)
    input_length = int(batch["input_ids"].shape[1])

    recorders = (
        [_LogprobRecorder() for _ in active]
        if logprob_sink is not None
        else None
    )
    kwargs: dict = {
        "max_new_tokens": int(step),
        "min_new_tokens": 0 if stop_on_eos else int(step),
        "pad_token_id": pad_token_id,
        "eos_token_id": sorted(eos_ids) if stop_on_eos else -1,
        "do_sample": False,
    }
    if sampling.temperature > 0:
        # Same warper order as the padded engine, then one Gumbel draw per row
        # from the request's persistent generator.
        kwargs["logits_processor"] = LogitsProcessorList(
            [
                TemperatureLogitsWarper(sampling.temperature),
                TopKLogitsWarper(sampling.top_k),
                TopPLogitsWarper(sampling.top_p),
                _PerRequestGumbel(
                    [request.generator for request in active], recorders
                ),
            ]
        )
    elif recorders is not None:
        kwargs["logits_processor"] = LogitsProcessorList(
            [_RecordGreedyLogprobs(recorders)]
        )
    if input_length > MAX_PREFILL_TOKENS:
        new_tokens = _decode_round_chunked(
            model,
            active,
            sampling,
            step,
            pad_token_id,
            eos_ids,
            stop_on_eos,
            batch,
            input_length,
            kwargs,
        )
        if logprob_sink is not None and recorders is not None:
            logprob_sink.extend(recorders)
        return new_tokens
    cache = _round_cache(model, input_length, step)
    if cache is not None:
        kwargs["past_key_values"] = cache
    with torch.inference_mode():
        output = model.generate(**batch, **kwargs)
    del cache
    if logprob_sink is not None and recorders is not None:
        logprob_sink.extend(recorders)
    return output[:, input_length:]


def _decode_round_chunked(
    model,
    active: list[_Request],
    sampling: Sampling,
    step: int,
    pad_token_id: int,
    eos_ids: set[int],
    stop_on_eos: bool,
    batch: dict,
    input_length: int,
    kwargs: dict,
) -> torch.Tensor:
    """Re-prefill long histories in <=2048-token forwards, then decode.

    The KV cache is filled chunk by chunk; only the final token is handed to
    ``generate`` for the decode steps.  This bounds one forward's activation
    memory by ``MAX_PREFILL_TOKENS`` instead of batch x full history.
    """

    from transformers.cache_utils import DynamicCache

    if any(any(request.prompt_mm) for request in active):
        raise RuntimeError(
            "chunked re-prefill is text-only; image conversations must stay "
            "on the padded media path"
        )
    text_config = _text_config(model)
    total = int(input_length) + int(step)
    cache = _round_cache(model, int(input_length), int(step))
    if cache is None:
        cache = DynamicCache(config=text_config)
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)
    prefix = int(input_length) - 1
    rows = max(1, int(ids.shape[0]))
    # Bound the total tokens in one forward, not just each row's chunk.
    chunk = max(1, MAX_PREFILL_TOKENS // rows)
    for start in range(0, prefix, chunk):
        end = min(start + chunk, prefix)
        with torch.inference_mode():
            model(
                input_ids=ids[:, start:end],
                attention_mask=mask[:, :end],
                position_ids=positions[:, start:end],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
    decode_kwargs = dict(kwargs)
    decode_kwargs["past_key_values"] = cache
    decode_kwargs["attention_mask"] = mask
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids[:, -1:],
            **decode_kwargs,
        )
    return output[:, 1:]


def _pending_key(provided_ids: bool, request: _Request):
    if provided_ids:
        return (request.request_id, request.index)
    return (request.index,)


def _sort_pending(pending: list[_Request], provided_ids: bool) -> None:
    pending.sort(key=lambda request: _pending_key(provided_ids, request))


def generate_batch_refill(
    model,
    processor,
    conversations: list[list[dict]],
    sampling: Sampling,
    tools: list[dict] | None = None,
    seed: int = 0,
    image_soft_tokens: int = 280,
    max_batch: int | None = None,
    round_size: int = DEFAULT_ROUND_SIZE,
    budgets: list[int] | None = None,
    stop_on_eos: bool = True,
    request_ids: list[str] | None = None,
    on_complete: Callable[[int, GenOut], None] | None = None,
    heartbeat: Callable[[dict], None] | None = None,
    record_top20: bool = False,
) -> list[GenOut]:
    """Generate with round-based refill; same contract as ``generate_batch``.

    ``round_size`` is the maximum number of new tokens decoded between admission
    boundaries.  ``budgets`` optionally gives every request its own
    completion-token budget (the long-tail harness uses it); it defaults to
    ``sampling.max_new_tokens`` for every request.  ``stop_on_eos=False`` makes
    every slot produce exactly its budget.  ``request_ids`` are the stable
    per-request identities used for sampling and for the deterministic
    admission order; they default to the input index.

    ``on_complete(index, output)`` fires as soon as a request finishes, so the
    caller can persist it before the rest of the bench runs.  ``heartbeat``
    receives a small progress dict after every round and is meant for a
    liveness file; neither callback changes the generated tokens.
    """
    if not conversations:
        return []
    if sampling.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if budgets is None:
        budgets = [sampling.max_new_tokens] * len(conversations)
    if len(budgets) != len(conversations) or any(budget < 1 for budget in budgets):
        raise ValueError("budgets must have one positive value per conversation")
    provided_ids = request_ids is not None
    if request_ids is None:
        request_ids = [str(index) for index in range(len(conversations))]
    if len(request_ids) != len(conversations):
        raise ValueError(
            f"got {len(request_ids)} request ids for "
            f"{len(conversations)} conversations"
        )
    if max_batch is None:
        max_batch = max_batch_for(sampling.max_new_tokens, _model_size(model))
    if max_batch < 1:
        raise ValueError(f"max_batch must be at least 1, got {max_batch}")
    if round_size < 1:
        raise ValueError(f"round_size must be at least 1, got {round_size}")

    normalized = [normalize_messages(conversation) for conversation in conversations]
    _set_image_soft_tokens(processor, image_soft_tokens)
    if any(_has_media(conversation) for conversation in normalized):
        # Vision inputs use the one-at-a-time padded path.  Drive it request by
        # request so a freeze keeps every completed completion and the
        # heartbeat advances exactly like the text path.
        outputs: list[GenOut] = []
        for index, conversation in enumerate(normalized):
            per_request = replace(sampling, max_new_tokens=int(budgets[index]))
            output = generate_batch(
                model,
                processor,
                [conversation],
                per_request,
                tools=tools,
                seed=seed,
                image_soft_tokens=image_soft_tokens,
                max_batch=1,
                request_ids=[request_ids[index]],
            )[0]
            outputs.append(output)
            if on_complete is not None:
                on_complete(index, output)
            if heartbeat is not None:
                heartbeat(
                    {
                        "round": index + 1,
                        "finished": len(outputs),
                        "active": 0,
                        "pending": len(normalized) - len(outputs),
                        "generated_tokens": sum(
                            int(item.completion_tokens) for item in outputs
                        ),
                        "rounds": 0,
                        "rep_refills": 0,
                        "late_admissions": 0,
                        "mixed_histories": False,
                    }
                )
        return outputs

    rendered = [
        _render(processor, conversation, sampling, tools)
        for conversation in normalized
    ]
    prompts = [_prompt_text(processor, item) for item in rendered]
    prompt_tokens = [_prompt_token_count(item) for item in rendered]
    prompt_ids = [_prompt_ids(item) for item in rendered]
    device = _model_device(model)
    generators = (
        _gumbel_generators(request_ids, seed, device)
        if sampling.temperature > 0
        else [None] * len(conversations)
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requests = [
        _Request(
            index=index,
            request_id=request_ids[index],
            prompt_text=prompts[index],
            prompt_tokens=prompt_tokens[index],
            prompt_ids=prompt_ids[index][0],
            prompt_mm=prompt_ids[index][1],
            budget=int(budgets[index]),
            generator=generators[index],
        )
        for index in range(len(conversations))
    ]
    pending = list(requests)
    _sort_pending(pending, provided_ids)
    active: list[_Request] = []
    results: list[GenOut | None] = [None] * len(requests)
    current_batch = int(max_batch)
    beat = 0
    rounds = 0
    rep_refills = 0
    late_admissions = 0
    mixed_histories = False
    eos_ids = set(_eos_token_ids(model)) if stop_on_eos else set()
    pad_token_id = _pad_token_id(processor)

    while pending or active:
        # A spill retries this round from the state before admission.
        before_active = list(active)
        before_pending = list(pending)
        before_rounds = rounds
        before_rep_refills = rep_refills
        before_late_admissions = late_admissions
        admitted = 0
        while pending and len(active) < current_batch:
            active.append(pending.pop(0))
            admitted += 1
        if not active:
            break
        if rounds > 0:
            late_admissions += admitted
            continuing = [
                request for request in before_active if request.generated
            ]
            if admitted and continuing:
                histories = {
                    len(request.prompt_ids) + len(request.generated)
                    for request in active
                }
                if len(histories) > 1:
                    mixed_histories = True
        # Every continuing request is re-prefilled from prompt + generated.
        rep_refills += sum(bool(request.generated) for request in active)

        step = min(
            round_size,
            min(request.budget - len(request.generated) for request in active),
        )
        if step < 1:  # pragma: no cover - defensive; finished requests are dropped
            raise RuntimeError("refill engine selected a non-positive round")

        # The round is re-decoded after a spill/OOM.  Gumbel draws made by the
        # discarded attempt must be undone so the retried round sees exactly
        # the generator state of an uninterrupted run.
        generator_states = {
            id(request): request.generator.get_state()
            for request in active
            if request.generator is not None
        }
        _reset_peak()
        spilled = False
        oom: torch.cuda.OutOfMemoryError | None = None
        round_logprobs: list[_LogprobRecorder] = []
        try:
            new_tokens = _decode_round(
                model,
                active,
                sampling,
                step,
                pad_token_id,
                eos_ids,
                stop_on_eos,
                logprob_sink=round_logprobs if record_top20 else None,
            )
        except torch.cuda.OutOfMemoryError as exc:
            new_tokens = None
            spilled = True
            oom = exc
        if not spilled and _spilled():
            spilled = True

        if spilled:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch > 1:
                for request in active:
                    state = generator_states.get(id(request))
                    if state is not None:
                        request.generator.set_state(state)
                current_batch = max(1, current_batch // 2)
                active = before_active
                pending = before_pending
                rounds = before_rounds
                rep_refills = before_rep_refills
                late_admissions = before_late_admissions
                if len(active) > current_batch:
                    # Keep the deterministic prefix active and return the rest
                    # to the queue with their generated tokens intact.
                    active.sort(key=lambda request: _pending_key(provided_ids, request))
                    pending.extend(active[current_batch:])
                    active = active[:current_batch]
                    _sort_pending(pending, provided_ids)
                warnings.warn(
                    f"refill: spill detected; retrying at batch {current_batch}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if oom is not None:
                raise oom
            warnings.warn(
                "refill: spill at batch 1; continuing (results stay valid, "
                "throughput may degrade)",
                RuntimeWarning,
                stacklevel=2,
            )

        if record_top20 and round_logprobs:
            for request, recorder in zip(active, round_logprobs):
                request.top20.extend(recorder.top20)
                request.chosen.extend(recorder.chosen)
        tokens_per_row = new_tokens.tolist()
        next_active: list[_Request] = []
        for row, request in enumerate(active):
            tokens = [int(token) for token in tokens_per_row[row]]
            stop_at = (
                next(
                    (
                        position
                        for position, token in enumerate(tokens)
                        if token in eos_ids
                    ),
                    None,
                )
                if stop_on_eos
                else None
            )
            if stop_at is not None:
                request.generated.extend(tokens[: stop_at + 1])
                request.finish_reason = "stop"
            else:
                request.generated.extend(tokens)
                if len(request.generated) >= request.budget:
                    request.finish_reason = "length"

            if request.finish_reason is not None:
                ids = torch.tensor(request.generated, dtype=torch.long)
                output = _finish(
                    processor,
                    ids,
                    request.prompt_text,
                    request.prompt_tokens,
                    replace(sampling, max_new_tokens=request.budget),
                    tools,
                    eos_token_ids=_eos_token_ids(model),
                    top20_logprobs=tuple(request.top20) if record_top20 else None,
                    chosen_logprobs=tuple(request.chosen) if record_top20 else None,
                )
                results[request.index] = output
                if on_complete is not None:
                    on_complete(request.index, output)
            else:
                next_active.append(request)
        active = next_active
        rounds += 1
        if heartbeat is not None:
            heartbeat(
                {
                    "round": int(beat) + 1,
                    "rounds": rounds,
                    "rep_refills": rep_refills,
                    "late_admissions": late_admissions,
                    "mixed_histories": mixed_histories,
                    "finished": sum(value is not None for value in results),
                    "active": len(active),
                    "pending": len(pending),
                    "generated_tokens": sum(
                        len(request.generated) for request in requests
                    ),
                }
            )
            beat += 1

    missing = [index for index, value in enumerate(results) if value is None]
    if missing:
        raise RuntimeError(f"refill engine produced no output for requests {missing}")
    return [value for value in results if value is not None]


def _model_size(model) -> str:
    from gemmaloop.serve.generate import model_size

    return model_size(model)


__all__ = ["DEFAULT_ROUND_SIZE", "generate_batch_refill"]
