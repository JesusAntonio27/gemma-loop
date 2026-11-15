"""Token-weighted, decontaminated healing mixture.

``build_mix`` streams each source, rejects rows that fail the source filter or
the evaluation index, and interleaves the accepted rows in one deterministic
order.  The interleave key is ``tokens_consumed_before_row / weight``: a row is
included when that key is below ``target_tokens``, which fills each source's
share (``weight * target_tokens``) while guaranteeing that a larger target with
the same seed is an exact extension of the smaller mixture.  The whole set is
never reshuffled.

The manifest's ``sha256`` is the SHA-256 of the exact bytes written by
:func:`write_jsonl`, so ``hashlib.sha256(train.jsonl.read_bytes())`` matches it.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from gemmaloop.model.snapshot import chat_template_sha256

from .decontam import EvalIndex, tokenizer_name
from .sources import (
    SOURCES,
    SourceSpec,
    iter_rows,
    reasoning_defect,
    strip_defect_markers,
    to_messages,
)

# Full renders are capped here only for the *raw* token statistic.  The
# trained contribution is always ``min(raw, max_length)``, so the cap is raised
# to ``max_length`` whenever the caller asks for a longer sequence.
_RENDER_CAP = 65_536
REASONING_TOKEN_SHARE = 0.75
REASONING_SHARE_TOLERANCE = 0.01
LENGTH_MODES = ("truncate", "filter")


class Manifest(BaseModel):
    """Provenance and accounting for one generated mixture."""

    sha256: str
    tokens_per_source: dict[str, int]
    rows_per_source: dict[str, int]
    dropped: dict[str, int]
    licenses: dict[str, str]
    target_tokens: int = 0
    seed: int = 0
    ngram_n: int = 13
    tokenizer: str = ""
    tokenizer_version: str = ""
    chat_template_sha256: str = ""
    weights: dict[str, float] = Field(default_factory=dict)
    revisions: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    max_length: int = 0
    length_mode: str = "truncate"
    raw_tokens_per_source: dict[str, int] = Field(default_factory=dict)
    trained_tokens_per_source: dict[str, int] = Field(default_factory=dict)
    requested_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    achieved_weights: dict[str, float] = Field(default_factory=dict)
    reasoning_sources: list[str] = Field(default_factory=list)
    requested_reasoning_share: float = 0.0
    achieved_reasoning_share: float = 0.0
    target_reasoning_share: float = 0.0
    allow_low_reasoning: bool = False
    unavailable_sources: dict[str, str] = Field(default_factory=dict)


def _count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = encode(text)
        return len(token_ids)
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    if isinstance(encoded, dict):
        return len(encoded["input_ids"])
    return len(encoded)


def message_tokens(tokenizer, messages: list[dict[str, Any]]) -> int:
    """Token count used for quota accounting and the manifest."""

    total = 0
    for message in messages:
        total += _count_tokens(tokenizer, str(message.get("content") or ""))
        total += _count_tokens(tokenizer, str(message.get("reasoning") or ""))
    return total


def _rendered_token_counts(
    tokenizer,
    messages: list[dict[str, Any]],
    max_length: int,
    tools: Any = None,
) -> tuple[int, int, bool]:
    """Return ``(raw, trained, trainable)`` for one row.

    Rendering goes through :func:`gemmaloop.train.labels.assistant_labels`, the
    exact path the trainer uses, so the accounting matches the training
    sequences.  ``raw`` is the full rendered length (capped at a large limit),
    ``trained`` is ``min(raw, max_length)`` and ``trainable`` is False when
    truncation leaves no assistant label (the trainer drops such rows).
    """

    from gemmaloop.train.labels import assistant_labels

    render_limit = max(int(max_length), _RENDER_CAP)
    output = assistant_labels(tokenizer, messages, render_limit, tools=tools)
    raw = len(output["input_ids"])
    trained = min(raw, int(max_length))
    trainable = any(
        label != -100 for label in output["labels"][: int(max_length)]
    )
    return raw, trained, trainable


def _reasoning_share(weights: dict[str, float], reasoning: set[str]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(weights.get(name, 0.0) for name in reasoning) / total


def _rebalance_weights(
    requested: dict[str, float],
    available: set[str],
    reasoning: set[str],
) -> tuple[dict[str, float], dict[str, str], float]:
    """Re-normalise available weights so reasoning sources keep >= 75% tokens.

    Unavailable sources are reported and their weight is redistributed to the
    available reasoning sources.  The returned weights sum to one (when any
    source is available), so a larger target remains a prefix of a smaller one
    for the same ``(max_length, length_mode)``.
    """

    unavailable = {
        name: "no row fits max_length"
        for name, weight in requested.items()
        if weight > 0 and name not in available
    }
    available_reasoning = [
        name
        for name in requested
        if name in available and name in reasoning and requested[name] > 0
    ]
    if not available_reasoning:
        if any(requested.get(name, 0.0) > 0 for name in reasoning):
            starved = sorted(
                name
                for name in reasoning
                if requested.get(name, 0.0) > 0 and name not in available
            )
            raise ValueError(
                "no reasoning source can contribute at this max_length; "
                f"starved source(s): {', '.join(starved)}. "
                "Cannot reach the 75% reasoning-token share."
            )
        total = sum(requested[name] for name in available)
        if total <= 0:
            raise ValueError("no available source has a positive weight")
        effective = {
            name: (requested[name] / total if name in available else 0.0)
            for name in requested
        }
        return effective, unavailable, 0.0

    base = {
        name: (requested[name] if name in available else 0.0)
        for name in requested
    }
    unavailable_weight = sum(
        requested[name] for name in requested if name not in available
    )
    if unavailable_weight > 0:
        reasoning_requested = sum(requested[name] for name in available_reasoning)
        if reasoning_requested > 0:
            for name in available_reasoning:
                base[name] += (
                    unavailable_weight * requested[name] / reasoning_requested
                )
        else:
            share = unavailable_weight / len(available_reasoning)
            for name in available_reasoning:
                base[name] += share

    reasoning_total = sum(base[name] for name in available_reasoning)
    other_names = [
        name for name in available if name not in reasoning and base[name] > 0
    ]
    other_total = sum(base[name] for name in other_names)
    total = reasoning_total + other_total
    if total <= 0:
        raise ValueError("no available source has a positive weight")

    current_share = reasoning_total / total
    target = max(REASONING_TOKEN_SHARE, current_share)
    if other_total == 0:
        target = 1.0
    elif target > current_share:
        reasoning_scale = target * total / reasoning_total
        other_scale = (1.0 - target) * total / other_total
        for name in available_reasoning:
            base[name] *= reasoning_scale
        for name in other_names:
            base[name] *= other_scale

    normalised = sum(base.values())
    effective = {
        name: (base[name] / normalised if normalised > 0 else 0.0)
        for name in requested
    }
    return effective, unavailable, target


def _contamination(
    eval_index: EvalIndex | None, messages: list[dict[str, Any]]
) -> str | None:
    if eval_index is None:
        return None
    for message in messages:
        for key in ("content", "reasoning"):
            value = message.get(key)
            if not value:
                continue
            verdict = eval_index.check(str(value))
            if verdict is not None:
                return verdict
    return None


def serialize_row(row: dict[str, Any]) -> str:
    """The exact JSON line encoding used by :func:`write_jsonl`."""

    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows atomically using the canonical line encoding."""

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(serialize_row(row))
            handle.write("\n")
    os.replace(tmp, path)


def _manifest_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(serialize_row(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _accepted_rows(
    name: str,
    spec: SourceSpec,
    tokenizer,
    eval_index: EvalIndex | None,
    *,
    seed: int,
    limit_rows: int | None,
    stream_shuffle_buffer: int,
    max_length: int | None,
    length_mode: str,
    dropped: dict[str, int],
):
    """Yield ``(trained_before_row, messages, raw_tokens, trained_tokens)``.

    ``trained_before_row`` is the cumulative *trained* token count (i.e. the
    sum of ``min(rendered_len, max_length)``) before this row, which is the
    interleave key.  With ``max_length=None`` the legacy quota accounting is
    used so callers that do not opt into length-aware targets keep working.

    In ``filter`` mode rows whose full render exceeds ``max_length`` are
    dropped (``too_long``) instead of truncated.
    """

    used = 0
    for row in iter_rows(
        spec,
        seed=seed,
        limit_rows=limit_rows,
        shuffle_buffer=stream_shuffle_buffer,
    ):
        if spec.row_filter is not None:
            criterion = spec.row_filter(row)
            if criterion is not None:
                dropped[criterion] += 1
                continue

        messages = to_messages(row, name)
        defect = reasoning_defect(messages)
        if defect is not None:
            dropped[defect] += 1
            continue
        strip_defect_markers(messages)
        if not messages or not any(
            message.get("content") or message.get("reasoning")
            for message in messages
        ):
            dropped["empty"] += 1
            continue

        verdict = _contamination(eval_index, messages)
        if verdict is not None:
            dropped[verdict] += 1
            continue

        if max_length is None:
            raw = trained = message_tokens(tokenizer, messages)
        else:
            raw, trained, trainable = _rendered_token_counts(
                tokenizer, messages, max_length, tools=row.get("tools")
            )
            if length_mode == "filter":
                if raw > max_length:
                    dropped["too_long"] += 1
                    continue
                trained = raw
            if not trainable:
                dropped["no_labels"] += 1
                continue

        yield used, messages, raw, trained
        used += trained


def build_mix(
    target_tokens: int,
    tokenizer,
    eval_index: EvalIndex | None,
    seed: int = 0,
    *,
    sources: dict[str, SourceSpec] | None = None,
    limit_rows: int | None = None,
    stream_shuffle_buffer: int = 0,
    max_length: int | None = None,
    length_mode: str = "truncate",
    allow_low_reasoning: bool = False,
) -> tuple[list[dict[str, Any]], Manifest]:
    """Build a prefix-stable, token-weighted mixture and its manifest.

    ``sources`` defaults to :data:`SOURCES`; tests inject local specs with a
    ``loader``.  ``limit_rows`` bounds how many rows are *scanned* per source,
    which is useful for smoke runs.

    When ``max_length`` is given, each row contributes
    ``min(rendered_len, max_length)`` toward ``target_tokens``.  ``rendered_len``
    comes from :func:`gemmaloop.train.labels.assistant_labels`, the same render
    the trainer uses, so the target is a *trained-token* target.  The manifest
    records the raw and trained totals per source.

    ``length_mode="truncate"`` keeps the start of long rows;
    ``length_mode="filter"`` drops rows whose full render exceeds ``max_length``
    and rebalances weights so reasoning sources supply at least 75% of trained
    tokens.  With ``max_length=None`` the legacy quota-token accounting is used.
    """

    if target_tokens < 0:
        raise ValueError("target_tokens must be non-negative")
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length!r}")
    if length_mode not in LENGTH_MODES:
        raise ValueError(
            f"length_mode must be one of {LENGTH_MODES}, got {length_mode!r}"
        )
    if length_mode == "filter" and max_length is None:
        raise ValueError("length_mode='filter' requires max_length")

    chosen_sources = sources if sources is not None else SOURCES
    dropped: dict[str, int] = defaultdict(int)
    tokens_per_source: dict[str, int] = {}
    raw_tokens_per_source: dict[str, int] = {}
    trained_tokens_per_source: dict[str, int] = {}
    rows_per_source: dict[str, int] = {}
    licenses: dict[str, str] = {}
    requested_weights: dict[str, float] = {}
    revisions: dict[str, str] = {}
    provenance: dict[str, str] = {}
    reasoning_names = {
        name for name, spec in chosen_sources.items() if spec.reasoning
    }

    # One live generator per source.  Each yields accepted rows in its own
    # stream order together with the trained token count accumulated before it.
    # Peeking one row also tells us which sources can contribute at this
    # ``(max_length, length_mode)``; that set fixes the effective weights, so a
    # bigger target remains a prefix of a smaller one.
    streams: dict[str, Any] = {}
    first_rows: dict[str, Any] = {}
    available: set[str] = set()

    for name, spec in chosen_sources.items():
        licenses[name] = spec.license
        requested_weights[name] = float(spec.weight)
        revisions[name] = spec.revision or ""
        provenance[name] = spec.provenance
        tokens_per_source[name] = 0
        raw_tokens_per_source[name] = 0
        trained_tokens_per_source[name] = 0
        rows_per_source[name] = 0

        if spec.weight <= 0:
            continue
        stream = _accepted_rows(
            name,
            spec,
            tokenizer,
            eval_index,
            seed=seed,
            limit_rows=limit_rows,
            stream_shuffle_buffer=stream_shuffle_buffer,
            max_length=max_length,
            length_mode=length_mode,
            dropped=dropped,
        )
        streams[name] = stream
        try:
            first_rows[name] = next(stream)
        except StopIteration:
            continue
        available.add(name)

    if length_mode == "filter":
        effective_weights, unavailable, target_share = _rebalance_weights(
            requested_weights, available, reasoning_names
        )
    else:
        effective_weights = {
            name: (requested_weights[name] if name in available else 0.0)
            for name in requested_weights
        }
        unavailable = {
            name: "no accepted rows"
            for name, weight in requested_weights.items()
            if weight > 0 and name not in available
        }
        target_share = _reasoning_share(requested_weights, reasoning_names)

    heap: list[tuple[float, int, str, list[dict[str, Any]], int, int]] = []
    for index, (name, spec) in enumerate(chosen_sources.items()):
        if name not in first_rows:
            continue
        weight = effective_weights.get(name, 0.0)
        if weight <= 0:
            continue
        start, messages, raw, trained = first_rows[name]
        heapq.heappush(
            heap,
            (start / weight, index, name, messages, raw, trained),
        )

    rows: list[dict[str, Any]] = []
    while heap:
        key, index, name, messages, raw, trained = heap[0]
        if key >= target_tokens:
            break
        heapq.heappop(heap)

        rows.append({"messages": messages, "source": name})
        tokens_per_source[name] += trained
        raw_tokens_per_source[name] += raw
        trained_tokens_per_source[name] += trained
        rows_per_source[name] += 1

        try:
            start, next_messages, next_raw, next_trained = next(streams[name])
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                start / effective_weights[name],
                index,
                name,
                next_messages,
                next_raw,
                next_trained,
            ),
        )

    try:
        import transformers

        version = transformers.__version__
    except Exception:  # pragma: no cover - transformers is a hard dependency
        version = ""

    total_trained = sum(trained_tokens_per_source.values())
    achieved_weights = {
        name: (
            trained_tokens_per_source[name] / total_trained
            if total_trained > 0
            else 0.0
        )
        for name in chosen_sources
    }
    achieved_reasoning = (
        sum(trained_tokens_per_source[name] for name in reasoning_names)
        / total_trained
        if total_trained > 0
        else 0.0
    )
    if (
        length_mode == "filter"
        and not allow_low_reasoning
        and achieved_reasoning < target_share - REASONING_SHARE_TOLERANCE
    ):
        starved = sorted(
            name
            for name in reasoning_names
            if name in unavailable
            or trained_tokens_per_source.get(name, 0) == 0
        )
        if not starved:
            starved = sorted(
                reasoning_names,
                key=lambda name: trained_tokens_per_source.get(name, 0),
            )
        raise ValueError(
            "filter mode reasoning share "
            f"{achieved_reasoning:.1%} < target {target_share:.1%}; "
            f"starved source(s): {', '.join(starved)}. "
            "Pass --allow-low-reasoning to build anyway."
        )

    manifest = Manifest(
        sha256=_manifest_hash(rows),
        tokens_per_source=tokens_per_source,
        rows_per_source=rows_per_source,
        dropped=dict(sorted(dropped.items())),
        licenses=licenses,
        target_tokens=int(target_tokens),
        seed=int(seed),
        ngram_n=int(getattr(eval_index, "n", 13)),
        tokenizer=tokenizer_name(tokenizer),
        tokenizer_version=version,
        chat_template_sha256=chat_template_sha256(tokenizer),
        weights=requested_weights,
        revisions=revisions,
        provenance=provenance,
        max_length=int(max_length) if max_length is not None else 0,
        length_mode=length_mode,
        raw_tokens_per_source=raw_tokens_per_source,
        trained_tokens_per_source=trained_tokens_per_source,
        requested_weights=requested_weights,
        effective_weights=effective_weights,
        achieved_weights=achieved_weights,
        reasoning_sources=sorted(reasoning_names),
        requested_reasoning_share=_reasoning_share(
            requested_weights, reasoning_names
        ),
        achieved_reasoning_share=achieved_reasoning,
        target_reasoning_share=target_share,
        allow_low_reasoning=bool(allow_low_reasoning),
        unavailable_sources=unavailable,
    )
    return rows, manifest

__all__ = [
    "Manifest",
    "build_mix",
    "message_tokens",
    "serialize_row",
    "write_jsonl",
]
