"""Local Stage-0-lite sampler support (registered engine: padded HF).

The full Stage 0 job runs on Modal with stock vLLM.  The pilot needs the same
registered data earlier and locally, so this module re-implements the *data*
identity of that job on the registered padded evaluation engine:

* the same pool and deterministic bucket splits,
* the registered sampling (T=1.0, top_p=0.95, top_k=64, thinking on) and four
  samples per prompt,
* the repo's canonical per-request identity ``"{problem_id}:{sample_index}"``
  (the eval runner's ``sha256(seed:item:sample)`` scheme; the Modal job writes
  the same ``(problem_id, sample_index)`` pair into every record),
* a per-request generation budget of ``max_length - rendered_prompt_tokens``,
  so the registered ``finish_reason = stop`` / ``rendered <= 8192`` filter keeps
  exactly the samples a 32,768-token run would keep.  A request whose natural
  stop length is below its budget is unaffected by the cap; a request that would
  run past the 8192 rendered limit is dropped either way.  Calls may batch
  requests with different budgets: the engine cap is the largest per-request
  budget in the call, so no request is cut before its own budget.
* resumable, per-request persistence in the Modal shard layout, with atomic
  shard finalisation and a manifest carrying the pool SHA, buckets, request
  ids, caps, token counts, acceptance rates and per-file SHA-256 values.

The module is pure Python (no model is loaded) so the CPU tests can drive it
with a fake generator and fake tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from gemmaloop.stage0.postprocess import (
    MAX_LENGTH,
    MIN_REASONING_SHARE,
    keep_reason,
    reasoning_share,
    sample_messages,
)

ENGINE = "local-padded"
MODEL_ID = "google/gemma-4-E2B-it"
SAMPLES_PER_PROMPT = 4
REGISTERED_MAX_NEW_TOKENS = 32768
REGISTERED_MAX_LENGTH = 8192
REGISTERED_MIN_REASONING_SHARE = 0.75
REGISTERED_TRAIN_TOKENS = 3_000_000
# Version of the registered selection order recorded in the run marker.  Bump
# this whenever the order changes so an old output directory cannot be resumed.
SELECTION_ORDER_VERSION = "s0lite-selection-v1-request-hash"
SELECTION_ORDER_RULE = (
    "sha256(seed:request_id) ascending over the whole bucket"
)


class RunIdentityError(ValueError):
    """A shard directory belongs to a different run/plan/verifier identity."""

SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_new_tokens": REGISTERED_MAX_NEW_TOKENS,
    "enable_thinking": True,
}

DEFAULT_SHARD_SIZE = 25  # nominal prompt granularity; requests/shard = this * samples
PROMPT_BIN_WIDTH = 256  # batch requests with similar prompt lengths together

_THOUGHT_RE = re.compile(r"<\|channel>thought\b(.*?)<channel\|>", re.DOTALL)


# ---------------------------------------------------------------------------
# Identity conventions
# ---------------------------------------------------------------------------
def per_problem_seed(problem_id: str) -> int:
    """The Modal job's per-prompt vLLM seed for ``problem_id``.

    ``scripts/modal_stage0.py`` seeds one request per problem with the first
    eight hex digits of ``sha256(problem_id)``.  Reference only: the local
    engine uses ``generate_batch``'s per-request seed
    ``sha256(seed:request_id)``, not this per-problem Modal seed; they identify
    the same ``(problem_id, sample_index)`` records but do not produce the same
    token stream.
    """

    return int(
        hashlib.sha256(str(problem_id).encode("utf-8")).hexdigest()[:8], 16
    )


def stage0_request_id(problem_id: str, sample_index: int) -> str:
    """Stable per-request identity, identical to the Modal record pair.

    The eval runner names every request ``f"{item_id}:{sample_idx}"`` and seeds
    it with ``sha256(seed:request_id)``.  The Modal Stage 0 records carry the
    same ``problem_id`` and ``sample_index``; this is their canonical identity.
    """

    return f"{problem_id}:{int(sample_index)}"


def parse_stage0_request_id(request_id: str) -> tuple[str, int]:
    """Inverse of :func:`stage0_request_id` (problem ids may contain ``:``)."""

    problem_id, separator, sample = str(request_id).rpartition(":")
    if not separator or not sample.isdigit():
        raise ValueError(f"not a Stage 0 request id: {request_id!r}")
    return problem_id, int(sample)


def compute_run_id(
    pool_manifest_sha256: str,
    *,
    shard_size: int,
    samples_per_prompt: int,
    model_revision: str,
    sampling: Mapping[str, Any] | None = None,
) -> str:
    """Run namespace, same basis as the Modal job's ``compute_run_id``."""

    payload = {
        "pool_manifest_sha256": str(pool_manifest_sha256),
        "shard_size": int(shard_size),
        "samples_per_prompt": int(samples_per_prompt),
        "model_revision": str(model_revision),
        "sampling": dict(sampling or SAMPLING),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Budgets and batching
# ---------------------------------------------------------------------------
def prompt_budget(max_length: int, prompt_tokens: int) -> int:
    """Per-request ``max_new_tokens``: the registered cap minus the prompt."""

    return int(max_length) - int(prompt_tokens)


def _max_batch_for(cap: int, model_size: str = "e2b") -> int:
    from gemmaloop.serve.generate import max_batch_for

    return int(max_batch_for(int(cap), model_size))


@dataclass(frozen=True)
class Stage0Request:
    """One sample request (one of the four samples for a problem)."""

    problem: Mapping[str, Any]
    sample_index: int
    prompt_tokens: int
    budget: int

    @property
    def problem_id(self) -> str:
        return str(self.problem["problem_id"])

    @property
    def request_id(self) -> str:
        return stage0_request_id(self.problem_id, self.sample_index)

    @property
    def prompt(self) -> str:
        return str(self.problem.get("prompt") or "")


@dataclass(frozen=True)
class RequestCall:
    """One ``generate_batch`` call: requests sharing an engine cap."""

    requests: tuple[Stage0Request, ...]
    max_new_tokens: int
    max_batch: int

    @property
    def request_ids(self) -> list[str]:
        return [request.request_id for request in self.requests]


def plan_request_calls(
    requests: Sequence[Stage0Request],
    *,
    max_batch_override: int | None = None,
    model_size: str = "e2b",
    bin_width: int = PROMPT_BIN_WIDTH,
) -> list[RequestCall]:
    """Group requests into deterministic, length-similar generation calls.

    Requests are binned by rendered prompt length (so a call's engine cap is at
    most ``bin_width`` tokens above any member's own budget) and chunked at the
    measured non-spilling batch for that cap.  The engine cap is the *largest*
    per-request budget in the call: no request is truncated before its own
    registered budget, so the accepted set is unchanged.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    ordered = sorted(
        requests,
        key=lambda request: (
            int(request.prompt_tokens) // int(bin_width),
            int(request.prompt_tokens),
            request.problem_id,
            request.sample_index,
        ),
    )
    calls: list[RequestCall] = []
    for _key, group in _group_by(
        ordered, lambda request: int(request.prompt_tokens) // int(bin_width)
    ):
        cap = max(int(request.budget) for request in group)
        batch = (
            int(max_batch_override)
            if max_batch_override is not None
            else _max_batch_for(cap, model_size)
        )
        if batch < 1:
            raise ValueError("max_batch must be positive")
        for start in range(0, len(group), batch):
            chunk = tuple(group[start : start + batch])
            calls.append(
                RequestCall(
                    requests=chunk,
                    max_new_tokens=cap,
                    max_batch=batch,
                )
            )
    return calls


def _group_by(items: Iterable[Any], key: Callable[[Any], Any]):
    current_key = object()
    group: list[Any] = []
    for item in items:
        item_key = key(item)
        if item_key != current_key and group:
            yield current_key, group
            group = []
        current_key = item_key
        group.append(item)
    if group:
        yield current_key, group


def build_requests(
    problems: Sequence[Mapping[str, Any]],
    *,
    samples_per_prompt: int,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int],
    max_length: int,
    done_request_ids: Iterable[str] = (),
) -> tuple[list[Stage0Request], dict[str, str]]:
    """Plan the not-yet-finished requests for a shard.

    Returns ``(requests, skipped)`` where ``skipped`` maps a problem id to a
    reason when no sample can pass the registered filter (for example a prompt
    that already renders longer than ``max_length``).
    """

    done = set(done_request_ids)
    requests: list[Stage0Request] = []
    skipped: dict[str, str] = {}
    for problem in problems:
        problem_id = str(problem["problem_id"])
        prompt_tokens = int(prompt_tokens_for(problem))
        budget = prompt_budget(max_length, prompt_tokens)
        expected = [
            stage0_request_id(problem_id, index)
            for index in range(int(samples_per_prompt))
        ]
        if budget < 1:
            if not all(request_id in done for request_id in expected):
                skipped[problem_id] = "prompt_too_long"
            continue
        for index in range(int(samples_per_prompt)):
            request_id = stage0_request_id(problem_id, index)
            if request_id in done:
                continue
            requests.append(
                Stage0Request(
                    problem=problem,
                    sample_index=index,
                    prompt_tokens=prompt_tokens,
                    budget=budget,
                )
            )
    return requests, skipped


def build_requests_from_refs(
    refs: Sequence["RequestRef"],
    *,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int],
    max_length: int,
    done_request_ids: Iterable[str] = (),
) -> tuple[list[Stage0Request], dict[str, str]]:
    """Plan the not-yet-finished requests from the hash-ordered refs."""

    done = set(done_request_ids)
    requests: list[Stage0Request] = []
    skipped: dict[str, str] = {}
    prompt_cache: dict[str, int] = {}
    for ref in refs:
        problem_id = ref.problem_id
        if problem_id in skipped:
            continue
        if problem_id not in prompt_cache:
            prompt_cache[problem_id] = int(prompt_tokens_for(ref.problem))
        budget = prompt_budget(max_length, prompt_cache[problem_id])
        if budget < 1:
            skipped[problem_id] = "prompt_too_long"
            continue
        if ref.request_id in done:
            continue
        requests.append(
            Stage0Request(
                problem=ref.problem,
                sample_index=ref.sample_index,
                prompt_tokens=prompt_cache[problem_id],
                budget=budget,
            )
        )
    return requests, skipped


# ---------------------------------------------------------------------------
# Raw records (Modal shard format)
# ---------------------------------------------------------------------------
def parse_completion(text: str) -> dict[str, Any]:
    """Split a raw completion into reasoning/content/thought-closed.

    Byte-identical to ``scripts/modal_stage0.parse_completion`` so a local
    record and a Modal record parse the same way.
    """

    text = text or ""
    match = _THOUGHT_RE.search(text)
    reasoning = match.group(1).strip() if match else ""
    content = _THOUGHT_RE.sub("", text).strip()
    content = re.sub(r"(?:<turn\|>|<eos>|<end_of_turn>)+$", "", content).strip()
    has_open = "<|channel>thought" in text
    has_close = "<channel|>" in text
    thought_closed = bool(match) or (not has_open) or (has_open and has_close)
    return {
        "reasoning": reasoning,
        "content": content,
        "thought_closed": thought_closed,
    }


def build_sample_record(
    *,
    problem: Mapping[str, Any],
    sample_index: int,
    request: Stage0Request | None,
    output: Any,
    run_id: str,
    pool_manifest_sha256: str,
    shard_index: int,
    prompt_tokens: int,
    budget: int,
    engine_cap: int,
    model_revision: str,
    decode: Callable[[Sequence[int]], str] | None = None,
    samples_per_prompt: int = SAMPLES_PER_PROMPT,
    max_length: int = REGISTERED_MAX_LENGTH,
    engine: str = ENGINE,
    gpu: str = "",
    seed: int | None = None,
    engine_version: str | None = None,
) -> dict[str, Any]:
    """Build one raw sample record in the Modal Stage 0 shard format.

    ``engine``/``gpu``/``seed``/``engine_version`` let another backend (the
    Modal vLLM lite sampler) write the identical schema.  The optional fields
    are only added when supplied, so local records are unchanged.
    """

    token_ids = getattr(output, "token_ids", None)
    reasoning = getattr(output, "reasoning", "") or ""
    content = getattr(output, "text", "") or ""
    if token_ids is not None and decode is not None:
        raw = decode(token_ids)
        parsed = parse_completion(raw)
        reasoning = parsed["reasoning"]
        content = parsed["content"]
        thought_closed = parsed["thought_closed"]
    else:
        raw = str(getattr(output, "text", "") or "")
        thought_closed = True
    sample_index = int(sample_index)
    request_id = stage0_request_id(str(problem["problem_id"]), sample_index)
    record = {
        "problem_id": str(problem["problem_id"]),
        "sample_index": sample_index,
        "request_id": request_id,
        "prompt": str(problem.get("prompt") or ""),
        "completion": str(raw),
        "reasoning": str(reasoning),
        "content": str(content),
        "finish_reason": str(getattr(output, "finish_reason", "")),
        "thought_closed": bool(thought_closed),
        "tokens": int(
            len(token_ids)
            if token_ids is not None
            else getattr(output, "completion_tokens", 0)
        ),
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(budget),
        "engine_cap": int(engine_cap),
        "engine": str(engine),
        "model_id": MODEL_ID,
        "model_revision": str(model_revision),
        "run_id": str(run_id),
        "pool_manifest_sha256": str(pool_manifest_sha256),
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "top_k": SAMPLING["top_k"],
        "samples_per_prompt": int(samples_per_prompt),
        "max_model_len": int(max_length),
        "max_prompt_tokens": int(prompt_tokens),
        "skip_special_tokens": False,
        "gpu": str(gpu),
        "shard_index": int(shard_index),
    }
    if seed is not None:
        record["seed"] = int(seed)
    if engine_version is not None:
        record["engine_version"] = str(engine_version)
    return record


def generate_calls(
    requests: Sequence[Stage0Request],
    *,
    generate_fn: Callable[..., Sequence[Any]],
    model: Any,
    processor: Any,
    decode: Callable[[Sequence[int]], str] | None,
    run_id: str,
    pool_manifest_sha256: str,
    shard_index: int,
    model_revision: str,
    seed: int = 0,
    samples_per_prompt: int = SAMPLES_PER_PROMPT,
    max_length: int = REGISTERED_MAX_LENGTH,
    max_batch_override: int | None = None,
) -> list[dict[str, Any]]:
    """Run every call and return the new raw records."""

    from gemmaloop.serve.generate import Sampling

    records: list[dict[str, Any]] = []
    for call in plan_request_calls(requests, max_batch_override=max_batch_override):
        conversations = [
            [{"role": "user", "content": request.prompt}] for request in call.requests
        ]
        sampling = Sampling(
            temperature=float(SAMPLING["temperature"]),
            top_p=float(SAMPLING["top_p"]),
            top_k=int(SAMPLING["top_k"]),
            max_new_tokens=int(call.max_new_tokens),
            enable_thinking=bool(SAMPLING["enable_thinking"]),
        )
        outputs = list(
            generate_fn(
                model,
                processor,
                conversations,
                sampling,
                seed=int(seed),
                request_ids=call.request_ids,
                max_batch=int(call.max_batch),
            )
        )
        if len(outputs) != len(call.requests):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for "
                f"{len(call.requests)} requests"
            )
        for request, output in zip(call.requests, outputs):
            records.append(
                build_sample_record(
                    problem=request.problem,
                    sample_index=request.sample_index,
                    request=request,
                    output=output,
                    run_id=run_id,
                    pool_manifest_sha256=pool_manifest_sha256,
                    shard_index=shard_index,
                    prompt_tokens=request.prompt_tokens,
                    budget=request.budget,
                    engine_cap=call.max_new_tokens,
                    model_revision=model_revision,
                    decode=decode,
                    samples_per_prompt=samples_per_prompt,
                    max_length=max_length,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Registered filter and verification
# ---------------------------------------------------------------------------
@dataclass
class SampleVerdict:
    kept: bool
    drop_reason: str | None
    rendered_tokens: int
    reasoning_share: float
    reasoning_tokens: int
    answer_tokens: int
    assistant_tokens: int
    correct: bool | None = None
    grade: dict[str, Any] | None = None
    verify_duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "drop_reason": self.drop_reason,
            "rendered_tokens": self.rendered_tokens,
            "reasoning_share": self.reasoning_share,
            "reasoning_tokens": self.reasoning_tokens,
            "answer_tokens": self.answer_tokens,
            "assistant_tokens": self.assistant_tokens,
            "correct": self.correct,
            "verify_duration_s": self.verify_duration_s,
        }


def _count_tokens(text: str, counter: Callable[[str], int] | None) -> int:
    if not text:
        return 0
    if counter is not None:
        return int(counter(text))
    return max(1, (len(text) + 3) // 4)


def _rendered_tokens(
    sample: Mapping[str, Any],
    problem: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
    max_length: int = MAX_LENGTH,
) -> int:
    if renderer is not None:
        return int(renderer(sample_messages(dict(sample), dict(problem))))
    if tokenizer is not None:
        from gemmaloop.train.labels import assistant_labels

        output = assistant_labels(
            tokenizer,
            sample_messages(dict(sample), dict(problem)),
            int(max_length) + 1,
        )
        return len(output["input_ids"])
    stored = sample.get("rendered_tokens")
    if stored is not None:
        return int(stored)
    return int(sample.get("prompt_tokens") or 0) + int(sample.get("tokens") or 0)


def evaluate_sample(
    sample: Mapping[str, Any],
    problem: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    token_counter: Callable[[str], int] | None = None,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
    max_length: int = MAX_LENGTH,
    min_reasoning_share: float = MIN_REASONING_SHARE,
    verify_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
) -> SampleVerdict:
    """Apply the registered Stage-0 filter, then (optionally) verify."""

    rendered = _rendered_tokens(
        sample,
        problem,
        tokenizer=tokenizer,
        renderer=renderer,
        max_length=max_length,
    )
    reason = keep_reason(
        dict(sample),
        dict(problem),
        token_counter=token_counter,
        renderer=lambda _messages: rendered,
        max_length=max_length,
        min_reasoning_share=min_reasoning_share,
    )
    reasoning_tokens = _count_tokens(str(sample.get("reasoning") or ""), token_counter)
    answer_tokens = _count_tokens(str(sample.get("content") or ""), token_counter)
    correct: bool | None = None
    grade: dict[str, Any] | None = None
    verify_duration_s = 0.0
    if reason is None and verify_fn is not None:
        started = time.perf_counter()
        result = verify_fn(sample, problem)
        verify_duration_s = time.perf_counter() - started
        if isinstance(result, tuple) and len(result) == 2:
            correct = None if result[0] is None else bool(result[0])
            grade = result[1]
        else:  # pragma: no cover - defensive
            raise TypeError("verify_fn must return (correct, grade)")
    return SampleVerdict(
        kept=reason is None,
        drop_reason=reason,
        rendered_tokens=rendered,
        reasoning_share=reasoning_share(dict(sample), token_counter),
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
        assistant_tokens=reasoning_tokens + answer_tokens,
        correct=correct,
        grade=grade,
        verify_duration_s=float(verify_duration_s),
    )


def sample_grade_row(
    sample: Mapping[str, Any],
    problem: Mapping[str, Any],
    verdict: SampleVerdict,
) -> dict[str, Any]:
    """Per-sample verification provenance for ``sample_grades.jsonl``."""

    grade = verdict.grade or {}
    decided_by = grade.get("decided_by")
    if decided_by is None and verdict.correct is not None:
        decided_by = (
            "code_grader"
            if str(problem.get("kind") or "") == "code"
            else "normaliser"
        )
    if decided_by == "both":
        verifiers = ["math_verify", "normaliser"]
    elif decided_by == "math_verify":
        verifiers = ["math_verify"]
    elif decided_by == "normaliser":
        verifiers = ["normaliser"]
    elif decided_by == "timeout":
        verifiers = ["normaliser"]
    elif decided_by == "code_grader":
        verifiers = ["code_grader"]
    else:
        verifiers = []
    return {
        "problem_id": str(sample["problem_id"]),
        "sample_index": int(sample.get("sample_index") or 0),
        "request_id": str(
            sample.get("request_id")
            or stage0_request_id(
                str(sample["problem_id"]), int(sample.get("sample_index") or 0)
            )
        ),
        "kind": problem.get("kind"),
        "source": problem.get("source"),
        "split": problem.get("split"),
        "kept": bool(verdict.kept),
        "drop_reason": verdict.drop_reason,
        "correct": verdict.correct,
        "decided_by": decided_by,
        "verifiers": verifiers,
        "timed_out": bool(grade.get("timed_out", False)),
        "timing_s": round(float(verdict.verify_duration_s), 6),
        "rendered_tokens": int(verdict.rendered_tokens),
        "reasoning_tokens": int(verdict.reasoning_tokens),
        "answer_tokens": int(verdict.answer_tokens),
        "assistant_tokens": int(verdict.assistant_tokens),
        "grade": grade,
    }


def trace_record(
    sample: Mapping[str, Any],
    problem: Mapping[str, Any],
    verdict: SampleVerdict,
) -> dict[str, Any]:
    """The verified-trace record written by Stage 0 post-processing."""

    return {
        "problem_id": str(sample["problem_id"]),
        "source": problem.get("source"),
        "kind": problem.get("kind"),
        "split": problem.get("split"),
        "pilot_subsplit": problem.get("pilot_subsplit"),
        "sample_index": int(sample.get("sample_index") or 0),
        "messages": sample_messages(dict(sample), dict(problem)),
        "rendered_tokens": int(verdict.rendered_tokens),
        "reasoning_share": float(verdict.reasoning_share),
        "engine": sample.get("engine"),
        "model_revision": sample.get("model_revision"),
    }


def heal_row(
    sample: Mapping[str, Any],
    problem: Mapping[str, Any],
    verdict: SampleVerdict,
) -> dict[str, Any]:
    """A heal row: the ``messages`` schema ``precompute_teacher_topk`` reads."""

    return {
        "messages": sample_messages(dict(sample), dict(problem)),
        "problem_id": str(sample["problem_id"]),
        "source": problem.get("source"),
        "kind": problem.get("kind"),
        "split": problem.get("split"),
        "sample_index": int(sample.get("sample_index") or 0),
        "rendered_tokens": int(verdict.rendered_tokens),
        "assistant_tokens": int(verdict.assistant_tokens),
        "engine": sample.get("engine"),
        "model_revision": sample.get("model_revision"),
    }


# ---------------------------------------------------------------------------
# Targets from the pre-registration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    name: str
    low: int
    high: int
    problems: int
    token_target: int = 0
    verified: bool = True
    rows_target: int = 0
    # 0 means "use the run-level samples_per_prompt"; a target that needs a
    # different draw count (the held-out MONITOR corpus: 1) sets it explicitly.
    samples_per_prompt: int = 0

    @property
    def buckets(self) -> tuple[int, int]:
        return (self.low, self.high)

    def contains(self, bucket: int) -> bool:
        return self.low <= int(bucket) <= self.high


def parse_bucket_range(value: str | int) -> tuple[int, int]:
    text = str(value).strip()
    if "-" in text:
        low, high = text.split("-", 1)
        return int(low), int(high)
    number = int(text)
    return number, number


def targets_from_prereg(prereg: Any) -> list[Target]:
    """The pilot-relevant bucket targets, read from ``prereg.yaml``."""

    splits = prereg.data.splits
    pilot = prereg.pilot
    train_low, train_high = parse_bucket_range(splits.train_bucket)
    p1_low, p1_high = parse_bucket_range(splits.pilot_p1_bucket)
    tfit_low, tfit_high = parse_bucket_range(splits.pilot_tfit_bucket)
    sel_low, sel_high = parse_bucket_range(splits.pilot_selection_bucket)
    return [
        Target(
            "train",
            train_low,
            train_high,
            int(splits.train_problems),
            token_target=int(pilot.data.pilot_train_tokens),
            verified=False,
        ),
        Target(
            "pilot_p1",
            p1_low,
            p1_high,
            int(splits.pilot_p1_problems),
            verified=True,
            rows_target=int(pilot.p1.rows),
        ),
        Target(
            "monitor",
            train_low,
            train_high,
            int(getattr(splits, "monitor_problems", 64)),
            verified=False,
            rows_target=int(getattr(splits, "monitor_problems", 64)),
            samples_per_prompt=int(
                getattr(splits, "monitor_samples_per_prompt", 1)
            ),
        ),
        Target(
            "pilot_tfit",
            tfit_low,
            tfit_high,
            int(splits.pilot_tfit_problems),
            verified=True,
            rows_target=0,  # the prereg registers problems, not a row target
        ),
        Target(
            "pilot_selection",
            sel_low,
            sel_high,
            int(splits.pilot_selection_problems),
            verified=True,
            rows_target=0,  # the prereg registers problems, not a row target
        ),
    ]


def bucket_of(problem: Mapping[str, Any]) -> int:
    from gemmaloop.stage0.pool import bucket_for

    if problem.get("bucket") is not None:
        return int(problem["bucket"])
    return int(bucket_for(str(problem["problem_id"])))


def target_for_bucket(bucket: int, targets: Sequence[Target]) -> Target | None:
    for target in targets:
        if target.contains(bucket):
            return target
    return None


def request_order_key(seed: int, request_id: str) -> str:
    """Registered source-neutral order key: ``sha256(seed:request_id)``."""

    return hashlib.sha256(
        f"{int(seed)}:{request_id}".encode("utf-8")
    ).hexdigest()


def problem_order_key(seed: int, problem_id: str) -> str:
    """Registered problem order key: ``sha256(seed:problem_id)``.

    The MONITOR corpus is the largest keys inside the train bucket; every
    training subset takes the smallest keys and excludes the monitor ids.
    """

    return hashlib.sha256(
        f"{int(seed)}:{problem_id}".encode("utf-8")
    ).hexdigest()


def monitor_problem_ids(
    problems: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    *,
    seed: int = 0,
) -> set[str]:
    """The held-out MONITOR problem ids (largest train-bucket order keys)."""
    monitor = next((target for target in targets if target.name == "monitor"), None)
    if monitor is None:
        return set()
    train = next((target for target in targets if target.name == "train"), None)
    low, high = (
        (train.low, train.high) if train is not None else (monitor.low, monitor.high)
    )
    candidates = [
        problem
        for problem in problems
        if low <= bucket_of(problem) <= high
    ]
    ordered = sorted(
        candidates,
        key=lambda problem: (
            problem_order_key(seed, str(problem["problem_id"])),
            str(problem["problem_id"]),
        ),
    )
    count = max(0, int(monitor.problems))
    return {str(problem["problem_id"]) for problem in ordered[-count:]} if count else set()


def target_for_problem(
    problem: Mapping[str, Any],
    targets: Sequence[Target],
    monitor_ids: Iterable[str] | None = None,
) -> Target | None:
    """Resolve a problem to its target, keeping monitor ids out of train."""
    if monitor_ids and str(problem.get("problem_id")) in set(monitor_ids):
        return next((target for target in targets if target.name == "monitor"), None)
    return target_for_bucket(bucket_of(problem), targets)


@dataclass(frozen=True)
class RequestRef:
    """A planned request before its prompt length (and cap) are known."""

    problem: Mapping[str, Any]
    sample_index: int

    @property
    def problem_id(self) -> str:
        return str(self.problem["problem_id"])

    @property
    def request_id(self) -> str:
        return stage0_request_id(self.problem_id, self.sample_index)


def _bucket_refs(
    problems: Sequence[Mapping[str, Any]],
    target: Target,
    *,
    samples_per_prompt: int,
    seed: int,
    monitor_ids: Iterable[str] | None = None,
) -> list[RequestRef]:
    monitor_set = {str(value) for value in (monitor_ids or ())}
    refs = [
        RequestRef(problem, index)
        for problem in problems
        if target.contains(bucket_of(problem))
        and not (
            target.name == "train"
            and str(problem["problem_id"]) in monitor_set
        )
        and not (
            target.name == "monitor"
            and monitor_set
            and str(problem["problem_id"]) not in monitor_set
        )
        for index in range(int(target.samples_per_prompt or samples_per_prompt))
    ]
    if target.name == "monitor":
        refs.sort(
            key=lambda ref: (
                problem_order_key(seed, ref.problem_id),
                ref.problem_id,
                ref.sample_index,
            ),
            reverse=True,
        )
    elif target.name == "train":
        refs.sort(
            key=lambda ref: (
                problem_order_key(seed, ref.problem_id),
                ref.problem_id,
                ref.sample_index,
            )
        )
    else:
        refs.sort(
            key=lambda ref: (request_order_key(seed, ref.request_id), ref.request_id)
        )
    return refs


def _truncate_refs_to_problems(
    refs: Sequence[RequestRef], limit: int
) -> list[RequestRef]:
    """Keep the first ``limit`` distinct problems in request-hash order.

    The request order is the registered ``sha256(seed:request_id)`` ascending
    order over the whole bucket; a problem enters the subset when its earliest
    request appears.  All four samples of a selected problem are kept.
    """

    if int(limit) <= 0:
        return list(refs)
    chosen: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.problem_id in seen:
            continue
        seen.add(ref.problem_id)
        chosen.append(ref.problem_id)
        if len(chosen) >= int(limit):
            break
    keep = set(chosen)
    return [ref for ref in refs if ref.problem_id in keep]


def select_problems(
    problems: Sequence[Mapping[str, Any]],
    target: Target,
    *,
    samples_per_prompt: int = SAMPLES_PER_PROMPT,
    seed: int = 0,
    monitor_ids: Iterable[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Bucket members in the registered source-neutral order, capped.

    The order is the registered ``sha256(seed:request_id)`` ascending order over
    all requests in the bucket, not the source-prefixed problem-id order.
    """

    refs = _bucket_refs(
        problems,
        target,
        samples_per_prompt=samples_per_prompt,
        seed=seed,
        monitor_ids=monitor_ids,
    )
    refs = _truncate_refs_to_problems(refs, int(target.problems))
    ordered: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.problem_id in seen:
            continue
        seen.add(ref.problem_id)
        ordered.append(ref.problem)
    return ordered


# ---------------------------------------------------------------------------
# Shard plan and resumable execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedShard:
    index: int
    target: Target
    requests: tuple[RequestRef, ...]

    @property
    def problems(self) -> tuple[Mapping[str, Any], ...]:
        seen: set[str] = set()
        out: list[Mapping[str, Any]] = []
        for ref in self.requests:
            if ref.problem_id not in seen:
                seen.add(ref.problem_id)
                out.append(ref.problem)
        return tuple(out)

    @property
    def problem_ids(self) -> list[str]:
        return [str(problem["problem_id"]) for problem in self.problems]


def build_shard_plan(
    problems: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    *,
    shard_size: int,
    samples_per_prompt: int = SAMPLES_PER_PROMPT,
    seed: int = 0,
) -> list[PlannedShard]:
    """Chunk the registered request order into resumable shards.

    A shard holds at most ``shard_size * samples_per_prompt`` requests, contiguous
    in the registered ``sha256(seed:request_id)`` order.  A problem's four
    requests are hash-scattered, so a shard can contain more than ``shard_size``
    distinct problems; the request bound (not a prompt count) is the resume
    granularity.  The train token cut-off stops at a shard boundary, so the
    accepted subset is a prefix of that order.
    """

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    monitor_ids = monitor_problem_ids(problems, targets, seed=seed)
    plan: list[PlannedShard] = []
    for target in targets:
        refs = _bucket_refs(
            problems,
            target,
            samples_per_prompt=samples_per_prompt,
            seed=seed,
            monitor_ids=monitor_ids,
        )
        refs = _truncate_refs_to_problems(refs, int(target.problems))
        per_shard = max(
            1, int(shard_size) * int(target.samples_per_prompt or samples_per_prompt)
        )
        for start in range(0, len(refs), per_shard):
            plan.append(
                PlannedShard(
                    index=len(plan),
                    target=target,
                    requests=tuple(refs[start : start + per_shard]),
                )
            )
    return plan


@dataclass
class RunStats:
    shards_committed: list[int] = field(default_factory=list)
    shards_skipped: list[int] = field(default_factory=list)
    records: int = 0
    samples_generated: int = 0
    kept: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    generated_tokens: int = 0
    accepted_assistant_tokens: int = 0
    train_tokens: int = 0
    caps: dict[str, int] = field(
        default_factory=lambda: {"min": 0, "max": 0, "sum": 0, "calls": 0}
    )
    per_target: dict[str, dict[str, int]] = field(default_factory=dict)
    problems_skipped: dict[str, str] = field(default_factory=dict)
    sampled_problem_ids: dict[str, set[str]] = field(default_factory=dict)
    source_mix: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def add_cap(self, cap: int) -> None:
        cap = int(cap)
        if self.caps["calls"] == 0:
            self.caps["min"] = cap
            self.caps["max"] = cap
        else:
            self.caps["min"] = min(self.caps["min"], cap)
            self.caps["max"] = max(self.caps["max"], cap)
        self.caps["sum"] += cap
        self.caps["calls"] += 1

    @property
    def acceptance_rate(self) -> float:
        return self.kept / self.samples_generated if self.samples_generated else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "shards_committed": list(self.shards_committed),
            "shards_skipped": list(self.shards_skipped),
            "records": self.records,
            "samples_generated": self.samples_generated,
            "kept": self.kept,
            "dropped_by_reason": dict(sorted(self.dropped_by_reason.items())),
            "generated_tokens": self.generated_tokens,
            "accepted_assistant_tokens": self.accepted_assistant_tokens,
            "train_tokens": self.train_tokens,
            "caps": dict(self.caps),
            "per_target": {
                name: dict(values) for name, values in sorted(self.per_target.items())
            },
            "problems_skipped": dict(sorted(self.problems_skipped.items())),
            "source_mix": {
                target: {
                    source: dict(values)
                    for source, values in sorted(sources.items())
                }
                for target, sources in sorted(self.source_mix.items())
            },
        }


def _empty_target_stats(target: Target) -> dict[str, int]:
    return {
        "problems": int(target.problems),
        "sampled_problems": 0,
        "requests": 0,
        "records": 0,
        "kept": 0,
        "accepted_assistant_tokens": 0,
        "rows_target": int(target.rows_target),
    }


def _accumulate(
    stats: RunStats,
    record: Mapping[str, Any],
    problem: Mapping[str, Any],
    verdict: SampleVerdict,
    targets: Sequence[Target],
    *,
    monitor_ids: Iterable[str] | None = None,
) -> None:
    target = target_for_problem(problem, targets, monitor_ids)
    stats.records += 1
    stats.samples_generated += 1
    stats.generated_tokens += int(record.get("tokens") or 0)
    stats.add_cap(int(record.get("max_new_tokens") or 0))
    if target is not None:
        entry = stats.per_target.setdefault(target.name, _empty_target_stats(target))
        entry["records"] += 1
        entry["requests"] += 1
        stats.sampled_problem_ids.setdefault(target.name, set()).add(
            str(problem["problem_id"])
        )
        source = str(problem.get("source") or "unknown")
        mix = stats.source_mix.setdefault(target.name, {}).setdefault(
            source, {"records": 0, "kept": 0, "accepted_assistant_tokens": 0}
        )
        mix["records"] += 1
        if verdict.kept:
            entry["kept"] += 1
            entry["accepted_assistant_tokens"] += verdict.assistant_tokens
            mix["kept"] += 1
            mix["accepted_assistant_tokens"] += verdict.assistant_tokens
    if verdict.kept:
        stats.kept += 1
        stats.accepted_assistant_tokens += verdict.assistant_tokens
        if target is not None and target.name == "train":
            stats.train_tokens += verdict.assistant_tokens
    else:
        reason = verdict.drop_reason or "unknown"
        stats.dropped_by_reason[reason] = stats.dropped_by_reason.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# JSONL and manifest helpers
# ---------------------------------------------------------------------------
def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path, *, tolerant: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if tolerant:
                    break
                raise
            if not isinstance(row, dict):
                if tolerant:
                    break
                raise ValueError(f"{path}: expected JSON objects")
            rows.append(row)
    return rows


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append durable partial records (fsync per call)."""

    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_newline:
            handle.write("\n")
        for row in rows:
            handle.write(_canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_partial(path: Path) -> list[dict[str, Any]]:
    """Read a partial shard, discarding and rewriting a torn trailing record."""

    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    valid: list[dict[str, Any]] = []
    torn = False
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            torn = True
            break
        if not isinstance(row, dict):
            torn = True
            break
        valid.append(row)
    if torn or not raw.endswith(b"\n"):
        write_jsonl_atomic(path, valid)
    return valid


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                line = _canonical(row) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_text_atomic(path: Path, text: str) -> str:
    """Write text atomically with fsync; return the content sha256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = str(text).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_entry(path: Path, *, base: Path, rows: int | None = None) -> dict[str, Any]:
    path = Path(path)
    entry: dict[str, Any] = {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    entry["path"] = str(path.relative_to(base)).replace("\\", "/")
    return entry


def verify_manifest(path: Path) -> tuple[bool, str]:
    """Check the manifest hash and every recorded file hash."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable manifest {path}: {exc}"
    stored = payload.get("manifest_sha256")
    if not stored or stored != manifest_sha256(payload):
        return False, "manifest_sha256 is missing or does not recompute"
    base = path.parent
    for relative, entry in (payload.get("files") or {}).items():
        target = base / relative
        if not target.exists():
            return False, f"missing file {relative}"
        if target.stat().st_size != int(entry.get("bytes", -1)):
            return False, f"size mismatch for {relative}"
        if file_sha256(target) != entry.get("sha256"):
            return False, f"sha256 mismatch for {relative}"
    return True, "verified"


def build_manifest(
    *,
    output_dir: Path,
    pool_dir: Path,
    pool_manifest: Mapping[str, Any],
    problems_path: Path,
    run_id: str,
    stats: RunStats,
    targets: Sequence[Target],
    request_ids: Sequence[str],
    model_revision: str,
    created: str,
    build_command: str | None,
    verifiers: Mapping[str, Any],
    files: Mapping[str, int],
    selection_seed: int = 0,
    engine: str = ENGINE,
    engine_note: str | None = None,
) -> dict[str, Any]:
    """Assemble the sampler manifest (without ``manifest_sha256``)."""

    base = Path(output_dir)
    per_target = stats.per_target
    target_payload: dict[str, Any] = {}
    shortfalls: dict[str, Any] = {}
    for target in targets:
        entry = dict(
            per_target.get(target.name, _empty_target_stats(target))
        )
        entry["buckets"] = [target.low, target.high]
        entry["token_target"] = int(target.token_target)
        entry["rows_target"] = int(target.rows_target)
        entry["sources"] = {
            source: dict(values)
            for source, values in sorted(
                (stats.source_mix.get(target.name) or {}).items()
            )
        }
        if target.token_target:
            entry["token_target_reached"] = (
                int(entry.get("accepted_assistant_tokens", 0)) >= target.token_target
            )
        target_payload[target.name] = entry
    train_target = next(
        (target for target in targets if target.name == "train"), None
    )
    train_goal = int(train_target.token_target) if train_target else 0
    if train_goal and stats.train_tokens < train_goal:
        shortfalls["train_tokens"] = {
            "accepted_assistant_tokens": stats.train_tokens,
            "target": train_goal,
        }
    payload = {
        "kind": "stage0_local_samples",
        "version": 1,
        "created": created,
        "engine": str(engine),
        "engine_note": (
            engine_note
            if engine_note is not None
            else (
                "padded adaptive-batch HF engine; per-request max_new_tokens = "
                "max_length - rendered prompt tokens"
            )
        ),
        "model": {
            "id": MODEL_ID,
            "revision": model_revision,
            "dtype": "bfloat16",
        },
        "sampling": dict(SAMPLING),
        "max_length": REGISTERED_MAX_LENGTH,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "reasoning_share_min": REGISTERED_MIN_REASONING_SHARE,
        "run_id": run_id,
        "pool": {
            "dir": str(pool_dir),
            "manifest_sha256": pool_manifest.get("manifest_sha256"),
            "problems_sha256": file_sha256(problems_path),
            "problems_rows": len(read_jsonl(problems_path)),
            "creative_briefs": pool_manifest.get("creative_briefs"),
        },
        "selection_order": {
            "rule": "sha256(seed:request_id) ascending over the whole bucket",
            "seed": int(selection_seed),
        },
        "targets": target_payload,
        "buckets": {
            target.name: [target.low, target.high] for target in targets
        },
        "requests": {
            "count": len(request_ids),
            "sha256": hashlib.sha256(
                "\n".join(sorted(str(value) for value in request_ids)).encode("utf-8")
            ).hexdigest(),
            "file": "request_ids.txt",
        },
        "caps": {
            "max_length": REGISTERED_MAX_LENGTH,
            "registered_max_new_tokens": REGISTERED_MAX_NEW_TOKENS,
            "per_request": {
                "rule": "max_length - rendered_prompt_tokens",
                "count": int(stats.caps.get("calls", 0)),
                "min": int(stats.caps.get("min", 0)),
                "max": int(stats.caps.get("max", 0)),
                "mean": (
                    float(stats.caps.get("sum", 0)) / stats.caps["calls"]
                    if stats.caps.get("calls")
                    else 0.0
                ),
            },
        },
        "tokens": {
            "generated": int(stats.generated_tokens),
            "accepted_assistant": int(stats.accepted_assistant_tokens),
            "train_accepted_assistant": int(stats.train_tokens),
        },
        "acceptance": {
            "samples": int(stats.samples_generated),
            "kept": int(stats.kept),
            "rate": float(stats.acceptance_rate),
            "dropped_by_reason": dict(sorted(stats.dropped_by_reason.items())),
        },
        "shards": {
            "committed": list(stats.shards_committed),
            "skipped": list(stats.shards_skipped),
        },
        "problems_skipped": dict(sorted(stats.problems_skipped.items())),
        "shortfalls": shortfalls,
        "verifiers": dict(verifiers),
        "build_command": build_command,
        "files": {
            str(relative).replace("\\", "/"): file_entry(
                base / relative, base=base, rows=rows
            )
            for relative, rows in files.items()
        },
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload


# ---------------------------------------------------------------------------
# Resumable shard execution
# ---------------------------------------------------------------------------
def shard_paths(output_dir: Path, index: int) -> tuple[Path, Path, Path]:
    directory = Path(output_dir) / "shards"
    shard = directory / f"shard_{int(index):05d}.jsonl"
    return shard, shard.with_name(shard.name + ".committed"), shard.with_name(
        shard.name + ".partial"
    )


def _dedupe(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        request_id = str(record.get("request_id") or "")
        if not request_id or request_id in seen:
            continue
        seen.add(request_id)
        unique.append(dict(record))
    return unique


def target_plan_payload(targets: Sequence[Target]) -> dict[str, Any]:
    """The effective target plan bound by the run marker."""

    return {
        target.name: {
            "buckets": [int(target.low), int(target.high)],
            "problems": int(target.problems),
            "token_target": int(target.token_target),
            "rows_target": int(target.rows_target),
            "verified": bool(target.verified),
            "samples_per_prompt": int(target.samples_per_prompt),
        }
        for target in targets
    }


def build_run_marker(
    *,
    run_id: str,
    pool_manifest_sha256: str,
    targets: Sequence[Target],
    samples_per_prompt: int,
    shard_size: int,
    seed: int = 0,
    prereg_sha256: str | None = None,
    hidden_tests_sha256: str | None = None,
    selection_order_version: str = SELECTION_ORDER_VERSION,
    engine: str | None = None,
) -> dict[str, Any]:
    """Build the complete run identity bound into ``run_marker.json``."""

    payload = {
        "run_id": str(run_id),
        "pool_manifest_sha256": str(pool_manifest_sha256),
        "selection_order": {
            "rule": SELECTION_ORDER_RULE,
            "version": str(selection_order_version),
            "seed": int(seed),
        },
        "targets": target_plan_payload(targets),
        "prereg_sha256": prereg_sha256,
        "hidden_tests_sha256": hidden_tests_sha256,
        "samples_per_prompt": int(samples_per_prompt),
        "shard_size": int(shard_size),
    }
    if engine is not None:
        payload["engine"] = str(engine)
    return payload


def ensure_run_marker(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write or verify the output directory's complete run identity marker.

    A directory that already contains shard files but no marker is refused: it
    may have been written by an older selection order or target plan.  Any
    marker mismatch is a refusal (the CLI exits 2), never a silent adoption.
    """

    path = Path(output_dir) / "run_marker.json"
    expected = {key: payload[key] for key in sorted(payload)}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - malformed marker is a refusal
            raise RunIdentityError(
                f"{path} is not a valid run marker ({exc}); use a fresh "
                "--output-dir or move the old run aside"
            ) from None
        if not isinstance(existing, dict) or existing != expected:
            raise RunIdentityError(
                f"{path} belongs to a different Stage 0-lite run/selection order/"
                "target plan/verifier; use a fresh --output-dir or move the old "
                "run aside"
            )
    else:
        shards = Path(output_dir) / "shards"
        leftovers = sorted(
            list(shards.glob("*.jsonl"))
            + list(shards.glob("*.partial"))
            + list(shards.glob("*.committed"))
        )
        if leftovers:
            raise RunIdentityError(
                f"{output_dir} contains shard files ({leftovers[0].name}) but has "
                "no run_marker.json; use a fresh --output-dir or move the old "
                "run aside"
            )
        write_json_atomic(path, expected)
    return expected


def _check_run_identity(
    records: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    pool_manifest_sha256: str,
    source: Path,
) -> None:
    for record in records:
        if str(record.get("run_id") or "") != str(run_id):
            raise RunIdentityError(
                f"{source} belongs to run {record.get('run_id')!r}, not {run_id!r}; "
                "use a fresh --output-dir or move the old run aside"
            )
        if str(record.get("pool_manifest_sha256") or "") != str(
            pool_manifest_sha256
        ):
            raise RunIdentityError(
                f"{source} was sampled from pool "
                f"{record.get('pool_manifest_sha256')!r}, not "
                f"{pool_manifest_sha256!r}; refusing to mix pools"
            )

def run_sampling(
    *,
    problems: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    output_dir: Path,
    shard_size: int,
    samples_per_prompt: int,
    max_length: int,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int],
    evaluate_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], SampleVerdict],
    generate_fn: Callable[..., Sequence[Any]],
    model: Any,
    processor: Any,
    decode: Callable[[Sequence[int]], str] | None,
    run_id: str,
    pool_manifest_sha256: str,
    model_revision: str,
    max_batch_override: int | None = None,
    limit_shards: int | None = None,
    seed: int = 0,
    prereg_sha256: str | None = None,
    hidden_tests_sha256: str | None = None,
    selection_order_version: str = SELECTION_ORDER_VERSION,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> RunStats:
    """Sample every unfinished request, resuming from the per-request partials.

    Requests are planned in the registered ``sha256(seed:request_id)`` ascending
    order over each whole bucket; the train token cut-off and the pilot bucket
    truncation therefore use that source-neutral order.  A shard is committed
    when every request it planned is either persisted in its partial or belongs
    to a problem whose prompt already exceeds ``max_length``.  Committed shards
    are never regenerated.  A crash mid-shard leaves the durable ``.partial``
    records; the next run asks the generator only for the missing request ids.
    """

    output_dir = Path(output_dir)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    ensure_run_marker(
        output_dir,
        build_run_marker(
            run_id=run_id,
            pool_manifest_sha256=pool_manifest_sha256,
            targets=targets,
            samples_per_prompt=samples_per_prompt,
            shard_size=shard_size,
            seed=seed,
            prereg_sha256=prereg_sha256,
            hidden_tests_sha256=hidden_tests_sha256,
            selection_order_version=selection_order_version,
        ),
    )
    plan = build_shard_plan(
        problems,
        targets,
        shard_size=shard_size,
        samples_per_prompt=samples_per_prompt,
        seed=seed,
    )
    problems_by_id = {str(problem["problem_id"]): problem for problem in problems}
    target_by_name = {target.name: target for target in targets}
    train_target = target_by_name.get("train")
    train_goal = int(train_target.token_target) if train_target else 0
    prompt_cache: dict[str, int] = {}
    stats = RunStats()
    committed_indices: set[int] = set()
    last_emitted_tokens = 0

    def prompt_tokens(problem: Mapping[str, Any]) -> int:
        problem_id = str(problem["problem_id"])
        if problem_id not in prompt_cache:
            prompt_cache[problem_id] = int(prompt_tokens_for(problem))
        return prompt_cache[problem_id]

    def summarize(records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            problem = problems_by_id.get(str(record.get("problem_id")))
            if problem is None:
                continue
            stats.problems_skipped.pop(str(problem["problem_id"]), None)
            verdict = evaluate_fn(record, problem)
            _accumulate(stats, record, problem, verdict, targets)

    def emit_progress() -> None:
        nonlocal last_emitted_tokens
        if progress is None:
            return
        payload = stats.as_dict()
        payload["items_done"] = stats.records
        payload["train_goal"] = train_goal
        payload["shards_planned"] = len(plan)
        payload["generated_tokens_delta"] = max(
            0, stats.generated_tokens - last_emitted_tokens
        )
        last_emitted_tokens = stats.generated_tokens
        progress(payload)

    # Pre-scan committed shards so a resumed run starts with the correct token
    # total (and does not report the already-generated tokens as new).
    for planned in plan:
        shard, committed, _partial = shard_paths(output_dir, planned.index)
        if not (committed.exists() and shard.exists()):
            continue
        rows = read_jsonl(shard)
        _check_run_identity(
            rows,
            run_id=run_id,
            pool_manifest_sha256=pool_manifest_sha256,
            source=shard,
        )
        summarize(rows)
        committed_indices.add(planned.index)
        stats.shards_committed.append(planned.index)
    last_emitted_tokens = stats.generated_tokens
    emit_progress()

    processed = 0
    for planned in plan:
        if planned.index in committed_indices:
            continue
        if limit_shards is not None and processed >= int(limit_shards):
            break
        processed += 1
        target = planned.target
        if (
            target.name == "train"
            and train_goal > 0
            and stats.train_tokens >= train_goal
        ):
            stats.shards_skipped.append(planned.index)
            continue

        shard, committed, partial = shard_paths(output_dir, planned.index)
        existing: list[dict[str, Any]] = []
        if partial.exists():
            existing.extend(load_partial(partial))
        if shard.exists():
            # Written but not marked: a crash between replace and the marker.
            existing.extend(read_jsonl(shard, tolerant=True))
        existing = _dedupe(existing)
        _check_run_identity(
            existing,
            run_id=run_id,
            pool_manifest_sha256=pool_manifest_sha256,
            source=partial if partial.exists() else shard,
        )
        summarize(existing)
        done = {str(row.get("request_id")) for row in existing}

        expected = {ref.request_id for ref in planned.requests}
        requests, skipped = build_requests_from_refs(
            planned.requests,
            prompt_tokens_for=prompt_tokens,
            max_length=max_length,
            done_request_ids=done,
        )
        for problem_id, reason in skipped.items():
            stats.problems_skipped[problem_id] = reason

        if requests:
            new_records = generate_calls(
                requests,
                generate_fn=generate_fn,
                model=model,
                processor=processor,
                decode=decode,
                run_id=run_id,
                pool_manifest_sha256=pool_manifest_sha256,
                shard_index=planned.index,
                model_revision=model_revision,
                seed=seed,
                samples_per_prompt=samples_per_prompt,
                max_length=max_length,
                max_batch_override=max_batch_override,
            )
            append_jsonl(partial, new_records)
            summarize(new_records)
            done.update(str(row.get("request_id")) for row in new_records)
            existing.extend(new_records)
            emit_progress()

        skipped_problem_ids = set(skipped)
        complete = all(
            request_id in done
            or parse_stage0_request_id(request_id)[0] in skipped_problem_ids
            for request_id in expected
        )
        if complete:
            records = _dedupe(existing)
            records.sort(
                key=lambda row: (
                    str(row.get("problem_id")),
                    int(row.get("sample_index") or 0),
                )
            )
            write_jsonl_atomic(shard, records)
            write_json_atomic(committed, {"shard": planned.index, "committed": True})
            try:
                partial.unlink()
            except OSError:
                pass
            stats.shards_committed.append(planned.index)
            emit_progress()

    for name, problem_ids in stats.sampled_problem_ids.items():
        entry = stats.per_target.setdefault(
            name, _empty_target_stats(target_by_name[name])
        )
        entry["sampled_problems"] = len(problem_ids)
    for target in targets:
        entry = stats.per_target.setdefault(target.name, _empty_target_stats(target))
        entry.setdefault("sampled_problems", 0)
    return stats


def merge_hidden_tests(
    problems: Sequence[Mapping[str, Any]], hidden_path: Path
) -> dict[str, Any]:
    """Merge the registered Stage 0 hidden tests into the pool problems.

    Mirrors ``scripts/postprocess_stage0.py``: append tests that are not already
    present and record the hidden-tests file provenance.
    """

    hidden_path = Path(hidden_path)
    rows = read_jsonl(hidden_path)
    hidden: dict[str, list[Any]] = {}
    for row in rows:
        problem_id = str(row.get("problem_id") or "")
        if not problem_id:
            raise ValueError(f"{hidden_path}: a hidden-tests row has no problem_id")
        hidden[problem_id] = list(row.get("tests") or [])
    merged = 0
    for problem in problems:
        extra = hidden.get(str(problem["problem_id"]))
        if not extra:
            continue
        existing = list(problem.get("tests") or [])
        additions = [test for test in extra if test not in existing]
        if additions:
            problem["tests"] = existing + additions
            merged += 1
    return {
        "path": str(hidden_path),
        "sha256": file_sha256(hidden_path),
        "rows": len(rows),
        "problems_merged": merged,
    }


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------
def load_pool(pool_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and validate a frozen Stage 0 pool (refusing smoke builds)."""

    pool_dir = Path(pool_dir)
    manifest_path = pool_dir / "manifest.json"
    problems_path = pool_dir / "problems.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 0 pool manifest missing: {manifest_path}")
    if not problems_path.exists():
        raise FileNotFoundError(f"Stage 0 problems.jsonl missing: {problems_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = manifest.get("manifest_sha256")
    if not stored or stored != manifest_sha256(manifest):
        raise ValueError("pool manifest_sha256 is missing or does not recompute")
    artifact = (manifest.get("artifacts") or {}).get("problems.jsonl") or {}
    actual = file_sha256(problems_path)
    if artifact.get("sha256") != actual:
        raise ValueError("problems.jsonl does not match the frozen manifest hash")
    decontam = manifest.get("decontamination")
    if not isinstance(decontam, dict):
        raise ValueError("pool manifest has no decontamination section")
    if decontam.get("status") == "not_built":
        raise ValueError("pool decontamination was not built")
    targets = decontam.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("pool manifest has no decontamination targets")
    bad_targets = {
        name: entry
        for name, entry in targets.items()
        if isinstance(entry, dict) and entry.get("status") not in {"ok"}
    }
    if bad_targets:
        raise ValueError(f"pool has incomplete decontamination targets: {bad_targets}")
    from gemmaloop.stage0.evals import DEFAULT_BRIEFS, EVAL_SPECS
    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

    required = {spec.name for spec in EVAL_SPECS}
    missing = sorted(required - set(targets))
    if missing:
        raise ValueError(
            f"pool decontamination is missing registered targets: {missing}"
        )
    if manifest.get("creative_briefs") != PART1_CREATIVE_BRIEFS:
        briefs_key = f"file:{DEFAULT_BRIEFS.name}"
        briefs = targets.get(briefs_key) or {}
        if briefs.get("status") != "ok":
            raise ValueError(
                "pool has no creative-briefs decontamination target; use the "
                "Part-1 pool mode or freeze the briefs"
            )
    limits = manifest.get("limits") or {}
    smoke = {
        "eval_limit": limits.get("eval_limit"),
        "limit_rows": limits.get("limit_rows"),
        "allow_missing_eval": limits.get("allow_missing_eval"),
        "allow_missing_briefs": limits.get("allow_missing_briefs"),
        "allow_short_sources": limits.get("allow_short_sources"),
        "allow_duplicates": limits.get("allow_duplicates"),
    }
    if any(bool(value) for value in smoke.values()):
        raise ValueError(f"refusing a smoke/partial pool: {smoke}")
    problems = read_jsonl(problems_path)
    if not problems:
        raise ValueError("the Stage 0 pool is empty")
    return problems, manifest
