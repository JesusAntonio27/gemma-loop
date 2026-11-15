"""Stage 0-lite planning for the Modal vLLM backend (no Modal import).

The pilot's Stage 0-lite plan is normally sampled on the registered padded
local engine (:mod:`gemmaloop.stage0.local`).  This module keeps that plan
*identical* while the requests are executed by stock vLLM on Modal:

* the same targets read from ``prereg/prereg.yaml``,
* the same deterministic shard/request order (``local.build_shard_plan`` and
  ``local.build_requests_from_refs``),
* the same request ids (``"{problem_id}:{sample_index}"``), prompts and
  per-request budget ``max_new_tokens = 8192 - rendered_prompt_tokens``,
* the same sampling parameters (T=1.0, top_p=0.95, top_k=64, thinking on).

Only the execution backend changes.  One Modal call submits *all* of its
requests to vLLM in a single ``generate`` pass (the probe's shape), then writes
and commits the complete shards.  Non-train requests are one call; train
requests are whole-shard calls of at most :data:`MAX_TRAIN_CALL_SIZE` requests
in registered plan order so the accepted train subset remains a prefix of that
order.

The cost estimate uses the measured 2026-11-05 A100-80GB probe (2,000 requests,
11.7M generated tokens, 1,802 s, $2.07 booked): ~$0.18 per 1M generated tokens
plus ~$0.26 launch overhead per call.  The train request estimate uses the
measured local acceptance (0.534 accepted/generated at ~4,213 generated
tokens/request), which needs ~1,333 train requests for the 3M-token goal.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from gemmaloop.stage0 import local
from gemmaloop.stage0.postprocess import MAX_LENGTH, MIN_REASONING_SHARE

ENGINE = "modal-vllm"
ENGINE_NOTE = (
    "stock vLLM on Modal; per-request max_new_tokens = "
    "max_length - rendered prompt tokens"
)
DEFAULT_CALL_SIZE = 1400
MAX_TRAIN_CALL_SIZE = 1400
MAX_CALL_SIZE = 2000  # absolute per-call capacity proven by the probe

# Measured 2026-11-05 A100-80GB probe: 2,000 requests -> 11.7M generated
# tokens in 1,802 s wall, $2.07 booked (including launch overhead).
MEASURED_REQUESTS = 2_000
MEASURED_GENERATED_TOKENS = 11_700_000
MEASURED_WALL_SECONDS = 1_802.0
MEASURED_COST_USD = 2.07
MEASURED_COST_PER_MILLION_GENERATED_TOKENS_USD = 0.18
MEASURED_LAUNCH_OVERHEAD_USD = 0.26
MEASURED_GENERATED_TOKENS_PER_REQUEST = (
    MEASURED_GENERATED_TOKENS / MEASURED_REQUESTS
)

# Measured on the partial local Stage 0-lite run: 0.534 accepted/generated at
# ~4,213 generated tokens/request, i.e. ~2,250 accepted tokens/request and
# ~1,333 train requests for the 3,000,000-token goal.
LOCAL_MEASURED_GENERATED_TOKENS_PER_REQUEST = 4_213.0
LOCAL_MEASURED_TRAIN_ACCEPTED_SHARE = 0.534
DEFAULT_TRAIN_ACCEPTED_SHARE = LOCAL_MEASURED_TRAIN_ACCEPTED_SHARE
DEFAULT_IMAGE_BUILD_USD = 0.25

# vLLM stores per-request seeds in a C ``long`` (signed 64-bit on x86-64).
SEED_MODULUS = 2**63 - 1


class LitePlanError(ValueError):
    """The lite plan cannot be sampled in the requested call shape."""


# ---------------------------------------------------------------------------
# Seeds and prompt lengths
# ---------------------------------------------------------------------------
def lite_request_seed(seed: int, request_id: str) -> int:
    """The registered per-request seed, reduced to vLLM's signed 64-bit range.

    Mirrors ``gemmaloop.serve.generate._gumbel_generators`` without importing
    torch: ``sha256(f"{seed}:{request_id}")`` first eight digest bytes
    little-endian (``_request_seed``), then modulo ``2**63 - 1`` exactly as the
    local sampler does before ``torch.Generator.manual_seed``.  vLLM stores
    ``SamplingParams.seed`` in a C ``long``; the full unsigned 64-bit digest can
    exceed ``2**63 - 1`` and crashes its EngineCore with
    ``OverflowError: Python int too large to convert to C long``.
    """

    digest = hashlib.sha256(f"{int(seed)}:{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % SEED_MODULUS


def validate_payload_seeds(payload: Mapping[str, Any]) -> None:
    """Refuse seeds that do not fit vLLM's signed 64-bit ``long``.

    Called by the launcher on every shard payload *before* any Modal contact so
    an out-of-range seed can never reach a remote EngineCore.
    """

    for shard in payload.get("shards") or ():
        for request in shard.get("requests") or ():
            seed = int(request["seed"])
            if not 0 <= seed < SEED_MODULUS:
                raise LitePlanError(
                    f"request seed {seed} for "
                    f"{request.get('request_id')!r} does not fit a signed "
                    f"64-bit integer ([0, {SEED_MODULUS}))"
                )


def make_prompt_tokens_fn(tokenizer: Any) -> Callable[[Mapping[str, Any]], int]:
    """Return the local sampler's rendered-prompt token counter."""

    def prompt_tokens(problem: Mapping[str, Any]) -> int:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(problem.get("prompt") or "")}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(local.SAMPLING["enable_thinking"]),
            return_dict=True,
        )
        ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return int(len(ids))

    return prompt_tokens


def load_tokenizer(*, local_files_only: bool = True) -> Any:
    """Load the pinned Gemma tokenizer; local cache first, then the Hub."""

    from transformers import AutoTokenizer  # noqa: PLC0415 - CPU launcher only

    from gemmaloop.model.snapshot import MODEL_REVISION

    kwargs: dict[str, Any] = {"revision": MODEL_REVISION}
    if local_files_only:
        try:
            return AutoTokenizer.from_pretrained(
                local.MODEL_ID, local_files_only=True, **kwargs
            )
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(local.MODEL_ID, **kwargs)


# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiteRequest:
    """One ``(problem_id, sample_index)`` request with its registered budget."""

    shard_index: int
    target_name: str
    problem: Mapping[str, Any]
    sample_index: int
    prompt_tokens: int
    max_new_tokens: int
    seed: int

    @property
    def problem_id(self) -> str:
        return str(self.problem["problem_id"])

    @property
    def request_id(self) -> str:
        return local.stage0_request_id(self.problem_id, self.sample_index)

    @property
    def prompt(self) -> str:
        return str(self.problem.get("prompt") or "")

    def as_payload(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "sample_index": int(self.sample_index),
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_tokens": int(self.prompt_tokens),
            "max_new_tokens": int(self.max_new_tokens),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class LiteShard:
    index: int
    target_name: str
    requests: tuple[LiteRequest, ...]
    skipped: Mapping[str, str]

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True)
class LiteCall:
    index: int
    kind: str  # "train" or "non_train"
    requests: tuple[LiteRequest, ...]

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def shard_indices(self) -> tuple[int, ...]:
        seen: list[int] = []
        for request in self.requests:
            if request.shard_index not in seen:
                seen.append(request.shard_index)
        return tuple(seen)

    def as_payload(self) -> list[dict[str, Any]]:
        grouped: dict[int, list[LiteRequest]] = {}
        for request in self.requests:
            grouped.setdefault(request.shard_index, []).append(request)
        return [
            {
                "shard_index": int(index),
                "commit": True,
                "requests": [request.as_payload() for request in requests],
            }
            for index, requests in grouped.items()
        ]


@dataclass(frozen=True)
class LitePlan:
    run_id: str
    targets: tuple[local.Target, ...]
    shards: tuple[LiteShard, ...]
    train_calls: tuple[LiteCall, ...]
    other_calls: tuple[LiteCall, ...]
    call_size: int

    @property
    def train_requests(self) -> int:
        return sum(len(shard.requests) for shard in self.shards if shard.target_name == "train")

    @property
    def other_requests(self) -> int:
        return sum(len(shard.requests) for shard in self.shards if shard.target_name != "train")

    @property
    def requests_total(self) -> int:
        return self.train_requests + self.other_requests

    @property
    def train_goal(self) -> int:
        target = next((target for target in self.targets if target.name == "train"), None)
        return int(target.token_target) if target is not None else 0

    def shard_by_index(self, index: int) -> LiteShard:
        for shard in self.shards:
            if shard.index == index:
                return shard
        raise KeyError(index)

    def requests_by_target(self) -> dict[str, int]:
        counts: dict[str, int] = {target.name: 0 for target in self.targets}
        for shard in self.shards:
            counts[shard.target_name] = counts.get(shard.target_name, 0) + len(
                shard.requests
            )
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "targets": [target.name for target in self.targets],
            "planned_shards": len(self.shards),
            "requests_total": self.requests_total,
            "train_requests": self.train_requests,
            "non_train_requests": self.other_requests,
            "requests_by_target": self.requests_by_target(),
            "train_goal": self.train_goal,
            "call_size": self.call_size,
            "train_calls": len(self.train_calls),
            "non_train_calls": len(self.other_calls),
        }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def _chunk_shards(
    shards: Sequence[LiteShard], *, call_size: int, kind: str, start_index: int = 0
) -> list[LiteCall]:
    calls: list[LiteCall] = []
    current: list[LiteRequest] = []
    count = 0

    def flush() -> None:
        nonlocal current, count
        if current:
            calls.append(
                LiteCall(
                    index=start_index + len(calls),
                    kind=kind,
                    requests=tuple(current),
                )
            )
            current = []
            count = 0

    for shard in shards:
        requests = list(shard.requests)
        if not requests:
            continue
        if len(requests) > int(call_size):
            raise LitePlanError(
                f"shard {shard.index} has {len(requests)} requests, more than "
                f"the lite call size {call_size}; lower --shard-size or raise "
                f"--lite-call-size (max {MAX_TRAIN_CALL_SIZE})"
            )
        if current and count + len(requests) > int(call_size):
            flush()
        current.extend(requests)
        count += len(requests)
    flush()
    return calls


def build_lite_plan(
    problems: Sequence[Mapping[str, Any]],
    targets: Sequence[local.Target],
    *,
    shard_size: int = local.DEFAULT_SHARD_SIZE,
    samples_per_prompt: int = local.SAMPLES_PER_PROMPT,
    seed: int = 0,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int],
    max_length: int = local.REGISTERED_MAX_LENGTH,
    call_size: int = DEFAULT_CALL_SIZE,
    run_id: str = "",
) -> LitePlan:
    """Build the exact local-sampler request plan, grouped into whole-shard calls.

    ``prompt_tokens_for`` must be the same renderer/tokenizer the local sampler
    uses; the resulting request ids, prompts and per-request
    ``max_new_tokens`` are then identical to ``local.run_sampling``.
    """

    call_size = int(call_size)
    if not 1 <= call_size <= MAX_TRAIN_CALL_SIZE:
        raise LitePlanError(
            f"train call size must be in (0, {MAX_TRAIN_CALL_SIZE}]; got {call_size}"
        )
    if int(samples_per_prompt) != int(local.SAMPLES_PER_PROMPT):
        raise LitePlanError(
            "the lite plan is registered with four samples per prompt; got "
            f"{samples_per_prompt}"
        )

    cache: dict[str, int] = {}

    def cached_prompt_tokens(problem: Mapping[str, Any]) -> int:
        problem_id = str(problem["problem_id"])
        if problem_id not in cache:
            cache[problem_id] = int(prompt_tokens_for(problem))
        return cache[problem_id]

    planned = local.build_shard_plan(
        problems,
        targets,
        shard_size=int(shard_size),
        samples_per_prompt=int(samples_per_prompt),
        seed=int(seed),
    )
    shards: list[LiteShard] = []
    for planned_shard in planned:
        requests, skipped = local.build_requests_from_refs(
            planned_shard.requests,
            prompt_tokens_for=cached_prompt_tokens,
            max_length=int(max_length),
        )
        lite_requests = tuple(
            LiteRequest(
                shard_index=int(planned_shard.index),
                target_name=planned_shard.target.name,
                problem=request.problem,
                sample_index=int(request.sample_index),
                prompt_tokens=int(request.prompt_tokens),
                max_new_tokens=int(request.budget),
                seed=lite_request_seed(seed, request.request_id),
            )
            for request in requests
        )
        shards.append(
            LiteShard(
                index=int(planned_shard.index),
                target_name=planned_shard.target.name,
                requests=lite_requests,
                skipped=dict(skipped),
            )
        )

    train_shards = [shard for shard in shards if shard.target_name == "train"]
    other_shards = [shard for shard in shards if shard.target_name != "train"]
    train_calls = _chunk_shards(train_shards, call_size=call_size, kind="train")
    # Every non-train request goes to vLLM in ONE generation pass.  There is no
    # shard-at-a-time loop on the remote side; the whole-shard boundaries only
    # decide how the returned rows are written and committed.
    other_requests = tuple(
        request for shard in other_shards for request in shard.requests
    )
    if len(other_requests) > MAX_CALL_SIZE:
        raise LitePlanError(
            f"{len(other_requests)} non-train requests exceed the one-call "
            f"capacity {MAX_CALL_SIZE}"
        )
    other_calls = (
        [
            LiteCall(
                index=len(train_calls),
                kind="non_train",
                requests=other_requests,
            )
        ]
        if other_requests
        else []
    )
    return LitePlan(
        run_id=str(run_id),
        targets=tuple(targets),
        shards=tuple(shards),
        train_calls=tuple(train_calls),
        other_calls=tuple(other_calls),
        call_size=call_size,
    )


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
def _calls_for_request_prefix(
    shards: Sequence[LiteShard], limit: int, *, call_size: int, kind: str
) -> tuple[list[LiteCall], int]:
    """Calls and requests actually sent to cover the first ``limit`` requests.

    Shards are sent whole, so the returned request count includes the overshoot
    up to the end of the last shard and the last call.
    """

    if int(limit) <= 0:
        return [], 0
    selected: list[LiteShard] = []
    count = 0
    for shard in shards:
        if count >= int(limit):
            break
        selected.append(shard)
        count += len(shard.requests)
    calls = _chunk_shards(selected, call_size=call_size, kind=kind)
    return calls, sum(len(shard.requests) for shard in selected)


def estimate_lite_run(
    plan: LitePlan,
    *,
    train_accepted_share: float = DEFAULT_TRAIN_ACCEPTED_SHARE,
    generated_tokens_per_request: float = (
        LOCAL_MEASURED_GENERATED_TOKENS_PER_REQUEST
    ),
    cost_per_million_generated_tokens_usd: float = (
        MEASURED_COST_PER_MILLION_GENERATED_TOKENS_USD
    ),
    launch_overhead_usd: float = MEASURED_LAUNCH_OVERHEAD_USD,
    image_build_usd: float = DEFAULT_IMAGE_BUILD_USD,
) -> dict[str, Any]:
    """Conservative expected calls/tokens/cost using the measured rates.

    The expected call sequence is in execution order: the single non-train
    call first, then whole-shard train calls in plan order until the accepted
    token goal is expected to be reached.  Each entry carries its expected
    generated tokens and cost so the launcher can print the cap-check sequence.
    """

    if not 0 < float(train_accepted_share) <= 1:
        raise ValueError("train_accepted_share must be in (0, 1]")
    if float(generated_tokens_per_request) <= 0:
        raise ValueError("generated_tokens_per_request must be positive")

    train_shards = [shard for shard in plan.shards if shard.target_name == "train"]
    goal = int(plan.train_goal)
    if goal > 0 and train_accepted_share > 0:
        accepted_per_request = (
            float(generated_tokens_per_request) * float(train_accepted_share)
        )
        expected_train_requests = int(
            math.ceil(goal / max(accepted_per_request, 1e-9))
        )
    else:
        expected_train_requests = int(plan.train_requests)
    expected_train_requests = min(expected_train_requests, int(plan.train_requests))

    train_calls, train_requests_sent = _calls_for_request_prefix(
        train_shards,
        expected_train_requests,
        call_size=plan.call_size,
        kind="train",
    )
    other_calls = list(plan.other_calls)

    call_plan: list[dict[str, Any]] = []
    compute_total = 0.0
    overhead_total = 0.0
    generated_total = 0

    def add_call(kind: str, requests: int, shards: Sequence[int]) -> None:
        nonlocal compute_total, overhead_total, generated_total
        tokens = int(round(int(requests) * float(generated_tokens_per_request)))
        compute = (
            tokens / 1_000_000.0 * float(cost_per_million_generated_tokens_usd)
        )
        overhead = float(launch_overhead_usd)
        if not call_plan:
            # The image build is paid once per launch, on the first call.
            overhead += float(image_build_usd)
        generated_total += tokens
        compute_total += compute
        overhead_total += overhead
        call_plan.append(
            {
                "call_index": len(call_plan),
                "kind": kind,
                "requests": int(requests),
                "shard_indices": [int(index) for index in shards],
                "generated_tokens": tokens,
                "compute_usd": float(compute),
                "overhead_usd": float(overhead),
                "expected_usd": float(compute + overhead),
            }
        )

    for call in other_calls:
        add_call("non_train", call.request_count, call.shard_indices)
    for call in train_calls:
        add_call("train", call.request_count, call.shard_indices)

    calls = len(call_plan)
    full_train_calls = len(plan.train_calls)
    return {
        "train_goal": goal,
        "train_requests_planned": int(plan.train_requests),
        "train_requests_expected": int(expected_train_requests),
        "train_requests_expected_sent": int(train_requests_sent),
        "non_train_requests": int(plan.other_requests),
        "expected_calls": int(calls),
        "train_calls_expected": len(train_calls),
        "non_train_calls": len(other_calls),
        "full_run_calls": int(full_train_calls + len(plan.other_calls)),
        "expected_generated_tokens": int(generated_total),
        "compute_usd": float(compute_total),
        "launch_overhead_usd": float(overhead_total),
        "usd": float(compute_total + overhead_total),
        "expected_call_plan": call_plan,
        "measured_rate": {
            "probe_requests": MEASURED_REQUESTS,
            "probe_generated_tokens": MEASURED_GENERATED_TOKENS,
            "probe_wall_seconds": MEASURED_WALL_SECONDS,
            "probe_cost_usd": MEASURED_COST_USD,
            "generated_tokens_per_request": float(generated_tokens_per_request),
            "usd_per_million_generated_tokens": float(
                cost_per_million_generated_tokens_usd
            ),
            "launch_overhead_usd_per_call": float(launch_overhead_usd),
            "train_accepted_share": float(train_accepted_share),
            "accepted_tokens_per_request": float(
                generated_tokens_per_request * train_accepted_share
            ),
            "source": "2026-11-05 probe rates + partial local acceptance",
        },
    }


def _cap_checks(
    call_plan: Sequence[Mapping[str, Any]],
    *,
    reservations_usd: Sequence[float] | None,
    max_usd: float,
    month_total_usd: float,
    hard_cap_usd: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    spent = 0.0
    for index, call in enumerate(call_plan):
        reservation = (
            float(reservations_usd[index])
            if reservations_usd is not None and index < len(reservations_usd)
            else float(call["expected_usd"])
        )
        run_after = spent + reservation
        month_after = float(month_total_usd) + run_after
        checks.append(
            {
                "call_index": int(call["call_index"]),
                "kind": str(call["kind"]),
                "requests": int(call["requests"]),
                "shard_indices": list(call["shard_indices"]),
                "reservation_usd": round(reservation, 4),
                "run_spend_before_usd": round(spent, 4),
                "run_spend_after_reservation_usd": round(run_after, 4),
                "run_cap_usd": float(max_usd),
                "run_cap_ok": run_after <= float(max_usd) + 1e-9,
                "month_total_before_usd": round(float(month_total_usd) + spent, 4),
                "month_total_after_reservation_usd": round(month_after, 4),
                "global_cap_usd": float(hard_cap_usd),
                "global_cap_ok": month_after <= float(hard_cap_usd) + 1e-9,
                "expected_call_usd": round(float(call["expected_usd"]), 4),
            }
        )
        spent += float(call["expected_usd"])
    return checks


def format_plan(
    plan: LitePlan,
    estimate: Mapping[str, Any],
    *,
    max_usd: float,
    reservations_usd: Sequence[float] | None = None,
    month_total_usd: float = 0.0,
    hard_cap_usd: float = 30.0,
) -> str:
    call_plan = list(estimate.get("expected_call_plan") or [])
    checks = _cap_checks(
        call_plan,
        reservations_usd=reservations_usd,
        max_usd=float(max_usd),
        month_total_usd=float(month_total_usd),
        hard_cap_usd=float(hard_cap_usd),
    )
    payload = {
        "mode": "sample-lite",
        "engine": ENGINE,
        **plan.as_dict(),
        "expected": dict(estimate),
        "calls": call_plan,
        "cap_checks": checks,
        "reservations_usd": (
            [float(value) for value in reservations_usd]
            if reservations_usd is not None
            else []
        ),
        "run_cap_usd": float(max_usd),
        "month_total_usd": round(float(month_total_usd), 4),
        "global_cap_usd": float(hard_cap_usd),
        "month_total_after_expected_usd": round(
            float(month_total_usd) + float(estimate["usd"]), 4
        ),
        "expected_run_fits": all(
            check["run_cap_ok"] and check["global_cap_ok"] for check in checks
        ),
        "within_cap": float(estimate["usd"]) <= float(max_usd),
        "within_global_ceiling": (
            float(month_total_usd) + float(estimate["usd"])
            <= float(hard_cap_usd) + 1e-9
        ),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_CALL_SIZE",
    "DEFAULT_IMAGE_BUILD_USD",
    "DEFAULT_TRAIN_ACCEPTED_SHARE",
    "ENGINE",
    "ENGINE_NOTE",
    "LiteCall",
    "LitePlan",
    "LitePlanError",
    "LiteRequest",
    "LiteShard",
    "LOCAL_MEASURED_GENERATED_TOKENS_PER_REQUEST",
    "LOCAL_MEASURED_TRAIN_ACCEPTED_SHARE",
    "MAX_CALL_SIZE",
    "MAX_TRAIN_CALL_SIZE",
    "MEASURED_COST_PER_MILLION_GENERATED_TOKENS_USD",
    "MEASURED_COST_USD",
    "MEASURED_GENERATED_TOKENS",
    "MEASURED_GENERATED_TOKENS_PER_REQUEST",
    "MEASURED_LAUNCH_OVERHEAD_USD",
    "MEASURED_REQUESTS",
    "MEASURED_WALL_SECONDS",
    "SEED_MODULUS",
    "build_lite_plan",
    "estimate_lite_run",
    "format_plan",
    "lite_request_seed",
    "load_tokenizer",
    "make_prompt_tokens_fn",
    "validate_payload_seeds",
]
