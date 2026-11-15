"""Time/token budget arithmetic and spill detection for the preregistration.

The evaluation budget is expressed in hours per variant.  ``tokens_per_item``
is a **per-completion** measurement (one answer per item); :func:`fit_eval_sizes`
multiplies it by the registered number of samples per benchmark (LCB 4, AIME 8,
GPQA 4, MMLU-Pro 1, vision/spatial 1).  The fitting routine assigns every
benchmark its declared minimum and then spends the remaining hours in
``priority`` order, growing each benchmark up to its cap.

:func:`spill_detected` implements the RTX 5070 / Windows WDDM rule used by the
calibration script: a training length has spilled when PyTorch's peak reserved
memory crosses 95 % of the dedicated VRAM or when any timed step collapses to
less than half the median throughput of that length.  Driver settings are never
changed; the calibration only detects the condition.

Reference: ``references/gemma-loop/04`` Know-how section 6 and
``references/gemma-loop/07`` Know-how section 3.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "DEFAULT_SAMPLES_PER_BENCH",
    "bench_rates_from_calibration",
    "eval_hours",
    "fit_eval_sizes",
    "gen_rates_from_calibration",
    "max_trainable_length",
    "rate_for_completion_length",
    "spill_detected",
    "train_hours",
]

# Number of generations per item registered in prereg/prereg.yaml (Task 10).
# Vision and spatial benchmarks use one sample.
DEFAULT_SAMPLES_PER_BENCH: dict[str, int] = {
    "lcb": 4,
    "aime": 8,
    "gpqa": 4,
    "mmlupro": 1,
    "mmmupro": 1,
    "mathvision": 1,
    "spatialeval": 1,
    "cvbench": 1,
}

# Fraction of dedicated VRAM that counts as spilling, and the per-step
# throughput collapse threshold relative to that length's median.
SPILL_RESERVED_FRACTION = 0.95
SPILL_COLLAPSE_FRACTION = 0.5

# Floating-point cushion for "the minimums exactly fill the budget" (seconds).
_EPSILON_TOKENS = 1e-6

# Generation buckets are stored as ``n<loops>_len<length>``.
_GEN_BUCKET_KEY = re.compile(r"^n(\d+)_len(\d+)$")


def _positive(value: float, name: str) -> float:
    try:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return float(value)
    except TypeError:
        raise ValueError(
            f"{name} must be a positive number, got {value!r}"
        ) from None


def _usable_rate(value: Any) -> float | None:
    """Return ``value`` as a finite positive rate, or ``None`` when unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        return None
    return rate


def _clean_attempt_rate(attempt: Any) -> float | None:
    """Rate of one stored attempt, only when the attempt itself is valid."""
    if not isinstance(attempt, Mapping):
        return None
    if attempt.get("invalid", False):
        return None
    if attempt.get("spill", False) or attempt.get("oom", False):
        return None
    return _usable_rate(attempt.get("tok_s"))


def _bucket_rate(record: Any) -> float | None:
    """Usable rate for one stored generation bucket, or ``None`` to skip it.

    A bucket is used only when it is marked valid and has a positive ``tok_s``.
    When the top-level record is invalid (e.g. the full-batch attempt spilled),
    a clean half-batch retry stored in ``attempts`` is still a valid measurement
    of the same length bucket and is used instead.
    """
    if not isinstance(record, Mapping):
        return None
    if not record.get("invalid", False):
        if not record.get("spill", False) and not record.get("oom", False):
            rate = _usable_rate(record.get("tok_s"))
            if rate is not None:
                return rate
    attempts = record.get("attempts")
    if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)):
        # Attempts are chronological (full batch first, retry second), so this
        # naturally prefers the clean retry when the first attempt spilled.
        for attempt in attempts:
            rate = _clean_attempt_rate(attempt)
            if rate is not None:
                return rate
    return None


def gen_rates_from_calibration(
    calibration: Mapping[str, Any], size: str, n_loops: int
) -> dict[int, float]:
    """Valid generation rates for one model size and loop count.

    Reads ``calibration["gen_tok_per_s"][size]`` and returns
    ``{length_bucket: tok_per_s}`` for keys ``n<n_loops>_len<bucket>``.
    Invalid buckets (``invalid``/``spill``/``oom`` or ``tok_s is None``) are
    skipped; a clean half-batch retry in the record's ``attempts`` is used when
    present.  Missing sizes, loops or records yield an empty mapping rather than
    an error so callers can decide whether the gap is fatal.
    """
    if int(n_loops) < 1:
        raise ValueError(f"n_loops must be at least 1, got {n_loops!r}")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"calibration must be a mapping, got {calibration!r}")

    by_size = calibration.get("gen_tok_per_s")
    if not isinstance(by_size, Mapping):
        return {}
    size_records = by_size.get(size)
    if not isinstance(size_records, Mapping):
        return {}

    rates: dict[int, float] = {}
    for key, record in size_records.items():
        match = _GEN_BUCKET_KEY.match(str(key))
        if match is None:
            continue
        loops, bucket = int(match.group(1)), int(match.group(2))
        if loops != int(n_loops) or bucket < 1:
            continue
        rate = _bucket_rate(record)
        if rate is not None:
            rates[bucket] = rate
    return dict(sorted(rates.items()))


def rate_for_completion_length(
    rates: Mapping[int, float], completion_tokens: float
) -> float:
    """Rate for a benchmark whose p90 completion is ``completion_tokens``.

    Picks the smallest valid bucket whose length is at least the expected
    completion (the next larger bucket when the length falls between two
    measured buckets).  When the completion exceeds every measured bucket, the
    longest valid bucket is used rather than extrapolating.

    Raises:
        ValueError: if there are no usable buckets or the length is negative.
    """
    if completion_tokens < 0:
        raise ValueError(
            f"completion_tokens must be non-negative, got {completion_tokens!r}"
        )
    usable = sorted(
        bucket
        for bucket, rate in rates.items()
        if int(bucket) > 0 and _usable_rate(rate) is not None
    )
    if not usable:
        raise ValueError(
            "no valid generation-rate buckets; refusing to invent a rate"
        )
    for bucket in usable:
        if bucket >= completion_tokens:
            return float(rates[bucket])
    return float(rates[usable[-1]])


def bench_rates_from_calibration(
    calibration: Mapping[str, Any],
    size: str,
    n_loops: int,
    benches: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-benchmark generation rates from ``tokens_per_item`` p90 lengths.

    ``benches`` defaults to every benchmark present in
    ``calibration["tokens_per_item"]``.  Each benchmark uses
    :func:`rate_for_completion_length` on its stored p90 completion length, so
    invalid generation buckets are never selected.

    Raises:
        ValueError: if a requested benchmark has no ``p90`` token statistic, or
            there are no valid generation buckets at all.
    """
    rates = gen_rates_from_calibration(calibration, size, n_loops)
    stats = calibration.get("tokens_per_item")
    if not isinstance(stats, Mapping):
        raise ValueError("calibration has no tokens_per_item token statistics")
    if benches is None:
        benches = sorted(str(name) for name in stats)
    result: dict[str, float] = {}
    for bench in benches:
        stat = stats.get(bench)
        if not isinstance(stat, Mapping) or "p90" not in stat:
            raise ValueError(
                f"no p90 completion-length statistic for benchmark {bench!r}"
            )
        result[bench] = rate_for_completion_length(rates, float(stat["p90"]))
    return result


def _samples_for(bench: str, samples_per_bench: dict[str, int]) -> int:
    if bench not in samples_per_bench:
        raise ValueError(
            f"missing samples_per_bench entry for {bench!r}; "
            "registered sample counts cannot be inferred silently"
        )
    samples = int(samples_per_bench[bench])
    if samples < 1:
        raise ValueError(f"samples for {bench!r} must be at least 1, got {samples}")
    return samples


def eval_hours(
    items: int, samples: int, tokens_per_item: float, tok_per_s: float
) -> float:
    """Hours needed to evaluate ``items`` items with ``samples`` samples each.

    ``tokens_per_item`` is the number of generated tokens for a single
    completion, so the total is ``items * samples * tokens_per_item``.
    """
    if items < 0:
        raise ValueError(f"items must be non-negative, got {items!r}")
    if samples < 0:
        raise ValueError(f"samples must be non-negative, got {samples!r}")
    if tokens_per_item < 0:
        raise ValueError(f"tokens_per_item must be non-negative, got {tokens_per_item!r}")
    _positive(tok_per_s, "tok_per_s")
    return items * samples * tokens_per_item / float(tok_per_s) / 3600.0


def fit_eval_sizes(
    tok_per_s: float | Mapping[str, float],
    tokens_per_item: dict[str, float],
    hours_per_variant: float,
    minimums: dict[str, int],
    caps: dict[str, int],
    priority: list[str],
    samples_per_bench: dict[str, int],
) -> dict[str, int]:
    """Fit per-benchmark item counts into ``hours_per_variant``.

    ``tokens_per_item`` is **per completion**.  ``samples_per_bench`` is
    **required** and multiplies the cost of every item; the registered map is
    :data:`DEFAULT_SAMPLES_PER_BENCH`.  There is no implicit one-sample
    fallback: forgetting the map would under-budget LCB/GPQA/AIME by 4-8x.

    ``tok_per_s`` is either one aggregate rate or a per-benchmark mapping.  The
    mapping form is what the Task 5b calibration feeds (see
    :func:`bench_rates_from_calibration`): each benchmark uses the generation
    rate measured at its expected completion length (short fast benchmarks must
    not inherit the 2k/64 batch rate, and long reasoning must not inherit it
    either).  Missing or ``None`` entries raise; there is no silent fallback or
    coercion of an invalid bucket into a usable rate.

    Every benchmark in ``minimums`` starts at its minimum.  Then the remaining
    time budget is spent in ``priority`` order, growing a benchmark to ``caps``
    before moving to the next one.  Sizes are whole items (floor division).

    Raises:
        ValueError: if a benchmark is missing from ``tokens_per_item``, the
            rate map or ``samples_per_bench``, a per-item cost/rate is not
            positive, a minimum exceeds its cap, a priority benchmark has no
            cap, or the minimums alone do not fit in the budget.
    """
    if hours_per_variant < 0:
        raise ValueError(
            f"hours_per_variant must be non-negative, got {hours_per_variant!r}"
        )
    per_bench_rate: dict[str, float] = {}
    if isinstance(tok_per_s, Mapping):
        for bench in set(tokens_per_item) | set(minimums) | set(priority):
            if bench not in tok_per_s:
                raise ValueError(f"no tok_per_s entry for benchmark {bench!r}")
            per_bench_rate[bench] = _positive(
                tok_per_s[bench], f"tok_per_s[{bench!r}]"
            )
    else:
        aggregate = _positive(tok_per_s, "tok_per_s")
        per_bench_rate = {
            bench: aggregate
            for bench in set(tokens_per_item) | set(minimums) | set(priority)
        }

    budget_seconds = float(hours_per_variant) * 3600.0

    per_item: dict[str, float] = {}
    for bench, cost in tokens_per_item.items():
        per_item[bench] = _positive(cost, f"tokens_per_item[{bench!r}]")

    samples: dict[str, int] = {}
    for bench in set(minimums) | set(priority):
        samples[bench] = _samples_for(bench, samples_per_bench)

    def cost_seconds(bench: str, items: int) -> float:
        return items * per_item[bench] * samples[bench] / per_bench_rate[bench]

    sizes: dict[str, int] = {}
    for bench, minimum in minimums.items():
        if bench not in per_item:
            raise ValueError(f"minimum for {bench!r} has no tokens_per_item entry")
        if minimum < 0:
            raise ValueError(f"minimum for {bench!r} must be non-negative")
        cap = caps.get(bench)
        if cap is not None and minimum > cap:
            raise ValueError(
                f"minimum for {bench!r} ({minimum}) exceeds its cap ({cap})"
            )
        sizes[bench] = int(minimum)

    used_seconds = sum(cost_seconds(bench, count) for bench, count in sizes.items())
    if used_seconds > budget_seconds + _EPSILON_TOKENS:
        raise ValueError(
            f"minimum evaluation sizes need {used_seconds / 3600:.3f} h but the "
            f"budget is {hours_per_variant:g} h"
        )

    for bench in priority:
        if bench not in per_item:
            raise ValueError(f"priority benchmark {bench!r} has no tokens_per_item entry")
        if bench not in caps:
            raise ValueError(
                f"priority benchmark {bench!r} has no cap; refusing to silently "
                "under-allocate it"
            )
        current = sizes.setdefault(bench, 0)
        cap = caps[bench]
        limit = max(cap, current)
        remaining = max(0.0, budget_seconds - used_seconds)
        cost = cost_seconds(bench, 1)
        affordable = int(remaining // cost)
        add = max(0, min(limit - current, affordable))
        sizes[bench] = current + add
        used_seconds += add * cost

    return sizes


def spill_detected(
    step_tok_per_s: Sequence[float],
    peak_reserved_bytes: int,
    total_bytes: int,
    *,
    reserved_fraction: float = SPILL_RESERVED_FRACTION,
    collapse_fraction: float = SPILL_COLLAPSE_FRACTION,
) -> bool:
    """Return ``True`` when a training measurement spilled into shared memory.

    Two independent signals, matching the controller's rule:

    * ``peak_reserved_bytes > reserved_fraction * total_bytes`` (WDDM has
      started backing allocations with system RAM), or
    * any timed step is slower than ``collapse_fraction`` times the median
      throughput of that same length (the step collapsed).

    An empty measurement is treated as a spill: there is no clean step to
    average.
    """
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes!r}")
    if reserved_fraction <= 0 or collapse_fraction <= 0:
        raise ValueError("threshold fractions must be positive")
    if peak_reserved_bytes < 0:
        raise ValueError(
            f"peak_reserved_bytes must be non-negative, got {peak_reserved_bytes!r}"
        )

    rates = [float(rate) for rate in step_tok_per_s]
    if not rates:
        return True
    if peak_reserved_bytes > reserved_fraction * total_bytes:
        return True

    median = statistics.median(rates)
    if median <= 0:
        return True
    return any(rate < collapse_fraction * median for rate in rates)


def max_trainable_length(
    try_fn: Callable[[int], bool], candidates: Sequence[int] = (4096, 6144, 8192)
) -> int:
    """Return the largest candidate accepted by ``try_fn``.

    ``try_fn(length)`` must return ``True`` when a short training probe at that
    sequence length succeeds **without spilling** and ``False`` when it runs out
    of memory or spills.  Candidates are tried from smallest to largest and the
    scan stops at the first failure: memory use is monotonic in sequence length,
    so every larger candidate would fail too.  Probing upward also means a
    caught CUDA OOM (which can leave a few GiB of autograd state alive until the
    process exits) happens *after* the successful probe, never before it.

    Raises:
        ValueError: if no candidate succeeds.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    best: int | None = None
    for length in sorted({int(candidate) for candidate in candidates}):
        if length <= 0:
            raise ValueError(f"candidate lengths must be positive, got {length}")
        if not try_fn(length):
            break
        best = length
    if best is None:
        raise ValueError(
            f"none of the candidate lengths {tuple(candidates)} fit in GPU memory"
        )
    return best


def train_hours(tokens: int, train_tok_per_s: float) -> float:
    """Hours needed to train on ``tokens`` at ``train_tok_per_s`` tokens/s."""
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens!r}")
    _positive(train_tok_per_s, "train_tok_per_s")
    return tokens / float(train_tok_per_s) / 3600.0
