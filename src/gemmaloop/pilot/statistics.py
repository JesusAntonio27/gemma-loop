"""Problem-cluster bootstrap and the registered pilot endpoints.

Every NLL here is a per-token calibrated NLL averaged within a problem
cluster.  Resampling is paired by cluster (the same clusters for both arms) and
uses 10,000 iterations with seed 0 by default, exactly as registered.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "arm_endpoints",
    "bootstrap_ci",
    "cluster_mean",
    "delta_bar",
    "monotonic_ok",
    "paired_bootstrap_ci",
    "percentile_ci",
]

DEFAULT_ITERS = 10_000
DEFAULT_SEED = 0
MONOTONIC_TOLERANCE = 0.005


def _as_sum_tokens(value) -> tuple[float, float]:
    """Read ``(summed loss, token count)`` from one cluster value.

    A bare number is a per-token mean whose token count is unknown; treating it
    as one token keeps the old synthetic fixtures meaningful.  The production
    scorer stores ``{"sum_loss": ..., "tokens": ...}``.
    """

    if isinstance(value, Mapping):
        total = value.get("sum_loss", value.get("sum"))
        tokens = value.get("tokens", value.get("n"))
        if total is None or tokens is None:
            raise ValueError(f"cluster value lacks sum_loss/tokens: {value!r}")
        return float(total), float(tokens)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return float(value), 1.0


def cluster_mean(values: Mapping[str, Any]) -> float:
    """Token-weighted ``C_N = sum_i S_iN / sum_i n_i`` over problems."""

    if not values:
        raise ValueError("cannot average an empty cluster mapping")
    total = 0.0
    tokens = 0.0
    for value in values.values():
        sum_loss, count = _as_sum_tokens(value)
        total += sum_loss
        tokens += count
    if tokens <= 0:
        raise ValueError("cluster token counts must be positive")
    return float(total / tokens)


def _arrays(values: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = sorted(values)
    sums = np.asarray([_as_sum_tokens(values[key])[0] for key in keys], dtype=float)
    tokens = np.asarray([_as_sum_tokens(values[key])[1] for key in keys], dtype=float)
    return keys, sums, tokens


def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    if len(samples) == 0:
        raise ValueError("no bootstrap samples")
    low, high = np.percentile(np.asarray(samples, dtype=float), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_ci(
    values: Mapping[str, Any],
    *,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of the token-weighted problem mean."""

    if iters <= 0:
        raise ValueError("iters must be positive")
    if not values:
        raise ValueError("cannot bootstrap an empty cluster mapping")
    _keys, sums, tokens = _arrays(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sums.size, size=(iters, sums.size))
    ratios = sums[indices].sum(axis=1) / tokens[indices].sum(axis=1)
    return percentile_ci(ratios, alpha=alpha)


def paired_bootstrap_ci(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of the paired token-weighted difference ``C_a - C_b``.

    Whole problems are resampled with their token counts; tokens are never
    resampled independently.
    """

    if iters <= 0:
        raise ValueError("iters must be positive")
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two arms share no problem clusters")
    sums_a = np.asarray([_as_sum_tokens(a[key])[0] for key in common], dtype=float)
    sums_b = np.asarray([_as_sum_tokens(b[key])[0] for key in common], dtype=float)
    tokens_a = np.asarray([_as_sum_tokens(a[key])[1] for key in common], dtype=float)
    tokens_b = np.asarray([_as_sum_tokens(b[key])[1] for key in common], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(common), size=(iters, len(common)))
    ratios = (
        sums_a[indices].sum(axis=1) / tokens_a[indices].sum(axis=1)
        - sums_b[indices].sum(axis=1) / tokens_b[indices].sum(axis=1)
    )
    return percentile_ci(ratios, alpha=alpha)


def monotonic_ok(
    nll: Mapping[int, float], tolerance: float = MONOTONIC_TOLERANCE
) -> bool:
    """The registered point-estimate monotonicity filter.

    ``cNLL(N+1) - cNLL(N) <= 0.005`` for N = 1, 2, 3 and ``cNLL(8) <= cNLL(1)``.
    Missing registered N values fail closed (the filter is then not satisfied).
    """

    required = (1, 2, 3, 8)
    if any(int(n) not in nll for n in required):
        return False
    for n in (1, 2, 3):
        if float(nll[n + 1]) - float(nll[n]) > tolerance:
            return False
    return float(nll[8]) <= float(nll[1])


def arm_endpoints(
    nll_by_n: Mapping[int, Mapping[str, Any]],
    oh_by_n: Mapping[int, Mapping[str, Any]] | None = None,
    *,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compute the registered endpoints from per-problem loss plus token counts.

    ``nll_by_n`` maps N -> {problem -> {"sum_loss", "tokens"}}; O-H is the
    block-matched control at N=1.  Point estimates are the registered
    token-weighted ``C_N`` and the bootstrap resamples whole problems carrying
    their token counts.
    """

    point = {int(n): cluster_mean(values) for n, values in nll_by_n.items()}
    required = (1, 2, 3, 4, 6, 8)
    missing = [n for n in required if n not in point]
    if missing:
        raise ValueError(f"missing registered N values: {missing}")

    result: dict = {"nll": {str(n): point[n] for n in required}}

    result["delta_loop"] = point[1] - point[2]
    result["delta_loop_ci"] = list(
        paired_bootstrap_ci(nll_by_n[1], nll_by_n[2], iters=iters, seed=seed)
    )
    # Lower bound > 0 is the eligibility test.
    result["delta_loop_lower"] = result["delta_loop_ci"][0]

    if oh_by_n:
        oh_point = {int(n): cluster_mean(values) for n, values in oh_by_n.items()}
        oh_clusters = oh_by_n.get(1) or oh_by_n[sorted(oh_by_n)[0]]
        point_oh = cluster_mean(oh_clusters)
        result["oh_nll"] = {str(n): value for n, value in oh_point.items()}
        result["delta_arm"] = point_oh - point[2]
        result["delta_arm_ci"] = list(
            paired_bootstrap_ci(oh_clusters, nll_by_n[2], iters=iters, seed=seed)
        )
        result["delta_bar_arm"] = float(
            sum(point_oh - point[n] for n in (2, 3, 4)) / 3.0
        )
        result["delta_bar_arms"] = {
            str(n): point_oh - point[n] for n in (2, 3, 4)
        }
    else:
        result["delta_arm"] = None
        result["delta_bar_arm"] = None

    result["monotonic"] = monotonic_ok({n: point[n] for n in required})
    return result
