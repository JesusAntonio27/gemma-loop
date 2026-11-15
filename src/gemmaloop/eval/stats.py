"""Statistics for evaluation results.

Implements the error bars of the evaluation plan (reference 04, Know-how §5):
Wilson score intervals for a single accuracy, paired item-level differences
with a CLT standard error, a paired bootstrap as a robustness check, Holm's
step-down correction, and Miller's minimum detectable effect (MDE).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

# 95 % bilateral normal quantile; kept as an explicit default in ``wilson``.
Z_95 = 1.959964


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")

    z2 = z * z
    denominator = n + z2
    centre = (k + z2 / 2) / denominator
    margin = z * math.sqrt(k * (n - k) / n + z2 / 4) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _paired_differences(
    a: dict[str, float], b: dict[str, float]
) -> tuple[list[str], np.ndarray]:
    """Return the sorted common item ids and ``a - b`` for those items."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("a and b have no items in common")
    return common, np.array([a[item] - b[item] for item in common], dtype=float)


def paired_diff(
    a: dict[str, float], b: dict[str, float]
) -> tuple[float, float, float, float]:
    """Paired mean difference ``a - b`` over common items, with a CLT interval.

    Returns ``(mean, se, lo, hi)`` where ``se`` is the sample standard
    deviation of the per-item differences divided by ``sqrt(n)``.  A single
    common item has no estimable spread, so its ``se`` is reported as ``0``.
    """
    _, diffs = _paired_differences(a, b)
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    half_width = float(norm.ppf(0.975)) * se
    return mean, se, mean - half_width, mean + half_width


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    iters: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the paired mean difference.

    Items (pairs) are resampled with replacement; the two models are never
    resampled independently.  Returns the 2.5th and 97.5th percentiles.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    _, diffs = _paired_differences(a, b)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(iters, len(diffs)))
    means = diffs[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down rejection decisions, returned in the input order."""
    m = len(pvalues)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[index] <= threshold:
            rejected[index] = True
        else:
            break
    return rejected


def mde(
    n: int,
    k: int,
    alpha: float = 0.05,
    power: float = 0.8,
    omega2: float = 1 / 12,
    sigma2: float = 1 / 6,
) -> float:
    """Minimum detectable effect (Miller 2024, eq. 10).

    ``(z_{1-alpha/2} + z_power) * sqrt((omega2 + 2 * sigma2 / k) / n)``
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * math.sqrt((omega2 + 2 * sigma2 / k) / n))
