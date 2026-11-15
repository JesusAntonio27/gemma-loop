"""Agreement between the blind AI judge and the human rater.

The human rates the same rendered pairs as the judge, in a randomised A/B
order and without seeing variant names (see ``scripts/human_rate.py``).  The
headline number is not raw agreement but Cohen's kappa, because the raters can
tie and the classes are often imbalanced: with skewed marginals a high ``p_o``
can coexist with a low kappa (the kappa paradox from ``references/06`` §d).

``kappa`` comes from :func:`sklearn.metrics.cohen_kappa_score`; the 95 %
interval is a percentile bootstrap over pairs, seeded so a rerun is identical.
Report ``p_o``, ``kappa``, its interval and the confusion matrix together — a
single kappa hides whether the raters disagreed on ties or on a minority class.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

DEFAULT_ITERS = 10000
DEFAULT_SEED = 0

# Canonical order for the labels of the creative evaluation.  Other labels are
# accepted (the function is generic) and appended in sorted order.
RATING_LABELS = ("A", "B", "tie")


def _as_ratings(values: Sequence[str], name: str) -> list[str]:
    ratings = list(values)
    if not ratings:
        raise ValueError(f"{name} needs at least one rating")
    for rating in ratings:
        if not isinstance(rating, str) or not rating:
            raise ValueError(f"{name} contains an empty or non-string label")
    return ratings


def _labels(judge: list[str], human: list[str]) -> list[str]:
    """Observed labels, canonical first, so the confusion matrix is stable."""
    observed = set(judge) | set(human)
    ordered = [label for label in RATING_LABELS if label in observed]
    return ordered + sorted(observed.difference(RATING_LABELS))


def _kappa(judge: np.ndarray, human: np.ndarray, labels: list[str]) -> float:
    with warnings.catch_warnings():
        # A bootstrap replicate can contain a single class; sklearn then warns
        # and returns nan, which is the mathematically correct value here.
        warnings.simplefilter("ignore")
        value = cohen_kappa_score(judge, human, labels=labels)
    return float(value)


def _confusion(
    judge: list[str], human: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    """Rows are the human label, columns the judge label."""
    matrix = {
        human_label: {judge_label: 0 for judge_label in labels}
        for human_label in labels
    }
    for judge_label, human_label in zip(judge, human):
        matrix[human_label][judge_label] += 1
    return matrix


def agreement(
    judge: list[str],
    human: list[str],
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Return ``p_o``, Cohen's kappa, its seeded bootstrap CI and confusion.

    ``judge`` and ``human`` are aligned lists of labels (``A``/``B``/``tie``
    for the creative evaluation) of equal, non-zero length.  ``p_o`` is the
    fraction of exact agreements.  ``kappa_ci95`` is the ``(2.5, 97.5)``
    percentile interval of ``iters`` pair resamples drawn with ``seed``;
    replicates with an undefined kappa (a single observed class) are skipped.
    ``confusion`` is a nested mapping ``human label -> judge label -> count``
    whose labels preserve the canonical ``A``, ``B``, ``tie`` order.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")

    judge = _as_ratings(judge, "judge")
    human = _as_ratings(human, "human")
    if len(judge) != len(human):
        raise ValueError("judge and human must have the same number of ratings")

    labels = _labels(judge, human)
    judge_array = np.asarray(judge, dtype=object)
    human_array = np.asarray(human, dtype=object)

    p_o = float(np.mean(judge_array == human_array))
    kappa = _kappa(judge_array, human_array, labels)

    rng = np.random.default_rng(seed)
    n = len(judge)
    boot = np.empty(iters, dtype=float)
    for iteration in range(iters):
        indices = rng.integers(0, n, n)
        boot[iteration] = _kappa(
            judge_array[indices], human_array[indices], labels
        )
    finite = boot[np.isfinite(boot)]
    if finite.size:
        low, high = np.percentile(finite, [2.5, 97.5])
        ci95 = (float(low), float(high))
    else:
        ci95 = (float("nan"), float("nan"))

    return {
        "p_o": p_o,
        "kappa": kappa,
        "kappa_ci95": ci95,
        "confusion": _confusion(judge, human, labels),
    }
