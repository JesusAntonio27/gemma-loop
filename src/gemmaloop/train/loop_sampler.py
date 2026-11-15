"""Deterministic sampling of the number of block passes during training.

The sampler is a pure function of ``(seed, micro_step)``: it hashes the pair
with SHA-256 and maps the digest uniformly onto the cumulative distribution of
``{1, 2, 3, 4}``.  Python's built-in :func:`hash` is deliberately not used: it
is salted per process and would make resumed runs pick a different sequence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

DEFAULT_LOOP_PROBS: tuple[float, ...] = (0.25, 0.40, 0.20, 0.15)

_TWO_POW_256 = float(2**256)


def _validate_probs(probs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(p) for p in probs)
    if not values:
        raise ValueError("probs must not be empty")
    if any(p < 0.0 for p in values):
        raise ValueError(f"probs must be non-negative, got {values!r}")
    total = sum(values)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"probs must sum to 1, got {total!r}")
    return values


def sample_n_loops(
    seed: int,
    micro_step: int,
    probs: tuple[float, ...] = DEFAULT_LOOP_PROBS,
) -> int:
    """Return the number of block passes (1..len(probs)) for this micro step.

    ``micro_step`` is the 0-based index of the training micro batch, so a run
    resumed at micro step ``k`` continues the exact same sequence.
    """
    values = _validate_probs(probs)
    digest = hashlib.sha256(f"{seed}:{micro_step}".encode()).hexdigest()
    u = int(digest, 16) / _TWO_POW_256

    cumulative = 0.0
    for n, probability in enumerate(values, start=1):
        cumulative += probability
        if u < cumulative:
            return n
    # Floating-point rounding can only land here for u ~= 1; the last bucket wins.
    return len(values)
