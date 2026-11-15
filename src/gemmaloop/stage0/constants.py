"""Dependency-free Stage 0 constants shared by local and remote entry points.

This module must not import anything from ``gemmaloop``: ``scripts/modal_stage0.py``
is mounted into a small Modal image that only installs the sampling stack, and
its module-level import closure has to stay inside that image.
"""

from __future__ import annotations

# Part 1 does not decontaminate against the Part 2 creative eval briefs; when the
# briefs are frozen, Part 2 decontaminates them against this pool (reverse
# direction).  The string is recorded in the pool manifest so a paid Modal run
# can distinguish this registered mode from the refused smoke flag.
PART1_CREATIVE_BRIEFS = (
    "excluded: Part 2 decontaminates its briefs against this pool"
)

__all__ = ["PART1_CREATIVE_BRIEFS"]
