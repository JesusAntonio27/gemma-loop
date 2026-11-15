"""Pinned upstream snapshot and chat-template identity helpers.

Every model, tokenizer and processor load in the project must pass
:data:`MODEL_REVISION` so a rerun sees the same weights and the same chat
template.  Google already revised the Gemma 4 template once (the cached header
says 2026-07-09), so the template bytes are part of the data/run provenance:
the data manifest records their SHA-256 and training refuses to start when the
template in the cache no longer matches the one the data was built with.
"""

from __future__ import annotations

import hashlib
import json

E2B_REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"
# Teacher snapshots pinned in prereg-v1 section 11.  These are the revisions
# the offline Stage 1 top-k precompute must load; an unknown teacher key still
# fails closed in :func:`teacher_revision`.
E4B_REVISION = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
TWELVE_B_QAT_REVISION = "b6ed86275a6a5735884e208bfed95b445a684ca2"

TEACHER_MODEL_IDS: dict[str, str] = {
    "e4b": "google/gemma-4-E4B-it",
    "12b_qat": "google/gemma-4-12B-it-qat-q4_0-unquantized",
}

TEACHER_REVISIONS: dict[str, str] = {
    "e4b": E4B_REVISION,
    "12b_qat": TWELVE_B_QAT_REVISION,
}

MODEL_REVISIONS: dict[str, str | None] = {
    "e2b": E2B_REVISION,
    "e4b": E4B_REVISION,
    "12b_qat": TWELVE_B_QAT_REVISION,
}

# Backwards-compatible alias for E2B-only call sites (data build, tests).
MODEL_REVISION = E2B_REVISION


def model_revision(size: str) -> str:
    """Return the pinned snapshot for ``size`` or fail closed when unknown."""

    key = str(size).lower()
    if key not in MODEL_REVISIONS:
        raise ValueError(f"unknown model size {size!r}")
    revision = MODEL_REVISIONS[key]
    if not revision:
        raise RuntimeError(
            f"no pinned snapshot commit for {key!r}: set MODEL_REVISIONS[{key!r}] "
            "before loading it (TODO M2). Refusing to load an unpinned checkpoint."
        )
    return revision


def teacher_revision(name: str) -> str:
    """Return the pinned revision for a Stage 1 teacher, failing closed.

    ``name`` is ``"e4b"`` (the regular E4B checkpoint) or ``"12b_qat"`` (the
    Google QAT q4_0 *unquantized* checkpoint loaded with bitsandbytes nf4).
    """

    key = str(name).lower()
    if key not in TEACHER_REVISIONS:
        raise ValueError(
            f"unknown teacher {name!r}; expected one of {tuple(TEACHER_REVISIONS)}"
        )
    revision = TEACHER_REVISIONS[key]
    if not revision:
        raise RuntimeError(
            f"no pinned snapshot commit for teacher {key!r}: refusing to load "
            "an unpinned checkpoint"
        )
    return revision


def teacher_model_id(name: str) -> str:
    """Return the Hugging Face repo id for a Stage 1 teacher."""

    key = str(name).lower()
    if key not in TEACHER_MODEL_IDS:
        raise ValueError(
            f"unknown teacher {name!r}; expected one of {tuple(TEACHER_MODEL_IDS)}"
        )
    return TEACHER_MODEL_IDS[key]


def chat_template_sha256(tokenizer) -> str:  # noqa: ANN001 - HF tokenizer protocol
    """Return the SHA-256 of ``tokenizer.chat_template`` (``""`` when absent)."""

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return ""
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


__all__ = [
    "E2B_REVISION",
    "E4B_REVISION",
    "MODEL_REVISION",
    "MODEL_REVISIONS",
    "TEACHER_MODEL_IDS",
    "TEACHER_REVISIONS",
    "TWELVE_B_QAT_REVISION",
    "chat_template_sha256",
    "model_revision",
    "teacher_model_id",
    "teacher_revision",
]
