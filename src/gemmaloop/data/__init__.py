"""Decontaminated SFT data for the short adaptation phase."""

from .decontam import EvalIndex, ngrams, normalize
from .mix import Manifest, build_mix, message_tokens
from .sources import (
    SOURCES,
    SourceSpec,
    iter_rows,
    reasoning_defect,
    strip_defect_markers,
    to_messages,
)

__all__ = [
    "EvalIndex",
    "Manifest",
    "SOURCES",
    "SourceSpec",
    "build_mix",
    "iter_rows",
    "message_tokens",
    "ngrams",
    "normalize",
    "reasoning_defect",
    "strip_defect_markers",
    "to_messages",
]
