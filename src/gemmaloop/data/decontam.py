"""Benchmark decontamination (OpenThoughts3 recipe).

Three levels are applied in increasing cost, exactly as reference 07, section 2:

1. exact match after :func:`normalize`,
2. shared 13-grams,
3. normalised Indel (LCS) similarity ``>= 0.75``.

The third level is exact over every length-compatible evaluation text: for
normalised Indel similarity ``1 - distance/(len_a + len_b)`` to be at least
0.75, the lengths must satisfy ``2*min/(len_a+len_b) >= 0.75`` (equivalently
``min >= 0.6*max``).  RapidFuzz's ``cdist`` runs the comparison with
``workers=-1``.  The embedding/LLM level of the reference is intentionally not
implemented.

The tokenizer is stored on the index so ``build`` and ``check`` always use the
same tokenisation.  :func:`EvalIndex.build` defaults to a tiny word tokenizer so
unit tests stay offline; the data pipeline passes the Gemma 4 tokenizer and
records its name/version in the manifest.
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache
from typing import Iterable, Literal

from rapidfuzz import process
from rapidfuzz.distance import Indel

Verdict = Literal["exact", "ngram13", "indel75"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)

INDEL_THRESHOLD = 0.75
# 2*min/(a+b) >= 0.75  <=>  min >= 0.6*max
MIN_LENGTH_RATIO = 0.6

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace and drop punctuation."""

    text = (s or "").lower()
    text = _WS_RE.sub(" ", text)
    text = _NON_WORD_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def ngrams(tokens: list[str], n: int = 13) -> set[tuple[str, ...]]:
    """Return all contiguous ``n``-grams of ``tokens``.

    Texts shorter than ``n`` tokens produce an empty set, which matches the
    reference implementation.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


class _WordTokenizer:
    """Dependency-free tokenizer used when no tokenizer is supplied."""

    name = "word-regex"

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall((text or "").lower())


_DEFAULT_TOKENIZER = _WordTokenizer()


def _tokenize(tokenizer, text: str) -> list[str]:
    if tokenizer is None:
        tokenizer = _DEFAULT_TOKENIZER
    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        return [str(token) for token in tokenize(text or "")]
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text or "", add_special_tokens=False)
        except TypeError:
            encoded = encode(text or "")
        return [str(token) for token in encoded]
    if callable(tokenizer):
        return [str(token) for token in tokenizer(text or "")]
    raise TypeError(f"{type(tokenizer).__name__} has neither tokenize() nor encode()")


def tokenizer_name(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "name", None)
    return str(name) if name else type(tokenizer).__name__


@lru_cache(maxsize=1_000_000)
def _stable_token_hash(token: str) -> int:
    """Process-independent token hash (never Python's salted ``hash``)."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _stable_gram_hash(gram: tuple[str, ...]) -> int:
    """FNV-1a over stable token hashes."""

    value = _FNV_OFFSET
    for token in gram:
        value ^= _stable_token_hash(token)
        value = (value * _FNV_PRIME) & _MASK64
    return value


class EvalIndex:
    """An immutable index of evaluation texts to reject training rows against."""

    def __init__(
        self,
        texts: Iterable[str],
        *,
        n: int = 13,
        tokenizer=None,
    ) -> None:
        self.n = int(n)
        self.tokenizer = tokenizer if tokenizer is not None else _DEFAULT_TOKENIZER

        raw: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text is None:
                continue
            value = str(text)
            normalized = normalize(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            raw.append(value)

        self.texts = tuple(raw)
        self.normalized = tuple(normalize(text) for text in raw)
        self._exact = frozenset(self.normalized)
        self._ngram_hashes: set[int] = set()
        for text in raw:
            tokens = _tokenize(self.tokenizer, text)
            for gram in ngrams(tokens, self.n):
                self._ngram_hashes.add(_stable_gram_hash(gram))

        # Length-sorted texts make the exact Indel pass cheap without a
        # lossy candidate prefilter: only pairs that can reach the threshold
        # are handed to RapidFuzz.
        ordered = sorted(zip((len(text) for text in self.normalized), self.normalized))
        self._sorted_lengths = [length for length, _ in ordered]
        self._sorted_texts = [text for _, text in ordered]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        n: int = 13,
        tokenizer=None,
    ) -> "EvalIndex":
        return cls(texts, n=n, tokenizer=tokenizer)

    def __len__(self) -> int:
        return len(self.texts)

    def _indel_similarity(self, query: str) -> bool:
        length = len(query)
        if not length:
            return False
        low = bisect_left(self._sorted_lengths, MIN_LENGTH_RATIO * length)
        high = bisect_right(self._sorted_lengths, length / MIN_LENGTH_RATIO)
        if low >= high:
            return False
        scores = process.cdist(
            [query],
            self._sorted_texts[low:high],
            scorer=Indel.normalized_similarity,
            score_cutoff=INDEL_THRESHOLD,
            workers=-1,
        )
        return bool(scores.size and scores.max() >= INDEL_THRESHOLD)

    def check(self, text: str) -> Verdict | None:
        """Return the first contamination criterion that fires, or ``None``."""

        if not self.texts or not (text or "").strip():
            return None

        normalized = normalize(text)
        if normalized and normalized in self._exact:
            return "exact"

        tokens = _tokenize(self.tokenizer, text)
        for gram in ngrams(tokens, self.n):
            if _stable_gram_hash(gram) in self._ngram_hashes:
                return "ngram13"

        if self._indel_similarity(normalized):
            return "indel75"
        return None


__all__ = [
    "EvalIndex",
    "INDEL_THRESHOLD",
    "MIN_LENGTH_RATIO",
    "Verdict",
    "ngrams",
    "normalize",
    "tokenizer_name",
]
