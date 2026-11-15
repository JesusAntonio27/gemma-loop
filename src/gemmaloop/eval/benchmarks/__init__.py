"""Shared helpers for the benchmark adapters.

Every adapter exposes the same small surface::

    @dataclass class Item: item_id: str; prompt: str; answer: str | None; meta: dict
    load_items(limit: int | None = None, seed: int = 0) -> list[Item]
    build_messages(item: Item) -> list[dict]
    score(item: Item, completion: str) -> bool

The helpers here only cover what the adapters share: loading rows (streaming
for small samples so tests never download a full dataset), deterministic
subsampling, thinking-block removal, multiple-choice prompt formatting and
answer-letter extraction.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from typing import Any, Callable, Iterable, Sequence

from datasets import load_dataset

THINKING_RE = re.compile(r"<\|channel>thought\b.*?<channel\|>", re.DOTALL)
STREAMING_LIMIT_MAX = 64


def strip_thinking(text: str) -> str:
    """Drop Gemma 4 ``<|channel>thought ... <channel|>`` sections.

    The template puts the reasoning trace before the final answer.  Keeping it
    would let the answer-extraction regexes pick numbers or letters out of the
    trace (reference 04, trap 11; reference 08, trap 16).
    """

    return THINKING_RE.sub("", text or "")


def load_rows(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "test",
    revision: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> list[dict]:
    """Load dataset rows, optionally with a small seeded subsample.

    Real runs load the complete split and select with :func:`stable_sample`, so
    every variant sees the same items for the same seed and the sample is not
    biased to the first rows of the stream.  Only tiny test limits
    (``<= STREAMING_LIMIT_MAX``) use streaming, which keeps download tests from
    pulling hundreds of megabytes of vision parquet files.
    """

    call_kwargs: dict[str, Any] = {"split": split, **kwargs}
    if config is not None:
        call_kwargs["name"] = config
    if revision is not None:
        call_kwargs["revision"] = revision

    if limit is not None and limit <= STREAMING_LIMIT_MAX:
        iterable = load_dataset(dataset_id, streaming=True, **call_kwargs)
        buffer_size = max(64, min(1_000, limit * 10))
        shuffled = iterable.shuffle(seed=seed, buffer_size=buffer_size)
        return list(itertools.islice(shuffled, limit))

    rows = list(load_dataset(dataset_id, **call_kwargs))
    return stable_sample(rows, limit, seed) if limit is not None else rows


def _default_sample_key(row: Any, index: int) -> str:
    if isinstance(row, dict):
        for field in ("question_id", "id", "idx"):
            if row.get(field) is not None:
                return str(row[field])
        return str(index)
    return str(row)


def stable_sample(
    rows: Sequence[Any],
    limit: int | None,
    seed: int = 0,
    key: Callable[[Any, int], str] | None = None,
) -> list[Any]:
    """Return a SHA-256 ranked, order-independent sample of ``rows``.

    Ranking by ``sha256(f"{seed}:{stable_id}")`` is deterministic across
    processes and independent of row order, so the same seed selects the same
    items for every model variant as long as the dataset ids are unchanged.
    Items are returned in their original dataset order.
    """

    if limit is None or limit >= len(rows):
        return list(rows)

    def rank(index: int) -> str:
        value = key(rows[index], index) if key else _default_sample_key(rows[index], index)
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    chosen = sorted(range(len(rows)), key=lambda index: (rank(index), index))[:limit]
    return [rows[index] for index in sorted(chosen)]


def image_part(image: Any) -> dict:
    """Gemma 4 processor content part for an image.

    The processor accepts ``{"type": "image", "image": PIL.Image}`` (see
    reference 08, Know-how section 4).  Callers may also use
    ``{"type": "image", "url": ...}``; both render as ``<|image|>``.
    """

    return {"type": "image", "image": image}


def user_message(content: str | list[dict]) -> list[dict]:
    return [{"role": "user", "content": content}]


def format_options(options: Sequence[str], letters: str = "ABCDEFGHIJ") -> str:
    lines = [f"({letter}) {option}" for letter, option in zip(letters, options)]
    return "\n".join(lines)


def mcq_prompt(question: str, options: Sequence[str], letters: str = "ABCDEFGHIJ") -> str:
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{format_options(options, letters)}\n\n"
        "Answer with the option letter only."
    )


_GOLD_PAREN_RE = re.compile(r"^\s*\(\s*([A-Za-z])\s*\)")
_GOLD_LEADING_RE = re.compile(r"^\s*([A-Za-z])(?:$|[\s.,:;)])")


def extract_gold_letter(value: Any, letters: str = "ABCDEFGHIJ") -> str | None:
    """Return the letter encoded in a benchmark's gold answer.

    Handles the formats seen in the datasets: ``"A"``, ``"(A)"``, ``"A."``,
    ``"A. Northeast"`` and ``"(C) 1"``.  It deliberately does *not* scan for
    an arbitrary letter inside a word: a gold field like ``"Northeast"`` is not
    silently turned into ``"N"``.
    """

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for entry in value:
            letter = extract_gold_letter(entry, letters)
            if letter is not None:
                return letter
        return None

    text = str(value).strip()
    if not text:
        return None
    allowed = letters.upper()

    match = _GOLD_PAREN_RE.match(text)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()

    match = _GOLD_LEADING_RE.match(text)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()

    if len(text) == 1 and text.upper() in allowed:
        return text.upper()
    return None


_MARKDOWN_RE = re.compile(r"[*_$`]")


def _strip_markdown(text: str) -> str:
    """Remove the formatting characters that hide a bare answer letter."""

    return _MARKDOWN_RE.sub("", text or "")


def extract_choice_status(
    text: str, letters: str = "ABCDEFGHIJ"
) -> tuple[str | None, str]:
    """Extract a multiple-choice letter and report whether parsing succeeded.

    The letter must be an **uppercase** member of ``letters`` and must not be
    the first letter of a word (``B.`` and ``(B)`` are fine, ``Box`` is not).
    Markdown ``*``/``_``/``$``/backticks are stripped first, and the thinking
    block is always removed.  The last explicit answer statement wins; a
    parenthesised letter is the second choice and a line-leading standalone
    letter the third.  Returns ``(letter_or_None, "ok"|"unparsed")``.
    """

    cleaned = _strip_markdown(strip_thinking(text))
    allowed = letters.upper()
    char_class = re.escape(allowed)

    explicit_re = re.compile(
        rf"(?i:final\s+answer|answer)\s*(?:is|:)\s*[\(\[]?\s*"
        rf"([{char_class}])\s*[\)\]]?(?![A-Za-z0-9])"
    )
    boxed_re = re.compile(rf"\\boxed\s*\{{\s*([{char_class}])\s*\}}")
    paren_re = re.compile(rf"\(\s*([{char_class}])\s*\)")
    standalone_re = re.compile(
        rf"(?m)^[ \t]*[\(\[]?[ \t]*([{char_class}])[ \t]*[\)\]]?"
        rf"[ \t]*(?:[.):]|[ \t]*$)"
    )
    sentence_re = re.compile(rf"(?<![A-Za-z0-9])([{char_class}])(?=[.):])")

    # Explicit statements and ``\boxed{X}`` are equivalent: take the rightmost.
    explicit = [(match.start(), match.group(1)) for match in explicit_re.finditer(cleaned)]
    explicit += [(match.start(), match.group(1)) for match in boxed_re.finditer(cleaned)]
    if explicit:
        return max(explicit, key=lambda pair: pair[0])[1], "ok"

    parenthesised = paren_re.findall(cleaned)
    if parenthesised:
        return parenthesised[-1], "ok"

    standalone = standalone_re.findall(cleaned)
    if standalone:
        return standalone[-1], "ok"

    # Last resort: a letter followed by sentence punctuation ("We choose I.").
    sentence = sentence_re.findall(cleaned)
    if sentence:
        return sentence[-1], "ok"

    return None, "unparsed"


def extract_choice(text: str, letters: str = "ABCDEFGHIJ") -> str | None:
    """Return the extracted multiple-choice letter, or ``None``."""

    return extract_choice_status(text, letters)[0]


def _balanced_braces(text: str, opening: int) -> str | None:
    """Content inside the brace that opens at ``opening`` (nested-safe)."""

    depth = 0
    index = opening
    while index < len(text):
        character = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
        index += 1
    return None


def last_boxed(text: str, commands: tuple[str, ...] = ("boxed", "fbox")) -> str | None:
    r"""Return the last ``\boxed{...}``/``\fbox{...}`` content, braces balanced.

    The official MATH-V and MathArena graders take the last box; unlike a flat
    regex, this handles ``\boxed{\text{70}}`` and ``\boxed{\frac{1}{2}}``.
    """

    pattern = re.compile(
        r"\\(?:" + "|".join(re.escape(command) for command in commands) + r")\s*\{"
    )
    matches = list(pattern.finditer(text or ""))
    for match in reversed(matches):
        content = _balanced_braces(text, match.end() - 1)
        if content is not None:
            return content
    return None


def rows_of(items: Iterable[dict]) -> list[dict]:
    return [dict(item) for item in items]
