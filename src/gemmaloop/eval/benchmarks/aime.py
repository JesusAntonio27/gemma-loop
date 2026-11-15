"""AIME 2026 adapter.

Dataset: ``math-ai/aime26`` (Apache 2.0), split ``test``, 30 problems with a
string ``answer`` in 0-999.

Scoring follows the MathArena grader (reference 04, section 3): remove the
Gemma 4 thinking block, take the *last* ``\\boxed{}``/``\\fbox{}`` and compare
the normalised integer.  No judge is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import last_boxed, load_rows, strip_thinking, user_message

DATASET_ID = "math-ai/aime26"
REVISION = "79037aebdb6580008fb960d17cb21fd3099083e3"
SPLIT = "test"


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def _normalise_integer(value: str) -> str:
    """Normalise the content of a box to a canonical integer string."""

    value = value.strip().strip("$").strip()
    value = value.replace(",", "").replace(" ", "").replace("\u00a0", "")
    value = value.replace("{,}", "").replace("\\,", "")
    wrapped = re.fullmatch(r"\\(?:text|mbox)\{([+-]?\d+)\}", value)
    if wrapped:
        value = wrapped.group(1)
    if re.fullmatch(r"[+-]?\d+", value):
        return str(int(value))
    return value


def format_prompt(problem: str) -> str:
    return (
        f"{problem.strip()}\n\n"
        "Put the final integer (0-999) alone inside \\boxed{}."
    )


def item_from_row(row: dict[str, Any]) -> Item:
    answer = row.get("answer")
    return Item(
        item_id=str(row["id"]),
        prompt=format_prompt(str(row["problem"])),
        answer=None if answer is None else _normalise_integer(str(answer)),
        meta={"source": DATASET_ID},
    )


def load_items(limit: int | None = None, seed: int = 0) -> list[Item]:
    rows = load_rows(
        DATASET_ID,
        split=SPLIT,
        revision=REVISION,
        limit=limit,
        seed=seed,
    )
    return [item_from_row(row) for row in rows]


def build_messages(item: Item) -> list[dict]:
    return user_message(item.prompt)


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the last boxed integer (if any) plus its parse status."""

    boxed = last_boxed(strip_thinking(completion))
    if boxed is None:
        return None, "unparsed"
    return boxed, "ok"


def score(item: Item, completion: str) -> bool:
    if item.answer is None:
        return False

    text = strip_thinking(completion)
    boxed = last_boxed(text)
    if boxed is None:
        return False

    predicted = _normalise_integer(boxed)
    return predicted == _normalise_integer(str(item.answer))
