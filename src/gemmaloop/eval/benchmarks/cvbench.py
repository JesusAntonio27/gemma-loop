"""CV-Bench 2D adapter.

Dataset: ``nyu-visionx/CV-Bench`` (Apache 2.0), config ``2D``, split ``test``
with 1,438 examples (spatial relations and counting).  The 3D split (1,200) is
a negative control and is not loaded here.

Answers are letters ``(A)``-``(F)`` in the ``answer`` field; scoring is exact
letter accuracy (reference 08, section (c)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import (
    extract_choice,
    extract_choice_status,
    extract_gold_letter,
    format_options,
    image_part,
    load_rows,
    user_message,
)

DATASET_ID = "nyu-visionx/CV-Bench"
REVISION = "bc284db50d036958861cb60cdd7b77612052ce0d"
CONFIG = "2D"
SPLIT = "test"
LETTERS = "ABCDEF"


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def format_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt:
        return f"{str(prompt).strip()}\n\nAnswer with the option letter only."
    choices = list(row.get("choices") or [])
    return (
        f"{str(row.get('question', '')).strip()}\n\n"
        f"Options:\n{format_options(choices, LETTERS)}\n\n"
        "Answer with the option letter only."
    )


def item_from_row(row: dict[str, Any]) -> Item:
    return Item(
        item_id=str(row.get("idx", row.get("filename", ""))),
        prompt=format_prompt(row),
        answer=extract_gold_letter(row.get("answer"), LETTERS),
        meta={
            "type": row.get("type"),
            "task": row.get("task"),
            "source": row.get("source"),
            "image": row.get("image"),
            "choices": list(row.get("choices") or []),
            "filename": row.get("filename"),
        },
    )


def load_items(limit: int | None = None, seed: int = 0) -> list[Item]:
    rows = load_rows(
        DATASET_ID,
        config=CONFIG,
        split=SPLIT,
        revision=REVISION,
        limit=limit,
        seed=seed,
    )
    return [item_from_row(row) for row in rows]


def build_messages(item: Item) -> list[dict]:
    parts: list[dict] = []
    image = item.meta.get("image")
    if image is not None:
        parts.append(image_part(image))
    parts.append({"type": "text", "text": item.prompt})
    return user_message(parts)


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the extracted option letter plus its parse status."""

    return extract_choice_status(completion, LETTERS)


def score(item: Item, completion: str) -> bool:
    expected = extract_gold_letter(item.answer, LETTERS)
    if expected is None:
        return False
    predicted = extract_choice(completion, LETTERS)
    return predicted is not None and predicted == expected
