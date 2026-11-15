"""GPQA Diamond adapter.

Dataset: ``Idavidrein/gpqa`` config ``gpqa_diamond``, revision
``83022cefff930aea54f654c0b282e74b9eeda5c6`` (CC BY 4.0, gated).  It has 198
questions and four answer choices.

Scoring follows lm-evaluation-harness ``gpqa_diamond_cot_zeroshot``: a
generative answer with a flexible ``(A)``-``(D)`` / "The answer is" extractor
(reference 04, section 2 and trap 7).

The four options are shuffled deterministically from
``sha256(f"{seed}:{item_id}")``.  Never use Python's built-in ``hash()``: it is
randomised per process, which would break paired comparisons between variants
(reference 04, trap 8).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from . import (
    extract_choice,
    extract_choice_status,
    format_options,
    load_rows,
    user_message,
)

DATASET_ID = "Idavidrein/gpqa"
REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
CONFIG = "gpqa_diamond"
SPLIT = "train"
LETTERS = "ABCD"


def shuffle_map(item_id: str, seed: int) -> list[int]:
    """Return a deterministic permutation of the four option indices.

    ``random.Random`` is seeded from a SHA-256 digest so the permutation is
    identical across processes and Python invocations.
    """

    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest, 16))
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    return order


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def format_prompt(question: str, options: list[str]) -> str:
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{format_options(options, LETTERS)}\n\n"
        "Answer with the option letter (A-D) only."
    )


def _question_id(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"gpqa_diamond:{digest}"


def item_from_row(
    row: dict[str, Any],
    *,
    item_id: str | None = None,
    seed: int = 0,
) -> Item:
    question = str(row["Question"])
    question_id = str(item_id) if item_id is not None else _question_id(question)
    correct = str(row["Correct Answer"]).strip()
    original = [
        correct,
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    order = shuffle_map(question_id, seed)
    displayed = [original[index] for index in order]
    correct_position = order.index(0)
    answer = LETTERS[correct_position]

    return Item(
        item_id=question_id,
        prompt=format_prompt(question, displayed),
        answer=answer,
        meta={
            "options": displayed,
            "letter_map": order,
            "correct_answer": correct,
            "domain": row.get("domain"),
            "subdomain": row.get("subdomain"),
            "explanation": row.get("Explanation"),
            "seed": seed,
            "source": DATASET_ID,
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
    return [item_from_row(row, seed=seed) for row in rows]


def build_messages(item: Item) -> list[dict]:
    return user_message(item.prompt)


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the extracted option letter plus its parse status."""

    return extract_choice_status(completion, LETTERS)


def score(item: Item, completion: str) -> bool:
    if item.answer is None:
        return False
    predicted = extract_choice(completion, LETTERS)
    return predicted is not None and predicted == item.answer.upper()
