"""SpatialEval adapter (VQA primary, TQA control).

Dataset: ``MilaWang/SpatialEval`` (no declared licence; evaluation only, never
training, never redistribution), configs ``vqa`` and ``tqa``, split ``test``
with 4,635 items each.  VQA and TQA ask the same questions; only the modality
differs, so they share a ``base_item_id`` (reference 08, section (c) and
trap 19).

Scoring is exact match against ``oracle_option`` (A-D).  The paper/repo use
``evals/evaluation.py --mode vqa|tqa`` with the same prompt for both modalities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import (
    extract_choice,
    extract_choice_status,
    extract_gold_letter,
    image_part,
    load_rows,
    user_message,
)

DATASET_ID = "MilaWang/SpatialEval"
REVISION = "59ce0450bf65ae8dfa3590062442cf642474d7fe"
SPLIT = "test"
LETTERS = "ABCD"
MODALITIES = ("vqa", "tqa")

_MODALITY_RE = re.compile(r"\.(?:vqa|tqa|vtqa)\.")


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def base_item_id(item_id: str) -> str:
    """Remove the modality segment so VQA/TQA/VTQA items line up."""

    return _MODALITY_RE.sub(".", str(item_id), count=1)


def format_prompt(text: str) -> str:
    return f"{text.strip()}\n\nAnswer with the option letter (A-D) only."


def item_from_row(row: dict[str, Any], modality: str) -> Item:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")

    item_id = str(row["id"])
    answer = row.get("oracle_option")
    return Item(
        item_id=item_id,
        prompt=format_prompt(str(row.get("text", ""))),
        answer=extract_gold_letter(answer, LETTERS),
        meta={
            "modality": modality,
            "base_item_id": base_item_id(item_id),
            "image": row.get("image"),
            "oracle_answer": row.get("oracle_answer"),
            "oracle_full_answer": row.get("oracle_full_answer"),
        },
    )


def load_items(
    limit: int | None = None,
    seed: int = 0,
    modality: str = "vqa",
) -> list[Item]:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    rows = load_rows(
        DATASET_ID,
        config=modality,
        split=SPLIT,
        revision=REVISION,
        limit=limit,
        seed=seed,
    )
    return [item_from_row(row, modality) for row in rows]


def build_messages(item: Item) -> list[dict]:
    parts: list[dict] = []
    image = item.meta.get("image")
    if item.meta.get("modality") == "vqa" and image is not None:
        parts.append(image_part(image))
    parts.append({"type": "text", "text": item.prompt})
    return user_message(parts)


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the extracted option letter plus its parse status."""

    return extract_choice_status(completion, LETTERS)


def score(item: Item, completion: str) -> bool:
    if item.answer is None:
        return False
    predicted = extract_choice(completion, LETTERS)
    return predicted is not None and predicted == item.answer.upper()
