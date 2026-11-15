"""MMMU-Pro adapter.

Dataset: ``MMMU/MMMU_Pro`` (Apache 2.0), ``test`` split, 1,730 questions per
config.  Google only publishes "MMMU Pro", without saying whether it means
Standard(10), Standard(4), Vision or the Overall average, so the adapter
measures **Standard(10)** and **Vision** separately (reference 08, section (b)
and trap 13).  ``Overall = (Standard(10) + Vision) / 2`` is defined by the
MMMU-Pro paper, not by the Gemma model card.

Answers are letters A-J.  Prompt construction follows lmms-eval's
``mmmu_pro_standard`` / ``mmmu_pro_vision`` tasks: Standard uses the official
question + ``A. option`` list + post-prompt, and each image part is emitted at
its ``<image i>`` placeholder position so ``<image 3>`` always refers to
``image_3``.  Vision uses the official vision-only post-prompt, because the
question and options are rendered inside the image (reference 08, trap 14).
"""

from __future__ import annotations

import ast
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

DATASET_ID = "MMMU/MMMU_Pro"
REVISION = "0d7426df4ccb3d8704a992fe6850abee7c262127"
STANDARD_CONFIG = "standard (10 options)"
VISION_CONFIG = "vision"
SPLIT = "test"
LETTERS = "ABCDEFGHIJ"
IMAGE_FIELDS = tuple(f"image_{index}" for index in range(1, 8))
_IMAGE_REF_RE = re.compile(r"<image[ _](\d+)\s*>", re.IGNORECASE)

# Exact default post-prompts from lmms_eval/tasks/mmmu_pro/mmmu_pro_*.yaml.
STANDARD_POST_PROMPT = "Answer with the option letter from the given choices directly."
VISION_POST_PROMPT = (
    "Answer with the option letter from the given choices directly. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
)


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def parse_options(raw: Any) -> list[str]:
    """Normalise the dataset's options field (a list or its repr) to a list."""

    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(option).strip() for option in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(option).strip() for option in parsed]
    return [text]


def _format_options(options: list[str]) -> str:
    return "\n".join(
        f"{letter}. {option.strip()}"
        for letter, option in zip(LETTERS, options)
    )


def format_prompt(question: str, options: list[str], *, vision: bool = False) -> str:
    if vision:
        return VISION_POST_PROMPT
    return (
        f"{question.strip()}\n"
        f"{_format_options(options)}\n\n"
        f"{STANDARD_POST_PROMPT}"
    )


def item_from_row(row: dict[str, Any], config: str = STANDARD_CONFIG) -> Item:
    options = parse_options(row.get("options"))
    answer_letter = extract_gold_letter(row.get("answer"), LETTERS)

    images: list[tuple[str, Any]] = []
    if config == VISION_CONFIG:
        image = row.get("image")
        if image is not None:
            images.append(("image", image))
        prompt = format_prompt("", [], vision=True)
    else:
        question = str(row.get("question", ""))
        # lmms-eval's doc_to_visual includes exactly the images referenced by
        # the prompt, in ascending image-index order.
        referenced = sorted({int(index) for index in _IMAGE_REF_RE.findall(question)})
        names = [
            f"image_{index}"
            for index in referenced
            if row.get(f"image_{index}") is not None
        ]
        for name in names:
            images.append((name, row[name]))
        prompt = format_prompt(question, options)

    return Item(
        item_id=str(row["id"]),
        prompt=prompt,
        answer=answer_letter,
        meta={
            "config": config,
            "options": options,
            "images": images,
            "image_names": [name for name, _ in images],
            "subject": row.get("subject"),
            "img_type": row.get("img_type"),
        },
    )


def load_items(
    limit: int | None = None,
    seed: int = 0,
    config: str = STANDARD_CONFIG,
) -> list[Item]:
    rows = load_rows(
        DATASET_ID,
        config=config,
        split=SPLIT,
        revision=REVISION,
        limit=limit,
        seed=seed,
    )
    return [item_from_row(row, config=config) for row in rows]


def build_messages(item: Item) -> list[dict]:
    images = item.meta.get("images", [])
    if item.meta.get("config") == VISION_CONFIG:
        parts: list[dict] = [image_part(image) for _, image in images]
        parts.append({"type": "text", "text": item.prompt})
        return user_message(parts)

    images_by_name = dict(images)
    parts = []
    cursor = 0
    for match in _IMAGE_REF_RE.finditer(item.prompt):
        text = item.prompt[cursor:match.start()]
        if text:
            parts.append({"type": "text", "text": text})
        name = f"image_{int(match.group(1))}"
        if name in images_by_name:
            parts.append(image_part(images_by_name[name]))
        else:
            parts.append({"type": "text", "text": match.group(0)})
        cursor = match.end()
    tail = item.prompt[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return user_message(parts)


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the extracted option letter plus its parse status."""

    return extract_choice_status(completion, LETTERS)


def score(item: Item, completion: str) -> bool:
    if item.answer is None:
        return False
    predicted = extract_choice(completion, LETTERS)
    return predicted is not None and predicted == item.answer.upper()
