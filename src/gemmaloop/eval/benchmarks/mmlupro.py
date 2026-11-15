"""MMLU-Pro adapter (0-shot).

Dataset: ``TIGER-Lab/MMLU-Pro`` pinned to revision
``b189ec765aa7ed75c8acfea42df31fdae71f97be`` (MIT), ``test`` split with 12,032
questions and up to ten options.

Scoring uses the exact regex of the lm-evaluation-harness MMLU-Pro task:
``answer is \\(?([ABCDEFGHIJ])\\)?`` (reference 04, section 2).  Google does not
publish its shot count, so this adapter implements the declared 0-shot setting
(reference 04, trap 12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import extract_gold_letter, load_rows, strip_thinking, user_message

DATASET_ID = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SPLIT = "test"
NUM_FEWSHOT = 0
COT_CUE = "Finish your response with 'The answer is (X)' where X is the correct letter."

# Exact regex from lm_eval/tasks/mmlu_pro/_default_template_yaml.  No extra
# case-insensitivity or whitespace leniency beyond the official pattern.
_ANSWER_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def format_prompt(question: str, options: list[str], answer: str | None = None) -> str:
    del answer  # the answer never appears in the prompt
    choices = "\n".join(
        f"{letter}. {option.strip()}"
        for letter, option in zip("ABCDEFGHIJ", options)
    )
    return (
        f"Question:\n{question.strip()}\n"
        f"Options:\n{choices}\n"
        f"Answer: Let's think step by step. {COT_CUE}"
    )


def item_from_row(row: dict[str, Any]) -> Item:
    options = list(row.get("options") or [])
    answer = extract_gold_letter(row.get("answer"), "ABCDEFGHIJ")
    return Item(
        item_id=str(row["question_id"]),
        prompt=format_prompt(str(row["question"]), options),
        answer=answer,
        meta={
            "options": options,
            "category": row.get("category"),
            "src": row.get("src"),
            "num_fewshot": NUM_FEWSHOT,
        },
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
    """Return the MMLU-Pro letter plus its parse status (official regex)."""

    matches = _ANSWER_RE.findall(strip_thinking(completion))
    if not matches:
        return None, "unparsed"
    return matches[-1].upper(), "ok"


def score(item: Item, completion: str) -> bool:
    if item.answer is None:
        return False

    matches = _ANSWER_RE.findall(strip_thinking(completion))
    if not matches:
        return False
    return matches[-1].upper() == item.answer
