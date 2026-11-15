"""MATH-Vision adapter.

Dataset: ``MathLLMs/MathVision`` (MIT), ``test`` split with 3,040 problems
(MATH-V).

There are two scorers:

- :func:`score` is a **fast approximate, non-official** scorer kept for unit
  tests and quick iteration.  It ports the official answer cleanup and
  ``find_math_answer`` but uses a numeric fallback instead of the official
  ``latex2sympy2`` symbolic equivalence.
- :func:`run_official_scorer` runs the real ``MATH-V/evaluation/evaluate.py``
  in the eval-only ``.venv-mathv`` environment created by
  ``scripts/setup_mathv.ps1``.  Use it for reported numbers; no judge is used
  (reference 08, section (b) and trap 15).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import (
    format_options,
    image_part,
    last_boxed,
    load_rows,
    strip_thinking,
    user_message,
)

DATASET_ID = "MathLLMs/MathVision"
REVISION = "2837ddb3f13abaf6b3997c12d80753e5470bd46a"
SPLIT = "test"
LETTERS = "ABCDEFGHIJ"
BOXED_INSTRUCTION = (
    'Please solve the problem and put your answer in one "\\boxed{}". '
    'If it is a multiple choice question, only one letter is allowed in '
    'the "\\boxed{}".'
)


@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Official MATH-V normalisation helpers (ported from evaluation/utils.py)
# ---------------------------------------------------------------------------
def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                if len(substr) > 2:
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}{" + b + "}"
            else:
                if len(substr) > 2:
                    new_str += "{" + a + "}" + b + substr[2:]
                else:
                    new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        int_a, int_b = int(a), int(b)
        assert string == f"{int_a}/{int_b}"
        return f"\\frac{{{int_a}}}{{{int_b}}}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    return string.split("\\text{ ")[0]


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if "sqrt" in string:
        string = _fix_sqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def find_math_answer(text: str) -> str:
    text = (text or "").lower()
    if "{}" in text:
        text = text.replace("{}", "")
    answer = last_boxed(text)
    if answer is None:
        answer = text

    if answer.find("}") != -1 and (
        answer.find("{") == -1 or answer.find("}") < answer.find("{")
    ):
        answer = answer.split("}")[0]

    wrapped = re.fullmatch(r"\\(?:text|mbox)\{([^{}]*)\}", answer)
    if wrapped:
        answer = wrapped.group(1)

    answer = answer.split("=")[-1]
    answer = answer.split("\\approx")[-1]
    answer = answer.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    answer = answer.replace("+\\infty", "\\infty").replace("\\\\", "\\").replace("\n", "")
    answer = answer.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    answer = answer.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    answer = answer.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    answer = (
        answer.replace("{units}", "")
        .replace("units", "")
        .replace("{km}", "")
        .replace("km", "")
    )
    return _strip_string(answer)


def eval_tuple(text: str) -> str:
    """Evaluate a simple numeric tuple/list the way the official helper does."""

    if len(text) < 2 or text[0] not in "([" or text[-1] not in ")]":
        return text
    inner = text[1:-1]
    parts = inner.split(",")
    if len(parts) <= 1:
        return text
    evaluated = []
    for part in parts:
        try:
            evaluated.append(str(round(float(part), 2)))
        except ValueError:
            evaluated.append(part)
    return ("(" if text[0] == "(" else "[") + ",".join(evaluated) + (")" if text[-1] == ")" else "]")


def _frac_to_float(text: str) -> float | None:
    match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if match:
        try:
            return float(match.group(1)) / float(match.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _as_number(text: str) -> float | None:
    text = text.strip().strip("$").replace("\\,", "").replace(",", "").replace(" ", "")
    if text.startswith("+") or text.startswith("-"):
        sign = -1.0 if text.startswith("-") else 1.0
        unsigned = text[1:]
    else:
        sign = 1.0
        unsigned = text
    frac = _frac_to_float(unsigned)
    if frac is not None:
        return sign * frac
    if re.fullmatch(r"[+-]?\d+", text):
        return float(text)
    try:
        return float(text)
    except ValueError:
        return None


def is_equal(answer: str, ground_truth: str) -> bool:
    answer = str(answer).lower()
    ground_truth = str(ground_truth).lower()

    if answer.replace(" ", "") == "" or ground_truth.replace(" ", "") == "":
        return False
    if ground_truth.strip() == answer.strip():
        return True

    answer = eval_tuple(answer)
    ground_truth = eval_tuple(ground_truth)
    if ground_truth == answer:
        return True

    try:  # optional, exactly like the official grader when the package exists
        from latex2sympy2 import latex2sympy

        if round(eval(str(latex2sympy(ground_truth))), 2) == round(
            eval(str(latex2sympy(answer))), 2
        ):
            return True
    except Exception:
        pass

    answer_number = _as_number(answer)
    ground_truth_number = _as_number(ground_truth)
    if (
        answer_number is not None
        and ground_truth_number is not None
        and round(answer_number, 6) == round(ground_truth_number, 6)
    ):
        return True

    return _strip_string(answer) == _strip_string(ground_truth)


# ---------------------------------------------------------------------------
# Adapter surface
# ---------------------------------------------------------------------------
def format_prompt(question: str, options: list[str], answer: str | None = None) -> str:
    r"""Official MATH-V prompt (``models/GPT4V.py``).

    The instruction to put the answer in one ``\boxed{}`` is mandatory: the
    official scorer extracts the last box, so a response without one is graded
    as wrong even when the value is correct.
    """

    del answer
    parts = [BOXED_INSTRUCTION, question.strip()]
    if options and "".join(str(option) for option in options) != "ABCDE":
        parts.append(format_options(options, LETTERS))
    return "\n".join(parts)


def item_from_row(row: dict[str, Any]) -> Item:
    options = list(row.get("options") or [])
    answer = row.get("answer")
    image = row.get("decoded_image")
    if image is None:
        image = row.get("image")
    return Item(
        item_id=str(row["id"]),
        prompt=format_prompt(str(row.get("question", "")), options),
        answer=None if answer is None else str(answer),
        meta={
            "options": options,
            "image": image,
            "subject": row.get("subject"),
            "level": row.get("level"),
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
    parts: list[dict] = []
    image = item.meta.get("image")
    if image is not None:
        parts.append(image_part(image))
    parts.append({"type": "text", "text": item.prompt})
    return user_message(parts)


def _extract_model_answer(completion: str) -> str:
    """Port of the answer cleanup in ``MATH-V/evaluation/evaluate.py``."""

    model_answer = strip_thinking(completion).strip()
    for letter in "ABCDE":
        if (
            model_answer.endswith(f" {letter}.")
            or model_answer.endswith(f" ({letter}).")
            or model_answer.startswith(f"{letter}\n")
            or model_answer.startswith(f"({letter})\n")
            or model_answer.startswith(f"({letter}) {letter}\n")
        ):
            model_answer = letter

    tail = model_answer.split("is ")[-1].rstrip(".")
    if is_number(tail):
        model_answer = tail

    if "oxed{" not in model_answer:
        for flag in (
            "the final answer is",
            "the answer is",
            "the correct answer is",
            "the answer should be",
        ):
            for variant in (flag, flag.replace("the", "The")):
                raw = model_answer
                model_answer = model_answer.split(variant)[-1].strip()
                if variant in raw:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
    elif model_answer.count("oxed{") > 1:
        model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

    model_answer = find_math_answer(model_answer)
    for old, new in (
        ("(a)", "a"),
        ("(b)", "b"),
        ("(c)", "c"),
        ("(d)", "d"),
        ("(e)", "e"),
        ("{a}", "a"),
        ("{b}", "b"),
        ("{c}", "c"),
        ("{d}", "d"),
        ("{e}", "e"),
    ):
        model_answer = model_answer.replace(old, new)
    return model_answer.rstrip(".").lstrip(":").strip()


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the official-style extracted answer plus its parse status."""

    if not strip_thinking(completion).strip():
        return None, "unparsed"
    return _extract_model_answer(completion), "ok"


def score(item: Item, completion: str) -> bool:
    """Fast approximate, **non-official** MATH-Vision scorer.

    Use :func:`run_official_scorer` for the official exact-match result.
    """

    if item.answer is None:
        return False

    ground_truth = str(item.answer)
    options = item.meta.get("options") or []
    if options and len(ground_truth) == 1 and ground_truth.upper() in LETTERS:
        index = ord(ground_truth.upper()) - ord("A")
        ground_truth_value = options[index] if index < len(options) else ""
    else:
        ground_truth_value = ""

    predicted = _extract_model_answer(completion)
    return is_equal(ground_truth, predicted) or is_equal(ground_truth_value, predicted)


# ---------------------------------------------------------------------------
# Official MATH-V scorer (separate eval-only environment)
# ---------------------------------------------------------------------------
def _mathv_repo() -> Path:
    override = os.environ.get("MATHV_REPO")
    return Path(override) if override else Path.home() / "ML" / "MATH-V"


def _mathv_python(repo: Path) -> Path:
    override = os.environ.get("MATHV_PYTHON")
    if override:
        return Path(override)
    windows = repo / ".venv-mathv" / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    return repo / ".venv-mathv" / "bin" / "python"


def _mathv_grade_script() -> Path:
    override = os.environ.get("MATHV_GRADE_SCRIPT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "scripts" / "mathv_grade.py"


def run_official_scorer(outputs: list[dict]) -> dict[str, bool]:
    """Run MATH-V's official ``evaluation/evaluate.py`` on our completions.

    ``outputs`` is a list of dictionaries with ``id`` (or ``item_id``),
    ``completion`` (or ``response``), ``answer`` and ``options``.  ``answer``
    and ``options`` are required because the official script looks each id up
    in ``data/test.jsonl``; passing them lets the same code grade ad-hoc
    fixtures without modifying the cloned repo.

    Each completion has its ``<|channel>thought ... <channel|>`` block removed
    first (the official script only strips whitespace), so a boxed distractor in
    the reasoning trace cannot be graded.  The runner script executes inside
    ``C:\\Users\\<user>\\ML\\MATH-V`` with the eval-only ``.venv-mathv``
    environment (``latex2sympy2`` included) and returns ``{id: correct}``.
    """

    repo = _mathv_repo()
    python = _mathv_python(repo)
    script = _mathv_grade_script()
    if not python.exists():
        raise FileNotFoundError(
            f"MATH-V Python not found at {python}. Run scripts/setup_mathv.ps1 first."
        )
    if not script.exists():
        raise FileNotFoundError(f"MATH-V grade runner not found at {script}")

    records: list[dict] = []
    for entry in outputs:
        item_id = entry.get("item_id", entry.get("id"))
        if item_id is None:
            raise ValueError("Every output needs an 'id' (or 'item_id').")
        response = entry.get("completion", entry.get("response", ""))
        if "answer" not in entry:
            raise ValueError(f"Output {item_id!r} needs the gold 'answer'.")
        records.append(
            {
                "id": str(item_id),
                # The official evaluate.py only strips whitespace, so remove
                # Gemma's reasoning trace here before it can leak a boxed
                # distractor into the exact-match.
                "response": strip_thinking(str(response)),
                "answer": str(entry["answer"]),
                "options": list(entry.get("options") or []),
                "subject": entry.get("subject", ""),
                "level": entry.get("level", 0),
            }
        )

    with tempfile.TemporaryDirectory(prefix="gemmaloop_mathv_") as tmp:
        input_path = Path(tmp) / "outputs.json"
        output_path = Path(tmp) / "grades.json"
        input_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(
            [str(python), str(script), str(input_path), str(output_path), str(repo)],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "MATH-V official evaluator failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        grades = json.loads(output_path.read_text(encoding="utf-8"))
    return {str(key): bool(value) for key, value in grades.items()}
