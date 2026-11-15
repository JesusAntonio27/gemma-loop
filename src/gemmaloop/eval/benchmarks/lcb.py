"""LiveCodeBench v6 adapter.

Dataset: ``livecodebench/code_generation_lite`` at the fixed ``release_v6``
revision ``0fe84c3912ea0c4d4a78037083943e8f0c4dd505`` (1055 problems, 2023-05
to 2025-04).  The post-cut subset used as the S2 set is **every** row with
``contest_date >= 2025-01-01`` (182 problems: 175 from test6 plus 7 from
test5).  The ``lcb_google`` alias loads a stable seeded sample of the FULL
release (``post_cut=None``) and is the only LCB bench carrying Google's
full-v6 figure; never compare the post-cut-only ``lcb`` accuracy with it.
Always pin ``release_v6``; ``release_latest`` is not a stable benchmark window
(reference 04, trap 2).

``datasets`` 5.x no longer executes the LiveCodeBench dataset script, so the
rows are downloaded directly from the dataset repository.  Scoring is delegated
to LiveCodeBench's official ``compute_code_generation_metrics.codegen_metrics``
through ``scripts/lcb_grade.py``, executed in the separate ``.venv-lcb``
environment created by ``scripts/setup_lcb.ps1``.  Only the sampled problems
are sent to the grader; ``custom_evaluator.py`` is not used because it expects
one entry per problem in the release and misattributes grades for subsets.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from . import stable_sample

DATASET_ID = "livecodebench/code_generation_lite"
RELEASE = "release_v6"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
POST_CUT_DATE = "2025-01-01"
DEFAULT_TIMEOUT = 12

# Official lcb_runner prompt constants (lcb_runner/prompts/code_generation.py).
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)

RELEASE_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
RELEASE_FILES["release_latest"] = RELEASE_FILES["release_v6"]

@dataclass
class Item:
    item_id: str
    prompt: str
    answer: str | None
    meta: dict = field(default_factory=dict)


def extract_code(completion: str) -> str:
    """Return the last fenced block in any language, like ``lcb_runner``.

    This mirrors ``lcb_runner.utils.extraction_utils.extract_code`` for the
    non-``GenericBase`` styles: fence lines are found by the substring
    ```````, and the content between the last two fence lines is returned.  A
    completion with fewer than two fence lines yields ``""``.
    """

    lines = (completion or "").split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def parse_answer(completion: str) -> tuple[str | None, str]:
    """Return the last fenced code block plus its parse status."""

    code = extract_code(completion)
    if not code:
        return None, "unparsed"
    return code, "ok"


def format_prompt(question_content: str, starter_code: str = "") -> str:
    """Official ``get_generic_question_template_answer`` prompt.

    The stdin/stdout instruction must stay in the prompt: without it the model
    may write a function-only answer that the official harness never runs.
    """

    prompt = f"### Question:\n{question_content.strip()}\n\n"
    if starter_code.strip():
        prompt += (
            f"### Format: {FORMATTING_WITH_STARTER_CODE}\n"
            f"```python\n{starter_code.strip()}\n```\n\n"
        )
    else:
        prompt += (
            f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _release_files(release: str) -> list[str]:
    try:
        return RELEASE_FILES[release]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LiveCodeBench release {release!r}; "
            f"expected one of {sorted(RELEASE_FILES)}"
        ) from exc


@functools.lru_cache(maxsize=None)
def _download_release(release: str) -> tuple[dict, ...]:
    rows: list[dict] = []
    for filename in _release_files(release):
        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=REVISION,
        )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return tuple(rows)


@functools.lru_cache(maxsize=None)
def _release_question_ids(release: str = RELEASE) -> tuple[str, ...]:
    return tuple(str(row["question_id"]) for row in _download_release(release))


def clear_release_cache() -> None:
    """Drop the cached release rows and id list.

    The full ``release_v6`` rows are several gigabytes.  They are only needed
    while :func:`load_items` turns them into the small, seeded :class:`Item`
    list; keeping them cached for the rest of the run would leave no room for
    the model.
    """

    for cached in (_download_release, _release_question_ids):
        clear = getattr(cached, "cache_clear", None)
        if clear is not None:
            clear()


def item_from_row(row: dict[str, Any]) -> Item:
    return Item(
        item_id=str(row["question_id"]),
        prompt=format_prompt(
            str(row.get("question_content", "")),
            str(row.get("starter_code", "") or ""),
        ),
        answer=None,
        meta={
            "question_title": row.get("question_title"),
            "platform": row.get("platform"),
            "difficulty": row.get("difficulty"),
            "contest_date": row.get("contest_date"),
            "post_cut": str(row.get("contest_date", "")) >= POST_CUT_DATE,
            "release": RELEASE,
        },
    )


def load_items(
    limit: int | None = None,
    seed: int = 0,
    release: str = RELEASE,
    *,
    post_cut: bool | None = None,
) -> list[Item]:
    """Load the requested LiveCodeBench rows.

    ``post_cut=True`` returns **every** post-cutoff problem (the registered S2
    set) and ignores ``limit``; ``post_cut=False`` returns a stable seeded
    sample of pre-cutoff problems; ``None`` samples the whole release, which is
    what the harness-only ``lcb_google`` bench uses.
    """

    try:
        rows = list(_download_release(release))
        if post_cut is True:
            rows = [
                row
                for row in rows
                if str(row.get("contest_date", "")) >= POST_CUT_DATE
            ]
        elif post_cut is False:
            rows = [
                row
                for row in rows
                if str(row.get("contest_date", "")) < POST_CUT_DATE
            ]
            if limit is not None:
                rows = stable_sample(rows, limit, seed)
        elif limit is not None:
            rows = stable_sample(rows, limit, seed)
        return [item_from_row(row) for row in rows]
    finally:
        # The selected items no longer reference the release rows.  Free the
        # multi-gigabyte cache immediately; a later call will re-download.
        clear_release_cache()


def build_messages(item: Item) -> list[dict]:
    """Official chat messages: generic system prompt + templated user prompt."""

    return [
        {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": item.prompt},
    ]


def _lcb_repo() -> Path:
    override = os.environ.get("LCB_REPO")
    return Path(override) if override else Path.home() / "ML" / "LiveCodeBench"


def _lcb_python(repo: Path) -> Path:
    override = os.environ.get("LCB_PYTHON")
    if override:
        return Path(override)
    windows = repo / ".venv-lcb" / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    return repo / ".venv-lcb" / "bin" / "python"


def _lcb_grade_script() -> Path:
    override = os.environ.get("LCB_GRADE_SCRIPT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "scripts" / "lcb_grade.py"


# LCB's release_v6 has at most 103 public+private tests per problem.  Use a
# conservative ceiling so the caller never has to load the 4+ GB release just
# to size the subprocess timeout.
MAX_TESTS_PER_PROBLEM = 120


def _grade_samples(
    outputs: list[dict], release: str, timeout: int
) -> tuple[dict[str, list[bool]], dict[str, list[dict]]]:
    provided: dict[str, list[str]] = {}
    for entry in outputs:
        question_id = str(entry["question_id"])
        code_list = list(entry.get("code_list") or [])
        if not code_list:
            raise ValueError(f"question_id {question_id!r} has no code samples")
        provided[question_id] = code_list

    if not provided:
        return {}, {}

    sample_counts = {len(code_list) for code_list in provided.values()}
    if len(sample_counts) != 1:
        raise ValueError(
            "every sampled problem needs the same number of code samples, "
            f"got {sorted(sample_counts)}"
        )

    # Existence checks come first: a missing .venv-lcb should fail immediately,
    # without downloading or parsing the release.
    repo = _lcb_repo()
    python = _lcb_python(repo)
    script = _lcb_grade_script()
    if not python.exists():
        raise FileNotFoundError(
            f"LiveCodeBench Python not found at {python}. "
            "Run scripts/setup_lcb.ps1 first."
        )
    if not script.exists():
        raise FileNotFoundError(f"LiveCodeBench grade runner not found at {script}")

    sample_count = next(iter(sample_counts))
    estimated_seconds = (
        len(provided)
        * sample_count
        * ((timeout + 1) * MAX_TESTS_PER_PROBLEM + 5)
    )
    outer_timeout = max(300, 60 + int(1.5 * estimated_seconds))

    with tempfile.TemporaryDirectory(prefix="gemmaloop_lcb_") as tmp:
        samples_path = Path(tmp) / "samples.json"
        grades_path = Path(tmp) / "grades.json"
        samples_path.write_text(
            json.dumps(provided, ensure_ascii=False), encoding="utf-8"
        )

        try:
            completed = subprocess.run(
                [
                    str(python),
                    str(script),
                    str(samples_path),
                    str(grades_path),
                    release,
                    str(4),
                    str(timeout),
                ],
                cwd=str(repo),
                check=False,
                capture_output=True,
                text=True,
                timeout=outer_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "LiveCodeBench grader exceeded its outer timeout of "
                f"{outer_timeout:.0f}s ({len(provided)} problem(s), "
                f"{sample_count} sample(s)/problem, per-test timeout="
                f"{timeout}s, conservative budget={estimated_seconds}s). "
                "A candidate may be blocked in a C call that ignores per-test "
                "async timeouts."
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "LiveCodeBench official grader failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        payload = json.loads(grades_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "grades" in payload:
        raw_grades = payload.get("grades") or {}
        raw_metadata = payload.get("metadata") or {}
    else:  # tolerate the old grades-only shape
        raw_grades = payload
        raw_metadata = {}

    grades = {
        str(question_id): [bool(value) for value in values]
        for question_id, values in raw_grades.items()
    }
    metadata = {
        str(question_id): list(values)
        for question_id, values in raw_metadata.items()
    }
    return grades, metadata


def run_official_evaluator(
    outputs: list[dict], release: str = RELEASE, timeout: int = DEFAULT_TIMEOUT
) -> dict[str, list[bool]]:
    """Grade sampled LiveCodeBench problems with the official metric code.

    ``outputs`` is a list of ``{"question_id": ..., "code_list": [...]}`` for
    the problems that were actually sampled.  Every problem must have the same
    number of samples, matching ``codegen_metrics``'s contract.  The mapping is
    written to disk and graded by ``scripts/lcb_grade.py`` inside ``.venv-lcb``;
    the returned mapping has the same question ids and preserves sample order.

    ``timeout`` is our declared per-test timeout in seconds (default 12).
    LCB's harness default is 6 s; reference 04 recommends 12 s to reduce false
    negatives from slow tests.  The subprocess gets a generous outer timeout
    derived from the problem/sample counts and a conservative per-problem test
    ceiling, so a candidate stuck in a C call that ignores the per-test async
    exception cannot hang the run forever.

    The former padded ``custom_evaluator`` workflow is intentionally gone: it
    requires one entry per problem in the release, and for a post-cut subset it
    zipped the full benchmark with the shorter graded list, assigning grades to
    the wrong question ids.
    """

    grades, _ = _grade_samples(outputs, release, timeout)
    return grades


def run_official_evaluator_with_metadata(
    outputs: list[dict], release: str = RELEASE, timeout: int = DEFAULT_TIMEOUT
) -> tuple[dict[str, list[bool]], dict[str, list[dict]]]:
    """Like :func:`run_official_evaluator` but also return LCB's sample metadata.

    The metadata keeps the official ``error_code`` (``-3`` is Time Limit
    Exceeded, ``-2`` Wrong Answer, etc.) and is useful for asserting that a
    timeout came from the per-test alarm rather than the global watchdog.
    """

    return _grade_samples(outputs, release, timeout)
