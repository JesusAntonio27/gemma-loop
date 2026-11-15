"""Pinned Stage 0 problem sources and the normalised problem schema.

The pre-registration names the sources but not their Hub revisions.  The
revisions below are the immutable Hub commits read when this module was
written; every row and every manifest therefore names an exact snapshot:

===================  ================================================  ==========================================
source               Hub dataset                                       revision
===================  ================================================  ==========================================
``taco``             ``BAAI/TACO``                                     ``d593ed0a2becbbc952230bb89be09189bf1056dc``
``apps``             ``codeparrot/apps``                               ``0f10e424e13e1c2a69f851e153097b71b6734a1f``
``codecontests``     ``deepmind/code_contests``                        ``802411c3010cb00d1b05bad57ca77365a3c699d6``
``omr``              ``nvidia/OpenMathReasoning``                      ``d3d08664755704f422af97d43a7ff0ded4bd95df``
``dapo``             ``BytedTsinghua-SIA/DAPO-Math-17k``               ``65877096c24ffa7abc4e4fa5edb95cf3413a5674``
``deepscaler``       ``agentica-org/DeepScaleR-Preview-Dataset``       ``b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29``
``chat``             ``HuggingFaceH4/ultrachat_200k``                  ``8049631c405ae6576f93f445c6b8166f76f5505a``
===================  ================================================  ==========================================

Loader corrections (the first pool build failed on these):

* ``apps``: the dataset script is no longer supported, so the loader reads
  ``all/train/0000.parquet`` from the pinned ``refs/convert/parquet`` commit
  ``0f10e424e13e1c2a69f851e153097b71b6734a1f`` (5000 rows, the same ``all``
  train split as the original ``train.jsonl`` at source commit
  ``21e74ddf8de1a21436da12e3e653065c5213e9d1``).
* ``taco``: the dataset script is no longer supported, but the pinned repo
  already contains ``ALL/train-*.parquet`` (9 shards, 25443 rows, identical to
  the pinned ``ALL`` train arrow shards).  The loader reads those files at the
  pinned commit; no mirror or different dataset is involved.
* ``codecontests``: the parquet schema stores ``solutions`` as a struct of two
  parallel lists (``language`` enum ints + ``solution`` strings), not a dict
  keyed by language name.  The loader maps the enum (1=PYTHON, 2=CPP,
  3=PYTHON3, 4=JAVA) and requires a Python reference solution, preferring
  PYTHON3.  The dataset has no per-row date field; the pinned snapshot predates
  2023-05, which is what the manifest records.

``DeepScaleR`` is the registered *DeepScaleR* pool source (the preview
dataset).  ``dapo`` is the original ByteDance/Tsinghua DAPO-Math-17k release.
CodeContests is already a pre-2023-05 snapshot; the filter is recorded in the
manifest as ``pre_2023_05`` so a later rebuild cannot silently widen the window.

Rows are converted to a single schema::

    {
      "problem_id": "<source>:<stable-id>",
      "source": "taco",
      "kind": "code" | "math" | "chat",
      "prompt": "...",                 # chat prompt sent to O
      "tests": [...],                  # code only, hidden tests
      "answer": "...",                 # math only, exact answer
      "metadata": {...},               # provenance and source fields
      "split": "...",                  # filled by the pool builder
      "bucket": 0..99,                 # filled by the pool builder
    }

No network access happens at import time.  Tests inject ``loader`` callables.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Literal

from gemmaloop.data.decontam import normalize

Kind = Literal["code", "math", "chat"]
Loader = Callable[[], Iterable[dict[str, Any]]]

TACO_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
# Source repo commit of the original (script-based) APPS dataset...
APPS_SOURCE_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
# ...and the full sha of its ``refs/convert/parquet`` branch, which is what we
# actually read now that dataset scripts are unsupported.
APPS_PARQUET_REVISION = "0f10e424e13e1c2a69f851e153097b71b6734a1f"
APPS_REVISION = APPS_PARQUET_REVISION
CODECONTESTS_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
OMR_REVISION = "d3d08664755704f422af97d43a7ff0ded4bd95df"
DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DEEPSCALER_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
CHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"

MAX_HIDDEN_TESTS = 40

# CodeContests language enum, from the pinned deepmind/code_contests
# ``contest_problem.proto``.  Only these four values occur in the parquet.
CODECONTESTS_LANGUAGE_NAMES = {
    0: "UNKNOWN",
    1: "PYTHON",
    2: "CPP",
    3: "PYTHON3",
    4: "JAVA",
}
CODECONTESTS_PYTHON_LANGUAGES = ("PYTHON3", "PYTHON")

# Bounded retries for Hub downloads.  Non-network errors are never retried and
# a persistent network failure is re-raised after the last attempt.
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

# One source of truth for the prereg-v1-data addendum.  The builder reads these
# constants and every manifest records them.
DATASET_REVISIONS: dict[str, dict[str, Any]] = {
    "taco": {
        "hf_id": "BAAI/TACO",
        "config": "ALL",
        "split": "train",
        "revision": TACO_REVISION,
        "license": "Apache 2.0",
        "load_mode": "data_files",
        "data_files": ["ALL/train-*.parquet"],
        "source_revision": TACO_REVISION,
        "rows": 25443,
        "notes": "pinned ALL/train parquet shards (same dataset, no mirror)",
    },
    "apps": {
        "hf_id": "codeparrot/apps",
        "config": "all",
        "split": "train",
        "revision": APPS_REVISION,
        "license": "MIT",
        "load_mode": "refs/convert/parquet",
        "data_files": ["all/train/0000.parquet"],
        "source_revision": APPS_SOURCE_REVISION,
        "parquet_revision": APPS_PARQUET_REVISION,
        "rows": 5000,
        "notes": "pinned auto-converted parquet branch (script dataset unsupported)",
    },
    "codecontests": {
        "hf_id": "deepmind/code_contests",
        "config": None,
        "split": "train",
        "revision": CODECONTESTS_REVISION,
        "license": "CC BY 4.0",
        "date_window": "pre_2023_05",
        "load_mode": "parquet",
        "language_filter": "PYTHON3 then PYTHON",
        "notes": "solutions struct: parallel language enum + solution string lists",
    },
    "omr": {
        "hf_id": "nvidia/OpenMathReasoning",
        "config": "default",
        "split": "cot",
        "revision": OMR_REVISION,
        "license": "CC BY 4.0",
    },
    "dapo": {
        "hf_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "config": None,
        "split": "train",
        "revision": DAPO_REVISION,
        "license": "Apache 2.0",
    },
    "deepscaler": {
        "hf_id": "agentica-org/DeepScaleR-Preview-Dataset",
        "config": None,
        "split": "train",
        "revision": DEEPSCALER_REVISION,
        "license": "MIT",
    },
    "chat": {
        "hf_id": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "train_sft",
        "revision": CHAT_REVISION,
        "license": "MIT",
    },
}


CODE_PROMPT_SUFFIX = (
    "\n\nRead the inputs from stdin, solve the problem, and write the answer "
    "to stdout. Enclose the final Python program in a ```python code block."
)
MATH_PROMPT_SUFFIX = "\n\nPut the final answer alone inside \\boxed{}."


@dataclass(frozen=True)
class PoolSource:
    """One pinned Stage 0 source.

    ``quota`` is the number of problems the pool asks from this source.  The
    registered pool sizes are 2,000 code, 2,000 math and 1,000 chat; the quotas
    below divide each family evenly across its named sources.
    """

    name: str
    kind: Kind
    hf_id: str
    config: str | None
    split: str
    revision: str
    license: str
    quota: int
    columns: tuple[str, ...] | None = None
    data_files: tuple[str, ...] = ()
    load_mode: str = "dataset"
    loader: Loader | None = field(default=None, compare=False, repr=False)
    notes: str = ""

    def load(self) -> Iterable[dict[str, Any]]:
        if self.loader is not None:
            return self.loader()
        from datasets import load_dataset

        if self.data_files:
            files = [
                f"hf://datasets/{self.hf_id}@{self.revision}/{pattern}"
                for pattern in self.data_files
            ]
            return load_dataset(
                "parquet",
                data_files={self.split: files},
                split=self.split,
                streaming=True,
            )

        kwargs: dict[str, Any] = {}
        if self.config is not None:
            kwargs["name"] = self.config
        if self.columns:
            try:
                return load_dataset(
                    self.hf_id,
                    split=self.split,
                    streaming=True,
                    columns=list(self.columns),
                    revision=self.revision,
                    **kwargs,
                )
            except (TypeError, ValueError):
                pass
        return load_dataset(
            self.hf_id,
            split=self.split,
            streaming=True,
            revision=self.revision,
            **kwargs,
        )


POOL_SOURCES: tuple[PoolSource, ...] = (
    PoolSource(
        name="taco",
        kind="code",
        hf_id="BAAI/TACO",
        config="ALL",
        split="train",
        revision=TACO_REVISION,
        license="Apache 2.0 (see dataset card)",
        quota=667,
        columns=("question", "solutions", "input_output", "starter_code", "difficulty", "url"),
        data_files=("ALL/train-*.parquet",),
        load_mode="data_files",
        notes="pinned ALL/train parquet shards (script dataset unsupported)",
    ),
    PoolSource(
        name="apps",
        kind="code",
        hf_id="codeparrot/apps",
        config="all",
        split="train",
        revision=APPS_REVISION,
        license="MIT (see dataset card)",
        quota=667,
        columns=("problem_id", "question", "solutions", "input_output", "starter_code"),
        data_files=("all/train/0000.parquet",),
        load_mode="refs/convert/parquet",
        notes="pinned refs/convert/parquet branch (script dataset unsupported)",
    ),
    PoolSource(
        name="codecontests",
        kind="code",
        hf_id="deepmind/code_contests",
        config=None,
        split="train",
        revision=CODECONTESTS_REVISION,
        license="CC BY 4.0 (see dataset card)",
        quota=666,
        columns=(
            "name",
            "description",
            "public_tests",
            "private_tests",
            "generated_tests",
            "solutions",
            "source",
            "difficulty",
            "cf_contest_id",
            "cf_rating",
        ),
        notes="pre_2023_05 snapshot",
    ),
    PoolSource(
        name="omr",
        kind="math",
        hf_id="nvidia/OpenMathReasoning",
        config="default",
        split="cot",
        revision=OMR_REVISION,
        license="CC BY 4.0",
        quota=667,
        columns=("problem", "generated_solution", "expected_answer", "problem_source"),
    ),
    PoolSource(
        name="dapo",
        kind="math",
        hf_id="BytedTsinghua-SIA/DAPO-Math-17k",
        config=None,
        split="train",
        revision=DAPO_REVISION,
        license="Apache 2.0 (see dataset card)",
        quota=667,
        columns=("data_source", "prompt", "reward_model", "extra_info"),
    ),
    PoolSource(
        name="deepscaler",
        kind="math",
        hf_id="agentica-org/DeepScaleR-Preview-Dataset",
        config=None,
        split="train",
        revision=DEEPSCALER_REVISION,
        license="MIT (see dataset card)",
        quota=666,
        columns=("problem", "solution", "answer"),
    ),
    PoolSource(
        name="chat",
        kind="chat",
        hf_id="HuggingFaceH4/ultrachat_200k",
        config="default",
        split="train_sft",
        revision=CHAT_REVISION,
        license="MIT",
        quota=1000,
        columns=("messages",),
        notes="existing decontaminated chat source (data-length-audit.md)",
    ),
)

SOURCES_BY_NAME = {source.name: source for source in POOL_SOURCES}


def source_with_loader(name: str, loader: Loader, **changes: Any) -> PoolSource:
    """Return a copy of a registered source with a test-only loader."""

    base = SOURCES_BY_NAME[name]
    return replace(base, loader=loader, **changes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_WS_RE = re.compile(r"\s+")


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8"))
    return digest.hexdigest()[:24]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _last_boxed(text: str) -> str | None:
    text = text or ""
    matches = list(_BOXED_RE.finditer(text))
    for match in reversed(matches):
        index = match.end() - 1
        depth = 0
        for position in range(index, len(text)):
            char = text[position]
            if char == "{" and (position == 0 or text[position - 1] != "\\"):
                depth += 1
            elif char == "}" and (position == 0 or text[position - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    return text[index + 1 : position].strip()
    return None


def _json_load(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def parse_input_output(value: Any, *, fn_name: str | None = None) -> list[dict[str, Any]]:
    """Convert TACO/APPS ``input_output`` JSON to the normalised test list."""

    payload = _json_load(value)
    if not isinstance(payload, dict):
        return []
    inputs = _as_list(payload.get("inputs", payload.get("input")))
    outputs = _as_list(payload.get("outputs", payload.get("output")))
    if not inputs or not outputs:
        return []
    fn = payload.get("fn_name") or fn_name
    tests: list[dict[str, Any]] = []
    for test_input, test_output in zip(inputs, outputs):
        if fn:
            tests.append(
                {
                    "input": test_input if isinstance(test_input, str) else json.dumps(test_input),
                    "output": test_output if isinstance(test_output, str) else json.dumps(test_output),
                    "type": "functional",
                    "fn_name": str(fn),
                }
            )
        else:
            tests.append(
                {
                    "input": test_input if isinstance(test_input, str) else json.dumps(test_input),
                    "output": test_output if isinstance(test_output, str) else json.dumps(test_output),
                    "type": "stdin",
                }
            )
        if len(tests) >= MAX_HIDDEN_TESTS:
            break
    return tests


def _codecontests_tests(row: dict[str, Any]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for key in ("public_tests", "private_tests", "generated_tests"):
        block = row.get(key) or {}
        if not isinstance(block, dict):
            continue
        for test_input, test_output in zip(
            _as_list(block.get("input")), _as_list(block.get("output"))
        ):
            tests.append({"input": str(test_input), "output": str(test_output), "type": "stdin"})
            if len(tests) >= MAX_HIDDEN_TESTS:
                return tests
    return tests


def _codecontests_language_name(value: Any) -> str:
    """Map a parquet language enum int (or legacy name) to its proto name."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return CODECONTESTS_LANGUAGE_NAMES.get(value, str(value))
    if isinstance(value, float) and value.is_integer():
        return CODECONTESTS_LANGUAGE_NAMES.get(int(value), str(int(value)))
    text = _text(value)
    if text.isdigit():
        return CODECONTESTS_LANGUAGE_NAMES.get(int(text), text)
    return text.upper()


def _codecontests_solutions(solutions: Any) -> dict[str, list[Any]]:
    """Return ``{language_name: [solutions]}`` from the parquet struct.

    The pinned parquet stores ``solutions`` as ``struct<language: list<int>,
    solution: list<string>>`` (two parallel lists).  Older dataset-script
    exports were already grouped by language name; both shapes are accepted.
    """

    if not isinstance(solutions, dict):
        return {}
    languages = _as_list(solutions.get("language"))
    entries = _as_list(solutions.get("solution"))
    grouped: dict[str, list[Any]] = {}
    if languages and entries and len(languages) == len(entries):
        for language, entry in zip(languages, entries):
            grouped.setdefault(_codecontests_language_name(language), []).append(entry)
        return grouped
    for language, value in solutions.items():
        if language in {"language", "solution"}:
            continue
        grouped.setdefault(_codecontests_language_name(language), []).extend(
            _as_list(value)
        )
    return grouped


def _reference_solution(row: dict[str, Any], source: str) -> str:
    solutions = row.get("solutions")
    if source == "codecontests":
        grouped = _codecontests_solutions(solutions)
        for language in CODECONTESTS_PYTHON_LANGUAGES:
            for entry in grouped.get(language, []):
                if _text(entry):
                    return str(entry)
        return ""
    # TACO and APPS store ``solutions`` as a JSON-encoded list in parquet.
    payload = _json_load(solutions)
    for entry in _as_list(payload if payload is not None else solutions):
        if _text(entry):
            return str(entry)
    return ""


def _first_user_message(row: dict[str, Any]) -> str:
    for key in ("messages", "conversations", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, list):
            continue
        for message in value:
            if not isinstance(message, dict):
                continue
            role = _text(message.get("role", message.get("from", ""))).lower()
            if role in {"user", "human"}:
                content = message.get("content", message.get("value", ""))
                if _text(content):
                    return _text(content)
    return ""


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------
def convert_row(source: str, row: dict[str, Any], index: int = 0) -> dict[str, Any] | None:
    """Return a normalised problem, or ``None`` when the row is unusable."""

    if source == "taco":
        question = _text(row.get("question") or row.get("problem"))
        tests = parse_input_output(row.get("input_output"))
        solution = _reference_solution(row, source)
        if not question or not tests or not solution:
            return None
        raw_id = row.get("url") or _stable_id(question)
        return {
            "problem_id": f"{source}:{_stable_id(raw_id)}",
            "source": source,
            "kind": "code",
            "prompt": question + CODE_PROMPT_SUFFIX,
            "tests": tests,
            "answer": None,
            "metadata": {
                "raw_id": raw_id,
                "difficulty": row.get("difficulty"),
                "url": row.get("url"),
                "starter_code": row.get("starter_code") or "",
                "reference_solution": solution,
                "source_index": index,
                "hf_revision": TACO_REVISION,
            },
        }

    if source == "apps":
        question = _text(row.get("question") or row.get("problem"))
        tests = parse_input_output(row.get("input_output"))
        solution = _reference_solution(row, source)
        if not question or not tests or not solution:
            return None
        raw_id = row.get("problem_id")
        if raw_id is None:
            raw_id = _stable_id(question)
        return {
            "problem_id": f"{source}:{_stable_id(raw_id)}",
            "source": source,
            "kind": "code",
            "prompt": question + CODE_PROMPT_SUFFIX,
            "tests": tests,
            "answer": None,
            "metadata": {
                "raw_id": raw_id,
                "difficulty": row.get("difficulty"),
                "starter_code": row.get("starter_code") or "",
                "reference_solution": solution,
                "source_index": index,
                "hf_revision": APPS_REVISION,
            },
        }

    if source == "codecontests":
        description = _text(row.get("description") or row.get("name"))
        tests = _codecontests_tests(row)
        solution = _reference_solution(row, source)
        if not description or not tests or not solution:
            return None
        raw_id = row.get("name") or row.get("cf_contest_id") or _stable_id(description)
        return {
            "problem_id": f"{source}:{_stable_id(raw_id)}",
            "source": source,
            "kind": "code",
            "prompt": description + CODE_PROMPT_SUFFIX,
            "tests": tests,
            "answer": None,
            "metadata": {
                "raw_id": raw_id,
                "difficulty": row.get("difficulty"),
                "cf_contest_id": row.get("cf_contest_id"),
                "cf_rating": row.get("cf_rating"),
                "reference_solution": solution,
                "source_index": index,
                "date_window": "pre_2023_05",
                "hf_revision": CODECONTESTS_REVISION,
            },
        }

    if source == "omr":
        problem_source = _text(row.get("problem_source")).lower()
        if "amc_aime" in problem_source or problem_source in {"amc", "aime"}:
            return None
        problem = _text(row.get("problem"))
        answer = _text(row.get("expected_answer"))
        if not answer:
            answer = _last_boxed(_text(row.get("generated_solution"))) or ""
        if not problem or not answer:
            return None
        raw_id = _stable_id(problem)
        return {
            "problem_id": f"{source}:{raw_id}",
            "source": source,
            "kind": "math",
            "prompt": problem + MATH_PROMPT_SUFFIX,
            "tests": [],
            "answer": answer,
            "metadata": {
                "raw_id": raw_id,
                "problem_source": row.get("problem_source"),
                "source_index": index,
                "hf_revision": OMR_REVISION,
            },
        }

    if source == "dapo":
        prompt = _first_user_message(row) or _text(row.get("problem"))
        reward = row.get("reward_model")
        answer = ""
        if isinstance(reward, dict):
            answer = _text(reward.get("ground_truth") or reward.get("answer"))
        extra = row.get("extra_info")
        if not answer and isinstance(extra, dict):
            answer = _text(extra.get("answer") or extra.get("ground_truth"))
        if not answer:
            answer = _last_boxed(_text(row.get("solution"))) or ""
        if not prompt or not answer:
            return None
        raw_id = None
        if isinstance(extra, dict):
            raw_id = extra.get("index") or extra.get("id") or extra.get("problem_id")
        raw_id = raw_id if raw_id is not None else _stable_id(prompt)
        return {
            "problem_id": f"{source}:{_stable_id(raw_id)}",
            "source": source,
            "kind": "math",
            "prompt": prompt + MATH_PROMPT_SUFFIX,
            "tests": [],
            "answer": answer,
            "metadata": {
                "raw_id": raw_id,
                "data_source": row.get("data_source"),
                "source_index": index,
                "hf_revision": DAPO_REVISION,
            },
        }

    if source == "deepscaler":
        problem = _text(row.get("problem"))
        answer = _text(row.get("answer"))
        if not answer:
            answer = _last_boxed(_text(row.get("solution"))) or ""
        if not problem or not answer:
            return None
        raw_id = _stable_id(problem)
        return {
            "problem_id": f"{source}:{raw_id}",
            "source": source,
            "kind": "math",
            "prompt": problem + MATH_PROMPT_SUFFIX,
            "tests": [],
            "answer": answer,
            "metadata": {
                "raw_id": raw_id,
                "source_index": index,
                "hf_revision": DEEPSCALER_REVISION,
            },
        }

    if source == "chat":
        prompt = _first_user_message(row)
        if not prompt:
            return None
        raw_id = _stable_id(prompt)
        return {
            "problem_id": f"{source}:{raw_id}",
            "source": source,
            "kind": "chat",
            "prompt": prompt,
            "tests": [],
            "answer": None,
            "metadata": {
                "raw_id": raw_id,
                "source_index": index,
                "hf_revision": CHAT_REVISION,
            },
        }

    raise KeyError(f"unknown Stage 0 source {source!r}")


def _is_transient_error(exc: BaseException) -> bool:
    """True for download/connection errors that a retry can plausibly fix."""

    if isinstance(exc, (http.client.IncompleteRead, http.client.RemoteDisconnected)):
        return True
    if isinstance(exc, OSError):  # ConnectionError, TimeoutError, SSL, ...
        return True
    try:
        import httpx

        if isinstance(exc, httpx.HTTPError):
            return True
    except ImportError:  # pragma: no cover - httpx is a datasets dependency
        pass
    try:
        import requests

        if isinstance(exc, requests.RequestException):
            return True
    except ImportError:  # pragma: no cover - requests is a hub dependency
        pass
    try:
        from urllib3.exceptions import HTTPError as Urllib3HTTPError

        if isinstance(exc, Urllib3HTTPError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def iter_source_rows(
    source: PoolSource,
    *,
    limit_rows: int | None = None,
    shuffle_buffer: int = 0,
    seed: int = 0,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, row)`` in deterministic stream order.

    Transient download errors are retried with exponential backoff.  Each
    retry reopens the stream at the pinned revision and skips the rows already
    yielded, so the output contains no duplicates.  Persistent failures are
    re-raised after ``retry_attempts``; non-network errors are never retried.
    """

    if limit_rows is not None and limit_rows <= 0:
        return
    attempts = 0
    next_index = 0
    while True:
        try:
            stream = source.load()
            if shuffle_buffer and shuffle_buffer > 0:
                stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
            for index, row in enumerate(stream):
                if index < next_index:
                    continue
                if limit_rows is not None and next_index >= limit_rows:
                    return
                yield next_index, row
                next_index += 1
            return
        except Exception as exc:  # noqa: BLE001 - classified below
            attempts += 1
            if attempts >= max(1, retry_attempts) or not _is_transient_error(exc):
                raise
            delay = max(0.0, retry_backoff) * (2 ** (attempts - 1))
            if delay > 0:
                time.sleep(delay)


def dedup_key(problem: dict[str, Any]) -> str:
    """Across-source duplicate key: the normalised prompt."""

    return normalize(problem.get("prompt") or "")


__all__ = [
    "APPS_PARQUET_REVISION",
    "APPS_REVISION",
    "APPS_SOURCE_REVISION",
    "CHAT_REVISION",
    "CODECONTESTS_LANGUAGE_NAMES",
    "CODECONTESTS_PYTHON_LANGUAGES",
    "CODECONTESTS_REVISION",
    "DAPO_REVISION",
    "DATASET_REVISIONS",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_BACKOFF_SECONDS",
    "DEEPSCALER_REVISION",
    "MAX_HIDDEN_TESTS",
    "OMR_REVISION",
    "POOL_SOURCES",
    "PoolSource",
    "SOURCES_BY_NAME",
    "TACO_REVISION",
    "convert_row",
    "dedup_key",
    "iter_source_rows",
    "parse_input_output",
    "source_with_loader",
]
