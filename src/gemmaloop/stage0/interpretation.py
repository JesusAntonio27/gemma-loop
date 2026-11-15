"""Authenticated APPS/TACO test interpretation (X1 ruling Q1).

The frozen v2 pool converted every non-string ``input_output`` element with
``json.dumps``.  For stdin tests that is not an execution text: an original
flat list of strings must be joined with newlines, while an original string
must be used unchanged even when it looks like JSON.  This module recovers the
original element types from the already pinned APPS/TACO records, replays the
old conversion byte-for-byte, and writes a hash-bound interpretation manifest.

The representation is derived *only* from the original Python element type.
A leading ``[`` or a successful ``json.loads`` never decides anything.  A
problem with any unauthenticated, ambiguous, malformed or unsupported test is
excluded in its entirety before grading; its complete test inventory and the
exclusion reason are retained.

CodeContests is not affected by the ``input_output`` JSON conversion, but its
pinned proto records are still recovered, bound to their actual locator and
problem identity, and replayed through ``_codecontests_tests`` byte-for-byte.
There is no literal-source passthrough: a missing or mismatched record excludes
the whole problem exactly like an APPS/TACO record does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemmaloop.stage0.sources import (
    APPS_PARQUET_REVISION,
    APPS_REVISION,
    APPS_SOURCE_REVISION,
    CODECONTESTS_REVISION,
    DATASET_REVISIONS,
    MAX_HIDDEN_TESTS,
    TACO_REVISION,
    _as_list,
    _codecontests_tests,
    _json_load,
    parse_input_output,
)

INTERPRETATION_MANIFEST_NAME = "interpretation_manifest.json"
EXECUTION_TESTS_NAME = "execution_tests.jsonl"
INTERPRETATION_KIND = "pilot_shortfall_interpretation"
EXECUTION_TESTS_KIND = "pilot_shortfall_execution_tests"

REPRESENTATION_STRING = "string_unchanged"
REPRESENTATION_LIST = "newline_join_string_list"
REPRESENTATION_FUNCTIONAL_STRING = "functional_string_unchanged"
REPRESENTATION_FUNCTIONAL_JSON = "functional_json_arguments"

STATUS_AUTHENTICATED = "authenticated"
STATUS_EXCLUDED = "excluded"

_SOURCE_REVISIONS: dict[str, dict[str, Any]] = {
    "apps": {
        "hf_id": "codeparrot/apps",
        "revision": APPS_REVISION,
        "parquet_revision": APPS_PARQUET_REVISION,
        "source_revision": APPS_SOURCE_REVISION,
        "config": "all",
        "split": "train",
        "data_files": ["all/train/0000.parquet"],
        "load_mode": "refs/convert/parquet",
    },
    "taco": {
        "hf_id": "BAAI/TACO",
        "revision": TACO_REVISION,
        "source_revision": TACO_REVISION,
        "config": "ALL",
        "split": "train",
        "data_files": ["ALL/train-*.parquet"],
        "load_mode": "data_files",
    },
    "codecontests": {
        "hf_id": "deepmind/code_contests",
        "revision": CODECONTESTS_REVISION,
        "source_revision": CODECONTESTS_REVISION,
        "config": None,
        "split": "train",
        "load_mode": "parquet",
        "language_filter": "PYTHON3 then PYTHON",
    },
}


class InterpretationError(RuntimeError):
    """The interpretation manifest cannot authenticate a required test."""


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(str(text).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hash_payload(payload: Mapping[str, Any], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _type_name(value: Any) -> str:
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        element_types = sorted({_type_name(item) for item in value})
        return "list[" + ",".join(element_types) + "]"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    return type(value).__name__


def _problem_tests_sha256(tests: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for test in tests:
        data = _canonical(test).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _stored_element(value: Any) -> str:
    """Exactly the old ``parse_input_output`` serialisation."""

    return value if isinstance(value, str) else json.dumps(value)


def _representation(
    element: Any,
    stored: str,
    *,
    functional: bool,
) -> dict[str, Any]:
    """Derive the representation from the original element type only."""

    expected_stored = _stored_element(element)
    entry: dict[str, Any] = {
        "original_type": _type_name(element),
        "original_value": element,
        "stored_sha256": _sha256_text(stored),
        "supported": False,
        "representation": None,
        "execution_text": None,
        "execution_sha256": None,
        "reason": None,
    }
    if expected_stored != stored:
        entry["reason"] = (
            "stored element does not replay the old conversion "
            f"(expected {expected_stored!r})"
        )
        return entry
    if functional:
        # Functional semantics are unchanged: the stored JSON/string is what
        # the runner consumes.
        entry["supported"] = True
        entry["representation"] = (
            REPRESENTATION_FUNCTIONAL_STRING
            if isinstance(element, str)
            else REPRESENTATION_FUNCTIONAL_JSON
        )
        entry["execution_text"] = stored
        entry["execution_sha256"] = _sha256_text(stored)
        return entry
    if isinstance(element, str):
        entry["supported"] = True
        entry["representation"] = REPRESENTATION_STRING
        entry["execution_text"] = element
        entry["execution_sha256"] = _sha256_text(element)
        return entry
    if isinstance(element, list) and all(isinstance(item, str) for item in element):
        text = "\n".join(element)
        entry["supported"] = True
        entry["representation"] = REPRESENTATION_LIST
        entry["execution_text"] = text
        entry["execution_sha256"] = _sha256_text(text)
        return entry
    entry["reason"] = (
        "stdin element is neither a string nor a flat list of strings "
        f"(original type {_type_name(element)})"
    )
    return entry


# ---------------------------------------------------------------------------
# Original-record replay
# ---------------------------------------------------------------------------
def _replayed_tests(source: str, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    if source in {"apps", "taco"}:
        return parse_input_output(raw.get("input_output"))
    if source == "codecontests":
        return _codecontests_tests(raw)
    raise InterpretationError(f"unsupported code source {source!r}")


def _original_elements(
    source: str, raw: Mapping[str, Any]
) -> tuple[list[Any], list[Any], str | None]:
    """Return ``(inputs, outputs, fn_name)`` in frozen test order."""

    if source in {"apps", "taco"}:
        payload = _json_load(raw.get("input_output"))
        if not isinstance(payload, dict):
            return [], [], None
        inputs = _as_list(payload.get("inputs", payload.get("input")))
        outputs = _as_list(payload.get("outputs", payload.get("output")))
        fn = payload.get("fn_name")
        return list(inputs), list(outputs), None if fn is None else str(fn)
    if source == "codecontests":
        inputs: list[Any] = []
        outputs: list[Any] = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            block = raw.get(key) or {}
            if not isinstance(block, Mapping):
                continue
            for test_input, test_output in zip(
                _as_list(block.get("input")), _as_list(block.get("output"))
            ):
                inputs.append(test_input)
                outputs.append(test_output)
                if len(inputs) >= MAX_HIDDEN_TESTS:
                    return inputs, outputs, None
        return inputs, outputs, None
    raise InterpretationError(f"unsupported code source {source!r}")


def _exclusion(
    *,
    problem_id: str,
    source: str,
    split: str,
    reason: str,
    frozen_tests: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    inventory = list(tests or [])
    if not inventory:
        inventory = [
            {
                "index": index,
                "type": test.get("type"),
                "stored_test_sha256": _sha256_text(_canonical(test)),
                "stored_test": test,
            }
            for index, test in enumerate(frozen_tests)
        ]
    return {
        "problem_id": problem_id,
        "source": source,
        "split": split,
        "status": STATUS_EXCLUDED,
        "reason": reason,
        "n_tests": len(frozen_tests),
        "test_inventory": inventory,
    }


def derive_problem_interpretation(
    problem: Mapping[str, Any],
    frozen_row: Mapping[str, Any],
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Authenticate one problem and derive its per-test execution text."""

    problem_id = str(frozen_row["problem_id"])
    split = str(frozen_row.get("split") or "")
    source = str((record or {}).get("source") or problem.get("source") or "")
    frozen_bytes = [str(value) for value in frozen_row.get("test_bytes") or []]
    frozen_tests = [json.loads(value) for value in frozen_bytes]

    if source not in _SOURCE_REVISIONS:
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"unsupported code source {source!r}",
            frozen_tests=frozen_tests,
        )

    raw = (record or {}).get("raw")
    authentication_basis = "replayed_original_record"

    if record is None:
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"original {source} record is missing",
            frozen_tests=frozen_tests,
        )
    if record.get("identity_ok") is False:
        detail = record.get("identity_error") or "identity does not replay"
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"recovered original record identity does not match: {detail}",
            frozen_tests=frozen_tests,
        )
    if not isinstance(raw, Mapping):
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"original {source} record is missing",
            frozen_tests=frozen_tests,
        )

    try:
        replayed = _replayed_tests(source, raw)
    except Exception as exc:  # noqa: BLE001 - malformed original record
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"old conversion raised {type(exc).__name__}: {exc}",
            frozen_tests=frozen_tests,
        )
    replayed_bytes = [_canonical(test) for test in replayed]
    if replayed_bytes != frozen_bytes:
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=(
                "replayed old conversion does not match the frozen ordered "
                "test bytes"
            ),
            frozen_tests=frozen_tests,
        )
    try:
        inputs, outputs, fn_name = _original_elements(source, raw)
    except Exception as exc:  # noqa: BLE001 - malformed original record
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=f"original elements are malformed: {type(exc).__name__}: {exc}",
            frozen_tests=frozen_tests,
        )
    if len(inputs) < len(frozen_tests) or len(outputs) < len(frozen_tests):
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason="original record has fewer input/output elements than tests",
            frozen_tests=frozen_tests,
        )

    tests: list[dict[str, Any]] = []
    for index, frozen_test in enumerate(frozen_tests):
        test_type = str(frozen_test.get("type") or "stdin")
        functional = test_type == "functional"
        stored_input = str(frozen_test.get("input") or "")
        stored_output = str(frozen_test.get("output") or "")
        input_rep = _representation(
            inputs[index], stored_input, functional=functional
        )
        output_rep = _representation(
            outputs[index], stored_output, functional=functional
        )
        test_entry = {
            "index": index,
            "type": test_type,
            "fn_name": frozen_test.get("fn_name"),
            "stored_test_sha256": _sha256_text(_canonical(frozen_test)),
            "stored_input_sha256": _sha256_text(stored_input),
            "stored_output_sha256": _sha256_text(stored_output),
            "input": input_rep,
            "output": output_rep,
        }
        tests.append(test_entry)

    unsupported = [
        (test["index"], test["input"].get("reason") or test["output"].get("reason"))
        for test in tests
        if not (test["input"]["supported"] and test["output"]["supported"])
    ]
    if unsupported:
        first_index, first_reason = unsupported[0]
        return _exclusion(
            problem_id=problem_id,
            source=source,
            split=split,
            reason=(
                f"unsupported or ambiguous representation at test {first_index}: "
                f"{first_reason}"
            ),
            frozen_tests=frozen_tests,
            tests=tests,
        )

    locator = dict((record or {}).get("locator") or {})
    locator.setdefault("source", source)
    locator.setdefault("hf_id", _SOURCE_REVISIONS[source].get("hf_id"))
    locator.setdefault("revision", _SOURCE_REVISIONS[source].get("revision"))
    return {
        "problem_id": problem_id,
        "source": source,
        "split": split,
        "status": STATUS_AUTHENTICATED,
        "reason": None,
        "authentication_basis": authentication_basis,
        "record_locator": locator,
        "source_revision": (record or {}).get(
            "revision", _SOURCE_REVISIONS[source].get("revision")
        ),
        "original_problem_sha256": (
            _sha256_text(_canonical(raw)) if isinstance(raw, Mapping) else None
        ),
        "n_tests": len(frozen_tests),
        "stored_tests_sha256": _problem_tests_sha256(frozen_tests),
        "tests": tests,
    }


# ---------------------------------------------------------------------------
# Manifest build / load
# ---------------------------------------------------------------------------
def _execution_tests(problem_entry: Mapping[str, Any]) -> dict[str, Any]:
    tests = []
    for test in problem_entry.get("tests") or []:
        tests.append(
            {
                "index": int(test["index"]),
                "type": str(test["type"]),
                "fn_name": test.get("fn_name"),
                "input": str(test["input"]["execution_text"]),
                "output": str(test["output"]["execution_text"]),
                "input_representation": test["input"]["representation"],
                "output_representation": test["output"]["representation"],
                "stored_test_sha256": test["stored_test_sha256"],
                "input_sha256": test["input"]["execution_sha256"],
                "output_sha256": test["output"]["execution_sha256"],
            }
        )
    return {
        "kind": EXECUTION_TESTS_KIND,
        "problem_id": problem_entry["problem_id"],
        "source": problem_entry.get("source"),
        "split": problem_entry.get("split"),
        "n_tests": len(tests),
        "tests": tests,
    }


def build_interpretation_manifest(
    frozen_rows: Sequence[Mapping[str, Any]],
    problems_by_id: Mapping[str, Mapping[str, Any]],
    records_by_problem_id: Mapping[str, Mapping[str, Any]],
    *,
    source_revisions: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(manifest, execution_rows)`` for every frozen code problem."""

    entries: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    for frozen_row in frozen_rows:
        problem_id = str(frozen_row["problem_id"])
        problem = problems_by_id.get(problem_id)
        if problem is None:
            frozen_tests = [json.loads(value) for value in frozen_row.get("test_bytes") or []]
            entries.append(
                _exclusion(
                    problem_id=problem_id,
                    source=str(frozen_row.get("source") or ""),
                    split=str(frozen_row.get("split") or ""),
                    reason="problem is missing from the frozen pool",
                    frozen_tests=frozen_tests,
                )
            )
            continue
        record = records_by_problem_id.get(problem_id)
        entry = derive_problem_interpretation(problem, frozen_row, record)
        entries.append(entry)
        if entry["status"] == STATUS_AUTHENTICATED:
            execution_rows.append(_execution_tests(entry))

    exclusions = [
        {
            "problem_id": entry["problem_id"],
            "source": entry.get("source"),
            "split": entry.get("split"),
            "reason": entry["reason"],
            "n_tests": entry.get("n_tests"),
            "test_inventory": entry.get("test_inventory") or [],
        }
        for entry in entries
        if entry["status"] == STATUS_EXCLUDED
    ]
    authenticated = [entry for entry in entries if entry["status"] == STATUS_AUTHENTICATED]
    manifest: dict[str, Any] = {
        "kind": INTERPRETATION_KIND,
        "version": 1,
        "created_utc": created_utc,
        "rule": (
            "representation derived from the original element type; flat lists "
            "of strings are newline-joined without a terminal newline or stdin "
            "stripping; strings are unchanged; unsupported problems are "
            "excluded in their entirety"
        ),
        "source_revisions": dict(source_revisions or _SOURCE_REVISIONS),
        "problems": entries,
        "exclusions": exclusions,
        "authenticated_problem_count": len(authenticated),
        "excluded_problem_count": len(exclusions),
        "authenticated_test_count": sum(int(entry["n_tests"]) for entry in authenticated),
        "execution_tests": {
            "file": EXECUTION_TESTS_NAME,
            "sha256": None,
            "rows": len(execution_rows),
            "tests": sum(int(row["n_tests"]) for row in execution_rows),
        },
        "manifest_sha256": "",
    }
    return manifest, execution_rows


def write_interpretation_manifest(
    out_dir: Path | str,
    frozen_rows: Sequence[Mapping[str, Any]],
    problems_by_id: Mapping[str, Mapping[str, Any]],
    records_by_problem_id: Mapping[str, Mapping[str, Any]],
    *,
    source_revisions: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, execution_rows = build_interpretation_manifest(
        frozen_rows,
        problems_by_id,
        records_by_problem_id,
        source_revisions=source_revisions,
        created_utc=created_utc,
    )
    execution_text = "".join(
        _canonical(row) + "\n" for row in execution_rows
    )
    execution_path = out_dir / EXECUTION_TESTS_NAME
    execution_bytes = execution_text.encode("utf-8")
    execution_path.write_bytes(execution_bytes)
    manifest["execution_tests"]["sha256"] = _sha256_text(execution_text)
    manifest["execution_tests"]["bytes"] = len(execution_bytes)
    manifest["manifest_sha256"] = _hash_payload(manifest, "manifest_sha256")
    (out_dir / INTERPRETATION_MANIFEST_NAME).write_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    return manifest


def load_interpretation_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InterpretationError(f"{path}: expected a JSON object")
    if payload.get("kind") != INTERPRETATION_KIND:
        raise InterpretationError(f"{path}: not a pilot-shortfall interpretation")
    if payload.get("manifest_sha256") != _hash_payload(payload, "manifest_sha256"):
        raise InterpretationError(f"{path}: manifest self hash does not recompute")
    execution = payload.get("execution_tests") or {}
    execution_path = path.parent / str(execution.get("file") or "")
    if not execution_path.is_file():
        raise InterpretationError(f"{path}: execution tests file is missing")
    if _sha256_file(execution_path) != execution.get("sha256"):
        raise InterpretationError(f"{path}: execution tests file changed")
    return payload


def execution_tests_by_problem(
    manifest: Mapping[str, Any], manifest_path: Path | str
) -> dict[str, dict[str, Any]]:
    path = Path(manifest_path).parent / str(
        (manifest.get("execution_tests") or {}).get("file") or EXECUTION_TESTS_NAME
    )
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("kind") != EXECUTION_TESTS_KIND:
            raise InterpretationError(f"{path}:{line_number}: not an execution row")
        problem_id = str(row.get("problem_id") or "")
        if not problem_id or problem_id in rows:
            raise InterpretationError(
                f"{path}:{line_number}: duplicate or empty problem id"
            )
        rows[problem_id] = row
    expected = {
        str(entry["problem_id"])
        for entry in manifest.get("problems") or []
        if entry.get("status") == STATUS_AUTHENTICATED
    }
    if set(rows) != expected:
        raise InterpretationError(
            "execution tests do not cover exactly the authenticated problems "
            f"(missing={sorted(expected - set(rows))[:3]}, "
            f"extra={sorted(set(rows) - expected)[:3]})"
        )
    return rows


def excluded_problem_ids(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(entry["problem_id"]): str(entry.get("reason") or "excluded")
        for entry in manifest.get("problems") or []
        if entry.get("status") == STATUS_EXCLUDED
    }


def verify_interpretation_against_test_manifest(
    manifest: Mapping[str, Any],
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Check every frozen problem is present and every test set is accounted."""

    by_id = {str(entry["problem_id"]): entry for entry in manifest.get("problems") or []}
    frozen_ids = [str(row["problem_id"]) for row in frozen_rows]
    missing = [problem_id for problem_id in frozen_ids if problem_id not in by_id]
    extra = sorted(set(by_id) - set(frozen_ids))
    if missing or extra:
        raise InterpretationError(
            "interpretation manifest does not cover exactly the frozen problems "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )
    for frozen_row in frozen_rows:
        problem_id = str(frozen_row["problem_id"])
        entry = by_id[problem_id]
        expected_tests = [json.loads(value) for value in frozen_row.get("test_bytes") or []]
        if entry["status"] == STATUS_EXCLUDED:
            inventory = entry.get("test_inventory") or []
            if len(inventory) != len(expected_tests):
                raise InterpretationError(
                    f"{problem_id}: exclusion inventory does not retain every test"
                )
            continue
        if int(entry.get("n_tests") or -1) != len(expected_tests):
            raise InterpretationError(f"{problem_id}: test count changed")
        for index, (test, expected) in enumerate(zip(entry.get("tests") or [], expected_tests)):
            if int(test["index"]) != index:
                raise InterpretationError(f"{problem_id}: test order changed")
            if str(test["type"]) != str(expected.get("type") or "stdin"):
                raise InterpretationError(f"{problem_id}: test type changed")
            if test["stored_test_sha256"] != _sha256_text(_canonical(expected)):
                raise InterpretationError(
                    f"{problem_id}: stored test bytes changed at index {index}"
                )
    if manifest_path is not None:
        execution_tests_by_problem(manifest, manifest_path)
    return {
        "problems": len(frozen_ids),
        "authenticated": int(manifest.get("authenticated_problem_count") or 0),
        "excluded": int(manifest.get("excluded_problem_count") or 0),
        "tests": sum(int(row.get("n_tests") or 0) for row in frozen_rows),
    }


__all__ = [
    "EXECUTION_TESTS_KIND",
    "EXECUTION_TESTS_NAME",
    "INTERPRETATION_KIND",
    "INTERPRETATION_MANIFEST_NAME",
    "InterpretationError",
    "REPRESENTATION_FUNCTIONAL_JSON",
    "REPRESENTATION_FUNCTIONAL_STRING",
    "REPRESENTATION_LIST",
    "REPRESENTATION_STRING",
    "STATUS_AUTHENTICATED",
    "STATUS_EXCLUDED",
    "build_interpretation_manifest",
    "derive_problem_interpretation",
    "excluded_problem_ids",
    "execution_tests_by_problem",
    "load_interpretation_manifest",
    "verify_interpretation_against_test_manifest",
    "write_interpretation_manifest",
]
