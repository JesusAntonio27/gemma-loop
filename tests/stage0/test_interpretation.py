"""Authenticated source-type recovery and interpretation manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.stage0 import interpretation, source_recovery
from gemmaloop.stage0.shortfall import _canonical_test_bytes
from gemmaloop.stage0.sources import (
    CODECONTESTS_REVISION,
    _codecontests_tests,
    convert_row,
    parse_input_output,
)


def _frozen_row(problem_id: str, tests, *, split: str = "pilot_tfit") -> dict:
    return {
        "problem_id": problem_id,
        "split": split,
        "source": "apps",
        "n_tests": len(tests),
        "test_bytes": [
            _canonical_test_bytes(test).decode("utf-8") for test in tests
        ],
    }


def _record(payload: dict, *, source: str = "apps") -> dict:
    return {
        "source": source,
        "raw": {"input_output": json.dumps(payload)},
        "locator": {"row_index": 7, "raw_id": 7},
        "revision": "pinned-revision",
    }


def _problem(problem_id: str = "apps:x", *, source: str = "apps") -> dict:
    return {"problem_id": problem_id, "source": source, "kind": "code"}


def test_flat_list_of_strings_is_newline_joined_without_terminal_newline():
    raw = {
        "inputs": [["2", "2 chef", "4 enjoyourapplepie", "", ""]],
        "outputs": [["cehf", "eayejpuinpopolre"]],
    }
    tests = parse_input_output(json.dumps(raw))
    row = _frozen_row("apps:x", tests)
    result = interpretation.derive_problem_interpretation(
        _problem(), row, _record(raw)
    )
    assert result["status"] == "authenticated"
    assert result["tests"][0]["input"]["representation"] == (
        interpretation.REPRESENTATION_LIST
    )
    assert result["tests"][0]["input"]["execution_text"] == (
        "2\n2 chef\n4 enjoyourapplepie\n\n"
    )
    assert result["tests"][0]["output"]["execution_text"] == (
        "cehf\neayejpuinpopolre"
    )
    # No *extra* terminal newline is added; the trailing newlines above come
    # only from the preserved empty elements.
    simple = {"inputs": [["a", "b"]], "outputs": [["c"]]}
    simple_tests = parse_input_output(json.dumps(simple))
    simple_result = interpretation.derive_problem_interpretation(
        _problem(), _frozen_row("apps:x", simple_tests), _record(simple)
    )
    assert simple_result["tests"][0]["input"]["execution_text"] == "a\nb"


def test_literal_json_looking_string_stays_unchanged():
    raw = {"inputs": ['["1","2"]'], "outputs": ["[3, 4]"]}
    tests = parse_input_output(json.dumps(raw))
    assert tests[0]["input"] == '["1","2"]'
    result = interpretation.derive_problem_interpretation(
        _problem(), _frozen_row("apps:x", tests), _record(raw)
    )
    assert result["status"] == "authenticated"
    assert result["tests"][0]["input"]["representation"] == (
        interpretation.REPRESENTATION_STRING
    )
    assert result["tests"][0]["input"]["execution_text"] == '["1","2"]'
    assert result["tests"][0]["output"]["execution_text"] == "[3, 4]"


def test_unsupported_original_type_excludes_the_whole_problem():
    raw = {"inputs": [5], "outputs": [7]}
    tests = parse_input_output(json.dumps(raw))
    result = interpretation.derive_problem_interpretation(
        _problem(), _frozen_row("apps:x", tests), _record(raw)
    )
    assert result["status"] == "excluded"
    assert "unsupported" in result["reason"]
    assert len(result["test_inventory"]) == 1
    assert result["test_inventory"][0]["stored_test_sha256"]


def test_replay_mismatch_excludes_the_whole_problem():
    raw = {"inputs": ["1 2"], "outputs": ["3"]}
    tests = parse_input_output(json.dumps(raw))
    row = _frozen_row("apps:x", tests)
    row["test_bytes"] = [
        _canonical_test_bytes(
            {"input": "different", "output": "3", "type": "stdin"}
        ).decode("utf-8")
    ]
    result = interpretation.derive_problem_interpretation(
        _problem(), row, _record(raw)
    )
    assert result["status"] == "excluded"
    assert "replayed old conversion" in result["reason"]


def test_missing_apps_record_excludes_the_problem():
    tests = parse_input_output(
        json.dumps({"inputs": ["1 2"], "outputs": ["3"]})
    )
    result = interpretation.derive_problem_interpretation(
        _problem(), _frozen_row("apps:x", tests), None
    )
    assert result["status"] == "excluded"
    assert "missing" in result["reason"]


def test_manifest_is_hash_bound_and_tampering_is_refused(tmp_path: Path):
    raw = {"inputs": ["1 2"], "outputs": ["3"]}
    tests = parse_input_output(json.dumps(raw))
    row = _frozen_row("apps:x", tests)
    out = tmp_path / "interpretation"
    manifest = interpretation.write_interpretation_manifest(
        out,
        [row],
        {"apps:x": _problem()},
        {"apps:x": _record(raw)},
    )
    loaded = interpretation.load_interpretation_manifest(
        out / interpretation.INTERPRETATION_MANIFEST_NAME
    )
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert loaded["authenticated_problem_count"] == 1
    execution_path = out / interpretation.EXECUTION_TESTS_NAME
    execution_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(interpretation.InterpretationError, match="changed"):
        interpretation.load_interpretation_manifest(
            out / interpretation.INTERPRETATION_MANIFEST_NAME
        )


def _codecontests_raw(*, inputs, outputs, name="contest-x"):
    return {
        "name": name,
        "description": "Read stdin and write stdout.",
        "public_tests": {"input": list(inputs), "output": list(outputs)},
        "private_tests": {"input": [], "output": []},
        "generated_tests": {"input": [], "output": []},
        "solutions": {"language": [3], "solution": ["print('ok')"]},
        "source": 1,
    }


def _codecontests_record(raw, *, identity_ok=True, identity_error=None):
    return {
        "source": "codecontests",
        "raw": raw,
        "locator": {
            "source": "codecontests",
            "row_index": 3,
            "raw_id": raw.get("name"),
            "hf_revision": CODECONTESTS_REVISION,
        },
        "revision": CODECONTESTS_REVISION,
        "identity_ok": identity_ok,
        "identity_error": identity_error,
    }


def test_codecontests_ordered_record_is_recovered_and_replayed():
    nl = chr(10)
    raw = _codecontests_raw(
        inputs=["1 2" + nl, "3 4" + nl], outputs=["3" + nl, "7" + nl]
    )
    problem = convert_row("codecontests", raw, 3)
    assert problem is not None
    tests = _codecontests_tests(raw)
    row = _frozen_row(problem["problem_id"], tests)
    result = interpretation.derive_problem_interpretation(
        problem, row, _codecontests_record(raw)
    )
    assert result["status"] == "authenticated"
    assert result["authentication_basis"] == "replayed_original_record"
    assert result["record_locator"]["row_index"] == 3
    assert result["record_locator"]["raw_id"] == raw["name"]
    assert result["source_revision"] == CODECONTESTS_REVISION
    assert result["original_problem_sha256"]
    assert [test["index"] for test in result["tests"]] == [0, 1]
    # Original elements, their types and hashes are recorded in order.
    assert [test["input"]["original_value"] for test in result["tests"]] == [
        "1 2" + nl,
        "3 4" + nl,
    ]
    assert [test["output"]["execution_text"] for test in result["tests"]] == [
        "3" + nl,
        "7" + nl,
    ]
    assert all(
        test["input"]["original_type"] == "str"
        and test["input"]["representation"] == interpretation.REPRESENTATION_STRING
        and test["input"]["execution_sha256"]
        for test in result["tests"]
    )


def test_codecontests_missing_record_excludes_the_whole_problem():
    nl = chr(10)
    raw = _codecontests_raw(inputs=["1 2" + nl], outputs=["3" + nl])
    problem = convert_row("codecontests", raw, 3)
    assert problem is not None
    row = _frozen_row(problem["problem_id"], _codecontests_tests(raw))
    result = interpretation.derive_problem_interpretation(problem, row, None)
    assert result["status"] == "excluded"
    assert "missing" in result["reason"]
    assert len(result["test_inventory"]) == 1


def test_codecontests_identity_mismatch_excludes_the_whole_problem():
    nl = chr(10)
    raw = _codecontests_raw(inputs=["1 2" + nl], outputs=["3" + nl])
    problem = convert_row("codecontests", raw, 3)
    assert problem is not None
    row = _frozen_row(problem["problem_id"], _codecontests_tests(raw))
    result = interpretation.derive_problem_interpretation(
        problem,
        row,
        _codecontests_record(
            raw,
            identity_ok=False,
            identity_error="recovered identity does not match",
        ),
    )
    assert result["status"] == "excluded"
    assert "identity" in result["reason"]
    assert len(result["test_inventory"]) == 1


def test_codecontests_replay_mismatch_excludes_the_whole_problem():
    nl = chr(10)
    raw = _codecontests_raw(
        inputs=["1 2" + nl, "3 4" + nl], outputs=["3" + nl, "7" + nl]
    )
    problem = convert_row("codecontests", raw, 3)
    assert problem is not None
    reordered = list(reversed(_codecontests_tests(raw)))
    row = _frozen_row(problem["problem_id"], reordered)
    result = interpretation.derive_problem_interpretation(
        problem, row, _codecontests_record(raw)
    )
    assert result["status"] == "excluded"
    assert "replayed old conversion" in result["reason"]
    assert len(result["test_inventory"]) == 2


def test_recover_records_binds_codecontests_locator_and_flags_missing_identity():
    nl = chr(10)
    raw = _codecontests_raw(inputs=["1 2" + nl], outputs=["3" + nl])
    problem = convert_row("codecontests", raw, 3)
    assert problem is not None
    row = _frozen_row(problem["problem_id"], _codecontests_tests(raw))
    seen = []

    def loader(source, needed):
        seen.append((source, set(needed)))
        if 3 in needed:
            yield 3, raw

    records = source_recovery.recover_records(
        [row],
        {problem["problem_id"]: problem},
        loader=loader,
        sources=("codecontests",),
    )
    record = records[problem["problem_id"]]
    assert seen == [("codecontests", {3})]
    assert record["raw"] is raw
    assert record["identity_ok"] is True
    assert record["locator"]["row_index"] == 3
    assert record["locator"]["raw_id"] == raw["name"]
    assert record["revision"] == CODECONTESTS_REVISION

    missing = source_recovery.recover_records(
        [row],
        {problem["problem_id"]: problem},
        loader=lambda source, needed: iter(()),
        sources=("codecontests",),
    )[problem["problem_id"]]
    assert missing["identity_ok"] is False
    assert "no codecontests record" in missing["identity_error"]
