"""Unit tests for the separate synthetic real-Docker verifier selftest."""

from __future__ import annotations

import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "selftest_verifier_sandbox_under_test",
        ROOT / "scripts" / "selftest_verifier_sandbox.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _fake_runner(returncode=0, stdout="", stderr="", calls=None):
    def runner(command, **kwargs):
        if calls is not None:
            calls.append(list(command))
        return SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    return runner


def test_selftest_refuses_a_missing_local_image_and_never_pulls():
    script = _load_script()
    calls: list[list[str]] = []
    with pytest.raises(script.SyntheticValidationError, match="never pulls"):
        script.inspect_local_image(
            "python:3.12-slim",
            runner=_fake_runner(returncode=1, stderr="No such image", calls=calls),
        )
    assert calls
    assert all("pull" not in command for command in calls)


def test_selftest_digest_mismatch_is_refused_before_any_candidate(tmp_path: Path):
    script = _load_script()
    entry = json.dumps(
        [{"Id": "sha256:" + "a" * 64, "RepoDigests": []}]
    )
    with pytest.raises(script.SyntheticValidationError, match="not the pinned"):
        script.verify_local_digest(
            "python:3.12-slim",
            expected_digest="sha256:" + "0" * 64,
            runner=_fake_runner(stdout=entry),
        )


def test_selftest_marker_cases_are_parameterised_and_fail_closed():
    script = _load_script()
    assert script.MARKER_CASES
    for case in script.MARKER_CASES:
        classification, error = script.classify_marker_case(case)
        assert script._class_is_infrastructure(classification), case
    names = {case["name"] for case in script.MARKER_CASES}
    assert {"missing_markers", "altered_markers", "out_of_order_markers"} <= names


def test_selftest_late_receipt_case_is_deterministic_and_rejects_the_late_entry():
    script = _load_script()
    case = script.late_receipt_deterministic_case()
    assert case["ok"] is True
    assert case["detail"]["late_entry_received"] is True
    assert case["detail"]["late_entry_rejected"] is True
    assert case["detail"]["host_start_received_within_budget"] is False
    assert case["classification"] in {
        "startup_failure",
        "unresolved",
    }
    # The tightened late-start matcher accepts only the two documented
    # dispositions; other entry errors cannot satisfy it.
    assert script.LATE_START_RECORD_ERRORS == {
        "execution entry was not received by the host before its registered "
        "deadline",
        "expected exactly one successful execution-entry record, found 0",
    }
    assert not case["detail"]["record_error"].startswith("expected exactly")


def test_selftest_receipt_is_strictly_synthetic(tmp_path: Path):
    script = _load_script()
    out = tmp_path / "receipt.json"
    receipt = script._write_receipt(
        out,
        image="python:3.12-slim",
        local={"id": "sha256:" + "a" * 64, "repo_digests": []},
        cases=[
            {
                "name": "known_pass",
                "ok": True,
                "classification": "completed",
                "detail": {},
            }
        ],
    )
    assert receipt["official"] is False
    assert receipt["count_result"] is False
    assert receipt["synthetic_only"] is True
    stored = json.loads(out.read_text(encoding="utf-8"))
    claimed = stored.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(stored, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert claimed == expected


def test_selftest_skip_docker_writes_only_synthetic_marker_cases(tmp_path: Path):
    script = _load_script()
    entry = json.dumps(
        [{"Id": "sha256:" + "a" * 64, "RepoDigests": ["python@sha256:" + "a" * 64]}]
    )
    receipt = script.run_synthetic_validation(
        image="python:3.12-slim",
        expected_digest="sha256:" + "a" * 64,
        out_path=tmp_path / "receipt.json",
        runner=_fake_runner(stdout=entry),
        skip_docker=True,
    )
    names = {case["name"] for case in receipt["cases"]}
    assert "digest_mismatch_refused" in names
    assert "missing_markers" in names
    assert receipt["official"] is False
    assert receipt["count_result"] is False
