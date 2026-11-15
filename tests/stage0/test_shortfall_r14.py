"""Fix round 14: v3 verifier-identity repair, void receipt and v3r guards.

The production-path regression deliberately keeps the real
``default count adapter -> grade_code -> DockerSandbox.run_tests`` call chain
and controls only the external Docker transport (by running the real Docker
daemon).  It is skipped only when Docker itself is unavailable; the mandatory
real-Docker dry run in ``scripts/dry_run_v3r.py`` is the release-gating
exercise.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

from gemmaloop.stage0 import grader, shortfall
from tests.stage0 import test_shortfall as ts


ROOT = Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return grader.DockerSandbox().available()
    except Exception:  # pragma: no cover - defensive
        return False


DOCKER_AVAILABLE = _docker_available()


# ---------------------------------------------------------------------------
# Interface guard and identity forwarding
# ---------------------------------------------------------------------------
def _called_keywords(source: str, function_name: str) -> set[str]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = (
            target.id
            if isinstance(target, ast.Name)
            else target.attr
            if isinstance(target, ast.Attribute)
            else None
        )
        if name == function_name:
            found.update(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
    return found


def test_production_adapter_passes_only_accepted_identity_keywords():
    grade_code_signature = inspect.signature(grader.grade_code)
    adapter_keywords = _called_keywords(
        inspect.getsource(shortfall._grade_code_default), "grade_code"
    )
    assert "container_names" in adapter_keywords
    assert "attempts" in adapter_keywords
    assert adapter_keywords <= set(grade_code_signature.parameters), (
        "the production adapter passes a keyword grade_code does not accept: "
        f"{sorted(adapter_keywords - set(grade_code_signature.parameters))}"
    )

    run_tests_signature = inspect.signature(grader.DockerSandbox.run_tests)
    forwarding_keywords = _called_keywords(
        inspect.getsource(grader.grade_code), "run_tests"
    )
    assert {"container_names", "attempts"} <= forwarding_keywords
    assert forwarding_keywords <= set(run_tests_signature.parameters), (
        "grade_code passes a keyword DockerSandbox.run_tests does not accept: "
        f"{sorted(forwarding_keywords - set(run_tests_signature.parameters))}"
    )


def test_call_adapter_refuses_to_drop_identity_keywords():
    def three_arg_grader(code, tests, timeout):  # pragma: no cover - never called
        raise AssertionError("must not be called")

    with pytest.raises(shortfall.ShortfallVerifierInterfaceError, match="never be dropped"):
        shortfall._call_grade_fn(
            three_arg_grader,
            "code",
            [{"input": "", "output": ""}],
            1.0,
            container_name="c",
            attempt="a",
        )


def test_verifier_type_error_blocks_dispatch_without_consuming_an_attempt(
    tmp_path: Path,
):
    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    count_dir = ts._count_out(tmp_path, "blocked-run")
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME

    def broken_grader(code, tests, timeout, **kwargs):
        raise TypeError("verifier signature mismatch")

    with pytest.raises(shortfall.ShortfallVerifierInterfaceError, match="blocked"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=ts._interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=count_dir,
            grade_fn=broken_grader,
            workers=1,
            max_infrastructure_retries=1,
            recovery_check=ts._recovery_ok,
            authorization=ts._test_authorization(
                tmp_path, fixture["lite"], out_dir=count_dir
            ),
        )

    records = shortfall._read_attempt_ledger(ledger_path)
    errors = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_VERIFIER_INTERFACE_ERROR
    ]
    assert errors, "the interface error was not recorded durably"
    assert errors[0]["blocks_dispatch"] is True
    assert errors[0]["error_type"] == "TypeError"
    assert not [
        record
        for record in records
        if record.get("kind")
        in {shortfall.LEDGER_ATTEMPT_RESULT, shortfall.LEDGER_ATTEMPT_INTERRUPTED}
    ], "a programming error must never become a per-candidate attempt result"
    assert not (count_dir / shortfall.COUNT_PER_REQUEST_NAME).exists()


def test_resume_after_verifier_interface_error_stays_blocked_across_invocations(
    tmp_path: Path,
):
    """X1 B1: the durable error is a run-wide block on every later resume."""

    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    interpretation_path = ts._interpretation(fixture, tmp_path, freeze_path)
    count_dir = ts._count_out(tmp_path, "interface-blocked")
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME

    def broken_grader(code, tests, timeout, **kwargs):
        raise TypeError("verifier signature mismatch")

    observed: list[tuple[int, list[int]]] = []
    # Initial run, then the first and second resumes from X1's reproduction.
    for invocation, resume in enumerate((False, True, True)):
        with pytest.raises(
            shortfall.ShortfallVerifierInterfaceError, match="run-wide|blocked|error"
        ):
            shortfall.count_verified(
                freeze_path=freeze_path,
                interpretation_path=interpretation_path,
                topup_plan_path=plan_path,
                out_dir=count_dir,
                grade_fn=broken_grader,
                workers=1,
                max_infrastructure_retries=1,
                recovery_check=ts._recovery_ok,
                authorization=ts._test_authorization(
                    tmp_path, fixture["lite"], out_dir=count_dir
                ),
                resume=resume,
            )
        records = shortfall._read_attempt_ledger(ledger_path)
        errors = shortfall._ledger_verifier_interface_error_records(records)
        starts = [
            record
            for record in records
            if record.get("kind") == shortfall.LEDGER_ATTEMPT_START
        ]
        assert len(errors) == 1, (invocation, errors)
        # No retry, no new attempt start, no interrupted/terminal disposition.
        assert [int(record["attempt"]) for record in starts] == [1], invocation
        assert not [
            record
            for record in records
            if record.get("kind")
            in {
                shortfall.LEDGER_ATTEMPT_INTERRUPTED,
                shortfall.LEDGER_ATTEMPT_RESULT,
            }
        ], invocation
        observed.append((len(errors), [int(record["attempt"]) for record in starts]))

    assert observed == [(1, [1]), (1, [1]), (1, [1])]
    assert not (count_dir / shortfall.COUNT_PER_REQUEST_NAME).exists()
    assert not (count_dir / shortfall.COUNT_SUMMARY_NAME).exists()
    # Planning itself refuses the error; it is never treated as an ordinary
    # interrupted in-flight attempt that offers the registered retry.
    error_records = [
        record
        for record in shortfall._read_attempt_ledger(ledger_path)
        if record.get("kind") == shortfall.LEDGER_VERIFIER_INTERFACE_ERROR
    ]
    assert error_records
    with pytest.raises(shortfall.ShortfallVerifierInterfaceError, match="run-wide"):
        shortfall._ledger_plan(error_records)


def test_loader_refuses_a_forged_complete_count_whose_ledger_hides_a_verifier_error(
    tmp_path: Path,
):
    """X1 B1: a forged complete count may not launder the durable error."""

    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    interpretation_path = ts._interpretation(fixture, tmp_path, freeze_path)
    count_dir = ts._count_out(tmp_path, "forged-loader")
    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=interpretation_path,
        topup_plan_path=plan_path,
        out_dir=count_dir,
        grade_fn=ts._fake_grade,
        workers=1,
        max_infrastructure_retries=0,
        recovery_check=ts._recovery_ok,
        authorization=ts._test_authorization(
            tmp_path, fixture["lite"], out_dir=count_dir
        ),
    )
    assert summary["status"] == shortfall.COUNT_STATUS_COMPLETE

    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    shortfall._append_attempt_ledger(
        ledger_path,
        {
            "kind": shortfall.LEDGER_VERIFIER_INTERFACE_ERROR,
            "request_id": "forged:request",
            "problem_id": "forged",
            "test_index": 0,
            "attempt": 1,
            "error_type": "TypeError",
            "error": "forged verifier signature mismatch",
            "failed_utc": shortfall._utc_now(),
            "blocks_dispatch": True,
        },
    )

    # Forge the derived hashes so the summary/manifest still claim a complete,
    # internally consistent count while hiding the error from all accounting.
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    ledger_sha = shortfall._sha256_file(ledger_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["attempt_ledger"]["sha256"] = ledger_sha
    summary_payload["artifacts"][shortfall.ATTEMPT_LEDGER_NAME][
        "sha256"
    ] = ledger_sha
    summary_path.write_text(
        json.dumps(
            summary_payload, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["files"][shortfall.ATTEMPT_LEDGER_NAME] = (
        shortfall._file_entry(shortfall.ATTEMPT_LEDGER_NAME, ledger_path)
    )
    manifest_payload["files"][shortfall.COUNT_SUMMARY_NAME] = (
        shortfall._file_entry(shortfall.COUNT_SUMMARY_NAME, summary_path)
    )
    manifest_payload["manifest_sha256"] = shortfall._hash_payload(
        manifest_payload, "manifest_sha256"
    )
    manifest_path.write_text(
        json.dumps(
            manifest_payload, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verifier interface/programming error"):
        shortfall.load_verified_count(
            count_dir,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
        )


def _append_forged_interface_error(count_dir: Path) -> Path:
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    shortfall._append_attempt_ledger(
        ledger_path,
        {
            "kind": shortfall.LEDGER_VERIFIER_INTERFACE_ERROR,
            "request_id": "forged:request",
            "problem_id": "forged",
            "test_index": 0,
            "attempt": 1,
            "error_type": "TypeError",
            "error": "forged verifier signature mismatch",
            "failed_utc": shortfall._utc_now(),
            "blocks_dispatch": True,
        },
    )
    return ledger_path


def _rebind_count_ledger(count_dir: Path, *, drop_binding: bool = False) -> Path:
    """Recompute the summary/manifest bindings around the current ledger bytes."""

    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    ledger_sha = shortfall._sha256_file(ledger_path)
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["attempt_ledger"]["sha256"] = ledger_sha
    summary["artifacts"][shortfall.ATTEMPT_LEDGER_NAME]["sha256"] = ledger_sha
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if drop_binding:
        manifest["files"].pop(shortfall.ATTEMPT_LEDGER_NAME, None)
    else:
        manifest["files"][shortfall.ATTEMPT_LEDGER_NAME] = shortfall._file_entry(
            shortfall.ATTEMPT_LEDGER_NAME, ledger_path
        )
    manifest["files"][shortfall.COUNT_SUMMARY_NAME] = shortfall._file_entry(
        shortfall.COUNT_SUMMARY_NAME, summary_path
    )
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return ledger_path


def _relabel_count_v3r(count_dir: Path) -> None:
    """Give a synthetic ready fixture the operative v3r count identity."""

    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["version"] = "3r"
    summary["run_version"] = "v3r"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "3r"
    manifest["run_version"] = "v3r"
    manifest["files"][shortfall.COUNT_SUMMARY_NAME] = shortfall._file_entry(
        shortfall.COUNT_SUMMARY_NAME, summary_path
    )
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _complete_count_index(
    tmp_path: Path, fixture: dict, freeze_path: Path, plan_path: Path, label: str
) -> tuple[Path, Path]:
    count_dir = ts._official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    _relabel_count_v3r(count_dir)
    manifest_path, _manifest = ts._final_corpus(
        tmp_path, fixture, freeze_path, count_dir, name=f"corpus-{label}"
    )
    index_path = tmp_path / f"index-{label}.json"
    shortfall.write_freeze_index(
        repo_root=ROOT,
        freeze_path=freeze_path,
        count_summary_path=count_dir / shortfall.COUNT_SUMMARY_NAME,
        topup_plan_path=plan_path,
        out_path=index_path,
        final_manifest_path=manifest_path,
        tokenizer=object(),
        renderer=ts._fake_renderer,
        run_version="v3r",
    )
    return count_dir, index_path


def _refresh_index_count_bindings(index_path: Path, count_dir: Path) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    replacements = {
        "count_manifest": count_dir / shortfall.COUNT_MANIFEST_NAME,
        "count_summary": count_dir / shortfall.COUNT_SUMMARY_NAME,
    }
    for entry in index["bindings"]:
        target = replacements.get(str(entry.get("label") or ""))
        if target is not None:
            entry["sha256"] = shortfall._sha256_file(target)
            entry["bytes"] = target.stat().st_size
    index["index_sha256"] = shortfall._hash_payload(index, "index_sha256")
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_loader_and_readiness_refuse_an_omitted_attempt_ledger_binding(
    tmp_path: Path,
):
    """X1 B1: omitting the ledger binding must not launder a durable error."""

    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    count_dir, index_path = _complete_count_index(
        tmp_path, fixture, freeze_path, plan_path, "omit-ledger"
    )

    _append_forged_interface_error(count_dir)
    _rebind_count_ledger(count_dir, drop_binding=True)

    with pytest.raises(ValueError, match="not bound"):
        shortfall.load_verified_count(
            count_dir,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
        )

    _refresh_index_count_bindings(index_path, count_dir)
    with pytest.raises(shortfall.ShortfallReadinessError, match="not bound"):
        shortfall.check_freeze_readiness(index_path)


def test_loader_refuses_summary_manifest_ledger_disagreement(tmp_path: Path):
    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    count_dir = ts._official_count(tmp_path, fixture, freeze_path, plan_path)

    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["attempt_ledger"]["sha256"] = "0" * 64
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][shortfall.COUNT_SUMMARY_NAME] = shortfall._file_entry(
        shortfall.COUNT_SUMMARY_NAME, summary_path
    )
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree on the attempt ledger hash"):
        shortfall.load_verified_count(
            count_dir,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
        )


def test_loader_and_readiness_refuse_changed_attempt_ledger_bytes(
    tmp_path: Path,
):
    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    count_dir, index_path = _complete_count_index(
        tmp_path, fixture, freeze_path, plan_path, "changed-ledger"
    )
    shortfall._append_attempt_ledger(
        count_dir / shortfall.ATTEMPT_LEDGER_NAME,
        {"kind": "synthetic-ledger-tamper", "note": "changed after binding"},
    )

    with pytest.raises(ValueError, match="changed"):
        shortfall.load_verified_count(
            count_dir,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
        )

    with pytest.raises(shortfall.ShortfallReadinessError, match="changed"):
        shortfall.check_freeze_readiness(index_path)


def test_loader_and_readiness_refuse_a_ledger_header_from_another_run(
    tmp_path: Path,
):
    fixture = ts._fixture(tmp_path)
    freeze_path = ts._freeze(fixture, tmp_path)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    count_dir, index_path = _complete_count_index(
        tmp_path, fixture, freeze_path, plan_path, "foreign-ledger"
    )
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    records = shortfall._read_attempt_ledger(ledger_path)
    header = next(
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_HEADER
    )
    header["freeze_manifest_sha256"] = "0" * 64
    header["header_sha256"] = shortfall._hash_payload(header, "header_sha256")
    ledger_path.write_bytes(
        b"".join(shortfall._encode_ledger_record(record) for record in records)
    )
    _rebind_count_ledger(count_dir)

    with pytest.raises(ValueError, match="does not agree with the claimed run"):
        shortfall.load_verified_count(
            count_dir,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
        )

    _refresh_index_count_bindings(index_path, count_dir)
    with pytest.raises(
        shortfall.ShortfallReadinessError, match="does not agree with the claimed run"
    ):
        shortfall.check_freeze_readiness(index_path)


# ---------------------------------------------------------------------------
# Real production path: default adapter -> grade_code -> DockerSandbox.run_tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker daemon is not available")
def test_real_docker_count_path_forwards_identity_and_loads_independently(
    tmp_path: Path,
):
    fixture = ts._fixture(tmp_path)
    # Keep the fixture small but real: one candidate request per problem.
    request_ids = [rid for rid in fixture["request_ids"] if rid.endswith(":0")]
    (fixture["lite"] / "request_ids.txt").write_text(
        "".join(rid + "\n" for rid in request_ids), encoding="utf-8"
    )
    records = [
        record
        for record in fixture["records"]
        if record["request_id"] in set(request_ids)
    ]
    grades = [
        grade
        for grade in fixture["grades"]
        if grade["request_id"] in set(request_ids)
    ]
    for record in records:
        problem = next(
            item
            for item in fixture["problems"]
            if item["problem_id"] == record["problem_id"]
        )
        if problem["kind"] != "code":
            continue
        code = "import sys\nsys.stdout.write(sys.stdin.read())"
        record["content"] = f"```python\n{code}\n```"
        record["completion"] = "step " * 20 + record["content"]
    (fixture["lite"] / "shards" / "shard_00000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (fixture["lite"] / "sample_grades.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in grades), encoding="utf-8"
    )

    pin = shortfall.resolve_docker_image("python:3.12-slim")
    grader.DockerSandbox(
        image="python:3.12-slim", expected_digest=pin["pinned"]
    ).verify_image_digest()
    freeze_path = ts._freeze(fixture, tmp_path, image_pin=pin)
    plan_path = ts._plan(fixture, tmp_path, freeze_path)
    interpretation_path = ts._interpretation(fixture, tmp_path, freeze_path)
    count_dir = ts._count_out(tmp_path, "real-run")
    authorization = ts._test_authorization(
        tmp_path, fixture["lite"], out_dir=count_dir
    )

    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=interpretation_path,
        topup_plan_path=plan_path,
        out_dir=count_dir,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=ts._recovery_ok,
        authorization=authorization,
    )
    assert summary["status"] == shortfall.COUNT_STATUS_COMPLETE

    records = shortfall._read_attempt_ledger(
        count_dir / shortfall.ATTEMPT_LEDGER_NAME
    )
    starts = {
        (
            record["request_id"],
            int(record["test_index"]),
            int(record["attempt"]),
        ): record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_START
    }
    results = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
    ]
    assert results
    for result in results:
        key = (
            result["request_id"],
            int(result["test_index"]),
            int(result["attempt"]),
        )
        start = starts[key]
        evidence = result["evidence"]
        assert evidence["container_name"] == start["container_name"]
        assert evidence["lifecycle_attempt"] == start["attempt_token"]
        assert evidence["lifecycle_valid"] is True
        assert evidence["lifecycle_frames_raw"]
        assert start["container_name"].startswith("gemmaloop-")
        assert start["attempt"] == 1

    loaded = shortfall.load_verified_count(
        count_dir,
        topup_plan_path=plan_path,
        freeze_path=freeze_path,
    )
    assert loaded["status"] == summary["status"]
    assert loaded["accounting"]["requests"] == summary["accounting"]["requests"]
    assert loaded["run_version"] == "v3"

    per_request = [
        json.loads(line)
        for line in (
            count_dir / shortfall.COUNT_PER_REQUEST_NAME
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    code_rows = [row for row in per_request if row["domain"] == "code"]
    assert code_rows
    for row in code_rows:
        for test in row["per_test"]:
            assert test["container_name"] in {
                start["container_name"]
                for start in starts.values()
                if start["request_id"] == row["request_id"]
                and int(start["test_index"]) == int(test["index"])
            }
            assert test["lifecycle_attempt"] in {
                start["attempt_token"]
                for start in starts.values()
                if start["request_id"] == row["request_id"]
                and int(start["test_index"]) == int(test["index"])
            }


# ---------------------------------------------------------------------------
# Void receipt, continuity and the one-time v3r guard
# ---------------------------------------------------------------------------
def _synthetic_crash_tree(tmp_path: Path) -> tuple[Path, dict]:
    lite = tmp_path / "lite"
    shortfall_dir = lite / "shortfall"
    v3 = shortfall_dir / "v3"
    (v3 / "verifier_v3").mkdir(parents=True)
    (v3 / "count_v3").mkdir(parents=True)
    registry = {
        "kind": shortfall.V3_REGISTRY_KIND,
        "version": 1,
        "status": shortfall.V3_STATUS_CRASHED,
        "official": True,
        "created_utc": "2026-11-06T20:23:38+00:00",
    }
    (shortfall_dir / "v3_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (v3 / "verifier_v3" / shortfall.FREEZE_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.COUNT_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.ATTEMPT_LEDGER_NAME).write_text(
        "", encoding="utf-8"
    )
    ruling = tmp_path / "ruling.md"
    ruling.write_text("synthetic void ruling\n", encoding="utf-8")
    receipt = shortfall.write_void_receipt(
        lite,
        ruling_path=ruling,
        enforce_known_hashes=False,
        void_out_root=v3,
    )
    shortfall.write_v3r_continuity(lite, ruling_path=ruling)
    return lite, receipt


def test_void_receipt_binds_bytes_and_rejects_void_artifacts(tmp_path: Path):
    lite, receipt = _synthetic_crash_tree(tmp_path)
    loaded = shortfall.load_void_receipt(lite)
    assert loaded["receipt_sha256"] == receipt["receipt_sha256"]
    assert loaded["synthetic"] is True
    assert loaded["disposition"] == shortfall.VOID_DISPOSITION
    with pytest.raises(shortfall.ShortfallVoidRunError, match="audit"):
        shortfall.load_freeze_manifest(
            lite
            / "shortfall"
            / "v3"
            / "verifier_v3"
            / shortfall.FREEZE_MANIFEST_NAME
        )
    with pytest.raises(shortfall.ShortfallVoidRunError, match="audit"):
        shortfall.load_verified_count(lite / "shortfall" / "v3" / "count_v3")
    with pytest.raises(shortfall.ShortfallVoidRunError, match="VOID"):
        shortfall._claim_v3_registry(
            repo_root=ROOT, lite_dir=lite, out_root=tmp_path / "v3"
        )


def test_void_receipt_refuses_a_changed_preserved_file(tmp_path: Path):
    lite, _receipt = _synthetic_crash_tree(tmp_path)
    ledger = lite / "shortfall" / "v3" / "count_v3" / shortfall.ATTEMPT_LEDGER_NAME
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(shortfall.ShortfallVoidRunError, match="changed"):
        shortfall.load_void_receipt(lite)


def test_v3r_guard_requires_review_and_rejects_bypass_and_second_claim(
    tmp_path: Path,
):
    lite, _receipt = _synthetic_crash_tree(tmp_path)
    fixed_out = lite / "shortfall" / "v3r"
    with pytest.raises(shortfall.ShortfallV3RunError, match="review"):
        shortfall.require_v3r_guard(lite, out_root=fixed_out, require_official=False)
    with pytest.raises(shortfall.ShortfallV3RunError, match="alternate-directory"):
        shortfall.require_v3r_guard(
            lite, out_root=tmp_path / "elsewhere", require_official=False
        )

    review = shortfall.record_v3r_replacement_review(
        lite,
        reviewer="synthetic-review",
        implementation={"sha256": "0" * 64},
        require_official=False,
    )
    assert review["replacement_review"]["status"] == "recorded"
    continued = shortfall.record_v3r_replacement_review
    with pytest.raises(shortfall.ShortfallV3RunError, match="already recorded"):
        continued(
            lite,
            reviewer="second",
            implementation={},
            require_official=False,
        )

    registry_path, registry = shortfall._claim_v3r_registry(
        repo_root=ROOT, lite_dir=lite, out_root=fixed_out
    )
    assert registry["run_version"] == "v3r"
    assert registry["attempt_base"] == 1
    assert registry["max_infrastructure_retries"] == 1
    void_receipt_file = (
        lite / "shortfall" / shortfall.VOID_RECEIPT_NAME
    )
    assert registry["void_receipt_sha256"] == shortfall._sha256_file(void_receipt_file)
    assert registry["replaces_void_receipt_sha256"] == registry["void_receipt_sha256"]
    with pytest.raises(shortfall.ShortfallV3RunError, match="second fresh"):
        shortfall._claim_v3r_registry(
            repo_root=ROOT, lite_dir=lite, out_root=fixed_out
        )
    assert registry_path.is_file()


def test_v3r_retry_policy_is_the_registered_single_retry(tmp_path: Path):
    fixture = ts._pipeline_fixture(tmp_path)
    with pytest.raises(shortfall.ShortfallV3RunError, match="exactly one"):
        shortfall.run_v3r_pipeline(
            **{
                **ts._v3_kwargs(fixture, tmp_path, tmp_path / "v3r"),
                "max_infrastructure_retries": 2,
            }
        )


def test_v3r_attempts_start_at_one_and_retry_once(tmp_path: Path, monkeypatch):
    fixture = ts._pipeline_fixture(tmp_path)
    lite, _receipt = _synthetic_crash_tree(tmp_path / "continuity")
    # Reuse the pipeline fixture's lite/pool but install a synthetic
    # continuity tree beside it so the v3r claim is satisfied.
    shortfall_dir = fixture["lite"] / "shortfall"
    shortfall_dir.mkdir(parents=True, exist_ok=True)
    _other_lite, _ = _synthetic_crash_tree(tmp_path / "other")
    for name in (
        shortfall.V3_REGISTRY_NAME,
        shortfall.VOID_RECEIPT_NAME,
        shortfall.V3R_CONTINUITY_NAME,
    ):
        shutil.copyfile(_other_lite / "shortfall" / name, shortfall_dir / name)
    ruling = shortfall_dir / "ruling.md"
    ruling.write_text("synthetic void ruling\n", encoding="utf-8")
    # Rebuild the receipt/continuity against this lite so paths agree.
    for name in (
        shortfall.V3_REGISTRY_NAME,
        shortfall.VOID_RECEIPT_NAME,
        shortfall.V3R_CONTINUITY_NAME,
    ):
        (shortfall_dir / name).unlink()
    v3 = shortfall_dir / "v3"
    (v3 / "verifier_v3").mkdir(parents=True)
    (v3 / "count_v3").mkdir(parents=True)
    (shortfall_dir / shortfall.V3_REGISTRY_NAME).write_text(
        json.dumps(
            {
                "kind": shortfall.V3_REGISTRY_KIND,
                "version": 1,
                "status": shortfall.V3_STATUS_CRASHED,
                "official": True,
            }
        ),
        encoding="utf-8",
    )
    (v3 / "verifier_v3" / shortfall.FREEZE_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.COUNT_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.ATTEMPT_LEDGER_NAME).write_text(
        "", encoding="utf-8"
    )
    shortfall.write_void_receipt(
        fixture["lite"],
        ruling_path=ruling,
        enforce_known_hashes=False,
        void_out_root=v3,
    )
    shortfall.write_v3r_continuity(fixture["lite"], ruling_path=ruling)
    shortfall.record_v3r_replacement_review(
        fixture["lite"],
        reviewer="synthetic",
        implementation={},
        require_official=False,
    )
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)
    result = shortfall.run_v3r_pipeline(
        **{
            **ts._v3_kwargs(fixture, tmp_path, fixture["lite"] / "shortfall" / "v3r"),
            "max_infrastructure_retries": 1,
        }
    )
    count_dir = Path(result["count"]).parent
    records = shortfall._read_attempt_ledger(
        count_dir / shortfall.ATTEMPT_LEDGER_NAME
    )
    starts = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_START
    ]
    assert starts
    assert {int(record["attempt"]) for record in starts} == {1}
    assert result["registry_status"] == shortfall.V3_STATUS_COMPLETE


def test_v3r_guard_rejects_missing_and_inconsistent_continuity(tmp_path: Path):
    empty_lite = tmp_path / "empty-lite"
    with pytest.raises(shortfall.ShortfallV3RunError, match="continuity record is missing"):
        shortfall.require_v3r_guard(empty_lite, require_official=False)

    lite, _receipt = _synthetic_crash_tree(tmp_path / "inconsistent")
    continuity_path = shortfall.v3r_continuity_path(lite)
    payload = json.loads(continuity_path.read_text(encoding="utf-8"))
    payload["allowed_claims"] = 2
    continuity_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(shortfall.ShortfallV3RunError, match="self hash"):
        shortfall.require_v3r_guard(lite, require_official=False)

    shortfall.write_v3r_continuity  # keep the writer import surface exercised


# ---------------------------------------------------------------------------
# Production v3r CLI binding and literal dry-run entry point
# ---------------------------------------------------------------------------
def test_v3r_production_cli_is_bound_as_an_entry_point(tmp_path: Path):
    implementation = shortfall.executing_implementation(repo_root=ROOT)
    labels = {entry["label"] for entry in implementation["entry_points"]}
    assert "scripts/freeze_recount_v3r.py" in labels
    assert "scripts/freeze_recount_v3.py" in labels


def test_dry_run_invokes_the_production_v3r_cli_main():
    source = (ROOT / "scripts" / "dry_run_v3r.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "main"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "freeze_recount_v3r"
    ]
    assert main_calls, "the dry run must invoke freeze_recount_v3r.main()"
    direct_pipeline_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"run_v3r_pipeline", "resume_v3r_pipeline"}
    ]
    assert not direct_pipeline_calls, (
        "the dry run must go through the production CLI main(), not call the "
        "pipeline directly"
    )
