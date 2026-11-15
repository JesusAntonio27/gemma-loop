"""Fix round 16: immutable v3r resume identity and fail-closed readiness.

R1: the durable attempt ledger binds the immutable claim-time run identity and
an audit-only count-start registry digest.  Registry updates (including a
resume) advance the registry file hash but never invalidate an authorized
resume.  The production v3r orchestration is interrupted after conclusive
records are durable and resumed twice to completion without regrading them.

R2: operative readiness always requires the declared v3r run version and the
bound, fully verified count; omitting that metadata can never disable the
loader or interface-error scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.stage0 import shortfall
from tests.stage0 import test_shortfall as ts


def _install_continuity(lite: Path) -> None:
    """Install a synthetic (non-official) void -> continuity -> v3r claim tree."""

    shortfall_dir = lite / "shortfall"
    shortfall_dir.mkdir(parents=True, exist_ok=True)
    (shortfall_dir / "v3").mkdir(parents=True, exist_ok=True)
    (shortfall_dir / "v3" / "verifier_v3").mkdir(
        parents=True, exist_ok=True
    )
    (shortfall_dir / "v3" / "count_v3").mkdir(parents=True, exist_ok=True)
    (shortfall_dir / shortfall.V3_REGISTRY_NAME).write_text(
        json.dumps(
            {
                "kind": shortfall.V3_REGISTRY_KIND,
                "version": 1,
                "status": shortfall.V3_STATUS_CRASHED,
                "official": True,
                "created_utc": "2026-11-06T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (shortfall_dir / "v3" / "verifier_v3" / shortfall.FREEZE_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (shortfall_dir / "v3" / "count_v3" / shortfall.COUNT_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (shortfall_dir / "v3" / "count_v3" / shortfall.ATTEMPT_LEDGER_NAME).write_text(
        "", encoding="utf-8"
    )
    ruling = shortfall_dir / "synthetic-void-ruling.md"
    ruling.write_text("synthetic void ruling\n", encoding="utf-8")
    shortfall.write_void_receipt(
        lite,
        ruling_path=ruling,
        enforce_known_hashes=False,
        void_out_root=shortfall_dir / "v3",
    )
    shortfall.write_v3r_continuity(lite, ruling_path=ruling)
    shortfall.record_v3r_replacement_review(
        lite,
        reviewer="synthetic",
        implementation={},
        require_official=False,
    )


def _records(ledger: Path) -> list[dict]:
    return shortfall._read_attempt_ledger(ledger)


def _conclusive_keys(records) -> set[tuple[str, int, int]]:
    return {
        (
            str(record.get("request_id")),
            int(record.get("test_index", -1)),
            int(record.get("attempt") or 1),
        )
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
        and record.get("conclusive") is True
    }


def _result_counts(records) -> dict[tuple[str, int, int], int]:
    counts: dict[tuple[str, int, int], int] = {}
    for record in records:
        if record.get("kind") != shortfall.LEDGER_ATTEMPT_RESULT:
            continue
        key = (
            str(record.get("request_id")),
            int(record.get("test_index", -1)),
            int(record.get("attempt") or 1),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_v3r_resume_after_durable_conclusive_records_preserves_the_ledger_prefix(
    tmp_path: Path, monkeypatch
):
    """Production v3r orchestration: interrupt, resume twice, never regrade."""

    fixture = ts._pipeline_fixture(tmp_path)
    _install_continuity(fixture["lite"])
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)
    out_root = fixture["lite"] / "shortfall" / "v3r"
    kwargs = ts._v3_kwargs(fixture, tmp_path, out_root)
    count_dir = out_root / shortfall.V3R_STAGE_DIRECTORIES["count"]
    ledger = count_dir / shortfall.ATTEMPT_LEDGER_NAME

    real_append = shortfall._append_attempt_ledger

    def interrupt_after_first_result(message: str):
        state = {"results": 0}

        def append(path, record):
            real_append(path, record)
            if record.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT:
                state["results"] += 1
                if state["results"] == 1:
                    raise SystemExit(message)

        return append

    monkeypatch.setattr(
        shortfall,
        "_append_attempt_ledger",
        interrupt_after_first_result("interrupt after a durable conclusive result"),
    )
    with pytest.raises(SystemExit, match="durable conclusive"):
        shortfall.run_v3r_pipeline(**{**kwargs, "grade_fn": ts._fake_grade})

    assert ledger.is_file()
    prefix_records = _records(ledger)
    prefix_conclusive = _conclusive_keys(prefix_records)
    assert prefix_conclusive
    prefix_bytes = ledger.read_bytes()
    registry_path = shortfall.v3r_registry_path(fixture["lite"])
    crashed = json.loads(registry_path.read_text(encoding="utf-8"))
    assert crashed["status"] == shortfall.V3_STATUS_CRASHED
    claim = crashed["claim_registry_sha256"]
    assert claim
    count_start_digest = next(
        record["count_start_registry_digest"]
        for record in prefix_records
        if record.get("kind") == shortfall.LEDGER_HEADER
    )
    assert len(count_start_digest) == 64
    assert count_start_digest != crashed["registry_sha256"]

    # First resume: registry updates advance its self hash, then another
    # synthetic interruption leaves more durable records.
    monkeypatch.setattr(
        shortfall,
        "_append_attempt_ledger",
        interrupt_after_first_result("second synthetic interruption"),
    )
    with pytest.raises(SystemExit, match="second synthetic"):
        shortfall.resume_v3r_pipeline(
            **{**kwargs, "grade_fn": ts._fake_grade}
        )
    after_first_resume = ledger.read_bytes()
    assert after_first_resume.startswith(prefix_bytes)
    resumed_records = _records(ledger)
    resumed_counts = _result_counts(resumed_records)
    for key in prefix_conclusive:
        assert resumed_counts.get(key) == 1
    registry_after_first = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_after_first["claim_registry_sha256"] == claim
    assert registry_after_first["resume_count"] == 1
    assert registry_after_first["registry_sha256"] != crashed["registry_sha256"]

    # Second resume: complete, still preserving every earlier ledger byte.
    monkeypatch.setattr(shortfall, "_append_attempt_ledger", real_append)
    result = shortfall.resume_v3r_pipeline(**{**kwargs, "grade_fn": ts._fake_grade})
    assert result["registry_status"] == shortfall.V3_STATUS_COMPLETE
    final_bytes = ledger.read_bytes()
    assert final_bytes.startswith(after_first_resume)
    final_records = _records(ledger)
    final_counts = _result_counts(final_records)
    for key in prefix_conclusive:
        assert final_counts.get(key) == 1
    assert _conclusive_keys(final_records) >= prefix_conclusive
    final_header = next(
        record
        for record in final_records
        if record.get("kind") == shortfall.LEDGER_HEADER
    )
    assert final_header["count_start_registry_digest"] == count_start_digest
    assert final_header["claim_registry_sha256"] == claim
    final_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert final_registry["claim_registry_sha256"] == claim
    assert final_registry["resume_count"] == 2
    assert final_registry["status"] == shortfall.V3_STATUS_COMPLETE

    # The completed count still loads under independent verification.
    loaded = shortfall.load_verified_count(
        count_dir,
        topup_plan_path=out_root / shortfall.V3R_PLAN_NAME,
        freeze_path=(
            out_root
            / shortfall.V3R_STAGE_DIRECTORIES["freeze"]
            / shortfall.FREEZE_MANIFEST_NAME
        ),
    )
    assert loaded["status"] == shortfall.COUNT_STATUS_COMPLETE


def _mutate_index(index_path: Path, mutate) -> dict:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["index_sha256"] = shortfall._hash_payload(payload, "index_sha256")
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_readiness_refuses_an_index_that_omits_run_version_and_count_binding(
    tmp_path: Path,
):
    """X1 R2 mutation: removing both fields must never produce acceptance."""

    index = ts._write_index(tmp_path, tfit=20, selection=30, name="r2-base.json")
    index_path = index["path"]
    assert shortfall.check_freeze_readiness(index_path)

    def strip_version_and_count(payload):
        payload.pop("run_version", None)
        payload["bindings"] = [
            entry
            for entry in payload["bindings"]
            if entry.get("label")
            not in {"count_manifest", "count_summary"}
        ]

    mutated = _mutate_index(index_path, strip_version_and_count)
    with pytest.raises(shortfall.ShortfallReadinessError, match="run_version"):
        shortfall.check_freeze_readiness(index_path)
    with pytest.raises(shortfall.ShortfallReadinessError, match="run_version"):
        shortfall.check_pilot_inputs(
            mutated,
            fit_data=index["rendered"]["pilot_tfit"],
            test_data=index["rendered"]["pilot_selection"],
        )

    # A declared non-operative version is refused just as a missing one is.
    for version in ("v1", "v2", "v3", "void"):
        def set_version(payload, value=version):
            payload["run_version"] = value

        _mutate_index(index_path, set_version)
        with pytest.raises(shortfall.ShortfallReadinessError, match="run_version"):
            shortfall.check_freeze_readiness(index_path)


def test_readiness_refuses_a_v3r_index_without_its_count_binding(
    tmp_path: Path,
):
    """X1 R2 mutation: an explicit v3r index cannot omit its count evidence."""

    index = ts._write_index(tmp_path, tfit=20, selection=30, name="r2-v3r.json")
    index_path = index["path"]

    def strip_count_only(payload):
        payload["bindings"] = [
            entry
            for entry in payload["bindings"]
            if entry.get("label") not in {"count_manifest", "count_summary"}
        ]

    mutated = _mutate_index(index_path, strip_count_only)
    with pytest.raises(shortfall.ShortfallReadinessError, match="count"):
        shortfall.check_freeze_readiness(index_path)
    with pytest.raises(shortfall.ShortfallReadinessError, match="count"):
        shortfall.check_pilot_inputs(
            mutated,
            fit_data=index["rendered"]["pilot_tfit"],
            test_data=index["rendered"]["pilot_selection"],
        )


def test_readiness_runs_the_full_count_loader_including_interface_errors(
    tmp_path: Path,
):
    """A bound count with a durable verifier error is refused by readiness."""

    index = ts._write_index(tmp_path, tfit=20, selection=30, name="r2-error.json")
    count_dir = index["index_dir"] / "count"
    shortfall._append_attempt_ledger(
        count_dir / shortfall.ATTEMPT_LEDGER_NAME,
        {
            "kind": shortfall.LEDGER_VERIFIER_INTERFACE_ERROR,
            "request_id": "tfit:c0:0",
            "test_index": 0,
            "attempt": 1,
            "error_type": "TypeError",
            "error": "synthetic interface error",
        },
    )
    # Rebind the changed ledger bytes exactly as a forger would.
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    summary["attempt_ledger"]["sha256"] = shortfall._sha256_file(ledger_path)
    summary["artifacts"][shortfall.ATTEMPT_LEDGER_NAME]["sha256"] = (
        shortfall._sha256_file(ledger_path)
    )
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in (
        (shortfall.COUNT_SUMMARY_NAME, summary_path),
        (shortfall.ATTEMPT_LEDGER_NAME, ledger_path),
    ):
        manifest["files"][name] = {
            "path": str(path.resolve()),
            "sha256": shortfall._sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    payload = json.loads(index["path"].read_text(encoding="utf-8"))
    for entry in payload["bindings"]:
        if entry.get("label") == "count_summary":
            entry["sha256"] = shortfall._sha256_file(summary_path)
        elif entry.get("label") == "count_manifest":
            entry["sha256"] = shortfall._sha256_file(manifest_path)
    payload["index_sha256"] = shortfall._hash_payload(payload, "index_sha256")
    index["path"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(shortfall.ShortfallReadinessError, match="interface"):
        shortfall.check_freeze_readiness(index["path"])
    with pytest.raises(shortfall.ShortfallReadinessError, match="interface"):
        shortfall.check_pilot_inputs(
            payload,
            fit_data=index["rendered"]["pilot_tfit"],
            test_data=index["rendered"]["pilot_selection"],
        )
