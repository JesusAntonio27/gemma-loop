"""Pilot-shortfall amendment: freeze, count, plan, corpus and guard tests."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaloop.stage0 import grader, modal_lite, shortfall

ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW = lambda: datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _count_out(base: Path | str, name: str = "count") -> Path:
    """A stage-count target whose parent is the claimed run's out_root."""

    return Path(base) / name / "count_v3"


def _test_authorization(
    base: Path | str,
    lite_dir: Path | str,
    *,
    out_dir: Path | str | None = None,
    stage: str = "count",
) -> shortfall.V3RunAuthorization:
    """A real authorization token bound to a temporary registry.

    This exercises the production guard (it is not a bypass): tests supply the
    same claimed-run token/registry the official stage requires, while their
    fake grader provably never reaches Docker.
    """

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    out_root = (
        Path(out_dir).parent if out_dir is not None else base / "test-v3-root"
    )
    token = hashlib.sha256(
        (str(Path(out_dir or out_root).resolve()) + str(Path(lite_dir).resolve())).encode()
    ).hexdigest()
    payload = {
        "kind": shortfall.V3_REGISTRY_KIND,
        "version": 1,
        "status": shortfall.V3_STATUS_CLAIMED,
        "created_utc": "2026-11-06T00:00:00+00:00",
        "updated_utc": "2026-11-06T00:00:00+00:00",
        "official": True,
        "run_token": token,
        "repo_root": str(ROOT.resolve()),
        "lite_dir": str(Path(lite_dir).resolve()),
        "out_root": str(Path(out_root).resolve()),
        "current_stage": None,
        "last_completed_stage": None,
        "failed_stage": None,
        "stages": [],
        "resume_count": 0,
        "error": None,
        "claim_registry_sha256": "",
        "registry_sha256": "",
    }
    shortfall._seal_registry_claim(payload)
    path = base / f"test-v3-registry-{token[:12]}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return shortfall.V3RunAuthorization(
        registry_path=path, run_token=token, stage=stage
    )


# ---------------------------------------------------------------------------
# Fixtures: a tiny frozen pool + lite run
# ---------------------------------------------------------------------------
def _problem(problem_id: str, kind: str, *, tests=None, split="pilot_dev") -> dict:
    return {
        "problem_id": problem_id,
        "prompt": f"prompt for {problem_id}",
        "kind": kind,
        "answer": "1" if kind == "math" else "None",
        "tests": list(tests or []),
        "source": problem_id.split(":")[0],
        "split": split,
        "pilot_subsplit": None,
        "bucket": 59,
    }


def _stdin_test(value: str) -> dict:
    return {"input": value, "output": value, "type": "stdin"}


def _fixture(tmp_path: Path) -> dict:
    pool = tmp_path / "pool"
    lite = tmp_path / "modal_lite"
    (pool / "splits").mkdir(parents=True, exist_ok=True)
    (lite / "shards").mkdir(parents=True, exist_ok=True)
    p1 = [
        _problem("p1:m0", "math"),
        _problem("p1:m1", "math"),
        _problem("p1:c0", "code", tests=[_stdin_test("1")]),
        _problem("p1:notest", "code", tests=[]),
        _problem("p1:chat", "chat"),
        _problem("p1:omitted", "code", tests=[_stdin_test("9")]),
    ]
    tfit = [
        _problem("tfit:m0", "math"),
        _problem("tfit:m1", "math"),
        _problem("tfit:m2", "math"),
        _problem("tfit:c0", "code", tests=[_stdin_test("2"), _stdin_test("3")]),
        _problem("tfit:c1", "code", tests=[_stdin_test("4")]),
        _problem("tfit:notest", "code", tests=[]),
        _problem("tfit:chat", "chat"),
        _problem("tfit:omitted", "code", tests=[_stdin_test("8")]),
    ]
    selection = [
        _problem("sel:m0", "math"),
        _problem("sel:m1", "math"),
        _problem("sel:m2", "math"),
        _problem("sel:c0", "code", tests=[_stdin_test("5")]),
        _problem("sel:c1", "code", tests=[_stdin_test("6")]),
        _problem("sel:notest", "code", tests=[]),
        _problem("sel:chat", "chat"),
        _problem("sel:omitted", "code", tests=[_stdin_test("7")]),
    ]
    split_rows = {
        "pilot_p1": p1,
        "pilot_tfit": tfit,
        "pilot_selection": selection,
    }
    all_problems = p1 + tfit + selection
    (pool / "problems.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_problems), encoding="utf-8"
    )
    for name, rows in split_rows.items():
        text = "".join(json.dumps(row) + "\n" for row in rows)
        (pool / f"split_{name}.jsonl").write_text(text, encoding="utf-8")
    pool_manifest = {
        "stage": "stage0_pool",
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(
                    (pool / "problems.jsonl").read_bytes()
                ).hexdigest(),
                "rows": len(all_problems),
            }
        },
    }
    pool_manifest["manifest_sha256"] = shortfall._hash_payload(
        pool_manifest, "manifest_sha256"
    )
    (pool / "manifest.json").write_text(
        json.dumps(pool_manifest), encoding="utf-8"
    )

    selected_ids = [
        row["problem_id"]
        for rows in split_rows.values()
        for row in rows
        if not row["problem_id"].endswith(":omitted")
    ]
    request_ids = [f"{pid}:{index}" for pid in selected_ids for index in range(4)]
    (lite / "request_ids.txt").write_text(
        "".join(rid + "\n" for rid in request_ids), encoding="utf-8"
    )
    (lite / "run_marker.json").write_text(
        json.dumps({"run_id": "b605af7a625f93c7e3e08a02"}), encoding="utf-8"
    )
    (lite / "pilot").mkdir(parents=True, exist_ok=True)
    (lite / "pilot" / "manifest.json").write_text(
        json.dumps({"model_revision": "rev-test"}), encoding="utf-8"
    )

    records = []
    grades = []
    for request_id in request_ids:
        pid, index_text = request_id.rsplit(":", 1)
        index = int(index_text)
        kind = next(row["kind"] for row in all_problems if row["problem_id"] == pid)
        passed = pid in {
            "tfit:c0",
            "tfit:c1",
            "sel:c0",
            "sel:c1",
        }
        content = (
            (
                "```python\nPASS " + request_id + "\n```"
                if passed
                else "```python\nFAIL " + request_id + "\n```"
            )
            if kind == "code"
            else "The answer is 1."
        )
        reasoning = "step " * 20
        record = {
            "problem_id": pid,
            "sample_index": index,
            "request_id": request_id,
            "prompt": f"prompt for {pid}",
            "completion": reasoning + content,
            "reasoning": reasoning,
            "content": content,
            "finish_reason": "stop",
            "thought_closed": True,
            "tokens": 80,
            "prompt_tokens": 40,
            "engine": "modal-vllm",
            "model_revision": "rev-test",
            "shard_index": 0,
        }
        records.append(record)
        correct = {
            "p1:m0": True,
            "tfit:m0": True,
            "tfit:m1": True,
            "sel:m0": True,
            "sel:m1": True,
            "sel:m2": True,
        }.get(pid)
        kept = not (pid == "p1:c0" and index == 0)
        grades.append(
            {
                "request_id": request_id,
                "problem_id": pid,
                "sample_index": index,
                "kind": kind,
                "split": "pilot_dev",
                "kept": kept,
                "drop_reason": None if kept else "reasoning_share",
                "correct": correct if kept and kind == "math" else None,
                "rendered_tokens": 120,
                "reasoning_share": 0.8,
                "timing_s": 0.0,
            }
        )
    (lite / "shards" / "shard_00000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (lite / "sample_grades.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in grades), encoding="utf-8"
    )
    return {
        "pool": pool,
        "lite": lite,
        "problems": all_problems,
        "request_ids": request_ids,
        "records": records,
        "grades": grades,
    }


def _freeze(fixture: dict, tmp_path: Path, **kwargs) -> Path:
    out = tmp_path / "verifier"
    pin = kwargs.pop(
        "image_pin",
        {
            "tag": "python:3.12-slim",
            "id": "sha256:" + "a" * 64,
            "digest": "sha256:" + "a" * 64,
            "pinned": "python@sha256:" + "a" * 64,
            "repo_digests": ["python@sha256:" + "a" * 64],
        },
    )
    shortfall.freeze_verifier(
        repo_root=ROOT,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        request_ids_path=fixture["lite"] / "request_ids.txt",
        out_dir=out,
        image_pin=pin,
        **kwargs,
    )
    return out / shortfall.FREEZE_MANIFEST_NAME


def _plan(fixture: dict, tmp_path: Path, freeze_path: Path) -> Path:
    out = tmp_path / "pilot_topup_plan.json"
    shortfall.plan_topup(freeze_path, out)
    return out


def _interpretation(
    fixture: dict, tmp_path: Path, freeze_path: Path, *, name: str = "interpretation"
) -> Path:
    """Build an authenticated interpretation manifest for the tiny fixture."""

    from gemmaloop.stage0 import interpretation

    freeze = shortfall.load_freeze_manifest(freeze_path)
    frozen_rows = [
        json.loads(line)
        for line in (
            freeze_path.parent / freeze["tests"]["manifest_file"]
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = {row["problem_id"]: row for row in fixture["problems"]}
    records: dict[str, dict] = {}
    for row in frozen_rows:
        problem = problems[row["problem_id"]]
        inputs = [test["input"] for test in problem["tests"]]
        outputs = [test["output"] for test in problem["tests"]]
        raw = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs})
        }
        records[row["problem_id"]] = {
            "source": "apps",
            "raw": raw,
            "locator": {"row_index": 0},
            "revision": "fixture",
        }
    out = tmp_path / name
    interpretation.write_interpretation_manifest(
        out, frozen_rows, problems, records
    )
    return out / interpretation.INTERPRETATION_MANIFEST_NAME


def _signed_fake_lifecycle(index: int, *, completed: bool = True):
    """Build re-verifiable authenticated frames for a fake conclusion."""

    key = "d" * 64
    attempt = f"fake-attempt-{index}"
    records = [
        {
            "v": 1,
            "attempt": attempt,
            "seq": 1,
            "event": grader.LIFECYCLE_EVENT_READY,
            "wall": 1.0,
            "monotonic": 100.0,
            "pid": 1,
        },
        {
            "v": 1,
            "attempt": attempt,
            "seq": 2,
            "event": grader.LIFECYCLE_EVENT_BUDGET,
            "wall": 1.1,
            "monotonic": 100.1,
            "pid": 1,
            "remaining_s": 5.0,
        },
        {
            "v": 1,
            "attempt": attempt,
            "seq": 3,
            "event": grader.LIFECYCLE_EVENT_START,
            "wall": 1.2,
            "monotonic": 100.2,
            "pid": 1,
            "child_pid": 77,
        },
    ]
    if completed:
        records.append(
            {
                "v": 1,
                "attempt": attempt,
                "seq": 4,
                "event": grader.LIFECYCLE_EVENT_FINISH,
                "wall": 1.3,
                "monotonic": 100.3,
                "pid": 1,
                "child_pid": 77,
                "returncode": 0,
                "terminated": True,
                "exit_observed": True,
            }
        )
    frames = []
    for record in records:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            key.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        frames.append(mac + " GML1 " + body + "\n")
    return key, attempt, records, frames


def _fake_grade(code: str, tests, timeout, **kwargs):
    passed = "PASS" in code
    rows = []
    for index, test in enumerate(tests):
        key, attempt, records, frames = _signed_fake_lifecycle(index)
        rows.append(
            {
                "index": index,
                "passed": passed,
                "input": str(test.get("input") or ""),
                "expected": str(test.get("output") or ""),
                "stdout": (
                    str(test.get("output") or "")
                    if passed
                    else "not the expected output"
                ),
                "stderr": "",
                "error": None if passed else "wrong answer",
                "timed_out": False,
                "duration_s": 0.01,
                "exit_code": 0,
                "test_type": str(test.get("type") or "stdin"),
                "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
                "lifecycle_valid": True,
                "cleanup_verified": True,
                "handoff_success": True,
                "termination_verified": True,
                "deadline_alive": False,
                "exit_observed": True,
                "completed_within_budget": True,
                "lifecycle_attempt": attempt,
                "lifecycle_auth_key": key,
                "lifecycle_frames_raw": frames,
                "lifecycle_records": records,
                "lifecycle_markers": records,
            }
        )
    return SimpleNamespace(
        to_dict=lambda: {
            "passed": passed,
            "sandboxed": True,
            "n_tests": len(tests),
            "error": None if passed else "wrong answer",
            "duration_s": 0.01,
            "timed_out": False,
            "tests": rows,
        }
    )


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def test_freeze_manifest_is_ordered_and_hash_bound(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    payload = shortfall.load_freeze_manifest(freeze_path)

    assert payload["verifier"]["docker_image"]["pinned"] == (
        "python@sha256:" + "a" * 64
    )
    assert payload["verifier"]["docker_image"]["tag"] == "python:3.12-slim"
    assert payload["verifier"]["docker"]["memory_mb"] == 512
    assert payload["verifier"]["docker"]["tmpfs"].endswith("size=64m")
    assert payload["verifier"]["timeout_s"] == 6.0
    assert payload["verifier"]["lifecycle"]["enforcement_actions"] == list(
        grader.DEADLINE_ENFORCEMENT_ACTIONS
    )
    assert payload["verifier"]["lifecycle"]["enforcement_routes"] == list(
        grader.DEADLINE_ENFORCEMENT_ROUTES
    )
    assert "wrapper" in payload["verifier"]["lifecycle"]["enforcement_route_rule"]
    assert (
        payload["verifier"]["lifecycle"]["enforcement_route_bound_s"]
        == grader.LIFECYCLE_NOTICE_GRACE_S
    )
    assert payload["tests"]["problems"] == 5  # p1:c0, tfit:c0, tfit:c1, sel:c0, sel:c1
    assert payload["tests"]["tests"] == 6
    assert payload["tests"]["exclusions"] == [
        {"problem_id": "p1:notest", "split": "pilot_p1", "reason": "no_tests"},
        {"problem_id": "tfit:notest", "split": "pilot_tfit", "reason": "no_tests"},
        {"problem_id": "sel:notest", "split": "pilot_selection", "reason": "no_tests"},
    ]
    # The no-test code problems are recorded, not silently dropped.
    assert payload["tests"]["problems"] == 5
    assert payload["selected_inputs"]["splits"]["pilot_p1"]["omitted_problem_ids"] == [
        "p1:omitted"
    ]

    manifest_path = freeze_path.parent / shortfall.FREEZE_TEST_MANIFEST_NAME
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["problem_id"] for row in rows] == [
        "p1:c0",
        "tfit:c0",
        "tfit:c1",
        "sel:c0",
        "sel:c1",
    ]
    tfit_c0 = rows[1]
    assert tfit_c0["test_bytes"] == [
        json.dumps(_stdin_test("2"), ensure_ascii=False, sort_keys=True),
        json.dumps(_stdin_test("3"), ensure_ascii=False, sort_keys=True),
    ]
    assert tfit_c0["tests_sha256"] == shortfall._problem_tests_sha256(
        [_stdin_test("2"), _stdin_test("3")]
    )
    # Mutating a bound evidence byte must be refused.
    requests = fixture["lite"] / "request_ids.txt"
    requests.write_text(requests.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        shortfall.load_freeze_manifest(freeze_path)


def test_no_test_code_problem_is_an_explicit_exclusion(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    payload = shortfall.load_freeze_manifest(freeze_path)
    excluded = {
        (row["problem_id"], row["split"], row["reason"])
        for row in payload["tests"]["exclusions"]
    }
    assert ("p1:notest", "pilot_p1", "no_tests") in excluded
    assert ("tfit:notest", "pilot_tfit", "no_tests") in excluded
    assert ("sel:notest", "pilot_selection", "no_tests") in excluded
    # No empty test set ever enters the ordered verifier manifest.
    assert all(
        json.loads(line)["n_tests"] > 0
        for line in (
            freeze_path.parent / shortfall.FREEZE_TEST_MANIFEST_NAME
        ).read_text(encoding="utf-8").splitlines()
        if line
    )


def test_grader_accepts_and_enforces_an_image_digest(monkeypatch):
    digest = "sha256:" + "b" * 64

    def fake_inspect(image):
        return {
            "Id": digest,
            "RepoDigests": [f"python@{digest}"],
        }

    monkeypatch.setattr(grader, "inspect_docker_image", fake_inspect)
    sandbox = grader.DockerSandbox("python:3.12-slim", expected_digest=digest)
    assert sandbox.verify_image_digest()["digest"] == digest

    mismatch = grader.DockerSandbox(
        "python:3.12-slim", expected_digest="sha256:" + "c" * 64
    )
    with pytest.raises(grader.DockerImageDigestError):
        mismatch.verify_image_digest()
    with pytest.raises(ValueError):
        grader.DockerSandbox("python:3.12-slim", expected_digest="python:3.12-slim")


# ---------------------------------------------------------------------------
# Count and decision boundaries
# ---------------------------------------------------------------------------
def test_extension_decision_boundaries_19_20_and_29_30():
    assert shortfall.decide_extension(19, 30) is True
    assert shortfall.decide_extension(20, 30) is False
    assert shortfall.decide_extension(20, 29) is True
    assert shortfall.decide_extension(20, 31) is False


def _count_entries(tfit: int, selection: int, p1: int = 0):
    entries = []
    for index in range(tfit):
        entries.append(
            {
                "problem_id": f"t{index}",
                "split": "pilot_tfit",
                "domain": "math",
                "verified": True,
                "kept": True,
            }
        )
    for index in range(selection):
        entries.append(
            {
                "problem_id": f"s{index}",
                "split": "pilot_selection",
                "domain": "math",
                "verified": True,
                "kept": True,
            }
        )
    for index in range(p1):
        entries.append(
            {
                "problem_id": f"p{index}",
                "split": "pilot_p1",
                "domain": "math",
                "verified": True,
                "kept": True,
            }
        )
    return entries


def test_compute_counts_uses_distinct_problems_and_ignores_p1_gate():
    counts = shortfall.compute_counts(_count_entries(19, 30, p1=0))
    assert counts["nT0"] == 19 and counts["nS0"] == 30
    assert counts["extension_required"] is True
    counts = shortfall.compute_counts(_count_entries(20, 30, p1=0))
    assert counts["extension_required"] is False
    counts = shortfall.compute_counts(_count_entries(20, 29, p1=50))
    assert counts["extension_required"] is True
    # Repeated rows never inflate distinct counts.
    repeated = _count_entries(20, 30, p1=0) + _count_entries(20, 30, p1=0)
    assert shortfall.compute_counts(repeated)["nT0"] == 20


def test_count_grades_only_kept_code_and_records_the_plan(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    graded_ids: list[str] = []

    def grade_fn(code, tests, timeout, **kwargs):
        graded_ids.append(code)
        return _fake_grade(code, tests, timeout)

    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=_count_out(tmp_path, "count"),
        authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count")),
        grade_fn=grade_fn,
        workers=1,
    )
    assert summary["nT0"] == 4  # two math + two code
    assert summary["nS0"] == 5  # three math + two code
    assert summary["pilot_p1"] == 1  # descriptive only
    assert summary["extension_required"] is True
    assert summary["topup_plan"]["sha256"] == hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    entries = [
        json.loads(line)
        for line in (_count_out(tmp_path, "count") / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(entries) == len(fixture["request_ids"])
    # Only kept code samples with a nonempty frozen test set reach the
    # verifier (p1:c0:0 is filtered; no-test problems are never run).  The
    # ledger commits and dispatches one test at a time, so the 19 kept code
    # requests with tests expand to 23 per-test dispatches.
    assert len(graded_ids) == 23
    assert any(
        entry["error"] == "filtered:reasoning_share" for entry in entries
    )
    assert (_count_out(tmp_path, "count") / shortfall.COUNT_MANIFEST_NAME).is_file()


def test_count_refuses_without_the_precommitted_plan(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    with pytest.raises(FileNotFoundError, match="before the count/decision"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            topup_plan_path=tmp_path / "missing_plan.json",
            out_dir=_count_out(tmp_path, "count"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count")),
            grade_fn=_fake_grade,
            workers=1,
        )


# ---------------------------------------------------------------------------
# Top-up plan
# ---------------------------------------------------------------------------
def test_topup_plan_lists_exact_indices_seeds_and_no_p1_or_chat(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan = shortfall.plan_topup(freeze_path, tmp_path / "plan.json")
    plan = shortfall.validate_topup_plan(plan)

    assert plan["request_indices"] == [4, 5, 6, 7]
    assert plan["run_cap_usd"] == 3.0
    assert plan["timeout_seconds"] == shortfall.topup_timeout_seconds(
        plan["request_count"]
    )
    problems = {
        request["problem_id"]
        for request in plan["requests"]
        if request["kind"] in {"math", "code"}
    }
    assert len(plan["requests"]) == 4 * len(problems)
    assert all(item["request_id"] == f"{item['problem_id']}:{item['sample_index']}" for item in plan["requests"])
    assert all(
        item["seed"] == modal_lite.lite_request_seed(0, item["request_id"])
        for item in plan["requests"]
    )
    assert all(item["kind"] in {"math", "code"} for item in plan["requests"])
    assert all("pilot_p1" not in item["splits"] for item in plan["requests"])
    assert plan["requests_by_split"] == {
        "pilot_tfit": 4 * 6,
        "pilot_selection": 4 * 6,
    }
    assert plan["topup_run_id"] != plan["parent_run_id"]


# ---------------------------------------------------------------------------
# Final corpus
# ---------------------------------------------------------------------------
def test_corpus_order_dedupes_and_counts_distinct_problems():
    def trace(request_id, pid, index):
        return {
            "request_id": request_id,
            "problem_id": pid,
            "sample_index": index,
            "kind": "math",
            "messages": [],
            "rendered_tokens": 10,
        }

    rows = [
        trace("b:0", "b", 0),
        trace("a:0", "a", 0),
        trace("a:1", "a", 1),
        trace("a:0", "a", 0),  # exact duplicate collapses
    ]
    ordered, manifest = shortfall.assemble_corpus(rows, split_name="pilot_tfit")
    expected = sorted(
        {row["request_id"]: row for row in rows},
        key=shortfall._request_order_key,
    )
    assert [row["request_id"] for row in ordered] == expected
    assert manifest["rows"] == 3
    assert manifest["distinct_problems"] == 2
    with pytest.raises(ValueError, match="P1"):
        shortfall.assemble_corpus(
            [trace("p1:4", "p1", 4)], split_name="pilot_p1"
        )
    conflicting = [trace("a:0", "a", 0), trace("a:0", "a", 0)]
    conflicting[1]["kind"] = "code"
    with pytest.raises(ValueError, match="conflicting"):
        shortfall.assemble_corpus(conflicting, split_name="pilot_tfit")
    # The realized decision was no_extension_original_only: this build refuses
    # the conditional extension instead of grading or assembling it.
    union = [trace("u:0", "u", 0), trace("u:4", "u", 4)]
    with pytest.raises(ValueError, match="extension branch not implemented"):
        shortfall.assemble_corpus(union, split_name="pilot_tfit")
    with pytest.raises(ValueError, match="extension branch not implemented"):
        shortfall.assemble_corpus(
            union, split_name="pilot_tfit", allow_extra=True
        )


def test_final_corpus_uses_every_verified_trace_and_renders(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=_count_out(tmp_path, "count"),
        authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count")),
        grade_fn=_fake_grade,
        workers=1,
    )
    freeze = shortfall.load_freeze_manifest(freeze_path)
    entries = [
        json.loads(line)
        for line in (_count_out(tmp_path, "count") / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    records = {row["request_id"]: row for row in fixture["records"]}
    grades = {row["request_id"]: row for row in fixture["grades"]}
    traces = shortfall.build_eligible_traces(
        freeze=freeze,
        pool_dir=fixture["pool"],
        records=records,
        grade_rows=grades,
        verified_request_ids={
            entry["request_id"] for entry in entries if entry["verified"]
        },
    )
    rendered_calls = []

    def fake_renderer(rows, tokenizer, **kwargs):
        rendered_calls.append((len(rows), kwargs.get("source")))
        return [
            {
                "problem_id": row["problem_id"],
                "sample_index": row.get("sample_index"),
                "row_id": index,
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "label_positions": [1, 2],
            }
            for index, row in enumerate(rows)
        ]

    manifest = shortfall.write_final_corpus(
        tmp_path / "corpus",
        traces,
        tokenizer=object(),
        renderer=fake_renderer,
    )
    assert manifest["counts"]["pilot_tfit"]["distinct_problems"] == 4
    assert manifest["counts"]["pilot_selection"]["distinct_problems"] == 5
    assert manifest["counts"]["pilot_p1"]["distinct_problems"] == 1
    assert len(rendered_calls) == 3
    assert (tmp_path / "corpus" / "pilot_tfit.rendered.jsonl").is_file()
    # Every eligible trace is kept, not just one per problem.
    tfit_rows = (
        tmp_path / "corpus" / "pilot_tfit.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(tfit_rows) == len(traces["pilot_tfit"])


# ---------------------------------------------------------------------------
# Freeze index / execution guard
# ---------------------------------------------------------------------------
def _write_synthetic_count(
    index_dir: Path, *, status: str = shortfall.COUNT_STATUS_COMPLETE
) -> dict[str, Path]:
    """Write a minimal count whose manifest/summary/ledger loaders accept.

    The rows are intentionally non-code unless a terminal-infrastructure count
    is requested, because this fixture exists only to exercise readiness'
    binding and verification gates, not the grader.
    """

    count_dir = index_dir / "count"
    count_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = count_dir / shortfall.ATTEMPT_LEDGER_NAME
    shortfall._ensure_attempt_ledger(ledger_path, {})
    terminal = status == shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE
    if terminal:
        inventory = {
            "n_tests": 1,
            "tests_sha256": shortfall._sha256_text("synthetic frozen test"),
        }
        attempts = [
            {
                "attempt": number,
                "status": "infrastructure_error",
                "cleanup_verified": True,
                "lifecycle_class": "docker_enforcement_unknown",
                "lifecycle_valid": False,
            }
            for number in (1, 2)
        ]
        row = {
            "request_id": "tfit:c0:0",
            "problem_id": "tfit:c0",
            "split": "pilot_tfit",
            "domain": "code",
            "kept": True,
            "verified": False,
            "tests_run": 1,
            "failure_class": "infrastructure",
            "infrastructure_failure": True,
            "terminal_infrastructure": True,
            "blocking_infrastructure": False,
            "coverage_error": None,
            "test_inventory": inventory,
            "per_test": [
                {
                    "index": 0,
                    "status": "infrastructure_error",
                    "infrastructure_failure": True,
                    "terminal_infrastructure": True,
                    "terminal_status": (
                        shortfall.TEST_STATUS_TERMINAL_INFRASTRUCTURE
                    ),
                    "terminal_reason": "unknown after the single retry",
                    "cleanup_verified": True,
                    "timed_out": False,
                    "lifecycle_class": "docker_enforcement_unknown",
                    "lifecycle_valid": False,
                    "attempts": attempts,
                }
            ],
        }
        accounting = {
            "requests": 1,
            "kept": 1,
            "code_requests": 1,
            "code_requests_kept": 1,
            "code_requests_graded": 0,
            "code_requests_passed": 0,
            "errors": 0,
            "infrastructure_failures": 1,
            "blocking_infrastructure_failures": 0,
            "terminal_infrastructure_failures": 1,
            "infrastructure_tests": 1,
            "terminal_infrastructure_tests": 1,
            "timeouts": 0,
            "answer_failures": 0,
            "per_test_rows": 1,
            "interpretation_excluded_requests": 0,
            "interpretation_excluded_problems": 0,
        }
        terminal_ids = {
            "count": 1,
            "request_count": 1,
            "tests": [{"request_id": "tfit:c0:0", "test_index": 0}],
        }
    else:
        row = {
            "request_id": "tfit:m0:0",
            "problem_id": "tfit:m0",
            "split": "pilot_tfit",
            "domain": "math",
            "kept": True,
            "verified": True,
            "failure_class": "verified",
            "infrastructure_failure": False,
            "per_test": [],
        }
        accounting = {
            "requests": 1,
            "kept": 1,
            "code_requests": 0,
            "code_requests_kept": 0,
            "code_requests_graded": 0,
            "code_requests_passed": 0,
            "errors": 0,
            "infrastructure_failures": 0,
            "infrastructure_tests": 0,
            "timeouts": 0,
            "answer_failures": 0,
            "per_test_rows": 0,
            "interpretation_excluded_requests": 0,
            "interpretation_excluded_problems": 0,
        }
        terminal_ids = None
    per_request_path = count_dir / shortfall.COUNT_PER_REQUEST_NAME
    per_request_text = json.dumps(row, sort_keys=True) + "\n"
    per_request_path.write_text(per_request_text, encoding="utf-8")
    summary: dict = {
        "kind": "pilot_shortfall_official_count",
        "version": "3r",
        "run_version": "v3r",
        "status": status,
        "accounting": accounting,
        "expected_requests": {
            "count": 1,
            "code_request_count": 1 if terminal else 0,
            "request_ids_sha256": shortfall._sha256_text(row["request_id"]),
        },
        "per_split": {},
        "infrastructure_failures": accounting["infrastructure_failures"],
        "extension_required": False,
        "extension_decision": "no_extension_original_only",
        "artifacts": {
            shortfall.ATTEMPT_LEDGER_NAME: {
                "sha256": shortfall._sha256_file(ledger_path)
            }
        },
        "attempt_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": shortfall._sha256_file(ledger_path),
        },
    }
    if terminal_ids is not None:
        summary["terminal_infrastructure"] = terminal_ids
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    manifest = {
        "kind": "pilot_shortfall_count_manifest",
        "version": "3r",
        "run_version": "v3r",
        "status": status,
        "files": {
            shortfall.COUNT_SUMMARY_NAME: {
                "path": str(summary_path.resolve()),
                "sha256": shortfall._sha256_file(summary_path),
                "bytes": summary_path.stat().st_size,
            },
            shortfall.COUNT_PER_REQUEST_NAME: {
                "path": str(per_request_path.resolve()),
                "sha256": shortfall._sha256_file(per_request_path),
                "bytes": per_request_path.stat().st_size,
            },
            shortfall.ATTEMPT_LEDGER_NAME: {
                "path": str(ledger_path.resolve()),
                "sha256": shortfall._sha256_file(ledger_path),
                "bytes": ledger_path.stat().st_size,
            },
        },
    }
    manifest["manifest_sha256"] = shortfall._hash_payload(manifest, "manifest_sha256")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return {
        "dir": count_dir,
        "summary": summary_path,
        "manifest": manifest_path,
        "ledger": ledger_path,
    }


def _write_index(
    tmp_path: Path,
    *,
    tfit: int,
    selection: int,
    p1: int = 0,
    name: str = "index.json",
    status: str = shortfall.INDEX_STATUS_COMPLETE,
    pending: list[dict] | None = None,
    amendment: bool = True,
    final_corpus: bool = True,
    ext_expected: bool = False,
    count_status: str | None = None,
    terminal_infrastructure: dict | None = None,
) -> dict:
    index_dir = tmp_path / f"index-{name}"
    index_dir.mkdir(parents=True, exist_ok=True)
    bindings: list[dict] = []

    def bind(label: str, path: Path) -> None:
        bindings.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    if amendment:
        amendment_path = index_dir / "pilot-shortfall-amendment.md"
        amendment_path.write_text("amendment\n", encoding="utf-8")
        bind("pilot_shortfall_amendment", amendment_path)

    if count_status == shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE:
        synthetic_status = shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE
    else:
        synthetic_status = shortfall.COUNT_STATUS_COMPLETE
    count = _write_synthetic_count(index_dir, status=synthetic_status)
    bind("count_summary", count["summary"])
    bind("count_manifest", count["manifest"])

    rendered_paths: dict[str, Path] = {}
    for split, count in (
        ("pilot_tfit", tfit),
        ("pilot_selection", selection),
        ("pilot_p1", p1),
    ):
        path = index_dir / f"{split}.rendered.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "problem_id": f"{split}-p{index}",
                        "input_ids": [1, 2, 3],
                        "labels": [-100, 2, 3],
                    }
                )
                + "\n"
                for index in range(count)
            ),
            encoding="utf-8",
        )
        rendered_paths[split] = path
        if split != "pilot_p1":
            bind(f"final_corpus/{split}.rendered.jsonl", path)

    final_files: dict = {}
    if final_corpus:
        manifest_path = index_dir / "manifest.json"
        manifest = {
            "kind": "pilot_shortfall_final_corpus",
            "allow_extra": False,
            "splits": {
                split: {
                    "distinct_problems": count,
                    "request_ids": [],
                    "rendered": {
                        "file": f"{split}.rendered.jsonl",
                        "sha256": hashlib.sha256(
                            rendered_paths[split].read_bytes()
                        ).hexdigest(),
                    },
                }
                for split, count in (
                    ("pilot_tfit", tfit),
                    ("pilot_selection", selection),
                    ("pilot_p1", p1),
                )
            },
            "files": {
                f"{split}.rendered.jsonl": {
                    "path": str(rendered_paths[split].resolve())
                }
                for split in (
                    "pilot_tfit",
                    "pilot_selection",
                    "pilot_p1",
                )
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        bind("final_corpus_manifest", manifest_path)
        final_files = {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "splits": manifest["splits"],
        }

    payload = {
        "kind": "pilot_shortfall_freeze_index",
        "run_version": "v3r",
        "status": status,
        "final_counts": {
            "pilot_p1": p1,
            "pilot_tfit": tfit,
            "pilot_selection": selection,
        },
        "decision": {
            "extension_required": ext_expected,
            "extension_decision": (
                "extension_required"
                if ext_expected
                else "no_extension_original_only"
            ),
            "realized_decision": "no_extension_original_only",
        },
        "bindings": bindings,
        "pending_bindings": list(pending or []),
        "final_corpus": final_files,
        "index_sha256": "",
    }
    payload["count_status"] = (
        count_status if count_status is not None else synthetic_status
    )
    if terminal_infrastructure is not None:
        payload["terminal_infrastructure"] = terminal_infrastructure
    payload["index_sha256"] = shortfall._hash_payload(payload, "index_sha256")
    path = index_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": path,
        "rendered": rendered_paths,
        "payload": payload,
        "index_dir": index_dir,
    }


def test_guard_refuses_missing_hash_mismatch_and_below_minimums(tmp_path: Path):
    with pytest.raises(shortfall.ShortfallReadinessError, match="missing"):
        shortfall.check_freeze_readiness(tmp_path / "missing.json")

    # A skeleton/partial index is rejected before its hashes are trusted.
    skeleton = tmp_path / "skeleton.json"
    skeleton.write_text(
        json.dumps(
            {
                "kind": "prereg_pilot_shortfall_freeze",
                "status": "pending-data-addendum",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(shortfall.ShortfallReadinessError, match="pending"):
        shortfall.check_freeze_readiness(skeleton)

    index = _write_index(tmp_path, tfit=20, selection=30)
    shortfall.check_freeze_readiness(index["path"])

    # P1 never gates: zero P1 is allowed when P2 minimums hold.
    assert shortfall.check_freeze_readiness(
        _write_index(tmp_path, tfit=20, selection=30, p1=0, name="p1zero.json")[
            "path"
        ]
    )

    # An unresolved required binding (the amendment) always refuses.
    with pytest.raises(shortfall.ShortfallReadinessError, match="pending"):
        shortfall.check_freeze_readiness(
            _write_index(
                tmp_path,
                tfit=20,
                selection=30,
                name="pending.json",
                status=shortfall.INDEX_STATUS_PENDING,
                pending=[{"label": "pilot_shortfall_amendment", "path": "x"}],
            )["path"]
        )
    with pytest.raises(shortfall.ShortfallReadinessError, match="pending_bindings|required"):
        shortfall.check_freeze_readiness(
            _write_index(
                tmp_path,
                tfit=20,
                selection=30,
                name="unresolved.json",
                pending=[{"label": "pilot_shortfall_amendment", "path": "x"}],
            )["path"]
        )
    with pytest.raises(shortfall.ShortfallReadinessError, match="amendment"):
        shortfall.check_freeze_readiness(
            _write_index(
                tmp_path, tfit=20, selection=30, name="noamend.json", amendment=False
            )["path"]
        )
    with pytest.raises(shortfall.ShortfallReadinessError, match="final corpus"):
        shortfall.check_freeze_readiness(
            _write_index(
                tmp_path,
                tfit=20,
                selection=30,
                name="nocorpus.json",
                final_corpus=False,
            )["path"]
        )

    with pytest.raises(shortfall.ShortfallReadinessError, match="T-fit"):
        shortfall.check_freeze_readiness(
            _write_index(tmp_path, tfit=19, selection=30, name="t19.json")["path"]
        )
    with pytest.raises(shortfall.ShortfallReadinessError, match="selection"):
        shortfall.check_freeze_readiness(
            _write_index(tmp_path, tfit=20, selection=29, name="s29.json")["path"]
        )

    index_path = index["path"]
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["sha256"] = "0" * 64
    payload["index_sha256"] = shortfall._hash_payload(payload, "index_sha256")
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(shortfall.ShortfallReadinessError, match="changed"):
        shortfall.check_freeze_readiness(index_path)


def test_pilot_inputs_must_be_byte_identical_and_recounted(tmp_path: Path):
    index = _write_index(tmp_path, tfit=20, selection=30, name="inputs.json")
    payload = index["payload"]
    fit = index["rendered"]["pilot_tfit"]
    test = index["rendered"]["pilot_selection"]

    assert shortfall.check_pilot_inputs(
        payload, fit_data=fit, test_data=test
    ) == {"pilot_tfit": 20, "pilot_selection": 30}

    substituted = tmp_path / "substituted.jsonl"
    text = fit.read_text(encoding="utf-8")
    substituted.write_text(
        text.replace('"input_ids": [1, 2, 3]', '"input_ids": [9, 9]'),
        encoding="utf-8",
    )
    with pytest.raises(shortfall.ShortfallReadinessError, match="byte-identical"):
        shortfall.check_pilot_inputs(
            payload, fit_data=substituted, test_data=test
        )

    # A missing final-corpus section is refused even when the callers pass files.
    without = dict(payload)
    without["final_corpus"] = {}
    with pytest.raises(shortfall.ShortfallReadinessError, match="final corpus"):
        shortfall.check_pilot_inputs(without, fit_data=fit, test_data=test)

    # The count is recomputed from the actual files, not trusted from the index.
    under = _write_index(tmp_path, tfit=20, selection=30, name="under.json")
    under_payload = dict(under["payload"])
    under_payload["final_counts"] = {"pilot_tfit": 19, "pilot_selection": 30}
    with pytest.raises(shortfall.ShortfallReadinessError, match="distinct problems"):
        shortfall.check_pilot_inputs(
            under_payload,
            fit_data=under["rendered"]["pilot_tfit"],
            test_data=under["rendered"]["pilot_selection"],
        )


def _run_pilot_main(
    tmp_path: Path,
    monkeypatch,
    index_path: Path,
    *,
    fit_data: Path,
    test_data: Path,
    require_tag: str | None = None,
    frozen_ok: bool = False,
) -> int:
    import scripts.run_pilot as run_pilot

    def boom(*args, **kwargs):
        raise AssertionError("no precompute or training may start")

    monkeypatch.setattr(run_pilot, "run_stage", boom)
    if frozen_ok:
        monkeypatch.setattr(run_pilot, "check_frozen_run", lambda *a, **k: None)
        monkeypatch.setattr(run_pilot, "assert_code_under", lambda *a, **k: None)
    else:
        monkeypatch.setattr(run_pilot, "check_frozen_run", boom)
        monkeypatch.setattr(run_pilot, "assert_code_under", boom)

    argv = [
        "--prereg", str(ROOT / "prereg" / "prereg.yaml"),
        "--heal-data", str(tmp_path / "heal"),
        "--fit-data", str(fit_data),
        "--test-data", str(test_data),
        "--shortfall-freeze", str(index_path),
        "--out", str(tmp_path / "pilot-out"),
    ]
    if require_tag:
        argv.extend(["--require-tag", require_tag])
    return run_pilot.main(argv)


def test_run_pilot_real_readiness_refuses_pending_amendment_with_tag(
    tmp_path: Path, monkeypatch, capsys
):
    """Through run_pilot.main with the real check_freeze_readiness (no mock)."""

    index = _write_index(
        tmp_path,
        tfit=27,
        selection=40,
        name="pending-tag.json",
        status=shortfall.INDEX_STATUS_PENDING,
        pending=[{"label": "pilot_shortfall_amendment", "path": "prereg/x.md"}],
    )
    fit = index["rendered"]["pilot_tfit"]
    test = index["rendered"]["pilot_selection"]
    code = _run_pilot_main(
        tmp_path,
        monkeypatch,
        index["path"],
        fit_data=fit,
        test_data=test,
        require_tag="prereg-v1",
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "pilot-shortfall readiness refused" in err
    assert "pending" in err or "amendment" in err
    assert not (tmp_path / "pilot-out").exists()


def test_run_pilot_real_readiness_refuses_amendment_skeleton(
    tmp_path: Path, monkeypatch, capsys
):
    import scripts.run_pilot as run_pilot

    skeleton = tmp_path / "prereg-skeleton.json"
    skeleton.write_text(
        json.dumps(
            {
                "kind": "prereg_pilot_shortfall_freeze",
                "status": "pending-data-addendum",
            }
        ),
        encoding="utf-8",
    )
    def boom(*args, **kwargs):
        raise AssertionError("no frozen-run check may start")

    monkeypatch.setattr(run_pilot, "check_frozen_run", boom)
    monkeypatch.setattr(run_pilot, "assert_code_under", boom)
    code = run_pilot.main(
        [
            "--prereg", str(ROOT / "prereg" / "prereg.yaml"),
            "--heal-data", str(tmp_path / "heal"),
            "--fit-data", str(tmp_path / "fit"),
            "--test-data", str(tmp_path / "test"),
            "--shortfall-freeze", str(skeleton),
            "--require-tag", "prereg-v1",
            "--out", str(tmp_path / "pilot-out"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "pending" in err
    assert not (tmp_path / "pilot-out").exists()


def test_run_pilot_refuses_substituted_and_missing_final_corpus(
    tmp_path: Path, monkeypatch, capsys
):
    index = _write_index(tmp_path, tfit=27, selection=40, name="inputs.json")
    fit = index["rendered"]["pilot_tfit"]
    test = index["rendered"]["pilot_selection"]
    substituted = tmp_path / "sub-fit.jsonl"
    substituted.write_text(
        fit.read_text(encoding="utf-8").replace(
            '"input_ids": [1, 2, 3]', '"input_ids": [4, 5, 6]'
        ),
        encoding="utf-8",
    )
    code = _run_pilot_main(
        tmp_path,
        monkeypatch,
        index["path"],
        fit_data=substituted,
        test_data=test,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "byte-identical" in err
    assert not (tmp_path / "pilot-out").exists()

    missing = _write_index(
        tmp_path,
        tfit=27,
        selection=40,
        name="missing-corpus.json",
        final_corpus=False,
    )
    code = _run_pilot_main(
        tmp_path,
        monkeypatch,
        missing["path"],
        fit_data=missing["rendered"]["pilot_tfit"],
        test_data=missing["rendered"]["pilot_selection"],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "final corpus" in err
    assert not (tmp_path / "pilot-out").exists()


def test_run_pilot_accepts_matching_inputs_then_stops_at_monitor(
    tmp_path: Path, monkeypatch, capsys
):
    index = _write_index(tmp_path, tfit=27, selection=40, name="ok.json")
    code = _run_pilot_main(
        tmp_path,
        monkeypatch,
        index["path"],
        fit_data=index["rendered"]["pilot_tfit"],
        test_data=index["rendered"]["pilot_selection"],
        frozen_ok=True,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "MONITOR" in err or "monitor" in err
    assert "readiness refused" not in err
    assert "inputs refused" not in err


# ---------------------------------------------------------------------------
# Modal top-up mode (fakes only; never contacts Modal)
# ---------------------------------------------------------------------------
def _load_modal_script():
    spec = importlib.util.spec_from_file_location(
        "modal_stage0_topup_under_test", ROOT / "scripts" / "modal_stage0.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeEntry:
    def __init__(self, path: str, size: int | None = None):
        self.path = path
        self.size = size


class FakeVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commits = 0

    def commit(self):
        self.commits += 1

    def listdir(self, path: str, recursive: bool = False):
        prefix = str(path).strip("/") + "/"
        return [
            FakeEntry(name, len(data))
            for name, data in self.files.items()
            if name.startswith(prefix) and (recursive or "/" not in name[len(prefix):])
        ]

    def read_file(self, path: str):
        data = self.files[str(path).lstrip("/")]
        return iter([data[: len(data) // 2], data[len(data) // 2 :]])


class FakeHeartbeat(dict):
    pass


class FakeDictContext:
    def __init__(self, heartbeat):
        self.heartbeat = heartbeat

    def __enter__(self):
        return self.heartbeat

    def __exit__(self, *_exc):
        return False


class FakeDictManager:
    def ephemeral(self):
        return FakeDictContext(FakeHeartbeat())


def _fake_modal(volume: FakeVolume):
    class VolumeClass:
        @classmethod
        def from_name(cls, name, create_if_missing=False):
            return volume

    return SimpleNamespace(Volume=VolumeClass, Dict=FakeDictManager())


def _heartbeat(payload, container_id="container-1", sequence=1):
    heartbeat = payload.get("_heartbeat")
    key = payload.get("heartbeat_key")
    if heartbeat is not None and key is not None:
        heartbeat[key] = {
            "container_id": container_id,
            "sequence": sequence,
            "phase": "entered",
        }


class FakeCall:
    def __init__(self, result, owner=None):
        self.result = result
        self.owner = owner

    def get(self, timeout=None):
        assert timeout is not None
        if self.owner is not None:
            self.owner.active = max(0, self.owner.active - 1)
        return self.result

    def cancel(self, terminate_containers=False):
        return None


class FakeTopupFunction:
    def __init__(self, volume: FakeVolume):
        self.volume = volume
        self.calls = 0
        self.payloads: list[dict] = []
        self.sent_ids: list[str] = []
        self.options: dict = {}
        self.active = 0

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def get_current_stats(self):
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)

    def spawn(self, payload):
        self.calls += 1
        self.active += 1
        _heartbeat(payload)
        self.payloads.append(payload)
        results = []
        for shard in payload["shards"]:
            rows = []
            for request in shard["requests"]:
                self.sent_ids.append(request["request_id"])
                content = (
                    "```python\nPASS\n```"
                    if request.get("kind") == "code"
                    else "The answer is 1."
                )
                rows.append(
                    {
                        "problem_id": request["problem_id"],
                        "sample_index": request["sample_index"],
                        "request_id": request["request_id"],
                        "prompt": request["prompt"],
                        "completion": "step " * 20 + content,
                        "reasoning": "step " * 20,
                        "content": content,
                        "finish_reason": "stop",
                        "thought_closed": True,
                        "tokens": 80,
                        "prompt_tokens": request["prompt_tokens"],
                        "seed": request["seed"],
                        "shard_index": shard["shard_index"],
                    }
                )
            data = ("".join(json.dumps(row) + "\n" for row in rows)).encode("utf-8")
            name = f"{payload['run_dir_rel']}/shard_{shard['shard_index']:05d}.jsonl"
            self.volume.files[name] = data
            self.volume.files[name + ".committed"] = b"committed\n"
            results.append(
                {
                    "shard": shard["shard_index"],
                    "committed": True,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        return FakeCall({"shards": results}, owner=self)


class FakePreflight:
    def spawn(self, payload):
        _heartbeat(payload)
        return FakeCall(
            {
                "ok": True,
                "versions": {},
                "heartbeat": {
                    "container_id": "container-1",
                    "delivered": 2,
                    "background_delivered": 1,
                    "failed_writes": 0,
                },
            }
        )

    def get_current_stats(self):
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)


class FakeLedger:
    def __init__(self, *_args, **_kwargs):
        import tempfile

        path = _kwargs.get("path")
        if path is None and _args:
            path = _args[0]
        if path is None:
            path = Path(tempfile.mkdtemp()) / "ledger.json"
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{}", encoding="utf-8")
        self._entries: list[dict] = []
        self._next = 0

    def entries(self):
        return list(self._entries)

    def cumulative_usd(self, month, provider=None):
        return sum(float(entry.get("usd") or 0.0) for entry in self._entries)

    def reserve(self, **kwargs):
        self._next += 1
        reservation_id = f"r{self._next}"
        self._entries.append(
            {
                "kind": "reservation",
                "id": reservation_id,
                "usd": kwargs.get("usd", 0.0),
                "month": kwargs.get("month"),
                "provider": kwargs.get("provider"),
                "metadata": dict(kwargs.get("metadata") or {}),
            }
        )
        return reservation_id

    def settle(self, reservation_id, **kwargs):
        self._entries.append(
            {
                "kind": "actual",
                "reservation_id": reservation_id,
                "usd": kwargs.get("actual_usd", 0.0),
                "month": kwargs.get("month"),
                "provider": kwargs.get("provider"),
                "metadata": dict(kwargs.get("metadata") or {}),
            }
        )


def _set_count_decision(count_dir: Path, required: bool) -> None:
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["extension_required"] = bool(required)
    summary["extension_decision"] = (
        "extension_required" if required else "no_extension_original_only"
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][shortfall.COUNT_SUMMARY_NAME]
    entry["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    entry["bytes"] = summary_path.stat().st_size
    manifest["manifest_sha256"] = shortfall._hash_payload(manifest, "manifest_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _official_count(
    tmp_path: Path,
    fixture: dict,
    freeze_path: Path,
    plan_path: Path,
    *,
    required: bool = True,
) -> Path:
    out = _count_out(tmp_path, f"count-{'yes' if required else 'no'}")
    shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=out,
        grade_fn=_fake_grade,
        workers=1,
        authorization=_test_authorization(
            tmp_path, fixture["lite"], out_dir=out
        ),
    )
    if not required:
        _set_count_decision(out, False)
    return out


def _modal_topup_call(
    script,
    tmp_path,
    fixture,
    plan_path,
    fake_fn,
    fake_modal,
    *,
    topup_count=None,
    topup_freeze=None,
    **overrides,
):
    kwargs = dict(
        mode="sample-lite-topup",
        topup_plan=plan_path,
        topup_count=topup_count,
        topup_freeze=topup_freeze,
        pool_dir=fixture["pool"],
        output_dir=tmp_path / "topup_out",
        gpu="A100-80GB",
        gpu_id="A100-80GB",
        max_usd=3.0,
        shard_size=100,
        avg_generated_tokens=3200,
        samples_per_prompt=1,
        yes=True,
        fn=fake_fn,
        preflight_fn=FakePreflight(),
        modal_module=fake_modal,
        month_now=FIXED_NOW,
        timeout_s=None,
        require_frozen_pool=False,
        prompt_tokens_for=lambda problem: 10,
        init_ledger=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_modal_topup_runs_exactly_the_plan_in_one_call(tmp_path: Path, monkeypatch):
    script = _load_modal_script()
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    count_dir = _official_count(tmp_path, fixture, freeze_path, plan_path)
    volume = FakeVolume()
    fake_fn = FakeTopupFunction(volume)

    monkeypatch.setattr(
        script,
        "_lite_load_pool",
        lambda *a, **k: (
            fixture["problems"],
            {"manifest_sha256": plan["parent_pool_manifest_sha256"]},
        ),
    )
    monkeypatch.setattr(script, "SpendLedger", FakeLedger)
    monkeypatch.setattr(script, "require_spend_lock", lambda *a, **k: None)

    code = script.run_mode(
        **_modal_topup_call(
            script,
            tmp_path,
            fixture,
            plan_path,
            fake_fn,
            _fake_modal(volume),
            topup_count=count_dir,
            topup_freeze=freeze_path,
        )
    )
    assert code == 0
    assert fake_fn.calls == 1
    assert fake_fn.sent_ids == [item["request_id"] for item in plan["requests"]]
    assert fake_fn.options["timeout"] == plan["timeout_seconds"]
    assert fake_fn.options["max_containers"] == 1
    manifest = json.loads(
        (tmp_path / "topup_out" / "topup_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["parent_run_id"] == plan["parent_run_id"]
    assert manifest["plan_sha256"] == plan["plan_sha256"]
    assert manifest["request_count"] == plan["request_count"]
    assert manifest["calls_completed"] == 1
    assert (tmp_path / "topup_out" / "topup_receipt.json").is_file()


def test_modal_topup_refuses_over_the_run_cap_before_any_call(
    tmp_path: Path, monkeypatch
):
    script = _load_modal_script()
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    count_dir = _official_count(tmp_path, fixture, freeze_path, plan_path)
    volume = FakeVolume()
    fake_fn = FakeTopupFunction(volume)

    monkeypatch.setattr(
        script,
        "_lite_load_pool",
        lambda *a, **k: (
            fixture["problems"],
            {"manifest_sha256": plan["parent_pool_manifest_sha256"]},
        ),
    )
    monkeypatch.setattr(script, "SpendLedger", FakeLedger)
    monkeypatch.setattr(script, "require_spend_lock", lambda *a, **k: None)

    code = script.run_mode(
        **_modal_topup_call(
            script,
            tmp_path,
            fixture,
            plan_path,
            fake_fn,
            _fake_modal(volume),
            topup_count=count_dir,
            topup_freeze=freeze_path,
            max_usd=0.001,
        )
    )
    assert code == 2
    assert fake_fn.calls == 0


def test_modal_topup_refuses_without_true_decision_and_leaves_ledger_untouched(
    tmp_path: Path, monkeypatch, capsys
):
    script = _load_modal_script()
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    false_count = _official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    volume = FakeVolume()
    fake_fn = FakeTopupFunction(volume)

    def no_ledger(*args, **kwargs):
        raise AssertionError("the ledger must not be touched on refusal")

    monkeypatch.setattr(script, "SpendLedger", no_ledger)
    monkeypatch.setattr(script, "require_spend_lock", no_ledger)
    monkeypatch.setattr(
        script,
        "_lite_load_pool",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the pool must not be loaded on refusal")
        ),
    )

    # 1. The decision is false.
    code = script.run_mode(
        **_modal_topup_call(
            script,
            tmp_path,
            fixture,
            plan_path,
            fake_fn,
            _fake_modal(volume),
            topup_count=false_count,
            topup_freeze=freeze_path,
        )
    )
    assert code == 2
    assert "extension_required is false" in capsys.readouterr().err
    assert fake_fn.calls == 0

    # 2. No count at all.
    code = script.run_mode(
        **_modal_topup_call(
            script,
            tmp_path,
            fixture,
            plan_path,
            fake_fn,
            _fake_modal(volume),
            topup_count=None,
            topup_freeze=freeze_path,
        )
    )
    assert code == 2
    assert "--topup-count" in capsys.readouterr().err
    assert fake_fn.calls == 0

    # 3. A count for a different plan.
    other_plan = tmp_path / "other_plan.json"
    other = json.loads(plan_path.read_text(encoding="utf-8"))
    other["created_utc"] = "2026-11-06T00:00:00+00:00"
    other["plan_sha256"] = shortfall._hash_payload(other, "plan_sha256")
    other_plan.write_text(json.dumps(other), encoding="utf-8")
    code = script.run_mode(
        **_modal_topup_call(
            script,
            tmp_path,
            fixture,
            other_plan,
            fake_fn,
            _fake_modal(volume),
            topup_count=_official_count(
                tmp_path, fixture, freeze_path, plan_path
            ),
            topup_freeze=freeze_path,
        )
    )
    assert code == 2
    assert "top-up plan" in capsys.readouterr().err
    assert fake_fn.calls == 0


def test_modal_topup_main_refuses_before_importing_modal(
    tmp_path: Path, monkeypatch, capsys
):
    script = _load_modal_script()
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)

    def no_import():
        raise AssertionError("the Modal client must not be imported on refusal")

    monkeypatch.setattr(script, "_import_modal", no_import)
    code = script.main(
        [
            "--mode", "sample-lite-topup",
            "--topup-plan", str(plan_path),
            "--topup-freeze", str(freeze_path),
            "--yes",
            "--output-dir", str(tmp_path / "topup_out"),
        ]
    )
    assert code == 2
    assert "--topup-count" in capsys.readouterr().err
    assert not (tmp_path / "topup_out").exists()


def _fake_renderer(rows, tokenizer, **kwargs):
    return [
        {
            "problem_id": row["problem_id"],
            "sample_index": row.get("sample_index"),
            "row_id": index,
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "label_positions": [1, 2],
        }
        for index, row in enumerate(rows)
    ]


def _final_corpus(
    tmp_path: Path,
    fixture: dict,
    freeze_path: Path,
    count_dir: Path,
    *,
    name: str,
    renderer=None,
) -> tuple[Path, dict]:
    freeze = shortfall.load_freeze_manifest(freeze_path)
    entries = [
        json.loads(line)
        for line in (count_dir / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    records = {row["request_id"]: row for row in fixture["records"]}
    grades = {row["request_id"]: row for row in fixture["grades"]}
    traces = shortfall.build_eligible_traces(
        freeze=freeze,
        pool_dir=Path(freeze["inputs"]["pool_dir"]),
        records=records,
        grade_rows=grades,
        verified_request_ids={
            entry["request_id"] for entry in entries if entry.get("verified")
        },
    )

    out = tmp_path / name
    manifest = shortfall.write_final_corpus(
        out, traces, tokenizer=object(), renderer=renderer or _fake_renderer
    )
    return out / "manifest.json", manifest


def test_extension_branch_is_refused_and_final_counts_are_derived(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    required_count = _official_count(tmp_path, fixture, freeze_path, plan_path)
    manifest_path, manifest = _final_corpus(
        tmp_path, fixture, freeze_path, required_count, name="corpus-ext"
    )

    with pytest.raises(ValueError, match="extension branch not implemented"):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=required_count / shortfall.COUNT_SUMMARY_NAME,
            topup_plan_path=plan_path,
            out_path=tmp_path / "index-ext.json",
            final_manifest_path=manifest_path,
            allow_pending=True,
            tokenizer=object(),
            renderer=_fake_renderer,
        )

    # An original-only count whose final corpus was built with extra traces is
    # refused even if its counts look plausible.
    original_count = _official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    original_manifest_path, original_manifest = _final_corpus(
        tmp_path, fixture, freeze_path, original_count, name="corpus-orig"
    )
    payload = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    payload["allow_extra"] = True
    payload["manifest_sha256"] = shortfall._hash_payload(payload, "manifest_sha256")
    original_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="extension branch not implemented"):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=original_count / shortfall.COUNT_SUMMARY_NAME,
            topup_plan_path=plan_path,
            out_path=tmp_path / "index-extra.json",
            final_manifest_path=original_manifest_path,
            allow_pending=True,
            tokenizer=object(),
            renderer=_fake_renderer,
        )
    payload["allow_extra"] = False
    payload["manifest_sha256"] = shortfall._hash_payload(payload, "manifest_sha256")
    original_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    # The integrated tree ships the registered prereg amendment, so the
    # missing-amendment path is exercised against an explicitly absent path.
    absent_amendment = tmp_path / "absent-pilot-shortfall-amendment.md"

    # Without --allow-pending the missing prereg amendment refuses the index.
    with pytest.raises(FileNotFoundError, match="pilot_shortfall_amendment"):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=original_count / shortfall.COUNT_SUMMARY_NAME,
            topup_plan_path=plan_path,
            out_path=tmp_path / "index-pending.json",
            final_manifest_path=original_manifest_path,
            amendment_path=absent_amendment,
            tokenizer=object(),
            renderer=_fake_renderer,
        )

    # With --allow-pending the index is pending and derives its counts from the
    # final corpus, while the guard rejects it.
    pending = shortfall.write_freeze_index(
        repo_root=ROOT,
        freeze_path=freeze_path,
        count_summary_path=original_count / shortfall.COUNT_SUMMARY_NAME,
        topup_plan_path=plan_path,
        out_path=tmp_path / "index-pending-ok.json",
        final_manifest_path=original_manifest_path,
        amendment_path=absent_amendment,
        allow_pending=True,
        tokenizer=object(),
        renderer=_fake_renderer,
    )
    assert pending["status"] == shortfall.INDEX_STATUS_PENDING
    assert pending["decision"]["realized_decision"] == "no_extension_original_only"
    assert pending["final_counts"]["pilot_tfit"] == manifest["counts"][
        "pilot_tfit"
    ]["distinct_problems"]
    assert pending["final_counts"]["pilot_selection"] == manifest["counts"][
        "pilot_selection"
    ]["distinct_problems"]
    with pytest.raises(shortfall.ShortfallReadinessError, match="pending"):
        shortfall.check_freeze_readiness(tmp_path / "index-pending-ok.json")


def test_build_pilot_corpus_refuses_topup_inputs_and_extension(tmp_path: Path):
    import scripts.build_pilot_corpus as builder

    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    required_count = _official_count(tmp_path, fixture, freeze_path, plan_path)
    args = [
        "--repo-root", str(ROOT),
        "--freeze", str(freeze_path),
        "--count-summary", str(required_count / shortfall.COUNT_SUMMARY_NAME),
        "--topup-plan", str(plan_path),
        "--out", str(tmp_path / "corpus-build"),
        "--index-out", str(tmp_path / "index-build.json"),
        "--allow-pending",
    ]
    with pytest.raises(SystemExit, match="extension branch not implemented"):
        builder.main(args)
    with pytest.raises(SystemExit, match="extension branch not implemented"):
        builder.main([*args, "--topup-dir", str(tmp_path / "topup")])


# ---------------------------------------------------------------------------
# Count input/implementation binding, plan equality and per-request accounting
# ---------------------------------------------------------------------------
def _rehash_plan(plan: dict) -> dict:
    plan["plan_sha256"] = shortfall._hash_payload(plan, "plan_sha256")
    return plan


def _rewrite_freeze(payload: dict, path: Path) -> Path:
    payload = json.loads(json.dumps(payload))
    payload["manifest_sha256"] = shortfall._hash_payload(payload, "manifest_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_count_consumes_frozen_paths_and_refuses_a_substitute(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    other_pool = tmp_path / "other-pool"
    other_pool.mkdir()
    with pytest.raises(ValueError, match="differs from the verified freeze"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            pool_dir=other_pool,
            lite_dir=fixture["lite"],
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-sub"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-sub")),
            grade_fn=_fake_grade,
        )


def test_count_refuses_changed_implementation_and_dependency(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))

    changed_code = json.loads(json.dumps(payload))
    first = sorted(changed_code["verifier"]["code_hashes"])[0]
    changed_code["verifier"]["code_hashes"][first] = "0" * 64
    code_freeze = _rewrite_freeze(changed_code, tmp_path / "verifier" / "code.json")
    with pytest.raises(ValueError, match="implementation changed"):
        shortfall.count_verified(
            freeze_path=code_freeze,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-code"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-code")),
            grade_fn=_fake_grade,
        )

    changed_deps = json.loads(json.dumps(payload))
    changed_deps["verifier"]["dependencies"]["transformers"] = "0.0.0-changed"
    dep_freeze = _rewrite_freeze(changed_deps, tmp_path / "verifier" / "deps.json")
    with pytest.raises(ValueError, match="dependency version changed"):
        shortfall.count_verified(
            freeze_path=dep_freeze,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-deps"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-deps")),
            grade_fn=_fake_grade,
        )


def test_plan_must_equal_the_frozen_request_map(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shortfall.validate_topup_plan(plan, freeze=freeze)

    # Omission: a self-consistent plan that drops one frozen problem.
    dropped = json.loads(json.dumps(plan))
    victim = dropped["requests"][0]["problem_id"]
    dropped["requests"] = [
        request
        for request in dropped["requests"]
        if request["problem_id"] != victim
    ]
    dropped["request_count"] = len(dropped["requests"])
    dropped["problem_count"] = len(
        {request["problem_id"] for request in dropped["requests"]}
    )
    dropped["timeout_seconds"] = shortfall.topup_timeout_seconds(
        dropped["request_count"]
    )
    _rehash_plan(dropped)
    with pytest.raises(ValueError, match="does not match the request map"):
        shortfall.validate_topup_plan(dropped, freeze=freeze)
    omitted_plan = tmp_path / "omitted-plan.json"
    omitted_plan.write_text(json.dumps(dropped), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the request map"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=omitted_plan,
            out_dir=_count_out(tmp_path, "count-omitted"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-omitted")),
            grade_fn=_fake_grade,
        )

    # Uneven indices: every problem must carry exactly 4, 5, 6 and 7.
    uneven = json.loads(json.dumps(plan))
    first_problem = uneven["requests"][0]["problem_id"]
    uneven["requests"] = [
        request
        for request in uneven["requests"]
        if not (
            request["problem_id"] == first_problem
            and request["sample_index"] == 7
        )
    ]
    uneven["request_count"] = len(uneven["requests"])
    uneven["problem_count"] = len(
        {request["problem_id"] for request in uneven["requests"]}
    )
    uneven["timeout_seconds"] = shortfall.topup_timeout_seconds(
        uneven["request_count"]
    )
    _rehash_plan(uneven)
    with pytest.raises(ValueError, match="indices must be exactly"):
        shortfall.validate_topup_plan(uneven, freeze=freeze)

    # Wrong seed (the plan still self-validates once rehashed).
    wrong_seed = json.loads(json.dumps(plan))
    wrong_seed["requests"][0]["seed"] = int(wrong_seed["requests"][0]["seed"]) + 1
    _rehash_plan(wrong_seed)
    with pytest.raises(ValueError, match="seed does not match"):
        shortfall.validate_topup_plan(wrong_seed, freeze=freeze)


def test_per_request_accounting_keeps_test_details_and_infrastructure(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=_fake_grade,
        workers=1,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    assert {"status", "exit_code", "timed_out", "stderr", "infrastructure_failure"} <= set(
        graded[0]["per_test"][0]
    )
    passed = next(
        entry for entry in graded if entry["failure_class"] == "verified"
    )
    assert passed["per_test"][0]["exit_code"] == 0
    failed = next(
        entry
        for entry in entries
        if entry["domain"] == "code"
        and entry["kept"]
        and entry["failure_class"] == "answer_failure"
    )
    assert failed["per_test"][0]["exit_code"] == 0
    assert failed["infrastructure_failure"] is False

    plan_path = _plan(fixture, tmp_path, freeze_path)

    def infra_grade(code, tests, timeout, **kwargs):
        return SimpleNamespace(
            to_dict=lambda: {
                "passed": False,
                "sandboxed": True,
                "n_tests": len(tests),
                "error": None,
                "duration_s": 0.1,
                "timed_out": False,
                "tests": [
                    {
                        "index": index,
                        "passed": False,
                        "input": "",
                        "expected": "",
                        "stdout": "",
                        "stderr": "Cannot connect to the Docker daemon " + "x" * 2000,
                        "error": "docker exit code 125",
                        "timed_out": False,
                        "duration_s": 0.1,
                        "exit_code": 125,
                        "cleanup_verified": True,
                    }
                    for index in range(len(tests))
                ],
            }
        )

    with pytest.raises(shortfall.ShortfallCountError, match="infrastructure"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-infra"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-infra")),
            grade_fn=infra_grade,
            workers=1,
        )
    rows = [
        json.loads(line)
        for line in (_count_out(tmp_path, "count-infra") / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    infra = [row for row in rows if row["infrastructure_failure"]]
    assert infra
    assert infra[0]["per_test"][0]["status"] == "infrastructure_error"
    assert "truncated" in infra[0]["per_test"][0]["stderr"]
    summary = json.loads(
        (_count_out(tmp_path, "count-infra") / shortfall.COUNT_SUMMARY_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == shortfall.COUNT_STATUS_BLOCKED_INFRASTRUCTURE
    assert summary["infrastructure_failures"] > 0
    with pytest.raises(ValueError, match="infrastructure|status"):
        shortfall.load_verified_count(_count_out(tmp_path, "count-infra"))


def test_grader_exception_is_recorded_as_infrastructure_and_blocks(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)

    def exploding_grade(code, tests, timeout, **kwargs):
        raise RuntimeError("Cannot connect to the Docker daemon")

    with pytest.raises(shortfall.ShortfallCountError, match="infrastructure"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-explode"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-explode")),
            grade_fn=exploding_grade,
            workers=1,
        )
    rows = [
        json.loads(line)
        for line in (_count_out(tmp_path, "count-explode") / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    infra = [row for row in rows if row["infrastructure_failure"]]
    assert infra
    assert infra[0]["failure_class"] == "infrastructure"
    assert "grader failure" in (infra[0]["error"] or "")




# ---------------------------------------------------------------------------
# Fix round 2: lifecycle, retries, legacy refusal, executing binding, v3 pipeline
# ---------------------------------------------------------------------------
def test_lifecycle_classes_control_timeout_attribution():
    from gemmaloop.stage0 import grader

    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": True,
            "lifecycle_class": grader.LIFECYCLE_CANDIDATE_RUNNING,
            "lifecycle_valid": True,
            "cleanup_verified": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": True,
            "exit_observed": True,
        }
    ) == (False, "timeout")
    for lifecycle in (
        grader.LIFECYCLE_STARTUP_FAILURE,
        grader.LIFECYCLE_COMPLETED_BEFORE,
        grader.LIFECYCLE_UNRESOLVED,
        None,
    ):
        assert shortfall._classify_test(
            {"passed": False, "timed_out": True, "lifecycle_class": lifecycle}
        ) == (True, "infrastructure_error")
    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": True,
            "lifecycle_class": grader.LIFECYCLE_LOCAL_TIMEOUT,
            "lifecycle_valid": True,
            "cleanup_verified": True,
        }
    ) == (False, "timeout")


def _result(payload_tests, *, passed, sandboxed=True, error=None):
    return SimpleNamespace(
        to_dict=lambda: {
            "passed": passed,
            "sandboxed": sandboxed,
            "n_tests": len(payload_tests),
            "error": error,
            "duration_s": 0.01,
            "timed_out": any(test.get("timed_out") for test in payload_tests),
            "tests": payload_tests,
        }
    )


def _infra_tests(tests):
    return [
        {
            "index": index,
            "passed": False,
            "input": "",
            "expected": "",
            "stdout": "",
            "stderr": "Cannot connect to the Docker daemon",
            "error": "docker exit code 125",
            "timed_out": False,
            "duration_s": 0.01,
            "exit_code": 125,
            "cleanup_verified": True,
        }
        for index in range(len(tests))
    ]


def _recovery_ok():
    return {
        "check": "docker daemon server version",
        "checked_utc": "2026-11-06T12:00:00+00:00",
        "ok": True,
        "detail": "test recovery",
    }


def test_infrastructure_retry_is_at_most_once_and_first_conclusive_controls(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        test_key = (code, id(tests[0]))
        count = calls.get(test_key, 0) + 1
        calls[test_key] = count
        if count == 1:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    for entry in graded:
        for test in entry["per_test"]:
            assert len(test["attempts"]) == 2
            assert test["attempts"][0]["status"] == "infrastructure_error"
            assert test["attempts"][1]["status"] in {
                "passed",
                "wrong_answer",
                "candidate_error",
            }
    assert all(
        len(test["attempts"]) <= 2
        for entry in graded
        for test in entry["per_test"]
    )


def test_retry_stays_infrastructure_and_blocks_without_a_third_attempt(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        test_key = (code, id(tests[0]))
        calls[test_key] = calls.get(test_key, 0) + 1
        return _result(_infra_tests(tests), passed=False)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    for entry in graded:
        assert entry["failure_class"] == "infrastructure"
        assert entry["infrastructure_failure"] is True
        for test in entry["per_test"]:
            assert len(test["attempts"]) == 2
            assert test["status"] == "infrastructure_error"
    # One full attempt plus one retry per affected test, never more.
    assert max(calls.values()) == 2


def test_conclusive_pass_and_candidate_failure_are_not_retried(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        calls[code] = calls.get(code, 0) + 1
        return _fake_grade(code, tests, timeout)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    # One dispatch per test (the ledger streams per test), and no retry.
    assert max(calls.values()) == 2
    for entry in graded:
        for test in entry["per_test"]:
            assert len(test["attempts"]) == 1


def test_legacy_count_without_status_or_accounting_is_refused(tmp_path: Path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    summary = legacy / shortfall.COUNT_SUMMARY_NAME
    summary.write_text(
        json.dumps({"kind": "pilot_shortfall_official_count"}),
        encoding="utf-8",
    )
    manifest = {
        "kind": "pilot_shortfall_count_manifest",
        "files": {
            shortfall.COUNT_SUMMARY_NAME: shortfall._file_entry(
                shortfall.COUNT_SUMMARY_NAME, summary
            )
        },
    }
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    (legacy / shortfall.COUNT_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="status"):
        shortfall.load_verified_count(legacy)


def test_v2_style_count_without_expected_coverage_is_refused(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    count_dir = _official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("expected_requests")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][shortfall.COUNT_SUMMARY_NAME] = shortfall._file_entry(
        shortfall.COUNT_SUMMARY_NAME, summary_path
    )
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_requests|legacy"):
        shortfall.load_verified_count(count_dir)


def test_second_checkout_and_changed_shortfall_are_refused(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    actual = shortfall._module_files()

    checkout = tmp_path / "second-checkout"
    copied: dict[str, Path] = {}
    for name, path in actual.items():
        relative = path.relative_to(ROOT)
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[name] = target
    with pytest.raises(ValueError, match="path differs|entry point|module set"):
        shortfall.verify_count_implementation(freeze, module_files=copied)

    changed = tmp_path / "changed-shortfall.py"
    changed.write_text(
        actual["gemmaloop.stage0.shortfall"].read_text(encoding="utf-8")
        + "\n# changed\n",
        encoding="utf-8",
    )
    changed_map = dict(actual)
    changed_map["gemmaloop.stage0.shortfall"] = changed
    with pytest.raises(ValueError, match="path differs|changed"):
        shortfall.verify_count_implementation(freeze, module_files=changed_map)


def _rehash_final_manifest(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for _name, entry in (payload.get("files") or {}).items():
        path = Path(entry["path"])
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["bytes"] = path.stat().st_size
    payload["manifest_sha256"] = shortfall._hash_payload(
        payload, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_corpus_substitution_with_unchanged_counts_is_refused(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    count_dir = _official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    manifest_path, _manifest = _final_corpus(
        tmp_path, fixture, freeze_path, count_dir, name="corpus-sub"
    )
    corpus_dir = manifest_path.parent
    raw_path = corpus_dir / "pilot_tfit.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows[0]["messages"][1]["content"] = "```python\nSUBSTITUTED\n```"
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    payload = _rehash_final_manifest(manifest_path)
    payload["splits"]["pilot_tfit"]["sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    payload["manifest_sha256"] = shortfall._hash_payload(
        payload, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="completion content changed|does not equal"
    ):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=count_dir / shortfall.COUNT_SUMMARY_NAME,
            topup_plan_path=plan_path,
            out_path=tmp_path / "index-sub.json",
            final_manifest_path=manifest_path,
            allow_pending=True,
            tokenizer=object(),
            renderer=_fake_renderer,
        )


def _pipeline_fixture(tmp_path: Path) -> dict:
    """A small fixture with real ``apps:`` ids so source recovery replays."""

    from gemmaloop.stage0.sources import convert_row

    pool = tmp_path / "pipeline_pool"
    lite = tmp_path / "pipeline_lite"
    (lite / "shards").mkdir(parents=True, exist_ok=True)
    raw_by_index: dict[int, dict] = {}
    problems: list[dict] = []
    splits: dict[str, list[str]] = {
        "pilot_p1": [],
        "pilot_tfit": [],
        "pilot_selection": [],
    }

    def add_code(raw_id: int, split: str, *, list_looking: bool = False) -> dict:
        payload = (
            {"inputs": [["1", "2"]], "outputs": [["3"]]}
            if list_looking
            else {"inputs": [f"{raw_id} {raw_id}\n"], "outputs": ["ok\n"]}
        )
        raw = {
            "problem_id": raw_id,
            "question": f"question {raw_id}",
            "solutions": ['print("ok")'],
            "input_output": json.dumps(payload),
        }
        raw_by_index[raw_id] = raw
        problem = convert_row("apps", raw, raw_id)
        assert problem is not None
        problem.update(
            {"split": "train", "pilot_subsplit": None, "bucket": 59}
        )
        problems.append(problem)
        splits[split].append(problem["problem_id"])
        return problem

    def add_math(name: str, split: str) -> dict:
        problem = {
            "problem_id": f"math:{name}",
            "source": "omr",
            "kind": "math",
            "prompt": f"math {name}"
            + "\n\nPut the final answer alone inside \\boxed{}.",
            "answer": "1",
            "tests": [],
            "metadata": {"raw_id": name},
            "split": "train",
            "pilot_subsplit": None,
            "bucket": 59,
        }
        problems.append(problem)
        splits[split].append(problem["problem_id"])
        return problem

    add_math("p1", "pilot_p1")
    add_code(101, "pilot_p1")
    add_math("t0", "pilot_tfit")
    add_math("t1", "pilot_tfit")
    add_code(201, "pilot_tfit", list_looking=True)
    add_code(202, "pilot_tfit")
    for index in range(3):
        add_math(f"s{index}", "pilot_selection")
    add_code(301, "pilot_selection")
    add_code(302, "pilot_selection")

    pool.mkdir(parents=True, exist_ok=True)
    (pool / "problems.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in problems), encoding="utf-8"
    )
    for name, ids in splits.items():
        rows = [row for row in problems if row["problem_id"] in set(ids)]
        (pool / f"split_{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (pool / "manifest.json").write_text(json.dumps({}), encoding="utf-8")

    request_ids = [
        f"{row['problem_id']}:{index}" for row in problems for index in range(4)
    ]
    (lite / "request_ids.txt").write_text(
        "".join(rid + "\n" for rid in request_ids), encoding="utf-8"
    )
    (lite / "run_marker.json").write_text(
        json.dumps({"run_id": "b605af7a625f93c7e3e08a02"}), encoding="utf-8"
    )
    (lite / "pilot").mkdir(parents=True, exist_ok=True)
    (lite / "pilot" / "manifest.json").write_text(
        json.dumps({"model_revision": "rev-test"}), encoding="utf-8"
    )
    records = []
    grades = []
    for request_id in request_ids:
        pid, index_text = request_id.rsplit(":", 1)
        problem = next(row for row in problems if row["problem_id"] == pid)
        index = int(index_text)
        if problem["kind"] == "code":
            content = "```python\nPASS\n```"
            correct = None
        else:
            content = "The answer is 1."
            correct = True
        reasoning = "step " * 20
        records.append(
            {
                "problem_id": pid,
                "sample_index": index,
                "request_id": request_id,
                "prompt": problem["prompt"],
                "completion": reasoning + content,
                "reasoning": reasoning,
                "content": content,
                "finish_reason": "stop",
                "thought_closed": True,
                "tokens": 80,
                "prompt_tokens": 40,
                "engine": "modal-vllm",
                "model_revision": "rev-test",
                "shard_index": 0,
            }
        )
        grades.append(
            {
                "request_id": request_id,
                "problem_id": pid,
                "sample_index": index,
                "kind": problem["kind"],
                "split": "pilot_dev",
                "kept": True,
                "drop_reason": None,
                "correct": correct,
                "rendered_tokens": 120,
                "reasoning_share": 0.8,
                "timing_s": 0.0,
            }
        )
    (lite / "shards" / "shard_00000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (lite / "sample_grades.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in grades), encoding="utf-8"
    )
    return {
        "pool": pool,
        "lite": lite,
        "problems": problems,
        "records": records,
        "grades": grades,
        "raw_by_index": raw_by_index,
        "request_ids": request_ids,
    }


def test_v3_pipeline_end_to_end_with_fake_docker_runner(tmp_path: Path, monkeypatch):
    fixture = _pipeline_fixture(tmp_path)
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)
    captured_inputs: list[str] = []

    def grade_fn(code, tests, timeout, **kwargs):
        for test in tests:
            captured_inputs.append(str(test.get("input")))
        return _fake_grade(code, tests, timeout)

    def record_loader(source, needed):
        assert source == "apps"
        for index in sorted(needed):
            yield index, fixture["raw_by_index"][index]

    def fake_renderer(rows, tokenizer, **kwargs):
        return [
            {
                "problem_id": row["problem_id"],
                "sample_index": row["sample_index"],
                "row_id": index,
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "label_positions": [1, 2],
            }
            for index, row in enumerate(rows)
        ]

    pin = {
        "tag": "python:3.12-slim",
        "id": "sha256:" + "b" * 64,
        "digest": "sha256:" + "b" * 64,
        "pinned": "python@sha256:" + "b" * 64,
        "repo_digests": ["python@sha256:" + "b" * 64],
    }
    result = shortfall.run_v3_pipeline(
        repo_root=ROOT,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        request_ids_path=fixture["lite"] / "request_ids.txt",
        out_root=tmp_path / "v3",
        image_pin=pin,
        workers=1,
        grade_fn=grade_fn,
        record_loader=record_loader,
        recovery_check=_recovery_ok,
        tokenizer=object(),
        renderer=fake_renderer,
        allow_pending=True,
    )
    assert Path(result["interpretation"]).is_file()
    assert Path(result["freeze"]).is_file()
    assert Path(result["count"]).is_file()
    assert Path(result["corpus"]).is_dir()
    # The integrated tree binds the registered prereg amendment, so the index
    # is complete rather than pending-data-addendum.
    assert result["index_status"] == shortfall.INDEX_STATUS_COMPLETE
    # The list-looking original was decoded by newline joining, not JSON.
    assert "1\n2" in captured_inputs
    assert '["1", "2"]' not in captured_inputs
    summary = result["count_summary"]
    assert summary["status"] == shortfall.COUNT_STATUS_COMPLETE
    assert summary["accounting"]["requests"] == len(fixture["request_ids"])
    assert summary["accounting"]["code_requests"] == 20
    index = json.loads(Path(result["index"]).read_text(encoding="utf-8"))
    assert index["final_counts"]["pilot_tfit"] == summary["nT0"]
    assert index["final_counts"]["pilot_selection"] == summary["nS0"]


def test_v3_cli_wires_the_documented_command(tmp_path: Path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "freeze_recount_v3_under_test", ROOT / "scripts" / "freeze_recount_v3.py"
    )
    script = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(script)
    pin = {
        "tag": "python:3.12-slim",
        "id": "sha256:" + "c" * 64,
        "digest": "sha256:" + "c" * 64,
        "pinned": "python@sha256:" + "c" * 64,
        "repo_digests": ["python@sha256:" + "c" * 64],
    }
    monkeypatch.setattr(script, "resolve_docker_image", lambda image: pin)
    monkeypatch.setattr(
        script,
        "DockerSandbox",
        lambda **kwargs: SimpleNamespace(verify_image_digest=lambda: None),
    )
    monkeypatch.setattr(script.modal_lite, "load_tokenizer", lambda: object())
    captured: dict = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "interpretation": "i",
            "interpretation_manifest_sha256": "i",
            "freeze": "f",
            "freeze_manifest_sha256": "f",
            "count": "c",
            "plan": "p",
            "corpus": "d",
            "corpus_manifest_sha256": "d",
            "index": "x",
            "index_sha256": "x",
            "index_status": "pending-data-addendum",
            "count_summary": {
                "nT0": 27,
                "nS0": 40,
                "pilot_p1": 16,
                "extension_required": False,
                "extension_decision": "no_extension_original_only",
                "infrastructure_failures": 0,
                "accounting": {},
            },
        }

    monkeypatch.setattr(script, "run_v3_pipeline", fake_pipeline)
    argv = [
        "--repo-root", str(ROOT),
        "--pool-dir", str(tmp_path / "pool"),
        "--lite-dir", str(tmp_path / "lite"),
        "--request-ids", str(tmp_path / "ids.txt"),
        "--out-root", str(tmp_path / "v3"),
    ]
    assert script.main(argv) == 0
    assert captured["allow_pending"] is True
    assert captured["recovery_check"] is script._docker_recovery_check
    assert script.main([*argv, "--no-pending"]) == 0
    assert captured["allow_pending"] is False


# ---------------------------------------------------------------------------
# Fix round 3: cleanup, retry evidence, count coverage, rendered bytes, one-time v3
# ---------------------------------------------------------------------------
def test_verified_cleanup_is_required_even_for_a_pass():
    assert shortfall._classify_test(
        {"passed": True, "cleanup_verified": None}
    ) == (True, "infrastructure_error")
    assert shortfall._classify_test(
        {"passed": True, "cleanup_verified": False}
    ) == (True, "infrastructure_error")
    assert shortfall._classify_test(
        {
            "passed": True,
            "exit_code": 0,
            "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            "lifecycle_valid": True,
            "cleanup_verified": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": False,
            "exit_observed": True,
        }
    ) == (False, "passed")
    # A Docker pass without a valid lifecycle record is infrastructure unknown.
    assert shortfall._classify_test(
        {"passed": True, "cleanup_verified": True}
    ) == (True, "infrastructure_error")


def test_retry_requires_documented_persisted_recovery(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        calls[code] = calls.get(code, 0) + 1
        if calls[code] == 1:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=lambda: True,  # undocumented bare boolean
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    # No documented recovery evidence, so no retry and no attempt reset.
    # Each test is dispatched once (max two tests on one problem in the
    # fixture); the per-test attempt lists below prove no retry occurred.
    assert max(calls.values()) <= 2
    for entry in graded:
        assert entry["recovery_evidence"] is None
        for test in entry["per_test"]:
            assert len(test["attempts"]) == 1


def test_attempt_summaries_keep_lifecycle_and_cleanup_evidence(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        calls[code] = calls.get(code, 0) + 1
        if calls[code] == 1:
            payload = _infra_tests(tests)
            for row in payload:
                row.update(
                    {
                        "lifecycle_class": "startup_failure",
                        "container_name": "container-x",
                        "container_started_at": "2026-11-06T00:00:00+00:00",
                        "container_finished_at": "2026-11-06T00:00:01+00:00",
                        "lifecycle_events": [{"observed_epoch": 1.0}],
                    }
                )
            return _result(payload, passed=False)
        return _fake_grade(code, tests, timeout)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    retried = [entry for entry in graded if entry["recovery_evidence"]]
    assert retried
    for entry in retried:
        assert entry["recovery_evidence"]["check"]
        assert entry["recovery_evidence"]["checked_utc"]
        for test in entry["per_test"]:
            if len(test["attempts"]) != 2:
                continue
            first, second = test["attempts"]
            for attempt in (first, second):
                assert "lifecycle_events" in attempt
                assert "cleanup_verified" in attempt
                assert "container_started_at" in attempt
            assert first["lifecycle_class"] == "startup_failure"
            assert first["container_name"] == "container-x"


def test_partial_per_test_result_is_refused_when_producing_a_count(tmp_path: Path):
    from gemmaloop.stage0 import local

    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)

    def partial_grade(code, tests, timeout, **kwargs):
        return SimpleNamespace(
            to_dict=lambda: {
                "passed": True,
                "sandboxed": True,
                "n_tests": 1,
                "error": None,
                "duration_s": 0.01,
                "timed_out": False,
                "tests": [
                    {
                        "index": 0,
                        "passed": True,
                        "input": "",
                        "expected": "",
                        "stdout": "",
                        "stderr": "",
                        "error": None,
                        "timed_out": False,
                        "duration_s": 0.01,
                        "exit_code": 0,
                        "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
                        "lifecycle_valid": True,
                        "cleanup_verified": True,
                        "handoff_success": True,
                        "termination_verified": True,
                        "deadline_alive": False,
                        "exit_observed": True,
                    }
                ],
            }
        )

    with pytest.raises(shortfall.ShortfallCountError, match="infrastructure"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=_count_out(tmp_path, "count-partial"),
            authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-partial")),
            grade_fn=partial_grade,
            workers=1,
        )
    rows = local.read_jsonl(
        _count_out(tmp_path, "count-partial") / shortfall.COUNT_PER_REQUEST_NAME
    )
    coverage = [row for row in rows if row.get("coverage_error")]
    assert coverage
    assert all(row["verified"] is False for row in coverage)
    assert any(
        "per-test rows" in row["coverage_error"]
        or "raw lifecycle frames" in row["coverage_error"]
        for row in coverage
    )


def test_loader_refuses_a_partial_per_test_count(tmp_path: Path):
    from gemmaloop.stage0 import local

    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    count_dir = _official_count(tmp_path, fixture, freeze_path, plan_path)
    per_request_path = count_dir / shortfall.COUNT_PER_REQUEST_NAME
    rows = local.read_jsonl(per_request_path)
    target = next(
        row
        for row in rows
        if row.get("problem_id") == "tfit:c0" and len(row.get("per_test") or []) == 2
    )
    target["per_test"] = target["per_test"][:1]
    local.write_jsonl_atomic(per_request_path, rows)
    summary_path = count_dir / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accounting"]["per_test_rows"] = sum(
        len(row.get("per_test") or []) for row in rows
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = count_dir / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][shortfall.COUNT_SUMMARY_NAME] = shortfall._file_entry(
        shortfall.COUNT_SUMMARY_NAME, summary_path
    )
    manifest["files"][shortfall.COUNT_PER_REQUEST_NAME] = shortfall._file_entry(
        shortfall.COUNT_PER_REQUEST_NAME, per_request_path
    )
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete test coverage"):
        shortfall.load_verified_count(count_dir, freeze_path=freeze_path)


def _content_renderer(rows, tokenizer, **kwargs):
    rendered = []
    for index, row in enumerate(rows):
        prompt = str(row["messages"][0]["content"])
        completion = str(row["messages"][1]["content"])
        ids = [len(prompt) % 97, len(completion) % 89]
        rendered.append(
            {
                "problem_id": row["problem_id"],
                "sample_index": row.get("sample_index"),
                "row_id": index,
                "input_ids": ids,
                "labels": [-100, ids[1]],
                "label_positions": [1],
            }
        )
    return rendered


def test_self_consistent_rendered_substitution_is_refused(tmp_path: Path):
    from gemmaloop.stage0 import local

    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    count_dir = _official_count(
        tmp_path, fixture, freeze_path, plan_path, required=False
    )
    manifest_path, _manifest = _final_corpus(
        tmp_path,
        fixture,
        freeze_path,
        count_dir,
        name="corpus-rendered-sub",
        renderer=_content_renderer,
    )
    corpus_dir = manifest_path.parent
    rendered_path = corpus_dir / "pilot_tfit.rendered.jsonl"
    rendered_rows = local.read_jsonl(rendered_path)
    rendered_rows[0]["input_ids"] = [
        int(value) for value in rendered_rows[0]["input_ids"]
    ] + [7]
    rendered_rows[0]["labels"] = [
        int(value) for value in rendered_rows[0]["labels"]
    ] + [7]
    rendered_rows[0]["label_positions"] = [
        int(value) for value in rendered_rows[0]["label_positions"]
    ] + [len(rendered_rows[0]["labels"]) - 1]
    local.write_jsonl_atomic(rendered_path, rendered_rows)
    payload = _rehash_final_manifest(manifest_path)
    payload["splits"]["pilot_tfit"]["rendered"]["sha256"] = hashlib.sha256(
        rendered_path.read_bytes()
    ).hexdigest()
    payload["splits"]["pilot_tfit"]["rendered"][
        "assistant_label_positions"
    ] = sum(len(row.get("label_positions") or []) for row in rendered_rows)
    payload["manifest_sha256"] = shortfall._hash_payload(
        payload, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="re-render|rendered row"):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=count_dir / shortfall.COUNT_SUMMARY_NAME,
            topup_plan_path=plan_path,
            out_path=tmp_path / "index-rendered-sub.json",
            final_manifest_path=manifest_path,
            allow_pending=True,
            tokenizer=object(),
            renderer=_content_renderer,
        )


def _v3_kwargs(fixture: dict, tmp_path: Path, out_root: Path) -> dict:
    pin = {
        "tag": "python:3.12-slim",
        "id": "sha256:" + "b" * 64,
        "digest": "sha256:" + "b" * 64,
        "pinned": "python@sha256:" + "b" * 64,
        "repo_digests": ["python@sha256:" + "b" * 64],
    }

    def record_loader(source, needed):
        for index in sorted(needed):
            yield index, fixture["raw_by_index"][index]

    def renderer(rows, tokenizer, **kwargs):
        return _fake_renderer(rows, tokenizer, **kwargs)

    return {
        "repo_root": ROOT,
        "pool_dir": fixture["pool"],
        "lite_dir": fixture["lite"],
        "request_ids_path": fixture["lite"] / "request_ids.txt",
        "out_root": out_root,
        "image_pin": pin,
        "workers": 1,
        "grade_fn": _fake_grade,
        "record_loader": record_loader,
        "recovery_check": _recovery_ok,
        "tokenizer": object(),
        "renderer": renderer,
        "allow_pending": True,
    }


def test_v3_is_a_one_time_freeze_and_refuses_an_alternate_out_root(
    tmp_path: Path, monkeypatch
):
    fixture = _pipeline_fixture(tmp_path)
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)
    first = shortfall.run_v3_pipeline(**_v3_kwargs(fixture, tmp_path, tmp_path / "v3"))
    assert Path(first["registry"]).is_file()
    registry = json.loads(Path(first["registry"]).read_text(encoding="utf-8"))
    assert registry["status"] == shortfall.V3_STATUS_COMPLETE
    with pytest.raises(shortfall.ShortfallV3RunError, match="already started"):
        shortfall.run_v3_pipeline(**_v3_kwargs(fixture, tmp_path, tmp_path / "v3"))
    with pytest.raises(shortfall.ShortfallV3RunError, match="already started"):
        shortfall.run_v3_pipeline(
            **_v3_kwargs(fixture, tmp_path, tmp_path / "v3-alternate")
        )


def test_v3_refuses_a_preexisting_output_root_without_a_registry(tmp_path: Path):
    fixture = _pipeline_fixture(tmp_path)
    out_root = tmp_path / "v3-foreign"
    out_root.mkdir()
    (out_root / "leftover.json").write_text("{}", encoding="utf-8")
    with pytest.raises(shortfall.ShortfallV3RunError, match="may not be clobbered"):
        shortfall.run_v3_pipeline(**_v3_kwargs(fixture, tmp_path, out_root))


@pytest.mark.parametrize(
    "stage,target,module_target",
    [
        ("interpretation", "recover_source_types", "source_recovery"),
        ("freeze", "freeze_verifier", None),
        ("decision", "plan_topup", None),
        ("count", "count_verified", None),
        ("corpus", "write_final_corpus", None),
        ("index", "write_freeze_index", None),
    ],
)
def test_v3_crash_at_each_stage_leaves_a_refusing_crash_state(
    tmp_path: Path, monkeypatch, stage, target, module_target
):
    fixture = _pipeline_fixture(tmp_path)
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)

    def explode(*args, **kwargs):
        raise RuntimeError(f"crash at {stage}")

    if module_target == "source_recovery":
        from gemmaloop.stage0 import source_recovery

        monkeypatch.setattr(source_recovery, target, explode)
    else:
        monkeypatch.setattr(shortfall, target, explode)
    out_root = tmp_path / f"v3-crash-{stage}"
    with pytest.raises(RuntimeError, match="crash at"):
        shortfall.run_v3_pipeline(**_v3_kwargs(fixture, tmp_path, out_root))
    registry_path = shortfall.v3_registry_path(fixture["lite"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["status"] == shortfall.V3_STATUS_CRASHED
    assert registry["failed_stage"] == stage
    assert registry["error"]
    with pytest.raises(shortfall.ShortfallV3RunError, match="already started"):
        shortfall.run_v3_pipeline(
            **_v3_kwargs(fixture, tmp_path, tmp_path / f"v3-after-{stage}")
        )


# ---------------------------------------------------------------------------
# Fix round 4: lifecycle on every outcome, durable ledger/resume, stage guard,
# failure-class-specific recovery
# ---------------------------------------------------------------------------
def _ledger_grade(
    fixture: dict,
    freeze: dict,
    ledger: Path,
    grade_fn,
    *,
    resume: bool,
    max_retries: int = 1,
    identity: dict | None = None,
    recovery_check=None,
):
    return shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=max_retries,
        recovery_check=recovery_check or _recovery_ok,
        ledger_path=ledger,
        ledger_identity=identity or {},
        resume=resume,
        authorization=_test_authorization(ledger.parent, fixture["lite"]),
    )


def _first_code_key(
    fixture: dict,
) -> tuple[str, str, list[dict], str]:
    kept = {
        row["request_id"]
        for row in fixture["grades"]
        if row.get("kept") and row.get("kind") == "code"
    }
    for request_id in fixture["request_ids"]:
        if request_id not in kept:
            continue
        problem_id = request_id.rsplit(":", 1)[0]
        problem = next(
            row for row in fixture["problems"] if row["problem_id"] == problem_id
        )
        if problem["kind"] == "code" and problem.get("tests"):
            record = next(
                row
                for row in fixture["records"]
                if row["request_id"] == request_id
            )
            return (
                request_id,
                problem_id,
                list(problem["tests"]),
                shortfall.extract_code(record),
            )
    raise AssertionError("fixture has no kept code request with tests")


def test_every_docker_outcome_requires_a_valid_lifecycle_record(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))

    def grade_rows(rows):
        return SimpleNamespace(
            to_dict=lambda: {
                "passed": all(row.get("passed") for row in rows),
                "sandboxed": True,
                "n_tests": len(rows),
                "error": None,
                "duration_s": 0.01,
                "timed_out": any(row.get("timed_out") for row in rows),
                "tests": rows,
            }
        )

    def row(**updates):
        base = {
            "index": 0,
            "passed": False,
            "input": "",
            "expected": "",
            "stdout": "",
            "stderr": "",
            "error": None,
            "timed_out": False,
            "duration_s": 0.01,
            "exit_code": 0,
            "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            "lifecycle_valid": True,
            "cleanup_verified": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": False,
            "exit_observed": True,
        }
        base.update(updates)
        return base

    valid_pass = row(passed=True, stdout="1")
    valid_wrong = row(passed=False, stdout="2", error="wrong answer")
    valid_error = row(passed=False, exit_code=1, error="docker exit code 1")
    valid_timeout = row(
        timed_out=True,
        lifecycle_class=grader.LIFECYCLE_CANDIDATE_RUNNING,
        error="timeout after 6s",
        handoff_success=True,
        termination_verified=True,
        deadline_alive=True,
        exit_observed=True,
    )
    invalid_pass = row(
        passed=True,
        lifecycle_class=grader.LIFECYCLE_COMPLETED_BEFORE,
        lifecycle_valid=False,
        marker_error="missing lifecycle evidence",
        error="untrusted lifecycle evidence",
    )
    invalid_wrong = row(
        passed=False,
        lifecycle_class=grader.LIFECYCLE_UNRESOLVED,
        lifecycle_valid=False,
        marker_error="HMAC invalid",
        error="untrusted lifecycle evidence",
    )
    cases = {
        "valid_pass": (valid_pass, (False, "passed")),
        "valid_wrong": (valid_wrong, (False, "wrong_answer")),
        "valid_candidate_error": (valid_error, (False, "candidate_error")),
        "valid_timeout": (valid_timeout, (False, "timeout")),
        "invalid_pass": (invalid_pass, (True, "infrastructure_error")),
        "invalid_wrong": (invalid_wrong, (True, "infrastructure_error")),
    }
    for name, (test_row, expected) in cases.items():
        classified = shortfall._classify_test(test_row)
        assert classified == expected, (name, classified)

    # The exact r4 probe: an unresolved lifecycle class may never be accepted
    # as a pass or a wrong answer.
    for passed, error in ((True, None), (False, "wrong answer")):
        assert shortfall._classify_test(
            {
                "passed": passed,
                "error": error,
                "timed_out": False,
                "exit_code": 0 if passed else None,
                "lifecycle_class": grader.LIFECYCLE_UNRESOLVED,
                "lifecycle_valid": False,
                "cleanup_verified": True,
            }
        ) == (True, "infrastructure_error")


def test_real_grade_path_rejects_untrusted_lifecycle_evidence(monkeypatch):
    sandbox = grader.DockerSandbox()
    monkeypatch.setattr(sandbox, "ensure_image", lambda: None)

    def fake_run(command, *, input_text, timeout, container_name, **kwargs):
        return (
            0,
            "1",
            "",
            False,
            0.01,
            {
                "class": grader.LIFECYCLE_COMPLETED_BEFORE,
                "container_name": container_name,
                "container_started_at": "2026-11-06T00:00:00+00:00",
                "container_finished_at": "2026-11-06T00:00:01+00:00",
                "events": [],
                "markers": [],
                "marker_error": "forged marker channel",
                "record_error": "forged marker channel",
                "lifecycle_valid": False,
                "cleanup_verified": True,
            },
        )

    monkeypatch.setattr(grader, "_run_process_docker", fake_run)
    result = sandbox.run_tests(
        "print(1)", [{"input": "", "output": "1", "type": "stdin"}], timeout=6
    )
    test = result.tests[0]
    assert test.passed is False
    assert test.lifecycle_valid is False
    assert "untrusted lifecycle evidence" in (test.error or "")
    assert shortfall._classify_test(test.__dict__) == (
        True,
        "infrastructure_error",
    )


def test_out_of_order_lifecycle_markers_are_unknown():
    assert grader.classify_lifecycle_evidence(
        markers=[
            {"event": grader.LIFECYCLE_EVENT_START, "epoch": 90.0},
            {"event": grader.LIFECYCLE_EVENT_FINISH, "epoch": 80.0},
        ],
        marker_error=None,
        timed_out=False,
        cleanup_verified=True,
        deadline_epoch=100.0,
        status="exited",
    ) == grader.LIFECYCLE_UNRESOLVED


def test_recovery_check_is_specific_to_the_failure_class():
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        text = " ".join(str(value) for value in command)
        if " info " in f" {text} ":
            return SimpleNamespace(returncode=0, stdout="28.0\n", stderr="")
        if "recovery-mount" in text and "run" in command:
            return SimpleNamespace(returncode=0, stdout="mount-ok\n", stderr="")
        if "recovery-noop" in text and "run" in command:
            return SimpleNamespace(returncode=0, stdout="recovery-ok\n", stderr="")
        if command[:2] == ["docker", "rm"]:
            return SimpleNamespace(returncode=0, stdout="gone\n", stderr="")
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(
                returncode=1, stdout="", stderr="Error: No such object"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    startup = shortfall._docker_recovery_check(
        ["startup", "daemon"], image="python@sha256:" + "a" * 64, runner=runner
    )
    assert startup["ok"] is True
    assert {"daemon", "noop_container"} <= set(startup["checks"])
    assert startup["checks"]["noop_container"]["removed"] is True

    mount = shortfall._docker_recovery_check(
        ["mount"], image="python@sha256:" + "a" * 64, runner=runner
    )
    assert mount["ok"] is True
    assert "bind_mount" in mount["checks"]

    cleanup = shortfall._docker_recovery_check(
        ["cleanup"],
        image="python@sha256:" + "a" * 64,
        container_name="stale-container",
        runner=runner,
    )
    assert cleanup["ok"] is True
    assert "cleanup_removal" in cleanup["checks"]
    assert any("stale-container" in command for command in commands)


def test_recovery_failure_classes_track_the_observed_defect():
    assert shortfall._recovery_failure_classes(
        [
            {
                "infrastructure_failure": True,
                "stderr": "bind mount failed",
                "cleanup_verified": True,
                "lifecycle_class": grader.LIFECYCLE_UNRESOLVED,
            }
        ]
    ) == ["mount", "startup"]
    assert shortfall._recovery_failure_classes(
        [
            {
                "infrastructure_failure": True,
                "stderr": "",
                "cleanup_verified": False,
                "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            }
        ]
    ) == ["cleanup"]
    assert shortfall._recovery_failure_classes(
        [
            {
                "infrastructure_failure": True,
                "stderr": "cannot connect to the docker daemon",
                "cleanup_verified": True,
                "lifecycle_class": grader.LIFECYCLE_STARTUP_FAILURE,
            }
        ]
    ) == ["startup"]
    # An interrupted in-flight attempt also recovers its own recorded
    # container cleanup, so the cleanup class is part of the check.
    assert shortfall._recovery_failure_classes([], in_flight=True) == [
        "cleanup",
        "mount",
        "startup",
    ]


def test_retry_persists_class_specific_recovery_evidence(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    seen: list[dict] = []
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        calls[code] = calls.get(code, 0) + 1
        if calls[code] == 1:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    def recovery(**kwargs):
        seen.append(kwargs)
        return {
            "check": "startup recovery",
            "failure_classes": kwargs.get("failure_classes"),
            "checks": {"daemon": {"ok": True}, "noop_container": {"ok": True}},
            "checked_utc": "2026-11-06T12:00:00+00:00",
            "ok": True,
            "detail": "ok",
        }

    ledger = tmp_path / "attempts.jsonl"
    entries = _ledger_grade(
        fixture,
        freeze,
        ledger,
        grade_fn,
        resume=False,
        recovery_check=recovery,
    )
    assert seen
    assert any("startup" in item.get("failure_classes", []) for item in seen)
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry.get("recovery_evidence")
    ]
    assert graded
    for entry in graded:
        assert entry["recovery_evidence"]["checks"]["noop_container"]["ok"]
    rows = shortfall._read_attempt_ledger(ledger)
    retried = [
        row
        for row in rows
        if row.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
        and row.get("recovery_evidence")
    ]
    assert retried
    assert all(
        row["recovery_evidence"]["checks"]["daemon"]["ok"] for row in retried
    )


def test_attempt_ledger_is_append_only_and_resume_never_regrades(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "attempts.jsonl"
    first_calls: list[str] = []

    def first_grade(code, tests, timeout, **kwargs):
        first_calls.append(code)
        return _fake_grade(code, tests, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, first_grade, resume=False, max_retries=0
    )
    rows = shortfall._read_attempt_ledger(ledger)
    starts = [
        index
        for index, row in enumerate(rows)
        if row.get("kind") == shortfall.LEDGER_ATTEMPT_START
    ]
    results = [
        index
        for index, row in enumerate(rows)
        if row.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
    ]
    assert starts and results
    assert min(starts) < min(results)
    assert len(rows) == 1 + len(starts) + len(results)

    def must_not_grade(*args, **kwargs):
        raise AssertionError("a conclusive attempt was regraded")

    resumed = _ledger_grade(
        fixture,
        freeze,
        ledger,
        must_not_grade,
        resume=True,
        max_retries=0,
    )
    baseline = {
        entry["request_id"]: (entry["verified"], entry["failure_class"])
        for entry in entries
    }
    for entry in resumed:
        assert baseline[entry["request_id"]] == (
            entry["verified"],
            entry["failure_class"],
        )


def test_ledger_crash_at_every_write_boundary_resumes_to_the_same_count(
    tmp_path: Path, monkeypatch
):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    original = shortfall._append_attempt_ledger

    counter = {"appends": 0}

    def recording(path, record):
        counter["appends"] += 1
        original(path, record)

    monkeypatch.setattr(shortfall, "_append_attempt_ledger", recording)
    baseline = _ledger_grade(
        fixture,
        freeze,
        tmp_path / "baseline-ledger.jsonl",
        _fake_grade,
        resume=False,
    )
    total = counter["appends"]
    assert total > 3
    baseline_state = {
        entry["request_id"]: (entry["verified"], entry["failure_class"])
        for entry in baseline
    }
    monkeypatch.setattr(shortfall, "_append_attempt_ledger", original)

    for boundary in range(total):
        ledger = tmp_path / f"crash-ledger-{boundary}.jsonl"
        seen = {"count": 0}

        def flaky(path, record, *, boundary=boundary, seen=seen):
            if seen["count"] == boundary:
                raise RuntimeError("simulated crash at write boundary")
            seen["count"] += 1
            original(path, record)

        monkeypatch.setattr(shortfall, "_append_attempt_ledger", flaky)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _ledger_grade(
                fixture,
                freeze,
                ledger,
                _fake_grade,
                resume=False,
            )
        monkeypatch.setattr(shortfall, "_append_attempt_ledger", original)
        crashed = shortfall._read_attempt_ledger(ledger)
        interrupted_requests: set[str] = set()
        for (request_id, _index), entries in shortfall._ledger_entries_by_test(
            crashed
        ).items():
            _result, _next_attempt, in_flight, _used = shortfall._ledger_plan(
                entries
            )
            if in_flight is not None:
                interrupted_requests.add(request_id)
        resumed = _ledger_grade(
            fixture,
            freeze,
            ledger,
            _fake_grade,
            resume=True,
        )
        state = {
            entry["request_id"]: (entry["verified"], entry["failure_class"])
            for entry in resumed
        }
        for request_id, outcome in state.items():
            if request_id in interrupted_requests:
                # An interrupted attempt has no verified cleanup, so its
                # request stays blocked instead of silently reusing the retry.
                assert outcome[0] is False, (boundary, request_id)
                assert outcome[1] in {
                    "coverage_error",
                    "infrastructure",
                }, (boundary, request_id, outcome)
            else:
                assert outcome == baseline_state[request_id], (
                    boundary,
                    request_id,
                    outcome,
                )
        rows = shortfall._read_attempt_ledger(ledger)
        assert all(int(row.get("attempt") or 1) <= 2 for row in rows), boundary


def test_in_flight_attempt_waits_for_cleanup_recovery_before_retry(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "attempts.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    shortfall._ensure_attempt_ledger(ledger, {})
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": 1,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "container_name": "gemmaloop-manual-inflight",
            "started_utc": shortfall._utc_now(),
        },
    )

    def passing(code, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(fixture, freeze, ledger, passing, resume=True)
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    # No container-bound cleanup recovery exists yet, so the interrupted
    # attempt stays pending and the retry is not dispatched.
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error"
    ]
    assert test["attempts"][0].get("interrupted") is True
    assert test["attempts"][0].get("cleanup_verified") is False
    assert target["verified"] is False
    assert "unverified cleanup" in (target.get("coverage_error") or "")


def test_second_unknown_after_the_retry_is_terminal(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "attempts.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    shortfall._ensure_attempt_ledger(ledger, {})
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": 1,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "container_name": "gemmaloop-manual-terminal",
            "started_utc": shortfall._utc_now(),
        },
    )

    def infra(code, tests_, timeout, **kwargs):
        rows = _infra_tests(tests_)
        for row in rows:
            # The retry's own container removal is unverified, so the terminal
            # disposition must remain blocked even though the interrupted
            # attempt's cleanup was recovered.
            row["cleanup_verified"] = False
        return _result(rows, passed=False)

    def recovery(**kwargs):
        return _cleanup_recovery(str(kwargs.get("container_name")))

    entries = _ledger_grade(
        fixture, freeze, ledger, infra, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "infrastructure_error",
    ]
    assert target["infrastructure_failure"] is True
    entries_again = _ledger_grade(
        fixture, freeze, ledger, infra, resume=True, recovery_check=recovery
    )
    target_again = next(
        entry for entry in entries_again if entry["request_id"] == request_id
    )
    test_again = next(
        row for row in target_again["per_test"] if row["index"] == 0
    )
    assert len(test_again["attempts"]) == 2
    # The retry attempt has no verified cleanup, so even after the interrupted
    # attempt's recovery a second infrastructure/unknown retry cannot receive
    # the terminal disposition.
    assert test_again.get("terminal_infrastructure") is not True
    assert target_again["terminal_infrastructure"] is False
    assert target_again["blocking_infrastructure"] is True


def test_ledger_resume_refuses_changed_candidate_and_test_bytes(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    request_id, problem_id, tests, code = _first_code_key(fixture)
    for field in ("request_sha256", "test_sha256"):
        ledger = tmp_path / f"changed-{field}.jsonl"
        shortfall._ensure_attempt_ledger(ledger, {})
        record = {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": 1,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "started_utc": shortfall._utc_now(),
        }
        record[field] = "0" * 64
        shortfall._append_attempt_ledger(ledger, record)
        with pytest.raises(
            ValueError, match="candidate bytes changed|test bytes changed"
        ):
            _ledger_grade(fixture, freeze, ledger, _fake_grade, resume=True)


def test_ledger_header_identity_mismatch_is_refused(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "identity.jsonl"
    shortfall._ensure_attempt_ledger(
        ledger, {"freeze_manifest_sha256": "a" * 64}
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _ledger_grade(
            fixture,
            freeze,
            ledger,
            _fake_grade,
            resume=True,
            identity={"freeze_manifest_sha256": "b" * 64},
        )


def test_closed_run_stages_require_the_claimed_run_authorization(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    out_root = tmp_path / "v3-guard"
    registry_path, registry = shortfall._claim_v3_registry(
        repo_root=ROOT,
        lite_dir=fixture["lite"],
        out_root=out_root,
    )
    assert registry["status"] == shortfall.V3_STATUS_CLAIMED

    with pytest.raises(shortfall.ShortfallV3RunError, match="is not authorization"):
        shortfall.freeze_verifier(
            repo_root=ROOT,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            request_ids_path=fixture["lite"] / "request_ids.txt",
            out_dir=out_root / "verifier_v3",
            image_pin={
                "pinned": "python@sha256:" + "a" * 64,
                "id": "sha256:" + "a" * 64,
            },
            exclusive=True,
        )
    with pytest.raises(shortfall.ShortfallV3RunError, match="is not authorization"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            topup_plan_path=plan_path,
            out_dir=out_root / "count_v3",
            exclusive=True,
        )
    with pytest.raises(shortfall.ShortfallV3RunError, match="may only run"):
        shortfall.plan_topup(freeze_path, out_root / "pilot_topup_plan_v3.json")
    with pytest.raises(shortfall.ShortfallV3RunError, match="is not authorization"):
        shortfall.write_final_corpus(out_root / "corpus_v3", {}, exclusive=True)
    with pytest.raises(shortfall.ShortfallV3RunError, match="is not authorization"):
        shortfall.write_freeze_index(
            repo_root=ROOT,
            freeze_path=freeze_path,
            count_summary_path=tmp_path / "missing.json",
            topup_plan_path=tmp_path / "plan.json",
            out_path=out_root / "index.json",
            exclusive=True,
        )
    stolen = shortfall.V3RunAuthorization(
        registry_path=registry_path,
        run_token=registry["run_token"],
        stage="freeze",
    )
    with pytest.raises(shortfall.ShortfallV3RunError, match="not the directory"):
        shortfall.freeze_verifier(
            repo_root=ROOT,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            request_ids_path=fixture["lite"] / "request_ids.txt",
            out_dir=tmp_path / "elsewhere",
            image_pin={
                "pinned": "python@sha256:" + "a" * 64,
                "id": "sha256:" + "a" * 64,
            },
            authorization=stolen,
            resume=True,
        )


def test_v3_resume_reuses_completed_stages_and_is_not_a_second_freeze(
    tmp_path: Path, monkeypatch
):
    fixture = _pipeline_fixture(tmp_path)
    monkeypatch.setattr(shortfall, "decide_extension", lambda n_t0, n_s0: False)
    kwargs = _v3_kwargs(fixture, tmp_path, tmp_path / "v3-resume")
    original_count = shortfall.count_verified

    def crashing_count(**inner):
        raise RuntimeError("crash during count")

    monkeypatch.setattr(shortfall, "count_verified", crashing_count)
    with pytest.raises(RuntimeError, match="crash during count"):
        shortfall.run_v3_pipeline(**kwargs)
    monkeypatch.setattr(shortfall, "count_verified", original_count)

    registry_path = Path(kwargs["lite_dir"]) / "shortfall" / "v3_registry.json"
    crashed = json.loads(registry_path.read_text(encoding="utf-8"))
    assert crashed["status"] == shortfall.V3_STATUS_CRASHED
    assert crashed["failed_stage"] == "count"
    freeze_path = (
        kwargs["out_root"] / "verifier_v3" / shortfall.FREEZE_MANIFEST_NAME
    )
    freeze_before = freeze_path.read_bytes()

    graded: list[str] = []

    def counting_grade(code, tests, timeout, **kwargs):
        graded.append(code)
        return _fake_grade(code, tests, timeout)

    resumed = shortfall.resume_v3_pipeline(
        **{**kwargs, "grade_fn": counting_grade}
    )
    assert freeze_path.read_bytes() == freeze_before
    assert graded
    assert Path(resumed["index"]).is_file()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["status"] == shortfall.V3_STATUS_COMPLETE
    assert registry["resume_count"] == 1

    # A crash after the count must not grade again.
    fixture2 = _pipeline_fixture(tmp_path / "late")
    kwargs2 = _v3_kwargs(
        fixture2, tmp_path, tmp_path / "v3-resume-late"
    )
    real_corpus = shortfall.write_final_corpus

    def crashing_corpus(*args, **inner):
        raise RuntimeError("crash during corpus")

    monkeypatch.setattr(shortfall, "write_final_corpus", crashing_corpus)
    with pytest.raises(RuntimeError, match="crash during corpus"):
        shortfall.run_v3_pipeline(**kwargs2)
    monkeypatch.setattr(shortfall, "write_final_corpus", real_corpus)

    graded.clear()
    resumed2 = shortfall.resume_v3_pipeline(
        **{**kwargs2, "grade_fn": counting_grade}
    )
    assert graded == []
    assert Path(resumed2["index"]).is_file()


def test_attempt_ledger_recovers_a_torn_trailing_write(tmp_path: Path):
    ledger = tmp_path / "attempts.jsonl"
    shortfall._ensure_attempt_ledger(ledger, {"freeze_manifest_sha256": "a" * 64})
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": "torn:0",
            "problem_id": "torn",
            "test_index": 0,
            "attempt": 1,
            "request_sha256": shortfall._sha256_text("code"),
            "test_sha256": shortfall._sha256_text("test"),
            "started_utc": shortfall._utc_now(),
        },
    )
    with ledger.open("ab") as handle:
        handle.write(b'{"kind":"attempt_result"')
    rows = shortfall._read_attempt_ledger(ledger)
    assert rows
    assert rows[-1]["kind"] == shortfall.LEDGER_ATTEMPT_START
    assert ledger.read_bytes().endswith(b"\n")

# Fix round 5: terminal infrastructure disposition, state-specific loader and
# readiness propagation.
def test_second_unknown_after_the_single_retry_is_terminal_not_blocked(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    calls: dict[str, int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        calls[code] = calls.get(code, 0) + 1
        return _result(_infra_tests(tests), passed=False)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    for entry in graded:
        assert entry["terminal_infrastructure"] is True
        assert entry["blocking_infrastructure"] is False
        assert entry["infrastructure_failure"] is True
        # Never relabeled as a candidate failure or timeout.
        assert entry["failure_class"] == "infrastructure"
        for test in entry["per_test"]:
            assert test["status"] == "infrastructure_error"
            assert test["terminal_status"] == (
                shortfall.TEST_STATUS_TERMINAL_INFRASTRUCTURE
            )
            assert test["terminal_reason"]
            assert len(test["attempts"]) == 2


def test_terminal_infrastructure_count_completes_and_loader_accepts(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    out = _count_out(tmp_path, "count-terminal")

    def infra_grade(code, tests, timeout, **kwargs):
        return _result(_infra_tests(tests), passed=False)

    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=out,
        grade_fn=infra_grade,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(
            tmp_path, fixture["lite"], out_dir=out
        ),
    )
    assert summary["status"] == (
        shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE
    )
    assert summary["blocking_infrastructure_failures"] == 0
    assert summary["terminal_infrastructure_failures"] > 0
    terminal = summary["terminal_infrastructure"]
    assert terminal["count"] > 0 and terminal["request_count"] > 0
    assert terminal["tests"]
    assert terminal["per_split"]
    assert set(terminal["per_domain"]) == {"math", "code"}
    for item in terminal["tests"]:
        assert item["reason"]
        assert item["terminal_status"] == (
            shortfall.TEST_STATUS_TERMINAL_INFRASTRUCTURE
        )
        assert item["attempts"] == 2
    # The affected code completions are not verified; other domains can still
    # qualify.
    assert summary["per_split"]["pilot_tfit"]["code_count"] == 0
    assert summary["per_split"]["pilot_tfit"]["math_count"] > 0

    loaded = shortfall.load_verified_count(
        out, freeze_path=freeze_path, topup_plan_path=plan_path
    )
    assert loaded["status"] == summary["status"]
    assert loaded["terminal_infrastructure"]["count"] == terminal["count"]

    rows = [
        json.loads(line)
        for line in (out / shortfall.COUNT_PER_REQUEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    infra = [row for row in rows if row["infrastructure_failure"]]
    assert infra
    for row in infra:
        assert row["blocking_infrastructure"] is False
        assert row["failure_class"] == "infrastructure"
        for test in row["per_test"]:
            assert test["terminal_infrastructure"] is True
            assert test["status"] == "infrastructure_error"


def test_loader_refuses_terminal_count_without_terminal_accounting(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    out = _count_out(tmp_path, "count-terminal")

    def infra_grade(code, tests, timeout, **kwargs):
        return _result(_infra_tests(tests), passed=False)

    shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=out,
        grade_fn=infra_grade,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(
            tmp_path, fixture["lite"], out_dir=out
        ),
    )
    summary_path = out / shortfall.COUNT_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("terminal_infrastructure")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = out / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][shortfall.COUNT_SUMMARY_NAME]
    entry["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    entry["bytes"] = summary_path.stat().st_size
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identities|terminal"):
        shortfall.load_verified_count(
            out, freeze_path=freeze_path, topup_plan_path=plan_path
        )


def test_pending_retry_without_recovery_stays_a_blocker(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)

    def infra_grade(code, tests, timeout, **kwargs):
        return _result(_infra_tests(tests), passed=False)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=tmp_path / "direct-attempts.jsonl",
        grade_fn=infra_grade,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=None,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    graded = [
        entry
        for entry in entries
        if entry["domain"] == "code" and entry["per_test"]
    ]
    assert graded
    for entry in graded:
        assert entry["terminal_infrastructure"] is False
        assert entry["blocking_infrastructure"] is True
        for test in entry["per_test"]:
            assert len(test["attempts"]) == 1
            assert not test.get("terminal_infrastructure")


def test_readiness_guard_accepts_terminal_count_and_rejects_blocked(
    tmp_path: Path,
):
    terminal = {
        "count": 1,
        "tests": [{"request_id": "x:0", "test_index": 0}],
    }
    index = _write_index(
        tmp_path,
        tfit=20,
        selection=30,
        name="terminal.json",
        count_status=shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE,
        terminal_infrastructure=terminal,
    )
    assert shortfall.check_freeze_readiness(index["path"])

    bad = _write_index(
        tmp_path,
        tfit=20,
        selection=30,
        name="blocked.json",
        count_status=shortfall.COUNT_STATUS_BLOCKED_INFRASTRUCTURE,
        terminal_infrastructure=terminal,
    )
    with pytest.raises(shortfall.ShortfallReadinessError, match="blocked"):
        shortfall.check_freeze_readiness(bad["path"])

def test_interrupted_retry_exhausts_the_allowance_and_unverified_cleanup_blocks(
    tmp_path: Path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "attempts.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    shortfall._ensure_attempt_ledger(ledger, {})
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": 1,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "started_utc": shortfall._utc_now(),
        },
    )
    shortfall._append_attempt_ledger(
        ledger,
        shortfall._ledger_result_record(
            request_id=request_id,
            problem_id=problem_id,
            test_index=0,
            attempt=1,
            evidence={
                "status": "infrastructure_error",
                "passed": False,
                "infrastructure_failure": True,
                "error": "first unknown",
            },
        ),
    )
    # The retry started but the host crashed before its result was committed.
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": 2,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "started_utc": shortfall._utc_now(),
        },
    )

    def must_not_grade(*args, **kwargs):
        raise AssertionError("an interrupted retry must not create a third attempt")

    entries = _ledger_grade(
        fixture, freeze, ledger, must_not_grade, resume=True
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "infrastructure_error",
    ]
    assert test["attempts"][1].get("interrupted") is True
    # The retry allowance is exhausted (no third attempt), but an interrupted
    # attempt has no verified termination/cleanup, so it stays blocking.
    assert test.get("terminal_infrastructure") is not True
    assert target["terminal_infrastructure"] is False
    assert target["blocking_infrastructure"] is True
    rows = shortfall._read_attempt_ledger(ledger)
    assert all(int(row.get("attempt") or 1) <= 2 for row in rows)


def test_resume_revalidates_the_pinned_image_digest(tmp_path: Path, monkeypatch):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    calls: list[str] = []
    registry_path, registry = shortfall._claim_v3_registry(
        repo_root=ROOT,
        lite_dir=fixture["lite"],
        out_root=tmp_path / "v3-auth",
    )
    authorization = shortfall.V3RunAuthorization(
        registry_path=registry_path,
        run_token=registry["run_token"],
        stage="count",
    )

    class FakeSandbox:
        def __init__(self, image, expected_digest=None):
            calls.append(image)

        def verify_image_digest(self):
            raise grader.DockerImageDigestError("image digest changed")

    monkeypatch.setattr(grader, "DockerSandbox", FakeSandbox)
    with pytest.raises(grader.DockerImageDigestError, match="digest"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            topup_plan_path=plan_path,
            out_dir=tmp_path / "v3-auth" / "count_v3",
            workers=1,
            resume=True,
            authorization=authorization,
        )
    assert calls

def test_terminal_infrastructure_only_excludes_the_affected_completion(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)

    def grade_fn(code, tests, timeout, **kwargs):
        if "tfit:c0:0" in code:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    summary = shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=_count_out(tmp_path, "count-terminal-mixed"),
        authorization=_test_authorization(tmp_path, fixture["lite"], out_dir=_count_out(tmp_path, "count-terminal-mixed")),
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
    )
    assert summary["status"] == (
        shortfall.COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE
    )
    assert "tfit:c0" in summary["per_split"]["pilot_tfit"]["code_verified_problems"]
    assert summary["per_split"]["pilot_tfit"]["code_count"] > 0
    assert any(
        item["request_id"].startswith("tfit:c0:0")
        for item in summary["terminal_infrastructure"]["tests"]
    )

# ---------------------------------------------------------------------------
# Fix round 6: causal deadline notice, positive handoff, re-verifiable loader,
# per-test ledger streaming, terminal cleanup and bound grading entry points.
# ---------------------------------------------------------------------------
def _valid_r6_count(tmp_path: Path, name: str = "count-r6"):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    interpretation_path = _interpretation(fixture, tmp_path, freeze_path)
    out = _count_out(tmp_path, name)
    shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=interpretation_path,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=out,
        grade_fn=_fake_grade,
        workers=1,
        authorization=_test_authorization(
            tmp_path, fixture["lite"], out_dir=out
        ),
    )
    return fixture, freeze_path, plan_path, out


def _rewrite_count_per_request(out: Path, mutate) -> None:
    path = out / shortfall.COUNT_PER_REQUEST_NAME
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutate(rows)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_path = out / shortfall.COUNT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][shortfall.COUNT_PER_REQUEST_NAME]
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["bytes"] = path.stat().st_size
    manifest["manifest_sha256"] = shortfall._hash_payload(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _first_conclusive_row(rows):
    for row in rows:
        if row.get("verified"):
            for test in row.get("per_test") or []:
                if not test.get("infrastructure_failure"):
                    return row, test
    raise AssertionError("count has no conclusive pass row")


def test_loader_refuses_a_conclusive_row_without_raw_lifecycle_frames(
    tmp_path: Path,
):
    fixture, freeze_path, plan_path, out = _valid_r6_count(tmp_path)

    def strip(rows):
        _row, test = _first_conclusive_row(rows)
        test.pop("lifecycle_frames_raw", None)
        test.pop("lifecycle_auth_key", None)

    _rewrite_count_per_request(out, strip)
    with pytest.raises(ValueError, match="raw lifecycle frames"):
        shortfall.load_verified_count(
            out, freeze_path=freeze_path, topup_plan_path=plan_path
        )


def test_loader_rejects_forged_frames_and_forged_stdout(tmp_path: Path):
    fixture, freeze_path, plan_path, out = _valid_r6_count(
        tmp_path, "count-r6-forged"
    )

    def wrong_key(rows):
        _row, test = _first_conclusive_row(rows)
        test["lifecycle_auth_key"] = "0" * 64

    _rewrite_count_per_request(out, wrong_key)
    with pytest.raises(ValueError, match="re-authenticate"):
        shortfall.load_verified_count(
            out, freeze_path=freeze_path, topup_plan_path=plan_path
        )

    fixture, freeze_path, plan_path, out = _valid_r6_count(
        tmp_path, "count-r6-stdout"
    )

    def forged_stdout(rows):
        _row, test = _first_conclusive_row(rows)
        test["stdout"] = "a forged passing answer"

    _rewrite_count_per_request(out, forged_stdout)
    with pytest.raises(ValueError, match="stdout does not reproduce"):
        shortfall.load_verified_count(
            out, freeze_path=freeze_path, topup_plan_path=plan_path
        )


def test_unverified_cleanup_cannot_receive_terminal_disposition():
    def row(cleanup: bool) -> dict:
        return {
            "index": 0,
            "status": "infrastructure_error",
            "passed": False,
            "infrastructure_failure": True,
            "cleanup_verified": cleanup,
            "timed_out": False,
            "error": "cleanup failed",
            "attempts": [
                {
                    "status": "infrastructure_error",
                    "error": "first",
                    "cleanup_verified": cleanup,
                },
                {
                    "status": "infrastructure_error",
                    "error": "retry",
                    "cleanup_verified": cleanup,
                },
            ],
        }

    blocked = [row(False)]
    shortfall._apply_terminal_infrastructure(blocked, 1)
    assert blocked[0].get("terminal_infrastructure") is not True

    terminal = [row(True)]
    shortfall._apply_terminal_infrastructure(terminal, 1)
    assert terminal[0]["terminal_infrastructure"] is True


def test_grade_original_requests_refuses_unbound_calls_before_docker(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    interpretation_path = _interpretation(fixture, tmp_path, freeze_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)

    # Default real-Docker grader, no authorization and no ledger.
    with pytest.raises(shortfall.ShortfallV3RunError, match="claimed official"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
        )
    # A supplied grade_fn never substitutes for the claimed run.
    with pytest.raises(shortfall.ShortfallV3RunError, match="claimed official"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            grade_fn=_fake_grade,
        )
    # A ledger is mandatory even with authorization and an explicit grade_fn.
    with pytest.raises(shortfall.ShortfallV3RunError, match="durable attempt ledger"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            grade_fn=_fake_grade,
            authorization=_test_authorization(tmp_path, fixture["lite"]),
        )
    # A forged stage token is refused against the claimed registry.
    registry_path, registry = shortfall._claim_v3_registry(
        repo_root=ROOT,
        lite_dir=fixture["lite"],
        out_root=tmp_path / "v3-r6-auth",
    )
    with pytest.raises(shortfall.ShortfallV3RunError, match="token"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            grade_fn=_fake_grade,
            ledger_path=tmp_path / "auth-ledger.jsonl",
            authorization=shortfall.V3RunAuthorization(
                registry_path=registry_path,
                run_token="0" * 64,
                stage="count",
            ),
        )
    # count_verified's real-Docker path is bound too, with or without a fake.
    with pytest.raises(shortfall.ShortfallV3RunError, match="claimed official"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=interpretation_path,
            topup_plan_path=plan_path,
            out_dir=tmp_path / "unbound-count",
        )
    with pytest.raises(shortfall.ShortfallV3RunError, match="claimed official"):
        shortfall.count_verified(
            freeze_path=freeze_path,
            interpretation_path=interpretation_path,
            topup_plan_path=plan_path,
            out_dir=tmp_path / "unbound-count-fake",
            grade_fn=_fake_grade,
        )


def _add_test_to_problem(fixture: dict, problem_id: str, test: dict) -> None:
    for problem in fixture["problems"]:
        if problem["problem_id"] == problem_id:
            problem["tests"].append(dict(test))
    pool = fixture["pool"]
    for path in (pool / "problems.jsonl", pool / "split_pilot_tfit.jsonl"):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        for row in rows:
            if row["problem_id"] == problem_id:
                row["tests"].append(dict(test))
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_ledger_streams_per_test_and_a_midway_crash_loses_only_the_in_flight_test(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    _add_test_to_problem(fixture, "tfit:c0", _stdin_test("98"))
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    ledger = tmp_path / "stream-ledger.jsonl"
    calls: dict[str, int] = {}

    def crashing_grade(code, tests, timeout, **kwargs):
        if "tfit:c0:0" in code:
            calls[code] = calls.get(code, 0) + 1
            if calls[code] == 2:
                raise KeyboardInterrupt("simulated host crash midway")
        return _fake_grade(code, tests, timeout)

    with pytest.raises(KeyboardInterrupt, match="midway"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            ledger_path=ledger,
            grade_fn=crashing_grade,
            workers=1,
            max_infrastructure_retries=1,
            recovery_check=_recovery_ok,
            resume=False,
            authorization=_test_authorization(tmp_path, fixture["lite"]),
        )
    records = shortfall._read_attempt_ledger(ledger)
    grouped = shortfall._ledger_entries_by_test(records)
    request_id = "tfit:c0:0"
    test0 = grouped[(request_id, 0)]
    test1 = grouped[(request_id, 1)]
    test2 = grouped.get((request_id, 2), [])
    assert [row["kind"] for row in test0] == [
        shortfall.LEDGER_ATTEMPT_START,
        shortfall.LEDGER_ATTEMPT_RESULT,
    ]
    assert [row["kind"] for row in test1] == [shortfall.LEDGER_ATTEMPT_START]
    assert test2 == []
    # The actual container identity was committed before dispatch.
    assert test0[0]["container_name"].startswith("gemmaloop-")
    assert test1[0]["container_name"].startswith("gemmaloop-")
    assert test0[0]["container_name"] != test1[0]["container_name"]

    dispatched: list[str] = []
    recovery_seen: list[dict] = []

    def recovery(**kwargs):
        recovery_seen.append(kwargs)
        # An interrupted in-flight attempt's cleanup must be bound to its
        # actual recorded container.
        return _cleanup_recovery(str(kwargs.get("container_name")))

    def resume_grade(code, tests, timeout, **kwargs):
        if "tfit:c0:0" in code:
            dispatched.append(str(tests[0].get("input")))
        return _fake_grade(code, tests, timeout)

    entries = shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        ledger_path=ledger,
        grade_fn=resume_grade,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=recovery,
        resume=True,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    # Test 0's durable result is never regraded; the interrupted test is the
    # only one that consumes its retry; test 2 starts for the first time.
    assert dispatched == ["3", "98"]
    assert any(item.get("container_name") for item in recovery_seen)
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    rows = {row["index"]: row for row in target["per_test"]}
    assert len(rows[0]["attempts"]) == 1
    assert rows[1]["attempts"][0].get("interrupted") is True
    assert len(rows[1]["attempts"]) == 2
    assert len(rows[2]["attempts"]) == 1


def test_malformed_middle_ledger_record_is_refused_without_rewriting():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.jsonl"
        first = shortfall._encode_ledger_record({"kind": "first"})
        third = shortfall._encode_ledger_record({"kind": "third"})
        malformed = b"GML-L1 3 not-a-checksum {}\n"
        original = first + malformed + third
        ledger.write_bytes(original)
        with pytest.raises(ValueError, match="malformed ledger record"):
            shortfall._read_attempt_ledger(ledger)
        # The valid record after the malformed one is neither discarded nor
        # the file rewritten.
        assert ledger.read_bytes() == original


# ---------------------------------------------------------------------------
# Fix round 7: complete-ledger refusal, per-attempt cleanup/recovery, and
# authorization for every original-grading route.
# ---------------------------------------------------------------------------
def test_complete_invalid_final_ledger_record_is_refused_not_truncated(
    tmp_path: Path,
):
    ledger = tmp_path / "complete-invalid-final.jsonl"
    valid = shortfall._encode_ledger_record({"kind": "first"})
    # A complete, newline-terminated but invalid record is committed history,
    # not a recoverable torn tail.
    invalid = b"GML-L1 3 not-a-checksum {}\n"
    original = valid + invalid
    ledger.write_bytes(original)
    with pytest.raises(ValueError, match="malformed ledger record"):
        shortfall._read_attempt_ledger(ledger)
    assert ledger.read_bytes() == original


def test_loader_requires_verified_cleanup_for_every_attempt(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    plan_path = _plan(fixture, tmp_path, freeze_path)
    calls: dict[tuple[str, str], int] = {}

    def grade_fn(code, tests, timeout, **kwargs):
        key = (code, str(tests[0].get("input")))
        calls[key] = calls.get(key, 0) + 1
        if calls[key] == 1:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    out = _count_out(tmp_path, "count-r7-attempt-cleanup")
    shortfall.count_verified(
        freeze_path=freeze_path,
        interpretation_path=_interpretation(fixture, tmp_path, freeze_path),
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        topup_plan_path=plan_path,
        out_dir=out,
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=_recovery_ok,
        authorization=_test_authorization(
            tmp_path, fixture["lite"], out_dir=out
        ),
    )

    def unverify_first_attempt(rows):
        for row in rows:
            for test in row.get("per_test") or []:
                attempts = test.get("attempts") or []
                if len(attempts) == 2:
                    # The final row still reports verified cleanup; the first
                    # attempt's own container cleanup is unverified.
                    attempts[0]["cleanup_verified"] = False
                    assert test.get("cleanup_verified") is True
                    return
        raise AssertionError("fixture produced no retried test")

    _rewrite_count_per_request(out, unverify_first_attempt)
    with pytest.raises(ValueError, match="unverified cleanup"):
        shortfall.load_verified_count(
            out, freeze_path=freeze_path, topup_plan_path=plan_path
        )


def test_terminal_disposition_requires_cleanup_for_every_attempt():
    def row(attempt_cleanups: list[bool]) -> dict:
        return {
            "index": 0,
            "status": "infrastructure_error",
            "passed": False,
            "infrastructure_failure": True,
            "cleanup_verified": True,
            "timed_out": False,
            "error": "cleanup failed",
            "attempts": [
                {
                    "status": "infrastructure_error",
                    "error": "first",
                    "cleanup_verified": cleanup,
                }
                for cleanup in attempt_cleanups
            ],
        }

    blocked = [row([False, True])]
    shortfall._apply_terminal_infrastructure(blocked, 1)
    assert blocked[0].get("terminal_infrastructure") is not True

    terminal = [row([True, True])]
    shortfall._apply_terminal_infrastructure(terminal, 1)
    assert terminal[0]["terminal_infrastructure"] is True


def test_retry_recovery_is_per_item_and_bound_to_its_own_container(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    ledger = tmp_path / "per-item-recovery.jsonl"
    calls: dict[tuple[str, str], int] = {}
    recovery_calls: list[dict] = []

    def grade_fn(code, tests, timeout, **kwargs):
        key = (code, str(tests[0].get("input")))
        calls[key] = calls.get(key, 0) + 1
        if calls[key] == 1:
            return _result(_infra_tests(tests), passed=False)
        return _fake_grade(code, tests, timeout)

    def recovery(**kwargs):
        recovery_calls.append(dict(kwargs))
        return {
            "check": "test recovery",
            "checked_utc": "2026-11-06T12:00:00+00:00",
            "ok": True,
            "container_name": kwargs.get("container_name"),
        }

    shortfall.grade_original_requests(
        freeze,
        pool_dir=fixture["pool"],
        lite_dir=fixture["lite"],
        grade_fn=grade_fn,
        workers=1,
        max_infrastructure_retries=1,
        recovery_check=recovery,
        ledger_path=ledger,
        authorization=_test_authorization(tmp_path, fixture["lite"]),
    )
    records = shortfall._read_attempt_ledger(ledger)
    first_attempt_container = {
        (str(record["request_id"]), int(record["test_index"]), int(record["attempt"])): record.get(
            "container_name"
        )
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_START
    }
    retried = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
        and record.get("recovery_evidence")
    ]
    assert retried
    # Two tests on tfit:c0 both need retries, so the first (broken) behaviour
    # reused one container's recovery for both items.  Each result must carry
    # recovery bound to its own first-attempt container.
    seen_containers = set()
    for record in retried:
        key = (str(record["request_id"]), int(record["test_index"]), 1)
        actual = first_attempt_container.get(key)
        evidence = record["recovery_evidence"]
        assert actual is not None
        assert evidence.get("container_name") == actual
        seen_containers.add(actual)
    assert len(seen_containers) == len(retried)


def test_supplied_wrapper_around_real_grader_is_not_authorization(
    tmp_path: Path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    freeze_path = _freeze(fixture, tmp_path)
    freeze = shortfall.load_freeze_manifest(freeze_path)
    reached: list[str] = []

    def fake_default(*args, **kwargs):
        reached.append("docker")
        raise AssertionError("the unbound wrapper must be refused before Docker")

    monkeypatch.setattr(shortfall, "_grade_code_default", fake_default)

    def wrapper(code, tests, timeout, **kwargs):
        return fake_default(code, tests, timeout=timeout, **kwargs)

    with pytest.raises(shortfall.ShortfallV3RunError, match="claimed official"):
        shortfall.grade_original_requests(
            freeze,
            pool_dir=fixture["pool"],
            lite_dir=fixture["lite"],
            grade_fn=wrapper,
            ledger_path=tmp_path / "unbound-wrapper-ledger.jsonl",
        )
    assert reached == []


# ---------------------------------------------------------------------------
# Fix round 8: container-bound cleanup recovery for interrupted attempts.
# ---------------------------------------------------------------------------
def _cleanup_recovery(container_name: str) -> dict:
    return {
        "check": "docker recovery: cleanup",
        "failure_classes": ["cleanup"],
        "checked_utc": "2026-11-06T12:00:00+00:00",
        "image": "python:3.12-slim",
        "container_name": container_name,
        "ok": True,
        "detail": "all required recovery checks passed",
        "checks": {
            "cleanup_removal": {
                "command": ["docker", "rm", "-f", container_name],
                "returncode": 0,
                "stdout": container_name,
                "stderr": "",
                "ok": True,
                "removed": True,
                "removal_probe": {
                    "returncode": 1,
                    "detail": "Error: No such container",
                },
            }
        },
    }


def _append_interrupted_start(ledger: Path, request_id: str, problem_id: str,
                              tests: list[dict], code: str, *,
                              attempt: int, container_name: str,
                              attempt_token: str) -> None:
    shortfall._append_attempt_ledger(
        ledger,
        {
            "kind": shortfall.LEDGER_ATTEMPT_START,
            "request_id": request_id,
            "problem_id": problem_id,
            "test_index": 0,
            "attempt": attempt,
            "request_sha256": shortfall._sha256_text(code),
            "test_sha256": shortfall._test_identity_sha256(tests[0]),
            "container_name": container_name,
            "attempt_token": attempt_token,
            "started_utc": shortfall._utc_now(),
        },
    )


def test_interrupted_attempt_cleanup_recovery_accepts_the_conclusive_retry(
    tmp_path: Path,
):
    """Reviewer probe: attempt cleanup [False, True] with verified recovery."""

    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "cleanup-recovery.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    container = "gemmaloop-r8-recovered"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=container,
        attempt_token="token-r8-recovered",
    )
    recovery_seen: list[dict] = []

    def recovery(**kwargs):
        recovery_seen.append(dict(kwargs))
        return _cleanup_recovery(str(kwargs.get("container_name")))

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    interrupted = test["attempts"][0]
    assert interrupted["interrupted"] is True
    # The interrupted attempt keeps its infrastructure outcome and its own
    # cleanup remains unverified ([False, True] across the two attempts) ...
    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["cleanup_verified"] is False
    # ... but acceptance comes from the separate recovery record.
    assert interrupted["cleanup_recovery"]["container_name"] == container
    assert len(test["attempts"]) == 2
    assert target["verified"] is True
    assert not target.get("coverage_error")
    assert any(call.get("container_name") == container for call in recovery_seen)

    records = shortfall._read_attempt_ledger(ledger)
    interrupted_records = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_INTERRUPTED
        and str(record.get("request_id")) == request_id
        and int(record.get("test_index", -1)) == 0
        and int(record.get("attempt") or 1) == 1
    ]
    assert len(interrupted_records) == 1
    # The interrupted record was never rewritten: its own cleanup stays false.
    assert interrupted_records[0]["evidence"]["cleanup_verified"] is False
    recovered = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
    ]
    assert len(recovered) == 1
    assert recovered[0]["request_id"] == request_id
    assert int(recovered[0]["test_index"]) == 0
    assert int(recovered[0]["attempt"]) == 1
    assert recovered[0]["container_name"] == container
    assert recovered[0] is not interrupted_records[0]


def test_unverifiable_interrupted_container_keeps_blocking(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "cleanup-unverifiable.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    container = "gemmaloop-r8-unverifiable"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=container,
        attempt_token="token-r8-unverifiable",
    )

    def wrong_container_recovery(**kwargs):
        # The checker reports removal of a different container, so it is not
        # bound to this attempt's recorded container.
        return _cleanup_recovery("gemmaloop-r8-someone-else")

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture,
        freeze,
        ledger,
        passing,
        resume=True,
        recovery_check=wrong_container_recovery,
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert test["attempts"][0]["interrupted"] is True
    assert test["attempts"][0]["cleanup_verified"] is False
    # The unbound check never unlocks the retry or a cleanup acceptance.
    assert len(test["attempts"]) == 1
    assert target["verified"] is False
    assert "unverified cleanup" in (target.get("coverage_error") or "")
    records = shortfall._read_attempt_ledger(ledger)
    assert not any(
        record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
        for record in records
    )
    assert any(
        record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERY_FAILED
        and record.get("container_name") == container
        for record in records
    )


def test_interrupted_retry_cleanup_recovery_becomes_terminal(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "interrupted-retry-recovery.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    first_container = "gemmaloop-r8-retry-first"
    retry_container = "gemmaloop-r8-retry-second"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=first_container,
        attempt_token="token-r8-retry-first",
    )
    shortfall._append_attempt_ledger(
        ledger,
        shortfall._ledger_result_record(
            request_id=request_id,
            problem_id=problem_id,
            test_index=0,
            attempt=1,
            evidence={
                "status": "infrastructure_error",
                "passed": False,
                "infrastructure_failure": True,
                "cleanup_verified": True,
                "container_name": first_container,
                "error": "first unknown",
            },
        ),
    )
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=2,
        container_name=retry_container,
        attempt_token="token-r8-retry-second",
    )

    def recovery(**kwargs):
        return _cleanup_recovery(str(kwargs.get("container_name")))

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "infrastructure_error",
    ]
    retry = test["attempts"][1]
    assert retry["interrupted"] is True
    assert retry["cleanup_verified"] is False
    assert retry["cleanup_recovery"]["container_name"] == retry_container
    # The retry allowance is exhausted; recovered cleanup allows the terminal
    # infrastructure-unverified disposition instead of permanent blocking.
    assert test.get("terminal_infrastructure") is True
    assert target["terminal_infrastructure"] is True
    assert target["blocking_infrastructure"] is False
    records = shortfall._read_attempt_ledger(ledger)
    assert all(int(record.get("attempt") or 1) <= 2 for record in records)
    recovered = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
    ]
    assert len(recovered) == 1
    assert int(recovered[0]["attempt"]) == 2
    assert recovered[0]["container_name"] == retry_container


# ---------------------------------------------------------------------------
# Fix round 9: repeatable cleanup recovery for interrupted attempts.
# ---------------------------------------------------------------------------
def _failed_cleanup_recovery(container_name: str) -> dict:
    evidence = _cleanup_recovery(container_name)
    evidence["ok"] = False
    evidence["detail"] = "removal of the recorded container was not verified"
    check = evidence["checks"]["cleanup_removal"]
    check["ok"] = False
    check["removed"] = False
    check["returncode"] = 1
    check["removal_probe"] = {
        "returncode": 0,
        "detail": "container is still present",
    }
    return evidence


def _direct_interrupted_record(
    request_id: str,
    problem_id: str,
    *,
    attempt: int,
    container_name: str,
    attempt_token: str,
) -> dict:
    return shortfall._ledger_result_record(
        request_id=request_id,
        problem_id=problem_id,
        test_index=0,
        attempt=attempt,
        evidence={
            "status": "infrastructure_error",
            "passed": False,
            "infrastructure_failure": True,
            "exit_code": None,
            "timed_out": False,
            "duration_s": 0.0,
            "error": "attempt interrupted by host crash; no result was persisted",
            "stderr": "attempt interrupted by host crash; no result was persisted",
            "lifecycle_class": None,
            "lifecycle_valid": False,
            "cleanup_verified": False,
            "container_name": container_name,
            "lifecycle_attempt": attempt_token,
        },
        sandboxed=True,
        kind=shortfall.LEDGER_ATTEMPT_INTERRUPTED,
    )


def test_cleanup_recovery_rechecks_after_a_failed_check(tmp_path: Path):
    """Reviewer probe: first check fails, second verifies, retry passes."""

    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "cleanup-recheck.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    container = "gemmaloop-r9-recheck"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=container,
        attempt_token="token-r9-recheck",
    )
    calls: list[str] = []

    def recovery(**kwargs):
        name = str(kwargs.get("container_name"))
        calls.append(name)
        if len(calls) == 1:
            return _failed_cleanup_recovery(name)
        return _cleanup_recovery(name)

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)

    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "passed",
    ]
    interrupted = test["attempts"][0]
    assert interrupted["interrupted"] is True
    assert interrupted["cleanup_verified"] is False
    assert interrupted["cleanup_recovery"]["container_name"] == container
    assert test["attempts"][1]["cleanup_verified"] is True
    assert target["verified"] is True
    assert not target.get("coverage_error")
    assert len(calls) >= 2

    records = shortfall._read_attempt_ledger(ledger)
    interrupted_records = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_ATTEMPT_INTERRUPTED
    ]
    failed = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERY_FAILED
    ]
    recovered = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
    ]
    assert len(interrupted_records) == 1
    assert interrupted_records[0]["evidence"]["cleanup_verified"] is False
    assert failed and failed[0]["container_name"] == container
    assert len(recovered) == 1
    assert recovered[0]["container_name"] == container
    assert int(recovered[0]["attempt"]) == 1
    # No attempt was added beyond the registered single retry.
    assert all(int(record.get("attempt") or 1) <= 2 for record in records)


def test_failed_cleanup_recovery_keeps_the_count_blocked(tmp_path: Path):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "cleanup-still-pending.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    container = "gemmaloop-r9-still-pending"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=container,
        attempt_token="token-r9-still-pending",
    )

    def failing(**kwargs):
        return _failed_cleanup_recovery(str(kwargs.get("container_name")))

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=failing
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    assert target["verified"] is False
    assert "unverified cleanup" in (target.get("coverage_error") or "")
    records = shortfall._read_attempt_ledger(ledger)
    assert any(
        record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERY_FAILED
        for record in records
    )
    assert not any(
        record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
        for record in records
    )
    assert all(
        int(record.get("attempt") or 1) <= 1
        for record in records
        if record.get("kind")
        in {shortfall.LEDGER_ATTEMPT_START, shortfall.LEDGER_ATTEMPT_INTERRUPTED}
    )


def test_crash_between_interrupted_and_recovered_appends_is_recovered(
    tmp_path: Path,
):
    """Reviewer probe: interrupted append committed, recovery append lost."""

    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "cleanup-crash-between-appends.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    container = "gemmaloop-r9-crash-between"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=container,
        attempt_token="token-r9-crash-between",
    )
    # The interruption was committed; the host crashed before the separate
    # cleanup-recovered append.  The next pass must recover it.
    shortfall._append_attempt_ledger(
        ledger,
        _direct_interrupted_record(
            request_id,
            problem_id,
            attempt=1,
            container_name=container,
            attempt_token="token-r9-crash-between",
        ),
    )

    def recovery(**kwargs):
        return _cleanup_recovery(str(kwargs.get("container_name")))

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "passed",
    ]
    assert test["attempts"][0]["interrupted"] is True
    assert test["attempts"][0]["cleanup_verified"] is False
    assert test["attempts"][0]["cleanup_recovery"]["container_name"] == container
    assert target["verified"] is True
    assert not target.get("coverage_error")

    records = shortfall._read_attempt_ledger(ledger)
    assert (
        sum(
            1
            for record in records
            if record.get("kind") == shortfall.LEDGER_ATTEMPT_INTERRUPTED
        )
        == 1
    )
    recovered = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
    ]
    assert len(recovered) == 1
    assert recovered[0]["container_name"] == container
    assert int(recovered[0]["attempt"]) == 1

def test_interrupted_retry_cleanup_recovery_survives_a_crash_between_appends(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    freeze = shortfall.load_freeze_manifest(_freeze(fixture, tmp_path))
    ledger = tmp_path / "retry-crash-between-appends.jsonl"
    request_id, problem_id, tests, code = _first_code_key(fixture)
    first_container = "gemmaloop-r9-retry-crash-first"
    retry_container = "gemmaloop-r9-retry-crash-second"
    shortfall._ensure_attempt_ledger(ledger, {})
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=1,
        container_name=first_container,
        attempt_token="token-r9-retry-crash-first",
    )
    shortfall._append_attempt_ledger(
        ledger,
        shortfall._ledger_result_record(
            request_id=request_id,
            problem_id=problem_id,
            test_index=0,
            attempt=1,
            evidence={
                "status": "infrastructure_error",
                "passed": False,
                "infrastructure_failure": True,
                "cleanup_verified": True,
                "container_name": first_container,
                "error": "first unknown",
            },
        ),
    )
    _append_interrupted_start(
        ledger,
        request_id,
        problem_id,
        tests,
        code,
        attempt=2,
        container_name=retry_container,
        attempt_token="token-r9-retry-crash-second",
    )
    # The retry interruption was committed; the host crashed before the
    # separate cleanup-recovered append.
    shortfall._append_attempt_ledger(
        ledger,
        _direct_interrupted_record(
            request_id,
            problem_id,
            attempt=2,
            container_name=retry_container,
            attempt_token="token-r9-retry-crash-second",
        ),
    )

    def recovery(**kwargs):
        return _cleanup_recovery(str(kwargs.get("container_name")))

    def passing(code_, tests_, timeout, **kwargs):
        return _fake_grade("```python\nPASS\n```", tests_, timeout)

    entries = _ledger_grade(
        fixture, freeze, ledger, passing, resume=True, recovery_check=recovery
    )
    target = next(entry for entry in entries if entry["request_id"] == request_id)
    test = next(row for row in target["per_test"] if row["index"] == 0)
    assert [attempt["status"] for attempt in test["attempts"]] == [
        "infrastructure_error",
        "infrastructure_error",
    ]
    retry = test["attempts"][1]
    assert retry["interrupted"] is True
    assert retry["cleanup_verified"] is False
    assert retry["cleanup_recovery"]["container_name"] == retry_container
    assert test.get("terminal_infrastructure") is True
    assert target["terminal_infrastructure"] is True
    assert target["blocking_infrastructure"] is False
    records = shortfall._read_attempt_ledger(ledger)
    assert all(int(record.get("attempt") or 1) <= 2 for record in records)
    recovered = [
        record
        for record in records
        if record.get("kind") == shortfall.LEDGER_CLEANUP_RECOVERED
    ]
    assert len(recovered) == 1
    assert int(recovered[0]["attempt"]) == 2
    assert recovered[0]["container_name"] == retry_container

