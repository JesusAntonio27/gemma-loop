"""Mandatory synthetic, non-official real-Docker dry run for the v3r repair.

This script runs the *production* v3r orchestration
(:func:`gemmaloop.stage0.shortfall.run_v3r_pipeline`) and the production
default grading path (default count adapter -> real ``grade_code`` ->
``DockerSandbox.run_tests``) end to end on hand-authored synthetic, non-pilot
candidates:

* interpretation (synthetic pinned source records);
* verifier freeze (real image content digest);
* complete synthetic count with durable attempt ledger and per-request
  serialization;
* independent count loading;
* final corpus and freeze-index build.

It never reads or writes the official pool, lite run, v3/v3r registry or
official output directory.  A synthetic void receipt and continuity record
exercise the same one-time v3r guard without touching the official ones.

The receipt is written to ``docs/validation/v3r-dry-run-receipt.json`` and is
explicitly synthetic and non-official.

Usage::

    uv run python scripts/dry_run_v3r.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from gemmaloop.pilot.eval_rows import render_trace_rows  # noqa: E402
from gemmaloop.stage0 import grader, shortfall  # noqa: E402
from gemmaloop.stage0.sources import convert_row  # noqa: E402

import freeze_recount_v3r  # noqa: E402  (resolved from scripts/ on sys.path)

RECEIPT_PATH = REPO / "docs" / "validation" / "v3r-dry-run-receipt.json"
IMAGE = "python:3.12-slim"
T_FIT_PASSING = 20
SELECTION_PASSING = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _add_problem(
    *,
    index: int,
    split: str,
    kind: str,
    question: str,
    code: str,
    tests: list[dict[str, Any]],
    raw_by_index: dict[int, dict],
    problems: list[dict],
    split_ids: dict[str, list[str]],
) -> str:
    if kind == "functional":
        input_output = {
            "inputs": [json.loads(test["input"]) for test in tests],
            "outputs": [json.loads(test["output"]) for test in tests],
            "fn_name": tests[0]["fn_name"],
        }
    else:
        input_output = {
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
        }
    raw = {
        "problem_id": index,
        "question": question,
        "solutions": [code],
        "input_output": json.dumps(input_output),
    }
    problem = convert_row("apps", raw, index)
    assert problem is not None
    problem.update(
        {"split": "train", "pilot_subsplit": None, "bucket": 59}
    )
    raw_by_index[index] = raw
    problems.append(problem)
    split_ids[split].append(problem["problem_id"])
    return problem["problem_id"]


def build_fixture(root: Path) -> dict[str, Any]:
    NL = chr(10)
    pool = root / "pool"
    lite = root / "lite"
    (pool / "splits").mkdir(parents=True, exist_ok=True)
    (lite / "shards").mkdir(parents=True, exist_ok=True)
    (lite / "pilot").mkdir(parents=True, exist_ok=True)
    split_ids: dict[str, list[str]] = {
        "pilot_p1": [],
        "pilot_tfit": [],
        "pilot_selection": [],
    }
    problems: list[dict] = []
    raw_by_index: dict[int, dict] = {}
    completions: dict[str, str] = {}
    next_index = 0

    def add(
        split: str,
        kind: str,
        question: str,
        code: str,
        tests: list[dict[str, Any]],
    ) -> str:
        nonlocal next_index
        index = next_index
        next_index += 1
        problem_id = _add_problem(
            index=index,
            split=split,
            kind=kind,
            question=question,
            code=code,
            tests=tests,
            raw_by_index=raw_by_index,
            problems=problems,
            split_ids=split_ids,
        )
        completions[problem_id] = code
        return problem_id

    stdin_pass = "import sys" + NL + "sys.stdout.write(sys.stdin.read())"
    stdin_fail = 'print("wrong")'
    stdin_timeout = "while True:" + NL + "    pass"
    functional_src = "def solve(a, b):" + NL + "    return a + b"

    # P1 is descriptive only and never gates the count: one pass, one known
    # failure and one known timeout.  The timeout runs inside the same
    # production count so the wrapper enforcement route (A) is exercised end
    # to end and must not block the approved successful count.
    add(
        "pilot_p1",
        "stdin",
        "synthetic p1 pass",
        stdin_pass,
        [{"input": "p1-ok" + NL, "output": "p1-ok" + NL, "type": "stdin"}],
    )
    add(
        "pilot_p1",
        "stdin",
        "synthetic p1 known failure",
        stdin_fail,
        [{"input": "x" + NL, "output": "expected" + NL, "type": "stdin"}],
    )
    timeout_problem_id = add(
        "pilot_p1",
        "stdin",
        "synthetic p1 known timeout",
        stdin_timeout,
        [{"input": "", "output": "never" + NL, "type": "stdin"}],
    )

    # T-fit: 19 stdin passes + one functional pass.
    for number in range(T_FIT_PASSING - 1):
        value = "tfit-" + str(number)
        add(
            "pilot_tfit",
            "stdin",
            "synthetic tfit " + str(number),
            stdin_pass,
            [{"input": value + NL, "output": value + NL, "type": "stdin"}],
        )
    add(
        "pilot_tfit",
        "functional",
        "synthetic tfit functional",
        functional_src,
        [
            {
                "input": "[1, 2]",
                "output": "3",
                "type": "functional",
                "fn_name": "solve",
            }
        ],
    )

    # Selection: only passing stdin candidates so the synthetic count clears
    # the unchanged extension thresholds and actually reaches corpus/index.
    for number in range(SELECTION_PASSING):
        value = "selection-" + str(number)
        add(
            "pilot_selection",
            "stdin",
            "synthetic selection " + str(number),
            stdin_pass,
            [{"input": value + NL, "output": value + NL, "type": "stdin"}],
        )

    _write_jsonl(pool / "problems.jsonl", problems)
    for name in ("pilot_p1", "pilot_tfit", "pilot_selection"):
        rows = [row for row in problems if row["problem_id"] in set(split_ids[name])]
        _write_jsonl(pool / ("split_" + name + ".jsonl"), rows)
    (pool / "manifest.json").write_text("{}", encoding="utf-8")

    request_ids = [row["problem_id"] + ":0" for row in problems]
    (lite / "request_ids.txt").write_text(
        "".join(request_id + NL for request_id in request_ids),
        encoding="utf-8",
    )
    (lite / "run_marker.json").write_text(
        json.dumps({"run_id": "synthetic-v3r-dry-run"}), encoding="utf-8"
    )
    (lite / "manifest.json").write_text("{}", encoding="utf-8")
    (lite / "pilot" / "manifest.json").write_text(
        json.dumps({"model_revision": "synthetic"}), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for row in problems:
        request_id = row["problem_id"] + ":0"
        code = completions[row["problem_id"]]
        content = "```python" + NL + code + NL + "```"
        records.append(
            {
                "problem_id": row["problem_id"],
                "sample_index": 0,
                "request_id": request_id,
                "prompt": row["prompt"],
                "completion": ("step " * 20) + content,
                "reasoning": "step " * 20,
                "content": content,
                "finish_reason": "stop",
                "thought_closed": True,
                "tokens": 80,
                "prompt_tokens": 40,
                "engine": "synthetic",
                "model_revision": "synthetic",
                "shard_index": 0,
            }
        )
        grades.append(
            {
                "request_id": request_id,
                "problem_id": row["problem_id"],
                "sample_index": 0,
                "kind": "code",
                "split": row["split"],
                "kept": True,
                "drop_reason": None,
                "correct": None,
                "rendered_tokens": 120,
                "reasoning_share": 0.8,
                "timing_s": 0.0,
            }
        )
    _write_jsonl(lite / "shards" / "shard_00000.jsonl", records)
    _write_jsonl(lite / "sample_grades.jsonl", grades)

    def record_loader(source: str, needed: set[int]):
        assert source == "apps"
        for index in sorted(needed):
            yield index, raw_by_index[index]

    return {
        "root": root,
        "pool": pool,
        "lite": lite,
        "problems": problems,
        "request_ids": request_ids,
        "raw_by_index": raw_by_index,
        "record_loader": record_loader,
        "timeout_problem_id": timeout_problem_id,
    }


def _timeout_exercise(
    count_dir: Path, fixture: dict[str, Any]
) -> dict[str, Any]:
    """Read the known timeout that ran inside the production count.

    The timeout is graded by the same ``_grade_code_default -> grade_code ->
    DockerSandbox.run_tests`` chain, the frozen 6 s outer timeout and the real
    lifecycle wrapper as every other candidate.  Under the registered
    lifecycle ruling a wrapper-observed timeout is conclusive through route A,
    so it is part of the successful count rather than a blocking probe.
    """

    rows = _read_jsonl(count_dir / shortfall.COUNT_PER_REQUEST_NAME)
    timeout_problem_id = str(fixture["timeout_problem_id"])
    request_row = next(
        row for row in rows if str(row.get("problem_id")) == timeout_problem_id
    )
    row = (request_row.get("per_test") or [{}])[0]
    record = row.get("deadline_enforcement") or {}
    deadline = row.get("host_deadline_monotonic")
    host_exit = row.get("host_exit_monotonic")
    try:
        host_exit_within_bound = (
            deadline is not None
            and host_exit is not None
            and float(host_exit)
            <= float(deadline) + float(grader.LIFECYCLE_NOTICE_GRACE_S) + 1e-6
        )
    except (TypeError, ValueError):
        host_exit_within_bound = False
    return {
        "expected": "candidate running at the registered 6 s outer deadline",
        "problem_id": timeout_problem_id,
        "request_id": request_row.get("request_id"),
        "counted_timeouts": sum(
            1 for item in rows if item.get("failure_class") == "timeout"
        ),
        "failure_class": request_row.get("failure_class"),
        "observed_status": row.get("status"),
        "infrastructure_failure": bool(row.get("infrastructure_failure")),
        "timed_out": bool(row.get("timed_out")),
        "lifecycle_class": row.get("lifecycle_class"),
        "lifecycle_valid": row.get("lifecycle_valid"),
        "enforcement_route": row.get("enforcement_route"),
        "deadline_enforcement_route": record.get("enforcement_route"),
        "host_deadline_monotonic": deadline,
        "host_exit_monotonic": host_exit,
        "host_exit_within_bound": bool(host_exit_within_bound),
        "wrapper_kill_verified": bool(record.get("wrapper_kill_verified")),
        "start_received_within_budget": row.get(
            "start_received_within_budget"
        ),
        "deadline_alive": row.get("deadline_alive"),
        "termination_verified": row.get("termination_verified"),
        "cleanup_verified": row.get("cleanup_verified"),
        "container_name": row.get("container_name"),
        "lifecycle_attempt": row.get("lifecycle_attempt"),
        "identity_agreement": bool(row.get("container_name"))
        and bool(row.get("lifecycle_attempt")),
        "count_coverage_policy": (
            "conclusive via the wrapper enforcement route (A): the frozen "
            "wrapper received the authenticated deadline notice, observed the "
            "identified child alive, killed it and exited before the host "
            "enforcement window; the host observed the process exit inside "
            "deadline + 0.25 s and verified cleanup, so no host enforcement "
            "action is required"
        ),
    }


def _install_synthetic_void_continuity(fixture: dict[str, Any]) -> dict[str, Any]:
    """A synthetic void identity + continuity, never the official artifacts."""

    lite = fixture["lite"]
    shortfall_dir = lite / "shortfall"
    v3 = shortfall_dir / "v3"
    (v3 / "verifier_v3").mkdir(parents=True, exist_ok=True)
    (v3 / "count_v3").mkdir(parents=True, exist_ok=True)
    registry = {
        "kind": shortfall.V3_REGISTRY_KIND,
        "version": 1,
        "status": shortfall.V3_STATUS_CRASHED,
        "official": True,
        "created_utc": "2026-11-06T00:00:00+00:00",
    }
    (shortfall_dir / shortfall.V3_REGISTRY_NAME).write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (v3 / "verifier_v3" / shortfall.FREEZE_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.COUNT_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    (v3 / "count_v3" / shortfall.ATTEMPT_LEDGER_NAME).write_text("", encoding="utf-8")
    ruling = fixture["root"] / "synthetic-void-ruling.md"
    ruling.write_text(
        "SYNTHETIC DRY-RUN ONLY: represents codex-ruling-v3-void.md.\n",
        encoding="utf-8",
    )
    receipt = shortfall.write_void_receipt(
        lite,
        ruling_path=ruling,
        enforce_known_hashes=False,
        void_out_root=v3,
    )
    shortfall.write_v3r_continuity(lite, ruling_path=ruling)
    review = shortfall.record_v3r_replacement_review(
        lite,
        reviewer="synthetic-dry-run",
        implementation={"sha256": "synthetic"},
        require_official=False,
    )
    return {
        "void_receipt": shortfall.void_receipt_path(lite),
        "void_receipt_sha256": _sha256(shortfall.void_receipt_path(lite)),
        "continuity": shortfall.v3r_continuity_path(lite),
        "continuity_sha256": _sha256(shortfall.v3r_continuity_path(lite)),
        "review": shortfall.v3r_review_path(lite),
        "registry": shortfall.v3_registry_path(lite),
        "receipt_payload_sha256": receipt["receipt_sha256"],
        "continuity_review": review["replacement_review"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _docker_environment() -> dict[str, Any]:
    def run(args: list[str]) -> dict[str, Any]:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
        }

    return {
        "docker_version": run(["docker", "version", "--format", "{{json .}}"]),
        "docker_info": run(["docker", "info", "--format", "{{.ServerVersion}}"]),
    }


def _audit_identity(count_dir: Path) -> dict[str, Any]:
    records = shortfall._read_attempt_ledger(
        count_dir / shortfall.ATTEMPT_LEDGER_NAME
    )
    starts = {
        (r["request_id"], int(r["test_index"]), int(r["attempt"])): r
        for r in records
        if r.get("kind") == shortfall.LEDGER_ATTEMPT_START
    }
    results = [
        r for r in records if r.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
    ]
    per_request = _read_jsonl(count_dir / shortfall.COUNT_PER_REQUEST_NAME)
    problems: list[str] = []
    for result in results:
        key = (
            result["request_id"],
            int(result["test_index"]),
            int(result["attempt"]),
        )
        start = starts.get(key)
        if start is None:
            problems.append(f"missing start for {key}")
            continue
        evidence = result.get("evidence") or {}
        if evidence.get("container_name") != start.get("container_name"):
            problems.append(f"container mismatch for {key}")
        if evidence.get("lifecycle_attempt") != start.get("attempt_token"):
            problems.append(f"attempt token mismatch for {key}")
        if evidence.get("lifecycle_valid") is not True:
            problems.append(f"invalid lifecycle for {key}: {evidence.get('lifecycle_class')}")
        if not evidence.get("lifecycle_frames_raw"):
            problems.append(f"no raw lifecycle frames for {key}")
    for row in per_request:
        for test in row.get("per_test") or []:
            if test.get("container_name") is None:
                problems.append(
                    f"per-request row lacks container for {row['request_id']} test {test.get('index')}"
                )
            if test.get("lifecycle_attempt") is None:
                problems.append(
                    f"per-request row lacks attempt token for {row['request_id']} test {test.get('index')}"
                )
    attempts = sorted({int(r["attempt"]) for r in records if r.get("kind") == shortfall.LEDGER_ATTEMPT_START})
    return {
        "starts": len(starts),
        "results": len(results),
        "attempts_seen": attempts,
        "first_attempt_is_one": bool(attempts) and attempts[0] == 1,
        "max_attempt_within_policy": max(attempts, default=0) <= 2,
        "all_attempts_one": attempts == [1],
        "identity_agreement": not problems,
        "identity_problems": problems,
        "start_records": [
            {
                "request_id": start["request_id"],
                "test_index": int(start["test_index"]),
                "attempt": int(start["attempt"]),
                "container_name": start["container_name"],
                "attempt_token": start["attempt_token"],
            }
            for start in starts.values()
        ],
    }


def _checks(
    *,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    loaded: Mapping[str, Any],
    audit: Mapping[str, Any],
    fixture: dict[str, Any],
    corpus_manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    continuity: Mapping[str, Any],
    timeout_exercise: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": "v3r registry claimed and complete",
            "expected": shortfall.V3_STATUS_COMPLETE,
            "actual": result.get("registry_status"),
            "ok": result.get("registry_status") == shortfall.V3_STATUS_COMPLETE,
        },
        {
            "name": "count status complete",
            "expected": shortfall.COUNT_STATUS_COMPLETE,
            "actual": summary.get("status"),
            "ok": summary.get("status") == shortfall.COUNT_STATUS_COMPLETE,
        },
        {
            "name": "independent count loader agrees",
            "expected": summary.get("run_fingerprint"),
            "actual": loaded.get("run_fingerprint"),
            "ok": loaded.get("run_fingerprint") == summary.get("run_fingerprint")
            and loaded.get("accounting") == summary.get("accounting"),
        },
        {
            "name": "v3r attempts start at 1",
            "expected": [1],
            "actual": audit.get("attempts_seen"),
            "ok": audit.get("first_attempt_is_one") is True,
        },
        {
            "name": "single registered retry not exceeded",
            "expected": True,
            "actual": audit.get("max_attempt_within_policy"),
            "ok": audit.get("max_attempt_within_policy") is True,
        },
        {
            "name": "container names and attempt tokens agree with lifecycle/ledger",
            "expected": True,
            "actual": audit.get("identity_problems"),
            "ok": audit.get("identity_agreement") is True,
        },
        {
            "name": "all kept code requests were executed",
            "expected": len(fixture["request_ids"]),
            "actual": summary.get("accounting", {}).get("code_requests_graded"),
            "ok": int(summary.get("accounting", {}).get("code_requests_graded") or 0)
            == len(fixture["request_ids"]),
        },
        {
            "name": "known passing candidates verified",
            "expected": T_FIT_PASSING + SELECTION_PASSING + 1,
            "actual": summary.get("accounting", {}).get("code_requests_passed"),
            "ok": int(summary.get("accounting", {}).get("code_requests_passed") or 0)
            == T_FIT_PASSING + SELECTION_PASSING + 1,
        },
        {
            "name": "known failing candidate accounted as a candidate failure",
            "expected": 1,
            "actual": summary.get("accounting", {}).get("answer_failures"),
            "ok": int(summary.get("accounting", {}).get("answer_failures") or 0) == 1,
        },
        {
            "name": "production default path identifies a candidate timeout",
            "expected": "candidate_running_at_deadline",
            "actual": timeout_exercise.get("lifecycle_class"),
            "ok": bool(timeout_exercise.get("timed_out"))
            and timeout_exercise.get("lifecycle_class")
            == "candidate_running_at_deadline"
            and timeout_exercise.get("identity_agreement") is True,
        },
        {
            "name": "timeout classification is a registered outer-deadline timeout",
            "expected": "timeout",
            "actual": timeout_exercise.get("observed_status"),
            "ok": timeout_exercise.get("observed_status") == "timeout"
            and timeout_exercise.get("infrastructure_failure") is False,
        },
        {
            "name": "timeout is conclusive through the wrapper enforcement route",
            "expected": grader.ENFORCEMENT_ROUTE_WRAPPER,
            "actual": timeout_exercise.get("enforcement_route"),
            "ok": timeout_exercise.get("enforcement_route")
            == grader.ENFORCEMENT_ROUTE_WRAPPER
            and timeout_exercise.get("deadline_enforcement_route")
            == grader.ENFORCEMENT_ROUTE_WRAPPER
            and timeout_exercise.get("wrapper_kill_verified") is True
            and timeout_exercise.get("host_exit_within_bound") is True
            and timeout_exercise.get("start_received_within_budget") is True
            and timeout_exercise.get("deadline_alive") is True
            and timeout_exercise.get("termination_verified") is True
            and timeout_exercise.get("cleanup_verified") is True,
        },
        {
            "name": "timeout inside the count does not block it",
            "expected": [shortfall.COUNT_STATUS_COMPLETE, 1],
            "actual": [
                summary.get("status"),
                timeout_exercise.get("counted_timeouts"),
            ],
            "ok": summary.get("status") == shortfall.COUNT_STATUS_COMPLETE
            and int(summary.get("accounting", {}).get("timeouts") or 0) == 1
            and timeout_exercise.get("counted_timeouts") == 1
            and timeout_exercise.get("failure_class") == "timeout"
            and timeout_exercise.get("infrastructure_failure") is False,
        },
        {
            "name": "T-fit / selection distinct verified counts clear thresholds",
            "expected": [T_FIT_PASSING, SELECTION_PASSING],
            "actual": [summary.get("nT0"), summary.get("nS0")],
            "ok": int(summary.get("nT0") or 0) == T_FIT_PASSING
            and int(summary.get("nS0") or 0) == SELECTION_PASSING,
        },
        {
            "name": "synthetic continuity links the void receipt",
            "expected": bool((continuity.get("void_receipt") or {}).get("sha256")),
            "actual": (continuity.get("void_receipt") or {}).get("sha256"),
            "ok": bool((continuity.get("void_receipt") or {}).get("sha256"))
            and continuity.get("replacement_review", {}).get("status") == "recorded",
        },
        {
            "name": "corpus manifest built",
            "expected": True,
            "actual": bool(corpus_manifest.get("manifest_sha256")),
            "ok": bool(corpus_manifest.get("manifest_sha256")),
        },
        {
            "name": "freeze index built",
            "expected": True,
            "actual": bool(index.get("index_sha256")),
            "ok": bool(index.get("index_sha256")),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=REPO / ".v3r-dry-run",
        help="synthetic-only workspace (default: .v3r-dry-run)",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=RECEIPT_PATH,
        help="synthetic validation receipt path",
    )
    parser.add_argument("--keep", action="store_true", help="keep the workspace")
    args = parser.parse_args(argv)

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    official = Path(
        os.environ.get(
            "GEMMALOOP_ARTIFACTS",
            str(Path.home() / "ML" / "gemmaloop-artifacts"),
        )
    ) / "stage0" / "modal_lite" / "shortfall"
    official_before = {
        name: _sha256(official / name)
        for name in (
            "v3_registry.json",
            "v3/verifier_v3/verifier_freeze.json",
            "v3/count_v3/count_manifest.json",
            "v3/count_v3/attempt_ledger.jsonl",
        )
        if (official / name).is_file()
    }

    started = time.time()
    fixture = build_fixture(workdir)
    synthetic = _install_synthetic_void_continuity(fixture)
    pin = shortfall.resolve_docker_image(IMAGE)
    out_root = fixture["lite"] / "shortfall" / "v3r"

    command = [
        "uv",
        "run",
        "--offline",
        "python",
        "scripts/dry_run_v3r.py",
        *(["--keep"] if args.keep else []),
    ]
    # Exercise the production v3r CLI itself: freeze_recount_v3r.main() runs
    # the guard, image-digest verification, tokenizer load and the production
    # run_v3r_pipeline.  Only paths/registry are redirected to the synthetic
    # fixture, plus the offline synthetic source-record loader.
    cli_argv = [
        "--repo-root",
        str(REPO),
        "--pool-dir",
        str(fixture["pool"]),
        "--lite-dir",
        str(fixture["lite"]),
        "--request-ids",
        str(fixture["lite"] / "request_ids.txt"),
        "--out-root",
        str(out_root),
        "--image",
        IMAGE,
        "--workers",
        "1",
        "--max-length",
        "8192",
    ]
    cli_returncode = freeze_recount_v3r.main(
        cli_argv, record_loader=fixture["record_loader"]
    )
    if cli_returncode != 0:
        raise RuntimeError(
            "the production v3r CLI main() failed with exit "
            f"{cli_returncode}"
        )
    count_dir = out_root / shortfall.V3R_STAGE_DIRECTORIES["count"]
    freeze_path = (
        out_root
        / shortfall.V3R_STAGE_DIRECTORIES["freeze"]
        / shortfall.FREEZE_MANIFEST_NAME
    )
    plan_path = out_root / shortfall.V3R_PLAN_NAME
    summary = shortfall.load_verified_count(
        count_dir,
        topup_plan_path=plan_path,
        freeze_path=freeze_path,
    )
    registry_candidate = shortfall.v3r_registry_path(fixture["lite"])
    result = {
        "interpretation": str(
            out_root
            / shortfall.V3R_STAGE_DIRECTORIES["interpretation"]
            / shortfall.INTERPRETATION_MANIFEST_NAME
        ),
        "freeze": str(freeze_path),
        "count": str(count_dir / shortfall.COUNT_SUMMARY_NAME),
        "plan": str(plan_path),
        "corpus": str(out_root / shortfall.V3R_STAGE_DIRECTORIES["corpus"]),
        "index": str(out_root / shortfall.V3R_INDEX_NAME),
        "registry": str(registry_candidate),
        "registry_status": json.loads(
            registry_candidate.read_text(encoding="utf-8")
        ).get("status"),
    }
    loaded = shortfall.load_verified_count(
        count_dir,
        topup_plan_path=plan_path,
        freeze_path=freeze_path,
    )
    audit = _audit_identity(count_dir)
    timeout_exercise = _timeout_exercise(count_dir, fixture)
    corpus_manifest = shortfall.load_final_corpus_manifest(Path(result["corpus"]) / "manifest.json")
    index = json.loads(Path(result["index"]).read_text(encoding="utf-8"))
    continuity = json.loads(synthetic["continuity"].read_text(encoding="utf-8"))
    registry = json.loads(Path(result["registry"]).read_text(encoding="utf-8"))

    checks = _checks(
        result=result,
        summary=summary,
        loaded=loaded,
        audit=audit,
        fixture=fixture,
        corpus_manifest=corpus_manifest,
        index=index,
        continuity=continuity,
        timeout_exercise=timeout_exercise,
    )
    official_after = {
        name: _sha256(official / name)
        for name in official_before
        if (official / name).is_file()
    }
    checks.append(
        {
            "name": "official v3 artifacts untouched",
            "expected": official_before,
            "actual": official_after,
            "ok": official_before == official_after,
        }
    )
    checks.append(
        {
            "name": "production v3r CLI main() executed the synthetic run",
            "expected": 0,
            "actual": cli_returncode,
            "ok": cli_returncode == 0,
        }
    )
    bound_entry_points = sorted(
        str(entry["label"])
        for entry in shortfall.executing_implementation(repo_root=REPO)[
            "entry_points"
        ]
    )
    checks.append(
        {
            "name": "production v3r CLI is bound as an executing entry point",
            "expected": "scripts/freeze_recount_v3r.py",
            "actual": bound_entry_points,
            "ok": "scripts/freeze_recount_v3r.py" in bound_entry_points,
        }
    )
    checks.append(
        {
            "name": "synthetic workspace is outside the official artifacts",
            "expected": True,
            "actual": str(out_root),
            "ok": "gemmaloop-artifacts" not in str(out_root),
        }
    )

    receipt = {
        "kind": "gemmaloop_v3r_synthetic_dry_run_receipt",
        "version": 1,
        "official": False,
        "synthetic": True,
        "label": "SYNTHETIC-ONLY, NON-OFFICIAL v3r dry-run validation",
        "created_utc": shortfall._utc_now(),
        "duration_s": round(time.time() - started, 3),
        "command": command,
        "python_argv": sys.argv,
        "production_cli": {
            "module": "scripts/freeze_recount_v3r.py",
            "entry_point": "main",
            "argv": cli_argv,
            "returncode": cli_returncode,
            "redirected": (
                "Only pool/lite/request-id/output-root paths are synthetic; the "
                "source-record loader is the offline synthetic fixture loader. "
                "The official guard, image digest check, tokenizer load and "
                "run_v3r_pipeline are the production main() path."
            ),
        },
        "workspace": str(workdir),
        "fixture": {
            "synthetic_candidates": len(fixture["request_ids"]),
            "t_fit_passing": T_FIT_PASSING,
            "selection_passing": SELECTION_PASSING,
            "known_failing": 1,
            "known_timeout": 1,
            "functional_tests": 1,
            "stdin_tests": len(fixture["request_ids"]) - 1,
            "notes": (
                "Hand-authored synthetic candidates only. No pilot pool, lite "
                "completion, request id or official output was read or written."
            ),
            "files": {
                str(path.relative_to(workdir)): _sha256(path)
                for path in sorted(workdir.rglob("*"))
                if path.is_file()
                and not str(path).startswith(str(out_root))
            },
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "docker": _docker_environment(),
        },
        "image": pin,
        "executing_code_hashes": shortfall.code_implementation_hashes(REPO),
        "executing_implementation": shortfall.executing_implementation(repo_root=REPO),
        "synthetic_void_continuity": {
            **synthetic,
            "receipt_payload_sha256": synthetic["receipt_payload_sha256"],
        },
        "artifacts": {
            "interpretation": result["interpretation"],
            "freeze": result["freeze"],
            "plan": result["plan"],
            "count": result["count"],
            "corpus": result["corpus"],
            "index": result["index"],
            "registry": result["registry"],
            "ledger": str(count_dir / shortfall.ATTEMPT_LEDGER_NAME),
        },
        "artifact_sha256": {
            "freeze": _sha256(Path(result["freeze"])),
            "count_manifest": _sha256(count_dir / shortfall.COUNT_MANIFEST_NAME),
            "count_summary": _sha256(count_dir / shortfall.COUNT_SUMMARY_NAME),
            "per_request": _sha256(count_dir / shortfall.COUNT_PER_REQUEST_NAME),
            "ledger": _sha256(count_dir / shortfall.ATTEMPT_LEDGER_NAME),
            "corpus_manifest": _sha256(Path(result["corpus"]) / "manifest.json"),
            "index": _sha256(Path(result["index"])),
            "v3r_registry": _sha256(Path(result["registry"])),
        },
        "expectations": [
            "The exact production v3r CLI main() (scripts/freeze_recount_v3r.py) runs the guard, image digest check, tokenizer load and the production v3r orchestration end to end.",
            "The production v3r CLI is bound in the executing implementation entry points.",
            "The exact production default grading path (default count adapter -> grade_code -> DockerSandbox.run_tests) runs end to end.",
            "Every synthetic stdin and functional passing candidate verifies.",
            "The known failing candidate is a conclusive candidate failure, not infrastructure.",
            "The timeout candidate runs inside the same production count; the real 6 s outer deadline identifies candidate_running_at_deadline, the wrapper enforcement route (A) makes it a conclusive candidate timeout, and the count remains complete.",
            "Container names and attempt tokens agree across ledger, lifecycle evidence and per-request rows.",
            "v3r attempts begin at 1 and never exceed the single registered retry.",
            "Independent count loading reproduces the serialized count.",
            "Corpus and freeze index build successfully.",
            "No official v3 artifact or registry byte changes.",
        ],
        "actual": {
            "registry_status": registry.get("status"),
            "registry_attempt_base": registry.get("attempt_base"),
            "registry_max_infrastructure_retries": registry.get(
                "max_infrastructure_retries"
            ),
            "count_status": summary.get("status"),
            "nT0": summary.get("nT0"),
            "nS0": summary.get("nS0"),
            "pilot_p1": summary.get("pilot_p1"),
            "extension_required": summary.get("extension_required"),
            "accounting": summary.get("accounting"),
            "infrastructure_failures": summary.get("infrastructure_failures"),
            "ledger_audit": audit,
            "timeout_exercise": timeout_exercise,
            "independent_loading": {
                "run_fingerprint": loaded.get("run_fingerprint"),
                "accounting": loaded.get("accounting"),
                "matches_serialized": loaded.get("run_fingerprint")
                == summary.get("run_fingerprint"),
            },
            "corpus_status": corpus_manifest.get("kind"),
            "index_status": index.get("status"),
            "official_artifacts": official_after,
        },
        "checks": checks,
        "all_checks_passed": all(check["ok"] for check in checks),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            receipt, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "all_checks_passed": receipt["all_checks_passed"],
                "failed_checks": [
                    check["name"] for check in checks if not check["ok"]
                ],
                "count_status": summary.get("status"),
                "nT0": summary.get("nT0"),
                "nS0": summary.get("nS0"),
                "duration_s": receipt["duration_s"],
            },
            indent=2,
        )
    )
    return 0 if receipt["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
