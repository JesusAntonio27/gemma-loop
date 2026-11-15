"""Fix round 15: exactly two conclusive outer-deadline enforcement routes.

The registered lifecycle ruling makes a timeout conclusive with established
execution entry, positive evidence of liveness at the deadline and verified
eventual termination/cleanup.  Host enforcement is required only when the host
has to enforce.  This round accepts exactly two routes:

* route A (``wrapper``): the frozen wrapper received the authenticated deadline
  notice, observed the identified child alive, killed it and exited before the
  host enforcement window; the host observed the docker run process/container
  exit no later than ``deadline + 0.25 s`` and cleanup is verified.  No host
  enforcement action is required.
* route B (``host``): the existing complete frozen action-set rule.

Any other combination is infrastructure/unknown with cleanup pending.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest

from gemmaloop.stage0 import grader, shortfall

KEY = "a" * 64
ATTEMPT = "attempt-r15"
DEADLINE = 100.0
BOUND = DEADLINE + grader.LIFECYCLE_NOTICE_GRACE_S


def _record(event: str, seq: int, monotonic: float, **extra) -> dict:
    return {
        "v": 1,
        "attempt": ATTEMPT,
        "seq": seq,
        "event": event,
        "wall": 1_000_000.0 + monotonic,
        "monotonic": monotonic,
        "pid": 1,
        **extra,
    }


def _markers() -> list[dict]:
    return [
        _record(grader.LIFECYCLE_EVENT_READY, 1, 99.0),
        _record(grader.LIFECYCLE_EVENT_BUDGET, 2, 99.1, remaining_s=1.0),
        _record(grader.LIFECYCLE_EVENT_START, 3, 99.2, child_pid=77),
        _record(
            grader.LIFECYCLE_EVENT_ALIVE,
            4,
            100.01,
            child_pid=77,
            notice_monotonic=DEADLINE,
        ),
        _record(
            grader.LIFECYCLE_EVENT_FINISH,
            5,
            100.02,
            child_pid=77,
            returncode=-9,
            terminated=True,
            exit_observed=True,
        ),
    ]


def _frames(records: list[dict]) -> list[str]:
    lines = []
    for record in records:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            KEY.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        lines.append(mac + " GML1 " + body + "\n")
    return lines


def _wrapper_record(*, host_exit: float = 100.1, **overrides) -> dict:
    record = {
        "triggered": False,
        "bound_s": grader.LIFECYCLE_NOTICE_GRACE_S,
        "deadline_monotonic": DEADLINE,
        "bound_monotonic": BOUND,
        "enforcement_route": grader.ENFORCEMENT_ROUTE_WRAPPER,
        "host_exit_monotonic": host_exit,
        "host_exit_within_bound": True,
        "overrun": False,
        "overrun_reason": None,
        "wrapper_kill_verified": True,
        "errors": [],
    }
    record.update(overrides)
    return record


def _route_a_attempt(
    *,
    host_exit: float = 100.1,
    with_frames: bool = True,
) -> dict:
    records = _markers()
    attempt = {
        "attempt": 1,
        "status": "timeout",
        "passed": False,
        "infrastructure_failure": False,
        "timed_out": True,
        "lifecycle_class": grader.LIFECYCLE_CANDIDATE_RUNNING,
        "lifecycle_valid": True,
        "cleanup_verified": True,
        "handoff_success": True,
        "termination_verified": True,
        "deadline_alive": True,
        "exit_observed": True,
        "start_received_within_budget": True,
        "start_received_monotonic": 99.5,
        "host_deadline_monotonic": DEADLINE,
        "host_exit_monotonic": host_exit,
        "enforcement_route": grader.ENFORCEMENT_ROUTE_WRAPPER,
        "deadline_enforcement": _wrapper_record(host_exit=host_exit),
        "lifecycle_attempt": ATTEMPT,
        "lifecycle_auth_key": KEY,
        "lifecycle_records": records,
        "lifecycle_markers": records,
    }
    if with_frames:
        attempt["lifecycle_frames_raw"] = _frames(records)
    return attempt


def _route_a_row(attempt: dict) -> dict:
    row = dict(attempt)
    row.update(
        {
            "index": 0,
            "status": "timeout",
            "input": "",
            "expected": "never\n",
            "stdout": "",
            "test_type": "stdin",
            "attempts": [attempt],
        }
    )
    return row


def _validate(row: dict):
    return shortfall._validate_per_test_coverage(
        [row],
        expected_n=1,
        expected_sha="a" * 64,
        inventory={"n_tests": 1, "tests_sha256": "a" * 64},
        test_bytes=[
            json.dumps(
                {"input": "", "output": "never\n", "type": "stdin"}
            )
        ],
        request_id="request-r15",
        enforcement_bound_s=grader.LIFECYCLE_NOTICE_GRACE_S,
    )


# ---------------------------------------------------------------------------
# Route A through the production Docker supervisor
# ---------------------------------------------------------------------------
class _WrapperExitProcess:
    """A docker-run process the wrapper kills after the host deadline."""

    pid = 5150
    returncode = -9

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        time.sleep(self.delay)
        return "", ""


def _run_wrapper_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, delay: float, timeout: float
):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    records = _markers()
    (evidence / grader.LIFECYCLE_EVIDENCE_FILE).write_text(
        "".join(_frames(records)), encoding="utf-8"
    )
    process = _WrapperExitProcess(delay)
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    return grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=timeout,
        container_name="container-r15",
        evidence_dir=evidence,
        key=KEY,
        attempt=ATTEMPT,
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
        marker_reader=lambda *a, **k: (records, None),
    )


def test_genuine_timeout_through_wrapper_route_is_conclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    timeout = 0.15
    # The process exits after the deadline but well inside the frozen window
    # and without any host enforcement.
    result = _run_wrapper_route(
        tmp_path, monkeypatch, delay=timeout + 0.02, timeout=timeout
    )
    _code, _stdout, _stderr, timed_out, _duration, lifecycle = result

    assert timed_out is True
    assert lifecycle["class"] == grader.LIFECYCLE_CANDIDATE_RUNNING
    assert lifecycle["enforcement_route"] == grader.ENFORCEMENT_ROUTE_WRAPPER
    assert lifecycle["deadline_enforcement"]["triggered"] is False
    assert (
        lifecycle["deadline_enforcement"]["enforcement_route"]
        == grader.ENFORCEMENT_ROUTE_WRAPPER
    )
    assert lifecycle["deadline_enforcement"]["wrapper_kill_verified"] is True
    assert lifecycle["deadline_enforcement"]["host_exit_within_bound"] is True
    assert lifecycle["lifecycle_valid"] is True
    assert lifecycle["cleanup_verified"] is True
    assert lifecycle["record_error"] is None

    row = {
        "passed": False,
        "timed_out": True,
        "lifecycle_class": lifecycle["class"],
        "lifecycle_valid": lifecycle["lifecycle_valid"],
        "cleanup_verified": lifecycle["cleanup_verified"],
        "handoff_success": lifecycle["handoff_success"],
        "termination_verified": lifecycle["termination_verified"],
        "deadline_alive": lifecycle["deadline_alive"],
        "enforcement_route": lifecycle["enforcement_route"],
        "host_exit_monotonic": lifecycle["host_exit_monotonic"],
        "host_deadline_monotonic": lifecycle["host_deadline_monotonic"],
        "deadline_enforcement": lifecycle["deadline_enforcement"],
        "lifecycle_frames_raw": lifecycle["lifecycle_frames_raw"],
        "lifecycle_auth_key": lifecycle["lifecycle_auth_key"],
        "lifecycle_attempt": lifecycle["lifecycle_attempt"],
        "start_received_within_budget": lifecycle[
            "start_received_within_budget"
        ],
    }
    assert shortfall._classify_test(row) == (False, "timeout")


def test_container_exit_after_the_window_without_host_enforcement_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    timeout = 0.1
    result = _run_wrapper_route(
        tmp_path, monkeypatch, delay=timeout + 0.35, timeout=timeout
    )
    _code, _stdout, _stderr, timed_out, _duration, lifecycle = result

    assert timed_out is True
    assert lifecycle["enforcement_route"] is None
    assert lifecycle["deadline_enforcement"]["host_exit_within_bound"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES
    assert lifecycle["cleanup_verified"] is False
    assert "no conclusive deadline enforcement route" in (
        lifecycle["record_error"] or ""
    )
    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": True,
            "lifecycle_class": lifecycle["class"],
            "lifecycle_valid": lifecycle["lifecycle_valid"],
            "cleanup_verified": lifecycle["cleanup_verified"],
            "enforcement_route": lifecycle["enforcement_route"],
            "deadline_enforcement": lifecycle["deadline_enforcement"],
        }
    ) == (True, "infrastructure_error")


# ---------------------------------------------------------------------------
# Loader and count classifier re-check both routes
# ---------------------------------------------------------------------------
def test_loader_accepts_a_conclusive_wrapper_route_timeout():
    attempt = _route_a_attempt()
    row = _route_a_row(attempt)
    assert shortfall._classify_test(row) == (False, "timeout")
    assert _validate(row) is None


def test_loader_refuses_route_a_host_exit_after_the_bound():
    attempt = _route_a_attempt(host_exit=BOUND + 0.2)
    attempt["deadline_enforcement"] = _wrapper_record(
        host_exit=BOUND + 0.2, host_exit_within_bound=False
    )
    row = _route_a_row(attempt)
    refusal = _validate(row)
    assert refusal is not None
    assert "host exit is after" in refusal
    assert "deadline enforcement issuance" in refusal


def test_loader_refuses_route_a_missing_wrapper_kill_record():
    attempt = _route_a_attempt(with_frames=False)
    row = _route_a_row(attempt)
    refusal = _validate(row)
    assert refusal is not None
    assert "wrapper kill record is missing" in refusal


def test_loader_refuses_route_a_unauthenticated_wrapper_kill_record():
    attempt = _route_a_attempt()
    tampered = list(attempt["lifecycle_frames_raw"])
    tampered[0] = "0" * 64 + tampered[0][64:]
    attempt["lifecycle_frames_raw"] = tampered
    row = _route_a_row(attempt)
    refusal = _validate(row)
    assert refusal is not None
    assert "wrapper kill record is not authenticated" in refusal


def test_host_route_b_is_unchanged():
    client = {
        "action": grader.CLIENT_HARD_KILL_ACTION,
        "status": "returned",
        "issued": True,
        "returned": True,
        "error": None,
        "started_monotonic": DEADLINE + 0.10,
        "returned_monotonic": DEADLINE + 0.11,
    }
    primitives = [
        {
            "action": action,
            "command": command,
            "primitive": command[1],
            "status": "returned",
            "issued": True,
            "returned": True,
            "error": None,
            "launch_started_monotonic": DEADLINE + 0.12,
            "launch_returned_monotonic": DEADLINE + 0.13,
        }
        for action, command in (
            (grader.DOCKER_RM_ACTION, ["docker", "rm", "-f", "c"]),
            (grader.DOCKER_KILL_ACTION, ["docker", "kill", "-s", "KILL", "c"]),
        )
    ]
    record = {
        "triggered": True,
        "deadline_monotonic": DEADLINE,
        "bound_monotonic": BOUND,
        "enforcement_route": grader.ENFORCEMENT_ROUTE_HOST,
        "actions": [client, *primitives],
        "client_kill": client,
        "primitives": primitives,
        "overrun": False,
        "overrun_reason": None,
    }
    assert (
        shortfall._deadline_enforcement_reason(
            record, bound_s=grader.LIFECYCLE_NOTICE_GRACE_S, required=True
        )
        is None
    )
    overrun = dict(record)
    overrun["actions"] = [client, primitives[0]]
    overrun["primitives"] = [primitives[0]]
    reason = shortfall._deadline_enforcement_reason(
        overrun, bound_s=grader.LIFECYCLE_NOTICE_GRACE_S, required=True
    )
    assert reason is not None
    assert "missing expected action" in reason
