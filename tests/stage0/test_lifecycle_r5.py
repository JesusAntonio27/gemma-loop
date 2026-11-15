"""Fix round 5: frozen bootstrap/pipe lifecycle, state-specific validation."""

from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
from pathlib import Path

import pytest

from gemmaloop.stage0 import grader, shortfall


def _rec(event: str, seq: int, mono: float, **extra) -> dict:
    return {
        "v": 1,
        "attempt": "attempt-1",
        "seq": seq,
        "event": event,
        "wall": 1.0 + mono,
        "monotonic": mono,
        "pid": 1,
        **extra,
    }


def _base(deadline: float = 106.0) -> list[dict]:
    return [
        _rec(grader.LIFECYCLE_EVENT_READY, 1, 100.0),
        _rec(grader.LIFECYCLE_EVENT_BUDGET, 2, 100.1, deadline_monotonic=deadline),
        _rec(grader.LIFECYCLE_EVENT_START, 3, 100.2, child_pid=77),
    ]


def _finish(seq: int, mono: float, *, terminated: bool = True) -> dict:
    return _rec(
        grader.LIFECYCLE_EVENT_FINISH,
        seq,
        mono,
        child_pid=77,
        returncode=0,
        terminated=terminated,
        exit_observed=True,
    )


def _alive(seq: int, mono: float, deadline: float = 106.0) -> dict:
    return _rec(
        grader.LIFECYCLE_EVENT_ALIVE,
        seq,
        mono,
        child_pid=77,
        notice_monotonic=deadline,
    )


def test_normal_completion_requires_handoff_exit_and_cleanup():
    records = _base() + [_finish(4, 101.0)]
    assert grader.classify_lifecycle_evidence(
        markers=records, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_COMPLETED_BEFORE
    assert (
        grader.lifecycle_record_error(
            markers=records, cleanup_verified=True, timed_out=False
        )
        is None
    )
    # A missing successful-handoff record is not a normal completion.
    assert (
        grader.lifecycle_record_error(
            markers=[_rec(grader.LIFECYCLE_EVENT_READY, 1, 100.0)],
            cleanup_verified=True,
            timed_out=False,
        )
    )
    # Cleanup is part of every outcome.
    assert grader.lifecycle_record_error(
        markers=records, cleanup_verified=False, timed_out=False
    )


def test_failed_or_incomplete_handoff_is_infrastructure_unknown():
    failed = _base()[:2] + [
        _rec(
            grader.LIFECYCLE_EVENT_HANDOFF_FAILED,
            3,
            100.2,
            child_pid=77,
            error="FileNotFoundError",
        )
    ]
    assert grader.classify_lifecycle_evidence(
        markers=failed, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_UNRESOLVED
    assert "handoff failed" in (
        grader.lifecycle_record_error(
            markers=failed, cleanup_verified=True, timed_out=False
        )
        or ""
    )
    # Entry observed but no successful exec confirmation and no failure frame:
    # incomplete handoff, still unknown.
    incomplete = _base()[:2] + [_rec("entry", 3, 100.2, pid=77)]
    assert grader.classify_lifecycle_evidence(
        markers=incomplete, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_UNRESOLVED


def test_timeout_requires_positive_liveness_and_verified_termination():
    deadline = 106.0
    records = _base(deadline) + [
        _alive(4, 106.001, deadline),
        _finish(5, 106.002),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=records, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_CANDIDATE_RUNNING
    assert (
        grader.lifecycle_record_error(
            markers=records, cleanup_verified=True, timed_out=True
        )
        is None
    )
    # A timeout with no liveness observation is unknown even if it timed out.
    no_alive = _base(deadline) + [_finish(4, 106.002)]
    assert grader.classify_lifecycle_evidence(
        markers=no_alive, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_UNRESOLVED
    assert (
        "liveness"
        in (
            grader.lifecycle_record_error(
                markers=no_alive, cleanup_verified=True, timed_out=True
            )
            or ""
        )
        or "spans" in (
            grader.lifecycle_record_error(
                markers=no_alive, cleanup_verified=True, timed_out=True
            )
            or ""
        )
    )
    # Liveness before the deadline is not deadline evidence.
    early_alive = _base(deadline) + [
        _alive(4, 105.0, deadline),
        _finish(5, 106.5),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=early_alive, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_UNRESOLVED
    # Alive at the deadline but termination is not verified.
    unverified = _base(deadline) + [
        _alive(4, 106.001, deadline),
        _finish(5, 106.002, terminated=False),
    ]
    assert grader.lifecycle_record_error(
        markers=unverified, cleanup_verified=True, timed_out=True
    )


def test_exit_interval_spanning_the_deadline_is_unknown_not_timeout():
    deadline = 106.0
    spans = _base(deadline) + [_finish(4, 106.5)]
    assert grader.classify_lifecycle_evidence(
        markers=spans, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_UNRESOLVED
    assert "spans" in (
        grader.lifecycle_record_error(
            markers=spans, cleanup_verified=True, timed_out=True
        )
        or ""
    )
    # A finish before the deadline is conclusive even if the host timed out
    # during teardown; the classifier returns completion, and the counting
    # classifier then treats teardown failure as infrastructure.
    completed = _base(deadline) + [_finish(4, 105.0)]
    assert grader.classify_lifecycle_evidence(
        markers=completed, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_COMPLETED_BEFORE
    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": True,
            "exit_code": 0,
            "cleanup_verified": True,
            "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            "lifecycle_valid": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": False,
            "exit_observed": True,
        }
    ) == (True, "infrastructure_error")


def test_host_epoch_is_not_compared_with_the_wrapper_clock():
    records = _base(deadline=106.0) + [_finish(4, 101.0)]
    # A wildly different host epoch/deadline value must not change the result;
    # only the wrapper's own monotonic deadline is compared.
    for host_deadline in (-1_000_000.0, 0.0, 1_000_000.0):
        assert grader.classify_lifecycle_evidence(
            markers=records,
            cleanup_verified=True,
            deadline_epoch=host_deadline,
        ) == grader.LIFECYCLE_COMPLETED_BEFORE


def test_timely_nonzero_exit_with_valid_evidence_is_candidate_error():
    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": False,
            "exit_code": 1,
            "error": "docker exit code 1",
            "cleanup_verified": True,
            "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            "lifecycle_valid": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": False,
            "exit_observed": True,
        }
    ) == (False, "candidate_error")
    # A nonzero infrastructure exit code remains infrastructure/unknown.
    assert shortfall._classify_test(
        {
            "passed": False,
            "timed_out": False,
            "exit_code": 125,
            "error": "docker exit code 125",
            "cleanup_verified": True,
            "lifecycle_class": grader.LIFECYCLE_COMPLETED_BEFORE,
            "lifecycle_valid": True,
            "handoff_success": True,
            "termination_verified": True,
            "deadline_alive": False,
            "exit_observed": True,
        }
    ) == (True, "infrastructure_error")


def test_entry_after_deadline_is_startup_failure():
    late = [
        _rec(grader.LIFECYCLE_EVENT_READY, 1, 100.0),
        _rec(grader.LIFECYCLE_EVENT_BUDGET, 2, 100.1, deadline_monotonic=101.0),
        _rec(grader.LIFECYCLE_EVENT_START, 3, 102.0, child_pid=77),
        _finish(4, 102.1),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=late, cleanup_verified=True, deadline_epoch=999.0
    ) == grader.LIFECYCLE_STARTUP_FAILURE


def _write_signed(path: Path, key: str, records: list[dict]) -> None:
    lines = []
    for record in records:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            key.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        lines.append(mac + " GML1 " + body)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_records_are_bound_to_attempt_child_pid_and_sequence():
    key = "b" * 64
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / grader.LIFECYCLE_EVIDENCE_FILE
        records = _base(deadline=106.0) + [_finish(4, 101.0)]
        _write_signed(path, key, records)
        parsed, error = grader.read_lifecycle_markers(Path(tmp), key, "attempt-1")
        assert error is None and len(parsed) == 4

        # Wrong attempt identity.
        parsed, error = grader.read_lifecycle_markers(Path(tmp), key, "attempt-2")
        assert parsed == [] and "different attempt" in error

        # Duplicate/replayed sequence.
        replayed = records + [records[-1]]
        _write_signed(path, key, replayed)
        parsed, error = grader.read_lifecycle_markers(Path(tmp), key, "attempt-1")
        assert parsed == [] and "replayed" in error

        # Wrong key cannot authenticate any record.
        _write_signed(path, key, records)
        parsed, error = grader.read_lifecycle_markers(Path(tmp), "c" * 64, "attempt-1")
        assert parsed == [] and "HMAC" in error

        # Child identity mismatch is rejected by record validation.
        mismatched = records[:-1] + [
            {**records[-1], "child_pid": 999},
        ]
        assert grader.lifecycle_record_error(
            markers=mismatched, cleanup_verified=True, timed_out=False
        ) == "lifecycle record is not bound to the tracked child"


def test_wrapper_is_a_frozen_bootstrap_with_private_pipe_and_no_key_file():
    source = grader._LIFECYCLE_WRAPPER_SOURCE
    assert "hmac.new" in source
    assert "os.fork()" in source
    assert "os.execv" in source
    assert 'os.set_inheritable(entry_write, False)' in source
    assert source.index('os.fork()') < source.index('key = read_exact(0, KEY_LEN)')
    assert "os.close(evidence_fd)" in source
    assert "PR_SET_DUMPABLE" in source or "prctl(4, 0" in source
    # The old nonce/key file channel is gone.
    assert "/evidence/nonce" not in source
    assert "sha256((nonce" not in source


def test_grader_rejects_a_pass_without_a_genuine_exit_observation(monkeypatch):
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
                "lifecycle_valid": True,
                "cleanup_verified": True,
                "handoff_success": True,
                "termination_verified": True,
                "deadline_alive": False,
                "exit_observed": False,
                "markers": [],
            },
        )

    monkeypatch.setattr(grader, "_run_process_docker", fake_run)
    result = sandbox.run_tests("print(1)", [{"input": "", "output": "1"}], timeout=6)
    assert result.tests[0].passed is False
    assert "untrusted lifecycle evidence" in (result.tests[0].error or "")
    assert shortfall._classify_test(result.tests[0].__dict__) == (
        True,
        "infrastructure_error",
    )


def _host_route_record() -> dict:
    """A complete host-route (B) action set for the classification probe."""

    client = {
        "action": grader.CLIENT_HARD_KILL_ACTION,
        "status": "returned",
        "issued": True,
        "returned": True,
        "error": None,
        "started_monotonic": 10.10,
        "returned_monotonic": 10.11,
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
            "launch_started_monotonic": 10.12,
            "launch_returned_monotonic": 10.13,
        }
        for action, command in (
            (grader.DOCKER_RM_ACTION, ["docker", "rm", "-f", "c"]),
            (grader.DOCKER_KILL_ACTION, ["docker", "kill", "-s", "KILL", "c"]),
        )
    ]
    return {
        "triggered": True,
        "deadline_monotonic": 10.0,
        "bound_monotonic": 10.25,
        "actions": [client, *primitives],
        "client_kill": client,
        "primitives": primitives,
        "overrun": False,
        "overrun_reason": None,
    }


def test_grader_accepts_timeout_only_with_liveness_and_termination(monkeypatch):
    sandbox = grader.DockerSandbox()
    monkeypatch.setattr(sandbox, "ensure_image", lambda: None)

    def run_with(*, alive: bool, terminated: bool):
        def fake_run(command, *, input_text, timeout, container_name, **kwargs):
            return (
                -9,
                "",
                "",
                True,
                6.0,
                {
                    "class": grader.LIFECYCLE_CANDIDATE_RUNNING,
                    "container_name": container_name,
                    "lifecycle_valid": True,
                    "cleanup_verified": True,
                    "handoff_success": True,
                    "termination_verified": terminated,
                    "deadline_alive": alive,
                    "exit_observed": True,
                    "markers": [],
                    "enforcement_route": grader.ENFORCEMENT_ROUTE_HOST,
                    "deadline_enforcement": _host_route_record(),
                },
            )

        return fake_run

    monkeypatch.setattr(
        grader, "_run_process_docker", run_with(alive=True, terminated=True)
    )
    result = sandbox.run_tests(
        "import time\ntime.sleep(30)\n",
        [{"input": "", "output": "1"}],
        timeout=6,
    )
    assert result.tests[0].timed_out is True
    assert result.tests[0].error == "timeout after 6s"
    assert shortfall._classify_test(result.tests[0].__dict__) == (False, "timeout")

    monkeypatch.setattr(
        grader, "_run_process_docker", run_with(alive=False, terminated=True)
    )
    result = sandbox.run_tests(
        "import time\ntime.sleep(30)\n",
        [{"input": "", "output": "1"}],
        timeout=6,
    )
    assert result.tests[0].passed is False
    assert shortfall._classify_test(result.tests[0].__dict__) == (
        True,
        "infrastructure_error",
    )


def test_local_lifecycle_is_explicit_and_not_docker_evidence():
    assert shortfall._classify_test(
        {
            "cleanup_verified": True,
            "lifecycle_class": grader.LIFECYCLE_LOCAL,
            "timed_out": False,
            "passed": True,
        }
    ) == (False, "passed")
