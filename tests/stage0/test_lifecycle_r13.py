"""Fix round 13: pre-create every expected enforcement slot before launching.

The r12 review found that a worker delayed by the scheduler before it entered
its target never initialized its evidence slot, the collector omitted the empty
slot, and the timely sibling made the grader report ``overrun=false``.  The
loader only validated the primitives that happened to be present, so it
accepted the incomplete enforcement record.

Round 13 fixes the action set itself:

* the frozen expected actions are the client hard kill, ``docker rm -f`` and
  ``docker kill -s KILL``;
* one evidence slot per expected action is pre-created with status
  ``scheduled`` and a scheduling timestamp before any launcher thread starts;
* workers only fill their pre-created slot, and the collector never drops one;
* the final overrun includes any action without a terminal return inside
  ``deadline + 0.25 s`` and every earlier late/in-flight observation;
* the loader requires the complete expected set with no missing or extra
  action and refuses anything else.
"""

from __future__ import annotations

import subprocess
import threading
import time

from gemmaloop.stage0 import grader, shortfall


class _NoThreads:
    """Silence the lifecycle monitor; enforcement uses its own real threads."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_set(self) -> bool:
        return False

    def set(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> bool:
        return False


class _DelayedRmThread(threading.Thread):
    """A real thread that delays the rm worker *before* it enters its target."""

    rm_delay_s = 0.8

    def run(self) -> None:  # noqa: D401 - deliberate scheduler-delay simulation
        if "rm" in (self.name or ""):
            time.sleep(self.rm_delay_s)
        super().run()


class _EnforcementClient:
    returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


class _TimeoutProcess:
    pid = 6001
    returncode = -9

    def __init__(self) -> None:
        self.calls = 0
        self.killed_at: float | None = None

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        self.calls += 1
        if self.calls == 1:
            time.sleep(float(timeout or 0.0))
            raise subprocess.TimeoutExpired(["docker"], float(timeout or 0.0))
        return "", ""

    def kill(self) -> None:
        self.killed_at = time.perf_counter()


def _run(monkeypatch, process, spawner, *, timeout=0.05):
    monkeypatch.setattr(grader, "_ENFORCEMENT_THREAD_CLASS", _DelayedRmThread)
    # Keep the bounded post-classification audit short; the rm worker is still
    # sleeping when the collector snapshots the pre-created scheduled slot.
    monkeypatch.setattr(grader, "ENFORCEMENT_RETURN_AUDIT_S", 0.1)
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(grader.threading, "Thread", _NoThreads)
    return grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=timeout,
        container_name="container-r13",
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
        enforcement_spawner=spawner,
    )


def test_delayed_rm_worker_cannot_disappear_from_the_evidence(monkeypatch):
    """Reviewer probe: the rm worker is still scheduled when collection starts.

    ``docker kill`` returns inside the frozen bound, but the rm worker has not
    entered its target.  Its pre-created slot must still be collected and the
    attempt must be an overrun.
    """

    process = _TimeoutProcess()

    def spawner(command, **kwargs):
        return _EnforcementClient()

    result = _run(monkeypatch, process, spawner)
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    assert process.killed_at is not None
    evidence = lifecycle["deadline_enforcement"]
    actions = {item["action"]: item for item in evidence["actions"]}
    assert set(actions) == set(grader.DEADLINE_ENFORCEMENT_ACTIONS)
    assert evidence["actions_complete"] is True
    assert evidence["expected_actions"] == list(
        grader.DEADLINE_ENFORCEMENT_ACTIONS
    )

    rm = actions[grader.DOCKER_RM_ACTION]
    assert rm["status"] == "scheduled"
    assert rm["scheduled_monotonic"] is not None
    assert rm["launch_started_monotonic"] is None
    assert rm["launch_returned_monotonic"] is None
    assert rm["terminal"] is False
    assert rm["late_or_in_flight_at_bound"] is True

    kill = actions[grader.DOCKER_KILL_ACTION]
    assert kill["status"] == "returned"
    assert kill["launch_returned_within_bound"] is True
    actions_client = actions[grader.CLIENT_HARD_KILL_ACTION]
    assert actions_client["status"] == "returned"
    assert actions_client["within_bound"] is True

    # A timely sibling never rescues the missing terminal action.
    assert evidence["overrun"] is True
    assert evidence["any_action_not_terminal"] is True
    assert evidence["overrun_reason"]
    assert lifecycle["deadline_enforcement_overrun"] is True
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def test_all_timely_actions_still_pass_unchanged(monkeypatch):
    """When every pre-created slot is returned in time, nothing is overrun."""

    process = _TimeoutProcess()

    def spawner(command, **kwargs):
        return _EnforcementClient()

    original = _DelayedRmThread.rm_delay_s
    monkeypatch.setattr(_DelayedRmThread, "rm_delay_s", 0.0)
    try:
        result = _run(monkeypatch, process, spawner)
    finally:
        monkeypatch.setattr(_DelayedRmThread, "rm_delay_s", original)
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    evidence = lifecycle["deadline_enforcement"]
    actions = {item["action"]: item for item in evidence["actions"]}
    assert set(actions) == set(grader.DEADLINE_ENFORCEMENT_ACTIONS)
    assert all(item["status"] == "returned" for item in actions.values())
    assert all(item["terminal"] is True for item in actions.values())
    assert all(
        item["launch_returned_within_bound"] is True
        for item in evidence["primitives"]
    )
    assert evidence["actions_complete"] is True
    assert evidence["any_action_not_terminal"] is False
    assert evidence["any_late_or_in_flight"] is False
    assert evidence["overrun"] is False
    assert lifecycle["deadline_enforcement_overrun"] is False
    assert lifecycle["cleanup_verified"] is True


# ---------------------------------------------------------------------------
# Loader probes
# ---------------------------------------------------------------------------
def _client_slot(**overrides):
    slot = {
        "action": grader.CLIENT_HARD_KILL_ACTION,
        "status": "returned",
        "scheduled_monotonic": 10.10,
        "started_monotonic": 10.12,
        "returned_monotonic": 10.13,
        "issued": True,
        "returned": True,
        "error": None,
        "terminal": True,
        "within_bound": True,
        "late_or_in_flight_at_bound": False,
    }
    slot.update(overrides)
    return slot


def _primitive_slot(action, command, primitive, started, returned, **overrides):
    slot = {
        "action": action,
        "command": command,
        "primitive": primitive,
        "status": "returned",
        "scheduled_monotonic": 10.10,
        "launch_started_monotonic": started,
        "launch_returned_monotonic": returned,
        "issued": True,
        "returned": True,
        "error": None,
        "terminal": True,
        "launch_returned_within_bound": True,
        "late_or_in_flight_at_bound": False,
    }
    slot.update(overrides)
    return slot


def _record():
    client = _client_slot()
    rm = _primitive_slot(
        grader.DOCKER_RM_ACTION,
        ["docker", "rm", "-f", "name"],
        "rm",
        10.14,
        10.20,
    )
    kill = _primitive_slot(
        grader.DOCKER_KILL_ACTION,
        ["docker", "kill", "-s", "KILL", "name"],
        "kill",
        10.14,
        10.18,
    )
    return {
        "triggered": True,
        "deadline_monotonic": 10.0,
        "bound_monotonic": 10.25,
        "bound_s": 0.25,
        "expected_actions": list(grader.DEADLINE_ENFORCEMENT_ACTIONS),
        "actions": [dict(client), dict(rm), dict(kill)],
        "client_kill": dict(client),
        "primitives": [dict(rm), dict(kill)],
        "actions_complete": True,
        "container_issued_within_bound": True,
        "any_primitive_failed": False,
        "any_action_not_terminal": False,
        "any_late_or_in_flight": False,
        "issued": True,
        "overrun": False,
        "overrun_reason": None,
    }


def test_loader_refuses_omitted_action_from_delayed_worker():
    """Reviewer loader probe: the delayed rm slot is absent from the record."""

    omitted = _record()
    omitted["actions"] = [
        item
        for item in omitted["actions"]
        if item["action"] != grader.DOCKER_RM_ACTION
    ]
    omitted["primitives"] = [
        item
        for item in omitted["primitives"]
        if item["action"] != grader.DOCKER_RM_ACTION
    ]
    reason = shortfall._deadline_enforcement_reason(
        omitted, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "missing expected action" in reason
    assert "overrun" in reason

    # The same incomplete record cannot back a conclusive timeout row.
    row = {
        "index": 0,
        "status": "timeout",
        "passed": False,
        "timed_out": True,
        "lifecycle_class": grader.LIFECYCLE_CANDIDATE_RUNNING,
        "lifecycle_valid": True,
        "cleanup_verified": True,
        "infrastructure_failure": False,
        "attempts": [
            {
                "attempt": 1,
                "status": "timeout",
                "timed_out": True,
                "lifecycle_class": grader.LIFECYCLE_CANDIDATE_RUNNING,
                "cleanup_verified": True,
                "deadline_enforcement": omitted,
            }
        ],
    }
    refusal = shortfall._validate_per_test_coverage(
        [row],
        expected_n=1,
        expected_sha="a" * 64,
        inventory={"n_tests": 1, "tests_sha256": "a" * 64},
        enforcement_bound_s=0.25,
    )
    assert refusal is not None
    assert "deadline enforcement issuance" in refusal
    assert "missing expected action" in refusal


def test_loader_refuses_missing_slot_without_actions_field():
    """Legacy-shaped evidence missing the rm primitive is also refused."""

    record = _record()
    record.pop("actions")
    record["primitives"] = [
        item
        for item in record["primitives"]
        if item["action"] != grader.DOCKER_RM_ACTION
    ]
    reason = shortfall._deadline_enforcement_reason(
        record, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "missing expected action" in reason
    assert "overrun" in reason


def test_loader_refuses_extra_or_unexpected_action_slot():
    record = _record()
    extra = _primitive_slot(
        "docker pause",
        ["docker", "pause", "name"],
        "pause",
        10.14,
        10.15,
    )
    record["actions"].append(extra)
    record["primitives"].append(dict(extra))
    reason = shortfall._deadline_enforcement_reason(
        record, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "unexpected action" in reason
    assert "overrun" in reason


def test_loader_still_accepts_all_timely_complete_action_set():
    assert (
        shortfall._deadline_enforcement_reason(
            _record(), bound_s=0.25, required=True
        )
        is None
    )
    legacy = _record()
    legacy.pop("actions")
    assert (
        shortfall._deadline_enforcement_reason(
            legacy, bound_s=0.25, required=True
        )
        is None
    )
