"""Fix round 11: issue enforcement concurrently and fail closed on overrun.

The r10 review simulated a slow process launch (0.4 s) before the first
container-kill launch returned.  The old sequential path delayed both the
alternative primitive and the docker-client hard kill past the frozen
deadline-plus-250 ms window and recorded no issuance time.

This round hard-kills the docker client *first*, launches both primitives in
their own threads at the same instant, stamps every action on the host
monotonic clock, and fails closed when the client kill or at least one
primitive was not issued by the frozen bound.
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


class _EnforcementClient:
    returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


class _TimeoutProcess:
    pid = 4001
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


def _run(monkeypatch, process, spawner, *, timeout=0.05, state_probe=None, cleanup=None):
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    # The monitor threads are disabled, but enforcement must still launch real
    # concurrent threads.
    monkeypatch.setattr(grader.threading, "Thread", _NoThreads)
    return grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=timeout,
        container_name="container-r11",
        state_probe=state_probe or (lambda name: (None, "no state")),
        cleanup=cleanup or (lambda name: True),
        enforcement_spawner=spawner,
    )


def _wait_for(predicate, *, timeout=2.0):
    limit = time.perf_counter() + timeout
    while time.perf_counter() < limit:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_slow_first_launch_is_overrun_even_when_alternative_is_timely(
    monkeypatch,
):
    """The r11 reviewer simulation, now classified as an overrun.

    The first launch sleeps 0.4 s while ``docker kill`` returns inside the
    frozen bound.  The timely alternative must not mask the launch that was
    still in flight at the bound.
    """

    process = _TimeoutProcess()
    calls: list[dict[str, float]] = []
    lock = threading.Lock()

    def spawner(command, **kwargs):
        started = time.perf_counter()
        if command[1] == "rm":
            time.sleep(0.4)
        returned = time.perf_counter()
        with lock:
            calls.append(
                {
                    "primitive": command[1],
                    "started": started,
                    "returned": returned,
                }
            )
        return _EnforcementClient()

    started = time.perf_counter()
    result = _run(monkeypatch, process, spawner, cleanup=lambda name: True)
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    assert process.killed_at is not None
    # The docker client is killed first, well inside deadline + 250 ms.
    assert process.killed_at - started <= 0.300
    assert grader.LIFECYCLE_NOTICE_GRACE_S <= 0.25
    assert grader.DEADLINE_ENFORCEMENT_START_S < grader.LIFECYCLE_NOTICE_GRACE_S

    by_name = {call["primitive"]: call for call in calls}
    assert set(by_name) == {"rm", "kill"}
    # The alternative primitive is not queued behind the 0.4 s first launch.
    assert by_name["kill"]["started"] < by_name["rm"]["returned"]
    assert by_name["kill"]["started"] - started <= 0.300
    assert by_name["kill"]["returned"] - started <= 0.300

    evidence = lifecycle["deadline_enforcement"]
    bound = evidence["bound_monotonic"]
    assert evidence["triggered"] is True
    # The slow launch returned late, so the attempt overran even though the
    # alternative was timely.
    assert evidence["overrun"] is True
    assert evidence["container_issued_within_bound"] is False
    assert evidence["client_kill"]["issued"] is True
    assert evidence["client_kill"]["returned_monotonic"] <= bound
    primitives = {item["primitive"]: item for item in evidence["primitives"]}
    assert primitives["kill"]["launch_returned_monotonic"] <= bound
    assert primitives["rm"]["launch_returned_monotonic"] is not None
    assert primitives["rm"]["launch_returned_monotonic"] > bound
    assert primitives["rm"]["launch_returned_within_bound"] is False
    assert primitives["kill"]["launch_returned_within_bound"] is True
    assert lifecycle["deadline_enforcement_overrun"] is True
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def test_no_primitive_returned_inside_bound_is_overrun_and_fails_closed(monkeypatch):
    """Both launches sleep 0.4 s: nothing returns inside the frozen bound."""

    process = _TimeoutProcess()
    calls: list[dict[str, float]] = []
    lock = threading.Lock()

    def spawner(command, **kwargs):
        started = time.perf_counter()
        time.sleep(0.4)
        returned = time.perf_counter()
        with lock:
            calls.append(
                {
                    "primitive": command[1],
                    "started": started,
                    "returned": returned,
                }
            )
        return _EnforcementClient()

    started = time.perf_counter()
    result = _run(monkeypatch, process, spawner, cleanup=lambda name: True)
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result
    elapsed = time.perf_counter() - started

    # Enforcement classified at the frozen bound, then used the bounded audit
    # wait to record each late launch's eventual return.
    assert elapsed >= 0.25
    assert process.killed_at is not None
    assert process.killed_at - started <= 0.300
    evidence = lifecycle["deadline_enforcement"]
    bound = evidence["bound_monotonic"]
    assert evidence["client_kill"]["issued"] is True
    assert evidence["client_kill"]["returned_monotonic"] <= bound
    assert evidence["container_issued_within_bound"] is False
    assert evidence["overrun"] is True
    assert evidence["overrun_reason"]
    assert all(
        primitive["launch_started_monotonic"] is not None
        and primitive["launch_returned_monotonic"] is not None
        and primitive["launch_returned_monotonic"] > bound
        and primitive["issued"] is True
        and primitive["error"] is None
        and primitive["launch_returned_within_bound"] is False
        for primitive in evidence["primitives"]
    )

    # Both were launched concurrently; the alternative is not behind the first.
    assert _wait_for(lambda: len(calls) == 2)
    by_name = {call["primitive"]: call for call in calls}
    assert by_name["kill"]["started"] < by_name["rm"]["returned"]
    assert by_name["kill"]["started"] - by_name["rm"]["started"] <= 0.15

    assert lifecycle["deadline_enforcement_overrun"] is True
    assert lifecycle["deadline_enforcement_issued"] is False
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def _record(**overrides):
    record = {
        "triggered": True,
        "deadline_monotonic": 10.0,
        "bound_monotonic": 10.25,
        "bound_s": 0.25,
        "client_kill": {
            "issued": True,
            "started_monotonic": 10.12,
            "returned_monotonic": 10.13,
            "within_bound": True,
            "error": None,
        },
        "primitives": [
            {
                "command": ["docker", "rm", "-f", "name"],
                "primitive": "rm",
                "launch_started_monotonic": 10.14,
                "launch_returned_monotonic": 10.20,
                "issued": True,
                "error": None,
            },
            {
                "command": ["docker", "kill", "-s", "KILL", "name"],
                "primitive": "kill",
                "launch_started_monotonic": 10.14,
                "launch_returned_monotonic": 10.18,
                "issued": True,
                "error": None,
            },
        ],
        "issued": True,
        "overrun": False,
        "overrun_reason": None,
    }
    record.update(overrides)
    return record


def test_loader_rechecks_recorded_issuance_against_frozen_bound():
    assert (
        shortfall._deadline_enforcement_reason(
            _record(), bound_s=0.25, required=True
        )
        is None
    )
    late_primitive = _record()
    late_primitive["primitives"] = [
        {
            **late_primitive["primitives"][0],
            "launch_returned_monotonic": 10.31,
        }
    ]
    reason = shortfall._deadline_enforcement_reason(
        late_primitive, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "overrun" in reason

    inconsistent = _record()
    inconsistent["overrun"] = True
    reason = shortfall._deadline_enforcement_reason(
        inconsistent, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "overrun flag does not recompute" in reason

    assert (
        shortfall._deadline_enforcement_reason(
            None, bound_s=0.25, required=True
        )
        == "missing deadline enforcement issuance evidence"
    )
    # The frozen policy records the deadline + 0.25 s bound.
    assert shortfall._frozen_deadline_enforcement_bound(
        {"verifier": {"lifecycle": {"deadline_enforcement_issue_bound_s": 0.25}}}
    ) == 0.25


def test_coverage_loader_refuses_timed_out_attempt_with_overrun_evidence():
    """The coverage loader itself re-checks the frozen issuance bound."""

    def row_with(record):
        attempt = {
            "attempt": 1,
            "status": "infrastructure_error",
            "timed_out": True,
            "lifecycle_class": grader.LIFECYCLE_UNRESOLVED,
            "cleanup_verified": True,
            "deadline_enforcement": record,
        }
        return {
            "index": 0,
            "status": "infrastructure_error",
            "passed": False,
            "timed_out": True,
            "lifecycle_class": grader.LIFECYCLE_UNRESOLVED,
            "lifecycle_valid": False,
            "cleanup_verified": True,
            "infrastructure_failure": True,
            "attempts": [attempt],
        }

    inventory = {"n_tests": 1, "tests_sha256": "a" * 64}
    assert (
        shortfall._validate_per_test_coverage(
            [row_with(_record())],
            expected_n=1,
            expected_sha="a" * 64,
            inventory=inventory,
            enforcement_bound_s=0.25,
        )
        is None
    )

    late = _record()
    late["overrun"] = True
    late["overrun_reason"] = "launch returned after deadline + 0.25 s"
    late["container_issued_within_bound"] = False
    reason = shortfall._validate_per_test_coverage(
        [row_with(late)],
        expected_n=1,
        expected_sha="a" * 64,
        inventory=inventory,
        enforcement_bound_s=0.25,
    )
    assert reason is not None
    assert "deadline enforcement issuance" in reason
    assert "overrun" in reason

    missing = shortfall._validate_per_test_coverage(
        [row_with(None)],
        expected_n=1,
        expected_sha="a" * 64,
        inventory=inventory,
        enforcement_bound_s=0.25,
    )
    assert missing is not None
    assert "missing deadline enforcement issuance evidence" in missing
