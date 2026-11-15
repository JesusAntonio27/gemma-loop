"""Fix round 10: enforcement must be *issued* inside the frozen window.

The registered six-second outer budget is over the whole ``docker run`` with no
restart.  After the deadline the wrapper gets only the short observation
sub-window; the host then starts enforcement and must issue every immediate
container-kill primitive (and hard-kill the docker client) inside the frozen
250 ms window, before any state probe.  If every enforcement launch fails the
attempt stays infrastructure/unknown with cleanup pending.
"""

from __future__ import annotations

import subprocess
import time

from gemmaloop.stage0 import grader


class _NoThreads:
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

    def __init__(self) -> None:
        self.reaped = False

    def wait(self, timeout: float | None = None) -> int:
        self.reaped = True
        return 0

    def kill(self) -> None:
        self.reaped = True


class _TimeoutProcess:
    pid = 3001
    returncode = -9

    def __init__(self) -> None:
        self.calls = 0
        self.communicate_timeout: float | None = None
        self.killed_at: float | None = None

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        self.calls += 1
        if self.calls == 1:
            self.communicate_timeout = timeout
            time.sleep(float(timeout or 0.0))
            raise subprocess.TimeoutExpired(["docker"], float(timeout or 0.0))
        return "", ""

    def kill(self) -> None:
        self.killed_at = time.perf_counter()


def _run(monkeypatch, process, spawner, *, timeout=0.05, state_probe=None, cleanup=None):
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(grader.threading, "Thread", _NoThreads)
    return grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=timeout,
        container_name="container-r10",
        state_probe=state_probe or (lambda name: (None, "no state")),
        cleanup=cleanup or (lambda name: True),
        enforcement_spawner=spawner,
    )


def test_frozen_enforcement_constants_are_ordered_inside_the_window():
    assert grader.LIFECYCLE_NOTICE_GRACE_S <= 0.25
    assert 0.0 < grader.LIFECYCLE_NOTICE_OBSERVATION_S <= 0.15
    assert (
        grader.DEADLINE_ENFORCEMENT_START_S
        == grader.LIFECYCLE_NOTICE_OBSERVATION_S
    )
    assert (
        grader.DEADLINE_ENFORCEMENT_START_S
        + grader.DEADLINE_ENFORCEMENT_EXTRA_S
        <= grader.LIFECYCLE_NOTICE_GRACE_S
    )
    # The module fails closed at import if the ordering drifts.
    grader._assert_lifecycle_enforcement_bounds()


def test_hard_kill_and_container_kill_are_issued_by_deadline_plus_window(
    monkeypatch,
):
    """Reviewer probe (a): 0.05 s budget and a 100 ms taskkill mock."""

    process = _TimeoutProcess()
    container_kills: list[float] = []
    taskkill_calls: list[float] = []

    def spawner(command, **kwargs):
        if "kill" in command:
            container_kills.append(time.perf_counter())
        return _EnforcementClient()

    def fake_run(command, **kwargs):
        taskkill_calls.append(time.perf_counter())
        time.sleep(0.1)
        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""
        return Completed()

    monkeypatch.setattr(grader.subprocess, "run", fake_run)
    started = time.perf_counter()
    _run(monkeypatch, process, spawner)

    assert taskkill_calls == []
    assert process.killed_at is not None
    assert process.killed_at - started <= 0.300
    assert container_kills
    assert container_kills[0] - started <= 0.300


def test_failed_first_primitive_is_an_overrun_before_any_slow_probe(
    monkeypatch,
):
    """Reviewer probe (b): removal launch raises and the probe takes 0.4 s.

    Round 12 tightens the rule: any primitive launch failure is itself an
    overrun, so the timely alternative cannot make the attempt conclusive and
    no state probe is allowed to run.
    """

    process = _TimeoutProcess()
    issued: list[tuple[str, float]] = []
    probe_calls: list[float] = []

    def spawner(command, **kwargs):
        if command[1] == "rm":
            raise OSError("removal client could not start")
        issued.append((command[1], time.perf_counter()))
        return _EnforcementClient()

    def slow_probe(name):
        probe_calls.append(time.perf_counter())
        time.sleep(0.4)
        return None, "no state"

    started = time.perf_counter()
    result = _run(
        monkeypatch,
        process,
        spawner,
        state_probe=slow_probe,
        cleanup=lambda name: False,
    )
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    assert issued and issued[0][0] == "kill"
    assert issued[0][1] - started <= 0.300
    # A failed primitive is an overrun, so the probe never runs.
    assert probe_calls == []
    assert process.killed_at is not None
    evidence = lifecycle["deadline_enforcement"]
    assert evidence["overrun"] is True
    assert evidence["any_primitive_failed"] is True
    assert evidence["overrun_reason"]
    # Cleanup is pending because enforcement was not completely issued.
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES
    assert lifecycle["deadline_enforcement_issued"] is True
    assert lifecycle["deadline_enforcement_overrun"] is True


def test_failed_first_primitive_keeps_cleanup_pending_even_if_cleanup_succeeds(
    monkeypatch,
):
    process = _TimeoutProcess()
    issued: list[str] = []
    probe_calls: list[float] = []

    def spawner(command, **kwargs):
        if command[1] == "rm":
            raise OSError("removal client could not start")
        issued.append(command[1])
        return _EnforcementClient()

    def slow_probe(name):
        probe_calls.append(time.perf_counter())
        time.sleep(0.4)
        return None, "no state"

    result = _run(
        monkeypatch,
        process,
        spawner,
        state_probe=slow_probe,
        cleanup=lambda name: True,
    )
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    assert issued == ["kill"]
    assert probe_calls == []
    # An overrun is never conclusive, even when the cleanup callback reports
    # success; the recovery path must verify the container is gone.
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["deadline_enforcement_overrun"] is True
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def test_all_enforcement_launches_failing_leaves_cleanup_pending(monkeypatch):
    process = _TimeoutProcess()
    probe_calls: list[float] = []
    attempts: list[str] = []

    def spawner(command, **kwargs):
        attempts.append(command[1])
        raise OSError("docker client unavailable")

    def probe(name):
        probe_calls.append(time.perf_counter())
        return None, "no state"

    result = _run(
        monkeypatch,
        process,
        spawner,
        state_probe=probe,
        cleanup=lambda name: True,
    )
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    # Every primitive was attempted (both alternatives, twice) but none was
    # issued; the state probe is not allowed to stand in for enforcement.
    assert probe_calls == []
    assert attempts.count("kill") >= 2
    assert attempts.count("rm") >= 2
    assert process.killed_at is not None
    assert lifecycle["deadline_enforcement_issued"] is False
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def test_deterministic_late_receipt_is_received_and_rejected():
    """The late entry was positively received, then rejected as unknown."""

    markers = [
        {"event": grader.LIFECYCLE_EVENT_BUDGET, "remaining_s": 0.1},
        {
            "event": grader.LIFECYCLE_EVENT_START,
            "monotonic": 1.0,
            "child_pid": 4242,
        },
        {
            "event": grader.LIFECYCLE_EVENT_ALIVE,
            "monotonic": 2.0,
            "notice_monotonic": 1.9,
            "child_pid": 4242,
        },
        {
            "event": grader.LIFECYCLE_EVENT_FINISH,
            "monotonic": 2.1,
            "child_pid": 4242,
            "terminated": True,
            "exit_observed": True,
        },
    ]
    starts = [m for m in markers if m["event"] == grader.LIFECYCLE_EVENT_START]
    assert len(starts) == 1
    classification = grader.classify_lifecycle_evidence(
        markers=markers,
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        deadline_epoch=100.0,
        status="exited",
        host_completed_within_budget=False,
        host_start_received_within_budget=False,
    )
    error = grader.lifecycle_record_error(
        markers=markers,
        marker_error=None,
        cleanup_verified=True,
        timed_out=True,
        host_completed_within_budget=False,
        host_start_received_within_budget=False,
    )
    assert classification in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert classification not in grader.CANDIDATE_DEADLINE_CLASSES
    assert error == (
        "execution entry was not received by the host before its registered "
        "deadline"
    )
