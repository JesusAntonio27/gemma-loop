"""Fix round 8: the outer deadline and its bounded enforcement window.

Registered lifecycle paragraph 3 keeps the six-second outer timeout over the
whole ``docker run`` without restart.  The host may spend at most a tightly
bounded post-deadline window on the liveness observation and kill, but no
stdout/stderr bytes or exit status observed after the deadline may be used.
"""

from __future__ import annotations

import subprocess
import time

from gemmaloop.stage0 import grader


class _ImmediateProcess:
    pid = 1001
    returncode = 0

    def __init__(self) -> None:
        self.communicate_timeout: float | None = None

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        self.communicate_timeout = timeout
        return "", ""


def test_requested_six_second_budget_uses_only_the_frozen_notice_window(
    monkeypatch,
):
    process = _ImmediateProcess()
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)

    grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=6.0,
        container_name="container-r8-budget",
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
    )

    assert grader.LIFECYCLE_NOTICE_GRACE_S <= 0.25
    assert process.communicate_timeout is not None
    # The absolute enforcement instant is the registered 6.0 s deadline plus
    # the frozen window; it is never a fresh timeout from a later clock.
    assert 6.0 <= process.communicate_timeout <= (
        6.0 + grader.LIFECYCLE_NOTICE_GRACE_S + 0.05
    )
    assert process.communicate_timeout <= 6.25


class _KillProcess:
    pid = 1002
    returncode = -9

    def __init__(self) -> None:
        self.communicate_timeout: float | None = None
        self.calls = 0
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


def test_deadline_kill_happens_within_the_bounded_window(monkeypatch):
    process = _KillProcess()

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(grader.subprocess, "run", lambda *a, **k: Completed())

    started = time.perf_counter()
    result = grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=0.05,
        container_name="container-r8-kill",
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
    )
    elapsed = time.perf_counter() - started
    _code, _stdout, _stderr, timed_out, _duration, lifecycle = result

    assert process.communicate_timeout is not None
    assert process.communicate_timeout <= (
        0.05 + grader.LIFECYCLE_NOTICE_GRACE_S + 0.01
    )
    assert process.killed_at is not None
    # The kill is inside the registered deadline plus the frozen window (the
    # extra tolerance is test scheduling, not a second candidate budget).
    assert process.killed_at - started <= (
        0.05 + grader.LIFECYCLE_NOTICE_GRACE_S + 0.10
    )
    assert elapsed <= 0.05 + grader.LIFECYCLE_NOTICE_GRACE_S + 0.20
    assert timed_out is True
    assert lifecycle["late_completion"] is True


class _LateOutputProcess:
    pid = 1003
    returncode = 0

    def __init__(self) -> None:
        self.communicate_timeout: float | None = None

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        self.communicate_timeout = timeout
        # Finish after the registered deadline but inside the observation
        # window.  The bytes and exit status must never reach the outcome.
        time.sleep(0.08)
        return "PASS\n", "late candidate stderr"


def test_post_deadline_output_and_exit_status_are_excluded(monkeypatch):
    process = _LateOutputProcess()
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)

    result = grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=0.05,
        container_name="container-r8-late",
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
    )
    code, stdout, stderr, timed_out, _duration, lifecycle = result

    assert code is None
    assert stdout == ""
    assert stderr == ""
    assert timed_out is False
    assert lifecycle["completed_within_budget"] is False
    assert lifecycle["late_completion"] is True
