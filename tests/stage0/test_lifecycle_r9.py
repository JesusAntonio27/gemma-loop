"""Fix round 9: enforce termination before probing after the outer deadline.

Registered lifecycle amendment paragraph 3 keeps the single six-second outer
budget over the whole ``docker run`` without restart and allows only the frozen
bounded post-deadline window (``LIFECYCLE_NOTICE_GRACE_S``).  Once that window
expires the host must issue the forced container/client kill *first*; a Docker
state probe, a Windows ``taskkill`` wait or any cleanup step must never delay
enforcement.
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


class _SlowProbeKillProcess:
    pid = 2001
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


def test_slow_state_probe_cannot_delay_deadline_enforcement(monkeypatch):
    """Reviewer probe: a 0.4 s probe on a 0.05 s budget must not delay the kill."""

    process = _SlowProbeKillProcess()
    probe_calls: list[float] = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def state_probe(name: str):
        probe_calls.append(time.perf_counter())
        time.sleep(0.4)
        return None, "no state"

    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(grader.subprocess, "run", lambda *a, **k: Completed())
    monkeypatch.setattr(grader.threading, "Thread", _NoThreads)

    started = time.perf_counter()
    grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=0.05,
        container_name="container-r9-slow-probe",
        state_probe=state_probe,
        cleanup=lambda name: True,
    )

    assert process.communicate_timeout is not None
    assert process.communicate_timeout <= (
        0.05 + grader.LIFECYCLE_NOTICE_GRACE_S + 0.05
    )
    assert process.killed_at is not None
    # Enforcement must be issued inside the registered deadline plus the frozen
    # window plus the documented enforcement bound.  It must not include any
    # part of the 0.4 s state probe.
    assert process.killed_at - started <= (
        0.05
        + grader.LIFECYCLE_NOTICE_GRACE_S
        + grader.DEADLINE_ENFORCEMENT_EXTRA_S
        + 0.05
    )
    # The timeout-handling probe runs only after the kill was issued.
    assert probe_calls
    assert probe_calls[0] >= process.killed_at
