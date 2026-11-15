"""Fix round 12: a timely primitive cannot mask a late/in-flight launch.

The r11 review found a wrong-acceptance path.  With a 0.05 s budget the docker
client was hard-killed and ``docker kill`` returned inside the frozen
deadline + 250 ms window, but a concurrently launched ``docker rm`` returned
0.471 s after the bound.  Because at least one primitive was timely, the
attempt was classified as a conclusive candidate timeout and the loader
accepted it.

Round 12 requires *every* enforcement action (the client kill and every
container-kill primitive launch) to have returned successfully inside the
frozen bound, or the attempt is an overrun: infrastructure/unknown with cleanup
pending.  An in-flight snapshot with no return timestamp is itself an overrun.
After classification the launcher still waits a bounded interval to record each
late launch's eventual return time and result for auditability.
"""

from __future__ import annotations

import subprocess
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
    pid = 5001
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


def _run(
    monkeypatch,
    process,
    spawner,
    *,
    timeout=0.05,
    state_probe=None,
    cleanup=None,
):
    monkeypatch.setattr(grader.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(grader.threading, "Thread", _NoThreads)
    return grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=timeout,
        container_name="container-r12",
        state_probe=state_probe or (lambda name: (None, "no state")),
        cleanup=cleanup or (lambda name: True),
        enforcement_spawner=spawner,
    )


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
        "container_issued_within_bound": True,
        "any_primitive_failed": False,
        "issued": True,
        "overrun": False,
        "overrun_reason": None,
    }
    record.update(overrides)
    return record


def test_reviewer_grader_probe_is_now_an_overrun(monkeypatch):
    """The reviewer's 0.05 s probe: kill timely, rm returns ~0.47 s late.

    Enforcement starts about 0.12 s after the registered deadline, so a launch
    that sleeps 0.6 s returns roughly 0.47 s after the frozen 0.25 s bound.
    """

    process = _TimeoutProcess()
    calls: list[dict[str, float]] = []

    def spawner(command, **kwargs):
        started = time.perf_counter()
        if command[1] == "rm":
            time.sleep(0.6)
        returned = time.perf_counter()
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
    assert process.killed_at - started <= 0.300
    by_name = {call["primitive"]: call for call in calls}
    assert set(by_name) == {"rm", "kill"}

    evidence = lifecycle["deadline_enforcement"]
    bound = evidence["bound_monotonic"]
    assert evidence["client_kill"]["returned_monotonic"] <= bound
    primitives = {item["primitive"]: item for item in evidence["primitives"]}
    assert primitives["kill"]["launch_returned_monotonic"] <= bound
    # The late launch's eventual return is retained for auditability.
    assert primitives["rm"]["launch_returned_monotonic"] is not None
    assert primitives["rm"]["launch_returned_monotonic"] > bound
    assert primitives["rm"]["issued"] is True
    assert primitives["rm"]["error"] is None
    assert primitives["rm"]["launch_returned_within_bound"] is False

    # ANY late launch is an overrun; the timely alternative cannot mask it.
    assert evidence["overrun"] is True
    assert evidence["container_issued_within_bound"] is False
    assert evidence["overrun_reason"]
    assert lifecycle["deadline_enforcement_overrun"] is True
    assert lifecycle["cleanup_verified"] is False
    assert lifecycle["class"] in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
    assert lifecycle["class"] not in grader.CANDIDATE_DEADLINE_CLASSES


def test_all_timely_enforcement_is_unchanged(monkeypatch):
    """When every action returns inside the bound the attempt is not overrun."""

    process = _TimeoutProcess()
    probe_calls: list[float] = []

    def spawner(command, **kwargs):
        return _EnforcementClient()

    def state_probe(name):
        probe_calls.append(time.perf_counter())
        return None, "no state"

    result = _run(
        monkeypatch,
        process,
        spawner,
        state_probe=state_probe,
        cleanup=lambda name: True,
    )
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result

    evidence = lifecycle["deadline_enforcement"]
    bound = evidence["bound_monotonic"]
    assert evidence["overrun"] is False
    assert evidence["container_issued_within_bound"] is True
    assert evidence["any_primitive_failed"] is False
    assert lifecycle["deadline_enforcement_overrun"] is False
    assert lifecycle["cleanup_verified"] is True
    # Ordinary post-enforcement attribution still runs.
    assert probe_calls
    assert all(
        primitive["launch_returned_within_bound"] is True
        and primitive["launch_returned_monotonic"] <= bound
        for primitive in evidence["primitives"]
    )


def test_loader_refuses_in_flight_primitive_masked_by_timely_alternative():
    """The reviewer's loader probe: one primitive has no return timestamp."""

    in_flight = _record()
    in_flight["primitives"] = [
        in_flight["primitives"][0],
        {
            **in_flight["primitives"][1],
            "launch_returned_monotonic": None,
            "issued": False,
        },
    ]
    # The grader must not have recorded this as conclusive, but a forged or
    # stale row that does is still refused by the loader's recomputation.
    in_flight["overrun"] = False
    in_flight["overrun_reason"] = None
    reason = shortfall._deadline_enforcement_reason(
        in_flight, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "overrun" in reason

    # The same evidence on a would-be conclusive timeout row is refused.
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
                "deadline_enforcement": in_flight,
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
    assert "overrun" in refusal


def test_loader_still_accepts_all_timely_enforcement():
    assert (
        shortfall._deadline_enforcement_reason(
            _record(), bound_s=0.25, required=True
        )
        is None
    )

    late = _record()
    late["primitives"] = [
        {
            **late["primitives"][0],
            "launch_returned_monotonic": 10.471,
        },
        late["primitives"][1],
    ]
    late["overrun"] = True
    late["overrun_reason"] = "the rm launch returned after the frozen bound"
    reason = shortfall._deadline_enforcement_reason(
        late, bound_s=0.25, required=True
    )
    assert reason is not None
    assert "overrun" in reason or "returned after" in reason
