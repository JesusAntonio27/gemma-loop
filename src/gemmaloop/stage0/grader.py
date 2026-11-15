"""Persistent CPU grader service for Stage 0 / Stage 2 verification.

Code grading reuses the semantics of LiveCodeBench's ``run_test``
(``lcb_runner.evaluation.testing_util``): each hidden test is a separate
process, stdin tests feed the input string and compare stdout, functional tests
call ``fn_name`` with JSON-decoded arguments, and every test has its own
timeout.  The registered per-test timeout is **6 s**.

The default runtime is the Docker sandbox: CPU-only with ``--network none``, a
read-only root, a writable ``tmpfs``, memory/PID limits and a non-root user.  The
local subprocess path is an explicit, non-sandboxed fallback (``sandboxed=False``)
that requires ``allow_fallback=True``; scripts expose it only via
``--allow-unsandboxed``.  Docker is not needed for the unit tests.

Math grading follows ``gemmaloop.eval.benchmarks.aime``: strip the thinking
block, take the last balance-checked ``\\boxed{}``/``\\fbox{}`` and compare the
normalised value.  If the optional ``math_verify`` package is installed it is
used first; otherwise the AIME normaliser is the documented fallback (the repo
does not add ``math-verify`` to ``pyproject.toml``).
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import inspect
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from gemmaloop.eval.benchmarks import last_boxed, strip_thinking
from gemmaloop.eval.benchmarks.aime import _normalise_integer

DEFAULT_TIMEOUT_S = 6.0
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"
MATH_VERIFY_AVAILABLE = importlib.util.find_spec("math_verify") is not None
# The outer grading-subprocess timeout for math-verify.  math-verify's own
# timeouts use multiprocessing, which fails on Windows; the registered rule runs
# it with its internal timeouts disabled and enforces this outer limit instead.
DEFAULT_MATH_VERIFY_TIMEOUT_S = 10.0

# Trusted lifecycle classes for the registered six-second outer timeout.  The
# timeout supervises the whole ``docker run`` process (including startup); the
# clock is never restarted after the container or candidate starts.
LIFECYCLE_CANDIDATE_RUNNING = "candidate_running_at_deadline"
LIFECYCLE_CANDIDATE_EXITED_AFTER = "candidate_exited_after_deadline"
LIFECYCLE_STARTUP_FAILURE = "startup_failure"
LIFECYCLE_COMPLETED_BEFORE = "candidate_completed_before_deadline"
LIFECYCLE_UNRESOLVED = "unresolved"
LIFECYCLE_LOCAL = "local_process"
LIFECYCLE_LOCAL_TIMEOUT = "local_process_timeout"

# Trusted lifecycle evidence produced by the frozen bootstrap inside the
# container.  The grader generates a fresh per-attempt key and sends it on the
# container's standard input before the test input.  The frozen bootstrap
# (container PID 1) reads exactly 64 key bytes, forks a bootstrap child and
# only the wrapper ever sees the key afterwards.  The child performs all
# verifier setup, closes the evidence descriptor, writes an execution-entry
# frame on a private child-to-wrapper pipe and then ``exec``s the candidate.
# The wrapper turns those private frames into HMAC-SHA-256 authenticated
# records on the shared evidence mount.  Candidate stdout/stderr and container
# state alone never prove candidate lifecycle.
LIFECYCLE_EVIDENCE_FILE = "lifecycle.jsonl"
LIFECYCLE_DEADLINE_FILE = "deadline_notice.txt"
LIFECYCLE_DEADLINE_FRAME_PREFIX = "GML2"
LIFECYCLE_BUDGET_FILE = "budget.json"
LIFECYCLE_WRAPPER_NAME = "wrapper.py"
LIFECYCLE_RUNNER_NAME = "post_exec_runner.py"
LIFECYCLE_FRAME_PREFIX = "GML1"
LIFECYCLE_KEY_BYTES = 64
LIFECYCLE_BUDGET_WAIT_S = 3.0
# The host sends an authenticated deadline-reached notice at the registered
# outer deadline and only then enforces the kill.  The registered lifecycle
# amendment paragraph 3 requires the six-second outer budget over the whole
# docker run, without restart, and says post-deadline evidence may establish
# attribution and cleanup but cannot make a candidate answer acceptable.
# This bounded post-deadline observation window lets the wrapper make its
# non-blocking liveness observation and enforce its own termination, and lets
# the host kill a wrapper that misses the notice.  It is at most 250 ms, is
# part of the frozen lifecycle policy, never grants a fresh or extended
# candidate budget, and any stdout/stderr bytes or exit status observed after
# the deadline are excluded from the outcome.
LIFECYCLE_NOTICE_GRACE_S = 0.25
# The wrapper gets only this short sub-window after the registered deadline to
# make and forward its authenticated liveness observation.  The host begins
# enforcement at the end of this sub-window (comfortably before
# ``deadline + 150 ms``), so every enforcement primitive is *issued* inside the
# frozen 250 ms window even when a launch needs a fallback primitive.
LIFECYCLE_NOTICE_OBSERVATION_S = 0.12
# Both constants are frozen and recorded in the v3 freeze.  Host enforcement
# starts at the registered deadline plus the observation sub-window; the
# remaining part of the frozen window is the allowance for issuing the
# immediate primitives.
DEADLINE_ENFORCEMENT_START_S = LIFECYCLE_NOTICE_OBSERVATION_S
DEADLINE_ENFORCEMENT_EXTRA_S = (
    LIFECYCLE_NOTICE_GRACE_S - LIFECYCLE_NOTICE_OBSERVATION_S
)
# Registered lifecycle amendment paragraph 3 enforcement order: at the
# enforcement start the host FIRST issues a non-blocking forced kill/removal
# of the recorded container and hard-kills the docker client, and only
# AFTERWARDS runs a state probe (or any cleanup/wait) for attribution.  A
# state probe is never allowed to sit between the deadline and the kill.
CONTAINER_KILL_TIMEOUT_S = 0.10
# After the frozen issuance bound, keep waiting only this long (bounded) so a
# late launch's eventual return time and result can be recorded in the attempt
# evidence for auditability.  This never changes the classification: a launch
# that was in flight at the bound is already an overrun.
ENFORCEMENT_RETURN_AUDIT_S = 3.0
# The local (non-Docker) subprocess fallback still bounds its tree kill.
TASKKILL_TIMEOUT_S = 0.10
# The exact expected set of deadline-enforcement actions.  This tuple is a
# frozen constant, is recorded in the v3 freeze, and is re-checked by the
# loader.  One evidence slot per action is pre-created before any launcher
# thread starts; a missing, extra or non-terminal slot is an overrun.
CLIENT_HARD_KILL_ACTION = "client hard-kill"
DOCKER_RM_ACTION = "docker rm -f"
DOCKER_KILL_ACTION = "docker kill -s KILL"
DEADLINE_ENFORCEMENT_ACTIONS = (
    CLIENT_HARD_KILL_ACTION,
    DOCKER_RM_ACTION,
    DOCKER_KILL_ACTION,
)
# Exactly two routes may establish a conclusive outer-deadline failure.  A
# registered lifecycle timeout is definitive with established execution entry,
# a positive liveness observation at the deadline notice and verified eventual
# termination, so host enforcement is required only when the host itself has to
# enforce:
#
# * ``wrapper`` (route A): the frozen wrapper received the authenticated
#   deadline notice, observed the identified child alive, killed it and exited
#   before the host enforcement window.  The host enforces nothing; it only has
#   to observe the docker run process/container exit inside the frozen bound.
# * ``host`` (route B): the container did not exit by itself inside the
#   observation sub-window, so the host issues the complete frozen expected
#   action set and every action returns inside the frozen bound.
ENFORCEMENT_ROUTE_WRAPPER = "wrapper"
ENFORCEMENT_ROUTE_HOST = "host"
DEADLINE_ENFORCEMENT_ROUTES = (
    ENFORCEMENT_ROUTE_WRAPPER,
    ENFORCEMENT_ROUTE_HOST,
)
# Deadline-enforcement launches must stay concurrent even when a test (or an
# embedder) replaces ``threading.Thread`` to silence the lifecycle monitor.
# Capture the real constructor at import time for those short-lived launchers.
_ENFORCEMENT_THREAD_CLASS = threading.Thread


def _assert_lifecycle_enforcement_bounds() -> None:
    """Fail closed if the frozen observation/enforcement ordering drifts."""

    ok = (
        0.0 < LIFECYCLE_NOTICE_OBSERVATION_S <= 0.15
        and 0.0 < DEADLINE_ENFORCEMENT_START_S
        and DEADLINE_ENFORCEMENT_START_S == LIFECYCLE_NOTICE_OBSERVATION_S
        and DEADLINE_ENFORCEMENT_START_S <= 0.15
        and DEADLINE_ENFORCEMENT_START_S < LIFECYCLE_NOTICE_GRACE_S
        and DEADLINE_ENFORCEMENT_START_S + DEADLINE_ENFORCEMENT_EXTRA_S
        <= LIFECYCLE_NOTICE_GRACE_S
    )
    if not ok:
        raise RuntimeError(
            "lifecycle enforcement ordering violates the frozen bound: "
            f"observation={LIFECYCLE_NOTICE_OBSERVATION_S}s, "
            f"start={DEADLINE_ENFORCEMENT_START_S}s, "
            f"grace={LIFECYCLE_NOTICE_GRACE_S}s"
        )


_assert_lifecycle_enforcement_bounds()


def wrapper_enforcement_route_verified(
    *,
    lifecycle_valid: bool | None,
    start_received_within_budget: bool | None,
    deadline_alive: bool | None,
    termination_verified: bool | None,
    cleanup_verified: bool | None,
    host_exit_monotonic: float | None,
    host_deadline_monotonic: float | None,
    bound_s: float = LIFECYCLE_NOTICE_GRACE_S,
) -> bool:
    """Return True only for the wrapper (route A) enforcement evidence.

    Route A requires the authenticated wrapper record (entry received by the
    host before the deadline, a positive liveness observation at the
    authenticated deadline notice and the wrapper's kill of the identified
    child), verified termination/cleanup, and the host's own observation of the
    docker run process/container exit no later than ``deadline + bound_s``.  It
    deliberately does **not** require any host enforcement action: the wrapper
    already enforced the kill before the host window opened.
    """

    if lifecycle_valid is not True:
        return False
    if start_received_within_budget is not True:
        return False
    if deadline_alive is not True:
        return False
    if termination_verified is not True:
        return False
    if cleanup_verified is not True:
        return False
    if host_exit_monotonic is None or host_deadline_monotonic is None:
        return False
    try:
        exit_monotonic = float(host_exit_monotonic)
        deadline_monotonic = float(host_deadline_monotonic)
        window = float(bound_s)
    except (TypeError, ValueError):
        return False
    if window <= 0.0:
        return False
    return exit_monotonic <= deadline_monotonic + window + 1e-6


LIFECYCLE_EVENT_READY = "bootstrap_ready"
LIFECYCLE_EVENT_BUDGET = "budget_applied"
LIFECYCLE_EVENT_START = "candidate_start"
LIFECYCLE_EVENT_FINISH = "candidate_finish"
LIFECYCLE_EVENT_ALIVE = "candidate_alive_at_deadline_notice"
LIFECYCLE_EVENT_EXITED_AT_NOTICE = "candidate_exited_at_deadline_notice"
LIFECYCLE_EVENT_HANDOFF_FAILED = "handoff_failed"
LIFECYCLE_EVENT_BOOTSTRAP_FAILURE = "bootstrap_failure"

# The classes that are a registered outer-deadline (candidate) failure rather
# than infrastructure/unknown.  Startup failures, completion before the
# deadline and unresolved attribution stay infrastructure/unknown.
CANDIDATE_DEADLINE_CLASSES = frozenset(
    {LIFECYCLE_CANDIDATE_RUNNING, LIFECYCLE_CANDIDATE_EXITED_AFTER}
)
INFRASTRUCTURE_LIFECYCLE_CLASSES = frozenset(
    {
        LIFECYCLE_STARTUP_FAILURE,
        LIFECYCLE_COMPLETED_BEFORE,
        LIFECYCLE_UNRESOLVED,
    }
)
# Classes that a complete, authentic lifecycle record can produce; a Docker
# outcome is conclusive only when its class is one of these and the record
# itself validates.
TRUSTED_LIFECYCLE_CLASSES = frozenset(
    {
        LIFECYCLE_CANDIDATE_RUNNING,
        LIFECYCLE_CANDIDATE_EXITED_AFTER,
        LIFECYCLE_COMPLETED_BEFORE,
    }
)

MATH_VERIFY_SCRIPT = r"""
import sys

from math_verify import parse, verify

candidate, answer = sys.argv[1], sys.argv[2]
try:
    gold = parse(answer, parsing_timeout=None)
    target = parse(candidate, parsing_timeout=None)
    result = bool(verify(gold, target, timeout_seconds=None))
except BaseException:
    print("MATH_VERIFY_ERROR")
    raise SystemExit(3)
print("MATH_VERIFY_TRUE" if result else "MATH_VERIFY_FALSE")
"""
_SAFE_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
)


def sanitized_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for candidate code: no secrets, no tokens, no API keys."""

    source = os.environ if environ is None else environ
    return {key: source[key] for key in _SAFE_ENV_KEYS if key in source}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    index: int
    passed: bool
    input: str
    expected: str
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    exit_code: int | None = None
    # Trusted lifecycle evidence (Docker state, never candidate output).
    lifecycle_class: str | None = None
    container_name: str | None = None
    container_started_at: str | None = None
    container_finished_at: str | None = None
    lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_markers: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_records: list[dict[str, Any]] = field(default_factory=list)
    # Exact HMAC-framed bytes retained so a loader can re-authenticate an
    # attempt after it ended.  The per-attempt key is only persisted after the
    # candidate process has exited and is never visible inside the container.
    lifecycle_frames_raw: list[str] = field(default_factory=list)
    lifecycle_auth_key: str | None = None
    marker_error: str | None = None
    # True only when the whole docker run reached the host inside the
    # registered outer budget.
    completed_within_budget: bool | None = None
    test_type: str = "stdin"
    # True only when the lifecycle record was complete, authentic and ordered.
    # Every Docker outcome -- pass, wrong answer, candidate error or timeout --
    # requires this before it may be accepted as conclusive.
    lifecycle_valid: bool | None = None
    cleanup_verified: bool | None = None
    # State-specific lifecycle facts.  A normal result needs a successful
    # handoff and a genuine child exit; a timeout also needs a positive
    # non-exit observation at/after the deadline and verified termination.
    handoff_success: bool | None = None
    termination_verified: bool | None = None
    deadline_alive: bool | None = None
    lifecycle_attempt: str | None = None
    exit_observed: bool | None = None
    # Host-side receipt of the execution-entry record (same host monotonic
    # clock as the registered deadline).  A deadline failure requires the
    # entry evidence to have reached the host before its deadline.
    start_received_within_budget: bool | None = None
    start_received_monotonic: float | None = None
    host_deadline_monotonic: float | None = None
    # Host monotonic issuance times for every deadline-enforcement action
    # (client hard kill, each container-kill primitive launch start/return) and
    # the recorded enforcement route (``wrapper`` or ``host``).
    deadline_enforcement: dict[str, Any] = field(default_factory=dict)
    # Exactly one of the two accepted enforcement routes for a conclusive
    # outer-deadline failure, or ``None`` when neither route is established.
    enforcement_route: str | None = None
    # Host monotonic time at which the docker run process/container was
    # observed to exit.  Route A requires it no later than deadline + bound.
    host_exit_monotonic: float | None = None
    # Attempt history for the precommitted infrastructure retry policy.
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GradeResult:
    passed: bool
    n_tests: int
    passed_tests: int
    sandboxed: bool
    tests: list[TestResult] = field(default_factory=list)
    error: str | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    sandbox: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_tests": self.n_tests,
            "passed_tests": self.passed_tests,
            "sandboxed": self.sandboxed,
            "sandbox": self.sandbox,
            "error": self.error,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "tests": [test.__dict__ for test in self.tests],
        }


@dataclass
class MathGrade:
    correct: bool
    predicted: str | None
    normalized_prediction: str | None
    normalized_answer: str
    method: str
    parsed: bool
    # Which verifier produced the decision: math_verify / normaliser / both /
    # timeout (the outer subprocess timeout fired; the normaliser still votes).
    decided_by: str = "normaliser"
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------
def normalize_output(text: str) -> str:
    """Comparison form used for stdout: strip trailing spaces/newlines."""

    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def _resource_limits(memory_mb: int, cpu_seconds: int):
    """POSIX-only ``preexec_fn``; ignored on Windows."""

    if os.name == "nt":
        return None

    def _limit() -> None:  # pragma: no cover - POSIX only
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if memory_mb > 0:
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return _limit


def _container_enforcement_plan(
    container_name: str,
) -> tuple[tuple[str, list[str]], ...]:
    """The exact container-kill actions and their commands, in issuance order.

    ``docker rm -f`` is the primary forced removal/kill primitive; ``docker
    kill -s KILL`` is the alternative that stops the container even when
    removal cannot start.  Both are launched without waiting so nothing slow
    can sit between the deadline and enforcement.  The action names are the
    frozen loader-visible names, not the shorter process labels.
    """

    name = str(container_name)
    return (
        (DOCKER_RM_ACTION, ["docker", "rm", "-f", name]),
        (DOCKER_KILL_ACTION, ["docker", "kill", "-s", "KILL", name]),
    )


def _container_enforcement_commands(
    container_name: str,
) -> tuple[list[str], ...]:
    """Immediate container-kill primitive commands, in issuance order."""

    return tuple(
        command for _action, command in _container_enforcement_plan(container_name)
    )


def _enforcement_client_launch_error(client: Any) -> str | None:
    """Return an immediate launch error, or ``None`` for a live client.

    A real ``Popen`` reports ``returncode is None`` until it is reaped.  Tests
    and alternate spawners may return a completed result immediately; a
    non-zero code from such a synchronous result means the primitive was not
    accepted and the alternative must be issued at once.
    """

    if client is None:
        return "spawner returned no process"
    code = getattr(client, "returncode", None)
    if code is None or callable(getattr(client, "communicate", None)):
        return None
    try:
        if int(code) != 0:
            return f"spawner returned error code {code}"
    except (TypeError, ValueError):
        return None
    return None


def _launch_enforcement_primitive(
    command: Sequence[str],
    slot: dict[str, Any],
    *,
    spawner: Any,
    round_index: int,
) -> None:
    """Launch one container-kill primitive and stamp its issuance times.

    The launched process is not awaited here.  ``launch_started_monotonic`` is
    stamped immediately before the spawner call and
    ``launch_returned_monotonic`` immediately after it returns or raises, so a
    slow launch is visible in the attempt evidence and can be checked against
    the frozen post-deadline bound.
    """

    # The slot was pre-created by the collector with a "scheduled" status and
    # a scheduling timestamp.  Workers only fill their own pre-created slot;
    # they never create or drop one.
    slot.update(
        {
            "status": "running",
            "round": int(round_index),
            "launch_started_monotonic": time.perf_counter(),
        }
    )
    client: Any = None
    error: str | None = None
    try:
        client = spawner(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=sanitized_env(),
        )
        error = _enforcement_client_launch_error(client)
    except Exception as exc:  # noqa: BLE001 - the alternative primitive runs too
        error = (
            f"{command[1] if len(command) > 1 else 'docker'}: "
            f"{type(exc).__name__}: {exc}"
        )
    slot.update(
        {
            "launch_returned_monotonic": time.perf_counter(),
            "issued": error is None,
            "returned": True,
            "status": "failed" if error is not None else "returned",
            "error": error,
            "_client": client if error is None else None,
        }
    )


def _action_slot_terminal(slot: Any) -> bool:
    """Return True only when an action slot reached a terminal state."""

    if not isinstance(slot, dict):
        return False
    if slot.get("returned") is not True:
        return False
    status = slot.get("status")
    return status in {None, "returned", "failed"}


def _action_slot_late_or_in_flight(
    slot: Any, bound_monotonic: float | None
) -> bool:
    """Return True when an action was late, scheduled or still running.

    A launch that failed quickly is terminally recorded but is not a retry
    blocker; it is still an overrun through the failure rule.
    """

    if not _action_slot_terminal(slot):
        return True
    returned = slot.get("launch_returned_monotonic")
    if returned is None:
        returned = slot.get("returned_monotonic")
    if returned is None:
        return True
    if bound_monotonic is None:
        return False
    try:
        return float(returned) > bound_monotonic
    except (TypeError, ValueError):
        return True


def _action_slot_within_bound(
    slot: Any, bound_monotonic: float | None
) -> bool:
    """Return True only for a terminal, successful slot returned in time.

    A "scheduled" slot whose worker never entered its target, a "running" slot
    whose launch never returned, and a "failed" slot are all non-conforming.
    """

    if not _action_slot_terminal(slot):
        return False
    if slot.get("status") != "returned":
        return False
    if slot.get("issued") is not True or slot.get("error") is not None:
        return False
    returned = slot.get("launch_returned_monotonic")
    if returned is None:
        returned = slot.get("returned_monotonic")
    if returned is None:
        return False
    try:
        returned_value = float(returned)
    except (TypeError, ValueError):
        return False
    return bound_monotonic is None or returned_value <= bound_monotonic


def _issue_container_enforcement(
    container_name: str,
    *,
    spawner: Any = None,
    deadline_monotonic: float | None = None,
) -> tuple[list[Any], list[str], bool, list[dict[str, Any]]]:
    """Issue every container-kill primitive concurrently, not in sequence.

    Both immediate primitives are started in their own threads at the same
    instant, so a slow first launch can never sit in front of the alternative.
    Every launch is classified at the frozen bound: any primitive that has not
    returned by then (or that failed) is an overrun, even when its concurrent
    alternative returned in time.  After that classification the launcher keeps
    waiting only for the bounded audit interval
    (``ENFORCEMENT_RETURN_AUDIT_S``) so each late launch's eventual return time
    and result can be recorded.  The whole command set is retried once when
    *no* primitive could be launched early (matching the previous fallback
    rule), but a retry never follows a launch that was in flight or late at the
    frozen bound.
    """

    spawner = subprocess.Popen if spawner is None else spawner
    plan = _container_enforcement_plan(container_name)
    bound_monotonic = (
        None
        if deadline_monotonic is None
        else float(deadline_monotonic) + float(LIFECYCLE_NOTICE_GRACE_S)
    )
    clients: list[Any] = []
    errors: list[str] = []
    evidence: list[dict[str, Any]] = []
    issued = False
    for round_index in range(2):
        # Pre-create one evidence slot per expected action BEFORE starting any
        # launcher thread.  A scheduler delay or a hung spawner can therefore
        # never make an action disappear from the recorded evidence.
        scheduled = time.perf_counter()
        slots: list[dict[str, Any]] = [
            {
                "action": action,
                "command": list(command),
                "primitive": (
                    command[1] if len(command) > 1 else "docker"
                ),
                "round": int(round_index),
                "status": "scheduled",
                "scheduled_monotonic": scheduled,
                "launch_started_monotonic": None,
                "launch_returned_monotonic": None,
                "issued": False,
                "returned": False,
                "error": None,
                "late_or_in_flight_at_bound": False,
            }
            for action, command in plan
        ]
        threads: list[threading.Thread] = []
        for index, (_action, command) in enumerate(plan):
            thread = _ENFORCEMENT_THREAD_CLASS(
                target=_launch_enforcement_primitive,
                args=(command, slots[index]),
                kwargs={"spawner": spawner, "round_index": round_index},
                name=(
                    "container-enforcement-"
                    f"{command[1] if len(command) > 1 else 'docker'}"
                ),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        # Wait for launches only up to the frozen bound.  A launch that has not
        # entered its target or returned by then is already an overrun and can
        # never become a conclusive outcome.
        for thread in threads:
            if bound_monotonic is None:
                remaining = CONTAINER_KILL_TIMEOUT_S
            else:
                remaining = bound_monotonic - time.perf_counter()
            if remaining > 0.0:
                thread.join(timeout=remaining)
        # Fix the late/in-flight classification at the bound and carry it on
        # every slot.  A slot that was in flight here can never become timely
        # even if the bounded audit wait later records its return.
        for slot in slots:
            slot["late_or_in_flight_at_bound"] = (
                _action_slot_late_or_in_flight(slot, bound_monotonic)
            )
        late_or_in_flight = any(
            bool(slot.get("late_or_in_flight_at_bound")) for slot in slots
        )
        # Classification is fixed at the bound.  Keep waiting afterwards only
        # for a bounded interval to record each late launch's eventual return
        # time and result for auditability.
        if late_or_in_flight:
            audit_deadline = time.perf_counter() + float(
                ENFORCEMENT_RETURN_AUDIT_S
            )
            for thread in threads:
                remaining = audit_deadline - time.perf_counter()
                if remaining > 0.0 and (
                    not callable(getattr(thread, "is_alive", None))
                    or thread.is_alive()
                ):
                    thread.join(timeout=remaining)
        round_issued = False
        for slot in slots:
            # Never drop a slot: a delayed worker's pre-created "scheduled"
            # slot is collected as-is and is itself an overrun.
            snapshot = {
                key: value
                for key, value in slot.items()
                if key != "_client"
            }
            snapshot["terminal"] = _action_slot_terminal(snapshot)
            snapshot["launch_returned_within_bound"] = (
                _action_slot_within_bound(snapshot, bound_monotonic)
            )
            evidence.append(snapshot)
            if slot.get("issued"):
                round_issued = True
                if slot.get("_client") is not None:
                    clients.append(slot["_client"])
            elif slot.get("error"):
                errors.append(str(slot["error"]))
        issued = issued or round_issued
        if round_issued:
            break
        if late_or_in_flight:
            # Do not retry behind a launch that was in flight or late at the
            # frozen bound; that launch is the overrun this path fails closed
            # on.
            break
    return clients, errors, issued, evidence


def _kill_enforcement_client(process: Any) -> tuple[bool, str | None]:
    """Hard-kill the docker client immediately (no new process, no wait)."""

    already_gone = getattr(process, "returncode", None) is not None
    error: str | None = None
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except Exception as exc:  # noqa: BLE001 - fall back to the direct kill
            try:
                process.kill()
            except Exception as direct_exc:  # noqa: BLE001 - may already be gone
                error = (
                    f"{type(exc).__name__}: {exc}; "
                    f"{type(direct_exc).__name__}: {direct_exc}"
                )
    else:
        try:
            process.kill()
        except Exception as exc:  # noqa: BLE001 - may already be gone
            error = f"{type(exc).__name__}: {exc}"
    issued = error is None or already_gone
    return issued, None if issued else error


def _primitive_returned_within_bound(
    record: Any, bound_monotonic: float | None
) -> bool:
    """Return True only for a successful primitive that returned in time.

    An in-flight or still-scheduled snapshot has no return timestamp and is
    therefore a failure of the frozen issuance bound, not a partial success.
    """

    return _action_slot_within_bound(record, bound_monotonic)


def _enforce_deadline_termination(
    process: Any,
    container_name: str | None,
    *,
    spawner: Any = None,
    deadline_monotonic: float | None = None,
) -> tuple[list[Any], bool, list[str], dict[str, Any]]:
    """Enforce termination before any post-deadline observation.

    Order at enforcement time:

    1. hard-kill the docker client immediately (``Popen.kill`` /
       ``TerminateProcess`` on Windows), never a ``taskkill`` wait;
    2. launch the container-kill primitives concurrently, each in its own
       thread started at the same instant, never sequentially behind a slow
       launch and never behind a state probe;
    3. only then is the caller allowed to run a state probe or cleanup.

    The registered deadline plus ``LIFECYCLE_NOTICE_GRACE_S`` is the frozen
    issuance bound.  Every action's host monotonic issuance time is recorded in
    the returned evidence.  The attempt overruns if the client kill is not
    returned inside the bound, if *any* container-kill primitive is not
    returned inside the bound (a timely alternative cannot mask another launch
    that was still in flight), or if any primitive failed.  On an overrun the
    caller must keep the attempt infrastructure/unknown with cleanup pending.
    """

    _assert_lifecycle_enforcement_bounds()
    bound_monotonic = (
        None
        if deadline_monotonic is None
        else float(deadline_monotonic) + float(LIFECYCLE_NOTICE_GRACE_S)
    )
    # 1. Pre-create the client-kill evidence slot before any launcher thread
    # starts, then hard-kill the docker client.  This is a system call and does
    # not create a process, so it cannot be delayed by a slow launch.
    client_scheduled = time.perf_counter()
    client_slot: dict[str, Any] = {
        "action": CLIENT_HARD_KILL_ACTION,
        "status": "scheduled",
        "scheduled_monotonic": client_scheduled,
        "started_monotonic": None,
        "returned_monotonic": None,
        "issued": False,
        "returned": False,
        "error": None,
        "late_or_in_flight_at_bound": False,
    }
    client_started = time.perf_counter()
    client_issued, client_error = _kill_enforcement_client(process)
    client_returned = time.perf_counter()
    client_slot.update(
        {
            "status": "returned" if client_issued else "failed",
            "started_monotonic": client_started,
            "returned_monotonic": client_returned,
            "issued": bool(client_issued),
            "returned": True,
            "error": client_error,
        }
    )
    client_within_bound = _action_slot_within_bound(
        client_slot, bound_monotonic
    )
    client_slot["within_bound"] = client_within_bound
    client_slot["terminal"] = _action_slot_terminal(client_slot)
    client_slot["late_or_in_flight_at_bound"] = (
        _action_slot_late_or_in_flight(client_slot, bound_monotonic)
    )
    # 2. Container-kill primitives launch concurrently.  Their slots are also
    # pre-created before any thread starts.
    enforcement_clients: list[Any] = []
    enforcement_errors: list[str] = []
    container_issued = not bool(container_name)
    primitive_evidence: list[dict[str, Any]] = []
    if container_name:
        (
            enforcement_clients,
            enforcement_errors,
            container_issued,
            primitive_evidence,
        ) = _issue_container_enforcement(
            str(container_name),
            spawner=spawner,
            deadline_monotonic=deadline_monotonic,
        )
    # Every recorded primitive must have returned successfully inside the
    # bound.  A timely alternative never masks another launch that was still in
    # flight (or failed) at the bound.
    container_issued_within_bound = (not bool(container_name)) or any(
        _primitive_returned_within_bound(record, bound_monotonic)
        for record in primitive_evidence
    )
    container_within_bound = (not bool(container_name)) or all(
        _primitive_returned_within_bound(record, bound_monotonic)
        for record in primitive_evidence
    )
    any_primitive_failed = any(
        record.get("issued") is not True or record.get("error") is not None
        for record in primitive_evidence
    )
    # The loader-visible expected action set is exact.  Missing, extra or
    # duplicated action names are refused; every slot must be terminal and
    # must have returned inside the frozen bound.  Every earlier
    # late/in-flight observation is carried into this final computation.
    expected_actions = list(DEADLINE_ENFORCEMENT_ACTIONS)
    action_slots = [client_slot, *primitive_evidence]
    recorded_actions = [
        str(slot.get("action"))
        for slot in action_slots
        if slot.get("action") is not None
    ]
    recorded_set = set(recorded_actions)
    missing_actions = [
        action for action in expected_actions if action not in recorded_set
    ]
    extra_actions = sorted(recorded_set - set(expected_actions))
    actions_complete = not missing_actions and not extra_actions
    any_action_not_terminal = any(
        not _action_slot_terminal(slot) for slot in action_slots
    )
    any_late_or_in_flight = any(
        bool(slot.get("late_or_in_flight_at_bound")) for slot in action_slots
    )
    all_actions_within_bound = actions_complete and all(
        _action_slot_within_bound(slot, bound_monotonic)
        for slot in action_slots
    )
    container_issued = container_issued_within_bound
    enforcement_issued = client_within_bound and container_issued_within_bound
    overrun = (
        not client_within_bound
        or not container_within_bound
        or any_primitive_failed
        or not actions_complete
        or any_action_not_terminal
        or any_late_or_in_flight
        or not all_actions_within_bound
    )
    overrun_reason: str | None = None
    if overrun:
        details: list[str] = []
        if missing_actions:
            details.append(
                "missing expected action(s): " + ", ".join(missing_actions)
            )
        if extra_actions:
            details.append(
                "unexpected action(s): " + ", ".join(extra_actions)
            )
        if not client_within_bound:
            details.append("docker client kill")
        for record in primitive_evidence:
            if _primitive_returned_within_bound(record, bound_monotonic):
                continue
            name = str(
                record.get("action")
                or record.get("primitive")
                or (record.get("command") or ["docker"])[0]
            )
            status = record.get("status")
            if status == "scheduled":
                details.append(
                    f"{name} was still scheduled (its worker never entered "
                    "the target)"
                )
            elif status == "running":
                details.append(
                    f"{name} was still running (its launch never returned)"
                )
            elif record.get("issued") is not True:
                if record.get("error"):
                    details.append(f"{name} failed: {record['error']}")
                else:
                    details.append(
                        f"{name} had not returned by the frozen bound"
                    )
            elif record.get("launch_returned_monotonic") is None:
                details.append(
                    f"{name} had no return timestamp at the frozen bound"
                )
            else:
                details.append(
                    f"{name} returned after the frozen bound "
                    f"({float(record['launch_returned_monotonic']) - float(bound_monotonic):.6f}s late)"
                )
        if any_late_or_in_flight and not details:
            details.append(
                "an action was late or in flight at the frozen bound"
            )
        if any_action_not_terminal and not details:
            details.append("an action never reached a terminal state")
        if not details:
            details.append("deadline enforcement action set")
        overrun_reason = (
            "deadline enforcement did not return every expected action "
            f"inside the frozen {float(LIFECYCLE_NOTICE_GRACE_S):g}s window: "
            + ", ".join(details)
        )
    evidence: dict[str, Any] = {
        "triggered": True,
        "bound_s": float(LIFECYCLE_NOTICE_GRACE_S),
        "deadline_monotonic": (
            None if deadline_monotonic is None else float(deadline_monotonic)
        ),
        "bound_monotonic": bound_monotonic,
        "expected_actions": list(DEADLINE_ENFORCEMENT_ACTIONS),
        "actions": [dict(client_slot), *primitive_evidence],
        "actions_complete": actions_complete,
        "missing_actions": missing_actions,
        "extra_actions": extra_actions,
        "any_action_not_terminal": any_action_not_terminal,
        "any_late_or_in_flight": any_late_or_in_flight,
        "client_kill": dict(client_slot),
        "primitives": primitive_evidence,
        "container_issued": container_issued,
        "container_issued_within_bound": container_within_bound,
        "any_primitive_failed": any_primitive_failed,
        "issued": enforcement_issued,
        "overrun": overrun,
        "overrun_reason": overrun_reason,
        "errors": list(enforcement_errors),
    }
    # Reap the enforcement clients only after every primitive was issued.
    # Each wait is bounded; a stuck client is hard-killed.
    for client in enforcement_clients:
        try:
            wait = getattr(client, "wait", None)
            if callable(wait):
                try:
                    wait(timeout=CONTAINER_KILL_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - hard kill below
                    killer = getattr(client, "kill", None)
                    if callable(killer):
                        try:
                            killer()
                        except Exception:  # noqa: BLE001
                            pass
            else:
                poll = getattr(client, "poll", None)
                if callable(poll):
                    try:
                        poll()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001 - enforcement already issued
            pass
    return enforcement_clients, enforcement_issued, enforcement_errors, evidence


def _run_process(
    command: Sequence[str],
    *,
    input_text: str,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, bool, float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=sanitized_env() if env is None else env,
        preexec_fn=_resource_limits(1024, max(1, int(timeout) + 1)),
        start_new_session=(os.name != "nt"),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input_text, timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except Exception:
                process.kill()
        else:
            # Kill the whole tree: a candidate that spawns children must not
            # outlive the registered per-test timeout.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    env=sanitized_env(),
                    timeout=TASKKILL_TIMEOUT_S,
                )
            except Exception:
                # A timed-out or unavailable taskkill is followed by the hard
                # kill immediately; the wait is bounded.
                process.kill()
            try:
                process.kill()
            except Exception:
                pass
        stdout, stderr = process.communicate()
        code = process.returncode
    return code, stdout or "", stderr or "", timed_out, time.perf_counter() - started


def _parse_docker_timestamp(value: Any) -> float | None:
    """Parse a Docker RFC3339 timestamp into a POSIX epoch, or ``None``."""

    text = str(value or "").strip()
    if not text or text.startswith("0001-01-01"):
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _events(records: Sequence[Mapping[str, Any]] | None, name: str) -> list[Mapping[str, Any]]:
    return [record for record in (records or []) if record.get("event") == name]


def _record_number(record: Mapping[str, Any], key: str) -> float | None:
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def _notice_time(
    record: Mapping[str, Any], budget: Mapping[str, Any] | None
) -> float | None:
    """Return the causal deadline time for a liveness observation.

    New records carry the wrapper monotonic time at which it verified the
    host's authenticated deadline notice.  The ``deadline_monotonic`` fallback
    exists only for older synthetic fixtures; the runtime never compares the
    host epoch with the container clock.
    """

    value = _record_number(record, "notice_monotonic")
    if value is not None:
        return value
    return None


def _host_start_receipt_ok(
    host_start_received_within_budget: bool | None,
    start_mono: float | None,
    deadline_mono: float | None,
) -> bool:
    """Whether the host proved it received execution entry before its deadline.

    Production records have no container-clock deadline (the wrapper never
    compares clocks); the host stamps the moment its monitor read the
    authenticated ``candidate_start`` record and compares that with the same
    host monotonic clock used for the registered deadline.  Legacy synthetic
    fixtures may still carry ``deadline_monotonic`` in the budget record; that
    fallback keeps their meaning but is never used by the live wrapper.
    """

    if host_start_received_within_budget is True:
        return True
    if host_start_received_within_budget is False:
        return False
    return (
        deadline_mono is not None
        and start_mono is not None
        and start_mono <= deadline_mono
    )


def lifecycle_record_error(
    *,
    markers: Sequence[Mapping[str, Any]] | None = None,
    marker_error: str | None = None,
    cleanup_verified: bool | None = None,
    deadline_epoch: float | None = None,
    timed_out: bool = False,
    host_completed_within_budget: bool | None = None,
    host_start_received_within_budget: bool | None = None,
) -> str | None:
    """Return why a lifecycle record is not complete/valid, or ``None``.

    Complete is state-specific.  Every conclusive Docker outcome requires
    verified cleanup, exactly one authenticated successful-handoff record, a
    consistent child identity and a budget record proving the execution entry
    boundary was established before the deadline.  A normal result additionally
    requires a genuine child exit record that reached the host inside the
    registered budget; a timeout requires a positive non-exit observation made
    *after* the authenticated deadline-reached notice arrived and a verified
    termination record.  The host and container monotonic clocks are never
    compared with each other.
    """

    if cleanup_verified is not True:
        return "cleanup was not verified"
    if marker_error:
        return str(marker_error)
    if markers is None:
        return "no lifecycle records"
    failures = _events(markers, LIFECYCLE_EVENT_HANDOFF_FAILED)
    if failures:
        reason = failures[0].get("error") or failures[0].get("reason")
        return f"execution handoff failed: {reason or 'no detail'}"
    boot_failures = _events(markers, LIFECYCLE_EVENT_BOOTSTRAP_FAILURE)
    if boot_failures:
        reason = boot_failures[0].get("reason") or "bootstrap failure"
        return f"bootstrap failed before execution: {reason}"
    budgets = _events(markers, LIFECYCLE_EVENT_BUDGET)
    if len(budgets) != 1:
        return f"expected exactly one budget record, found {len(budgets)}"
    budget = budgets[0]
    deadline_mono = _record_number(budget, "deadline_monotonic")
    starts = _events(markers, LIFECYCLE_EVENT_START)
    finishes = _events(markers, LIFECYCLE_EVENT_FINISH)
    alives = _events(markers, LIFECYCLE_EVENT_ALIVE)
    exited_notice = _events(markers, LIFECYCLE_EVENT_EXITED_AT_NOTICE)
    if len(starts) != 1:
        return f"expected exactly one successful execution-entry record, found {len(starts)}"
    if len(finishes) > 1:
        return f"expected at most one candidate_finish record, found {len(finishes)}"
    if len(alives) > 1:
        return f"expected at most one deadline liveness record, found {len(alives)}"
    if len(exited_notice) > 1:
        return (
            "expected at most one deadline-notice exit observation, found "
            f"{len(exited_notice)}"
        )
    start = starts[0]
    start_mono = _record_number(start, "monotonic")
    child_pid = start.get("child_pid")
    if start_mono is None or child_pid is None:
        return "execution-entry record has no child identity or monotonic time"
    if deadline_mono is not None and start_mono > deadline_mono:
        return "execution entry was established after the wrapper deadline"
    for record in (*finishes, *alives, *exited_notice):
        if record.get("child_pid") != child_pid:
            return "lifecycle record is not bound to the tracked child"
    alive_mono = None
    if alives:
        alive = alives[0]
        alive_mono = _record_number(alive, "monotonic")
        notice_mono = _notice_time(alive, budget)
        if alive_mono is None or notice_mono is None:
            return "deadline liveness observation has no causal notice time"
        if alive_mono < notice_mono:
            return "deadline liveness observation precedes the deadline notice"
        if not _host_start_receipt_ok(
            host_start_received_within_budget, start_mono, deadline_mono
        ):
            return (
                "execution entry was not received by the host before its "
                "registered deadline"
            )
        if start_mono > notice_mono:
            return "execution entry was established after the deadline notice"
        if not finishes:
            return "candidate was alive at the deadline but termination was not verified"
        if finishes[0].get("terminated") is not True:
            return "candidate termination was not verified after the deadline"
        finish_mono = _record_number(finishes[0], "monotonic")
        if finish_mono is None or finish_mono < alive_mono:
            return "candidate_finish precedes the deadline liveness observation"
        if finishes[0].get("exit_observed") is not True:
            return "candidate exit status was not observed after the deadline"
        return None
    if exited_notice:
        return (
            "candidate exited at the deadline notice; its completion did not "
            "reach the host before the deadline"
        )
    if finishes:
        finish = finishes[0]
        finish_mono = _record_number(finish, "monotonic")
        if finish_mono is None or finish_mono < start_mono:
            return "candidate_finish precedes the successful execution entry"
        if finish.get("exit_observed") is not True:
            return "candidate exit status was not observed"
        if alive_mono is not None and finish_mono < alive_mono:
            return "candidate_finish precedes the deadline liveness observation"
        if host_completed_within_budget is False:
            return "candidate completion did not reach the host before its deadline"
        if (
            host_completed_within_budget is None
            and deadline_mono is not None
            and finish_mono > deadline_mono
        ):
            # The exit interval spans the deadline; a delayed finish record
            # does not prove the candidate was still running at the deadline.
            return "candidate exit interval spans the wrapper deadline"
    elif not timed_out:
        return "no candidate exit record for a completed outcome"
    if timed_out and not alives:
        return "timeout has no positive candidate-liveness observation at the deadline notice"
    return None


def classify_lifecycle_evidence(
    *,
    markers: Sequence[Mapping[str, Any]] | None = None,
    marker_error: str | None = None,
    timed_out: bool = False,
    cleanup_verified: bool | None = None,
    deadline_epoch: float,
    status: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    probe_error: str | None = None,
    host_completed_within_budget: bool | None = None,
    host_start_received_within_budget: bool | None = None,
) -> str:
    """Classify trusted wrapper lifecycle records causally.

    A candidate timeout requires the wrapper's positive non-exit observation
    made after the host's authenticated deadline-reached notice.  The host's
    own registered deadline is the only timeout clock; it is never restarted
    and the wrapper computes no deadline of its own.  Legacy marker-shaped
    records (epoch only) are still classified with the earlier epoch rule so
    synthetic classifier fixtures remain meaningful.
    """

    if cleanup_verified is not True:
        return LIFECYCLE_UNRESOLVED
    if marker_error or markers is None:
        if probe_error:
            return LIFECYCLE_UNRESOLVED
        if status in {None, "created", "restarting", "exited", "dead"}:
            return LIFECYCLE_STARTUP_FAILURE
        return LIFECYCLE_UNRESOLVED
    new_style = any(
        "monotonic" in record or record.get("event") in {
            LIFECYCLE_EVENT_READY,
            LIFECYCLE_EVENT_BUDGET,
            LIFECYCLE_EVENT_ALIVE,
            LIFECYCLE_EVENT_EXITED_AT_NOTICE,
            LIFECYCLE_EVENT_HANDOFF_FAILED,
            LIFECYCLE_EVENT_BOOTSTRAP_FAILURE,
        }
        for record in markers
    )
    if not new_style:
        return _classify_legacy_lifecycle(
            markers, timed_out=timed_out, deadline_epoch=deadline_epoch
        )
    if _events(markers, LIFECYCLE_EVENT_HANDOFF_FAILED):
        return LIFECYCLE_UNRESOLVED
    if _events(markers, LIFECYCLE_EVENT_BOOTSTRAP_FAILURE):
        return LIFECYCLE_STARTUP_FAILURE
    budgets = _events(markers, LIFECYCLE_EVENT_BUDGET)
    if len(budgets) != 1:
        return LIFECYCLE_UNRESOLVED
    budget = budgets[0]
    deadline_mono = _record_number(budget, "deadline_monotonic")
    starts = _events(markers, LIFECYCLE_EVENT_START)
    finishes = _events(markers, LIFECYCLE_EVENT_FINISH)
    alives = _events(markers, LIFECYCLE_EVENT_ALIVE)
    exited_notice = _events(markers, LIFECYCLE_EVENT_EXITED_AT_NOTICE)
    if len(starts) != 1 or len(finishes) > 1 or len(alives) > 1:
        return LIFECYCLE_UNRESOLVED
    start_mono = _record_number(starts[0], "monotonic")
    if start_mono is None:
        return LIFECYCLE_UNRESOLVED
    if deadline_mono is not None and start_mono > deadline_mono:
        return LIFECYCLE_STARTUP_FAILURE
    if alives:
        alive_mono = _record_number(alives[0], "monotonic")
        notice_mono = _notice_time(alives[0], budget)
        if (
            alive_mono is None
            or notice_mono is None
            or alive_mono < notice_mono
            or (deadline_mono is not None and alive_mono < deadline_mono)
            or not _host_start_receipt_ok(
                host_start_received_within_budget, start_mono, deadline_mono
            )
            or start_mono > notice_mono
            or not finishes
            or finishes[0].get("terminated") is not True
        ):
            return LIFECYCLE_UNRESOLVED
        return LIFECYCLE_CANDIDATE_RUNNING
    if exited_notice:
        # The child had already exited when the deadline notice arrived.  Its
        # completion did not reach the host before the deadline, so the cause
        # is not a registered outer-deadline failure.
        return LIFECYCLE_UNRESOLVED
    if not finishes:
        return LIFECYCLE_UNRESOLVED
    finish_mono = _record_number(finishes[0], "monotonic")
    if finish_mono is None or finish_mono < start_mono:
        return LIFECYCLE_UNRESOLVED
    if host_completed_within_budget is True:
        return LIFECYCLE_COMPLETED_BEFORE
    if host_completed_within_budget is False:
        return LIFECYCLE_UNRESOLVED
    if deadline_mono is not None:
        if finish_mono <= deadline_mono:
            return LIFECYCLE_COMPLETED_BEFORE
        return LIFECYCLE_UNRESOLVED
    # No host deadline result and no legacy deadline: only an untimed normal
    # completion is conclusive.
    if timed_out:
        return LIFECYCLE_UNRESOLVED
    return LIFECYCLE_COMPLETED_BEFORE


def _classify_legacy_lifecycle(
    markers: Sequence[Mapping[str, Any]],
    *,
    timed_out: bool,
    deadline_epoch: float,
) -> str:
    starts = _events(markers, LIFECYCLE_EVENT_START)
    finishes = _events(markers, LIFECYCLE_EVENT_FINISH)
    if len(starts) != 1 or len(finishes) > 1:
        return LIFECYCLE_UNRESOLVED
    try:
        start_epoch = float(starts[0]["epoch"])
        finish_epoch = float(finishes[0]["epoch"]) if finishes else None
    except (KeyError, TypeError, ValueError):
        return LIFECYCLE_UNRESOLVED
    if finish_epoch is not None and finish_epoch < start_epoch:
        return LIFECYCLE_UNRESOLVED
    if start_epoch > float(deadline_epoch):
        return LIFECYCLE_STARTUP_FAILURE
    if finish_epoch is not None:
        if finish_epoch <= float(deadline_epoch):
            return LIFECYCLE_COMPLETED_BEFORE
        return LIFECYCLE_CANDIDATE_EXITED_AFTER
    if timed_out:
        return LIFECYCLE_CANDIDATE_RUNNING
    return LIFECYCLE_UNRESOLVED


def _docker_container_state(
    container_name: str,
    *,
    runner: Any = subprocess.run,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read trusted container state from the Docker daemon."""

    try:
        completed = runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .State}}",
                str(container_name),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env=sanitized_env(),
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is unresolved
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout or "").strip() or "inspect failed"
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return None, f"non-JSON inspect output: {exc}"
    if not isinstance(state, dict):
        return None, "inspect output is not an object"
    return state, None


def _remove_container(
    container_name: str,
    *,
    runner: Any = subprocess.run,
) -> bool:
    """Force-remove a container and verify it is gone."""

    try:
        runner(
            ["docker", "rm", "-f", str(container_name)],
            capture_output=True,
            text=True,
            timeout=15,
            env=sanitized_env(),
        )
    except Exception:  # noqa: BLE001 - cleanup failure is recorded
        return False
    state, error = _docker_container_state(container_name, runner=runner)
    return state is None and bool(error) and "no such" in error.lower()


_LIFECYCLE_RUNNER_SOURCE = r'''import json
import os
import runpy
import sys
import time


def main():
    target = sys.argv[1]
    entry_fd = int(sys.argv[2])
    try:
        body = json.dumps(
            {
                "event": "post_exec_entry",
                "pid": os.getpid(),
                "monotonic": time.monotonic(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.write(entry_fd, body + b"\n")
    finally:
        # The write end exists only in this exec'd image and is closed before
        # any candidate code runs.  The per-attempt key was never visible to
        # this runner or to the candidate.
        try:
            os.close(entry_fd)
        except BaseException:
            pass
    sys.argv = [target]
    runpy.run_path(os.path.join("/work", target), run_name="__main__")


raise SystemExit(main())
'''

_LIFECYCLE_WRAPPER_SOURCE = r'''import hashlib
import hmac
import json
import os
import select
import signal
import sys
import time

EVIDENCE = "/evidence/lifecycle.jsonl"
BUDGET = "/evidence/budget.json"
DEADLINE_NOTICE = "/evidence/deadline_notice.txt"
LAUNCHER = "/tmp/gemmaloop_launcher.py"
RUNNER = "/work/post_exec_runner.py"
KEY_LEN = 64
EVENT_READY = "bootstrap_ready"
EVENT_BUDGET = "budget_applied"
EVENT_START = "candidate_start"
EVENT_FINISH = "candidate_finish"
EVENT_ALIVE = "candidate_alive_at_deadline_notice"
EVENT_EXITED_AT_NOTICE = "candidate_exited_at_deadline_notice"
EVENT_HANDOFF_FAILED = "handoff_failed"
EVENT_BOOTSTRAP_FAILURE = "bootstrap_failure"
NOTICE_PREFIX = "GML2"
NOTICE_EVENT = "deadline_reached"
WRAPPER_STARTED = time.monotonic()

LAUNCHER_SOURCE = r"""import json
import os
import sys
import time


def main():
    target = sys.argv[1]
    runner = sys.argv[2]
    setup_fd = int(sys.argv[3])
    # This pre-exec frame is diagnostic only.  It is emitted before the
    # candidate image is exec'd, so it is never execution-entry evidence.
    try:
        body = json.dumps(
            {
                "event": "exec_ok",
                "pid": os.getpid(),
                "monotonic": time.monotonic(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.write(setup_fd, body + b"\n")
    except BaseException:
        pass
    try:
        # Only a successful exec of the frozen runner lets that exec'd image
        # write the post-exec entry frame on this descriptor.  A death in this
        # interval closes the descriptor without any post-exec entry, which
        # must remain infrastructure/unknown.
        os.set_inheritable(setup_fd, True)
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-I",
                runner,
                os.path.join("/work", target),
                str(setup_fd),
            ],
        )
    except BaseException as exc:
        try:
            body = json.dumps(
                {
                    "event": "handoff_failed",
                    "pid": os.getpid(),
                    "monotonic": time.monotonic(),
                    "error": (type(exc).__name__ + ": " + str(exc))[:200],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(setup_fd, body + b"\n")
        except BaseException:
            pass
    raise SystemExit(124)


raise SystemExit(main())
"""


def read_exact(fd, size):
    chunks = b""
    while len(chunks) < size:
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError("short read on the key channel")
        chunks += chunk
    return chunks


def read_deadline_notice(key, attempt):
    """Verify the host's authenticated causal deadline-reached notice."""

    try:
        with open(DEADLINE_NOTICE, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
        mac, separator, framed = text.partition(" ")
        if not separator or not framed.startswith(NOTICE_PREFIX + " "):
            return None
        body = framed[len(NOTICE_PREFIX) + 1 :]
        expected = hmac.new(
            key.encode("ascii"),
            (NOTICE_PREFIX + " " + body).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return None
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("v") != 1
            or payload.get("attempt") != attempt
            or payload.get("event") != NOTICE_EVENT
        ):
            return None
        return payload
    except BaseException:
        return None


def main():
    target = sys.argv[1]
    attempt = sys.argv[2]
    evidence_fd = os.open(
        EVIDENCE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    os.set_inheritable(evidence_fd, False)
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_DUMPABLE = 4.  The candidate must not read the wrapper's
        # memory or descriptors through /proc even though both run as the
        # same container UID.
        libc.prctl(4, 0, 0, 0, 0)
    except BaseException:
        pass

    # The launcher is frozen verifier code with no key.  It runs after the
    # bootstrap exec and writes a post-exec confirmation before it execs the
    # candidate, so EOF on the close-on-exec setup pipe is positive evidence
    # that the bootstrap handoff succeeded.
    try:
        with open(LAUNCHER, "w", encoding="utf-8") as handle:
            handle.write(LAUNCHER_SOURCE)
        os.chmod(LAUNCHER, 0o500)
    except BaseException:
        pass

    # The child is forked *before* the parent reads the per-attempt key, so
    # the bootstrap that becomes the candidate never has the key in its
    # memory, globals, frames or descriptors.  It waits on a private
    # parent-to-child gate until verifier setup and the budget handshake have
    # completed.
    entry_read, entry_write = os.pipe()
    setup_read, setup_write = os.pipe()
    gate_read, gate_write = os.pipe()
    try:
        pid = os.fork()
    except BaseException:
        try:
            sys.stderr.write("bootstrap: fork failed\n")
        except BaseException:
            pass
        raise SystemExit(122)
    if pid == 0:
        code = 121
        try:
            # Verifier setup is complete before the execution-entry record.
            # The evidence descriptor is closed before candidate-controlled
            # execution.  The entry descriptor is close-on-exec; the setup
            # descriptor survives the bootstrap exec so the frozen launcher
            # can positively report that the exec happened.
            os.close(evidence_fd)
            os.close(entry_read)
            os.close(setup_read)
            os.close(gate_write)
            if os.read(gate_read, 1) != b"g":
                raise RuntimeError("host did not authorize execution entry")
            os.close(gate_read)
            try:
                os.setpgid(0, 0)
            except BaseException:
                pass
            entry = json.dumps(
                {
                    "event": "entry",
                    "pid": os.getpid(),
                    "monotonic": time.monotonic(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(entry_write, entry + b"\n")
            os.set_inheritable(entry_write, False)
            os.set_inheritable(setup_write, True)
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-I",
                    LAUNCHER,
                    target,
                    RUNNER,
                    str(setup_write),
                ],
            )
        except BaseException as exc:
            try:
                message = json.dumps(
                    {
                        "event": "handoff_failed",
                        "pid": os.getpid(),
                        "monotonic": time.monotonic(),
                        "error": (type(exc).__name__ + ": " + str(exc))[:200],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                os.write(entry_write, message + b"\n")
            except BaseException:
                pass
        finally:
            os._exit(code)
    os.close(entry_write)
    os.close(gate_read)
    os.close(setup_write)
    key = read_exact(0, KEY_LEN).decode("ascii")
    if len(key) != KEY_LEN:
        raise ValueError("bad key length")
    int(key, 16)
    seq = [0]

    def emit(event, **extra):
        seq[0] += 1
        body = {
            "v": 1,
            "attempt": attempt,
            "seq": seq[0],
            "event": event,
            "wall": time.time(),
            "monotonic": time.monotonic(),
            "pid": os.getpid(),
            **extra,
        }
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            key.encode("ascii"),
            ("GML1 " + text).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        data = (mac + " GML1 " + text + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(evidence_fd, view)
            view = view[written:]
        os.fsync(evidence_fd)

    def fail(reason):
        try:
            emit(EVENT_BOOTSTRAP_FAILURE, reason=reason)
        except BaseException:
            pass
        try:
            os.close(gate_write)
        except BaseException:
            pass
        raise SystemExit(122)

    emit(EVENT_READY)
    budget = None
    limit = time.monotonic() + 3.0
    while time.monotonic() < limit:
        try:
            with open(BUDGET, encoding="utf-8") as handle:
                budget = json.load(handle)
            break
        except BaseException:
            time.sleep(0.005)
    if not isinstance(budget, dict):
        fail("host did not provide a remaining budget")
    try:
        remaining = float(budget["remaining_s"])
    except BaseException:
        fail("host budget is invalid")
    if remaining <= 0:
        fail("outer budget was exhausted before execution entry")
    # No wrapper-side deadline exists.  The host owns the single registered
    # outer deadline and sends an authenticated causal notice when it is
    # reached; the wrapper only observes the identified child at that notice.
    emit(EVENT_BUDGET, remaining_s=remaining)
    os.write(gate_write, b"g")
    os.close(gate_write)

    entry_buffer = b""
    setup_buffer = b""
    entry_open = True
    setup_open = True
    entry_seen = False
    exec_confirmed = False
    post_exec_confirmed = False
    start_emitted = False
    handoff_reason = None
    handoff_emitted = False
    notice_verified = False
    notice_mono = None

    def handle_frame(frame):
        nonlocal entry_seen, exec_confirmed, post_exec_confirmed, handoff_reason
        if not isinstance(frame, dict):
            return
        event = frame.get("event")
        if event == "entry":
            entry_seen = True
        elif event == "exec_ok":
            # Diagnostic only: emitted before the candidate image is exec'd
            # and therefore never execution-entry evidence.
            exec_confirmed = True
        elif event == "post_exec_entry":
            # Written by the frozen runner after it is the exec'd image and
            # before it runs any candidate code.
            post_exec_confirmed = True
        elif event == "handoff_failed":
            handoff_reason = frame.get("error") or "exec handoff failed"

    def consume(fd, which):
        nonlocal entry_buffer, setup_buffer, entry_open, setup_open
        try:
            data = os.read(fd, 4096)
        except OSError:
            data = b""
        if data == b"":
            if which == "entry":
                entry_open = False
            else:
                setup_open = False
            return
        if which == "entry":
            entry_buffer += data
            buffer = entry_buffer
        else:
            setup_buffer += data
            buffer = setup_buffer
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            try:
                frame = json.loads(line.decode("utf-8"))
            except BaseException:
                frame = {"event": "bad_frame"}
            handle_frame(frame)
        if which == "entry":
            entry_buffer = buffer
        else:
            setup_buffer = buffer

    def read_frames(wait):
        fds = []
        if entry_open:
            fds.append(entry_read)
        if setup_open:
            fds.append(setup_read)
        if not fds:
            return
        try:
            ready, _, _ = select.select(fds, [], [], 0.01 if wait else 0)
        except BaseException:
            ready = []
        for fd in ready:
            consume(fd, "entry" if fd == entry_read else "setup")

    def maybe_start():
        nonlocal start_emitted
        if (
            not start_emitted
            and post_exec_confirmed
            and not setup_open
            and not entry_open
            and handoff_reason is None
        ):
            emit(EVENT_START, child_pid=pid, handoff="post_exec_entry")
            start_emitted = True

    finish_status = None
    killed = False
    exited_at_notice = False
    while True:
        read_frames(True)
        maybe_start()
        try:
            exited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            exited, status = pid, None
        if exited:
            finish_status = status
            break
        if not notice_verified:
            notice = read_deadline_notice(key, attempt)
            if notice is not None:
                notice_verified = True
                notice_mono = time.monotonic()
                probed, probe_status = os.waitpid(pid, os.WNOHANG)
                if probed == 0:
                    if start_emitted and handoff_reason is None:
                        # Positive trusted evidence: the identified child had
                        # not exited when the host's registered outer deadline
                        # notice arrived, immediately before enforcement.
                        emit(
                            EVENT_ALIVE,
                            child_pid=pid,
                            notice_monotonic=notice_mono,
                        )
                        killed = True
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except BaseException:
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except BaseException:
                                pass
                        try:
                            _probed, finish_status = os.waitpid(pid, 0)
                        except BaseException:
                            finish_status = None
                    else:
                        if handoff_reason is None:
                            handoff_reason = (
                                "execution entry was not established before "
                                "the deadline notice"
                            )
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except BaseException:
                            pass
                        try:
                            _probed, finish_status = os.waitpid(pid, 0)
                        except BaseException:
                            finish_status = None
                    break
                finish_status = probe_status
                emit(
                    EVENT_EXITED_AT_NOTICE,
                    child_pid=pid,
                    notice_monotonic=notice_mono,
                )
                exited_at_notice = True
                break
        time.sleep(0.002)
    for _ in range(50):
        read_frames(True)
        maybe_start()
        if not entry_open and not setup_open:
            break
    if not start_emitted and handoff_reason is None:
        handoff_reason = "successful post-exec execution entry was not observed"
    returncode = None
    signal_number = None
    exit_observed = False
    if finish_status is not None:
        exit_observed = True
        if os.WIFSIGNALED(finish_status):
            signal_number = os.WTERMSIG(finish_status)
            returncode = -signal_number
        else:
            returncode = os.WEXITSTATUS(finish_status)
    if handoff_reason is not None or not start_emitted or not exit_observed:
        reason = handoff_reason or (
            "successful post-exec execution entry was not observed"
            if not start_emitted
            else "candidate exit status was not observed"
        )
        if not handoff_emitted:
            emit(EVENT_HANDOFF_FAILED, child_pid=pid, error=reason)
        code = 121
    else:
        emit(
            EVENT_FINISH,
            child_pid=pid,
            returncode=returncode,
            signal_number=signal_number,
            wait_status=finish_status,
            terminated=True,
            exit_observed=True,
            killed=killed,
            exited_at_deadline_notice=exited_at_notice,
        )
        code = returncode if returncode is not None else 121
    try:
        os.close(entry_read)
    except BaseException:
        pass
    try:
        os.close(setup_read)
    except BaseException:
        pass
    try:
        os.close(evidence_fd)
    except BaseException:
        pass
    if code < 0 or code > 255:
        code = 121
    return code


raise SystemExit(main())
'''

def _write_lifecycle_runner(workdir: Path) -> Path:
    path = workdir / LIFECYCLE_RUNNER_NAME
    path.write_text(_LIFECYCLE_RUNNER_SOURCE, encoding="utf-8")
    return path


def _write_lifecycle_wrapper(workdir: Path) -> Path:
    path = workdir / LIFECYCLE_WRAPPER_NAME
    path.write_text(_LIFECYCLE_WRAPPER_SOURCE, encoding="utf-8")
    _write_lifecycle_runner(workdir)
    return path


def authenticate_lifecycle_frames(
    frames: Sequence[str] | None,
    key: str | None,
    attempt: str | None = None,
    *,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Authenticate exact retained ``<hmac> GML1 <json>`` lifecycle lines.

    The same function is used when the evidence is first read and when the
    count loader independently re-verifies the raw frames persisted with an
    attempt.  Each frame is HMAC-SHA-256 authenticated with the per-attempt
    key, bound to the attempt and child order, and duplicate/replayed or
    out-of-order sequences fail closed.
    """

    if key is None:
        return [], "no lifecycle authentication key"
    if not frames:
        return [], "no lifecycle evidence frames"
    lines = [str(line) for line in frames]
    if lines and not lines[-1].endswith("\n"):
        if not allow_partial:
            return [], "lifecycle evidence has a torn trailing record"
        lines = lines[:-1]
    records: list[dict[str, Any]] = []
    last_seq = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        mac, separator, framed = line.partition(" ")
        if not separator or not framed.startswith(LIFECYCLE_FRAME_PREFIX + " "):
            return [], f"lifecycle marker HMAC invalid at line {line_number}"
        body = framed[len(LIFECYCLE_FRAME_PREFIX) + 1 :]
        expected = hmac.new(
            str(key).encode("utf-8"),
            (LIFECYCLE_FRAME_PREFIX + " " + body).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return [], f"lifecycle marker HMAC invalid at line {line_number}"
        try:
            record = json.loads(body)
        except json.JSONDecodeError:
            return [], f"lifecycle marker JSON invalid at line {line_number}"
        if not isinstance(record, dict):
            return [], f"lifecycle marker is not an object at line {line_number}"
        if record.get("v") != 1:
            return [], f"unsupported lifecycle record version at line {line_number}"
        if attempt is not None and record.get("attempt") != attempt:
            return [], f"lifecycle record is bound to a different attempt at line {line_number}"
        try:
            seq = int(record["seq"])
        except (KeyError, TypeError, ValueError):
            return [], f"lifecycle record has no sequence at line {line_number}"
        if seq <= last_seq:
            return [], f"duplicate or replayed lifecycle record at line {line_number}"
        last_seq = seq
        if not isinstance(record.get("event"), str):
            return [], f"lifecycle record has no event at line {line_number}"
        if record.get("event") == LIFECYCLE_EVENT_START and "child_pid" not in record:
            return [], f"execution-entry record has no child identity at line {line_number}"
        records.append(record)
    return records, None


def read_lifecycle_frames(evidence_dir: Path | str) -> list[str] | None:
    """Return the exact raw lifecycle lines retained for independent auditing."""

    path = Path(evidence_dir) / LIFECYCLE_EVIDENCE_FILE
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return None


def read_lifecycle_markers(
    evidence_dir: Path | str,
    key: str | None,
    attempt: str | None = None,
    *,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Read and authenticate the wrapper's HMAC-SHA-256 lifecycle records."""

    if key is None:
        return [], "no lifecycle authentication key"
    path = Path(evidence_dir) / LIFECYCLE_EVIDENCE_FILE
    if not path.is_file():
        return [], "no lifecycle evidence file"
    try:
        frames = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        return [], f"lifecycle evidence unreadable: {type(exc).__name__}: {exc}"
    return authenticate_lifecycle_frames(
        frames, key, attempt, allow_partial=allow_partial
    )


def _read_partial_records(
    evidence_dir: Path, key: str, attempt: str
) -> list[dict[str, Any]]:
    records, _error = read_lifecycle_markers(
        evidence_dir, key, attempt, allow_partial=True
    )
    return records


def _call_marker_reader(
    reader: Any,
    evidence_dir: Path,
    key: str,
    attempt: str,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        parameters = inspect.signature(reader).parameters
    except (TypeError, ValueError):
        return reader(evidence_dir, key)
    if "attempt" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return reader(evidence_dir, key, attempt=attempt)
    return reader(evidence_dir, key)


def _write_deadline_notice(
    evidence_dir: Path,
    *,
    key: str,
    attempt: str,
    host_deadline_epoch: float,
) -> bool:
    """Write the host's authenticated causal deadline-reached notice."""

    body = json.dumps(
        {
            "v": 1,
            "attempt": attempt,
            "event": "deadline_reached",
            "host_epoch": time.time(),
            "host_deadline_epoch": float(host_deadline_epoch),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    mac = hmac.new(
        str(key).encode("utf-8"),
        (LIFECYCLE_DEADLINE_FRAME_PREFIX + " " + body).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    text = f"{mac} {LIFECYCLE_DEADLINE_FRAME_PREFIX} {body}\n"
    target = Path(evidence_dir) / LIFECYCLE_DEADLINE_FILE
    temporary = target.with_suffix(".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        return False
    return True


def _run_process_docker(
    command: Sequence[str],
    *,
    input_text: str,
    timeout: float,
    container_name: str,
    evidence_dir: Path | str | None = None,
    key: str | None = None,
    attempt: str | None = None,
    state_probe: Any = _docker_container_state,
    cleanup: Any = _remove_container,
    marker_reader: Any = read_lifecycle_markers,
    enforcement_spawner: Any = None,
) -> tuple[int, str, str, bool, float, dict[str, Any]]:
    """Run one ``docker run`` under the unchanged outer timeout.

    The host owns the single registered outer deadline over the whole ``docker
    run`` (including startup).  It carries only the *remaining duration* to the
    frozen wrapper and, when the registered deadline is reached, writes an
    authenticated ``deadline_reached`` notice over the private evidence
    channel *before* enforcing the kill.  The wrapper then makes a non-blocking
    observation of the identified child and records either
    ``candidate_alive_at_deadline_notice`` or
    ``candidate_exited_at_deadline_notice``.  Only the former, causally after
    the notice, plus established handoff and verified termination/cleanup is a
    registered outer-deadline failure.  No result obtained after the deadline
    is accepted, and cleanup does not extend the acceptance budget.
    """

    started_wall = time.time()
    started = time.perf_counter()
    deadline_epoch = started_wall + float(timeout)
    host_deadline = started + float(timeout)
    events: list[dict[str, Any]] = []
    budget_sent: dict[str, Any] = {}
    notice_sent: dict[str, Any] = {}
    start_received: dict[str, Any] = {}
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.005):
            if evidence_dir is not None and key and attempt and (
                not budget_sent or not start_received
            ):
                try:
                    partial = _read_partial_records(
                        Path(evidence_dir), key, attempt
                    )
                except Exception:  # noqa: BLE001 - partial reads are best effort
                    partial = []
                if not budget_sent and any(
                    record.get("event") == LIFECYCLE_EVENT_READY
                    for record in partial
                ):
                    remaining = deadline_epoch - time.time()
                    payload = {
                        "remaining_s": max(0.0, remaining),
                        "host_sent_epoch": time.time(),
                        "host_deadline_epoch": deadline_epoch,
                    }
                    budget_sent.update(payload)
                    target = Path(evidence_dir) / LIFECYCLE_BUDGET_FILE
                    temporary = target.with_suffix(".tmp")
                    try:
                        temporary.write_text(
                            json.dumps(payload), encoding="utf-8"
                        )
                        os.replace(temporary, target)
                    except OSError:
                        budget_sent.clear()
                if not start_received and any(
                    record.get("event") == LIFECYCLE_EVENT_START
                    for record in partial
                ):
                    # Stamp the host receipt on the same monotonic clock as
                    # the registered deadline.  This is what makes a
                    # late-entry timeout attribution impossible.
                    receipt = time.perf_counter()
                    start_received.update(
                        {
                            "monotonic": receipt,
                            "offset_s": receipt - started,
                            "within_budget": receipt < host_deadline,
                        }
                    )
            state, error = state_probe(container_name)
            observed = time.time()
            events.append(
                {
                    "observed_epoch": observed,
                    "status": (state or {}).get("Status") if state else None,
                    "started_at": (state or {}).get("StartedAt") if state else None,
                    "finished_at": (state or {}).get("FinishedAt") if state else None,
                    "error": error,
                }
            )
            if state is not None and state.get("Status") in {"exited", "dead"}:
                break

    def deadline_notifier() -> None:
        while True:
            delay = host_deadline - time.perf_counter()
            if delay <= 0:
                break
            if stop.wait(delay):
                return
        if stop.is_set():
            return
        if evidence_dir is None or not key or not attempt:
            return
        sent = _write_deadline_notice(
            Path(evidence_dir),
            key=str(key),
            attempt=str(attempt),
            host_deadline_epoch=deadline_epoch,
        )
        notice_sent.update(
            {
                "sent": bool(sent),
                "host_epoch": time.time(),
                "host_deadline_epoch": deadline_epoch,
            }
        )

    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=sanitized_env(),
        start_new_session=(os.name != "nt"),
    )
    monitor_thread = threading.Thread(target=monitor, name="docker-lifecycle", daemon=True)
    notice_thread = threading.Thread(
        target=deadline_notifier, name="deadline-notice", daemon=True
    )
    monitor_thread.start()
    notice_thread.start()
    timed_out = False
    final_state: dict[str, Any] | None = None
    probe_error: str | None = None
    enforcement_issued = True
    enforcement_errors: list[str] = []
    enforcement_overrun = False
    enforcement_evidence: dict[str, Any] = {
        "triggered": False,
        "bound_s": float(LIFECYCLE_NOTICE_GRACE_S),
        "deadline_monotonic": float(host_deadline),
        "bound_monotonic": float(host_deadline)
        + float(LIFECYCLE_NOTICE_GRACE_S),
        "overrun": False,
        "overrun_reason": None,
        "errors": [],
    }
    # The single registered outer deadline is never restarted.  The wrapper may
    # use only the frozen observation sub-window after it; enforcement begins
    # at ``deadline + LIFECYCLE_NOTICE_OBSERVATION_S`` and every primitive is
    # issued inside the frozen 250 ms window.  The timeout passed to
    # ``communicate`` is the *remaining* time until that absolute instant,
    # never a fresh timeout + grace from a later clock.
    enforcement_deadline = (
        started + float(timeout) + float(DEADLINE_ENFORCEMENT_START_S)
    )
    finished_monotonic: float | None = None
    try:
        remaining_to_enforcement = enforcement_deadline - time.perf_counter()
        stdout, stderr = process.communicate(
            (str(key) if key is not None else "") + input_text,
            timeout=max(0.0, remaining_to_enforcement),
        )
        finished_monotonic = time.perf_counter()
        code = process.returncode
        final_state, probe_error = state_probe(container_name)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Registered lifecycle amendment paragraph 3 enforcement order: the
        # observation sub-window has expired, so the host FIRST hard-kills the
        # docker client and launches both non-blocking container-kill
        # primitives concurrently.  No state probe, taskkill wait or cleanup
        # step may run before enforcement has been issued.
        (
            _enforcement_clients,
            enforcement_issued,
            enforcement_errors,
            enforcement_evidence,
        ) = _enforce_deadline_termination(
            process,
            container_name,
            spawner=enforcement_spawner,
            deadline_monotonic=host_deadline,
        )
        enforcement_overrun = bool(enforcement_evidence.get("overrun"))
        if enforcement_issued and not enforcement_overrun:
            # Only after enforcement may the host observe daemon state for
            # attribution and cleanup.
            final_state, probe_error = state_probe(container_name)
        else:
            # Enforcement was not issued inside the frozen bound (or no
            # primitive could be issued at all).  Skip the probe and keep the
            # attempt infrastructure/unknown with cleanup pending so the
            # documented recovery path must verify the container is gone.
            final_state = None
            probe_error = (
                str(enforcement_evidence.get("overrun_reason") or "")
                or (
                    "deadline enforcement could not be issued: "
                    + "; ".join(
                        enforcement_errors or ["unknown launch failure"]
                    )
                )
            )
        stdout, stderr = process.communicate()
        finished_monotonic = time.perf_counter()
        code = process.returncode
    finally:
        stop.set()
        monitor_thread.join(timeout=2.0)
        notice_thread.join(timeout=2.0)

    if finished_monotonic is None:
        finished_monotonic = time.perf_counter()
    host_completed_within_budget = (
        finished_monotonic - started
    ) <= float(timeout)
    late_completion = not host_completed_within_budget
    if late_completion:
        # Post-deadline bytes and exit status may establish attribution and
        # cleanup only; they can never make a candidate answer acceptable.
        stdout = ""
        stderr = ""
        code = None
    cleanup_verified = bool(cleanup(container_name))
    if not enforcement_issued or enforcement_overrun:
        # Never a conclusive outcome when container enforcement was not
        # issued inside the frozen bound: leave cleanup pending for the
        # recovery path to verify the container is gone.
        cleanup_verified = False
    status = (final_state or {}).get("Status")
    raw_frames: list[str] | None = None
    if evidence_dir is not None and key is not None:
        markers, marker_error = _call_marker_reader(
            marker_reader, Path(evidence_dir), key, str(attempt or "")
        )
        raw_frames = read_lifecycle_frames(Path(evidence_dir))
    else:
        markers, marker_error = [], "no trusted lifecycle evidence channel"
    lifecycle_class = classify_lifecycle_evidence(
        markers=markers,
        marker_error=marker_error,
        timed_out=timed_out,
        cleanup_verified=cleanup_verified,
        deadline_epoch=deadline_epoch,
        status=status,
        started_at=(final_state or {}).get("StartedAt"),
        finished_at=(final_state or {}).get("FinishedAt"),
        events=events,
        probe_error=probe_error,
        host_completed_within_budget=host_completed_within_budget,
        host_start_received_within_budget=start_received.get("within_budget"),
    )
    if enforcement_overrun:
        # A late issuance can never become a conclusive candidate deadline
        # failure.  Fail closed as infrastructure/unknown and keep cleanup
        # pending until the recovery path verifies the container is gone.
        lifecycle_class = LIFECYCLE_UNRESOLVED
        cleanup_verified = False
    deadline_class = lifecycle_class in CANDIDATE_DEADLINE_CLASSES
    if deadline_class:
        # A wrapper-certified candidate still running at the outer deadline is
        # a timed-out attempt even when the docker run process exited before the
        # host enforcement window opened.
        timed_out = True
    record_error = lifecycle_record_error(
        markers=markers,
        marker_error=marker_error,
        cleanup_verified=cleanup_verified,
        timed_out=timed_out or deadline_class,
        host_completed_within_budget=host_completed_within_budget,
        host_start_received_within_budget=start_received.get("within_budget"),
    )
    finishes = _events(markers, LIFECYCLE_EVENT_FINISH)
    alives = _events(markers, LIFECYCLE_EVENT_ALIVE)
    starts = _events(markers, LIFECYCLE_EVENT_START)
    handoff_success = bool(starts) and marker_error is None
    termination_verified = bool(
        finishes and finishes[0].get("terminated") is True
    )
    deadline_alive = bool(alives)
    exit_observed = bool(finishes and finishes[0].get("exit_observed") is True)
    child_exit_code: int | None = None
    if finishes and exit_observed and not late_completion:
        try:
            child_exit_code = int(finishes[0]["returncode"])
        except (KeyError, TypeError, ValueError):
            child_exit_code = None
    # Keep the evidence bounded: the first and last few daemon observations.
    trimmed_events = (events[:4] + events[-6:]) if len(events) > 10 else events
    lifecycle_valid = bool(
        record_error is None and lifecycle_class in TRUSTED_LIFECYCLE_CLASSES
    )
    # Exactly two routes can establish a conclusive outer-deadline failure:
    # route A when the wrapper enforced its own kill before the host window
    # opened, and route B when the host had to enforce.  Anything else remains
    # infrastructure/unknown with cleanup pending.
    host_enforcement_triggered = bool(enforcement_evidence.get("triggered"))
    enforcement_route: str | None = None
    if timed_out:
        if host_enforcement_triggered:
            if not enforcement_overrun:
                enforcement_route = ENFORCEMENT_ROUTE_HOST
        elif deadline_class and wrapper_enforcement_route_verified(
            lifecycle_valid=lifecycle_valid,
            start_received_within_budget=start_received.get("within_budget"),
            deadline_alive=deadline_alive,
            termination_verified=termination_verified,
            cleanup_verified=cleanup_verified,
            host_exit_monotonic=finished_monotonic,
            host_deadline_monotonic=host_deadline,
        ):
            enforcement_route = ENFORCEMENT_ROUTE_WRAPPER
    if timed_out and enforcement_route is None:
        lifecycle_class = LIFECYCLE_UNRESOLVED
        deadline_class = False
        cleanup_verified = False
        lifecycle_valid = False
        if not record_error:
            record_error = (
                "no conclusive deadline enforcement route: route A needs the "
                "verified wrapper kill and host exit inside the frozen window; "
                "route B needs the complete host enforcement action set"
            )
    host_exit_bound = float(host_deadline) + float(LIFECYCLE_NOTICE_GRACE_S)
    host_exit_within_bound = bool(
        finished_monotonic is not None
        and float(finished_monotonic) <= host_exit_bound + 1e-6
    )
    enforcement_evidence.update(
        {
            "enforcement_route": enforcement_route,
            "host_exit_monotonic": finished_monotonic,
            "host_exit_within_bound": host_exit_within_bound,
            "wrapper_kill_verified": bool(
                enforcement_route == ENFORCEMENT_ROUTE_WRAPPER
                and termination_verified
                and exit_observed
            ),
        }
    )
    return (
        child_exit_code if child_exit_code is not None else code,
        stdout or "",
        stderr or "",
        timed_out,
        time.perf_counter() - started,
        {
            "class": lifecycle_class,
            "container_name": container_name,
            "container_started_at": (final_state or {}).get("StartedAt"),
            "container_finished_at": (final_state or {}).get("FinishedAt"),
            "container_status": status,
            "events": trimmed_events,
            "markers": list(markers),
            "lifecycle_records": list(markers),
            "lifecycle_frames_raw": list(raw_frames or []),
            "lifecycle_auth_key": key,
            "marker_error": marker_error,
            "record_error": record_error,
            "lifecycle_valid": lifecycle_valid,
            "handoff_success": handoff_success,
            "termination_verified": termination_verified,
            "deadline_alive": deadline_alive,
            "exit_observed": exit_observed,
            "lifecycle_attempt": attempt,
            "cleanup_verified": cleanup_verified,
            "probe_error": probe_error,
            "deadline_enforcement_issued": enforcement_issued,
            "deadline_enforcement_errors": list(enforcement_errors),
            "deadline_enforcement_overrun": enforcement_overrun,
            "deadline_enforcement": dict(enforcement_evidence),
            "enforcement_route": enforcement_route,
            "host_exit_monotonic": finished_monotonic,
            "deadline_epoch": deadline_epoch,
            "completed_within_budget": host_completed_within_budget,
            "budget_sent": dict(budget_sent),
            "deadline_notice": dict(notice_sent),
            "late_completion": not host_completed_within_budget,
            "start_received_monotonic": start_received.get("monotonic"),
            "start_received_offset_s": start_received.get("offset_s"),
            "start_received_within_budget": start_received.get("within_budget"),
            "host_deadline_monotonic": host_deadline,
        },
    )


def _write_candidate(workdir: Path, code: str) -> Path:
    path = workdir / "candidate.py"
    path.write_text(code or "", encoding="utf-8")
    return path


def _functional_runner(workdir: Path, fn_name: str) -> Path:
    runner = workdir / "runner.py"
    runner.write_text(
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('candidate', 'candidate.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "args = json.loads(sys.stdin.read())\n"
        f"fn = getattr(module, {fn_name!r})\n"
        "result = fn(*args) if isinstance(args, list) else fn(**args)\n"
        "print(json.dumps(result, sort_keys=True, default=str))\n",
        encoding="utf-8",
    )
    return runner


def _compare_output(actual: str, expected: str, test_type: str) -> bool:
    if test_type == "functional":
        try:
            return json.loads(actual) == json.loads(expected)
        except (json.JSONDecodeError, TypeError):
            return normalize_output(actual) == normalize_output(expected)
    return normalize_output(actual) == normalize_output(expected)


def _test_local(
    code: str,
    test: dict[str, Any],
    index: int,
    *,
    timeout: float,
    workdir: Path,
    python: str,
) -> TestResult:
    test_type = str(test.get("type") or "stdin")
    test_input = str(test.get("input") or "")
    expected = str(test.get("output") or "")
    candidate = _write_candidate(workdir, code)
    if test_type == "functional":
        fn_name = str(test.get("fn_name") or "")
        if not fn_name:
            return TestResult(index, False, test_input, expected, error="missing fn_name")
        command = [python, "-I", str(_functional_runner(workdir, fn_name))]
    else:
        command = [python, "-I", str(candidate)]
    code_rc, stdout, stderr, timed_out, duration = _run_process(
        command, input_text=test_input, timeout=timeout, cwd=workdir
    )
    passed = (not timed_out) and code_rc == 0 and _compare_output(stdout, expected, test_type)
    error = None
    if timed_out:
        error = f"timeout after {timeout:g}s"
    elif code_rc != 0:
        error = f"exit code {code_rc}"
    elif not passed:
        error = "wrong answer"
    return TestResult(
        index=index,
        passed=passed,
        input=test_input,
        expected=expected,
        stdout=stdout,
        stderr=stderr,
        error=error,
        timed_out=timed_out,
        duration_s=duration,
        exit_code=code_rc,
        lifecycle_class=(
            LIFECYCLE_LOCAL_TIMEOUT if timed_out else LIFECYCLE_LOCAL
        ),
        lifecycle_valid=True,
        cleanup_verified=True,
        completed_within_budget=not timed_out,
        test_type=test_type,
    )


def run_local_tests(
    code: str,
    tests: Iterable[dict[str, Any]],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    python: str | None = None,
) -> GradeResult:
    """Run every test in a separate local subprocess (non-sandboxed fallback)."""

    tests = list(tests)
    python = python or sys.executable
    results: list[TestResult] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gemmaloop_grade_") as tmp:
        workdir = Path(tmp)
        for index, test in enumerate(tests):
            results.append(
                _test_local(
                    code,
                    test,
                    index,
                    timeout=float(timeout),
                    workdir=workdir,
                    python=python,
                )
            )
    passed_tests = sum(1 for result in results if result.passed)
    passed = bool(results) and passed_tests == len(results)
    return GradeResult(
        passed=passed,
        n_tests=len(results),
        passed_tests=passed_tests,
        sandboxed=False,
        tests=results,
        error=None if results else "no tests",
        timed_out=any(result.timed_out for result in results),
        duration_s=time.perf_counter() - started,
        sandbox="local",
    )


class DockerImageDigestError(RuntimeError):
    """The running image content digest differs from the frozen digest."""


def inspect_docker_image(
    image: str,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Return the first ``docker image inspect`` record for ``image``."""

    completed = runner(
        ["docker", "image", "inspect", str(image)],
        capture_output=True,
        text=True,
        timeout=30,
        env=sanitized_env(),
    )
    if completed.returncode != 0:
        raise DockerImageDigestError(
            f"docker image inspect {image!r} failed: {completed.stderr!r}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DockerImageDigestError(
            f"docker image inspect {image!r} returned non-JSON"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise DockerImageDigestError(
            f"docker image inspect {image!r} returned no image"
        )
    return payload[0]


class DockerSandbox:
    """CPU-only Docker sandbox command builder and runner.

    The container has no network, a read-only root filesystem, a writable
    ``tmpfs`` at ``/tmp``, memory/PID/CPU limits and runs as an unprivileged
    user.  The candidate directory is mounted read-only at ``/work``.

    ``expected_digest`` pins the *content*: when set, ``ensure_image`` refuses
    an image whose ``Id``/``RepoDigests`` do not carry that digest.  A mutable
    tag alone is never accepted as the pinned verifier.
    """

    def __init__(
        self,
        image: str = DEFAULT_DOCKER_IMAGE,
        *,
        memory_mb: int = 512,
        pids: int = 64,
        cpus: float = 1.0,
        user: str = "65534:65534",
        expected_digest: str | None = None,
    ) -> None:
        self.image = image
        self.memory_mb = int(memory_mb)
        self.pids = int(pids)
        self.cpus = float(cpus)
        self.user = user
        if expected_digest is not None and "sha256:" not in str(expected_digest):
            raise ValueError(
                "expected_digest must be a content digest (sha256:...), not a tag"
            )
        self.expected_digest = expected_digest

    def available(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            completed = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        return completed.returncode == 0

    def verify_image_digest(self) -> dict[str, Any]:
        """Refuse unless the local image matches the frozen content digest."""

        if self.expected_digest is None:
            return {}
        entry = inspect_docker_image(self.image)
        image_id = str(entry.get("Id") or "")
        digests = [image_id]
        for value in entry.get("RepoDigests") or []:
            _repository, separator, content = str(value).partition("@")
            if separator and content:
                digests.append(content)
        expected = str(self.expected_digest)
        expected_content = expected.partition("@")[2] if "@" in expected else expected
        if expected_content not in digests:
            raise DockerImageDigestError(
                f"docker image {self.image!r} resolves to {digests!r}, not the "
                f"frozen digest {expected!r}"
            )
        return {
            "tag": self.image,
            "id": image_id,
            "digest": expected_content,
            "repo_digests": list(entry.get("RepoDigests") or []),
        }

    def ensure_image(self) -> None:
        """Pre-pull the image so the first test does not pay the pull time."""

        try:
            found = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=30,
                env=sanitized_env(),
            )
            if found.returncode != 0:
                subprocess.run(
                    ["docker", "pull", self.image],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=sanitized_env(),
                    check=True,
                )
            self.verify_image_digest()
        except DockerImageDigestError:
            raise
        except Exception:
            # A genuine Docker failure is surfaced by the first test itself,
            # but an explicit digest pin may never be silently skipped.
            if self.expected_digest is not None:
                raise
            return

    def command(
        self,
        workdir: Path,
        script: str = "candidate.py",
        *,
        container_name: str | None = None,
        evidence_dir: Path | str | None = None,
        attempt: str | None = None,
    ) -> list[str]:
        workdir = Path(workdir).resolve()
        command = ["docker", "run", "--rm"]
        if container_name is not None:
            command += ["--name", str(container_name)]
        command += [
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--memory",
            f"{self.memory_mb}m",
            "--memory-swap",
            f"{self.memory_mb}m",
            "--pids-limit",
            str(self.pids),
            "--cpus",
            str(self.cpus),
            "--user",
            self.user,
            "--security-opt",
            "no-new-privileges",
            "-i",
            "-v",
            f"{workdir}:/work:ro",
            *(
                ["-v", f"{Path(evidence_dir).resolve()}:/evidence:rw"]
                if evidence_dir is not None
                else []
            ),
            "-w",
            "/work",
            self.image,
            "python",
            "-I",
            f"/work/{LIFECYCLE_WRAPPER_NAME}",
            script,
            str(attempt or ""),
        ]
        return command

    def run_tests(
        self,
        code: str,
        tests: Iterable[dict[str, Any]],
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        container_names: Sequence[str] | None = None,
        attempts: Sequence[str] | None = None,
    ) -> GradeResult:
        tests = list(tests)
        results: list[TestResult] = []
        self.ensure_image()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="gemmaloop_docker_") as tmp:
            workdir = Path(tmp)
            _write_candidate(workdir, code)
            _write_lifecycle_wrapper(workdir)
            for index, test in enumerate(tests):
                test_type = str(test.get("type") or "stdin")
                expected = str(test.get("output") or "")
                test_input = str(test.get("input") or "")
                script = "candidate.py"
                if test_type == "functional":
                    fn_name = str(test.get("fn_name") or "")
                    if not fn_name:
                        results.append(
                            TestResult(index, False, test_input, expected, error="missing fn_name")
                        )
                        continue
                    _functional_runner(workdir, fn_name)
                    script = "runner.py"
                # The registered per-test timeout supervises the whole docker
                # run process, including startup.  The clock is never restarted
                # and no result after the deadline is accepted.
                container_name = (
                    str(container_names[index])
                    if container_names is not None
                    and index < len(container_names)
                    else f"gemmaloop-{uuid.uuid4().hex[:16]}"
                )
                key = secrets.token_hex(LIFECYCLE_KEY_BYTES // 2)
                attempt = (
                    str(attempts[index])
                    if attempts is not None and index < len(attempts)
                    else uuid.uuid4().hex
                )
                with tempfile.TemporaryDirectory(
                    prefix="gemmaloop_evidence_"
                ) as evidence_tmp:
                    evidence_dir = Path(evidence_tmp)
                    try:
                        os.chmod(evidence_dir, 0o777)
                    except OSError:  # pragma: no cover - platform specific
                        pass
                    (
                        code_rc,
                        stdout,
                        stderr,
                        timed_out,
                        duration,
                        lifecycle,
                    ) = _run_process_docker(
                        self.command(
                            workdir,
                            script,
                            container_name=container_name,
                            evidence_dir=evidence_dir,
                            attempt=attempt,
                        ),
                        input_text=test_input,
                        timeout=float(timeout),
                        container_name=container_name,
                        evidence_dir=evidence_dir,
                        key=key,
                        attempt=attempt,
                    )
                lifecycle_class = lifecycle.get("class")
                lifecycle_valid = lifecycle.get("lifecycle_valid") is True
                deadline_class = lifecycle_class in CANDIDATE_DEADLINE_CLASSES
                if deadline_class:
                    # A wrapper-certified candidate still running at the outer
                    # deadline is a deadline failure.  A delayed finish record
                    # without a liveness observation is already unresolved and
                    # never reaches this branch.
                    timed_out = True
                enforcement_route = lifecycle.get("enforcement_route")
                handoff_success = lifecycle.get("handoff_success") is True
                termination_verified = (
                    lifecycle.get("termination_verified") is True
                )
                deadline_alive = lifecycle.get("deadline_alive") is True
                exit_observed = lifecycle.get("exit_observed") is True
                if timed_out and deadline_class:
                    # Timeout: established entry before the deadline, positive
                    # non-exit observation at/after it, verified eventual
                    # termination and exactly one accepted enforcement route
                    # (wrapper route A or host route B).  waitpid-after-deadline
                    # alone is not enough.
                    trusted_lifecycle = bool(
                        lifecycle_valid
                        and handoff_success
                        and deadline_alive
                        and termination_verified
                        and enforcement_route in DEADLINE_ENFORCEMENT_ROUTES
                    )
                elif timed_out:
                    # A timeout/transport/teardown failure after an established
                    # completion is infrastructure, never a candidate timeout.
                    trusted_lifecycle = False
                else:
                    # Normal result: successful handoff, genuine child exit,
                    # completed lifecycle before the wrapper deadline and a
                    # verified exit observation.
                    trusted_lifecycle = bool(
                        lifecycle_valid
                        and lifecycle_class == LIFECYCLE_COMPLETED_BEFORE
                        and handoff_success
                        and termination_verified
                        and exit_observed
                        and code_rc is not None
                    )
                passed = bool(
                    trusted_lifecycle
                    and not timed_out
                    and code_rc == 0
                    and _compare_output(stdout, expected, test_type)
                )
                error = None
                if not trusted_lifecycle:
                    # Every Docker outcome -- pass, wrong answer, candidate
                    # error and timeout -- requires the state-specific complete
                    # lifecycle record.  Otherwise it is infrastructure
                    # unknown, never a conclusive result.
                    detail = (
                        lifecycle.get("record_error")
                        or lifecycle.get("marker_error")
                        or lifecycle_class
                        or "missing lifecycle record"
                    )
                    error = f"untrusted lifecycle evidence: {detail}"
                elif timed_out:
                    error = f"timeout after {timeout:g}s"
                elif code_rc != 0:
                    error = f"docker exit code {code_rc}"
                elif not passed:
                    error = "wrong answer"
                results.append(
                    TestResult(
                        index=index,
                        passed=passed,
                        input=test_input,
                        expected=expected,
                        stdout=stdout,
                        stderr=stderr,
                        error=error,
                        timed_out=timed_out,
                        duration_s=duration,
                        exit_code=code_rc,
                        lifecycle_class=lifecycle.get("class"),
                        container_name=lifecycle.get("container_name"),
                        container_started_at=lifecycle.get("container_started_at"),
                        container_finished_at=lifecycle.get("container_finished_at"),
                        lifecycle_events=list(lifecycle.get("events") or []),
                        lifecycle_markers=list(lifecycle.get("markers") or []),
                        lifecycle_records=list(lifecycle.get("lifecycle_records") or []),
                        lifecycle_frames_raw=list(
                            lifecycle.get("lifecycle_frames_raw") or []
                        ),
                        lifecycle_auth_key=lifecycle.get("lifecycle_auth_key"),
                        marker_error=lifecycle.get("marker_error"),
                        completed_within_budget=lifecycle.get(
                            "completed_within_budget"
                        ),
                        test_type=test_type,
                        lifecycle_valid=lifecycle.get("lifecycle_valid"),
                        cleanup_verified=lifecycle.get("cleanup_verified"),
                        handoff_success=lifecycle.get("handoff_success"),
                        termination_verified=lifecycle.get("termination_verified"),
                        deadline_alive=lifecycle.get("deadline_alive"),
                        lifecycle_attempt=lifecycle.get("lifecycle_attempt"),
                        exit_observed=lifecycle.get("exit_observed"),
                        start_received_within_budget=lifecycle.get(
                            "start_received_within_budget"
                        ),
                        start_received_monotonic=lifecycle.get(
                            "start_received_monotonic"
                        ),
                        host_deadline_monotonic=lifecycle.get(
                            "host_deadline_monotonic"
                        ),
                        deadline_enforcement=dict(
                            lifecycle.get("deadline_enforcement") or {}
                        ),
                        enforcement_route=lifecycle.get("enforcement_route"),
                        host_exit_monotonic=lifecycle.get(
                            "host_exit_monotonic"
                        ),
                    )
                )
        passed_tests = sum(1 for result in results if result.passed)
        return GradeResult(
            passed=bool(results) and passed_tests == len(results),
            n_tests=len(results),
            passed_tests=passed_tests,
            sandboxed=True,
            tests=results,
            error=None if results else "no tests",
            timed_out=any(result.timed_out for result in results),
            duration_s=time.perf_counter() - started,
            sandbox="docker",
        )


def grade_code(
    code: str,
    tests: Iterable[dict[str, Any]],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    sandbox: str = "docker",
    docker: DockerSandbox | None = None,
    allow_fallback: bool = False,
    container_names: Sequence[str] | None = None,
    attempts: Sequence[str] | None = None,
) -> GradeResult:
    """Grade one code sample.

    ``sandbox="local"`` is the non-sandboxed fallback.  ``sandbox="docker"``
    uses :class:`DockerSandbox`; when Docker is unavailable it raises unless
    ``allow_fallback`` is true, in which case the local result explicitly says
    ``sandboxed=False``.

    ``container_names`` and ``attempts`` are the per-test identities chosen by
    the production count adapter.  They are *forwarded unchanged* to
    :meth:`DockerSandbox.run_tests`, which uses the container name for the
    actual ``docker run --name`` and the attempt token for the wrapper
    lifecycle evidence.  Dropping or ignoring either is prohibited: the
    ledger, lifecycle evidence and serialized count row must all agree.
    """

    tests = list(tests)
    if sandbox not in {"local", "docker"}:
        raise ValueError(f"unknown sandbox {sandbox!r}")
    if sandbox == "local":
        if container_names is not None or attempts is not None:
            raise ValueError(
                "container_names/attempts are only meaningful for the Docker "
                "sandbox; the local fallback cannot carry container identity"
            )
        return run_local_tests(code, tests, timeout=timeout)
    docker = docker or DockerSandbox()
    if docker.available():
        return docker.run_tests(
            code,
            tests,
            timeout=timeout,
            container_names=container_names,
            attempts=attempts,
        )
    if not allow_fallback:
        raise RuntimeError("Docker sandbox requested but Docker is not available")
    result = run_local_tests(code, tests, timeout=timeout)
    result.sandbox = "local-fallback"
    result.error = (result.error + "; " if result.error else "") + (
        "Docker unavailable: ran in the NON-SANDBOXED local fallback"
    )
    return result


# ---------------------------------------------------------------------------
# Math grading
# ---------------------------------------------------------------------------
def _normalise_expression(value: str) -> str:
    value = (value or "").strip().strip("$").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace(",", "").replace("\u00a0", "")
    value = value.replace(" ", "")
    for wrapper in ("\\text", "\\mbox", "\\mathrm"):
        prefix = wrapper + "{"
        if value.startswith(prefix) and value.endswith("}"):
            value = value[len(prefix) : -1]
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value


def _math_verify_grade(
    prediction: str,
    answer: str,
    *,
    timeout: float = DEFAULT_MATH_VERIFY_TIMEOUT_S,
) -> tuple[bool | None, bool]:
    """Run math-verify in a grading subprocess.

    Returns ``(result, timed_out)``.  ``result`` is ``None`` when math-verify is
    unavailable or the subprocess failed; ``timed_out`` is true when the outer
    timeout fired.  The subprocess has math-verify's multiprocessing-based
    internal timeouts disabled (``parsing_timeout=None`` / ``timeout_seconds=None``)
    because those fail on Windows; the outer timeout is the registered guard.
    """

    if not MATH_VERIFY_AVAILABLE:
        return None, False
    command = [
        sys.executable,
        "-c",
        MATH_VERIFY_SCRIPT,
        str(prediction),
        str(answer),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(timeout),
            env=sanitized_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, True
    except Exception:  # pragma: no cover - interpreter/exec failures
        return None, False
    if completed.returncode != 0:
        return None, False
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None, False
    if lines[-1] == "MATH_VERIFY_TRUE":
        return True, False
    if lines[-1] == "MATH_VERIFY_FALSE":
        return False, False
    return None, False


def grade_math(
    prediction: str,
    answer: str,
    *,
    use_math_verify: bool = True,
    math_verify_timeout: float = DEFAULT_MATH_VERIFY_TIMEOUT_S,
) -> MathGrade:
    """Grade one math completion under the registered OR rule.

    ``correct`` is true iff math-verify says correct **or** the registered
    AIME/integer normalisation says correct.  ``decided_by`` records which
    verifier decided (``math_verify``, ``normaliser``, ``both`` or ``timeout``).
    """

    answer_text = str(answer or "")
    text = strip_thinking(str(prediction or ""))
    boxed = last_boxed(text)
    candidate = boxed if boxed is not None else text
    normal_answer = _normalise_integer(answer_text)
    if normal_answer == answer_text.strip():
        normal_answer = _normalise_expression(normal_answer)
    else:
        normal_answer = _normalise_expression(normal_answer)
    parsed = boxed is not None
    normal_prediction = _normalise_expression(candidate) if candidate else None
    normaliser_correct = bool(
        parsed
        and normal_prediction is not None
        and normal_prediction == normal_answer
    )

    verified: bool | None = None
    timed_out = False
    if use_math_verify:
        try:
            verified, timed_out = _math_verify_grade(
                candidate, answer_text, timeout=math_verify_timeout
            )
        except subprocess.TimeoutExpired:
            verified, timed_out = None, True
        except Exception:  # pragma: no cover - defensive: fall back to AIME
            verified, timed_out = None, False

    if timed_out:
        correct = normaliser_correct
        decided_by = "timeout"
    elif verified is True and normaliser_correct:
        correct, decided_by = True, "both"
    elif verified is True:
        correct, decided_by = True, "math_verify"
    elif normaliser_correct:
        correct, decided_by = True, "normaliser"
    elif verified is False:
        correct, decided_by = False, "math_verify"
    else:
        correct, decided_by = False, "normaliser"

    method = "math-verify" if verified is not None else "aime-normalise"
    return MathGrade(
        correct=correct,
        predicted=candidate if candidate else None,
        normalized_prediction=normal_prediction,
        normalized_answer=normal_answer,
        method=method,
        parsed=parsed,
        decided_by=decided_by,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Persistent worker pool
# ---------------------------------------------------------------------------
def _code_worker(payload: dict[str, Any]) -> dict[str, Any]:
    return grade_code(
        payload["code"],
        payload["tests"],
        timeout=payload.get("timeout", DEFAULT_TIMEOUT_S),
        sandbox=payload.get("sandbox", "local"),
        allow_fallback=payload.get("allow_fallback", True),
    ).to_dict()


def _math_worker(payload: dict[str, Any]) -> dict[str, Any]:
    return grade_math(
        payload["prediction"],
        payload["answer"],
        use_math_verify=payload.get("use_math_verify", True),
    ).to_dict()


class GraderService:
    """A persistent CPU process pool for code and math verification.

    Workers are spawned once and reused for every batch, so the service can sit
    next to Stage 2 rollouts without paying process-start cost per sample.  Each
    code sample still runs its tests in isolated subprocesses.
    """

    def __init__(
        self,
        workers: int | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        sandbox: str = "docker",
        allow_fallback: bool = False,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
    ) -> None:
        self.workers = int(workers if workers is not None else max(1, (os.cpu_count() or 2) - 1))
        self.timeout = float(timeout)
        self.sandbox = sandbox
        self.allow_fallback = bool(allow_fallback)
        self.docker_image = docker_image
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("GraderService is closed")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)
        return self._executor

    def submit_code(self, code: str, tests: list[dict[str, Any]]):
        return self._get_executor().submit(
            _code_worker,
            {
                "code": code,
                "tests": tests,
                "timeout": self.timeout,
                "sandbox": self.sandbox,
                "allow_fallback": self.allow_fallback,
            },
        )

    def submit_math(self, prediction: str, answer: str):
        return self._get_executor().submit(
            _math_worker,
            {"prediction": prediction, "answer": answer},
        )

    def grade_code(self, code: str, tests: list[dict[str, Any]]) -> GradeResult:
        return _result_from_dict(self.submit_code(code, tests).result())

    def grade_math(self, prediction: str, answer: str) -> MathGrade:
        return MathGrade(**self.submit_math(prediction, answer).result())

    def grade_code_batch(
        self, items: Iterable[tuple[str, list[dict[str, Any]]]]
    ) -> list[GradeResult]:
        futures = [self.submit_code(code, tests) for code, tests in items]
        return [_result_from_dict(future.result()) for future in futures]

    def grade_math_batch(
        self, items: Iterable[tuple[str, str]]
    ) -> list[MathGrade]:
        futures = [self.submit_math(prediction, answer) for prediction, answer in items]
        return [MathGrade(**future.result()) for future in futures]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._closed = True

    def __enter__(self) -> "GraderService":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _result_from_dict(payload: dict[str, Any]) -> GradeResult:
    tests = [TestResult(**test) for test in payload.get("tests", [])]
    return GradeResult(
        passed=payload["passed"],
        n_tests=payload["n_tests"],
        passed_tests=payload["passed_tests"],
        sandboxed=payload["sandboxed"],
        tests=tests,
        error=payload.get("error"),
        timed_out=payload.get("timed_out", False),
        duration_s=payload.get("duration_s", 0.0),
        sandbox=payload.get("sandbox", "local"),
    )


__all__ = [
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_TIMEOUT_S",
    "DockerImageDigestError",
    "DockerSandbox",
    "GradeResult",
    "GraderService",
    "MATH_VERIFY_AVAILABLE",
    "MathGrade",
    "TestResult",
    "grade_code",
    "grade_math",
    "inspect_docker_image",
    "normalize_output",
    "read_lifecycle_markers",
    "lifecycle_record_error",
    "classify_lifecycle_evidence",
    "run_local_tests",
    "sanitized_env",
]
