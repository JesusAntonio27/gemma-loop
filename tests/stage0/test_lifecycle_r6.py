"""Fix round 6: causal host-deadline notice and positive exec handoff."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest

from gemmaloop.stage0 import grader


def _rec(event: str, seq: int, mono: float, **extra) -> dict:
    return {
        "v": 1,
        "attempt": "attempt-r6",
        "seq": seq,
        "event": event,
        "wall": 1.0 + mono,
        "monotonic": mono,
        "pid": 1,
        **extra,
    }


def _base() -> list[dict]:
    return [
        _rec(grader.LIFECYCLE_EVENT_READY, 1, 100.0),
        _rec(grader.LIFECYCLE_EVENT_BUDGET, 2, 100.1, remaining_s=5.0),
        _rec(grader.LIFECYCLE_EVENT_START, 3, 100.2, child_pid=77),
    ]


def _finish(seq: int, mono: float) -> dict:
    return _rec(
        grader.LIFECYCLE_EVENT_FINISH,
        seq,
        mono,
        child_pid=77,
        returncode=0,
        terminated=True,
        exit_observed=True,
    )


def test_wrapper_has_no_wrapper_side_deadline_or_safety_margin():
    source = grader._LIFECYCLE_WRAPPER_SOURCE
    # The old wrapper computed ``now + remaining - 50 ms`` and enforced its
    # own deadline.  That cannot prove liveness at the registered host
    # deadline, so it must be gone.
    assert "SAFETY_MARGIN" not in source
    assert "deadline = time.monotonic() + remaining" not in source
    assert "- 0.05" not in source
    # Liveness is only accepted after the host's authenticated notice.
    assert "read_deadline_notice(key, attempt)" in source
    assert "notice_monotonic=notice_mono" in source
    assert "EVENT_EXITED_AT_NOTICE" in source


def test_positive_exec_confirmation_requires_post_exec_frame_and_closed_pipe():
    source = grader._LIFECYCLE_WRAPPER_SOURCE
    # The frozen launcher proves the bootstrap exec happened; the setup pipe is
    # close-on-exec only after the launcher itself execs the candidate.
    assert 'body = json.dumps(\n            {\n                "event": "exec_ok"' in source
    assert 'os.execv(\n            sys.executable,\n            [sys.executable, "-I", os.path.join("/work", target)],\n        )' in source or "handoff_failed" in source
    assert "exec_confirmed" in source
    assert "and not setup_open" in source
    assert "and not entry_open" in source
    # An entry frame alone can never emit a start record.
    assert (
        "if any(frame.get(\"event\") == \"entry\" for frame in frames):\n"
        "            emit(EVENT_START"
    ) not in source


def test_deadline_notice_is_authenticated_and_bound_to_attempt(tmp_path: Path):
    key = "e" * 64
    assert grader._write_deadline_notice(
        tmp_path, key=key, attempt="attempt-r6", host_deadline_epoch=123.0
    )
    text = (tmp_path / grader.LIFECYCLE_DEADLINE_FILE).read_text(encoding="utf-8")
    mac, rest = text.strip().split(" ", 1)
    prefix, body = rest.split(" ", 1)
    assert prefix == grader.LIFECYCLE_DEADLINE_FRAME_PREFIX
    expected = hmac.new(
        key.encode(),
        (prefix + " " + body).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert hmac.compare_digest(mac, expected)
    payload = json.loads(body)
    assert payload["attempt"] == "attempt-r6"
    assert payload["event"] == "deadline_reached"
    assert payload["host_deadline_epoch"] == 123.0


def test_alive_only_after_notice_is_a_deadline_failure():
    records = _base() + [
        _rec(
            grader.LIFECYCLE_EVENT_ALIVE,
            4,
            106.01,
            child_pid=77,
            notice_monotonic=106.0,
        ),
        _finish(5, 106.2),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=records,
        cleanup_verified=True,
        deadline_epoch=999.0,
        timed_out=True,
        host_start_received_within_budget=True,
    ) == grader.LIFECYCLE_CANDIDATE_RUNNING
    assert (
        grader.lifecycle_record_error(
            markers=records,
            cleanup_verified=True,
            timed_out=True,
            host_start_received_within_budget=True,
        )
        is None
    )
    # An observation before the notice is not deadline evidence.
    early = _base() + [
        _rec(
            grader.LIFECYCLE_EVENT_ALIVE,
            4,
            105.0,
            child_pid=77,
            notice_monotonic=106.0,
        ),
        _finish(5, 106.2),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=early, cleanup_verified=True, deadline_epoch=999.0, timed_out=True
    ) == grader.LIFECYCLE_UNRESOLVED


def test_child_exited_at_notice_is_unknown_not_timeout():
    # The child exited, but its completion did not reach the host before the
    # deadline.  The wrapper observed that at the notice; this is unknown.
    records = _base() + [
        _rec(
            grader.LIFECYCLE_EVENT_EXITED_AT_NOTICE,
            4,
            106.01,
            child_pid=77,
            notice_monotonic=106.0,
        ),
        _finish(5, 106.02),
    ]
    assert grader.classify_lifecycle_evidence(
        markers=records,
        cleanup_verified=True,
        deadline_epoch=999.0,
        timed_out=True,
    ) == grader.LIFECYCLE_UNRESOLVED
    assert "exited at the deadline notice" in (
        grader.lifecycle_record_error(
            markers=records, cleanup_verified=True, timed_out=True
        )
        or ""
    )


def test_completion_after_host_budget_is_unknown():
    records = _base() + [_finish(4, 101.0)]
    assert grader.classify_lifecycle_evidence(
        markers=records,
        cleanup_verified=True,
        deadline_epoch=999.0,
        timed_out=False,
        host_completed_within_budget=False,
    ) == grader.LIFECYCLE_UNRESOLVED
    assert grader.lifecycle_record_error(
        markers=records,
        cleanup_verified=True,
        timed_out=False,
        host_completed_within_budget=False,
    )


class _FakeTimeoutProcess:
    """A docker-run process that is still alive when the host deadline hits."""

    pid = 4242
    returncode = -9

    def __init__(self, evidence_dir: Path) -> None:
        self.evidence_dir = evidence_dir
        self.calls = 0

    def communicate(self, input_text=None, timeout=None):  # noqa: A002
        self.calls += 1
        if self.calls == 1:
            notice = self.evidence_dir / grader.LIFECYCLE_DEADLINE_FILE
            limit = time.time() + 2.0
            while not notice.exists() and time.time() < limit:
                time.sleep(0.005)
            assert notice.exists(), "host killed before sending the deadline notice"
            raise subprocess_timeout()
        return "", ""


def subprocess_timeout():
    import subprocess

    return subprocess.TimeoutExpired(["docker"], 0.2)


def test_host_notice_is_sent_before_the_kill_and_alive_notice_is_accepted(
    tmp_path: Path, monkeypatch
):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    key = "f" * 64
    attempt = "attempt-r6"
    records = _base() + [
        _rec(
            grader.LIFECYCLE_EVENT_ALIVE,
            4,
            106.01,
            child_pid=77,
            notice_monotonic=106.0,
        ),
        _finish(5, 106.2),
    ]

    lines = []
    for record in records:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            key.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        lines.append(mac + " GML1 " + body + "\n")
    (evidence / grader.LIFECYCLE_EVIDENCE_FILE).write_text(
        "".join(lines), encoding="utf-8"
    )

    process = _FakeTimeoutProcess(evidence)

    def fake_popen(*args, **kwargs):
        return process

    def fake_run(*args, **kwargs):
        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    monkeypatch.setattr(grader.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(grader.subprocess, "run", fake_run)
    result = grader._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=0.2,
        container_name="container-r6",
        evidence_dir=evidence,
        key=key,
        attempt=attempt,
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
        marker_reader=lambda *args, **kwargs: (records, None),
    )
    _code, _stdout, _stderr, timed_out, _duration, lifecycle = result
    assert timed_out is True
    assert lifecycle["class"] == grader.LIFECYCLE_CANDIDATE_RUNNING
    assert lifecycle["deadline_alive"] is True
    assert lifecycle["deadline_notice"]["sent"] is True
    assert (evidence / grader.LIFECYCLE_DEADLINE_FILE).is_file()
