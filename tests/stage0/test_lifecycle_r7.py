"""Fix round 7: host entry receipt and post-exec entry boundary."""

from __future__ import annotations

import re

from gemmaloop.stage0 import grader


def _rec(event: str, seq: int, mono: float, **extra) -> dict:
    return {
        "v": 1,
        "attempt": "attempt-r7",
        "seq": seq,
        "event": event,
        "wall": 1.0 + mono,
        "monotonic": mono,
        "pid": 1,
        **extra,
    }


def _late_start_records() -> list[dict]:
    # Reviewer probe: host deadline 6.0, start 6.2, delayed notice 6.4.
    return [
        _rec(grader.LIFECYCLE_EVENT_READY, 1, 0.1),
        _rec(grader.LIFECYCLE_EVENT_BUDGET, 2, 0.2, remaining_s=5.8),
        _rec(grader.LIFECYCLE_EVENT_START, 3, 6.2, child_pid=77),
        _rec(
            grader.LIFECYCLE_EVENT_ALIVE,
            4,
            6.4,
            child_pid=77,
            notice_monotonic=6.4,
        ),
        _rec(
            grader.LIFECYCLE_EVENT_FINISH,
            5,
            6.5,
            child_pid=77,
            returncode=-9,
            terminated=True,
            exit_observed=True,
        ),
    ]


def test_post_deadline_start_with_delayed_notice_is_not_a_deadline_failure():
    records = _late_start_records()
    # No host receipt, or a receipt stamped after the host deadline, is
    # infrastructure/unknown.  Before the fix this was candidate_running.
    assert grader.classify_lifecycle_evidence(
        markers=records,
        cleanup_verified=True,
        deadline_epoch=999.0,
        timed_out=True,
    ) == grader.LIFECYCLE_UNRESOLVED
    assert grader.classify_lifecycle_evidence(
        markers=records,
        cleanup_verified=True,
        deadline_epoch=999.0,
        timed_out=True,
        host_start_received_within_budget=False,
    ) == grader.LIFECYCLE_UNRESOLVED
    error = grader.lifecycle_record_error(
        markers=records,
        cleanup_verified=True,
        timed_out=True,
        host_start_received_within_budget=False,
    )
    assert error and "received by the host" in error

    # The same records with a receipt before 6.0 are a genuine deadline
    # failure: entry was established, the child was alive at the notice, and
    # termination was verified.
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


def test_host_receipt_is_stamped_on_the_host_clock_not_the_wrapper_clock(
    tmp_path, monkeypatch
):
    """The runtime monitor must stamp receipt before the registered deadline."""

    import hashlib
    import hmac
    import json
    import time

    from gemmaloop.stage0 import grader as module

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    key = "a" * 64
    attempt = "attempt-r7"
    records = _late_start_records()[:3]  # ready, budget, start at wrapper 6.2

    def signed(record: dict) -> str:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(
            key.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        return mac + " GML1 " + body + "\n"

    # Only the pre-start frames exist before the host deadline.  The start
    # arrives on the evidence channel after the host deadline, so the host
    # stamps its receipt late.
    (evidence / module.LIFECYCLE_EVIDENCE_FILE).write_text(
        "".join(signed(record) for record in records[:2]), encoding="utf-8"
    )

    class FakeProcess:
        pid = 111
        returncode = 0

        def communicate(self, input_text=None, timeout=None):  # noqa: A002
            time.sleep(0.1)
            with (evidence / module.LIFECYCLE_EVIDENCE_FILE).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(signed(records[2]))
            # Let the host monitor poll the evidence channel once more before
            # the process stops.
            time.sleep(0.05)
            return "", ""

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: FakeProcess())
    result = module._run_process_docker(
        ["docker", "run"],
        input_text="",
        timeout=0.05,
        container_name="container-r7",
        evidence_dir=evidence,
        key=key,
        attempt=attempt,
        state_probe=lambda name: (None, "no state"),
        cleanup=lambda name: True,
        marker_reader=lambda *a, **k: (records, None),
    )
    _code, _stdout, _stderr, _timed_out, _duration, lifecycle = result
    # The wrapper's start record claims monotonic 6.2, but the host stamped the
    # receipt on its own clock relative to a 0.05 s deadline: it is late.
    assert lifecycle["start_received_within_budget"] is False
    assert lifecycle["start_received_monotonic"] is not None
    assert lifecycle["host_deadline_monotonic"] is not None
    assert (
        lifecycle["start_received_monotonic"]
        >= lifecycle["host_deadline_monotonic"]
    )


def test_launcher_death_before_post_exec_entry_is_not_entry_evidence():
    wrapper = grader._LIFECYCLE_WRAPPER_SOURCE
    runner = grader._LIFECYCLE_RUNNER_SOURCE
    # There is a separate frozen runner whose bytes are part of the freeze.
    assert "post_exec_entry" in runner
    assert "runpy.run_path" in runner
    assert runner.index("post_exec_entry") < runner.index("runpy.run_path")
    assert runner.index("os.close(entry_fd)") < runner.index("runpy.run_path")
    # The launcher execs that runner; its pre-exec exec_ok frame is diagnostic.
    assert "exec_ok" in wrapper
    assert '"post_exec_entry"' in wrapper
    launcher = wrapper.split("LAUNCHER_SOURCE = r", 1)[1].split('"""', 2)[1]
    assert "os.execv(" in launcher
    assert "runner" in launcher
    assert "os.path.join(\"/work\", target)" in launcher
    # Entry evidence is the runner's post-exec frame, never exec_ok.
    maybe_start = wrapper.split("def maybe_start():", 1)[1].split(
        "finish_status = None", 1
    )[0]
    assert "post_exec_confirmed" in maybe_start
    assert not re.search(r"\bexec_confirmed\b", maybe_start)
    # A wrapper death after the launcher's exec_ok but before the runner frame
    # closes both pipes without a start, which is unresolved.
    assert "successful post-exec execution entry was not observed" in wrapper
