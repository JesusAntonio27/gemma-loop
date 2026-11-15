"""The retry orchestrator's crash-safety v2 contract (Task 14b)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.orchestrate as orchestrate
from gemmaloop.train.checkpoint import DONE_FILE, finalize_checkpoint

FAKE = """
import json, pathlib, sys
log = pathlib.Path(sys.argv[1])
state = pathlib.Path(sys.argv[2])
args = sys.argv[3:]
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
count = int(state.read_text()) if state.exists() else 0
state.write_text(str(count + 1))
mode = {mode!r}
if mode == "retry-twice":
    sys.exit(0 if count >= 2 else 75)
if mode == "fail":
    sys.exit(1)
if mode == "always-busy":
    sys.exit(75)
"""


def _fake(tmp_path, mode: str):
    script = tmp_path / "fake.py"
    script.write_text(FAKE.format(mode=mode), encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    state = tmp_path / "state.txt"
    command = [sys.executable, str(script), str(log), str(state)]
    return command, log, state


def _events(out):
    return [
        json.loads(line)
        for line in (out / orchestrate.ORCHESTRATOR_LOG).read_text(encoding="utf-8").splitlines()
    ]


def _checkpoint(out: Path, step: int, tokens: int) -> Path:
    staging = out / f"checkpoint-{step}.tmp"
    staging.mkdir(parents=True)
    (staging / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    finalize_checkpoint(
        staging,
        out / f"checkpoint-{step}",
        {"step": step, "tokens_seen": tokens, "complete": True},
    )
    return out / f"checkpoint-{step}"


def _run_cli(argv):
    return orchestrate.main(argv)


def test_retries_75_with_resume_auto_then_succeeds(tmp_path):
    out = tmp_path / "run"
    command, calls, _ = _fake(tmp_path, "retry-twice")
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "5",
            "--",
            *command,
            "--arm", "L",
        ]
    )
    assert code == 0
    invocations = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(invocations) == 3
    assert invocations[0] == ["--arm", "L"]
    assert invocations[1] == ["--arm", "L", "--resume", "auto"]
    assert invocations[2] == ["--arm", "L", "--resume", "auto"]
    assert [event["event"] for event in _events(out)].count("launch") == 3
    assert _events(out)[-1]["event"] == "done"


def test_any_nonzero_is_retried_until_no_progress_limit(tmp_path, monkeypatch):
    out = tmp_path / "run"
    sleeps: list[float] = []
    monkeypatch.setattr(orchestrate.time, "sleep", sleeps.append)
    calls = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del command, env, cwd
        calls["count"] += 1
        return SimpleNamespace(returncode=1)  # killed child / driver reset

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(
        ["--out", str(out), "--backoff-initial", "0", "--max-no-progress", "5"]
        + ["--", "run-train"]
    )
    assert code == 75
    assert calls["count"] == 5
    assert _events(out)[-1]["reason"] == "no progress"


def test_exit_2_is_terminal(tmp_path, monkeypatch):
    out = tmp_path / "run"
    calls = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del command, env, cwd
        calls["count"] += 1
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(["--out", str(out), "--", "run-train"])
    assert code == 2
    assert calls["count"] == 1
    assert _events(out)[-1]["reason"] == "terminal refusal"


def test_progress_resets_the_retry_counter_and_backoff(tmp_path, monkeypatch):
    out = tmp_path / "run"
    sleeps: list[float] = []
    monkeypatch.setattr(orchestrate.time, "sleep", sleeps.append)
    attempts = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del command, env, cwd
        attempts["count"] += 1
        step = attempts["count"]
        if step > 25:
            return SimpleNamespace(returncode=0)
        _checkpoint(out, step, step * 100)
        return SimpleNamespace(returncode=75)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(
        ["--out", str(out), "--backoff-initial", "60", "--max-no-progress", "5"]
        + ["--", "run-train"]
    )
    assert code == 0
    assert attempts["count"] == 26
    waits = [event["seconds"] for event in _events(out) if event["event"] == "wait"]
    assert len(waits) == 25
    assert all(seconds == 60 for seconds in waits)


def test_done_is_a_no_op_without_launching(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / DONE_FILE).write_text(
        json.dumps({"status": "complete", "run_identity_hash": "x"}),
        encoding="utf-8",
    )
    called = {"count": 0}

    def fake_run(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("must not launch a completed run")

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(["--out", str(out), "--", "run-train"])
    assert code == 0
    assert called["count"] == 0
    events = _events(out)
    assert [event["event"] for event in events] == ["start", "done"]
    assert events[-1]["reason"] == "already complete"


def test_first_attempt_resumes_when_the_run_already_has_a_checkpoint(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _checkpoint(out, 4, 400)
    command, calls, _ = _fake(tmp_path, "retry-twice")
    code = _run_cli(
        ["--out", str(out), "--backoff-initial", "0", "--max-no-progress", "5", "--", *command]
    )
    assert code == 0
    invocations = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert invocations[0][-2:] == ["--resume", "auto"]
    assert any(event["event"] == "auto-resume" for event in _events(out))


def test_explicit_resume_is_preserved_without_newer_progress(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    explicit = _checkpoint(out, 1, 100)
    commands: list[list[str]] = []
    monkeypatch.setattr(orchestrate.time, "sleep", lambda _s: None)

    def fake_run(command, env=None, cwd=None):
        del env, cwd
        commands.append(list(command))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "3",
            "--",
            "run-train",
            "--resume",
            str(explicit),
        ]
    )
    assert code == 75
    assert len(commands) == 3
    assert all(command[-2:] == ["--resume", str(explicit)] for command in commands)


def test_explicit_resume_switches_to_auto_after_newer_checkpoint(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    explicit = _checkpoint(out, 1, 100)
    commands: list[list[str]] = []
    attempts = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del env, cwd
        commands.append(list(command))
        attempts["count"] += 1
        if attempts["count"] == 1:
            _checkpoint(out, 2, 200)
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrate.time, "sleep", lambda _s: None)
    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "5",
            "--",
            "run-train",
            "--resume",
            str(explicit),
        ]
    )
    assert code == 0
    assert commands[0][-2:] == ["--resume", str(explicit)]
    assert commands[1][-2:] == ["--resume", "auto"]


def test_lock_busy_is_retried_and_does_not_consume_crash_budget(tmp_path, monkeypatch):
    out = tmp_path / "run"
    command, calls, _ = _fake(tmp_path, "retry-twice")
    states = iter([False, True, True, True])
    monkeypatch.setattr(orchestrate, "_gpu_lock_free", lambda: next(states, True))
    code = _run_cli(
        ["--out", str(out), "--backoff-initial", "0", "--max-no-progress", "5", "--", *command]
    )
    assert code == 0
    events = [event["event"] for event in _events(out)]
    assert "blocked" in events
    assert events[-1] == "done"


def test_with_resume_replaces_existing_resume_value():
    assert orchestrate.with_resume(["train.py", "--resume", "old", "--x"]) == [
        "train.py",
        "--x",
        "--resume",
        "auto",
    ]
    assert orchestrate.with_resume(["train.py", "--resume=old"]) == [
        "train.py",
        "--resume",
        "auto",
    ]


def test_backoff_schedule():
    assert orchestrate.backoff_seconds(0) == pytest.approx(60)
    assert orchestrate.backoff_seconds(1) == pytest.approx(120)
    assert orchestrate.backoff_seconds(2) == pytest.approx(300)
    assert orchestrate.backoff_seconds(3) == pytest.approx(600)
    assert orchestrate.backoff_seconds(100, maximum=1800) == pytest.approx(1800)


def test_run_progress_uses_only_complete_checkpoints(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    assert orchestrate.run_progress(out) == (0, 0)
    _checkpoint(out, 3, 333)
    assert orchestrate.run_progress(out) == (333, 3)
    (out / "status.json").write_text(
        json.dumps({"tokens_seen": 999, "step": 9}), encoding="utf-8"
    )
    assert orchestrate.run_progress(out) == (333, 3)


def test_fresh_is_dropped_after_the_first_attempt(tmp_path, monkeypatch):
    out = tmp_path / "run"
    commands: list[list[str]] = []
    attempts = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del env, cwd
        commands.append(list(command))
        attempts["count"] += 1
        return SimpleNamespace(returncode=0 if attempts["count"] > 1 else 1)

    monkeypatch.setattr(orchestrate.time, "sleep", lambda _s: None)
    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "3",
            "--",
            "run-train",
            "--fresh",
        ]
    )
    assert code == 0
    assert "--fresh" in commands[0]
    assert "--fresh" not in commands[1]
    assert commands[1][-2:] == ["--resume", "auto"]


def test_per_run_lock_blocks_a_duplicate_orchestrator(tmp_path, monkeypatch):
    from gemmaloop.gpulock import GpuLockBusy

    out = tmp_path / "run"

    def busy(_path):
        raise GpuLockBusy("held")

    monkeypatch.setattr(orchestrate, "acquire", busy)
    called = {"count": 0}

    def fake_run(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("must not launch")

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(["--out", str(out), "--no-lock-check", "--", "run-train"])
    assert code == 75
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# Exit 76 is a terminal heal-monitor abort with an explicit ack
# ---------------------------------------------------------------------------


def _write_monitor_abort(out: Path, alert_id_value: str = "abc123") -> None:
    out.mkdir(parents=True, exist_ok=True)
    alert = {
        "severity": "abort",
        "rule": "nonfinite",
        "message": "non-finite loss",
        "step": 7,
        "tokens_seen": 700,
        "alert_id": alert_id_value,
    }
    (out / orchestrate.MONITOR_ALERTS_FILE).write_text(
        json.dumps({"aborts": [], "alerts": [alert]}), encoding="utf-8"
    )
    orchestrate.write_abort_guard(out, alert)
    (out / orchestrate.MONITOR_STATE_FILE).write_text(
        json.dumps(
            {
                "rules": {
                    "abort_triggered": True,
                    "abort_emitted": ["nonfinite"],
                    "warn_last": {"nonfinite": 700},
                }
            }
        ),
        encoding="utf-8",
    )
    (out / "run.json").write_text(
        json.dumps({"arm": "L", "status": "running"}), encoding="utf-8"
    )
    (out / "status.json").write_text(
        json.dumps({"phase": "train", "exit_code": None}), encoding="utf-8"
    )


def test_exit_76_is_terminal_and_records_the_alert(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    calls = {"count": 0}

    def fake_run(command, env=None, cwd=None):
        del command, env, cwd
        calls["count"] += 1
        # The aborted step had progressed; a retry would otherwise reset the
        # no-progress counter and loop forever.
        _checkpoint(out, 2, 200)
        return SimpleNamespace(returncode=orchestrate.MONITOR_ABORT_EXIT_CODE)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = _run_cli(["--out", str(out), "--", "run-train"])
    assert code == orchestrate.MONITOR_ABORT_EXIT_CODE
    assert calls["count"] == 1
    events = _events(out)
    assert [event["event"] for event in events].count("launch") == 1
    assert events[-1]["event"] == "stop"
    assert events[-1]["reason"] == "heal-monitor abort"
    assert events[-1]["alert_id"]
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == orchestrate.MONITOR_ABORT_EXIT_CODE
    assert status["heal_monitor_abort"]["rule"]
    assert (out / orchestrate.ABORT_GUARD_FILE).is_file()


def test_abort_requires_the_matching_ack_and_clears_flags(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _write_monitor_abort(out, "abc123")
    # A valid checkpoint keeps the ack on the resume path; the separate N1
    # test covers an abort before the first checkpoint (archive + restart).
    _checkpoint(out, 1, 100)
    calls = {"count": 0, "command": []}

    def fake_run(command, env=None, cwd=None):
        del env, cwd
        calls["count"] += 1
        calls["command"] = list(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    assert _run_cli(["--out", str(out), "--", "run-train"]) == 2
    assert calls["count"] == 0
    assert _events(out)[-1]["reason"] == "abort not acknowledged"

    assert _run_cli(["--out", str(out), "--ack-abort", "wrong", "--", "run-train"]) == 2
    assert calls["count"] == 0

    assert (
        _run_cli(["--out", str(out), "--ack-abort", "abc123", "--", "run-train"])
        == 0
    )
    assert calls["count"] == 1
    # The acknowledged abort resumes the last COMPLETE checkpoint in place,
    # and the child receives the ack it must record before training.
    assert calls["command"] == [
        "run-train",
        "--ack-abort",
        "abc123",
        "--resume",
        "auto",
    ]
    assert not (out / orchestrate.ABORT_GUARD_FILE).exists()
    events = _events(out)
    assert any(event["event"] == "ack-abort" for event in events)
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["abort_acks"][-1]["alert_id"] == "abc123"
    state = json.loads((out / orchestrate.MONITOR_STATE_FILE).read_text(encoding="utf-8"))
    assert state["rules"]["abort_triggered"] is False
    assert state["rules"]["abort_emitted"] == []


def test_ack_is_recorded_once_when_train_and_orchestrator_both_record(tmp_path):
    """R3-M2: the child records the ack, then the orchestrator must not add one."""

    import scripts.train as train_script

    out = tmp_path / "run"
    _write_monitor_abort(out, "abc123")
    guard = {
        "alert_id": "abc123",
        "alert": {
            "alert_id": "abc123",
            "rule": "nonfinite",
            "message": "non-finite loss",
        },
    }

    train_script.record_abort_ack(out, guard)
    orchestrate.record_abort_ack(out, guard["alert"])

    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    acks = [
        item for item in payload["abort_acks"] if item["alert_id"] == "abc123"
    ]
    assert len(acks) == 1
    assert acks[0]["source"] == "train --ack-abort"


def test_final_run_json_preserves_orchestrator_abort_acks(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    out.mkdir()
    (out / "run.json").write_text(
        json.dumps({"abort_acks": [{"alert_id": "abc123"}]}), encoding="utf-8"
    )
    train_script.write_run_json(out, {"status": "complete", "arm": "L"})
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["abort_acks"][0]["alert_id"] == "abc123"


def test_rerun_complete_relaunches_a_done_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / DONE_FILE).write_text(
        json.dumps({"status": "complete", "run_identity_hash": "x"}),
        encoding="utf-8",
    )
    marker = tmp_path / "child-ran.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('ran')\n",
        encoding="utf-8",
    )
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--",
            sys.executable,
            str(child),
            str(marker),
            "--rerun-complete",
        ]
    )
    assert code == 0
    assert marker.is_file()
    events = _events(out)
    assert any(event["event"] == "launch" for event in events)


def test_partial_run_without_a_checkpoint_is_archived_and_restarted(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "run_identity.json").write_text('{"git_sha": "x"}', encoding="utf-8")
    (out / "status.json").write_text('{"phase": "loading-model"}', encoding="utf-8")
    (out / "heal_monitor_state.json").write_text(
        '{"last_probe_tokens": 3000000}', encoding="utf-8"
    )
    (out / "alerts.json").write_text('{"alerts": []}', encoding="utf-8")
    marker = tmp_path / "child-ran.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('ran')\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv[3:]))\n",
        encoding="utf-8",
    )
    args_file = tmp_path / "child-args.json"
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--",
            sys.executable,
            str(child),
            str(marker),
            str(args_file),
        ]
    )
    assert code == 0
    assert marker.is_file()
    assert json.loads(args_file.read_text(encoding="utf-8")) == []
    assert not (out / "run_identity.json").exists()
    assert not (out / "heal_monitor_state.json").exists()
    archived = list((out / "attic").glob("partial-no-checkpoint-*"))
    assert archived
    names = {child.name for child in archived[0].iterdir()}
    assert {"run_identity.json", "status.json", "heal_monitor_state.json", "alerts.json"} <= names
    assert any(event["event"] == "archive-partial" for event in _events(out))


def test_acked_abort_before_the_first_checkpoint_archives_and_restarts_once(tmp_path):
    """N1: an acked early abort archives and launches from zero in one step."""

    out = tmp_path / "run"
    _write_monitor_abort(out, "early-abort")
    (out / "run_identity.json").write_text('{"git_sha": "x"}', encoding="utf-8")
    (out / "status.json").write_text(
        '{"phase": "aborted-heal-monitor"}', encoding="utf-8"
    )
    marker = tmp_path / "child-ran.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "state = out / 'run_identity.json'\n"
        "if state.exists():\n"
        "    sys.exit(2)\n"
        "pathlib.Path(sys.argv[2]).write_text('ran')\n",
        encoding="utf-8",
    )
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--ack-abort", "early-abort",
            "--",
            sys.executable,
            str(child),
            str(out),
            str(marker),
        ]
    )
    assert code == 0
    assert marker.is_file()
    assert not (out / "run_identity.json").exists()
    assert list((out / "attic").glob("partial-no-checkpoint-*"))
    events = [event["event"] for event in _events(out)]
    assert "ack-abort" in events
    assert events.index("archive-partial") < events.index("launch")


def test_forced_rerun_keeps_the_force_flag_until_the_stale_done_is_gone(tmp_path):
    """N2: a 75 before the archive must retry with --rerun-complete."""

    out = tmp_path / "run"
    out.mkdir()
    (out / DONE_FILE).write_text(
        json.dumps({"status": "complete", "run_identity_hash": "old"}),
        encoding="utf-8",
    )
    log = tmp_path / "calls.jsonl"
    counter = tmp_path / "count.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "log = pathlib.Path(sys.argv[1])\n"
        "count = pathlib.Path(sys.argv[2])\n"
        "out = pathlib.Path(sys.argv[3])\n"
        "args = sys.argv[4:]\n"
        "with log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "n = int(count.read_text()) if count.exists() else 0\n"
        "count.write_text(str(n + 1))\n"
        "if n == 0:\n"
        "    sys.exit(75)\n"  # archive failed before DONE moved
        "done = out / 'DONE.json'\n"
        "if done.exists():\n"
        "    done.unlink()\n"
        "(out / 'run.json').write_text(json.dumps({'status': 'complete'}))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "5",
            "--",
            sys.executable,
            str(child),
            str(log),
            str(counter),
            str(out),
            "--rerun-complete",
        ]
    )
    assert code == 0
    invocations = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 2
    assert "--rerun-complete" in invocations[0]
    assert "--rerun-complete" in invocations[1]
    assert "--resume" not in invocations[1]


def test_forced_rerun_does_not_relaunch_a_fresh_done_that_exited_nonzero(tmp_path):
    """R2-M1: the stale DONE must be distinguished from the child's new DONE."""

    out = tmp_path / "run"
    out.mkdir()
    stale = {"status": "complete", "run_identity_hash": "old", "completed_utc": "old"}
    (out / DONE_FILE).write_text(json.dumps(stale), encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "log = pathlib.Path(sys.argv[1])\n"
        "out = pathlib.Path(sys.argv[2])\n"
        "args = sys.argv[3:]\n"
        "with log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "(out / 'DONE.json').write_text(\n"
        "    json.dumps({'status': 'complete', 'run_identity_hash': 'new',\n"
        "                'completed_utc': 'new'}), encoding='utf-8')\n"
        "sys.exit(1)\n",  # crash after publishing the fresh result
        encoding="utf-8",
    )
    code = _run_cli(
        [
            "--out", str(out),
            "--backoff-initial", "0",
            "--max-no-progress", "5",
            "--",
            sys.executable,
            str(child),
            str(log),
            str(out),
            "--rerun-complete",
        ]
    )
    assert code == 0
    invocations = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 1
    assert "--rerun-complete" in invocations[0]
    assert json.loads((out / DONE_FILE).read_text(encoding="utf-8"))[
        "run_identity_hash"
    ] == "new"
    assert not (out / "attic").exists()
    events = _events(out)
    assert events[-1]["event"] == "done"
    assert events[-1]["reason"] == "fresh DONE written by the forced rerun"
