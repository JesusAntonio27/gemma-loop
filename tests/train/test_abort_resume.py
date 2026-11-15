"""The registered resume-over-abort rule shared by train.py and the orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.train as train
from gemmaloop.train.checkpoint import (
    ABORT_GUARD_FILE,
    decide_abort_resume,
    finalize_checkpoint,
)


def _guard(out: Path, alert_id: str = "abc123") -> None:
    out.mkdir(parents=True, exist_ok=True)
    alert = {
        "severity": "abort",
        "rule": "nonfinite",
        "message": "non-finite loss",
        "step": 7,
        "tokens_seen": 700,
        "alert_id": alert_id,
    }
    (out / ABORT_GUARD_FILE).write_text(
        json.dumps(
            {
                "alert_id": alert_id,
                "alert": alert,
                "returncode": 76,
                "created_utc": "2026-11-05T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


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


class _NullGuard:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_no_abort_record_needs_no_ack(tmp_path):
    decision = decide_abort_resume(tmp_path, None)
    assert decision.action == "none"
    assert decision.guard is None


def test_train_refuses_a_resume_over_an_abort_record_without_the_ack(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "run"
    _guard(out)
    monkeypatch.setattr(train, "prevent_sleep", lambda: _NullGuard())

    with pytest.raises(SystemExit) as exc:
        train.main(
            [
                "--arm", "O",
                "--size", "e2b",
                "--tokens", "1",
                "--data", str(tmp_path / "data"),
                "--seed", "0",
                "--out", str(out),
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "abc123" in stderr
    assert "--ack-abort" in stderr
    # The refusal must not consume or move the persisted abort.
    assert (out / ABORT_GUARD_FILE).is_file()


def test_ack_resumes_the_last_complete_checkpoint(tmp_path):
    out = tmp_path / "run"
    _guard(out)
    _checkpoint(out, 1, 100)
    latest = _checkpoint(out, 2, 200)
    (out / "heal_monitor_state.json").write_text(
        json.dumps({"rules": {"abort_triggered": True, "abort_emitted": ["x"]}}),
        encoding="utf-8",
    )
    decision = decide_abort_resume(out, "abc123")
    assert decision.action == "resume"
    assert decision.checkpoint is not None
    assert decision.checkpoint.path == latest
    assert decision.checkpoint.step == 2

    args = SimpleNamespace(out=out, resume=None)
    assert train.apply_abort_decision(args, decision) is False
    assert args.resume == str(latest)
    assert train.handle_abort_resume(args, decision) is None

    assert not (out / ABORT_GUARD_FILE).exists()
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["abort_acks"][-1]["alert_id"] == "abc123"
    state = json.loads(
        (out / "heal_monitor_state.json").read_text(encoding="utf-8")
    )
    assert state["rules"]["abort_triggered"] is False
    assert state["rules"]["abort_emitted"] == []
    # A checkpoint means no archive: the run resumes in place.
    assert not (out / "attic").exists()
    assert (out / "checkpoint-2").is_dir()


def test_ack_without_a_checkpoint_archives_and_restarts_from_zero(tmp_path):
    out = tmp_path / "run"
    _guard(out)
    (out / "run_identity.json").write_text('{"git_sha": "x"}', encoding="utf-8")
    (out / "status.json").write_text('{"phase": "train"}', encoding="utf-8")
    (out / "heal_monitor_state.json").write_text(
        json.dumps({"last_probe_tokens": 520000, "rules": {}}),
        encoding="utf-8",
    )
    decision = decide_abort_resume(out, "abc123")
    assert decision.action == "restart"
    assert decision.checkpoint is None

    args = SimpleNamespace(out=out, resume=None)
    assert train.apply_abort_decision(args, decision) is True
    assert args.resume is None
    assert train.handle_abort_resume(args, decision) is None

    assert not (out / "run_identity.json").exists()
    assert not (out / "status.json").exists()
    assert not (out / "heal_monitor_state.json").exists()
    assert not (out / ABORT_GUARD_FILE).exists()
    archived = list((out / "attic").glob("ack-abort-*"))
    assert archived
    names = {child.name for child in archived[0].iterdir()}
    assert {"run_identity.json", "status.json", "heal_monitor_state.json"} <= names
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["abort_acks"][-1]["alert_id"] == "abc123"


def test_wrong_ack_is_a_refusal(tmp_path):
    out = tmp_path / "run"
    _guard(out, "right")
    decision = decide_abort_resume(out, "wrong")
    assert decision.action == "refuse"
    assert decision.alert_id == "right"


class _StopBeforeModel(RuntimeError):
    """Raised by the fake loader once main has passed the resume decision."""


def _minimal_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps({"sha256": "abort-resume", "max_length": 1024}),
        encoding="utf-8",
    )
    return data


def _run_main_args(tmp_path: Path, out: Path, data: Path, *extra: str) -> list[str]:
    return [
        "--arm", "O",
        "--size", "e2b",
        "--tokens", "1",
        "--data", str(data),
        "--seed", "0",
        "--out", str(out),
        "--max-length", "1024",
        "--vram-cap", "0",
        *extra,
    ]


def test_resume_auto_with_ack_and_no_checkpoint_restarts_in_one_invocation(
    tmp_path, monkeypatch
):
    """R3-I2: the restart branch must clear a still-set --resume auto."""

    out = tmp_path / "run"
    _guard(out)
    (out / "run_identity.json").write_text('{"git_sha": "x"}', encoding="utf-8")
    (out / "status.json").write_text('{"phase": "train"}', encoding="utf-8")
    data = _minimal_data(tmp_path)
    monkeypatch.setattr(train, "prevent_sleep", lambda: _NullGuard())
    monkeypatch.setattr(train, "check_frozen_run", lambda *a, **k: None)
    monkeypatch.setattr(train, "assert_code_under", lambda *a, **k: None)
    monkeypatch.setenv(
        "GEMMALOOP_GPU_LOCK_PATH", str(tmp_path / "gpu.lock")
    )

    def fake_load_model(spec):  # noqa: ANN001 - the model must never load
        del spec
        raise _StopBeforeModel

    monkeypatch.setattr(train, "load_model", fake_load_model)

    try:
        with pytest.raises(_StopBeforeModel):
            train.main(
                _run_main_args(
                    tmp_path, out, data, "--resume", "auto", "--ack-abort", "abc123"
                )
            )
    finally:
        train.release_run_lock()

    # The run reached the model load: resolve_resume_checkpoint did not exit 2.
    assert not (out / ABORT_GUARD_FILE).exists()
    archived = list((out / "attic").glob("ack-abort-*"))
    assert archived
    identity = json.loads(
        (out / "run_identity.json").read_text(encoding="utf-8")
    )
    assert identity.get("git_sha") != "x"
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    acks = [
        item
        for item in run_json["abort_acks"]
        if item["alert_id"] == "abc123"
    ]
    assert len(acks) == 1


def test_fresh_with_ack_is_refused_before_any_side_effect(tmp_path, monkeypatch):
    """R3-M1: a fresh start and an acknowledged resume are contradictory."""

    out = tmp_path / "run"
    _guard(out)
    _checkpoint(out, 1, 100)
    (out / "marker.txt").write_text("untouched", encoding="utf-8")
    data = tmp_path / "missing-data"
    monkeypatch.setattr(train, "prevent_sleep", lambda: _NullGuard())

    with pytest.raises(SystemExit) as exc:
        train.main(
            _run_main_args(
                tmp_path, out, data, "--fresh", "--ack-abort", "abc123"
            )
        )
    assert exc.value.code == 2

    # Nothing was archived, consumed, created or rewritten.
    assert (out / ABORT_GUARD_FILE).is_file()
    assert (out / "checkpoint-1").is_dir()
    assert (out / "marker.txt").read_text(encoding="utf-8") == "untouched"
    assert not (out / "attic").exists()
    assert not (out / "run.json").exists()
