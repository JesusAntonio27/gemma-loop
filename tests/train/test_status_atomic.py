"""status.json / heartbeat.json are written atomically (Task 14b)."""

from __future__ import annotations

import json

import pytest

import gemmaloop.train.checkpoint as checkpoint_module
from gemmaloop.train.status import StatusWriter


def test_status_and_heartbeat_are_written_atomically(tmp_path):
    writer = StatusWriter(
        tmp_path / "status.json",
        run="pilot",
        identity={"git_sha": "abc"},
        thresholds={"ram": {"min_available_gib": 2.0}},
    )
    writer.update(phase="train", step=3, tokens_seen=120, tok_per_s=42.5)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))

    assert status["phase"] == "train" and status["step"] == 3
    assert status["identity"]["git_sha"] == "abc"
    assert status["thresholds"]["ram"]["min_available_gib"] == 2.0
    assert status["started_utc"].endswith("+00:00")
    assert status["updated_utc"].endswith("+00:00")
    assert heartbeat["step"] == 3 and heartbeat["tokens_seen"] == 120
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))

    writer.set_exit("guard: ram", 75)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 75
    assert status["exit_reason"] == "guard: ram"


def test_failed_replace_is_logged_and_keeps_the_run_alive(tmp_path, monkeypatch, caplog):
    writer = StatusWriter(tmp_path / "status.json")
    writer.update(step=1)
    original = (tmp_path / "status.json").read_text(encoding="utf-8")

    def boom(src, dst):
        raise RuntimeError("disk full")

    monkeypatch.setattr(checkpoint_module.os, "replace", boom)
    payload = writer.update(step=2)  # must NOT raise

    assert payload["step"] == 2
    assert writer.last_error and "disk full" in writer.last_error
    assert (tmp_path / "status.json").read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".*.tmp"))
    assert "could not write" in caplog.text

    # The next step retries the write and recovers.
    monkeypatch.undo()
    writer.update(step=3)
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["step"] == 3
