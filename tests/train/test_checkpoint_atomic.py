"""Atomic checkpoint manifests and stale-staging handling (Task 14b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.train.checkpoint import (
    COMPLETE_FILE,
    CheckpointCorrupt,
    CheckpointIncomplete,
    assert_resume_compatible,
    cleanup_stale_tmp_dirs,
    finalize_checkpoint,
    find_latest_valid_checkpoint,
    list_complete_checkpoints,
    read_complete,
    rotate_checkpoints,
)


def _checkpoint(out, step: int, *, payload: str = "weights") -> None:
    staging = out / f"checkpoint-{step}.tmp"
    staging.mkdir(parents=True)
    (staging / "adapter_model.safetensors").write_text(payload, encoding="utf-8")
    (staging / "token_state.json").write_text(
        json.dumps({"tokens_seen": step * 10}), encoding="utf-8"
    )
    finalize_checkpoint(
        staging,
        out / f"checkpoint-{step}",
        {
            "step": step,
            "tokens_seen": step * 10,
            "run_config_sha256": "cfg",
            "data_manifest_sha256": "data",
            "git_sha": "abc",
            "prereg_sha": None,
            "has_optimizer": False,
            "has_scheduler": False,
            "has_scaler": False,
            "rng_states_saved": True,
        },
    )


def test_complete_marker_lists_every_file_with_sha(tmp_path):
    _checkpoint(tmp_path, 3)
    marker = json.loads(
        (tmp_path / "checkpoint-3" / COMPLETE_FILE).read_text(encoding="utf-8")
    )
    assert marker["complete"] is True
    assert marker["step"] == 3
    assert marker["rng_states_saved"] is True
    listed = {entry["path"]: entry for entry in marker["files"]}
    assert set(listed) == {"adapter_model.safetensors", "token_state.json"}
    assert all(len(entry["sha256"]) == 64 for entry in listed.values())
    assert read_complete(tmp_path / "checkpoint-3")["step"] == 3


def test_partial_tmp_dir_is_never_used(tmp_path):
    _checkpoint(tmp_path, 2)
    leftover = tmp_path / "checkpoint-9.tmp"
    leftover.mkdir()
    (leftover / "adapter_model.safetensors").write_text("half", encoding="utf-8")

    latest = find_latest_valid_checkpoint(tmp_path)
    assert latest is not None and latest.step == 2
    assert not (tmp_path / "checkpoint-9").exists()
    with pytest.raises(CheckpointIncomplete):
        read_complete(leftover)


def test_sha_mismatch_is_rejected_and_newer_corrupt_checkpoint_skipped(tmp_path):
    _checkpoint(tmp_path, 2)
    _checkpoint(tmp_path, 3)
    # Corrupt the newest checkpoint after COMPLETE.json was written.
    (tmp_path / "checkpoint-3" / "adapter_model.safetensors").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(CheckpointCorrupt):
        read_complete(tmp_path / "checkpoint-3")
    latest = find_latest_valid_checkpoint(tmp_path)
    assert latest is not None and latest.step == 2


def test_missing_marker_is_rejected(tmp_path):
    directory = tmp_path / "checkpoint-1"
    directory.mkdir()
    (directory / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    with pytest.raises(CheckpointIncomplete):
        read_complete(directory)
    assert list_complete_checkpoints(tmp_path) == []


def test_cleanup_deletes_stale_tmp_and_logs_paths(tmp_path):
    stale = tmp_path / "checkpoint-7.tmp"
    stale.mkdir()
    final_stage = tmp_path / ".final.tmp"
    final_stage.mkdir()
    keep = tmp_path / "checkpoint-1"
    keep.mkdir()
    removed = cleanup_stale_tmp_dirs(tmp_path)
    assert {path.name for path in removed} == {"checkpoint-7.tmp", ".final.tmp"}
    assert not stale.exists() and not final_stage.exists()
    assert keep.exists()


def test_rotation_keeps_the_last_two_complete_checkpoints(tmp_path):
    for step in (1, 2, 3, 4):
        _checkpoint(tmp_path, step)
    removed = rotate_checkpoints(tmp_path, keep=2)
    assert {path.name for path in removed} == {"checkpoint-1", "checkpoint-2"}
    assert [info.step for info in list_complete_checkpoints(tmp_path)] == [3, 4]


def test_resume_compatibility_refuses_identity_changes():
    metadata = {
        "run_config_sha256": "cfg",
        "data_manifest_sha256": "data",
        "git_sha": "abc",
        "prereg_sha": "prereg",
    }
    assert (
        assert_resume_compatible(
            metadata,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha="prereg",
        )
        == []
    )
    with pytest.raises(Exception, match="data manifest"):
        assert_resume_compatible(
            metadata,
            run_config_sha256="cfg",
            data_manifest_sha256="other",
            git_sha="abc",
            prereg_sha="prereg",
        )
    mismatches = assert_resume_compatible(
        metadata,
        run_config_sha256="new",
        data_manifest_sha256="data",
        git_sha="def",
        prereg_sha="prereg",
        allow_mismatch=True,
    )
    assert any("run configuration" in item for item in mismatches)
    assert any("code git SHA" in item for item in mismatches)


def test_required_resume_state_is_enforced(tmp_path):
    from gemmaloop.train.checkpoint import (
        CheckpointCorrupt,
        assert_resume_state_present,
    )

    metadata = {
        "has_optimizer": True,
        "has_scheduler": True,
        "rng_states_saved": True,
        "loop_sampler": {"seed": 0, "micro_step": 3},
        "data_order": {"epoch": 0, "row_position": 3},
    }
    assert_resume_state_present(metadata)
    for field in ("has_optimizer", "has_scheduler", "rng_states_saved"):
        broken = dict(metadata)
        broken[field] = False
        with pytest.raises(CheckpointCorrupt, match="missing required resume state"):
            assert_resume_state_present(broken)
    for field in ("loop_sampler", "data_order"):
        broken = dict(metadata)
        broken.pop(field)
        with pytest.raises(CheckpointCorrupt, match="missing required resume state"):
            assert_resume_state_present(broken)


def test_replacing_a_same_step_checkpoint_keeps_it_valid(tmp_path):
    _checkpoint(tmp_path, 4, payload="first")
    _checkpoint(tmp_path, 4, payload="second")
    assert read_complete(tmp_path / "checkpoint-4")["step"] == 4
    assert (tmp_path / "checkpoint-4" / "adapter_model.safetensors").read_text(
        encoding="utf-8"
    ) == "second"
    assert not list(tmp_path.glob("*.old.tmp"))


def test_publish_retries_windows_sharing_violation(tmp_path, monkeypatch):
    import gemmaloop.train.checkpoint as checkpoint

    _checkpoint(tmp_path, 1)
    staging = tmp_path / "checkpoint-2.tmp"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    real_replace = checkpoint.os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        if Path(src).name == "checkpoint-2.tmp":
            attempts["count"] += 1
            if attempts["count"] <= 2:
                raise PermissionError(13, "sharing violation", None, 5)
        return real_replace(src, dst)

    monkeypatch.setattr(checkpoint.os, "replace", flaky_replace)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda _s: None)
    finalize_checkpoint(staging, tmp_path / "checkpoint-2", {"step": 2})
    assert (tmp_path / "checkpoint-2" / COMPLETE_FILE).is_file()
    assert attempts["count"] == 3


def test_publish_failure_is_retryable_and_keeps_the_old_checkpoint(tmp_path, monkeypatch):
    import gemmaloop.train.checkpoint as checkpoint

    _checkpoint(tmp_path, 4, payload="old")
    staging = tmp_path / "checkpoint-4.tmp"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_text("new", encoding="utf-8")
    real_replace = checkpoint.os.replace

    def blocked_replace(src, dst):
        if Path(src).name == "checkpoint-4.tmp":
            raise PermissionError(13, "held open", None, 5)
        return real_replace(src, dst)

    monkeypatch.setattr(checkpoint.os, "replace", blocked_replace)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda _s: None)
    with pytest.raises(checkpoint.RetryableIOError):
        finalize_checkpoint(staging, tmp_path / "checkpoint-4", {"step": 4})
    # The old checkpoint was restored, never left as a backup/partial.
    assert read_complete(tmp_path / "checkpoint-4")["step"] == 4
    assert (tmp_path / "checkpoint-4" / "adapter_model.safetensors").read_text(
        encoding="utf-8"
    ) == "old"
    assert not (tmp_path / "checkpoint-4.old.tmp").exists()


def test_fsync_is_actually_called_on_publish(tmp_path, monkeypatch):
    import gemmaloop.train.checkpoint as checkpoint

    calls: list[int] = []
    real_fsync = checkpoint.os.fsync

    def recording_fsync(fd):
        calls.append(int(fd))
        return real_fsync(fd)

    monkeypatch.setattr(checkpoint.os, "fsync", recording_fsync)
    _checkpoint(tmp_path, 1)
    # Two checkpoint files are fsynced by fsync_file, plus COMPLETE.json by the
    # atomic writer.  A no-op fsync_file mutant leaves only the latter.
    assert len(calls) >= 3


def test_archive_run_state_moves_the_old_run_to_attic(tmp_path):
    from gemmaloop.train.checkpoint import archive_run_state

    _checkpoint(tmp_path, 1)
    (tmp_path / "run.json").write_text('{"status": "complete"}', encoding="utf-8")
    (tmp_path / "status.json").write_text('{"step": 1}', encoding="utf-8")
    destination = archive_run_state(tmp_path, reason="fresh")
    assert destination.is_dir()
    assert not (tmp_path / "checkpoint-1").exists()
    assert not (tmp_path / "run.json").exists()
    assert (destination / "checkpoint-1" / COMPLETE_FILE).is_file()
    assert (destination / "run.json").is_file()


def test_archive_newer_checkpoints_keeps_the_rewound_branch(tmp_path):
    from gemmaloop.train.checkpoint import archive_newer_checkpoints

    for step in (1, 2, 3):
        _checkpoint(tmp_path, step)
    (tmp_path / "run.json").write_text('{"status": "complete"}', encoding="utf-8")
    destination = archive_newer_checkpoints(tmp_path, resume_step=1)
    assert (tmp_path / "checkpoint-1").is_dir()
    assert not (tmp_path / "checkpoint-2").exists()
    assert not (tmp_path / "checkpoint-3").exists()
    assert (destination / "checkpoint-2").is_dir()
    assert (destination / "run.json").is_file()


def test_archive_newer_checkpoints_also_archives_monitor_files(tmp_path):
    from gemmaloop.train.checkpoint import archive_newer_checkpoints

    for step in (1, 2):
        _checkpoint(tmp_path, step)
    (tmp_path / "run.json").write_text('{"status": "complete"}', encoding="utf-8")
    (tmp_path / "heal_monitor_state.json").write_text(
        '{"last_probe_tokens": 3000000}', encoding="utf-8"
    )
    (tmp_path / "alerts.json").write_text('{"alerts": []}', encoding="utf-8")
    (tmp_path / "heal_metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (tmp_path / "heal_probes.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    texts = tmp_path / "heal_probe_texts"
    texts.mkdir()
    (texts / "sample.txt").write_text("old", encoding="utf-8")
    (tmp_path / "heal_monitor_abort.json").write_text("{}", encoding="utf-8")

    destination = archive_newer_checkpoints(tmp_path, resume_step=1)
    for name in (
        "heal_monitor_state.json",
        "alerts.json",
        "heal_metrics.jsonl",
        "heal_probes.jsonl",
        "heal_probe_texts",
        "heal_monitor_abort.json",
    ):
        assert (destination / name).exists(), name
        assert not (tmp_path / name).exists(), name
