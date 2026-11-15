"""Provenance guards in scripts/train.py (pinned template + manifest)."""

from __future__ import annotations

import json

import pytest

import scripts.train as train_script
from gemmaloop.model.snapshot import (
    E2B_REVISION,
    MODEL_REVISION,
    chat_template_sha256,
    model_revision,
)


class _Tokenizer:
    def __init__(self, template: str | None = "template-v1") -> None:
        self.chat_template = template


def test_model_revision_is_the_pinned_snapshot():
    pinned = "4adf15748de9e1a0097b6274cee9c15836c41c37"
    assert MODEL_REVISION == E2B_REVISION == pinned
    assert model_revision("e2b") == pinned
    assert model_revision("E2B") == pinned


def test_e4b_and_12b_qat_revisions_are_pinned():
    assert model_revision("e4b") == "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    assert model_revision("12b_qat") == "b6ed86275a6a5735884e208bfed95b445a684ca2"
    with pytest.raises(ValueError, match="unknown model size"):
        model_revision("13b")


def test_verify_chat_template_accepts_the_manifest_hash():
    tokenizer = _Tokenizer()
    expected = chat_template_sha256(tokenizer)
    assert train_script.verify_chat_template(expected, tokenizer) == expected


def test_verify_chat_template_refuses_a_different_template(capsys):
    with pytest.raises(SystemExit) as exc:
        train_script.verify_chat_template("0" * 64, _Tokenizer())
    assert exc.value.code == 2
    assert "chat template changed" in capsys.readouterr().err


def test_verify_chat_template_accepts_a_legacy_manifest():
    assert train_script.verify_chat_template("", _Tokenizer()) == chat_template_sha256(
        _Tokenizer()
    )


def test_read_data_manifest_details_reads_the_template_hash(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": "deadbeef",
                "target_tokens": 10,
                "chat_template_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    details = train_script.read_data_manifest_details(data)
    assert details.chat_template_sha256 == "a" * 64


def test_logging_quiets_accelerate():
    import logging

    train_script.configure_logging()
    assert logging.getLogger("accelerate").level == logging.WARNING


def test_assert_no_silent_repeat():
    from types import SimpleNamespace

    trainer = SimpleNamespace(micro_batches_seen=5)
    train_script.assert_no_silent_repeat(trainer, 5, allow_repeat=False)
    with pytest.raises(RuntimeError, match="without --allow-repeat"):
        train_script.assert_no_silent_repeat(
            SimpleNamespace(micro_batches_seen=5), 4, allow_repeat=False
        )
    train_script.assert_no_silent_repeat(trainer, 4, allow_repeat=True)


def test_check_oh_ext_budget(capsys):
    train_script.check_oh_ext_budget("O-H-ext", 66_150_274, 66_150_274)
    with pytest.raises(SystemExit) as exc:
        train_script.check_oh_ext_budget("O-H-ext", 66_150_274, 66_298_701)
    assert exc.value.code == 2
    assert "budget mismatch" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        train_script.check_oh_ext_budget("L", 1, 1)
    assert exc.value.code == 2
    assert "only valid for the O-H-ext" in capsys.readouterr().err
    train_script.check_oh_ext_budget("L", None, 1)


def test_prereg_length_default_and_manifest_guard():
    from types import SimpleNamespace

    prereg = SimpleNamespace(
        training=SimpleNamespace(max_length=8192, length_mode="filter")
    )
    manifest = train_script.DataManifest(
        sha256="deadbeef",
        target_tokens=None,
        max_length=8192,
        trained_tokens=None,
    )
    args = SimpleNamespace(max_length=None)
    assert train_script.resolve_run_length(
        args, prereg, manifest, "e2b", "O-H"
    ) == (8192, "filter")

    args.max_length = 1024
    with pytest.raises(SystemExit) as exc:
        train_script.resolve_run_length(args, prereg, manifest, "e2b", "O-H")
    assert exc.value.code == 2

    legacy = train_script.DataManifest(
        sha256="deadbeef",
        target_tokens=None,
        max_length=None,
        trained_tokens=None,
    )
    assert train_script.resolve_run_length(
        SimpleNamespace(max_length=2048), None, legacy, "e2b", "O-H"
    ) == (2048, None)


def test_deterministic_configuration_errors_are_terminal_exit_2():
    """M10: these must not be retried by the orchestrator as exit 1."""

    with pytest.raises(SystemExit) as exc:
        train_script.resolve_num_epochs(50, 100, max_length=1024, allow_repeat=False)
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        train_script.check_oh_ext_budget("O-H-ext", 1, 2)
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        train_script.verify_chat_template("0" * 64, _Tokenizer())
    assert exc.value.code == 2

    from types import SimpleNamespace

    prereg = SimpleNamespace(
        training=SimpleNamespace(max_length=8192, length_mode="filter")
    )
    manifest = train_script.DataManifest(
        sha256="deadbeef", target_tokens=None, max_length=8192, trained_tokens=None
    )
    with pytest.raises(SystemExit) as exc:
        train_script.resolve_run_length(
            SimpleNamespace(max_length=1024), prereg, manifest, "e2b", "O-H"
        )
    assert exc.value.code == 2


def test_monitor_teacher_transient_io_errors_are_retryable():
    """R2-M4: filesystem contention exits 75; content errors stay terminal."""

    from gemmaloop.data.teacher_topk import TeacherManifestMismatch
    from gemmaloop.train.checkpoint import RetryableIOError

    assert train_script.monitor_teacher_load_is_retryable(OSError("locked"))
    assert train_script.monitor_teacher_load_is_retryable(
        PermissionError("sharing violation")
    )
    assert train_script.monitor_teacher_load_is_retryable(
        RetryableIOError("kept failing")
    )

    wrapped = TeacherManifestMismatch("cannot read teacher manifest")
    wrapped.__cause__ = OSError("OneDrive lock")
    assert train_script.monitor_teacher_load_is_retryable(wrapped)

    assert not train_script.monitor_teacher_load_is_retryable(
        TeacherManifestMismatch("teacher manifest not found")
    )
    assert not train_script.monitor_teacher_load_is_retryable(
        ValueError("bad content")
    )
