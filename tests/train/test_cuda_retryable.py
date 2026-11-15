"""CUDA/OOM and checkpoint-I/O failures map to retryable exit 75 (Task 14b)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import scripts.train as train_script
from gemmaloop.train.checkpoint import RetryableIOError


class _Status:
    def __init__(self):
        self.reason = None
        self.code = None

    def set_exit(self, reason, code):
        self.reason = reason
        self.code = code


class _Trainer:
    def __init__(self):
        self.state = SimpleNamespace(global_step=4)
        self.tokens_seen = 400
        self._last_checkpoint_path = None
        self.model = object()
        self.saved = 0

    def _save_checkpoint(self, model, trial):  # noqa: ANN001
        del model, trial
        self.saved += 1
        self._last_checkpoint_path = SimpleNamespace(name="checkpoint-4")


@pytest.mark.parametrize(
    "exc",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUDA error: unspecified launch failure"),
        RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"),
        RuntimeError("cuDNN error"),
    ],
)
def test_gpu_failures_are_retryable(monkeypatch, exc):
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: True)
    status = _Status()
    trainer = _Trainer()
    code = train_script.handle_retryable_training_exception(exc, trainer, status)
    assert code == 75
    assert status.code == 75 and "gpu-error" in status.reason
    assert trainer.saved == 0  # dead context: no emergency save


def test_never_saves_a_checkpoint_after_an_exception(monkeypatch):
    # N1: even a live context must not save mid-step counters/parameters.
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: True)
    status = _Status()
    trainer = _Trainer()
    code = train_script.handle_retryable_training_exception(
        torch.OutOfMemoryError("CUDA out of memory"), trainer, status
    )
    assert code == 75
    assert trainer.saved == 0


def test_checkpoint_io_failure_is_retryable():
    status = _Status()
    code = train_script.handle_retryable_training_exception(
        RetryableIOError("publish checkpoint kept failing"), _Trainer(), status
    )
    assert code == 75
    assert "retryable IO" in status.reason


def test_unrelated_exceptions_stay_fatal():
    assert (
        train_script.handle_retryable_training_exception(
            ValueError("bug"), _Trainer(), _Status()
        )
        is None
    )
