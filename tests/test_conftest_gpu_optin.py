"""GPU tests are opt-in: ``--run-gpu`` is required even when CUDA is visible."""

from __future__ import annotations

from types import SimpleNamespace

import tests.conftest as conftest


class _Item:
    def __init__(self, *, gpu: bool) -> None:
        self.keywords = {"gpu"} if gpu else set()
        self.markers = []

    def add_marker(self, marker) -> None:  # noqa: ANN001
        self.markers.append(marker)


def _config(run_gpu: bool):
    return SimpleNamespace(getoption=lambda name: run_gpu if name == "--run-gpu" else None)


def _skip_names(item: _Item) -> list[str]:
    return [marker.mark.name for marker in item.markers]


def test_gpu_tests_skip_without_run_gpu_without_touching_torch(monkeypatch):
    def _boom():
        raise AssertionError("CPU pytest must not query torch.cuda")

    monkeypatch.setattr(conftest, "_cuda_available", _boom)
    gpu = _Item(gpu=True)
    conftest.pytest_collection_modifyitems(_config(False), [gpu])
    assert _skip_names(gpu) == ["skip"]

    cpu = _Item(gpu=False)
    conftest.pytest_collection_modifyitems(_config(False), [cpu])
    assert _skip_names(cpu) == []


def test_pytest_configure_hides_cuda_before_torch(monkeypatch):
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    conftest.pytest_configure(_config(False))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    conftest.pytest_configure(_config(True))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_gpu_tests_run_with_run_gpu_and_cuda(monkeypatch):
    monkeypatch.setattr(conftest, "_cuda_available", lambda: True)
    gpu = _Item(gpu=True)
    conftest.pytest_collection_modifyitems(_config(True), [gpu])
    assert _skip_names(gpu) == []


def test_run_gpu_still_skips_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(conftest, "_cuda_available", lambda: False)
    gpu = _Item(gpu=True)
    conftest.pytest_collection_modifyitems(_config(True), [gpu])
    assert _skip_names(gpu) == ["skip"]
