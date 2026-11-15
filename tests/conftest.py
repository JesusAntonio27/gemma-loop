"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency
        return False
    return torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _isolate_gpu_lock(tmp_path, monkeypatch):
    """Keep script lock/heartbeat files inside the test's temporary directory."""
    monkeypatch.setenv("GEMMALOOP_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("GEMMALOOP_HEARTBEAT_PATH", str(tmp_path / "heartbeat.json"))
    yield
    from gemmaloop.gpulock import release_held_lock

    release_held_lock()


@pytest.fixture(autouse=True)
def _offpeak_deepseek_clock(monkeypatch):
    """Default DeepSeek clocks to a fixed off-peak instant in unit tests.

    Production code still uses the real UTC clock; tests that need another
    instant inject ``now_fn``/``sleep_fn`` explicitly.  This prevents the
    mandatory off-peak gate from sleeping for hours when the suite happens to
    run inside a peak window.  ``HiddenTestWriter`` looks the symbol up at call
    time, so it is covered too.
    """

    from datetime import datetime, timezone

    try:
        from gemmaloop import deepseek_money
    except Exception:  # pragma: no cover - package always present
        yield
        return
    monkeypatch.setattr(
        deepseek_money,
        "utcnow",
        lambda: datetime(2026, 11, 2, 5, 0, tzinfo=timezone.utc),
    )
    yield


@pytest.fixture(autouse=True)
def _isolate_deepseek_money_state(tmp_path, monkeypatch):
    """Keep the shared DeepSeek ledger/lock and run guard inside ``tmp_path``."""

    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", str(tmp_path / "artifacts"))
    try:
        from gemmaloop.judge import client as judge_client

        judge_client.reset_default_judge_guard()
    except Exception:  # pragma: no cover
        judge_client = None
    yield
    from gemmaloop.stage0.spend import release_held_spend_lock

    release_held_spend_lock()
    if judge_client is not None:
        judge_client.reset_default_judge_guard()



def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run gpu-marked tests (they load real E2B weights and need a GPU)",
    )


def pytest_configure(config):
    """Hide CUDA before torch is imported unless GPU tests were requested.

    Torch 2.11 on Windows also reports ``is_available() == True`` with
    ``CUDA_VISIBLE_DEVICES=""`` while ``device_count() == 0``; ``-1`` is the
    documented CPU-only value.  This runs in ``pytest_configure`` (before test
    modules import torch), and the CPU path never calls ``torch.cuda``.
    """

    if not config.getoption("--run-gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_collection_modifyitems(config, items):
    run_gpu = bool(config.getoption("--run-gpu"))
    if not run_gpu:
        reason = "GPU tests are opt-in: pass --run-gpu"
    elif not _cuda_available():
        reason = "CUDA GPU not available"
    else:
        return
    skip_gpu = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
