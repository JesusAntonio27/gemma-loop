"""The GPU lock must exclude a second process and exit 75 when busy."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gemmaloop.gpulock import LOCK_BUSY_EXIT_CODE, try_acquire

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def test_lock_refuses_a_second_process(tmp_path):
    lock = tmp_path / "gpu.lock"
    holder_code = (
        "import sys, time\n"
        "from gemmaloop.gpulock import acquire\n"
        f"lock = acquire({str(lock)!r})\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code],
        stdout=subprocess.PIPE,
        text=True,
        env=_python_env(),
    )
    try:
        assert holder.stdout is not None
        first = holder.stdout.readline().strip()
        assert first == "locked", first

        # Same-process probe also refuses while the child owns the lock.
        assert try_acquire(lock) is None

        busy_code = (
            "from gemmaloop.gpulock import require_gpu_lock\n"
            f"require_gpu_lock({str(lock)!r})\n"
        )
        second = subprocess.run(
            [sys.executable, "-c", busy_code],
            capture_output=True,
            text=True,
            env=_python_env(),
            timeout=30,
        )
        assert second.returncode == LOCK_BUSY_EXIT_CODE, second.stderr
        assert "GPU lock is held" in second.stderr
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()


def test_lock_is_released_with_the_process(tmp_path):
    lock = tmp_path / "gpu.lock"
    code = (
        "from gemmaloop.gpulock import acquire\n"
        f"held = acquire({str(lock)!r})\n"
        "print('locked', flush=True)\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_python_env(),
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    time.sleep(0.1)
    held = try_acquire(lock)
    assert held is not None
    held.release()
