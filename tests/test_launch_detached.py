"""PowerShell 5.1 detached launcher: dry-run parsing and GPU-busy refusal."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_detached.ps1"
POWERSHELL = "powershell"


def _skip_unless_windows():
    if os.name != "nt":
        pytest.skip("the detached launcher targets Windows PowerShell 5.1")


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # The launcher honours GEMMALOOP_GPU_LOCK_PATH; the tests exercise the
    # <artifacts>/gpu.lock default.
    env.pop("GEMMALOOP_GPU_LOCK_PATH", None)
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_dry_run_parses_the_command(tmp_path):
    _skip_unless_windows()
    artifacts = tmp_path / "artifacts"
    result = _run_launcher(
        "-Name",
        "smoke",
        "-Python",
        "python",
        "-Arguments",
        "-m scripts.trainfree_campaign --stage A",
        "-Artifacts",
        str(artifacts),
        "-DryRun",
        "-SkipChecks",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["name"] == "smoke"
    assert payload["arguments"] == "-m scripts.trainfree_campaign --stage A"
    assert payload["log"].endswith(".log")
    assert "runs\\detached" in payload["log"]
    assert payload["pid_file"].endswith(".pid")
    assert "-m scripts.trainfree_campaign --stage A" in payload["command"]
    assert "2>&1" in payload["command"]


def test_refuses_when_gpu_lock_is_held(tmp_path):
    _skip_unless_windows()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    lock = artifacts / "gpu.lock"
    code = (
        "import time\n"
        "from gemmaloop.gpulock import acquire\n"
        f"lock = acquire({str(lock)!r})\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        result = _run_launcher(
            "-Name",
            "should-refuse",
            "-Python",
            "python",
            "-Arguments",
            "-c pass",
            "-Artifacts",
            str(artifacts),
            "-DryRun",
        )
        assert result.returncode == 75, (result.stdout, result.stderr)
        assert "GPU lock is held" in result.stderr
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()


DESKTOP_GPU_APPS = r"""13004, C:\Windows\System32\ShellHost.exe
13212, C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\CrossDeviceResume.exe
14460, C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe
12844, C:\Windows\explorer.exe
19488, C:\Program Files\LGHUB\system_tray\lghub_system_tray.exe
19560, C:\Program Files\WindowsApps\Claude_4.16336.0.0_x64__pzs8sxrjxfjjc\app\claude.exe
29632, C:\Program Files\WindowsApps\Claude_4.16336.0.0_x64__pzs8sxrjxfjjc\app\claude.exe
4124, C:\Program Files (x86)\Microsoft\EdgeWebView\Application\156.0.4314.58\msedgewebview2.exe
5484, C:\Program Files\WindowsApps\Microsoft.ScreenSketch_11.2608.41.0_x64__8wekyb3d8bbwe\SnippingTool\SnippingTool.exe
"""


def test_desktop_gpu_apps_are_ignored_and_defaults_are_sane(tmp_path):
    _skip_unless_windows()
    fake = tmp_path / "nvidia-smi.txt"
    fake.write_text(DESKTOP_GPU_APPS, encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    result = _run_launcher(
        "-Name",
        "desktop-apps",
        "-Arguments",
        "-m scripts.trainfree_campaign --stage A",
        "-Artifacts",
        str(artifacts),
        "-NvidiaSmiOutputFile",
        str(fake),
        "-DryRun",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["ignored_desktop_gpu_processes"]) == 9
    assert any("explorer.exe" in line for line in payload["ignored_desktop_gpu_processes"])
    assert payload["python"].endswith(r".venv\Scripts\python.exe")
    assert Path(payload["working_directory"]).resolve() == REPO_ROOT.resolve()
    assert r"C:\path\to" not in payload["command"]


def test_python_compute_process_still_refuses(tmp_path):
    _skip_unless_windows()
    fake = tmp_path / "nvidia-smi.txt"
    fake.write_text(
        r"12999, C:\Windows\explorer.exe" + "\n"
        + r"13000, C:\Python312\python.exe" + "\n"
        + r"13001, C:\tools\python3.12.exe" + "\n",
        encoding="utf-8",
    )
    result = _run_launcher(
        "-Name",
        "python-app",
        "-Arguments",
        "-m scripts.trainfree_campaign --stage A",
        "-Artifacts",
        str(tmp_path / "artifacts"),
        "-NvidiaSmiOutputFile",
        str(fake),
        "-DryRun",
    )
    assert result.returncode == 75
    assert "python compute process" in result.stderr
    assert "explorer.exe" not in result.stderr
