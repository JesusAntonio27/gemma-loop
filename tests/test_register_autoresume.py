"""The auto-resume helper prints commands and registers nothing (Task 14b)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "register_autoresume.ps1"


def test_script_prints_windows_and_linux_commands():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "schtasks /Create" in text
    assert "systemctl --user enable" in text
    assert "@reboot" in text
    assert "Nothing below was executed" in text


def test_script_never_executes_a_registration():
    text = SCRIPT.read_text(encoding="utf-8")
    # It may print schtasks/systemctl/Register-ScheduledTask, never start them.
    assert "Start-Process" not in text
    assert "& schtasks" not in text
    assert "& systemctl" not in text
    assert "Invoke-Expression" not in text


def test_printed_task_sets_working_directory_and_resume(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is not available")
    run_dir = tmp_path / "run"
    repo = tmp_path / "repo"
    python = repo / ".venv" / "Scripts" / "python.exe"
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Name",
            "pilot",
            "-RunDir",
            str(run_dir),
            "-Python",
            str(python),
            "-Repo",
            str(repo),
            "-Arguments",
            "scripts/train.py --arm O-H",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "cd /d" in output
    assert str(repo) in output
    assert "--resume auto" in output
    assert "WorkingDirectory=" in output
    assert "-WorkingDirectory" in output
    assert "ExecutionTimeLimit" in output
