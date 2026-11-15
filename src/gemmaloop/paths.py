"""Filesystem locations for Gemma-Loop artifacts.

Heavy artifacts (models, checkpoints, data and raw results) live outside the
repository. The location defaults to ``~/ML/gemmaloop-artifacts`` and can be
overridden with the ``GEMMALOOP_ARTIFACTS`` environment variable. Paths inside
OneDrive are rejected: syncing multi-gigabyte checkpoints blocks files and
stalls training.
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACTS_ENV_VAR = "GEMMALOOP_ARTIFACTS"
_DEFAULT_ARTIFACTS = Path.home() / "ML" / "gemmaloop-artifacts"


def artifacts_dir() -> Path:
    """Return the root directory for heavy artifacts."""
    raw = os.environ.get(ARTIFACTS_ENV_VAR)
    root = Path(raw).expanduser() if raw else _DEFAULT_ARTIFACTS
    if "onedrive" in str(root).lower():
        raise ValueError(
            f"Refusing to use {root}: artifacts must not live inside OneDrive. "
            f"Set {ARTIFACTS_ENV_VAR} to a local directory."
        )
    return root


def run_dir(name: str) -> Path:
    """Return (and create) the directory of a named training or evaluation run."""
    path = artifacts_dir() / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Return (and create) the directory holding datasets."""
    path = artifacts_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    """Return (and create) the directory holding evaluation results."""
    path = artifacts_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
