"""Frozen-run identity checks (Task 14b).

A multi-day run must not silently continue on different code, data or
pre-registration.  ``scripts/train.py`` records the git SHA, dirty flag, data
manifest SHA and prereg SHA in the run directory; these helpers produce those
values and enforce ``--allow-dirty`` / ``--require-tag``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmaloop.train.checkpoint import atomic_write_json, sha256_file, sha256_json

IDENTITY_FILE = "run_identity.json"

__all__ = [
    "FrozenRunError",
    "GitState",
    "IDENTITY_FILE",
    "assert_code_under",
    "check_frozen_run",
    "git_state",
    "prereg_sha256",
    "run_config_sha256",
    "run_identity_hash",
    "tag_commit",
    "write_run_identity",
]


class FrozenRunError(RuntimeError):
    """The worktree/tag does not satisfy the frozen-run contract."""


@dataclass(frozen=True)
class GitState:
    sha: str
    dirty: bool
    branch: str | None


def _git(repo: str | Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FrozenRunError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def git_state(repo: str | Path = ".") -> GitState:
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    try:
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except FrozenRunError:  # pragma: no cover - detached HEAD names itself
        branch = None
    return GitState(sha=sha, dirty=bool(status), branch=branch)


def tag_commit(repo: str | Path, tag: str) -> str:
    try:
        return _git(repo, "rev-parse", f"{tag}^{{commit}}")
    except FrozenRunError as exc:
        raise FrozenRunError(f"tag {tag!r} does not exist in {repo}") from exc


def check_frozen_run(
    repo: str | Path = ".",
    *,
    allow_dirty: bool = False,
    require_tag: str | None = None,
    prereg: str | Path | None = None,
) -> GitState:
    """Refuse a dirty worktree, a wrong tag or a prereg run without a tag."""
    if allow_dirty and require_tag:
        raise FrozenRunError(
            "--require-tag is incompatible with --allow-dirty; a tagged run "
            "must start from the tagged, clean commit"
        )
    if prereg is not None and not require_tag:
        raise FrozenRunError(
            "--prereg requires --require-tag so results are tied to a frozen "
            "pre-registration commit"
        )
    state = git_state(repo)
    if state.dirty and not allow_dirty:
        raise FrozenRunError(
            "the worktree is dirty; long runs must start from a clean, tagged "
            "commit (use --allow-dirty only for a smoke run)"
        )
    if require_tag:
        expected = tag_commit(repo, require_tag)
        if state.sha != expected:
            raise FrozenRunError(
                f"HEAD is {state.sha} but --require-tag {require_tag!r} is "
                f"{expected}; check out the tagged commit"
            )
    return state


def assert_code_under(
    repo_root: str | Path, *, module_file: str | Path | None = None
) -> Path:
    """Refuse when the imported ``gemmaloop`` package is not this worktree's src."""
    root = Path(repo_root).resolve()
    if module_file is None:
        import gemmaloop

        module_file = gemmaloop.__file__
    if module_file is None:  # pragma: no cover - namespace package
        raise FrozenRunError("cannot locate the imported gemmaloop package")
    imported = Path(module_file).resolve()
    source_root = (root / "src").resolve()
    try:
        imported.relative_to(source_root)
    except ValueError as exc:
        raise FrozenRunError(
            f"the imported gemmaloop package is {imported}, outside this "
            f"worktree's {source_root}; refusing to train code that was not "
            "the checked tree"
        ) from exc
    return imported


def prereg_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise FrozenRunError(f"pre-registration file not found: {target}")
    return sha256_file(target)


def run_config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(config)


def run_identity_hash(
    *,
    run_config_sha256: str,
    data_manifest_sha256: str,
    git_sha: str,
    seed: int,
) -> str:
    """The run identity used by DONE/COMPLETE and by ``--resume auto``."""
    return sha256_json(
        {
            "run_config_sha256": run_config_sha256,
            "data_manifest_sha256": data_manifest_sha256,
            "git_sha": git_sha,
            "seed": int(seed),
        }
    )


def write_run_identity(
    out_dir: str | Path, identity: dict[str, Any]
) -> Path:
    """Atomically write ``<out>/run_identity.json`` and return its path."""
    path = Path(out_dir) / IDENTITY_FILE
    atomic_write_json(path, identity)
    return path
