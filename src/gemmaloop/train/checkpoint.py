"""Atomic, verifiable training checkpoints (Task 14b).

A checkpoint is written into ``<out>/checkpoint-<step>.tmp`` and only becomes
visible as ``<out>/checkpoint-<step>`` after every file has been flushed and
``COMPLETE.json`` (written last) lists the SHA-256 of every checkpoint file.
A directory without a valid ``COMPLETE.json`` -- or whose file hashes no longer
match -- is never treated as resumable.

The functions here are deliberately independent of ``transformers`` so they can
be unit-tested on a plain filesystem.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "checkpoint-"
COMPLETE_FILE = "COMPLETE.json"
DONE_FILE = "DONE.json"
ABORT_GUARD_FILE = "heal_monitor_abort.json"
TMP_SUFFIX = ".tmp"
ATTIC_DIR = "attic"
DEFAULT_KEEP_CHECKPOINTS = 2
FS_RETRY_ATTEMPTS = 10
FS_RETRY_INITIAL = 0.1
FS_RETRY_MAX = 5.0

_SHARING_WINERRORS = {5, 32, 33}
_SHARING_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EBUSY, errno.ENOTEMPTY}

__all__ = [
    "ABORT_GUARD_FILE",
    "AbortResumeDecision",
    "CHECKPOINT_PREFIX",
    "COMPLETE_FILE",
    "CheckpointCorrupt",
    "CheckpointIncomplete",
    "ATTIC_DIR",
    "CheckpointInfo",
    "DONE_FILE",
    "ResumeMismatch",
    "RetryableIOError",
    "archive_newer_checkpoints",
    "archive_run_state",
    "assert_resume_compatible",
    "assert_resume_state_present",
    "atomic_write_json",
    "atomic_write_text",
    "checkpoint_invalid_reasons",
    "cleanup_stale_tmp_dirs",
    "decide_abort_resume",
    "finalize_checkpoint",
    "find_latest_valid_checkpoint",
    "fsync_directory",
    "fsync_file",
    "has_run_state",
    "list_checkpoint_dirs",
    "list_complete_checkpoints",
    "read_abort_guard",
    "read_complete",
    "read_done",
    "retry_fs",
    "retry_rmtree",
    "rotate_checkpoints",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_checkpoint",
    "write_done",
]


class CheckpointIncomplete(RuntimeError):
    """The checkpoint has no ``COMPLETE.json``."""


class CheckpointCorrupt(RuntimeError):
    """``COMPLETE.json`` is invalid or a listed file hash does not match."""


class ResumeMismatch(RuntimeError):
    """A checkpoint was produced by a different run configuration."""


class RetryableIOError(RuntimeError):
    """A filesystem operation kept failing (Windows sharing violation, etc.)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a JSON-able mapping (sorted keys, no whitespace)."""
    encoded = json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def is_retryable_io_error(exc: BaseException) -> bool:
    """Windows sharing violations and transient access errors are retryable."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror in _SHARING_WINERRORS:
            return True
        if getattr(exc, "errno", None) in _SHARING_ERRNOS:
            return True
    return False


def retry_fs(
    operation: Callable[..., Any],
    *args: Any,
    attempts: int = FS_RETRY_ATTEMPTS,
    initial: float = FS_RETRY_INITIAL,
    maximum: float = FS_RETRY_MAX,
    what: str = "filesystem operation",
    **kwargs: Any,
) -> Any:
    """Run ``operation`` with exponential backoff on Windows sharing violations.

    Ten attempts with 0.1 s -> 5 s backoff total about 30 s.  The final failure
    is raised as :class:`RetryableIOError` so ``train.py`` can exit 75.
    """

    delay = float(initial)
    last: BaseException | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            return operation(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - filtered below
            if not is_retryable_io_error(exc):
                raise
            last = exc
            if attempt + 1 >= attempts:
                break
            LOGGER.warning(
                "%s failed (%s); retrying in %.2fs", what, exc, delay
            )
            time.sleep(delay)
            delay = min(float(maximum), delay * 2.0)
    raise RetryableIOError(f"{what} kept failing: {last}") from last


def retry_rmtree(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return

    def _rmtree() -> None:
        shutil.rmtree(path)

    retry_fs(_rmtree, what=f"remove {path}")


def retry_replace(src: str | Path, dst: str | Path, *, what: str | None = None) -> None:
    retry_fs(
        os.replace,
        str(src),
        str(dst),
        what=what or f"rename {Path(src).name}",
    )


def fsync_file(path: str | Path) -> None:
    """Best-effort ``fsync`` of one file (never raises)."""
    try:
        with Path(path).open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        try:
            with Path(path).open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass


def fsync_directory(path: str | Path) -> None:
    """Best-effort directory ``fsync``; Windows does not allow opening dirs."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` through a sibling temporary and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        retry_replace(temporary, path, what=f"publish {path.name}")
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return None
    suffix = name[len(CHECKPOINT_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def read_complete(directory: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Read and (by default) verify a checkpoint's ``COMPLETE.json``.

    Raises :class:`CheckpointIncomplete` when the marker is missing and
    :class:`CheckpointCorrupt` when it cannot be parsed or a listed file is
    absent / has a different SHA-256.
    """

    directory = Path(directory)
    marker = directory / COMPLETE_FILE
    if not marker.is_file():
        raise CheckpointIncomplete(f"{directory} has no {COMPLETE_FILE}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorrupt(f"cannot read {marker}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointCorrupt(f"{marker} is not a JSON object")
    if not payload.get("complete"):
        raise CheckpointCorrupt(f"{marker} is not marked complete")
    if verify:
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise CheckpointCorrupt(f"{marker} has no file list")
        for entry in files:
            if not isinstance(entry, dict):
                raise CheckpointCorrupt(f"{marker} has a malformed file entry")
            name = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(name, str) or not isinstance(expected, str):
                raise CheckpointCorrupt(f"{marker} has a malformed file entry")
            path = directory / name
            if not path.is_file():
                raise CheckpointCorrupt(f"{path} listed in {marker} is missing")
            actual = sha256_file(path)
            if actual != expected:
                raise CheckpointCorrupt(
                    f"{path} sha256 {actual} does not match {expected} in {marker}"
                )
    return payload


def validate_checkpoint(directory: str | Path) -> dict[str, Any]:
    return read_complete(directory, verify=True)


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    metadata: Mapping[str, Any]


def list_complete_checkpoints(
    out_dir: str | Path, *, verify: bool = False
) -> list[CheckpointInfo]:
    """Return checkpoint directories that have a (by default unchecked) marker."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found: list[CheckpointInfo] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_step(child)
        if step is None:
            continue
        if not (child / COMPLETE_FILE).is_file():
            continue
        if verify:
            try:
                metadata = read_complete(child, verify=True)
            except (CheckpointIncomplete, CheckpointCorrupt) as exc:
                LOGGER.warning("ignoring checkpoint %s: %s", child, exc)
                continue
        else:
            try:
                metadata = json.loads(
                    (child / COMPLETE_FILE).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("ignoring checkpoint %s: %s", child, exc)
                continue
        found.append(CheckpointInfo(path=child, step=step, metadata=metadata))
    return sorted(found, key=lambda info: info.step)


def find_latest_valid_checkpoint(
    out_dir: str | Path, *, identity_hash: str | None = None
) -> CheckpointInfo | None:
    """Newest verified checkpoint, optionally only for ``identity_hash``."""
    for info in reversed(list_complete_checkpoints(out_dir, verify=False)):
        try:
            metadata = read_complete(info.path, verify=True)
        except (CheckpointIncomplete, CheckpointCorrupt) as exc:
            LOGGER.warning("skipping invalid checkpoint %s: %s", info.path, exc)
            continue
        if identity_hash is not None and metadata.get("run_identity_hash") != identity_hash:
            LOGGER.warning(
                "skipping checkpoint %s: run identity %s != current %s",
                info.path,
                metadata.get("run_identity_hash"),
                identity_hash,
            )
            continue
        return CheckpointInfo(path=info.path, step=info.step, metadata=metadata)
    return None


@dataclass(frozen=True)
class AbortResumeDecision:
    """The single resume rule shared by ``train.py`` and the orchestrator.

    ``action`` is one of:

    * ``none``: no persisted heal-monitor abort record;
    * ``refuse``: an abort record exists and the launch did not acknowledge it;
    * ``resume``: acknowledged and a COMPLETE checkpoint exists;
    * ``restart``: acknowledged but there is no COMPLETE checkpoint, so the
      partial run directory must be archived and training starts from zero.
    """

    action: str
    guard: Mapping[str, Any] | None = None
    checkpoint: CheckpointInfo | None = None

    @property
    def alert_id(self) -> str | None:
        if not self.guard:
            return None
        value = self.guard.get("alert_id")
        return None if value is None else str(value)


def read_abort_guard(out_dir: str | Path) -> dict[str, Any] | None:
    """Read ``heal_monitor_abort.json``; malformed/absent files mean no guard."""

    path = Path(out_dir) / ABORT_GUARD_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("alert_id"):
        return None
    return payload


def decide_abort_resume(
    out_dir: str | Path,
    ack_alert_id: str | None,
    *,
    identity_hash: str | None = None,
) -> AbortResumeDecision:
    """Decide whether a launch may resume, must restart, or is refused.

    This is the only implementation of the registered resume-over-abort rule.
    ``train.py`` and ``scripts/orchestrate.py`` both call it, so the prereg
    text and the executed behaviour cannot drift apart.
    """

    guard = read_abort_guard(out_dir)
    if guard is None:
        return AbortResumeDecision("none")
    wanted = str(guard.get("alert_id"))
    if ack_alert_id is None or str(ack_alert_id) != wanted:
        return AbortResumeDecision("refuse", guard)
    info = find_latest_valid_checkpoint(out_dir, identity_hash=identity_hash)
    if info is None:
        return AbortResumeDecision("restart", guard)
    return AbortResumeDecision("resume", guard, info)


def checkpoint_invalid_reasons(out_dir: str | Path) -> list[str]:
    """Human-readable reasons why no checkpoint in ``out_dir`` can be resumed."""
    reasons: list[str] = []
    for info in list_complete_checkpoints(out_dir, verify=False):
        try:
            read_complete(info.path, verify=True)
        except (CheckpointIncomplete, CheckpointCorrupt) as exc:
            reasons.append(f"{info.path.name}: {exc}")
    for child in list_checkpoint_dirs(out_dir):
        if _checkpoint_step(child) is None:
            reasons.append(f"{child.name}: not a published checkpoint")
        elif not (child / COMPLETE_FILE).is_file():
            reasons.append(f"{child.name}: no {COMPLETE_FILE}")
    return reasons


_RUN_ROOT_FILES = (
    "DONE.json",
    "run.json",
    "status.json",
    "heartbeat.json",
    "run_identity.json",
    "metrics.json",
    "trainer_state.json",
    "token_state.json",
    "adapters_chain.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "loop_config.json",
    "loop_state.safetensors",
    # Heal-monitor state.  A fresh/rerun/rewind must start a new monitor
    # trajectory: leaving these behind hides the first probe (the old
    # last_probe_tokens can exceed the whole new run) and mixes old alerts and
    # metric rows into the new run.
    "heal_monitor_state.json",
    "alerts.json",
    "heal_metrics.jsonl",
    "heal_probes.jsonl",
    "heal_probe_texts",
    "heal_monitor_abort.json",
)


def list_checkpoint_dirs(out_dir: str | Path) -> list[Path]:
    """Every ``checkpoint-*`` directory (valid, torn or staging), oldest first."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found = [
        child
        for child in out_dir.iterdir()
        if child.is_dir() and child.name.startswith(CHECKPOINT_PREFIX)
    ]
    return sorted(found, key=lambda path: (_checkpoint_step(path) or -1, path.name))


def has_run_state(out_dir: str | Path) -> bool:
    """True when the run directory already contains any run/checkpoint artefact."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return False
    if list_checkpoint_dirs(out_dir):
        return True
    for name in ("run_identity.json", "DONE.json", "run.json", "status.json"):
        if (out_dir / name).is_file():
            return True
    return False


def read_done(out_dir: str | Path) -> dict[str, Any] | None:
    path = Path(out_dir) / DONE_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_done(out_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(out_dir) / DONE_FILE
    atomic_write_json(path, dict(payload))
    return path


def _archive_destination(out_dir: Path, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in reason
    ).strip("-") or "superseded"
    return out_dir / ATTIC_DIR / f"{safe}-{stamp}-{os.getpid()}"


def _move_to_archive(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if source.is_dir():
        retry_fs(
            shutil.move, str(source), str(target), what=f"archive {source.name}"
        )
    else:
        retry_replace(source, target, what=f"archive {source.name}")


def archive_run_state(
    out_dir: str | Path, *, reason: str = "fresh"
) -> Path:
    """Move every old checkpoint and run-root file under ``<out>/attic/``.

    Used by ``--fresh`` (and ``--rerun-complete``) so rotation and auto-resume
    can never pick an old trajectory from the same directory.
    """
    out_dir = Path(out_dir)
    destination = _archive_destination(out_dir, reason)
    moved = False
    for child in list_checkpoint_dirs(out_dir):
        _move_to_archive(child, destination)
        moved = True
    for name in _RUN_ROOT_FILES:
        path = out_dir / name
        if path.exists():
            _move_to_archive(path, destination)
            moved = True
    if not moved:
        # Remove the empty attic budget directory to keep an empty run dir tidy.
        try:
            destination.rmdir()
        except OSError:
            pass
    return destination


def archive_newer_checkpoints(
    out_dir: str | Path, *, resume_step: int, reason: str = "rewind"
) -> Path:
    """Archive checkpoints newer than ``resume_step`` plus the stale run root."""
    out_dir = Path(out_dir)
    destination = _archive_destination(out_dir, reason)
    moved = False
    for child in list_checkpoint_dirs(out_dir):
        step = _checkpoint_step(child)
        if step is not None and step <= int(resume_step):
            continue
        _move_to_archive(child, destination)
        moved = True
    for name in _RUN_ROOT_FILES:
        path = out_dir / name
        if path.exists():
            _move_to_archive(path, destination)
            moved = True
    if not moved:
        try:
            destination.rmdir()
        except OSError:
            pass
    return destination


def _manifest(directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == COMPLETE_FILE:
            continue
        entries.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return entries


def finalize_checkpoint(
    staging_dir: str | Path,
    final_dir: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Flush ``staging_dir``, write ``COMPLETE.json`` last, then rename atomically.

    ``staging_dir`` must be on the same filesystem as ``final_dir`` (it normally
    is a sibling ``checkpoint-<step>.tmp`` directory).
    """

    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    if not staging_dir.is_dir():
        raise CheckpointIncomplete(f"staging checkpoint {staging_dir} does not exist")

    files = _manifest(staging_dir)
    for entry in files:
        fsync_file(staging_dir / entry["path"])

    payload = dict(metadata)
    payload.update(
        {
            "complete": True,
            "completed_utc": utc_now(),
            "files": files,
        }
    )
    atomic_write_json(staging_dir / COMPLETE_FILE, payload)
    fsync_directory(staging_dir)

    if final_dir.exists():
        # Keep the old checkpoint as a backup until the new one is in place;
        # a crash between the two renames must not delete the only good copy.
        LOGGER.warning("replacing existing checkpoint %s", final_dir)
        backup = final_dir.with_name(final_dir.name + ".old.tmp")
        if backup.exists():
            retry_rmtree(backup)
        retry_replace(final_dir, backup, what=f"backup {final_dir.name}")
        try:
            retry_replace(staging_dir, final_dir, what=f"publish {final_dir.name}")
        except BaseException:
            retry_replace(backup, final_dir, what=f"restore {final_dir.name}")
            raise
        retry_rmtree(backup)
    else:
        retry_replace(staging_dir, final_dir, what=f"publish {final_dir.name}")
    fsync_directory(final_dir.parent)
    return payload


def rotate_checkpoints(
    out_dir: str | Path,
    *,
    keep: int = DEFAULT_KEEP_CHECKPOINTS,
    protect: Iterable[str | Path] = (),
) -> list[Path]:
    """Delete complete checkpoints beyond the newest ``keep`` (plus protected)."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    protected = {Path(path).resolve() for path in protect}
    checkpoints = list_complete_checkpoints(out_dir, verify=False)
    keep_paths = {
        info.path.resolve() for info in checkpoints[-keep:] if keep
    } | protected
    removed: list[Path] = []
    for info in checkpoints:
        if info.path.resolve() in keep_paths:
            continue
        try:
            retry_rmtree(info.path)
        except RetryableIOError as exc:  # pragma: no cover - filesystem dependent
            LOGGER.warning("could not remove old checkpoint %s: %s", info.path, exc)
            continue
        removed.append(info.path)
    return removed


def cleanup_stale_tmp_dirs(out_dir: str | Path) -> list[Path]:
    """Delete leftover ``checkpoint-*.tmp`` / staging directories from a crash."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    removed: list[Path] = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if (name.startswith(CHECKPOINT_PREFIX) and TMP_SUFFIX in name) or name == ".final.tmp":
            try:
                retry_rmtree(child)
            except RetryableIOError as exc:  # pragma: no cover - filesystem dependent
                LOGGER.warning("could not remove stale temp %s: %s", child, exc)
                continue
            removed.append(child)
    return removed


_REQUIRED_STATE_FIELDS = (
    ("has_optimizer", "optimizer state"),
    ("has_scheduler", "scheduler state"),
    ("rng_states_saved", "RNG states"),
)


def assert_resume_state_present(metadata: Mapping[str, Any]) -> None:
    """Refuse a checkpoint that cannot restore the full training state.

    A valid ``COMPLETE.json`` is not enough: without the optimizer, scheduler,
    RNG states, the N-sampler position and the data-order position the resumed
    run would silently reset Adam / dropout / the loop sampler / the next row.
    """

    bounded = getattr(metadata, "get", None)
    if not callable(bounded):  # pragma: no cover - defensive
        raise CheckpointCorrupt("checkpoint metadata is not a mapping")
    missing = [
        label
        for field, label in _REQUIRED_STATE_FIELDS
        if not metadata.get(field)
    ]
    if not isinstance(metadata.get("loop_sampler"), dict):
        missing.append("N-sampler state")
    if not isinstance(metadata.get("data_order"), dict):
        missing.append("data-order position")
    if missing:
        raise CheckpointCorrupt(
            "checkpoint is missing required resume state: " + ", ".join(missing)
        )


_RESUME_FIELDS = (
    ("run_config_sha256", "run configuration"),
    ("data_manifest_sha256", "data manifest"),
    ("git_sha", "code git SHA"),
    ("prereg_sha", "pre-registration SHA"),
)


def assert_resume_compatible(
    metadata: Mapping[str, Any],
    *,
    run_config_sha256: str,
    data_manifest_sha256: str,
    git_sha: str,
    prereg_sha: str | None,
    run_identity_hash: str | None = None,
    allow_mismatch: bool = False,
) -> list[str]:
    """Refuse (or with ``allow_mismatch`` warn about) a changed run identity.

    Returns the list of human-readable mismatch descriptions.  Raises
    :class:`ResumeMismatch` when mismatches exist and ``allow_mismatch`` is
    false.
    """

    current = {
        "run_config_sha256": run_config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "git_sha": git_sha,
        "prereg_sha": prereg_sha,
    }
    if run_identity_hash is not None:
        current["run_identity_hash"] = run_identity_hash
    mismatches: list[str] = []
    for field, label in _RESUME_FIELDS:
        if field not in metadata:
            mismatches.append(f"{label}: checkpoint has no {field}")
            continue
        saved = metadata.get(field)
        if saved != current[field]:
            mismatches.append(
                f"{label}: checkpoint {saved!r} != current {current[field]!r}"
            )
    if run_identity_hash is not None and metadata.get("run_identity_hash") != run_identity_hash:
        mismatches.append(
            "run identity hash: checkpoint "
            f"{metadata.get('run_identity_hash')!r} != current {run_identity_hash!r}"
        )
    if mismatches and not allow_mismatch:
        raise ResumeMismatch(
            "refusing to resume: " + "; ".join(mismatches)
            + ". Pass --allow-resume-mismatch only for a smoke run."
        )
    return mismatches
