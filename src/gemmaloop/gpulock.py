"""Process-lifetime GPU job lock.

Only one process should drive the GPU at a time while the training-free
campaign is exploratory and launched by hand.  The lock is a one-byte byte-range
lock on ``<artifacts>/gpu.lock`` (``msvcrt.locking`` on Windows,
``fcntl.flock`` elsewhere).  The OS releases it when the process exits, so a
crash never leaves a stale lock.

Entry points used by the scripts:

``require_gpu_lock()``
    Acquire the lock or exit with :data:`LOCK_BUSY_EXIT_CODE` (75) when another
    process holds it.  Safe to call from a script's ``main`` before doing any
    GPU work.

``acquire()`` / ``try_acquire()``
    Library API.  ``acquire`` raises :class:`GpuLockBusy`; ``try_acquire``
    returns ``None``.

The lock file path can be overridden with ``GEMMALOOP_GPU_LOCK_PATH`` (tests and
the campaign).  There is deliberately no inherited-lock bypass: every process
that touches the GPU must own the lock itself, so a crashed parent can never
leave an unlocked child running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO

from gemmaloop.paths import artifacts_dir

LOCK_BUSY_EXIT_CODE = 75
LOCK_PATH_ENV_VAR = "GEMMALOOP_GPU_LOCK_PATH"
LOCK_NAME = "gpu.lock"

__all__ = [
    "LOCK_BUSY_EXIT_CODE",
    "LOCK_NAME",
    "LOCK_PATH_ENV_VAR",
    "GpuLock",
    "GpuLockBusy",
    "acquire",
    "lock_path",
    "release_held_lock",
    "require_gpu_lock",
    "try_acquire",
]


class GpuLockBusy(RuntimeError):
    """Raised when another process already holds the GPU lock."""


class GpuLock:
    """A held lock; keep the instance alive for the process lifetime."""

    def __init__(self, path: Path, handle: IO[bytes]) -> None:
        self.path = path
        self._handle = handle
        self._released = False

    def release(self) -> None:
        """Release the byte-range lock and close the file (idempotent)."""
        if self._released:
            return
        self._released = True
        handle = self._handle
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                handle.close()
            except OSError:
                pass

    def __enter__(self) -> "GpuLock":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def lock_path(path: str | Path | None = None) -> Path:
    """Return the configured lock file path (created parent not guaranteed)."""
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(LOCK_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return artifacts_dir() / LOCK_NAME


def _try_lock(handle: IO[bytes]) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def try_acquire(path: str | Path | None = None) -> GpuLock | None:
    """Try to take the lock; return ``None`` when another process holds it."""
    target = lock_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a+b")
    if _try_lock(handle):
        return GpuLock(target, handle)
    handle.close()
    return None


def acquire(path: str | Path | None = None) -> GpuLock:
    """Take the lock or raise :class:`GpuLockBusy`."""
    held = try_acquire(path)
    if held is None:
        raise GpuLockBusy(f"GPU lock is held: {lock_path(path)}")
    return held


_HELD: GpuLock | None = None


def release_held_lock() -> None:
    """Release the module-level lock (used by tests and by the campaign)."""
    global _HELD
    if _HELD is not None:
        _HELD.release()
        _HELD = None


def require_gpu_lock(path: str | Path | None = None) -> GpuLock | None:
    """Acquire the lock for the lifetime of this process or exit with 75."""
    global _HELD
    target = lock_path(path)
    if _HELD is not None and _HELD.path == target:
        return _HELD
    try:
        _HELD = acquire(target)
    except GpuLockBusy as exc:
        print(f"error: {exc}; another GPU job is running", file=sys.stderr)
        raise SystemExit(LOCK_BUSY_EXIT_CODE) from None
    return _HELD
