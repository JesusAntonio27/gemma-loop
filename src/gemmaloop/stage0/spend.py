"""Persistent Modal spend ledger.

The author pays with a personal credit card and the Modal account has a $30
monthly usage limit.  A single process-local counter is not enough: crashes,
2-hour timeouts, failed commits and resumed runs all start a new process.  Every
launch therefore writes a **reservation** for the next container's worst-case
cost to a local JSON ledger *before* contacting Modal, then replaces it with the
observed cost afterwards.  The guard always uses the cumulative current-month
value (actual spend + outstanding reservations).

The ledger is append-only and written atomically (``.tmp`` + ``os.replace``).
A reservation without a matching actual entry is treated as spent; this is
conservative after a crash and cannot lead to overspending.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Iterator

LEDGER_VERSION = 1
LOCK_BUSY_EXIT_CODE = 75
SPEND_LOCK_NAME = "spend.lock"
SPEND_LOCK_ENV_VAR = "GEMMALOOP_SPEND_LOCK_PATH"


def month_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m")


def default_ledger_path() -> Path:
    from gemmaloop.paths import artifacts_dir

    return artifacts_dir() / "stage0" / "modal_spend_ledger.json"


def spend_lock_path(path: Path | str | None = None) -> Path:
    """Default process-lifetime spend lock under the artifacts directory."""

    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(SPEND_LOCK_ENV_VAR)
    if override:
        return Path(override).expanduser()
    from gemmaloop.paths import artifacts_dir

    return artifacts_dir() / "stage0" / SPEND_LOCK_NAME


class SpendLockBusy(RuntimeError):
    """Raised when another process already holds the spend lock."""


class SpendLock:
    """A held process-lifetime lock; keep the instance alive until exit."""

    def __init__(self, path: Path, handle: IO[bytes]) -> None:
        self.path = path
        self._handle = handle
        self._released = False

    def release(self) -> None:
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

    def __enter__(self) -> "SpendLock":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


_LEDGER_THREAD_LOCK = threading.RLock()


@contextmanager
def _ledger_file_lock(path: Path) -> Iterator[None]:
    """Short-lived cross-process lock around one ledger append.

    The process-lifetime spend lock keeps separate live runs apart; this
    per-append lock additionally protects a library caller (a judge or the
    vision gate) that did not take the process lock.  The thread lock makes
    the same guarantee deterministic for threads in one process.
    """

    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    with _LEDGER_THREAD_LOCK:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    try:
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
                handle.close()


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


def try_acquire_spend_lock(path: Path | str | None = None) -> SpendLock | None:
    target = spend_lock_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a+b")
    if _try_lock(handle):
        return SpendLock(target, handle)
    handle.close()
    return None


def acquire_spend_lock(path: Path | str | None = None) -> SpendLock:
    held = try_acquire_spend_lock(path)
    if held is None:
        raise SpendLockBusy(f"spend lock is held: {spend_lock_path(path)}")
    return held


_HELD: SpendLock | None = None


def release_held_spend_lock() -> None:
    global _HELD
    if _HELD is not None:
        _HELD.release()
        _HELD = None


def held_spend_lock_path() -> Path | None:
    """Return the path of the lock this process currently holds, if any."""

    return _HELD.path if _HELD is not None else None


def require_spend_lock(path: Path | str | None = None) -> SpendLock:
    """Acquire the lock for this process or exit 75 with a clear message."""

    global _HELD
    target = spend_lock_path(path)
    if _HELD is not None:
        if _HELD.path == target:
            return _HELD
        _HELD.release()
        _HELD = None
    try:
        _HELD = acquire_spend_lock(target)
    except SpendLockBusy as exc:
        print(
            f"error: {exc}; another money-spending process is running "
            "(Modal launcher or DeepSeek live writer). Refusing to start.",
            file=sys.stderr,
        )
        raise SystemExit(LOCK_BUSY_EXIT_CODE) from None
    return _HELD


def _timestamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat()


def _entry_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:24]


class SpendLedger:
    """Append-only, atomically written monthly spend ledger."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else default_ledger_path()
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": LEDGER_VERSION, "entries": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid spend ledger at {self.path}")
        payload.setdefault("version", LEDGER_VERSION)
        payload.setdefault("entries", [])
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        try:
            directory = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        except OSError:
            pass
        finally:
            os.close(directory)

    def _append(self, entry: dict[str, Any]) -> str:
        with _ledger_file_lock(self.path):
            payload = self.load()
            entry_id = entry.get("id") or _entry_id(
                f"{_timestamp(self._now())}:{entry.get('kind')}:{entry.get('shard')}:"
                f"{entry.get('usd')}:{len(payload['entries'])}"
            )
            entry["id"] = entry_id
            entry.setdefault("timestamp", _timestamp(self._now()))
            payload["entries"].append(entry)
            self._write(payload)
            return entry_id

    def reserve(
        self,
        *,
        shard: int,
        usd: float,
        gpu: str = "",
        timeout_s: float = 0.0,
        month: str | None = None,
        provider: str = "modal",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        month = month or month_key(self._now())
        return self._append(
            {
                "kind": "reservation",
                "month": month,
                "provider": provider,
                "shard": int(shard),
                "usd": float(usd),
                "gpu": gpu,
                "timeout_s": float(timeout_s),
                "metadata": metadata or {},
            }
        )

    def settle(
        self,
        reservation_id: str,
        *,
        shard: int,
        actual_usd: float,
        month: str | None = None,
        provider: str = "modal",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        month = month or month_key(self._now())
        return self._append(
            {
                "kind": "actual",
                "month": month,
                "provider": provider,
                "shard": int(shard),
                "usd": float(actual_usd),
                "reservation_id": reservation_id,
                "metadata": metadata or {},
            }
        )

    def entries(self) -> list[dict[str, Any]]:
        return list(self.load()["entries"])

    def cumulative_usd(
        self, month: str | None = None, *, provider: str | None = "modal"
    ) -> float:
        month = month or month_key(self._now())
        entries = self.load()["entries"]

        def matches(entry: dict[str, Any]) -> bool:
            return entry.get("month") == month and (
                provider is None or entry.get("provider", "modal") == provider
            )

        selected = [entry for entry in entries if matches(entry)]
        actual = sum(
            float(entry.get("usd", 0.0))
            for entry in selected
            if entry.get("kind") == "actual"
        )
        settled = {
            entry.get("reservation_id")
            for entry in selected
            if entry.get("kind") == "actual"
        }
        outstanding = sum(
            float(entry.get("usd", 0.0))
            for entry in selected
            if entry.get("kind") == "reservation"
            and entry.get("id") not in settled
        )
        return actual + outstanding


__all__ = [
    "LEDGER_VERSION",
    "LOCK_BUSY_EXIT_CODE",
    "SPEND_LOCK_ENV_VAR",
    "SPEND_LOCK_NAME",
    "SpendLedger",
    "SpendLock",
    "SpendLockBusy",
    "acquire_spend_lock",
    "default_ledger_path",
    "held_spend_lock_path",
    "month_key",
    "release_held_spend_lock",
    "require_spend_lock",
    "spend_lock_path",
    "try_acquire_spend_lock",
]
