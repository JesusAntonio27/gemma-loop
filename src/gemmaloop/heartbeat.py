"""Atomic job heartbeat for detached GPU runs.

The controller polls one small JSON file to tell whether a job is alive and how
far it has got.  Every write is atomic (temporary file + :func:`os.replace`) so
a poll never sees a half-written document.

Fields (all optional except ``ts``)::

    ts                 ISO-8601 UTC timestamp
    run / phase        labels for the job and the current stage/benchmark
    step               completed items (or plan jobs for the campaign)
    items_done         alias of ``step`` for item-level runners
    items_total        total items in the current phase
    tok_per_s          generated tokens per wall-clock second since start
    torch_reserved_gib PyTorch CUDA reserved memory, 0 when CUDA is absent
    ram_available_gib  currently available system RAM
    batch              size of the batch that produced this heartbeat
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gemmaloop.paths import artifacts_dir

__all__ = [
    "Heartbeat",
    "default_heartbeat_path",
    "ram_available_gib",
    "torch_reserved_gib",
]

_HEARTBEAT_DIR = "runs" / Path("detached")


def default_heartbeat_path(name: str = "heartbeat") -> Path:
    """Return ``<artifacts>/runs/detached/<name>.json``.

    ``GEMMALOOP_HEARTBEAT_PATH`` overrides the location (tests and the
    campaign pass explicit paths; this keeps unit tests off the real artifact
    tree).
    """
    override = os.environ.get("GEMMALOOP_HEARTBEAT_PATH")
    if override:
        return Path(override).expanduser()
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in str(name)
    ).strip("-") or "heartbeat"
    return artifacts_dir() / _HEARTBEAT_DIR / f"{safe}.json"


def ram_available_gib() -> float:
    """Return available system RAM in GiB (best effort, never raises)."""
    try:
        import psutil  # type: ignore

        return round(float(psutil.virtual_memory().available) / 2**30, 3)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(float(status.ullAvailPhys) / 2**30, 3)
        except Exception:
            pass
        return 0.0

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(float(pages * page_size) / 2**30, 3)
    except Exception:
        return 0.0


def torch_reserved_gib() -> float:
    """Return CUDA reserved memory in GiB, or 0 when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(float(torch.cuda.memory_reserved()) / 2**30, 3)
    except Exception:
        pass
    return 0.0


class Heartbeat:
    """Write a heartbeat JSON after every finished batch."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        run: str = "",
        phase: str = "",
    ) -> None:
        self.path = Path(path) if path is not None else default_heartbeat_path()
        self.run = str(run)
        self.phase = str(phase)
        self.items_done = 0
        self.items_total = 0
        self.tokens = 0
        self.batch = 0
        self._started = time.monotonic()
        self._last_ts = time.time()
        self._extra: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, interval: float = 30.0) -> "Heartbeat":
        """Start a daemon writer so a long phase is never silent for >interval.

        Callers still update counters through :meth:`update`; this only
        guarantees the file is refreshed while a single arm/chunk is running.
        """
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(max(0.05, float(interval))):
                try:
                    self.write()
                except Exception:
                    pass

        self._thread = threading.Thread(
            target=_loop, name="gemmaloop-heartbeat", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the background writer and flush one final heartbeat."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        try:
            self.write()
        except Exception:
            pass

    def set_phase(
        self,
        phase: str,
        *,
        total: int | None = None,
        reset_items: bool = True,
    ) -> None:
        """Start a new phase (benchmark, stage) and write immediately."""
        self.phase = str(phase)
        if total is not None:
            self.items_total = int(total)
        if reset_items:
            self.items_done = 0
        self.batch = 0
        self.write()

    def update(
        self,
        *,
        items_done: int | None = None,
        items_total: int | None = None,
        tokens: int | None = None,
        step: int | None = None,
        batch: int | None = None,
        phase: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Update counters and write atomically.

        ``tokens`` is the number of tokens generated since the last call (a
        delta).  ``step`` overrides ``items_done`` so the campaign can report a
        plan-job counter through the same file.
        """
        if phase is not None:
            self.phase = str(phase)
        if tokens is not None:
            self.tokens += max(0, int(tokens))
        if items_total is not None:
            self.items_total = int(items_total)
        if items_done is not None:
            self.items_done = int(items_done)
        if step is not None:
            self.items_done = int(step)
        if batch is not None:
            self.batch = int(batch)
        if extra:
            self._extra.update(dict(extra))
        try:
            self.write()
        except Exception:
            # Progress must never abort a GPU job because a poller briefly
            # holds the destination file on Windows.
            pass

    def _elapsed(self) -> float:
        return max(1e-9, time.monotonic() - self._started)

    def payload(self) -> dict[str, Any]:
        elapsed = self._elapsed()
        now = time.time()
        return {
            "ts": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "epoch": round(now, 3),
            "run": self.run,
            "phase": self.phase,
            "step": int(self.items_done),
            "items_done": int(self.items_done),
            "items_total": int(self.items_total),
            "tok_per_s": round(self.tokens / elapsed, 3),
            "generated_tokens": int(self.tokens),
            "elapsed_s": round(elapsed, 3),
            "batch": int(self.batch),
            "torch_reserved_gib": torch_reserved_gib(),
            "ram_available_gib": ram_available_gib(),
            **getattr(self, "_extra", {}),
        }

    def write(self) -> Path:
        """Write the heartbeat atomically and return its path."""
        payload = self.payload()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._last_ts = payload["epoch"]
        return self.path
