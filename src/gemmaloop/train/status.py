"""Atomic run status and heartbeat files (Task 14b).

``<out>/status.json`` is rewritten every optimizer step and at exit.  It is the
single place the author/controller looks at to know whether a detached run is
alive, how far it got and why it stopped.  ``<out>/heartbeat.json`` is the small
liveness file the watchdog can poll without parsing the full status.

Every write goes to a sibling temporary file, is flushed, and then moved with
:func:`os.replace`, so a crash can leave the previous document but never a
truncated one.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from transformers import TrainerCallback

from gemmaloop.train.checkpoint import atomic_write_json

LOGGER = logging.getLogger(__name__)

STATUS_FILE = "status.json"
HEARTBEAT_FILE = "heartbeat.json"

__all__ = [
    "HEARTBEAT_FILE",
    "STATUS_FILE",
    "StatusCallback",
    "StatusWriter",
    "utc_iso",
]


def utc_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp, timezone.utc
    ).isoformat()


class StatusWriter:
    """Small atomic JSON status document for one run directory."""

    def __init__(
        self,
        path: str | Path,
        *,
        run: str = "",
        phase: str = "starting",
        identity: Mapping[str, Any] | None = None,
        thresholds: Mapping[str, Any] | None = None,
        heartbeat_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.heartbeat_path = (
            Path(heartbeat_path)
            if heartbeat_path is not None
            else self.path.with_name(HEARTBEAT_FILE)
        )
        self.clock = clock
        self.started_monotonic = float(clock())
        self.started_utc = utc_iso()
        self.payload: dict[str, Any] = {
            "run": str(run),
            "phase": str(phase),
            "pid": os.getpid(),
            "step": 0,
            "tokens_seen": 0,
            "tok_per_s": 0.0,
            "last_complete_checkpoint": None,
            "exit_reason": None,
            "exit_code": None,
            "started_utc": self.started_utc,
            "updated_utc": self.started_utc,
            "identity": dict(identity or {}),
            "thresholds": dict(thresholds or {}),
        }
        self.last_error: str | None = None

    def update(self, **fields: Any) -> dict[str, Any]:
        extra = fields.pop("extra", None)
        for key, value in fields.items():
            if key in {"identity", "thresholds"} and isinstance(value, Mapping):
                merged = dict(self.payload.get(key) or {})
                merged.update(dict(value))
                self.payload[key] = merged
            else:
                self.payload[key] = value
        if extra:
            merged_extra = dict(self.payload.get("extra") or {})
            merged_extra.update(dict(extra))
            self.payload["extra"] = merged_extra
        self.payload["updated_utc"] = utc_iso()
        try:
            self.write()
        except Exception as exc:  # pragma: no cover - write() already swallows
            self.last_error = str(exc)
            LOGGER.warning("status update failed (training continues): %s", exc)
        return self.payload

    def set_exit(self, reason: str, code: int) -> dict[str, Any]:
        return self.update(
            phase="exited",
            exit_reason=str(reason),
            exit_code=int(code),
            exited_utc=utc_iso(),
        )

    def _heartbeat_payload(self) -> dict[str, Any]:
        return {
            "ts": utc_iso(),
            "run": self.payload.get("run", ""),
            "phase": self.payload.get("phase", ""),
            "step": int(self.payload.get("step", 0)),
            "tokens_seen": int(self.payload.get("tokens_seen", 0)),
            "tok_per_s": float(self.payload.get("tok_per_s", 0.0)),
            "last_complete_checkpoint": self.payload.get(
                "last_complete_checkpoint"
            ),
            "reserved_gib": self.payload.get("reserved_gib"),
            "ram_available_gib": self.payload.get("ram_available_gib"),
            "elapsed_s": round(float(self.clock()) - self.started_monotonic, 3),
        }

    @staticmethod
    def _write_with_retry(path: Path, payload: Mapping[str, Any]) -> None:
        delay = 0.02
        last: Exception | None = None
        for attempt in range(3):
            try:
                atomic_write_json(path, payload)
                return
            except Exception as exc:  # noqa: BLE001 - filesystem races on Windows
                last = exc
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
        assert last is not None
        raise last

    def write(self) -> Path:
        """Best-effort atomic write; a poller holding the file never kills a run."""
        errors: list[str] = []
        try:
            self._write_with_retry(self.path, self.payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{self.path.name}: {exc}")
            LOGGER.warning("could not write %s (retried next step): %s", self.path, exc)
        if self.heartbeat_path is not None:
            try:
                self._write_with_retry(
                    self.heartbeat_path, self._heartbeat_payload()
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{self.heartbeat_path.name}: {exc}")
                LOGGER.warning(
                    "could not write %s (retried next step): %s",
                    self.heartbeat_path,
                    exc,
                )
        self.last_error = "; ".join(errors) or None
        return self.path


class StatusCallback(TrainerCallback):
    """Update the status document after every optimizer step and on save/exit."""

    def __init__(
        self,
        writer: StatusWriter,
        trainer,  # noqa: ANN001 - LoopSFTTrainer
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.writer = writer
        self.trainer = trainer
        self.clock = clock
        self._last_tokens = int(getattr(trainer, "tokens_seen", 0))
        self._last_clock = float(clock())

    def _tok_per_s(self) -> float:
        """Per-step tok/s (never the cumulative count divided by this run)."""
        now = float(self.clock())
        tokens = int(getattr(self.trainer, "tokens_seen", 0))
        elapsed = max(1e-9, now - self._last_clock)
        delta = max(0, tokens - self._last_tokens)
        self._last_tokens = tokens
        self._last_clock = now
        return float(delta) / elapsed

    def _last_checkpoint(self) -> str | None:
        value = getattr(self.trainer, "_last_checkpoint_path", None)
        return None if value is None else str(value)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, kwargs
        try:
            from gemmaloop.train.guards import ram_available_gib, reserved_gib
        except Exception:  # pragma: no cover - defensive
            ram_available_gib = lambda: None  # type: ignore
            reserved_gib = lambda: None  # type: ignore
        self.writer.update(
            phase="train",
            step=int(state.global_step),
            tokens_seen=int(getattr(self.trainer, "tokens_seen", 0)),
            tok_per_s=self._tok_per_s(),
            last_complete_checkpoint=self._last_checkpoint(),
            ram_available_gib=ram_available_gib(),
            reserved_gib=reserved_gib(),
            throughput_baseline_tok_s=getattr(
                self.trainer, "_throughput_baseline", None
            ),
            throughput_low_windows=getattr(
                self.trainer, "_throughput_low_windows", 0
            ),
        )
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, kwargs
        self.writer.update(
            phase="checkpoint",
            step=int(state.global_step),
            tokens_seen=int(getattr(self.trainer, "tokens_seen", 0)),
            tok_per_s=self._tok_per_s(),
            last_complete_checkpoint=self._last_checkpoint(),
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, kwargs
        self.writer.update(
            phase=self.writer.payload.get("phase", "train"),
            step=int(state.global_step),
            tokens_seen=int(getattr(self.trainer, "tokens_seen", 0)),
            tok_per_s=self._tok_per_s(),
            last_complete_checkpoint=self._last_checkpoint(),
        )
        return control
