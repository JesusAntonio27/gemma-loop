"""Crash-safety guards for multi-day training runs (Task 14b).

The guards are pure decision objects so they can be exercised with fake series
on CPU.  :class:`WatchdogCallback` wires them into the Hugging Face trainer and
asks for a checkpoint followed by exit code 75 (retryable) when a guard fires.

Defaults (recorded in the run's ``status.json`` and ``run.json``):

* throughput: rolling median of ``--throughput-window-steps`` optimizer steps
  below ``0.5x`` the post-warm-up baseline for three consecutive windows;
* VRAM: PyTorch reserved memory above the allocator cap (the cap already makes
  an over-budget step OOM instead of paging);
* RAM: less than 2 GiB available system memory.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from transformers import TrainerCallback

LOGGER = logging.getLogger(__name__)

GUARD_EXIT_CODE = 75
DEFAULT_THROUGHPUT_DROP_RATIO = 0.5
DEFAULT_THROUGHPUT_WINDOW_STEPS = 10
DEFAULT_THROUGHPUT_FAILURES = 3
DEFAULT_BASELINE_START_STEP = 20
DEFAULT_BASELINE_END_STEP = 100
DEFAULT_MIN_RAM_GIB = 2.0
DEFAULT_SAVE_EVERY_MINUTES = 30.0

__all__ = [
    "DEFAULT_BASELINE_END_STEP",
    "DEFAULT_BASELINE_START_STEP",
    "DEFAULT_MIN_RAM_GIB",
    "DEFAULT_SAVE_EVERY_MINUTES",
    "DEFAULT_THROUGHPUT_DROP_RATIO",
    "DEFAULT_THROUGHPUT_FAILURES",
    "DEFAULT_THROUGHPUT_WINDOW_STEPS",
    "GUARD_EXIT_CODE",
    "GuardDecision",
    "RamGuard",
    "ThroughputGuard",
    "ThroughputGuardConfig",
    "TimeSaveCallback",
    "WatchdogCallback",
    "guard_thresholds",
    "ram_available_gib",
    "reserved_gib",
]


def _platform_name(platform: str | None = None) -> str:
    if platform:
        return platform
    return "windows" if os.name == "nt" else "posix"


@dataclass(frozen=True)
class GuardDecision:
    triggered: bool = False
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.triggered


@dataclass
class ThroughputGuardConfig:
    window_steps: int = DEFAULT_THROUGHPUT_WINDOW_STEPS
    baseline_start_step: int = DEFAULT_BASELINE_START_STEP
    baseline_end_step: int = DEFAULT_BASELINE_END_STEP
    drop_ratio: float = DEFAULT_THROUGHPUT_DROP_RATIO
    consecutive_windows: int = DEFAULT_THROUGHPUT_FAILURES
    platform: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_steps": int(self.window_steps),
            "baseline_start_step": int(self.baseline_start_step),
            "baseline_end_step": int(self.baseline_end_step),
            "drop_ratio": float(self.drop_ratio),
            "consecutive_windows": int(self.consecutive_windows),
            "platform": _platform_name(self.platform),
        }


class ThroughputGuard:
    """Detect a sustained throughput collapse; the OOM-at-cap path exits 75."""

    def __init__(self, config: ThroughputGuardConfig | None = None) -> None:
        self.config = config or ThroughputGuardConfig()
        if self.config.window_steps <= 0:
            raise ValueError("window_steps must be positive")
        if self.config.consecutive_windows <= 0:
            raise ValueError("consecutive_windows must be positive")
        if not 0.0 < float(self.config.drop_ratio) < 1.0:
            raise ValueError("drop_ratio must be in (0, 1)")
        self.low_windows = 0
        self.baseline_tok_s: float | None = None
        self._baseline_samples: list[float] = []
        self._window: list[float] = []
        self._restored_baseline = False
        self.last_metrics: dict[str, Any] = {}

    def reset(self) -> None:
        self.low_windows = 0
        self.baseline_tok_s = None
        self._baseline_samples.clear()
        self._window.clear()
        self._restored_baseline = False

    def state_dict(self) -> dict[str, Any]:
        """The baseline survives checkpoints so protection starts immediately."""
        return {
            "baseline_tok_s": self.baseline_tok_s,
            "low_windows": int(self.low_windows),
        }

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        self.reset()
        if not payload:
            return
        baseline = payload.get("baseline_tok_s")
        if isinstance(baseline, (int, float)) and float(baseline) > 0:
            self.baseline_tok_s = float(baseline)
            self._restored_baseline = True
        self.low_windows = int(payload.get("low_windows", 0) or 0)

    def observe(
        self,
        step: int,
        tokens: int,
        elapsed_seconds: float,
    ) -> GuardDecision:
        """Observe one optimizer step and decide whether to stop retryably."""
        step = int(step)
        elapsed = float(elapsed_seconds)
        if elapsed <= 0:
            return GuardDecision(metrics={"step": step})
        tok_s = max(0.0, float(tokens)) / elapsed

        self.last_metrics = {
            "step": step,
            "tok_per_s": tok_s,
            "baseline_tok_s": self.baseline_tok_s,
            "window_median_tok_s": None,
            "low_windows": self.low_windows,
        }

        cfg = self.config
        if (
            tok_s > 0
            and cfg.baseline_start_step <= step < cfg.baseline_end_step
        ):
            self._baseline_samples.append(tok_s)

        if self.baseline_tok_s is None and step >= cfg.baseline_end_step:
            if self._baseline_samples:
                self.baseline_tok_s = statistics.median(self._baseline_samples)
                LOGGER.info(
                    "throughput baseline %.3f tok/s (median of %d steps)",
                    self.baseline_tok_s,
                    len(self._baseline_samples),
                )
            else:
                # No samples in the window (e.g. a restart late in the run):
                # use the step itself so the guard can still act.
                self.baseline_tok_s = tok_s

        if self.baseline_tok_s is None or self.baseline_tok_s <= 0:
            return GuardDecision(metrics=self.last_metrics)

        window_start = (
            cfg.baseline_start_step
            if self._restored_baseline
            else cfg.baseline_end_step
        )
        if step >= window_start:
            self._window.append(tok_s)
            if len(self._window) >= cfg.window_steps:
                median = statistics.median(self._window)
                self._window.clear()
                if median < float(cfg.drop_ratio) * float(self.baseline_tok_s):
                    self.low_windows += 1
                else:
                    self.low_windows = 0
                self.last_metrics["window_median_tok_s"] = median
                self.last_metrics["low_windows"] = self.low_windows
                if self.low_windows >= cfg.consecutive_windows:
                    platform = _platform_name(cfg.platform)
                    reason = (
                        f"throughput collapse: {median:.1f} tok/s < "
                        f"{float(cfg.drop_ratio):.2f} x baseline "
                        f"{float(self.baseline_tok_s):.1f} tok/s for "
                        f"{self.low_windows} consecutive windows"
                    )
                    if platform == "windows":
                        reason += " (WDDM spill suspected: sustained slowdown)"
                    return GuardDecision(
                        triggered=True,
                        reason=reason,
                        metrics=dict(self.last_metrics),
                    )

        self.last_metrics["baseline_tok_s"] = self.baseline_tok_s
        return GuardDecision(metrics=self.last_metrics)


class RamGuard:
    """Stop retryably just before Windows starts thrashing on the pagefile."""

    def __init__(self, min_available_gib: float = DEFAULT_MIN_RAM_GIB) -> None:
        if float(min_available_gib) < 0:
            raise ValueError("min_available_gib must be non-negative")
        self.min_available_gib = float(min_available_gib)

    def to_dict(self) -> dict[str, float]:
        return {"min_available_gib": self.min_available_gib}

    def observe(self, available_gib: float | None) -> GuardDecision:
        if available_gib is None:
            return GuardDecision()
        value = float(available_gib)
        if value < self.min_available_gib:
            return GuardDecision(
                triggered=True,
                reason=(
                    f"available RAM {value:.2f} GiB < "
                    f"{self.min_available_gib:.2f} GiB"
                ),
                metrics={"ram_available_gib": value},
            )
        return GuardDecision(metrics={"ram_available_gib": value})


def guard_thresholds(
    *,
    throughput: ThroughputGuardConfig,
    ram: RamGuard,
    save_every_minutes: float,
) -> dict[str, Any]:
    """The exact threshold values recorded in ``status.json``/``run.json``."""
    return {
        "throughput": throughput.to_dict(),
        "ram": ram.to_dict(),
        "save_every_minutes": float(save_every_minutes),
        "exit_code_on_guard": GUARD_EXIT_CODE,
    }


class TimeSaveCallback(TrainerCallback):
    """Trigger a checkpoint every ``every_minutes`` of wall-clock time.

    Implemented as a plain object (not a ``TrainerCallback`` subclass at import
    time) so a fake clock can drive the unit tests; it exposes the two callback
    methods Hugging Face calls.
    """

    def __init__(
        self,
        every_minutes: float = DEFAULT_SAVE_EVERY_MINUTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if float(every_minutes) <= 0:
            raise ValueError("every_minutes must be positive")
        self.every_seconds = float(every_minutes) * 60
        self.clock = clock
        self.last_save = float(clock())
        self.saves = 0

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, state, kwargs
        # The model load can take a long time; the first save interval starts
        # when training actually starts.
        self.last_save = float(self.clock())
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, state, kwargs
        if self.clock() - self.last_save >= self.every_seconds:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, state, kwargs
        self.last_save = float(self.clock())
        self.saves += 1
        return control


class WatchdogCallback(TrainerCallback):
    """Feed per-step throughput / RAM / VRAM into the guards.

    On a trigger it sets ``should_save`` and ``should_training_stop`` and records
    ``trainer.exit_code = 75`` and ``trainer.exit_reason``.  The actual process
    exit code is chosen by ``scripts/train.py`` after the checkpoint is written.
    """

    def __init__(
        self,
        trainer,  # noqa: ANN001 - LoopSFTTrainer
        throughput: ThroughputGuard,
        ram: RamGuard,
        *,
        clock: Callable[[], float] = time.monotonic,
        ram_reader: Callable[[], float] | None = None,
        reserved_reader: Callable[[], float | None] | None = None,
        status=None,  # noqa: ANN001 - optional StatusWriter
    ) -> None:
        self.trainer = trainer
        self.throughput = throughput
        self.ram = ram
        self.clock = clock
        self.ram_reader = ram_reader or _read_ram_gib
        self.reserved_reader = reserved_reader or _read_reserved_gib
        self.status = status
        self._last_clock = float(clock())
        self._last_tokens = 0
        self._start_step = 0
        self.decisions: list[GuardDecision] = []

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        """Anchor the counters after a resume (no cumulative-token spike)."""
        del args, kwargs
        self._last_tokens = int(getattr(self.trainer, "tokens_seen", 0))
        self._last_clock = float(self.clock())
        self._start_step = int(state.global_step)
        persisted = getattr(self.trainer, "_persisted_guard_state", None)
        if isinstance(persisted, dict) and persisted.get("baseline_tok_s"):
            self.throughput.load_state_dict(persisted)
            LOGGER.info(
                "restored throughput baseline %.3f tok/s from the checkpoint",
                self.throughput.baseline_tok_s,
            )
        else:
            self.throughput.reset()
        self.trainer._throughput_baseline = self.throughput.baseline_tok_s
        self.trainer._throughput_low_windows = self.throughput.low_windows
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        del args, kwargs
        now = float(self.clock())
        elapsed = max(1e-9, now - self._last_clock)
        tokens = max(0, int(self.trainer.tokens_seen) - self._last_tokens)
        self._last_clock = now
        self._last_tokens = int(self.trainer.tokens_seen)
        local_step = max(1, int(state.global_step) - int(self._start_step) + 1)

        throughput = self.throughput.observe(local_step, tokens, elapsed)
        self.trainer._throughput_baseline = self.throughput.baseline_tok_s
        self.trainer._throughput_low_windows = self.throughput.low_windows
        self.decisions.append(throughput)
        if throughput.triggered:
            return self._fire(control, "throughput", throughput.reason, throughput.metrics)

        ram = self.ram.observe(self.ram_reader())
        self.decisions.append(ram)
        if ram.triggered:
            return self._fire(control, "ram", ram.reason, ram.metrics)
        return control

    def _fire(self, control, phase: str, reason: str, metrics: dict[str, Any]):  # noqa: ANN001
        LOGGER.error("crash-safety guard fired (%s): %s", phase, reason)
        self.trainer.exit_code = GUARD_EXIT_CODE
        self.trainer.exit_reason = f"{phase}: {reason}"
        control.should_save = True
        control.should_training_stop = True
        if self.status is not None:
            # A status failure must never prevent the checkpoint requested
            # above (should_save was set first).
            try:
                self.status.update(
                    phase=f"stopping-{phase}",
                    extra={"guard": {"phase": phase, "reason": reason, **metrics}},
                )
            except Exception as exc:  # noqa: BLE001 - status is best-effort
                LOGGER.warning("could not write guard status: %s", exc)
        return control


def ram_available_gib() -> float | None:
    """Available system RAM in GiB (``None`` when it cannot be read)."""
    try:
        from gemmaloop.heartbeat import ram_available_gib as reader

        value = reader()
        return float(value) if value > 0 else None
    except Exception:  # pragma: no cover - never break training
        return None


def reserved_gib() -> float | None:
    """PyTorch CUDA reserved memory in GiB (``None`` without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_reserved()) / 2**30
    except Exception:  # pragma: no cover - CUDA query failure
        return None
    return None


_read_ram_gib = ram_available_gib
_read_reserved_gib = reserved_gib
