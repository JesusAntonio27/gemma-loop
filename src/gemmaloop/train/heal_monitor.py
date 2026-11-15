"""Heal monitoring and early detection (observe-only).

This module implements the execution-safety monitor for the Stage 1 heal:

* cheap per-optimizer-step JSONL metrics in the run directory;
* inference-only probes on a dedicated MONITOR split every 0.5M tokens and at
  the end;
* registered early-warning and abort rules with the defaults in
  :class:`MonitorConfig`;
* an ``alerts.json`` file and a dedicated abort exit code.

Nothing here is a pilot selection input.  The monitor never changes the
optimiser, the data order, the RNG or the loss; the only mutation it performs
is the registered abort path (save the last complete checkpoint and exit with
:data:`MONITOR_ABORT_EXIT_CODE`).

The only hook in ``scripts/train.py`` is
:func:`install_heal_monitor`, which installs :class:`HealMonitorCallback`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from transformers import TrainerCallback

from gemmaloop.train.checkpoint import atomic_write_json
from gemmaloop.train.devices import model_compute_device
from gemmaloop.train.dial_objective import TeacherTopK

_LOG = logging.getLogger(__name__)

MONITOR_SPLIT_NAME = "monitor"
MONITOR_ABORT_EXIT_CODE = 76
MONITOR_METRICS_FILE = "heal_metrics.jsonl"
MONITOR_PROBES_FILE = "heal_probes.jsonl"
MONITOR_ALERTS_FILE = "alerts.json"
MONITOR_STATE_FILE = "heal_monitor_state.json"
MONITOR_GENERATION_DIR = "heal_probe_texts"
PRIMARY_N = 4

# Registered execution-safety rules.  The prereg cites this module and these
# default values; ``MonitorConfig.to_dict`` is the machine-readable copy.
ABORT_RULES = (
    {
        "id": "nonfinite",
        "condition": "any non-finite loss, gradient or logit",
        "default": None,
    },
    {
        "id": "exit1_bit_exact",
        "condition": "exit 1 (N=1, adapters off) is not bit-exactly O",
        "default": None,
    },
    {
        "id": "teacher_integrity",
        "condition": "teacher top-k+tail sums off by more than 2e-3, or wrong shapes/ids",
        "default": 2e-3,
    },
    {
        "id": "parent_kl_runaway",
        "condition": "KL(O || student exit primary N) > 1.0 nat/token for 3 consecutive probes",
        "default": 1.0,
    },
)

WARNING_RULES = (
    {
        "id": "loss_spike",
        "condition": "loss > 3x its rolling median for 20 consecutive steps",
        "default": 3.0,
    },
    {
        "id": "grad_spike",
        "condition": "gradient norm > 10x its rolling median",
        "default": 10.0,
    },
    {
        "id": "teacher_stall",
        "condition": "teacher KL improved by less than 5% over the last 1M tokens",
        "default": 0.05,
    },
    {
        "id": "hinge_active",
        "condition": "monotone hinge active on more than 50% of rows after 1M tokens",
        "default": 0.5,
    },
    {
        "id": "overfitting_gap",
        "condition": "monitor/train teacher-KL gap grew for 3 consecutive probes",
        "default": 3,
    },
    {
        "id": "generation_sanity",
        "condition": (
            "repetition loops at N <= 4, or a missing answer / open thought "
            "channel when the generation finished before the 512-token cap "
            "(a length-truncated 8192-token thinking trace is expected)"
        ),
        "default": None,
    },
    {
        "id": "bridge_saturated",
        "condition": "more than 10% of bridge channels with |g| > 0.5",
        "default": 0.1,
    },
    {
        "id": "bridge_zero",
        "condition": "bridge g is all zero after 1M tokens",
        "default": None,
    },
    {
        "id": "memory_margin",
        "condition": "reserved memory within 0.3 GiB of the allocator cap",
        "default": 0.3,
    },
    {
        "id": "throughput_spill",
        "condition": "tokens/s below 50% of the post-warm-up median",
        "default": 0.5,
    },
)

__all__ = [
    "ABORT_RULES",
    "HealMonitor",
    "HealMonitorCallback",
    "MONITOR_ABORT_EXIT_CODE",
    "MONITOR_ALERTS_FILE",
    "MONITOR_GENERATION_DIR",
    "MONITOR_METRICS_FILE",
    "MONITOR_PROBES_FILE",
    "MONITOR_SPLIT_NAME",
    "MonitorAlert",
    "MonitorConfig",
    "MonitorRules",
    "PRIMARY_N",
    "WARNING_RULES",
    "atomic_append_jsonl",
    "bridge_g_stats",
    "build_probe_batches",
    "build_prompt_tensors",
    "gradient_stats",
    "install_heal_monitor",
    "json_safe",
    "load_reserved_ids",
    "lora_b_norms",
    "rolling_median",
    "select_monitor_rows",
    "teacher_integrity",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` (strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def atomic_append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one line in a single ``O_APPEND`` write and flush it to disk.

    ``O_APPEND`` makes concurrent writers add whole lines rather than
    interleaving bytes on the platforms this project supports.  The monitor is
    the only writer in normal use; this keeps a stray ``heal_report.py
    --watch`` reader safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(dict(record)), ensure_ascii=False, sort_keys=True)
    data = (payload + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def rolling_median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(values) if values else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


# ---------------------------------------------------------------------------
# Configuration (the thresholds the prereg can cite)
# ---------------------------------------------------------------------------


@dataclass
class MonitorConfig:
    """All monitor thresholds, with the values registered by the brief."""

    # Probe schedule and split.
    monitor_rows: int = 32
    probe_interval_tokens: int = 500_000
    generation_prompts: int = 8
    generation_max_new_tokens: int = 512
    generation_batch_size: int = 4
    generation_ns: tuple[int, ...] = (1, 2, 4)
    anchor_ns: tuple[int, ...] = (2, 3, 4, 6, 8)
    nll_ns: tuple[int, ...] = (1, 2, 3, 4)
    probe_token_chunk: int = 1024
    probe_vocab_chunk: int = 16384
    # One probe row per forward micro-batch.  At the registered 8192-token row
    # length four rows in one forward reserve ~9.9 GiB on the 12 GiB card and
    # OOM at N=4/N=8; one row keeps the same metrics for ~6.7 GiB (headroom
    # 3.6 GiB).  Metrics aggregate token-weighted across micro-batches, so the
    # reported values are identical to one combined batch of four rows.
    probe_batch_size: int = 1
    nll_temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

    # Abort rules.
    primary_n: int = PRIMARY_N
    abort_exit_code: int = MONITOR_ABORT_EXIT_CODE
    teacher_sum_tolerance: float = 2e-3
    mask_token_id: int | None = 249788
    parent_kl_abort: float = 1.0
    parent_kl_consecutive_probes: int = 3
    probe_failure_abort_consecutive: int = 2

    # Warning rules.
    loss_spike_factor: float = 3.0
    loss_spike_steps: int = 20
    loss_median_window: int = 100
    loss_warmup_steps: int = 20
    grad_spike_factor: float = 10.0
    grad_median_window: int = 100
    grad_warmup_steps: int = 10
    teacher_stall_fraction: float = 0.05
    teacher_stall_tokens: int = 1_000_000
    hinge_active_fraction: float = 0.5
    hinge_after_tokens: int = 1_000_000
    gap_consecutive_probes: int = 3
    bridge_sat_fraction: float = 0.1
    bridge_sat_abs: float = 0.5
    bridge_active_abs: float = 1e-3
    bridge_zero_after_tokens: int = 1_000_000
    memory_margin_gib: float = 0.3
    throughput_drop_ratio: float = 0.5
    throughput_warmup_steps: int = 20
    throughput_window_steps: int = 50
    ngram: int = 8
    repetition_rate_threshold: float = 0.5
    repetition_max_count: int = 3
    warn_repeat_tokens: int = 1_000_000

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "monitor_split": MONITOR_SPLIT_NAME,
            "monitor_rows": int(self.monitor_rows),
            "probe_interval_tokens": int(self.probe_interval_tokens),
            "generation_prompts": int(self.generation_prompts),
            "generation_max_new_tokens": int(self.generation_max_new_tokens),
            "generation_batch_size": int(self.generation_batch_size),
            "probe_batch_size": int(self.probe_batch_size),
            "generation_ns": [int(n) for n in self.generation_ns],
            "anchor_ns": [int(n) for n in self.anchor_ns],
            "nll_ns": [int(n) for n in self.nll_ns],
            "nll_temperatures": [float(value) for value in self.nll_temperatures],
            "primary_n": int(self.primary_n),
            "abort_exit_code": int(self.abort_exit_code),
            "teacher_sum_tolerance": float(self.teacher_sum_tolerance),
            "mask_token_id": self.mask_token_id,
            "parent_kl_abort": float(self.parent_kl_abort),
            "parent_kl_consecutive_probes": int(
                self.parent_kl_consecutive_probes
            ),
            "probe_failure_abort_consecutive": int(
                self.probe_failure_abort_consecutive
            ),
            "loss_spike_factor": float(self.loss_spike_factor),
            "loss_spike_steps": int(self.loss_spike_steps),
            "grad_spike_factor": float(self.grad_spike_factor),
            "teacher_stall_fraction": float(self.teacher_stall_fraction),
            "teacher_stall_tokens": int(self.teacher_stall_tokens),
            "hinge_active_fraction": float(self.hinge_active_fraction),
            "hinge_after_tokens": int(self.hinge_after_tokens),
            "gap_consecutive_probes": int(self.gap_consecutive_probes),
            "bridge_sat_fraction": float(self.bridge_sat_fraction),
            "bridge_sat_abs": float(self.bridge_sat_abs),
            "bridge_active_abs": float(self.bridge_active_abs),
            "bridge_zero_after_tokens": int(self.bridge_zero_after_tokens),
            "memory_margin_gib": float(self.memory_margin_gib),
            "throughput_drop_ratio": float(self.throughput_drop_ratio),
            "throughput_warmup_steps": int(self.throughput_warmup_steps),
            "ngram": int(self.ngram),
            "repetition_rate_threshold": float(self.repetition_rate_threshold),
            "repetition_max_count": int(self.repetition_max_count),
        }
        return payload


# ---------------------------------------------------------------------------
# Alerts and rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorAlert:
    severity: str
    rule: str
    message: str
    step: int | None = None
    tokens_seen: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": str(self.severity),
            "rule": str(self.rule),
            "message": str(self.message),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "metrics": json_safe(self.metrics),
        }


class MonitorRules:
    """Stateful machine-checkable rules over the step and probe series."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._losses: deque[float] = deque(maxlen=int(self.config.loss_median_window))
        self._grad_norms: deque[float] = deque(
            maxlen=int(self.config.grad_median_window)
        )
        self._loss_spike_run = 0
        self._teacher_smoothed: deque[tuple[int, float]] = deque(maxlen=500)
        self._teacher_recent: deque[float] = deque(maxlen=20)
        self._throughput: deque[float] = deque(
            maxlen=int(self.config.throughput_window_steps)
        )
        self._parent_kl_run = 0
        self._probe_failure_run = 0
        self._gaps: deque[float] = deque(maxlen=int(self.config.gap_consecutive_probes))
        self._warn_last: dict[str, int] = {}
        self._abort_emitted: set[str] = set()
        self.abort_triggered = False
        self.steps_seen = 0
        self.probes_seen = 0

    # -- helpers -----------------------------------------------------------
    def _emit(
        self,
        severity: str,
        rule: str,
        message: str,
        *,
        step: int | None,
        tokens: int | None,
        metrics: Mapping[str, Any] | None = None,
    ) -> MonitorAlert | None:
        tokens_value = int(tokens or 0)
        if severity == "abort":
            if rule in self._abort_emitted:
                return None
            self._abort_emitted.add(rule)
            self.abort_triggered = True
        else:
            last = self._warn_last.get(rule)
            if last is not None and tokens_value - last < int(
                self.config.warn_repeat_tokens
            ):
                return None
            self._warn_last[rule] = tokens_value
        return MonitorAlert(
            severity=severity,
            rule=rule,
            message=message,
            step=None if step is None else int(step),
            tokens_seen=None if tokens is None else int(tokens),
            metrics=dict(metrics or {}),
        )

    # -- per step ----------------------------------------------------------
    def observe_step(self, record: Mapping[str, Any]) -> list[MonitorAlert]:
        self.steps_seen += 1
        alerts: list[MonitorAlert] = []
        step = record.get("step")
        tokens = record.get("tokens_seen")
        loss = record.get("loss")
        grad = record.get("grad_norm_total")
        nonfinite = record.get("nonfinite") or {}

        if any(int(value or 0) > 0 for value in nonfinite.values()):
            alert = self._emit(
                "abort",
                "nonfinite",
                "non-finite loss, gradient or logit detected",
                step=step,
                tokens=tokens,
                metrics={"nonfinite": nonfinite},
            )
            if alert:
                alerts.append(alert)

        integrity = record.get("teacher_integrity") or {}
        if integrity.get("ok") is False:
            alert = self._emit(
                "abort",
                "teacher_integrity",
                "teacher data failed integrity validation",
                step=step,
                tokens=tokens,
                metrics={"teacher_integrity": integrity},
            )
            if alert:
                alerts.append(alert)

        # Loss spike: > factor x the median of the preceding window, for
        # ``loss_spike_steps`` consecutive steps.
        if _finite(loss):
            value = float(loss)
            median = rolling_median(self._losses)
            if (
                median is not None
                and median > 0
                and len(self._losses) >= int(self.config.loss_warmup_steps)
            ):
                if value > float(self.config.loss_spike_factor) * median:
                    self._loss_spike_run += 1
                else:
                    self._loss_spike_run = 0
                if self._loss_spike_run >= int(self.config.loss_spike_steps):
                    alert = self._emit(
                        "warn",
                        "loss_spike",
                        (
                            f"loss spike: {value:.3f} > "
                            f"{float(self.config.loss_spike_factor):.1f}x "
                            f"rolling median {median:.3f} for "
                            f"{self._loss_spike_run} consecutive steps"
                        ),
                        step=step,
                        tokens=tokens,
                        metrics={"loss": value, "median": median},
                    )
                    if alert:
                        alerts.append(alert)
                        self._loss_spike_run = 0
            self._losses.append(value)

        # Gradient norm spike against its rolling median.
        if _finite(grad):
            value = float(grad)
            median = rolling_median(self._grad_norms)
            if (
                median is not None
                and median > 0
                and len(self._grad_norms) >= int(self.config.grad_warmup_steps)
                and value > float(self.config.grad_spike_factor) * median
            ):
                alert = self._emit(
                    "warn",
                    "grad_spike",
                    (
                        f"gradient norm {value:.3f} > "
                        f"{float(self.config.grad_spike_factor):.1f}x rolling "
                        f"median {median:.3f}"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"grad_norm_total": value, "median": median},
                )
                if alert:
                    alerts.append(alert)
            self._grad_norms.append(value)

        # Teacher KL stall over the last 1M tokens (smoothed to keep step
        # noise from firing the rule).
        primary_kl = record.get("primary_kl_teacher")
        if _finite(primary_kl):
            self._teacher_recent.append(float(primary_kl))
            smoothed = rolling_median(self._teacher_recent)
            if smoothed is not None and tokens is not None:
                current_tokens = int(tokens)
                self._teacher_smoothed.append((current_tokens, smoothed))
                cutoff = current_tokens - int(self.config.teacher_stall_tokens)
                older = [
                    (token, value)
                    for token, value in self._teacher_smoothed
                    if token <= cutoff
                ]
                if older:
                    old_tokens, old_value = older[-1]
                    if old_value > 0 and smoothed > (
                        1.0 - float(self.config.teacher_stall_fraction)
                    ) * old_value:
                        alert = self._emit(
                            "warn",
                            "teacher_stall",
                            (
                                "teacher KL has not improved by "
                                f"{100 * float(self.config.teacher_stall_fraction):.0f}% "
                                f"over {current_tokens - old_tokens} tokens "
                                f"({old_value:.3f} -> {smoothed:.3f})"
                            ),
                            step=step,
                            tokens=tokens,
                            metrics={
                                "teacher_kl": smoothed,
                                "reference": old_value,
                            },
                        )
                        if alert:
                            alerts.append(alert)

        if tokens is not None and int(tokens) >= int(self.config.hinge_after_tokens):
            active = record.get("hinge_active_max")
            if _finite(active) and float(active) > float(
                self.config.hinge_active_fraction
            ):
                alert = self._emit(
                    "warn",
                    "hinge_active",
                    (
                        f"monotone hinge active on {100 * float(active):.0f}% of "
                        f"rows (> {100 * float(self.config.hinge_active_fraction):.0f}% "
                        f"after {int(self.config.hinge_after_tokens)} tokens)"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"hinge_active_fraction": float(active)},
                )
                if alert:
                    alerts.append(alert)

        bridge = record.get("bridge_g") or {}
        if bridge.get("available"):
            # Saturation is an unconditional WARN (early detection); only the
            # "all zero after 1M tokens" condition is gated.
            saturation = float(bridge.get("saturation_fraction") or 0.0)
            if saturation > float(self.config.bridge_sat_fraction):
                alert = self._emit(
                    "warn",
                    "bridge_saturated",
                    (
                        f"bridge g saturation: {100 * saturation:.1f}% of "
                        f"channels with |g| > "
                        f"{float(self.config.bridge_sat_abs):.2f}"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"bridge_g": bridge},
                )
                if alert:
                    alerts.append(alert)
            if (
                tokens is not None
                and int(tokens) >= int(self.config.bridge_zero_after_tokens)
                and bool(bridge.get("all_zero"))
            ):
                alert = self._emit(
                    "warn",
                    "bridge_zero",
                    f"bridge g is all zero after {int(tokens)} tokens",
                    step=step,
                    tokens=tokens,
                    metrics={"bridge_g": bridge},
                )
                if alert:
                    alerts.append(alert)

        reserved = record.get("reserved_gib")
        cap = record.get("memory_cap_gib")
        if _finite(reserved) and _finite(cap):
            if float(reserved) >= float(cap) - float(self.config.memory_margin_gib):
                alert = self._emit(
                    "warn",
                    "memory_margin",
                    (
                        f"reserved memory {float(reserved):.2f} GiB is within "
                        f"{float(self.config.memory_margin_gib):.1f} GiB of the "
                        f"{float(cap):.2f} GiB cap"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"reserved_gib": float(reserved), "cap_gib": float(cap)},
                )
                if alert:
                    alerts.append(alert)

        rate = record.get("tok_per_s")
        if _finite(rate) and self.steps_seen > int(
            self.config.throughput_warmup_steps
        ):
            value = float(rate)
            median = rolling_median(self._throughput)
            if median is not None and median > 0 and value < float(
                self.config.throughput_drop_ratio
            ) * median:
                alert = self._emit(
                    "warn",
                    "throughput_spill",
                    (
                        f"throughput {value:.1f} tok/s < "
                        f"{100 * float(self.config.throughput_drop_ratio):.0f}% of "
                        f"the post-warm-up median {median:.1f} tok/s (spill?)"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"tok_per_s": value, "median": median},
                )
                if alert:
                    alerts.append(alert)
            self._throughput.append(value)

        return alerts

    # -- per probe ---------------------------------------------------------
    def observe_probe(self, probe: Mapping[str, Any]) -> list[MonitorAlert]:
        self.probes_seen += 1
        alerts: list[MonitorAlert] = []
        step = probe.get("step")
        tokens = probe.get("tokens_seen")

        invariant = probe.get("invariant") or {}
        if invariant.get("applicable") and invariant.get("ok") is False:
            alert = self._emit(
                "abort",
                "exit1_bit_exact",
                "exit 1 is not bit-exactly O on the fixed monitor probe",
                step=step,
                tokens=tokens,
                metrics={"invariant": invariant},
            )
            if alert:
                alerts.append(alert)

        for section in ("anchor_kl", "teacher_kl", "calibrated_nll"):
            for key, value in (probe.get(section) or {}).items():
                if value is not None and not _finite(value):
                    alert = self._emit(
                        "abort",
                        "nonfinite_probe",
                        f"non-finite probe metric {section}[{key}]",
                        step=step,
                        tokens=tokens,
                        metrics={"section": section, "key": key},
                    )
                    if alert:
                        alerts.append(alert)

        primary_key = str(int(self.config.primary_n))
        parent_kl = (probe.get("anchor_kl") or {}).get(primary_key)
        if _finite(parent_kl):
            if float(parent_kl) > float(self.config.parent_kl_abort):
                self._parent_kl_run += 1
            else:
                self._parent_kl_run = 0
            if self._parent_kl_run >= int(self.config.parent_kl_consecutive_probes):
                alert = self._emit(
                    "abort",
                    "parent_kl_runaway",
                    (
                        f"KL(O || student N={self.config.primary_n}) = "
                        f"{float(parent_kl):.3f} nat/token for "
                        f"{self._parent_kl_run} consecutive probes"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"parent_kl": float(parent_kl)},
                )
                if alert:
                    alerts.append(alert)
                    self._parent_kl_run = 0

        gap = (probe.get("teacher_kl_gap") or {}).get(primary_key)
        if _finite(gap):
            self._gaps.append(float(gap))
            if len(self._gaps) >= int(self.config.gap_consecutive_probes):
                values = list(self._gaps)
                if all(
                    values[index] > values[index - 1]
                    for index in range(1, len(values))
                ):
                    alert = self._emit(
                        "warn",
                        "overfitting_gap",
                        (
                            "monitor/train teacher-KL gap grew for "
                            f"{len(values)} consecutive probes "
                            f"(latest {values[-1]:.4f} nat)"
                        ),
                        step=step,
                        tokens=tokens,
                        metrics={"gap": values[-1]},
                    )
                    if alert:
                        alerts.append(alert)

        if probe.get("probe_failed"):
            self._probe_failure_run += 1
            detail = str(probe.get("error") or "unknown probe error")
            if self._probe_failure_run >= int(
                self.config.probe_failure_abort_consecutive
            ):
                alert = self._emit(
                    "abort",
                    "probe_failure",
                    (
                        "the monitor probe failed "
                        f"{self._probe_failure_run} consecutive times; the "
                        f"safety net is down ({detail})"
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"error": detail, "consecutive": self._probe_failure_run},
                )
                if alert:
                    alerts.append(alert)
                self._probe_failure_run = 0
            else:
                alert = self._emit(
                    "warn",
                    "probe_failure",
                    f"monitor probe failed ({detail}); retrying at the next probe point",
                    step=step,
                    tokens=tokens,
                    metrics={"error": detail, "consecutive": self._probe_failure_run},
                )
                if alert:
                    alerts.append(alert)
        else:
            self._probe_failure_run = 0

        for error in probe.get("probe_errors") or []:
            alert = self._emit(
                "warn",
                "probe_error",
                f"monitor probe failed: {error}",
                step=step,
                tokens=tokens,
                metrics={"error": str(error)},
            )
            if alert:
                alerts.append(alert)

        for row in probe.get("generation") or []:
            if int(row.get("n_loops", 99)) > 4:
                continue
            problems: list[str] = []
            if row.get("repetition_loop"):
                problems.append("repetition loop")
            # M1: with a 512-token generation cap on 8192-token thinking
            # traces, an unfinished (finish_reason == "length") completion
            # usually has no answer yet and its thought channel is still open.
            # That is expected at this cap, not an anomaly; only judge the
            # answer/thought-channel state on a generation that actually
            # stopped before the cap.
            truncated = str(row.get("finish_reason") or "") == "length"
            if not truncated:
                if row.get("answer_present") is False:
                    problems.append("missing answer")
                if row.get("thinking_closed") is False:
                    problems.append("open thinking channel")
            if problems:
                alert = self._emit(
                    "warn",
                    "generation_sanity",
                    (
                        f"generation sanity probe at N={row.get('n_loops')} "
                        f"prompt {row.get('prompt_index')}: "
                        + ", ".join(problems)
                    ),
                    step=step,
                    tokens=tokens,
                    metrics={"generation": row},
                )
                if alert:
                    alerts.append(alert)

        return alerts

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "loss_spike_run": int(self._loss_spike_run),
            "parent_kl_run": int(self._parent_kl_run),
            "probe_failure_run": int(self._probe_failure_run),
            "throughput": list(self._throughput),
            "losses": list(self._losses),
            "grad_norms": list(self._grad_norms),
            "teacher_smoothed": [list(item) for item in self._teacher_smoothed],
            "warn_last": dict(self._warn_last),
            "abort_emitted": sorted(self._abort_emitted),
            "abort_triggered": bool(self.abort_triggered),
        }

    def load_state_dict(
        self,
        payload: Mapping[str, Any] | None,
        *,
        restore_abort: bool = False,
    ) -> None:
        if not payload:
            return
        self._loss_spike_run = int(payload.get("loss_spike_run", 0) or 0)
        self._parent_kl_run = int(payload.get("parent_kl_run", 0) or 0)
        self._probe_failure_run = int(payload.get("probe_failure_run", 0) or 0)
        self._throughput = deque(
            [float(v) for v in payload.get("throughput", [])],
            maxlen=int(self.config.throughput_window_steps),
        )
        self._losses = deque(
            [float(v) for v in payload.get("losses", [])],
            maxlen=int(self.config.loss_median_window),
        )
        self._grad_norms = deque(
            [float(v) for v in payload.get("grad_norms", [])],
            maxlen=int(self.config.grad_median_window),
        )
        self._teacher_smoothed = deque(
            [(int(token), float(value)) for token, value in payload.get("teacher_smoothed", [])],
            maxlen=500,
        )
        self._warn_last = {str(k): int(v) for k, v in payload.get("warn_last", {}).items()}
        self._abort_emitted = {str(v) for v in payload.get("abort_emitted", [])}
        self.abort_triggered = bool(payload.get("abort_triggered", False))
        if not restore_abort:
            # A new process must re-observe the condition, not inherit the
            # previous attempt's terminal state.
            self._loss_spike_run = 0
            self._parent_kl_run = 0
            self._probe_failure_run = 0
            self._warn_last = {}
            self._gaps.clear()
            self._abort_emitted.clear()
            self.abort_triggered = False


# ---------------------------------------------------------------------------
# Monitor split
# ---------------------------------------------------------------------------


def _normalise_id(value: Any) -> str:
    return str(value)


def load_reserved_ids(paths: str | Iterable[str | Path] | None) -> set[str]:
    """Load reserved split identifiers from one or more JSON files.

    Accepted formats: a list of ids, ``{"ids": [...]}``,
    ``{"problem_ids": [...]}`` or a mapping from split name to a list.  The
    audit trail is deliberately simple so a controller can save the exact
    pilot/evaluation/T-fit split list next to the run.
    """
    if not paths:
        return set()
    if isinstance(paths, (str, Path)):
        raw = str(paths)
        if Path(raw).exists():
            candidates = [raw]
        else:
            candidates = [
                part
                for part in re.split(r"[;" + os.pathsep + r"]", raw)
                if part.strip()
            ]
    else:
        candidates = [str(path) for path in paths]
    reserved: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            raise FileNotFoundError(f"reserved split file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = []
            for key, item in payload.items():
                if isinstance(item, (list, tuple, set)):
                    values.extend(item)
                elif key in ("ids", "problem_ids") and isinstance(
                    item, (str, int)
                ):
                    values.append(item)
        else:
            raise ValueError(f"unsupported reserved split format in {path}")
        reserved.update(_normalise_id(value) for value in values)
    return reserved


def select_monitor_rows(
    row_ids: Sequence[Any],
    *,
    reserved: Iterable[Any] = (),
    count: int = 32,
    seed: int = 0,
) -> list[str]:
    """Deterministically choose ``count`` monitor rows disjoint from reserved ids.

    Ranking is ``sha256(f"{seed}:{row_id}")``; ties break on the string id.
    The result is stable across processes and independent of dataset order.
    """
    if int(count) <= 0:
        raise ValueError("count must be positive")
    reserved_ids = {_normalise_id(value) for value in reserved}
    available = sorted(
        {_normalise_id(value) for value in row_ids} - reserved_ids,
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    if len(available) < int(count):
        raise ValueError(
            f"the monitor split needs {int(count)} rows disjoint from "
            f"{len(reserved_ids)} reserved ids, but only {len(available)} are "
            "available"
        )
    return available[: int(count)]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _dist_stats(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": None, "min": None, "max": None, "p95": None}
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "min": None, "max": None, "p95": None}
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p95": _quantile(values.tolist(), 0.95),
    }


def teacher_integrity(
    teacher: TeacherTopK,
    vocab_size: int,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mask_token_id: int | None = None,
    ignore_index: int = -100,
    tolerance: float = 2e-3,
) -> dict[str, Any]:
    """Measure top-k + tail mass before/after fp32 renormalisation.

    Accepts the trainer's real unshifted ``[B, S, K]`` / ``[B, S]`` batch
    layout as well as the flattened ``[tokens, K]`` form used by the objective.
    ``masked_tokens`` and ``supervised_tokens`` are counted on the *next-token*
    positions (labels shifted by one), which is what the loss supervises.
    """
    raw_probs = teacher.probs
    raw_indices = teacher.indices
    raw_tail = teacher.tail_prob
    if raw_probs.ndim == 3:
        if raw_indices.shape != raw_probs.shape:
            return {
                "available": True,
                "ok": False,
                "error": (
                    "teacher probs and indices must have the same [B,S,K] "
                    f"shape, got {tuple(raw_probs.shape)} and "
                    f"{tuple(raw_indices.shape)}"
                ),
            }
        if raw_tail.ndim != 2 or raw_tail.shape != raw_probs.shape[:2]:
            return {
                "available": True,
                "ok": False,
                "error": (
                    "teacher tail_prob must have shape [B,S] = "
                    f"{tuple(raw_probs.shape[:2])}, got {tuple(raw_tail.shape)}"
                ),
            }
        k = int(raw_probs.shape[-1])
        probs = raw_probs.reshape(-1, k).to(torch.float32)
        indices = raw_indices.reshape(-1, k)
        tail = raw_tail.reshape(-1).to(torch.float32)
        layout = "batch_3d"
    elif raw_probs.ndim == 2:
        probs = raw_probs.reshape(-1, raw_probs.shape[-1]).to(torch.float32)
        indices = raw_indices.reshape(-1, raw_indices.shape[-1])
        tail = raw_tail.reshape(-1).to(torch.float32)
        layout = "flat_2d"
    else:
        return {
            "available": True,
            "ok": False,
            "error": (
                "teacher probs must be [B,S,K] or [tokens,K], got "
                f"{tuple(raw_probs.shape)}"
            ),
        }
    if probs.shape != indices.shape or tail.numel() != probs.shape[0]:
        return {
            "available": True,
            "ok": False,
            "error": (
                "teacher shapes disagree after flattening: probs "
                f"{tuple(probs.shape)}, indices {tuple(indices.shape)}, tail "
                f"{tuple(tail.shape)}"
            ),
        }
    before = probs.sum(dim=-1) + tail
    masked_tokens: int | None = None
    supervised_tokens: int | None = None
    if labels is not None:
        target_labels = (
            labels.reshape(labels.shape[0], -1)[:, 1:]
            if labels.ndim == 2
            else labels.reshape(-1)[1:]
        )
        masked_tokens = int((target_labels == int(ignore_index)).sum().item())
        supervised_tokens = int((target_labels != int(ignore_index)).sum().item())
    # Only real, supervised positions belong to the teacher contract.  Padding
    # rows are zero-filled by the collator (sum 0) and must never false-abort a
    # valid ragged batch.
    valid = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)
    if labels is not None and labels.ndim == 2:
        shifted_valid = torch.zeros_like(labels, dtype=torch.bool)
        shifted_valid[:, 1:] = labels[:, 1:] != int(ignore_index)
        if attention_mask is not None:
            shifted_valid[:, 1:] = shifted_valid[:, 1:] & (
                attention_mask[:, 1:].to(torch.bool)
            )
        flat_valid = shifted_valid.reshape(-1)
        if flat_valid.numel() == probs.shape[0]:
            valid = flat_valid.to(probs.device)
    elif attention_mask is not None:
        flat_attention = attention_mask.reshape(-1).to(probs.device)
        if flat_attention.numel() == probs.shape[0]:
            valid = flat_attention.bool()
    if not bool(valid.any()):
        return {
            "available": True,
            "ok": None,
            "layout": layout,
            "skipped": "no supervised teacher positions",
            "positions": int(probs.shape[0]),
            "valid_tokens": 0,
            "masked_tokens": masked_tokens,
            "supervised_tokens": supervised_tokens,
        }
    valid_before = before[valid]
    valid_indices = indices[valid]
    deviation = (valid_before - 1.0).abs()
    report: dict[str, Any] = {
        "available": True,
        "ok": None,
        "layout": layout,
        "tolerance": float(tolerance),
        "topk_mass": _dist_stats(valid_before - tail[valid]),
        "tail_mass": _dist_stats(tail[valid]),
        "sum_before": _dist_stats(valid_before),
        "abs_deviation_before": _dist_stats(deviation),
        "max_abs_deviation_before": float(deviation.max().item())
        if deviation.numel()
        else None,
        "tokens": int(valid.sum().item()),
        "positions": int(probs.shape[0]),
        "valid_tokens": int(valid.sum().item()),
        "masked_tokens": masked_tokens,
        "supervised_tokens": supervised_tokens,
    }
    if mask_token_id is not None:
        report["mask_token_count"] = int(
            (valid_indices == int(mask_token_id)).sum().item()
        )
    flat_teacher = TeacherTopK(
        probs=probs[valid],
        indices=indices[valid],
        tail_prob=tail[valid],
    )
    try:
        renorm = flat_teacher.renormalized(vocab_size, tolerance=float(tolerance))
    except Exception as exc:  # noqa: BLE001 - report the exact failure
        report["ok"] = False
        report["error"] = str(exc)
        return report
    after = renorm.probs.float().sum(dim=-1) + renorm.tail_prob.float()
    valid_after = after
    after_deviation = (valid_after - 1.0).abs()
    report.update(
        {
            "sum_after": _dist_stats(valid_after),
            "abs_deviation_after": _dist_stats(after_deviation),
            "max_abs_deviation_after": float(after_deviation.max().item())
            if after_deviation.numel()
            else None,
        }
    )
    before_max = report.get("max_abs_deviation_before")
    after_max = report.get("max_abs_deviation_after")
    report["ok"] = bool(
        before_max is not None
        and after_max is not None
        and float(before_max) <= float(tolerance)
        and float(after_max) <= float(tolerance)
    )
    return report


def bridge_g_stats(
    g: torch.Tensor | None,
    *,
    active_abs: float = 1e-3,
    saturation_abs: float = 0.5,
) -> dict[str, Any]:
    """Mean/max ``|g|``, active and saturated channel fractions."""
    if g is None:
        return {"available": False}
    values = g.detach().float().reshape(-1).abs()
    if values.numel() == 0:
        return {"available": True, "channels": 0}
    return {
        "available": True,
        "channels": int(values.numel()),
        "mean_abs": float(values.mean().item()),
        "max_abs": float(values.max().item()),
        "signed_mean": float(g.detach().float().mean().item()),
        "active_fraction": float(
            (values > float(active_abs)).float().mean().item()
        ),
        "saturation_fraction": float(
            (values > float(saturation_abs)).float().mean().item()
        ),
        "all_zero": bool((values == 0).all().item()),
    }


def lora_b_norms(model: torch.nn.Module) -> dict[str, float]:
    """Frobenius norm of every LoRA ``B`` matrix (per layer)."""
    layers: list[str] = []
    values: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if ".lora_B." not in name and not name.endswith(".lora_B.weight"):
            continue
        layers.append(
            name.replace(".lora_B.default.weight", "").replace(
                ".lora_B.weight", ""
            )
        )
        values.append(torch.linalg.vector_norm(parameter.detach().float()))
    if not values:
        return {}
    # One host sync for the whole per-layer table.
    return {
        layer: float(value)
        for layer, value in zip(layers, torch.stack(values).tolist())
    }


def _group_of(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("loop_bridge_logit", "loop_bridge_g"):
        return "bridge"
    if leaf == "loop_ple":
        return "loop_ple"
    if ".lora_" in name or name.endswith(".lora_A.weight") or name.endswith(
        ".lora_B.weight"
    ):
        return "lora"
    return "other"


def gradient_stats(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Pre-clipping gradient norm and first-order update/weight per group.

    Squared norms are accumulated on the parameter's own device and the host
    is synchronised once per group, not once per parameter; this keeps the
    per-step monitor overhead negligible next to a ~26k-token optimizer step.
    """
    grad_squares: dict[str, torch.Tensor] = {}
    weight_squares: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = _group_of(name)
        detached = parameter.detach().float()
        weight_square = detached.pow(2).sum()
        weight_squares[group] = weight_squares.get(
            group, torch.zeros((), device=detached.device)
        ) + weight_square
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        grad_square = grad.pow(2).sum()
        grad_squares[group] = grad_squares.get(
            group, torch.zeros((), device=grad.device)
        ) + grad_square
    out: dict[str, dict[str, float]] = {}
    for group in sorted(set(grad_squares) | set(weight_squares)):
        grad_norm = math.sqrt(float(grad_squares[group].item())) if group in grad_squares else 0.0
        weight_norm = (
            math.sqrt(float(weight_squares[group].item()))
            if group in weight_squares
            else 0.0
        )
        out[group] = {
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "grad_to_weight": grad_norm / weight_norm if weight_norm > 0 else 0.0,
        }
    return out


def _optimizer_lrs(optimizer: Any) -> dict[str, float]:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return {}
    out: dict[str, float] = {}
    for index, group in enumerate(groups):
        name = group.get("name") if isinstance(group, Mapping) else None
        key = str(name) if name else f"group{index}"
        lr = group.get("lr") if isinstance(group, Mapping) else None
        if _finite(lr):
            out[key] = float(lr)
    return out


def _memory_gib() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            float(torch.cuda.memory_allocated()) / 2**30,
            float(torch.cuda.memory_reserved()) / 2**30,
        )
    except Exception:  # pragma: no cover - CUDA query failure
        return None, None


def _peak_memory_gib() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            float(torch.cuda.max_memory_allocated()) / 2**30,
            float(torch.cuda.max_memory_reserved()) / 2**30,
        )
    except Exception:  # pragma: no cover
        return None, None


# ---------------------------------------------------------------------------
# Persistent monitor
# ---------------------------------------------------------------------------


class HealMonitor:
    """Owns the JSONL streams, alert list and rule state for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: MonitorConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or MonitorConfig()
        self.clock = clock
        self.rules = MonitorRules(self.config)
        self.metrics_path = self.run_dir / MONITOR_METRICS_FILE
        self.probes_path = self.run_dir / MONITOR_PROBES_FILE
        self.alerts_path = self.run_dir / MONITOR_ALERTS_FILE
        self.state_path = self.run_dir / MONITOR_STATE_FILE
        self.generation_dir = self.run_dir / MONITOR_GENERATION_DIR
        self.alerts: list[MonitorAlert] = []
        self._horizon_counts: Counter[int] = Counter()
        self._step_integrity: list[dict[str, Any]] = []
        self._last_dial_result: Any = None
        self._pending_grad_groups: dict[str, dict[str, float]] | None = None
        # Micro-batch -> optimizer-step aggregation.  Values are token-weighted
        # means except the ``*_tokens``/``*_rows`` counters, which are sums.
        self._step_weighted: dict[str, list[float]] = {}
        self._step_sums: dict[str, float] = {}
        self._last_horizon: int | None = None
        self._pending_nonfinite_logits = False
        self._pending_nonfinite_loss = False
        self._last_probe_tokens = 0
        self._last_step_clock: float | None = None
        self._last_tokens = 0
        self._load_state()
        # Preserve an existing alert list across a resume.
        if self.alerts_path.is_file():
            try:
                payload = json.loads(self.alerts_path.read_text(encoding="utf-8"))
                for entry in payload.get("alerts", []):
                    if isinstance(entry, dict):
                        self.alerts.append(
                            MonitorAlert(
                                severity=str(entry.get("severity", "warn")),
                                rule=str(entry.get("rule", "")),
                                message=str(entry.get("message", "")),
                                step=entry.get("step"),
                                tokens_seen=entry.get("tokens_seen"),
                                metrics=dict(entry.get("metrics") or {}),
                            )
                        )
            except (OSError, json.JSONDecodeError):
                pass

    # -- state -------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.rules.load_state_dict(payload.get("rules"))
        self._horizon_counts = Counter(
            {int(k): int(v) for k, v in (payload.get("horizon_counts") or {}).items()}
        )
        self._last_probe_tokens = int(payload.get("last_probe_tokens", 0) or 0)
        self._last_tokens = int(payload.get("last_tokens", 0) or 0)

    def save_state(self) -> Path:
        return atomic_write_json(
            self.state_path,
            {
                "rules": self.rules.state_dict(),
                "horizon_counts": {
                    str(key): int(value) for key, value in self._horizon_counts.items()
                },
                "last_probe_tokens": int(self._last_probe_tokens),
                "last_tokens": int(self._last_tokens),
                "updated_utc": _utc_iso(),
            },
        )

    # -- observation -------------------------------------------------------
    def observe_batch(self, inputs: Mapping[str, Any], model: torch.nn.Module) -> None:
        """Record teacher-data integrity for one training micro batch."""
        names = ("teacher_topk_probs", "teacher_topk_indices", "teacher_tail_prob")
        if not all(inputs.get(name) is not None for name in names):
            return
        try:
            weight = _head_weight(model)
            teacher = TeacherTopK(
                probs=inputs["teacher_topk_probs"],
                indices=inputs["teacher_topk_indices"],
                tail_prob=inputs["teacher_tail_prob"],
            )
            labels = inputs.get("labels")
            report = teacher_integrity(
                teacher,
                int(weight.shape[0]),
                labels=labels if isinstance(labels, torch.Tensor) else None,
                attention_mask=(
                    inputs.get("attention_mask")
                    if isinstance(inputs.get("attention_mask"), torch.Tensor)
                    else None
                ),
                mask_token_id=self.config.mask_token_id,
                tolerance=float(self.config.teacher_sum_tolerance),
            )
        except Exception as exc:  # noqa: BLE001 - malformed batch is an abort
            report = {
                "available": True,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._step_integrity.append(report)

    def observe_dial_result(self, result: Any) -> None:
        self._last_dial_result = result

    def observe_hidden_states(self, outputs: Mapping[str, Any]) -> None:
        """Flag non-finite exit hidden states (the source of non-finite logits)."""
        for hidden in (outputs.get("exit_hidden_states") or {}).values():
            if isinstance(hidden, torch.Tensor) and not bool(
                torch.isfinite(hidden).all().item()
            ):
                self._pending_nonfinite_logits = True
                return

    def _accumulate(self, key: str, value: Any, weight: float) -> None:
        if value is None:
            return
        if not _finite(value):
            if key in ("dial_total", "heal_loss", "loss"):
                self._pending_nonfinite_loss = True
            return
        numeric = float(value)
        if key.endswith("_tokens") or key.endswith("_rows") or key in {
            "dial_tokens",
            "dial_rows",
            "dial_token_sum",
        }:
            self._step_sums[key] = self._step_sums.get(key, 0.0) + numeric
            return
        entry = self._step_weighted.setdefault(key, [0.0, 0.0])
        entry[0] += numeric * float(weight)
        entry[1] += float(weight)

    def observe_microbatch(self, trainer: Any) -> None:
        """Aggregate one micro-batch into the optimizer-step metrics.

        ``trainer._last_dial_metrics`` / ``_last_heal_metrics`` are overwritten
        by every micro-batch, so reading them once at ``on_step_end`` would log
        only the last draw of a GA step.  This accumulates token-weighted means
        and counts every T draw; the loss-spike rule then sees the step value.
        """
        dial = dict(getattr(trainer, "_last_dial_metrics", {}) or {})
        if dial:
            weight = float(dial.get("dial_tokens") or dial.get("dial_rows") or 1.0)
            for key, value in dial.items():
                if key in ("dial_horizon",):
                    continue
                self._accumulate(key, value, weight)
            result = self._last_dial_result
            if result is not None and hasattr(result, "per_exit_hinge"):
                counts = getattr(result, "row_token_counts", None)
                if counts is not None:
                    valid = counts > 0
                    valid_count = float(valid.sum().item())
                    for exit_index, hinge in getattr(
                        result, "per_exit_hinge", {}
                    ).items():
                        if hinge is None or valid_count <= 0:
                            continue
                        fraction = float((hinge[valid] > 0).float().mean().item())
                        self._accumulate(
                            f"dial_exit{int(exit_index)}_hinge_active_fraction",
                            fraction,
                            valid_count,
                        )
            horizon = dial.get("dial_horizon")
            if horizon is not None:
                self._last_horizon = int(horizon)
                self._horizon_counts[int(horizon)] += 1
            self._last_dial_result = None
            return
        heal = dict(getattr(trainer, "_last_heal_metrics", {}) or {})
        if heal:
            weight = float(heal.get("heal_tokens") or 1.0)
            for key, value in heal.items():
                self._accumulate(key, value, weight)

    def _drain_step_metrics(self, trainer: Any) -> dict[str, float]:
        if not self._step_weighted and not self._step_sums:
            return {
                **dict(getattr(trainer, "_last_dial_metrics", {}) or {}),
                **dict(getattr(trainer, "_last_heal_metrics", {}) or {}),
            }
        values: dict[str, float] = {}
        for key, (total, weight) in self._step_weighted.items():
            if weight > 0:
                values[key] = total / weight
        for key, total in self._step_sums.items():
            values[key] = total
        self._step_weighted.clear()
        self._step_sums.clear()
        return values

    def capture_gradients(self, model: torch.nn.Module) -> None:
        """Snapshot per-group gradient norms while the grads are still alive.

        Hugging Face zeroes the gradients at the end of the optimizer step,
        before ``on_step_end``; ``on_optimizer_step`` is the last callback that
        still sees them.  Monitoring must not change the run, so capture there
        and drain the snapshot in :meth:`record_step`.
        """
        self._pending_grad_groups = gradient_stats(model)

    def _integrity_summary(self) -> dict[str, Any]:
        if not self._step_integrity:
            return {"available": False, "ok": None}
        reports = self._step_integrity
        ok_values = [report.get("ok") for report in reports]
        ok: bool | None
        if any(value is False for value in ok_values):
            ok = False
        elif any(value is True for value in ok_values):
            ok = True
        else:
            ok = None
        summaries = {
            key: [
                report[key]
                for report in reports
                if report.get(key) is not None
            ]
            for key in (
                "max_abs_deviation_before",
                "max_abs_deviation_after",
                "masked_tokens",
                "mask_token_count",
            )
        }
        return {
            "available": True,
            "ok": ok,
            "batches": len(reports),
            "max_abs_deviation_before": max(
                summaries["max_abs_deviation_before"], default=None
            ),
            "max_abs_deviation_after": max(
                summaries["max_abs_deviation_after"], default=None
            ),
            "masked_tokens": int(sum(summaries["masked_tokens"])),
            "mask_token_count": int(sum(summaries["mask_token_count"])),
            "tail_mass": _merge_stats(
                [report.get("tail_mass") for report in reports]
            ),
            "abs_deviation_before": _merge_stats(
                [report.get("abs_deviation_before") for report in reports]
            ),
            "errors": [
                report["error"] for report in reports if report.get("error")
            ],
        }

    def record_step(
        self,
        trainer: Any,
        *,
        step: int,
        tokens_seen: int,
        step_seconds: float | None,
        memory_cap_gib: float | None = None,
    ) -> dict[str, Any]:
        step_metrics = self._drain_step_metrics(trainer)
        dial = {
            key: value
            for key, value in step_metrics.items()
            if key.startswith("dial_")
        }
        heal = {
            key: value
            for key, value in step_metrics.items()
            if key.startswith("heal_")
        }
        exits = _exit_metrics(dial)
        result = self._last_dial_result
        if result is not None and hasattr(result, "per_exit_hinge"):
            for exit_index in exits:
                hinge = getattr(result, "per_exit_hinge", {}).get(exit_index)
                counts = getattr(result, "row_token_counts", None)
                if hinge is not None and counts is not None:
                    valid = counts > 0
                    if bool(valid.any()):
                        exits[exit_index]["hinge_active_fraction"] = float(
                            (hinge[valid] > 0).float().mean().item()
                        )
        loss = dial.get("dial_total")
        if loss is None:
            loss = heal.get("heal_loss")
        if loss is None:
            log_history = getattr(getattr(trainer, "state", None), "log_history", [])
            if log_history and "loss" in log_history[-1]:
                loss = log_history[-1]["loss"]

        model = getattr(trainer, "model", None)
        groups = (
            self._pending_grad_groups
            if self._pending_grad_groups is not None
            else (gradient_stats(model) if isinstance(model, torch.nn.Module) else {})
        )
        self._pending_grad_groups = None
        lrs = _optimizer_lrs(getattr(trainer, "optimizer", None))
        for group, values in groups.items():
            lr = _group_lr(group, lrs)
            if lr is not None:
                values["lr"] = lr
                values["update_to_weight"] = (
                    lr * values["grad_norm"] / values["weight_norm"]
                    if values["weight_norm"] > 0
                    else 0.0
                )
        bridge = _bridge_parameter(trainer)
        bridge_stats = bridge_g_stats(
            bridge,
            active_abs=float(self.config.bridge_active_abs),
            saturation_abs=float(self.config.bridge_sat_abs),
        )
        lora_norms = lora_b_norms(model) if isinstance(model, torch.nn.Module) else {}
        horizon = dial.get("dial_horizon", self._last_horizon)
        if horizon is not None and self._last_horizon is None:
            # Direct record_step use (tests, non-dial fallback): count the draw
            # here because no micro-batch callback saw it.
            self._horizon_counts[int(horizon)] += 1
        registered = _registered_horizon(trainer)
        histogram = {
            str(key): int(value) for key, value in sorted(self._horizon_counts.items())
        }
        total_draws = sum(self._horizon_counts.values())
        fractions = {
            str(key): float(value) / total_draws
            for key, value in sorted(self._horizon_counts.items())
        } if total_draws else {}

        allocated, reserved = _memory_gib()
        peak_allocated, peak_reserved = _peak_memory_gib()
        now = float(self.clock())
        if step_seconds is None:
            if self._last_step_clock is not None:
                step_seconds = max(1e-9, now - self._last_step_clock)
            else:
                step_seconds = 0.0
        delta_tokens = max(0, int(tokens_seen) - int(self._last_tokens))
        tok_per_s = float(delta_tokens) / step_seconds if step_seconds > 0 else 0.0
        self._last_step_clock = now
        self._last_tokens = int(tokens_seen)

        nonfinite = {
            "loss": (
                1
                if self._pending_nonfinite_loss
                or (loss is not None and not _finite(loss))
                else 0
            ),
            "grads": 0
            if getattr(trainer, "_last_grad_norm_total", None) is None
            or _finite(getattr(trainer, "_last_grad_norm_total"))
            else 1,
            "logits": (
                1
                if self._pending_nonfinite_logits
                or not all(
                    value is None or _finite(value)
                    for value in _flatten_metrics(dial)
                )
                else 0
            ),
        }
        self._pending_nonfinite_logits = False
        self._pending_nonfinite_loss = False
        primary_key = str(int(self.config.primary_n))
        primary = exits.get(int(self.config.primary_n), {})
        primary_kl_teacher = primary.get("kl_teacher")
        if primary_kl_teacher is None:
            values = [
                float(entry["kl_teacher"])
                for entry in exits.values()
                if entry.get("kl_teacher") is not None
            ]
            primary_kl_teacher = sum(values) / len(values) if values else None
        primary_kl_parent = primary.get("kl_parent")
        if primary_kl_parent is None:
            values = [
                float(entry["kl_parent"])
                for entry in exits.values()
                if entry.get("kl_parent") is not None
            ]
            primary_kl_parent = sum(values) / len(values) if values else None
        record = {
            "kind": "step",
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "loss": None if loss is None else float(loss),
            "horizon": None if horizon is None else int(horizon),
            "exits": exits,
            "grad_norm_total": getattr(trainer, "_last_grad_norm_total", None),
            "grad_groups": groups,
            "clip_fired": bool(getattr(trainer, "_last_clip_fired", False)),
            "clip_rate": float(getattr(trainer, "clip_rate", 0.0) or 0.0),
            "bridge_g": bridge_stats,
            "lora_b_norms": lora_norms,
            "lr": lrs,
            "supervised_tokens": _supervised_tokens(trainer),
            "horizon_histogram": histogram,
            "horizon_fractions": fractions,
            "registered_horizon": registered,
            "teacher_integrity": self._integrity_summary(),
            "nonfinite": nonfinite,
            "tok_per_s": tok_per_s,
            "step_seconds": float(step_seconds),
            "allocated_gib": allocated,
            "reserved_gib": reserved,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "memory_cap_gib": memory_cap_gib,
            "primary_n": int(self.config.primary_n),
            "primary_kl_teacher": primary_kl_teacher,
            "primary_kl_parent": primary_kl_parent,
            "hinge_active_max": _max_exit(exits, "hinge_active_fraction"),
            "recorded_utc": _utc_iso(),
        }
        self._last_horizon = None
        self._step_integrity = []
        self._last_dial_result = None
        atomic_append_jsonl(self.metrics_path, record)
        alerts = self.rules.observe_step(record)
        self._absorb(alerts)
        return record

    def record_probe(self, probe: Mapping[str, Any]) -> list[MonitorAlert]:
        atomic_append_jsonl(self.probes_path, probe)
        self._last_probe_tokens = int(probe.get("tokens_seen", 0) or 0)
        alerts = self.rules.observe_probe(probe)
        self._absorb(alerts)
        self.save_state()
        return alerts

    def _absorb(self, alerts: Sequence[MonitorAlert]) -> None:
        if not alerts:
            return
        self.alerts.extend(alerts)
        self.write_alerts()
        for alert in alerts:
            level = _LOG.error if alert.severity == "abort" else _LOG.warning
            level("heal monitor %s [%s]: %s", alert.severity, alert.rule, alert.message)

    def write_alerts(self) -> Path:
        return atomic_write_json(
            self.alerts_path,
            {
                "alerts": [alert.to_dict() for alert in self.alerts],
                "abort_triggered": bool(self.rules.abort_triggered),
                "updated_utc": _utc_iso(),
            },
        )

    def should_probe(self, tokens_seen: int) -> bool:
        return int(tokens_seen) - int(self._last_probe_tokens) >= int(
            self.config.probe_interval_tokens
        )

    def due_at_end(self, tokens_seen: int) -> bool:
        return int(tokens_seen) > int(self._last_probe_tokens)


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


class HealMonitorCallback(TrainerCallback):
    """Feed the trainer's state into :class:`HealMonitor` every step."""

    def __init__(
        self,
        monitor: HealMonitor,
        trainer: Any,
        *,
        probe_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        memory_cap_gib: float | None = None,
    ) -> None:
        self.monitor = monitor
        self.trainer = trainer
        self.probe_factory = probe_factory
        self.clock = clock
        self.memory_cap_gib = memory_cap_gib
        self._step_started = float(clock())

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, kwargs
        self._step_started = float(self.clock())
        self.monitor._last_tokens = int(getattr(self.trainer, "tokens_seen", 0))
        self.monitor._last_step_clock = float(self.clock())
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, state, kwargs
        self._step_started = float(self.clock())
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
        # Gradients are still populated here (HF zeroes them after this
        # callback); capture the per-group norms before they disappear.
        del args, state, kwargs
        model = getattr(self.trainer, "model", None)
        if (
            isinstance(model, torch.nn.Module)
            and self.monitor._pending_grad_groups is None
        ):
            # Clipping disabled: no earlier capture, and grads are still intact
            # here (HF zeroes them after on_optimizer_step).
            self.monitor.capture_gradients(model)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, kwargs
        elapsed = max(0.0, float(self.clock()) - self._step_started)
        record = self.monitor.record_step(
            self.trainer,
            step=int(state.global_step),
            tokens_seen=int(getattr(self.trainer, "tokens_seen", 0)),
            step_seconds=elapsed,
            memory_cap_gib=self.memory_cap_gib,
        )
        del record
        if self.monitor.rules.abort_triggered:
            return self._abort(control)
        if self.probe_factory is not None and self.monitor.should_probe(
            int(getattr(self.trainer, "tokens_seen", 0))
        ):
            return self._run_probe(control)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, kwargs
        if self.probe_factory is not None and self.monitor.due_at_end(
            int(getattr(self.trainer, "tokens_seen", 0))
        ):
            control = self._run_probe(control)
        self.monitor.save_state()
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, state, kwargs
        self.monitor.save_state()
        return control

    def _run_probe(self, control):  # noqa: ANN001
        try:
            runner = self.probe_factory()
            dial = dict(getattr(self.trainer, "_last_dial_metrics", {}) or {})
            train_kl = {
                key.split("_")[1][4:]: value
                for key, value in dial.items()
                if key.startswith("dial_exit") and key.endswith("_kl_teacher")
            }
            probe = runner.run(
                tokens_seen=int(getattr(self.trainer, "tokens_seen", 0)),
                step=int(getattr(getattr(self.trainer, "state", None), "global_step", 0)),
                train_teacher_kl=train_kl,
            )
            self.monitor.record_probe(probe)
        except Exception as exc:  # noqa: BLE001 - record, then fail closed on repeat
            tokens_seen = int(getattr(self.trainer, "tokens_seen", 0) or 0)
            step = int(
                getattr(getattr(self.trainer, "state", None), "global_step", 0) or 0
            )
            detail = f"{type(exc).__name__}: {exc}"
            _LOG.warning(
                "heal probe failed (training continues for now): %s", detail
            )
            # Persist an error probe row so the failure is visible in
            # heal_probes.jsonl and advances the probe cursor: a broken probe
            # is not retried at every subsequent optimizer step.
            self.monitor.record_probe(
                {
                    "kind": "probe",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "probe_failed": True,
                    "error": detail,
                    "invariant": {"applicable": False, "ok": None},
                    "anchor_kl": {},
                    "teacher_kl": {},
                    "teacher_fit_available": False,
                    "teacher_kl_gap": {},
                    "calibrated_nll": {},
                    "generation": [],
                    "probe_errors": [],
                }
            )
            if self.monitor.rules.abort_triggered:
                return self._abort(control)
            return control
        if self.monitor.rules.abort_triggered:
            return self._abort(control)
        return control

    def _abort(self, control):  # noqa: ANN001
        self.monitor.write_alerts()
        code = int(self.monitor.config.abort_exit_code)
        try:
            self.trainer.exit_code = code
            self.trainer.exit_reason = "heal-monitor abort"
        except Exception:  # pragma: no cover - defensive
            pass
        _LOG.error(
            "heal monitor abort: leaving the last COMPLETE checkpoint intact "
            "and exiting %d",
            code,
        )
        # Ruling: an abort never writes a checkpoint.  The anomalous optimizer
        # step may already have applied a non-finite update, and publishing it
        # would poison the newest COMPLETE checkpoint that auto-resume picks.
        control.should_save = False
        control.should_training_stop = True
        raise SystemExit(code)


# ---------------------------------------------------------------------------
# Trainer record helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _head_weight(model: torch.nn.Module) -> torch.Tensor:
    from gemmaloop.train.heal_probes import text_model_and_head

    return text_model_and_head(model)[1]


_SIZE_RE = re.compile(r"dial_exit(\d+)_(.+)")


def _exit_metrics(dial: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    exits: dict[int, dict[str, float]] = {}
    for key, value in dial.items():
        match = _SIZE_RE.match(key)
        if not match or value is None:
            continue
        index = int(match.group(1))
        if index < 2:
            continue
        if not _finite(value):
            continue
        exits.setdefault(index, {})[match.group(2)] = float(value)
    return exits


def _flatten_metrics(metrics: Mapping[str, Any]) -> Iterable[Any]:
    for value in metrics.values():
        if isinstance(value, Mapping):
            yield from _flatten_metrics(value)
        else:
            yield value


def _max_exit(exits: Mapping[int, Mapping[str, float]], key: str) -> float | None:
    values = [float(item[key]) for item in exits.values() if _finite(item.get(key))]
    return max(values) if values else None


def _merge_stats(stats: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    values = [item for item in stats if isinstance(item, Mapping)]
    if not values:
        return {"mean": None, "min": None, "max": None, "p95": None}
    keys = ("mean", "min", "max", "p95")
    out: dict[str, Any] = {}
    for key in keys:
        present = [float(item[key]) for item in values if _finite(item.get(key))]
        if not present:
            out[key] = None
        elif key == "mean":
            out[key] = sum(present) / len(present)
        elif key == "min":
            out[key] = min(present)
        elif key == "max":
            out[key] = max(present)
        else:
            out[key] = _quantile(present, 0.95)
    return out


def _group_lr(group: str, lrs: Mapping[str, float]) -> float | None:
    if group == "bridge":
        return lrs.get("bridge")
    if group == "loop_ple":
        return lrs.get("loop_ple")
    for key in ("lora_decay", "lora_no_decay"):
        if key in lrs:
            return lrs[key]
    return next(iter(lrs.values()), None)


def _bridge_parameter(trainer: Any) -> torch.Tensor | None:
    text_model = getattr(trainer, "loop_text_model", None)
    if text_model is None:
        return None
    parameter = getattr(text_model, "loop_bridge_g", None)
    if parameter is not None:
        return parameter
    return None


def _registered_horizon(trainer: Any) -> dict[str, float]:
    probs = list(getattr(trainer, "dial_horizon_probs", ()) or ())
    return {
        str(index + 1): float(prob)
        for index, prob in enumerate(probs)
        if _finite(prob) and float(prob) > 0
    }


def _supervised_tokens(trainer: Any) -> int | None:
    values = getattr(trainer, "_supervised_tokens_per_step", None)
    if values:
        return int(values[-1])
    return None


# ---------------------------------------------------------------------------
# Probe construction and install hook
# ---------------------------------------------------------------------------


def _pad_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids, attention, labels = [], [], []
    for row in rows:
        ids = list(row["input_ids"])
        mask = list(row.get("attention_mask") or [1] * len(ids))
        lab = list(row["labels"])
        pad = max_length - len(ids)
        input_ids.append(ids + [int(pad_token_id)] * pad)
        attention.append(mask + [0] * pad)
        labels.append(lab + [-100] * pad)
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }
    # Teacher fields are optional and only present in smoke/embedded batches.
    rows_with_teacher = [
        row
        for row in rows
        if all(
            row.get(name) is not None
            for name in ("teacher_topk_probs", "teacher_topk_indices", "teacher_tail_prob")
        )
    ]
    if len(rows_with_teacher) == len(rows):
        k = len(rows[0]["teacher_topk_probs"][0])
        probs, indices, tails = [], [], []
        for row in rows:
            pad = max_length - len(row["input_ids"])
            probs.append(list(row["teacher_topk_probs"]) + [[0.0] * k for _ in range(pad)])
            indices.append(list(row["teacher_topk_indices"]) + [[0] * k for _ in range(pad)])
            tails.append(list(row["teacher_tail_prob"]) + [0.0] * pad)
        batch["teacher_topk_probs"] = torch.tensor(
            probs, dtype=torch.float16, device=device
        )
        batch["teacher_topk_indices"] = torch.tensor(
            indices, dtype=torch.int32, device=device
        )
        batch["teacher_tail_prob"] = torch.tensor(
            tails, dtype=torch.float32, device=device
        )
    return batch


def split_monitor_halves(
    indices: Sequence[int],
    *,
    seed: int,
    keys: Mapping[int, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Deterministically split MONITOR rows into fit and score halves."""
    values = list(indices)
    if len(values) < 2:
        return values, values

    def sort_key(index: int) -> tuple[str, int]:
        key = str(keys[index]) if keys is not None and index in keys else str(index)
        digest = hashlib.sha256(f"{int(seed)}:nll:{key}".encode("utf-8")).hexdigest()
        return digest, int(index)

    ranked = sorted(values, key=sort_key)
    half = max(1, len(ranked) // 2)
    fit = ranked[:half]
    score = ranked[half:]
    if not score:
        _LOG.warning(
            "MONITOR split has fewer than two rows; NLL fit and score use the "
            "same half"
        )
        score = fit
    return fit, score


def build_probe_batches(
    dataset: Any,
    indices: Sequence[int],
    *,
    model: torch.nn.Module,
    tokenizer: Any = None,
    data_collator: Callable[..., Any] | None = None,
    batch_size: int = 1,
    on_collator_error: Callable[[Exception], None] | None = None,
) -> list[dict[str, Any]]:
    """Collate the monitor rows into fixed probe batches.

    ``batch_size`` is the probe *forward* micro-batch size, not the number of
    probe rows: the registered rows are 8192 tokens long and four of them do not
    fit in one forward on a 12 GiB card.  One row per micro-batch keeps the
    probe metrics identical because ``ProbeRunner`` aggregates every per-token
    statistic over all rows token-weighted.

    A collator failure (for example a MONITOR teacher row that no longer
    aligns) falls back to a teacher-free batch, but never silently: it emits a
    WARN and calls ``on_collator_error`` so the split output can record that
    teacher-fit KL is unavailable.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    device = model_compute_device(model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    batches: list[dict[str, Any]] = []
    indices = list(indices)
    for start in range(0, len(indices), int(batch_size)):
        chunk = indices[start : start + int(batch_size)]
        if data_collator is not None:
            rows = [
                {name: dataset[int(index)][name] for name in dataset.column_names}
                for index in chunk
            ]
            try:
                batch = data_collator(rows)
                batches.append(
                    {
                        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                        for key, value in batch.items()
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001 - fall back to a plain batch
                _LOG.warning(
                    "MONITOR teacher collator failed on probe rows %s: %s; "
                    "falling back to a teacher-free batch, teacher-fit KL is "
                    "unavailable",
                    list(chunk),
                    exc,
                )
                if on_collator_error is not None:
                    on_collator_error(exc)
        rows = [dataset[int(index)] for index in chunk]
        batches.append(_pad_batch(rows, pad_token_id=int(pad_id), device=device))
    return batches


def _mark_teacher_fit_unavailable(run_dir: str | Path, exc: Exception) -> None:
    """Record a failed MONITOR teacher collator in the split output (N5)."""

    path = Path(run_dir) / "heal_monitor_split.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if payload.get("teacher_fit") == "unavailable":
        return
    payload["teacher_fit"] = "unavailable"
    payload["teacher_fit_error"] = f"{type(exc).__name__}: {exc}"
    atomic_write_json(path, payload)


def build_prompt_tensors(
    dataset: Any,
    indices: Sequence[int],
    *,
    device: torch.device,
    count: int = 8,
) -> list[torch.Tensor]:
    """Prompt-only prefixes of monitor rows (up to the first supervised token)."""
    prompts: list[torch.Tensor] = []
    for index in indices:
        row = dataset[int(index)]
        ids = [int(value) for value in row["input_ids"]]
        labels = row["labels"]
        first = next(
            (position for position, value in enumerate(labels) if int(value) != -100),
            None,
        )
        if first is None or first < 1:
            continue
        prompts.append(torch.tensor(ids[:first], dtype=torch.long, device=device))
        if len(prompts) >= int(count):
            break
    return prompts


_TRAINING_KEYS = {"training_ids", "train_ids", "training_rows", "train_rows"}


def is_registered_run(args: Any) -> bool:
    """True for a prereg-bound or pilot-arm run (where disjointness is mandatory)."""
    prereg = getattr(args, "prereg", None)
    pilot_arm = getattr(args, "pilot_arm", None)
    # A smoke run is explicitly not a registered run; the fail-closed split
    # rule applies to real --prereg/--pilot-arm executions.
    return bool(prereg is not None or pilot_arm) and not bool(
        getattr(args, "smoke", False)
    )


def _split_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, (str, int)):
        return [value]
    return []


def load_registered_splits(
    args: Any,
    *,
    env_paths: str | Iterable[str | Path] | None = None,
) -> tuple[set[str], set[str], set[str], list[str]]:
    """Read ``reserved_ids.json`` and related manifests.

    The registered contract is a map of split name -> ids, e.g.
    ``{"monitor": [...], "pilot_eval": [...], "pilot_fit": [...],
    "heal_train": [...], "confirmatory": [...]}``; unknown extra keys are
    allowed and treated as reserved.  The legacy
    ``{"reserved_ids": [...], "splits": {...}}`` shape is still accepted, with
    the monitor split subtracted from the legacy union.

    Returns ``(reserved, training, monitor, sources)``.  The monitor split's
    own ids are returned separately and are never part of the forbidden
    reserved union.
    """
    candidates: list[Path] = []

    def add(value: Any) -> None:
        if not value:
            return
        if isinstance(value, (str, Path)):
            text = str(value)
            # A single existing path wins over platform separator splitting.
            if Path(text).exists():
                candidates.append(Path(text))
            else:
                for part in re.split(r"[;" + re.escape(os.pathsep) + r"]", text):
                    if part.strip():
                        candidates.append(Path(part.strip()))
        else:
            for item in value:
                add(item)

    add(env_paths)
    for name in ("monitor_reserved", "pilot_manifest", "pilot_splits", "splits"):
        add(getattr(args, name, None))
    data = getattr(args, "data", None)
    if data:
        base = Path(data) if Path(data).is_dir() else Path(data).parent
        for name in (
            "splits.json",
            "reserved_ids.json",
            "monitor_splits.json",
            "manifest.json",
        ):
            path = base / name
            if path.is_file():
                candidates.append(path)

    reserved: set[str] = set()
    training: set[str] = set()
    monitor: set[str] = set()
    legacy_reserved: set[str] = set()
    seen: set[Path] = set()
    sources: list[str] = []

    def absorb(name: str, value: Any) -> None:
        key = str(name).strip().lower()
        values = {_normalise_id(item) for item in _split_values(value)}
        if key in {"monitor", "monitor_ids"}:
            monitor.update(values)
        elif key in _TRAINING_KEYS or key in {"heal", "heal_train", "train"}:
            training.update(values)
        else:
            # Unknown extra split names are allowed and are forbidden for the
            # monitor; they are part of the reserved proof.
            reserved.update(values)

    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        if isinstance(payload, list):
            reserved.update(_normalise_id(value) for value in payload)
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"unsupported split manifest format in {path}")
        for key, value in payload.items():
            key_lower = str(key).strip().lower()
            if key_lower == "splits" and isinstance(value, dict):
                for split_name, split_value in value.items():
                    absorb(split_name, split_value)
            elif key_lower == "reserved_ids":
                legacy_reserved.update(
                    _normalise_id(item) for item in _split_values(value)
                )
            elif key_lower in _TRAINING_KEYS:
                training.update(
                    _normalise_id(item) for item in _split_values(value)
                )
            elif isinstance(value, (list, tuple, set)):
                absorb(key, value)
            elif key_lower in {"monitor", "monitor_ids"}:
                monitor.update(
                    _normalise_id(item) for item in _split_values(value)
                )
            else:
                # Scalar metadata (seed, sha, ...) is not an id list.
                continue
    # Legacy unions often include the monitor and training splits; remove
    # them so the monitor's own ids never count as an overlap.
    reserved.update(legacy_reserved - monitor - training)
    return reserved, training, monitor, sources

def _explicit_row_ids(row: Mapping[str, Any]) -> set[str]:
    """Every id field a row carries (the producer collects all of them)."""

    values: set[str] = set()
    for key in (
        "problem_id",
        "cluster_id",
        "problem",
        "task_id",
        "request_id",
        "id",
        "source_id",
    ):
        value = row.get(key)
        if value is not None:
            values.add(_normalise_id(value))
    return values


def _explicit_row_id(row: Mapping[str, Any]) -> str | None:
    ids = _explicit_row_ids(row)
    return next(iter(sorted(ids))) if ids else None


def _monitor_rows_source(path: str | Path) -> Path:
    """The monitor JSONL actually selected for a file-or-directory argument."""
    source = Path(path)
    if source.is_dir():
        for name in ("monitor_rows.jsonl", "monitor.jsonl"):
            candidate = source / name
            if candidate.is_file():
                return candidate
    return source


def load_monitor_rows_file(path: str | Path) -> list[dict[str, Any]]:
    """Read a heal-format monitor JSONL (``messages`` rows)."""
    path = _monitor_rows_source(path)
    if not path.is_file():
        raise FileNotFoundError(f"monitor rows file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("messages"):
            raise ValueError(
                f"{path}:{line_number}: expected a heal-format row with 'messages'"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"monitor rows file is empty: {path}")
    return rows


def tokenize_monitor_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    raw_indices: Sequence[int],
    *,
    max_length: int,
    include_teacher: bool = True,
) -> Any:
    """Tokenize selected raw rows into a probe dataset preserving ``row_id``."""
    from datasets import Dataset

    from gemmaloop.train.labels import assistant_labels
    from gemmaloop.train.teacher_collator import TEACHER_FIELDS

    records: list[dict[str, Any]] = []
    for raw_index in raw_indices:
        row = rows[int(raw_index)]
        output = assistant_labels(
            tokenizer, row["messages"], int(max_length), tools=row.get("tools")
        )
        if not output["input_ids"] or all(
            label == -100 for label in output["labels"]
        ):
            continue
        record: dict[str, Any] = dict(output)
        record["row_id"] = int(raw_index)
        if include_teacher and all(
            field in row and row[field] is not None for field in TEACHER_FIELDS
        ):
            length = len(output["input_ids"])
            for field in TEACHER_FIELDS:
                value = row[field]
                if len(value) < length:
                    raise ValueError(
                        f"{field} has {len(value)} positions but the monitor row "
                        f"has {length}"
                    )
                record[field] = value[:length]
            for field in ("teacher_manifest_sha256", "teacher_id", "teacher_revision"):
                if row.get(field) is not None:
                    record[field] = row[field]
        records.append(record)
    if not records:
        raise ValueError("the monitor split has no tokenizable rows")
    return Dataset.from_list(records)


def _monitor_indices(
    dataset: Any,
    *,
    count: int,
    seed: int,
    reserved: Iterable[Any] = (),
) -> list[int]:
    if "row_id" in getattr(dataset, "column_names", []):
        ids = [str(value) for value in dataset["row_id"]]
    else:
        ids = [str(index) for index in range(len(dataset))]
    chosen = set(select_monitor_rows(ids, reserved=reserved, count=count, seed=seed))
    # Preserve dataset order for a stable probe batch.
    return [index for index, value in enumerate(ids) if value in chosen]


def install_heal_monitor(
    trainer: Any,
    args: Any,
    *,
    dataset: Any = None,
    tokenizer: Any = None,
    data_collator: Callable[..., Any] | None = None,
    monitor_data_collator: Callable[..., Any] | None = None,
    monitor_teacher_index: Any = None,
    rows: Any = None,
    max_length: int | None = None,
    enabled: bool | None = None,
    config: MonitorConfig | None = None,
) -> HealMonitorCallback | None:
    """The single ``scripts/train.py`` hook: install the monitor callback.

    Monitoring is on by default for dial/heal runs (not smoke runs) and can be
    forced on/off with ``GEMMALOOP_HEAL_MONITOR=1|0``.  The monitor split is
    selected deterministically and excludes any ids listed in the JSON file
    named by ``GEMMALOOP_HEAL_MONITOR_RESERVED``.  Monitoring off touches
    nothing and therefore reproduces training bit-exactly.
    """
    if enabled is None:
        explicit = os.environ.get("GEMMALOOP_HEAL_MONITOR")
        if explicit is not None:
            enabled = explicit.strip().lower() not in ("0", "false", "no", "off", "")
        else:
            heal_run = bool(
                getattr(trainer, "dial_enabled", False)
                or getattr(trainer, "heal_teacher", False)
            )
            enabled = bool(heal_run and not getattr(args, "smoke", False))
    if not enabled or dataset is None:
        return None

    config = config or MonitorConfig()
    registered = is_registered_run(args)
    reserved_path = os.environ.get("GEMMALOOP_HEAL_MONITOR_RESERVED")
    reserved_sources: list[str] = []
    manifest_training: set[str] = set()
    manifest_monitor: set[str] = set()
    try:
        (
            reserved,
            manifest_training,
            manifest_monitor,
            reserved_sources,
        ) = load_registered_splits(args, env_paths=reserved_path)
    except Exception as exc:  # noqa: BLE001 - a bad split manifest is fail-closed
        _LOG.error("heal monitor split manifest could not be read: %s", exc)
        if registered:
            raise SystemExit(2) from exc
        reserved = set()
    if not reserved:
        message = (
            "heal monitor: no reserved split ids were supplied; the MONITOR "
            "split cannot be proven disjoint from the pilot/evaluation/T-fit/"
            "confirmatory sets"
        )
        if registered:
            _LOG.error("%s; refusing the registered run (exit 2)", message)
            raise SystemExit(2)
        if not getattr(args, "smoke", False):
            _LOG.warning(message)

    probe_dataset = dataset
    probe_collator = data_collator
    monitor_collator = monitor_data_collator
    indices: list[int] = []
    split_ids: list[str] = []
    monitor_problem_ids: set[str] = set()
    monitor_rows_path = getattr(args, "monitor_rows", None)
    probe_source = "training_dataset"
    loaded_monitor_rows: list[dict[str, Any]] | None = None
    resolved_max_length = max_length
    if resolved_max_length is None:
        resolved_max_length = getattr(
            getattr(trainer, "args", None), "max_length", None
        )

    if monitor_rows_path:
        try:
            loaded_monitor_rows = load_monitor_rows_file(monitor_rows_path)
        except Exception as exc:  # noqa: BLE001 - explicit corpus must be valid
            _LOG.error("heal monitor rows could not be read: %s (exit 2)", exc)
            raise SystemExit(2) from exc
        if loaded_monitor_rows is not None:
            explicit_ids = [_explicit_row_ids(row) for row in loaded_monitor_rows]
            if any(not values for values in explicit_ids):
                message = (
                    "every monitor row must carry a problem/id field to prove "
                    "disjointness by id"
                )
                if registered:
                    _LOG.error("%s (exit 2)", message)
                    raise SystemExit(2)
                _LOG.warning("%s; disabling the monitor", message)
                return None
            monitor_problem_ids = {value for values in explicit_ids for value in values}
            training_identities: set[str] = set()
            if rows is not None:
                explicit_training = [_explicit_row_ids(row) for row in rows]
                training_identities = {
                    value for values in explicit_training for value in values
                }
                if registered and (
                    not training_identities
                    or any(not values for values in explicit_training)
                ):
                    _LOG.error(
                        "heal monitor: training rows do not all carry problem ids; "
                        "cannot prove the MONITOR split is disjoint (exit 2)"
                    )
                    raise SystemExit(2)
            elif registered:
                _LOG.error(
                    "heal monitor: a registered run needs the raw training rows "
                    "to prove MONITOR disjointness by id (exit 2)"
                )
                raise SystemExit(2)
            training_identities.update(manifest_training)
            if manifest_monitor and monitor_problem_ids != manifest_monitor:
                message = (
                    "the MONITOR rows do not match the manifest's monitor split "
                    "(rows: %d, manifest: %d)" % (
                        len(monitor_problem_ids),
                        len(manifest_monitor),
                    )
                )
                if registered:
                    _LOG.error("%s (exit 2)", message)
                    raise SystemExit(2)
                _LOG.warning("%s; disabling the monitor", message)
                return None
            overlap = monitor_problem_ids & (training_identities | reserved)
            if overlap:
                message = (
                    "the MONITOR rows overlap training/reserved ids: "
                    + ", ".join(sorted(overlap)[:8])
                )
                if registered:
                    _LOG.error("%s (exit 2)", message)
                    raise SystemExit(2)
                _LOG.warning("%s; disabling the monitor", message)
                return None
            if tokenizer is None or resolved_max_length is None:
                _LOG.error(
                    "heal monitor: --monitor-rows needs the tokenizer and "
                    "max_length to tokenize the monitor corpus"
                )
                if registered:
                    raise SystemExit(2)
                return None
            try:
                probe_dataset = tokenize_monitor_rows(
                    tokenizer,
                    loaded_monitor_rows,
                    list(range(len(loaded_monitor_rows))),
                    max_length=int(resolved_max_length),
                )
            except Exception as exc:  # noqa: BLE001 - fail closed for registered
                _LOG.error("heal monitor: cannot tokenize the MONITOR rows: %s", exc)
                if registered:
                    raise SystemExit(2) from exc
                return None
            if monitor_teacher_index is not None:
                from gemmaloop.data.teacher_topk import validate_teacher_alignment

                try:
                    validate_teacher_alignment(
                        monitor_teacher_index,
                        [probe_dataset[index] for index in range(len(probe_dataset))],
                    )
                except Exception as exc:  # noqa: BLE001 - teacher corpus mismatch
                    _LOG.error(
                        "heal monitor: monitor teacher data does not align with "
                        "the MONITOR rows: %s",
                        exc,
                    )
                    if registered:
                        raise SystemExit(2) from exc
                    return None
            probe_collator = monitor_collator
            try:
                indices = _monitor_indices(
                    probe_dataset,
                    count=int(config.monitor_rows),
                    seed=int(getattr(args, "seed", 0) or 0),
                    reserved=set(),
                )
            except ValueError as exc:
                _LOG.error("heal monitor: %s (exit 2)", exc)
                raise SystemExit(2) from exc
            split_ids = [
                str(
                    _explicit_row_id(
                        loaded_monitor_rows[int(probe_dataset[index]["row_id"])]
                    )
                )
                for index in indices
            ]
            probe_source = "monitor_rows_file"
            if len(indices) < int(config.monitor_rows):
                _LOG.error(
                    "heal monitor: the MONITOR corpus produced %d probe rows, "
                    "need at least %d (exit 2)",
                    len(indices),
                    int(config.monitor_rows),
                )
                raise SystemExit(2)

    if probe_source == "training_dataset":
        if registered:
            _LOG.error(
                "heal monitor: a registered run requires --monitor-rows "
                "<jsonl> from the dedicated Stage 0 monitor bucket (exit 2)"
            )
            raise SystemExit(2)
        try:
            indices = _monitor_indices(
                dataset,
                count=int(config.monitor_rows),
                seed=int(getattr(args, "seed", 0) or 0),
                reserved=reserved,
            )
            split_ids = [
                str(dataset[int(index)]["row_id"])
                if "row_id" in getattr(dataset, "column_names", [])
                else str(index)
                for index in indices
            ]
        except ValueError as exc:
            _LOG.error("heal monitor disabled: %s", exc)
            return None
    if not indices:
        if registered:
            raise SystemExit(2)
        _LOG.error("heal monitor disabled: the monitor split is empty")
        return None

    run_dir = Path(getattr(args, "out", getattr(trainer.args, "output_dir", ".")))
    monitor = HealMonitor(run_dir, config=config)
    monitor_rows_sha256 = None
    monitor_manifest_rows_sha256 = None
    monitor_manifest_sha256_matches = None
    if monitor_rows_path:
        monitor_file = _monitor_rows_source(monitor_rows_path)
        try:
            monitor_rows_sha256 = hashlib.sha256(monitor_file.read_bytes()).hexdigest()
        except OSError:
            monitor_rows_sha256 = None
        manifest_path = monitor_file.parent / "manifest.json"
        if manifest_path.is_file():
            try:
                monitor_manifest_rows_sha256 = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("sha256")
            except (OSError, json.JSONDecodeError):
                monitor_manifest_rows_sha256 = None
        if monitor_rows_sha256 and monitor_manifest_rows_sha256:
            monitor_manifest_sha256_matches = (
                monitor_rows_sha256 == monitor_manifest_rows_sha256
            )
    atomic_write_json(
        run_dir / "heal_monitor_split.json",
        {
            "split": MONITOR_SPLIT_NAME,
            "seed": int(getattr(args, "seed", 0) or 0),
            "count": len(split_ids),
            "problem_ids": split_ids,
            "row_ids": [int(index) for index in indices],
            "monitor_rows_path": (
                None if monitor_rows_path is None else str(monitor_rows_path)
            ),
            "monitor_rows_sha256": monitor_rows_sha256,
            "monitor_manifest_rows_sha256": monitor_manifest_rows_sha256,
            "monitor_manifest_sha256_matches": monitor_manifest_sha256_matches,
            "monitor_problem_ids": sorted(monitor_problem_ids),
            "monitor_row_indices": [int(index) for index in indices],
            "monitor_teacher_data": (
                None
                if getattr(args, "monitor_teacher_data", None) is None
                else str(getattr(args, "monitor_teacher_data"))
            ),
            "teacher_fit": "available" if monitor_collator is not None else "unavailable",
            "reserved_file": reserved_path,
            "reserved_sources": reserved_sources,
            "reserved_count": len(reserved),
            "manifest_training_count": len(manifest_training),
            "probe_source": probe_source,
            "registered": bool(registered),
            "config": config.to_dict(),
        },
    )
    device = model_compute_device(trainer.model)
    text_dir = run_dir / MONITOR_GENERATION_DIR
    batches: list[dict[str, Any]] = []
    prompts: list[torch.Tensor] = []
    nll_fit_batches: list[int] = []
    nll_score_batches: list[int] = []

    def _probe_factory():
        from gemmaloop.train.heal_probes import ProbeConfig, ProbeRunner, text_model_and_head

        nonlocal batches, prompts, nll_fit_batches, nll_score_batches
        if not batches:
            keys = {
                index: (
                    probe_dataset[index]["row_id"]
                    if "row_id" in getattr(probe_dataset, "column_names", [])
                    else index
                )
                for index in indices
            }
            fit_indices, score_indices = split_monitor_halves(
                indices,
                seed=int(getattr(args, "seed", 0) or 0),
                keys=keys,
            )
            # Both halves use the MONITOR collator (or none for a teacher-free
            # monitor corpus).  The training teacher collator would attach the
            # TRAINING teacher row to colliding monitor row_ids and corrupt
            # teacher_kl / teacher_kl_gap.
            fit_batches = build_probe_batches(
                probe_dataset,
                fit_indices,
                model=trainer.model,
                tokenizer=tokenizer,
                data_collator=probe_collator,
                batch_size=int(config.probe_batch_size),
                on_collator_error=lambda exc: _mark_teacher_fit_unavailable(
                    run_dir, exc
                ),
            )
            score_batches = build_probe_batches(
                probe_dataset,
                score_indices,
                model=trainer.model,
                tokenizer=tokenizer,
                data_collator=probe_collator,
                batch_size=int(config.probe_batch_size),
                on_collator_error=lambda exc: _mark_teacher_fit_unavailable(
                    run_dir, exc
                ),
            )
            batches = [*fit_batches, *score_batches]
            nll_fit_batches = list(range(len(fit_batches)))
            nll_score_batches = list(
                range(len(fit_batches), len(fit_batches) + len(score_batches))
            )
        if not prompts:
            prompts = build_prompt_tensors(
                probe_dataset,
                indices,
                device=device,
                count=int(config.generation_prompts),
            )
        _, weight, softcap = text_model_and_head(trainer.model)
        return ProbeRunner(
            model=trainer.model,
            text_model=trainer.loop_text_model or trainer.model,
            weight=weight,
            softcap=softcap,
            batches=batches,
            config=ProbeConfig(
                anchor_ns=tuple(config.anchor_ns),
                eval_ns=tuple(sorted(set(config.nll_ns) | set(config.anchor_ns) | {1})),
                nll_ns=tuple(config.nll_ns),
                generation_ns=tuple(config.generation_ns),
                generation_prompts=int(config.generation_prompts),
                generation_max_new_tokens=int(config.generation_max_new_tokens),
                generation_batch_size=int(config.generation_batch_size),
                nll_temperatures=tuple(config.nll_temperatures),
                token_chunk=int(config.probe_token_chunk),
                vocab_chunk=int(config.probe_vocab_chunk),
                ngram=int(config.ngram),
                repetition_rate_threshold=float(config.repetition_rate_threshold),
                repetition_max_count=int(config.repetition_max_count),
            ),
            tokenizer=tokenizer,
            prompts=prompts,
            text_dir=text_dir,
            nll_fit_batches=nll_fit_batches,
            nll_score_batches=nll_score_batches,
        )

    memory_cap = _memory_cap_gib(args)
    callback = HealMonitorCallback(
        monitor,
        trainer,
        probe_factory=_probe_factory,
        memory_cap_gib=memory_cap,
    )
    trainer.add_callback(callback)
    _install_capture(trainer, monitor)
    _LOG.info(
        "heal monitor enabled: %d monitor rows, probe every %d tokens, "
        "%d row(s) per probe micro-batch, metrics -> %s",
        len(indices),
        int(config.probe_interval_tokens),
        int(config.probe_batch_size),
        monitor.metrics_path,
    )
    return callback


def _install_capture(trainer: Any, monitor: HealMonitor) -> None:
    """Wrap ``compute_loss`` and the dial loss to capture integrity/result.

    Both wrappers are observation only: the returned loss object is exactly the
    one the original method returned.
    """
    original_compute = trainer.compute_loss

    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        monitor.observe_batch(inputs, model)
        result = original_compute(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if bool(getattr(model, "training", False)):
            monitor.observe_microbatch(trainer)
        return result

    trainer.compute_loss = compute_loss

    # The per-group norms must be captured *before* HF clips.  LoopSFTTrainer
    # already overrides _clip_grad_norm, so wrapping the instance method is the
    # last hook that runs before the clipping arithmetic and before the
    # optimizer step mutates the weights.
    original_clip = getattr(trainer, "_clip_grad_norm", None)
    if callable(original_clip):

        def clip_grad_norm(model, *clip_args, **clip_kwargs):
            monitor.capture_gradients(model)
            return original_clip(model, *clip_args, **clip_kwargs)

        trainer._clip_grad_norm = clip_grad_norm

    original_dial = getattr(trainer, "_dial_compute_loss", None)
    if callable(original_dial):

        def dial_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss, outputs = original_dial(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            if isinstance(outputs, Mapping):
                monitor.observe_hidden_states(outputs)
                monitor.observe_dial_result(outputs.get("dial"))
            return (loss, outputs) if return_outputs else loss

        trainer._dial_compute_loss = dial_compute_loss


def _memory_cap_gib(args: Any) -> float | None:
    """The allocator cap actually used at start: min(fraction*total, free-margin)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        fraction = float(getattr(args, "vram_cap", 0.0) or 0.0)
        if fraction <= 0:
            return None
        free, total = torch.cuda.mem_get_info()
        margin = float(getattr(args, "vram_free_margin_gib", 0.5) or 0.0)
        cap = min(float(fraction) * float(total), float(free) - margin * 2**30)
        return cap / 2**30 if cap > 0 else None
    except Exception:  # pragma: no cover - CUDA query failure
        return None
