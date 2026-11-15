"""Render heal-monitoring reports from a live or finished run.

Modes
-----
* ``--run <run_dir>`` -- final report (default when a path is given).
* ``--watch <run_dir>`` -- CPU-only, low-priority loop that re-renders every
  five minutes (``--interval``) while the run is live.
* ``--compare <run_dir> ...`` -- overlay the four teacher-step arms (or any
  set of runs) on the same charts.

Outputs under the run directory (or ``--out``): ``heal_report.md`` (plain
language), ``heal_report.json`` and the PNG charts.  The charts use the
project's series palette (``#2a78d6``, ``#eb6834``, then the reference
palette), gray reference lines, direct labels, small multiples by N and never
a dual axis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from gemmaloop.train.checkpoint import atomic_write_json, atomic_write_text  # noqa: E402
from gemmaloop.train.heal_monitor import (  # noqa: E402
    MONITOR_ALERTS_FILE,
    MONITOR_METRICS_FILE,
    MONITOR_PROBES_FILE,
    MonitorConfig,
)

DPI = 150
FIG_WIDTH = 8.0
SERIES_COLORS = ("#2a78d6", "#eb6834", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")
REFERENCE_GRAY = "0.55"
GRID_ALPHA = 0.25
GENERATION_NS = (1, 2, 4)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_run(run_dir: str | Path, *, label: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {
        "name": label or run_dir.name,
        "dir": run_dir,
        "metrics": _read_jsonl(run_dir / MONITOR_METRICS_FILE),
        "probes": _read_jsonl(run_dir / MONITOR_PROBES_FILE),
        "alerts": _read_json(run_dir / MONITOR_ALERTS_FILE).get("alerts", []),
        "status": _read_json(run_dir / "status.json"),
        "run_json": _read_json(run_dir / "run.json"),
    }


# ---------------------------------------------------------------------------
# Traffic lights
# ---------------------------------------------------------------------------


def _alert_rules(run: Mapping[str, Any]) -> dict[str, str]:
    rules: dict[str, str] = {}
    for alert in run.get("alerts") or []:
        if not isinstance(alert, dict):
            continue
        rule = str(alert.get("rule", ""))
        severity = str(alert.get("severity", "warn"))
        if rule and (rule not in rules or severity == "abort"):
            rules[rule] = severity
    return rules


def _check(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def build_traffic_light(run: Mapping[str, Any]) -> list[dict[str, str]]:
    metrics = run.get("metrics") or []
    probes = run.get("probes") or []
    rules = _alert_rules(run)
    checks: list[dict[str, str]] = []

    def alert_status(rule: str) -> str | None:
        severity = rules.get(rule)
        if severity == "abort":
            return "red"
        if severity == "warn":
            return "yellow"
        return None

    def item(name: str, rule: str, healthy: str, bad: str = "") -> None:
        status = alert_status(rule)
        if status is None:
            status = "green"
            detail = healthy
        else:
            detail = bad or f"rule fired ({rule})"
        checks.append({"check": name, **_check(status, detail)})

    nonfinite = [
        record.get("nonfinite", {})
        for record in metrics
        if isinstance(record.get("nonfinite"), Mapping)
    ]
    if any(int(value or 0) > 0 for entry in nonfinite for value in entry.values()):
        checks.append({"check": "Non-finite loss/gradient", **_check("red", "non-finite value recorded")})
    else:
        checks.append({"check": "Non-finite loss/gradient", **_check("green", "none")})

    integrity_bad = any(
        (record.get("teacher_integrity") or {}).get("ok") is False for record in metrics
    )
    checks.append(
        {
            "check": "Teacher data integrity",
            **_check("red" if integrity_bad else "green", "renormalisation failed" if integrity_bad else "top-k + tail sums valid"),
        }
    )

    invariant = [probe.get("invariant") or {} for probe in probes]
    applicable = [entry for entry in invariant if entry.get("applicable")]
    if applicable:
        ok = all(entry.get("ok") is True for entry in applicable)
        checks.append(
            {
                "check": "Exit 1 == O",
                **_check("green" if ok else "red", "bit-exact on every probe" if ok else "probe mismatch"),
            }
        )
    else:
        checks.append({"check": "Exit 1 == O", **_check("gray", "not applicable (unlooped model)")})

    primary_key = str(MonitorConfig().primary_n)
    parent_values = [
        (probe.get("anchor_kl") or {}).get(primary_key) for probe in probes
    ]
    parent_bad = any(
        value is not None and math.isfinite(float(value)) and float(value) > 1.0
        for value in parent_values
    )
    item(
        f"Parent anchor (KL at N={primary_key})",
        "parent_kl_runaway",
        "below 1.0 nat/token",
        f"KL(O || student N={primary_key}) exceeded 1.0 nat/token",
    )
    if parent_bad and checks[-1]["status"] == "green":
        checks[-1] = {
            "check": f"Parent anchor (KL at N={primary_key})",
            "status": "yellow",
            "detail": "KL above 1.0 nat/token on at least one probe (abort needs 3 consecutive)",
        }

    item("Loss spike", "loss_spike", "loss within 3x its rolling median")
    item("Gradient spike", "grad_spike", "gradient norm within 10x its rolling median")
    item("Teacher learning", "teacher_stall", "teacher KL improving over the last 1M tokens")
    item(
        "Monotone hinge",
        "hinge_active",
        "hinge active on <= 50% of rows after 1M tokens",
    )
    teacher_fit = [
        probe.get("teacher_fit_available")
        for probe in probes
        if probe.get("teacher_fit_available") is not None
    ]
    if teacher_fit and not any(teacher_fit):
        checks.append(
            {
                "check": "Teacher fit (KL to teacher)",
                "status": "gray",
                "detail": "unavailable: no --monitor-teacher-data shards for the MONITOR rows",
            }
        )
    elif teacher_fit:
        item("Teacher fit (KL to teacher)", "overfitting_gap", "monitor/train teacher-KL gap stable")
    else:
        checks.append(
            {
                "check": "Teacher fit (KL to teacher)",
                "status": "gray",
                "detail": "no probe yet",
            }
        )
    item("Generation sanity", "generation_sanity", "no repetition loops or missing answers")
    item("Probe execution", "probe_error", "every monitor probe completed")
    item("Bridge g", "bridge_saturated", "bridge channels learning, below saturation")
    item("Bridge g", "bridge_zero", "bridge channels learning")
    item("Memory margin", "memory_margin", "comfortably below the allocator cap")
    item("Throughput", "throughput_spill", "no sustained throughput collapse")

    if "teacher_integrity" in rules and rules["teacher_integrity"] == "abort":
        # Already reported above; keep the explicit red entry.
        pass
    if checks and any(entry["status"] == "red" for entry in checks):
        overall = "red"
    elif any(entry["status"] == "yellow" for entry in checks):
        overall = "yellow"
    else:
        overall = "green"
    checks.insert(0, {"check": "Overall", **_check(overall, _overall_text(overall))})
    return checks


def _overall_text(status: str) -> str:
    if status == "red":
        return "a registered abort condition fired"
    if status == "yellow":
        return "at least one early-warning rule fired"
    return "all registered checks are green"


def _summary(run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics") or []
    probes = run.get("probes") or []
    last = metrics[-1] if metrics else {}
    last_probe = probes[-1] if probes else {}
    return {
        "run": run.get("name"),
        "run_dir": str(run.get("dir")),
        "arm": (run.get("run_json") or {}).get("arm"),
        "steps": len(metrics),
        "probes": len(probes),
        "tokens_seen": last.get("tokens_seen"),
        "last_loss": last.get("loss"),
        "last_primary_kl_teacher": last.get("primary_kl_teacher"),
        "last_primary_kl_parent": last.get("primary_kl_parent"),
        "last_probe_tokens": last_probe.get("tokens_seen"),
        "alerts": len(run.get("alerts") or []),
        "status_phase": (run.get("status") or {}).get("phase"),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _finish_axis(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=7)


def _direct_label(ax, x: float, y: float, text: str, color: str) -> None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=6,
        color=color,
        va="center",
    )


def _values(records: Sequence[Mapping[str, Any]], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for record in records:
        tokens = record.get("tokens_seen")
        value = record.get(key)
        if tokens is None or value is None:
            continue
        try:
            x, y = float(tokens), float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def _mean_exit(record: Mapping[str, Any], key: str) -> float | None:
    exits = record.get("exits") or {}
    values = [
        float(entry[key])
        for entry in exits.values()
        if isinstance(entry, Mapping) and entry.get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def _metric_series(
    records: Sequence[Mapping[str, Any]], getter: Callable[[Mapping[str, Any]], Any]
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for record in records:
        tokens = record.get("tokens_seen")
        if tokens is None:
            continue
        value = getter(record)
        if value is None:
            continue
        try:
            x, y = float(tokens), float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def _plot_xy(ax, points, *, color, label, linestyle="-", marker=None, logy=False):
    if not points:
        return
    xs = [point[0] / 1e6 for point in points]
    ys = [point[1] for point in points]
    ax.plot(xs, ys, color=color, linewidth=1.4, label=label, linestyle=linestyle, marker=marker, markersize=2)
    _direct_label(ax, xs[-1], ys[-1], label, color)
    if logy and any(value > 0 for value in ys):
        ax.set_yscale("log")


def _runs_palette(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(runs))
    ]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def plot_loss_components(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    panels = (
        ("total loss", lambda record: record.get("loss")),
        ("teacher KL", lambda record: _mean_exit(record, "kl_teacher")),
        ("parent KL", lambda record: _mean_exit(record, "kl_parent")),
        ("hinge value", lambda record: _mean_exit(record, "hinge")),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_WIDTH, 1.8 * len(panels)), sharex=True)
    axes = list(axes)
    palette = _runs_palette(runs)
    for ax, (title, getter) in zip(axes, panels):
        for run, color in zip(runs, palette):
            _plot_xy(
                ax,
                _metric_series(run.get("metrics") or [], getter),
                color=color,
                label=run["name"],
            )
        _finish_axis(ax, xlabel="Tokens seen (M)", ylabel=title, title=title)
    fig.suptitle("Stage 1 heal loss components", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "heal_loss_components.png")


def _group_value(record: Mapping[str, Any], group: str, key: str) -> float | None:
    value = (record.get("grad_groups") or {}).get(group)
    return None if not isinstance(value, Mapping) else value.get(key)


def plot_gradient_norms(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    panels = ("total", "lora", "bridge", "loop_ple")
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_WIDTH, 1.8 * len(panels)), sharex=True)
    axes = list(axes)
    palette = _runs_palette(runs)
    for ax, group in zip(axes, panels):
        for run, color in zip(runs, palette):
            if group == "total":
                points = _values(run.get("metrics") or [], "grad_norm_total")
            else:
                points = _metric_series(
                    run.get("metrics") or [],
                    lambda record, g=group: _group_value(record, g, "grad_norm"),
                )
            _plot_xy(ax, points, color=color, label=run["name"], logy=True)
        _finish_axis(ax, xlabel="Tokens seen (M)", ylabel="grad norm", title=f"{group} (pre-clip)")
        if group != "total":
            ax.set_yscale("log")
    fig.suptitle("Gradient norms by parameter group (before clipping)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "heal_grad_norms.png")


def plot_bridge_lora(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    panels = (
        ("bridge mean |g|", lambda record: (record.get("bridge_g") or {}).get("mean_abs")),
        ("bridge saturation fraction", lambda record: (record.get("bridge_g") or {}).get("saturation_fraction")),
        (
            "LoRA B mean norm",
            lambda record: (
                sum((record.get("lora_b_norms") or {}).values())
                / len(record.get("lora_b_norms") or {})
                if record.get("lora_b_norms")
                else None
            ),
        ),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_WIDTH, 1.8 * len(panels)), sharex=True)
    axes = list(axes)
    palette = _runs_palette(runs)
    for ax, (title, getter) in zip(axes, panels):
        for run, color in zip(runs, palette):
            _plot_xy(
                ax,
                _metric_series(run.get("metrics") or [], getter),
                color=color,
                label=run["name"],
            )
        if "saturation" in title:
            ax.axhline(0.1, color=REFERENCE_GRAY, linewidth=0.8, linestyle="--")
            ax.text(0.01, 0.11, "warn > 10%", transform=ax.get_yaxis_transform(), fontsize=6, color=REFERENCE_GRAY)
        _finish_axis(ax, xlabel="Tokens seen (M)", ylabel=title, title=title)
    fig.suptitle("Bridge g and LoRA B health", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "heal_bridge_lora.png")


def plot_horizon_histogram(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    registered = {}
    for run in runs:
        for record in reversed(run.get("metrics") or []):
            registered = record.get("registered_horizon") or {}
            if registered:
                break
        if registered:
            break
    horizons = sorted({int(key) for run in runs for record in (run.get("metrics") or []) for key in (record.get("horizon_fractions") or {})})
    if not horizons:
        horizons = [4, 5, 6, 7, 8]
    fig, axes = plt.subplots(1, max(1, len(runs)), figsize=(FIG_WIDTH, 2.8), squeeze=False)
    palette = _runs_palette(runs)
    for ax, run, color in zip(axes[0], runs, palette):
        fractions = {}
        for record in reversed(run.get("metrics") or []):
            fractions = record.get("horizon_fractions") or {}
            if fractions:
                break
        empirical = [float(fractions.get(str(n), 0.0)) for n in horizons]
        registered_values = [float(registered.get(str(n), 0.0)) for n in horizons]
        positions = list(range(len(horizons)))
        ax.bar([p - 0.2 for p in positions], empirical, width=0.4, color=color, label="empirical")
        ax.bar([p + 0.2 for p in positions], registered_values, width=0.4, color="0.75", label="registered")
        ax.set_xticks(positions, [str(n) for n in horizons])
        _finish_axis(ax, xlabel="Horizon T", ylabel="fraction", title=run["name"])
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Sampled horizon vs registered (T=4 at 0.70, tail 0.30)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out / "heal_horizon_histogram.png")


def _probe_value(probe: Mapping[str, Any], section: str, n: int) -> float | None:
    value = (probe.get(section) or {}).get(str(n))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def plot_probe_kl(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    ns = [2, 3, 4, 6, 8]
    fig, axes = plt.subplots(len(ns), 1, figsize=(FIG_WIDTH, 1.7 * len(ns)), sharex=True)
    axes = list(axes)
    palette = _runs_palette(runs)
    for ax, n in zip(axes, ns):
        for run, color in zip(runs, palette):
            anchor = [
                (float(probe["tokens_seen"]) / 1e6, value)
                for probe in (run.get("probes") or [])
                if (value := _probe_value(probe, "anchor_kl", n)) is not None
            ]
            teacher = [
                (float(probe["tokens_seen"]) / 1e6, value)
                for probe in (run.get("probes") or [])
                if (value := _probe_value(probe, "teacher_kl", n)) is not None
            ]
            _plot_xy(ax, anchor, color=color, label=f"{run['name']} O", marker="o")
            _plot_xy(ax, teacher, color=color, label=f"{run['name']} teacher", linestyle="--", marker="s")
        ax.axhline(1.0, color=REFERENCE_GRAY, linewidth=0.8, linestyle=":")
        ax.text(0.01, 1.02, "parent run-away warn/abort at 1.0", transform=ax.get_yaxis_transform(), fontsize=6, color=REFERENCE_GRAY)
        _finish_axis(ax, xlabel="Probe tokens (M)", ylabel="KL (nat/token)", title=f"N={n}")
    fig.suptitle("Probe KL: distance to O (solid) and to the teacher (dashed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "heal_probe_kl.png")


def plot_probe_nll(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    ns = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))
    palette = _runs_palette(runs)
    markers = ("o", "s", "^", "D")
    single = len(runs) == 1
    for run_index, run in enumerate(runs):
        for n_index, n in enumerate(ns):
            points = [
                (float(probe["tokens_seen"]) / 1e6, value)
                for probe in (run.get("probes") or [])
                if (value := _probe_value(probe, "calibrated_nll", n)) is not None
            ]
            if single:
                color = SERIES_COLORS[n_index % len(SERIES_COLORS)]
                label = f"N={n}"
            else:
                color = palette[run_index]
                label = f"{run['name']} N={n}"
            _plot_xy(ax, points, color=color, label=label, marker=markers[n_index % len(markers)])
    _finish_axis(
        ax,
        xlabel="Probe tokens (M)",
        ylabel="NLL (nat/token)",
        title="Temperature-calibrated NLL by N (fit half, scored on the other)",
    )
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(
        "Calibrated NLL on the MONITOR split (temperature fitted per N)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / "heal_probe_nll.png")


def plot_hidden_change(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    panels = (
        ("relative hidden change", lambda record: _mean_exit(record, "hidden_change")),
        ("period-2 ratio", lambda record: _mean_exit(record, "period2")),
    )
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 3.8), sharex=True)
    palette = _runs_palette(runs)
    for ax, (title, getter) in zip(list(axes), panels):
        for run, color in zip(runs, palette):
            _plot_xy(ax, _metric_series(run.get("metrics") or [], getter), color=color, label=run["name"])
        _finish_axis(ax, xlabel="Tokens seen (M)", ylabel=title, title=title)
    fig.suptitle("Loop dynamics: per-exit hidden change and period-2", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / "heal_hidden_change.png")


def plot_throughput_memory(runs: Sequence[Mapping[str, Any]], out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 3.8), sharex=True)
    palette = _runs_palette(runs)
    for run, color in zip(runs, palette):
        _plot_xy(axes[0], _values(run.get("metrics") or [], "tok_per_s"), color=color, label=run["name"])
        reserved = _values(run.get("metrics") or [], "reserved_gib")
        _plot_xy(axes[1], reserved, color=color, label=run["name"])
    cap = next(
        (
            record.get("memory_cap_gib")
            for run in runs
            for record in (run.get("metrics") or [])
            if record.get("memory_cap_gib") is not None
        ),
        None,
    )
    if cap is not None:
        axes[1].axhline(float(cap), color=REFERENCE_GRAY, linewidth=0.8, linestyle="--")
        axes[1].text(0.01, float(cap), "allocator cap", transform=axes[1].get_yaxis_transform(), fontsize=6, color=REFERENCE_GRAY, va="bottom")
    _finish_axis(axes[0], xlabel="Tokens seen (M)", ylabel="tok/s", title="Throughput (per optimizer step)")
    _finish_axis(axes[1], xlabel="Tokens seen (M)", ylabel="GiB", title="Reserved GPU memory")
    fig.suptitle("Throughput and memory", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / "heal_throughput_memory.png")


# ---------------------------------------------------------------------------
# Markdown / JSON rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}g}"


def _generation_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in run.get("probes") or []:
        for row in probe.get("generation") or []:
            rows.append({"tokens_seen": probe.get("tokens_seen"), **row})
    return rows


def _link(path_value: str | None, run_dir: Path) -> str:
    if not path_value:
        return "—"
    path = Path(path_value)
    try:
        relative = path.resolve().relative_to(run_dir.resolve())
        return f"[text]({relative.as_posix()})"
    except Exception:
        return f"`{path}`"


def render_markdown(run: Mapping[str, Any], *, charts: Mapping[str, Path]) -> str:
    checks = build_traffic_light(run)
    summary = _summary(run)
    run_dir = Path(run.get("dir"))
    lines = [
        f"# Heal monitor — {run.get('name')}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `{run_dir}`.",
        "",
        "## Traffic lights",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚪"}.get(
            check["status"], check["status"]
        )
        lines.append(f"| {check['check']} | {icon} {check['status']} | {check['detail']} |")
    lines.extend(
        [
            "",
            "## In plain language",
            "",
            (
                f"This report covers {summary['steps']} optimizer steps and "
                f"{summary['probes']} inference-only probes. The last recorded "
                f"loss is {_fmt(summary['last_loss'])}, the last teacher KL at "
                f"the primary N is {_fmt(summary['last_primary_kl_teacher'])} nat/token "
                f"and the last parent anchor KL is {_fmt(summary['last_primary_kl_parent'])}. "
                "The monitor never changes training except through the registered "
                "abort rules."
            ),
            "",
            "## Alerts",
            "",
        ]
    )
    alerts = run.get("alerts") or []
    if alerts:
        lines.extend(["| Severity | Rule | Step | Tokens | Message |", "|---|---|---|---|---|"])
        for alert in alerts:
            lines.append(
                f"| {alert.get('severity')} | {alert.get('rule')} | "
                f"{alert.get('step')} | {alert.get('tokens_seen')} | "
                f"{alert.get('message')} |"
            )
    else:
        lines.append("No alert fired.")
    lines.extend(
        [
            "",
            "## Probe trend",
            "",
            "| Probe tokens | Invariant | " + " | ".join(f"KL(O||N={n})" for n in (2, 3, 4, 6, 8)) + " | " + " | ".join(f"KL(teacher||N={n})" for n in (2, 3, 4, 6, 8)) + " |",
            "|---" * (2 + 10) + "|",
        ]
    )
    for probe in run.get("probes") or []:
        invariant = (probe.get("invariant") or {}).get("ok")
        invariant_text = "—" if invariant is None else ("ok" if invariant else "FAIL")
        cells = [str(probe.get("tokens_seen")), invariant_text]
        for section, ns in (("anchor_kl", (2, 3, 4, 6, 8)), ("teacher_kl", (2, 3, 4, 6, 8))):
            for n in ns:
                cells.append(_fmt((probe.get(section) or {}).get(str(n))))
        lines.append("| " + " | ".join(cells) + " |")
    if not run.get("probes"):
        lines.append("| — | — | " + " | ".join("—" for _ in range(10)) + " |")
    if run.get("probes") and not any(
        probe.get("teacher_fit_available") for probe in run.get("probes") or []
    ):
        lines.append("")
        lines.append(
            "**Teacher fit: unavailable** — the MONITOR corpus has no teacher "
            "top-k shards (`--monitor-teacher-data`); the overfitting-gap WARN "
            "cannot fire for this run."
        )

    lines.extend(["", "## Generation sanity", ""])
    generation = _generation_rows(run)
    if generation:
        lines.extend(
            [
                "| Tokens | Prompt | N | Finish | Thinking closed | Answer | Repetition | Text |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in generation:
            lines.append(
                f"| {row.get('tokens_seen')} | {row.get('prompt_index')} | "
                f"{row.get('n_loops')} | {row.get('finish_reason')} | "
                f"{'yes' if row.get('thinking_closed') else 'NO'} | "
                f"{'yes' if row.get('answer_present') else 'NO'} | "
                f"{'loop' if row.get('repetition_loop') else 'ok'} | "
                f"{_link(row.get('text_path'), run_dir)} |"
            )
    else:
        lines.append("No generation probe was recorded.")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            *[f"- [{path.name}]({path.name})" for path in charts.values()],
            "",
            "## Files",
            "",
            f"- `{MONITOR_METRICS_FILE}` — per-step JSONL (source of the trend charts).",
            f"- `{MONITOR_PROBES_FILE}` — probe JSONL.",
            f"- `{MONITOR_ALERTS_FILE}` — machine-readable alerts.",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(
    run: Mapping[str, Any],
    *,
    out_dir: str | Path | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    run_dir = Path(run.get("dir"))
    output = Path(out_dir) if out_dir is not None else run_dir
    output.mkdir(parents=True, exist_ok=True)
    runs = [run]
    charts: dict[str, Path] = {}
    chart_functions = {
        "loss": plot_loss_components,
        "grad": plot_gradient_norms,
        "bridge": plot_bridge_lora,
        "horizon": plot_horizon_histogram,
        "probe_kl": plot_probe_kl,
        "probe_nll": plot_probe_nll,
        "hidden": plot_hidden_change,
        "throughput": plot_throughput_memory,
    }
    for name, function in chart_functions.items():
        path = function(runs, output)
        if prefix:
            prefixed = output / f"{prefix}{path.name}"
            path.replace(prefixed)
            path = prefixed
        charts[name] = path
    markdown = render_markdown(run, charts=charts)
    md_path = output / f"{prefix}heal_report.md"
    atomic_write_text(md_path, markdown)
    payload = {
        "run": run.get("name"),
        "run_dir": str(run_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "traffic_light": build_traffic_light(run),
        "summary": _summary(run),
        "alerts": run.get("alerts") or [],
        "generation": _generation_rows(run),
        "charts": {name: str(path) for name, path in charts.items()},
        "metric_records": len(run.get("metrics") or []),
        "probe_records": len(run.get("probes") or []),
    }
    atomic_write_json(output / f"{prefix}heal_report.json", payload)
    return {"markdown": md_path, "json": output / f"{prefix}heal_report.json", **charts}


def render_comparison(
    runs: Sequence[Mapping[str, Any]],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".heal_compare.tmp"
    if staging.is_dir():
        for item in staging.iterdir():
            item.unlink()
        staging.rmdir()
    elif staging.exists():
        staging.unlink()
    staging.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Path] = {}
    chart_functions = {
        "loss": plot_loss_components,
        "grad": plot_gradient_norms,
        "bridge": plot_bridge_lora,
        "horizon": plot_horizon_histogram,
        "probe_kl": plot_probe_kl,
        "probe_nll": plot_probe_nll,
        "hidden": plot_hidden_change,
        "throughput": plot_throughput_memory,
    }
    source_names = {
        "loss": "heal_loss_components.png",
        "grad": "heal_grad_norms.png",
        "bridge": "heal_bridge_lora.png",
        "horizon": "heal_horizon_histogram.png",
        "probe_kl": "heal_probe_kl.png",
        "probe_nll": "heal_probe_nll.png",
        "hidden": "heal_hidden_change.png",
        "throughput": "heal_throughput_memory.png",
    }
    try:
        for name, function in chart_functions.items():
            function(runs, staging)
            source = staging / source_names[name]
            path = output / f"heal_compare_{name}.png"
            if source.exists():
                source.replace(path)
                charts[name] = path
    finally:
        try:
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
        except OSError:
            pass
    lines = [
        "# Heal monitor comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}.",
        "",
    ]
    for run in runs:
        summary = _summary(run)
        checks = build_traffic_light(run)
        overall = next((check for check in checks if check["check"] == "Overall"), {})
        lines.append(
            f"## {run['name']} — {overall.get('status', 'unknown')}"
        )
        lines.append("")
        lines.append(
            f"{summary['steps']} steps, {summary['probes']} probes, "
            f"{summary['alerts']} alerts, last loss {_fmt(summary['last_loss'])}."
        )
        lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.extend(f"- [{path.name}]({path.name})" for path in charts.values())
    lines.append("")
    atomic_write_text(output / "heal_compare.md", "\n".join(lines))
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"name": run["name"], "summary": _summary(run), "traffic_light": build_traffic_light(run)}
            for run in runs
        ],
        "charts": {name: str(path) for name, path in charts.items()},
    }
    atomic_write_json(output / "heal_compare.json", payload)
    return {"markdown": output / "heal_compare.md", "json": output / "heal_compare.json", **charts}


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def _lower_priority() -> None:
    """Best-effort low-priority process (the report must never slow a heal)."""
    try:
        os.nice(10)
        return
    except (AttributeError, OSError):
        pass
    if os.name == "nt":  # pragma: no cover - Windows-only branch
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            return


def is_live(run_dir: Path) -> bool:
    status = _read_json(run_dir / "status.json")
    if not status:
        # A watch started before the run writes its first status should keep
        # rendering rather than exit immediately.
        return not (run_dir / "DONE.json").is_file()
    phase = str(status.get("phase", ""))
    return phase not in ("complete", "exited", "aborted-heal-monitor")


def watch(run_dir: str | Path, *, interval: float = 300.0, once: bool = False) -> int:
    run_dir = Path(run_dir)
    _lower_priority()
    while True:
        run = load_run(run_dir)
        render_report(run)
        print(f"rendered {run_dir / 'heal_report.md'}")
        if once:
            return 0
        if not is_live(run_dir):
            return 0
        time.sleep(max(1.0, float(interval)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="run directory")
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--watch", type=Path, default=None)
    parser.add_argument("--compare", nargs="+", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--once", action="store_true", help="watch mode: render once")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.compare:
        runs = [
            load_run(path, label=(args.labels[index] if args.labels and index < len(args.labels) else None))
            for index, path in enumerate(args.compare)
        ]
        out = args.out or Path(args.compare[0]) / "heal_compare"
        render_comparison(runs, out_dir=out)
        print(f"rendered {Path(out) / 'heal_compare.md'}")
        return 0
    if args.watch:
        return watch(args.watch, interval=args.interval, once=args.once)
    run_dir = args.run or args.path
    if run_dir is None:
        raise SystemExit("provide a run directory (positional, --run, --watch or --compare)")
    render_report(load_run(run_dir), out_dir=args.out)
    print(f"rendered {run_dir / 'heal_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
