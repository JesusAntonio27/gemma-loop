"""Plain-language pilot report: markdown, JSON and small-multiple charts.

The report is descriptive.  It never computes or changes a selection; the
machine-readable ``selection`` object is copied from
:mod:`gemmaloop.pilot.selection` unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["write_report"]

SERIES_COLOR = "#2a78d6"
REFERENCE_COLOR = "#eb6834"
GRAY = "#8a8a8a"
DPI = 150
FIG_WIDTH = 9.0


def _display_teacher(value: str | None) -> str:
    return {"e4b": "E4B", "12b": "12B QAT q4_0"}.get(str(value), str(value))


def _display_arm(arm: Mapping) -> str:
    block = arm.get("block") or [0, 0]
    kind = "O-H" if str(arm.get("kind")) == "O-H" else "L"
    bridge = "" if kind == "O-H" else f" + {arm.get('bridge', '')}"
    single = " (single-exit)" if arm.get("single_exit") else ""
    return f"{kind}[{block[0]}-{block[1]}]{bridge} + {_display_teacher(arm.get('teacher'))}{single}"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _placeholder(title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{title}: no results yet", ha="center", va="center", color=GRAY)
    return _save(fig, path)


def _style(ax) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_per_n(payload: Mapping, path: Path) -> Path:
    loop_arms = [
        arm for arm in (payload.get("arms") or {}).values() if str(arm.get("kind")) == "L"
    ]
    loop_arms = [arm for arm in loop_arms if arm.get("nll")]
    if not loop_arms:
        return _placeholder("Per-N calibrated NLL", path)
    loop_arms.sort(key=lambda arm: (str(arm.get("teacher")), str(arm.get("arm_id"))))
    ncols = min(3, len(loop_arms))
    nrows = (len(loop_arms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 2.5 * nrows), squeeze=False)
    oh_by_block = {
        (tuple(arm.get("block") or []), str(arm.get("teacher"))): arm
        for arm in (payload.get("arms") or {}).values()
        if str(arm.get("kind")) == "O-H" and arm.get("nll")
    }
    baseline = payload.get("baseline_nll")
    for ax, arm in zip(axes.ravel(), loop_arms):
        _style(ax)
        nll = {int(n): value for n, value in (arm.get("nll") or {}).items()}
        xs = sorted(nll)
        ys = [float(nll[n]) for n in xs]
        ax.plot(xs, ys, marker="o", color=SERIES_COLOR, linewidth=1.8, label="L")
        if xs:
            ax.annotate(
                _display_arm(arm),
                xy=(xs[-1], ys[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=6.5,
                va="center",
                color=SERIES_COLOR,
            )
        oh = oh_by_block.get(
            (tuple(arm.get("block") or []), str(arm.get("teacher")))
        )
        if oh and oh.get("nll"):
            value = float(next(iter(oh["nll"].values())))
            ax.axhline(value, color=REFERENCE_COLOR, linestyle="--", linewidth=1.4, label="O-H (block-matched)")
            ax.annotate("O-H", xy=(xs[-1] if xs else 1, value), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=REFERENCE_COLOR)
        if baseline is not None:
            ax.axhline(float(baseline), color=GRAY, linestyle=":", linewidth=1.0, label="O")
        ax.set_title(_display_arm(arm), fontsize=8)
        ax.set_xlabel("loop passes N", fontsize=7)
        ax.set_ylabel("calibrated NLL", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(loop_arms):]:
        ax.set_axis_off()
    fig.suptitle("Calibrated NLL by loop count (lower is better)", fontsize=10)
    return _save(fig, path)


def _plot_loss(payload: Mapping, path: Path) -> Path:
    arms = [
        arm for arm in (payload.get("arms") or {}).values() if (arm.get("training") or {}).get("loss")
    ]
    if not arms:
        return _placeholder("Training loss", path)
    arms.sort(key=lambda arm: str(arm.get("arm_id")))
    ncols = min(3, len(arms))
    nrows = (len(arms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 2.4 * nrows), squeeze=False)
    for ax, arm in zip(axes.ravel(), arms):
        _style(ax)
        history = arm["training"]["loss"]
        xs = [entry.get("step") for entry in history]
        ys = [entry.get("loss") for entry in history]
        ax.plot(xs, ys, color=SERIES_COLOR, linewidth=1.6)
        if xs and ys:
            ax.annotate(_display_arm(arm), xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, va="center", color=SERIES_COLOR)
        ax.set_title(_display_arm(arm), fontsize=8)
        ax.set_xlabel("optimizer step", fontsize=7)
        ax.set_ylabel("training loss", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(arms):]:
        ax.set_axis_off()
    fig.suptitle("Training loss", fontsize=10)
    return _save(fig, path)


def _plot_teacher(payload: Mapping, path: Path) -> Path:
    teachers = payload.get("teachers") or {}
    qualification = payload.get("qualification") or {}
    if not teachers and not qualification:
        return _placeholder("Teacher diagnostics", path)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0), squeeze=False)
    ax = axes[0][0]
    _style(ax)
    keys = [key for key in ("e4b", "12b") if key in teachers]
    if keys:
        xs = range(len(keys))
        entropy = [float((teachers[key] or {}).get("entropy", 0.0) or 0.0) for key in keys]
        top1 = [float((teachers[key] or {}).get("top1_confidence", 0.0) or 0.0) for key in keys]
        ax.bar([x - 0.18 for x in xs], entropy, width=0.34, color=SERIES_COLOR, label="entropy")
        ax.bar([x + 0.18 for x in xs], top1, width=0.34, color=REFERENCE_COLOR, label="top-1 confidence")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([_display_teacher(key) for key in keys], fontsize=7)
        for x, value in zip(xs, entropy):
            ax.annotate(_fmt(value, 2), xy=(x - 0.18, value), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
        for x, value in zip(xs, top1):
            ax.annotate(_fmt(value, 2), xy=(x + 0.18, value), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
        ax.legend(fontsize=6)
    ax.set_title("Teacher sharpness on heal rows", fontsize=8)
    ax.tick_params(labelsize=6)

    ax = axes[0][1]
    _style(ax)
    arms = [
        arm for arm in (payload.get("arms") or {}).values() if arm.get("teacher_kl")
    ]

    def _kl_n(arm):
        return 1 if str(arm.get("kind")) == "O-H" else 2

    arms = [
        arm
        for arm in arms
        if isinstance((arm.get("teacher_kl") or {}).get("initial"), Mapping)
        and str(_kl_n(arm)) in (arm.get("teacher_kl") or {}).get("initial", {})
    ]
    if arms:
        labels = [_display_arm(arm) for arm in arms]

        def _kl_at(payload_value, n):
            initial = (payload_value or {}).get("initial")
            if isinstance(initial, Mapping):
                key = str(n)
                return float(initial.get(key, 0.0) or 0.0)
            return float(initial or 0.0)

        def _kl_final(payload_value, n):
            final = (payload_value or {}).get("final")
            if isinstance(final, Mapping):
                key = str(n)
                return float(final.get(key, 0.0) or 0.0)
            return float(final or 0.0)

        initial = [_kl_at(arm.get("teacher_kl"), _kl_n(arm)) for arm in arms]
        final = [_kl_final(arm.get("teacher_kl"), _kl_n(arm)) for arm in arms]
        xs = range(len(arms))
        ax.bar([x - 0.18 for x in xs], initial, width=0.34, color=SERIES_COLOR, label="initial KL")
        ax.bar([x + 0.18 for x in xs], final, width=0.34, color=REFERENCE_COLOR, label="final KL")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=5, rotation=20, ha="right")
        ax.legend(fontsize=6)
    ax.set_title("KL(teacher || student) at N=2", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.suptitle("Teacher qualification and distillation gap", fontsize=10)
    return _save(fig, path)


def _plot_gains(payload: Mapping, path: Path) -> Path:
    arms = [
        arm
        for arm in (payload.get("arms") or {}).values()
        if str(arm.get("kind")) == "L" and (arm.get("spans") or arm.get("positions"))
    ]
    if not arms:
        return _placeholder("Gain breakdown", path)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0), squeeze=False)
    for ax, field, title in (
        (axes[0][0], "spans", "Gain by span (nats)"),
        (axes[0][1], "positions", "Gain by position bucket (nats)"),
    ):
        _style(ax)
        labels = []
        values = []
        for arm in arms:
            for key, value in (arm.get(field) or {}).items():
                labels.append(f"{_display_arm(arm)} / {key}")
                values.append(float(value))
        colors = [SERIES_COLOR if index % 2 == 0 else REFERENCE_COLOR for index in range(len(values))]
        ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=5)
        ax.axvline(0.0, color=GRAY, linewidth=1.0)
        for index, value in enumerate(values):
            ax.annotate(_fmt(value, 3), xy=(value, index), xytext=(3, 0), textcoords="offset points", fontsize=5, va="center")
        ax.set_title(title, fontsize=8)
    fig.suptitle("Where the calibrated-NLL gain comes from", fontsize=10)
    return _save(fig, path)


def _recipe_paragraph(payload: Mapping) -> list[str]:
    teacher_step = payload.get("teacher_step") or {}
    selection = payload.get("selection") or {}
    qualification = payload.get("qualification") or {}
    teachers = payload.get("teachers") or {}
    chosen_teacher = (selection or {}).get("chosen_teacher") or teacher_step.get(
        "chosen"
    )
    chosen_arm = selection.get("selected")
    lines: list[str] = []
    if not chosen_teacher:
        lines.append(
            "No teacher has been selected yet. The teacher step and rule B are still pending."
        )
        return lines

    if teacher_step.get("single_teacher"):
        excluded = ", ".join(teacher_step.get("excluded_teachers") or []) or "the other teacher"
        lines.append(
            "The teacher comparison is **not estimable**: the registration "
            f"excludes {excluded} (no qualified resident runtime), so there is "
            "a single qualified teacher. The recipe teacher is **E4B**."
        )
    else:
        lines.append(
            f"The registered rule chose **{_display_teacher(chosen_teacher)}** as the teacher."
        )
        gains = teacher_step.get("gains") or {}
        if gains:
            e4b = (gains.get("e4b") or {}).get("mean")
            twelve = (gains.get("12b") or {}).get("mean")
            if chosen_teacher == "12b":
                lines.append(
                    f"Its mean calibrated-NLL gain over O was {_fmt(twelve)} nats "
                    f"versus E4B's {_fmt(e4b)}: the larger teacher won on the measured "
                    "gain, not merely on capacity."
                )
            else:
                lines.append(
                    f"Its mean calibrated-NLL gain over O was {_fmt(e4b)} nats "
                    f"versus 12B's {_fmt(twelve)}: the capacity gap did not make the "
                    "larger teacher the better teacher."
                )
    entropy = {key: (value or {}).get("entropy") for key, value in teachers.items()}
    if any(value is not None for value in entropy.values()):
        lines.append(
            "Sharpness on the heal rows: "
            + ", ".join(
                f"{_display_teacher(key)} entropy {_fmt(value, 2)}"
                for key, value in entropy.items()
                if value is not None
            )
            + "."
        )
    if qualification:
        lines.append(
            "Qualification error (teacher vs its bf16 reference, when present): "
            + ", ".join(
                f"{_display_teacher(key)} mean KL {_fmt(value.get('kl_mean'), 4)}"
                for key, value in qualification.items()
                if isinstance(value, Mapping) and value.get("kl_mean") is not None
            )
            + "."
        )

    if selection.get("stop"):
        lines.append(
            "No loop arm cleared the registered eligibility bar, so rule B **STOPs**: "
            "the honest result is the mechanism null, not a loop recipe."
        )
        return lines
    if not chosen_arm:
        lines.append("Rule B has not selected a loop arm yet.")
        return lines

    arm = (payload.get("arms") or {}).get(chosen_arm) or {}
    lines.append(
        f"Rule B then chose **{_display_arm(arm) if arm else chosen_arm}** for the loop recipe"
        + (" after the monotonicity filter." if selection.get("filter_applied") else " without the monotonicity filter.")
    )
    if arm.get("delta_bar_arm") is not None:
        lines.append(
            f"Its mean-over-N ranking gain is {_fmt(arm.get('delta_bar_arm'))} nats, with "
            f"delta_loop {_fmt(arm.get('delta_loop'))} and delta_arm {_fmt(arm.get('delta_arm'))}."
        )
    spans = arm.get("spans") or {}
    if spans:
        lines.append(
            "The gain comes mostly from "
            + " and ".join(f"{key} tokens ({_fmt(value)} nats)" for key, value in spans.items())
            + "."
        )
    positions = arm.get("positions") or {}
    if positions:
        lines.append(
            "By position it is "
            + ", ".join(f"{key}: {_fmt(value)}" for key, value in positions.items())
            + "."
        )
    return lines


def _markdown(payload: Mapping) -> str:
    pilot = payload.get("pilot") or {}
    lines = [
        "# Pilot: what happened and why",
        "",
        "This report is **descriptive only**. It is never a selection input; the",
        "machine-readable `selection.json` is produced by the registered rule alone.",
        "",
        f"- Heal budget: {pilot.get('heal_tokens', '—')} tokens per arm",
        f"- Registered local cost: {pilot.get('cost_gpu_hours', '—')} GPU-hours",
        "- Held-out traces only; no benchmark accuracy is computed on a pilot checkpoint.",
        "",
    ]

    lines += ["## Teacher step", ""]
    teacher_step = payload.get("teacher_step") or {}
    if teacher_step.get("single_teacher"):
        excluded = ", ".join(teacher_step.get("excluded_teachers") or []) or "the other teacher"
        lines += [
            f"The teacher comparison is **not estimable**: the registration "
            f"excludes {excluded} (no qualified resident runtime), so there is a "
            "single qualified teacher.",
            "",
            "Recipe teacher: **E4B**.",
            "",
        ]
    elif teacher_step:
        rows = []
        for teacher, gain in (teacher_step.get("gains") or {}).items():
            rows.append(
                [
                    _display_teacher(teacher),
                    _fmt(gain.get("l_n4")),
                    _fmt(gain.get("oh_n1")),
                    _fmt(gain.get("mean")),
                ]
            )
        lines += _markdown_table(["Teacher", "L N=4 gain", "O-H N=1 gain", "mean gain"], rows)
        lines += [
            "",
            f"Chosen: **{_display_teacher(teacher_step.get('chosen'))}**"
            + (" (E4B tie-break)." if teacher_step.get("tie_break") else "."),
            "",
        ]
    else:
        lines += ["Pending: the qualified teacher-step arms have no results yet.", ""]

    qualification = payload.get("qualification") or {}
    lines += ["## Teacher qualification (from the qualification JSON when present)", ""]
    if qualification:
        rows = []
        for teacher, values in qualification.items():
            rows.append(
                [
                    _display_teacher(teacher),
                    _fmt(values.get("kl_mean"), 4),
                    _fmt(values.get("top1"), 3),
                    _fmt(values.get("tail_error_mean"), 4),
                    _fmt(values.get("memory_headroom_gib"), 2),
                ]
            )
        lines += _markdown_table(
            ["Teacher", "mean KL", "top-1 agreement", "tail error", "headroom GiB"], rows
        )
    else:
        lines.append("No qualification JSON was available; this section is pending.")
    lines.append("")

    teacher_stats = payload.get("teachers") or {}
    lines += ["## Teacher sharpness on the heal rows", ""]
    if teacher_stats:
        rows = [
            [_display_teacher(key), _fmt(value.get("entropy"), 2), _fmt(value.get("top1_confidence"), 3)]
            for key, value in teacher_stats.items()
        ]
        lines += _markdown_table(
            ["Teacher", "entropy (top-k+tail approx.)", "top-1 confidence (token-weighted)"],
            rows,
        )
    else:
        lines.append("Pending: no teacher-distribution diagnostics yet.")
    lines.append("")

    lines += ["## Selection (rule B)", ""]
    selection = payload.get("selection") or {}
    if selection:
        lines.append(f"- STOP: **{bool(selection.get('stop'))}**")
        lines.append(f"- Selected arm: **{selection.get('selected') or '—'}**")
        lines.append(f"- Eligible set: {', '.join(selection.get('eligible') or []) or '—'}")
        lines.append(f"- Monotonicity filter applied: {bool(selection.get('filter_applied'))}")
        lines.append(f"- Passing arms: {', '.join(selection.get('monotonic_pass') or []) or '—'}")
    else:
        lines.append("Pending: rule B has not run yet.")
    lines.append("")

    lines += ["## Per-N calibrated NLL and the gain breakdown", ""]
    lines.append("![per-N curves](why_per_n.png)")
    lines.append("")
    lines.append("![gain breakdown](why_gains.png)")
    lines.append("")

    arms = payload.get("arms") or {}
    lines += ["## KL(teacher || student) and the fraction of the gap closed", ""]
    rows = []
    for arm in arms.values():
        kl = arm.get("teacher_kl") or {}
        if not kl:
            continue
        initial = kl.get("initial")
        final = kl.get("final")
        gaps = kl.get("gap_closed")
        if isinstance(initial, Mapping) and isinstance(final, Mapping):
            for n in kl.get("n_values") or sorted(initial, key=int):
                key = str(n)
                rows.append(
                    [
                        _display_arm(arm),
                        key,
                        _fmt(initial.get(key)),
                        _fmt(final.get(key)),
                        _fmt((gaps or {}).get(key) if isinstance(gaps, Mapping) else None, 2),
                    ]
                )
        else:
            rows.append(
                [
                    _display_arm(arm),
                    "2",
                    _fmt(initial),
                    _fmt(final),
                    _fmt(gaps, 2),
                ]
            )
    if rows:
        lines += _markdown_table(
            ["Arm", "N", "initial KL", "final KL", "fraction closed"], rows
        )
        lines.append("")
        lines.append(
            "N=1 is structurally the original model for L arms; its gap is shown "
            "for completeness and is expected to be zero."
        )
        cost = next(
            (
                (arm.get("teacher_kl") or {}).get("cost_seconds")
                for arm in arms.values()
                if isinstance((arm.get("teacher_kl") or {}).get("cost_seconds"), (int, float))
            ),
            None,
        )
        if cost is not None:
            lines.append("")
            lines.append(
                "Teacher KL is a descriptive diagnostic on a fixed deterministic "
                f"64-row heal subsample; measured cost on this arm: {_fmt(cost, 1)} s."
            )
    else:
        lines.append("Pending: no teacher-KL measurements yet.")
    lines.append("")

    lines += ["## Training cost", ""]
    rows = []
    for arm in arms.values():
        training = arm.get("training") or {}
        if not training:
            continue
        rows.append(
            [
                _display_arm(arm),
                _fmt(training.get("tokens"), 0),
                _fmt(training.get("wall_seconds"), 0),
                _fmt(training.get("tokens_per_second"), 0),
                _fmt(arm.get("flops_ratio"), 2),
            ]
        )
    if rows:
        lines += _markdown_table(
            ["Arm", "tokens", "wall seconds", "tokens/s", "FLOPs ratio at N=4"],
            rows,
        )
        lines.append("")
        lines.append("![training loss](why_loss.png)")
    else:
        lines.append("Pending: no finished training runs yet.")
    lines.append("")

    lines += ["## Recipe so far", ""]
    lines += [paragraph for paragraph in _recipe_paragraph(payload)]
    lines.append("")
    lines.append("![teacher diagnostics](why_teacher.png)")
    lines.append("")
    return "\n".join(lines)


def write_report(payload: Mapping, out_dir: str | Path) -> dict[str, str]:
    """Write ``why.json``, ``why.md`` and the chart set; return their paths."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "why.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paths = {
        "json": str(json_path),
        "markdown": str(directory / "why.md"),
        "per_n": str(directory / "why_per_n.png"),
        "loss": str(directory / "why_loss.png"),
        "teacher": str(directory / "why_teacher.png"),
        "gains": str(directory / "why_gains.png"),
    }
    _plot_per_n(payload, Path(paths["per_n"]))
    _plot_loss(payload, Path(paths["loss"]))
    _plot_teacher(payload, Path(paths["teacher"]))
    _plot_gains(payload, Path(paths["gains"]))
    Path(paths["markdown"]).write_text(_markdown(payload), encoding="utf-8")
    return paths
