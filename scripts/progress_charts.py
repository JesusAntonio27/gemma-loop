"""Progress charts generated from whatever evaluation results exist.

The script scans ``$GEMMALOOP_ARTIFACTS/results`` for :class:`RunResult` JSON
files and writes a small, self-contained report to
``$GEMMALOOP_ARTIFACTS/report/progress``:

* ``before_after_<size>.png`` -- Google's official numbers (hatched) next to
  O, O-H and L at N=2 for each benchmark that has data;
* ``acc_vs_n_<size>_<bench>.png`` -- L, R and R-damped across N=1..4, with O
  and O-H as horizontal references;
* ``acc_vs_tokens_<size>_<bench>.png`` and
  ``acc_vs_flops_<size>_<bench>.png`` -- the same curve against generated
  tokens and analytic FLOPs;
* ``summary.md`` -- what was produced, the number of samples behind each
  point, and what is still missing.

Missing variants are omitted, never drawn as zero.  The script runs with zero,
partial or full results: with no results it writes only ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from gemmaloop.eval.results import RunResult  # noqa: E402
from gemmaloop.model.load import MODEL_IDS  # noqa: E402
from gemmaloop.model.flops import flops_per_token, per_layer_flops  # noqa: E402
from gemmaloop.model.layer_plan import make_layer_plan  # noqa: E402
from gemmaloop.paths import artifacts_dir, results_dir  # noqa: E402
from gemmaloop.prereg import (  # noqa: E402
    default_prereg_path,
    load_prereg,
)
from gemmaloop.report.benches import (  # noqa: E402
    BENCH_LABELS,
    BENCH_ORDER,
    canonical_bench,
    canonical_variant,
)
from gemmaloop.report.figures import (  # noqa: E402
    AccuracyPoint,
    google_official,
    item_scores,
    pool_overall_runs,
    pool_runs,
)

DPI = 150
FIG_WIDTH = 8.0  # 8 in * 150 dpi = 1200 px

VARIANT_ORDER = (
    "Google (official)",
    "O",
    "O-H",
    "O-H-ext",
    "L",
    "R",
    "R-damped",
)
VARIANT_COLORS = {
    "Google (official)": "0.62",
    "O": "tab:blue",
    "O-H": "tab:orange",
    "O-H-ext": "tab:brown",
    "L": "tab:green",
    "R": "tab:purple",
    "R-damped": "tab:red",
}
LOOP_VARIANTS = ("L", "R", "R-damped")
REFERENCE_VARIANTS = ("O", "O-H")
BASELINE_VARIANTS = ("O", "O-H", "O-H-ext")
EXPECTED_SIZES = ("e2b", "e4b")
EXPECTED_LOOPS = (1, 2, 3, 4)


def load_text_config(size: str) -> dict:
    """Read the real ``text_config`` from the checkpoint's ``config.json``.

    The analytic FLOPs model needs fields the old hand-copied tables omitted --
    most importantly ``vocab_size`` for the LM head.  ``GEMMALOOP_CONFIG_DIR``
    (one ``<size>_config.json`` per model) and ``GEMMALOOP_CONFIG_JSON`` are
    test/offline overrides.
    """

    config_dir = os.environ.get("GEMMALOOP_CONFIG_DIR")
    override = os.environ.get("GEMMALOOP_CONFIG_JSON")
    if override:
        path = Path(override)
    elif config_dir:
        path = Path(config_dir) / f"{size}_config.json"
    else:
        repo_id = MODEL_IDS[size]
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    local_files_only=True,
                )
            )
        except Exception:
            path = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw.get("text_config", raw))


@dataclass(frozen=True)
class Bar:
    """One bar of a before/after chart."""

    label: str
    accuracy: float
    low: float
    high: float
    n: int


def load_results(
    root: str | Path, prereg_tag: str | None = "prereg-v1"
) -> list[RunResult]:
    """Load valid ``RunResult`` files, optionally filtered to one prereg tag.

    Invalid or unrelated JSON files are skipped: a half-copied result must not
    stop the progress report.  A duplicate ``(variant, size, n_loops, bench,
    seed)`` under the selected tag is an error: re-runs must not be silently
    pooled into one chart point.
    """

    root = Path(root)
    if not root.exists():
        return []
    results: list[RunResult] = []
    for path in sorted(root.rglob("*.json")):
        try:
            run = RunResult.load(path)
        except Exception:
            continue
        if prereg_tag is not None and run.prereg_tag != prereg_tag:
            continue
        results.append(run)

    counts: dict[tuple, int] = defaultdict(int)
    for run in results:
        counts[run.result_identity(canonical_bench(run.bench))] += 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "duplicate evaluation results for the same full identity "
            f"(variant, size, n_loops, bench, seed, geometry): {duplicates}"
        )
    return results


def _geometry(run: RunResult) -> tuple:
    return run.geometry_key()


def _geometry_label(geometry: tuple) -> str:
    block, bridge, policy, target_layers = geometry
    parts: list[str] = []
    if block is not None:
        parts.append(f"block {block[0]}-{block[1]}")
    if bridge is not None:
        parts.append(str(bridge))
    if policy not in (None, "always"):
        parts.append(str(policy))
    if target_layers not in (None, "language"):
        parts.append(str(target_layers))
    return " / ".join(parts) if parts else "legacy"


def _variant_label(variant: str, geometry: tuple, multiple: bool) -> str:
    return f"{variant} [{_geometry_label(geometry)}]" if multiple else variant


def _safe(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in str(value)
    )
    return cleaned.strip("-") or "run"


def _point(runs: Iterable[RunResult]) -> AccuracyPoint | None:
    return pool_runs(runs)


def _variant_runs(
    runs: Sequence[RunResult],
    variant: str,
    *,
    n_loops: int | None = None,
    geometry: tuple | None = None,
) -> list[RunResult]:
    matching = [run for run in runs if canonical_variant(run.variant) == variant]
    if n_loops is not None:
        matching = [run for run in matching if run.n_loops == n_loops]
    elif variant in BASELINE_VARIANTS:
        # Single-point baselines are unlooped: n_loops=0 is the canonical marker
        # (Task 14 ruling).  Never mix a baseline with a looped run.
        matching = [run for run in matching if run.n_loops == 0]
    if geometry is not None:
        matching = [run for run in matching if _geometry(run) == geometry]
    return matching


def _group_by_geometry(runs: Sequence[RunResult]) -> dict[tuple, list[RunResult]]:
    grouped: dict[tuple, list[RunResult]] = defaultdict(list)
    for run in runs:
        grouped[_geometry(run)].append(run)
    return dict(grouped)


def _variant_bars(
    runs: Sequence[RunResult], variant: str, *, n_loops: int | None = None
) -> list[Bar]:
    grouped = _group_by_geometry(_variant_runs(runs, variant, n_loops=n_loops))
    multiple = len(grouped) > 1
    bars: list[Bar] = []
    for geometry, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        point = _point(group)
        if point is not None:
            bars.append(
                Bar(
                    _variant_label(variant, geometry, multiple),
                    point.accuracy,
                    point.low,
                    point.high,
                    point.n,
                )
            )
    return bars


def _project_bars(runs: Sequence[RunResult]) -> list[Bar]:
    """O, O-H and L@N=2 bars available in ``runs`` (one per geometry)."""
    bars: list[Bar] = []
    for variant in ("O", "O-H"):
        bars.extend(_variant_bars(runs, variant))
    bars.extend(_variant_bars(runs, "L", n_loops=2))
    return bars


def _overall_bars(
    standard_runs: Sequence[RunResult], vision_runs: Sequence[RunResult]
) -> list[Bar]:
    """MMMU-Pro Overall bars from the per-item means on shared question ids."""

    bars: list[Bar] = []
    for label in ("O", "O-H", "L"):
        if label == "L":
            left_all = _variant_runs(standard_runs, label, n_loops=2)
            right_all = _variant_runs(vision_runs, label, n_loops=2)
        else:
            left_all = _variant_runs(standard_runs, label)
            right_all = _variant_runs(vision_runs, label)
        left_by = _group_by_geometry(left_all)
        right_by = _group_by_geometry(right_all)
        shared = sorted(set(left_by) & set(right_by), key=str)
        multiple = len(shared) > 1
        for geometry in shared:
            point = pool_overall_runs(left_by[geometry], right_by[geometry])
            if point is not None:
                bars.append(
                    Bar(
                        _variant_label(label, geometry, multiple),
                        point.accuracy,
                        point.low,
                        point.high,
                        point.n,
                    )
                )
    return bars


def before_after_series(
    results: Sequence[RunResult], size: str, bench: str
) -> list[Bar]:
    """Return the available bars for ``(size, bench)`` in a fixed order.

    Google's official number is shown only where the source names that exact
    benchmark.  MMMU-Pro is special: Google publishes one ambiguous "MMMU Pro"
    number, so it is attached only to ``mmmu_pro_overall``, the mean of our
    Standard(10) and Vision accuracies.  O and O-H are pooled over their seeds;
    L is only used at N=2.  Absent variants are simply not in the list.
    """
    bench = canonical_bench(bench)
    runs = [
        run
        for run in results
        if run.size == size and canonical_bench(run.bench) == bench
    ]

    if bench == "mmmu_pro_overall":
        standard_runs = [
            run
            for run in results
            if run.size == size and canonical_bench(run.bench) == "mmmupro"
        ]
        vision_runs = [
            run
            for run in results
            if run.size == size and canonical_bench(run.bench) == "mmmupro_vision"
        ]
        derived = _overall_bars(standard_runs, vision_runs)
    else:
        derived = _project_bars(runs)

    bars: list[Bar] = []
    official = google_official().get(size, {}).get(bench)
    if official is not None:
        value = official / 100.0
        bars.append(Bar("Google (official)", value, value, value, 0))
    return bars + derived


def _loop_series(
    runs: Sequence[RunResult], variant: str
) -> dict[tuple, list[tuple[int, AccuracyPoint]]]:
    """Points per loop geometry, never pooling different blocks/policies."""
    by_geometry: dict[tuple, dict[int, list[RunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        if canonical_variant(run.variant) != variant or run.n_loops < 1:
            continue
        by_geometry[_geometry(run)][run.n_loops].append(run)
    series: dict[tuple, list[tuple[int, AccuracyPoint]]] = {}
    for geometry, by_loops in by_geometry.items():
        points: list[tuple[int, AccuracyPoint]] = []
        for n_loops, grouped in sorted(by_loops.items()):
            point = _point(grouped)
            if point is not None:
                points.append((n_loops, point))
        if points:
            series[geometry] = points
    return series


def _ordered_benches(runs: Sequence[RunResult]) -> list[str]:
    present = {canonical_bench(run.bench) for run in runs}
    known = [bench for bench in BENCH_ORDER if bench in present]
    return known + sorted(present - set(known))


def _before_after_benches(runs: Sequence[RunResult]) -> list[str]:
    """Benchmarks to show; add MMMU-Pro Overall when both parts exist."""
    benches = _ordered_benches(runs)
    present = {canonical_bench(run.bench) for run in runs}
    if {"mmmupro", "mmmupro_vision"} <= present:
        benches = set(benches) | {"mmmu_pro_overall"}
    order = {bench: index for index, bench in enumerate(BENCH_ORDER)}
    return sorted(benches, key=lambda bench: (order.get(bench, len(order)), bench))


def _plot_before_after(
    size: str, benches: dict[str, list[Bar]], out_png: Path
) -> None:
    present = {
        bar.label for bars in benches.values() for bar in bars
    }
    labels = [label for label in VARIANT_ORDER if label in present]
    labels += sorted(label for label in present if label not in VARIANT_ORDER)
    if not labels:
        return

    def _color(label: str) -> str:
        return VARIANT_COLORS.get(label.split(" [")[0], "0.45")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.6))
    names = list(benches)
    positions = list(range(len(names)))
    width = 0.78 / max(len(labels), 1)

    for index, bench in enumerate(names):
        by_label = {bar.label: bar for bar in benches[bench]}
        for slot, label in enumerate(labels):
            bar = by_label.get(label)
            if bar is None:
                continue
            offset = (slot - (len(labels) - 1) / 2) * width
            # The bar height is in percentage points, so the Wilson interval
            # must be converted to percentage points as well (matplotlib reads
            # yerr in data units, i.e. percent on this axis).
            error = [
                [(bar.accuracy - bar.low) * 100],
                [(bar.high - bar.accuracy) * 100],
            ]
            hatch = "//" if label == "Google (official)" else None
            ax.bar(
                index + offset,
                bar.accuracy * 100,
                width=width * 0.92,
                color=_color(label),
                hatch=hatch,
                edgecolor="white" if hatch else None,
                yerr=error if label != "Google (official)" else None,
                capsize=3,
                label=label,
            )
            ax.text(
                index + offset,
                bar.accuracy * 100 + 1.2,
                f"{bar.accuracy * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [BENCH_LABELS.get(bench, bench) for bench in names],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    has_n2_loop = any(
        bar.label.split(" [")[0] == "L"
        for bars in benches.values()
        for bar in bars
    )
    title = f"{size.upper()} - before/after"
    if has_n2_loop:
        title += " at N=2"
    ax.set_title(
        f"{title} (accuracy, Wilson 95% CI)",
        fontsize=12,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(legend_labels, handles))
        ordered = [label for label in labels if label in by_label]
        ax.legend(
            [by_label[label] for label in ordered],
            ordered,
            fontsize=8,
            frameon=False,
            ncol=min(len(ordered), 4),
        )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(
        0.01,
        0.005,
        "Missing variants are omitted, never drawn as zero. "
        "Error bars: Wilson 95% score interval.",
        fontsize=7,
        color="0.35",
    )
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plan_for_geometry(flops: tuple, geometry: tuple | None):
    layer_flops, plan = flops
    if isinstance(plan, dict):
        if geometry is None or geometry not in plan:
            raise ValueError(f"no FLOPs plan for loop geometry {geometry!r}")
        return layer_flops, plan[geometry]
    return layer_flops, plan


def _metric_value(
    metric: str,
    n_loops: int,
    point: AccuracyPoint,
    flops: tuple[list[int], object] | None,
    geometry: tuple | None = None,
) -> float:
    if metric == "n":
        return float(n_loops)
    if metric == "tokens":
        return point.tokens
    if metric == "flops":
        if flops is None:
            raise ValueError("FLOPs requested without a text configuration")
        layer_flops, plan = _plan_for_geometry(flops, geometry)
        return point.tokens * flops_per_token(layer_flops, plan, n_loops)  # type: ignore[arg-type]
    raise ValueError(f"unknown metric {metric!r}")


def _plot_loop_curves(
    size: str,
    bench: str,
    runs: Sequence[RunResult],
    out_png: Path,
    *,
    metric: str,
    flops: tuple[list[int], object] | None = None,
) -> bool:
    curves = {variant: _loop_series(runs, variant) for variant in LOOP_VARIANTS}
    if not any(curves.values()):
        return False

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.4))
    for variant in LOOP_VARIANTS:
        series = curves[variant]
        multiple = len(series) > 1
        for geometry, points in sorted(series.items(), key=lambda item: str(item[0])):
            if not points:
                continue
            xs = [
                _metric_value(metric, n_loops, point, flops, geometry)
                for n_loops, point in points
            ]
            ys = [point.accuracy * 100 for _n_loops, point in points]
            lower = [(point.accuracy - point.low) * 100 for _n, point in points]
            upper = [(point.high - point.accuracy) * 100 for _n, point in points]
            ax.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                marker="o",
                markersize=5,
                capsize=3,
                linewidth=1.6,
                color=VARIANT_COLORS[variant],
                label=_variant_label(variant, geometry, multiple),
            )

    for variant in BASELINE_VARIANTS:
        by_geometry = _group_by_geometry(_variant_runs(runs, variant))
        multiple = len(by_geometry) > 1
        for geometry, group in sorted(by_geometry.items(), key=lambda item: str(item[0])):
            point = _point(group)
            if point is None:
                continue
            label = _variant_label(variant, geometry, multiple)
            ax.axhline(
                point.accuracy * 100,
                linestyle="--",
                linewidth=1.2,
                color=VARIANT_COLORS[variant],
                label=f"{label} (reference)",
            )
            if metric in ("tokens", "flops"):
                # Baselines are single points too: put them at their own measured
                # token/compute value on the x axis, not only on a flat line.
                x = _metric_value(metric, 1, point, flops, geometry)
                lower = (point.accuracy - point.low) * 100
                upper = (point.high - point.accuracy) * 100
                ax.errorbar(
                    [x],
                    [point.accuracy * 100],
                    yerr=[[lower], [upper]],
                    marker="D",
                    linestyle="none",
                    markersize=7,
                    capsize=3,
                    color=VARIANT_COLORS[variant],
                    label=f"{label} (measured)",
                    zorder=5,
                )

    if metric == "n":
        xlabel = "Loops (N)"
        title_metric = "accuracy vs loops"
        xticks = sorted(
            {
                n_loops
                for series in curves.values()
                for points in series.values()
                for n_loops, _point_value in points
            }
        )
        if xticks:
            ax.set_xticks(xticks)
    elif metric == "tokens":
        xlabel = "Generated tokens per problem"
        title_metric = "accuracy vs generated tokens"
    else:
        xlabel = "FLOPs per problem"
        title_metric = "accuracy vs compute"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{size.upper()} - {BENCH_LABELS.get(bench, bench)} - {title_metric}",
        fontsize=12,
    )
    ax.legend(fontsize=8, frameon=False)
    bottom = 0.11 if metric == "flops" else 0.04
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    note = "Error bars: Wilson 95% score interval."
    if metric == "flops":
        note += (
            "\nFLOPs per problem = generated tokens x analytic FLOPs per token.\n"
            "Counts all matmuls: attention q/k/v/o, MLP, PLE gate/projection, "
            "model-level PLE projection and LM head.\n"
            "Excludes the attention score/softmax product and embedding/PLE "
            "table lookups."
        )
    fig.text(0.01, 0.005, note, fontsize=7, color="0.35")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _summary_rows(
    results: Sequence[RunResult],
) -> list[tuple[str, str, str, int, int, float, str]]:
    grouped: dict[tuple[str, str, str, int, tuple], list[RunResult]] = defaultdict(list)
    for run in results:
        grouped[
            (
                run.size,
                canonical_bench(run.bench),
                canonical_variant(run.variant),
                run.n_loops,
                _geometry(run),
            )
        ].append(run)
    rows: list[tuple[str, str, str, int, int, float, str]] = []
    for (size, bench, variant, n_loops, geometry), runs in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        point = _point(runs)
        if point is None:
            continue
        rows.append(
            (
                size,
                bench,
                variant,
                n_loops,
                point.n,
                point.accuracy,
                _geometry_label(geometry),
            )
        )
    return rows


def _lcb_google_lines(results: Sequence[RunResult]) -> list[str]:
    """Report the full-v6 sample and its post-cut subgroup separately."""

    grouped: dict[tuple[str, str, int], list[RunResult]] = defaultdict(list)
    for run in results:
        if canonical_bench(run.bench) == "lcb_google":
            grouped[(run.size, canonical_variant(run.variant), run.n_loops)].append(run)

    lines: list[str] = []
    for (size, variant, n_loops), runs in sorted(grouped.items()):
        full = item_scores(runs)
        post_items: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for item in run.items:
                if item.meta.get("post_cut") is True:
                    post_items[item.item_id].append(1.0 if item.correct else 0.0)
        post = {
            item_id: sum(values) / len(values)
            for item_id, values in post_items.items()
            if values
        }
        full_accuracy = sum(full.values()) / len(full)
        line = (
            f"- `{size}/{variant}` N={n_loops}: full-v6 sample "
            f"{full_accuracy * 100:.1f}% (n={len(full)} items)"
        )
        if post:
            post_accuracy = sum(post.values()) / len(post)
            line += (
                f"; post-cut subgroup {post_accuracy * 100:.1f}% "
                f"(n={len(post)} items; reported separately, never compared "
                "to the full-v6 Google figure)"
            )
        lines.append(line)
    return lines


def _missing_lines(results: Sequence[RunResult]) -> list[str]:
    lines: list[str] = []
    sizes = sorted({run.size for run in results})
    if not sizes:
        return [
            "No result files found yet; nothing is available for e2b or e4b."
        ]

    for size in EXPECTED_SIZES:
        if size not in sizes:
            lines.append(f"No results at all for `{size}` yet.")

    for size in sizes:
        size_runs = [run for run in results if run.size == size]
        present = {canonical_bench(run.bench) for run in size_runs}
        for bench in BENCH_ORDER:
            if bench == "mmmu_pro_overall":
                continue
            if bench not in present:
                lines.append(
                    f"`{size}`: no results for "
                    f"{BENCH_LABELS.get(bench, bench)}."
                )
        for bench in _ordered_benches(size_runs):
            if bench == "mmmu_pro_overall":
                continue
            bench_runs = [
                run for run in size_runs if canonical_bench(run.bench) == bench
            ]
            for variant in REFERENCE_VARIANTS:
                if _point(_variant_runs(bench_runs, variant)) is None:
                    lines.append(
                        f"`{size}/{bench}`: no {variant} baseline."
                    )
            if _point(_variant_runs(bench_runs, "L", n_loops=2)) is None:
                lines.append(
                    f"`{size}/{bench}`: no L at N=2 (before/after bar omitted)."
                )
            for variant in LOOP_VARIANTS:
                for geometry, series in sorted(
                    _loop_series(bench_runs, variant).items(),
                    key=lambda item: str(item[0]),
                ):
                    present_loops = {n_loops for n_loops, _point_value in series}
                    missing = [
                        n for n in EXPECTED_LOOPS if n not in present_loops
                    ]
                    if not missing:
                        continue
                    suffix = (
                        ""
                        if geometry == (None, None, None, None)
                        else f" [{_geometry_label(geometry)}]"
                    )
                    missing_text = ", ".join(f"N={n}" for n in missing)
                    lines.append(
                        f"`{size}/{bench}`: no {variant}{suffix} at "
                        f"{missing_text}."
                    )
        has_standard = "mmmupro" in present
        has_vision = "mmmupro_vision" in present
        if has_standard != has_vision:
            lines.append(
                f"`{size}`: cannot compute MMMU-Pro Overall; it needs both "
                "Standard(10) and Vision results."
            )
    return lines


def _write_summary(
    path: Path,
    results: Sequence[RunResult],
    produced: Sequence[Path],
    *,
    root: str | Path | None,
) -> None:
    rows = _summary_rows(results)
    lines = ["# Gemma-Loop progress charts", ""]
    if root is not None:
        lines.append(f"Scanned results under `{Path(root)}`.")
    lines.append(f"Loaded {len(results)} result file(s).")
    lines.append("")

    if not results:
        lines.extend(
            [
                "Nothing is available yet: no `RunResult` JSON files were found.",
                "",
            ]
        )
    lines.append("## Charts produced")
    lines.append("")
    if produced:
        for output in produced:
            lines.append(f"- `{output.name}`")
    else:
        lines.append("- None yet.")
    lines.append("")

    lines.append("## Data points")
    lines.append("")
    if rows:
        lines.append("| Size | Benchmark | Variant | N | n | Accuracy | Geometry |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for size, bench, variant, n_loops, n, accuracy, geometry in rows:
            lines.append(
                f"| {size} | {bench} | {variant} | {n_loops} | {n} | "
                f"{accuracy * 100:.1f}% | {geometry} |"
            )
    else:
        lines.append("No data points yet.")
    lines.append("")

    lines.append("## LiveCodeBench full-v6 sample (Google comparison)")
    lines.append("")
    google_lines = _lcb_google_lines(results)
    if google_lines:
        lines.extend(google_lines)
    else:
        lines.append("- No `lcb_google` results yet.")
    lines.append("")

    lines.append("## Missing data")
    lines.append("")
    for missing in _missing_lines(results):
        lines.append(f"- {missing}")
    lines.append("")
    lines.append(
        "Official-number provenance: text benchmarks come from "
        "`global-constraints.md` line 15 (Gemma 4 model card / arXiv "
        "2607.02770 Table 5); vision numbers come from "
        "`docs/superpowers/references/gemma-loop/08-vision-and-spatial.md` "
        "lines 50-53 (model card / arXiv 2607.02770 Table 6) at 1120 vision "
        "tokens. Google publishes only an ambiguous \"MMMU Pro\" number, so it "
        "is compared only with MMMU-Pro Overall = the mean of our Standard(10) "
        "and Vision accuracies; Standard(10) and Vision alone get no official "
        "bar."
    )
    lines.append("")

    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def generate_progress_charts(
    results: Iterable[RunResult],
    out_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Write every chart that the available results allow and the summary.

    Returns the list of written chart paths (excluding ``summary.md``).
    """
    results = list(results)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    sizes = sorted({run.size for run in results})
    for size in sizes:
        size_runs = [run for run in results if run.size == size]
        benches = _ordered_benches(size_runs)

        bench_bars: dict[str, list[Bar]] = {}
        for bench in _before_after_benches(size_runs):
            bars = before_after_series(size_runs, size, bench)
            # Never write a "before/after" chart that only has the Google bar.
            if any(bar.label != "Google (official)" for bar in bars):
                bench_bars[bench] = bars
        if bench_bars:
            path = out_dir / f"before_after_{_safe(size)}.png"
            _plot_before_after(size, bench_bars, path)
            produced.append(path)

        try:
            config = load_text_config(size)
        except Exception:
            config = None

        for bench in benches:
            bench_runs = [
                run for run in size_runs if canonical_bench(run.bench) == bench
            ]
            flops_info = None
            if config is not None:
                # Each geometry uses its own block, so a chart with two blocks
                # computes FLOPs from the block that produced each point.
                layer_flops = per_layer_flops(config)
                plans = {
                    _geometry(run): make_layer_plan(
                        config, block=run.loop_block
                    )
                    for run in bench_runs
                }
                plans.setdefault(
                    (None, None, None, None), make_layer_plan(config)
                )
                flops_info = (layer_flops, plans)
            base = f"{_safe(size)}_{_safe(bench)}"
            charts = (
                ("n", f"acc_vs_n_{base}.png"),
                ("tokens", f"acc_vs_tokens_{base}.png"),
                ("flops", f"acc_vs_flops_{base}.png"),
            )
            for metric, filename in charts:
                if metric == "flops" and flops_info is None:
                    continue
                path = out_dir / filename
                if _plot_loop_curves(
                    size,
                    bench,
                    bench_runs,
                    path,
                    metric=metric,
                    flops=flops_info,
                ):
                    produced.append(path)

    _write_summary(
        out_dir / "summary.md", results, produced, root=root
    )
    return produced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate progress charts from RunResult JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="result root (default: $GEMMALOOP_ARTIFACTS/results)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: $GEMMALOOP_ARTIFACTS/report/progress)",
    )
    parser.add_argument(
        "--prereg-tag",
        default=None,
        help=(
            "only chart runs with this tag (default: the tag in prereg.yaml); "
            "pass an empty string to disable"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.results_dir) if args.results_dir else results_dir()
    out_dir = Path(args.out) if args.out else artifacts_dir() / "report" / "progress"
    tag = args.prereg_tag
    if tag is None:
        tag = load_prereg(default_prereg_path()).tag
    results = load_results(root, prereg_tag=tag or None)
    produced = generate_progress_charts(results, out_dir, root=root)
    print(
        f"progress_charts: {len(results)} result file(s), "
        f"{len(produced)} chart(s) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
