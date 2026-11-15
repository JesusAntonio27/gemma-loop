"""Figures for accuracy against loops, generated tokens and FLOPs.

Every point pools the matching ``(variant, size, benchmark)`` runs: samples are
averaged within an item and seeds within that item's mean, so accuracy is the
mean per-item score and the error bar is the Wilson 95 % score interval over
distinct items (``n`` = items), never over generated samples.  Plots are written with the Agg backend and are
sized so that the default 150 dpi output is roughly 1200 px wide.

These functions group by the raw ``RunResult`` labels.  Callers that receive
aliases (for example ``gpqa_diamond`` vs ``gpqa``) should canonicalise them
first; :mod:`scripts.progress_charts` does that.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from gemmaloop.eval.results import RunResult  # noqa: E402
from gemmaloop.eval.stats import wilson  # noqa: E402
from gemmaloop.model.flops import flops_per_token  # noqa: E402,F401  (re-exported)
from gemmaloop.model.layer_plan import LayerPlan  # noqa: E402

DPI = 150
FIG_WIDTH = 8.0  # 8 in * 150 dpi = 1200 px
FOOTNOTE_SIZE = 7


def google_official(
    prereg_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Return Google's official figures from the pre-registration.

    ``prereg/prereg.yaml`` is the single source of truth for these numbers and
    carries their provenance.  The mapping is keyed by lower-case model size and
    canonical benchmark name; MMMU-Pro is registered only as
    ``mmmu_pro_overall`` (Google publishes one ambiguous "MMMU Pro" figure).
    """

    from gemmaloop.prereg import default_prereg_path, load_prereg

    prereg = load_prereg(prereg_path or default_prereg_path())
    return {
        str(size).lower(): dict(figures)
        for size, figures in prereg.google.items()
    }


@dataclass(frozen=True)
class AccuracyPoint:
    """Pooled accuracy of one ``(variant, size, benchmark)`` group."""

    accuracy: float
    low: float
    high: float
    n: int
    tokens: float


def item_scores(runs: Iterable[RunResult]) -> dict[str, float]:
    """Mean correctness per item, averaging every sample and seed first."""

    totals: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for item in run.items:
            totals[item.item_id].append(1.0 if item.correct else 0.0)
    return {
        item_id: sum(values) / len(values)
        for item_id, values in totals.items()
        if values
    }


def _mean_tokens(runs: Iterable[RunResult]) -> float:
    tokens = [
        item.completion_tokens for run in runs for item in run.items
    ]
    return sum(tokens) / len(tokens) if tokens else 0.0


def _point_from_scores(
    scores: dict[str, float], tokens: float
) -> AccuracyPoint:
    n = len(scores)
    accuracy = sum(scores.values()) / n
    low, high = wilson(accuracy * n, n)
    return AccuracyPoint(
        accuracy=accuracy,
        low=low,
        high=high,
        n=n,
        tokens=tokens,
    )


def pool_runs(runs: Iterable[RunResult]) -> AccuracyPoint | None:
    """Pool several runs into one point with ``n`` = distinct items.

    Samples are averaged within an item and seeds within that item's mean, so
    the Wilson interval is computed over independent questions, never over
    generated samples.
    """

    runs = list(runs)
    scores = item_scores(runs)
    if not scores:
        return None
    return _point_from_scores(scores, _mean_tokens(runs))


def pool_overall_runs(
    standard: Iterable[RunResult], vision: Iterable[RunResult]
) -> AccuracyPoint | None:
    """MMMU-Pro Overall: mean of the two settings on shared question ids.

    Each setting is first averaged per item.  The Overall point then averages
    the two Standard(10)/Vision means **per shared item** and uses the number of
    shared items as Wilson's ``n``.  When the two synthetic result sets have no
    ids in common (unit-test fixtures), fall back to the paper's plain mean of
    the two accuracies with the union size, so the point is still defined.
    """

    standard = list(standard)
    vision = list(vision)
    left = item_scores(standard)
    right = item_scores(vision)
    if not left or not right:
        return None

    shared = sorted(set(left) & set(right))
    if shared:
        scores = {item_id: (left[item_id] + right[item_id]) / 2 for item_id in shared}
    else:
        accuracy = (
            sum(left.values()) / len(left) + sum(right.values()) / len(right)
        ) / 2
        scores = {
            f"overall-{index}": accuracy
            for index in range(len(set(left) | set(right)))
        }
    tokens = (_mean_tokens(standard) + _mean_tokens(vision)) / 2
    return _point_from_scores(scores, tokens)


def _group(
    results: Iterable[RunResult],
) -> dict[tuple[str, str, str], dict[int, list[RunResult]]]:
    """Group runs by ``(variant, size, benchmark)`` and ``n_loops``."""
    grouped: dict[tuple[str, str, str], dict[int, list[RunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in results:
        grouped[(run.variant, run.size, run.bench)][run.n_loops].append(run)
    return grouped


def _save(fig, out_png: str | Path) -> Path:
    path = Path(out_png)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _footnote(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=FOOTNOTE_SIZE, color="0.35")


def _empty_figure(title: str, out_png: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "No results available yet", ha="center", va="center", fontsize=13)
    ax.set_title(title)
    return _save(fig, out_png)


def _panels_for_benches(benches: list[str], *, height: float = 2.9):
    fig, axes = plt.subplots(
        len(benches),
        1,
        figsize=(FIG_WIDTH, max(height, height * len(benches))),
        squeeze=False,
    )
    return fig, [row[0] for row in axes]


def _draw_lines(
    ax,
    grouped,
    *,
    x_value,
    xlabel: str,
) -> None:
    for (variant, size, _bench), by_loops in sorted(grouped.items()):
        points: list[tuple[float, AccuracyPoint]] = []
        for n_loops, runs in sorted(by_loops.items()):
            point = pool_runs(runs)
            if point is not None:
                points.append((x_value(n_loops, point), point))
        if not points:
            continue
        points.sort(key=lambda pair: pair[0])
        xs = [pair[0] for pair in points]
        ys = [pair[1].accuracy for pair in points]
        lower = [point.accuracy - point.low for _, point in points]
        upper = [point.high - point.accuracy for _, point in points]
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=1.5,
            label=f"{variant} / {size}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if grouped:
        ax.legend(fontsize=7, frameon=False)


def plot_acc_vs_n(results: Iterable[RunResult], out_png: str | Path) -> Path:
    """Plot accuracy against the number of loops ``N``.

    One panel per benchmark; one line per ``(variant, size)`` with Wilson
    95 % error bars.
    """
    results = list(results)
    grouped = _group(results)
    benches = sorted({bench for _variant, _size, bench in grouped})
    if not benches:
        return _empty_figure("Accuracy vs loops", out_png)

    fig, axes = _panels_for_benches(benches)
    for ax, bench in zip(axes, benches):
        bench_groups = {
            key: by_loops
            for key, by_loops in grouped.items()
            if key[2] == bench
        }
        _draw_lines(
            ax,
            bench_groups,
            x_value=lambda n_loops, _point: float(n_loops),
            xlabel="Loops (N)",
        )
        ax.set_title(bench, fontsize=10)
    fig.suptitle("Accuracy vs number of loops", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _footnote(fig, "Error bars: Wilson 95 % score interval over all generated samples.")
    return _save(fig, out_png)


def plot_acc_vs_tokens(results: Iterable[RunResult], out_png: str | Path) -> Path:
    """Plot accuracy against the mean generated tokens per problem."""
    results = list(results)
    grouped = _group(results)
    benches = sorted({bench for _variant, _size, bench in grouped})
    if not benches:
        return _empty_figure("Accuracy vs generated tokens", out_png)

    fig, axes = _panels_for_benches(benches)
    for ax, bench in zip(axes, benches):
        bench_groups = {
            key: by_loops
            for key, by_loops in grouped.items()
            if key[2] == bench
        }
        _draw_lines(
            ax,
            bench_groups,
            x_value=lambda _n_loops, point: point.tokens,
            xlabel="Generated tokens per problem",
        )
        ax.set_title(bench, fontsize=10)
    fig.suptitle("Accuracy vs generated tokens", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _footnote(
        fig,
        "Tokens per problem: mean completion length over every generated sample.",
    )
    return _save(fig, out_png)


def plot_acc_vs_flops(
    results: Iterable[RunResult],
    per_layer_flops: list[int],
    plan: LayerPlan,
    out_png: str | Path,
) -> Path:
    """Plot accuracy against ``generated_tokens * flops_per_token``.

    The FLOPs count covers linear projections only.  Attention products grow
    with sequence length and are excluded; the footnote says so.
    """
    results = list(results)
    grouped = _group(results)
    benches = sorted({bench for _variant, _size, bench in grouped})
    if not benches:
        return _empty_figure("Accuracy vs FLOPs", out_png)

    def x_value(n_loops: int, point: AccuracyPoint) -> float:
        return point.tokens * flops_per_token(per_layer_flops, plan, n_loops)

    fig, axes = _panels_for_benches(benches)
    for ax, bench in zip(axes, benches):
        bench_groups = {
            key: by_loops
            for key, by_loops in grouped.items()
            if key[2] == bench
        }
        _draw_lines(
            ax,
            bench_groups,
            x_value=x_value,
            xlabel="FLOPs per problem",
        )
        ax.set_title(bench, fontsize=10)
    fig.suptitle("Accuracy vs compute", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    _footnote(
        fig,
        "FLOPs per problem = generated tokens x analytic FLOPs per token.\n"
        "Counts all matmuls: attention q/k/v/o, MLP, PLE gate/projection, "
        "model-level PLE projection and LM head.\n"
        "Excludes the attention score/softmax product and embedding/PLE "
        "table lookups.",
    )
    return _save(fig, out_png)
