"""Report the training-free loop campaign.

Produces, under ``<artifacts>/report/trainfree/``:

* ``trainfree_table.md`` / ``.csv`` — accuracy with a Wilson 95% interval over
  **items**, the paired difference vs O with a paired item bootstrap
  (10,000 resamples, seed 0), mean generated tokens and truncation rate;
* ``accuracy_lcb.png`` and ``accuracy_gpqa.png`` — accuracy vs N per block with
  O as a horizontal reference and Wilson error bars;
* ``calibrated_nll.png`` — temperature-calibrated NLL vs N per block from stage A;
* ``stageA_summary.md``, ``stageA_calibrated_nll.png`` and
  ``stageA_top1.png`` — the P1-lite block/bridge/N table and small multiples,
  also available on their own with ``--stage-a-only``;
* ``trainfree_summary.md`` — a one-page, clearly labelled summary for sharing.

Chart code lives here on purpose: another branch is changing the keys used by
``progress_charts.py`` and this campaign must not depend on them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from gemmaloop.eval.results import RunResult
from gemmaloop.eval.stats import paired_bootstrap, wilson
from gemmaloop.paths import artifacts_dir

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 0
LABEL = (
    "**Training-free, exploratory, measured before preregistration.** "
    "Descriptive campaign evidence only: not an input to block selection, "
    "the prereg pilot rule, or any confirmatory criterion."
)

# Stage A charts (P1-lite sweep).  Fixed design tokens: one panel per block,
# shared y, two series, a neutral reference and no chart junk.
STAGE_A_TITLE = "Untrained loops vs the original Gemma 4 E2B"
STAGE_A_BLOCK_ORDER = ((25, 29), (16, 18), (15, 29))
STAGE_A_SERIES = (
    ("none", "#2a78d6", "R (pure repeat)", "R"),
    ("damped", "#eb6834", "R-damped", "R-damped"),
)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
REFERENCE = "#8a8984"
GRID = "#dedcd6"


# ---------------------------------------------------------------------------
# Loading / summaries
# ---------------------------------------------------------------------------
def load_runs(root: str | Path) -> list[RunResult]:
    """Load every valid ``RunResult`` under ``root`` (newest file wins)."""
    newest: dict[tuple, tuple[float, RunResult]] = {}
    root = Path(root)
    for path in sorted(root.rglob("*.json")):
        try:
            run = RunResult.load(path)
        except Exception:
            continue
        key = run_key(run)
        mtime = path.stat().st_mtime
        if key not in newest or mtime >= newest[key][0]:
            newest[key] = (mtime, run)
    return [item[1] for item in newest.values()]


def run_key(run: RunResult) -> tuple:
    if int(run.n_loops) == 0:
        return ("O", run.bench)
    block = None if run.loop_block is None else tuple(int(v) for v in run.loop_block)
    return (run.bench, block, run.bridge, int(run.n_loops))


def per_item_scores(run: RunResult) -> dict[str, float]:
    return run.per_item_mean()


def _round_k(scores: dict[str, float]) -> float:
    return float(sum(scores.values()))


def summarize_run(run: RunResult) -> dict[str, Any]:
    scores = per_item_scores(run)
    n = len(scores)
    correct = _round_k(scores)
    accuracy = correct / n if n else 0.0
    lo, hi = wilson(correct, n) if n else (0.0, 0.0)
    tokens = [int(item.completion_tokens) for item in run.items]
    lengths = sum(item.finish_reason == "length" for item in run.items)
    return {
        "key": run_key(run),
        "bench": run.bench,
        "variant": run.variant,
        "block": None if run.loop_block is None else tuple(run.loop_block),
        "bridge": run.bridge,
        "n": 1 if int(run.n_loops) == 0 else int(run.n_loops),
        "n_items": n,
        "accuracy": accuracy,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "truncation_rate": lengths / len(run.items) if run.items else 0.0,
    }


def paired_vs_o(run: RunResult, o_run: RunResult) -> dict[str, float] | None:
    a = per_item_scores(run)
    b = per_item_scores(o_run)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diffs = np.array([a[item] - b[item] for item in common], dtype=float)
    lo, hi = paired_bootstrap(
        a, b, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED
    )
    return {
        "mean": float(diffs.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_pairs": len(common),
    }


def build_table(runs: Iterable[RunResult]) -> list[dict[str, Any]]:
    """One row per (bench, block, bridge, N), paired against O."""
    runs = list(runs)
    by_bench_o: dict[str, RunResult] = {}
    for run in runs:
        if int(run.n_loops) == 0:
            by_bench_o[run.bench] = run

    rows = []
    for run in runs:
        summary = summarize_run(run)
        pair = None if int(run.n_loops) == 0 else paired_vs_o(
            run, by_bench_o.get(run.bench)
        ) if run.bench in by_bench_o else None
        summary["diff_vs_o"] = None if pair is None else pair["mean"]
        summary["diff_lo"] = None if pair is None else pair["lo"]
        summary["diff_hi"] = None if pair is None else pair["hi"]
        summary["diff_pairs"] = None if pair is None else pair["n_pairs"]
        rows.append(summary)

    def order(row):
        block = row["block"]
        return (
            row["bench"],
            -1 if block is None else block[0],
            -1 if block is None else block[1],
            "" if row["bridge"] is None else row["bridge"],
            row["n"],
        )

    return sorted(rows, key=order)


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:+.1f}"


def _block_label(row: dict[str, Any]) -> str:
    block = row["block"]
    if block is None:
        return "original"
    return f"{block[0]}-{block[1]}"


def write_markdown_table(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Training-free loop campaign — accuracy",
        "",
        LABEL,
        "",
        "Wilson intervals are over items; differences are paired item "
        "bootstraps vs O (10,000 resamples, seed 0).",
        "",
        "| bench | block | bridge | N | items | accuracy | 95% CI | "
        "diff vs O (pp) | bootstrap 95% CI | mean tokens | truncation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        diff_ci = (
            "—"
            if row["diff_lo"] is None
            else f"[{100 * row['diff_lo']:+.1f}, {100 * row['diff_hi']:+.1f}]"
        )
        lines.append(
            "| {bench} | {block} | {bridge} | {n} | {items} | {acc:.1%} | "
            "[{lo:.1%}, {hi:.1%}] | {diff} | {ci} | {tokens:.0f} | "
            "{trunc:.1%} |".format(
                bench=row["bench"],
                block=_block_label(row),
                bridge=row["bridge"] or "—",
                n=row["n"],
                items=row["n_items"],
                acc=row["accuracy"],
                lo=row["ci_lo"],
                hi=row["ci_hi"],
                diff=_fmt_pct(row["diff_vs_o"]),
                ci=diff_ci,
                tokens=row["mean_tokens"],
                trunc=row["truncation_rate"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    columns = [
        "bench",
        "block",
        "bridge",
        "n",
        "n_items",
        "accuracy",
        "ci_lo",
        "ci_hi",
        "diff_vs_o",
        "diff_lo",
        "diff_hi",
        "diff_pairs",
        "mean_tokens",
        "truncation_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["block"] = _block_label(row)
            writer.writerow(value)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _series(rows: Sequence[dict[str, Any]], bench: str):
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["bench"] != bench or row["block"] is None:
            continue
        grouped[(row["block"], row["bridge"])].append(row)
    return grouped


def plot_accuracy(
    rows: Sequence[dict[str, Any]], bench: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    o_rows = [row for row in rows if row["bench"] == bench and row["block"] is None]
    o_accuracy = o_rows[0]["accuracy"] if o_rows else None
    o_lo = o_rows[0]["ci_lo"] if o_rows else None
    o_hi = o_rows[0]["ci_hi"] if o_rows else None

    for (block, bridge), points in sorted(
        _series(rows, bench).items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        points = sorted(points, key=lambda row: row["n"])
        x = [row["n"] for row in points]
        y = [row["accuracy"] for row in points]
        lower = [row["accuracy"] - row["ci_lo"] for row in points]
        upper = [row["ci_hi"] - row["accuracy"] for row in points]
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            marker="o",
            capsize=3,
            linewidth=1.6,
            label=f"{block[0]}-{block[1]} / {bridge}",
        )

    if o_accuracy is not None:
        ax.axhline(
            o_accuracy,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=f"O ({100 * o_accuracy:.1f}%)",
        )
        if o_lo is not None and o_hi is not None:
            ax.axhspan(o_lo, o_hi, color="black", alpha=0.06)

    ax.set_xlabel("loop passes N (O shown at N=1)")
    ax.set_ylabel("accuracy (items)")
    ax.set_title(
        f"Training-free (exploratory): {bench} accuracy vs N", fontsize=10
    )
    ax.set_xticks(sorted({1, 2, 3, 4}))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_a_series(
    payloads: Sequence[dict[str, Any]]
) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for record in payload.get("variants", []):
            block = tuple(int(value) for value in record["block"])
            grouped[(block, str(record["bridge"]))].append(record)
    for points in grouped.values():
        points.sort(key=lambda record: int(record["n_loops"]))
    return grouped


def plot_calibrated_nll(
    payloads: Sequence[dict[str, Any]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (block, bridge), points in sorted(
        stage_a_series(payloads).items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        ax.plot(
            [int(record["n_loops"]) for record in points],
            [float(record["nll_n_calibrated"]) for record in points],
            marker="o",
            linewidth=1.6,
            label=f"{block[0]}-{block[1]} / {bridge}",
        )
    ax.set_xlabel("loop passes N")
    ax.set_ylabel("temperature-calibrated NLL")
    ax.set_title(
        "Training-free (exploratory): calibrated NLL vs N", fontsize=10
    )
    ax.set_xticks(sorted({1, 2, 3, 4}))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage A only (P1-lite): one table plus small-multiple charts
# ---------------------------------------------------------------------------
def load_stage_a_payloads(stage_a_dir: str | Path) -> list[dict[str, Any]]:
    """Load every ``n*.json`` payload written by the P1-lite probe."""
    root = Path(stage_a_dir)
    payloads = []
    for path in sorted(root.glob("n*.json")):
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _as_block(record: dict[str, Any]) -> tuple[int, int]:
    return tuple(int(value) for value in record["block"])


def _block_rank(block: tuple[int, int]) -> int:
    if block in STAGE_A_BLOCK_ORDER:
        return STAGE_A_BLOCK_ORDER.index(block)
    return len(STAGE_A_BLOCK_ORDER)


def build_stage_a_rows(
    payloads: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flatten stage-A ``variants`` into one row per (block, bridge, N).

    Newer payloads win when the same (block, bridge, N) appears twice.  The
    ``delta`` and reference fields fall back to the N=1 variant when a payload
    omits the pre-computed values, which keeps synthetic fixtures simple.
    """
    records: dict[tuple[tuple[int, int], str, int], dict[str, Any]] = {}
    for payload in payloads:
        for record in payload.get("variants", []):
            key = (
                _as_block(record),
                str(record["bridge"]),
                int(record["n_loops"]),
            )
            records[key] = record

    baseline: dict[tuple[tuple[int, int], str], tuple[float, float]] = {}
    for (block, bridge, n), record in records.items():
        if n == 1:
            baseline[(block, bridge)] = (
                float(record["nll_n_calibrated"]),
                float(record["top1_n"]),
            )

    rows = []
    for (block, bridge, n), record in records.items():
        nll = float(record["nll_n_calibrated"])
        top1 = float(record["top1_n"])
        base_nll, base_top1 = baseline.get((block, bridge), (nll, top1))
        base_nll = float(record.get("nll_n1_calibrated", base_nll))
        base_top1 = float(record.get("top1_n1", base_top1))
        delta = record.get("delta_nll_calibrated")
        rows.append(
            {
                "block": block,
                "bridge": bridge,
                "n": n,
                "nll": nll,
                "delta_nll": (
                    float(delta) if delta is not None else nll - base_nll
                ),
                "top1": top1,
                "kl": float(
                    record.get("kl_n1_n", record.get("kl_n1_n2", 0.0))
                ),
                "flops_share": float(record.get("flops_share_per_pass", 0.0)),
                "original_nll": base_nll,
                "original_top1": base_top1,
            }
        )

    bridge_rank = {bridge: index for index, (bridge, *_rest) in enumerate(
        STAGE_A_SERIES
    )}
    return sorted(
        rows,
        key=lambda row: (
            _block_rank(row["block"]),
            bridge_rank.get(row["bridge"], len(bridge_rank)),
            row["bridge"],
            row["n"],
        ),
    )


def _stage_a_block_label(block: Sequence[int]) -> str:
    return f"{block[0]}-{block[1]}"


def write_stage_a_summary(
    rows: Sequence[dict[str, Any]], path: Path
) -> None:
    lines = [
        "# Stage A — training-free P1-lite sweep",
        "",
        LABEL,
        "",
        "60 calibration + 60 test verified traces. Calibrated NLL is fitted "
        "on the calibration half and scored on the test half; delta is vs the "
        "original model (N=1). FLOPs are the extra share per loop pass.",
        "",
        "| block | bridge | N | calibrated NLL | delta vs original | top-1 | "
        "KL(N=1||N) | extra FLOPs % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {block} | {bridge} | {n} | {nll:.4f} | {delta:+.4f} | "
            "{top1:.4f} | {kl:.4f} | {flops:+.1f}% |".format(
                block=_stage_a_block_label(row["block"]),
                bridge=row["bridge"],
                n=row["n"],
                nll=row["nll"],
                delta=row["delta_nll"],
                top1=row["top1"],
                kl=row["kl"],
                flops=100.0 * row["flops_share"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage_a_groups(
    rows: Sequence[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["block"], {}).setdefault(
            row["bridge"], []
        ).append(row)
    for bridged in grouped.values():
        for points in bridged.values():
            points.sort(key=lambda row: row["n"])
    return grouped


def _label_positions(
    pairs: Sequence[tuple[float, str]], metric: str
) -> list[float]:
    """Return end-label y positions, nudging apart any that would collide."""
    ordered = sorted(pairs, key=lambda item: item[0])
    positions = [item[0] for item in ordered]
    for index in range(1, len(positions)):
        previous = positions[index - 1]
        current = positions[index]
        if metric == "nll":
            if current < previous * 1.08:
                positions[index] = previous * 1.12
        elif current - previous < 0.03:
            positions[index] = previous + 0.035
    return positions


def _style_stage_a_axis(ax: Any, metric: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 5.15)
    ax.set_xlabel("loop passes N", fontsize=7, color=TEXT, labelpad=2)
    if metric == "nll":
        ax.set_yscale("log")
        ax.set_ylim(0.14, 3.4)
        ax.set_yticks([0.2, 0.5, 1.0, 2.0, 3.0])
        ax.set_yticklabels(["0.2", "0.5", "1", "2", "3"])
        ax.tick_params(axis="y", which="minor", left=False, labelleft=False)
    else:
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"])
    ax.grid(
        True,
        which="major",
        color=GRID,
        linewidth=0.6,
        alpha=0.75,
        zorder=0,
    )
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c7c1")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=TEXT, labelsize=7, length=2, width=0.6)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)


def plot_stage_a_chart(
    rows: Sequence[dict[str, Any]], path: Path, *, metric: str
) -> None:
    """Small multiples: one panel per block, two bridges, shared y."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    grouped = _stage_a_groups(rows)
    blocks = [block for block in STAGE_A_BLOCK_ORDER if block in grouped]
    blocks += sorted(block for block in grouped if block not in blocks)
    if not blocks:
        raise ValueError("no stage-A rows to plot")

    fig, axes = plt.subplots(
        1,
        len(blocks),
        figsize=(8, 3),
        dpi=200,
        sharey=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    axes = list(axes[0])
    reference = rows[0]["original_nll" if metric == "nll" else "original_top1"]

    for index, (ax, block) in enumerate(zip(axes, blocks)):
        _style_stage_a_axis(ax, metric)
        panel = grouped[block]
        flops = panel[next(iter(panel))][0]["flops_share"]
        ax.set_title(
            f"Layers {block[0]}-{block[1]} "
            f"(+{100.0 * flops:.0f}% FLOPs per pass)",
            fontsize=8,
            color=TEXT,
            pad=4,
        )

        ax.axhline(
            reference,
            color=REFERENCE,
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )
        if metric == "nll":
            ax.text(
                2.5,
                reference * 0.93,
                "original model (N=1)",
                ha="center",
                va="top",
                fontsize=6,
                color=REFERENCE,
                zorder=2,
            )
        else:
            ax.text(
                2.5,
                reference + 0.012,
                "original model (N=1)",
                ha="center",
                va="bottom",
                fontsize=6,
                color=REFERENCE,
                zorder=2,
            )

        endpoints = []
        for bridge, color, _legend, short_label in STAGE_A_SERIES:
            points = panel.get(bridge)
            if not points:
                continue
            xs = [row["n"] for row in points]
            ys = [row["nll" if metric == "nll" else "top1"] for row in points]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2.0,
                solid_capstyle="round",
                zorder=3,
            )
            endpoints.append((ys[-1], short_label))

        positions = _label_positions(endpoints, metric)
        for (y_value, short_label), y_position in zip(
            sorted(endpoints, key=lambda item: item[0]), positions
        ):
            ax.text(
                4.12,
                y_position,
                short_label,
                ha="left",
                va="bottom" if metric == "nll" else "top",
                fontsize=6.5,
                color=SECONDARY_TEXT,
                zorder=4,
            )

        if index == 0:
            ax.set_ylabel(
                "calibrated NLL (lower is better)"
                if metric == "nll"
                else "top-1 accuracy (higher is better)",
                fontsize=7,
                color=TEXT,
                labelpad=3,
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=6,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                    label=legend,
                )
                for _bridge, color, legend, _short in STAGE_A_SERIES
            ]
            ax.legend(
                handles=handles,
                loc="upper left" if metric == "nll" else "lower left",
                frameon=False,
                fontsize=6.5,
                labelcolor=TEXT,
                handlelength=1.6,
                borderaxespad=0.2,
                handletextpad=0.4,
            )

    subtitle = (
        "Training-free, exploratory, measured before preregistration. "
        "60 held-out verified traces; "
        + (
            "calibrated NLL (lower is better)"
            if metric == "nll"
            else "top-1 next-token accuracy (higher is better)"
        )
    )
    fig.suptitle(
        STAGE_A_TITLE,
        x=0.5,
        y=0.985,
        fontsize=11,
        color=TEXT,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        subtitle,
        ha="center",
        va="top",
        fontsize=6.5,
        color=SECONDARY_TEXT,
    )
    fig.tight_layout(rect=(0.005, 0.01, 0.995, 0.83))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_stage_a_report(
    rows: Sequence[dict[str, Any]], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stage_a_summary(rows, out_dir / "stageA_summary.md")
    plot_stage_a_chart(
        rows, out_dir / "stageA_calibrated_nll.png", metric="nll"
    )
    plot_stage_a_chart(rows, out_dir / "stageA_top1.png", metric="top1")


# ---------------------------------------------------------------------------
# One-page summary
# ---------------------------------------------------------------------------
def write_summary(
    rows: Sequence[dict[str, Any]],
    payloads: Sequence[dict[str, Any]],
    path: Path,
    *,
    b2_block: Sequence[int] | None = None,
    corpus_manifest: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Training-free loop evaluation — exploratory",
        "",
        LABEL,
        "",
    ]
    if corpus_manifest:
        lines += [
            f"Corpus: {corpus_manifest.get('n_verified', '?')} verified "
            f"traces from `{corpus_manifest.get('source', '?')}` "
            f"(revision `{str(corpus_manifest.get('source_revision', '?'))[:12]}`).",
            "",
        ]
    lines += [
        "## Headline",
        "",
        "| bench | config | accuracy (95% CI) | diff vs O |",
        "| --- | --- | ---: | ---: |",
    ]
    for bench in sorted({row["bench"] for row in rows}):
        subset = [row for row in rows if row["bench"] == bench]
        o_row = next((row for row in subset if row["block"] is None), None)
        if o_row is not None:
            lines.append(
                f"| {bench} | O | {100 * o_row['accuracy']:.1f}% "
                f"({100 * o_row['ci_lo']:.1f}–{100 * o_row['ci_hi']:.1f}) | — |"
            )
        loop_rows = [row for row in subset if row["block"] is not None]
        if loop_rows:
            best = max(loop_rows, key=lambda row: row["accuracy"])
            diff = _fmt_pct(best["diff_vs_o"])
            lines.append(
                f"| {bench} | {_block_label(best)} / {best['bridge']} / "
                f"N={best['n']} | {100 * best['accuracy']:.1f}% "
                f"({100 * best['ci_lo']:.1f}–{100 * best['ci_hi']:.1f}) | "
                f"{diff} pp |"
            )
    lines += ["", "## Stage A (P1-lite)", ""]
    if payloads:
        n2 = [
            record
            for payload in payloads
            for record in payload.get("variants", [])
            if int(record["n_loops"]) == 2
        ]
        if n2:
            best = min(n2, key=lambda record: float(record["nll_n_calibrated"]))
            lines.append(
                "Lowest calibrated NLL at N=2: "
                f"block {best['block'][0]}–{best['block'][1]}, "
                f"bridge {best['bridge']}, "
                f"{float(best['nll_n_calibrated']):.4f}."
            )
    if b2_block is not None:
        lines.append(
            f"Stage-B2 block (descriptive rule): {b2_block[0]}–{b2_block[1]}."
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Exploratory and descriptive; no confirmatory claim and no effect on "
        "block selection or the preregistration.",
        "- Accuracy is one sample per item; intervals are Wilson over items.",
        "- Paired differences are item bootstraps vs O (10,000 resamples, seed 0).",
        "- Calibrated NLL fits one temperature per (config, N) on half the "
        "P1-lite traces and scores the other half.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--stage-a-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--corpus-manifest", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument(
        "--stage-a-only",
        action="store_true",
        help=(
            "read only the P1-lite n*.json payloads and write "
            "stageA_summary.md, stageA_calibrated_nll.png and stageA_top1.png"
        ),
    )
    return parser


def _defaults() -> dict[str, Path]:
    root = artifacts_dir()
    return {
        "results_root": root / "results" / "trainfree",
        "stage_a_dir": root / "results" / "trainfree" / "stageA",
        "out_dir": root / "report" / "trainfree",
        "corpus_manifest": root / "data" / "p1lite" / "manifest.json",
        "plan": root / "runs" / "trainfree" / "plan.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = _defaults()
    results_root = args.results_root or defaults["results_root"]
    stage_a_dir = args.stage_a_dir or defaults["stage_a_dir"]
    out_dir = args.out_dir or defaults["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_stage_a_payloads(stage_a_dir)
    runs = load_runs(results_root)

    if args.stage_a_only or (not runs and payloads):
        if not payloads:
            raise SystemExit(f"no stage-A payloads under {stage_a_dir}")
        stage_rows = build_stage_a_rows(payloads)
        if not stage_rows:
            raise SystemExit(f"no stage-A variants in {stage_a_dir}")
        write_stage_a_report(stage_rows, out_dir)
        print(f"wrote stage-A report to {out_dir}")
        return 0

    if not runs:
        raise SystemExit(f"no RunResult files under {results_root}")
    rows = build_table(runs)
    write_markdown_table(rows, out_dir / "trainfree_table.md")
    write_csv(rows, out_dir / "trainfree_table.csv")

    if payloads:
        plot_calibrated_nll(payloads, out_dir / "calibrated_nll.png")
    for bench in sorted({row["bench"] for row in rows}):
        plot_accuracy(rows, bench, out_dir / f"accuracy_{bench}.png")

    b2_block = None
    plan_path = args.plan or defaults["plan"]
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            b2_block = plan.get("b2_block")
        except (OSError, ValueError):
            b2_block = None
    corpus_manifest = None
    manifest_path = args.corpus_manifest or defaults["corpus_manifest"]
    if manifest_path.exists():
        try:
            corpus_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            corpus_manifest = None

    write_summary(
        rows,
        payloads,
        out_dir / "trainfree_summary.md",
        b2_block=b2_block,
        corpus_manifest=corpus_manifest,
    )
    print(f"wrote report to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
