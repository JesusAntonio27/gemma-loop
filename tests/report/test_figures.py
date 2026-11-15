"""Tests for FLOPs accounting and the accuracy figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import pytest
from matplotlib.container import ErrorbarContainer

from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.model.flops import (
    flops_per_token,
    model_flops,
    per_layer_flops,
)
from gemmaloop.model.layer_plan import make_layer_plan, schedule
from gemmaloop.report import figures

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _text_config(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}_text_config.json").read_text("utf-8"))


def _run(
    *,
    variant: str = "L",
    size: str = "e2b",
    n_loops: int = 2,
    bench: str = "gpqa_diamond",
    correct: list[bool] | None = None,
    tokens: int = 100,
) -> RunResult:
    correct = correct if correct is not None else [True, False, True, True]
    items = [
        ItemResult(
            item_id=f"item-{index}",
            sample_idx=0,
            correct=value,
            completion_tokens=tokens + index,
            finish_reason="stop",
        )
        for index, value in enumerate(correct)
    ]
    return RunResult(
        variant=variant,
        size=size,
        n_loops=n_loops,
        bench=bench,
        prereg_tag="prereg-v1",
        items=items,
    )


def _png_ok(path: Path) -> None:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 1000


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list = []
    real_savefig = matplotlib.figure.Figure.savefig

    def fake_savefig(self, *args, **kwargs):
        captured.append(self)
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)
    return captured


def _errorbars(fig) -> list[tuple[list[float], list[float], list[float], list[float]]]:
    """Return ``(x, y, low, high)`` for every error-bar container."""
    errors = []
    for ax in fig.axes:
        for container in ax.containers:
            if not isinstance(container, ErrorbarContainer):
                continue
            line = container.lines[0]
            xs = [float(value) for value in line.get_xdata()]
            ys = [float(value) for value in line.get_ydata()]
            segments = [
                segment
                for collection in container.lines[2]
                for segment in collection.get_segments()
            ]
            lows = [float(segment[0][1]) for segment in segments]
            highs = [float(segment[1][1]) for segment in segments]
            errors.append((xs, ys, lows, highs))
    return errors


def test_flops_per_token_counts_scheduled_layers() -> None:
    config = _text_config("e2b")
    plan = make_layer_plan(config)
    assert flops_per_token([1] * 35, plan, 2) == 50
    assert flops_per_token([1] * 35, plan, 1) == 35
    assert flops_per_token([1] * 35, plan, 4) == 80


def test_model_flops_counts_ple_projection_and_lm_head() -> None:
    e2b = _text_config("e2b")
    e4b = _text_config("e4b")
    # PLE model projection + LM head.
    assert model_flops(e2b) == 2 * 1536 * 35 * 256 + 2 * 1536 * 262144
    assert model_flops(e4b) == 2 * 2560 * 42 * 256 + 2 * 2560 * 262144
    assert model_flops(e2b) == 832_831_488
    assert model_flops(e4b) == 1_397_227_520


def test_per_layer_flops_e2b_exact_values() -> None:
    config = _text_config("e2b")
    values = per_layer_flops(config)
    assert len(values) == 35

    # Producer sliding, producer full, consumer sliding (double-wide MLP) and
    # consumer full.  Consumers have no k_proj/v_proj but a 2x wide MLP.
    assert values[0] == 72_351_744
    assert values[4] == 86_507_520
    assert values[15] == 127_401_984
    assert values[34] == 139_984_896


def test_per_layer_flops_e4b_exact_values() -> None:
    config = _text_config("e4b")
    values = per_layer_flops(config)
    assert len(values) == 42

    assert values[0] == 186_122_240
    assert values[5] == 212_336_640
    assert values[24] == 180_879_360
    assert values[41] == 201_850_880


def test_flops_per_token_matches_schedule_sum_plus_model_constant() -> None:
    for name in ("e2b", "e4b"):
        config = _text_config(name)
        plan = make_layer_plan(config)
        values = per_layer_flops(config)
        constant = model_flops(config)
        for n_loops in (1, 2, 3, 4):
            expected = constant + sum(
                values[layer] for layer in schedule(plan, n_loops)
            )
            assert flops_per_token(values, plan, n_loops) == expected
        assert flops_per_token(values, plan, 1) == constant + sum(values)


def test_flops_per_token_treats_zero_and_none_as_one_pass() -> None:
    config = _text_config("e2b")
    plan = make_layer_plan(config)
    values = per_layer_flops(config)
    one_pass = flops_per_token(values, plan, 1)
    assert flops_per_token(values, plan, 0) == one_pass
    assert flops_per_token(values, plan, None) == one_pass


def test_flops_per_token_rejects_short_layer_list() -> None:
    plan = make_layer_plan(_text_config("e2b"))
    with pytest.raises(ValueError, match="entries"):
        flops_per_token([1] * 10, plan, 1)


def _synthetic_results() -> list[RunResult]:
    results: list[RunResult] = []
    for variant, base in (("O", 0.30), ("O-H", 0.45), ("L", 0.55), ("R", 0.35)):
        for n_loops in (1, 2, 3):
            successes = round((base + 0.02 * n_loops) * 20)
            correct = [index < successes for index in range(20)]
            results.append(
                _run(
                    variant=variant,
                    n_loops=n_loops,
                    correct=correct,
                    tokens=50 + 10 * n_loops,
                )
            )
    return results


def test_pool_runs_uses_item_count_not_samples_or_seeds() -> None:
    def run(seed: int, corrects: list[bool]) -> RunResult:
        items = [
            ItemResult(
                item_id=item_id,
                sample_idx=sample_idx,
                correct=correct,
                completion_tokens=10,
                finish_reason="stop",
            )
            for item_id, correct in zip(("q1", "q2"), corrects)
            for sample_idx in range(3)
        ]
        return RunResult(
            variant="O",
            size="e2b",
            n_loops=0,
            bench="gpqa",
            prereg_tag="prereg-v1",
            seed=seed,
            items=items,
        )

    point = figures.pool_runs([run(0, [True, False]), run(1, [True, False])])
    assert point is not None
    assert point.n == 2
    assert point.accuracy == pytest.approx(0.5)
    low, high = figures.wilson(1.0, 2)
    assert point.low == pytest.approx(low)
    assert point.high == pytest.approx(high)


def test_pool_overall_runs_pairs_shared_item_ids() -> None:
    def run(bench: str, corrects: list[bool]) -> RunResult:
        return RunResult(
            variant="O",
            size="e2b",
            n_loops=0,
            bench=bench,
            prereg_tag="prereg-v1",
            items=[
                ItemResult(
                    item_id=item_id,
                    sample_idx=0,
                    correct=correct,
                    completion_tokens=10,
                    finish_reason="stop",
                )
                for item_id, correct in zip(("q1", "q2"), corrects)
            ],
        )

    standard = run("mmmupro", [True, False])
    vision = run("mmmupro_vision", [False, True])
    point = figures.pool_overall_runs([standard], [vision])
    assert point is not None
    assert point.n == 2
    assert point.accuracy == pytest.approx(0.5)
    low, high = figures.wilson(1.0, 2)
    assert point.low == pytest.approx(low)
    assert point.high == pytest.approx(high)


def test_plot_acc_vs_n_x_values_and_error_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_figures(monkeypatch)
    correct = [index < 15 for index in range(20)]
    out = tmp_path / "acc_vs_n.png"

    figures.plot_acc_vs_n(
        [_run(variant="L", n_loops=1, correct=correct, tokens=100)], out
    )
    _png_ok(out)

    errors = _errorbars(captured[-1])
    assert len(errors) == 1
    xs, ys, lows, highs = errors[0]
    assert xs == [1.0]
    assert ys == pytest.approx([0.75])
    low, high = figures.wilson(15, 20)
    assert lows == pytest.approx([low])
    assert highs == pytest.approx([high])


def test_plot_acc_vs_tokens_x_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_figures(monkeypatch)
    out = tmp_path / "acc_vs_tokens.png"
    figures.plot_acc_vs_tokens([_run(variant="L", n_loops=2, tokens=100)], out)
    _png_ok(out)

    xs, _ys, _lows, _highs = _errorbars(captured[-1])[0]
    # The default fixture has 4 items with completion tokens 100..103.
    assert xs == pytest.approx([101.5])


def test_plot_acc_vs_flops_x_values_include_unlooped_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _text_config("e2b")
    plan = make_layer_plan(config)
    values = per_layer_flops(config)
    captured = _capture_figures(monkeypatch)
    out = tmp_path / "acc_vs_flops.png"

    results = [
        _run(variant="O", n_loops=0, tokens=100),
        _run(variant="L", n_loops=2, tokens=100),
    ]
    figures.plot_acc_vs_flops(results, values, plan, out)
    _png_ok(out)

    errors = _errorbars(captured[-1])
    assert len(errors) == 2
    x_values = sorted(xs[0] for xs, _y, _lo, _hi in errors)
    expected = sorted(
        [
            101.5 * flops_per_token(values, plan, 1),  # n_loops=0 -> N=1
            101.5 * flops_per_token(values, plan, 2),
        ]
    )
    assert x_values == pytest.approx(expected)

    # The plotted half-widths are the Wilson bounds of 3/4 correct samples.
    low, high = figures.wilson(3, 4)
    for _xs, ys, lows, highs in errors:
        assert ys == pytest.approx([0.75])
        assert lows == pytest.approx([low])
        assert highs == pytest.approx([high])


def test_plots_without_results_still_write_png(tmp_path: Path) -> None:
    for name, call in (
        ("n", lambda out: figures.plot_acc_vs_n([], out)),
        ("tokens", lambda out: figures.plot_acc_vs_tokens([], out)),
    ):
        out = tmp_path / f"empty_{name}.png"
        call(out)
        _png_ok(out)
