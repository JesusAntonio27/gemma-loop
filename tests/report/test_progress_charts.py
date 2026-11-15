"""Tests for the progress chart scanner and generator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import pytest
from matplotlib.container import ErrorbarContainer

from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.model.flops import flops_per_token, model_flops, per_layer_flops
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.report import figures
from scripts import progress_charts

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _run(
    *,
    variant: str,
    size: str = "e2b",
    n_loops: int,
    bench: str = "gpqa",
    correct: list[bool] | None = None,
    tokens: int = 80,
    prereg_tag: str = "prereg-v1",
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
        prereg_tag=prereg_tag,
        items=items,
    )


def _save(root: Path, result: RunResult) -> None:
    name = f"{result.size}_n{result.n_loops}_{result.bench.replace('/', '-')}.json"
    result.save(root / result.variant / name)


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list = []
    real_savefig = matplotlib.figure.Figure.savefig

    def fake_savefig(self, *args, **kwargs):
        captured.append(self)
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)
    return captured


def _errorbars(fig) -> list[tuple[list[float], list[float], list[float], list[float]]]:
    errors = []
    for ax in fig.axes:
        for container in ax.containers:
            if not isinstance(container, ErrorbarContainer):
                continue
            line = container.lines[0]
            segments = [
                segment
                for collection in container.lines[2]
                for segment in collection.get_segments()
            ]
            if line is None:
                xs = [float(segment[0][0]) for segment in segments]
                ys: list[float] = []
            else:
                xs = [float(value) for value in line.get_xdata()]
                ys = [float(value) for value in line.get_ydata()]
            lows = [float(segment[0][1]) for segment in segments]
            highs = [float(segment[1][1]) for segment in segments]
            errors.append((xs, ys, lows, highs))
    return errors


def _png_ok(path: Path) -> None:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 1000


def test_no_results_writes_only_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    out = tmp_path / "out"

    results = progress_charts.load_results(results_root)
    assert results == []
    produced = progress_charts.generate_progress_charts(
        results, out, root=results_root
    )

    assert produced == []
    assert sorted(path.name for path in out.iterdir()) == ["summary.md"]
    text = (out / "summary.md").read_text("utf-8").lower()
    assert "nothing is available yet" in text


def test_one_baseline_omits_missing_variants(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _save(results_root, _run(variant="O", n_loops=0, correct=[True, False] * 10))
    out = tmp_path / "out"

    results = progress_charts.load_results(results_root)
    assert len(results) == 1
    produced = progress_charts.generate_progress_charts(
        results, out, root=results_root
    )

    names = {path.name for path in produced}
    assert names == {"before_after_e2b.png"}
    assert not list(out.glob("acc_vs_*.png"))
    _png_ok(out / "before_after_e2b.png")

    labels = [
        bar.label
        for bar in progress_charts.before_after_series(results, "e2b", "gpqa")
    ]
    assert labels == ["Google (official)", "O"]
    assert "O-H" not in labels
    assert "L" not in labels

    summary = (out / "summary.md").read_text("utf-8")
    assert "no O-H baseline" in summary
    assert "no L at N=2" in summary
    assert "no results for AIME 2026" in summary


def test_full_synthetic_set_produces_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "GEMMALOOP_CONFIG_JSON", str(FIXTURES / "e2b_text_config.json")
    )
    results_root = tmp_path / "results"
    _save(results_root, _run(variant="O", n_loops=0, correct=[True, False] * 5))
    _save(results_root, _run(variant="O-H", n_loops=0, correct=[True, True, False] * 3))
    for variant in ("L", "R", "R-damped"):
        for n_loops in (1, 2, 3, 4):
            successes = min(10, 5 + n_loops)
            correct = [index < successes for index in range(10)]
            _save(
                results_root,
                _run(
                    variant=variant,
                    n_loops=n_loops,
                    correct=correct,
                    tokens=60 + 10 * n_loops,
                ),
            )

    results = progress_charts.load_results(results_root)
    assert len(results) == 14
    out = tmp_path / "out"
    produced = progress_charts.generate_progress_charts(
        results, out, root=results_root
    )

    expected = {
        "before_after_e2b.png",
        "acc_vs_n_e2b_gpqa.png",
        "acc_vs_tokens_e2b_gpqa.png",
        "acc_vs_flops_e2b_gpqa.png",
    }
    assert {path.name for path in produced} == expected
    assert {path.name for path in out.iterdir()} == expected | {"summary.md"}
    for name in expected:
        _png_ok(out / name)

    summary = (out / "summary.md").read_text("utf-8")
    assert "| e2b | gpqa | L | 2 | 10 |" in summary
    assert "no L at N=2" not in summary


def test_bench_aliases_are_normalized(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _save(results_root, _run(variant="O", n_loops=0, bench="gpqa_diamond"))
    results = progress_charts.load_results(results_root)
    bars = progress_charts.before_after_series(results, "e2b", "gpqa")
    assert [bar.label for bar in bars] == ["Google (official)", "O"]


def test_load_results_filters_by_prereg_tag(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _run(variant="O", n_loops=0).save(results_root / "o" / "published.json")
    _run(
        variant="O-H", n_loops=0, bench="gpqa", prereg_tag="smoke"
    ).save(results_root / "oh" / "smoke.json")
    # Give the smoke file its own result so it is not a duplicate.
    _run(variant="O-H", n_loops=0, bench="aime", prereg_tag="smoke").save(
        results_root / "oh" / "smoke2.json"
    )

    published = progress_charts.load_results(results_root)
    assert [run.prereg_tag for run in published] == ["prereg-v1"]

    smoke = progress_charts.load_results(results_root, prereg_tag="smoke")
    assert {run.prereg_tag for run in smoke} == {"smoke"}

    everything = progress_charts.load_results(results_root, prereg_tag=None)
    assert len(everything) == 3


def test_load_results_rejects_duplicate_identity(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _run(variant="L", n_loops=2, correct=[True, False]).save(
        results_root / "l" / "first.json"
    )
    _run(variant="L", n_loops=2, correct=[False, True]).save(
        results_root / "l" / "second.json"
    )

    with pytest.raises(ValueError, match="duplicate evaluation results"):
        progress_charts.load_results(results_root)


def test_invalid_json_is_skipped(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "broken.json").write_text("{not json", "utf-8")
    (results_root / "config.json").write_text('{"temperature": 1.0}', "utf-8")
    _save(results_root, _run(variant="O", n_loops=0))
    assert len(progress_charts.load_results(results_root)) == 1


def test_before_after_error_bars_are_percentage_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        _run(variant="O", n_loops=0, correct=[True, False] * 10, tokens=100)
    ]
    bars = progress_charts.before_after_series(results, "e2b", "gpqa")
    o_bar = next(bar for bar in bars if bar.label == "O")

    captured = _capture_figures(monkeypatch)
    out = tmp_path / "before_after.png"
    progress_charts._plot_before_after("e2b", {"gpqa": bars}, out)
    _png_ok(out)

    errors = _errorbars(captured[-1])
    assert len(errors) == 1  # the Google bar has no interval
    _xs, _ys, lows, highs = errors[0]
    assert o_bar.accuracy == pytest.approx(0.5)
    assert lows == pytest.approx([o_bar.low * 100])
    assert highs == pytest.approx([o_bar.high * 100])
    # A true Wilson interval is many percentage points wide; the old bug drew
    # the same numbers divided by 100.
    assert highs[0] - o_bar.accuracy * 100 > 1.0
    assert o_bar.accuracy * 100 - lows[0] > 1.0


def test_official_mmmu_only_maps_to_derived_overall() -> None:
    standard = _run(
        variant="O",
        n_loops=0,
        bench="mmmupro",
        correct=[True] * 10 + [False] * 10,
    )
    vision = _run(
        variant="O",
        n_loops=0,
        bench="mmmupro_vision",
        correct=[True] * 14 + [False] * 6,
    )
    results = [standard, vision]

    standard_bars = progress_charts.before_after_series(
        results, "e2b", "mmmupro"
    )
    vision_bars = progress_charts.before_after_series(
        results, "e2b", "mmmupro_vision"
    )
    assert [bar.label for bar in standard_bars] == ["O"]
    assert [bar.label for bar in vision_bars] == ["O"]

    overall_bars = progress_charts.before_after_series(
        results, "e2b", "mmmu_pro_overall"
    )
    assert [bar.label for bar in overall_bars] == ["Google (official)", "O"]
    assert overall_bars[0].accuracy == pytest.approx(44.2 / 100)
    assert overall_bars[1].accuracy == pytest.approx((0.5 + 0.7) / 2)


def test_mmmu_overall_interval_uses_shared_item_ids() -> None:
    def overall_run(bench: str, corrects: list[bool]) -> RunResult:
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

    results = [
        overall_run("mmmupro", [True, False]),
        overall_run("mmmu_pro_vision", [False, True]),
    ]
    bars = progress_charts.before_after_series(results, "e2b", "mmmu_pro_overall")
    overall = next(bar for bar in bars if bar.label == "O")
    assert overall.accuracy == pytest.approx(0.5)
    assert overall.n == 2
    low, high = figures.wilson(1.0, 2)
    assert overall.low == pytest.approx(low)
    assert overall.high == pytest.approx(high)


def test_load_text_config_includes_lm_head(tmp_path: Path, monkeypatch) -> None:
    config = json.loads((FIXTURES / "e2b_text_config.json").read_text("utf-8"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": config}), "utf-8")
    monkeypatch.setenv("GEMMALOOP_CONFIG_JSON", str(path))

    loaded = progress_charts.load_text_config("e2b")
    assert loaded["vocab_size"] == 262144
    assert model_flops(loaded) == 2 * 1536 * 35 * 256 + 2 * 1536 * 262144


def test_lcb_google_is_the_only_lcb_google_bar() -> None:
    results = [
        _run(variant="O", n_loops=0, bench="lcb"),
        _run(variant="O", n_loops=0, bench="lcb_google"),
    ]

    lcb_bars = progress_charts.before_after_series(results, "e2b", "lcb")
    assert [bar.label for bar in lcb_bars] == ["O"]

    google_bars = progress_charts.before_after_series(
        results, "e2b", "lcb_google"
    )
    assert [bar.label for bar in google_bars] == ["Google (official)", "O"]
    assert google_bars[0].accuracy == pytest.approx(44.0 / 100)


def test_mathvision_official_value_and_tokens() -> None:
    results = [_run(variant="O", n_loops=0, bench="mathvision")]
    bars = progress_charts.before_after_series(results, "e2b", "mathvision")
    assert bars[0].label == "Google (official)"
    assert bars[0].accuracy == pytest.approx(52.4 / 100)


def test_official_only_benchmark_does_not_produce_before_after(
    tmp_path: Path,
) -> None:
    # L exists only at N=1, so there is no project bar for the before/after
    # chart; the Google number alone must not create a misleading figure.
    results_root = tmp_path / "results"
    _save(results_root, _run(variant="L", n_loops=1))
    results = progress_charts.load_results(results_root)
    out = tmp_path / "out"

    produced = progress_charts.generate_progress_charts(
        results, out, root=results_root
    )
    assert "before_after_e2b.png" not in {path.name for path in produced}
    assert (out / "acc_vs_n_e2b_gpqa.png").exists()


def test_baseline_marker_x_uses_measured_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(
        (FIXTURES / "e2b_text_config.json").read_text("utf-8")
    )
    plan = make_layer_plan(config)
    values = per_layer_flops(config)
    results = [
        _run(variant="O", n_loops=0, tokens=100),
        _run(variant="O-H", n_loops=0, tokens=100),
        _run(variant="O-H-ext", n_loops=0, tokens=100),
        _run(variant="L", n_loops=2, tokens=100),
    ]
    captured = _capture_figures(monkeypatch)
    out = tmp_path / "acc_vs_flops.png"

    progress_charts._plot_loop_curves(
        "e2b",
        "gpqa",
        results,
        out,
        metric="flops",
        flops=(values, plan),
    )
    _png_ok(out)

    ax = captured[-1].axes[0]
    markers = {
        container.get_label(): container.lines[0].get_xdata()
        for container in ax.containers
        if isinstance(container, ErrorbarContainer)
        and container.get_label().endswith("(measured)")
    }
    mean_tokens = 101.5  # completion tokens 100..103
    expected = mean_tokens * flops_per_token(values, plan, 1)
    assert markers["O (measured)"] == pytest.approx([expected])
    assert markers["O-H-ext (measured)"] == pytest.approx([expected])


def _geometry_run(
    *,
    variant: str = "L",
    n_loops: int = 2,
    block: tuple[int, int] | None = (16, 18),
    bridge: str = "gch",
    policy: str = "loop_only",
    targets: str = "loop_block_coda",
    correct: list[bool] | None = None,
    seed: int = 0,
) -> RunResult:
    result = _run(
        variant=variant,
        n_loops=n_loops,
        correct=correct if correct is not None else [True, False, True, True],
    )
    return result.model_copy(
        update={
            "loop_block": block,
            "bridge": bridge,
            "adapter_policy": policy,
            "lora_target_layers": targets,
            "seed": seed,
        }
    )


def test_load_results_does_not_reject_different_blocks_or_pool_them(tmp_path: Path):
    results_root = tmp_path / "results"
    first = _geometry_run(block=(16, 18), correct=[True] * 4)
    second = _geometry_run(block=(25, 29), correct=[False] * 4)
    first.save(results_root / "l" / "first.json")
    second.save(results_root / "l" / "second.json")

    results = progress_charts.load_results(results_root)
    assert len(results) == 2

    bars = progress_charts.before_after_series(results, "e2b", "gpqa")
    l_bars = [bar for bar in bars if bar.label.startswith("L")]
    assert len(l_bars) == 2
    assert {round(bar.accuracy, 4) for bar in l_bars} == {1.0, 0.0}
    assert any("16-18" in bar.label for bar in l_bars)
    assert any("25-29" in bar.label for bar in l_bars)


def test_flops_chart_uses_each_runs_own_block(tmp_path: Path, monkeypatch):
    config = json.loads(
        (FIXTURES / "e2b_text_config.json").read_text("utf-8")
    )
    values = per_layer_flops(config)
    runs = [
        _geometry_run(n_loops=1, block=(16, 18)),
        _geometry_run(n_loops=2, block=(16, 18)),
        _geometry_run(n_loops=1, block=(25, 29)),
        _geometry_run(n_loops=2, block=(25, 29)),
    ]
    captured = _capture_figures(monkeypatch)
    out = tmp_path / "flops.png"
    progress_charts._plot_loop_curves(
        "e2b",
        "gpqa",
        runs,
        out,
        metric="flops",
        flops=(
            values,
            {
                run.geometry_key(): make_layer_plan(
                    config, block=run.loop_block
                )
                for run in runs
            },
        ),
    )
    _png_ok(out)
    xs = {
        round(float(x), 6)
        for container in captured[-1].axes[0].containers
        if isinstance(container, ErrorbarContainer)
        for x in container.lines[0].get_xdata()
    }
    mean_tokens = 81.5
    for run in runs:
        expected = mean_tokens * flops_per_token(
            values, make_layer_plan(config, block=run.loop_block), run.n_loops
        )
        assert round(float(expected), 6) in xs
