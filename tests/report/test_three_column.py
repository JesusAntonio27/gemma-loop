"""Tests for the three-column baseline table."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.report import three_column as tc

ROOT = Path(__file__).resolve().parents[2]


def _run(
    *,
    variant: str = "O",
    size: str = "e2b",
    n_loops: int = 0,
    bench: str = "lcb",
    correct: list[bool] | None = None,
    samples: int = 1,
    seed: int = 0,
    tokens: int = 80,
    prereg_tag: str = "prereg-v1",
) -> RunResult:
    correct = correct if correct is not None else [True, False, True, True]
    items = [
        ItemResult(
            item_id=f"{bench}-item-{index}",
            sample_idx=sample,
            correct=value,
            completion_tokens=tokens + index,
            finish_reason="stop",
        )
        for index, value in enumerate(correct)
        for sample in range(samples)
    ]
    return RunResult(
        variant=variant,
        size=size,
        n_loops=n_loops,
        bench=bench,
        prereg_tag=prereg_tag,
        seed=seed,
        items=items,
    )


def _rows(table: str) -> list[list[str]]:
    rows = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _data_rows(table: str) -> list[list[str]]:
    return _rows(table)[2:]


def _row(table: str, label: str) -> list[str]:
    for cells in _data_rows(table):
        if cells[0] == label:
            return cells
    raise AssertionError(f"row {label!r} not found in:\n{table}")


def test_three_column_flags_invalid():
    run = _run(bench="lcb", correct=[True] + [False] * 19)
    table = tc.three_column_table({"lcb": 90.0}, {"lcb": run}, None, 3.0)

    cells = _row(table, "LiveCodeBench v6")
    assert cells[1] == "90.0"
    assert cells[2] == "5.0"
    assert "no validado" in cells[4]


def test_low_power_flag_text():
    # 1/2 correct = 50 %, Google 52.0: within 3 pp but a very wide interval.
    run = _run(bench="aime", correct=[True, False])
    table = tc.three_column_table({"aime": 52.0}, {"aime": run}, None, 3.0)

    validity = _row(table, "AIME 2026")[4]
    assert "validado" in validity
    assert "no validado" not in validity
    assert "chequeo de baja potencia" in validity


def test_mmmu_overall_is_mean_of_standard_and_vision():
    standard = _run(bench="mmmupro", correct=[True] * 5 + [False] * 5)
    vision = _run(bench="mmmu_pro_vision", correct=[True] * 7 + [False] * 3)
    table = tc.three_column_table(
        {"mmmu_pro_overall": 52.6},
        {"mmmupro": standard, "mmmu_pro_vision": vision},
        None,
        3.0,
    )

    overall = _row(table, "MMMU-Pro Overall")
    assert overall[1] == "52.6"
    assert overall[2] == "60.0"

    # Google publishes only the overall figure: the components get no bar.
    standard_row = _row(table, "MMMU-Pro Standard(10)")
    vision_row = _row(table, "MMMU-Pro Vision")
    assert standard_row[1] == tc.EM_DASH
    assert vision_row[1] == tc.EM_DASH


def test_mmmu_overall_wilson_n_uses_shared_item_ids():
    standard = RunResult(
        variant="O",
        size="e2b",
        n_loops=0,
        bench="mmmupro",
        prereg_tag="prereg-v1",
        items=[
            ItemResult(
                item_id="q1",
                sample_idx=0,
                correct=True,
                completion_tokens=10,
                finish_reason="stop",
            ),
            ItemResult(
                item_id="q2",
                sample_idx=0,
                correct=False,
                completion_tokens=10,
                finish_reason="stop",
            ),
        ],
    )
    vision = RunResult(
        variant="O",
        size="e2b",
        n_loops=0,
        bench="mmmu_pro_vision",
        prereg_tag="prereg-v1",
        items=[
            ItemResult(
                item_id="q1",
                sample_idx=0,
                correct=False,
                completion_tokens=10,
                finish_reason="stop",
            ),
            ItemResult(
                item_id="q2",
                sample_idx=0,
                correct=True,
                completion_tokens=10,
                finish_reason="stop",
            ),
        ],
    )
    table = tc.three_column_table(
        {"mmmu_pro_overall": 52.6},
        {"mmmupro": standard, "mmmu_pro_vision": vision},
        None,
        3.0,
    )
    overall = _row(table, "MMMU-Pro Overall")
    assert overall[2] == "50.0"
    assert "n=2 items" in overall[4]


def test_cli_uses_prereg_primary_n(tmp_path: Path):
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(
        (ROOT / "prereg" / "prereg.yaml")
        .read_text("utf-8")
        .replace("primary_n: 4", "primary_n: 3"),
        "utf-8",
    )
    results = tmp_path / "results"
    _run(bench="lcb", correct=[True, False, True, True]).save(
        results / "o" / "e2b_lcb.json"
    )
    _run(
        variant="L", n_loops=3, bench="lcb", correct=[True, True, True, True]
    ).save(results / "l" / "e2b_lcb_n3.json")
    out = tmp_path / "table.md"

    assert (
        tc.main(
            [
                "--size",
                "e2b",
                "--results-dir",
                str(results),
                "--prereg",
                str(prereg),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert _row(out.read_text("utf-8"), "LiveCodeBench v6")[3] == "100.0"


def test_lcb_has_no_google_figure_and_lcb_google_does():
    lcb_run = _run(bench="lcb", correct=[True, False, True, True])
    google_run = _run(bench="lcb_google", correct=[True, True, False, False])
    table = tc.three_column_table(
        {"lcb_google": 44.0},
        {"lcb": lcb_run, "lcb_google": google_run},
        None,
        3.0,
    )

    assert _row(table, "LiveCodeBench v6")[1] == tc.EM_DASH
    google_row = _row(table, "LiveCodeBench v6 (full-v6 sample)")
    assert google_row[1] == "44.0"
    assert google_row[2] == "50.0"


def test_cli_rejects_duplicate_selected_results(tmp_path: Path):
    results = tmp_path / "results"
    _run(bench="lcb", correct=[True, False]).save(results / "o" / "first.json")
    _run(bench="lcb", correct=[False, True]).save(results / "o" / "second.json")

    with pytest.raises(ValueError, match="duplicate evaluation results"):
        tc.main(
            [
                "--size",
                "e2b",
                "--results-dir",
                str(results),
                "--out",
                str(tmp_path / "table.md"),
            ]
        )


def test_spatial_rows_are_exploratory_without_google_figure():
    run = _run(bench="spatialeval_vqa", correct=[True, False, True])
    table = tc.three_column_table({}, {"spatialeval_vqa": run}, None, 3.0)

    cells = _row(table, "SpatialEval VQA")
    assert cells[1] == tc.EM_DASH
    assert "exploratorio" in cells[4]
    assert "sin cifra de Google" in cells[4]


def test_missing_modified_column_renders_em_dash():
    run = _run(bench="gpqa", correct=[True, False])
    table = tc.three_column_table({"gpqa": 43.4}, {"gpqa": run}, None, 3.0)

    rows = _data_rows(table)
    assert rows
    for cells in rows:
        assert cells[3] == tc.EM_DASH


def test_modified_column_renders_accuracy():
    original = _run(bench="gpqa", correct=[True, False])
    modified = _run(
        variant="L", n_loops=2, bench="gpqa", correct=[True, True, True, False]
    )
    table = tc.three_column_table(
        {"gpqa": 43.4}, {"gpqa": original}, {"gpqa": modified}, 3.0
    )

    assert _row(table, "GPQA Diamond")[3] == "75.0"


def test_wilson_uses_item_count_not_samples():
    # 2 items x 4 samples = 8 generated samples, but only 2 independent items.
    run = _run(bench="aime", correct=[True, False], samples=4)
    table = tc.three_column_table({"aime": 40.0}, {"aime": run}, None, 3.0)

    validity = _row(table, "AIME 2026")[4]
    assert "n=2 items" in validity
    assert "n=8" not in validity


def test_multiple_seeds_average_per_item():
    first = _run(bench="lcb", correct=[True, False], seed=0)
    second = _run(bench="lcb", correct=[True, True], seed=1)
    original = tc._collect([first, second], "e2b", "O", {0, 1})
    table = tc.three_column_table({"lcb": 60.0}, original, None, 3.0)

    # Item 1: (1 + 1) / 2 = 1.0; item 2: (0 + 1) / 2 = 0.5 -> 75 %.
    cells = _row(table, "LiveCodeBench v6")
    assert cells[2] == "75.0"
    assert "n=2 items" in cells[4]


def test_cli_writes_report_from_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts = tmp_path / "artifacts"
    results = artifacts / "results"
    results.mkdir(parents=True)

    _run(bench="lcb", correct=[True, False, True, True]).save(
        results / "o" / "e2b_lcb.json"
    )
    _run(
        variant="L", n_loops=4, bench="lcb", correct=[True, True, True, True]
    ).save(results / "l" / "e2b_lcb_n4.json")

    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", str(artifacts))
    assert tc.main(["--size", "e2b"]) == 0

    out = artifacts / "report" / "three_column_e2b.md"
    text = out.read_text(encoding="utf-8")
    assert "# Three-column baseline table" in text
    assert "LiveCodeBench v6" in text
    assert "44.0" in text
    row = _row(text, "LiveCodeBench v6")
    assert row[2] == "75.0"
    assert row[3] == "100.0"
    assert not list(out.parent.glob("three_column_e2b.md.tmp.*"))


def test_cli_without_modified_results_renders_em_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts = tmp_path / "artifacts"
    results = artifacts / "results"
    results.mkdir(parents=True)
    _run(bench="gpqa", correct=[True, False]).save(results / "o" / "e2b_gpqa.json")

    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", str(artifacts))
    assert tc.main(["--size", "e2b", "--out", str(tmp_path / "table.md")]) == 0

    text = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert _row(text, "GPQA Diamond")[3] == tc.EM_DASH


def test_mmmu_overall_is_plain_mean_with_unequal_component_counts():
    # 10 Standard items at 50 % + 30 Vision items at 70 % -> plain mean 60.0,
    # not the item-weighted (10*50 + 30*70) / 40 = 65.0.
    standard = _run(bench="mmmupro", correct=[True] * 5 + [False] * 5)
    vision = _run(
        bench="mmmu_pro_vision", correct=[True] * 21 + [False] * 9
    )
    table = tc.three_column_table(
        {"mmmu_pro_overall": 52.6},
        {"mmmupro": standard, "mmmu_pro_vision": vision},
        None,
        3.0,
    )

    overall = _row(table, "MMMU-Pro Overall")
    assert overall[2] == "60.0"
    assert "n=40 items" in overall[4]


def test_mmmu_overall_can_be_derived_from_component_google_keys():
    standard = _run(bench="mmmupro", correct=[True] * 5 + [False] * 5)
    vision = _run(bench="mmmu_pro_vision", correct=[True] * 7 + [False] * 3)
    table = tc.three_column_table(
        {"mmmu_pro_standard": 44.2, "mmmu_pro_vision": 44.2},
        {"mmmupro": standard, "mmmu_pro_vision": vision},
        None,
        3.0,
    )

    assert _row(table, "MMMU-Pro Overall")[1] == "44.2"
    assert _row(table, "MMMU-Pro Standard(10)")[1] == tc.EM_DASH
    assert _row(table, "MMMU-Pro Vision")[1] == tc.EM_DASH


def test_every_registered_benchmark_renders_a_row_without_results():
    run = _run(bench="lcb", correct=[True, False])
    table = tc.three_column_table({"lcb": 44.0}, {"lcb": run}, None, 3.0)

    for label in (
        "AIME 2026",
        "GPQA Diamond",
        "MMLU-Pro",
        "MMMU-Pro Standard(10)",
        "MMMU-Pro Vision",
        "MMMU-Pro Overall",
        "MATH-Vision",
        "SpatialEval VQA",
        "SpatialEval TQA",
        "CV-Bench 2D",
    ):
        cells = _row(table, label)
        assert cells[2] == tc.EM_DASH
        assert "sin datos" in cells[4] or "sin cifra de Google" in cells[4]

    assert "exploratorio" in _row(table, "SpatialEval VQA")[4]
    assert "exploratorio" in _row(table, "SpatialEval TQA")[4]
    assert "exploratorio" in _row(table, "CV-Bench 2D")[4]


def test_google_official_reads_the_prereg_single_source():
    from gemmaloop.prereg import default_prereg_path, load_prereg
    from gemmaloop.report.figures import google_official

    prereg = load_prereg(default_prereg_path())
    official = google_official()

    assert official["e2b"]["lcb_google"] == prereg.google["E2B"]["lcb_google"] == 44.0
    assert official["e4b"]["mmlupro"] == prereg.google["E4B"]["mmlupro"] == 69.4
    assert official["e2b"]["mmmu_pro_overall"] == 44.2
    assert "mmmu_pro_standard" not in official["e2b"]
    assert "mmmu_pro_vision" not in official["e2b"]


OFFICIAL_VALUES = (
    "43.4",
    "60.0",
    "44.0",
    "37.5",
    "69.4",
    "58.6",
    "52.0",
    "42.5",
    "44.2",
    "52.6",
    "52.4",
    "59.5",
)


def test_official_values_only_live_in_prereg_yaml():
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo / "src").rglob("*.py")) + list(
        (repo / "scripts").rglob("*.py")
    )
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for value in OFFICIAL_VALUES:
            if value in text:
                offenders.append(f"{path.relative_to(repo)}: {value}")
    assert not offenders, "hard-coded official Google values: " + ", ".join(
        offenders
    )


def test_cli_filters_size_loop_count_and_prereg_tag(
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()

    _run(bench="lcb", correct=[True, False, True, True]).save(
        results / "o" / "e2b_lcb.json"
    )
    _run(
        variant="L", n_loops=4, bench="lcb", correct=[True, True, True, True]
    ).save(results / "l" / "e2b_lcb_n4.json")
    _run(
        variant="L", n_loops=3, bench="lcb", correct=[False, False, False, False]
    ).save(results / "l" / "e2b_lcb_n3.json")
    _run(size="e4b", bench="lcb", correct=[True, True, True, True]).save(
        results / "o" / "e4b_lcb.json"
    )
    _run(bench="gpqa", correct=[True, False], prereg_tag="smoke").save(
        results / "o" / "e2b_gpqa_smoke.json"
    )

    out = tmp_path / "table.md"
    assert (
        tc.main(
            [
                "--size",
                "e2b",
                "--results-dir",
                str(results),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    text = out.read_text(encoding="utf-8")
    lcb = _row(text, "LiveCodeBench v6")
    assert lcb[2] == "75.0"
    assert lcb[3] == "100.0"
    # The smoke-tagged GPQA run is not part of the published baseline.
    assert _row(text, "GPQA Diamond")[2] == tc.EM_DASH


def test_cli_tolerance_defaults_to_prereg(tmp_path: Path):
    out = tmp_path / "table.md"
    assert (
        tc.main(
            [
                "--size",
                "e2b",
                "--results-dir",
                str(tmp_path / "empty"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "tolerancia 3 pp" in out.read_text(encoding="utf-8")


def test_cli_uses_zero_loop_marker_for_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts = tmp_path / "artifacts"
    results = artifacts / "results"
    results.mkdir(parents=True)
    _run(bench="gpqa", n_loops=0, correct=[True, False]).save(
        results / "o" / "e2b_gpqa_n0.json"
    )
    # A looped-looking original run must not satisfy the unlooped baseline.
    _run(bench="gpqa", n_loops=1, correct=[True, True]).save(
        results / "o" / "e2b_gpqa_n1.json"
    )

    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", str(artifacts))
    assert tc.main(["--size", "e2b", "--out", str(tmp_path / "table.md")]) == 0

    text = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert _row(text, "GPQA Diamond")[2] == "50.0"


def test_collect_refuses_to_pool_two_blocks_and_can_filter_one():
    first = _run(variant="L", n_loops=2, seed=0).model_copy(
        update={"loop_block": (16, 18), "bridge": "gch"}
    )
    second = _run(variant="L", n_loops=2, seed=1).model_copy(
        update={"loop_block": (25, 29), "bridge": "gch"}
    )
    with pytest.raises(ValueError, match="multiple loop geometries"):
        tc._collect([first, second], "e2b", "L", {2})

    selected = tc._collect(
        [first, second],
        "e2b",
        "L",
        {2},
        geometry=((16, 18), "gch", None, None),
    )
    assert selected["lcb"].loop_block == (16, 18)


def test_cli_rejects_mixed_engine_across_arms(tmp_path: Path):
    results = tmp_path / "results"
    _run(bench="lcb", correct=[True, False]).save(results / "O" / "o.json")
    _run(bench="lcb", correct=[False, True]).model_copy(
        update={"metadata": {"engine": "refill"}}
    ).save(results / "L" / "l.json")
    with pytest.raises(ValueError, match="across arms"):
        tc.main(
            [
                "--size",
                "e2b",
                "--results-dir",
                str(results),
                "--out",
                str(tmp_path / "table.md"),
            ]
        )
