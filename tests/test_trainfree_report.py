"""Report maths and chart generation on synthetic campaign results."""

from __future__ import annotations

import json

import pytest

import scripts.trainfree_report as report
from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.eval.stats import wilson


def _run(
    *,
    variant: str,
    bench: str,
    n_loops: int,
    correct: int,
    block=None,
    bridge=None,
    truncations: int = 0,
    tokens: int = 100,
) -> RunResult:
    items = []
    for index in range(10):
        items.append(
            ItemResult(
                item_id=f"q{index}",
                sample_idx=0,
                correct=index < correct,
                completion_tokens=tokens,
                finish_reason="length" if index < truncations else "stop",
            )
        )
    return RunResult(
        variant=variant,
        size="e2b",
        n_loops=n_loops,
        bench=bench,
        prereg_tag="trainfree-exploratory",
        loop_block=block,
        bridge=bridge,
        items=items,
    )


def test_table_accuracy_wilson_paired_diff_and_tokens():
    o = _run(variant="O", bench="gpqa", n_loops=0, correct=5, truncations=1)
    loop = _run(
        variant="R_b25-29_n2",
        bench="gpqa",
        n_loops=2,
        correct=7,
        block=(25, 29),
        bridge="none",
        truncations=2,
        tokens=200,
    )
    rows = report.build_table([o, loop])
    loop_row = next(row for row in rows if row["block"] is not None)
    o_row = next(row for row in rows if row["block"] is None)

    assert o_row["accuracy"] == pytest.approx(0.5)
    assert loop_row["n_items"] == 10
    assert loop_row["accuracy"] == pytest.approx(0.7)
    assert loop_row["ci_lo"] == pytest.approx(wilson(7, 10)[0])
    assert loop_row["ci_hi"] == pytest.approx(wilson(7, 10)[1])
    assert loop_row["diff_vs_o"] == pytest.approx(0.2)
    assert loop_row["diff_pairs"] == 10
    assert loop_row["mean_tokens"] == pytest.approx(200)
    assert loop_row["truncation_rate"] == pytest.approx(0.2)
    assert o_row["diff_vs_o"] is None


def test_charts_and_summary_are_written(tmp_path):
    o = _run(variant="O", bench="lcb", n_loops=0, correct=4)
    loop2 = _run(
        variant="R_b25-29_n2",
        bench="lcb",
        n_loops=2,
        correct=6,
        block=(25, 29),
        bridge="none",
    )
    loop3 = _run(
        variant="R_b25-29_n3",
        bench="lcb",
        n_loops=3,
        correct=5,
        block=(25, 29),
        bridge="none",
    )
    rows = report.build_table([o, loop2, loop3])
    payloads = [
        {
            "variants": [
                {
                    "block": [25, 29],
                    "bridge": "none",
                    "n_loops": n,
                    "nll_n_calibrated": 2.0 - 0.1 * n,
                }
                for n in (1, 2, 3, 4)
            ]
        }
    ]

    accuracy_png = tmp_path / "accuracy_lcb.png"
    nll_png = tmp_path / "calibrated_nll.png"
    report.plot_accuracy(rows, "lcb", accuracy_png)
    report.plot_calibrated_nll(payloads, nll_png)
    assert accuracy_png.stat().st_size > 0
    assert nll_png.stat().st_size > 0

    summary = tmp_path / "summary.md"
    report.write_summary(
        rows, payloads, summary, b2_block=(25, 29), corpus_manifest={"n_verified": 120}
    )
    text = summary.read_text(encoding="utf-8")
    assert "Training-free, exploratory, measured before preregistration" in text
    assert "Stage-B2 block" in text
    assert "25–29" in text


def test_main_writes_table_csv_and_charts(tmp_path):
    root = tmp_path / "results"
    o = _run(variant="O", bench="lcb", n_loops=0, correct=4)
    loop = _run(
        variant="R_b16-18_n2",
        bench="lcb",
        n_loops=2,
        correct=6,
        block=(16, 18),
        bridge="none",
    )
    o.save(root / "stageB" / "B_O" / "O" / "lcb.json")
    loop.save(root / "stageB" / "B_R_b16-18" / "R_b16-18_n2" / "lcb.json")
    stage_a = tmp_path / "stageA"
    stage_a.mkdir()
    (stage_a / "n2.json").write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "block": [16, 18],
                        "bridge": "none",
                        "n_loops": 2,
                        "nll_n_calibrated": 1.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report"
    code = report.main(
        [
            "--results-root",
            str(root),
            "--stage-a-dir",
            str(stage_a),
            "--out-dir",
            str(out),
            "--corpus-manifest",
            str(tmp_path / "missing.json"),
            "--plan",
            str(tmp_path / "missing-plan.json"),
        ]
    )
    assert code == 0
    assert (out / "trainfree_table.md").exists()
    assert (out / "trainfree_table.csv").exists()
    assert (out / "trainfree_summary.md").exists()
    assert (out / "accuracy_lcb.png").exists()
    assert (out / "calibrated_nll.png").exists()


def _stage_a_payloads() -> list[dict]:
    """Synthetic P1-lite fixture: 3 blocks x 2 bridges x N=1..4."""
    blocks = [(25, 29), (16, 18), (15, 29)]
    flops = {(25, 29): 0.142, (16, 18): 0.084, (15, 29): 0.427}
    payloads = []
    for n in (1, 2, 3, 4):
        variants = []
        for block_index, block in enumerate(blocks):
            for bridge, factor in (("none", 1.0), ("damped", 0.25)):
                nll = 0.18 + 0.3 * (block_index + 1) * factor * (n - 1)
                top1 = 0.94 - 0.04 * (block_index + 1) * factor * (n - 1)
                variants.append(
                    {
                        "block": list(block),
                        "bridge": bridge,
                        "n_loops": n,
                        "nll_n_calibrated": nll,
                        "top1_n": top1,
                        "kl_n1_n": 0.01 * (block_index + 1) * factor * (n - 1),
                        "flops_share_per_pass": flops[block],
                    }
                )
        payloads.append({"variants": variants})
    return payloads


def test_stage_a_rows_parse_delta_and_write_report(tmp_path):
    payloads = _stage_a_payloads()
    rows = report.build_stage_a_rows(payloads)

    assert len(rows) == 24
    assert [(row["block"], row["bridge"], row["n"]) for row in rows[:4]] == [
        ((25, 29), "none", 1),
        ((25, 29), "none", 2),
        ((25, 29), "none", 3),
        ((25, 29), "none", 4),
    ]
    baseline = next(
        row
        for row in rows
        if row["block"] == (25, 29) and row["bridge"] == "none" and row["n"] == 1
    )
    assert baseline["delta_nll"] == pytest.approx(0.0)
    assert baseline["original_nll"] == pytest.approx(0.18)

    third = next(
        row
        for row in rows
        if row["block"] == (25, 29) and row["bridge"] == "none" and row["n"] == 3
    )
    assert third["nll"] == pytest.approx(0.78)
    assert third["delta_nll"] == pytest.approx(0.6)
    assert third["original_nll"] == pytest.approx(0.18)
    assert third["kl"] == pytest.approx(0.02)
    assert third["flops_share"] == pytest.approx(0.142)

    out = tmp_path / "report"
    report.write_stage_a_report(rows, out)
    summary = (out / "stageA_summary.md").read_text(encoding="utf-8")
    assert "| block | bridge | N | calibrated NLL | delta vs original |" in summary
    assert "| 25-29 | none | 3 | 0.7800 | +0.6000 |" in summary
    assert "| 15-29 | damped | 4 |" in summary

    from PIL import Image

    for name in ("stageA_calibrated_nll.png", "stageA_top1.png"):
        with Image.open(out / name) as image:
            assert image.size == (1600, 600)


def test_stage_a_only_flag_needs_no_run_results(tmp_path):
    stage_a = tmp_path / "stageA"
    stage_a.mkdir()
    for index, payload in enumerate(_stage_a_payloads(), start=1):
        (stage_a / f"n{index}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    out = tmp_path / "report"
    code = report.main(
        [
            "--stage-a-only",
            "--stage-a-dir",
            str(stage_a),
            "--results-root",
            str(tmp_path / "missing-results"),
            "--out-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "stageA_summary.md").exists()
    assert (out / "stageA_calibrated_nll.png").exists()
    assert (out / "stageA_top1.png").exists()
    assert not (out / "trainfree_table.md").exists()
