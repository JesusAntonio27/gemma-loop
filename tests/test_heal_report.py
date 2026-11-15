"""Report rendering from fake monitoring logs (CPU, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.heal_report as heal_report
from gemmaloop.train.heal_monitor import (
    MONITOR_ALERTS_FILE,
    MONITOR_METRICS_FILE,
    MONITOR_PROBES_FILE,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "run-arm"
    run.mkdir(parents=True, exist_ok=True)
    metrics = []
    for index in range(5):
        tokens = 500_000 * (index + 1)
        metrics.append(
            {
                "kind": "step",
                "step": index + 1,
                "tokens_seen": tokens,
                "loss": 3.0 - 0.1 * index,
                "grad_norm_total": 1.0 + 0.1 * index,
                "primary_kl_teacher": 1.0 - 0.05 * index,
                "primary_kl_parent": 0.1,
                "hinge_active_max": 0.2,
                "bridge_g": {
                    "available": True,
                    "mean_abs": 0.01 * index,
                    "saturation_fraction": 0.0,
                    "all_zero": False,
                },
                "lora_b_norms": {"layer0": 1.0 + index},
                "grad_groups": {
                    "lora": {"grad_norm": 1.0 + index, "weight_norm": 2.0},
                    "bridge": {"grad_norm": 0.1, "weight_norm": 1.0},
                },
                "horizon_fractions": {"4": 0.7, "5": 0.075, "6": 0.075, "7": 0.075, "8": 0.075},
                "registered_horizon": {"4": 0.7, "5": 0.075, "6": 0.075, "7": 0.075, "8": 0.075},
                "teacher_integrity": {"available": True, "ok": True},
                "nonfinite": {"loss": 0, "grads": 0, "logits": 0},
                "tok_per_s": 500.0,
                "reserved_gib": 8.0,
                "memory_cap_gib": 10.0,
                "exits": {
                    "2": {"kl_teacher": 1.0, "kl_parent": 0.1, "hinge": 0.0, "hidden_change": 0.4, "period2": 0.1},
                    "3": {"kl_teacher": 0.9, "kl_parent": 0.1, "hinge": 0.1, "hidden_change": 0.3, "period2": 0.2},
                },
            }
        )
    _write_jsonl(run / MONITOR_METRICS_FILE, metrics)
    probes = []
    for index in range(2):
        tokens = 1_000_000 * (index + 1)
        probes.append(
            {
                "kind": "probe",
                "step": index + 1,
                "tokens_seen": tokens,
                "invariant": {"applicable": True, "ok": True},
                "anchor_kl": {"2": 0.1, "4": 0.2},
                "teacher_kl": {"2": 1.0, "4": 0.9},
                "teacher_kl_gap": {"4": 0.05},
                "calibrated_nll": {"1": 2.0, "2": 1.9, "3": 1.8, "4": 1.7},
                "generation": [
                    {
                        "prompt_index": 0,
                        "n_loops": 2,
                        "finish_reason": "eos",
                        "thinking_closed": True,
                        "answer_present": True,
                        "repetition_loop": False,
                        "text_path": str(run / "heal_probe_texts" / "sample.txt"),
                    }
                ],
            }
        )
    _write_jsonl(run / MONITOR_PROBES_FILE, probes)
    (run / MONITOR_ALERTS_FILE).write_text(
        json.dumps(
            {
                "alerts": [
                    {
                        "severity": "warn",
                        "rule": "grad_spike",
                        "message": "gradient norm spike",
                        "step": 3,
                        "tokens_seen": 1_500_000,
                        "metrics": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps({"arm": "L", "status": "complete"}), encoding="utf-8"
    )
    (run / "status.json").write_text(
        json.dumps({"phase": "complete"}), encoding="utf-8"
    )
    text_dir = run / "heal_probe_texts"
    text_dir.mkdir()
    (text_dir / "sample.txt").write_text("<|channel>thought\nx<channel|>answer", encoding="utf-8")
    return run


def test_report_rendering_on_fake_logs(tmp_path):
    run = _fake_run(tmp_path)
    assert heal_report.main(["--run", str(run)]) == 0
    markdown = (run / "heal_report.md").read_text(encoding="utf-8")
    assert "Traffic lights" in markdown
    assert "Gradient spike" in markdown
    assert "Generation sanity" in markdown
    assert "heal_probe_texts/sample.txt" in markdown
    payload = json.loads((run / "heal_report.json").read_text(encoding="utf-8"))
    assert payload["summary"]["probes"] == 2
    assert any(check["check"] == "Overall" for check in payload["traffic_light"])
    for name in (
        "heal_loss_components.png",
        "heal_grad_norms.png",
        "heal_bridge_lora.png",
        "heal_horizon_histogram.png",
        "heal_probe_kl.png",
        "heal_probe_nll.png",
        "heal_hidden_change.png",
        "heal_throughput_memory.png",
    ):
        assert (run / name).is_file(), name


def test_comparison_report_overlays_runs(tmp_path):
    first = _fake_run(tmp_path / "a")
    second = _fake_run(tmp_path / "b")
    out = tmp_path / "compare"
    assert (
        heal_report.main(
            [
                "--compare",
                str(first),
                str(second),
                "--labels",
                "arm-a",
                "arm-b",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    markdown = (out / "heal_compare.md").read_text(encoding="utf-8")
    assert "arm-a" in markdown and "arm-b" in markdown
    payload = json.loads((out / "heal_compare.json").read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 2
    assert (out / "heal_compare_loss.png").is_file()


def test_watch_once_renders_and_exits(tmp_path):
    run = _fake_run(tmp_path)
    assert heal_report.main(["--watch", str(run), "--once"]) == 0
    assert (run / "heal_report.md").is_file()


def test_is_live_before_status_file(tmp_path):
    run = tmp_path / "starting-run"
    run.mkdir()
    assert heal_report.is_live(run) is True
    (run / "DONE.json").write_text("{}", encoding="utf-8")
    assert heal_report.is_live(run) is False


def test_report_marks_teacher_fit_unavailable(tmp_path):
    run = _fake_run(tmp_path)
    probes = [
        json.loads(line)
        for line in (run / MONITOR_PROBES_FILE).read_text(encoding="utf-8").splitlines()
    ]
    for probe in probes:
        probe["teacher_fit_available"] = False
        probe["teacher_kl"] = {}
        probe["teacher_kl_gap"] = {}
    (run / MONITOR_PROBES_FILE).write_text(
        "\n".join(json.dumps(probe) for probe in probes) + "\n", encoding="utf-8"
    )
    assert heal_report.main(["--run", str(run)]) == 0
    markdown = (run / "heal_report.md").read_text(encoding="utf-8")
    assert "Teacher fit: unavailable" in markdown
    payload = json.loads((run / "heal_report.json").read_text(encoding="utf-8"))
    assert any(
        check["check"].startswith("Teacher fit") and check["status"] == "gray"
        for check in payload["traffic_light"]
    )
