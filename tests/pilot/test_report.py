"""The descriptive "why" report on fake results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.pilot.qualification import extract_qualification, load_qualification
from gemmaloop.pilot.report import write_report


def _payload() -> dict:
    return {
        "pilot": {
            "heal_tokens": 3_000_000,
            "cost_gpu_hours": "43-52",
            "held_out_only": True,
            "benchmarks": False,
        },
        "arms": {
            "L-25-29-gch-e4b": {
                "arm_id": "L-25-29-gch-e4b",
                "kind": "L",
                "block": [25, 29],
                "bridge": "gch",
                "teacher": "e4b",
                "nll": {"1": 1.20, "2": 1.10, "3": 1.08, "4": 1.06, "6": 1.05, "8": 1.04},
                "delta_loop": 0.10,
                "delta_arm": 0.20,
                "delta_bar_arm": 0.21,
                "monotonic": True,
                "temperatures": {"1": 1.0, "2": 1.1, "3": 1.1, "4": 1.1, "6": 1.1, "8": 1.1},
                "training": {
                    "loss": [{"step": 10, "loss": 2.4}, {"step": 20, "loss": 2.1}],
                    "tokens": 3_000_000,
                    "wall_seconds": 3600.0,
                    "tokens_per_second": 833.3,
                },
                "spans": {"thinking": 0.08, "answer": 0.12},
                "positions": {"<=1k": 0.09, "1-2k": 0.11, "2-4k": 0.13},
                "teacher_kl": {"initial": 0.50, "final": 0.20, "gap_closed": 0.60},
            },
            "OH-25-29-e4b": {
                "arm_id": "OH-25-29-e4b",
                "kind": "O-H",
                "block": [25, 29],
                "bridge": "gch",
                "teacher": "e4b",
                "nll": {"1": 1.30},
                "training": {"loss": [{"step": 10, "loss": 2.6}], "tokens": 3_000_000, "wall_seconds": 4000.0, "tokens_per_second": 750.0},
            },
        },
        "teacher_step": {
            "chosen": "e4b",
            "tie_break": False,
            "gains": {
                "e4b": {"l_n4": 0.12, "oh_n1": 0.10, "mean": 0.11},
                "12b": {"l_n4": 0.08, "oh_n1": 0.07, "mean": 0.075},
            },
        },
        "selection": {
            "stop": False,
            "selected": "L-25-29-gch-e4b",
            "eligible": ["L-25-29-gch-e4b"],
            "monotonic_pass": ["L-25-29-gch-e4b"],
            "filter_applied": True,
        },
        "teachers": {
            "e4b": {"entropy": 2.1, "top1_confidence": 0.55},
            "12b": {"entropy": 1.7, "top1_confidence": 0.64},
        },
        "qualification": {
            "e4b": {"kl_mean": 0.0004, "top1": 0.999, "tail_error_mean": 0.0001},
            "12b": {"kl_mean": 0.003, "top1": 0.97, "tail_error_mean": 0.001},
        },
    }


def _png(path: Path) -> None:
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_report_writes_markdown_json_and_charts(tmp_path):
    paths = write_report(_payload(), tmp_path)
    assert {"json", "markdown", "per_n", "loss", "teacher", "gains"} <= set(paths)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["selection"]["selected"] == "L-25-29-gch-e4b"
    text = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "descriptive" in text.lower()
    assert "never a selection input" in text.lower()
    assert "Recipe so far" in text
    assert "E4B" in text
    _png(Path(paths["per_n"]))
    _png(Path(paths["loss"]))
    _png(Path(paths["teacher"]))
    _png(Path(paths["gains"]))


def test_report_is_resilient_to_partial_results(tmp_path):
    payload = _payload()
    payload["arms"] = {}
    payload["selection"] = None
    payload["teacher_step"] = None
    paths = write_report(payload, tmp_path)
    assert Path(paths["markdown"]).is_file()
    text = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "no results" in text.lower() or "pending" in text.lower()


def test_qualification_loader_tolerates_missing_files_and_fields(tmp_path):
    assert load_qualification(tmp_path / "absent.json") == {}
    assert load_qualification(None) == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_qualification(broken) == {}

    payload = {
        "teachers": {
            "e4b": {"metrics": {"mean_kl": 0.0004, "top1_agreement": 0.999}},
            "12b": {"runtime": {"quant": "nf4"}, "extra": "ignored"},
        }
    }
    extracted = extract_qualification(payload)
    assert extracted["e4b"]["kl_mean"] == 0.0004
    assert extracted["e4b"]["top1"] == 0.999
    assert extracted["12b"]["quant"] == "nf4"
    assert "extra" not in extracted["12b"]


def test_qualification_loader_accepts_a_list_of_results():
    payload = [
        {"teacher": "google/gemma-4-12B-it", "kl_mean": 0.002, "top1": 0.97},
        {"teacher": "google/gemma-4-E4B-it", "full_kl_mean": 0.0003},
    ]
    extracted = extract_qualification(payload)
    assert extracted["12b"]["kl_mean"] == 0.002
    assert extracted["e4b"]["kl_mean"] == 0.0003


def test_report_renders_teacher_kl_for_every_n(tmp_path):
    payload = _payload()
    payload["arms"]["L-25-29-gch-e4b"]["teacher_kl"] = {
        "initial": {"1": 0.50, "2": 0.60},
        "final": {"1": 0.50, "2": 0.20},
        "gap_closed": {"1": 0.0, "2": 2.0 / 3.0},
        "n_values": [1, 2],
        "note": "N=1 is structurally the original model for L arms",
    }
    paths = write_report(payload, tmp_path)
    text = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "initial KL" in text
    assert "structurally the original" in text
    assert "0.67" in text
