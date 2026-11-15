"""Tests for the evaluation-index fail-closed policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_heal_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_heal_data_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec(module, name, loader, *, mandatory=True, expected=2):
    return module.EvalSpec(
        name=name,
        loader=loader,
        expected=expected,
        hf_id=f"fake/{name}",
        split="test",
        mandatory=mandatory,
    )


def test_mandatory_eval_failure_fails_closed(monkeypatch):
    module = _load_script()

    def boom(limit):
        raise RuntimeError("gated")

    monkeypatch.setattr(
        module,
        "EVAL_SPECS",
        (
            _spec(module, "mandatory", boom),
            _spec(module, "optional", lambda limit: iter(["optional text"]), mandatory=False),
        ),
    )
    with pytest.raises(RuntimeError, match="mandatory evaluation index incomplete"):
        module.gather_eval_texts(eval_limit=1, extra_eval_files=[])


def test_allow_missing_eval_records_failure(monkeypatch):
    module = _load_script()

    def boom(limit):
        raise RuntimeError("gated")

    monkeypatch.setattr(
        module,
        "EVAL_SPECS",
        (_spec(module, "mandatory", boom),),
    )
    texts, statuses = module.gather_eval_texts(
        eval_limit=1, extra_eval_files=[], allow_missing=True
    )
    assert texts == []
    assert statuses["mandatory"]["status"] == "failed"
    assert "gated" in statuses["mandatory"]["error"]


def test_short_mandatory_benchmark_fails_closed(monkeypatch):
    module = _load_script()
    monkeypatch.setattr(
        module,
        "EVAL_SPECS",
        (_spec(module, "short", lambda limit: iter(["only one"]), expected=2),),
    )
    with pytest.raises(RuntimeError, match="expected at least 2"):
        module.gather_eval_texts(eval_limit=None, extra_eval_files=[])


def test_optional_extra_file_missing_only_warns(monkeypatch, tmp_path):
    module = _load_script()
    monkeypatch.setattr(module, "EVAL_SPECS", ())
    texts, statuses = module.gather_eval_texts(
        eval_limit=1,
        extra_eval_files=[tmp_path / "absent.jsonl"],
    )
    assert texts == []
    assert statuses["file:absent.jsonl"]["status"] == "missing"


def test_parse_args_accepts_max_length_and_length_mode():
    module = _load_script()
    args = module._parse_args(["--max-length", "512"])
    assert args.max_length == 512
    assert args.length_mode == "filter"
    args = module._parse_args(["--max-length", "512", "--length-mode", "truncate"])
    assert args.length_mode == "truncate"

