"""The registered pilot mode in scripts/train.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.train as train
from gemmaloop.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


def _args(arm_id: str, *extra: str):
    return train._parse_args(
        [
            "--pilot-arm",
            arm_id,
            "--prereg",
            str(PREREG),
            "--data",
            str(ROOT / "data"),
            "--seed",
            "0",
            "--out",
            str(ROOT / "runs" / "pilot-test"),
            *extra,
        ]
    )


def test_pilot_mode_binds_the_registered_loop_arm_and_budget():
    args = _args("L-16-18-gch-e4b")
    train.apply_pilot_arm(args, load_prereg(PREREG))
    assert args.arm == "L"
    assert args.size == "e2b"
    assert args.tokens == 3_000_000
    assert args.bridge == "gch"
    assert args.block == "16-18"
    assert args.adapter_policy == "loop_only"
    assert args.lora_target_layers == "loop_block"
    assert args.dial_objective is True
    assert args.heal_teacher is False
    assert args.loop_probs is None


def test_pilot_mode_binds_the_oh_control():
    args = _args("OH-15-29-e4b")
    train.apply_pilot_arm(args, load_prereg(PREREG))
    assert args.arm == "O-H"
    assert args.block == "15-29"
    assert args.heal_teacher is True
    assert args.dial_objective is False


def test_pilot_mode_binds_the_single_exit_anytime_tax_horizon():
    args = _args("L-25-29-gch-e4b-single-exit-0901")
    train.apply_pilot_arm(args, load_prereg(PREREG))
    assert args.arm == "L"
    assert args.dial_objective is False
    assert args.heal_teacher is True
    assert args.pilot_single_exit is True
    assert args.loop_probs == "0,0.9,0.1"

    args = _args("L-25-29-gch-e4b-single-exit-n4")
    train.apply_pilot_arm(args, load_prereg(PREREG))
    assert args.loop_probs == "0,0,0,1"


def test_pilot_mode_refuses_cli_overrides_and_unknown_arms():
    args = _args("L-25-29-gch-e4b", "--tokens", "10")
    with pytest.raises(SystemExit) as exc:
        train.apply_pilot_arm(args, load_prereg(PREREG))
    assert exc.value.code == 2

    args = _args("L-25-29-gch-e4b", "--arm", "O-H")
    with pytest.raises(SystemExit) as exc:
        train.apply_pilot_arm(args, load_prereg(PREREG))
    assert exc.value.code == 2

    args = train._parse_args(
        [
            "--pilot-arm",
            "L-99-99-gch-e4b",
            "--prereg",
            str(PREREG),
            "--data",
            str(ROOT / "data"),
            "--seed",
            "0",
            "--out",
            str(ROOT / "runs" / "pilot-test"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        train.apply_pilot_arm(args, load_prereg(PREREG))
    assert exc.value.code == 2


def test_pilot_mode_refuses_registered_recipe_overrides():
    for extra in (
        ("--bridge-init", "0.5"),
        ("--max-length", "4096"),
        ("--init-adapter", str(ROOT / "runs" / "adapter")),
        ("--block", "16-18"),
    ):
        args = _args("L-25-29-gch-e4b", *extra)
        with pytest.raises(SystemExit) as exc:
            train.apply_pilot_arm(args, load_prereg(PREREG))
        assert exc.value.code == 2, extra


def test_without_pilot_mode_arm_size_tokens_stay_required():
    with pytest.raises(SystemExit):
        train._parse_args(
            [
                "--data",
                str(ROOT / "data"),
                "--seed",
                "0",
                "--out",
                str(ROOT / "runs" / "x"),
            ]
        )


def test_registered_run_refuses_smoke_and_a_disabled_monitor(monkeypatch):
    args = _args("L-25-29-gch-e4b", "--smoke")
    with pytest.raises(SystemExit) as exc:
        train.refuse_registered_monitor_escapes(args)
    assert exc.value.code == 2

    args = _args("L-25-29-gch-e4b")
    monkeypatch.setenv("GEMMALOOP_HEAL_MONITOR", "0")
    with pytest.raises(SystemExit) as exc:
        train.refuse_registered_monitor_escapes(args)
    assert exc.value.code == 2

    monkeypatch.delenv("GEMMALOOP_HEAL_MONITOR")
    train.refuse_registered_monitor_escapes(args)  # no refusal


def test_run_config_records_the_monitor_on_off_state(monkeypatch):
    args = _args("L-25-29-gch-e4b")
    train.apply_pilot_arm(args, load_prereg(PREREG))
    objective = train.resolve_objective(args, load_prereg(PREREG))
    monkeypatch.delenv("GEMMALOOP_HEAL_MONITOR", raising=False)
    config = train.build_run_config(
        args, quant="bf16", max_length=32, objective=objective
    )
    assert config["heal_monitor"] is True

    monkeypatch.setenv("GEMMALOOP_HEAL_MONITOR", "0")
    config = train.build_run_config(
        args, quant="bf16", max_length=32, objective=objective
    )
    assert config["heal_monitor"] is False
