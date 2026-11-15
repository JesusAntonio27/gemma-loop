"""Stage 1 optimiser recipe binding (R7b-2).

Under ``--prereg`` train.py must read the registered recipe from prereg.yaml and
refuse any CLI value that differs (exit 2).  The loader accepts the current
``tokens_per_optimizer_step`` schema and the updated
``rows_per_optimizer_step`` schema so the two branches merge cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.train as train_script
from gemmaloop.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


def _base(*extra: str) -> list[str]:
    return [
        "--arm", "L",
        "--size", "e2b",
        "--data", "unused",
        "--tokens", "1",
        "--seed", "0",
        "--out", "unused",
        # The registered dial objective requires these; without them main
        # exits 2 before it reaches the recipe binding.
        "--adapter-policy", "loop_only",
        "--lora-target-layers", "loop_block",
        *extra,
    ]


def _registered_recipe(prereg):
    lora = prereg.architecture.lora
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": prereg.stage1.rows_per_optimizer_step,
        "learning_rate": lora.lr,
        "bridge_lr": lora.bridge_lr,
        "loop_ple_lr": lora.loop_ple_lr,
        "gradient_clip": prereg.stage1.gradient_clip,
        "warmup_fraction": 0.03,
        "optimizer": prereg.stage1.optimizer.strip().lower(),
    }


def test_prereg_run_reads_every_recipe_value():
    prereg = load_prereg(PREREG)
    args = train_script._parse_args(_base("--prereg", str(PREREG)))

    recipe = train_script.resolve_recipe(args, prereg)
    expected = _registered_recipe(prereg)

    assert recipe.registered is True
    assert recipe.per_device_train_batch_size == expected["per_device_train_batch_size"] == 1
    assert recipe.gradient_accumulation_steps == expected["gradient_accumulation_steps"] == 16
    assert recipe.learning_rate == pytest.approx(expected["learning_rate"]) == pytest.approx(2e-4)
    assert recipe.bridge_lr == pytest.approx(expected["bridge_lr"]) == pytest.approx(3e-3)
    assert recipe.loop_ple_lr == pytest.approx(expected["loop_ple_lr"]) == pytest.approx(1e-3)
    assert recipe.gradient_clip == pytest.approx(expected["gradient_clip"]) == pytest.approx(1.0)
    assert recipe.warmup_fraction == pytest.approx(expected["warmup_fraction"]) == pytest.approx(0.03)
    assert recipe.optimizer == expected["optimizer"]
    assert recipe.optimizer in train_script.ADAMW_ALIASES
    # The resolved name is the one that runs (torch on CPU, 8-bit on GPU).
    assert recipe.resolved_optimizer == train_script.resolve_optimizer_name(
        expected["optimizer"]
    )


def test_train_defaults_match_the_registered_recipe():
    prereg = load_prereg(PREREG)
    expected = _registered_recipe(prereg)
    args = train_script._parse_args(_base())

    # The CLI defaults themselves are the registered values.
    assert args.per_device_train_batch_size == expected["per_device_train_batch_size"]
    assert args.gradient_accumulation_steps == expected["gradient_accumulation_steps"]
    assert args.lr == pytest.approx(expected["learning_rate"])
    assert args.bridge_lr == pytest.approx(expected["bridge_lr"])
    assert args.loop_ple_lr == pytest.approx(expected["loop_ple_lr"])
    assert args.gradient_clip == pytest.approx(expected["gradient_clip"])
    assert args.warmup_ratio == pytest.approx(expected["warmup_fraction"])
    # ``adamw`` and ``adamw_8bit`` are the same registered optimizer.
    assert args.optimizer in train_script.ADAMW_ALIASES
    assert train_script.resolve_optimizer_name(args.optimizer) == (
        train_script.resolve_optimizer_name(expected["optimizer"])
    )

    config = train_script.build_sft_config(
        Path("unused"),
        max_length=32,
        seed=0,
        num_train_epochs=1,
    )
    assert config.per_device_train_batch_size == expected["per_device_train_batch_size"]
    assert config.gradient_accumulation_steps == expected["gradient_accumulation_steps"]
    assert config.learning_rate == pytest.approx(expected["learning_rate"])
    assert config.max_grad_norm == pytest.approx(expected["gradient_clip"])
    assert config.get_warmup_steps(1000) == 30


_TOKENS_LINE = "tokens_per_optimizer_step: 8192"
_ROWS_LINE = "rows_per_optimizer_step: 16"


def _legacy_and_updated(rows: int) -> tuple[str, str]:
    """Legacy (tokens key) and updated (rows key) variants of the committed prereg."""
    text = PREREG.read_text(encoding="utf-8")
    if _ROWS_LINE in text:
        legacy = text.replace(_ROWS_LINE, _TOKENS_LINE, 1)
        updated = text.replace(_ROWS_LINE, f"rows_per_optimizer_step: {rows}", 1)
    else:
        assert _TOKENS_LINE in text
        legacy = text
        updated = text.replace(_TOKENS_LINE, f"rows_per_optimizer_step: {rows}", 1)
    return legacy, updated


def test_loader_accepts_tokens_or_rows_schema(tmp_path):
    legacy_text, updated_text = _legacy_and_updated(16)
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(legacy_text, encoding="utf-8")
    updated = tmp_path / "updated.yaml"
    updated.write_text(updated_text, encoding="utf-8")

    for path in (legacy, updated):
        prereg = load_prereg(path)
        assert prereg.stage1.rows_per_optimizer_step == 16
        args = train_script._parse_args(
            _base("--prereg", str(path), "--out", str(tmp_path / "out"))
        )
        assert (
            train_script.resolve_recipe(args, prereg).gradient_accumulation_steps
            == 16
        )


def test_updated_rows_field_is_the_effective_accumulation(tmp_path):
    _legacy, updated_text = _legacy_and_updated(24)
    updated = tmp_path / "updated.yaml"
    updated.write_text(updated_text, encoding="utf-8")
    prereg = load_prereg(updated)
    args = train_script._parse_args(
        _base(
            "--prereg", str(updated),
            "--out", str(tmp_path / "out"),
            "--gradient-accumulation-steps", "24",
        )
    )
    assert train_script.resolve_recipe(args, prereg).gradient_accumulation_steps == 24


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lr", "1e-4"),
        ("--bridge-lr", "1e-2"),
        ("--loop-ple-lr", "2e-3"),
        ("--gradient-accumulation-steps", "8"),
        ("--gradient-clip", "0.5"),
        ("--warmup-ratio", "0.1"),
        ("--per-device-train-batch-size", "2"),
        # ``--optimizer adamw_8bit`` is not an override: it is the registered
        # optimizer under its explicit name.  A real mismatch needs a CUDA
        # device (see test_optimizer_resolution.py).
    ],
)
def test_each_cli_recipe_override_is_refused_with_exit_2(tmp_path, flag, value):
    args = _base(
        "--prereg", str(PREREG),
        "--out", str(tmp_path / "out"),
        flag, value,
    )
    with pytest.raises(SystemExit) as excinfo:
        train_script.main(args)
    assert excinfo.value.code == 2


def test_recipe_values_reach_the_sft_config(tmp_path):
    prereg = load_prereg(PREREG)
    args = train_script._parse_args(_base("--prereg", str(PREREG)))
    recipe = train_script.resolve_recipe(args, prereg)

    config = train_script.build_sft_config(
        tmp_path,
        max_length=32,
        seed=0,
        num_train_epochs=1,
        per_device_train_batch_size=recipe.per_device_train_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        warmup_fraction=recipe.warmup_fraction,
        gradient_clip=recipe.gradient_clip,
        optimizer=recipe.optimizer,
    )
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 16
    assert config.learning_rate == pytest.approx(2e-4)
    assert config.max_grad_norm == pytest.approx(1.0)
    assert config.get_warmup_steps(1000) == 30
    assert str(getattr(config.optim, "value", config.optim)) in {
        "adamw_torch", "adamw_8bit"
    }
