"""The resolved optimizer is explicit in the run config and run.json (R7c-2).

``train.py`` has always turned the registered ``AdamW`` into bitsandbytes
``adamw_8bit`` on a CUDA device (and ``adamw_torch`` on CPU).  These tests pin
that resolution, the ``--prereg`` binding (a real mismatch is exit 2 on GPU),
the CPU fallback, and the bitsandbytes version recorded when 8-bit is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

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
        "--adapter-policy", "loop_only",
        "--lora-target-layers", "loop_block",
        *extra,
    ]


def _prereg_with_optimizer(tmp_path: Path, value: str) -> Path:
    """A copy of the real prereg whose Stage 1 optimizer is ``value``.

    Only the Stage 1 block is edited: the fallback RFT section has its own
    optimizer line, and the schema may move Stage 1's other fields.
    """
    text = PREREG.read_text(encoding="utf-8")
    start = text.index("\nstage1:\n")
    end = text.index("\nstage2:\n", start)
    block = text[start:end]
    pattern = re.compile(r"(?m)^  optimizer:.*$")
    if pattern.search(block):
        block = pattern.sub(f"  optimizer: {value}", block, count=1)
    else:
        block = block.rstrip("\n") + f"\n  optimizer: {value}\n"
    updated = tmp_path / f"prereg-{value}.yaml"
    updated.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return updated


def _force_cuda(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: available)


@pytest.mark.parametrize(
    ("requested", "cuda", "resolved"),
    [
        ("adamw", True, "adamw_8bit"),
        ("AdamW", True, "adamw_8bit"),
        ("adamw_8bit", True, "adamw_8bit"),
        ("adamw_torch", True, "adamw_torch"),
        ("adamw", False, "adamw_torch"),
        ("AdamW", False, "adamw_torch"),
        ("adamw_8bit", False, "adamw_torch"),
        ("adamw_torch", False, "adamw_torch"),
    ],
)
def test_resolve_optimizer_name(requested, cuda, resolved):
    assert train_script.resolve_optimizer_name(requested, cuda=cuda) == resolved


def test_build_sft_config_uses_resolved_name_on_gpu(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, True)
    config = train_script.build_sft_config(
        tmp_path,
        max_length=32,
        seed=0,
        num_train_epochs=1,
        optimizer="AdamW",
        bf16=False,
        use_cpu=True,
    )
    assert train_script.optimizer_display_name(config.optim) == "adamw_8bit"


def test_build_sft_config_falls_back_to_torch_on_cpu(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, False)
    config = train_script.build_sft_config(
        tmp_path,
        max_length=32,
        seed=0,
        num_train_epochs=1,
        optimizer="adamw_8bit",
        bf16=False,
        use_cpu=True,
    )
    assert train_script.optimizer_display_name(config.optim) == "adamw_torch"


def test_recipe_records_the_resolved_optimizer_on_gpu(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, True)
    prereg_path = _prereg_with_optimizer(tmp_path, "adamw_8bit")
    prereg = load_prereg(prereg_path)
    args = train_script._parse_args(_base("--prereg", str(prereg_path)))

    recipe = train_script.resolve_recipe(args, prereg)
    assert recipe.optimizer == "adamw_8bit"
    assert recipe.resolved_optimizer == "adamw_8bit"


def test_recipe_records_the_cpu_fallback(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, False)
    prereg_path = _prereg_with_optimizer(tmp_path, "adamw_8bit")
    prereg = load_prereg(prereg_path)
    args = train_script._parse_args(_base("--prereg", str(prereg_path)))

    recipe = train_script.resolve_recipe(args, prereg)
    assert recipe.optimizer == "adamw_8bit"
    assert recipe.resolved_optimizer == "adamw_torch"


def test_legacy_adamw_prereg_resolves_to_8bit_on_gpu(monkeypatch):
    _force_cuda(monkeypatch, True)
    prereg = load_prereg(PREREG)
    args = train_script._parse_args(_base("--prereg", str(PREREG)))

    recipe = train_script.resolve_recipe(args, prereg)
    assert recipe.optimizer in {"adamw", "adamw_8bit"}
    assert recipe.resolved_optimizer == "adamw_8bit"


@pytest.mark.parametrize("registered", ["AdamW", "adamw_8bit"])
def test_prereg_optimizer_mismatch_exits_2_on_gpu(monkeypatch, tmp_path, registered):
    _force_cuda(monkeypatch, True)
    path = _prereg_with_optimizer(tmp_path, registered)
    prereg = load_prereg(path)
    # --optimizer adamw_torch resolves to torch on GPU, the registered recipe
    # requires 8-bit: this is a real mismatch, not an alias.
    args = train_script._parse_args(_base("--prereg", str(path), "--optimizer", "adamw_torch"))
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_recipe(args, prereg)
    assert excinfo.value.code == 2


@pytest.mark.parametrize("registered", ["AdamW", "adamw_8bit"])
def test_equivalent_optimizer_alias_is_not_a_mismatch_on_gpu(monkeypatch, tmp_path, registered):
    _force_cuda(monkeypatch, True)
    path = _prereg_with_optimizer(tmp_path, registered)
    prereg = load_prereg(path)
    for requested in ("adamw", "adamw_8bit"):
        args = train_script._parse_args(_base("--prereg", str(path), "--optimizer", requested))
        recipe = train_script.resolve_recipe(args, prereg)
        assert recipe.resolved_optimizer == "adamw_8bit"


def test_cpu_fallback_is_not_a_prereg_mismatch(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, False)
    path = _prereg_with_optimizer(tmp_path, "adamw_8bit")
    prereg = load_prereg(path)
    args = train_script._parse_args(_base("--prereg", str(path), "--optimizer", "adamw_torch"))
    recipe = train_script.resolve_recipe(args, prereg)
    assert recipe.resolved_optimizer == "adamw_torch"


def test_run_config_records_resolved_optimizer_and_bitsandbytes(monkeypatch, tmp_path):
    _force_cuda(monkeypatch, True)
    path = _prereg_with_optimizer(tmp_path, "adamw_8bit")
    prereg = load_prereg(path)
    args = train_script._parse_args(_base("--prereg", str(path)))
    recipe = train_script.resolve_recipe(args, prereg)

    config = train_script.build_run_config(
        args, quant="bf16", max_length=32, recipe=recipe
    )
    assert config["optimizer"] == "adamw_8bit"
    assert config["optimizer_requested"] == "adamw_8bit"
    assert config["bitsandbytes_version"]
    assert config["bitsandbytes_version"] == train_script.bitsandbytes_version_for(
        "adamw_8bit"
    )


def test_run_config_records_cpu_fallback_without_bitsandbytes(monkeypatch):
    _force_cuda(monkeypatch, False)
    prereg = load_prereg(PREREG)
    args = train_script._parse_args(_base("--prereg", str(PREREG)))
    recipe = train_script.resolve_recipe(args, prereg)

    config = train_script.build_run_config(
        args, quant="bf16", max_length=32, recipe=recipe
    )
    assert config["optimizer"] == "adamw_torch"
    assert config["optimizer_requested"] in {"adamw", "adamw_8bit"}
    assert config["bitsandbytes_version"] is None


def _fake_trainer(optim: str):
    return SimpleNamespace(
        state=SimpleNamespace(global_step=3),
        tokens_seen=120,
        micro_batches_seen=2,
        args=SimpleNamespace(learning_rate=1e-4, optim=optim),
        _optimizer_steps=3,
        _clipped_steps=0,
        clip_rate=0.0,
        _last_grad_norm_total=None,
        _metric_history=[],
        loop_metrics=None,
    )


def test_run_json_records_resolved_optimizer_and_version(tmp_path):
    metadata = train_script.build_run_metadata(
        arm="O-H",
        size="e2b",
        quant="bf16",
        data=tmp_path / "data",
        base_tokens=100,
        target_tokens=100,
        max_length=32,
        seed=0,
        loop=None,
        adapters_chain=[],
        trainer=_fake_trainer("adamw_8bit"),
        manifest_sha256="deadbeef",
        manifest_target_tokens=100,
    )
    assert metadata["optimizer"] == "adamw_8bit"
    assert metadata["bitsandbytes_version"] == train_script.bitsandbytes_version_for(
        "adamw_8bit"
    )


def test_run_json_derives_the_cpu_fallback_from_the_trainer(tmp_path):
    metadata = train_script.build_run_metadata(
        arm="O-H",
        size="e2b",
        quant="bf16",
        data=tmp_path / "data",
        base_tokens=100,
        target_tokens=100,
        max_length=32,
        seed=0,
        loop=None,
        adapters_chain=[],
        trainer=_fake_trainer("adamw_torch"),
        manifest_sha256="deadbeef",
        manifest_target_tokens=100,
    )
    assert metadata["optimizer"] == "adamw_torch"
    assert metadata["bitsandbytes_version"] is None
    run_json = json.loads(
        train_script.write_run_json(tmp_path / "run", metadata).read_text(
            encoding="utf-8"
        )
    )
    assert run_json["optimizer"] == "adamw_torch"
