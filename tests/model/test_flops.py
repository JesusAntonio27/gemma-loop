"""FLOPs head-dim resolution against transformers' per-layer config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.snapshot import MODEL_REVISION
from gemmaloop.train.budget import flops_matched_tokens
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "e2b_text_config.json"
BASE_TOKENS = 40_000_000
EXPECTED = 66_150_274


def _budget(text_config: dict) -> int:
    return flops_matched_tokens(
        BASE_TOKENS,
        per_layer_flops(text_config),
        make_layer_plan(text_config),
        DEFAULT_LOOP_PROBS,
    )


def test_fixture_budget_is_the_registered_flops_matched_value():
    assert _budget(json.loads(FIXTURE.read_text(encoding="utf-8"))) == EXPECTED


def test_autoconfig_budget_matches_the_fixture():
    transformers = pytest.importorskip("transformers")
    try:
        config = transformers.AutoConfig.from_pretrained(
            "google/gemma-4-E2B-it",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - cache dependent
        pytest.skip(f"AutoConfig for the pinned snapshot is not cached: {exc}")
    text_config = config.text_config.to_dict()
    # transformers 5.22 moved global_head_dim into per_layer_config.
    assert text_config.get("per_layer_config")
    assert _budget(text_config) == EXPECTED


def test_per_layer_head_dim_overrides_global_and_default():
    text_config = {
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_kv_shared_layers": 0,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": None,
        "attention_k_eq_v": False,
        "vocab_size": 32,
        "per_layer_config": {"01": {"head_dim": 16}},
    }
    values = per_layer_flops(text_config)
    # Layer 1's q/o projections use 2 * 16 instead of 2 * 4 head dims.
    assert values[1] > values[0]
