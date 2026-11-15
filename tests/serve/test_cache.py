"""CPU tests for the preallocated producer-only KV cache."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import Gemma4TextConfig
from transformers.cache_utils import DynamicSlidingWindowLayer

from gemmaloop.serve.cache import (
    _ExactStaticLayer,
    build_text_cache,
    cache_bytes_estimate,
    num_cache_layers,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "e2b_text_config.json"


def _config() -> Gemma4TextConfig:
    return Gemma4TextConfig(**json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_build_text_cache_only_creates_producer_layers():
    config = _config()
    assert num_cache_layers(config) == 15
    cache = build_text_cache(config, max_cache_len=1024)
    assert len(cache.layers) == 15
    assert sum(isinstance(layer, _ExactStaticLayer) for layer in cache.layers) == 3
    assert (
        sum(isinstance(layer, DynamicSlidingWindowLayer) for layer in cache.layers) == 12
    )
    assert cache.is_compileable is False  # dynamic sliding layers disable compilation


def test_exact_static_layer_returns_only_the_written_prefix():
    layer = _ExactStaticLayer(max_cache_len=16)
    key = torch.randn(2, 1, 5, 4)
    value = torch.randn(2, 1, 5, 4)
    cached_key, cached_value = layer.update(key, value)
    assert cached_key.shape == (2, 1, 5, 4)
    assert cached_value.shape == (2, 1, 5, 4)
    assert layer.get_mask_sizes(1) == (6, 0)  # 5 prompt + 1 query

    more_key = torch.randn(2, 1, 1, 4)
    cached_key, _ = layer.update(more_key, more_key)
    assert cached_key.shape == (2, 1, 6, 4)
    assert layer.get_mask_sizes(1) == (7, 0)


def test_cache_bytes_estimate_is_linear_and_uses_full_attention_head_dim():
    config = _config()
    small = cache_bytes_estimate(config, 1024, 1)
    large = cache_bytes_estimate(config, 2048, 1)
    # Only the three full-attention producer layers grow past the window; the
    # rest are capped at sliding_window=512, so doubling the length is not 2x.
    assert large > small
    full_only = 3 * 2 * 1 * (2048 - 1024) * 512 * 2
    assert large - small == full_only


def test_cache_policy_threshold_and_size():
    from types import SimpleNamespace

    import torch as _torch

    from gemmaloop.serve.generate import Sampling, _cache_for

    config = _config()
    model = SimpleNamespace(config=config)
    inputs = {"input_ids": _torch.zeros(1, 8, dtype=_torch.long)}
    assert _cache_for(model, inputs, Sampling(max_new_tokens=4096)) is None
    cache = _cache_for(model, inputs, Sampling(max_new_tokens=4097))
    assert cache is not None and len(cache.layers) == 15
    # cache_new_tokens lets calibration preallocate a 32k bucket while decoding
    # only a short window.
    assert _cache_for(
        model, inputs, Sampling(max_new_tokens=512), cache_new_tokens=32768
    ) is not None
