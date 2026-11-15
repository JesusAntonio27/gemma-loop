"""KV cache used by the padded generation engine.

The default ``DynamicCache`` grows with ``torch.cat`` every decode step.  For
the three full-attention *producer* layers of Gemma 4 (the only layers that
store a full-length KV cache; the looped/consumer layers share their states)
this produces a new allocation whose size grows by one token at every step, and
on Windows/WDDM the caching allocator fragments badly: the live cache is
~49 KB/token at batch 8, but reserved VRAM grew from 5.5 GiB at 6k tokens to
25.4 GiB at 16k (Task 5b measurement).

:func:`build_text_cache` returns a cache that preallocates the full-attention
layers at ``prompt + max_new_tokens`` and keeps the sliding-window layers
dynamic (they are capped at ``sliding_window``, so their ``cat`` costs at most
one window and never fragments meaningfully).  Only the KV-producer layers are
created: consumer layers never update the cache (``is_kv_shared_layer``), so
the 20 consumer layers of E2B need no cache slots at all.

``_ExactStaticLayer`` returns *only the written prefix* of its preallocated
tensor and reports the same mask sizes as ``DynamicLayer``.  Attention
therefore sees exactly the same tensor shapes as with the dynamic cache, which
keeps greedy decoding token-identical while removing the fragmentation.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import (
    Cache,
    DynamicSlidingWindowLayer,
    StaticLayer,
)

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class _ExactStaticLayer(StaticLayer):
    """A preallocated ``StaticLayer`` that behaves like a ``DynamicLayer``.

    ``StaticLayer.update`` returns the whole ``max_cache_len`` buffer, which is
    what a compiled model wants.  Here we return a view of the written prefix,
    and ``get_mask_sizes`` reports that prefix, so the attention shapes and the
    causal mask are identical to ``DynamicLayer``.  Allocation is static, so
    the allocator never has to grow a block.
    """

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__(max_cache_len=max_cache_len, **kwargs)
        # Python-side mirror of ``cumulative_length`` so the decode loop does not
        # force a device sync per attention layer.
        self._written = 0

    def update(self, key_states, value_states, *args, **kwargs):
        super().update(key_states, value_states, *args, **kwargs)
        self._written += int(key_states.shape[-2])
        return (
            self.keys[:, :, : self._written, :],
            self.values[:, :, : self._written, :],
        )

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Same contract as ``DynamicLayer``: the KV length *after* this update.
        return self._written + int(query_length), 0

    def reset(self):
        super().reset()
        self._written = 0


def num_cache_layers(config) -> int:
    """Number of layers that actually own a KV cache (producers only)."""
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return int(text_config.num_hidden_layers) - int(
        getattr(text_config, "num_kv_shared_layers", 0)
    )


def build_text_cache(config, max_cache_len: int) -> Cache:
    """Build the preallocated producer-only text cache for ``config``.

    ``config`` may be the multimodal top-level config or the text config.
    ``max_cache_len`` is the prompt length plus ``max_new_tokens``.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if max_cache_len < 1:
        raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")
    layer_types = list(getattr(text_config, "layer_types", None) or [])
    producer_layers = num_cache_layers(text_config)
    sliding_window = getattr(text_config, "sliding_window", None)
    layers = []
    for layer_idx in range(producer_layers):
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else FULL_ATTENTION
        if layer_type == SLIDING_ATTENTION:
            if sliding_window is None:
                raise ValueError(
                    f"layer {layer_idx} is sliding but the config has no sliding_window"
                )
            layers.append(DynamicSlidingWindowLayer(sliding_window=int(sliding_window)))
        elif layer_type == FULL_ATTENTION:
            layers.append(_ExactStaticLayer(max_cache_len=max_cache_len))
        else:
            raise ValueError(f"unsupported text layer type {layer_type!r} at layer {layer_idx}")
    return Cache(layers=layers, offloading=False)


def cache_bytes_estimate(config, max_cache_len: int, batch_size: int) -> int:
    """Upper bound of the cache bytes (diagnostic/derivation helper)."""
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    producer_layers = num_cache_layers(text_config)
    layer_types = list(getattr(text_config, "layer_types", None) or [])
    per_layer_config = getattr(text_config, "per_layer_config", None)
    first_layer_cfg = per_layer_config[0] if per_layer_config is not None else text_config
    kv_heads = int(getattr(first_layer_cfg, "num_key_value_heads", 1))
    total = 0
    for layer_idx in range(producer_layers):
        if per_layer_config is not None:
            try:
                layer_cfg = per_layer_config[layer_idx]
            except (IndexError, KeyError):
                layer_cfg = first_layer_cfg
        else:
            layer_cfg = first_layer_cfg
        head_dim = int(layer_cfg.head_dim)
        layer_len = max_cache_len
        if layer_types and layer_types[layer_idx] == SLIDING_ATTENTION:
            layer_len = min(layer_len, int(text_config.sliding_window))
        total += 2 * batch_size * kv_heads * layer_len * head_dim * 2  # bf16
    return total


__all__ = [
    "build_text_cache",
    "cache_bytes_estimate",
    "num_cache_layers",
]
