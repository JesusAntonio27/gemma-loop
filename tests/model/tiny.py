"""A tiny, CPU-runnable Gemma 4 text model for the looped-forward tests.

The shape of the config mirrors the E2B layout on a much smaller scale: 15
layers, a period of 4 sliding + 1 full attention layers, and 10 KV-sharing
layers (the last two attention periods).  That makes the layer plan
prelude 0-4 / loop 5-9 / coda 10-14, i.e. the looped block lives entirely in
the KV-consumer region.
"""

from __future__ import annotations

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

NUM_LAYERS = 15
LAYER_TYPES = (["sliding_attention"] * 4 + ["full_attention"]) * 3


def tiny_config() -> Gemma4TextConfig:
    """Return the tiny ``Gemma4TextConfig`` shared by the model and LM head."""
    return Gemma4TextConfig(
        num_hidden_layers=NUM_LAYERS,
        layer_types=list(LAYER_TYPES),
        num_kv_shared_layers=10,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        vocab_size=256,
        vocab_size_per_layer_input=256,
        sliding_window=16,
        use_double_wide_mlp=False,
        max_position_embeddings=64,
    )


def tiny_text_model(seed: int = 0) -> Gemma4TextModel:
    """Build a deterministic tiny ``Gemma4TextModel`` in eval mode."""
    torch.manual_seed(seed)
    model = Gemma4TextModel(tiny_config())
    model.eval()
    return model
