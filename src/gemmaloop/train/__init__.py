"""LoRA supervised fine-tuning with sampled loop counts.

The public helpers mirror the interfaces of Task 13: the deterministic loop
sampler, the FLOPs-matched token budget for the ``O-H-ext`` control, the
assistant-only label builder (Gemma 4 has no ``{% generation %}`` markers), the
trainer class and the callbacks used by ``scripts/train.py``.
"""

from __future__ import annotations

from gemmaloop.train.budget import (
    expected_layer_counts,
    flops_matched_tokens,
    per_layer_flops,
)
from gemmaloop.train.labels import assistant_labels
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS, sample_n_loops
from gemmaloop.train.trainer import (
    LoopSFTTrainer,
    SpillGuardCallback,
    TokenBudgetCallback,
    default_lora_config,
)

__all__ = [
    "DEFAULT_LOOP_PROBS",
    "LoopSFTTrainer",
    "SpillGuardCallback",
    "TokenBudgetCallback",
    "assistant_labels",
    "default_lora_config",
    "expected_layer_counts",
    "flops_matched_tokens",
    "per_layer_flops",
    "sample_n_loops",
]
