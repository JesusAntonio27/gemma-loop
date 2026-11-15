"""Banded SDPA for Gemma 4's sliding-window layers during training (Task 13b).

With ``attn_implementation="sdpa"`` a sliding-window layer longer than the
window (512 tokens in E2B) gets an explicit ``[seq, seq]`` boolean mask.  No
flash/cuDNN kernel accepts it, so PyTorch's memory-efficient kernel computes
the full ``seq x seq`` score matrix and masks it: the cost of a sliding layer
grows like ``seq**2`` although each query only sees 512 keys.

:func:`banded_sliding_sdpa` computes the same attention with the same SDPA
kernel, one block of ``block`` queries at a time against the only keys it can
see, ``[block_start - window, block_end)``; all blocks run as one batched SDPA
call.  The mask applied inside a block is exactly transformers' sliding
pattern (``k <= q`` and ``q - k < window``), so every query attends to the same
keys with the same weights; only the kernel's internal tiling (hence bf16
rounding) differs.  Work drops from ``seq**2`` to ``seq * (block + window)``.

The banded path is used only when it is valid without looking at the mask:
training mode, batch size 1 (no padding), no KV cache (``q_len == kv_len``),
no dropout and the module's own ``sliding_window``.  Everything else falls back
to transformers' ``sdpa_attention_forward``.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

__all__ = [
    "banded_sliding_sdpa",
    "install_banded_sliding_attention",
    "sliding_band_mask",
]

DEFAULT_BLOCK = 512
DEFAULT_MIN_LENGTH = 2048
_FLAG = "_gemmaloop_banded_sliding"


@functools.lru_cache(maxsize=16)
def _band_mask_cached(n_blocks: int, block: int, window: int, device: str) -> torch.Tensor:
    b = torch.arange(n_blocks, device=device).view(n_blocks, 1, 1)
    i = torch.arange(block, device=device).view(1, block, 1)
    j = torch.arange(block + window, device=device).view(1, 1, block + window)
    # q = b*block + i, k = b*block - window + j  ->  q - k = i - j + window
    band = (j > i) & (j <= i + window)  # k <= q and q - k < window
    real_key = (b * block - window + j) >= 0  # keys before position 0 are padding
    return (band & real_key).unsqueeze(1)  # [n_blocks, 1, block, block + window]


def sliding_band_mask(n_blocks: int, block: int, window: int, device) -> torch.Tensor:
    """Boolean mask ``[n_blocks, 1, block, block + window]`` (True = attend)."""
    return _band_mask_cached(int(n_blocks), int(block), int(window), str(torch.device(device)))


def banded_sliding_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int,
    scaling: float | None,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Sliding-window causal attention for ``[1, heads, seq, dim]`` inputs.

    ``key``/``value`` may have fewer (GQA) heads.  Returns ``[1, heads, seq, dim]``.
    """
    batch, heads, length, dim = query.shape
    if batch != 1:
        raise ValueError("banded_sliding_sdpa supports batch size 1 only")
    kv_heads = key.shape[1]
    n_blocks = -(-length // block)
    pad_end = n_blocks * block - length

    q = F.pad(query, (0, 0, 0, pad_end)) if pad_end else query
    q = q.reshape(heads, n_blocks, block, dim).transpose(0, 1)  # [nb, H, B, D]

    def _bands(x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 0, window, pad_end))  # [1, Hkv, W + nb*B, D]
        bands = padded.unfold(2, block + window, block)  # [1, Hkv, nb, D, B + W]
        bands = bands.permute(2, 0, 1, 4, 3).reshape(n_blocks, kv_heads, block + window, dim)
        if kv_heads != heads:
            bands = repeat_kv(bands, heads // kv_heads)
        return bands

    mask = sliding_band_mask(n_blocks, block, window, query.device)
    out = F.scaled_dot_product_attention(q, _bands(key), _bands(value), attn_mask=mask, scale=scaling)
    out = out.transpose(0, 1).reshape(1, heads, n_blocks * block, dim)
    return out[:, :, :length] if pad_end else out


def _banded_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    settings = getattr(module, _FLAG, None)
    if (
        settings is not None
        and module.training
        and query.shape[0] == 1
        and query.shape[2] == key.shape[2]
        and query.shape[2] >= settings["min_length"]
        and dropout == 0.0
        and kwargs.get("sliding_window") == settings["window"]
        and kwargs.get("position_bias") is None
        and kwargs.get("cache") is None
        and (attention_mask is None or attention_mask.shape[0] == 1)
    ):
        out = banded_sliding_sdpa(
            query, key, value, window=settings["window"], scaling=scaling, block=settings["block"]
        )
        return out.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )


def install_banded_sliding_attention(
    text_model: torch.nn.Module,
    *,
    block: int = DEFAULT_BLOCK,
    min_length: int = DEFAULT_MIN_LENGTH,
) -> int:
    """Mark the sliding-window attention modules of ``text_model``; return how many.

    The process-wide ``"sdpa"`` entry of transformers' attention registry is
    pointed at a wrapper that only changes behaviour for marked modules.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if block < 1:
        raise ValueError("block must be positive")
    marked = 0
    for module in text_model.modules():
        window = getattr(module, "sliding_window", None)
        if getattr(module, "is_sliding", False) and window:
            setattr(module, _FLAG, {"window": int(window), "block": int(block), "min_length": int(min_length)})
            marked += 1
    if marked:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _banded_sdpa_attention_forward
    return marked
