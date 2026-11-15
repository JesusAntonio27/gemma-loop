"""CPU equivalence of the banded sliding-window SDPA (Task 13b)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Gemma4ForCausalLM
from transformers.integrations.sdpa_attention import repeat_kv

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.train.sliding_attention import (
    banded_sliding_sdpa,
    install_banded_sliding_attention,
)
from tests.model.tiny import tiny_config


def _full_mask_sdpa(q, k, v, window, scaling):
    length = q.shape[2]
    i = torch.arange(length)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    groups = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(
        q, repeat_kv(k, groups), repeat_kv(v, groups), attn_mask=mask[None, None], scale=scaling
    )


@pytest.mark.parametrize(
    "length,window,block,kv_heads",
    [(64, 16, 16, 1), (70, 16, 16, 1), (70, 16, 8, 2), (5, 16, 16, 1), (100, 7, 32, 4), (33, 32, 64, 1)],
)
def test_banded_matches_full_mask(length, window, block, kv_heads):
    torch.manual_seed(length + window + block)
    heads, dim = 4, 8
    q = torch.randn(1, heads, length, dim, requires_grad=True)
    k = torch.randn(1, kv_heads, length, dim, requires_grad=True)
    v = torch.randn(1, kv_heads, length, dim, requires_grad=True)
    out = banded_sliding_sdpa(q, k, v, window=window, scaling=0.7, block=block)

    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref = _full_mask_sdpa(q2, k2, v2, window, 0.7)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    upstream = torch.randn_like(ref)
    (out * upstream).sum().backward()
    (ref * upstream).sum().backward()
    for got, expected in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def _tiny(seed: int = 0):
    config = tiny_config()
    config.final_logit_softcapping = 30.0
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(config)
    apply_loops(model.model, make_layer_plan(config.to_dict()), LoopConfig(n_loops=1))
    torch.manual_seed(5)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj", "down_proj"])
    return get_peft_model(model, lora)


def _loss_and_grads(model, input_ids):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
    out.loss.backward()
    return out.loss.detach(), {
        n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
    }


@pytest.mark.parametrize("n_loops", [1, 3])
def test_banded_attention_keeps_model_loss_and_gradients(n_loops):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # float64: the banded and full-mask paths then agree to rounding of the
    # same sums (in fp32 the CPU flash kernel's blocking adds ~1e-7 noise that
    # this random tiny model amplifies through 3 loops)
    base = _tiny().double()
    banded = copy.deepcopy(base)
    for model in (base, banded):
        set_n_loops(model.get_base_model().model, n_loops)
        model.train()
    torch.manual_seed(9)
    input_ids = torch.randint(3, 256, (1, 57))  # > 3 windows (sliding_window=16), not a block multiple

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        for model in (base, banded):
            model.config._attn_implementation = "sdpa"
        ref_loss, ref_grads = _loss_and_grads(base, input_ids)
        marked = install_banded_sliding_attention(banded.get_base_model().model, block=8, min_length=1)
        assert marked == sum(1 for t in tiny_config().layer_types if t == "sliding_attention")
        loss, grads = _loss_and_grads(banded, input_ids)
        # marked modules in eval mode fall back to the standard path
        banded.eval()
        with torch.no_grad():
            eval_logits = banded(input_ids=input_ids).logits
            base.eval()
            ref_logits = base(input_ids=input_ids).logits
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original

    torch.testing.assert_close(loss, ref_loss, rtol=1e-12, atol=1e-12)
    assert grads.keys() == ref_grads.keys() and grads
    for name, grad in ref_grads.items():
        torch.testing.assert_close(grads[name], grad, rtol=1e-9, atol=1e-12, msg=name)
    assert torch.equal(eval_logits, ref_logits)


def test_banded_path_skips_short_sequences_and_batches(monkeypatch):
    import gemmaloop.train.sliding_attention as sa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    calls = []
    real = sa.banded_sliding_sdpa
    monkeypatch.setattr(sa, "banded_sliding_sdpa", lambda *a, **k: calls.append(a[0].shape) or real(*a, **k))
    model = _tiny()
    model.config._attn_implementation = "sdpa"
    model.train()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        install_banded_sliding_attention(model.get_base_model().model, block=8, min_length=40)
        model(input_ids=torch.randint(3, 256, (1, 30)))  # shorter than min_length
        assert calls == []
        model(input_ids=torch.randint(3, 256, (2, 48)))  # batch of 2
        assert calls == []
        model(input_ids=torch.randint(3, 256, (1, 48)))
        assert calls  # every sliding layer of the single long sequence
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
