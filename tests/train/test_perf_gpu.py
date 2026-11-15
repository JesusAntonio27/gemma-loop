"""GPU equivalence of the training perf switches on real Gemma 4 E2B (Task 13b).

Loads E2B once, wraps it with the project LoRA config, gradient checkpointing
(non-reentrant) and TRL's fused head, and checks under bf16 autocast that

* the lean LM head gives exactly TRL's loss and the same LoRA/bridge gradients;
* removing the input-grad hook and offloading the vision tower change neither;
* both fused heads agree with transformers' plain full-logits loss.

The GPU backward is not deterministic: two identical TRL steps (same seed, same
dropout masks) already differ by ~1% relative gradient norm (cosine ~0.99995).
Gradients are therefore compared against that measured noise floor, while the
forward loss must match exactly.

Run with ``uv run pytest -m gpu tests/train/test_perf_gpu.py -q``.
"""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LENGTH = 768
N_LOOPS = 3


@pytest.fixture(scope="module")
def e2b():
    from peft import get_peft_model
    from trl.trainer.utils import add_fused_lm_head

    from gemmaloop.model.load import ModelSpec, load_model
    from gemmaloop.model.looped import LoopConfig, set_n_loops
    from gemmaloop.train.trainer import default_lora_config, enable_loop_parameters, freeze_base_model

    model, processor, _plan = load_model(
        ModelSpec("e2b", "bf16", loop=LoopConfig(n_loops=1, ple_mode="shared"))
    )
    text_model = model.model.language_model
    freeze_base_model(model)
    torch.manual_seed(0)
    peft_model = get_peft_model(model, default_lora_config())
    enable_loop_parameters(text_model)
    with torch.no_grad():  # non-zero B so every LoRA gradient is exercised
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 1e-3)
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.enable_input_require_grads()
    peft_model.train()
    set_n_loops(text_model, N_LOOPS)
    add_fused_lm_head(peft_model.get_base_model(), outputs=("log_probs", "entropy", "is_top1"))

    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(3, 255000, (1, LENGTH), generator=generator)
    labels = input_ids.clone()
    labels[:, :64] = -100
    batch = {
        "input_ids": input_ids.cuda(),
        "attention_mask": torch.ones_like(input_ids).cuda(),
        "labels": labels.cuda(),
    }
    yield peft_model, batch, getattr(processor, "tokenizer", processor)
    del peft_model, model
    gc.collect()
    torch.cuda.empty_cache()


def _step(model, batch, **kwargs):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)  # identical LoRA dropout masks in every run
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch, **kwargs)
    out.loss.backward()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return out.loss.detach().float().item(), grads


def _distance(a: dict, b: dict) -> tuple[float, float]:
    """(cosine, relative L2 distance) over all trainable gradients."""
    assert a.keys() == b.keys() and a
    num = sum(float((a[k] * b[k]).sum()) for k in a)
    na = sum(float((a[k] ** 2).sum()) for k in a) ** 0.5
    nb = sum(float((b[k] ** 2).sum()) for k in a) ** 0.5
    diff = sum(float(((a[k] - b[k]) ** 2).sum()) for k in a) ** 0.5
    return num / (na * nb), diff / na


def test_perf_switches_keep_loss_and_gradients(e2b):
    from gemmaloop.train.fused_ce import install_lean_lm_head
    from gemmaloop.train.perf import VisionOffloadCallback, disable_input_require_grads

    model, batch, _tokenizer = e2b
    plain_loss, plain_grads = _step(model, batch)  # transformers' full-logits loss
    trl_loss, trl_grads = _step(model, batch, fused_lm_head=True)
    trl_again, trl_grads_again = _step(model, batch, fused_lm_head=True)
    assert any("loop_bridge_logit" in name for name in trl_grads)
    assert trl_again == trl_loss
    noise_cos, noise_rel = _distance(trl_grads, trl_grads_again)
    rel_bound = max(2.5 * noise_rel, 0.02)
    cos_bound = min(1.0 - 4.0 * (1.0 - noise_cos), 0.9998)

    install_lean_lm_head(model, token_chunk=4096, vocab_chunk=32768, compact=False)
    same_tiles_loss, same_tiles_grads = _step(model, batch, fused_lm_head=True)
    install_lean_lm_head(model)
    lean_loss, lean_grads = _step(model, batch, fused_lm_head=True)
    assert disable_input_require_grads(model) >= 1
    nograd_loss, nograd_grads = _step(model, batch, fused_lm_head=True)
    callback = VisionOffloadCallback(model)
    callback.offload()
    try:
        offload_loss, offload_grads = _step(model, batch, fused_lm_head=True)
    finally:
        callback.restore()

    # TRL vs transformers: softcap in fp32 vs bf16 -> ~1e-4 relative on the loss
    assert abs(trl_loss - plain_loss) / plain_loss < 1e-3
    cos, rel = _distance(trl_grads, plain_grads)
    assert cos >= cos_bound and rel <= rel_bound, (cos, rel, noise_cos, noise_rel)
    for name, loss, grads in (
        ("lean/TRL tiles", same_tiles_loss, same_tiles_grads),
        ("lean/default", lean_loss, lean_grads),
        ("lean/no input grads", nograd_loss, nograd_grads),
        ("lean/no input grads/vision offload", offload_loss, offload_grads),
    ):
        assert abs(loss - trl_loss) <= 1e-6 * abs(trl_loss), (name, loss, trl_loss)
        cos, rel = _distance(trl_grads, grads)
        assert cos >= cos_bound and rel <= rel_bound, (name, cos, rel, noise_cos, noise_rel)


def _text_batch(tokenizer, length: int) -> dict:
    """Deterministic, predictable English (a random-token input is chaotic:
    there any kernel change, even SDPA math vs memory-efficient, moves the
    gradient direction arbitrarily)."""
    text = " ".join(
        f"Step {i}: the farmer adds {i % 7} apples to basket {i % 5}, which now holds {i * (i % 7)} apples."
        for i in range(600)
    )
    ids = torch.tensor([tokenizer(text, add_special_tokens=True)["input_ids"][:length]])
    assert ids.shape[1] == length
    labels = ids.clone()
    labels[:, :64] = -100
    return {"input_ids": ids.cuda(), "attention_mask": torch.ones_like(ids).cuda(), "labels": labels.cuda()}


def test_banded_sliding_attention_matches_every_layer_call(e2b):
    """Every sliding-layer call of the real model: banded == transformers' SDPA
    path on the same q/k/v/mask (fp32: to fp32 rounding; bf16: within 2 ulp)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gemmaloop.train import sliding_attention as sa

    model, _, tokenizer = e2b
    batch = _text_batch(tokenizer, 2560)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    records = []
    try:
        assert sa.install_banded_sliding_attention(model.get_base_model()) == 28
        banded = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def probe(module, q, k, v, mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
            call = dict(dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
            ref, _ = sa.sdpa_attention_forward(module, q, k, v, mask, **call)
            if getattr(module, sa._FLAG, None) is not None:
                out, _ = banded(module, q, k, v, mask, **call)
                with torch.autocast("cuda", enabled=False):
                    q32, k32, v32 = q.float(), k.float(), v.float()
                    ref32, _ = sa.sdpa_attention_forward(module, q32, k32, v32, mask, **call)
                    out32, _ = banded(module, q32, k32, v32, mask, **call)
                scale = float(ref.float().abs().max())
                records.append(
                    (
                        module.layer_idx,
                        float((out.float() - ref.float()).abs().max()) / scale,
                        float((out32 - ref32).abs().max()) / scale,
                    )
                )
            return ref, None

        ALL_ATTENTION_FUNCTIONS["sdpa"] = probe
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**batch, fused_lm_head=True)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
    assert len({layer for layer, _, _ in records}) == 28
    worst_bf16 = max(r[1] for r in records)
    worst_fp32 = max(r[2] for r in records)
    assert worst_fp32 < 1e-5, records
    assert worst_bf16 <= 2.0**-7, records  # 2 bf16 ulp of the largest output


def test_banded_sliding_attention_keeps_loss_and_gradients(e2b):
    """Model level, at a length where the banded path engages: banded vs the
    full-mask path must differ no more than two equivalent SDPA kernels do
    (math vs memory-efficient on the same sliding layers)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gemmaloop.train import sliding_attention as sa
    from gemmaloop.train.fused_ce import install_lean_lm_head

    model, _, tokenizer = e2b
    install_lean_lm_head(model)
    batch = _text_batch(tokenizer, 2560)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def math_sliding(module, *args, **kwargs):
        if getattr(module, "is_sliding", False):
            with sdpa_kernel([SDPBackend.MATH]):
                return sa.sdpa_attention_forward(module, *args, **kwargs)
        return sa.sdpa_attention_forward(module, *args, **kwargs)

    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = sa.sdpa_attention_forward
        ref_loss, ref_grads = _step(model, batch, fused_lm_head=True)
        again_loss, again_grads = _step(model, batch, fused_lm_head=True)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = math_sliding
        math_loss, math_grads = _step(model, batch, fused_lm_head=True)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        assert sa.install_banded_sliding_attention(model.get_base_model()) == 28
        banded_loss, banded_grads = _step(model, batch, fused_lm_head=True)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
    assert again_loss == ref_loss
    kernel_rel_loss = abs(math_loss - ref_loss) / ref_loss
    kernel_cos, kernel_rel = _distance(ref_grads, math_grads)
    noise_cos, noise_rel = _distance(ref_grads, again_grads)
    cos, rel = _distance(ref_grads, banded_grads)
    report = dict(
        banded=(abs(banded_loss - ref_loss) / ref_loss, cos, rel),
        math_kernel=(kernel_rel_loss, kernel_cos, kernel_rel),
        backward_noise=(noise_cos, noise_rel),
    )
    assert abs(banded_loss - ref_loss) / ref_loss <= max(2.0 * kernel_rel_loss, 2e-4), report
    assert cos >= 1.0 - 2.0 * (1.0 - min(kernel_cos, noise_cos)), report
    assert rel <= 2.0 * max(kernel_rel, noise_rel), report
