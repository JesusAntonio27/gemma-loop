"""CPU equivalence tests for the lean fused LM head (Task 13b).

The reference is (a) a plain ``log_softmax`` over the full softcapped logits and
(b) TRL's own ``ChunkedLogProbFunction`` / ``add_fused_lm_head``, which is what
``SFTTrainer`` trains with today.  Everything runs in fp32 on the CPU.
"""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4ForCausalLM
from trl.kernels import ChunkedLogProbFunction
from trl.trainer.utils import add_fused_lm_head

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.train.fused_ce import chunked_softcap_logprobs, install_lean_lm_head
from gemmaloop.train.perf import (
    VisionOffloadCallback,
    disable_input_require_grads,
)
from tests.model.tiny import tiny_config

SOFTCAP = 30.0


def _reference(hidden, weight, targets, softcap):
    logits = hidden.double() @ weight.double().t()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    log_p = torch.log_softmax(logits, dim=-1)
    lp = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    top1 = logits.argmax(-1) == targets
    return lp, entropy, top1


@pytest.mark.parametrize("softcap", [None, SOFTCAP])
@pytest.mark.parametrize(
    "token_chunk,vocab_chunk", [(1, 7), (5, 64), (16, 50), (1000, 1000)]
)
def test_chunked_logprobs_match_full_softmax(softcap, token_chunk, vocab_chunk):
    torch.manual_seed(0)
    n, h, v = 13, 24, 211
    hidden = (torch.randn(n, h) * 3).requires_grad_(True)
    weight = torch.randn(v, h)
    targets = torch.randint(0, v, (n,))

    lp, entropy, top1 = chunked_softcap_logprobs(
        hidden, weight, targets, softcap, token_chunk=token_chunk, vocab_chunk=vocab_chunk
    )
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_lp, ref_entropy, ref_top1 = _reference(ref_hidden, weight, targets, softcap)

    torch.testing.assert_close(lp.double(), ref_lp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy.double(), ref_entropy, rtol=1e-5, atol=1e-5)
    assert torch.equal(top1, ref_top1)

    upstream = torch.randn(n)
    (lp * upstream).sum().backward()
    (ref_lp * upstream.double()).sum().backward()
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-5)


def test_top1_breaks_ties_like_argmax():
    # identical rows in the LM head produce exact logit ties; argmax picks the first
    weight = torch.zeros(10, 4)
    weight[3] = weight[7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden = torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 2)
    targets = torch.tensor([3, 7])
    _, _, top1 = chunked_softcap_logprobs(hidden, weight, targets, None, token_chunk=1, vocab_chunk=4)
    assert top1.tolist() == [True, False]


@pytest.mark.parametrize("softcap", [None, SOFTCAP])
@pytest.mark.parametrize("token_chunk,tol", [(4096, 1e-6), (6, 2e-5)])
def test_chunked_logprobs_match_trl_kernel(softcap, token_chunk, tol):
    # Same tile shapes as TRL (one token chunk): fp32 rounding only.  Smaller
    # token chunks change the GEMM blocking, i.e. ~1e-6 relative noise on logits
    # of magnitude ~20.
    torch.manual_seed(1)
    n, h, v = 17, 32, 300
    hidden = (torch.randn(n, h) * 4).requires_grad_(True)
    weight = torch.randn(v, h)
    targets = torch.randint(0, v, (n,))

    lp, entropy, top1 = chunked_softcap_logprobs(
        hidden, weight, targets, softcap, token_chunk=token_chunk, vocab_chunk=128
    )
    trl_hidden = hidden.detach().clone().requires_grad_(True)
    trl_lp, trl_entropy, _sq, _mean, trl_top1, _score = ChunkedLogProbFunction.apply(
        trl_hidden,
        weight,
        None,
        targets,
        1.0,
        128,
        softcap,
        1.0,
        ("log_probs", "entropy", "is_top1"),
        None,
        None,
    )
    torch.testing.assert_close(lp, trl_lp, rtol=tol, atol=tol)
    torch.testing.assert_close(entropy, trl_entropy, rtol=tol, atol=tol)
    assert torch.equal(top1, trl_top1)

    upstream = torch.randn(n)
    (lp * upstream).sum().backward()
    (trl_lp * upstream).sum().backward()
    torch.testing.assert_close(hidden.grad, trl_hidden.grad, rtol=tol, atol=tol)


def test_high_confidence_target_keeps_loss_and_gradient():
    # p_target rounds to 1 in fp32: the target-excluding sum keeps both finite and exact
    weight = torch.zeros(5, 2)
    weight[2, 0] = 1.0
    hidden = torch.tensor([[60.0, 0.0]], requires_grad=True)
    lp, _, _ = chunked_softcap_logprobs(hidden, weight, torch.tensor([2]), None, vocab_chunk=2)
    expected = -torch.log1p(4 * torch.exp(torch.tensor(-60.0, dtype=torch.float64)))
    torch.testing.assert_close(lp.double(), expected.reshape(1), rtol=1e-6, atol=0)
    lp.sum().backward()
    assert torch.isfinite(hidden.grad).all()


# ---------------------------------------------------------------------------
# Model-level: the lean head inside the PEFT model vs TRL's fused head
# ---------------------------------------------------------------------------


def _tiny_softcap_model(seed: int = 0, loops: bool = False) -> Gemma4ForCausalLM:
    config = tiny_config()
    config.final_logit_softcapping = SOFTCAP
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(config)
    if loops:
        apply_loops(model.model, make_layer_plan(config.to_dict()), LoopConfig(n_loops=1))
    return model


def _peft(model, seed: int = 123):
    torch.manual_seed(seed)  # identical LoRA init for every copy
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj", "down_proj"])
    return get_peft_model(model, lora)


def _batch(seed: int = 0, length: int = 24, prompt: int = 6):
    torch.manual_seed(seed)
    input_ids = torch.randint(3, 256, (1, length))
    labels = input_ids.clone()
    labels[:, :prompt] = -100
    labels[:, length // 2 : length // 2 + 2] = -100  # a tool-response style hole
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _run(model, batch, num_items=None):
    model.zero_grad(set_to_none=True)
    out = model(**batch, fused_lm_head=True, num_items_in_batch=num_items)
    out.loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return out, grads


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("n_loops", [1, 3])
def test_lean_head_matches_trl_fused_head(compact, n_loops):
    base = _tiny_softcap_model(loops=True)
    trl_model = _peft(copy.deepcopy(base))
    lean_model = _peft(copy.deepcopy(base))
    for model in (trl_model, lean_model):
        set_n_loops(model.get_base_model().model, n_loops)
        model.get_base_model().model.loop_bridge_logit.requires_grad_(True)
        model.train()
    add_fused_lm_head(trl_model.get_base_model(), outputs=("log_probs", "entropy", "is_top1"))
    add_fused_lm_head(lean_model.get_base_model(), outputs=("log_probs", "entropy", "is_top1"))
    install_lean_lm_head(lean_model, token_chunk=5, vocab_chunk=96, compact=compact)

    batch = _batch()
    for num_items in (None, torch.tensor(40)):
        trl_out, trl_grads = _run(trl_model, batch, num_items)
        lean_out, lean_grads = _run(lean_model, batch, num_items)
        torch.testing.assert_close(lean_out.loss, trl_out.loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(lean_out.log_probs, trl_out.log_probs, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(lean_out.entropy, trl_out.entropy, rtol=1e-6, atol=1e-6)
        assert torch.equal(lean_out.is_top1, trl_out.is_top1)
        assert torch.equal(lean_out.label_mask, trl_out.label_mask)
        assert trl_grads.keys() == lean_grads.keys() and trl_grads
        for name, grad in trl_grads.items():
            torch.testing.assert_close(lean_grads[name], grad, rtol=1e-5, atol=1e-5, msg=name)


def test_lean_head_matches_plain_transformers_loss():
    model = _peft(_tiny_softcap_model())
    model.train()
    batch = _batch(seed=3)
    model.zero_grad(set_to_none=True)
    plain = model(**batch)  # transformers' own loss over the full logits
    plain.loss.backward()
    plain_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    add_fused_lm_head(model.get_base_model(), outputs=("log_probs", "entropy", "is_top1"))
    install_lean_lm_head(model, token_chunk=4, vocab_chunk=100)
    lean_out, lean_grads = _run(model, batch)
    torch.testing.assert_close(lean_out.loss, plain.loss, rtol=1e-5, atol=1e-6)
    for name, grad in plain_grads.items():
        torch.testing.assert_close(lean_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_lean_head_passes_through_without_fused_flag():
    model = _peft(_tiny_softcap_model())
    batch = _batch(seed=4)
    with torch.no_grad():
        expected = model(**batch).logits
    install_lean_lm_head(model)
    with torch.no_grad():
        got = model(**batch)
    torch.testing.assert_close(got.logits, expected)
    assert got.loss is not None


def test_lean_head_all_masked_batch_is_a_connected_zero():
    model = _peft(_tiny_softcap_model())
    model.train()
    install_lean_lm_head(model)
    batch = _batch(seed=5)
    batch["labels"][:] = -100
    out = model(**batch, fused_lm_head=True)
    assert out.loss.item() == 0.0
    out.loss.backward()  # must not raise


def test_lean_head_refuses_a_trainable_lm_head():
    model = _peft(_tiny_softcap_model())
    model.train()
    install_lean_lm_head(model)
    model.get_base_model().get_output_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen LM-head"):
        model(**_batch(seed=6), fused_lm_head=True)


# ---------------------------------------------------------------------------
# Input-grad hook and vision offload
# ---------------------------------------------------------------------------


def test_disabling_input_grads_keeps_lora_gradients():
    base = _tiny_softcap_model(seed=7, loops=True)
    grads = []
    for hook in (True, False):
        model = _peft(copy.deepcopy(base))
        set_n_loops(model.get_base_model().model, 2)
        model.get_base_model().model.loop_bridge_logit.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if not hook:
            assert disable_input_require_grads(model)
        model.train()
        add_fused_lm_head(model.get_base_model(), outputs=("log_probs",))
        _, g = _run(model, _batch(seed=8))
        grads.append(g)
    with_hook, without_hook = grads
    assert with_hook.keys() == without_hook.keys() and with_hook
    for name, grad in with_hook.items():
        assert torch.equal(without_hook[name], grad), name


class _FakeVLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Linear(4, 4)
        self.model.embed_vision = torch.nn.Linear(4, 4)
        self.model.language_model = torch.nn.Linear(4, 4)
        for p in list(self.model.vision_tower.parameters()) + list(self.model.embed_vision.parameters()):
            p.requires_grad_(False)


def test_vision_offload_guards_and_restores():
    fake = _FakeVLM()
    callback = VisionOffloadCallback(fake)
    assert sorted(callback.modules) == ["embed_vision", "vision_tower"]
    fake.model.vision_tower(torch.zeros(1, 4))  # allowed before training
    callback.offload()
    with pytest.raises(RuntimeError, match="offloaded"):
        fake.model.vision_tower(torch.zeros(1, 4))
    fake.model.language_model(torch.zeros(1, 4))  # text path untouched
    callback.restore()
    fake.model.embed_vision(torch.zeros(1, 4))


def test_vision_offload_refuses_trainable_modules():
    fake = _FakeVLM()
    fake.model.vision_tower.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        VisionOffloadCallback(fake).offload()


# ---------------------------------------------------------------------------
# Trainer level: the switches survive Trainer.train() and keep the loss curve
# ---------------------------------------------------------------------------


def _perf_trainer(tmp_path, name, perf_config, lora_dropout=0.0):
    from datasets import Dataset
    from trl import SFTConfig

    from gemmaloop.train.perf import apply_train_perf
    from gemmaloop.train.trainer import LoopSFTTrainer
    from tests.train.test_trainer_tiny import _tiny_tokenizer

    model = _tiny_softcap_model(seed=11, loops=True)
    rows = []
    for i in range(8):
        torch.manual_seed(100 + i)
        ids = torch.randint(3, 256, (20 + i,)).tolist()
        labels = [-100] * 5 + ids[5:]
        rows.append({"input_ids": ids, "labels": labels})
    args = SFTConfig(
        output_dir=str(tmp_path / name),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=4,
        learning_rate=1e-2,
        lr_scheduler_type="linear",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=64,
        train_sampling_strategy="sequential",
        seed=0,
        data_seed=0,
        disable_tqdm=True,
    )
    torch.manual_seed(123)
    trainer = LoopSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=_tiny_tokenizer(),
        peft_config=LoraConfig(
            r=4, lora_alpha=8, lora_dropout=lora_dropout, target_modules=["q_proj", "v_proj"]
        ),
        loop_text_model=model.model,
        loop_enabled=True,
        loop_seed=0,
    )
    apply_train_perf(trainer, perf_config)
    seen = []
    embed = trainer.model.get_base_model().model.embed_tokens
    embed.register_forward_hook(lambda module, inputs, output: seen.append(output.requires_grad))
    return trainer, seen


def _losses(trainer):
    return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]


def _trainable(trainer):
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}


def test_perf_switches_are_bit_identical_at_trl_tile_sizes(tmp_path):
    """Same tiles as TRL + no input-grad hook: identical weights after training."""
    from gemmaloop.train.perf import BASELINE_PERF, TrainPerfConfig

    baseline, base_seen = _perf_trainer(tmp_path, "base", BASELINE_PERF)
    baseline.train()
    lean, lean_seen = _perf_trainer(
        tmp_path,
        "lean",
        TrainPerfConfig(lm_head_token_chunk=4096, lm_head_vocab_chunk=32768, lm_head_compact=False),
    )
    lean.train()

    # Trainer.train() re-installs the hook; the callback removes it again
    assert base_seen and all(base_seen)
    assert lean_seen and not any(lean_seen)
    assert lean._gemmaloop_perf_callback.removed_input_grad_hook >= 2  # TRL init + train()

    assert _losses(lean) == _losses(baseline) and len(_losses(lean)) == 4
    expected, got = _trainable(baseline), _trainable(lean)
    assert expected.keys() == got.keys()
    for name, value in expected.items():
        assert torch.equal(got[name], value), name


def test_small_tiles_keep_the_loss_curve(tmp_path):
    """Smaller tiles only reorder fp32 sums: same loss curve to ~1e-6."""
    from gemmaloop.train.perf import BASELINE_PERF, TrainPerfConfig

    baseline, _ = _perf_trainer(tmp_path, "base", BASELINE_PERF)
    baseline.train()
    lean, _ = _perf_trainer(
        tmp_path, "lean", TrainPerfConfig(lm_head_token_chunk=3, lm_head_vocab_chunk=50)
    )
    lean.train()
    assert len(_losses(lean)) == 4
    for got, expected in zip(_losses(lean), _losses(baseline)):
        assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)
    expected, got = _trainable(baseline), _trainable(lean)
    for name, value in expected.items():
        # Adam (lr 1e-2) amplifies 1e-7 gradient reorderings on near-zero entries
        torch.testing.assert_close(got[name], value, rtol=0, atol=2e-3, msg=name)


def test_adaptive_checkpointing_keeps_the_training_run(tmp_path):
    """Skipping checkpointing for small micro steps changes nothing: the same
    dropout masks (non-reentrant checkpointing replays and restores the RNG)
    and the same loss curve and weights."""
    from gemmaloop.train.perf import TrainPerfConfig, decoder_layer_calls

    always, _ = _perf_trainer(tmp_path, "always", TrainPerfConfig(), lora_dropout=0.2)
    always.train()
    # rows are 20-27 tokens; tiny plan: 15 + 5 (N - 1) layer calls
    adaptive, _ = _perf_trainer(
        tmp_path, "adaptive", TrainPerfConfig(checkpoint_skip_token_calls=24 * 25), lora_dropout=0.2
    )
    assert decoder_layer_calls(adaptive.loop_text_model) == 15
    adaptive.train()
    policy = adaptive.checkpoint_policy
    assert policy.skipped > 0 and policy.checkpointed > 0
    assert len(_losses(adaptive)) == 4
    assert _losses(adaptive) == pytest.approx(_losses(always), rel=1e-6, abs=1e-7)
    expected, got = _trainable(always), _trainable(adaptive)
    for name, value in expected.items():
        torch.testing.assert_close(got[name], value, rtol=1e-5, atol=1e-6, msg=name)


def test_adaptive_checkpointing_never_enables_checkpointing():
    from gemmaloop.train.perf import AdaptiveCheckpointing

    model = _tiny_softcap_model(loops=True)
    policy = AdaptiveCheckpointing(model.model, max_token_calls=100)  # 15 layer calls
    inputs = {"input_ids": torch.zeros(1, 50, dtype=torch.long)}
    assert policy(inputs, enabled=False) is False
    assert not any(layer.gradient_checkpointing for layer in model.model.layers)
    assert policy(inputs, enabled=True) is True
    assert policy({"input_ids": torch.zeros(1, 2, dtype=torch.long)}, enabled=True) is False
    assert (policy.skipped, policy.checkpointed) == (1, 1)


def test_reentrant_checkpointing_would_drop_shared_kv_gradients(tmp_path):
    """Why use_reentrant=False is kept: Gemma 4's KV-shared layers get the
    producer layers' K/V through the ``shared_kv_states`` kwarg, and reentrant
    checkpointing propagates gradients only through positional inputs."""
    from gemmaloop.train.perf import BASELINE_PERF

    baseline, _ = _perf_trainer(tmp_path, "base", BASELINE_PERF)
    baseline.train()
    reentrant, _ = _perf_trainer(tmp_path, "reentrant", BASELINE_PERF)
    reentrant.args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    reentrant.train()
    assert _losses(reentrant)[0] == pytest.approx(_losses(baseline)[0], rel=1e-6)
    assert _losses(reentrant)[1:] != pytest.approx(_losses(baseline)[1:], rel=1e-4)


def test_cap_cuda_allocator_leaves_desktop_memory_alone(monkeypatch):
    from gemmaloop.train import perf

    calls = []
    gib = 2**30
    monkeypatch.setattr(perf.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf.torch.cuda, "set_per_process_memory_fraction", calls.append)
    monkeypatch.setattr(perf.torch.cuda, "mem_get_info", lambda: (int(10.78 * gib), int(11.91 * gib)))
    info = perf.cap_cuda_allocator(0.92, 0.5)
    assert info["cap_gib"] == pytest.approx(10.28, abs=1e-3)
    assert calls == [pytest.approx(10.28 / 11.91, rel=1e-4)]
    monkeypatch.setattr(perf.torch.cuda, "mem_get_info", lambda: (int(11.8 * gib), int(11.91 * gib)))
    assert perf.cap_cuda_allocator(0.92, 0.5)["cap_gib"] == pytest.approx(0.92 * 11.91, abs=1e-3)
    monkeypatch.setattr(perf.torch.cuda, "mem_get_info", lambda: (int(0.4 * gib), int(11.91 * gib)))
    with pytest.raises(RuntimeError, match="another job"):
        perf.cap_cuda_allocator(0.92, 0.5)
