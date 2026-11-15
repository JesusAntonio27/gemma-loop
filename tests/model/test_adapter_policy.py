"""LoRA activation policies and configurable target layer sets."""

import copy

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.load import lora_target_modules
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from tests.model.tiny import tiny_text_model

IDS = torch.randint(0, 256, (2, 10), generator=torch.Generator().manual_seed(0))
TARGETS = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _scaled(model, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        layer.layer_scalar.copy_(torch.rand(1, generator=generator) + 0.4)
    return model


def _lora_config():
    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
        target_modules=TARGETS,
    )


def _randomise_nonzero_adapters(model):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator))


def _build(block=(5, 7), policy="loop_only", bridge="gch", seed=0):
    torch.manual_seed(seed)
    base = _scaled(tiny_text_model())
    plan = make_layer_plan(base.config.to_dict(), block=block)
    apply_loops(
        base,
        plan,
        LoopConfig(n_loops=1, bridge=bridge, adapter_policy=policy),
    )
    model = get_peft_model(base, _lora_config())
    _randomise_nonzero_adapters(model)
    return model, base


def test_loop_only_keeps_n1_bit_identical_with_nonzero_lora_and_g():
    baseline = _scaled(tiny_text_model())
    model, base = _build()
    with torch.no_grad():
        base.loop_bridge_g.normal_()
        assert torch.count_nonzero(base.loop_bridge_g) > 0
        set_n_loops(base, 1)
        expected = baseline(input_ids=IDS).last_hidden_state
        got = model(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_always_policy_changes_n1_when_adapters_are_nonzero():
    baseline = _scaled(tiny_text_model())
    model, _ = _build(policy="always")
    with torch.no_grad():
        expected = baseline(input_ids=IDS).last_hidden_state
        got = model(input_ids=IDS).last_hidden_state
    assert not torch.equal(got, expected)


def test_loop_only_activates_adapters_at_n2():
    baseline = _scaled(tiny_text_model())
    model, base = _build()
    with torch.no_grad():
        set_n_loops(base, 2)
        two = model(input_ids=IDS).last_hidden_state
    assert not torch.equal(two, baseline(input_ids=IDS).last_hidden_state)
    # ... and N=1 stays the original even after that.
    with torch.no_grad():
        set_n_loops(base, 1)
        one = model(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(
        one, baseline(input_ids=IDS).last_hidden_state, atol=0, rtol=0
    )


def _grads(with_checkpointing):
    model, base = _build(block=(5, 7), policy="loop_only")
    if with_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        assert all(
            getattr(layer, "gradient_checkpointing", False)
            for layer in base.layers
        )
    model.train()
    set_n_loops(base, 2)
    out = model(input_ids=IDS).last_hidden_state
    out.float().pow(2).sum().backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and ("lora_" in name or name.endswith("loop_bridge_g"))
    }


def test_checkpointing_recomputation_sees_the_right_pass_index():
    plain = _grads(False)
    checkpointed = _grads(True)
    assert set(plain) == set(checkpointed)
    assert plain, "test needs at least one adapter/bridge gradient"
    for name, gradient in plain.items():
        torch.testing.assert_close(
            checkpointed[name], gradient, atol=1e-6, rtol=1e-6
        )


def test_generation_cached_equals_uncached_with_loop_only_adapters():
    from transformers import Gemma4ForCausalLM

    model, base = _build(block=(5, 6), policy="loop_only")
    set_n_loops(base, 2)
    lm = Gemma4ForCausalLM(base.config)
    lm.model = model
    lm.lm_head.weight = base.embed_tokens.weight
    lm.eval()

    with torch.no_grad():
        cached = lm.generate(
            input_ids=IDS,
            do_sample=False,
            max_new_tokens=6,
            use_cache=True,
            pad_token_id=base.config.pad_token_id,
        )
        ids = IDS.clone()
        for _ in range(6):
            logits = lm(input_ids=ids, use_cache=False).logits
            ids = torch.cat(
                [ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
            )
    assert torch.equal(cached, ids)


def test_loop_only_roundtrips_through_peft_save_load(tmp_path):
    model, base = _build(block=(5, 6), policy="loop_only")
    set_n_loops(base, 2)
    with torch.no_grad():
        expected = model(input_ids=IDS).last_hidden_state
    model.save_pretrained(tmp_path)

    fresh = _scaled(tiny_text_model())
    plan = make_layer_plan(fresh.config.to_dict(), block=(5, 6))
    apply_loops(
        fresh,
        plan,
        LoopConfig(n_loops=1, bridge="gch", adapter_policy="loop_only"),
    )
    loaded = PeftModel.from_pretrained(fresh, tmp_path)
    set_n_loops(fresh, 2)
    with torch.no_grad():
        got = loaded(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_lora_target_layer_sets():
    config = tiny_text_model().config.to_dict()
    plan = make_layer_plan(config, block=(5, 6))
    everything = lora_target_modules("language", plan)
    assert everything == lora_target_modules(None, plan)

    import re

    restricted = lora_target_modules("loop_block_coda", plan)
    for layer in list(plan.loop) + list(plan.coda):
        name = f"model.language_model.layers.{layer}.self_attn.q_proj"
        assert re.fullmatch(restricted, name)
    for layer in plan.prelude:
        name = f"model.language_model.layers.{layer}.self_attn.q_proj"
        assert not re.fullmatch(restricted, name)

    loop_only = lora_target_modules("loop_block", plan)
    for layer in plan.loop:
        name = f"model.language_model.layers.{layer}.self_attn.q_proj"
        assert re.fullmatch(loop_only, name)
    for layer in list(plan.prelude) + list(plan.coda):
        name = f"model.language_model.layers.{layer}.self_attn.q_proj"
        assert not re.fullmatch(loop_only, name)

    for name in ("loop_block", "loop_block_coda"):
        with pytest.raises(ValueError):
            lora_target_modules(name, None)


def test_loop_only_heldout_loss_and_calibrated_nll_are_n1_identical():
    from transformers import Gemma4ForCausalLM

    from gemmaloop.eval.calibrated_nll import score_model
    from gemmaloop.train.trainer import heldout_loss

    baseline = _scaled(tiny_text_model())
    looped = copy.deepcopy(baseline)
    plan = make_layer_plan(baseline.config.to_dict(), block=(5, 7))
    apply_loops(
        looped,
        plan,
        LoopConfig(n_loops=1, bridge="gch", adapter_policy="loop_only"),
    )
    model = get_peft_model(looped, _lora_config())
    _randomise_nonzero_adapters(model)
    with torch.no_grad():
        looped.loop_bridge_g.normal_()

    labels = IDS.clone()
    labels[:, :3] = -100
    batch = {
        "input_ids": IDS,
        "attention_mask": torch.ones_like(IDS),
        "labels": labels,
        "num_items_in_batch": int((labels != -100).sum()),
    }

    causal = Gemma4ForCausalLM(baseline.config)
    causal.model = baseline
    causal.lm_head.weight = baseline.embed_tokens.weight
    baseline_loss = heldout_loss(causal, batch, 1)

    wrapped = Gemma4ForCausalLM(looped.config)
    wrapped.model = model
    wrapped.lm_head.weight = looped.embed_tokens.weight
    looped_loss = heldout_loss(wrapped, batch, 1)
    torch.testing.assert_close(looped_loss, baseline_loss, atol=0, rtol=0)

    plain_scores = score_model(baseline, [batch], n_loops=1)
    loop_scores = score_model(model, [batch], n_loops=1)
    torch.testing.assert_close(loop_scores.nll, plain_scores.nll, atol=0, rtol=0)


def test_loop_only_static_cache_generation_matches_uncached():
    from transformers import Gemma4ForCausalLM

    from gemmaloop.serve.cache import build_text_cache

    model, base = _build(block=(5, 6), policy="loop_only")
    set_n_loops(base, 2)
    lm = Gemma4ForCausalLM(base.config)
    lm.model = model
    lm.lm_head.weight = base.embed_tokens.weight
    lm.eval()
    with torch.no_grad():
        cache = build_text_cache(base.config, IDS.shape[1] + 6)
        cached = lm.generate(
            input_ids=IDS,
            do_sample=False,
            max_new_tokens=6,
            past_key_values=cache,
            pad_token_id=base.config.pad_token_id,
        )
        ids = IDS.clone()
        for _ in range(6):
            logits = lm(input_ids=ids, use_cache=False).logits
            ids = torch.cat(
                [ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
            )
    assert torch.equal(cached, ids)


def _raw_target_modules(name: str, plan) -> str:
    """Drop the production wrapper prefix for a bare Gemma4TextModel test."""
    from gemmaloop.model.load import lora_target_modules

    return lora_target_modules(name, plan).replace(
        r"model\.language_model\.", ""
    )


def _expected_lora_parameters(base, plan, rank: int = 4) -> int:
    from torch import nn

    total = 0
    for layer in plan.loop:
        prefix = f"layers.{layer}."
        for name, module in base.named_modules():
            if not name.startswith(prefix) or not isinstance(module, nn.Linear):
                continue
            if name.rsplit(".", 1)[-1] in TARGETS:
                total += rank * module.in_features + module.out_features * rank
    return total


def test_loop_block_target_set_matches_the_block_parameter_count():
    from gemmaloop.model.load import lora_target_modules

    config = tiny_text_model().config.to_dict()
    problems = {
        (5, 5): 0,
        (5, 7): 0,
        (6, 10): 0,
    }
    for block in problems:
        base = _scaled(tiny_text_model())
        expected = _expected_lora_parameters(base, make_layer_plan(config, block=block))
        plan = make_layer_plan(config, block=block)
        model = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
                target_modules=_raw_target_modules("loop_block", plan),
            ),
        )
        count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        )
        assert count == expected
        problems[block] = count
        lora_names = [
            name for name, _ in model.named_parameters() if "lora_" in name
        ]
        for name in lora_names:
            assert any(f"layers.{layer}." in name for layer in plan.loop), name
            assert not any(f"layers.{layer}." in name for layer in plan.coda)

    assert problems[(5, 5)] < problems[(5, 7)] < problems[(6, 10)]


def test_loop_block_and_loop_only_keep_n1_bit_identical():
    from gemmaloop.model.load import lora_target_modules

    baseline = _scaled(tiny_text_model())
    looped = copy.deepcopy(baseline)
    plan = make_layer_plan(baseline.config.to_dict(), block=(5, 7))
    apply_loops(
        looped,
        plan,
        LoopConfig(n_loops=1, bridge="gch", adapter_policy="loop_only"),
    )
    model = get_peft_model(
        looped,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type=None,
            target_modules=_raw_target_modules("loop_block", plan),
        ),
    )
    _randomise_nonzero_adapters(model)
    with torch.no_grad():
        looped.loop_bridge_g.normal_()
        got = model(input_ids=IDS).last_hidden_state
        expected = baseline(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_loop_block_targets_are_active_on_the_unlooped_single_pass():
    from gemmaloop.model.load import lora_target_modules

    baseline = _scaled(tiny_text_model())
    plan = make_layer_plan(baseline.config.to_dict(), block=(5, 7))
    model = get_peft_model(
        baseline,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type=None,
            target_modules=_raw_target_modules("loop_block", plan),
        ),
    )
    _randomise_nonzero_adapters(model)
    with torch.no_grad():
        got = model(input_ids=IDS).last_hidden_state
    # This is the O-H shape: no loop patch, so the single pass runs the
    # loop-block adapters (they must not be inert).
    assert torch.count_nonzero(got) > 0
    plain = _scaled(tiny_text_model())
    with torch.no_grad():
        expected = plain(input_ids=IDS).last_hidden_state
    assert not torch.equal(got, expected)


def _tiny35_text_model():
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    config = Gemma4TextConfig(
        num_hidden_layers=35,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 7,
        num_kv_shared_layers=10,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        hidden_size_per_layer_input=4,
        vocab_size=256,
        vocab_size_per_layer_input=256,
        sliding_window=16,
        use_double_wide_mlp=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Gemma4TextModel(config)
    model.eval()
    return model


def test_oh_loop_block_25_29_trains_the_same_modules_as_l():
    import argparse

    import scripts.train as train_script

    looped_model = _tiny35_text_model()
    config = looped_model.config.to_dict()
    plan = make_layer_plan(config, block=(25, 29))
    args = argparse.Namespace(
        block="25-29",
        bridge="gch",
        bridge_init=0.99,
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
    )
    loop_cfg = train_script._loop_config("L", args)
    assert loop_cfg is not None
    apply_loops(looped_model, plan, loop_cfg)

    oh_model = _tiny35_text_model()
    assert train_script.resolve_lora_target_layers(loop_cfg, args) == "loop_block"
    assert train_script.resolve_lora_target_layers(None, args) == "loop_block"
    assert train_script.resolve_lora_layers(plan, "loop_block") == list(
        range(25, 30)
    )

    def _lora(module):
        target = train_script.resolve_lora_target_layers(
            loop_cfg if module is looped_model else None, args
        )
        return get_peft_model(
            module,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
                target_modules=_raw_target_modules(target, plan),
            ),
        )

    loop_lora = _lora(looped_model)
    oh_lora = _lora(oh_model)

    def _lora_params(model):
        return {
            name: parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }

    loop_params = _lora_params(loop_lora)
    oh_params = _lora_params(oh_lora)
    assert loop_params == oh_params
    assert sum(loop_params.values()) == sum(oh_params.values())
    for name in loop_params:
        layer = int(name.split("layers.")[1].split(".")[0])
        assert 25 <= layer <= 29, name
