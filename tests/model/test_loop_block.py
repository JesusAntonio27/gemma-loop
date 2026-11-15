"""Configurable loop-block geometry (consumer-only, non-empty coda)."""

import copy

import pytest
import torch

from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.flops import flops_per_token
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.serve.cache import build_text_cache
from tests.model.tiny import tiny_config, tiny_text_model

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))
CONSUMER_BLOCKS = [(5, 5), (5, 7), (5, 9), (6, 10), (7, 9), (8, 9), (5, 13)]


def _with_nonunit_scalar(model, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        layer.layer_scalar.copy_(
            torch.rand(1, generator=generator) * 1.5 + 0.25
        )
    return model


@pytest.mark.parametrize("block", CONSUMER_BLOCKS)
def test_make_layer_plan_accepts_consumer_block(block):
    config = tiny_config().to_dict()
    plan = make_layer_plan(config, block=block)
    assert plan.prelude == range(0, block[0])
    assert plan.loop == range(block[0], block[1] + 1)
    assert plan.coda == range(block[1] + 1, 15)
    assert len(plan.coda) > 0


@pytest.mark.parametrize(
    "block", [(4, 9), (0, 4), (5, 14), (15, 15), (10, 9)]
)
def test_make_layer_plan_rejects_invalid_blocks(block):
    with pytest.raises(ValueError):
        make_layer_plan(tiny_config().to_dict(), block=block)


def test_apply_loops_rejects_hand_built_producer_block():
    config = tiny_config()
    model = _with_nonunit_scalar(tiny_text_model())
    plan = make_layer_plan(config.to_dict(), block=(5, 7))
    # Splice a producer layer into the loop while keeping the schedule a
    # contiguous partition: apply_loops must still refuse it.
    from dataclasses import replace

    bad = replace(plan, prelude=range(0, 4), loop=range(4, 8), coda=range(8, 15))
    with pytest.raises(ValueError, match="KV"):
        apply_loops(model, bad, LoopConfig(n_loops=1))


@pytest.mark.parametrize("block", CONSUMER_BLOCKS)
@pytest.mark.parametrize("bridge", ["scalar", "none", "damped", "gch"])
def test_n1_is_bit_identical_for_every_block_and_bridge(block, bridge):
    base = _with_nonunit_scalar(tiny_text_model())
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict(), block=block)
    apply_loops(looped, plan, LoopConfig(n_loops=1, bridge=bridge))
    set_n_loops(looped, 1)
    with torch.no_grad():
        looped.loop_bridge_logit.fill_(1.25)
        if bridge == "gch":
            looped.loop_bridge_g.normal_()
        a = base(input_ids=IDS).last_hidden_state
        b = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_loop_config_block_drives_apply_loops():
    base = _with_nonunit_scalar(tiny_text_model())
    looped = copy.deepcopy(base)
    default_plan = make_layer_plan(base.config.to_dict())
    assert default_plan.loop == range(5, 10)
    apply_loops(looped, default_plan, LoopConfig(n_loops=1, block=(5, 7)))
    assert looped._gemmaloop_plan.loop == range(5, 8)
    assert looped._gemmaloop_plan.coda == range(8, 15)
    set_n_loops(looped, 1)
    with torch.no_grad():
        torch.testing.assert_close(
            base(input_ids=IDS).last_hidden_state,
            looped(input_ids=IDS).last_hidden_state,
            atol=0,
            rtol=0,
        )


def test_flops_follow_the_configured_block():
    config = tiny_config().to_dict()
    layer_flops = per_layer_flops(config)
    constant = layer_flops.constant_flops
    plans = {block: make_layer_plan(config, block=block) for block in CONSUMER_BLOCKS}
    values = {
        block: flops_per_token(layer_flops, plan, 2)
        for block, plan in plans.items()
    }
    for block, plan in plans.items():
        expected = constant
        expected += sum(layer_flops[i] for i in range(0, block[0]))
        expected += 2 * sum(layer_flops[i] for i in range(block[0], block[1] + 1))
        expected += sum(layer_flops[i] for i in range(block[1] + 1, 15))
        assert values[block] == expected
    # A smaller, later block is cheaper per extra pass.
    assert values[(5, 5)] < values[(5, 9)]


def test_static_cache_generation_matches_uncached_for_non_default_block():
    from transformers import Gemma4ForCausalLM

    base = _with_nonunit_scalar(tiny_text_model())
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict(), block=(5, 7))
    apply_loops(looped, plan, LoopConfig(n_loops=2))
    set_n_loops(looped, 2)

    lm = Gemma4ForCausalLM(looped.config)
    lm.model = looped
    lm.lm_head.weight = looped.embed_tokens.weight
    lm.eval()

    with torch.no_grad():
        cache = build_text_cache(looped.config, IDS.shape[1] + 8)
        static = lm.generate(
            input_ids=IDS,
            do_sample=False,
            max_new_tokens=8,
            past_key_values=cache,
            pad_token_id=looped.config.pad_token_id,
        )
        ids = IDS.clone()
        for _ in range(8):
            logits = lm(input_ids=ids, use_cache=False).logits
            ids = torch.cat(
                [ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
            )
    assert torch.equal(static, ids)


def test_cached_generation_matches_uncached_for_non_default_block():
    from transformers import Gemma4ForCausalLM

    base = _with_nonunit_scalar(tiny_text_model())
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict(), block=(5, 7))
    apply_loops(looped, plan, LoopConfig(n_loops=2))
    set_n_loops(looped, 2)

    lm = Gemma4ForCausalLM(looped.config)
    lm.model = looped
    lm.lm_head.weight = looped.embed_tokens.weight
    lm.eval()

    with torch.no_grad():
        cached = lm.generate(
            input_ids=IDS,
            do_sample=False,
            max_new_tokens=8,
            use_cache=True,
            pad_token_id=looped.config.pad_token_id,
        )
        ids = IDS.clone()
        for _ in range(8):
            logits = lm(input_ids=ids, use_cache=False).logits
            ids = torch.cat(
                [ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
            )
    assert torch.equal(cached, ids)


def test_e2b_block_flops_ratios_match_the_review():
    import json
    from pathlib import Path

    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "e2b_text_config.json"
        ).read_text(encoding="utf-8")
    )
    values = per_layer_flops(config)
    expected = {(15, 29): 1.428, (25, 29): 1.143, (16, 18): 1.084}
    for block, ratio in expected.items():
        plan = make_layer_plan(config, block=block)
        one = flops_per_token(values, plan, 1)
        two = flops_per_token(values, plan, 2)
        assert two / one == pytest.approx(ratio, abs=1e-3)
