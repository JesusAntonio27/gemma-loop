"""Zero-init norm-matched per-channel bridge (``gch``)."""

import copy

import torch

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import (
    LoopConfig,
    apply_loops,
    bridge_input,
    gch_bridge_input,
    load_loop_state,
    save_loop_state,
    set_n_loops,
)
from tests.model.tiny import tiny_text_model

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))


def _looped(block=(5, 7), bridge="gch", seed=0):
    torch.manual_seed(seed)
    model = tiny_text_model()
    for layer in model.layers:
        layer.layer_scalar.fill_(0.7)
    plan = make_layer_plan(model.config.to_dict(), block=block)
    apply_loops(model, plan, LoopConfig(n_loops=1, bridge=bridge))
    return model, plan


def test_gch_formula_is_norm_matched_and_single_cast():
    torch.manual_seed(0)
    hp = torch.randn(3, 16)
    prev = torch.randn(3, 16)
    g = torch.randn(16)
    got = gch_bridge_input(hp, prev, g)
    hp32, prev32 = hp.float(), prev.float()
    hp_rms = hp32.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    prev_rms = prev32.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    expected = hp32 + g.float() * hp_rms * (prev32 / prev_rms)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_gch_zero_init_makes_n2_equal_n1():
    looped, _ = _looped()
    set_n_loops(looped, 1)
    with torch.no_grad():
        one = looped(input_ids=IDS).last_hidden_state
    assert torch.count_nonzero(looped.loop_bridge_g) == 0
    set_n_loops(looped, 2)
    with torch.no_grad():
        two = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(one, two, atol=0, rtol=0)


def test_gch_gradient_nonzero_at_init():
    looped, _ = _looped()
    set_n_loops(looped, 2)
    looped(input_ids=IDS).last_hidden_state.pow(2).sum().backward()
    assert looped.loop_bridge_g.grad is not None
    assert torch.count_nonzero(looped.loop_bridge_g.grad) > 0


def test_gch_n1_identity_with_nonzero_g():
    base = tiny_text_model()
    for layer in base.layers:
        layer.layer_scalar.fill_(0.7)
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict(), block=(6, 10))
    apply_loops(looped, plan, LoopConfig(n_loops=1, bridge="gch"))
    with torch.no_grad():
        looped.loop_bridge_g.normal_()
        assert torch.count_nonzero(looped.loop_bridge_g) > 0
        a = base(input_ids=IDS).last_hidden_state
        b = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_gch_and_block_roundtrip_in_loop_state(tmp_path):
    looped, plan = _looped(block=(6, 10))
    set_n_loops(looped, 2)
    with torch.no_grad():
        looped.loop_bridge_g.normal_()
    save_loop_state(looped, tmp_path)
    fresh, fresh_plan = _looped(block=(6, 10))
    set_n_loops(fresh, 2)
    load_loop_state(fresh, tmp_path)
    assert fresh._gemmaloop_loop_config.bridge == "gch"
    assert fresh._gemmaloop_loop_config.block == (6, 10)
    torch.testing.assert_close(
        looped.loop_bridge_g, fresh.loop_bridge_g, atol=0, rtol=0
    )
    with torch.no_grad():
        torch.testing.assert_close(
            looped(input_ids=IDS).last_hidden_state,
            fresh(input_ids=IDS).last_hidden_state,
        )


def test_gch_bridge_formula_matches_bridge_input():
    hp = torch.randn(2, 16)
    prev = torch.randn(2, 16)
    g = torch.randn(16)
    w = torch.tensor(0.5)
    torch.testing.assert_close(
        bridge_input(hp, prev, 2, "gch", w, g),
        gch_bridge_input(hp, prev, g),
        atol=0,
        rtol=0,
    )
    # Pass 1 is always the prelude, even with a non-zero g.
    torch.testing.assert_close(
        bridge_input(hp, prev, 1, "gch", w, g), hp, atol=0, rtol=0
    )


def test_gch_bridge_never_touches_the_coda():
    looped, plan = _looped(block=(5, 7))
    set_n_loops(looped, 2)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def hook(index):
        def _capture(module, inputs, output):
            captured[index] = (inputs[0].detach(), output.detach())

        return _capture

    handles = [
        looped.layers[index].register_forward_hook(hook(index))
        for index in (*plan.loop, *plan.coda)
    ]
    try:
        with torch.no_grad():
            looped.loop_bridge_g.normal_()
            looped(input_ids=IDS)
    finally:
        for handle in handles:
            handle.remove()

    loop_output = captured[plan.loop[-1]][1]
    coda_input = captured[plan.coda[0]][0]
    torch.testing.assert_close(coda_input, loop_output, atol=0, rtol=0)
