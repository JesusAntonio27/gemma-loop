import copy, json, pytest, torch
from gemmaloop.model.layer_plan import LayerPlan, make_layer_plan
from gemmaloop.model.looped import (LoopConfig, apply_loops, set_n_loops, bridge_input,
                                    damped_update, save_loop_state, load_loop_state,
                                    add_loop_ple)
from tests.model.tiny import tiny_text_model

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))

def _pair():
    base = tiny_text_model(); looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict())
    apply_loops(looped, plan, LoopConfig(n_loops=1)); return base, looped

def test_n1_identical_hidden_states():
    base, looped = _pair()
    with torch.no_grad():
        a = base(input_ids=IDS).last_hidden_state; b = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)

def test_n2_changes_output():
    base, looped = _pair(); set_n_loops(looped, 2)
    with torch.no_grad():
        assert not torch.equal(base(input_ids=IDS).last_hidden_state, looped(input_ids=IDS).last_hidden_state)

def test_param_count_constant_across_n():
    _, looped = _pair(); n0 = sum(p.numel() for p in looped.parameters())
    for n in (2, 3, 4):
        set_n_loops(looped, n); assert sum(p.numel() for p in looped.parameters()) == n0

def test_bridge_gets_gradient():
    _, looped = _pair(); set_n_loops(looped, 2)
    looped(input_ids=IDS).last_hidden_state.sum().backward()
    assert looped.loop_bridge_logit.grad is not None

def test_bridge_formulas():
    hp, hprev, w = torch.ones(3), torch.zeros(3), torch.tensor(0.99)
    assert torch.equal(bridge_input(hp, hprev, 1, "scalar", w), hp)
    torch.testing.assert_close(bridge_input(hp, hprev, 2, "scalar", w), torch.full((3,), 0.99))
    assert torch.equal(bridge_input(hp, hprev, 2, "none", w), hprev)
    torch.testing.assert_close(damped_update(torch.zeros(3), torch.ones(3), 4), torch.full((3,), 0.25))
    torch.testing.assert_close(damped_update(torch.zeros(3), torch.ones(3), 1), torch.ones(3))

def test_loop_state_roundtrip(tmp_path):
    _, looped = _pair(); set_n_loops(looped, 2)
    with torch.no_grad(): looped.loop_bridge_logit.fill_(0.3)
    save_loop_state(looped, tmp_path)
    _, fresh = _pair(); load_loop_state(fresh, tmp_path); set_n_loops(fresh, 2)
    with torch.no_grad():
        torch.testing.assert_close(looped(input_ids=IDS).last_hidden_state,
                                   fresh(input_ids=IDS).last_hidden_state)

def test_loop_state_load_ignores_saved_training_n(tmp_path):
    _, looped = _pair(); set_n_loops(looped, 4)
    save_loop_state(looped, tmp_path)
    payload = json.loads((tmp_path / "loop_config.json").read_text())
    assert payload["n_loops_last_sampled"] == 4
    assert "n_loops" not in payload

    _, fresh = _pair(); set_n_loops(fresh, 2)
    load_loop_state(fresh, tmp_path)
    assert fresh._gemmaloop_n_loops == 2
    assert fresh._gemmaloop_loop_config.bridge == "scalar"

def test_scalar_bridge_uses_fp32_arithmetic():
    dtype = torch.bfloat16
    h_prelude = torch.tensor([0.9, 1.7], dtype=dtype)
    h_prev = torch.zeros(2, dtype=dtype)
    w = torch.tensor(0.99, dtype=torch.float32)
    got = bridge_input(h_prelude, h_prev, 2, "scalar", w)
    expected = torch.lerp(h_prev.float(), h_prelude.float(), w).to(dtype)
    assert got.dtype == dtype
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    # Rounding w (and the product) to bf16 first gives a different answer.
    old = (w * h_prelude + (1 - w) * h_prev).to(dtype)
    assert not torch.equal(got, old)

def test_n1_bf16_identical_with_nonzero_loop_state():
    base = tiny_text_model().to(torch.bfloat16)
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict())
    apply_loops(looped, plan, LoopConfig(n_loops=1, bridge="scalar",
                                         ple_mode="loop_vector"))
    set_n_loops(looped, 1)
    with torch.no_grad():
        looped.loop_bridge_logit.fill_(1.25)
        looped.loop_ple.fill_(0.7)
        a = base(input_ids=IDS).last_hidden_state
        b = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)

def test_loop_ple_addition_helper_rounds_fp32_once():
    x = torch.tensor([2.265625], dtype=torch.bfloat16)
    row = torch.tensor([-1.573498249053955], dtype=torch.float32)
    got = add_loop_ple(x, row)
    expected = (x.float() + row).to(torch.bfloat16)
    assert torch.equal(got, expected)
    old = (x + row.to(torch.bfloat16)).to(torch.bfloat16)
    assert not torch.equal(got, old)


def test_loop_ple_addition_is_fp32():
    base = tiny_text_model()
    looped = copy.deepcopy(base).to(torch.bfloat16)
    plan = make_layer_plan(base.config.to_dict())
    apply_loops(looped, plan, LoopConfig(n_loops=1, ple_mode="loop_vector"))
    set_n_loops(looped, 2)
    with torch.no_grad():
        looped.loop_ple[1].fill_(1.0)
        a = looped(input_ids=IDS).last_hidden_state
        # 1.003 and 1.0 are the same bfloat16; a bf16 addition would leave the
        # forward unchanged, an fp32 addition must not.
        looped.loop_ple[1].fill_(1.003)
        b = looped(input_ids=IDS).last_hidden_state
    assert not torch.equal(a, b)

def test_cached_generation_matches_uncached():
    from transformers import Gemma4ForCausalLM

    _, looped = _pair()
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
            ids = torch.cat([ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)

    assert torch.equal(cached, ids)

def test_rejects_loop_over_kv_producers():
    # The tiny model's KV-sharing boundary is layer 5 (15 - 10). These hand-built
    # plans keep the partition valid (they still cover 0..14) but either start the
    # loop in the producer region or carry a wrong boundary; repeating producer
    # layers would update past_key_values twice per decoding step.
    plans = [
        LayerPlan(
            num_layers=15,
            period=5,
            full_layers=(4, 9, 14),
            first_shared=5,
            prelude=range(0, 0),
            loop=range(0, 5),
            coda=range(5, 15),
        ),
        LayerPlan(
            num_layers=15,
            period=5,
            full_layers=(4, 9, 14),
            first_shared=0,
            prelude=range(0, 5),
            loop=range(5, 10),
            coda=range(10, 15),
        ),
    ]
    for plan in plans:
        with pytest.raises(ValueError, match="KV"):
            apply_loops(tiny_text_model(), plan, LoopConfig())


@pytest.mark.parametrize("n", [1, 3])
def test_gradient_checkpointing_wraps_every_loop_pass(n):
    """Every decoder-layer call, including each pass of the looped block, is
    checkpointed (activation memory grows by checkpoint inputs only, not by
    whole layers, as N grows)."""
    _, looped = _pair(); set_n_loops(looped, n)
    looped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    looped.train()
    calls = []
    for idx, layer in enumerate(looped.layers):
        inner = layer._gradient_checkpointing_func
        def counted(fn, *args, _i=idx, _inner=inner, **kwargs):
            calls.append(_i); return _inner(fn, *args, **kwargs)
        layer._gradient_checkpointing_func = counted
    looped(input_ids=IDS).last_hidden_state.sum().backward()
    plan = looped._gemmaloop_plan
    assert calls == list(plan.prelude) + list(plan.loop) * n + list(plan.coda)
    if n > 1:  # the bridge only mixes inputs from the second pass on
        assert looped.loop_bridge_logit.grad is not None
