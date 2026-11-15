"""Multi-exit ("anytime") forward of the looped block (Task T-dial-train).

CPU-only, tiny Gemma 4 config.  The tests pin the exactness contract of
``dial-design.md`` section 8: exit ``t`` of the multi-exit forward is the
fixed-``N=t`` forward, exit 1 is O bit-exactly, the truncated window keeps the
full-unroll gradient restricted to its last ``k`` passes, activation-memory
savings are constant in ``T``, and consumer-layer states are per-token
independent.
"""

from __future__ import annotations

import copy

import pytest
import torch

from peft import LoraConfig, get_peft_model
from transformers import Gemma4ForCausalLM

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import (
    LoopConfig,
    apply_loops,
    bridge_input,
    set_n_loops,
)
from tests.model.tiny import tiny_config, tiny_text_model

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))


def _looped(dtype: torch.dtype = torch.float32, bridge: str = "gch"):
    base = tiny_text_model().to(dtype)
    looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict())
    apply_loops(
        looped,
        plan,
        LoopConfig(n_loops=1, bridge=bridge, bridge_init=0.99, ple_mode="shared"),
    )
    if bridge == "gch":
        with torch.no_grad():
            looped.loop_bridge_g.copy_(torch.linspace(-0.2, 0.3, 64, dtype=torch.float32))
    return base, looped, plan


def _set_horizon(model, n: int) -> None:
    set_n_loops(model, n)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("exit_pass", [1, 2, 3, 4, 8])
def test_exit_equals_fixed_n_forward(dtype, exit_pass):
    """Exit t of a T-run is bit-identical to the fixed-N=t forward."""
    _, looped, _ = _looped(dtype)
    with torch.no_grad():
        _set_horizon(looped, exit_pass)
        expected = looped(input_ids=IDS).last_hidden_state
        _set_horizon(looped, 8)
        outputs = looped(input_ids=IDS, exits=[exit_pass], window=3)
    torch.testing.assert_close(
        outputs.exit_hidden_states[exit_pass], expected, atol=0, rtol=0
    )


def test_exit_one_is_o_bit_exactly_with_nonzero_loop_state():
    base, looped, _ = _looped(torch.float32)
    with torch.no_grad():
        looped.loop_bridge_g.fill_(0.37)
        reference = base(input_ids=IDS).last_hidden_state
        _set_horizon(looped, 4)
        outputs = looped(input_ids=IDS, exits=[1, 2, 3, 4], window=3)
    torch.testing.assert_close(outputs.exit_hidden_states[1], reference, atol=0, rtol=0)


def _record_reference_gradient(looped, plan, total_passes: int, window: int):
    """Full-unroll gradient with the block state detached at the window start.

    The prelude and passes before the window still run, but their outputs are
    replaced by detached copies at the first window pass, exactly the truncated
    BPTT graph.  This is an independent autograd reference (it does not use
    ``exits=``/``window=`` at all).
    """
    start, _ = max(2, total_passes - window + 1), total_passes
    first_loop_layer = looped.layers[plan.loop.start]
    last_loop_layer = looped.layers[plan.loop.stop - 1]
    last_prelude_layer = looped.layers[plan.prelude.stop - 1]
    first_loop_calls = 0
    loop_calls = 0
    state = {"prelude": None, "previous": None}

    def record_prelude(_module, _args, output):
        state["prelude"] = output.detach()

    def record_previous(_module, _args, output):
        nonlocal loop_calls
        loop_calls += 1
        if loop_calls == start - 1:
            state["previous"] = output.detach()

    def detach_window_input(_module, args, kwargs):
        nonlocal first_loop_calls
        first_loop_calls += 1
        if first_loop_calls == start:
            w = torch.sigmoid(looped.loop_bridge_logit)
            g = getattr(looped, "loop_bridge_g", None)
            new_x = bridge_input(
                state["prelude"], state["previous"], start, "gch", w, g
            )
            return (new_x, *args[1:]), kwargs
        return args, kwargs

    handles = [
        last_prelude_layer.register_forward_hook(record_prelude),
        last_loop_layer.register_forward_hook(record_previous),
        first_loop_layer.register_forward_pre_hook(
            detach_window_input, with_kwargs=True
        ),
    ]
    try:
        _set_horizon(looped, total_passes)
        hidden = looped(input_ids=IDS).last_hidden_state
        hidden.sum().backward()
    finally:
        for handle in handles:
            handle.remove()


def test_window_gradient_matches_full_unroll_restricted_to_window():
    _, looped, plan = _looped(torch.float32)
    total_passes, window = 8, 3
    start, end = total_passes - window + 1, total_passes

    # Reference: full forward with the window input detached at the boundary.
    _record_reference_gradient(looped, plan, total_passes, window)
    reference_grad = looped.loop_bridge_g.grad.detach().clone()
    looped.zero_grad(set_to_none=True)

    _set_horizon(looped, total_passes)
    exits = [1, *range(start, end + 1)]
    outputs = looped(input_ids=IDS, exits=exits, window=window)
    outputs.exit_hidden_states[end].sum().backward()
    torch.testing.assert_close(
        looped.loop_bridge_g.grad, reference_grad, atol=1e-6, rtol=1e-6
    )


def _saved_tensor_count(looped, exits) -> int:
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        _set_horizon(looped, max(exits))
        looped(input_ids=IDS, exits=exits, window=3)
    return len(saved)


@pytest.mark.parametrize("total_passes", [4, 5, 6, 7, 8])
def test_saved_tensors_constant_in_horizon(total_passes):
    """The truncated window keeps the CPU activation-memory proxy flat in T."""
    _, looped, _ = _looped(torch.float32)
    start = max(2, total_passes - 3 + 1)
    exits = [1, *range(start, total_passes + 1)]
    count = _saved_tensor_count(looped, exits)
    # Same number of window passes and exits for every T: no early-pass saves.
    _, reference, _ = _looped(torch.float32)
    expected = _saved_tensor_count(reference, [1, 2, 3, 4])
    assert count == expected


def test_consumer_state_is_per_token_independent():
    """Perturbing token j's consumer state never changes token i != j."""
    _, base, plan = _looped(torch.float32)
    _, perturbed, _ = _looped(torch.float32)
    token = 5
    first_loop_layer = base.layers[plan.loop.start]
    calls = 0

    def perturb(_module, args, kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            x = args[0].clone()
            x[:, token, :] = x[:, token, :] + 0.5
            return (x, *args[1:]), kwargs
        return args, kwargs

    handle = first_loop_layer.register_forward_pre_hook(perturb, with_kwargs=True)
    try:
        with torch.no_grad():
            _set_horizon(base, 4)
            _set_horizon(perturbed, 4)
            clean = base(input_ids=IDS, exits=[1, 2, 3, 4], window=3).exit_hidden_states[4]
            noisy = perturbed(
                input_ids=IDS, exits=[1, 2, 3, 4], window=3
            ).exit_hidden_states[4]
    finally:
        handle.remove()

    assert not torch.equal(clean[:, token], noisy[:, token])
    torch.testing.assert_close(clean[:, :token], noisy[:, :token], atol=0, rtol=0)
    torch.testing.assert_close(clean[:, token + 1 :], noisy[:, token + 1 :], atol=0, rtol=0)


def test_multi_exit_rejects_damped_bridge_and_bad_exits():
    _, looped, _ = _looped(torch.float32, bridge="damped")
    _set_horizon(looped, 4)
    with pytest.raises(ValueError, match="time-invariant"):
        looped(input_ids=IDS, exits=[1, 2, 3, 4], window=3)

    _, looped, _ = _looped(torch.float32)
    with pytest.raises(ValueError, match="exits"):
        looped(input_ids=IDS, exits=[5], window=3)
    with pytest.raises(ValueError, match="exits"):
        looped(input_ids=IDS, exits=[], window=3)


def _peft_looped_coda():
    """A PEFT-wrapped looped model whose coda adapters are active above N=1."""
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(tiny_config())
    apply_loops(
        model.model,
        make_layer_plan(model.config.to_dict()),
        LoopConfig(
            n_loops=1,
            bridge="gch",
            ple_mode="shared",
            adapter_policy="loop_only",
            lora_target_layers="loop_block_coda",
        ),
    )
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    text_model = wrapped.base_model.model.model
    with torch.no_grad():
        text_model.loop_bridge_g.fill_(0.2)
        # get_peft_model zero-initialises lora_B; make the adapters non-inert so
        # an active coda adapter really changes the output.
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.5)
    return text_model


def test_coda_uses_the_exits_pass_count_for_adapters():
    text_model = _peft_looped_coda()
    with torch.no_grad():
        _set_horizon(text_model, 1)
        o = text_model(input_ids=IDS).last_hidden_state
        references = {}
        for exit_pass in (2, 3, 4):
            _set_horizon(text_model, exit_pass)
            references[exit_pass] = text_model(input_ids=IDS).last_hidden_state
        _set_horizon(text_model, 4)
        outputs = text_model(input_ids=IDS, exits=[1, 2, 3, 4], window=3)
    torch.testing.assert_close(outputs.exit_hidden_states[1], o, atol=0, rtol=0)
    for exit_pass in (2, 3, 4):
        torch.testing.assert_close(
            outputs.exit_hidden_states[exit_pass],
            references[exit_pass],
            atol=0,
            rtol=0,
        )


def test_window_gradient_matches_reference_for_a_lora_parameter():
    text_model = _peft_looped_coda()
    plan = make_layer_plan(text_model.config.to_dict())
    parameter = dict(text_model.named_parameters())[
        f"layers.{plan.loop.start}.self_attn.q_proj.lora_A.default.weight"
    ]
    total_passes, window = 8, 3
    start, end = total_passes - window + 1, total_passes

    text_model.zero_grad(set_to_none=True)
    _record_reference_gradient(text_model, plan, total_passes, window)
    reference = parameter.grad.detach().clone()
    text_model.zero_grad(set_to_none=True)

    _set_horizon(text_model, total_passes)
    outputs = text_model(
        input_ids=IDS, exits=[1, *range(start, end + 1)], window=window
    )
    outputs.exit_hidden_states[end].sum().backward()
    torch.testing.assert_close(parameter.grad, reference, atol=1e-6, rtol=1e-6)


def test_full_window_objective_gradient_matches_finite_differences():
    from gemmaloop.train.dial_objective import dial_objective

    text_model = _peft_looped_coda()
    plan = make_layer_plan(text_model.config.to_dict())
    parameter = dict(text_model.named_parameters())[
        f"layers.{plan.loop.start}.self_attn.q_proj.lora_A.default.weight"
    ]
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn(256, 64, generator=generator)
    labels = IDS.clone()

    def loss():
        _set_horizon(text_model, 4)
        outputs = text_model(input_ids=IDS, exits=[1, 2, 3, 4], window=3)
        return dial_objective(
            outputs.exit_hidden_states,
            weight,
            labels,
            window=3,
            token_chunk=4,
            vocab_chunk=64,
            # Keep the objective smooth for the numerical check; the hinge is
            # covered by its own unit test.
            lambda_hinge=0.0,
        ).total

    value = loss()
    gradient = torch.autograd.grad(value, parameter)[0]
    direction = torch.randn(parameter.shape, generator=generator)
    epsilon = 1e-3
    with torch.no_grad():
        original = parameter.detach().clone()
        parameter.add_(epsilon * direction)
        plus = float(loss().detach())
        parameter.copy_(original - epsilon * direction)
        minus = float(loss().detach())
        parameter.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = float((gradient * direction).sum())
    assert abs(finite_difference - analytic) <= 1e-2 * max(1.0, abs(analytic))


def test_prelude_runs_without_autograd_in_dial_mode():
    text_model = _peft_looped_coda()
    parameter = text_model.layers[0].self_attn.q_proj.base_layer.weight
    parameter.requires_grad_(True)

    text_model.zero_grad(set_to_none=True)
    _set_horizon(text_model, 4)
    outputs = text_model(input_ids=IDS, exits=[1, 2, 3, 4], window=3)
    outputs.exit_hidden_states[4].sum().backward()
    assert parameter.grad is None

    text_model.zero_grad(set_to_none=True)
    _set_horizon(text_model, 1)
    text_model(input_ids=IDS).last_hidden_state.sum().backward()
    assert parameter.grad is not None
