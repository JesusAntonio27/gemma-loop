"""CPU-only coverage for bitsandbytes int8 device state repair."""

import pytest
import torch
from bitsandbytes.nn import Int8Params, Linear8bitLt

from gemmaloop.teacher.runtime import teacher_runtime
from scripts.precompute_teacher_topk import (
    _assert_int8_on_device,
    _repair_int8_device_state,
)


def _cpu_int8_module():
    return Linear8bitLt(8, 4, has_fp16_weights=False).to("cpu")


def test_repair_stale_state_after_model_move():
    module = _cpu_int8_module()
    assert module.weight.dtype == torch.int8
    cb, scb = module.weight.CB, module.weight.SCB
    # Mimic Module._apply retaining old auxiliary tensors while weight data moves.
    module.weight = Int8Params(
        torch.empty_like(module.weight.data, device="meta"),
        requires_grad=False,
        has_fp16_weights=False,
        CB=cb,
        SCB=scb,
    )
    module.state.CB = cb.clone()
    module.state.SCB = scb.clone()
    model = torch.nn.Sequential(module)

    with pytest.raises(RuntimeError, match=r"0.weight.CB.*expected meta, got cpu"):
        _assert_int8_on_device(model, "meta")
    _repair_int8_device_state(model, "meta")
    _assert_int8_on_device(model, "meta")
    assert torch._C._is_alias_of(module.weight.CB, module.weight.data)
    assert module.weight.SCB.device.type == "meta"
    assert module.state.CB.device.type == "meta"
    assert module.state.SCB.device.type == "meta"


@pytest.mark.parametrize(
    "field", ["weight.data", "weight.CB", "weight.SCB", "state.CB", "state.SCB"]
)
def test_assert_checks_each_tensor(field):
    module = _cpu_int8_module()
    if field == "weight.data":
        module.weight = Int8Params(
            torch.empty_like(module.weight.data, device="meta"),
            requires_grad=False,
            has_fp16_weights=False,
        )
    else:
        owner, name = field.split(".")
        setattr(getattr(module, owner), name, torch.empty(1, device="meta"))
    pattern = field.replace(".", r"\.") + ".*expected cpu, got meta"
    with pytest.raises(RuntimeError, match=pattern):
        _assert_int8_on_device(module, "cpu")


def test_cpu_repair_and_forward():
    module = _cpu_int8_module()
    module.weight.CB = module.weight.CB.clone()
    _repair_int8_device_state(module, "cpu")
    _assert_int8_on_device(module, "cpu")
    assert module.weight.CB.data_ptr() == module.weight.data.data_ptr()
    assert module(torch.randn(2, 8)).shape == (2, 4)
    _repair_int8_device_state(module, "cpu")
    _assert_int8_on_device(module, "cpu")


def test_no_int8_modules_fails_closed():
    with pytest.raises(RuntimeError, match="no Linear8bitLt modules"):
        _assert_int8_on_device(torch.nn.Linear(8, 4), "cpu")


def test_registered_int8_identity_records_repair():
    settings = teacher_runtime("e4b", "int8").identity()["settings"]
    assert settings["int8_device_state_repair"] == "cb_scb_moved_after_module_apply"
