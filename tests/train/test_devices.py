"""Regression tests for probe inputs following the text embeddings.

The real GPU smoke failed every probe because the frozen vision tower is
offloaded to the CPU and is the first registered parameter; deriving the probe
device with ``next(model.parameters()).device`` therefore put token ids on the
CPU while the text model ran on CUDA.  These tests use a ``meta`` first
parameter (a stand-in for the offloaded tower) so the CPU-only suite catches a
regression immediately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gemmaloop.train.devices import model_compute_device
from gemmaloop.train.heal_monitor import build_probe_batches, build_prompt_tensors
from gemmaloop.train.heal_probes import ProbeConfig, ProbeRunner


class _FakeTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)


class _OffloadedVisionModel(torch.nn.Module):
    """First parameter on ``meta``; text embeddings on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Linear(2, 2, device="meta")
        self.model = torch.nn.Module()
        self.model.language_model = _FakeTextModel()


class _PeftLikeWrapper(torch.nn.Module):
    """PEFT-style ``base_model`` wrapper whose ``get_input_embeddings`` fails."""

    def __init__(self) -> None:
        super().__init__()
        self.base_model = _OffloadedVisionModel()

    def get_base_model(self):
        return self.base_model

    def get_input_embeddings(self):
        raise NotImplementedError("wrapper does not expose embeddings")


class _Dataset:
    column_names = ["input_ids", "labels", "attention_mask"]

    def __init__(self) -> None:
        self.rows = [
            {
                "input_ids": [3, 4, 5],
                "labels": [-100, 4, 5],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [6, 7],
                "labels": [-100, 7],
                "attention_mask": [1, 1],
            },
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


@pytest.mark.parametrize("model_type", [_OffloadedVisionModel, _PeftLikeWrapper])
def test_model_compute_device_ignores_the_offloaded_first_parameter(model_type):
    model = model_type()
    assert next(model.parameters()).device.type == "meta"
    assert model_compute_device(model) == torch.device("cpu")


def test_build_probe_batches_uses_the_embedding_device_not_the_first_parameter():
    model = _OffloadedVisionModel()
    batches = build_probe_batches(
        _Dataset(),
        [0, 1],
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0),
        batch_size=2,
    )
    assert batches
    for batch in batches:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu")
                assert value.device.type != "meta"


def test_build_prompt_tensors_uses_the_embedding_device_not_the_first_parameter():
    model = _OffloadedVisionModel()
    prompts = build_prompt_tensors(
        _Dataset(),
        [0, 1],
        device=model_compute_device(model),
        count=2,
    )
    assert [prompt.tolist() for prompt in prompts] == [[3], [6]]
    assert all(prompt.device == torch.device("cpu") for prompt in prompts)


class _GeneratingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Linear(2, 2, device="meta")
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.seen_devices: list[torch.device] = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def generate(self, *, input_ids=None, attention_mask=None, **kwargs):
        assert attention_mask is not None
        self.seen_devices.append(input_ids.device)
        next_token = torch.full(
            (input_ids.shape[0], 1),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, next_token], dim=1)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


def _generation_runner(model: _GeneratingModel) -> ProbeRunner:
    return ProbeRunner(
        model=model,
        text_model=model,
        weight=model.embed_tokens.weight,
        softcap=None,
        batches=[],
        config=ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(1,),
            generation_ns=(1,),
            generation_max_new_tokens=1,
        ),
        tokenizer=_Tokenizer(),
    )


def test_probe_runner_moves_inputs_to_the_embedding_device():
    model = _GeneratingModel()
    assert next(model.parameters()).device.type == "meta"
    runner = _generation_runner(model)

    assert runner._generate_one(torch.tensor([3, 4]), 1) == [2]
    assert runner._generate_group([torch.tensor([3, 4]), torch.tensor([5])], 1) == [
        [2],
        [2],
    ]
    assert model.seen_devices == [torch.device("cpu"), torch.device("cpu")]


@pytest.mark.parametrize(
    "model",
    [
        torch.nn.Linear(2, 2),
        SimpleNamespace(get_input_embeddings=lambda: None),
    ],
)
def test_model_compute_device_fails_closed_without_input_embeddings(model):
    with pytest.raises(RuntimeError, match="cannot locate the text input embeddings"):
        model_compute_device(model)


class _TinyOffloadedCausalLM:
    """Real HF ``GenerationMixin`` model with an offloaded first parameter."""

    @staticmethod
    def build():
        from transformers import GPT2Config, GPT2LMHeadModel

        class _Model(GPT2LMHeadModel):
            def __init__(self, config):
                super().__init__(config)
                self.vision_tower = torch.nn.Linear(2, 2).to("meta")
                # Registration order is what makes ``model.device`` wrong.
                modules = self._modules
                self._modules = {
                    "vision_tower": modules.pop("vision_tower"),
                    **modules,
                }

        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        return _Model(config)


def test_real_generation_mixin_survives_an_offloaded_first_parameter(monkeypatch):
    """B1 regression: HF generate must see the text compute device.

    transformers 5.22.1 skips the per-step move only for a ``meta`` device; a
    CUDA model with a CPU vision tower is not skipped.  This CPU test restores
    the audited unconditional move so the real ``GenerationMixin`` path fails
    on the offloaded first parameter without the fix.
    """

    from transformers.generation import utils as generation_utils

    original = generation_utils.GenerationMixin.prepare_inputs_for_generation

    def forced_move(self, input_ids, *args, **kwargs):
        model_inputs = original(self, input_ids, *args, **kwargs)
        tensor = model_inputs.get("inputs_embeds", model_inputs.get("input_ids"))
        if tensor is not None and tensor.device != self.device:
            model_inputs = generation_utils.move_to_device(
                model_inputs, self.device
            )
        return model_inputs

    monkeypatch.setattr(
        generation_utils.GenerationMixin,
        "prepare_inputs_for_generation",
        forced_move,
    )
    model = _TinyOffloadedCausalLM.build()
    assert next(model.parameters()).device.type == "meta"
    assert model.get_input_embeddings().weight.device.type == "cpu"

    # Without the probe's device context, generate moves ids to the offloaded
    # module's device and the real forward fails.
    with pytest.raises(Exception):
        model.generate(
            torch.tensor([[1, 3, 4]]),
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
        )

    runner = ProbeRunner(
        model=model,
        text_model=model,
        weight=model.get_input_embeddings().weight,
        softcap=None,
        batches=[],
        config=ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(1,),
            generation_ns=(1,),
            generation_max_new_tokens=2,
        ),
        tokenizer=_Tokenizer(),
    )
    generated = runner._generate_one(torch.tensor([3, 4]), 1)
    assert generated
    # The offload state is restored exactly.
    assert next(model.parameters()).device.type == "meta"
    assert model.get_input_embeddings().weight.device.type == "cpu"


class _FakeTensor:
    """Faithful enough for the non-meta branch of ``_move_tensor``.

    A CPU-only suite cannot move a real tensor between CPU and CUDA.  This fake
    records the requested device, returns a data object with that device, and
    can fail one specific move so the partial-failure restore can be exercised.
    """

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.shape = (2,)
        self.dtype = torch.float32
        self._data = None
        self.to_calls: list[torch.device] = []
        self.fail_to: torch.device | None = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # Assigning ``Parameter.data`` moves the tensor in production, so the
        # fake's visible device must follow the assigned payload.
        self._data = value
        self.device = torch.device(value.device)

    def to(self, device):
        target = torch.device(device)
        self.to_calls.append(target)
        if self.fail_to is not None and target == self.fail_to:
            raise RuntimeError("fake CUDA OOM while moving in")
        return SimpleNamespace(device=target)


def test_move_tensor_covers_the_production_non_meta_branch():
    """Covers ``_move_tensor``'s non-meta ``.to()`` branch and its early exit.

    Not covered: a real CPU<->CUDA copy (no GPU in the CPU suite).  Covered:
    the exact production code path selected for the CPU-offloaded production
    tensors, including the no-op when the tensor is already on its home device.
    """

    from gemmaloop.train.heal_probes import _move_tensor

    tensor = _FakeTensor("cuda:0")
    _move_tensor(tensor, torch.device("cpu"))
    assert tensor.to_calls == [torch.device("cpu")]
    assert tensor.data.device == torch.device("cpu")

    _move_tensor(tensor, torch.device("cpu"))
    assert tensor.to_calls == [torch.device("cpu")]

    _move_tensor(tensor, torch.device("cuda:0"))
    assert tensor.to_calls[-1] == torch.device("cuda:0")
    assert tensor.data.device == torch.device("cuda:0")


def test_generation_device_restores_every_tensor_on_success(monkeypatch):
    import gemmaloop.train.heal_probes as heal_probes

    model = _FakeTextModel()
    tensors = [_FakeTensor("cuda:0"), _FakeTensor("cuda:1")]
    monkeypatch.setattr(
        heal_probes,
        "_generation_offload_modules",
        lambda model: [("vision_tower", object())],
    )
    monkeypatch.setattr(
        heal_probes, "_module_tensors", lambda module: list(tensors)
    )

    with heal_probes._generation_device(model) as compute:
        assert compute == torch.device("cpu")
        assert all(tensor.data.device == torch.device("cpu") for tensor in tensors)

    assert tensors[0].data.device == torch.device("cuda:0")
    assert tensors[1].data.device == torch.device("cuda:1")


def test_generation_device_restores_a_partial_move_when_a_later_move_fails(
    monkeypatch,
):
    """The move loop is inside ``try``: a mid-loop OOM still restores homes."""

    import gemmaloop.train.heal_probes as heal_probes

    model = _FakeTextModel()
    first = _FakeTensor("cuda:0")
    second = _FakeTensor("cuda:0")
    second.fail_to = torch.device("cpu")
    monkeypatch.setattr(
        heal_probes,
        "_generation_offload_modules",
        lambda model: [("vision_tower", object())],
    )
    monkeypatch.setattr(
        heal_probes, "_module_tensors", lambda module: [first, second]
    )

    with pytest.raises(RuntimeError, match="fake CUDA OOM"):
        with heal_probes._generation_device(model):
            pass

    # The first tensor had already moved in; the finally block put it back.
    assert first.to_calls == [torch.device("cpu"), torch.device("cuda:0")]
    assert first.data.device == torch.device("cuda:0")
    # The failing tensor never moved, so its restore is a no-op.
    assert second.to_calls == [torch.device("cpu")]
    assert second.data is None
