"""Gemma 4 loader for a 12 GB GPU (RTX 5070).

The official checkpoints are multimodal (``model.language_model.*``,
``model.vision_tower.*``, ``model.audio_tower.*``).  Vision is part of the
project, so this loader keeps the frozen, unquantized vision tower on the GPU
and drops only the audio tower (``config.audio_config = None`` before
``from_pretrained``, so it is never materialised).

Per-Layer Embeddings (``embed_tokens_per_layer``, 4.4-4.7 GiB bf16 for E2B) are
kept on the CPU and frozen.  The embedding module is replaced by a thin wrapper
that performs the index lookup on the CPU and copies the gathered rows to the
model device, so a forward pass with ``input_ids`` on the GPU still works.
Because the lookup is a pure gather (and PLE scaling is a power of two), the
result is bit-identical to a GPU lookup, which keeps the ``N=1`` looped forward
equal to the original model.

The multimodal wrapper computes ``per_layer_inputs`` before merging the image
features and passes them to the language model; the patched looped forward
consumes that tensor unchanged and never recomputes it.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    ProcessorMixin,
)

from gemmaloop.model.layer_plan import LayerPlan, make_layer_plan, parse_block
from gemmaloop.model.snapshot import model_revision
from gemmaloop.model.looped import (
    LOOP_CONFIG_FILE,
    LOOP_STATE_FILE,
    LoopConfig,
    apply_loops,
    assert_adapter_merge_allowed,
    load_loop_state,
    set_n_loops,
)

MODEL_IDS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}

# Begin-of-image, image and end-of-image token ids (Gemma 4 config/processor).
IMAGE_TOKEN_IDS = (255999, 258880, 258882)


@dataclass
class ModelSpec:
    """Everything that determines how :func:`load_model` builds a model."""

    size: Literal["e2b", "e4b"]
    quant: Literal["bf16", "nf4"]
    adapters: tuple[str, ...] = ()
    loop: LoopConfig | None = None


_LANGUAGE_LORA_SUFFIX = (
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def language_lora_target_modules() -> str:
    """Regex for LoRA ``target_modules`` limited to the language stack.

    PEFT matches ``target_modules`` strings with :func:`re.fullmatch` against the
    module names of the wrapped model, hence the full ``model.language_model``
    prefix.  Vision and audio modules can never match.  KV-consumer layers have
    no ``k_proj``/``v_proj``; the regex simply matches fewer modules there.
    """
    return rf"model\.language_model\.layers\.\d+\.{_LANGUAGE_LORA_SUFFIX}"


def layer_set_lora_target_modules(layers: Iterable[int]) -> str:
    """Regex targeting the linear modules of exactly ``layers``.

    Used by the configurable ``"loop_block"`` / ``"loop_block_coda"`` LoRA
    target sets.  Layer
    indices are matched by an alternation, so this works for non-period-aligned
    loop blocks (e.g. 16-18).
    """
    indices = sorted({int(layer) for layer in layers})
    if not indices:
        raise ValueError("layer_set_lora_target_modules needs at least one layer")
    if any(index < 0 for index in indices):
        raise ValueError(f"layer indices must be non-negative, got {indices}")
    alternatives = "|".join(str(index) for index in indices)
    return rf"model\.language_model\.layers\.(?:{alternatives})\.{_LANGUAGE_LORA_SUFFIX}"


def lora_target_modules(
    target_layers: str | None, plan: LayerPlan | None = None
) -> str:
    """Resolve the configured LoRA target layer set to a PEFT regex.

    ``"language"`` (the default) covers every language layer.
    ``"loop_block"`` covers only the repeated block, and ``"loop_block_coda"``
    covers the block plus the coda; prelude layers never get adapters in either
    case (matching the ``loop_only`` activation policy).
    """
    name = target_layers or "language"
    if name == "language":
        return language_lora_target_modules()
    if name in ("loop_block", "loop_block_coda"):
        if plan is None:
            raise ValueError(
                f"lora_target_layers={name!r} requires the layer plan"
            )
        layers = list(plan.loop)
        if name == "loop_block_coda":
            layers += list(plan.coda)
        return layer_set_lora_target_modules(layers)
    raise ValueError(
        f"unknown lora_target_layers {target_layers!r}; expected "
        "'language', 'loop_block' or 'loop_block_coda'"
    )


class _CPUOffloadedEmbedding(nn.Module):
    """``embed_tokens_per_layer`` lookup on CPU, result on the model device.

    It exposes ``weight`` directly so the state dict name stays
    ``...embed_tokens_per_layer.weight`` and the table can still be
    saved/loaded like a plain embedding.

    The weight must survive whole-model moves: ``model.to("cuda")`` (also
    called internally by accelerate/``SFTTrainer`` for non-quantized models)
    would otherwise copy the multi-gigabyte table to the GPU.  ``_apply`` is
    overridden so the PLE table never leaves the CPU, while the registered
    (non-persistent) probe buffer follows the rest of the model and tells the
    forward pass where to send the gathered rows.  ``embed_scale`` is kept as
    a Python float so it cannot go stale on a device/dtype move either.
    """

    def __init__(self, embedding: nn.Embedding, output_device: torch.device):
        super().__init__()
        self.weight = embedding.weight
        self.padding_idx = embedding.padding_idx
        self.embed_scale = float(embedding.scalar_embed_scale)
        self.register_buffer(
            "_device_probe", torch.zeros(1, device=output_device), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        out = F.embedding(input_ids.to(weight.device), weight, self.padding_idx)
        out = out * self.embed_scale
        return out.to(self._device_probe.device)

    def _apply(self, fn, recurse: bool = True):
        """Move buffers (the probe) but keep the PLE table pinned to CPU."""
        with torch.no_grad():
            for key, buffer in self._buffers.items():
                if buffer is not None:
                    self._buffers[key] = fn(buffer)
        return self


def free_vram_gib() -> float:
    """Return the currently free VRAM of device 0 in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / 2**30


def _validate_spec(spec: ModelSpec) -> None:
    if spec.size not in MODEL_IDS:
        raise ValueError(f"unknown model size {spec.size!r}, expected one of {tuple(MODEL_IDS)}")
    if spec.quant not in ("bf16", "nf4"):
        raise ValueError(f"unknown quantization {spec.quant!r}, expected 'bf16' or 'nf4'")
    # PEFT cannot stack multiple live adapters on a quantized model.  Reject
    # the chain before any download/allocation instead of nesting wrappers.
    if spec.quant == "nf4" and len(spec.adapters) > 1:
        raise ValueError("adapter chains require bf16")


def _freeze_vision(model) -> None:
    """Freeze the vision tower and its projector (kept in bf16, no LoRA)."""
    for module in (model.model.vision_tower, model.model.embed_vision):
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


def _move_to_device_except(model: nn.Module, device: torch.device, skip: nn.Module) -> None:
    """Move parameters and buffers to ``device``, leaving ``skip``'s own tensors."""
    for module in model.modules():
        if module is skip:
            continue
        for param in module.parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device)
        for buffer in module.buffers(recurse=False):
            if buffer.device != device:
                buffer.data = buffer.data.to(device)


def _has_loop_state(directory: str | Path) -> bool:
    directory = Path(directory)
    return (directory / LOOP_STATE_FILE).is_file() and (directory / LOOP_CONFIG_FILE).is_file()


def _loop_config_from_dir(directory: str | Path) -> LoopConfig:
    """Read an adapter's loop config (bridge/ple_mode are checkpoint data).

    The saved N (new ``n_loops_last_sampled`` or legacy ``n_loops``) is a
    transient training sample and is always ignored; evaluation N comes from
    the caller's :class:`LoopConfig` (see :func:`_resolve_loop_config`).
    """
    payload = json.loads((Path(directory) / LOOP_CONFIG_FILE).read_text(encoding="utf-8"))
    block = payload.get("block")
    if block is None and isinstance(payload.get("plan"), dict):
        loop_range = payload["plan"].get("loop")
        if isinstance(loop_range, (list, tuple)) and len(loop_range) == 2:
            block = (int(loop_range[0]), int(loop_range[1]) - 1)
    return LoopConfig(
        n_loops=1,
        bridge=payload["bridge"],
        bridge_init=float(payload["bridge_init"]),
        ple_mode=payload["ple_mode"],
        block=None if block is None else parse_block(block),
        adapter_policy=payload.get("adapter_policy"),
        lora_target_layers=payload.get("lora_target_layers"),
    )


def _resolve_loop_config(
    spec_loop: LoopConfig | None, adapters: tuple[str, ...]
) -> LoopConfig | None:
    """Combine the evaluation spec with the adapter's checkpoint metadata.

    The bridge and ``ple_mode`` describe how the checkpoint was trained, so
    they come from the adapter's ``loop_config.json`` (the CLI only exposes
    ``--loops``/``--bridge``).  For a chain this is the last adapter with loop
    state, matching what :func:`_apply_adapters` finally restores.  The
    evaluation N always comes from ``spec_loop``: the adapter records the last
    N drawn by the training sampler for diagnostics only, and it must never
    decide how many passes an evaluation runs.  An explicitly requested bridge
    that disagrees with the checkpoint is an error, not a silent substitution.
    """
    adapter_loop: LoopConfig | None = None
    adapter_dir: str | Path | None = None
    for candidate in adapters:
        if _has_loop_state(candidate):
            adapter_loop = _loop_config_from_dir(candidate)
            adapter_dir = candidate

    if adapter_loop is None:
        return spec_loop
    if spec_loop is None:
        return adapter_loop
    if spec_loop.bridge != adapter_loop.bridge:
        raise ValueError(
            f"adapter {str(adapter_dir)!r} was trained with "
            f"bridge={adapter_loop.bridge!r} but the evaluation spec requests "
            f"bridge={spec_loop.bridge!r}; pass --bridge {adapter_loop.bridge} "
            "or omit --bridge"
        )
    block = adapter_loop.block
    if spec_loop.block is not None:
        if block is not None and tuple(spec_loop.block) != tuple(block):
            raise ValueError(
                f"adapter {str(adapter_dir)!r} was trained with loop block "
                f"{block[0]}-{block[1]} but the evaluation spec requests "
                f"{spec_loop.block[0]}-{spec_loop.block[1]}"
            )
        block = tuple(spec_loop.block)

    def _inherit(name: str):
        adapter_value = getattr(adapter_loop, name)
        spec_value = getattr(spec_loop, name)
        if (
            adapter_value is not None
            and spec_value is not None
            and adapter_value != spec_value
        ):
            raise ValueError(
                f"adapter {str(adapter_dir)!r} was trained with {name}="
                f"{adapter_value!r} but the evaluation spec requests "
                f"{spec_value!r}"
            )
        return adapter_value if adapter_value is not None else spec_value

    return LoopConfig(
        n_loops=int(spec_loop.n_loops),
        bridge=adapter_loop.bridge,
        bridge_init=adapter_loop.bridge_init,
        ple_mode=adapter_loop.ple_mode,
        block=block,
        adapter_policy=_inherit("adapter_policy"),
        lora_target_layers=_inherit("lora_target_layers"),
    )


def _apply_adapters(model, text_model, adapters: tuple[str, ...], quant: str):
    """Load ``adapters`` in order.

    bf16: every adapter but the last is merged into the weights; the last one
    stays a live PEFT adapter.  nf4: the single allowed adapter stays live
    (merging into a quantized model is not supported).  A ``loop_only``
    adapter is refused before merging: its weights must stay switchable per
    pass, otherwise N=1 would silently become ``base + LoRA``.
    """
    from peft import PeftModel

    if not adapters:
        return model

    peft_model = None
    for index, adapter_dir in enumerate(adapters):
        is_last = index == len(adapters) - 1
        will_merge = quant == "bf16" and not is_last
        if will_merge:
            # Merged weights cannot be switched off on pass 1, so a
            # loop_only adapter would lose its N=1 no-harm invariant.
            assert_adapter_merge_allowed(adapter_dir)
        if _has_loop_state(adapter_dir):
            load_loop_state(text_model, Path(adapter_dir))

        if will_merge:
            model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
            peft_model = None
            continue

        adapter_name = "default" if peft_model is None else f"adapter_{index}"
        peft_model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=adapter_name)
        model = peft_model
        peft_model.set_adapter(adapter_name)

    return model


def load_model(
    spec: ModelSpec, min_free_gib: float = 3.0
) -> tuple[nn.Module, ProcessorMixin, LayerPlan]:
    """Load Gemma 4 with a frozen bf16 vision tower and no audio tower.

    * ``min_free_gib`` guards against launching into a VRAM-starved GPU; the
      check runs before any download or allocation.
    * ``embed_tokens_per_layer`` (PLE) stays on the CPU, frozen; a CPU-side
      gather wrapper moves its output to the model device.
    * The vision tower and ``embed_vision`` stay in bf16 on the GPU, in
      ``eval()`` with ``requires_grad=False``; even in ``nf4`` they are skipped
      by bitsandbytes.
    * ``attn_implementation="sdpa"`` for text and vision.
    * ``spec.adapters`` are loaded in order (see :func:`_apply_adapters`).
    * ``spec.loop`` patches the inner text model with :func:`apply_loops`; loop
      state found next to an adapter is restored with :func:`load_loop_state`.
      The bridge and ``ple_mode`` come from the adapter's ``loop_config.json``
      (an explicitly requested bridge must match); the evaluation N always
      comes from ``spec.loop``.
    """
    _validate_spec(spec)

    free_gib = free_vram_gib()
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Not enough free VRAM: {free_gib:.2f} GiB available, "
            f"{min_free_gib:.2f} GiB required. Close the browser, games or other "
            "GPU applications and try again."
        )

    model_id = MODEL_IDS[spec.size]
    processor = AutoProcessor.from_pretrained(
        model_id, revision=model_revision(spec.size)
    )

    # Drop the audio tower before it is materialised.  The checkpoint still has
    # `model.audio_tower.*`/`model.embed_audio.*` keys; transformers reports
    # them as unexpected and leaves them out of the model.
    config = AutoConfig.from_pretrained(
        model_id, revision=model_revision(spec.size)
    )
    config.audio_config = None

    load_kwargs: dict = {
        "config": config,
        "dtype": torch.bfloat16,
        "attn_implementation": {
            "text_config": "sdpa",
            "vision_config": "sdpa",
        },
        "low_cpu_mem_usage": True,
    }
    if spec.quant == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Passing this list disables bitsandbytes' automatic skips, so
            # `lm_head` (tied to the embeddings) must be listed explicitly.
            llm_int8_skip_modules=[
                r".*vision_tower.*",
                r".*embed_vision.*",
                "lm_head",
            ],
        )
        # Quantize on the CPU: bitsandbytes supports it, and it means neither
        # the vision tower nor the PLE table ever reaches the GPU as a bf16
        # block.  Only the quantized text stack + bf16 vision are moved to the
        # GPU afterwards.
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=model_revision(spec.size), **load_kwargs
    )
    text_model = model.model.language_model
    text_config = model.config.text_config

    # Resolve the adapter's saved loop metadata *before* building the plan: the
    # block is part of the checkpoint geometry, not a free evaluation knob.
    loop_cfg = _resolve_loop_config(spec.loop, spec.adapters)
    plan = make_layer_plan(
        text_config.to_dict(),
        block=None if loop_cfg is None else loop_cfg.block,
    )

    _freeze_vision(model)

    ple = text_model.embed_tokens_per_layer
    ple.weight.requires_grad_(False)

    device = torch.device("cuda")
    if spec.quant == "nf4":
        # Temporarily detach the PLE so the whole-model move does not copy the
        # multi-gigabyte table to the GPU; the CPU table is re-attached below.
        text_model.embed_tokens_per_layer = nn.Identity()
        model.to(device)
    else:
        _move_to_device_except(model, device, skip=ple)

    # Freeze before wrapping so the wrapper's copy keeps requires_grad=False.
    text_model.embed_tokens_per_layer = _CPUOffloadedEmbedding(ple, device)
    gc.collect()
    torch.cuda.empty_cache()

    model.generation_config = GenerationConfig.from_pretrained(
        model_id, revision=model_revision(spec.size)
    )
    model.eval()

    if loop_cfg is not None:
        apply_loops(text_model, plan, loop_cfg)
        if spec.loop is not None:
            # Record the resolved geometry on the caller's config so RunResult,
            # run.json and the fingerprint carry the block actually executed.
            spec.loop.block = (plan.loop.start, plan.loop.stop - 1)
            spec.loop.adapter_policy = loop_cfg.adapter_policy or "always"
            spec.loop.lora_target_layers = (
                loop_cfg.lora_target_layers or "language"
            )

    model = _apply_adapters(model, text_model, spec.adapters, spec.quant)
    if spec.loop is not None:
        # Adapter loading must never decide the evaluation N (the vanished
        # train-N override): re-assert the spec's N after the adapters.
        set_n_loops(text_model, spec.loop.n_loops)
    model.eval()

    return model, processor, plan
