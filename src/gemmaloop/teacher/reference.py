"""Layer-streamed bf16 reference for the 12B teacher qualification.

The 12B bf16 checkpoint is ~24 GB and cannot sit in the 32 GB PC's RAM next to
the running qualification process.  Review X3 section 7 therefore asks for a
reference that loads **one decoder layer at a time** from the safetensors
files, runs every qualification row's hidden states through it, and keeps the
hidden states on the CPU between layers.  The fp32 vocabulary-chunked head is
then applied separately (in :mod:`gemmaloop.teacher.qualification`).

:class:`StreamingTextReference` is deliberately independent of
``from_pretrained``: it builds the few non-layer modules (embedding, norm,
rotary, optional PLE) and materialises each decoder layer from a
:class:`WeightSource`.  Tests use :class:`InMemoryWeightSource` with a tiny
model to prove the streamed output equals the normal forward.
"""

from __future__ import annotations

import ctypes
import gc
import json
from collections import UserDict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from dataclasses import dataclass

__all__ = [
    "InMemoryWeightSource",
    "ReferenceFamily",
    "SafetensorsWeightSource",
    "StreamingTextReference",
    "WeightSource",
    "reference_family",
]


@dataclass(frozen=True)
class ReferenceFamily:
    """The transformers class family used to stream a text checkpoint."""

    name: str
    decoder_layer: type
    rotary_embedding: type
    scaled_word_embedding: type
    rms_norm: type
    passes_per_layer_input: bool


def reference_family(config: Any) -> ReferenceFamily:
    """Pick the class family from the checkpoint's ``model_type``.

    The pinned 12B QAT checkpoint is ``gemma4_unified``; E2B/E4B are
    ``gemma4``.  The two implementations are currently numerically equivalent
    for the 12B text config (no PLE/MoE/KV sharing), but selecting by the
    checkpoint's own family keeps a pin or config change from silently using
    the wrong layer/attention/rope classes.
    """

    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        config = text_config
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if "unified" in model_type:
        from transformers.models.gemma4_unified import (
            modeling_gemma4_unified as unified,
        )

        return ReferenceFamily(
            name="gemma4_unified",
            decoder_layer=unified.Gemma4UnifiedTextDecoderLayer,
            rotary_embedding=unified.Gemma4UnifiedTextRotaryEmbedding,
            scaled_word_embedding=unified.Gemma4UnifiedTextScaledWordEmbedding,
            rms_norm=unified.Gemma4UnifiedRMSNorm,
            passes_per_layer_input=False,
        )
    if model_type.startswith("gemma4"):
        from transformers.models.gemma4 import modeling_gemma4 as base

        return ReferenceFamily(
            name="gemma4",
            decoder_layer=base.Gemma4TextDecoderLayer,
            rotary_embedding=base.Gemma4TextRotaryEmbedding,
            scaled_word_embedding=base.Gemma4TextScaledWordEmbedding,
            rms_norm=base.Gemma4RMSNorm,
            passes_per_layer_input=True,
        )
    raise ValueError(
        f"no layer-streamed reference family for model_type {model_type!r}"
    )


class WeightSource(Protocol):
    """Reads one named bf16 checkpoint tensor on the CPU."""

    def tensor(self, name: str) -> torch.Tensor:  # pragma: no cover - protocol
        ...


class InMemoryWeightSource:
    """A weight source backed by an in-memory state dict (CPU tests)."""

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        prefix: str = "",
        dtype: torch.dtype | None = None,
    ) -> None:
        self._state = dict(state_dict)
        self.prefix = prefix
        self.dtype = dtype

    def tensor(self, name: str) -> torch.Tensor:
        key = self.prefix + name
        if key not in self._state:
            raise KeyError(key)
        tensor = self._state[key]
        return tensor.to(self.dtype) if self.dtype is not None else tensor


_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class _TensorEntry:
    """One tensor's dtype/shape plus its byte span in the data section."""

    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class _SafetensorsHeader:
    """Parsed safetensors header (the data section starts at ``data_start``)."""

    data_start: int
    data_length: int
    entries: Mapping[str, _TensorEntry]


def _parse_safetensors_header(path: Path) -> _SafetensorsHeader:
    """Parse and validate one safetensors file header without mapping the file.

    The file is ``8-byte little-endian header length | JSON header | data``.
    Fail closed: every tensor span must match its dtype/shape, the spans must
    tile the data section exactly, and the file size must equal
    ``8 + header_length + max(end)``.
    """

    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(
            f"{path}: not a safetensors file: {file_size} bytes is too short "
            "for the 8-byte header length"
        )
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:  # pragma: no cover - stat race guard
            raise ValueError(f"{path}: could not read the 8-byte header length")
        header_length = int.from_bytes(length_bytes, "little")
        remaining = file_size - 8
        if header_length > remaining:
            raise ValueError(
                f"{path}: truncated safetensors header: header length "
                f"{header_length} exceeds the {remaining} bytes after the "
                "length prefix"
            )
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:  # pragma: no cover - stat race
            raise ValueError(
                f"{path}: truncated safetensors header: expected "
                f"{header_length} bytes, read {len(raw_header)}"
            )
    try:
        payload = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: invalid safetensors header: expected a JSON object")

    entries: dict[str, _TensorEntry] = {}
    spans: list[tuple[int, int, str]] = []
    for name, info in payload.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(
                f"{path}: invalid safetensors header entry for {name!r}: "
                "expected a JSON object"
            )
        dtype_name = info.get("dtype")
        dtype = (
            _SAFETENSORS_DTYPES.get(dtype_name)
            if isinstance(dtype_name, str)
            else None
        )
        if dtype is None:
            raise ValueError(
                f"{path}: unsupported safetensors dtype {dtype_name!r} for "
                f"tensor {name!r}"
            )
        shape = info.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in shape
        ):
            raise ValueError(
                f"{path}: invalid shape for tensor {name!r}: {shape!r}"
            )
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
                for offset in offsets
            )
        ):
            raise ValueError(
                f"{path}: invalid data_offsets for tensor {name!r}: {offsets!r}"
            )
        start, end = int(offsets[0]), int(offsets[1])
        if end < start:
            raise ValueError(
                f"{path}: invalid data_offsets for tensor {name!r}: "
                f"end {end} precedes start {start}"
            )
        numel = 1
        for dim in shape:
            numel *= int(dim)
        expected_bytes = numel * dtype.itemsize
        if end - start != expected_bytes:
            raise ValueError(
                f"{path}: data_offsets for tensor {name!r} span {end - start} "
                f"bytes but dtype {dtype_name} and shape {shape!r} need "
                f"{expected_bytes}"
            )
        entries[name] = _TensorEntry(
            dtype=dtype, shape=tuple(int(dim) for dim in shape), start=start, end=end
        )
        spans.append((start, end, name))

    data_start = 8 + header_length
    data_length = file_size - data_start
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if end > data_length:
            raise ValueError(
                f"{path}: data_offsets for tensor {name!r} end at {end}, "
                f"beyond the {data_length}-byte data section"
            )
        if start != cursor:
            raise ValueError(
                f"{path}: data_offsets are not contiguous at tensor {name!r}: "
                f"expected {cursor}, got {start}"
            )
        cursor = end
    if cursor != data_length:
        raise ValueError(
            f"{path}: file size {file_size} does not match "
            f"8 + header_length + max(end) = {data_start + cursor}"
        )
    return _SafetensorsHeader(
        data_start=data_start,
        data_length=data_length,
        entries=entries,
    )


class SafetensorsWeightSource:
    """Reads tensors lazily from a sharded safetensors checkpoint.

    The 12B checkpoint is one ~24 GB file.  Mapping that file once per tensor
    (``safe_open(...).get_tensor(...)``) keeps every copy-on-write view alive
    and blows the process commit budget on Windows.  This source instead reads
    each header once, then seeks to the tensor's bytes and copies exactly that
    span into a fresh writable buffer.  Returned tensors therefore reference
    ordinary process memory and never alias a file mapping.
    """

    def __init__(self, directory: str | Path, *, prefix: str | None = None) -> None:
        self.directory = Path(directory)
        self._shards: dict[str, Path] = {}
        self._keys: set[str] = set()
        self._headers: dict[Path, _SafetensorsHeader] = {}
        self._load_index()
        self.prefix = (
            _infer_prefix(self._keys) if prefix is None else str(prefix)
        )

    def _header(self, path: Path) -> _SafetensorsHeader:
        header = self._headers.get(path)
        if header is None:
            header = _parse_safetensors_header(path)
            self._headers[path] = header
        return header

    def _load_index(self) -> None:
        index_path = self.directory / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") or {}
            for name, filename in weight_map.items():
                self._shards[str(name)] = self.directory / str(filename)
                self._keys.add(str(name))
            return
        files = sorted(self.directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"no safetensors files in {self.directory}"
            )
        for path in files:
            for key in self._header(path).entries:
                self._keys.add(key)
                self._shards[key] = path

    def tensor(self, name: str) -> torch.Tensor:
        key = self.prefix + name
        path = self._shards.get(key)
        if path is None:
            raise KeyError(key)
        header = self._header(path)
        entry = header.entries.get(key)
        if entry is None:
            raise KeyError(key)
        byte_count = entry.end - entry.start
        if byte_count == 0:
            return torch.empty(entry.shape, dtype=entry.dtype)
        # Allocate the destination with torch so the returned tensor owns a
        # normal, resizable storage, then read the file span straight into it.
        flat = torch.empty(byte_count, dtype=torch.uint8)
        raw = (ctypes.c_char * byte_count).from_address(flat.data_ptr())
        view = memoryview(raw)
        filled = 0
        with path.open("rb") as handle:
            handle.seek(header.data_start + entry.start)
            while filled < byte_count:
                read = handle.readinto(view[filled:])
                if not read:
                    raise ValueError(
                        f"{path}: truncated safetensors file while reading "
                        f"{key!r}: expected {byte_count} bytes at offset "
                        f"{header.data_start + entry.start}, got {filled}"
                    )
                filled += read
        del view, raw
        return flat.view(entry.dtype).reshape(entry.shape)


def _empty_cache_or_gc(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        gc.collect()


def _rss_gib() -> float:
    """Best-effort process RSS in GiB (0 when psutil is unavailable)."""

    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 2**30
    except Exception:  # pragma: no cover - optional dependency
        return 0.0


def _cuda_peak_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated()) / 2**30,
        float(torch.cuda.max_memory_reserved()) / 2**30,
    )


def _infer_prefix(keys: Sequence[str]) -> str:
    """Find the text-model prefix that owns ``layers.0`` and ``embed_tokens``."""

    for key in keys:
        if key.endswith("layers.0.self_attn.q_proj.weight"):
            return key[: -len("layers.0.self_attn.q_proj.weight")]
    for key in keys:
        if key.endswith("embed_tokens.weight"):
            return key[: -len("embed_tokens.weight")]
    return ""


class StreamingTextReference:
    """The text decoder evaluated one layer at a time."""

    def __init__(
        self,
        config: Any,
        source: WeightSource,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefix: str = "",
        ram_limit_gib: float | None = 6.0,
        vram_limit_gib: float | None = None,
        record_memory: bool = True,
        family: ReferenceFamily | None = None,
    ) -> None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            config = text_config
        self.family = family or reference_family(config)
        self.config = config
        self.source = source
        self.prefix = prefix
        self.device = torch.device(device)
        self.dtype = dtype
        self.ram_limit_gib = (
            None if ram_limit_gib is None else float(ram_limit_gib)
        )
        self.vram_limit_gib = (
            None if vram_limit_gib is None else float(vram_limit_gib)
        )
        self.record_memory = bool(record_memory)
        self.memory_report: dict[str, Any] = {}
        self.num_layers = int(config.num_hidden_layers)
        self.unique_layer_types = sorted(set(config.layer_types))

        self.embed_tokens = self._embedding(
            int(config.vocab_size),
            int(config.hidden_size),
            float(config.hidden_size**0.5),
            "embed_tokens.weight",
        )
        self.norm = self.family.rms_norm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.norm.load_state_dict({"weight": self._tensor("norm.weight")})
        self.rotary_emb = self.family.rotary_embedding(
            config, device=self.device
        )

        self.hidden_size_per_layer_input = int(
            getattr(config, "hidden_size_per_layer_input", 0) or 0
        )
        if self.hidden_size_per_layer_input:
            from torch import nn as _nn

            self.embed_tokens_per_layer = self._embedding(
                int(config.vocab_size_per_layer_input),
                self.num_layers * self.hidden_size_per_layer_input,
                float(self.hidden_size_per_layer_input**0.5),
                "embed_tokens_per_layer.weight",
            )
            self.per_layer_model_projection = _nn.Linear(
                int(config.hidden_size),
                self.num_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection.load_state_dict(
                {"weight": self._tensor("per_layer_model_projection.weight")}
            )
            self.per_layer_projection_norm = self.family.rms_norm(
                self.hidden_size_per_layer_input,
                eps=float(config.rms_norm_eps),
            )
            self.per_layer_projection_norm.load_state_dict(
                {"weight": self._tensor("per_layer_projection_norm.weight")}
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection_scale = float(config.hidden_size) ** -0.5

        for module in self._non_layer_modules():
            module.to(device=self.device, dtype=self.dtype)
            module.eval()
        for parameter in self._non_layer_parameters():
            parameter.requires_grad_(False)

    # -- component construction ---------------------------------------------

    def _tensor(self, name: str) -> torch.Tensor:
        return self.source.tensor(self.prefix + name)

    def _embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float,
        name: str,
    ) -> nn.Module:
        padding_idx = getattr(self.config, "pad_token_id", None)
        embedding = self.family.scaled_word_embedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            embed_scale=embed_scale,
        )
        embedding.load_state_dict({"weight": self._tensor(name)})
        return embedding

    def _non_layer_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            self.embed_tokens,
            self.norm,
            self.rotary_emb,
        ]
        if self.hidden_size_per_layer_input:
            modules += [
                self.embed_tokens_per_layer,
                self.per_layer_model_projection,
                self.per_layer_projection_norm,
            ]
        return modules

    def _non_layer_parameters(self):
        for module in self._non_layer_modules():
            for parameter in module.parameters():
                yield parameter
            for buffer in module.buffers():
                yield buffer

    def _load_layer(self, index: int) -> nn.Module:
        """Materialise exactly one decoder layer on the target device."""

        with torch.device("meta"):
            layer = self.family.decoder_layer(self.config, index)
        state: dict[str, torch.Tensor] = {}
        for name, _ in list(layer.named_parameters()) + list(
            layer.named_buffers()
        ):
            try:
                state[name] = self._tensor(f"layers.{index}.{name}")
            except KeyError:
                continue
        layer.load_state_dict(state, strict=False, assign=True)

        missing = [
            name for name, parameter in layer.named_parameters() if parameter.is_meta
        ]
        if missing:
            raise KeyError(
                f"layer {index} is missing checkpoint parameters: {missing[:4]}"
            )
        # Non-persistent rotary buffers are not in the checkpoint; replace the
        # deterministic rotary module so no meta buffer survives.
        for module_name, module in list(layer.named_modules()):
            if isinstance(module, self.family.rotary_embedding):
                parent_name, _, child_name = module_name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(
                    parent,
                    child_name,
                    self.family.rotary_embedding(
                        self.config, device=self.device
                    ),
                )
        layer.to(device=self.device, dtype=self.dtype)
        layer.eval()
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        return layer

    # -- forward -------------------------------------------------------------

    def _per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.hidden_size_per_layer_input:
            return None
        token_identity = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.num_layers,
            self.hidden_size_per_layer_input,
        )
        projection = self.per_layer_model_projection(inputs_embeds)
        projection = projection * self.per_layer_model_projection_scale
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_layers,
            self.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        return (projection + token_identity) * self.per_layer_input_scale

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the final hidden states on the CPU (bf16 by default)."""

        return self.forward_many([input_ids], attention_masks=[attention_mask])[0]

    @torch.inference_mode()
    def forward_many(
        self,
        input_ids_batch: Sequence[torch.Tensor],
        *,
        attention_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Run all rows through each decoder layer before freeing it.

        Every layer is loaded from the safetensors files once, applied to all
        qualification rows' hidden states (kept on the CPU between layers), then
        freed.  Only the hidden states, PLE inputs (when the family has PLE) and
        position ids stay on the CPU; masks and rotary tensors are rebuilt from
        the current hidden shape for each row/layer.  GPU memory therefore stays
        bounded by one layer plus one row's activations, and the RAM report is
        asserted against ``ram_limit_gib``.
        """

        device = self.device
        if not input_ids_batch:
            return []
        baseline_ram = _rss_gib()
        peak_ram = baseline_ram
        per_layer_memory: list[dict[str, Any]] = []
        peak_vram_allocated = 0.0
        peak_vram_reserved = 0.0
        masks_in = (
            [None] * len(input_ids_batch)
            if attention_masks is None
            else list(attention_masks)
        )
        if len(masks_in) != len(input_ids_batch):
            raise ValueError("attention_masks must match the input batch")

        state: list[dict[str, Any]] = []
        for input_ids, attention_mask in zip(input_ids_batch, masks_in):
            ids = input_ids.to(device)
            inputs_embeds = self.embed_tokens(ids)
            per_layer_inputs = self._per_layer_inputs(ids, inputs_embeds)
            position_ids = torch.arange(
                ids.shape[1], device=device, dtype=torch.long
            ).unsqueeze(0)
            state.append(
                {
                    "hidden": inputs_embeds.detach().to("cpu"),
                    "per_layer_inputs": (
                        None
                        if per_layer_inputs is None
                        else per_layer_inputs.detach().to("cpu")
                    ),
                    "position_ids": position_ids.detach().to("cpu"),
                    "attention_mask": (
                        None
                        if attention_mask is None
                        else attention_mask.detach().to("cpu")
                    ),
                }
            )
            del inputs_embeds, per_layer_inputs, ids
            _empty_cache_or_gc(device)
            if self.record_memory:
                peak_ram = max(peak_ram, _rss_gib())

        shared_kv_states = [UserDict() for _ in state]
        for index in range(self.num_layers):
            layer_start_ram = _rss_gib() if self.record_memory else 0.0
            layer_peak_ram = layer_start_ram
            layer_peak_vram_allocated = 0.0
            layer_peak_vram_reserved = 0.0
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            layer = self._load_layer(index)
            if self.record_memory:
                layer_peak_ram = max(layer_peak_ram, _rss_gib())
                allocated, reserved = _cuda_peak_gib()
                layer_peak_vram_allocated = max(
                    layer_peak_vram_allocated, allocated
                )
                layer_peak_vram_reserved = max(
                    layer_peak_vram_reserved, reserved
                )
            try:
                layer_type = self.config.layer_types[index]
                for row, item in enumerate(state):
                    position_ids = item["position_ids"].to(device)
                    hidden_input = item["hidden"].to(device)
                    attention_mask = (
                        None
                        if item["attention_mask"] is None
                        else item["attention_mask"].to(device)
                    )
                    mask_kwargs = {
                        "config": self.config,
                        "inputs_embeds": hidden_input,
                        "attention_mask": attention_mask,
                        "past_key_values": None,
                        "position_ids": position_ids,
                    }
                    if layer_type == "full_attention":
                        causal_mask = create_causal_mask(**mask_kwargs)
                    else:
                        causal_mask = create_sliding_window_causal_mask(**mask_kwargs)
                    position_embedding = tuple(
                        value.to(device)
                        for value in self.rotary_emb(
                            hidden_input, position_ids, layer_type
                        )
                    )
                    per_layer_input = (
                        None
                        if item["per_layer_inputs"] is None
                        else item["per_layer_inputs"][:, :, index, :].to(device)
                    )
                    layer_kwargs = {
                        "shared_kv_states": shared_kv_states[row],
                        "position_embeddings": position_embedding,
                        "attention_mask": causal_mask,
                        "position_ids": position_ids,
                        "past_key_values": None,
                    }
                    if self.family.passes_per_layer_input:
                        output = layer(
                            hidden_input, per_layer_input, **layer_kwargs
                        )
                    else:
                        output = layer(hidden_input, **layer_kwargs)
                    item["hidden"] = output.detach().to("cpu")
                    if self.record_memory:
                        # Sample while the layer, its output, the hidden input
                        # and any attention workspace are still alive.
                        layer_peak_ram = max(layer_peak_ram, _rss_gib())
                        allocated, reserved = _cuda_peak_gib()
                        layer_peak_vram_allocated = max(
                            layer_peak_vram_allocated, allocated
                        )
                        layer_peak_vram_reserved = max(
                            layer_peak_vram_reserved, reserved
                        )
                    del (
                        output,
                        position_ids,
                        position_embedding,
                        per_layer_input,
                        hidden_input,
                        attention_mask,
                        causal_mask,
                    )
            finally:
                if self.record_memory:
                    # The layer is still materialised here: its checkpoint
                    # tensors and buffers are part of the real per-layer peak.
                    layer_peak_ram = max(layer_peak_ram, _rss_gib())
                    allocated, reserved = _cuda_peak_gib()
                    layer_peak_vram_allocated = max(
                        layer_peak_vram_allocated, allocated
                    )
                    layer_peak_vram_reserved = max(
                        layer_peak_vram_reserved, reserved
                    )
                    peak_ram = max(peak_ram, layer_peak_ram)
                    peak_vram_allocated = max(
                        peak_vram_allocated, layer_peak_vram_allocated
                    )
                    peak_vram_reserved = max(
                        peak_vram_reserved, layer_peak_vram_reserved
                    )
                    per_layer_memory.append(
                        {
                            "layer": int(index),
                            "ram_above_baseline_gib": round(
                                max(0.0, layer_peak_ram - baseline_ram), 4
                            ),
                            "vram_allocated_gib": round(
                                layer_peak_vram_allocated, 4
                            ),
                            "vram_reserved_gib": round(
                                layer_peak_vram_reserved, 4
                            ),
                        }
                    )
                del layer
                _empty_cache_or_gc(device)

        results: list[torch.Tensor] = []
        for item in state:
            final = self.norm(item["hidden"].to(device))
            item["hidden"] = None
            results.append(final.detach().to("cpu"))
            del final
        if self.record_memory:
            peak_ram = max(peak_ram, _rss_gib())
            above = max(0.0, peak_ram - baseline_ram)
            self.memory_report = {
                "baseline_ram_gib": round(baseline_ram, 4),
                "peak_ram_gib": round(peak_ram, 4),
                "peak_ram_above_baseline_gib": round(above, 4),
                "ram_limit_gib": self.ram_limit_gib,
                "within_ram_limit": (
                    self.ram_limit_gib is None
                    or above <= float(self.ram_limit_gib)
                ),
                "vram_limit_gib": self.vram_limit_gib,
                "within_vram_limit": (
                    self.vram_limit_gib is None
                    or peak_vram_reserved <= float(self.vram_limit_gib)
                ),
                "peak_vram_allocated_gib": round(peak_vram_allocated, 4),
                "peak_vram_reserved_gib": round(peak_vram_reserved, 4),
                "per_layer": per_layer_memory,
            }
            if (
                self.ram_limit_gib is not None
                and above > float(self.ram_limit_gib)
            ):
                raise RuntimeError(
                    "layer-streamed reference exceeded its RAM bound: "
                    f"{above:.3f} GiB above baseline > {self.ram_limit_gib:.3f} GiB"
                )
            if (
                self.vram_limit_gib is not None
                and peak_vram_reserved > float(self.vram_limit_gib)
            ):
                raise RuntimeError(
                    "layer-streamed reference exceeded its VRAM bound: "
                    f"{peak_vram_reserved:.3f} GiB reserved > "
                    f"{self.vram_limit_gib:.3f} GiB"
                )
        return results

    def close(self) -> None:
        for module in self._non_layer_modules():
            module.to("cpu")
        del self.rotary_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
