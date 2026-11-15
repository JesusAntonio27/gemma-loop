"""Memory/speed switches for LoRA training on a 12 GiB GPU (Task 13b).

None of these switches changes what the model learns: same data, same loss
(see :mod:`gemmaloop.train.fused_ce`), same LoRA/loop parameters and the same
optimizer.  They only change where tensors live and which tensors autograd
keeps.

* ``lean_lm_head`` - replace TRL's fused LM head (torch fallback, large fp32
  tiles) with :func:`gemmaloop.train.fused_ce.install_lean_lm_head`.
* ``input_grads=False`` - remove the ``enable_input_require_grads`` hook that
  TRL installs for PEFT + gradient checkpointing.  The hook only matters for
  *reentrant* checkpointing; with ``use_reentrant=False`` it merely makes
  autograd record (and keep activations for) the frozen PLE projection and the
  input gradient of layer 0, neither of which reaches a trainable parameter.
* ``offload_vision`` - the training data is text only, so the frozen vision
  tower and its projector are moved to the CPU when training starts (after
  ``accelerator.prepare`` placed the model) and moved back when it ends.  A
  forward pre-hook raises if anything calls them while they are offloaded.
* ``banded_sliding`` - sliding-window layers attend through
  :func:`gemmaloop.train.sliding_attention.banded_sliding_sdpa` (same SDPA
  kernel and mask semantics, only the visible key band of each query block)
  instead of masking a full ``[seq, seq]`` score matrix.
* ``checkpoint_skip_token_calls`` - gradient checkpointing only pays off for
  long micro steps; short ones are bound by kernel-launch (CPU) overhead,
  which the recompute forward adds to.  A micro step whose activations are
  small - ``tokens * decoder_layer_calls <= checkpoint_skip_token_calls``,
  with ``35 + 15 (N - 1)`` layer calls for E2B - runs without checkpointing.
  Non-reentrant checkpointing replays the same RNG state (dropout masks) and
  restores the RNG afterwards, so the loss and gradients are the same with or
  without it.  ``0`` (the default) checkpoints every micro step.
  NOT RECOMMENDED: on fixed-length benchmarks it is 1.5-2x faster for short
  rows (-7% projected GPU hours), but in a real mixed-length run (train.py,
  E2B, L arm, max_length 8192) the large non-checkpointed activations
  fragment the caching allocator: peak reserved 10.26 GiB (at the cap)
  instead of 9.25 GiB and 2.6x slower (168 s vs 65 s, same loss).
* ``host_performance_cores`` - short micro steps are bound by the Python /
  kernel-launch thread.  On a hybrid CPU (i5-14600KF: 6 P-cores, 8 E-cores)
  Windows may run a background console process on E-cores or with EcoQoS
  throttling, which roughly doubles the per-step host time (bimodal step
  times between runs).  This opts the training process (only this process,
  nothing system-wide) out of power throttling and gives it the P-cores as
  default CPU sets.  No-op elsewhere.
* ``offload_checkpoints`` - transformers' own ``gradient_checkpointing_enable(
  offload=True)``: the per-layer checkpoint inputs (``[seq, hidden]`` per
  decoder-layer call, i.e. ``35 + 15 (N - 1)`` of them) live in pinned host
  memory between forward and backward.  Off by default (it costs copies).

``Trainer.train()`` re-runs ``gradient_checkpointing_enable()``, which
re-installs the input-grad hook, so the hook removal (and the checkpoint
offload) happen in :class:`TrainPerfCallback` at ``on_train_begin``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import torch
from transformers import TrainerCallback

from gemmaloop.train.fused_ce import (
    DEFAULT_TOKEN_CHUNK,
    DEFAULT_VOCAB_CHUNK,
    install_lean_lm_head,
)
from gemmaloop.train.sliding_attention import (
    DEFAULT_BLOCK,
    DEFAULT_MIN_LENGTH,
    install_banded_sliding_attention,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AdaptiveCheckpointing",
    "TrainPerfCallback",
    "TrainPerfConfig",
    "VisionOffloadCallback",
    "apply_train_perf",
    "prevent_sleep",
    "SleepGuard",
    "disable_input_require_grads",
    "cap_cuda_allocator",
    "prefer_performance_cores",
    "unused_text_only_modules",
]


@dataclass(frozen=True)
class TrainPerfConfig:
    """Performance switches; defaults are the configuration proven equivalent."""

    lean_lm_head: bool = True
    lm_head_token_chunk: int = DEFAULT_TOKEN_CHUNK
    lm_head_vocab_chunk: int = DEFAULT_VOCAB_CHUNK
    lm_head_compact: bool = True
    input_grads: bool = False
    offload_vision: bool = True
    offload_checkpoints: bool = False
    banded_sliding: bool = True
    banded_block: int = DEFAULT_BLOCK
    banded_min_length: int = DEFAULT_MIN_LENGTH
    checkpoint_skip_token_calls: int = 0
    host_performance_cores: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE_PERF = TrainPerfConfig(
    lean_lm_head=False,
    input_grads=True,
    offload_vision=False,
    banded_sliding=False,
    host_performance_cores=False,
)


def cap_cuda_allocator(fraction: float = 0.92, free_margin_gib: float = 0.5) -> dict | None:
    """Cap this process's caching allocator so over-budget steps OOM instead of spilling.

    Under WDDM an allocation beyond physical VRAM does not fail: the driver
    pages to system RAM (tens of times slower; two overlapping GPU jobs
    ended in a driver crash).  The cap is ``min(fraction * total, free now -
    free_margin_gib)``: what the desktop and the CUDA context already use is
    left alone.  When the cap is reached the allocator first frees its cache
    and retries, so fragmentation alone does not fail a step.  Call it after
    CUDA is initialised and before the model is loaded.  Returns the numbers
    (or None without CUDA).
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    cap = min(float(fraction) * total, free - float(free_margin_gib) * 2**30)
    if cap <= 0:
        raise RuntimeError(f"only {free / 2**30:.2f} GiB VRAM free; another job is probably running")
    torch.cuda.set_per_process_memory_fraction(cap / total)
    info = {
        "total_gib": round(total / 2**30, 3),
        "free_at_start_gib": round(free / 2**30, 3),
        "cap_gib": round(cap / 2**30, 3),
    }
    LOGGER.info("CUDA allocator cap %s", info)
    return info


def prefer_performance_cores() -> dict:
    """Windows: run this process on the most performant cores, without EcoQoS.

    Uses ``SetProcessInformation(ProcessPowerThrottling)`` (execution-speed
    throttling off) and ``SetProcessDefaultCpuSets`` (a soft affinity to the
    CPU sets of the highest efficiency class).  Both only affect the calling
    process and end with it.  Returns what was applied; never raises.
    """
    import os

    applied: dict = {"power_throttling_off": False, "cpu_sets": None}
    if os.name != "nt":
        return applied
    try:
        import ctypes
        import ctypes.wintypes as wt
        import struct

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        process = kernel32.GetCurrentProcess()

        class _Throttling(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

        kernel32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        state = _Throttling(1, 0x1, 0x0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED off
        applied["power_throttling_off"] = bool(
            kernel32.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))
        )

        kernel32.GetSystemCpuSetInformation.argtypes = [
            ctypes.c_void_p, wt.ULONG, ctypes.POINTER(wt.ULONG), wt.HANDLE, wt.ULONG
        ]
        needed = wt.ULONG(0)
        kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
        buffer = ctypes.create_string_buffer(needed.value)
        if not kernel32.GetSystemCpuSetInformation(buffer, needed, ctypes.byref(needed), None, 0):
            return applied
        sets: list[tuple[int, int]] = []
        offset = 0
        while offset < needed.value:
            size, kind = struct.unpack_from("<II", buffer, offset)
            if kind == 0:  # CpuSetInformation
                set_id = struct.unpack_from("<I", buffer, offset + 8)[0]
                efficiency = struct.unpack_from("<B", buffer, offset + 18)[0]
                sets.append((set_id, efficiency))
            offset += size
        classes = {efficiency for _, efficiency in sets}
        if len(classes) > 1:
            best = max(classes)
            ids = [set_id for set_id, efficiency in sets if efficiency == best]
            array = (wt.ULONG * len(ids))(*ids)
            kernel32.SetProcessDefaultCpuSets.argtypes = [wt.HANDLE, ctypes.POINTER(wt.ULONG), wt.ULONG]
            if kernel32.SetProcessDefaultCpuSets(process, array, len(ids)):
                applied["cpu_sets"] = ids
    except Exception as exc:  # noqa: BLE001 - a scheduling hint must never break training
        LOGGER.warning("prefer_performance_cores failed: %s", exc)
    return applied


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def unused_text_only_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Frozen modules a text-only forward never calls (vision tower, projector)."""
    inner = getattr(_unwrap(model), "model", None)
    modules: dict[str, torch.nn.Module] = {}
    for name in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
        module = getattr(inner, name, None)
        if isinstance(module, torch.nn.Module):
            modules[name] = module
    return modules


# Forward hooks that only do ``output.requires_grad_(True)`` on the input
# embeddings: transformers' ``enable_input_require_grads`` and PEFT's fallback.
_INPUT_GRAD_HOOK_NAMES = frozenset({"make_inputs_require_grads", "make_inputs_require_grad"})


def disable_input_require_grads(model: torch.nn.Module) -> int:
    """Remove every input-embedding ``requires_grad`` hook; return how many.

    ``PreTrainedModel.disable_input_require_grads`` only removes the hooks of
    the *last* ``enable_input_require_grads`` call (each call overwrites the
    handle list), while TRL's ``SFTTrainer.__init__`` and ``Trainer.train()``
    each install a set.  The hooks are therefore removed from the modules
    themselves.
    """
    base = _unwrap(model)
    removed = 0
    for module in base.modules():
        hooks = getattr(module, "_forward_hooks", None)
        if not hooks:
            continue
        for key, fn in list(hooks.items()):
            if getattr(fn, "__name__", "") in _INPUT_GRAD_HOOK_NAMES:
                del hooks[key]
                getattr(module, "_forward_hooks_with_kwargs", {}).pop(key, None)
                getattr(module, "_forward_hooks_always_called", {}).pop(key, None)
                removed += 1
    if hasattr(base, "_require_grads_hooks"):
        base._require_grads_hooks = []
    if hasattr(base, "_require_grads_hook"):
        del base._require_grads_hook
    return removed


class _OffloadedGuard:
    """Forward pre-hook that refuses to run an offloaded module."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.active = False

    def __call__(self, module, args):  # noqa: ANN001 - torch hook API
        if self.active:
            raise RuntimeError(
                f"{self.name} was offloaded to the CPU for text-only training but was "
                "called; image inputs need offload_vision=False"
            )
        return None


class VisionOffloadCallback(TrainerCallback):
    """Move frozen, unused vision modules to the CPU for the duration of training."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.modules = unused_text_only_modules(model)
        self._guards: dict[str, _OffloadedGuard] = {}
        self._handles = []
        self._home: dict[str, torch.device] = {}
        for name, module in self.modules.items():
            guard = _OffloadedGuard(name)
            self._guards[name] = guard
            self._handles.append(module.register_forward_pre_hook(guard))
        self.offloaded_bytes = 0

    def offload(self) -> int:
        moved = 0
        for name, module in self.modules.items():
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} has trainable parameters; refusing to offload it")
            params = list(module.parameters()) + list(module.buffers())
            if not params:
                continue
            self._home.setdefault(name, params[0].device)
            if params[0].device.type == "cpu":
                self._guards[name].active = True
                continue
            moved += sum(t.numel() * t.element_size() for t in params)
            module.to("cpu")
            self._guards[name].active = True
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.offloaded_bytes = max(self.offloaded_bytes, moved)
        if moved:
            LOGGER.info("offloaded %.2f GiB of unused vision modules to the CPU", moved / 2**30)
        return moved

    def restore(self) -> None:
        for name, module in self.modules.items():
            self._guards[name].active = False
            home = self._home.get(name)
            if home is not None and home.type != "cpu":
                module.to(home)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        self.offload()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        self.restore()
        return control


def decoder_layer_calls(text_model: torch.nn.Module) -> int:
    """Decoder-layer calls of one forward: ``prelude + loop * N + coda`` (looped) or all layers."""
    plan = getattr(text_model, "_gemmaloop_plan", None)
    if plan is None:
        return len(text_model.layers)
    n_loops = int(getattr(text_model, "_gemmaloop_n_loops", 1))
    return len(plan.prelude) + len(plan.loop) * n_loops + len(plan.coda)


class AdaptiveCheckpointing:
    """Turn gradient checkpointing off for micro steps with small activations.

    Called by :meth:`LoopSFTTrainer.compute_loss` after ``n_loops`` is set and
    before the forward.  It only ever *disables* checkpointing that the
    trainer enabled (``args.gradient_checkpointing``); it never enables it.
    """

    def __init__(self, text_model: torch.nn.Module, max_token_calls: int) -> None:
        self.text_model = text_model
        self.max_token_calls = int(max_token_calls)
        self.layers = [layer for layer in text_model.layers if hasattr(layer, "gradient_checkpointing")]
        self.skipped = 0
        self.checkpointed = 0

    def __call__(self, inputs: dict, enabled: bool) -> bool:
        """Set the layers' flag for this micro step; return whether it checkpoints."""
        input_ids = inputs.get("input_ids")
        tokens = int(input_ids.numel()) if isinstance(input_ids, torch.Tensor) else None
        use = bool(enabled) and (
            tokens is None or tokens * decoder_layer_calls(self.text_model) > self.max_token_calls
        )
        for layer in self.layers:
            layer.gradient_checkpointing = use
        if enabled:
            if use:
                self.checkpointed += 1
            else:
                self.skipped += 1
        return use


class TrainPerfCallback(TrainerCallback):
    """Apply the switches that ``Trainer.train()`` would otherwise undo."""

    def __init__(self, model: torch.nn.Module, config: TrainPerfConfig) -> None:
        self.model = model
        self.config = config
        self.vision = VisionOffloadCallback(model) if config.offload_vision else None
        self.removed_input_grad_hook: int | None = None  # number of hooks removed

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        if self.config.offload_checkpoints and getattr(args, "gradient_checkpointing", False):
            gc_kwargs = dict(args.gradient_checkpointing_kwargs or {})
            gc_kwargs.pop("offload", None)
            gc_kwargs.pop("every_n_layers", None)
            if gc_kwargs.get("use_reentrant", False):
                # Gemma 4's KV-shared layers read the producer layers' K/V from the
                # ``shared_kv_states`` kwarg; reentrant checkpointing drops the
                # gradient through kwargs, so it is never used here.
                raise ValueError("reentrant gradient checkpointing gives wrong gradients for Gemma 4")
            gc_kwargs["use_reentrant"] = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs,
                offload=True,
            )
        if not self.config.input_grads:
            self.removed_input_grad_hook = disable_input_require_grads(self.model)
        if self.vision is not None:
            self.vision.offload()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        if self.vision is not None:
            self.vision.restore()
        return control


def apply_train_perf(trainer, config: TrainPerfConfig) -> dict:
    """Apply ``config`` to a constructed ``LoopSFTTrainer``; return a summary.

    Must be called after the trainer is built (TRL installs its fused head in
    ``SFTTrainer.__init__``) and before ``train()``.
    """
    summary: dict = {"config": config.to_dict()}
    if config.host_performance_cores:
        summary["host"] = prefer_performance_cores()
    model = trainer.model
    if config.lean_lm_head:
        install_lean_lm_head(
            model,
            token_chunk=config.lm_head_token_chunk,
            vocab_chunk=config.lm_head_vocab_chunk,
            compact=config.lm_head_compact,
        )
    if config.banded_sliding:
        summary["banded_sliding_modules"] = install_banded_sliding_attention(
            _unwrap(model), block=config.banded_block, min_length=config.banded_min_length
        )
    if config.checkpoint_skip_token_calls > 0:
        text_model = getattr(trainer, "loop_text_model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            raise ValueError("checkpoint_skip_token_calls needs the trainer's loop_text_model")
        trainer.checkpoint_policy = AdaptiveCheckpointing(text_model, config.checkpoint_skip_token_calls)
    callback = TrainPerfCallback(model, config)
    trainer.add_callback(callback)
    trainer._gemmaloop_perf_callback = callback
    if callback.vision is not None:
        summary["vision_offload_modules"] = sorted(callback.vision.modules)
    LOGGER.info("training perf: %s", summary)
    return summary


class SleepGuard:
    """Keep Windows awake for the trainer's lifetime (no system setting change).

    ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` is
    process/thread-scoped, needs no admin and is released automatically when
    the process exits (including a crash).  On non-Windows platforms this is a
    no-op.  It cannot stop a forced Windows Update restart.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self, *, platform: str | None = None, ctypes_module=None) -> None:
        self.platform = platform if platform is not None else os.name
        self._ctypes = ctypes_module
        self.active = False
        self.calls: list[int] = []

    def _set(self, flags: int) -> bool:
        if self.platform != "nt":
            return False
        module = self._ctypes
        if module is None:
            import ctypes as module  # type: ignore
        self.calls.append(int(flags))
        try:
            module.windll.kernel32.SetThreadExecutionState(int(flags))
            return True
        except Exception:  # pragma: no cover - defensive
            return False

    def __enter__(self) -> "SleepGuard":
        self.active = self._set(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.active:
            self._set(self.ES_CONTINUOUS)
            self.active = False


def prevent_sleep(
    *, platform: str | None = None, ctypes_module=None
) -> SleepGuard:
    return SleepGuard(platform=platform, ctypes_module=ctypes_module)
