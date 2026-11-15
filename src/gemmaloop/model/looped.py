"""Looped forward for a Gemma 4 text model.

This module re-programs the layer loop of
:class:`transformers.models.gemma4.modeling_gemma4.Gemma4TextModel` so that a
contiguous block of layers (``plan.loop``) is executed ``n_loops`` times with
shared weights.  Everything else in the forward is a line-for-line mirror of
``Gemma4TextModel.forward`` in transformers 5.22.1, so with ``n_loops == 1`` the
patched model is bit-identical to the original.

Design notes (see ``docs/superpowers/references/gemma-loop``):

* Per-layer inputs, position embeddings and causal masks are always indexed by
  the **source** layer index ``i``, never by the loop counter.
* Loop 1 is the identity: its input is the prelude output unchanged, so the
  bridge is inert at ``N=1`` and the S1 control (identity) survives.  The
  bridge only feeds loop passes ``t >= 2``; the coda is never bridged.
* The ``loop_only`` activation policy is only installed by ``apply_loops``.
  Unlooped arms (O-H/O-H-ext) therefore run their adapters on the single pass;
  with ``lora_target_layers='loop_block'`` that is exactly the ruling for O-H.
* The looped block lives in the KV-consumer region of Gemma 4 (the layers after
  ``num_hidden_layers - num_kv_shared_layers``): consumers only read
  ``shared_kv_states`` and never update ``past_key_values``, so repeating them
  needs no extra KV cache.
* The bridge is an ``nn.Parameter`` on the text model, not a LoRA adapter, so it
  must be saved separately (:func:`save_loop_state`).
"""

from __future__ import annotations

import functools
import json
import math
import types
from collections import UserDict
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import nn

from safetensors.torch import load_file, save_file
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModelOutputWithPast
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from gemmaloop.model.layer_plan import LayerPlan, make_layer_plan, parse_block, schedule

BRIDGES = ("scalar", "none", "damped", "gch")
ADAPTER_POLICIES = ("always", "loop_only")
LORA_TARGET_LAYERS = ("language", "loop_block", "loop_block_coda")
PLE_MODES = ("shared", "loop_vector")
LOOP_STATE_FILE = "loop_state.safetensors"
LOOP_CONFIG_FILE = "loop_config.json"

# True/False is the activation decision seen by the patched PEFT LoRA layers.
# The default (True) keeps models that were never patched by apply_loops alive.
_ADAPTER_ACTIVE: ContextVar[bool] = ContextVar("gemmaloop_adapter_active", default=True)
_LORA_PATCH_INSTALLED = False


@dataclass
class LoopConfig:
    """Runtime configuration of the looped block."""

    n_loops: int = 1
    bridge: Literal["scalar", "none", "damped", "gch"] = "scalar"
    bridge_init: float = 0.99
    ple_mode: Literal["shared", "loop_vector"] = "shared"
    # Inclusive (start, end) of the looped block.  ``None`` keeps the legacy
    # split (E2B: 15-29).  ``make_layer_plan`` validates it.
    block: tuple[int, int] | None = None
    # Adapter activation policy: ``None`` means the legacy ``"always"``.
    adapter_policy: Literal["always", "loop_only"] | None = None
    # LoRA target layer set: ``None`` means the legacy all-language set.
    lora_target_layers: Literal["language", "loop_block", "loop_block_coda"] | None = None


def gch_bridge_input(
    h_prelude: torch.Tensor,
    h_prev: torch.Tensor,
    g: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Zero-init, norm-matched, per-channel bridge.

    ``x = h_prelude + g * (rms(h_prelude) * RMSNorm(h_prev))`` where RMSNorm has
    no learned weight and both RMS values are per token.  Everything is computed
    in fp32 and rounded to the residual dtype exactly once.  With ``g = 0`` the
    result is exactly ``h_prelude``, so ``N=2`` equals ``N=1`` at initialisation
    while ``d(loss)/dg`` is non-zero (the multiplicative factor is non-zero).
    """
    if g is None:
        raise ValueError("bridge mode 'gch' requires the loop_bridge_g parameter")
    if g.numel() != h_prelude.shape[-1]:
        raise ValueError(
            f"loop_bridge_g has {g.numel()} channels but hidden size is "
            f"{h_prelude.shape[-1]}"
        )
    hp = h_prelude.float()
    prev = h_prev.float()
    hp_rms = hp.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    prev_rms = prev.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    normed_prev = prev / prev_rms
    bridged = hp + g.float() * hp_rms * normed_prev
    return bridged.to(h_prelude.dtype)


def bridge_input(
    h_prelude: torch.Tensor,
    h_prev: torch.Tensor,
    t: int,
    mode: str,
    w: torch.Tensor,
    g: torch.Tensor | None = None,
) -> torch.Tensor:
    """Input of block pass ``t`` (1-based).

    Loop 1 always receives the prelude output unchanged; the bridge only acts
    on loop passes ``t >= 2`` and is **never** applied to the coda (the coda
    reads the final loop output directly):

    * ``scalar``: ``w * h_prelude + (1 - w) * h_prev`` (Closing the Loop);
    * ``none``: ``h_prev``;
    * ``damped``: ``h_prev`` (the damping is applied by :func:`damped_update`
      after the block, see the loop in ``_looped_forward``);
    * ``gch``: :func:`gch_bridge_input`.
    """
    if t == 1:
        return h_prelude
    if mode == "scalar":
        # Sum in fp32 and round once: the 0-dim fp32 w would otherwise be
        # rounded to bf16 (and the product with it), biasing every bridged
        # pass towards the prelude.
        return torch.lerp(h_prev.float(), h_prelude.float(), w.float()).to(
            h_prelude.dtype
        )
    if mode == "none":
        return h_prev
    if mode == "damped":
        return h_prev
    if mode == "gch":
        return gch_bridge_input(h_prelude, h_prev, g)  # type: ignore[arg-type]
    raise ValueError(f"unknown bridge mode: {mode!r} (expected one of {BRIDGES})")


def add_loop_ple(
    per_layer_input: torch.Tensor, loop_ple_row: torch.Tensor
) -> torch.Tensor:
    """Add one loop-PLE row in fp32 and cast back exactly once.

    A bf16 addition would round the small learned vector away before it can
    affect the layer input.
    """
    return (per_layer_input.float() + loop_ple_row.float()).to(
        per_layer_input.dtype
    )


def damped_update(x: torch.Tensor, block_out: torch.Tensor, n_loops: int) -> torch.Tensor:
    """Damped Euler step ``x <- (1 - 1/N) * x + (1/N) * block_out``.

    At ``N=1`` this is exactly ``block_out``.
    """
    if n_loops < 1:
        raise ValueError(f"n_loops must be at least 1, got {n_loops}")
    return (1.0 - 1.0 / n_loops) * x + (1.0 / n_loops) * block_out


@merge_with_config_defaults
@capture_outputs
def _looped_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    use_cache: bool | None = None,
    exits: Sequence[int] | None = None,
    window: int | None = None,
    **kwargs,
) -> Gemma4TextModelOutputWithPast:
    """Patched ``Gemma4TextModel.forward``.

    Mirrors transformers 5.22.1 line by line; the only change is the layer
    iteration, which runs ``prelude + loop * N + coda`` instead of
    ``layers[0:num_hidden_layers]``.

    Multi-exit ("anytime" / dial) mode is enabled by passing ``exits``:
    the forward emits the coda output of every requested block pass and stores
    them in ``outputs.exit_hidden_states`` (a ``dict[int, Tensor]``).
    ``window`` is the number of *last* block passes whose gradients are kept
    (``None`` keeps all of them).  In dial mode the prelude and every pass
    outside the window run under ``torch.no_grad()``; the window starts from a
    detached state, so the gradient is exact for that window (truncated BPTT).
    The legacy forward (``exits=None``) is untouched, bit for bit.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None and per_layer_inputs is not None:
        raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self.hidden_size_per_layer_input:
        if per_layer_inputs is None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = {}
    for layer_type in self.unique_layer_types:
        position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

    # Initialize as empty dict, or reuse past shared states (UserDict for fsdp2).
    shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

    plan: LayerPlan = self._gemmaloop_plan
    cfg: LoopConfig = self._gemmaloop_loop_config
    n_loops: int = self._gemmaloop_n_loops
    layer_types = self.config.layer_types
    loop_ple = getattr(self, "loop_ple", None)
    if cfg.ple_mode == "loop_vector" and loop_ple is None:
        raise RuntimeError("ple_mode='loop_vector' but the model has no loop_ple parameter")

    def run_layer(
        i: int, x: torch.Tensor, t: int, pass_count: int | None = None
    ) -> torch.Tensor:
        """Run source layer ``i`` on ``x`` during block pass ``t`` (0 = not looped).

        ``pass_count`` is the loop count the adapter policy sees.  Loop layers
        use the forward's own ``n_loops``; the coda of a multi-exit forward
        uses the **exit's** pass count, so exit 1 stays adapter-free and exit
        t matches the fixed-N=t forward even for ``loop_block_coda``.
        """
        per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        if (
            per_layer_input is not None
            and cfg.ple_mode == "loop_vector"
            and t >= 2
        ):
            per_layer_input = add_loop_ple(per_layer_input, loop_ple[t - 1])

        return self.layers[i](
            x,
            per_layer_input,
            shared_kv_states=shared_kv_states,
            position_embeddings=position_embeddings[layer_types[i]],
            attention_mask=causal_mask_mapping[layer_types[i]],
            position_ids=position_ids,
            past_key_values=past_key_values,
            _gemmaloop_pass=t,
            _gemmaloop_n_loops=n_loops if pass_count is None else pass_count,
            **kwargs,
        )

    def run_coda(h_in: torch.Tensor, exit_n: int) -> torch.Tensor:
        """Coda + final norm on ``h_in`` for an exit with ``exit_n`` block passes."""
        hs = h_in
        for i in plan.coda:
            hs = run_layer(i, hs, 0, pass_count=exit_n)
        return self.norm(hs)

    if exits is None:
        for i in plan.prelude:
            hidden_states = run_layer(i, hidden_states, 0)

        h_prelude = hidden_states
        h = h_prelude
        w = torch.sigmoid(self.loop_bridge_logit)
        g = getattr(self, "loop_bridge_g", None)
        for t in range(1, n_loops + 1):
            x = (
                h
                if cfg.bridge == "damped"
                else bridge_input(h_prelude, h, t, cfg.bridge, w, g)
            )
            for i in plan.loop:
                x = run_layer(i, x, t)
            h = damped_update(h, x, n_loops) if cfg.bridge == "damped" else x
        hidden_states = h

        # The coda is never bridged: it reads the last loop output directly (the
        # bridge only produces inputs for loop passes t >= 2).
        for i in plan.coda:
            hidden_states = run_layer(i, hidden_states, 0)

        hidden_states = self.norm(hidden_states)

        return Gemma4TextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
        )

    # --- multi-exit ("dial") forward --------------------------------------
    requested = tuple(sorted({int(t) for t in exits}))
    if not requested:
        raise ValueError("exits must not be empty")
    if requested[0] < 1 or requested[-1] > n_loops:
        raise ValueError(
            f"exits must lie in 1..{n_loops}, got {requested}"
        )
    if window is not None:
        window = int(window)
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
    if cfg.bridge == "damped":
        # The damped update depends on the *total* loop count, so pass t of a
        # T-run is not the same function as the fixed-N=t forward.  The dial
        # is defined on the time-invariant bridge modes only.
        raise ValueError("multi-exit forwards require a time-invariant bridge (not 'damped')")

    window_start = 2 if window is None or window >= n_loops else n_loops - window + 1

    w = torch.sigmoid(self.loop_bridge_logit)
    g = getattr(self, "loop_bridge_g", None)
    exit_hidden_states: dict[int, torch.Tensor] = {}
    block_states: dict[int, torch.Tensor] = {}

    def block_pass(t: int, h_prev: torch.Tensor) -> torch.Tensor:
        x = (
            h_prev
            if cfg.bridge == "damped"
            else bridge_input(h_prelude, h_prev, t, cfg.bridge, w, g)
        )
        for i in plan.loop:
            x = run_layer(i, x, t)
        return damped_update(h_prev, x, n_loops) if cfg.bridge == "damped" else x

    # The frozen prefix (prelude + pass 1) never holds a trainable parameter
    # under loop_block + loop_only, so it runs without autograd.
    with torch.no_grad():
        for i in plan.prelude:
            hidden_states = run_layer(i, hidden_states, 0)
        h_prelude = hidden_states
        h = block_pass(1, h_prelude)
        block_states[1] = h
        if 1 in requested:
            exit_hidden_states[1] = run_coda(h, 1)
        for t in range(2, max(2, window_start)):
            h = block_pass(t, h)
            block_states[t] = h
            if t in requested:
                exit_hidden_states[t] = run_coda(h, t)
        h = h.detach()

    for t in range(window_start, n_loops + 1):
        h = block_pass(t, h)
        block_states[t] = h
        if t in requested:
            exit_hidden_states[t] = run_coda(h, t)

    hidden_states = exit_hidden_states[requested[-1]]

    outputs = Gemma4TextModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
    )
    # Keep the per-exit states out of the dataclass fields: the output class is
    # a fixed transformers schema, and the caller only needs this attribute.
    outputs.exit_hidden_states = exit_hidden_states
    outputs.exit_block_states = block_states
    return outputs


def _install_lora_policy_patch() -> None:
    """Make PEFT LoRA layers honour the per-forward pass context.

    The patch is process-wide but inert (adapters active) unless a looped
    forward sets the context.  It never copies weights or toggles ``merged``;
    it only routes the existing ``disable_adapters`` fast path for the duration
    of one module call.
    """
    global _LORA_PATCH_INSTALLED
    if _LORA_PATCH_INSTALLED:
        return
    try:
        from peft.tuners.lora.layer import Embedding, Linear
    except Exception:  # pragma: no cover - peft is a hard dependency
        return

    def _wrap(original):
        @functools.wraps(original)
        def forward(self, x, *args, **kwargs):
            if _ADAPTER_ACTIVE.get() is False:
                previous = getattr(self, "_disable_adapters", False)
                self._disable_adapters = True
                try:
                    return original(self, x, *args, **kwargs)
                finally:
                    self._disable_adapters = previous
            return original(self, x, *args, **kwargs)

        return forward

    Linear.forward = _wrap(Linear.forward)
    Embedding.forward = _wrap(Embedding.forward)
    _LORA_PATCH_INSTALLED = True


def _install_adapter_policy_hooks(text_model) -> None:
    """Set the LoRA activation context for every decoder-layer call.

    The pass index travel as a keyword argument to the layer, so
    gradient-checkpointing recomputation (which replays the same captured
    kwargs) sees exactly the pass that produced the saved activations.
    """
    for handle in getattr(text_model, "_gemmaloop_adapter_hooks", ()):
        handle.remove()

    def _pre(module, args, kwargs):
        t = int(kwargs.pop("_gemmaloop_pass", 0))
        n = int(kwargs.pop("_gemmaloop_n_loops", 1))
        cfg = getattr(text_model, "_gemmaloop_loop_config", None)
        plan = getattr(text_model, "_gemmaloop_plan", None)
        policy = getattr(cfg, "adapter_policy", None) or "always"
        if policy == "loop_only" and plan is not None:
            layer_idx = int(getattr(module, "layer_idx", -1))
            if layer_idx in plan.loop:
                active = t >= 2
            elif layer_idx in plan.coda:
                active = n >= 2
            else:
                active = False
        else:
            active = True
        _ADAPTER_ACTIVE.set(active)
        return args, kwargs

    def _post(module, args, output):
        _ADAPTER_ACTIVE.set(True)
        return output

    num_layers = int(text_model.config.num_hidden_layers)
    handles = []
    for layer in text_model.layers[:num_layers]:
        handles.append(layer.register_forward_pre_hook(_pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(_post))
    text_model._gemmaloop_adapter_hooks = handles


def _validate_loop_config(cfg: LoopConfig) -> None:
    if cfg.n_loops < 1:
        raise ValueError(f"n_loops must be at least 1, got {cfg.n_loops}")
    if cfg.bridge not in BRIDGES:
        raise ValueError(f"bridge must be one of {BRIDGES}, got {cfg.bridge!r}")
    if cfg.ple_mode not in PLE_MODES:
        raise ValueError(f"ple_mode must be one of {PLE_MODES}, got {cfg.ple_mode!r}")
    if cfg.adapter_policy is not None and cfg.adapter_policy not in ADAPTER_POLICIES:
        raise ValueError(
            f"adapter_policy must be one of {ADAPTER_POLICIES}, "
            f"got {cfg.adapter_policy!r}"
        )
    if cfg.lora_target_layers is not None and cfg.lora_target_layers not in LORA_TARGET_LAYERS:
        raise ValueError(
            f"lora_target_layers must be one of {LORA_TARGET_LAYERS}, "
            f"got {cfg.lora_target_layers!r}"
        )
    if cfg.block is not None:
        parse_block(cfg.block)
    if not 0.0 < cfg.bridge_init < 1.0:
        raise ValueError(f"bridge_init must be in (0, 1), got {cfg.bridge_init}")


def _validate_plan(text_model, plan: LayerPlan) -> None:
    num_layers = text_model.config.num_hidden_layers
    if plan.num_layers != num_layers:
        raise ValueError(
            f"plan covers {plan.num_layers} layers but the model has {num_layers}"
        )
    if len(text_model.layers) < num_layers:
        raise ValueError(
            f"model has {len(text_model.layers)} layers, expected at least {num_layers}"
        )
    # A single pass must visit every source layer in its original order.
    expected = list(range(num_layers))
    if schedule(plan, 1) != expected:
        raise ValueError(
            "prelude + loop + coda must partition 0..num_layers-1 contiguously; "
            f"got {schedule(plan, 1)}"
        )

    # The looped block must live entirely in the KV-consumer region: repeating a
    # producer layer would call `past_key_values.update` twice for the same
    # layer_idx (DynamicCache concatenates), desynchronising the cache from
    # position_ids/masks and overwriting shared_kv_states.
    num_kv_shared_layers = text_model.config.num_kv_shared_layers
    first_shared = num_layers - num_kv_shared_layers
    if plan.first_shared != first_shared:
        raise ValueError(
            f"plan.first_shared={plan.first_shared} does not match the model's "
            f"KV-sharing boundary (num_hidden_layers - num_kv_shared_layers = "
            f"{num_layers} - {num_kv_shared_layers} = {first_shared})"
        )
    if plan.prelude != range(0, plan.loop.start):
        raise ValueError(
            f"plan.prelude must be range(0, {plan.loop.start}); got {plan.prelude}"
        )
    if plan.loop.start < first_shared:
        raise ValueError(
            f"plan.loop starts at layer {plan.loop.start}, inside the KV-producer "
            f"region [0, {first_shared}); repeating producer layers would update "
            "past_key_values twice"
        )
    if len(plan.loop) < 1:
        raise ValueError("plan.loop must contain at least one layer")
    if len(plan.coda) < 1:
        raise ValueError(
            "plan.coda must be non-empty; the looped block cannot reach the "
            "last layer"
        )
    if plan.coda != range(plan.loop.stop, num_layers):
        raise ValueError(
            f"plan.coda must be range({plan.loop.stop}, {num_layers}); "
            f"got {plan.coda}"
        )


def apply_loops(text_model, plan: LayerPlan, cfg: LoopConfig) -> "Gemma4TextModel":
    """Patch ``text_model`` in place so its layer loop follows ``plan``.

    Registers ``loop_bridge_logit`` (scalar ``nn.Parameter``, initialised to
    ``log(bridge_init / (1 - bridge_init))``), for ``bridge="gch"`` a
    zero-initialised ``loop_bridge_g`` (``[hidden_size]``, fp32), and, for
    ``ple_mode="loop_vector"``, ``loop_ple`` (``[4,
    hidden_size_per_layer_input]``, zeros).  Weights are never copied: the same
    layer modules are re-run.
    """
    _validate_loop_config(cfg)
    if cfg.block is not None:
        start, end = parse_block(cfg.block)
        if plan.loop != range(start, end + 1):
            # ``LoopConfig.block`` is the requested geometry: deriving the plan
            # here keeps a bare ``apply_loops(model, default_plan, cfg)`` call
            # consistent with the config instead of silently ignoring it.
            plan = make_layer_plan(
                text_model.config.to_dict(), block=(start, end)
            )
    _validate_plan(text_model, plan)

    if cfg.ple_mode == "loop_vector":
        if not text_model.hidden_size_per_layer_input:
            raise ValueError("ple_mode='loop_vector' requires per-layer embeddings (PLE)")

    device = next(text_model.parameters()).device

    if not hasattr(text_model, "_gemmaloop_original_forward"):
        text_model._gemmaloop_original_forward = text_model.forward

    text_model._gemmaloop_plan = plan
    text_model._gemmaloop_loop_config = cfg

    if not hasattr(text_model, "loop_bridge_logit"):
        logit = math.log(cfg.bridge_init / (1.0 - cfg.bridge_init))
        text_model.loop_bridge_logit = nn.Parameter(
            torch.tensor(logit, dtype=torch.float32, device=device)
        )

    if cfg.bridge == "gch" and not hasattr(text_model, "loop_bridge_g"):
        hidden_size = int(text_model.config.hidden_size)
        text_model.loop_bridge_g = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32, device=device)
        )

    if cfg.ple_mode == "loop_vector" and not hasattr(text_model, "loop_ple"):
        text_model.loop_ple = nn.Parameter(
            torch.zeros(
                4,
                text_model.hidden_size_per_layer_input,
                dtype=torch.float32,
                device=device,
            )
        )

    _install_lora_policy_patch()
    _install_adapter_policy_hooks(text_model)
    text_model.forward = types.MethodType(_looped_forward, text_model)
    set_n_loops(text_model, cfg.n_loops)
    return text_model


def set_n_loops(text_model, n: int) -> None:
    """Set the number of block passes used by the patched forward."""
    cfg: LoopConfig | None = getattr(text_model, "_gemmaloop_loop_config", None)
    if cfg is None or getattr(text_model, "_gemmaloop_plan", None) is None:
        raise ValueError("apply_loops() must be called before set_n_loops()")
    n = int(n)
    if n < 1:
        raise ValueError(f"n_loops must be at least 1, got {n}")
    if cfg.ple_mode == "loop_vector" and n > text_model.loop_ple.shape[0]:
        raise ValueError(
            f"ple_mode='loop_vector' supports up to {text_model.loop_ple.shape[0]} "
            f"loops (loop_ple rows), got {n}"
        )
    text_model._gemmaloop_n_loops = n
    cfg.n_loops = n


def saved_adapter_policy(directory: str | Path) -> str | None:
    """Return the adapter-policy recorded beside an adapter, if any."""
    path = Path(directory) / LOOP_CONFIG_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupt dir
        raise ValueError(f"cannot read loop config from {path}: {exc}") from exc
    return payload.get("adapter_policy")


def assert_adapter_merge_allowed(directory: str | Path) -> None:
    """Refuse to merge a ``loop_only`` adapter into the base weights.

    ``loop_only`` keeps LoRA inactive on pass 1 and for unlooped N=1 forwards.
    Merging bakes the adapter into the weights permanently, so N=1 would become
    ``base + LoRA`` and the no-harm invariant would be silently lost.
    """
    policy = saved_adapter_policy(directory)
    if policy == "loop_only":
        raise ValueError(
            f"adapter {str(directory)!r} was trained with "
            "adapter_policy='loop_only' and cannot be merged into the base "
            "weights; evaluate it as a live PEFT adapter (single adapter or "
            "the last adapter in a chain) or train it with adapter_policy="
            "'always' if merging is required"
        )


def _plan_to_json(plan: LayerPlan) -> dict:
    return {
        "num_layers": plan.num_layers,
        "period": plan.period,
        "first_shared": plan.first_shared,
        "full_layers": list(plan.full_layers),
        "prelude": [plan.prelude.start, plan.prelude.stop],
        "loop": [plan.loop.start, plan.loop.stop],
        "coda": [plan.coda.start, plan.coda.stop],
    }


def save_loop_state(text_model, out_dir: Path) -> None:
    """Save the bridge (and per-loop PLE) plus the loop config to ``out_dir``.

    Writes ``loop_state.safetensors`` (``loop_bridge_logit`` and ``loop_ple``
    when present) and ``loop_config.json`` (bridge, ``bridge_init``,
    ``ple_mode``, the plan ranges, and the diagnostic
    ``n_loops_last_sampled``).  The bridge is not a LoRA parameter, so it would
    otherwise be lost when an adapter is saved.
    """
    cfg: LoopConfig | None = getattr(text_model, "_gemmaloop_loop_config", None)
    plan: LayerPlan | None = getattr(text_model, "_gemmaloop_plan", None)
    if cfg is None or plan is None:
        raise ValueError("apply_loops() must be called before save_loop_state()")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {"loop_bridge_logit": text_model.loop_bridge_logit.detach().cpu().clone()}
    if hasattr(text_model, "loop_bridge_g"):
        state["loop_bridge_g"] = text_model.loop_bridge_g.detach().cpu().clone()
    if hasattr(text_model, "loop_ple"):
        state["loop_ple"] = text_model.loop_ple.detach().cpu().clone()
    save_file(state, out_dir / LOOP_STATE_FILE)

    payload = {
        # The training sampler's last draw is a diagnostic only.  The N used
        # for evaluation always comes from the evaluation spec (see load.py).
        "n_loops_last_sampled": cfg.n_loops,
        "bridge": cfg.bridge,
        "bridge_init": cfg.bridge_init,
        "ple_mode": cfg.ple_mode,
        "block": None if cfg.block is None else [int(cfg.block[0]), int(cfg.block[1])],
        "adapter_policy": cfg.adapter_policy or "always",
        "lora_target_layers": cfg.lora_target_layers or "language",
        "plan": _plan_to_json(plan),
    }
    (out_dir / LOOP_CONFIG_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_loop_state(text_model, in_dir: Path) -> LoopConfig:
    """Restore the loop state written by :func:`save_loop_state`.

    Returns the restored :class:`LoopConfig`.  ``n_loops`` in the saved
    ``loop_config.json`` is only the training sampler's last draw; it is
    **ignored** here so that loading an adapter cannot override the evaluation
    N selected by the caller.  ``apply_loops()`` must have been called first.
    """
    cfg: LoopConfig | None = getattr(text_model, "_gemmaloop_loop_config", None)
    plan: LayerPlan | None = getattr(text_model, "_gemmaloop_plan", None)
    if cfg is None or plan is None:
        raise ValueError("apply_loops() must be called before load_loop_state()")

    in_dir = Path(in_dir)
    payload = json.loads((in_dir / LOOP_CONFIG_FILE).read_text(encoding="utf-8"))
    saved_block = payload.get("block")
    if saved_block is None and isinstance(payload.get("plan"), dict):
        loop_range = payload["plan"].get("loop")
        if isinstance(loop_range, (list, tuple)) and len(loop_range) == 2:
            saved_block = (int(loop_range[0]), int(loop_range[1]) - 1)
    loaded_cfg = LoopConfig(
        # Keep the N owned by the evaluation spec; the file only carries the
        # transient training sample (n_loops_last_sampled / legacy n_loops).
        n_loops=int(getattr(text_model, "_gemmaloop_n_loops", cfg.n_loops)),
        bridge=payload["bridge"],
        bridge_init=float(payload["bridge_init"]),
        ple_mode=payload["ple_mode"],
        block=None if saved_block is None else (int(saved_block[0]), int(saved_block[1])),
        adapter_policy=payload.get("adapter_policy"),
        lora_target_layers=payload.get("lora_target_layers"),
    )
    _validate_loop_config(loaded_cfg)
    if loaded_cfg.ple_mode == "loop_vector" and not hasattr(text_model, "loop_ple"):
        raise ValueError(
            "loop_config.json requests ple_mode='loop_vector' but the model was "
            "patched without a loop_ple parameter"
        )

    saved_plan = payload.get("plan")
    if saved_plan is not None and saved_plan != _plan_to_json(plan):
        raise ValueError(
            "loop_config.json was saved for a different layer plan: "
            f"{saved_plan} != {_plan_to_json(plan)}"
        )

    state = load_file(in_dir / LOOP_STATE_FILE)
    with torch.no_grad():
        if "loop_bridge_logit" in state:
            text_model.loop_bridge_logit.copy_(
                state["loop_bridge_logit"].to(
                    device=text_model.loop_bridge_logit.device,
                    dtype=text_model.loop_bridge_logit.dtype,
                )
            )
        if "loop_bridge_g" in state:
            if not hasattr(text_model, "loop_bridge_g"):
                raise ValueError(
                    "loop_state.safetensors contains loop_bridge_g but the "
                    "model was patched with a bridge other than 'gch'"
                )
            text_model.loop_bridge_g.copy_(
                state["loop_bridge_g"].to(
                    device=text_model.loop_bridge_g.device,
                    dtype=text_model.loop_bridge_g.dtype,
                )
            )
        if "loop_ple" in state:
            if not hasattr(text_model, "loop_ple"):
                raise ValueError(
                    "loop_state.safetensors contains loop_ple but the model was "
                    "patched without ple_mode='loop_vector'"
                )
            text_model.loop_ple.copy_(
                state["loop_ple"].to(
                    device=text_model.loop_ple.device,
                    dtype=text_model.loop_ple.dtype,
                )
            )

    text_model._gemmaloop_loop_config = loaded_cfg
    # N is deliberately not touched: the caller (load_model/serve/trainer)
    # decides it from the evaluation spec or the training sampler.
    return loaded_cfg
