"""Analytic FLOPs for the Gemma 4 text stack.

Every matrix multiplication executed once per generated token is counted:

* attention ``q``/``k``/``v``/``o`` projections (no ``k``/``v`` in KV-consumer
  layers); full-attention layers use ``global_head_dim`` and the
  ``num_global_key_value_heads`` override when ``attention_k_eq_v`` is on,
  sliding layers use ``head_dim``;
* the MLP (double-wide in E2B's KV-consumer layers);
* the per-layer PLE input gate and projection;
* the model-level PLE input projection (``per_layer_model_projection``);
* the LM head.

Loop layers are counted once per pass by :func:`flops_per_token`.  The
following are explicitly **excluded**: the attention score/softmax product
(its cost grows with sequence length), embedding and PLE table lookups,
normalisations, residual/bridge additions and every other element-wise op.
``enable_moe_block`` and ``attention_bias`` are not modelled; the E2B/E4B
configs used here have both disabled.

The result is reported per token.  A problem costs
``generated_tokens * flops_per_token``.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from gemmaloop.model.layer_plan import FULL_ATTENTION, LayerPlan, schedule


class LayerFlops(list[int]):
    """Per-layer FLOPs plus a model-level cost paid once per token.

    The extra attribute keeps the public ``flops_per_token(list[int], ...)``
    signature from the brief while still charging the model-level PLE
    projection and the LM head exactly once, independent of ``n_loops``.
    Plain ``list`` objects are accepted by :func:`flops_per_token` and have no
    model-level constant.
    """

    def __init__(self, values: Iterable[int], constant_flops: int = 0) -> None:
        super().__init__(int(value) for value in values)
        self.constant_flops = int(constant_flops)


def _per_layer_override(
    text_config: Mapping[str, object], layer_idx: int
) -> Mapping[str, object]:
    """Return transformers' ``per_layer_config`` entry for ``layer_idx``.

    transformers 5.22 turns the raw ``global_head_dim`` into
    ``per_layer_config = {"04": {"head_dim": 512}, ...}`` and clears
    ``global_head_dim``.  Keys are the zero-padded layer indices (``"04"``,
    ``"34"``), but accept the unpadded form too.
    """

    per_layer = text_config.get("per_layer_config")
    if not isinstance(per_layer, Mapping):
        return {}
    for key in (str(layer_idx), f"{int(layer_idx):02d}"):
        value = per_layer.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _head_dim(
    text_config: Mapping[str, object], layer_type: str, layer_idx: int
) -> int:
    override = _per_layer_override(text_config, layer_idx)
    if override.get("head_dim") is not None:
        return int(override["head_dim"])  # type: ignore[arg-type]
    if layer_type == FULL_ATTENTION and text_config.get("global_head_dim") is not None:
        return int(text_config["global_head_dim"])  # type: ignore[arg-type]
    return int(text_config["head_dim"])  # type: ignore[arg-type]


def _num_kv_heads(text_config: Mapping[str, object], layer_type: str) -> int:
    if layer_type == FULL_ATTENTION and text_config.get("attention_k_eq_v", True):
        override = text_config.get("num_global_key_value_heads")
        if override is not None:
            return int(override)  # type: ignore[arg-type]
    return int(text_config["num_key_value_heads"])  # type: ignore[arg-type]


def _layer_flops(
    text_config: Mapping[str, object],
    layer_types: Sequence[str],
    layer_idx: int,
) -> int:
    hidden_size = int(text_config["hidden_size"])  # type: ignore[arg-type]
    num_heads = int(text_config["num_attention_heads"])  # type: ignore[arg-type]

    layer_type = layer_types[layer_idx]
    head_dim = _head_dim(text_config, layer_type, layer_idx)
    q_out = num_heads * head_dim

    # q_proj: hidden -> heads * head_dim; o_proj: heads * head_dim -> hidden.
    linear_parameters = hidden_size * q_out + q_out * hidden_size

    num_layers = int(text_config["num_hidden_layers"])  # type: ignore[arg-type]
    num_kv_shared = int(text_config.get("num_kv_shared_layers", 0))  # type: ignore[arg-type]
    first_shared = num_layers - num_kv_shared
    # transformers only enables shared-KV consumers when the first shared layer
    # is a real layer (``first_kv_shared_layer_idx > 0``).
    is_kv_consumer = first_shared > 0 and layer_idx >= first_shared

    if not is_kv_consumer:
        kv_out = _num_kv_heads(text_config, layer_type) * head_dim
        linear_parameters += hidden_size * kv_out  # k_proj
        # ``attention_k_eq_v`` makes full-attention layers reuse k as v.
        if not (layer_type == FULL_ATTENTION and text_config.get("attention_k_eq_v", False)):
            linear_parameters += hidden_size * kv_out  # v_proj

    intermediate_size = int(text_config["intermediate_size"])  # type: ignore[arg-type]
    if is_kv_consumer and text_config.get("use_double_wide_mlp", False):
        intermediate_size *= 2
    # gate_proj + up_proj + down_proj.
    linear_parameters += 3 * hidden_size * intermediate_size

    per_layer_dim = text_config.get("hidden_size_per_layer_input")
    if per_layer_dim:
        per_layer_dim = int(per_layer_dim)  # type: ignore[arg-type]
        # per_layer_input_gate and per_layer_projection.
        linear_parameters += 2 * hidden_size * per_layer_dim

    # One multiply and one add per parameter.
    return 2 * linear_parameters


def model_flops(text_config: Mapping[str, object]) -> int:
    """Return model-level FLOPs paid once per token.

    Counts the model-level PLE input projection and the LM head.  Embedding and
    PLE table lookups are not matmuls and are excluded.
    """
    hidden_size = int(text_config["hidden_size"])  # type: ignore[arg-type]
    num_layers = int(text_config["num_hidden_layers"])  # type: ignore[arg-type]
    per_layer_dim = int(text_config.get("hidden_size_per_layer_input") or 0)  # type: ignore[arg-type]

    flops = 0
    if per_layer_dim:
        # per_layer_model_projection: hidden -> num_layers * per_layer_dim.
        flops += 2 * hidden_size * num_layers * per_layer_dim
    vocab_size = text_config.get("vocab_size")
    if vocab_size is not None:
        # lm_head: hidden -> vocab (even when the embedding is tied).
        flops += 2 * hidden_size * int(vocab_size)  # type: ignore[arg-type]
    return flops


def per_layer_flops(text_config: Mapping[str, object]) -> list[int]:
    """Return the analytic FLOPs per token for each physical layer.

    ``text_config`` is the ``text_config`` mapping from a Gemma 4
    ``config.json`` (or ``Gemma4TextConfig.to_dict()``).  The returned
    :class:`LayerFlops` list also carries the model-level constant so that
    :func:`flops_per_token` includes the PLE projection and LM head once.
    """
    num_layers = int(text_config["num_hidden_layers"])  # type: ignore[arg-type]
    layer_types = list(text_config["layer_types"])  # type: ignore[arg-type]
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries, expected {num_layers}"
        )
    values = [
        _layer_flops(text_config, layer_types, layer_idx)
        for layer_idx in range(num_layers)
    ]
    return LayerFlops(values, model_flops(text_config))


def flops_per_token(
    per_layer_flops: list[int], plan: LayerPlan, n_loops: int | None
) -> int:
    """Return FLOPs per token for ``n_loops`` passes of the looped block.

    ``n_loops`` 0 or ``None`` means the unlooped model and is evaluated with
    the N=1 schedule.  The model-level constant carried by
    :func:`per_layer_flops` is added exactly once.
    """
    if n_loops is None or n_loops < 1:
        n_loops = 1
    if len(per_layer_flops) < plan.num_layers:
        raise ValueError(
            f"per_layer_flops has {len(per_layer_flops)} entries, "
            f"expected at least {plan.num_layers}"
        )
    constant = int(getattr(per_layer_flops, "constant_flops", 0))
    return constant + sum(
        per_layer_flops[layer] for layer in schedule(plan, n_loops)
    )
