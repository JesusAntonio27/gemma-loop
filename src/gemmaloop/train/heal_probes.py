"""Inference-only probes used by heal monitoring.

The probes answer three different questions:

* **invariant** -- is multi-exit exit 1 still bit-for-bit the adapter-off
  parent ``O``?
* **health** -- how far has every operating point drifted from ``O``, from the
  teacher, and what is its calibrated NLL?  (Descriptive only.)
* **sanity** -- does greedy generation still close the thinking channel and
  produce an answer without n-gram loops?

Everything here observes: it never writes to the model, never consumes the
global RNG (greedy generation and ``torch.inference_mode``), and restores the
model's previous ``n_loops`` and ``training`` flag in ``finally`` blocks.  The
module has no dependency on the trainer so it can be exercised on tiny CPU
models.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from gemmaloop.train.devices import model_compute_device
from gemmaloop.train.dial_objective import TeacherTopK, chunked_teacher_kl

__all__ = [
    "DEFAULT_ANCHOR_NS",
    "DEFAULT_EVAL_NS",
    "DEFAULT_GENERATION_NS",
    "DEFAULT_NLL_NS",
    "DEFAULT_PROBE_CONFIG",
    "ProbeConfig",
    "ProbeRunner",
    "chunked_kl_hidden",
    "chunked_lse",
    "chunked_nll_hidden",
    "chunked_nll_temperatures",
    "exit1_bit_exact",
    "generation_offload_devices",
    "generation_sanity",
    "merge_invariants",
    "nll_from_hidden",
    "nll_from_hidden_temperatures",
    "teacher_kl_from_batch",
    "text_model_and_head",
]

DEFAULT_ANCHOR_NS = (2, 3, 4, 6, 8)
DEFAULT_EVAL_NS = (1, 2, 3, 4, 6, 8)
DEFAULT_NLL_NS = (1, 2, 3, 4)
DEFAULT_GENERATION_NS = (1, 2, 4)

_TINY = 1e-12
_THOUGHT_OPEN = "<|channel>thought"
_THOUGHT_CLOSE = "<channel|>"
_TURN_TOKENS = ("<turn|>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>")


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def text_model_and_head(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.Tensor, float | None]:
    """Return ``(text_model, lm_head_weight, softcap)`` for any wrapper."""
    base = _base_model(model)
    inner = getattr(base, "model", None)
    text_model = base if inner is None else getattr(inner, "language_model", inner)
    if text_model is None or not hasattr(text_model, "layers"):
        raise ValueError("cannot locate the Gemma 4 text model inside the wrapper")
    head = getattr(base, "get_output_embeddings", lambda: None)()
    weight = getattr(head, "weight", None)
    if weight is None:
        weight = getattr(getattr(text_model, "embed_tokens", None), "weight", None)
    if weight is None:
        raise ValueError("cannot locate the LM-head weight")
    config = getattr(text_model, "config", None)
    return text_model, weight, getattr(config, "final_logit_softcapping", None)


def _set_n_loops(text_model: torch.nn.Module, n: int) -> None:
    from gemmaloop.model.looped import set_n_loops

    set_n_loops(text_model, int(n))


def _loop_state(text_model: torch.nn.Module) -> tuple[bool, int]:
    looped = getattr(text_model, "_gemmaloop_loop_config", None) is not None
    return looped, int(getattr(text_model, "_gemmaloop_n_loops", 1))


def _adapter_context(model: torch.nn.Module, adapters_off: bool):
    if not adapters_off:
        return nullcontext()
    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


@contextmanager
def _eval_mode(model: torch.nn.Module):
    """Temporarily set ``model`` to eval and restore every module's own flag.

    A plain ``model.train(was_training)`` would switch submodules that were
    deliberately left in eval (the vision tower) back to train mode.  Probes
    must be perfectly observation-only, so snapshot each module.
    """
    states = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def _attention_mask(batch: Mapping[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
    mask = batch.get("attention_mask")
    if isinstance(mask, torch.Tensor):
        return mask
    return torch.ones_like(input_ids)


def _generation_offload_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Frozen text-unused modules (vision/audio) that training offloads."""

    base = _base_model(model)
    containers = (base, getattr(base, "model", None))
    seen: set[int] = set()
    modules: list[tuple[str, torch.nn.Module]] = []
    for container in containers:
        if not isinstance(container, torch.nn.Module):
            continue
        for name in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
            module = getattr(container, name, None)
            if isinstance(module, torch.nn.Module) and id(module) not in seen:
                seen.add(id(module))
                modules.append((name, module))
    return modules


def _module_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    return list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))


def generation_offload_devices(
    model: torch.nn.Module | None,
) -> dict[str, tuple[str, ...]]:
    """Device of every tensor in the modules generation temporarily moves.

    The smoke records this before and after a generation phase to prove the
    ``finally`` restore actually ran.
    """

    if model is None:
        return {}
    return {
        name: tuple(str(tensor.device) for tensor in _module_tensors(module))
        for name, module in _generation_offload_modules(model)
    }


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device == device:
        return
    if tensor.device.type == "meta":
        data = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
    else:
        data = tensor.to(device)
    tensor.data = data


@contextmanager
def _generation_device(model: torch.nn.Module):
    """Keep every module touched by ``generate`` on the text compute device.

    ``PreTrainedModel.device`` is the device of the first registered parameter.
    During training the frozen vision tower is offloaded to the CPU and is
    registered before the language model, so ``model.device`` is CPU while the
    text stack runs on CUDA.  HF ``generate`` moves every input to
    ``model.device`` on each decode step, which then hands CPU ids to the CUDA
    embeddings.  Temporarily moving the offloaded modules to the compute device
    makes ``model.device`` correct for the duration of generation; the exact
    per-tensor devices are restored in ``finally``.
    """

    compute = model_compute_device(model)
    moved_tensors: list[tuple[torch.Tensor, torch.device]] = []
    moved_modules: list[tuple[torch.nn.Module, torch.device]] = []
    try:
        # Record every tensor's home before moving it.  The loop is inside the
        # ``try`` so an OOM or any other failure part-way through still
        # restores everything already moved instead of pinning it on the GPU
        # for the rest of the arm.
        for _name, module in _generation_offload_modules(model):
            tensors = _module_tensors(module)
            if not tensors or all(tensor.device == compute for tensor in tensors):
                continue
            if any(tensor.device.type == "meta" for tensor in tensors):
                # ``Parameter.data`` cannot change a meta tensor's type; use
                # the module API.  Production offload is CPU, but the CPU
                # regression test uses meta as the "different device" stand-in.
                moved_modules.append((module, tensors[0].device))
                module.to_empty(device=compute)
                continue
            states = [(tensor, tensor.device) for tensor in tensors]
            moved_tensors.extend(states)
            for tensor in tensors:
                _move_tensor(tensor, compute)
        first = next(model.parameters(), None)
        if first is not None and first.device != compute:
            raise RuntimeError(
                "generation inputs are on "
                f"{compute} but the model's first parameter is on "
                f"{first.device}; cannot keep HF generate on one device"
            )
        yield compute
    finally:
        for module, device in reversed(moved_modules):
            module.to_empty(device=device)
        for tensor, device in reversed(moved_tensors):
            _move_tensor(tensor, device)


@torch.inference_mode()
def _forward_hidden(
    model: torch.nn.Module,
    text_model: torch.nn.Module,
    batch: Mapping[str, Any],
    n_loops: int,
    *,
    adapters_off: bool = False,
    exits: Sequence[int] | None = None,
) -> torch.Tensor:
    """Run one forward and return the coda/norm hidden state (detached, CPU-kept)."""
    looped, previous_n = _loop_state(text_model)
    input_ids = batch["input_ids"]
    attention_mask = _attention_mask(batch, input_ids)
    try:
        with _eval_mode(model):
            if looped:
                _set_n_loops(text_model, int(n_loops))
            with _adapter_context(model, adapters_off):
                if exits is not None and looped:
                    outputs = text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        exits=tuple(int(t) for t in exits),
                    )
                    hidden = outputs.exit_hidden_states[int(exits[-1])]
                else:
                    outputs = text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    hidden = outputs.last_hidden_state
        return hidden.detach()
    finally:
        if looped:
            _set_n_loops(text_model, previous_n)


# ---------------------------------------------------------------------------
# Chunked distributions (never materialise [tokens, vocab])
# ---------------------------------------------------------------------------


def _softcap(z: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return z
    return torch.tanh(z / float(softcap)) * float(softcap)


@torch.inference_mode()
def chunked_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> torch.Tensor:
    """Full-vocabulary log-sum-exp of ``softcap(hidden @ weight.T)`` per token."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be [tokens, hidden]")
    if token_chunk < 1 or vocab_chunk < 1:
        raise ValueError("token_chunk and vocab_chunk must be positive")
    hidden = hidden.float()
    vocab = int(weight.shape[0])
    out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    for t0 in range(0, hidden.shape[0], token_chunk):
        t1 = min(t0 + token_chunk, hidden.shape[0])
        h = hidden[t0:t1]
        running_max = torch.full((t1 - t0,), float("-inf"), device=h.device)
        sum_exp = torch.zeros((t1 - t0,), device=h.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            z = _softcap(h @ weight[v0:v1].float().t(), softcap)
            new_max = torch.maximum(running_max, z.max(dim=1).values)
            sum_exp = sum_exp * torch.exp(running_max - new_max) + torch.exp(
                z - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max
        out[t0:t1] = running_max + torch.log(sum_exp)
    return out


@torch.inference_mode()
def chunked_nll_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float | None = None,
    temperature: float = 1.0,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> torch.Tensor:
    """Per-token NLL of ``softmax(softcap(hidden @ weight.T) / T)[target]``."""
    if hidden.shape[0] != targets.numel():
        raise ValueError("hidden and targets must agree on the token count")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    hidden = hidden.float()
    targets = targets.long()
    vocab = int(weight.shape[0])
    nll = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    for t0 in range(0, hidden.shape[0], token_chunk):
        t1 = min(t0 + token_chunk, hidden.shape[0])
        h = hidden[t0:t1]
        tgt = targets[t0:t1]
        running_max = torch.full((t1 - t0,), float("-inf"), device=h.device)
        sum_exp = torch.zeros((t1 - t0,), device=h.device)
        target_logit = torch.zeros((t1 - t0,), device=h.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            z = _softcap(h @ weight[v0:v1].float().t(), softcap)
            if float(temperature) != 1.0:
                z = z / float(temperature)
            local = tgt - v0
            in_tile = (local >= 0) & (local < (v1 - v0))
            local_index = local.clamp(0, (v1 - v0) - 1).unsqueeze(1)
            gathered = z.gather(1, local_index).squeeze(1)
            target_logit = torch.where(in_tile, gathered, target_logit)
            new_max = torch.maximum(running_max, z.max(dim=1).values)
            sum_exp = sum_exp * torch.exp(running_max - new_max) + torch.exp(
                z - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max
        nll[t0:t1] = (running_max + torch.log(sum_exp)) - target_logit
    return nll


@torch.inference_mode()
def chunked_nll_temperatures(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    temperatures: Sequence[float],
    *,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> dict[float, torch.Tensor]:
    """Per-token NLL for several temperatures from one logits pass per chunk.

    For every vocabulary tile the raw ``softcap(hidden @ weight.T)`` is
    computed once and each temperature only rescales that tile.  This is the
    same arithmetic, in the same order, as one ``chunked_nll_hidden`` call per
    temperature, so the results are bit-identical (verified by
    ``test_multi_temperature_nll_is_bit_identical``).
    """
    if hidden.shape[0] != targets.numel():
        raise ValueError("hidden and targets must agree on the token count")
    # A duplicated T would update the same per-temperature accumulator twice.
    # Deduplicate explicitly (order preserved) so ``(1.0, 1.0)`` is exactly
    # the single-temperature result.
    requested = tuple(dict.fromkeys(float(value) for value in temperatures))
    if not requested:
        return {}
    if any(value <= 0 for value in requested):
        raise ValueError("temperature must be positive")
    hidden = hidden.float()
    targets = targets.long()
    vocab = int(weight.shape[0])
    out = {
        value: torch.empty(
            hidden.shape[0], dtype=torch.float32, device=hidden.device
        )
        for value in requested
    }
    for t0 in range(0, hidden.shape[0], token_chunk):
        t1 = min(t0 + token_chunk, hidden.shape[0])
        h = hidden[t0:t1]
        tgt = targets[t0:t1]
        running_max = {
            value: torch.full((t1 - t0,), float("-inf"), device=h.device)
            for value in requested
        }
        sum_exp = {
            value: torch.zeros((t1 - t0,), device=h.device)
            for value in requested
        }
        target_logit = {
            value: torch.zeros((t1 - t0,), device=h.device)
            for value in requested
        }
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            z = _softcap(h @ weight[v0:v1].float().t(), softcap)
            local = tgt - v0
            in_tile = (local >= 0) & (local < (v1 - v0))
            local_index = local.clamp(0, (v1 - v0) - 1).unsqueeze(1)
            gathered = z.gather(1, local_index).squeeze(1)
            for value in requested:
                if value != 1.0:
                    scaled = z / value
                    gathered_value = gathered / value
                else:
                    scaled = z
                    gathered_value = gathered
                target_logit[value] = torch.where(
                    in_tile, gathered_value, target_logit[value]
                )
                new_max = torch.maximum(
                    running_max[value], scaled.max(dim=1).values
                )
                sum_exp[value] = sum_exp[value] * torch.exp(
                    running_max[value] - new_max
                ) + torch.exp(scaled - new_max.unsqueeze(1)).sum(dim=1)
                running_max[value] = new_max
        for value in requested:
            out[value][t0:t1] = (
                running_max[value] + torch.log(sum_exp[value])
            ) - target_logit[value]
    return out


@torch.inference_mode()
def chunked_kl_hidden(
    hidden_p: torch.Tensor,
    hidden_q: torch.Tensor,
    weight: torch.Tensor,
    *,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> torch.Tensor:
    """Per-token ``KL(P || Q)`` for two hidden-state distributions.

    Two vocabulary passes per token chunk: the first computes both
    log-normalisers, the second accumulates ``sum_v P_v (log P_v - log Q_v)``.
    """
    if hidden_p.shape != hidden_q.shape:
        raise ValueError("hidden_p and hidden_q must have the same shape")
    if hidden_p.ndim != 2:
        raise ValueError("hidden must be [tokens, hidden]")
    p = hidden_p.float()
    q = hidden_q.float()
    vocab = int(weight.shape[0])
    out = torch.empty(p.shape[0], dtype=torch.float32, device=p.device)
    for t0 in range(0, p.shape[0], token_chunk):
        t1 = min(t0 + token_chunk, p.shape[0])
        hp, hq = p[t0:t1], q[t0:t1]
        max_p = torch.full((t1 - t0,), float("-inf"), device=p.device)
        max_q = torch.full((t1 - t0,), float("-inf"), device=p.device)
        sum_p = torch.zeros((t1 - t0,), device=p.device)
        sum_q = torch.zeros((t1 - t0,), device=p.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            w = weight[v0:v1].float()
            zp = _softcap(hp @ w.t(), softcap)
            zq = _softcap(hq @ w.t(), softcap)
            new_p = torch.maximum(max_p, zp.max(dim=1).values)
            new_q = torch.maximum(max_q, zq.max(dim=1).values)
            sum_p = sum_p * torch.exp(max_p - new_p) + torch.exp(
                zp - new_p.unsqueeze(1)
            ).sum(dim=1)
            sum_q = sum_q * torch.exp(max_q - new_q) + torch.exp(
                zq - new_q.unsqueeze(1)
            ).sum(dim=1)
            max_p, max_q = new_p, new_q
        lse_p = max_p + torch.log(sum_p)
        lse_q = max_q + torch.log(sum_q)
        kl = torch.zeros((t1 - t0,), device=p.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            w = weight[v0:v1].float()
            zp = _softcap(hp @ w.t(), softcap)
            zq = _softcap(hq @ w.t(), softcap)
            logp = zp - lse_p.unsqueeze(1)
            logq = zq - lse_q.unsqueeze(1)
            kl = kl + (torch.exp(logp) * (logp - logq)).sum(dim=1)
        out[t0:t1] = kl
    return out


# ---------------------------------------------------------------------------
# Probe helpers over labelled batches
# ---------------------------------------------------------------------------


def _compact(hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted = labels[:, 1:]
    mask = shifted.reshape(-1) != -100
    compact = hidden[:, :-1].reshape(-1, hidden.shape[-1])[mask]
    return compact, shifted.reshape(-1)[mask]


def _batch_rows(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def _select_batch_rows(batch: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    rows = _batch_rows(batch)
    selected = [int(index) for index in indices]
    return {
        key: (
            value[selected]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
            else value
        )
        for key, value in batch.items()
    }


@torch.inference_mode()
def nll_from_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    *,
    softcap: float | None = None,
    temperature: float = 1.0,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> tuple[float, int] | None:
    """Mean NLL over labelled next-token positions, or ``None`` if empty."""
    compact, targets = _compact(hidden.detach(), labels)
    if compact.numel() == 0:
        return None
    values = chunked_nll_hidden(
        compact,
        weight,
        targets,
        softcap=softcap,
        temperature=float(temperature),
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    return float(values.mean().item()), int(targets.numel())


@torch.inference_mode()
def nll_from_hidden_temperatures(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    temperatures: Sequence[float],
    *,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> dict[float, tuple[float, int]]:
    """Mean NLL per temperature from one shared logits pass per token chunk."""

    compact, targets = _compact(hidden.detach(), labels)
    if compact.numel() == 0:
        return {}
    values = chunked_nll_temperatures(
        compact,
        weight,
        targets,
        temperatures,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    tokens = int(targets.numel())
    return {
        float(temperature): (float(nll.mean().item()), tokens)
        for temperature, nll in values.items()
    }


@torch.inference_mode()
def teacher_kl_from_batch(
    hidden: torch.Tensor,
    batch: Mapping[str, Any],
    weight: torch.Tensor,
    *,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> tuple[float, int] | None:
    """Mean ``KL(teacher || student)`` over labelled tokens of one batch."""
    probs = batch.get("teacher_topk_probs")
    indices = batch.get("teacher_topk_indices")
    tail = batch.get("teacher_tail_prob")
    labels = batch.get("labels")
    if probs is None or indices is None or tail is None or labels is None:
        return None
    labels = labels.long()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.reshape(-1) != -100
    k = int(probs.shape[-1])
    teacher = TeacherTopK(
        probs=probs[:, 1:].reshape(-1, k)[mask],
        indices=indices[:, 1:].reshape(-1, k)[mask],
        tail_prob=tail[:, 1:].reshape(-1)[mask],
    )
    compact = hidden.detach()[:, :-1].reshape(-1, hidden.shape[-1])[mask]
    if compact.numel() == 0:
        return None
    values = chunked_teacher_kl(
        compact,
        weight,
        teacher,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    return float(values.mean().item()), int(compact.shape[0])


@torch.inference_mode()
def exit1_bit_exact(
    model: torch.nn.Module,
    text_model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    parent: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare multi-exit exit 1 with the adapter-off ``O`` forward.

    ``parent`` is the already-computed adapter-off N=1 hidden state for
    ``batch`` when the caller has it (the probe does), which avoids a second
    identical forward.  Returns
    ``{"applicable": bool, "ok": bool|None, "max_abs_diff": float|None}``.
    An unlooped model has no exit 1 and is reported as not applicable rather
    than falsely green.
    """
    looped, _ = _loop_state(text_model)
    if not looped:
        return {"applicable": False, "ok": None, "max_abs_diff": None}
    if parent is None:
        parent = _forward_hidden(model, text_model, batch, 1, adapters_off=True)
    exit1 = _forward_hidden(
        model, text_model, batch, 1, adapters_off=False, exits=(1,)
    )
    if parent.shape != exit1.shape:
        return {
            "applicable": True,
            "ok": False,
            "max_abs_diff": None,
            "reason": f"shape mismatch {tuple(parent.shape)} vs {tuple(exit1.shape)}",
        }
    max_abs = float((parent.float() - exit1.float()).abs().max().item())
    return {
        "applicable": True,
        "ok": bool(torch.equal(parent, exit1)),
        "max_abs_diff": max_abs,
    }


def merge_invariants(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine per-micro-batch exit-1 checks into one whole-probe result.

    The registered invariant runs on the fixed first ``invariant_rows`` probe
    rows (the rows that formed the old four-row batch 0), split across their
    training micro-batches.  ``ok`` is the conjunction over the checked rows
    and ``max_abs_diff`` the maximum over them.  A part that raised is kept out
    of the ``ok`` conjunction: the exception is surfaced through
    ``probe_errors`` / ``probe_failed`` instead of becoming a terminal
    ``ok=False`` abort.
    """
    if not parts:
        return {"applicable": False, "ok": None}
    if any(not part.get("applicable") for part in parts):
        # ``_loop_state`` is a property of the model, identical for every
        # micro-batch; the first result is the registered not-applicable form.
        return dict(parts[0])
    errored = [part for part in parts if part.get("error") is not None]
    valid = [part for part in parts if part.get("error") is None]
    diffs = [
        float(part["max_abs_diff"])
        for part in valid
        if part.get("max_abs_diff") is not None
    ]
    ok: bool | None
    if any(part.get("ok") is False for part in valid):
        ok = False
    elif errored:
        ok = None
    elif valid:
        ok = all(part.get("ok") is True for part in valid)
    else:
        ok = None
    merged: dict[str, Any] = {
        "applicable": True,
        "ok": ok,
        "max_abs_diff": max(diffs) if diffs else None,
    }
    reasons = [str(part["reason"]) for part in valid if part.get("reason")]
    reasons.extend(f"probe failed: {part['error']}" for part in errored)
    if reasons:
        merged["reason"] = "; ".join(reasons)
    return merged


# ---------------------------------------------------------------------------
# Generation sanity
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _strip_thought(text: str) -> str:
    return re.sub(
        r"<\|channel>thought\b.*?<channel\|>", "", text, flags=re.DOTALL | re.IGNORECASE
    )


def _answer_text(text: str) -> str:
    cleaned = _strip_thought(text)
    # An unclosed thought channel means everything after it is still thinking.
    if _THOUGHT_OPEN in cleaned and _THOUGHT_CLOSE not in cleaned:
        cleaned = cleaned.split(_THOUGHT_OPEN, 1)[0]
    for token in _TURN_TOKENS:
        cleaned = cleaned.replace(token, " ")
    return cleaned.strip()


def generation_sanity(
    text: str,
    *,
    ngram: int = 8,
    repetition_rate_threshold: float = 0.5,
    repetition_max_count: int = 3,
) -> dict[str, Any]:
    """Deterministic sanity features of one completion."""
    open_pos = text.find(_THOUGHT_OPEN)
    has_thinking = open_pos >= 0
    close_pos = text.rfind(_THOUGHT_CLOSE)
    thinking_closed = (not has_thinking) or close_pos > open_pos
    answer = _answer_text(text)
    answer_present = bool(answer.strip(" \t\r\n`'\".,:;"))

    words = _WORD_RE.findall(text)
    ngram = max(1, int(ngram))
    grams = [
        tuple(words[index : index + ngram])
        for index in range(max(0, len(words) - ngram + 1))
    ]
    if grams:
        counts = Counter(grams)
        max_repeat = int(max(counts.values()))
        repetition_rate = 1.0 - (len(counts) / float(len(grams)))
    else:
        max_repeat = 0
        repetition_rate = 0.0
    repetition_loop = bool(
        max_repeat >= int(repetition_max_count)
        or repetition_rate > float(repetition_rate_threshold)
    )

    length = len(text)
    non_ascii = sum(1 for char in text if ord(char) > 127)
    replacement = text.count("\ufffd")
    return {
        "has_thinking": bool(has_thinking),
        "thinking_closed": bool(thinking_closed),
        "answer_present": bool(answer_present),
        "repetition_rate": float(repetition_rate),
        "max_ngram_repeat": int(max_repeat),
        "repetition_loop": repetition_loop,
        "length": int(length),
        "non_ascii_rate": float(non_ascii / length) if length else 0.0,
        "replacement_chars": int(replacement),
        "garbage_rate": float(replacement / length) if length else 0.0,
        "answer_excerpt": answer[:240],
    }


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    anchor_ns: tuple[int, ...] = DEFAULT_ANCHOR_NS
    eval_ns: tuple[int, ...] = DEFAULT_EVAL_NS
    nll_ns: tuple[int, ...] = DEFAULT_NLL_NS
    generation_ns: tuple[int, ...] = DEFAULT_GENERATION_NS
    generation_prompts: int = 8
    generation_max_new_tokens: int = 512
    generation_batch_size: int = 4
    # The registered exit-1 invariant is evaluated on the fixed first four
    # probe rows (the rows that formed the old four-row batch 0), independent
    # of the training forward micro-batch size.
    invariant_rows: int = 4
    nll_temperatures: tuple[float, ...] = (
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
    )
    token_chunk: int = 1024
    vocab_chunk: int = 16384
    ngram: int = 8
    repetition_rate_threshold: float = 0.5
    repetition_max_count: int = 3


DEFAULT_PROBE_CONFIG = ProbeConfig()


@dataclass
class ProbeRunner:
    """Run the monitor probes for one checkpoint state.

    ``batches`` are already-collated probe batches (input_ids, attention_mask,
    labels and, when available, the unshifted teacher fields).  ``prompts`` is
    a list of 1-D prompt tensors used for the generation sanity check.

    ``nll_fit_batches`` / ``nll_score_batches`` are *micro-batch* indices.
    Callers must derive them from row-level halves (the training wiring builds
    the two halves as separate batches), so changing the probe forward
    micro-batch size never moves a row between the fit and score populations.
    All per-token statistics (anchor KL, teacher KL, NLL) are accumulated as
    ``sum * tokens / tokens`` over every micro-batch, so the reported means are
    identical to one combined batch of all probe rows.
    """

    model: torch.nn.Module
    text_model: torch.nn.Module
    weight: torch.Tensor
    softcap: float | None
    batches: Sequence[Mapping[str, Any]]
    config: ProbeConfig = DEFAULT_PROBE_CONFIG
    tokenizer: Any = None
    prompts: Sequence[torch.Tensor] = field(default_factory=tuple)
    text_dir: Path | None = None
    generate_fn: Any = None
    nll_fit_batches: Sequence[int] | None = None
    nll_score_batches: Sequence[int] | None = None

    # -- public ------------------------------------------------------------
    def run(
        self,
        *,
        tokens_seen: int,
        step: int,
        train_teacher_kl: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        eval_ns = tuple(sorted({int(n) for n in self.config.eval_ns if int(n) >= 1}))
        if 1 not in eval_ns:
            eval_ns = (1, *eval_ns)
        anchor_sum: dict[int, float] = {n: 0.0 for n in eval_ns}
        anchor_count: dict[int, int] = {n: 0 for n in eval_ns}
        teacher_sum: dict[int, float] = {n: 0.0 for n in eval_ns}
        teacher_count: dict[int, int] = {n: 0 for n in eval_ns}
        temperatures = tuple(float(value) for value in self.config.nll_temperatures)
        nll_fit: dict[tuple[int, float], list[float]] = {}
        nll_score: dict[tuple[int, float], list[float]] = {}
        invariant_parts: list[dict[str, Any]] = []
        errors: list[str] = []
        nll_batches = len(self.batches)
        if self.nll_fit_batches is None and self.nll_score_batches is None:
            fit_batches = set(range(nll_batches))
            score_batches = set(range(nll_batches))
        else:
            fit_batches = {int(index) for index in (self.nll_fit_batches or [])}
            score_batches = {int(index) for index in (self.nll_score_batches or [])}
            for index in range(nll_batches):
                if index not in fit_batches and index not in score_batches:
                    score_batches.add(index)

        invariant_rows = max(0, int(self.config.invariant_rows))
        row_offset = 0
        invariant_errors = False
        invariant_error_messages: list[str] = []
        for batch_index, batch in enumerate(self.batches):
            labels = batch.get("labels")
            rows = _batch_rows(batch)
            hidden_by_n: dict[int, torch.Tensor] = {}
            for n in eval_ns:
                hidden_by_n[n] = _forward_hidden(
                    self.model,
                    self.text_model,
                    batch,
                    n,
                    adapters_off=n == 1,
                )
            selected = [
                index
                for index in range(rows)
                if row_offset + index < invariant_rows
            ]
            if selected:
                whole_batch = len(selected) == rows
                invariant_batch = (
                    batch if whole_batch else _select_batch_rows(batch, selected)
                )
                parent = hidden_by_n.get(1)
                if parent is not None and not whole_batch:
                    # A row sliced out of a larger bf16 forward can differ
                    # bitwise from the same row forwarded alone (the GEMM may
                    # pick another kernel), which would false-abort exit 1.
                    # Recompute the adapter-off parent on exactly the sliced
                    # batch instead of reusing the larger batch's hidden state.
                    parent = _forward_hidden(
                        self.model,
                        self.text_model,
                        invariant_batch,
                        1,
                        adapters_off=True,
                    )
                try:
                    invariant_part = exit1_bit_exact(
                        self.model,
                        self.text_model,
                        invariant_batch,
                        parent=parent,
                    )
                except Exception as exc:  # noqa: BLE001 - probe must not kill training
                    invariant_errors = True
                    message = f"invariant: {exc}"
                    errors.append(message)
                    invariant_error_messages.append(message)
                    invariant_part = {
                        "applicable": True,
                        "ok": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                invariant_parts.append(invariant_part)
            row_offset += rows

            if isinstance(labels, torch.Tensor) and 1 in hidden_by_n:
                parent = hidden_by_n[1]
                for n in eval_ns:
                    if n == 1:
                        continue
                    compact_parent, _ = _compact(parent, labels)
                    compact_student, _ = _compact(hidden_by_n[n], labels)
                    if compact_parent.numel() == 0:
                        continue
                    values = chunked_kl_hidden(
                        compact_parent,
                        compact_student,
                        self.weight,
                        softcap=self.softcap,
                        token_chunk=self.config.token_chunk,
                        vocab_chunk=self.config.vocab_chunk,
                    )
                    anchor_sum[n] += float(values.sum().item())
                    anchor_count[n] += int(values.numel())

            for n in eval_ns:
                result = teacher_kl_from_batch(
                    hidden_by_n[n],
                    batch,
                    self.weight,
                    softcap=self.softcap,
                    token_chunk=self.config.token_chunk,
                    vocab_chunk=self.config.vocab_chunk,
                )
                if result is not None:
                    mean_value, tokens = result
                    teacher_sum[n] += float(mean_value) * int(tokens)
                    teacher_count[n] += int(tokens)

                if n in self.config.nll_ns and isinstance(labels, torch.Tensor):
                    # One shared logits pass per exit/row for every registered
                    # temperature; the per-temperature arithmetic is identical
                    # to one ``nll_from_hidden`` call each (I2).
                    nll_results = nll_from_hidden_temperatures(
                        hidden_by_n[n],
                        labels,
                        self.weight,
                        temperatures,
                        softcap=self.softcap,
                        token_chunk=self.config.token_chunk,
                        vocab_chunk=self.config.vocab_chunk,
                    )
                    for temperature, nll_result in nll_results.items():
                        mean_value, tokens = nll_result
                        for bucket, indices in (
                            (nll_fit, fit_batches),
                            (nll_score, score_batches),
                        ):
                            if batch_index not in indices:
                                continue
                            entry = bucket.setdefault(
                                (int(n), float(temperature)), [0.0, 0.0]
                            )
                            entry[0] += float(mean_value) * int(tokens)
                            entry[1] += int(tokens)
            del hidden_by_n

        invariant = merge_invariants(invariant_parts)
        anchor = {
            str(n): 0.0
            if n == 1
            else float(anchor_sum[n] / anchor_count[n])
            for n in eval_ns
            if n == 1 or anchor_count[n] > 0
        }
        teacher = {
            str(n): float(teacher_sum[n] / teacher_count[n])
            for n in eval_ns
            if teacher_count[n] > 0
        }
        nll: dict[str, float] = {}
        nll_temperature: dict[str, float] = {}
        nll_fit_value: dict[str, float] = {}
        nll_fit_tokens: dict[str, int] = {}
        nll_score_tokens: dict[str, int] = {}
        for n in eval_ns:
            candidates = [
                (nll_fit[(n, temperature)][0] / nll_fit[(n, temperature)][1], temperature)
                for temperature in temperatures
                if (n, temperature) in nll_fit and nll_fit[(n, temperature)][1] > 0
            ]
            if not candidates:
                continue
            fit_mean, chosen = min(candidates, key=lambda item: (item[0], item[1]))
            score_entry = nll_score.get((n, chosen))
            if score_entry is not None and score_entry[1] > 0:
                score_mean = score_entry[0] / score_entry[1]
                nll_score_tokens[str(n)] = int(score_entry[1])
            else:
                score_mean = fit_mean
                nll_score_tokens[str(n)] = int(nll_fit[(n, chosen)][1])
            nll[str(n)] = float(score_mean)
            nll_temperature[str(n)] = float(chosen)
            nll_fit_value[str(n)] = float(fit_mean)
            nll_fit_tokens[str(n)] = int(nll_fit[(n, chosen)][1])
        teacher_tokens = {
            str(n): int(teacher_count[n]) for n in eval_ns if teacher_count[n] > 0
        }

        generation = []
        try:
            generation = self._generation(tokens_seen=tokens_seen, step=step)
        except Exception as exc:  # noqa: BLE001 - a probe failure is a warning
            errors.append(f"generation: {exc}")

        train_kl = {
            str(key): float(value)
            for key, value in (train_teacher_kl or {}).items()
            if value is not None and math.isfinite(float(value))
        }
        gap = {
            key: float(teacher[key] - train_kl[key])
            for key in teacher
            if key in train_kl
        }
        result = {
            "kind": "probe",
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "invariant": invariant,
            "anchor_kl": anchor,
            "teacher_kl": teacher,
            "teacher_fit_available": bool(teacher),
            "teacher_tokens": teacher_tokens,
            "train_teacher_kl": train_kl,
            "teacher_kl_gap": gap,
            "calibrated_nll": nll,
            "calibrated_nll_temperature": nll_temperature,
            "calibrated_nll_fit": nll_fit_value,
            "calibrated_nll_fit_tokens": nll_fit_tokens,
            "calibrated_nll_tokens": nll_score_tokens,
            "calibrated_nll_note": (
                "temperature fitted per (checkpoint, N) on one deterministic "
                "half of the MONITOR rows and scored on the other half; the "
                "pilot temperature-fit split is never touched"
            ),
            "generation": generation,
            "probe_errors": errors,
        }
        if invariant_errors:
            # A raised invariant check is a probe failure, handled by the
            # registered two-consecutive-failures WARN/ABORT rule; it is never
            # an immediate exit-1 mismatch abort.  Surface the message on the
            # probe itself so the WARN/ABORT does not read "unknown probe
            # error".
            result["probe_failed"] = True
            result["error"] = "; ".join(invariant_error_messages)
        return result

    # -- generation --------------------------------------------------------
    def _decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(int(i)) for i in ids)
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return " ".join(str(int(i)) for i in ids)
        return str(decode(list(ids), skip_special_tokens=False))

    def _eos_ids(self) -> set[int]:
        values: list[Any] = []
        config = getattr(self.model, "generation_config", None)
        for candidate in (
            getattr(config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        ):
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple, set)):
                values.extend(candidate)
            else:
                values.append(candidate)
        return {int(value) for value in values if value is not None}

    def _generation(self, *, tokens_seen: int, step: int) -> list[dict[str, Any]]:
        prompts = list(self.prompts)[: int(self.config.generation_prompts)]
        if not prompts:
            return []
        looped, previous_n = _loop_state(self.text_model)
        rows: list[dict[str, Any]] = []
        batch_size = max(1, int(self.config.generation_batch_size))
        try:
            # Move the offloaded vision modules once for the whole generation
            # loop so HF ``generate`` sees the compute device as model.device.
            with _generation_device(self.model):
                for n in self.config.generation_ns:
                    if looped:
                        _set_n_loops(self.text_model, int(n))
                    context = _adapter_context(self.model, int(n) == 1)
                    with _eval_mode(self.model), torch.inference_mode(), context:
                        for start in range(0, len(prompts), batch_size):
                            group = prompts[start : start + batch_size]
                            generated_group = self._generate_group(
                                group, int(n), manage_device=False
                            )
                            for offset, generated in enumerate(generated_group):
                                prompt_index = start + offset
                                text = self._decode(generated)
                                sanity = generation_sanity(
                                    text,
                                    ngram=self.config.ngram,
                                    repetition_rate_threshold=self.config.repetition_rate_threshold,
                                    repetition_max_count=self.config.repetition_max_count,
                                )
                                eos_ids = self._eos_ids()
                                if generated and int(generated[-1]) in eos_ids:
                                    finish_reason = "eos"
                                elif len(generated) >= int(
                                    self.config.generation_max_new_tokens
                                ):
                                    finish_reason = "length"
                                else:
                                    finish_reason = "stop"
                                path = self._write_text(
                                    tokens_seen=tokens_seen,
                                    prompt_index=prompt_index,
                                    n_loops=int(n),
                                    text=text,
                                )
                                rows.append(
                                    {
                                        "prompt_index": int(prompt_index),
                                        "n_loops": int(n),
                                        "completion_tokens": int(len(generated)),
                                        "finish_reason": finish_reason,
                                        "text_path": None if path is None else str(path),
                                        **sanity,
                                    }
                                )
        finally:
            if looped:
                _set_n_loops(self.text_model, previous_n)
        return rows

    def _trim_eos(self, ids: Sequence[int]) -> list[int]:
        eos = self._eos_ids()
        trimmed: list[int] = []
        for token in ids:
            trimmed.append(int(token))
            if int(token) in eos:
                break
        return trimmed

    def _generate_group(
        self,
        prompts: Sequence[torch.Tensor],
        n_loops: int,
        *,
        manage_device: bool = True,
    ) -> list[list[int]]:
        """Greedy generation for a small prompt group (left padded on the GPU)."""
        if self.generate_fn is not None:
            return [
                list(self.generate_fn(prompt, n_loops, self.config.generation_max_new_tokens))
                for prompt in prompts
            ]
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if self.tokenizer is None or pad is None or len(prompts) == 1:
            return [
                self._generate_one(prompt, n_loops, manage_device=manage_device)
                for prompt in prompts
            ]
        context = (
            _generation_device(self.model)
            if manage_device
            else nullcontext(model_compute_device(self.model))
        )
        with context as device:
            max_length = max(int(prompt.numel()) for prompt in prompts)
            input_ids = torch.full(
                (len(prompts), max_length), int(pad), dtype=torch.long, device=device
            )
            attention = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                ids = prompt.reshape(-1).to(device)
                input_ids[row, max_length - ids.numel() :] = ids
                attention[row, max_length - ids.numel() :] = 1
            generate = getattr(self.model, "generate", None)
            if not callable(generate):
                raise RuntimeError("the model has no generate() method")
            output = generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=int(self.config.generation_max_new_tokens),
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=int(pad),
                eos_token_id=(sorted(self._eos_ids()) or None),
            )
            return [
                self._trim_eos(output[row, max_length:].tolist())
                for row in range(len(prompts))
            ]

    def _generate_one(
        self,
        prompt: torch.Tensor,
        n_loops: int,
        *,
        manage_device: bool = True,
    ) -> list[int]:
        if self.generate_fn is not None:
            return list(
                self.generate_fn(prompt, n_loops, self.config.generation_max_new_tokens)
            )
        context = (
            _generation_device(self.model)
            if manage_device
            else nullcontext(model_compute_device(self.model))
        )
        with context as device:
            input_ids = prompt.reshape(1, -1).to(device)
            generate = getattr(self.model, "generate", None)
            if not callable(generate):
                raise RuntimeError("the model has no generate() method")
            eos = self._eos_ids()
            pad = getattr(self.tokenizer, "pad_token_id", None)
            output = generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(self.config.generation_max_new_tokens),
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad if pad is not None else 0,
                eos_token_id=(sorted(eos) if eos else None),
            )
            sequence = output[0].tolist()
            return self._trim_eos(sequence[input_ids.shape[1] :])

    def _write_text(
        self,
        *,
        tokens_seen: int,
        prompt_index: int,
        n_loops: int,
        text: str,
    ) -> Path | None:
        if self.text_dir is None:
            return None
        directory = Path(self.text_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"tokens{tokens_seen:09d}_p{prompt_index:02d}_N{n_loops}.txt"
        path.write_text(text, encoding="utf-8")
        return path


# Keep ``Iterable`` imported for callers that annotate batch sequences with it.
_ = Iterable
