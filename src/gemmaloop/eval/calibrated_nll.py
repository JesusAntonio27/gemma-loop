"""Temperature-calibrated NLL for looped and unlooped Gemma 4 models.

The pilot endpoint from the methodology review is **not** raw NLL: an untrained
loop can lower raw NLL by flattening the distribution, and a per-variant
temperature fit removes that artefact.  This module fits one scalar temperature
per model on a calibration split by minimising NLL with a bounded 1-D golden
section search, then reports on a separate test split:

* raw NLL (``T = 1``) and NLL at the fitted temperature;
* top-1 token accuracy (temperature-invariant for ``T > 0``);
* the same three quantities per position bucket (``<=1k``, ``1-2k``, ``2-4k``,
  ``4-8k``; an ``>8k`` bucket catches longer contexts).

Scoring never materialises the full ``[tokens, vocab]`` logits tensor: the LM
head is evaluated in token and vocabulary chunks and only the running
log-sum-exp, the target-scaled logit and the argmax are kept.  This is the same
lean-head arithmetic used by :mod:`gemmaloop.train.fused_ce`, with the
temperature applied after Gemma's final logit softcap.

The model helpers walk the usual wrappers (PEFT, ``Gemma4ForCausalLM`` and
``Gemma4ForConditionalGeneration``), set ``n_loops`` explicitly and restore it
afterwards, so they work for looped and unlooped models at any valid N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

__all__ = [
    "OVERFLOW_BUCKET",
    "POSITION_BUCKETS",
    "TokenScores",
    "bucket_report",
    "calibrated_nll_report",
    "fit_model_temperature",
    "fit_temperature_from_logits",
    "per_token_nll_from_logits",
    "position_bucket",
    "score_model",
    "top1_from_logits",
]

# Inclusive lower bound -> label.  A target at absolute token position ``p``
# (0-based, derived from the attention mask) falls in the first bucket whose
# upper bound is ``p < bound``.
POSITION_BUCKETS = (
    (1024, "<=1k"),
    (2048, "1-2k"),
    (4096, "2-4k"),
    (8192, "4-8k"),
)
OVERFLOW_BUCKET = ">8k"
BUCKET_LABELS = tuple(label for _bound, label in POSITION_BUCKETS)
TEMPERATURE_BOUNDS = (0.5, 4.0)


def position_bucket(position: int) -> str:
    """Return the bucket label for a 0-based absolute target-token position."""
    value = int(position)
    for bound, label in POSITION_BUCKETS:
        if value < bound:
            return label
    return OVERFLOW_BUCKET


# ---------------------------------------------------------------------------
# Pure tensor helpers (used directly by the synthetic-logit tests)
# ---------------------------------------------------------------------------
def per_token_nll_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Per-token NLL of ``softmax(logits / temperature)``.

    ``logits`` is ``[N, vocab]`` and ``targets`` is ``[N]``; the result is fp32
    and has shape ``[N]``.  The temperature is applied to the already-computed
    logits (i.e. after any final logit softcap).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return F.cross_entropy(
        logits.float() / float(temperature), targets, reduction="none"
    )


def top1_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Boolean ``[N]`` mask of correctly predicted targets."""
    return logits.argmax(dim=-1) == targets


def _golden_minimise(fn, low: float, high: float, max_iter: int, tol: float) -> float:
    if low <= 0 or high <= low:
        raise ValueError(f"invalid temperature bounds ({low}, {high})")
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = high - inv_phi * (high - low)
    d = low + inv_phi * (high - low)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if high - low <= tol:
            break
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - inv_phi * (high - low)
            fc = fn(c)
        else:
            low, c, fc = c, d, fd
            d = low + inv_phi * (high - low)
            fd = fn(d)
    return (low + high) / 2.0


def fit_temperature_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    *,
    max_iter: int = 60,
    tol: float = 1e-3,
) -> float:
    """Fit a scalar temperature by minimising mean NLL on ``logits``.

    A bounded 1-D golden-section search is sufficient because the NLL of an
    exponential-family tilt is unimodal in ``T`` on the positive half-line.
    """

    def objective(temperature: float) -> float:
        return float(
            per_token_nll_from_logits(logits, targets, temperature).mean().item()
        )

    return _golden_minimise(
        objective, float(bounds[0]), float(bounds[1]), max_iter, tol
    )


@dataclass
class TokenScores:
    """Per-token scores in the order the labelled tokens were produced."""

    nll: torch.Tensor
    top1: torch.Tensor
    positions: torch.Tensor

    def __len__(self) -> int:
        return int(self.nll.numel())


def _bucket_values(
    raw: torch.Tensor,
    calibrated: torch.Tensor,
    top1: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any] | None:
    count = int(mask.sum().item())
    if count == 0:
        return None
    return {
        "tokens": count,
        "raw_nll": float(raw[mask].mean().item()),
        "calibrated_nll": float(calibrated[mask].mean().item()),
        "top1_accuracy": float(top1[mask].float().mean().item()),
    }


def bucket_report(
    raw_nll: torch.Tensor,
    calibrated_nll: torch.Tensor,
    top1: torch.Tensor,
    positions: torch.Tensor,
    *,
    include_overflow: bool = True,
) -> dict[str, Any]:
    """Aggregate per-token values overall and per position bucket."""
    raw = torch.as_tensor(raw_nll).float().reshape(-1)
    calibrated = torch.as_tensor(calibrated_nll).float().reshape(-1)
    correct = torch.as_tensor(top1).bool().reshape(-1)
    pos = torch.as_tensor(positions).long().reshape(-1)
    if not (raw.numel() == calibrated.numel() == correct.numel() == pos.numel()):
        raise ValueError("raw_nll, calibrated_nll, top1 and positions must align")

    labels = list(BUCKET_LABELS)
    if include_overflow:
        labels.append(OVERFLOW_BUCKET)
    buckets: dict[str, Any] = {}
    labels_by_position = [position_bucket(value) for value in pos.tolist()]
    for label in labels:
        mask = torch.tensor(
            [value == label for value in labels_by_position], dtype=torch.bool
        )
        buckets[label] = _bucket_values(raw, calibrated, correct, mask)

    everything = _bucket_values(
        raw, calibrated, correct, torch.ones_like(raw, dtype=torch.bool)
    )
    assert everything is not None
    return {**everything, "buckets": buckets}


# ---------------------------------------------------------------------------
# Lean head over hidden states
# ---------------------------------------------------------------------------
def _input_device(text_model: torch.nn.Module) -> torch.device:
    """Device of the token embedding (inputs must live there; PLE handles CPU)."""
    embeddings = getattr(text_model, "embed_tokens", None)
    weight = getattr(embeddings, "weight", None)
    if weight is not None:
        return weight.device
    return next(text_model.parameters()).device


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _text_model_and_head(model: torch.nn.Module):
    base = _base_model(model)
    inner = getattr(base, "model", None)
    text_model = base if inner is None else getattr(inner, "language_model", inner)
    if text_model is None or not hasattr(text_model, "layers"):
        raise ValueError("cannot locate the Gemma 4 text model inside the wrapper")
    head = None
    get_head = getattr(base, "get_output_embeddings", None)
    if callable(get_head):
        head = get_head()
    weight = getattr(head, "weight", None)
    if weight is None:
        embeddings = getattr(text_model, "embed_tokens", None)
        weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise ValueError("cannot locate the LM-head / embedding weight")
    config = getattr(text_model, "config", None)
    softcap = getattr(config, "final_logit_softcapping", None)
    return text_model, weight, softcap


@torch.no_grad()
def _chunked_nll_top1(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float | None,
    temperature: float,
    token_chunk: int,
    vocab_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NLL and top-1 per token without materialising full logits."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    device = weight.device
    n_tokens = int(hidden.shape[0])
    vocab = int(weight.shape[0])
    nll = torch.empty(n_tokens, dtype=torch.float32)
    correct = torch.empty(n_tokens, dtype=torch.bool)
    for t0 in range(0, n_tokens, token_chunk):
        t1 = min(t0 + token_chunk, n_tokens)
        # Compute the LM-head matmul in fp32 with both operands explicitly cast:
        # the hidden states may be bf16 and the head weight may have a different
        # dtype (bf16 checkpoint, fp32 adapter scaffolding, ...).
        h = hidden[t0:t1].to(device=device, dtype=torch.float32)
        tgt = targets[t0:t1].to(device)
        running_max = torch.full((t1 - t0,), float("-inf"), device=device)
        sum_exp = torch.zeros((t1 - t0,), device=device)
        target_logit = torch.zeros((t1 - t0,), device=device)
        best = torch.full((t1 - t0,), float("-inf"), device=device)
        best_idx = torch.zeros((t1 - t0,), dtype=torch.long, device=device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            weight_tile = weight[v0:v1].to(device=device, dtype=torch.float32)
            logits = h @ weight_tile.t()
            if softcap is not None:
                logits = torch.tanh(logits / float(softcap)) * float(softcap)
            logits = logits / float(temperature)

            local = tgt - v0
            in_tile = (local >= 0) & (local < (v1 - v0))
            local_index = local.clamp(0, (v1 - v0) - 1).unsqueeze(1)
            tile_target = logits.gather(1, local_index).squeeze(1)
            target_logit = torch.where(in_tile, tile_target, target_logit)

            tile_max, tile_arg = logits.max(dim=1)
            better = tile_max > best
            best = torch.where(better, tile_max, best)
            best_idx = torch.where(better, tile_arg + v0, best_idx)

            new_max = torch.maximum(running_max, tile_max)
            sum_exp = sum_exp * torch.exp(running_max - new_max) + torch.exp(
                logits - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max

        nll[t0:t1] = (running_max + torch.log(sum_exp)) - target_logit
        correct[t0:t1] = best_idx == tgt
    return nll, correct


@dataclass
class _Collected:
    hidden: list[torch.Tensor]
    targets: list[torch.Tensor]
    positions: list[torch.Tensor]


def _collect(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    n_loops: int,
) -> tuple[_Collected, torch.Tensor, float | None]:
    text_model, weight, softcap = _text_model_and_head(model)
    previous_n = int(getattr(text_model, "_gemmaloop_n_loops", 1))
    looped = getattr(text_model, "_gemmaloop_loop_config", None) is not None
    hidden_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        if looped:
            from gemmaloop.model.looped import set_n_loops

            set_n_loops(text_model, int(n_loops))
        device = _input_device(text_model)
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels", input_ids).to(device)
            # Respect padding: derive absolute positions from the mask (the
            # same layout `generate`/the trainer use) instead of assuming a
            # right-padded, zero-offset batch.  Buckets use the target token's
            # position, not its physical index in the padded tensor.
            if attention_mask is None:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .unsqueeze(0)
                    .expand(input_ids.shape[0], -1)
                )
            else:
                position_ids = (
                    attention_mask.to(torch.long).cumsum(dim=-1) - 1
                ).clamp(min=0)
            with torch.no_grad():
                outputs = text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            hidden = outputs.last_hidden_state
            if hidden.shape[1] < 2:
                continue
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            positions = position_ids[:, 1:]
            hidden_chunks.append(
                hidden[:, :-1].reshape(-1, hidden.shape[-1])[mask.reshape(-1)].cpu()
            )
            target_chunks.append(
                shifted_labels.reshape(-1)[mask.reshape(-1)].cpu()
            )
            position_chunks.append(
                positions.reshape(-1)[mask.reshape(-1)].cpu()
            )
    finally:
        if looped:
            from gemmaloop.model.looped import set_n_loops

            set_n_loops(text_model, previous_n)
        if hasattr(model, "train"):
            model.train(was_training)
    return _Collected(hidden_chunks, target_chunks, position_chunks), weight, softcap


def _score_collected(
    collected: _Collected,
    weight: torch.Tensor,
    softcap: float | None,
    temperature: float,
    *,
    token_chunk: int,
    vocab_chunk: int,
) -> TokenScores:
    nll_parts = []
    top1_parts = []
    for hidden, targets in zip(collected.hidden, collected.targets):
        if hidden.numel() == 0:
            continue
        nll, top1 = _chunked_nll_top1(
            hidden,
            weight,
            targets,
            softcap=softcap,
            temperature=temperature,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        nll_parts.append(nll)
        top1_parts.append(top1)
    if nll_parts:
        nll = torch.cat(nll_parts)
        top1 = torch.cat(top1_parts)
    else:
        nll = torch.empty(0, dtype=torch.float32)
        top1 = torch.empty(0, dtype=torch.bool)
    positions = (
        torch.cat(collected.positions)
        if collected.positions
        else torch.empty(0, dtype=torch.long)
    )
    return TokenScores(nll=nll, top1=top1, positions=positions)


def score_model(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    n_loops: int,
    temperature: float = 1.0,
    *,
    token_chunk: int = 512,
    vocab_chunk: int = 16384,
) -> TokenScores:
    """Score a calibration/test split at ``n_loops`` and ``temperature``."""
    collected, weight, softcap = _collect(model, batches, n_loops)
    return _score_collected(
        collected,
        weight,
        softcap,
        temperature,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )


def fit_model_temperature(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    n_loops: int,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    *,
    token_chunk: int = 512,
    vocab_chunk: int = 16384,
    max_iter: int = 40,
    tol: float = 5e-3,
) -> float:
    """Fit the model's scalar temperature on ``batches`` (the calibration split).

    Hidden states are collected once; each golden-section evaluation only
    rescans the vocabulary in chunks.  This is still a bounded search (no full
    logits tensor is ever allocated).
    """
    collected, weight, softcap = _collect(model, batches, n_loops)
    if not any(chunk.numel() for chunk in collected.hidden):
        raise ValueError("calibration split produced no labelled tokens")

    def objective(temperature: float) -> float:
        scores = _score_collected(
            collected,
            weight,
            softcap,
            temperature,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        return float(scores.nll.mean().item())

    return _golden_minimise(
        objective, float(bounds[0]), float(bounds[1]), max_iter, tol
    )


def calibrated_nll_report(
    model: torch.nn.Module,
    calibration_batches: Iterable[Mapping[str, Any]],
    test_batches: Iterable[Mapping[str, Any]],
    n_loops: int,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    *,
    token_chunk: int = 512,
    vocab_chunk: int = 16384,
    max_iter: int = 40,
    tol: float = 5e-3,
) -> dict[str, Any]:
    """Fit T on the calibration split and report the test split at T and T=1."""
    temperature = fit_model_temperature(
        model,
        calibration_batches,
        n_loops,
        bounds,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        max_iter=max_iter,
        tol=tol,
    )
    collected, weight, softcap = _collect(model, test_batches, n_loops)
    raw = _score_collected(
        collected,
        weight,
        softcap,
        1.0,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    calibrated = _score_collected(
        collected,
        weight,
        softcap,
        temperature,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    if not torch.equal(raw.top1, calibrated.top1):
        # Temperature scaling with T > 0 must not change the argmax.  A mismatch
        # means an implementation bug, not a modelling decision.
        raise RuntimeError("temperature changed top-1 predictions")
    report = bucket_report(raw.nll, calibrated.nll, raw.top1, raw.positions)
    report["temperature"] = float(temperature)
    report["n_loops"] = int(n_loops)
    return report
