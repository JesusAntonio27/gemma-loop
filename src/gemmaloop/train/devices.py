"""Devices the model expects its inputs on.

``next(model.parameters()).device`` is wrong for a multimodal Gemma 4 while it
trains: :class:`gemmaloop.train.perf.VisionOffloadCallback` moves the frozen
vision tower to the CPU, and that tower is registered *before* the language
model, so the first parameter reports ``cpu`` even though text forwards run on
CUDA.  Token ids and attention masks must follow the text input embeddings.
"""

from __future__ import annotations

import torch

__all__ = ["model_compute_device"]


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _text_model(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort text stack of a multimodal/PEFT/looped wrapper."""
    base = _base_model(model)
    inner = getattr(base, "model", None)
    if inner is not None:
        language_model = getattr(inner, "language_model", None)
        if language_model is not None:
            return language_model
        if hasattr(inner, "embed_tokens"):
            return inner
    language_model = getattr(base, "language_model", None)
    if language_model is not None:
        return language_model
    return base


def _embedding_candidates(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Modules that may be the text ``embed_tokens`` embedding."""
    seen: set[int] = set()
    candidates: list[torch.nn.Module] = []

    def add(candidate: object) -> None:
        if isinstance(candidate, torch.nn.Module) and id(candidate) not in seen:
            seen.add(id(candidate))
            candidates.append(candidate)

    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            add(getter())
        except (AttributeError, NotImplementedError, TypeError):
            pass

    base = _base_model(model)
    text_model = _text_model(model)
    for module in (
        base,
        getattr(base, "model", None),
        getattr(base, "language_model", None),
        text_model,
        getattr(text_model, "language_model", None),
    ):
        add(getattr(module, "embed_tokens", None))
    return candidates


def model_compute_device(model: torch.nn.Module) -> torch.device:
    """Device of ``model``'s text input embeddings.

    Use this wherever token ids or attention masks are built for a forward
    pass.  It unwraps PEFT/looped wrappers and the multimodal wrapper to reach
    the text ``embed_tokens`` table, so an offloaded vision tower on the CPU
    cannot drag probe inputs off the compute device.

    Raises ``RuntimeError`` when the embeddings cannot be found: silently
    returning some other parameter's device is exactly the bug this helper
    exists to prevent.
    """
    for embeddings in _embedding_candidates(model):
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.device
    raise RuntimeError(
        "cannot locate the text input embeddings for "
        f"{type(model).__name__}; expected get_input_embeddings().weight or "
        "model.language_model.embed_tokens.weight"
    )
