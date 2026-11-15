"""Multi-exit ("anytime") distillation objective for the loop dial.

The objective follows ``dial-design.md`` section 2.2:

* every supervised exit ``t`` (the last ``k`` block passes of a sampled
  horizon ``T``) is scored with the Stage 1 loss

  ``ell_t = KL_teacher(t) + beta * KL(p_O || p_t)``

  where ``KL_teacher`` is a forward KL against the teacher's top-k logits plus
  one tail bucket (``k = 64`` in the registered recipe) and ``p_O`` is the
  adapter-off, exit-1 distribution (the parent anchor, ``beta = 0.1``);
* exits are averaged with uniform weights;
* a stop-gradient monotone hinge ``relu(ell_t - sg(ell_{t-1}))`` with
  ``lambda_m = 1`` only pushes a later exit that is worse *on that row*;
* ``T`` is drawn from the heavy-tailed horizon ``P(T=4)=0.7`` and
  ``P(5)=...=P(8)=0.075``.

The module is deliberately independent of the model: it takes the coda
hidden states of each exit plus the LM-head weight and returns per-token KLs
and the scalar loss.  All student projections are chunked over tokens and
vocabulary, so a 262k-vocabulary LM head is never materialised.  The parent
log-normaliser is computed once per step; the parent logits are recomputed
inside each exit's pass (never stored as a vocab tile).

Teacher contract
----------------
A real teacher is stored offline as, per assistant token, its top-k
probabilities, their vocabulary indices, and the probability of the single
"tail" bucket (the remaining vocabulary).  :class:`TeacherTopK` carries
exactly that.  When no teacher is supplied the supervised label is used as a
one-hot teacher, which makes ``KL_teacher`` the usual next-token
cross-entropy (the current Stage 1 target) without a teacher-logit pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from gemmaloop.train.loop_sampler import sample_n_loops

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_LAMBDA_HINGE",
    "DEFAULT_WINDOW",
    "DIAL_LOOP_PROBS",
    "DialLossResult",
    "TeacherTopK",
    "chunked_exit_kl",
    "chunked_logsumexp",
    "chunked_teacher_kl",
    "compute_window",
    "dial_objective",
    "monotone_hinge",
    "sample_horizon",
    "teacher_from_labels",
    "teacher_from_topk_logits",
    "teacher_heal_loss",
    "window_bounds",
]

# P(T=4)=0.70, P(5)=P(6)=P(7)=P(8)=0.075.  ``sample_n_loops`` maps the draw
# onto these buckets starting at N=1, so the leading zeros are intentional.
DIAL_LOOP_PROBS: tuple[float, ...] = (0.0, 0.0, 0.0, 0.70, 0.075, 0.075, 0.075, 0.075)

DEFAULT_WINDOW = 3
DEFAULT_BETA = 0.1
DEFAULT_LAMBDA_HINGE = 1.0

# Teacher shards store float16 probabilities; their batch sum may deviate from
# 1 by a few 1e-4 (and more at K=64).  The check uses fp32 and this tolerance,
# and ``TeacherTopK.renormalized`` restores an exact fp32 sum afterwards.
TEACHER_SUM_TOLERANCE = 2e-3

DEFAULT_TOKEN_CHUNK = 2048
DEFAULT_VOCAB_CHUNK = 16384

_TINY = 1e-7


# ---------------------------------------------------------------------------
# Horizon and window
# ---------------------------------------------------------------------------


def sample_horizon(
    seed: int,
    micro_step: int,
    probs: Sequence[float] = DIAL_LOOP_PROBS,
) -> int:
    """Sample the training horizon ``T`` (block passes per step).

    Reuses the deterministic ``sha256(seed:micro_step)`` sampler so a resumed
    run continues the exact same horizon sequence.  The dial horizon starts at
    2: a horizon of 1 would not train the loop at all.
    """
    horizon = sample_n_loops(seed, micro_step, tuple(probs))
    if horizon < 2:
        raise ValueError(
            f"the dial horizon must be at least 2, got {horizon}; "
            "use DIAL_LOOP_PROBS or another distribution without mass on N=1"
        )
    return horizon


def window_bounds(total_passes: int, window: int = DEFAULT_WINDOW) -> tuple[int, int]:
    """Return the inclusive ``(start, end)`` of the last ``window`` trainable passes.

    ``start = max(2, T - window + 1)`` because pass 1 is always the frozen,
    adapter-off identity.  This is ``W = {max(2, T-2), ..., T}`` at ``k = 3``.
    """
    total_passes = int(total_passes)
    window = int(window)
    if total_passes < 1:
        raise ValueError(f"total_passes must be at least 1, got {total_passes}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    return max(2, total_passes - window + 1), total_passes


def compute_window(total_passes: int, window: int = DEFAULT_WINDOW) -> range:
    """The supervised exit indices: ``range(*window_bounds(...))`` inclusive."""
    start, end = window_bounds(total_passes, window)
    return range(start, end + 1)


# ---------------------------------------------------------------------------
# Teacher representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherTopK:
    """Teacher distribution as top-k probabilities plus a tail bucket.

    After :meth:`_teacher_for_sequence` normalisation this is the flattened
    ``[N, K]`` / ``[N]`` kernel form.  The batch contract at the trainer
    boundary is ``[B, S, K]`` for ``probs``/``indices`` and ``[B, S]`` for
    ``tail_prob`` (see :func:`_teacher_for_sequence`).
    """

    probs: torch.Tensor
    indices: torch.Tensor
    tail_prob: torch.Tensor

    def validate(
        self,
        vocab_size: int,
        *,
        tolerance: float = TEACHER_SUM_TOLERANCE,
    ) -> "TeacherTopK":
        if self.probs.ndim != 2 or self.indices.shape != self.probs.shape:
            raise ValueError(
                "teacher probs and indices must both have shape [N, K], got "
                f"{tuple(self.probs.shape)} and {tuple(self.indices.shape)}"
            )
        if self.tail_prob.ndim != 1 or self.tail_prob.shape[0] != self.probs.shape[0]:
            raise ValueError(
                f"teacher tail_prob must have shape [{self.probs.shape[0]}], "
                f"got {tuple(self.tail_prob.shape)}"
            )
        if self.probs.shape[1] < 1:
            raise ValueError("teacher needs at least one top-k entry per token")
        # Shards store float16 probabilities; validate in fp32.
        probs = self.probs.to(torch.float32)
        tail = self.tail_prob.to(torch.float32)
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("teacher top-k probabilities must be finite and non-negative")
        if not torch.isfinite(tail).all() or (tail < 0).any():
            raise ValueError("teacher tail probabilities must be finite and non-negative")
        indices = self.indices
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"teacher indices must be integers, got {indices.dtype}")
        if (indices < 0).any() or (indices >= vocab_size).any():
            raise ValueError(
                f"teacher indices must lie in [0, {vocab_size})"
            )
        total = probs.sum(dim=-1) + tail
        if (total - 1.0).abs().max().item() > float(tolerance):
            raise ValueError(
                "teacher top-k probabilities and the tail bucket must sum to 1 "
                f"(tolerance {float(tolerance)}); max deviation "
                f"{(total - 1.0).abs().max().item():.2e}"
            )
        return self

    def renormalized(
        self,
        vocab_size: int,
        *,
        tolerance: float = TEACHER_SUM_TOLERANCE,
    ) -> "TeacherTopK":
        """Validate and return fp32 probabilities whose sum is exactly one.

        The tail mass is kept as a separate bucket; both it and the top-k
        entries are divided by their fp32 total.
        """
        self.validate(vocab_size, tolerance=tolerance)
        probs = self.probs.to(torch.float32)
        tail = self.tail_prob.to(torch.float32)
        total = (probs.sum(dim=-1) + tail).clamp(min=_TINY)
        return TeacherTopK(
            probs=probs / total.unsqueeze(1),
            indices=self.indices,
            tail_prob=tail / total,
        )


def teacher_from_labels(
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> TeacherTopK:
    """One-hot teacher from next-token labels (``KL_teacher`` becomes CE).

    Ignored positions are replaced by index 0 with probability 1; the caller's
    token mask removes them from the row means.
    """
    flat = labels.reshape(-1).to(torch.long)
    flat = torch.where(flat == int(ignore_index), torch.zeros_like(flat), flat)
    probs = torch.ones(flat.shape[0], 1, dtype=torch.float32, device=flat.device)
    tail = torch.zeros(flat.shape[0], dtype=torch.float32, device=flat.device)
    return TeacherTopK(probs=probs, indices=flat.unsqueeze(1), tail_prob=tail)


def teacher_from_topk_logits(
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    tail_logit: torch.Tensor,
) -> TeacherTopK:
    """Convert teacher top-k logits plus a tail logit into a distribution.

    The ``K+1`` logits are normalised together, so the tail entry carries
    whatever mass the teacher's remaining vocabulary holds.
    """
    if topk_logits.shape != topk_indices.shape:
        raise ValueError("topk_logits and topk_indices must have the same shape")
    tail = tail_logit.reshape(-1, 1).to(torch.float32)
    logits = torch.cat([topk_logits.to(torch.float32), tail], dim=1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return TeacherTopK(
        probs=log_probs[:, :-1].exp(),
        indices=topk_indices,
        tail_prob=log_probs[:, -1].exp(),
    )


# ---------------------------------------------------------------------------
# Chunked student kernels
# ---------------------------------------------------------------------------


def _apply_softcap(z: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return z
    return torch.tanh(z / float(softcap)) * float(softcap)


def _project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(hidden, weight.t())


def _validate_chunks(token_chunk: int, vocab_chunk: int) -> None:
    if token_chunk < 1 or vocab_chunk < 1:
        raise ValueError("token_chunk and vocab_chunk must be positive")


def _gather_topk(
    z_tile: torch.Tensor,
    indices: torch.Tensor,
    v0: int,
    v1: int,
    topk_z: torch.Tensor,
) -> None:
    """Copy the top-k logits that fall in ``[v0, v1)`` into ``topk_z``."""
    n, width = z_tile.shape
    local = indices - v0
    in_tile = (local >= 0) & (local < width)
    local = local.clamp(0, width - 1)
    rows = torch.arange(n, device=z_tile.device).unsqueeze(1)
    flat_index = (rows * width + local).reshape(-1)
    gathered = z_tile.reshape(-1).gather(0, flat_index).reshape_as(indices)
    topk_z.copy_(torch.where(in_tile, gathered, topk_z))


def chunked_logsumexp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    softcap: float | None = None,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
) -> torch.Tensor:
    """Full-vocabulary ``logsumexp`` of ``softcap(hidden @ weight.T)`` per token."""
    _validate_chunks(token_chunk, vocab_chunk)
    n_tokens = hidden.shape[0]
    vocab = weight.shape[0]
    out = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
    for t0 in range(0, n_tokens, token_chunk):
        t1 = min(t0 + token_chunk, n_tokens)
        n = t1 - t0
        h = hidden[t0:t1]
        running_max = torch.full((n,), float("-inf"), dtype=torch.float32, device=h.device)
        sum_exp = torch.zeros(n, dtype=torch.float32, device=h.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            z = _apply_softcap(_project(h, weight[v0:v1]).float(), softcap)
            tile_max = z.max(dim=1).values
            new_max = torch.maximum(running_max, tile_max)
            alpha = torch.exp(running_max - new_max)
            sum_exp = sum_exp * alpha + torch.exp(z - new_max.unsqueeze(1)).sum(dim=1)
            running_max = new_max
        out[t0:t1] = running_max + torch.log(sum_exp)
    return out


def _teacher_kl_from_lse(
    topk_z: torch.Tensor,
    lse: torch.Tensor,
    probs: torch.Tensor,
    tail_prob: torch.Tensor,
) -> torch.Tensor:
    q_logp = topk_z - lse.unsqueeze(1)
    q_mass = torch.exp(q_logp).sum(dim=1)
    tail_logp = torch.log1p(-q_mass.clamp(max=1.0 - _TINY))
    safe_probs = probs.clamp(min=_TINY)
    topk_term = torch.where(
        probs > 0,
        probs * (torch.log(safe_probs) - q_logp),
        torch.zeros_like(probs),
    ).sum(dim=1)
    tail_term = torch.where(
        tail_prob > 0,
        tail_prob * (torch.log(tail_prob.clamp(min=_TINY)) - tail_logp),
        torch.zeros_like(tail_prob),
    )
    return topk_term + tail_term


def _softcap_derivative(z_capped: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return torch.ones_like(z_capped)
    t = z_capped / float(softcap)
    return 1.0 - t * t


class _ChunkedKL(torch.autograd.Function):
    """Teacher and parent KL with vocab tiles recomputed in backward.

    The forward saves only ``h_exit``, ``h_parent``, the teacher arrays and the
    per-token normalisers (``lse_exit``, ``lse_parent``, ``Q``).  The backward
    re-projects each vocabulary tile and accumulates ``dL/dh`` directly, so
    saved activation memory is ``O(tokens * (hidden + K))`` rather than
    ``O(tokens * vocab)`` (the fused-CE pattern).
    """

    @staticmethod
    def forward(  # noqa: PLR0913 - autograd signature
        ctx,
        parent_hidden: torch.Tensor | None,
        exit_hidden: torch.Tensor,
        weight: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
        tail_prob: torch.Tensor,
        parent_lse: torch.Tensor | None,
        exit_lse: torch.Tensor | None,
        softcap: float | None,
        token_chunk: int,
        vocab_chunk: int,
        compute_parent: bool,
    ):
        _validate_chunks(token_chunk, vocab_chunk)
        n_tokens = exit_hidden.shape[0]
        vocab = weight.shape[0]
        device = exit_hidden.device
        teacher_kl = torch.empty(n_tokens, dtype=torch.float32, device=device)
        parent_kl = torch.zeros(n_tokens, dtype=torch.float32, device=device)
        lse_exit_out = (
            torch.empty(n_tokens, dtype=torch.float32, device=device)
            if exit_lse is None
            else exit_lse.to(torch.float32)
        )
        lse_parent_out = (
            parent_lse.to(torch.float32)
            if compute_parent and parent_lse is not None
            else lse_exit_out
        )
        if compute_parent and parent_lse is None:
            raise ValueError("parent_lse is required when compute_parent=True")
        q_out = torch.empty(n_tokens, dtype=torch.float32, device=device)

        for t0 in range(0, n_tokens, token_chunk):
            t1 = min(t0 + token_chunk, n_tokens)
            n = t1 - t0
            e_hidden = exit_hidden[t0:t1]
            p_hidden = parent_hidden[t0:t1] if compute_parent else e_hidden
            lse_parent = lse_parent_out[t0:t1]
            topk_indices = indices[t0:t1]
            p_chunk = probs[t0:t1].to(torch.float32)
            tail_chunk = tail_prob[t0:t1].to(torch.float32)
            topk_z = torch.full(
                topk_indices.shape, float("-inf"), dtype=torch.float32, device=device
            )
            running_max = torch.full(
                (n,), float("-inf"), dtype=torch.float32, device=device
            )
            sum_exp = torch.zeros(n, dtype=torch.float32, device=device)
            sum_p = torch.zeros(n, dtype=torch.float32, device=device)
            sum_p_zp = torch.zeros(n, dtype=torch.float32, device=device)
            sum_p_ze = torch.zeros(n, dtype=torch.float32, device=device)
            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                z_exit = _apply_softcap(
                    _project(e_hidden, weight[v0:v1]).float(), softcap
                )
                _gather_topk(z_exit, topk_indices, v0, v1, topk_z)
                if exit_lse is None:
                    tile_max = z_exit.max(dim=1).values
                    new_max = torch.maximum(running_max, tile_max)
                    alpha = torch.exp(running_max - new_max)
                    sum_exp = sum_exp * alpha + torch.exp(
                        z_exit - new_max.unsqueeze(1)
                    ).sum(dim=1)
                    running_max = new_max
                if compute_parent:
                    z_parent = _apply_softcap(
                        _project(p_hidden, weight[v0:v1]).float(), softcap
                    )
                    p_tile = torch.exp(z_parent - lse_parent.unsqueeze(1))
                    sum_p = sum_p + p_tile.sum(dim=1)
                    sum_p_zp = sum_p_zp + (p_tile * z_parent).sum(dim=1)
                    sum_p_ze = sum_p_ze + (p_tile * z_exit).sum(dim=1)
            lse_e = (
                running_max + torch.log(sum_exp)
                if exit_lse is None
                else lse_exit_out[t0:t1]
            )
            lse_exit_out[t0:t1] = lse_e
            topk_q = torch.exp(topk_z - lse_e.unsqueeze(1))
            q_out[t0:t1] = topk_q.sum(dim=1)
            teacher_kl[t0:t1] = _teacher_kl_from_lse(
                topk_z, lse_e, p_chunk, tail_chunk
            )
            if compute_parent:
                parent_kl[t0:t1] = (
                    sum_p_zp - sum_p * lse_parent - sum_p_ze + sum_p * lse_e
                )

        ctx.save_for_backward(
            exit_hidden,
            parent_hidden if compute_parent else exit_hidden,
            weight,
            probs,
            indices,
            tail_prob,
            lse_exit_out,
            lse_parent_out,
            q_out,
            parent_kl,
        )
        ctx.softcap = softcap
        ctx.token_chunk = int(token_chunk)
        ctx.vocab_chunk = int(vocab_chunk)
        ctx.compute_parent = bool(compute_parent)
        return teacher_kl, parent_kl

    @staticmethod
    def backward(ctx, grad_teacher_kl, grad_parent_kl):  # type: ignore[override]
        (
            exit_hidden,
            parent_hidden,
            weight,
            probs,
            indices,
            tail_prob,
            lse_exit,
            lse_parent,
            q_mass,
            parent_kl,
        ) = ctx.saved_tensors
        softcap = ctx.softcap
        token_chunk = ctx.token_chunk
        vocab_chunk = ctx.vocab_chunk
        compute_parent = ctx.compute_parent
        n_tokens = exit_hidden.shape[0]
        vocab = weight.shape[0]
        device = exit_hidden.device
        grad_exit = torch.zeros(exit_hidden.shape, dtype=torch.float32, device=device)
        grad_parent = (
            torch.zeros(parent_hidden.shape, dtype=torch.float32, device=device)
            if compute_parent
            else None
        )
        zeros = torch.zeros(n_tokens, dtype=torch.float32, device=device)
        g_teacher = (
            grad_teacher_kl.to(torch.float32)
            if grad_teacher_kl is not None
            else zeros
        )
        g_parent = (
            grad_parent_kl.to(torch.float32)
            if compute_parent and grad_parent_kl is not None
            else zeros
        )
        if not ctx.needs_input_grad[1] and not (
            compute_parent and ctx.needs_input_grad[0]
        ):
            return (None,) * 12

        for t0 in range(0, n_tokens, token_chunk):
            t1 = min(t0 + token_chunk, n_tokens)
            e_hidden = exit_hidden[t0:t1]
            p_hidden = parent_hidden[t0:t1]
            topk_indices = indices[t0:t1].to(torch.long)
            p_chunk = probs[t0:t1].to(torch.float32)
            tail_chunk = tail_prob[t0:t1].to(torch.float32)
            lse_e = lse_exit[t0:t1]
            lse_p = lse_parent[t0:t1]
            q_mass_chunk = q_mass[t0:t1].clamp(max=1.0 - _TINY)
            one_minus_q = (1.0 - q_mass_chunk).clamp(min=_TINY)
            sum_p = p_chunk.sum(dim=1)
            gT = g_teacher[t0:t1]
            gP = g_parent[t0:t1]
            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                w = weight[v0:v1]
                z_exit = _apply_softcap(_project(e_hidden, w).float(), softcap)
                q = torch.exp(z_exit - lse_e.unsqueeze(1))
                base = q * (
                    sum_p.unsqueeze(1)
                    - tail_chunk.unsqueeze(1)
                    * q_mass_chunk.unsqueeze(1)
                    / one_minus_q.unsqueeze(1)
                )
                local = topk_indices - v0
                in_tile = (local >= 0) & (local < (v1 - v0))
                local = local.clamp(0, v1 - v0 - 1)
                q_at = q.gather(1, local)
                topk_delta = torch.where(
                    in_tile,
                    tail_chunk.unsqueeze(1)
                    * q_at
                    / one_minus_q.unsqueeze(1)
                    - p_chunk,
                    torch.zeros_like(q_at),
                )
                teacher_grad = base.scatter_add(1, local, topk_delta)
                grad_z = gT.unsqueeze(1) * teacher_grad
                if compute_parent:
                    z_parent = _apply_softcap(
                        _project(p_hidden, w).float(), softcap
                    )
                    parent_p = torch.exp(z_parent - lse_p.unsqueeze(1))
                    grad_z = grad_z + gP.unsqueeze(1) * (q - parent_p)
                    # d KL(P || q) / d z_parent_v
                    #   = P_v * (log P_v - log q_v - KL(P || q))
                    kl_chunk = parent_kl[t0:t1].unsqueeze(1)
                    log_p_tile = z_parent - lse_p.unsqueeze(1)
                    log_q_tile = z_exit - lse_e.unsqueeze(1)
                    grad_z_parent = gP.unsqueeze(1) * parent_p * (
                        log_p_tile - log_q_tile - kl_chunk
                    )
                    grad_z_parent = grad_z_parent * _softcap_derivative(
                        z_parent, softcap
                    )
                    grad_parent[t0:t1] += torch.mm(grad_z_parent, w.float())
                grad_z = grad_z * _softcap_derivative(z_exit, softcap)
                grad_exit[t0:t1] += torch.mm(grad_z, w.float())

        grad_exit = grad_exit.to(exit_hidden.dtype)
        if grad_parent is not None:
            grad_parent = grad_parent.to(parent_hidden.dtype)
        return (
            grad_parent,
            grad_exit,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_teacher_kl(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher: TeacherTopK,
    *,
    softcap: float | None = None,
    lse: torch.Tensor | None = None,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token forward ``KL(teacher || student)`` with a top-k + tail teacher.

    ``lse`` may carry a previously computed full-vocabulary log-normaliser
    (the parent anchor's), which avoids a duplicate LSE pass for exit 1.
    """
    _validate_chunks(token_chunk, vocab_chunk)
    teacher = teacher.renormalized(weight.shape[0])
    if lse is not None and lse.shape != (hidden.shape[0],):
        raise ValueError(
            f"lse must have shape [{hidden.shape[0]}], got {tuple(lse.shape)}"
        )
    return _ChunkedKL.apply(
        None,
        hidden,
        weight,
        teacher.probs,
        teacher.indices,
        teacher.tail_prob,
        None,
        lse,
        softcap,
        int(token_chunk),
        int(vocab_chunk),
        False,
    )[0]


def chunked_exit_kl(
    parent_hidden: torch.Tensor,
    exit_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher: TeacherTopK,
    *,
    softcap: float | None = None,
    parent_lse: torch.Tensor | None = None,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``(KL_teacher, KL(parent || exit))`` in one vocabulary pass.

    The parent log-normaliser is computed once (pass ``parent_lse`` to reuse it
    across exits); both the parent and exit logits are recomputed inside the
    autograd backward, which keeps saved memory independent of the vocabulary.
    """
    _validate_chunks(token_chunk, vocab_chunk)
    if parent_hidden.shape != exit_hidden.shape:
        raise ValueError(
            "parent_hidden and exit_hidden must have the same shape, got "
            f"{tuple(parent_hidden.shape)} and {tuple(exit_hidden.shape)}"
        )
    teacher = teacher.renormalized(weight.shape[0])
    if parent_lse is None:
        with torch.no_grad():
            parent_lse = chunked_logsumexp(
                parent_hidden,
                weight,
                softcap=softcap,
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
            )
    return _ChunkedKL.apply(
        parent_hidden,
        exit_hidden,
        weight,
        teacher.probs,
        teacher.indices,
        teacher.tail_prob,
        parent_lse,
        None,
        softcap,
        int(token_chunk),
        int(vocab_chunk),
        True,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def monotone_hinge(
    loss: torch.Tensor,
    previous_loss: torch.Tensor,
    lambda_m: float = DEFAULT_LAMBDA_HINGE,
) -> torch.Tensor:
    """``lambda_m * relu(loss - sg(previous_loss))`` per row."""
    if loss.shape != previous_loss.shape:
        raise ValueError(
            f"hinge operands must have the same shape, got {tuple(loss.shape)} "
            f"and {tuple(previous_loss.shape)}"
        )
    return float(lambda_m) * torch.relu(loss - previous_loss.detach())


@dataclass
class DialLossResult:
    """Result of :func:`dial_objective`; tensors keep their gradients."""

    total: torch.Tensor
    per_exit_loss: dict[int, torch.Tensor]
    per_exit_kl_teacher: dict[int, torch.Tensor]
    per_exit_kl_parent: dict[int, torch.Tensor]
    per_exit_hinge: dict[int, torch.Tensor]
    metrics: dict[str, float]
    per_exit_token_sum: dict[int, torch.Tensor] = field(default_factory=dict)
    row_token_counts: torch.Tensor | None = None
    token_sum: torch.Tensor | None = None
    window_size: int = 0


def _row_counts(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    counts = torch.zeros(n_rows, dtype=torch.float32, device=rows.device)
    counts.index_add_(0, rows, torch.ones(rows.shape[0], dtype=torch.float32, device=rows.device))
    return counts


def _row_sum(
    values: torch.Tensor, rows: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    sums = torch.zeros(counts.shape[0], dtype=values.dtype, device=values.device)
    sums.index_add_(0, rows, values)
    return sums


def _row_mean(
    values: torch.Tensor, rows: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    return _row_sum(values, rows, counts) / counts.clamp(min=1.0)


def _teacher_for_sequence(
    labels: torch.Tensor,
    teacher: TeacherTopK | None,
    ignore_index: int,
) -> tuple[torch.Tensor, TeacherTopK]:
    """Shift labels by one and return ``(shifted_labels, shifted teacher)``.

    **Batch contract (shared with T-heal):** teacher fields are *unshifted* and
    aligned 1:1 with ``labels``/``input_ids``.  ``probs``/``indices`` are
    ``[B, S, K]`` (ids int32, probs float16) and ``tail_prob`` is ``[B, S]``
    (float32).  The single left shift by one happens here, immediately before
    the row mask, so the collator must not shift again.  A flattened
    ``[B*S, K]`` / ``[B*S]`` layout is also accepted for direct API use.
    """
    shifted = labels[:, 1:]
    if teacher is None:
        return shifted, teacher_from_labels(shifted, ignore_index=ignore_index)

    labels_shape = labels.shape
    k = int(teacher.probs.shape[-1]) if teacher.probs.ndim >= 2 else 0
    if teacher.probs.ndim == 3:
        if teacher.probs.shape != (*labels_shape, k):
            raise ValueError(
                "teacher probs must have shape [B, S, K] = "
                f"{(*labels_shape, k)}, got {tuple(teacher.probs.shape)}"
            )
        if teacher.indices.shape != teacher.probs.shape:
            raise ValueError(
                "teacher indices must have the same [B, S, K] shape as probs, "
                f"got {tuple(teacher.indices.shape)}"
            )
        if teacher.tail_prob.shape != labels_shape:
            raise ValueError(
                f"teacher tail_prob must have shape {labels_shape}, got "
                f"{tuple(teacher.tail_prob.shape)}"
            )
        probs = teacher.probs[:, 1:]
        indices = teacher.indices[:, 1:]
        tail = teacher.tail_prob[:, 1:]
    elif teacher.probs.ndim == 2:
        if teacher.probs.shape[0] != labels.numel():
            raise ValueError(
                "a flattened teacher must cover every position of labels "
                f"({labels.numel()} tokens), got {teacher.probs.shape[0]}"
            )
        probs = teacher.probs.reshape(*labels_shape, k)[:, 1:]
        indices = teacher.indices.reshape(*labels_shape, k)[:, 1:]
        tail = teacher.tail_prob.reshape(labels_shape)[:, 1:]
    else:
        raise ValueError(
            "teacher probs must have shape [B, S, K] or [B*S, K], got "
            f"{tuple(teacher.probs.shape)}"
        )
    if indices.shape != probs.shape:
        raise ValueError("teacher probs and indices must have the same K")
    return (
        shifted,
        TeacherTopK(
            probs=probs.reshape(-1, k),
            indices=indices.reshape(-1, k),
            tail_prob=tail.reshape(-1),
        ),
    )


def dial_objective(
    exit_hidden_states: Mapping[int, torch.Tensor],
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    teacher: TeacherTopK | None = None,
    softcap: float | None = None,
    beta: float = DEFAULT_BETA,
    lambda_hinge: float = DEFAULT_LAMBDA_HINGE,
    window: int = DEFAULT_WINDOW,
    ignore_index: int = -100,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
) -> DialLossResult:
    """Compute the multi-exit objective.

    ``exit_hidden_states`` maps a block-pass count to that exit's **coda
    output** before the LM head, shaped ``[batch, seq, hidden]``.  It must
    contain exit 1 (the parent) and every pass of the window.  Labels are
    shifted internally (``hidden[:, :-1]`` predicts ``labels[:, 1:]``), as in
    the lean LM head.  Losses are row means over assistant tokens.
    """
    if 1 not in exit_hidden_states:
        raise ValueError("exit_hidden_states must contain exit 1 (the parent)")
    supervised = sorted(t for t in exit_hidden_states if int(t) > 1)
    if not supervised:
        raise ValueError("exit_hidden_states must contain at least one exit above 1")
    expected = sorted(compute_window(max(supervised), window))
    if supervised != expected:
        raise ValueError(
            "the supervised exits must be exactly the last "
            f"{int(window)} passes {expected}, got {supervised}"
        )

    labels = labels.reshape(labels.shape[0], -1)
    for t, hidden in exit_hidden_states.items():
        if hidden.shape[:2] != labels.shape:
            raise ValueError(
                f"exit {t} hidden states must have shape {tuple(labels.shape) + (hidden.shape[-1],)}, "
                f"got {tuple(hidden.shape)}"
            )

    shifted_labels, shifted_teacher = _teacher_for_sequence(
        labels, teacher, ignore_index
    )
    mask = shifted_labels != int(ignore_index)
    rows = torch.arange(labels.shape[0], device=labels.device).repeat_interleave(
        shifted_labels.shape[1]
    )[mask.reshape(-1)]
    n_rows = int(labels.shape[0])
    n_tokens = int(mask.sum().item())
    if n_tokens == 0:
        raise ValueError("dial_objective needs at least one assistant token")

    compact_teacher = TeacherTopK(
        probs=shifted_teacher.probs[mask.reshape(-1)],
        indices=shifted_teacher.indices[mask.reshape(-1)],
        tail_prob=shifted_teacher.tail_prob[mask.reshape(-1)],
    ).renormalized(lm_head_weight.shape[0])
    compact_hidden = {
        int(t): hidden[:, :-1].reshape(-1, hidden.shape[-1])[mask.reshape(-1)]
        for t, hidden in exit_hidden_states.items()
    }
    counts = _row_counts(rows, n_rows)
    valid_rows = counts > 0
    if not bool(valid_rows.any()):
        raise ValueError("dial_objective needs at least one row with assistant tokens")
    n_valid_rows = int(valid_rows.sum().item())

    parent_hidden = compact_hidden[1]
    lse_parent = chunked_logsumexp(
        parent_hidden,
        lm_head_weight,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )

    per_exit_loss: dict[int, torch.Tensor] = {}
    per_exit_kl_teacher: dict[int, torch.Tensor] = {}
    per_exit_kl_parent: dict[int, torch.Tensor] = {}
    per_exit_hinge: dict[int, torch.Tensor] = {}
    per_exit_token_sum: dict[int, torch.Tensor] = {}

    # Exit 1's teacher KL is only needed when pass 2 is in the window (the
    # predecessor of the first hinge).  Reuse the parent LSE: no second pass.
    if 2 in supervised:
        teacher_parent = chunked_teacher_kl(
            parent_hidden,
            lm_head_weight,
            compact_teacher,
            softcap=softcap,
            lse=lse_parent,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        parent_kl_parent = torch.zeros_like(teacher_parent)
        per_exit_loss[1] = _row_mean(
            teacher_parent.detach() + beta * parent_kl_parent, rows, counts
        )
        per_exit_kl_teacher[1] = _row_mean(teacher_parent, rows, counts)
        per_exit_kl_parent[1] = _row_mean(parent_kl_parent, rows, counts)

    for t in supervised:
        teacher_kl, parent_kl = chunked_exit_kl(
            parent_hidden,
            compact_hidden[t],
            lm_head_weight,
            compact_teacher,
            softcap=softcap,
            parent_lse=lse_parent,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        per_exit_kl_teacher[t] = _row_mean(teacher_kl, rows, counts)
        per_exit_kl_parent[t] = _row_mean(parent_kl, rows, counts)
        token_loss = teacher_kl + float(beta) * parent_kl
        per_exit_loss[t] = _row_mean(token_loss, rows, counts)
        per_exit_token_sum[t] = _row_sum(token_loss, rows, counts)

    for t in supervised:
        previous = t - 1
        if previous in per_exit_loss:
            per_exit_hinge[t] = monotone_hinge(
                per_exit_loss[t], per_exit_loss[previous], lambda_hinge
            )
        else:
            # The first exit of a window that starts above pass 2 has no
            # supervised predecessor: the design only adds the hinge when
            # t-1 is in the window or is exit 1.
            per_exit_hinge[t] = torch.zeros_like(per_exit_loss[t])

    mean_loss = torch.stack(
        [per_exit_loss[t][valid_rows] for t in supervised], dim=0
    ).mean(dim=0)
    mean_hinge = torch.stack(
        [per_exit_hinge[t][valid_rows] for t in supervised], dim=0
    ).sum(dim=0)
    # ``monotone_hinge`` already multiplies by lambda_hinge; do not apply it
    # a second time here.
    total = (mean_loss + mean_hinge).mean()

    window_size = len(supervised)
    mean_token_sum = torch.stack(
        [per_exit_token_sum[t] for t in supervised], dim=0
    ).sum()
    hinge_token_sum = torch.stack(
        [per_exit_hinge[t] * counts for t in supervised], dim=0
    ).sum()
    token_sum = mean_token_sum / float(window_size) + hinge_token_sum

    metrics: dict[str, float] = {
        "dial_total": float(total.detach().item()),
        "dial_tokens": int(n_tokens),
        "dial_horizon": int(max(supervised)),
        "dial_rows": n_valid_rows,
        "dial_token_sum": float(token_sum.detach().item()),
    }
    for t in supervised:
        metrics[f"dial_exit{t}_loss"] = float(
            per_exit_loss[t][valid_rows].detach().mean().item()
        )
        metrics[f"dial_exit{t}_kl_teacher"] = float(
            per_exit_kl_teacher[t][valid_rows].detach().mean().item()
        )
        metrics[f"dial_exit{t}_kl_parent"] = float(
            per_exit_kl_parent[t][valid_rows].detach().mean().item()
        )
        metrics[f"dial_exit{t}_hinge"] = float(
            per_exit_hinge[t][valid_rows].detach().mean().item()
        )
        metrics[f"dial_exit{t}_tokens"] = int(n_tokens)
    if 1 in per_exit_kl_teacher:
        metrics["dial_exit1_kl_teacher"] = float(
            per_exit_kl_teacher[1][valid_rows].detach().mean().item()
        )
        metrics["dial_exit1_tokens"] = int(n_tokens)

    return DialLossResult(
        total=total,
        per_exit_loss=per_exit_loss,
        per_exit_kl_teacher=per_exit_kl_teacher,
        per_exit_kl_parent=per_exit_kl_parent,
        per_exit_hinge=per_exit_hinge,
        metrics=metrics,
        per_exit_token_sum=per_exit_token_sum,
        row_token_counts=counts,
        token_sum=token_sum,
        window_size=window_size,
    )


def teacher_heal_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    teacher: TeacherTopK | None = None,
    parent_hidden: torch.Tensor | None = None,
    beta: float = DEFAULT_BETA,
    softcap: float | None = None,
    ignore_index: int = -100,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token non-dial (O-H) heal loss: teacher KL + beta * KL(parent||student).

    This is the single-exit version of the same objective L uses: the teacher
    contract is the shared unshifted ``[B, S, K]`` layout, the one shift is in
    :func:`_teacher_for_sequence`, and both KLs go through the same autograd
    kernel.  ``parent_hidden`` must be the adapter-off, N=1 hidden state
    (``[B, S, H]``); when omitted it defaults to ``hidden`` (the anchor term is
    then exactly zero, used only by direct unit tests).

    Returns the **per-token** tensor; the trainer reduces it by
    ``num_items_in_batch`` so the whole accumulation is a true token mean.
    """

    if hidden.ndim != 3:
        raise ValueError("hidden must be [B, S, H]")
    labels = labels.reshape(labels.shape[0], -1)
    if hidden.shape[:2] != labels.shape:
        raise ValueError(
            "hidden and labels must agree on [B, S]: "
            f"{tuple(hidden.shape[:2])} vs {tuple(labels.shape)}"
        )
    if parent_hidden is None:
        parent_hidden = hidden
    if parent_hidden.shape != hidden.shape:
        raise ValueError(
            "parent_hidden and hidden must have the same shape, got "
            f"{tuple(parent_hidden.shape)} and {tuple(hidden.shape)}"
        )
    shifted_labels, shifted_teacher = _teacher_for_sequence(
        labels, teacher, ignore_index
    )
    mask = shifted_labels != int(ignore_index)
    if int(mask.sum().item()) == 0:
        raise ValueError("teacher_heal_loss needs at least one assistant token")
    flat_mask = mask.reshape(-1)
    compact_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])[flat_mask]
    compact_parent = parent_hidden[:, :-1].reshape(-1, parent_hidden.shape[-1])[flat_mask]
    compact_teacher = TeacherTopK(
        probs=shifted_teacher.probs[flat_mask],
        indices=shifted_teacher.indices[flat_mask],
        tail_prob=shifted_teacher.tail_prob[flat_mask],
    ).renormalized(lm_head_weight.shape[0])
    with torch.no_grad():
        parent_lse = chunked_logsumexp(
            compact_parent,
            lm_head_weight,
            softcap=softcap,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
    teacher_kl, parent_kl = chunked_exit_kl(
        compact_parent,
        compact_hidden,
        lm_head_weight,
        compact_teacher,
        softcap=softcap,
        parent_lse=parent_lse,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    return teacher_kl + float(beta) * parent_kl
