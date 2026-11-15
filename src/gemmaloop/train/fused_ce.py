"""Memory-lean fused LM head + cross-entropy for Gemma 4 SFT without Triton.

TRL 1.17.1's ``SFTTrainer`` already trains through a fused LM head
(:func:`trl.trainer.utils.add_fused_lm_head`): the ``[seq, vocab]`` logits are
never materialised and the loss is ``-sum(log p(label)) / n_label_tokens``.
Without Triton (Windows) the per-tile work falls back to plain torch ops on
``[min(seq, 4096), 32768]`` tiles that keep about four (forward) to nine
(backward) fp32 copies of the tile alive at once: ~2.2 GiB forward and ~4.5 GiB
backward at 4096 tokens, which caps the trainable sequence length.

:class:`ChunkedSoftcapLogProb` computes exactly the same quantities as TRL's
``ChunkedLogProbFunction`` for this project's case (no temperature, no logit
scale, Gemma's ``final_logit_softcapping``, no support/score ids):

* the tile is the same bf16 GEMM ``hidden @ weight.T`` (logits rounded to the
  compute dtype, as in TRL and in the plain transformers forward);
* softcapping, the online log-sum-exp and the target-excluding normalisation
  ``log p_t = -softplus(m - z_t + log sum_{j != t} exp(z_j - m))`` are TRL's;
* the backward recomputes each tile, forms ``d log p_t / dz`` with the target
  entry ``-expm1(log p_t)``, applies the softcap chain rule in fp32, rounds the
  tile gradient to the compute dtype and accumulates ``grad_hidden`` in fp32.

The differences are purely about memory and memory traffic: smaller tiles
(``token_chunk x vocab_chunk``), in-place fp32 ops (two fp32 tile buffers at
most), and only labelled positions are projected (``compact``).  The
``entropy`` and ``is_top1`` outputs feed TRL's logged metrics and carry no
gradient.

:func:`install_lean_lm_head` re-patches the model's forward that TRL patched,
returning TRL's own :class:`~trl.trainer.utils.FusedCausalLMOutput`, so
``SFTTrainer.compute_loss`` is unchanged.
"""

from __future__ import annotations

import functools
import types

import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_TOKEN_CHUNK",
    "DEFAULT_VOCAB_CHUNK",
    "ChunkedSoftcapLogProb",
    "chunked_softcap_logprobs",
    "install_lean_lm_head",
]

DEFAULT_TOKEN_CHUNK = 2048
DEFAULT_VOCAB_CHUNK = 16384

# ``torch.addmm(..., out_dtype=torch.float32)`` from bf16/fp16 inputs (CUDA, torch >= 2.8)
_MM_OUT_DTYPE = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 8)


def _addmm_fp32(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """``acc += a @ b`` into an fp32 ``acc`` without rounding ``a @ b`` first (as TRL)."""
    if _MM_OUT_DTYPE and acc.device.type == "cuda" and a.dtype in (torch.float16, torch.bfloat16):
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    else:
        acc.addmm_(a.float(), b.float())


def _project(hidden: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """bf16 (compute dtype) logits tile ``hidden @ weight.T`` written into ``out``."""
    torch.mm(hidden, weight.t(), out=out)
    return out


class ChunkedSoftcapLogProb(torch.autograd.Function):
    """Per-token ``log p(target)`` of ``softcap(hidden @ weight.T)``, tile by tile.

    Inputs: ``hidden`` ``[N, H]`` and ``weight`` ``[V, H]`` in the compute dtype,
    ``targets`` ``[N]`` int64 in ``[0, V)``.  Returns ``(log_probs, entropy,
    is_top1)``; ``entropy``/``is_top1`` are ``None`` unless requested and never
    receive a gradient.  Only ``hidden`` gets a gradient (the LM head is tied to
    the frozen input embeddings).
    """

    @staticmethod
    def forward(  # noqa: PLR0913 - mirrors the autograd signature
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        softcap: float | None,
        token_chunk: int,
        vocab_chunk: int,
        want_entropy: bool,
        want_top1: bool,
    ):
        ctx.set_materialize_grads(False)
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        device = hidden.device
        f32 = torch.float32

        log_probs = torch.empty(n_tokens, device=device, dtype=f32)
        log_z = torch.empty(n_tokens, device=device, dtype=f32)
        entropy = torch.empty(n_tokens, device=device, dtype=f32) if want_entropy else None
        is_top1 = torch.empty(n_tokens, device=device, dtype=torch.bool) if want_top1 else None

        rows = min(n_tokens, token_chunk)
        width = min(vocab, vocab_chunk)
        tile_buf = torch.empty((rows, width), device=device, dtype=hidden.dtype)
        z_buf = torch.empty((rows, width), device=device, dtype=f32)
        e_buf = torch.empty((rows, width), device=device, dtype=f32)

        for t0 in range(0, n_tokens, token_chunk):
            t1 = min(t0 + token_chunk, n_tokens)
            n = t1 - t0
            h = hidden[t0:t1]
            tgt = targets[t0:t1]
            running_max = torch.full((n,), float("-inf"), device=device, dtype=f32)
            sum_exp = torch.zeros(n, device=device, dtype=f32)
            other_sum_exp = torch.zeros(n, device=device, dtype=f32)
            x_sum_exp = torch.zeros(n, device=device, dtype=f32) if want_entropy else None
            target_logit = torch.zeros(n, device=device, dtype=f32)
            best = torch.full((n,), float("-inf"), device=device, dtype=f32)
            best_idx = torch.zeros(n, device=device, dtype=torch.long)

            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                c = v1 - v0
                tile = _project(h, weight[v0:v1], tile_buf[:n, :c])
                z = z_buf[:n, :c]
                z.copy_(tile)
                if softcap is not None:
                    z.div_(softcap).tanh_().mul_(softcap)

                local = tgt - v0
                in_tile = (local >= 0) & (local < c)
                local = local.clamp(0, c - 1).unsqueeze(1)
                zt = z.gather(1, local).squeeze(1)
                target_logit = torch.where(in_tile, zt, target_logit)

                tile_max, tile_arg = z.max(dim=1)
                # strictly greater keeps the first maximum across tiles (argmax semantics)
                better = tile_max > best
                best_idx = torch.where(better, tile_arg + v0, best_idx)
                best = torch.where(better, tile_max, best)

                # Same reductions, in the same order, as TRL's ``_forward_tile``.
                new_max = torch.maximum(running_max, tile_max)
                alpha = torch.exp(running_max - new_max)
                e = e_buf[:n, :c]
                torch.sub(z, new_max.unsqueeze(1), out=e).exp_()  # e_j = exp(z_j - m)
                sum_exp = sum_exp * alpha + e.sum(dim=1)
                if want_entropy:
                    # z is no longer needed: z <- e * z (absolute logits), as TRL's (e * z)
                    x_sum_exp = x_sum_exp * alpha + z.mul_(e).sum(dim=1)
                # exclude the target before summing (TRL's target-excluding normalisation)
                kept = torch.where(in_tile, 0.0, e.gather(1, local).squeeze(1))
                e.scatter_(1, local, kept.unsqueeze(1))
                other_sum_exp = other_sum_exp * alpha + e.sum(dim=1)
                running_max = new_max

            lz = running_max + torch.log(sum_exp)
            log_z[t0:t1] = lz
            log_probs[t0:t1] = -F.softplus((running_max - target_logit) + torch.log(other_sum_exp))
            if want_entropy:
                entropy[t0:t1] = lz - x_sum_exp / sum_exp
            if want_top1:
                is_top1[t0:t1] = best_idx == tgt

        ctx.save_for_backward(hidden, weight, targets, log_z, log_probs.clone())
        ctx.softcap = softcap
        ctx.token_chunk = int(token_chunk)
        ctx.vocab_chunk = int(vocab_chunk)
        if entropy is not None:
            ctx.mark_non_differentiable(entropy)
        if is_top1 is not None:
            ctx.mark_non_differentiable(is_top1)
        return log_probs, entropy, is_top1

    @staticmethod
    def backward(ctx, grad_log_probs, _grad_entropy, _grad_top1):  # type: ignore[override]
        hidden, weight, targets, log_z, log_probs = ctx.saved_tensors
        if grad_log_probs is None or not ctx.needs_input_grad[0]:
            return (None,) * 8
        softcap = ctx.softcap
        token_chunk, vocab_chunk = ctx.token_chunk, ctx.vocab_chunk
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        device = hidden.device
        f32 = torch.float32

        grad_hidden = torch.zeros(hidden.shape, device=device, dtype=f32)
        grad_lp = grad_log_probs.float().contiguous()
        rows = min(n_tokens, token_chunk)
        width = min(vocab, vocab_chunk)
        tile_buf = torch.empty((rows, width), device=device, dtype=hidden.dtype)
        a_buf = torch.empty((rows, width), device=device, dtype=f32)
        b_buf = torch.empty((rows, width), device=device, dtype=f32)

        for t0 in range(0, n_tokens, token_chunk):
            t1 = min(t0 + token_chunk, n_tokens)
            n = t1 - t0
            h = hidden[t0:t1]
            tgt = targets[t0:t1]
            g = grad_lp[t0:t1]
            lz = log_z[t0:t1]
            # d log p_t / dz_t = 1 - p_t, computed without cancellation
            target_grad = g * -torch.expm1(log_probs[t0:t1])
            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                c = v1 - v0
                w = weight[v0:v1]
                tile = _project(h, w, tile_buf[:n, :c])
                t = a_buf[:n, :c]
                t.copy_(tile)
                p = b_buf[:n, :c]
                if softcap is not None:
                    t.div_(softcap).tanh_()
                    torch.mul(t, softcap, out=p)
                else:
                    p.copy_(t)
                # p_j = exp(z_j - log Z);  grad_z_j = -g * p_j  (target fixed below)
                p.sub_(lz.unsqueeze(1)).exp_().mul_(-g.unsqueeze(1))
                local = tgt - v0
                in_tile = (local >= 0) & (local < c)
                local = local.clamp(0, c - 1).unsqueeze(1)
                current = p.gather(1, local).squeeze(1)
                p.scatter_(1, local, torch.where(in_tile, target_grad, current).unsqueeze(1))
                if softcap is not None:
                    # d softcap(y) / dy = 1 - tanh(y / softcap)^2
                    t.mul_(t).neg_().add_(1.0)
                    p.mul_(t)
                tile.copy_(p)
                _addmm_fp32(grad_hidden[t0:t1], tile, w)

        return grad_hidden.to(hidden.dtype), None, None, None, None, None, None, None


def chunked_softcap_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None,
    *,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
    want_entropy: bool = True,
    want_top1: bool = True,
):
    """Functional wrapper around :class:`ChunkedSoftcapLogProb`."""
    if token_chunk < 1 or vocab_chunk < 1:
        raise ValueError("token_chunk and vocab_chunk must be positive")
    return ChunkedSoftcapLogProb.apply(
        hidden,
        weight,
        targets,
        None if softcap is None else float(softcap),
        int(token_chunk),
        int(vocab_chunk),
        bool(want_entropy),
        bool(want_top1),
    )


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def install_lean_lm_head(
    model: torch.nn.Module,
    *,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
    compact: bool = True,
) -> torch.nn.Module:
    """Replace TRL's fused LM head on ``model`` with :class:`ChunkedSoftcapLogProb`.

    ``model`` is the (PEFT-wrapped) model held by the ``SFTTrainer``.  The
    forward keeps TRL's contract for ``fused_lm_head=True`` (labels shifted by
    one, ``-100`` ignored, loss divided by ``num_items_in_batch`` or by the
    number of labelled tokens, metrics ``entropy``/``is_top1``) and falls back to
    the original forward otherwise, so generation and plain calls are
    unchanged.  With ``compact`` only labelled positions are projected through
    the LM head; the per-token results are scattered back to ``[batch, seq-1]``.
    """
    from trl.trainer.utils import FusedCausalLMOutput

    base = _base_model(model)
    text_config = base.config.get_text_config()
    softcap = getattr(text_config, "final_logit_softcapping", None)
    for name in ("logit_scale", "output_multiplier", "lm_head_multiplier", "logits_scaling"):
        if getattr(text_config, name, None) not in (None, 1, 1.0):
            raise ValueError(f"install_lean_lm_head does not support {name}={getattr(text_config, name)!r}")
    if getattr(text_config, "output_router_logits", False):
        raise ValueError("install_lean_lm_head does not support MoE router losses")

    original_forward = types.MethodType(type(base).forward, base)

    @functools.wraps(type(base).forward)
    def _lean_forward(
        self,
        *args,
        fused_lm_head=False,
        labels=None,
        shift_labels=None,
        num_items_in_batch=None,
        support_ids=None,
        score_ids=None,
        **kwargs,
    ):
        if not fused_lm_head:
            if labels is not None:
                kwargs["labels"] = labels
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return original_forward(*args, **kwargs)
        if support_ids is not None or score_ids is not None:
            raise ValueError("the lean LM head does not support support_ids/score_ids")

        kwargs["use_cache"] = False
        backbone_outputs = self.base_model(*args, **kwargs)
        if shift_labels is None:
            hidden_states = backbone_outputs.last_hidden_state[:, :-1]
            labels = labels[:, 1:]
        else:
            hidden_states = backbone_outputs.last_hidden_state
            labels = shift_labels
        lm_head = self.get_output_embeddings()
        weight = lm_head.weight
        if weight.requires_grad and torch.is_grad_enabled():
            # ChunkedSoftcapLogProb only returns grad_hidden: a trainable
            # (or tied-and-trainable) LM head would silently get no gradient.
            raise ValueError("the lean LM head requires a frozen LM-head weight")
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("the lean LM head does not support an LM-head bias")
        hidden_states = hidden_states.to(weight.device)
        labels = labels.to(weight.device)
        mask = labels != -100

        compute_dtype = (
            torch.get_autocast_dtype(weight.device.type)
            if torch.is_autocast_enabled(weight.device.type)
            else hidden_states.dtype
        )
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_labels = labels.reshape(-1)
        flat_mask = mask.reshape(-1)
        if compact:
            flat_hidden = flat_hidden[flat_mask]
            flat_labels = flat_labels[flat_mask]
        else:
            flat_labels = flat_labels.clamp(min=0)

        log_probs, entropy, is_top1 = chunked_softcap_logprobs(
            flat_hidden.to(compute_dtype),
            weight.to(compute_dtype),
            flat_labels,
            softcap,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            want_entropy=True,
            want_top1=True,
        )

        def _expand(values: torch.Tensor) -> torch.Tensor:
            if compact:
                # masked_scatter keeps an all-masked micro batch connected to the graph
                return values.new_zeros(mask.shape).masked_scatter(mask, values)
            return values.view(mask.shape).masked_fill(~mask, 0)

        log_probs = _expand(log_probs)
        entropy = _expand(entropy)
        is_top1 = _expand(is_top1)
        denominator = mask.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch
        loss = -log_probs.sum() / denominator
        return FusedCausalLMOutput(
            loss=loss,
            log_probs=log_probs,
            entropy=entropy,
            is_top1=is_top1,
            label_mask=mask,
        )

    base.forward = types.MethodType(_lean_forward, base)
    base._gemmaloop_lean_lm_head = {
        "token_chunk": int(token_chunk),
        "vocab_chunk": int(vocab_chunk),
        "compact": bool(compact),
    }
    return model
