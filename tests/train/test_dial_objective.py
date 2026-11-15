"""Unit tests for the multi-exit dial objective (Task T-dial-train).

The chunked kernels are compared against a naive full-vocabulary reference;
the horizon, window, hinge and the full objective are pinned independently of
the model.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from gemmaloop.train.dial_objective import (
    DIAL_LOOP_PROBS,
    TeacherTopK,
    chunked_exit_kl,
    chunked_logsumexp,
    chunked_teacher_kl,
    compute_window,
    dial_objective,
    monotone_hinge,
    sample_horizon,
    teacher_from_labels,
    teacher_from_topk_logits,
    window_bounds,
)


def test_horizon_sampler_matches_registered_distribution():
    draws = [sample_horizon(0, step) for step in range(20000)]
    assert set(draws) == {4, 5, 6, 7, 8}
    counts = Counter(draws)
    expected = {4: 0.70, 5: 0.075, 6: 0.075, 7: 0.075, 8: 0.075}
    for horizon, probability in expected.items():
        assert abs(counts[horizon] / 20000 - probability) < 0.01
    assert draws == [sample_horizon(0, step) for step in range(20000)]
    assert DIAL_LOOP_PROBS[1:3] == (0.0, 0.0)


@pytest.mark.parametrize(
    "total,expected",
    [(1, (2, 1)), (2, (2, 2)), (3, (2, 3)), (4, (2, 4)), (5, (3, 5)), (8, (6, 8))],
)
def test_window_is_the_last_three_trainable_passes(total, expected):
    start, end = window_bounds(total, 3)
    # For T < 2 the empty window is represented by start > end; the training
    # sampler never draws such horizons.
    if total < 2:
        assert start == 2 and end == total
    else:
        assert (start, end) == expected
        assert list(compute_window(total, 3)) == list(range(expected[0], expected[1] + 1))


def _random_teacher(n: int, vocab: int, k: int, generator: torch.Generator):
    indices = torch.stack(
        [torch.randperm(vocab, generator=generator)[:k] for _ in range(n)]
    )
    mass = torch.rand(n, k, generator=generator)
    mass = mass / mass.sum(dim=1, keepdim=True) * 0.82
    tail = (1.0 - mass.sum(dim=1)).clamp(min=0)
    return TeacherTopK(probs=mass, indices=indices, tail_prob=tail)


def _naive_teacher_kl(hidden, weight, teacher):
    logits = hidden @ weight.t()
    log_probs = torch.log_softmax(logits, dim=-1)
    q = log_probs.gather(1, teacher.indices)
    q_tail = 1.0 - q.exp().sum(dim=1)
    topk = (
        teacher.probs * (torch.log(teacher.probs.clamp(min=1e-30)) - q)
    ).sum(dim=1)
    tail = teacher.tail_prob * (
        torch.log(teacher.tail_prob.clamp(min=1e-30)) - torch.log(q_tail.clamp(min=1e-30))
    )
    return topk + tail


def test_chunked_teacher_kl_matches_naive_topk_plus_tail():
    generator = torch.Generator().manual_seed(0)
    n, hidden_size, vocab, k = 11, 6, 37, 5
    hidden = torch.randn(n, hidden_size, generator=generator)
    weight = torch.randn(vocab, hidden_size, generator=generator)
    teacher = _random_teacher(n, vocab, k, generator)

    got = chunked_teacher_kl(
        hidden, weight, teacher, token_chunk=4, vocab_chunk=7
    )
    torch.testing.assert_close(
        got, _naive_teacher_kl(hidden, weight, teacher), atol=1e-5, rtol=1e-5
    )


def test_chunked_parent_kl_matches_naive_and_reuses_parent_lse():
    generator = torch.Generator().manual_seed(1)
    n, hidden_size, vocab, k = 9, 5, 29, 4
    parent = torch.randn(n, hidden_size, generator=generator)
    exit_hidden = torch.randn(n, hidden_size, generator=generator)
    weight = torch.randn(vocab, hidden_size, generator=generator)
    teacher = _random_teacher(n, vocab, k, generator)

    parent_lse = chunked_logsumexp(parent, weight, token_chunk=3, vocab_chunk=6)
    teacher_kl, parent_kl = chunked_exit_kl(
        parent,
        exit_hidden,
        weight,
        teacher,
        parent_lse=parent_lse,
        token_chunk=3,
        vocab_chunk=6,
    )

    log_p = torch.log_softmax(parent @ weight.t(), dim=-1)
    log_e = torch.log_softmax(exit_hidden @ weight.t(), dim=-1)
    naive_parent = (log_p.exp() * (log_p - log_e)).sum(dim=1)
    torch.testing.assert_close(parent_kl, naive_parent, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        teacher_kl, _naive_teacher_kl(exit_hidden, weight, teacher), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(parent_lse, torch.logsumexp(parent @ weight.t(), dim=-1))


def test_monotone_hinge_has_stop_gradient_on_the_previous_exit():
    previous = torch.tensor(2.0, requires_grad=True)
    later = torch.tensor(2.5, requires_grad=True)
    hinge = monotone_hinge(later, previous, 1.0)
    torch.testing.assert_close(hinge, torch.tensor(0.5))
    hinge.backward()
    assert previous.grad is None
    assert later.grad is not None and float(later.grad) == 1.0

    better = torch.tensor(1.5, requires_grad=True)
    zero = monotone_hinge(better, torch.tensor(2.0), 1.0)
    assert float(zero.detach()) == 0.0


def test_teacher_from_topk_logits_is_normalised():
    teacher = teacher_from_topk_logits(
        torch.tensor([[2.0, 0.5, -1.0]]),
        torch.tensor([[3, 7, 1]]),
        torch.tensor([0.25]),
    )
    total = teacher.probs.sum(dim=1) + teacher.tail_prob
    torch.testing.assert_close(total, torch.ones(1))


def test_full_objective_matches_manual_row_means_and_hinge():
    generator = torch.Generator().manual_seed(2)
    batch, seq, hidden_size, vocab = 2, 5, 7, 23
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    labels[0, 3] = -100
    exits = {
        1: torch.randn(batch, seq, hidden_size, generator=generator),
        2: torch.randn(batch, seq, hidden_size, generator=generator),
        3: torch.randn(batch, seq, hidden_size, generator=generator),
        4: torch.randn(batch, seq, hidden_size, generator=generator),
    }
    beta = 0.1
    result = dial_objective(
        exits, weight, labels, beta=beta, window=3, token_chunk=2, vocab_chunk=5
    )

    shifted = labels[:, 1:]
    mask = shifted != -100
    rows = torch.arange(batch).repeat_interleave(seq - 1)[mask.reshape(-1)]
    teacher_token = teacher_from_labels(shifted.reshape(-1)[mask.reshape(-1)])

    def hidden_compact(exit_index):
        return exits[exit_index][:, :-1].reshape(-1, hidden_size)[mask.reshape(-1)]

    def row_mean(values):
        sums = torch.zeros(batch)
        sums.index_add_(0, rows, values)
        counts = torch.zeros(batch)
        counts.index_add_(0, rows, torch.ones_like(values))
        return sums / counts

    parent_compact = hidden_compact(1)
    parent_logp = torch.log_softmax(parent_compact @ weight.t(), dim=-1)
    teacher_parent = _naive_teacher_kl(parent_compact, weight, teacher_token)
    ell_1 = row_mean(teacher_parent)
    ell = {}
    for t in (2, 3, 4):
        compact = hidden_compact(t)
        teacher_kl = _naive_teacher_kl(compact, weight, teacher_token)
        exit_logp = torch.log_softmax(compact @ weight.t(), dim=-1)
        parent_kl = (parent_logp.exp() * (parent_logp - exit_logp)).sum(dim=1)
        ell[t] = row_mean(teacher_kl + beta * parent_kl)

    manual = (
        torch.stack([ell[2], ell[3], ell[4]]).mean(dim=0)
        + torch.relu(ell[2] - ell_1.detach())
        + torch.relu(ell[3] - ell[2].detach())
        + torch.relu(ell[4] - ell[3].detach())
    ).mean()

    torch.testing.assert_close(result.total, manual, atol=1e-5, rtol=1e-5)
    assert result.metrics["dial_tokens"] == int(mask.sum())
    for t in (2, 3, 4):
        assert f"dial_exit{t}_loss" in result.metrics
        assert f"dial_exit{t}_kl_teacher" in result.metrics
        assert f"dial_exit{t}_kl_parent" in result.metrics
        assert f"dial_exit{t}_hinge" in result.metrics
        assert result.metrics[f"dial_exit{t}_tokens"] == int(mask.sum())


def test_objective_requires_exit_one_and_the_contiguous_window():
    weight = torch.randn(17, 4)
    labels = torch.randint(0, 17, (1, 4))
    hidden = torch.randn(1, 4, 4)
    with pytest.raises(ValueError, match="exit 1"):
        dial_objective({2: hidden, 3: hidden, 4: hidden}, weight, labels, window=3)
    with pytest.raises(ValueError, match="last 3"):
        dial_objective(
            {1: hidden, 2: hidden, 4: hidden}, weight, labels, window=3
        )


def _bsk_teacher(batch, seq, vocab, k, generator, *, tail=0.1):
    indices = torch.stack(
        [torch.randperm(vocab, generator=generator)[:k] for _ in range(batch * seq)]
    ).reshape(batch, seq, k)
    probs = torch.rand(batch, seq, k, generator=generator)
    probs = probs / probs.sum(dim=-1, keepdim=True) * (1.0 - tail)
    probs = probs.to(torch.float16)
    tail_prob = torch.full((batch, seq), tail, dtype=torch.float32)
    return TeacherTopK(probs=probs, indices=indices.to(torch.int32), tail_prob=tail_prob)


def test_teacher_bsk_layout_fp16_tolerance_and_renormalisation():
    generator = torch.Generator().manual_seed(3)
    batch, seq, hidden_size, vocab, k = 2, 5, 6, 21, 3
    exits = {
        t: torch.randn(batch, seq, hidden_size, generator=generator)
        for t in (1, 2, 3, 4)
    }
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    teacher = _bsk_teacher(batch, seq, vocab, k, generator)

    # fp16 rows can deviate from 1 by a few 1e-4; validation and fp32
    # renormalisation must accept them.
    sums = (
        teacher.probs.to(torch.float32).sum(dim=-1) + teacher.tail_prob
    )
    assert (sums - 1.0).abs().max() < 2e-3
    flat = TeacherTopK(
        probs=teacher.probs.reshape(-1, k),
        indices=teacher.indices.reshape(-1, k),
        tail_prob=teacher.tail_prob.reshape(-1),
    )
    flat.validate(vocab)
    normalised = flat.renormalized(vocab)
    torch.testing.assert_close(
        normalised.probs.sum(dim=-1) + normalised.tail_prob,
        torch.ones(batch * seq),
    )

    with_teacher = dial_objective(
        exits, weight, labels, teacher=teacher, window=3, token_chunk=2, vocab_chunk=4
    )
    one_hot = dial_objective(
        exits, weight, labels, teacher=None, window=3, token_chunk=2, vocab_chunk=4
    )
    assert torch.isfinite(with_teacher.total)
    assert with_teacher.metrics["dial_exit4_kl_teacher"] != pytest.approx(
        one_hot.metrics["dial_exit4_kl_teacher"], abs=1e-3
    )


def test_hinge_lambda_is_applied_once():
    generator = torch.Generator().manual_seed(4)
    batch, seq, hidden_size, vocab = 2, 5, 6, 19
    exits = {
        t: torch.randn(batch, seq, hidden_size, generator=generator)
        for t in (1, 2, 3, 4)
    }
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    base = dial_objective(
        exits, weight, labels, lambda_hinge=0.0, window=3, token_chunk=2, vocab_chunk=4
    )
    once = dial_objective(
        exits, weight, labels, lambda_hinge=1.0, window=3, token_chunk=2, vocab_chunk=4
    )
    twice = dial_objective(
        exits, weight, labels, lambda_hinge=2.0, window=3, token_chunk=2, vocab_chunk=4
    )
    torch.testing.assert_close(
        twice.total - base.total, 2.0 * (once.total - base.total)
    )


def test_empty_rows_do_not_dilute_the_loss():
    generator = torch.Generator().manual_seed(5)
    _, seq, hidden_size, vocab = 2, 4, 6, 17
    exits = {
        t: torch.randn(2, seq, hidden_size, generator=generator)
        for t in (1, 2, 3, 4)
    }
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (2, seq), generator=generator)
    labels[0, :] = -100

    with_empty = dial_objective(
        exits, weight, labels, window=3, token_chunk=2, vocab_chunk=5
    )
    only_valid = dial_objective(
        {t: hidden[1:] for t, hidden in exits.items()},
        weight,
        labels[1:],
        window=3,
        token_chunk=2,
        vocab_chunk=5,
    )
    torch.testing.assert_close(with_empty.total, only_valid.total)
    assert with_empty.metrics["dial_rows"] == 1


def test_bsk_one_hot_teacher_is_shifted_exactly_once():
    generator = torch.Generator().manual_seed(6)
    batch, seq, hidden_size, vocab = 2, 5, 6, 17
    exits = {
        t: torch.randn(batch, seq, hidden_size, generator=generator)
        for t in (1, 2, 3, 4)
    }
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    teacher = TeacherTopK(
        probs=torch.ones(batch, seq, 1),
        indices=labels.unsqueeze(-1),
        tail_prob=torch.zeros(batch, seq),
    )
    with_teacher = dial_objective(
        exits, weight, labels, teacher=teacher, window=3, token_chunk=2, vocab_chunk=4
    )
    from_labels = dial_objective(
        exits, weight, labels, teacher=None, window=3, token_chunk=2, vocab_chunk=4
    )
    torch.testing.assert_close(with_teacher.total, from_labels.total)
    for t in (2, 3, 4):
        torch.testing.assert_close(
            with_teacher.per_exit_kl_teacher[t],
            from_labels.per_exit_kl_teacher[t],
        )


# ---------------------------------------------------------------------------
# Autograd kernels: gradients, saved memory, hinge and renormalisation
# ---------------------------------------------------------------------------


def _naive_pair(parent, exit_hidden, weight, teacher, softcap=None):
    def log_probs(hidden):
        logits = hidden @ weight.t()
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return torch.log_softmax(logits, dim=-1)

    log_p = log_probs(parent)
    log_e = log_probs(exit_hidden)
    p = teacher.probs.float()
    idx = teacher.indices.to(torch.long)
    tail = teacher.tail_prob.float()
    q_top = log_e.gather(1, idx)
    q_mass = q_top.exp().sum(dim=1)
    teacher_kl = (
        p * (torch.log(p.clamp(min=1e-30)) - q_top)
    ).sum(dim=1) + tail * (
        torch.log(tail.clamp(min=1e-30))
        - torch.log1p(-q_mass.clamp(max=1.0 - 1e-7))
    )
    parent_kl = (log_p.exp() * (log_p - log_e)).sum(dim=1)
    return teacher_kl, parent_kl


def _normalized_teacher(n, vocab, k, generator):
    teacher = _random_teacher(n, vocab, k, generator)
    return teacher.renormalized(vocab)


@pytest.mark.parametrize("softcap", [None, 3.0])
def test_chunked_exit_kl_gradients_match_the_naive_implementation(softcap):
    generator = torch.Generator().manual_seed(11)
    n, hidden_size, vocab, k = 7, 5, 19, 4
    parent = torch.randn(n, hidden_size, generator=generator, requires_grad=True)
    exit_hidden = torch.randn(n, hidden_size, generator=generator, requires_grad=True)
    weight = torch.randn(vocab, hidden_size, generator=generator)
    teacher = _normalized_teacher(n, vocab, k, generator)

    got_t, got_p = chunked_exit_kl(
        parent,
        exit_hidden,
        weight,
        teacher,
        softcap=softcap,
        token_chunk=3,
        vocab_chunk=6,
    )
    (got_t.sum() * 1.3 + got_p.sum() * 0.7).backward()

    parent_ref = parent.detach().clone().requires_grad_(True)
    exit_ref = exit_hidden.detach().clone().requires_grad_(True)
    ref_t, ref_p = _naive_pair(parent_ref, exit_ref, weight, teacher, softcap=softcap)
    (ref_t.sum() * 1.3 + ref_p.sum() * 0.7).backward()

    torch.testing.assert_close(parent.grad, parent_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(exit_hidden.grad, exit_ref.grad, atol=1e-5, rtol=1e-5)


def test_chunked_teacher_kl_gradient_matches_the_naive_implementation():
    generator = torch.Generator().manual_seed(12)
    n, hidden_size, vocab, k = 9, 4, 17, 3
    hidden = torch.randn(n, hidden_size, generator=generator, requires_grad=True)
    weight = torch.randn(vocab, hidden_size, generator=generator)
    teacher = _normalized_teacher(n, vocab, k, generator)

    got = chunked_teacher_kl(
        hidden, weight, teacher, softcap=2.0, token_chunk=4, vocab_chunk=5
    )
    got.sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    logits = torch.tanh((hidden_ref @ weight.t()) / 2.0) * 2.0
    log_q = torch.log_softmax(logits, dim=-1)
    p = teacher.probs.float()
    idx = teacher.indices.to(torch.long)
    tail = teacher.tail_prob.float()
    q_top = log_q.gather(1, idx)
    q_mass = q_top.exp().sum(dim=1)
    ref = (
        p * (torch.log(p.clamp(min=1e-30)) - q_top)
    ).sum(dim=1) + tail * (
        torch.log(tail.clamp(min=1e-30))
        - torch.log1p(-q_mass.clamp(max=1.0 - 1e-7))
    )
    ref.sum().backward()
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, atol=1e-5, rtol=1e-5)


def test_objective_saved_tensors_are_o_tokens_not_o_tokens_times_vocab():
    n, hidden_size, k = 256, 8, 8

    def saved_bytes(vocab):
        generator = torch.Generator().manual_seed(13)
        parent = torch.randn(n, hidden_size, generator=generator)
        exit_hidden = torch.randn(n, hidden_size, generator=generator, requires_grad=True)
        weight = torch.randn(vocab, hidden_size, generator=generator)
        teacher = _normalized_teacher(n, vocab, k, generator)
        total = [0]

        def pack(tensor):
            total[0] += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            teacher_kl, parent_kl = chunked_exit_kl(
                parent,
                exit_hidden,
                weight,
                teacher,
                token_chunk=64,
                vocab_chunk=512,
            )
            (teacher_kl.sum() + parent_kl.sum()).backward()
        return total[0]

    small = saved_bytes(2048)
    large = saved_bytes(8192)
    # One fp32 [N, V] tile at V=8192 would be 8 MB; the kernel must stay far
    # below that and must not grow linearly with V.
    assert large < n * 8192 * 4 / 8
    assert large < 4.0 * small


def test_hinge_uses_exit_one_as_the_predecessor_of_exit_two():
    vocab = 7
    weight = torch.eye(vocab)
    labels = torch.tensor([[-100, 1, 2]])
    teacher = TeacherTopK(
        probs=torch.ones(1, 3, 1),
        indices=labels.unsqueeze(-1).clamp(min=0),
        tail_prob=torch.zeros(1, 3),
    )
    good = torch.stack([weight[1] * 8.0, weight[2] * 8.0, weight[2] * 8.0]).unsqueeze(0)
    bad = -good
    exits = {1: good, 2: bad, 3: bad, 4: bad}
    result = dial_objective(
        exits, weight, labels, teacher=teacher, window=3, token_chunk=2, vocab_chunk=3
    )
    assert float(result.per_exit_loss[2][0]) > float(result.per_exit_loss[1][0])
    assert float(result.per_exit_hinge[2][0]) > 0.0
    manual = (
        torch.stack([result.per_exit_loss[t] for t in (2, 3, 4)]).mean()
        + result.per_exit_hinge[2]
        + result.per_exit_hinge[3]
        + result.per_exit_hinge[4]
    )
    torch.testing.assert_close(result.total, manual.mean())


def test_in_tolerance_teacher_scaling_is_renormalised():
    generator = torch.Generator().manual_seed(14)
    batch, seq, hidden_size, vocab = 2, 4, 5, 13
    exits = {
        t: torch.randn(batch, seq, hidden_size, generator=generator)
        for t in (1, 2, 3, 4)
    }
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    base = _normalized_teacher(batch * seq, vocab, 3, generator)
    base_bsk = TeacherTopK(
        probs=base.probs.reshape(batch, seq, -1),
        indices=base.indices.reshape(batch, seq, -1),
        tail_prob=base.tail_prob.reshape(batch, seq),
    )
    scaled = TeacherTopK(
        probs=base_bsk.probs.float() * 1.001,
        indices=base_bsk.indices,
        tail_prob=base_bsk.tail_prob.float() * 1.001,
    )
    reference = dial_objective(exits, weight, labels, teacher=base_bsk, window=3)
    rescaled = dial_objective(exits, weight, labels, teacher=scaled, window=3)
    torch.testing.assert_close(reference.total, rescaled.total, atol=1e-7, rtol=1e-7)
