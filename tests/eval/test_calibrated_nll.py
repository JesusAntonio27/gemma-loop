"""Temperature-calibrated NLL helper."""

import pytest
import torch

from gemmaloop.eval.calibrated_nll import (
    BUCKET_LABELS,
    _chunked_nll_top1,
    OVERFLOW_BUCKET,
    bucket_report,
    calibrated_nll_report,
    fit_temperature_from_logits,
    per_token_nll_from_logits,
    position_bucket,
    score_model,
    top1_from_logits,
)
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from tests.model.tiny import tiny_text_model


def _synthetic(n=20000, vocab=32, temperature=1.7):
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(n, vocab, generator=generator)
    # Sample targets from softmax(logits / temperature), then hand the model the
    # unscaled logits: the NLL-minimising temperature is the data temperature.
    targets = torch.distributions.Categorical(
        logits=logits / temperature
    ).sample()
    return logits, targets


def test_position_buckets_are_inclusive_lower_bounds():
    assert [position_bucket(p) for p in (0, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192)] == [
        "<=1k",
        "<=1k",
        "1-2k",
        "1-2k",
        "2-4k",
        "2-4k",
        "4-8k",
        "4-8k",
        OVERFLOW_BUCKET,
    ]


def test_temperature_fit_recovers_the_scale_and_improves_nll():
    logits, targets = _synthetic(temperature=1.7)
    fitted = fit_temperature_from_logits(logits, targets, bounds=(0.5, 4.0))
    assert fitted == pytest.approx(1.7, abs=0.08)
    raw = per_token_nll_from_logits(logits, targets, 1.0)
    calibrated = per_token_nll_from_logits(logits, targets, fitted)
    assert calibrated.mean() < raw.mean()
    # Top-1 is unchanged by a positive temperature.
    assert torch.equal(top1_from_logits(logits, targets), top1_from_logits(logits / fitted, targets))


def test_bucket_report_counts_and_means():
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    calibrated = raw - 0.5
    top1 = torch.tensor([True, False, True, True, False])
    positions = torch.tensor([0, 1023, 1024, 4095, 9000])
    report = bucket_report(raw, calibrated, top1, positions)
    assert report["tokens"] == 5
    assert report["raw_nll"] == pytest.approx(3.0)
    assert report["calibrated_nll"] == pytest.approx(2.5)
    assert report["top1_accuracy"] == pytest.approx(0.6)
    assert report["buckets"]["<=1k"]["tokens"] == 2
    assert report["buckets"]["1-2k"]["tokens"] == 1
    assert report["buckets"]["2-4k"]["tokens"] == 1
    assert report["buckets"]["4-8k"] is None
    assert report["buckets"][OVERFLOW_BUCKET]["tokens"] == 1


def test_chunked_lean_head_matches_full_logits_with_softcap():
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(11, 16, generator=generator)
    weight = torch.randn(29, 16, generator=generator)
    targets = torch.randint(0, 29, (11,), generator=generator)
    softcap = 30.0
    temperature = 1.35
    nll, top1 = _chunked_nll_top1(
        hidden,
        weight,
        targets,
        softcap=softcap,
        temperature=temperature,
        token_chunk=4,
        vocab_chunk=7,
    )
    logits = torch.tanh((hidden @ weight.t()) / softcap) * softcap
    expected = per_token_nll_from_logits(logits, targets, temperature)
    torch.testing.assert_close(nll, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(top1, top1_from_logits(logits, targets))


def _batches(seed=0, rows=2, length=40):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (rows, length), generator=generator)
    labels = ids.clone()
    labels[:, :5] = -100
    mask = torch.ones_like(ids)
    return [{"input_ids": ids, "attention_mask": mask, "labels": labels}]


def test_model_scoring_works_looped_and_unlooped_at_a_given_n():
    model = tiny_text_model()
    plain_one = score_model(model, _batches(), n_loops=1)
    plain_three = score_model(model, _batches(), n_loops=3)
    assert torch.equal(plain_one.nll, plain_three.nll)

    looped = tiny_text_model()
    plan = make_layer_plan(looped.config.to_dict(), block=(5, 6))
    apply_loops(looped, plan, LoopConfig(n_loops=1, bridge="gch"))
    one = score_model(looped, _batches(), n_loops=1)
    assert looped._gemmaloop_n_loops == 1
    assert len(one) == 2 * (40 - 1 - 4)
    with torch.no_grad():
        looped.loop_bridge_g.normal_()
    two = score_model(looped, _batches(), n_loops=2)
    assert not torch.equal(one.nll, two.nll)
    # The helper restores the caller's N.
    assert looped._gemmaloop_n_loops == 1


def test_calibrated_report_on_a_looped_model():
    model = tiny_text_model()
    plan = make_layer_plan(model.config.to_dict(), block=(5, 6))
    apply_loops(model, plan, LoopConfig(n_loops=1, bridge="gch"))
    report = calibrated_nll_report(
        model,
        _batches(seed=1),
        _batches(seed=2),
        n_loops=2,
        bounds=(0.7, 1.4),
        max_iter=20,
        tol=1e-2,
    )
    assert 0.7 <= report["temperature"] <= 1.4
    assert report["n_loops"] == 2
    assert set(BUCKET_LABELS).issubset(report["buckets"])
    assert report["tokens"] > 0
    assert report["top1_accuracy"] == pytest.approx(
        sum(
            entry["tokens"] * entry["top1_accuracy"]
            for entry in report["buckets"].values()
            if entry is not None
        )
        / report["tokens"]
    )


def test_left_padding_positions_and_buckets_follow_the_mask():
    model = tiny_text_model()
    # Two rows with one and two left-pad tokens.
    input_ids = torch.tensor(
        [
            [0, 5, 6, 7, 8],
            [0, 0, 9, 10, 11],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    scores = score_model(
        model,
        [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        ],
        n_loops=1,
    )
    assert scores.positions.tolist() == [0, 1, 2, 3, 0, 1, 2]


def test_chunked_head_explicitly_casts_mixed_hidden_and_weight_dtypes():
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(7, 16, generator=generator)  # fp32
    weight = torch.randn(29, 16, generator=generator).to(torch.bfloat16)
    targets = torch.randint(0, 29, (7,), generator=generator)
    nll, top1 = _chunked_nll_top1(
        hidden,
        weight,
        targets,
        softcap=None,
        temperature=1.0,
        token_chunk=3,
        vocab_chunk=5,
    )
    assert torch.isfinite(nll).all()
    logits = hidden @ weight.float().t()
    expected = per_token_nll_from_logits(logits, targets, 1.0)
    torch.testing.assert_close(nll, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(top1, top1_from_logits(logits, targets))


def test_score_model_on_a_bf16_tiny_model():
    model = tiny_text_model().to(torch.bfloat16)
    scores = score_model(model, _batches(), n_loops=1)
    assert len(scores) > 0
    assert torch.isfinite(scores.nll).all()


def test_chunked_head_memory_does_not_scale_with_sequence_length():
    from torch.profiler import ProfilerActivity, profile

    tokens, vocab, hidden = 8192, 2048, 64
    generator = torch.Generator().manual_seed(0)
    h = torch.randn(tokens, hidden, generator=generator)
    weight = torch.randn(vocab, hidden, generator=generator)
    targets = torch.randint(0, vocab, (tokens,), generator=generator)

    def max_matmul_bytes(length: int) -> int:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            _chunked_nll_top1(
                h[:length],
                weight,
                targets[:length],
                softcap=None,
                temperature=1.0,
                token_chunk=512,
                vocab_chunk=512,
            )
        values = [
            event.cpu_memory_usage
            for event in prof.events()
            if event.key in {"aten::mm", "aten::matmul"}
            and event.cpu_memory_usage > 0
        ]
        assert values
        return max(values)

    short = max_matmul_bytes(512)
    long = max_matmul_bytes(tokens)
    assert short <= 512 * 512 * 4
    assert long <= 512 * 512 * 4
    assert long < tokens * vocab * 4 // 4
