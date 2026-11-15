"""Tests for token-weighted mixing, prefix stability and the manifest."""

from __future__ import annotations

import hashlib

import pytest

from gemmaloop.data.decontam import EvalIndex
from gemmaloop.data.mix import build_mix, write_jsonl
from gemmaloop.data.sources import SOURCES, SourceSpec

EVAL = [
    "Given an array of n integers, find the maximum sum of a contiguous "
    "subarray of length exactly k and print it."
]


class WordTokenizer:
    """Minimal tokenizer used to make token accounting explicit in tests."""

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return text.split()


def _rows(prefix: str, count: int = 400):
    return [
        {
            "input": f"{prefix} question {i} " + "alpha beta gamma delta " * 20,
            "output": f"{prefix} answer {i} " + "one two three four " * 20,
        }
        for i in range(count)
    ]


def _loader(rows):
    def load():
        return iter(list(rows))

    return load


def _sources(alpha_rows=None, beta_rows=None):
    return {
        "alpha": SourceSpec(
            hf_id="fake/alpha",
            config=None,
            split="train",
            license="MIT",
            weight=0.5,
            loader=_loader(alpha_rows if alpha_rows is not None else _rows("alpha")),
        ),
        "beta": SourceSpec(
            hf_id="fake/beta",
            config=None,
            split="train",
            license="Apache 2.0",
            weight=0.5,
            loader=_loader(beta_rows if beta_rows is not None else _rows("beta")),
        ),
    }


def _manifest_sources():
    return _sources()


def test_token_proportions_match_weights_within_one_point():
    rows, manifest = build_mix(
        target_tokens=20_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["Completely unrelated evaluation text."]),
        seed=0,
        sources=_sources(),
    )
    total = sum(manifest.tokens_per_source.values())
    assert total > 0
    for name, weight in (("alpha", 0.5), ("beta", 0.5)):
        share = manifest.tokens_per_source[name] / total
        assert abs(share - weight) <= 0.01
    assert sum(manifest.rows_per_source.values()) == len(rows)
    assert manifest.licenses == {"alpha": "MIT", "beta": "Apache 2.0"}


def test_manifest_sha_is_stable_for_same_seed():
    kwargs = dict(
        target_tokens=5_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["Completely unrelated evaluation text."]),
        seed=7,
        sources=_manifest_sources(),
    )
    rows_a, manifest_a = build_mix(**kwargs)
    rows_b, manifest_b = build_mix(**kwargs)
    assert manifest_a.sha256 == manifest_b.sha256
    assert rows_a == rows_b


def test_manifest_sha_is_hash_of_written_file(tmp_path):
    rows, manifest = build_mix(
        target_tokens=5_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["Completely unrelated evaluation text."]),
        seed=3,
        sources=_sources(),
    )
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest.sha256


def test_bigger_target_is_an_exact_prefix():
    kwargs = dict(
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["Completely unrelated evaluation text."]),
        seed=11,
        sources=_sources(),
    )
    small, _ = build_mix(4_000, **kwargs)
    large, _ = build_mix(8_000, **kwargs)
    assert len(small) < len(large)
    assert large[: len(small)] == small


def test_contaminated_rows_are_counted_in_dropped():
    alpha = _rows("alpha")
    alpha[0] = {"input": EVAL[0], "output": "contaminated " * 100}
    beta = _rows("beta")
    beta[1] = {"input": "Intro. " + EVAL[0][:-10] + " please.", "output": "x " * 100}

    _, manifest = build_mix(
        target_tokens=10_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(EVAL),
        seed=0,
        sources=_sources(alpha_rows=alpha, beta_rows=beta),
    )

    assert manifest.dropped
    assert manifest.dropped.get("exact", 0) >= 1
    assert any(
        manifest.dropped.get(criterion, 0) >= 1
        for criterion in ("ngram13", "indel75")
    )


def test_real_sources_contract():
    assert set(SOURCES) == {"ocr", "oci", "omr", "ot3", "chat"}
    assert sum(spec.weight for spec in SOURCES.values()) == 1.0
    assert SOURCES["chat"].license in {"Apache 2.0", "MIT", "CC BY 4.0"}
    for spec in SOURCES.values():
        assert spec.hf_id and spec.split and spec.license
        assert spec.revision and spec.provenance


# ---------------------------------------------------------------------------
# Reasoning defects (audit finding 1)
# ---------------------------------------------------------------------------
def _conversation(answer: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": answer},
        ]
    }


def _ot3_source(rows):
    return {
        "ot3": SourceSpec(
            hf_id="fake/ot3",
            config=None,
            split="train",
            license="Apache 2.0",
            weight=1.0,
            loader=_loader(rows),
        )
    }


def test_unterminated_reasoning_and_empty_answer_are_dropped():
    rows = [
        _conversation("<think" + ">" + "truncated plan"),
        _conversation("<think" + ">" + "plan</think" + ">"),
        _conversation("<think" + ">" + "</think" + ">"),
        _conversation("<think" + ">" + "plan</think" + ">" + "answer"),
        _conversation("<|channel>thought\nunclosed channel"),
    ]
    out, manifest = build_mix(
        target_tokens=1_000_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=_ot3_source(rows),
    )
    assert manifest.dropped["unterminated_think"] == 2
    assert manifest.dropped["empty_answer"] == 2
    assert manifest.rows_per_source["ot3"] == 1
    assert len(out) == 1
    # Internal defect markers never leak into the training rows.
    assert all(
        "_data_defect" not in message
        for row in out
        for message in row["messages"]
    )


def test_rows_without_reasoning_are_not_empty_answer_drops():
    rows = [
        _conversation("plain answer"),
        _conversation(""),
    ]
    out, manifest = build_mix(
        target_tokens=1_000_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=_ot3_source(rows),
    )
    # The reasoning-specific rule must not touch rows without a thinking trace.
    assert manifest.dropped.get("empty_answer", 0) == 0
    assert manifest.dropped.get("unterminated_think", 0) == 0
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Length-aware targets (audit finding 6)
# ---------------------------------------------------------------------------
def _fake_assistant_labels(tokenizer, messages, max_length, tools=None):
    """Deterministic render length: one token per word of content/reasoning."""
    length = 0
    for message in messages:
        length += len(str(message.get("content") or "").split())
        length += len(str(message.get("reasoning") or "").split())
    length = max(length, 1)
    ids = list(range(length))[:max_length]
    labels = [1] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


@pytest.fixture
def fake_labels(monkeypatch):
    import gemmaloop.train.labels as labels_module

    monkeypatch.setattr(labels_module, "assistant_labels", _fake_assistant_labels)


def _single_source(weight: float = 1.0):
    return {
        "alpha": SourceSpec(
            hf_id="fake/alpha",
            config=None,
            split="train",
            license="MIT",
            weight=weight,
            loader=_loader(_rows("alpha")),
        )
    }


def test_rows_contribute_min_rendered_length(fake_labels):
    # Each _rows row renders to 166 word tokens (83 input + 83 output).
    rows, manifest = build_mix(
        target_tokens=250,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=_single_source(),
        max_length=100,
    )
    assert len(rows) == 3  # keys 0, 100, 200 < 250
    assert manifest.max_length == 100
    assert manifest.raw_tokens_per_source["alpha"] == 3 * 166
    assert manifest.trained_tokens_per_source["alpha"] == 3 * 100
    assert manifest.tokens_per_source["alpha"] == 3 * 100


def test_bigger_target_is_prefix_at_same_max_length(fake_labels):
    kwargs = dict(
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        seed=11,
        sources=_sources(),
        max_length=100,
    )
    small, small_manifest = build_mix(500, **kwargs)
    large, large_manifest = build_mix(1_500, **kwargs)
    assert len(small) < len(large)
    assert large[: len(small)] == small
    assert small_manifest.max_length == large_manifest.max_length == 100
    for name in ("alpha", "beta"):
        assert (
            small_manifest.raw_tokens_per_source[name]
            >= small_manifest.trained_tokens_per_source[name]
        )
        assert (
            large_manifest.trained_tokens_per_source[name]
            >= small_manifest.trained_tokens_per_source[name]
        )



# ---------------------------------------------------------------------------
# Filter length mode and reasoning rebalancing (training research R3/R4)
# ---------------------------------------------------------------------------
def _rows_of_length(lengths):
    return [
        {"input": "q", "output": " ".join(["w"] * (length - 1))}
        for length in lengths
    ]


def _reasoning_spec(rows, *, weight=1.0, reasoning=True):
    return SourceSpec(
        hf_id="fake/source",
        config=None,
        split="train",
        license="MIT",
        weight=weight,
        reasoning=reasoning,
        loader=_loader(rows),
    )


def test_filter_mode_drops_long_rows_without_truncating(fake_labels):
    rows, manifest = build_mix(
        target_tokens=1_000_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources={"reason": _reasoning_spec(_rows_of_length([50, 60, 200, 70]))},
        max_length=100,
        length_mode="filter",
    )
    assert manifest.length_mode == "filter"
    assert manifest.dropped["too_long"] == 1
    assert len(rows) == 3
    assert manifest.raw_tokens_per_source["reason"] == 180
    assert manifest.trained_tokens_per_source["reason"] == 180


def test_filter_mode_rebalances_reasoning_to_75_percent(fake_labels):
    sources = {
        "reason": _reasoning_spec(_rows_of_length([50] * 1000), weight=0.4),
        "other": _reasoning_spec(
            _rows_of_length([40] * 1000), weight=0.6, reasoning=False
        ),
    }
    _, manifest = build_mix(
        target_tokens=20_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=sources,
        max_length=100,
        length_mode="filter",
    )
    assert manifest.requested_reasoning_share == pytest.approx(0.4)
    assert manifest.target_reasoning_share == pytest.approx(0.75)
    assert manifest.effective_weights["reason"] == pytest.approx(0.75)
    assert manifest.effective_weights["other"] == pytest.approx(0.25)
    assert manifest.achieved_reasoning_share >= 0.75
    assert manifest.achieved_weights["reason"] == pytest.approx(
        manifest.achieved_reasoning_share
    )
    assert manifest.trained_tokens_per_source["reason"] > 0
    assert manifest.trained_tokens_per_source["other"] > 0


def test_filter_mode_redistributes_unavailable_reasoning_source(fake_labels):
    sources = {
        "reason": _reasoning_spec(_rows_of_length([50] * 1000), weight=0.3),
        "reason2": _reasoning_spec(_rows_of_length([200] * 20), weight=0.3),
        "other": _reasoning_spec(
            _rows_of_length([40] * 1000), weight=0.4, reasoning=False
        ),
    }
    _, manifest = build_mix(
        target_tokens=20_000,
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=sources,
        max_length=100,
        length_mode="filter",
    )
    assert manifest.unavailable_sources == {
        "reason2": "no row fits max_length"
    }
    assert manifest.effective_weights["reason2"] == 0.0
    assert manifest.effective_weights["reason"] == pytest.approx(0.75)
    assert manifest.effective_weights["other"] == pytest.approx(0.25)
    assert manifest.achieved_reasoning_share >= 0.75
    assert manifest.trained_tokens_per_source["reason2"] == 0


def test_filter_mode_bigger_target_is_prefix(fake_labels):
    sources = {
        "reason": _reasoning_spec(_rows_of_length([50] * 1000), weight=0.4),
        "other": _reasoning_spec(
            _rows_of_length([40] * 1000), weight=0.6, reasoning=False
        ),
    }
    kwargs = dict(
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        seed=11,
        sources=sources,
        max_length=100,
        length_mode="filter",
    )
    small, small_manifest = build_mix(2_000, **kwargs)
    large, large_manifest = build_mix(20_000, **kwargs)
    assert len(small) < len(large)
    assert large[: len(small)] == small
    assert small_manifest.effective_weights == large_manifest.effective_weights
    assert small_manifest.achieved_reasoning_share >= 0.74
    assert large_manifest.achieved_reasoning_share >= 0.75


def test_filter_mode_requires_max_length():
    with pytest.raises(ValueError, match="requires max_length"):
        build_mix(
            target_tokens=100,
            tokenizer=WordTokenizer(),
            eval_index=None,
            sources={"reason": _reasoning_spec(_rows_of_length([10]))},
            length_mode="filter",
        )


def test_filter_mode_refuses_low_reasoning_share(fake_labels):
    sources = {
        "reason": _reasoning_spec(_rows_of_length([50] * 200), weight=0.4),
        "other": _reasoning_spec(
            _rows_of_length([40] * 1000), weight=0.6, reasoning=False
        ),
    }
    kwargs = dict(
        tokenizer=WordTokenizer(),
        eval_index=EvalIndex.build(["unrelated evaluation text"]),
        sources=sources,
        max_length=100,
        length_mode="filter",
    )
    with pytest.raises(ValueError, match="starved source.*reason"):
        build_mix(target_tokens=20_000, **kwargs)

    _, manifest = build_mix(
        target_tokens=20_000, allow_low_reasoning=True, **kwargs
    )
    assert manifest.allow_low_reasoning is True
    assert manifest.achieved_reasoning_share < manifest.target_reasoning_share


class TemplateTokenizer(WordTokenizer):
    """Word tokenizer carrying a chat template for provenance tests."""

    chat_template = "{{ messages }} v1"


def test_manifest_records_chat_template_sha256():
    tokenizer = TemplateTokenizer()
    _, manifest = build_mix(
        target_tokens=2_000,
        tokenizer=tokenizer,
        eval_index=EvalIndex.build(["Completely unrelated evaluation text."]),
        seed=0,
        sources=_sources(),
    )
    expected = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    assert manifest.chat_template_sha256 == expected
