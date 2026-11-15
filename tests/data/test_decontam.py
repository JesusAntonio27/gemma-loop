"""Tests for the evaluation-set decontamination index."""

from __future__ import annotations

import pytest

from gemmaloop.data.decontam import EvalIndex, ngrams, normalize
from gemmaloop.data.sources import SOURCES, to_messages

EVAL = [
    "Given an array of n integers, find the maximum sum of a contiguous "
    "subarray of length exactly k and print it."
]


def test_exact_and_ngram_and_indel():
    idx = EvalIndex.build(EVAL)
    assert idx.check(EVAL[0]) == "exact"
    assert idx.check("Intro. " + EVAL[0][:-10] + " please.") in {
        "ngram13",
        "indel75",
    }
    assert idx.check("Write a haiku about autumn leaves falling slowly.") is None


def test_reasoning_goes_to_reasoning_key():
    msgs = to_messages(
        {"input": "q", "output": "<think>plan</think>```python\nx=1\n```"},
        "ocr",
    )
    assert msgs[-1]["reasoning"] == "plan" and "<think>" not in msgs[-1]["content"]


def test_indel_only_paraphrase_without_shared_ngrams():
    # Matching runs of three tokens only: no shared 5-token window, and no
    # shared 13-gram.  Indel similarity is still >= 0.75.
    eval_tokens = [f"t{index}" for index in range(40)]
    query_tokens = list(eval_tokens)
    for index in range(3, 40, 4):
        query_tokens[index] = f"changed{index}"
    eval_text = " ".join(eval_tokens)
    query = " ".join(query_tokens)

    idx = EvalIndex.build([eval_text])
    assert not any(
        tuple(eval_tokens[i : i + 5]) == tuple(query_tokens[i : i + 5])
        for i in range(36)
    )
    assert idx.check(query) == "indel75"


def test_gemma_channel_reasoning_is_split():
    msgs = to_messages(
        {
            "input": "q",
            "output": "<|channel>thought\nplan<channel|>final answer",
        },
        "ocr",
    )
    assert msgs[-1]["reasoning"] == "plan"
    assert msgs[-1]["content"] == "final answer"


def test_unclosed_think_is_split():
    msgs = to_messages({"input": "q", "output": "<think>truncated plan"}, "ocr")
    assert msgs[-1]["reasoning"] == "truncated plan"
    assert msgs[-1]["content"] == ""


@pytest.mark.parametrize(
    ("source", "row", "expected_reasoning"),
    [
        (
            "ocr",
            {"input": "q", "output": "<think>ocr plan</think>answer"},
            "ocr plan",
        ),
        (
            "oci",
            {"input": "q", "output": "```python\nx=1\n```"},
            None,
        ),
        (
            "omr",
            {
                "problem": "p",
                "generated_solution": "<think>omr plan</think>answer",
            },
            "omr plan",
        ),
        (
            "ot3",
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "<think>ot3 plan</think>answer"},
                ]
            },
            "ot3 plan",
        ),
        (
            "chat",
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<think>chat plan</think>answer"},
                ]
            },
            "chat plan",
        ),
    ],
)
def test_each_source_splitter(source, row, expected_reasoning):
    msgs = to_messages(row, source)
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1].get("reasoning") == expected_reasoning
    assert "<think>" not in msgs[-1]["content"]


def test_normalize_and_ngrams_contract():
    assert normalize("  Hello,   WORLD! ") == "hello world"
    assert normalize("a-b_c") == "ab_c"
    assert ngrams(["a", "b", "c"], n=3) == {("a", "b", "c")}
    assert ngrams(["a", "b"], n=3) == set()


def test_source_row_filters():
    ocr_filter = SOURCES["ocr"].row_filter
    assert ocr_filter({"input": "q", "output": "x" * 500}) == "ocr_length"
    assert ocr_filter({"input": "q", "output": "x" * 900}) is None

    oci_filter = SOURCES["oci"].row_filter
    assert oci_filter({"input": "q", "average_test_score": "0.9"}) == "oci_score"
    assert oci_filter({"input": "q", "average_test_score": "1"}) is None

    omr_filter = SOURCES["omr"].row_filter
    assert omr_filter({"problem": "p", "problem_source": "amc_aime"}) == "omr_amc_aime"
    assert omr_filter({"problem": "p", "problem_source": "aops_c6"}) is None

    ot3_filter = SOURCES["ot3"].row_filter
    assert ot3_filter({"domain": "math"}) == "ot3_domain"
    assert ot3_filter({"domain": "code"}) is None
