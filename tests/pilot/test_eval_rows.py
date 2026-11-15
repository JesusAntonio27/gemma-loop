"""Deterministic Stage 0 -> evaluation-row renderer (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.pilot import eval_rows
from gemmaloop.pilot.evaluate import load_eval_rows
from gemmaloop.train.labels import assistant_labels

MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer", "reasoning": "thought"},
]


class FakeTokenizer:
    unk_token_id = None
    added_tokens_encoder = {}

    def convert_tokens_to_ids(self, token):
        return {
            "<|turn>": 250,
            "<turn|>": 251,
            "<channel|>": 252,
        }.get(token)

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [0]

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return {"input_ids": [250, 0, 3, 4, 252, 5, 6, 251]}


def _stage0(tmp_path: Path) -> Path:
    stage0 = tmp_path / "stage0"
    for name, problem_id in (
        ("pilot_tfit", "fit-problem"),
        ("pilot_selection", "eval-problem"),
        ("pilot_p1", "p1-problem"),
    ):
        path = stage0 / "pilot" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"problem_id": problem_id, "messages": MESSAGES}) + "\n",
            encoding="utf-8",
        )
    return stage0


def test_rendered_rows_equal_the_training_rendering(tmp_path):
    stage0 = _stage0(tmp_path)
    tokenizer = FakeTokenizer()
    payload = eval_rows.render_stage0_eval_rows(
        stage0,
        tmp_path / "rows",
        tokenizer=tokenizer,
        max_length=32,
        tokenizer_revision="rev",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "rows" / "pilot_tfit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = assistant_labels(tokenizer, MESSAGES, 32)
    row = rows[0]
    assert row["input_ids"] == expected["input_ids"]
    assert row["labels"] == expected["labels"]
    assert row["label_positions"] == [
        index for index, value in enumerate(expected["labels"]) if value != -100
    ]
    assert row["labels_start"] == row["label_positions"][0]
    assert row["answer_start"] == 5
    assert row["problem_id"] == "fit-problem"
    assert row["tokenizer_revision"] == "rev"
    assert payload["splits"]["pilot_tfit"]["source_sha256"]
    assert payload["splits"]["pilot_tfit"]["output_sha256"]

    loaded = load_eval_rows(tmp_path / "rows" / "pilot_tfit.jsonl")
    assert loaded[0].input_ids == expected["input_ids"]
    assert loaded[0].labels_start == row["labels_start"]
    assert loaded[0].answer_start == 5
    assert loaded[0].cluster == "fit-problem"


def test_second_render_reuses_the_cache_without_the_tokenizer(tmp_path, monkeypatch):
    stage0 = _stage0(tmp_path)
    first = eval_rows.render_stage0_eval_rows(
        stage0,
        tmp_path / "rows",
        tokenizer=FakeTokenizer(),
        max_length=32,
        tokenizer_revision="rev",
    )

    def boom(*args, **kwargs):
        raise AssertionError("cached rows must not be re-rendered")

    monkeypatch.setattr(eval_rows, "assistant_labels", boom)
    second = eval_rows.render_stage0_eval_rows(
        stage0,
        tmp_path / "rows",
        tokenizer=FakeTokenizer(),
        max_length=32,
        tokenizer_revision="rev",
    )
    assert second == first


def test_missing_problem_id_is_refused_for_registered_splits(tmp_path):
    with pytest.raises(ValueError, match="problem_id"):
        eval_rows.render_trace_rows(
            [{"messages": MESSAGES}],
            FakeTokenizer(),
            max_length=32,
            source="p1lite.jsonl",
        )

    path = tmp_path / "p1lite.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "input_ids": [1, 2, 3], "labels_start": 1})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="problem_id"):
        load_eval_rows(path, require_problem_id=True)
    # Without the registered-run requirement the legacy shape still loads.
    assert load_eval_rows(path)[0].cluster == "0"


def test_renderer_cli_cannot_misstate_the_pinned_tokenizer_revision():
    """N3: the tokenizer is always loaded at the pinned revision."""

    import scripts.render_pilot_eval_rows as renderer

    with pytest.raises(SystemExit):
        renderer._parse_args(
            ["--stage0", "stage0", "--out", "rows", "--tokenizer-revision", "x"]
        )
