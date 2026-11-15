"""Assistant-only labels for the Gemma 4 chat template (Task 13)."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from gemmaloop.model.load import IMAGE_TOKEN_IDS
from gemmaloop.train.labels import assistant_labels

TOKENIZER_ID = "google/gemma-4-E2B-it"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as exc:  # pragma: no cover - offline without HF cache
        pytest.skip(f"Gemma 4 tokenizer unavailable: {exc}")


def _messages():
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Run the tests."},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "I should run pytest.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "pytest -q"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "bash",
            "content": "1 failed, 4 passed",
        },
        {"role": "assistant", "content": "The bug is in add().", "reasoning": "Check the traceback."},
    ]


def _labelled_text(tokenizer, output):
    labelled = [
        token
        for token, label in zip(output["input_ids"], output["labels"])
        if label != -100
    ]
    return tokenizer.decode(labelled)


def _assistant_positions(tokenizer, ids):
    """Return the token positions inside assistant turns, excluding tool responses."""
    turn_open = tokenizer.convert_tokens_to_ids("<|turn>")
    turn_close = tokenizer.convert_tokens_to_ids("<turn|>")
    model_id = tokenizer.encode("model", add_special_tokens=False)[0]
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    tool_open = tokenizer.convert_tokens_to_ids("<|tool_response>")
    tool_close = tokenizer.convert_tokens_to_ids("<tool_response|>")

    positions = set()
    index = 0
    while index < len(ids):
        if ids[index] == turn_open and ids[index + 1] == model_id:
            start = index + 2
            while start < len(ids) and ids[start] in newline_ids:
                start += 1
            end = start
            while end < len(ids) and ids[end] != turn_close:
                end += 1
            if end < len(ids):
                end += 1
            inside_tool = False
            for position in range(start, end):
                if ids[position] == tool_open:
                    inside_tool = True
                if not inside_tool:
                    positions.add(position)
                if ids[position] == tool_close:
                    inside_tool = False
            index = end
        else:
            index += 1
    return positions


def _assert_exact_assistant_labels(tokenizer, output):
    expected = _assistant_positions(tokenizer, output["input_ids"])
    for position, (token_id, label) in enumerate(
        zip(output["input_ids"], output["labels"])
    ):
        if token_id in IMAGE_TOKEN_IDS:
            assert label == -100
        elif position in expected:
            assert label == token_id
        else:
            assert label == -100


def test_only_assistant_turns_are_labelled(tokenizer):
    output = assistant_labels(tokenizer, _messages(), max_length=4096)
    assert len(output["input_ids"]) == len(output["labels"]) == len(output["attention_mask"])

    text = _labelled_text(tokenizer, output)
    # Final assistant content, reasoning and tool-call arguments survive.
    assert "Check the traceback." in text
    assert "The bug is in add()." in text
    assert "pytest -q" in text
    # Earlier reasoning is not part of the training history.
    assert "I should run pytest." not in text
    # System, user and tool response tokens are masked.
    assert "You are a coding agent." not in text
    assert "Run the tests." not in text
    assert "1 failed, 4 passed" not in text
    assert "<|tool_response>" not in text

    assert all(flag == 1 for flag in output["attention_mask"])
    _assert_exact_assistant_labels(tokenizer, output)


def test_reasoning_only_on_final_assistant_turn(tokenizer):
    """Regression: a reasoning turn followed by a user turn must not leak labels."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first answer", "reasoning": "first reasoning"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "second answer", "reasoning": "second reasoning"},
    ]
    output = assistant_labels(tokenizer, messages, max_length=4096)
    text = _labelled_text(tokenizer, output)

    assert "first answer" in text
    assert "second answer" in text
    assert "second reasoning" in text
    assert "first reasoning" not in text
    assert "first user" not in text
    assert "second user" not in text
    assert "You are helpful." not in text
    _assert_exact_assistant_labels(tokenizer, output)


def test_truncation_keeps_the_start(tokenizer):
    messages = [
        {"role": "user", "content": "u " * 100},
        {"role": "assistant", "content": "a " * 100},
    ]
    full = assistant_labels(tokenizer, messages, max_length=4096)
    short = assistant_labels(tokenizer, messages, max_length=8)
    assert short["input_ids"] == full["input_ids"][:8]
    assert short["labels"] == full["labels"][:8]
    assert len(short["attention_mask"]) == 8


def test_image_tokens_are_masked(tokenizer):
    output = assistant_labels(
        tokenizer,
        [
            {"role": "user", "content": "look at <|image|>"},
            {"role": "assistant", "content": "the image <|image|> is red"},
        ],
        max_length=512,
    )
    positions = [
        index
        for index, token_id in enumerate(output["input_ids"])
        if token_id in IMAGE_TOKEN_IDS
    ]
    assert positions
    for position in positions:
        assert output["labels"][position] == -100
    _assert_exact_assistant_labels(tokenizer, output)


def test_tool_declarations_are_masked(tokenizer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    output = assistant_labels(
        tokenizer,
        [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": "done"},
        ],
        max_length=1024,
        tools=tools,
    )
    text = _labelled_text(tokenizer, output)
    assert "Run a shell command" not in text
    assert "done" in text
    _assert_exact_assistant_labels(tokenizer, output)


def test_thinking_flag_follows_reasoning_presence(tokenizer):
    with_reasoning = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning": "trace"},
    ]
    without_reasoning = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    thinking_text = tokenizer.decode(
        assistant_labels(tokenizer, with_reasoning, max_length=512)["input_ids"]
    )
    plain_text = tokenizer.decode(
        assistant_labels(tokenizer, without_reasoning, max_length=512)["input_ids"]
    )

    assert "<|think|>" in thinking_text
    assert "<|think|>" not in plain_text


class _RecordingTokenizer:
    """Minimal tokenizer that records the ``enable_thinking`` template flag."""

    unk_token_id = None

    def __init__(self) -> None:
        self.enable_thinking: list[bool] = []

    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
        self.enable_thinking.append(bool(kwargs.get("enable_thinking")))
        return {"input_ids": [1, 2, 3]}

    def convert_tokens_to_ids(self, token):  # noqa: ANN001
        return None

    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        return [1]


def test_enable_thinking_is_chosen_per_row():
    tokenizer = _RecordingTokenizer()
    assistant_labels(
        tokenizer,
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        max_length=16,
    )
    assistant_labels(
        tokenizer,
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": "r"},
        ],
        max_length=16,
    )
    assert tokenizer.enable_thinking == [False, True]
