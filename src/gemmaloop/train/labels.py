"""Assistant-only labels for Gemma 4 SFT.

Gemma 4's chat template has no ``{% generation %}`` markers, so TRL's
``assistant_only_loss=True`` raises (reference 07, trap 1).  This module builds
the labels from **one** render of the whole conversation and locates assistant
turns by their delimiter tokens:

* the conversation is rendered once with the real Gemma 4 chat template;
* reasoning is kept only on the final assistant turn (the template drops it
  from earlier turns when the conversation continues with a user message);
* tokens between ``<|turn>model\\n`` and the matching ``<turn|>`` are labelled,
  except tool-response blocks (``<|tool_response> ... <tool_response|>``),
  which belong to the tool role, and image tokens;
* system and user turns stay at ``-100``;
* truncation uses ``keep_start``.

Reasoning must live in the message's ``reasoning`` key: the template deletes
``<|channel>thought ... <channel|>`` found inside ``content`` (trap 4).

Rendering the conversation once (instead of once per message prefix) matters:
the template decides whether to emit a turn's reasoning from later messages, so
prefix renders can be longer than the assistant span in the final sequence and
would bleed labels onto the next user turn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from gemmaloop.model.load import IMAGE_TOKEN_IDS

__all__ = ["assistant_labels"]

_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"
_MODEL_ROLE = "model"
_TOOL_RESPONSE_OPEN = "<|tool_response>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    tools: Sequence[Any] | None = None,
    *,
    enable_thinking: bool | None = None,
) -> list[int]:
    if not messages:
        return []
    if enable_thinking is None:
        enable_thinking = _messages_have_reasoning(messages)
    kwargs: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    if tools:
        kwargs["tools"] = list(tools)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        **kwargs,
    )
    if isinstance(rendered, Mapping):
        return list(rendered["input_ids"])
    return list(rendered)


def _messages_have_reasoning(messages: Sequence[Mapping[str, Any]]) -> bool:
    """True when any assistant turn carries a non-empty ``reasoning`` trace."""

    return any(
        str(message.get("role", "")).lower() == "assistant"
        and bool(message.get("reasoning"))
        for message in messages
    )


def _special_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # pragma: no cover - defensive for exotic tokenizers
        return None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(token_id)


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:  # pragma: no cover - tokenizers without the kwarg
        return list(tokenizer.encode(text))


def _without_earlier_reasoning(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep ``reasoning`` only on the final assistant message.

    This mirrors the inference-time history rule: earlier assistant turns are
    rendered without their thinking trace.  The template also applies a
    last-user guard, but making it explicit keeps the labels stable for tool
    loops, where every assistant turn lies after the last user message.
    """
    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if str(message.get("role", "")).lower() == "assistant"
    ]
    if not assistant_positions:
        return [dict(message) for message in messages]
    final = assistant_positions[-1]

    cleaned: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index != final and str(message.get("role", "")).lower() == "assistant":
            message = {
                key: value
                for key, value in message.items()
                if key not in ("reasoning", "reasoning_content")
            }
        else:
            message = dict(message)
        cleaned.append(message)
    return cleaned


def _assistant_spans(
    token_ids: list[int], tokenizer: PreTrainedTokenizerBase
) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` spans of assistant turn bodies."""
    turn_open = _special_id(tokenizer, _TURN_OPEN)
    turn_close = _special_id(tokenizer, _TURN_CLOSE)
    role_ids = _encode(tokenizer, _MODEL_ROLE)
    newline_ids = set(_encode(tokenizer, "\n"))
    if turn_open is None or turn_close is None or not role_ids:
        return []

    spans: list[tuple[int, int]] = []
    index = 0
    role_length = len(role_ids)
    while index < len(token_ids):
        if (
            token_ids[index] == turn_open
            and token_ids[index + 1 : index + 1 + role_length] == role_ids
        ):
            start = index + 1 + role_length
            while start < len(token_ids) and token_ids[start] in newline_ids:
                start += 1
            end = start
            while end < len(token_ids) and token_ids[end] != turn_close:
                end += 1
            if end < len(token_ids):
                end += 1  # include the end-of-turn token
            spans.append((start, end))
            index = max(end, start)
        else:
            index += 1
    return spans


def assistant_labels(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    max_length: int,
    tools: Sequence[Any] | None = None,
    *,
    enable_thinking: bool | None = None,
) -> dict[str, list[int]]:
    """Tokenize ``messages`` and mask everything outside assistant turns.

    Rows with a ``reasoning`` trace are rendered with ``enable_thinking=True``;
    rows without one use ``enable_thinking=False`` so they do not teach the
    model that a thinking-mode prompt can be answered without a thought.
    ``enable_thinking`` overrides the automatic choice.

    Returns ``{"input_ids", "labels", "attention_mask"}`` as plain lists so the
    result can be fed straight to ``datasets.Dataset.from_list``.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length!r}")
    if not messages:
        raise ValueError("messages must not be empty")

    if enable_thinking is None:
        enable_thinking = _messages_have_reasoning(messages)
    cleaned = _without_earlier_reasoning(messages)
    full_ids = _render(
        tokenizer, cleaned, tools=tools, enable_thinking=enable_thinking
    )
    labels = [-100] * len(full_ids)

    tool_open = _special_id(tokenizer, _TOOL_RESPONSE_OPEN)
    tool_close = _special_id(tokenizer, _TOOL_RESPONSE_CLOSE)
    image_ids = set(int(token) for token in IMAGE_TOKEN_IDS)

    for start, end in _assistant_spans(full_ids, tokenizer):
        inside_tool_response = False
        for position in range(start, end):
            token_id = full_ids[position]
            if tool_open is not None and token_id == tool_open:
                inside_tool_response = True
            if inside_tool_response:
                labels[position] = -100
            elif token_id in image_ids:
                labels[position] = -100
            else:
                labels[position] = token_id
            if tool_close is not None and token_id == tool_close:
                inside_tool_response = False

    input_ids = full_ids[:max_length]
    labels = labels[:max_length]
    return {
        "input_ids": [int(token) for token in input_ids],
        "labels": [int(label) for label in labels],
        "attention_mask": [1] * len(input_ids),
    }
