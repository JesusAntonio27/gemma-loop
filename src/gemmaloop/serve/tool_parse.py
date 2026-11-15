"""Gemma 4 output parsing (reasoning channel and native tool calls).

The canonical Gemma 4 chat template (published 2026-07-09) emits reasoning as
``<|channel>thought\\n...<channel|>`` and tool calls as
``<|tool_call>call:NAME{arg:<|"|>value<|"|>}<tool_call|>``.  transformers 5.22.1
ships a declarative parser for exactly that format (the tokenizer's
``response_template``); this module wraps it, converts the parsed message to the
OpenAI Chat Completions shape and adds the degradation path for malformed tool
calls, which the upstream parser reports as an exception.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from transformers.utils.chat_parsing import parse_response

# Mirrors ``response_template`` in the Gemma 4 tokenizer_config.json
# (google/gemma-4-*-it, transformers 5.22.1).  Kept here so parsing does not
# require loading a processor (the offline evaluator imports this module
# without a model).
GEMMA4_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text",
        },
        "thinking": {
            "close": "<channel|>",
            "content": "text",
            "open": "<|channel>thought\n",
        },
        "tool_calls": {
            "close": "<tool_call|>",
            "content": "json",
            "content_args": {
                "string_delims": [["<|\"|>", "<|\"|>"]],
                "unquoted_keys": True,
            },
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "repeats": True,
            "transform": {
                "function": {"arguments": "{content}", "name": "{name}"},
                "type": "function",
            },
        },
    },
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
}

_THOUGHT_OPEN = "<|channel>thought\n"
_THOUGHT_CLOSE = "<channel|>"
_TURN_OPEN = "<|turn>model\n"


def _to_openai_calls(calls: list[dict] | None) -> list[dict]:
    """Convert parser tool calls to ``{id, type, function{name, arguments}}``."""
    out: list[dict] = []
    for index, call in enumerate(calls or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments_json = arguments
        elif arguments is None:
            arguments_json = "{}"
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
        out.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments_json,
                },
            }
        )
    return out


def _degrade(raw: str, prefix: str = "") -> tuple[str, str]:
    """Best-effort split of a response the upstream parser rejected.

    Returns ``(content, reasoning)``.  A well-formed thinking channel is still
    separated from the rest so callers keep whatever reasoning the model
    produced before the malformed tool call.
    """
    if _THOUGHT_OPEN in raw:
        before, rest = raw.split(_THOUGHT_OPEN, 1)
        if _THOUGHT_CLOSE in rest:
            reasoning, after = rest.split(_THOUGHT_CLOSE, 1)
            return (before + after).strip(), reasoning.strip()
        return before.strip(), rest.strip()
    if prefix.endswith(_THOUGHT_OPEN) and _THOUGHT_CLOSE in raw:
        reasoning, after = raw.split(_THOUGHT_CLOSE, 1)
        return after.strip(), reasoning.strip()
    return raw.strip(), ""


def parse_gemma_output(
    raw: str, prefix: str = "", tools: list[dict] | None = None
) -> tuple[str, str, list[dict], bool]:
    """Parse a decoded Gemma 4 completion.

    Returns ``(content, reasoning, tool_calls_openai, invalid)``.

    ``prefix`` is the decoded prompt that preceded the generation.  It is only
    needed when the prompt itself opened the thinking channel (the canonical
    template ends tool-response turns with ``<|channel>thought\\n``); pass it
    whenever it is available.  ``invalid`` is true when a tool call was present
    but could not be parsed, in which case the raw text is returned as content
    and the call is dropped.
    """
    try:
        message = parse_response(
            raw, GEMMA4_RESPONSE_TEMPLATE, prefix=prefix, tools=tools
        )
    except Exception:
        # The upstream parser raises on truncated/invalid tool-call JSON.  A
        # broken call must never lose the rest of the turn: keep it as content
        # and flag it so the caller can retry or surface the failure.
        content, reasoning = _degrade(raw, prefix)
        return content, reasoning, [], True

    content = message.get("content") or ""
    reasoning = message.get("thinking") or ""
    calls = _to_openai_calls(message.get("tool_calls"))
    return content, reasoning, calls, False


def openai_history_to_gemma(messages: list[dict]) -> list[dict]:
    """Make an OpenAI history renderable by the Gemma 4 chat template.

    * ``tool_calls[].function.arguments`` must be a mapping for the template;
      JSON strings coming from OpenAI clients are deserialised.
    * reasoning travels in ``reasoning`` (or ``reasoning_content``); the
      template strips ``<|channel>`` blocks from ``content``, so the canonical
      key is used here.
    """
    converted = copy.deepcopy(messages)
    for message in converted:
        if "reasoning_content" in message:
            message.setdefault("reasoning", message["reasoning_content"])
            del message["reasoning_content"]

        for call in message.get("tool_calls") or []:
            function = call.get("function")
            if not function:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid tool_call arguments for {function.get('name')!r}: "
                        f"{arguments!r}"
                    ) from exc
            elif arguments is None:
                function["arguments"] = {}
    return converted
