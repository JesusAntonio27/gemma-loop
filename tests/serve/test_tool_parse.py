"""Parsing of Gemma 4 assistant output (Task 5)."""

from __future__ import annotations

from pathlib import Path

from gemmaloop.serve.tool_parse import openai_history_to_gemma, parse_gemma_output

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gemma_outputs"


def test_single_tool_call():
    raw = '<|tool_call>call:bash{command:<|"|>pytest -q<|"|>}<tool_call|>'
    content, reasoning, calls, invalid = parse_gemma_output(raw)
    assert calls[0]["function"]["name"] == "bash" and not invalid
    assert calls[0]["function"]["arguments"] == '{"command": "pytest -q"}'


def test_reasoning_split():
    raw = "<|channel>thought\nplan<channel|>Hola"
    assert parse_gemma_output(raw)[:2] == ("Hola", "plan")


def test_malformed_tool_call_degrades_to_content():
    raw = '<|tool_call>call:bash{command:<|"|>ls'
    content, _, calls, invalid = parse_gemma_output(raw)
    assert calls == [] and invalid and "bash" in content


def test_history_arguments_become_dict():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}}]}]
    assert openai_history_to_gemma(msgs)[0]["tool_calls"][0]["function"]["arguments"] == {"filePath": "a.py"}


def test_history_reasoning_content_becomes_reasoning():
    msgs = [{"role": "assistant", "content": "ok", "reasoning_content": "because"}]
    converted = openai_history_to_gemma(msgs)[0]
    assert converted["reasoning"] == "because"
    assert "reasoning_content" not in converted


def test_parse_fixture_files():
    raw = (FIXTURES / "single_tool_call.txt").read_text(encoding="utf-8")
    content, reasoning, calls, invalid = parse_gemma_output(raw)
    assert not invalid and reasoning == ""
    assert calls[0]["function"]["name"] == "bash"
    assert calls[0]["function"]["arguments"] == '{"command": "pytest -q"}'

    raw = (FIXTURES / "reasoning_tool_call.txt").read_text(encoding="utf-8")
    content, reasoning, calls, invalid = parse_gemma_output(raw)
    assert not invalid and reasoning == "plan"
    assert content == "Hola"
    assert calls[0]["function"]["name"] == "bash"


def test_parse_with_prompt_prefix_detects_thinking_channel():
    # `generate_batch` decodes only the new tokens, so the opening thinking
    # marker lives in the prompt.  The parser must be told about it.
    raw = "plan<channel|>Hola"
    prefix = "<|turn>model\n<|channel>thought\n"
    assert parse_gemma_output(raw, prefix=prefix)[:2] == ("Hola", "plan")
