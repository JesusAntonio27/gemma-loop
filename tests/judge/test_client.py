"""Tests for the vision-judge client and rubric (Task 18).

Every test uses a fake OpenAI-compatible client: no network, no API key and no
spend.  The fake mirrors the subset of the SDK the judge relies on
(``client.chat.completions.create`` and the response fields).
"""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import gemmaloop.judge.client as client_module
from gemmaloop.deepseek_money import WasteGuard
from gemmaloop.judge.client import (
    JudgeClientConfig,
    JudgeRecord,
    JudgeVersionError,
    Verdict,
    build_client,
    judge_pair,
)
from gemmaloop.judge.render import RenderResult
from gemmaloop.judge.rubric import SYSTEM_PROMPT, USER_TEMPLATE, user_text

DEFAULT_VERSION_PATH_AT_IMPORT = client_module.DEFAULT_JUDGE_VERSION_PATH
DEFAULT_GATES_PATH_AT_IMPORT = client_module.DEFAULT_GATES_PATH

BRIEF_SPEC = {
    "brief_id": "b1",
    "kind": "page",
    "brief": "A landing page for a specialty coffee shop.",
    "required_sections": ["Hero", "Pricing"],
    "required_copy": ["Buy now"],
    "forbidden": ["lorem ipsum"],
}


def _verdict_json(**overrides) -> str:
    payload = {
        "render_ok": {"A": True, "B": True},
        "criteria": {
            "brief_fidelity": {"A": 9, "B": 7, "note": "Both complete"},
            "hierarchy": {"A": 8, "B": 7, "note": "A reads better"},
            "layout": {"A": 8, "B": 7, "note": "A is balanced"},
            "color_a11y": {"A": 8, "B": 7, "note": "Both fine"},
            "craft": {"A": 8, "B": 7, "note": "A is consistent"},
            "mobile": {"A": 8, "B": 7, "note": "Both reflow"},
        },
        "weighted": {"A": 8.2, "B": 7.0},
        "winner": "A",
        "confidence": 85,
        "decisive_reason": "Clearer hierarchy",
        "flags": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _response(
    content: str | None,
    *,
    model: str = "deepseek-flash",
    fingerprint: str | None = "fp-test",
    usage: dict | None = None,
    finish_reason: str = "stop",
):
    return SimpleNamespace(
        id="resp_1",
        model=model,
        system_fingerprint=fingerprint,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            **(usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        ),
    )


class FakeClient:
    """Minimal stand-in for ``openai.OpenAI``."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


@pytest.fixture
def fake_client():
    def _make(*responses):
        return FakeClient(*responses)

    return _make


@pytest.fixture(autouse=True)
def _reset_registered_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        client_module,
        "DEFAULT_JUDGE_VERSION_PATH",
        tmp_path / "judge_version.json",
    )
    client_module.reset_judge_version()
    client_module.register_judge_version("deepseek-flash", "fp-test")
    yield
    client_module.reset_judge_version()


def _render(tmp_path, name: str) -> RenderResult:
    directory = tmp_path / name
    directory.mkdir()
    desktop = directory / "desktop.png"
    mobile = directory / "mobile.png"
    desktop.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"-desktop")
    mobile.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"-mobile")
    return RenderResult(True, desktop, mobile, None)


def _render_desktop_only(tmp_path, name: str) -> RenderResult:
    directory = tmp_path / name
    directory.mkdir()
    desktop = directory / "desktop.png"
    desktop.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"-desktop")
    return RenderResult(True, desktop, None, None)


def _judge(tmp_path, client, *, a=None, b=None, model="deepseek-flash", max_tokens=4096):
    a = _render(tmp_path, "a") if a is None else a
    b = _render(tmp_path, "b") if b is None else b
    return judge_pair(
        BRIEF_SPEC,
        a,
        b,
        client=client,
        model=model,
        max_tokens=max_tokens,
        waste_guard=WasteGuard(window=10_000, rate=1.0),
    )


def test_render_error_does_not_call_the_judge(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json()))
    failed = RenderResult(False, None, None, "blank mobile screenshot")

    record = _judge(tmp_path=tmp_path, client=client, a=_render(tmp_path, "a"), b=failed)

    assert isinstance(record, JudgeRecord)
    assert record.status == "render_error"
    assert record.verdict is None
    assert record.response is None
    assert client.calls == []
    assert record.error and "render" in record.error.lower()


def test_empty_content_is_retried_once(tmp_path, fake_client):
    client = fake_client(_response("   "), _response(_verdict_json()))

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "ok"
    assert record.attempts == 2
    assert len(client.calls) == 2
    assert record.verdict is not None and record.verdict.winner == "A"


def test_empty_content_twice_is_invalid(tmp_path, fake_client):
    client = fake_client(_response(""), _response("   "))

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "invalid"
    assert record.verdict is None
    assert record.attempts == 2
    assert len(client.calls) == 2


def test_invalid_json_is_retried_once_then_invalid(tmp_path, fake_client):
    client = fake_client(_response("{not json"), _response("[]"))

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "invalid"
    assert record.verdict is None
    assert len(client.calls) == 2


def test_schema_violation_is_retried_once_then_invalid(tmp_path, fake_client):
    client = fake_client(
        _response(_verdict_json(winner="C")),
        _response(_verdict_json(confidence="high")),
    )

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "invalid"
    assert record.verdict is None
    assert len(client.calls) == 2


def test_invalid_json_then_valid_is_ok(tmp_path, fake_client):
    client = fake_client(_response("{broken"), _response(_verdict_json()))

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "ok"
    assert record.verdict is not None and record.verdict.winner == "A"
    assert len(client.calls) == 2


def test_request_puts_rubric_first_and_images_in_user_message(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json()))
    a = _render(tmp_path, "a")
    b = _render(tmp_path, "b")

    _judge(tmp_path=tmp_path, client=client, a=a, b=b)

    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "deepseek-flash"
    assert request["max_tokens"] == 4096
    assert request["response_format"] == {"type": "json_object"}
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}

    messages = request["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"][0] == {
        "type": "text",
        "text": user_text(
            BRIEF_SPEC["brief"],
            BRIEF_SPEC["required_sections"],
            BRIEF_SPEC["required_copy"],
            BRIEF_SPEC["forbidden"],
        ),
    }

    images = [part for part in messages[1]["content"] if part["type"] == "image_url"]
    assert len(images) == 4
    for part, path in zip(images, [a.desktop_png, a.mobile_png, b.desktop_png, b.mobile_png]):
        url = part["image_url"]["url"]
        assert part["image_url"]["detail"] == "original"
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == path.read_bytes()


def test_model_and_max_tokens_are_configurable(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json(), model="deepseek-flash"))

    _judge(tmp_path=tmp_path, client=client, model="deepseek-flash", max_tokens=2048)

    assert client.calls[0]["model"] == "deepseek-flash"
    assert client.calls[0]["max_tokens"] == 2048


def test_record_keeps_request_raw_response_usage_and_fingerprint(tmp_path, fake_client):
    raw = _verdict_json()
    usage = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}
    client = fake_client(_response(raw, usage=usage))

    record = _judge(tmp_path=tmp_path, client=client)

    assert record.status == "ok"
    assert record.request == client.calls[0]
    assert record.response == raw
    assert record.usage == usage
    assert record.model == "deepseek-flash"
    assert record.system_fingerprint == "fp-test"
    assert record.verdict == Verdict.model_validate(json.loads(raw))


def test_missing_capture_file_is_a_render_error(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json()))
    a = _render(tmp_path, "a")
    b = _render(tmp_path, "b")
    b.mobile_png.unlink()

    record = _judge(tmp_path=tmp_path, client=client, a=a, b=b)

    assert record.status == "render_error"
    assert client.calls == []


def test_changed_model_version_aborts_the_run(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json(), model="deepseek-other"))

    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_changed_system_fingerprint_aborts_the_run(tmp_path, fake_client):
    client_module.register_judge_version("deepseek-flash", "fp-test")
    client = fake_client(_response(_verdict_json(), fingerprint="fp-changed"))

    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_verdict_rejects_unknown_winner():
    with pytest.raises(ValidationError):
        Verdict(
            render_ok={"A": True, "B": True},
            criteria={},
            weighted={"A": 1.0, "B": 0.0},
            winner="C",
            confidence=50,
            decisive_reason="x",
            flags=[],
        )


def test_system_prompt_is_the_fixed_rubric():
    assert SYSTEM_PROMPT.startswith(
        "You are a senior front-end design reviewer."
    )
    for phrase in (
        "Brief fidelity & content completeness (25)",
        "Visual hierarchy & typography (20)",
        "Layout, spacing & composition (15)",
        "Color, contrast & accessibility (15)",
        "Craft & consistency (15)",
        "Responsive / mobile readiness (10)",
        "A tie is allowed.",
        "output JSON only, no markdown.",
    ):
        assert phrase in SYSTEM_PROMPT


def test_user_text_fills_the_template_and_names_empty_lists():
    text = user_text("Build a shop", ["Hero"], ["Buy now"], [])

    expected = (
        USER_TEMPLATE.replace("{{brief}}", "Build a shop")
        .replace("{{required_sections}}", "Hero")
        .replace("{{required_copy}}", "Buy now")
        .replace("{{forbidden}}", "none")
    )
    assert text == expected
    assert "BRIEF: Build a shop" in text
    assert "Must include: Hero; must appear verbatim: Buy now; forbidden: none." in text
    assert "CANDIDATE A: images 1 (desktop full) and 2 (mobile full)." in text
    assert '"winner":"A|B|tie"' in text


def test_single_image_layout_marks_mobile_not_applicable():
    text = user_text("Draw a logo", [], [], [], image_layout="single")

    assert "CANDIDATE A: image 1. CANDIDATE B: image 2." in text
    assert "images 3 and 4" not in text
    assert "HARD CONSTRAINTS: desktop 1440x900, mobile 390x844." not in text
    assert "HARD CONSTRAINTS: desktop 1440x900." in text
    assert '"mobile":' not in text
    assert "not applicable" in text.lower()

    with pytest.raises(ValueError):
        user_text("Draw a logo", [], [], [], image_layout="weird")


def test_build_client_reads_the_configured_environment_variable(monkeypatch):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("MY_JUDGE_KEY", "secret-value")

    build_client(
        JudgeClientConfig(
            base_url="https://api.deepseek.com",
            model="other-model",
            api_key_env="MY_JUDGE_KEY",
        ),
        client_factory=factory,
    )

    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["api_key"] == "secret-value"
    assert captured["max_retries"] == 0
    assert captured["timeout"] <= 300


def test_build_client_reports_the_missing_variable_without_a_key(monkeypatch):
    monkeypatch.delenv("MISSING_JUDGE_KEY", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        build_client(
            JudgeClientConfig(api_key_env="MISSING_JUDGE_KEY"),
            client_factory=lambda **kwargs: object(),
        )

    message = str(excinfo.value)
    assert "MISSING_JUDGE_KEY" in message
    assert "secret" not in message.lower()


def test_build_client_refuses_an_opencode_endpoint():
    with pytest.raises(ValueError):
        build_client(
            JudgeClientConfig(base_url="https://opencode-go.example", api_key_env="X"),
            client_factory=lambda **kwargs: object(),
        )


def test_single_image_pair_uses_two_image_prompt(tmp_path, fake_client):
    client = fake_client(_response(_verdict_json()))
    a = _render_desktop_only(tmp_path, "a")
    b = _render_desktop_only(tmp_path, "b")

    record = _judge(tmp_path=tmp_path, client=client, a=a, b=b)

    assert record.status == "ok"
    content = client.calls[0]["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 2
    for part, path in zip(images, [a.desktop_png, b.desktop_png]):
        assert base64.b64decode(
            part["image_url"]["url"].split(",", 1)[1]
        ) == path.read_bytes()

    text = content[0]["text"]
    assert "CANDIDATE A: image 1. CANDIDATE B: image 2." in text
    assert "images 3 and 4" not in text
    assert '"mobile":' not in text
    assert "not applicable" in text.lower()


def test_gate_version_record_survives_fresh_registration(tmp_path, fake_client):
    version_path = tmp_path / "judge_version.json"
    client_module.save_judge_version(
        "deepseek-flash", "fp-1", path=version_path
    )
    client_module.reset_judge_version()
    assert client_module.registered_judge_version() is None

    assert client_module.load_judge_version(version_path) == (
        "deepseek-flash",
        "fp-1",
    )

    client = fake_client(_response(_verdict_json(), fingerprint="fp-2"))
    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_judge_pair_auto_loads_the_version_file(tmp_path, monkeypatch, fake_client):
    version_path = tmp_path / "judge_version.json"
    client_module.save_judge_version(
        "deepseek-flash", "fp-1", path=version_path
    )
    client_module.reset_judge_version()
    monkeypatch.setattr(
        client_module, "DEFAULT_JUDGE_VERSION_PATH", version_path
    )

    client = fake_client(_response(_verdict_json(), fingerprint="fp-2"))
    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_registered_fingerprint_must_be_present_in_the_response(tmp_path, fake_client):
    client_module.register_judge_version("deepseek-flash", "fp-1")
    client = fake_client(_response(_verdict_json(), fingerprint=None))

    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_registered_model_must_be_present_in_the_response(tmp_path, fake_client):
    client_module.register_judge_version("deepseek-flash", "fp-1")
    client = fake_client(_response(_verdict_json(), model=None))

    with pytest.raises(JudgeVersionError):
        _judge(tmp_path=tmp_path, client=client)


def test_opencode_opt_in_never_builds_a_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPT_IN_KEY", "secret")
    gates = tmp_path / "gates.md"
    gates.write_text(
        "OPENCODE_WRITTEN_PERMISSION: granted 2026-11-03\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        build_client(
            JudgeClientConfig(
                base_url="https://opencode-go.example/v1",
                api_key_env="OPT_IN_KEY",
            ),
            client_factory=lambda **kwargs: object(),
            allow_opencode_hosted=True,
            gates_path=gates,
        )


def _gate_args(tmp_path, out, version):
    return [
        "--out",
        str(out),
        "--version-out",
        str(version),
        "--ledger",
        str(tmp_path / "spend.json"),
        "--spend-lock",
        str(tmp_path / "spend.lock"),
        "--init-ledger",
    ]


@pytest.fixture(scope="module")
def gate_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "judge_vision_gate.py"
    )
    spec = importlib.util.spec_from_file_location("judge_vision_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_defaults_to_4096_max_tokens(gate_module):
    assert gate_module.parse_args([]).max_tokens == 4096


def test_gate_retries_once_and_records_finish_reason_and_raw_content(
    tmp_path, monkeypatch, fake_client, gate_module
):
    client = fake_client(
        _response("   ", finish_reason="length"),
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr(gate_module, "build_client", lambda config: client)
    out = tmp_path / "gates.md"
    version = tmp_path / "judge_version.json"

    code = gate_module.main(_gate_args(tmp_path, out, version))

    assert code == 0
    assert len(client.calls) == 2
    log = out.read_text(encoding="utf-8")
    assert "finish_reason: stop" in log
    assert "raw_content:" in log
    assert "red square" in log
    assert json.loads(version.read_text(encoding="utf-8")) == {
        "model": "deepseek-flash",
        "system_fingerprint": "fp-test",
    }


def test_gate_fails_and_pins_nothing_after_two_empty_answers(
    tmp_path, monkeypatch, fake_client, gate_module
):
    client = fake_client(
        _response("", finish_reason="length"),
        _response("   ", finish_reason="length"),
    )
    monkeypatch.setattr(gate_module, "build_client", lambda config: client)
    out = tmp_path / "gates.md"
    version = tmp_path / "judge_version.json"

    code = gate_module.main(_gate_args(tmp_path, out, version))

    assert code == 1
    assert len(client.calls) == 2
    log = out.read_text(encoding="utf-8")
    assert "status: FAIL" in log
    assert "finish_reason: length" in log
    assert not version.exists()


def test_default_paths_are_repo_relative_not_cwd(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]

    assert client_module.REPO_ROOT == root
    assert DEFAULT_VERSION_PATH_AT_IMPORT.is_absolute()
    assert DEFAULT_VERSION_PATH_AT_IMPORT == root / "docs" / "judge_version.json"
    assert DEFAULT_GATES_PATH_AT_IMPORT == root / "docs" / "gates.md"

    monkeypatch.chdir(tmp_path)
    assert DEFAULT_VERSION_PATH_AT_IMPORT == root / "docs" / "judge_version.json"
    assert DEFAULT_GATES_PATH_AT_IMPORT == root / "docs" / "gates.md"


def test_judge_pair_refuses_without_a_registered_version(tmp_path, fake_client):
    client_module.reset_judge_version()
    client = fake_client(_response(_verdict_json()))

    with pytest.raises(JudgeVersionError, match="registered judge version"):
        _judge(tmp_path=tmp_path, client=client)

    assert client.calls == []


def test_gate_records_model_mismatch_as_fail(
    tmp_path, monkeypatch, fake_client, gate_module
):
    client = fake_client(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            model="deepseek-other",
        )
    )
    monkeypatch.setattr(gate_module, "build_client", lambda config: client)
    out = tmp_path / "gates.md"
    version = tmp_path / "judge_version.json"

    code = gate_module.main(_gate_args(tmp_path, out, version))

    assert code == 1
    log = out.read_text(encoding="utf-8")
    assert "status: FAIL" in log
    assert "deepseek-other" in log
    assert "finish_reason: stop" in log
    assert not version.exists()
