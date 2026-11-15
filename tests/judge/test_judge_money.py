"""Money guards at the vision-judge and vision-gate call sites (fakes only)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import gemmaloop.judge.client as client_module
from gemmaloop.deepseek_money import (
    ModelMismatch,
    ModelRefused,
    SpendingCapExceeded,
    WasteGuard,
    WasteGuardTriggered,
)
from gemmaloop.judge.client import (
    JudgeClientConfig,
    JudgeVersionError,
    build_client,
    complete_json,
    judge_pair,
    judge_waste_journal_path,
)
from gemmaloop.judge.render import RenderResult
from gemmaloop.stage0.spend import SpendLedger

UTC = timezone.utc
OFFPEAK = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)
MONDAY_0050 = datetime(2026, 11, 2, 0, 50, tzinfo=UTC)
MONDAY_0405 = datetime(2026, 11, 2, 4, 5, tzinfo=UTC)

BRIEF_SPEC = {
    "brief_id": "b1",
    "kind": "page",
    "brief": "A landing page for a specialty coffee shop.",
    "required_sections": ["Hero", "Pricing"],
    "required_copy": ["Buy now"],
    "forbidden": ["lorem ipsum"],
}


def _verdict_json() -> str:
    return json.dumps(
        {
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
    )


def _response(
    content: str | None,
    *,
    model: object = "deepseek-flash",
    fingerprint: object = "fp-test",
    finish_reason: object = "stop",
    usage: dict | None = None,
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
            **(
                usage
                or {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            )
        ),
    )


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now = self.now + timedelta(seconds=float(seconds))


class FakeClient:
    def __init__(self, *responses, clock: FakeClock | None = None) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.call_times: list[datetime] = []
        self.clock = clock
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.clock is not None:
            self.call_times.append(self.clock.now)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


@pytest.fixture(autouse=True)
def _registered_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        client_module,
        "DEFAULT_JUDGE_VERSION_PATH",
        tmp_path / "judge_version.json",
    )
    client_module.reset_judge_version()
    client_module.register_judge_version("deepseek-flash", "fp-test")
    client_module.reset_default_judge_guard()
    yield
    client_module.reset_judge_version()
    client_module.reset_default_judge_guard()


def _render(tmp_path, name: str) -> RenderResult:
    directory = tmp_path / name
    directory.mkdir()
    desktop = directory / "desktop.png"
    mobile = directory / "mobile.png"
    desktop.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"-desktop")
    mobile.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"-mobile")
    return RenderResult(True, desktop, mobile, None)


def _judge(tmp_path, client, **kwargs):
    a = _render(tmp_path, f"a{len(client.calls)}")
    b = _render(tmp_path, f"b{len(client.calls)}")
    kwargs.setdefault("now_fn", lambda: OFFPEAK)
    kwargs.setdefault("waste_guard", WasteGuard(window=10_000, rate=1.0))
    return judge_pair(BRIEF_SPEC, a, b, client=client, **kwargs)


def test_judge_refuses_a_legacy_model_before_any_request(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    with pytest.raises(ModelRefused):
        _judge(tmp_path, client, model="deepseek-chat")
    assert client.calls == []


def test_judge_sends_explicit_thinking_and_low_effort_by_default(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    _judge(tmp_path, client)
    request = client.calls[0]
    assert request["model"] == "deepseek-flash"
    assert request["max_tokens"] == 4096
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_judge_accepts_high_effort_explicitly(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    _judge(tmp_path, client, reasoning_effort="high")
    assert client.calls[0]["reasoning_effort"] == "high"


def test_judge_rejects_a_non_integer_max_tokens(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    with pytest.raises(ValueError, match="max_tokens"):
        _judge(tmp_path, client, max_tokens=None)
    assert client.calls == []


def test_judge_waits_off_peak_before_the_request(tmp_path):
    clock = FakeClock(MONDAY_0050)
    client = FakeClient(_response(_verdict_json()), clock=clock)
    _judge(tmp_path, client, now_fn=clock, sleep_fn=clock.sleep)
    assert clock.sleeps == [pytest.approx(11700.0)]
    assert client.call_times == [MONDAY_0405]


def test_judge_reserves_and_settles_in_the_shared_ledger(tmp_path):
    client = FakeClient(
        _response(
            _verdict_json(),
            usage={
                "prompt_tokens": 2_000_000,
                "completion_tokens": 1_000_000,
                "total_tokens": 3_000_000,
                "prompt_cache_hit_tokens": 1_000_000,
                "prompt_cache_miss_tokens": 1_000_000,
            },
        )
    )
    ledger = SpendLedger(tmp_path / "spend.json")
    _judge(
        tmp_path,
        client,
        ledger=ledger,
        ledger_month="2026-11",
        max_usd=5.0,
    )
    entries = ledger.entries()
    reservation = next(e for e in entries if e["kind"] == "reservation")
    actual = next(e for e in entries if e["kind"] == "actual")
    assert reservation["provider"] == "deepseek"
    assert actual["usd"] == pytest.approx(0.003 + 0.15 + 0.60)
    assert ledger.cumulative_usd("2026-11", provider="deepseek") == pytest.approx(0.753)


def test_judge_cap_refuses_before_any_request(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    with pytest.raises(SpendingCapExceeded):
        _judge(tmp_path, client, max_usd=0.0000001)
    assert client.calls == []


def test_judge_default_ledger_makes_the_cap_run_level(tmp_path):
    client = FakeClient(
        _response(
            _verdict_json(),
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 4096},
        )
    )
    _judge(tmp_path, client, max_usd=0.004)
    with pytest.raises(SpendingCapExceeded):
        _judge(tmp_path, client, max_usd=0.004)
    assert len(client.calls) == 1


def test_judge_stops_after_the_first_non_flash_response(tmp_path):
    for index, bad in enumerate(
        ("deepseek-v3", "DeepSeek-V5-Flash-0301", "deepseek-v4-flash-lite")
    ):
        case = tmp_path / f"case{index}"
        case.mkdir()
        client = FakeClient(
            _response(_verdict_json(), model=bad),
            _response(_verdict_json()),
        )
        with pytest.raises((ModelMismatch, JudgeVersionError)):
            _judge(case, client)
        assert len(client.calls) == 1


def test_judge_accepts_the_exact_registered_served_version(tmp_path):
    client_module.reset_judge_version()
    client_module.register_judge_version("DeepSeek-V4.1-Flash-1017", "fp-test")
    client = FakeClient(_response(_verdict_json(), model="DeepSeek-V4.1-Flash-1017"))
    record = _judge(tmp_path, client)
    assert record.status == "ok"
    assert record.model == "DeepSeek-V4.1-Flash-1017"


def test_judge_requires_a_registered_fingerprint(tmp_path):
    client_module.reset_judge_version()
    client = FakeClient(_response(_verdict_json()))
    with pytest.raises(JudgeVersionError, match="registered judge version"):
        _judge(tmp_path, client)
    assert client.calls == []


def test_judge_default_waste_guard_stops_and_keeps_the_paid_record(tmp_path):
    client = FakeClient(
        _response("", finish_reason="length"),
        _response("", finish_reason="length"),
        _response("", finish_reason="length"),
    )
    first = judge_pair(
        BRIEF_SPEC,
        _render(tmp_path, "a"),
        _render(tmp_path, "b"),
        client=client,
        now_fn=lambda: OFFPEAK,
    )
    assert first.status == "invalid"
    assert first.waste_stopped is False
    with pytest.raises(WasteGuardTriggered) as excinfo:
        judge_pair(
            BRIEF_SPEC,
            _render(tmp_path, "c"),
            _render(tmp_path, "d"),
            client=client,
            now_fn=lambda: OFFPEAK,
        )
    record = excinfo.value.record
    assert record.status == "invalid"
    assert record.waste_stopped is True
    assert len(client.calls) == 3


def test_complete_json_refuses_an_unpinned_or_implicit_request(tmp_path):
    client = FakeClient(_response(_verdict_json()))
    from gemmaloop.deepseek_money import CallAccountant

    accountant = CallAccountant(max_usd=1.0, now_fn=lambda: OFFPEAK)
    base = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 16,
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    for overrides in (
        {"model": "deepseek-chat"},
        {"max_tokens": None},
        {"extra_body": {}},
    ):
        with pytest.raises((ValueError, ModelRefused)):
            complete_json(client, {**base, **overrides}, accountant=accountant)
    assert client.calls == []


def test_build_client_disables_sdk_retries_and_bounds_the_timeout(monkeypatch):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("MY_JUDGE_KEY", "secret-value")
    build_client(
        JudgeClientConfig(
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key_env="MY_JUDGE_KEY",
        ),
        client_factory=factory,
    )
    assert captured["max_retries"] == 0
    assert captured["timeout"] <= 300
    assert captured["base_url"] == "https://api.deepseek.com"


def test_build_client_refuses_every_non_official_endpoint(monkeypatch):
    monkeypatch.setenv("MY_JUDGE_KEY", "secret")
    for base_url in (
        "https://opencode.ai/v1",
        "http://api.deepseek.com",
        "https://example.com/v1",
    ):
        with pytest.raises(ValueError):
            build_client(
                JudgeClientConfig(base_url=base_url, api_key_env="MY_JUDGE_KEY"),
                client_factory=lambda **kwargs: object(),
                allow_opencode_hosted=True,
            )


# ---------------------------------------------------------------------------
# Vision gate CLI
# ---------------------------------------------------------------------------


def _load_gate_module():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "judge_vision_gate.py"
    )
    spec = importlib.util.spec_from_file_location("judge_vision_gate_money", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _gate_args(tmp_path, **overrides):
    args = {
        "--out": str(tmp_path / "gates.md"),
        "--version-out": str(tmp_path / "judge_version.json"),
        "--ledger": str(tmp_path / "spend.json"),
        "--spend-lock": str(tmp_path / "spend.lock"),
        "--init-ledger": None,
    }
    return [item for key, value in args.items() for item in ([key] if value is None else [key, value])]


def test_gate_refuses_a_legacy_requested_model_before_building_a_client(
    tmp_path, monkeypatch
):
    module = _load_gate_module()

    def explode(config):
        raise AssertionError("gate must refuse the model before building a client")

    monkeypatch.setattr(module, "build_client", explode)
    with pytest.raises(ModelRefused):
        module.main(["--model", "deepseek-chat", *_gate_args(tmp_path)])
    assert not (tmp_path / "spend.json").exists()


def test_gate_requires_the_shared_ledger_and_lock(tmp_path, monkeypatch):
    module = _load_gate_module()
    monkeypatch.setattr(
        module,
        "build_client",
        lambda config: pytest.fail("must refuse before building a client"),
    )
    with pytest.raises(SystemExit):
        module.main(["--out", str(tmp_path / "gates.md")])


def test_gate_stops_after_one_non_flash_response_and_records_fail(
    tmp_path, monkeypatch
):
    module = _load_gate_module()
    client = FakeClient(
        _response("not json", model="deepseek-v3"),
        _response(json.dumps({"left": "red square", "right": "blue circle"})),
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 1
    assert len(client.calls) == 1
    log = (tmp_path / "gates.md").read_text(encoding="utf-8")
    assert "status: FAIL" in log
    assert "deepseek-v3" in log


def test_gate_accepts_a_full_flash_version_and_pins_it(tmp_path, monkeypatch):
    module = _load_gate_module()
    client = FakeClient(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            model="DeepSeek-V4.1-Flash-1017",
        )
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    version = tmp_path / "judge_version.json"
    assert module.main(_gate_args(tmp_path)) == 0
    assert json.loads(version.read_text(encoding="utf-8")) == {
        "model": "DeepSeek-V4.1-Flash-1017",
        "system_fingerprint": "fp-test",
    }


def test_gate_request_is_explicit_and_cheap(tmp_path, monkeypatch):
    module = _load_gate_module()
    client = FakeClient(
        _response(json.dumps({"left": "red square", "right": "blue circle"}))
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 0
    request = client.calls[0]
    assert request["max_tokens"] == 4096
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_gate_waits_off_peak(tmp_path, monkeypatch):
    module = _load_gate_module()
    clock = FakeClock(MONDAY_0050)
    client = FakeClient(
        _response(json.dumps({"left": "red square", "right": "blue circle"})),
        clock=clock,
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    real_accountant = module.CallAccountant

    def factory(**kwargs):
        kwargs["now_fn"] = clock
        kwargs["sleep_fn"] = clock.sleep
        return real_accountant(**kwargs)

    monkeypatch.setattr(module, "CallAccountant", factory)
    assert module.main(_gate_args(tmp_path)) == 0
    assert clock.sleeps == [pytest.approx(11700.0)]
    assert client.call_times == [MONDAY_0405]


def test_gate_writes_the_default_shared_ledger(tmp_path, monkeypatch):
    module = _load_gate_module()
    client = FakeClient(
        _response(json.dumps({"left": "red square", "right": "blue circle"}))
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    args = [
        "--out",
        str(tmp_path / "gates.md"),
        "--version-out",
        str(tmp_path / "judge_version.json"),
        "--init-ledger",
        "--spend-lock",
        str(tmp_path / "spend.lock"),
    ]
    assert module.main(args) == 0
    from gemmaloop.stage0.spend import default_ledger_path

    ledger = SpendLedger(default_ledger_path())
    entries = ledger.entries()
    assert any(entry["kind"] == "reservation" for entry in entries)
    assert any(entry["kind"] == "actual" for entry in entries)

# ---------------------------------------------------------------------------
# Round 2: judge retry waste, attached records, registry and gate pins
# ---------------------------------------------------------------------------


def test_judge_waste_guard_counts_every_retry_and_persists(tmp_path):
    from gemmaloop.deepseek_money import read_journal

    journal = tmp_path / "waste.jsonl"
    client = FakeClient(
        _response("", finish_reason="length"),
        _response(_verdict_json()),
        _response("", finish_reason="length"),
        _response(_verdict_json()),
        _response("", finish_reason="length"),
        _response(_verdict_json()),
    )
    for index in range(2):
        case = tmp_path / f"case{index}"
        case.mkdir()
        record = judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=client,
            waste_journal=journal,
            now_fn=lambda: OFFPEAK,
        )
        assert record.status == "ok"

    case = tmp_path / "case2"
    case.mkdir()
    with pytest.raises(WasteGuardTriggered) as excinfo:
        judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=client,
            waste_journal=journal,
            now_fn=lambda: OFFPEAK,
        )
    assert excinfo.value.record.waste_stopped is True
    entries = read_journal(journal)
    assert sum(1 for entry in entries if entry.get("waste")) == 3

    late = FakeClient(_response(_verdict_json()))
    case = tmp_path / "case3"
    case.mkdir()
    with pytest.raises(WasteGuardTriggered):
        judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=late,
            waste_journal=journal,
            now_fn=lambda: OFFPEAK,
        )
    assert late.calls == []


def test_complete_json_requires_an_accountant():
    client = FakeClient(_response(_verdict_json()))
    request = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 16,
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    with pytest.raises(ValueError, match="CallAccountant"):
        complete_json(client, request, accountant=None)
    assert client.calls == []


def test_judge_fingerprint_mismatch_attaches_the_paid_record(tmp_path):
    client = FakeClient(_response(_verdict_json(), fingerprint="fp-other"))
    with pytest.raises(JudgeVersionError) as excinfo:
        _judge(tmp_path, client)
    record = getattr(excinfo.value, "record", None)
    assert record is not None
    assert record.stop_reason == "JudgeVersionError"
    assert record.attempts_log


def test_judge_peak_created_response_attaches_the_paid_record(tmp_path):
    from gemmaloop.deepseek_money import PeakRequestDetected

    peak_response = _response(_verdict_json())
    peak_response.created = datetime(2026, 11, 2, 2, 0, tzinfo=UTC).timestamp()
    client = FakeClient(peak_response)
    with pytest.raises(PeakRequestDetected) as excinfo:
        _judge(tmp_path, client)
    record = getattr(excinfo.value, "record", None)
    assert record is not None
    assert record.stop_reason == "PeakRequestDetected"
    assert excinfo.value.response is peak_response


def test_judge_records_both_attempts(tmp_path):
    client = FakeClient(_response(""), _response(_verdict_json()))
    record = _judge(tmp_path, client)
    assert record.status == "ok"
    assert record.attempts == 2
    assert len(record.attempts_log) == 2


def test_judge_uses_the_registered_fingerprint_record(tmp_path):
    from gemmaloop.deepseek_money import write_fingerprint_record

    path = tmp_path / "registered.json"
    write_fingerprint_record(
        model="deepseek-flash", fingerprint="fp-reg", source="test", path=path
    )
    client_module.reset_judge_version()
    client = FakeClient(_response(_verdict_json(), fingerprint="fp-reg"))
    record = judge_pair(
        BRIEF_SPEC,
        _render(tmp_path, "a"),
        _render(tmp_path, "b"),
        client=client,
        fingerprint_path=path,
        now_fn=lambda: OFFPEAK,
        waste_guard=WasteGuard(window=10_000, rate=1.0),
    )
    assert record.status == "ok"

    client_module.reset_judge_version()
    client = FakeClient(_response(_verdict_json(), fingerprint="fp-other"))
    with pytest.raises(JudgeVersionError):
        judge_pair(
            BRIEF_SPEC,
            _render(tmp_path, "c"),
            _render(tmp_path, "d"),
            client=client,
            fingerprint_path=path,
            now_fn=lambda: OFFPEAK,
            waste_guard=WasteGuard(window=10_000, rate=1.0),
        )


def test_gate_pass_writes_the_fingerprint_record(tmp_path, monkeypatch):
    from gemmaloop.deepseek_money import (
        fingerprint_record_path,
        load_fingerprint_record,
    )

    module = _load_gate_module()
    client = FakeClient(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            fingerprint="fp-gate",
        )
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 0
    record = load_fingerprint_record(fingerprint_record_path())
    assert record["fingerprint"] == "fp-gate"
    assert record["source"] == "vision_gate"
    assert record["date"]


def test_gate_pass_requires_a_non_empty_fingerprint(tmp_path, monkeypatch):
    from gemmaloop.deepseek_money import (
        fingerprint_record_path,
        load_fingerprint_record,
    )

    module = _load_gate_module()
    client = FakeClient(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            fingerprint=None,
        )
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 1
    assert not (tmp_path / "judge_version.json").exists()
    assert load_fingerprint_record(fingerprint_record_path()) is None


def test_gate_repin_is_deliberate(tmp_path, monkeypatch):
    from gemmaloop.deepseek_money import (
        fingerprint_record_path,
        load_fingerprint_record,
        write_fingerprint_record,
    )

    module = _load_gate_module()
    path = fingerprint_record_path()
    write_fingerprint_record(
        model="deepseek-flash",
        fingerprint="fp-old",
        source="test",
        path=path,
        repin=True,
        created="2026-01-01T00:00:00+00:00",
    )
    client = FakeClient(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            fingerprint="fp-new",
        )
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 1
    assert load_fingerprint_record(path)["fingerprint"] == "fp-old"

    client = FakeClient(
        _response(
            json.dumps({"left": "red square", "right": "blue circle"}),
            fingerprint="fp-new",
        )
    )
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path) + ["--repin"]) == 0
    record = load_fingerprint_record(path)
    assert record["fingerprint"] == "fp-new"
    assert record["date"] != "2026-01-01T00:00:00+00:00"


def test_gate_peak_created_response_records_fail(tmp_path, monkeypatch):
    module = _load_gate_module()
    peak_response = _response(
        json.dumps({"left": "red square", "right": "blue circle"}),
        fingerprint="fp-test",
    )
    peak_response.created = datetime(2026, 11, 2, 2, 0, tzinfo=UTC).timestamp()
    client = FakeClient(peak_response)
    monkeypatch.setattr(module, "build_client", lambda config: client)
    assert module.main(_gate_args(tmp_path)) == 1
    log = (tmp_path / "gates.md").read_text(encoding="utf-8")
    assert "status: FAIL" in log
    assert "peak window" in log
    assert "attempts:" in log

def test_judge_persists_waste_by_default_and_has_memory_mode(tmp_path):
    out_dir = tmp_path / "judge-run"
    client = FakeClient(
        _response("", finish_reason="length"),
        _response(_verdict_json()),
        _response("", finish_reason="length"),
        _response(_verdict_json()),
        _response("", finish_reason="length"),
        _response(_verdict_json()),
    )
    for index in range(2):
        case = tmp_path / f"run{index}"
        case.mkdir()
        record = judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=client,
            out_dir=out_dir,
            now_fn=lambda: OFFPEAK,
        )
        assert record.status == "ok"

    case = tmp_path / "run2"
    case.mkdir()
    with pytest.raises(WasteGuardTriggered):
        judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=client,
            out_dir=out_dir,
            now_fn=lambda: OFFPEAK,
        )
    journal = judge_waste_journal_path(out_dir)
    assert journal.exists()
    from gemmaloop.deepseek_money import read_journal

    assert sum(1 for entry in read_journal(journal) if entry.get("waste")) == 3

    late = FakeClient(_response(_verdict_json()))
    case = tmp_path / "run3"
    case.mkdir()
    with pytest.raises(WasteGuardTriggered):
        judge_pair(
            BRIEF_SPEC,
            _render(case, "a"),
            _render(case, "b"),
            client=late,
            out_dir=out_dir,
            now_fn=lambda: OFFPEAK,
        )
    assert late.calls == []

    memory_dir = tmp_path / "memory-run"
    memory_client = FakeClient(_response(_verdict_json()))
    case = tmp_path / "run4"
    case.mkdir()
    judge_pair(
        BRIEF_SPEC,
        _render(case, "a"),
        _render(case, "b"),
        client=memory_client,
        out_dir=memory_dir,
        in_memory_waste_guard=True,
        now_fn=lambda: OFFPEAK,
    )
    assert not judge_waste_journal_path(memory_dir).exists()
