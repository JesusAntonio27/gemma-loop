"""Money guards for every DeepSeek call site.

All clients are fake and all clocks are injected: no network, no API key and
no real sleep.  Each test targets one guard from the round-1 reviews.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gemmaloop.deepseek_money import (
    API_MODEL,
    CallAccountant,
    ModelMismatch,
    ModelRefused,
    OffPeakGateError,
    PeakRequestDetected,
    PilotReportError,
    SpendingCapExceeded,
    WasteGuard,
    WasteTripAcknowledgmentRequired,
    build_pilot_report,
    check_pilot_projection,
    enforce_flash_response,
    estimate_cost,
    estimate_request_prompt_tokens,
    estimate_tokens,
    is_peak,
    load_fingerprint_record,
    load_pilot_report,
    pilot_config_sha256,
    seconds_until_off_peak,
    validate_model,
    validate_pilot_report,
    wait_for_off_peak,
    write_fingerprint_record,
    write_pilot_report,
)
from gemmaloop.stage0.deepseek import HiddenTestWriter
from gemmaloop.stage0.spend import SpendLedger

UTC = timezone.utc
OFFPEAK = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)  # Monday, 05:00 UTC
MONDAY_0045 = datetime(2026, 11, 2, 0, 45, tzinfo=UTC)
MONDAY_0050 = datetime(2026, 11, 2, 0, 50, tzinfo=UTC)
MONDAY_0200 = datetime(2026, 11, 2, 2, 0, tzinfo=UTC)
MONDAY_0350 = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)
MONDAY_0400 = datetime(2026, 11, 2, 4, 0, tzinfo=UTC)
MONDAY_0405 = datetime(2026, 11, 2, 4, 5, tzinfo=UTC)
MONDAY_0550 = datetime(2026, 11, 2, 5, 50, tzinfo=UTC)
FRIDAY_0550 = datetime(2026, 11, 6, 5, 50, tzinfo=UTC)
FRIDAY_1004 = datetime(2026, 11, 6, 10, 4, tzinfo=UTC)
SATURDAY_0050 = datetime(2026, 11, 7, 0, 50, tzinfo=UTC)


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
    def __init__(self, responses, clock: FakeClock | None = None) -> None:
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


def response(
    content: str = '{"tests": [{"input": "1", "output": "1"}]}',
    *,
    model: object = "deepseek-flash",
    fingerprint: object = "fp-test",
    finish_reason: object = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
    usage: object = "auto",
):
    if usage == "auto":
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cache_hit_tokens is not None:
            usage["prompt_cache_hit_tokens"] = cache_hit_tokens
        if cache_miss_tokens is not None:
            usage["prompt_cache_miss_tokens"] = cache_miss_tokens
    return SimpleNamespace(
        id="resp",
        model=model,
        system_fingerprint=fingerprint,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=(
            SimpleNamespace(**usage)
            if isinstance(usage, dict)
            else usage
        ),
    )


def _problem(problem_id: str = "p:1") -> dict:
    return {
        "problem_id": problem_id,
        "source": "taco",
        "prompt": "Add two numbers from stdin.",
        "tests": [],
        "metadata": {"reference_solution": "print(1)"},
    }


class FakeGrader:
    def grade_code(self, code, tests):
        return SimpleNamespace(passed=True)


def _writer(client, clock, **kwargs):
    kwargs.setdefault("max_usd", 5.0)
    kwargs.setdefault("dry_run", False)
    kwargs.setdefault("grader", FakeGrader())
    return HiddenTestWriter(
        client=client,
        now_fn=clock,
        sleep_fn=clock.sleep,
        **kwargs,
    )


def test_peak_gate_has_margins_at_both_ends():
    # Inside a peak window: sleep to its end plus the 5-minute margin.
    assert seconds_until_off_peak(MONDAY_0200) == pytest.approx(7500.0)
    assert seconds_until_off_peak(MONDAY_0350) == pytest.approx(900.0)
    # Within 15 minutes before a peak: sleep through the whole window + margin.
    assert seconds_until_off_peak(MONDAY_0045) == pytest.approx(12000.0)
    assert seconds_until_off_peak(MONDAY_0050) == pytest.approx(11700.0)
    assert seconds_until_off_peak(MONDAY_0550) == pytest.approx(15300.0)
    assert seconds_until_off_peak(FRIDAY_0550) == pytest.approx(15300.0)
    # The 5 minutes after a window are gated too.
    assert seconds_until_off_peak(MONDAY_0400) == pytest.approx(300.0)
    assert seconds_until_off_peak(FRIDAY_1004) == pytest.approx(60.0)
    # Safe gaps and weekends are untouched.
    assert seconds_until_off_peak(MONDAY_0405) == 0.0
    assert seconds_until_off_peak(SATURDAY_0050) == 0.0
    assert not is_peak(MONDAY_0045)
    assert is_peak(MONDAY_0200)


def test_wait_for_off_peak_sleeps_to_the_safe_boundary():
    clock = FakeClock(MONDAY_0050)
    waited = wait_for_off_peak(now_fn=clock, sleep_fn=clock.sleep)
    assert waited == pytest.approx(11700.0)
    assert clock.sleeps == [pytest.approx(11700.0)]
    assert clock.now == MONDAY_0405


def test_wait_for_off_peak_fails_closed_on_a_non_advancing_clock():
    clock = FakeClock(MONDAY_0200)
    slept: list[float] = []
    with pytest.raises(OffPeakGateError):
        wait_for_off_peak(now_fn=clock, sleep_fn=slept.append)
    assert slept == [pytest.approx(7500.0)]


def test_naive_datetimes_are_refused():
    naive = datetime(2026, 11, 2, 2, 0)
    with pytest.raises(ValueError, match="naive"):
        is_peak(naive)
    with pytest.raises(ValueError, match="naive"):
        seconds_until_off_peak(naive)
    with pytest.raises(ValueError, match="naive"):
        wait_for_off_peak(now_fn=lambda: naive, sleep_fn=lambda _s: None)
    with pytest.raises(ValueError, match="naive"):
        CallAccountant(max_usd=1.0, now_fn=lambda: naive).before_request(
            prompt_tokens=10, max_output_tokens=10
        )


def test_only_deepseek_flash_is_accepted_as_the_request_model():
    assert validate_model("deepseek-flash") == "deepseek-flash"
    for legacy in ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro", ""):
        with pytest.raises(ModelRefused):
            validate_model(legacy)


def test_response_model_must_be_the_exact_served_version_or_alias():
    assert enforce_flash_response(response(model="DeepSeek-V4.1-Flash-1017"))[0] == (
        "DeepSeek-V4.1-Flash-1017"
    )
    assert enforce_flash_response(response(model="deepseek-flash"))[0] == "deepseek-flash"
    for bad in (
        "deepseek-chat",
        "deepseek-v3",
        "DeepSeek-V5-Flash-0301",
        "deepseek-v4-flash-lite",
        None,
    ):
        with pytest.raises(ModelMismatch):
            enforce_flash_response(response(model=bad))


def test_estimate_tokens_is_an_upper_bound():
    assert estimate_tokens("a" * 100) >= 50
    request = {
        "messages": [
            {"role": "system", "content": "x" * 100},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "y" * 100},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
    }
    assert estimate_request_prompt_tokens(request) >= 50 + 50 + 2048


def test_writer_refuses_a_legacy_model_before_any_request():
    client = FakeClient([response()])
    with pytest.raises(ModelRefused):
        HiddenTestWriter(client=client, model="deepseek-chat", dry_run=False)
    assert client.calls == []


def test_writer_waits_off_peak_before_each_request():
    clock = FakeClock(MONDAY_0050)
    client = FakeClient([response()], clock=clock)
    writer = _writer(client, clock, max_output_tokens=4096)
    result = writer.write([_problem("p:1")])
    assert result.calls == 1
    assert clock.sleeps == [pytest.approx(11700.0)]
    assert client.call_times == [MONDAY_0405]


def test_writer_sends_explicit_thinking_disabled_and_max_tokens():
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response()], clock=clock)
    writer = _writer(client, clock, max_output_tokens=4096)
    writer.write([_problem()])
    kwargs = client.calls[0]
    assert kwargs["model"] == API_MODEL == "deepseek-flash"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_writer_default_clock_is_the_patched_module_clock():
    writer = HiddenTestWriter(dry_run=False, max_usd=1.0)
    assert writer._now() == OFFPEAK


def test_writer_stops_after_the_first_response_is_not_flash():
    clock = FakeClock(OFFPEAK)
    client = FakeClient(
        [response(model="deepseek-v3"), response(model="deepseek-flash")],
        clock=clock,
    )
    writer = _writer(client, clock)
    result = writer.write([_problem("p:1"), _problem("p:2")])
    assert len(client.calls) == 1
    assert result.calls == 1
    assert result.stopped_reason == "model_mismatch"


def test_writer_pins_the_first_system_fingerprint():
    clock = FakeClock(OFFPEAK)
    client = FakeClient(
        [
            response(fingerprint="fp-1"),
            response(fingerprint="fp-2"),
            response(fingerprint="fp-2"),
        ],
        clock=clock,
    )
    writer = _writer(client, clock)
    result = writer.write([_problem("p:1"), _problem("p:2"), _problem("p:3")])
    assert len(client.calls) == 2
    assert result.stopped_reason == "fingerprint_mismatch"


def test_writer_records_response_model_and_fingerprint():
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response(model="deepseek-flash", fingerprint="fp-1")], clock=clock)
    writer = _writer(client, clock)
    result = writer.write([_problem("p:1")])
    record = result.records[0]
    assert record["response_model"] == "deepseek-flash"
    assert record["system_fingerprint"] == "fp-1"
    assert result.response_models == ["deepseek-flash"]
    assert result.system_fingerprints == ["fp-1"]


def test_writer_has_no_ungated_request_helper():
    assert not hasattr(HiddenTestWriter, "_call")


def test_writer_reserves_worst_case_and_settles_cache_hits_cheaply(tmp_path):
    clock = FakeClock(OFFPEAK)
    client = FakeClient(
        [
            response(
                prompt_tokens=2_000_000,
                completion_tokens=1_000_000,
                cache_hit_tokens=1_000_000,
                cache_miss_tokens=1_000_000,
            )
        ],
        clock=clock,
    )
    ledger = SpendLedger(tmp_path / "spend.json")
    writer = _writer(
        client,
        clock,
        ledger=ledger,
        ledger_month="2026-11",
        max_output_tokens=4096,
    )
    problem = _problem("p:1")
    writer.write([problem])
    entries = ledger.entries()
    reservation = next(e for e in entries if e["kind"] == "reservation")
    actual = next(e for e in entries if e["kind"] == "actual")
    expected_reservation = estimate_cost(
        writer._prompt_tokens(problem), 4096
    )["total_usd"]
    assert reservation["provider"] == "deepseek"
    assert reservation["usd"] == pytest.approx(expected_reservation)
    # 1M cache-hit input at $0.003 + 1M miss at $0.15 + 1M output at $0.60.
    assert actual["usd"] == pytest.approx(0.003 + 0.15 + 0.60)
    assert ledger.cumulative_usd("2026-11", provider="deepseek") == pytest.approx(0.753)


def test_writer_run_spend_excludes_previous_monthly_entries(tmp_path):
    clock = FakeClock(OFFPEAK)
    client = FakeClient(
        [response(prompt_tokens=1_000_000, completion_tokens=0)], clock=clock
    )
    ledger = SpendLedger(tmp_path / "spend.json")
    previous = ledger.reserve(shard=0, usd=1.0, month="2026-11", provider="deepseek")
    ledger.settle(
        previous, shard=0, actual_usd=1.0, month="2026-11", provider="deepseek"
    )
    writer = _writer(client, clock, ledger=ledger, ledger_month="2026-11")
    result = writer.write([_problem("p:1")])
    assert result.run_usd == pytest.approx(0.15)
    assert result.spent_usd == pytest.approx(1.15)


def test_missing_usage_settles_at_the_full_reservation(tmp_path):
    clock = FakeClock(OFFPEAK)
    missing = response(usage=None)
    client = FakeClient([missing], clock=clock)
    ledger = SpendLedger(tmp_path / "spend.json")
    accountant = CallAccountant(
        ledger=ledger, month="2026-11", max_usd=5.0, now_fn=clock, sleep_fn=clock.sleep
    )
    reservation, estimate = accountant.before_request(
        prompt_tokens=1000, max_output_tokens=4096
    )
    actual, _usage = accountant.after_response(reservation, missing)
    assert actual["usage_missing"] is True
    assert actual["total_usd"] == pytest.approx(estimate["total_usd"])
    assert ledger.cumulative_usd("2026-11", provider="deepseek") == pytest.approx(
        estimate["total_usd"]
    )


def test_actual_above_reservation_is_flagged():
    clock = FakeClock(OFFPEAK)
    accountant = CallAccountant(max_usd=5.0, now_fn=clock, sleep_fn=clock.sleep)
    reservation, _estimate = accountant.before_request(
        prompt_tokens=10, max_output_tokens=10
    )
    accountant.after_response(
        reservation, response(prompt_tokens=100, completion_tokens=1_000_000)
    )
    assert accountant.last_actual_exceeds_reservation is True


def test_server_created_timestamp_in_peak_settles_at_double_and_stops():
    clock = FakeClock(OFFPEAK)
    accountant = CallAccountant(max_usd=5.0, now_fn=clock, sleep_fn=clock.sleep)
    reservation, _estimate = accountant.before_request(
        prompt_tokens=1000, max_output_tokens=100
    )
    peak_response = response(prompt_tokens=1_000_000, completion_tokens=0)
    peak_response.created = datetime(2026, 11, 2, 2, 0, tzinfo=UTC).timestamp()
    with pytest.raises(PeakRequestDetected) as excinfo:
        accountant.after_response(reservation, peak_response)
    assert excinfo.value.actual["peak"] is True
    # 1M input at $0.15/M x 2 (peak).
    assert accountant.cumulative_usd() == pytest.approx(0.30)


def test_writer_client_disables_sdk_retries_and_sets_timeout(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-not-a-real-key")
    writer = HiddenTestWriter(dry_run=False, max_usd=1.0)
    writer._get_client()
    assert captured["max_retries"] == 0
    assert captured["timeout"] <= 300


def test_waste_guard_uses_an_absolute_count_of_three():
    guard = WasteGuard()
    assert guard.record("stop", "ok") is False
    assert guard.record("length", "") is False  # 1 waste does not stop
    assert guard.record("length", "") is False  # 2 waste does not stop
    assert guard.record("length", "") is True  # 3 waste stops
    assert guard.triggered is True
    assert guard.waste_count == 3
    # The window slides: three waste in the last 20 trips even after clean ones.
    clean = WasteGuard()
    for _ in range(19):
        assert clean.record("stop", "ok") is False
    assert clean.record("length", "") is False  # 1/20
    assert clean.record("length", "") is False  # 2/20
    assert clean.record("stop", "ok") is False
    assert clean.record("length", "") is True  # 3/20


def test_writer_waste_guard_journals_and_resume_never_repays(tmp_path):
    clock = FakeClock(OFFPEAK)
    waste = [response(content="", finish_reason="length") for _ in range(3)]
    client = FakeClient(waste + [response() for _ in range(5)], clock=clock)
    partial = tmp_path / "hidden.partial.jsonl"
    writer = _writer(client, clock, partial_path=partial)
    result = writer.write([_problem(f"p:{index}") for index in range(5)])
    assert len(client.calls) == 3
    assert result.stopped_reason == "waste_guard"
    assert result.waste_trip_id
    journal_path = partial.with_name(partial.name + ".responses.jsonl")
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    responses = [entry for entry in journal if entry.get("kind") == "response"]
    assert [entry["problem_id"] for entry in responses] == ["p:0", "p:1", "p:2"]
    assert all(entry["waste"] is True for entry in responses)
    assert any(entry.get("kind") == "waste_trip" for entry in journal)

    # Without the deliberate ack the output path refuses to start.
    second = FakeClient([response()], clock=clock)
    with pytest.raises(WasteTripAcknowledgmentRequired):
        _writer(second, clock, partial_path=partial).write(
            [_problem(f"p:{index}") for index in range(3)]
        )
    assert second.calls == []

    # With the ack a new window starts, and recorded items are never re-bought.
    third = FakeClient([response()], clock=clock)
    resumed = _writer(
        third,
        clock,
        partial_path=partial,
        ack_waste_trip=result.waste_trip_id,
    )
    resumed_result = resumed.write([_problem(f"p:{index}") for index in range(3)])
    assert third.calls == []
    assert resumed_result.resumed == 3


def test_pilot_report_projection_and_cap(tmp_path):
    report = build_pilot_report(
        requests=20,
        input_miss_tokens=1000,
        input_hit_tokens=500,
        output_tokens=200,
        waste_count=1,
        usd=0.02,
        pilot_items=20,
        total_items=200,
    )
    assert report["requests"] == 20
    assert report["waste_rate"] == pytest.approx(0.05)
    assert report["projected_usd"] == pytest.approx(0.2)
    assert report["projected_total_usd"] == pytest.approx(0.2)
    path = write_pilot_report(tmp_path / "pilot.json", report)
    assert load_pilot_report(path)["projected_usd"] == pytest.approx(0.2)
    assert check_pilot_projection(report, 5.0) == pytest.approx(0.2)
    with pytest.raises(SpendingCapExceeded):
        check_pilot_projection(report, 0.1)


def test_pilot_projection_includes_month_to_date_and_failures():
    report = build_pilot_report(
        requests=4,
        usd=0.01,
        pilot_items=4,
        total_items=40,
        waste_count=0,
        month_to_date_usd=0.5,
    )
    assert report["projected_usd"] == pytest.approx(0.1)
    assert report["projected_total_usd"] == pytest.approx(0.6)
    with pytest.raises(SpendingCapExceeded):
        check_pilot_projection(report, 0.55)


@pytest.mark.parametrize(
    "overrides",
    [
        {"requests": 0},
        {"stopped_reason": "error:RuntimeError"},
        {"stopped_reason": "waste_guard"},
        {"waste_count": 4},
        {"usd": float("nan")},
        {"usd": -1.0},
        {"projected_usd": float("inf")},
    ],
)
def test_pilot_report_rejects_invalid_values(overrides):
    with pytest.raises(PilotReportError):
        report = build_pilot_report(
            requests=20,
            usd=0.02,
            pilot_items=20,
            total_items=200,
            waste_count=0,
        )
        report.update(overrides)
        validate_pilot_report(report)


def test_pilot_report_binds_config_and_candidates():
    config = {"model": "deepseek-flash", "max_output_tokens": 4096, "limit": None}
    report = build_pilot_report(
        requests=2,
        usd=0.001,
        pilot_items=2,
        total_items=2,
        config=config,
        candidate_ids=["a", "b"],
        pilot_candidate_ids=["a", "b"],
    )
    assert report["config_sha256"] == pilot_config_sha256(config)
    validate_pilot_report(
        report,
        max_usd=5.0,
        expected_config_sha256=pilot_config_sha256(config),
        expected_candidate_ids=["a", "b"],
    )
    with pytest.raises(PilotReportError):
        validate_pilot_report(
            report,
            max_usd=5.0,
            expected_config_sha256=pilot_config_sha256({**config, "max_output_tokens": 8192}),
        )
    with pytest.raises(PilotReportError):
        validate_pilot_report(
            report, max_usd=5.0, expected_candidate_ids=["a", "b", "c"]
        )


def _load_cli():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "write_stage0_hidden_tests.py"
    )
    spec = importlib.util.spec_from_file_location("write_stage0_offpeak_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_cli_has_no_offpeak_switch_and_pilots_by_default():
    module = _load_cli()
    args = module._parse_args([])
    assert not hasattr(args, "off_peak")
    assert args.pilot == 20
    assert args.after_pilot is None
    with pytest.raises(SystemExit):
        module._parse_args(["--off-peak"])
    with pytest.raises(SystemExit):
        module.main(["--pilot", "51", "--pool-dir", "missing"])


def _write_pool(path, count: int):
    path.mkdir()
    rows = [
        {
            "problem_id": f"code:{index}",
            "source": "taco",
            "kind": "code",
            "prompt": "add",
            "tests": [],
            "metadata": {"reference_solution": "print(1)"},
        }
        for index in range(count)
    ]
    (path / "problems.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class _FakeGraderService:
    def __init__(self, *args, **kwargs):
        pass

    def grade_code(self, code, tests):
        return SimpleNamespace(passed=True)

    def close(self):
        pass


def _install_cli_fakes(module, monkeypatch, client, clock):
    real_writer = HiddenTestWriter

    def factory(**kwargs):
        return real_writer(
            client=client,
            now_fn=clock,
            sleep_fn=clock.sleep,
            **kwargs,
        )

    monkeypatch.setattr(module, "HiddenTestWriter", factory)
    monkeypatch.setattr(module, "GraderService", _FakeGraderService)


def _base_args(tmp_path, pool, output, extra):
    return [
        "--live",
        "--pool-dir",
        str(pool),
        "--output",
        str(output),
        "--max-usd",
        "5",
        "--spend-lock",
        str(tmp_path / "spend.lock"),
        "--ledger",
        str(tmp_path / "spend.json"),
        "--init-ledger",
        *extra,
    ]


def test_cli_live_run_never_calls_the_api_at_a_peak_time(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(MONDAY_0050)
    client = FakeClient([response()], clock=clock)
    _install_cli_fakes(module, monkeypatch, client, clock)
    pool = _write_pool(tmp_path / "pool", 1)
    assert (
        module.main(
            _base_args(
                tmp_path,
                pool,
                tmp_path / "hidden.jsonl",
                ["--pilot", "1"],
            )
        )
        == 0
    )
    assert clock.sleeps == [pytest.approx(11700.0)]
    assert client.call_times == [MONDAY_0405]


def test_cli_pilot_run_stops_and_writes_a_report(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response() for _ in range(5)], clock=clock)
    _install_cli_fakes(module, monkeypatch, client, clock)
    pool = _write_pool(tmp_path / "pool", 3)
    output = tmp_path / "hidden.jsonl"
    assert (
        module.main(_base_args(tmp_path, pool, output, ["--pilot", "2"]))
        == 0
    )
    assert len(client.calls) == 2
    report = json.loads(output.with_suffix(output.suffix + ".pilot.json").read_text())
    assert report["requests"] == 2
    assert report["pilot_items"] == 2
    assert report["total_items"] == 3
    assert report["stopped_reason"] is None
    assert report["projected_usd"] == pytest.approx(report["usd"] * 3 / 2)
    assert report["candidate_ids"] == ["code:0", "code:1", "code:2"]
    assert report["pilot_candidate_ids"] == ["code:0", "code:1"]


def test_cli_pilot_report_cannot_be_overwritten(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response() for _ in range(5)], clock=clock)
    _install_cli_fakes(module, monkeypatch, client, clock)
    pool = _write_pool(tmp_path / "pool", 3)
    output = tmp_path / "hidden.jsonl"
    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "2"])) == 0
    first = output.with_suffix(output.suffix + ".pilot.json").read_text()
    calls = len(client.calls)
    with pytest.raises(SystemExit):
        module.main(_base_args(tmp_path, pool, output, ["--pilot", "2"]))
    assert len(client.calls) == calls
    assert output.with_suffix(output.suffix + ".pilot.json").read_text() == first


def test_cli_after_pilot_refuses_when_projection_exceeds_cap(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response()], clock=clock)
    _install_cli_fakes(module, monkeypatch, client, clock)
    pool = _write_pool(tmp_path / "pool", 3)
    config = {
        "model": "deepseek-flash",
        "max_output_tokens": 4096,
        "limit": None,
    }
    report_path = tmp_path / "pilot.json"
    write_pilot_report(
        report_path,
        build_pilot_report(
            requests=2,
            input_miss_tokens=10,
            input_hit_tokens=0,
            output_tokens=5,
            waste_count=0,
            usd=2.0,
            pilot_items=2,
            total_items=100,
            config=config,
            candidate_ids=["code:0", "code:1", "code:2"],
        ),
    )
    with pytest.raises(SpendingCapExceeded):
        module.main(
            _base_args(
                tmp_path,
                pool,
                tmp_path / "hidden.jsonl",
                ["--after-pilot", str(report_path)],
            )
        )
    assert client.calls == []


def test_cli_after_pilot_refuses_a_different_config(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    client = FakeClient([response() for _ in range(4)], clock=clock)
    _install_cli_fakes(module, monkeypatch, client, clock)
    pool = _write_pool(tmp_path / "pool", 4)
    output = tmp_path / "hidden.jsonl"
    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "2"])) == 0
    report_path = output.with_suffix(output.suffix + ".pilot.json")
    calls = len(client.calls)
    with pytest.raises(PilotReportError):
        module.main(
            _base_args(
                tmp_path,
                pool,
                output,
                ["--after-pilot", str(report_path), "--max-output-tokens", "8192"],
            )
        )
    assert len(client.calls) == calls


def test_cli_after_pilot_success_processes_the_remaining_items(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    first_client = FakeClient([response() for _ in range(4)], clock=clock)
    _install_cli_fakes(module, monkeypatch, first_client, clock)
    pool = _write_pool(tmp_path / "pool", 4)
    output = tmp_path / "hidden.jsonl"
    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "2"])) == 0
    assert len(first_client.calls) == 2

    second_client = FakeClient([response() for _ in range(4)], clock=clock)
    _install_cli_fakes(module, monkeypatch, second_client, clock)
    report_path = output.with_suffix(output.suffix + ".pilot.json")
    assert (
        module.main(
            _base_args(tmp_path, pool, output, ["--after-pilot", str(report_path)])
        )
        == 0
    )
    assert len(second_client.calls) == 2
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["problem_id"] for record in records] == [
        "code:0",
        "code:1",
        "code:2",
        "code:3",
    ]

# ---------------------------------------------------------------------------
# Round 2: interrupted pilots, waste ack, registry, deadline and minors
# ---------------------------------------------------------------------------


def test_writer_two_early_waste_do_not_stop():
    clock = FakeClock(OFFPEAK)
    client = FakeClient(
        [
            response(content="", finish_reason="length"),
            response(content="", finish_reason="length"),
            response(),
            response(),
        ],
        clock=clock,
    )
    writer = _writer(client, clock)
    result = writer.write([_problem(f"p:{index}") for index in range(4)])
    assert len(client.calls) == 4
    assert result.stopped_reason is None
    assert result.waste == 2


def test_pilot_report_rejects_unacknowledged_waste():
    report = build_pilot_report(
        requests=20,
        usd=0.02,
        pilot_items=20,
        total_items=20,
        waste_count=3,
        waste_acknowledged=False,
    )
    with pytest.raises(PilotReportError):
        validate_pilot_report(report)
    acknowledged = build_pilot_report(
        requests=20,
        usd=0.02,
        pilot_items=20,
        total_items=20,
        waste_count=3,
        waste_acknowledged=True,
        waste_trip_ids=["trip-1"],
        acked_waste_trip_ids=["trip-1"],
    )
    assert validate_pilot_report(acknowledged) == pytest.approx(0.02)


def test_cli_interrupted_pilot_resumes_and_report_aggregates(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)

    class FailOnFourth(FakeClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 4:
                raise RuntimeError("network reset")
            return response()

    first = FailOnFourth([], clock=clock)
    _install_cli_fakes(module, monkeypatch, first, clock)
    pool = _write_pool(tmp_path / "pool", 30)
    output = tmp_path / "hidden.jsonl"

    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "20"])) == 0
    report1 = json.loads(
        output.with_suffix(output.suffix + ".pilot.json").read_text(encoding="utf-8")
    )
    assert report1["requests"] == 4  # 3 responses + the failed 4th attempt
    assert report1["accounted_items"] == 3
    assert report1["stopped_reason"].startswith("error:")

    second = FakeClient([response() for _ in range(40)], clock=clock)
    _install_cli_fakes(module, monkeypatch, second, clock)
    report2_path = tmp_path / "p2.json"
    assert (
        module.main(
            _base_args(
                tmp_path,
                pool,
                output,
                ["--pilot", "20", "--pilot-report", str(report2_path)],
            )
        )
        == 0
    )
    report2 = json.loads(report2_path.read_text(encoding="utf-8"))
    assert report2["requests"] == 21
    assert report2["pilot_items"] == 20
    assert report2["accounted_items"] == 20
    assert report2["stopped_reason"] is None
    assert report2["config_sha256"] == report1["config_sha256"]

    third = FakeClient([response() for _ in range(20)], clock=clock)
    _install_cli_fakes(module, monkeypatch, third, clock)
    assert (
        module.main(
            _base_args(
                tmp_path, pool, output, ["--after-pilot", str(report2_path)]
            )
        )
        == 0
    )
    assert len(third.calls) == 10


def test_cli_waste_trip_requires_ack_and_then_resumes(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    first = FakeClient(
        [response(content="", finish_reason="length") for _ in range(3)]
        + [response() for _ in range(30)],
        clock=clock,
    )
    _install_cli_fakes(module, monkeypatch, first, clock)
    pool = _write_pool(tmp_path / "pool", 30)
    output = tmp_path / "hidden.jsonl"

    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "20"])) == 0
    report_path = output.with_suffix(output.suffix + ".pilot.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stopped_reason"] == "waste_guard"
    trip_id = report["waste_trip_id"]
    assert trip_id

    without_ack = FakeClient([response() for _ in range(30)], clock=clock)
    _install_cli_fakes(module, monkeypatch, without_ack, clock)
    with pytest.raises(SystemExit):
        module.main(_base_args(tmp_path, pool, output, ["--pilot", "20"]))
    assert without_ack.calls == []

    with_ack = FakeClient([response() for _ in range(30)], clock=clock)
    _install_cli_fakes(module, monkeypatch, with_ack, clock)
    assert (
        module.main(
            _base_args(
                tmp_path,
                pool,
                output,
                ["--pilot", "20", "--ack-waste-trip", trip_id],
            )
        )
        == 0
    )
    assert len(with_ack.calls) == 17  # the 3 paid waste items are not re-bought
    completed = json.loads(report_path.read_text(encoding="utf-8"))
    assert completed["stopped_reason"] is None
    assert completed["waste_acknowledged"] is True
    assert completed["accounted_items"] == 20


def test_writer_registers_and_checks_the_fingerprint_record(tmp_path):
    clock = FakeClock(OFFPEAK)
    fingerprint_path = tmp_path / "fingerprint.json"

    first = FakeClient([response(fingerprint="fp-1")], clock=clock)
    result = _writer(
        first, clock, fingerprint_path=fingerprint_path
    ).write([_problem("p:1")])
    assert result.stopped_reason is None
    record = load_fingerprint_record(fingerprint_path)
    assert record["fingerprint"] == "fp-1"
    assert record["model"] == "deepseek-flash"
    assert record["source"] == "writer"
    assert record["date"]

    mismatch = FakeClient([response(fingerprint="fp-2")], clock=clock)
    result = _writer(
        mismatch, clock, fingerprint_path=fingerprint_path
    ).write([_problem("p:2")])
    assert result.stopped_reason == "fingerprint_mismatch"

    write_fingerprint_record(
        model="DeepSeek-V4.1-Flash-1017",
        fingerprint="fp-alias",
        source="test",
        path=fingerprint_path,
        repin=True,
    )
    alias = FakeClient(
        [response(model="deepseek-flash", fingerprint="fp-alias")], clock=clock
    )
    result = _writer(alias, clock, fingerprint_path=fingerprint_path).write(
        [_problem("p:3")]
    )
    assert result.stopped_reason is None


def test_writer_model_mismatch_does_not_pin_a_fingerprint(tmp_path):
    clock = FakeClock(OFFPEAK)
    fingerprint_path = tmp_path / "fingerprint.json"
    partial = tmp_path / "hidden.partial.jsonl"

    bad = FakeClient(
        [response(model="deepseek-chat", fingerprint="fp-bad")], clock=clock
    )
    first = _writer(
        bad, clock, partial_path=partial, fingerprint_path=fingerprint_path
    ).write([_problem("p:1")])
    assert first.stopped_reason == "model_mismatch"
    assert not fingerprint_path.exists()

    good = FakeClient([response(fingerprint="fp-good")], clock=clock)
    second = _writer(
        good, clock, partial_path=partial, fingerprint_path=fingerprint_path
    ).write([_problem("p:1"), _problem("p:2")])
    assert second.stopped_reason is None
    assert load_fingerprint_record(fingerprint_path)["fingerprint"] == "fp-good"


def test_send_with_deadline_enforces_a_total_deadline():
    import time

    from gemmaloop.deepseek_money import DeepSeekError, send_with_deadline

    with pytest.raises(DeepSeekError, match="deadline"):
        send_with_deadline(lambda: time.sleep(0.05), deadline_s=0.01)


def test_missing_usage_at_peak_settles_at_twice_the_reservation():
    clock = FakeClock(OFFPEAK)
    peak_response = response(usage=None)
    peak_response.created = datetime(2026, 11, 2, 2, 0, tzinfo=UTC).timestamp()
    accountant = CallAccountant(max_usd=5.0, now_fn=clock, sleep_fn=clock.sleep)
    reservation, estimate = accountant.before_request(
        prompt_tokens=1000, max_output_tokens=4096
    )
    with pytest.raises(PeakRequestDetected) as excinfo:
        accountant.after_response(reservation, peak_response)
    assert excinfo.value.actual["total_usd"] == pytest.approx(
        estimate["total_usd"] * 2
    )


def test_shared_accountant_refuses_to_replace_another_spend_lock(tmp_path):
    from gemmaloop.deepseek_money import shared_accountant
    from gemmaloop.stage0.spend import (
        SpendLockBusy,
        release_held_spend_lock,
        require_spend_lock,
    )

    require_spend_lock(tmp_path / "first.lock")
    try:
        with pytest.raises(SpendLockBusy):
            shared_accountant(max_usd=1.0, lock_path=tmp_path / "second.lock")
    finally:
        release_held_spend_lock()


def test_writer_default_max_usd_is_the_prepaid_balance():
    module = _load_cli()
    assert module._parse_args([]).max_usd == 5.0
    assert HiddenTestWriter(dry_run=True).max_usd == 5.0

def test_binding_check_allows_a_zero_request_interrupted_report():
    config = {"model": "deepseek-flash", "max_output_tokens": 4096, "limit": None}
    report = build_pilot_report(
        requests=0,
        usd=0.0,
        pilot_items=2,
        total_items=20,
        stopped_reason="error:RuntimeError",
        config=config,
        candidate_ids=["a", "b"],
        planned_candidate_ids=["a", "b"],
        accounted_candidate_ids=[],
    )
    assert (
        validate_pilot_report(
            report,
            expected_config_sha256=pilot_config_sha256(config),
            expected_candidate_ids=["a", "b"],
            require_complete=False,
        )
        == 0.0
    )
    with pytest.raises(PilotReportError):
        validate_pilot_report(report)

def test_validate_pilot_report_accepts_an_explicit_waste_ack():
    report = build_pilot_report(
        requests=20,
        usd=0.02,
        pilot_items=20,
        total_items=20,
        waste_count=3,
        waste_trip_ids=["trip-1"],
        acked_waste_trip_ids=[],
        waste_acknowledged=False,
    )
    with pytest.raises(PilotReportError):
        validate_pilot_report(report)
    assert validate_pilot_report(
        report, ack_waste_trip="trip-1"
    ) == pytest.approx(0.02)

# ---------------------------------------------------------------------------
# Round 3: pilot events, yield fields, writer repin and raw event content
# ---------------------------------------------------------------------------


def test_pilot_journal_summary_lists_events_and_yield_fields():
    from gemmaloop.deepseek_money import summarize_pilot_journal

    entries = [
        {
            "kind": "response",
            "problem_id": "p:1",
            "paid": True,
            "sent": True,
            "actual_usd": 0.01,
            "recorded": True,
            "dropped_tests": 2,
            "waste": False,
        },
        {
            "kind": "response",
            "problem_id": "p:2",
            "paid": True,
            "sent": True,
            "actual_usd": 0.01,
            "recorded": False,
            "error": "parse_error",
            "waste": False,
        },
        {
            "kind": "response",
            "problem_id": "p:3",
            "paid": True,
            "sent": True,
            "actual_usd": 0.01,
            "recorded": False,
            "error": "model_mismatch",
            "event_id": "evt-1",
            "response_model": "deepseek-v3",
            "waste": False,
        },
    ]
    summary = summarize_pilot_journal(entries, ["p:1", "p:2", "p:3"])
    assert summary["records_written"] == 1
    assert summary["parse_failures"] == 1
    assert summary["dropped_tests"] == 2
    assert [event["event_id"] for event in summary["pilot_events"]] == ["evt-1"]
    assert summary["unacked_pilot_event_ids"] == ["evt-1"]


def test_cli_non_flash_pilot_event_requires_ack(tmp_path, monkeypatch):
    module = _load_cli()
    clock = FakeClock(OFFPEAK)
    first = FakeClient(
        [response() for _ in range(19)] + [response(model="deepseek-v3")],
        clock=clock,
    )
    _install_cli_fakes(module, monkeypatch, first, clock)
    pool = _write_pool(tmp_path / "pool", 30)
    output = tmp_path / "hidden.jsonl"

    assert module.main(_base_args(tmp_path, pool, output, ["--pilot", "20"])) == 0
    report_path = output.with_suffix(output.suffix + ".pilot.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stopped_reason"] == "pilot_events"
    assert report["pilot_events"]
    assert report["pilot_events"][0]["kind"] == "model_mismatch"
    event_id = report["pilot_events"][0]["event_id"]
    assert event_id in report["unacked_pilot_event_ids"]

    without_ack = FakeClient([response() for _ in range(20)], clock=clock)
    _install_cli_fakes(module, monkeypatch, without_ack, clock)
    with pytest.raises(PilotReportError):
        module.main(
            _base_args(tmp_path, pool, output, ["--after-pilot", str(report_path)])
        )
    assert without_ack.calls == []

    with_ack = FakeClient([response() for _ in range(20)], clock=clock)
    _install_cli_fakes(module, monkeypatch, with_ack, clock)
    assert (
        module.main(
            _base_args(
                tmp_path,
                pool,
                output,
                ["--after-pilot", str(report_path), "--ack-pilot-event", event_id],
            )
        )
        == 0
    )
    assert len(with_ack.calls) == 10

    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert event_id in updated["acked_pilot_event_ids"]
    assert updated["unacked_pilot_event_ids"] == []
    assert updated["events_acknowledged"] is True
    assert updated["stopped_reason"] is None

    repeat = FakeClient([response() for _ in range(5)], clock=clock)
    _install_cli_fakes(module, monkeypatch, repeat, clock)
    assert (
        module.main(
            _base_args(tmp_path, pool, output, ["--after-pilot", str(report_path)])
        )
        == 0
    )
    assert repeat.calls == []


def test_writer_repin_replaces_the_registry_with_history(tmp_path):
    clock = FakeClock(OFFPEAK)
    fingerprint_path = tmp_path / "fingerprint.json"
    write_fingerprint_record(
        model="deepseek-flash",
        fingerprint="fp-old",
        source="test",
        path=fingerprint_path,
    )
    client = FakeClient([response(fingerprint="fp-new")], clock=clock)
    result = _writer(
        client,
        clock,
        fingerprint_path=fingerprint_path,
        repin=True,
    ).write([_problem("p:1")])
    assert result.stopped_reason is None
    record = load_fingerprint_record(fingerprint_path)
    assert record["fingerprint"] == "fp-new"
    assert record["history"][0]["fingerprint"] == "fp-old"


def test_writer_event_journal_keeps_the_raw_content(tmp_path):
    from gemmaloop.deepseek_money import read_journal

    clock = FakeClock(OFFPEAK)
    partial = tmp_path / "hidden.partial.jsonl"
    content = json.dumps({"tests": [{"input": "1", "output": "1"}]})
    client = FakeClient([response(model="deepseek-v3", content=content)], clock=clock)
    result = _writer(client, clock, partial_path=partial).write([_problem("p:1")])
    assert result.stopped_reason == "model_mismatch"
    journal = read_journal(partial.with_name(partial.name + ".responses.jsonl"))
    entry = next(item for item in journal if item.get("kind") == "response")
    assert entry["error"] == "model_mismatch"
    assert entry["event_id"]
    assert entry["raw_content"] == content


def test_writer_missing_fingerprint_is_a_recorded_event(tmp_path):
    from gemmaloop.deepseek_money import read_journal

    clock = FakeClock(OFFPEAK)
    partial = tmp_path / "hidden.partial.jsonl"
    client = FakeClient([response(fingerprint=None)], clock=clock)
    result = _writer(client, clock, partial_path=partial).write([_problem("p:1")])
    assert result.stopped_reason == "fingerprint_mismatch"
    journal = read_journal(partial.with_name(partial.name + ".responses.jsonl"))
    entry = next(item for item in journal if item.get("kind") == "response")
    assert entry["error"] == "fingerprint_mismatch"
    assert entry["event_id"]
    assert entry["raw_content"] is not None


def test_writer_cli_exposes_repin_and_pilot_event_ack():
    module = _load_cli()
    args = module._parse_args([])
    assert args.repin is False
    assert args.ack_pilot_event is None
    assert module._parse_args(["--repin"]).repin is True
    assert module._parse_args(
        ["--ack-pilot-event", "evt-1", "--ack-pilot-event", "evt-2"]
    ).ack_pilot_event == ["evt-1", "evt-2"]

# ---------------------------------------------------------------------------
# Round 4: repin scope, generic waste ack, report ack write-back, history cap
# ---------------------------------------------------------------------------


def test_writer_repin_applies_only_to_the_first_paid_response(tmp_path):
    clock = FakeClock(OFFPEAK)
    fingerprint_path = tmp_path / "fingerprint.json"
    write_fingerprint_record(
        model="deepseek-flash",
        fingerprint="fpA",
        source="test",
        path=fingerprint_path,
    )
    client = FakeClient(
        [
            response(fingerprint="fpA"),
            response(fingerprint="fpB"),
            response(fingerprint="fpA"),
            response(fingerprint="fpC"),
        ],
        clock=clock,
    )
    result = _writer(
        client,
        clock,
        fingerprint_path=fingerprint_path,
        repin=True,
    ).write([_problem(f"p:{index}") for index in range(4)])
    assert len(client.calls) == 2  # first response repins, second mismatches
    assert result.stopped_reason == "fingerprint_mismatch"
    record = load_fingerprint_record(fingerprint_path)
    assert record["fingerprint"] == "fpA"
    assert len(record["history"]) == 1  # exactly one repin

    later = FakeClient([response(fingerprint="fpC")], clock=clock)
    later_result = _writer(
        later, clock, fingerprint_path=fingerprint_path
    ).write([_problem("p:9")])
    assert later_result.stopped_reason == "fingerprint_mismatch"


def test_pilot_event_ack_of_a_waste_trip_restarts_the_window(tmp_path):
    from gemmaloop.deepseek_money import read_journal

    clock = FakeClock(OFFPEAK)
    partial = tmp_path / "hidden.partial.jsonl"
    client = FakeClient(
        [response(content="", finish_reason="length") for _ in range(3)],
        clock=clock,
    )
    writer = _writer(client, clock, partial_path=partial)
    result = writer.write([_problem(f"p:{index}") for index in range(5)])
    assert result.stopped_reason == "waste_guard"
    trip_id = result.waste_trip_id

    second = FakeClient([response(), response()], clock=clock)
    resumed = _writer(
        second, clock, partial_path=partial, ack_pilot_events=[trip_id]
    ).write([_problem(f"p:{index}") for index in range(5)])
    assert resumed.stopped_reason is None
    assert len(second.calls) == 2  # p:3 and p:4, no re-buys
    journal = read_journal(partial.with_name(partial.name + ".responses.jsonl"))
    assert sum(1 for entry in journal if entry.get("kind") == "waste_trip") == 1


def test_fingerprint_history_keeps_the_first_plus_the_last_twenty(tmp_path):
    fingerprint_path = tmp_path / "fingerprint.json"
    write_fingerprint_record(
        model="deepseek-flash",
        fingerprint="fp-0",
        source="test",
        path=fingerprint_path,
    )
    for index in range(1, 31):
        write_fingerprint_record(
            model="deepseek-flash",
            fingerprint=f"fp-{index}",
            source="test",
            path=fingerprint_path,
            repin=True,
        )
    record = load_fingerprint_record(fingerprint_path)
    assert record["fingerprint"] == "fp-30"
    assert len(record["history"]) == 21
    assert record["history"][0]["fingerprint"] == "fp-0"
    assert record["history"][-1]["fingerprint"] == "fp-29"
