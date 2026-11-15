"""DeepSeek hidden-test writer: fake client only, never the real API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gemmaloop.stage0.deepseek import (
    EndpointRefused,
    HiddenTestWriter,
    MissingAPIKey,
    SpendingCapExceeded,
    api_key_from_env,
    estimate_cost,
    is_peak,
    seconds_until_off_peak,
    validate_endpoint,
    wait_for_off_peak,
)


class FakeCompletions:
    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            model="deepseek-flash",
            system_fingerprint="fp-test",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            ),
        )


def _fake_client(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    completions = FakeCompletions(content, prompt_tokens, completion_tokens)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _problem(problem_id: str = "taco:abc") -> dict:
    return {
        "problem_id": problem_id,
        "source": "taco",
        "prompt": "Add two numbers from stdin.",
        "tests": [],
        "metadata": {"reference_solution": "a, b = map(int, input().split())\nprint(a + b)"},
    }


class FakeGrader:
    def __init__(self, failing_inputs: set[str] | None = None):
        self.failing_inputs = failing_inputs or set()

    def grade_code(self, code, tests):
        test = tests[0]
        return SimpleNamespace(passed=test["input"] not in self.failing_inputs)


def test_endpoint_validation_refuses_opencode_and_non_official():
    assert validate_endpoint("https://api.deepseek.com/") == "https://api.deepseek.com"
    for url in (
        "https://opencode.ai/v1",
        "https://api.opencode.ai/v1",
        "http://api.deepseek.com",
        "https://example.com/v1",
    ):
        with pytest.raises(EndpointRefused):
            validate_endpoint(url)


def test_api_key_is_read_only_from_environment(monkeypatch):
    assert api_key_from_env({"DEEPSEEK_API_KEY": "sk-test-secret"}) == "sk-test-secret"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey) as excinfo:
        api_key_from_env({})
    assert "DEEPSEEK_API_KEY" in str(excinfo.value)
    assert "sk-" not in str(excinfo.value)


def test_cost_estimate_and_peak_windows():
    off = estimate_cost(1_000_000, 1_000_000, n_requests=1, peak=False)
    assert off["total_usd"] == pytest.approx(0.75)
    peak = estimate_cost(1_000_000, 1_000_000, n_requests=1, peak=True)
    assert peak["total_usd"] == pytest.approx(1.50)

    monday_2 = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)
    monday_5 = datetime(2026, 11, 2, 5, 0, tzinfo=timezone.utc)
    saturday_2 = datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)
    assert is_peak(monday_2)
    assert not is_peak(monday_5)
    assert not is_peak(saturday_2)
    # 2 h to the window end + the 5-minute post-peak margin.
    assert seconds_until_off_peak(monday_2) == pytest.approx(2 * 3600 + 5 * 60)
    assert seconds_until_off_peak(monday_5) == 0

    now = monday_2
    slept: list[float] = []

    def clock():
        return now

    def sleep(seconds):
        nonlocal now
        slept.append(seconds)
        now = now + timedelta(seconds=seconds)

    wait_for_off_peak(now_fn=clock, sleep_fn=sleep)
    assert slept == [pytest.approx(2 * 3600 + 5 * 60)]


def test_dry_run_does_not_call_client_and_estimates_cost():
    def explode(**kwargs):  # pragma: no cover
        raise AssertionError("dry-run must not call the API")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=explode)))
    writer = HiddenTestWriter(client=client, dry_run=True, max_usd=5.0)
    result = writer.write([_problem()])
    assert result.calls == 0
    assert result.records == []
    assert result.estimated["total_usd"] > 0
    assert result.estimated["input_tokens"] > 0
    assert result.estimated["output_tokens"] == writer.max_output_tokens


def test_cap_refuses_before_any_request():
    client, completions = _fake_client('{"tests": []}')
    writer = HiddenTestWriter(client=client, max_usd=0.0001, dry_run=False)
    with pytest.raises(SpendingCapExceeded):
        writer.write([_problem()])
    assert completions.calls == []


def test_writer_drops_tests_the_reference_fails_and_shares_output():
    content = json.dumps(
        {
            "tests": [
                {"input": "1 2", "output": "3"},
                {"input": "bad", "output": "0"},
                {"input": "4 5", "output": "9"},
            ]
        }
    )
    client, completions = _fake_client(content)
    writer = HiddenTestWriter(
        client=client,
        max_usd=5.0,
        dry_run=False,
        grader=FakeGrader(failing_inputs={"bad"}),
    )
    result = writer.write([_problem()])
    assert result.calls == 1
    assert result.dropped_tests == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert record["shared_by_arms"] is True
    assert "arm" not in record
    assert [test["input"] for test in record["tests"]] == ["1 2", "4 5"]
    assert completions.calls[0]["model"] == "deepseek-flash"


def test_writer_requires_env_key_for_real_client(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    writer = HiddenTestWriter(max_usd=5.0, dry_run=False)
    with pytest.raises(MissingAPIKey):
        writer._get_client()


def test_writer_refuses_opencode_even_with_fake_client():
    client, _ = _fake_client('{"tests": []}')
    with pytest.raises(EndpointRefused):
        HiddenTestWriter(client=client, base_url="https://opencode.ai/v1")


def test_mid_run_cap_stop_keeps_paid_output():
    content = json.dumps({"tests": [{"input": "1 2", "output": "3"}]})
    client, completions = _fake_client(
        content, prompt_tokens=100, completion_tokens=1_000_000
    )
    writer = HiddenTestWriter(
        client=client,
        max_usd=0.001,
        max_output_tokens=10,
        dry_run=False,
        grader=FakeGrader(),
    )
    result = writer.write([_problem("p:1"), _problem("p:2")])
    assert result.stopped_reason == "spending_cap"
    assert result.calls == 1
    assert len(result.records) == 1
    assert result.records[0]["problem_id"] == "p:1"


def test_cli_is_dry_run_by_default_and_requires_live_for_requests():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "write_stage0_hidden_tests.py"
    spec = importlib.util.spec_from_file_location("write_stage0_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    assert module._parse_args([]).live is False
    assert module._parse_args(["--live"]).live is True
    assert module._parse_args(["--dry-run"]).dry_run is True


def test_non_cap_abort_keeps_paid_records():
    calls = {"n": 0}

    class FlakyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("network reset")
            message = SimpleNamespace(content='{"tests": [{"input": "1", "output": "1"}]}')
            return SimpleNamespace(
                model="deepseek-flash",
                system_fingerprint="fp-test",
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
            )

    writer = HiddenTestWriter(
        client=FlakyClient(), max_usd=5.0, dry_run=False, grader=FakeGrader()
    )
    result = writer.write([_problem("p:1"), _problem("p:2")])
    assert result.stopped_reason == "error:RuntimeError"
    assert result.calls == 1
    assert [record["problem_id"] for record in result.records] == ["p:1"]


def _write_pool(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "problems.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "code:1",
                "source": "taco",
                "kind": "code",
                "prompt": "add",
                "tests": [],
                "metadata": {"reference_solution": "print(1)"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pool


def test_cli_main_default_builds_no_client_or_grader(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / "scripts" / "write_stage0_hidden_tests.py"
    spec = importlib.util.spec_from_file_location("write_stage0_cli_main_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    captured = {}

    class SpyWriter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def write(self, problems):
            return SimpleNamespace(
                estimated={"total_usd": 0.0, "input_tokens": 1, "output_tokens": 1},
                records=[],
                dropped_tests=0,
                failed_problems=0,
                calls=0,
                spent_usd=0.0,
                stopped_reason=None,
            )

    class BombGrader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("default dry-run must not create a grader")

    monkeypatch.setattr(module, "HiddenTestWriter", SpyWriter)
    monkeypatch.setattr(module, "GraderService", BombGrader)
    pool = _write_pool(tmp_path)
    assert module.main(["--pool-dir", str(pool), "--output", str(tmp_path / "out.jsonl")]) == 0
    assert captured["dry_run"] is True


def test_cli_main_live_without_key_refuses(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "write_stage0_hidden_tests.py"
    spec = importlib.util.spec_from_file_location("write_stage0_cli_live_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class FakeGrader:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "GraderService", FakeGrader)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    pool = _write_pool(tmp_path)
    with pytest.raises(MissingAPIKey):
        module.main(
            [
                "--live",
                "--pool-dir",
                str(pool),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--spend-lock",
                str(tmp_path / "spend.lock"),
                "--ledger",
                str(tmp_path / "spend.json"),
                "--init-ledger",
            ]
        )


def test_deepseek_spend_ledger_reserves_and_persists(tmp_path):
    from gemmaloop.stage0.spend import SpendLedger

    content = json.dumps({"tests": [{"input": "1", "output": "1"}]})
    client, completions = _fake_client(content, prompt_tokens=10, completion_tokens=10)
    ledger = SpendLedger(tmp_path / "spend.json")
    writer = HiddenTestWriter(
        client=client,
        max_usd=5.0,
        dry_run=False,
        grader=FakeGrader(),
        ledger=ledger,
        ledger_month="2026-11",
    )
    result = writer.write([_problem("p:1")])
    assert result.calls == 1
    assert ledger.cumulative_usd("2026-11", provider="deepseek") > 0
    entries = ledger.entries()
    assert any(entry["kind"] == "reservation" and entry["provider"] == "deepseek" for entry in entries)
    assert any(entry["kind"] == "actual" and entry["provider"] == "deepseek" for entry in entries)


def test_partial_output_is_appended_and_resumed(tmp_path):
    content = json.dumps({"tests": [{"input": "1", "output": "1"}]})
    client, completions = _fake_client(content)

    calls = {"n": 0}

    class Flaky:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash")
            message = SimpleNamespace(content=content)
            return SimpleNamespace(
                model="deepseek-flash",
                system_fingerprint="fp-test",
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
            )

    partial = tmp_path / "hidden.partial.jsonl"
    writer = HiddenTestWriter(
        client=Flaky(), max_usd=5.0, dry_run=False, grader=FakeGrader(),
        partial_path=partial,
    )
    result = writer.write([_problem("p:1"), _problem("p:2")])
    assert result.stopped_reason == "error:RuntimeError"
    assert [json.loads(line)["problem_id"] for line in partial.read_text().splitlines()] == ["p:1"]

    second_client, second_completions = _fake_client(content)
    resumed = HiddenTestWriter(
        client=second_client, max_usd=5.0, dry_run=False, grader=FakeGrader(),
        partial_path=partial,
    )
    resumed_result = resumed.write([_problem("p:1"), _problem("p:2")])
    assert resumed_result.resumed == 1
    assert second_completions.calls and second_completions.calls[0]["messages"]
    assert [record["problem_id"] for record in resumed_result.records] == ["p:2"]


def test_cli_live_requires_the_spend_lock(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "write_stage0_hidden_tests.py"
    spec = importlib.util.spec_from_file_location("write_stage0_lock_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "require_spend_lock", lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(75)))
    pool = _write_pool(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--live", "--pool-dir", str(pool), "--output", str(tmp_path / "out.jsonl"), "--ledger", str(tmp_path / "spend.json"), "--init-ledger", "--spend-lock", str(tmp_path / "spend.lock")])
    assert excinfo.value.code == 75


def test_deepseek_guard_starts_from_monthly_ledger_total(tmp_path):
    from gemmaloop.stage0.spend import SpendLedger

    ledger = SpendLedger(tmp_path / "spend.json")
    reservation = ledger.reserve(shard=0, usd=5.0, month="2026-11", provider="deepseek")
    ledger.settle(reservation, shard=0, actual_usd=4.999, month="2026-11", provider="deepseek")
    client, completions = _fake_client('{"tests": [{"input": "1", "output": "1"}]}')
    writer = HiddenTestWriter(
        client=client, max_usd=4.999, max_output_tokens=100, dry_run=False,
        grader=FakeGrader(), ledger=ledger, ledger_month="2026-11",
    )
    result = writer.write([_problem("p:1")])
    assert result.calls == 0
    assert result.stopped_reason == "spending_cap"
    assert completions.calls == []
