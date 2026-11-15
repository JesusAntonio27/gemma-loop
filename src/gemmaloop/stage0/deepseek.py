"""Hidden-test writer for the official DeepSeek Open Platform API.

Rules implemented here are the registered ones:

* the endpoint is the **official** API (``https://api.deepseek.com``); any
  ``opencode.ai`` / OpenCode endpoint is refused before a request is built,
* the key is read from the ``DEEPSEEK_API_KEY`` environment variable only --
  never from a file in the repo and never logged,
* the only accepted request model id is ``deepseek-flash``; responses must be
  exactly ``DeepSeek-V4.1-Flash-1017`` or the documented alias, and the single
  registered ``system_fingerprint`` record must match,
* every request goes through :mod:`gemmaloop.deepseek_money`: the mandatory
  off-peak gate, explicit thinking (disabled here), a shared monthly ledger
  reservation and the sliding waste guard,
* dry-run mode only counts tokens and estimates cost with the registered
  off-peak prices (input ``$0.15``, output ``$0.60`` per million tokens),
* the default cap is the author's $5 prepaid balance (global ceiling $30),
* every paid response is journaled before a stop, so a resume never pays for
  the same item twice; an interrupted pilot can be resumed on the same output
  path and its report aggregates all runs,
* tests that the reference solution fails are dropped.

The output records only ``problem_id`` and ``tests`` plus provenance, so the
same file is used by both arms.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from gemmaloop import deepseek_money as _money
from gemmaloop.deepseek_money import (
    API_KEY_ENV,
    API_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    HARD_GLOBAL_CAP_USD,
    OFFICIAL_BASE_URL,
    PEAK_MULTIPLIER,
    PRICES_PER_MILLION,
    REGISTERED_MODEL,
    CallAccountant,
    DeepSeekError,
    EndpointRefused,
    FingerprintMismatch,
    MissingAPIKey,
    ModelMismatch,
    PeakRequestDetected,
    SpendingCapExceeded,
    WasteGuard,
    WasteTripAcknowledgmentRequired,
    api_key_from_env,
    append_journal,
    enforce_flash_response,
    ensure_fingerprint,
    estimate_cost,
    estimate_tokens,
    is_peak,
    read_journal,
    response_content,
    response_finish_reason,
    response_journal_path,
    seconds_until_off_peak,
    send_with_deadline,
    shared_accountant,
    usage_breakdown,
    validate_endpoint,
    validate_model,
    wait_for_off_peak,
)

PEAK_HOURS_UTC = ((1, 4), (6, 10))


def _json_from_text(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _parse_tests(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("tests") or payload.get("hidden_tests") or []
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    tests: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        test_input = item.get("input", item.get("stdin", ""))
        test_output = item.get("output", item.get("stdout", item.get("expected", "")))
        if test_input is None or test_output is None:
            continue
        tests.append(
            {
                "input": str(test_input),
                "output": str(test_output),
                "type": str(item.get("type") or "stdin"),
            }
        )
    return tests


def _response_usage(response: Any) -> tuple[int, int]:
    usage = usage_breakdown(response)
    return usage["prompt_tokens"], usage["completion_tokens"]


def _response_field(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _build_prompt(problem: dict[str, Any]) -> str:
    metadata = problem.get("metadata") or {}
    reference = metadata.get("reference_solution") or ""
    existing = problem.get("tests") or []
    return (
        "You are writing hidden tests for a competitive-programming problem.\n"
        "Return only JSON with this exact shape:\n"
        '{"tests": [{"input": "<stdin>", "output": "<stdout>"}, ...]}\n'
        "Use stdin/stdout tests. Include edge cases, but do not include an "
        "explanation. Every output must be exactly what the reference program "
        "prints.\n\n"
        f"Problem statement:\n{problem.get('prompt', '')}\n\n"
        f"Reference solution:\n{reference}\n\n"
        f"Existing tests ({len(existing)}):\n{json.dumps(existing[:5], ensure_ascii=False)}\n"
    )


@dataclass
class WriteResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    dropped_tests: int = 0
    failed_problems: int = 0
    estimated: dict[str, Any] = field(default_factory=dict)
    spent_usd: float = 0.0
    run_usd: float = 0.0
    calls: int = 0
    stopped_reason: str | None = None
    resumed: int = 0
    waste: int = 0
    waste_trip_id: str | None = None
    input_miss_tokens: int = 0
    input_hit_tokens: int = 0
    output_tokens: int = 0
    response_models: list[Any] = field(default_factory=list)
    system_fingerprints: list[Any] = field(default_factory=list)


class HiddenTestWriter:
    """Write hidden tests with the official DeepSeek API, or plan a dry run."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        base_url: str = OFFICIAL_BASE_URL,
        model: str = API_MODEL,
        max_usd: float = 5.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        dry_run: bool = True,
        prices: dict[str, float] | None = None,
        token_counter: Callable[[str], int] | None = None,
        grader: Any | None = None,
        ledger: Any | None = None,
        ledger_month: str | None = None,
        partial_path: Path | None = None,
        resume: bool = True,
        sleep_fn: Callable[[float], None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        waste_guard: WasteGuard | None = None,
        ack_waste_trip: str | None = None,
        ack_pilot_events: list[str] | None = None,
        repin: bool = False,
        fingerprint_path: Path | str | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        self.base_url = validate_endpoint(base_url)
        self.model = validate_model(model)
        if not 0 < float(max_usd) <= HARD_GLOBAL_CAP_USD:
            raise ValueError(
                f"max_usd must be in (0, {HARD_GLOBAL_CAP_USD:g}]; got {max_usd!r}"
            )
        self.max_usd = float(max_usd)
        self.max_output_tokens = int(max_output_tokens)
        self.dry_run = bool(dry_run)
        self.prices = dict(prices or PRICES_PER_MILLION)
        self.token_counter = token_counter
        self.grader = grader
        self.partial_path = Path(partial_path) if partial_path is not None else None
        self.journal_path = (
            response_journal_path(self.partial_path)
            if self.partial_path is not None
            else None
        )
        self.resume = bool(resume)
        self.ack_waste_trip = ack_waste_trip
        self.ack_pilot_events = list(ack_pilot_events or [])
        self.repin = bool(repin)
        self._repin_pending = bool(repin)
        self.fingerprint_path = (
            Path(fingerprint_path) if fingerprint_path is not None else None
        )
        self.request_timeout_s = float(
            request_timeout_s
            if request_timeout_s is not None
            else _money.REQUEST_TIMEOUT_SECONDS
        )
        self._client = client
        self._sleep = sleep_fn or time.sleep
        self._now = now_fn or (lambda: _money.utcnow())
        self.waste_guard = waste_guard or WasteGuard()

        if ledger is None and not self.dry_run:
            accountant, ledger, month = shared_accountant(
                max_usd=self.max_usd,
                now_fn=self._now,
                sleep_fn=self._sleep,
                prices=self.prices,
            )
            self.ledger = ledger
            self.ledger_month = month
            self.accountant = accountant
        else:
            self.ledger = ledger
            self.ledger_month = ledger_month
            self.accountant = CallAccountant(
                ledger=ledger,
                month=ledger_month,
                max_usd=self.max_usd,
                now_fn=self._now,
                sleep_fn=self._sleep,
                prices=self.prices,
            )

    def _get_client(self):
        if self._client is not None:
            return self._client
        # The key is fetched only when a real request is about to be made.
        secret = api_key_from_env()
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - openai is a dependency
            raise DeepSeekError("the openai client is required for real requests") from exc
        self._client = OpenAI(
            api_key=secret,
            base_url=self.base_url,
            max_retries=_money.CLIENT_MAX_RETRIES,
            timeout=_money.REQUEST_TIMEOUT_SECONDS,
        )
        return self._client

    def _prompt_tokens(self, problem: dict[str, Any]) -> int:
        return estimate_tokens(_build_prompt(problem), self.token_counter)

    def plan(self, problems: Iterable[dict[str, Any]]) -> dict[str, Any]:
        problems = list(problems)
        prompt_tokens = sum(self._prompt_tokens(problem) for problem in problems)
        # Peak is impossible behind the mandatory gate, so the plan is
        # off-peak at the cache-miss input price.
        estimate = estimate_cost(
            prompt_tokens,
            self.max_output_tokens,
            n_requests=len(problems),
            peak=False,
            prices=self.prices,
        )
        estimate["n_problems"] = len(problems)
        estimate["model"] = self.model
        estimate["registered_model"] = REGISTERED_MODEL
        estimate["max_usd"] = self.max_usd
        estimate["within_cap"] = estimate["total_usd"] <= self.max_usd
        return estimate

    def _check_cap(self, estimate: dict[str, Any]) -> None:
        if estimate["total_usd"] > self.max_usd:
            raise SpendingCapExceeded(
                f"estimated ${estimate['total_usd']:.4f} exceeds the hard cap "
                f"${self.max_usd:.4f}; refusing to start"
            )

    def _request(self, problem: dict[str, Any]) -> Any:
        client = self._get_client()
        prompt = _build_prompt(problem)
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": 0.0,
            # Never rely on the provider default (thinking on, effort high,
            # 64K output): hidden tests are plain JSON and must be cheap.
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if hasattr(getattr(client, "chat", None), "completions"):
            return send_with_deadline(
                lambda: client.chat.completions.create(**kwargs),
                deadline_s=self.request_timeout_s,
            )
        if callable(client):
            return send_with_deadline(
                lambda: client(**kwargs), deadline_s=self.request_timeout_s
            )
        raise DeepSeekError("client has no chat.completions.create")  # pragma: no cover

    def _drop_failing_tests(
        self, problem: dict[str, Any], tests: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        reference = (problem.get("metadata") or {}).get("reference_solution") or ""
        if not reference:
            return [], len(tests)
        kept: list[dict[str, Any]] = []
        dropped = 0
        grader = self.grader
        if grader is None:
            from .grader import grade_code

            def grade_one(test: dict[str, Any]):
                return grade_code(reference, [test])
        else:
            def grade_one(test: dict[str, Any]):
                return grader.grade_code(reference, [test])

        for test in tests:
            result = grade_one(test)
            if getattr(result, "passed", False):
                kept.append(test)
            else:
                dropped += 1
        return kept, dropped

    def _read_partial(self) -> list[dict[str, Any]]:
        if self.partial_path is None or not self.partial_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.partial_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _read_journal(self) -> list[dict[str, Any]]:
        if self.journal_path is None:
            return []
        return read_journal(self.journal_path)

    def _append_partial(self, record: dict[str, Any]) -> None:
        if self.partial_path is None:
            return
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        with self.partial_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _append_journal(self, entry: dict[str, Any]) -> None:
        if self.journal_path is None:
            return
        append_journal(self.journal_path, entry)

    def _waste_flags_after_last_ack(
        self, journal: list[dict[str, Any]]
    ) -> list[bool]:
        trip_ids = {
            str(entry.get("trip_id"))
            for entry in journal
            if entry.get("kind") == "waste_trip" and entry.get("trip_id")
        }
        last_ack = -1
        for index, entry in enumerate(journal):
            if entry.get("kind") == "waste_ack":
                last_ack = index
            elif (
                entry.get("kind") == "pilot_event_ack"
                and str(entry.get("event_id")) in trip_ids
            ):
                last_ack = index
        recent = journal[last_ack + 1 :]
        return [
            bool(entry.get("waste"))
            for entry in recent
            if entry.get("kind") == "response" and entry.get("paid", True)
        ]

    def _new_trip(self, result: WriteResult, journal: list[dict[str, Any]], problem_id: Any) -> str:
        trip_id = uuid.uuid4().hex[:12]
        self._append_journal(
            {
                "kind": "waste_trip",
                "trip_id": trip_id,
                "problem_id": problem_id,
                "waste_count": self.waste_guard.waste_count,
                "total_count": self.waste_guard.total_count,
            }
        )
        journal.append(
            {
                "kind": "waste_trip",
                "trip_id": trip_id,
                "problem_id": problem_id,
                "waste_count": self.waste_guard.waste_count,
                "total_count": self.waste_guard.total_count,
            }
        )
        result.waste_trip_id = trip_id
        result.waste = self.waste_guard.waste_count
        result.stopped_reason = "waste_guard"
        return trip_id

    def _prepare_waste_window(
        self, result: WriteResult, journal: list[dict[str, Any]]
    ) -> bool:
        trips = [
            str(entry.get("trip_id"))
            for entry in journal
            if entry.get("kind") == "waste_trip" and entry.get("trip_id")
        ]
        acks = {
            str(entry.get("trip_id"))
            for entry in journal
            if entry.get("kind") in ("waste_ack", "pilot_event_ack")
            and entry.get("trip_id")
        }
        acks.update(
            str(entry.get("event_id"))
            for entry in journal
            if entry.get("kind") == "pilot_event_ack" and entry.get("event_id")
        )
        unacked = [trip for trip in trips if trip not in acks]
        if unacked:
            if self.ack_waste_trip not in unacked:
                raise WasteTripAcknowledgmentRequired(
                    "an earlier run tripped the waste guard (trip "
                    f"{unacked[-1]}); pass --ack-waste-trip {unacked[-1]} to "
                    "start a new window deliberately"
                )
            ack = {
                "kind": "waste_ack",
                "trip_id": self.ack_waste_trip,
                "timestamp": _money.utcnow().isoformat(),
            }
            self._append_journal(ack)
            journal.append(ack)
            still_unacked = [trip for trip in unacked if trip != self.ack_waste_trip]
            if still_unacked:
                raise WasteTripAcknowledgmentRequired(
                    "more than one waste trip is unacknowledged; pass "
                    f"--ack-waste-trip {still_unacked[-1]} as well"
                )
            self.waste_guard = WasteGuard(
                window=self.waste_guard.window, rate=self.waste_guard.rate
            )
            return True

        flags = self._waste_flags_after_last_ack(journal)
        self.waste_guard = WasteGuard(
            window=self.waste_guard.window, rate=self.waste_guard.rate
        )
        self.waste_guard.prime(flags[-self.waste_guard.window :])
        if self.waste_guard.triggered:
            # A legacy/manual journal can already be over the trip count.
            self._new_trip(result, journal, None)
            return False
        return True

    def write(self, problems: Iterable[dict[str, Any]]) -> WriteResult:
        problems = list(problems)
        estimate = self.plan(problems)
        result = WriteResult(estimated=estimate)
        if self.dry_run:
            return result
        self._check_cap(estimate)
        self._repin_pending = self.repin

        journal = self._read_journal()
        partial = self._read_partial()
        finished = {row.get("problem_id") for row in partial}
        finished.update(
            entry.get("problem_id")
            for entry in journal
            if entry.get("kind") == "response" and entry.get("paid", True)
        )
        if self.resume and finished:
            problems = [p for p in problems if p.get("problem_id") not in finished]
            result.resumed = len(finished)

        for event_id in self.ack_pilot_events:
            ack = {
                "kind": "pilot_event_ack",
                "event_id": str(event_id),
                "timestamp": _money.utcnow().isoformat(),
            }
            self._append_journal(ack)
            journal.append(ack)

        if not self._prepare_waste_window(result, journal):
            result.spent_usd = self.accountant.cumulative_usd()
            return result

        for problem in problems:
            shard = int(problem.get("_index", 0))
            metadata = {"problem_id": problem.get("problem_id")}
            try:
                reservation, _reserved = self.accountant.before_request(
                    prompt_tokens=self._prompt_tokens(problem),
                    max_output_tokens=self.max_output_tokens,
                    shard=shard,
                    metadata=metadata,
                )
            except SpendingCapExceeded:
                result.stopped_reason = "spending_cap"
                break

            try:
                response = self._request(problem)
            except (MissingAPIKey, EndpointRefused):
                settled = self.accountant.settle_failure(
                    reservation, shard=shard, metadata=metadata
                )
                result.run_usd += settled
                self._append_journal(
                    {
                        "kind": "failure",
                        "problem_id": problem.get("problem_id"),
                        "sent": False,
                        "paid": False,
                        "actual_usd": settled,
                        "waste": False,
                        "error": "missing_key_or_endpoint",
                    }
                )
                raise
            except Exception as exc:
                # The provider may have billed a failed request; keep the
                # reservation as spent and count it in the pilot projection.
                settled = self.accountant.settle_failure(
                    reservation, shard=shard, metadata=metadata
                )
                result.run_usd += settled
                self._append_journal(
                    {
                        "kind": "failure",
                        "problem_id": problem.get("problem_id"),
                        "sent": True,
                        "paid": False,
                        "actual_usd": settled,
                        "waste": False,
                        "error": f"error:{type(exc).__name__}",
                    }
                )
                result.stopped_reason = f"error:{type(exc).__name__}"
                break

            result.calls += 1
            try:
                actual, usage = self.accountant.after_response(
                    reservation, response, shard=shard, metadata=metadata
                )
            except PeakRequestDetected as exc:
                peak_actual = exc.actual or {"total_usd": 0.0}
                result.run_usd += float(peak_actual.get("total_usd", 0.0))
                self._append_journal(
                    {
                        "kind": "response",
                        "problem_id": problem.get("problem_id"),
                        "sent": True,
                        "paid": True,
                        "actual_usd": float(peak_actual.get("total_usd", 0.0)),
                        "input_miss_tokens": int(
                            peak_actual.get("prompt_cache_miss_tokens", 0)
                        ),
                        "input_hit_tokens": int(
                            peak_actual.get("prompt_cache_hit_tokens", 0)
                        ),
                        "output_tokens": int(
                            peak_actual.get("completion_tokens", 0)
                        ),
                        "waste": False,
                        "finish_reason": response_finish_reason(response),
                        "response_model": _response_field(response, "model"),
                        "system_fingerprint": _response_field(
                            response, "system_fingerprint"
                        ),
                        "recorded": False,
                        "error": "peak_request",
                        "event_id": uuid.uuid4().hex[:12],
                        "raw_content": response_content(response),
                    }
                )
                result.stopped_reason = "peak_request"
                break
            result.spent_usd = self.accountant.cumulative_usd()
            result.run_usd += actual["total_usd"]
            result.input_miss_tokens += usage["prompt_cache_miss_tokens"]
            result.input_hit_tokens += usage["prompt_cache_hit_tokens"]
            result.output_tokens += usage["completion_tokens"]

            model = _response_field(response, "model")
            fingerprint = _response_field(response, "system_fingerprint")
            result.response_models.append(model)
            result.system_fingerprints.append(fingerprint)
            content = response_content(response)
            finish_reason = response_finish_reason(response)
            entry: dict[str, Any] = {
                "kind": "response",
                "problem_id": problem.get("problem_id"),
                "sent": True,
                "paid": True,
                "actual_usd": actual["total_usd"],
                "input_miss_tokens": usage["prompt_cache_miss_tokens"],
                "input_hit_tokens": usage["prompt_cache_hit_tokens"],
                "output_tokens": usage["completion_tokens"],
                "finish_reason": finish_reason,
                "waste": self.waste_guard.is_waste(finish_reason, content),
                "response_model": model,
                "system_fingerprint": fingerprint,
                "recorded": False,
                "error": None,
                "raw_content": None,
            }

            try:
                enforce_flash_response(response)
            except ModelMismatch:
                entry["error"] = "model_mismatch"
                entry["event_id"] = uuid.uuid4().hex[:12]
                entry["raw_content"] = content
                self._append_journal(entry)
                result.stopped_reason = "model_mismatch"
                break

            try:
                ensure_fingerprint(
                    response,
                    allow_register=True,
                    repin=self._repin_pending,
                    source="writer",
                    path=self.fingerprint_path,
                )
            except FingerprintMismatch:
                entry["error"] = "fingerprint_mismatch"
                entry["event_id"] = uuid.uuid4().hex[:12]
                entry["raw_content"] = content
                self._append_journal(entry)
                result.stopped_reason = "fingerprint_mismatch"
                break

            self._repin_pending = False
            try:
                tests = _parse_tests(_json_from_text(content)) if not entry["waste"] else []
            except Exception:
                entry["error"] = "parse_error"
                entry["raw_content"] = content
                self._append_journal(entry)
                result.failed_problems += 1
                if self.waste_guard.record(finish_reason, content):
                    self._new_trip(result, journal, problem.get("problem_id"))
                    break
                continue

            if not entry["waste"]:
                try:
                    tests, dropped = self._drop_failing_tests(problem, tests)
                except (MissingAPIKey, EndpointRefused):
                    raise
                except Exception as exc:
                    entry["error"] = f"error:{type(exc).__name__}"
                    entry["raw_content"] = content
                    self._append_journal(entry)
                    result.stopped_reason = f"error:{type(exc).__name__}"
                    break
                result.dropped_tests += dropped
                entry["dropped_tests"] = dropped
                if not tests:
                    result.failed_problems += 1
                else:
                    record = {
                        "problem_id": problem["problem_id"],
                        "source": problem.get("source"),
                        "tests": tests,
                        "written_by": REGISTERED_MODEL,
                        "api_model": self.model,
                        "response_model": model,
                        "system_fingerprint": fingerprint,
                        "estimated_usd": actual["total_usd"],
                        "shared_by_arms": True,
                    }
                    result.records.append(record)
                    self._append_partial(record)
                    entry["recorded"] = True

            # Journal every paid response *before* the guard can stop the run.
            self._append_journal(entry)
            if self.waste_guard.record(finish_reason, content):
                self._new_trip(result, journal, problem.get("problem_id"))
                break

        result.waste = max(result.waste, self.waste_guard.waste_count)
        result.spent_usd = self.accountant.cumulative_usd()
        result.estimated = estimate
        return result


__all__ = [
    "API_KEY_ENV",
    "API_MODEL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DeepSeekError",
    "EndpointRefused",
    "HARD_GLOBAL_CAP_USD",
    "HiddenTestWriter",
    "MissingAPIKey",
    "OFFICIAL_BASE_URL",
    "PEAK_MULTIPLIER",
    "PRICES_PER_MILLION",
    "REGISTERED_MODEL",
    "SpendingCapExceeded",
    "WriteResult",
    "api_key_from_env",
    "estimate_cost",
    "estimate_tokens",
    "is_peak",
    "seconds_until_off_peak",
    "validate_endpoint",
    "wait_for_off_peak",
]
