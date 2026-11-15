"""Shared money, model and waste guards for every DeepSeek call site.

The author pays DeepSeek with a credit card and ordered that all inference
happens in off-peak hours and that expensive inference is avoided.  This module
is the single implementation used by the hidden-test writer, the vision judge,
the vision gate and their scripts:

* **Off-peak gate.**  A request may only start outside the peak windows
  (01:00-04:00 and 06:00-10:00 UTC, Monday-Friday), at least 15 minutes before
  the next peak and at least 5 minutes after the previous one.  The gate sleeps
  to the safe boundary and re-checks after every sleep; a clock that does not
  advance is an error, never a silent pass.
* **Model pin.**  Only ``deepseek-flash`` is accepted as a request model.  The
  first response must be exactly the served version
  ``DeepSeek-V4.1-Flash-1017`` or the documented API alias ``deepseek-flash``;
  anything else stops the run.  The first ``system_fingerprint`` is pinned.
* **Explicit thinking.**  Call sites must set ``extra_body`` thinking
  explicitly; the provider default (thinking on, effort ``high``, 64K output)
  is never relied on.
* **Spend.**  Every call reserves prompt estimate + ``max_tokens`` at the
  off-peak cache-miss input price before the request and settles with actual
  usage (cache hits at $0.003 per million).  A response without usage settles
  at the full reservation, never at $0.
* **Waste guard.**  A sliding window over the last 20 responses (partial windows
  included) counts ``finish_reason == "length"`` and empty content; more than
  10% stops the run.  Paid responses are journaled before stopping.
* **Paid pilot.**  A live run processes at most ``--pilot`` items (<=50), writes
  a report bound to the exact run config and stops.  ``--after-pilot`` refuses
  stale, overwritten, incomplete or mismatched reports.

Nothing here observes the network or the environment at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

API_KEY_ENV = "DEEPSEEK_API_KEY"
OFFICIAL_BASE_URL = "https://api.deepseek.com"
API_MODEL = "deepseek-flash"
REGISTERED_MODEL = "DeepSeek-V4.1-Flash-1017"
# The only response model strings that identify the registered served version.
ACCEPTED_RESPONSE_MODELS = frozenset({API_MODEL, REGISTERED_MODEL})

DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_JUDGE_MAX_USD = 5.0
HARD_GLOBAL_CAP_USD = 30.0
PEAK_MULTIPLIER = 2.0

# Every DeepSeek SDK client uses these; retries are ours (and re-gated).
CLIENT_MAX_RETRIES = 0
REQUEST_TIMEOUT_SECONDS = 240.0

# Official DeepSeek V4.1 Flash prices, USD per million tokens (off-peak).
PRICES_PER_MILLION = {
    "input": 0.15,
    "output": 0.60,
    "cached_input": 0.003,
}

# Peak windows in UTC, Monday-Friday.  Chinese public holidays are treated as
# normal weekdays on purpose (conservative).
PEAK_HOURS_UTC = ((1, 4), (6, 10))
PEAK_LOOKAHEAD_SECONDS = 15 * 60
PEAK_END_MARGIN_SECONDS = 5 * 60
IMAGE_TOKEN_ESTIMATE = 2048
MESSAGE_OVERHEAD_TOKENS = 16

WASTE_WINDOW_SIZE = 20
WASTE_RATE_LIMIT = 0.10

PILOT_DEFAULT_ITEMS = 20
PILOT_MAX_ITEMS = 50
PILOT_REPORT_VERSION = 3

FINGERPRINT_RECORD_FILENAME = "deepseek_fingerprint.json"
# Rule (e): more than 10% of a 20-response window, expressed as an absolute
# count so partial windows behave the same way.
WASTE_TRIP_COUNT = max(1, int(math.floor(WASTE_WINDOW_SIZE * WASTE_RATE_LIMIT)) + 1)

# Journal error values that must be surfaced as pilot events (N1).
PILOT_EVENT_ERRORS = frozenset(
    {"model_mismatch", "fingerprint_mismatch", "peak_request"}
)


def utcnow() -> datetime:
    """Current UTC time; tests replace this symbol with a fake clock."""

    return datetime.now(timezone.utc)


class DeepSeekError(RuntimeError):
    """Base class for every DeepSeek guard failure."""


class EndpointRefused(DeepSeekError):
    pass


class MissingAPIKey(DeepSeekError):
    pass


class SpendingCapExceeded(DeepSeekError):
    pass


class ModelRefused(DeepSeekError):
    pass


class ModelMismatch(DeepSeekError):
    pass


class FingerprintMismatch(DeepSeekError):
    pass


class WasteGuardTriggered(DeepSeekError):
    pass


class PilotReportError(DeepSeekError):
    pass


class WasteTripAcknowledgmentRequired(DeepSeekError):
    """An earlier run tripped the waste guard and needs a human ack."""


class OffPeakGateError(DeepSeekError):
    pass


class PeakRequestDetected(DeepSeekError):
    """The provider says a request was created inside a peak window."""

    def __init__(self, message: str, *, actual: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.actual = actual


# ---------------------------------------------------------------------------
# Endpoint / key helpers (kept here so every call site shares them)
# ---------------------------------------------------------------------------


def validate_endpoint(base_url: str) -> str:
    """Refuse anything except the official DeepSeek Open Platform API."""

    value = (base_url or "").strip()
    lowered = value.lower()
    if "opencode" in lowered:
        raise EndpointRefused(
            "OpenCode-hosted endpoints are refused for anything that enters "
            "training or results; use the official DeepSeek API."
        )
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc.lower() != "api.deepseek.com":
        raise EndpointRefused(
            f"refusing non-official endpoint {value!r}; only "
            f"{OFFICIAL_BASE_URL} is allowed"
        )
    return value.rstrip("/")


def api_key_from_env(environ: dict[str, str] | None = None) -> str:
    """Return the key from the environment, never from a file."""

    source = os.environ if environ is None else environ
    value = source.get(API_KEY_ENV, "").strip()
    if not value:
        raise MissingAPIKey(
            f"{API_KEY_ENV} is not set; the key is read from the environment only"
        )
    return value


def estimate_tokens(text: str, counter: Callable[[str], int] | None = None) -> int:
    """Conservative upper-bound token estimate with no tokenizer.

    ``chars / 2`` is deliberately pessimistic for code and English prompts and
    also covers the roughly 0.6 tokens per CJK character reported by the
    provider, so the spend reservation never undercounts the prompt.
    """

    if counter is not None:
        return int(counter(text))
    return max(1, int(math.ceil(len(text or "") / 2)))


# ---------------------------------------------------------------------------
# Off-peak gate
# ---------------------------------------------------------------------------


def _as_utc(now: datetime | None = None) -> datetime:
    now = now or utcnow()
    if now.tzinfo is None:
        raise ValueError(
            "naive datetimes are refused; pass a timezone-aware UTC datetime"
        )
    return now.astimezone(timezone.utc)


def _peak_end_for_start(start: datetime) -> datetime:
    for low, high in PEAK_HOURS_UTC:
        if start.hour == low:
            return start.replace(hour=high, minute=0, second=0, microsecond=0)
    raise ValueError(f"not a peak start: {start!r}")


def _peak_window(now: datetime) -> tuple[datetime, datetime] | None:
    if now.weekday() >= 5:
        return None
    for low, high in PEAK_HOURS_UTC:
        start = now.replace(hour=low, minute=0, second=0, microsecond=0)
        end = now.replace(hour=high, minute=0, second=0, microsecond=0)
        if start <= now < end:
            return start, end
    return None


def _post_peak_margin(now: datetime) -> tuple[datetime, datetime] | None:
    """The 5-minute safety margin immediately after a peak window."""

    if now.weekday() >= 5:
        return None
    for _low, high in PEAK_HOURS_UTC:
        end = now.replace(hour=high, minute=0, second=0, microsecond=0)
        margin_end = end + timedelta(seconds=PEAK_END_MARGIN_SECONDS)
        if end <= now < margin_end:
            return end, margin_end
    return None


def _next_peak_start(now: datetime) -> datetime | None:
    for offset in range(0, 8):
        day: date = (now + timedelta(days=offset)).date()
        if day.weekday() >= 5:
            continue
        for low, _high in PEAK_HOURS_UTC:
            candidate = datetime(
                day.year, day.month, day.day, low, tzinfo=timezone.utc
            )
            if candidate > now:
                return candidate
    return None


def is_peak(now: datetime | None = None) -> bool:
    """True only while inside a peak window (margins excluded)."""

    return _peak_window(_as_utc(now)) is not None


def seconds_until_off_peak(now: datetime | None = None) -> float:
    """Seconds to sleep so that a request starts safely off-peak.

    Inside a peak window this is the time to its end plus the 5-minute margin.
    Within 15 minutes before a peak window it is the time to the end of *that*
    window plus the margin.  The 5 minutes after a window are also gated.
    """

    current = _as_utc(now)
    window = _peak_window(current)
    if window is not None:
        return max(
            0.0,
            (window[1] - current).total_seconds() + PEAK_END_MARGIN_SECONDS,
        )

    margin = _post_peak_margin(current)
    if margin is not None:
        return max(0.0, (margin[1] - current).total_seconds())

    next_start = _next_peak_start(current)
    if next_start is not None:
        lead = (next_start - current).total_seconds()
        if 0 <= lead <= PEAK_LOOKAHEAD_SECONDS:
            end = _peak_end_for_start(next_start)
            return max(
                0.0,
                (end - current).total_seconds() + PEAK_END_MARGIN_SECONDS,
            )
    return 0.0


def wait_for_off_peak(
    *,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    """Sleep until a request is safe; returns the total seconds slept.

    Fails closed: after every sleep the clock is re-read and checked, and a
    clock that does not advance raises :class:`OffPeakGateError` instead of
    letting the caller send at peak.
    """

    now_fn = now_fn or utcnow
    total = 0.0
    for _ in range(8):
        current = _as_utc(now_fn())
        delay = seconds_until_off_peak(current)
        if delay <= 0:
            break
        sleep_fn(delay)
        total += delay
        after = _as_utc(now_fn())
        if after <= current:
            raise OffPeakGateError(
                "off-peak gate clock did not advance after sleeping; refusing "
                "to send a request at an unverified time"
            )
        if seconds_until_off_peak(after) <= 0:
            return total

    final = _as_utc(now_fn())
    if seconds_until_off_peak(final) > 0:
        raise OffPeakGateError(
            f"still inside/next to a peak window at {final.isoformat()}; "
            "refusing to send a request"
        )
    return total


# ---------------------------------------------------------------------------
# Model pin
# ---------------------------------------------------------------------------


def validate_model(model: str) -> str:
    """Only the registered API alias is accepted."""

    value = str(model or "").strip()
    if value != API_MODEL:
        raise ModelRefused(
            f"model {value!r} is refused; DeepSeek requests must use "
            f"{API_MODEL!r} (the legacy deepseek-chat/deepseek-reasoner "
            "aliases are retired)"
        )
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def response_content(response: Any) -> str:
    choices = _field(response, "choices") or []
    if not choices:
        return ""
    message = _field(choices[0], "message")
    content = _field(message, "content")
    return content if isinstance(content, str) else ""


def response_finish_reason(response: Any) -> Any:
    choices = _field(response, "choices") or []
    if not choices:
        return None
    return _field(choices[0], "finish_reason")


def response_created_at(response: Any) -> datetime | None:
    created = _field(response, "created")
    if created is None:
        return None
    try:
        return datetime.fromtimestamp(float(created), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def enforce_flash_response(response: Any) -> tuple[Any, Any]:
    """Record and validate ``response.model`` / ``system_fingerprint``.

    Only the exact served version or the exact documented API alias identify
    DeepSeek-V4.1-Flash.  A missing model or any look-alike (for example
    ``DeepSeek-V5-Flash-0301``) is refused.  The caller stops after that one
    response.
    """

    model = _field(response, "model")
    fingerprint = _field(response, "system_fingerprint")
    if not isinstance(model, str) or model not in ACCEPTED_RESPONSE_MODELS:
        raise ModelMismatch(
            f"provider model {model!r} is not {REGISTERED_MODEL!r} or the "
            f"documented alias {API_MODEL!r}; stopping the run"
        )
    return model, fingerprint


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def estimate_cost(
    prompt_tokens: int,
    max_output_tokens: int,
    *,
    n_requests: int = 1,
    peak: bool = False,
    prices: dict[str, float] | None = None,
    multiplier: float = PEAK_MULTIPLIER,
) -> dict[str, Any]:
    """Worst-case USD estimate at the cache-miss input price (off-peak)."""

    prices = dict(prices or PRICES_PER_MILLION)
    factor = float(multiplier) if peak else 1.0
    input_tokens = int(prompt_tokens) * int(n_requests)
    output_tokens = int(max_output_tokens) * int(n_requests)
    input_cost = input_tokens / 1_000_000 * float(prices["input"]) * factor
    output_cost = output_tokens / 1_000_000 * float(prices["output"]) * factor
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_usd": input_cost + output_cost,
        "peak": bool(peak),
        "price_multiplier": factor,
        "prices_per_million": prices,
    }


def _usage_source(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        if "usage" in value:
            usage = value.get("usage") or {}
            return usage if isinstance(usage, dict) else _namespace_dict(usage)
        return value
    usage = getattr(value, "usage", None)
    if usage is None:
        return {}
    return usage if isinstance(usage, dict) else _namespace_dict(usage)


def _namespace_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    except TypeError:
        return {}


def usage_breakdown(value: Any) -> dict[str, int]:
    """Normalise prompt/cache/output token counts from a response or usage."""

    usage = _usage_source(value)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss_raw = usage.get("prompt_cache_miss_tokens")
    miss = int(miss_raw) if miss_raw is not None else max(0, prompt - hit)
    return {
        "prompt_tokens": max(prompt, hit + miss),
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "total_tokens": max(prompt, hit + miss) + completion,
    }


def cost_from_usage(
    value: Any,
    prices: dict[str, float] | None = None,
    *,
    peak: bool = False,
    multiplier: float = PEAK_MULTIPLIER,
) -> dict[str, Any]:
    """Actual USD cost using cache-hit/miss input prices."""

    prices = dict(prices or PRICES_PER_MILLION)
    tokens = usage_breakdown(value)
    factor = float(multiplier) if peak else 1.0
    hit = tokens["prompt_cache_hit_tokens"]
    miss = tokens["prompt_cache_miss_tokens"]
    output = tokens["completion_tokens"]
    hit_cost = hit / 1_000_000 * float(prices["cached_input"]) * factor
    miss_cost = miss / 1_000_000 * float(prices["input"]) * factor
    output_cost = output / 1_000_000 * float(prices["output"]) * factor
    return {
        **tokens,
        "cache_hit_cost_usd": hit_cost,
        "cache_miss_cost_usd": miss_cost,
        "output_cost_usd": output_cost,
        "total_usd": hit_cost + miss_cost + output_cost,
        "peak": bool(peak),
        "price_multiplier": factor,
        "prices_per_million": prices,
    }


def estimate_request_prompt_tokens(request: dict[str, Any]) -> int:
    """Conservative upper-bound prompt estimate for an OpenAI-style request."""

    total = 0
    for message in request.get("messages") or []:
        total += MESSAGE_OVERHEAD_TOKENS
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind in ("text", "input_text"):
                    total += estimate_tokens(str(part.get("text") or ""))
                elif kind in ("image_url", "input_image", "file"):
                    total += IMAGE_TOKEN_ESTIMATE
        elif isinstance(content, dict):
            total += estimate_tokens(str(content.get("text") or ""))
    return max(1, total)


# ---------------------------------------------------------------------------
# Hard total deadline and SDK-free registry
# ---------------------------------------------------------------------------


def send_with_deadline(
    fn: Callable[[], Any],
    *,
    deadline_s: float = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """Run one provider call with a wall-clock deadline.

    The OpenAI SDK's ``timeout`` is a per-read timeout, so a provider that
    streams keep-alive bytes can hold a request open past the peak margin.
    This wrapper enforces a total deadline for our process; on expiry the run
    stops and the caller settles the reservation conservatively.
    """

    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # pragma: no cover - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(float(deadline_s))
    if thread.is_alive():
        raise DeepSeekError(
            f"DeepSeek request exceeded the {deadline_s:g}s total deadline"
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def fingerprint_record_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    from gemmaloop.paths import artifacts_dir

    return artifacts_dir() / "stage0" / FINGERPRINT_RECORD_FILENAME


def load_fingerprint_record(path: str | Path | None = None) -> dict[str, Any] | None:
    target = fingerprint_record_path(path)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FingerprintMismatch(f"invalid DeepSeek fingerprint record at {target}")
    model = payload.get("model")
    fingerprint = payload.get("fingerprint")
    if not isinstance(model, str) or model not in ACCEPTED_RESPONSE_MODELS:
        raise FingerprintMismatch(
            f"registered DeepSeek fingerprint has invalid model {model!r}"
        )
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise FingerprintMismatch(
            f"registered DeepSeek fingerprint at {target} is empty"
        )
    return payload


def write_fingerprint_record(
    *,
    model: str,
    fingerprint: str,
    source: str,
    path: str | Path | None = None,
    repin: bool = False,
    created: str | None = None,
) -> dict[str, Any]:
    """Write the single registered fingerprint record.

    An existing record is never changed silently; callers must pass
    ``repin=True`` (the CLI's deliberate ``--repin`` decision).
    """

    if not isinstance(model, str) or model not in ACCEPTED_RESPONSE_MODELS:
        raise ModelMismatch(
            f"cannot register fingerprint for model {model!r}; expected "
            f"{REGISTERED_MODEL!r} or {API_MODEL!r}"
        )
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise FingerprintMismatch(
            "a non-empty system_fingerprint is required to register DeepSeek"
        )
    target = fingerprint_record_path(path)
    existing = load_fingerprint_record(target)
    if existing is not None and not repin:
        if existing.get("fingerprint") == fingerprint:
            return existing
        raise FingerprintMismatch(
            "registered DeepSeek fingerprint changed: recorded "
            f"{existing.get('fingerprint')!r}, provider returned "
            f"{fingerprint!r}; pass --repin for a deliberate new record"
        )
    history = list(existing.get("history", [])) if existing is not None else []
    if existing is not None and repin:
        previous = {key: value for key, value in existing.items() if key != "history"}
        history.append(previous)
        if len(history) > 21:
            history = [history[0]] + history[-20:]
    record = {
        "version": 1,
        "date": created or utcnow().isoformat(),
        "model": model,
        "fingerprint": fingerprint,
        "source": source,
        "history": history,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return record


def ensure_fingerprint(
    response: Any,
    *,
    allow_register: bool = False,
    repin: bool = False,
    source: str = "writer",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Check one paid response against the single registered fingerprint."""

    model, fingerprint = enforce_flash_response(response)
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise FingerprintMismatch(
            "response has no system_fingerprint; refusing to use an unpinned "
            "DeepSeek response"
        )
    existing = load_fingerprint_record(path)
    if existing is None:
        if not allow_register:
            raise FingerprintMismatch(
                "no registered DeepSeek fingerprint; run the vision gate or a "
                "paid writer first"
            )
        return write_fingerprint_record(
            model=model,
            fingerprint=fingerprint,
            source=source,
            path=path,
            repin=repin,
        )
    if repin:
        return write_fingerprint_record(
            model=model,
            fingerprint=fingerprint,
            source=source,
            path=path,
            repin=True,
        )
    if existing.get("fingerprint") != fingerprint:
        raise FingerprintMismatch(
            "DeepSeek system_fingerprint changed: registered "
            f"{existing.get('fingerprint')!r}, provider returned "
            f"{fingerprint!r}"
        )
    return existing


def clear_fingerprint_record(path: str | Path | None = None) -> None:
    target = fingerprint_record_path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Paid-response journal helpers
# ---------------------------------------------------------------------------


def response_journal_path(partial_path: str | Path) -> Path:
    target = Path(partial_path)
    return target.with_name(target.name + ".responses.jsonl")


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_journal(path: str | Path, entry: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stable_event_id(entry: dict[str, Any], index: int) -> str:
    seed = ":".join(
        str(entry.get(key))
        for key in (
            "kind",
            "problem_id",
            "error",
            "response_model",
            "system_fingerprint",
            index,
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def summarize_pilot_journal(
    entries: Iterable[dict[str, Any]], planned_ids: Iterable[str]
) -> dict[str, Any]:
    """Aggregate all pilot runs on one output path from its journal."""

    planned = [str(item) for item in planned_ids]
    planned_set = set(planned)
    entries = list(entries)
    attempts = [
        entry
        for entry in entries
        if entry.get("kind") in ("response", "failure")
        and str(entry.get("problem_id")) in planned_set
    ]
    sent = [entry for entry in attempts if entry.get("sent", True)]
    paid = [
        entry
        for entry in attempts
        if entry.get("kind") == "response" and entry.get("paid", True)
    ]
    paid_ids = {str(entry.get("problem_id")) for entry in paid}
    accounted = [pid for pid in planned if pid in paid_ids]
    trips = [
        str(entry.get("trip_id"))
        for entry in entries
        if entry.get("kind") == "waste_trip" and entry.get("trip_id")
    ]
    waste_acks = {
        str(entry.get("trip_id"))
        for entry in entries
        if entry.get("kind") == "waste_ack" and entry.get("trip_id")
    }
    event_acks = {
        str(entry.get("event_id"))
        for entry in entries
        if entry.get("kind") == "pilot_event_ack" and entry.get("event_id")
    }
    acked_event_ids = waste_acks | event_acks

    events: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        error = entry.get("error")
        if (
            entry.get("kind") == "response"
            and str(entry.get("problem_id")) in planned_set
            and error in PILOT_EVENT_ERRORS
        ):
            events.append(
                {
                    "event_id": entry.get("event_id")
                    or _stable_event_id(entry, index),
                    "kind": str(error),
                    "problem_id": entry.get("problem_id"),
                    "response_model": entry.get("response_model"),
                    "system_fingerprint": entry.get("system_fingerprint"),
                }
            )
        elif entry.get("kind") == "waste_trip" and entry.get("trip_id"):
            events.append(
                {
                    "event_id": str(entry.get("trip_id")),
                    "kind": "waste_guard",
                    "problem_id": entry.get("problem_id"),
                    "waste_count": entry.get("waste_count"),
                }
            )

    unacked = [
        event["event_id"]
        for event in events
        if event["event_id"] not in acked_event_ids
    ]
    return {
        "requests": len(sent),
        "usd": sum(float(entry.get("actual_usd", 0.0) or 0.0) for entry in attempts),
        "input_miss_tokens": sum(
            int(entry.get("input_miss_tokens", 0) or 0) for entry in paid
        ),
        "input_hit_tokens": sum(
            int(entry.get("input_hit_tokens", 0) or 0) for entry in paid
        ),
        "output_tokens": sum(
            int(entry.get("output_tokens", 0) or 0) for entry in paid
        ),
        "waste_count": sum(1 for entry in paid if entry.get("waste")),
        "records_written": sum(1 for entry in paid if entry.get("recorded")),
        "parse_failures": sum(1 for entry in paid if entry.get("error") == "parse_error"),
        "dropped_tests": sum(int(entry.get("dropped_tests", 0) or 0) for entry in paid),
        "planned_candidate_ids": planned,
        "accounted_candidate_ids": accounted,
        "all_accounted": bool(planned) and planned_set <= paid_ids,
        "waste_trip_ids": trips,
        "acked_waste_trip_ids": sorted(waste_acks & set(trips)),
        "latest_waste_trip_id": trips[-1] if trips else None,
        "waste_acknowledged": bool(trips) and set(trips) <= acked_event_ids,
        "pilot_events": events,
        "acked_pilot_event_ids": sorted(acked_event_ids),
        "unacked_pilot_event_ids": unacked,
        "events_acknowledged": bool(events) and not unacked,
    }




# ---------------------------------------------------------------------------
# Waste guard
# ---------------------------------------------------------------------------


class WasteGuard:
    """Sliding window over the last 20 paid responses.

    The trip rule is an absolute count (3 waste responses), so a partial window
    with one or two waste responses does not stop a short pilot.  The window is
    evaluated after every response and never raises; call sites journal the
    paid response before they stop.
    """

    def __init__(
        self,
        *,
        window: int = WASTE_WINDOW_SIZE,
        rate: float = WASTE_RATE_LIMIT,
        trip_count: int | None = None,
    ) -> None:
        self.window = int(window)
        self.rate = float(rate)
        self.trip_count = (
            int(trip_count)
            if trip_count is not None
            else max(1, int(math.floor(self.window * self.rate)) + 1)
        )
        self.waste_count = 0  # total waste in the run
        self.total_count = 0
        self.triggered = False
        self._window: deque[bool] = deque(maxlen=self.window)

    @staticmethod
    def is_waste(finish_reason: Any, content: Any) -> bool:
        return finish_reason == "length" or not str(content or "").strip()

    def _append(self, is_waste: bool) -> bool:
        self._window.append(bool(is_waste))
        self.total_count += 1
        if is_waste:
            self.waste_count += 1
        if sum(1 for item in self._window if item) >= self.trip_count:
            self.triggered = True
        return self.triggered

    def record(self, finish_reason: Any, content: Any) -> bool:
        if self.triggered:
            return True
        return self._append(self.is_waste(finish_reason, content))

    def prime(self, history: Iterable[bool]) -> bool:
        """Rebuild the sliding window from a persisted journal."""

        for item in history:
            self._append(bool(item))
        return self.triggered

    def snapshot(self) -> list[bool]:
        return list(self._window)


# ---------------------------------------------------------------------------
# Spend accountant
# ---------------------------------------------------------------------------


class CallAccountant:
    """Off-peak gate + monthly ledger reservation/settlement for one call site.

    With ``ledger=None`` the same checks run against process-local counters,
    which keeps unit tests and one-off calls honest without touching the
    author's real ledger.  Every call site in the product passes the shared
    ledger (and holds the spend lock) by default.
    """

    def __init__(
        self,
        *,
        ledger: Any | None = None,
        month: str | None = None,
        provider: str = "deepseek",
        max_usd: float | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        prices: dict[str, float] | None = None,
    ) -> None:
        if max_usd is not None and not 0 < float(max_usd) <= HARD_GLOBAL_CAP_USD:
            raise ValueError(
                f"max_usd must be in (0, {HARD_GLOBAL_CAP_USD:g}]; "
                f"got {max_usd!r}"
            )
        self.ledger = ledger
        self.month = month
        self.provider = provider
        self.max_usd = None if max_usd is None else float(max_usd)
        self.prices = dict(prices or PRICES_PER_MILLION)
        self._now = now_fn or utcnow
        self._sleep = sleep_fn
        self._spent = 0.0
        self._reserved: dict[str, float] = {}
        self.last_actual: dict[str, Any] | None = None
        self.last_actual_exceeds_reservation = False

    def cumulative_usd(self) -> float:
        if self.ledger is not None:
            return float(
                self.ledger.cumulative_usd(self.month, provider=self.provider)
            )
        return self._spent + sum(self._reserved.values())

    def _assert_cap(self, usd: float) -> None:
        if self.max_usd is not None and self.cumulative_usd() + usd > self.max_usd + 1e-12:
            raise SpendingCapExceeded(
                f"next request reserves ${usd:.6f}; cumulative "
                f"${self.cumulative_usd():.6f} would exceed the hard cap "
                f"${self.max_usd:.6f}"
            )

    def before_request(
        self,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
        shard: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        estimate = estimate_cost(
            prompt_tokens, max_output_tokens, prices=self.prices
        )
        usd = float(estimate["total_usd"])
        self._assert_cap(usd)
        wait_for_off_peak(now_fn=self._now, sleep_fn=self._sleep)
        self._assert_cap(usd)
        if self.ledger is not None:
            reservation = self.ledger.reserve(
                shard=int(shard),
                usd=usd,
                month=self.month,
                provider=self.provider,
                metadata=metadata or {},
            )
        else:
            reservation = f"memory:{uuid.uuid4().hex}"
        self._reserved[reservation] = usd
        return reservation, estimate

    def after_response(
        self,
        reservation: str,
        response: Any,
        *,
        shard: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        reserved = float(self._reserved.get(reservation, 0.0))
        usage = _usage_source(response)
        tokens = usage_breakdown(response)
        usage_missing = (not usage) or tokens["total_tokens"] == 0
        created = response_created_at(response)
        peak_observed = created is not None and is_peak(created)

        if usage_missing:
            # With no usage the reservation is the worst case; if the provider
            # timestamp is inside peak, the worst case is twice the reservation.
            factor = PEAK_MULTIPLIER if peak_observed else 1.0
            actual = {
                **tokens,
                "cache_hit_cost_usd": 0.0,
                "cache_miss_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_usd": reserved * factor,
                "peak": bool(peak_observed),
                "price_multiplier": factor,
                "prices_per_million": self.prices,
                "usage_missing": True,
            }
        else:
            actual = cost_from_usage(
                response, prices=self.prices, peak=peak_observed
            )

        self.last_actual = actual
        self.last_actual_exceeds_reservation = (
            not usage_missing
            and actual["total_usd"] > reserved + 1e-12
        )
        merged = dict(metadata or {})
        if usage_missing:
            merged["usage_missing"] = True
        if self.last_actual_exceeds_reservation:
            merged["actual_exceeds_reservation"] = True
        if peak_observed:
            merged["peak_observed"] = True
        self._settle(
            reservation, actual["total_usd"], shard=shard, metadata=merged
        )
        if peak_observed:
            raise PeakRequestDetected(
                f"provider created a response at {created.isoformat()} inside "
                "a peak window; settled at the peak multiplier and stopped",
                actual=actual,
            )
        return actual, tokens

    def settle_failure(
        self,
        reservation: str,
        *,
        shard: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Settle at the reserved worst case after an error (conservative).

        Returns the amount settled so the caller can account for failed sends
        in a pilot projection.
        """

        reserved = self._reserved.get(reservation, 0.0)
        self._settle(reservation, reserved, shard=shard, metadata=metadata)
        return reserved

    def _settle(
        self,
        reservation: str,
        actual_usd: float,
        *,
        shard: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.ledger is not None:
            self.ledger.settle(
                reservation,
                shard=int(shard),
                actual_usd=float(actual_usd),
                month=self.month,
                provider=self.provider,
                metadata=metadata or {},
            )
        else:
            self._spent += float(actual_usd)
        self._reserved.pop(reservation, None)


# ---------------------------------------------------------------------------
# Shared default ledger / spend lock
# ---------------------------------------------------------------------------


def shared_accountant(
    *,
    max_usd: float,
    ledger: Any | None = None,
    month: str | None = None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    provider: str = "deepseek",
    ledger_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    prices: dict[str, float] | None = None,
) -> tuple[CallAccountant, Any, str]:
    """Build an accountant over the shared monthly ledger and spend lock.

    Every product call site uses this by default; unit tests redirect the
    artifacts directory and the lock path with environment variables.
    """

    from gemmaloop.stage0.spend import (
        SpendLedger,
        SpendLockBusy,
        default_ledger_path,
        held_spend_lock_path,
        month_key,
        require_spend_lock,
        spend_lock_path,
    )

    held = held_spend_lock_path()
    wanted = spend_lock_path(lock_path)
    if held is not None and held != wanted:
        raise SpendLockBusy(
            f"this process already holds the spend lock {held}; refusing to "
            f"replace it with {wanted}"
        )
    require_spend_lock(lock_path)
    target = Path(ledger_path) if ledger_path is not None else default_ledger_path()
    shared_ledger = ledger if ledger is not None else SpendLedger(target)
    now = _as_utc((now_fn or utcnow)())
    shared_month = month or month_key(now)
    accountant = CallAccountant(
        ledger=shared_ledger,
        month=shared_month,
        provider=provider,
        max_usd=max_usd,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        prices=prices,
    )
    return accountant, shared_ledger, shared_month


# ---------------------------------------------------------------------------
# Paid pilot
# ---------------------------------------------------------------------------


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PilotReportError(f"pilot report {name} is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise PilotReportError(
            f"pilot report {name} must be finite and non-negative; got {value!r}"
        )
    return number


def pilot_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pilot_report(
    *,
    requests: int,
    usd: float,
    pilot_items: int,
    total_items: int,
    input_miss_tokens: int = 0,
    input_hit_tokens: int = 0,
    output_tokens: int = 0,
    waste_count: int = 0,
    stopped_reason: str | None = None,
    month_to_date_usd: float = 0.0,
    config: dict[str, Any] | None = None,
    candidate_ids: list[str] | None = None,
    pilot_candidate_ids: list[str] | None = None,
    planned_candidate_ids: list[str] | None = None,
    accounted_candidate_ids: list[str] | None = None,
    waste_acknowledged: bool = False,
    waste_trip_ids: list[str] | None = None,
    acked_waste_trip_ids: list[str] | None = None,
    waste_trip_id: str | None = None,
    pilot_events: list[dict[str, Any]] | None = None,
    acked_pilot_event_ids: list[str] | None = None,
    unacked_pilot_event_ids: list[str] | None = None,
    records_written: int = 0,
    parse_failures: int = 0,
    dropped_tests: int = 0,
    limit: int | None = None,
    pool_manifest_sha256: str | None = None,
    model: str = API_MODEL,
    registered_model: str = REGISTERED_MODEL,
    created: str | None = None,
) -> dict[str, Any]:
    """Build the JSON payload written after a paid pilot."""

    requests = int(requests)
    pilot_items = int(pilot_items)
    total_items = int(total_items)
    usd = _finite(usd, "usd")
    month_to_date_usd = _finite(month_to_date_usd, "month_to_date_usd")
    projected = float(usd) * total_items / requests if requests > 0 else 0.0
    projected_total = month_to_date_usd + projected
    report = {
        "kind": "deepseek_paid_pilot",
        "version": PILOT_REPORT_VERSION,
        "model": model,
        "registered_model": registered_model,
        "requests": requests,
        "pilot_items": pilot_items,
        "total_items": total_items,
        "input_miss_tokens": int(input_miss_tokens),
        "input_hit_tokens": int(input_hit_tokens),
        "output_tokens": int(output_tokens),
        "waste_count": int(waste_count),
        "waste_rate": (float(waste_count) / requests if requests else 0.0),
        "usd": usd,
        "month_to_date_usd": month_to_date_usd,
        "projected_usd": projected,
        "projected_total_usd": projected_total,
        "stopped_reason": stopped_reason,
        "waste_acknowledged": bool(waste_acknowledged),
        "waste_trip_id": waste_trip_id,
        "waste_trip_ids": list(waste_trip_ids or ([waste_trip_id] if waste_trip_id else [])),
        "acked_waste_trip_ids": list(acked_waste_trip_ids or []),
        "pilot_events": list(pilot_events or []),
        "acked_pilot_event_ids": list(acked_pilot_event_ids or []),
        "unacked_pilot_event_ids": list(unacked_pilot_event_ids or []),
        "events_acknowledged": bool(pilot_events)
        and not (unacked_pilot_event_ids or []),
        "records_written": int(records_written),
        "parse_failures": int(parse_failures),
        "dropped_tests": int(dropped_tests),
        "created": created or utcnow().isoformat(),
    }
    if planned_candidate_ids is not None:
        report["planned_candidate_ids"] = list(planned_candidate_ids)
        report["accounted_items"] = len(list(accounted_candidate_ids or []))
    if accounted_candidate_ids is not None:
        report["accounted_candidate_ids"] = list(accounted_candidate_ids)
    if config is not None:
        report["config"] = config
        report["config_sha256"] = pilot_config_sha256(config)
    if candidate_ids is not None:
        report["candidate_ids"] = list(candidate_ids)
    if pilot_candidate_ids is not None:
        report["pilot_candidate_ids"] = list(pilot_candidate_ids)
    report["limit"] = limit
    report["pool_manifest_sha256"] = pool_manifest_sha256
    return report


def write_pilot_report(
    path: str | Path, report: dict[str, Any], *, overwrite: bool = False
) -> Path:
    """Write a pilot report; refuses to overwrite an existing report."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise PilotReportError(
            f"pilot report already exists at {target}; refusing to overwrite "
            "it (use --pilot-report with a new path for a new pilot)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_pilot_report(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise PilotReportError(f"pilot report not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "projected_usd" not in payload:
        raise PilotReportError(f"{target} is not a DeepSeek pilot report")
    return payload


def validate_pilot_report(
    report: dict[str, Any],
    *,
    max_usd: float | None = None,
    expected_config_sha256: str | None = None,
    expected_candidate_ids: Iterable[str] | None = None,
    require_complete: bool = True,
    ack_waste_trip: str | None = None,
    ack_pilot_events: Iterable[str] | None = None,
) -> float:
    """Validate a pilot report and return its full-run projection.

    With ``require_complete`` (the default) the report must cover every planned
    pilot item, may span several resumed runs, and must not have stopped early.
    ``require_complete=False`` is used to check the binding of an interrupted
    report before resuming it.
    """

    if not isinstance(report, dict):
        raise PilotReportError("pilot report must be a JSON object")

    requests = int(report.get("requests", 0) or 0)
    pilot_items = int(report.get("pilot_items", 0) or 0)
    total_items = int(report.get("total_items", 0) or 0)
    if require_complete and requests <= 0:
        raise PilotReportError("pilot report has zero sent requests")
    if pilot_items <= 0:
        raise PilotReportError("pilot report has zero pilot items")
    if total_items < pilot_items:
        raise PilotReportError(
            f"pilot total_items {total_items} is smaller than pilot_items "
            f"{pilot_items}"
        )

    planned = report.get("planned_candidate_ids")
    accounted = report.get("accounted_candidate_ids")
    if planned is not None:
        planned_list = [str(item) for item in planned]
        if pilot_items != len(planned_list):
            raise PilotReportError(
                f"pilot report pilot_items {pilot_items} does not match the "
                f"{len(planned_list)} planned candidate ids"
            )
        if require_complete:
            accounted_list = [str(item) for item in (accounted or [])]
            if not set(planned_list) <= set(accounted_list):
                raise PilotReportError(
                    "pilot planned items are not all accounted for: "
                    f"{len(accounted_list)}/{len(planned_list)}"
                )
    elif require_complete and requests < pilot_items:
        raise PilotReportError(
            f"pilot report has {requests} requests for {pilot_items} planned "
            "items"
        )

    supplied_acks = {str(item) for item in (ack_pilot_events or [])}
    event_ids = {
        str(event.get("event_id"))
        for event in (report.get("pilot_events") or [])
        if event.get("event_id")
    }
    reported_acks = {
        str(item) for item in (report.get("acked_pilot_event_ids") or [])
    } | {
        str(item) for item in (report.get("acked_waste_trip_ids") or [])
    }
    acknowledged_events = reported_acks | supplied_acks
    unacked_events = event_ids - acknowledged_events

    stopped_reason = report.get("stopped_reason")
    if require_complete and stopped_reason is not None:
        if not (stopped_reason == "pilot_events" and not unacked_events):
            raise PilotReportError(
                f"pilot stopped early ({stopped_reason!r}); refusing "
                "to continue from an incomplete pilot"
            )
    if require_complete and unacked_events:
        raise PilotReportError(
            "pilot has unacknowledged event(s): "
            f"{sorted(unacked_events)}; pass --ack-pilot-event for each one"
        )

    for key in (
        "input_miss_tokens",
        "input_hit_tokens",
        "output_tokens",
        "waste_count",
        "usd",
        "month_to_date_usd",
        "waste_rate",
        "projected_usd",
        "projected_total_usd",
    ):
        if key in report:
            _finite(report[key], key)

    waste_count = int(report.get("waste_count", 0) or 0)
    acknowledged = bool(report.get("waste_acknowledged", False))
    trip_ids = {str(item) for item in (report.get("waste_trip_ids") or [])}
    acked_ids = {
        str(item) for item in (report.get("acked_waste_trip_ids") or [])
    }
    acknowledged_trips = acknowledged_events | acked_ids
    if ack_waste_trip is not None:
        acknowledged_trips = acknowledged_trips | {ack_waste_trip}
    unacked_trips = trip_ids - acknowledged_trips
    if require_complete and unacked_trips:
        raise PilotReportError(
            "pilot has unacknowledged waste trip(s): "
            f"{sorted(unacked_trips)}; pass --ack-waste-trip or "
            "--ack-pilot-event for each one"
        )
    if require_complete and waste_count >= WASTE_TRIP_COUNT:
        acknowledged = (
            acknowledged
            or bool(trip_ids and not unacked_trips)
            or bool(acknowledged_events & trip_ids)
        )
        if not acknowledged:
            raise PilotReportError(
                f"pilot has {waste_count} waste responses and no acknowledged "
                "waste trip"
            )

    projected = _finite(report.get("projected_usd", 0.0), "projected_usd")
    projected_total = _finite(
        report.get("projected_total_usd", projected), "projected_total_usd"
    )
    if max_usd is not None:
        for name, value in (
            ("projected_usd", projected),
            ("projected_total_usd", projected_total),
        ):
            if value > float(max_usd) + 1e-12:
                raise SpendingCapExceeded(
                    f"pilot {name} ${value:.4f} for the full run is above the "
                    f"hard cap ${float(max_usd):.4f}; refusing --after-pilot"
                )

    if expected_config_sha256 is not None:
        actual = report.get("config_sha256")
        if not actual:
            raise PilotReportError("pilot report is not bound to a run config")
        if actual != expected_config_sha256:
            raise PilotReportError(
                "pilot report belongs to a different run config "
                f"(model/max_tokens/limit/pool/prices); expected "
                f"{expected_config_sha256[:12]}..., got {str(actual)[:12]}..."
            )

    if expected_candidate_ids is not None:
        expected = list(expected_candidate_ids)
        if report.get("candidate_ids") != expected:
            raise PilotReportError(
                "pilot report candidate list does not match the current pool/"
                "limit; re-run the pilot for this exact run"
            )
    return projected


def check_pilot_projection(
    report: dict[str, Any],
    max_usd: float,
    *,
    expected_config_sha256: str | None = None,
    expected_candidate_ids: Iterable[str] | None = None,
    ack_waste_trip: str | None = None,
    ack_pilot_events: Iterable[str] | None = None,
) -> float:
    """Refuse to continue a paid run when the report is invalid or over cap."""

    return validate_pilot_report(
        report,
        max_usd=max_usd,
        expected_config_sha256=expected_config_sha256,
        expected_candidate_ids=expected_candidate_ids,
        ack_waste_trip=ack_waste_trip,
        ack_pilot_events=ack_pilot_events,
    )


__all__ = [
    "ACCEPTED_RESPONSE_MODELS",
    "API_KEY_ENV",
    "API_MODEL",
    "CLIENT_MAX_RETRIES",
    "CallAccountant",
    "DEFAULT_JUDGE_MAX_USD",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DeepSeekError",
    "EndpointRefused",
    "FINGERPRINT_RECORD_FILENAME",
    "FingerprintMismatch",
    "HARD_GLOBAL_CAP_USD",
    "IMAGE_TOKEN_ESTIMATE",
    "MESSAGE_OVERHEAD_TOKENS",
    "MissingAPIKey",
    "ModelMismatch",
    "ModelRefused",
    "OFFICIAL_BASE_URL",
    "OffPeakGateError",
    "PEAK_END_MARGIN_SECONDS",
    "PEAK_HOURS_UTC",
    "PEAK_LOOKAHEAD_SECONDS",
    "PEAK_MULTIPLIER",
    "PILOT_DEFAULT_ITEMS",
    "PILOT_MAX_ITEMS",
    "PILOT_REPORT_VERSION",
    "PRICES_PER_MILLION",
    "PeakRequestDetected",
    "PilotReportError",
    "REGISTERED_MODEL",
    "REQUEST_TIMEOUT_SECONDS",
    "SpendingCapExceeded",
    "WASTE_RATE_LIMIT",
    "WASTE_TRIP_COUNT",
    "WASTE_WINDOW_SIZE",
    "WasteGuard",
    "WasteGuardTriggered",
    "WasteTripAcknowledgmentRequired",
    "api_key_from_env",
    "append_journal",
    "clear_fingerprint_record",
    "build_pilot_report",
    "check_pilot_projection",
    "cost_from_usage",
    "enforce_flash_response",
    "ensure_fingerprint",
    "estimate_cost",
    "estimate_request_prompt_tokens",
    "estimate_tokens",
    "fingerprint_record_path",
    "is_peak",
    "load_fingerprint_record",
    "load_pilot_report",
    "pilot_config_sha256",
    "response_content",
    "response_created_at",
    "read_journal",
    "response_finish_reason",
    "response_journal_path",
    "seconds_until_off_peak",
    "send_with_deadline",
    "shared_accountant",
    "summarize_pilot_journal",
    "usage_breakdown",
    "utcnow",
    "validate_endpoint",
    "validate_model",
    "validate_pilot_report",
    "wait_for_off_peak",
    "write_fingerprint_record",
    "write_pilot_report",
]
