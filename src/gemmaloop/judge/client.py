"""DeepSeek vision judge client.

The judge is blind and pairwise: it receives the fixed rubric first (so the
provider's prefix cache can hit), then the screenshots in the ``user`` message
(two per candidate for HTML, one per candidate for SVG).  The response is JSON
mode and is validated with :class:`Verdict`.

Nothing here is executed at import time: the API key is only read when a
client is built, and it is never logged.  Every call keeps the request, the raw
response, the token usage and the provider's ``system_fingerprint`` so a run
can be audited after the fact.
"""

from __future__ import annotations

import base64
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from gemmaloop.deepseek_money import (
    ACCEPTED_RESPONSE_MODELS,
    API_MODEL,
    CLIENT_MAX_RETRIES,
    DEFAULT_JUDGE_MAX_USD,
    HARD_GLOBAL_CAP_USD,
    REGISTERED_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    CallAccountant,
    EndpointRefused,
    FingerprintMismatch,
    ModelMismatch,
    PeakRequestDetected,
    WasteGuard,
    WasteGuardTriggered,
    append_journal,
    enforce_flash_response,
    ensure_fingerprint,
    estimate_request_prompt_tokens,
    load_fingerprint_record,
    read_journal,
    response_content as _raw_response_content,
    response_finish_reason,
    send_with_deadline,
    shared_accountant,
    validate_endpoint,
    validate_model,
)
from gemmaloop.judge.render import RenderResult
from gemmaloop.judge.rubric import SYSTEM_PROMPT, user_text

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = API_MODEL
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
JUDGE_REASONING_EFFORTS = ("low", "high", "max")

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JUDGE_VERSION_PATH = REPO_ROOT / "docs" / "judge_version.json"
DEFAULT_GATES_PATH = REPO_ROOT / "docs" / "gates.md"
OPENCODE_PERMISSION_MARKER = "OPENCODE_WRITTEN_PERMISSION:"


class Verdict(BaseModel):
    """Schema the judge must return (``references/06`` Know-how §4)."""

    render_ok: dict[str, bool]
    criteria: dict[str, dict]
    weighted: dict[str, float]
    winner: Literal["A", "B", "tie"]
    confidence: int
    decisive_reason: str
    flags: list[str]


class JudgeVersionError(RuntimeError):
    """The provider is no longer the model registered by the vision gate."""


@dataclass
class JudgeRecord:
    """Everything needed to audit one pairwise judgment.

    ``status`` is ``render_error`` when a capture was unusable (the client was
    never called), ``invalid`` when both attempts failed to produce a valid
    JSON verdict, and ``ok`` otherwise.
    """

    request: dict | None
    response: str | None
    usage: dict | None
    system_fingerprint: str | None
    verdict: Verdict | None
    status: Literal["ok", "invalid", "render_error"]
    model: str | None = None
    response_id: str | None = None
    attempts: int = 0
    error: str | None = None
    waste_stopped: bool = False
    stop_reason: str | None = None
    attempts_log: list[dict] | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view (the verdict is expanded to its plain fields)."""
        return {
            "status": self.status,
            "request": self.request,
            "response": self.response,
            "usage": self.usage,
            "system_fingerprint": self.system_fingerprint,
            "verdict": self.verdict.model_dump() if self.verdict else None,
            "model": self.model,
            "response_id": self.response_id,
            "attempts": self.attempts,
            "error": self.error,
            "waste_stopped": self.waste_stopped,
            "stop_reason": self.stop_reason,
            "attempts_log": self.attempts_log or [],
        }


@dataclass(frozen=True)
class JudgeClientConfig:
    """Connection settings; all three values can be overridden per run."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV


_expected_version: tuple[str, str | None] | None = None


def register_judge_version(
    model: str, system_fingerprint: str | None = None
) -> None:
    """Pin the version recorded by the vision gate.

    Once registered, every :func:`judge_pair` response must carry the same
    model and (when both sides have one) the same ``system_fingerprint``;
    otherwise :class:`JudgeVersionError` aborts the run.
    """
    global _expected_version
    _expected_version = (model, system_fingerprint)


def registered_judge_version() -> tuple[str, str | None] | None:
    """Return the pinned ``(model, system_fingerprint)``, if any."""
    return _expected_version


def reset_judge_version() -> None:
    """Forget the pinned version (used by tests and by a new gate run)."""
    global _expected_version
    _expected_version = None


def save_judge_version(
    model: str,
    system_fingerprint: str | None = None,
    *,
    path: str | Path | None = None,
) -> Path:
    """Write the machine-readable judge version (used by the vision gate)."""
    target = Path(path) if path is not None else DEFAULT_JUDGE_VERSION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "system_fingerprint": system_fingerprint}
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_judge_version(
    path: str | Path | None = None,
    *,
    register: bool = True,
) -> tuple[str, str | None] | None:
    """Read and (by default) register the version saved by the vision gate.

    Returns ``None`` when no record exists yet.  Every judging entry point
    calls this through :func:`_ensure_version_loaded` before its first API
    call, so a gate run in one process pins the judge for later processes.
    """
    target = Path(path) if path is not None else DEFAULT_JUDGE_VERSION_PATH
    if not target.exists():
        return None

    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise ValueError(f"{target} does not contain a judge model")
    version = (payload["model"], payload.get("system_fingerprint"))
    if register:
        register_judge_version(*version)
    return version


def _ensure_version_loaded(fingerprint_path: str | Path | None = None) -> None:
    """Load the single artifacts fingerprint record (or the legacy mirror)."""

    record = load_fingerprint_record(fingerprint_path)
    if record is not None:
        register_judge_version(record["model"], record["fingerprint"])
        return
    if registered_judge_version() is None:
        load_judge_version()
    version = registered_judge_version()
    if version is None:
        raise JudgeVersionError(
            "judge_pair is running without a registered judge version; run "
            "scripts/judge_vision_gate.py first so the served model and "
            "system_fingerprint are pinned."
        )
    if version[1] is None:
        raise JudgeVersionError(
            "registered judge version has no system_fingerprint; re-run the "
            "vision gate so the provider can be pinned."
        )


def _has_opencode_permission(path: str | Path | None = None) -> bool:
    target = Path(path) if path is not None else DEFAULT_GATES_PATH
    if not target.exists():
        return False
    return any(
        line.startswith(OPENCODE_PERMISSION_MARKER)
        for line in target.read_text(encoding="utf-8").splitlines()
    )


def build_client(
    config: JudgeClientConfig | None = None,
    *,
    client_factory: Any | None = None,
    allow_opencode_hosted: bool = False,
    gates_path: str | Path | None = None,
):
    """Build an ``openai.OpenAI`` client for the judge.

    The API key is read from the environment variable named in ``config``;
    it is never accepted as a literal argument (so it cannot leak into logs
    or command lines).  Only the official DeepSeek endpoint is accepted: the
    OpenCode opt-in is kept as a parameter for source compatibility but is
    ignored, so no money path can bypass :func:`validate_endpoint`.

    SDK retries are disabled and a bounded timeout is set on every client, so
    every retry is ours and goes through the off-peak gate again.
    """
    config = config or JudgeClientConfig()
    try:
        validate_endpoint(config.base_url)
    except EndpointRefused as exc:
        raise ValueError(str(exc)) from exc

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set; the judge needs the official "
            "DeepSeek API key in that environment variable."
        )

    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    return client_factory(
        base_url=config.base_url,
        api_key=api_key,
        max_retries=CLIENT_MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_dict(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    asdict = getattr(value, "dict", None)
    if callable(asdict):
        return asdict()
    try:
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    except TypeError:
        return None


def _response_content(response: Any) -> str | None:
    choices = _field(response, "choices") or []
    if not choices:
        return None
    message = _field(choices[0], "message")
    content = _field(message, "content")
    return content if isinstance(content, str) else None


def parse_json_object(content: str | None) -> tuple[dict | None, str | None]:
    """Parse raw content as a JSON object; return ``(object, error)``."""
    if content is None or not content.strip():
        return None, "empty content"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "JSON response is not an object"
    return payload, None


def _parse_verdict(content: str | None) -> tuple[Verdict | None, str | None]:
    payload, error = parse_json_object(content)
    if payload is None:
        return None, error
    try:
        return Verdict.model_validate(payload), None
    except ValidationError as exc:
        return None, f"verdict schema: {exc.error_count()} error(s)"


def _validate_request_envelope(request: dict) -> None:
    """Refuse a request that is not pinned and explicitly cheap."""

    validate_model(request.get("model"))
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("judge requests must set an integer max_tokens")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive; got {max_tokens!r}")
    extra = request.get("extra_body") or {}
    thinking = extra.get("thinking") if isinstance(extra, dict) else None
    if not isinstance(thinking, dict) or thinking.get("type") not in (
        "enabled",
        "disabled",
    ):
        raise ValueError(
            "judge requests must set extra_body.thinking.type explicitly"
        )


def complete_json(
    client: Any,
    request: dict,
    *,
    accountant: CallAccountant,
    parse=parse_json_object,
    enforce_version: bool = True,
    waste_guard: WasteGuard | None = None,
    waste_journal: str | Path | None = None,
    response_log: list[Any] | None = None,
    shard: int = 0,
    metadata: dict | None = None,
    enforce_flash: bool = True,
    request_timeout_s: float = 240.0,
) -> tuple[Any, Any, str | None, int]:
    """Call the judge, retrying once on empty/invalid JSON.

    ``accountant`` is mandatory: a send without the off-peak gate and a ledger
    reservation is not allowed.  Every paid response (including an intermediate
    retry) is recorded in the run's waste journal and, when requested, in
    ``response_log``; the waste guard uses the absolute trip count so the
    judge's single retry remains possible.
    """
    if accountant is None:
        raise ValueError(
            "complete_json requires a CallAccountant; refusing an ungated, "
            "unreserved DeepSeek request"
        )
    _validate_request_envelope(request)
    response = None
    parsed = None
    error = None
    attempts = 0

    for attempts in range(1, 3):
        reservation, _estimate = accountant.before_request(
            prompt_tokens=estimate_request_prompt_tokens(request),
            max_output_tokens=int(request["max_tokens"]),
            shard=shard,
            metadata=metadata,
        )
        try:
            response = send_with_deadline(
                lambda: client.chat.completions.create(**request),
                deadline_s=request_timeout_s,
            )
        except Exception:
            accountant.settle_failure(reservation, shard=shard, metadata=metadata)
            raise
        if response_log is not None:
            response_log.append(response)
        try:
            accountant.after_response(
                reservation, response, shard=shard, metadata=metadata
            )
        except PeakRequestDetected as exc:
            exc.response = response  # type: ignore[attr-defined]
            raise

        if waste_guard is not None:
            tripped = waste_guard.record(
                response_finish_reason(response), _raw_response_content(response)
            )
            if waste_journal is not None:
                append_journal(
                    waste_journal,
                    {
                        "kind": "response",
                        "waste": bool(
                            waste_guard.is_waste(
                                response_finish_reason(response),
                                _raw_response_content(response),
                            )
                        ),
                        "finish_reason": response_finish_reason(response),
                        "response_model": _field(response, "model"),
                        "system_fingerprint": _field(response, "system_fingerprint"),
                    },
                )
        else:
            tripped = False

        if enforce_flash:
            try:
                enforce_flash_response(response)
            except ModelMismatch as exc:
                err = JudgeVersionError(str(exc))
                err.response = response  # type: ignore[attr-defined]
                raise err from exc
        if enforce_version:
            try:
                _enforce_version(response, request["model"])
            except JudgeVersionError as exc:
                exc.response = response  # type: ignore[attr-defined]
                raise
        parsed, error = parse(_response_content(response))
        if parsed is not None or tripped:
            return response, parsed, error if parsed is None else None, attempts

    return response, None, error, attempts


def _enforce_version(response: Any, requested_model: str) -> None:
    expected = registered_judge_version()
    response_model = _field(response, "model")
    fingerprint = _field(response, "system_fingerprint")

    # Only the exact served version or the exact documented alias identify the
    # registered model.  Look-alikes such as "DeepSeek-V5-Flash-0301" are not
    # accepted just because they contain "flash".
    if not isinstance(response_model, str) or response_model not in ACCEPTED_RESPONSE_MODELS:
        raise JudgeVersionError(
            f"judge model changed: requested {requested_model!r}, provider "
            f"returned {response_model!r}"
        )
    if expected is None:
        raise JudgeVersionError(
            "no registered judge version; run scripts/judge_vision_gate.py "
            "first so the served model and fingerprint are pinned"
        )

    expected_model, expected_fingerprint = expected
    if not isinstance(expected_model, str) or expected_model not in ACCEPTED_RESPONSE_MODELS:
        raise JudgeVersionError(
            f"registered judge model {expected_model!r} is not "
            f"{REGISTERED_MODEL!r} or {API_MODEL!r}"
        )
    if expected_fingerprint is None:
        raise JudgeVersionError(
            "registered judge version has no system_fingerprint; the run "
            "cannot pin the provider"
        )
    if fingerprint != expected_fingerprint:
        raise JudgeVersionError(
            f"judge system_fingerprint changed: registered "
            f"{expected_fingerprint!r}, provider returned {fingerprint!r}"
        )

def image_data_url(path: str | Path) -> str:
    """Return a ``data:image/...;base64,...`` URL for a capture file."""
    path = Path(path)
    mime = _MIME_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_error(a: RenderResult, b: RenderResult) -> str | None:
    problems = []
    for label, result in (("A", a), ("B", b)):
        if not result.ok:
            problems.append(f"{label}: {result.error or 'render failed'}")
            continue
        if result.desktop_png is None and result.mobile_png is None:
            problems.append(f"{label}: no captures")
            continue
        if result.desktop_png is None:
            problems.append(f"{label}: missing desktop capture")
            continue
        for label_size, path in (
            ("desktop", result.desktop_png),
            ("mobile", result.mobile_png),
        ):
            if path is not None and not Path(path).exists():
                problems.append(f"{label} {label_size}: capture file missing")
    if (a.mobile_png is None) != (b.mobile_png is None):
        problems.append("A/B: inconsistent capture set (one mobile missing)")
    return "; ".join(problems) if problems else None


def _image_layout(a: RenderResult, b: RenderResult) -> str:
    """One capture per candidate (SVG) or desktop+mobile for both."""
    if a.mobile_png is None and b.mobile_png is None:
        return "single"
    return "full"


def _brief_lists(brief_spec: dict) -> tuple[str, list[str], list[str], list[str]]:
    def _list(key: str) -> list[str]:
        values = brief_spec.get(key) or []
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values]

    return (
        str(brief_spec.get("brief", "")),
        _list("required_sections"),
        _list("required_copy"),
        _list("forbidden"),
    )


def _image_paths(a: RenderResult, b: RenderResult) -> list[Path]:
    """Images in prompt order: A desktop, A mobile, B desktop, B mobile.

    SVG renders only produce a desktop capture; the mobile slot is skipped.
    """
    images: list[Path] = []
    for result in (a, b):
        for path in (result.desktop_png, result.mobile_png):
            if path is not None:
                images.append(Path(path))
    return images


def _build_request(
    brief_spec: dict,
    a: RenderResult,
    b: RenderResult,
    *,
    model: str,
    max_tokens: int,
    reasoning_effort: str = "low",
) -> dict:
    brief, required_sections, required_copy, forbidden = _brief_lists(brief_spec)

    content: list[dict] = [
        {
            "type": "text",
            "text": user_text(
                brief,
                required_sections,
                required_copy,
                forbidden,
                image_layout=_image_layout(a, b),
            ),
        }
    ]
    for path in _image_paths(a, b):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(path), "detail": "original"},
            }
        )

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "json_object"},
        # Explicit thinking: never rely on the provider default (on, high).
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def _attempt_entry(response: Any) -> dict:
    choices = _field(response, "choices") or []
    return {
        "response": _response_content(response),
        "usage": _as_dict(_field(response, "usage")),
        "model": _field(response, "model"),
        "system_fingerprint": _field(response, "system_fingerprint"),
        "finish_reason": _field(choices[0], "finish_reason") if choices else None,
        "response_id": _field(response, "id"),
    }


def _record_from_response(
    response: Any,
    *,
    status: Literal["ok", "invalid"],
    request: dict,
    attempts: int,
    verdict: Verdict | None,
    error: str | None,
    attempts_log: list[Any] | None = None,
    stop_reason: str | None = None,
) -> JudgeRecord:
    return JudgeRecord(
        status=status,
        request=request,
        response=_response_content(response),
        usage=_as_dict(_field(response, "usage")),
        system_fingerprint=_field(response, "system_fingerprint"),
        verdict=verdict,
        model=_field(response, "model"),
        response_id=_field(response, "id"),
        attempts=attempts,
        error=error,
        stop_reason=stop_reason,
        attempts_log=[_attempt_entry(item) for item in (attempts_log or [])],
    )


_DEFAULT_JUDGE_WASTE_GUARD = WasteGuard()


def reset_default_judge_guard() -> WasteGuard:
    """Start a new run-level waste guard (tests and a new judging run)."""

    global _DEFAULT_JUDGE_WASTE_GUARD
    _DEFAULT_JUDGE_WASTE_GUARD = WasteGuard()
    return _DEFAULT_JUDGE_WASTE_GUARD


def default_judge_guard() -> WasteGuard:
    return _DEFAULT_JUDGE_WASTE_GUARD


def judge_waste_journal_path(out_dir: str | Path | None = None) -> Path:
    """Default persisted waste journal for a judging run."""

    if out_dir is not None:
        return Path(out_dir) / "judge_waste.jsonl"
    from gemmaloop.paths import artifacts_dir

    return artifacts_dir() / "stage0" / "judge_waste.jsonl"


def _guard_from_journal(
    path: str | Path, *, window: int = 20, rate: float = 0.10
) -> WasteGuard:
    guard = WasteGuard(window=window, rate=rate)
    flags = [
        bool(entry.get("waste"))
        for entry in read_journal(path)
        if entry.get("kind") == "response"
    ]
    guard.prime(flags[-window:])
    return guard


def judge_pair(
    brief_spec: dict,
    a: RenderResult,
    b: RenderResult,
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    reasoning_effort: str = "low",
    ledger: Any | None = None,
    max_usd: float = DEFAULT_JUDGE_MAX_USD,
    ledger_month: str | None = None,
    now_fn: Any | None = None,
    sleep_fn: Any | None = None,
    waste_guard: WasteGuard | None = None,
    waste_journal: str | Path | None = None,
    out_dir: str | Path | None = None,
    in_memory_waste_guard: bool = False,
    fingerprint_path: str | Path | None = None,
    request_timeout_s: float = 240.0,
    accountant: CallAccountant | None = None,
    shard: int = 0,
) -> JudgeRecord:
    """Judge one pair in one order and return an auditable record.

    ``a`` and ``b`` are the two candidates in the order the judge sees them.
    If either render is unusable the client is not called at all
    (``status="render_error"``).  Empty or invalid JSON/verdicts are retried
    once; a second failure yields ``status="invalid"``.  Changes in the
    provider's model or registered ``system_fingerprint`` raise
    :class:`JudgeVersionError` instead of producing a record.

    Every attempt goes through the shared monthly ledger and spend lock
    (unless the caller injects an ``accountant`` for tests).  Thinking is
    always explicit and enabled with effort ``low`` by default; callers that
    need ``high`` must ask for it.  Every paid response is counted by the
    waste guard and can be persisted in ``waste_journal``.  When the guard
    trips, the paid record is attached to the raised
    :class:`WasteGuardTriggered` exception.
    """
    model = validate_model(model)
    if not 0 < float(max_usd) <= HARD_GLOBAL_CAP_USD:
        raise ValueError(
            f"max_usd must be in (0, {HARD_GLOBAL_CAP_USD:g}]; got {max_usd!r}"
        )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError(f"max_tokens must be an integer; got {max_tokens!r}")
    if not 1 <= max_tokens <= 393216:
        raise ValueError(f"max_tokens must be in [1, 393216]; got {max_tokens!r}")
    if reasoning_effort not in JUDGE_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {JUDGE_REASONING_EFFORTS}; "
            f"got {reasoning_effort!r}"
        )

    render_problem = _render_error(a, b)
    if render_problem is not None:
        return JudgeRecord(
            status="render_error",
            request=None,
            response=None,
            usage=None,
            system_fingerprint=None,
            verdict=None,
            error=f"render failed: {render_problem}",
        )

    _ensure_version_loaded(fingerprint_path)
    if waste_guard is not None:
        guard = waste_guard
    elif waste_journal is not None:
        guard = _guard_from_journal(waste_journal)
    elif in_memory_waste_guard:
        # Explicit unit-test mode only; product runs persist the window.
        guard = _DEFAULT_JUDGE_WASTE_GUARD
    else:
        waste_journal = judge_waste_journal_path(out_dir)
        guard = _guard_from_journal(waste_journal)
    if guard.triggered:
        raise WasteGuardTriggered(
            "judge run stopped: the waste guard already tripped on a previous "
            "pair"
        )

    request = _build_request(
        brief_spec,
        a,
        b,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    if accountant is None:
        if ledger is None:
            accountant, _ledger, ledger_month = shared_accountant(
                max_usd=float(max_usd),
                now_fn=now_fn,
                sleep_fn=sleep_fn or time.sleep,
            )
        else:
            from gemmaloop.stage0.spend import require_spend_lock

            require_spend_lock()
            accountant = CallAccountant(
                ledger=ledger,
                month=ledger_month,
                max_usd=float(max_usd),
                now_fn=now_fn,
                sleep_fn=sleep_fn or time.sleep,
            )

    response_log: list[Any] = []
    try:
        response, verdict, parse_error, attempts = complete_json(
            client,
            request,
            parse=_parse_verdict,
            accountant=accountant,
            waste_guard=guard,
            waste_journal=waste_journal,
            response_log=response_log,
            shard=shard,
            metadata={"brief_id": brief_spec.get("brief_id")},
            request_timeout_s=request_timeout_s,
        )
    except (JudgeVersionError, PeakRequestDetected, FingerprintMismatch) as exc:
        response = getattr(exc, "response", None)
        if response is None and response_log:
            response = response_log[-1]
        if response is not None:
            record = _record_from_response(
                response,
                status="invalid",
                request=request,
                attempts=max(1, len(response_log)),
                verdict=None,
                error=str(exc),
                attempts_log=response_log,
                stop_reason=type(exc).__name__,
            )
            exc.record = record  # type: ignore[attr-defined]
        raise

    if verdict is not None:
        record = _record_from_response(
            response,
            status="ok",
            request=request,
            attempts=attempts,
            verdict=verdict,
            error=None,
            attempts_log=response_log,
        )
    else:
        assert response is not None  # two attempts always run
        record = _record_from_response(
            response,
            status="invalid",
            request=request,
            attempts=attempts,
            verdict=None,
            error=parse_error,
            attempts_log=response_log,
        )

    if guard.triggered:
        record.waste_stopped = True
        record.stop_reason = "waste_guard"
        exc = WasteGuardTriggered(
            f"waste guard tripped: {guard.waste_count}/{guard.total_count} "
            "responses were waste; stopping the run with the paid record kept"
        )
        exc.record = record  # type: ignore[attr-defined]
        raise exc
    return record
