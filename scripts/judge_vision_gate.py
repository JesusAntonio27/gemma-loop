"""Minimum-spend vision gate for the DeepSeek judge.

Sends one synthetic PNG — a red square on the left, a blue circle on the
right — and requires the judge to name both colours and both positions.  A
failure means the provider cannot see (or cannot describe) images reliably,
so Part E must stop before any pair is judged.

This script is intentionally *not* run by the unit-test suite: it spends
money and needs ``DEEPSEEK_API_KEY``.  The provider, base URL and model are
arguments so a future approved provider needs no code change.  The API key is
only read from the named environment variable and is never printed or logged.

Usage (later, after the author approves the spend and the legal gate):

    uv run python scripts/judge_vision_gate.py

The gate appends ``response.model``, ``system_fingerprint``, ``finish_reason``
and the raw content to ``docs/gates.md``.  On PASS it also writes
``docs/judge_version.json``; :func:`gemmaloop.judge.client.judge_pair` loads
that file before its first call and aborts if the model, fingerprint or the
presence of the fingerprint changes.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from gemmaloop.deepseek_money import (
    DEFAULT_JUDGE_MAX_USD,
    CallAccountant,
    FingerprintMismatch,
    PeakRequestDetected,
    WasteGuard,
    fingerprint_record_path,
    validate_model,
    write_fingerprint_record,
)
from gemmaloop.stage0.spend import (
    SpendLedger,
    default_ledger_path,
    month_key,
    require_spend_lock,
)
from gemmaloop.judge.client import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_JUDGE_VERSION_PATH,
    DEFAULT_MODEL,
    JudgeClientConfig,
    JudgeVersionError,
    build_client,
    complete_json,
    parse_json_object,
    save_judge_version,
)
GATE_SYSTEM = (
    "You are a visual perception checker. Describe only what is visible in the "
    "image. Output JSON only, no markdown."
)
GATE_USER = (
    "The image contains two shapes on a white background. Name the colour and "
    "the horizontal position (left or right) of each shape. Return exactly: "
    '{"left": "<colour and shape on the left>", '
    '"right": "<colour and shape on the right>"}'
)
GATE_SECTION = "Vision judge gate"


def make_gate_image() -> Image.Image:
    """White 640x360 canvas with a red square left and a blue circle right."""
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 110, 220, 270), fill=(255, 0, 0))
    draw.ellipse((420, 110, 580, 270), fill=(0, 0, 255))
    return image


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def check_answer(payload: dict) -> tuple[bool, str]:
    """Require red under the ``left`` key and blue under the ``right`` key.

    The JSON keys are the positions the model assigned, so a colour under the
    wrong key fails just as a missing colour does.
    """
    left = str(payload.get("left", "")).lower()
    right = str(payload.get("right", "")).lower()
    problems = []
    if "red" not in left:
        problems.append("left shape is not described as red")
    if "blue" not in right:
        problems.append("right shape is not described as blue")
    if problems:
        return False, "; ".join(problems)
    return True, "both colours and both positions recognised"


def _attempt_record(response: object) -> dict:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    return {
        "response": getattr(message, "content", None),
        "model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": getattr(choice, "finish_reason", None),
    }


def _append_gate_record(
    out_path: Path,
    *,
    passed: bool,
    base_url: str,
    requested_model: str,
    response_model: str | None,
    system_fingerprint: str | None,
    finish_reason: str | None,
    raw_content: str | None,
    detail: str,
    answer: dict | None,
    attempts_log: list[dict] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        f"## {GATE_SECTION} — {timestamp}",
        "",
        f"- status: {'PASS' if passed else 'FAIL'}",
        f"- base_url: {base_url}",
        f"- requested_model: {requested_model}",
        f"- response_model: {response_model}",
        f"- system_fingerprint: {system_fingerprint}",
        f"- finish_reason: {finish_reason}",
        f"- raw_content: {json.dumps(raw_content, ensure_ascii=False)}",
        f"- detail: {detail}",
        f"- attempts: {json.dumps(attempts_log or [], ensure_ascii=False)}",
    ]
    if answer is not None:
        lines.append(f"- answer: {json.dumps(answer, ensure_ascii=False)}")
    existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    out_path.write_text(
        existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-usd", type=float, default=DEFAULT_JUDGE_MAX_USD)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="low"
    )
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--spend-lock", type=Path, default=None)
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument(
        "--repin",
        action="store_true",
        help="deliberately replace the registered fingerprint record",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/gates.md"))
    parser.add_argument(
        "--version-out", type=Path, default=DEFAULT_JUDGE_VERSION_PATH
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_model(args.model)
    require_spend_lock(args.spend_lock)
    ledger = SpendLedger(args.ledger or default_ledger_path())
    if not ledger.path.exists() and not args.init_ledger:
        raise SystemExit(
            f"refusing: spend ledger {ledger.path} is missing; pass "
            "--init-ledger for the first live gate run"
        )
    ledger_month = month_key()
    config = JudgeClientConfig(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
    )

    image = make_gate_image()
    client = build_client(config)
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": GATE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GATE_USER},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(image),
                            "detail": "original",
                        },
                    },
                ],
            },
        ],
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    accountant = CallAccountant(
        ledger=ledger,
        month=ledger_month,
        max_usd=args.max_usd,
    )
    version_error = None
    response_log: list[object] = []
    try:
        response, answer, parse_error, attempts = complete_json(
            client,
            request,
            parse=parse_json_object,
            enforce_version=False,
            enforce_flash=True,
            accountant=accountant,
            waste_guard=WasteGuard(),
            response_log=response_log,
        )
    except (JudgeVersionError, PeakRequestDetected, FingerprintMismatch) as exc:
        # Any paid response that triggers a stop is terminal: the gate records
        # the FAIL and never sends a second paid request.
        response = getattr(exc, "response", None)
        if response is None and response_log:
            response = response_log[-1]
        answer = None
        parse_error = None
        attempts = max(1, len(response_log))
        version_error = str(exc)

    response_model = getattr(response, "model", None)
    system_fingerprint = getattr(response, "system_fingerprint", None)
    choices = getattr(response, "choices", None) or []
    finish_reason = None
    raw_content = None
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        raw_content = getattr(getattr(choices[0], "message", None), "content", None)

    passed = False
    detail = ""
    if version_error is not None:
        detail = version_error
    elif not isinstance(response_model, str) or response_model not in (
        "deepseek-flash",
        "DeepSeek-V4.1-Flash-1017",
    ):
        detail = (
            f"provider returned model {response_model!r}, "
            f"requested a Flash model {args.model!r}"
        )
    elif not isinstance(system_fingerprint, str) or not system_fingerprint.strip():
        detail = "provider returned no system_fingerprint; refusing an unpinned PASS"
    elif answer is None:
        detail = parse_error or "empty JSON content"
    else:
        passed, detail = check_answer(answer)

    registered = None
    if passed:
        try:
            registered = write_fingerprint_record(
                model=response_model,
                fingerprint=system_fingerprint,
                source="vision_gate",
                repin=args.repin,
            )
        except (FingerprintMismatch, ValueError) as exc:
            passed = False
            detail = str(exc)

    _append_gate_record(
        args.out,
        passed=passed,
        base_url=args.base_url,
        requested_model=args.model,
        response_model=response_model,
        system_fingerprint=system_fingerprint,
        finish_reason=finish_reason,
        raw_content=raw_content,
        detail=detail,
        answer=answer,
        attempts_log=[_attempt_record(item) for item in response_log],
    )

    if passed:
        save_judge_version(
            response_model or args.model,
            system_fingerprint,
            path=args.version_out,
        )

    print(f"vision gate: {'PASS' if passed else 'FAIL'} - {detail}")
    print(f"attempts: {attempts}")
    if response_model is not None:
        print(f"model: {response_model}")
    if system_fingerprint is not None:
        print(f"system_fingerprint: {system_fingerprint}")
    print(f"recorded in {args.out}")
    if passed:
        print(f"judge version pinned in {args.version_out}")
        if registered is not None:
            print(f"fingerprint record: {fingerprint_record_path()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
