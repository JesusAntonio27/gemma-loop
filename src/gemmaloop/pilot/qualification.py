"""Tolerant reader for the teacher-qualification JSON.

The qualification script is owned by another branch and its exact schema is
not frozen here.  The "why" report must never fail because a field is absent:
this module accepts the file when it exists, extracts only the fields it knows,
and returns an empty mapping otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["extract_qualification", "load_qualification"]

_ALIASES: dict[str, tuple[str, ...]] = {
    "kl_mean": ("kl_mean", "mean_kl", "full_kl_mean", "mean_full_kl", "teacher_kl_mean"),
    "kl_p95": ("kl_p95", "p95_kl", "full_kl_p95"),
    "top1": ("top1", "top1_agreement", "top1_confidence_agreement"),
    "tail_error_mean": ("tail_error_mean", "tail_mean", "tail_kl_mean"),
    "tail_error_p95": ("tail_error_p95", "tail_p95", "tail_kl_p95"),
    "memory_headroom_gib": ("memory_headroom_gib", "headroom_gib"),
    "rows": ("rows", "queries"),
    "quant": ("quant", "quantization", "weight_quant"),
    "runtime": ("runtime", "backend"),
}


def _lower_key(value: Any) -> str:
    return str(value).strip().lower()


def _teacher_key(value: Any) -> str | None:
    text = _lower_key(value)
    if "12b" in text:
        return "12b"
    if "e4b" in text:
        return "e4b"
    return None


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        out[_lower_key(key)] = value
        out[name.lower()] = value
        if isinstance(value, Mapping):
            out.update(_flatten(value, name))
    return out


def _pick(flat: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in flat and flat[alias] is not None:
            return flat[alias]
    return None


def _clean_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    flat = _flatten(entry)
    out: dict[str, Any] = {}
    for field, aliases in _ALIASES.items():
        value = _pick(flat, aliases)
        if value is None:
            continue
        if field in ("quant", "runtime") and isinstance(value, Mapping):
            value = value.get("quant") or value.get("name") or value.get("backend")
        if isinstance(value, Mapping):
            continue
        out[field] = value
    return out


def _entries(payload: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    if isinstance(payload, list):
        return [
            (_teacher_key(item.get("teacher") or item.get("name") or item.get("model_id")), item)
            for item in payload
            if isinstance(item, Mapping)
        ]
    if not isinstance(payload, Mapping):
        return []
    for container in ("teachers", "results", "qualification", "runtimes"):
        section = payload.get(container)
        if isinstance(section, Mapping) or isinstance(section, list):
            return _entries(section)
    result: list[tuple[str | None, Mapping[str, Any]]] = []
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        teacher = _teacher_key(key) or _teacher_key(
            value.get("teacher") or value.get("name") or value.get("model_id") or ""
        )
        if teacher is not None:
            result.append((teacher, value))
    return result


def extract_qualification(payload: Any) -> dict[str, dict[str, Any]]:
    """Extract per-teacher metrics without trusting any particular schema."""

    output: dict[str, dict[str, Any]] = {}
    try:
        for teacher, entry in _entries(payload):
            if teacher is None:
                continue
            cleaned = _clean_entry(entry)
            if cleaned:
                output[teacher] = cleaned
    except Exception:
        return {}
    return output


def load_qualification(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Read the qualification JSON when possible; never raise."""

    if path is None:
        return {}
    try:
        source = Path(path)
        if not source.is_file():
            return {}
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return extract_qualification(payload)
