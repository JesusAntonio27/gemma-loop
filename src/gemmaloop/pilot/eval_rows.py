"""Render Stage 0 verified traces into evaluation rows (CPU, deterministic).

The pilot fit/selection splits are ``trace_record`` rows: ``messages`` plus a
``problem_id`` and no ``input_ids``.  Evaluation needs rendered ids, the
assistant-label mask and positions.  This module renders them with exactly the
tokenisation training uses (:func:`gemmaloop.train.labels.assistant_labels`)
and caches the result with a manifest of the source hashes, the tokenizer
identity/revision and the output hashes.

The renderer is deterministic: the same trace, tokenizer and ``max_length``
produce byte-identical rows, so a cached directory is reused until a source
hash, the tokenizer revision or the max length changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemmaloop.model.snapshot import model_revision
from gemmaloop.train.checkpoint import atomic_write_json, atomic_write_text
from gemmaloop.train.labels import assistant_labels

__all__ = [
    "DEFAULT_EVAL_SPLITS",
    "RENDERER_VERSION",
    "render_stage0_eval_rows",
    "render_trace_rows",
]

RENDERER_VERSION = "stage0-eval-rows-v1"
DEFAULT_EVAL_SPLITS = ("pilot_tfit", "pilot_selection", "pilot_p1")
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
_CHANNEL_CLOSE = "<channel|>"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _special_id(tokenizer: Any, token: str) -> int | None:
    try:
        value = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # noqa: BLE001 - tokenizer without the token
        return None
    if value is None or value == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(value)


def _answer_start(
    input_ids: Sequence[int], labels_start: int, tokenizer: Any
) -> int | None:
    """First token after the thinking channel (``None`` when not present)."""

    close = _special_id(tokenizer, _CHANNEL_CLOSE)
    if close is None:
        return None
    for position in range(int(labels_start), len(input_ids)):
        if int(input_ids[position]) == close:
            return position + 1
    return None


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield payload


def render_trace_rows(
    traces: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    source: str | Path = "<traces>",
    tokenizer_revision: str | None = None,
    require_problem_id: bool = True,
) -> list[dict[str, Any]]:
    """Render trace rows with the training tokenisation, preserving problem ids."""

    if int(max_length) <= 0:
        raise ValueError("max_length must be positive")
    revision = tokenizer_revision or model_revision("e2b")
    rendered: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        messages = trace.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{source}: row {index} has no messages")
        problem_id = (
            trace.get("problem_id")
            or trace.get("cluster_id")
            or trace.get("problem")
        )
        if problem_id is None and require_problem_id:
            raise ValueError(
                f"{source}: row {index} has no problem_id; a registered pilot "
                "evaluation split must carry problem_id (P1-lite rows need it "
                "for the reserved split map and the problem-cluster bootstrap)"
            )
        output = assistant_labels(
            tokenizer,
            list(messages),
            int(max_length),
            tools=trace.get("tools"),
        )
        labels = [int(value) for value in output["labels"]]
        positions = [position for position, value in enumerate(labels) if value != -100]
        if not positions:
            raise ValueError(f"{source}: row {index} has no assistant tokens")
        labels_start = int(positions[0])
        identifier = None if problem_id is None else str(problem_id)
        rendered.append(
            {
                "problem_id": identifier,
                "cluster_id": identifier,
                "row_id": index,
                "source": trace.get("source"),
                "kind": trace.get("kind"),
                "split": trace.get("split"),
                "sample_index": trace.get("sample_index"),
                "input_ids": [int(value) for value in output["input_ids"]],
                "labels": labels,
                "label_positions": positions,
                "labels_start": labels_start,
                "thinking_start": labels_start,
                "answer_start": _answer_start(
                    output["input_ids"], labels_start, tokenizer
                ),
                "max_length": int(max_length),
                "tokenizer_revision": revision,
                "renderer_version": RENDERER_VERSION,
            }
        )
    if not rendered:
        raise ValueError(f"{source}: no rows to render")
    return rendered


def _cache_payload(
    stage0: Path,
    out: Path,
    splits: Sequence[str],
    *,
    max_length: int,
    tokenizer: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for name in splits:
        source = stage0 / "pilot" / f"{name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Stage 0 split not found: {source}")
        sources[name] = {
            "source": str(source),
            "source_sha256": _sha256_file(source),
            "output": f"{name}.jsonl",
        }
    return {
        "kind": "pilot_eval_rows",
        "version": RENDERER_VERSION,
        "max_length": int(max_length),
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "splits": sources,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _cache_valid(manifest_path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key in (
        "kind",
        "version",
        "max_length",
        "tokenizer",
        "tokenizer_revision",
    ):
        if cached.get(key) != payload.get(key):
            return False
    out = manifest_path.parent
    for name, entry in (cached.get("splits") or {}).items():
        if entry.get("output") != payload["splits"][name]["output"]:
            return False
        if entry.get("source_sha256") != payload["splits"][name]["source_sha256"]:
            return False
        output = out / str(entry.get("output"))
        if not output.is_file():
            return False
        if entry.get("output_sha256") != _sha256_file(output):
            return False
    return True


def render_stage0_eval_rows(
    stage0: str | Path,
    out: str | Path,
    *,
    tokenizer: Any,
    max_length: int,
    splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    tokenizer_revision: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Render (or reuse) the Stage 0 pilot splits as evaluation JSONL rows."""

    stage0 = Path(stage0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    revision = tokenizer_revision or model_revision("e2b")
    manifest_path = out / "manifest.json"
    payload = _cache_payload(
        stage0,
        out,
        list(splits),
        max_length=int(max_length),
        tokenizer=tokenizer_name,
        tokenizer_revision=revision,
    )
    if not force and _cache_valid(manifest_path, payload):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    for name in splits:
        source = Path(payload["splits"][name]["source"])
        rendered = render_trace_rows(
            list(_iter_jsonl(source)),
            tokenizer,
            max_length=int(max_length),
            source=source,
            tokenizer_revision=revision,
        )
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rendered)
        target = out / payload["splits"][name]["output"]
        atomic_write_text(target, text)
        payload["splits"][name]["rows"] = len(rendered)
        payload["splits"][name]["output_sha256"] = _sha256_bytes(
            text.encode("utf-8")
        )
    atomic_write_json(manifest_path, payload)
    return payload
