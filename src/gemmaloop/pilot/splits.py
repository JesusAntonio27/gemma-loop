"""Fail-closed disjointness checks for the pilot splits.

The fit split, evaluation traces, training rows and the monitor split must
never share a row.  Identity is the rendered ``input_ids`` hash, an explicit
``input_ids_sha256`` field, or a canonical hash of the raw JSON row, plus any
problem/cluster id the row carries.  Numeric row indices are not identities.

``reserved_ids_payload`` writes the manifest the heal monitor reads.  Its split
lists hold *problem ids only* so the monitor corpus can be compared exactly
against the run's ``--monitor-rows`` problem-id set; content hashes are kept
under ``content_hashes`` as extra metadata for audits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SplitOverlapError",
    "assert_disjoint_splits",
    "reserved_ids_payload",
    "row_identities",
    "split_content_hashes",
    "split_problem_ids",
    "split_row_ids",
]

# The ruled manifest names for the pilot's split paths.
_SPLIT_NAMES = {
    "heal": "heal_train",
    "heal_train": "heal_train",
    "fit": "pilot_fit",
    "pilot_fit": "pilot_fit",
    "evaluation": "pilot_eval",
    "eval": "pilot_eval",
    "test": "pilot_eval",
    "pilot_eval": "pilot_eval",
    "monitor": "monitor",
    "confirmatory": "confirmatory",
}
_TRAINING_SPLITS = {"heal_train"}
_MONITOR_SPLIT = "monitor"


class SplitOverlapError(ValueError):
    """Two pilot splits contain the same row or problem."""


def _canonical_row(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_identity(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("input_ids_sha256")
    if explicit:
        return f"sha256:{explicit}"
    input_ids = payload.get("input_ids")
    if isinstance(input_ids, list):
        joined = ",".join(str(int(value)) for value in input_ids)
        return "ids:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return "row:" + hashlib.sha256(_canonical_row(payload).encode("utf-8")).hexdigest()


def _problem_ids(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("cluster_id", "problem_id", "problem", "task_id"):
        value = payload.get(key)
        if value is not None:
            values.append(str(value))
    return values


def _source_path(path: str | Path) -> Path:
    """The JSONL file behind a path: a file, ``train.jsonl`` or the monitor file."""

    source = Path(path)
    if source.is_dir():
        for name in ("train.jsonl", "monitor_rows.jsonl", "monitor.jsonl"):
            candidate = source / name
            if candidate.is_file():
                return candidate
    return source


def _iter_payloads(path: str | Path):
    source = _source_path(path)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON row in {source}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"row in {source} is not an object")
            yield payload


def row_identities(path: str | Path) -> set[str]:
    """Return every identity token a split file carries."""

    source = _source_path(path)
    identities: set[str] = set()
    for payload in _iter_payloads(source):
        identities.add(_content_identity(payload))
        for key in ("cluster_id", "problem_id", "problem", "task_id"):
            value = payload.get(key)
            if value is not None:
                identities.add(f"problem:{key}:{value}")
    if not identities:
        raise ValueError(f"no usable rows in {source}")
    return identities


def assert_disjoint_splits(paths: Mapping[str, str | Path | None]) -> None:
    """Refuse any overlap among the named non-empty splits."""

    loaded = {
        name: row_identities(path)
        for name, path in paths.items()
        if path is not None
    }
    names = list(loaded)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = loaded[left] & loaded[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:3])
                raise SplitOverlapError(
                    f"{left} and {right} overlap on {len(overlap)} identity "
                    f"token(s), e.g. {sample}"
                )


def split_row_ids(path: str | Path) -> list[str]:
    """Every raw row/problem id and content hash a split file exposes."""

    source = _source_path(path)
    values: set[str] = set()
    for payload in _iter_payloads(source):
        if payload.get("row_id") is not None:
            values.add(str(payload["row_id"]))
        values.update(_problem_ids(payload))
        values.add(_content_identity(payload).split(":", 1)[-1])
    return sorted(values)


def split_problem_ids(path: str | Path) -> list[str]:
    """The explicit problem/cluster ids of a split (no row indices or hashes)."""

    values: set[str] = set()
    for payload in _iter_payloads(path):
        values.update(_problem_ids(payload))
    return sorted(values)


def split_content_hashes(path: str | Path) -> list[str]:
    """The rendered-row content identities of a split (audit metadata only)."""

    return sorted({_content_identity(payload) for payload in _iter_payloads(path)})


def _ruled_name(name: str) -> str:
    key = str(name).strip().lower()
    return _SPLIT_NAMES.get(key, key)


def reserved_ids_payload(
    paths: Mapping[str, str | Path | None],
) -> dict[str, Any]:
    """A monitor-readable manifest of every pilot split's problem ids.

    The returned object is a ruled split map (``monitor``, ``pilot_eval``,
    ``pilot_fit``, ``heal_train``, ``confirmatory``) plus:

    * ``splits``: the same map, kept for older readers;
    * ``content_hashes``: per-split rendered-row hashes (metadata only);
    * ``reserved_ids``: the union of every non-monitor, non-training split, so
      the monitor and heal rows never become their own reserved ids.
    """

    splits: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        ruled = _ruled_name(name)
        splits[ruled] = split_problem_ids(path)
        hashes[ruled] = split_content_hashes(path)
    reserved = sorted(
        {
            value
            for name, values in splits.items()
            if name != _MONITOR_SPLIT and name not in _TRAINING_SPLITS
            for value in values
        }
    )
    return {
        **splits,
        "splits": splits,
        "content_hashes": hashes,
        "reserved_ids": reserved,
    }
