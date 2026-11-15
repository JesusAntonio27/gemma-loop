"""Stage 0 problem-pool builder and deterministic split assignment.

The pool is about 2,000 code problems with hidden tests, 2,000 math problems
with exact answers and 1,000 chat prompts.  Every accepted problem is checked
with :class:`gemmaloop.data.decontam.EvalIndex` (13-gram overlap and normalised
Indel ``>= 0.75``) against every registered evaluation corpus and the creative
briefs, then assigned exactly once to a bucket by the registered rule::

    bucket = int(sha256(problem_id), 16) % 100

Bucket roles (prereg ``data.splits``):

=====  =========================================================
0-49   Stage 1 / Stage 2 train pool
50-59  pilot dev / selection
60-69  NLL/KL test
70-79  long traces
80-99  confirmatory held-out test
=====  =========================================================

The pilot bucket 50-59 is further split into disjoint P3 (50-53), selection
(54-56), T-fit (57-58) and P1 (59) sub-ranges.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gemmaloop.data.decontam import EvalIndex, tokenizer_name

from .constants import PART1_CREATIVE_BRIEFS
from .sources import (
    DATASET_REVISIONS,
    POOL_SOURCES,
    PoolSource,
    convert_row,
    dedup_key,
    iter_source_rows,
)

SPLIT_BUCKETS: dict[str, tuple[int, int]] = {
    "train": (0, 49),
    "pilot_dev": (50, 59),
    "nll_kl": (60, 69),
    "long_traces": (70, 79),
    "heldout_test": (80, 99),
}
PILOT_SUBRANGES: dict[str, tuple[int, int]] = {
    "pilot_p3": (50, 53),
    "pilot_selection": (54, 56),
    "pilot_tfit": (57, 58),
    "pilot_p1": (59, 59),
}
REGISTERED_COUNTS = {"code": 2000, "math": 2000, "chat": 1000}
SPLIT_RULE = "bucket = int(sha256(problem_id), 16) % 100, seed 0"


def bucket_for(problem_id: str) -> int:
    return int(hashlib.sha256(problem_id.encode("utf-8")).hexdigest(), 16) % 100


def primary_split(bucket: int) -> str:
    if not 0 <= bucket <= 99:
        raise ValueError(f"bucket must be in 0..99, got {bucket!r}")
    for name, (low, high) in SPLIT_BUCKETS.items():
        if low <= bucket <= high:
            return name
    raise AssertionError("unreachable")


def pilot_subsplit(bucket: int) -> str | None:
    if not 50 <= bucket <= 59:
        return None
    for name, (low, high) in PILOT_SUBRANGES.items():
        if low <= bucket <= high:
            return name
    raise AssertionError("unreachable")


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row))
            handle.write("\n")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _contamination(
    eval_index: EvalIndex | None, problem: dict[str, Any]
) -> tuple[str, str] | None:
    """Return ``(field, verdict)`` for the first contaminated field."""

    if eval_index is None:
        return None
    fields = [("prompt", problem.get("prompt"))]
    metadata = problem.get("metadata") or {}
    fields.append(("reference_solution", metadata.get("reference_solution")))
    fields.append(("answer", problem.get("answer")))
    for name, value in fields:
        if not value:
            continue
        verdict = eval_index.check(str(value))
        if verdict is not None:
            return name, verdict
    return None


@dataclass
class PoolBuildResult:
    problems: list[dict[str, Any]]
    manifest: dict[str, Any]
    dropped: dict[str, dict[str, int]] = field(default_factory=dict)


def build_pool(
    sources: Iterable[PoolSource] = POOL_SOURCES,
    *,
    eval_index: EvalIndex | None = None,
    eval_status: dict[str, Any] | None = None,
    quotas: dict[str, int] | None = None,
    limit_rows: int | None = None,
    shuffle_buffer: int = 0,
    seed: int = 0,
    allow_duplicates: bool = False,
    tokenizer=None,
    tokenizer_revision: str | None = None,
    allow_short_sources: bool = False,
    eval_limit: int | None = None,
    allow_missing_eval: bool = False,
    allow_missing_briefs: bool = False,
    creative_briefs: str | None = None,
) -> PoolBuildResult:
    """Build the decontaminated pool.

    ``sources`` and ``quotas`` are injectable for tests.  A source is scanned in
    its pinned stream order until its quota is filled or the stream ends; a
    short source is reported, never silently back-filled from another source.
    """

    quotas = dict(quotas or {})
    sources = list(sources)
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_counts: dict[str, int] = {}
    source_rows: dict[str, int] = {}
    source_status: dict[str, dict[str, Any]] = {}
    source_errors: list[str] = []

    for source in sources:
        quota = int(quotas.get(source.name, source.quota))
        source_rows[source.name] = 0
        source_counts[source.name] = 0
        status: dict[str, Any] = {
            "status": "ok",
            "hf_id": source.hf_id,
            "config": source.config,
            "split": source.split,
            "revision": source.revision,
            "license": source.license,
            "quota": quota,
            "kind": source.kind,
            "notes": source.notes,
        }
        try:
            rows = iter_source_rows(
                source,
                limit_rows=limit_rows,
                shuffle_buffer=shuffle_buffer,
                seed=seed,
            )
            for index, row in rows:
                source_rows[source.name] += 1
                if source_counts[source.name] >= quota:
                    break
                try:
                    problem = convert_row(source.name, row, index)
                except Exception as exc:  # bad source rows are counted, not fatal
                    dropped[source.name][f"convert_{type(exc).__name__}"] += 1
                    continue
                if problem is None:
                    dropped[source.name]["unusable_row"] += 1
                    continue
                key = dedup_key(problem)
                if not key:
                    dropped[source.name]["empty_prompt"] += 1
                    continue
                if not allow_duplicates and key in seen:
                    dropped[source.name]["duplicate_prompt"] += 1
                    continue
                contaminated = _contamination(eval_index, problem)
                if contaminated is not None:
                    field_name, verdict = contaminated
                    dropped[source.name][f"contaminated_{field_name}_{verdict}"] += 1
                    continue

                seen.add(key)
                bucket = bucket_for(problem["problem_id"])
                problem["bucket"] = bucket
                problem["split"] = primary_split(bucket)
                problem["pilot_subsplit"] = pilot_subsplit(bucket)
                accepted.append(problem)
                source_counts[source.name] += 1
        except Exception as exc:  # pragma: no cover - network dependent
            status["status"] = "failed"
            status["error"] = f"{type(exc).__name__}: {exc}"
        if status["status"] == "ok" and source_counts[source.name] < quota:
            status["status"] = "short"
            status["error"] = f"quota {quota}, accepted {source_counts[source.name]}"
        if status["status"] in {"failed", "short"}:
            source_errors.append(f"{source.name}: {status.get('error', status['status'])}")
        status["rows_scanned"] = source_rows[source.name]
        status["accepted"] = source_counts[source.name]
        source_status[source.name] = status

    if source_errors and not allow_short_sources:
        raise RuntimeError(
            "mandatory Stage 0 source(s) failed or fell short: "
            + "; ".join(source_errors)
            + ". Pass --allow-short-sources only for a smoke run."
        )

    accepted.sort(key=lambda problem: problem["problem_id"])

    counts_by_kind = Counter(problem["kind"] for problem in accepted)
    counts_by_source = Counter(problem["source"] for problem in accepted)
    counts_by_split = Counter(problem["split"] for problem in accepted)
    counts_by_subsplit = Counter(
        problem["pilot_subsplit"]
        for problem in accepted
        if problem.get("pilot_subsplit")
    )

    manifest: dict[str, Any] = {
        "stage": "stage0_pool",
        "version": 1,
        "split_rule": SPLIT_RULE,
        "seed": int(seed),
        "split_buckets": {name: list(bounds) for name, bounds in SPLIT_BUCKETS.items()},
        "pilot_subranges": {
            name: list(bounds) for name, bounds in PILOT_SUBRANGES.items()
        },
        "registered_counts": dict(REGISTERED_COUNTS),
        "counts": {
            "total": len(accepted),
            "by_kind": dict(sorted(counts_by_kind.items())),
            "by_source": dict(sorted(counts_by_source.items())),
            "by_split": dict(sorted(counts_by_split.items())),
            "by_pilot_subsplit": dict(sorted(counts_by_subsplit.items())),
        },
        "sources": source_status,
        "dataset_revisions": {
            name: dict(entry) for name, entry in sorted(DATASET_REVISIONS.items())
        },
        "dropped": {
            source: dict(sorted(criteria.items()))
            for source, criteria in sorted(dropped.items())
        },
        "decontamination": {
            "method": "13-gram overlap + normalised Indel >= 0.75",
            "ngram_n": int(getattr(eval_index, "n", 13)) if eval_index else 13,
            "texts": len(eval_index) if eval_index is not None else 0,
            "tokenizer": tokenizer_name(tokenizer) if tokenizer is not None else "word-regex",
            "tokenizer_revision": tokenizer_revision,
            "eval_index_sha256": (
                _sha256_text("\n".join(eval_index.normalized))
                if eval_index is not None
                else None
            ),
            "targets": eval_status or {},
        },
        "creative_briefs": creative_briefs or "included",
        "limits": {
            "limit_rows": limit_rows,
            "shuffle_buffer": shuffle_buffer,
            "allow_duplicates": bool(allow_duplicates),
            "allow_short_sources": bool(allow_short_sources),
            "eval_limit": eval_limit,
            "allow_missing_eval": bool(allow_missing_eval),
            "allow_missing_briefs": bool(allow_missing_briefs),
        },
        "artifacts": {},
    }
    if eval_index is None:
        manifest["decontamination"]["status"] = "not_built"
    return PoolBuildResult(problems=accepted, manifest=manifest, dropped=dict(dropped))


def split_rows(problems: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "all": list(problems),
        **{name: [] for name in SPLIT_BUCKETS},
        **{name: [] for name in PILOT_SUBRANGES},
    }
    for problem in out["all"]:
        out[problem["split"]].append(problem)
        subsplit = problem.get("pilot_subsplit")
        if subsplit:
            out[subsplit].append(problem)
    return out


def write_pool(
    result: PoolBuildResult,
    output_dir: Path | str,
    *,
    tokenizer=None,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Write ``problems.jsonl``, one file per split, and ``manifest.json``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = split_rows(result.problems)

    files: dict[str, Path] = {"problems.jsonl": output_dir / "problems.jsonl"}
    _write_jsonl(files["problems.jsonl"], rows["all"])
    for name in SPLIT_BUCKETS:
        files[f"split_{name}.jsonl"] = output_dir / f"split_{name}.jsonl"
        _write_jsonl(files[f"split_{name}.jsonl"], rows[name])
    for name in PILOT_SUBRANGES:
        files[f"split_{name}.jsonl"] = output_dir / f"split_{name}.jsonl"
        _write_jsonl(files[f"split_{name}.jsonl"], rows[name])

    manifest = dict(result.manifest)
    manifest["artifacts"] = {
        name: {"sha256": _sha256_file(path), "rows": len(rows[key])}
        for (name, path), key in zip(
            files.items(),
            ["all"] + list(SPLIT_BUCKETS) + list(PILOT_SUBRANGES),
        )
    }
    manifest["decontamination"]["tokenizer"] = (
        tokenizer_name(tokenizer) if tokenizer is not None else manifest["decontamination"]["tokenizer"]
    )
    if tokenizer_revision is not None:
        manifest["decontamination"]["tokenizer_revision"] = tokenizer_revision
    hash_payload = dict(manifest)
    hash_payload.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _sha256_text(
        json.dumps(hash_payload, ensure_ascii=False, sort_keys=True)
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "PART1_CREATIVE_BRIEFS",
    "PILOT_SUBRANGES",
    "PoolBuildResult",
    "REGISTERED_COUNTS",
    "SPLIT_BUCKETS",
    "SPLIT_RULE",
    "bucket_for",
    "build_pool",
    "pilot_subsplit",
    "primary_split",
    "split_rows",
    "write_pool",
]
