"""Build the P1-lite held-out corpus from verified original-model traces.

The P1-lite sweep needs a small, clean set of math problems whose answers the
**original** Gemma 4 E2B can produce correctly in one thinking sample.  A
training-free loop can look better than the original on rows the original
cannot answer at all, so the campaign uses only verified-correct traces.

Pipeline:

1. stream DAPO-Math-17k (a public RL-math source that no registered benchmark
   uses) and keep problems with an exact integer answer;
2. reject every problem whose raw id or canonical DAPO alias appears in a
   registered protected set (pilot selection/T-fit, MONITOR or confirmatory);
   train-pool rows are allowed.  Then reject every problem that the
   repository's :class:`EvalIndex` matches against **all** registered
   evaluation sets (exact / 13-gram / Indel >= 0.75);
3. sample the original checkpoint once per candidate with the registered eval
   sampling (thinking on, T=1.0, top_p=0.95, top_k=64) and
   ``max_new_tokens=8192``;
4. verify the final ``\\boxed{}`` answer with the AIME normalisation;
5. keep the first ``--target-verified`` correct traces and write them as JSONL
   (rendered prompt + full response + exact token ids/label offset) with a
   manifest carrying the source revision, ids, hashes and decontamination
   status.

With ``--from-stage0-local DIR`` the corpus is built without a GPU from the
committed Stage 0 shards (O's own samples), joining the pool by ``problem_id``,
keeping train-pool rows, excluding the protected sets and the run's MONITOR
ids, verifying math answers, and stratifying by length exactly as section 7
requires.  With ``--extend-from DIR`` every existing row that passes the Stage
0 disjointness filter is kept byte-for-byte and new candidates are sampled in
the registered deterministic order until ``--target-verified`` disjoint rows
exist.  ``--dry-run`` prints either plan without loading a model.

The run needs the GPU.  Candidate filtering, verification and manifest writing
are pure functions and are covered by CPU tests with a fake generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemmaloop.eval.benchmarks.aime import (
    _normalise_integer,
    last_boxed,
    strip_thinking,
)
from gemmaloop.gpulock import require_gpu_lock
from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path
from gemmaloop.paths import artifacts_dir
from gemmaloop.serve.generate import Sampling
from gemmaloop.teacher.qualification import (
    canonical_dapo_id,
    default_protected_sources,
    extract_row_ids,
    filter_disjoint,
    load_protected_ids,
    select_development_rows,
)
from gemmaloop.train.perf import cap_cuda_allocator

DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
SPLIT = "train"
LICENSE = "Apache-2.0"
URL = f"https://huggingface.co/datasets/{DATASET_ID}"

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
INTEGER_RE = re.compile(r"[+-]?\d+")
BOXED_PROMPT = "\n\nPut the final answer alone inside \\boxed{}."

DEFAULT_TARGET_VERIFIED = 140
DEFAULT_CANDIDATE_LIMIT = 360
DEFAULT_MAX_NEW_TOKENS = 8192
FULL_TRACE_FORMAT_VERSION = "p1lite-full-trace-v1"
STAGE0_FILTER_VERSION = "p1lite-stage0-protected-v2"
STAGE0_LOCAL_KIND = "p1lite_stage0_local"
DEFAULT_STAGE0_ROOT = artifacts_dir() / "stage0"
DEFAULT_STAGE0_POOL = DEFAULT_STAGE0_ROOT / "pool"
DEFAULT_STAGE0_LOCAL = DEFAULT_STAGE0_ROOT / "local"
DEFAULT_LOCAL_CORPUS_DIR = artifacts_dir() / "data" / "p1lite-stage0-local"
MEASURED_REQUEST_SECONDS = 26.0
DEFAULT_SCREEN_ROWS = 20
DEFAULT_EXPAND_ROWS = 100
LOCAL_MAX_LENGTH = 8192
LOCAL_MIN_REASONING_SHARE = 0.75
SCREEN_QUOTAS = {"short": 4, "common": 8, "medium": 4, "long": 4}


class CorpusBuildError(RuntimeError):
    """The registered corpus cannot be built from the available inputs."""


def default_corpus_heartbeat_path() -> Path:
    """The heartbeat path the detached report polling commands read."""
    return default_heartbeat_path("corpus")


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
def prompt_text(row: dict[str, Any]) -> str:
    """Return the user prompt content from a DAPO row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and str(message.get("content") or "").strip():
                return str(message["content"])
    if isinstance(prompt, str):
        return prompt
    return str(row.get("question") or row.get("problem") or "")


def extract_problem(text: str) -> str:
    """Strip DAPO's generic instruction wrapper, leaving the problem."""
    value = (text or "").strip()
    if value.startswith(DAPO_PREFIX):
        value = value[len(DAPO_PREFIX) :]
    if value.endswith(DAPO_SUFFIX):
        value = value[: -len(DAPO_SUFFIX)]
    return value.strip()


def gold_answer(row: dict[str, Any]) -> str | None:
    """Return the canonical integer gold answer, or ``None``."""
    reward = row.get("reward_model")
    raw = None
    if isinstance(reward, dict):
        raw = reward.get("ground_truth")
    if raw is None:
        raw = row.get("answer")
    if raw is None:
        return None
    value = _normalise_integer(str(raw))
    return str(int(value)) if INTEGER_RE.fullmatch(value) else None


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a candidate from a DAPO row, or ``None`` when it is not usable."""
    problem = extract_problem(prompt_text(row))
    answer = gold_answer(row)
    if not problem or answer is None:
        return None

    extra = row.get("extra_info")
    raw_id = None
    if isinstance(extra, dict):
        raw_id = extra.get("index") or extra.get("id")
    if raw_id in (None, ""):
        raw_id = None
    item_id = (
        str(raw_id)
        if raw_id is not None
        else "dapo:" + hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
    )
    return {
        "id": item_id,
        "raw_id": str(raw_id) if raw_id is not None else None,
        "problem": problem,
        "answer": answer,
        "prompt": problem + BOXED_PROMPT,
    }


def candidate_aliases(candidate: Mapping[str, Any]) -> set[str]:
    """Return the raw id and the Stage 0 canonical DAPO alias of a candidate."""

    aliases = {str(candidate["id"])}
    raw_id = candidate.get("raw_id")
    if raw_id not in (None, ""):
        aliases.add(canonical_dapo_id(raw_id))
    return aliases


def select_candidates(
    rows: Iterable[dict[str, Any]],
    limit: int,
    *,
    eval_index: Any = None,
    protected_ids: set[str] | None = None,
    skip_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return up to ``limit`` clean candidates and simple skip counters.

    Candidates are rejected before sampling when their raw id or canonical
    DAPO alias appears in a registered protected set (pilot selection/T-fit,
    MONITOR or confirmatory).  Train-pool rows are allowed.  ``skip_ids``
    additionally drops ids already retained in or rejected from an extended
    corpus.
    """

    chosen: list[dict[str, Any]] = []
    stats = {
        "rows": 0,
        "non_numeric": 0,
        "contaminated": 0,
        "protected_overlap": 0,
        "already_used": 0,
        "selected": 0,
    }
    seen: set[str] = set()
    protected_ids = protected_ids or set()
    skip_ids = skip_ids or set()
    for row in rows:
        stats["rows"] += 1
        candidate = candidate_from_row(row)
        if candidate is None:
            stats["non_numeric"] += 1
            continue
        if candidate["id"] in seen:
            continue
        seen.add(candidate["id"])
        aliases = candidate_aliases(candidate)
        if aliases & skip_ids:
            stats["already_used"] += 1
            continue
        if aliases & protected_ids:
            stats["protected_overlap"] += 1
            continue
        if eval_index is not None:
            if eval_index.check(candidate["problem"]) or eval_index.check(
                candidate["prompt"]
            ):
                stats["contaminated"] += 1
                continue
        chosen.append(candidate)
        if len(chosen) >= limit:
            break
    stats["selected"] = len(chosen)
    return chosen, stats


def verify_completion(expected: str, completion: str) -> bool:
    """Compare the last non-thinking ``\\boxed{}`` answer with ``expected``."""
    boxed = last_boxed(strip_thinking(completion or ""))
    if boxed is None:
        return False
    predicted = _normalise_integer(boxed)
    return INTEGER_RE.fullmatch(predicted) is not None and str(
        int(predicted)
    ) == str(int(expected))


# ---------------------------------------------------------------------------
# Decontamination against every registered eval set
# ---------------------------------------------------------------------------
def build_eval_index(
    tokenizer: Any = None, *, allow_missing: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Build the repository EvalIndex over all registered benchmark texts."""
    try:
        from scripts.prepare_heal_data import gather_eval_texts
    except ImportError:  # pragma: no cover - direct ``python scripts/...``
        from prepare_heal_data import gather_eval_texts

    from gemmaloop.data.decontam import EvalIndex

    texts, statuses = gather_eval_texts(
        eval_limit=None,
        extra_eval_files=[],
        allow_missing=allow_missing,
    )
    if not texts:
        raise RuntimeError("empty evaluation index; refusing to build a corpus")
    index = EvalIndex.build(texts, tokenizer=tokenizer)
    return index, {"texts": len(index), "benchmarks": statuses}


# ---------------------------------------------------------------------------
# Stage 0 pool disjointness and corpus extension
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl_by_newline(path: Path) -> list[dict[str, Any]]:
    """Read JSONL by ``\\n`` only.

    ``str.splitlines`` also splits on U+2028/U+2029, which are legal inside
    JSON strings and occur in real Stage 0 problems; splitting the file that
    way corrupts valid rows.
    """

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: expected a JSON object per line")
        rows.append(row)
    return rows


def load_pool_manifest(pool_dir: str | Path) -> dict[str, Any]:
    """Read the frozen Stage 0 pool manifest and record both hashes.

    The stored ``manifest_sha256`` is the value the paid pipeline freezes.  It
    is recorded verbatim rather than recomputed from the file bytes, because
    post-processing rewrites the manifest without changing the frozen value.
    """

    pool = Path(pool_dir)
    manifest_path = pool / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 0 pool manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = str(manifest.get("manifest_sha256") or "")
    if not manifest_sha:
        raise ValueError(
            f"Stage 0 pool manifest has no manifest_sha256: {manifest_path}"
        )
    return {
        "pool_dir": str(pool),
        "pool_manifest_file": str(manifest_path),
        "pool_manifest_file_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "pool_manifest_sha256": manifest_sha,
    }


def load_pool_index(
    pool_dir: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Join ``problems.jsonl`` with every split file by ``problem_id``."""

    pool = Path(pool_dir)
    info = load_pool_manifest(pool)
    problems_path = pool / "problems.jsonl"
    if not problems_path.is_file():
        raise FileNotFoundError(f"Stage 0 problems.jsonl not found: {problems_path}")
    rows = _read_jsonl_by_newline(problems_path)
    problems: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = str(row.get("problem_id") or "")
        if not problem_id:
            raise ValueError(f"{problems_path}: row without problem_id")
        problems[problem_id] = dict(row)
    files = {
        "problems.jsonl": {
            "path": str(problems_path),
            "rows": len(rows),
            "sha256": _sha256_bytes(problems_path.read_bytes()),
        }
    }
    for path in sorted(pool.glob("split_*.jsonl")):
        split_rows = _read_jsonl_by_newline(path)
        for row in split_rows:
            problem_id = str(row.get("problem_id") or "")
            if not problem_id:
                raise ValueError(f"{path}: row without problem_id")
            merged = dict(problems.get(problem_id, {}))
            merged.update(row)
            merged.setdefault("split", path.stem.removeprefix("split_"))
            problems[problem_id] = merged
        files[path.name] = {
            "path": str(path),
            "rows": len(split_rows),
            "sha256": _sha256_bytes(path.read_bytes()),
        }
    info["files"] = files
    info["problems"] = len(problems)
    return problems, info


def _protected_set_info(
    sources: Mapping[str, Sequence[Path]],
    report: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sets: dict[str, Any] = {}
    for name, paths in sources.items():
        files = []
        for path in paths:
            candidate = Path(path)
            if candidate.is_file():
                files.append(
                    {
                        "path": str(candidate),
                        "sha256": _sha256_bytes(candidate.read_bytes()),
                    }
                )
        entry = dict(report.get(name, {}))
        entry["files"] = files
        sets[str(name)] = entry
    return sets


def load_registered_protected(
    root: str | Path,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load exactly the registered protected sets under the Stage 0 root."""

    sources = default_protected_sources(root)
    report, ids_by_source = load_protected_ids(sources)
    info = {
        "filter_version": STAGE0_FILTER_VERSION,
        "protected_root": str(Path(root)),
        "sets": _protected_set_info(sources, report),
    }
    return ids_by_source, report, info


def disjointness_info(
    protected_root: str | Path, pool_dir: str | Path
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    """The manifest record shared by the extension and Stage 0 source modes."""

    ids_by_source, report, protected = load_registered_protected(protected_root)
    pool = load_pool_manifest(pool_dir)
    record = {
        "filter_version": STAGE0_FILTER_VERSION,
        "protected_root": str(Path(protected_root)),
        "protected_sets": protected["sets"],
        **pool,
    }
    return ids_by_source, report, record


def split_protected(
    aliases: Sequence[Sequence[str] | set[str]],
    ids_by_source: Mapping[str, Sequence[str] | set[str]],
    report: Mapping[str, Mapping[str, Any]],
) -> tuple[list[int], dict[str, Any], list[dict[str, Any]]]:
    """Reuse :func:`filter_disjoint` and attach a per-row removal reason."""

    kept, exclusion = filter_disjoint(
        aliases, ids_by_source, protected_report=report
    )
    kept_set = set(kept)
    removed: list[dict[str, Any]] = []
    for index, values in enumerate(aliases):
        if index in kept_set:
            continue
        aliases_set = {str(value) for value in values if str(value)}
        hits = sorted(
            name
            for name, ids in ids_by_source.items()
            if aliases_set & {str(value) for value in ids}
        )
        removed.append(
            {
                "index": index,
                "reason": (
                    "protected_overlap:" + ",".join(hits)
                    if hits
                    else "missing_id"
                ),
                "aliases": sorted(aliases_set)[:6],
            }
        )
    return kept, exclusion, removed


def read_source_corpus(source_dir: str | Path) -> dict[str, Any]:
    """Read an existing corpus and its manifest without rewriting its rows."""

    directory = Path(source_dir)
    corpus_path = directory / "corpus.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"existing corpus not found: {corpus_path}")
    raw = corpus_path.read_bytes()
    text = raw.decode("utf-8")
    records: list[tuple[str, dict[str, Any]]] = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{corpus_path}: expected a JSON object per row")
        records.append((line, row))
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    return {
        "dir": str(directory),
        "path": str(corpus_path),
        "manifest_path": str(manifest_path),
        "sha256": _sha256_bytes(raw),
        "records": records,
        "manifest": manifest,
    }


def filter_retained_rows(
    records: Sequence[tuple[str, Mapping[str, Any]]],
    ids_by_source: Mapping[str, Sequence[str] | set[str]],
    report: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Keep byte-identical rows that are disjoint; record every removal reason."""

    aliases = [extract_row_ids(row) for _, row in records]
    kept, _, removed = split_protected(aliases, ids_by_source, report)
    kept_lines: list[str] = []
    kept_ids: list[str] = []
    seen: set[str] = set()
    kept_set = set(kept)
    for index in kept:
        line, row = records[index]
        row_id = str(row.get("id") or row.get("row_id") or "")
        if row_id in seen:
            removed.append(
                {
                    "index": index,
                    "id": row_id,
                    "reason": "duplicate_id",
                    "aliases": sorted(aliases[index])[:6],
                }
            )
            continue
        seen.add(row_id)
        kept_lines.append(line.rstrip("\r\n") + "\n")
        kept_ids.append(row_id)
    for entry in removed:
        index = int(entry.get("index", 0))
        line, row = records[index]
        entry["id"] = str(row.get("id") or row.get("row_id") or "")
    removed.sort(key=lambda entry: int(entry.get("index", 0)))
    return kept_lines, kept_ids, removed


def _raw_sample_ids(source_dir: str | Path) -> set[str]:
    path = Path(source_dir) / "samples.jsonl"
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for record in _read_jsonl(path):
        value = record.get("id")
        if value not in (None, ""):
            ids.add(str(value))
    return ids


def length_stratum(length: int) -> str:
    """The section-7 length label used for the dry-run counts."""

    value = int(length)
    if value <= 1000:
        return "short"
    if value <= 2000:
        return "common"
    if value <= 4000:
        return "medium"
    return "long"


def build_dry_run_plan(
    source_dir: str | Path,
    pool_dir: str | Path,
    *,
    protected_root: str | Path | None = None,
    monitor_local_dir: str | Path | None = None,
    require_monitor_manifest: bool = False,
    target_verified: int,
    candidate_limit: int,
    seconds_per_request: float = MEASURED_REQUEST_SECONDS,
) -> dict[str, Any]:
    """Describe an extension without touching a model or the GPU."""

    root = Path(protected_root) if protected_root is not None else Path(pool_dir).parent
    if monitor_local_dir is None:
        candidate = Path(pool_dir).parent / "local"
        if (candidate / "run_marker.json").is_file():
            monitor_local_dir = candidate
    ids_by_source, report, disjointness = disjointness_info(root, pool_dir)
    monitor_ids, monitor_info = monitor_for_run(
        monitor_local_dir,
        pool_dir,
        disjointness,
        ids_by_source,
        require_manifest=require_monitor_manifest,
    )
    merged_ids = {name: set(values) for name, values in ids_by_source.items()}
    merged_ids["monitor"] = monitor_ids
    merged_report = {name: dict(entry) for name, entry in report.items()}
    merged_report["monitor"] = {
        **merged_report.get("monitor", {}),
        **monitor_info,
    }
    protected_sets = _protected_set_info(
        default_protected_sources(root), merged_report
    )
    protected_sets["monitor"]["files"] = [
        *protected_sets["monitor"].get("files", []),
        *monitor_info.get("files", []),
    ]
    disjointness = {
        **disjointness,
        "protected_sets": protected_sets,
        "monitor": monitor_info,
    }
    source = read_source_corpus(source_dir)
    _, kept_ids, removed = filter_retained_rows(
        source["records"], merged_ids, merged_report
    )
    by_stratum: dict[str, int] = {
        "short": 0,
        "common": 0,
        "medium": 0,
        "long": 0,
    }
    for _, row in source["records"]:
        if "input_ids" in row:
            by_stratum[length_stratum(len(row["input_ids"]))] += 1
    rows_kept = len(kept_ids)
    rows_needed = max(0, int(target_verified) - rows_kept)
    upper_bound_s = (
        float(candidate_limit) * float(seconds_per_request)
        if rows_needed > 0
        else 0.0
    )
    return {
        "mode": "extend",
        "source_corpus_dir": source["dir"],
        "source_corpus_file": source["path"],
        "source_corpus_sha256": source["sha256"],
        "rows_in_source": len(source["records"]),
        "rows_kept": rows_kept,
        "rows_removed": len(removed),
        "target_verified": int(target_verified),
        "rows_needed": rows_needed,
        "candidate_limit": int(candidate_limit),
        "seconds_per_request": float(seconds_per_request),
        "gpu_upper_bound_s": upper_bound_s,
        "eligible_by_stratum": by_stratum,
        "stage0_disjointness": disjointness,
        "monitor": monitor_info,
        "removed": removed,
    }


def format_dry_run_plan(plan: Mapping[str, Any]) -> str:
    """Render either dry-run plan as a short, human-readable block."""

    stage0 = plan["stage0_disjointness"]
    lines = [
        "P1-lite corpus dry-run plan (no model, no GPU)",
        f"  mode: {plan['mode']}",
        f"  Stage 0 pool: {stage0['pool_dir']}",
        f"  Stage 0 manifest sha256: {stage0['pool_manifest_sha256']}",
        f"  filter version: {stage0['filter_version']}",
        f"  protected sets: {', '.join(sorted(stage0['protected_sets']))}",
        f"  MONITOR status: {plan.get('monitor', {}).get('status', 'not_targeted')}",
        f"  MONITOR excluded ids: {plan.get('monitor', {}).get('ids', 0)}",
    ]
    if plan["mode"] == "stage0_local":
        lines += [
            f"  Stage 0 local samples: {plan['local_dir']}",
            f"  run_id: {plan['run_id']}",
            f"  committed shards: {plan['committed_shards']}",
            f"  eligible rows: {plan['eligible_rows']}",
            "  eligible rows by length stratum:",
            *[
                f"    {name}: {plan['eligible_by_stratum'][name]}"
                for name in ("short", "common", "medium", "long")
            ],
            f"  rows selected (screen {plan['screen_rows']}, "
            f"expansion {plan['expand_rows']}): {plan['rows_selected']}",
            "  selected rows by length stratum:",
            *[
                f"    {name}: {plan['selected_by_stratum'][name]}"
                for name in ("short", "common", "medium", "long")
            ],
        ]
        if plan.get("screen_shortfall"):
            lines.append(
                "  SECTION-7 SCREEN SHORTFALL (build exits 2): "
                + ", ".join(
                    f"{name} missing {missing}"
                    for name, missing in plan["screen_shortfall"].items()
                )
            )
    else:
        hours = float(plan["gpu_upper_bound_s"]) / 3600.0
        lines += [
            f"  source corpus: {plan['source_corpus_file']}",
            f"  source sha256: {plan['source_corpus_sha256']}",
            f"  rows in source: {plan['rows_in_source']}",
            f"  kept rows: {plan['rows_kept']}",
            f"  removed rows: {plan['rows_removed']}",
            f"  target verified rows: {plan['target_verified']}",
            f"  rows still needed: {plan['rows_needed']}",
            f"  candidate limit: {plan['candidate_limit']}",
            (
                "  upper bound GPU sampling time: "
                f"{hours:.1f} h "
                f"({plan['candidate_limit']} requests x "
                f"{plan['seconds_per_request']:.0f} s)"
            ),
        ]
    return "\n".join(lines)


def load_committed_shards(
    local_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Read only ``shard_*.jsonl`` files with a ``.committed`` marker."""

    local = Path(local_dir)
    marker_path = local / "run_marker.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Stage 0 run marker not found: {marker_path}")
    run_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    run_id = str(run_marker.get("run_id") or "")
    pool_sha = str(run_marker.get("pool_manifest_sha256") or "")
    shards_dir = local / "shards"
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard in sorted(shards_dir.glob("shard_*.jsonl")):
        committed = shard.with_name(shard.name + ".committed")
        if not committed.is_file():
            continue
        marker = json.loads(committed.read_text(encoding="utf-8"))
        if marker.get("committed") is not True:
            continue
        rows = _read_jsonl_by_newline(shard)
        for row in rows:
            if run_id and row.get("run_id") not in (None, run_id):
                raise ValueError(
                    f"{shard}: record run_id {row.get('run_id')!r} != {run_id!r}"
                )
            if pool_sha and row.get("pool_manifest_sha256") not in (None, pool_sha):
                raise ValueError(
                    f"{shard}: record pool manifest sha differs from the run marker"
                )
        records.extend(rows)
        shards.append(
            {
                "index": int(marker.get("shard", len(shards))),
                "path": str(shard),
                "sha256": _sha256_bytes(shard.read_bytes()),
                "marker_path": str(committed),
                "marker_sha256": _sha256_bytes(committed.read_bytes()),
                "rows": len(rows),
            }
        )
    if not shards:
        raise FileNotFoundError(f"no committed Stage 0 shards under {shards_dir}")
    return records, run_marker, shards


def _run_targets(run_marker: Mapping[str, Any]) -> list[Any]:
    from gemmaloop.stage0.local import Target

    targets = []
    for name, entry in sorted((run_marker.get("targets") or {}).items()):
        buckets = list(entry.get("buckets") or [0, -1])
        targets.append(
            Target(
                str(name),
                int(buckets[0]),
                int(buckets[1]),
                int(entry.get("problems") or 0),
                token_target=int(entry.get("token_target") or 0),
                verified=bool(entry.get("verified", True)),
                rows_target=int(entry.get("rows_target") or 0),
            )
        )
    return targets


def validate_run_pool(
    run_marker: Mapping[str, Any], pool_info: Mapping[str, Any]
) -> str:
    """Bind the committed shards to the pool passed on the command line."""

    run_sha = str(run_marker.get("pool_manifest_sha256") or "")
    if not run_sha:
        raise CorpusBuildError(
            "Stage 0 run marker has no pool_manifest_sha256; refusing to join"
        )
    pool_sha = str(pool_info.get("pool_manifest_sha256") or "")
    if not pool_sha:
        raise CorpusBuildError(
            "Stage 0 pool manifest has no manifest_sha256; refusing to join"
        )
    if run_sha != pool_sha:
        raise CorpusBuildError(
            "Stage 0 run was sampled from a different pool manifest: "
            f"run {run_sha}, --stage0-pool {pool_sha}"
        )
    return run_sha


def resolve_monitor_ids(
    local_dir: str | Path | None,
    pool_dir: str | Path,
    problems: Mapping[str, Mapping[str, Any]],
    pool_info: Mapping[str, Any],
    protected_ids: Mapping[str, Sequence[str] | set[str]],
    *,
    require_manifest: bool = True,
) -> tuple[set[str], dict[str, Any]]:
    """Registered monitor ids, the run's derived ids and the monitor files."""

    ids = {str(value) for value in protected_ids.get("monitor", set())}
    info: dict[str, Any] = {
        "status": "not_targeted",
        "target": None,
        "derived_ids": 0,
        "ids": 0,
        "files": [],
    }
    if local_dir is None:
        info["ids"] = len(ids)
        return ids, info
    local = Path(local_dir)
    marker_path = local / "run_marker.json"
    if not marker_path.is_file():
        info["status"] = "run-marker-missing"
        info["ids"] = len(ids)
        return ids, info
    run_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    validate_run_pool(run_marker, pool_info)
    targets = _run_targets(run_marker)
    monitor_target = next(
        (target for target in targets if target.name == "monitor"), None
    )
    if monitor_target is None:
        info["status"] = "not_targeted"
        info["ids"] = len(ids)
        return ids, info
    from gemmaloop.stage0.local import monitor_problem_ids

    selection = run_marker.get("selection_order") or {}
    seed = int(selection.get("seed") or 0)
    derived = monitor_problem_ids(
        list(problems.values()), targets, seed=seed
    )
    ids |= derived
    info.update(
        {
            "target": {
                "buckets": [int(monitor_target.low), int(monitor_target.high)],
                "problems": int(monitor_target.problems),
                "seed": seed,
                "rule": (
                    "the train-bucket problems with the largest "
                    "sha256(seed:problem_id) keys"
                ),
            },
            "derived_ids": len(derived),
            "ids": len(ids),
        }
    )
    monitor_dir = local / "monitor"
    manifest_path = monitor_dir / "manifest.json"
    rows_path = monitor_dir / "monitor_rows.jsonl"
    manifest_ids: set[str] = set()
    files: list[dict[str, Any]] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ids = {str(value) for value in payload.get("problem_ids", [])}
        ids |= manifest_ids
        files.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256_bytes(manifest_path.read_bytes()),
            }
        )
    if rows_path.is_file():
        for row in _read_jsonl_by_newline(rows_path):
            ids |= extract_row_ids(row)
        files.append(
            {
                "path": str(rows_path),
                "sha256": _sha256_bytes(rows_path.read_bytes()),
                "rows": len(_read_jsonl_by_newline(rows_path)),
            }
        )
    outside = manifest_ids - derived
    if outside:
        raise CorpusBuildError(
            "the local MONITOR manifest has ids outside the registered "
            f"largest-order-key set: {sorted(outside)[:3]}"
        )
    info["manifest_ids"] = len(manifest_ids)
    info["manifest_is_subset"] = bool(manifest_ids) and manifest_ids < derived
    if not manifest_path.is_file():
        if require_manifest:
            raise CorpusBuildError(
                "the Stage 0 run declares a MONITOR target but "
                f"{manifest_path} does not exist; finish the run before "
                "building the qualification corpus"
            )
        info["status"] = "missing"
    else:
        info["status"] = "present"
    info["files"] = files
    info["ids"] = len(ids)
    return ids, info


def monitor_for_run(
    monitor_local_dir: str | Path | None,
    pool_dir: str | Path,
    pool_info: Mapping[str, Any],
    ids_by_source: Mapping[str, Sequence[str] | set[str]],
    *,
    require_manifest: bool = True,
) -> tuple[set[str], dict[str, Any]]:
    """Resolve the run's MONITOR ids for the extension mode."""

    if monitor_local_dir is None:
        ids = {str(value) for value in ids_by_source.get("monitor", set())}
        return ids, {
            "status": "not_targeted",
            "target": None,
            "derived_ids": 0,
            "ids": len(ids),
            "files": [],
        }
    marker_path = Path(monitor_local_dir) / "run_marker.json"
    if not marker_path.is_file():
        ids = {str(value) for value in ids_by_source.get("monitor", set())}
        return ids, {
            "status": "run-marker-missing",
            "target": None,
            "derived_ids": 0,
            "ids": len(ids),
            "files": [],
        }
    problems, _ = load_pool_index(pool_dir)
    return resolve_monitor_ids(
        monitor_local_dir,
        pool_dir,
        problems,
        pool_info,
        ids_by_source,
        require_manifest=require_manifest,
    )


def screen_shortfall(
    eligible_by_stratum: Mapping[str, int], screen_rows: int
) -> dict[str, int]:
    """The per-stratum shortage against the registered section-7 quotas."""

    shortfall: dict[str, int] = {}
    total = sum(int(value) for value in eligible_by_stratum.values())
    if total < int(screen_rows):
        shortfall["total"] = int(screen_rows) - total
    if int(screen_rows) >= 20:
        scale = max(1, (int(screen_rows) + 19) // 20)
        for name, quota in SCREEN_QUOTAS.items():
            needed = int(quota) * scale
            present = int(eligible_by_stratum.get(name, 0))
            if present < needed:
                shortfall[name] = needed - present
    return shortfall


def _count_token_fn(tokenizer: Any) -> Callable[[str], int] | None:
    if tokenizer is None:
        return None

    def count(text: str) -> int:
        if not text:
            return 0
        encode = getattr(tokenizer, "encode", None)
        if encode is not None:
            return len(encode(text, add_special_tokens=False))
        output = tokenizer(text, add_special_tokens=False)
        ids = getattr(output, "input_ids", output)
        return len(ids)

    return count


def collect_stage0_candidates(
    local_dir: str | Path,
    pool_dir: str | Path,
    *,
    protected_root: str | Path,
    tokenizer: Any = None,
    screen_rows: int = DEFAULT_SCREEN_ROWS,
    expand_rows: int = DEFAULT_EXPAND_ROWS,
    seed: int = 0,
    math_grade: Callable[..., Any] | None = None,
    max_length: int = LOCAL_MAX_LENGTH,
    min_reasoning_share: float = LOCAL_MIN_REASONING_SHARE,
    require_monitor_manifest: bool = True,
) -> dict[str, Any]:
    """Select complete, verified, disjoint train-pool O samples."""

    from gemmaloop.stage0.grader import grade_math
    from gemmaloop.stage0.local import request_order_key
    from gemmaloop.stage0.postprocess import keep_reason

    records, run_marker, shards = load_committed_shards(local_dir)
    problems, pool_info = load_pool_index(pool_dir)
    validate_run_pool(run_marker, pool_info)
    ids_by_source, report, disjointness = disjointness_info(
        protected_root, pool_dir
    )
    monitor_ids, monitor_info = resolve_monitor_ids(
        local_dir,
        pool_dir,
        problems,
        pool_info,
        ids_by_source,
        require_manifest=require_monitor_manifest,
    )
    merged_ids = {name: set(values) for name, values in ids_by_source.items()}
    merged_ids["monitor"] = monitor_ids
    merged_report = {name: dict(entry) for name, entry in report.items()}
    merged_report["monitor"] = {
        **merged_report.get("monitor", {}),
        **monitor_info,
        "rows": merged_report.get("monitor", {}).get("rows", 0),
    }
    protected_sets = _protected_set_info(
        default_protected_sources(protected_root), merged_report
    )
    protected_sets["monitor"]["files"] = [
        *protected_sets["monitor"].get("files", []),
        *monitor_info.get("files", []),
    ]
    disjointness = {
        **disjointness,
        "protected_sets": protected_sets,
        "monitor": monitor_info,
    }

    known_pids = sorted(
        {
            str(record.get("problem_id"))
            for record in records
            if str(record.get("problem_id") or "") in problems
        }
    )
    if not known_pids:
        raise ValueError("no committed Stage 0 sample joins the Stage 0 pool")
    aliases = [extract_row_ids(problems[pid]) for pid in known_pids]
    kept, exclusion, _ = split_protected(aliases, merged_ids, merged_report)
    allowed = {known_pids[index] for index in kept}

    counter = _count_token_fn(tokenizer)
    verifier = math_grade or grade_math
    ordered = sorted(
        records,
        key=lambda record: (
            request_order_key(seed, str(record.get("request_id") or "")),
            str(record.get("request_id") or ""),
        ),
    )
    dropped: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    for record in ordered:
        problem_id = str(record.get("problem_id") or "")
        problem = problems.get(problem_id)
        if problem is None:
            dropped["missing_problem"] += 1
            continue
        if problem_id not in allowed:
            dropped["protected_overlap"] += 1
            continue
        if str(problem.get("split") or "") != "train":
            dropped["not_train"] += 1
            continue
        if not str(record.get("reasoning") or "").strip():
            dropped["empty_reasoning"] += 1
            continue
        reason = keep_reason(
            record,
            problem,
            token_counter=counter,
            max_length=int(max_length),
            min_reasoning_share=float(min_reasoning_share),
        )
        if reason is not None:
            dropped[str(reason)] += 1
            continue
        grade = None
        if str(problem.get("kind") or "") == "math":
            answer = str(problem.get("answer") or "")
            if not answer:
                dropped["missing_answer"] += 1
                continue
            result = verifier(
                str(record.get("completion") or record.get("content") or ""),
                answer,
            )
            if not bool(getattr(result, "correct", False)):
                dropped["math_unverified"] += 1
                continue
            grade = result.to_dict() if hasattr(result, "to_dict") else None
        if problem_id in seen_problems:
            continue
        seen_problems.add(problem_id)
        length = int(record.get("prompt_tokens") or 0) + int(
            record.get("tokens") or 0
        )
        candidates.append(
            {
                "record": record,
                "problem": problem,
                "request_id": str(record.get("request_id") or ""),
                "length": length,
                "grade": grade,
            }
        )

    eligible_by_stratum = {
        "short": 0,
        "common": 0,
        "medium": 0,
        "long": 0,
    }
    for candidate in candidates:
        eligible_by_stratum[length_stratum(candidate["length"])] += 1
    shortfall = screen_shortfall(eligible_by_stratum, int(screen_rows))
    selected: list[tuple[int, str]] = []
    if candidates:
        selection = select_development_rows(
            [candidate["length"] for candidate in candidates],
            screen_rows=min(int(screen_rows), len(candidates)),
            expand_rows=min(int(expand_rows), len(candidates)),
            seed=int(seed),
        )
        selected = list(selection["expansion"])
    selected_by_stratum = {
        "short": 0,
        "common": 0,
        "medium": 0,
        "long": 0,
    }
    for _, label in selected:
        selected_by_stratum[label] += 1
    screen = list(selected[: int(screen_rows)])
    return {
        "mode": "stage0_local",
        "local_dir": str(Path(local_dir)),
        "run_id": str(run_marker.get("run_id") or ""),
        "run_marker": run_marker,
        "shards": shards,
        "committed_shards": len(shards),
        "pool": pool_info,
        "disjointness": disjointness,
        "records": len(records),
        "problems": len(problems),
        "eligible_rows": len(candidates),
        "eligible_by_stratum": eligible_by_stratum,
        "screen_shortfall": shortfall,
        "rows_selected": len(selected),
        "selected_by_stratum": selected_by_stratum,
        "screen_rows": len(screen),
        "expand_rows": int(expand_rows),
        "selected": [
            {**candidates[index], "stratum": label} for index, label in selected
        ],
        "selected_ids": [
            candidates[index]["request_id"] for index, _ in selected
        ],
        "screen_ids": [
            candidates[index]["request_id"] for index, _ in screen
        ],
        "dropped": dict(sorted(dropped.items())),
        "exclusion": exclusion,
        "tokenizer": tokenizer,
        "seed": int(seed),
    }


def stage0_local_dry_run_plan(
    local_dir: str | Path,
    pool_dir: str | Path,
    *,
    protected_root: str | Path | None = None,
    tokenizer: Any = None,
    screen_rows: int = DEFAULT_SCREEN_ROWS,
    expand_rows: int = DEFAULT_EXPAND_ROWS,
    seed: int = 0,
    math_grade: Callable[..., Any] | None = None,
    require_monitor_manifest: bool = False,
) -> dict[str, Any]:
    """Count eligible rows per length stratum without a model or the GPU."""

    root = Path(protected_root) if protected_root is not None else Path(pool_dir).parent
    result = collect_stage0_candidates(
        local_dir,
        pool_dir,
        protected_root=root,
        tokenizer=tokenizer,
        screen_rows=screen_rows,
        expand_rows=expand_rows,
        seed=seed,
        math_grade=math_grade,
        require_monitor_manifest=require_monitor_manifest,
    )
    result.pop("tokenizer", None)
    result.pop("selected", None)
    result["stage0_disjointness"] = result.pop("disjointness")
    result["monitor"] = result["stage0_disjointness"]["monitor"]
    return result


def _stage0_local_row(
    candidate: Mapping[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    from gemmaloop.stage0.postprocess import sample_messages
    from gemmaloop.train.labels import assistant_labels

    record = candidate["record"]
    problem = candidate["problem"]
    if not str(record.get("reasoning") or "").strip():
        raise CorpusBuildError(
            f"selected local row {candidate['request_id']} has empty reasoning "
            "although Stage 0 sampling used thinking on"
        )
    output = assistant_labels(
        tokenizer,
        sample_messages(dict(record), dict(problem)),
        LOCAL_MAX_LENGTH,
        enable_thinking=True,
    )
    input_ids = [int(value) for value in output["input_ids"]]
    labels = [int(value) for value in output["labels"]]
    first_label = next(
        (index for index, value in enumerate(labels) if value != -100),
        len(input_ids),
    )
    completion = str(record.get("completion") or "")
    return {
        "id": str(candidate["request_id"]),
        "request_id": str(candidate["request_id"]),
        "problem_id": str(problem.get("problem_id")),
        "source": str(problem.get("source") or ""),
        "split": str(problem.get("split") or ""),
        "kind": str(problem.get("kind") or ""),
        "bucket": problem.get("bucket"),
        "sample_index": record.get("sample_index"),
        "stratum": candidate.get("stratum"),
        "answer": problem.get("answer"),
        "answer_grade": candidate.get("grade"),
        "prompt": str(problem.get("prompt") or ""),
        "response": completion,
        "completion": completion,
        "reasoning": str(record.get("reasoning") or ""),
        "input_ids": input_ids,
        "labels": labels,
        "labels_start": first_label,
        "length": len(input_ids),
    }


def build_stage0_local_corpus(
    local_dir: str | Path,
    pool_dir: str | Path,
    *,
    protected_root: str | Path | None = None,
    out_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    screen_rows: int = DEFAULT_SCREEN_ROWS,
    expand_rows: int = DEFAULT_EXPAND_ROWS,
    seed: int = 0,
    math_grade: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build the qualification corpus from committed Stage 0 local samples."""

    if tokenizer is None:
        raise ValueError("the Stage 0 local corpus needs a tokenizer")
    root = Path(protected_root) if protected_root is not None else Path(pool_dir).parent
    collected = collect_stage0_candidates(
        local_dir,
        pool_dir,
        protected_root=root,
        tokenizer=tokenizer,
        screen_rows=screen_rows,
        expand_rows=expand_rows,
        seed=seed,
        math_grade=math_grade,
    )
    if collected["screen_shortfall"]:
        detail = ", ".join(
            f"{name} missing {missing}"
            for name, missing in collected["screen_shortfall"].items()
        )
        raise CorpusBuildError(
            "the section-7 screen cannot be filled: " + detail
        )
    selected = list(collected["selected"])
    if not selected:
        raise CorpusBuildError("no eligible Stage 0 local samples for qualification")
    rows = [
        _stage0_local_row(candidate, tokenizer=tokenizer)
        for candidate in selected
    ]
    corpus_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    _atomic_write_text(out_path, corpus_text)
    template = getattr(tokenizer, "chat_template", None)
    manifest = {
        "kind": STAGE0_LOCAL_KIND,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_mode": "stage0_local",
        "run_id": collected["run_id"],
        "run_marker": collected["run_marker"],
        "run_marker_sha256": _sha256_bytes(
            (Path(local_dir) / "run_marker.json").read_bytes()
        ),
        "shards": collected["shards"],
        "stage0_pool": collected["pool"],
        "stage0_pool_manifest_sha256": collected["pool"]["pool_manifest_sha256"],
        "filter_version": STAGE0_FILTER_VERSION,
        "stage0_disjointness": collected["disjointness"],
        "protected_sets": collected["disjointness"]["protected_sets"],
        "monitor": collected["disjointness"]["monitor"],
        "selection": {
            "seed": int(seed),
            "order_rule": (
                "sha256(seed:request_id) ascending over committed shards"
            ),
            "screen_rows": int(collected["screen_rows"]),
            "expand_rows": int(expand_rows),
            "eligible_rows": int(collected["eligible_rows"]),
            "eligible_by_stratum": collected["eligible_by_stratum"],
            "screen_quotas": dict(SCREEN_QUOTAS),
            "screen_shortfall": collected["screen_shortfall"],
            "selected_by_stratum": collected["selected_by_stratum"],
            "selected_ids": collected["selected_ids"],
            "screen_ids": collected["screen_ids"],
            "dropped": collected["dropped"],
            "excluded_protected": collected["exclusion"]["excluded_rows"],
        },
        "n_verified": len(rows),
        "ids": [row["id"] for row in rows],
        "problem_ids": [row["problem_id"] for row in rows],
        "rows": [
            {
                "id": row["id"],
                "problem_id": row["problem_id"],
                "length": row["length"],
                "stratum": row["stratum"],
                "labels_start": row["labels_start"],
            }
            for row in rows
        ],
        "corpus_file": str(out_path),
        "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "tokenizer": (
            getattr(tokenizer, "name_or_path", None)
            or getattr(tokenizer, "model_id", None)
            or type(tokenizer).__name__
        ),
        "chat_template_sha256": (
            hashlib.sha256(str(template).encode("utf-8")).hexdigest()
            if template
            else None
        ),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def extend_corpus(
    source_dir: str | Path,
    pool_dir: str | Path,
    candidate_rows: Iterable[dict[str, Any]],
    *,
    generate_fn: Callable,
    target_verified: int,
    out_path: Path,
    manifest_path: Path,
    raw_path: Path,
    candidate_limit: int,
    protected_root: str | Path | None = None,
    monitor_local_dir: str | Path | None = None,
    require_monitor_manifest: bool = True,
    eval_index: Any = None,
    batch: int = 8,
    model: Any = None,
    processor: Any = None,
    tokenizer: Any = None,
    sampling: Sampling | None = None,
    heartbeat: Heartbeat | None = None,
    extra_manifest: dict[str, Any] | None = None,
    raw_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep disjoint existing rows and add verified candidates up to the target."""

    root = Path(protected_root) if protected_root is not None else Path(pool_dir).parent
    if monitor_local_dir is None:
        candidate = Path(pool_dir).parent / "local"
        if (candidate / "run_marker.json").is_file():
            monitor_local_dir = candidate
    ids_by_source, report, stage0_info = disjointness_info(root, pool_dir)
    monitor_ids, monitor_info = monitor_for_run(
        monitor_local_dir,
        pool_dir,
        stage0_info,
        ids_by_source,
        require_manifest=require_monitor_manifest,
    )
    merged_ids = {name: set(values) for name, values in ids_by_source.items()}
    merged_ids["monitor"] = monitor_ids
    merged_report = {name: dict(entry) for name, entry in report.items()}
    merged_report["monitor"] = {
        **merged_report.get("monitor", {}),
        **monitor_info,
    }
    protected_sets = _protected_set_info(
        default_protected_sources(root), merged_report
    )
    protected_sets["monitor"]["files"] = [
        *protected_sets["monitor"].get("files", []),
        *monitor_info.get("files", []),
    ]
    stage0_info = {
        **stage0_info,
        "protected_sets": protected_sets,
        "monitor": monitor_info,
    }
    protected_ids = set()
    for values in merged_ids.values():
        protected_ids |= {str(value) for value in values}
    source = read_source_corpus(source_dir)
    kept_lines, kept_ids, removed = filter_retained_rows(
        source["records"], merged_ids, merged_report
    )
    skip_ids = set(kept_ids)
    skip_ids.update(
        str(row.get("id")) for _, row in source["records"] if row.get("id")
    )
    skip_ids.update(_raw_sample_ids(source_dir))

    candidates, selection_stats = select_candidates(
        candidate_rows,
        candidate_limit,
        eval_index=eval_index,
        protected_ids=protected_ids,
        skip_ids=skip_ids,
    )
    extension = {
        "source_corpus_dir": source["dir"],
        "source_corpus_file": source["path"],
        "source_corpus_sha256": source["sha256"],
        "source_manifest_sha256": (
            source["manifest"].get("corpus_sha256")
            if isinstance(source["manifest"], dict)
            else None
        ),
        "rows_in_source": len(source["records"]),
        "kept_ids": kept_ids,
        "removed": removed,
        "removed_ids": [entry["id"] for entry in removed],
    }
    merged = {
        "filter_version": STAGE0_FILTER_VERSION,
        "stage0_disjointness": stage0_info,
        "protected_sets": stage0_info["protected_sets"],
        "monitor": stage0_info["monitor"],
        "stage0_pool_manifest_sha256": stage0_info["pool_manifest_sha256"],
        "source_corpus_sha256": source["sha256"],
        "kept_ids": kept_ids,
        "removed_ids": extension["removed_ids"],
        "removed": removed,
        "extension": extension,
        "selection": selection_stats,
        **(extra_manifest or {}),
    }
    return build_corpus(
        candidates,
        generate_fn=generate_fn,
        target_verified=int(target_verified),
        out_path=out_path,
        manifest_path=manifest_path,
        raw_path=raw_path,
        retained_lines=kept_lines,
        retained_ids=kept_ids,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        sampling=sampling,
        batch=batch,
        extra_manifest=merged,
        raw_fingerprint=raw_fingerprint,
        heartbeat=heartbeat,
    )


# ---------------------------------------------------------------------------
# Sampling / writing
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _default_render(processor: Any, prompt: str) -> str:
    if processor is None:
        return prompt
    tokenizer = getattr(processor, "tokenizer", processor)
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        return prompt
    return template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def render_completed_turn(
    tokenizer: Any,
    user_prompt: str,
    reasoning: str,
    content: str,
) -> tuple[str, str]:
    """Render prompt + the exact completed assistant turn.

    ``GenOut.text`` excludes the thinking channel; the template puts it back
    only when the assistant message carries ``reasoning``.  Rendering the full
    conversation gives the same string a completed turn would have, so the
    corpus stores the reasoning tokens Stage A must score.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        raise TypeError("tokenizer has no apply_chat_template")
    user = [{"role": "user", "content": user_prompt}]
    prompt = template(
        user,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        assistant["reasoning"] = reasoning
    full = template(
        [*user, assistant],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "completed assistant turn does not round-trip through the chat "
            "template (prompt is not a prefix)"
        )
    return prompt, full[len(prompt) :]


def _token_ids(tokenizer: Any, text: str) -> list[int] | None:
    if tokenizer is None:
        return None
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def _corpus_row(
    candidate: dict[str, Any],
    completion: str,
    reasoning: str,
    *,
    processor: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        rendered, response = render_completed_turn(
            tokenizer, candidate["prompt"], reasoning, completion
        )
    else:
        rendered = _default_render(processor, candidate["prompt"])
        thought = (
            f"<|channel>thought\n{reasoning}<channel|>" if reasoning else ""
        )
        response = thought + completion
    prompt_ids = _token_ids(tokenizer, rendered)
    row: dict[str, Any] = {
        "id": candidate["id"],
        # Registered pilot evaluation rows must expose a problem id for the
        # reserved split map and the problem-cluster bootstrap.
        "problem_id": str(candidate["id"]),
        "cluster_id": str(candidate["id"]),
        "source": DATASET_ID,
        "source_revision": REVISION,
        "answer": candidate["answer"],
        "prompt": rendered,
        "response": response,
        "completion": response,
        "reasoning": reasoning,
    }
    if prompt_ids is not None:
        full_ids = _token_ids(tokenizer, rendered + response) or []
        if full_ids[: len(prompt_ids)] == prompt_ids:
            row["input_ids"] = full_ids
        else:  # pragma: no cover - BPE boundary fallback
            row["input_ids"] = prompt_ids + (
                _token_ids(tokenizer, response) or []
            )
        row["labels_start"] = len(prompt_ids)
    return row


def build_corpus(
    candidates: Sequence[dict[str, Any]],
    *,
    generate_fn: Callable,
    target_verified: int,
    out_path: Path,
    manifest_path: Path,
    raw_path: Path | None = None,
    retained_lines: Sequence[str] = (),
    retained_ids: Sequence[str] = (),
    model: Any = None,
    processor: Any = None,
    tokenizer: Any = None,
    sampling: Sampling | None = None,
    batch: int = 8,
    extra_manifest: dict[str, Any] | None = None,
    raw_fingerprint: dict[str, Any] | None = None,
    heartbeat: Heartbeat | None = None,
) -> dict[str, Any]:
    """Sample once per candidate, verify, and write verified-correct traces.

    ``retained_lines`` are byte-exact rows kept from an existing corpus; they
    count toward ``target_verified`` and are written before any new row.
    """

    if target_verified < 1:
        raise ValueError("target_verified must be positive")
    if batch < 1:
        raise ValueError("batch must be positive")
    sampling = sampling or Sampling(max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
    raw_path = raw_path or out_path.with_suffix(".samples.jsonl")
    raw_meta_path = raw_path.with_suffix(raw_path.suffix + ".meta.json")
    fingerprint = {
        "format_version": FULL_TRACE_FORMAT_VERSION,
        **dict(
            raw_fingerprint
            or {
                "source_revision": REVISION,
                "sampling": asdict(sampling),
            }
        ),
    }
    new_target = max(0, int(target_verified) - len(retained_lines))
    raw_records: dict[str, dict[str, Any]] = {}
    if new_target > 0:
        raw_has_data = raw_path.exists() and raw_path.stat().st_size > 0
        if raw_has_data and not raw_meta_path.exists():
            raise RuntimeError(
                f"{raw_path} has no {raw_meta_path.name} declaring the "
                f"{FULL_TRACE_FORMAT_VERSION} full-trace format; it may be an "
                "answer-only pre-fix sample file. Move it aside and regenerate "
                "the P1-lite corpus."
            )
        if raw_meta_path.exists():
            previous = json.loads(raw_meta_path.read_text(encoding="utf-8"))
            if previous.get("format_version") != FULL_TRACE_FORMAT_VERSION:
                raise RuntimeError(
                    f"{raw_meta_path.name} does not declare "
                    f"{FULL_TRACE_FORMAT_VERSION}; move it aside and regenerate "
                    "the P1-lite corpus."
                )
            if previous != fingerprint:
                raise RuntimeError(
                    "raw P1-lite samples were produced with a different model, "
                    "sampling, chat template or source revision; move them aside "
                    "or pass a new --raw-samples path"
                )
        else:
            _atomic_write_text(
                raw_meta_path,
                json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
            )

        raw_records = {str(row["id"]): row for row in _read_jsonl(raw_path)}
        if sampling.enable_thinking:
            missing_reasoning = [
                str(record.get("id"))
                for record in raw_records.values()
                if not str(record.get("reasoning") or "").strip()
            ]
            if missing_reasoning:
                raise RuntimeError(
                    "reused raw P1-lite samples lack the required reasoning field "
                    f"(for example {missing_reasoning[:3]}); move {raw_path} aside "
                    "and regenerate with full traces."
                )
    started = time.time()
    considered = 0
    for start in range(0, len(candidates), batch):
        chunk = list(candidates[start : start + batch])
        pending = [c for c in chunk if c["id"] not in raw_records]
        if pending:
            conversations = [
                [{"role": "user", "content": candidate["prompt"]}]
                for candidate in pending
            ]
            outputs = generate_fn(
                model,
                processor,
                conversations,
                sampling,
                seed=0,
                image_soft_tokens=280,
                request_ids=[candidate["id"] for candidate in pending],
            )
            if len(outputs) != len(pending):
                raise RuntimeError(
                    f"generator returned {len(outputs)} outputs for "
                    f"{len(pending)} candidates"
                )
            new_records = []
            for candidate, output in zip(pending, outputs):
                record = {
                    "id": candidate["id"],
                    "completion": output.text,
                    "reasoning": getattr(output, "reasoning", "") or "",
                    "finish_reason": str(output.finish_reason),
                    "completion_tokens": int(output.completion_tokens),
                    "correct": verify_completion(candidate["answer"], output.text),
                }
                raw_records[candidate["id"]] = record
                new_records.append(record)
            _append_jsonl(raw_path, new_records)
            if heartbeat is not None:
                heartbeat.update(
                    items_done=len(raw_records),
                    items_total=len(candidates),
                    tokens=sum(
                        int(record.get("completion_tokens", 0))
                        for record in new_records
                    ),
                    batch=len(new_records),
                    phase="sampling",
                    extra={"target_verified": int(target_verified)},
                )
        considered = min(len(candidates), start + len(chunk))
        verified = sum(
            1
            for candidate in candidates
            if raw_records.get(candidate["id"], {}).get("correct")
        )
        if len(retained_lines) + verified >= target_verified:
            break

    selected = [
        candidate
        for candidate in candidates
        if raw_records.get(candidate["id"], {}).get("correct")
    ][:new_target]
    if len(selected) < new_target:
        raise RuntimeError(
            f"only {len(selected)} verified-correct traces after "
            f"{considered} candidates; raise --candidate-limit"
        )

    if heartbeat is not None:
        heartbeat.update(
            items_done=len(retained_lines) + len(selected),
            items_total=int(target_verified),
            phase="verification",
            extra={"verified": len(retained_lines) + len(selected)},
        )
    rows = []
    for candidate in selected:
        record = raw_records[candidate["id"]]
        rows.append(
            _corpus_row(
                candidate,
                str(record["completion"]),
                str(record.get("reasoning") or ""),
                processor=processor,
                tokenizer=tokenizer,
            )
        )

    all_rows = [*[json.loads(line) for line in retained_lines], *rows]
    corpus_text = "".join(
        [*retained_lines]
        + [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    )
    _atomic_write_text(out_path, corpus_text)
    digest = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    token_lengths = [
        len(row["input_ids"]) for row in all_rows if "input_ids" in row
    ]
    response_lengths = [
        len(row["input_ids"]) - int(row["labels_start"])
        for row in all_rows
        if "input_ids" in row
    ]

    def _distribution(values: list[int]) -> dict[str, Any] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": ordered[0],
            "mean": round(statistics.fmean(ordered), 1),
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": ordered[-1],
        }

    template = getattr(tokenizer, "chat_template", None) if tokenizer else None
    manifest: dict[str, Any] = {
        "kind": "p1lite_corpus",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": DATASET_ID,
        "source_revision": REVISION,
        "source_split": SPLIT,
        "source_url": URL,
        "license": LICENSE,
        "target_verified": int(target_verified),
        "n_verified": len(all_rows),
        "n_kept": len(retained_lines),
        "n_new_verified": len(rows),
        "n_candidates": len(candidates),
        "n_considered": considered,
        "n_sampled": len(raw_records),
        "ids": [
            *[str(value) for value in retained_ids],
            *[row["id"] for row in rows],
        ],
        "kept_ids": [str(value) for value in retained_ids],
        "new_ids": [row["id"] for row in rows],
        "corpus_file": str(out_path),
        "corpus_sha256": digest,
        "raw_samples_file": str(raw_path),
        "raw_samples_sha256": hashlib.sha256(
            raw_path.read_bytes() if raw_path.exists() else b""
        ).hexdigest(),
        "sampling": asdict(sampling),
        "raw_fingerprint": fingerprint,
        "token_distribution": _distribution(token_lengths),
        "response_token_distribution": _distribution(response_lengths),
        "chat_template_sha256": (
            hashlib.sha256(str(template).encode("utf-8")).hexdigest()
            if template
            else None
        ),
        "elapsed_s": round(time.time() - started, 1),
        **(extra_manifest or {}),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="corpus JSONL")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-samples", type=Path, default=None)
    parser.add_argument(
        "--extend-from",
        type=Path,
        default=None,
        help="fallback: existing corpus directory whose disjoint rows are kept",
    )
    parser.add_argument(
        "--from-stage0-local",
        type=Path,
        default=None,
        help=(
            "preferred: build from committed Stage 0 local O samples in this "
            "directory (for example <artifacts>/stage0/local)"
        ),
    )
    parser.add_argument(
        "--stage0-root",
        type=Path,
        default=DEFAULT_STAGE0_ROOT,
        help="Stage 0 root holding the registered protected sets",
    )
    parser.add_argument(
        "--stage0-pool",
        type=Path,
        default=DEFAULT_STAGE0_POOL,
        help="Stage 0 pool directory (default: <artifacts>/stage0/pool)",
    )
    parser.add_argument(
        "--screen-rows",
        type=int,
        default=DEFAULT_SCREEN_ROWS,
        help="section-7 screen rows for Stage 0 local selection",
    )
    parser.add_argument(
        "--expand-rows",
        type=int,
        default=DEFAULT_EXPAND_ROWS,
        help="section-7 expansion rows for Stage 0 local selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without loading a model or the GPU",
    )
    parser.add_argument("--target-verified", type=int, default=DEFAULT_TARGET_VERIFIED)
    parser.add_argument(
        "--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", choices=("e2b", "e4b"), default="e2b")
    parser.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--allow-missing-eval",
        action="store_true",
        help="smoke only: do not fail when an eval set cannot be loaded",
    )
    parser.add_argument(
        "--skip-decontam",
        action="store_true",
        help="smoke only: no EvalIndex (the manifest records this)",
    )
    parser.add_argument(
        "--heartbeat",
        default=None,
        help=(
            "heartbeat JSON path (default: "
            "<artifacts>/runs/detached/corpus.json)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_verified < 1:
        raise ValueError("--target-verified must be positive")
    if args.candidate_limit < 1:
        raise ValueError("--candidate-limit must be positive")
    if args.screen_rows < 1 or args.expand_rows < args.screen_rows:
        raise ValueError("--expand-rows must be >= --screen-rows >= 1")
    if args.from_stage0_local is not None and args.extend_from is not None:
        raise ValueError("--from-stage0-local and --extend-from are exclusive")

    def _tokenizer():
        from transformers import AutoTokenizer

        from gemmaloop.model.load import MODEL_IDS
        from gemmaloop.model.snapshot import model_revision

        return MODEL_IDS[args.size], AutoTokenizer.from_pretrained(
            MODEL_IDS[args.size], revision=model_revision(args.size)
        )

    if args.from_stage0_local is not None:
        local_dir = args.from_stage0_local
        if args.dry_run:
            tokenizer = None
            try:
                _, tokenizer = _tokenizer()
            except Exception as exc:  # pragma: no cover - cached in production
                print(f"warning: falling back to stored token counts: {exc}")
            plan = stage0_local_dry_run_plan(
                local_dir,
                args.stage0_pool,
                protected_root=args.stage0_root,
                tokenizer=tokenizer,
                screen_rows=args.screen_rows,
                expand_rows=args.expand_rows,
            )
            print(format_dry_run_plan(plan))
            return 0
        out_path = args.out or DEFAULT_LOCAL_CORPUS_DIR / "corpus.jsonl"
        manifest_path = args.manifest or DEFAULT_LOCAL_CORPUS_DIR / "manifest.json"
        tokenizer_id, tokenizer = _tokenizer()
        try:
            manifest = build_stage0_local_corpus(
                local_dir,
                args.stage0_pool,
                protected_root=args.stage0_root,
                out_path=out_path,
                manifest_path=manifest_path,
                tokenizer=tokenizer,
                screen_rows=args.screen_rows,
                expand_rows=args.expand_rows,
            )
        except CorpusBuildError as exc:
            print(f"p1lite corpus refused: {exc}", file=sys.stderr)
            return 2
        manifest.setdefault("tokenizer_id", tokenizer_id)
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        print(
            f"wrote {manifest['corpus_file']} "
            f"({manifest['n_verified']} verified traces) and {manifest_path}"
        )
        return 0

    root = artifacts_dir() / "data" / "p1lite"
    source_dir = args.extend_from
    out_path = args.out or (
        source_dir / "corpus.jsonl" if source_dir else root / "corpus.jsonl"
    )
    manifest_path = args.manifest or (
        source_dir / "manifest.json" if source_dir else root / "manifest.json"
    )
    raw_path = args.raw_samples or (
        source_dir / "samples.jsonl" if source_dir else root / "samples.jsonl"
    )

    if args.dry_run:
        if source_dir is None:
            raise ValueError("--dry-run requires --extend-from or --from-stage0-local")
        plan = build_dry_run_plan(
            source_dir,
            args.stage0_pool,
            protected_root=args.stage0_root,
            target_verified=args.target_verified,
            candidate_limit=args.candidate_limit,
        )
        print(format_dry_run_plan(plan))
        return 0

    require_gpu_lock()
    vram_cap = cap_cuda_allocator(0.90, 0.5)
    heartbeat = Heartbeat(
        args.heartbeat or default_corpus_heartbeat_path(),
        run="p1lite-corpus",
    )
    heartbeat.start(30.0)
    heartbeat.update(extra={"vram_cap": vram_cap})

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from gemmaloop.model.load import MODEL_IDS, ModelSpec, load_model
    from gemmaloop.model.snapshot import model_revision

    tokenizer_id = MODEL_IDS[args.size]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, revision=model_revision(args.size)
    )
    eval_index = None
    decontam_status: dict[str, Any] = {"enabled": False}
    if not args.skip_decontam:
        eval_index, decontam_status = build_eval_index(
            tokenizer, allow_missing=args.allow_missing_eval
        )
        decontam_status["enabled"] = True

    dataset = load_dataset(DATASET_ID, split=SPLIT, revision=REVISION)

    model, processor, _ = load_model(
        ModelSpec(size=args.size, quant=args.quant, loop=None)
    )
    sampling = Sampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=True,
    )
    from gemmaloop.serve.generate import generate_batch

    common_extra = {
        "model": {"size": args.size, "quant": args.quant, "id": tokenizer_id},
        "tokenizer": tokenizer_id,
        "decontamination": decontam_status,
        "vram_cap": vram_cap,
        "command": [sys.executable, *sys.argv],
    }
    raw_fingerprint = {
        "source_revision": REVISION,
        "model": {"size": args.size, "quant": args.quant, "id": tokenizer_id},
        "sampling": asdict(sampling),
        "chat_template_sha256": (
            hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest()
            if getattr(tokenizer, "chat_template", None)
            else None
        ),
    }
    try:
        if source_dir is not None:
            try:
                manifest = extend_corpus(
                    source_dir,
                    args.stage0_pool,
                    dataset,
                    generate_fn=generate_batch,
                    target_verified=args.target_verified,
                    out_path=out_path,
                    manifest_path=manifest_path,
                    raw_path=raw_path,
                    candidate_limit=args.candidate_limit,
                    protected_root=args.stage0_root,
                    eval_index=eval_index,
                    batch=args.batch,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    sampling=sampling,
                    heartbeat=heartbeat,
                    extra_manifest=common_extra,
                    raw_fingerprint=raw_fingerprint,
                )
            except CorpusBuildError as exc:
                print(f"p1lite corpus refused: {exc}", file=sys.stderr)
                return 2
        else:
            ids_by_source, _, stage0_info = disjointness_info(
                args.stage0_root, args.stage0_pool
            )
            protected_ids = set()
            for values in ids_by_source.values():
                protected_ids |= {str(value) for value in values}
            candidates, selection_stats = select_candidates(
                dataset,
                args.candidate_limit,
                eval_index=eval_index,
                protected_ids=protected_ids,
            )
            if not candidates:
                raise RuntimeError(
                    "no decontaminated numeric candidates were found"
                )
            manifest = build_corpus(
                candidates,
                generate_fn=generate_batch,
                target_verified=args.target_verified,
                out_path=out_path,
                manifest_path=manifest_path,
                raw_path=raw_path,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                sampling=sampling,
                batch=args.batch,
                extra_manifest={
                    **common_extra,
                    "selection": selection_stats,
                    "filter_version": STAGE0_FILTER_VERSION,
                    "stage0_disjointness": stage0_info,
                    "protected_sets": stage0_info["protected_sets"],
                    "stage0_pool_manifest_sha256": stage0_info[
                        "pool_manifest_sha256"
                    ],
                },
                raw_fingerprint=raw_fingerprint,
                heartbeat=heartbeat,
            )
    finally:
        del model
        heartbeat.stop()

    print(
        f"wrote {manifest['corpus_file']} "
        f"({manifest['n_verified']} verified traces) and {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
