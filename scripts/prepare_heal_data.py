"""Build the decontaminated healing-data mixture.

Usage::

    uv run python scripts/prepare_heal_data.py --target-tokens 200000 \
        --output-dir artifacts/smoke --eval-limit 64

The script streams every training source, builds an evaluation index from the
benchmark adapters (LiveCodeBench is read line-by-line and never materialised),
and writes ``train.jsonl`` plus ``manifest.json`` under
``gemmaloop.paths.artifacts_dir() / "data" / "heal"`` by default.  Re-running the
same command is idempotent: streaming selection, the prefix-stable interleave
and the content hash all depend only on the arguments and the upstream data.

The benchmark index fails closed: a load/auth failure or an unexpectedly small
benchmark aborts the run unless ``--allow-missing-eval`` is passed (smoke runs
only).  The creative briefs file is an optional extra and only warns when
missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from gemmaloop.data.decontam import EvalIndex
from gemmaloop.data.mix import build_mix, write_jsonl
from gemmaloop.eval.benchmarks import (
    aime,
    cvbench,
    gpqa,
    lcb,
    mathvision,
    mmlupro,
    mmmupro,
    spatialeval,
)
from gemmaloop.model.snapshot import MODEL_REVISION
from gemmaloop.paths import artifacts_dir

LOGGER = logging.getLogger("prepare_heal_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIEFS = REPO_ROOT / "data" / "creative" / "briefs_eval.jsonl"

# Snapshots used when the smoke was written.  Pinning these makes "same command,
# same output" true as long as the local cache is intact.
LCB_REVISION = lcb.REVISION
AIME_REVISION = aime.REVISION
MMMU_PRO_REVISION = mmmupro.REVISION
MATH_VISION_REVISION = mathvision.REVISION
SPATIAL_EVAL_REVISION = spatialeval.REVISION
CV_BENCH_REVISION = cvbench.REVISION


# ---------------------------------------------------------------------------
# Evaluation corpus
# ---------------------------------------------------------------------------
def _stream_texts(
    hf_id: str,
    fields: tuple[str, ...],
    *,
    config: str | None = None,
    split: str = "test",
    revision: str | None = None,
    limit: int | None = None,
    columns: bool = True,
) -> Iterator[str]:
    """Yield the first non-empty text field of each streamed row."""

    kwargs: dict[str, Any] = {}
    if config is not None:
        kwargs["name"] = config
    if revision is not None:
        kwargs["revision"] = revision

    def dataset_iter():
        if columns:
            try:
                stream = load_dataset(
                    hf_id,
                    split=split,
                    streaming=True,
                    columns=list(fields),
                    **kwargs,
                )
                return iter(stream)
            except (TypeError, ValueError):
                pass
        return iter(load_dataset(hf_id, split=split, streaming=True, **kwargs))

    for index, row in enumerate(dataset_iter()):
        if limit is not None and index >= limit:
            break
        for field in fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                yield str(value)
                break


def _lcb_texts(limit: int | None = None) -> Iterator[str]:
    """Stream ``release_v6`` prompts without keeping the release in memory."""

    count = 0
    for filename in lcb.RELEASE_FILES[lcb.RELEASE]:
        path = hf_hub_download(
            repo_id=lcb.DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=LCB_REVISION,
        )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = str(row.get("question_content") or "")
                starter = str(row.get("starter_code") or "")
                text = question.strip()
                if starter.strip():
                    text = f"{text}\n\nStarter code:\n{starter.strip()}"
                if text.strip():
                    yield text
                    count += 1
                    if limit is not None and count >= limit:
                        return


def _collect_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _collect_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_strings(item)


def _jsonl_texts(path: Path) -> Iterator[str]:
    preferred = ("prompt", "brief", "text", "question", "instruction", "content")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, str):
                yield record
                continue
            if not isinstance(record, dict):
                yield from _collect_strings(record)
                continue
            found = False
            for key in preferred:
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    yield value
                    found = True
            if not found:
                yield from _collect_strings(record)


@dataclass(frozen=True)
class EvalSpec:
    name: str
    loader: Callable[[int | None], Iterable[str]]
    expected: int
    hf_id: str
    split: str
    config: str | None = None
    revision: str | None = None
    mandatory: bool = True


EVAL_SPECS: tuple[EvalSpec, ...] = (
    EvalSpec(
        "livecodebench_v6",
        _lcb_texts,
        1055,
        lcb.DATASET_ID,
        lcb.RELEASE,
        revision=LCB_REVISION,
    ),
    EvalSpec(
        "gpqa_diamond",
        lambda limit: _stream_texts(
            gpqa.DATASET_ID,
            ("Question",),
            config=gpqa.CONFIG,
            split=gpqa.SPLIT,
            revision=gpqa.REVISION,
            limit=limit,
            columns=False,
        ),
        198,
        gpqa.DATASET_ID,
        gpqa.SPLIT,
        config=gpqa.CONFIG,
        revision=gpqa.REVISION,
    ),
    EvalSpec(
        "aime26",
        lambda limit: _stream_texts(
            aime.DATASET_ID,
            ("problem",),
            split=aime.SPLIT,
            revision=AIME_REVISION,
            limit=limit,
            columns=False,
        ),
        30,
        aime.DATASET_ID,
        aime.SPLIT,
        revision=AIME_REVISION,
    ),
    EvalSpec(
        "mmlu_pro",
        lambda limit: _stream_texts(
            mmlupro.DATASET_ID,
            ("question",),
            split=mmlupro.SPLIT,
            revision=mmlupro.REVISION,
            limit=limit,
        ),
        12032,
        mmlupro.DATASET_ID,
        mmlupro.SPLIT,
        revision=mmlupro.REVISION,
    ),
    EvalSpec(
        "mmmu_pro",
        lambda limit: _stream_texts(
            mmmupro.DATASET_ID,
            ("question",),
            config=mmmupro.STANDARD_CONFIG,
            split=mmmupro.SPLIT,
            revision=MMMU_PRO_REVISION,
            limit=limit,
        ),
        1730,
        mmmupro.DATASET_ID,
        mmmupro.SPLIT,
        config=mmmupro.STANDARD_CONFIG,
        revision=MMMU_PRO_REVISION,
    ),
    EvalSpec(
        "math_vision",
        lambda limit: _stream_texts(
            mathvision.DATASET_ID,
            ("question",),
            split=mathvision.SPLIT,
            revision=MATH_VISION_REVISION,
            limit=limit,
        ),
        3040,
        mathvision.DATASET_ID,
        mathvision.SPLIT,
        revision=MATH_VISION_REVISION,
    ),
    EvalSpec(
        "spatial_eval_tqa",
        lambda limit: _stream_texts(
            spatialeval.DATASET_ID,
            ("text",),
            config="tqa",
            split=spatialeval.SPLIT,
            revision=SPATIAL_EVAL_REVISION,
            limit=limit,
        ),
        4635,
        spatialeval.DATASET_ID,
        spatialeval.SPLIT,
        config="tqa",
        revision=SPATIAL_EVAL_REVISION,
    ),
    EvalSpec(
        "cv_bench_2d",
        lambda limit: _stream_texts(
            cvbench.DATASET_ID,
            ("prompt", "question"),
            config=cvbench.CONFIG,
            split=cvbench.SPLIT,
            revision=CV_BENCH_REVISION,
            limit=limit,
        ),
        1438,
        cvbench.DATASET_ID,
        cvbench.SPLIT,
        config=cvbench.CONFIG,
        revision=CV_BENCH_REVISION,
    ),
)


def gather_eval_texts(
    *,
    eval_limit: int | None,
    extra_eval_files: list[Path],
    allow_missing: bool = False,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load the mandatory benchmark suite and any optional extra files.

    Mandatory failures (exception, empty result, or fewer rows than the pinned
    snapshot) abort the run with :class:`RuntimeError` unless ``allow_missing``
    is set.  Optional extras only log a warning.
    """

    texts: list[str] = []
    statuses: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for spec in EVAL_SPECS:
        entry: dict[str, Any] = {
            "status": "ok",
            "texts": 0,
            "mandatory": bool(spec.mandatory),
            "hf_id": spec.hf_id,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "expected": spec.expected,
        }
        try:
            part = [text for text in spec.loader(eval_limit)]
        except Exception as exc:  # noqa: BLE001 - surfaced in status/manifest
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            statuses[spec.name] = entry
            message = f"{spec.name}: {entry['error']}"
            if spec.mandatory:
                failures.append(message)
                LOGGER.error("mandatory eval benchmark failed: %s", message)
            else:
                LOGGER.warning("optional eval benchmark failed: %s", message)
            continue

        required = spec.expected if eval_limit is None else min(spec.expected, eval_limit)
        entry["texts"] = len(part)
        if len(part) == 0:
            entry["status"] = "failed"
            entry["error"] = "loader returned no texts"
        elif len(part) < required:
            entry["status"] = "short"
            entry["error"] = f"expected at least {required} texts, got {len(part)}"

        if entry["status"] != "ok":
            message = f"{spec.name}: {entry['error']}"
            if spec.mandatory:
                failures.append(message)
                LOGGER.error("mandatory eval benchmark incomplete: %s", message)
            else:
                LOGGER.warning("optional eval benchmark incomplete: %s", message)
        else:
            LOGGER.info("eval benchmark %-18s %6d texts", spec.name, len(part))
        statuses[spec.name] = entry
        texts.extend(part)

    for path in extra_eval_files:
        key = f"file:{path.name}"
        entry = {
            "status": "ok",
            "texts": 0,
            "mandatory": False,
            "path": str(path),
        }
        if not path.exists():
            entry["status"] = "missing"
            entry["error"] = "file not found"
            LOGGER.warning("optional eval file missing: %s", path)
            statuses[key] = entry
            continue
        try:
            part = list(_jsonl_texts(path))
        except Exception as exc:  # noqa: BLE001 - optional file only warns
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("optional eval file failed: %s: %s", path, exc)
            statuses[key] = entry
            continue
        entry["texts"] = len(part)
        if not part:
            entry["status"] = "failed"
            entry["error"] = "file produced no texts"
            LOGGER.warning("optional eval file produced no texts: %s", path)
        else:
            LOGGER.info("eval file %-25s %6d texts", str(path), len(part))
        statuses[key] = entry
        texts.extend(part)

    if failures and not allow_missing:
        raise RuntimeError(
            "mandatory evaluation index incomplete: "
            + "; ".join(failures)
            + ". Pass --allow-missing-eval only for smoke runs."
        )
    if failures:
        LOGGER.warning(
            "--allow-missing-eval set; continuing without: %s", "; ".join(failures)
        )
    return texts, statuses


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _peak_ram_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        for attribute in ("peak_wset", "peak_pagefile"):
            value = getattr(info, attribute, None)
            if value:
                return round(value / (1024 * 1024), 1)
        return round(info.rss / (1024 * 1024), 1)
    except Exception:
        pass

    # psutil is not a declared project dependency.  On Windows, query the
    # kernel's peak working set directly; on POSIX use getrusage.
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
        else:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        pass
    return float("nan")


def _assert_not_onedrive(path: Path) -> None:
    if "onedrive" in str(path).lower():
        raise ValueError(
            f"Refusing to write artifacts to {path}: heavy artifacts must not "
            "live inside OneDrive."
        )


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-tokens",
        "--tokens",
        dest="target_tokens",
        type=int,
        default=40_000_000,
        help="token budget of the mixture (default: 40,000,000)",
    )
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="maximum rows scanned per source")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "training sequence length; each row contributes "
            "min(rendered_len, max_length) toward --target-tokens"
        ),
    )
    parser.add_argument(
        "--length-mode",
        choices=("truncate", "filter"),
        default="filter",
        help=(
            "truncate keeps the start of long rows; filter drops rows that do "
            "not fit and rebalances reasoning sources to >=75%% of trained "
            "tokens (default: filter; requires --max-length)"
        ),
    )
    parser.add_argument(
        "--allow-low-reasoning",
        action="store_true",
        help=(
            "filter mode only: build even when the reasoning share of trained "
            "tokens is below 75%%"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="maximum texts per evaluation benchmark (smoke runs)",
    )
    parser.add_argument(
        "--extra-eval-file",
        type=Path,
        action="append",
        default=[],
        help="additional JSONL file with evaluation prompts; may be repeated",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/gemma-4-E2B-it",
        help="tokenizer used for 13-grams and token accounting",
    )
    parser.add_argument(
        "--stream-shuffle-buffer",
        type=int,
        default=0,
        help="optional streaming shuffle buffer per source (0 keeps stream order)",
    )
    parser.add_argument(
        "--allow-missing-eval",
        action="store_true",
        help="smoke only: continue when a mandatory benchmark cannot be loaded",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "datasets", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args(argv)

    extra_files = list(args.extra_eval_file)
    if DEFAULT_BRIEFS.exists():
        extra_files.append(DEFAULT_BRIEFS)
    else:
        LOGGER.warning(
            "creative briefs not found at %s; skipping (pass --extra-eval-file "
            "after Task 19 freezes them)",
            DEFAULT_BRIEFS,
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else artifacts_dir() / "data" / "heal"
    )
    _assert_not_onedrive(output_dir)

    eval_limit = args.eval_limit if args.eval_limit and args.eval_limit > 0 else None
    eval_texts, eval_status = gather_eval_texts(
        eval_limit=eval_limit,
        extra_eval_files=extra_files,
        allow_missing=args.allow_missing_eval,
    )
    if not eval_texts:
        raise RuntimeError("empty evaluation index; refusing to build a mixture")

    LOGGER.info("loading tokenizer %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION)
    eval_index = EvalIndex.build(eval_texts, tokenizer=tokenizer)
    LOGGER.info("evaluation index: %d texts, %d-grams", len(eval_index), eval_index.n)

    length_mode = args.length_mode
    if args.max_length is None:
        if length_mode == "filter":
            LOGGER.warning(
                "--length-mode filter requires --max-length; falling back to "
                "legacy quota-token accounting"
            )
        length_mode = "truncate"

    build_started = time.perf_counter()
    rows, manifest = build_mix(
        args.target_tokens,
        tokenizer,
        eval_index,
        seed=args.seed,
        limit_rows=args.limit_rows,
        stream_shuffle_buffer=args.stream_shuffle_buffer,
        max_length=args.max_length,
        length_mode=length_mode,
        allow_low_reasoning=args.allow_low_reasoning,
    )
    build_seconds = time.perf_counter() - build_started
    if (
        length_mode == "filter"
        and manifest.achieved_reasoning_share
        < manifest.target_reasoning_share - 0.01
    ):
        LOGGER.warning(
            "--allow-low-reasoning: reasoning share %.1f%% is below the %.1f%% "
            "target",
            manifest.achieved_reasoning_share * 100.0,
            manifest.target_reasoning_share * 100.0,
        )
    if args.max_length is None:
        LOGGER.warning(
            "no --max-length: --target-tokens counts raw quota tokens; the "
            "trainer will count trained tokens at its own max_length"
        )
    else:
        LOGGER.info(
            "length-aware target: mode=%s, max_length=%d, reasoning share "
            "requested=%.1f%% effective=%.1f%%",
            length_mode,
            args.max_length,
            manifest.requested_reasoning_share * 100.0,
            manifest.target_reasoning_share * 100.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(train_path, rows)
    payload = manifest.model_dump()
    payload.update(
        {
            "output": str(train_path),
            "eval_texts": len(eval_texts),
            "eval_benchmarks": eval_status,
            "eval_limit": eval_limit,
        }
    )
    _write_json_atomic(manifest_path, payload)

    peak_mb = _peak_ram_mb()
    total_tokens = sum(manifest.trained_tokens_per_source.values())
    total_raw_tokens = sum(manifest.raw_tokens_per_source.values())
    rows_per_second = len(rows) / build_seconds if build_seconds > 0 else float("nan")
    tokens_per_second = (
        total_tokens / build_seconds if build_seconds > 0 else float("nan")
    )
    projected_40m_hours = (
        40_000_000 / tokens_per_second / 3600
        if tokens_per_second and tokens_per_second > 0
        else float("nan")
    )
    summary = {
        "train": str(train_path),
        "manifest": str(manifest_path),
        "sha256": manifest.sha256,
        "target_tokens": manifest.target_tokens,
        "max_length": manifest.max_length,
        "length_mode": manifest.length_mode,
        "tokens_per_source": manifest.tokens_per_source,
        "raw_tokens_per_source": manifest.raw_tokens_per_source,
        "trained_tokens_per_source": manifest.trained_tokens_per_source,
        "requested_weights": manifest.requested_weights,
        "effective_weights": manifest.effective_weights,
        "achieved_weights": manifest.achieved_weights,
        "requested_reasoning_share": manifest.requested_reasoning_share,
        "achieved_reasoning_share": manifest.achieved_reasoning_share,
        "target_reasoning_share": manifest.target_reasoning_share,
        "allow_low_reasoning": manifest.allow_low_reasoning,
        "unavailable_sources": manifest.unavailable_sources,
        "rows_per_source": manifest.rows_per_source,
        "total_tokens": total_tokens,
        "total_raw_tokens": total_raw_tokens,
        "dropped": manifest.dropped,
        "peak_ram_mb": peak_mb,
        "build_seconds": round(build_seconds, 2),
        "rows_per_second": round(rows_per_second, 2),
        "tokens_per_second": round(tokens_per_second, 1),
        "projected_40m_hours": round(projected_40m_hours, 2),
        "eval_benchmarks": eval_status,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("wrote %d rows to %s (peak RAM %.1f MB)",
                len(rows), train_path, peak_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
