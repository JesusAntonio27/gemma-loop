"""Build the decontaminated Stage 0 problem pool and its deterministic splits.

Example (CPU, after the Modal download or from the pinned Hub snapshots)::

    uv run python scripts/build_stage0_pool.py \
        --output-dir artifacts/stage0/pool --eval-limit 64

The command downloads the pinned source snapshots and every registered
evaluation corpus, builds the repo's 13-gram + Indel-0.75 index, writes
``problems.jsonl`` plus one file per registered split, and freezes counts and
SHA-256 hashes in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

from gemmaloop.model.snapshot import MODEL_REVISION
from gemmaloop.stage0.evals import (
    DEFAULT_BRIEFS,
    build_eval_index,
    default_extra_eval_files,
    gather_eval_texts,
)
from gemmaloop.stage0.pool import (
    PART1_CREATIVE_BRIEFS,
    REGISTERED_COUNTS,
    build_pool,
    write_pool,
)
from gemmaloop.stage0.sources import (
    POOL_SOURCES,
    PoolSource,
    convert_row,
    iter_source_rows,
)

LOGGER = logging.getLogger("build_stage0_pool")
DRY_RUN_ROWS = 50


def dry_run_sources(
    rows: int = DRY_RUN_ROWS,
    sources: Iterable[PoolSource] | None = None,
    stream=None,
) -> int:
    """Load a few rows of every source through the real loaders and report.

    Returns 0 when every source produced at least one usable problem, else 1.
    This is a controller pre-flight check; it performs no decontamination and
    writes no pool artifacts.
    """

    stream = stream or sys.stdout
    sources = POOL_SOURCES if sources is None else sources
    failures: list[str] = []
    for source in sources:
        scanned = 0
        accepted = 0
        dropped: Counter[str] = Counter()
        try:
            for index, row in iter_source_rows(source, limit_rows=rows):
                scanned += 1
                problem = convert_row(source.name, row, index)
                if problem is None:
                    dropped["unusable_row"] += 1
                else:
                    accepted += 1
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print(
                f"{source.name}: ERROR {type(exc).__name__}: {exc}",
                file=stream,
            )
            failures.append(source.name)
            continue
        detail = ""
        if dropped:
            detail = " dropped=" + ",".join(
                f"{name}:{count}" for name, count in sorted(dropped.items())
            )
        print(
            f"{source.name}: rows={scanned} accepted={accepted}{detail}",
            file=stream,
        )
        if accepted == 0:
            failures.append(source.name)
    if failures:
        print("dry-run FAILED sources: " + ", ".join(failures), file=stream)
        return 1
    return 0


def _quotas_for(counts: dict[str, int]) -> dict[str, int]:
    """Scale each source's registered quota to the requested family count."""

    by_kind: dict[str, list] = {}
    for source in POOL_SOURCES:
        by_kind.setdefault(source.kind, []).append(source)
    quotas: dict[str, int] = {}
    for kind, sources in by_kind.items():
        registered = REGISTERED_COUNTS.get(kind, sum(s.quota for s in sources))
        target = int(counts.get(kind, registered))
        for source in sources:
            quotas[source.name] = int(round(source.quota * target / registered))
    return quotas


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage0/pool"))
    parser.add_argument("--code-count", type=int, default=REGISTERED_COUNTS["code"])
    parser.add_argument("--math-count", type=int, default=REGISTERED_COUNTS["math"])
    parser.add_argument("--chat-count", type=int, default=REGISTERED_COUNTS["chat"])
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="maximum rows scanned per source (smoke runs)")
    parser.add_argument("--stream-shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="maximum texts per evaluation benchmark (smoke runs)")
    parser.add_argument("--extra-eval-file", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    parser.add_argument("--allow-missing-eval", action="store_true",
                        help="smoke only: continue if a mandatory eval cannot be loaded")
    parser.add_argument("--allow-missing-briefs", action="store_true",
                        help="smoke only: continue without the creative-brief decontamination target")
    parser.add_argument("--part1", action="store_true",
                        help=("Part 1 mode: build the pool without the Part 2 creative "
                              "eval briefs (not the smoke flag).  The manifest records "
                              "that Part 2 decontaminates its briefs against this pool."))
    parser.add_argument("--allow-short-sources", action="store_true",
                        help="smoke only: continue when a mandatory source is short")
    parser.add_argument("--dry-run-sources", action="store_true",
                        help=("pre-flight: load the first --dry-run-rows rows of every "
                              "source through the real loaders and print accepted "
                              "counts; no eval index, tokenizer or pool is built"))
    parser.add_argument("--dry-run-rows", type=int, default=DRY_RUN_ROWS,
                        help="rows per source for --dry-run-sources (default 50)")
    return parser.parse_args(argv)


def _briefs_config(args: argparse.Namespace) -> tuple[bool, bool, str]:
    """Return ``(include_default_briefs, allow_gate, manifest_note)``.

    Part 1 is an explicit registered mode: the default briefs target is not
    loaded, the briefs gate is skipped, and the manifest records the reverse
    decontamination direction.  The smoke flag is not reused.
    """

    if bool(getattr(args, "part1", False)):
        return False, True, PART1_CREATIVE_BRIEFS
    return True, bool(getattr(args, "allow_missing_briefs", False)), "included"


def _briefs_gate(
    extra_status: dict[str, dict],
    allow_missing: bool,
) -> None:
    """Require the creative briefs target itself, not just any extra file."""

    def is_briefs(entry: dict) -> bool:
        path = entry.get("path")
        return bool(path) and Path(path).name == DEFAULT_BRIEFS.name

    briefs_ok = any(
        entry.get("status") == "ok" and is_briefs(entry)
        for entry in extra_status.values()
    )
    briefs_failed = any(
        entry.get("status") == "failed" and is_briefs(entry)
        for entry in extra_status.values()
    )
    if allow_missing:
        return
    if briefs_failed:
        raise RuntimeError(
            "creative-brief decontamination failed (corrupt or unreadable file): "
            f"{extra_status}. Freeze a valid briefs file or pass "
            "--allow-missing-briefs only for a smoke run."
        )
    if not briefs_ok:
        raise RuntimeError(
            "creative-brief decontamination target is missing or empty: "
            f"{extra_status}. Freeze the briefs, pass the briefs file with "
            "--extra-eval-file, or use --allow-missing-briefs for a smoke run."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    if args.dry_run_sources:
        return dry_run_sources(args.dry_run_rows)

    include_default_briefs, allow_gate, creative_briefs = _briefs_config(args)
    extra = list(args.extra_eval_file)
    if include_default_briefs:
        extra.extend(default_extra_eval_files())
    eval_texts, eval_status = gather_eval_texts(
        eval_limit=args.eval_limit,
        extra_eval_files=extra,
        allow_missing=args.allow_missing_eval,
    )
    extra_status = {
        name: entry
        for name, entry in eval_status.items()
        if name.startswith("file:")
    }
    _briefs_gate(extra_status, allow_gate)
    if not eval_texts:
        raise RuntimeError("empty evaluation index; refusing to build the Stage 0 pool")

    LOGGER.info("loading tokenizer %s @ %s", args.tokenizer, MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION)
    eval_index = build_eval_index(eval_texts, tokenizer=tokenizer)
    LOGGER.info("evaluation index: %d texts, %d-grams", len(eval_index), eval_index.n)

    quotas = _quotas_for(
        {
            "code": args.code_count,
            "math": args.math_count,
            "chat": args.chat_count,
        }
    )
    result = build_pool(
        eval_index=eval_index,
        eval_status=eval_status,
        quotas=quotas,
        limit_rows=args.limit_rows,
        shuffle_buffer=args.stream_shuffle_buffer,
        seed=args.seed,
        tokenizer=tokenizer,
        tokenizer_revision=MODEL_REVISION,
        allow_short_sources=args.allow_short_sources,
        eval_limit=args.eval_limit,
        allow_missing_eval=args.allow_missing_eval,
        allow_missing_briefs=args.allow_missing_briefs,
        creative_briefs=creative_briefs,
    )
    manifest = write_pool(
        result,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_revision=MODEL_REVISION,
    )
    LOGGER.info(
        "wrote %d problems to %s (manifest %s)",
        len(result.problems),
        args.output_dir,
        manifest["manifest_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
