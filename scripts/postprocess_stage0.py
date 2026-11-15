"""Verify Modal samples and build the Stage 0 corpora.

Example::

    uv run python scripts/postprocess_stage0.py \
        --pool-dir artifacts/stage0/pool \
        --samples artifacts/stage0/samples \
        --hidden-tests artifacts/stage0/hidden_tests.jsonl \
        --output-dir artifacts/stage0/postprocess \
        --sandbox docker

The command writes ``sample_grades.jsonl``, ``stage1_heal.jsonl``,
``stage2_prompts.jsonl``, ``pilot_verified.jsonl``, ``nll_kl_verified.jsonl``
and ``long_traces.jsonl``, then records their counts and SHA-256 values in the
pool manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from gemmaloop.model.snapshot import MODEL_REVISION
from gemmaloop.stage0.grader import GraderService
from gemmaloop.stage0.postprocess import (
    MAX_LENGTH,
    MIN_REASONING_SHARE,
    STAGE2_MEDIAN_MAX,
    postprocess,
    write_postprocess,
)

LOGGER = logging.getLogger("postprocess_stage0")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sample_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def _merge_hidden_tests(
    problems: list[dict[str, Any]], hidden_path: Path | None
) -> None:
    if hidden_path is None or not hidden_path.exists():
        return
    hidden = {
        record["problem_id"]: record.get("tests") or []
        for record in _read_jsonl(hidden_path)
    }
    for problem in problems:
        extra = hidden.get(problem["problem_id"])
        if extra:
            existing = list(problem.get("tests") or [])
            problem["tests"] = existing + [test for test in extra if test not in existing]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=Path("artifacts/stage0/pool"))
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--hidden-tests", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage0/postprocess"))
    parser.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--reasoning-share", type=float, default=MIN_REASONING_SHARE)
    parser.add_argument("--stage2-median-max", type=int, default=STAGE2_MEDIAN_MAX)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker",
                        help="code verification sandbox (Docker required by default)")
    parser.add_argument("--allow-unsandboxed", action="store_true",
                        help="explicitly allow the NON-SANDBOXED local fallback")
    parser.add_argument("--timeout", type=float, default=6.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    problems = _read_jsonl(args.pool_dir / "problems.jsonl")
    _merge_hidden_tests(problems, args.hidden_tests)
    samples: list[dict[str, Any]] = []
    for path in _sample_files(args.samples):
        samples.extend(_read_jsonl(path))

    if args.sandbox == "local" and not args.allow_unsandboxed:
        raise SystemExit("refusing --sandbox local without --allow-unsandboxed")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION)

    def token_counter(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    with GraderService(
        workers=args.workers,
        timeout=args.timeout,
        sandbox=args.sandbox,
        allow_fallback=args.allow_unsandboxed,
    ) as grader:
        result = postprocess(
            problems,
            samples,
            grader=grader,
            tokenizer=tokenizer,
            token_counter=token_counter,
            max_length=args.max_length,
            min_reasoning_share=args.reasoning_share,
            stage2_median_max=args.stage2_median_max,
            token_counter_method=f"tokenizer:{args.tokenizer}@{MODEL_REVISION}",
        )
    manifest = write_postprocess(
        result,
        args.output_dir,
        pool_manifest_path=args.pool_dir / "manifest.json",
    )
    LOGGER.info(
        "kept %d/%d samples; %d Stage 2 prompts; %d heal rows",
        manifest["counts"]["samples_kept"],
        manifest["counts"]["samples_input"],
        manifest["counts"]["stage2_prompts"],
        manifest["counts"]["stage1_heal_rows"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
