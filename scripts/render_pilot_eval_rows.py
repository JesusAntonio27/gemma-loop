"""Render the Stage 0 pilot splits into evaluation rows (CPU, deterministic).

Reads ``<stage0>/pilot/pilot_tfit.jsonl``, ``pilot_selection.jsonl`` and
``pilot_p1.jsonl`` (``messages`` traces with ``problem_id``) and writes
``<out>/pilot_tfit.jsonl``, ... with the training tokenisation (assistant-only
labels) plus a manifest of source hashes, tokenizer revision and output hashes.

The renderer is idempotent: a matching manifest is reused without touching the
tokenizer.  Generated rows carry ``problem_id`` so ``run_pilot`` can build the
reserved split map and the problem-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.pilot.eval_rows import (  # noqa: E402
    DEFAULT_EVAL_SPLITS,
    DEFAULT_TOKENIZER,
    render_stage0_eval_rows,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_EVAL_SPLITS),
        help="comma-separated Stage 0 pilot split names",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = tuple(
        part.strip() for part in str(args.splits).split(",") if part.strip()
    )
    if not splits:
        print("error: --splits is empty", file=sys.stderr)
        return 2
    from scripts.precompute_teacher_topk import load_student_tokenizer

    tokenizer = load_student_tokenizer(args.tokenizer)
    payload = render_stage0_eval_rows(
        args.stage0,
        args.out,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        splits=splits,
        tokenizer_name=args.tokenizer,
        # The tokenizer is always loaded at the pinned revision
        # (load_student_tokenizer); the manifest records that same revision, so
        # there is no --tokenizer-revision flag that could misstate it.
        force=args.force,
    )
    print(json.dumps({"out": str(args.out), "manifest": payload}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
