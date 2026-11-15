"""Recover original APPS/TACO element types and freeze the interpretation.

Usage::

    uv run python scripts/recover_source_types.py \
        --freeze "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/verifier_v3/verifier_freeze.json" \
        --out "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/interpretation_v3"

The tool reads only the already pinned APPS/TACO/CodeContests revisions
registered in ``gemmaloop.stage0.sources`` (the local Hugging Face cache is used
first; the pinned public revision may be fetched when absent).  Every frozen
code problem is recovered from its pinned record and replayed through the old
converter; a mismatch, missing/mismatched record, mapping failure or
unsupported original type excludes the *whole* problem before grading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.paths import artifacts_dir  # noqa: E402
from gemmaloop.stage0 import local  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    FREEZE_MANIFEST_NAME,
    load_freeze_manifest,
)
from gemmaloop.stage0.source_recovery import recover_source_types  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shortfall = artifacts_dir() / "stage0" / "modal_lite" / "shortfall"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=shortfall / "verifier" / FREEZE_MANIFEST_NAME,
    )
    parser.add_argument(
        "--out", type=Path, default=shortfall / "interpretation"
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=("apps", "taco", "codecontests"),
        default=None,
        help=(
            "network-backed original source (default: apps, taco and "
            "codecontests)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    freeze = load_freeze_manifest(args.freeze)
    test_manifest = Path(args.freeze).parent / freeze["tests"]["manifest_file"]
    frozen_rows = [
        json.loads(line)
        for line in test_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool_dir = Path(freeze["inputs"]["pool_dir"])
    problems_by_id = {
        str(row["problem_id"]): row
        for row in local.read_jsonl(pool_dir / "problems.jsonl")
    }
    manifest = recover_source_types(
        frozen_rows,
        problems_by_id,
        out_dir=args.out,
        sources=tuple(args.source or ("apps", "taco", "codecontests")),
    )
    print(
        json.dumps(
            {
                "manifest": str(
                    (Path(args.out) / "interpretation_manifest.json").resolve()
                ),
                "manifest_sha256": manifest["manifest_sha256"],
                "problems": len(manifest["problems"]),
                "authenticated_problems": manifest["authenticated_problem_count"],
                "excluded_problems": manifest["excluded_problem_count"],
                "authenticated_tests": manifest["authenticated_test_count"],
                "exclusions": [
                    {
                        "problem_id": item["problem_id"],
                        "source": item.get("source"),
                        "split": item.get("split"),
                        "reason": item["reason"],
                    }
                    for item in manifest["exclusions"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
