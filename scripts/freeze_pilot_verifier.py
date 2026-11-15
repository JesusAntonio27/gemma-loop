"""Freeze the pilot code verifier and its ordered test manifest.

Usage::

    uv run python scripts/freeze_pilot_verifier.py \
        --pool-dir "$GEMMALOOP_ARTIFACTS/stage0/pool" \
        --lite-dir "$GEMMALOOP_ARTIFACTS/stage0/modal_lite" \
        --out "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/verifier"

The script pins ``python:3.12-slim`` by ``docker image inspect`` content digest
(not the tag), records the exact ordered per-problem test bytes, and hashes the
extraction/wrapper/comparison implementation and dependency versions.
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
from gemmaloop.stage0.shortfall import (  # noqa: E402
    FREEZE_MANIFEST_NAME,
    freeze_verifier,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stage0 = artifacts_dir() / "stage0"
    lite = stage0 / "modal_lite"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--pool-dir", type=Path, default=stage0 / "pool")
    parser.add_argument("--lite-dir", type=Path, default=lite)
    parser.add_argument(
        "--request-ids", type=Path, default=lite / "request_ids.txt"
    )
    parser.add_argument(
        "--out", type=Path, default=lite / "shortfall" / "verifier"
    )
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--timeout-seconds", type=float, default=6.0)
    parser.add_argument(
        "--interpretation",
        type=Path,
        default=None,
        help="authenticated interpretation manifest (required for v3)",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = freeze_verifier(
        repo_root=args.repo_root,
        pool_dir=args.pool_dir,
        lite_dir=args.lite_dir,
        request_ids_path=args.request_ids,
        out_dir=args.out,
        image=args.image,
        timeout_s=args.timeout_seconds,
        interpretation_path=args.interpretation,
        workers=max(1, int(args.workers)),
    )
    print(
        json.dumps(
            {
                "freeze": str((args.out / FREEZE_MANIFEST_NAME).resolve()),
                "manifest_sha256": payload["manifest_sha256"],
                "docker_image": payload["verifier"]["docker_image"],
                "test_manifest_sha256": payload["tests"]["manifest_sha256"],
                "test_problems": payload["tests"]["problems"],
                "tests": payload["tests"]["tests"],
                "selected": {
                    name: split["problem_count"]
                    for name, split in payload["selected_inputs"]["splits"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
