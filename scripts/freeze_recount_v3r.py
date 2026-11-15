"""One documented command for the single authorized v3r replacement run.

Usage (only after the X1 replacement review is recorded and GO is issued)::

    uv run python scripts/freeze_recount_v3r.py

It performs, in order:

1. verify the synthetic-free official void receipt and continuity record;
2. claim ``<lite-dir>/shortfall/v3r_registry.json`` atomically and durably;
3. build/verify the authenticated source-type interpretation manifest;
4. freeze v3r (same candidates, requests, filters, interpretation/execution
   bytes, math decisions, comparison, 6 s outer timeout, sandbox restrictions,
   image digest, dependencies and concurrency as v3; only the disclosed
   verifier-identity repair differs);
5. full recount of all original selected requests into ``count_v3r`` with
   attempts starting at 1 and the single registered infrastructure retry;
6. extension decision from the actual v3r counts;
7. ``corpus_v3r`` and ``pilot-shortfall-freeze-v3r.json``.

A second fresh claim, an alternate output root, a missing/corrupt continuity
record or the void v3 identity are all refused before any output or dispatch.
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
from gemmaloop.pilot.eval_rows import render_trace_rows  # noqa: E402
from gemmaloop.stage0 import modal_lite  # noqa: E402
from gemmaloop.stage0.grader import DockerSandbox  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    LITE_IMAGE_TAG,
    V3R_MAX_INFRASTRUCTURE_RETRIES,
    V3R_OUT_DIR_NAME,
    _docker_recovery_check,
    load_void_receipt,
    require_v3r_guard,
    resolve_docker_image,
    resume_v3r_pipeline,
    run_v3r_pipeline,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stage0 = artifacts_dir() / "stage0"
    lite = stage0 / "modal_lite"
    shortfall = lite / "shortfall"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--pool-dir", type=Path, default=stage0 / "pool")
    parser.add_argument("--lite-dir", type=Path, default=lite)
    parser.add_argument(
        "--request-ids", type=Path, default=lite / "request_ids.txt"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=shortfall / V3R_OUT_DIR_NAME,
        help="must equal the fixed continuity output root",
    )
    parser.add_argument("--image", default=LITE_IMAGE_TAG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--no-pending",
        action="store_true",
        help="refuse to write the index while the amendment file is absent",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the claimed v3r run from verified completed artifacts and "
            "the durable attempt ledger; never claim or re-freeze"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    record_loader: object | None = None,
) -> int:
    args = _parse_args(argv)
    # A synthetic void receipt can never authorize an official run; the
    # production path (a real, non-synthetic receipt) keeps require_official
    # True.  This lets the mandatory synthetic real-Docker dry run exercise
    # this exact production main() with only its paths redirected.
    require_official = not bool(
        load_void_receipt(args.lite_dir).get("synthetic")
    )
    require_v3r_guard(
        args.lite_dir,
        out_root=args.out_root,
        require_review=True,
        require_official=require_official,
    )
    pin = resolve_docker_image(args.image)
    DockerSandbox(image=args.image, expected_digest=pin["pinned"]).verify_image_digest()
    tokenizer = modal_lite.load_tokenizer()
    pipeline = resume_v3r_pipeline if args.resume else run_v3r_pipeline
    result = pipeline(
        repo_root=args.repo_root,
        pool_dir=args.pool_dir,
        lite_dir=args.lite_dir,
        request_ids_path=args.request_ids,
        out_root=args.out_root,
        image=args.image,
        image_pin=pin,
        workers=max(1, int(args.workers)),
        record_loader=record_loader,
        recovery_check=_docker_recovery_check,
        tokenizer=tokenizer,
        renderer=render_trace_rows,
        max_length=int(args.max_length),
        allow_pending=not args.no_pending,
        max_infrastructure_retries=V3R_MAX_INFRASTRUCTURE_RETRIES,
    )
    summary = result.pop("count_summary")
    print(
        json.dumps(
            {
                **result,
                "run_version": "v3r",
                "nT0": summary["nT0"],
                "nS0": summary["nS0"],
                "pilot_p1": summary["pilot_p1"],
                "extension_required": summary["extension_required"],
                "extension_decision": summary["extension_decision"],
                "infrastructure_failures": summary["infrastructure_failures"],
                "accounting": summary["accounting"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
