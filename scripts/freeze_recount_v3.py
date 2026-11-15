"""One documented command for the v3 interpretation/freeze/recount/corpus/index.

Usage::

    uv run python scripts/freeze_recount_v3.py

It performs, in order:

1. build/verify the source-type interpretation manifest;
2. freeze v3 (interpretation artifacts, actual executing files, dependencies,
   Docker image content digest, sandbox limits, concurrency, lifecycle/retry
   policy);
3. full recount of every original selected request into ``count_v3``;
4. extension decision from the actual v3 counts;
5. ``corpus_v3``;
6. ``pilot-shortfall-freeze-v3.json`` (pending only for the amendment file if
   it is absent from this worktree).

The run is one-time.  Before any output it claims the fixed registry/lock
``<lite-dir>/shortfall/v3_registry.json``; a second invocation (even with a
different ``--out-root``) is refused, and a crash leaves a durable ``crashed``
state that also refuses a fresh freeze or recount.

The real Docker verifier is used.  ``--fake-docker-fixture`` is intentionally
not provided: fixture coverage lives in the CPU test suite and injects a fake
``grade_fn`` through :func:`gemmaloop.stage0.shortfall.run_v3_pipeline`.
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
from gemmaloop.stage0 import modal_lite  # noqa: E402
from gemmaloop.stage0.grader import DockerSandbox  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    LITE_IMAGE_TAG,
    _docker_recovery_check,
    resolve_docker_image,
    resume_v3_pipeline,
    run_v3_pipeline,
)
from gemmaloop.pilot.eval_rows import render_trace_rows  # noqa: E402


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
        "--out-root", type=Path, default=shortfall / "v3"
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
            "resume the claimed official run from verified completed artifacts "
            "and the durable attempt ledger; never claim or re-freeze"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pin = resolve_docker_image(args.image)
    DockerSandbox(image=args.image, expected_digest=pin["pinned"]).verify_image_digest()
    tokenizer = modal_lite.load_tokenizer()
    pipeline = resume_v3_pipeline if args.resume else run_v3_pipeline
    result = pipeline(
        repo_root=args.repo_root,
        pool_dir=args.pool_dir,
        lite_dir=args.lite_dir,
        request_ids_path=args.request_ids,
        out_root=args.out_root,
        image=args.image,
        image_pin=pin,
        workers=max(1, int(args.workers)),
        recovery_check=_docker_recovery_check,
        tokenizer=tokenizer,
        renderer=render_trace_rows,
        max_length=int(args.max_length),
        allow_pending=not args.no_pending,
    )
    summary = result.pop("count_summary")
    print(
        json.dumps(
            {
                **result,
                "nT0": summary["nT0"],
                "nS0": summary["nS0"],
                "pilot_p1": summary["pilot_p1"],
                "extension_required": summary["extension_required"],
                "extension_decision": summary["extension_decision"],
                "infrastructure_failures": summary["infrastructure_failures"],
                "blocking_infrastructure_failures": summary.get(
                    "blocking_infrastructure_failures"
                ),
                "terminal_infrastructure_failures": summary.get(
                    "terminal_infrastructure_failures"
                ),
                "terminal_infrastructure": summary.get("terminal_infrastructure"),
                "accounting": summary["accounting"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
