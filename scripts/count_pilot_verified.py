"""Official verified math+code count for the three original pilot input sets.

Usage::

    uv run python scripts/count_pilot_verified.py \
        --freeze "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/verifier/verifier_freeze.json" \
        --topup-plan "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/pilot_topup_plan.json" \
        --out "$GEMMALOOP_ARTIFACTS/stage0/modal_lite/shortfall/count"

Every kept original code sample is graded in the frozen CPU-only Docker
sandbox.  The summary contains the per-request accounting, ``nT0``/``nS0``,
the descriptive-only P1 count and the extension Boolean.  The precommitted
top-up plan must already exist; this script refuses to decide without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.paths import artifacts_dir  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    COUNT_SUMMARY_NAME,
    _docker_recovery_check,
    count_verified,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stage0 = artifacts_dir() / "stage0"
    lite = stage0 / "modal_lite"
    shortfall = lite / "shortfall"
    parser = argparse.ArgumentParser(description=__doc__)
    # The count consumes the pool/lite paths recorded in the verified freeze.
    # These remain explicit only so a mismatched path is refused, never used.
    parser.add_argument("--pool-dir", type=Path, default=None)
    parser.add_argument("--lite-dir", type=Path, default=None)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=shortfall / "verifier" / "verifier_freeze.json",
    )
    parser.add_argument(
        "--topup-plan", type=Path, default=shortfall / "pilot_topup_plan.json"
    )
    parser.add_argument(
        "--interpretation",
        type=Path,
        default=shortfall / "interpretation" / "interpretation_manifest.json",
    )
    parser.add_argument("--out", type=Path, default=shortfall / "count")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = count_verified(
        freeze_path=args.freeze,
        interpretation_path=args.interpretation,
        pool_dir=args.pool_dir,
        lite_dir=args.lite_dir,
        topup_plan_path=args.topup_plan,
        out_dir=args.out,
        workers=max(1, int(args.workers)),
        max_infrastructure_retries=1,
        recovery_check=_docker_recovery_check,
    )
    print(
        json.dumps(
            {
                "summary": str((args.out / COUNT_SUMMARY_NAME).resolve()),
                "status": summary["status"],
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
                "per_split": {
                    name: {
                        "math": value["math_count"],
                        "code": value["code_count"],
                        "union": value["union_count"],
                    }
                    for name, value in summary["per_split"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
