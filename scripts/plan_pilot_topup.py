"""Write the complete potential pilot top-up request/seed plan.

This must run after the verifier freeze and *before* the official count makes
the extension decision.  The plan always lists every potential request (sample
indices 4-7 for every original T-fit/selection math/code problem); no P1 and no
chat requests are ever planned.
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
from gemmaloop.stage0.shortfall import plan_topup  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shortfall = artifacts_dir() / "stage0" / "modal_lite" / "shortfall"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=shortfall / "verifier" / "verifier_freeze.json",
    )
    parser.add_argument(
        "--out", type=Path, default=shortfall / "pilot_topup_plan.json"
    )
    parser.add_argument("--parent-run-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = plan_topup(
        args.freeze, args.out, parent_run_id=args.parent_run_id
    )
    print(
        json.dumps(
            {
                "plan": str(Path(args.out).resolve()),
                "plan_sha256": payload["plan_sha256"],
                "topup_run_id": payload["topup_run_id"],
                "request_count": payload["request_count"],
                "problem_count": payload["problem_count"],
                "requests_by_split": payload["requests_by_split"],
                "requests_by_kind": payload["requests_by_kind"],
                "timeout_seconds": payload["timeout_seconds"],
                "run_cap_usd": payload["run_cap_usd"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
