"""Write the real, immutable v3 void receipt and the v3r continuity record.

This binds the preserved crashed v3 registry, verifier freeze, count manifest,
attempt ledger and the X1 ruling by file-byte SHA-256.  It refuses on any
mismatch and never overwrites.  It does **not** claim v3r and does not record a
replacement review.

Usage::

    uv run python scripts/record_v3_void.py \
        --ruling /path/to/codex-ruling-v3-void.md
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
from gemmaloop.stage0 import shortfall  # noqa: E402

DEFAULT_RULING = (
    Path.home()
    / "ML"
    / "GemmaLoop"
    / ".superpowers"
    / "sdd"
    / "2026-11-03-gemma-loop"
    / shortfall.VOID_RULING_NAME
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lite-dir",
        type=Path,
        default=artifacts_dir() / "stage0" / "modal_lite",
    )
    parser.add_argument("--ruling", type=Path, default=DEFAULT_RULING)
    args = parser.parse_args(argv)
    receipt = shortfall.write_void_receipt(
        args.lite_dir, ruling_path=args.ruling, enforce_known_hashes=True
    )
    continuity = shortfall.write_v3r_continuity(
        args.lite_dir, ruling_path=args.ruling
    )
    print(
        json.dumps(
            {
                "void_receipt": str(
                    shortfall.void_receipt_path(args.lite_dir)
                ),
                "void_receipt_sha256": shortfall._sha256_file(
                    shortfall.void_receipt_path(args.lite_dir)
                ),
                "void_receipt_payload_sha256": receipt["receipt_sha256"],
                "continuity": str(
                    shortfall.v3r_continuity_path(args.lite_dir)
                ),
                "continuity_sha256": shortfall._sha256_file(
                    shortfall.v3r_continuity_path(args.lite_dir)
                ),
                "continuity_payload_sha256": continuity["continuity_sha256"],
                "replacement_review_status": continuity["replacement_review"][
                    "status"
                ],
                "v3r_registry_status": continuity["v3r_registry"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
