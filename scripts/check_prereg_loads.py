"""Check that a prereg YAML file loads through the strict ``Prereg`` model.

Release gate for the tag: run this on the exact file that will be frozen
before creating ``prereg-v1``.

    uv run python scripts/check_prereg_loads.py prereg/prereg.yaml
    uv run python scripts/check_prereg_loads.py prereg/prereg.yaml --tag-mode

Plain mode loads and validates the file.  ``--tag-mode`` additionally runs the
full tag gate (``assert_tag_ready``): any ``PIN-PENDING`` or
``ALIGN-WITH-CODE`` placeholder anywhere in the document exits 2.

Exit 0 when the file loads and validates (and, in tag mode, the tag gate
passes); exit 2 otherwise (with the pydantic error on stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.prereg import assert_tag_ready, load_prereg  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="prereg YAML to validate")
    parser.add_argument(
        "--tag-mode",
        action="store_true",
        help=(
            "also run the full tag gate: refuse any PIN-PENDING or "
            "ALIGN-WITH-CODE placeholder"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.path)
    if not path.is_file():
        print(f"error: prereg not found: {path}", file=sys.stderr)
        return 2
    try:
        prereg = load_prereg(path)
        if args.tag_mode:
            assert_tag_ready(prereg)
    except Exception as exc:  # noqa: BLE001 - any validation error is fatal
        print(f"error: {path} did not load: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.tag_mode:
        print(f"ok (tag mode): {path}")
    else:
        print(f"ok: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
