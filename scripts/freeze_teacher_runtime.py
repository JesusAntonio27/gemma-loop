"""Build the pinned qualified teacher-runtime record from qualification JSON.

The pilot harness reads this record and passes its runtime to every teacher
precompute.  A teacher whose supplied qualification decision is not ``pass`` is
refused (exit 2), so the record can never bind E4B bf16, 12B NF4 or an
unimplemented fallback.

Real E4B build (the only teacher with a qualified runtime today)::

    uv run python scripts/freeze_teacher_runtime.py \
        --qualification "C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\qualification\\e4b-int8-r2\\qualification.json" \
        --out "C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\qualification\\teacher_runtime_freeze.json"

12B has no qualified runtime (NF4 failed fidelity and q4_0_gguf is not
implemented); supplying its qualification file is refused, which is the
intended fail-closed behaviour.
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

from gemmaloop.teacher.runtime_freeze import (  # noqa: E402
    TeacherRuntimeFreezeError,
    build_teacher_runtime_freeze,
    default_teacher_runtime_freeze_path,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification",
        action="append",
        type=Path,
        required=True,
        help="qualification.json for one teacher runtime (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output freeze path; defaults to "
            "<artifacts>/qualification/teacher_runtime_freeze.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out = args.out or default_teacher_runtime_freeze_path()
    try:
        freeze = build_teacher_runtime_freeze(args.qualification, out=out)
    except TeacherRuntimeFreezeError as exc:
        print(f"teacher-runtime freeze refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(freeze.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
