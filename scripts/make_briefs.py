"""Validate and freeze the creative evaluation briefs.

The 150 briefs (60 % pages, 25 % components, 15 % SVG) are frozen by the
SHA-256 of this JSONL file before any creative run.  Generation through the
official DeepSeek API is deliberately **not implemented yet**: it waits for
the author's provider and spend decision.  Until then the author writes the
JSONL by hand (or accepts a generated file) and this script validates it and
prints the hash to record in ``prereg.yaml`` (``creative.briefs_sha256``).

Usage::

    uv run python scripts/make_briefs.py \
        --briefs data/creative/briefs_eval.jsonl --check-mix --expect 150

Exit status is 0 when the file is valid (and, with ``--check-mix``, the
60/25/15 mix is within one percentage point) and 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BRIEF_KEYS = (
    "brief_id",
    "kind",
    "brief",
    "required_sections",
    "required_copy",
    "forbidden",
)
KINDS = ("page", "component", "svg")
MIX = {"page": 0.60, "component": 0.25, "svg": 0.15}
MIX_TOLERANCE = 0.01
EXPECTED_BRIEFS = 150


class BriefError(ValueError):
    """The briefs file does not follow the frozen format."""


def validate_brief(record: object, position: int) -> None:
    """Validate one brief; ``position`` is only used in the error message."""
    if not isinstance(record, dict):
        raise BriefError(f"brief {position}: must be a JSON object")

    missing = [key for key in BRIEF_KEYS if key not in record]
    extra = [key for key in record if key not in BRIEF_KEYS]
    if missing:
        raise BriefError(f"brief {position}: missing field(s) {', '.join(missing)}")
    if extra:
        raise BriefError(f"brief {position}: unexpected field(s) {', '.join(extra)}")

    brief_id = record["brief_id"]
    if not isinstance(brief_id, str) or not brief_id.strip():
        raise BriefError(f"brief {position}: brief_id must be a non-empty string")
    if record["kind"] not in KINDS:
        raise BriefError(
            f"brief {position}: kind must be one of {', '.join(KINDS)}"
        )
    if not isinstance(record["brief"], str) or not record["brief"].strip():
        raise BriefError(f"brief {position}: brief must be a non-empty string")
    for key in ("required_sections", "required_copy", "forbidden"):
        value = record[key]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise BriefError(
                f"brief {position}: {key} must be a list of non-empty strings"
            )


def validate_briefs(records: list[dict]) -> dict:
    """Validate the whole set and return ``{"n", "kinds"}`` counts."""
    records = list(records)
    if not records:
        raise BriefError("no briefs")

    seen: set[str] = set()
    kinds = {kind: 0 for kind in KINDS}
    for position, record in enumerate(records, start=1):
        validate_brief(record, position)
        brief_id = record["brief_id"]
        if brief_id in seen:
            raise BriefError(f"brief {position}: duplicate brief_id {brief_id!r}")
        seen.add(brief_id)
        kinds[record["kind"]] += 1

    return {"n": len(records), "kinds": kinds}


def check_mix(
    records: list[dict], tolerance: float = MIX_TOLERANCE
) -> dict[str, int]:
    """Enforce the 60/25/15 page/component/SVG mix within ``tolerance``."""
    records = list(records)
    if not records:
        raise BriefError("no briefs")

    counts = {kind: 0 for kind in KINDS}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("kind") not in KINDS:
            raise BriefError(f"brief {position}: unknown kind")
        counts[record["kind"]] += 1

    total = len(records)
    for kind, expected in MIX.items():
        share = counts[kind] / total
        if abs(share - expected) > tolerance:
            raise BriefError(
                f"{kind} share is {share:.1%}, expected {expected:.0%} "
                f"+/- {tolerance:.0%}"
            )
    return counts


def hash_file(path: str | Path) -> str:
    """SHA-256 of the raw file bytes (the frozen value goes in prereg)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_briefs(path: str | Path) -> list[dict]:
    """Parse the JSONL, validating the schema line by line."""
    path = Path(path)
    records: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BriefError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from None
        try:
            validate_brief(record, line_number)
        except BriefError as exc:
            raise BriefError(f"{path}:{line_number}: {exc}") from None
        records.append(record)

    if not records:
        raise BriefError(f"{path} contains no briefs")
    validate_briefs(records)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--briefs", required=True, type=Path, help="briefs JSONL")
    parser.add_argument(
        "--check-mix",
        action="store_true",
        help="require the 60/25/15 page/component/SVG mix",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help=(
            "require exactly this many briefs "
            f"(the frozen set uses {EXPECTED_BRIEFS})"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = load_briefs(args.briefs)
        summary = validate_briefs(records)
        if args.expect is not None and summary["n"] != args.expect:
            raise BriefError(
                f"expected {args.expect} briefs, found {summary['n']}"
            )
        if args.check_mix:
            check_mix(records)
    except (OSError, BriefError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    digest = hash_file(args.briefs)
    if args.json:
        print(
            json.dumps(
                {
                    "n": summary["n"],
                    "kinds": summary["kinds"],
                    "sha256": digest,
                    "mix_checked": args.check_mix,
                },
                ensure_ascii=False,
            )
        )
    else:
        counts = summary["kinds"]
        print(f"briefs: {summary['n']}")
        for kind in KINDS:
            share = counts[kind] / summary["n"]
            print(f"  {kind}: {counts[kind]} ({share:.1%})")
        if args.check_mix:
            print("mix: ok (60/25/15)")
        print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
