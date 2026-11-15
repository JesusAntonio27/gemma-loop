"""Build the final pilot corpus and the shortfall freeze index.

The realized count was ``no_extension_original_only``.  This registration build
keeps every eligible original trace, orders by
``sha256(UTF8("0:"+request_id))`` and counts distinct canonical problem ids.
The conditional extension branch is not implemented here and any top-up input
is refused.  ``--allow-pending`` may write the index while the prereg amendment
file is absent (the index is then marked pending and the execution guard
rejects it); it never marks a complete index pending.
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
from gemmaloop.stage0 import local  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    EXTENSION_NOT_IMPLEMENTED,
    FREEZE_MANIFEST_NAME,
    _load_grades_by_request,
    _load_records_by_request,
    build_eligible_traces,
    load_freeze_manifest,
    load_verified_count,
    write_final_corpus,
    write_freeze_index,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shortfall = artifacts_dir() / "stage0" / "modal_lite" / "shortfall"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=shortfall / "verifier" / FREEZE_MANIFEST_NAME,
    )
    parser.add_argument("--count-summary", type=Path, default=shortfall / "count" / "summary.json")
    parser.add_argument("--topup-plan", type=Path, default=shortfall / "pilot_topup_plan.json")
    parser.add_argument("--out", type=Path, default=shortfall / "corpus")
    parser.add_argument(
        "--index-out",
        type=Path,
        default=shortfall / "pilot-shortfall-freeze.json",
    )
    parser.add_argument("--topup-per-request", type=Path, default=None)
    parser.add_argument("--topup-dir", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--run-version",
        default="v3",
        help=(
            "count/freeze version the index declares; operative readiness "
            "accepts only the authorized v3r run"
        ),
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help=(
            "write the index while the prereg amendment file is absent; the "
            "index is marked pending-data-addendum and the execution guard "
            "rejects it"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.topup_per_request is not None or args.topup_dir is not None:
        raise SystemExit(EXTENSION_NOT_IMPLEMENTED)
    freeze = load_freeze_manifest(args.freeze)
    summary = load_verified_count(
        args.count_summary,
        topup_plan_path=args.topup_plan,
        freeze_path=args.freeze,
    )
    if bool(summary.get("extension_required")) or str(
        summary.get("extension_decision")
    ) != "no_extension_original_only":
        raise SystemExit(
            f"{EXTENSION_NOT_IMPLEMENTED}; the realized decision must be "
            "no_extension_original_only"
        )
    per_request = Path(args.count_summary).parent / "per_request.jsonl"
    entries = local.read_jsonl(per_request)
    verified_ids = {
        str(entry["request_id"]) for entry in entries if entry.get("verified")
    }
    lite_dir = Path(freeze["inputs"]["lite_dir"])
    records = _load_records_by_request(lite_dir)
    grades = _load_grades_by_request(lite_dir)
    pool_dir = Path(freeze["inputs"]["pool_dir"])
    traces = build_eligible_traces(
        freeze=freeze,
        pool_dir=pool_dir,
        records=records,
        grade_rows=grades,
        verified_request_ids=verified_ids,
    )

    tokenizer = None
    if args.render:
        from gemmaloop.stage0 import modal_lite  # noqa: PLC0415

        tokenizer = modal_lite.load_tokenizer()
    manifest = write_final_corpus(
        args.out,
        traces,
        tokenizer=tokenizer,
        renderer=render_trace_rows if tokenizer is not None else None,
        max_length=int(args.max_length),
    )
    index = write_freeze_index(
        repo_root=args.repo_root,
        freeze_path=args.freeze,
        count_summary_path=args.count_summary,
        topup_plan_path=args.topup_plan,
        out_path=args.index_out,
        final_manifest_path=Path(args.out) / "manifest.json",
        allow_pending=bool(args.allow_pending),
        tokenizer=tokenizer,
        renderer=render_trace_rows if tokenizer is not None else None,
        max_length=int(args.max_length),
        run_version=str(args.run_version),
    )
    print(
        json.dumps(
            {
                "corpus": str(Path(args.out).resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "index": str(Path(args.index_out).resolve()),
                "index_sha256": index["index_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
