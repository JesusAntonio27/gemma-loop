"""Write Stage 0 hidden tests with the official DeepSeek API.

Every request is off-peak, uses the pinned ``deepseek-flash`` model, explicit
non-thinking mode and the shared monthly spend ledger.  The key is read from
``DEEPSEEK_API_KEY`` only.  A dry run counts tokens and estimates cost without
constructing a client::

    uv run python scripts/write_stage0_hidden_tests.py \
        --pool-dir artifacts/stage0/pool --dry-run --max-usd 12

Real runs are paid pilots.  The first invocation processes at most
``--pilot`` items (default 20), writes ``<output>.pilot.json`` and stops::

    set DEEPSEEK_API_KEY=...   # Windows PowerShell: $env:DEEPSEEK_API_KEY="..."
    uv run python scripts/write_stage0_hidden_tests.py \
        --live --pool-dir artifacts/stage0/pool \
        --output artifacts/stage0/hidden_tests.jsonl \
        --max-usd 12 --sandbox docker --init-ledger

The rest needs a second invocation with ``--after-pilot <report>``; it refuses
when the report's projected full-run cost exceeds ``--max-usd``::

    uv run python scripts/write_stage0_hidden_tests.py \
        --live --after-pilot artifacts/stage0/hidden_tests.jsonl.pilot.json \
        --pool-dir artifacts/stage0/pool \
        --output artifacts/stage0/hidden_tests.jsonl \
        --max-usd 12 --sandbox docker

Any ``opencode.ai`` endpoint is refused.  Tests that the reference solution
fails are dropped, and the output is shared by both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from gemmaloop.deepseek_money import (
    API_MODEL,
    OFFICIAL_BASE_URL,
    PILOT_DEFAULT_ITEMS,
    PILOT_MAX_ITEMS,
    PRICES_PER_MILLION,
    PilotReportError,
    WasteTripAcknowledgmentRequired,
    build_pilot_report,
    check_pilot_projection,
    load_pilot_report,
    pilot_config_sha256,
    read_journal,
    response_journal_path,
    summarize_pilot_journal,
    validate_pilot_report,
    write_pilot_report,
)
from gemmaloop.stage0.deepseek import HiddenTestWriter
from gemmaloop.stage0.grader import GraderService
from gemmaloop.stage0.spend import (
    SpendLedger,
    default_ledger_path,
    month_key,
    require_spend_lock,
)

LOGGER = logging.getLogger("write_stage0_hidden_tests")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(tmp, path)
    return digest.hexdigest()


def _default_pilot_report_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".pilot.json")


def _pool_manifest_sha256(pool_dir: Path) -> str:
    manifest = pool_dir / "manifest.json"
    target = manifest if manifest.exists() else pool_dir / "problems.jsonl"
    if not target.exists():
        return ""
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _pilot_config(
    *,
    pool_manifest_sha256: str,
    candidate_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "pool_manifest_sha256": pool_manifest_sha256,
        "candidate_ids": list(candidate_ids),
        "limit": args.limit,
        "max_output_tokens": int(args.max_output_tokens),
        "model": args.model,
        "thinking": "disabled",
        "base_url": args.base_url,
        "prices_per_million": dict(PRICES_PER_MILLION),
        "provider": "deepseek",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=Path("artifacts/stage0/pool"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage0/hidden_tests.jsonl"))
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--min-existing-tests", type=int, default=2,
                        help="only problems with fewer tests are sent to the writer")
    parser.add_argument("--all-code", action="store_true",
                        help="send every code problem, not only weakly tested ones")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true",
                        help="make real API requests (default is a dry run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="explicit dry run (the default)")
    parser.add_argument("--pilot", type=int, default=PILOT_DEFAULT_ITEMS,
                        help="maximum items in the paid pilot (default 20)")
    parser.add_argument("--after-pilot", type=Path, default=None,
                        help="continue after this pilot report (projected cost must fit --max-usd)")
    parser.add_argument("--pilot-report", type=Path, default=None,
                        help="where the pilot writes its JSON report")
    parser.add_argument("--ack-waste-trip", default=None,
                        help="acknowledge a recorded waste trip and start a new window")
    parser.add_argument("--ack-pilot-event", action="append", default=None,
                        help="acknowledge a recorded pilot event (repeatable)")
    parser.add_argument("--repin", action="store_true",
                        help="deliberately replace the fingerprint registry record")
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--base-url", default=OFFICIAL_BASE_URL)
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker",
                        help="reference-solution validation sandbox (Docker required by default)")
    parser.add_argument("--allow-unsandboxed", action="store_true",
                        help="explicitly allow the NON-SANDBOXED local fallback")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--spend-lock", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--init-ledger", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if args.live and args.dry_run:
        raise SystemExit("--live and --dry-run are mutually exclusive")
    if not 1 <= args.pilot <= PILOT_MAX_ITEMS:
        raise SystemExit(
            f"--pilot must be between 1 and {PILOT_MAX_ITEMS}; got {args.pilot}"
        )
    dry_run = not args.live
    problems = _read_jsonl(args.pool_dir / "problems.jsonl")
    candidates = [
        problem
        for problem in problems
        if problem.get("kind") == "code"
        and (
            args.all_code
            or len(problem.get("tests") or []) < args.min_existing_tests
        )
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    if args.sandbox == "local" and not args.allow_unsandboxed:
        raise SystemExit("refusing --sandbox local without --allow-unsandboxed")

    if not dry_run:
        # One money-spending process at a time; held until process exit.
        require_spend_lock(args.spend_lock)

    pool_sha = _pool_manifest_sha256(args.pool_dir)
    candidate_ids = [str(problem.get("problem_id")) for problem in candidates]
    config = _pilot_config(
        pool_manifest_sha256=pool_sha,
        candidate_ids=candidate_ids,
        args=args,
    )
    config_sha = pilot_config_sha256(config)

    # A live run is a pilot unless it is explicitly continuing one.
    pilot_mode = (not dry_run) and args.after_pilot is None
    report_path = args.pilot_report or _default_pilot_report_path(args.output)
    planned_candidates = candidates[: args.pilot]
    planned_ids = [str(problem.get("problem_id")) for problem in planned_candidates]
    partial_path = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    journal_path = response_journal_path(partial_path)

    resuming_report = False
    if pilot_mode:
        run_candidates = planned_candidates
        if report_path.exists():
            existing = load_pilot_report(report_path)
            try:
                validate_pilot_report(
                    existing,
                    max_usd=None,
                    expected_config_sha256=config_sha,
                    expected_candidate_ids=candidate_ids,
                    require_complete=False,
                )
            except PilotReportError as exc:
                raise SystemExit(str(exc)) from exc
            summary = summarize_pilot_journal(
                read_journal(journal_path), planned_ids
            )
            unacked = [
                trip
                for trip in summary["waste_trip_ids"]
                if trip not in summary["acked_waste_trip_ids"]
            ]
            if (
                summary["all_accounted"]
                and not unacked
                and not summary["unacked_pilot_event_ids"]
            ):
                raise SystemExit(
                    f"pilot report {report_path} is already complete; refuse "
                    "to overwrite it (use a new --output/--pilot-report)"
                )
            resuming_report = True
    else:
        run_candidates = candidates

    if not dry_run and args.after_pilot is not None:
        report = load_pilot_report(args.after_pilot)
        check_pilot_projection(
            report,
            args.max_usd,
            expected_config_sha256=config_sha,
            expected_candidate_ids=candidate_ids,
            ack_waste_trip=args.ack_waste_trip,
            ack_pilot_events=args.ack_pilot_event,
        )
        if args.ack_pilot_event or args.ack_waste_trip:
            acked_events = {
                str(item) for item in (report.get("acked_pilot_event_ids") or [])
            }
            acked_waste = {
                str(item) for item in (report.get("acked_waste_trip_ids") or [])
            }
            for event_id in args.ack_pilot_event or []:
                acked_events.add(str(event_id))
            if args.ack_waste_trip:
                acked_waste.add(str(args.ack_waste_trip))
                acked_events.add(str(args.ack_waste_trip))
            event_ids = {
                str(event.get("event_id"))
                for event in (report.get("pilot_events") or [])
                if event.get("event_id")
            } | {str(item) for item in (report.get("waste_trip_ids") or [])}
            trip_ids = {
                str(item) for item in (report.get("waste_trip_ids") or [])
            }
            unacked_events = sorted(event_ids - acked_events)
            acknowledged_trips = acked_waste | acked_events
            report["acked_pilot_event_ids"] = sorted(acked_events)
            report["acked_waste_trip_ids"] = sorted(acked_waste)
            report["unacked_pilot_event_ids"] = unacked_events
            report["events_acknowledged"] = bool(event_ids) and not unacked_events
            report["waste_acknowledged"] = bool(trip_ids) and trip_ids <= acknowledged_trips
            if (
                not unacked_events
                and (not trip_ids or trip_ids <= acknowledged_trips)
                and report.get("stopped_reason") in ("pilot_events", "waste_guard")
            ):
                report["stopped_reason"] = None
            write_pilot_report(args.after_pilot, report, overwrite=True)
    ledger = None
    ledger_month = None
    if not dry_run:
        ledger = SpendLedger(args.ledger or default_ledger_path())
        if not ledger.path.exists() and not args.init_ledger:
            raise SystemExit(
                f"refusing: spend ledger {ledger.path} is missing; pass --init-ledger "
                "for the first live run"
            )
        ledger_month = month_key()

    grader = None
    if not dry_run:
        grader = GraderService(
            workers=1,
            timeout=args.timeout,
            sandbox=args.sandbox,
            allow_fallback=args.allow_unsandboxed,
        )
    try:
        writer = HiddenTestWriter(
            base_url=args.base_url,
            model=args.model,
            max_usd=args.max_usd,
            max_output_tokens=args.max_output_tokens,
            dry_run=dry_run,
            grader=grader,
            ledger=ledger,
            ledger_month=ledger_month,
            partial_path=None if dry_run else partial_path,
            ack_waste_trip=args.ack_waste_trip,
            ack_pilot_events=args.ack_pilot_event,
            repin=args.repin,
        )
        try:
            result = writer.write(run_candidates)
        except WasteTripAcknowledgmentRequired as exc:
            raise SystemExit(str(exc)) from exc
    finally:
        if grader is not None:
            grader.close()

    if dry_run:
        LOGGER.info("dry run: %s", json.dumps(result.estimated, sort_keys=True))
        print(json.dumps(result.estimated, indent=2, sort_keys=True))
        return 0

    if pilot_mode:
        summary = summarize_pilot_journal(read_journal(journal_path), planned_ids)
        month_to_date = max(0.0, float(result.spent_usd) - float(summary["usd"]))
        unacked_events = summary["unacked_pilot_event_ids"]
        if summary["all_accounted"] and not unacked_events:
            stopped_reason = None
        elif summary["all_accounted"]:
            stopped_reason = "pilot_events"
        else:
            stopped_reason = result.stopped_reason or "incomplete"
        report = build_pilot_report(
            requests=summary["requests"],
            input_miss_tokens=summary["input_miss_tokens"],
            input_hit_tokens=summary["input_hit_tokens"],
            output_tokens=summary["output_tokens"],
            waste_count=summary["waste_count"],
            usd=summary["usd"],
            pilot_items=len(planned_ids),
            total_items=len(candidates),
            stopped_reason=stopped_reason,
            month_to_date_usd=month_to_date,
            config=config,
            candidate_ids=candidate_ids,
            pilot_candidate_ids=planned_ids,
            planned_candidate_ids=planned_ids,
            accounted_candidate_ids=summary["accounted_candidate_ids"],
            waste_acknowledged=summary["waste_acknowledged"],
            waste_trip_ids=summary["waste_trip_ids"],
            acked_waste_trip_ids=summary["acked_waste_trip_ids"],
            waste_trip_id=summary["latest_waste_trip_id"],
            pilot_events=summary["pilot_events"],
            acked_pilot_event_ids=summary["acked_pilot_event_ids"],
            unacked_pilot_event_ids=summary["unacked_pilot_event_ids"],
            records_written=summary["records_written"],
            parse_failures=summary["parse_failures"],
            dropped_tests=summary["dropped_tests"],
            limit=args.limit,
            pool_manifest_sha256=pool_sha,
            model=args.model,
        )
        report["stage"] = "stage0_hidden_tests_pilot"
        report["partial_output"] = str(partial_path)
        try:
            write_pilot_report(
                report_path, report, overwrite=resuming_report
            )
        except PilotReportError as exc:
            raise SystemExit(str(exc)) from exc
        LOGGER.info("paid pilot stopped after %d items; report %s", len(run_candidates), report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    all_records = result.records
    if partial_path.exists():
        all_records = []
        with partial_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    all_records.append(json.loads(line))
    sha256 = _write_jsonl(args.output, all_records)
    manifest = {
        "stage": "stage0_hidden_tests",
        "problems_considered": len(candidates),
        "records": len(all_records),
        "resumed": result.resumed,
        "dropped_tests": result.dropped_tests,
        "failed_problems": result.failed_problems,
        "calls": result.calls,
        "spent_usd": result.spent_usd,
        "run_usd": result.run_usd,
        "stopped_reason": result.stopped_reason,
        "waste": result.waste,
        "input_miss_tokens": result.input_miss_tokens,
        "input_hit_tokens": result.input_hit_tokens,
        "output_tokens": result.output_tokens,
        "response_models": sorted({str(value) for value in result.response_models}),
        "system_fingerprints": sorted(
            {str(value) for value in result.system_fingerprints if value is not None}
        ),
        "live": not dry_run,
        "estimated": result.estimated,
        "output": str(args.output),
        "output_sha256": sha256,
        "shared_by_arms": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("wrote %d records to %s", len(all_records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
