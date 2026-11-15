"""Print and persist the pre-registered S1-S5 verdicts.

Examples::

    uv run python scripts/check_criteria.py \
        --size e2b \
        --prereg prereg/prereg.yaml \
        --results artifacts/eval/e2b \
        --out artifacts/report/criteria_e2b.json

    uv run python scripts/check_criteria.py \
        --size all \
        --prereg prereg/prereg.yaml \
        --results artifacts/eval \
        --creative artifacts/report/creative_phase1.json

``--results`` accepts either one ``RunResult`` JSON file or a directory (all
``*.json`` files are scanned; files that are not run results are skipped).
``--size all`` evaluates each model size separately and writes one verdict list
per size; it never pools E2B and E4B.

The exit status is 1 when any emitted verdict has ``passed is False`` (``None``
verdicts do not fail the command).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gemmaloop.eval.results import RunResult
from gemmaloop.prereg import (
    AgentSummary,
    CreativeSummary,
    Verdict,
    check_criteria,
    load_prereg,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("prereg/prereg.yaml"),
        help="path to prereg.yaml (default: prereg/prereg.yaml)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="RunResult JSON file or directory of RunResult JSON files",
    )
    parser.add_argument(
        "--size",
        required=True,
        choices=["e2b", "e4b", "all"],
        help="model size to evaluate; 'all' loops over the sizes present",
    )
    parser.add_argument(
        "--creative",
        type=Path,
        help="CreativeSummary JSON (a single object or a name -> object map)",
    )
    parser.add_argument(
        "--agent",
        type=Path,
        help="AgentSummary JSON (a single object or a name -> object map)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output JSON path (default: <results>/criteria.json)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="only pool runs with this prereg tag (default: prereg.yaml tag; '' disables)",
    )
    parser.add_argument(
        "--require-seeds",
        dest="require_seeds",
        action="store_true",
        default=True,
        help=(
            "fail unless every registered O-H/L training seed is present "
            "(default)"
        ),
    )
    parser.add_argument(
        "--allow-partial-seeds",
        dest="require_seeds",
        action="store_false",
        help="relax the seed check for smoke/exploratory runs only",
    )
    return parser.parse_args(argv)


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"results path does not exist: {path}")
    return sorted(path.rglob("*.json"))


def _load_results(path: Path) -> list[RunResult]:
    loaded: list[RunResult] = []
    for file in _result_files(path):
        if file.name.startswith("criteria"):
            continue
        try:
            run = RunResult.load(file)
        except Exception as exc:  # noqa: BLE001 - mixed result directories
            print(f"warning: skipping {file}: {exc}", file=sys.stderr)
            continue
        loaded.append(run)
    if not loaded:
        raise SystemExit(f"no RunResult files found under {path}")
    return loaded


def _load_creative(path: Path | None) -> dict[str, CreativeSummary] | None:
    if path is None:
        return None
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "win_rate" in raw:
        return {"L": CreativeSummary.model_validate(raw)}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return {
        name: CreativeSummary.model_validate(value) for name, value in raw.items()
    }


def _load_agent(path: Path | None) -> dict[str, AgentSummary] | None:
    if path is None:
        return None
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "success_rate" in raw:
        return {"L+OC": AgentSummary.model_validate(raw)}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return {name: AgentSummary.model_validate(value) for name, value in raw.items()}


def _status(verdict: Verdict) -> str:
    if verdict.passed is True:
        return "PASS"
    if verdict.passed is False:
        return "FAIL"
    return "NONE"


def _print_verdicts(verdicts: list[Verdict]) -> None:
    for verdict in verdicts:
        print(f"{verdict.name:<28} {_status(verdict):<5} {verdict.detail}")


def main(argv: list[str] | None = None) -> int:
    print(
        "check_criteria: S1-S5 are retired in prereg-v1; the confirmatory "
        "family is E1 (precondition) -> E2 -> E3 -> E4.",
        file=sys.stderr,
    )
    return 2
    args = _parse_args(argv)
    prereg = load_prereg(args.prereg)
    results = _load_results(args.results)
    creative = _load_creative(args.creative)
    agent = _load_agent(args.agent)

    if args.size == "all":
        sizes = sorted({run.size.upper() for run in results})
        if not sizes:
            raise SystemExit("no model sizes found in results")
        payload: dict[str, list[dict]] = {}
        any_failed = False
        for size in sizes:
            verdicts = check_criteria(
                prereg,
                results,
                creative=creative,
                agent=agent,
                size=size,
                tag=args.tag,
                require_seeds=args.require_seeds,
            )
            print(f"=== {size} ===")
            _print_verdicts(verdicts)
            print()
            payload[size] = [verdict.model_dump() for verdict in verdicts]
            any_failed = any_failed or any(
                verdict.passed is False for verdict in verdicts
            )
        output: Any = payload
    else:
        verdicts = check_criteria(
            prereg,
            results,
            creative=creative,
            agent=agent,
            size=args.size,
            tag=args.tag,
            require_seeds=args.require_seeds,
        )
        _print_verdicts(verdicts)
        payload = [verdict.model_dump() for verdict in verdicts]
        any_failed = any(verdict.passed is False for verdict in verdicts)
        output = payload

    out = args.out
    if out is None:
        base = args.results if args.results.is_dir() else args.results.parent
        out = base / "criteria.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
