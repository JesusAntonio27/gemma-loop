"""Three-column baseline table for the Phase 0 runs.

The table compares Google's official Gemma 4 numbers with our reproduction of
the unmodified model (``O``) and, once available, the modified Gemma-Loop model
(``L`` at the primary loop count).  The last column applies the pre-registered
harness-validity rule from :mod:`gemmaloop.prereg`; it is not reimplemented
here.

Layout (markdown)::

    | Benchmark | Google (%) | Mi original (%) | Mi modificado (%) | Validez y Wilson 95 % |

Special cases required by the Task 15 ruling:

* Google publishes only one ambiguous "MMMU Pro" figure.  It is attached only
  to the derived ``MMMU-Pro Overall`` row, the **plain mean of our Standard(10)
  and Vision accuracies**; the two component rows get no Google figure.
* Every registered benchmark renders a row even without results, as dashes.
* The spatial benchmarks are exploratory and have no Google figure.

The command line reads every ``RunResult`` JSON below the artifacts results
directory and writes ``artifacts/report/three_column_<size>.md``::

    python -m gemmaloop.report.three_column --size e2b
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from gemmaloop.eval.results import RunResult
from gemmaloop.paths import artifacts_dir, results_dir
from gemmaloop.prereg import (
    HarnessCheck,
    default_prereg_path,
    duplicate_result_keys,
    harness_validity_from_accuracy,
    load_prereg,
    validate_single_engine,
)
from gemmaloop.report.benches import (
    BENCH_LABELS,
    BENCH_ORDER,
    EXPLORATORY_BENCHES,
    canonical_bench,
    canonical_variant,
)

EM_DASH = "\u2014"

MMMU_STANDARD = "mmmupro"
MMMU_VISION = "mmmupro_vision"
MMMU_OVERALL = "mmmu_pro_overall"

HEADER = (
    "| Benchmark | Google (%) | Mi original (%) | Mi modificado (%) | "
    "Validez y Wilson 95 % |"
)
SEPARATOR = "|---|---:|---:|---:|---|"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _scores(run: RunResult) -> dict[str, float]:
    """Per-item mean correctness, averaging every sample and seed first."""
    return run.per_item_mean()


def _merge_runs(runs: Sequence[RunResult]) -> RunResult:
    """Pool several result files of the same arm into one ``RunResult``.

    Items keep their ids, so :meth:`RunResult.per_item_mean` averages matching
    items across seeds (the Task 10 aggregation rule) while sample indices stay
    independent.
    """
    first = runs[0]
    metadata: dict = {}
    for run in runs:
        metadata.update(run.metadata)
    return RunResult(
        variant=first.variant,
        size=first.size,
        n_loops=first.n_loops,
        bench=first.bench,
        prereg_tag=first.prereg_tag,
        seed=first.seed,
        loop_block=first.loop_block,
        bridge=first.bridge,
        adapter_policy=first.adapter_policy,
        lora_target_layers=first.lora_target_layers,
        items=[item for run in runs for item in run.items],
        metadata=metadata,
    )


def _normalise(runs: Mapping[str, RunResult] | None) -> dict[str, RunResult]:
    """Canonicalise keys and merge duplicate aliases per benchmark."""
    if not runs:
        return {}
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for name, run in runs.items():
        grouped[canonical_bench(name)].append(run)
    return {bench: _merge_runs(group) for bench, group in grouped.items()}


def _row_point(
    key: str, runs: Mapping[str, RunResult]
) -> tuple[float | None, int]:
    """Return ``(accuracy, n_items)`` for a table row.

    MMMU-Pro Overall is the plain mean of the Standard(10) and Vision
    accuracies, exactly as the Task 15 ruling specifies.  When both settings
    share question ids, the mean is taken per shared item and Wilson's ``n`` is
    that shared count (the two configurations evaluate the same questions).
    For disjoint synthetic fixtures it falls back to the plain component means
    with the union size.
    """
    if key == MMMU_OVERALL:
        component_scores = []
        item_ids: list[set[str]] = []
        for bench in (MMMU_STANDARD, MMMU_VISION):
            run = runs.get(bench)
            if run is None:
                return None, 0
            scores = _scores(run)
            if not scores:
                return None, 0
            component_scores.append(scores)
            item_ids.append(set(scores))
        shared = item_ids[0] & item_ids[1]
        if shared:
            per_item = [
                (component_scores[0][item_id] + component_scores[1][item_id]) / 2
                for item_id in shared
            ]
            return _mean(per_item), len(shared)
        return (
            _mean([_mean(list(scores.values())) for scores in component_scores]),
            len(item_ids[0] | item_ids[1]),
        )

    run = runs.get(key)
    if run is None:
        return None, 0
    scores = _scores(run)
    if not scores:
        return None, 0
    return _mean(list(scores.values())), len(scores)


def _google_value(key: str, google: Mapping[str, float]) -> float | None:
    """Google's figure for a row, honouring the MMMU-Pro only-overall ruling."""
    if key in (MMMU_STANDARD, MMMU_VISION):
        return None
    if key in google:
        return google[key]
    if key == MMMU_OVERALL:
        standard = google.get(MMMU_STANDARD)
        vision = google.get(MMMU_VISION)
        if standard is not None and vision is not None:
            return (standard + vision) / 2.0
    return None


def _format_points(value: float | None) -> str:
    if value is None:
        return EM_DASH
    return f"{value * 100.0:.1f}"


def _format_google(value: float | None) -> str:
    if value is None:
        return EM_DASH
    return f"{value:.1f}"


def _wilson_suffix(check: HarnessCheck) -> list[str]:
    parts: list[str] = []
    if check.n:
        parts.append(f"n={check.n} items")
    if check.low is not None and check.high is not None:
        parts.append(
            f"Wilson95=[{check.low * 100.0:.1f}, {check.high * 100.0:.1f}]"
        )
    if check.low_power:
        parts.append("chequeo de baja potencia")
    return parts


def _format_validity(
    key: str, google: float | None, check: HarnessCheck
) -> str:
    if google is None:
        if key in EXPLORATORY_BENCHES:
            prefix = "exploratorio, sin cifra de Google"
        elif key in (MMMU_STANDARD, MMMU_VISION):
            prefix = "sin cifra de Google (solo Overall)"
        else:
            prefix = "sin cifra de Google"
        return "; ".join([prefix, *_wilson_suffix(check)])

    if check.accuracy is None:
        return "sin datos"
    status = "validado" if check.passed else "no validado"
    return "; ".join([status, *_wilson_suffix(check)])


def _ordered_keys(
    google: Mapping[str, float],
    original: Mapping[str, RunResult],
    modified: Mapping[str, RunResult],
    registered_benches: Sequence[str],
) -> list[str]:
    keys = set(registered_benches) | set(google) | set(original) | set(modified)
    if MMMU_STANDARD in keys and MMMU_VISION in keys:
        keys.add(MMMU_OVERALL)
    order = {bench: index for index, bench in enumerate(BENCH_ORDER)}
    return sorted(keys, key=lambda key: (order.get(key, len(order)), key))


def three_column_table(
    google: dict[str, float],
    mine_original: dict[str, RunResult],
    mine_modified: dict[str, RunResult] | None,
    tolerance: float,
    registered_benches: Iterable[str] | None = None,
) -> str:
    """Render the Google / mine-original / mine-modified markdown table.

    ``google`` maps canonical benchmark keys to official figures in percent.
    ``mine_original`` and ``mine_modified`` map the same keys to the
    corresponding ``RunResult``.  ``registered_benches`` defaults to every
    benchmark known to the report; every one of them gets a row, with dashes
    when a result is missing.  The validity column always refers to the
    original model, because it checks whether our evaluation harness reproduces
    Google's number.
    """
    original = _normalise(mine_original)
    modified = _normalise(mine_modified)
    google_map = {canonical_bench(name): float(value) for name, value in google.items()}
    registered = {
        canonical_bench(name) for name in (registered_benches or BENCH_ORDER)
    }

    lines = [HEADER, SEPARATOR]
    for key in _ordered_keys(google_map, original, modified, tuple(registered)):
        official = _google_value(key, google_map)
        original_accuracy, original_n = _row_point(key, original)
        modified_accuracy, _ = _row_point(key, modified)
        check = harness_validity_from_accuracy(
            official, original_accuracy, original_n, tolerance
        )

        lines.append(
            "| {label} | {google} | {original} | {modified} | {validity} |".format(
                label=BENCH_LABELS.get(key, key),
                google=_format_google(official),
                original=_format_points(original_accuracy),
                modified=_format_points(modified_accuracy),
                validity=_format_validity(key, official, check),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_results(root: str | Path) -> list[RunResult]:
    """Load every valid ``RunResult`` JSON below ``root``; warn on bad files."""
    root = Path(root)
    if not root.exists():
        return []
    results: list[RunResult] = []
    skipped: list[tuple[Path, Exception]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            results.append(RunResult.load(path))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            skipped.append((path, exc))
    if skipped:
        print(
            f"three_column: skipped {len(skipped)} invalid result file(s)",
            file=sys.stderr,
        )
        for path, exc in skipped[:10]:
            print(f"  {path}: {exc}", file=sys.stderr)
    return results


def _collect(
    results: Iterable[RunResult],
    size: str,
    variant: str,
    n_loops: set[int] | None = None,
    prereg_tag: str | None = None,
    geometry: tuple | None = None,
) -> dict[str, RunResult]:
    """Merge matching runs into one result per canonical benchmark.

    ``geometry`` pins the loop block/bridge/policy/target set.  Without it, a
    benchmark with more than one geometry is ambiguous and refused instead of
    being pooled.
    """
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for run in results:
        if run.size.strip().upper() != size.strip().upper():
            continue
        if canonical_variant(run.variant) != canonical_variant(variant):
            continue
        if n_loops is not None and run.n_loops not in n_loops:
            continue
        if prereg_tag is not None and run.prereg_tag != prereg_tag:
            continue
        if geometry is not None and run.geometry_key() != geometry:
            continue
        grouped[canonical_bench(run.bench)].append(run)

    selected: dict[str, RunResult] = {}
    for bench, runs in grouped.items():
        geometries = {run.geometry_key() for run in runs}
        if len(geometries) > 1:
            raise ValueError(
                f"{bench}: multiple loop geometries at the same arm/N; "
                "select one block/bridge/policy/target set instead of pooling "
                f"({sorted(map(str, geometries))})"
            )
        selected[bench] = _merge_runs(runs)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the three-column baseline table from RunResults."
    )
    parser.add_argument("--size", required=True, choices=("e2b", "e4b"))
    parser.add_argument(
        "--results-dir",
        default=None,
        help="result root (default: $GEMMALOOP_ARTIFACTS/results)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output file (default: $GEMMALOOP_ARTIFACTS/report/three_column_<size>.md)",
    )
    parser.add_argument(
        "--prereg",
        default=None,
        help="pre-registration YAML (default: the repository's prereg/prereg.yaml)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="harness-validity tolerance in points (default: prereg value)",
    )
    parser.add_argument(
        "--n-loops",
        type=int,
        default=None,
        help="loop count used for the modified column (default: prereg.primary_n)",
    )
    parser.add_argument(
        "--prereg-tag",
        default="prereg-v1",
        help="only pool runs with this tag; pass an empty string to disable",
    )
    parser.add_argument(
        "--loop-block",
        default=None,
        help="inclusive loop block for the modified arm (e.g. 16-18)",
    )
    parser.add_argument(
        "--bridge",
        default=None,
        choices=("scalar", "none", "damped", "gch"),
    )
    parser.add_argument(
        "--adapter-policy",
        default=None,
        choices=("always", "loop_only"),
    )
    parser.add_argument(
        "--lora-target-layers",
        default=None,
        choices=("language", "loop_block", "loop_block_coda"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    size = args.size.lower()
    prereg = load_prereg(args.prereg or default_prereg_path())
    google = {
        str(model_size).lower(): dict(figures)
        for model_size, figures in prereg.google.items()
    }.get(size, {})
    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else prereg.harness_tolerance_pts
    )
    prereg_tag = args.prereg_tag or None

    root = Path(args.results_dir) if args.results_dir else results_dir()
    results = _load_results(root)
    selected = [
        run
        for run in results
        if prereg_tag is None or run.prereg_tag == prereg_tag
    ]
    duplicates = duplicate_result_keys(selected)
    if duplicates:
        raise ValueError(
            "duplicate evaluation results for the same full identity "
            f"(variant, size, n_loops, bench, seed, geometry, engine): "
            f"{duplicates}"
        )
    validate_single_engine(selected, size=size)

    n_loops = args.n_loops if args.n_loops is not None else prereg.primary_n
    from gemmaloop.model.layer_plan import parse_block

    block = parse_block(args.loop_block) if args.loop_block else None
    geometry = None
    if any(
        value is not None
        for value in (
            block,
            args.bridge,
            args.adapter_policy,
            args.lora_target_layers,
        )
    ):
        geometry = (
            block,
            args.bridge,
            args.adapter_policy,
            args.lora_target_layers,
        )
    original = _collect(results, size, "O", {0}, prereg_tag)
    modified = _collect(
        results, size, "L", {n_loops}, prereg_tag, geometry=geometry
    )
    registered = [canonical_bench(bench) for bench in prereg.benches]

    table = three_column_table(
        google,
        original,
        modified or None,
        tolerance,
        registered_benches=registered,
    )

    out = (
        Path(args.out)
        if args.out
        else artifacts_dir() / "report" / f"three_column_{size}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    tmp.write_text(
        "\n".join(
            [
                f"# Three-column baseline table \u2014 {size.upper()}",
                "",
                table,
                "",
                "Validez: regla preregistrada de harness (tolerancia "
                f"{tolerance:g} pp; IC de Wilson al 95 % sobre items, "
                "promediando muestras por item).",
                "",
                "Google: cifras oficiales en modo razonamiento (model card / "
                "arXiv 2607.02770 Tablas 5 y 6). Google publica un solo valor "
                "ambiguo \"MMMU Pro\", comparado solo con MMMU-Pro Overall = "
                "media simple de Standard(10) y Vision. Las filas espaciales "
                "son exploratorias y no tienen cifra oficial.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.replace(tmp, out)
    print(f"three_column: {len(results)} result file(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
