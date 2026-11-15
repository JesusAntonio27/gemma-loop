"""Summarise ``scripts/bench_train.py`` JSONL output as Markdown tables.

Usage::

    uv run python scripts/bench_train_report.py bench.jsonl [--projection-length 4096]

Prints one table per perf preset (rows: length, columns: N) with
``tok/s`` and peak reserved GiB, and the projected GPU hours of the Phase-1
plan (O-H and L at 40M tokens x 3 seeds, O-H-ext at 66M, L-plevec at 40M) for
the requested length.  ``L`` time per token is the expectation over the
training distribution of N, ``sum_n p_n / tok_s(n)``; a missing ``N=3`` rate is
interpolated linearly in time per token between N=2 and N=4.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LOOP_PROBS = (0.25, 0.40, 0.20, 0.15)
PLAN = (
    ("O-H (3 seeds x 40M)", "oh", 3 * 40e6),
    ("L (3 seeds x 40M)", "l", 3 * 40e6),
    ("O-H-ext (66M)", "oh", 66e6),
    ("L-plevec (40M)", "l", 40e6),
)


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "length" in record:
                rows.append(record)
    return rows


def seconds_per_token(rates: dict[int, float]) -> dict[str, float] | None:
    if 1 not in rates:
        return None
    spt = {n: 1.0 / r for n, r in rates.items() if r}
    if 3 not in spt and 2 in spt and 4 in spt:
        spt[3] = 0.5 * (spt[2] + spt[4])
    if not all(n in spt for n in (1, 2, 3, 4)):
        return {"oh": spt[1]}
    return {"oh": spt[1], "l": sum(p * spt[n] for n, p in zip((1, 2, 3, 4), LOOP_PROBS))}


def fit_step_time(points: dict[int, float]):
    """Least-squares ``t(L) = a + b L + c L^2`` through measured (L, seconds) points."""
    import numpy as np

    lengths = np.array(sorted(points), dtype=float)
    times = np.array([points[int(x)] for x in lengths])
    if len(lengths) >= 3:
        design = np.stack([np.ones_like(lengths), lengths, lengths**2], axis=1)
    else:
        design = np.stack([np.ones_like(lengths), lengths], axis=1)
    coef, *_ = np.linalg.lstsq(design, times, rcond=None)
    coef = list(coef) + [0.0] * (3 - len(coef))
    return lambda x: coef[0] + coef[1] * x + coef[2] * x * x


def mix_rows(records: Path, lengths_json: Path) -> list[tuple[int, float]]:
    """(length, weight) per audited row; weights sum to 1 over the mix."""
    shares = json.loads(lengths_json.read_text(encoding="utf-8"))["mix_row_share"]
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["source"]] += 1
    return [(int(row["total"]), shares[row["source"]] / counts[row["source"]]) for row in rows]


def mix_projection(items: list[dict], rows: list[tuple[int, float]], max_length: int) -> dict | None:
    """Effective tok/s per N over the length mix, truncating at ``max_length``."""
    per_n: dict[int, dict[int, float]] = defaultdict(dict)
    for r in items:
        if r["fits"] and r["tok_s"] and r["length"] <= max_length:
            per_n[r["n_loops"]][r["length"]] = r["length"] / r["tok_s"]
    if any(len(per_n.get(n, {})) < 2 for n in (1, 2, 4)) or max(per_n[4]) < max_length:
        return None
    fits = {n: fit_step_time(points) for n, points in per_n.items()}
    if 3 not in fits:
        fits[3] = lambda x, f2=fits[2], f4=fits[4]: 0.5 * (f2(x) + f4(x))
    tokens = sum(w * min(length, max_length) for length, w in rows)
    rates = {}
    for n in (1, 2, 3, 4):
        seconds = sum(w * fits[n](min(length, max_length)) for length, w in rows)
        rates[n] = tokens / seconds
    spt_l = sum(p / rates[n] for n, p in zip((1, 2, 3, 4), LOOP_PROBS))
    return {"rates": rates, "oh": 1.0 / rates[1], "l": spt_l, "mean_tokens_per_row": tokens}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--projection-length", type=int, action="append", default=None)
    parser.add_argument("--length-audit", type=Path, default=None, help="length-audit directory")
    args = parser.parse_args(argv)
    audit_rows = (
        mix_rows(args.length_audit / "records.jsonl", args.length_audit / "lengths.json")
        if args.length_audit
        else None
    )

    rows = load(args.jsonl)
    by_perf: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_perf[row["perf"]].append(row)

    for perf, items in by_perf.items():
        lengths = sorted({r["length"] for r in items})
        loops = sorted({r["n_loops"] for r in items})
        latest = {(r["length"], r["n_loops"]): r for r in items}
        print(f"\n### {perf}\n")
        print("| length | " + " | ".join(f"N={n} tok/s | N={n} peak GiB" for n in loops) + " |")
        print("|---" * (1 + 2 * len(loops)) + "|")
        for length in lengths:
            cells = []
            for n in loops:
                r = latest.get((length, n))
                if r is None:
                    cells += ["-", "-"]
                    continue
                if r["oom"] or not r["fits"]:
                    tag = "OOM" if r["oom"] else ("retry" if r["alloc_retries"] else "no")
                    cells += [tag, f"{r['peak_reserved_gib']:.2f}"]
                else:
                    cells += [f"{r['tok_s']:.0f}", f"{r['peak_reserved_gib']:.2f}"]
            print(f"| {length} | " + " | ".join(cells) + " |")
        fits_n4 = [r["length"] for r in items if r["n_loops"] == 4 and r["fits"]]
        if fits_n4:
            print(f"\nlargest measured length that fits at N=4: {max(fits_n4)}")

        for length in args.projection_length or []:
            if audit_rows is not None:
                projection = mix_projection(items, audit_rows, length)
                if projection is None:
                    print(f"\nno mix projection at {length}: need >=2 clean lengths per N up to {length}")
                    continue
                rates_txt = ", ".join(f"N={n} {r:.0f}" for n, r in projection["rates"].items())
                print(
                    f"\nLength-mix tok/s at max_length={length} (mean {projection['mean_tokens_per_row']:.0f} "
                    f"trained tokens/row): {rates_txt}"
                )
                spt = {"oh": projection["oh"], "l": projection["l"]}
            else:
                rates = {r["n_loops"]: r["tok_s"] for r in items if r["length"] == length and r["fits"]}
                spt = seconds_per_token(rates)
            if not spt or "l" not in spt:
                print(f"\nno projection at {length}: need clean N=1,2,4 rates, got {sorted(rates)}")
                continue
            total = 0.0
            print(f"\nProjected GPU hours at max_length={length} (L: E[s/token] over N~{LOOP_PROBS}):\n")
            print("| run | tokens | hours |")
            print("|---|---|---|")
            for name, kind, tokens in PLAN:
                hours = tokens * spt[kind] / 3600
                total += hours
                print(f"| {name} | {tokens / 1e6:.0f}M | {hours:.1f} |")
            print(f"| total | | {total:.1f} |")
            print(f"\nO-H {1 / spt['oh']:.0f} tok/s, L (mixture) {1 / spt['l']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
