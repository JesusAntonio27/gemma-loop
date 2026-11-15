"""Padded vs round-based refill on real benchmark prompts (Task 5c Part B).

This is the acceptance benchmark for the refill engine.  It uses *natural*
completion lengths (no forced budgets) on the two text benchmarks that dominate
the registered evaluation:

* 64 LiveCodeBench ``release_v6`` prompts from the post-cutoff window
  (``contest_date >= 2025-01-01``);
* 64 GPQA Diamond prompts.

Both engines decode with the registered sampling settings
(``max_new_tokens=32768``, temperature 1.0, top_p 0.95, top_k 64) at N=2 loop
passes, and the script reports wall-clock time, effective tokens/s (all
completion tokens divided by wall time), peak reserved VRAM and whether the run
spilled (>95 % of the card).  Requests are admitted in item-id order.

The GPU must be free before this runs.  It loads E2B bf16 once and runs the
padded engine first, then the refill engine, for each benchmark/loop count.
Results are written incrementally to ``<artifacts>/calibration/refill.json``
plus a markdown table in the same directory; re-running skips completed keys.

Usage::

    uv run python scripts/bench_refill.py --loops 2 --limit 64 \
        --max-new-tokens 32768 --round-size 1024
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from gemmaloop import paths
from gemmaloop.eval.benchmarks import stable_sample
from gemmaloop.eval.benchmarks import gpqa, lcb
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling, generate_batch, max_batch_for
from gemmaloop.serve.refill import generate_batch_refill
from gemmaloop.train.perf import cap_cuda_allocator

SPILL_RESERVED_FRACTION = 0.95
DEFAULT_BENCHES = "lcb,gpqa"
DEFAULT_LIMIT = 64
DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_ROUND_SIZE = 1024
DEFAULT_LOOPS = "2"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "| bench | N | engine | requests | tokens | wall (s) | tok/s | "
        "p50/p90/max tokens | peak VRAM (GiB) | spill |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    current = data.get("meta", {}).get("config_key")
    for key in sorted(data.get("results", {})):
        record = data["results"][key]
        if current is not None and record.get("config_key") != current:
            continue
        lines.append(
            f"| {record['bench']} | {record['n_loops']} | {record['engine']} | "
            f"{record['requests']} | {record['tokens']} | "
            f"{record['elapsed_s']:.1f} | {record['tok_s']:.1f} | "
            f"{record['p50_tokens']:.0f}/{record['p90_tokens']:.0f}/"
            f"{record['max_tokens']} | {record['peak_vram_gib']:.2f} | "
            f"{'yes' if record['spill'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _load_lcb_post_cut(limit: int, seed: int) -> list:
    """Post-cutoff LCB items without materialising the whole release.

    The 182 post-cut problems live in ``test5.jsonl`` (7) and ``test6.jsonl``
    (175), so only those two files are downloaded; the multi-gigabyte
    ``test.jsonl``..``test4.jsonl`` rows are never loaded in this process.
    """
    rows: list[dict] = []
    for filename in ("test5.jsonl", "test6.jsonl"):
        path = hf_hub_download(
            repo_id=lcb.DATASET_ID,
            filename=filename,
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("contest_date", "")) >= lcb.POST_CUT_DATE:
                    rows.append(row)
    items = [lcb.item_from_row(row) for row in rows]
    return stable_sample(
        items, limit, seed, key=lambda item, index: str(item.item_id)
    )


def _load_items(bench: str, limit: int, seed: int, *, lcb_post_cut: bool) -> list:
    if bench == "lcb":
        if lcb_post_cut:
            return _load_lcb_post_cut(limit, seed)
        return lcb.load_items(limit=limit, seed=seed)
    if bench == "gpqa":
        return gpqa.load_items(limit=limit, seed=seed)
    raise ValueError(f"unsupported benchmark {bench!r}; expected lcb or gpqa")


def _build_conversations(bench: str, items: list) -> list[list[dict]]:
    module = lcb if bench == "lcb" else gpqa
    return [module.build_messages(item) for item in items]


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 2**30


def _total_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def _run_engine(
    engine: str,
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    request_ids: list[str],
    sampling: Sampling,
    seed: int,
    max_batch: int,
    round_size: int,
) -> list[Any]:
    if engine == "padded":
        return generate_batch(
            model,
            processor,
            conversations,
            sampling,
            seed=seed,
            max_batch=max_batch,
            request_ids=request_ids,
        )
    return generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        seed=seed,
        max_batch=max_batch,
        round_size=round_size,
        request_ids=request_ids,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Padded vs round-based refill on real LCB/GPQA prompts."
    )
    parser.add_argument("--benches", default=DEFAULT_BENCHES)
    parser.add_argument("--size", default="e2b", choices=("e2b", "e4b"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--loops", default=DEFAULT_LOOPS)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--round-size", type=int, default=DEFAULT_ROUND_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help="batch cap for both engines (0 = adaptive for max_new_tokens)",
    )
    parser.add_argument(
        "--engines",
        default="padded,refill",
        help="comma-separated engines to run",
    )
    parser.add_argument(
        "--lcb-all-dates",
        action="store_true",
        help="do not restrict LiveCodeBench to the post-cutoff window",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help="allocator cap fraction (0 disables)",
    )
    args = parser.parse_args(argv)

    benches = [part.strip().lower() for part in args.benches.split(",") if part.strip()]
    engines = [part.strip().lower() for part in args.engines.split(",") if part.strip()]
    loops = [int(part) for part in args.loops.split(",") if part.strip()]
    unknown = [engine for engine in engines if engine not in ("padded", "refill")]
    if unknown:
        parser.error(f"unknown engine(s) {unknown}; expected padded,refill")
    if not benches or any(bench not in ("lcb", "gpqa") for bench in benches):
        parser.error("--benches must be a comma-separated subset of lcb,gpqa")
    if args.limit < 1 or args.samples < 1:
        parser.error("--limit and --samples must be at least 1")
    if args.max_new_tokens < 1 or args.round_size < 1:
        parser.error("--max-new-tokens and --round-size must be at least 1")
    if args.max_batch < 0:
        parser.error("--max-batch must be >= 0 (0 = auto)")
    if not loops or any(loop < 1 for loop in loops):
        parser.error("--loops must be positive integers")

    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir() / "calibration" / "refill.json"
    )
    data = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    run_config = {
        "size": args.size,
        "benches": benches,
        "limit": args.limit,
        "samples": args.samples,
        "loops": loops,
        "max_new_tokens": args.max_new_tokens,
        "round_size": args.round_size,
        "seed": args.seed,
        "max_batch": args.max_batch,
        "lcb_post_cut": not args.lcb_all_dates,
    }
    config_key = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    data.setdefault("results", {})
    data.setdefault("meta", {})
    data["meta"].update(
        {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "config": run_config,
            "config_key": config_key,
            **run_config,
        }
    )

    sampling = Sampling(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_new_tokens=args.max_new_tokens,
    )
    max_batch = args.max_batch or max_batch_for(args.max_new_tokens, args.size)
    _log(
        f"size={args.size} max_new_tokens={args.max_new_tokens} "
        f"round_size={args.round_size} max_batch={max_batch} loops={loops} "
        f"samples={args.samples} config={config_key}"
    )

    model = processor = None
    try:
        if args.vram_cap > 0:
            cap_cuda_allocator(args.vram_cap)
        model, processor, _ = load_model(
            ModelSpec(
                size=args.size,
                quant="bf16",
                loop=LoopConfig(n_loops=loops[0]),
            )
        )
        for bench in benches:
            items = _load_items(
                bench,
                args.limit,
                args.seed,
                lcb_post_cut=not args.lcb_all_dates,
            )
            conversations = _build_conversations(bench, items)
            request_ids = [
                f"{item.item_id}:{sample_idx}"
                for sample_idx in range(args.samples)
                for item in items
            ]
            if args.samples > 1:
                conversations = conversations * args.samples
            _log(f"{bench}: {len(items)} items, {len(conversations)} requests")
            try:
                for n_loops in loops:
                    set_n_loops(model.model.language_model, n_loops)
                    for engine in engines:
                        key = f"{bench}_{config_key}_n{n_loops}_{engine}"
                        if key in data["results"]:
                            _log(f"{key} already measured, skipping")
                            continue
                        _log(f"{key}: running {len(conversations)} requests ...")
                        _reset_peak()
                        started = time.perf_counter()
                        outputs = _run_engine(
                            engine,
                            model,
                            processor,
                            conversations,
                            request_ids,
                            sampling,
                            args.seed,
                            max_batch,
                            args.round_size,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        tokens = [int(output.completion_tokens) for output in outputs]
                        total_tokens = sum(tokens)
                        peak = _peak_gib()
                        total = _total_gib()
                        record = {
                            "bench": bench,
                            "engine": engine,
                            "config_key": config_key,
                            "n_loops": n_loops,
                            "samples": args.samples,
                            "requests": len(conversations),
                            "elapsed_s": round(elapsed, 2),
                            "tokens": total_tokens,
                            "tok_s": round(total_tokens / elapsed, 2)
                            if elapsed > 0
                            else 0.0,
                            "p50_tokens": round(_percentile(tokens, 0.50), 1),
                            "p90_tokens": round(_percentile(tokens, 0.90), 1),
                            "max_tokens": max(tokens) if tokens else 0,
                            "finish_stop_rate": round(
                                sum(
                                    output.finish_reason == "stop"
                                    for output in outputs
                                )
                                / len(outputs),
                                4,
                            )
                            if outputs
                            else 0.0,
                            "peak_vram_gib": round(peak, 3),
                            "total_vram_gib": round(total, 3),
                            "spill": bool(
                                total and peak > SPILL_RESERVED_FRACTION * total
                            ),
                        }
                        data["results"][key] = record
                        _save(output, data)
                        _log(
                            f"{key}: {record['tok_s']} tok/s in "
                            f"{record['elapsed_s']}s (tokens={total_tokens}, "
                            f"peak={record['peak_vram_gib']} GiB, "
                            f"spill={'yes' if record['spill'] else 'no'})"
                        )
                        del outputs
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            finally:
                # Release the benchmark before the next one starts.
                del items, conversations
                gc.collect()
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save(output, data)
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(data), encoding="utf-8")
    _log(f"wrote {output} and {markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
