"""Padded vs slot-refill throughput on a forced long-tailed length mix.

The evaluation's reasoning traces have a long tail: most completions are a few
thousand tokens, a few reach the 32k cap.  A fixed padded batch keeps every slot
busy until the longest sequence in the chunk finishes; the refill engine puts a
new request in each freed slot at the next round boundary.

To compare the two schedulers on the same work, every request has a forced
completion budget (``min_new_tokens == max_new_tokens`` for the padded engine,
per-slot budgets for refill).  The padded engine is given the advantage of
knowing the budget: requests are grouped by length and each group uses its own
adaptive batch.  Both engines therefore generate exactly the same number of
tokens and the comparison is wall-clock only.

Usage::

    uv run python scripts/bench_tail.py --loops 1,2

Results are merged into ``<artifacts>/calibration/tail.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from gemmaloop import paths
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling, generate_batch, max_batch_for
from gemmaloop.serve.refill import generate_batch_refill

DEFAULT_MIX = "1024:16,4096:22,8192:16,16384:10"
PROMPT = (
    "Write a detailed, step-by-step explanation of how a modern computer works, "
    "covering the CPU, memory hierarchy, storage and the operating system. "
    "Do not stop early."
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _parse_mix(value: str) -> list[tuple[int, int]]:
    mix = []
    for part in value.split(","):
        budget, count = part.split(":")
        mix.append((int(budget), int(count)))
    return mix


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _padded(model, processor, conversations, budgets, seed, max_batch) -> list[Any]:
    """Group requests by budget; each group uses the same batch cap as refill."""
    outputs: list[Any] = []
    previous_min = getattr(model.generation_config, "min_new_tokens", 0)
    try:
        for budget in sorted({*budgets}, reverse=True):
            indices = [i for i, value in enumerate(budgets) if value == budget]
            model.generation_config.min_new_tokens = budget
            group = [conversations[i] for i in indices]
            outputs.extend(
                generate_batch(
                    model,
                    processor,
                    group,
                    Sampling(max_new_tokens=budget),
                    seed=seed,
                    max_batch=min(max_batch, max_batch_for(budget)),
                )
            )
    finally:
        model.generation_config.min_new_tokens = previous_min
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Padded vs refill long-tail throughput.")
    parser.add_argument("--loops", default="1,2")
    parser.add_argument("--mix", default=DEFAULT_MIX)
    parser.add_argument("--round-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help="slot/batch cap for both engines (0 = adaptive for the longest request)",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    loops = [int(part) for part in args.loops.split(",") if part.strip()]
    mix = _parse_mix(args.mix)
    budgets = [budget for budget, count in mix for _ in range(count)]
    conversations = [[{"role": "user", "content": PROMPT}] for _ in budgets]
    max_budget = max(budgets)
    max_batch = args.max_batch or max_batch_for(max_budget)
    _log(
        f"requests={len(budgets)} max_budget={max_budget} max_batch={max_batch} "
        f"total_tokens={sum(budgets)}"
    )

    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir() / "calibration" / "tail.json"
    )
    data = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    data.setdefault("results", {})
    data.setdefault("meta", {})
    data["meta"].update(
        {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mix": mix,
            "total_tokens": sum(budgets),
            "round_size": args.round_size,
            "seed": args.seed,
        }
    )

    model = processor = None
    try:
        model, processor, _ = load_model(
            ModelSpec(size="e2b", quant="bf16", loop=LoopConfig(n_loops=1))
        )
        for n_loops in loops:
            set_n_loops(model.model.language_model, n_loops)
            for engine in ("padded", "refill"):
                key = f"n{n_loops}_{engine}"
                if key in data["results"]:
                    _log(f"{key} already measured, skipping")
                    continue
                _log(f"{key}: generating {sum(budgets)} tokens ...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                if engine == "padded":
                    outputs = _padded(
                        model, processor, conversations, budgets, args.seed, max_batch
                    )
                else:
                    outputs = generate_batch_refill(
                        model,
                        processor,
                        conversations,
                        Sampling(max_new_tokens=max_budget),
                        seed=args.seed,
                        max_batch=max_batch,
                        round_size=args.round_size,
                        budgets=budgets,
                        stop_on_eos=False,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                tokens = sum(output.completion_tokens for output in outputs)
                record = {
                    "engine": engine,
                    "n_loops": n_loops,
                    "elapsed_s": round(elapsed, 2),
                    "tokens": tokens,
                    "expected_tokens": sum(budgets),
                    "tok_s": round(tokens / elapsed, 2),
                    "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                }
                data["results"][key] = record
                _save(output, data)
                _log(
                    f"{key}: {record['tok_s']} tok/s in {record['elapsed_s']}s "
                    f"(tokens={tokens}, peak={record['peak_vram_gib']} GiB)"
                )
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save(output, data)
    _log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
