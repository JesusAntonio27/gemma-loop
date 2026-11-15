"""Registered Part B: padded vs refill long-context throughput.

The refill engine becomes eligible only if BOTH the equivalence gate passes and
this benchmark shows at least the registered speed-up.  Registered settings live
in ``prereg/prereg.yaml`` under ``part_b`` and are not CLI-overridable; prompts
come only from the non-confirmatory P1-lite / Stage-0 dev pools, never from
confirmatory benchmark items.

Registered run::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_partb.py \
        --prereg prereg/prereg.yaml --out artifacts/engine_gate/partb.json

GPU-only; the CPU suite never runs it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from gemmaloop import paths
from gemmaloop.model.layer_plan import parse_block
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig
from gemmaloop.serve.generate import Sampling, generate_batch
from gemmaloop.serve.refill import generate_batch_refill
from gemmaloop.train.perf import cap_cuda_allocator

MIN_SPEEDUP = 1.3
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64


def speedup(refill_tok_s: float, padded_tok_s: float) -> float:
    if padded_tok_s <= 0:
        return 0.0
    return float(refill_tok_s) / float(padded_tok_s)


def summarize(rows: Sequence[dict], minimum: float = MIN_SPEEDUP) -> dict:
    refill_rows = [row for row in rows if row.get("engine") == "refill"]
    passed = bool(refill_rows) and all(
        float(row.get("speedup", 0.0)) >= float(minimum)
        for row in refill_rows
    )
    return {
        "passed": passed,
        "min_speedup": float(minimum),
        "rows": list(rows),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Part B - padded vs refill throughput",
        "",
        f"**Overall: {'PASS' if report['passed'] else 'FAIL'}** "
        f"(required speed-up {report['min_speedup']:.2f}x)",
        "",
        "| max_new_tokens | engine | requests | tokens | wall (s) | tok/s | speed-up |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['max_new_tokens']} | {row['engine']} | "
            f"{row['requests']} | {row['tokens']} | {row['elapsed_s']:.1f} | "
            f"{row['tok_s']:.1f} | {row.get('speedup', 0.0):.2f}x |"
        )
    return "\n".join(lines) + "\n"


def load_part_b(path: str | Path = "prereg/prereg.yaml") -> dict[str, Any]:
    """Read the registered Part B settings."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    part_b = payload.get("part_b")
    if not isinstance(part_b, dict):
        raise ValueError(f"{path}: no 'part_b' registration")
    config = {
        "p1lite_path": str(part_b["p1lite_path"]),
        "stage0_pool_path": (
            None
            if part_b.get("stage0_pool_path") is None
            else str(part_b["stage0_pool_path"])
        ),
        "requests": int(part_b.get("requests", 128)),
        "max_batch": int(part_b.get("eval_batch", part_b.get("max_batch", 24))),
        "eval_batch": int(part_b.get("eval_batch", 24)),
        "candidates": int(part_b.get("candidates", 360)),
        "candidate_pool": str(part_b.get("candidate_pool", "")),
        "chosen_prompts_sha256": str(
            part_b.get("chosen_prompts_sha256", "")
        ),
        "estimated_gpu_hours": str(part_b.get("estimated_gpu_hours", "")),
        "max_tokens": [int(v) for v in part_b.get("max_tokens", [8192, 16384])],
        "round_size": int(part_b.get("round_size", 1024)),
        "min_speedup": float(part_b.get("min_speedup", MIN_SPEEDUP)),
        "seed": int(part_b.get("seed", 0)),
        "block": str(part_b.get("block", "16-18")),
        "bridge": str(part_b.get("bridge", "gch")),
        "loops": int(part_b.get("loops", 2)),
    }
    if config["requests"] < 3 * config["max_batch"]:
        raise ValueError(
            f"{path}: part_b.requests must be at least 3x max_batch "
            f"({config['requests']} < {3 * config['max_batch']})"
        )
    if not config["max_tokens"] or any(value < 1 for value in config["max_tokens"]):
        raise ValueError(f"{path}: part_b.max_tokens must be positive")
    return config


def select_prompts_from_rows(
    rows: Sequence[dict[str, Any]], limit: int
) -> tuple[list[list[dict]], dict]:
    """Select prompts with the P1-lite builder's own candidate selection."""

    from scripts.make_p1lite_corpus import select_candidates

    candidates, stats = select_candidates(rows, limit, eval_index=None)
    conversations = [
        [{"role": "user", "content": candidate["prompt"]}]
        for candidate in candidates
    ]
    return conversations, stats


def prompt_sha256(prompts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()


def load_pool_mix(config: dict[str, Any]) -> tuple[list[list[dict]], dict]:
    """Reproduce the builder's 360 DAPO candidates offline and verify them.

    Uses the same Gemma 4 tokenizer EvalIndex, the non-streaming cached dataset
    at the pinned revision, and the builder's own ``select_candidates``; the
    result is checked against the P1-lite manifest and the registered 128-prompt
    sha256.
    """

    import json as _json

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from gemmaloop.model.load import MODEL_IDS
    from gemmaloop.model.snapshot import model_revision
    from scripts.make_p1lite_corpus import (
        DATASET_ID,
        REVISION,
        SPLIT,
        build_eval_index,
        select_candidates,
    )

    manifest_path = Path(config["p1lite_path"]) / "manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source") != DATASET_ID
        or manifest.get("source_revision") != REVISION
    ):
        raise RuntimeError(
            "P1-lite manifest source/revision does not match the builder"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_IDS["e2b"], revision=model_revision("e2b")
    )
    eval_index, _ = build_eval_index(tokenizer, allow_missing=False)
    dataset = load_dataset(DATASET_ID, split=SPLIT, revision=REVISION)
    candidates, stats = select_candidates(
        dataset, int(config["candidates"]), eval_index=eval_index
    )
    expected = manifest.get("selection") or {}
    if (
        int(stats.get("selected", 0)) != int(manifest.get("n_candidates", 0))
        or any(stats.get(key) != expected.get(key) for key in expected)
    ):
        raise RuntimeError(
            f"DAPO selection {stats} does not match manifest {expected}"
        )
    prompts = [candidate["prompt"] for candidate in candidates]
    chosen = prompts[: int(config["requests"])]
    if len(chosen) < int(config["requests"]):
        raise RuntimeError(
            f"DAPO selection produced {len(chosen)} candidates, "
            f"Part B needs {config['requests']}"
        )
    digest = prompt_sha256(chosen)
    registered = str(config.get("chosen_prompts_sha256") or "")
    if registered and digest != registered:
        raise RuntimeError(
            f"chosen 128-prompt sha256 {digest} != registered {registered}"
        )
    conversations = [
        [{"role": "user", "content": prompt}] for prompt in chosen
    ]
    sources = {
        "dapo": {
            "dataset": DATASET_ID,
            "revision": REVISION,
            "selected": len(candidates),
            "used": len(chosen),
            "selection": stats,
            "chosen_prompts_sha256": digest,
            "confirmatory_items_used": False,
        }
    }
    return conversations, sources


def _run(
    engine: str,
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    *,
    max_batch: int,
    round_size: int,
    seed: int,
    request_ids: list[str],
) -> tuple[list[Any], float]:
    started = time.perf_counter()
    if engine == "padded":
        outputs = generate_batch(
            model,
            processor,
            conversations,
            sampling,
            seed=seed,
            max_batch=max_batch,
            request_ids=request_ids,
        )
    else:
        outputs = generate_batch_refill(
            model,
            processor,
            conversations,
            sampling,
            seed=seed,
            max_batch=max_batch,
            round_size=round_size,
            request_ids=request_ids,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return list(outputs), time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", default="prereg/prereg.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="narrow the run; never the registered Part B result",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loops", type=int, default=None)
    parser.add_argument("--block", default=None)
    parser.add_argument(
        "--bridge",
        default=None,
        choices=("scalar", "none", "damped", "gch"),
    )
    parser.add_argument("--max-new-tokens", default=None)
    parser.add_argument("--round-size", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("--vram-cap", type=float, default=0.92)
    parser.add_argument("--out", default="")
    return parser


def resolve_settings(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    overrides = {
        "limit": args.limit,
        "loops": args.loops,
        "block": args.block,
        "bridge": args.bridge,
        "max_tokens": args.max_new_tokens,
        "round_size": args.round_size,
        "max_batch": args.max_batch,
        "seed": args.seed,
        "min_speedup": args.min_speedup,
    }
    if not args.smoke and any(value is not None for value in overrides.values()):
        raise SystemExit(
            "registered Part B uses prereg/prereg.yaml; CLI overrides need "
            "--smoke and never count as the registered result"
        )
    settings = dict(config)
    if args.smoke:
        if args.limit is not None:
            settings["requests"] = max(int(args.limit), 3 * settings["max_batch"])
        if args.loops is not None:
            settings["loops"] = int(args.loops)
        if args.block is not None:
            settings["block"] = args.block
        if args.bridge is not None:
            settings["bridge"] = args.bridge
        if args.max_new_tokens is not None:
            settings["max_tokens"] = [
                int(part)
                for part in str(args.max_new_tokens).split(",")
                if part.strip()
            ]
        if args.round_size is not None:
            settings["round_size"] = int(args.round_size)
        if args.max_batch is not None:
            settings["max_batch"] = int(args.max_batch)
        if args.seed is not None:
            settings["seed"] = int(args.seed)
        if args.min_speedup is not None:
            settings["min_speedup"] = float(args.min_speedup)
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_part_b(args.prereg)
    settings = resolve_settings(args, config)
    if not torch.cuda.is_available():
        raise SystemExit("bench_partb: CUDA GPU required; never run in CPU tests")
    if args.vram_cap > 0:
        cap_cuda_allocator(args.vram_cap)

    conversations, sources = load_pool_mix(settings)
    request_ids = [f"partb-{index}" for index in range(len(conversations))]
    model = processor = None
    rows: list[dict] = []
    try:
        model, processor, _ = load_model(
            ModelSpec(
                size="e2b",
                quant="bf16",
                loop=LoopConfig(
                    n_loops=int(settings["loops"]),
                    bridge=settings["bridge"],
                    block=parse_block(settings["block"]),
                ),
            )
        )
        for max_new_tokens in settings["max_tokens"]:
            sampling = Sampling(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_new_tokens=int(max_new_tokens),
            )
            stats: dict[str, dict] = {}
            for engine in ("padded", "refill"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                outputs, elapsed = _run(
                    engine,
                    model,
                    processor,
                    conversations,
                    sampling,
                    max_batch=int(settings["max_batch"]),
                    round_size=int(settings["round_size"]),
                    seed=int(settings["seed"]),
                    request_ids=request_ids,
                )
                tokens = sum(
                    int(output.completion_tokens) for output in outputs
                )
                stats[engine] = {
                    "max_new_tokens": int(max_new_tokens),
                    "engine": engine,
                    "requests": len(outputs),
                    "tokens": tokens,
                    "elapsed_s": round(elapsed, 3),
                    "tok_s": round(tokens / elapsed, 3) if elapsed > 0 else 0.0,
                }
                del outputs
            stats["refill"]["speedup"] = speedup(
                stats["refill"]["tok_s"], stats["padded"]["tok_s"]
            )
            rows.extend([stats["padded"], stats["refill"]])
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **summarize(rows, settings["min_speedup"]),
        "registered": not args.smoke,
        "config": settings,
        "sources": sources,
    }
    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir()
        / "engine_gate"
        / ("partb_smoke.json" if args.smoke else "partb.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    output.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(f"Part B: {'PASS' if report['passed'] else 'FAIL'} -> {output}")
    return 0 if (report["passed"] and not args.smoke) else 1


if __name__ == "__main__":
    sys.exit(main())
