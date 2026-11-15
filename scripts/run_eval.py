"""Run the preregistered benchmark suite for one model variant.

Example::

    uv run python scripts/run_eval.py --variant L --size e2b --quant bf16 \
        --loops 2 --bridge scalar --adapter runs/lora \
        --bench lcb,aime,gpqa,mmlupro --prereg prereg/prereg.yaml

Variant naming follows the plan: ``R`` is ``--bridge none`` without an adapter,
``R-damped`` is ``--bridge damped`` without an adapter, and ``O`` is the original
model (no ``--loops``).  Sampling and per-benchmark sizes come from the YAML run
config; the CLI never invents sampling parameters.

The model is loaded once and runs every requested benchmark in sequence, so two
models are never resident at the same time.  Each benchmark's items are loaded
once and released (``del`` + ``gc.collect``) before the next one starts; this is
essential for LiveCodeBench, whose release is several gigabytes.  Results are
resumable: an interrupted benchmark continues from its ``*.partial.jsonl`` file
on the next invocation.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Sequence

from gemmaloop.eval.runner import (
    fingerprint_digest,
    load_items_for_bench,
    read_run_config,
    resolve_bench,
    run_bench,
    run_fingerprint,
)
from gemmaloop.gpulock import require_gpu_lock
from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path
from gemmaloop.model.load import ModelSpec, _resolve_loop_config, load_model
from gemmaloop.model.layer_plan import parse_block
from gemmaloop.model.looped import LoopConfig
from gemmaloop.paths import results_dir
from gemmaloop.train.perf import cap_cuda_allocator

DEFAULT_BATCH = 0  # 0 = auto from sampling.max_new_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preregistered Gemma-Loop benchmarks resumably."
    )
    parser.add_argument("--variant", required=True, help="e.g. O, L, R, R-damped")
    parser.add_argument("--size", required=True, choices=("e2b", "e4b"))
    parser.add_argument(
        "--quant", default="bf16", choices=("bf16", "nf4")
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="number of loop passes; omit for the original (O) model",
    )
    parser.add_argument(
        "--bridge",
        default=None,
        choices=("scalar", "none", "damped", "gch"),
        help=(
            "loop bridge; defaults to the adapter's bridge when an adapter "
            "carries loop state, else scalar"
        ),
    )
    parser.add_argument(
        "--block",
        default=None,
        help=(
            "inclusive loop block START-END inside the KV-consumer region "
            "(e.g. 16-18); defaults to the adapter's block, else legacy 15-29"
        ),
    )
    parser.add_argument(
        "--adapter-policy",
        default=None,
        choices=("always", "loop_only"),
        help="LoRA activation policy; defaults to the adapter's value, else always",
    )
    parser.add_argument(
        "--lora-target-layers",
        default=None,
        choices=("language", "loop_block", "loop_block_coda"),
        help="LoRA target layer set; defaults to the adapter's value, else language",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="adapter directory; repeat for a chain (bf16 only)",
    )
    parser.add_argument(
        "--bench",
        required=True,
        help="comma-separated benchmark names, e.g. lcb,aime,gpqa,mmlupro",
    )
    parser.add_argument(
        "--prereg",
        required=True,
        help="YAML run config (sampling + benches), e.g. prereg/prereg.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="result root (default: $GEMMALOOP_ARTIFACTS/results)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="generation batch size (0 = auto from the measured non-spilling ceiling)",
    )
    parser.add_argument(
        "--engine",
        default="padded",
        choices=("padded", "refill"),
        help=(
            "generation engine: padded (default) or the round-based refill "
            "engine (registered only after the Task 5d gate passes)"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="rotate any existing partial file instead of resuming it",
    )
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help=(
            "cap the CUDA caching allocator at this fraction of VRAM so an "
            "over-budget refill round OOMs (and halves the batch) instead of "
            "paging to RAM; 0 disables"
        ),
    )
    parser.add_argument(
        "--vram-free-margin-gib",
        type=float,
        default=0.5,
        help="leave this much VRAM free when deriving the allocator cap",
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help=(
            "tolerate skipped exploratory benchmarks only; failures and skips "
            "of registered (non-exploratory) benchmarks still exit non-zero"
        ),
    )
    parser.add_argument(
        "--heartbeat",
        default=None,
        help=(
            "heartbeat JSON path (default: "
            "$GEMMALOOP_ARTIFACTS/runs/detached/heartbeat.json)"
        ),
    )
    return parser


def _is_gated_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "gated" in text
        or "gatedrepo" in name
        or "401" in text
        or "authentication" in text
        or "permission" in name
    )


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )
    return cleaned.strip("-") or "run"


def _result_path(
    root: Path,
    variant: str,
    size: str,
    quant: str,
    n_loops: int,
    bridge: str | None,
    bench: str,
    *,
    prereg_tag: str,
    fingerprint: dict,
) -> Path:
    """Result path keyed by the run config, not just variant/size/bench.

    The fingerprint digest changes with the sampling, adapters, item set and
    prereg tag, so two runs that would otherwise collide get distinct files and
    cannot silently overwrite or reuse each other's partial.
    """

    safe_variant = _safe_name(variant)
    loop = f"n{n_loops}" if n_loops else "n0"
    bridge_part = bridge if n_loops and bridge else "base"
    digest = fingerprint_digest(fingerprint)[:12]
    name = (
        f"{size}_{quant}_{loop}_{bridge_part}_{bench}_"
        f"{_safe_name(prereg_tag)}_{digest}.json"
    )
    return root / safe_variant / name


def _loop_config(args: argparse.Namespace) -> LoopConfig | None:
    if args.loops is None:
        return None
    if args.loops < 1:
        raise ValueError("--loops must be at least 1")
    bridge = args.bridge
    if bridge is None:
        # Omitted --bridge: inherit the checkpoint's bridge when an adapter
        # carries loop state, otherwise use the scalar default.
        adapter_loop = _resolve_loop_config(None, tuple(args.adapter))
        bridge = adapter_loop.bridge if adapter_loop is not None else "scalar"
    # Validate an explicit --bridge against the checkpoint *before* loading
    # the multi-GB model; this also returns the adapter's bridge/ple_mode/block.
    block_text = getattr(args, "block", None)
    block = parse_block(block_text) if block_text else None
    resolved = _resolve_loop_config(
        LoopConfig(
            n_loops=args.loops,
            bridge=bridge,
            block=block,
            adapter_policy=getattr(args, "adapter_policy", None),
            lora_target_layers=getattr(args, "lora_target_layers", None),
        ),
        tuple(args.adapter),
    )
    assert resolved is not None
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # One GPU job at a time: take the process-lifetime lock before touching the
    # model, datasets or result tree.
    require_gpu_lock()
    # One allocator cap, computed once and recorded in both the heartbeat and
    # every result file.
    vram_cap = (
        cap_cuda_allocator(args.vram_cap, args.vram_free_margin_gib)
        if args.vram_cap > 0
        else None
    )
    heartbeat = Heartbeat(
        args.heartbeat or default_heartbeat_path("eval"),
        run=args.variant,
    )
    heartbeat.start(30.0)
    heartbeat.update(extra={"vram_cap": vram_cap})
    benches = [part.strip() for part in args.bench.split(",") if part.strip()]
    if not benches:
        raise ValueError("--bench needs at least one benchmark name")

    config = read_run_config(args.prereg)
    missing = [bench for bench in benches if bench not in config.benches]
    if missing:
        raise ValueError(
            f"benchmark(s) {missing} are not defined under 'benches' in "
            f"{args.prereg}"
        )

    spec = ModelSpec(
        size=args.size,
        quant=args.quant,
        adapters=tuple(args.adapter),
        loop=_loop_config(args),
    )
    model, processor, _ = load_model(spec)

    root = Path(args.out_dir) if args.out_dir else results_dir()
    n_loops = spec.loop.n_loops if spec.loop is not None else 0

    ran: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    try:
        for bench in benches:
            items = None
            try:
                bench_config = config.benches[bench]
                items = load_items_for_bench(
                    bench, bench_config.limit, bench_config.seed
                )
                resolved = resolve_bench(bench)
                clear_cache = getattr(
                    resolved.module, "clear_release_cache", None
                )
                if resolved.kind == "lcb" and clear_cache is not None:
                    clear_cache()

                fingerprint = run_fingerprint(
                    spec,
                    bench,
                    items,
                    bench_config.samples,
                    config.sampling,
                    seed=bench_config.seed,
                    image_soft_tokens=bench_config.image_soft_tokens,
                    prereg_tag=config.tag,
                    engine=args.engine,
                )
                out_path = _result_path(
                    root,
                    args.variant,
                    args.size,
                    args.quant,
                    n_loops,
                    spec.loop.bridge if spec.loop is not None else None,
                    bench,
                    prereg_tag=config.tag,
                    fingerprint=fingerprint,
                )
                heartbeat.set_phase(
                    bench, total=len(items) * bench_config.samples
                )
                result = run_bench(
                    spec,
                    args.variant,
                    bench,
                    items,
                    bench_config.samples,
                    config.sampling,
                    out_path,
                    batch=args.batch if args.batch > 0 else None,
                    seed=bench_config.seed,
                    image_soft_tokens=bench_config.image_soft_tokens,
                    prereg_tag=config.tag,
                    model=model,
                    processor=processor,
                    fresh=args.fresh,
                    engine=args.engine,
                    on_batch=lambda progress: heartbeat.update(
                        items_done=progress["items_done"],
                        items_total=progress["items_total"],
                        tokens=progress["tokens"],
                        batch=progress["batch"],
                        phase=progress["bench"],
                    ),
                )
                result.metadata["vram_cap"] = vram_cap
                result.save(out_path)
            except Exception as exc:
                if "gpqa" in bench.lower() and _is_gated_error(exc):
                    print(
                        f"{bench}: skipped: gated dataset ({exc})",
                        file=sys.stderr,
                    )
                    skipped.append(bench)
                else:
                    print(f"{bench}: failed: {exc}", file=sys.stderr)
                    failed.append(bench)
                continue
            finally:
                # Release the benchmark before loading the next one (LCB's
                # release is >4 GB in the worst case).
                del items
                gc.collect()

            ran.append(bench)
            n_length = sum(
                item.finish_reason == "length" for item in result.items
            )
            length_rate = n_length / len(result.items) if result.items else 0.0
            print(
                f"{bench}: {len(result.items)} samples, "
                f"accuracy={result.accuracy():.4f}, "
                f"length={n_length} ({length_rate:.1%}), "
                f"result={out_path}"
            )
            if result.metadata.get("scoring") == "non-official":
                print(
                    f"{bench}: WARNING non-official scorer used "
                    f"({result.metadata.get('scoring_reason', '')})",
                    file=sys.stderr,
                )
    finally:
        del model
        gc.collect()

    heartbeat.stop()
    print(
        "summary: "
        f"run={','.join(ran)} "
        f"skipped={','.join(skipped)} "
        f"failed={','.join(failed)}"
    )

    fatal_skips = [
        bench
        for bench in skipped
        if not (
            args.allow_skip
            and config.benches[bench].exploratory
        )
    ]
    if failed or fatal_skips:
        problems = []
        if failed:
            problems.append(f"failed={','.join(failed)}")
        if fatal_skips:
            problems.append(f"skipped={','.join(fatal_skips)}")
        print(
            "ERROR: run_eval refused to treat this run as complete: "
            + "; ".join(problems),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
