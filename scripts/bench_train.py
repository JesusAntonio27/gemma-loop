"""Measure LoRA training memory and throughput through the real trainer path.

``scripts/calibrate.py`` times ``model(**batch)``, i.e. transformers' own
full-logits loss without autocast.  The trainer does something else: TRL's
``SFTTrainer`` calls the model with ``fused_lm_head=True`` under bf16 autocast.
This benchmark therefore builds the model, the ``SFTConfig`` and the
``LoopSFTTrainer`` exactly as ``scripts/train.py`` does (``build_sft_config``,
``default_lora_config``, ``apply_train_perf``) and times ``trainer.train()`` on
synthetic fixed-length sequences, forcing one loop count ``N`` per run.

Spill handling: the process is capped with
``torch.cuda.set_per_process_memory_fraction(--vram-cap)`` (0.92, under the
project's 0.95 spill threshold) so a configuration that would page to system RAM under WDDM
fails fast with an OOM instead of running at a few tokens/s.  A length counts
as ``fits`` only if it neither OOMed nor made the caching allocator flush and
retry (``num_alloc_retries == 0``): then its peak reserved memory is what an
uncapped run would reserve as well.

Usage::

    uv run python scripts/bench_train.py --perf baseline --out bench.jsonl \
        --lengths 1024,2048,3072,4096 --loops 1,2,4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from gemmaloop.model.load import IMAGE_TOKEN_IDS, ModelSpec, load_model  # noqa: E402
from gemmaloop.model.looped import LoopConfig  # noqa: E402
from gemmaloop.train.perf import BASELINE_PERF, TrainPerfConfig, apply_train_perf  # noqa: E402
from gemmaloop.train.trainer import (  # noqa: E402
    LoopSFTTrainer,
    default_lora_config,
    freeze_base_model,
)
from scripts.train import build_sft_config  # noqa: E402

LOGGER = logging.getLogger("bench_train")

PRESETS: dict[str, TrainPerfConfig] = {
    # the trainer as of c8ce43b: TRL fused head, input-grad hook, vision on GPU
    "baseline": BASELINE_PERF,
    # cumulative stages: each adds one lever to the previous one
    "s1_lean_head": TrainPerfConfig(input_grads=True, offload_vision=False, banded_sliding=False),
    "s2_no_input_grads": TrainPerfConfig(offload_vision=False, banded_sliding=False),
    "s3_vision_offload": TrainPerfConfig(banded_sliding=False),
    "s4_banded_sliding": TrainPerfConfig(),
    "s5_offload_checkpoints": TrainPerfConfig(offload_checkpoints=True),
    "all": TrainPerfConfig(),
}


class SlowStepAbort(RuntimeError):
    """A micro step took so long that the GPU is almost certainly paging (WDDM spill)."""


class StepTimer(TrainerCallback):
    """Wall time of every micro step (the last one of an optimizer step includes it).

    Raises :class:`SlowStepAbort` when a micro step is slower than
    ``max_seconds_per_token * length`` so a spilling run cannot stall the
    benchmark for many minutes.
    """

    def __init__(self, max_seconds_per_token: float = 0.02) -> None:
        self.durations: list[float] = []
        self._last: float | None = None
        self.max_seconds_per_token = max_seconds_per_token
        self.length = 0
        self.min_free_bytes = 192 * 2**20

    @staticmethod
    def _now() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self) -> None:
        self.durations = []
        self._last = None

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._last = self._now()
        return control

    def _mark(self) -> None:
        now = self._now()
        free, _total = torch.cuda.mem_get_info()
        if free < self.min_free_bytes:
            # device-wide: another process grew (or our own non-allocator use did);
            # stop before WDDM starts paging VRAM to system RAM
            raise SlowStepAbort(f"device free memory {free / 2**30:.2f} GiB < {self.min_free_bytes / 2**30:.2f} GiB")
        if self._last is not None:
            self.durations.append(now - self._last)
            limit = self.max_seconds_per_token * max(self.length, 1024)
            if len(self.durations) > 1 and self.durations[-1] > limit:
                raise SlowStepAbort(f"micro step took {self.durations[-1]:.1f}s > {limit:.1f}s")
        self._last = now

    def on_substep_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self._mark()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self._mark()
        return control


def synthetic_rows(length: int, count: int, *, prompt: int, seed: int) -> list[dict[str, list[int]]]:
    generator = torch.Generator().manual_seed(seed)
    high = min(IMAGE_TOKEN_IDS) - 1  # stay clear of image placeholder ids
    rows = []
    for _ in range(count):
        ids = torch.randint(3, high, (length,), generator=generator).tolist()
        labels = list(ids)
        for position in range(min(prompt, length)):
            labels[position] = -100
        rows.append({"input_ids": ids, "labels": labels})
    return rows


def onehot_probs(n_loops: int) -> tuple[float, ...]:
    return tuple(1.0 if n == n_loops else 0.0 for n in range(1, 5))


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure(trainer: LoopSFTTrainer, timer: StepTimer, length: int, n_loops: int, args) -> dict[str, Any]:
    trainer.loop_probs = onehot_probs(n_loops)
    trainer.train_dataset = Dataset.from_list(
        synthetic_rows(
            length,
            args.grad_accum * args.opt_steps,
            prompt=args.prompt_tokens,
            seed=length * 10 + n_loops,
        )
    )
    total = torch.cuda.get_device_properties(0).total_memory
    _cleanup()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()
    timer.reset()
    timer.length = length
    if getattr(trainer, "checkpoint_policy", None) is not None:
        trainer.checkpoint_policy.skipped = trainer.checkpoint_policy.checkpointed = 0
    oom = False
    slow = False
    error = None
    started = time.perf_counter()
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:  # also raised when the cap is hit
        oom = True
        error = str(exc).splitlines()[0][:200]
    except SlowStepAbort as exc:
        slow = True
        error = str(exc)
    wall = time.perf_counter() - started
    stats = torch.cuda.memory_stats()
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    retries = stats.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0)
    if oom or slow:
        trainer.model.zero_grad(set_to_none=True)
        if trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
    _cleanup()

    durations = list(timer.durations)
    timed = durations[args.warmup_micro:]
    rates = [length / d for d in timed if d > 0]
    tok_s = (length * len(timed) / sum(timed)) if timed and not (oom or slow) else None
    median = statistics.median(rates) if rates else None
    collapsed = slow or bool(median and min(rates) < 0.5 * median)
    fits = (
        not oom
        and retries == 0
        and peak_reserved <= args.cap_bytes
        and peak_reserved <= args.vram_cap * total
        and not collapsed
        and tok_s is not None
    )
    policy = getattr(trainer, "checkpoint_policy", None)
    record = {
        "perf": args.perf,
        "ckpt_skipped_micro_steps": None if policy is None else policy.skipped,
        "perf_config": trainer._gemmaloop_perf_summary,
        "length": length,
        "n_loops": n_loops,
        "micro_steps": len(durations),
        "timed_micro_steps": len(timed),
        "tok_s": round(tok_s, 1) if tok_s else None,
        "step_tok_s": [round(r, 1) for r in rates],
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "total_vram_gib": round(total / 2**30, 3),
        "free_at_start_gib": round(args.free_at_start / 2**30, 3),
        "cap_gib": round(args.cap_bytes / 2**30, 3),
        "alloc_retries": int(retries),
        "oom": oom,
        "collapsed": collapsed,
        "fits": fits,
        "wall_s": round(wall, 1),
        "error": error,
    }
    LOGGER.info(
        "L=%d N=%d tok/s=%s peak_reserved=%.2f GiB peak_alloc=%.2f GiB retries=%d oom=%s fits=%s",
        length,
        n_loops,
        record["tok_s"],
        record["peak_reserved_gib"],
        record["peak_allocated_gib"],
        retries,
        oom,
        fits,
    )
    return record


def profile_run(trainer, timer, length: int, n_loops: int, args) -> dict[str, Any]:
    """``measure`` under torch.profiler; writes CUDA/CPU time tables next to ``--out``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        record = measure(trainer, timer, length, n_loops, args)
    averages = prof.key_averages()
    device_total = sum(item.self_device_time_total for item in averages)
    path = args.out.with_suffix(f".L{length}.N{n_loops}.profile.txt")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"perf={args.perf} L={length} N={n_loops} record={json.dumps(record)}\n")
        handle.write(
            f"total self device time: {device_total / 1e6:.3f} s over {record['micro_steps']} micro steps\n\n"
        )
        handle.write(averages.table(sort_by="self_device_time_total", row_limit=40, max_name_column_width=90))
        handle.write("\n\n")
        handle.write(averages.table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=90))
    record["profile"] = str(path)
    record["device_busy_s"] = round(device_total / 1e6, 3)
    LOGGER.info("profile written to %s", path)
    return record


def sdpa_report(lengths: list[int]) -> list[dict[str, Any]]:
    """Which SDPA kernels accept Gemma 4 E2B's attention shapes (fwd+bwd), and their speed."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = torch.device("cuda")
    results = []
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    for length in lengths:
        for kind, head_dim, masked in (("sliding", 256, True), ("full", 512, False)):
            q = torch.randn(1, 8, length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 8, length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(1, 8, length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            mask = None
            if masked:
                i = torch.arange(length, device=device)
                mask = ((i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 512))[None, None]
            for backend in backends:
                entry = {"length": length, "layer": kind, "head_dim": head_dim, "backend": backend.name}
                try:
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    with sdpa_kernel([backend]):
                        for _ in range(2):
                            out = torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, attn_mask=mask, is_causal=mask is None
                            )
                            out.sum().backward()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(3):
                            out = torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, attn_mask=mask, is_causal=mask is None
                            )
                            out.sum().backward()
                        torch.cuda.synchronize()
                    entry["ms_fwd_bwd"] = round((time.perf_counter() - start) / 3 * 1e3, 2)
                    entry["peak_extra_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
                    entry["ok"] = True
                except Exception as exc:  # noqa: BLE001 - report every refusal
                    entry["ok"] = False
                    entry["error"] = str(exc).splitlines()[0][:160]
                q.grad = k.grad = v.grad = None
                results.append(entry)
                _cleanup()
            del q, k, v, mask
            _cleanup()
    return results


def other_gpu_python_processes() -> list[str]:
    """Python processes other than this one that hold a CUDA context.

    Two training/benchmark processes on the 12 GiB card together exceed VRAM
    and WDDM pages the overflow to system RAM (an earlier run of two
    benchmark scripts overlapping ended in a driver crash).  The per-process
    allocator cap cannot see other processes, so the benchmark refuses to
    start next to one.
    """
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    found = []
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        if "python" in name.lower() and pid.strip() != str(os.getpid()):
            found.append(line.strip())
    return found


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perf", default="all", choices=sorted(PRESETS))
    parser.add_argument("--token-chunk", type=int, default=None, help="override lm_head_token_chunk")
    parser.add_argument("--vocab-chunk", type=int, default=None, help="override lm_head_vocab_chunk")
    parser.add_argument("--no-host-pcores", action="store_true", help="host_performance_cores=False")
    parser.add_argument(
        "--ckpt-skip",
        type=int,
        default=None,
        help="override checkpoint_skip_token_calls (micro steps with tokens * layer calls <= this "
        "run without gradient checkpointing)",
    )
    parser.add_argument("--lengths", type=_int_list, default=[1024, 2048, 3072, 4096])
    parser.add_argument("--loops", type=_int_list, default=[1, 2, 4])
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--opt-steps", type=int, default=2)
    parser.add_argument(
        "--warmup-micro",
        type=int,
        default=None,
        help="micro steps excluded from tok/s (default: the whole first optimizer step, "
        "which includes bitsandbytes' 8-bit state initialisation)",
    )
    parser.add_argument("--prompt-tokens", type=int, default=32, help="unlabelled prefix per sequence")
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help="fraction of total VRAM the allocator may reserve (0.95 is the project spill rule; "
        "0.92 leaves margin for the CUDA context and cuBLAS/cuDNN workspaces)",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="start even if another python process holds the GPU (unsafe: the cap is per process)",
    )
    parser.add_argument(
        "--free-margin-gib",
        type=float,
        default=1.0,
        help="the cap is also <= VRAM free at start minus this margin (desktop headroom, CUDA context "
        "growth: with 0.5 a run at the cap left 0.13 GiB free on the device)",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSONL file (appended)")
    parser.add_argument("--sdpa-report", action="store_true", help="only probe SDPA kernels and exit")
    parser.add_argument("--keep-going", action="store_true", help="measure longer lengths after an OOM")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the first (length, N) run with torch.profiler and write <out>.profile.txt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.warmup_micro is None:
        args.warmup_micro = args.grad_accum
    if not torch.cuda.is_available():
        raise SystemExit("bench_train.py needs a CUDA GPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    others = [] if args.allow_shared_gpu else other_gpu_python_processes()
    if others:
        raise SystemExit(f"another python process holds the GPU, refusing to start: {others}")
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    args.free_at_start = int(free)
    args.cap_bytes = int(min(args.vram_cap * total, free - args.free_margin_gib * 2**30))
    torch.cuda.set_per_process_memory_fraction(args.cap_bytes / total)
    LOGGER.info(
        "VRAM total %.2f GiB, free at start %.2f GiB, allocator cap %.2f GiB",
        total / 2**30,
        free / 2**30,
        args.cap_bytes / 2**30,
    )
    if args.sdpa_report:
        for entry in sdpa_report(args.lengths):
            LOGGER.info("sdpa %s", entry)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sdpa": entry}) + "\n")
        return 0

    perf = PRESETS[args.perf]
    overrides = {}
    if args.token_chunk:
        overrides["lm_head_token_chunk"] = args.token_chunk
    if args.vocab_chunk:
        overrides["lm_head_vocab_chunk"] = args.vocab_chunk
    if args.ckpt_skip is not None:
        overrides["checkpoint_skip_token_calls"] = args.ckpt_skip
    if args.no_host_pcores:
        overrides["host_performance_cores"] = False
    if overrides:
        perf = TrainPerfConfig(**{**perf.to_dict(), **overrides})

    loop = LoopConfig(n_loops=1, bridge="scalar", bridge_init=0.99, ple_mode="shared")
    model, processor, _plan = load_model(ModelSpec("e2b", "bf16", adapters=(), loop=loop))
    text_model = model.model.language_model
    freeze_base_model(model)

    max_length = max(args.lengths)
    out_dir = Path(os.environ.get("TEMP", "/tmp")) / "gemmaloop-bench-train"
    training_args = build_sft_config(
        out_dir,
        max_length=max_length,
        seed=0,
        num_train_epochs=1,
        max_steps=args.opt_steps,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="no",
        logging_steps=1000,
        disable_tqdm=True,
        warmup_steps=0,
    )
    timer = StepTimer()
    trainer = LoopSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(synthetic_rows(64, 4, prompt=8, seed=0)),
        processing_class=processor,
        peft_config=default_lora_config(),
        loop_text_model=text_model,
        loop_enabled=True,
        loop_seed=0,
        loop_probs=onehot_probs(1),
        callbacks=[timer],
    )
    summary = apply_train_perf(trainer, perf)
    trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    summary["trainable_tensors"] = len(trainable)
    summary["trainable_params"] = int(sum(p.numel() for _, p in trainable))
    summary["non_lora_trainable"] = [n for n, _ in trainable if "lora_" not in n]
    trainer._gemmaloop_perf_summary = summary
    LOGGER.info("perf=%s summary=%s", args.perf, summary)

    failed_at: dict[int, int] = {}
    for length in sorted(args.lengths):
        for n_loops in sorted(args.loops):
            if not args.keep_going and any(
                n <= n_loops and fail <= length for n, fail in failed_at.items()
            ):
                LOGGER.info("skip L=%d N=%d (a smaller config already failed)", length, n_loops)
                continue
            if args.profile:
                record = profile_run(trainer, timer, length, n_loops, args)
                args.profile = False
            else:
                record = measure(trainer, timer, length, n_loops, args)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if not record["fits"]:
                failed_at[n_loops] = min(failed_at.get(n_loops, length), length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
