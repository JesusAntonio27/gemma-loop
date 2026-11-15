"""T-dial-bench: measure dial training throughput and peak VRAM (GPU only).

The benchmark runs the **actual** dial training path (``LoopSFTTrainer`` with
``dial_enabled=True``) at ``--length 8192`` for the three registered blocks
``25-29``, ``16-18`` and ``15-29`` at ``T=4`` and ``T=8``.  Each measurement
uses the truncated ``k=3`` window, ``bridge='gch'`` and
``adapter_policy='loop_only'`` / ``lora_target_layers='loop_block'``, i.e. the
registered dial recipe.

It is GPU-only and **must not** be run on CPU.  The job takes the project GPU
lock (``<artifacts>/gpu.lock``) before touching the device and writes one JSON
record per configuration to ``<artifacts>/perf/dial_bench.json`` (override with
``--out``).  Six configurations at two timed steps plus one warm-up are well
inside the 0.5 GPU-h budget:

    uv run python scripts/bench_dial.py \
        --blocks 25-29,16-18,15-29 --horizons 4,8 --length 8192 \
        --opt-steps 2 --warmup-steps 1

Expected: roughly 10-25 min on the RTX 5070 (12 GiB) including six model
loads; abort with ``--keep-going`` off if a configuration OOMs.  It never runs
in CI (no GPU).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
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

from gemmaloop import paths  # noqa: E402
from gemmaloop.gpulock import require_gpu_lock  # noqa: E402
from gemmaloop.model.layer_plan import parse_block  # noqa: E402
from gemmaloop.model.load import IMAGE_TOKEN_IDS, ModelSpec, load_model  # noqa: E402
from gemmaloop.model.looped import LoopConfig, set_n_loops  # noqa: E402
from gemmaloop.data.teacher_topk import write_json_atomic  # noqa: E402
from gemmaloop.train.devices import model_compute_device  # noqa: E402
from gemmaloop.train.fused_ce import chunked_softcap_logprobs  # noqa: E402
from gemmaloop.train.perf import TrainPerfConfig, apply_train_perf  # noqa: E402
from gemmaloop.train.teacher_collator import TeacherDataCollator  # noqa: E402
from gemmaloop.train.trainer import (  # noqa: E402
    LoopSFTTrainer,
    default_lora_config,
    enable_loop_parameters,
    freeze_base_model,
)
from scripts.bench_train import other_gpu_python_processes  # noqa: E402
from scripts.train import build_sft_config  # noqa: E402

LOGGER = logging.getLogger("bench_dial")
DEFAULT_BLOCKS = "25-29,16-18,15-29"
DEFAULT_HORIZONS = "4,8"


class StepTimer(TrainerCallback):
    """Wall time of every optimizer (and micro) step, with CUDA synchronisation."""

    def __init__(self) -> None:
        self.durations: list[float] = []
        self._last: float | None = None

    @staticmethod
    def _now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self) -> None:
        self.durations = []
        self._last = None

    def _mark(self) -> None:
        now = self._now()
        if self._last is not None:
            self.durations.append(now - self._last)
        self._last = now

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._last = self._now()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self._mark()
        return control


def _onehot_horizon(horizon: int) -> tuple[float, ...]:
    return tuple(1.0 if index + 1 == horizon else 0.0 for index in range(horizon))


def synthetic_rows(
    length: int,
    count: int,
    *,
    seed: int,
    teacher_top_k: int = 64,
    vocab_size: int | None = None,
) -> list[dict[str, Any]]:
    """Token rows plus a synthetic top-k teacher, at exactly ``length``.

    The teacher has the T-heal layout/probability shape so the benchmark
    measures the real dial kernel (top-64 gather + tail), not a one-hot CE.
    """
    generator = torch.Generator().manual_seed(seed)
    high = (min(IMAGE_TOKEN_IDS) - 1) if vocab_size is None else int(vocab_size) - 1
    # A fixed, unique top-k support per position is enough to exercise the
    # gather/tail kernel; it avoids materialising a 262k permutation per token.
    indices = torch.arange(teacher_top_k, dtype=torch.int32).unsqueeze(0).repeat(length, 1)
    probs = torch.softmax(
        torch.randn(length, teacher_top_k, generator=generator), dim=-1
    ) * 0.9
    probs = probs.to(torch.float16)
    tail = torch.full((length,), 0.1, dtype=torch.float32)
    rows = []
    for _ in range(count):
        ids = torch.randint(3, high, (length,), generator=generator).tolist()
        rows.append(
            {
                "input_ids": ids,
                "labels": list(ids),
                "teacher_topk_probs": probs.tolist(),
                "teacher_topk_indices": indices.tolist(),
                "teacher_tail_prob": tail.tolist(),
            }
        )
    return rows


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def measure(
    block: str,
    horizon: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load the block, train the dial objective for a few steps, measure."""
    _cleanup()
    loop = LoopConfig(
        n_loops=1,
        bridge="gch",
        bridge_init=0.99,
        ple_mode="shared",
        block=parse_block(block),
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
    )
    model, processor, plan = load_model(ModelSpec("e2b", "bf16", adapters=(), loop=loop))
    text_model = getattr(model.model, "language_model", model.model)
    freeze_base_model(model)

    out_dir = Path(args.work_dir) / f"dial-{block.replace('-', '_')}-T{horizon}"
    training_args = build_sft_config(
        out_dir,
        max_length=args.length,
        seed=0,
        num_train_epochs=1,
        max_steps=args.opt_steps + args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="no",
        logging_steps=1000,
        disable_tqdm=True,
        warmup_steps=0,
    )
    timer = StepTimer()
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    tokenizer = getattr(processor, "tokenizer", processor)
    train_dataset = Dataset.from_list(
        synthetic_rows(
            args.length,
            (args.opt_steps + args.warmup_steps) * args.grad_accum,
            seed=int(block.split("-")[0]) * 100 + horizon,
            vocab_size=args.vocab_size,
        )
    )
    data_collator = TeacherDataCollator(
        DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or 0)
    )
    trainer = LoopSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        peft_config=default_lora_config(target_layers="loop_block", plan=plan),
        loop_text_model=text_model,
        loop_enabled=True,
        loop_seed=0,
        loop_probs=_onehot_horizon(horizon),
        dial_enabled=True,
        dial_window=args.window,
        dial_horizon_probs=_onehot_horizon(horizon),
        callbacks=[timer],
    )
    apply_train_perf(trainer, TrainPerfConfig())

    torch.cuda.reset_peak_memory_stats()
    oom = False
    error = None
    started = time.perf_counter()
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:  # cap or real OOM
        oom = True
        error = str(exc).splitlines()[0][:200]
    wall = time.perf_counter() - started
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory

    timed = timer.durations[args.warmup_steps :]
    tokens_per_step = args.length * args.grad_accum
    tok_s = (
        tokens_per_step * len(timed) / sum(timed)
        if timed and not oom and sum(timed) > 0
        else None
    )
    record = {
        "objective": "dial",
        "block": block,
        "T": horizon,
        "window": args.window,
        "length": args.length,
        "grad_accum": args.grad_accum,
        "gradient_checkpointing": bool(training_args.gradient_checkpointing),
        "tok_s": round(tok_s, 1) if tok_s else None,
        "step_s": [round(value, 3) for value in timed],
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "total_vram_gib": round(total / 2**30, 3),
        "wall_s": round(wall, 1),
        "oom": oom,
        "fits": bool(not oom and peak_reserved <= args.vram_cap * total),
        "error": error,
    }
    LOGGER.info(
        "block=%s T=%d tok/s=%s peak_reserved=%.2f GiB oom=%s",
        block,
        horizon,
        record["tok_s"],
        record["peak_reserved_gib"],
        oom,
    )
    del trainer, model, text_model
    _cleanup()
    return record


def _time_trainer(trainer, timer, args, record):
    torch.cuda.reset_peak_memory_stats()
    oom = False
    error = None
    started = time.perf_counter()
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:  # cap or real OOM
        oom = True
        error = str(exc).splitlines()[0][:200]
    wall = time.perf_counter() - started
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    timed = timer.durations[args.warmup_steps :]
    tokens_per_step = args.length * args.grad_accum
    tok_s = (
        tokens_per_step * len(timed) / sum(timed)
        if timed and not oom and sum(timed) > 0
        else None
    )
    record.update(
        {
            "tok_s": round(tok_s, 1) if tok_s else None,
            "step_s": [round(value, 3) for value in timed],
            "peak_reserved_gib": round(peak_reserved / 2**30, 3),
            "peak_allocated_gib": round(peak_allocated / 2**30, 3),
            "total_vram_gib": round(total / 2**30, 3),
            "wall_s": round(wall, 1),
            "oom": oom,
            "fits": bool(not oom and peak_reserved <= args.vram_cap * total),
            "error": error,
        }
    )
    return record


def measure_heal(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the registered O-H single-exit heal step (teacher KL + anchor)."""
    _cleanup()
    # O-H is unlooped: no LoopConfig, adapters always active on the single pass.
    model, processor, plan = load_model(
        ModelSpec("e2b", "bf16", adapters=(), loop=None)
    )
    text_model = getattr(model.model, "language_model", model.model)
    freeze_base_model(model)
    out_dir = Path(args.work_dir) / "heal-oh"
    training_args = build_sft_config(
        out_dir,
        max_length=args.length,
        seed=0,
        num_train_epochs=1,
        max_steps=args.opt_steps + args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="no",
        logging_steps=1000,
        disable_tqdm=True,
        warmup_steps=0,
    )
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    tokenizer = getattr(processor, "tokenizer", processor)
    dataset = Dataset.from_list(
        synthetic_rows(
            args.length,
            (args.opt_steps + args.warmup_steps) * args.grad_accum,
            seed=919,
            vocab_size=args.vocab_size,
        )
    )
    data_collator = TeacherDataCollator(
        DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or 0)
    )
    timer = StepTimer()
    grad_counts = {"n": 0}

    class _GradCheck(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
            grad_counts["n"] = max(
                grad_counts["n"],
                sum(
                    1
                    for parameter in model.parameters()
                    if parameter.grad is not None
                    and bool(parameter.grad.detach().abs().sum() > 0)
                ),
            )
            return control

    trainer = LoopSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
        peft_config=default_lora_config(target_layers="loop_block", plan=plan),
        loop_text_model=text_model,
        loop_enabled=False,
        heal_teacher=True,
        callbacks=[timer, _GradCheck()],
    )
    apply_train_perf(trainer, TrainPerfConfig())
    record = _time_trainer(
        trainer,
        timer,
        args,
        {"objective": "heal_oh", "block": "", "T": 1, "window": args.window,
         "length": args.length, "grad_accum": args.grad_accum,
         "gradient_checkpointing": bool(training_args.gradient_checkpointing)},
    )
    record["gradient_parameters"] = int(grad_counts["n"])
    if not record["oom"] and int(grad_counts["n"]) <= 0:
        raise RuntimeError(
            "the O-H heal bench produced no gradients; the LoRA path is inactive"
        )
    LOGGER.info(
        "heal O-H tok/s=%s peak_reserved=%.2f GiB oom=%s",
        record["tok_s"],
        record["peak_reserved_gib"],
        record["oom"],
    )
    del trainer, model, text_model
    _cleanup()
    return record


def measure_stage2(
    block: str, horizon: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure a Stage 2 RL-like step at N=6/8 on a prompt + 8192 completion.

    The step is a looped forward with the k=3 window, per-token log-prob on the
    completion and a backward through the fused chunked head, at the registered
    gradient accumulation.  This is the memory/speed shape Stage 2 charges.
    """
    from peft import get_peft_model

    _cleanup()
    loop = LoopConfig(
        n_loops=1,
        bridge="gch",
        bridge_init=0.99,
        ple_mode="shared",
        block=parse_block(block),
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
    )
    model, processor, plan = load_model(ModelSpec("e2b", "bf16", adapters=(), loop=loop))
    text_model = getattr(model.model, "language_model", model.model)
    freeze_base_model(model)
    model = get_peft_model(model, default_lora_config(target_layers="loop_block", plan=plan))
    enable_loop_parameters(text_model)
    checkpointing_enabled = False
    try:
        text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        checkpointing_enabled = True
    except (AttributeError, TypeError):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            checkpointing_enabled = True
        except (AttributeError, TypeError):
            checkpointing_enabled = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5)
    generator = torch.Generator().manual_seed(int(block.split("-")[0]) * 100 + horizon)
    length = args.length
    prompt_length = max(1, length // 16)
    device = model_compute_device(model)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.base_model.model.lm_head
    weight = head.weight
    softcap = getattr(text_model.config, "final_logit_softcapping", None)

    torch.cuda.reset_peak_memory_stats()
    oom = False
    error = None
    durations: list[float] = []
    started = time.perf_counter()
    try:
        for step in range(args.opt_steps + args.warmup_steps):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            for _micro in range(args.grad_accum):
                input_ids = torch.randint(
                    3,
                    int(args.vocab_size) - 1,
                    (1, length),
                    generator=generator,
                ).to(device)
                set_n_loops(text_model, int(horizon))
                outputs = text_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    exits=[int(horizon)],
                    window=int(args.window),
                )
                hidden = outputs.exit_hidden_states[int(horizon)][0]
                targets = input_ids[0, prompt_length:]
                log_probs, _entropy, _top1 = chunked_softcap_logprobs(
                    hidden[prompt_length - 1 : -1],
                    weight,
                    targets,
                    softcap,
                    token_chunk=256,
                    vocab_chunk=8192,
                )
                loss = -log_probs.mean()
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if step >= args.warmup_steps:
                durations.append(time.perf_counter() - step_started)
    except torch.OutOfMemoryError as exc:
        oom = True
        error = str(exc).splitlines()[0][:200]
    wall = time.perf_counter() - started
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    tokens_per_step = (length - prompt_length) * args.grad_accum
    tok_s = (
        tokens_per_step * len(durations) / sum(durations)
        if durations and not oom and sum(durations) > 0
        else None
    )
    record = {
        "objective": "stage2_rl",
        "block": block,
        "T": horizon,
        "window": args.window,
        "length": length,
        "grad_accum": args.grad_accum,
        "gradient_checkpointing": checkpointing_enabled,
        "tok_s": round(tok_s, 1) if tok_s else None,
        "step_s": [round(value, 3) for value in durations],
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "total_vram_gib": round(total / 2**30, 3),
        "wall_s": round(wall, 1),
        "oom": oom,
        "fits": bool(not oom and peak_reserved <= args.vram_cap * total),
        "error": error,
    }
    LOGGER.info(
        "stage2 block=%s N=%d tok/s=%s peak_reserved=%.2f GiB oom=%s",
        block,
        horizon,
        record["tok_s"],
        record["peak_reserved_gib"],
        oom,
    )
    del model, text_model
    _cleanup()
    return record


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _block_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blocks", default=DEFAULT_BLOCKS, help="comma-separated blocks")
    parser.add_argument(
        "--horizons", default=DEFAULT_HORIZONS, help="comma-separated T values"
    )
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=min(IMAGE_TOKEN_IDS) - 1,
        help="tokenizer vocabulary size (CPU smoke uses the tiny model's)",
    )
    parser.add_argument("--opt-steps", type=int, default=2, help="timed optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="registered gradient accumulation (Stage 1 and Stage 2)",
    )
    parser.add_argument(
        "--stage2-horizons",
        default="6,8",
        help="Stage 2 RL step horizons (comma-separated N values)",
    )
    parser.add_argument(
        "--skip-heal",
        action="store_true",
        help="skip the O-H heal step (it is part of the registered bench)",
    )
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help="fraction of VRAM the allocator may reserve; 0.95 is the spill rule",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.artifacts_dir() / "perf" / "dial_bench.json",
        help="JSON file (rewritten after every configuration)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch directory for trainer output (default: under the system temp dir)",
    )
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    require_gpu_lock()
    if not torch.cuda.is_available():
        raise SystemExit("bench_dial.py needs a CUDA GPU (CPU is forbidden)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir is None:
        args.work_dir = Path(tempfile_root()) / "gemmaloop-bench-dial"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    others = [] if args.allow_shared_gpu else other_gpu_python_processes()
    if others:
        raise SystemExit(
            f"another python process holds the GPU, refusing to start: {others}"
        )

    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    cap = min(args.vram_cap * total, free - 0.5 * 2**30)
    torch.cuda.set_per_process_memory_fraction(max(cap, 0.1 * total) / total)
    LOGGER.info(
        "VRAM total %.2f GiB, free %.2f GiB, allocator cap %.2f GiB",
        total / 2**30,
        free / 2**30,
        cap / 2**30,
    )

    records: list[dict[str, Any]] = []

    def append(record: dict[str, Any]) -> None:
        records.append(record)
        write_json_atomic(args.out, records)

    for block in _block_list(args.blocks):
        for horizon in _int_list(args.horizons):
            record = measure(block, horizon, args)
            append(record)
            if record["oom"] and not args.keep_going:
                LOGGER.error("stopping after OOM at block=%s T=%d", block, horizon)
                return 1
        for horizon in _int_list(args.stage2_horizons):
            record = measure_stage2(block, horizon, args)
            append(record)
            if record["oom"] and not args.keep_going:
                LOGGER.error(
                    "stopping after OOM at Stage 2 block=%s N=%d", block, horizon
                )
                return 1
    if not args.skip_heal:
        record = measure_heal(args)
        append(record)
        if record["oom"] and not args.keep_going:
            LOGGER.error("stopping after OOM in the O-H heal step")
            return 1
    LOGGER.info("wrote %d records to %s", len(records), args.out)
    return 0


def tempfile_root() -> str:
    import os

    return os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"


if __name__ == "__main__":
    sys.exit(main())
