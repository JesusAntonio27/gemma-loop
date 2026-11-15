"""GPU probe-memory smoke for the registered heal monitor (operational check).

The native healpilot audit requires one measured probe on the real 12 GiB card:
``monitor_rows`` are rendered at ``max_length=8192`` and a probe evaluates four
such rows at ``N=8`` while the *training state* is resident (model + LoRA +
loop bridge + the real optimizer and its materialised state + gradient
buffers).  Probe memory had never been measured, and with the monitor aborting
after two consecutive probe failures a deterministic probe OOM would stop every
registered arm.

This script deliberately does not read the registered MONITOR/pilot/evaluation/
T-fit/selection/confirmatory corpora.  The probe rows are synthetic: a fixed
word pattern inside the real Gemma 4 chat template, rendered with
``assistant_labels`` (thinking mode, so the final channel is left open) and
padded/truncated to exactly ``--max-length`` tokens with a tokenizer.

The training state is built the same way ``scripts/train.py`` builds the
heaviest registered dial arm (E2B, block 25-29, zero-init per-channel bridge,
``loop_only`` adapters on the loop block, bf16, CPU PLE, lean LM head, banded
sliding attention, vision offload, bitsandbytes 8-bit AdamW).  One real
optimizer step on a tiny synthetic batch materialises the optimizer state; the
probe then calls the same training entry point used by
``HealMonitorCallback._run_probe``: :meth:`gemmaloop.train.heal_probes.ProbeRunner.run`
with batches built by :func:`gemmaloop.train.heal_monitor.build_probe_batches`
and prompts by :func:`gemmaloop.train.heal_monitor.build_prompt_tensors`.

Measurement: ``torch.cuda.reset_peak_memory_stats()`` before the training-state
baseline and before every probe phase, then ``max_memory_allocated`` /
``max_memory_reserved``.  A phase passes when it does not OOM, has no
``probe_errors`` entry, produced the expected number of generation rows when
generation was requested, and its reserved headroom against the CUDA allocator
cap is at least ``--headroom-gib`` (0.25 GiB, the qualification rule).  OOMs
and other probe errors are caught, recorded as FAIL and the report is still
written.

``--rows`` is the number of probe rows; ``--probe-batch-size`` is the forward
micro-batch size (training default: one row, because four registered 8192-token
rows do not fit in one forward on the 12 GiB card).  They are independent, so
the orchestrator can measure the registered 4 rows at micro-batch 1 with 8
generation prompts, exactly matching training.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path
from gemmaloop.paths import artifacts_dir
from gemmaloop.train.devices import model_compute_device
from gemmaloop.train.heal_monitor import (
    MonitorConfig,
    build_probe_batches,
    build_prompt_tensors,
)
from gemmaloop.train.heal_probes import (
    ProbeConfig,
    ProbeRunner,
    generation_offload_devices,
    text_model_and_head,
)

__all__ = [
    "HEADROOM_GIB",
    "REGISTERED_ANCHOR_NS",
    "REGISTERED_BLOCK",
    "REGISTERED_BRIDGE",
    "REGISTERED_EVAL_NS",
    "REGISTERED_GENERATION_MAX_NEW_TOKENS",
    "REGISTERED_GENERATION_NS",
    "REGISTERED_LOOP_PROBS",
    "REGISTERED_MAX_LENGTH",
    "REGISTERED_NLL_NS",
    "build_parser",
    "build_probe_batches_for_smoke",
    "build_synthetic_messages",
    "build_synthetic_probe_records",
    "build_synthetic_probe_dataset",
    "decide_status",
    "probe_config_for_n",
    "write_reports",
]

# The heaviest registered dial arm is an L arm (E2B student, dial objective,
# loop block 25-29, zero-init per-channel bridge).  These are the registered
# values the smoke must reproduce; N=8 is the largest anchor the dial horizon
# can sample and the probe must survive.
REGISTERED_BLOCK = (25, 29)
REGISTERED_BRIDGE = "gch"
REGISTERED_LOOP_PROBS = (0.0, 0.0, 0.0, 0.70, 0.075, 0.075, 0.075, 0.075)
REGISTERED_DIAL_WINDOW = 3
REGISTERED_MAX_LENGTH = 8192
REGISTERED_ROWS = 4
REGISTERED_EVAL_NS = tuple(MonitorConfig().anchor_ns) + tuple(
    MonitorConfig().nll_ns
) + (1,)
REGISTERED_EVAL_NS = tuple(sorted(set(REGISTERED_EVAL_NS)))
REGISTERED_ANCHOR_NS = tuple(MonitorConfig().anchor_ns)
REGISTERED_NLL_NS = tuple(MonitorConfig().nll_ns)
REGISTERED_GENERATION_NS = tuple(MonitorConfig().generation_ns)
REGISTERED_GENERATION_MAX_NEW_TOKENS = int(
    MonitorConfig().generation_max_new_tokens
)
HEADROOM_GIB = 0.25

_PATTERN = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
)

_PROTECTED_NAMES = (
    "reserved_ids.json",
    "pilot_manifest.json",
    "pilot_splits.json",
    "monitor_splits.json",
    "monitor_rows.jsonl",
    "monitor.jsonl",
    "splits.json",
    "manifest.json",
    "pilot_tfit.jsonl",
    "pilot_selection.jsonl",
    "confirmatory.jsonl",
    "eval_rows.jsonl",
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def _default_out() -> Path:
    return artifacts_dir() / "smoke" / "probe_memory"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the heal-monitor probe memory on the real GPU while the "
            "heaviest registered dial training state is resident."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "directory for probe_memory.json and probe_memory.md "
            "(default: <artifacts>/smoke/probe_memory)"
        ),
    )
    parser.add_argument(
        "--size",
        default="e2b",
        choices=("e2b",),
        help="student size; the registered heaviest dial arm uses E2B",
    )
    parser.add_argument(
        "--quant",
        default="bf16",
        choices=("bf16", "nf4"),
        help="student quantisation (registered E2B student is bf16)",
    )
    parser.add_argument(
        "--arm",
        default="L",
        choices=("L",),
        help="registered objective; only the dial arm L is accepted here",
    )
    parser.add_argument(
        "--block",
        default="-".join(str(value) for value in REGISTERED_BLOCK),
        help="inclusive loop block START-END (registered 25-29)",
    )
    parser.add_argument(
        "--bridge",
        default=REGISTERED_BRIDGE,
        choices=("gch", "scalar", "damped", "none"),
        help="loop bridge (registered zero-init per-channel g)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=REGISTERED_ROWS,
        help="probe rows per batch (registered probe batch is 4)",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=int(MonitorConfig().probe_batch_size),
        help=(
            "rows per probe forward micro-batch, independent of --rows "
            "(training default 1; 4 rows at micro-batch 1 matches training)"
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=REGISTERED_MAX_LENGTH,
        help="probe row length in tokens (registered training/probe 8192)",
    )
    parser.add_argument(
        "--ns",
        default="1,4,8",
        help="comma-separated probe eval N values to measure (N=8 is required)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="do not exercise the registered generation path (1/2/4, cap 512)",
    )
    parser.add_argument(
        "--generation-prompts",
        type=int,
        default=int(MonitorConfig().generation_prompts),
        help="generation prompts per N (registered monitor uses 8)",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=REGISTERED_GENERATION_MAX_NEW_TOKENS,
        help="registered probe generation cap (512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic synthetic-row seed",
    )
    parser.add_argument(
        "--train-step-max-length",
        type=int,
        default=128,
        help=(
            "max_length of the tiny synthetic batch used for the single real "
            "optimizer step (optimizer state size does not depend on it)"
        ),
    )
    parser.add_argument(
        "--headroom-gib",
        type=float,
        default=HEADROOM_GIB,
        help="minimum reserved headroom against the allocator cap (0.25 GiB)",
    )
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help="CUDA allocator cap fraction (registered training default 0.92)",
    )
    parser.add_argument(
        "--vram-free-margin-gib",
        type=float,
        default=0.5,
        help="also cap at free VRAM minus this margin (training default 0.5)",
    )
    parser.add_argument(
        "--heartbeat",
        type=Path,
        default=None,
        help=(
            "heartbeat JSON path (default: gemmaloop.heartbeat default location)"
        ),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="do not force HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE (default offline)",
    )
    return parser


def _parse_ns(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError as exc:
            raise ValueError(f"--ns expects integers, got {part!r}") from exc
        if number < 1:
            raise ValueError(f"--ns values must be >= 1, got {number}")
        values.append(number)
    if not values:
        raise ValueError("--ns must name at least one N")
    if 8 not in values:
        values.append(8)
    return tuple(sorted(set(values)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = _default_out()
    args.out = Path(args.out)
    try:
        args.ns = _parse_ns(args.ns)
    except ValueError as exc:
        parser.error(str(exc))
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.probe_batch_size < 1:
        parser.error("--probe-batch-size must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.train_step_max_length < 1:
        parser.error("--train-step-max-length must be positive")
    if args.generation_prompts < 1:
        parser.error("--generation-prompts must be positive")
    if args.generation_max_new_tokens < 1:
        parser.error("--generation-max-new-tokens must be positive")
    if args.headroom_gib < 0:
        parser.error("--headroom-gib must be non-negative")
    if args.vram_cap < 0:
        parser.error("--vram-cap must be non-negative")
    return args


# ---------------------------------------------------------------------------
# Synthetic, protected-set-free probe rows
# ---------------------------------------------------------------------------


def build_synthetic_messages(
    index: int,
    *,
    seed: int = 0,
    target_chars: int = 65_536,
) -> list[dict[str, Any]]:
    """One deterministic conversation whose thinking trace is longer than the cap.

    The assistant carries a ``reasoning`` field, so the real chat template
    renders ``<|channel>thought ...`` (enable_thinking=True).  With a trace much
    longer than ``--max-length`` the rendered row is truncated inside the open
    thought channel, exactly the shape the audit asks the probe to handle.
    """
    user = (
        f"Synthetic probe row {int(index)} (seed {int(seed)}). "
        "This row is not registered data. Reply with one short sentence."
    )
    repeats = max(1, int(target_chars) // len(_PATTERN) + 1)
    reasoning = (_PATTERN * repeats)[: int(target_chars)]
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "reasoning": reasoning,
            "content": "The answer is four.",
        },
    ]


def _fit_record(
    record: Mapping[str, Any],
    *,
    max_length: int,
    pad_token_id: int,
) -> dict[str, Any]:
    """Truncate or right-pad one tokenized record to exactly ``max_length``."""
    ids = [int(value) for value in record.get("input_ids", [])][: int(max_length)]
    labels = [int(value) for value in record.get("labels", [])][: int(max_length)]
    attention = [
        int(value) for value in record.get("attention_mask", [1] * len(ids))
    ][: int(max_length)]
    while len(ids) < int(max_length):
        ids.append(int(pad_token_id))
        labels.append(-100)
        attention.append(0)
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attention,
    }


def build_synthetic_probe_records(
    tokenizer: Any,
    *,
    rows: int = REGISTERED_ROWS,
    max_length: int = REGISTERED_MAX_LENGTH,
    seed: int = 0,
    include_row_id: bool = False,
) -> list[dict[str, Any]]:
    """Render synthetic rows with the real chat template and force exact length.

    This function touches no filesystem path: it is the guarantee that the
    probe never reads a protected MONITOR/pilot/eval manifest.
    """
    from gemmaloop.train.labels import assistant_labels

    if int(rows) < 1:
        raise ValueError("rows must be positive")
    if int(max_length) < 1:
        raise ValueError("max_length must be positive")
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = 0 if pad is None else int(pad)
    target_chars = max(int(max_length) * 8, 1024)
    records: list[dict[str, Any]] = []
    for index in range(int(rows)):
        messages = build_synthetic_messages(
            index, seed=int(seed), target_chars=target_chars
        )
        rendered = assistant_labels(
            tokenizer, messages, int(max_length), enable_thinking=True
        )
        record = _fit_record(
            rendered, max_length=int(max_length), pad_token_id=pad
        )
        if include_row_id:
            record["row_id"] = int(index)
        records.append(record)
    return records


def build_synthetic_probe_dataset(
    tokenizer: Any,
    *,
    rows: int = REGISTERED_ROWS,
    max_length: int = REGISTERED_MAX_LENGTH,
    seed: int = 0,
    include_row_id: bool = False,
) -> Any:
    from datasets import Dataset

    return Dataset.from_list(
        build_synthetic_probe_records(
            tokenizer,
            rows=rows,
            max_length=max_length,
            seed=seed,
            include_row_id=include_row_id,
        )
    )


def build_probe_batches_for_smoke(
    dataset: Any,
    *,
    model: Any,
    tokenizer: Any,
    rows: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Collate ``rows`` probe rows at ``batch_size`` rows per forward.

    ``--rows`` is the number of probe rows (registered: 4); ``batch_size`` is
    the probe forward micro-batch (training default: 1).  They are independent
    so the smoke measures exactly the registered shape: 4 rows, one row per
    forward.
    """
    return build_probe_batches(
        dataset,
        list(range(int(rows))),
        model=model,
        tokenizer=tokenizer,
        data_collator=None,
        batch_size=int(batch_size),
    )


# ---------------------------------------------------------------------------
# Probe configuration for one N
# ---------------------------------------------------------------------------


def probe_config_for_n(
    n: int,
    *,
    generation: bool = False,
    generation_prompts: int = int(MonitorConfig().generation_prompts),
    generation_max_new_tokens: int = REGISTERED_GENERATION_MAX_NEW_TOKENS,
    generation_batch_size: int = int(MonitorConfig().generation_batch_size),
) -> ProbeConfig:
    """Registered probe config restricted to the exits up to ``n``.

    For ``n=8`` this is exactly the training configuration
    (``eval_ns=(1, 2, 3, 4, 6, 8)``, ``nll_ns=(1, 2, 3, 4)``,
    ``anchor_ns=(2, 3, 4, 6, 8)``).  Lower-N phases are the same probe with the
    higher anchors removed, so each phase reports an isolated per-N peak.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    eval_ns = tuple(sorted({value for value in REGISTERED_EVAL_NS if value <= n} | {n, 1}))
    nll_ns = tuple(value for value in REGISTERED_NLL_NS if value <= n) or (1,)
    anchor_ns = tuple(value for value in REGISTERED_ANCHOR_NS if value <= n)
    return ProbeConfig(
        anchor_ns=anchor_ns,
        eval_ns=eval_ns,
        nll_ns=nll_ns,
        generation_ns=REGISTERED_GENERATION_NS if generation else (),
        generation_prompts=int(generation_prompts),
        generation_max_new_tokens=int(generation_max_new_tokens),
        generation_batch_size=int(generation_batch_size),
        # Keep the registered full-vocabulary chunking; the probe's memory is
        # dominated by these chunks, so they are part of what is measured.
        token_chunk=int(MonitorConfig().probe_token_chunk),
        vocab_chunk=int(MonitorConfig().probe_vocab_chunk),
        nll_temperatures=tuple(MonitorConfig().nll_temperatures),
        ngram=int(MonitorConfig().ngram),
        repetition_rate_threshold=float(MonitorConfig().repetition_rate_threshold),
        repetition_max_count=int(MonitorConfig().repetition_max_count),
    )


# ---------------------------------------------------------------------------
# CUDA measurement helpers (small and monkeypatchable in CPU tests)
# ---------------------------------------------------------------------------


def _synchronize() -> None:
    torch.cuda.synchronize()


def _reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()


def _empty_cache() -> None:
    torch.cuda.empty_cache()


def _memory_totals_gib() -> dict[str, float]:
    return {
        "allocated_gib": float(torch.cuda.memory_allocated()) / 2**30,
        "reserved_gib": float(torch.cuda.memory_reserved()) / 2**30,
    }


def _peak_memory_gib() -> dict[str, float]:
    return {
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2**30,
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved()) / 2**30,
    }


def _phase_status(phase: Mapping[str, Any], headroom_gib: float) -> tuple[str, str | None]:
    if phase.get("oom"):
        return "FAIL", "CUDA OOM"
    probe_errors = [
        str(error) for error in (phase.get("probe_errors") or []) if error
    ]
    if probe_errors:
        return "FAIL", "; ".join(probe_errors)
    invariant = phase.get("invariant")
    if invariant is None:
        invariant = (phase.get("probe") or {}).get("invariant")
    if isinstance(invariant, Mapping):
        if invariant.get("applicable") and invariant.get("ok") is False:
            detail = invariant.get("reason") or (
                f"max_abs_diff={invariant.get('max_abs_diff')}"
            )
            return "FAIL", f"exit-1 invariant failed ({detail})"
    expected_generation = phase.get("expected_generation_rows")
    if expected_generation is not None:
        actual = phase.get("generation_rows")
        if actual is None:
            actual = (phase.get("probe") or {}).get("generation_rows")
        if actual != expected_generation:
            return (
                "FAIL",
                f"generation produced {actual} rows, expected "
                f"{expected_generation}",
            )
    if phase.get("error"):
        return "FAIL", str(phase["error"])
    headroom = phase.get("headroom_gib")
    if headroom is not None and float(headroom) < float(headroom_gib):
        return (
            "FAIL",
            f"headroom {float(headroom):.3f} GiB < {float(headroom_gib):.3f} GiB",
        )
    return "PASS", None


def decide_status(
    phases: Sequence[Mapping[str, Any]], headroom_gib: float = HEADROOM_GIB
) -> dict[str, Any]:
    """Apply the qualification rule to every measured phase.

    A phase passes iff it did not OOM (or hit another error) and its reserved
    headroom against the CUDA allocator cap is at least ``headroom_gib``.
    Missing headroom (a disabled cap) cannot fail on memory alone, but an OOM
    still does.
    """
    statuses: list[dict[str, Any]] = []
    failures: list[str] = []
    for phase in phases:
        status, reason = _phase_status(phase, headroom_gib)
        statuses.append(
            {
                "phase": phase.get("phase"),
                "status": status,
                "reason": reason,
            }
        )
        if reason is not None:
            failures.append(f"{phase.get('phase')}: {reason}")
    return {
        "status": "FAIL" if failures else "PASS",
        "pass": not failures,
        "failures": failures,
        "phase_status": statuses,
    }


# ---------------------------------------------------------------------------
# Training state (same construction as scripts/train.py)
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    model: Any
    text_model: Any
    tokenizer: Any
    processor: Any
    plan: Any
    trainer: Any
    optimizer: Any
    info: dict[str, Any] = field(default_factory=dict)


def _parse_block(value: str) -> tuple[int, int]:
    parts = str(value).split("-")
    if len(parts) != 2:
        raise ValueError(f"block must be START-END, got {value!r}")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError(f"block end must be >= start, got {value!r}")
    return start, end


def _count_tensors(value: Any) -> tuple[int, int]:
    tensors = 0
    elements = 0
    if torch.is_tensor(value):
        return 1, int(value.numel())
    if isinstance(value, Mapping):
        for item in value.values():
            count, size = _count_tensors(item)
            tensors += count
            elements += size
    elif isinstance(value, (list, tuple)):
        for item in value:
            count, size = _count_tensors(item)
            tensors += count
            elements += size
    return tensors, elements


def _optimizer_state_stats(optimizer: Any) -> dict[str, Any]:
    tensors = 0
    elements = 0
    parameters = 0
    if optimizer is None:
        return {"parameters_with_state": 0, "tensors": 0, "elements": 0}
    for group in getattr(optimizer, "param_groups", []):
        for parameter in group.get("params", []):
            state = optimizer.state.get(parameter)
            if not state:
                continue
            parameters += 1
            count, size = _count_tensors(state)
            tensors += count
            elements += size
    return {
        "parameters_with_state": int(parameters),
        "tensors": int(tensors),
        "elements": int(elements),
    }


def materialize_gradient_buffers(model: Any) -> int:
    """Keep one gradient tensor per trainable parameter (worst-case resident)."""
    created = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            created += 1
    return created


def build_training_state(
    args: argparse.Namespace,
    *,
    out_dir: Path,
) -> TrainingState:
    """Build the registered heaviest dial training state exactly as train.py does."""
    import scripts.train as train_script
    from gemmaloop.model.load import ModelSpec
    from gemmaloop.model.looped import LoopConfig
    from gemmaloop.train.dial_objective import DEFAULT_BETA, DEFAULT_LAMBDA_HINGE
    from gemmaloop.train.perf import TrainPerfConfig, apply_train_perf
    from gemmaloop.train.trainer import LoopSFTTrainer, default_lora_config

    block = _parse_block(args.block)
    loop = LoopConfig(
        n_loops=1,
        bridge=str(args.bridge),
        bridge_init=0.99,
        ple_mode="shared",
        block=block,
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
    )
    spec = ModelSpec(size=str(args.size), quant=str(args.quant), adapters=(), loop=loop)
    model, processor, plan = train_script.load_model(spec)
    tokenizer = getattr(processor, "tokenizer", processor)
    text_model = model.model.language_model

    train_script.freeze_base_model(model)

    # A tiny tokenized batch is enough to create the real optimizer and run one
    # real update.  Optimizer state size depends on the trainable parameter
    # count, not on the sequence length.
    train_records = build_synthetic_probe_records(
        tokenizer,
        rows=2,
        max_length=int(args.train_step_max_length),
        seed=int(args.seed),
        include_row_id=False,
    )
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = train_script.build_sft_config(
        out_dir,
        max_length=int(args.train_step_max_length),
        seed=int(args.seed),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=train_script.REGISTERED_LORA_LR,
        warmup_fraction=train_script.REGISTERED_WARMUP_FRACTION,
        gradient_clip=train_script.REGISTERED_GRADIENT_CLIP,
        optimizer=train_script.REGISTERED_OPTIMIZER,
        max_steps=1,
        save_strategy="no",
        logging_steps=1,
    )
    trainer = LoopSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=default_lora_config(
            target_layers="loop_block",
            plan=plan,
        ),
        loop_text_model=text_model,
        loop_enabled=True,
        loop_seed=int(args.seed),
        loop_probs=REGISTERED_LOOP_PROBS,
        bridge_lr=train_script.REGISTERED_BRIDGE_LR,
        loop_ple_lr=train_script.REGISTERED_LOOP_PLE_LR,
        warmup_fraction=train_script.REGISTERED_WARMUP_FRACTION,
        dial_enabled=True,
        dial_window=REGISTERED_DIAL_WINDOW,
        dial_beta=DEFAULT_BETA,
        dial_lambda=DEFAULT_LAMBDA_HINGE,
        dial_horizon_probs=REGISTERED_LOOP_PROBS,
        allow_no_teacher=True,  # smoke only: no teacher shards are read
        heal_teacher=False,
        data_collator=None,
        token_budget=None,
    )
    # The registered memory/speed switches: lean chunked LM head, banded
    # sliding attention, vision offload, no input-grad hook.
    apply_train_perf(trainer, TrainPerfConfig())

    trainer.train()
    optimizer = trainer.optimizer
    if optimizer is None:
        raise RuntimeError("the single optimizer step did not create an optimizer")

    # ``TrainPerfCallback.on_train_end`` restores the frozen vision tower after
    # ``train()`` returns.  A probe during training runs while that tower is
    # offloaded, so re-apply the mid-training state before measuring.
    perf_callback = getattr(trainer, "_gemmaloop_perf_callback", None)
    vision = getattr(perf_callback, "vision", None)
    if vision is not None:
        vision.offload()

    # A real probe happens next to a step boundary with gradients allocated.
    # HF may drop them after the step (set_to_none); materialise the buffers so
    # the measured baseline is the conservative resident case.
    gradients = materialize_gradient_buffers(trainer.model)

    total_params = sum(int(p.numel()) for p in trainer.model.parameters())
    trainable_params = sum(
        int(p.numel()) for p in trainer.model.parameters() if p.requires_grad
    )
    lora_params = sum(
        int(p.numel())
        for name, p in trainer.model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    )
    bridge_params = sum(
        int(p.numel())
        for name, p in trainer.model.named_parameters()
        if p.requires_grad and "loop_bridge" in name
    )
    info = {
        "arm": "L",
        "size": str(args.size),
        "quant": str(args.quant),
        "block": list(block),
        "bridge": str(args.bridge),
        "loop_probs": list(REGISTERED_LOOP_PROBS),
        "dial_window": int(REGISTERED_DIAL_WINDOW),
        "optimizer": str(getattr(training_args, "optim", "")),
        "optimizer_state": _optimizer_state_stats(optimizer),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "lora_parameters": int(lora_params),
        "bridge_parameters": int(bridge_params),
        "gradient_tensors_materialized": int(gradients),
        "train_step": {
            "rows": len(train_dataset),
            "max_length": int(args.train_step_max_length),
            "per_device_batch": 1,
            "gradient_accumulation": 1,
            "max_steps": 1,
            "loss": "dial objective with allow_no_teacher (smoke)",
        },
        "perf": {
            "lean_lm_head": True,
            "banded_sliding": True,
            "offload_vision": True,
            "input_grads": False,
        },
        "ple": "cpu_offloaded",
        "attn_implementation": "sdpa",
    }
    return TrainingState(
        model=trainer.model,
        text_model=trainer.loop_text_model,
        tokenizer=tokenizer,
        processor=processor,
        plan=plan,
        trainer=trainer,
        optimizer=optimizer,
        info=info,
    )


# ---------------------------------------------------------------------------
# Phases and reports
# ---------------------------------------------------------------------------


def measure_baseline(*, cap_gib: float | None, headroom_gib: float) -> dict[str, Any]:
    """Training-state-only peak after the single optimizer step."""
    _synchronize()
    _reset_peak_memory()
    _synchronize()
    phase = {
        "phase": "training_state",
        "n_loops": None,
        "rows": None,
        "max_length": None,
        "wall_s": None,
        "oom": False,
        "error": None,
    }
    phase.update({key: round(value, 4) for key, value in _memory_totals_gib().items()})
    phase.update({key: round(value, 4) for key, value in _peak_memory_gib().items()})
    peak_reserved = float(phase["peak_reserved_gib"])
    phase["headroom_gib"] = (
        None if cap_gib is None else round(float(cap_gib) - peak_reserved, 4)
    )
    status, reason = _phase_status(phase, headroom_gib)
    phase["status"] = status
    phase["reason"] = reason
    return phase


def _probe_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    generation = result.get("generation") or []
    return {
        "invariant": result.get("invariant"),
        "anchor_kl": result.get("anchor_kl"),
        "teacher_kl": result.get("teacher_kl"),
        "calibrated_nll": result.get("calibrated_nll"),
        "probe_errors": list(result.get("probe_errors") or []),
        "generation_rows": len(generation),
        "generation_finish_reasons": sorted(
            {str(row.get("finish_reason")) for row in generation if row}
        ),
    }


def run_probe_phase(
    *,
    name: str,
    n_loops: int,
    make_runner: Callable[[], ProbeRunner],
    rows: int,
    max_length: int,
    cap_gib: float | None,
    headroom_gib: float,
    expected_generation_rows: int | None = None,
) -> dict[str, Any]:
    """Run one probe phase and return its memory/peak record.

    ``make_runner`` builds the probe lazily so the runner's own allocations are
    inside the measured window.  OOM (and any other error) is recorded instead
    of escaping: the report is always written.
    """
    phase: dict[str, Any] = {
        "phase": str(name),
        "n_loops": int(n_loops),
        "rows": int(rows),
        "max_length": int(max_length),
        "wall_s": None,
        "oom": False,
        "error": None,
        "probe_errors": [],
        "generation_rows": None,
        "expected_generation_rows": (
            None
            if expected_generation_rows is None
            else int(expected_generation_rows)
        ),
    }
    generation_timing: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        _synchronize()
        _reset_peak_memory()
        _synchronize()
        runner = make_runner()
        offload_model = getattr(runner, "model", None)
        offload_before = (
            generation_offload_devices(offload_model)
            if expected_generation_rows is not None and offload_model is not None
            else None
        )
        original_generation = getattr(runner, "_generation", None)
        if callable(original_generation):
            def _timed_generation(*args, **kwargs):
                generation_timing["allocated_before_gib"] = _memory_totals_gib()[
                    "allocated_gib"
                ]
                generation_started = time.perf_counter()
                try:
                    return original_generation(*args, **kwargs)
                finally:
                    generation_timing["wall_s"] = round(
                        time.perf_counter() - generation_started, 3
                    )
                    generation_timing["allocated_after_gib"] = _memory_totals_gib()[
                        "allocated_gib"
                    ]

            runner._generation = _timed_generation
        run_started = time.perf_counter()
        result = runner.run(tokens_seen=0, step=1, train_teacher_kl=None)
        run_wall_s = time.perf_counter() - run_started
        _synchronize()
        phase["wall_s"] = round(time.perf_counter() - started, 3)
        phase.update(
            {key: round(value, 4) for key, value in _memory_totals_gib().items()}
        )
        phase.update(
            {key: round(value, 4) for key, value in _peak_memory_gib().items()}
        )
        summary = _probe_summary(result)
        phase["probe"] = summary
        phase["probe_errors"] = list(summary.get("probe_errors") or [])
        phase["generation_rows"] = int(summary.get("generation_rows", 0))
        phase["invariant"] = summary.get("invariant")
        generation_wall_s = generation_timing.get("wall_s")
        phase["generation_wall_s"] = generation_wall_s
        phase["probe_forward_wall_s"] = round(
            max(
                0.0,
                run_wall_s
                - (0.0 if generation_wall_s is None else float(generation_wall_s)),
            ),
            3,
        )
        phase["generation_allocated_before_gib"] = generation_timing.get(
            "allocated_before_gib"
        )
        phase["generation_allocated_after_gib"] = generation_timing.get(
            "allocated_after_gib"
        )
        # ProbeRunner catches generation errors (including CUDA OOM) so a
        # single failed generation cannot kill training.  For this smoke every
        # probe error is a FAIL, and a generation phase must produce exactly
        # len(generation_ns) * generation_prompts rows.
        errors = [str(error) for error in phase["probe_errors"] if error]
        if offload_before is not None:
            offload_after = generation_offload_devices(offload_model)
            if offload_after != offload_before:
                changed = {
                    name: (offload_before.get(name), offload_after.get(name))
                    for name in set(offload_before) | set(offload_after)
                    if offload_before.get(name) != offload_after.get(name)
                }
                errors.append(
                    "offloaded generation modules were not restored to their "
                    f"offload devices: {changed}"
                )
        oom_errors = [
            error
            for error in errors
            if "outofmemory" in error.lower() or "out of memory" in error.lower()
        ]
        if oom_errors:
            phase["oom"] = True
        expected = phase.get("expected_generation_rows")
        if expected is not None and int(phase["generation_rows"]) != int(expected):
            errors.append(
                f"generation produced {phase['generation_rows']} rows, "
                f"expected {expected}"
            )
        if errors:
            phase["error"] = "; ".join(errors)
    except torch.cuda.OutOfMemoryError as exc:
        phase["wall_s"] = round(time.perf_counter() - started, 3)
        phase["oom"] = True
        phase["error"] = f"{type(exc).__name__}: {exc}"
        try:
            _empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass
    except Exception as exc:  # noqa: BLE001 - reported, never lost
        phase["wall_s"] = round(time.perf_counter() - started, 3)
        phase["error"] = f"{type(exc).__name__}: {exc}"
        phase["traceback"] = traceback.format_exc(limit=6)
        try:
            _empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass
    peak_reserved = phase.get("peak_reserved_gib")
    phase["headroom_gib"] = (
        None
        if cap_gib is None or peak_reserved is None
        else round(float(cap_gib) - float(peak_reserved), 4)
    )
    status, reason = _phase_status(phase, headroom_gib)
    phase["status"] = status
    phase["reason"] = reason
    return phase


def _render_markdown(payload: Mapping[str, Any]) -> str:
    config = payload.get("config", {})
    allocator = payload.get("allocator", {})
    lines = [
        f"# Probe memory smoke: {payload.get('status', 'FAIL')}",
        "",
        f"- created: `{payload.get('created_utc')}`",
        f"- out: `{payload.get('out')}`",
        f"- probe rows: {config.get('rows')} x {config.get('max_length')} tokens",
        f"- probe forward micro-batch: {config.get('probe_batch_size')} row(s)",
        f"- N values: {config.get('ns')}",
        f"- allocator cap: {allocator.get('cap_gib')} GiB "
        f"(total {allocator.get('total_gib')} GiB, "
        f"free at start {allocator.get('free_at_start_gib')} GiB)",
        f"- pass rule: no OOM and reserved headroom >= {payload.get('headroom_gib')} GiB",
        "",
        "## Training state",
        "",
        f"- model: {payload.get('training_state', {}).get('size')} "
        f"{payload.get('training_state', {}).get('quant')}, "
        f"arm {payload.get('training_state', {}).get('arm')}, "
        f"block {payload.get('training_state', {}).get('block')}, "
        f"bridge {payload.get('training_state', {}).get('bridge')}",
        f"- optimizer: {payload.get('training_state', {}).get('optimizer')} "
        f"(state: {payload.get('training_state', {}).get('optimizer_state')})",
        f"- trainable parameters: {payload.get('training_state', {}).get('trainable_parameters')}",
        "",
        "## Phases",
        "",
        "| phase | N | rows | seq | peak alloc GiB | peak reserved GiB | headroom GiB | wall s | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for phase in payload.get("phases", []):
        lines.append(
            "| {phase} | {n} | {rows} | {seq} | {alloc} | {res} | {head} | {wall} | {status} |".format(
                phase=phase.get("phase"),
                n=phase.get("n_loops", "-"),
                rows=phase.get("rows", "-"),
                seq=phase.get("max_length", "-"),
                alloc=phase.get("peak_allocated_gib", "-"),
                res=phase.get("peak_reserved_gib", "-"),
                head=phase.get("headroom_gib", "-"),
                wall=phase.get("wall_s", "-"),
                status=phase.get("status", "-"),
            )
        )
    timing = [
        (
            f"- {phase.get('phase')}: probe forward wall "
            f"{phase.get('probe_forward_wall_s')} s (run total "
            f"{phase.get('wall_s')} s), generation wall "
            f"{phase.get('generation_wall_s')} s, allocated before generation "
            f"{phase.get('generation_allocated_before_gib')} GiB, after "
            f"{phase.get('generation_allocated_after_gib')} GiB"
        )
        for phase in payload.get("phases", [])
        if phase.get("generation_wall_s") is not None
    ]
    if timing:
        lines.extend(["", "## Generation timing", ""])
        lines.extend(timing)
    lines.extend(["", "## Probe errors", ""])
    error_lines = [
        f"- {phase.get('phase')}: {error}"
        for phase in payload.get("phases", [])
        for error in (phase.get("probe_errors") or [])
    ]
    lines.extend(error_lines or ["- none"])
    failures = payload.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_reports(out_dir: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    from gemmaloop.train.checkpoint import atomic_write_json, atomic_write_text

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "probe_memory.json"
    md_path = out_dir / "probe_memory.md"
    atomic_write_json(json_path, dict(payload))
    atomic_write_text(md_path, _render_markdown(payload) + "\n")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _base_payload(args: argparse.Namespace) -> dict[str, Any]:
    import datetime

    return {
        "kind": "probe_memory_smoke",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "out": str(args.out),
        "command": " ".join(sys.argv),
        "headroom_gib": float(args.headroom_gib),
        "config": {
            "size": str(args.size),
            "quant": str(args.quant),
            "arm": str(args.arm),
            "block": str(args.block),
            "bridge": str(args.bridge),
            "rows": int(args.rows),
            "probe_batch_size": int(args.probe_batch_size),
            "max_length": int(args.max_length),
            "ns": list(args.ns),
            "generation": not bool(args.skip_generation),
            "generation_prompts": int(args.generation_prompts),
            "generation_max_new_tokens": int(args.generation_max_new_tokens),
            "seed": int(args.seed),
            "train_step_max_length": int(args.train_step_max_length),
            "vram_cap": float(args.vram_cap),
            "vram_free_margin_gib": float(args.vram_free_margin_gib),
        },
        "allocator": {},
        "training_state": {},
        "phases": [],
        "failures": [],
        "notes": [
            "Probe inputs are synthetic; no MONITOR, pilot, T-fit, selection, "
            "confirmatory or evaluation manifest is read.",
            "Each N phase runs the registered probe restricted to exits <= N "
            "(with N=1 always included by ProbeRunner); the N=8 phase is the "
            "full training probe config and also runs the registered "
            "generation path (N=1/2/4, cap 512).",
            "The probe forward runs one row per micro-batch by default, which "
            "is the training default at the registered 8192-token row length; "
            "--rows (probe rows) and --probe-batch-size (rows per forward) are "
            "independent, so 4 rows at micro-batch 1 matches training.",
            "The one real optimizer step uses a tiny 128-token synthetic batch "
            "only to materialise optimizer state; the probe rows themselves are "
            "exactly 8192 tokens.",
            "The frozen vision tower is offloaded for the measurement, as it is "
            "during training.  Teacher top-k fields are absent, so teacher-fit "
            "KL is reported as unavailable; memory is dominated by the 8192-token "
            "forwards, the chunked full-vocabulary KL/NLL and the generation KV "
            "cache.",
        ],
    }


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "CUDA OOM"
    return f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _base_payload(args)

    if not args.allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    heartbeat = Heartbeat(
        args.heartbeat or default_heartbeat_path("probe-memory"),
        run="probe-memory",
        phase="start",
    )
    heartbeat.start(interval=30.0)
    heartbeat.update(items_total=len(args.ns) + 1, phase="start")

    if not torch.cuda.is_available():
        payload["failures"] = ["no CUDA device is visible"]
        payload["status"] = "FAIL"
        payload["notes"].append(
            "CUDA is not available; set CUDA_VISIBLE_DEVICES to the target GPU "
            "and run this smoke on the 12 GiB card."
        )
        write_reports(out_dir, payload)
        heartbeat.stop()
        return 2

    current_phase = "start"
    try:
        from gemmaloop.train.perf import cap_cuda_allocator

        current_phase = "cuda-init"
        torch.cuda.init()
        cap_info = cap_cuda_allocator(
            float(args.vram_cap), float(args.vram_free_margin_gib)
        )
        payload["allocator"] = dict(cap_info or {})
        cap_gib = None if cap_info is None else float(cap_info["cap_gib"])

        current_phase = "build-training-state"
        heartbeat.set_phase("build-training-state", total=len(args.ns) + 1)
        state = build_training_state(args, out_dir=out_dir / "training_step")
        payload["training_state"] = dict(state.info)

        # Free Python-only temporaries from the build, but keep the allocator
        # cache: training does not empty it before a probe either.
        gc.collect()

        current_phase = "baseline"
        heartbeat.set_phase("baseline", total=len(args.ns) + 1)
        baseline = measure_baseline(
            cap_gib=cap_gib, headroom_gib=float(args.headroom_gib)
        )
        payload["phases"].append(baseline)
        heartbeat.update(items_done=1)

        current_phase = "probe-setup"
        # Probe forwards use ``--rows``; generation prompts may need more rows
        # than that (the registered monitor generates from 8 prompts while the
        # probe forward evaluates 4 rows).  Prompts are short prefixes, so the
        # extra rows do not change the measured probe-forward peak.
        prompt_rows = max(int(args.rows), int(args.generation_prompts))
        probe_dataset = build_synthetic_probe_dataset(
            state.tokenizer,
            rows=prompt_rows,
            max_length=int(args.max_length),
            seed=int(args.seed),
            include_row_id=False,
        )
        device = model_compute_device(state.model)
        probe_batches = build_probe_batches_for_smoke(
            probe_dataset,
            model=state.model,
            tokenizer=state.tokenizer,
            rows=int(args.rows),
            batch_size=int(args.probe_batch_size),
        )
        prompts = build_prompt_tensors(
            probe_dataset,
            list(range(prompt_rows)),
            device=device,
            count=int(args.generation_prompts),
        )

        text_model, weight, softcap = text_model_and_head(state.model)

        def _make_runner(config: ProbeConfig) -> ProbeRunner:
            return ProbeRunner(
                model=state.model,
                text_model=state.text_model or text_model,
                weight=weight,
                softcap=softcap,
                batches=probe_batches,
                config=config,
                tokenizer=state.tokenizer,
                prompts=prompts if config.generation_ns else (),
                text_dir=None,
                nll_fit_batches=None,
                nll_score_batches=None,
            )

        done = 1
        for n in args.ns:
            current_phase = f"probe-N{n}"
            generation = bool(n == 8 and not args.skip_generation)
            config = probe_config_for_n(
                int(n),
                generation=generation,
                generation_prompts=int(args.generation_prompts),
                generation_max_new_tokens=int(args.generation_max_new_tokens),
                generation_batch_size=int(MonitorConfig().generation_batch_size),
            )
            expected_generation_rows = (
                len(config.generation_ns) * int(args.generation_prompts)
                if config.generation_ns
                else None
            )
            phase = run_probe_phase(
                name=(
                    f"probe-N{n}"
                    + ("+generation" if config.generation_ns else "")
                ),
                n_loops=int(n),
                make_runner=lambda config=config: _make_runner(config),
                rows=int(args.rows),
                max_length=int(args.max_length),
                cap_gib=cap_gib,
                headroom_gib=float(args.headroom_gib),
                expected_generation_rows=expected_generation_rows,
            )
            payload["phases"].append(phase)
            done += 1
            heartbeat.update(
                items_done=done,
                extra={
                    "phase_status": phase.get("status"),
                    "peak_reserved_gib": phase.get("peak_reserved_gib"),
                },
            )

        decision = decide_status(payload["phases"], float(args.headroom_gib))
        payload["status"] = decision["status"]
        payload["pass"] = bool(decision["pass"])
        payload["failures"] = list(decision["failures"])
        payload["phase_status"] = decision["phase_status"]
        heartbeat.set_phase("complete")
        heartbeat.update(
            items_done=len(payload["phases"]),
            items_total=len(payload["phases"]),
            extra={"status": payload["status"]},
        )
        json_path, md_path = write_reports(out_dir, payload)
        print(f"{payload['status']}: {json_path} / {md_path}")
        return 0 if payload["pass"] else 1
    except BaseException as exc:  # noqa: BLE001 - always write a report
        detail = _classify_error(exc)
        if not any(phase.get("phase") == current_phase for phase in payload["phases"]):
            payload["phases"].append(
                {
                    "phase": current_phase,
                    "n_loops": None,
                    "rows": None,
                    "max_length": None,
                    "wall_s": None,
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                    "error": detail,
                    "status": "FAIL",
                    "reason": detail,
                }
            )
        payload["status"] = "FAIL"
        payload["pass"] = False
        payload["failures"] = [f"{current_phase}: {detail}"]
        payload["notes"].append(
            "The smoke stopped before all phases ran; the failure is recorded "
            "above with the phase that raised it."
        )
        payload["traceback"] = traceback.format_exc(limit=12)
        try:
            json_path, md_path = write_reports(out_dir, payload)
            print(f"FAIL: {json_path} / {md_path}", file=sys.stderr)
        except Exception as write_exc:  # pragma: no cover - last resort
            print(f"FAIL and the report could not be written: {write_exc}", file=sys.stderr)
        return 1
    finally:
        try:
            heartbeat.stop()
        except Exception:  # pragma: no cover - defensive
            pass


if __name__ == "__main__":
    sys.exit(main())
