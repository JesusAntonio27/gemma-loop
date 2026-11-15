"""Train the Phase 1/2 LoRA arms with sampled loop counts.

Usage::

    uv run python scripts/train.py --arm L --size e2b \
        --data artifacts/data/heal --tokens 200000 --seed 0 \
        --out artifacts/runs/l-e2b-seed0

Arms (design section 5.3):

* ``O``       original, no loops, no healing adapters;
* ``O-H``     original + healing LoRA, no loops;
* ``O-H-ext`` same as ``O-H`` but on ``flops_matched_tokens(...)`` tokens;
* ``L``       looped block, N sampled from ``{1,2,3,4}`` during training;
* ``L-plevec`` ``L`` with ``ple_mode="loop_vector"`` (ablation).

The E2B run is bf16 LoRA (PLE on CPU); E4B is NF4 QLoRA.  Vision and audio are
handled by :func:`gemmaloop.model.load.load_model`; ``prepare_model_for_kbit_training``
is never called.
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from datasets import Dataset
from peft import PeftModel

from gemmaloop.gpulock import (
    LOCK_BUSY_EXIT_CODE,
    GpuLockBusy,
    acquire,
    require_gpu_lock,
)
from gemmaloop.train.checkpoint import (
    ABORT_GUARD_FILE,
    DONE_FILE,
    AbortResumeDecision,
    CheckpointCorrupt,
    CheckpointIncomplete,
    ResumeMismatch,
    RetryableIOError,
    archive_newer_checkpoints,
    archive_run_state,
    assert_resume_compatible,
    assert_resume_state_present,
    atomic_write_json,
    checkpoint_invalid_reasons,
    cleanup_stale_tmp_dirs,
    decide_abort_resume,
    find_latest_valid_checkpoint,
    has_run_state,
    list_checkpoint_dirs,
    read_complete,
    read_done,
    sha256_json,
    validate_checkpoint,
    write_done,
)
from gemmaloop.train.frozen import (
    FrozenRunError,
    assert_code_under,
    check_frozen_run,
    git_state,
    prereg_sha256,
    run_config_sha256,
    run_identity_hash,
    write_run_identity,
)
from gemmaloop.train.guards import (
    DEFAULT_SAVE_EVERY_MINUTES,
    GUARD_EXIT_CODE,
    RamGuard,
    ThroughputGuard,
    ThroughputGuardConfig,
    TimeSaveCallback,
    WatchdogCallback,
    guard_thresholds,
)
from gemmaloop.train.heal_monitor import install_heal_monitor
from gemmaloop.train.status import StatusCallback, StatusWriter

from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.layer_plan import make_layer_plan, parse_block
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.data.teacher_topk import (
    DEFAULT_TOP_K,
    TeacherIndex,
    TeacherManifestMismatch,
    TeacherTopKCollator,
    tokenizer_vocab_sha256,
    validate_teacher_alignment,
    validate_teacher_manifest_identity,
    vocab_ex_mask_sha256,
)
from gemmaloop.teacher.runtime_freeze import (
    TeacherRuntimeFreeze,
    TeacherRuntimeFreezeError,
    load_teacher_runtime_freeze,
)
from gemmaloop.model.snapshot import E2B_REVISION, chat_template_sha256, model_revision
from gemmaloop.model.looped import (
    LOOP_CONFIG_FILE,
    LoopConfig,
    assert_adapter_merge_allowed,
    load_loop_state,
)
from gemmaloop.paths import artifacts_dir
from gemmaloop.prereg import load_prereg
from gemmaloop.train.budget import flops_matched_tokens
from gemmaloop.train.dial_objective import (
    DEFAULT_BETA,
    DEFAULT_LAMBDA_HINGE,
    DEFAULT_WINDOW,
    DIAL_LOOP_PROBS,
)
from gemmaloop.train.labels import assistant_labels
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS
from gemmaloop.train.teacher_collator import TEACHER_FIELDS, TeacherDataCollator
from gemmaloop.train.perf import (
    TrainPerfConfig,
    apply_train_perf,
    cap_cuda_allocator,
    prevent_sleep,
)
from gemmaloop.train.trainer import (
    ADAPTERS_CHAIN_FILE,
    LoopSFTTrainer,
    MissingTeacherError,
    SingleEpochGuard,
    SpillGuardCallback,
    TokenBudgetCallback,
    default_lora_config,
    freeze_base_model,
)

LOGGER = logging.getLogger("train")

LOOP_ARMS = {"L", "L-plevec"}
ALL_ARMS = ("O", "O-H", "L", "O-H-ext", "L-plevec")
SIZES = ("e2b", "e4b")
FALLBACK_MAX_LENGTH = 1024

# The registered Stage 1 recipe (R7b-2).  The CLI defaults equal these values
# so an unregistered run and a --prereg run cannot silently diverge.
REGISTERED_PER_DEVICE_BATCH = 1
REGISTERED_ROWS_PER_OPTIMIZER_STEP = 16
REGISTERED_LORA_LR = 2e-4
REGISTERED_BRIDGE_LR = 3e-3
REGISTERED_LOOP_PLE_LR = 1e-3
REGISTERED_GRADIENT_CLIP = 1.0
REGISTERED_WARMUP_FRACTION = 0.03
REGISTERED_OPTIMIZER = "adamw"

# The registered ``AdamW`` runs as bitsandbytes 8-bit AdamW on a CUDA device
# (memory) and falls back to torch AdamW on CPU.  ``adamw`` is the historical
# registered name; ``adamw_8bit`` is the explicit one.  Both resolve to the
# same optimizer, and the resolved name (not the request) is recorded.
ADAMW_8BIT = "adamw_8bit"
ADAMW_TORCH = "adamw_torch"
BITSANDBYTES_OPTIMIZERS = frozenset({ADAMW_8BIT})
ADAMW_ALIASES = frozenset({"adamw", ADAMW_8BIT})


def optimizer_display_name(value: Any) -> str | None:
    """The canonical lower-case name of an optimizer setting or enum."""
    if value is None:
        return None
    name = str(getattr(value, "value", value)).strip().lower()
    return name or None


def resolve_optimizer_name(optimizer: str, *, cuda: bool | None = None) -> str:
    """Return the optimizer name that is actually instantiated.

    ``adamw`` and ``adamw_8bit`` mean bitsandbytes 8-bit AdamW on a CUDA
    device.  On CPU both fall back to ``adamw_torch`` (the 8-bit implementation
    is CUDA-only here); the fallback is explicit so the run record never claims
    an optimizer that did not run.
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    requested = optimizer_display_name(optimizer)
    if requested in ADAMW_ALIASES:
        return ADAMW_8BIT if cuda else ADAMW_TORCH
    return requested or ""


def bitsandbytes_version_for(optimizer: str | None) -> str | None:
    """The bitsandbytes version when ``optimizer`` is a bitsandbytes one."""
    if optimizer_display_name(optimizer) not in BITSANDBYTES_OPTIMIZERS:
        return None
    try:
        import bitsandbytes
    except ImportError:  # pragma: no cover - pinned dependency
        return None
    return str(getattr(bitsandbytes, "__version__", "")).strip() or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-arm",
        default=None,
        help=(
            "registered pilot arm id (from prereg.yaml); the arm, E2B student "
            "size and pilot heal budget come from the prereg, not the CLI"
        ),
    )
    parser.add_argument("--arm", required=False, choices=ALL_ARMS)
    parser.add_argument("--size", required=False, choices=SIZES)
    parser.add_argument("--data", required=True, type=Path, help="Task 12 output directory or JSONL file")
    parser.add_argument(
        "--monitor-rows",
        type=Path,
        default=None,
        help=(
            "heal-format JSONL of MONITOR rows from the dedicated Stage 0 "
            "monitor bucket (required for registered --prereg/--pilot-arm runs)"
        ),
    )
    parser.add_argument(
        "--teacher-data",
        type=Path,
        default=None,
        help=(
            "directory with precomputed teacher top-k shards and "
            "teacher_manifest.json (scripts/precompute_teacher_topk.py)"
        ),
    )
    parser.add_argument(
        "--monitor-teacher-data",
        type=Path,
        default=None,
        help=(
            "teacher top-k shards precomputed for the --monitor-rows corpus "
            "(same contract as --teacher-data); enables teacher-fit KL on the "
            "MONITOR probes"
        ),
    )
    parser.add_argument(
        "--teacher-runtime-freeze",
        type=Path,
        default=None,
        help=(
            "pinned qualified teacher-runtime record; teacher shards must "
            "match its runtime (default: <artifacts>/qualification/"
            "teacher_runtime_freeze.json)"
        ),
    )
    parser.add_argument(
        "--tokens",
        required=False,
        type=int,
        help="data-token budget before FLOPs matching (pilot mode reads pilot.p2.heal_tokens)",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path, help="run directory")
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--lr",
        type=float,
        default=REGISTERED_LORA_LR,
        help="LoRA learning rate (the loop parameters use --bridge-lr/--loop-ple-lr)",
    )
    parser.add_argument(
        "--bridge-lr",
        type=float,
        default=REGISTERED_BRIDGE_LR,
        help="learning rate for the loop bridge logit (weight decay 0)",
    )
    parser.add_argument(
        "--loop-ple-lr",
        type=float,
        default=REGISTERED_LOOP_PLE_LR,
        help="learning rate for loop_ple (weight decay 0)",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        "--per-device-batch-size",
        dest="per_device_train_batch_size",
        type=int,
        default=REGISTERED_PER_DEVICE_BATCH,
        help="per-device batch size in rows (the registered recipe is 1)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--grad-accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=REGISTERED_ROWS_PER_OPTIMIZER_STEP,
        help=(
            "rows accumulated per optimizer step (the registered recipe is "
            "stage1.rows_per_optimizer_step = 16)"
        ),
    )
    parser.add_argument(
        "--gradient-clip",
        "--max-grad-norm",
        dest="gradient_clip",
        type=float,
        default=REGISTERED_GRADIENT_CLIP,
        help="gradient-norm clip (the registered recipe is 1.0)",
    )
    parser.add_argument(
        "--warmup-ratio",
        "--warmup-fraction",
        dest="warmup_ratio",
        type=float,
        default=REGISTERED_WARMUP_FRACTION,
        help="warmup fraction of the token-cosine schedule (registered 3%%)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adamw_8bit", "adamw_torch"),
        default=REGISTERED_OPTIMIZER,
        help=(
            "optimizer; 'adamw' (the registered name) and 'adamw_8bit' run "
            "bitsandbytes 8-bit AdamW on GPU and adamw_torch on CPU"
        ),
    )
    parser.add_argument(
        "--bridge",
        choices=("scalar", "none", "damped", "gch"),
        default="scalar",
        help="loop bridge (gch = zero-init norm-matched per-channel)",
    )
    parser.add_argument(
        "--bridge-init",
        type=float,
        default=0.99,
        help="initial scalar-bridge weight w (ignored by gch/none/damped)",
    )
    parser.add_argument(
        "--block",
        default=None,
        help="inclusive loop block START-END inside the KV-consumer region",
    )
    parser.add_argument(
        "--adapter-policy",
        choices=("always", "loop_only"),
        default="always",
        help="LoRA activation policy",
    )
    parser.add_argument(
        "--lora-target-layers",
        choices=("language", "loop_block", "loop_block_coda"),
        default="language",
        help="LoRA target layer set",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help="pre-registration YAML; its loop_probs drive the sampler and FLOPs match",
    )
    parser.add_argument(
        "--loop-probs",
        default=None,
        help=(
            "comma-separated block-pass distribution (overrides the default "
            "horizon; refused when it disagrees with --prereg)"
        ),
    )
    dial = parser.add_argument_group(
        "loop dial (multi-exit anytime training; off by default)"
    )
    dial.add_argument(
        "--dial-objective",
        action="store_true",
        help=(
            "train every exit in the last --dial-window passes with teacher KL + "
            "0.1*KL(parent||exit) and the monotone hinge, sampling T from the "
            "heavy-tailed dial horizon"
        ),
    )
    dial.add_argument(
        "--dial-window",
        type=int,
        default=DEFAULT_WINDOW,
        help="number of last trainable passes supervised (truncated BPTT window)",
    )
    dial.add_argument(
        "--dial-beta",
        type=float,
        default=DEFAULT_BETA,
        help="weight of KL(parent || exit) at every exit",
    )
    dial.add_argument(
        "--dial-lambda",
        type=float,
        default=DEFAULT_LAMBDA_HINGE,
        help="weight of the stop-gradient monotone hinge",
    )
    dial.add_argument(
        "--allow-no-teacher",
        action="store_true",
        help=(
            "SMOKE TEST ONLY: replace the missing teacher KL with one-hot "
            "next-token CE and record teacher_mode=onehot in run.json; a "
            "registered dial/heal run refuses to start without teacher shards"
        ),
    )
    dial.add_argument(
        "--heal-teacher",
        action="store_true",
        help=(
            "non-dial single-exit heal: use the precomputed teacher forward KL "
            "as the loss (O-H Stage 1); requires --teacher-data or embedded "
            "teacher columns"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint directory, or 'auto' for the newest COMPLETE checkpoint",
    )
    parser.add_argument(
        "--ack-abort",
        default=None,
        metavar="ALERT_ID",
        help=(
            "acknowledge the persisted heal-monitor abort (exit 76); resuming "
            "a run with an abort record without this exits 2, and with it the "
            "run resumes the last COMPLETE checkpoint or restarts from zero"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="archive every old checkpoint/run file under <out>/attic and start over",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="archive a completed run and train it again (normally a relaunch is a no-op)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="test-only: relax frozen-run/prereg tag checks (never for real runs)",
    )
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="resume despite a changed config/data/code/prereg sha (smoke runs only)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a dirty worktree (smoke runs only; long runs must be clean)",
    )
    parser.add_argument(
        "--require-tag",
        default=None,
        help="refuse to run unless HEAD is exactly this tag's commit",
    )
    parser.add_argument(
        "--save-every-minutes",
        type=float,
        default=DEFAULT_SAVE_EVERY_MINUTES,
        help="wall-clock checkpoint interval at optimizer-step boundaries",
    )
    parser.add_argument("--throughput-window-steps", type=int, default=10)
    parser.add_argument("--throughput-baseline-start", type=int, default=20)
    parser.add_argument("--throughput-baseline-end", type=int, default=100)
    parser.add_argument("--throughput-drop-ratio", type=float, default=0.5)
    parser.add_argument("--throughput-failures", type=int, default=3)
    parser.add_argument("--min-ram-gib", type=float, default=2.0)
    parser.add_argument("--guard-platform", default=None)
    parser.add_argument(
        "--oh-ext-budget",
        type=int,
        default=None,
        help=(
            "the O-H-ext FLOPs-matched token budget from the plan; refused when "
            "it differs from train.py's own computation"
        ),
    )
    perf = parser.add_argument_group(
        "performance (memory/speed only; the loss and the recipe are unchanged)"
    )
    perf.add_argument(
        "--no-lean-lm-head",
        dest="lean_lm_head",
        action="store_false",
        help="use TRL's fused LM head (torch fallback) instead of the lean chunked one",
    )
    perf.add_argument("--lm-head-token-chunk", type=int, default=TrainPerfConfig.lm_head_token_chunk)
    perf.add_argument("--lm-head-vocab-chunk", type=int, default=TrainPerfConfig.lm_head_vocab_chunk)
    perf.add_argument(
        "--input-grads",
        action="store_true",
        help="keep TRL's enable_input_require_grads hook (only needed for reentrant checkpointing)",
    )
    perf.add_argument(
        "--no-offload-vision",
        dest="offload_vision",
        action="store_false",
        help="keep the frozen vision tower on the GPU during (text-only) training",
    )
    perf.add_argument(
        "--no-banded-sliding",
        dest="banded_sliding",
        action="store_false",
        help="use the full-mask SDPA path for sliding-window layers",
    )
    perf.add_argument(
        "--checkpoint-skip-token-calls",
        type=int,
        default=TrainPerfConfig.checkpoint_skip_token_calls,
        help="run micro steps with tokens * decoder-layer calls <= this without gradient "
        "checkpointing (0: always checkpoint; not recommended: fragments the allocator in "
        "mixed-length runs)",
    )
    perf.add_argument(
        "--no-host-performance-cores",
        dest="host_performance_cores",
        action="store_false",
        help="do not opt this process out of EcoQoS / onto P-cores (Windows hybrid CPUs)",
    )
    perf.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help="cap the CUDA caching allocator at this fraction of VRAM (and at free VRAM minus "
        "--vram-free-margin-gib) so an over-budget step OOMs instead of paging to RAM; 0 disables",
    )
    perf.add_argument("--vram-free-margin-gib", type=float, default=0.5)
    perf.add_argument(
        "--offload-checkpoints",
        action="store_true",
        help="keep gradient-checkpoint inputs in pinned host memory (longer sequences, slower)",
    )
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help=(
            "repeat the dataset when its trained tokens at --max-length are "
            "below --tokens (the epoch count is recorded in run.json)"
        ),
    )
    args = parser.parse_args(argv)
    if args.pilot_arm is None:
        for name in ("arm", "size", "tokens"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required unless --pilot-arm is given")
    return args


def apply_pilot_arm(args: argparse.Namespace, prereg) -> None:
    """Fill a pilot run's arm and budget from ``prereg.pilot.p2``.

    Only the registered pilot fields are changed.  Every recipe binding still
    goes through :func:`resolve_recipe`; a CLI value that disagrees with the
    registered arm is a terminal refusal (exit 2).
    """

    from gemmaloop.pilot.arms import pilot_arms

    if prereg.pilot is None or prereg.pilot.p2 is None:
        _refuse("--pilot-arm needs prereg.pilot.p2")
    arms = pilot_arms(prereg)
    matches = [arm for arm in arms if arm.arm_id == str(args.pilot_arm)]
    if not matches:
        known = ", ".join(arm.arm_id for arm in arms)
        _refuse(f"unknown --pilot-arm {args.pilot_arm!r}; registered arms: {known}")
    arm = matches[0]

    student_arm = "L" if arm.kind == "L" else "O-H"
    budget = int(prereg.pilot.p2.heal_tokens)
    if args.arm is not None and args.arm != student_arm:
        _refuse(
            f"--pilot-arm {arm.arm_id} is a {student_arm} arm, but --arm={args.arm}"
        )
    if args.size is not None and args.size != "e2b":
        _refuse(f"pilot arms train the E2B student; got --size={args.size}")
    if args.tokens is not None and int(args.tokens) != budget:
        _refuse(
            f"--tokens={args.tokens} overrides the registered pilot budget "
            f"pilot.p2.heal_tokens={budget}"
        )

    registered_max_length = int(prereg.training.max_length)
    registered_bridge_init = float(prereg.architecture.bridge.legacy_scalar_w0)
    if args.max_length is not None and int(args.max_length) != registered_max_length:
        _refuse(
            f"--max-length={args.max_length} overrides the registered "
            f"training.max_length={registered_max_length}"
        )
    if abs(float(args.bridge_init) - registered_bridge_init) > 1e-12:
        _refuse(
            f"--bridge-init={args.bridge_init} overrides the registered "
            f"architecture.bridge.legacy_scalar_w0={registered_bridge_init}"
        )
    if args.init_adapter is not None:
        _refuse("--init-adapter is not part of the registered pilot arm")
    if args.block is not None and str(args.block) != f"{arm.block[0]}-{arm.block[1]}":
        _refuse(
            f"--block={args.block} overrides the registered pilot block "
            f"{arm.block[0]}-{arm.block[1]}"
        )

    args.arm = student_arm
    args.size = "e2b"
    args.tokens = budget
    args.max_length = registered_max_length
    args.bridge = arm.bridge
    args.block = f"{arm.block[0]}-{arm.block[1]}"
    args.adapter_policy = "loop_only"
    args.lora_target_layers = "loop_block"
    args.pilot_arm_id = arm.arm_id
    args.pilot_single_exit = bool(arm.single_exit)
    if arm.kind == "L":
        args.dial_objective = arm.single_exit is None
        args.heal_teacher = arm.single_exit is not None
        if arm.single_exit_horizon is not None:
            args.loop_probs = ",".join(
                f"{value:g}" for value in arm.single_exit_horizon
            )
        else:
            args.loop_probs = None
    else:
        args.dial_objective = False
        args.heal_teacher = True
        args.loop_probs = None


def perf_config_from_args(args: argparse.Namespace) -> TrainPerfConfig:
    return TrainPerfConfig(
        lean_lm_head=bool(args.lean_lm_head),
        lm_head_token_chunk=int(args.lm_head_token_chunk),
        lm_head_vocab_chunk=int(args.lm_head_vocab_chunk),
        input_grads=bool(args.input_grads),
        offload_vision=bool(args.offload_vision),
        offload_checkpoints=bool(args.offload_checkpoints),
        banded_sliding=bool(args.banded_sliding),
        checkpoint_skip_token_calls=int(args.checkpoint_skip_token_calls),
        host_performance_cores=bool(args.host_performance_cores),
    )


def build_sft_config(
    output_dir: str | Path,
    *,
    max_length: int,
    seed: int,
    num_train_epochs: float,
    per_device_train_batch_size: int = REGISTERED_PER_DEVICE_BATCH,
    gradient_accumulation_steps: int = REGISTERED_ROWS_PER_OPTIMIZER_STEP,
    learning_rate: float = REGISTERED_LORA_LR,
    warmup_fraction: float = REGISTERED_WARMUP_FRACTION,
    gradient_clip: float = REGISTERED_GRADIENT_CLIP,
    optimizer: str = REGISTERED_OPTIMIZER,
    **overrides: Any,
):
    """The Task 13 training recipe (identical for every arm).

    ``overrides`` exists for benchmarks and tests (e.g. ``max_steps``); the
    training CLI never passes any.  The keyword defaults are the registered
    Stage 1 values, so a caller that omits them cannot silently use the old
    recipe.
    """
    from trl import SFTConfig

    cuda = torch.cuda.is_available()
    optim = resolve_optimizer_name(optimizer, cuda=cuda)
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        learning_rate=float(learning_rate),
        lr_scheduler_type="cosine",
        # warmup_ratio=0.03 of the recipe: transformers 5.x removed
        # ``warmup_ratio``; a float < 1 in ``warmup_steps`` is that ratio
        # (``ceil(total_steps * 0.03)`` warm-up steps, the same rule).
        warmup_steps=float(warmup_fraction),
        bf16=cuda,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=4,
        save_only_model=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=not cuda,
        max_grad_norm=float(gradient_clip),
        # Teacher columns must survive the Trainer's column pruning; a
        # TeacherDataCollator is installed below when the dataset has them.
        remove_unused_columns=False,
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


@dataclass(frozen=True)
class DataManifest:
    """The training-data manifest fields the trainer needs for the budget check."""

    sha256: str
    target_tokens: int | None
    max_length: int | None
    trained_tokens: int | None
    rows: int | None = None
    chat_template_sha256: str = ""


def read_data_manifest_details(data: str | Path) -> DataManifest:
    """Read the full manifest contract used by the token-budget check.

    ``max_length`` and ``trained_tokens`` are optional: manifests produced
    before the length-aware data build have neither, in which case the trainer
    counts the rendered dataset itself.
    """

    path = Path(data)
    base = path if path.is_dir() else path.parent
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training data manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {manifest_path}: {exc}") from exc
    sha = payload.get("sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"{manifest_path} has no non-empty 'sha256'")
    target = payload.get("target_tokens")
    if target is None:
        target_value: int | None = None
    else:
        try:
            target_value = int(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path} has a non-integer 'target_tokens'"
            ) from exc

    max_length = payload.get("max_length")
    max_length_value: int | None = None
    if max_length is not None:
        try:
            max_length_value = int(max_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path} has a non-integer 'max_length'"
            ) from exc
        if max_length_value <= 0:
            max_length_value = None

    trained = payload.get("trained_tokens_per_source")
    trained_value: int | None = None
    if isinstance(trained, dict):
        try:
            trained_value = sum(int(value) for value in trained.values())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path} has non-integer trained token counts"
            ) from exc

    rows = payload.get("rows_per_source")
    rows_value: int | None = None
    if isinstance(rows, dict):
        try:
            rows_value = sum(int(value) for value in rows.values())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path} has non-integer row counts"
            ) from exc

    template_sha = payload.get("chat_template_sha256")
    template_value = template_sha if isinstance(template_sha, str) else ""

    return DataManifest(
        sha256=sha,
        target_tokens=target_value,
        max_length=max_length_value,
        trained_tokens=trained_value,
        rows=rows_value,
        chat_template_sha256=template_value,
    )


def read_data_manifest(data: str | Path) -> tuple[str, int | None]:
    """Return ``(sha256, target_tokens)`` from the training-data manifest.

    ``--data`` is normally the Task 12 output directory; a direct JSONL path is
    also accepted and looks for ``manifest.json`` next to it.  The manifest is
    required: without its hash the run cannot be audited against the other arms.
    """

    details = read_data_manifest_details(data)
    return details.sha256, details.target_tokens


def select_dataset_tokens(
    manifest: DataManifest, max_length: int, actual_tokens: int
) -> int:
    """Use the manifest's trained-token count when it was built at this length."""

    if manifest.max_length == int(max_length) and manifest.trained_tokens is not None:
        return int(manifest.trained_tokens)
    return int(actual_tokens)


def resolve_num_epochs(
    dataset_tokens: int,
    budget: int,
    *,
    max_length: int,
    allow_repeat: bool,
) -> float:
    """Return the epoch count, refusing silent repetition of a short dataset.

    Without ``--allow-repeat`` the trainer always runs one epoch and the token
    budget callback stops it; a dataset below the budget is an error naming the
    exact shortfall.  With ``--allow-repeat`` the dataset is repeated and the
    epoch count is returned so it can be recorded in ``run.json``.
    """

    if dataset_tokens <= 0:
        raise ValueError("dataset has no trained tokens")
    if dataset_tokens >= budget:
        return 1.0
    if not allow_repeat:
        shortfall = int(budget) - int(dataset_tokens)
        _refuse(
            "training data is too short at max_length="
            f"{max_length}: {dataset_tokens} trained tokens < {budget}-token "
            f"budget (shortfall {shortfall} tokens). Build more data or pass "
            "--allow-repeat to repeat the dataset."
        )
    return max(1.0, budget / dataset_tokens * 1.05)


def mean_trained_tokens_per_row(
    manifest: DataManifest,
    max_length: int,
    dataset_tokens: int,
    dataset_rows: int,
) -> float:
    """Mean trained tokens per row, from the matching manifest when possible."""

    if (
        manifest.max_length == int(max_length)
        and manifest.trained_tokens is not None
        and manifest.rows
    ):
        return manifest.trained_tokens / manifest.rows
    if dataset_rows <= 0:
        raise ValueError("cannot estimate tokens per row from an empty dataset")
    return dataset_tokens / dataset_rows


def resolve_max_steps(
    budget: int,
    tokens_per_row: float,
    *,
    gradient_accumulation_steps: int,
    per_device_train_batch_size: int,
) -> tuple[int, float]:
    """Return ``(max_steps, tokens_per_step)`` for a schedule ending at budget.

    No epoch clamp: the cosine horizon is the number of optimizer steps needed
    to consume ``budget`` tokens at the estimated mean row length.  The token
    budget callback still stops the run exactly when the budget is reached.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    if tokens_per_row <= 0:
        raise ValueError("mean trained tokens per row must be positive")
    if gradient_accumulation_steps <= 0 or per_device_train_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    tokens_per_step = (
        float(tokens_per_row)
        * int(gradient_accumulation_steps)
        * int(per_device_train_batch_size)
    )
    return max(1, math.ceil(int(budget) / tokens_per_step)), tokens_per_step


def resolve_max_steps_bound(
    budget: int,
    min_tokens_per_row: float,
    *,
    gradient_accumulation_steps: int,
    per_device_train_batch_size: int,
) -> int:
    """A safe upper bound on optimizer steps before the budget callback stops.

    Every optimizer step consumes at least ``min_tokens_per_row`` per
    micro-batch, so this bound cannot stop the run before the callback reaches
    the budget.  The token-based scheduler makes the exact horizon irrelevant.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    if min_tokens_per_row <= 0:
        raise ValueError("minimum trained tokens per row must be positive")
    tokens_per_step_min = (
        float(min_tokens_per_row)
        * int(gradient_accumulation_steps)
        * int(per_device_train_batch_size)
    )
    return max(1, math.ceil(int(budget) / tokens_per_step_min))


def assert_no_silent_repeat(trainer, dataset_size: int, *, allow_repeat: bool) -> None:
    """Refuse a run that consumed more micro-batches than the dataset has rows.

    A manifest whose ``trained_tokens`` overstates the rendered rows can pass
    the budget pre-check and then silently start a reshuffled second epoch.
    """

    if allow_repeat:
        return
    consumed = int(getattr(trainer, "micro_batches_seen", 0))
    if consumed > int(dataset_size):
        raise RuntimeError(
            "trainer consumed "
            f"{consumed} micro-batches from a {dataset_size}-row dataset "
            "without --allow-repeat; the manifest probably overstates the "
            "trained tokens. Rebuild the data or pass --allow-repeat."
        )


def check_oh_ext_budget(arm: str, explicit: int | None, computed: int) -> None:
    """Refuse a plan/train FLOPs-budget mismatch (C1 of the native review)."""

    if explicit is None:
        return
    if arm != "O-H-ext":
        _refuse(
            f"--oh-ext-budget is only valid for the O-H-ext arm, got {arm!r}"
        )
    if int(explicit) != int(computed):
        _refuse(
            "O-H-ext budget mismatch: plan/train.py says "
            f"{int(explicit)} tokens, this model/config computes {int(computed)}. "
            "The data build and the training run would disagree."
        )


def verify_chat_template(manifest_sha256: str, tokenizer) -> str:  # noqa: ANN001
    """Return the loaded template's hash, refusing a mismatch with the manifest.

    A legacy manifest without the field cannot be checked; it is accepted so
    pre-pin data stays trainable, but new builds always record the hash.
    """

    current = chat_template_sha256(tokenizer)
    expected = str(manifest_sha256 or "")
    if expected and current != expected:
        _refuse(
            "chat template changed since the data build: manifest sha256 "
            f"{expected}, loaded tokenizer sha256 {current or '<none>'}. "
            f"Cache the pinned snapshot ({E2B_REVISION}) or rebuild the data."
        )
    return current


def build_teacher_metadata(
    teacher_index, teacher_data: str | Path | None
) -> dict[str, Any] | None:
    """The ``run.json`` teacher section, in the layout the harness reads.

    The runtime lives at ``run["teacher"]["teacher"]["runtime"]`` (the nested
    ``TeacherManifest.teacher`` record).  ``run_config`` deliberately does not
    carry a runtime key: it is not part of the frozen resume identity.
    """

    if teacher_index is None:
        return None
    manifest = teacher_index.manifest
    return {
        "path": str(teacher_data),
        "manifest_sha256": teacher_index.manifest_sha256,
        "teacher": dict(manifest.teacher),
        "data_manifest_sha256": manifest.data_manifest_sha256,
        "top_k": int(manifest.top_k),
        "rows": int(manifest.rows_total),
        "tokens": int(manifest.tokens_total),
        "mask_token": manifest.mask_token,
    }


def build_run_metadata(
    *,
    arm: str,
    size: str,
    quant: str,
    data: Path,
    base_tokens: int,
    target_tokens: int,
    max_length: int,
    seed: int,
    loop: LoopConfig | None,
    adapters_chain: Sequence[str],
    trainer,
    manifest_sha256: str,
    manifest_target_tokens: int | None,
    chat_template_sha256: str | None = None,
    manifest_chat_template_sha256: str | None = None,
    length_mode: str | None = None,
    model_revision_sha: str | None = None,
    num_epochs: float = 1.0,
    dataset_trained_tokens: int | None = None,
    allow_repeat: bool = False,
    max_steps: int | None = None,
    estimated_steps: int | None = None,
    tokens_per_step: float | None = None,
    mean_tokens_per_row: float | None = None,
    loop_block: tuple[int, int] | None = None,
    lora_target_layers: str | None = None,
    lora_layers: Sequence[int] | None = None,
    optimizer: str | None = None,
    optimizer_requested: str | None = None,
) -> dict[str, Any]:
    """Build the ``run.json`` payload written after training.

    ``target_tokens`` is the arm's actual budget (FLOPs-matched for
    ``O-H-ext``), while ``base_tokens`` is the pre-registered budget shared by
    O-H and L.  ``tokens_consumed`` and ``global_step`` are read from the
    finished trainer so the orchestrator can audit the run.

    ``optimizer`` defaults to the optimizer the trainer actually built
    (``trainer.args.optim``), so a record cannot claim a recipe that did not
    run.  ``bitsandbytes_version`` is recorded whenever that optimizer is a
    bitsandbytes one.
    """

    resolved_optimizer = optimizer_display_name(optimizer) or optimizer_display_name(
        getattr(getattr(trainer, "args", None), "optim", None)
    )
    requested_optimizer = optimizer_display_name(optimizer_requested)

    payload = {
        "status": "complete",
        "arm": arm,
        "size": size,
        "quant": quant,
        "data": str(data),
        "base_tokens": int(base_tokens),
        "budget_tokens": int(target_tokens),
        "target_tokens": int(target_tokens),
        "max_length": int(max_length),
        "length_mode": length_mode,
        "seed": int(seed),
        "ple_mode": loop.ple_mode if loop is not None else None,
        "bridge": loop.bridge if loop is not None else None,
        "bridge_init": loop.bridge_init if loop is not None else None,
        "loop_block": (
            list(loop.block)
            if loop is not None and loop.block is not None
            else (list(loop_block) if loop_block is not None else None)
        ),
        "adapter_policy": (
            loop.adapter_policy if loop is not None else None
        ),
        "lora_target_layers": (
            loop.lora_target_layers
            if loop is not None and loop.lora_target_layers is not None
            else lora_target_layers
        ),
        "lora_layers": (
            [int(layer) for layer in lora_layers]
            if lora_layers is not None
            else None
        ),
        "n_loops_sampled": bool(loop is not None),
        "adapters_chain": list(adapters_chain),
        "global_step": int(trainer.state.global_step),
        "tokens_consumed": int(trainer.tokens_seen),
        "micro_batches_seen": int(getattr(trainer, "micro_batches_seen", 0)),
        "num_epochs": float(num_epochs),
        "dataset_trained_tokens": (
            None if dataset_trained_tokens is None else int(dataset_trained_tokens)
        ),
        "allow_repeat": bool(allow_repeat),
        "max_steps": None if max_steps is None else int(max_steps),
        "estimated_steps": (
            None if estimated_steps is None else int(estimated_steps)
        ),
        "tokens_per_step": (
            None if tokens_per_step is None else float(tokens_per_step)
        ),
        "mean_tokens_per_row": (
            None if mean_tokens_per_row is None else float(mean_tokens_per_row)
        ),
        "data_manifest_sha256": manifest_sha256,
        "data_manifest_target_tokens": manifest_target_tokens,
        "data_manifest_chat_template_sha256": manifest_chat_template_sha256,
        "chat_template_sha256": chat_template_sha256,
        "model_revision": model_revision_sha,
        "learning_rate": float(getattr(trainer.args, "learning_rate", 0.0)),
        "optimizer": resolved_optimizer,
        "optimizer_requested": requested_optimizer,
        "bitsandbytes_version": bitsandbytes_version_for(resolved_optimizer),
        "bridge_lr": (
            float(getattr(trainer, "bridge_lr", 0.0)) if loop is not None else None
        ),
        "loop_ple_lr": (
            float(getattr(trainer, "loop_ple_lr", 0.0)) if loop is not None else None
        ),
        "optimizer_steps": int(getattr(trainer, "_optimizer_steps", 0)),
        "supervised_tokens_per_step": getattr(
            trainer, "supervised_tokens_per_step", None
        ),
        "clipped_steps": int(getattr(trainer, "_clipped_steps", 0)),
        "clip_rate": float(getattr(trainer, "clip_rate", 0.0)),
        "final_grad_norm": getattr(trainer, "_last_grad_norm_total", None),
        "final_bridge_w": None,
        "final_bridge_g_norm": None,
        "final_loop_ple_norm": None,
        "training_metrics": list(getattr(trainer, "_metric_history", [])),
    }
    loop_metrics = getattr(trainer, "loop_metrics", None)
    if callable(loop_metrics):
        payload.update(loop_metrics())
        if "bridge_w" in payload:
            payload["final_bridge_w"] = payload.pop("bridge_w")
        if "bridge_g_norm" in payload:
            payload["final_bridge_g_norm"] = payload.pop("bridge_g_norm")
        if "loop_ple_norm" in payload:
            payload["final_loop_ple_norm"] = payload.pop("loop_ple_norm")
    return payload


def write_run_json(directory: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Atomically write ``metadata`` to ``<directory>/run.json``.

    The temporary file is created in the same directory and renamed with
    :func:`os.replace`, so a crash can leave the old file or no file but never a
    truncated ``run.json`` that the orchestrator would treat as complete.
    """
    path = Path(directory) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    # An orchestrator ``--ack-abort`` recorded before the relaunch must survive
    # the final run.json rewrite (the ack is part of the run's audit trail).
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("abort_acks"):
            payload.setdefault("abort_acks", existing["abort_acks"])
    atomic_write_json(path, payload)
    return path


def is_complete_payload(payload: Mapping[str, Any] | None) -> bool:
    return bool(payload) and payload.get("status") == "complete"


def check_already_complete(
    out_dir: str | Path,
    identity_hash: str,
    *,
    allow_rerun: bool = False,
) -> bool:
    """True when DONE/run.json says this run already finished.

    A relaunch must return 0 before the GPU lock without touching the adapter
    or ``run.json``.  A completed run with a different identity is exit 2.
    """
    if allow_rerun:
        return False
    done = read_done(out_dir)
    run_json_path = Path(out_dir) / "run.json"
    run_payload: dict[str, Any] | None = None
    if run_json_path.is_file():
        try:
            loaded = json.loads(run_json_path.read_text(encoding="utf-8"))
            run_payload = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            run_payload = None

    payload = done if is_complete_payload(done) else (
        run_payload if is_complete_payload(run_payload) else None
    )
    if payload is None:
        return False
    saved = payload.get("run_identity_hash")
    if saved is not None and saved != identity_hash:
        print(
            f"error: {out_dir} was completed by a different run identity "
            f"({saved}); use --fresh/--rerun-complete to start over",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"already complete: {out_dir} (identity {identity_hash[:12]})")
    if not is_complete_payload(done):
        write_done(
            out_dir,
            {
                "status": "complete",
                "run_identity_hash": identity_hash,
                "run_config_sha256": payload.get("run_config_sha256"),
                "data_manifest_sha256": payload.get("data_manifest_sha256"),
                "git_sha": payload.get("git_sha"),
                "seed": payload.get("seed"),
                "completed_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "recovered_from": "run.json",
            },
        )
    return True


def cuda_failure_kind(exc: BaseException) -> str | None:
    """Classify a CUDA/OOM failure that should be retried with exit 75."""
    if isinstance(exc, torch.OutOfMemoryError):
        return "out-of-memory at the allocator cap"
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return "CUDA accelerator error"
    message = str(exc).lower()
    for token in ("cuda error", "cublas_status", "cudnn", "unspecified launch failure"):
        if token in message:
            return "CUDA error"
    return None


def handle_retryable_training_exception(exc: BaseException, trainer, status) -> int | None:  # noqa: ANN001
    """Map checkpoint I/O and CUDA/OOM failures to exit 75, else return None."""
    if isinstance(exc, RetryableIOError):
        status.set_exit(f"retryable IO error: {exc}", GUARD_EXIT_CODE)
        LOGGER.error("retryable checkpoint I/O failure (exit 75): %s", exc)
        return GUARD_EXIT_CODE
    kind = cuda_failure_kind(exc)
    if kind is None:
        return None
    # Never checkpoint after an exception: counters/RNG may be mid-step and
    # the parameters partially updated.  Resume from the last COMPLETE
    # checkpoint (the documented <= 30 min loss).
    LOGGER.error(
        "retryable GPU failure (%s): %s; resuming from the last COMPLETE checkpoint",
        kind,
        exc,
    )
    status.set_exit(f"gpu-error: {kind}: {exc}", GUARD_EXIT_CODE)
    return GUARD_EXIT_CODE


def guard_is_at_bound(trainer, max_steps: int) -> bool:
    """True when a guard fires on the last budgeted optimizer step.

    Returning 75 there would make the resumed run take one extra step beyond
    ``max_steps`` or the token budget.
    """
    token_budget = getattr(trainer, "token_budget", None)
    if token_budget is not None and int(getattr(trainer, "tokens_seen", 0)) >= int(
        token_budget
    ):
        return True
    return int(max_steps) > 0 and int(getattr(trainer.state, "global_step", 0)) >= int(
        max_steps
    )


def teacher_data_sha256(path: str | Path | None) -> str | None:
    """SHA-256 of the teacher manifest file, without validating its shards."""

    if path is None:
        return None
    target = Path(path)
    manifest = target if target.is_file() else target / "teacher_manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def build_run_config(
    args: argparse.Namespace,
    *,
    quant: str,
    max_length: int,
    objective: ObjectiveConfig | None = None,
    teacher_manifest_sha256: str | None = None,
    recipe: RecipeConfig | None = None,
) -> dict[str, Any]:
    """The fields that must not change between a checkpoint and its resume."""
    dial = bool(objective.dial) if objective is not None else bool(args.dial_objective)
    heal = bool(objective.heal) if objective is not None else bool(args.heal_teacher)
    window = int(objective.window) if objective is not None else int(args.dial_window)
    beta = float(objective.beta) if objective is not None else float(args.dial_beta)
    lambda_hinge = (
        float(objective.lambda_hinge)
        if objective is not None
        else float(args.dial_lambda)
    )
    horizon = (
        list(objective.horizon)
        if objective is not None
        else list(
            _loop_probs_arg(args.loop_probs)
            or (DIAL_LOOP_PROBS if args.dial_objective else DEFAULT_LOOP_PROBS)
        )
    )
    requested_optimizer = optimizer_display_name(
        recipe.optimizer if recipe is not None else args.optimizer
    )
    resolved_optimizer = (
        optimizer_display_name(recipe.resolved_optimizer)
        if recipe is not None and recipe.resolved_optimizer
        else resolve_optimizer_name(
            recipe.optimizer if recipe is not None else args.optimizer
        )
    )
    return {
        "arm": args.arm,
        "size": args.size,
        "quant": quant,
        "data": str(Path(args.data).resolve()),
        "monitor_rows": (
            None
            if getattr(args, "monitor_rows", None) is None
            else str(Path(args.monitor_rows).resolve())
        ),
        "monitor_teacher_data": (
            None
            if getattr(args, "monitor_teacher_data", None) is None
            else str(Path(args.monitor_teacher_data).resolve())
        ),
        "tokens": int(args.tokens),
        "seed": int(args.seed),
        "max_length": int(max_length),
        "learning_rate": (
            float(recipe.learning_rate) if recipe is not None else float(args.lr)
        ),
        "bridge_lr": (
            float(recipe.bridge_lr) if recipe is not None else float(args.bridge_lr)
        ),
        "loop_ple_lr": (
            float(recipe.loop_ple_lr)
            if recipe is not None
            else float(args.loop_ple_lr)
        ),
        "per_device_train_batch_size": (
            int(recipe.per_device_train_batch_size)
            if recipe is not None
            else int(args.per_device_train_batch_size)
        ),
        "gradient_accumulation_steps": (
            int(recipe.gradient_accumulation_steps)
            if recipe is not None
            else int(args.gradient_accumulation_steps)
        ),
        "gradient_clip": (
            float(recipe.gradient_clip)
            if recipe is not None
            else float(args.gradient_clip)
        ),
        "warmup_fraction": (
            float(recipe.warmup_fraction)
            if recipe is not None
            else float(args.warmup_ratio)
        ),
        # The resolved name is what actually runs (bitsandbytes adamw_8bit on
        # GPU, adamw_torch on CPU); the request is kept for provenance.
        "optimizer": resolved_optimizer,
        "optimizer_requested": requested_optimizer,
        "bitsandbytes_version": bitsandbytes_version_for(resolved_optimizer),
        "bridge": args.bridge,
        "bridge_init": float(args.bridge_init),
        "block": args.block,
        "adapter_policy": args.adapter_policy,
        "lora_target_layers": args.lora_target_layers,
        "allow_repeat": bool(args.allow_repeat),
        # Cross-platform resume must not silently change numerics: a CPU CPU
        # run and a CUDA bf16 run are different configurations.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "bf16": bool(torch.cuda.is_available()),
        "init_adapter": (
            None if args.init_adapter is None else str(Path(args.init_adapter).resolve())
        ),
        "prereg": None if args.prereg is None else str(Path(args.prereg).resolve()),
        "dial_objective": dial,
        "dial_window": window,
        "dial_beta": beta,
        "dial_lambda": lambda_hinge,
        "dial_horizon_probs": horizon,
        "heal_teacher": heal,
        # M7: the monitor on/off state is part of the run identity.
        "heal_monitor": _monitor_enabled(args, dial=dial, heal=heal),
        "pilot_arm": getattr(args, "pilot_arm_id", None),
        "allow_no_teacher": bool(
            objective.allow_no_teacher
            if objective is not None
            else args.allow_no_teacher
        ),
        "teacher_data": (
            None
            if args.teacher_data is None
            else str(Path(args.teacher_data).resolve())
        ),
        "teacher_manifest_sha256": teacher_manifest_sha256,
    }


def _read_run_identity(out_dir: str | Path) -> dict[str, Any] | None:
    path = Path(out_dir) / "run_identity.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_resume_checkpoint(
    requested: str | None,
    out_dir: str | Path,
    *,
    run_config_sha256: str,
    data_manifest_sha256: str,
    git_sha: str,
    prereg_sha: str | None,
    identity_hash: str,
    allow_mismatch: bool = False,
) -> Path | None:
    """Resolve ``--resume auto|<dir>`` and enforce the run identity.

    ``auto`` only considers checkpoints of the *same* run identity (config +
    data manifest + code sha + seed).  An empty run directory is a normal first
    start; any existing but unusable checkpoint state is exit 2 unless the
    caller already archived it with ``--fresh``.
    """
    if requested is None:
        return None
    if requested == "auto":
        info = find_latest_valid_checkpoint(out_dir, identity_hash=identity_hash)
        if info is None and allow_mismatch:
            # N6: the escape hatch must work with auto as well (it then warns
            # about every changed identity field below).
            info = find_latest_valid_checkpoint(out_dir, identity_hash=None)
        if info is None:
            checkpoint_dirs = list_checkpoint_dirs(out_dir)
            identity_file = _read_run_identity(out_dir)
            if not checkpoint_dirs and identity_file is not None:
                saved_identity = identity_file.get("run_identity_hash")
                if saved_identity == identity_hash:
                    LOGGER.warning(
                        "--resume auto: no checkpoint yet for the same run "
                        "identity in %s; starting fresh",
                        out_dir,
                    )
                    return None
                if not allow_mismatch:
                    print(
                        f"error: {out_dir} belongs to a different run identity "
                        f"({saved_identity}); use --fresh to archive it",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                LOGGER.warning(
                    "--resume auto: identity mismatch ignored by "
                    "--allow-resume-mismatch; starting fresh"
                )
                return None
            if not checkpoint_dirs and not has_run_state(out_dir):
                LOGGER.warning(
                    "--resume auto: empty run directory %s; starting fresh", out_dir
                )
                return None
            reasons = checkpoint_invalid_reasons(out_dir)
            detail = "; ".join(reasons) if reasons else "no matching identity"
            print(
                f"error: --resume auto found no usable checkpoint for this "
                f"run in {out_dir}: {detail}. Use --fresh to archive the old "
                "run and start over.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        path = info.path
        metadata = info.metadata
    else:
        path = Path(requested)
        try:
            metadata = validate_checkpoint(path)
        except (CheckpointIncomplete, CheckpointCorrupt) as exc:
            print(f"error: refusing to resume from {path}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    try:
        assert_resume_state_present(metadata)
    except CheckpointCorrupt as exc:
        print(f"error: refusing to resume from {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        mismatches = assert_resume_compatible(
            metadata,
            run_config_sha256=run_config_sha256,
            data_manifest_sha256=data_manifest_sha256,
            git_sha=git_sha,
            prereg_sha=prereg_sha,
            run_identity_hash=identity_hash,
            allow_mismatch=allow_mismatch,
        )
    except ResumeMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if mismatches:
        LOGGER.warning(
            "resuming despite run identity mismatch (--allow-resume-mismatch): %s",
            "; ".join(mismatches),
        )
    LOGGER.info("resuming from %s", path)
    return path


def resolve_run_length(
    args, prereg, manifest: DataManifest, size: str, arm: str
) -> tuple[int, str | None]:
    """Resolve the run length from CLI/prereg and guard the data manifest.

    With ``--prereg`` an omitted ``--max-length`` takes the registered value
    instead of falling back to calibration.  A manifest that records a
    different length cannot be used: the token accounting and the labels were
    rendered at that length.
    """

    requested = args.max_length
    length_mode: str | None = None
    if prereg is not None:
        length_mode = str(prereg.training.length_mode)
        if requested is None:
            requested = int(prereg.training.max_length)
    max_length = resolve_max_length(size, requested, arm)
    if manifest.max_length is not None and int(manifest.max_length) != int(max_length):
        _refuse(
            f"training data was built at max_length={int(manifest.max_length)} but "
            f"this run uses {int(max_length)}; rebuild the data or pass the "
            "matching --max-length"
        )
    return max_length, length_mode


def _calibration_max_length(size: str, arm: str | None = None) -> int | None:
    path = artifacts_dir() / "calibration" / "calibration.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("cannot read %s: %s", path, exc)
        return None

    value: Any = payload.get("max_nonspill_train_seq_len")
    if isinstance(value, dict):
        value = value.get(size)
    elif value is not None and size != "e2b":
        # Task 6 probes E2B only; a scalar is not valid evidence for E4B.
        value = None
    if value is None:
        by_size = payload.get("max_nonspill_train_seq_len_by_size")
        if isinstance(by_size, dict):
            value = by_size.get(size)
    if value is None and arm is not None:
        by_arm = payload.get("max_nonspill_train_seq_len_by_arm")
        if isinstance(by_arm, dict):
            value = by_arm.get(arm)
    if value is None and isinstance(payload.get(size), dict):
        value = payload[size].get("max_nonspill_train_seq_len")
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_max_length(
    size: str, requested: int | None, arm: str | None = None
) -> int:
    """CLI rule: explicit value, else calibration, else 1024 with a warning."""
    if requested is not None:
        if requested <= 0:
            raise ValueError(f"--max-length must be positive, got {requested}")
        return int(requested)
    calibrated = _calibration_max_length(size, arm)
    if calibrated is not None:
        LOGGER.info("max_length=%d from calibration.json", calibrated)
        return calibrated
    LOGGER.warning(
        "no max_nonspill_train_seq_len for %s in %s; falling back to %d",
        size,
        artifacts_dir() / "calibration" / "calibration.json",
        FALLBACK_MAX_LENGTH,
    )
    return FALLBACK_MAX_LENGTH


def _load_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"training data not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("messages"):
                raise ValueError(f"{path}:{line_number}: expected a row with 'messages'")
            rows.append(row)
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def build_dataset(
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    *,
    include_row_id: bool = False,
) -> Dataset:
    """Tokenize messages with assistant-only labels (no TRL assistant_only_loss).

    If a row carries teacher fields (``teacher_topk_probs`` ``[S,K]``,
    ``teacher_topk_indices`` ``[S,K]``, ``teacher_tail_prob`` ``[S]``), they are
    truncated with the input ids and kept on the dataset; the teacher is
    unshifted and aligned 1:1 with ``input_ids``.  The T-heal shard collator may
    instead synthesise them from shards, in which case these columns are absent
    and the collator supplies them.

    ``include_row_id`` keeps the original ``train.jsonl`` line index so a
    teacher shard index can join on it lazily in the collator.
    """
    records: list[dict[str, Any]] = []
    for row_id, row in enumerate(rows):
        output = assistant_labels(
            tokenizer, row["messages"], max_length, tools=row.get("tools")
        )
        if not output["input_ids"] or all(label == -100 for label in output["labels"]):
            continue
        record: dict[str, Any] = dict(output)
        if include_row_id:
            record["row_id"] = row_id
        present = [field in row and row[field] is not None for field in TEACHER_FIELDS]
        if any(present):
            if not all(present):
                raise ValueError(
                    "a row with teacher fields must carry all of "
                    + ", ".join(TEACHER_FIELDS)
                )
            length = len(output["input_ids"])
            for field in TEACHER_FIELDS:
                value = row[field]
                if len(value) < length:
                    raise ValueError(
                        f"{field} has {len(value)} positions but the truncated "
                        f"input has {length}"
                    )
                record[field] = value[:length]
            for field in ("teacher_manifest_sha256", "teacher_id", "teacher_revision"):
                if row.get(field) is not None:
                    record[field] = row[field]
        records.append(record)
    if not records:
        raise ValueError("every row was empty or had no assistant tokens after truncation")
    return Dataset.from_list(records)


def resolve_teacher_index(
    manifest_sha256: str,
    teacher_data: Path | None,
    *,
    allow_smoke: bool = False,
    render_template_sha256: str | None = None,
    student_vocab_sha256: str | None = None,
    student_vocab_ex_mask_sha256: str | None = None,
    teacher_vocab_sha256: str | None = None,
    teacher_runtime_freeze: TeacherRuntimeFreeze | None = None,
) -> TeacherIndex | None:
    """Load and validate precomputed teacher shards, or refuse the run.

    The refusal decision (teacher required but absent) is made once, in
    :func:`main`, after the dataset is rendered so embedded teacher columns and
    ``--teacher-data`` shards share one path.  A supplied shard directory whose
    manifest/data SHA does not match is always a hard exit 2.
    """

    if teacher_data is None:
        return None
    try:
        index = TeacherIndex.from_directory(
            teacher_data,
            data_manifest_sha256=manifest_sha256,
            expect_top_k=DEFAULT_TOP_K,
            verify_shards=True,
        )
        frozen_runtime = (
            None
            if teacher_runtime_freeze is None
            else teacher_runtime_freeze.runtime_for_teacher_key(
                index.manifest.teacher_key
            )
        )
        validate_teacher_manifest_identity(
            index.manifest,
            allow_smoke=allow_smoke,
            render_template_sha256=render_template_sha256,
            student_vocab_sha256=student_vocab_sha256,
            student_vocab_ex_mask_sha256=student_vocab_ex_mask_sha256,
            teacher_vocab_sha256=teacher_vocab_sha256,
            frozen_runtime=frozen_runtime,
        )
        return index
    except (TeacherManifestMismatch, TeacherRuntimeFreezeError) as exc:
        _refuse(f"teacher data unusable: {exc}")
    raise AssertionError("unreachable")


def teacher_collator(index: TeacherIndex, tokenizer):
    """TRL's language-modeling collator wrapped with the teacher fields."""

    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    if pad_token_id is None:
        raise ValueError("the tokenizer has neither a pad nor an eos token id")
    base = DataCollatorForLanguageModeling(pad_token_id=int(pad_token_id))
    return TeacherTopKCollator(index, base)


def ensure_teacher_available(
    *,
    needs_teacher: bool,
    teacher_available: bool,
    allow_no_teacher: bool,
) -> None:
    """The single refusal path for a registered teacher-distillation run."""

    if needs_teacher and not teacher_available and not allow_no_teacher:
        _refuse(
            "the teacher-distillation objective needs teacher_topk_probs / "
            "teacher_topk_indices / teacher_tail_prob (--teacher-data shards or "
            "embedded columns); refusing to start. Pass --allow-no-teacher only "
            "for an explicit one-hot smoke run."
        )


def _read_adapters_chain(directory: Path) -> list[str]:
    path = Path(directory) / ADAPTERS_CHAIN_FILE
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("adapters", [])
    return [str(item) for item in payload]


def _merge_init_adapter(model, text_model, init_dir: Path, size: str, quant: str):
    """Merge the full previous adapter chain and return ``(model, chain)``.

    The adapters are merged in the same order that ``load_model`` uses for
    ``ModelSpec.adapters``: every adapter recorded by the init adapter, then the
    init adapter itself.  The returned chain is written beside the new adapter
    so evaluation can reproduce exactly the weights that were trained.
    """
    if size != "e2b" or quant != "bf16":
        raise ValueError("--init-adapter is only supported for bf16 E2B runs")
    chain = _read_adapters_chain(Path(init_dir))
    chain.append(str(Path(init_dir)))
    for adapter_dir in chain:
        assert_adapter_merge_allowed(adapter_dir)
    if (Path(init_dir) / LOOP_CONFIG_FILE).is_file():
        load_loop_state(text_model, Path(init_dir))
    for adapter_dir in chain:
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    return model, chain


def _loop_config(arm: str, args: argparse.Namespace | None = None) -> LoopConfig | None:
    if arm not in LOOP_ARMS:
        return None
    block = None
    if args is not None and getattr(args, "block", None):
        block = parse_block(args.block)
    return LoopConfig(
        n_loops=1,
        bridge=getattr(args, "bridge", "scalar") if args is not None else "scalar",
        bridge_init=(
            getattr(args, "bridge_init", 0.99) if args is not None else 0.99
        ),
        ple_mode="loop_vector" if arm == "L-plevec" else "shared",
        block=block,
        adapter_policy=(
            getattr(args, "adapter_policy", "always") if args is not None else "always"
        ),
        lora_target_layers=(
            getattr(args, "lora_target_layers", "language")
            if args is not None
            else "language"
        ),
    )


def resolve_lora_target_layers(
    loop: LoopConfig | None, args: argparse.Namespace | None = None
) -> str:
    """Target set for every arm, looped or not.

    Unlooped O-H/O-H-ext arms have no ``LoopConfig``; they still honour
    ``--lora-target-layers`` so ``loop_block`` resolves against the same plan
    the L arm uses.
    """
    if loop is not None and loop.lora_target_layers:
        return loop.lora_target_layers
    return getattr(args, "lora_target_layers", None) or "language"


def resolve_lora_layers(plan, target_layers: str) -> list[int]:
    """Concrete layer indices selected by a LoRA target set."""
    name = target_layers or "language"
    if name == "language":
        return list(range(plan.num_layers))
    if name == "loop_block":
        return list(plan.loop)
    if name == "loop_block_coda":
        return list(plan.loop) + list(plan.coda)
    raise ValueError(f"unknown lora target set {target_layers!r}")


def configure_logging() -> None:
    """Project logging with accelerate quiet (it logs once per micro-batch)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("accelerate").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ObjectiveConfig:
    """The effective (possibly prereg-registered) Stage 1 objective."""

    dial: bool
    heal: bool
    window: int
    beta: float
    lambda_hinge: float
    horizon: tuple[float, ...]
    allow_no_teacher: bool

    @property
    def needs_teacher(self) -> bool:
        return self.dial or self.heal


def _loop_probs_arg(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("--loop-probs must contain at least one value")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"--loop-probs must sum to 1, got {sum(values)}")
    return values


def resolve_objective(
    args: argparse.Namespace, prereg=None
) -> ObjectiveConfig:
    """Bind a registered run to prereg.yaml; refuse CLI overrides.

    Without ``--prereg`` the CLI is authoritative.  With ``--prereg`` the arm
    decides the objective (L/L-plevec -> dial, O-H/O-H-ext -> single-exit heal,
    O -> legacy SFT) and every registered value comes from the prereg.  Any
    override, a one-hot smoke opt-in, or a flag for the wrong arm is exit 2.
    """

    requested_probs = _loop_probs_arg(args.loop_probs)
    pilot_single_exit = bool(getattr(args, "pilot_single_exit", False))
    registered = prereg is not None and (
        prereg.dial_objective is not None or prereg.stage1 is not None
    )
    if not registered:
        horizon = requested_probs
        if horizon is None:
            if prereg is not None:
                horizon = tuple(prereg.loop_probs)
            else:
                horizon = (
                    DIAL_LOOP_PROBS if args.dial_objective else DEFAULT_LOOP_PROBS
                )
        return ObjectiveConfig(
            dial=bool(args.dial_objective),
            heal=bool(args.heal_teacher),
            window=int(args.dial_window),
            beta=float(args.dial_beta),
            lambda_hinge=float(args.dial_lambda),
            horizon=tuple(float(value) for value in horizon),
            allow_no_teacher=bool(args.allow_no_teacher),
        )

    dial_spec = prereg.dial_objective
    stage1 = prereg.stage1
    reg_window = int(dial_spec.window_trainable_passes) if dial_spec else DEFAULT_WINDOW
    reg_beta = (
        float(stage1.teacher.beta_parent_kl) if stage1 is not None else DEFAULT_BETA
    )
    reg_lambda = (
        float(dial_spec.monotone_hinge.get("lambda", DEFAULT_LAMBDA_HINGE))
        if dial_spec
        else DEFAULT_LAMBDA_HINGE
    )
    if dial_spec is not None and dial_spec.horizon:
        reg_horizon = tuple(float(value) for value in dial_spec.horizon)
    else:
        reg_horizon = tuple(float(value) for value in prereg.loop_probs)
    if abs(sum(reg_horizon) - 1.0) > 1e-6:
        _refuse("prereg.loop_probs / dial_objective.horizon must sum to 1")

    if args.dial_window != DEFAULT_WINDOW and int(args.dial_window) != reg_window:
        _refuse(
            f"--dial-window={args.dial_window} overrides the registered "
            f"window_trainable_passes={reg_window}"
        )
    if abs(float(args.dial_beta) - DEFAULT_BETA) > 1e-12 and abs(
        float(args.dial_beta) - reg_beta
    ) > 1e-12:
        _refuse(
            f"--dial-beta={args.dial_beta} overrides the registered "
            f"stage1.teacher.beta_parent_kl={reg_beta}"
        )
    if abs(float(args.dial_lambda) - DEFAULT_LAMBDA_HINGE) > 1e-12 and abs(
        float(args.dial_lambda) - reg_lambda
    ) > 1e-12:
        _refuse(
            f"--dial-lambda={args.dial_lambda} overrides the registered "
            f"monotone_hinge.lambda={reg_lambda}"
        )
    if (
        requested_probs is not None
        and tuple(requested_probs) != reg_horizon
        and not pilot_single_exit
    ):
        _refuse("--loop-probs overrides the registered loop_probs/horizon")
    if pilot_single_exit and requested_probs is not None:
        reg_horizon = tuple(requested_probs)

    dial_enabled = args.arm in LOOP_ARMS and not pilot_single_exit
    heal_enabled = args.arm in {"O-H", "O-H-ext"} or pilot_single_exit
    if args.dial_objective and not dial_enabled:
        _refuse(
            f"--dial-objective is not registered for arm {args.arm!r}"
        )
    if args.heal_teacher and not heal_enabled:
        _refuse(
            f"--heal-teacher is not registered for arm {args.arm!r}"
        )
    if args.allow_no_teacher:
        _refuse("--allow-no-teacher is not allowed for a --prereg run")
    return ObjectiveConfig(
        dial=bool(dial_enabled),
        heal=bool(heal_enabled),
        window=reg_window,
        beta=reg_beta,
        lambda_hinge=reg_lambda,
        horizon=reg_horizon,
        allow_no_teacher=False,
    )


def _refuse(message: str) -> None:
    """Refuse to start with exit code 2 (the launcher's refusal convention)."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def apply_abort_decision(
    args: argparse.Namespace, decision: AbortResumeDecision
) -> bool:
    """Apply the shared abort-resume decision to ``args``.

    Returns ``True`` when the acknowledged abort has no COMPLETE checkpoint and
    the caller must archive the partial directory before training.  A refusal
    is terminal (exit 2).  With an acknowledged checkpoint, an unset
    ``--resume`` becomes that exact COMPLETE checkpoint.
    """

    if decision.action == "refuse":
        alert = dict(decision.guard or {}).get("alert") or {}
        _refuse(
            f"heal monitor aborted {args.out} (alert {decision.alert_id}): "
            f"{alert.get('rule', '?')}: {alert.get('message', '')}. "
            f"Pass --ack-abort {decision.alert_id} to resume."
        )
    if decision.action == "resume" and decision.checkpoint is not None:
        if args.resume is None:
            args.resume = str(decision.checkpoint.path)
    if decision.action == "restart":
        # R3-I2: there is no COMPLETE checkpoint, so a still-set ``--resume
        # auto`` would make resolve_resume_checkpoint() exit 2 on the fresh
        # run.json written below.  The registered rule is restart from zero.
        args.resume = None
    return decision.action == "restart"


def record_abort_ack(out_dir: str | Path, guard: Mapping[str, Any]) -> None:
    """Record ``--ack-abort`` in run.json and clear the persisted flags."""

    out_dir = Path(out_dir)
    alert = dict(guard.get("alert") or {"alert_id": guard.get("alert_id")})
    run_path = out_dir / "run.json"
    payload: dict[str, Any] = {}
    if run_path.is_file():
        try:
            loaded = json.loads(run_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    alert_id = str(alert.get("alert_id") or guard.get("alert_id"))
    acks = [item for item in (payload.get("abort_acks") or []) if isinstance(item, dict)]
    # R3-M2: the orchestrator records the same ack after the child exits; an
    # acknowledgement is one audit entry, not one per recorder.
    if not any(str(item.get("alert_id")) == alert_id for item in acks):
        acks.append(
            {
                "alert_id": alert_id,
                "acked_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "source": "train --ack-abort",
            }
        )
    payload["abort_acks"] = acks
    atomic_write_json(run_path, payload)
    state_path = out_dir / "heal_monitor_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if isinstance(state, dict):
            rules = state.get("rules")
            if isinstance(rules, dict):
                rules["abort_triggered"] = False
                rules["abort_emitted"] = []
                rules["warn_last"] = {}
            state["abort_ack"] = {
                "alert_id": str(alert.get("alert_id") or guard.get("alert_id")),
                "acked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            atomic_write_json(state_path, state)
    try:
        (out_dir / ABORT_GUARD_FILE).unlink()
    except FileNotFoundError:
        pass


def handle_abort_resume(
    args: argparse.Namespace, decision: AbortResumeDecision
) -> int | None:
    """Post-lock side effects of an acknowledged abort.

    Returns ``GUARD_EXIT_CODE`` when the partial directory cannot be archived,
    otherwise ``None``.  ``restart`` archives the partial run before training;
    both ``resume`` and ``restart`` record and clear the acknowledgement.
    """

    if decision.action not in ("resume", "restart"):
        return None
    if decision.action == "restart":
        try:
            archived = archive_run_state(args.out, reason="ack-abort")
        except RetryableIOError as exc:
            LOGGER.error("could not archive the aborted run (exit 75): %s", exc)
            return GUARD_EXIT_CODE
        if archived.is_dir():
            LOGGER.warning("archived the aborted run under %s", archived)
    record_abort_ack(args.out, dict(decision.guard or {}))
    return None


def monitor_teacher_load_is_retryable(exc: BaseException) -> bool:
    """True for transient filesystem failures while reading monitor shards.

    Content and contract errors (bad JSON, wrong teacher identity, missing
    manifest keys) stay terminal; an ``OSError`` from OneDrive/antivirus/disk
    contention must be retried by the orchestrator with exit 75 (R2-M4).
    """

    if isinstance(exc, (OSError, RetryableIOError)):
        return True
    if isinstance(exc, TeacherManifestMismatch):
        cause: BaseException | None = exc.__cause__
        while cause is not None:
            if isinstance(cause, (OSError, RetryableIOError)):
                return True
            cause = cause.__cause__
    return False


def _monitor_env_off() -> bool:
    """True when ``GEMMALOOP_HEAL_MONITOR`` explicitly disables the monitor."""

    explicit = os.environ.get("GEMMALOOP_HEAL_MONITOR")
    if explicit is None:
        return False
    return explicit.strip().lower() in ("0", "false", "no", "off", "")


def _monitor_enabled(args: argparse.Namespace, *, dial: bool, heal: bool) -> bool:
    """The monitor on/off state the run will actually use."""

    if os.environ.get("GEMMALOOP_HEAL_MONITOR") is not None:
        return not _monitor_env_off()
    return bool((dial or heal) and not args.smoke)


def refuse_registered_monitor_escapes(args: argparse.Namespace) -> None:
    """M7: a registered run may not silently disable or bypass the monitor."""

    if args.prereg is None and getattr(args, "pilot_arm", None) is None:
        return
    if _monitor_env_off():
        _refuse(
            "GEMMALOOP_HEAL_MONITOR=0 disables the heal monitor; a "
            "--prereg/--pilot-arm run cannot turn its fail-closed split checks "
            "off. Unset the variable for the registered run."
        )
    if args.smoke:
        _refuse(
            "--smoke relaxes the frozen-run and registered monitor checks; it "
            "is not allowed together with --prereg/--pilot-arm"
        )


@dataclass(frozen=True)
class RecipeConfig:
    """The effective Stage 1 optimiser recipe (CLI or registered)."""

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    bridge_lr: float
    loop_ple_lr: float
    gradient_clip: float
    warmup_fraction: float
    optimizer: str
    resolved_optimizer: str = ""
    registered: bool = False


def resolve_recipe(args: argparse.Namespace, prereg=None) -> RecipeConfig:
    """Bind the Stage 1 recipe to prereg.yaml under ``--prereg``.

    Without ``--prereg`` (or with a prereg that has no Stage 1 section) the CLI
    is authoritative.  With ``--prereg`` the registered values are applied and
    any CLI value that differs is exit 2.

    Optimizers are compared after resolution, not by spelling: the registered
    ``AdamW`` and an explicit ``adamw_8bit`` are the same GPU optimizer, and on
    CPU both are the ``adamw_torch`` fallback.  A genuine mismatch (the
    registered recipe asks for 8-bit and the run would use torch, or vice
    versa) is exit 2 on a CUDA device.
    """
    cuda = torch.cuda.is_available()
    cli = RecipeConfig(
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.lr),
        bridge_lr=float(args.bridge_lr),
        loop_ple_lr=float(args.loop_ple_lr),
        gradient_clip=float(args.gradient_clip),
        warmup_fraction=float(args.warmup_ratio),
        optimizer=optimizer_display_name(args.optimizer) or "",
        resolved_optimizer=resolve_optimizer_name(args.optimizer, cuda=cuda),
        registered=False,
    )
    stage1 = prereg.stage1 if prereg is not None else None
    if stage1 is None:
        return cli

    lora = (
        prereg.architecture.lora
        if prereg.architecture is not None
        else None
    )
    registered = RecipeConfig(
        per_device_train_batch_size=REGISTERED_PER_DEVICE_BATCH,
        gradient_accumulation_steps=int(stage1.rows_per_optimizer_step),
        learning_rate=(
            float(lora.lr) if lora is not None else REGISTERED_LORA_LR
        ),
        bridge_lr=(
            float(lora.bridge_lr) if lora is not None else REGISTERED_BRIDGE_LR
        ),
        loop_ple_lr=(
            float(lora.loop_ple_lr)
            if lora is not None
            else REGISTERED_LOOP_PLE_LR
        ),
        gradient_clip=float(stage1.gradient_clip),
        warmup_fraction=float(stage1.warmup_fraction),
        optimizer=optimizer_display_name(stage1.optimizer) or "",
        resolved_optimizer=resolve_optimizer_name(stage1.optimizer, cuda=cuda),
        registered=True,
    )

    if cli.per_device_train_batch_size != registered.per_device_train_batch_size:
        _refuse(
            f"--per-device-train-batch-size={cli.per_device_train_batch_size} "
            "overrides the registered per-device batch of "
            f"{registered.per_device_train_batch_size} row(s)"
        )
    if cli.gradient_accumulation_steps != registered.gradient_accumulation_steps:
        _refuse(
            "--gradient-accumulation-steps="
            f"{cli.gradient_accumulation_steps} overrides the registered "
            "stage1.rows_per_optimizer_step="
            f"{registered.gradient_accumulation_steps}"
        )
    if abs(cli.learning_rate - registered.learning_rate) > 1e-12:
        _refuse(
            f"--lr={cli.learning_rate} overrides the registered "
            f"architecture.lora.lr={registered.learning_rate}"
        )
    if abs(cli.bridge_lr - registered.bridge_lr) > 1e-12:
        _refuse(
            f"--bridge-lr={cli.bridge_lr} overrides the registered "
            f"architecture.lora.bridge_lr={registered.bridge_lr}"
        )
    if abs(cli.loop_ple_lr - registered.loop_ple_lr) > 1e-12:
        _refuse(
            f"--loop-ple-lr={cli.loop_ple_lr} overrides the registered "
            f"architecture.lora.loop_ple_lr={registered.loop_ple_lr}"
        )
    if abs(cli.gradient_clip - registered.gradient_clip) > 1e-12:
        _refuse(
            f"--gradient-clip={cli.gradient_clip} overrides the registered "
            f"stage1.gradient_clip={registered.gradient_clip}"
        )
    if abs(cli.warmup_fraction - registered.warmup_fraction) > 1e-12:
        _refuse(
            f"--warmup-ratio={cli.warmup_fraction} overrides the registered "
            f"warmup of {registered.warmup_fraction:.4g} "
            f"(stage1.schedule: {stage1.schedule!r})"
        )
    if cli.resolved_optimizer != registered.resolved_optimizer:
        _refuse(
            f"--optimizer={cli.optimizer} resolves to "
            f"{cli.resolved_optimizer}, but the registered "
            f"stage1.optimizer={stage1.optimizer} resolves to "
            f"{registered.resolved_optimizer}"
        )
    if (
        cuda
        and registered.optimizer in ADAMW_ALIASES
        and registered.resolved_optimizer != ADAMW_8BIT
    ):
        _refuse(
            f"the registered stage1.optimizer={stage1.optimizer} must resolve "
            f"to {ADAMW_8BIT} on a CUDA device, got "
            f"{registered.resolved_optimizer}"
        )
    return registered


def _checkpoint_step(path: Path) -> int | None:
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        return None
    suffix = name[len("checkpoint-"):]
    return int(suffix) if suffix.isdigit() else None


_RUN_LOCK = None


def acquire_run_lock(out_dir: str | Path) -> object:
    """Process-lifetime lock so duplicate launches cannot touch a live run."""
    global _RUN_LOCK
    path = Path(out_dir) / ".run.lock"
    inherited = os.environ.get("GEMMALOOP_RUN_LOCK_HELD")
    if inherited and Path(inherited).resolve() == path.resolve():
        # The orchestrator owns this run lock for its lifetime; the child only
        # needs to acquire the GPU lock.
        return None
    if _RUN_LOCK is not None and getattr(_RUN_LOCK, "path", None) == path:
        return _RUN_LOCK
    try:
        _RUN_LOCK = acquire(path)
        return _RUN_LOCK
    except GpuLockBusy as exc:
        print(f"error: {exc}; another process owns this run", file=sys.stderr)
        raise SystemExit(LOCK_BUSY_EXIT_CODE) from None


def release_run_lock() -> None:
    """Release the per-run lock (normal process exit; tests call it explicitly)."""
    global _RUN_LOCK
    if _RUN_LOCK is not None:
        _RUN_LOCK.release()
        _RUN_LOCK = None


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    refuse_registered_monitor_escapes(args)
    if getattr(args, "pilot_arm", None) is not None:
        if args.prereg is None:
            _refuse("--pilot-arm requires --prereg")
        apply_pilot_arm(args, load_prereg(args.prereg))
    sleep_guard = prevent_sleep()
    sleep_guard.__enter__()
    atexit.register(sleep_guard.__exit__, None, None, None)
    if args.allow_no_teacher and not args.smoke:
        _refuse("--allow-no-teacher requires --smoke")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.lr <= 0 or args.bridge_lr < 0 or args.loop_ple_lr < 0:
        raise ValueError("--lr must be positive; loop LRs must be non-negative")
    if args.allow_repeat and args.resume is not None:
        print(
            "error: --resume is not supported together with --allow-repeat "
            "until multi-epoch resume is bit-exactly tested; rerun without "
            "--allow-repeat or start a fresh --out",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.fresh and args.resume is not None:
        print("error: --fresh cannot be combined with --resume", file=sys.stderr)
        raise SystemExit(2)
    if args.rerun_complete and args.resume is not None:
        print(
            "error: --rerun-complete cannot be combined with --resume",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # R2-I2: the shared abort-resume rule.  A resume over a persisted abort
    # record without the matching --ack-abort is refused before any run state
    # is touched; with the ack, use the last COMPLETE checkpoint or restart.
    # R3-M1: --fresh/--rerun-complete with an ack is contradictory (the fresh
    # archive would destroy the checkpoint the ack resumes), so it is refused
    # here, before the output directory or any run state is touched.
    abort_decision = decide_abort_resume(args.out, args.ack_abort)
    if (
        (args.fresh or args.rerun_complete)
        and abort_decision.action in ("resume", "restart")
    ):
        _refuse(
            "--fresh/--rerun-complete cannot be combined with --ack-abort "
            f"{args.ack_abort}: an acknowledged abort resumes the last "
            "COMPLETE checkpoint or restarts the same registered run; use "
            "--fresh without --ack-abort to abandon it"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    ack_restart = apply_abort_decision(args, abort_decision)
    done_pre = read_done(args.out)
    if not is_complete_payload(done_pre):
        if (
            args.resume is None
            and not (args.fresh or args.rerun_complete)
            and not ack_restart
        ):
            if has_run_state(args.out):
                reasons = checkpoint_invalid_reasons(args.out)
                detail = "; ".join(reasons) if reasons else "existing run state"
                print(
                    f"error: {args.out} already contains run state ({detail}); pass "
                    "--resume auto to continue it or --fresh to archive it",
                    file=sys.stderr,
                )
                raise SystemExit(2)

    prereg = load_prereg(args.prereg) if args.prereg is not None else None
    objective = resolve_objective(args, prereg)
    recipe = resolve_recipe(args, prereg)
    if args.dial_objective and args.arm not in LOOP_ARMS and prereg is None:
        _refuse(
            "--dial-objective is only valid for the looped arms "
            f"({sorted(LOOP_ARMS)}), got {args.arm!r}"
        )
    if objective.window < 1:
        raise ValueError(f"--dial-window must be at least 1, got {objective.window}")
    if objective.dial and args.adapter_policy != "loop_only":
        _refuse(
            "the dial objective requires --adapter-policy loop_only; otherwise "
            "exit 1 is not O and the no-grad prefix trains nothing"
        )
    if objective.dial and args.lora_target_layers != "loop_block":
        _refuse(
            "the dial objective requires --lora-target-layers loop_block "
            f"(got {args.lora_target_layers!r})"
        )
    if objective.allow_no_teacher:
        logging.getLogger("train").warning(
            "DIAL TEACHER DISABLED: --allow-no-teacher is a smoke test; the "
            "teacher KL becomes one-hot CE and run.json records teacher_mode=onehot"
        )
    manifest = read_data_manifest_details(args.data)
    manifest_sha256 = manifest.sha256
    manifest_target_tokens = manifest.target_tokens
    loop_probs = tuple(objective.horizon)

    max_length, length_mode = resolve_run_length(
        args, prereg, manifest, args.size, args.arm
    )
    quant = "nf4" if args.size == "e4b" else "bf16"
    loop = _loop_config(args.arm, args)
    configured_block = parse_block(args.block) if args.block else (
        loop.block if loop is not None else None
    )

    repo_root = Path(__file__).resolve().parents[1]
    try:
        git = git_state(repo_root)
        prereg_sha = prereg_sha256(args.prereg)
    except FrozenRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    teacher_manifest_sha256 = teacher_data_sha256(args.teacher_data)
    run_config_payload = build_run_config(
        args,
        quant=quant,
        max_length=max_length,
        objective=objective,
        teacher_manifest_sha256=teacher_manifest_sha256,
        recipe=recipe,
    )
    config_sha = run_config_sha256(run_config_payload)
    identity_hash = run_identity_hash(
        run_config_sha256=config_sha,
        data_manifest_sha256=manifest_sha256,
        git_sha=git.sha,
        seed=args.seed,
    )
    prereg_tag = args.require_tag if args.prereg is not None else None
    identity = {
        "git_sha": git.sha,
        "git_dirty": bool(git.dirty),
        "git_branch": git.branch,
        "allow_dirty": bool(args.allow_dirty),
        "data_manifest_sha256": manifest_sha256,
        "prereg_sha": prereg_sha,
        "prereg_tag": prereg_tag,
        "run_config_sha256": config_sha,
        "run_identity_hash": identity_hash,
        "smoke": bool(args.smoke),
        "run_config": run_config_payload,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Only the read-only DONE check may run before the locks.
    done_before_lock = read_done(args.out)
    if is_complete_payload(done_before_lock):
        saved = done_before_lock.get("run_identity_hash")
        if saved is not None and saved != identity_hash:
            print(
                f"error: {args.out} was completed by a different run identity "
                f"({saved}); use --fresh/--rerun-complete to start over",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(f"already complete: {args.out} (identity {identity_hash[:12]})")
        return 0

    # N4: both locks before cleanup, archive or resume work.
    _run_lock = acquire_run_lock(args.out)
    require_gpu_lock()

    cleaned = cleanup_stale_tmp_dirs(args.out)
    if cleaned:
        LOGGER.warning(
            "removed %d stale checkpoint staging dir(s): %s",
            len(cleaned),
            ", ".join(str(path) for path in cleaned),
        )

    if args.fresh or args.rerun_complete:
        try:
            archived = archive_run_state(
                args.out, reason="fresh" if args.fresh else "rerun-complete"
            )
        except RetryableIOError as exc:
            LOGGER.error("could not archive the old run state (exit 75): %s", exc)
            return GUARD_EXIT_CODE
        if archived.is_dir():
            LOGGER.warning("archived the previous run state under %s", archived)

    abort_exit = handle_abort_resume(args, abort_decision)
    if abort_exit is not None:
        return abort_exit

    # Teacher/objective checks run after the locks, cleanup, archive and resume
    # work (t14b order); a missing/invalid teacher exits 2 before the model.
    if (
        prereg is not None
        and (prereg.dial_objective is not None or prereg.stage1 is not None)
        and objective.needs_teacher
        and args.teacher_data is None
    ):
        _refuse(
            "a registered Stage 1 run needs --teacher-data (precomputed top-k "
            "shards); refusing to start without a teacher"
        )
    teacher_runtime_freeze = None
    if args.teacher_data is not None or args.monitor_teacher_data is not None:
        try:
            teacher_runtime_freeze = load_teacher_runtime_freeze(
                args.teacher_runtime_freeze
            )
        except TeacherRuntimeFreezeError as exc:
            _refuse(str(exc))
    teacher_index = resolve_teacher_index(
        manifest_sha256,
        args.teacher_data,
        allow_smoke=objective.allow_no_teacher,
        render_template_sha256=manifest.chat_template_sha256,
        teacher_runtime_freeze=teacher_runtime_freeze,
    )

    # C1: recover a missing DONE marker from a complete run.json (now locked).
    if check_already_complete(
        args.out,
        identity_hash,
        allow_rerun=bool(args.fresh or args.rerun_complete),
    ):
        return 0

    try:
        check_frozen_run(
            repo_root,
            allow_dirty=bool(args.allow_dirty or args.smoke),
            require_tag=args.require_tag,
            prereg=None if args.smoke else args.prereg,
        )
        assert_code_under(repo_root)
    except FrozenRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if (
        args.resume is None
        and not (args.fresh or args.rerun_complete)
        and not ack_restart
    ):
        if has_run_state(args.out):
            reasons = checkpoint_invalid_reasons(args.out)
            detail = "; ".join(reasons) if reasons else "existing run state"
            print(
                f"error: {args.out} already contains run state ({detail}); pass "
                "--resume auto to continue it or --fresh to archive it",
                file=sys.stderr,
            )
            raise SystemExit(2)

    vram_cap = cap_cuda_allocator(args.vram_cap, args.vram_free_margin_gib) if args.vram_cap > 0 else None

    resume_path = resolve_resume_checkpoint(
        args.resume,
        args.out,
        run_config_sha256=config_sha,
        data_manifest_sha256=manifest_sha256,
        git_sha=git.sha,
        prereg_sha=prereg_sha,
        identity_hash=identity_hash,
        allow_mismatch=args.allow_resume_mismatch,
    )
    if resume_path is not None and args.resume not in (None, "auto"):
        step = _checkpoint_step(resume_path)
        if step is not None:
            newer = [
                path
                for path in list_checkpoint_dirs(args.out)
                if (_checkpoint_step(path) or -1) > step
            ]
            if newer:
                archived = archive_newer_checkpoints(
                    args.out, resume_step=step, reason="rewind"
                )
                LOGGER.warning(
                    "rewind: archived %d newer checkpoint(s) and run files under %s",
                    len(newer),
                    archived,
                )
    # Write the identity only after the resume check so a refused resume cannot
    # destroy the original root identity (checkpoints still carry it).
    try:
        write_run_identity(args.out, identity)
    except RetryableIOError as exc:
        LOGGER.error("could not write the run identity (exit 75): %s", exc)
        return GUARD_EXIT_CODE

    throughput_config = ThroughputGuardConfig(
        window_steps=int(args.throughput_window_steps),
        baseline_start_step=int(args.throughput_baseline_start),
        baseline_end_step=int(args.throughput_baseline_end),
        drop_ratio=float(args.throughput_drop_ratio),
        consecutive_windows=int(args.throughput_failures),
        platform=args.guard_platform,
    )
    ram_guard = RamGuard(float(args.min_ram_gib))
    thresholds = guard_thresholds(
        throughput=throughput_config,
        ram=ram_guard,
        save_every_minutes=float(args.save_every_minutes),
    )
    status = StatusWriter(
        args.out / "status.json",
        run=args.out.name,
        identity=identity,
        thresholds=thresholds,
    )
    status.update(phase="loading-model")

    spec = ModelSpec(args.size, quant, adapters=(), loop=loop)
    LOGGER.info("loading %s (%s) for arm %s", args.size, quant, args.arm)
    model, processor, plan = load_model(spec)
    if loop is None and configured_block is not None:
        # Unlooped controls still need the configured block for the O-H-ext
        # FLOPs-matched budget (the model itself is not looped).
        plan = make_layer_plan(
            model.config.text_config.to_dict(), block=configured_block
        )
    lora_target_layers = resolve_lora_target_layers(loop, args)
    lora_layers = resolve_lora_layers(plan, lora_target_layers)
    text_model = model.model.language_model
    tokenizer = getattr(processor, "tokenizer", processor)
    template_sha256 = verify_chat_template(manifest.chat_template_sha256, tokenizer)
    if teacher_index is not None:
        try:
            validate_teacher_manifest_identity(
                teacher_index.manifest,
                allow_smoke=objective.allow_no_teacher,
                render_template_sha256=template_sha256,
                student_vocab_sha256=tokenizer_vocab_sha256(tokenizer),
                student_vocab_ex_mask_sha256=vocab_ex_mask_sha256(tokenizer),
                frozen_runtime=(
                    None
                    if teacher_runtime_freeze is None
                    else teacher_runtime_freeze.runtime_for_teacher_key(
                        teacher_index.manifest.teacher_key
                    )
                ),
            )
        except (TeacherManifestMismatch, TeacherRuntimeFreezeError) as exc:
            _refuse(f"teacher identity does not match this run: {exc}")

    chain: list[str] = []
    if args.init_adapter is not None:
        model, chain = _merge_init_adapter(
            model, text_model, args.init_adapter, args.size, quant
        )
        text_model = model.model.language_model
        LOGGER.info("merged init adapter; chain=%s", chain)

    freeze_base_model(model)

    rows = _load_rows(args.data)
    dataset = build_dataset(
        tokenizer,
        rows,
        max_length,
        include_row_id=teacher_index is not None,
    )
    if teacher_index is not None:
        try:
            validate_teacher_alignment(teacher_index, dataset)
        except TeacherManifestMismatch as exc:
            _refuse(f"teacher data does not align with the rows: {exc}")

    teacher_columns = all(field in dataset.column_names for field in TEACHER_FIELDS)
    ensure_teacher_available(
        needs_teacher=objective.needs_teacher,
        teacher_available=bool(teacher_columns) or teacher_index is not None,
        allow_no_teacher=objective.allow_no_teacher,
    )
    data_collator = None
    if teacher_index is not None:
        data_collator = teacher_collator(teacher_index, tokenizer)
        LOGGER.info(
            "teacher shards loaded; shard-aware collator installed "
            "(teacher mode=teacher)"
        )
    elif teacher_columns:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        data_collator = TeacherDataCollator(
            DataCollatorForLanguageModeling(
                pad_token_id=tokenizer.pad_token_id or 0
            )
        )
        LOGGER.info(
            "teacher columns present; teacher-aware collator installed "
            "(teacher mode=teacher)"
        )
    monitor_teacher_index = None
    monitor_data_collator = None
    if args.monitor_teacher_data is not None:
        try:
            monitor_teacher_index = TeacherIndex.from_directory(
                args.monitor_teacher_data,
                expect_top_k=DEFAULT_TOP_K,
                verify_shards=True,
            )
            validate_teacher_manifest_identity(
                monitor_teacher_index.manifest,
                allow_smoke=objective.allow_no_teacher,
                render_template_sha256=template_sha256,
                student_vocab_sha256=tokenizer_vocab_sha256(tokenizer),
                student_vocab_ex_mask_sha256=vocab_ex_mask_sha256(tokenizer),
                frozen_runtime=(
                    None
                    if teacher_runtime_freeze is None
                    else teacher_runtime_freeze.runtime_for_teacher_key(
                        monitor_teacher_index.manifest.teacher_key
                    )
                ),
            )
        except TeacherManifestMismatch as exc:
            if monitor_teacher_load_is_retryable(exc):
                LOGGER.error(
                    "transient I/O error loading monitor teacher data (exit 75): %s",
                    exc,
                )
                return GUARD_EXIT_CODE
            _refuse(f"monitor teacher identity does not match this run: {exc}")
        except (OSError, RetryableIOError) as exc:
            LOGGER.error(
                "transient I/O error loading monitor teacher data (exit 75): %s",
                exc,
            )
            return GUARD_EXIT_CODE
        except Exception as exc:  # noqa: BLE001 - deterministic config refusal
            _refuse(f"monitor teacher data unusable: {type(exc).__name__}: {exc}")
        monitor_data_collator = teacher_collator(monitor_teacher_index, tokenizer)

    actual_tokens = sum(len(record["input_ids"]) for record in dataset)
    if actual_tokens <= 0:
        raise ValueError("tokenized dataset is empty")
    dataset_tokens = select_dataset_tokens(manifest, max_length, actual_tokens)
    if manifest.max_length == max_length and dataset_tokens != actual_tokens:
        LOGGER.warning(
            "manifest trained tokens (%d) differ from the rendered dataset (%d)",
            dataset_tokens,
            actual_tokens,
        )

    budget = int(args.tokens)
    if args.arm == "O-H-ext":
        text_config = model.config.text_config.to_dict()
        budget = flops_matched_tokens(
            budget, per_layer_flops(text_config), plan, loop_probs
        )
        LOGGER.info("O-H-ext FLOPs-matched token budget: %d", budget)
    check_oh_ext_budget(args.arm, args.oh_ext_budget, budget)
    if not args.allow_repeat and actual_tokens < budget:
        _refuse(
            f"the rendered dataset has {actual_tokens} tokens (at "
            f"max_length={max_length}) but the budget is {budget}; rebuild the "
            "data or pass --allow-repeat"
        )

    num_epochs = resolve_num_epochs(
        dataset_tokens,
        budget,
        max_length=max_length,
        allow_repeat=args.allow_repeat,
    )
    if num_epochs > 1.0:
        LOGGER.warning(
            "--allow-repeat: dataset is below budget; running %.3f epochs",
            num_epochs,
        )
    mean_tokens_per_row = mean_trained_tokens_per_row(
        manifest, max_length, dataset_tokens, len(dataset)
    )
    min_tokens_per_row = min(
        len(record["input_ids"]) for record in dataset
    )
    per_device_train_batch_size = recipe.per_device_train_batch_size
    gradient_accumulation_steps = recipe.gradient_accumulation_steps
    estimated_steps, tokens_per_step = resolve_max_steps(
        budget,
        mean_tokens_per_row,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
    )
    max_steps = resolve_max_steps_bound(
        budget,
        min_tokens_per_row,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
    )
    LOGGER.info(
        "token LR schedule: budget=%d, ~%d steps (%.1f tokens/step, "
        "%.1f tokens/row), max_steps bound=%d",
        budget,
        estimated_steps,
        tokens_per_step,
        mean_tokens_per_row,
        max_steps,
    )
    time_save = TimeSaveCallback(float(args.save_every_minutes))
    callbacks = [TokenBudgetCallback(budget), SpillGuardCallback(args.out), time_save]

    training_args = build_sft_config(
        args.out,
        max_length=max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        warmup_fraction=recipe.warmup_fraction,
        gradient_clip=recipe.gradient_clip,
        optimizer=recipe.resolved_optimizer or recipe.optimizer,
        seed=args.seed,
        num_train_epochs=1.0,
        max_steps=max_steps,
    )

    trainer = LoopSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=default_lora_config(
            target_layers=lora_target_layers,
            plan=plan,
        ),
        loop_text_model=text_model,
        loop_enabled=loop is not None,
        loop_seed=args.seed,
        loop_probs=loop_probs,
        bridge_lr=recipe.bridge_lr,
        loop_ple_lr=recipe.loop_ple_lr,
        warmup_fraction=recipe.warmup_fraction,
        dial_enabled=bool(objective.dial),
        dial_window=int(objective.window),
        dial_beta=float(objective.beta),
        dial_lambda=float(objective.lambda_hinge),
        dial_horizon_probs=tuple(objective.horizon),
        allow_no_teacher=bool(objective.allow_no_teacher),
        heal_teacher=bool(objective.heal),
        data_collator=data_collator,
        adapters_chain=chain or None,
        token_budget=budget,
        callbacks=callbacks,
    )
    trainer.add_callback(
        SingleEpochGuard(trainer, len(dataset), allow_repeat=args.allow_repeat)
    )
    trainer.set_checkpoint_context(
        git_sha=git.sha,
        git_dirty=bool(git.dirty),
        git_branch=git.branch,
        data_manifest_sha256=manifest_sha256,
        prereg_sha=prereg_sha,
        prereg_tag=prereg_tag,
        run_config_sha256=config_sha,
        run_config=run_config_payload,
        run_identity_hash=identity_hash,
        smoke=bool(args.smoke),
        guard_thresholds=thresholds,
    )
    status_callback = StatusCallback(status, trainer)
    watchdog = WatchdogCallback(
        trainer,
        ThroughputGuard(throughput_config),
        ram_guard,
        status=status,
    )
    trainer.add_callback(watchdog)
    trainer.add_callback(status_callback)
    install_heal_monitor(
        trainer,
        args,
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        monitor_data_collator=monitor_data_collator,
        monitor_teacher_index=monitor_teacher_index,
        rows=rows,
        max_length=max_length,
    )
    perf_config = perf_config_from_args(args)
    apply_train_perf(trainer, perf_config)

    status.update(phase="train")
    try:
        trainer.train(
            resume_from_checkpoint=None if resume_path is None else str(resume_path)
        )
    except MissingTeacherError as exc:
        # A registered teacher-distillation run must refuse (exit 2) before the
        # generic retryable classifier maps the failure to exit 75.
        _refuse(str(exc))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        status.set_exit(f"exit {code}", code)
        raise
    except BaseException as exc:  # noqa: BLE001 - classify before re-raising
        retryable = handle_retryable_training_exception(exc, trainer, status)
        if retryable is not None:
            return retryable
        status.set_exit(f"error: {exc}", 1)
        raise

    if trainer.exit_code == GUARD_EXIT_CODE:
        if guard_is_at_bound(trainer, int(training_args.max_steps)):
            # The guard fired on the last budgeted step: the run is complete.
            # Returning 75 would make a resumed run take one more optimizer
            # step beyond max_steps/the token budget.
            LOGGER.warning(
                "guard fired on the final budget step (step=%d, tokens=%d); "
                "treating the run as complete instead of retrying",
                int(trainer.state.global_step),
                int(trainer.tokens_seen),
            )
            trainer.exit_code = None
        else:
            expected = f"checkpoint-{trainer.state.global_step}"
            last = trainer._last_checkpoint_path
            if last is None or last.name != expected:
                trainer._save_checkpoint(trainer.model, None)
            reason = trainer.exit_reason or "guard"
            status.set_exit(f"guard: {reason}", GUARD_EXIT_CODE)
            LOGGER.error("stopping for a retryable failure (exit 75): %s", reason)
            return GUARD_EXIT_CODE

    try:
        assert_no_silent_repeat(trainer, len(dataset), allow_repeat=args.allow_repeat)
        trainer.save_final_artifacts(args.out)
    except RetryableIOError as exc:
        status.set_exit(f"retryable final-write error: {exc}", GUARD_EXIT_CODE)
        LOGGER.error("retryable final-write failure (exit 75): %s", exc)
        return GUARD_EXIT_CODE

    metadata = build_run_metadata(
        arm=args.arm,
        size=args.size,
        quant=quant,
        data=args.data,
        base_tokens=args.tokens,
        target_tokens=budget,
        max_length=max_length,
        seed=args.seed,
        loop=loop,
        adapters_chain=chain,
        trainer=trainer,
        manifest_sha256=manifest_sha256,
        manifest_target_tokens=manifest_target_tokens,
        chat_template_sha256=template_sha256,
        manifest_chat_template_sha256=manifest.chat_template_sha256,
        length_mode=length_mode,
        model_revision_sha=model_revision(args.size),
        num_epochs=num_epochs,
        dataset_trained_tokens=dataset_tokens,
        allow_repeat=args.allow_repeat,
        max_steps=max_steps,
        estimated_steps=estimated_steps,
        tokens_per_step=tokens_per_step,
        mean_tokens_per_row=mean_tokens_per_row,
        loop_block=(
            (plan.loop.start, plan.loop.stop - 1)
            if args.arm == "O-H-ext"
            else None
        ),
        lora_target_layers=lora_target_layers,
        lora_layers=lora_layers,
        optimizer=recipe.resolved_optimizer or (trainer.args.optim),
        optimizer_requested=recipe.optimizer,
    )
    metadata["train_perf"] = perf_config.to_dict()
    metadata["vram_cap"] = vram_cap
    metadata["dial_objective"] = {
        "enabled": bool(objective.dial),
        "window": int(objective.window),
        "beta": float(objective.beta),
        "lambda_hinge": float(objective.lambda_hinge),
        "horizon_probs": list(objective.horizon),
        "teacher_mode": trainer.dial_teacher_mode,
        "teacher": dict(trainer.dial_teacher_metadata),
        "allow_no_teacher": bool(objective.allow_no_teacher),
        "registered": prereg is not None,
    }
    metadata["heal_teacher"] = {
        "enabled": bool(objective.heal),
        "beta": float(objective.beta),
        "teacher_mode": trainer.dial_teacher_mode,
        "teacher": dict(trainer.dial_teacher_metadata),
        "registered": prereg is not None,
    }
    metadata["teacher"] = build_teacher_metadata(teacher_index, args.teacher_data)
    metadata.update(
        {
            "git_sha": git.sha,
            "git_dirty": bool(git.dirty),
            "git_branch": git.branch,
            "allow_dirty": bool(args.allow_dirty),
            "prereg_sha": prereg_sha,
            "prereg_tag": prereg_tag,
            "run_config_sha256": config_sha,
            "run_identity_hash": identity_hash,
            "smoke": bool(args.smoke),
            "guard_thresholds": thresholds,
            "save_every_minutes": float(args.save_every_minutes),
            "resumed_from": None if resume_path is None else str(resume_path),
            "status_file": str(args.out / "status.json"),
        }
    )
    try:
        run_json_path = write_run_json(args.out, metadata)
        write_done(
            args.out,
            {
                "status": "complete",
                "run_identity_hash": identity_hash,
                "smoke": bool(args.smoke),
                "run_config_sha256": config_sha,
                "data_manifest_sha256": manifest_sha256,
                "git_sha": git.sha,
                "seed": int(args.seed),
                "completed_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "run_json": run_json_path.name,
                "final_step": int(trainer.state.global_step),
                "tokens_seen": int(trainer.tokens_seen),
            },
        )
    except RetryableIOError as exc:
        status.set_exit(f"retryable completion-write error: {exc}", GUARD_EXIT_CODE)
        LOGGER.error("retryable completion-write failure (exit 75): %s", exc)
        return GUARD_EXIT_CODE
    status.update(
        phase="complete",
        step=int(trainer.state.global_step),
        tokens_seen=int(trainer.tokens_seen),
        last_complete_checkpoint=(
            None
            if trainer._last_checkpoint_path is None
            else str(trainer._last_checkpoint_path)
        ),
    )
    status.set_exit("complete", 0)
    LOGGER.info(
        "finished %s: step=%d tokens_consumed=%d budget=%d",
        args.arm,
        trainer.state.global_step,
        trainer.tokens_seen,
        budget,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
