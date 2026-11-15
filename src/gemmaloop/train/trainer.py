"""SFT trainer and callbacks for Gemma-Loop (Task 13).

* :class:`LoopSFTTrainer` draws ``N`` deterministically for every training
  micro batch and calls :func:`gemmaloop.model.looped.set_n_loops` before the
  forward.  The micro-step counter is restored from the checkpoint's
  ``global_step`` (or its sidecar) so a resumed run continues the same loop
  sequence.
* Loop state (the bridge and, for ``L-plevec``, the per-loop PLE vectors) is
  saved next to every adapter checkpoint and restored on resume.  It is
  re-enabled for training **after** PEFT wraps the model; PEFT otherwise
  freezes every non-adapter parameter.
* :class:`TokenBudgetCallback` stops training once the requested number of
  non-padding input tokens has been consumed.  The actual count is persisted in
  the checkpoint and in ``metrics.json``.
* Loop parameters (bridge logit, ``loop_ple``) get their own optimizer groups
  with dedicated learning rates, and every logging step records ``w``, the
  bridge gradient norm, ``loop_ple`` norm, total gradient norm and whether
  clipping fired (see :meth:`LoopSFTTrainer.write_metrics`).
* :class:`SpillGuardCallback` records CUDA shared-memory spill in
  ``metrics.json`` (no-op on CPU).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import TrainerCallback, TrainerState
from transformers.trainer_optimizer import is_optimizer_factory
from transformers.trainer_utils import TrainOutput
from torch.utils.data import Sampler
from trl import SFTTrainer

from gemmaloop.train.checkpoint import (
    CheckpointCorrupt,
    RetryableIOError,
    assert_resume_state_present,
    atomic_write_json,
    cleanup_stale_tmp_dirs,
    finalize_checkpoint,
    find_latest_valid_checkpoint,
    fsync_directory,
    fsync_file,
    read_complete,
    retry_fs,
    retry_rmtree,
    rotate_checkpoints,
    validate_checkpoint,
)

from gemmaloop.model.layer_plan import LayerPlan
from gemmaloop.model.load import lora_target_modules
from gemmaloop.model.looped import (
    LOOP_CONFIG_FILE,
    load_loop_state,
    save_loop_state,
    set_n_loops,
)
from gemmaloop.train.dial_objective import (
    DEFAULT_BETA,
    DEFAULT_LAMBDA_HINGE,
    DEFAULT_WINDOW,
    DIAL_LOOP_PROBS,
    TeacherTopK,
    dial_objective,
    sample_horizon,
    teacher_heal_loss,
    window_bounds,
)
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS, sample_n_loops
from gemmaloop.train.teacher_collator import TEACHER_FIELDS, TEACHER_METADATA_FIELDS

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "checkpoint-"
METRICS_FILE = "metrics.json"
ADAPTERS_CHAIN_FILE = "adapters_chain.json"
TOKEN_STATE_FILE = "token_state.json"
TRAINER_STATE_FILE = "trainer_state.json"
CHECKPOINT_KEEP = 2

_EPOCH_MULTIPLIER = 1_000_003


class DeterministicResumeSampler(Sampler):
    """Epoch-stable permutation with an explicit next-position on resume.

    Hugging Face's ``RandomSampler`` draws a fresh permutation from the global
    RNG for every epoch, so after a crash the resumed dataloader can visit a
    different order even when the RNG state is restored.  This sampler derives
    the epoch permutation from a dedicated generator (``seed + epoch``) and can
    start at the exact next row position saved in the checkpoint.
    """

    def __init__(
        self,
        data_source_size: int,
        *,
        seed: int,
        shuffle: bool = True,
        start_epoch: int = 0,
        start_position: int = 0,
    ) -> None:
        self.size = int(data_source_size)
        if self.size <= 0:
            raise ValueError("sampler needs a non-empty dataset")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = int(start_epoch)
        self.position = int(start_position) % self.size

    def _order(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.size))
        generator = torch.Generator()
        generator.manual_seed(
            (self.seed * _EPOCH_MULTIPLIER + int(epoch)) % (2**63 - 1)
        )
        return torch.randperm(self.size, generator=generator).tolist()

    def __iter__(self):
        epoch = self.epoch
        start = self.position
        self.epoch += 1
        self.position = 0
        order = self._order(epoch)
        for index in order[start:]:
            yield index

    def __len__(self) -> int:
        return max(0, self.size - self.position)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

LOOP_BRIDGE_NAME = "loop_bridge_logit"
LOOP_BRIDGE_G_NAME = "loop_bridge_g"
LOOP_PLE_NAME = "loop_ple"
GROUP_LORA_DECAY = "lora_decay"
GROUP_LORA_NO_DECAY = "lora_no_decay"
GROUP_BRIDGE = "bridge"
GROUP_LOOP_PLE = "loop_ple"


class MissingTeacherError(RuntimeError):
    """A registered teacher-distillation run has no teacher fields in its batch.

    ``scripts/train.py`` turns this into exit code 2 (a refusal to start), and
    only ``--allow-no-teacher`` converts it into the loud one-hot smoke mode.
    """

__all__ = [
    "ADAPTERS_CHAIN_FILE",
    "CHECKPOINT_KEEP",
    "DeterministicResumeSampler",
    "GROUP_BRIDGE",
    "GROUP_LOOP_PLE",
    "GROUP_LORA_DECAY",
    "GROUP_LORA_NO_DECAY",
    "LOOP_BRIDGE_G_NAME",
    "LOOP_BRIDGE_NAME",
    "LOOP_PLE_NAME",
    "MissingTeacherError",
    "TOKEN_STATE_FILE",
    "LoopSFTTrainer",
    "SingleEpochGuard",
    "SpillGuardCallback",
    "TokenBudgetCallback",
    "default_lora_config",
    "enable_loop_parameters",
    "freeze_base_model",
    "heldout_loss",
    "token_lr_multiplier",
]


def token_lr_multiplier(
    consumed_tokens: int,
    budget_tokens: int,
    warmup_fraction: float = 0.03,
) -> float:
    """Cosine multiplier over the token budget with linear token warmup.

    The multiplier depends only on the trainer's non-pad token counter, never on
    the optimizer step count, so every arm anneals to zero exactly when its own
    budget is consumed.
    """

    if budget_tokens <= 0:
        return 0.0
    consumed = max(0, min(int(consumed_tokens), int(budget_tokens)))
    warmup_tokens = float(budget_tokens) * float(warmup_fraction)
    if warmup_tokens > 0 and consumed < warmup_tokens:
        return consumed / warmup_tokens
    remaining = float(budget_tokens) - warmup_tokens
    if remaining <= 0:
        return 0.0
    progress = (consumed - warmup_tokens) / remaining
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the model the lean LM head was installed on (PEFT-aware)."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _loop_text_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the ``Gemma4TextModel`` inside any model wrapper.

    Works for both looped and unlooped models: the caller decides whether the
    loop configuration has to be present.
    """
    base = _base_model(model)
    inner = getattr(base, "model", None)
    text_model = getattr(inner, "language_model", inner)
    if text_model is None:
        raise ValueError("cannot locate the text model inside the wrapper")
    return text_model


def heldout_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    n_loops: int,
) -> torch.Tensor:
    """Cross-entropy loss of ``batch`` at exactly ``n_loops`` block passes.

    Held-out evaluation must not inherit the N left behind by the training
    sampler (``compute_loss`` only sets it while training).  This helper sets N
    explicitly, switches to ``eval()``, and restores both the previous N and
    the previous training flag.
    """
    text_model = _loop_text_model(model)
    looped = getattr(text_model, "_gemmaloop_loop_config", None) is not None
    previous_n = int(getattr(text_model, "_gemmaloop_n_loops", 1))
    was_training = bool(model.training)
    # When the training perf pass installed the lean fused head, evaluate
    # through it: it projects only labelled positions and never materialises
    # the full [tokens, vocab] logits tensor (2 GiB bf16 at 4096 tokens).
    lean = getattr(_base_model(model), "_gemmaloop_lean_lm_head", None) is not None
    try:
        model.eval()
        if looped:
            set_n_loops(text_model, int(n_loops))
        with torch.no_grad():
            if lean:
                outputs = model(**batch, fused_lm_head=True)
            else:
                outputs = model(**batch)
        return outputs.loss
    finally:
        if looped:
            set_n_loops(text_model, previous_n)
        model.train(was_training)


def default_lora_config(
    target_layers: str | None = None, plan: LayerPlan | None = None
) -> LoraConfig:
    """LoRA configuration from the global constraints (Task 13).

    ``target_layers`` selects the configurable target layer set: ``"language"``
    (default, every language layer), ``"loop_block"`` (only the repeated block)
    or ``"loop_block_coda"`` (the looped block plus coda, both resolved from
    ``plan``).
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(target_layers, plan),
    )


def freeze_base_model(model: torch.nn.Module) -> None:
    """Freeze every parameter; the caller adds LoRA afterwards.

    ``prepare_model_for_kbit_training`` is deliberately **not** used: it casts
    the PLE table and the vision tower to fp32 (9-11 GiB) and does not fit in
    12 GiB (reference 08, trap 7).
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vision = getattr(getattr(model, "model", None), "vision_tower", None)
    if vision is not None:
        vision.eval()
    embed_vision = getattr(getattr(model, "model", None), "embed_vision", None)
    if embed_vision is not None:
        embed_vision.eval()


def enable_loop_parameters(text_model: torch.nn.Module) -> None:
    """Mark the bridge (and per-loop PLE vectors) as trainable."""
    for name in (LOOP_BRIDGE_NAME, LOOP_BRIDGE_G_NAME, LOOP_PLE_NAME):
        parameter = getattr(text_model, name, None)
        if parameter is not None:
            parameter.requires_grad_(True)


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _tensor_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(torch.linalg.vector_norm(value.detach().float()).item())


def _relative_hidden_change(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Mean per-token ``||h_t - h_{t-1}|| / ||h_{t-1}||`` (design §2.2)."""
    with torch.no_grad():
        cur = current.detach().float()
        prev = previous.detach().float()
        diff = torch.linalg.vector_norm(cur - prev, dim=-1)
        denom = torch.linalg.vector_norm(prev, dim=-1).clamp(min=1e-12)
        return float((diff / denom).mean().item())


def _period2_ratio(
    current: torch.Tensor, previous: torch.Tensor, two_back: torch.Tensor
) -> float:
    """Mean per-token ``||h_t - h_{t-2}|| / ||h_t - h_{t-1}||`` (§2.2/§7.2)."""
    with torch.no_grad():
        cur = current.detach().float()
        prev = previous.detach().float()
        older = two_back.detach().float()
        numerator = torch.linalg.vector_norm(cur - older, dim=-1)
        denominator = torch.linalg.vector_norm(cur - prev, dim=-1).clamp(min=1e-12)
        return float((numerator / denominator).mean().item())


class TokenBudgetCallback(TrainerCallback):
    """Stop training when ``budget_tokens`` non-padding input tokens are seen."""

    def __init__(self, budget_tokens: int) -> None:
        if budget_tokens < 0:
            raise ValueError(f"budget_tokens must be non-negative, got {budget_tokens!r}")
        self.budget_tokens = int(budget_tokens)
        self.tokens_seen = 0
        self.stopped = False

    def add_tokens(self, count: int) -> None:
        self.tokens_seen += int(count)

    @property
    def exhausted(self) -> bool:
        return self.tokens_seen >= self.budget_tokens

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        if self.exhausted:
            if not self.stopped:
                LOGGER.info(
                    "token budget reached: %d >= %d tokens at step %d",
                    self.tokens_seen,
                    self.budget_tokens,
                    state.global_step,
                )
                self.stopped = True
            control.should_training_stop = True
        return control


class SingleEpochGuard(TrainerCallback):
    """Stop training before a dataset is silently reused.

    Checks the persisted micro-batch counter after every optimizer step and at
    every epoch end.  Raising (rather than only setting ``should_training_stop``)
    prevents the run from writing ``status=complete`` for a second epoch.
    """

    def __init__(self, trainer, dataset_size: int, allow_repeat: bool = False) -> None:
        self.trainer = trainer
        self.dataset_size = int(dataset_size)
        self.allow_repeat = bool(allow_repeat)

    def _check(self, control):  # noqa: ANN001 - HF control object
        if self.allow_repeat:
            return control
        consumed = int(getattr(self.trainer, "micro_batches_seen", 0))
        if consumed > self.dataset_size:
            control.should_training_stop = True
            raise RuntimeError(
                f"trainer consumed {consumed} micro-batches from a "
                f"{self.dataset_size}-row dataset without --allow-repeat; "
                "the manifest probably overstates the trained tokens. "
                "Rebuild the data or pass --allow-repeat."
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        return self._check(control)

    def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        return self._check(control)


class SpillGuardCallback(TrainerCallback):
    """Warn once and record ``spill_detected`` when CUDA memory spills.

    The heuristic is ``max_memory_reserved() > 0.95 * total VRAM``.  On CPU it
    is a no-op and writes nothing.
    """

    def __init__(self, output_dir: str | Path | None = None, threshold: float = 0.95) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.threshold = float(threshold)
        self.spill_detected = False
        self._warned = False
        self._peak_reserved = 0

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        if not torch.cuda.is_available():
            return control
        try:
            total = float(torch.cuda.get_device_properties(0).total_memory)
            reserved = float(torch.cuda.max_memory_reserved())
        except Exception:  # pragma: no cover - CUDA device query failure
            return control
        self._peak_reserved = max(self._peak_reserved, int(reserved))
        if reserved > self.threshold * total:
            self.spill_detected = True
            if not self._warned:
                LOGGER.warning(
                    "CUDA spill detected: %.2f GiB reserved > %.0f%% of %.2f GiB VRAM",
                    reserved / 2**30,
                    self.threshold * 100.0,
                    total / 2**30,
                )
                self._warned = True
                self.write(args)
        return control

    def _metrics_path(self, args) -> Path:  # noqa: ANN001 - HF API
        output_dir = self.output_dir or Path(args.output_dir)
        return output_dir / METRICS_FILE

    def write(self, args) -> None:  # noqa: ANN001 - HF API
        if not torch.cuda.is_available():
            return
        path = self._metrics_path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload["spill_detected"] = bool(self.spill_detected)
        payload["peak_reserved_bytes"] = int(self._peak_reserved)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001 - HF API
        self.write(args)
        return control


class LoopSFTTrainer(SFTTrainer):
    """``SFTTrainer`` that sets ``n_loops`` before every training forward.

    ``loop_enabled`` is false for ``O``, ``O-H`` and ``O-H-ext`` (unlooped
    controls) and true for ``L`` and ``L-plevec``.  ``loop_text_model`` is the
    patched ``Gemma4TextModel`` inside the (possibly multimodal) wrapper.

    Loop parameters get their own optimizer groups (research ruling R1): the
    bridge logit at ``bridge_lr`` (default ``3e-3``) and ``loop_ple`` at
    ``loop_ple_lr`` (default ``1e-3``), both with zero weight decay.  The LoRA
    parameters keep ``args.learning_rate`` and ``args.weight_decay``.  The
    scheduler multiplies every group's LR, so the cosine/warmup shape is shared.

    The token cosine multiplier is 0 at 0 consumed tokens, so the first optimizer
    step always runs at LR 0.  With the registered 3 % warm-up this is harmless
    for the 40M-token arms (about 1.7k steps), but a small pilot/smoke budget can
    have fewer than two warm-up steps: the first step is then LR 0 and the second
    already sits at the cosine peak.  That behaviour is identical for every arm
    and is documented in ``prereg/PREREGISTRATION.md`` section 6.
    """

    def __init__(
        self,
        *args,
        loop_text_model: torch.nn.Module | None = None,
        loop_enabled: bool = False,
        loop_seed: int = 0,
        loop_probs: tuple[float, ...] = DEFAULT_LOOP_PROBS,
        micro_step: int = 0,
        adapters_chain: list[str] | None = None,
        token_budget: int | None = None,
        warmup_fraction: float = 0.03,
        bridge_lr: float = 3e-3,
        loop_ple_lr: float = 1e-3,
        dial_enabled: bool = False,
        dial_window: int = DEFAULT_WINDOW,
        dial_beta: float = DEFAULT_BETA,
        dial_lambda: float = DEFAULT_LAMBDA_HINGE,
        dial_horizon_probs: tuple[float, ...] = DIAL_LOOP_PROBS,
        allow_no_teacher: bool = False,
        heal_teacher: bool = False,
        **kwargs,
    ) -> None:
        if heal_teacher and dial_enabled:
            raise ValueError(
                "heal_teacher is the non-dial single-exit loss; do not combine "
                "it with dial_enabled"
            )
        if dial_enabled:
            if not loop_enabled or loop_text_model is None:
                raise ValueError(
                    "dial_enabled=True requires loop_enabled=True and a "
                    "loop_text_model"
                )
            if int(dial_window) < 1:
                raise ValueError(
                    f"dial_window must be at least 1, got {dial_window}"
                )
            loop_config = getattr(loop_text_model, "_gemmaloop_loop_config", None)
            if loop_config is None:
                raise ValueError(
                    "dial_enabled=True requires an apply_loops()-patched text model"
                )
            if loop_config.bridge == "damped":
                raise ValueError(
                    "the dial objective requires a time-invariant bridge "
                    "(not 'damped')"
                )
            if loop_config.ple_mode == "loop_vector":
                raise ValueError(
                    "the dial objective samples horizons up to 8, which "
                    "ple_mode='loop_vector' cannot represent (4 rows)"
                )
            # Exit 1 == O and the frozen prefix only hold when adapters are off
            # on pass 1 and bound to the repeated block.  ``always`` would train
            # adapters that exit 1 omits and that the no-grad prefix never sees.
            if loop_config.adapter_policy != "loop_only":
                raise ValueError(
                    "the dial objective requires adapter_policy='loop_only' "
                    f"(got {loop_config.adapter_policy!r}); otherwise exit 1 is "
                    "not O and the frozen prefix trains nothing"
                )
            if loop_config.lora_target_layers != "loop_block":
                raise ValueError(
                    "the dial objective requires lora_target_layers='loop_block' "
                    f"(got {loop_config.lora_target_layers!r})"
                )
        super().__init__(*args, **kwargs)
        # Gemma4ForConditionalGeneration declares ``accepts_loss_kwargs = False``,
        # which would make TRL average each row's own NLL (every row weight
        # 1/GA regardless of length).  The project's contract is that weights
        # are token shares, so ask the Trainer to pass ``num_items_in_batch``
        # and let the model divide the summed NLL by the whole step's tokens.
        self.model_accepts_loss_kwargs = True
        if loop_enabled and loop_text_model is None:
            raise ValueError("loop_enabled=True requires loop_text_model")
        self.loop_text_model = loop_text_model
        self.loop_enabled = bool(loop_enabled)
        if self.loop_enabled:
            # PEFT's _mark_only_adapters_as_trainable() freezes every plain
            # parameter, including the bridge and loop_ple added by apply_loops.
            # The optimizer is created lazily in train(), so re-enabling them
            # here is enough for them to enter the optimizer.
            enable_loop_parameters(self.loop_text_model)
        if bridge_lr < 0 or loop_ple_lr < 0:
            raise ValueError("bridge_lr and loop_ple_lr must be non-negative")
        self.bridge_lr = float(bridge_lr)
        self.loop_ple_lr = float(loop_ple_lr)
        self.loop_seed = int(loop_seed)
        self.loop_probs = tuple(float(p) for p in loop_probs)
        self._micro_step = int(micro_step)
        self.adapters_chain = list(adapters_chain) if adapters_chain else None
        self.tokens_seen = 0
        self.micro_batches_seen = 0
        self.token_budget = None if token_budget is None else int(token_budget)
        self.warmup_fraction = float(warmup_fraction)
        self.last_n_loops: int | None = None
        self.dial_enabled = bool(dial_enabled)
        self.dial_window = int(dial_window)
        self.dial_beta = float(dial_beta)
        self.dial_lambda = float(dial_lambda)
        self.dial_horizon_probs = tuple(float(p) for p in dial_horizon_probs)
        self.allow_no_teacher = bool(allow_no_teacher)
        self.heal_teacher = bool(heal_teacher)
        self.dial_teacher_mode: str | None = None
        self.dial_teacher_metadata: dict[str, Any] = {}
        self._no_teacher_warned = False
        self._last_dial_metrics: dict[str, float] = {}
        self._last_heal_metrics: dict[str, float] = {}
        self._metric_history: list[dict[str, Any]] = []
        self._optimizer_steps = 0
        # Measured supervised (labels != -100) tokens, one entry per optimizer
        # step.  ``_supervised_tokens_pending`` accumulates over the micro
        # batches of the step in progress.
        self._supervised_tokens_pending = 0
        self._supervised_tokens_per_step: list[int] = []
        self._clipped_steps = 0
        self._last_grad_norm_total: float | None = None
        self._last_bridge_grad_norm: float | None = None
        self._last_clip_fired = False
        self._pending_clip_fired = False
        # Crash-safety state (Task 14b).
        self.checkpoint_keep = CHECKPOINT_KEEP
        self._checkpoint_context: dict[str, Any] = {}
        self._train_started = time.time()
        self._last_checkpoint_path: Path | None = None
        self._sampler_initial_epoch = 0
        self._sampler_initial_position = 0
        self._resume_global_step = 0
        self.exit_code: int | None = None
        self.exit_reason: str | None = None
        # Crash-safety v2: guard baseline survives process restarts.
        self._throughput_baseline: float | None = None
        self._throughput_low_windows = 0
        self._persisted_guard_state: dict[str, Any] = {}
        self._restored_log_history: list[dict[str, Any]] = []

        tokenizer = getattr(self, "_tokenizer", None) or getattr(
            self, "processing_class", None
        )
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self._pad_token_id is None:
            config = getattr(getattr(self, "model", None), "config", None)
            self._pad_token_id = getattr(config, "pad_token_id", None)

    def create_scheduler(self, num_training_steps, optimizer=None):  # noqa: ANN001
        """Use a token-based cosine schedule when a token budget is configured.

        ``max_steps`` remains only a safety bound: the multiplier is computed
        from ``self.tokens_seen`` at every optimizer step, for every parameter
        group, and reaches zero exactly at ``token_budget``.
        """

        if self.token_budget is None:
            return super().create_scheduler(
                num_training_steps, optimizer=optimizer
            )
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _step: token_lr_multiplier(
                self.tokens_seen,
                self.token_budget,
                self.warmup_fraction,
            ),
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler

    # -- loop sampling and token accounting ---------------------------------
    @property
    def micro_step(self) -> int:
        return self._micro_step

    # -- optimizer groups and metrics ---------------------------------------
    @property
    def clip_rate(self) -> float:
        """Fraction of optimizer steps on which gradient clipping fired."""
        if self._optimizer_steps <= 0:
            return 0.0
        return self._clipped_steps / self._optimizer_steps

    @property
    def supervised_tokens_per_step(self) -> dict[str, Any]:
        """Summary of the measured supervised tokens per optimizer step."""
        values = self._supervised_tokens_per_step
        if not values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def _loop_parameter(self, name: str) -> torch.nn.Parameter | None:
        if not self.loop_enabled or self.loop_text_model is None:
            return None
        return getattr(self.loop_text_model, name, None)

    def build_optimizer_groups(self, model: torch.nn.Module | None = None):
        """Return the LoRA + loop parameter groups, or ``None`` if not looped.

        ``None`` means the arm has no trainable loop parameters (O, O-H,
        O-H-ext), so :meth:`create_optimizer` keeps the stock HF grouping and
        those arms are bit-for-bit unaffected.
        """
        opt_model = self.model if model is None else model
        decay_names = set(self.get_decay_parameter_names(opt_model))

        lora_decay: list[torch.nn.Parameter] = []
        lora_no_decay: list[torch.nn.Parameter] = []
        bridge: list[torch.nn.Parameter] = []
        loop_ple: list[torch.nn.Parameter] = []
        for name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue
            leaf = _leaf_name(name)
            if leaf in (LOOP_BRIDGE_NAME, LOOP_BRIDGE_G_NAME):
                bridge.append(parameter)
            elif leaf == LOOP_PLE_NAME:
                loop_ple.append(parameter)
            elif name in decay_names:
                lora_decay.append(parameter)
            else:
                lora_no_decay.append(parameter)

        if not bridge and not loop_ple:
            return None

        lora_lr = float(self.args.learning_rate)
        lora_weight_decay = float(self.args.weight_decay)
        groups = []
        if lora_decay:
            groups.append(
                {
                    "name": GROUP_LORA_DECAY,
                    "params": lora_decay,
                    "lr": lora_lr,
                    "weight_decay": lora_weight_decay,
                }
            )
        if lora_no_decay:
            groups.append(
                {
                    "name": GROUP_LORA_NO_DECAY,
                    "params": lora_no_decay,
                    "lr": lora_lr,
                    "weight_decay": 0.0,
                }
            )
        if bridge:
            groups.append(
                {
                    "name": GROUP_BRIDGE,
                    "params": bridge,
                    "lr": self.bridge_lr,
                    "weight_decay": 0.0,
                }
            )
        if loop_ple:
            groups.append(
                {
                    "name": GROUP_LOOP_PLE,
                    "params": loop_ple,
                    "lr": self.loop_ple_lr,
                    "weight_decay": 0.0,
                }
            )
        return groups

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """Build the optimizer with separate LR groups for the loop parameters."""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        groups = self.build_optimizer_groups(opt_model)
        if groups is None:
            return super().create_optimizer(model)

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
        if is_optimizer_factory(optimizer_cls):
            # A factory would ignore our explicit groups and silently put the
            # loop parameters back at the LoRA LR, which is the exact failure
            # this trainer exists to avoid.
            raise RuntimeError(
                f"LoopSFTTrainer needs a class-based optimizer for loop groups, "
                f"got factory {optimizer_cls!r}"
            )

        optimizer_kwargs = dict(optimizer_kwargs)
        if "params" in optimizer_kwargs:
            groups = optimizer_kwargs.pop("params")
        if "model" in optimizer_kwargs:
            groups = optimizer_kwargs.pop("model")
        if "optimizer_dict" in optimizer_kwargs:
            groups = optimizer_kwargs.pop("optimizer_dict")
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def _capture_loop_grads(self) -> None:
        self._last_bridge_grad_norm = None
        norms = [
            norm
            for name in (LOOP_BRIDGE_NAME, LOOP_BRIDGE_G_NAME)
            if (parameter := self._loop_parameter(name)) is not None
            and (norm := _tensor_norm(parameter.grad)) is not None
        ]
        if norms:
            self._last_bridge_grad_norm = math.sqrt(sum(norm * norm for norm in norms))

    def _clip_grad_norm(self, model):
        self._capture_loop_grads()
        grad_norm = super()._clip_grad_norm(model)
        value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
        self._pending_clip_fired = bool(
            self.args.max_grad_norm > 0 and value > float(self.args.max_grad_norm)
        )
        return grad_norm

    def _get_grad_norm(self, model, grad_norm=None):
        if grad_norm is None:
            # Clipping is disabled, so _clip_grad_norm was never called: capture
            # the loop gradient norm here instead, keeping the log shape stable.
            self._capture_loop_grads()
        value = super()._get_grad_norm(model, grad_norm=grad_norm)
        self._optimizer_steps += 1
        # One entry per optimizer step: all micro batches since the last step.
        self._supervised_tokens_per_step.append(
            int(self._supervised_tokens_pending)
        )
        self._supervised_tokens_pending = 0
        self._last_clip_fired = bool(self._pending_clip_fired)
        if self._last_clip_fired:
            self._clipped_steps += 1
        self._pending_clip_fired = False
        if value is not None:
            self._last_grad_norm_total = (
                float(value.item()) if hasattr(value, "item") else float(value)
            )
        return value

    def loop_metrics(self) -> dict[str, float]:
        """Current bridge ``w`` and ``loop_ple`` norm (empty for unlooped arms)."""
        metrics: dict[str, float] = {}
        bridge = self._loop_parameter(LOOP_BRIDGE_NAME)
        if bridge is not None:
            metrics["bridge_w"] = float(
                torch.sigmoid(bridge.detach().float()).item()
            )
        bridge_g = self._loop_parameter(LOOP_BRIDGE_G_NAME)
        if bridge_g is not None:
            norm = _tensor_norm(bridge_g)
            if norm is not None:
                metrics["bridge_g_norm"] = norm
        loop_ple = self._loop_parameter(LOOP_PLE_NAME)
        if loop_ple is not None:
            norm = _tensor_norm(loop_ple)
            if norm is not None:
                metrics["loop_ple_norm"] = norm
        return metrics

    def _snapshot_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "grad_norm_total": self._last_grad_norm_total,
            "clip_fired": bool(self._last_clip_fired),
            "clip_rate": self.clip_rate,
        }
        if self.loop_enabled:
            metrics["bridge_grad_norm"] = self._last_bridge_grad_norm
            metrics.update(self.loop_metrics())
        if self.dial_enabled and self._last_dial_metrics:
            metrics.update(self._last_dial_metrics)
        if self.heal_teacher and self._last_heal_metrics:
            metrics.update(self._last_heal_metrics)
        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None, **kwargs) -> None:
        """Attach loop/gradient metrics to every training logging step."""
        if "loss" in logs or "grad_norm" in logs:
            metrics = self._snapshot_metrics()
            logs.update(metrics)
            self._metric_history.append(
                {"step": int(self.state.global_step), **metrics}
            )
        super().log(logs, start_time, **kwargs)

    def _account_tokens(self, count: int) -> None:
        self.tokens_seen += int(count)
        for callback in getattr(self.callback_handler, "callbacks", []):
            add_tokens = getattr(callback, "add_tokens", None)
            if callable(add_tokens):
                add_tokens(int(count))

    def _count_input_tokens(self, inputs: dict[str, Any]) -> None:
        attention_mask = inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            self._account_tokens(int(attention_mask.sum().item()))
            return
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return
        if self._pad_token_id is None:
            self._account_tokens(int(input_ids.numel()))
        else:
            self._account_tokens(int((input_ids != self._pad_token_id).sum().item()))

    def _count_supervised_tokens(self, inputs: dict[str, Any]) -> None:
        """Accumulate the step's supervised tokens (``labels != -100``)."""
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            self._supervised_tokens_pending += int(
                (labels != -100).sum().item()
            )

    def _dial_teacher(self, inputs: dict[str, Any]) -> TeacherTopK | None:
        """Read the offline top-k teacher from the batch, or refuse.

        The batch contract is ``teacher_topk_probs`` ``[B, S, K]`` (float16),
        ``teacher_topk_indices`` ``[B, S, K]`` (int32) and
        ``teacher_tail_prob`` ``[B, S]`` (float32), unshifted and aligned with
        ``labels``.  Missing fields raise :class:`MissingTeacherError` unless
        ``allow_no_teacher=True``; that smoke mode logs a WARNING once and uses
        the labelled next token as a one-hot teacher.
        """
        names = ("teacher_topk_probs", "teacher_topk_indices", "teacher_tail_prob")
        present = [name in inputs and inputs[name] is not None for name in names]
        if not any(present):
            if self.allow_no_teacher:
                if not self._no_teacher_warned:
                    LOGGER.warning(
                        "DIAL TEACHER MISSING: --allow-no-teacher is set, so the "
                        "teacher KL is replaced by one-hot next-token CE; this run "
                        "is a smoke test and must not be registered as a teacher "
                        "distillation run"
                    )
                    self._no_teacher_warned = True
                self.dial_teacher_mode = "onehot"
                self.dial_teacher_metadata = {"reason": "allow_no_teacher"}
                return None
            raise MissingTeacherError(
                "the teacher-distillation objective requires teacher fields "
                "(teacher_topk_probs, teacher_topk_indices, teacher_tail_prob) "
                "in every batch; build the T-heal teacher shards or pass "
                "--allow-no-teacher for an explicit one-hot smoke run"
            )
        if not all(present):
            raise ValueError(
                "a teacher batch must provide all of teacher_topk_probs, "
                "teacher_topk_indices and teacher_tail_prob"
            )
        self.dial_teacher_mode = "teacher"
        metadata: dict[str, Any] = {}
        for field in TEACHER_METADATA_FIELDS:
            value = inputs.get(field)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                metadata[field] = value
        self.dial_teacher_metadata = metadata
        return TeacherTopK(
            probs=inputs["teacher_topk_probs"],
            indices=inputs["teacher_topk_indices"],
            tail_prob=inputs["teacher_tail_prob"],
        )

    def _dial_compute_loss(
        self, model, inputs, return_outputs: bool, num_items_in_batch=None
    ):
        """Multi-exit objective for the current horizon.

        The outer model's logits are bypassed: the text model emits the coda
        hidden states of every supervised exit and the objective projects them
        through the (frozen) LM head in vocabulary chunks.
        """
        if self.last_n_loops is None:
            raise RuntimeError("dial loss needs a sampled horizon before the forward")
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        if input_ids is None or labels is None:
            raise ValueError(
                "the dial objective needs input_ids and labels in the batch"
            )
        horizon = int(self.last_n_loops)
        start, end = window_bounds(horizon, self.dial_window)
        exits = [1, *range(start, end + 1)]
        text_outputs = self.loop_text_model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
            exits=exits,
            window=self.dial_window,
        )
        exit_hidden_states = text_outputs.exit_hidden_states
        base = _base_model(model)
        head = base.get_output_embeddings()
        if head is None:
            raise ValueError("cannot locate the LM head for the dial objective")
        weight = head.weight
        if weight.requires_grad and torch.is_grad_enabled():
            raise ValueError(
                "the dial objective requires a frozen LM-head weight"
            )
        softcap = getattr(
            getattr(self.loop_text_model, "config", None),
            "final_logit_softcapping",
            None,
        )
        result = dial_objective(
            exit_hidden_states,
            weight,
            labels,
            teacher=self._dial_teacher(inputs),
            softcap=softcap,
            beta=self.dial_beta,
            lambda_hinge=self.dial_lambda,
            window=self.dial_window,
        )
        metrics = dict(result.metrics)
        block_states = getattr(text_outputs, "exit_block_states", {})
        for t in range(start, end + 1):
            current = block_states.get(t)
            previous = block_states.get(t - 1)
            if current is None or previous is None:
                continue
            metrics[f"dial_exit{t}_hidden_change"] = _relative_hidden_change(
                current, previous
            )
            two_back = block_states.get(t - 2)
            if two_back is not None:
                metrics[f"dial_exit{t}_period2"] = _period2_ratio(
                    current, previous, two_back
                )
        token_sum = result.token_sum if result.token_sum is not None else result.total
        normalizer = self._normalizer(
            num_items_in_batch, int(metrics.get("dial_tokens", 1))
        )
        loss = token_sum / normalizer
        metrics["dial_normalizer"] = normalizer
        self._last_dial_metrics = metrics
        if return_outputs:
            return loss, {
                "loss": loss,
                "exit_hidden_states": exit_hidden_states,
                "dial": result,
            }
        return loss

    @staticmethod
    def _normalizer(num_items_in_batch, fallback: int) -> float:
        if num_items_in_batch is None:
            return float(max(int(fallback), 1))
        if torch.is_tensor(num_items_in_batch):
            value = float(num_items_in_batch.detach().item())
        else:
            value = float(num_items_in_batch)
        return value if value > 0 else float(max(int(fallback), 1))

    def _parent_hidden_for_heal(self, model, inputs):
        """Adapter-off, N=1 parent hidden state under ``torch.no_grad()``.

        The parent must be the LoRA-free O.  A model with trainable LoRA
        parameters but no ``disable_adapter()`` is refused: silently using the
        adapted model would make the anchor exactly zero (the original C3 bug).
        A genuinely LoRA-free model is its own explicit reference.
        """
        text_model = _loop_text_model(model)
        looped = getattr(text_model, "_gemmaloop_loop_config", None) is not None
        previous_n = int(getattr(text_model, "_gemmaloop_n_loops", 1))
        trainable_lora = any(
            "lora" in name.lower() and parameter.requires_grad
            for name, parameter in model.named_parameters()
        )
        disable = getattr(model, "disable_adapter", None)
        if callable(disable):
            adapter_ctx = disable()
        elif trainable_lora:
            raise RuntimeError(
                "the heal parent anchor needs model.disable_adapter() to compute "
                "the LoRA-free O; refusing to use the adapted model as its own "
                "parent"
            )
        else:
            adapter_ctx = nullcontext()
        try:
            with torch.no_grad(), adapter_ctx:
                if looped:
                    set_n_loops(text_model, 1)
                outputs = text_model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                )
        finally:
            if looped:
                set_n_loops(text_model, previous_n)
        return outputs.last_hidden_state.detach()

    def _heal_compute_loss(
        self, model, inputs, return_outputs: bool, num_items_in_batch=None
    ):
        """Single-exit teacher KL for the non-dial O-H heal.

        Consumes the same unshifted ``[B, S, K]`` teacher fields as the dial
        objective.  Missing fields raise :class:`MissingTeacherError` unless
        the explicit one-hot smoke mode is enabled.
        """
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        if input_ids is None or labels is None:
            raise ValueError("the heal teacher loss needs input_ids and labels")
        text_model = self.loop_text_model or _loop_text_model(model)
        text_outputs = text_model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
        hidden = text_outputs.last_hidden_state
        head = _base_model(model).get_output_embeddings()
        if head is None:
            raise ValueError("cannot locate the LM head for the heal teacher loss")
        weight = head.weight
        if weight.requires_grad and torch.is_grad_enabled():
            raise ValueError("the heal teacher loss requires a frozen LM-head weight")
        softcap = getattr(
            getattr(text_model, "config", None), "final_logit_softcapping", None
        )
        parent_hidden = self._parent_hidden_for_heal(model, inputs)
        per_token = teacher_heal_loss(
            hidden,
            weight,
            labels,
            teacher=self._dial_teacher(inputs),
            parent_hidden=parent_hidden,
            beta=self.dial_beta,
            softcap=softcap,
        )
        tokens = int(per_token.numel())
        normalizer = self._normalizer(num_items_in_batch, tokens)
        loss = per_token.sum() / normalizer
        self._last_heal_metrics = {
            "heal_loss": float(per_token.detach().mean().item()),
            "heal_tokens": tokens,
            "heal_normalizer": normalizer,
        }
        if return_outputs:
            return loss, {
                "loss": loss,
                "last_hidden_state": hidden,
                "heal": dict(self._last_heal_metrics),
            }
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        training = bool(getattr(model, "training", False))
        if training:
            self.micro_batches_seen += 1
            self._count_input_tokens(inputs)
            self._count_supervised_tokens(inputs)
            if self.loop_enabled:
                if self.dial_enabled:
                    n_loops = sample_horizon(
                        self.loop_seed, self._micro_step, self.dial_horizon_probs
                    )
                else:
                    n_loops = sample_n_loops(
                        self.loop_seed, self._micro_step, self.loop_probs
                    )
                set_n_loops(self.loop_text_model, n_loops)
                self.last_n_loops = n_loops
                self._micro_step += 1
            policy = getattr(self, "checkpoint_policy", None)
            if policy is not None:  # gemmaloop.train.perf.AdaptiveCheckpointing
                policy(inputs, enabled=bool(self.args.gradient_checkpointing))
            if self.dial_enabled:
                return self._dial_compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            if self.heal_teacher:
                return self._heal_compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
        # ``remove_unused_columns=False`` keeps teacher columns in the batch;
        # any forward that is not the dial path must not receive them as model
        # kwargs.
        extras = (*TEACHER_FIELDS, *TEACHER_METADATA_FIELDS)
        if any(name in inputs for name in extras):
            inputs = {
                key: value for key, value in inputs.items() if key not in extras
            }
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    # -- deterministic data order -------------------------------------------
    def _detach_dataloader_rng(self, dataloader):  # noqa: ANN001 - HF API
        """Give a DataLoader its own generator so it cannot consume the model RNG.

        PyTorch draws a "base seed" from the loader generator on every
        ``iter(dataloader)``.  With ``generator=None`` that draw uses the global
        torch RNG, so a resumed run would consume one extra draw after
        ``_load_rng_state`` and every dropout mask would shift.  A dedicated
        generator keeps the global RNG sequence identical to the uninterrupted
        run.
        """
        generator = torch.Generator()
        generator.manual_seed((int(self.loop_seed) * _EPOCH_MULTIPLIER + 17) % (2**63 - 1))
        targets = [dataloader]
        base = getattr(dataloader, "base_dataloader", None)
        if base is not None:
            targets.append(base)
        rest = getattr(dataloader, "dataloader", None)
        if rest is not None:
            targets.append(rest)
        for target in targets:
            try:
                target.generator = generator
            except Exception:  # pragma: no cover - read-only wrapper
                pass
        return dataloader

    def _get_dataloader(self, *args, **kwargs):  # noqa: ANN001 - HF API
        return self._detach_dataloader_rng(super()._get_dataloader(*args, **kwargs))

    def _get_train_sampler(self, train_dataset=None):  # noqa: ANN001 - HF API
        """Use an epoch-stable sampler so a resumed run visits the same rows."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None or not hasattr(dataset, "__len__"):
            return super()._get_train_sampler(dataset)
        strategy = getattr(self.args, "train_sampling_strategy", "random") or "random"
        if strategy not in ("random", "sequential"):
            return super()._get_train_sampler(dataset)
        return DeterministicResumeSampler(
            len(dataset),
            seed=int(getattr(self.args, "data_seed", 0) or getattr(self.args, "seed", 0)),
            shuffle=strategy != "sequential",
            start_epoch=self._sampler_initial_epoch,
            start_position=self._sampler_initial_position,
        )

    def _data_epoch_and_position(self) -> tuple[int, int]:
        size = len(self.train_dataset) if self.train_dataset is not None else 0
        if size <= 0:
            return 0, 0
        return self.micro_batches_seen // size, self.micro_batches_seen % size

    # -- checkpoint sidecars and resume -------------------------------------
    def _checkpoint_dir(self, trial) -> Path:  # noqa: ANN001 - HF API
        try:
            output_dir = Path(self._get_output_dir(trial=trial))
        except TypeError:  # pragma: no cover - older transformers signature
            output_dir = Path(self.args.output_dir)
        return output_dir / f"{CHECKPOINT_PREFIX}{self.state.global_step}"

    def set_checkpoint_context(self, **values: Any) -> None:
        """Attach the frozen-run identity recorded in every ``COMPLETE.json``."""
        self._checkpoint_context.update(values)

    def _save_sidecars(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.loop_enabled:
            save_loop_state(self.loop_text_model, directory)
        if self.adapters_chain:
            atomic_write_json(directory / ADAPTERS_CHAIN_FILE, self.adapters_chain)
        epoch, position = self._data_epoch_and_position()
        atomic_write_json(
            directory / TOKEN_STATE_FILE,
            {
                "tokens_seen": int(self.tokens_seen),
                "micro_step": int(self._micro_step),
                "micro_batches_seen": int(self.micro_batches_seen),
                "optimizer_steps": int(self._optimizer_steps),
                "supervised_tokens_per_step": list(
                    self._supervised_tokens_per_step
                ),
                "clipped_steps": int(self._clipped_steps),
                # Exact next row for DeterministicResumeSampler.
                "data_epoch": int(epoch),
                "row_position": int(position),
                "loop_sampler_seed": int(self.loop_seed),
                "loop_enabled": bool(self.loop_enabled),
                # Keep the per-logging-step metric table across a resume.
                "metric_history": list(self._metric_history),
            },
        )

    def _checkpoint_metadata(self, directory: Path, step: int) -> dict[str, Any]:
        epoch, position = self._data_epoch_and_position()
        metadata: dict[str, Any] = {
            "step": int(step),
            "global_step": int(step),
            "tokens_seen": int(self.tokens_seen),
            "micro_step": int(self._micro_step),
            "micro_batches_seen": int(self.micro_batches_seen),
            "optimizer_steps": int(self._optimizer_steps),
            "clipped_steps": int(self._clipped_steps),
            "data_order": {"epoch": int(epoch), "row_position": int(position)},
            "loop_sampler": {
                "seed": int(self.loop_seed),
                "micro_step": int(self._micro_step),
                "enabled": bool(self.loop_enabled),
            },
            "has_optimizer": (directory / "optimizer.pt").is_file(),
            "has_scheduler": (directory / "scheduler.pt").is_file(),
            "has_scaler": (directory / "scaler.pt").is_file(),
            "rng_states_saved": (directory / "rng_state.pth").is_file(),
            "wall_time_seconds": round(float(time.time()) - self._train_started, 3),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "throughput_guard": {
                "baseline_tok_s": self._throughput_baseline,
                "low_windows": int(self._throughput_low_windows),
            },
        }
        metadata.update(self._checkpoint_context)
        return metadata

    def _save_checkpoint(self, model, trial):  # noqa: ANN001 - HF API
        """Write a checkpoint into ``.tmp`` and publish it with a COMPLETE marker."""
        run_dir = Path(self._get_output_dir(trial=trial))
        step = int(self.state.global_step)
        final_dir = run_dir / f"{CHECKPOINT_PREFIX}{step}"
        staging_dir = run_dir / f"{CHECKPOINT_PREFIX}{step}{'.tmp'}"

        if staging_dir.exists():
            retry_rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        original_output = self.args.output_dir
        keep_staging = False
        # Hugging Face appends ``checkpoint-<step>`` to its run directory, so
        # point the run directory at our staging directory.
        self.args.output_dir = str(staging_dir)
        try:
            super()._save_checkpoint(model, trial)
            inner = staging_dir / f"{CHECKPOINT_PREFIX}{step}"
            if not inner.is_dir():  # pragma: no cover - HF invariant
                raise CheckpointCorrupt(
                    f"Hugging Face did not write a checkpoint under {staging_dir}"
                )
            self._save_sidecars(inner)
            metadata = self._checkpoint_metadata(inner, step)
            finalize_checkpoint(inner, final_dir, metadata)
        except RetryableIOError:
            # Leave the fully written staging directory for a later promotion;
            # the previous published checkpoint is still intact.
            keep_staging = True
            raise
        finally:
            self.args.output_dir = original_output
            if staging_dir.exists() and not keep_staging:
                try:
                    retry_rmtree(staging_dir)
                except RetryableIOError as exc:  # pragma: no cover
                    LOGGER.warning("could not remove staging %s: %s", staging_dir, exc)

        self._last_checkpoint_path = final_dir
        rotate_checkpoints(
            run_dir,
            keep=int(self.checkpoint_keep),
            protect=[final_dir],
        )
        LOGGER.info("checkpoint published: %s", final_dir)

    def _sync_budget_callbacks(self) -> None:
        for callback in getattr(self.callback_handler, "callbacks", []):
            if isinstance(callback, TokenBudgetCallback):
                callback.tokens_seen = int(self.tokens_seen)

    def _restore_resume_state(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        state_path = checkpoint / TRAINER_STATE_FILE
        if not state_path.is_file():
            raise CheckpointCorrupt(
                f"{checkpoint} has no {TRAINER_STATE_FILE}; refusing a partial resume"
            )
        state = TrainerState.load_from_json(str(state_path))
        global_step = int(state.global_step)
        self._resume_global_step = global_step
        self._last_checkpoint_path = checkpoint
        self._restored_log_history = list(getattr(state, "log_history", []) or [])
        try:
            complete = read_complete(checkpoint, verify=False)
        except (CheckpointCorrupt, OSError):
            complete = {}
        guard = complete.get("throughput_guard")
        self._persisted_guard_state = dict(guard) if isinstance(guard, dict) else {}
        baseline = self._persisted_guard_state.get("baseline_tok_s")
        self._throughput_baseline = (
            float(baseline) if isinstance(baseline, (int, float)) else None
        )
        self._throughput_low_windows = int(
            self._persisted_guard_state.get("low_windows", 0) or 0
        )
        token_state = checkpoint / TOKEN_STATE_FILE
        if not token_state.is_file():
            raise CheckpointCorrupt(
                f"{checkpoint} has no {TOKEN_STATE_FILE}; refusing a partial resume"
            )
        payload = json.loads(token_state.read_text(encoding="utf-8"))
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        fallback = global_step * int(self.args.gradient_accumulation_steps)
        self._micro_step = int(payload.get("micro_step", fallback))
        self.micro_batches_seen = int(payload.get("micro_batches_seen", fallback))
        self._optimizer_steps = int(payload.get("optimizer_steps", global_step))
        supervised = payload.get("supervised_tokens_per_step")
        if isinstance(supervised, list):
            self._supervised_tokens_per_step = [int(value) for value in supervised]
        self._clipped_steps = int(payload.get("clipped_steps", 0))
        size = len(self.train_dataset) if self.train_dataset is not None else 0
        if size > 0:
            self._sampler_initial_epoch = int(
                payload.get("data_epoch", self.micro_batches_seen // size)
            )
            self._sampler_initial_position = int(
                payload.get("row_position", self.micro_batches_seen % size)
            )
        history = payload.get("metric_history")
        if isinstance(history, list):
            self._metric_history = [
                entry for entry in history if isinstance(entry, dict)
            ]
        self._sync_budget_callbacks()
        if self.loop_enabled:
            if not (checkpoint / LOOP_CONFIG_FILE).exists():
                raise CheckpointCorrupt(
                    f"{checkpoint} has no {LOOP_CONFIG_FILE}; refusing to restart "
                    "the bridge from its current value"
                )
            load_loop_state(self.loop_text_model, checkpoint)
        LOGGER.info(
            "resuming at micro step %d with %d consumed tokens (global step %d, "
            "next row epoch=%d position=%d)",
            self._micro_step,
            self.tokens_seen,
            global_step,
            self._sampler_initial_epoch,
            self._sampler_initial_position,
        )

    def _resume_is_complete(self) -> bool:
        """True when the restored checkpoint already consumed the whole budget."""
        max_steps = int(getattr(self.args, "max_steps", -1) or -1)
        if max_steps > 0 and int(self._resume_global_step) >= max_steps:
            return True
        return self.token_budget is not None and int(self.tokens_seen) >= int(
            self.token_budget
        )

    def write_metrics(self, directory: str | Path | None = None) -> Path:
        """Merge actual consumption counters and training metrics into ``metrics.json``."""
        path = Path(directory or self.args.output_dir) / METRICS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload.update(
            {
                "tokens_consumed": int(self.tokens_seen),
                "micro_step": int(self._micro_step),
                "global_step": int(self.state.global_step),
                "last_n_loops": self.last_n_loops,
                "optimizer_steps": int(self._optimizer_steps),
                "clipped_steps": int(self._clipped_steps),
                "clip_rate": self.clip_rate,
                "final_grad_norm": self._last_grad_norm_total,
            }
        )
        loop_metrics = self.loop_metrics()
        if "bridge_w" in loop_metrics:
            payload["final_bridge_w"] = loop_metrics["bridge_w"]
        if "bridge_g_norm" in loop_metrics:
            payload["final_bridge_g_norm"] = loop_metrics["bridge_g_norm"]
        if "loop_ple_norm" in loop_metrics:
            payload["final_loop_ple_norm"] = loop_metrics["loop_ple_norm"]

        if self._metric_history:
            by_step = {
                int(entry["step"]): entry
                for entry in payload.get("training_metrics", [])
                if isinstance(entry, dict) and "step" in entry
            }
            for entry in self._metric_history:
                by_step[int(entry["step"])] = entry
            payload["training_metrics"] = [by_step[key] for key in sorted(by_step)]

        atomic_write_json(path, payload)
        return path

    def train(self, resume_from_checkpoint=None, *args, **kwargs):  # noqa: ANN001
        resolved = resume_from_checkpoint
        if isinstance(resolved, bool) and resolved:
            info = find_latest_valid_checkpoint(self.args.output_dir)
            if info is None:
                raise ValueError(
                    f"No valid COMPLETE checkpoint in {self.args.output_dir}"
                )
            resolved = str(info.path)
        if isinstance(resolved, str) and resolved:
            metadata = validate_checkpoint(resolved)
            assert_resume_state_present(metadata)
            # The sampler restarts at the exact saved row; Hugging Face must not
            # skip batches again on top of that (which would also skip the RNG
            # restore).  The token_state sidecar remains the authority.
            self.args.ignore_data_skip = True
            self._restore_resume_state(resolved)
        if isinstance(resolved, str) and resolved and self._resume_is_complete():
            # The guard may have fired on the final budget step.  Loading the
            # checkpoint and stopping here is exact: a second train() pass must
            # not take one extra optimizer step beyond max_steps / the budget.
            LOGGER.info(
                "checkpoint %s is already at the training bound (step=%d, "
                "tokens=%d); finalising without another step",
                resolved,
                int(self._resume_global_step),
                int(self.tokens_seen),
            )
            self._load_from_checkpoint(resolved)
            self.state.global_step = int(self._resume_global_step)
            self.state.log_history = list(self._restored_log_history)
            self.write_metrics()
            return TrainOutput(int(self._resume_global_step), 0.0, {})
        self._train_started = time.time()
        cleaned = cleanup_stale_tmp_dirs(self.args.output_dir)
        if cleaned:
            LOGGER.warning(
                "removed %d stale checkpoint staging dir(s): %s",
                len(cleaned),
                ", ".join(str(path) for path in cleaned),
            )
        output = super().train(
            *args, resume_from_checkpoint=resolved, **kwargs
        )
        self.write_metrics()
        return output

    def save_final_artifacts(self, directory: str | Path | None = None) -> Path:
        """Write the final adapter/state/metrics through ``.tmp`` + ``os.replace``."""
        out_dir = Path(directory or self.args.output_dir)
        staging = out_dir / ".final.tmp"
        if staging.exists():
            retry_rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        existing_metrics = out_dir / METRICS_FILE
        if existing_metrics.is_file():
            # Preserve the pre-crash metric history across a resume.
            shutil.copy2(existing_metrics, staging / METRICS_FILE)
        original_output = self.args.output_dir
        self.args.output_dir = str(staging)
        try:
            self.save_state()
            self.save_model(str(staging))
            self._save_sidecars(staging)
            self.write_metrics(staging)
        finally:
            self.args.output_dir = original_output

        for item in sorted(staging.iterdir()):
            destination = out_dir / item.name
            if item.is_dir():
                if destination.exists():
                    retry_rmtree(destination)
                retry_fs(os.replace, str(item), str(destination), what=f"publish {item.name}")
            else:
                retry_fs(os.replace, str(item), str(destination), what=f"publish {item.name}")
            if destination.is_file():
                fsync_file(destination)
        fsync_directory(out_dir)
        try:
            retry_rmtree(staging)
        except RetryableIOError:  # pragma: no cover - cleanup only
            pass
        return out_dir
