"""CPU training tests with a tiny random Gemma 4 model (Task 13).

No Gemma weights and no GPU are used: the model is a few-layer
``Gemma4ForCausalLM`` built from ``tests.model.tiny`` and the dataset is already
tokenized, so these tests exercise the trainer, the deterministic loop sampler,
checkpoint sidecars, resume and the adapter chain merge.
"""

from __future__ import annotations

import json

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    PreTrainedTokenizerFast,
    TrainerCallback,
)
from trl import SFTConfig

from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.train.loop_sampler import sample_n_loops
from gemmaloop.train.trainer import (
    LoopSFTTrainer,
    TokenBudgetCallback,
    _loop_text_model,
    default_lora_config,
    heldout_loss,
)
from tests.model.tiny import tiny_config

TEXT_MODEL_ARGS = {"target_modules": ["q_proj", "v_proj"]}


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    vocab.update({f"tok{i}": i for i in range(3, 256)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )


def _tiny_model(
    seed: int = 0,
    loops: bool = False,
    ple_mode: str = "shared",
    adapter_policy: str | None = None,
    lora_target_layers: str | None = None,
) -> Gemma4ForCausalLM:
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(tiny_config())
    if loops:
        plan = make_layer_plan(tiny_config().to_dict())
        apply_loops(
            model.model,
            plan,
            LoopConfig(
                n_loops=1,
                ple_mode=ple_mode,
                adapter_policy=adapter_policy,
                lora_target_layers=lora_target_layers,
            ),
        )
    return model


def _tiny_conditional_model(seed: int = 0) -> Gemma4ForConditionalGeneration:
    """The real multimodal class TRL sees in production (tiny config)."""
    torch.manual_seed(seed)
    return Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=tiny_config(), vision_config=None, audio_config=None)
    )


def _pattern() -> list[int]:
    return [2] + [10 + (i % 4) for i in range(12)] + [1]


def _rows(count: int = 8) -> list[dict[str, list[int]]]:
    pattern = _pattern()
    return [{"input_ids": list(pattern), "labels": list(pattern)} for _ in range(count)]


def _make_trainer(
    output_dir,
    *,
    seed: int = 0,
    loops: bool = False,
    ple_mode: str = "shared",
    max_steps: int = 6,
    save_steps: int | None = None,
    callbacks=None,
    gradient_accumulation_steps: int = 1,
    adapters_chain: list[str] | None = None,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    bridge_lr: float = 3e-3,
    loop_ple_lr: float = 1e-3,
    lr_scheduler_type: str = "linear",
    rows: list[dict[str, list[int]]] | None = None,
    token_budget: int | None = None,
    model: torch.nn.Module | None = None,
    train_sampling_strategy: str = "sequential",
    lora_dropout: float = 0.0,
    adapter_policy: str | None = None,
    lora_target_layers: str | None = None,
    data_collator=None,
    dial_enabled: bool = False,
    dial_horizon_probs: tuple[float, ...] | None = None,
    heal_teacher: bool = False,
) -> LoopSFTTrainer:
    if model is None:
        model = _tiny_model(
            seed=seed,
            loops=loops,
            ple_mode=ple_mode,
            adapter_policy=adapter_policy,
            lora_target_layers=lora_target_layers,
        )
    inner = getattr(model, "model", model)
    loop_text_model = getattr(inner, "language_model", inner)
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=1,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=2,
        save_only_model=False,
        report_to="none",
        use_cpu=True,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,
        max_length=32,
        train_sampling_strategy=train_sampling_strategy,
        seed=seed,
        data_seed=seed,
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    return LoopSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows if rows is not None else _rows()),
        processing_class=_tiny_tokenizer(),
        data_collator=data_collator,
        peft_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=lora_dropout,
            **TEXT_MODEL_ARGS,
        ),
        loop_text_model=loop_text_model,
        loop_enabled=loops,
        loop_seed=seed,
        bridge_lr=bridge_lr,
        loop_ple_lr=loop_ple_lr,
        adapters_chain=adapters_chain,
        token_budget=token_budget,
        dial_enabled=dial_enabled,
        **(
            {}
            if dial_horizon_probs is None
            else {"dial_horizon_probs": dial_horizon_probs}
        ),
        heal_teacher=heal_teacher,
        callbacks=callbacks or [],
    )


def _losses(trainer: LoopSFTTrainer) -> list[float]:
    return [float(entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]


class _GradCapture(TrainerCallback):
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.grads: list[torch.Tensor | None] = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
        grad = self.parameter.grad
        self.grads.append(None if grad is None else grad.detach().clone())
        return control


def test_tiny_training_reduces_loss(tmp_path):
    trainer = _make_trainer(tmp_path / "run", max_steps=6)
    trainer.train()
    losses = _losses(trainer)
    assert len(losses) == 6
    assert losses[-1] < losses[0]


def _varying_rows() -> list[dict[str, list[int]]]:
    """Two different-length rows: per-row and per-token means differ."""
    rows = []
    for length in (5, 25):
        ids = [2] + [10 + (i % 4) for i in range(length)] + [1]
        rows.append({"input_ids": ids, "labels": list(ids)})
    return rows


def test_logged_loss_is_the_per_token_mean_over_the_optimizer_step(tmp_path):
    """The real multimodal class must weight tokens, not rows (audit C1)."""

    rows = _varying_rows()
    trainer = _make_trainer(
        tmp_path / "per-token",
        model=_tiny_conditional_model(),
        rows=rows,
        gradient_accumulation_steps=2,
        max_steps=1,
    )
    assert trainer.model_accepts_loss_kwargs is True
    assert trainer.model.base_model.accepts_loss_kwargs is False

    model = trainer.model
    was_training = model.training
    model.eval()
    per_row: list[tuple[float, int]] = []
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            loss = model(input_ids=ids, labels=ids.clone()).loss
            per_row.append((float(loss), len(row["labels"]) - 1))
    model.train(was_training)

    total_tokens = sum(count for _, count in per_row)
    expected = sum(loss * count for loss, count in per_row) / total_tokens
    row_mean = sum(loss for loss, _ in per_row) / len(per_row)
    assert abs(expected - row_mean) > 1e-3

    trainer.train()
    logged = _losses(trainer)[0]
    assert logged == pytest.approx(expected, rel=1e-5, abs=1e-5)
    # The pre-fix behaviour (each row weight 1/GA) would log this instead.
    assert abs(logged - row_mean) > 1e-3


def test_tiny_looped_training_reduces_loss(tmp_path):
    trainer = _make_trainer(tmp_path / "loops", loops=True, max_steps=6)
    trainer.train()
    losses = _losses(trainer)
    assert len(losses) == 6
    assert trainer.micro_step == 6
    assert losses[-1] < losses[0]


def test_bridge_is_trainable_and_updated(tmp_path):
    trainer = _make_trainer(tmp_path / "bridge", loops=True, max_steps=1)
    bridge = trainer.loop_text_model.loop_bridge_logit
    assert bridge.requires_grad
    before = bridge.detach().clone()
    capture = _GradCapture(bridge)
    trainer.add_callback(capture)

    trainer.train()

    assert capture.grads and capture.grads[0] is not None
    assert capture.grads[0].abs().sum().item() > 0
    assert not torch.equal(bridge.detach(), before)


def test_loop_ple_is_trainable_and_updated(tmp_path):
    trainer = _make_trainer(
        tmp_path / "ple", loops=True, ple_mode="loop_vector", max_steps=1
    )
    loop_ple = trainer.loop_text_model.loop_ple
    assert loop_ple.requires_grad
    before = loop_ple.detach().clone()
    capture = _GradCapture(loop_ple)
    trainer.add_callback(capture)

    trainer.train()

    assert capture.grads and capture.grads[0] is not None
    assert capture.grads[0].abs().sum().item() > 0
    assert not torch.equal(loop_ple.detach(), before)


def test_unlooped_arm_has_no_loop_parameters(tmp_path):
    trainer = _make_trainer(tmp_path / "oh", loops=False)
    assert not hasattr(trainer.loop_text_model, "loop_bridge_logit")
    assert not hasattr(trainer.loop_text_model, "loop_ple")


def _groups_by_name(trainer: LoopSFTTrainer) -> dict[str, dict]:
    return {
        group["name"]: group
        for group in trainer.optimizer.param_groups
        if group.get("name")
    }


def test_optimizer_groups_isolate_loop_parameters(tmp_path):
    trainer = _make_trainer(
        tmp_path / "groups",
        loops=True,
        ple_mode="loop_vector",
        learning_rate=2e-3,
        weight_decay=0.1,
    )
    trainer.create_optimizer()
    groups = _groups_by_name(trainer)
    bridge = trainer.loop_text_model.loop_bridge_logit
    loop_ple = trainer.loop_text_model.loop_ple

    assert groups["bridge"]["params"] == [bridge]
    assert groups["bridge"]["lr"] == pytest.approx(3e-3)
    assert groups["bridge"]["weight_decay"] == 0.0
    assert len(groups["loop_ple"]["params"]) == 1
    assert groups["loop_ple"]["params"][0] is loop_ple
    assert groups["loop_ple"]["lr"] == pytest.approx(1e-3)
    assert groups["loop_ple"]["weight_decay"] == 0.0

    lora_groups = [groups[name] for name in ("lora_decay", "lora_no_decay") if name in groups]
    assert lora_groups
    lora_params = [parameter for group in lora_groups for parameter in group["params"]]
    assert lora_params
    for parameter in lora_params:
        assert parameter is not bridge
        assert parameter is not loop_ple
        assert "lora_" in next(
            name for name, candidate in trainer.model.named_parameters() if candidate is parameter
        )
    assert groups["lora_decay"]["lr"] == pytest.approx(2e-3)
    assert groups["lora_decay"]["weight_decay"] == pytest.approx(0.1)

    grouped = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
    assert len(grouped) == len(trainable)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}


def test_unlooped_optimizer_has_no_loop_groups(tmp_path):
    trainer = _make_trainer(tmp_path / "oh-groups", loops=False)
    trainer.create_optimizer()
    names = [group.get("name") for group in trainer.optimizer.param_groups]
    assert "bridge" not in names
    assert "loop_ple" not in names


def test_token_scheduler_scales_every_param_group(tmp_path):
    budget = 100
    steps = 8
    trainer = _make_trainer(
        tmp_path / "token-cosine",
        loops=True,
        ple_mode="loop_vector",
        max_steps=steps,
        learning_rate=2e-3,
        bridge_lr=3e-3,
        loop_ple_lr=1e-3,
        token_budget=budget,
    )
    trainer.create_optimizer()
    groups = trainer.optimizer.param_groups
    peaks = {
        group["name"]: group["lr"]
        for group in groups
        if group.get("name")
    }
    trainer.create_scheduler(num_training_steps=steps)
    assert peaks["bridge"] == pytest.approx(3e-3)
    assert peaks["loop_ple"] == pytest.approx(1e-3)
    assert peaks["lora_decay"] == pytest.approx(2e-3)
    assert {"lora_decay", "bridge", "loop_ple"} <= set(peaks)

    # Mark optimizer.step() as called so PyTorch does not warn about the
    # scheduler-only stepping below; there are no gradients, so weights stay put.
    trainer.optimizer.step()
    # The multiplier is a function of consumed tokens, not of scheduler steps.
    for tokens in (int(0.03 * budget), budget // 2, budget - 1):
        trainer.tokens_seen = tokens
        trainer.lr_scheduler.step()
        factors = {
            group["name"]: group["lr"] / peaks[group["name"]]
            for group in groups
            if group.get("name")
        }
        reference = factors["bridge"]
        assert reference > 0
        for name, factor in factors.items():
            assert factor == pytest.approx(reference, rel=1e-6, abs=1e-12), (
                tokens,
                name,
            )

    # Reaching the token budget anneals the cosine to zero in every group,
    # whatever its peak, regardless of the max_steps safety bound.
    trainer.tokens_seen = budget
    trainer.lr_scheduler.step()
    assert all(group["lr"] == pytest.approx(0.0, abs=1e-12) for group in groups)


def _bridge_displacement(tmp_path, bridge_lr: float, steps: int = 15) -> float:
    """Run a synthetic loss that rewards using the loop; return |Δa|."""
    trainer = _make_trainer(
        tmp_path / f"displacement-{bridge_lr}",
        loops=True,
        max_steps=steps,
        learning_rate=1e-4,
        bridge_lr=bridge_lr,
    )
    trainer.create_optimizer()
    bridge = trainer.loop_text_model.loop_bridge_logit
    before = float(bridge.detach())
    for _ in range(steps):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = -10.0 * torch.sigmoid(bridge)
        loss.backward()
        trainer.optimizer.step()
    return abs(float(bridge.detach()) - before)


def test_bridge_lr_moves_logit_far_more_than_lora_lr(tmp_path):
    slow = _bridge_displacement(tmp_path / "slow", 1e-4)
    fast = _bridge_displacement(tmp_path / "fast", 3e-3)
    assert slow > 0
    assert fast > 10.0 * slow


def test_loop_and_grad_metrics_are_logged(tmp_path):
    trainer = _make_trainer(
        tmp_path / "logged", loops=True, ple_mode="loop_vector", max_steps=3
    )
    trainer.train()

    entries = [entry for entry in trainer.state.log_history if "bridge_w" in entry]
    assert len(entries) == 3
    bridge = trainer.loop_text_model.loop_bridge_logit
    expected_w = float(torch.sigmoid(bridge.detach().float()).item())
    assert entries[-1]["bridge_w"] == pytest.approx(expected_w, abs=1e-6)
    assert entries[-1]["bridge_grad_norm"] is not None
    assert entries[-1]["bridge_grad_norm"] > 0
    assert entries[-1]["loop_ple_norm"] > 0
    assert entries[-1]["grad_norm_total"] is not None
    assert isinstance(entries[-1]["clip_fired"], bool)

    metrics = json.loads(
        (tmp_path / "logged" / "metrics.json").read_text(encoding="utf-8")
    )
    assert len(metrics["training_metrics"]) == 3
    assert metrics["training_metrics"][-1]["bridge_w"] == pytest.approx(
        expected_w, abs=1e-6
    )
    assert metrics["optimizer_steps"] == 3
    assert metrics["final_bridge_w"] == pytest.approx(expected_w, abs=1e-6)
    assert metrics["final_loop_ple_norm"] > 0
    assert metrics["clip_rate"] == pytest.approx(
        metrics["clipped_steps"] / metrics["optimizer_steps"]
    )


def test_unlooped_arm_logs_grad_stats_without_loop_fields(tmp_path):
    trainer = _make_trainer(tmp_path / "oh-logged", loops=False, max_steps=3)
    trainer.train()

    entries = [entry for entry in trainer.state.log_history if "grad_norm_total" in entry]
    assert len(entries) == 3
    assert "bridge_w" not in entries[-1]
    assert "bridge_grad_norm" not in entries[-1]
    assert isinstance(entries[-1]["clip_fired"], bool)

    metrics = json.loads(
        (tmp_path / "oh-logged" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["optimizer_steps"] == 3
    assert metrics["clip_rate"] == pytest.approx(
        metrics["clipped_steps"] / metrics["optimizer_steps"]
    )
    assert "final_bridge_w" not in metrics



def test_set_n_loops_sequence_matches_sampler_with_accumulation(tmp_path, monkeypatch):
    import gemmaloop.train.trainer as trainer_module

    real_set_n_loops = trainer_module.set_n_loops
    calls: list[int] = []

    def spy(model, n):  # noqa: ANN001 - test spy
        calls.append(int(n))
        return real_set_n_loops(model, n)

    monkeypatch.setattr(trainer_module, "set_n_loops", spy)

    trainer = _make_trainer(
        tmp_path / "sequence",
        seed=3,
        loops=True,
        max_steps=4,
        gradient_accumulation_steps=2,
    )
    forwards: list[int | None] = []
    handle = trainer.loop_text_model.register_forward_pre_hook(
        lambda module, args: forwards.append(
            getattr(module, "_gemmaloop_n_loops", None)
        )
    )
    try:
        trainer.train()
    finally:
        handle.remove()

    expected = [sample_n_loops(3, step) for step in range(8)]
    assert calls == expected
    assert forwards == expected


def test_unlooped_trainer_never_samples(tmp_path, monkeypatch):
    import gemmaloop.train.trainer as trainer_module

    calls: list[int] = []
    monkeypatch.setattr(
        trainer_module,
        "set_n_loops",
        lambda model, n: calls.append(int(n)),
    )
    trainer = _make_trainer(tmp_path / "unlooped", loops=False, max_steps=2)
    trainer.train()
    assert calls == []


def test_single_epoch_guard_stops_a_silent_second_epoch(tmp_path):
    from gemmaloop.train.trainer import SingleEpochGuard

    rows = _rows(2)
    trainer = _make_trainer(tmp_path / "repeat", max_steps=3, rows=rows)
    trainer.add_callback(
        SingleEpochGuard(trainer, len(rows), allow_repeat=False)
    )
    with pytest.raises(RuntimeError, match="without --allow-repeat"):
        trainer.train()
    assert trainer.micro_batches_seen == 3

    allowed = _make_trainer(tmp_path / "allowed", max_steps=3, rows=rows)
    allowed.add_callback(SingleEpochGuard(allowed, len(rows), allow_repeat=True))
    allowed.train()
    assert allowed.micro_batches_seen == 3


def test_resume_matches_uninterrupted_and_restores_tokens(tmp_path):
    uninterrupted = _make_trainer(tmp_path / "full", seed=0, loops=True, max_steps=6)
    uninterrupted.train()
    expected = _losses(uninterrupted)[-1]

    # The first run is configured for the same 6-step horizon as the final run
    # so the LR scheduler is identical; a token budget stops it after step 3.
    first = _make_trainer(
        tmp_path / "first",
        seed=0,
        loops=True,
        max_steps=6,
        save_steps=3,
        callbacks=[TokenBudgetCallback(36)],
    )
    first.train()
    checkpoint = tmp_path / "first" / "checkpoint-3"
    assert (checkpoint / "loop_state.safetensors").is_file()
    assert (checkpoint / "loop_config.json").is_file()
    assert (checkpoint / "adapter_model.safetensors").is_file()
    token_state = json.loads((checkpoint / "token_state.json").read_text(encoding="utf-8"))
    assert token_state["tokens_seen"] == first.tokens_seen
    assert token_state["micro_step"] == 3
    assert token_state["micro_batches_seen"] == 3
    assert token_state["optimizer_steps"] == first.state.global_step
    assert token_state["clipped_steps"] == first._clipped_steps

    resumed = _make_trainer(tmp_path / "resumed", seed=0, loops=True, max_steps=6)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    final = _losses(resumed)[-1]
    assert resumed.micro_step == 6
    assert resumed.micro_batches_seen == 6
    assert abs(final - expected) <= 1e-4
    assert resumed.tokens_seen == uninterrupted.tokens_seen
    assert json.loads(
        (tmp_path / "resumed" / "metrics.json").read_text(encoding="utf-8")
    )["tokens_consumed"] == uninterrupted.tokens_seen

    # Counters are cumulative, not reset by the resume.
    assert resumed.state.global_step == 6
    assert resumed._optimizer_steps == resumed.state.global_step
    assert resumed._clipped_steps >= first._clipped_steps
    assert resumed.clip_rate == pytest.approx(
        resumed._clipped_steps / resumed._optimizer_steps
    )

    # The resumed optimizer keeps the loop group and its scheduler base LR.
    resumed_groups = _groups_by_name(resumed)
    assert "bridge" in resumed_groups
    bridge = resumed.loop_text_model.loop_bridge_logit
    assert resumed_groups["bridge"]["params"] == [bridge]
    assert int(resumed.optimizer.state[bridge]["step"]) == resumed.state.global_step
    bridge_index = next(
        index
        for index, group in enumerate(resumed.optimizer.param_groups)
        if group.get("name") == "bridge"
    )
    assert resumed.lr_scheduler.base_lrs[bridge_index] == pytest.approx(3e-3)

    # run.json's counter is the whole run, not just the post-resume window.
    import scripts.train as train_script

    metadata = train_script.build_run_metadata(
        arm="L",
        size="e2b",
        quant="bf16",
        data=tmp_path / "data",
        base_tokens=100,
        target_tokens=100,
        max_length=32,
        seed=0,
        loop=LoopConfig(n_loops=1),
        adapters_chain=[],
        trainer=resumed,
        manifest_sha256="deadbeef",
        manifest_target_tokens=100,
    )
    run_json = json.loads(
        train_script.write_run_json(tmp_path / "resumed", metadata).read_text(
            encoding="utf-8"
        )
    )
    assert run_json["optimizer_steps"] == run_json["global_step"] == 6
    assert run_json["clip_rate"] == pytest.approx(resumed.clip_rate)


def test_token_budget_boundary_and_metrics(tmp_path):
    budget = TokenBudgetCallback(20)
    trainer = _make_trainer(
        tmp_path / "budget",
        max_steps=10,
        gradient_accumulation_steps=2,
        callbacks=[budget],
    )
    trainer.train()

    tokens_per_step = 2 * len(_pattern())
    assert 20 <= trainer.tokens_seen < 20 + tokens_per_step
    assert budget.tokens_seen == trainer.tokens_seen
    assert trainer.state.global_step < 10
    metrics = json.loads(
        (tmp_path / "budget" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["tokens_consumed"] == trainer.tokens_seen


def test_supervised_tokens_per_optimizer_step_are_measured_and_recorded(tmp_path):
    import scripts.train as train_script

    trainer = _make_trainer(
        tmp_path / "supervised",
        max_steps=3,
        gradient_accumulation_steps=2,
    )
    trainer.train()

    expected = 2 * len(_pattern())
    stats = trainer.supervised_tokens_per_step
    assert stats["count"] == trainer._optimizer_steps == 3
    assert stats["min"] == stats["max"] == expected
    assert stats["mean"] == pytest.approx(expected)

    metadata = train_script.build_run_metadata(
        arm="O-H",
        size="e2b",
        quant="bf16",
        data=tmp_path / "data",
        base_tokens=100,
        target_tokens=100,
        max_length=32,
        seed=0,
        loop=None,
        adapters_chain=[],
        trainer=trainer,
        manifest_sha256="deadbeef",
        manifest_target_tokens=100,
    )
    payload = json.loads(
        train_script.write_run_json(
            tmp_path / "supervised", metadata
        ).read_text(encoding="utf-8")
    )
    assert payload["supervised_tokens_per_step"]["mean"] == pytest.approx(expected)
    assert payload["supervised_tokens_per_step"]["min"] == expected
    assert payload["supervised_tokens_per_step"]["max"] == expected


def _save_tiny_adapter(directory, *, seed: int, fill: float):
    model = _tiny_model(seed=seed)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            **TEXT_MODEL_ARGS,
        ),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(fill)
    peft_model.save_pretrained(directory)
    return str(directory)


def test_init_adapter_merges_full_chain(tmp_path):
    import scripts.train as train_script

    first = _save_tiny_adapter(tmp_path / "a1", seed=1, fill=0.1)
    second = _save_tiny_adapter(tmp_path / "a2", seed=2, fill=0.2)
    third = _save_tiny_adapter(tmp_path / "a3", seed=3, fill=0.3)
    (tmp_path / "a3" / "adapters_chain.json").write_text(
        json.dumps([first, second]), encoding="utf-8"
    )

    expected = _tiny_model(seed=42)
    for adapter_dir in (first, second, third):
        expected = PeftModel.from_pretrained(expected, str(adapter_dir)).merge_and_unload()

    target = _tiny_model(seed=42)
    actual, chain = train_script._merge_init_adapter(
        target, target.model, third, "e2b", "bf16"
    )
    assert chain == [first, second, third]
    q_name = "layers.0.self_attn.q_proj.weight"
    torch.testing.assert_close(
        actual.model.get_parameter(q_name).detach(),
        expected.model.get_parameter(q_name).detach(),
    )

    trainer = _make_trainer(tmp_path / "chain", adapters_chain=chain)
    trainer._save_sidecars(tmp_path / "chain_sidecar")
    assert json.loads(
        (tmp_path / "chain_sidecar" / "adapters_chain.json").read_text(encoding="utf-8")
    ) == chain

    with pytest.raises(ValueError):
        train_script._merge_init_adapter(target, target.model, third, "e4b", "nf4")


def test_default_lora_config_matches_global_constraints():
    config = default_lora_config()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == pytest.approx(0.05)
    assert config.target_modules is not None


def test_resolve_max_length_uses_calibration(tmp_path, monkeypatch):
    import scripts.train as train_script

    monkeypatch.setattr(train_script, "artifacts_dir", lambda: tmp_path)
    assert train_script.resolve_max_length("e2b", None) == 1024
    assert train_script.resolve_max_length("e2b", 2048) == 2048

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    path = calibration / "calibration.json"
    # Task 6 writes a scalar probed on E2B: it must not leak to E4B.
    path.write_text(
        json.dumps({"max_nonspill_train_seq_len": 5120}), encoding="utf-8"
    )
    assert train_script.resolve_max_length("e2b", None) == 5120
    assert train_script.resolve_max_length("e4b", None) == 1024

    # A per-size dict is also accepted when present.
    path.write_text(
        json.dumps({"max_nonspill_train_seq_len": {"e2b": 6144, "e4b": 3072}}),
        encoding="utf-8",
    )
    assert train_script.resolve_max_length("e2b", None) == 6144
    assert train_script.resolve_max_length("e4b", None) == 3072


def test_run_metadata_records_manifest_budget_and_final_step(tmp_path):
    import scripts.train as train_script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"sha256": "deadbeef", "target_tokens": 20}),
        encoding="utf-8",
    )

    trainer = _make_trainer(
        tmp_path / "run",
        max_steps=10,
        gradient_accumulation_steps=2,
        callbacks=[TokenBudgetCallback(20)],
    )
    trainer.train()
    trainer.save_state()

    sha, manifest_target = train_script.read_data_manifest(data_dir)
    metadata = train_script.build_run_metadata(
        arm="O-H",
        size="e2b",
        quant="bf16",
        data=data_dir,
        base_tokens=20,
        target_tokens=20,
        max_length=32,
        seed=3,
        loop=None,
        adapters_chain=[],
        trainer=trainer,
        manifest_sha256=sha,
        manifest_target_tokens=manifest_target,
        max_steps=5,
        tokens_per_step=28.0,
        mean_tokens_per_row=14.0,
    )
    path = train_script.write_run_json(tmp_path / "run", metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = json.loads(
        (tmp_path / "run" / "trainer_state.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "complete"
    assert payload["data_manifest_sha256"] == "deadbeef"
    assert payload["data_manifest_target_tokens"] == 20
    assert payload["target_tokens"] == 20
    assert payload["tokens_consumed"] == trainer.tokens_seen
    assert payload["tokens_consumed"] >= payload["target_tokens"]
    assert payload["max_steps"] == 5
    assert payload["tokens_per_step"] == pytest.approx(28.0)
    assert payload["mean_tokens_per_row"] == pytest.approx(14.0)
    assert payload["global_step"] == trainer.state.global_step
    assert payload["global_step"] == state["global_step"]
    assert payload["clip_rate"] == pytest.approx(trainer.clip_rate)
    assert payload["optimizer_steps"] == trainer.state.global_step
    assert payload["final_bridge_w"] is None
    assert not list((tmp_path / "run").glob("run.json.tmp.*"))


def test_read_data_manifest_requires_manifest(tmp_path):
    import scripts.train as train_script

    with pytest.raises(FileNotFoundError):
        train_script.read_data_manifest(tmp_path / "missing")


def test_build_sft_config_is_the_task13_recipe(tmp_path):
    import scripts.train as train_script

    config = train_script.build_sft_config(
        tmp_path, max_length=2048, seed=3, num_train_epochs=1.5
    )
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 16
    # Round 7c recipe binding: the registered LoRA LR is 2e-4.
    assert config.learning_rate == pytest.approx(2e-4)
    assert getattr(config.lr_scheduler_type, "value", config.lr_scheduler_type) == "cosine"
    # warmup_ratio=0.03 (transformers 5.x spells it warmup_steps=0.03)
    assert config.get_warmup_steps(1000) == 30
    assert config.get_warmup_steps(101) == 4
    assert config.gradient_checkpointing is True
    assert config.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert config.max_grad_norm == pytest.approx(1.0)
    assert config.max_length == 2048
    assert config.save_steps == 200 and config.save_total_limit == 4
    assert config.save_only_model is False


def test_train_cli_perf_flags_default_to_the_proven_configuration():
    import scripts.train as train_script
    from gemmaloop.train.perf import TrainPerfConfig

    args = train_script._parse_args(
        ["--arm", "L", "--size", "e2b", "--data", "d", "--tokens", "10", "--seed", "0", "--out", "o"]
    )
    assert train_script.perf_config_from_args(args) == TrainPerfConfig()
    args = train_script._parse_args(
        [
            "--arm", "L", "--size", "e2b", "--data", "d", "--tokens", "10", "--seed", "0", "--out", "o",
            "--no-lean-lm-head", "--input-grads", "--no-offload-vision", "--offload-checkpoints",
            "--no-banded-sliding", "--checkpoint-skip-token-calls", "24576",
            "--no-host-performance-cores",
        ]
    )
    perf = train_script.perf_config_from_args(args)
    assert not perf.lean_lm_head and perf.input_grads
    assert not perf.offload_vision and perf.offload_checkpoints
    assert not perf.banded_sliding
    assert perf.checkpoint_skip_token_calls == 24576 and not perf.host_performance_cores
    assert args.vram_cap == pytest.approx(0.92) and args.vram_free_margin_gib == pytest.approx(0.5)
def test_train_budget_refuses_silent_repetition():
    import scripts.train as train_script

    with pytest.raises(SystemExit) as exc:
        train_script.resolve_num_epochs(
            50, 100, max_length=1024, allow_repeat=False
        )
    assert exc.value.code == 2
    assert (
        train_script.resolve_num_epochs(
            100, 100, max_length=1024, allow_repeat=False
        )
        == 1.0
    )
    epochs = train_script.resolve_num_epochs(
        50, 100, max_length=1024, allow_repeat=True
    )
    assert epochs == pytest.approx(2.1)


def test_manifest_trained_tokens_are_used_only_at_matching_max_length(tmp_path):
    import scripts.train as train_script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": "deadbeef",
                "target_tokens": 100,
                "max_length": 1024,
                "trained_tokens_per_source": {"a": 60, "b": 40},
            }
        ),
        encoding="utf-8",
    )
    details = train_script.read_data_manifest_details(data_dir)
    assert details.max_length == 1024
    assert details.trained_tokens == 100
    assert train_script.select_dataset_tokens(details, 1024, 5) == 100
    assert train_script.select_dataset_tokens(details, 2048, 5) == 5


def test_dataset_tokens_counted_with_tiny_tokenizer(tmp_path, monkeypatch):
    import scripts.train as train_script

    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]

    def fake_labels(tokenizer, messages, max_length, tools=None):
        del tokenizer, messages, tools
        ids = list(range(min(7, max_length)))
        labels = [-100] * len(ids)
        labels[-1] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    monkeypatch.setattr(train_script, "assistant_labels", fake_labels)
    dataset = train_script.build_dataset(_tiny_tokenizer(), rows, max_length=32)
    actual = sum(len(record["input_ids"]) for record in dataset)
    assert actual == 7
    manifest = train_script.DataManifest("deadbeef", 100, 32, actual)
    assert train_script.select_dataset_tokens(manifest, 32, 999) == 7



def test_resolve_max_steps_uses_mean_row_tokens():
    import scripts.train as train_script

    max_steps, tokens_per_step = train_script.resolve_max_steps(
        56,
        14,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
    )
    assert tokens_per_step == pytest.approx(28.0)
    assert max_steps == 2

    max_steps, _ = train_script.resolve_max_steps(
        280,
        14,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
    )
    assert max_steps == 10

    # ceil, never floor: 57 tokens still needs a third step.
    max_steps, _ = train_script.resolve_max_steps(
        57,
        14,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
    )
    assert max_steps == 3


def test_manifest_mean_tokens_per_row_used_at_matching_length():
    import scripts.train as train_script

    manifest = train_script.DataManifest(
        sha256="deadbeef",
        target_tokens=100,
        max_length=32,
        trained_tokens=140,
        rows=10,
    )
    assert train_script.mean_trained_tokens_per_row(
        manifest, 32, dataset_tokens=10, dataset_rows=2
    ) == pytest.approx(14.0)
    # A manifest built at another length is not used.
    assert train_script.mean_trained_tokens_per_row(
        manifest, 64, dataset_tokens=10, dataset_rows=2
    ) == pytest.approx(5.0)


def test_cosine_schedule_reaches_end_of_short_and_long_budgets(tmp_path):
    import scripts.train as train_script

    tokens_per_row = len(_pattern())
    grad_accum = 2
    for budget in (56, 280):
        max_steps, _ = train_script.resolve_max_steps(
            budget,
            tokens_per_row,
            gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=1,
        )
        trainer = _make_trainer(
            tmp_path / f"cosine-{budget}",
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
            callbacks=[TokenBudgetCallback(budget)],
            lr_scheduler_type="cosine",
        )
        trainer.train()
        assert trainer.state.global_step == max_steps
        assert trainer.tokens_seen >= budget
        assert trainer.lr_scheduler.get_last_lr()[0] == pytest.approx(
            0.0, abs=1e-6
        )


def _mixed_rows() -> list[dict[str, list[int]]]:
    """Rows with different lengths so the token/step ratio is not constant."""
    lengths = [5, 9, 13, 17, 21, 25, 7, 11]
    rows = []
    for index, length in enumerate(lengths):
        ids = [2] + [10 + ((index + position) % 4) for position in range(length)] + [1]
        rows.append({"input_ids": ids, "labels": list(ids)})
    return rows


def test_token_lr_multiplier_depends_on_tokens_not_steps():
    from gemmaloop.train.trainer import token_lr_multiplier

    budget = 1_000
    # Same token position, different (ignored) step counters.
    assert token_lr_multiplier(500, budget) == token_lr_multiplier(500, budget)
    assert token_lr_multiplier(0, budget) == 0.0
    assert token_lr_multiplier(budget, budget) == 0.0
    assert token_lr_multiplier(0.03 * budget, budget) == pytest.approx(1.0)
    assert token_lr_multiplier(2 * budget, budget) == 0.0
    # Warmup is linear over the first 3% of the budget.
    assert token_lr_multiplier(0.015 * budget, budget) == pytest.approx(0.5)


def test_token_schedule_reaches_zero_at_short_and_long_budgets(tmp_path):
    import scripts.train as train_script

    rows = _mixed_rows()
    min_tokens = min(len(row["input_ids"]) for row in rows)
    max_step_tokens = max(len(row["input_ids"]) for row in rows) * 2
    grad_accum = 2
    for budget in (60, 300):
        max_steps = train_script.resolve_max_steps_bound(
            budget,
            min_tokens,
            gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=1,
        )
        trainer = _make_trainer(
            tmp_path / f"token-schedule-{budget}",
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
            callbacks=[TokenBudgetCallback(budget)],
            rows=rows,
            token_budget=budget,
        )
        trainer.train()
        assert trainer.tokens_seen >= budget
        assert trainer.tokens_seen < budget + max_step_tokens
        assert trainer.state.global_step <= max_steps
        assert trainer.lr_scheduler.get_last_lr()[0] == pytest.approx(
            0.0, abs=1e-6
        )
def test_heldout_loss_sets_explicit_n_and_restores_state():
    model = _tiny_model(seed=0, loops=True)
    model.train()
    ids = torch.tensor([_pattern()], dtype=torch.long)
    batch = {"input_ids": ids, "labels": ids.clone()}
    set_n_loops(model.model, 3)

    loss_n1 = heldout_loss(model, batch, 1)
    assert torch.isfinite(loss_n1)
    assert model.model._gemmaloop_n_loops == 3
    assert model.training is True

    loss_n2 = heldout_loss(model, batch, 2)
    assert torch.isfinite(loss_n2)
    assert model.model._gemmaloop_n_loops == 3
    assert model.training is True
    assert not torch.isclose(loss_n1, loss_n2)

    assert model.model._gemmaloop_n_loops == 3
    model.eval()
    set_n_loops(model.model, 2)
    with torch.no_grad():
        expected = model(**batch).loss
    torch.testing.assert_close(loss_n2, expected)
    assert model.training is False


def test_heldout_loss_walks_real_peft_wrapper():
    from peft import LoraConfig, get_peft_model

    base = _tiny_model(seed=0, loops=True)
    wrapped = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"]),
    )
    assert _loop_text_model(wrapped) is base.model

    ids = torch.tensor([_pattern()], dtype=torch.long)
    batch = {"input_ids": ids, "labels": ids.clone()}
    wrapped.train()
    set_n_loops(base.model, 3)

    for n in (1, 2, 3, 4):
        loss = heldout_loss(wrapped, batch, n)
        assert torch.isfinite(loss)
        assert base.model._gemmaloop_n_loops == 3
        assert wrapped.training is True


def test_heldout_loss_uses_the_lean_head_and_not_full_logits():
    from gemmaloop.train.fused_ce import install_lean_lm_head

    model = _tiny_model(seed=0, loops=True)
    ids = torch.tensor([_pattern()], dtype=torch.long)
    batch = {"input_ids": ids, "labels": ids.clone()}

    with torch.no_grad():
        set_n_loops(model.model, 1)
        expected_n1 = model(**batch).loss
        set_n_loops(model.model, 2)
        expected_n2 = model(**batch).loss

    install_lean_lm_head(model)

    def _no_full_logits(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("heldout_loss must not materialise full logits")

    model.lm_head.forward = _no_full_logits

    torch.testing.assert_close(
        heldout_loss(model, batch, 1), expected_n1, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        heldout_loss(model, batch, 2), expected_n2, rtol=1e-4, atol=1e-4
    )


def test_heldout_loss_supports_unlooped_models():
    from gemmaloop.train.fused_ce import install_lean_lm_head

    model = _tiny_model(seed=0, loops=False)
    ids = torch.tensor([_pattern()], dtype=torch.long)
    batch = {"input_ids": ids, "labels": ids.clone()}
    model.train()
    with torch.no_grad():
        model.eval()
        expected = model(**batch).loss
    model.train()

    torch.testing.assert_close(heldout_loss(model, batch, 2), expected)
    assert model.training is True

    install_lean_lm_head(model)
    model.lm_head.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("full logits")
    )
    torch.testing.assert_close(
        heldout_loss(model, batch, 2), expected, rtol=1e-4, atol=1e-4
    )


def test_gch_bridge_parameter_joins_the_bridge_optimizer_group(tmp_path):
    model = Gemma4ForCausalLM(tiny_config())
    plan = make_layer_plan(tiny_config().to_dict(), block=(5, 7))
    apply_loops(
        model.model,
        plan,
        LoopConfig(
            n_loops=1,
            bridge="gch",
            block=(5, 7),
            adapter_policy="loop_only",
        ),
    )
    trainer = _make_trainer(tmp_path / "gch", loops=True, model=model)
    groups = {group["name"]: group for group in trainer.build_optimizer_groups()}
    bridge_params = groups["bridge"]["params"]
    assert any(parameter is model.model.loop_bridge_g for parameter in bridge_params)
    assert groups["bridge"]["lr"] == pytest.approx(3e-3)
    assert groups["bridge"]["weight_decay"] == 0.0


def test_run_metadata_records_lora_target_set_for_unlooped_arms(tmp_path):
    from types import SimpleNamespace

    import scripts.train as train_script

    fake_trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=3),
        tokens_seen=120,
        micro_batches_seen=2,
        args=SimpleNamespace(learning_rate=1e-4),
        _optimizer_steps=3,
        _clipped_steps=0,
        clip_rate=0.0,
        _last_grad_norm_total=None,
        _metric_history=[],
        loop_metrics=None,
    )
    metadata = train_script.build_run_metadata(
        arm="O-H",
        size="e2b",
        quant="bf16",
        data=tmp_path / "data",
        base_tokens=100,
        target_tokens=100,
        max_length=32,
        seed=0,
        loop=None,
        adapters_chain=[],
        trainer=fake_trainer,
        manifest_sha256="deadbeef",
        manifest_target_tokens=100,
        lora_target_layers="loop_block",
        lora_layers=[25, 26, 27, 28, 29],
    )
    assert metadata["lora_target_layers"] == "loop_block"
    assert metadata["lora_layers"] == [25, 26, 27, 28, 29]
    assert metadata["loop_block"] is None


def test_trainer_retains_exactly_two_checkpoints(tmp_path):
    """CHECKPOINT_KEEP=1 (or a dropped rotation) must fail this test."""
    trainer = _make_trainer(
        tmp_path / "keep", seed=0, loops=True, max_steps=3, save_steps=1
    )
    trainer.train()
    checkpoints = sorted(
        path.name for path in (tmp_path / "keep").glob("checkpoint-*")
        if path.is_dir() and (path / "COMPLETE.json").is_file()
    )
    assert len(checkpoints) == 2
    assert checkpoints == ["checkpoint-2", "checkpoint-3"]


def test_throughput_baseline_is_persisted_in_the_checkpoint(tmp_path):
    trainer = _make_trainer(
        tmp_path / "guard-state", seed=0, loops=True, max_steps=2, save_steps=1
    )
    trainer._throughput_baseline = 123.5
    trainer._throughput_low_windows = 2
    trainer.train()
    marker = (tmp_path / "guard-state" / "checkpoint-1" / "COMPLETE.json")
    import json

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["throughput_guard"]["baseline_tok_s"] == pytest.approx(123.5)
    assert payload["throughput_guard"]["low_windows"] == 2
