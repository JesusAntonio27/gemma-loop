"""End-to-end CPU teacher path: shards -> collator -> trainer -> loss.

The reference values are computed directly from the full teacher and student
softmaxes, independently of the shard writer, the collator's alignment and the
chunked kernels.  This is the regression that catches a second shift, a wrong
assistant mask or a broken non-dial heal loss.
"""

from __future__ import annotations

import copy
import importlib.util
import sys

import pytest
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from transformers import Gemma4ForCausalLM
from trl import SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from gemmaloop.data.teacher_topk import (
    DEFAULT_MASK_TOKEN,
    TeacherIndex,
    TeacherTopKCollator,
    teacher_spec,
)
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.train.trainer import LoopSFTTrainer
from tests.model.tiny import tiny_config, tiny_text_model
from tests.train.test_trainer_tiny import _tiny_tokenizer

class _FakeVocabTokenizer:
    def __init__(self):
        self.chat_template = "tiny-template"

    def get_vocab(self):
        return {f"tok{i}": i for i in range(256)}


K = 8
MASK_ID = 7
IGNORE = -100


def _load_precompute_script():
    script = Path(__file__).resolve().parents[2] / "scripts" / "precompute_teacher_topk.py"
    spec = importlib.util.spec_from_file_location("precompute_teacher_topk_e2e", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ids_labels(row_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    offset = int(row_id) % 32
    input_ids = torch.tensor([[5 + offset, 6, 7, 8, 9, 10, 11]])
    labels = torch.tensor([[-100, -100, 7, 8, -100, 10, 11]])
    return input_ids, labels


def _student(looped: bool, weight: torch.Tensor) -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(tiny_config())
    if looped:
        apply_loops(
            model.model,
            make_layer_plan(model.config.to_dict()),
            LoopConfig(
                n_loops=1,
                bridge="gch",
                ple_mode="shared",
                adapter_policy="loop_only",
                lora_target_layers="loop_block",
            ),
        )
    with torch.no_grad():
        model.lm_head.weight.copy_(weight)
    model.lm_head.weight.requires_grad_(False)
    return model


def _teacher_and_shards(tmp_path, count: int = 1):
    module = _load_precompute_script()
    torch.manual_seed(0)
    student_weight = Gemma4ForCausalLM(tiny_config()).lm_head.weight.detach().clone()
    torch.manual_seed(1)
    teacher_text = tiny_text_model(seed=1)
    loaded = SimpleNamespace(
        spec=teacher_spec("e4b"),
        model=None,
        tokenizer=_FakeVocabTokenizer(),
        text_model=teacher_text,
        weight=student_weight,
        softcap=None,
        hidden=lambda input_ids: teacher_text(input_ids).last_hidden_state,
    )
    prepared = []
    for row_id in range(count):
        input_ids, labels = _ids_labels(row_id)
        prepared.append(
            module.PreparedRow(
                row_id=row_id,
                input_ids=input_ids[0].tolist(),
                labels=labels[0].tolist(),
            )
        )
    module.precompute_prepared(
        prepared,
        loaded,
        out=tmp_path / "teacher",
        data_manifest_sha256="a" * 64,
        top_k=K,
        max_length=16,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=MASK_ID,
        shard_rows=1,
        token_chunk=3,
        vocab_chunk=5,
        expected_lm_head_rows=None,
    )
    index = TeacherIndex.from_directory(
        tmp_path / "teacher",
        data_manifest_sha256="a" * 64,
        expect_top_k=K,
        verify_shards=True,
    )
    return index, student_weight, teacher_text


def _batch(index: TeacherIndex, row_id: int = 0) -> dict:
    input_ids, labels = _ids_labels(row_id)
    collator = TeacherTopKCollator(
        index, DataCollatorForLanguageModeling(pad_token_id=0)
    )
    return collator(
        [
            {
                "row_id": int(row_id),
                "input_ids": input_ids[0].tolist(),
                "labels": labels[0].tolist(),
                "attention_mask": [1] * input_ids.shape[1],
            }
        ]
    )


def _trainer(
    tmp_path,
    model,
    batch,
    *,
    dial: bool,
    heal: bool,
) -> LoopSFTTrainer:
    args = SFTConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        max_length=16,
        disable_tqdm=True,
        warmup_steps=0,
        remove_unused_columns=False,
    )
    return LoopSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(
            [
                {
                    "input_ids": batch["input_ids"][0].tolist(),
                    "labels": batch["labels"][0].tolist(),
                    "attention_mask": batch["attention_mask"][0].tolist(),
                    "row_id": 0,
                }
            ]
        ),
        processing_class=_tiny_tokenizer(),
        loop_text_model=model.model,
        loop_enabled=dial,
        dial_enabled=dial,
        dial_horizon_probs=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        heal_teacher=heal,
        data_collator=None,
        callbacks=[],
    )


def _teacher_log_probs(teacher_text, weight, input_ids, positions):
    """Full-softmax teacher log-probs for target positions (state ``pos-1``)."""

    with torch.no_grad():
        hidden = teacher_text(input_ids).last_hidden_state[0]
        logits = hidden @ weight.t()
        logits[:, MASK_ID] = float("-inf")
        log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[torch.tensor(positions) - 1]


def _naive_teacher_kl(teacher_log_probs, student_hidden, weight):
    """Full-softmax q; teacher top-k plus its remaining tail bucket."""

    log_q = torch.log_softmax(student_hidden @ weight.t(), dim=-1)
    top_log_p, top_indices = torch.topk(teacher_log_probs, K, dim=-1)
    p = top_log_p.exp()
    q_at_top = log_q.gather(1, top_indices)
    q_mass = q_at_top.exp().sum(dim=1)
    tail = (1.0 - p.sum(dim=1)).clamp(min=1e-12)
    tail_log_q = torch.log1p(-q_mass.clamp(max=1.0 - 1e-7))
    return (p * (top_log_p - q_at_top)).sum(dim=1) + tail * (
        torch.log(tail) - tail_log_q
    )


def _positions(labels: torch.Tensor) -> list[int]:
    return [
        index
        for index, label in enumerate(labels[0].tolist())
        if label != IGNORE and index > 0
    ]


def test_dial_loss_matches_the_full_softmax_reference(tmp_path):
    index, weight, teacher_text = _teacher_and_shards(tmp_path)
    batch = _batch(index)
    model = _student(looped=True, weight=weight)
    trainer = _trainer(tmp_path / "dial", model, batch, dial=True, heal=False)
    model.train()
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)

    input_ids, labels = _ids_labels()
    positions = _positions(labels)
    teacher_log_probs = _teacher_log_probs(
        teacher_text, weight, input_ids, positions
    )
    mask = labels[0, 1:] != IGNORE

    with torch.no_grad():
        set_n_loops(model.model, 4)
        exits = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            exits=[1, 2, 3, 4],
            window=3,
        ).exit_hidden_states
        compact = {
            t: exits[t][0, :-1][mask] for t in (1, 2, 3, 4)
        }
        ell = {}
        ell[1] = _naive_teacher_kl(
            teacher_log_probs, compact[1], weight
        ).mean()
        log_parent = torch.log_softmax(compact[1] @ weight.t(), dim=-1)
        for exit_index in (2, 3, 4):
            teacher_kl = _naive_teacher_kl(
                teacher_log_probs, compact[exit_index], weight
            ).mean()
            log_exit = torch.log_softmax(
                compact[exit_index] @ weight.t(), dim=-1
            )
            parent_kl = (
                log_parent.exp() * (log_parent - log_exit)
            ).sum(dim=1).mean()
            ell[exit_index] = teacher_kl + 0.1 * parent_kl
        hinge = sum(
            torch.relu(ell[t] - ell[t - 1].detach()) for t in (2, 3, 4)
        )
        expected = torch.stack([ell[2], ell[3], ell[4]]).mean() + hinge

    torch.testing.assert_close(loss, expected, atol=2e-4, rtol=2e-4)
    assert outputs["dial"].metrics["dial_tokens"] == int(mask.sum())
    assert trainer.dial_teacher_mode == "teacher"


def test_non_dial_heal_loss_matches_the_full_softmax_reference(tmp_path):
    index, weight, teacher_text = _teacher_and_shards(tmp_path)
    batch = _batch(index)
    model = _student(looped=False, weight=weight)
    trainer = _trainer(tmp_path / "heal", model, batch, dial=False, heal=True)
    model.train()
    loss = trainer.compute_loss(model, batch)

    input_ids, labels = _ids_labels()
    positions = _positions(labels)
    teacher_log_probs = _teacher_log_probs(
        teacher_text, weight, input_ids, positions
    )
    mask = labels[0, 1:] != IGNORE
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state[0]
        compact = hidden[:-1][mask]
        # The registered O-H objective carries the same parent anchor as L.
        parent_log = torch.log_softmax(compact @ weight.t(), dim=-1)
        student_log = torch.log_softmax(compact @ weight.t(), dim=-1)
        parent_kl = (
            parent_log.exp() * (parent_log - student_log)
        ).sum(dim=1).mean()
        expected = (
            _naive_teacher_kl(teacher_log_probs, compact, weight).mean()
            + 0.1 * parent_kl
        )

    torch.testing.assert_close(loss, expected, atol=2e-4, rtol=2e-4)
    assert trainer.dial_teacher_mode == "teacher"
    assert trainer._last_heal_metrics["heal_tokens"] == int(mask.sum())


def test_heal_loss_includes_a_nonzero_parent_anchor(tmp_path, monkeypatch):
    index, weight, teacher_text = _teacher_and_shards(tmp_path)
    batch = _batch(index)
    model = _student(looped=False, weight=weight)
    trainer = _trainer(tmp_path / "heal-anchor", model, batch, dial=False, heal=True)
    model.train()
    torch.manual_seed(99)
    parent = torch.randn(1, batch["input_ids"].shape[1], model.config.hidden_size)
    monkeypatch.setattr(
        trainer, "_parent_hidden_for_heal", lambda model, inputs: parent
    )
    loss = trainer.compute_loss(model, batch)

    input_ids, labels = _ids_labels()
    positions = _positions(labels)
    teacher_log_probs = _teacher_log_probs(teacher_text, weight, input_ids, positions)
    mask = labels[0, 1:] != IGNORE
    with torch.no_grad():
        student_hidden = model.model(input_ids=input_ids).last_hidden_state[0]
        parent_log = torch.log_softmax(parent[0, :-1][mask] @ weight.t(), dim=-1)
        student_log = torch.log_softmax(
            student_hidden[:-1][mask] @ weight.t(), dim=-1
        )
        parent_kl = (
            parent_log.exp() * (parent_log - student_log)
        ).sum(dim=1).mean()
        expected = (
            _naive_teacher_kl(teacher_log_probs, student_hidden[:-1][mask], weight).mean()
            + 0.1 * parent_kl
        )
    torch.testing.assert_close(loss, expected, atol=2e-4, rtol=2e-4)


def test_ga4_and_ga1_give_the_same_dial_gradient(tmp_path):
    index, weight, _ = _teacher_and_shards(tmp_path, count=4)
    batches = [_batch(index, row_id) for row_id in range(4)]
    total_items = int(
        sum((batch["labels"][:, 1:] != IGNORE).sum().item() for batch in batches)
    )

    def micro_gradient(batch, *, normalized: bool) -> torch.Tensor:
        # A fresh model per micro-step keeps the two gradient accumulations
        # independent (no graph is reused across micro-steps).
        model = _student(looped=True, weight=weight)
        trainer = _trainer(
            tmp_path / f"ga-{normalized}-{len(batch['input_ids'])}",
            model,
            batch,
            dial=True,
            heal=False,
        )
        model.train()
        loss = trainer.compute_loss(
            model,
            batch,
            num_items_in_batch=total_items if normalized else None,
        )
        if not normalized:
            count = int((batch["labels"][:, 1:] != IGNORE).sum().item())
            loss = loss * (float(count) / float(total_items))
        parameter = model.model.layers[7].mlp.down_proj.weight
        return torch.autograd.grad(loss, parameter)[0]

    accumulated = sum(
        micro_gradient(batch, normalized=True) for batch in batches
    )
    reference = sum(
        micro_gradient(batch, normalized=False) for batch in batches
    )
    torch.testing.assert_close(accumulated, reference, atol=1e-6, rtol=1e-6)


def test_real_parent_hidden_uses_the_adapter_free_model(tmp_path):
    from peft import LoraConfig, get_peft_model

    index, weight, _ = _teacher_and_shards(tmp_path)
    batch = _batch(index)
    student = _student(looped=False, weight=weight)
    peft_model = get_peft_model(
        student,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, 0.1)
    trainer = _trainer(tmp_path / "parent", peft_model, batch, dial=False, heal=True)
    from gemmaloop.train.trainer import _loop_text_model

    text_model = _loop_text_model(peft_model)
    with torch.no_grad():
        with peft_model.disable_adapter():
            reference = text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).last_hidden_state
        adapted = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).last_hidden_state
    parent = trainer._parent_hidden_for_heal(peft_model, batch)
    torch.testing.assert_close(parent, reference, atol=0, rtol=0)
    assert not torch.allclose(parent, adapted)


def test_parent_requires_disable_adapter_when_lora_is_trainable():
    import torch.nn as nn

    from gemmaloop.train.trainer import LoopSFTTrainer

    class _LoraModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B = nn.Parameter(torch.ones(1))

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Linear(1, 1)
            self.lora_B = nn.Parameter(torch.ones(1))

    trainer = object.__new__(LoopSFTTrainer)
    with pytest.raises(RuntimeError, match="disable_adapter"):
        trainer._parent_hidden_for_heal(
            _Wrapper(), {"input_ids": torch.zeros(1, 1, dtype=torch.long)}
        )
