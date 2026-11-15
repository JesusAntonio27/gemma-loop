"""Teacher-aware data collator for the dial / heal objective.

T-heal writes per-position teacher top-k shards; the batch contract consumed by
:mod:`gemmaloop.train.dial_objective` is fixed here and must stay stable:

``teacher_topk_probs``   ``[B, S, K]`` float16 (or any float dtype)
``teacher_topk_indices`` ``[B, S, K]`` int32
``teacher_tail_prob``    ``[B, S]``    float32

The arrays are **unshifted** and aligned 1:1 with ``input_ids`` / ``labels``.
:func:`gemmaloop.train.dial_objective._teacher_for_sequence` applies the single
left shift to the next-token prediction positions; a collator must not shift
again.

The TRL ``DataCollatorForLanguageModeling`` (and ``transformers``' default
collator) only return the signature columns, so teacher columns are dropped
even with ``remove_unused_columns=False``.  This wrapper delegates the language
model fields to any collator and pads/returns the teacher columns itself.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "TEACHER_FIELDS",
    "TEACHER_METADATA_FIELDS",
    "TeacherDataCollator",
    "has_teacher_fields",
]

TEACHER_FIELDS = (
    "teacher_topk_probs",
    "teacher_topk_indices",
    "teacher_tail_prob",
)
TEACHER_METADATA_FIELDS = (
    "teacher_manifest_sha256",
    "teacher_id",
    "teacher_revision",
)


def has_teacher_fields(example: dict[str, Any]) -> bool:
    """True when ``example`` carries the full teacher triple."""
    return all(field in example and example[field] is not None for field in TEACHER_FIELDS)


class TeacherDataCollator:
    """Pad the LM fields with ``base_collator`` and keep the teacher columns."""

    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator
        signature = list(getattr(base_collator, "__signature_columns__", None) or [])
        for field in (*TEACHER_FIELDS, *TEACHER_METADATA_FIELDS):
            if field not in signature:
                signature.append(field)
        self.__signature_columns__ = signature

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(examples)
        present = [any(field in example for example in examples) for field in TEACHER_FIELDS]
        if not any(present):
            self._copy_metadata(examples, batch)
            return batch
        if not all(present):
            raise ValueError(
                "teacher columns must be all present or all absent; got "
                + ", ".join(
                    field for field, found in zip(TEACHER_FIELDS, present) if found
                )
            )

        probs, indices, tails = [], [], []
        for example in examples:
            if not has_teacher_fields(example):
                raise ValueError("every example in a teacher batch needs all teacher fields")
            sequence_length = len(example["input_ids"])
            p = torch.as_tensor(example["teacher_topk_probs"]).to(torch.float16)
            i = torch.as_tensor(example["teacher_topk_indices"]).to(torch.int32)
            t = torch.as_tensor(example["teacher_tail_prob"]).to(torch.float32)
            if p.ndim != 2 or i.shape != p.shape:
                raise ValueError(
                    "teacher_topk_probs and teacher_topk_indices must both be "
                    f"[S, K], got {tuple(p.shape)} and {tuple(i.shape)}"
                )
            if t.shape != (sequence_length,):
                raise ValueError(
                    f"teacher_tail_prob must be [{sequence_length}], got {tuple(t.shape)}"
                )
            if p.shape[0] != sequence_length:
                raise ValueError(
                    "teacher rows must align with input_ids: "
                    f"{p.shape[0]} vs {sequence_length}"
                )
            if (i < 0).any():
                raise ValueError("teacher indices must be non-negative")
            probs.append(p)
            indices.append(i)
            tails.append(t)

        batch["teacher_topk_probs"] = torch.nn.utils.rnn.pad_sequence(
            probs, batch_first=True, padding_value=0.0
        )
        batch["teacher_topk_indices"] = torch.nn.utils.rnn.pad_sequence(
            indices, batch_first=True, padding_value=0
        )
        batch["teacher_tail_prob"] = torch.nn.utils.rnn.pad_sequence(
            tails, batch_first=True, padding_value=0.0
        )
        self._copy_metadata(examples, batch)
        return batch

    @staticmethod
    def _copy_metadata(
        examples: list[dict[str, Any]], batch: dict[str, Any]
    ) -> None:
        for field in TEACHER_METADATA_FIELDS:
            if any(field in example for example in examples):
                batch[field] = [example.get(field) for example in examples]
