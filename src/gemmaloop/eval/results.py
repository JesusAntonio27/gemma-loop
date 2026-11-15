"""Evaluation result files.

Every evaluation run writes a single :class:`RunResult` as JSON.  The schema
keeps one entry per generated sample so that the statistics in
:mod:`gemmaloop.eval.stats` can be computed from the raw records (multiple
samples per item, aggregate accuracy, paired comparisons).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    """Outcome of a single generated sample for one benchmark item.

    ``extracted_answer``/``parse_status`` record what the automatic extractor
    saw (``None`` when the benchmark has no extractor), and ``reasoning`` keeps
    Gemma's thought channel for post-hoc inspection.  ``meta`` carries item
    provenance such as LiveCodeBench's ``post_cut`` flag.  All four are optional
    so result files written before they existed still load.
    """

    item_id: str
    sample_idx: int
    correct: bool
    completion_tokens: int
    finish_reason: str
    extracted_answer: str | None = None
    parse_status: str | None = None
    reasoning: str | None = None
    meta: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    """All samples produced by one (variant, size, loops, benchmark) run.

    ``seed`` identifies the training seed of the arm so multi-seed criteria can
    aggregate matching seeds and pair them.  It defaults to 0 for older files.
    """

    variant: str
    size: str
    n_loops: int
    bench: str
    prereg_tag: str
    seed: int = 0
    # Loop geometry/policy actually executed.  Defaults keep every result file
    # written before these fields existed loadable (None = legacy 15-29 for E2B,
    # bridge/policy unknown, all-language adapter targets).
    loop_block: tuple[int, int] | None = None
    bridge: str | None = None
    adapter_policy: str | None = None
    lora_target_layers: str | None = None
    items: list[ItemResult]
    # Optional run-level annotations (for example whether MATH-Vision used the
    # official scorer).  Defaulting to an empty mapping keeps every result file
    # written before this field existed loadable without migration.
    metadata: dict = Field(default_factory=dict)

    def geometry_key(self) -> tuple:
        """Loop geometry/policy identity used for duplicate checks and pooling.

        Two runs that differ in block, bridge, adapter policy or LoRA target
        set are different experiments and must never be pooled into one chart
        point or rejected as duplicate re-runs.
        """
        block = tuple(self.loop_block) if self.loop_block is not None else None
        return (
            block,
            self.bridge,
            self.adapter_policy,
            self.lora_target_layers,
        )

    def seed_scheme_key(self) -> str:
        """Per-request seed convention recorded in the partial fingerprint."""

        fingerprint = self.metadata.get("fingerprint") or {}
        if int(fingerprint.get("format_version", 1)) < 2:
            return "legacy"
        return str(
            fingerprint.get("seed_scheme", "sha256(seed:item:sample)")
        )

    def engine_key(self) -> str:
        """Generation engine plus seed scheme that produced this run.

        Older result files predate the engine field; they were all written by
        the padded engine, so ``"padded"`` is the safe default.
        """
        return (
            str(self.metadata.get("engine", "padded"))
            + "|"
            + self.seed_scheme_key()
        )

    def result_identity(self, bench: str | None = None) -> tuple:
        """Full identity of a run (variant, size, N, bench, seed, geometry, engine)."""
        return (
            self.variant.upper(),
            self.size.upper(),
            int(self.n_loops),
            self.bench if bench is None else bench,
            int(self.seed),
            *self.geometry_key(),
            self.engine_key(),
        )

    def accuracy(self) -> float:
        """Fraction of correct samples across every item and sample."""
        if not self.items:
            return 0.0
        return sum(item.correct for item in self.items) / len(self.items)

    def per_item_mean(self) -> dict[str, float]:
        """Mean correctness per item, averaging over that item's samples."""
        totals: dict[str, list[bool]] = defaultdict(list)
        for item in self.items:
            totals[item.item_id].append(item.correct)
        return {
            item_id: sum(corrects) / len(corrects)
            for item_id, corrects in totals.items()
        }

    def save(self, path: str | Path) -> None:
        """Write the result as JSON, creating parent directories if needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        """Read a result previously written by :meth:`save`."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
