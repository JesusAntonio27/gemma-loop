"""Evaluation corpora used to decontaminate the Stage 0 pool.

The registered targets are LiveCodeBench ``release_v6``, GPQA Diamond, AIME
2026, MMLU-Pro, MMMU-Pro, MATH-Vision, SpatialEval, CV-Bench 2D and the
creative briefs.  The loaders mirror ``scripts/prepare_heal_data.py`` so the
Stage 0 pool and the healing mix reject contamination with the same index.

Tests inject ``texts_by_name``; no network access is performed at import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from gemmaloop.data.decontam import EvalIndex
from gemmaloop.eval.benchmarks import aime, cvbench, gpqa, lcb, mathvision, mmlupro, mmmupro, spatialeval

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BRIEFS = REPO_ROOT / "data" / "creative" / "briefs_eval.jsonl"


def _stream_texts(
    hf_id: str,
    fields: tuple[str, ...],
    *,
    config: str | None = None,
    split: str = "test",
    revision: str | None = None,
    limit: int | None = None,
    columns: bool = True,
) -> Iterator[str]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if config is not None:
        kwargs["name"] = config
    if revision is not None:
        kwargs["revision"] = revision

    def dataset_iter():
        if columns:
            try:
                stream = load_dataset(
                    hf_id,
                    split=split,
                    streaming=True,
                    columns=list(fields),
                    **kwargs,
                )
                return iter(stream)
            except (TypeError, ValueError):
                pass
        return iter(load_dataset(hf_id, split=split, streaming=True, **kwargs))

    for index, row in enumerate(dataset_iter()):
        if limit is not None and index >= limit:
            break
        for field in fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                yield str(value)
                break


def _lcb_texts(limit: int | None = None) -> Iterator[str]:
    from huggingface_hub import hf_hub_download

    count = 0
    for filename in lcb.RELEASE_FILES[lcb.RELEASE]:
        path = hf_hub_download(
            repo_id=lcb.DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=lcb.REVISION,
        )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = str(row.get("question_content") or "").strip()
                starter = str(row.get("starter_code") or "").strip()
                text = question
                if starter:
                    text = f"{text}\n\nStarter code:\n{starter}"
                if text.strip():
                    yield text
                    count += 1
                    if limit is not None and count >= limit:
                        return


def _collect_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _collect_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_strings(item)


def jsonl_texts(path: Path) -> Iterator[str]:
    preferred = ("prompt", "brief", "text", "question", "instruction", "content")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, str):
                yield record
                continue
            if not isinstance(record, dict):
                yield from _collect_strings(record)
                continue
            found = False
            for key in preferred:
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    yield value
                    found = True
            if not found:
                yield from _collect_strings(record)


@dataclass(frozen=True)
class EvalSpec:
    name: str
    loader: Callable[[int | None], Iterable[str]]
    expected: int
    hf_id: str
    split: str
    config: str | None = None
    revision: str | None = None
    mandatory: bool = True


EVAL_SPECS: tuple[EvalSpec, ...] = (
    EvalSpec(
        "livecodebench_v6",
        _lcb_texts,
        1055,
        lcb.DATASET_ID,
        lcb.RELEASE,
        revision=lcb.REVISION,
    ),
    EvalSpec(
        "gpqa_diamond",
        lambda limit: _stream_texts(
            gpqa.DATASET_ID,
            ("Question",),
            config=gpqa.CONFIG,
            split=gpqa.SPLIT,
            revision=gpqa.REVISION,
            limit=limit,
            columns=False,
        ),
        198,
        gpqa.DATASET_ID,
        gpqa.SPLIT,
        config=gpqa.CONFIG,
        revision=gpqa.REVISION,
    ),
    EvalSpec(
        "aime26",
        lambda limit: _stream_texts(
            aime.DATASET_ID,
            ("problem",),
            split=aime.SPLIT,
            revision=aime.REVISION,
            limit=limit,
            columns=False,
        ),
        30,
        aime.DATASET_ID,
        aime.SPLIT,
        revision=aime.REVISION,
    ),
    EvalSpec(
        "mmlu_pro",
        lambda limit: _stream_texts(
            mmlupro.DATASET_ID,
            ("question",),
            split=mmlupro.SPLIT,
            revision=mmlupro.REVISION,
            limit=limit,
        ),
        12032,
        mmlupro.DATASET_ID,
        mmlupro.SPLIT,
        revision=mmlupro.REVISION,
    ),
    EvalSpec(
        "mmmu_pro",
        lambda limit: _stream_texts(
            mmmupro.DATASET_ID,
            ("question",),
            config=mmmupro.STANDARD_CONFIG,
            split=mmmupro.SPLIT,
            revision=mmmupro.REVISION,
            limit=limit,
        ),
        1730,
        mmmupro.DATASET_ID,
        mmmupro.SPLIT,
        config=mmmupro.STANDARD_CONFIG,
        revision=mmmupro.REVISION,
    ),
    EvalSpec(
        "math_vision",
        lambda limit: _stream_texts(
            mathvision.DATASET_ID,
            ("question",),
            split=mathvision.SPLIT,
            revision=mathvision.REVISION,
            limit=limit,
        ),
        3040,
        mathvision.DATASET_ID,
        mathvision.SPLIT,
        revision=mathvision.REVISION,
    ),
    EvalSpec(
        "spatial_eval_tqa",
        lambda limit: _stream_texts(
            spatialeval.DATASET_ID,
            ("text",),
            config="tqa",
            split=spatialeval.SPLIT,
            revision=spatialeval.REVISION,
            limit=limit,
        ),
        4635,
        spatialeval.DATASET_ID,
        spatialeval.SPLIT,
        config="tqa",
        revision=spatialeval.REVISION,
    ),
    EvalSpec(
        "cv_bench_2d",
        lambda limit: _stream_texts(
            cvbench.DATASET_ID,
            ("prompt", "question"),
            config=cvbench.CONFIG,
            split=cvbench.SPLIT,
            revision=cvbench.REVISION,
            limit=limit,
        ),
        1438,
        cvbench.DATASET_ID,
        cvbench.SPLIT,
        config=cvbench.CONFIG,
        revision=cvbench.REVISION,
    ),
)


def gather_eval_texts(
    *,
    eval_limit: int | None = None,
    extra_eval_files: Iterable[Path] = (),
    allow_missing: bool = False,
    specs: Iterable[EvalSpec] = EVAL_SPECS,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load every registered evaluation corpus and any extra JSONL files."""

    texts: list[str] = []
    statuses: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for spec in specs:
        entry: dict[str, Any] = {
            "status": "ok",
            "texts": 0,
            "mandatory": bool(spec.mandatory),
            "hf_id": spec.hf_id,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "expected": spec.expected,
        }
        try:
            part = list(spec.loader(eval_limit))
        except Exception as exc:  # pragma: no cover - network dependent
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            statuses[spec.name] = entry
            if spec.mandatory:
                failures.append(f"{spec.name}: {entry['error']}")
            continue

        required = spec.expected if eval_limit is None else min(spec.expected, eval_limit)
        entry["texts"] = len(part)
        if len(part) == 0:
            entry["status"] = "failed"
            entry["error"] = "loader returned no texts"
        elif len(part) < required:
            entry["status"] = "short"
            entry["error"] = f"expected at least {required} texts, got {len(part)}"
        if entry["status"] != "ok" and spec.mandatory:
            failures.append(f"{spec.name}: {entry['error']}")
        statuses[spec.name] = entry
        texts.extend(part)

    for path in extra_eval_files:
        path = Path(path)
        key = f"file:{path.name}"
        entry = {"status": "ok", "texts": 0, "mandatory": False, "path": str(path)}
        if not path.exists():
            entry["status"] = "missing"
            entry["error"] = "file not found"
            statuses[key] = entry
            continue
        try:
            part = list(jsonl_texts(path))
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            statuses[key] = entry
            continue
        entry["texts"] = len(part)
        if not part:
            entry["status"] = "failed"
            entry["error"] = "file produced no texts"
        statuses[key] = entry
        texts.extend(part)

    if failures and not allow_missing:
        raise RuntimeError(
            "mandatory evaluation index incomplete: "
            + "; ".join(failures)
            + ". Pass allow_missing=True only for smoke runs."
        )
    return texts, statuses


def build_eval_index(
    texts: Iterable[str], tokenizer=None, *, n: int = 13
) -> EvalIndex:
    """Build the repo's 13-gram + Indel 0.75 index."""

    return EvalIndex.build(texts, n=n, tokenizer=tokenizer)


def default_extra_eval_files() -> list[Path]:
    """Always name the creative briefs so the manifest records their status."""

    return [DEFAULT_BRIEFS]


__all__ = [
    "DEFAULT_BRIEFS",
    "EVAL_SPECS",
    "EvalSpec",
    "build_eval_index",
    "default_extra_eval_files",
    "gather_eval_texts",
    "jsonl_texts",
]
