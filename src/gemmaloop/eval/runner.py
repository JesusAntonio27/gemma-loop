"""Resumable benchmark runner.

The runner turns a benchmark adapter (Task 8) into a :class:`RunResult`.  It
writes every finished batch to a sibling ``*.partial.jsonl`` file, so a machine
freeze loses at most the batch that was in flight.  A resumed run reads that
file and skips the ``(item_id, sample_idx)`` pairs that are already present;
nothing is generated twice and no item is dropped.

Scoring follows the reference protocol (``04-evaluation`` Know-how §6 and
Trampa 1):

* ordinary benchmarks are scored per completion as soon as it is generated;
* a ``finish_reason == "length"`` completion is always wrong, whatever the text
  contains (the thinking trace was truncated);
* LiveCodeBench accumulates the ``code_list`` of every sampled problem and calls
  :func:`gemmaloop.eval.benchmarks.lcb.run_official_evaluator` once at the end;
* MATH-Vision calls :func:`...mathvision.run_official_scorer` for reported
  numbers.  If the eval-only ``.venv-mathv`` is missing it falls back to the
  in-package approximate scorer and marks ``RunResult.metadata["scoring"]`` as
  ``"non-official"``.

The generation engine is selected with ``engine`` (``"padded"`` by default or
``"refill"`` for the round-based scheduler in :mod:`gemmaloop.serve.refill`);
the choice is part of the run fingerprint, so partials from different engines
cannot be mixed.  The refill engine is the registered evaluation engine only
after the Task 5d gate passes before ``analysis-v1``; otherwise the padded
adaptive-batch engine is used.  One engine is used per comparison.

The module also exposes :func:`read_run_config`, the small YAML reader for the
``prereg/prereg.yaml`` schema that Task 10 will freeze::

    sampling:
      temperature: 1.0
      top_p: 0.95
      top_k: 64
      max_new_tokens: 16384
      enable_thinking: true
    benches:
      aime: {limit: 30, samples: 8, seed: 0}
      mmmu_pro_vision: {limit: 300, samples: 1, seed: 0, image_soft_tokens: 1120}

Labels and aliases are resolved by :func:`resolve_bench`, so the CLI accepts
``mmlupro``/``mmlu_pro``, ``gpqa``/``gpqa_diamond``, ``mathvision``,
``mmmu_pro_standard``/``mmmu_pro_vision``, ``spatialeval_vqa``/
``spatialeval_tqa`` and ``cvbench``/``cvbench_2d``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml

from gemmaloop.eval.benchmarks import (
    aime,
    cvbench,
    gpqa,
    lcb,
    mathvision,
    mmlupro,
    mmmupro,
    spatialeval,
)
from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.model.load import (
    MODEL_IDS,
    ModelSpec,
    _resolve_loop_config,
    load_model,
)
from gemmaloop.serve.generate import (
    BATCH_POLICY_VERSION,
    IMAGE_SOFT_TOKEN_CHOICES,
    GenOut,
    Sampling,
    generate_batch,
    max_batch_for,
)
from gemmaloop.serve.refill import generate_batch_refill

DEFAULT_PREREG_TAG = "prereg-v1"
_PARTIAL_SUFFIX = ".partial.jsonl"
_HEADER_TYPE = "run_fingerprint"
_OFFICIAL_KINDS = ("lcb", "mathvision")


class PartialProvenanceError(ValueError, RuntimeError):
    """Raised when an existing partial file belongs to another run config.

    Inherits from both :class:`ValueError` and :class:`RuntimeError` so callers
    that expect either kind of "refused, do not mix" failure catch it.
    """


# ---------------------------------------------------------------------------
# Run configuration (the subset of Task 10's prereg.yaml consumed here)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchConfig:
    """Per-benchmark prereg values."""

    limit: int | None = None
    samples: int = 1
    seed: int = 0
    image_soft_tokens: int = 280
    exploratory: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Sampling plus per-benchmark sizes read from ``prereg.yaml``."""

    sampling: Sampling
    benches: dict[str, BenchConfig]
    tag: str = DEFAULT_PREREG_TAG


def _require(mapping: dict, key: str, where: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"missing required key {where}.{key}") from exc


def read_run_config(path: str | Path) -> RunConfig:
    """Read sampling and per-benchmark settings from a YAML run config.

    Only the keys Task 9 needs are interpreted; unknown keys are ignored so the
    future preregistration can grow without breaking older runners.  ``tag``
    (or its alias ``prereg_tag``) is optional and defaults to ``prereg-v1``.
    """

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    sampling_raw = raw.get("sampling")
    if not isinstance(sampling_raw, dict):
        raise ValueError(f"{path}: missing 'sampling' mapping")

    sampling = Sampling(
        temperature=float(_require(sampling_raw, "temperature", "sampling")),
        top_p=float(_require(sampling_raw, "top_p", "sampling")),
        top_k=int(_require(sampling_raw, "top_k", "sampling")),
        max_new_tokens=int(_require(sampling_raw, "max_new_tokens", "sampling")),
        enable_thinking=bool(
            _require(sampling_raw, "enable_thinking", "sampling")
        ),
    )
    if sampling.max_new_tokens < 1:
        raise ValueError("sampling.max_new_tokens must be at least 1")

    benches_raw = raw.get("benches")
    if not isinstance(benches_raw, dict):
        raise ValueError(f"{path}: missing 'benches' mapping")

    benches: dict[str, BenchConfig] = {}
    for name, values in benches_raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path}: bench {name!r} must be a mapping")
        limit = values.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError(f"{path}: bench {name!r} has a negative limit")
        samples = int(values.get("samples", 1))
        if samples < 1:
            raise ValueError(f"{path}: bench {name!r} needs samples >= 1")
        image_soft_tokens = int(values.get("image_soft_tokens", 280))
        if image_soft_tokens not in IMAGE_SOFT_TOKEN_CHOICES:
            raise ValueError(
                f"{path}: bench {name!r} image_soft_tokens must be one of "
                f"{IMAGE_SOFT_TOKEN_CHOICES}, got {image_soft_tokens}"
            )
        benches[str(name)] = BenchConfig(
            limit=limit,
            samples=samples,
            seed=int(values.get("seed", 0)),
            image_soft_tokens=image_soft_tokens,
            exploratory=bool(values.get("exploratory", False)),
        )

    tag = str(raw.get("tag", raw.get("prereg_tag", DEFAULT_PREREG_TAG)))
    return RunConfig(sampling=sampling, benches=benches, tag=tag)


# ---------------------------------------------------------------------------
# Benchmark resolution / loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedBench:
    module: Any
    load_kwargs: dict = field(default_factory=dict)
    kind: str = "score"


def _key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def resolve_bench(bench: str) -> ResolvedBench:
    """Map a CLI/prereg benchmark name to its adapter module.

    The aliases cover the names used by the plan, the model card and the
    lmms-eval task names, so a preregistration can spell a benchmark naturally
    without the runner needing a second mapping.
    """

    key = _key(bench)
    if key in ("lcb", "livecodebench", "live_code_bench"):
        return ResolvedBench(lcb, {"post_cut": True}, kind="lcb")
    if key in ("lcb_google", "lcb_full_v6", "livecodebench_google"):
        return ResolvedBench(lcb, {"post_cut": None}, kind="lcb")
    if key in ("aime", "aime26", "aime_2026"):
        return ResolvedBench(aime)
    if key in ("gpqa", "gpqa_diamond", "gpqa_d"):
        return ResolvedBench(gpqa)
    if key in ("mmlupro", "mmlu_pro"):
        return ResolvedBench(mmlupro)
    if key in ("mathvision", "math_vision", "mathvision_test"):
        return ResolvedBench(mathvision, kind="mathvision")
    if key in ("mmmupro", "mmmu_pro", "mmmu_pro_standard", "mmmu_pro_10c"):
        return ResolvedBench(mmmupro, {"config": mmmupro.STANDARD_CONFIG})
    if key in ("mmmupro_vision", "mmmu_pro_vision"):
        return ResolvedBench(mmmupro, {"config": mmmupro.VISION_CONFIG})
    if key in ("spatialeval", "spatialeval_vqa", "spatial_eval", "spatial_eval_vqa"):
        return ResolvedBench(spatialeval, {"modality": "vqa"})
    if key in ("spatialeval_tqa", "spatial_eval_tqa"):
        return ResolvedBench(spatialeval, {"modality": "tqa"})
    if key in ("cvbench", "cv_bench", "cvbench_2d", "cv_bench_2d"):
        return ResolvedBench(cvbench)
    raise ValueError(
        f"unknown benchmark {bench!r}; expected one of "
        "lcb, lcb_google, aime, gpqa, mmlupro, mathvision, "
        "mmmu_pro_standard, mmmu_pro_vision, spatialeval_vqa, "
        "spatialeval_tqa, cvbench"
    )


def load_items_for_bench(bench: str, limit: int | None, seed: int) -> list:
    """Load one benchmark's items exactly once, with its adapter-specific kwargs."""

    resolved = resolve_bench(bench)
    return resolved.module.load_items(limit=limit, seed=seed, **resolved.load_kwargs)


# ---------------------------------------------------------------------------
# Generator plumbing
# ---------------------------------------------------------------------------
def _call_gen(
    gen_fn: Callable,
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
    image_soft_tokens: int,
    request_ids: list[str] | None = None,
    max_batch: int | None = None,
    on_complete: Callable[[int, GenOut], None] | None = None,
    heartbeat: Callable[[dict], None] | None = None,
) -> list[GenOut]:
    """Call ``gen_fn`` with only the keyword arguments it accepts.

    Real callers use :func:`generate_batch`; unit tests inject a small fake that
    often ignores the extra keywords.  Filtering keeps both working without
    silently swallowing an unexpected ``TypeError`` raised inside the function.
    """

    kwargs: dict[str, Any] = {
        "tools": None,
        "seed": seed,
        "image_soft_tokens": image_soft_tokens,
        "request_ids": request_ids,
        "max_batch": max_batch,
        "on_complete": on_complete,
        "heartbeat": heartbeat,
    }
    try:
        parameters = inspect.signature(gen_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = {}
    if parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    outputs = gen_fn(model, processor, conversations, sampling, **kwargs)
    return list(outputs)


def _completion_record(
    item: Any,
    sample_idx: int,
    output: GenOut,
    kind: str,
    score_fn: Callable | None,
    parse_fn: Callable[[str], tuple[str | None, str | None]] | None = None,
) -> dict:
    """One partial-file record, shared by the padded and refill paths."""
    item_id = str(item.item_id)
    truncated = output.finish_reason == "length"
    if kind in _OFFICIAL_KINDS:
        correct = None  # recomputed from the official grader
    else:
        correct = False if truncated else bool(score_fn(item, output.text))
    extracted_answer = None
    parse_status = None
    if parse_fn is not None:
        extracted_answer, parse_status = parse_fn(output.text)
    return {
        "item_id": item_id,
        "sample_idx": sample_idx,
        "completion_tokens": int(output.completion_tokens),
        "finish_reason": output.finish_reason,
        "completion": output.text,
        "reasoning": getattr(output, "reasoning", "") or "",
        "extracted_answer": extracted_answer,
        "parse_status": parse_status,
        # Diagnostic only: finalize re-scores every completion with the
        # current scorer instead of trusting this.
        "correct": correct,
    }


def _needs_loading(gen_fn: Callable) -> bool:
    if gen_fn in (generate_batch, generate_batch_refill):
        return True
    wrapped = getattr(gen_fn, "func", None)
    return wrapped in (generate_batch, generate_batch_refill)


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_spilled(fraction: float = 0.95) -> bool:
    """True when the last chunk reserved more than ``fraction`` of the card."""
    if not torch.cuda.is_available():
        return False
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return bool(total and torch.cuda.max_memory_reserved() > fraction * total)


def _log(message: str) -> None:
    print(f"[runner] {message}", file=sys.stderr, flush=True)


def _heartbeat_writer(path: Path, bench: str, engine: str) -> Callable[[dict], None]:
    """Atomically update a small liveness file after every refill round.

    A freeze loses only the round in progress: the partial keeps every finished
    completion and the heartbeat tells the controller the process is alive.
    """

    def beat(status: dict) -> None:
        payload = {
            "bench": str(bench),
            "engine": str(engine),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **status,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    return beat


# ---------------------------------------------------------------------------
# Partial-file persistence
# ---------------------------------------------------------------------------
def _partial_path(out_path: Path) -> Path:
    return out_path.with_suffix(_PARTIAL_SUFFIX)


def run_fingerprint(
    spec: ModelSpec,
    bench: str,
    items: list,
    samples: int,
    sampling: Sampling,
    *,
    seed: int = 0,
    image_soft_tokens: int = 280,
    prereg_tag: str = DEFAULT_PREREG_TAG,
    batch: int | None = None,
    batch_policy: str = BATCH_POLICY_VERSION,
    engine: str = "padded",
) -> dict:
    """Describe everything that makes a partial file reusable.

    A partial file is only a cache of completions generated under one exact
    configuration.  Mixing completions from a different sampling, adapter,
    benchmark size or item set would silently corrupt the result, so the
    fingerprint travels with the partial file and with ``RunResult.metadata``.
    """

    loop = getattr(spec, "loop", None)
    size = str(getattr(spec, "size", ""))
    adapters = [
        str(Path(adapter).expanduser().resolve())
        for adapter in (getattr(spec, "adapters", ()) or ())
    ]
    item_ids = [str(item.item_id) for item in items]
    item_hash = hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()
    return {
        "format_version": 2,
        "seed_scheme": "sha256(seed:item:sample)",
        "bench": str(bench),
        "size": size,
        "quant": str(getattr(spec, "quant", "")),
        "n_loops": int(getattr(loop, "n_loops", 0)) if loop is not None else 0,
        "bridge": getattr(loop, "bridge", None) if loop is not None else None,
        "loop_block": (
            list(getattr(loop, "block", None))
            if loop is not None and getattr(loop, "block", None) is not None
            else None
        ),
        "adapter_policy": (
            getattr(loop, "adapter_policy", None) if loop is not None else None
        ),
        "lora_target_layers": (
            getattr(loop, "lora_target_layers", None) if loop is not None else None
        ),
        "adapters": adapters,
        "samples": int(samples),
        "seed": int(seed),
        "image_soft_tokens": int(image_soft_tokens),
        "sampling": asdict(sampling),
        "model_id": MODEL_IDS.get(size, size),
        "prereg_tag": str(prereg_tag),
        "batch": None if batch is None else int(batch),
        "batch_policy": str(batch_policy),
        "engine": str(engine),
        "item_count": len(item_ids),
        "item_ids_sha256": item_hash,
    }


def fingerprint_digest(fingerprint: dict) -> str:
    """Stable SHA-256 of a fingerprint, used in output file names."""

    payload = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint_diff(old: dict, new: dict) -> list[str]:
    lines = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            lines.append(f"  {key}: partial={old.get(key)!r} current={new.get(key)!r}")
    return lines


def _rotate_partial(path: Path) -> Path | None:
    """Move an old partial aside so ``--fresh`` can start from zero."""

    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.{stamp}.{os.getpid()}.bak")
    path.replace(backup)
    return backup


def _truncate_torn_tail(path: Path) -> None:
    """Drop an incomplete final line left by a freeze.

    A crash can leave bytes after the last newline.  Appending a new record
    would glue it to that fragment and produce one unparsable line, losing the
    pair it describes.  Truncating at the last newline keeps every complete
    record and lets the fragment's pair be regenerated.
    """

    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    path.write_bytes(data[: cut + 1] if cut >= 0 else b"")


def _read_partial(path: Path) -> tuple[dict | None, dict[tuple[str, int], dict]]:
    """Read the fingerprint header and the completed records from a partial."""

    records: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return None, records

    header: dict | None = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # ``_truncate_torn_tail`` removes the usual trailing fragment;
            # this also tolerates a corrupt line in the middle of a file.
            continue
        if not isinstance(record, dict):
            continue
        if index == 0 and record.get("type") == _HEADER_TYPE:
            candidate = record.get("fingerprint")
            if isinstance(candidate, dict):
                header = candidate
            continue
        if record.get("type") == _HEADER_TYPE:
            continue
        key = (str(record["item_id"]), int(record["sample_idx"]))
        records[key] = record
    return header, records


def _write_header(handle, fingerprint: dict) -> None:
    handle.write(
        json.dumps(
            {"type": _HEADER_TYPE, "fingerprint": fingerprint},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def _open_partial(
    path: Path, fingerprint: dict, *, fresh: bool
) -> tuple[dict[tuple[str, int], dict], Any]:
    """Prepare the partial file, refusing to mix incompatible completions.

    Returns the already-completed records and an append handle for new records.
    The caller owns and closes the handle.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        _rotate_partial(path)

    _truncate_torn_tail(path)
    header, records = _read_partial(path)

    if header is not None:
        if header != fingerprint:
            details = "\n".join(_fingerprint_diff(header, fingerprint))
            raise PartialProvenanceError(
                f"partial file {path} belongs to a different run configuration:\n"
                f"{details}\n"
                "Refusing to mix completions into a new result. Re-run with "
                "--fresh to rotate the old partial, or restore the original "
                "configuration."
            )
        return records, path.open("a", encoding="utf-8")

    if path.exists() and path.stat().st_size > 0:
        raise PartialProvenanceError(
            f"partial file {path} has no run fingerprint (created by an older "
            "runner). Refusing to reuse it; re-run with --fresh to rotate it."
        )

    handle = path.open("a", encoding="utf-8")
    _write_header(handle, fingerprint)
    return records, handle


def _write_records(handle, records: Iterable[dict]) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Final scoring
# ---------------------------------------------------------------------------
def _sort_items(items: list, samples: int) -> list[tuple[Any, int]]:
    """Sample-major order, matching the generation order and stable on resume."""

    return [(item, sample_idx) for sample_idx in range(samples) for item in items]


def _finalize_lcb(
    items: list, samples: int, records: dict[tuple[str, int], dict]
) -> tuple[dict[tuple[str, int], bool], dict]:
    outputs: list[dict] = []
    for item in items:
        item_id = str(item.item_id)
        code_list: list[str] = []
        for sample_idx in range(samples):
            record = records[(item_id, sample_idx)]
            code_list.append(lcb.extract_code(record.get("completion", "")))
        outputs.append({"question_id": item_id, "code_list": code_list})

    grades = lcb.run_official_evaluator(outputs)
    correctness: dict[tuple[str, int], bool] = {}
    for item in items:
        item_id = str(item.item_id)
        per_sample = grades.get(item_id)
        if per_sample is None:
            raise RuntimeError(
                f"LiveCodeBench official evaluator returned no grades for "
                f"{item_id!r}"
            )
        if len(per_sample) != samples:
            raise RuntimeError(
                f"LiveCodeBench official evaluator returned {len(per_sample)} "
                f"grades for {samples} samples of {item_id!r}"
            )
        for sample_idx, correct in enumerate(per_sample):
            correctness[(item_id, sample_idx)] = bool(correct)
    return correctness, {
        "scoring": "official",
        "lcb_release": lcb.RELEASE,
        "lcb_timeout_s": lcb.DEFAULT_TIMEOUT,
    }


def _finalize_mathvision(
    items: list,
    samples: int,
    records: dict[tuple[str, int], dict],
    score_fn: Callable[[Any, str], bool],
) -> tuple[dict[tuple[str, int], bool], dict]:
    outputs: list[dict] = []
    official_ids: dict[tuple[str, int], str] = {}
    for item in items:
        item_id = str(item.item_id)
        for sample_idx in range(samples):
            record = records[(item_id, sample_idx)]
            record_id = item_id if samples == 1 else f"{item_id}::{sample_idx}"
            official_ids[(item_id, sample_idx)] = record_id
            outputs.append(
                {
                    "id": record_id,
                    "item_id": record_id,
                    "completion": record.get("completion", ""),
                    "answer": "" if item.answer is None else item.answer,
                    "options": list(item.meta.get("options") or []),
                    "subject": item.meta.get("subject", ""),
                    "level": item.meta.get("level", 0),
                }
            )

    try:
        grades = mathvision.run_official_scorer(outputs)
    except FileNotFoundError as exc:
        # The official eval-only environment is absent.  Use the approximate
        # in-package scorer, but never present its numbers as official.
        correctness = {}
        for item in items:
            item_id = str(item.item_id)
            for sample_idx in range(samples):
                record = records[(item_id, sample_idx)]
                correctness[(item_id, sample_idx)] = bool(
                    score_fn(item, record.get("completion", ""))
                )
        return correctness, {
            "scoring": "non-official",
            "scoring_reason": str(exc),
        }

    correctness = {}
    for item in items:
        item_id = str(item.item_id)
        for sample_idx in range(samples):
            record_id = official_ids[(item_id, sample_idx)]
            if record_id not in grades:
                raise RuntimeError(
                    f"MATH-Vision official scorer returned no grade for "
                    f"{record_id!r}"
                )
            correctness[(item_id, sample_idx)] = bool(grades[record_id])
    return correctness, {"scoring": "official"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def training_seed_from_adapters(adapters: Iterable[str | Path]) -> int:
    """Return the training seed recorded by the evaluated adapter.

    ``RunResult.seed`` is the training seed of the evaluated arm, not the
    generation seed.  Unlooped/untrained models (``O``, ``R``, ``R-damped``)
    have no adapter and default to 0.  For an adapter chain the last adapter is
    the model actually evaluated, so its ``run.json`` is required: silently
    falling back to an earlier adapter or to seed 0 would mislabel the run.
    """

    chain = list(adapters)
    if not chain:
        return 0
    path = Path(chain[-1]) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"evaluated adapter has no run.json (training seed unknown): {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training seed from {path}: {exc}") from exc
    if "seed" not in payload:
        raise ValueError(f"{path} has no 'seed' field")
    try:
        return int(payload["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has a non-integer 'seed'") from exc


def _executed_n_loops(model: Any) -> int | None:
    """Return the loop count actually running in a loaded model.

    Handles the PEFT wrapper and both Gemma 4 wrappers
    (``Gemma4ForCausalLM.model`` is the text model;
    ``Gemma4ForConditionalGeneration.model.language_model`` is).
    """
    base = model
    get_base_model = getattr(base, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
    inner = getattr(base, "model", None)
    text_model = getattr(inner, "language_model", inner)
    return getattr(text_model, "_gemmaloop_n_loops", None)


def run_bench(
    spec: ModelSpec,
    variant: str,
    bench: str,
    items: list,
    samples: int,
    sampling: Sampling,
    out_path: Path,
    gen_fn: Callable | None = None,
    batch: int | None = None,
    *,
    seed: int = 0,
    image_soft_tokens: int = 280,
    prereg_tag: str = DEFAULT_PREREG_TAG,
    model: Any = None,
    processor: Any = None,
    adapter: Any = None,
    load_fn: Callable | None = None,
    fresh: bool = False,
    engine: str = "padded",
    on_batch: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run one benchmark, resuming from ``out_path.with_suffix(".partial.jsonl")``.

    Parameters that are not in the brief's minimal signature are keyword-only:
    ``seed`` (first sample); ``image_soft_tokens`` (280 or 1120 for vision
    benchmarks); ``prereg_tag``; preloaded ``model``/``processor``; a fake
    ``adapter`` for tests; ``load_fn`` for an injectable loader; and ``fresh``,
    which rotates the old partial instead of resuming it.  ``batch`` defaults to
    the largest measured non-spilling batch for ``sampling.max_new_tokens``
    (64 at 2k, 48 at 4k, 32 at 8k, 24 at 16k, 16 at 32k).  ``engine`` selects
    the padded :func:`generate_batch` (default) or the round-based
    :func:`generate_batch_refill`; it is recorded in the fingerprint.  With the
    real generator and no preloaded model, the model is loaded from ``spec``
    exactly once, at the start of this benchmark.

    The first line of the partial file is a fingerprint of the run
    configuration.  A resumed run refuses to append to a partial generated with
    different sampling, adapter, benchmark size or item set, so old completions
    can never leak into a result with a new pre-registration tag.  Every
    finished batch is appended and fsynced before the next batch starts.
    Completed pairs are never regenerated.  The final :class:`RunResult` is
    written to ``out_path``.  With ``engine="refill"`` each completion is
    appended and fsynced as soon as it finishes (not at the end of the whole
    bench), so a freeze keeps every finished request.
    """

    if engine not in ("padded", "refill"):
        raise ValueError(
            f"engine must be 'padded' or 'refill', got {engine!r}"
        )
    if gen_fn is None:
        gen_fn = generate_batch if engine == "padded" else generate_batch_refill
    if load_fn is None:
        load_fn = load_model

    training_seed = training_seed_from_adapters(
        getattr(spec, "adapters", ()) or ()
    )
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    if batch is None:
        batch = max_batch_for(
            sampling.max_new_tokens, str(getattr(spec, "size", "e2b"))
        )
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    if image_soft_tokens not in IMAGE_SOFT_TOKEN_CHOICES:
        raise ValueError(
            f"image_soft_tokens must be one of {IMAGE_SOFT_TOKEN_CHOICES}, "
            f"got {image_soft_tokens}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_path(out_path)
    loop = getattr(spec, "loop", None)
    if loop is not None and getattr(spec, "adapters", ()):
        # Resolve checkpoint geometry *before* writing the partial header: the
        # public path may not have loaded the model yet, and the fingerprint
        # must not record None for a block/policy the adapter supplies.
        resolved = _resolve_loop_config(loop, tuple(spec.adapters))
        if resolved is not None:
            loop.block = resolved.block
            loop.bridge = resolved.bridge
            loop.adapter_policy = resolved.adapter_policy
            loop.lora_target_layers = resolved.lora_target_layers
    requested_n_loops = int(getattr(loop, "n_loops", 0)) if loop is not None else 0
    fingerprint = run_fingerprint(
        spec,
        bench,
        items,
        samples,
        sampling,
        seed=seed,
        image_soft_tokens=image_soft_tokens,
        prereg_tag=prereg_tag,
        batch=batch,
        batch_policy=BATCH_POLICY_VERSION,
        engine=engine,
    )

    if not items:
        if fresh:
            _rotate_partial(partial_path)
        executed_n_loops = _executed_n_loops(model) if model is not None else None
        result = RunResult(
            variant=str(variant),
            size=str(getattr(spec, "size", "")),
            n_loops=int(getattr(loop, "n_loops", 0)) if loop is not None else 0,
            bench=str(bench),
            prereg_tag=str(prereg_tag),
            seed=training_seed,
            loop_block=getattr(loop, "block", None) if loop is not None else None,
            bridge=getattr(loop, "bridge", None) if loop is not None else None,
            adapter_policy=(
                getattr(loop, "adapter_policy", None) if loop is not None else None
            ),
            lora_target_layers=(
                getattr(loop, "lora_target_layers", None) if loop is not None else None
            ),
            items=[],
            metadata={
                "fingerprint": fingerprint,
                "batch": int(batch),
                "batch_policy": BATCH_POLICY_VERSION,
                "engine": engine,
                "executed_n_loops": executed_n_loops,
            },
        )
        result.save(out_path)
        return result

    parse_fn = None
    if adapter is not None:
        build_messages = adapter.build_messages
        score_fn = adapter.score
        try:
            kind = resolve_bench(bench).kind
        except ValueError:
            kind = "score"
    else:
        resolved = resolve_bench(bench)
        build_messages = resolved.module.build_messages
        # LCB has no per-completion scorer: it is graded by the official
        # evaluator after every code sample has been collected.
        score_fn = getattr(resolved.module, "score", None)
        parse_fn = getattr(resolved.module, "parse_answer", None)
        kind = resolved.kind

    # Provision (or validate) the partial before loading the model: a
    # provenance mismatch must fail fast, not after a multi-GB model load.
    records, handle = _open_partial(partial_path, fingerprint, fresh=fresh)
    done = set(records)

    try:
        if model is None and processor is None and _needs_loading(gen_fn):
            model, processor, _ = load_fn(spec)

        executed_n_loops = _executed_n_loops(model) if model is not None else None
        if requested_n_loops >= 1:
            if executed_n_loops is None:
                raise RuntimeError(
                    f"{bench}: requested N={requested_n_loops} but the loaded "
                    "model is not looped"
                )
            if int(executed_n_loops) != requested_n_loops:
                raise RuntimeError(
                    f"{bench}: requested N={requested_n_loops} but the loaded "
                    f"model executes N={executed_n_loops}"
                )

        used_batch = int(batch)
        heartbeat = _heartbeat_writer(
            out_path.with_suffix(".heartbeat.json"), str(bench), engine
        )
        if engine == "refill":
            # The engine owns the slot scheduler: give it the whole benchmark
            # (every unfinished item x sample) as one queue and cap the number
            # of slots with ``batch``.  Padded stays chunked below.  Each
            # finished request is appended and fsynced through ``on_complete``,
            # so a freeze keeps every completed completion and a resume skips
            # them.
            current_batch = int(batch)
            while True:
                pairs = [
                    (item, sample_idx)
                    for sample_idx in range(samples)
                    for item in items
                    if (str(item.item_id), sample_idx) not in records
                ]
                if not pairs:
                    break
                conversations = [build_messages(item) for item, _ in pairs]
                request_ids = [
                    f"{item.item_id}:{sample_idx}"
                    for item, sample_idx in pairs
                ]

                def on_complete(index: int, output: GenOut) -> None:
                    item, sample_idx = pairs[index]
                    record = _completion_record(
                        item, sample_idx, output, kind, score_fn, parse_fn
                    )
                    key = (str(record["item_id"]), int(record["sample_idx"]))
                    records[key] = record
                    _write_records(handle, [record])
                    if on_batch is not None:
                        on_batch(
                            {
                                "bench": str(bench),
                                "sample_idx": int(record["sample_idx"]),
                                "items_done": len(records),
                                "items_total": len(items) * int(samples),
                                "tokens": int(
                                    record.get("completion_tokens", 0)
                                ),
                                "batch": len(pairs),
                                "truncated": record.get("finish_reason")
                                == "length",
                            }
                        )

                _reset_peak_memory()
                try:
                    outputs = _call_gen(
                        gen_fn,
                        model,
                        processor,
                        conversations,
                        sampling,
                        # Same convention as the padded engine:
                        # sha256(seed:item:sample), no arm name.
                        seed,
                        image_soft_tokens,
                        request_ids=request_ids,
                        max_batch=current_batch,
                        on_complete=on_complete,
                        heartbeat=heartbeat,
                    )
                    if len(outputs) != len(pairs):
                        raise RuntimeError(
                            f"generator returned {len(outputs)} outputs for "
                            f"{len(pairs)} conversations"
                        )
                    spilled = _peak_spilled()
                except torch.cuda.OutOfMemoryError:
                    if current_batch == 1:
                        raise
                    outputs = None
                    spilled = True
                if spilled:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        _log(
                            f"{bench}: OOM/spill at batch {used_batch}; "
                            f"retrying queue at batch {current_batch}"
                        )
                        used_batch = current_batch
                        # Finished requests are already persisted and excluded
                        # from the rebuilt queue; nothing computed is lost.
                        continue
                    _log(
                        f"{bench}: spill at batch 1; continuing (results "
                        "stay valid, throughput may degrade)"
                    )
                used_batch = current_batch

                # Generators that do not implement ``on_complete`` (or fake
                # generators in tests) still need their outputs persisted.
                for index, output in enumerate(outputs or []):
                    item, sample_idx = pairs[index]
                    key = (str(item.item_id), sample_idx)
                    if key in records:
                        continue
                    record = _completion_record(
                        item, sample_idx, output, kind, score_fn, parse_fn
                    )
                    records[key] = record
                    _write_records(handle, [record])
                    if on_batch is not None:
                        on_batch(
                            {
                                "bench": str(bench),
                                "sample_idx": int(record["sample_idx"]),
                                "items_done": len(records),
                                "items_total": len(items) * int(samples),
                                "tokens": int(
                                    record.get("completion_tokens", 0)
                                ),
                                "batch": len(pairs),
                                "truncated": record.get("finish_reason")
                                == "length",
                            }
                        )
                del outputs, conversations
                break
        else:
            for sample_idx in range(samples):
                pending = [
                    item
                    for item in items
                    if (str(item.item_id), sample_idx) not in done
                ]
                current_batch = int(batch)
                start = 0
                while start < len(pending):
                    chunk = pending[start : start + current_batch]
                    conversations = [build_messages(item) for item in chunk]
                    request_ids = [
                        f"{item.item_id}:{sample_idx}" for item in chunk
                    ]
                    _reset_peak_memory()
                    try:
                        outputs = _call_gen(
                            gen_fn,
                            model,
                            processor,
                            conversations,
                            sampling,
                            seed,
                            image_soft_tokens,
                            request_ids=request_ids,
                        )
                        if len(outputs) != len(chunk):
                            raise RuntimeError(
                                f"generator returned {len(outputs)} outputs for "
                                f"{len(chunk)} conversations"
                            )
                        spilled = _peak_spilled()
                    except torch.cuda.OutOfMemoryError:
                        if current_batch == 1:
                            raise
                        outputs = None
                        spilled = True
                    if spilled:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if current_batch > 1:
                            current_batch = max(1, current_batch // 2)
                            _log(
                                f"{bench}: OOM/spill at batch {used_batch}; "
                                f"retrying chunk at batch {current_batch}"
                            )
                            used_batch = current_batch
                            continue
                        _log(
                            f"{bench}: spill at batch 1; continuing (results "
                            "stay valid, throughput may degrade)"
                        )
                    used_batch = current_batch

                    new_records = []
                    for item, output in zip(chunk, outputs):
                        record = _completion_record(
                            item, sample_idx, output, kind, score_fn, parse_fn
                        )
                        records[
                            (str(item.item_id), sample_idx)
                        ] = record
                        new_records.append(record)

                    if new_records:
                        _write_records(handle, new_records)
                        if on_batch is not None:
                            on_batch(
                                {
                                    "bench": str(bench),
                                    "sample_idx": int(sample_idx),
                                    "items_done": len(records),
                                    "items_total": len(items) * int(samples),
                                    "tokens": sum(
                                        int(
                                            record.get(
                                                "completion_tokens", 0
                                            )
                                        )
                                        for record in new_records
                                    ),
                                    "batch": len(chunk),
                                    "truncated": sum(
                                        record.get("finish_reason")
                                        == "length"
                                        for record in new_records
                                    ),
                                }
                            )
                    heartbeat(
                        {
                            "round": 0,
                            "finished": len(records),
                            "active": 0,
                            "pending": len(pending) - start - len(chunk),
                            "generated_tokens": sum(
                                int(record.get("completion_tokens", 0))
                                for record in records.values()
                            ),
                        }
                    )
                    start += len(chunk)
                    del outputs, conversations
    finally:
        handle.close()

    if kind == "lcb":
        correctness, metadata = _finalize_lcb(items, samples, records)
    elif kind == "mathvision":
        correctness, metadata = _finalize_mathvision(
            items, samples, records, score_fn
        )
    else:
        correctness = {}
        for item in items:
            item_id = str(item.item_id)
            for sample_idx in range(samples):
                record = records[(item_id, sample_idx)]
                truncated = record.get("finish_reason") == "length"
                # Re-score with the current scorer.  The value cached in the
                # partial file is useful when debugging a scorer change, but it
                # must never decide the final result.
                rescored = bool(score_fn(item, record.get("completion", "")))
                correctness[(item_id, sample_idx)] = (
                    False if truncated else rescored
                )
        metadata = {}

    metadata = dict(metadata)
    parsed = [
        record.get("parse_status")
        for record in records.values()
        if record.get("parse_status") is not None
    ]
    if parsed:
        failures = sum(status != "ok" for status in parsed)
        metadata["parse_failures"] = failures
        metadata["parse_failure_rate"] = failures / len(parsed)
        metadata["parse_status_counts"] = {
            status: parsed.count(status) for status in sorted(set(parsed))
        }
    metadata["fingerprint"] = fingerprint
    metadata["batch"] = int(batch)
    metadata["batch_policy"] = BATCH_POLICY_VERSION
    metadata["batch_after_fallback"] = int(used_batch if items else batch)
    metadata["engine"] = engine
    metadata["executed_n_loops"] = executed_n_loops

    item_results: list[ItemResult] = []
    for item, sample_idx in _sort_items(items, samples):
        item_id = str(item.item_id)
        record = records[(item_id, sample_idx)]
        truncated = record.get("finish_reason") == "length"
        correct = False if truncated else bool(
            correctness[(item_id, sample_idx)]
        )
        item_meta: dict = {}
        if "post_cut" in item.meta:
            item_meta["post_cut"] = bool(item.meta["post_cut"])
        item_results.append(
            ItemResult(
                item_id=item_id,
                sample_idx=sample_idx,
                correct=correct,
                completion_tokens=int(record.get("completion_tokens", 0)),
                finish_reason=str(record.get("finish_reason", "stop")),
                extracted_answer=record.get("extracted_answer"),
                parse_status=record.get("parse_status"),
                reasoning=record.get("reasoning") or None,
                meta=item_meta,
            )
        )

    result = RunResult(
        variant=str(variant),
        size=str(getattr(spec, "size", "")),
        n_loops=int(getattr(loop, "n_loops", 0)) if loop is not None else 0,
        bench=str(bench),
        prereg_tag=str(prereg_tag),
        seed=training_seed,
        loop_block=getattr(loop, "block", None) if loop is not None else None,
        bridge=getattr(loop, "bridge", None) if loop is not None else None,
        adapter_policy=(
            getattr(loop, "adapter_policy", None) if loop is not None else None
        ),
        lora_target_layers=(
            getattr(loop, "lora_target_layers", None) if loop is not None else None
        ),
        items=item_results,
        metadata=metadata,
    )
    result.save(out_path)
    return result
