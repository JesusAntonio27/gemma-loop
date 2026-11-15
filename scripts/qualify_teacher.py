"""Qualify one teacher runtime against its bf16 reference (review X3 section 7).

The script is a one-teacher-per-process engineering screen, not a training or
selection run.  It renders a deterministic, length-stratified development set,
computes the reference hidden states (resident bf16 for E4B, or a layer-streamed
bf16 pass over the 12B safetensors because 24 GB cannot sit in RAM), then loads
the chosen registered runtime and compares the two full distributions.

Default development corpus
--------------------------
``C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\data\\p1lite\\corpus.jsonl`` is a
**dev stand-in**: the Stage-0-lite pilot O-samples do not exist yet.  It is
stratified by rendered length (4 short, 8 around 1-2k, 4 medium, 4 near the
upper limit for the 20-row screen; the same pattern scales to 100 rows).  Rows
are selected by length and a seeded hash, never by teacher losses.

Usage (GPU, after the CPU suite is green)::

    uv run --offline python scripts/qualify_teacher.py \\
        --teacher e4b --runtime bf16 \\
        --rows C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\data\\p1lite\\corpus.jsonl \\
        --out C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\qualification\\e4b-bf16 \\
        --vram-cap 0.857 --vram-free-margin-gib 0.5

    uv run --offline python scripts/qualify_teacher.py \\
        --teacher 12b_qat --runtime nf4 \\
        --rows C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\data\\p1lite\\corpus.jsonl \\
        --out C:\\Users\\<user>\\ML\\gemmaloop-artifacts\\qualification\\12b-nf4 \\
        --vram-cap 0.857 --vram-free-margin-gib 0.5

The script takes the project GPU lock, caps the caching allocator, writes a
heartbeat, and never loads two teachers at once in the same process.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gemmaloop.data.teacher_topk import (  # noqa: E402
    DEFAULT_TOP_K,
    GLITCH_TOKEN_ID,
    NonFiniteTeacherRow,
    compute_teacher_row,
    teacher_spec,
)
from gemmaloop.gpulock import require_gpu_lock  # noqa: E402
from gemmaloop.heartbeat import Heartbeat  # noqa: E402
from gemmaloop.model.snapshot import chat_template_sha256, model_revision  # noqa: E402
from gemmaloop.paths import artifacts_dir  # noqa: E402
from gemmaloop.teacher.qualification import (  # noqa: E402
    ACCEPTANCE_THRESHOLDS,
    ProtectionError,
    aggregate_metrics,
    default_protected_sources,
    evaluate_acceptance,
    extract_row_ids,
    filter_disjoint,
    load_protected_ids,
    resolve_decision,
    select_development_rows,
    token_spans,
)
from gemmaloop.train.labels import assistant_labels  # noqa: E402
from gemmaloop.train.perf import cap_cuda_allocator  # noqa: E402

LOGGER = logging.getLogger("qualify_teacher")

DEFAULT_P1LITE_CORPUS = Path(
    r"C:\Users\<user>\ML\gemmaloop-artifacts\data\p1lite\corpus.jsonl"
)
DEV_CORPUS_NOTE = (
    "The registered default is the P1-lite DAPO corpus, outside the Stage 0 "
    "pool and decontaminated against every eval; selection is by length and a "
    "seeded hash, never by teacher loss.  By-id disjointness against the "
    "registered pilot/fit/monitor/confirmatory manifests is checked at run time."
)
REPORT_FORMAT = "gemmaloop-teacher-qualification-v1"
DEFAULT_PROTECTED_ROOT = artifacts_dir() / "stage0"
DEFAULT_STAGE0_POOL = artifacts_dir() / "stage0" / "pool"
STAGE0_FILTER_VERSION = "p1lite-stage0-protected-v2"
SUPPORTED_CORPUS_KINDS = {"p1lite_corpus", "p1lite_stage0_local"}


class CorpusManifestError(RuntimeError):
    """The corpus was not filtered, or not filtered against this Stage 0 pool."""


def read_stage0_pool_manifest_sha(pool_dir: str | Path) -> str:
    """Return the frozen ``manifest_sha256`` of the Stage 0 pool."""

    manifest_path = Path(pool_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusManifestError(
            f"Stage 0 pool manifest not found: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusManifestError(
            f"Stage 0 pool manifest is unreadable: {manifest_path}: {exc}"
        ) from exc
    sha = payload.get("manifest_sha256") if isinstance(payload, dict) else None
    if not sha:
        raise CorpusManifestError(
            f"Stage 0 pool manifest has no manifest_sha256: {manifest_path}"
        )
    return str(sha)


def check_corpus_disjointness(
    rows_path: str | Path,
    pool_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refuse a corpus whose manifest lacks or mismatches the Stage 0 record."""

    rows = Path(rows_path)
    manifest = (
        Path(manifest_path) if manifest_path is not None else rows.parent / "manifest.json"
    )
    if not manifest.is_file():
        raise CorpusManifestError(
            f"corpus manifest not found: {manifest}; the P1-lite corpus must "
            "record its Stage 0 disjointness filter"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusManifestError(
            f"corpus manifest is unreadable: {manifest}: {exc}"
        ) from exc
    kind = payload.get("kind") if isinstance(payload, dict) else None
    if kind not in (None, "") and str(kind) not in SUPPORTED_CORPUS_KINDS:
        raise CorpusManifestError(
            f"unsupported corpus kind {kind!r}; expected one of "
            f"{sorted(SUPPORTED_CORPUS_KINDS)}"
        )
    record = payload.get("stage0_disjointness") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise CorpusManifestError(
            f"corpus manifest lacks the Stage 0 disjointness record: {manifest}"
        )
    filter_version = str(record.get("filter_version") or "")
    if not filter_version:
        raise CorpusManifestError(
            f"corpus manifest has a Stage 0 disjointness record without a "
            f"filter version: {manifest}"
        )
    corpus_pool_sha = str(record.get("pool_manifest_sha256") or "")
    if not corpus_pool_sha:
        raise CorpusManifestError(
            f"corpus manifest has a Stage 0 disjointness record without the "
            f"pool manifest sha256: {manifest}"
        )
    current_pool_sha = read_stage0_pool_manifest_sha(pool_dir)
    if corpus_pool_sha != current_pool_sha:
        raise CorpusManifestError(
            "corpus was filtered against a different pool manifest: "
            f"corpus {corpus_pool_sha}, current {current_pool_sha} "
            f"({Path(pool_dir) / 'manifest.json'})"
        )
    return record


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualificationRow:
    row_id: int
    input_ids: list[int]
    labels: list[int]
    source: str = ""
    item_id: str = ""
    item_ids: tuple[str, ...] = ()
    spans: tuple[str | None, ...] = ()

    @property
    def length(self) -> int:
        return len(self.input_ids)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL development corpus (``messages`` or rendered ``input_ids``)."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"qualification rows not found: {source}. Pass --rows with a JSONL "
            "file; the default P1-lite corpus is documented as a dev stand-in."
        )
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            if "messages" not in row and "input_ids" not in row:
                raise ValueError(
                    f"{source}:{line_number}: expected 'messages' or 'input_ids'"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"no development rows in {source}")
    return rows


def _marker_ids(tokenizer, token: str, fallback: Sequence[int]) -> list[int]:
    try:
        ids = list(tokenizer.encode(token, add_special_tokens=False))
    except Exception:
        return list(fallback)
    return ids or list(fallback)


def prepare_rows(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    *,
    label_fn: Callable[..., dict[str, list[int]]] = assistant_labels,
    thought_open_ids: Sequence[int] = (),
    thought_close_ids: Sequence[int] = (),
) -> list[QualificationRow]:
    """Render exact student IDs and assistant labels for every raw row."""

    prepared: list[QualificationRow] = []
    open_ids = list(thought_open_ids)
    close_ids = list(thought_close_ids)
    for row_id, row in enumerate(rows):
        if "messages" in row:
            output = label_fn(
                tokenizer,
                row["messages"],
                int(max_length),
                tools=row.get("tools"),
            )
            input_ids = [int(value) for value in output["input_ids"]]
            labels = [int(value) for value in output["labels"]]
        else:
            input_ids = [int(value) for value in row["input_ids"]][: int(max_length)]
            start = int(row.get("labels_start", 0))
            labels = [-100] * len(input_ids)
            for position in range(max(0, start), len(input_ids)):
                labels[position] = input_ids[position]
            if "labels" in row:
                labels = [int(value) for value in row["labels"]][: len(input_ids)]
        if len(labels) != len(input_ids):
            raise ValueError(
                f"row {row_id}: {len(input_ids)} input ids but {len(labels)} labels"
            )
        spans = tuple(
            token_spans(
                input_ids,
                labels,
                thought_open_ids=open_ids,
                thought_close_ids=close_ids,
            )
        )
        aliases = tuple(sorted(extract_row_ids(row)))
        prepared.append(
            QualificationRow(
                row_id=row_id,
                input_ids=input_ids,
                labels=labels,
                source=str(row.get("source") or row.get("id") or ""),
                item_id=aliases[0] if aliases else "",
                item_ids=aliases,
                spans=spans,
            )
        )
    return prepared


def select_rows(
    prepared: Sequence[QualificationRow],
    *,
    screen_rows: int = 20,
    expand_rows: int = 100,
    seed: int = 0,
) -> tuple[list[tuple[QualificationRow, str]], list[tuple[QualificationRow, str]]]:
    """Return the independent stratified screen and the expansion superset."""

    selected = select_development_rows(
        [row.length for row in prepared],
        screen_rows=int(screen_rows),
        expand_rows=int(expand_rows),
        seed=int(seed),
    )
    return (
        [(prepared[index], label) for index, label in selected["screen"]],
        [(prepared[index], label) for index, label in selected["expansion"]],
    )


def prediction_slice(
    hidden: torch.Tensor, labels: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hidden at predecessor positions, actual tokens and positions."""

    if hidden.ndim != 2:
        raise ValueError(
            f"prediction_slice expects [S, H] hidden states, got "
            f"{tuple(hidden.shape)}"
        )
    positions = [
        position
        for position, label in enumerate(labels)
        if int(label) != -100 and position > 0
    ]
    if not positions:
        empty = torch.zeros((0, hidden.shape[1]), dtype=hidden.dtype)
        return empty, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    index = torch.tensor([position - 1 for position in positions], device=hidden.device)
    actual = torch.tensor(
        [int(labels[position]) for position in positions], dtype=torch.long
    )
    return (
        hidden.index_select(0, index).detach().to("cpu"),
        actual,
        torch.tensor(positions, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Hidden-state caches / references
# ---------------------------------------------------------------------------


def _teacher_hidden(loaded, input_ids: torch.Tensor) -> torch.Tensor:
    """Return one row's hidden states without the batch dimension."""

    return loaded.hidden(input_ids)[0]


def build_hidden_cache(
    rows: Sequence[QualificationRow],
    hidden_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: str,
    heartbeat: Heartbeat | None = None,
    phase: str = "",
) -> dict[int, dict[str, Any]]:
    """Run every row through a teacher and keep only prediction positions."""

    cache: dict[int, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        input_ids = torch.tensor([row.input_ids], dtype=torch.long, device=device)
        hidden = _two_dim_hidden(hidden_fn(input_ids), "hidden_fn")
        prediction, actual, positions = prediction_slice(hidden, row.labels)
        spans = [row.spans[int(position)] for position in positions.tolist()]
        cache[row.row_id] = {
            "hidden": prediction.to(torch.bfloat16),
            "actual": actual,
            "positions": positions,
            "spans": spans,
        }
        del hidden, input_ids
        if heartbeat is not None:
            heartbeat.update(
                phase=phase,
                items_done=number,
                items_total=len(rows),
            )
    return cache


def _two_dim_hidden(hidden: torch.Tensor, source: str) -> torch.Tensor:
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        return hidden.squeeze(0)
    if hidden.ndim != 2:
        raise ValueError(
            f"{source} must return [S, H] or [1, S, H], got {tuple(hidden.shape)}"
        )
    return hidden


def build_hidden_cache_from_hidden(
    rows: Sequence[QualificationRow],
    hidden_states: Sequence[torch.Tensor],
) -> dict[int, dict[str, Any]]:
    """Select prediction positions from precomputed full-row hidden states."""

    cache: dict[int, dict[str, Any]] = {}
    states = list(hidden_states)
    for index, row in enumerate(rows):
        hidden = _two_dim_hidden(states[index], "the streamed reference")
        prediction, actual, positions = prediction_slice(hidden, row.labels)
        spans = [row.spans[int(position)] for position in positions.tolist()]
        cache[row.row_id] = {
            "hidden": prediction.to(torch.bfloat16),
            "actual": actual,
            "positions": positions,
            "spans": spans,
        }
        # Release the full-row reference tensor as soon as its prediction slice
        # has been copied into the cache; holding both would double peak RAM.
        states[index] = None
        del hidden, prediction
    return cache


def build_reference_cache(
    spec,
    rows: Sequence[QualificationRow],
    *,
    args: argparse.Namespace,
    heartbeat: Heartbeat | None,
) -> tuple[dict[int, dict[str, Any]] | None, dict[str, Any] | None]:
    """Compute the bf16 reference hidden states, then free the reference."""

    if spec.key == "e4b" and spec.runtime == "bf16":
        LOGGER.info("E4B bf16 is both the runtime and its own reference")
        return None, {"kind": "resident-bf16-self"}
    if spec.key == "e4b":
        LOGGER.info("loading the E4B bf16 reference for the int8 runtime")
        from scripts.precompute_teacher_topk import load_teacher

        reference = load_teacher("e4b", device=args.device, runtime="bf16")
        try:
            cache = build_hidden_cache(
                rows,
                lambda ids: _teacher_hidden(reference, ids),
                device=args.device,
                heartbeat=heartbeat,
                phase="reference",
            )
        finally:
            del reference
            gc.collect()
            _empty_cuda()
        return cache, {"kind": "resident-bf16"}

    if spec.key == "12b_qat":
        LOGGER.info("loading the layer-streamed bf16 12B reference")
        from huggingface_hub import snapshot_download

        from gemmaloop.teacher.reference import (
            SafetensorsWeightSource,
            StreamingTextReference,
        )

        directory = snapshot_download(
            spec.model_id,
            revision=spec.revision,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
        )
        source = SafetensorsWeightSource(directory)
        config = _text_config(spec)
        LOGGER.info(
            "streamed reference family=%s model_type=%s softcap=%s",
            _reference_family_name(config),
            getattr(config, "model_type", ""),
            getattr(config, "final_logit_softcapping", None),
        )
        reference = StreamingTextReference(
            config,
            source,
            device=args.device,
            dtype=torch.bfloat16,
            ram_limit_gib=args.reference_ram_limit_gib,
            vram_limit_gib=args.reference_vram_limit_gib,
        )
        try:
            hidden_states = reference.forward_many(
                [torch.tensor([row.input_ids], dtype=torch.long) for row in rows]
            )
            cache = build_hidden_cache_from_hidden(rows, hidden_states)
            if heartbeat is not None:
                heartbeat.update(
                    phase="reference",
                    items_done=len(rows),
                    items_total=len(rows),
                )
        finally:
            reference.close()
            reference_memory = {
                "kind": "layer-streamed-bf16",
                "family": reference.family.name,
                "model_type": str(getattr(config, "model_type", "")),
                "final_logit_softcapping": getattr(
                    config, "final_logit_softcapping", None
                ),
                **dict(reference.memory_report),
            }
            del reference
            gc.collect()
            _empty_cuda()
        return cache, reference_memory
    raise ValueError(f"no reference implementation for {spec.key}/{spec.runtime}")


def _text_config(spec) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision)
    text = getattr(config, "text_config", config)
    required = (
        "num_hidden_layers",
        "hidden_size",
        "layer_types",
        "rms_norm_eps",
        "vocab_size",
        "max_position_embeddings",
    )
    missing = [
        name for name in required if getattr(text, name, None) is None
    ]
    if missing:
        raise ValueError(
            f"{spec.model_id} text config is missing streamed-reference "
            f"fields: {missing}"
        )
    if spec.key == "12b_qat":
        if "unified" not in str(getattr(text, "model_type", "")):
            raise ValueError(
                f"{spec.model_id} text config is not gemma4_unified: "
                f"{getattr(text, 'model_type', None)!r}"
            )
        if int(getattr(text, "hidden_size_per_layer_input", 0) or 0) != 0:
            raise ValueError("the pinned 12B reference must not need PLE")
        if getattr(text, "final_logit_softcapping", None) is None:
            raise ValueError("the pinned 12B config has no final_logit_softcapping")
    try:
        text._attn_implementation = "sdpa"
    except Exception:  # pragma: no cover - config property differences
        pass
    return text


def _file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_family_name(config) -> str:
    from gemmaloop.teacher.reference import reference_family

    return reference_family(config).name


def _empty_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _concat_metrics(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[int], list[str], list[str]]:
    keys = None
    chunks: dict[str, list[torch.Tensor]] = {}
    row_ids: list[int] = []
    strata: list[str] = []
    spans: list[str] = []
    for entry in entries:
        metrics = entry["metrics"]
        if keys is None:
            keys = tuple(metrics)
            chunks = {key: [] for key in keys}
        count = int(metrics["kl"].shape[0])
        for key in keys:
            chunks[key].append(metrics[key])
        row_ids.extend([int(entry["row_id"])] * count)
        strata.extend([str(entry["stratum"])] * count)
        spans.extend(str(value) for value in entry["spans"])
    if keys is None:
        return {}, [], [], []
    return {key: torch.cat(values) for key, values in chunks.items()}, row_ids, strata, spans


def _memory_report(
    cap_info: Mapping[str, Any] | None,
    allocation_samples: Sequence[float],
    smoke: Mapping[str, Any] | None,
    *,
    row_peak_allocated_gib: float = 0.0,
    row_peak_reserved_gib: float = 0.0,
) -> dict[str, Any]:
    peak_allocated = max(
        row_peak_allocated_gib,
        float(torch.cuda.max_memory_allocated()) / 2**30
        if torch.cuda.is_available()
        else 0.0,
    )
    peak_reserved = max(
        row_peak_reserved_gib,
        float(torch.cuda.max_memory_reserved()) / 2**30
        if torch.cuda.is_available()
        else 0.0,
    )
    cap_gib = float(cap_info["cap_gib"]) if cap_info else None
    headroom = cap_gib - peak_reserved if cap_gib is not None else None
    samples = [float(value) for value in allocation_samples]
    steady = True
    if len(samples) >= 4:
        first = samples[: max(1, len(samples) // 4)]
        last = samples[-max(1, len(samples) // 4) :]
        growth = sum(last) / len(last) - sum(first) / len(first)
        steady = growth <= 0.10
    return {
        "cap_gib": cap_gib,
        "peak_allocated_gib": round(peak_allocated, 4),
        "peak_reserved_gib": round(peak_reserved, 4),
        "headroom_gib": None if headroom is None else round(headroom, 4),
        "steady": bool(steady),
        "allocation_samples_gib": [round(value, 4) for value in samples],
        "smoke": dict(smoke) if smoke else None,
    }


def _memory_smoke(
    loaded,
    *,
    max_length: int,
    top_k: int,
    token_chunk: int,
    vocab_chunk: int,
    mask_token_id: int | None,
) -> dict[str, Any]:
    """Synthetic max-length allocation/causal-boundary run (not a fidelity row)."""

    length = int(max_length)
    ids = [2] + [1000 + ((index * 37) % 20000) for index in range(length - 1)]
    labels = [-100] + ids[1:]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.weight.device)
    hidden = loaded.hidden(input_ids)
    row = compute_teacher_row(
        hidden[0].float(),
        loaded.weight,
        torch.tensor(labels, dtype=torch.long, device=loaded.weight.device),
        row_id=0,
        input_ids=ids,
        top_k=top_k,
        mask_token_id=mask_token_id,
        softcap=loaded.softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - started
    result = {
        "length": length,
        "tokens": int(row.tokens),
        "seconds": round(elapsed, 3),
        "peak_allocated_gib": round(
            float(torch.cuda.max_memory_allocated()) / 2**30, 4
        )
        if torch.cuda.is_available()
        else 0.0,
        "peak_reserved_gib": round(
            float(torch.cuda.max_memory_reserved()) / 2**30, 4
        )
        if torch.cuda.is_available()
        else 0.0,
    }
    del row, hidden, input_ids
    _empty_cuda()
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _check_table(checks: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| criterion | value | threshold | pass | near boundary |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for check in checks:
        value = check.get("value")
        lines.append(
            "| {name} | {value:.6g} | {threshold:.6g} | {passed} | {near} |".format(
                name=check.get("name"),
                value=float(value) if value is not None else float("nan"),
                threshold=float(check["threshold"]),
                passed="yes" if check["passed"] else "NO",
                near="yes" if check["near_boundary"] else "no",
            )
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON report as a compact, human-readable markdown page."""

    lines = [
        f"# Teacher qualification: {report['teacher']['key']} / "
        f"{report['teacher']['runtime']}",
        "",
        f"- corpus: `{report['corpus']['path']}`",
        f"- note: {report['corpus']['note']}",
        f"- reference: {report.get('reference', {}).get('kind', '')}",
        f"- rows: screen {report['corpus']['screen_rows']}, "
        f"expansion {report['corpus']['expand_rows']}",
        f"- generated: {report.get('generated_at', '')}",
        "",
        "## Decision",
        "",
        f"**{report['decision']['decision']}** "
        f"({'pass' if report['decision']['passed'] else 'fail'})",
    ]
    if report["decision"].get("next_runtime"):
        lines.append(f"- registered next runtime: `{report['decision']['next_runtime']}`")
    if report["decision"].get("reasons"):
        lines.append("- reasons:")
        for reason in report["decision"]["reasons"]:
            lines.append(f"  - {reason}")
    lines += ["", "## Acceptance checks", ""]
    lines += _check_table(report["decision"].get("checks", []))
    lines += [
        "",
        "## Metrics",
        "",
        "| metric | screen | expanded |",
        "|---|---:|---:|",
    ]
    screen = report["metrics"]["screen"]
    expanded = report["metrics"].get("expanded") or {}
    keys = [
        "mean_full_kl",
        "mean_full_kl_ci_low",
        "mean_full_kl_ci_high",
        "p95_full_kl",
        "union_kl_mean",
        "ref65_kl_mean",
        "stored65_kl_mean",
        "stored65_ref_kl_mean",
        "top1_agreement",
        "top1_confident",
        "top64_overlap_mean",
        "ref_mass_captured_mean",
        "tail_abs_error_mean",
        "tail_abs_error_p95",
        "entropy_diff_mean",
        "entropy_abs_diff_mean",
        "logp_actual_diff_mean",
        "logp_actual_abs_diff_mean",
        "stored_top1_agreement",
        "stored_tail_abs_error_mean",
        "stored_tail_abs_error_p95",
    ]
    for key in keys:
        lines.append(
            "| {key} | {screen} | {expanded} |".format(
                key=key,
                screen=_fmt(screen.get(key)),
                expanded=_fmt(expanded.get(key)),
            )
        )
    lines += ["", "## Memory", ""]
    for key, value in (report.get("memory") or {}).items():
        if key in {"allocation_samples_gib", "smoke"}:
            continue
        lines.append(f"- {key}: {_fmt(value)}")
    smoke = (report.get("memory") or {}).get("smoke")
    if smoke:
        lines.append(
            "- smoke: length {length}, {peak_reserved_gib} GiB reserved, "
            "{seconds} s".format(**smoke)
        )
    lines += [
        "",
        "## Runtime identity",
        "",
        "```json",
        json.dumps(report.get("runtime_identity", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Limitations",
        "",
        "- Twenty rows are an engineering screen, not a population equivalence "
        "proof; the expansion is predetermined, not sampled until it passes.",
        "- The registered default corpus is the P1-lite DAPO corpus (outside the "
        "Stage 0 pool); protected-set manifests are checked when present and an "
        "unreadable registered manifest aborts the run.",
        "- A coarse 65-way KL is a lower bound on the omitted within-tail "
        "distribution; full forward KL is reported as the primary metric.",
        "- E4B bf16 uses itself as the reference, so its fidelity metrics are "
        "identically zero/one by construction and are reported as an "
        "allocation/self-consistency check only.",
        "- The fp16-stored 65-way KL on the runtime's own support is reported "
        "beside the stored KL gathered on the bf16 reference support; the "
        "stored top-1 agreement is invariant to the scalar fp32 renormalisation.",
        "- A stored log p(actual) outside the runtime top-64 assigns the whole "
        "tail bucket to that token: it is an upper-bound proxy for the "
        "serialised distribution, not the token's exact log-probability.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="e4b or 12b_qat")
    parser.add_argument(
        "--runtime",
        default=None,
        help="registered runtime (e4b: bf16/int8; 12b_qat: nf4); default primary",
    )
    parser.add_argument("--rows", type=Path, default=DEFAULT_P1LITE_CORPUS)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=None,
        help="corpus manifest (default: <rows dir>/manifest.json)",
    )
    parser.add_argument(
        "--stage0-pool",
        type=Path,
        default=DEFAULT_STAGE0_POOL,
        help="Stage 0 pool whose frozen manifest sha the corpus must match",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--screen-rows", type=int, default=20)
    parser.add_argument("--expand-rows", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--token-chunk", type=int, default=256)
    parser.add_argument("--vocab-chunk", type=int, default=8192)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-cap", type=float, default=0.857)
    parser.add_argument("--vram-free-margin-gib", type=float, default=0.5)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--skip-memory-smoke", action="store_true")
    parser.add_argument(
        "--reference-ram-limit-gib",
        type=float,
        default=6.0,
        help="layer-streamed reference RAM bound above baseline (default 6 GiB)",
    )
    parser.add_argument(
        "--reference-vram-limit-gib",
        type=float,
        default=None,
        help="optional per-layer VRAM reserved bound for the streamed reference",
    )
    parser.add_argument("--skip-gpu-lock", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--protected-root",
        type=Path,
        default=DEFAULT_PROTECTED_ROOT,
        help="root of the Stage 0 pool/manifests used for disjointness checks",
    )
    parser.add_argument(
        "--protected-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override or add a protected row-id manifest (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if args.screen_rows < 1 or args.expand_rows < args.screen_rows:
        raise ValueError("--expand-rows must be >= --screen-rows >= 1")
    if any(
        value <= 0
        for value in (args.max_length, args.top_k, args.token_chunk, args.vocab_chunk)
    ):
        raise ValueError("chunk sizes, top-k and max length must be positive")
    try:
        corpus_disjointness = check_corpus_disjointness(
            args.rows,
            args.stage0_pool,
            manifest_path=args.corpus_manifest,
        )
    except CorpusManifestError as exc:
        print(f"teacher qualification refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    spec = teacher_spec(args.teacher, runtime=args.runtime)
    from gemmaloop.teacher.runtime import teacher_runtime

    runtime = teacher_runtime(spec.key, spec.runtime)

    if not args.skip_gpu_lock:
        require_gpu_lock()
    cap_info = cap_cuda_allocator(args.vram_cap, args.vram_free_margin_gib)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=model_revision("e2b")
    )
    raw_rows = load_rows(args.rows)
    thought_open = _marker_ids(tokenizer, "<|channel>thought", (100, 45518))
    thought_close = _marker_ids(tokenizer, "<channel|>", (101,))
    prepared = prepare_rows(
        tokenizer,
        raw_rows,
        args.max_length,
        thought_open_ids=thought_open,
        thought_close_ids=thought_close,
    )
    sources = default_protected_sources(args.protected_root)
    for override in args.protected_manifest:
        if "=" not in override:
            raise ValueError("--protected-manifest must be NAME=PATH")
        name, path = override.split("=", 1)
        candidate = Path(path)
        if not candidate.is_file():
            raise SystemExit(
                f"teacher qualification refused: explicit protected manifest "
                f"not found: {candidate}"
            )
        sources[name] = (candidate,)
    try:
        protected_report, protected_ids = load_protected_ids(sources)
        kept_indices, disjointness = filter_disjoint(
            [row.item_ids for row in prepared],
            protected_ids,
            protected_report=protected_report,
        )
    except ProtectionError as exc:
        raise SystemExit(f"teacher qualification refused: {exc}") from exc
    kept_set = set(kept_indices)
    excluded = [
        row for index, row in enumerate(prepared) if index not in kept_set
    ]
    prepared = [prepared[index] for index in kept_indices]

    screen_selected, expansion_selected = select_rows(
        prepared,
        screen_rows=args.screen_rows,
        expand_rows=args.expand_rows,
        seed=args.seed,
    )
    selected = expansion_selected
    selected_rows = [row for row, _ in selected]

    args.out.mkdir(parents=True, exist_ok=True)
    heartbeat = None
    if args.heartbeat is not None:
        heartbeat = Heartbeat(
            args.heartbeat, run=f"qualify-{spec.key}-{spec.runtime}", phase="rows"
        ).start()
    try:
        reference_cache, reference_memory = build_reference_cache(
            spec, selected_rows, args=args, heartbeat=heartbeat
        )

        from scripts.precompute_teacher_topk import load_teacher

        loaded = load_teacher(spec.key, device=args.device, runtime=spec.runtime)
        try:
            from gemmaloop.teacher.qualification import pair_token_metrics as _pair

            entries: list[dict[str, Any]] = []
            allocation_samples: list[float] = []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for number, (row, stratum) in enumerate(selected, start=1):
                if heartbeat is not None:
                    heartbeat.update(
                        phase="runtime", items_done=number, items_total=len(selected)
                    )
                input_ids = torch.tensor(
                    [row.input_ids], dtype=torch.long, device=loaded.weight.device
                )
                hidden = loaded.hidden(input_ids)[0]
                prediction, actual, positions = prediction_slice(hidden, row.labels)
                del hidden, input_ids
                spans = [row.spans[int(position)] for position in positions.tolist()]
                if reference_cache is None:
                    reference_hidden = prediction
                    ref_actual = actual
                else:
                    cached = reference_cache[row.row_id]
                    reference_hidden = cached["hidden"]
                    ref_actual = cached["actual"]
                metrics = _pair(
                    reference_hidden.to(loaded.weight.device),
                    prediction.to(loaded.weight.device),
                    loaded.weight,
                    ref_actual.to(loaded.weight.device),
                    top_k=args.top_k,
                    mask_token_id=GLITCH_TOKEN_ID,
                    softcap=loaded.softcap,
                    token_chunk=args.token_chunk,
                    vocab_chunk=args.vocab_chunk,
                )
                entries.append(
                    {
                        "row_id": row.row_id,
                        "stratum": stratum,
                        "length": row.length,
                        "source": row.source,
                        "spans": spans,
                        "metrics": {
                            key: value.detach().to("cpu")
                            for key, value in metrics.items()
                        },
                    }
                )
                if torch.cuda.is_available():
                    allocation_samples.append(
                        float(torch.cuda.memory_allocated()) / 2**30
                    )

            row_peak_allocated = (
                float(torch.cuda.max_memory_allocated()) / 2**30
                if torch.cuda.is_available()
                else 0.0
            )
            row_peak_reserved = (
                float(torch.cuda.max_memory_reserved()) / 2**30
                if torch.cuda.is_available()
                else 0.0
            )
            smoke = None
            if not args.skip_memory_smoke:
                smoke = _memory_smoke(
                    loaded,
                    max_length=args.max_length,
                    top_k=args.top_k,
                    token_chunk=args.token_chunk,
                    vocab_chunk=args.vocab_chunk,
                    mask_token_id=GLITCH_TOKEN_ID,
                )
            memory = _memory_report(
                cap_info,
                allocation_samples,
                smoke,
                row_peak_allocated_gib=row_peak_allocated,
                row_peak_reserved_gib=row_peak_reserved,
            )
            memory["reference"] = reference_memory
        finally:
            del loaded
            gc.collect()
            _empty_cuda()

        screen_entries = entries[: len(screen_selected)]
        screen_metrics, screen_rows_ids, screen_strata, screen_spans = _concat_metrics(
            screen_entries
        )
        screen_overall = aggregate_metrics(
            screen_metrics,
            screen_rows_ids,
            strata=screen_strata,
            spans=screen_spans,
            bootstrap_samples=args.bootstrap,
            seed=args.seed,
        )
        screen_decision = evaluate_acceptance(
            screen_overall,
            memory=memory,
            teacher_key=spec.key,
            current_runtime=spec.runtime,
            n_rows=len(screen_entries),
            max_rows=int(args.expand_rows),
        )

        expanded_overall = None
        expanded_decision = None
        if screen_decision.decision == "expand" and len(entries) > len(screen_entries):
            all_metrics, all_rows_ids, all_strata, all_spans = _concat_metrics(entries)
            expanded_overall = aggregate_metrics(
                all_metrics,
                all_rows_ids,
                strata=all_strata,
                spans=all_spans,
                bootstrap_samples=args.bootstrap,
                seed=args.seed,
            )
            expanded_decision = evaluate_acceptance(
                expanded_overall,
                memory=memory,
                teacher_key=spec.key,
                current_runtime=spec.runtime,
                n_rows=len(entries),
                max_rows=int(args.expand_rows),
            )

        decision = resolve_decision(
            screen_decision,
            expanded_decision,
            selected_rows=len(entries),
            expand_rows=int(args.expand_rows),
        )
        report = {
            "format": REPORT_FORMAT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "teacher": spec.manifest_dict(),
            "runtime_identity": runtime.identity(
                token_chunk=args.token_chunk, vocab_chunk=args.vocab_chunk
            ),
            "reference": {
                "kind": (
                    "resident-bf16"
                    if spec.key == "e4b"
                    else "layer-streamed-bf16"
                ),
                "note": (
                    "E4B bf16 is its own reference: the fidelity metrics are a "
                    "self-consistency/allocation check, not weight-fidelity "
                    "evidence."
                    if spec.key == "e4b" and spec.runtime == "bf16"
                    else "bf16 reference on the same checkpoint and head policy"
                ),
                "checkpoint": spec.model_id,
                "revision": spec.revision,
            },
            "corpus": {
                "path": str(args.rows),
                "note": DEV_CORPUS_NOTE,
                "registered_default": args.rows == DEFAULT_P1LITE_CORPUS,
                "kind": (
                    "p1lite-dapo"
                    if args.rows == DEFAULT_P1LITE_CORPUS
                    else "explicit"
                ),
                "sha256": _file_sha256(args.rows),
                "stage0_disjointness": corpus_disjointness,
                "screen_rows": len(screen_selected),
                "expand_rows": int(args.expand_rows),
                "selected_rows": len(entries),
                "seed": int(args.seed),
                "chat_template_sha256": chat_template_sha256(tokenizer),
            },
            "disjointness": {
                **disjointness,
                "excluded_row_examples": [
                    {
                        "source": row.source,
                        "item_ids": list(row.item_ids[:3]),
                    }
                    for row in excluded[:5]
                ],
            },
            "rows": [
                {
                    "row_id": entry["row_id"],
                    "length": entry["length"],
                    "stratum": entry["stratum"],
                    "source": entry["source"],
                    "tokens": int(entry["metrics"]["kl"].shape[0]),
                }
                for entry in entries
            ],
            "thresholds": dict(ACCEPTANCE_THRESHOLDS),
            "metrics": {
                "screen": screen_overall,
                "expanded": expanded_overall,
            },
            "acceptance": {
                "screen": screen_decision.to_dict(),
                "expanded": expanded_decision.to_dict() if expanded_decision else None,
            },
            "decision": decision.to_dict(),
            "memory": memory,
        }
        (args.out / "qualification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.out / "qualification.md").write_text(
            render_markdown(report) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["decision"], indent=2, sort_keys=True))
        return 0 if decision.passed and decision.decision == "pass" else 2
    except NonFiniteTeacherRow as exc:
        # Deterministic config/data failure: a retry reproduces it, so exit 2
        # instead of the orchestrator's generic retry loop.
        print(f"error: teacher qualification refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    finally:
        if heartbeat is not None:
            heartbeat.stop()


if __name__ == "__main__":
    sys.exit(main())
