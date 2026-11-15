"""Precompute offline teacher top-k logits for the Stage 1 heal.

The teacher (E4B bf16 by default, with int8 as its registered fallback, or the
12B QAT q4_0 unquantized checkpoint in NF4; all pinned in
``gemmaloop.model.snapshot``) is teacher-forced on O's own Stage 0 samples,
rendered with the pinned E2B chat template.  For every labelled assistant token
the script stores the top-64 next-token probabilities (float16), their
vocabulary indices (int32) and the remaining tail mass (float32).  The
``<glitch_pruned_249788>`` logit is masked and the distribution renormalised,
as registered in prereg-v1.

Data layout (all atomic, resumable per shard)::

    <out>/
      teacher_manifest.json          # teacher identity + data sha + shard sha256
      shard-00000.pt                 # rows grouped by --shard-rows
      ...

The head projection is chunked over token positions and vocabulary, so the
full ``[tokens, 262144]`` logits tensor is never materialised and a 12B nf4
teacher (7-8 GB of weights) fits the 12 GiB card.  The script takes the project
GPU lock, caps the caching allocator and writes a heartbeat, like the other GPU
jobs.

Smoke run (few rows, into a separate directory)::

    uv run python scripts/precompute_teacher_topk.py \
        --teacher e4b --data artifacts/data/heal \
        --out artifacts/data/heal/teacher-e4b-smoke --max-rows 8

Full run (12M tokens, ~4.7 GB of shards)::

    uv run python scripts/precompute_teacher_topk.py \
        --teacher e4b --data artifacts/data/heal \
        --out artifacts/data/heal/teacher-e4b

Resume is automatic: an existing shard whose SHA-256 matches the manifest is
skipped, a torn or stale shard is recomputed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gemmaloop.data.teacher_topk import (  # noqa: E402
    DEFAULT_MASK_TOKEN,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    NonFiniteTeacherRow,
    ShardEntry,
    TeacherManifest,
    TeacherRow,
    TeacherSpec,
    assert_teacher_tokenizer_matches,
    compute_teacher_row,
    file_sha256,
    input_ids_sha256,
    teacher_spec,
    tokenizer_vocab_sha256,
    vocab_ex_mask_sha256,
    write_teacher_shard,
)
from gemmaloop.gpulock import require_gpu_lock  # noqa: E402
from gemmaloop.teacher.runtime import (  # noqa: E402
    DEFAULT_TOKEN_CHUNK,
    DEFAULT_VOCAB_CHUNK,
)
from gemmaloop.heartbeat import Heartbeat  # noqa: E402
from gemmaloop.model.snapshot import (  # noqa: E402
    E2B_REVISION,
    chat_template_sha256,
    model_revision,
    teacher_revision,
)
from gemmaloop.train.labels import assistant_labels  # noqa: E402
from gemmaloop.train.perf import cap_cuda_allocator  # noqa: E402

LOGGER = logging.getLogger("precompute_teacher_topk")

MANIFEST_NAME = "teacher_manifest.json"
DEFAULT_MAX_LENGTH = 8192


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedRow:
    """One rendered row: exact student ids and assistant-only labels."""

    row_id: int
    input_ids: list[int]
    labels: list[int]
    tokens: int = 0

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same length")
        if self.tokens == 0:
            object.__setattr__(
                self,
                "tokens",
                sum(1 for label in self.labels if label != -100),
            )


def load_rows(data: str | Path) -> list[dict[str, Any]]:
    """Read ``train.jsonl`` from a data directory or a direct JSONL path."""

    path = Path(data)
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
                raise ValueError(
                    f"{path}:{line_number}: expected a row with 'messages'"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def read_data_manifest(data: str | Path) -> tuple[dict[str, Any], str]:
    """Return ``(manifest payload, sha256)``; the sha is mandatory."""

    path = Path(data)
    base = path if path.is_dir() else path.parent
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training data manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = payload.get("sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"{manifest_path} has no non-empty 'sha256'")
    return payload, sha


def prepare_rows(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    *,
    max_rows: int | None = None,
    label_fn: Callable[..., dict[str, list[int]]] = assistant_labels,
) -> list[PreparedRow]:
    """Render every row with the pinned chat template and assistant labels."""

    limit = len(rows) if max_rows is None else min(len(rows), int(max_rows))
    prepared: list[PreparedRow] = []
    for row_id in range(limit):
        row = rows[row_id]
        output = label_fn(
            tokenizer,
            row["messages"],
            int(max_length),
            tools=row.get("tools"),
        )
        prepared.append(
            PreparedRow(
                row_id=row_id,
                input_ids=[int(value) for value in output["input_ids"]],
                labels=[int(value) for value in output["labels"]],
            )
        )
    return prepared


# ---------------------------------------------------------------------------
# Teacher loading (never exercised by CPU tests)
# ---------------------------------------------------------------------------


@dataclass
class LoadedTeacher:
    """The parts of a teacher the precompute needs."""

    spec: TeacherSpec
    model: Any
    tokenizer: Any
    text_model: Any
    weight: torch.Tensor
    softcap: float | None = None

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs.last_hidden_state.detach()


def _find_text_model(model: Any) -> Any:
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(getattr(inner, "language_model", None))
        candidates.append(inner)
    candidates.append(getattr(model, "language_model", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "forward"):
            return candidate
    raise ValueError("cannot locate the teacher's text model")


def _find_lm_head_weight(model: Any) -> torch.Tensor:
    head = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
    if head is None:
        for name in ("lm_head", "output"):
            candidate = getattr(model, name, None)
            if candidate is not None and hasattr(candidate, "weight"):
                head = candidate
                break
    if head is None or not hasattr(head, "weight"):
        raise ValueError("cannot locate the teacher's LM head")
    return head.weight


def _nf4_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=[".*vision_tower.*", ".*embed_vision.*", "lm_head"],
    )


def _int8_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=[".*vision_tower.*", ".*embed_vision.*", "lm_head"],
    )


def _repair_int8_device_state(model: Any, device: str | torch.device) -> None:
    """Repair bitsandbytes attributes missed by a whole-model ``_apply``."""
    from bitsandbytes.nn import Linear8bitLt

    for module in model.modules():
        if isinstance(module, Linear8bitLt):
            if module.weight.CB is not None:
                module.weight.CB = module.weight.data
            if module.weight.SCB is not None:
                module.weight.SCB = module.weight.SCB.to(device)
            for name in ("CB", "SCB"):
                value = getattr(module.state, name)
                if value is not None:
                    setattr(module.state, name, value.to(device))


def _assert_int8_on_device(model: Any, device: str | torch.device) -> None:
    """Fail before inference if any int8 tensor is on the wrong device."""
    from bitsandbytes.nn import Linear8bitLt

    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, Linear8bitLt):
            continue
        count += 1
        tensors = {
            "weight.data": module.weight.data,
            "weight.CB": module.weight.CB,
            "weight.SCB": module.weight.SCB,
            "state.CB": module.state.CB,
            "state.SCB": module.state.SCB,
        }
        for field, tensor in tensors.items():
            if tensor is not None and tensor.device != target:
                raise RuntimeError(
                    f"int8 device state mismatch in {name or '<root>'}.{field}: "
                    f"expected {target}, got {tensor.device}"
                )
    if count == 0:
        raise RuntimeError("int8 device state check found no Linear8bitLt modules")


def _offload_unused_vision(model: Any) -> None:
    """Keep the text-only teacher's vision tower off the GPU.

    The vision tower and its projector are never used by the export; E4B bf16
    leaves them on the GPU by default, which costs headroom for no benefit.
    """

    inner = getattr(model, "model", None)
    if inner is None:
        return
    for name in ("vision_tower", "embed_vision"):
        module = getattr(inner, name, None)
        if module is not None:
            module.to(torch.device("cpu"))


def _freeze_all(model: Any) -> None:
    model.eval()
    model.requires_grad_(False)


def _load_12b_qat(spec: TeacherSpec, device: str) -> LoadedTeacher:
    """Load the 12B QAT q4_0 unquantized checkpoint with nf4 weights.

    Assumptions (pinned by prereg §11 and the progress AUTHOR/Ruling lines):
    ``google/gemma-4-12B-it-qat-q4_0-unquantized`` is
    ``Gemma4UnifiedForConditionalGeneration`` (48 layers, hidden 3840, no
    PLE/KV sharing); its text stack is reachable as ``model.model.language_model``
    (the fallbacks in :func:`_find_text_model` cover the other wrappers), and
    ``lm_head`` is kept unquantized by the bitsandbytes skip list so its weight
    indexes the same 262,144-token vocabulary as E2B except the masked id.
    The audio tower is dropped before materialising the weights.
    """

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision)
    if getattr(config, "audio_config", None) is not None:
        config.audio_config = None
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        quantization_config=_nf4_config(),
        device_map="cpu",
    )
    model.to(torch.device(device))
    _offload_unused_vision(model)
    _freeze_all(model)
    processor = AutoProcessor.from_pretrained(spec.model_id, revision=spec.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    text_model = _find_text_model(model)
    return LoadedTeacher(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        text_model=text_model,
        weight=_find_lm_head_weight(model),
        softcap=getattr(text_model.config, "final_logit_softcapping", None),
    )


def _load_e4b_bf16(spec: TeacherSpec, device: str) -> LoadedTeacher:
    """Regular E4B in bf16 with CPU PLE and the unused vision tower on CPU."""

    from gemmaloop.model.load import ModelSpec, load_model

    model, processor, _plan = load_model(ModelSpec(size="e4b", quant="bf16"))
    _offload_unused_vision(model)
    _freeze_all(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    text_model = _find_text_model(model)
    return LoadedTeacher(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        text_model=text_model,
        weight=_find_lm_head_weight(model),
        softcap=getattr(text_model.config, "final_logit_softcapping", None),
    )


def _load_e4b_int8(spec: TeacherSpec, device: str) -> LoadedTeacher:
    """LLM.int8 fallback for E4B, preserving CPU PLE and skipping the tower.

    This mirrors :func:`gemmaloop.model.load.load_model`'s NF4 staging: the
    text stack is quantized on CPU, the PLE table is detached before the model
    move, and the bf16 embedding/head stay unquantized.  Bitsandbytes' int8
    CPU-quantized int8 auxiliary state is repaired after the model move.
    """

    from torch import nn
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )
    from gemmaloop.model.load import _CPUOffloadedEmbedding

    processor = AutoProcessor.from_pretrained(spec.model_id, revision=spec.revision)
    config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision)
    if getattr(config, "audio_config", None) is not None:
        config.audio_config = None
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation={"text_config": "sdpa", "vision_config": "sdpa"},
        quantization_config=_int8_config(),
        device_map="cpu",
    )
    text_model = _find_text_model(model)
    ple = text_model.embed_tokens_per_layer
    ple.weight.requires_grad_(False)
    text_model.embed_tokens_per_layer = nn.Identity()
    model.to(torch.device(device))
    _repair_int8_device_state(model, device)
    _assert_int8_on_device(model, device)
    text_model.embed_tokens_per_layer = _CPUOffloadedEmbedding(
        ple, torch.device(device)
    )
    _offload_unused_vision(model)
    _freeze_all(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    return LoadedTeacher(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        text_model=text_model,
        weight=_find_lm_head_weight(model),
        softcap=getattr(text_model.config, "final_logit_softcapping", None),
    )


def load_teacher(
    name: str, device: str = "cuda", runtime: str | None = None
) -> LoadedTeacher:
    """Load one registered teacher runtime (E4B bf16/int8, 12B QAT nf4).

    The runtime is part of the registered recipe, so an unknown or
    unimplemented runtime fails closed.  Every returned teacher is in ``eval``
    with all parameters frozen; the production pass additionally runs under
    inference mode.
    """

    if torch.device(device).type != "cuda":
        raise ValueError(
            "registered teacher runtimes load on CUDA only; CPU device support "
            "is provided by the CPU tests' fake models, not by this loader"
        )
    spec = teacher_spec(name, runtime=runtime)
    if spec.key == "e4b" and spec.runtime == "bf16":
        return _load_e4b_bf16(spec, device)
    if spec.key == "e4b" and spec.runtime == "int8":
        return _load_e4b_int8(spec, device)
    if spec.key == "12b_qat" and spec.runtime == "nf4":
        return _load_12b_qat(spec, device)
    raise ValueError(
        f"unsupported teacher runtime {spec.key}/{spec.runtime}"
    )


def load_student_tokenizer(name: str = "google/gemma-4-E2B-it"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, revision=model_revision("e2b"))


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


def _manifest_matches(
    manifest: TeacherManifest,
    *,
    spec: TeacherSpec,
    data_manifest_sha256: str,
    top_k: int,
    max_length: int,
    mask_token: str,
    max_rows: int | None,
    expected_template_sha: str,
    expected_tokenizer_sha: str,
    expected_lm_head_rows: int,
    token_chunk: int,
    vocab_chunk: int,
) -> bool:
    return (
        manifest.teacher.get("key") == spec.key
        and manifest.teacher.get("revision") == spec.revision
        and manifest.teacher.get("runtime") == spec.runtime
        and manifest.runtime_identity
        == spec.runtime_identity(
            token_chunk=int(token_chunk), vocab_chunk=int(vocab_chunk)
        )
        and manifest.data_manifest_sha256 == data_manifest_sha256
        and int(manifest.top_k) == int(top_k)
        and int(manifest.teacher.get("max_length", -1)) == int(max_length)
        and manifest.mask_token == mask_token
        and manifest.max_rows == max_rows
        and manifest.chat_template_sha256 == expected_template_sha
        and manifest.tokenizer_sha256 == expected_tokenizer_sha
        and int(manifest.lm_head_rows) == int(expected_lm_head_rows)
    )


def _freeze_teacher(loaded: LoadedTeacher) -> None:
    """Freeze every teacher tensor before the no-grad production pass.

    ``load_model`` already freezes PLE and vision, but the bf16 embedding/head
    and the NF4 skip-listed head can still carry ``requires_grad=True``.
    Inference mode makes this harmless; freezing is the additional explicit
    invariant required by review X3-C1.
    """

    model = getattr(loaded, "model", None)
    if model is not None and hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    weight = getattr(loaded, "weight", None)
    if isinstance(weight, torch.Tensor):
        weight.requires_grad_(False)


@torch.inference_mode()
def precompute_prepared(
    prepared: Sequence[PreparedRow],
    loaded: LoadedTeacher,
    *,
    out: str | Path,
    data_manifest_sha256: str,
    top_k: int = DEFAULT_TOP_K,
    max_length: int = DEFAULT_MAX_LENGTH,
    mask_token: str = DEFAULT_MASK_TOKEN,
    mask_token_id: int | None = None,
    shard_rows: int = 32,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
    heartbeat: Heartbeat | None = None,
    max_rows: int | None = None,
    expected_lm_head_rows: int | None = 262144,
    render_tokenizer=None,
) -> dict[str, int]:
    """Write all missing shards and return ``{computed, skipped, rows, tokens}``.

    ``loaded.weight`` may live on any device; hidden states are produced row by
    row (batch 1) and the LM-head projection is chunked inside
    :func:`compute_teacher_row`.
    """

    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    _freeze_teacher(loaded)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    render_tokenizer = loaded.tokenizer if render_tokenizer is None else render_tokenizer
    teacher_tokenizer_sha = tokenizer_vocab_sha256(loaded.tokenizer)
    render_template_sha = chat_template_sha256(render_tokenizer)
    student_vocab_sha = tokenizer_vocab_sha256(render_tokenizer)
    teacher_vocab_ex_mask_sha = vocab_ex_mask_sha256(loaded.tokenizer)

    existing: TeacherManifest | None = None
    if manifest_path.is_file():
        try:
            existing = TeacherManifest.read(manifest_path)
        except Exception:  # torn/corrupt manifest -> rebuild below
            existing = None
        if existing is not None and not _manifest_matches(
            existing,
            spec=loaded.spec,
            data_manifest_sha256=data_manifest_sha256,
            top_k=top_k,
            max_length=max_length,
            mask_token=mask_token,
            max_rows=max_rows,
            expected_template_sha=render_template_sha,
            expected_tokenizer_sha=student_vocab_sha,
            expected_lm_head_rows=int(loaded.weight.shape[0]),
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        ):
            existing = None

    completed: dict[int, ShardEntry] = {}
    if existing is not None:
        for entry in existing.shards:
            path = output / entry.file
            if path.is_file() and file_sha256(path) == entry.sha256:
                completed[int(entry.index)] = entry

    usable = (
        int(loaded.weight.shape[0])
        - (1 if mask_token_id is not None else 0)
    )
    if top_k > usable:
        raise ValueError(
            f"top_k={top_k} does not fit the teacher vocabulary "
            f"({loaded.weight.shape[0]} tokens, {usable} usable after masking)"
        )

    if (
        expected_lm_head_rows is not None
        and int(loaded.weight.shape[0]) != int(expected_lm_head_rows)
    ):
        raise ValueError(
            "teacher LM-head rows must be the registered "
            f"{int(expected_lm_head_rows)}-token vocabulary, got "
            f"{int(loaded.weight.shape[0])}"
        )
    shards: list[ShardEntry] = []
    computed = 0
    skipped = 0
    tokens_total = 0
    device = loaded.weight.device

    def write_manifest(*, complete: bool) -> TeacherManifest:
        manifest = TeacherManifest(
            teacher={
                **loaded.spec.manifest_dict(),
                "max_length": int(max_length),
            },
            data_manifest_sha256=str(data_manifest_sha256),
            top_k=int(top_k),
            temperature=DEFAULT_TEMPERATURE,
            mask_token=str(mask_token),
            mask_token_id=None if mask_token_id is None else int(mask_token_id),
            rows_total=len(prepared),
            tokens_total=int(tokens_total),
            shards=tuple(sorted(shards, key=lambda item: item.index)),
            max_rows=None if max_rows is None else int(max_rows),
            complete=bool(complete),
            chat_template_sha256=render_template_sha,
            tokenizer_sha256=student_vocab_sha,
            render_template_sha256=render_template_sha,
            student_vocab_sha256=student_vocab_sha,
            teacher_vocab_sha256=teacher_tokenizer_sha,
            teacher_vocab_ex_mask_sha256=teacher_vocab_ex_mask_sha,
            lm_head_rows=int(loaded.weight.shape[0]),
            runtime_identity=loaded.spec.runtime_identity(
                token_chunk=token_chunk, vocab_chunk=vocab_chunk
            ),
        )
        manifest.write(manifest_path)
        return manifest

    # Write an initial incomplete manifest so a crash before the first shard
    # still records the run identity and is never mistaken for a finished set.
    write_manifest(complete=False)
    for shard_index, start in enumerate(range(0, len(prepared), shard_rows)):
        group = prepared[start : start + shard_rows]
        filename = f"shard-{shard_index:05d}.pt"
        entry = completed.get(shard_index)
        if (
            entry is not None
            and entry.file == filename
            and tuple(entry.row_ids) == tuple(row.row_id for row in group)
        ):
            shards.append(entry)
            skipped += 1
            tokens_total += int(entry.tokens)
        else:
            rows: list[TeacherRow] = []
            for row in group:
                if not row.input_ids or all(label == -100 for label in row.labels):
                    teacher_row = TeacherRow(
                        row_id=row.row_id,
                        positions=torch.zeros(0, dtype=torch.int32),
                        indices=torch.zeros((0, top_k), dtype=torch.int32),
                        probs=torch.zeros((0, top_k), dtype=torch.float16),
                        tail=torch.zeros(0, dtype=torch.float32),
                        input_ids_sha256=input_ids_sha256(row.input_ids),
                    )
                else:
                    input_ids = torch.tensor(
                        [row.input_ids], dtype=torch.long, device=device
                    )
                    hidden = loaded.hidden(input_ids)[0].float()
                    teacher_row = compute_teacher_row(
                        hidden,
                        loaded.weight,
                        torch.tensor(row.labels, dtype=torch.long, device=device),
                        row_id=row.row_id,
                        input_ids=row.input_ids,
                        top_k=top_k,
                        mask_token_id=mask_token_id,
                        softcap=loaded.softcap,
                        token_chunk=token_chunk,
                        vocab_chunk=vocab_chunk,
                    )
                rows.append(teacher_row)
            digest = write_teacher_shard(output / filename, rows)
            entry = ShardEntry(
                index=shard_index,
                file=filename,
                sha256=digest,
                row_ids=tuple(row.row_id for row in group),
                tokens=sum(row.tokens for row in rows),
            )
            shards.append(entry)
            computed += 1
            tokens_total += entry.tokens
            if heartbeat is not None:
                heartbeat.update(
                    items_done=len(shards),
                    items_total=(len(prepared) + shard_rows - 1) // shard_rows,
                    tokens=entry.tokens,
                )
        # Per-shard atomic manifest entry: a driver reset resumes at the next
        # shard instead of losing the whole run.
        write_manifest(complete=False)

    shards.sort(key=lambda item: item.index)
    write_manifest(complete=True)
    return {
        "computed": computed,
        "skipped": skipped,
        "rows": len(prepared),
        "tokens": int(tokens_total),
        "shards": len(shards),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        default="e4b",
        help="registered teacher: e4b or 12b_qat",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        help="registered teacher runtime (e4b: bf16/int8; 12b_qat: nf4); "
        "defaults to the frozen qualified runtime from "
        "qualification/teacher_runtime_freeze.json and is required when no "
        "freeze entry exists",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="smoke mode: render and precompute only the first N rows",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--shard-rows", type=int, default=32)
    parser.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--token-chunk", type=int, default=DEFAULT_TOKEN_CHUNK)
    parser.add_argument("--vocab-chunk", type=int, default=DEFAULT_VOCAB_CHUNK)
    parser.add_argument("--vram-cap", type=float, default=0.90)
    parser.add_argument("--vram-free-margin-gib", type=float, default=0.5)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--skip-gpu-lock", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def resolve_precompute_runtime(
    teacher: str,
    runtime: str | None,
    *,
    freeze_path: str | Path | None = None,
) -> str:
    """Resolve the runtime fail-closed; never silently use the primary one.

    ``--runtime`` always wins.  Otherwise the frozen qualified-runtime record
    is the only source.  A hand run for E4B must not fall back to bf16 just
    because it is the registered primary: bf16 failed qualification and the
    harness/train would refuse its shards after the GPU hours were spent.
    """

    if runtime:
        return str(runtime)
    from gemmaloop.teacher.runtime_freeze import (
        TeacherRuntimeFreezeError,
        default_teacher_runtime_freeze_path,
        harness_teacher_key,
        load_teacher_runtime_freeze,
    )

    source = (
        default_teacher_runtime_freeze_path()
        if freeze_path is None
        else Path(freeze_path)
    )
    try:
        freeze = load_teacher_runtime_freeze(source)
        return freeze.runtime_for(harness_teacher_key(teacher))
    except TeacherRuntimeFreezeError as exc:
        raise ValueError(
            f"--runtime is required: no frozen qualified runtime is available "
            f"for teacher {teacher!r} from {source} ({exc}). Pass --runtime "
            "or build the freeze with scripts/freeze_teacher_runtime.py."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    _data_payload, data_sha = read_data_manifest(args.data)
    rows = load_rows(args.data)
    manifest_payload = _data_payload
    max_length = args.max_length
    if max_length is None:
        max_length = int(manifest_payload.get("max_length") or DEFAULT_MAX_LENGTH)
    if max_length <= 0:
        raise ValueError("--max-length must be positive")

    if not args.skip_gpu_lock:
        require_gpu_lock()
    cap_cuda_allocator(args.vram_cap, args.vram_free_margin_gib)

    tokenizer = load_student_tokenizer(args.tokenizer)
    prepared = prepare_rows(
        tokenizer,
        rows,
        max_length,
        max_rows=args.max_rows,
    )
    try:
        runtime = resolve_precompute_runtime(args.teacher, args.runtime)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    loaded = load_teacher(args.teacher, device="cuda", runtime=runtime)

    if not args.skip_tokenizer_check:
        check = assert_teacher_tokenizer_matches(
            tokenizer, loaded.tokenizer, mask_token=args.mask_token
        )
        LOGGER.info(
            "tokenizers match except the masked id: mask_id=%s vocab=%s",
            check["mask_token_id"],
            check["vocab_size"],
        )
    else:
        converted = getattr(loaded.tokenizer, "convert_tokens_to_ids", None)
        check = {
            "mask_token_id": (
                int(converted(args.mask_token))
                if callable(converted)
                else None
            )
        }
    mask_token_id = check.get("mask_token_id")

    heartbeat = None
    if args.heartbeat is not None:
        heartbeat = Heartbeat(
            args.heartbeat, run=f"teacher-{loaded.spec.key}", phase="precompute"
        ).start()
    try:
        result = precompute_prepared(
            prepared,
            loaded,
            out=args.out,
            data_manifest_sha256=data_sha,
            top_k=args.top_k,
            max_length=max_length,
            mask_token=args.mask_token,
            mask_token_id=mask_token_id,
            shard_rows=args.shard_rows,
            token_chunk=args.token_chunk,
            vocab_chunk=args.vocab_chunk,
            heartbeat=heartbeat,
            max_rows=args.max_rows,
            render_tokenizer=tokenizer,
        )
    except NonFiniteTeacherRow as exc:
        # Deterministic data/config failure: retrying reloads the teacher and
        # reproduces it.  Exit 2 prevents the orchestrator's retry loop.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if heartbeat is not None:
            heartbeat.stop()
    summary = {
        "teacher": loaded.spec.manifest_dict(),
        "runtime_identity": loaded.spec.runtime_identity(
            token_chunk=args.token_chunk, vocab_chunk=args.vocab_chunk
        ),
        "data_manifest_sha256": data_sha,
        "max_length": max_length,
        "mask_token": args.mask_token,
        "mask_token_id": mask_token_id,
        "out": str(args.out),
        **result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
