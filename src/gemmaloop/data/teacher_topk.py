"""Offline teacher top-k logits for the Stage 1 heal.

The Stage 1 objective (``gemmaloop.train.dial_objective``) consumes, for every
supervised assistant token, the teacher's top-``k`` next-token probabilities,
their vocabulary indices and one tail bucket.  This module builds and reads
that representation:

* :func:`topk_distribution_from_hidden` projects teacher hidden states through
  the LM head in token/vocabulary chunks, masks ``<glitch_pruned_249788>`` and
  renormalises, and returns ``(indices, probs, tail)``.  The full
  ``[tokens, vocab]`` logits tensor is never materialised.
* :func:`compute_teacher_row` writes the teacher distribution for target token
  ``i`` into slot ``i`` (the strict unshifted ``[B, S, K]`` contract: the
  objective applies the single left shift) and stores only labelled assistant
  tokens; the collator pads the rest with a valid filler distribution.
* :func:`write_teacher_shard` / :func:`read_teacher_shard` implement atomic,
  hashed shards; :class:`TeacherIndex` reads the manifest, verifies the data
  manifest SHA and lazily loads shards.
* :class:`TeacherTopKCollator` joins dataset rows (by ``row_id``) with their
  shard and emits the exact batch fields the objective consumes:
  ``teacher_topk_probs`` ``[B, S, K]`` (float16), ``teacher_topk_indices``
  ``[B, S, K]`` (int32) and ``teacher_tail_prob`` ``[B, S]`` (float32).

Nothing here downloads or loads a teacher; the precompute script owns that.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from gemmaloop.model.snapshot import (
    TEACHER_MODEL_IDS,
    TEACHER_REVISIONS,
    teacher_model_id,
    teacher_revision,
)

__all__ = [
    "DEFAULT_MASK_TOKEN",
    "DEFAULT_TOP_K",
    "GLITCH_TOKEN_ID",
    "TEACHER_FORMAT",
    "NonFiniteTeacherRow",
    "TeacherIndex",
    "TeacherManifestMismatch",
    "TeacherRow",
    "TeacherSpec",
    "TeacherTopKCollator",
    "assert_teacher_tokenizer_matches",
    "compute_teacher_row",
    "file_sha256",
    "input_ids_sha256",
    "read_teacher_shard",
    "teacher_spec",
    "tokenizer_vocab_sha256",
    "vocab_ex_mask_sha256",
    "topk_distribution_from_hidden",
    "validate_teacher_alignment",
    "validate_teacher_manifest_identity",
    "assert_teacher_row_finite",
    "write_json_atomic",
    "write_teacher_shard",
]

TEACHER_FORMAT = "gemmaloop-teacher-topk-v1"


class NonFiniteTeacherRow(ValueError):
    """A teacher row contains NaN/Inf: deterministic config/data failure."""


DEFAULT_TOP_K = 64
DEFAULT_MASK_TOKEN = "<glitch_pruned_249788>"
GLITCH_TOKEN_ID = 249788
DEFAULT_TEMPERATURE = 1.0

_SHARD_GLOB = "shard-*.pt"
_MANIFEST_NAME = "teacher_manifest.json"


# ---------------------------------------------------------------------------
# Teacher identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherSpec:
    """Pinned identity and registered runtime of an offline teacher."""

    key: str
    model_id: str
    revision: str
    runtime: str

    @property
    def quant(self) -> str:
        """Legacy name for the runtime (manifests now record ``runtime``)."""

        return self.runtime

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_id": self.model_id,
            "revision": self.revision,
            "runtime": self.runtime,
            # Kept so older readers that only knew ``quant`` see the same
            # value instead of an empty field; identity validation uses
            # ``runtime`` and the full runtime identity.
            "quant": self.runtime,
        }

    def runtime_identity(
        self,
        *,
        token_chunk: int | None = None,
        vocab_chunk: int | None = None,
    ) -> dict[str, Any]:
        from gemmaloop.teacher.runtime import runtime_identity

        return runtime_identity(
            self.key,
            self.runtime,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )


def teacher_spec(name: str, runtime: str | None = None) -> TeacherSpec:
    """Return the pinned teacher spec and registered runtime; fail closed."""

    key = str(name).lower()
    if key not in TEACHER_MODEL_IDS:
        raise ValueError(
            f"unknown teacher {name!r}; expected one of {tuple(TEACHER_MODEL_IDS)}"
        )
    from gemmaloop.teacher.runtime import teacher_runtime

    resolved = teacher_runtime(key, runtime).name
    return TeacherSpec(
        key=key,
        model_id=teacher_model_id(key),
        revision=teacher_revision(key),
        runtime=resolved,
    )


# ---------------------------------------------------------------------------
# Vocabulary check
# ---------------------------------------------------------------------------


def _vocab(tokenizer: Any) -> dict[str, int]:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise TypeError("tokenizers must expose get_vocab()")
    vocab = getter()
    if not isinstance(vocab, Mapping):
        raise TypeError("get_vocab() must return a mapping token -> id")
    return {str(token): int(index) for token, index in vocab.items()}


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    """Stable SHA-256 of the tokenizer vocabulary (token -> id, sorted)."""

    vocab = _vocab(tokenizer)
    payload = json.dumps(
        sorted(vocab.items()), separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vocab_ex_mask_sha256(
    tokenizer: Any, mask_token: str = DEFAULT_MASK_TOKEN
) -> str:
    """SHA-256 of the vocabulary with the masked id removed.

    Comparing this hash proves two vocabularies are identical except at the
    registered ``<glitch_pruned_249788>`` id.
    """

    vocab = _vocab(tokenizer)
    mask_id = GLITCH_TOKEN_ID
    for token, index in vocab.items():
        if token == mask_token:
            mask_id = int(index)
            break
    filtered = [
        (token, index) for token, index in vocab.items() if int(index) != mask_id
    ]
    payload = json.dumps(
        sorted(filtered), separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_ids_sha256(input_ids: Iterable[int]) -> str:
    """Stable SHA-256 of one rendered row's input ids."""

    payload = ",".join(str(int(value)) for value in input_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_teacher_tokenizer_matches(
    student: Any,
    teacher: Any,
    *,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> dict[str, Any]:
    """Assert the two tokenizers are identical except the masked id.

    The teacher must share the student vocabulary so a top-k teacher
    distribution indexes the student LM head directly.  ``<glitch_pruned_249788>``
    is the one registered exception.  Returns ``{"mask_token_id", "vocab_size"}``
    on success and raises :class:`ValueError` naming the first difference
    otherwise.
    """

    student_vocab = _vocab(student)
    teacher_vocab = _vocab(teacher)
    if len(student_vocab) != len(teacher_vocab):
        raise ValueError(
            "teacher and student tokenizers differ in size: "
            f"{len(teacher_vocab)} vs {len(student_vocab)}"
        )

    student_id_to_token = {index: token for token, index in student_vocab.items()}
    teacher_id_to_token = {index: token for token, index in teacher_vocab.items()}
    mask_ids = {
        index
        for index, token in student_id_to_token.items()
        if token == mask_token
    } | {
        index
        for index, token in teacher_id_to_token.items()
        if token == mask_token
    }
    if not mask_ids:
        raise ValueError(
            f"neither tokenizer contains the registered masked token {mask_token!r}"
        )
    if len(mask_ids) != 1:
        raise ValueError(
            f"{mask_token!r} has more than one vocabulary id: {sorted(mask_ids)}"
        )
    mask_id = next(iter(mask_ids))
    if student_id_to_token.get(mask_id) != teacher_id_to_token.get(mask_id):
        # The placeholder may legitimately differ, but both ids must exist.
        if (
            student_id_to_token.get(mask_id) != mask_token
            and teacher_id_to_token.get(mask_id) != mask_token
        ):
            raise ValueError(
                f"tokenizer id {mask_id} is neither side's {mask_token!r}: "
                f"{student_id_to_token.get(mask_id)!r} vs "
                f"{teacher_id_to_token.get(mask_id)!r}"
            )

    for token, index in student_vocab.items():
        if index == mask_id:
            continue
        teacher_index = teacher_vocab.get(token)
        if teacher_index != index:
            raise ValueError(
                f"teacher tokenizer differs at {token!r}: student id {index}, "
                f"teacher id {teacher_index}"
            )
    for token, index in teacher_vocab.items():
        if index == mask_id:
            continue
        if student_vocab.get(token) != index:
            raise ValueError(
                f"teacher tokenizer has an extra id at {token!r} (id {index})"
            )
    return {"mask_token_id": int(mask_id), "vocab_size": len(student_vocab)}


# ---------------------------------------------------------------------------
# Top-k computation
# ---------------------------------------------------------------------------


def _apply_softcap(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return torch.tanh(logits / float(softcap)) * float(softcap)


def _masked_logits(
    logits: torch.Tensor,
    v0: int,
    mask_token_id: int | None,
) -> torch.Tensor:
    if mask_token_id is None:
        return logits
    local = int(mask_token_id) - int(v0)
    if 0 <= local < logits.shape[1]:
        logits = logits.clone()
        logits[:, local] = float("-inf")
    return logits


@torch.inference_mode()
def topk_distribution_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    mask_token_id: int | None = GLITCH_TOKEN_ID,
    softcap: float | None = None,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(indices int32 [N,K], probs float16 [N,K], tail float32 [N])``.

    ``hidden`` is ``[N, H]`` and ``weight`` is ``[V, H]``.  The masked token is
    removed from the distribution before renormalising, so the teacher never
    assigns mass to ``<glitch_pruned_249788>``.  Both the token and vocabulary
    dimensions are processed in chunks; the full logits tensor is never built.

    The whole projection runs under :func:`torch.inference_mode` and every
    returned tensor is detached: a teacher never needs autograd, and an
    unquantized ``requires_grad`` head must not retain a graph through the
    vocabulary tiles (review X3-C1).
    """

    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must both be 2-D")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError(
            f"hidden width {hidden.shape[1]} != weight width {weight.shape[1]}"
        )
    if token_chunk < 1 or vocab_chunk < 1:
        raise ValueError("token_chunk and vocab_chunk must be positive")
    n_tokens = int(hidden.shape[0])
    vocab = int(weight.shape[0])
    usable = vocab - (1 if mask_token_id is not None and 0 <= int(mask_token_id) < vocab else 0)
    if top_k < 1 or top_k > usable:
        raise ValueError(
            f"top_k={top_k} does not fit a vocabulary of {vocab} "
            f"({usable} usable after masking)"
        )

    device = hidden.device
    indices_out = torch.zeros((n_tokens, top_k), dtype=torch.int32, device=device)
    probs_out = torch.zeros((n_tokens, top_k), dtype=torch.float16, device=device)
    tail_out = torch.zeros(n_tokens, dtype=torch.float32, device=device)

    for t0 in range(0, n_tokens, token_chunk):
        t1 = min(t0 + token_chunk, n_tokens)
        h = hidden[t0:t1].float()
        n = t1 - t0
        best_values = torch.full(
            (n, top_k), float("-inf"), dtype=torch.float32, device=device
        )
        best_indices = torch.zeros((n, top_k), dtype=torch.long, device=device)
        running_max = torch.full((n,), float("-inf"), dtype=torch.float32, device=device)
        sum_exp = torch.zeros(n, dtype=torch.float32, device=device)

        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            tile = _apply_softcap(h @ weight[v0:v1].float().t(), softcap)
            tile = _masked_logits(tile, v0, mask_token_id)

            local_k = min(top_k, tile.shape[1])
            local_values, local_indices = torch.topk(tile, local_k, dim=1)
            local_indices = local_indices + v0
            values = torch.cat([best_values, local_values], dim=1)
            indices = torch.cat([best_indices, local_indices], dim=1)
            best_values, picked = torch.topk(values, top_k, dim=1)
            best_indices = torch.gather(indices, 1, picked)

            tile_max = tile.max(dim=1).values
            new_max = torch.maximum(running_max, tile_max)
            alpha = torch.exp(running_max - new_max)
            sum_exp = sum_exp * alpha + torch.exp(tile - new_max.unsqueeze(1)).sum(dim=1)
            running_max = new_max

        lse = running_max + torch.log(sum_exp)
        probs = torch.exp(best_values - lse.unsqueeze(1))
        # The tail is the teacher mass outside the top-k, computed from the
        # **fp32** probabilities before rounding: the fp16 top-k rounding error
        # must not be dumped into a bucket registered as "the remaining mass".
        tail = (1.0 - probs.sum(dim=1)).clamp(min=0.0)
        indices_out[t0:t1] = best_indices.to(torch.int32)
        probs_out[t0:t1] = probs.to(torch.float16)
        tail_out[t0:t1] = tail

    return (
        indices_out.detach(),
        probs_out.detach(),
        tail_out.detach(),
    )


# ---------------------------------------------------------------------------
# Rows and shards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRow:
    """Teacher top-k for one dataset row (labelled assistant tokens only)."""

    row_id: int
    positions: torch.Tensor  # int32 [n] target-token positions
    indices: torch.Tensor  # int32 [n, K]
    probs: torch.Tensor  # float16 [n, K]
    tail: torch.Tensor  # float32 [n]
    input_ids_sha256: str = ""

    def __post_init__(self) -> None:
        if self.positions.ndim != 1:
            raise ValueError("positions must be 1-D")
        n = int(self.positions.shape[0])
        if self.indices.shape != (n, self.probs.shape[1]):
            raise ValueError("indices must have shape [n, K]")
        if self.probs.shape != self.indices.shape:
            raise ValueError("probs must have shape [n, K]")
        if self.tail.shape != (n,):
            raise ValueError("tail must have shape [n]")

    @property
    def tokens(self) -> int:
        return int(self.positions.shape[0])

    def token_positions(self) -> list[int]:
        return [int(value) for value in self.positions.tolist()]

    def to_payload(self) -> dict[str, Any]:
        return {
            "row_id": int(self.row_id),
            "positions": self.positions.to(torch.int32),
            "indices": self.indices.to(torch.int32),
            "probs": self.probs.to(torch.float16),
            "tail": self.tail.to(torch.float32),
            "input_ids_sha256": str(self.input_ids_sha256),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TeacherRow":
        return cls(
            row_id=int(payload["row_id"]),
            positions=payload["positions"].to(torch.int32),
            indices=payload["indices"].to(torch.int32),
            probs=payload["probs"].to(torch.float16),
            tail=payload["tail"].to(torch.float32),
            input_ids_sha256=str(payload.get("input_ids_sha256", "")),
        )


def assert_teacher_row_finite(row: TeacherRow) -> None:
    """Fail closed if a precomputed teacher row contains NaN or Inf.

    bitsandbytes LLM.int8 casts bf16 activations to fp16 before quantising; a
    rare overflow can reach the stored top-k probabilities or tail.  A
    non-finite row must never be written and only surface later as a training
    abort.
    """

    for name, tensor in (("probs", row.probs), ("tail", row.tail)):
        if tensor.numel() == 0:
            continue
        if not bool(torch.isfinite(tensor).all().item()):
            raise NonFiniteTeacherRow(
                f"teacher row {row.row_id} has non-finite {name}; refusing "
                "to write it"
            )


@torch.inference_mode()
def compute_teacher_row(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    row_id: int,
    input_ids: Iterable[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
    mask_token_id: int | None = GLITCH_TOKEN_ID,
    softcap: float | None = None,
    ignore_index: int = -100,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> TeacherRow:
    """Teacher-forced top-k for ``labels`` from whole-row ``hidden`` states.

    ``hidden`` is ``[S, H]`` (the teacher's hidden state for the same input
    ids) and ``labels`` is ``[S]``.  Target token ``i`` is scored by the
    teacher state at ``i - 1`` and stored at slot ``i``: the row is **unshifted**
    and aligned 1:1 with ``labels``, exactly the ``[B, S, K]`` batch contract.
    The objective applies the only shift (``[:, 1:]``); position 0 is never a
    target because it predicts nothing.
    """

    if hidden.ndim != 2:
        raise ValueError("hidden must be [S, H]")
    labels = labels.reshape(-1)
    if int(hidden.shape[0]) != int(labels.shape[0]):
        raise ValueError(
            f"hidden has {int(hidden.shape[0])} positions but labels has "
            f"{int(labels.shape[0])}"
        )
    labelled = labels != int(ignore_index)
    positions = torch.nonzero(labelled, as_tuple=False).flatten()
    positions = positions[positions > 0]
    if positions.numel() == 0:
        empty = torch.zeros((0, top_k), dtype=torch.float32)
        return TeacherRow(
            row_id=int(row_id),
            positions=torch.zeros(0, dtype=torch.int32),
            indices=empty.to(torch.int32),
            probs=empty.to(torch.float16),
            tail=torch.zeros(0, dtype=torch.float32),
            input_ids_sha256=(
                "" if input_ids is None else input_ids_sha256(input_ids)
            ),
        )
    predictions = hidden[(positions - 1).to(hidden.device)]
    if not bool(torch.isfinite(predictions).all().item()):
        raise NonFiniteTeacherRow(
            f"teacher row {row_id} has a non-finite hidden state; refusing "
            "to write it"
        )
    indices, probs, tail = topk_distribution_from_hidden(
        predictions,
        weight,
        top_k=top_k,
        mask_token_id=mask_token_id,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    row = TeacherRow(
        row_id=int(row_id),
        positions=positions.detach().to(torch.int32).cpu(),
        indices=indices.detach().cpu(),
        probs=probs.detach().cpu(),
        tail=tail.detach().cpu(),
        input_ids_sha256=(
            "" if input_ids is None else input_ids_sha256(input_ids)
        ),
    )
    assert_teacher_row_finite(row)
    return row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of ``path`` (public wrapper for shard checks)."""

    return _sha256_file(Path(path))


def _atomic_replace(temporary: Path, target: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            import time

            time.sleep(0.02 * (attempt + 1))


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as sorted JSON, atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_teacher_shard(path: Path, rows: Sequence[TeacherRow]) -> str:
    """Write a shard atomically and return its SHA-256."""

    for row in rows:
        assert_teacher_row_finite(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": TEACHER_FORMAT,
        "rows": [row.to_payload() for row in rows],
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        _atomic_replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return digest


def read_teacher_shard(path: Path, *, mmap: bool = True) -> list[TeacherRow]:
    """Read one shard, memory-mapped when the file format supports it."""

    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True, mmap=mmap
        )
    except (RuntimeError, TypeError, ValueError):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != TEACHER_FORMAT:
        raise ValueError(f"{path} is not a {TEACHER_FORMAT} shard")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no row list")
    return [TeacherRow.from_payload(item) for item in rows]


# ---------------------------------------------------------------------------
# Manifest and index
# ---------------------------------------------------------------------------


class TeacherManifestMismatch(RuntimeError):
    """Raised when teacher shards do not belong to the requested data."""


@dataclass(frozen=True)
class ShardEntry:
    index: int
    file: str
    sha256: str
    row_ids: tuple[int, ...]
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "file": self.file,
            "sha256": self.sha256,
            "row_ids": list(self.row_ids),
            "tokens": int(self.tokens),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardEntry":
        return cls(
            index=int(payload["index"]),
            file=str(payload["file"]),
            sha256=str(payload["sha256"]),
            row_ids=tuple(int(value) for value in payload["row_ids"]),
            tokens=int(payload.get("tokens", 0)),
        )


@dataclass(frozen=True)
class TeacherManifest:
    teacher: Mapping[str, Any]
    data_manifest_sha256: str
    top_k: int
    temperature: float
    mask_token: str
    mask_token_id: int | None
    rows_total: int
    tokens_total: int
    shards: tuple[ShardEntry, ...] = ()
    max_rows: int | None = None
    complete: bool = True
    chat_template_sha256: str = ""
    tokenizer_sha256: str = ""
    render_template_sha256: str = ""
    student_vocab_sha256: str = ""
    teacher_vocab_sha256: str = ""
    teacher_vocab_ex_mask_sha256: str = ""
    lm_head_rows: int = 0
    runtime_identity: Mapping[str, Any] = field(default_factory=dict)
    format: str = TEACHER_FORMAT

    @property
    def teacher_key(self) -> str:
        return str(self.teacher.get("key", ""))

    @property
    def teacher_revision(self) -> str:
        return str(self.teacher.get("revision", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "teacher": dict(self.teacher),
            "data_manifest_sha256": self.data_manifest_sha256,
            "top_k": int(self.top_k),
            "temperature": float(self.temperature),
            "mask_token": self.mask_token,
            "mask_token_id": None if self.mask_token_id is None else int(self.mask_token_id),
            "rows_total": int(self.rows_total),
            "tokens_total": int(self.tokens_total),
            "max_rows": None if self.max_rows is None else int(self.max_rows),
            "complete": bool(self.complete),
            "chat_template_sha256": self.chat_template_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "render_template_sha256": self.render_template_sha256,
            "student_vocab_sha256": self.student_vocab_sha256,
            "teacher_vocab_sha256": self.teacher_vocab_sha256,
            "teacher_vocab_ex_mask_sha256": self.teacher_vocab_ex_mask_sha256,
            "lm_head_rows": int(self.lm_head_rows),
            "runtime_identity": dict(self.runtime_identity),
            "shards": [entry.to_dict() for entry in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherManifest":
        if payload.get("format") != TEACHER_FORMAT:
            raise ValueError(
                f"unsupported teacher manifest format {payload.get('format')!r}"
            )
        mask_id = payload.get("mask_token_id")
        return cls(
            teacher=dict(payload["teacher"]),
            data_manifest_sha256=str(payload["data_manifest_sha256"]),
            top_k=int(payload["top_k"]),
            temperature=float(payload.get("temperature", DEFAULT_TEMPERATURE)),
            mask_token=str(payload.get("mask_token", DEFAULT_MASK_TOKEN)),
            mask_token_id=None if mask_id is None else int(mask_id),
            rows_total=int(payload["rows_total"]),
            tokens_total=int(payload["tokens_total"]),
            shards=tuple(ShardEntry.from_dict(item) for item in payload.get("shards", ())),
            max_rows=None if payload.get("max_rows") is None else int(payload["max_rows"]),
            complete=bool(payload.get("complete", True)),
            chat_template_sha256=str(payload.get("chat_template_sha256", "")),
            tokenizer_sha256=str(payload.get("tokenizer_sha256", "")),
            render_template_sha256=str(
                payload.get(
                    "render_template_sha256",
                    payload.get("chat_template_sha256", ""),
                )
            ),
            student_vocab_sha256=str(
                payload.get(
                    "student_vocab_sha256",
                    payload.get("tokenizer_sha256", ""),
                )
            ),
            teacher_vocab_sha256=str(payload.get("teacher_vocab_sha256", "")),
            teacher_vocab_ex_mask_sha256=str(
                payload.get("teacher_vocab_ex_mask_sha256", "")
            ),
            lm_head_rows=int(payload.get("lm_head_rows", 0)),
            runtime_identity=dict(payload.get("runtime_identity") or {}),
        )

    def with_shards(self, shards: Sequence[ShardEntry]) -> "TeacherManifest":
        return TeacherManifest(
            teacher=self.teacher,
            data_manifest_sha256=self.data_manifest_sha256,
            top_k=self.top_k,
            temperature=self.temperature,
            mask_token=self.mask_token,
            mask_token_id=self.mask_token_id,
            rows_total=self.rows_total,
            tokens_total=sum(entry.tokens for entry in shards),
            shards=tuple(shards),
            max_rows=self.max_rows,
            complete=self.complete,
            chat_template_sha256=self.chat_template_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            render_template_sha256=self.render_template_sha256,
            student_vocab_sha256=self.student_vocab_sha256,
            teacher_vocab_sha256=self.teacher_vocab_sha256,
            teacher_vocab_ex_mask_sha256=self.teacher_vocab_ex_mask_sha256,
            lm_head_rows=self.lm_head_rows,
            runtime_identity=self.runtime_identity,
            format=self.format,
        )

    def write(self, path: Path) -> None:
        write_json_atomic(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "TeacherManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeacherManifestMismatch(
                f"cannot read teacher manifest {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TeacherManifestMismatch(f"{path} is not a JSON object")
        try:
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise TeacherManifestMismatch(
                f"invalid teacher manifest {path}: {exc}"
            ) from exc


@dataclass
class TeacherIndex:
    """Lazily loaded teacher shards joined to dataset rows by ``row_id``."""

    directory: Path
    manifest: TeacherManifest
    manifest_sha256: str = ""
    max_cached_shards: int = 4
    _row_to_shard: dict[int, int] = field(default_factory=dict, repr=False)
    _shard_cache: OrderedDict[int, dict[int, TeacherRow]] = field(
        default_factory=OrderedDict, repr=False
    )
    _verified_shards: set[int] = field(default_factory=set, repr=False)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        data_manifest_sha256: str | None = None,
        expect_top_k: int | None = None,
        verify_shards: bool = False,
    ) -> "TeacherIndex":
        path = Path(directory)
        manifest_path = path if path.is_file() else path / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise TeacherManifestMismatch(
                f"teacher manifest not found: {manifest_path}"
            )
        manifest = TeacherManifest.read(manifest_path)
        if not manifest.complete:
            raise TeacherManifestMismatch(
                f"teacher manifest {manifest_path} is incomplete; rerun the "
                "precompute to finish the remaining shards"
            )
        if data_manifest_sha256 is not None and (
            manifest.data_manifest_sha256 != str(data_manifest_sha256)
        ):
            raise TeacherManifestMismatch(
                "teacher shards were built for a different data manifest: "
                f"{manifest.data_manifest_sha256} != {data_manifest_sha256}"
            )
        if expect_top_k is not None and int(manifest.top_k) != int(expect_top_k):
            raise TeacherManifestMismatch(
                f"teacher top_k={manifest.top_k} but the run needs {expect_top_k}"
            )
        base = manifest_path.parent
        row_to_shard: dict[int, int] = {}
        verified: set[int] = set()
        for entry in manifest.shards:
            shard_path = base / entry.file
            if not shard_path.is_file():
                raise TeacherManifestMismatch(f"teacher shard missing: {shard_path}")
            if verify_shards:
                digest = _sha256_file(shard_path)
                if digest != entry.sha256:
                    raise TeacherManifestMismatch(
                        f"teacher shard {shard_path} sha256 mismatch"
                    )
                verified.add(int(entry.index))
            for row_id in entry.row_ids:
                if row_id in row_to_shard:
                    raise TeacherManifestMismatch(
                        f"teacher row {row_id} appears in more than one shard"
                    )
                row_to_shard[int(row_id)] = int(entry.index)
        if len(row_to_shard) != int(manifest.rows_total):
            raise TeacherManifestMismatch(
                f"teacher manifest lists {manifest.rows_total} rows but the "
                f"shards cover {len(row_to_shard)}"
            )
        return cls(
            directory=base,
            manifest=manifest,
            manifest_sha256=_sha256_file(manifest_path),
            _row_to_shard=row_to_shard,
            _verified_shards=verified,
        )

    def row_ids(self) -> set[int]:
        return set(self._row_to_shard)

    def _load_shard(self, index: int) -> dict[int, TeacherRow]:
        if index in self._shard_cache:
            self._shard_cache.move_to_end(index)
            self._trim_cache()
            return self._shard_cache[index]
        entry = next(
            (item for item in self.manifest.shards if int(item.index) == int(index)),
            None,
        )
        if entry is None:
            raise TeacherManifestMismatch(f"teacher shard {index} is not in the manifest")
        path = self.directory / entry.file
        if int(index) not in self._verified_shards:
            digest = _sha256_file(path)
            if digest != entry.sha256:
                raise TeacherManifestMismatch(
                    f"teacher shard {path} sha256 mismatch"
                )
            self._verified_shards.add(int(index))
        rows = read_teacher_shard(path)
        loaded = {int(row.row_id): row for row in rows}
        if set(loaded) != set(entry.row_ids):
            raise TeacherManifestMismatch(
                f"teacher shard {path} rows do not match its manifest entry"
            )
        self._shard_cache[index] = loaded
        self._shard_cache.move_to_end(index)
        self._trim_cache()
        return loaded

    def _trim_cache(self) -> None:
        while len(self._shard_cache) > max(1, int(self.max_cached_shards)):
            self._shard_cache.popitem(last=False)

    def row(self, row_id: int) -> TeacherRow:
        if int(row_id) not in self._row_to_shard:
            raise TeacherManifestMismatch(
                f"dataset row {row_id} has no teacher shard entry"
            )
        return self._load_shard(self._row_to_shard[int(row_id)])[int(row_id)]

    def batch(self, row_ids: Sequence[int], seq_len: int) -> dict[str, torch.Tensor]:
        """Build the ``[B, S, K]`` teacher batch fields.

        Positions without a stored teacher token get ``probs=0``,
        ``indices=0`` and ``tail=1`` so the objective's ignore mask (which uses
        the labels) still sees a valid distribution everywhere.
        """

        top_k = int(self.manifest.top_k)
        batch = len(row_ids)
        probs = torch.zeros((batch, int(seq_len), top_k), dtype=torch.float16)
        indices = torch.zeros((batch, int(seq_len), top_k), dtype=torch.int32)
        tail = torch.ones((batch, int(seq_len)), dtype=torch.float32)
        for row_number, row_id in enumerate(row_ids):
            row = self.row(int(row_id))
            if row.probs.shape[1] != top_k:
                raise TeacherManifestMismatch(
                    f"teacher row {row_id} has K={row.probs.shape[1]}, "
                    f"manifest says {top_k}"
                )
            positions = row.positions.to(torch.long)
            keep = positions < int(seq_len)
            positions = positions[keep]
            if positions.numel() == 0:
                continue
            probs[row_number, positions] = row.probs[keep]
            indices[row_number, positions] = row.indices[keep]
            tail[row_number, positions] = row.tail[keep]
        return {
            "teacher_topk_probs": probs,
            "teacher_topk_indices": indices,
            "teacher_tail_prob": tail,
        }


class TeacherTopKCollator:
    """Wrap a base collator and attach the teacher fields for each row."""

    def __init__(self, index: TeacherIndex, base_collator: Any) -> None:
        if not callable(base_collator):
            raise TypeError("base_collator must be callable")
        self.index = index
        self.base_collator = base_collator

    def __call__(self, features: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        cleaned: list[dict[str, Any]] = []
        row_ids: list[int] = []
        for feature in features:
            record = dict(feature)
            if "row_id" not in record:
                raise ValueError(
                    "teacher batches need a 'row_id' on every dataset record"
                )
            row_ids.append(int(record.pop("row_id")))
            cleaned.append(record)
        batch = dict(self.base_collator(cleaned))
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("the base collator did not return input_ids")
        batch.update(self.index.batch(row_ids, int(input_ids.shape[1])))
        # Same provenance columns as the tdial TeacherDataCollator contract.
        count = len(row_ids)
        batch["teacher_manifest_sha256"] = [self.index.manifest_sha256] * count
        batch["teacher_id"] = [str(self.index.manifest.teacher.get("key", ""))] * count
        batch["teacher_revision"] = [
            str(self.index.manifest.teacher.get("revision", ""))
        ] * count
        return batch


def validate_teacher_alignment(
    index: TeacherIndex,
    records: Iterable[Mapping[str, Any]],
    *,
    ignore_index: int = -100,
) -> None:
    """Refuse a teacher run whose stored positions miss supervised tokens."""

    for record in records:
        row_id = int(record["row_id"])
        labels = list(record["labels"])
        expected = [position for position, label in enumerate(labels) if label != ignore_index]
        expected = [position for position in expected if position > 0]
        row = index.row(row_id)
        if row.input_ids_sha256 and row.input_ids_sha256 != input_ids_sha256(
            record["input_ids"]
        ):
            raise TeacherManifestMismatch(
                f"teacher row {row_id} was rendered from different input ids"
            )
        actual = row.token_positions()
        if actual != expected:
            raise TeacherManifestMismatch(
                f"teacher row {row_id} covers positions {actual[:8]}... but the "
                f"rendered labels need {expected[:8]}... ({len(actual)} vs "
                f"{len(expected)} tokens)"
            )


def validate_teacher_manifest_identity(
    manifest: TeacherManifest,
    *,
    allow_smoke: bool = False,
    render_template_sha256: str | None = None,
    student_vocab_sha256: str | None = None,
    student_vocab_ex_mask_sha256: str | None = None,
    teacher_vocab_sha256: str | None = None,
    frozen_runtime: str | None = None,
) -> None:
    """Refuse a shard set that was not built by a registered teacher runtime.

    Checks the pinned teacher key/revision, the registered runtime and its
    complete identity (method/settings, compute dtype, skipped modules, head
    projection/accumulation policy, attention implementation, software versions
    and chunk sizes), the glitch mask, the 262144 LM-head vocabulary, the
    rendered template hash and the tokenizer hash.  There is no broad "any
    quant" whitelist: each teacher is bound to its registered runtime.
    """

    from gemmaloop.model.snapshot import TEACHER_REVISIONS
    from gemmaloop.teacher.runtime import registered_runtimes, validate_runtime_identity

    teacher = manifest.teacher
    key = str(teacher.get("key", ""))
    if key not in TEACHER_REVISIONS:
        raise TeacherManifestMismatch(f"unknown teacher key {key!r}")
    if str(teacher.get("revision", "")) != str(TEACHER_REVISIONS[key]):
        raise TeacherManifestMismatch(
            f"teacher revision {teacher.get('revision')!r} is not the pinned "
            f"{TEACHER_REVISIONS[key]}"
        )
    runtime = str(teacher.get("runtime", ""))
    allowed = registered_runtimes(key)
    if runtime not in allowed:
        raise TeacherManifestMismatch(
            f"teacher {key!r} runtime {runtime!r} is not registered; "
            f"expected one of {allowed}"
        )
    try:
        validate_runtime_identity(key, runtime, manifest.runtime_identity)
    except ValueError as exc:
        raise TeacherManifestMismatch(str(exc)) from exc
    if frozen_runtime is not None and runtime != str(frozen_runtime):
        raise TeacherManifestMismatch(
            f"teacher {key!r} shards were built with runtime {runtime!r}, but "
            f"the frozen qualified runtime is {frozen_runtime!r}"
        )
    if manifest.mask_token != DEFAULT_MASK_TOKEN:
        raise TeacherManifestMismatch(
            f"teacher mask token must be {DEFAULT_MASK_TOKEN!r}, got "
            f"{manifest.mask_token!r}"
        )
    if manifest.mask_token_id != GLITCH_TOKEN_ID:
        raise TeacherManifestMismatch(
            f"teacher mask id must be {GLITCH_TOKEN_ID}, got "
            f"{manifest.mask_token_id}"
        )
    if not allow_smoke and manifest.max_rows is not None:
        raise TeacherManifestMismatch(
            "a registered teacher manifest cannot be a --max-rows smoke build"
        )
    if int(manifest.lm_head_rows) != 262144:
        raise TeacherManifestMismatch(
            f"teacher LM-head rows must be 262144, got {manifest.lm_head_rows}"
        )
    required = {
        "render_template_sha256": manifest.render_template_sha256,
        "student_vocab_sha256": manifest.student_vocab_sha256,
        "teacher_vocab_sha256": manifest.teacher_vocab_sha256,
        "teacher_vocab_ex_mask_sha256": manifest.teacher_vocab_ex_mask_sha256,
    }
    empty = [name for name, value in required.items() if not value]
    if empty:
        raise TeacherManifestMismatch(
            "teacher manifest is missing required identity fields: "
            + ", ".join(empty)
        )
    if render_template_sha256 is not None:
        if manifest.render_template_sha256 != render_template_sha256:
            raise TeacherManifestMismatch(
                "teacher shards were rendered with a different chat template"
            )
    if student_vocab_sha256 is not None:
        if manifest.student_vocab_sha256 != student_vocab_sha256:
            raise TeacherManifestMismatch(
                "teacher shards were rendered with a different student tokenizer"
            )
    if student_vocab_ex_mask_sha256 is not None:
        if manifest.teacher_vocab_ex_mask_sha256 != student_vocab_ex_mask_sha256:
            raise TeacherManifestMismatch(
                "teacher vocabulary differs from the student outside the "
                "registered masked id"
            )
    if teacher_vocab_sha256 is not None:
        if manifest.teacher_vocab_sha256 != teacher_vocab_sha256:
            raise TeacherManifestMismatch(
                "teacher shards were built with a different teacher tokenizer"
            )
