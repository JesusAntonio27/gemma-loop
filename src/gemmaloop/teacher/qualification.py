"""Teacher distribution-fidelity metrics and the X3 section 7 acceptance gate.

The runtime is compared with a bf16 reference on the same checkpoint, token
IDs, mask, softcap, temperature and head policy.  All metric formulas here are
pure and CPU-testable against a direct full-softmax computation; the production
harness in ``scripts/qualify_teacher.py`` feeds them reference and runtime
hidden states and the fp32 vocabulary-chunked head.

Reported views (review X3 section 7):

* full forward KL and reverse KL, streamed in two vocabulary passes;
* union-support KL and the 65-way KL on the bf16 top-64 support;
* top-1 agreement overall and on confident reference tokens, top-64 overlap and
  bf16 mass captured by the runtime top-64;
* absolute tail-mass errors on the fixed reference support;
* entropy and teacher-forced log p(O's actual next token) differences;
* every metric again after fp16 top-k serialisation and the trainer's fp32
  renormalisation;
* thinking/answer spans and length strata, with a row-cluster bootstrap.

Thresholds are engineering tolerances from the review, not published universal
standards.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from gemmaloop.data.teacher_topk import (
    DEFAULT_TOP_K,
    GLITCH_TOKEN_ID,
    _apply_softcap,
    _masked_logits,
    topk_distribution_from_hidden,
)
from gemmaloop.teacher.runtime import REGISTERED_FALLBACK_RUNTIMES

__all__ = [
    "ACCEPTANCE_THRESHOLDS",
    "AcceptanceResult",
    "ProtectionError",
    "aggregate_metrics",
    "assert_disjoint",
    "canonical_dapo_id",
    "extract_row_ids",
    "filter_disjoint",
    "bootstrap_mean_ci",
    "default_protected_sources",
    "evaluate_acceptance",
    "extract_row_id",
    "load_protected_ids",
    "pair_token_metrics",
    "percentile",
    "resolve_decision",
    "select_development_rows",
    "stratify_indices",
    "token_spans",
]

_METRIC_TINY = 1e-30

PROTECTED_SOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "pilot_evaluation": (
        "postprocess/pilot_verified.jsonl",
        "pool/pilot_verified.jsonl",
        "pool/split_pilot_dev.jsonl",
    ),
    "temperature_fit": (
        "pool/split_pilot_tfit.jsonl",
        "postprocess/split_pilot_tfit.jsonl",
    ),
    "monitor": (
        "pool/split_monitor.jsonl",
        "pool/split_monitor_dev.jsonl",
        "postprocess/monitor.jsonl",
        "monitor.jsonl",
    ),
    "confirmatory": (
        "pool/split_heldout_test.jsonl",
        "pool/split_confirmatory.jsonl",
        "postprocess/split_heldout_test.jsonl",
        "confirmatory.jsonl",
    ),
}


class ProtectionError(RuntimeError):
    """A registered protected corpus or its manifest cannot be verified."""


def stable_problem_id(*parts: Any) -> str:
    """Match ``gemmaloop.stage0.sources._stable_id`` exactly."""

    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    )
    return digest.hexdigest()[:24]


def canonical_dapo_id(raw_id: Any) -> str:
    """Return the Stage 0 canonical id for a DAPO-Math-17k raw id."""

    return f"dapo:{stable_problem_id(raw_id)}"


def _row_values(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("id", "row_id", "problem_id", "item_id", "uuid"):
        value = row.get(key)
        if value not in (None, ""):
            values.add(str(value))
    for container_key in ("metadata", "extra_info"):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            for key in ("raw_id", "index", "id", "problem_id", "row_id"):
                value = container.get(key)
                if value not in (None, ""):
                    values.add(str(value))
    return values


def extract_row_ids(row: Any) -> set[str]:
    """Return every identity alias used for by-id disjointness.

    P1-lite rows carry the raw DAPO UUID under ``id``; the Stage 0 pool stores
    ``problem_id = "dapo:<stable_id(raw)>"`` and keeps the raw id under
    ``metadata.raw_id``.  Both sides therefore contribute the raw and canonical
    forms, so a real overlap cannot hide in different namespaces.
    """

    if isinstance(row, str):
        return {row} if row else set()
    if not isinstance(row, Mapping):
        return set()
    values = _row_values(row)
    source = str(row.get("source") or "").lower()
    is_dapo = (
        "dapo" in source
        or any(value.startswith("dapo:") for value in values)
        or str(row.get("problem_id", "")).startswith("dapo:")
    )
    aliases = set(values)
    if is_dapo:
        for value in values:
            if not value.startswith("dapo:"):
                aliases.add(canonical_dapo_id(value))
    return aliases


def extract_row_id(row: Any) -> str | None:
    """Return one stable alias used for disjointness, or ``None``."""

    aliases = extract_row_ids(row)
    return sorted(aliases)[0] if aliases else None


def default_protected_sources(root: str | Path) -> dict[str, tuple[Path, ...]]:
    """Return the registered protected-set paths under the Stage 0 root."""

    base = Path(root)
    return {
        name: tuple(base / pattern for pattern in patterns)
        for name, patterns in PROTECTED_SOURCE_PATTERNS.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_from_payload(payload: Any) -> set[str]:
    ids: set[str] = set()
    items: Iterable[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        if isinstance(payload.get("ids"), list):
            items = payload["ids"]
        elif isinstance(payload.get("row_ids"), list):
            items = payload["row_ids"]
        elif isinstance(payload.get("rows"), list):
            items = payload["rows"]
        elif isinstance(payload.get("problems"), list):
            items = payload["problems"]
        else:
            items = [payload]
    else:
        raise ProtectionError("protected manifest is not a JSON object or list")
    for item in items:
        aliases = extract_row_ids(item)
        if not aliases and isinstance(item, (int, float)):
            aliases = {str(item)}
        ids |= aliases
    return ids


def load_protected_ids(
    sources: Mapping[str, Sequence[str | Path]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Read every existing protected source; fail closed when one is unreadable.

    Missing files are recorded as ``absent`` (the sets do not exist yet).  An
    existing file that cannot be parsed, or that yields no row ids, is a
    registered-manifest failure and aborts the qualification.
    """

    report: dict[str, dict[str, Any]] = {}
    all_ids: dict[str, set[str]] = {}
    for name, paths in sources.items():
        existing: list[Path] = []
        for path in paths:
            candidate = Path(path)
            if candidate.exists() and not candidate.is_file():
                raise ProtectionError(
                    f"registered protected manifest is not a readable file: "
                    f"{candidate}"
                )
            if candidate.is_file():
                existing.append(candidate)
        used_paths: list[str] = []
        ids: set[str] = set()
        rows = 0
        for path in existing:
            try:
                text = path.read_text(encoding="utf-8")
                if path.suffix.lower() == ".json":
                    payload = json.loads(text)
                else:
                    # Split on ``\n`` only: U+2028/U+2029 are legal inside
                    # JSON strings and occur in real Stage 0 pool rows.
                    payload = [
                        json.loads(line.rstrip("\r"))
                        for line in text.split("\n")
                        if line.strip()
                    ]
            except (OSError, json.JSONDecodeError) as exc:
                raise ProtectionError(
                    f"registered protected manifest is unreadable: {path}: {exc}"
                ) from exc
            file_ids = _ids_from_payload(payload)
            if not file_ids:
                raise ProtectionError(
                    f"registered protected manifest has no row ids: {path}"
                )
            ids |= file_ids
            rows += len(payload) if isinstance(payload, list) else 1
            used_paths.append(str(path))
        all_ids[str(name)] = ids
        report[str(name)] = {
            "status": "present" if existing else "absent",
            "paths": used_paths,
            "rows": rows,
            "ids": len(ids),
            "id_sha256": hashlib.sha256(
                "\n".join(sorted(ids)).encode("utf-8")
            ).hexdigest()
            if ids
            else "",
        }
    return report, all_ids


def _alias_set(value: Sequence[str] | str) -> set[str]:
    if isinstance(value, str):
        aliases = {value}
    else:
        aliases = {str(item) for item in value}
    aliases.discard("")
    return aliases


def filter_disjoint(
    corpus_ids: Sequence[Sequence[str] | str],
    protected_ids: Mapping[str, Sequence[str] | set[str]],
    *,
    protected_report: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Exclude overlapping qualification rows and record the exclusion count.

    ``corpus_ids`` has one alias collection per row (P1-lite/Stage 0 canonical
    and raw forms).  Missing/unreadable protected manifests are handled by
    :func:`load_protected_ids` (the former are "absent", the latter raise).
    Overlaps are excluded rather than aborting, but a corpus with no remaining
    row is refused.
    """

    aliases = [_alias_set(value) for value in corpus_ids]
    if any(not values for values in aliases):
        raise ProtectionError(
            "every qualification row needs an id to prove disjointness"
        )
    protected = {
        str(name): {str(value) for value in ids}
        for name, ids in protected_ids.items()
    }
    kept: list[int] = []
    excluded: list[int] = []
    excluded_by_source: dict[str, int] = {name: 0 for name in protected}
    for index, values in enumerate(aliases):
        hit = [name for name, ids in protected.items() if values & ids]
        if hit:
            excluded.append(index)
            for name in hit:
                excluded_by_source[name] += 1
        else:
            kept.append(index)
    present = [
        name
        for name, entry in (protected_report or {}).items()
        if entry.get("status") == "present"
    ]
    per_source: dict[str, Any] = {}
    for name in protected:
        entry = dict(protected_report.get(name, {})) if protected_report else {}
        entry["intersection"] = int(excluded_by_source.get(name, 0))
        per_source[name] = entry
    if not kept:
        raise ProtectionError(
            "every qualification row overlaps a protected development or "
            "evaluation set"
        )
    verification = (
        "unverified"
        if not present
        else ("verified" if not excluded else "overlap-excluded")
    )
    return kept, {
        "disjoint": not excluded,
        "verification": verification,
        "protected_sources_present": len(present),
        "protected_sources_absent": len(per_source) - len(present),
        "corpus_rows": len(aliases),
        "kept_rows": len(kept),
        "excluded_rows": len(excluded),
        "excluded_by_source": excluded_by_source,
        "excluded_examples": [sorted(aliases[index])[:3] for index in excluded[:5]],
        "sources": per_source,
    }


def assert_disjoint(
    corpus_ids: Sequence[str],
    protected_ids: Mapping[str, Sequence[str] | set[str]],
    *,
    protected_report: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refuse a qualification corpus that intersects any protected row set."""

    corpus = [str(value) for value in corpus_ids]
    if any(not value for value in corpus):
        raise ProtectionError(
            "every qualification row needs an id to prove disjointness"
        )
    if len(set(corpus)) != len(corpus):
        raise ProtectionError("qualification corpus has duplicate row ids")
    corpus_set = set(corpus)
    per_source: dict[str, Any] = {}
    overlaps: dict[str, list[str]] = {}
    for name, ids in protected_ids.items():
        overlap = sorted(corpus_set & {str(value) for value in ids})
        per_source[str(name)] = {
            **(dict(protected_report.get(str(name), {})) if protected_report else {}),
            "intersection": len(overlap),
        }
        if overlap:
            overlaps[str(name)] = overlap
    if overlaps:
        detail = "; ".join(
            f"{name}: {len(ids)} ids e.g. {ids[:3]}"
            for name, ids in overlaps.items()
        )
        raise ProtectionError(
            "qualification corpus overlaps protected development/evaluation "
            f"sets: {detail}"
        )
    return {
        "disjoint": True,
        "corpus_ids": len(corpus_set),
        "sources": per_source,
    }


ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "mean_full_kl": 0.010,
    "mean_full_kl_ci_high": 0.015,
    "p95_full_kl": 0.050,
    "top1_agreement": 0.95,
    "top1_confident": 0.99,
    "tail_abs_error_mean": 0.002,
    "tail_abs_error_p95": 0.010,
    "allocator_headroom_gib": 0.25,
}

_EXPANSION_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Chunked primitives
# ---------------------------------------------------------------------------


def _check_inputs(
    ref_hidden: torch.Tensor,
    q_hidden: torch.Tensor,
    weight: torch.Tensor,
    actual_ids: torch.Tensor,
) -> None:
    if ref_hidden.ndim != 2 or q_hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must be 2-D")
    if ref_hidden.shape != q_hidden.shape:
        raise ValueError("reference and runtime hidden states must have equal shape")
    if ref_hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden width does not match the head weight width")
    if actual_ids.reshape(-1).shape[0] != ref_hidden.shape[0]:
        raise ValueError("actual_ids must have one entry per hidden state")
    if ref_hidden.device != q_hidden.device or ref_hidden.device != weight.device:
        raise ValueError("hidden states and head weight must share a device")


def _chunk_limits(total: int, chunk: int) -> Iterable[tuple[int, int]]:
    if chunk < 1:
        raise ValueError("chunk sizes must be positive")
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def _logits_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    v0: int,
    v1: int,
    *,
    softcap: float | None,
    mask_token_id: int | None,
) -> torch.Tensor:
    tile = hidden.float() @ weight[v0:v1].float().t()
    tile = _apply_softcap(tile, softcap)
    return _masked_logits(tile, v0, mask_token_id)


def _normalizer(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    softcap: float | None,
    mask_token_id: int | None,
    token_chunk: int,
    vocab_chunk: int,
) -> torch.Tensor:
    """Stable full-vocabulary log-sum-exp per token, fp64 totals."""

    n = int(hidden.shape[0])
    out = torch.zeros(n, dtype=torch.float64, device=hidden.device)
    vocab = int(weight.shape[0])
    for t0, t1 in _chunk_limits(n, token_chunk):
        h = hidden[t0:t1]
        m = t1 - t0
        running_max = torch.full(
            (m,), float("-inf"), dtype=torch.float64, device=hidden.device
        )
        sum_exp = torch.zeros(m, dtype=torch.float64, device=hidden.device)
        for v0, v1 in _chunk_limits(vocab, vocab_chunk):
            tile = _logits_chunk(
                h,
                weight,
                v0,
                v1,
                softcap=softcap,
                mask_token_id=mask_token_id,
            )
            tile_max = tile.max(dim=1).values.double()
            new_max = torch.maximum(running_max, tile_max)
            alpha = torch.exp(running_max - new_max)
            shifted = tile - new_max.to(tile.dtype).unsqueeze(1)
            sum_exp = sum_exp * alpha + torch.exp(shifted).sum(
                dim=1, dtype=torch.float64
            )
            running_max = new_max
        out[t0:t1] = running_max + torch.log(sum_exp)
    return out


def _gather_in_chunk(
    values: torch.Tensor,
    ids: torch.Tensor,
    v0: int,
    v1: int,
) -> torch.Tensor:
    """Gather ``values[token, id]`` for ids that fall in ``[v0, v1)``."""

    local = ids - int(v0)
    inside = (local >= 0) & (local < (v1 - v0))
    local = local.clamp(0, max(v1 - v0 - 1, 0))
    gathered = values.gather(1, local)
    return torch.where(inside, gathered, torch.zeros_like(gathered))


def _safe_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    positive = p > 0
    terms = torch.where(
        positive,
        p * (torch.log(p.clamp_min(_METRIC_TINY)) - torch.log(q.clamp_min(_METRIC_TINY))),
        torch.zeros_like(p),
    )
    return terms.sum(dim=1)


def _safe_entropy(p: torch.Tensor) -> torch.Tensor:
    positive = p > 0
    return torch.where(
        positive,
        -p * torch.log(p.clamp_min(_METRIC_TINY)),
        torch.zeros_like(p),
    ).sum(dim=1)


def _union_indices(
    ref_indices: torch.Tensor, q_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the union of each token's top-k ID sets (``-1`` outside the union)."""

    n = int(ref_indices.shape[0])
    width = int(ref_indices.shape[1]) + int(q_indices.shape[1])
    ref_cpu = ref_indices.detach().to("cpu").tolist()
    q_cpu = q_indices.detach().to("cpu").tolist()
    padded = torch.full((n, width), -1, dtype=torch.long)
    counts = torch.zeros(n, dtype=torch.long)
    for row in range(n):
        values = sorted(set(ref_cpu[row]) | set(q_cpu[row]))
        padded[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        counts[row] = len(values)
    return padded.to(ref_indices.device), counts.to(ref_indices.device)


def _kl_on_support(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL contribution per token on a fixed support (last column = residual)."""

    return _safe_kl(p, q)


def _stored_probs(
    probs16: torch.Tensor, tail: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 top-k + fp32 tail, renormalised exactly like the trainer."""

    probs = probs16.float()
    tail = tail.float()
    total = (probs.sum(dim=1) + tail).clamp_min(_METRIC_TINY)
    return probs / total.unsqueeze(1), tail / total


@torch.inference_mode()
def pair_token_metrics(
    ref_hidden: torch.Tensor,
    q_hidden: torch.Tensor,
    weight: torch.Tensor,
    actual_ids: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    mask_token_id: int | None = GLITCH_TOKEN_ID,
    softcap: float | None = None,
    token_chunk: int = 256,
    vocab_chunk: int = 8192,
) -> dict[str, torch.Tensor]:
    """Return per-token metric tensors for a reference/runtime hidden pair.

    ``actual_ids`` are O's own next-token IDs at each prediction position.  The
    full logits matrix is never stored: the normalisers and the KL/entropy pass
    both stream the vocabulary in ``vocab_chunk`` tiles.
    """

    _check_inputs(ref_hidden, q_hidden, weight, actual_ids)
    actual_ids = actual_ids.reshape(-1).to(torch.long)
    n = int(ref_hidden.shape[0])
    if n == 0:
        return {
            "kl": torch.zeros(0),
            "reverse_kl": torch.zeros(0),
            "union_kl": torch.zeros(0),
            "ref65_kl": torch.zeros(0),
            "stored65_kl": torch.zeros(0),
            "top1_match": torch.zeros(0, dtype=torch.bool),
            "top1_match_stored": torch.zeros(0, dtype=torch.bool),
            "confident": torch.zeros(0, dtype=torch.bool),
            "overlap": torch.zeros(0, dtype=torch.long),
            "ref_mass_at_q": torch.zeros(0),
            "tail_err_ref": torch.zeros(0),
            "tail_err_own": torch.zeros(0),
            "tail_err_stored": torch.zeros(0),
            "entropy_ref": torch.zeros(0),
            "entropy_q": torch.zeros(0),
            "entropy_q_stored": torch.zeros(0),
            "logp_actual_ref": torch.zeros(0),
            "logp_actual_q": torch.zeros(0),
            "logp_actual_q_stored": torch.zeros(0),
            "valid": torch.zeros(0, dtype=torch.bool),
        }

    ref_indices, _, _ = topk_distribution_from_hidden(
        ref_hidden,
        weight,
        top_k=top_k,
        mask_token_id=mask_token_id,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    q_indices, q_probs16, q_tail_raw = topk_distribution_from_hidden(
        q_hidden,
        weight,
        top_k=top_k,
        mask_token_id=mask_token_id,
        softcap=softcap,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    q_probs_stored, q_tail_stored = _stored_probs(q_probs16, q_tail_raw)

    union_indices, _ = _union_indices(ref_indices, q_indices)

    lse_ref = _normalizer(
        ref_hidden,
        weight,
        softcap=softcap,
        mask_token_id=mask_token_id,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )
    lse_q = _normalizer(
        q_hidden,
        weight,
        softcap=softcap,
        mask_token_id=mask_token_id,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
    )

    device = ref_hidden.device
    kl = torch.zeros(n, dtype=torch.float64, device=device)
    reverse_kl = torch.zeros(n, dtype=torch.float64, device=device)
    entropy_ref = torch.zeros(n, dtype=torch.float64, device=device)
    entropy_q = torch.zeros(n, dtype=torch.float64, device=device)
    ref_logit_actual = torch.full(
        (n,), float("-inf"), dtype=torch.float32, device=device
    )
    q_logit_actual = torch.full(
        (n,), float("-inf"), dtype=torch.float32, device=device
    )
    ref_at_ref = torch.zeros((n, top_k), dtype=torch.float64, device=device)
    q_at_ref = torch.zeros((n, top_k), dtype=torch.float64, device=device)
    ref_at_q = torch.zeros((n, top_k), dtype=torch.float64, device=device)
    q_at_q = torch.zeros((n, top_k), dtype=torch.float64, device=device)
    ref_at_union = torch.zeros(
        (n, union_indices.shape[1]), dtype=torch.float64, device=device
    )
    q_at_union = torch.zeros(
        (n, union_indices.shape[1]), dtype=torch.float64, device=device
    )
    vocab = int(weight.shape[0])

    for t0, t1 in _chunk_limits(n, token_chunk):
        h_ref = ref_hidden[t0:t1]
        h_q = q_hidden[t0:t1]
        m = t1 - t0
        ref_idx = ref_indices[t0:t1].long()
        q_idx = q_indices[t0:t1].long()
        union = union_indices[t0:t1]
        actual = actual_ids[t0:t1]
        kl_chunk = torch.zeros(m, dtype=torch.float64, device=device)
        reverse_chunk = torch.zeros(m, dtype=torch.float64, device=device)
        entropy_ref_chunk = torch.zeros(m, dtype=torch.float64, device=device)
        entropy_q_chunk = torch.zeros(m, dtype=torch.float64, device=device)
        for v0, v1 in _chunk_limits(vocab, vocab_chunk):
            logits_ref = _logits_chunk(
                h_ref, weight, v0, v1, softcap=softcap, mask_token_id=mask_token_id
            )
            logits_q = _logits_chunk(
                h_q, weight, v0, v1, softcap=softcap, mask_token_id=mask_token_id
            )
            logp_ref = logits_ref - lse_ref[t0:t1].to(logits_ref.dtype).unsqueeze(1)
            logp_q = logits_q - lse_q[t0:t1].to(logits_q.dtype).unsqueeze(1)
            p_ref = torch.exp(logp_ref)
            p_q = torch.exp(logp_q)

            kl_chunk += torch.where(
                p_ref > 0,
                p_ref * (logp_ref - logp_q),
                torch.zeros_like(p_ref),
            ).sum(dim=1, dtype=torch.float64)
            reverse_chunk += torch.where(
                p_q > 0,
                p_q * (logp_q - logp_ref),
                torch.zeros_like(p_q),
            ).sum(dim=1, dtype=torch.float64)
            entropy_ref_chunk += _safe_entropy(p_ref).double()
            entropy_q_chunk += _safe_entropy(p_q).double()

            ref_at_ref[t0:t1] += _gather_in_chunk(p_ref, ref_idx, v0, v1).double()
            q_at_ref[t0:t1] += _gather_in_chunk(p_q, ref_idx, v0, v1).double()
            ref_at_q[t0:t1] += _gather_in_chunk(p_ref, q_idx, v0, v1).double()
            q_at_q[t0:t1] += _gather_in_chunk(p_q, q_idx, v0, v1).double()
            ref_at_union[t0:t1] += _gather_in_chunk(p_ref, union, v0, v1).double()
            q_at_union[t0:t1] += _gather_in_chunk(p_q, union, v0, v1).double()

            actual_ref = _gather_in_chunk(
                logits_ref, actual.unsqueeze(1), v0, v1
            ).squeeze(1)
            actual_q = _gather_in_chunk(
                logits_q, actual.unsqueeze(1), v0, v1
            ).squeeze(1)
            inside_ref = ((actual >= v0) & (actual < v1))
            ref_logit_actual[t0:t1] = torch.where(
                inside_ref, actual_ref, ref_logit_actual[t0:t1]
            )
            q_logit_actual[t0:t1] = torch.where(
                inside_ref, actual_q, q_logit_actual[t0:t1]
            )

        kl[t0:t1] = kl_chunk
        reverse_kl[t0:t1] = reverse_chunk
        entropy_ref[t0:t1] = entropy_ref_chunk
        entropy_q[t0:t1] = entropy_q_chunk

    # Support used by each distribution, plus one residual bucket.
    ref_support = torch.cat([ref_at_ref, (1.0 - ref_at_ref.sum(1, keepdim=True))], 1)
    q_on_ref = torch.cat([q_at_ref, (1.0 - q_at_ref.sum(1, keepdim=True))], 1)
    ref65_kl = _kl_on_support(ref_support, q_on_ref)

    q_support = torch.cat([ref_at_q, (1.0 - ref_at_q.sum(1, keepdim=True))], 1)
    q_stored_support = torch.cat([q_probs_stored.double(), q_tail_stored.double().unsqueeze(1)], 1)
    stored65_kl = _kl_on_support(q_support, q_stored_support)

    union_p = torch.cat([ref_at_union, (1.0 - ref_at_union.sum(1, keepdim=True))], 1)
    union_q = torch.cat([q_at_union, (1.0 - q_at_union.sum(1, keepdim=True))], 1)
    union_kl = _kl_on_support(union_p, union_q)

    top1_match = ref_indices[:, 0] == q_indices[:, 0]
    stored_order = q_probs_stored.argsort(dim=1, descending=True)
    stored_indices = q_indices.gather(1, stored_order)
    top1_match_stored = ref_indices[:, 0] == stored_indices[:, 0]
    if top_k >= 2:
        gap = torch.log(ref_at_ref[:, 0].clamp_min(_METRIC_TINY)) - torch.log(
            ref_at_ref[:, 1].clamp_min(_METRIC_TINY)
        )
        confident = gap >= 0.5
    else:
        confident = torch.zeros(n, dtype=torch.bool, device=device)

    ref_cpu = ref_indices.detach().to("cpu").tolist()
    q_cpu = q_indices.detach().to("cpu").tolist()
    overlap = torch.tensor(
        [
            len(set(ref_cpu[row]) & set(q_cpu[row]))
            for row in range(n)
        ],
        dtype=torch.long,
        device=device,
    )
    ref_mass_at_q = ref_at_q.sum(1)
    ref_tail_on_ref = 1.0 - ref_at_ref.sum(1)
    tail_err_ref = torch.abs(ref_tail_on_ref - (1.0 - q_at_ref.sum(1)))
    tail_err_own = torch.abs(ref_tail_on_ref - (1.0 - q_at_q.sum(1)))

    q_in_ref = torch.tensor(
        [
            [value in set(ref_cpu[row]) for value in q_cpu[row]]
            for row in range(n)
        ],
        dtype=torch.bool,
    )
    q_stored_at_ref = (q_probs_stored.detach().to("cpu") * q_in_ref).sum(1)
    tail_err_stored = torch.abs(
        ref_tail_on_ref.detach().to("cpu")
        - (1.0 - q_stored_at_ref.double())
    ).to(device)
    # The stored distribution gathered on the *reference* top-64 support.  The
    # review asked for this view because the existing stored 65-way KL is on the
    # runtime's own support.
    match = q_indices.unsqueeze(2) == ref_indices.unsqueeze(1)
    q_stored_ref_ids = (q_probs_stored.unsqueeze(2) * match).sum(dim=1)
    q_stored_ref_support = torch.cat(
        [
            q_stored_ref_ids.double(),
            (1.0 - q_stored_ref_ids.sum(1, keepdim=True)).double(),
        ],
        1,
    )
    stored65_ref_kl = _kl_on_support(ref_support, q_stored_ref_support)

    entropy_q_stored = _safe_entropy(q_probs_stored)
    tail_stored = q_tail_stored.double()
    entropy_q_stored = entropy_q_stored + torch.where(
        tail_stored > 0,
        -tail_stored * torch.log(tail_stored.clamp_min(_METRIC_TINY)),
        torch.zeros_like(tail_stored),
    )

    logp_ref_actual = ref_logit_actual.double() - lse_ref
    logp_q_actual = q_logit_actual.double() - lse_q
    actual_in_q = (q_indices == actual_ids.unsqueeze(1)).any(dim=1)
    stored_prob = (q_probs_stored * (q_indices == actual_ids.unsqueeze(1))).sum(1)
    logp_stored_actual = torch.where(
        actual_in_q,
        torch.log(stored_prob.clamp_min(_METRIC_TINY)),
        torch.log(q_tail_stored.clamp_min(_METRIC_TINY)),
    )

    if mask_token_id is not None:
        actual_valid = actual_ids != int(mask_token_id)
    else:
        actual_valid = torch.ones(n, dtype=torch.bool, device=device)
    index_valid = (
        ((ref_indices >= 0) & (ref_indices < vocab)).all(dim=1)
        & ((q_indices >= 0) & (q_indices < vocab)).all(dim=1)
        & (actual_ids >= 0)
        & (actual_ids < vocab)
    )
    stored_total = q_probs_stored.sum(dim=1) + q_tail_stored
    stored_valid = (
        torch.isfinite(q_probs_stored).all(dim=1)
        & torch.isfinite(q_tail_stored)
        & (q_probs_stored >= 0).all(dim=1)
        & (q_tail_stored >= 0)
        & (torch.abs(stored_total - 1.0) <= 1e-4)
    )
    valid = (
        torch.isfinite(kl)
        & torch.isfinite(reverse_kl)
        & torch.isfinite(union_kl)
        & torch.isfinite(ref65_kl)
        & torch.isfinite(stored65_kl)
        & torch.isfinite(entropy_ref)
        & torch.isfinite(entropy_q)
        & torch.isfinite(entropy_q_stored)
        & actual_valid
        & index_valid
        & stored_valid
    )

    return {
        "kl": kl,
        "reverse_kl": reverse_kl,
        "union_kl": union_kl,
        "ref65_kl": ref65_kl,
        "stored65_kl": stored65_kl,
        "stored65_ref_kl": stored65_ref_kl,
        "top1_match": top1_match,
        "top1_match_stored": top1_match_stored,
        "confident": confident,
        "overlap": overlap,
        "ref_mass_at_q": ref_mass_at_q,
        "tail_err_ref": tail_err_ref,
        "tail_err_own": tail_err_own,
        "tail_err_stored": tail_err_stored,
        "entropy_ref": entropy_ref,
        "entropy_q": entropy_q,
        "entropy_q_stored": entropy_q_stored,
        "logp_actual_ref": logp_ref_actual,
        "logp_actual_q": logp_q_actual,
        "logp_actual_q_stored": logp_stored_actual,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Aggregation and bootstrap
# ---------------------------------------------------------------------------


def percentile(values: torch.Tensor, q: float) -> float:
    """Return the ``q`` quantile (0-1) of ``values``; NaN for an empty set."""

    flat = values.reshape(-1).double()
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, float(q)))


def bootstrap_mean_ci(
    row_means: Sequence[float],
    row_tokens: Sequence[int],
    *,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Token-weighted row-cluster bootstrap percentile interval."""

    means = [float(value) for value in row_means]
    tokens = [int(value) for value in row_tokens]
    if not means or sum(tokens) == 0:
        return float("nan"), float("nan")
    rng = random.Random(int(seed))
    n = len(means)
    draws: list[float] = []
    for _ in range(max(1, int(samples))):
        total_weight = 0
        total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            weight = tokens[index]
            total_weight += weight
            total += means[index] * weight
        draws.append(total / total_weight if total_weight else float("nan"))
    draws.sort()
    low = _quantile_sorted(draws, 0.025)
    high = _quantile_sorted(draws, 0.975)
    return low, high


def _quantile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _summary(
    metrics: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, float]:
    count = int(mask.sum())
    if count == 0:
        return {"tokens": 0}

    def mean(name: str) -> float:
        return float(metrics[name][mask].double().mean())

    def p95(name: str) -> float:
        return percentile(metrics[name][mask], 0.95)

    top1 = float(metrics["top1_match"][mask].double().mean())
    stored_top1 = float(metrics["top1_match_stored"][mask].double().mean())
    confident = mask & metrics["confident"]
    confident_count = int(confident.sum())
    top1_confident = (
        float(metrics["top1_match"][confident].double().mean())
        if confident_count
        else float("nan")
    )
    stored_confident = (
        float(metrics["top1_match_stored"][confident].double().mean())
        if confident_count
        else float("nan")
    )
    entropy_diff = metrics["entropy_q"][mask] - metrics["entropy_ref"][mask]
    entropy_stored_diff = (
        metrics["entropy_q_stored"][mask] - metrics["entropy_ref"][mask]
    )
    logp_diff = metrics["logp_actual_q"][mask] - metrics["logp_actual_ref"][mask]
    logp_stored_diff = (
        metrics["logp_actual_q_stored"][mask] - metrics["logp_actual_ref"][mask]
    )
    return {
        "tokens": count,
        "mean_full_kl": mean("kl"),
        "p95_full_kl": p95("kl"),
        "mean_reverse_kl": mean("reverse_kl"),
        "union_kl_mean": mean("union_kl"),
        "ref65_kl_mean": mean("ref65_kl"),
        "stored65_kl_mean": mean("stored65_kl"),
        "stored65_ref_kl_mean": mean("stored65_ref_kl"),
        "top1_agreement": top1,
        "top1_confident": top1_confident,
        "confident_tokens": confident_count,
        "top64_overlap_mean": mean("overlap"),
        "ref_mass_captured_mean": mean("ref_mass_at_q"),
        "tail_abs_error_mean": mean("tail_err_ref"),
        "tail_abs_error_p95": p95("tail_err_ref"),
        "tail_own_error_mean": mean("tail_err_own"),
        "entropy_diff_mean": float(entropy_diff.double().mean()),
        "entropy_abs_diff_mean": float(entropy_diff.double().abs().mean()),
        "logp_actual_diff_mean": float(logp_diff.double().mean()),
        "logp_actual_abs_diff_mean": float(logp_diff.double().abs().mean()),
        "stored_top1_agreement": stored_top1,
        "stored_top1_confident": stored_confident,
        "stored_tail_abs_error_mean": mean("tail_err_stored"),
        "stored_tail_abs_error_p95": p95("tail_err_stored"),
        "stored_entropy_diff_mean": float(entropy_stored_diff.double().mean()),
        "stored_entropy_abs_diff_mean": float(
            entropy_stored_diff.double().abs().mean()
        ),
        "stored_logp_actual_diff_mean": float(
            logp_stored_diff.double().mean()
        ),
        "stored_logp_actual_abs_diff_mean": float(
            logp_stored_diff.double().abs().mean()
        ),
    }


def aggregate_metrics(
    metrics: Mapping[str, torch.Tensor],
    row_ids: Sequence[int] | torch.Tensor,
    *,
    strata: Sequence[str] | None = None,
    spans: Sequence[str] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Aggregate per-token metrics overall, by length stratum and by span."""

    valid = metrics["valid"].to(torch.bool)
    row_tensor = torch.as_tensor(list(row_ids), dtype=torch.long)
    overall = _summary(metrics, valid)
    overall["tokens_total"] = int(valid.numel())
    overall["invalid_tokens"] = int((~valid).sum())
    overall["rows"] = int(len(set(int(value) for value in row_ids)))
    overall["all_finite"] = bool(
        torch.isfinite(metrics["kl"][valid]).all()
        and torch.isfinite(metrics["reverse_kl"][valid]).all()
    )

    row_means: list[float] = []
    row_tokens: list[int] = []
    for row in sorted(set(int(value) for value in row_ids)):
        mask = valid & (row_tensor == row)
        count = int(mask.sum())
        if count:
            row_means.append(float(metrics["kl"][mask].double().mean()))
            row_tokens.append(count)
    low, high = bootstrap_mean_ci(
        row_means,
        row_tokens,
        samples=bootstrap_samples,
        seed=seed,
    )
    overall["mean_full_kl_ci_low"] = low
    overall["mean_full_kl_ci_high"] = high
    overall["row_mean_kl_dispersion"] = float(
        torch.tensor(row_means, dtype=torch.float64).std(unbiased=False)
    ) if row_means else float("nan")

    def group(labels: Sequence[str], prefix: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        tensor = list(labels)
        for label in sorted(set(tensor)):
            indices = torch.tensor(
                [index for index, value in enumerate(tensor) if value == label],
                dtype=torch.long,
            )
            if indices.numel() == 0:
                continue
            mask = torch.zeros_like(valid)
            mask[indices] = True
            result[str(label)] = _summary(metrics, valid & mask)
        return result

    if strata is not None:
        overall["by_stratum"] = group(strata, "stratum")
    if spans is not None:
        overall["span"] = group(spans, "span")
    return overall


# ---------------------------------------------------------------------------
# Acceptance logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    decision: str
    checks: tuple[dict[str, object], ...] = ()
    reasons: tuple[str, ...] = ()
    near_boundary: tuple[str, ...] = ()
    next_runtime: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": bool(self.passed),
            "decision": self.decision,
            "checks": [dict(check) for check in self.checks],
            "reasons": list(self.reasons),
            "near_boundary": list(self.near_boundary),
            "next_runtime": self.next_runtime,
        }


def _check(
    name: str,
    value: float,
    comparison: str,
    threshold: float,
    unit: str = "",
) -> dict[str, object]:
    finite = not (isinstance(value, float) and math.isnan(value))
    if comparison == "upper":
        passed = finite and value <= threshold
    else:
        passed = finite and value >= threshold
    if not finite:
        near = False
    elif comparison == "upper":
        # Only a value near the boundary on either side triggers expansion;
        # 0.999 against a 0.95 threshold is comfortably inside, not "near".
        margin = (
            float(threshold) - float(value)
            if value <= threshold
            else float(value) - float(threshold)
        )
        near = margin <= _EXPANSION_FRACTION * abs(threshold)
    else:
        margin = (
            float(value) - float(threshold)
            if value >= threshold
            else float(threshold) - float(value)
        )
        near = margin <= _EXPANSION_FRACTION * abs(1.0 - threshold)
    return {
        "name": name,
        "value": float(value) if finite else float("nan"),
        "comparison": comparison,
        "threshold": float(threshold),
        "unit": unit,
        "passed": bool(passed),
        "near_boundary": bool(near),
    }


def evaluate_acceptance(
    overall: Mapping[str, object],
    *,
    memory: Mapping[str, object] | None = None,
    teacher_key: str | None = None,
    current_runtime: str | None = None,
    n_rows: int = 20,
    max_rows: int = 100,
) -> AcceptanceResult:
    """Apply the frozen X3 section 7 table and the one-step expansion rule."""

    thresholds = ACCEPTANCE_THRESHOLDS
    checks = [
        _check(
            "mean_full_kl",
            float(overall.get("mean_full_kl", float("nan"))),
            "upper",
            thresholds["mean_full_kl"],
            "nat",
        ),
        _check(
            "mean_full_kl_upper95",
            float(overall.get("mean_full_kl_ci_high", float("nan"))),
            "upper",
            thresholds["mean_full_kl_ci_high"],
            "nat",
        ),
        _check(
            "p95_full_kl",
            float(overall.get("p95_full_kl", float("nan"))),
            "upper",
            thresholds["p95_full_kl"],
            "nat",
        ),
        _check(
            "top1_agreement",
            float(overall.get("top1_agreement", float("nan"))),
            "lower",
            thresholds["top1_agreement"],
        ),
        _check(
            "top1_confident",
            float(overall.get("top1_confident", float("nan"))),
            "lower",
            thresholds["top1_confident"],
        ),
        _check(
            "tail_abs_error_mean",
            float(overall.get("tail_abs_error_mean", float("nan"))),
            "upper",
            thresholds["tail_abs_error_mean"],
        ),
        _check(
            "tail_abs_error_p95",
            float(overall.get("tail_abs_error_p95", float("nan"))),
            "upper",
            thresholds["tail_abs_error_p95"],
        ),
        _check(
            "tensor_validity",
            1.0
            if int(overall.get("invalid_tokens", 1)) == 0
            and bool(overall.get("all_finite", False))
            else 0.0,
            "lower",
            1.0,
        ),
    ]
    if memory is not None:
        checks.append(
            _check(
                "allocator_headroom_gib",
                float(memory.get("headroom_gib") or float("nan")),
                "lower",
                thresholds["allocator_headroom_gib"],
                "GiB",
            )
        )
        checks.append(
            _check(
                "allocation_stable",
                1.0 if memory.get("steady", True) else 0.0,
                "lower",
                1.0,
            )
        )

    overall_passed = all(bool(check["passed"]) for check in checks)
    near = tuple(
        str(check["name"])
        for check in checks
        if check["near_boundary"]
        and check["name"]
        not in {"allocator_headroom_gib", "allocation_stable", "tensor_validity"}
    )

    def group_failures(key: str) -> list[str]:
        failures: list[str] = []
        groups = overall.get(key) or {}
        if not isinstance(groups, Mapping):
            return failures
        for label, summary in groups.items():
            if not isinstance(summary, Mapping):
                continue
            if float(summary.get("mean_full_kl", float("nan"))) > thresholds["mean_full_kl"]:
                failures.append(f"{key}:{label}:mean_full_kl")
            if float(summary.get("p95_full_kl", float("nan"))) > thresholds["p95_full_kl"]:
                failures.append(f"{key}:{label}:p95_full_kl")
            if float(summary.get("top1_agreement", float("nan"))) < thresholds["top1_agreement"]:
                failures.append(f"{key}:{label}:top1_agreement")
        return failures

    group_failure = group_failures("by_stratum") + group_failures("span")
    reasons: list[str] = [
        f"{check['name']}={check['value']} {check['comparison']} {check['threshold']}"
        for check in checks
        if not check["passed"]
    ]
    reasons += group_failure

    if overall_passed and not near and not group_failure:
        return AcceptanceResult(
            passed=True,
            decision="pass",
            checks=tuple(checks),
            reasons=(),
            near_boundary=near,
        )
    if int(n_rows) < int(max_rows) and (near or group_failure):
        return AcceptanceResult(
            passed=bool(overall_passed),
            decision="expand",
            checks=tuple(checks),
            reasons=tuple(reasons),
            near_boundary=near,
        )
    if overall_passed and not group_failure:
        return AcceptanceResult(
            passed=True,
            decision="pass",
            checks=tuple(checks),
            reasons=tuple(reasons),
            near_boundary=near,
        )
    fallback = None
    if teacher_key is not None:
        fallback = REGISTERED_FALLBACK_RUNTIMES.get(str(teacher_key).lower())
        if fallback == current_runtime:
            fallback = None
    return AcceptanceResult(
        passed=False,
        decision="fail_next_runtime",
        checks=tuple(checks),
        reasons=tuple(reasons),
        near_boundary=near,
        next_runtime=fallback,
    )


def resolve_decision(
    screen: AcceptanceResult,
    expanded: AcceptanceResult | None,
    *,
    selected_rows: int,
    expand_rows: int,
) -> AcceptanceResult:
    """Turn the intermediate ``"expand"`` into a terminal decision.

    A triggered expansion that cannot be run (too few rows) or that remains
    unresolved at its own limit is INCONCLUSIVE and must never be reported as a
    pass.  Only ``"pass"`` is a successful qualification.
    """

    if screen.decision != "expand":
        return screen
    if expanded is None or int(selected_rows) < int(expand_rows):
        return AcceptanceResult(
            passed=False,
            decision="insufficient_rows",
            checks=screen.checks,
            reasons=tuple(screen.reasons)
            + (
                f"expansion requested but only {int(selected_rows)} of the "
                f"required {int(expand_rows)} rows are available",
            ),
            near_boundary=screen.near_boundary,
        )
    if expanded.decision == "expand":
        return AcceptanceResult(
            passed=False,
            decision="unresolved",
            checks=expanded.checks,
            reasons=tuple(expanded.reasons)
            + (
                f"the {int(expand_rows)}-row expansion is still within the "
                "20% boundary; the qualification did not reach a decision",
            ),
            near_boundary=expanded.near_boundary,
        )
    return expanded


# ---------------------------------------------------------------------------
# Deterministic development-row selection
# ---------------------------------------------------------------------------


def _spread(pool: Sequence[int], count: int) -> list[int]:
    """Pick ``count`` evenly spaced indices from a length-sorted ``pool``."""

    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return list(pool)
    if count == 1:
        return [pool[0]]
    return [pool[round(index * (len(pool) - 1) / (count - 1))] for index in range(count)]


def stratify_indices(
    lengths: Sequence[int],
    *,
    n_short: int = 4,
    n_common: int = 8,
    n_medium: int = 4,
    n_long: int = 4,
    seed: int = 0,
    common_low: int = 1000,
    common_high: int = 2000,
) -> list[tuple[int, str]]:
    """Choose development rows by length, deterministically.

    The returned order is interleaved ``short, common, common, medium, long``
    blocks so the first 20 rows of a scaled 100-row set are still a
    stratified 4/8/4/4 screen.  Ties are broken by a seeded hash of the row
    index, never by a teacher loss.
    """

    import hashlib

    n = len(lengths)
    if n == 0:
        return []
    order = sorted(
        range(n),
        key=lambda index: (
            int(lengths[index]),
            hashlib.sha256(
                f"{int(seed)}:{index}:{int(lengths[index])}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    wanted_short = min(int(n_short), n)
    short = order[:wanted_short]
    wanted_long = min(int(n_long), n - len(short))
    # Long means the rows nearest the upper limit: take the largest and put the
    # largest first so a stratified prefix is not biased towards the lower end.
    long = list(reversed(order[len(order) - wanted_long :])) if wanted_long else []
    selected = set(short) | set(long)
    remaining = [index for index in order if index not in selected]

    pool = [
        index
        for index in remaining
        if int(common_low) <= int(lengths[index]) <= int(common_high)
    ]
    if len(pool) < int(n_common):
        nearest = sorted(
            (index for index in remaining if index not in set(pool)),
            key=lambda index: (
                abs(int(lengths[index]) - (int(common_low) + int(common_high)) // 2),
                int(lengths[index]),
                index,
            ),
        )
        pool = pool + nearest
    common = _spread(pool, min(int(n_common), len(pool)))
    selected = set(short) | set(long) | set(common)
    remaining = [index for index in order if index not in selected]
    medium = _spread(remaining, min(int(n_medium), len(remaining)))
    selected = set(short) | set(long) | set(common) | set(medium)
    leftover = [index for index in order if index not in selected]

    groups: dict[str, list[int]] = {
        "short": list(short),
        "common": list(common),
        "medium": list(medium),
        "long": list(long),
    }
    for index in leftover:
        length = int(lengths[index])
        if length <= int(common_low):
            groups["short"].append(index)
        elif length <= int(common_high):
            groups["common"].append(index)
        elif length <= 2 * int(common_high):
            groups["medium"].append(index)
        else:
            groups["long"].append(index)

    result: list[tuple[int, str]] = []
    cursors = {name: 0 for name in groups}
    block = (("short", 1), ("common", 2), ("medium", 1), ("long", 1))
    while any(cursors[name] < len(groups[name]) for name in groups):
        for name, take in block:
            for _ in range(take):
                if cursors[name] < len(groups[name]):
                    result.append((groups[name][cursors[name]], name))
                    cursors[name] += 1
        # Defensive progress guard for unexpected group shapes.
        if not any(cursors[name] < len(groups[name]) for name in groups):
            break
    return result


def _scaled_strata(rows: int) -> tuple[int, int, int, int]:
    scale = max(1, math.ceil(int(rows) / 20))
    return 4 * scale, 8 * scale, 4 * scale, 4 * scale


def select_development_rows(
    lengths: Sequence[int],
    *,
    screen_rows: int = 20,
    expand_rows: int = 100,
    seed: int = 0,
    common_low: int = 1000,
    common_high: int = 2000,
) -> dict[str, list[tuple[int, str]]]:
    """Return a true stratified screen and a predetermined expansion superset.

    The screen is drawn directly from the whole corpus with the section 7
    quotas (4 short, 8 common 1-2k, 4 medium, 4 near the upper limit) and does
    not depend on the expansion set.  The expansion is the screen plus extra
    rows selected deterministically from the remainder; it is only evaluated if
    the screen triggers the one-step expansion.
    """

    if screen_rows < 1 or expand_rows < 1:
        raise ValueError("screen_rows and expand_rows must be positive")
    scale_screen = max(1, math.ceil(int(screen_rows) / 20))
    screen = stratify_indices(
        lengths,
        n_short=4 * scale_screen,
        n_common=8 * scale_screen,
        n_medium=4 * scale_screen,
        n_long=4 * scale_screen,
        seed=seed,
        common_low=common_low,
        common_high=common_high,
    )[: int(screen_rows)]
    if len(screen) < int(screen_rows):
        raise ValueError(
            f"corpus has {len(screen)} selectable rows but the screen needs "
            f"{screen_rows}"
        )
    if int(expand_rows) <= int(screen_rows):
        return {"screen": screen, "expansion": screen}

    target = _scaled_strata(expand_rows)
    present = {"short": 0, "common": 0, "medium": 0, "long": 0}
    for _, label in screen:
        present[label] = present.get(label, 0) + 1
    need = {
        label: max(0, target[index] - present[label])
        for index, label in enumerate(("short", "common", "medium", "long"))
    }
    chosen = {index for index, _ in screen}
    remaining = [index for index in range(len(lengths)) if index not in chosen]
    remaining_lengths = [int(lengths[index]) for index in remaining]
    extras = stratify_indices(
        remaining_lengths,
        n_short=int(need["short"]),
        n_common=int(need["common"]),
        n_medium=int(need["medium"]),
        n_long=int(need["long"]),
        seed=int(seed) + 1,
        common_low=common_low,
        common_high=common_high,
    )
    expansion = list(screen)
    for index, label in extras:
        if len(expansion) >= int(expand_rows):
            break
        expansion.append((remaining[index], label))
    return {"screen": screen, "expansion": expansion}


# ---------------------------------------------------------------------------
# Thinking / answer spans
# ---------------------------------------------------------------------------


def token_spans(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    thought_open_ids: Sequence[int],
    thought_close_ids: Sequence[int],
    ignore_index: int = -100,
) -> list[str | None]:
    """Classify every position as ``"thinking"``, ``"answer"`` or ``None``.

    ``None`` means the position is not a supervised target (including position
    0, which predicts nothing).  Marker tokens belong to the thinking span.
    """

    n = len(input_ids)
    if len(labels) != n:
        raise ValueError("input_ids and labels must have the same length")
    open_ids = tuple(int(value) for value in thought_open_ids)
    close_ids = tuple(int(value) for value in thought_close_ids)
    thinking = [False] * n
    in_thought = False
    index = 0
    while index < n:
        if open_ids and tuple(input_ids[index : index + len(open_ids)]) == open_ids:
            in_thought = True
            for offset in range(len(open_ids)):
                if index + offset < n:
                    thinking[index + offset] = True
            index += len(open_ids)
            continue
        if close_ids and tuple(input_ids[index : index + len(close_ids)]) == close_ids:
            for offset in range(len(close_ids)):
                if index + offset < n:
                    thinking[index + offset] = True
            in_thought = False
            index += len(close_ids)
            continue
        thinking[index] = in_thought
        index += 1
    spans: list[str | None] = []
    for position, label in enumerate(labels):
        if position == 0 or int(label) == int(ignore_index):
            spans.append(None)
        else:
            spans.append("thinking" if thinking[position] else "answer")
    return spans
