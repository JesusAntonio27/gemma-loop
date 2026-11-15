"""Pilot-shortfall amendment: verifier freeze, verified counts and top-up plan.

This module implements the *code and data* half of the 2026-11-06 X1 ruling
(``codex-ruling-pilot-shortfall.md``).  It deliberately does not touch the
preregistration prose or schema; the prereg branch copies the freeze index this
module writes.

The flow is deliberately sequential and hash-bound:

1. :func:`freeze_verifier` writes an ordered per-problem test manifest from the
   frozen Stage 0 pool, pins the Docker image by content digest, and hashes the
   extraction/wrapper/comparison implementation and its dependency versions.
2. :func:`plan_topup` writes the complete *potential* extension plan (sample
   indices 4-7 for every original T-fit/selection math/code problem) before any
   original-code count is computed.
3. :func:`count_verified` grades every original code sample that passes the
   unchanged Stage 0 filter and writes the per-request accounting, ``nT0``,
   ``nS0``, the P1 descriptive count and the extension Boolean.
4. :func:`build_final_corpus` assembles the eligible original (and, only if the
   extension was triggered, extra) traces into the final pilot JSONLs.
5. :func:`write_freeze_index` binds every rule, evidence, verifier, count and
   final-data hash the ruling lists; :func:`check_freeze_readiness` is the
   fail-closed execution guard used by ``scripts/run_pilot.py``.

The module never imports a tokenizer, Modal or Docker at import time.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemmaloop.stage0 import local, modal_lite
from gemmaloop.stage0.grader import (
    CANDIDATE_DEADLINE_CLASSES,
    CLIENT_HARD_KILL_ACTION,
    DEADLINE_ENFORCEMENT_ACTIONS,
    DEADLINE_ENFORCEMENT_EXTRA_S,
    DEADLINE_ENFORCEMENT_ROUTES,
    DEADLINE_ENFORCEMENT_START_S,
    DOCKER_KILL_ACTION,
    DOCKER_RM_ACTION,
    DEFAULT_TIMEOUT_S,
    ENFORCEMENT_ROUTE_HOST,
    ENFORCEMENT_ROUTE_WRAPPER,
    INFRASTRUCTURE_LIFECYCLE_CLASSES,
    LIFECYCLE_COMPLETED_BEFORE,
    LIFECYCLE_LOCAL,
    LIFECYCLE_LOCAL_TIMEOUT,
    LIFECYCLE_NOTICE_GRACE_S,
    LIFECYCLE_NOTICE_OBSERVATION_S,
    LIFECYCLE_RUNNER_NAME,
    LIFECYCLE_STARTUP_FAILURE,
    LIFECYCLE_UNRESOLVED,
    TRUSTED_LIFECYCLE_CLASSES,
    _LIFECYCLE_RUNNER_SOURCE,
    _compare_output as _compare_grader_output,
    authenticate_lifecycle_frames,
    classify_lifecycle_evidence,
    lifecycle_record_error,
    sanitized_env,
    wrapper_enforcement_route_verified,
)
from gemmaloop.stage0.interpretation import (
    EXECUTION_TESTS_NAME,
    INTERPRETATION_MANIFEST_NAME,
    InterpretationError,
    excluded_problem_ids,
    execution_tests_by_problem,
    load_interpretation_manifest,
    verify_interpretation_against_test_manifest,
    write_interpretation_manifest,
)
from gemmaloop.stage0.postprocess import (
    MAX_LENGTH,
    MIN_REASONING_SHARE,
    extract_code,
    keep_reason,
    reasoning_share,
)
from gemmaloop.train.checkpoint import atomic_write_text

__all__ = [
    "ATTEMPT_LEDGER_NAME",
    "COUNT_LOG_NAME",
    "COUNT_MANIFEST_NAME",
    "COUNT_PER_REQUEST_NAME",
    "COUNT_SUMMARY_NAME",
    "FREEZE_MANIFEST_NAME",
    "FREEZE_TEST_MANIFEST_NAME",
    "PILOT_SPLITS",
    "ShortfallCountError",
    "ShortfallReadinessError",
    "TOPUP_INDICES",
    "assert_not_void_artifact",
    "assemble_corpus",
    "build_eligible_traces",
    "build_topup_plan",
    "check_freeze_readiness",
    "check_pilot_inputs",
    "code_implementation_hashes",
    "compute_counts",
    "count_verified",
    "decide_extension",
    "executing_implementation",
    "expected_topup_requests",
    "freeze_verifier",
    "grade_original_requests",
    "load_final_corpus_manifest",
    "load_freeze_manifest",
    "load_verified_count",
    "plan_topup",
    "resolve_docker_image",
    "resume_v3_pipeline",
    "resume_v3r_pipeline",
    "run_v3_pipeline",
    "run_v3r_pipeline",
    "select_original_inputs",
    "test_manifest_rows",
    "topup_timeout_seconds",
    "validate_topup_plan",
    "verify_count_implementation",
    "verify_final_corpus",
    "write_final_corpus",
    "write_freeze_index",
]

PILOT_SPLITS = ("pilot_p1", "pilot_tfit", "pilot_selection")
SPLITS_WITH_TOPUP = ("pilot_tfit", "pilot_selection")
TOPUP_INDICES = (4, 5, 6, 7)
P1_MIN = 20
TFIT_MIN = 20
SELECTION_MIN = 30

FREEZE_MANIFEST_NAME = "verifier_freeze.json"
FREEZE_TEST_MANIFEST_NAME = "test_manifest.jsonl"
COUNT_SUMMARY_NAME = "summary.json"
COUNT_PER_REQUEST_NAME = "per_request.jsonl"
COUNT_LOG_NAME = "grade.log"
COUNT_MANIFEST_NAME = "count_manifest.json"
ATTEMPT_LEDGER_NAME = "attempt_ledger.jsonl"
LEDGER_KIND = "pilot_shortfall_attempt_ledger"
LEDGER_HEADER = "ledger_header"
LEDGER_ATTEMPT_START = "attempt_start"
LEDGER_ATTEMPT_RESULT = "attempt_result"
LEDGER_ATTEMPT_INTERRUPTED = "attempt_interrupted"
# Append-only evidence that an interrupted attempt's *actual* recorded
# container was inspected and found gone (or removed and verified gone) during
# recovery.  It is bound to request/test/attempt/container and never rewrites
# the interrupted record or changes its infrastructure outcome.
LEDGER_CLEANUP_RECOVERED = "cleanup-recovered"
# Append-only evidence that a cleanup-recovery pass ran but did not verify the
# interrupted attempt's actual recorded container.  The attempt stays pending
# cleanup recovery; a later pass may append the matching cleanup-recovered
# record.  This record never changes the interrupted outcome or attempt count.
LEDGER_CLEANUP_RECOVERY_FAILED = "cleanup-recovery-failed"
# A verifier interface/programming error that stopped the official run.  It is
# never an attempt result: it does not count as a failed attempt, does not
# consume a retry, and blocks any further dispatch.
LEDGER_VERIFIER_INTERFACE_ERROR = "verifier-interface-error"
# Attempt statuses that are conclusive: they control the result and are never
# regraded by a resume.
CONCLUSIVE_ATTEMPT_STATUSES = frozenset(
    {"passed", "wrong_answer", "candidate_error", "timeout"}
)

INDEX_STATUS_COMPLETE = "complete"
INDEX_STATUS_PENDING = "pending-data-addendum"
COUNT_STATUS_COMPLETE = "complete"
COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE = (
    "complete_with_terminal_infrastructure"
)
COUNT_STATUS_BLOCKED_INFRASTRUCTURE = "blocked-infrastructure"
TEST_STATUS_TERMINAL_INFRASTRUCTURE = "infrastructure_unverified"

# One-time v3 registry/lock.  It lives in the fixed shortfall artifact
# directory (next to the v1/v2 artifacts), never under ``--out-root``, so an
# alternate output root cannot start a second official v3 run.
V3_REGISTRY_NAME = "v3_registry.json"
V3_REGISTRY_KIND = "pilot_shortfall_v3_registry"

# ---------------------------------------------------------------------------
# X1 v3-void disposition and the single authorized replacement (v3r).
#
# The failed v3 registry/freeze/count/ledger bytes are preserved verbatim.  The
# disposition is carried by a separate, fixed-location *void receipt* that
# binds those file-byte hashes and the ruling.  A fixed-location *continuity
# record* links the preserved crashed v3 registry, that receipt, the ruling, the
# (future) replacement review and the distinct v3r registry.  Neither the void
# identity nor an alternate output directory may ever run a count again.
# ---------------------------------------------------------------------------
VOID_RECEIPT_NAME = "v3_void_receipt.json"
VOID_RECEIPT_KIND = "pilot_shortfall_v3_void_receipt"
VOID_DISPOSITION = "VOID_VERIFIER_IMPLEMENTATION_FAILURE"
VOID_RULING_NAME = "codex-ruling-v3-void.md"
VOID_EXPECTED_FILE_SHA256 = {
    "v3_registry.json": "5e103b898ec9aac6a5b862cda4d607761c5bda815b701366625ab45823ce8a12",
    "verifier_v3/verifier_freeze.json": "3eed1f0be765e609e13f3f91460f3a15f1ff032cec1d1373c294477f914e1ea0",
    "count_v3/count_manifest.json": "5b0cd11b9b612914de865eb06b6419bab3376f01eb07380ed6ffd2134e854ea2",
    "count_v3/attempt_ledger.jsonl": "7d0e9fb47c6e73218f274cb8305590861bbbf763f33f1d9b43d09ad5455fced1",
}
V3R_CONTINUITY_NAME = "v3r_continuity.json"
V3R_CONTINUITY_KIND = "pilot_shortfall_v3r_continuity"
V3R_REGISTRY_NAME = "v3r_registry.json"
V3R_REGISTRY_KIND = "pilot_shortfall_v3r_registry"
V3R_REVIEW_NAME = "v3r_replacement_review.json"
V3R_REVIEW_KIND = "pilot_shortfall_v3r_replacement_review"
V3_STATUS_CLAIMED = "claimed"
V3_STATUS_RUNNING = "running"
V3_STATUS_COMPLETE = "complete"
V3_STATUS_CRASHED = "crashed"
V3_STATUS_RESUMABLE = "resumable"
V3_STAGE_DIRECTORIES = {
    "interpretation": "interpretation_v3",
    "freeze": "verifier_v3",
    "decision": None,
    "count": "count_v3",
    "corpus": "corpus_v3",
    "index": None,
}
V3_STAGES = (
    "interpretation",
    "freeze",
    "decision",
    "count",
    "corpus",
    "index",
)
V3R_STAGE_DIRECTORIES = {
    "interpretation": "interpretation_v3r",
    "freeze": "verifier_v3r",
    "decision": None,
    "count": "count_v3r",
    "corpus": "corpus_v3r",
    "index": None,
}
V3R_PLAN_NAME = "pilot_topup_plan_v3r.json"
V3R_INDEX_NAME = "pilot-shortfall-freeze-v3r.json"
V3R_OUT_DIR_NAME = "v3r"
V3R_FIRST_ATTEMPT = 1
V3R_MAX_INFRASTRUCTURE_RETRIES = 1


@dataclass(frozen=True)
class _RunSpec:
    """Naming/guard identity shared by the v3 and v3r pipelines."""

    version: str
    registry_name: str
    registry_kind: str
    stage_directories: Mapping[str, str | None]
    stages: tuple[str, ...]
    plan_name: str
    index_name: str

    def registry_path(self, lite_dir: Path | str) -> Path:
        return Path(lite_dir) / "shortfall" / self.registry_name

    def out_root(self, lite_dir: Path | str) -> Path:
        return Path(lite_dir) / "shortfall" / V3R_OUT_DIR_NAME


V3_RUN_SPEC = _RunSpec(
    version="v3",
    registry_name=V3_REGISTRY_NAME,
    registry_kind=V3_REGISTRY_KIND,
    stage_directories=V3_STAGE_DIRECTORIES,
    stages=V3_STAGES,
    plan_name="pilot_topup_plan_v3.json",
    index_name="pilot-shortfall-freeze-v3.json",
)
V3R_RUN_SPEC = _RunSpec(
    version="v3r",
    registry_name=V3R_REGISTRY_NAME,
    registry_kind=V3R_REGISTRY_KIND,
    stage_directories=V3R_STAGE_DIRECTORIES,
    stages=V3_STAGES,
    plan_name=V3R_PLAN_NAME,
    index_name=V3R_INDEX_NAME,
)


def _run_spec(version: str | None) -> _RunSpec:
    value = str(version or "v3").lower()
    if value == "v3r":
        return V3R_RUN_SPEC
    if value == "v3":
        return V3_RUN_SPEC
    raise ShortfallV3RunError(f"unknown official run version: {version!r}")
AMENDMENT_REPO_PATH = "prereg/pilot-shortfall-amendment.md"
DEPENDENCY_FINGERPRINT_KEYS = (
    "python",
    "python_full",
    "transformers",
    "math-verify",
    "huggingface-hub",
)

# Timing model for the one extension call.  The 2026-11-05 probe generated
# 2,000 requests in 1,802 s (0.901 s/request).  A 2x safety factor above the
# measured rate gives the registered per-call timeout; the 600 s floor covers
# vLLM engine warm-up for small synthetic plans.
TOPUP_MEASURED_SECONDS_PER_REQUEST = 1_802.0 / 2_000.0
TOPUP_TIMEOUT_SAFETY = 2.0
TOPUP_TIMEOUT_FLOOR_S = 600
TOPUP_RUN_CAP_USD = 3.0
HARD_GLOBAL_CAP_USD = 30.0
LITE_RUN_ID_FALLBACK = "b605af7a625f93c7e3e08a02"
LITE_IMAGE_TAG = "python:3.12-slim"
LITE_ENGINE = "modal-vllm"
LITE_ENGINE_VERSION = "vllm==0.30.0"
MODEL_ID = "google/gemma-4-E2B-it"
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "enable_thinking": True,
    "max_new_tokens_rule": "8192 - rendered_prompt_tokens",
}


class ShortfallReadinessError(RuntimeError):
    """The registered pilot corpus/verifier readiness gate refused execution."""


class ShortfallCountError(RuntimeError):
    """The official count cannot make a final decision (rerun required)."""


class ShortfallV3RunError(RuntimeError):
    """The one-time official v3 run already started or left a crash state."""


class ShortfallVoidRunError(ShortfallV3RunError):
    """Execution under the void v3 identity is prohibited."""


class ShortfallVerifierInterfaceError(ShortfallV3RunError, TypeError):
    """The verifier adapter/grader interface is unusable (a programming error).

    Such an error blocks any further official dispatch immediately.  It must
    never be recorded as a per-candidate infrastructure result and must never
    consume a registered retry.
    """


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(str(text).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return local.file_sha256(Path(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    local.write_json_atomic(Path(path), dict(payload))


def _write_text_exclusive(path: Path, text: str) -> None:
    """Create ``path`` durably; refuse to clobber an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=chr(10)) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
    )


def _require_new_dir(path: Path) -> None:
    """Refuse a stage directory that already contains any output."""

    path = Path(path)
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(
                f"v3 output already exists and may not be clobbered: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def v3_registry_path(lite_dir: Path | str) -> Path:
    """The fixed one-time registry location, independent of ``--out-root``."""

    return Path(lite_dir) / "shortfall" / V3_REGISTRY_NAME


def void_receipt_path(lite_dir: Path | str) -> Path:
    """The fixed, immutable v3 void receipt (next to ``v3_registry.json``)."""

    return Path(lite_dir) / "shortfall" / VOID_RECEIPT_NAME


def v3r_continuity_path(lite_dir: Path | str) -> Path:
    """The fixed continuity record linking crash -> receipt -> ruling -> v3r."""

    return Path(lite_dir) / "shortfall" / V3R_CONTINUITY_NAME


def v3r_registry_path(lite_dir: Path | str) -> Path:
    """The fixed one-time v3r registry, independent of ``--out-root``."""

    return Path(lite_dir) / "shortfall" / V3R_REGISTRY_NAME


def v3r_review_path(lite_dir: Path | str) -> Path:
    return Path(lite_dir) / "shortfall" / V3R_REVIEW_NAME


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    return _hash_payload(payload, "receipt_sha256")


def _continuity_hash(payload: Mapping[str, Any]) -> str:
    return _hash_payload(payload, "continuity_sha256")


def _binding(path: Path, *, role: str, expected: str | None = None) -> dict[str, Any]:
    path = Path(path)
    entry: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "bytes": int(path.stat().st_size) if path.is_file() else None,
    }
    if expected is not None:
        entry["expected_sha256"] = expected
    return entry


def write_void_receipt(
    lite_dir: Path | str,
    *,
    ruling_path: Path | str,
    registry_path: Path | str | None = None,
    freeze_path: Path | str | None = None,
    count_manifest_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
    void_out_root: Path | str | None = None,
    enforce_known_hashes: bool = True,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Write the immutable void receipt after verifying every bound file byte.

    The official invocation refuses on any hash mismatch and never overwrites
    an existing receipt.  ``enforce_known_hashes=False`` exists only for
    synthetic dry-run fixtures and is never used for the official receipt.
    """

    lite_dir = Path(lite_dir)
    shortfall_dir = lite_dir / "shortfall"
    registry_path = Path(registry_path or v3_registry_path(lite_dir))
    freeze_path = Path(
        freeze_path or shortfall_dir / "v3" / "verifier_v3" / FREEZE_MANIFEST_NAME
    )
    count_manifest_path = Path(
        count_manifest_path
        or shortfall_dir / "v3" / "count_v3" / COUNT_MANIFEST_NAME
    )
    ledger_path = Path(
        ledger_path or shortfall_dir / "v3" / "count_v3" / ATTEMPT_LEDGER_NAME
    )
    ruling_path = Path(ruling_path)
    bindings = {
        "v3_registry.json": registry_path,
        "verifier_v3/verifier_freeze.json": freeze_path,
        "count_v3/count_manifest.json": count_manifest_path,
        "count_v3/attempt_ledger.jsonl": ledger_path,
    }
    entries: dict[str, Any] = {}
    for label, path in bindings.items():
        if not path.is_file():
            raise ShortfallVoidRunError(f"void receipt binding is missing: {path}")
        expected = (
            VOID_EXPECTED_FILE_SHA256[label] if enforce_known_hashes else None
        )
        actual = _sha256_file(path)
        if expected is not None and actual != expected:
            raise ShortfallVoidRunError(
                f"void receipt refuses {path}: file sha256 {actual} does not "
                f"match the X1-published {expected}"
            )
        entries[label] = _binding(path, role=label, expected=expected)
    if not ruling_path.is_file():
        raise ShortfallVoidRunError(f"the v3 void ruling is missing: {ruling_path}")
    registry = _read_json(registry_path)
    if registry.get("kind") != V3_REGISTRY_KIND:
        raise ShortfallVoidRunError(
            f"{registry_path}: preserved registry is not a v3 registry"
        )
    if registry.get("official") is not True or registry.get("status") != V3_STATUS_CRASHED:
        raise ShortfallVoidRunError(
            f"{registry_path}: preserved v3 registry is not the crashed official run"
        )
    if void_out_root is None:
        void_out_root = (
            (shortfall_dir / "v3").resolve()
            if enforce_known_hashes
            else count_manifest_path.parent.parent.resolve()
        )
    payload: dict[str, Any] = {
        "kind": VOID_RECEIPT_KIND,
        "version": 1,
        "official": False,
        "synthetic": not enforce_known_hashes,
        "disposition": VOID_DISPOSITION,
        "created_utc": created_utc or _utc_now(),
        "ruling": _binding(ruling_path, role="ruling"),
        "files": entries,
        "void_out_root": str(Path(void_out_root).resolve()),
        "v3_registry_status": registry.get("status"),
        "v3_registry_created_utc": registry.get("created_utc"),
        "statement": (
            "Preserved v3 run VOID_VERIFIER_IMPLEMENTATION_FAILURE. The count "
            "adapter passed container_names/attempts to grade_code, whose "
            "signature accepted neither, before any pilot candidate dispatch. "
            "The preserved bytes remain the historical record and are readable "
            "for audit only; they are not operative count/corpus/index inputs."
        ),
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _receipt_hash(payload)
    _write_json_exclusive(void_receipt_path(lite_dir), payload)
    return payload


def load_void_receipt(
    lite_dir: Path | str, *, require_official: bool = False
) -> dict[str, Any]:
    """Load and fully verify the fixed void receipt (corruption blocks)."""

    path = void_receipt_path(lite_dir)
    if not path.is_file():
        raise ShortfallVoidRunError(f"the v3 void receipt is missing: {path}")
    payload = _read_json(path)
    if payload.get("kind") != VOID_RECEIPT_KIND:
        raise ShortfallVoidRunError(f"{path}: not a v3 void receipt")
    if payload.get("disposition") != VOID_DISPOSITION:
        raise ShortfallVoidRunError(
            f"{path}: void disposition is {payload.get('disposition')!r}"
        )
    if require_official and payload.get("synthetic") is True:
        raise ShortfallVoidRunError(
            f"{path}: synthetic receipt may not authorize an official run"
        )
    if payload.get("receipt_sha256") != _receipt_hash(payload):
        raise ShortfallVoidRunError(f"{path}: receipt self hash does not recompute")
    files = payload.get("files")
    if not isinstance(files, Mapping) or set(files) != set(VOID_EXPECTED_FILE_SHA256):
        raise ShortfallVoidRunError(
            f"{path}: receipt does not bind the complete preserved v3 file set"
        )
    for label, entry in files.items():
        bound = Path(str(entry.get("path") or ""))
        if not bound.is_file():
            raise ShortfallVoidRunError(
                f"{path}: bound void artifact {label} is missing ({bound})"
            )
        actual = _sha256_file(bound)
        if actual != entry.get("sha256"):
            raise ShortfallVoidRunError(
                f"{path}: bound void artifact {label} changed after the receipt"
            )
        expected = entry.get("expected_sha256")
        if expected is not None and actual != expected:
            raise ShortfallVoidRunError(
                f"{path}: bound void artifact {label} does not match the "
                "X1-published file-byte hash"
            )
    ruling = payload.get("ruling") or {}
    ruling_path = Path(str(ruling.get("path") or ""))
    if not ruling_path.is_file() or _sha256_file(ruling_path) != ruling.get("sha256"):
        raise ShortfallVoidRunError(f"{path}: the bound ruling file changed or is missing")
    registry_entry = files["v3_registry.json"]
    registry = _read_json(Path(str(registry_entry["path"])))
    if (
        registry.get("kind") != V3_REGISTRY_KIND
        or registry.get("official") is not True
        or registry.get("status") != V3_STATUS_CRASHED
    ):
        raise ShortfallVoidRunError(
            f"{path}: the preserved v3 registry is not the crashed official run"
        )
    return payload


def _ancestor_void_receipts(path: Path) -> list[Path]:
    ancestors = []
    path = Path(path).resolve()
    for parent in [path.parent, *path.parents]:
        candidate = parent / VOID_RECEIPT_NAME
        if candidate.is_file():
            ancestors.append(candidate)
    return ancestors


def void_artifact_role(path: Path | str) -> str | None:
    """Return the void binding label when ``path`` is a void v3 artifact.

    A corrupt receipt blocks (raises) rather than being ignored.  Paths under
    the recorded void output root are rejected even when they are audit copies
    that are not individually hash-bound.
    """

    resolved = Path(path).resolve()
    for receipt_path in _ancestor_void_receipts(resolved):
        lite_dir = receipt_path.parent.parent
        payload = load_void_receipt(lite_dir)
        void_root = Path(str(payload.get("void_out_root") or "")).resolve()
        try:
            resolved.relative_to(void_root)
        except ValueError:
            pass
        else:
            return f"path under void output root {void_root}"
        for label, entry in (payload.get("files") or {}).items():
            bound = Path(str(entry.get("path") or "")).resolve()
            if resolved == bound:
                return label
            if resolved.is_file() and _sha256_file(resolved) == entry.get("sha256"):
                return f"{label} (identical bytes)"
    return None


def assert_not_void_artifact(path: Path | str, *, role: str) -> None:
    label = void_artifact_role(path)
    if label is not None:
        raise ShortfallVoidRunError(
            f"refusing {role} {Path(path)}: it is the void v3 artifact {label!r}; "
            "void artifacts are readable for audit but are not operative inputs"
        )


def _bound_file(path: Path, expected: str | None) -> bool:
    if not Path(path).is_file():
        return False
    if expected is None:
        return True
    return _sha256_file(Path(path)) == expected


def write_v3r_continuity(
    lite_dir: Path | str,
    *,
    ruling_path: Path | str,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Write the fixed continuity record linking crash -> void -> ruling -> v3r.

    The replacement review is recorded later, before the v3r claim, by
    :func:`record_v3r_replacement_review`.  The v3r registry is a distinct,
    fixed path and is created only by the atomic claim.
    """

    lite_dir = Path(lite_dir)
    receipt = load_void_receipt(lite_dir)
    ruling_path = Path(ruling_path)
    if not ruling_path.is_file():
        raise ShortfallV3RunError(f"the v3 void ruling is missing: {ruling_path}")
    if _sha256_file(ruling_path) != (receipt.get("ruling") or {}).get("sha256"):
        raise ShortfallV3RunError(
            "the continuity ruling is not the ruling bound by the void receipt"
        )
    registry_path = v3_registry_path(lite_dir)
    registry = _read_json(registry_path)
    if registry.get("kind") != V3_REGISTRY_KIND or registry.get(
        "status"
    ) != V3_STATUS_CRASHED:
        raise ShortfallV3RunError(
            "the continuity record requires the preserved crashed v3 registry"
        )
    expected_out = (lite_dir / "shortfall" / V3R_OUT_DIR_NAME).resolve()
    payload: dict[str, Any] = {
        "kind": V3R_CONTINUITY_KIND,
        "version": 1,
        "official": True,
        "created_utc": created_utc or _utc_now(),
        "disposition": VOID_DISPOSITION,
        "replacement_run": "v3r",
        "v3_registry": _binding(registry_path, role="crashed_v3_registry"),
        "void_receipt": _binding(
            void_receipt_path(lite_dir),
            role="void_receipt",
            expected=receipt["receipt_sha256"],
        ),
        "ruling": _binding(ruling_path, role="ruling"),
        "replacement_review": {
            "role": "replacement_review",
            "path": str(v3r_review_path(lite_dir).resolve()),
            "sha256": None,
            "status": "pending",
        },
        "v3r_registry": {
            "role": "v3r_registry",
            "path": str(v3r_registry_path(lite_dir).resolve()),
            "sha256": None,
            "status": "unclaimed",
        },
        "allowed_out_root": str(expected_out),
        "allowed_claims": 1,
        "attempt_policy": {
            "first_attempt": V3R_FIRST_ATTEMPT,
            "max_infrastructure_retries": V3R_MAX_INFRASTRUCTURE_RETRIES,
            "identical_configuration": True,
        },
        "continuity_sha256": "",
    }
    payload["continuity_sha256"] = _continuity_hash(payload)
    _write_json_exclusive(v3r_continuity_path(lite_dir), payload)
    return payload


def _load_v3r_continuity(
    lite_dir: Path | str, *, require_official: bool = False
) -> dict[str, Any]:
    path = v3r_continuity_path(lite_dir)
    if not path.is_file():
        raise ShortfallV3RunError(
            f"the v3r continuity record is missing; execution is blocked ({path})"
        )
    payload = _read_json(path)
    if payload.get("kind") != V3R_CONTINUITY_KIND:
        raise ShortfallV3RunError(f"{path}: not a v3r continuity record")
    if payload.get("continuity_sha256") != _continuity_hash(payload):
        raise ShortfallV3RunError(f"{path}: continuity self hash does not recompute")
    if payload.get("disposition") != VOID_DISPOSITION:
        raise ShortfallV3RunError(f"{path}: continuity disposition is inconsistent")
    expected_out = (Path(lite_dir) / "shortfall" / V3R_OUT_DIR_NAME).resolve()
    if Path(str(payload.get("allowed_out_root") or "")).resolve() != expected_out:
        raise ShortfallV3RunError(
            f"{path}: continuity does not authorize the fixed v3r output root"
        )
    if int(payload.get("allowed_claims") or 0) != 1:
        raise ShortfallV3RunError(f"{path}: continuity does not authorize exactly one claim")
    receipt_path = void_receipt_path(lite_dir)
    receipt_entry = payload.get("void_receipt") or {}
    if Path(str(receipt_entry.get("path") or "")).resolve() != receipt_path.resolve():
        raise ShortfallV3RunError(f"{path}: continuity targets a different void receipt")
    receipt = load_void_receipt(lite_dir, require_official=require_official)
    if receipt_entry.get("sha256") != _sha256_file(receipt_path):
        raise ShortfallV3RunError(f"{path}: void receipt changed since continuity")
    if receipt_entry.get("expected_sha256") not in {
        None,
        receipt.get("receipt_sha256"),
    }:
        raise ShortfallV3RunError(
            f"{path}: continuity does not bind the void receipt self hash"
        )
    registry_path = v3_registry_path(lite_dir)
    registry_entry = payload.get("v3_registry") or {}
    if Path(str(registry_entry.get("path") or "")).resolve() != registry_path.resolve():
        raise ShortfallV3RunError(f"{path}: continuity targets a different v3 registry")
    registry = _read_json(registry_path)
    if registry.get("kind") != V3_REGISTRY_KIND or registry.get(
        "status"
    ) != V3_STATUS_CRASHED:
        raise ShortfallV3RunError(f"{path}: the preserved v3 registry is inconsistent")
    if _sha256_file(registry_path) != registry_entry.get("sha256"):
        raise ShortfallV3RunError(f"{path}: the preserved v3 registry changed")
    ruling_entry = payload.get("ruling") or {}
    ruling_path = Path(str(ruling_entry.get("path") or ""))
    if not ruling_path.is_file() or _sha256_file(ruling_path) != ruling_entry.get("sha256"):
        raise ShortfallV3RunError(f"{path}: the ruling file changed or is missing")
    if ruling_entry.get("sha256") != (receipt.get("ruling") or {}).get("sha256"):
        raise ShortfallV3RunError(f"{path}: continuity and void receipt disagree on the ruling")
    registry_bound = payload.get("v3r_registry") or {}
    if Path(str(registry_bound.get("path") or "")).resolve() != v3r_registry_path(
        lite_dir
    ).resolve():
        raise ShortfallV3RunError(f"{path}: continuity targets a different v3r registry")
    return payload


def require_v3r_review(lite_dir: Path | str) -> dict[str, Any]:
    """Refuse the v3r claim unless the replacement review is recorded and present."""

    continuity = _load_v3r_continuity(lite_dir)
    entry = continuity.get("replacement_review") or {}
    if entry.get("status") != "recorded" or not entry.get("sha256"):
        raise ShortfallV3RunError(
            "the v3r claim is blocked: the replacement review has not been "
            "recorded in the fixed continuity record"
        )
    path = Path(str(entry.get("path") or ""))
    if not path.is_file():
        raise ShortfallV3RunError(f"the recorded replacement review is missing: {path}")
    if _sha256_file(path) != entry.get("sha256"):
        raise ShortfallV3RunError(f"the replacement review changed after it was recorded: {path}")
    review = _read_json(path)
    if review.get("kind") != V3R_REVIEW_KIND:
        raise ShortfallV3RunError(f"{path}: not a v3r replacement review")
    if review.get("go") is not True:
        raise ShortfallV3RunError(f"{path}: the replacement review did not issue GO")
    return {"continuity": continuity, "review": review, "review_path": path}


def record_v3r_replacement_review(
    lite_dir: Path | str,
    *,
    review_path: Path | str | None = None,
    go: bool = True,
    reviewer: str,
    run_version: str = "v3r",
    implementation: Mapping[str, Any] | None = None,
    require_official: bool = True,
) -> dict[str, Any]:
    """Atomically record the one replacement review before the v3r claim.

    This is the only permitted update of the continuity record.  A second
    recording, a non-GO review, or a review for anything other than v3r is
    refused.
    """

    lite_dir = Path(lite_dir)
    continuity = _load_v3r_continuity(
        lite_dir, require_official=require_official
    )
    existing = continuity.get("replacement_review") or {}
    if existing.get("status") == "recorded":
        raise ShortfallV3RunError("a replacement review is already recorded")
    path = Path(review_path or v3r_review_path(lite_dir))
    payload = {
        "kind": V3R_REVIEW_KIND,
        "version": 1,
        "run_version": run_version,
        "go": bool(go),
        "reviewer": str(reviewer),
        "created_utc": _utc_now(),
        "implementation": dict(implementation or {}),
        "replaces_void_receipt_sha256": (continuity.get("void_receipt") or {}).get(
            "sha256"
        ),
        "review_sha256": "",
    }
    payload["review_sha256"] = _hash_payload(payload, "review_sha256")
    if not go or run_version != "v3r":
        raise ShortfallV3RunError(
            "only an explicit GO for the exact v3r replacement may be recorded"
        )
    _write_json(path, payload)
    continuity["replacement_review"] = {
        "role": "replacement_review",
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "status": "recorded",
        "reviewer": str(reviewer),
    }
    continuity["continuity_sha256"] = _continuity_hash(continuity)
    _write_json(v3r_continuity_path(lite_dir), continuity)
    return continuity


def require_v3r_guard(
    lite_dir: Path | str,
    *,
    out_root: Path | str | None = None,
    require_review: bool = True,
    require_official: bool = True,
) -> dict[str, Any]:
    """The fixed one-time guard: refuse void identity, bypasses and rereuns."""

    lite_dir = Path(lite_dir)
    continuity = _load_v3r_continuity(
        lite_dir, require_official=require_official
    )
    expected = Path(str(continuity.get("allowed_out_root") or "")).resolve()
    if out_root is not None and Path(out_root).resolve() != expected:
        raise ShortfallV3RunError(
            f"the v3r output root {Path(out_root).resolve()} is not the fixed "
            f"continuity root {expected}; alternate-directory bypass is refused"
        )
    if require_review:
        require_v3r_review(lite_dir)
    registry_path = v3r_registry_path(lite_dir)
    if registry_path.exists():
        registry = _load_run_registry(registry_path, V3R_RUN_SPEC)
        if registry.get("continuity_sha256") != continuity.get("continuity_sha256"):
            raise ShortfallV3RunError(
                "the v3r registry does not link to the fixed continuity record"
            )
        if registry.get("void_receipt_sha256") != (continuity.get("void_receipt") or {}).get(
            "sha256"
        ):
            raise ShortfallV3RunError(
                "the v3r registry does not link to the fixed void receipt"
            )
        return {"continuity": continuity, "registry": registry, "status": "claimed"}
    if expected.exists() and (not expected.is_dir() or any(expected.iterdir())):
        raise ShortfallV3RunError(
            f"v3r output already exists at {expected} without a registry; a "
            "second fresh replacement is refused"
        )
    return {"continuity": continuity, "registry": None, "status": "unclaimed"}


def _refuse_if_void_identity(lite_dir: Path | str) -> None:
    """Reject any execution under the preserved void v3 identity."""

    if void_receipt_path(lite_dir).exists():
        receipt = load_void_receipt(lite_dir)
        raise ShortfallVoidRunError(
            "the preserved v3 run is VOID_VERIFIER_IMPLEMENTATION_FAILURE "
            f"({receipt.get('disposition')}); execution under the void identity "
            "is refused. The only authorized replacement is v3r."
        )


def _refuse_if_official_v3_started(lite_dir: Path | str) -> None:
    """Refuse a fresh freeze or recount after the one-time run started."""

    _refuse_if_void_identity(lite_dir)
    path = v3_registry_path(lite_dir)
    if path.exists():
        raise ShortfallV3RunError(
            "the official v3 run has already started; refusing another freeze "
            f"or fresh recount ({path})"
        )


def _registry_hash(payload: Mapping[str, Any]) -> str:
    return _hash_payload(payload, "registry_sha256")


def _seal_registry_claim(payload: dict[str, Any]) -> dict[str, Any]:
    """Capture the immutable claim record digest, then the registry self hash.

    ``claim_registry_sha256`` is computed once, at claim time, over the claim
    record with both hash fields empty.  Registry updates never change it, so
    the durable attempt ledger can bind the immutable run identity without
    pinning the evolving registry file hash.
    """

    payload["claim_registry_sha256"] = ""
    payload["registry_sha256"] = ""
    payload["claim_registry_sha256"] = _hash_payload(payload, "registry_sha256")
    payload["registry_sha256"] = _registry_hash(payload)
    return payload


def _registry_claim_digest(payload: Mapping[str, Any]) -> str:
    """Return the immutable claim digest (legacy registries fall back once)."""

    return str(
        payload.get("claim_registry_sha256")
        or payload.get("registry_sha256")
        or ""
    )


def _registry_completed_stages(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(entry.get("stage") or "")
        for entry in payload.get("stages") or []
        if entry.get("status") == "complete"
    ]


def _validate_registry_stage_history(
    payload: Mapping[str, Any], *, stage_names: Sequence[str]
) -> None:
    """Refuse a malformed or non-appendable run registry stage history."""

    seen: set[str] = set()
    allowed = set(stage_names)
    for entry in payload.get("stages") or []:
        if not isinstance(entry, Mapping):
            raise ShortfallV3RunError("the run registry stage history is invalid")
        stage = str(entry.get("stage") or "")
        if stage not in allowed:
            raise ShortfallV3RunError(
                f"the run registry stage history names unknown stage {stage!r}"
            )
        if stage in seen:
            raise ShortfallV3RunError(
                f"the run registry stage history repeats stage {stage!r}"
            )
        seen.add(stage)
        if entry.get("status") not in {"running", "complete", "failed"}:
            raise ShortfallV3RunError(
                f"the run registry stage {stage!r} has invalid status "
                f"{entry.get('status')!r}"
            )
    last = payload.get("last_completed_stage")
    if last is not None and str(last) not in seen:
        raise ShortfallV3RunError(
            "the run registry last_completed_stage is not in its stage history"
        )
    completed = _registry_completed_stages(payload)
    if len(completed) != len(set(completed)):
        raise ShortfallV3RunError(
            "the run registry completes a stage more than once"
        )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _claim_v3_registry(
    *,
    repo_root: Path,
    lite_dir: Path,
    out_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Claim the one-time official v3 run before writing any output."""

    _refuse_if_void_identity(lite_dir)
    path = v3_registry_path(lite_dir)
    if path.exists():
        try:
            existing = _read_json(path)
            status = existing.get("status")
            stage = existing.get("last_completed_stage") or existing.get(
                "current_stage"
            )
            detail = f"status={status!r}, stage={stage!r}"
        except Exception:  # noqa: BLE001 - a torn registry still blocks a rerun
            detail = "unreadable registry"
        raise ShortfallV3RunError(
            "the official v3 run has already started; refusing another freeze "
            f"or fresh recount ({path}: {detail})"
        )
    if out_root.exists() and (
        not out_root.is_dir() or any(out_root.iterdir())
    ):
        raise ShortfallV3RunError(
            "v3 output already exists and may not be clobbered: "
            f"{out_root} (no official registry at {path})"
        )
    payload: dict[str, Any] = {
        "kind": V3_REGISTRY_KIND,
        "version": 1,
        "status": V3_STATUS_CLAIMED,
        "created_utc": _now_utc(),
        "updated_utc": _now_utc(),
        "official": True,
        "run_token": secrets.token_hex(32),
        "repo_root": str(repo_root.resolve()),
        "lite_dir": str(lite_dir.resolve()),
        "out_root": str(out_root.resolve()),
        "current_stage": None,
        "last_completed_stage": None,
        "failed_stage": None,
        "stages": [],
        "resume_count": 0,
        "error": None,
        "claim_registry_sha256": "",
        "registry_sha256": "",
    }
    _seal_registry_claim(payload)
    _write_json_exclusive(path, payload)
    return path, payload


def _update_v3_registry(
    path: Path,
    payload: dict[str, Any],
    *,
    stage: str | None = None,
    stage_status: str | None = None,
    status: str | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Durably advance the registry; a crash leaves a blocked crash state."""

    now = _now_utc()
    if status is not None:
        payload["status"] = status
    if stage is not None:
        payload["current_stage"] = stage
    if failed_stage is not None:
        payload["failed_stage"] = failed_stage
    if error is not None:
        payload["error"] = error
    if stage is not None and stage_status is not None:
        entries = list(payload.get("stages") or [])
        for entry in entries:
            if entry.get("stage") == stage:
                entry["status"] = stage_status
                entry["updated_utc"] = now
                if stage_status in {"complete", "failed"}:
                    entry["completed_utc"] = now
                if artifacts is not None:
                    entry["artifacts"] = dict(artifacts)
                break
        else:
            entry = {
                "stage": stage,
                "status": stage_status,
                "started_utc": now,
                "updated_utc": now,
                "artifacts": dict(artifacts or {}),
            }
            if stage_status in {"complete", "failed"}:
                entry["completed_utc"] = now
            entries.append(entry)
        payload["stages"] = entries
        if stage_status == "complete":
            payload["last_completed_stage"] = stage
    payload["updated_utc"] = now
    payload["registry_sha256"] = _registry_hash(payload)
    _write_json(path, payload)
    return payload


@dataclass(frozen=True)
class V3RunAuthorization:
    """Capability to execute one stage of the claimed official run.

    ``exclusive=True`` alone is never authorization: the caller must present
    the token persisted in the fixed one-time registry and target the stage
    directory recorded by that run.  ``run_version`` selects the v3 or v3r
    naming/guard identity; it defaults to v3 so existing callers are unchanged.
    """

    registry_path: Path
    run_token: str
    stage: str
    run_version: str = "v3"


def _load_run_registry(path: Path, spec: _RunSpec) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("kind") != spec.registry_kind:
        raise ShortfallV3RunError(
            f"{path}: not an official {spec.version} registry"
        )
    if payload.get("official") is not True:
        raise ShortfallV3RunError(f"{path}: the run is not marked official")
    if str(payload.get("run_version") or spec.version) != spec.version:
        raise ShortfallV3RunError(
            f"{path}: registry run version is not {spec.version!r}"
        )
    if payload.get("registry_sha256") != _registry_hash(payload):
        raise ShortfallV3RunError(f"{path}: registry hash does not recompute")
    if spec.version == "v3r" and not payload.get("claim_registry_sha256"):
        raise ShortfallV3RunError(
            f"{path}: the v3r registry has no immutable claim digest"
        )
    _validate_registry_stage_history(payload, stage_names=spec.stages)
    return payload


def _load_v3_registry(path: Path) -> dict[str, Any]:
    return _load_run_registry(Path(path), V3_RUN_SPEC)


def _authorize_v3_stage(
    authorization: V3RunAuthorization | None,
    *,
    stage: str,
    out_dir: Path | str | None = None,
    lite_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Refuse an unclaimed/unauthorized stage execution."""

    if authorization is None:
        raise ShortfallV3RunError(
            f"the {stage} stage may only run as part of the claimed official "
            "run or an authorized resume; exclusive=True is not authorization"
        )
    spec = _run_spec(getattr(authorization, "run_version", "v3"))
    if stage not in spec.stages:
        raise ShortfallV3RunError(f"unknown {spec.version} stage: {stage!r}")
    registry_path = Path(authorization.registry_path)
    if not registry_path.is_file():
        raise ShortfallV3RunError(
            f"the official {spec.version} registry is missing: {registry_path}"
        )
    registry = _load_run_registry(registry_path, spec)
    stage_lite = Path(lite_dir or registry.get("lite_dir") or "")
    if spec.version == "v3":
        _refuse_if_void_identity(stage_lite)
    else:
        expected_registry = v3r_registry_path(stage_lite).resolve()
        if registry_path.resolve() != expected_registry:
            raise ShortfallV3RunError(
                "a v3r stage must use the fixed v3r registry path "
                f"({expected_registry}), not {registry_path}"
            )
        guard = require_v3r_guard(
            stage_lite,
            out_root=Path(str(registry.get("out_root") or "")),
            require_review=False,
            require_official=not bool(
                (load_void_receipt(stage_lite).get("synthetic"))
            ),
        )
        if guard.get("status") != "claimed":
            raise ShortfallV3RunError(
                "the v3r stage is not authorized by the fixed continuity record"
            )
    if registry.get("status") not in {
        V3_STATUS_CLAIMED,
        V3_STATUS_RUNNING,
        V3_STATUS_CRASHED,
        V3_STATUS_RESUMABLE,
    }:
        raise ShortfallV3RunError(
            f"the official {spec.version} run is already complete; no stage "
            "may run again"
        )
    if not secrets.compare_digest(
        str(authorization.run_token), str(registry.get("run_token") or "")
    ):
        raise ShortfallV3RunError(
            f"the {spec.version} stage authorization token does not match the "
            "claimed run"
        )
    if lite_dir is not None and Path(lite_dir).resolve() != Path(
        str(registry.get("lite_dir") or "")
    ).resolve():
        raise ShortfallV3RunError(
            f"the {spec.version} stage targets a different lite directory than "
            "the claimed run"
        )
    if out_dir is not None:
        out_root = Path(str(registry.get("out_root") or "")).resolve()
        directory = spec.stage_directories.get(stage)
        expected = (
            (out_root / directory).resolve()
            if directory
            else (
                out_root / spec.plan_name
                if stage == "decision"
                else out_root / spec.index_name
            ).resolve()
        )
        if Path(out_dir).resolve() != expected:
            raise ShortfallV3RunError(
                f"the {stage} stage target {Path(out_dir).resolve()} is not the "
                f"directory recorded by the claimed run ({expected})"
            )
    return registry


def _claim_v3r_registry(
    *,
    repo_root: Path,
    lite_dir: Path,
    out_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Atomically claim the single authorized v3r replacement.

    The fixed continuity record, the immutable void receipt, the replacement
    review and the absent v3r registry are all verified *before* any output or
    dispatch.  Only the exact fixed output root is accepted; a second fresh
    replacement or a void-identity run is refused.
    """

    lite_dir = Path(lite_dir)
    out_root = Path(out_root)
    guard = require_v3r_guard(
        lite_dir,
        out_root=out_root,
        require_review=True,
        require_official=not bool(
            (load_void_receipt(lite_dir).get("synthetic"))
        ),
    )
    continuity = guard["continuity"]
    if guard.get("status") == "claimed":
        registry = guard.get("registry") or {}
        detail = (
            f"status={registry.get('status')!r}, "
            f"stage={registry.get('last_completed_stage')!r}"
        )
        raise ShortfallV3RunError(
            "the one authorized v3r replacement has already been claimed; "
            f"refusing a second fresh replacement ({detail})"
        )
    path = v3r_registry_path(lite_dir)
    if Path(str(continuity.get("allowed_out_root") or "")).resolve() != out_root.resolve():
        raise ShortfallV3RunError(
            "the v3r claim does not target the fixed continuity output root"
        )
    if out_root.exists() and (not out_root.is_dir() or any(out_root.iterdir())):
        raise ShortfallV3RunError(
            f"v3r output already exists and may not be clobbered: {out_root}"
        )
    void_receipt_sha = (continuity.get("void_receipt") or {}).get("sha256")
    payload: dict[str, Any] = {
        "kind": V3R_REGISTRY_KIND,
        "version": 1,
        "run_version": "v3r",
        "status": V3_STATUS_CLAIMED,
        "created_utc": _now_utc(),
        "updated_utc": _now_utc(),
        "official": True,
        "replacement": True,
        "replaces_void_receipt_sha256": void_receipt_sha,
        "continuity_sha256": continuity.get("continuity_sha256"),
        "void_receipt_sha256": void_receipt_sha,
        "attempt_base": V3R_FIRST_ATTEMPT,
        "max_infrastructure_retries": V3R_MAX_INFRASTRUCTURE_RETRIES,
        "run_token": secrets.token_hex(32),
        "repo_root": str(Path(repo_root).resolve()),
        "lite_dir": str(lite_dir.resolve()),
        "out_root": str(out_root.resolve()),
        "current_stage": None,
        "last_completed_stage": None,
        "failed_stage": None,
        "stages": [],
        "resume_count": 0,
        "error": None,
        "claim_registry_sha256": "",
        "registry_sha256": "",
    }
    _seal_registry_claim(payload)
    _write_json_exclusive(path, payload)
    return path, payload


def _hash_payload(payload: Mapping[str, Any], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _file_entry(label: str, path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _canonical_test_bytes(test: Mapping[str, Any]) -> bytes:
    return json.dumps(test, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _problem_tests_sha256(tests: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for test in tests:
        data = _canonical_test_bytes(test)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------
def _load_split_rows(pool_dir: Path) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for name in PILOT_SPLITS:
        path = Path(pool_dir) / f"split_{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing pilot split file: {path}")
        split_rows[name] = local.read_jsonl(path)
    return split_rows


def _load_request_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate request ids")
    for request_id in ids:
        if ":" not in request_id:
            raise ValueError(f"{path}: malformed request id {request_id!r}")
    return ids


def select_original_inputs(
    pool_dir: Path | str,
    request_ids_path: Path | str,
) -> dict[str, Any]:
    """The original selected input sets (not successful outputs).

    A problem belongs to a pilot split when it appears in that split file *and*
    the lite run requested it.  Omitted bucket members are kept visible and are
    never added back.  Problems with zero successful samples are included; the
    caller can derive the successful subset from the per-request grades.
    """

    pool_dir = Path(pool_dir)
    request_ids = _load_request_ids(Path(request_ids_path))
    split_rows = _load_split_rows(pool_dir)
    requested = set()
    indices_by_problem: dict[str, list[int]] = {}
    for request_id in request_ids:
        problem_id, separator, index_text = request_id.rpartition(":")
        if not separator or not index_text.isdigit():
            raise ValueError(f"malformed request id {request_id!r}")
        requested.add(problem_id)
        indices_by_problem.setdefault(problem_id, []).append(int(index_text))

    splits: dict[str, Any] = {}
    for name in PILOT_SPLITS:
        rows = split_rows[name]
        selected_rows = [row for row in rows if str(row["problem_id"]) in requested]
        omitted = [
            str(row["problem_id"])
            for row in rows
            if str(row["problem_id"]) not in requested
        ]
        problems = []
        requests = []
        for row in selected_rows:
            problem_id = str(row["problem_id"])
            kind = str(row.get("kind") or "")
            problems.append(
                {
                    "problem_id": problem_id,
                    "kind": kind,
                    "source": row.get("source"),
                    "split": name,
                }
            )
            for index in sorted(indices_by_problem.get(problem_id, [])):
                requests.append(
                    {
                        "request_id": f"{problem_id}:{index}",
                        "problem_id": problem_id,
                        "sample_index": index,
                        "kind": kind,
                        "source": row.get("source"),
                        "split": name,
                    }
                )
        counts: dict[str, int] = {}
        for row in selected_rows:
            counts[str(row.get("kind"))] = counts.get(str(row.get("kind")), 0) + 1
        splits[name] = {
            "problem_ids": [str(row["problem_id"]) for row in selected_rows],
            "problems": problems,
            "requests": requests,
            "omitted_problem_ids": omitted,
            "counts_by_kind": dict(sorted(counts.items())),
            "problem_count": len(selected_rows),
            "request_count": len(requests),
        }
    payload = {
        "version": 1,
        "rule": (
            "split-file members also requested by the immutable lite run; "
            "omitted bucket members are recorded, never added"
        ),
        "request_ids_file": str(Path(request_ids_path).resolve()),
        "request_ids_sha256": _sha256_file(Path(request_ids_path)),
        "splits": splits,
    }
    payload["selected_inputs_sha256"] = _hash_payload(payload, "selected_inputs_sha256")
    return payload


# ---------------------------------------------------------------------------
# Verifier test manifest
# ---------------------------------------------------------------------------
def test_manifest_rows(
    selected_inputs: Mapping[str, Any],
    pool_problems: Iterable[Mapping[str, Any]],
    *,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(rows, exclusions)`` for the ordered code-test manifest."""

    problems_by_id = {str(row["problem_id"]): row for row in pool_problems}
    provided_splits = {
        name: {str(row["problem_id"]): row for row in rows}
        for name, rows in (split_rows or {}).items()
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for name in PILOT_SPLITS:
        split = selected_inputs["splits"][name]
        for problem in split["problems"]:
            if problem["kind"] != "code":
                continue
            problem_id = problem["problem_id"]
            source = problems_by_id.get(problem_id)
            if source is None:
                raise ValueError(f"code problem {problem_id!r} is not in the pool")
            tests = source.get("tests")
            if not tests:
                exclusions.append(
                    {
                        "problem_id": problem_id,
                        "split": name,
                        "reason": "no_tests",
                    }
                )
                continue
            tests = list(tests)
            # The split files are a second copy of the same frozen pool rows.
            # A mismatch means the test bytes changed and the freeze must fail.
            if provided_splits.get(name, {}).get(problem_id) is not None:
                split_tests = provided_splits[name][problem_id].get("tests")
                if list(split_tests or []) != tests:
                    raise ValueError(
                        f"test bytes for {problem_id!r} differ between "
                        f"problems.jsonl and split_{name}.jsonl"
                    )
            canonical = [_canonical_test_bytes(test).decode("utf-8") for test in tests]
            rows.append(
                {
                    "problem_id": problem_id,
                    "split": name,
                    "kind": "code",
                    "source": problem.get("source"),
                    "problem_problem_id": problem_id,
                    "n_tests": len(tests),
                    "test_bytes": canonical,
                    "test_sha256": [
                        _sha256_text(text) for text in canonical
                    ],
                    "tests_sha256": _problem_tests_sha256(tests),
                }
            )
    return rows, exclusions


def _test_manifest_file_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Docker digest pinning
# ---------------------------------------------------------------------------
def _run_docker(argv: Sequence[str], *, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=float(timeout),
    )


def resolve_docker_image(
    image: str = LITE_IMAGE_TAG,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = _run_docker,
    allow_pull: bool = True,
) -> dict[str, Any]:
    """Resolve a tag/mutable reference to a content-addressed digest.

    ``docker image inspect`` is the only source of truth: a tag name or even a
    ``RepoDigests`` label is not accepted without the matching ``Id``.
    """

    completed = runner(["docker", "image", "inspect", str(image)])
    if completed.returncode != 0 and allow_pull:
        runner(["docker", "pull", str(image)], timeout=600.0)
        completed = runner(["docker", "image", "inspect", str(image)])
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker image inspect {image!r} failed: {completed.stderr!r}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"docker image inspect {image!r} returned non-JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"docker image inspect {image!r} returned no image")
    entry = payload[0]
    image_id = str(entry.get("Id") or "")
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"docker image {image!r} has no content Id")
    repo_digests = [str(value) for value in (entry.get("RepoDigests") or [])]
    digest = image_id
    pinned = None
    for value in repo_digests:
        repository, separator, content = value.partition("@")
        if separator and content.startswith("sha256:"):
            # Prefer a digest that matches the image id; otherwise the
            # repository digest is the stable pull reference.
            if content == image_id:
                pinned = value
                digest = content
                break
            if pinned is None:
                pinned = value
                digest = content
    return {
        "tag": str(image),
        "id": image_id,
        "digest": digest,
        "repo_digests": repo_digests,
        "pinned": pinned or image_id,
    }


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
_REPO_FILES = (
    "src/gemmaloop/stage0/shortfall.py",
    "src/gemmaloop/stage0/interpretation.py",
    "src/gemmaloop/stage0/source_recovery.py",
    "src/gemmaloop/stage0/sources.py",
    "src/gemmaloop/stage0/local.py",
    "src/gemmaloop/stage0/pool.py",
    "src/gemmaloop/stage0/grader.py",
    "src/gemmaloop/stage0/postprocess.py",
    "src/gemmaloop/eval/benchmarks/lcb.py",
    "src/gemmaloop/train/labels.py",
    "src/gemmaloop/pilot/eval_rows.py",
)

# The modules whose *actual resolved files* execute the count.  Binding module
# names is not enough: a second checkout or an edited module must be refused.
_EXECUTING_MODULES = (
    "gemmaloop.stage0.shortfall",
    "gemmaloop.stage0.interpretation",
    "gemmaloop.stage0.source_recovery",
    "gemmaloop.stage0.sources",
    "gemmaloop.stage0.local",
    "gemmaloop.stage0.pool",
    "gemmaloop.stage0.grader",
    "gemmaloop.stage0.postprocess",
    "gemmaloop.eval.benchmarks.lcb",
    "gemmaloop.train.labels",
    "gemmaloop.pilot.eval_rows",
)

_ENTRY_POINT_FILES = (
    "scripts/recover_source_types.py",
    "scripts/freeze_pilot_verifier.py",
    "scripts/count_pilot_verified.py",
    "scripts/plan_pilot_topup.py",
    "scripts/build_pilot_corpus.py",
    "scripts/freeze_recount_v3.py",
    "scripts/freeze_recount_v3r.py",
    "scripts/run_pilot.py",
    "scripts/modal_stage0.py",
)


def code_implementation_hashes(repo_root: Path | str) -> dict[str, str]:
    repo_root = Path(repo_root)
    return {
        str(path): _sha256_file(repo_root / path)
        for path in _REPO_FILES
        if (repo_root / path).is_file()
    }


def _module_files() -> dict[str, Path]:
    import importlib

    files: dict[str, Path] = {}
    for name in _EXECUTING_MODULES:
        module = importlib.import_module(name)
        location = getattr(module, "__file__", None)
        if not location:
            raise ValueError(f"executing module {name!r} has no resolved file")
        files[name] = Path(location).resolve()
    return files


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--always", "--dirty", "--tags"),
        "dirty": bool(status),
        "status_sha256": _sha256_text(status or ""),
    }


def _git_commit(repo_root: Path) -> str | None:
    return _git_provenance(Path(repo_root)).get("commit")


def executing_implementation(
    *,
    repo_root: Path | str | None = None,
    module_files: Mapping[str, Path | str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the actual imported implementation files and entry points."""

    files = (
        {str(name): Path(path).resolve() for name, path in module_files.items()}
        if module_files is not None
        else _module_files()
    )
    if not files:
        raise ValueError("no executing module files to bind")
    inferred_root = Path(next(iter(files.values()))).resolve().parents[3]
    root = Path(repo_root).resolve() if repo_root is not None else inferred_root
    modules = []
    for name, path in sorted(files.items()):
        if not path.is_file():
            raise ValueError(f"executing module {name!r} is missing: {path}")
        modules.append(
            {
                "module": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    entry_points = []
    for relative in _ENTRY_POINT_FILES:
        path = (root / relative).resolve()
        if path.is_file():
            entry_points.append(
                {
                    "label": relative,
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    return {
        "repo_root": str(root),
        "modules": modules,
        "entry_points": entry_points,
        "code_hashes": code_implementation_hashes(root),
        "git": dict(provenance or _git_provenance(root)),
        "interpreter": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "python": sys.version.split()[0],
        },
        "dependencies": _dependency_versions(),
    }


def _dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
    }
    for distribution in ("transformers", "math-verify", "huggingface-hub"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        completed = _run_docker(["docker", "version", "--format", "{{.Server.Version}}"])
        versions["docker_server"] = (
            completed.stdout.strip() if completed.returncode == 0 else None
        )
    except (OSError, subprocess.SubprocessError):
        versions["docker_server"] = None
    return versions


def freeze_verifier(
    *,
    repo_root: Path | str,
    pool_dir: Path | str,
    lite_dir: Path | str,
    request_ids_path: Path | str,
    out_dir: Path | str,
    image: str = LITE_IMAGE_TAG,
    image_pin: Mapping[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    git_commit: str | None = None,
    interpretation_path: Path | str | None = None,
    workers: int = 1,
    lifecycle_policy: Mapping[str, Any] | None = None,
    retry_policy: Mapping[str, Any] | None = None,
    exclusive: bool = False,
    authorization: V3RunAuthorization | None = None,
    resume: bool = False,
    run_version: str = "v3",
) -> dict[str, Any]:
    """Write the immutable verifier freeze and ordered test manifest."""

    repo_root = Path(repo_root)
    pool_dir = Path(pool_dir)
    lite_dir = Path(lite_dir)
    out_dir = Path(out_dir)
    if authorization is not None:
        _authorize_v3_stage(
            authorization, stage="freeze", out_dir=out_dir, lite_dir=lite_dir
        )
        if resume:
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            _require_new_dir(out_dir)
        exclusive_write = not resume
    elif exclusive:
        raise ShortfallV3RunError(
            "exclusive freeze requires the authorization of the claimed "
            "official v3 run; exclusive=True is not authorization"
        )
    else:
        _refuse_if_official_v3_started(lite_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exclusive_write = False

    pool_problems = local.read_jsonl(pool_dir / "problems.jsonl")
    split_rows = _load_split_rows(pool_dir)
    selected = select_original_inputs(pool_dir, request_ids_path)
    rows, exclusions = test_manifest_rows(
        selected, pool_problems, split_rows=split_rows
    )
    test_manifest_path = out_dir / FREEZE_TEST_MANIFEST_NAME
    test_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if exclusive_write:
        _write_text_exclusive(test_manifest_path, test_text)
    else:
        atomic_write_text(test_manifest_path, test_text)
    test_manifest_sha = _sha256_text(test_text)

    pin = dict(image_pin) if image_pin is not None else resolve_docker_image(image)

    interpretation_entry: dict[str, Any] | None = None
    if interpretation_path is not None:
        interpretation_path = Path(interpretation_path)
        interpretation = load_interpretation_manifest(interpretation_path)
        summary = verify_interpretation_against_test_manifest(
            interpretation, rows, manifest_path=interpretation_path
        )
        execution = interpretation.get("execution_tests") or {}
        interpretation_entry = {
            "manifest": str(interpretation_path.resolve()),
            "manifest_sha256": interpretation["manifest_sha256"],
            "manifest_file_sha256": _sha256_file(interpretation_path),
            "execution_tests_file": str(
                (interpretation_path.parent / str(execution.get("file"))).resolve()
            ),
            "execution_tests_sha256": execution.get("sha256"),
            "authenticated_problem_count": int(
                interpretation.get("authenticated_problem_count") or 0
            ),
            "excluded_problem_count": int(
                interpretation.get("excluded_problem_count") or 0
            ),
            "excluded_problem_ids": sorted(
                str(item["problem_id"])
                for item in interpretation.get("exclusions") or []
            ),
            "exclusions": list(interpretation.get("exclusions") or []),
            "verified": summary,
        }

    executing = executing_implementation(repo_root=repo_root)
    evidence_files: list[dict[str, Any]] = []
    for label, path in (
        ("pool_manifest", pool_dir / "manifest.json"),
        ("pool_problems", pool_dir / "problems.jsonl"),
        ("request_ids", Path(request_ids_path)),
        ("lite_manifest", lite_dir / "manifest.json"),
        ("lite_run_marker", lite_dir / "run_marker.json"),
        ("lite_modal_receipt", lite_dir / "modal_receipt.json"),
        ("lite_sample_grades", lite_dir / "sample_grades.jsonl"),
        ("lite_pilot_manifest", lite_dir / "pilot" / "manifest.json"),
        ("lite_pilot_p1", lite_dir / "pilot" / "pilot_p1.jsonl"),
        ("lite_pilot_tfit", lite_dir / "pilot" / "pilot_tfit.jsonl"),
        ("lite_pilot_selection", lite_dir / "pilot" / "pilot_selection.jsonl"),
    ):
        if Path(path).is_file():
            evidence_files.append(_file_entry(label, Path(path)))
    for name in PILOT_SPLITS:
        evidence_files.append(
            _file_entry(f"split_{name}", pool_dir / f"split_{name}.jsonl")
        )
    shard_dir = lite_dir / "shards"
    for path in sorted(shard_dir.glob("shard_*")):
        evidence_files.append(_file_entry(f"lite_shard/{path.name}", path))

    run_marker = (
        _read_json(lite_dir / "run_marker.json")
        if (lite_dir / "run_marker.json").is_file()
        else {}
    )
    lite_run_id = str(run_marker.get("run_id") or LITE_RUN_ID_FALLBACK)
    pilot_manifest = (
        _read_json(lite_dir / "pilot" / "manifest.json")
        if (lite_dir / "pilot" / "manifest.json").is_file()
        else {}
    )
    model_revision = pilot_manifest.get("model_revision")

    payload: dict[str, Any] = {
        "kind": "pilot_shortfall_verifier_freeze",
        "version": 1,
        "run_version": str(run_version),
        "created_utc": _utc_now(),
        "git_commit": git_commit or executing["git"].get("commit"),
        "lite_run_id": lite_run_id,
        "model_revision": model_revision,
        "inputs": {
            "repo_root": str(repo_root.resolve()),
            "pool_dir": str(pool_dir.resolve()),
            "lite_dir": str(lite_dir.resolve()),
            "files": evidence_files,
        },
        "selected_inputs": selected,
        "filter": {
            "finish_reason": "stop",
            "thought_closed": True,
            "non_empty_answer": True,
            "rendered_max_length": int(MAX_LENGTH),
            "reasoning_share_min": float(MIN_REASONING_SHARE),
            "domains": "math/code",
            "chat_verified": False,
        },
        "verifier": {
            "timeout_s": float(timeout_s),
            "interpreter": "python -I",
            "extraction": "src/gemmaloop/stage0/postprocess.py extract_code",
            "wrapper": (
                "src/gemmaloop/stage0/grader.py _write_lifecycle_wrapper + "
                "_write_lifecycle_runner + _functional_runner"
            ),
            "post_exec_runner": {
                "file": LIFECYCLE_RUNNER_NAME,
                "sha256": _sha256_text(_LIFECYCLE_RUNNER_SOURCE),
                "bytes": len(_LIFECYCLE_RUNNER_SOURCE.encode("utf-8")),
                "source": _LIFECYCLE_RUNNER_SOURCE,
                "entry_rule": (
                    "the frozen runner is the exec'd image; it writes the "
                    "post_exec_entry frame on the inherited write-only pipe fd "
                    "and closes that fd before running any candidate code with "
                    "runpy; a pre-exec exec_ok frame is never entry evidence"
                ),
            },
            "wrapper_evidence": {
                "record_file": "/evidence/lifecycle.jsonl",
                "budget_file": "/evidence/budget.json",
                "record_rule": (
                    "the grader sends a fresh 64-byte per-attempt key on stdin "
                    "before the test input; frozen container PID 1 reads it, "
                    "forks a bootstrap child and keeps the key out of candidate "
                    "arguments, environment, files, globals and live runner "
                    "frames; the child closes the evidence descriptor, writes "
                    "a pre-exec diagnostic frame on a private child-to-wrapper "
                    "pipe and execs a frozen launcher, which execs a minimal "
                    "frozen runner; the runner (now the exec'd image) writes "
                    "the post_exec_entry frame on the inherited write-only pipe "
                    "and closes it before any candidate code runs (explicit "
                    "handoff_failed on failure)"
                ),
                "authentication": (
                    "HMAC-SHA-256 over the literal GML1 prefix and canonical "
                    "JSON, with attempt identity, monotonic sequence, child pid "
                    "and framed records; the host's deadline-reached notice is "
                    "authenticated over the same per-attempt key"
                ),
                "handoff": (
                    "successful execution entry is positively established only "
                    "by the frozen runner's post_exec_entry frame (written by "
                    "the exec'd image and closed before candidate code), the "
                    "setup pipe EOF and the pre-exec diagnostic pipe EOF; a "
                    "launcher exec_ok frame emitted before the runner exec is "
                    "not entry evidence, so death in that interval is "
                    "infrastructure/unknown"
                ),
                "evidence_mount": "/evidence:rw",
                "candidate_channel": "stdout/stderr only (never trusted for lifecycle)",
                "shared_mount_role": "transports authenticated bytes only; it grants no authenticity or liveness",
            },
            "comparison": "src/gemmaloop/stage0/grader.py _compare_output",
            "implementation": "src/gemmaloop/stage0/grader.py DockerSandbox",
            "executing": executing,
            "code_hashes": executing["code_hashes"],
            "dependencies": executing["dependencies"],
            "lifecycle": {
                **(lifecycle_policy
                    or {
                    "outer_timeout_s": float(timeout_s),
                    "timeout_scope": "whole docker run process including startup",
                    "clock_restart": False,
                    "candidate_running_at_deadline": "outer_deadline_failure",
                    "candidate_exited_after_deadline": "infrastructure_unknown",
                    "startup_failure": "infrastructure_unknown",
                    "candidate_completed_before_deadline": "infrastructure_unknown",
                    "unresolved": "infrastructure_unknown",
                    "candidate_lifecycle_evidence": (
                        "authenticated bootstrap/entry/budget/liveness/finish "
                        "records; container state and candidate output alone "
                        "are insufficient"
                    ),
                    "candidate_running_class_requires": (
                        "authenticated execution entry received by the host "
                        "before its registered deadline on the same host "
                        "monotonic clock, a positive waitpid non-exit "
                        "observation made after the authenticated notice and a "
                        "verified termination record"
                    ),
                    "execution_entry_receipt": (
                        "the wrapper forwards the start frame as soon as the "
                        "post-exec entry is observed; the host stamps the "
                        "receipt on the same monotonic clock as its deadline"
                    ),
                    "deadline_notice": (
                        "host writes an authenticated deadline_reached notice "
                        "at the single registered outer deadline before it "
                        "enforces the kill inside the frozen bounded "
                        "observation/cleanup window; the wrapper computes no "
                        "deadline of its own and never compares host/container "
                        "epochs"
                    ),
                    "deadline_notice_grace_s": float(
                        LIFECYCLE_NOTICE_GRACE_S
                    ),
                    "post_deadline_observation_window_s": float(
                        LIFECYCLE_NOTICE_OBSERVATION_S
                    ),
                    "deadline_enforcement_start_s": float(
                        DEADLINE_ENFORCEMENT_START_S
                    ),
                    "deadline_enforcement_issue_bound_s": float(
                        LIFECYCLE_NOTICE_GRACE_S
                    ),
                    "deadline_enforcement_bound_s": float(
                        LIFECYCLE_NOTICE_GRACE_S
                    ),
                    "deadline_enforcement_bound_rule": (
                        "every expected enforcement action (the docker client "
                        "hard kill, docker rm -f and docker kill -s KILL) must "
                        "have a terminal, successful return timestamp by the "
                        "registered deadline plus 0.25 s; a missing, extra, "
                        "still-scheduled, in-flight, failed or late action is "
                        "an overrun and the attempt fails closed as "
                        "infrastructure/unknown with cleanup pending"
                    ),
                    "deadline_enforcement_extra_s": float(
                        DEADLINE_ENFORCEMENT_EXTRA_S
                    ),
                    "enforcement_actions": list(
                        DEADLINE_ENFORCEMENT_ACTIONS
                    ),
                    "enforcement_primitives": [
                        "docker rm -f <name>",
                        "docker kill -s KILL <name>",
                    ],
                    "enforcement_issuance_rule": (
                        "one evidence slot per expected enforcement action is "
                        "pre-created with a scheduled status and scheduling "
                        "timestamp before any launcher thread starts; "
                        "enforcement begins at the registered deadline plus "
                        "the frozen observation sub-window; the host first "
                        "hard-kills the docker client, then launches every "
                        "container-kill primitive concurrently in its own "
                        "thread before any state probe; workers only fill "
                        "their pre-created slot, the collector never drops a "
                        "slot, and any earlier late/in-flight observation is "
                        "carried into the final overrun computation"
                    ),
                    "post_deadline_bytes_and_exit_status": (
                        "excluded from the outcome; usable only as "
                        "attribution/cleanup evidence"
                    ),
                    "startup_failure": "infrastructure_unknown",
                    "candidate_completed_before_deadline": "infrastructure_unknown",
                    "post_deadline_results_accepted": False,
                    "termination_and_cleanup_verified": True,
                    "cleanup_required_for_any_outcome": True,
                    }),
                # Always record the exact frozen action set/routes, even when
                # a caller supplies a custom lifecycle policy.
                "enforcement_actions": list(
                    DEADLINE_ENFORCEMENT_ACTIONS
                ),
                "enforcement_routes": list(DEADLINE_ENFORCEMENT_ROUTES),
                "enforcement_route_rule": (
                    "exactly two routes establish a conclusive outer-deadline "
                    "failure and nothing else: (A) the wrapper route, where "
                    "the authenticated wrapper records show execution entry "
                    "received by the host before the deadline, the "
                    "alive-at-deadline-notice observation and the wrapper's "
                    "kill of the identified child, and the host observed the "
                    "docker run process/container exit no later than the "
                    "registered deadline plus 0.25 s, with verified "
                    "termination/cleanup; host enforcement actions are not "
                    "required; (B) the host route, where the container did not "
                    "exit by itself inside the observation sub-window and the "
                    "complete frozen expected action set returned within the "
                    "registered deadline plus 0.25 s with cleanup verified. "
                    "Any other combination is infrastructure/unknown with "
                    "cleanup pending."
                ),
                "wrapper_enforcement_route_requires": (
                    "authenticated entry received by the host before the "
                    "deadline, the alive-at-deadline-notice observation, the "
                    "wrapper's kill of the identified child, the host's own "
                    "observed docker run process/container exit no later than "
                    "deadline + 0.25 s, and verified termination/cleanup"
                ),
                "host_enforcement_route_requires": (
                    "the container did not exit by itself inside the "
                    "observation sub-window; the complete frozen expected "
                    "action set returned inside deadline + 0.25 s and cleanup "
                    "was verified"
                ),
                "enforcement_route_bound_s": float(
                    LIFECYCLE_NOTICE_GRACE_S
                ),
            },
            "retry": dict(
                retry_policy
                or {
                    "max_retries_per_infrastructure_test": 1,
                    "identical_configuration": True,
                    "first_conclusive_controls": True,
                    "preserve_both_attempts": True,
                    "conclusive_pass_or_candidate_failure_retried": False,
                    "retry_still_infrastructure": "terminal_infrastructure_unverified",
                    "requires_documented_recovery": True,
                }
            ),
            "concurrency": {"workers": int(workers)},
            "docker_image": pin,
            "docker": {
                "sandbox": "docker",
                "cpu_only": True,
                "network": "none",
                "read_only_root": True,
                "candidate_mount": "/work:ro",
                "tmpfs": "/tmp:rw,noexec,nosuid,size=64m",
                "memory_mb": 512,
                "memory_swap_mb": 512,
                "pids_limit": 64,
                "cpus": 1.0,
                "user": "65534:65534",
                "security_opt": "no-new-privileges",
                "lifecycle_evidence_mount": "/evidence:rw",
            },
        },
        "interpretation": interpretation_entry,
        "tests": {
            "manifest_file": FREEZE_TEST_MANIFEST_NAME,
            "manifest_sha256": test_manifest_sha,
            "manifest_bytes": len(test_text.encode("utf-8")),
            "problems": len(rows),
            "tests": sum(int(row["n_tests"]) for row in rows),
            "exclusions": exclusions,
            "ordering": "split order (P1, T-fit, selection), then split-file order",
        },
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _hash_payload(payload, "manifest_sha256")
    if exclusive_write:
        _write_json_exclusive(out_dir / FREEZE_MANIFEST_NAME, payload)
    else:
        _write_json(out_dir / FREEZE_MANIFEST_NAME, payload)
    return payload


def load_freeze_manifest(
    path: Path | str, *, verify_files: bool = True
) -> dict[str, Any]:
    """Load and fully verify a freeze manifest (self hash, tests, evidence)."""

    path = Path(path)
    assert_not_void_artifact(path, role="verifier freeze")
    payload = _read_json(path)
    if payload.get("kind") != "pilot_shortfall_verifier_freeze":
        raise ValueError(f"{path}: not a pilot-shortfall verifier freeze")
    expected = _hash_payload(payload, "manifest_sha256")
    if payload.get("manifest_sha256") != expected:
        raise ValueError(f"{path}: manifest self hash does not recompute")
    selected = payload.get("selected_inputs") or {}
    if selected.get("selected_inputs_sha256") != _hash_payload(
        selected, "selected_inputs_sha256"
    ):
        raise ValueError(f"{path}: selected-input hash does not recompute")
    tests = payload.get("tests") or {}
    test_path = path.parent / str(tests.get("manifest_file") or "")
    if not test_path.is_file():
        raise ValueError(f"{path}: test manifest {test_path} is missing")
    if _sha256_file(test_path) != tests.get("manifest_sha256"):
        raise ValueError(f"{path}: test manifest hash does not match")
    if verify_files:
        for entry in (payload.get("inputs") or {}).get("files") or []:
            bound = Path(str(entry.get("path") or ""))
            if not bound.is_file():
                raise ValueError(f"{path}: bound evidence {bound} is missing")
            if _sha256_file(bound) != entry.get("sha256"):
                raise ValueError(f"{path}: bound evidence {bound} changed")
        interpretation = payload.get("interpretation")
        if interpretation:
            manifest = Path(str(interpretation.get("manifest") or ""))
            execution = Path(str(interpretation.get("execution_tests_file") or ""))
            if not manifest.is_file() or _sha256_file(manifest) != interpretation.get(
                "manifest_file_sha256"
            ):
                raise ValueError(f"{path}: interpretation manifest changed")
            if not execution.is_file() or _sha256_file(execution) != interpretation.get(
                "execution_tests_sha256"
            ):
                raise ValueError(f"{path}: interpretation execution tests changed")
    return payload


# ---------------------------------------------------------------------------
# Top-up plan (precommitted before any decision)
# ---------------------------------------------------------------------------
def topup_timeout_seconds(request_count: int) -> int:
    count = int(request_count)
    if count <= 0:
        raise ValueError("request_count must be positive")
    seconds = math.ceil(
        count * TOPUP_MEASURED_SECONDS_PER_REQUEST * TOPUP_TIMEOUT_SAFETY
    )
    return int(max(TOPUP_TIMEOUT_FLOOR_S, seconds))


def _topup_problem_union(
    selected_inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    union: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for name in SPLITS_WITH_TOPUP:
        split = selected_inputs["splits"][name]
        for problem in split["problems"]:
            if problem["kind"] not in {"math", "code"}:
                continue
            problem_id = problem["problem_id"]
            if problem_id not in union:
                union[problem_id] = {
                    "problem_id": problem_id,
                    "kind": problem["kind"],
                    "source": problem.get("source"),
                    "splits": [name],
                }
                order.append(problem_id)
            elif name not in union[problem_id]["splits"]:
                union[problem_id]["splits"].append(name)
    return [union[problem_id] for problem_id in order]


def expected_topup_requests(
    freeze: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the complete request map the loaded freeze requires.

    The expected map is derived only from the freeze's immutable selected input
    sets: every math/code problem in the original T-fit ∪ selection input sets
    gets exactly indices 4, 5, 6 and 7 with the registered request id, split,
    kind and seed.  P1 and chat never appear.
    """

    selected = freeze.get("selected_inputs")
    if not isinstance(selected, Mapping) or not selected.get("splits"):
        raise ValueError("the verifier freeze has no selected input sets")
    requests: list[dict[str, Any]] = []
    for problem in _topup_problem_union(selected):
        problem_id = problem["problem_id"]
        for index in TOPUP_INDICES:
            request_id = f"{problem_id}:{index}"
            requests.append(
                {
                    "request_id": request_id,
                    "problem_id": problem_id,
                    "sample_index": int(index),
                    "seed": modal_lite.lite_request_seed(0, request_id),
                    "kind": problem["kind"],
                    "source": problem.get("source"),
                    "split": problem["splits"][0],
                    "splits": list(problem["splits"]),
                }
            )
    return requests


def build_topup_plan(
    freeze: Mapping[str, Any],
    *,
    parent_run_id: str | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Every potential extra request, exactly four per eligible P2 problem."""

    selected = freeze["selected_inputs"]
    parent = str(parent_run_id or freeze.get("lite_run_id") or LITE_RUN_ID_FALLBACK)
    problems = _topup_problem_union(selected)
    requests = expected_topup_requests(freeze)
    unique_ids = [request["request_id"] for request in requests]
    if len(unique_ids) != len(set(unique_ids)):
        raise ValueError("top-up plan request ids are not unique")
    request_count = len(requests)
    evidence = {
        str(entry.get("label")): entry.get("sha256")
        for entry in (freeze.get("inputs") or {}).get("files") or []
    }
    payload: dict[str, Any] = {
        "kind": "pilot_shortfall_topup_plan",
        "version": 1,
        "created_utc": created_utc or _utc_now(),
        "parent_run_id": parent,
        "parent_pool_manifest_sha256": evidence.get("pool_manifest"),
        "parent_sample_grades_sha256": evidence.get("lite_sample_grades"),
        "parent_request_ids_sha256": evidence.get("request_ids"),
        "request_indices": list(TOPUP_INDICES),
        "request_id_rule": "problem_id + ':' + str(sample_index)",
        "seed_rule": (
            "lite_request_seed(0, request_id) = "
            "int.from_bytes(sha256(UTF8('0:'+request_id)).digest()[:8], "
            "'little', signed=False) % (2**63-1)"
        ),
        "run_cap_usd": float(TOPUP_RUN_CAP_USD),
        "global_cap_usd": float(HARD_GLOBAL_CAP_USD),
        "timeout_seconds": topup_timeout_seconds(request_count),
        "timeout_basis": {
            "measured_seconds_per_request": TOPUP_MEASURED_SECONDS_PER_REQUEST,
            "safety_factor": TOPUP_TIMEOUT_SAFETY,
            "floor_seconds": TOPUP_TIMEOUT_FLOOR_S,
            "probe": "2026-11-05 A100-80GB: 2,000 requests / 1,802 s",
        },
        "sampling": dict(SAMPLING),
        "backend": {
            "engine": LITE_ENGINE,
            "engine_version": LITE_ENGINE_VERSION,
            "model_id": MODEL_ID,
            "model_revision": freeze.get("model_revision"),
            "dtype": "bfloat16",
            "gpu": "A100-80GB",
            "call_shape": "one vLLM generate pass; whole plan in one remote call",
        },
        "problem_count": len(problems),
        "requests_by_split": {
            name: sum(1 for item in requests if item["split"] == name)
            for name in SPLITS_WITH_TOPUP
        },
        "problems_by_split": {
            name: sum(1 for problem in problems if name in problem["splits"])
            for name in SPLITS_WITH_TOPUP
        },
        "requests_by_kind": {
            kind: sum(1 for item in requests if item["kind"] == kind)
            for kind in ("math", "code")
        },
        "request_count": request_count,
        "requests": requests,
        "plan_sha256": "",
    }
    # The extension run id is derived from the immutable plan content, never
    # from a decision or a result.
    stable = {
        "parent_run_id": parent,
        "request_indices": list(TOPUP_INDICES),
        "requests": requests,
        "sampling": SAMPLING,
        "backend": payload["backend"],
    }
    payload["topup_run_id"] = _sha256_text(
        json.dumps(stable, ensure_ascii=False, sort_keys=True)
    )[:24]
    payload["plan_sha256"] = _hash_payload(payload, "plan_sha256")
    return payload


def plan_topup(
    freeze_path: Path | str,
    out_path: Path | str,
    *,
    parent_run_id: str | None = None,
    authorization: V3RunAuthorization | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    freeze = load_freeze_manifest(freeze_path)
    if authorization is not None:
        _authorize_v3_stage(
            authorization,
            stage="decision",
            out_dir=out_path,
            lite_dir=(freeze.get("inputs") or {}).get("lite_dir") or "",
        )
    else:
        registry_path = v3_registry_path(
            str((freeze.get("inputs") or {}).get("lite_dir") or "")
        )
        if registry_path.exists():
            raise ShortfallV3RunError(
                "the top-up decision may only run as part of the claimed "
                "official v3 run or an authorized resume"
            )
    payload = build_topup_plan(freeze, parent_run_id=parent_run_id)
    _write_json(Path(out_path), payload)
    return payload


def _topup_request_identity(request: Mapping[str, Any]) -> tuple[Any, ...]:
    splits = request.get("splits")
    if splits is None:
        splits = [request.get("split")]
    return (
        str(request.get("request_id") or ""),
        str(request.get("problem_id") or ""),
        int(request.get("sample_index") or -1),
        int(request.get("seed") or -1),
        str(request.get("kind") or ""),
        str(request.get("split") or ""),
        tuple(sorted(str(value) for value in (splits or []))),
    )


def _require_frozen_plan_map(
    requests: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]
) -> None:
    """Require exact equality with the request map reconstructed from the freeze."""

    expected = {
        _topup_request_identity(request): request
        for request in expected_topup_requests(freeze)
    }
    actual = {_topup_request_identity(request): request for request in requests}
    if len(actual) != len(requests):
        raise ValueError("top-up plan has duplicate request identities")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "top-up plan does not match the request map frozen in the verifier "
            f"freeze (missing={len(missing)}, extra={len(extra)}; "
            f"first missing={missing[:1]}, first extra={extra[:1]})"
        )


def validate_topup_plan(
    plan: Mapping[str, Any], *, freeze: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if plan.get("kind") != "pilot_shortfall_topup_plan":
        raise ValueError("not a pilot-shortfall top-up plan")
    if tuple(int(value) for value in plan.get("request_indices") or ()) != TOPUP_INDICES:
        raise ValueError(f"top-up indices must be {TOPUP_INDICES}")
    if plan.get("plan_sha256") != _hash_payload(plan, "plan_sha256"):
        raise ValueError("top-up plan self hash does not recompute")
    requests = list(plan.get("requests") or [])
    if int(plan.get("request_count") or -1) != len(requests):
        raise ValueError("top-up request_count does not match requests")
    seen: set[str] = set()
    problems: set[str] = set()
    indices_by_problem: dict[str, list[int]] = {}
    for request in requests:
        request_id = str(request.get("request_id") or "")
        problem_id = str(request.get("problem_id") or "")
        index = int(request.get("sample_index") or -1)
        if index not in TOPUP_INDICES:
            raise ValueError(f"{request_id}: index {index} is not a top-up index")
        if request_id != f"{problem_id}:{index}":
            raise ValueError(f"{request_id}: request id does not match problem/index")
        if request_id in seen:
            raise ValueError(f"{request_id}: duplicate top-up request")
        seen.add(request_id)
        if int(request.get("seed")) != modal_lite.lite_request_seed(0, request_id):
            raise ValueError(f"{request_id}: seed does not match the registered rule")
        if str(request.get("kind")) not in {"math", "code"}:
            raise ValueError(f"{request_id}: only math/code top-ups are registered")
        if "pilot_p1" in (request.get("splits") or [request.get("split")]):
            raise ValueError(f"{request_id}: P1 top-ups are forbidden")
        problems.add(problem_id)
        indices_by_problem.setdefault(problem_id, []).append(index)
    for problem_id, indices in indices_by_problem.items():
        if sorted(indices) != list(TOPUP_INDICES):
            raise ValueError(
                f"{problem_id}: top-up indices must be exactly {TOPUP_INDICES}, "
                f"got {sorted(indices)}"
            )
    if int(plan.get("problem_count") or -1) != len(problems):
        raise ValueError("top-up problem_count does not match distinct problems")
    if len(requests) != len(problems) * len(TOPUP_INDICES):
        raise ValueError("top-up plan is not exactly four requests per problem")
    if float(plan.get("run_cap_usd") or 0) != TOPUP_RUN_CAP_USD:
        raise ValueError(f"top-up run cap must be ${TOPUP_RUN_CAP_USD:g}")
    if int(plan.get("timeout_seconds") or 0) != topup_timeout_seconds(len(requests)):
        raise ValueError("top-up timeout is not sized for the request count")
    if freeze is not None:
        _require_frozen_plan_map(requests, freeze)
    return dict(plan)


def verify_count_implementation(
    freeze: Mapping[str, Any],
    *,
    module_files: Mapping[str, Path | str] | None = None,
    repo_root: Path | str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse when the *actual executing* implementation differs from the freeze.

    The freeze records the resolved module files, entry points, full git
    provenance, interpreter and dependency versions that produced the
    registered count.  A second checkout, an edited ``shortfall.py`` or any
    other changed executing file is refused before grading.
    """

    verifier = freeze.get("verifier") or {}
    recorded_exec = verifier.get("executing")
    if not isinstance(recorded_exec, Mapping) or not recorded_exec.get("modules"):
        raise ValueError(
            "verifier freeze does not bind the actual executing implementation"
        )
    current = executing_implementation(
        repo_root=repo_root, module_files=module_files, provenance=provenance
    )
    recorded_modules = {
        str(item["module"]): item for item in recorded_exec.get("modules") or []
    }
    current_modules = {str(item["module"]): item for item in current["modules"]}
    if set(recorded_modules) != set(current_modules):
        missing = sorted(set(recorded_modules) - set(current_modules))
        extra = sorted(set(current_modules) - set(recorded_modules))
        raise ValueError(
            "executing verifier module set differs from the freeze "
            f"(missing={missing}, extra={extra})"
        )
    for name in sorted(recorded_modules):
        recorded = recorded_modules[name]
        actual = current_modules[name]
        if actual["path"] != recorded.get("path"):
            raise ValueError(
                "executing verifier implementation path differs from the freeze: "
                f"{name} ({actual['path']} != {recorded.get('path')})"
            )
        if actual["sha256"] != recorded.get("sha256"):
            raise ValueError(
                "executing verifier implementation changed since the freeze: "
                f"{name}"
            )
    recorded_entries = {
        str(item["label"]): item
        for item in recorded_exec.get("entry_points") or []
    }
    current_entries = {
        str(item["label"]): item for item in current.get("entry_points") or []
    }
    for label, recorded in sorted(recorded_entries.items()):
        actual = current_entries.get(label)
        if actual is None:
            raise ValueError(
                f"executing entry point is missing since the freeze: {label}"
            )
        if actual["path"] != recorded.get("path") or actual["sha256"] != recorded.get(
            "sha256"
        ):
            raise ValueError(
                f"executing entry point changed since the freeze: {label}"
            )
    recorded_git = recorded_exec.get("git") or {}
    current_git = current.get("git") or {}
    if recorded_git.get("commit") != current_git.get("commit"):
        raise ValueError(
            "executing source commit differs from the freeze "
            f"({current_git.get('commit')!r} != {recorded_git.get('commit')!r})"
        )
    recorded_interpreter = recorded_exec.get("interpreter") or {}
    current_interpreter = current.get("interpreter") or {}
    if (
        recorded_interpreter.get("executable")
        != current_interpreter.get("executable")
        or recorded_interpreter.get("version") != current_interpreter.get("version")
    ):
        raise ValueError("executing interpreter differs from the freeze")
    recorded = dict(verifier.get("code_hashes") or {})
    if recorded:
        current_hashes = current.get("code_hashes") or {}
        for relative, expected in sorted(recorded.items()):
            if current_hashes.get(relative) != expected:
                raise ValueError(
                    "executing verifier implementation changed since the freeze: "
                    f"{relative}"
                )
    recorded_deps = dict(verifier.get("dependencies") or {})
    current_deps = current.get("dependencies") or {}
    for key in DEPENDENCY_FINGERPRINT_KEYS:
        if key not in recorded_deps:
            raise ValueError(f"verifier freeze has no recorded dependency {key!r}")
        if current_deps.get(key) != recorded_deps[key]:
            raise ValueError(
                "executing dependency version changed since the freeze: "
                f"{key}={current_deps.get(key)!r}, frozen {recorded_deps[key]!r}"
            )
    return {
        "repo_root": str(current["repo_root"]),
        "modules": current["modules"],
        "entry_points": current["entry_points"],
        "code_hashes": current.get("code_hashes") or {},
        "git": current.get("git") or {},
        "interpreter": current.get("interpreter") or {},
        "dependencies": {
            key: current_deps.get(key) for key in DEPENDENCY_FINGERPRINT_KEYS
        },
    }


_COUNT_ACCOUNTING_KEYS = (
    "requests",
    "kept",
    "code_requests",
    "code_requests_kept",
    "code_requests_graded",
    "code_requests_passed",
    "errors",
    "infrastructure_failures",
    "infrastructure_tests",
    "timeouts",
    "answer_failures",
    "per_test_rows",
    "interpretation_excluded_requests",
    "interpretation_excluded_problems",
)

_PIPELINE_RUN_VERSIONS = frozenset({"v3", "v3r"})
OPERATIVE_RUN_VERSION = "v3r"


def _verify_count_attempt_ledger(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    manifest_path: Path,
    summary_path: Path,
    freeze: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Refuse a completed count unless its attempt ledger is fully bound.

    The attempt ledger is the durable record of every verifier dispatch.  It is
    never optional for an official count: the manifest must bind it, its bytes
    must match the bound hash, the summary must reference the same path and
    hash, and its header must name the same run (freeze identity, executing
    implementation and, when known, the run registry/token digest).  A count
    whose ledger contains a verifier interface/programming error is blocked
    run-wide even when every other artifact claims completion.
    """

    files = manifest.get("files") or {}
    ledger_entry = files.get(ATTEMPT_LEDGER_NAME)
    if not isinstance(ledger_entry, Mapping):
        raise ValueError(
            f"{manifest_path}: the attempt ledger ({ATTEMPT_LEDGER_NAME}) is not "
            "bound; an official v3/v3r count must bind its durable attempt ledger"
        )
    ledger_path = Path(str(ledger_entry.get("path") or ""))
    if not ledger_path.is_file():
        raise ValueError(
            f"{manifest_path}: bound attempt ledger is missing: {ledger_path}"
        )
    ledger_sha = _sha256_file(ledger_path)
    if ledger_sha != ledger_entry.get("sha256"):
        raise ValueError(
            f"{manifest_path}: bound attempt ledger changed after the manifest"
        )

    summary_ledger = summary.get("attempt_ledger")
    if not isinstance(summary_ledger, Mapping):
        raise ValueError(
            f"{summary_path}: the summary does not reference the bound attempt ledger"
        )
    summary_ledger_path = Path(str(summary_ledger.get("path") or ""))
    if summary_ledger_path.resolve() != ledger_path.resolve():
        raise ValueError(
            f"{summary_path}: summary and manifest disagree on the attempt ledger path"
        )
    if summary_ledger.get("sha256") != ledger_entry.get("sha256"):
        raise ValueError(
            f"{summary_path}: summary and manifest disagree on the attempt ledger hash"
        )
    artifact_entry = (summary.get("artifacts") or {}).get(ATTEMPT_LEDGER_NAME)
    if (
        not isinstance(artifact_entry, Mapping)
        or artifact_entry.get("sha256") != ledger_entry.get("sha256")
    ):
        raise ValueError(
            f"{summary_path}: summary artifacts do not bind the manifest attempt "
            "ledger hash"
        )

    records = _read_attempt_ledger(ledger_path)
    headers = [record for record in records if record.get("kind") == LEDGER_HEADER]
    if len(headers) != 1:
        raise ValueError(
            f"{ledger_path}: expected exactly one attempt ledger header, found "
            f"{len(headers)}"
        )
    header = headers[0]
    if header.get("ledger_kind") != LEDGER_KIND:
        raise ValueError(f"{ledger_path}: not a shortfall attempt ledger")
    if header.get("header_sha256") != _hash_payload(header, "header_sha256"):
        raise ValueError(f"{ledger_path}: attempt ledger header hash does not recompute")

    if summary.get("freeze_manifest_sha256") != manifest.get(
        "freeze_manifest_sha256"
    ):
        raise ValueError(
            f"{manifest_path}: count manifest and summary disagree on the freeze"
        )
    claimed = {
        "freeze_manifest_sha256": summary.get("freeze_manifest_sha256"),
        "freeze_test_manifest_sha256": summary.get("freeze_test_manifest_sha256"),
        "selected_inputs_sha256": summary.get("selected_inputs_sha256"),
    }
    implementation = summary.get("implementation")
    if isinstance(implementation, Mapping):
        claimed["executing_implementation_sha256"] = _sha256_text(
            json.dumps(dict(implementation), ensure_ascii=False, sort_keys=True)
        )
    for key, expected in claimed.items():
        if expected is None:
            continue
        if header.get(key) != expected:
            raise ValueError(
                f"{ledger_path}: attempt ledger header {key!r} does not agree with "
                "the claimed run"
            )
    run_registry = summary.get("run_registry")
    if isinstance(run_registry, Mapping):
        for key in ("run_token_sha256", "claim_registry_sha256"):
            expected = run_registry.get(key)
            if expected and header.get(key) != expected:
                raise ValueError(
                    f"{ledger_path}: attempt ledger header {key!r} does not agree "
                    "with the claimed run"
                )
        legacy = run_registry.get("registry_sha256")
        if (
            not run_registry.get("claim_registry_sha256")
            and legacy
            and header.get("registry_sha256") != legacy
        ):
            raise ValueError(
                f"{ledger_path}: attempt ledger header 'registry_sha256' does not "
                "agree with the claimed run"
            )

    if freeze is not None:
        identity = _ledger_identity(
            freeze,
            implementation if isinstance(implementation, Mapping) else {},
            timeout_s=float((freeze.get("verifier") or {}).get("timeout_s", 0.0)),
            retry_policy=(freeze.get("verifier") or {}).get("retry"),
        )
        for key, expected in identity.items():
            if header.get(key) != expected:
                raise ValueError(
                    f"{ledger_path}: attempt ledger header {key!r} does not agree "
                    "with the claimed run"
                )
        run_version = str(
            manifest.get("run_version") or summary.get("run_version") or ""
        )
        if run_version in _PIPELINE_RUN_VERSIONS:
            spec = _run_spec(run_version)
            lite_dir_value = (freeze.get("inputs") or {}).get("lite_dir")
            if lite_dir_value:
                registry_path = (
                    v3r_registry_path(lite_dir_value)
                    if spec.version == "v3r"
                    else v3_registry_path(lite_dir_value)
                )
                if registry_path.is_file():
                    try:
                        registry = _load_run_registry(registry_path, spec)
                    except ShortfallV3RunError as exc:
                        raise ValueError(
                            f"{registry_path}: the claimed run registry is invalid: "
                            f"{exc}"
                        ) from exc
                    if Path(str(registry.get("lite_dir") or "")).resolve() != Path(
                        str(lite_dir_value)
                    ).resolve():
                        raise ValueError(
                            f"{registry_path}: the claimed run registry targets a "
                            "different lite directory than the freeze"
                        )
                    expected_count_dir = (
                        Path(str(registry.get("out_root") or "")).resolve()
                        / spec.stage_directories["count"]
                    )
                    if Path(summary_path).parent.resolve() != expected_count_dir:
                        raise ValueError(
                            f"{summary_path}: the count is not in the directory "
                            f"recorded by the claimed run ({expected_count_dir})"
                        )
                    if isinstance(run_registry, Mapping):
                        # The registry file hash advances as stages complete;
                        # only the immutable run identity and append-only stage
                        # history are compared here.  The count-start registry
                        # digest in the header is audit-only.
                        try:
                            _verify_registry_is_same_run(
                                header, registry, stage_names=spec.stages
                            )
                        except ShortfallV3RunError as exc:
                            raise ValueError(
                                f"{registry_path}: {exc}"
                            ) from exc
                        expected_claim = str(
                            run_registry.get("claim_registry_sha256") or ""
                        )
                        if expected_claim and _registry_claim_digest(
                            registry
                        ) != expected_claim:
                            raise ValueError(
                                f"{registry_path}: the run claim digest does not "
                                "agree with the count summary"
                            )

    errors = _ledger_verifier_interface_error_records(records)
    if errors:
        raise ValueError(
            f"{summary_path}: {_verifier_interface_error_detail(errors)}; a count "
            "whose attempt ledger contains a verifier interface/programming error "
            "is blocked run-wide and may never be accepted"
        )
    return records


def load_verified_count(
    count_path: Path | str,
    *,
    topup_plan_path: Path | str | None = None,
    freeze_path: Path | str | None = None,
    require_extension: bool = False,
) -> dict[str, Any]:
    """Load a count only if it carries complete, explicit accounting.

    Legacy counts that omit the status, expected-request coverage or per-test
    execution/infrastructure accounting are refused; missing accounting is
    never defaulted to zero.
    """

    count_path = Path(count_path)
    assert_not_void_artifact(count_path, role="official count")
    summary_path = (
        count_path / COUNT_SUMMARY_NAME if count_path.is_dir() else count_path
    )
    summary_path = summary_path.resolve()
    count_dir = summary_path.parent
    manifest_path = count_dir / COUNT_MANIFEST_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"official count summary is missing: {summary_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"official count manifest is missing: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("kind") != "pilot_shortfall_count_manifest":
        raise ValueError(f"{manifest_path}: not a pilot-shortfall count manifest")
    if manifest.get("manifest_sha256") != _hash_payload(manifest, "manifest_sha256"):
        raise ValueError(f"{manifest_path}: manifest self hash does not recompute")
    manifest_status = manifest.get("status")
    if manifest_status not in {
        COUNT_STATUS_COMPLETE,
        COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE,
    }:
        raise ValueError(
            f"{manifest_path}: count manifest status is {manifest_status!r}, "
            "not a completed count"
        )
    files = manifest.get("files") or {}
    if COUNT_SUMMARY_NAME not in files:
        raise ValueError(f"{manifest_path}: summary.json is not bound")
    for name, entry in files.items():
        bound = Path(str(entry.get("path") or ""))
        if not bound.is_file():
            raise ValueError(f"{manifest_path}: bound count artifact {name} is missing")
        if _sha256_file(bound) != entry.get("sha256"):
            raise ValueError(f"{manifest_path}: bound count artifact {name} changed")
    if _sha256_file(summary_path) != (files[COUNT_SUMMARY_NAME].get("sha256")):
        raise ValueError(f"{manifest_path}: summary.json changed after the manifest")

    summary = _read_json(summary_path)
    # Every completed official count must bind its durable attempt ledger; the
    # binding is never optional.  This refuses the bypass where omitting the
    # ledger entry let loading accept a count whose ledger holds a verifier
    # interface/programming error.  The check also verifies that the summary
    # references the same ledger bytes and that the ledger header names the
    # claimed run.
    freeze_for_ledger = (
        load_freeze_manifest(freeze_path) if freeze_path is not None else None
    )
    _verify_count_attempt_ledger(
        manifest=manifest,
        summary=summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
        freeze=freeze_for_ledger,
    )
    if summary.get("kind") != "pilot_shortfall_official_count":
        raise ValueError(f"{summary_path}: not a pilot-shortfall official count")
    summary_status = summary.get("status")
    if summary_status not in {
        COUNT_STATUS_COMPLETE,
        COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE,
    }:
        raise ValueError(
            f"{summary_path}: count status is {summary_status!r}, not a "
            "completed count"
        )
    if manifest_status != summary_status:
        raise ValueError(
            f"{manifest_path}: manifest and summary disagree on count status"
        )
    for key in (
        "infrastructure_failures",
        "accounting",
        "expected_requests",
        "per_split",
        "extension_required",
        "extension_decision",
    ):
        if key not in summary:
            raise ValueError(
                f"{summary_path}: legacy count is missing explicit {key!r} accounting"
            )
    accounting = summary.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError(f"{summary_path}: count accounting is not an object")
    missing_accounting = [key for key in _COUNT_ACCOUNTING_KEYS if key not in accounting]
    if missing_accounting:
        raise ValueError(
            f"{summary_path}: count accounting is missing {missing_accounting}"
        )
    infrastructure = int(summary["infrastructure_failures"])
    if infrastructure != int(accounting["infrastructure_failures"]):
        raise ValueError(
            f"{summary_path}: infrastructure accounting disagrees with the summary"
        )
    # Terminal infrastructure accounting is recomputed from per-test rows
    # below; a bare summary number is never trusted.
    if manifest.get("freeze_manifest_sha256") != summary.get("freeze_manifest_sha256"):
        raise ValueError(
            f"{manifest_path}: count manifest and summary disagree on the freeze"
        )
    expected = summary.get("expected_requests")
    if not isinstance(expected, Mapping) or "count" not in expected:
        raise ValueError(f"{summary_path}: expected-request coverage is missing")
    per_request_entry = files.get(COUNT_PER_REQUEST_NAME)
    if per_request_entry is None:
        raise ValueError(f"{manifest_path}: per_request.jsonl is not bound")
    per_request_path = Path(str(per_request_entry.get("path") or ""))
    rows = local.read_jsonl(per_request_path)
    expected_count = int(expected["count"])
    if int(accounting["requests"]) != expected_count or len(rows) != expected_count:
        raise ValueError(
            f"{summary_path}: expected {expected_count} requests, accounting says "
            f"{accounting['requests']} and per_request has {len(rows)}"
        )
    request_ids = [str(row.get("request_id") or "") for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{per_request_path}: duplicate request ids")
    code_rows = [row for row in rows if str(row.get("domain") or "") == "code"]
    expected_code = expected.get("code_request_count")
    if expected_code is not None and int(expected_code) != len(code_rows):
        raise ValueError(
            f"{summary_path}: expected code-request coverage does not recompute"
        )
    if int(accounting["code_requests"]) != len(code_rows):
        raise ValueError(
            f"{summary_path}: code-request accounting does not recompute"
        )
    if expected.get("request_ids_sha256") != _sha256_text(
        "\n".join(sorted(request_ids))
    ):
        raise ValueError(
            f"{summary_path}: per-request ids do not match the expected coverage"
        )
    freeze: Mapping[str, Any] | None = freeze_for_ledger
    if freeze_path is not None:
        if freeze is None:
            freeze = load_freeze_manifest(freeze_path)
        if summary.get("freeze_manifest_sha256") != freeze.get("manifest_sha256"):
            raise ValueError(
                f"{summary_path}: count was not made against this verifier freeze"
            )
        expected_ids = [
            str(request["request_id"])
            for name in PILOT_SPLITS
            for request in freeze["selected_inputs"]["splits"][name]["requests"]
        ]
        if sorted(request_ids) != sorted(expected_ids):
            raise ValueError(
                f"{summary_path}: per-request coverage does not equal the frozen "
                "selected request set"
            )
    freeze_inventory: dict[str, dict[str, Any]] = {}
    freeze_test_bytes: dict[str, list[str]] = {}
    if freeze_path is not None:
        freeze_tests_path = Path(freeze_path).parent / load_freeze_manifest(
            freeze_path
        )["tests"]["manifest_file"]
        freeze_test_rows = [
            json.loads(line)
            for line in freeze_tests_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        freeze_inventory = _test_inventory_from_rows(freeze_test_rows)
        freeze_test_bytes = {
            str(row.get("problem_id") or ""): [
                str(value) for value in row.get("test_bytes") or []
            ]
            for row in freeze_test_rows
        }
        # When the freeze binds an authenticated interpretation, the loader
        # re-verifies that execution manifest and recomputes output comparison
        # against the execution text actually graded, not the raw stored
        # representation.
        interpretation_entry = freeze.get("interpretation") or {}
        execution_file = str(
            interpretation_entry.get("execution_tests_file") or ""
        )
        if execution_file:
            execution_path = Path(execution_file)
            expected_exec_sha = str(
                interpretation_entry.get("execution_tests_sha256") or ""
            )
            if not execution_path.is_file():
                raise ValueError(
                    f"{freeze_path}: bound execution tests file is missing"
                )
            if _sha256_file(execution_path) != expected_exec_sha:
                raise ValueError(
                    f"{freeze_path}: bound execution tests file changed"
                )
            execution_by_problem: dict[str, list[str]] = {}
            for line in execution_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                execution_row = json.loads(line)
                problem_id = str(execution_row.get("problem_id") or "")
                execution_by_problem[problem_id] = [
                    json.dumps(test, ensure_ascii=False, sort_keys=True)
                    for test in execution_row.get("tests") or []
                ]
            freeze_test_bytes = execution_by_problem
    # Explicit per-test execution/infrastructure accounting, recomputed.
    per_test_rows = 0
    infrastructure_tests = 0
    infrastructure_requests = 0
    blocking_infrastructure_requests = 0
    terminal_infrastructure_requests = 0
    terminal_infrastructure_tests = 0
    interpretation_excluded_requests = 0
    interpretation_excluded_problem_ids: set[str] = set()
    for row in rows:
        if row.get("failure_class") is None:
            raise ValueError(
                f"{per_request_path}: {row.get('request_id')} has no explicit "
                "failure_class"
            )
        per_test = list(row.get("per_test") or [])
        per_test_rows += len(per_test)
        infrastructure_tests += sum(
            1 for test in per_test if test.get("infrastructure_failure")
        )
        terminal_tests = [
            test
            for test in per_test
            if test.get("terminal_infrastructure")
        ]
        terminal_infrastructure_tests += len(terminal_tests)
        for test in per_test:
            if not test.get("infrastructure_failure"):
                continue
            if test.get("terminal_infrastructure"):
                if len(test.get("attempts") or []) != 2:
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} terminal infrastructure does not "
                        "have exactly the first attempt and its single retry"
                    )
                last = (test.get("attempts") or [])[-1]
                if last.get("status") != "infrastructure_error":
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} terminal infrastructure is not a "
                        "second infrastructure/unknown attempt"
                    )
                if test.get("terminal_status") != TEST_STATUS_TERMINAL_INFRASTRUCTURE:
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} has no terminal infrastructure "
                        "status"
                    )
                if not test.get("terminal_reason"):
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} terminal infrastructure has no "
                        "retained reason"
                    )
                if test.get("cleanup_verified") is not True:
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} terminal infrastructure has "
                        "unverified cleanup; it remains blocking"
                    )
                if (
                    test.get("timed_out")
                    and test.get("termination_verified") is not True
                ):
                    raise ValueError(
                        f"{per_request_path}: {row.get('request_id')} test "
                        f"{test.get('index')} terminal timeout has unverified "
                        "termination; it remains blocking"
                    )
        if row.get("infrastructure_failure"):
            infrastructure_requests += 1
            infra_tests = [
                test
                for test in per_test
                if test.get("infrastructure_failure")
            ]
            blocking_tests = [
                test
                for test in infra_tests
                if not test.get("terminal_infrastructure")
            ]
            if (
                blocking_tests
                or row.get("coverage_error")
                or not per_test
            ):
                blocking_infrastructure_requests += 1
            elif terminal_tests:
                terminal_infrastructure_requests += 1
            else:
                raise ValueError(
                    f"{per_request_path}: {row.get('request_id')} has "
                    "infrastructure failure without terminal or blocker status"
                )
        if row.get("failure_class") == "interpretation_excluded":
            interpretation_excluded_requests += 1
            interpretation_excluded_problem_ids.add(str(row.get("problem_id") or ""))
        if str(row.get("domain") or "") != "code":
            continue
        if not row.get("kept") or row.get("failure_class") in {
            "filtered",
            "interpretation_excluded",
            "no_frozen_tests",
            "not_applicable",
        }:
            continue
        inventory = row.get("test_inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError(
                f"{per_request_path}: {row.get('request_id')} has no frozen "
                "test inventory"
            )
        if freeze_inventory:
            expected = freeze_inventory.get(str(row.get("problem_id") or ""))
            if expected is None:
                raise ValueError(
                    f"{per_request_path}: {row.get('request_id')} problem is not "
                    "in the frozen test manifest"
                )
            if dict(inventory) != dict(expected):
                raise ValueError(
                    f"{per_request_path}: {row.get('request_id')} test inventory "
                    "does not equal the frozen inventory"
                )
        else:
            expected = dict(inventory)
        requires_tests = (
            row.get("failure_class") in _GRADED_CODE_FAILURE_CLASSES
            and (int(row.get("tests_run") or 0) > 0 or bool(per_test))
        )
        if not requires_tests and not per_test:
            continue
        reason = _validate_per_test_coverage(
            per_test,
            expected_n=int(expected.get("n_tests") or 0),
            expected_sha=str(expected.get("tests_sha256") or ""),
            inventory=inventory,
            test_bytes=(
                freeze_test_bytes.get(str(row.get("problem_id") or ""))
                if freeze_path is not None
                else None
            ),
            request_id=str(row.get("request_id") or ""),
            enforcement_bound_s=_frozen_deadline_enforcement_bound(freeze),
        )
        if reason:
            raise ValueError(
                f"{per_request_path}: {row.get('request_id')} incomplete test "
                f"coverage: {reason}"
            )
        derived_verified = bool(per_test) and not any(
            test.get("infrastructure_failure") for test in per_test
        ) and all(test.get("passed") for test in per_test)
        if bool(row.get("verified")) != derived_verified:
            raise ValueError(
                f"{per_request_path}: {row.get('request_id')} verified flag does "
                "not derive from its per-test rows"
            )
        if int(row.get("tests_run") or 0) != int(expected.get("n_tests") or 0):
            raise ValueError(
                f"{per_request_path}: {row.get('request_id')} tests_run does not "
                "equal the frozen test count"
            )
    if int(accounting["per_test_rows"]) != per_test_rows:
        raise ValueError(
            f"{summary_path}: per-test execution accounting does not recompute"
        )
    if int(accounting["infrastructure_tests"]) != infrastructure_tests:
        raise ValueError(
            f"{summary_path}: infrastructure test accounting does not recompute"
        )
    if int(accounting["infrastructure_failures"]) != infrastructure_requests:
        raise ValueError(
            f"{summary_path}: infrastructure request accounting does not recompute"
        )
    if infrastructure_requests != (
        blocking_infrastructure_requests + terminal_infrastructure_requests
    ):
        raise ValueError(
            f"{summary_path}: infrastructure requests are neither terminal nor "
            "blocking"
        )
    if summary_status == COUNT_STATUS_COMPLETE and infrastructure_requests:
        raise ValueError(
            f"{summary_path}: a complete count has infrastructure failures"
        )
    if summary_status == COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE:
        if not terminal_infrastructure_requests or not terminal_infrastructure_tests:
            raise ValueError(
                f"{summary_path}: terminal-infrastructure count has no terminal "
                "accounting"
            )
        if blocking_infrastructure_requests:
            raise ValueError(
                f"{summary_path}: terminal-infrastructure count still has "
                "blocking infrastructure"
            )
        for key, expected in (
            ("blocking_infrastructure_failures", blocking_infrastructure_requests),
            ("terminal_infrastructure_failures", terminal_infrastructure_requests),
            ("terminal_infrastructure_tests", terminal_infrastructure_tests),
        ):
            if key not in accounting:
                raise ValueError(
                    f"{summary_path}: terminal-infrastructure accounting is "
                    f"missing {key!r}"
                )
            if int(accounting[key]) != expected:
                raise ValueError(
                    f"{summary_path}: {key} accounting does not recompute"
                )
        terminal = summary.get("terminal_infrastructure")
        if not isinstance(terminal, Mapping):
            raise ValueError(
                f"{summary_path}: terminal infrastructure identities are missing"
            )
        if int(terminal.get("count") or 0) != terminal_infrastructure_tests:
            raise ValueError(
                f"{summary_path}: terminal infrastructure identity count does "
                "not recompute"
            )
        terminal_ids = {
            (
                str(item.get("request_id")),
                int(item.get("test_index", -1)),
            )
            for item in terminal.get("tests") or []
            if isinstance(item, Mapping)
        }
        expected_ids = {
            (str(row.get("request_id")), int(test.get("index", -1)))
            for row in rows
            for test in row.get("per_test") or []
            if test.get("terminal_infrastructure")
        }
        if terminal_ids != expected_ids:
            raise ValueError(
                f"{summary_path}: terminal infrastructure identities do not "
                "recompute"
            )
    if int(accounting["interpretation_excluded_requests"]) != (
        interpretation_excluded_requests
    ):
        raise ValueError(
            f"{summary_path}: interpretation-exclusion accounting does not recompute"
        )
    if int(accounting["interpretation_excluded_problems"]) != len(
        interpretation_excluded_problem_ids
    ):
        raise ValueError(
            f"{summary_path}: excluded-problem accounting does not recompute"
        )
    if topup_plan_path is not None:
        plan_path = Path(topup_plan_path).resolve()
        plan = validate_topup_plan(_read_json(plan_path))
        recorded = summary.get("topup_plan") or {}
        if Path(str(recorded.get("path") or "")).resolve() != plan_path:
            raise ValueError(
                f"{summary_path}: count was not made against this top-up plan"
            )
        if recorded.get("sha256") != _sha256_file(plan_path):
            raise ValueError(
                f"{summary_path}: top-up plan file hash does not match the count"
            )
        if recorded.get("plan_sha256") != plan.get("plan_sha256"):
            raise ValueError(
                f"{summary_path}: top-up plan content hash does not match the count"
            )
    if require_extension and not bool(summary.get("extension_required")):
        raise ValueError(
            f"{summary_path}: extension_required is false "
            f"({summary.get('extension_decision')!r}); the top-up branch may not "
            "launch"
        )
    return summary


# ---------------------------------------------------------------------------
# Official count
# ---------------------------------------------------------------------------
def decide_extension(n_t0: int, n_s0: int) -> bool:
    """True iff the single B extension is required (fail-closed)."""

    return not (int(n_t0) >= TFIT_MIN and int(n_s0) >= SELECTION_MIN)


def _load_records_by_request(lite_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(lite_dir) / "shards").glob("shard_*.jsonl")):
        for row in local.read_jsonl(path):
            request_id = str(row.get("request_id") or "")
            if not request_id:
                continue
            if request_id in records:
                raise ValueError(f"duplicate shard record for {request_id}")
            records[request_id] = row
    return records


def _load_grades_by_request(lite_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(lite_dir) / "sample_grades.jsonl"
    grades: dict[str, dict[str, Any]] = {}
    for row in local.read_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if request_id:
            grades[request_id] = row
    return grades


def _problem_by_id(pool_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["problem_id"]): row
        for row in local.read_jsonl(Path(pool_dir) / "problems.jsonl")
    }


def _grade_code_default(
    code: str,
    tests: list[dict[str, Any]],
    *,
    timeout: float,
    image: str,
    container_name: str | None = None,
    attempt: str | None = None,
) -> Any:
    from gemmaloop.stage0.grader import DockerSandbox, grade_code

    sandbox = DockerSandbox(image=image, expected_digest=image)
    return grade_code(
        code,
        tests,
        timeout=float(timeout),
        sandbox="docker",
        docker=sandbox,
        allow_fallback=False,
        container_names=(
            [container_name] if container_name is not None else None
        ),
        attempts=[attempt] if attempt is not None else None,
    )


_VERIFIER_PROGRAMMING_ERRORS = (
    TypeError,
    AttributeError,
    NameError,
    UnboundLocalError,
    SyntaxError,
)


def _is_verifier_programming_error(exc: BaseException) -> bool:
    """True for adapter/signature/programming defects, never infrastructure.

    A real sandbox failure is transport/daemon evidence (``RuntimeError``,
    ``OSError``, ``subprocess`` errors).  A ``TypeError``/``AttributeError`` at
    the verifier interface means the code could not even be called correctly;
    it must stop the run instead of being charged to a candidate or a retry.
    """

    return isinstance(exc, ShortfallVerifierInterfaceError) or isinstance(
        exc, _VERIFIER_PROGRAMMING_ERRORS
    )


def _call_grade_fn(
    grade_fn: Callable[..., Any],
    code: str,
    tests: list[dict[str, Any]],
    timeout: float,
    **extras: Any,
) -> Any:
    """Call ``grade_fn`` with the per-test container identity, never dropping it.

    A callable that does not accept the identity keywords is an interface
    programming error and blocks the official run; the identities are never
    silently omitted.
    """

    try:
        signature = inspect.signature(grade_fn)
    except (TypeError, ValueError) as exc:
        raise ShortfallVerifierInterfaceError(
            f"cannot inspect the verifier grade function signature: {exc}"
        ) from exc
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return grade_fn(code, tests, timeout, **extras)
    missing = sorted(key for key in extras if key not in parameters)
    if missing:
        raise ShortfallVerifierInterfaceError(
            "the verifier grade function does not accept the production "
            f"identity keyword(s) {missing}; container names and attempt tokens "
            "may never be dropped"
        )
    return grade_fn(code, tests, timeout, **extras)


STDERR_LIMIT = 800
_INFRASTRUCTURE_EXIT_CODES = {125, 126, 127, 137}
_INFRASTRUCTURE_STDERR_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "error response from daemon",
    "permission denied while trying to connect",
    "docker: error",
    "no such image",
    "exec format error",
)


def _truncate_stderr(text: Any) -> str:
    value = str(text or "")
    if len(value) <= STDERR_LIMIT:
        return value
    return value[:STDERR_LIMIT] + (
        f"... [truncated {len(value) - STDERR_LIMIT} chars]"
    )


def _enforcement_attempt_evidence(test: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the evidence mapping that carries the retained lifecycle frames."""

    if test.get("lifecycle_frames_raw"):
        return test
    attempts = list(test.get("attempts") or [])
    for attempt in reversed(attempts):
        if isinstance(attempt, Mapping) and attempt.get("lifecycle_frames_raw"):
            return attempt
    if attempts and isinstance(attempts[-1], Mapping):
        return attempts[-1]
    return test


def _test_enforcement_route(test: Mapping[str, Any]) -> str | None:
    record = test.get("deadline_enforcement")
    if test.get("enforcement_route") is not None:
        return str(test.get("enforcement_route"))
    if isinstance(record, Mapping):
        return _enforcement_route_from_evidence(
            record, _enforcement_attempt_evidence(test)
        )
    return None


def _classify_test(test: Mapping[str, Any]) -> tuple[bool, str]:
    """Return ``(infrastructure_failure, status)`` for one executed test.

    Validation is state-specific.  Verified cleanup is required before any
    outcome -- passes included -- may be accepted.  A normal result needs a
    successful execution handoff, a genuine child exit status, a completed
    lifecycle before the deadline and verified termination.  A timeout is a
    registered outer-deadline failure only with established entry before the
    deadline, a positive trusted non-exit observation at/after the deadline
    and verified eventual termination.  A delayed exit observation, a missing
    exit record, a handoff failure or completion before the deadline followed
    by transport/teardown failure are infrastructure/unknown.
    """

    if test.get("cleanup_verified") is not True:
        return True, "infrastructure_error"
    lifecycle = test.get("lifecycle_class")
    timed_out = bool(test.get("timed_out"))
    if lifecycle in {LIFECYCLE_LOCAL, LIFECYCLE_LOCAL_TIMEOUT}:
        if timed_out:
            return False, "timeout"
        if bool(test.get("passed")):
            return False, "passed"
        return False, "wrong_answer"
    # A Docker outcome is conclusive only with a complete and authentic
    # lifecycle record.  Missing, tampered, out-of-order or startup-only
    # evidence is infrastructure/unknown for passes and failures alike.
    if test.get("lifecycle_valid") is not True:
        return True, "infrastructure_error"
    if timed_out:
        if (
            lifecycle not in CANDIDATE_DEADLINE_CLASSES
            or test.get("handoff_success") is not True
            or test.get("deadline_alive") is not True
            or test.get("termination_verified") is not True
        ):
            return True, "infrastructure_error"
        # The count classifier applies the same two-route rule as the grader
        # and the independent loader.  A timeout row without a conclusive
        # wrapper or host enforcement route is infrastructure/unknown, never a
        # candidate timeout.  Rows that predate deadline enforcement evidence
        # entirely are handled by the loader, which always re-checks the route.
        if "deadline_enforcement" in test:
            attempt = _enforcement_attempt_evidence(test)
            reason = _deadline_enforcement_reason(
                test.get("deadline_enforcement"),
                bound_s=float(LIFECYCLE_NOTICE_GRACE_S),
                required=True,
                attempt=attempt,
            )
            if reason:
                return True, "infrastructure_error"
            if _test_enforcement_route(test) not in DEADLINE_ENFORCEMENT_ROUTES:
                return True, "infrastructure_error"
        return False, "timeout"
    if lifecycle in CANDIDATE_DEADLINE_CLASSES:
        # A completed result cannot have a deadline lifecycle class.
        return True, "infrastructure_error"
    if lifecycle != LIFECYCLE_COMPLETED_BEFORE:
        return True, "infrastructure_error"
    if (
        test.get("handoff_success") is not True
        or test.get("termination_verified") is not True
        or test.get("exit_observed") is not True
        or test.get("exit_code") is None
    ):
        return True, "infrastructure_error"
    if bool(test.get("passed")):
        return False, "passed"
    stderr = str(test.get("stderr") or "").lower()
    exit_code = test.get("exit_code")
    if exit_code is not None and int(exit_code) in _INFRASTRUCTURE_EXIT_CODES:
        return True, "infrastructure_error"
    if any(marker in stderr for marker in _INFRASTRUCTURE_STDERR_MARKERS):
        return True, "infrastructure_error"
    if exit_code not in (None, 0):
        return False, "candidate_error"
    if str(test.get("error") or "") == "wrong answer":
        return False, "wrong_answer"
    return False, "failed"


_GRADED_CODE_FAILURE_CLASSES = frozenset(
    {
        "verified",
        "timeout",
        "candidate_error",
        "answer_failure",
        "infrastructure",
        "coverage_error",
    }
)


def _test_inventory_from_rows(
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for row in test_rows:
        problem_id = str(row.get("problem_id") or "")
        if not problem_id:
            continue
        inventory[problem_id] = {
            "n_tests": int(row.get("n_tests") or 0),
            "tests_sha256": str(row.get("tests_sha256") or ""),
            "split": row.get("split"),
        }
    return inventory


def _frozen_test_at(
    test_bytes: Sequence[str] | None, index: int
) -> dict[str, Any] | None:
    if test_bytes is None or index < 0 or index >= len(test_bytes):
        return None
    try:
        value = json.loads(str(test_bytes[index]))
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _frozen_deadline_enforcement_bound(
    freeze: Mapping[str, Any] | None = None,
) -> float:
    """Return the frozen post-deadline issuance bound (deadline + 0.25 s)."""

    policy = (((freeze or {}).get("verifier") or {}).get("lifecycle") or {})
    value = policy.get("deadline_enforcement_issue_bound_s")
    try:
        bound = float(value)
    except (TypeError, ValueError):
        return float(LIFECYCLE_NOTICE_GRACE_S)
    if bound <= 0:
        return float(LIFECYCLE_NOTICE_GRACE_S)
    # A freeze can never widen the registered post-deadline window.
    return min(bound, float(LIFECYCLE_NOTICE_GRACE_S))


def _enforcement_slot_action(slot: Mapping[str, Any]) -> str | None:
    """Return the frozen action name for an evidence slot."""

    action = slot.get("action")
    if isinstance(action, str) and action.strip():
        return action.strip()
    command = slot.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        parts = [str(part) for part in command]
        if parts[:2] == ["docker", "rm"]:
            return DOCKER_RM_ACTION
        if parts[:2] == ["docker", "kill"]:
            return DOCKER_KILL_ACTION
    primitive = str(slot.get("primitive") or "")
    if primitive == "rm":
        return DOCKER_RM_ACTION
    if primitive == "kill":
        return DOCKER_KILL_ACTION
    return None


def _enforcement_slot_started(slot: Mapping[str, Any]) -> float | None:
    for key in ("launch_started_monotonic", "started_monotonic"):
        if slot.get(key) is not None:
            try:
                return float(slot[key])
            except (TypeError, ValueError):
                return None
    return None


def _enforcement_slot_returned(slot: Mapping[str, Any]) -> float | None:
    for key in ("launch_returned_monotonic", "returned_monotonic"):
        if slot.get(key) is not None:
            try:
                return float(slot[key])
            except (TypeError, ValueError):
                return None
    return None


def _deadline_enforcement_slots(record: Mapping[str, Any]) -> list[Any]:
    """Return the complete recorded action-slot set.

    New grader evidence carries an explicit ``actions`` list whose slots were
    pre-created before any launcher thread started.  Older retained evidence
    is normalised from ``client_kill`` plus ``primitives`` so its action set
    can still be checked exactly.
    """

    actions = record.get("actions")
    if isinstance(actions, list) and actions:
        return [
            dict(entry) if isinstance(entry, Mapping) else entry
            for entry in actions
        ]
    slots: list[Any] = []
    client = record.get("client_kill")
    if isinstance(client, Mapping):
        slot = dict(client)
        slot.setdefault("action", CLIENT_HARD_KILL_ACTION)
        if "status" not in slot:
            slot["status"] = (
                "returned"
                if slot.get("issued") is True
                and slot.get("returned_monotonic") is not None
                else "failed"
            )
        slot.setdefault(
            "returned", slot.get("returned_monotonic") is not None
        )
        slots.append(slot)
    primitives = record.get("primitives")
    if isinstance(primitives, list):
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                slots.append(primitive)
                continue
            slot = dict(primitive)
            if not slot.get("action"):
                slot["action"] = _enforcement_slot_action(primitive)
            if "status" not in slot:
                if (
                    slot.get("issued") is True
                    and slot.get("launch_returned_monotonic") is not None
                ):
                    slot["status"] = "returned"
                elif slot.get("error") is not None:
                    slot["status"] = "failed"
                else:
                    slot["status"] = "scheduled"
            slot.setdefault(
                "returned",
                slot.get("launch_returned_monotonic") is not None,
            )
            slots.append(slot)
    return slots


def _enforcement_route_from_evidence(
    record: Any, attempt: Any = None
) -> str | None:
    """Return the recorded enforcement route, or the legacy host inference."""

    for source in (record, attempt):
        if not isinstance(source, Mapping):
            continue
        value = source.get("enforcement_route")
        if value is None and source is record:
            value = source.get("route")
        if value is not None and str(value).strip():
            return str(value).strip()
    if isinstance(record, Mapping) and record.get("triggered") is True:
        # Retained host-route records predate the explicit route field.
        return ENFORCEMENT_ROUTE_HOST
    return None


def _wrapper_deadline_enforcement_reason(
    record: Any,
    *,
    bound_s: float,
    attempt: Any,
) -> str | None:
    """Re-check the wrapper (route A) evidence for a conclusive timeout.

    Route A does not require any host enforcement action.  It requires the
    authenticated wrapper records (entry received by the host before the
    deadline, the alive-at-deadline-notice observation and the wrapper's kill
    of the identified child), verified termination/cleanup and the host's own
    observation of the docker run process/container exit no later than
    ``deadline + bound_s``.
    """

    if not isinstance(record, Mapping) or not record:
        return "missing wrapper deadline enforcement evidence"
    if record.get("triggered") is True:
        return "wrapper enforcement route claims a host enforcement trigger"
    if bool(record.get("overrun")):
        return "wrapper enforcement route is marked as an overrun"
    try:
        deadline = float(record["deadline_monotonic"])
    except (KeyError, TypeError, ValueError):
        return "wrapper enforcement route deadline is missing"
    bound = record.get("bound_monotonic")
    if bound is not None:
        try:
            if abs((deadline + float(bound_s)) - float(bound)) > 1e-6:
                return (
                    "wrapper enforcement route bound does not equal the "
                    "registered deadline plus the frozen 0.25 s window"
                )
        except (TypeError, ValueError):
            return "wrapper enforcement route bound is malformed"
    host_exit = record.get("host_exit_monotonic")
    if host_exit is None and isinstance(attempt, Mapping):
        host_exit = attempt.get("host_exit_monotonic")
    try:
        host_exit_monotonic = float(host_exit)
    except (TypeError, ValueError):
        return "wrapper enforcement route has no host exit timestamp"
    if host_exit_monotonic > deadline + float(bound_s) + 1e-6:
        return (
            "wrapper enforcement route host exit is after the registered "
            "deadline plus the frozen 0.25 s window"
        )
    if not isinstance(attempt, Mapping):
        return "wrapper enforcement route has no attempt evidence"
    if attempt.get("start_received_within_budget") is not True:
        return (
            "wrapper enforcement route lacks a host receipt of execution entry "
            "before the deadline"
        )
    if attempt.get("deadline_alive") is not True:
        return (
            "wrapper enforcement route lacks the alive-at-deadline-notice "
            "observation"
        )
    if attempt.get("termination_verified") is not True:
        return "wrapper enforcement route lacks verified termination"
    if attempt.get("cleanup_verified") is not True:
        return "wrapper enforcement route lacks verified cleanup"
    if not wrapper_enforcement_route_verified(
        lifecycle_valid=True,
        start_received_within_budget=attempt.get("start_received_within_budget"),
        deadline_alive=attempt.get("deadline_alive"),
        termination_verified=attempt.get("termination_verified"),
        cleanup_verified=attempt.get("cleanup_verified"),
        host_exit_monotonic=host_exit_monotonic,
        host_deadline_monotonic=deadline,
        bound_s=bound_s,
    ):
        return "wrapper enforcement route evidence is inconsistent"
    frames = list(attempt.get("lifecycle_frames_raw") or [])
    key = attempt.get("lifecycle_auth_key")
    token = attempt.get("lifecycle_attempt")
    if not frames:
        return "wrapper kill record is missing"
    if not key or not token:
        return "wrapper kill record is not re-verifiable"
    authenticated, auth_error = authenticate_lifecycle_frames(
        frames, str(key), str(token)
    )
    if auth_error:
        return f"wrapper kill record is not authenticated: {auth_error}"
    if not authenticated:
        return "wrapper kill record is missing"
    reason = lifecycle_record_error(
        markers=authenticated,
        marker_error=None,
        cleanup_verified=True,
        timed_out=True,
        host_start_received_within_budget=True,
    )
    if reason:
        return f"wrapper kill record is not valid: {reason}"
    starts = [
        item for item in authenticated if item.get("event") == "candidate_start"
    ]
    finishes = [
        item for item in authenticated if item.get("event") == "candidate_finish"
    ]
    if len(starts) != 1 or len(finishes) != 1:
        return "wrapper kill record is missing or ambiguous"
    if finishes[0].get("terminated") is not True:
        return (
            "wrapper kill record does not show the wrapper's kill of the "
            "identified child"
        )
    if finishes[0].get("child_pid") != starts[0].get("child_pid"):
        return "wrapper kill record is not bound to the identified child"
    return None


def _host_deadline_enforcement_reason(
    record: Any,
    *,
    bound_s: float,
    required: bool,
) -> str | None:
    """Re-check the complete host action set against the frozen bound.

    The attempt evidence must carry the host monotonic deadline and bound and
    exactly one terminal slot for every frozen enforcement action: the docker
    client hard kill, ``docker rm -f`` and ``docker kill -s KILL``.  A missing
    or extra action, a slot without a terminal return, a failed action, or a
    return after the bound (including a still-scheduled or in-flight snapshot)
    is an overrun and must remain infrastructure/unknown.  A timely sibling
    never masks an omitted or late action.
    """

    if not isinstance(record, Mapping) or not record:
        return (
            "missing deadline enforcement issuance evidence"
            if required
            else None
        )
    if record.get("triggered") is not True:
        return (
            "deadline enforcement was not triggered for a timed-out attempt"
            if required
            else None
        )
    try:
        deadline = float(record["deadline_monotonic"])
        bound = float(record["bound_monotonic"])
    except (KeyError, TypeError, ValueError):
        return "deadline enforcement issuance times are missing"
    if abs((deadline + float(bound_s)) - bound) > 1e-6:
        return (
            "deadline enforcement bound does not equal the registered "
            "deadline plus the frozen 0.25 s window"
        )
    client = record.get("client_kill")
    if not isinstance(client, Mapping):
        return "deadline enforcement has no docker client-kill issuance"
    slots = _deadline_enforcement_slots(record)
    if not slots:
        return "deadline enforcement has no container-kill issuance"
    expected = list(DEADLINE_ENFORCEMENT_ACTIONS)
    names: list[str] = []
    for slot in slots:
        if not isinstance(slot, Mapping):
            return "deadline enforcement primitive evidence is malformed"
        name = _enforcement_slot_action(slot)
        if not name:
            return "deadline enforcement primitive evidence is malformed"
        names.append(name)
    missing = [action for action in expected if action not in names]
    extra = sorted(set(names) - set(expected))
    if missing:
        return (
            "deadline enforcement is missing expected action(s): "
            + ", ".join(missing)
            + "; this is an overrun"
        )
    if extra:
        return (
            "deadline enforcement has unexpected action(s): "
            + ", ".join(extra)
            + "; this is an overrun"
        )
    every_action_within_bound = True
    action_failures: list[str] = []
    for slot, name in zip(slots, names):
        status = slot.get("status")
        if status not in (None, "returned"):
            every_action_within_bound = False
            action_failures.append(
                f"{name} was not in a terminal successful state ({status})"
            )
            continue
        if slot.get("returned") is False:
            every_action_within_bound = False
            action_failures.append(f"{name} has no terminal return")
            continue
        if slot.get("issued") is not True:
            every_action_within_bound = False
            action_failures.append(f"{name} was not issued")
            continue
        if slot.get("error") is not None:
            every_action_within_bound = False
            action_failures.append(
                f"{name} failed: {slot.get('error')}"
            )
            continue
        launch_started = _enforcement_slot_started(slot)
        if launch_started is None:
            return f"{name} launch start is missing"
        launch_returned = _enforcement_slot_returned(slot)
        if launch_returned is None:
            every_action_within_bound = False
            action_failures.append(
                f"{name} has no return timestamp (in flight at the bound)"
            )
            continue
        if launch_returned < launch_started:
            return f"{name} issuance times are out of order"
        if launch_returned > bound + 1e-6:
            every_action_within_bound = False
            action_failures.append(
                f"{name} returned after the frozen bound"
            )
    actual_overrun = not every_action_within_bound
    recorded_overrun = bool(record.get("overrun"))
    if recorded_overrun != actual_overrun:
        return "deadline enforcement overrun flag does not recompute"
    if actual_overrun:
        reason = str(record.get("overrun_reason") or "").strip()
        if not reason:
            reason = (
                "deadline enforcement issuance overran the frozen 0.25 s bound"
            )
        if action_failures:
            reason = reason + ": " + "; ".join(action_failures)
        return reason
    return None


def _deadline_enforcement_reason(
    record: Any,
    *,
    bound_s: float,
    required: bool,
    attempt: Any = None,
) -> str | None:
    """Dispatch the two accepted enforcement routes, or fail closed.

    A conclusive outer-deadline failure is established either by the wrapper
    route (A: the wrapper killed the identified child before the host window)
    or the host route (B: the complete frozen action set returned in time).
    Missing, inconsistent, unknown or overrun evidence is never conclusive.
    """

    if not isinstance(record, Mapping) or not record:
        return (
            "missing deadline enforcement issuance evidence"
            if required
            else None
        )
    route = _enforcement_route_from_evidence(record, attempt)
    if route == ENFORCEMENT_ROUTE_WRAPPER:
        return _wrapper_deadline_enforcement_reason(
            record, bound_s=bound_s, attempt=attempt
        )
    if route == ENFORCEMENT_ROUTE_HOST:
        return _host_deadline_enforcement_reason(
            record, bound_s=bound_s, required=required
        )
    if route is not None:
        return (
            f"deadline enforcement route {route!r} is not one of "
            f"{list(DEADLINE_ENFORCEMENT_ROUTES)}"
        )
    if required:
        return (
            "deadline enforcement was not triggered for a timed-out attempt "
            "and no wrapper route was recorded"
        )
    return None


def _validate_per_test_coverage(
    per_test: Sequence[Mapping[str, Any]],
    *,
    expected_n: int,
    expected_sha: str,
    inventory: Mapping[str, Any] | None,
    test_bytes: Sequence[str] | None = None,
    request_id: str | None = None,
    enforcement_bound_s: float | None = None,
) -> str | None:
    """Return a refusal reason unless per-test rows cover the frozen inventory.

    Every conclusive row must carry the exact HMAC-framed lifecycle bytes and
    the per-attempt key needed to re-authenticate them after the attempt ended.
    The loader re-verifies the frames, recomputes the lifecycle class and state
    flags, and recomputes the frozen output comparison from the retained
    candidate stdout and the frozen test bytes.  A row with missing, forged,
    decoded-only or non-reproducible evidence is refused.
    """

    if not isinstance(inventory, Mapping):
        return "missing frozen test inventory"
    if int(inventory.get("n_tests") or -1) != int(expected_n):
        return "per-request inventory disagrees with the frozen test count"
    if str(inventory.get("tests_sha256") or "") != str(expected_sha):
        return "per-request inventory disagrees with the frozen test bytes"
    if len(per_test) != int(expected_n):
        return (
            f"grader returned {len(per_test)} per-test rows for "
            f"{int(expected_n)} frozen tests"
        )
    indexes = [int(row.get("index", -1)) for row in per_test]
    if indexes != list(range(int(expected_n))):
        return f"per-test indexes are not the complete ordered set: {indexes[:5]}"
    if len(set(indexes)) != len(indexes):
        return "per-test indexes are not unique"
    for row in per_test:
        index = int(row.get("index"))
        attempts = list(row.get("attempts") or [])
        if not attempts:
            return f"test {index} has no persisted attempt evidence"
        if len(attempts) > 2:
            return f"test {index} has more than the one precommitted retry"
        if len(attempts) == 2 and attempts[0].get("status") != "infrastructure_error":
            return f"test {index} retried a conclusive result"
        if attempts[-1].get("status") != row.get("status"):
            return f"test {index} attempt summary disagrees with its final row"
        # Cleanup is established per attempt, not just for the final row.  A
        # prior attempt whose container removal is unverified remains a
        # blocker even when the retry itself is conclusive.  An interrupted
        # attempt may only be accepted through a separate, validated,
        # container-bound cleanup-recovered record; it never rewrites the
        # interrupted record or changes its infrastructure outcome.
        for attempt in attempts:
            if attempt.get("interrupted"):
                # The interrupted attempt's own cleanup stays False; acceptance
                # comes only from a separate, validated, container-bound
                # cleanup-recovered record.
                if not _attempt_cleanup_recovery_valid(
                    attempt, request_id=request_id, test_index=index
                ):
                    return (
                        f"test {index} attempt {attempt.get('attempt')} has "
                        "unverified cleanup; it remains blocking"
                    )
            elif attempt.get("cleanup_verified") is not True:
                return (
                    f"test {index} attempt {attempt.get('attempt')} has "
                    "unverified cleanup; it remains blocking"
                )
            lifecycle = attempt.get("lifecycle_class")
            if lifecycle in {LIFECYCLE_LOCAL, LIFECYCLE_LOCAL_TIMEOUT}:
                continue
            reason = _deadline_enforcement_reason(
                attempt.get("deadline_enforcement"),
                bound_s=(
                    float(enforcement_bound_s)
                    if enforcement_bound_s is not None
                    else float(LIFECYCLE_NOTICE_GRACE_S)
                ),
                required=bool(attempt.get("timed_out")),
                attempt=attempt,
            )
            if reason:
                return (
                    f"test {index} attempt {attempt.get('attempt')} has "
                    f"invalid deadline enforcement issuance: {reason}"
                )
        if row.get("terminal_infrastructure"):
            if row.get("cleanup_verified") is not True:
                return (
                    f"test {index} terminal infrastructure has unverified "
                    "cleanup; it remains blocking"
                )
            if (
                row.get("timed_out")
                and row.get("termination_verified") is not True
            ):
                return (
                    f"test {index} terminal timeout has unverified termination; "
                    "it remains blocking"
                )
        recomputed_infrastructure, recomputed_status = _classify_test(row)
        if bool(row.get("infrastructure_failure")) != recomputed_infrastructure:
            return f"test {index} infrastructure flag does not recompute"
        if str(row.get("status")) != recomputed_status:
            return f"test {index} status does not recompute"
        if bool(row.get("passed")) != (recomputed_status == "passed"):
            return f"test {index} passed flag does not recompute"
        lifecycle = row.get("lifecycle_class")
        records = list(
            row.get("lifecycle_records") or row.get("lifecycle_markers") or []
        )
        if lifecycle in {LIFECYCLE_LOCAL, LIFECYCLE_LOCAL_TIMEOUT}:
            pass
        elif recomputed_status != "infrastructure_error":
            frames = list(row.get("lifecycle_frames_raw") or [])
            key = row.get("lifecycle_auth_key")
            attempt = row.get("lifecycle_attempt")
            if not frames:
                return (
                    f"test {index} conclusive outcome has no retained raw "
                    "lifecycle frames"
                )
            if not key or not attempt:
                return (
                    f"test {index} conclusive outcome has no re-verifiable "
                    "per-attempt key or attempt identity"
                )
            authenticated, auth_error = authenticate_lifecycle_frames(
                frames, str(key), str(attempt)
            )
            if auth_error:
                return (
                    f"test {index} retained lifecycle frames do not "
                    f"re-authenticate: {auth_error}"
                )
            if authenticated != records:
                return (
                    f"test {index} decoded lifecycle records do not equal the "
                    "authenticated frames"
                )
            reason = lifecycle_record_error(
                markers=authenticated,
                marker_error=None,
                cleanup_verified=row.get("cleanup_verified"),
                timed_out=bool(row.get("timed_out")),
                host_completed_within_budget=row.get(
                    "completed_within_budget"
                ),
                host_start_received_within_budget=row.get(
                    "start_received_within_budget"
                ),
            )
            if reason:
                return (
                    f"test {index} lifecycle record does not independently "
                    f"validate: {reason}"
                )
            recomputed_class = classify_lifecycle_evidence(
                markers=authenticated,
                marker_error=None,
                timed_out=bool(row.get("timed_out")),
                cleanup_verified=row.get("cleanup_verified"),
                deadline_epoch=0.0,
                host_completed_within_budget=row.get(
                    "completed_within_budget"
                ),
                host_start_received_within_budget=row.get(
                    "start_received_within_budget"
                ),
            )
            if recomputed_class != lifecycle:
                return (
                    f"test {index} lifecycle class does not independently "
                    "recompute"
                )
            if row.get("lifecycle_valid") is not True:
                return f"test {index} lifecycle evidence is not valid"
        if recomputed_status == "infrastructure_error":
            # Infrastructure/unknown is not a conclusive outcome; its row is
            # retained for terminal/blocker accounting rather than accepted as
            # a candidate result.
            continue
        # Independently recompute the frozen output comparison.
        frozen = _frozen_test_at(test_bytes, index)
        if frozen is None:
            return (
                f"test {index} has no frozen test bytes for an independent "
                "output comparison"
            )
        expected = str(frozen.get("output") or "")
        test_type = str(frozen.get("type") or "stdin")
        if str(row.get("expected") or "") != expected:
            return f"test {index} expected output disagrees with the frozen test"
        if str(row.get("input") or "") != str(frozen.get("input") or ""):
            return f"test {index} input disagrees with the frozen test"
        if str(row.get("test_type") or "stdin") != test_type:
            return f"test {index} test type disagrees with the frozen test"
        actual = str(row.get("stdout") or "")
        matches = _compare_grader_output(actual, expected, test_type)
        status = str(row.get("status"))
        if status == "passed":
            if not matches:
                return (
                    f"test {index} retained stdout does not reproduce the "
                    "frozen pass comparison"
                )
            if row.get("exit_code") != 0:
                return f"test {index} pass has a nonzero exit code"
        elif status == "wrong_answer" and matches:
            return (
                f"test {index} wrong-answer row reproduces a passing output "
                "comparison"
            )
    return None


def _per_test_accounting(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, test in enumerate(payload.get("tests") or []):
        if not isinstance(test, Mapping):
            continue
        infrastructure, status = _classify_test(test)
        rows.append(
            {
                "index": int(test.get("index", index)),
                "status": status,
                "passed": bool(test.get("passed")),
                "exit_code": test.get("exit_code"),
                "timed_out": bool(test.get("timed_out")),
                "duration_s": float(test.get("duration_s") or 0.0),
                "error": test.get("error"),
                "stderr": _truncate_stderr(test.get("stderr")),
                # Full candidate stdout is retained so a loader can recompute
                # the frozen output comparison independently.
                "stdout": str(test.get("stdout") or ""),
                "stdout_sha256": _sha256_text(str(test.get("stdout") or "")),
                "input": str(test.get("input") or ""),
                "expected": str(test.get("expected") or ""),
                "test_type": str(test.get("test_type") or "stdin"),
                "infrastructure_failure": infrastructure,
                "lifecycle_class": test.get("lifecycle_class"),
                "container_name": test.get("container_name"),
                "container_started_at": test.get("container_started_at"),
                "container_finished_at": test.get("container_finished_at"),
                "lifecycle_events": list(test.get("lifecycle_events") or []),
                "lifecycle_markers": list(test.get("lifecycle_markers") or []),
                "lifecycle_records": list(
                    test.get("lifecycle_records") or test.get("lifecycle_markers") or []
                ),
                "lifecycle_frames_raw": list(
                    test.get("lifecycle_frames_raw") or []
                ),
                "lifecycle_auth_key": test.get("lifecycle_auth_key"),
                "marker_error": test.get("marker_error"),
                "lifecycle_valid": test.get("lifecycle_valid"),
                "completed_within_budget": test.get("completed_within_budget"),
                "start_received_within_budget": test.get(
                    "start_received_within_budget"
                ),
                "start_received_monotonic": test.get(
                    "start_received_monotonic"
                ),
                "host_deadline_monotonic": test.get("host_deadline_monotonic"),
                "deadline_enforcement": dict(
                    test.get("deadline_enforcement") or {}
                ),
                "deadline_enforcement_overrun": test.get(
                    "deadline_enforcement_overrun"
                ),
                "enforcement_route": test.get("enforcement_route"),
                "host_exit_monotonic": test.get("host_exit_monotonic"),
                "cleanup_verified": test.get("cleanup_verified"),
                "handoff_success": test.get("handoff_success"),
                "termination_verified": test.get("termination_verified"),
                "deadline_alive": test.get("deadline_alive"),
                "exit_observed": test.get("exit_observed"),
                "lifecycle_attempt": test.get("lifecycle_attempt"),
                "attempts": list(test.get("attempts") or []),
            }
        )
    return rows


def _attempt_summary(test: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the complete evidence for one attempt, not just its verdict."""

    return {
        "status": test.get("status"),
        "passed": bool(test.get("passed")),
        "infrastructure_failure": bool(test.get("infrastructure_failure")),
        "exit_code": test.get("exit_code"),
        "timed_out": bool(test.get("timed_out")),
        "duration_s": float(test.get("duration_s") or 0.0),
        "stdout": str(test.get("stdout") or ""),
        "stdout_sha256": _sha256_text(str(test.get("stdout") or "")),
        "input": str(test.get("input") or ""),
        "expected": str(test.get("expected") or ""),
        "test_type": str(test.get("test_type") or "stdin"),
        "lifecycle_class": test.get("lifecycle_class"),
        "container_name": test.get("container_name"),
        "container_started_at": test.get("container_started_at"),
        "container_finished_at": test.get("container_finished_at"),
        "lifecycle_events": list(test.get("lifecycle_events") or []),
        "lifecycle_markers": list(test.get("lifecycle_markers") or []),
        "lifecycle_records": list(
            test.get("lifecycle_records") or test.get("lifecycle_markers") or []
        ),
        "lifecycle_frames_raw": list(test.get("lifecycle_frames_raw") or []),
        "lifecycle_auth_key": test.get("lifecycle_auth_key"),
        "marker_error": test.get("marker_error"),
        "lifecycle_valid": test.get("lifecycle_valid"),
        "completed_within_budget": test.get("completed_within_budget"),
        "start_received_within_budget": test.get("start_received_within_budget"),
        "start_received_monotonic": test.get("start_received_monotonic"),
        "host_deadline_monotonic": test.get("host_deadline_monotonic"),
        "deadline_enforcement": dict(test.get("deadline_enforcement") or {}),
        "deadline_enforcement_overrun": test.get(
            "deadline_enforcement_overrun"
        ),
        "enforcement_route": test.get("enforcement_route"),
        "host_exit_monotonic": test.get("host_exit_monotonic"),
        "cleanup_verified": test.get("cleanup_verified"),
        "handoff_success": test.get("handoff_success"),
        "termination_verified": test.get("termination_verified"),
        "deadline_alive": test.get("deadline_alive"),
        "exit_observed": test.get("exit_observed"),
        "lifecycle_attempt": test.get("lifecycle_attempt"),
        "partial_result": bool(test.get("partial_result")),
        "terminal_infrastructure": bool(test.get("terminal_infrastructure")),
        "terminal_status": test.get("terminal_status"),
        "terminal_reason": test.get("terminal_reason"),
        "recovery_evidence": test.get("recovery_evidence"),
        "error": test.get("error"),
        "stderr": _truncate_stderr(test.get("stderr")),
    }


def _apply_terminal_infrastructure(
    per_test: Sequence[dict[str, Any]], max_retries: int
) -> None:
    """Mark tests that exhausted the registered single retry as terminal.

    A test is terminal only after an infrastructure-affected first attempt and
    an infrastructure-affected (or interrupted) identical retry, with no third
    attempt.  The infrastructure reason is retained; the test is never
    relabeled wrong answer, candidate error or timeout.  A test whose retry
    is still pending, or whose coverage/inventory is invalid, remains a
    blocker and is deliberately not marked.
    """

    allowance = int(max_retries)
    if allowance < 1:
        return
    for row in per_test:
        if not row.get("infrastructure_failure"):
            continue
        attempts = list(row.get("attempts") or [])
        if len(attempts) < 1 + allowance:
            continue
        last = attempts[-1]
        if last.get("status") != "infrastructure_error":
            continue
        # X1 keeps unverified termination/cleanup blocking.  Terminal
        # infrastructure may only be assigned when *every* attempt of this test
        # has accepted cleanup (its own verified cleanup, or for an interrupted
        # attempt a separate container-bound cleanup-recovered record), and for
        # a deadline failure that the tracked child was terminated.
        test_index = int(row.get("index", -1))
        if any(
            not _attempt_cleanup_accepted(attempt, test_index=test_index)
            for attempt in attempts
        ):
            continue
        if row.get("cleanup_verified") is not True:
            continue
        if row.get("timed_out") and row.get("termination_verified") is not True:
            continue
        reason = (
            row.get("error")
            or last.get("error")
            or "infrastructure unknown after the single identical retry"
        )
        row["terminal_infrastructure"] = True
        row["terminal_status"] = TEST_STATUS_TERMINAL_INFRASTRUCTURE
        row["terminal_reason"] = str(reason)
        row["infrastructure_failure"] = True


_MOUNT_STDERR_MARKERS = (
    "mount",
    "bind",
    "no such file or directory",
    "invalid mount",
    "not a directory",
)
_CLEANUP_STDERR_MARKERS = ("removal of container", "device or resource busy")


def _recovery_failure_classes(
    rows: Sequence[Mapping[str, Any]],
    *,
    in_flight: bool = False,
) -> list[str]:
    """Classify the observed infrastructure failure for the recovery check."""

    classes: set[str] = set()
    for row in rows:
        if not row.get("infrastructure_failure"):
            continue
        if row.get("cleanup_verified") is not True or any(
            marker in str(row.get("stderr") or "").lower()
            for marker in _CLEANUP_STDERR_MARKERS
        ):
            classes.add("cleanup")
        stderr = str(row.get("stderr") or "").lower()
        if any(marker in stderr for marker in _MOUNT_STDERR_MARKERS):
            classes.add("mount")
        lifecycle = row.get("lifecycle_class")
        exit_code = row.get("exit_code")
        if (
            lifecycle in {LIFECYCLE_STARTUP_FAILURE, LIFECYCLE_UNRESOLVED, None}
            or exit_code in {125, 126, 127}
            or any(
                marker in stderr
                for marker in (
                    "cannot connect to the docker daemon",
                    "is the docker daemon running",
                    "error response from daemon",
                    "permission denied while trying to connect",
                    "no such image",
                    "exec format error",
                )
            )
        ):
            classes.add("startup")
    if in_flight:
        # An interrupted in-flight attempt has no observed class; recover the
        # daemon and prove a container can start and mount.  Also inspect the
        # interrupted attempt's own recorded container and prove it is gone (or
        # remove it and verify removal) so its cleanup can be recovered.
        classes.update({"startup", "mount", "cleanup"})
    if not classes:
        classes.add("unknown")
    return sorted(classes)


def _invoke_recovery_check(
    check: Callable[..., Mapping[str, Any] | bool] | None,
    *,
    failure_classes: Sequence[str],
    image: str,
    container_name: str | None,
) -> Mapping[str, Any] | bool | None:
    if check is None:
        return None
    kwargs = {
        "failure_classes": list(failure_classes),
        "image": image,
        "container_name": container_name,
    }
    try:
        signature = inspect.signature(check)
    except (TypeError, ValueError):
        return check()
    parameters = signature.parameters.values()
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return check(**kwargs)
    supported = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return check(**supported)


def _docker_recovery_check(
    failure_classes: Sequence[str] | None = None,
    *,
    image: str = LITE_IMAGE_TAG,
    container_name: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Documented recovery evidence specific to the observed failure class.

    A daemon-version response alone is not recovery.  Startup/daemon failures
    require a responsive daemon *and* a no-op container with the pinned image
    that starts, runs and is removed; mount failures require a real bind-mount
    read; cleanup failures require the stale container to be confirmed gone.
    The exact commands, return codes and observed output are persisted with the
    single precommitted retry.
    """

    classes = sorted({str(value) for value in (failure_classes or ["unknown"])})
    checked_utc = _utc_now()
    checks: dict[str, Any] = {}

    def run(command: list[str], *, timeout: float = 30) -> Any:
        try:
            return runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=sanitized_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(
                command, 1, "", f"{type(exc).__name__}: {exc}"
            )

    def record(label: str, command: list[str], completed: Any, ok: bool) -> None:
        checks[label] = {
            "command": command,
            "returncode": int(getattr(completed, "returncode", 1)),
            "stdout": str(getattr(completed, "stdout", "") or "").strip()[:STDERR_LIMIT],
            "stderr": str(getattr(completed, "stderr", "") or "").strip()[:STDERR_LIMIT],
            "ok": bool(ok),
        }

    def no_such_container(name: str) -> tuple[bool, Any]:
        completed = run(["docker", "inspect", "--format", "{{.Id}}", name], timeout=15)
        text = (str(completed.stdout or "") + str(completed.stderr or "")).lower()
        return completed.returncode != 0 and "no such" in text, completed

    needs_daemon = bool(
        {"startup", "daemon", "unknown", "mount"} & set(classes)
    )
    if needs_daemon:
        command = ["docker", "info", "--format", "{{.ServerVersion}}"]
        completed = run(command, timeout=15)
        version = str(completed.stdout or "").strip()
        record("daemon", command, completed, completed.returncode == 0 and bool(version))
    if {"startup", "daemon", "unknown"} & set(classes):
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            "gemmaloop-recovery-noop",
            "--network",
            "none",
            "--read-only",
            "--user",
            "65534:65534",
            "--security-opt",
            "no-new-privileges",
            image,
            "python",
            "-c",
            "print('recovery-ok')",
        ]
        completed = run(command, timeout=60)
        output = str(completed.stdout or "").strip()
        gone, inspect = no_such_container("gemmaloop-recovery-noop")
        record(
            "noop_container",
            command,
            completed,
            completed.returncode == 0 and output == "recovery-ok" and gone,
        )
        checks["noop_container"]["removed"] = gone
        checks["noop_container"]["removal_probe"] = {
            "returncode": int(getattr(inspect, "returncode", 1)),
            "detail": (
                str(getattr(inspect, "stderr", "") or "")
                or str(getattr(inspect, "stdout", "") or "")
            ).strip()[:STDERR_LIMIT],
        }
    if "mount" in classes:
        with tempfile.TemporaryDirectory(prefix="gemmaloop_recovery_mount_") as tmp:
            sentinel = Path(tmp) / "sentinel.txt"
            sentinel.write_text("mount-ok\n", encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                "gemmaloop-recovery-mount",
                "--network",
                "none",
                "--read-only",
                "--user",
                "65534:65534",
                "-v",
                f"{Path(tmp).resolve()}:/recovery-mount:ro",
                image,
                "python",
                "-c",
                "print(open('/recovery-mount/sentinel.txt').read().strip())",
            ]
            completed = run(command, timeout=60)
            gone, _inspect = no_such_container("gemmaloop-recovery-mount")
            record(
                "bind_mount",
                command,
                completed,
                completed.returncode == 0
                and str(completed.stdout or "").strip() == "mount-ok"
                and gone,
            )
    if "cleanup" in classes:
        name = container_name or "gemmaloop-recovery-cleanup"
        command = ["docker", "rm", "-f", name]
        completed = run(command, timeout=30)
        gone, inspect = no_such_container(name)
        record(
            "cleanup_removal",
            command,
            completed,
            gone,
        )
        checks["cleanup_removal"]["removed"] = gone
        checks["cleanup_removal"]["removal_probe"] = {
            "returncode": int(getattr(inspect, "returncode", 1)),
            "detail": (
                str(getattr(inspect, "stderr", "") or "")
                or str(getattr(inspect, "stdout", "") or "")
            ).strip()[:STDERR_LIMIT],
        }
    ok = bool(checks) and all(check["ok"] for check in checks.values())
    return {
        "check": "docker recovery: " + ", ".join(classes),
        "failure_classes": classes,
        "checks": checks,
        "checked_utc": checked_utc,
        "image": str(image),
        "container_name": container_name,
        "ok": ok,
        "detail": (
            "all required recovery checks passed"
            if ok
            else "; ".join(
                label for label, check in checks.items() if not check.get("ok")
            )
            or "no recovery check ran"
        ),
    }


def _normalise_recovery_evidence(value: Any) -> dict[str, Any] | None:
    """Return documented recovery evidence, or ``None`` when undocumented."""

    if not isinstance(value, Mapping):
        return None
    if value.get("ok") is not True:
        return None
    if not value.get("check") or not value.get("checked_utc"):
        return None
    checks = value.get("checks")
    if checks is not None:
        if not isinstance(checks, Mapping) or not checks:
            return None
        if not all(
            isinstance(check, Mapping) and check.get("ok") is True
            for check in checks.values()
        ):
            return None
    return dict(value)


def _cleanup_recovery_valid(
    record: Mapping[str, Any],
    *,
    request_id: str | None = None,
    test_index: int | None = None,
    attempt: int | None = None,
    container_name: str | None = None,
) -> bool:
    """Validate container-bound recovery evidence for an interrupted attempt.

    The recovery record must be bound to the interrupted request/test/attempt
    and to the *actual* container committed before dispatch.  It must contain a
    successful removal (or absence) observation whose command names that same
    container.  It never makes the interrupted attempt conclusive and never
    changes its infrastructure status.
    """

    if record.get("kind") != LEDGER_CLEANUP_RECOVERED:
        return False
    if record.get("cleanup_verified") is not True:
        return False
    name = str(record.get("container_name") or "")
    if not name:
        return False
    if container_name is not None and name != str(container_name):
        return False
    if request_id is not None and str(record.get("request_id") or "") != str(request_id):
        return False
    try:
        record_index = int(record.get("test_index", -1))
        record_attempt = int(record.get("attempt") or 1)
    except (TypeError, ValueError):
        return False
    if test_index is not None and record_index != int(test_index):
        return False
    if attempt is not None and record_attempt != int(attempt):
        return False
    evidence = record.get("recovery_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("ok") is not True:
        return False
    if str(evidence.get("container_name") or "") != name:
        return False
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        return False
    for check in checks.values():
        if not isinstance(check, Mapping):
            continue
        if check.get("ok") is not True or check.get("removed") is not True:
            continue
        command = check.get("command")
        if not isinstance(command, (list, tuple)):
            continue
        if name in [str(part) for part in command]:
            return True
    return False


def _cleanup_recovery_record(
    *,
    request_id: str,
    problem_id: str,
    test_index: int,
    attempt: int,
    container_name: str | None,
    attempt_token: str | None,
    recovery: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the append-only cleanup-recovered record, or ``None`` if invalid."""

    name = str(container_name or "")
    evidence = _normalise_recovery_evidence(recovery)
    if not name or evidence is None:
        return None
    record: dict[str, Any] = {
        "kind": LEDGER_CLEANUP_RECOVERED,
        "request_id": str(request_id),
        "problem_id": str(problem_id),
        "test_index": int(test_index),
        "attempt": int(attempt),
        "container_name": name,
        "attempt_token": str(attempt_token or ""),
        "recovered_utc": _utc_now(),
        "cleanup_verified": True,
        "recovery_evidence": evidence,
    }
    return record if _cleanup_recovery_valid(record) else None


def _attempt_cleanup_recovery_valid(
    attempt: Mapping[str, Any],
    *,
    request_id: str | None,
    test_index: int,
) -> bool:
    record = attempt.get("cleanup_recovery")
    if not isinstance(record, Mapping):
        return False
    return _cleanup_recovery_valid(
        record,
        request_id=request_id,
        test_index=test_index,
        attempt=int(attempt.get("attempt") or 1),
        container_name=attempt.get("container_name"),
    )


def _attempt_cleanup_accepted(
    attempt: Mapping[str, Any], *, test_index: int
) -> bool:
    """Whether an attempt's cleanup is accepted for terminal accounting."""

    if attempt.get("interrupted"):
        return _attempt_cleanup_recovery_valid(
            attempt, request_id=None, test_index=test_index
        )
    return attempt.get("cleanup_verified") is True


def _test_identity_sha256(test: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(dict(test), ensure_ascii=False, sort_keys=True))


def _ledger_identity(
    freeze: Mapping[str, Any],
    implementation: Mapping[str, Any] | None,
    *,
    timeout_s: float,
    retry_policy: Mapping[str, Any] | None = None,
    run_token: str | None = None,
    claim_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Everything a resume must re-verify before reusing the attempt ledger.

    When the count runs under the claimed official registry, the ledger header
    also binds the run token (by digest, never in clear text) and the immutable
    claim-record digest captured once when the run was claimed.  The evolving
    registry file hash is *never* an immutable resume requirement; it is
    recorded separately in the header as an audit-only count-start digest.
    Independent loading can therefore prove that the durable ledger belongs to
    the claimed run rather than merely to the same freeze.
    """

    verifier = freeze.get("verifier") or {}
    image = dict(verifier.get("docker_image") or {})
    docker = dict(verifier.get("docker") or {})
    identity: dict[str, Any] = {
        "freeze_manifest_sha256": str(freeze.get("manifest_sha256") or ""),
        "freeze_test_manifest_sha256": str(
            (freeze.get("tests") or {}).get("manifest_sha256") or ""
        ),
        "selected_inputs_sha256": str(
            (freeze.get("selected_inputs") or {}).get("selected_inputs_sha256")
            or ""
        ),
        "executing_implementation_sha256": _sha256_text(
            json.dumps(dict(implementation or {}), ensure_ascii=False, sort_keys=True)
        ),
        "image_digest": str(
            image.get("pinned") or image.get("digest") or image.get("id") or ""
        ),
        "image_config": image,
        "sandbox_configuration": docker,
        "timeout_s": float(timeout_s),
        "retry_policy": dict(retry_policy or verifier.get("retry") or {}),
    }
    if run_token is not None:
        identity["run_token_sha256"] = _sha256_text(str(run_token))
    if claim_registry_sha256 is not None:
        identity["claim_registry_sha256"] = str(claim_registry_sha256)
    return identity


_LEDGER_FRAME_PREFIX = "GML-L1"
_LEDGER_APPEND_LOCK = threading.Lock()


def _encode_ledger_record(record: Mapping[str, Any]) -> bytes:
    body = json.dumps(
        dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    return (
        f"{_LEDGER_FRAME_PREFIX} {len(body)} {digest} ".encode("ascii")
        + body
        + b"\n"
    )


def _decode_ledger_frame(line: bytes) -> dict[str, Any]:
    parts = line.split(b" ", 3)
    if len(parts) != 4 or parts[0] != _LEDGER_FRAME_PREFIX.encode("ascii"):
        raise ValueError("not a framed ledger record")
    try:
        length = int(parts[1])
    except ValueError as exc:
        raise ValueError("ledger record length is not an integer") from exc
    digest = parts[2].decode("ascii", "replace")
    body = parts[3]
    if len(body) != length:
        raise ValueError(
            f"ledger record length {len(body)} does not match its header {length}"
        )
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("ledger record checksum does not recompute")
    try:
        record = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ledger record is not valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError("ledger record is not an object")
    return record


def _read_attempt_ledger(path: Path) -> list[dict[str, Any]]:
    """Read the framed append-only ledger with constrained torn-tail recovery.

    Each record is ``GML-L1 <length> <sha256> <canonical-json>`` and carries
    its own integrity check.  Only a *truncated* final write -- no terminating
    newline, hence an incomplete frame -- may be dropped (and the file
    truncated to the last valid record).  Any complete, newline-terminated
    record is committed history: if it is invalid, anywhere including the last
    line, the ledger is inconsistent and is refused without rewriting it.
    """

    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    offset = 0
    valid_end = 0
    total = len(data)
    while offset < total:
        newline = data.find(b"\n", offset)
        if newline == -1:
            # An incomplete final write is the only recoverable torn tail.
            break
        line = data[offset:newline]
        try:
            record = _decode_ledger_frame(line)
        except ValueError as exc:
            raise ValueError(
                f"{path}: malformed ledger record at byte {offset}: {exc}"
            ) from exc
        records.append(record)
        offset = newline + 1
        valid_end = offset
    if valid_end < total:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
            handle.flush()
            os.fsync(handle.fileno())
    return records


def _ledger_verifier_interface_error_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every durable verifier interface/programming error in the run.

    Such records are a run-wide block: they are never candidate evidence, never
    a retry, and never a terminal infrastructure exclusion.  They must be
    visible to resume planning and to independent loading rather than filtered
    away as ordinary interrupted starts.
    """

    return [
        dict(record)
        for record in records
        if record.get("kind") == LEDGER_VERIFIER_INTERFACE_ERROR
    ]


def _verifier_interface_error_detail(
    errors: Sequence[Mapping[str, Any]],
) -> str:
    first = errors[0] if errors else {}
    return (
        f"{len(errors)} verifier interface/programming error record(s) "
        f"(first: {first.get('error_type')}: {first.get('error')})"
    )


def _refuse_verifier_interface_error_block(
    records: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Refuse any dispatch/acceptance once a programming error is durable."""

    errors = _ledger_verifier_interface_error_records(records)
    if errors:
        raise ShortfallVerifierInterfaceError(
            f"{context}: {_verifier_interface_error_detail(errors)}. The run is "
            "blocked run-wide; no retry, new attempt or terminal disposition is "
            "authorized (registered void paragraph 6)."
        )


def _append_attempt_ledger(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one framed record, serialized across threads."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _encode_ledger_record(record)
    with _LEDGER_APPEND_LOCK:
        with path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def _verify_registry_is_same_run(
    header: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    stage_names: Sequence[str],
) -> None:
    """Prove the current registry is the same run and an append-only extension.

    The ledger header binds the immutable run token and claim-record digest,
    plus the completed stage history observed when the count started.  Registry
    updates may advance status/history, but they may never change that identity
    or drop/rewrite a previously completed stage.  The evolving registry file
    hash is deliberately not compared.
    """

    _validate_registry_stage_history(registry, stage_names=stage_names)
    claim = str(registry.get("claim_registry_sha256") or "")
    if not claim:
        raise ShortfallV3RunError(
            "the run registry has no immutable claim digest; a resume cannot "
            "prove it is the same run"
        )
    if header.get("claim_registry_sha256") != claim:
        raise ShortfallV3RunError(
            "the run registry claim digest does not match the durable attempt "
            "ledger; refusing a different run"
        )
    if header.get("run_token_sha256") != _sha256_text(
        str(registry.get("run_token") or "")
    ):
        raise ShortfallV3RunError(
            "the run registry token does not match the durable attempt ledger"
        )
    captured = [str(stage) for stage in header.get("count_start_completed_stages") or []]
    current = _registry_completed_stages(registry)
    if current[: len(captured)] != captured:
        raise ShortfallV3RunError(
            "the run registry stage history is not an append-only extension of "
            "the count-start history; a completed stage was removed, reordered "
            "or rewritten"
        )


def _ensure_attempt_ledger(
    path: Path,
    identity: Mapping[str, Any],
    *,
    audit_fields: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    stage_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open or verify the durable ledger header for this exact run identity.

    ``audit_fields`` are written into a newly created header for audit (for
    example the count-start registry digest and completed-stage snapshot) but
    are never compared on resume.  ``registry`` proves same-run continuity and
    an append-only stage history without pinning the evolving file hash.
    """

    path = Path(path)
    records = _read_attempt_ledger(path)
    headers = [row for row in records if row.get("kind") == LEDGER_HEADER]
    if not headers:
        if records:
            raise ValueError(f"{path}: ledger has records but no verified header")
        header: dict[str, Any] = {
            "kind": LEDGER_HEADER,
            "ledger_kind": LEDGER_KIND,
            "version": 1,
            "created_utc": _utc_now(),
            **dict(identity),
            **dict(audit_fields or {}),
            "header_sha256": "",
        }
        header["header_sha256"] = _hash_payload(header, "header_sha256")
        _append_attempt_ledger(path, header)
        if registry is not None:
            _verify_registry_is_same_run(
                header,
                registry,
                stage_names=stage_names or tuple(),
            )
        return [header], header
    if len(headers) != 1:
        raise ValueError(f"{path}: duplicate ledger headers")
    header = headers[0]
    if header.get("ledger_kind") != LEDGER_KIND:
        raise ValueError(f"{path}: not a shortfall attempt ledger")
    if header.get("header_sha256") != _hash_payload(header, "header_sha256"):
        raise ValueError(f"{path}: ledger header hash does not recompute")
    for key, expected in identity.items():
        if header.get(key) != expected:
            raise ValueError(
                f"{path}: ledger identity mismatch for {key!r}; the frozen "
                "verifier, implementation or configuration changed"
            )
    if registry is not None:
        _verify_registry_is_same_run(
            header,
            registry,
            stage_names=stage_names or tuple(),
        )
    return records, header


def _ledger_entries_by_test(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("kind") not in {
            LEDGER_ATTEMPT_START,
            LEDGER_ATTEMPT_RESULT,
            LEDGER_ATTEMPT_INTERRUPTED,
            LEDGER_CLEANUP_RECOVERED,
            LEDGER_CLEANUP_RECOVERY_FAILED,
            LEDGER_VERIFIER_INTERFACE_ERROR,
        }:
            continue
        key = (str(record.get("request_id") or ""), int(record.get("test_index", -1)))
        grouped.setdefault(key, []).append(dict(record))
    return grouped


def _ledger_plan(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any] | None, int]:
    """Return ``(conclusive_result, next_attempt, in_flight_start, used)``.

    The first conclusive result controls and is never regraded.  A start with
    no result is an interrupted in-flight attempt; it consumes the single
    registered identical retry.  A verifier interface/programming error is a
    run-wide block and is never planned as a retry.
    """

    _refuse_verifier_interface_error_block(
        entries, context="attempt-ledger planning"
    )

    by_attempt: dict[int, list[Mapping[str, Any]]] = {}
    for entry in entries:
        by_attempt.setdefault(int(entry.get("attempt") or 1), []).append(entry)
    used = max(by_attempt) if by_attempt else 0
    for attempt in sorted(by_attempt):
        results = [
            entry
            for entry in by_attempt[attempt]
            if entry.get("kind") in {LEDGER_ATTEMPT_RESULT, LEDGER_ATTEMPT_INTERRUPTED}
        ]
        if results:
            last = results[-1]
            if last.get("conclusive") is True:
                return dict(last), None, None, used
            # Infra result: a retry may exist in a later attempt.
            continue
        start = next(
            entry
            for entry in by_attempt[attempt]
            if entry.get("kind") == LEDGER_ATTEMPT_START
        )
        return None, attempt + 1, dict(start), used
    return None, (used + 1 if used else 1), None, used


def _ledger_result_record(
    *,
    request_id: str,
    problem_id: str,
    test_index: int,
    attempt: int,
    evidence: Mapping[str, Any],
    sandboxed: bool = True,
    kind: str = LEDGER_ATTEMPT_RESULT,
    recovery_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(evidence.get("status") or "")
    record = {
        "kind": kind,
        "request_id": request_id,
        "problem_id": problem_id,
        "test_index": int(test_index),
        "attempt": int(attempt),
        "status": status,
        "passed": bool(evidence.get("passed")),
        "sandboxed": bool(sandboxed),
        "infrastructure_failure": status == "infrastructure_error",
        "conclusive": status in CONCLUSIVE_ATTEMPT_STATUSES,
        "evidence": dict(evidence),
        "ended_utc": _utc_now(),
    }
    if recovery_evidence is not None:
        record["recovery_evidence"] = dict(recovery_evidence)
    return record


def _ledger_row(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    evidence = dict(record.get("evidence") or {})
    evidence.setdefault("status", record.get("status"))
    evidence.setdefault(
        "infrastructure_failure", record.get("infrastructure_failure")
    )
    evidence["index"] = int(index)
    return evidence


def grade_original_requests(
    freeze: Mapping[str, Any],
    *,
    pool_dir: Path | str,
    lite_dir: Path | str,
    grade_fn: Callable[..., Any] | None = None,
    workers: int = 1,
    timeout_s: float | None = None,
    execution_tests: Mapping[str, Mapping[str, Any]] | None = None,
    test_inventory: Mapping[str, Mapping[str, Any]] | None = None,
    interpretation_excluded: Mapping[str, str] | None = None,
    max_infrastructure_retries: int = 0,
    recovery_check: Callable[..., Mapping[str, Any] | bool] | None = None,
    ledger_path: Path | str | None = None,
    ledger_identity: Mapping[str, Any] | None = None,
    ledger_audit: Mapping[str, Any] | None = None,
    ledger_registry: Mapping[str, Any] | None = None,
    resume: bool = False,
    authorization: V3RunAuthorization | None = None,
) -> list[dict[str, Any]]:
    """Per-request accounting for every original selected request.

    Every code request carries its per-test status, exit code, timeout flag,
    trusted lifecycle class, truncated stderr and an explicit
    ``failure_class``.  Infrastructure failures are kept separate from wrong
    answers so a final count decision can refuse them.

    ``execution_tests`` supplies the authenticated per-test execution text from
    the interpretation manifest.  ``interpretation_excluded`` names whole
    problems that may not be graded at all.  When a test is infrastructure-
    affected, the precommitted policy allows at most one identical-
    configuration retry after ``recovery_check`` documents recovery; both
    attempts are preserved and the first conclusive result controls.
    """

    # Execution of original-sample grading is bound to the claimed official
    # run (or an authorized resume) and to a durable attempt ledger.  This
    # holds regardless of whether a grade_fn is supplied: an arbitrary
    # callable (including a wrapper around the real Docker grader) is never
    # authorization.  Tests use a token bound to a temporary registry.
    if authorization is None:
        raise ShortfallV3RunError(
            "grade_original_requests may only execute original-sample grading "
            "as part of the claimed official v3 run or an authorized resume"
        )
    _authorize_v3_stage(
        authorization,
        stage="count",
        lite_dir=(freeze.get("inputs") or {}).get("lite_dir") or "",
    )
    if ledger_path is None:
        raise ShortfallV3RunError(
            "grade_original_requests requires the durable attempt ledger"
        )
    pool_dir = Path(pool_dir)
    lite_dir = Path(lite_dir)
    problems = _problem_by_id(pool_dir)
    records = _load_records_by_request(lite_dir)
    grades = _load_grades_by_request(lite_dir)
    excluded = dict(interpretation_excluded or {})
    timeout = float(
        timeout_s
        if timeout_s is not None
        else (freeze.get("verifier") or {}).get("timeout_s", DEFAULT_TIMEOUT_S)
    )
    image = str(
        ((freeze.get("verifier") or {}).get("docker_image") or {}).get("pinned")
        or ((freeze.get("verifier") or {}).get("docker_image") or {}).get("id")
        or LITE_IMAGE_TAG
    )
    if grade_fn is None:

        def default_grade(
            code: str,
            tests: list[dict[str, Any]],
            timeout: float,
            container_name: str | None = None,
            attempt: str | None = None,
        ) -> Any:
            return _grade_code_default(
                code,
                tests,
                timeout=timeout,
                image=image,
                container_name=container_name,
                attempt=attempt,
            )

        grade_fn = default_grade

    entries: list[dict[str, Any]] = []
    code_jobs: list[tuple[int, str, list[dict[str, Any]]]] = []
    for name in PILOT_SPLITS:
        split = freeze["selected_inputs"]["splits"][name]
        for request in split["requests"]:
            request_id = request["request_id"]
            problem_id = request["problem_id"]
            kind = request["kind"]
            record = records.get(request_id)
            grade_row = grades.get(request_id) or {}
            kept = bool(grade_row.get("kept"))
            drop_reason = grade_row.get("drop_reason")
            entry: dict[str, Any] = {
                "request_id": request_id,
                "problem_id": problem_id,
                "split": name,
                "domain": kind,
                "sample_index": int(request["sample_index"]),
                "kept": kept,
                "drop_reason": drop_reason,
                "passed": None,
                "verified": False,
                "tests_run": 0,
                "error": None,
                "timing_s": 0.0,
                "sandboxed": None,
                "per_test": [],
                "failure_class": None,
                "infrastructure_failure": False,
                "terminal_infrastructure": False,
                "blocking_infrastructure": False,
                "terminal_reason": None,
                "exit_codes": [],
                "recovery_evidence": None,
                "test_inventory": None,
                "coverage_error": None,
            }
            if record is None:
                entry["error"] = "missing shard record"
                entry["failure_class"] = "infrastructure"
                entry["infrastructure_failure"] = True
                entries.append(entry)
                continue
            if not kept:
                entry["error"] = f"filtered:{drop_reason or 'unknown'}"
                entry["failure_class"] = "filtered"
                entries.append(entry)
                continue
            if kind == "math":
                correct = grade_row.get("correct")
                entry["passed"] = bool(correct)
                entry["verified"] = bool(correct)
                entry["timing_s"] = float(grade_row.get("timing_s") or 0.0)
                entry["failure_class"] = (
                    "verified" if correct else "answer_failure"
                )
                entries.append(entry)
            elif kind == "code":
                if problem_id in excluded:
                    entry["error"] = (
                        "interpretation excluded: " + str(excluded[problem_id])
                    )
                    entry["failure_class"] = "interpretation_excluded"
                    entries.append(entry)
                    continue
                pool_tests = list(problems.get(problem_id, {}).get("tests") or [])
                if not pool_tests:
                    tests = []
                elif execution_tests is not None:
                    row = execution_tests.get(problem_id)
                    if row is None:
                        entry["error"] = "no authenticated execution tests"
                        entry["failure_class"] = "infrastructure"
                        entry["infrastructure_failure"] = True
                        entries.append(entry)
                        continue
                    tests = list(row.get("tests") or [])
                else:
                    tests = pool_tests
                if tests:
                    inventory = dict(
                        (test_inventory or {}).get(problem_id)
                        or {
                            "n_tests": len(tests),
                            "tests_sha256": _problem_tests_sha256(tests),
                        }
                    )
                    if int(inventory.get("n_tests") or -1) != len(tests):
                        entry["error"] = (
                            "frozen test inventory count disagrees with the "
                            "authenticated execution tests"
                        )
                        entry["failure_class"] = "coverage_error"
                        entry["infrastructure_failure"] = True
                        entry["coverage_error"] = entry["error"]
                        entries.append(entry)
                        continue
                    entry["test_inventory"] = inventory
                code = extract_code(record)
                if not tests:
                    # The freeze records no-test exclusions; the request is
                    # accounted for but can never be a verified pass.  It is
                    # not a Docker infrastructure failure.
                    entry["error"] = "no frozen tests"
                    entry["failure_class"] = "no_frozen_tests"
                    entries.append(entry)
                elif not code:
                    entry["error"] = "no fenced code block"
                    entry["failure_class"] = "answer_failure"
                    entries.append(entry)
                else:
                    code_jobs.append((len(entries), code, tests))
                    entries.append(entry)
            else:
                # Chat is never a verified target but remains in the
                # per-request accounting.
                entry["failure_class"] = "not_applicable"
                entries.append(entry)

    jobs_by_index = {index: (code, tests) for index, code, tests in code_jobs}
    results: dict[int, Any] = {}
    if ledger_path is not None:
        ledger_file = Path(ledger_path)
        ledger_stage_names = _run_spec(
            getattr(authorization, "run_version", "v3")
        ).stages
        ledger_records, _header = _ensure_attempt_ledger(
            ledger_file,
            ledger_identity or {},
            audit_fields=ledger_audit,
            registry=ledger_registry,
            stage_names=ledger_stage_names,
        )
        # Any persisted verifier interface/programming error blocks the whole
        # run before planning: no retry, no new attempt start and no terminal
        # disposition may follow it (neither on a fresh pass nor on a resume).
        _refuse_verifier_interface_error_block(
            ledger_records,
            context=f"{ledger_file}: durable verifier interface error",
        )
        if not resume and any(
            record.get("kind")
            in {LEDGER_ATTEMPT_START, LEDGER_ATTEMPT_RESULT, LEDGER_ATTEMPT_INTERRUPTED}
            for record in ledger_records
        ):
            raise ShortfallV3RunError(
                "the attempt ledger already contains attempts; only an "
                "authorized resume may continue this run"
            )
        ledger_entries = _ledger_entries_by_test(ledger_records)

        def register(record: Mapping[str, Any]) -> None:
            ledger_entries.setdefault(
                (str(record["request_id"]), int(record["test_index"])), []
            ).append(dict(record))
            _append_attempt_ledger(ledger_file, record)

        def latest_result(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
            found = [
                entry
                for entry in entries
                if entry.get("kind")
                in {LEDGER_ATTEMPT_RESULT, LEDGER_ATTEMPT_INTERRUPTED}
            ]
            return found[-1] if found else None

        def attempt_evidence(key: tuple[str, int]) -> list[dict[str, Any]]:
            cleanup_recoveries: dict[int, dict[str, Any]] = {}
            for record in ledger_entries.get(key, []):
                if record.get("kind") == LEDGER_CLEANUP_RECOVERED:
                    cleanup_recoveries[int(record.get("attempt") or 1)] = dict(
                        record
                    )
            evidence: list[dict[str, Any]] = []
            for record in ledger_entries.get(key, []):
                if record.get("kind") not in {
                    LEDGER_ATTEMPT_RESULT,
                    LEDGER_ATTEMPT_INTERRUPTED,
                }:
                    continue
                item = dict(record.get("evidence") or {})
                item.setdefault("status", record.get("status"))
                item.setdefault(
                    "infrastructure_failure", record.get("infrastructure_failure")
                )
                item["attempt"] = int(record.get("attempt") or 1)
                item["interrupted"] = record.get("kind") == LEDGER_ATTEMPT_INTERRUPTED
                if record.get("recovery_evidence") is not None:
                    item["recovery_evidence"] = dict(
                        record.get("recovery_evidence")
                    )
                if item["interrupted"]:
                    cleanup = cleanup_recoveries.get(item["attempt"])
                    if cleanup is not None and _cleanup_recovery_valid(
                        cleanup,
                        request_id=key[0],
                        test_index=key[1],
                        attempt=item["attempt"],
                        container_name=item.get("container_name"),
                    ):
                        # The interrupted record keeps its infrastructure
                        # outcome and its own cleanup_verified=False.  Only this
                        # separate, validated recovery record supplies the
                        # per-attempt cleanup acceptance.
                        item["cleanup_recovery"] = cleanup
                evidence.append(item)
            evidence.sort(key=lambda item: int(item.get("attempt") or 1))
            return evidence

        def cleanup_recovered_for(
            key: tuple[str, int],
            attempt: int,
            container_name: str | None,
        ) -> Mapping[str, Any] | None:
            for record in ledger_entries.get(key, []):
                if record.get("kind") != LEDGER_CLEANUP_RECOVERED:
                    continue
                if _cleanup_recovery_valid(
                    record,
                    request_id=key[0],
                    test_index=key[1],
                    attempt=attempt,
                    container_name=container_name,
                ):
                    return record
            return None

        def start_for_attempt(
            key: tuple[str, int], attempt: int
        ) -> Mapping[str, Any] | None:
            return next(
                (
                    record
                    for record in ledger_entries.get(key, [])
                    if record.get("kind") == LEDGER_ATTEMPT_START
                    and int(record.get("attempt") or 1) == int(attempt)
                ),
                None,
            )

        def append_cleanup_recovery(
            request_id: str,
            problem_id: str,
            key: tuple[str, int],
            start: Mapping[str, Any],
            recovery: Any,
            *,
            checked: bool,
        ) -> dict[str, Any] | None:
            """Append cleanup recovery evidence for one interrupted attempt.

            A valid, container-bound check appends ``cleanup-recovered``.  A
            check that ran but did not verify the recorded container appends
            ``cleanup-recovery-failed`` and leaves the attempt pending.  The
            interrupted record is never rewritten and no attempt is added.
            """

            attempt_no = int(start.get("attempt") or 1)
            container_name = start.get("container_name")
            if not container_name:
                return None
            existing = cleanup_recovered_for(key, attempt_no, str(container_name))
            if existing is not None:
                return dict(existing.get("recovery_evidence") or {})
            evidence = _normalise_recovery_evidence(recovery)
            record = _cleanup_recovery_record(
                request_id=request_id,
                problem_id=problem_id,
                test_index=key[1],
                attempt=attempt_no,
                container_name=str(container_name),
                attempt_token=start.get("attempt_token"),
                recovery=evidence,
            )
            if record is not None:
                register(record)
                return evidence
            if checked:
                raw = recovery if isinstance(recovery, Mapping) else None
                register(
                    {
                        "kind": LEDGER_CLEANUP_RECOVERY_FAILED,
                        "request_id": str(request_id),
                        "problem_id": str(problem_id),
                        "test_index": int(key[1]),
                        "attempt": attempt_no,
                        "container_name": str(container_name),
                        "attempt_token": str(start.get("attempt_token") or ""),
                        "cleanup_verified": False,
                        "recovery_evidence": dict(raw) if raw is not None else None,
                        "failed_utc": _utc_now(),
                        "detail": (
                            "cleanup recovery did not verify the recorded "
                            "container; the attempt remains pending"
                        ),
                    }
                )
            return None

        def recover_pending_cleanup(
            request_id: str, problem_id: str, key: tuple[str, int]
        ) -> None:
            """Re-check every interrupted attempt still pending cleanup.

            Recovery is append-only and idempotent: once a valid
            ``cleanup-recovered`` record exists the attempt is skipped.  A
            crash between the interrupted append and the recovery append is
            therefore recovered by the next pass, as is an earlier failed
            check.
            """

            interrupted = [
                record
                for record in ledger_entries.get(key, [])
                if record.get("kind") == LEDGER_ATTEMPT_INTERRUPTED
            ]
            seen: set[int] = set()
            for record in interrupted:
                attempt_no = int(record.get("attempt") or 1)
                if attempt_no in seen:
                    continue
                seen.add(attempt_no)
                start = start_for_attempt(key, attempt_no)
                if start is None:
                    continue
                container_name = start.get("container_name")
                if not container_name:
                    continue
                if cleanup_recovered_for(
                    key, attempt_no, str(container_name)
                ) is not None:
                    continue
                checked = recovery_check is not None
                raw = (
                    _invoke_recovery_check(
                        recovery_check,
                        failure_classes=["cleanup"],
                        image=image,
                        container_name=str(container_name),
                    )
                    if checked
                    else None
                )
                evidence = append_cleanup_recovery(
                    request_id,
                    problem_id,
                    key,
                    start,
                    raw,
                    checked=checked,
                )
                if evidence is not None and entry["recovery_evidence"] is None:
                    entry["recovery_evidence"] = evidence

        def interrupted_row(
            key: tuple[str, int], start: Mapping[str, Any]
        ) -> dict[str, Any]:
            detail = "attempt interrupted by host crash; no result was persisted"
            return {
                "index": int(key[1]),
                "status": "infrastructure_error",
                "passed": False,
                "infrastructure_failure": True,
                "exit_code": None,
                "timed_out": False,
                "duration_s": 0.0,
                "error": detail,
                "stderr": detail,
                "lifecycle_class": start.get("lifecycle_class"),
                "lifecycle_valid": False,
                "cleanup_verified": False,
                "container_name": start.get("container_name"),
                "lifecycle_attempt": start.get("attempt_token"),
            }

        def record_interrupted(
            request_id: str,
            problem_id: str,
            key: tuple[str, int],
            start: Mapping[str, Any],
            recovery: Any,
            *,
            checked: bool = True,
        ) -> None:
            row = interrupted_row(key, start)
            attempt_no = int(start.get("attempt") or 1)
            evidence = _normalise_recovery_evidence(recovery)
            register(
                _ledger_result_record(
                    request_id=request_id,
                    problem_id=problem_id,
                    test_index=key[1],
                    attempt=attempt_no,
                    evidence=_attempt_summary(row),
                    sandboxed=True,
                    kind=LEDGER_ATTEMPT_INTERRUPTED,
                    recovery_evidence=evidence,
                )
            )
            # Append separate, container-bound cleanup evidence; never rewrite
            # the interrupted record above.  This is the only route by which an
            # interrupted attempt's cleanup may become accepted.  The recovery
            # record and the interrupted record are separate appends, so a
            # crash between them is recovered by the next planning pass.
            append_cleanup_recovery(
                request_id,
                problem_id,
                key,
                start,
                recovery,
                checked=checked,
            )

        def start_attempt(
            request_id: str,
            problem_id: str,
            key: tuple[str, int],
            attempt: int,
            code: str,
            test: Mapping[str, Any],
            *,
            container_name: str,
            attempt_token: str,
        ) -> None:
            # The actual container identity is committed *before* dispatch so
            # a recovery check after a crash inspects the real container.
            register(
                {
                    "kind": LEDGER_ATTEMPT_START,
                    "request_id": request_id,
                    "problem_id": problem_id,
                    "test_index": int(key[1]),
                    "attempt": int(attempt),
                    "request_sha256": _sha256_text(code),
                    "test_sha256": _test_identity_sha256(test),
                    "container_name": str(container_name),
                    "attempt_token": str(attempt_token),
                    "started_utc": _utc_now(),
                    "freeze_manifest_sha256": str(
                        freeze.get("manifest_sha256") or ""
                    ),
                    "freeze_test_manifest_sha256": str(
                        (freeze.get("tests") or {}).get("manifest_sha256") or ""
                    ),
                }
            )

        def plan(
            request_id: str,
            code: str,
            tests: Sequence[Mapping[str, Any]],
        ) -> tuple[
            list[dict[str, Any]],
            dict[int, Mapping[str, Any]],
            dict[int, tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]],
        ]:
            pending: list[dict[str, Any]] = []
            reuse: dict[int, Mapping[str, Any]] = {}
            blocked: dict[
                int, tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]
            ] = {}
            for position, test in enumerate(tests):
                key = (request_id, position)
                entries = ledger_entries.get(key, [])
                for record in entries:
                    if record.get("kind") != LEDGER_ATTEMPT_START:
                        continue
                    if record.get("request_sha256") != _sha256_text(code):
                        raise ValueError(
                            "candidate bytes changed since the ledger attempt "
                            f"for {request_id} test {position}"
                        )
                    if record.get("test_sha256") != _test_identity_sha256(test):
                        raise ValueError(
                            "test bytes changed since the ledger attempt for "
                            f"{request_id} test {position}"
                        )
                result, next_attempt, in_flight, _used = _ledger_plan(entries)
                if result is not None:
                    reuse[position] = result
                elif (
                    next_attempt is not None
                    and next_attempt <= 1 + int(max_infrastructure_retries)
                ):
                    pending.append(
                        {
                            "position": position,
                            "attempt": next_attempt,
                            "in_flight": in_flight,
                            "test": test,
                        }
                    )
                else:
                    blocked[position] = (in_flight, entries)
            return pending, reuse, blocked

        def graded_payload(result: Any) -> dict[str, Any]:
            return result.to_dict() if hasattr(result, "to_dict") else dict(result)

        def failed_payload(exc: Exception) -> dict[str, Any]:
            return {
                "passed": False,
                "sandboxed": False,
                "n_tests": 0,
                "passed_tests": 0,
                "tests": [],
                "error": f"grader failure: {type(exc).__name__}: {exc}",
                "timed_out": False,
                "duration_s": 0.0,
            }

        for index, code, tests in code_jobs:
            entry = entries[index]
            request_id = str(entry["request_id"])
            problem_id = str(entry["problem_id"])
            rounds = 0
            while True:
                # Every planning pass first re-checks the actual recorded
                # container of any interrupted attempt that is still pending
                # cleanup recovery.  This is what recovers a crash between the
                # interrupted append and the cleanup-recovered append, and an
                # earlier failed recovery check, without rewriting the
                # interruption, adding an attempt or regrading a result.
                for position in range(len(tests)):
                    recover_pending_cleanup(
                        request_id, problem_id, (request_id, position)
                    )
                pending, _reuse, blocked = plan(request_id, code, tests)
                for position, (in_flight, entries_for) in blocked.items():
                    if in_flight is None:
                        continue
                    attempt_no = int(in_flight.get("attempt") or 1)
                    has_result = any(
                        entry.get("kind")
                        in {LEDGER_ATTEMPT_RESULT, LEDGER_ATTEMPT_INTERRUPTED}
                        and int(entry.get("attempt") or 1) == attempt_no
                        for entry in entries_for
                    )
                    if not has_result:
                        key = (request_id, position)
                        row = interrupted_row(key, in_flight)
                        container_name = row.get("container_name") or in_flight.get(
                            "container_name"
                        )
                        classes = _recovery_failure_classes(
                            [row], in_flight=True
                        )
                        # Even an interrupted retry (no third attempt allowed)
                        # still recovers its own recorded container cleanup.
                        checked = bool(
                            recovery_check is not None
                            and not ("cleanup" in classes and not container_name)
                        )
                        raw = (
                            _invoke_recovery_check(
                                recovery_check,
                                failure_classes=classes,
                                image=image,
                                container_name=container_name,
                            )
                            if checked
                            else None
                        )
                        recovery = _normalise_recovery_evidence(raw)
                        if (
                            recovery is not None
                            and entry["recovery_evidence"] is None
                        ):
                            entry["recovery_evidence"] = recovery
                        record_interrupted(
                            request_id,
                            problem_id,
                            key,
                            in_flight,
                            raw,
                            checked=checked,
                        )
                if not pending:
                    break
                retry_items = [
                    item
                    for item in pending
                    if item["attempt"] > 1 or item["in_flight"] is not None
                ]
                item_recovery: dict[int, dict[str, Any] | None] = {}
                item_recovery_raw: dict[int, Any] = {}
                item_recovery_checked: dict[int, bool] = {}
                if retry_items:
                    for item in retry_items:
                        key = (request_id, item["position"])
                        if item["in_flight"] is not None:
                            row = interrupted_row(key, item["in_flight"])
                            start = item["in_flight"]
                        else:
                            prior = latest_result(ledger_entries.get(key, []))
                            row = (
                                _ledger_row(prior, item["position"])
                                if prior is not None
                                else {}
                            )
                            start = next(
                                (
                                    record
                                    for record in ledger_entries.get(key, [])
                                    if record.get("kind") == LEDGER_ATTEMPT_START
                                    and int(record.get("attempt") or 1)
                                    == int(item["attempt"] or 1) - 1
                                ),
                                None,
                            )
                        container_name = row.get("container_name") or (
                            (start or {}).get("container_name")
                        )
                        classes = _recovery_failure_classes(
                            [row],
                            in_flight=item["in_flight"] is not None,
                        )
                        # Recovery is per retry item and bound to that item's
                        # own container identity.  A cleanup-class recovery
                        # without the actual container name is refused rather
                        # than reusing another item's container.
                        if "cleanup" in classes and not container_name:
                            item_recovery[item["position"]] = None
                            item_recovery_raw[item["position"]] = None
                            item_recovery_checked[item["position"]] = False
                            continue
                        raw = _invoke_recovery_check(
                            recovery_check,
                            failure_classes=classes,
                            image=image,
                            container_name=container_name,
                        )
                        item_recovery_raw[item["position"]] = raw
                        item_recovery_checked[item["position"]] = bool(
                            recovery_check is not None
                        )
                        item_recovery[item["position"]] = (
                            _normalise_recovery_evidence(raw)
                        )
                        if (
                            item_recovery[item["position"]] is not None
                            and entry["recovery_evidence"] is None
                        ):
                            entry["recovery_evidence"] = item_recovery[
                                item["position"]
                            ]
                allowed: list[dict[str, Any]] = []
                for item in pending:
                    if item["attempt"] > 1 or item["in_flight"] is not None:
                        key = (request_id, item["position"])
                        recovery = item_recovery.get(item["position"])
                        if recovery is None:
                            if item["in_flight"] is not None:
                                record_interrupted(
                                    request_id,
                                    problem_id,
                                    key,
                                    item["in_flight"],
                                    item_recovery_raw.get(item["position"]),
                                    checked=item_recovery_checked.get(
                                        item["position"], False
                                    ),
                                )
                            continue
                        if item["in_flight"] is not None:
                            record_interrupted(
                                request_id,
                                problem_id,
                                key,
                                item["in_flight"],
                                item_recovery_raw.get(item["position"]),
                                checked=item_recovery_checked.get(
                                    item["position"], False
                                ),
                            )
                            # The in-flight attempt stays pending cleanup
                            # recovery unless the recorded container now has a
                            # valid, separate cleanup-recovered record.  A
                            # non-cleanup or unbound recovery reply never
                            # unlocks the retry.
                            in_flight_attempt = int(
                                item["in_flight"].get("attempt") or 1
                            )
                            in_flight_container = item["in_flight"].get(
                                "container_name"
                            )
                            if in_flight_container and (
                                cleanup_recovered_for(
                                    key,
                                    in_flight_attempt,
                                    str(in_flight_container),
                                )
                                is None
                            ):
                                continue
                        else:
                            # A prior interrupted result stays pending until its
                            # actual recorded container has a valid, separate
                            # cleanup-recovered record.  Recovery evidence for
                            # an unrelated container never unlocks the retry.
                            prior = latest_result(ledger_entries.get(key, []))
                            if (
                                prior is not None
                                and prior.get("kind") == LEDGER_ATTEMPT_INTERRUPTED
                            ):
                                prior_attempt = int(prior.get("attempt") or 1)
                                prior_start = start_for_attempt(key, prior_attempt)
                                prior_container = (
                                    (prior_start or {}).get("container_name")
                                    or (prior.get("evidence") or {}).get(
                                        "container_name"
                                    )
                                )
                                if prior_container and (
                                    cleanup_recovered_for(
                                        key,
                                        prior_attempt,
                                        str(prior_container),
                                    )
                                    is None
                                ):
                                    continue
                        item["recovery_evidence"] = recovery
                    allowed.append(item)
                if not allowed:
                    # Keep the interrupted attempt pending; the next planning
                    # pass re-checks its recorded container.  The round guard
                    # bounds repeated failed recovery checks.
                    rounds += 1
                    if rounds > int(max_infrastructure_retries) + 2:
                        break
                    continue
                # Per-test streaming: commit this test's start immediately
                # before dispatching exactly this test, and its result
                # immediately after that test completes.  A crash can therefore
                # lose only the in-flight test, never already completed ones.
                for item in allowed:
                    container_name = f"gemmaloop-{uuid.uuid4().hex[:16]}"
                    attempt_token = uuid.uuid4().hex
                    start_attempt(
                        request_id,
                        problem_id,
                        (request_id, item["position"]),
                        int(item["attempt"]),
                        code,
                        item["test"],
                        container_name=container_name,
                        attempt_token=attempt_token,
                    )
                    try:
                        raw = _call_grade_fn(
                            grade_fn,
                            code,
                            [item["test"]],
                            timeout,
                            container_name=container_name,
                            attempt=attempt_token,
                        )
                        payload = graded_payload(raw)
                    except Exception as exc:  # noqa: BLE001 - infrastructure
                        if _is_verifier_programming_error(exc):
                            # Verifier interface/programming errors are not
                            # candidate evidence.  Persist a blocking error and
                            # stop dispatching immediately; this never becomes
                            # an attempt result and never consumes a retry.
                            register(
                                {
                                    "kind": LEDGER_VERIFIER_INTERFACE_ERROR,
                                    "request_id": request_id,
                                    "problem_id": problem_id,
                                    "test_index": int(item["position"]),
                                    "attempt": int(item["attempt"]),
                                    "container_name": container_name,
                                    "attempt_token": attempt_token,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "failed_utc": _utc_now(),
                                    "blocks_dispatch": True,
                                }
                            )
                            raise ShortfallVerifierInterfaceError(
                                "verifier interface/programming error blocked "
                                f"official dispatch at {request_id} test "
                                f"{item['position']} attempt {item['attempt']}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        payload = failed_payload(exc)
                    sandboxed_attempt = bool(payload.get("sandboxed"))
                    rows = _per_test_accounting(payload)
                    if rows:
                        row = dict(rows[0])
                        row["index"] = int(item["position"])
                        # Each attempt records its own container identity and
                        # lifecycle attempt token, independent of whether the
                        # grader supplied them.
                        row.setdefault("container_name", container_name)
                        row.setdefault("lifecycle_attempt", attempt_token)
                    else:
                        detail = str(
                            payload.get("error")
                            or "grader returned no row for this test"
                        )
                        row = {
                            "index": int(item["position"]),
                            "status": "infrastructure_error",
                            "passed": False,
                            "infrastructure_failure": True,
                            "exit_code": None,
                            "timed_out": False,
                            "duration_s": 0.0,
                            "error": detail,
                            "stderr": _truncate_stderr(detail),
                            "lifecycle_class": None,
                            "lifecycle_valid": False,
                            "cleanup_verified": False,
                            "container_name": container_name,
                            "lifecycle_attempt": attempt_token,
                            "partial_result": True,
                        }
                    register(
                        _ledger_result_record(
                            request_id=request_id,
                            problem_id=problem_id,
                            test_index=int(item["position"]),
                            attempt=int(item["attempt"]),
                            evidence=_attempt_summary(row),
                            sandboxed=sandboxed_attempt,
                            recovery_evidence=item.get("recovery_evidence"),
                        )
                    )
                rounds += 1
                if rounds > int(max_infrastructure_retries) + 2:
                    break

            full_rows: list[dict[str, Any]] = []
            sandboxed_flags: list[bool] = []
            all_row_errors: list[str] = []
            for position in range(len(tests)):
                key = (request_id, position)
                entries_for = ledger_entries.get(key, [])
                result, _next_attempt, in_flight, _used = _ledger_plan(entries_for)
                if result is not None:
                    row = _ledger_row(result, position)
                    row["interrupted"] = (
                        result.get("kind") == LEDGER_ATTEMPT_INTERRUPTED
                    )
                    row["attempt"] = int(result.get("attempt") or 1)
                    sandboxed_flags.append(bool(result.get("sandboxed", True)))
                else:
                    prior = latest_result(entries_for)
                    if prior is not None:
                        row = _ledger_row(prior, position)
                        row["interrupted"] = (
                            prior.get("kind") == LEDGER_ATTEMPT_INTERRUPTED
                        )
                        row["attempt"] = int(prior.get("attempt") or 1)
                        sandboxed_flags.append(bool(prior.get("sandboxed", True)))
                    elif in_flight is not None:
                        row = interrupted_row(key, in_flight)
                        row["interrupted"] = True
                        row["attempt"] = int(in_flight.get("attempt") or 1)
                        sandboxed_flags.append(True)
                    else:
                        detail = "no persisted attempt result"
                        row = {
                            "index": position,
                            "status": "infrastructure_error",
                            "passed": False,
                            "infrastructure_failure": True,
                            "exit_code": None,
                            "timed_out": False,
                            "duration_s": 0.0,
                            "error": detail,
                            "stderr": detail,
                            "lifecycle_class": None,
                            "lifecycle_valid": False,
                            "cleanup_verified": False,
                        }
                        sandboxed_flags.append(True)
                if row.get("error"):
                    all_row_errors.append(str(row["error"]))
                if row.get("partial_result"):
                    # A partial grader reply stays uncovered so the count
                    # refuses it rather than silently treating it as complete.
                    continue
                row["index"] = position
                row["attempts"] = attempt_evidence(key)
                if row.get("interrupted"):
                    current = next(
                        (
                            attempt
                            for attempt in row["attempts"]
                            if int(attempt.get("attempt") or 1)
                            == int(row.get("attempt") or 1)
                        ),
                        None,
                    )
                    if current is not None and _attempt_cleanup_recovery_valid(
                        current, request_id=request_id, test_index=position
                    ):
                        # The interrupted record itself stays unverified; the
                        # final row records the accepted cleanup supplied by
                        # its separate, container-bound recovery record.
                        row["cleanup_verified"] = True
                full_rows.append(row)
            results[index] = {
                "passed": bool(full_rows)
                and all(row.get("passed") for row in full_rows),
                "sandboxed": bool(sandboxed_flags) and all(sandboxed_flags),
                "n_tests": len(full_rows),
                "passed_tests": sum(1 for row in full_rows if row.get("passed")),
                "tests": full_rows,
                "error": (all_row_errors[0] if all_row_errors else None),
                "timed_out": any(row.get("timed_out") for row in full_rows),
                "duration_s": sum(
                    float(row.get("duration_s") or 0.0) for row in full_rows
                ),
            }
    else:

        def run_job(job: tuple[int, str, list[dict[str, Any]]]) -> tuple[int, Any]:
            _index, code, tests = job
            try:
                return _index, grade_fn(code, tests, timeout)
            except Exception as exc:  # noqa: BLE001 - an infrastructure failure
                if _is_verifier_programming_error(exc):
                    raise ShortfallVerifierInterfaceError(
                        "verifier interface/programming error blocked official "
                        f"dispatch: {type(exc).__name__}: {exc}"
                    ) from exc
                return _index, {
                    "passed": False,
                    "sandboxed": False,
                    "n_tests": 0,
                    "passed_tests": 0,
                    "tests": [],
                    "error": f"grader failure: {type(exc).__name__}: {exc}",
                    "timed_out": False,
                    "duration_s": 0.0,
                }

        if code_jobs:
            if int(workers) > 1:
                with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                    for index, result in pool.map(run_job, code_jobs):
                        results[index] = result
            else:
                for job in code_jobs:
                    index, result = run_job(job)
                    results[index] = result

    for index, result in results.items():
        entry = entries[index]
        request_id = str(entry["request_id"])
        code, tests = jobs_by_index[index]
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        sandboxed = bool(payload.get("sandboxed"))
        per_test = _per_test_accounting(payload)
        tests_by_index = {
            int(test.get("index", position)): test
            for position, test in enumerate(tests)
        }
        attempts_by_index: dict[int, list[dict[str, Any]]] = {}
        for row in per_test:
            existing_attempts = list(row.get("attempts") or [])
            attempts_by_index[int(row["index"])] = (
                existing_attempts or [_attempt_summary(row)]
            )

        # Precommitted retry policy: at most one identical-configuration retry
        # per infrastructure-affected test, only after documented recovery.
        # The durable ledger path already performed and persisted retries, so
        # it must never retry a second time here.
        infrastructure_indices = [
            int(row["index"]) for row in per_test if row["infrastructure_failure"]
        ]
        recovery_evidence = None
        if (
            ledger_path is None
            and infrastructure_indices
            and int(max_infrastructure_retries) > 0
        ):
            if recovery_check is not None:
                recovery_evidence = _normalise_recovery_evidence(
                    _invoke_recovery_check(
                        recovery_check,
                        failure_classes=_recovery_failure_classes(per_test),
                        image=image,
                        container_name=next(
                            (
                                row.get("container_name")
                                for row in per_test
                                if row.get("container_name")
                            ),
                            None,
                        ),
                    )
                )
        retry_allowed = bool(
            ledger_path is None
            and infrastructure_indices
            and int(max_infrastructure_retries) > 0
            and recovery_evidence is not None
        )
        if ledger_path is None:
            entry["recovery_evidence"] = recovery_evidence
        if retry_allowed:
            for test_index in infrastructure_indices:
                original_test = tests_by_index.get(test_index)
                if original_test is None:
                    continue
                try:
                    retry_result = grade_fn(code, [original_test], timeout)
                    retry_payload = (
                        retry_result.to_dict()
                        if hasattr(retry_result, "to_dict")
                        else dict(retry_result)
                    )
                except Exception as exc:  # noqa: BLE001 - still infrastructure
                    retry_payload = {
                        "passed": False,
                        "sandboxed": False,
                        "n_tests": 0,
                        "passed_tests": 0,
                        "tests": [],
                        "error": f"grader failure: {type(exc).__name__}: {exc}",
                        "timed_out": False,
                        "duration_s": 0.0,
                    }
                retry_rows = _per_test_accounting(retry_payload)
                if retry_rows:
                    retry_row = retry_rows[0]
                    retry_row["index"] = test_index
                else:
                    retry_row = {
                        "index": test_index,
                        "status": "infrastructure_error",
                        "passed": False,
                        "exit_code": None,
                        "timed_out": False,
                        "duration_s": 0.0,
                        "error": retry_payload.get("error"),
                        "stderr": _truncate_stderr(retry_payload.get("error")),
                        "infrastructure_failure": True,
                        "lifecycle_class": None,
                        "attempts": [],
                    }
                attempts_by_index.setdefault(test_index, []).append(
                    _attempt_summary(retry_row)
                )
                retry_row["attempts"] = attempts_by_index[test_index]
                for position, row in enumerate(per_test):
                    if int(row["index"]) == test_index:
                        if not retry_row["infrastructure_failure"]:
                            # The first conclusive result controls.
                            per_test[position] = retry_row
                        else:
                            row["attempts"] = attempts_by_index[test_index]
                        break

        for row in per_test:
            if not row.get("attempts"):
                row["attempts"] = attempts_by_index.get(int(row["index"]), [])
        _apply_terminal_infrastructure(per_test, max_infrastructure_retries)
        entry["per_test"] = per_test
        entry["exit_codes"] = [row["exit_code"] for row in per_test]
        infrastructure = any(row["infrastructure_failure"] for row in per_test)
        if not sandboxed:
            infrastructure = True
        inventory = entry.get("test_inventory")
        coverage_error = None
        grader_failure = str(payload.get("error") or "").startswith("grader failure:")
        if not per_test and grader_failure:
            # The grader raised before producing any test row; the recorded
            # exception is an infrastructure failure, not a partial result.
            infrastructure = True
        elif isinstance(inventory, Mapping):
            coverage_error = _validate_per_test_coverage(
                per_test,
                expected_n=int(inventory.get("n_tests") or 0),
                expected_sha=str(inventory.get("tests_sha256") or ""),
                inventory=inventory,
                test_bytes=[
                    _canonical_test_bytes(test).decode("utf-8")
                    for test in tests
                ],
                request_id=request_id,
                enforcement_bound_s=_frozen_deadline_enforcement_bound(freeze),
            )
        else:
            coverage_error = "missing frozen test inventory"
        if coverage_error:
            entry["coverage_error"] = coverage_error
            entry["error"] = coverage_error
            entry["failure_class"] = "coverage_error"
            entry["infrastructure_failure"] = True
            entry["passed"] = False
            entry["verified"] = False
        verified = (
            coverage_error is None
            and bool(per_test)
            and not infrastructure
            and all(row["passed"] for row in per_test)
        )
        entry["passed"] = verified
        entry["verified"] = verified
        entry["sandboxed"] = sandboxed
        entry["tests_run"] = int(
            payload.get("n_tests")
            if payload.get("n_tests") is not None
            else len(per_test)
        )
        entry["timing_s"] = float(payload.get("duration_s") or 0.0)
        if verified:
            entry["error"] = None
            entry["failure_class"] = "verified"
        elif entry.get("coverage_error"):
            entry["error"] = entry["coverage_error"]
            entry["failure_class"] = "coverage_error"
            entry["infrastructure_failure"] = True
        elif infrastructure or (not per_test and not verified):
            entry["error"] = payload.get("error") or "infrastructure failure"
            entry["failure_class"] = "infrastructure"
            entry["infrastructure_failure"] = True
        elif any(row["status"] == "timeout" for row in per_test):
            entry["error"] = payload.get("error") or "timeout"
            entry["failure_class"] = "timeout"
        elif any(row["status"] == "candidate_error" for row in per_test):
            entry["error"] = payload.get("error") or "candidate error"
            entry["failure_class"] = "candidate_error"
        else:
            entry["error"] = payload.get("error") or "wrong answer"
            entry["failure_class"] = "answer_failure"
        terminal_rows = [
            row for row in per_test if row.get("terminal_infrastructure")
        ]
        nonterminal_rows = [
            row
            for row in per_test
            if row.get("infrastructure_failure")
            and not row.get("terminal_infrastructure")
        ]
        if entry.get("infrastructure_failure"):
            if (
                entry.get("coverage_error")
                or not per_test
                or not sandboxed
                or nonterminal_rows
            ):
                entry["blocking_infrastructure"] = True
            elif terminal_rows:
                entry["terminal_infrastructure"] = True
                entry["terminal_reason"] = str(
                    terminal_rows[0].get("terminal_reason")
                    or terminal_rows[0].get("error")
                    or "infrastructure unverified after the single retry"
                )
    return entries


def compute_counts(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Distinct verified math/code problems per split and the decision."""

    per_split: dict[str, Any] = {}
    for name in PILOT_SPLITS:
        rows = [entry for entry in entries if entry.get("split") == name]
        math_verified = {
            str(entry["problem_id"])
            for entry in rows
            if entry.get("domain") == "math" and entry.get("verified")
        }
        code_verified = {
            str(entry["problem_id"])
            for entry in rows
            if entry.get("domain") == "code" and entry.get("verified")
        }
        union = math_verified | code_verified
        per_split[name] = {
            "math_verified_problems": sorted(math_verified),
            "code_verified_problems": sorted(code_verified),
            "union_verified_problems": sorted(union),
            "math_count": len(math_verified),
            "code_count": len(code_verified),
            "union_count": len(union),
            "requests": len(rows),
            "kept_requests": sum(1 for entry in rows if entry.get("kept")),
            "kept_code_samples": sum(
                1
                for entry in rows
                if entry.get("domain") == "code" and entry.get("kept")
            ),
            "graded_code_samples": sum(
                1
                for entry in rows
                if entry.get("domain") == "code" and entry.get("kept")
            ),
            "passed_code_samples": sum(
                1
                for entry in rows
                if entry.get("domain") == "code" and entry.get("verified")
            ),
        }
    n_t0 = int(per_split["pilot_tfit"]["union_count"])
    n_s0 = int(per_split["pilot_selection"]["union_count"])
    required = decide_extension(n_t0, n_s0)
    return {
        "per_split": per_split,
        "nT0": n_t0,
        "nS0": n_s0,
        "pilot_p1": int(per_split["pilot_p1"]["union_count"]),
        "p1_descriptive": True,
        "p1_small_sample": int(per_split["pilot_p1"]["union_count"]) < P1_MIN,
        "p1_unevaluable": int(per_split["pilot_p1"]["union_count"]) == 0,
        "extension_required": bool(required),
        "extension_decision": (
            "extension_required" if required else "no_extension_original_only"
        ),
        "thresholds": {
            "pilot_tfit_min": TFIT_MIN,
            "pilot_selection_min": SELECTION_MIN,
            "pilot_p1_gates": False,
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)
    return _sha256_text(text)


def _resolve_freeze_input_dir(
    supplied: Path | str | None, recorded: Any, label: str
) -> Path:
    """Consume the freeze-recorded input directory or refuse a substitute."""

    recorded_path = Path(str(recorded or "")).resolve()
    if not recorded_path.is_dir():
        raise ValueError(f"verifier freeze {label} is missing: {recorded_path}")
    if supplied is None:
        return recorded_path
    supplied_path = Path(supplied).resolve()
    if supplied_path != recorded_path:
        raise ValueError(
            f"{label} {supplied_path} differs from the verified freeze path "
            f"{recorded_path}; the count consumes only frozen inputs"
        )
    return recorded_path


def count_verified(
    *,
    freeze_path: Path | str,
    interpretation_path: Path | str | None = None,
    pool_dir: Path | str | None = None,
    lite_dir: Path | str | None = None,
    topup_plan_path: Path | str,
    out_dir: Path | str,
    grade_fn: Callable[..., Any] | None = None,
    workers: int = 1,
    max_infrastructure_retries: int = 0,
    recovery_check: Callable[..., Mapping[str, Any] | bool] | None = None,
    exclusive: bool = False,
    authorization: V3RunAuthorization | None = None,
    resume: bool = False,
    ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    """Grade original kept code, write accounting and the decision summary.

    The input directories are taken from the verified freeze.  Supplying a
    different directory is refused.  The actual executing implementation and
    the authenticated interpretation manifest are verified before any grading;
    the precommitted top-up plan is required *before* any decision is made.
    Any unresolved infrastructure failure blocks the final decision and raises
    :class:`ShortfallCountError`.
    """

    freeze_path = Path(freeze_path)
    freeze = load_freeze_manifest(freeze_path)
    out_dir = Path(out_dir)
    if authorization is None:
        if exclusive:
            raise ShortfallV3RunError(
                "exclusive count requires the authorization of the claimed "
                "official v3 run; exclusive=True is not authorization"
            )
        raise ShortfallV3RunError(
            "count execution of original samples requires the claimed official "
            "v3 run token or an authorized resume, even when a grade_fn is "
            "supplied"
        )
    count_registry = _authorize_v3_stage(
        authorization,
        stage="count",
        out_dir=out_dir,
        lite_dir=(freeze.get("inputs") or {}).get("lite_dir") or "",
    )
    run_version = str(count_registry.get("run_version") or "v3")
    if resume:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        _require_new_dir(out_dir)
    exclusive_write = not resume
    inputs = freeze.get("inputs") or {}
    resolved_pool = _resolve_freeze_input_dir(
        pool_dir, inputs.get("pool_dir"), "pool_dir"
    )
    resolved_lite = _resolve_freeze_input_dir(
        lite_dir, inputs.get("lite_dir"), "lite_dir"
    )
    implementation = verify_count_implementation(freeze)
    if resume and grade_fn is None:
        # A resume must re-establish the frozen image digest and sandbox
        # configuration before reusing any durable attempt.
        from gemmaloop.stage0.grader import DockerSandbox

        pinned_image = str(
            ((freeze.get("verifier") or {}).get("docker_image") or {}).get(
                "pinned"
            )
            or ((freeze.get("verifier") or {}).get("docker_image") or {}).get("id")
            or LITE_IMAGE_TAG
        )
        DockerSandbox(
            image=pinned_image, expected_digest=pinned_image
        ).verify_image_digest()
    if interpretation_path is None:
        raise ValueError(
            "the official count requires the authenticated interpretation manifest"
        )
    interpretation_path = Path(interpretation_path)
    interpretation = load_interpretation_manifest(interpretation_path)
    test_rows = [
        json.loads(line)
        for line in (
            freeze_path.parent / freeze["tests"]["manifest_file"]
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    interpretation_summary = verify_interpretation_against_test_manifest(
        interpretation, test_rows, manifest_path=interpretation_path
    )
    recorded_interpretation = freeze.get("interpretation") or {}
    if recorded_interpretation:
        execution_entry = interpretation.get("execution_tests") or {}
        if (
            recorded_interpretation.get("manifest_sha256")
            != interpretation.get("manifest_sha256")
            or recorded_interpretation.get("execution_tests_sha256")
            != execution_entry.get("sha256")
        ):
            raise ValueError(
                "the count interpretation manifest does not match the freeze"
            )
    execution = execution_tests_by_problem(interpretation, interpretation_path)
    test_inventory = _test_inventory_from_rows(test_rows)
    excluded = excluded_problem_ids(interpretation)
    plan_path = Path(topup_plan_path)
    if not plan_path.is_file():
        raise FileNotFoundError(
            "the complete top-up plan must be written before the count/decision: "
            f"{plan_path}"
        )
    plan = validate_topup_plan(_read_json(plan_path), freeze=freeze)
    plan_sha = _sha256_file(plan_path)

    ledger_file = (
        Path(ledger_path)
        if ledger_path is not None
        else out_dir / ATTEMPT_LEDGER_NAME
    )
    claim_registry_sha256 = _registry_claim_digest(count_registry)
    ledger_identity = _ledger_identity(
        freeze,
        implementation,
        timeout_s=float((freeze.get("verifier") or {}).get("timeout_s", 0.0)),
        retry_policy=(freeze.get("verifier") or {}).get("retry"),
        run_token=str(authorization.run_token),
        claim_registry_sha256=claim_registry_sha256,
    )
    # The count-start registry hash is audit-only.  It is captured once in the
    # durable ledger header and never becomes a resume requirement: the
    # registry self hash necessarily advances as stages complete.
    ledger_audit = {
        "count_start_registry_digest": str(
            count_registry.get("registry_sha256") or ""
        ),
        "count_start_completed_stages": _registry_completed_stages(count_registry),
        "count_start_run_version": run_version,
    }

    entries = grade_original_requests(
        freeze,
        pool_dir=resolved_pool,
        lite_dir=resolved_lite,
        grade_fn=grade_fn,
        workers=workers,
        execution_tests=execution,
        test_inventory=test_inventory,
        interpretation_excluded=excluded,
        max_infrastructure_retries=max_infrastructure_retries,
        recovery_check=recovery_check,
        ledger_path=ledger_file,
        ledger_identity=ledger_identity,
        ledger_audit=ledger_audit,
        ledger_registry=count_registry,
        resume=resume,
        authorization=authorization,
    )
    ledger_header = next(
        record
        for record in _read_attempt_ledger(ledger_file)
        if record.get("kind") == LEDGER_HEADER
    )
    run_registry_identity = {
        "run_token_sha256": _sha256_text(str(authorization.run_token)),
        "claim_registry_sha256": claim_registry_sha256,
        "count_start_registry_digest": str(
            ledger_header.get("count_start_registry_digest") or ""
        ),
    }
    expected_request_ids = [
        str(request["request_id"])
        for name in PILOT_SPLITS
        for request in freeze["selected_inputs"]["splits"][name]["requests"]
    ]
    actual_request_ids = [str(entry["request_id"]) for entry in entries]
    if actual_request_ids != expected_request_ids:
        raise ValueError(
            "per-request accounting does not exactly cover the frozen selected "
            "request set"
        )
    counts = compute_counts(entries)
    infrastructure = [
        entry for entry in entries if entry.get("infrastructure_failure")
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_request_path = out_dir / COUNT_PER_REQUEST_NAME
    per_request_sha = _write_jsonl(per_request_path, entries)
    log_lines = []
    for entry in entries:
        log_lines.append(
            "split={split} request_id={request_id} domain={domain} kept={kept} "
            "passed={passed} verified={verified} tests_run={tests_run} "
            "failure_class={failure_class} timed_out={timed_out} "
            "exit_codes={exit_codes} error={error} timing_s={timing_s}".format(
                **{
                    **entry,
                    "error": entry.get("error") or "none",
                    "timed_out": any(
                        row.get("timed_out")
                        for row in entry.get("per_test") or []
                    ),
                    "exit_codes": ",".join(
                        str(value) for value in entry.get("exit_codes") or []
                    )
                    or "none",
                }
            )
        )
    log_path = out_dir / COUNT_LOG_NAME
    log_text = "\n".join(log_lines) + "\n"
    atomic_write_text(log_path, log_text)
    blocking_infrastructure = [
        entry for entry in entries if entry.get("blocking_infrastructure")
    ]
    terminal_infrastructure = [
        entry
        for entry in entries
        if entry.get("terminal_infrastructure")
        and not entry.get("blocking_infrastructure")
    ]
    if blocking_infrastructure:
        status = COUNT_STATUS_BLOCKED_INFRASTRUCTURE
    elif terminal_infrastructure:
        status = COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE
    else:
        status = COUNT_STATUS_COMPLETE
    per_test_rows = sum(len(entry.get("per_test") or []) for entry in entries)
    infrastructure_tests = sum(
        1
        for entry in entries
        for row in entry.get("per_test") or []
        if row.get("infrastructure_failure")
    )
    terminal_infrastructure_tests = [
        {
            "request_id": str(entry.get("request_id")),
            "problem_id": str(entry.get("problem_id")),
            "split": str(entry.get("split")),
            "domain": str(entry.get("domain")),
            "test_index": int(row.get("index", -1)),
            "status": str(row.get("status")),
            "terminal_status": str(row.get("terminal_status")),
            "reason": str(
                row.get("terminal_reason")
                or row.get("error")
                or "infrastructure unverified after the single retry"
            ),
            "attempts": len(row.get("attempts") or []),
        }
        for entry in entries
        for row in entry.get("per_test") or []
        if row.get("terminal_infrastructure")
    ]
    terminal_by_split = {
        name: sum(
            1
            for item in terminal_infrastructure_tests
            if item["split"] == name
        )
        for name in PILOT_SPLITS
    }
    terminal_by_domain = {
        domain: sum(
            1
            for item in terminal_infrastructure_tests
            if item["domain"] == domain
        )
        for domain in ("math", "code")
    }
    interpretation_excluded_requests = [
        entry
        for entry in entries
        if entry.get("failure_class") == "interpretation_excluded"
    ]
    interpretation_excluded_problems = sorted(
        {str(entry["problem_id"]) for entry in interpretation_excluded_requests}
    )
    fingerprint = _sha256_text(
        json.dumps(
            {
                "freeze_manifest_sha256": freeze["manifest_sha256"],
                "freeze_test_manifest_sha256": freeze["tests"]["manifest_sha256"],
                "selected_inputs_sha256": freeze["selected_inputs"][
                    "selected_inputs_sha256"
                ],
                "interpretation_manifest_sha256": interpretation["manifest_sha256"],
                "interpretation_execution_tests_sha256": (
                    interpretation.get("execution_tests") or {}
                ).get("sha256"),
                "topup_plan_sha256": plan_sha,
                "plan_sha256": plan["plan_sha256"],
                "pool_dir": str(resolved_pool),
                "lite_dir": str(resolved_lite),
                "implementation": implementation,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    recovery_evidence = [
        {
            "request_id": entry.get("request_id"),
            "evidence": entry.get("recovery_evidence"),
        }
        for entry in entries
        if entry.get("recovery_evidence")
    ]
    summary: dict[str, Any] = {
        "kind": "pilot_shortfall_official_count",
        "version": 3 if run_version == "v3" else "3r",
        "run_version": run_version,
        "run_registry": dict(run_registry_identity),
        "status": status,
        "created_utc": _utc_now(),
        "freeze_manifest": str(freeze_path.resolve()),
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "freeze_test_manifest_sha256": freeze["tests"]["manifest_sha256"],
        "selected_inputs_sha256": freeze["selected_inputs"]["selected_inputs_sha256"],
        "pool_dir": str(resolved_pool),
        "lite_dir": str(resolved_lite),
        "implementation": implementation,
        "run_fingerprint": fingerprint,
        "interpretation": {
            "manifest": str(interpretation_path.resolve()),
            "manifest_sha256": interpretation["manifest_sha256"],
            "execution_tests_sha256": (
                interpretation.get("execution_tests") or {}
            ).get("sha256"),
            "authenticated_problem_count": int(
                interpretation.get("authenticated_problem_count") or 0
            ),
            "excluded_problem_count": int(
                interpretation.get("excluded_problem_count") or 0
            ),
            "excluded_problem_ids": sorted(excluded),
            "excluded_problem_reasons": dict(excluded),
            "verified": interpretation_summary,
        },
        "topup_plan": {
            "path": str(plan_path.resolve()),
            "sha256": plan_sha,
            "plan_sha256": plan["plan_sha256"],
            "topup_run_id": plan["topup_run_id"],
            "request_count": plan["request_count"],
            "problem_count": plan["problem_count"],
            **{key: counts[key] for key in ("nT0", "nS0", "extension_required") if key in counts},
        },
        **counts,
        "infrastructure_failures": len(infrastructure),
        "blocking_infrastructure_failures": len(blocking_infrastructure),
        "terminal_infrastructure_failures": len(terminal_infrastructure),
        "terminal_infrastructure": {
            "count": len(terminal_infrastructure_tests),
            "request_count": len(terminal_infrastructure),
            "tests": terminal_infrastructure_tests,
            "request_ids": [
                str(entry.get("request_id"))
                for entry in terminal_infrastructure
            ],
            "reasons": sorted(
                {
                    item["reason"]
                    for item in terminal_infrastructure_tests
                }
            ),
            "per_split": terminal_by_split,
            "per_domain": terminal_by_domain,
        },
        "pending_infrastructure_requests": [
            str(entry.get("request_id"))
            for entry in blocking_infrastructure
        ],
        "infrastructure_recovery": recovery_evidence,
        "infrastructure_request_ids": [
            str(entry.get("request_id")) for entry in infrastructure
        ],
        "expected_requests": {
            "count": len(expected_request_ids),
            "code_request_count": sum(
                1 for entry in entries if entry.get("domain") == "code"
            ),
            "request_ids_sha256": _sha256_text(
                "\n".join(sorted(expected_request_ids))
            ),
            "selected_inputs_sha256": freeze["selected_inputs"][
                "selected_inputs_sha256"
            ],
        },
        "accounting": {
            "requests": len(entries),
            "kept": sum(1 for entry in entries if entry.get("kept")),
            "code_requests": sum(
                1 for entry in entries if entry.get("domain") == "code"
            ),
            "code_requests_kept": sum(
                1
                for entry in entries
                if entry.get("domain") == "code" and entry.get("kept")
            ),
            "code_requests_graded": sum(
                1
                for entry in entries
                if entry.get("domain") == "code" and int(entry.get("tests_run") or 0) > 0
            ),
            "code_requests_passed": sum(
                1
                for entry in entries
                if entry.get("domain") == "code" and entry.get("verified")
            ),
            "errors": sum(1 for entry in entries if entry.get("error")),
            "infrastructure_failures": len(infrastructure),
            "blocking_infrastructure_failures": len(blocking_infrastructure),
            "terminal_infrastructure_failures": len(terminal_infrastructure),
            "infrastructure_tests": infrastructure_tests,
            "terminal_infrastructure_tests": len(
                terminal_infrastructure_tests
            ),
            "per_test_rows": per_test_rows,
            "timeouts": sum(
                1
                for entry in entries
                if entry.get("failure_class") == "timeout"
            ),
            "answer_failures": sum(
                1
                for entry in entries
                if entry.get("failure_class") == "answer_failure"
            ),
            "interpretation_excluded_requests": len(
                interpretation_excluded_requests
            ),
            "interpretation_excluded_problems": len(
                interpretation_excluded_problems
            ),
        },
        "artifacts": {
            COUNT_PER_REQUEST_NAME: {"sha256": per_request_sha, "rows": len(entries)},
            COUNT_LOG_NAME: {"sha256": _sha256_text(log_text), "rows": len(log_lines)},
            INTERPRETATION_MANIFEST_NAME: {
                "sha256": _sha256_file(interpretation_path),
            },
            EXECUTION_TESTS_NAME: {
                "sha256": _sha256_file(
                    interpretation_path.parent
                    / str(
                        (interpretation.get("execution_tests") or {}).get("file")
                        or EXECUTION_TESTS_NAME
                    )
                ),
            },
            ATTEMPT_LEDGER_NAME: {"sha256": _sha256_file(ledger_file)},
        },
        "attempt_ledger": {
            "path": str(ledger_file.resolve()),
            "sha256": _sha256_file(ledger_file),
        },
    }
    summary_path = out_dir / COUNT_SUMMARY_NAME
    if exclusive_write:
        _write_json_exclusive(summary_path, summary)
    else:
        _write_json(summary_path, summary)
    interpretation_execution_path = interpretation_path.parent / str(
        (interpretation.get("execution_tests") or {}).get("file")
        or EXECUTION_TESTS_NAME
    )
    manifest = {
        "kind": "pilot_shortfall_count_manifest",
        "version": 3 if run_version == "v3" else "3r",
        "run_version": run_version,
        "status": status,
        "created_utc": summary["created_utc"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "run_fingerprint": fingerprint,
        "files": {
            COUNT_SUMMARY_NAME: _file_entry(COUNT_SUMMARY_NAME, summary_path),
            COUNT_PER_REQUEST_NAME: _file_entry(
                COUNT_PER_REQUEST_NAME, per_request_path
            ),
            COUNT_LOG_NAME: _file_entry(COUNT_LOG_NAME, log_path),
            "topup_plan": _file_entry("topup_plan", plan_path),
            INTERPRETATION_MANIFEST_NAME: _file_entry(
                INTERPRETATION_MANIFEST_NAME, interpretation_path
            ),
            EXECUTION_TESTS_NAME: _file_entry(
                EXECUTION_TESTS_NAME, interpretation_execution_path
            ),
            ATTEMPT_LEDGER_NAME: _file_entry(
                ATTEMPT_LEDGER_NAME, ledger_file
            ),
        },
    }
    manifest["manifest_sha256"] = _hash_payload(manifest, "manifest_sha256")
    if exclusive_write:
        _write_json_exclusive(out_dir / COUNT_MANIFEST_NAME, manifest)
    else:
        _write_json(out_dir / COUNT_MANIFEST_NAME, manifest)
    if blocking_infrastructure:
        raise ShortfallCountError(
            "the official count has unresolved infrastructure failures "
            f"({len(blocking_infrastructure)} requests); rerun the identical "
            "input bytes under the identical frozen configuration before "
            "deciding: "
            + ", ".join(
                str(entry.get("request_id"))
                for entry in blocking_infrastructure[:5]
            )
        )
    return summary


# ---------------------------------------------------------------------------
# Final corpus
# ---------------------------------------------------------------------------
def _request_order_key(request_id: str) -> tuple[str, str]:
    return (_sha256_text(f"0:{request_id}"), str(request_id))


def _canonical_problem_id(problem_id: Any) -> str:
    return str(problem_id).strip()


def _trace_from_record(
    record: Mapping[str, Any],
    problem: Mapping[str, Any],
    grade_row: Mapping[str, Any],
    *,
    split_name: str,
) -> dict[str, Any]:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": str(record.get("content") or ""),
    }
    reasoning = str(record.get("reasoning") or "")
    if reasoning:
        assistant["reasoning"] = reasoning
    return {
        "problem_id": _canonical_problem_id(problem["problem_id"]),
        "source": problem.get("source"),
        "kind": str(problem.get("kind") or ""),
        "split": problem.get("split"),
        "pilot_subsplit": split_name,
        "request_id": str(record.get("request_id")),
        "sample_index": int(record.get("sample_index") or 0),
        "messages": [
            {"role": "user", "content": problem["prompt"]},
            assistant,
        ],
        "rendered_tokens": int(grade_row.get("rendered_tokens") or 0),
        "reasoning_share": float(grade_row.get("reasoning_share") or 0.0),
        "engine": record.get("engine"),
        "model_revision": record.get("model_revision"),
    }


def build_eligible_traces(
    *,
    freeze: Mapping[str, Any],
    pool_dir: Path | str,
    records: Mapping[str, Mapping[str, Any]],
    grade_rows: Mapping[str, Mapping[str, Any]],
    verified_request_ids: Iterable[str],
    allowed_splits: Sequence[str] = PILOT_SPLITS,
) -> dict[str, list[dict[str, Any]]]:
    """Turn verified per-request entries into trace rows, split by pilot set."""

    problems = _problem_by_id(Path(pool_dir))
    allowed = set(allowed_splits)
    verified = set(verified_request_ids)
    traces: dict[str, list[dict[str, Any]]] = {name: [] for name in PILOT_SPLITS}
    for name in PILOT_SPLITS:
        if name not in allowed:
            continue
        for request in freeze["selected_inputs"]["splits"][name]["requests"]:
            request_id = request["request_id"]
            if request_id not in verified:
                continue
            record = records.get(request_id)
            if record is None:
                raise ValueError(f"verified request {request_id} has no shard record")
            problem = problems[request["problem_id"]]
            traces[name].append(
                _trace_from_record(
                    record,
                    problem,
                    grade_rows.get(request_id) or {},
                    split_name=name,
                )
            )
    return traces


EXTENSION_NOT_IMPLEMENTED = (
    "extension branch not implemented in this registration build; required "
    "only if extension_required is true"
)


def assemble_corpus(
    traces: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    allow_extra: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sort, deduplicate and count one final pilot split.

    The realized count was ``no_extension_original_only``.  This registration
    build refuses every extra/top-up trace; it does not grade or assemble the
    conditional extension.
    """

    if allow_extra:
        raise ValueError(EXTENSION_NOT_IMPLEMENTED)
    by_request: dict[str, dict[str, Any]] = {}
    for trace in traces:
        request_id = str(trace.get("request_id") or "")
        if not request_id:
            raise ValueError("every final trace needs request_id")
        sample_index = int(trace.get("sample_index") or 0)
        if split_name == "pilot_p1" and sample_index >= min(TOPUP_INDICES):
            raise ValueError(f"{request_id}: P1 final corpus cannot include top-ups")
        if sample_index >= min(TOPUP_INDICES):
            raise ValueError(f"{request_id}: {EXTENSION_NOT_IMPLEMENTED}")
        if request_id in by_request:
            if by_request[request_id] != dict(trace):
                raise ValueError(f"{request_id}: conflicting duplicate trace")
            continue
        by_request[request_id] = dict(trace)
    ordered = [by_request[key] for key in sorted(by_request, key=_request_order_key)]
    problem_ids = sorted({_canonical_problem_id(row["problem_id"]) for row in ordered})
    domains: dict[str, int] = {}
    for row in ordered:
        kind = str(row.get("kind") or "")
        domains[kind] = domains.get(kind, 0) + 1
    return ordered, {
        "kind": "pilot_shortfall_final_split",
        "split": split_name,
        "rows": len(ordered),
        "distinct_problems": len(problem_ids),
        "problem_ids": problem_ids,
        "request_ids": [str(row["request_id"]) for row in ordered],
        "domains": dict(sorted(domains.items())),
        "scored_rendered_tokens": sum(
            int(row.get("rendered_tokens") or 0) for row in ordered
        ),
        "order_rule": (
            "ascending sha256(UTF8('0:'+request_id)); ties by request_id"
        ),
    }


def write_final_corpus(
    out_dir: Path | str,
    traces_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    allow_extra: bool = False,
    tokenizer: Any = None,
    renderer: Callable[..., list[dict[str, Any]]] | None = None,
    max_length: int = MAX_LENGTH,
    exclusive: bool = False,
    authorization: V3RunAuthorization | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if authorization is not None:
        _authorize_v3_stage(
            authorization, stage="corpus", out_dir=out_dir
        )
        if resume:
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            _require_new_dir(out_dir)
    elif exclusive:
        raise ShortfallV3RunError(
            "exclusive corpus writing requires the authorization of the "
            "claimed official v3 run; exclusive=True is not authorization"
        )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "pilot_shortfall_final_corpus",
        "version": 1,
        "created_utc": _utc_now(),
        "allow_extra": bool(allow_extra),
        "max_length": int(max_length),
        "rendering": {
            "renderer": (
                f"{getattr(renderer, '__module__', '')}."
                f"{getattr(renderer, '__qualname__', getattr(renderer, '__name__', ''))}"
                if renderer is not None
                else None
            ),
            "tokenizer": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
                if tokenizer is not None
                else None
            ),
            "rule": "re-render raw prompt + original completion; compare ids/labels/mask",
        },
        "splits": {},
    }
    files: dict[str, dict[str, Any]] = {}
    for name in PILOT_SPLITS:
        ordered, split_manifest = assemble_corpus(
            list(traces_by_split.get(name) or []),
            split_name=name,
            allow_extra=allow_extra,
        )
        path = out_dir / f"{name}.jsonl"
        sha = _write_jsonl(path, ordered)
        files[path.name] = _file_entry(name, path)
        split_manifest["sha256"] = sha
        if tokenizer is not None and renderer is not None and ordered:
            rendered = renderer(
                ordered, tokenizer, max_length=int(max_length), source=path
            )
            rendered_path = out_dir / f"{name}.rendered.jsonl"
            rendered_sha = _write_jsonl(rendered_path, rendered)
            files[rendered_path.name] = _file_entry(
                f"{name}_rendered", rendered_path
            )
            split_manifest["rendered"] = {
                "file": rendered_path.name,
                "sha256": rendered_sha,
                "rows": len(rendered),
                "assistant_label_positions": sum(
                    len(row.get("label_positions") or []) for row in rendered
                ),
            }
        manifest["splits"][name] = split_manifest
    manifest["files"] = files
    manifest["counts"] = {
        name: {
            "rows": manifest["splits"][name]["rows"],
            "distinct_problems": manifest["splits"][name]["distinct_problems"],
        }
        for name in PILOT_SPLITS
    }
    manifest["readiness"] = {
        "pilot_tfit_ok": manifest["splits"]["pilot_tfit"]["distinct_problems"]
        >= TFIT_MIN,
        "pilot_selection_ok": manifest["splits"]["pilot_selection"]["distinct_problems"]
        >= SELECTION_MIN,
        "pilot_p1_gates": False,
    }
    manifest["manifest_sha256"] = _hash_payload(manifest, "manifest_sha256")
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Freeze index and execution guard
# ---------------------------------------------------------------------------
_INDEX_BINDING_FILES = (
    ("prereg_yaml", "prereg/prereg.yaml"),
    ("prereg_md", "prereg/PREREGISTRATION.md"),
    ("prereg_python", "src/gemmaloop/prereg.py"),
    ("run_pilot", "scripts/run_pilot.py"),
    ("modal_stage0", "scripts/modal_stage0.py"),
    ("sample_stage0_local", "scripts/sample_stage0_local.py"),
    ("freeze_script", "scripts/freeze_pilot_verifier.py"),
    ("count_script", "scripts/count_pilot_verified.py"),
    ("recover_source_types_script", "scripts/recover_source_types.py"),
    ("freeze_recount_v3_script", "scripts/freeze_recount_v3.py"),
    ("topup_plan_script", "scripts/plan_pilot_topup.py"),
    ("corpus_script", "scripts/build_pilot_corpus.py"),
    ("shortfall_module", "src/gemmaloop/stage0/shortfall.py"),
    ("interpretation_module", "src/gemmaloop/stage0/interpretation.py"),
    ("source_recovery", "src/gemmaloop/stage0/source_recovery.py"),
    ("sources", "src/gemmaloop/stage0/sources.py"),
    ("stage0_local", "src/gemmaloop/stage0/local.py"),
    ("modal_lite", "src/gemmaloop/stage0/modal_lite.py"),
    ("postprocess", "src/gemmaloop/stage0/postprocess.py"),
    ("grader", "src/gemmaloop/stage0/grader.py"),
    ("lcb_extraction", "src/gemmaloop/eval/benchmarks/lcb.py"),
    ("train_labels", "src/gemmaloop/train/labels.py"),
    ("pilot_eval_rows", "src/gemmaloop/pilot/eval_rows.py"),
    ("pilot_harness", "src/gemmaloop/pilot/harness.py"),
    ("pilot_selection", "src/gemmaloop/pilot/selection.py"),
    ("calibrated_nll", "src/gemmaloop/eval/calibrated_nll.py"),
)


def load_final_corpus_manifest(
    path: Path | str, *, verify_files: bool = True
) -> dict[str, Any]:
    """Load a final corpus manifest and verify its self and file hashes."""

    path = Path(path)
    assert_not_void_artifact(path, role="final corpus manifest")
    payload = _read_json(path)
    if payload.get("kind") != "pilot_shortfall_final_corpus":
        raise ValueError(f"{path}: not a final pilot corpus manifest")
    if payload.get("manifest_sha256") != _hash_payload(payload, "manifest_sha256"):
        raise ValueError(f"{path}: final corpus manifest self hash does not recompute")
    if verify_files:
        for name, entry in (payload.get("files") or {}).items():
            bound = Path(str(entry.get("path") or ""))
            if not bound.is_file():
                raise ValueError(f"{path}: final corpus file {name} is missing")
            if _sha256_file(bound) != entry.get("sha256"):
                raise ValueError(f"{path}: final corpus file {name} changed")
            if int(entry.get("bytes") or -1) != int(bound.stat().st_size):
                raise ValueError(f"{path}: final corpus file {name} size changed")
    return payload


def _rendered_mask(row: Mapping[str, Any], *, source: str, index: int) -> list[int]:
    labels = [int(value) for value in row.get("labels") or []]
    input_ids = [int(value) for value in row.get("input_ids") or []]
    if len(labels) != len(input_ids):
        raise ValueError(f"{source}: rendered row {index} ids/labels length differ")
    positions = [int(value) for value in row.get("label_positions") or []]
    expected = [position for position, value in enumerate(labels) if value != -100]
    if positions != expected:
        raise ValueError(f"{source}: rendered row {index} assistant mask changed")
    if not positions:
        raise ValueError(f"{source}: rendered row {index} has no assistant labels")
    return positions


def verify_final_corpus(
    final_manifest_path: Path | str,
    *,
    freeze: Mapping[str, Any] | None = None,
    expected_verified: Mapping[str, set[str]] | None = None,
    records: Mapping[str, Mapping[str, Any]] | None = None,
    grade_rows: Mapping[str, Mapping[str, Any]] | None = None,
    problems: Mapping[str, Mapping[str, Any]] | None = None,
    tokenizer: Any = None,
    renderer: Callable[..., list[dict[str, Any]]] | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Derive identities/counts/domains/tokens from the actual corpus bytes.

    Every declared hash must match, the fixed request ordering must hold, each
    raw row must correspond to the frozen selected request and the original
    completion record, and each rendered row/mask must correspond to its raw
    row.  A substituted corpus is refused even when its counts are unchanged.
    """

    final_manifest_path = Path(final_manifest_path)
    out_dir = final_manifest_path.parent
    manifest = load_final_corpus_manifest(final_manifest_path)
    files = {
        str(name): dict(entry)
        for name, entry in (manifest.get("files") or {}).items()
    }
    splits: dict[str, Any] = {}
    for name in PILOT_SPLITS:
        declared = dict((manifest.get("splits") or {}).get(name) or {})
        raw_name = f"{name}.jsonl"
        raw_entry = files.get(raw_name)
        if raw_entry is None:
            raise ValueError(f"{final_manifest_path}: {raw_name} is not bound")
        raw_path = Path(str(raw_entry.get("path") or ""))
        if raw_path.resolve() != (out_dir / raw_name).resolve():
            raise ValueError(f"{final_manifest_path}: {raw_name} path is not canonical")
        rows = local.read_jsonl(raw_path)
        request_ids = [str(row.get("request_id") or "") for row in rows]
        if not all(request_ids):
            raise ValueError(f"{raw_path}: a raw row has no request_id")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"{raw_path}: duplicate request ids")
        expected_order = sorted(request_ids, key=_request_order_key)
        if request_ids != expected_order:
            raise ValueError(f"{raw_path}: request ordering is not the frozen order")
        problem_ids = sorted({_canonical_problem_id(row.get("problem_id")) for row in rows})
        domains: dict[str, int] = {}
        scored_tokens = 0
        for row in rows:
            request_id = str(row.get("request_id") or "")
            _, separator, index_text = request_id.rpartition(":")
            if not separator or not index_text.isdigit():
                raise ValueError(f"{raw_path}: malformed request id {request_id!r}")
            if int(index_text) >= min(TOPUP_INDICES):
                raise ValueError(f"{request_id}: {EXTENSION_NOT_IMPLEMENTED}")
            if str(row.get("pilot_subsplit") or "") != name:
                raise ValueError(
                    f"{raw_path}: {request_id} is not a member of split {name}"
                )
            kind = str(row.get("kind") or "")
            if not kind:
                raise ValueError(f"{raw_path}: {request_id} has no domain")
            domains[kind] = domains.get(kind, 0) + 1
            scored_tokens += int(row.get("rendered_tokens") or 0)
            if freeze is not None:
                split_inputs = freeze["selected_inputs"]["splits"][name]
                allowed_requests = {
                    str(request["request_id"])
                    for request in split_inputs["requests"]
                }
                if request_id not in allowed_requests:
                    raise ValueError(
                        f"{raw_path}: {request_id} is not in the frozen {name} "
                        "selected requests"
                    )
                allowed_problems = set(split_inputs["problem_ids"])
                if str(row.get("problem_id")) not in allowed_problems:
                    raise ValueError(
                        f"{raw_path}: {row.get('problem_id')} is not a frozen "
                        f"{name} problem"
                    )
            if records is not None and problems is not None:
                record = records.get(request_id)
                if record is None:
                    raise ValueError(
                        f"{raw_path}: verified request {request_id} has no "
                        "original completion record"
                    )
                problem = problems.get(str(row.get("problem_id")))
                if problem is None:
                    raise ValueError(
                        f"{raw_path}: {row.get('problem_id')} is not in the pool"
                    )
                messages = list(row.get("messages") or [])
                if len(messages) != 2:
                    raise ValueError(f"{raw_path}: {request_id} is not a two-turn trace")
                if messages[0].get("content") != problem["prompt"]:
                    raise ValueError(
                        f"{raw_path}: {request_id} prompt differs from the pool"
                    )
                if messages[1].get("content") != str(record.get("content") or ""):
                    raise ValueError(
                        f"{raw_path}: {request_id} completion content changed"
                    )
                reasoning = str(record.get("reasoning") or "")
                if reasoning and messages[1].get("reasoning") != reasoning:
                    raise ValueError(
                        f"{raw_path}: {request_id} completion reasoning changed"
                    )
                if int(row.get("sample_index") or -1) != int(
                    record.get("sample_index") or -1
                ):
                    raise ValueError(
                        f"{raw_path}: {request_id} sample index differs from the record"
                    )
                if str(row.get("source") or "") != str(problem.get("source") or ""):
                    raise ValueError(f"{raw_path}: {request_id} source changed")
                if str(row.get("kind") or "") != str(problem.get("kind") or ""):
                    raise ValueError(f"{raw_path}: {request_id} kind changed")
                if grade_rows is not None:
                    grade = grade_rows.get(request_id) or {}
                    if int(row.get("rendered_tokens") or 0) != int(
                        grade.get("rendered_tokens") or 0
                    ):
                        raise ValueError(
                            f"{raw_path}: {request_id} token denominator changed"
                        )
                    if float(row.get("reasoning_share") or 0.0) != float(
                        grade.get("reasoning_share") or 0.0
                    ):
                        raise ValueError(
                            f"{raw_path}: {request_id} reasoning share changed"
                        )
        if int(declared.get("rows") or -1) != len(rows):
            raise ValueError(f"{raw_path}: declared row count does not recompute")
        if declared.get("request_ids") != request_ids:
            raise ValueError(f"{raw_path}: declared request ids do not recompute")
        if declared.get("problem_ids") != problem_ids:
            raise ValueError(f"{raw_path}: declared problem ids do not recompute")
        if dict(declared.get("domains") or {}) != dict(sorted(domains.items())):
            raise ValueError(f"{raw_path}: declared domains do not recompute")
        if int(declared.get("distinct_problems") or -1) != len(problem_ids):
            raise ValueError(f"{raw_path}: declared distinct count does not recompute")
        if int(declared.get("scored_rendered_tokens") or -1) != scored_tokens:
            raise ValueError(f"{raw_path}: declared token denominator does not recompute")
        if declared.get("sha256") != raw_entry.get("sha256"):
            raise ValueError(f"{raw_path}: split hash differs from the file hash")
        if expected_verified is not None:
            expected_ids = set(expected_verified.get(name) or set())
            if set(request_ids) != expected_ids:
                raise ValueError(
                    f"{raw_path}: corpus request set does not equal the official "
                    f"verified {name} requests "
                    f"(missing={sorted(expected_ids - set(request_ids))[:3]}, "
                    f"extra={sorted(set(request_ids) - expected_ids)[:3]})"
                )
        rendered_info = declared.get("rendered") or {}
        if name in SPLITS_WITH_TOPUP and not rendered_info:
            raise ValueError(
                f"{raw_path}: split {name} has no rendered artifacts"
            )
        label_positions_total = 0
        if rendered_info:
            rendered_name = str(rendered_info.get("file") or "")
            rendered_entry = files.get(rendered_name)
            if rendered_entry is None:
                raise ValueError(f"{final_manifest_path}: {rendered_name} is not bound")
            rendered_path = Path(str(rendered_entry.get("path") or ""))
            if rendered_path.resolve() != (out_dir / rendered_name).resolve():
                raise ValueError(
                    f"{final_manifest_path}: {rendered_name} path is not canonical"
                )
            rendered_rows = local.read_jsonl(rendered_path)
            if len(rendered_rows) != len(rows):
                raise ValueError(
                    f"{rendered_path}: rendered row count does not match raw rows"
                )
            for index, (raw_row, rendered_row) in enumerate(zip(rows, rendered_rows)):
                if str(rendered_row.get("problem_id")) != str(raw_row.get("problem_id")):
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} problem id changed"
                    )
                if rendered_row.get("sample_index") is not None and int(
                    rendered_row["sample_index"]
                ) != int(raw_row.get("sample_index") or 0):
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} sample index changed"
                    )
                if int(rendered_row.get("row_id", -1)) != index:
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} identity changed"
                    )
                label_positions_total += len(
                    _rendered_mask(rendered_row, source=str(rendered_path), index=index)
                )
            if int(rendered_info.get("rows") or -1) != len(rendered_rows):
                raise ValueError(
                    f"{rendered_path}: declared rendered row count does not recompute"
                )
            if int(rendered_info.get("assistant_label_positions") or -1) != (
                label_positions_total
            ):
                raise ValueError(
                    f"{rendered_path}: declared assistant-label count does not recompute"
                )
            if rendered_info.get("sha256") != rendered_entry.get("sha256"):
                raise ValueError(
                    f"{rendered_path}: rendered split hash differs from the file hash"
                )
            if renderer is None or tokenizer is None:
                raise ValueError(
                    f"{rendered_path}: rendered bytes cannot be verified without "
                    "the bound tokenizer and renderer"
                )
            rendering = manifest.get("rendering") or {}
            bound_max_length = int(
                max_length
                if max_length is not None
                else (manifest.get("max_length") or rendering.get("max_length"))
                or MAX_LENGTH
            )
            expected_rows = renderer(
                rows, tokenizer, max_length=bound_max_length, source=raw_path
            )
            if len(expected_rows) != len(rendered_rows):
                raise ValueError(
                    f"{rendered_path}: re-rendering produced a different row count"
                )
            for index, (stored, expected_row) in enumerate(
                zip(rendered_rows, expected_rows)
            ):
                for key in ("problem_id", "sample_index", "row_id"):
                    if stored.get(key) != expected_row.get(key):
                        raise ValueError(
                            f"{rendered_path}: rendered row {index} {key} is not "
                            "the rendering of its raw row"
                        )
                if [int(value) for value in stored.get("input_ids") or []] != [
                    int(value) for value in expected_row.get("input_ids") or []
                ]:
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} input_ids do not "
                        "re-render from the raw prompt and original completion"
                    )
                if [int(value) for value in stored.get("labels") or []] != [
                    int(value) for value in expected_row.get("labels") or []
                ]:
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} labels do not "
                        "re-render from the raw prompt and original completion"
                    )
                if [int(value) for value in stored.get("label_positions") or []] != [
                    int(value) for value in expected_row.get("label_positions") or []
                ]:
                    raise ValueError(
                        f"{rendered_path}: rendered row {index} assistant mask does "
                        "not re-render from the raw prompt and original completion"
                    )
        splits[name] = {
            "rows": len(rows),
            "request_ids": request_ids,
            "problem_ids": problem_ids,
            "distinct_problems": len(problem_ids),
            "domains": dict(sorted(domains.items())),
            "scored_rendered_tokens": scored_tokens,
            "assistant_label_positions": label_positions_total,
            "raw_sha256": raw_entry.get("sha256"),
            "rendered_sha256": (
                files.get(str(rendered_info.get("file") or ""), {}).get("sha256")
                if rendered_info
                else None
            ),
        }
    declared_counts = manifest.get("counts") or {}
    for name in PILOT_SPLITS:
        expected = declared_counts.get(name) or {}
        if int(expected.get("rows") or -1) != splits[name]["rows"]:
            raise ValueError(f"{final_manifest_path}: declared {name} rows do not recompute")
        if int(expected.get("distinct_problems") or -1) != splits[name][
            "distinct_problems"
        ]:
            raise ValueError(
                f"{final_manifest_path}: declared {name} distinct count does not recompute"
            )
    return {
        "manifest": str(final_manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": splits,
        "files": files,
    }


def _amend_rendered_paths(
    final_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    files = {
        str(name): dict(entry)
        for name, entry in (final_manifest.get("files") or {}).items()
    }
    splits: dict[str, Any] = {}
    for name in PILOT_SPLITS:
        entry = dict((final_manifest.get("splits") or {}).get(name) or {})
        rendered = entry.get("rendered")
        if isinstance(rendered, Mapping) and rendered.get("file"):
            bound = files.get(str(rendered["file"])) or {}
            entry["rendered"] = {
                **dict(rendered),
                "path": str(bound.get("path") or ""),
            }
        splits[name] = entry
    return splits


def write_freeze_index(
    *,
    repo_root: Path | str,
    freeze_path: Path | str,
    count_summary_path: Path | str,
    topup_plan_path: Path | str,
    out_path: Path | str,
    final_manifest_path: Path | str | None = None,
    amendment_path: Path | str | None = None,
    allow_pending: bool = False,
    exclusive: bool = False,
    authorization: V3RunAuthorization | None = None,
    resume: bool = False,
    tokenizer: Any = None,
    renderer: Callable[..., list[dict[str, Any]]] | None = None,
    max_length: int | None = None,
    run_version: str = "v3",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    if authorization is not None:
        _authorize_v3_stage(
            authorization, stage="index", out_dir=out_path
        )
        if resume:
            if Path(out_path).exists():
                Path(out_path).unlink()
        elif Path(out_path).exists():
            raise FileExistsError(
                f"v3 index already exists and may not be clobbered: {out_path}"
            )
    elif exclusive:
        raise ShortfallV3RunError(
            "exclusive index writing requires the authorization of the "
            "claimed official v3 run; exclusive=True is not authorization"
        )
    freeze = load_freeze_manifest(freeze_path)
    summary = load_verified_count(
        count_summary_path,
        topup_plan_path=topup_plan_path,
        freeze_path=freeze_path,
    )
    plan = validate_topup_plan(_read_json(Path(topup_plan_path)), freeze=freeze)
    if bool(summary.get("extension_required")) or str(
        summary.get("extension_decision")
    ) != "no_extension_original_only":
        raise ValueError(
            f"{EXTENSION_NOT_IMPLEMENTED}; the realized decision must be "
            "no_extension_original_only"
        )
    if final_manifest_path is None:
        raise ValueError(
            "the freeze index requires a complete final corpus manifest with "
            "rendered T-fit/selection artifacts"
        )
    final_manifest_path = Path(final_manifest_path)
    final_manifest = load_final_corpus_manifest(final_manifest_path)
    if final_manifest.get("allow_extra"):
        raise ValueError(EXTENSION_NOT_IMPLEMENTED)
    count_dir = Path(count_summary_path).parent
    per_request_rows = local.read_jsonl(count_dir / COUNT_PER_REQUEST_NAME)
    expected_verified: dict[str, set[str]] = {name: set() for name in PILOT_SPLITS}
    for row in per_request_rows:
        if not row.get("verified"):
            continue
        split_name = str(row.get("split") or "")
        if split_name not in expected_verified:
            raise ValueError(f"verified request {row.get('request_id')} has no pilot split")
        expected_verified[split_name].add(str(row.get("request_id")))
    freeze_inputs = freeze.get("inputs") or {}
    pool_problems = _problem_by_id(Path(str(freeze_inputs.get("pool_dir") or "")))
    lite_records = _load_records_by_request(Path(str(freeze_inputs.get("lite_dir") or "")))
    lite_grades = _load_grades_by_request(Path(str(freeze_inputs.get("lite_dir") or "")))
    verified_corpus = verify_final_corpus(
        final_manifest_path,
        freeze=freeze,
        expected_verified=expected_verified,
        records=lite_records,
        grade_rows=lite_grades,
        problems=pool_problems,
        tokenizer=tokenizer,
        renderer=renderer,
        max_length=max_length,
    )
    final_counts = {
        name: int(verified_corpus["splits"][name]["distinct_problems"])
        for name in PILOT_SPLITS
    }

    bindings: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []

    def bind(label: str, path: Path | str, *, required: bool = True) -> None:
        path = Path(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"freeze-index binding {label}: {path}")
            pending.append({"label": label, "path": str(path)})
            return
        bindings.append({"label": label, **_file_entry(label, path)})

    for label, relative in _INDEX_BINDING_FILES:
        bind(label, repo_root / relative)
    amendment = (
        Path(amendment_path)
        if amendment_path is not None
        else repo_root / AMENDMENT_REPO_PATH
    )
    if amendment.is_file():
        bind("pilot_shortfall_amendment", amendment)
    elif allow_pending:
        pending.append(
            {"label": "pilot_shortfall_amendment", "path": str(amendment)}
        )
    else:
        bind("pilot_shortfall_amendment", amendment)

    bind("verifier_freeze", freeze_path)
    bind(
        "verifier_test_manifest",
        Path(freeze_path).parent / freeze["tests"]["manifest_file"],
    )
    interpretation = freeze.get("interpretation") or {}
    if interpretation:
        bind("interpretation_manifest", interpretation["manifest"])
        bind("interpretation_execution_tests", interpretation["execution_tests_file"])
    for entry in (freeze.get("inputs") or {}).get("files") or []:
        bind(f"evidence/{entry.get('label')}", entry["path"])
    bind("count_summary", count_summary_path)
    bind("count_manifest", Path(count_summary_path).parent / COUNT_MANIFEST_NAME)
    bind("count_per_request", Path(count_summary_path).parent / COUNT_PER_REQUEST_NAME)
    bind("count_log", Path(count_summary_path).parent / COUNT_LOG_NAME)
    bind("topup_plan", topup_plan_path)

    bind("final_corpus_manifest", final_manifest_path)
    for name, entry in (final_manifest.get("files") or {}).items():
        bind(f"final_corpus/{name}", entry["path"])
    final_files = {
        "manifest": str(final_manifest_path.resolve()),
        "manifest_sha256": _sha256_file(final_manifest_path),
        "splits": _amend_rendered_paths(final_manifest),
        "derived": verified_corpus["splits"],
    }

    if int(final_counts["pilot_tfit"]) != int(summary["nT0"]):
        raise ValueError(
            "final T-fit corpus count does not equal the original-only count "
            f"({final_counts['pilot_tfit']} != {summary['nT0']}); "
            + EXTENSION_NOT_IMPLEMENTED
        )
    if int(final_counts["pilot_selection"]) != int(summary["nS0"]):
        raise ValueError(
            "final selection corpus count does not equal the original-only count "
            f"({final_counts['pilot_selection']} != {summary['nS0']}); "
            + EXTENSION_NOT_IMPLEMENTED
        )
    if int(final_counts["pilot_p1"]) != int(summary["pilot_p1"]):
        raise ValueError(
            "final P1 corpus count does not equal the original-only count "
            f"({final_counts['pilot_p1']} != {summary['pilot_p1']})"
        )

    status = INDEX_STATUS_PENDING if pending else INDEX_STATUS_COMPLETE
    payload: dict[str, Any] = {
        "kind": "pilot_shortfall_freeze_index",
        "version": 1,
        "run_version": str(run_version),
        "status": status,
        "created_utc": _utc_now(),
        "git_commit": _git_commit(repo_root),
        "rule": {
            "source": "codex-ruling-pilot-shortfall.md (2026-11-06)",
            "amendment_adoption": (
                "verified math+code; descriptive-only P1; one conditional "
                "four-attempt extension per original P2 math/code problem"
            ),
        },
        "final_counts": final_counts,
        "count_status": summary.get("status"),
        "terminal_infrastructure": summary.get("terminal_infrastructure"),
        "decision": {
            "nT0": int(summary["nT0"]),
            "nS0": int(summary["nS0"]),
            "extension_required": False,
            "extension_decision": "no_extension_original_only",
            "realized_decision": "no_extension_original_only",
            "topup_run_id": plan["topup_run_id"],
            "topup_request_count": int(plan["request_count"]),
        },
        "thresholds": {
            "pilot_tfit_min": TFIT_MIN,
            "pilot_selection_min": SELECTION_MIN,
            "pilot_p1_gates": False,
        },
        "readiness": {
            "pilot_tfit_ok": final_counts["pilot_tfit"] >= TFIT_MIN,
            "pilot_selection_ok": final_counts["pilot_selection"] >= SELECTION_MIN,
            "pilot_p1_descriptive": True,
            "final_p2_runnable": (
                final_counts["pilot_tfit"] >= TFIT_MIN
                and final_counts["pilot_selection"] >= SELECTION_MIN
            ),
        },
        "bindings": bindings,
        "pending_bindings": pending,
        "final_corpus": final_files,
        "interpretation": {
            key: interpretation.get(key)
            for key in (
                "manifest",
                "manifest_sha256",
                "execution_tests_file",
                "execution_tests_sha256",
                "authenticated_problem_count",
                "excluded_problem_count",
                "excluded_problem_ids",
            )
        }
        if interpretation
        else None,
        "declaration": (
            "No pilot score or trained-arm result informed this amendment or "
            "this index. Original-model sampling/verification counts were "
            "inspected; access is declared, not denied."
        ),
        "index_sha256": "",
    }
    payload["index_sha256"] = _hash_payload(payload, "index_sha256")
    if exclusive and authorization is None:
        _write_json_exclusive(Path(out_path), payload)
    else:
        _write_json(Path(out_path), payload)
    return payload


def _require_operative_count_evidence(
    payload: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """Refuse operative readiness unless the index proves the claimed v3r run.

    The declared ``run_version`` must be exactly the authorized operative
    version.  Missing or unknown metadata never disables verification: v1, v2,
    v3 and void records are not operative.  The index must bind both the count
    manifest and its summary, and the full verified-count loader (including the
    durable attempt-ledger and verifier-interface-error scan) must accept the
    bound bytes.
    """

    declared = payload.get("run_version")
    if declared != OPERATIVE_RUN_VERSION:
        raise ShortfallReadinessError(
            f"{source}: operative readiness requires declared run_version "
            f"{OPERATIVE_RUN_VERSION!r}; got {declared!r} (v1/v2/v3/void are "
            "never operative)"
        )
    bindings = {
        str(entry.get("label") or ""): entry
        for entry in payload.get("bindings") or []
    }
    count_manifest_entry = bindings.get("count_manifest")
    if count_manifest_entry is None:
        raise ShortfallReadinessError(
            f"{source}: the completed {declared} index does not bind its "
            "count_manifest"
        )
    count_summary_entry = bindings.get("count_summary")
    if count_summary_entry is None:
        raise ShortfallReadinessError(
            f"{source}: the completed {declared} index does not bind its "
            "count_summary"
        )
    count_manifest_path = Path(str(count_manifest_entry.get("path") or ""))
    count_summary_path = Path(str(count_summary_entry.get("path") or ""))
    for label, path, entry in (
        ("count_manifest", count_manifest_path, count_manifest_entry),
        ("count_summary", count_summary_path, count_summary_entry),
    ):
        if not path.is_file():
            raise ShortfallReadinessError(
                f"{source}: the bound {label} is missing: {path}"
            )
        expected_sha = entry.get("sha256")
        if expected_sha and _sha256_file(path) != expected_sha:
            raise ShortfallReadinessError(
                f"{source}: the bound {label} changed after the index: {path}"
            )
    manifest = _read_json(count_manifest_path)
    if str(manifest.get("run_version") or "") != declared:
        raise ShortfallReadinessError(
            f"{source}: the bound count manifest run_version "
            f"{manifest.get('run_version')!r} is not the operative {declared!r}"
        )
    summary_entry = (manifest.get("files") or {}).get(COUNT_SUMMARY_NAME)
    if not isinstance(summary_entry, Mapping):
        raise ShortfallReadinessError(
            f"{count_manifest_path}: the count manifest does not bind summary.json"
        )
    manifest_summary_path = Path(str(summary_entry.get("path") or ""))
    if manifest_summary_path.resolve() != count_summary_path.resolve():
        raise ShortfallReadinessError(
            f"{source}: the index and count manifest disagree on the count summary"
        )
    if _sha256_file(count_summary_path) != summary_entry.get("sha256"):
        raise ShortfallReadinessError(
            f"{source}: the count summary changed after the count manifest"
        )
    freeze_path: Path | None = None
    freeze_entry = bindings.get("verifier_freeze")
    if freeze_entry is not None:
        freeze_path = Path(str(freeze_entry.get("path") or ""))
        if not freeze_path.is_file():
            raise ShortfallReadinessError(
                f"{source}: the bound verifier freeze is missing: {freeze_path}"
            )
    try:
        summary = load_verified_count(
            count_summary_path, freeze_path=freeze_path
        )
    except (ValueError, OSError, ShortfallV3RunError) as exc:
        raise ShortfallReadinessError(f"{source}: {exc}") from exc
    if str(summary.get("run_version") or "") != declared:
        raise ShortfallReadinessError(
            f"{source}: the verified count run_version "
            f"{summary.get('run_version')!r} is not the operative {declared!r}"
        )
    if payload.get("count_status") not in (None, summary.get("status")):
        raise ShortfallReadinessError(
            f"{source}: the index count_status {payload.get('count_status')!r} "
            f"does not agree with the verified count {summary.get('status')!r}"
        )
    return summary


def check_freeze_readiness(index_path: Path | str) -> dict[str, Any]:
    """Fail closed unless the completed freeze index and its 20/30 gate hold."""

    index_path = Path(index_path)
    try:
        assert_not_void_artifact(index_path, role="freeze index/readiness input")
    except ShortfallVoidRunError as exc:
        raise ShortfallReadinessError(str(exc)) from exc
    if not index_path.is_file():
        raise ShortfallReadinessError(
            f"pilot-shortfall freeze index is missing: {index_path}"
        )
    payload = _read_json(index_path)
    status = payload.get("status")
    if status != INDEX_STATUS_COMPLETE:
        raise ShortfallReadinessError(
            f"{index_path}: pilot-shortfall freeze index is not complete "
            f"(status={status!r}); the data amendment is pending and no pilot "
            "may run before it is completed and rebound"
        )
    if payload.get("kind") != "pilot_shortfall_freeze_index":
        raise ShortfallReadinessError(
            f"{index_path}: not a pilot-shortfall freeze index"
        )
    count_status = payload.get("count_status")
    if count_status not in {
        None,
        COUNT_STATUS_COMPLETE,
        COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE,
    }:
        raise ShortfallReadinessError(
            f"{index_path}: count status is {count_status!r}; a terminal "
            "infrastructure count must be completed, not blocked"
        )
    terminal = payload.get("terminal_infrastructure")
    if count_status == COUNT_STATUS_COMPLETE_TERMINAL_INFRASTRUCTURE:
        if not isinstance(terminal, Mapping) or int(terminal.get("count") or 0) < 1:
            raise ShortfallReadinessError(
                f"{index_path}: terminal-infrastructure count has no reported "
                "test identities"
            )
    pending = payload.get("pending_bindings") or []
    if pending:
        labels = ", ".join(str(item.get("label")) for item in pending)
        raise ShortfallReadinessError(
            f"{index_path}: unresolved required freeze-index binding(s): {labels}"
        )
    if payload.get("index_sha256") != _hash_payload(payload, "index_sha256"):
        raise ShortfallReadinessError(
            f"{index_path}: index self hash does not recompute"
        )
    labels: set[str] = set()
    for entry in payload.get("bindings") or []:
        label = str(entry.get("label") or "")
        labels.add(label)
        path = Path(str(entry.get("path") or ""))
        if not path.is_file():
            raise ShortfallReadinessError(
                f"bound artifact is missing: {entry.get('label')} ({path})"
            )
        if _sha256_file(path) != entry.get("sha256"):
            raise ShortfallReadinessError(
                f"bound artifact changed: {entry.get('label')} ({path})"
            )
    # Operative readiness always requires the authorized v3r run identity and
    # the bound, fully verified count.  Omitting run_version or the count
    # binding can never disable the loader/interface-error checks.
    _require_operative_count_evidence(payload, source=str(index_path))
    if "pilot_shortfall_amendment" not in labels:
        raise ShortfallReadinessError(
            f"{index_path}: the prereg pilot-shortfall amendment is not bound"
        )
    final_corpus = payload.get("final_corpus") or {}
    if not final_corpus.get("manifest") or "final_corpus_manifest" not in labels:
        raise ShortfallReadinessError(
            f"{index_path}: the index has no complete final corpus manifest"
        )
    for name in SPLITS_WITH_TOPUP:
        if f"final_corpus/{name}.rendered.jsonl" not in labels:
            raise ShortfallReadinessError(
                f"{index_path}: the index has no rendered {name} artifact"
            )
    counts = payload.get("final_counts") or {}
    tfit = int(counts.get("pilot_tfit") or 0)
    selection = int(counts.get("pilot_selection") or 0)
    if tfit < TFIT_MIN:
        raise ShortfallReadinessError(
            f"final T-fit has {tfit} distinct verified problems; need {TFIT_MIN}"
        )
    if selection < SELECTION_MIN:
        raise ShortfallReadinessError(
            f"final selection has {selection} distinct verified problems; "
            f"need {SELECTION_MIN}"
        )
    # P1 intentionally never gates.
    return dict(payload)


def _distinct_problem_count(path: Path) -> int:
    problem_ids: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        problem_id = row.get("problem_id")
        if problem_id is None or not str(problem_id).strip():
            raise ValueError(
                f"{path}:{line_number}: rendered row has no problem_id"
            )
        problem_ids.add(str(problem_id).strip())
    return len(problem_ids)


def check_pilot_inputs(
    index: Mapping[str, Any],
    *,
    fit_data: Path | str | None,
    test_data: Path | str | None,
) -> dict[str, int]:
    """Bind the actual pilot `--fit-data`/`--test-data` to the freeze index.

    The files must be byte-identical (SHA-256) to the rendered final artifacts
    bound in the index, and their distinct-problem counts must equal the indexed
    final counts.  A missing final corpus or a substituted file is refused.
    """

    final_corpus = index.get("final_corpus") or {}
    if not final_corpus.get("manifest"):
        raise ShortfallReadinessError(
            "the freeze index has no complete final corpus manifest; refusing "
            "before any precompute"
        )
    # The pilot entry point is an operative acceptance path too: it may never
    # accept an index that omitted its declared v3r identity or count binding.
    _require_operative_count_evidence(index, source="pilot inputs")
    bindings = {
        str(entry.get("label") or ""): entry
        for entry in index.get("bindings") or []
    }
    counts = index.get("final_counts") or {}
    actual = {
        "pilot_tfit": fit_data,
        "pilot_selection": test_data,
    }
    recomputed: dict[str, int] = {}
    for name, path_value in actual.items():
        if path_value is None:
            raise ShortfallReadinessError(
                f"the pilot needs an actual {name} input file to bind to the frozen corpus"
            )
        path = Path(path_value)
        if not path.is_file():
            raise ShortfallReadinessError(
                f"the actual {name} input is missing: {path}"
            )
        label = f"final_corpus/{name}.rendered.jsonl"
        entry = bindings.get(label)
        if entry is None:
            raise ShortfallReadinessError(
                f"the freeze index does not bind a rendered {name} artifact"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != entry.get("sha256"):
            raise ShortfallReadinessError(
                f"the actual {name} input {path} is not byte-identical to the "
                f"frozen rendered artifact {entry.get('path')} "
                f"(sha256 {actual_sha} != {entry.get('sha256')})"
            )
        count = _distinct_problem_count(path)
        indexed = int(counts.get(name) or 0)
        if count != indexed:
            raise ShortfallReadinessError(
                f"the actual {name} input has {count} distinct problems; the "
                f"freeze index records {indexed}"
            )
        recomputed[name] = count
    if recomputed["pilot_tfit"] < TFIT_MIN:
        raise ShortfallReadinessError(
            f"final T-fit has {recomputed['pilot_tfit']} distinct verified "
            f"problems; need {TFIT_MIN}"
        )
    if recomputed["pilot_selection"] < SELECTION_MIN:
        raise ShortfallReadinessError(
            f"final selection has {recomputed['pilot_selection']} distinct "
            f"verified problems; need {SELECTION_MIN}"
        )
    return recomputed


# Re-export the filter helper so corpus builders/tests share one definition.
__all__.extend(["keep_reason", "reasoning_share", "extract_code"])


# ---------------------------------------------------------------------------
# One-command v3 pipeline (interpretation -> freeze -> count -> corpus -> index)
# ---------------------------------------------------------------------------
def _registry_stage_complete(registry: Mapping[str, Any], stage: str) -> bool:
    for entry in registry.get("stages") or []:
        if entry.get("stage") == stage and entry.get("status") == "complete":
            return True
    return False


def _begin_v3_stage(
    registry_path: Path, registry: dict[str, Any], stage: str
) -> dict[str, Any]:
    return _update_v3_registry(
        registry_path,
        registry,
        stage=stage,
        stage_status="running",
        status=V3_STATUS_RUNNING,
    )


def _reset_incomplete_output(path: Path, *, directory: bool) -> None:
    """Remove only an unverified partial artifact of an incomplete stage."""

    path = Path(path)
    if not path.exists():
        return
    if directory:
        if not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    else:
        path.unlink()


def _run_v3_pipeline(
    *,
    repo_root: Path | str,
    pool_dir: Path | str,
    lite_dir: Path | str,
    request_ids_path: Path | str,
    out_root: Path | str,
    image: str = LITE_IMAGE_TAG,
    image_pin: Mapping[str, Any] | None = None,
    workers: int = 1,
    grade_fn: Callable[..., Any] | None = None,
    record_loader: Callable[..., Any] | None = None,
    sources: Sequence[str] = ("apps", "taco", "codecontests"),
    max_infrastructure_retries: int = 1,
    recovery_check: Callable[..., Mapping[str, Any] | bool] | None = None,
    tokenizer: Any = None,
    renderer: Callable[..., list[dict[str, Any]]] | None = None,
    max_length: int = MAX_LENGTH,
    allow_pending: bool = True,
    created_utc: str | None = None,
    resume: bool = False,
    run_version: str = "v3",
) -> dict[str, Any]:
    from gemmaloop.stage0.source_recovery import recover_source_types

    repo_root = Path(repo_root)
    pool_dir = Path(pool_dir)
    lite_dir = Path(lite_dir)
    out_root = Path(out_root)
    spec = _run_spec(run_version)
    if spec.version == "v3r" and int(max_infrastructure_retries) != int(
        V3R_MAX_INFRASTRUCTURE_RETRIES
    ):
        raise ShortfallV3RunError(
            "v3r retains exactly one registered infrastructure retry; "
            f"max_infrastructure_retries={max_infrastructure_retries!r} is refused"
        )

    if resume:
        registry_path = spec.registry_path(lite_dir)
        if spec.version == "v3":
            _refuse_if_void_identity(lite_dir)
        else:
            guard = require_v3r_guard(
                lite_dir,
                out_root=out_root,
                require_review=False,
                require_official=not bool(
                    (load_void_receipt(lite_dir).get("synthetic"))
                ),
            )
            if guard.get("status") != "claimed":
                raise ShortfallV3RunError(
                    "there is no claimed official v3r run to resume "
                    f"({registry_path})"
                )
        if not registry_path.is_file():
            raise ShortfallV3RunError(
                f"there is no claimed official {spec.version} run to resume "
                f"({registry_path})"
            )
        registry = _load_run_registry(registry_path, spec)
        if registry.get("status") == V3_STATUS_COMPLETE:
            raise ShortfallV3RunError(
                f"the official {spec.version} run is already complete; it may "
                "not resume"
            )
        if Path(str(registry.get("out_root") or "")).resolve() != out_root.resolve():
            raise ShortfallV3RunError(
                "resume targets a different output root than the claimed run"
            )
        if Path(str(registry.get("repo_root") or "")).resolve() != repo_root.resolve():
            raise ShortfallV3RunError(
                "resume targets a different checkout than the claimed run"
            )
        if Path(str(registry.get("lite_dir") or "")).resolve() != lite_dir.resolve():
            raise ShortfallV3RunError(
                "resume targets a different lite directory than the claimed run"
            )
        registry["resume_count"] = int(registry.get("resume_count") or 0) + 1
        registry = _update_v3_registry(
            registry_path,
            registry,
            status=V3_STATUS_RESUMABLE,
            error=None,
        )
    elif spec.version == "v3r":
        registry_path, registry = _claim_v3r_registry(
            repo_root=repo_root, lite_dir=lite_dir, out_root=out_root
        )
    else:
        registry_path, registry = _claim_v3_registry(
            repo_root=repo_root, lite_dir=lite_dir, out_root=out_root
        )

    def authorize(stage: str) -> V3RunAuthorization:
        return V3RunAuthorization(
            registry_path=registry_path,
            run_token=str(registry.get("run_token") or ""),
            stage=stage,
            run_version=spec.version,
        )

    current_stage = "started"
    try:
        current_stage = "interpretation"
        interpretation_dir = out_root / spec.stage_directories["interpretation"]
        interpretation_path = interpretation_dir / INTERPRETATION_MANIFEST_NAME
        interpretation: dict[str, Any] | None = None
        if (
            resume
            and _registry_stage_complete(registry, current_stage)
            and interpretation_path.is_file()
        ):
            try:
                interpretation = load_interpretation_manifest(interpretation_path)
            except Exception:  # noqa: BLE001 - incomplete artifact, redo
                interpretation = None
        if interpretation is None:
            if resume:
                _reset_incomplete_output(interpretation_dir, directory=True)
            _begin_v3_stage(registry_path, registry, current_stage)
            problems_by_id = {
                str(row["problem_id"]): row
                for row in local.read_jsonl(pool_dir / "problems.jsonl")
            }
            selected = select_original_inputs(pool_dir, request_ids_path)
            frozen_rows, _test_exclusions = test_manifest_rows(
                selected,
                problems_by_id.values(),
                split_rows=_load_split_rows(pool_dir),
            )
            interpretation = recover_source_types(
                frozen_rows,
                problems_by_id,
                out_dir=interpretation_dir,
                loader=record_loader,
                sources=tuple(sources),
                created_utc=created_utc,
            )
            load_interpretation_manifest(interpretation_path)
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                artifacts={"manifest": str(interpretation_path.resolve())},
            )

        current_stage = "freeze"
        freeze_dir = out_root / spec.stage_directories["freeze"]
        freeze_path = freeze_dir / FREEZE_MANIFEST_NAME
        freeze: dict[str, Any] | None = None
        if (
            resume
            and _registry_stage_complete(registry, current_stage)
            and freeze_path.is_file()
        ):
            try:
                freeze = load_freeze_manifest(freeze_path)
                recorded = (freeze.get("interpretation") or {}).get(
                    "manifest_sha256"
                )
                if recorded != interpretation["manifest_sha256"]:
                    raise ValueError(
                        "the completed freeze does not bind the resumed "
                        "interpretation manifest"
                    )
            except Exception:  # noqa: BLE001 - incomplete artifact, redo
                freeze = None
        if freeze is None:
            if resume:
                _reset_incomplete_output(freeze_dir, directory=True)
            _begin_v3_stage(registry_path, registry, current_stage)
            freeze = freeze_verifier(
                repo_root=repo_root,
                pool_dir=pool_dir,
                lite_dir=lite_dir,
                request_ids_path=request_ids_path,
                out_dir=freeze_dir,
                image=image,
                image_pin=image_pin,
                workers=workers,
                interpretation_path=interpretation_path,
                authorization=authorize("freeze"),
                resume=resume,
                run_version=spec.version,
            )
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                artifacts={"manifest": str(freeze_path.resolve())},
            )

        current_stage = "decision"
        plan_path = out_root / spec.plan_name
        plan_valid = False
        if resume and _registry_stage_complete(registry, current_stage):
            if plan_path.is_file():
                try:
                    validate_topup_plan(_read_json(plan_path), freeze=freeze)
                    plan_valid = True
                except Exception:  # noqa: BLE001 - incomplete artifact, redo
                    plan_valid = False
        if not plan_valid:
            if resume and plan_path.exists():
                plan_path.unlink()
            _begin_v3_stage(registry_path, registry, current_stage)
            plan_topup(
                freeze_path,
                plan_path,
                authorization=authorize("decision"),
                resume=resume,
            )
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                artifacts={"plan": str(plan_path.resolve())},
            )

        current_stage = "count"
        count_dir = out_root / spec.stage_directories["count"]
        summary_path = count_dir / COUNT_SUMMARY_NAME
        summary: dict[str, Any] | None = None
        if (
            resume
            and _registry_stage_complete(registry, current_stage)
            and summary_path.is_file()
        ):
            try:
                verify_count_implementation(freeze)
                summary = load_verified_count(
                    count_dir,
                    freeze_path=freeze_path,
                    topup_plan_path=plan_path,
                )
            except Exception:  # noqa: BLE001 - blocked/partial, resume grading
                summary = None
        if summary is None:
            if resume:
                ledger = count_dir / ATTEMPT_LEDGER_NAME
                if not ledger.is_file():
                    for name in (
                        COUNT_PER_REQUEST_NAME,
                        COUNT_SUMMARY_NAME,
                        COUNT_LOG_NAME,
                        COUNT_MANIFEST_NAME,
                    ):
                        _reset_incomplete_output(count_dir / name, directory=False)
            _begin_v3_stage(registry_path, registry, current_stage)
            summary = count_verified(
                freeze_path=freeze_path,
                interpretation_path=interpretation_path,
                topup_plan_path=plan_path,
                out_dir=count_dir,
                grade_fn=grade_fn,
                workers=workers,
                max_infrastructure_retries=max_infrastructure_retries,
                recovery_check=recovery_check,
                authorization=authorize("count"),
                resume=resume,
            )
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                artifacts={
                    "summary": str(summary_path.resolve()),
                    "status": summary.get("status"),
                },
            )
        if bool(summary.get("extension_required")):
            raise ShortfallCountError(
                "the actual v3 counts require the conditional extension; "
                f"{EXTENSION_NOT_IMPLEMENTED}"
            )

        current_stage = "corpus"
        corpus_dir = out_root / spec.stage_directories["corpus"]
        corpus_manifest_path = corpus_dir / "manifest.json"
        corpus: dict[str, Any] | None = None
        if (
            resume
            and _registry_stage_complete(registry, current_stage)
            and corpus_manifest_path.is_file()
        ):
            try:
                corpus = load_final_corpus_manifest(corpus_manifest_path)
            except Exception:  # noqa: BLE001 - incomplete artifact, redo
                corpus = None
        if corpus is None:
            if resume:
                _reset_incomplete_output(corpus_dir, directory=True)
            _begin_v3_stage(registry_path, registry, current_stage)
            per_request = local.read_jsonl(count_dir / COUNT_PER_REQUEST_NAME)
            verified_ids = {
                str(row["request_id"])
                for row in per_request
                if row.get("verified")
            }
            records = _load_records_by_request(lite_dir)
            grades = _load_grades_by_request(lite_dir)
            traces = build_eligible_traces(
                freeze=freeze,
                pool_dir=pool_dir,
                records=records,
                grade_rows=grades,
                verified_request_ids=verified_ids,
            )
            if tokenizer is None and renderer is not None:
                tokenizer = object()
            if tokenizer is not None and renderer is None:
                from gemmaloop.pilot.eval_rows import render_trace_rows

                renderer = render_trace_rows
            corpus = write_final_corpus(
                corpus_dir,
                traces,
                tokenizer=tokenizer,
                renderer=renderer,
                max_length=int(max_length),
                authorization=authorize("corpus"),
                resume=resume,
            )
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                artifacts={"manifest": str(corpus_manifest_path.resolve())},
            )

        current_stage = "index"
        index_path = out_root / spec.index_name
        index: dict[str, Any] | None = None
        if (
            resume
            and _registry_stage_complete(registry, current_stage)
            and index_path.is_file()
        ):
            try:
                index = _read_json(index_path)
                if index.get("index_sha256") != _hash_payload(
                    index, "index_sha256"
                ):
                    raise ValueError("index self hash does not recompute")
            except Exception:  # noqa: BLE001 - incomplete artifact, redo
                index = None
        if index is None:
            if resume and index_path.exists():
                index_path.unlink()
            _begin_v3_stage(registry_path, registry, current_stage)
            index = write_freeze_index(
                repo_root=repo_root,
                freeze_path=freeze_path,
                count_summary_path=summary_path,
                topup_plan_path=plan_path,
                out_path=index_path,
                final_manifest_path=corpus_manifest_path,
                allow_pending=bool(allow_pending),
                authorization=authorize("index"),
                resume=resume,
                tokenizer=tokenizer,
                renderer=renderer,
                max_length=int(max_length),
                run_version=spec.version,
            )
            _update_v3_registry(
                registry_path,
                registry,
                stage=current_stage,
                stage_status="complete",
                status=V3_STATUS_COMPLETE,
                artifacts={"index": str(index_path.resolve())},
            )
    except BaseException as exc:  # noqa: BLE001 - durable crash state
        try:
            _update_v3_registry(
                registry_path,
                registry,
                status=V3_STATUS_CRASHED,
                failed_stage=current_stage,
                error=f"{type(exc).__name__}: {exc}",
                stage=current_stage,
                stage_status="failed",
            )
        except Exception:
            pass
        raise

    return {
        "interpretation": str(interpretation_path.resolve()),
        "interpretation_manifest_sha256": interpretation["manifest_sha256"],
        "freeze": str(freeze_path.resolve()),
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "count": str(summary_path.resolve()),
        "count_summary": summary,
        "plan": str(plan_path.resolve()),
        "corpus": str(corpus_dir.resolve()),
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "index": str(index_path.resolve()),
        "index_sha256": index["index_sha256"],
        "index_status": index["status"],
        "registry": str(registry_path.resolve()),
        "registry_status": V3_STATUS_COMPLETE,
        "resumed": bool(resume),
    }


def run_v3_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Run the documented v3 sequence once (fresh official claim)."""

    return _run_v3_pipeline(resume=False, **kwargs)


def resume_v3_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Resume the claimed official v3 run without claiming or re-freezing.

    Completed and verified stages are reused.  The count resumes from the
    durable attempt ledger: conclusive attempts are never regraded, an
    interrupted in-flight attempt consumes its single registered retry, and a
    second unknown leaves the run blocked.
    """

    return _run_v3_pipeline(resume=True, run_version="v3", **kwargs)


def run_v3r_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Claim and run the single authorized v3r replacement once.

    The fixed continuity record, immutable void receipt and recorded
    replacement review are verified before the atomic v3r claim.  The frozen
    semantics are v3's except for the disclosed verifier-identity repair.
    """

    return _run_v3_pipeline(resume=False, run_version="v3r", **kwargs)


def resume_v3r_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Resume the claimed v3r run without claiming or re-freezing."""

    return _run_v3_pipeline(resume=True, run_version="v3r", **kwargs)
