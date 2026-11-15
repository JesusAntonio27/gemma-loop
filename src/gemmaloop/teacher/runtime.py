"""Registered offline teacher runtimes and their manifest identity.

The Stage 1 precompute is a one-off, text-only, teacher-forced export.  The
runtime is part of the registered recipe: E4B runs in bf16 with CPU PLE and the
unused vision tower off the GPU, with bitsandbytes LLM.int8 as its registered
fallback; the 12B QAT checkpoint runs in NF4.  A manifest records the exact
semantics of the runtime that produced it so a shard set can never be reused
after changing the method, compute dtype, skipped modules, head policy,
attention implementation, software stack or chunk sizes.

No teacher is loaded here; this module only defines and describes runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "DEFAULT_ATTENTION_IMPLEMENTATION",
    "DEFAULT_HEAD_ACCUMULATION",
    "DEFAULT_HEAD_PROJECTION",
    "DEFAULT_TOKEN_CHUNK",
    "DEFAULT_VOCAB_CHUNK",
    "PRIMARY_TEACHER_RUNTIMES",
    "REGISTERED_FALLBACK_RUNTIMES",
    "TEACHER_RUNTIMES",
    "TeacherRuntime",
    "registered_runtimes",
    "runtime_identity",
    "software_versions",
    "teacher_runtime",
]

DEFAULT_TOKEN_CHUNK = 256
DEFAULT_VOCAB_CHUNK = 8192
DEFAULT_HEAD_PROJECTION = "fp32-vocab-chunked"
DEFAULT_HEAD_ACCUMULATION = "fp32"
DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"

_SOFTWARE_VERSIONS: dict[str, str] | None = None

_VISION_SKIPS: tuple[str, ...] = (
    r".*vision_tower.*",
    r".*embed_vision.*",
    "lm_head",
)


@dataclass(frozen=True)
class TeacherRuntime:
    """One pinned teacher execution recipe."""

    name: str
    method: str
    settings: Mapping[str, Any]
    compute_dtype: str
    skipped_modules: tuple[str, ...]
    head_projection: str
    head_accumulation: str
    attention_implementation: str
    attention_backend_resolution: str
    token_chunk: int
    vocab_chunk: int
    kv_dtype: str

    def identity(
        self,
        *,
        token_chunk: int | None = None,
        vocab_chunk: int | None = None,
        software: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return the full JSON-ready identity recorded in a manifest."""

        return {
            "runtime": self.name,
            "method": self.method,
            "settings": dict(self.settings),
            "compute_dtype": self.compute_dtype,
            "skipped_modules": list(self.skipped_modules),
            "head_projection": self.head_projection,
            "head_accumulation": self.head_accumulation,
            "attention_implementation": self.attention_implementation,
            "attention_backend_resolution": self.attention_backend_resolution,
            "kv_dtype": self.kv_dtype,
            "token_chunk": int(
                self.token_chunk if token_chunk is None else token_chunk
            ),
            "vocab_chunk": int(
                self.vocab_chunk if vocab_chunk is None else vocab_chunk
            ),
            "software": dict(software or software_versions()),
        }


def _bf16_settings() -> dict[str, Any]:
    return {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "cpu_ple": True,
        "vision_device": "cpu",
        "embed_vision_device": "cpu",
    }


def _int8_settings() -> dict[str, Any]:
    return {
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "int8_device_state_repair": "cb_scb_moved_after_module_apply",
        "dtype": "bfloat16",
        "cpu_ple": True,
        "vision_device": "cpu",
        "embed_vision_device": "cpu",
    }


def _nf4_settings() -> dict[str, Any]:
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "vision_device": "cpu",
        "embed_vision_device": "cpu",
    }


_COMMON = dict(
    compute_dtype="bfloat16",
    skipped_modules=_VISION_SKIPS,
    head_projection=DEFAULT_HEAD_PROJECTION,
    head_accumulation=DEFAULT_HEAD_ACCUMULATION,
    attention_implementation=DEFAULT_ATTENTION_IMPLEMENTATION,
    attention_backend_resolution=(
        "sdpa requested; transformers dispatch resolved at load time "
        "(not introspected in the identity)"
    ),
    token_chunk=DEFAULT_TOKEN_CHUNK,
    vocab_chunk=DEFAULT_VOCAB_CHUNK,
    kv_dtype="bfloat16",
)

TEACHER_RUNTIMES: dict[str, dict[str, TeacherRuntime]] = {
    "e4b": {
        "bf16": TeacherRuntime(
            name="bf16", method="bf16", settings=_bf16_settings(), **_COMMON
        ),
        "int8": TeacherRuntime(
            name="int8",
            method="bitsandbytes-int8",
            settings=_int8_settings(),
            **_COMMON,
        ),
    },
    "12b_qat": {
        "nf4": TeacherRuntime(
            name="nf4",
            method="bitsandbytes-nf4",
            settings=_nf4_settings(),
            **_COMMON,
        ),
    },
}

# The registered primary runtime for each teacher.  E4B's bf16 path is the
# qualified default; 12B's four-bit NF4 path is the only resident 12B option.
PRIMARY_TEACHER_RUNTIMES: dict[str, str] = {
    "e4b": "bf16",
    "12b_qat": "nf4",
}

# Registered alternatives, used only after the primary fails qualification.
# ``q4_0_gguf`` is registered but deliberately has no exporter yet (review X3
# section 4): implement it only if 12B NF4 fails distribution qualification.
REGISTERED_FALLBACK_RUNTIMES: dict[str, str] = {
    "e4b": "int8",
    "12b_qat": "q4_0_gguf",
}


def registered_runtimes(teacher_key: str) -> tuple[str, ...]:
    """Return the implemented runtimes registered for ``teacher_key``."""

    key = str(teacher_key).lower()
    if key not in TEACHER_RUNTIMES:
        raise ValueError(
            f"unknown teacher {teacher_key!r}; expected one of "
            f"{tuple(TEACHER_RUNTIMES)}"
        )
    return tuple(TEACHER_RUNTIMES[key])


def teacher_runtime(teacher_key: str, name: str | None = None) -> TeacherRuntime:
    """Return a registered runtime; unknown or unimplemented runtimes fail closed."""

    key = str(teacher_key).lower()
    if key not in TEACHER_RUNTIMES:
        raise ValueError(
            f"unknown teacher {teacher_key!r}; expected one of "
            f"{tuple(TEACHER_RUNTIMES)}"
        )
    runtime = str(name or PRIMARY_TEACHER_RUNTIMES[key]).lower()
    try:
        return TEACHER_RUNTIMES[key][runtime]
    except KeyError:
        fallback = REGISTERED_FALLBACK_RUNTIMES.get(key)
        if runtime == fallback:
            raise NotImplementedError(
                f"teacher {key!r} fallback runtime {runtime!r} is registered but "
                "its exporter/loader is not implemented"
            ) from None
        raise ValueError(
            f"teacher {key!r} has no registered runtime {runtime!r}; "
            f"implemented runtimes: {registered_runtimes(key)}"
        ) from None


def software_versions() -> dict[str, str]:
    """Return the pinned-in-environment versions recorded in the identity."""

    global _SOFTWARE_VERSIONS
    if _SOFTWARE_VERSIONS is not None:
        return dict(_SOFTWARE_VERSIONS)
    versions = {"torch": _version("torch"), "transformers": _version("transformers")}
    try:
        import bitsandbytes as bnb  # noqa: F401

        versions["bitsandbytes"] = _version("bitsandbytes")
    except Exception:  # pragma: no cover - optional on CPU-only installs
        versions["bitsandbytes"] = "unavailable"
    _SOFTWARE_VERSIONS = versions
    return dict(versions)


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:  # pragma: no cover - hard dependency in production
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def runtime_identity(
    teacher_key: str,
    runtime: str | None = None,
    *,
    token_chunk: int | None = None,
    vocab_chunk: int | None = None,
    software: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Shorthand for the JSON identity of ``teacher_key``'s runtime."""

    return teacher_runtime(teacher_key, runtime).identity(
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        software=software,
    )


def identity_semantics(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the semantic subset of an identity (software versions excluded)."""

    keys = (
        "runtime",
        "method",
        "settings",
        "compute_dtype",
        "skipped_modules",
        "head_projection",
        "head_accumulation",
        "attention_implementation",
        "attention_backend_resolution",
        "kv_dtype",
    )
    return {key: identity.get(key) for key in keys}


def _normalise_versions(software: Any) -> dict[str, str]:
    if not isinstance(software, Mapping):
        return {}
    return {str(key): str(value) for key, value in software.items()}


def validate_runtime_identity(
    teacher_key: str,
    runtime: str,
    identity: Any,
) -> None:
    """Raise ``ValueError`` when ``identity`` is not the registered semantics."""

    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("teacher manifest is missing its runtime identity")
    expected_runtime = str(identity.get("runtime", ""))
    if expected_runtime != str(runtime):
        raise ValueError(
            f"runtime identity says {expected_runtime!r} but the teacher "
            f"runtime is {runtime!r}"
        )
    token_chunk = identity.get("token_chunk")
    vocab_chunk = identity.get("vocab_chunk")
    if not isinstance(token_chunk, int) or token_chunk < 1:
        raise ValueError("runtime identity has an invalid token_chunk")
    if not isinstance(vocab_chunk, int) or vocab_chunk < 1:
        raise ValueError("runtime identity has an invalid vocab_chunk")
    expected = runtime_identity(
        teacher_key,
        runtime,
        token_chunk=int(token_chunk),
        vocab_chunk=int(vocab_chunk),
        software=_normalise_versions(identity.get("software")),
    )
    actual_semantics = identity_semantics(identity)
    expected_semantics = identity_semantics(expected)
    if actual_semantics != expected_semantics:
        changed = [
            key
            for key in expected_semantics
            if actual_semantics.get(key) != expected_semantics.get(key)
        ]
        raise ValueError(
            "teacher runtime identity does not match the registered runtime: "
            + ", ".join(changed)
        )
    versions = _normalise_versions(identity.get("software"))
    current = software_versions()
    missing = [key for key in ("torch", "transformers") if not versions.get(key)]
    if missing:
        raise ValueError(
            "teacher runtime identity is missing software versions: "
            + ", ".join(missing)
        )
    changed_versions = [
        key for key in ("torch", "transformers", "bitsandbytes")
        if versions.get(key, "unavailable") != current.get(key, "unavailable")
    ]
    if changed_versions:
        raise ValueError(
            "teacher shards were built with a different software stack: "
            + ", ".join(changed_versions)
        )
