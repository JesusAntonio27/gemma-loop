"""The frozen, qualified teacher runtime for each teacher.

The prereg freezes one runtime per teacher before the pilot.  Qualification can
fail a runtime (E4B bf16 failed on headroom, 12B NF4 failed on fidelity), so
the harness must not fall back to the primary runtime implicitly.  This module
owns the pinned record that binds, per teacher, the qualified runtime name and
the qualification JSON that produced it (path, SHA-256 and decision ``pass``).

The record is a small JSON file (default
``<artifacts>/qualification/teacher_runtime_freeze.json``) built by
``scripts/freeze_teacher_runtime.py`` from the real qualification outputs.  It
is deliberately separate from ``gemmaloop.teacher.runtime``: that module lists
what is implemented, this one records what passed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from gemmaloop.paths import artifacts_dir
from gemmaloop.teacher.runtime import registered_runtimes

__all__ = [
    "DEFAULT_FREEZE_NAME",
    "FREEZE_FORMAT",
    "FrozenTeacherRuntime",
    "TeacherRuntimeFreeze",
    "TeacherRuntimeFreezeError",
    "TeacherRuntimeNotFrozen",
    "build_teacher_runtime_freeze",
    "default_teacher_runtime_freeze_path",
    "harness_teacher_key",
    "load_teacher_runtime_freeze",
    "qualification_record",
]

FREEZE_FORMAT = "gemmaloop-teacher-runtime-freeze-v1"
DEFAULT_FREEZE_NAME = "teacher_runtime_freeze.json"

# Qualification records use the registered teacher key (``12b_qat``); the
# pilot arms and cache directories use the short label (``12b``).
HARNESS_TEACHER_KEYS = {"e4b": "e4b", "12b": "12b_qat"}
QUALIFICATION_TO_HARNESS = {value: key for key, value in HARNESS_TEACHER_KEYS.items()}


class TeacherRuntimeFreezeError(RuntimeError):
    """The frozen teacher-runtime record is missing, invalid or not qualified."""


class TeacherRuntimeNotFrozen(TeacherRuntimeFreezeError):
    """A teacher has no passing qualified runtime in the freeze record."""


def default_teacher_runtime_freeze_path() -> Path:
    return artifacts_dir() / "qualification" / DEFAULT_FREEZE_NAME


def harness_teacher_key(teacher_key: str) -> str:
    """Map a registered teacher key (``12b_qat``) to the arm key (``12b``)."""

    key = str(teacher_key).lower()
    if key in QUALIFICATION_TO_HARNESS:
        return QUALIFICATION_TO_HARNESS[key]
    if key in HARNESS_TEACHER_KEYS:
        return key
    raise TeacherRuntimeFreezeError(f"unknown teacher {teacher_key!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualification_record(path: str | Path) -> dict[str, Any]:
    """Read one qualification JSON and extract the pass/fail decision.

    The record must name its teacher and runtime; the decision is the top-level
    ``decision`` object written by ``scripts/qualify_teacher.py`` (with the
    acceptance-table scopes as a fallback for older records).
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherRuntimeFreezeError(
            f"cannot read teacher qualification {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TeacherRuntimeFreezeError(f"{source} is not a JSON object")
    teacher = payload.get("teacher")
    if not isinstance(teacher, Mapping):
        raise TeacherRuntimeFreezeError(
            f"{source} has no teacher record"
        )
    teacher_key = str(teacher.get("key") or "")
    runtime = str(teacher.get("runtime") or teacher.get("quant") or "")
    if not teacher_key or not runtime:
        raise TeacherRuntimeFreezeError(
            f"{source} does not name both a teacher key and a runtime"
        )
    decision = _decision(payload)
    return {
        "teacher_key": teacher_key,
        "teacher": harness_teacher_key(teacher_key),
        "runtime": runtime,
        "decision": decision,
        "qualification_json": str(source.resolve()),
        "qualification_sha256": _sha256_file(source),
    }


def _decision(payload: Mapping[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, Mapping):
        if decision.get("passed") is True or str(decision.get("decision")) == "pass":
            return "pass"
        return str(decision.get("decision") or "fail")
    if isinstance(decision, str) and decision:
        return decision
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        for scope in ("expanded", "screen"):
            entry = acceptance.get(scope)
            if not isinstance(entry, Mapping):
                continue
            if entry.get("passed") is True or str(entry.get("decision")) == "pass":
                return "pass"
            return str(entry.get("decision") or "fail")
    return "unknown"


def _verify_qualification(frozen: FrozenTeacherRuntime) -> None:
    """Re-read the bound qualification JSON on every load.

    A freeze file is only meaningful together with the exact qualification
    output that produced it.  A stale record, a hand edit or a replaced
    qualification file must fail closed instead of silently binding a runtime
    that no longer passes.
    """

    source = Path(frozen.qualification_json)
    if not source.is_file():
        raise TeacherRuntimeFreezeError(
            f"teacher {frozen.teacher!r} qualification JSON {source} is missing"
        )
    actual = _sha256_file(source)
    if actual != frozen.qualification_sha256:
        raise TeacherRuntimeFreezeError(
            f"teacher {frozen.teacher!r} qualification JSON {source} changed "
            f"since the freeze was written (sha256 {actual} != "
            f"{frozen.qualification_sha256})"
        )
    record = qualification_record(source)
    if record["decision"] != "pass":
        raise TeacherRuntimeFreezeError(
            f"teacher {frozen.teacher!r} qualification JSON {source} no longer "
            f"passes (decision {record['decision']!r})"
        )
    if record["teacher"] != frozen.teacher:
        raise TeacherRuntimeFreezeError(
            f"teacher {frozen.teacher!r} qualification JSON {source} names "
            f"teacher {record['teacher']!r}"
        )
    if record["runtime"] != frozen.runtime:
        raise TeacherRuntimeFreezeError(
            f"teacher {frozen.teacher!r} qualification JSON {source} names "
            f"runtime {record['runtime']!r}, not {frozen.runtime!r}"
        )


@dataclass(frozen=True)
class FrozenTeacherRuntime:
    teacher: str
    teacher_key: str
    runtime: str
    qualification_json: str
    qualification_sha256: str
    decision: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_key": self.teacher_key,
            "runtime": self.runtime,
            "qualification_json": self.qualification_json,
            "qualification_sha256": self.qualification_sha256,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, teacher: str, payload: Mapping[str, Any]) -> "FrozenTeacherRuntime":
        return cls(
            teacher=str(teacher),
            teacher_key=str(payload.get("teacher_key") or HARNESS_TEACHER_KEYS.get(str(teacher), "")),
            runtime=str(payload.get("runtime") or ""),
            qualification_json=str(payload.get("qualification_json") or ""),
            qualification_sha256=str(payload.get("qualification_sha256") or ""),
            decision=str(payload.get("decision") or ""),
        )


@dataclass(frozen=True)
class TeacherRuntimeFreeze:
    teachers: Mapping[str, FrozenTeacherRuntime] = field(default_factory=dict)
    format: str = FREEZE_FORMAT

    def runtime_for(self, teacher: str) -> str:
        entry = self.teachers.get(str(teacher))
        if entry is None:
            raise TeacherRuntimeNotFrozen(
                f"teacher {teacher!r} has no qualified runtime in the frozen "
                "teacher-runtime record; refusing to precompute or train"
            )
        if entry.decision != "pass":
            raise TeacherRuntimeNotFrozen(
                f"teacher {teacher!r} runtime {entry.runtime!r} is not "
                f"qualified (decision {entry.decision!r})"
            )
        return entry.runtime

    def runtime_for_teacher_key(self, teacher_key: str) -> str:
        return self.runtime_for(harness_teacher_key(teacher_key))

    def require(self, teacher: str) -> str:
        return self.runtime_for(teacher)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "teachers": {
                teacher: entry.to_dict()
                for teacher, entry in sorted(self.teachers.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherRuntimeFreeze":
        if payload.get("format") != FREEZE_FORMAT:
            raise TeacherRuntimeFreezeError(
                f"unsupported teacher-runtime freeze format "
                f"{payload.get('format')!r}; expected {FREEZE_FORMAT!r}"
            )
        raw = payload.get("teachers")
        if not isinstance(raw, Mapping) or not raw:
            raise TeacherRuntimeFreezeError(
                "teacher-runtime freeze has no teacher entries"
            )
        teachers: dict[str, FrozenTeacherRuntime] = {}
        for teacher, entry in raw.items():
            if not isinstance(entry, Mapping):
                raise TeacherRuntimeFreezeError(
                    f"teacher-runtime freeze entry {teacher!r} is not an object"
                )
            frozen = FrozenTeacherRuntime.from_dict(str(teacher), entry)
            if frozen.decision != "pass":
                raise TeacherRuntimeFreezeError(
                    f"teacher {teacher!r} runtime {frozen.runtime!r} is not "
                    f"qualified (decision {frozen.decision!r})"
                )
            if not frozen.qualification_json or not frozen.qualification_sha256:
                raise TeacherRuntimeFreezeError(
                    f"teacher {teacher!r} does not bind its qualification JSON "
                    "path and SHA-256"
                )
            if len(frozen.qualification_sha256) != 64:
                raise TeacherRuntimeFreezeError(
                    f"teacher {teacher!r} has an invalid qualification SHA-256"
                )
            if harness_teacher_key(frozen.teacher_key) != str(teacher):
                raise TeacherRuntimeFreezeError(
                    f"teacher-runtime freeze entry {teacher!r} has teacher key "
                    f"{frozen.teacher_key!r}, which does not map back to that "
                    "entry"
                )
            _verify_qualification(frozen)
            allowed = registered_runtimes(frozen.teacher_key)
            if frozen.runtime not in allowed:
                raise TeacherRuntimeFreezeError(
                    f"teacher {teacher!r} runtime {frozen.runtime!r} is not "
                    f"implemented; registered runtimes: {allowed}"
                )
            teachers[str(teacher)] = frozen
        return cls(teachers=teachers)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target


def load_teacher_runtime_freeze(
    path: str | Path | None = None,
) -> TeacherRuntimeFreeze:
    """Load and structurally validate the pinned record; fail closed."""

    source = default_teacher_runtime_freeze_path() if path is None else Path(path)
    if not source.is_file():
        raise TeacherRuntimeFreezeError(
            f"teacher-runtime freeze not found at {source}; build it with "
            "scripts/freeze_teacher_runtime.py from the passing qualification "
            "outputs"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherRuntimeFreezeError(
            f"cannot read teacher-runtime freeze {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TeacherRuntimeFreezeError(f"{source} is not a JSON object")
    return TeacherRuntimeFreeze.from_dict(payload)


def build_teacher_runtime_freeze(
    qualification_paths: Iterable[str | Path],
    *,
    out: str | Path | None = None,
) -> TeacherRuntimeFreeze:
    """Build the pinned record from qualification outputs.

    Every teacher named by an input must have at least one passing runtime.  A
    teacher whose only inputs failed (12B NF4 today) is refused, so the record
    cannot silently omit it or bind a failed runtime.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in qualification_paths:
        record = qualification_record(path)
        grouped.setdefault(record["teacher"], []).append(record)
    if not grouped:
        raise TeacherRuntimeFreezeError(
            "no qualification JSON was supplied; refusing to write an empty freeze"
        )
    teachers: dict[str, FrozenTeacherRuntime] = {}
    failures: list[str] = []
    for teacher, records in sorted(grouped.items()):
        passing = [record for record in records if record["decision"] == "pass"]
        if not passing:
            details = ", ".join(
                f"{record['runtime']}={record['decision']} ({record['qualification_json']})"
                for record in records
            )
            failures.append(f"teacher {teacher!r} has no passing runtime: {details}")
            continue
        runtimes = {record["runtime"] for record in passing}
        if len(runtimes) > 1:
            raise TeacherRuntimeFreezeError(
                f"teacher {teacher!r} has multiple passing runtimes {sorted(runtimes)}; "
                "the freeze must name exactly one"
            )
        record = passing[0]
        frozen = FrozenTeacherRuntime(
            teacher=teacher,
            teacher_key=record["teacher_key"],
            runtime=record["runtime"],
            qualification_json=record["qualification_json"],
            qualification_sha256=record["qualification_sha256"],
            decision="pass",
        )
        # Validate the runtime is implemented before accepting it.
        if frozen.runtime not in registered_runtimes(frozen.teacher_key):
            raise TeacherRuntimeFreezeError(
                f"teacher {teacher!r} passing runtime {frozen.runtime!r} is not "
                "implemented"
            )
        teachers[teacher] = frozen
    if failures:
        raise TeacherRuntimeFreezeError("; ".join(failures))
    freeze = TeacherRuntimeFreeze(teachers=teachers)
    if out is not None:
        freeze.write(out)
    return freeze
