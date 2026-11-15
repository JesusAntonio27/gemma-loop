"""Crash-safe execution of the registered pilot stages.

Every child command goes through ``scripts/orchestrate.py``.  This module only
decides *what* to run: completed arms are skipped, partially finished arms are
redispatched (the orchestrator adds ``--resume auto``), and a teacher's shards
are precomputed once and reused by every arm that names that teacher.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from gemmaloop.teacher.runtime_freeze import (
    TeacherRuntimeFreeze,
    TeacherRuntimeFreezeError,
    TeacherRuntimeNotFrozen,
    default_teacher_runtime_freeze_path,
    load_teacher_runtime_freeze,
)

from .arms import ArmSpec

__all__ = [
    "EXCLUDED_RUNTIME_REASON",
    "EvaluationIdentityMismatch",
    "MonitorTeacherDataMissing",
    "RunIdentityMismatch",
    "TeacherRuntimeFreezeError",
    "TeacherRuntimeNotFrozen",
    "arm_run_dir",
    "build_monitor_precompute_command",
    "build_precompute_command",
    "build_train_command",
    "evaluation_path",
    "is_arm_complete",
    "monitor_teacher_cache_dir",
    "orchestrator_command",
    "registered_teacher_exclusions",
    "run_evaluation",
    "run_stage",
    "stage_actions",
    "teacher_cache_complete",
    "teacher_cache_dir",
]

# The exact record written for an arm whose teacher the registration excludes.
EXCLUDED_RUNTIME_REASON = "excluded: no qualified resident runtime (registered)"
_TEACHER_FAMILY_MARKERS = {
    "e4b": ("e4b", "e4 b"),
    "12b": ("12b", "12 b"),
}

ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
PRECOMPUTE_SCRIPT = ROOT / "scripts" / "precompute_teacher_topk.py"
ORCHESTRATE_SCRIPT = ROOT / "scripts" / "orchestrate.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate_pilot.py"
TEACHER_ARG = {"e4b": "e4b", "12b": "12b_qat"}
MANIFEST_NAME = "teacher_manifest.json"


class RunIdentityMismatch(RuntimeError):
    """A completed run directory was produced for a different pilot arm."""


class EvaluationIdentityMismatch(RuntimeError):
    """A stored evaluation belongs to a different arm, run or runtime."""


class MonitorTeacherDataMissing(RuntimeError):
    """The MONITOR teacher shards could not be produced before training."""


def arm_run_dir(root: str | Path, arm_id: str) -> Path:
    return Path(root) / "arms" / str(arm_id)


def teacher_cache_dir(root: str | Path, teacher: str) -> Path:
    return Path(root) / "teacher-shards" / str(teacher)


def registered_teacher_exclusions(prereg) -> dict[str, str]:
    """Teacher codes the round-8h data addendum declares excluded.

    The registration records the realized, and failed, resident runtimes in
    ``revision_round8h.data_addendum.teacher_runtime.excluded_teachers`` (for
    the registered pilot: both 12B runtimes).  When every runtime for a teacher
    family is excluded, the teacher has no qualified resident runtime and the
    teacher stage must not attempt it.  The exclusion is read from the loaded
    preregistration, never from a command-line flag.

    Returns a mapping of harness teacher code (``e4b``/``12b``) to the exact
    stage-output reason.  An unexpected missing runtime for a teacher that is
    *not* in this mapping still raises through
    :meth:`TeacherRuntimeFreeze.require`.
    """

    round8h = getattr(prereg, "revision_round8h", None)
    addendum = getattr(round8h, "data_addendum", None)
    teacher_runtime = getattr(addendum, "teacher_runtime", None)
    if teacher_runtime is None:
        return {}
    excluded_entries = [
        str(entry)
        for entry in (getattr(teacher_runtime, "excluded_teachers", None) or [])
    ]
    if not excluded_entries:
        return {}
    exclusions: dict[str, str] = {}
    for code, markers in _TEACHER_FAMILY_MARKERS.items():
        lowered = [entry.lower() for entry in excluded_entries]
        if any(marker in entry for entry in lowered for marker in markers):
            exclusions[code] = EXCLUDED_RUNTIME_REASON
    return exclusions


def monitor_teacher_cache_dir(root: str | Path, teacher: str) -> Path:
    """Teacher shards built on the MONITOR corpus (separate from heal rows)."""

    return teacher_cache_dir(root, teacher) / "monitor"


def evaluation_path(root: str | Path, arm_id: str) -> Path:
    return Path(root) / "eval" / str(arm_id) / "evaluation.json"


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_arm_complete(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    run_json = _read_json(directory / "run.json")
    if run_json.get("status") == "complete":
        return True
    done = _read_json(directory / "DONE.json")
    return done.get("status") == "complete"


def has_checkpoint(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    if not directory.is_dir():
        return False
    return any(
        child.is_dir() and child.name.startswith("checkpoint-")
        for child in directory.iterdir()
    )


def teacher_cache_complete(
    directory: str | Path,
    *,
    freeze: TeacherRuntimeFreeze | None = None,
    teacher: str | None = None,
) -> bool:
    """Return whether the cache is complete and built by the frozen runtime.

    Production callers pass the frozen record and the arm's teacher, so a cache
    built by E4B bf16 (failed qualification) or 12B NF4 (failed fidelity) is
    never reused for the qualified runtime.  Callers that omit ``freeze`` keep
    the structural check only; the harness always passes it.
    """

    directory = Path(directory)
    payload = _read_json(directory / MANIFEST_NAME)
    if not payload:
        return False
    if payload.get("complete") is False:
        return False
    if freeze is not None:
        manifest_teacher = payload.get("teacher") or {}
        manifest_runtime = str(manifest_teacher.get("runtime") or "")
        key = teacher
        if key is None:
            key = str(manifest_teacher.get("key") or "")
            if key:
                from gemmaloop.teacher.runtime_freeze import harness_teacher_key

                try:
                    key = harness_teacher_key(key)
                except TeacherRuntimeFreezeError:
                    return False
        if key is None:
            return False
        try:
            expected = freeze.runtime_for(key)
        except TeacherRuntimeNotFrozen:
            return False
        if manifest_runtime != expected:
            return False
    shards = payload.get("shards") or []
    if shards:
        for entry in shards:
            name = entry.get("file") if isinstance(entry, dict) else None
            if not name or not (directory / str(name)).is_file():
                return False
        return True
    return payload.get("rows_total") == 0


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_manifest(directory: str | Path) -> dict:
    return _read_json(Path(directory) / MANIFEST_NAME)


def expected_run_identity(
    arm: ArmSpec,
    *,
    teacher_data: str | Path | None,
    monitor_rows: str | Path | None = None,
    monitor_teacher_data: str | Path | None = None,
    teacher_runtime_freeze: TeacherRuntimeFreeze | None = None,
) -> dict:
    cache = None if teacher_data is None else Path(teacher_data)
    monitor_cache = (
        None if monitor_teacher_data is None else Path(monitor_teacher_data)
    )
    manifest = _teacher_manifest(cache) if cache is not None else {}
    frozen_runtime = (
        None
        if teacher_runtime_freeze is None
        else teacher_runtime_freeze.runtime_for(arm.teacher)
    )
    return {
        "pilot_arm": arm.arm_id,
        "arm": "L" if arm.kind == "L" else "O-H",
        "block": f"{arm.block[0]}-{arm.block[1]}",
        "bridge": arm.bridge,
        "teacher_data": None if cache is None else str(cache.resolve()),
        "monitor_rows": (
            None if monitor_rows is None else str(Path(monitor_rows).resolve())
        ),
        "monitor_teacher_data": (
            None if monitor_cache is None else str(monitor_cache.resolve())
        ),
        "teacher": dict(manifest.get("teacher") or {}),
        "teacher_runtime": frozen_runtime,
        "data_manifest_sha256": manifest.get("data_manifest_sha256"),
        "teacher_manifest_sha256": (
            _sha256_file(cache / MANIFEST_NAME)
            if cache is not None and (cache / MANIFEST_NAME).is_file()
            else None
        ),
    }


def run_identity_mismatches(
    run_dir: str | Path,
    arm: ArmSpec,
    *,
    teacher_data: str | Path | None,
    monitor_rows: str | Path | None = None,
    monitor_teacher_data: str | Path | None = None,
    teacher_runtime_freeze: TeacherRuntimeFreeze | None = None,
) -> list[str]:
    """Return why a completed run cannot be reused for ``arm``."""

    directory = Path(run_dir)
    run = _read_json(directory / "run.json")
    identity = _read_json(directory / "run_identity.json")
    if not run:
        return ["missing or unreadable run.json"]
    config = (
        identity.get("run_config")
        if isinstance(identity.get("run_config"), dict)
        else {}
    )
    expected = expected_run_identity(
        arm,
        teacher_data=teacher_data,
        monitor_rows=monitor_rows,
        monitor_teacher_data=monitor_teacher_data,
        teacher_runtime_freeze=teacher_runtime_freeze,
    )
    problems: list[str] = []

    stored_pilot_arm = config.get("pilot_arm") or identity.get("pilot_arm")
    if stored_pilot_arm != expected["pilot_arm"]:
        problems.append(
            f"pilot_arm={stored_pilot_arm!r} != {expected['pilot_arm']!r}"
        )
    stored_arm = config.get("arm") or run.get("arm")
    if stored_arm != expected["arm"]:
        problems.append(f"arm={stored_arm!r} != {expected['arm']!r}")
    stored_block = config.get("block")
    if stored_block is None and run.get("loop_block") is not None:
        block = run["loop_block"]
        stored_block = f"{block[0]}-{block[1]}"
    if stored_block != expected["block"]:
        problems.append(f"block={stored_block!r} != {expected['block']!r}")
    stored_bridge = config.get("bridge") or run.get("bridge")
    if arm.kind == "L" and stored_bridge != expected["bridge"]:
        problems.append(f"bridge={stored_bridge!r} != {expected['bridge']!r}")

    if expected["teacher_data"] is not None:
        stored_teacher_data = config.get("teacher_data")
        if stored_teacher_data is not None and (
            str(Path(stored_teacher_data).resolve()) != expected["teacher_data"]
        ):
            problems.append(
                f"teacher_data={stored_teacher_data!r} != "
                f"{expected['teacher_data']!r}"
            )
    for field in ("monitor_rows", "monitor_teacher_data"):
        if expected[field] is None:
            continue
        stored = config.get(field)
        if stored is None or Path(stored).resolve() != Path(expected[field]):
            problems.append(f"{field}={stored!r} != {expected[field]!r}")
    stored_teacher = (run.get("teacher") or {}).get("teacher")
    if expected["teacher"] and stored_teacher != expected["teacher"]:
        problems.append(
            f"teacher={stored_teacher!r} != {expected['teacher']!r}"
        )
    if expected["teacher_runtime"]:
        teacher_record = run.get("teacher") or {}
        nested_teacher = teacher_record.get("teacher") or {}
        # train.py writes the runtime inside the nested teacher manifest record
        # (``run["teacher"]["teacher"]["runtime"]``); ``run_config`` has no
        # runtime key because it is not part of the frozen resume identity.
        stored_runtime = (
            config.get("teacher_runtime")
            or nested_teacher.get("runtime")
            or teacher_record.get("runtime")
        )
        if stored_runtime != expected["teacher_runtime"]:
            problems.append(
                f"teacher_runtime={stored_runtime!r} != "
                f"{expected['teacher_runtime']!r}"
            )
    if expected["data_manifest_sha256"]:
        stored_data = run.get("data_manifest_sha256") or (
            (run.get("teacher") or {}).get("data_manifest_sha256")
        )
        if stored_data != expected["data_manifest_sha256"]:
            problems.append(
                f"data_manifest_sha256={stored_data!r} != "
                f"{expected['data_manifest_sha256']!r}"
            )
    expected_manifest = expected["teacher_manifest_sha256"]
    if expected_manifest:
        stored_manifest = (run.get("teacher") or {}).get("manifest_sha256")
        if stored_manifest != expected_manifest:
            problems.append(
                f"teacher_manifest_sha256={stored_manifest!r} != "
                f"{expected_manifest!r}"
            )
    if not run.get("run_identity_hash"):
        problems.append("run.json has no run_identity_hash")
    return problems


def evaluation_identity_mismatches(
    payload: dict,
    arm: ArmSpec,
    *,
    run_dir: str | Path,
    teacher_data: str | Path | None,
) -> list[str]:
    del teacher_data
    problems: list[str] = []
    if payload.get("arm_id") != arm.arm_id:
        problems.append(
            f"evaluation arm_id={payload.get('arm_id')!r} != {arm.arm_id!r}"
        )
    stored_arm = payload.get("arm") or {}
    expected = arm.to_dict()
    for field in ("block", "bridge", "teacher", "registered_teacher"):
        if stored_arm.get(field) != expected.get(field):
            problems.append(
                f"evaluation {field}={stored_arm.get(field)!r} != "
                f"{expected.get(field)!r}"
            )
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        problems.append("evaluation has no runtime identity")
    elif runtime.get("teacher") != arm.teacher:
        problems.append(
            f"evaluation runtime teacher={runtime.get('teacher')!r} != "
            f"{arm.teacher!r}"
        )
    run = _read_json(Path(run_dir) / "run.json")
    for field in (
        "run_identity_hash",
        "run_config_sha256",
        "data_manifest_sha256",
    ):
        stored = payload.get(field)
        if (
            stored is not None
            and run.get(field) is not None
            and stored != run.get(field)
        ):
            problems.append(
                f"evaluation {field}={stored!r} != run {run.get(field)!r}"
            )
    return problems


def validate_run_identity(
    arm: ArmSpec,
    *,
    run_dir: str | Path,
    teacher_data: str | Path | None,
) -> None:
    problems = run_identity_mismatches(
        run_dir, arm, teacher_data=teacher_data
    )
    if problems:
        raise RunIdentityMismatch(
            f"completed run {run_dir} cannot be reused for {arm.arm_id}: "
            + "; ".join(problems)
        )


def _precompute_command(
    teacher: str,
    *,
    runtime: str,
    data: str | Path,
    cache_dir: str | Path,
    max_length: int | None = None,
) -> list[str]:
    if not runtime:
        raise TeacherRuntimeNotFrozen(
            f"teacher {teacher!r} has no frozen runtime; refusing to precompute"
        )
    command = [
        sys.executable,
        str(PRECOMPUTE_SCRIPT),
        "--teacher",
        TEACHER_ARG.get(str(teacher), str(teacher)),
        "--runtime",
        str(runtime),
        "--data",
        str(Path(data)),
        "--out",
        str(Path(cache_dir)),
    ]
    if max_length is not None:
        command.extend(["--max-length", str(int(max_length))])
    return command


def build_precompute_command(
    teacher: str,
    *,
    runtime: str,
    heal_data: str | Path,
    cache_dir: str | Path,
    max_length: int | None = None,
) -> list[str]:
    return _precompute_command(
        teacher,
        runtime=runtime,
        data=heal_data,
        cache_dir=cache_dir,
        max_length=max_length,
    )


def build_monitor_precompute_command(
    teacher: str,
    *,
    runtime: str,
    monitor_rows: str | Path,
    cache_dir: str | Path,
    max_length: int | None = None,
) -> list[str]:
    """Precompute teacher shards for the MONITOR rows with the arm's teacher."""

    return _precompute_command(
        teacher,
        runtime=runtime,
        data=monitor_rows,
        cache_dir=cache_dir,
        max_length=max_length,
    )


def build_train_command(
    arm: ArmSpec,
    *,
    prereg_path: str | Path,
    heal_data: str | Path,
    run_dir: str | Path,
    seed: int,
    teacher_data: str | Path | None = None,
    monitor_rows: str | Path | None = None,
    monitor_teacher_data: str | Path | None = None,
    teacher_runtime_freeze: str | Path | None = None,
    require_tag: str | None = None,
    force: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--pilot-arm",
        arm.arm_id,
        "--prereg",
        str(Path(prereg_path)),
        "--data",
        str(Path(heal_data)),
        "--seed",
        str(int(seed)),
        "--out",
        str(Path(run_dir)),
    ]
    if teacher_data is not None:
        command.extend(["--teacher-data", str(Path(teacher_data))])
    if monitor_rows is not None:
        command.extend(["--monitor-rows", str(Path(monitor_rows))])
    if monitor_teacher_data is not None:
        command.extend(["--monitor-teacher-data", str(Path(monitor_teacher_data))])
    if teacher_runtime_freeze is not None:
        command.extend(
            ["--teacher-runtime-freeze", str(Path(teacher_runtime_freeze))]
        )
    if require_tag is not None:
        command.extend(["--require-tag", str(require_tag)])
    if force:
        command.append("--rerun-complete")
    return command


def orchestrator_command(
    name: str | None,
    out_dir: str | Path,
    child_command: Sequence[str],
    *,
    backoff_initial: float = float(60),
    ack_abort: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ORCHESTRATE_SCRIPT),
        "--out",
        str(Path(out_dir)),
        "--name",
        str(name or Path(out_dir).name),
        "--backoff-initial",
        str(float(backoff_initial)),
    ]
    if ack_abort is not None:
        command.extend(["--ack-abort", str(ack_abort)])
    command.extend(["--", *[str(part) for part in child_command]])
    return command


def stage_actions(
    arms: Sequence[ArmSpec],
    *,
    root: str | Path,
    force: bool = False,
) -> list[dict]:
    actions = []
    for arm in arms:
        run_dir = arm_run_dir(root, arm.arm_id)
        if is_arm_complete(run_dir) and not force:
            actions.append({"arm_id": arm.arm_id, "action": "skipped", "run_dir": str(run_dir)})
        else:
            actions.append(
                {
                    "arm_id": arm.arm_id,
                    "action": "train",
                    "run_dir": str(run_dir),
                    "resume": has_checkpoint(run_dir),
                }
            )
    return actions


def _evaluation_payload(path: Path) -> dict:
    payload = _read_json(path)
    return payload if payload.get("status") == "complete" else {}


def run_evaluation(
    arm: ArmSpec,
    *,
    root: str | Path,
    prereg_path: str | Path,
    heal_data: str | Path,
    fit_data: str | Path,
    test_data: str | Path,
    execute: Callable[[Sequence[str], Path], int] | None = None,
    force: bool = False,
    teacher_runtime_freeze: str | Path | None = None,
) -> dict:
    """Run one arm's evaluation as an orchestrated GPU child.

    The child owns the GPU lock and the allocator cap; the orchestrator makes
    the per-arm output crash-safe.  A complete ``evaluation.json`` is reused.
    """

    execute = execute or _default_execute
    root = Path(root)
    output = evaluation_path(root, arm.arm_id)
    cache_for_identity = teacher_cache_dir(root, arm.teacher)
    freeze_path = (
        default_teacher_runtime_freeze_path()
        if teacher_runtime_freeze is None
        else Path(teacher_runtime_freeze)
    )
    freeze = load_teacher_runtime_freeze(freeze_path)
    if not force:
        complete = _evaluation_payload(output)
        if complete:
            problems = evaluation_identity_mismatches(
                complete,
                arm,
                run_dir=arm_run_dir(root, arm.arm_id),
                teacher_data=cache_for_identity,
            )
            if problems:
                raise EvaluationIdentityMismatch(
                    f"stored evaluation {output} cannot be reused for "
                    f"{arm.arm_id}: " + "; ".join(problems)
                )
            return {
                "arm_id": arm.arm_id,
                "action": "skipped",
                "returncode": 0,
                "evaluation": complete,
                "path": str(output),
            }
    out_dir = output.parent
    cache = teacher_cache_dir(root, arm.teacher)
    child = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--arm-id",
        arm.arm_id,
        "--prereg",
        str(Path(prereg_path)),
        "--run-dir",
        str(arm_run_dir(root, arm.arm_id)),
        "--heal-data",
        str(Path(heal_data)),
        "--fit-data",
        str(Path(fit_data)),
        "--test-data",
        str(Path(test_data)),
        "--out",
        str(output),
    ]
    if teacher_cache_complete(cache, freeze=freeze, teacher=arm.teacher):
        child.extend(["--teacher-data", str(cache)])
    command = orchestrator_command(f"eval-{arm.arm_id}", out_dir, child)
    code = execute(command, out_dir)
    payload = _evaluation_payload(output) if code == 0 else {}
    return {
        "arm_id": arm.arm_id,
        "action": "evaluated" if code == 0 and payload else "eval_failed",
        "returncode": int(code),
        "evaluation": payload or None,
        "path": str(output),
    }


def _default_execute(command: Sequence[str], out: Path) -> int:
    started = time.time()
    completed = subprocess.run(list(command))
    elapsed = time.time() - started
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "pilot_wall_seconds.txt").write_text(
            f"{elapsed:.3f}\n", encoding="utf-8"
        )
    except OSError:
        pass
    return int(completed.returncode)


def run_stage(
    arms: Sequence[ArmSpec],
    *,
    root: str | Path,
    prereg_path: str | Path,
    heal_data: str | Path,
    seed: int,
    execute: Callable[[Sequence[str], Path], int] | None = None,
    force: bool = False,
    reserved_ids_path: str | Path | None = None,
    monitor_rows: str | Path | None = None,
    max_length: int | None = None,
    require_tag: str | None = None,
    ack_aborts: Mapping[str, str] | None = None,
    teacher_runtime_freeze: str | Path | None = None,
    prereg=None,
) -> list[dict]:
    """Run or resume every arm in ``arms``; return one action record per arm.

    ``reserved_ids_path`` is exported as the heal monitor's registered
    ``GEMMALOOP_HEAL_MONITOR_RESERVED`` environment variable for the training
    child, so the run can prove pilot split disjointness.  ``monitor_rows`` is
    the dedicated MONITOR corpus: its teacher shards are precomputed with the
    same teacher/runtime as the arm before any training child starts, and a
    failed precompute stops the stage with :class:`MonitorTeacherDataMissing`.
    ``ack_aborts`` maps an arm id to the monitor abort id acknowledged for that
    arm; it is passed to that arm's training orchestrator.

    ``prereg`` is the already-loaded :class:`gemmaloop.prereg.Prereg`; when it
    is omitted the file at ``prereg_path`` is loaded.  Its round-8h data
    addendum is the only source of teacher exclusions: an arm whose teacher the
    registration declares excluded is recorded with
    :data:`EXCLUDED_RUNTIME_REASON` and is never precomputed, trained or
    handed to evaluation.  A teacher that is not declared excluded still has
    to have a qualified frozen runtime before any child starts.
    """

    execute = execute or _default_execute
    root = Path(root)
    freeze_path = (
        default_teacher_runtime_freeze_path()
        if teacher_runtime_freeze is None
        else Path(teacher_runtime_freeze)
    )
    freeze = load_teacher_runtime_freeze(freeze_path)
    if prereg is None:
        from gemmaloop.prereg import load_prereg

        prereg = load_prereg(prereg_path)
    exclusions = registered_teacher_exclusions(prereg)
    # Fail before any GPU precompute if a selected teacher has no qualified
    # runtime and the registration does not declare that teacher excluded.
    for arm in arms:
        if arm.teacher in exclusions:
            continue
        freeze.require(arm.teacher)
    results: list[dict] = []
    ack_aborts = dict(ack_aborts or {})
    for arm in arms:
        run_dir = arm_run_dir(root, arm.arm_id)
        if arm.teacher in exclusions:
            results.append(
                {
                    "arm_id": arm.arm_id,
                    "action": "excluded",
                    "reason": exclusions[arm.teacher],
                    "run_dir": str(run_dir),
                    "returncode": None,
                    "wall_seconds": None,
                }
            )
            continue
        runtime = freeze.require(arm.teacher)
        cache = teacher_cache_dir(root, arm.teacher)
        monitor_cache = (
            None
            if monitor_rows is None
            else monitor_teacher_cache_dir(root, arm.teacher)
        )
        was_complete = is_arm_complete(run_dir)
        if was_complete:
            problems = run_identity_mismatches(
                run_dir,
                arm,
                teacher_data=cache,
                monitor_rows=monitor_rows,
                monitor_teacher_data=monitor_cache,
                teacher_runtime_freeze=freeze,
            )
            if problems and not force:
                raise RunIdentityMismatch(
                    f"completed run {run_dir} cannot be reused for "
                    f"{arm.arm_id}: " + "; ".join(problems)
                )
            if not problems and not force:
                if teacher_cache_complete(
                    cache, freeze=freeze, teacher=arm.teacher
                ):
                    results.append(
                        {
                            "arm_id": arm.arm_id,
                            "action": "skipped",
                            "run_dir": str(run_dir),
                            "returncode": 0,
                            "wall_seconds": None,
                        }
                    )
                    continue
                # The run is reusable but its teacher shards are gone; rebuild
                # them before accepting the arm so evaluation keeps its KL.
                skip_training = True
            else:
                skip_training = False
        else:
            skip_training = False

        precompute_seconds = None
        if not teacher_cache_complete(cache, freeze=freeze, teacher=arm.teacher):
            command = orchestrator_command(
                f"teacher-{arm.teacher}",
                cache,
                build_precompute_command(
                    arm.teacher,
                    runtime=runtime,
                    heal_data=heal_data,
                    cache_dir=cache,
                    max_length=max_length,
                ),
            )
            started = time.time()
            code = execute(command, cache)
            precompute_seconds = time.time() - started
            if code != 0:
                results.append(
                    {
                        "arm_id": arm.arm_id,
                        "action": "teacher_failed",
                        "run_dir": str(run_dir),
                        "returncode": int(code),
                        "wall_seconds": precompute_seconds,
                    }
                )
                continue

        if skip_training:
            results.append(
                {
                    "arm_id": arm.arm_id,
                    "action": "skipped",
                    "run_dir": str(run_dir),
                    "returncode": 0,
                    "wall_seconds": None,
                }
            )
            continue

        monitor_precompute_seconds = None
        if monitor_rows is not None and not teacher_cache_complete(
            monitor_cache, freeze=freeze, teacher=arm.teacher
        ):
            command = orchestrator_command(
                f"monitor-teacher-{arm.teacher}",
                monitor_cache,
                build_monitor_precompute_command(
                    arm.teacher,
                    runtime=runtime,
                    monitor_rows=monitor_rows,
                    cache_dir=monitor_cache,
                    max_length=max_length,
                ),
            )
            started = time.time()
            code = execute(command, monitor_cache)
            monitor_precompute_seconds = time.time() - started
            if code != 0 or not teacher_cache_complete(
                monitor_cache, freeze=freeze, teacher=arm.teacher
            ):
                raise MonitorTeacherDataMissing(
                    f"the MONITOR teacher shards for {arm.teacher} could not be "
                    f"built from {monitor_rows} (exit {code}); refusing to start "
                    "the training child"
                )

        command = orchestrator_command(
            arm.arm_id,
            run_dir,
            build_train_command(
                arm,
                prereg_path=prereg_path,
                heal_data=heal_data,
                run_dir=run_dir,
                seed=seed,
                teacher_data=cache,
                monitor_rows=monitor_rows,
                monitor_teacher_data=monitor_cache,
                teacher_runtime_freeze=freeze_path,
                require_tag=require_tag,
                # --force means "retrain a completed arm".  An incomplete arm
                # must keep auto-resume instead of restarting from zero.
                force=force and was_complete,
            ),
            ack_abort=ack_aborts.get(arm.arm_id),
        )
        started = time.time()
        import os

        key = "GEMMALOOP_HEAL_MONITOR_RESERVED"
        previous = os.environ.get(key)
        if reserved_ids_path is not None:
            os.environ[key] = str(reserved_ids_path)
        try:
            code = execute(command, run_dir)
        finally:
            if reserved_ids_path is not None:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        wall = time.time() - started
        record = {
            "arm_id": arm.arm_id,
            "action": "trained" if code == 0 else "train_failed",
            "run_dir": str(run_dir),
            "returncode": int(code),
            "wall_seconds": wall,
        }
        if precompute_seconds is not None:
            record["teacher_precompute_seconds"] = precompute_seconds
        if monitor_precompute_seconds is not None:
            record["monitor_teacher_precompute_seconds"] = (
                monitor_precompute_seconds
            )
        results.append(record)
    return results
