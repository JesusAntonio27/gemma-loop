"""Run the registered pilot: teacher step, block stage, selection and report.

Arm ids, teacher, block, bridge and the 3M-token pilot budget come from
``prereg.yaml``; the command line only says where the data and outputs live.
Every training/precompute child runs through ``scripts/orchestrate.py`` so the
pilot is crash-safe and resumable, and only one GPU job runs at a time.

Typical use::

    # The evaluation splits must be rendered with the training tokenisation
    # first: raw Stage 0 trace rows have no input_ids.
    uv run python scripts/render_pilot_eval_rows.py \
        --stage0 artifacts/stage0 --out artifacts/data/pilot-eval
    uv run python scripts/run_pilot.py \
        --heal-data artifacts/data/pilot/heal \
        --monitor-rows artifacts/stage0/monitor/monitor_rows.jsonl \
        --fit-data artifacts/data/pilot-eval/pilot_tfit.jsonl \
        --test-data artifacts/data/pilot-eval/pilot_selection.jsonl \
        --require-tag prereg-v1 \
        --out artifacts/pilot \
        --stage all

Stages can also be run separately (``--stage teacher`` then ``--stage block``).
The overnight order is **teacher first, then block**: the teacher stage trains
the registered E4B 25-29 pair (the 12B pair is recorded as excluded by the
round-8h data addendum and is never trained), and the block stage then trains
the seven pre-listed block arms.  Both stages write to the same ``--out`` so
selection sees the E4B 25-29 pair from the teacher stage.  The block stage
always trains the pre-listed E4B arms; the teacher-step winner only labels the
final recipe and is recorded in ``<out>/teacher_step.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.pilot.arms import (  # noqa: E402
    ArmSpec,
    block_stage_arms,
    pilot_arms,
    teacher_stage_arms,
)
from gemmaloop.pilot.evaluate import (  # noqa: E402
    build_arm_summary,
    load_eval_rows,
    read_training_summary,
    teacher_shard_stats,
)
from gemmaloop.pilot.harness import (  # noqa: E402
    EvaluationIdentityMismatch,
    MonitorTeacherDataMissing,
    RunIdentityMismatch,
    TeacherRuntimeFreezeError,
    arm_run_dir,
    is_arm_complete,
    registered_teacher_exclusions,
    run_evaluation,
    run_stage,
    teacher_cache_dir,
    teacher_cache_complete,
)
from gemmaloop.pilot.qualification import load_qualification  # noqa: E402
from gemmaloop.pilot.report import write_report  # noqa: E402
from gemmaloop.pilot.splits import (  # noqa: E402
    SplitOverlapError,
    assert_disjoint_splits,
    reserved_ids_payload,
)
from gemmaloop.pilot.selection import (  # noqa: E402
    select_block_rule_b,
    select_single_teacher,
    select_teacher,
)
from gemmaloop.pilot.statistics import (  # noqa: E402
    DEFAULT_ITERS,
    DEFAULT_SEED,
    cluster_mean,
)
from gemmaloop.prereg import default_prereg_path, load_prereg  # noqa: E402
from gemmaloop.stage0.shortfall import (  # noqa: E402
    ShortfallReadinessError,
    check_freeze_readiness,
    check_pilot_inputs,
)
from gemmaloop.train.checkpoint import atomic_write_json  # noqa: E402
from gemmaloop.train.frozen import (  # noqa: E402
    FrozenRunError,
    assert_code_under,
    check_frozen_run,
)

RESULTS_FILE = "pilot_results.json"
TEACHER_STEP_FILE = "teacher_step.json"
SELECTION_FILE = "selection.json"
SELECTION_PENDING_FILE = "selection_pending.json"


def plan_stage(prereg, stage: str) -> dict:
    """The registered plan for one stage (teacher, block or all)."""

    arms = pilot_arms(prereg)
    return {
        "arms": arms,
        "teacher_stage": teacher_stage_arms(arms),
        "block_stage": block_stage_arms(arms),
        "required": _required_for_selection(stage, arms),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--stage", choices=("teacher", "block", "all"), default="all")
    parser.add_argument("--heal-data", required=True, type=Path)
    parser.add_argument("--fit-data", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, default=None)
    parser.add_argument("--monitor-data", type=Path, default=None)
    parser.add_argument(
        "--monitor-rows",
        type=Path,
        default=None,
        help=(
            "heal-format MONITOR JSONL from the dedicated Stage 0 monitor "
            "bucket (for example <stage0>/monitor/monitor_rows.jsonl); its "
            "problem ids must match the reserved split map"
        ),
    )
    parser.add_argument(
        "--require-tag",
        default=None,
        help=(
            "required frozen-run tag; it is checked before any teacher "
            "precompute and passed to every train child"
        ),
    )
    parser.add_argument(
        "--shortfall-freeze",
        type=Path,
        default=None,
        help=(
            "pilot-shortfall freeze index; defaults to "
            "prereg/pilot-shortfall-freeze.json (or "
            "GEMMALOOP_SHORTFALL_FREEZE_INDEX). The run refuses before any "
            "precompute unless every bound hash matches and the final corpus "
            "has >=20 T-fit and >=30 selection verified problems. P1 never gates."
        ),
    )
    parser.add_argument("--confirmatory-data", type=Path, default=None)
    parser.add_argument("--qualification-json", type=Path, default=None)
    parser.add_argument(
        "--teacher-runtime-freeze",
        type=Path,
        default=None,
        help=(
            "pinned qualified teacher-runtime record; defaults to "
            "<artifacts>/qualification/teacher_runtime_freeze.json"
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--chosen-teacher",
        choices=("e4b", "12b"),
        default=None,
        help="block stage only: override the teacher_step.json choice",
    )
    parser.add_argument("--force", action="store_true", help="re-run completed arms")
    parser.add_argument("--force-evaluate", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "refused: a registered pilot cannot relax the frozen-run and "
            "monitor checks, so this flag stops before any precompute"
        ),
    )
    parser.add_argument(
        "--ack-abort",
        action="append",
        default=None,
        metavar="ARM=ALERT_ID",
        help=(
            "acknowledge one arm's persisted heal-monitor abort (exit 76); the "
            "ack is passed to that arm's training orchestrator, which archives "
            "a partial pre-checkpoint run and restarts it from zero"
        ),
    )
    return parser.parse_args(argv)


def _parse_ack_aborts(
    values: Sequence[str] | None, arms: Sequence[ArmSpec]
) -> tuple[dict[str, str], str | None]:
    """Parse ``--ack-abort ARM=ALERT_ID`` entries for registered arms."""

    if not values:
        return {}, None
    registered = {arm.arm_id for arm in arms}
    parsed: dict[str, str] = {}
    for value in values:
        arm_id, separator, abort_id = str(value).partition("=")
        if not separator or not arm_id or not abort_id:
            return {}, (
                f"--ack-abort must be ARM=ALERT_ID, got {value!r}; "
                f"registered arms: {', '.join(sorted(registered))}"
            )
        if arm_id not in registered:
            return {}, (
                f"--ack-abort names unregistered arm {arm_id!r}; "
                f"registered arms: {', '.join(sorted(registered))}"
            )
        parsed[arm_id] = abort_id
    return parsed, None


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _load_results(out: Path) -> dict:
    payload = _read_json(out / RESULTS_FILE)
    payload.setdefault("arms", {})
    payload.setdefault("actions", [])
    payload.setdefault("teacher_step", None)
    payload.setdefault("selection", None)
    payload.setdefault("stage", None)
    return payload


def _e4b_selection_arms(arms: Sequence[ArmSpec]) -> list[ArmSpec]:
    """The registered E4B L arms and their block-matched E4B O-H controls."""

    required: list[ArmSpec] = []
    for arm in arms:
        if (
            arm.kind == "L"
            and arm.teacher == "e4b"
            and arm.eligible
            and not arm.descriptive
        ):
            required.append(arm)
            oh = _matched_oh(arm, arms)
            if oh is not None and oh.teacher == "e4b":
                required.append(oh)
    return required


def _required_for_selection(
    stage: str,
    arms: Sequence[ArmSpec],
    excluded_teachers: Sequence[str] | None = None,
) -> list[ArmSpec]:
    excluded = set(excluded_teachers or ())
    if stage == "teacher":
        return [
            arm
            for arm in teacher_stage_arms(arms)
            if arm.teacher not in excluded
        ]
    if stage == "block":
        return _e4b_selection_arms(arms)
    # ``all`` needs the teacher step plus every registered E4B L arm and its
    # block-matched O-H.  The descriptive single-exit arms are never required.
    # A teacher the registration excludes is never required either.
    required = [
        arm for arm in teacher_stage_arms(arms) if arm.teacher not in excluded
    ]
    for arm in _e4b_selection_arms(arms):
        if arm.arm_id not in {item.arm_id for item in required}:
            required.append(arm)
    return required


def _teacher_stage_excluded_arms(
    arms: Sequence[ArmSpec], excluded_teachers: Mapping[str, str]
) -> dict[str, str]:
    """Arm id -> reason for teacher-stage arms whose teacher is excluded."""

    return {
        arm.arm_id: excluded_teachers[arm.teacher]
        for arm in teacher_stage_arms(arms)
        if arm.teacher in excluded_teachers
    }


def _selection_readiness(
    required: Sequence[ArmSpec], *, root: Path, results: Mapping
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    failed: list[str] = []
    for arm in required:
        record = (results.get("arms") or {}).get(arm.arm_id) or {}
        if not is_arm_complete(arm_run_dir(root, arm.arm_id)):
            missing.append(arm.arm_id)
            continue
        if record.get("status") != "evaluated" or not record.get("evaluation"):
            failed.append(arm.arm_id)
    return missing, failed


def _write_unevaluable(
    out: Path, *, stage: str, missing: Sequence[str], failed: Sequence[str]
) -> dict:
    payload = {
        "status": "unevaluable",
        "stage": stage,
        "missing": list(missing),
        "failed": list(failed),
        "reason": (
            "every arm required by this stage must finish training and "
            "evaluation before a selection or STOP is written"
        ),
    }
    _write_json(out / SELECTION_PENDING_FILE, payload)
    stale = out / SELECTION_FILE
    if stale.is_file():
        stale.unlink()
    return payload


def _matched_oh(arm: ArmSpec, arms: Sequence[ArmSpec]) -> ArmSpec | None:
    for other in arms:
        if other.kind != "O-H" or other.block != arm.block:
            continue
        if other.teacher == arm.teacher:
            return other
    return None


def _evaluation_ready(arm: ArmSpec, root: Path) -> bool:
    return is_arm_complete(arm_run_dir(root, arm.arm_id))


def _evaluate(
    arms: Sequence[ArmSpec],
    *,
    root: Path,
    results: dict,
    prereg_path: Path,
    heal_data: Path,
    fit_data: Path | None,
    test_data: Path | None,
    force: bool,
    execute=None,
    actions: Sequence[Mapping] | None = None,
    teacher_runtime_freeze: Path | None = None,
    excluded: Mapping[str, str] | None = None,
) -> None:
    by_action = {str(action.get("arm_id")): action for action in (actions or [])}
    excluded = dict(excluded or {})
    for arm in arms:
        record = results["arms"].setdefault(arm.arm_id, {"arm_id": arm.arm_id})
        record.update(arm.to_dict())
        if arm.arm_id in excluded:
            # Registered exclusion: never trained and never scored.
            record["status"] = "excluded"
            record["excluded"] = True
            record["reason"] = excluded[arm.arm_id]
            continue
        if not _evaluation_ready(arm, root):
            record["status"] = "pending-training"
            continue
        if not fit_data or not test_data:
            record["status"] = "pending-eval-data"
            continue
        result = run_evaluation(
            arm,
            root=root,
            prereg_path=prereg_path,
            heal_data=heal_data,
            fit_data=fit_data,
            test_data=test_data,
            execute=execute,
            force=force,
            teacher_runtime_freeze=teacher_runtime_freeze,
        )
        evaluation = result.get("evaluation") or {}
        run_payload = _read_json(arm_run_dir(root, arm.arm_id) / "run.json")
        record["run_identity"] = {
            "arm": run_payload.get("arm"),
            "bridge": run_payload.get("bridge"),
            "loop_block": run_payload.get("loop_block"),
            "run_identity_hash": run_payload.get("run_identity_hash"),
        }
        record["evaluation_status"] = result.get("action")
        record["evaluation_path"] = result.get("path")
        if evaluation.get("runtime"):
            record["runtime"] = evaluation["runtime"]
        if evaluation.get("evaluation"):
            evaluation = evaluation["evaluation"]
            training = evaluation.get("training") or read_training_summary(
                arm_run_dir(root, arm.arm_id)
            )
            wall = (by_action.get(arm.arm_id) or {}).get("wall_seconds")
            if wall is not None:
                training["wall_seconds"] = float(wall)
                if training.get("tokens"):
                    training["tokens_per_second"] = float(training["tokens"]) / float(wall)
            evaluation["training"] = training
            record["evaluation"] = evaluation
            record["training"] = training
            record["status"] = "evaluated"
        else:
            record["status"] = "eval-failed"
            record["evaluation_error"] = result.get("returncode")
        _write_json(Path(root) / RESULTS_FILE, results)


def _summaries(arms: Sequence[ArmSpec], results: dict) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for arm in arms:
        record = results["arms"].get(arm.arm_id) or {}
        evaluation = record.get("evaluation")
        if not evaluation:
            continue
        oh = _matched_oh(arm, arms) if arm.kind == "L" else None
        oh_eval = (results["arms"].get(oh.arm_id) or {}).get("evaluation") if oh else None
        summary = build_arm_summary(
            arm, evaluation, oh_eval, iters=DEFAULT_ITERS, seed=DEFAULT_SEED
        )
        summary["training"] = record.get("training") or evaluation.get("training") or {}
        summary["teacher_kl"] = evaluation.get("teacher_kl")
        summary["runtime"] = record.get("runtime") or arm.runtime
        # Descriptive gain split at N=2 against the block-matched O-H.
        final = ((evaluation.get("checkpoints") or {}).get("final") or {}).get("n") or {}
        reference = None
        if oh_eval:
            oh_final = ((oh_eval.get("checkpoints") or {}).get("final") or {}).get("n") or {}
            reference = oh_final.get("1")
        if arm.kind == "L" and reference and "2" in final:
            summary["spans"] = {
                key: float(value) - float((reference.get("spans") or {}).get(key, value))
                for key, value in (final["2"].get("spans") or {}).items()
            }
            positions = {}
            for key, value in (final["2"].get("buckets") or {}).items():
                if not isinstance(value, Mapping):
                    continue
                ours = value.get("calibrated_nll")
                theirs = (
                    ((reference.get("buckets") or {}).get(key) or {}).get("calibrated_nll")
                )
                if ours is not None and theirs is not None:
                    positions[key] = float(theirs) - float(ours)
            summary["positions"] = positions
        summaries[arm.arm_id] = summary
    return summaries


def _canonical_teacher_map(
    arms: Sequence[ArmSpec], summaries: Mapping[str, Mapping], teacher: str
) -> dict[str, Mapping]:
    output: dict[str, Mapping] = {}
    for arm in teacher_stage_arms(arms):
        if arm.teacher != teacher or arm.arm_id not in summaries:
            continue
        output[arm.arm_id] = summaries[arm.arm_id]
    return output


def _validate_identities(
    arms: Sequence[ArmSpec],
    summaries: Mapping[str, Mapping],
    results: Mapping | None = None,
) -> None:
    """Selection consumes only summaries whose identity matches the run records."""

    registered = {arm.arm_id: arm for arm in arms}
    records = (results or {}).get("arms") or {}
    for arm_id, summary in summaries.items():
        arm = registered.get(str(arm_id))
        if arm is None:
            raise ValueError(f"summary for unregistered arm {arm_id!r}")
        for field in ("block", "bridge", "teacher", "registered_teacher"):
            expected = list(arm.block) if field == "block" else getattr(arm, field)
            if summary.get(field) != expected:
                raise ValueError(
                    f"arm {arm_id} identity drift: {field}={summary.get(field)!r}, "
                    f"registered {expected!r}"
                )
        record = records.get(str(arm_id)) or {}
        run_identity = record.get("run_identity") or {}
        stored_arm = run_identity.get("arm")
        if stored_arm is not None and stored_arm != summary.get("kind"):
            raise ValueError(
                f"arm {arm_id} run identity drift: arm={stored_arm!r}, "
                f"summary kind {summary.get('kind')!r}"
            )
        stored_bridge = run_identity.get("bridge")
        if stored_bridge is not None and stored_bridge != summary.get("bridge"):
            raise ValueError(
                f"arm {arm_id} run identity drift: bridge={stored_bridge!r}, "
                f"summary bridge {summary.get('bridge')!r}"
            )
        runtime = summary.get("runtime")
        if not isinstance(runtime, Mapping) or not runtime:
            raise ValueError(f"arm {arm_id} has no teacher-runtime identity")


def _baseline_nll(arms: Sequence[ArmSpec], results: Mapping) -> float | None:
    for arm in arms:
        if arm.kind != "L":
            continue
        evaluation = (results.get("arms", {}).get(arm.arm_id) or {}).get("evaluation")
        if not evaluation:
            continue
        initial = ((evaluation.get("checkpoints") or {}).get("initial") or {}).get("n") or {}
        clusters = (initial.get("1") or {}).get("clusters") or {}
        if clusters:
            return cluster_mean(clusters)
    return None


def _teacher_stats(arms: Sequence[ArmSpec], root: Path) -> dict:
    teachers = {}
    for teacher in sorted({arm.teacher for arm in arms}):
        cache = teacher_cache_dir(root, teacher)
        if teacher_cache_complete(cache):
            stats = teacher_shard_stats(cache)
            if stats:
                teachers[teacher] = stats
    return teachers


def _report_payload(
    prereg,
    *,
    results: Mapping,
    arms: Sequence[ArmSpec],
    summaries: Mapping[str, Mapping],
    teachers: Mapping,
    qualification: Mapping,
) -> dict:
    return {
        "pilot": {
            "heal_tokens": int(prereg.pilot.p2.heal_tokens),
            "cost_gpu_hours": str(prereg.pilot.cost_gpu_hours),
            "held_out_only": bool(prereg.pilot.held_out_only),
            "benchmarks": bool(prereg.pilot.benchmarks),
        },
        "arms": {key: dict(value) for key, value in summaries.items()},
        "teacher_step": results.get("teacher_step"),
        "selection": results.get("selection"),
        "teachers": dict(teachers),
        "qualification": dict(qualification),
        "baseline_nll": _baseline_nll(arms, results),
        "arm_specs": [arm.to_dict() for arm in arms],
    }


def _resolve_monitor_rows(args: argparse.Namespace) -> Path | None:
    """The MONITOR JSONL (``--monitor-rows`` or a ``--monitor-data`` path)."""

    candidate = args.monitor_rows
    if candidate is None and args.monitor_data is not None:
        data = Path(args.monitor_data)
        if data.is_dir():
            for name in ("monitor_rows.jsonl", "monitor.jsonl"):
                if (data / name).is_file():
                    candidate = data / name
                    break
        else:
            candidate = data
    if candidate is None:
        return None
    path = Path(candidate)
    return path if path.is_file() else None


def _preflight_rendered_rows(path: Path, label: str) -> str | None:
    """Refuse a fit/test split that has no rendered ``input_ids``.

    Raw Stage 0 trace rows carry ``messages`` and ``problem_id`` but no
    ``input_ids``; evaluation would fail after the GPU training work.  The
    renderer (``scripts/render_pilot_eval_rows.py``) produces the rows
    evaluation consumes, so refuse before any precompute instead.
    """

    saw_row = False
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                saw_row = True
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    return f"{label} row {index} in {path} is not valid JSON"
                if not isinstance(payload, Mapping):
                    return f"{label} row {index} in {path} is not an object"
                input_ids = payload.get("input_ids")
                if not isinstance(input_ids, list) or len(input_ids) < 2:
                    return (
                        f"{label} row {index} in {path} has no rendered "
                        "input_ids; render the Stage 0 traces first with "
                        "scripts/render_pilot_eval_rows.py "
                        "(pilot_tfit.jsonl for --fit-data, "
                        "pilot_selection.jsonl for --test-data)"
                    )
    except OSError as exc:
        return f"{label} {path} could not be read: {exc}"
    if not saw_row:
        return f"{label} {path} has no rows"
    return None


def _monitor_env_off() -> bool:
    """True when ``GEMMALOOP_HEAL_MONITOR`` disables the registered monitor."""

    explicit = os.environ.get("GEMMALOOP_HEAL_MONITOR")
    if explicit is None:
        return False
    return explicit.strip().lower() in ("0", "false", "no", "off", "")


def _drop_excluded_teacher_records(
    results: dict, excluded_arms: Mapping[str, str]
) -> None:
    """Remove any recorded result for an arm whose teacher is excluded."""

    arms = results.get("arms") or {}
    for arm_id in excluded_arms:
        arms.pop(arm_id, None)


def _teacher_step_exclusion_state(step: Mapping) -> list[str]:
    """The exclusion state stored on a teacher-step record."""

    recorded = step.get("excluded_teachers")
    if recorded is None:
        return []
    if isinstance(recorded, str):
        return [recorded]
    return sorted(str(item) for item in recorded)


def _teacher_step_is_reusable(
    step: Mapping | None, excluded_teachers: Mapping[str, str]
) -> bool:
    """True when a saved teacher-step record matches the current exclusions.

    A record produced before the round-8h exclusions (or by a two-teacher
    comparison) has no ``excluded_teachers`` entry and cannot be reused once a
    teacher is excluded.  A record that names an excluded teacher in either
    ``chosen`` or ``recipe_teacher`` is stale regardless of its exclusion
    state.  Both cases are rejected so the stage recomputes from the current
    registration instead of carrying stale metadata forward.
    """

    if not isinstance(step, Mapping):
        return False
    expected = sorted(str(item) for item in excluded_teachers)
    if _teacher_step_exclusion_state(step) != expected:
        return False
    named = {
        str(step.get("chosen") or ""),
        str(step.get("recipe_teacher") or ""),
    }
    named.discard("")
    return not (named & set(expected))


def _drop_stale_teacher_step(
    out: Path, results: dict, excluded_teachers: Mapping[str, str]
) -> None:
    """Discard a teacher-step record that does not match the exclusions.

    The result entry and the standalone ``teacher_step.json`` are both checked
    because the block stage reads the file while the teacher stage reads the
    results payload.  A stale file is removed so a later stage cannot reuse it
    if recomputation is not possible yet.
    """

    step = results.get("teacher_step")
    if step is not None and not _teacher_step_is_reusable(
        step, excluded_teachers
    ):
        results["teacher_step"] = None
        results["chosen_teacher"] = None
    path = Path(out) / TEACHER_STEP_FILE
    saved = _read_json(path)
    if saved and not _teacher_step_is_reusable(saved, excluded_teachers):
        try:
            path.unlink()
        except OSError:
            pass


def _finalize_teacher_step(
    prereg,
    arms: Sequence[ArmSpec],
    *,
    out: Path,
    results: dict,
    summaries: Mapping[str, Mapping],
    excluded_teachers: Mapping[str, str],
) -> dict | None:
    """Write ``teacher_step.json``; handle the registered single-teacher case.

    With both 12B runtimes excluded there is no comparison to estimate, so the
    record states that fact and names E4B as the recipe teacher.  Otherwise the
    registered :func:`select_teacher` rule is used unchanged.
    """

    if results.get("teacher_step") is not None:
        if _teacher_step_is_reusable(
            results["teacher_step"], excluded_teachers
        ):
            return results["teacher_step"]
        results["teacher_step"] = None
        results["chosen_teacher"] = None
        try:
            (out / TEACHER_STEP_FILE).unlink()
        except OSError:
            pass
    baseline = _baseline_nll(arms, results)
    if baseline is None:
        return None
    e4b = _canonical_teacher_map(arms, summaries, "e4b")
    if not e4b:
        return None
    twelve = (
        {}
        if excluded_teachers
        else _canonical_teacher_map(arms, summaries, "12b")
    )
    if twelve:
        step = select_teacher(
            e4b,
            twelve,
            baseline,
            tie_nats=float(prereg.pilot.p2.tie_nats),
        )
    elif excluded_teachers:
        step = select_single_teacher(
            e4b,
            excluded_teachers=sorted(excluded_teachers),
            baseline_nll=baseline,
        )
    else:
        return None
    results["teacher_step"] = step
    _write_json(out / TEACHER_STEP_FILE, step)
    _write_json(out / RESULTS_FILE, results)
    return step


def main(argv: Sequence[str] | None = None, execute=None) -> int:
    args = _parse_args(argv)
    # R2-M3: a registered pilot must not spend hours on the teacher/MONITOR
    # precompute only for every train child to refuse after loading the model.
    if _monitor_env_off():
        print(
            "error: GEMMALOOP_HEAL_MONITOR=0 disables the registered monitor; "
            "a pilot run cannot turn the fail-closed split checks off. Unset it "
            "before any precompute.",
            file=sys.stderr,
        )
        return 2
    if args.smoke:
        print(
            "error: --smoke is not allowed for a registered pilot run; it "
            "relaxes the frozen-run and registered monitor checks. Refusing "
            "before any precompute.",
            file=sys.stderr,
        )
        return 2
    prereg_path = args.prereg or default_prereg_path()
    prereg = load_prereg(prereg_path)

    # Pilot-shortfall amendment execution guard: the corpus/verifier index must
    # exist, every bound hash must match, and the *final* T-fit/selection
    # corpora must meet 20/30.  P1 is descriptive and never gates.  This runs
    # before any monitor/teacher precompute.
    shortfall_freeze = args.shortfall_freeze or Path(
        os.environ.get(
            "GEMMALOOP_SHORTFALL_FREEZE_INDEX",
            str(Path(__file__).resolve().parents[1] / "prereg" / "pilot-shortfall-freeze.json"),
        )
    )
    try:
        shortfall_index = check_freeze_readiness(shortfall_freeze)
    except (ShortfallReadinessError, OSError, ValueError) as exc:
        print(
            f"error: pilot-shortfall readiness refused before any precompute: {exc}",
            file=sys.stderr,
        )
        return 2

    # Bind the *actual* evaluation inputs to the frozen final corpus before any
    # monitor/teacher precompute.  A caller-supplied file that is not
    # byte-identical to the indexed rendered artifact is refused, and the
    # distinct-problem counts are recomputed from those same files.
    try:
        check_pilot_inputs(
            shortfall_index,
            fit_data=args.fit_data,
            test_data=args.test_data,
        )
    except (ShortfallReadinessError, OSError, ValueError) as exc:
        print(
            "error: pilot-shortfall inputs refused before any precompute: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    arms = pilot_arms(prereg)
    # The registration, not the command line, decides whether a teacher is
    # excluded.  Round 8h declares both 12B runtimes excluded, leaving E4B int8
    # as the single qualified teacher.
    excluded_teachers = registered_teacher_exclusions(prereg)
    if args.chosen_teacher and args.chosen_teacher in excluded_teachers:
        print(
            f"error: --chosen-teacher {args.chosen_teacher} names a teacher the "
            "registration excludes (no qualified resident runtime); the recipe "
            "teacher is E4B",
            file=sys.stderr,
        )
        return 2
    ack_aborts, ack_problem = _parse_ack_aborts(args.ack_abort, arms)
    if ack_problem is not None:
        print(f"error: {ack_problem}", file=sys.stderr)
        return 2

    monitor_rows = _resolve_monitor_rows(args)
    if monitor_rows is None:
        print(
            "error: a registered pilot heal needs the dedicated MONITOR corpus; "
            "pass --monitor-rows <stage0>/monitor/monitor_rows.jsonl",
            file=sys.stderr,
        )
        return 2

    # A registered pilot without the fit/evaluation corpora has an empty
    # reserved map; every train child would exit 2 in the monitor split proof
    # AFTER the GPU teacher precomputes.  Refuse before any precompute.
    if args.fit_data is None or args.test_data is None:
        print(
            "error: a registered pilot needs --fit-data and --test-data (their "
            "problem ids are the reserved split proof); refusing before any "
            "precompute",
            file=sys.stderr,
        )
        return 2

    # B1: the frozen-run contract must fail before any GPU teacher precompute.
    repo_root = Path(__file__).resolve().parents[1]
    try:
        check_frozen_run(
            repo_root,
            require_tag=args.require_tag,
            prereg=str(prereg_path),
        )
        assert_code_under(repo_root)
    except FrozenRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Evaluation consumes rendered rows (input_ids + assistant labels).  Raw
    # Stage 0 traces pass the problem-id check but would fail only after the
    # GPU training work, so refuse them before any precompute.
    for label, path in (
        ("--fit-data", args.fit_data),
        ("--test-data", args.test_data),
    ):
        problem = _preflight_rendered_rows(Path(path), label)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2

    try:
        assert_disjoint_splits(
            {
                "heal": args.heal_data,
                "fit": args.fit_data,
                "test": args.test_data,
                "monitor": monitor_rows,
                "confirmatory": args.confirmatory_data,
            }
        )
    except (SplitOverlapError, ValueError, OSError) as exc:
        print(f"error: split overlap refused: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reserved_path = out / "reserved_ids.json"
    reserved_payload = reserved_ids_payload(
        {
            "heal": args.heal_data,
            "fit": args.fit_data,
            "evaluation": args.test_data,
            "monitor": monitor_rows,
            "confirmatory": args.confirmatory_data,
        }
    )
    if not reserved_payload.get("pilot_eval") or not reserved_payload.get(
        "pilot_fit"
    ):
        print(
            "error: --fit-data/--test-data must carry problem_id (or cluster_id) "
            "for the reserved split proof; P1-lite rows need problem_id "
            "(scripts/render_pilot_eval_rows.py or make_p1lite_corpus.py)",
            file=sys.stderr,
        )
        return 2
    _write_json(reserved_path, reserved_payload)
    results = _load_results(out)
    results["prereg"] = str(prereg_path)
    results["seed"] = int(args.seed)
    # Never let a record produced before the registered exclusion (or under a
    # different exclusion state) label this run, whichever stage starts.
    _drop_stale_teacher_step(out, results, excluded_teachers)

    def run(
        selected: Sequence[ArmSpec],
        stage_name: str,
        *,
        excluded_arms: Mapping[str, str] | None = None,
    ) -> list[dict]:
        try:
            stage_kwargs = {}
            if ack_aborts:
                stage_kwargs["ack_aborts"] = ack_aborts
            actions = run_stage(
                selected,
                root=out,
                prereg_path=prereg_path,
                prereg=prereg,
                heal_data=args.heal_data,
                seed=args.seed,
                force=args.force,
                execute=execute,
                reserved_ids_path=reserved_path,
                monitor_rows=monitor_rows,
                max_length=int(prereg.training.max_length),
                require_tag=args.require_tag,
                teacher_runtime_freeze=args.teacher_runtime_freeze,
                **stage_kwargs,
            )
        except (
            RunIdentityMismatch,
            EvaluationIdentityMismatch,
            MonitorTeacherDataMissing,
            TeacherRuntimeFreezeError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        results["actions"].extend(actions)
        results["stage"] = stage_name
        _evaluate(
            selected,
            root=out,
            results=results,
            prereg_path=prereg_path,
            heal_data=args.heal_data,
            fit_data=args.fit_data,
            test_data=args.test_data,
            force=args.force_evaluate,
            execute=execute,
            actions=actions,
            teacher_runtime_freeze=args.teacher_runtime_freeze,
            excluded=excluded_arms,
        )
        _write_json(out / RESULTS_FILE, results)
        return actions

    if args.stage in ("teacher", "all"):
        teacher_arms = teacher_stage_arms(arms)
        teacher_excluded = _teacher_stage_excluded_arms(arms, excluded_teachers)
        run(teacher_arms, "teacher", excluded_arms=teacher_excluded)
        _drop_excluded_teacher_records(results, teacher_excluded)
        active_teacher_arms = [
            arm for arm in teacher_arms if arm.arm_id not in teacher_excluded
        ]
        missing, failed = _selection_readiness(
            active_teacher_arms, root=out, results=results
        )
        if missing or failed:
            results["selection"] = None
            payload = _write_unevaluable(
                out, stage="teacher", missing=missing, failed=failed
            )
            _write_json(out / RESULTS_FILE, results)
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return 3
        summaries = _summaries(arms, results)
        _validate_identities(arms, summaries, results)
        _finalize_teacher_step(
            prereg,
            arms,
            out=out,
            results=results,
            summaries=summaries,
            excluded_teachers=excluded_teachers,
        )

    if args.stage in ("block", "all"):
        chosen = None
        if args.chosen_teacher:
            chosen = args.chosen_teacher
        else:
            saved = _read_json(out / TEACHER_STEP_FILE)
            chosen = saved.get("chosen") or (results.get("teacher_step") or {}).get("chosen")
        results["chosen_teacher"] = chosen
        # The block/bridge comparison is registered among the pre-listed E4B
        # arms.  ``chosen`` only labels the final recipe; it never relabels or
        # replaces a block-stage arm.
        run(block_stage_arms(arms), "block")

    teacher_excluded = _teacher_stage_excluded_arms(arms, excluded_teachers)
    _drop_excluded_teacher_records(results, teacher_excluded)
    summaries = _summaries(arms, results)
    if results.get("teacher_step") is None:
        active_teacher_arms = [
            arm
            for arm in teacher_stage_arms(arms)
            if arm.arm_id not in teacher_excluded
        ]
        teacher_ready = not any(
            _selection_readiness([arm], root=out, results=results)[0]
            + _selection_readiness([arm], root=out, results=results)[1]
            for arm in active_teacher_arms
        )
        if teacher_ready:
            step = _finalize_teacher_step(
                prereg,
                arms,
                out=out,
                results=results,
                summaries=summaries,
                excluded_teachers=excluded_teachers,
            )
            if step and not results.get("chosen_teacher"):
                results["chosen_teacher"] = step.get("chosen")

    required = _required_for_selection(
        args.stage, arms, excluded_teachers
    )
    missing, failed = _selection_readiness(required, root=out, results=results)
    if missing or failed:
        results["selection"] = None
        payload = _write_unevaluable(
            out, stage=args.stage, missing=missing, failed=failed
        )
        _write_json(out / RESULTS_FILE, results)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 3

    if args.stage in ("block", "all") and summaries:
        _validate_identities(arms, summaries, results)
        selection = select_block_rule_b(
            summaries,
            tie_nats=float(prereg.pilot.p2.tie_nats),
            min_delta_arm=float(prereg.pilot.p2.min_arm_vs_oh_nats),
        )
        selection["chosen_teacher"] = results.get("chosen_teacher")
        selection["chosen_teacher_source"] = (
            "cli" if args.chosen_teacher else "teacher_step"
        )
        results["selection"] = selection
        _write_json(out / SELECTION_FILE, results["selection"])
        pending = out / SELECTION_PENDING_FILE
        if pending.is_file():
            pending.unlink()

    qualification = load_qualification(args.qualification_json)
    payload = _report_payload(
        prereg,
        results=results,
        arms=arms,
        summaries=summaries,
        teachers=_teacher_stats(arms, out),
        qualification=qualification,
    )
    write_report(payload, out / "report")
    _write_json(out / RESULTS_FILE, results)
    print(json.dumps({"out": str(out), "selection": results.get("selection")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
