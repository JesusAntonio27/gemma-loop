"""Crash-safe pilot stage planning and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.pilot.arms import block_stage_arms, teacher_stage_arms
from gemmaloop.pilot.harness import (
    arm_run_dir,
    build_train_command,
    evaluation_path,
    is_arm_complete,
    orchestrator_command,
    run_evaluation,
    run_stage,
    stage_actions,
    teacher_cache_dir,
    teacher_cache_complete,
)
from gemmaloop.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


def _allow_frozen(monkeypatch):
    import scripts.run_pilot as run_pilot

    monkeypatch.setattr(run_pilot, "check_frozen_run", lambda *a, **k: None)
    monkeypatch.setattr(run_pilot, "assert_code_under", lambda *a, **k: None)
    monkeypatch.setattr(run_pilot, "check_freeze_readiness", lambda *a, **k: {})
    monkeypatch.setattr(run_pilot, "check_pilot_inputs", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _allow_shortfall_freeze(monkeypatch):
    import scripts.run_pilot as run_pilot

    monkeypatch.setattr(run_pilot, "check_freeze_readiness", lambda *a, **k: {})
    monkeypatch.setattr(run_pilot, "check_pilot_inputs", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _frozen_teacher_runtimes(monkeypatch, tmp_path_factory):
    """Every run_stage test uses a synthetic but structurally valid freeze."""

    import gemmaloop.pilot.harness as harness
    from gemmaloop.teacher.runtime_freeze import (
        FrozenTeacherRuntime,
        TeacherRuntimeFreeze,
    )

    def entry(teacher, teacher_key, runtime):
        return FrozenTeacherRuntime(
            teacher=teacher,
            teacher_key=teacher_key,
            runtime=runtime,
            qualification_json=f"{teacher}.json",
            qualification_sha256="a" * 64,
            decision="pass",
        )

    freeze = TeacherRuntimeFreeze(
        teachers={
            "e4b": entry("e4b", "e4b", "int8"),
            "12b": entry("12b", "12b_qat", "nf4"),
        }
    )
    path = tmp_path_factory.mktemp("freeze") / "teacher_runtime_freeze.json"
    monkeypatch.setattr(
        harness, "load_teacher_runtime_freeze", lambda path=None: freeze
    )
    monkeypatch.setattr(
        harness, "default_teacher_runtime_freeze_path", lambda: path
    )


def _arms():
    prereg = load_prereg(PREREG)
    from gemmaloop.pilot.arms import pilot_arms

    arms = pilot_arms(prereg)
    return teacher_stage_arms(arms), block_stage_arms(arms)


def _manifest_sha(cache: Path) -> str:
    import hashlib

    return hashlib.sha256((cache / "teacher_manifest.json").read_bytes()).hexdigest()


def _cache_teacher(directory: Path) -> tuple[str, str]:
    name = directory.parent.name if directory.name == "monitor" else directory.name
    if name == "12b":
        return "12b_qat", "nf4"
    return "e4b", "int8"


def _complete(
    directory: Path,
    arm,
    cache: Path,
    *,
    monitor_rows: Path | None = None,
    monitor_teacher_data: Path | None = None,
) -> None:
    """Write the exact run.json/run_identity.json layout train.py produces."""

    from tests.train_layout import train_shaped_complete_run

    teacher_key, runtime = _cache_teacher(cache)
    train_shaped_complete_run(
        directory,
        arm=arm,
        cache=cache,
        prereg=PREREG,
        teacher_key=teacher_key,
        runtime=runtime,
        monitor_rows=monitor_rows,
        monitor_teacher_data=monitor_teacher_data,
    )


def _partial(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoint-10").mkdir()


def _cache(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    teacher_key, runtime = _cache_teacher(directory)
    (directory / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "shards": [{"file": "shard-00000.pt"}],
                "teacher": {"key": teacher_key, "runtime": runtime},
            }
        ),
        encoding="utf-8",
    )
    (directory / "shard-00000.pt").write_bytes(b"shard")


def test_resume_skips_complete_arms_and_continues_partial_ones(tmp_path):
    teacher, _block = _arms()
    e4b = [arm for arm in teacher if arm.teacher == "e4b"]
    arm_a, arm_b = e4b[0], e4b[1]
    _cache(teacher_cache_dir(tmp_path, arm_a.teacher))
    _complete(
        arm_run_dir(tmp_path, arm_a.arm_id),
        arm_a,
        teacher_cache_dir(tmp_path, arm_a.teacher),
    )
    _partial(arm_run_dir(tmp_path, arm_b.arm_id))

    calls = []

    def execute(command, out):
        calls.append((list(command), Path(out)))
        if any("train.py" in part for part in command):
            arm_id = command[command.index("--pilot-arm") + 1]
            arm = next(a for a in teacher if a.arm_id == arm_id)
            _complete(
                Path(out),
                arm,
                teacher_cache_dir(tmp_path, arm.teacher),
            )
        return 0

    actions = run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
    )
    by_id = {action["arm_id"]: action for action in actions}
    assert by_id[arm_a.arm_id]["action"] == "skipped"
    assert by_id[arm_b.arm_id]["action"] == "trained"
    assert len(calls) == 1
    assert "--pilot-arm" in calls[0][0]
    assert arm_b.arm_id in calls[0][0]

    # A second launch is a no-op: both arms are now complete.
    calls.clear()
    again = run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
    )
    assert {action["action"] for action in again} == {"skipped"}
    assert calls == []


def test_missing_teacher_shards_are_precomputed_once_then_reused(tmp_path):
    teacher, _block = _arms()
    e4b = [arm for arm in teacher if arm.teacher == "e4b"]
    arm_a, arm_b = e4b[0], e4b[1]
    calls = []

    def execute(command, out):
        calls.append((list(command), Path(out)))
        if any("precompute_teacher_topk.py" in part for part in command):
            _cache(Path(out))
        elif any("train.py" in part for part in command):
            arm_id = command[command.index("--pilot-arm") + 1]
            arm = next(a for a in [arm_a, arm_b] if a.arm_id == arm_id)
            _complete(
                Path(out),
                arm,
                teacher_cache_dir(tmp_path, arm.teacher),
            )
        return 0

    run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
    )
    precompute = [call for call in calls if any("precompute_teacher_topk.py" in part for part in call[0])]
    trains = [call for call in calls if any("train.py" in part for part in call[0])]
    assert len(precompute) == 1
    assert len(trains) == 2
    assert teacher_cache_complete(teacher_cache_dir(tmp_path, "e4b"))


def test_stage_actions_report_partial_resume_without_running_it(tmp_path):
    teacher, _block = _arms()
    arm = teacher[0]
    _partial(arm_run_dir(tmp_path, arm.arm_id))
    actions = stage_actions(teacher, root=tmp_path)
    action = next(item for item in actions if item["arm_id"] == arm.arm_id)
    assert action["action"] == "train"
    assert action["resume"] is True
    assert is_arm_complete(arm_run_dir(tmp_path, arm.arm_id)) is False


def test_evaluation_runs_through_orchestrator_and_resumes(tmp_path):
    teacher, _block = _arms()
    arm = teacher[0]
    _cache(teacher_cache_dir(tmp_path, arm.teacher))
    calls = []

    def execute(command, out):
        calls.append((list(command), Path(out)))
        payload = {
            "status": "complete",
            "arm_id": arm.arm_id,
            "arm": arm.to_dict(),
            "runtime": arm.runtime,
            "evaluation": {"arm_id": arm.arm_id},
        }
        target = Path(out) / "evaluation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    first = run_evaluation(
        arm,
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        fit_data=tmp_path / "fit.jsonl",
        test_data=tmp_path / "test.jsonl",
        execute=execute,
    )
    assert first["action"] == "evaluated"
    command = calls[0][0]
    assert "orchestrate.py" in " ".join(command)
    assert "evaluate_pilot.py" in " ".join(command)
    assert command[command.index("--teacher-data") + 1] == str(
        teacher_cache_dir(tmp_path, arm.teacher)
    )
    assert evaluation_path(tmp_path, arm.arm_id).is_file()

    second = run_evaluation(
        arm,
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        fit_data=tmp_path / "fit.jsonl",
        test_data=tmp_path / "test.jsonl",
        execute=execute,
    )
    assert second["action"] == "skipped"
    assert len(calls) == 1


def test_train_and_orchestrator_commands_keep_the_registered_binding(tmp_path):
    teacher, _block = _arms()
    arm = teacher[0]
    command = build_train_command(
        arm,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        run_dir=tmp_path / "run",
        seed=7,
        teacher_data=tmp_path / "teacher-shards" / "e4b",
    )
    assert command[:3] == [__import__("sys").executable, str(ROOT / "scripts" / "train.py"), "--pilot-arm"]
    assert "--arm" not in command
    assert "--size" not in command
    assert "--tokens" not in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--teacher-data") + 1] == str(
        tmp_path / "teacher-shards" / "e4b"
    )
    wrapped = orchestrator_command("test", tmp_path / "run", command)
    assert "orchestrate.py" in " ".join(wrapped)
    assert wrapped[-len(command):] == command


def test_force_marks_the_child_for_a_rerun(tmp_path):
    teacher, _block = _arms()
    arm = teacher[0]
    command = build_train_command(
        arm,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        run_dir=tmp_path / "run",
        seed=0,
        teacher_data=tmp_path / "teacher-shards" / "e4b",
        force=True,
    )
    assert "--rerun-complete" in command


def test_evaluate_pilot_takes_the_gpu_lock_and_writes_resumable_json(tmp_path, monkeypatch):
    import scripts.evaluate_pilot as evaluate_pilot

    fit = tmp_path / "fit.jsonl"
    test = tmp_path / "test.jsonl"
    heal = tmp_path / "heal.jsonl"
    for path in (fit, test, heal):
        path.write_text(
            json.dumps({"input_ids": [1, 2, 3], "cluster_id": "p"}) + "\n",
            encoding="utf-8",
        )
    calls = []
    monkeypatch.setattr(
        evaluate_pilot, "require_gpu_lock", lambda: calls.append("lock")
    )
    monkeypatch.setattr(
        evaluate_pilot,
        "cap_cuda_allocator",
        lambda *args, **kwargs: calls.append("cap") or {"cap_gib": 10.0},
    )
    monkeypatch.setattr(
        evaluate_pilot,
        "_load_heal_rows",
        lambda path, max_length: [
            evaluate_pilot.load_eval_rows(path)[0]
        ],
    )
    monkeypatch.setattr(
        evaluate_pilot,
        "evaluate_arm",
        lambda *args, **kwargs: {"arm_id": "L-25-29-gch-e4b", "checkpoints": {}},
    )
    monkeypatch.setattr(
        evaluate_pilot, "read_training_summary", lambda run_dir: {"loss": []}
    )
    out = tmp_path / "eval" / "evaluation.json"
    code = evaluate_pilot.main(
        [
            "--arm-id", "L-25-29-gch-e4b",
            "--prereg", str(PREREG),
            "--run-dir", str(tmp_path / "run"),
            "--heal-data", str(heal),
            "--fit-data", str(fit),
            "--test-data", str(test),
            "--out", str(out),
        ]
    )
    assert code == 0
    assert calls == ["lock", "cap"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["evaluation"]["arm_id"] == "L-25-29-gch-e4b"


def test_evaluate_pilot_fails_closed_on_an_unknown_arm(tmp_path):
    import scripts.evaluate_pilot as evaluate_pilot

    code = evaluate_pilot.main(
        [
            "--arm-id", "L-99-99-gch-e4b",
            "--prereg", str(PREREG),
            "--run-dir", str(tmp_path),
            "--heal-data", str(tmp_path / "h"),
            "--fit-data", str(tmp_path / "f"),
            "--test-data", str(tmp_path / "t"),
            "--out", str(tmp_path / "o.json"),
        ]
    )
    assert code == 2


def test_every_registered_pilot_command_passes_train_validation(tmp_path):
    import scripts.train as train
    from gemmaloop.pilot.arms import pilot_arms
    from gemmaloop.prereg import load_prereg

    prereg = load_prereg(PREREG)
    for arm in pilot_arms(prereg):
        command = build_train_command(
            arm,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            run_dir=tmp_path / "run",
            seed=0,
            teacher_data=tmp_path / "teacher-shards" / arm.teacher,
        )
        args = train._parse_args(command[2:])
        train.apply_pilot_arm(args, prereg)
        objective = train.resolve_objective(args, prereg)
        train.resolve_recipe(args, prereg)
        assert objective.needs_teacher, arm.arm_id
        assert args.teacher_data is not None, arm.arm_id
        assert args.max_length == prereg.training.max_length


def _cache_with_teacher(directory: Path, teacher: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "shards": [{"file": "shard-00000.pt"}],
                "teacher": {"key": teacher, "revision": "rev", "quant": "nf4"},
                "data_manifest_sha256": "data-sha",
            }
        ),
        encoding="utf-8",
    )


def test_reused_run_with_a_different_teacher_is_refused(tmp_path):
    from gemmaloop.pilot.harness import RunIdentityMismatch

    teacher, block = _arms()
    arm = block[0]  # L-16-18-gch-e4b
    cache = teacher_cache_dir(tmp_path, arm.teacher)
    _cache_with_teacher(cache, arm.teacher)
    run_dir = arm_run_dir(tmp_path, arm.arm_id)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "arm": "L",
                "bridge": "gch",
                "loop_block": [16, 18],
                "teacher": {
                    "path": str(cache),
                    "manifest_sha256": _manifest_sha(cache),
                    "teacher": {"key": "12b_qat", "revision": "rev", "quant": "nf4"},
                    "data_manifest_sha256": "data-sha",
                },
                "data_manifest_sha256": "data-sha",
                "run_identity_hash": "sha",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_identity.json").write_text(
        json.dumps(
            {
                "run_config": {
                    "pilot_arm": arm.arm_id,
                    "arm": "L",
                    "block": "16-18",
                    "bridge": "gch",
                    "teacher_data": str(cache.resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RunIdentityMismatch):
        run_stage(
            [arm],
            root=tmp_path,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            seed=0,
            execute=lambda command, out: 0,
        )

    calls = []
    actions = run_stage(
        [arm],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        force=True,
        execute=lambda command, out: calls.append(list(command)) or 0,
    )
    assert actions[0]["action"] == "trained"
    assert any("--rerun-complete" in call for call in calls)


def test_stored_evaluation_with_a_different_arm_is_refused(tmp_path):
    from gemmaloop.pilot.harness import EvaluationIdentityMismatch

    teacher, _block = _arms()
    arm = teacher[0]
    _cache(teacher_cache_dir(tmp_path, arm.teacher))
    output = evaluation_path(tmp_path, arm.arm_id)
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(
            {
                "status": "complete",
                "arm_id": "L-16-18-gch-e4b",
                "arm": {"block": [16, 18], "bridge": "gch", "teacher": "e4b"},
                "runtime": {"teacher": "e4b"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationIdentityMismatch):
        run_evaluation(
            arm,
            root=tmp_path,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            fit_data=tmp_path / "fit.jsonl",
            test_data=tmp_path / "test.jsonl",
            execute=lambda command, out: 0,
        )
    calls = []

    def execute(command, out):
        calls.append(list(command))
        (Path(out) / "evaluation.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "arm_id": arm.arm_id,
                    "arm": arm.to_dict(),
                    "runtime": arm.runtime,
                    "evaluation": {"arm_id": arm.arm_id},
                }
            ),
            encoding="utf-8",
        )
        return 0

    result = run_evaluation(
        arm,
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        fit_data=tmp_path / "fit.jsonl",
        test_data=tmp_path / "test.jsonl",
        force=True,
        execute=execute,
    )
    assert result["action"] == "evaluated"
    assert len(calls) == 1


def test_reserved_ids_are_written_and_exported_to_the_train_child(tmp_path):
    import os

    from gemmaloop.pilot.splits import reserved_ids_payload

    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"messages": [], "row_id": 0, "cluster_id": "h"}) + "\n",
        encoding="utf-8",
    )
    fit = tmp_path / "fit.jsonl"
    fit.write_text(
        json.dumps({"input_ids": [1, 2], "row_id": 7, "cluster_id": "f"}) + "\n",
        encoding="utf-8",
    )
    payload = reserved_ids_payload({"heal": heal, "fit": fit})
    assert "heal_train" in payload
    assert "pilot_fit" in payload
    assert payload["heal_train"] == ["h"]
    assert payload["pilot_fit"] == ["f"]
    assert "f" in payload["reserved_ids"]
    assert "h" not in payload["reserved_ids"]
    assert "7" not in payload["reserved_ids"]

    reserved = tmp_path / "reserved_ids.json"
    reserved.write_text(json.dumps(payload), encoding="utf-8")
    teacher, _block = _arms()
    arm = teacher[0]
    _cache(teacher_cache_dir(tmp_path, arm.teacher))
    seen = []

    def execute(command, out):
        seen.append(os.environ.get("GEMMALOOP_HEAL_MONITOR_RESERVED"))
        _complete(Path(out), arm, teacher_cache_dir(tmp_path, arm.teacher))
        return 0

    run_stage(
        [arm],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=heal,
        seed=0,
        execute=execute,
        reserved_ids_path=reserved,
    )
    assert seen == [str(reserved)]
    assert os.environ.get("GEMMALOOP_HEAL_MONITOR_RESERVED") is None


def _write_monitor_corpus(path: Path, problem_ids) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "messages": [
                        {"role": "user", "content": problem_id},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            )
            + "\n"
            for problem_id in problem_ids
        ),
        encoding="utf-8",
    )
    return path


def test_every_registered_train_command_carries_the_monitor_flags(tmp_path):
    from gemmaloop.pilot.arms import pilot_arms

    monitor_rows = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    prereg = load_prereg(PREREG)
    for arm in pilot_arms(prereg):
        monitor_cache = tmp_path / "teacher-shards" / arm.teacher / "monitor"
        command = build_train_command(
            arm,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            run_dir=tmp_path / "run",
            seed=0,
            teacher_data=tmp_path / "teacher-shards" / arm.teacher,
            monitor_rows=monitor_rows,
            monitor_teacher_data=monitor_cache,
            require_tag="pilot-v1",
        )
        assert command[command.index("--monitor-rows") + 1] == str(monitor_rows)
        assert command[command.index("--monitor-teacher-data") + 1] == str(
            monitor_cache
        )
        assert command[command.index("--require-tag") + 1] == "pilot-v1"


def test_run_stage_precomputes_monitor_shards_before_training(tmp_path):
    teacher, _block = _arms()
    e4b = [arm for arm in teacher if arm.teacher == "e4b"]
    arm_a, arm_b = e4b[0], e4b[1]
    monitor_rows = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    calls = []

    def execute(command, out):
        calls.append((list(command), Path(out)))
        if any("precompute_teacher_topk.py" in part for part in command):
            data = Path(command[command.index("--data") + 1])
            if data.resolve() == monitor_rows.resolve():
                _cache(Path(out))
            else:
                _cache(Path(out))
        elif any("train.py" in part for part in command):
            arm_id = command[command.index("--pilot-arm") + 1]
            arm = next(a for a in [arm_a, arm_b] if a.arm_id == arm_id)
            _complete(Path(out), arm, teacher_cache_dir(tmp_path, arm.teacher))
        return 0

    run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
        monitor_rows=monitor_rows,
        max_length=32,
        require_tag="pilot-v1",
    )

    monitor_cache = teacher_cache_dir(tmp_path, "e4b") / "monitor"
    monitor_precompute = [
        call
        for call in calls
        if any("precompute_teacher_topk.py" in part for part in call[0])
        and call[0][call[0].index("--data") + 1] == str(monitor_rows)
    ]
    assert len(monitor_precompute) == 1
    command = monitor_precompute[0][0]
    assert command[command.index("--out") + 1] == str(monitor_cache)
    assert command[command.index("--teacher") + 1] == "e4b"
    assert command[command.index("--max-length") + 1] == "32"
    trains = [call[0] for call in calls if any("train.py" in part for part in call[0])]
    assert len(trains) == 2
    for train in trains:
        assert train[train.index("--monitor-rows") + 1] == str(monitor_rows)
        assert train[train.index("--monitor-teacher-data") + 1] == str(
            monitor_cache
        )
        assert train[train.index("--require-tag") + 1] == "pilot-v1"
    # The monitor shards are built before the first GPU training child.
    first_train = next(
        index
        for index, call in enumerate(calls)
        if any("train.py" in part for part in call[0])
    )
    first_monitor = next(
        index
        for index, call in enumerate(calls)
        if any("precompute_teacher_topk.py" in part for part in call[0])
        and call[0][call[0].index("--data") + 1] == str(monitor_rows)
    )
    assert first_monitor < first_train


def test_run_pilot_main_exports_the_reserved_file_and_ruled_split_map(
    tmp_path, monkeypatch
):
    import scripts.run_pilot as run_pilot

    def write_rows(path, problem_id, token):
        path.write_text(
            json.dumps({"problem_id": problem_id, "input_ids": [token, token + 1]})
            + "\n",
            encoding="utf-8",
        )
        return path

    heal = write_rows(tmp_path / "heal.jsonl", "heal-0", 1)
    fit = write_rows(tmp_path / "fit.jsonl", "fit-0", 3)
    test = write_rows(tmp_path / "test.jsonl", "eval-0", 5)
    confirmatory = write_rows(tmp_path / "confirmatory.jsonl", "confirm-0", 7)
    monitor = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    captured = {}

    def fake_stage(selected, *, root, prereg_path, heal_data, seed, force=False,
                   execute=None, reserved_ids_path=None, monitor_rows=None,
                   max_length=None, require_tag=None,
                   teacher_runtime_freeze=None, prereg=None):
        captured["reserved_ids_path"] = reserved_ids_path
        captured["monitor_rows"] = monitor_rows
        captured["max_length"] = max_length
        captured["require_tag"] = require_tag
        return []

    monkeypatch.setattr(run_pilot, "run_stage", fake_stage)
    _allow_frozen(monkeypatch)
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(fit),
            "--test-data", str(test),
            "--monitor-rows", str(monitor),
            "--confirmatory-data", str(confirmatory),
            "--require-tag", "pilot-v1",
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 3  # no arms finished, so selection stays pending
    out = tmp_path / "pilot"
    assert captured["reserved_ids_path"] == out / "reserved_ids.json"
    assert captured["monitor_rows"] == monitor
    assert captured["max_length"] is not None
    assert captured["require_tag"] == "pilot-v1"
    payload = json.loads((out / "reserved_ids.json").read_text(encoding="utf-8"))
    assert payload["monitor"] == ["m0", "m1"]
    assert payload["pilot_eval"] == ["eval-0"]
    assert payload["pilot_fit"] == ["fit-0"]
    assert payload["heal_train"] == ["heal-0"]
    assert payload["confirmatory"] == ["confirm-0"]
    assert "confirm-0" in payload["reserved_ids"]
    assert all(value not in payload["reserved_ids"] for value in ("m0", "m1", "heal-0"))


def test_missing_monitor_teacher_shards_stop_before_the_gpu_train_step(tmp_path):
    from gemmaloop.pilot.harness import MonitorTeacherDataMissing

    teacher, _block = _arms()
    arm = next(item for item in teacher if item.teacher == "e4b")
    _cache(teacher_cache_dir(tmp_path, arm.teacher))
    monitor = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    calls = []

    def execute(command, out):
        calls.append(list(command))
        if any("precompute_teacher_topk.py" in part for part in command):
            if command[command.index("--data") + 1] == str(monitor):
                return 1  # the monitor teacher precompute fails
            _cache(Path(out))
        return 0

    with pytest.raises(MonitorTeacherDataMissing):
        run_stage(
            [arm],
            root=tmp_path,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            seed=0,
            execute=execute,
            monitor_rows=monitor,
        )
    monitor_precompute = [
        command
        for command in calls
        if any("precompute_teacher_topk.py" in part for part in command)
        and command[command.index("--data") + 1] == str(monitor)
    ]
    assert monitor_precompute, "the monitor precompute must be attempted"
    assert not any(
        any("train.py" in part for part in command) for command in calls
    )


def test_run_pilot_refuses_to_start_without_a_monitor_corpus(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"problem_id": "h0", "input_ids": [1, 2]}) + "\n",
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(
        run_pilot, "run_stage", lambda *args, **kwargs: called.append(True) or []
    )
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 2
    assert called == []


def _pilot_data(tmp_path):
    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"problem_id": "h0", "input_ids": [1, 2]}) + "\n",
        encoding="utf-8",
    )
    fit = tmp_path / "fit.jsonl"
    fit.write_text(
        json.dumps({"problem_id": "f0", "input_ids": [3, 4]}) + "\n",
        encoding="utf-8",
    )
    test = tmp_path / "test.jsonl"
    test.write_text(
        json.dumps({"problem_id": "e0", "input_ids": [5, 6]}) + "\n",
        encoding="utf-8",
    )
    monitor = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    return heal, fit, test, monitor


def test_missing_frozen_tag_spends_zero_precompute(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    heal, fit, test, monitor = _pilot_data(tmp_path)
    calls = []

    def execute(command, out):
        calls.append(list(command))
        return 0

    base = [
        "--prereg", str(PREREG),
        "--heal-data", str(heal),
        "--fit-data", str(fit),
        "--test-data", str(test),
        "--monitor-rows", str(monitor),
        "--out", str(tmp_path / "pilot"),
        "--stage", "teacher",
    ]
    # No --require-tag at all: the prereg-bound run is refused up front.
    assert run_pilot.main(base, execute=execute) == 2
    assert calls == []
    # A tag that does not exist is likewise terminal before any GPU child.
    assert (
        run_pilot.main([*base, "--require-tag", "tag-that-does-not-exist"],
                       execute=execute)
        == 2
    )
    assert calls == []


def test_no_fit_or_test_data_is_refused_before_any_precompute(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"problem_id": "h0", "input_ids": [1, 2]}) + "\n",
        encoding="utf-8",
    )
    monitor = _write_monitor_corpus(
        tmp_path / "monitor" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    called = []
    monkeypatch.setattr(
        run_pilot, "run_stage", lambda *args, **kwargs: called.append(True) or []
    )
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--monitor-rows", str(monitor),
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 2
    assert called == []


def test_eval_splits_without_problem_ids_are_refused_before_any_precompute(
    tmp_path, monkeypatch, capsys
):
    import scripts.run_pilot as run_pilot

    heal, _fit, _test, monitor = _pilot_data(tmp_path)
    p1_fit = tmp_path / "p1lite-fit.jsonl"
    p1_fit.write_text(
        json.dumps({"id": "p1", "input_ids": [101, 102], "labels_start": 1}) + "\n",
        encoding="utf-8",
    )
    p1_test = tmp_path / "p1lite-test.jsonl"
    p1_test.write_text(
        json.dumps({"id": "p2", "input_ids": [103, 104], "labels_start": 1}) + "\n",
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(
        run_pilot, "run_stage", lambda *args, **kwargs: called.append(True) or []
    )
    _allow_frozen(monkeypatch)
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(p1_fit),
            "--test-data", str(p1_test),
            "--monitor-rows", str(monitor),
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 2
    assert called == []
    assert "problem_id" in capsys.readouterr().err


def test_eval_splits_without_rendered_input_ids_are_refused_before_precompute(
    tmp_path, monkeypatch, capsys
):
    import scripts.run_pilot as run_pilot

    heal, _fit, _test, monitor = _pilot_data(tmp_path)
    raw_fit = tmp_path / "raw-fit.jsonl"
    raw_fit.write_text(
        json.dumps(
            {
                "problem_id": "fit-raw",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_test = tmp_path / "raw-test.jsonl"
    raw_test.write_text(
        json.dumps(
            {
                "problem_id": "eval-raw",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(
        run_pilot, "run_stage", lambda *args, **kwargs: called.append(True) or []
    )
    _allow_frozen(monkeypatch)
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(raw_fit),
            "--test-data", str(raw_test),
            "--monitor-rows", str(monitor),
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 2
    assert called == []
    assert "render_pilot_eval_rows.py" in capsys.readouterr().err


def test_reused_run_with_a_different_monitor_corpus_is_refused(tmp_path):
    from gemmaloop.pilot.harness import RunIdentityMismatch

    teacher, _block = _arms()
    arm = teacher[0]
    cache = teacher_cache_dir(tmp_path, arm.teacher)
    _cache(cache)
    monitor_a = _write_monitor_corpus(
        tmp_path / "monitor-a" / "monitor_rows.jsonl", ["m0", "m1"]
    )
    monitor_b = _write_monitor_corpus(
        tmp_path / "monitor-b" / "monitor_rows.jsonl", ["m2", "m3"]
    )
    monitor_cache = teacher_cache_dir(tmp_path, arm.teacher) / "monitor"
    _cache(monitor_cache)
    run_dir = arm_run_dir(tmp_path, arm.arm_id)
    _complete(
        run_dir,
        arm,
        cache,
        monitor_rows=monitor_a,
        monitor_teacher_data=monitor_cache,
    )
    with pytest.raises(RunIdentityMismatch):
        run_stage(
            [arm],
            root=tmp_path,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            seed=0,
            execute=lambda command, out: 0,
            monitor_rows=monitor_b,
        )


def test_teacher_cache_complete_verifies_the_listed_shards(tmp_path):
    from gemmaloop.pilot.harness import teacher_cache_complete

    cache = tmp_path / "teacher"
    _cache_with_teacher(cache, "e4b")
    assert teacher_cache_complete(cache) is False
    (cache / "shard-00000.pt").write_bytes(b"x")
    assert teacher_cache_complete(cache) is True


def test_force_only_marks_completed_arms_for_a_rerun(tmp_path):
    teacher, _block = _arms()
    e4b = [arm for arm in teacher if arm.teacher == "e4b"]
    arm_a, arm_b = e4b[0], e4b[1]
    for arm in (arm_a, arm_b):
        _cache(teacher_cache_dir(tmp_path, arm.teacher))
    _complete(
        arm_run_dir(tmp_path, arm_a.arm_id),
        arm_a,
        teacher_cache_dir(tmp_path, arm_a.teacher),
    )
    _partial(arm_run_dir(tmp_path, arm_b.arm_id))
    calls = []

    def execute(command, out):
        calls.append(list(command))
        if any("train.py" in part for part in command):
            arm_id = command[command.index("--pilot-arm") + 1]
            arm = next(a for a in [arm_a, arm_b] if a.arm_id == arm_id)
            _complete(Path(out), arm, teacher_cache_dir(tmp_path, arm.teacher))
        return 0

    run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
        force=True,
    )
    trains = {
        command[command.index("--pilot-arm") + 1]: command
        for command in calls
        if any("train.py" in part for part in command)
    }
    assert "--rerun-complete" in trains[arm_a.arm_id]
    assert "--rerun-complete" not in trains[arm_b.arm_id]


def test_run_stage_passes_ack_abort_to_that_arms_train_child(tmp_path):
    teacher, _block = _arms()
    e4b = [arm for arm in teacher if arm.teacher == "e4b"]
    arm_a, arm_b = e4b[0], e4b[1]
    _cache(teacher_cache_dir(tmp_path, arm_a.teacher))
    calls = []

    def execute(command, out):
        calls.append(list(command))
        if any("train.py" in part for part in command):
            _complete(Path(out), arm_a, teacher_cache_dir(tmp_path, arm_a.teacher))
        return 0

    run_stage(
        [arm_a, arm_b],
        root=tmp_path,
        prereg_path=PREREG,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
        ack_aborts={arm_a.arm_id: "abort-123"},
    )
    trains = {
        command[command.index("--pilot-arm") + 1]: command
        for command in calls
        if any("train.py" in part for part in command)
    }
    assert trains[arm_a.arm_id][
        trains[arm_a.arm_id].index("--ack-abort") + 1
    ] == "abort-123"
    assert "--ack-abort" not in trains[arm_b.arm_id]


def test_run_pilot_parses_and_passes_the_ack_abort_map(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    heal, fit, test, monitor = _pilot_data(tmp_path)
    captured = {}

    def fake_stage(selected, *, root, prereg_path, heal_data, seed, force=False,
                   execute=None, reserved_ids_path=None, monitor_rows=None,
                   max_length=None, require_tag=None, ack_aborts=None,
                   teacher_runtime_freeze=None, prereg=None):
        captured["ack_aborts"] = ack_aborts
        return []

    monkeypatch.setattr(run_pilot, "run_stage", fake_stage)
    _allow_frozen(monkeypatch)
    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(fit),
            "--test-data", str(test),
            "--monitor-rows", str(monitor),
            "--ack-abort", "L-25-29-gch-e4b=abc123",
            "--out", str(tmp_path / "pilot"),
            "--stage", "teacher",
        ]
    )
    assert code == 3
    assert captured["ack_aborts"] == {"L-25-29-gch-e4b": "abc123"}

    assert (
        run_pilot.main(
            [
                "--prereg", str(PREREG),
                "--heal-data", str(heal),
                "--fit-data", str(fit),
                "--test-data", str(test),
                "--monitor-rows", str(monitor),
                "--ack-abort", "not-an-arm=abc",
                "--out", str(tmp_path / "pilot2"),
                "--stage", "teacher",
            ],
            execute=None,
        )
        == 2
    )


def test_run_pilot_writes_json_atomically(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    calls = []
    original = run_pilot.atomic_write_json

    def spy(path, payload):
        calls.append(Path(path))
        return original(path, payload)

    monkeypatch.setattr(run_pilot, "atomic_write_json", spy)
    target = tmp_path / "nested" / "reserved_ids.json"
    run_pilot._write_json(target, {"monitor": ["m0"]})
    assert calls == [target]
    assert json.loads(target.read_text(encoding="utf-8"))["monitor"] == ["m0"]


def test_evaluate_pilot_deterministic_input_errors_are_terminal_exit_2(
    tmp_path, monkeypatch
):
    import scripts.evaluate_pilot as evaluate_pilot

    monkeypatch.setattr(evaluate_pilot, "require_gpu_lock", lambda: None)
    monkeypatch.setattr(
        evaluate_pilot, "cap_cuda_allocator", lambda *a, **k: {"cap_gib": 10.0}
    )

    def bad_rows(path):
        raise ValueError("row 0 needs rendered input_ids")

    monkeypatch.setattr(evaluate_pilot, "load_eval_rows", bad_rows)
    out = tmp_path / "eval" / "evaluation.json"
    code = evaluate_pilot.main(
        [
            "--arm-id", "L-25-29-gch-e4b",
            "--prereg", str(PREREG),
            "--run-dir", str(tmp_path / "run"),
            "--heal-data", str(tmp_path / "h"),
            "--fit-data", str(tmp_path / "f"),
            "--test-data", str(tmp_path / "t"),
            "--out", str(out),
        ]
    )
    assert code == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "rendered input_ids" in payload["error"]


def test_run_pilot_refuses_a_disabled_monitor_and_smoke_before_precompute(
    tmp_path, monkeypatch, capsys
):
    """R2-M3: do not spend hours on precompute for a run that must refuse."""

    import scripts.run_pilot as run_pilot

    def boom(*args, **kwargs):
        raise AssertionError("no precompute or training may start")

    monkeypatch.setattr(run_pilot, "run_stage", boom)
    base = [
        "--heal-data", str(tmp_path / "heal"),
        "--out", str(tmp_path / "pilot"),
    ]

    assert run_pilot.main([*base, "--smoke"]) == 2
    smoke_err = capsys.readouterr().err
    assert "--smoke" in smoke_err
    assert not (tmp_path / "pilot").exists()

    monkeypatch.setenv("GEMMALOOP_HEAL_MONITOR", "0")
    assert run_pilot.main(base) == 2
    env_err = capsys.readouterr().err
    assert "GEMMALOOP_HEAL_MONITOR=0" in env_err
    assert not (tmp_path / "pilot").exists()


def test_run_pilot_refuses_before_precompute_without_shortfall_readiness(
    tmp_path, monkeypatch, capsys
):
    """The amendment guard must stop before monitor/teacher precompute."""

    import scripts.run_pilot as run_pilot
    from gemmaloop.stage0.shortfall import ShortfallReadinessError

    seen = {}

    def refuse(path):
        seen["path"] = path
        raise ShortfallReadinessError("final T-fit has 19; need 20")

    monkeypatch.setattr(run_pilot, "check_freeze_readiness", refuse)

    def boom(*args, **kwargs):
        raise AssertionError("no precompute or training may start")

    monkeypatch.setattr(run_pilot, "run_stage", boom)
    index = tmp_path / "shortfall" / "pilot-shortfall-freeze.json"
    base = [
        "--heal-data", str(tmp_path / "heal"),
        "--shortfall-freeze", str(index),
        "--out", str(tmp_path / "pilot"),
    ]
    assert run_pilot.main(base) == 2
    assert seen["path"] == index
    err = capsys.readouterr().err
    assert "pilot-shortfall readiness refused before any precompute" in err
    assert "need 20" in err
    assert not (tmp_path / "pilot").exists()
