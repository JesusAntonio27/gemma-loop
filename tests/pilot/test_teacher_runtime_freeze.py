"""B2: the frozen qualified teacher runtime binds precompute, cache and runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gemmaloop.pilot.arms import pilot_arms
from gemmaloop.pilot.harness import (
    TeacherRuntimeNotFrozen,
    build_precompute_command,
    run_stage,
    teacher_cache_complete,
)
from gemmaloop.prereg import load_prereg
from gemmaloop.teacher.runtime_freeze import (
    FREEZE_FORMAT,
    FrozenTeacherRuntime,
    TeacherRuntimeFreeze,
    TeacherRuntimeFreezeError,
    build_teacher_runtime_freeze,
    load_teacher_runtime_freeze,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


def _qualification(
    path: Path,
    *,
    key: str,
    runtime: str,
    decision: str,
) -> Path:
    payload = {
        "teacher": {"key": key, "runtime": runtime, "revision": "rev"},
        "decision": {
            "decision": decision,
            "passed": decision == "pass",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _freeze(
    path: Path,
    *,
    e4b_runtime: str = "int8",
    decision: str = "pass",
    teacher_key: str = "e4b",
) -> Path:
    qualification = _qualification(
        path.with_name("e4b.json"),
        key=teacher_key,
        runtime=e4b_runtime,
        decision=decision,
    )
    payload = {
        "format": FREEZE_FORMAT,
        "teachers": {
            "e4b": FrozenTeacherRuntime(
                teacher="e4b",
                teacher_key=teacher_key,
                runtime=e4b_runtime,
                qualification_json=str(qualification.resolve()),
                qualification_sha256=hashlib.sha256(
                    qualification.read_bytes()
                ).hexdigest(),
                decision=decision,
            ).to_dict()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_freeze_binds_the_passing_qualification_and_sha(tmp_path):
    qualification = _qualification(
        tmp_path / "e4b-int8.json",
        key="e4b",
        runtime="int8",
        decision="pass",
    )
    out = tmp_path / "teacher_runtime_freeze.json"
    freeze = build_teacher_runtime_freeze([qualification], out=out)
    assert freeze.runtime_for("e4b") == "int8"
    entry = freeze.teachers["e4b"]
    assert entry.qualification_json == str(qualification.resolve())
    assert entry.qualification_sha256 == hashlib.sha256(
        qualification.read_bytes()
    ).hexdigest()
    assert entry.decision == "pass"
    assert load_teacher_runtime_freeze(out).runtime_for("e4b") == "int8"


def test_build_freeze_refuses_a_failed_qualification(tmp_path):
    failed = _qualification(
        tmp_path / "12b-nf4.json",
        key="12b_qat",
        runtime="nf4",
        decision="fail_next_runtime",
    )
    with pytest.raises(TeacherRuntimeFreezeError, match="no passing runtime"):
        build_teacher_runtime_freeze([failed], out=tmp_path / "freeze.json")
    assert not (tmp_path / "freeze.json").exists()


def test_freeze_cli_exits_2_on_a_failed_runtime(tmp_path, capsys):
    import scripts.freeze_teacher_runtime as cli

    failed = _qualification(
        tmp_path / "12b-nf4.json",
        key="12b_qat",
        runtime="nf4",
        decision="fail_next_runtime",
    )
    code = cli.main(
        [
            "--qualification",
            str(failed),
            "--out",
            str(tmp_path / "freeze.json"),
        ]
    )
    assert code == 2
    assert "no passing runtime" in capsys.readouterr().err


def test_freeze_cli_writes_the_e4b_int8_record(tmp_path):
    import scripts.freeze_teacher_runtime as cli

    passing = _qualification(
        tmp_path / "e4b-int8.json",
        key="e4b",
        runtime="int8",
        decision="pass",
    )
    out = tmp_path / "freeze.json"
    assert cli.main(["--qualification", str(passing), "--out", str(out)]) == 0
    assert load_teacher_runtime_freeze(out).runtime_for("e4b") == "int8"


def test_precompute_command_passes_the_frozen_runtime():
    command = build_precompute_command(
        "e4b",
        runtime="int8",
        heal_data="heal",
        cache_dir="cache",
    )
    assert command[command.index("--runtime") + 1] == "int8"


def test_teacher_cache_complete_rejects_a_bf16_cache_for_an_int8_freeze(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "shard-00000.pt").write_bytes(b"shard")
    freeze = TeacherRuntimeFreeze(
        teachers={
            "e4b": FrozenTeacherRuntime(
                teacher="e4b",
                teacher_key="e4b",
                runtime="int8",
                qualification_json="q.json",
                qualification_sha256="a" * 64,
            )
        }
    )
    manifest = {
        "complete": True,
        "shards": [{"file": "shard-00000.pt"}],
        "teacher": {"key": "e4b", "runtime": "bf16"},
    }
    (cache / "teacher_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert (
        teacher_cache_complete(cache, freeze=freeze, teacher="e4b") is False
    )
    manifest["teacher"]["runtime"] = "int8"
    (cache / "teacher_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert teacher_cache_complete(cache, freeze=freeze, teacher="e4b") is True


def test_run_stage_refuses_a_non_excluded_teacher_with_no_frozen_runtime(
    tmp_path, monkeypatch
):
    import gemmaloop.pilot.harness as harness

    arms = pilot_arms(load_prereg(PREREG))
    arm_e4b = next(arm for arm in arms if arm.teacher == "e4b")
    freeze = TeacherRuntimeFreeze(teachers={})
    monkeypatch.setattr(
        harness, "load_teacher_runtime_freeze", lambda path=None: freeze
    )
    calls = []

    with pytest.raises(TeacherRuntimeNotFrozen, match="e4b"):
        run_stage(
            [arm_e4b],
            root=tmp_path,
            prereg_path=PREREG,
            heal_data=tmp_path / "heal.jsonl",
            seed=0,
            execute=lambda command, out: calls.append(command) or 0,
            teacher_runtime_freeze=tmp_path / "freeze.json",
        )
    assert calls == []


def _e4b_arm():
    return next(
        arm
        for arm in pilot_arms(load_prereg(PREREG))
        if arm.teacher == "e4b"
    )


def _cache_and_arm(tmp_path):
    arm = _e4b_arm()
    cache = tmp_path / "teacher-shards" / "e4b"
    cache.mkdir(parents=True)
    manifest = {
        "complete": True,
        "shards": [{"file": "shard-00000.pt"}],
        "teacher": {"key": "e4b", "runtime": "int8"},
    }
    (cache / "teacher_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (cache / "shard-00000.pt").write_bytes(b"shard")
    return arm, cache


def test_run_identity_accepts_a_train_shaped_completed_arm(tmp_path):
    from gemmaloop.pilot.harness import run_identity_mismatches
    from tests.train_layout import train_shaped_complete_run

    arm, cache = _cache_and_arm(tmp_path)
    run_dir = tmp_path / "arms" / arm.arm_id
    train_shaped_complete_run(
        run_dir,
        arm=arm,
        cache=cache,
        prereg=PREREG,
        teacher_key="e4b",
        runtime="int8",
    )
    freeze = load_teacher_runtime_freeze(
        _freeze(tmp_path / "freeze.json", e4b_runtime="int8")
    )
    problems = run_identity_mismatches(
        run_dir,
        arm,
        teacher_data=cache,
        teacher_runtime_freeze=freeze,
    )
    assert problems == []


def test_run_identity_rejects_a_run_bound_to_another_runtime(tmp_path):
    from gemmaloop.pilot.harness import run_identity_mismatches
    from tests.train_layout import train_shaped_complete_run

    arm, cache = _cache_and_arm(tmp_path)
    run_dir = tmp_path / "arms" / arm.arm_id
    train_shaped_complete_run(
        run_dir,
        arm=arm,
        cache=cache,
        prereg=PREREG,
        teacher_key="e4b",
        runtime="bf16",
    )
    freeze = load_teacher_runtime_freeze(
        _freeze(tmp_path / "freeze.json", e4b_runtime="int8")
    )
    problems = run_identity_mismatches(
        run_dir,
        arm,
        teacher_data=cache,
        teacher_runtime_freeze=freeze,
    )
    assert any("teacher_runtime" in problem for problem in problems)
    assert any("bf16" in problem and "int8" in problem for problem in problems)


def test_freeze_load_rechecks_the_qualification_sha(tmp_path):
    freeze_path = _freeze(tmp_path / "freeze.json")
    qualification = tmp_path / "e4b.json"
    qualification.write_text(
        json.dumps(
            {
                "teacher": {"key": "e4b", "runtime": "int8", "revision": "rev"},
                "decision": {"decision": "pass", "passed": True},
                "edited": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TeacherRuntimeFreezeError, match="changed"):
        load_teacher_runtime_freeze(freeze_path)


def test_freeze_load_rechecks_the_qualification_decision(tmp_path):
    qualification = _qualification(
        tmp_path / "e4b.json",
        key="e4b",
        runtime="int8",
        decision="fail_next_runtime",
    )
    payload = {
        "format": FREEZE_FORMAT,
        "teachers": {
            "e4b": FrozenTeacherRuntime(
                teacher="e4b",
                teacher_key="e4b",
                runtime="int8",
                qualification_json=str(qualification.resolve()),
                qualification_sha256=hashlib.sha256(
                    qualification.read_bytes()
                ).hexdigest(),
                decision="pass",
            ).to_dict()
        },
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TeacherRuntimeFreezeError, match="no longer passes"):
        load_teacher_runtime_freeze(freeze_path)


def test_freeze_load_requires_the_harness_teacher_key(tmp_path):
    qualification = _qualification(
        tmp_path / "e4b.json",
        key="12b_qat",
        runtime="nf4",
        decision="pass",
    )
    payload = {
        "format": FREEZE_FORMAT,
        "teachers": {
            "e4b": FrozenTeacherRuntime(
                teacher="e4b",
                teacher_key="12b_qat",
                runtime="nf4",
                qualification_json=str(qualification.resolve()),
                qualification_sha256=hashlib.sha256(
                    qualification.read_bytes()
                ).hexdigest(),
                decision="pass",
            ).to_dict()
        },
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TeacherRuntimeFreezeError, match="teacher key"):
        load_teacher_runtime_freeze(freeze_path)


def test_precompute_uses_the_freeze_and_never_defaults_to_primary(tmp_path):
    import scripts.precompute_teacher_topk as precompute

    missing = tmp_path / "missing_freeze.json"
    with pytest.raises(ValueError, match="--runtime is required"):
        precompute.resolve_precompute_runtime(
            "e4b", None, freeze_path=missing
        )

    freeze = _freeze(tmp_path / "freeze.json", e4b_runtime="int8")
    # The registered primary for E4B is bf16, but the freeze says int8.
    assert (
        precompute.resolve_precompute_runtime("e4b", None, freeze_path=freeze)
        == "int8"
    )
    # An explicit request still wins.
    assert (
        precompute.resolve_precompute_runtime("e4b", "bf16", freeze_path=freeze)
        == "bf16"
    )
