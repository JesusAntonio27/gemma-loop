"""Round-8h teacher exclusion: the pilot teacher stage runs the E4B pair only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmaloop.pilot.arms import (
    block_stage_arms,
    pilot_arms,
    teacher_stage_arms,
)
from gemmaloop.pilot.harness import (
    EXCLUDED_RUNTIME_REASON,
    registered_teacher_exclusions,
    run_stage,
    teacher_cache_dir,
    teacher_cache_complete,
)
from gemmaloop.pilot.report import write_report
from gemmaloop.pilot.selection import SINGLE_TEACHER_REASON, select_single_teacher
from gemmaloop.prereg import load_prereg
from gemmaloop.teacher.runtime_freeze import (
    FrozenTeacherRuntime,
    TeacherRuntimeFreeze,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


def _e4b_freeze() -> TeacherRuntimeFreeze:
    return TeacherRuntimeFreeze(
        teachers={
            "e4b": FrozenTeacherRuntime(
                teacher="e4b",
                teacher_key="e4b",
                runtime="int8",
                qualification_json="e4b-int8.json",
                qualification_sha256="a" * 64,
                decision="pass",
            )
        }
    )


def _e4b_cache(root: Path) -> Path:
    cache = teacher_cache_dir(root, "e4b")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "shards": [{"file": "shard-00000.pt"}],
                "teacher": {"key": "e4b", "runtime": "int8"},
            }
        ),
        encoding="utf-8",
    )
    (cache / "shard-00000.pt").write_bytes(b"shard")
    return cache


def _use_freeze(monkeypatch, freeze):
    import gemmaloop.pilot.harness as harness

    monkeypatch.setattr(
        harness, "load_teacher_runtime_freeze", lambda path=None: freeze
    )


def test_registered_exclusion_is_read_from_the_round8h_addendum():
    prereg = load_prereg(PREREG)
    assert registered_teacher_exclusions(prereg) == {
        "12b": EXCLUDED_RUNTIME_REASON
    }


def test_teacher_stage_runs_the_two_e4b_arms_and_records_both_exclusions(
    tmp_path, monkeypatch
):
    freeze = _e4b_freeze()
    _use_freeze(monkeypatch, freeze)
    cache = _e4b_cache(tmp_path)
    assert teacher_cache_complete(cache, freeze=freeze, teacher="e4b")

    prereg = load_prereg(PREREG)
    teacher_arms = teacher_stage_arms(pilot_arms(prereg))
    calls = []

    def execute(command, out):
        calls.append([str(part) for part in command])
        return 0

    actions = run_stage(
        teacher_arms,
        root=tmp_path,
        prereg_path=PREREG,
        prereg=prereg,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=execute,
        teacher_runtime_freeze=tmp_path / "freeze.json",
    )

    by_id = {action["arm_id"]: action for action in actions}
    assert by_id["L-25-29-gch-e4b"]["action"] == "trained"
    assert by_id["OH-25-29-e4b"]["action"] == "trained"
    for arm_id in ("L-25-29-gch-12b", "OH-25-29-12b"):
        assert by_id[arm_id]["action"] == "excluded"
        assert by_id[arm_id]["reason"] == EXCLUDED_RUNTIME_REASON
        assert by_id[arm_id]["returncode"] is None

    trained = {
        action["arm_id"]
        for action in actions
        if action["action"] == "trained"
    }
    assert trained == {"L-25-29-gch-e4b", "OH-25-29-e4b"}
    # No child command may mention the excluded teacher: no precompute and no
    # training child for either 12B arm.
    assert calls
    joined = " ".join(" ".join(command) for command in calls)
    assert "12b" not in joined.lower()


def test_block_stage_still_runs_every_registered_block_arm(tmp_path, monkeypatch):
    freeze = _e4b_freeze()
    _use_freeze(monkeypatch, freeze)
    _e4b_cache(tmp_path)

    prereg = load_prereg(PREREG)
    block_arms = block_stage_arms(pilot_arms(prereg))
    assert len(block_arms) == 7
    assert all(arm.teacher == "e4b" for arm in block_arms)

    actions = run_stage(
        block_arms,
        root=tmp_path,
        prereg_path=PREREG,
        prereg=prereg,
        heal_data=tmp_path / "heal.jsonl",
        seed=0,
        execute=lambda command, out: 0,
        teacher_runtime_freeze=tmp_path / "freeze.json",
    )
    assert [action["action"] for action in actions] == ["trained"] * 7
    assert not any(action["action"] == "excluded" for action in actions)


def test_select_single_teacher_records_a_non_estimable_comparison():
    step = select_single_teacher(
        {
            "L-25-29-gch-e4b": {"arm_id": "L-25-29-gch-e4b", "nll": {4: 0.8}},
            "OH-25-29-e4b": {"arm_id": "OH-25-29-e4b", "nll": {1: 0.95}},
        },
        excluded_teachers=("12b",),
        baseline_nll=1.0,
    )
    assert step["chosen"] == "e4b"
    assert step["recipe_teacher"] == "e4b"
    assert step["estimable"] is False
    assert step["single_teacher"] is True
    assert step["excluded_teachers"] == ["12b"]
    assert step["reason"] == SINGLE_TEACHER_REASON
    assert "not estimable" in step["reason"]
    assert step["e4b_gain"]["mean"] == pytest.approx(((1.0 - 0.8) + (1.0 - 0.95)) / 2.0)


def test_report_states_the_teacher_comparison_is_not_estimable(tmp_path):
    step = select_single_teacher(
        {
            "L-25-29-gch-e4b": {"arm_id": "L-25-29-gch-e4b", "nll": {4: 0.8}},
            "OH-25-29-e4b": {"arm_id": "OH-25-29-e4b", "nll": {1: 0.95}},
        },
        excluded_teachers=("12b",),
        baseline_nll=1.0,
    )
    payload = {
        "pilot": {"heal_tokens": 3_000_000},
        "arms": {},
        "teacher_step": step,
        "selection": {
            "stop": True,
            "selected": None,
            "eligible": [],
            "filter_applied": False,
            "monotonic_pass": [],
        },
        "teachers": {},
        "qualification": {},
        "baseline_nll": 1.0,
        "arm_specs": [],
    }
    paths = write_report(payload, tmp_path / "report")
    text = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "not estimable" in text
    assert "Recipe teacher: **E4B**" in text
    assert "versus 12B" not in text


def _e4b_summaries() -> dict[str, dict]:
    return {
        "L-25-29-gch-e4b": {
            "arm_id": "L-25-29-gch-e4b",
            "nll": {4: 0.8},
        },
        "OH-25-29-e4b": {
            "arm_id": "OH-25-29-e4b",
            "nll": {1: 0.95},
        },
    }


def _patch_teacher_step_inputs(monkeypatch):
    import scripts.run_pilot as run_pilot

    summaries = _e4b_summaries()
    monkeypatch.setattr(
        run_pilot, "_baseline_nll", lambda arms, results: 1.0
    )
    monkeypatch.setattr(
        run_pilot,
        "_canonical_teacher_map",
        lambda arms, summaries_arg, teacher: (
            dict(summaries) if teacher == "e4b" else {}
        ),
    )


def test_teacher_step_naming_an_excluded_teacher_is_not_reusable():
    import scripts.run_pilot as run_pilot

    excluded = {"12b": EXCLUDED_RUNTIME_REASON}
    assert not run_pilot._teacher_step_is_reusable(
        {"chosen": "12b"}, excluded
    )
    assert not run_pilot._teacher_step_is_reusable(
        {"chosen": "e4b", "recipe_teacher": "12b"}, excluded
    )
    # Produced before the exclusion: no recorded exclusion state.
    assert not run_pilot._teacher_step_is_reusable(
        {"chosen": "e4b"}, excluded
    )
    assert run_pilot._teacher_step_is_reusable(
        {"chosen": "e4b", "excluded_teachers": ["12b"]}, excluded
    )
    # With no registered exclusion, a two-teacher record is reusable.
    assert run_pilot._teacher_step_is_reusable({"chosen": "12b"}, {})


def test_finalize_teacher_step_rejects_a_stale_12b_record(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot

    prereg = load_prereg(PREREG)
    arms = pilot_arms(prereg)
    _patch_teacher_step_inputs(monkeypatch)
    stale = {"chosen": "12b", "tie_break": True, "reason": "stale"}
    (tmp_path / run_pilot.TEACHER_STEP_FILE).write_text(
        json.dumps(stale), encoding="utf-8"
    )
    results = {
        "teacher_step": dict(stale),
        "chosen_teacher": "12b",
        "arms": {},
    }

    step = run_pilot._finalize_teacher_step(
        prereg,
        arms,
        out=tmp_path,
        results=results,
        summaries={},
        excluded_teachers={"12b": EXCLUDED_RUNTIME_REASON},
    )

    assert step is not None
    assert step["chosen"] == "e4b"
    assert step["recipe_teacher"] == "e4b"
    assert step["single_teacher"] is True
    assert step["excluded_teachers"] == ["12b"]
    assert results["teacher_step"] == step
    saved = json.loads(
        (tmp_path / run_pilot.TEACHER_STEP_FILE).read_text(encoding="utf-8")
    )
    assert saved["chosen"] == "e4b"


def test_finalize_teacher_step_rejects_a_different_exclusion_state(
    tmp_path, monkeypatch
):
    import scripts.run_pilot as run_pilot

    prereg = load_prereg(PREREG)
    arms = pilot_arms(prereg)
    _patch_teacher_step_inputs(monkeypatch)
    # A pre-exclusion two-teacher record may even name E4B; it was still
    # produced under a different exclusion state and must be recomputed.
    old = {"chosen": "e4b", "tie_break": False, "margin_12b_minus_e4b": 0.0}
    results = {"teacher_step": dict(old), "arms": {}}

    step = run_pilot._finalize_teacher_step(
        prereg,
        arms,
        out=tmp_path,
        results=results,
        summaries={},
        excluded_teachers={"12b": EXCLUDED_RUNTIME_REASON},
    )

    assert step is not None
    assert step["single_teacher"] is True
    assert step["excluded_teachers"] == ["12b"]
    assert step is not old
    assert results["teacher_step"] == step


def test_finalize_teacher_step_reuses_a_record_with_the_same_exclusion_state(
    tmp_path,
):
    import scripts.run_pilot as run_pilot

    prereg = load_prereg(PREREG)
    valid = {
        "chosen": "e4b",
        "recipe_teacher": "e4b",
        "single_teacher": True,
        "excluded_teachers": ["12b"],
    }
    path = tmp_path / run_pilot.TEACHER_STEP_FILE
    path.write_text(json.dumps(valid), encoding="utf-8")
    results = {
        "teacher_step": dict(valid),
        "chosen_teacher": "e4b",
        "arms": {},
    }

    step = run_pilot._finalize_teacher_step(
        prereg,
        pilot_arms(prereg),
        out=tmp_path,
        results=results,
        summaries={},
        excluded_teachers={"12b": EXCLUDED_RUNTIME_REASON},
    )

    assert step == valid
    assert path.is_file()


def test_drop_stale_teacher_step_removes_the_file_and_results(tmp_path):
    import scripts.run_pilot as run_pilot

    stale = {"chosen": "12b"}
    path = tmp_path / run_pilot.TEACHER_STEP_FILE
    path.write_text(json.dumps(stale), encoding="utf-8")
    results = {"teacher_step": dict(stale), "chosen_teacher": "12b"}

    run_pilot._drop_stale_teacher_step(
        tmp_path, results, {"12b": EXCLUDED_RUNTIME_REASON}
    )

    assert results["teacher_step"] is None
    assert results["chosen_teacher"] is None
    assert not path.exists()


def test_drop_stale_teacher_step_keeps_a_matching_record(tmp_path):
    import scripts.run_pilot as run_pilot

    valid = {
        "chosen": "e4b",
        "recipe_teacher": "e4b",
        "excluded_teachers": ["12b"],
    }
    path = tmp_path / run_pilot.TEACHER_STEP_FILE
    path.write_text(json.dumps(valid), encoding="utf-8")
    results = {"teacher_step": dict(valid), "chosen_teacher": "e4b"}

    run_pilot._drop_stale_teacher_step(
        tmp_path, results, {"12b": EXCLUDED_RUNTIME_REASON}
    )

    assert results["teacher_step"] == valid
    assert results["chosen_teacher"] == "e4b"
    assert path.is_file()
