"""End-to-end pilot orchestration on fake arm results (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_pilot as run_pilot
from gemmaloop.pilot import harness
from gemmaloop.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "prereg.yaml"


@pytest.fixture(autouse=True)
def _frozen_ok(monkeypatch):
    import scripts.run_pilot as run_pilot

    monkeypatch.setattr(run_pilot, "check_frozen_run", lambda *a, **k: None)
    monkeypatch.setattr(run_pilot, "assert_code_under", lambda *a, **k: None)
    monkeypatch.setattr(run_pilot, "check_freeze_readiness", lambda *a, **k: {})
    monkeypatch.setattr(run_pilot, "check_pilot_inputs", lambda *a, **k: {})


def _pilot_monitor_rows(tmp_path: Path) -> Path:
    path = tmp_path / "monitor" / "monitor_rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "problem_id": f"monitor-{index}",
                    "messages": [
                        {"role": "user", "content": f"q{index}"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    return path


def _fake_eval(arm, **kwargs):
    if arm.kind == "O-H":
        values = {1: 1.12}
    elif arm.teacher == "12b":
        values = {1: 1.30, 2: 1.12, 3: 1.10, 4: 1.10, 6: 1.09, 8: 1.08}
    else:
        values = {1: 1.30, 2: 1.10, 3: 1.08, 4: 1.06, 6: 1.05, 8: 1.04}
    per_n = {
        str(n): {
            "clusters": {f"p{i}": value + 0.001 * i for i in range(4)},
            "temperature": 1.0,
            "spans": {"thinking": value, "answer": value + 0.1},
            "buckets": {"<=1k": {"calibrated_nll": value}},
        }
        for n, value in values.items()
    }
    initial = {"1": {"clusters": {f"p{i}": 1.30 for i in range(4)}, "temperature": 1.0}}
    evaluation = {
        "arm_id": arm.arm_id,
        "checkpoints": {"initial": {"n": initial}, "final": {"n": per_n}},
        "teacher_kl": {"initial": 0.5, "final": 0.2, "gap_closed": 0.6},
        "flops_ratio": 1.0,
        "training": {"loss": [{"step": 1, "loss": 2.0}], "tokens": 100, "wall_seconds": 10.0, "tokens_per_second": 10.0},
    }
    return {
        "arm_id": arm.arm_id,
        "action": "evaluated",
        "returncode": 0,
        "path": str(kwargs.get("root", "")) + "/eval.json",
        "evaluation": {
            "status": "complete",
            "arm_id": arm.arm_id,
            "arm": arm.to_dict(),
            "runtime": arm.runtime,
            "evaluation": evaluation,
        },
    }


def _fake_stage(arms, *, root, **kwargs):
    _fake_stage.selected.extend(arm.arm_id for arm in arms)
    actions = harness.stage_actions(arms, root=root)
    for action in actions:
        if action["action"] == "train":
            directory = Path(action["run_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            arm = next(a for a in arms if a.arm_id == action["arm_id"])
            (directory / "run.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "tokens_consumed": 100,
                        "arm": "L" if arm.kind == "L" else "O-H",
                        "bridge": arm.bridge,
                        "loop_block": list(arm.block),
                        "run_identity_hash": "sha-" + arm.arm_id,
                        "run_config_sha256": "cfg-" + arm.arm_id,
                    }
                ),
                encoding="utf-8",
            )
            (directory / "run_identity.json").write_text(
                json.dumps(
                    {
                        "run_config": {
                            "pilot_arm": arm.arm_id,
                            "arm": "L" if arm.kind == "L" else "O-H",
                            "block": f"{arm.block[0]}-{arm.block[1]}",
                            "bridge": arm.bridge,
                        }
                    }
                ),
                encoding="utf-8",
            )
            action["action"] = "trained"
            action["wall_seconds"] = 10.0
    return actions


_fake_stage.selected = []


def test_pilot_main_runs_both_stages_selects_and_writes_the_report(tmp_path, monkeypatch):
    _fake_stage.selected = []
    heal = tmp_path / "heal.jsonl"
    heal.write_text("{}\n", encoding="utf-8")
    fit = tmp_path / "fit.jsonl"
    fit.write_text(json.dumps({"input_ids": [1, 2, 3], "cluster_id": "f"}) + "\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({"input_ids": [4, 5, 6], "cluster_id": "p"}) + "\n", encoding="utf-8")
    monitor = _pilot_monitor_rows(tmp_path)

    # One teacher-stage arm is already finished: the first launch must skip it.
    prereg = load_prereg(PREREG)
    from gemmaloop.pilot.arms import pilot_arms, teacher_stage_arms

    first = teacher_stage_arms(pilot_arms(prereg))[0]
    complete_dir = harness.arm_run_dir(tmp_path / "pilot", first.arm_id)
    complete_dir.mkdir(parents=True)
    (complete_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "arm": "L" if first.kind == "L" else "O-H",
                "bridge": first.bridge,
                "loop_block": list(first.block),
                "run_identity_hash": "sha-" + first.arm_id,
            }
        ),
        encoding="utf-8",
    )
    (complete_dir / "run_identity.json").write_text(
        json.dumps(
            {
                "run_config": {
                    "pilot_arm": first.arm_id,
                    "arm": "L" if first.kind == "L" else "O-H",
                    "block": f"{first.block[0]}-{first.block[1]}",
                    "bridge": first.bridge,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_pilot, "run_stage", _fake_stage)
    monkeypatch.setattr(run_pilot, "run_evaluation", _fake_eval)
    monkeypatch.setattr(run_pilot, "read_training_summary", lambda run_dir: {"loss": [], "tokens": 100, "wall_seconds": 10.0})
    monkeypatch.setattr(run_pilot, "teacher_shard_stats", lambda cache: {"entropy": 2.0, "top1_confidence": 0.5})
    monkeypatch.setattr(run_pilot, "teacher_cache_complete", lambda cache: True)

    argv = [
        "--prereg",
        str(PREREG),
        "--heal-data",
        str(heal),
        "--fit-data",
        str(fit),
        "--test-data",
        str(test),
        "--monitor-rows",
        str(monitor),
        "--out",
        str(tmp_path / "pilot"),
        "--stage",
        "all",
    ]
    assert run_pilot.main(argv) == 0

    results = json.loads((tmp_path / "pilot" / "pilot_results.json").read_text(encoding="utf-8"))
    assert results["teacher_step"]["chosen"] == "e4b"
    assert results["teacher_step"]["single_teacher"] is True
    assert results["teacher_step"]["estimable"] is False
    assert results["selection"]["stop"] is False
    assert results["selection"]["selected"]
    assert (tmp_path / "pilot" / "selection.json").is_file()
    assert (tmp_path / "pilot" / "report" / "why.md").is_file()
    reserved = json.loads(
        (tmp_path / "pilot" / "reserved_ids.json").read_text(encoding="utf-8")
    )
    assert {"monitor", "heal_train", "pilot_fit", "pilot_eval"} <= set(
        reserved["splits"]
    )
    assert reserved["monitor"] == ["monitor-0", "monitor-1"]
    assert reserved["reserved_ids"]
    assert (tmp_path / "pilot" / "report" / "why_per_n.png").is_file()

    actions = results["actions"]
    by_arm = {action["arm_id"]: action for action in actions}
    assert by_arm[first.arm_id]["action"] == "skipped"
    # The block stage keeps the pre-listed E4B arms even when 12B is the
    # teacher-step winner; ``chosen`` only labels the final recipe.
    block_ids = [
        arm_id
        for arm_id in _fake_stage.selected
        if not arm_id.startswith("L-25-29-gch-12b") and arm_id != "OH-25-29-12b"
    ]
    assert block_ids
    assert all("12b" not in arm_id for arm_id in block_ids)
    assert results["chosen_teacher"] == "e4b"
    assert results["selection"]["eligible"]

    # A second launch is a no-op for every arm.
    assert run_pilot.main(argv) == 0
    rerun = json.loads((tmp_path / "pilot" / "pilot_results.json").read_text(encoding="utf-8"))
    latest = rerun["actions"][-11:]
    assert {action["action"] for action in latest} == {"skipped"}


def test_split_overlap_is_refused_before_any_arm_runs(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot
    from gemmaloop.pilot.splits import SplitOverlapError, assert_disjoint_splits

    shared = tmp_path / "shared.jsonl"
    shared.write_text(
        json.dumps({"input_ids": [1, 2, 3], "cluster_id": "same"}) + "\n",
        encoding="utf-8",
    )
    other = tmp_path / "other.jsonl"
    other.write_text(
        json.dumps({"input_ids": [4, 5, 6], "cluster_id": "other"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SplitOverlapError):
        assert_disjoint_splits({"fit": shared, "test": shared})
    with pytest.raises(SplitOverlapError):
        assert_disjoint_splits({"heal": shared, "fit": other, "test": shared})
    assert_disjoint_splits({"heal": shared, "fit": other})
    monitor = _pilot_monitor_rows(tmp_path)

    called = []
    monkeypatch.setattr(
        run_pilot, "run_stage", lambda *args, **kwargs: called.append(True) or []
    )
    code = run_pilot.main(
        [
            "--prereg",
            str(PREREG),
            "--heal-data",
            str(shared),
            "--fit-data",
            str(shared),
            "--test-data",
            str(shared),
            "--monitor-rows",
            str(monitor),
            "--out",
            str(tmp_path / "pilot"),
            "--stage",
            "teacher",
        ]
    )
    assert code == 2
    assert called == []


def test_partial_failure_is_unevaluable_not_stop(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot
    from gemmaloop.pilot import harness
    from gemmaloop.pilot.arms import block_stage_arms, pilot_arms
    from gemmaloop.prereg import load_prereg

    heal = tmp_path / "heal.jsonl"
    heal.write_text(json.dumps({"input_ids": [1, 2, 3], "cluster_id": "h"}) + "\n", encoding="utf-8")
    fit = tmp_path / "fit.jsonl"
    fit.write_text(json.dumps({"input_ids": [4, 5, 6], "cluster_id": "f"}) + "\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({"input_ids": [7, 8, 9], "cluster_id": "p"}) + "\n", encoding="utf-8")
    monitor = _pilot_monitor_rows(tmp_path)

    failed_id = block_stage_arms(pilot_arms(load_prereg(PREREG)))[0].arm_id

    def failing_stage(arms, *, root, **kwargs):
        actions = _fake_stage(arms, root=root, **kwargs)
        for action in actions:
            if action["arm_id"] == failed_id:
                (Path(action["run_dir"]) / "run.json").unlink()
                action["action"] = "train_failed"
                action["returncode"] = 1
        return actions

    monkeypatch.setattr(run_pilot, "run_stage", failing_stage)
    monkeypatch.setattr(run_pilot, "run_evaluation", _fake_eval)
    monkeypatch.setattr(run_pilot, "read_training_summary", lambda run_dir: {"loss": [], "tokens": 100})
    monkeypatch.setattr(run_pilot, "teacher_shard_stats", lambda cache: {})
    monkeypatch.setattr(run_pilot, "teacher_cache_complete", lambda cache: True)

    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(fit),
            "--test-data", str(test),
            "--monitor-rows", str(monitor),
            "--out", str(tmp_path / "pilot"),
            "--stage", "all",
        ]
    )
    assert code == 3
    out = tmp_path / "pilot"
    assert not (out / "selection.json").is_file()
    pending = json.loads((out / "selection_pending.json").read_text(encoding="utf-8"))
    assert pending["status"] == "unevaluable"
    assert failed_id in pending["missing"] + pending["failed"]
    assert "stop" not in pending


def test_evaluation_failure_is_unevaluable_not_stop(tmp_path, monkeypatch):
    import scripts.run_pilot as run_pilot
    from gemmaloop.pilot.arms import block_stage_arms, pilot_arms
    from gemmaloop.prereg import load_prereg

    heal = tmp_path / "heal.jsonl"
    heal.write_text(json.dumps({"input_ids": [1, 2, 3], "cluster_id": "h"}) + "\n", encoding="utf-8")
    fit = tmp_path / "fit.jsonl"
    fit.write_text(json.dumps({"input_ids": [4, 5, 6], "cluster_id": "f"}) + "\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({"input_ids": [7, 8, 9], "cluster_id": "p"}) + "\n", encoding="utf-8")
    monitor = _pilot_monitor_rows(tmp_path)
    failed_id = block_stage_arms(pilot_arms(load_prereg(PREREG)))[0].arm_id

    def evaluate(arm, **kwargs):
        if arm.arm_id == failed_id:
            return {
                "arm_id": arm.arm_id,
                "action": "eval_failed",
                "returncode": 1,
                "evaluation": None,
                "path": str(kwargs.get("root", "")) + "/eval.json",
            }
        return _fake_eval(arm, **kwargs)

    monkeypatch.setattr(run_pilot, "run_stage", _fake_stage)
    monkeypatch.setattr(run_pilot, "run_evaluation", evaluate)
    monkeypatch.setattr(run_pilot, "read_training_summary", lambda run_dir: {"loss": [], "tokens": 100})
    monkeypatch.setattr(run_pilot, "teacher_shard_stats", lambda cache: {})
    monkeypatch.setattr(run_pilot, "teacher_cache_complete", lambda cache: True)

    code = run_pilot.main(
        [
            "--prereg", str(PREREG),
            "--heal-data", str(heal),
            "--fit-data", str(fit),
            "--test-data", str(test),
            "--monitor-rows", str(monitor),
            "--out", str(tmp_path / "pilot"),
            "--stage", "all",
        ]
    )
    assert code == 3
    pending = json.loads(
        (tmp_path / "pilot" / "selection_pending.json").read_text(encoding="utf-8")
    )
    assert failed_id in pending["failed"]
    assert not (tmp_path / "pilot" / "selection.json").exists()
