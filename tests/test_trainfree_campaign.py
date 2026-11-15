"""Campaign plan, resume, ordering and the descriptive B2 block rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import scripts.trainfree_campaign as campaign
from gemmaloop.eval.results import ItemResult, RunResult
from gemmaloop.gpulock import acquire


def _paths(tmp_path: Path) -> dict[str, str]:
    return {
        "plan": str(tmp_path / "plan.json"),
        "corpus": str(tmp_path / "corpus.jsonl"),
        "results_root": str(tmp_path / "results"),
        "stage_a": str(tmp_path / "results" / "stageA"),
        "eval_config": str(tmp_path / "eval_config.yaml"),
        "heartbeat": str(tmp_path / "heartbeat.json"),
        "heartbeat_dir": str(tmp_path),
    }


def test_stage_a_has_every_block_bridge_and_n_in_order():
    configs = campaign.stage_a_configs()
    assert len(configs) == 3 * 2 * 4
    assert [config["n"] for config in configs[:6]] == [1] * 6
    assert [config["job"] for config in configs[:6]] == ["A_n1"] * 6
    blocks = {tuple(config["block"]) for config in configs}
    assert blocks == {(25, 29), (16, 18), (15, 29)}
    assert {config["bridge"] for config in configs} == {"none", "damped"}


def test_stage_b_and_b2_are_keyed_by_config():
    stage_b = campaign.stage_b_configs()
    assert len(stage_b) == 7 * 2
    assert stage_b[0]["variant"] == "O"
    assert stage_b[0]["n"] == 1
    assert {entry["variant"] for entry in stage_b} == {
        "O",
        "R_b25-29_n2",
        "R-damped_b25-29_n2",
        "R_b16-18_n2",
        "R-damped_b16-18_n2",
        "R_b15-29_n2",
        "R-damped_b15-29_n2",
    }
    stage_b2 = campaign.stage_b2_configs((16, 18))
    assert len(stage_b2) == 2 * 2 * 2
    assert {entry["n"] for entry in stage_b2} == {3, 4}
    assert all(tuple(entry["block"]) == (16, 18) for entry in stage_b2)


def test_b2_rule_takes_lowest_nll_at_n2_and_breaks_ties():
    payloads = [
        {
            "variants": [
                {"block": [25, 29], "bridge": "none", "n_loops": 2, "nll_n_calibrated": 3.0},
                {"block": [25, 29], "bridge": "damped", "n_loops": 2, "nll_n_calibrated": 2.5},
                {"block": [16, 18], "bridge": "none", "n_loops": 2, "nll_n_calibrated": 2.6},
                {"block": [16, 18], "bridge": "damped", "n_loops": 2, "nll_n_calibrated": 2.4},
                {"block": [15, 29], "bridge": "none", "n_loops": 3, "nll_n_calibrated": 0.1},
            ]
        }
    ]
    assert campaign.choose_b2_block(payloads) == (16, 18)

    tie = [
        {
            "variants": [
                {"block": [25, 29], "bridge": "none", "n_loops": 2, "nll_n_calibrated": 2.0},
                {"block": [15, 29], "bridge": "damped", "n_loops": 2, "nll_n_calibrated": 2.0},
            ]
        }
    ]
    assert campaign.choose_b2_block(tie) == (15, 29)


def test_eval_config_uses_one_sample_and_registered_sampling(tmp_path):
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(
        yaml.safe_dump(
            {
                "tag": "prereg-v1",
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_new_tokens": 32768,
                    "enable_thinking": True,
                },
                "benches": {
                    "lcb": {"limit": 182, "samples": 4, "seed": 0},
                    "gpqa": {"limit": 198, "samples": 4, "seed": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "derived.yaml"
    campaign.ensure_eval_config(prereg, out)
    derived = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert derived["tag"] == campaign.EVAL_TAG
    assert derived["sampling"]["max_new_tokens"] == 32768
    assert derived["benches"]["lcb"]["samples"] == 1
    assert derived["benches"]["gpqa"]["samples"] == 1
    assert derived["benches"]["lcb"]["limit"] == 182
    assert derived["benches"]["gpqa"]["limit"] == 198


def _write_stage_a_output(command: list[str], plan: dict) -> int:
    out = Path(command[command.index("--out-json") + 1])
    n = int(command[command.index("--n-loops") + 1])
    entries = [
        entry
        for entry in plan["configs"]
        if entry["stage"] == "A" and int(entry["n"]) == n
    ]
    payload = {
        "variants": [
            {
                "block": entry["block"],
                "bridge": entry["bridge"],
                "n_loops": n,
                "nll_n_calibrated": 1.0 + int(entry["n"]),
            }
            for entry in entries
        ]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return 0


def test_stage_a_resume_and_ordering(tmp_path):
    plan = campaign.build_plan(_paths(tmp_path))
    calls: list[list[str]] = []

    def runner(command, env):
        calls.append(list(command))
        return _write_stage_a_output(list(command), plan)

    stage_dir = tmp_path / "results" / "stageA"
    campaign.run_stage_a(
        plan,
        stage_dir=stage_dir,
        corpus=tmp_path / "corpus.jsonl",
        runner=runner,
        env={},
        plan_path=tmp_path / "plan.json",
    )
    assert plan["stages"]["A"] == "done"
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))[
        "stages"
    ]["A"] == "done"
    assert [int(command[command.index("--n-loops") + 1]) for command in calls] == [
        1,
        2,
        3,
        4,
    ]
    assert all(entry["status"] == "done" for entry in plan["configs"] if entry["stage"] == "A")

    calls.clear()
    campaign.run_stage_a(
        plan,
        stage_dir=stage_dir,
        corpus=tmp_path / "corpus.jsonl",
        runner=runner,
        env={},
        plan_path=tmp_path / "plan.json",
    )
    assert calls == []


def _fake_eval_runner(command: list[str], env: dict, plan: dict) -> int:
    out_dir = Path(command[command.index("--out-dir") + 1])
    variant = command[command.index("--variant") + 1]
    entries = [
        entry
        for entry in plan["configs"]
        if entry.get("variant") == variant and entry["stage"] == "B"
    ]
    for entry in entries:
        n_loops = 0 if variant == "O" else int(entry["n"])
        run = RunResult(
            variant=variant,
            size="e2b",
            n_loops=n_loops,
            bench=entry["bench"],
            prereg_tag=campaign.EVAL_TAG,
            loop_block=None if entry["block"] is None else tuple(entry["block"]),
            bridge=entry["bridge"],
            items=[
                ItemResult(
                    item_id="q1",
                    sample_idx=0,
                    correct=True,
                    completion_tokens=7,
                    finish_reason="stop",
                )
            ],
        )
        run.save(out_dir / variant / f"{entry['bench']}.json")
    return 0


def test_stage_b_resume_scans_results_and_keeps_configs_separate(tmp_path):
    plan = campaign.build_plan(_paths(tmp_path))
    calls: list[str] = []

    def runner(command, env):
        calls.append(command[command.index("--variant") + 1])
        return _fake_eval_runner(list(command), env, plan)

    campaign.run_eval_stage(
        plan,
        "B",
        results_root=tmp_path / "results",
        eval_config=tmp_path / "eval.yaml",
        size="e2b",
        quant="bf16",
        runner=runner,
        env={},
    )
    assert plan["stages"]["B"] == "done"
    assert calls[0] == "O"
    assert len(calls) == 7
    entries = [entry for entry in plan["configs"] if entry["stage"] == "B"]
    assert all(entry["status"] == "done" and entry["result"] for entry in entries)
    assert len({entry["result"] for entry in entries}) == 14

    calls.clear()
    campaign.run_eval_stage(
        plan,
        "B",
        results_root=tmp_path / "results",
        eval_config=tmp_path / "eval.yaml",
        size="e2b",
        quant="bf16",
        runner=runner,
        env={},
    )
    assert calls == []


def test_plan_only_chooses_and_records_b2_block(tmp_path, capsys):
    stage_a = tmp_path / "results" / "stageA"
    stage_a.mkdir(parents=True)
    (stage_a / "n2.json").write_text(
        json.dumps(
            {
                "variants": [
                    {"block": [25, 29], "bridge": "none", "n_loops": 2, "nll_n_calibrated": 2.0},
                    {"block": [16, 18], "bridge": "none", "n_loops": 2, "nll_n_calibrated": 1.5},
                    {"block": [15, 29], "bridge": "damped", "n_loops": 2, "nll_n_calibrated": 2.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    code = campaign.main(
        [
            "--plan-only",
            "--stage",
            "B2",
            "--plan",
            str(plan_path),
            "--stage-a-dir",
            str(stage_a),
            "--results-root",
            str(tmp_path / "results"),
            "--eval-config",
            str(tmp_path / "eval.yaml"),
            "--heartbeat",
            str(tmp_path / "hb.json"),
        ]
    )
    assert code == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["b2_block"] == [16, 18]
    assert len([entry for entry in plan["configs"] if entry["stage"] == "B2"]) == 8
    assert "b2 block: 16-18" in capsys.readouterr().out


def test_campaign_refuses_when_lock_is_held(tmp_path, monkeypatch):
    lock = tmp_path / "busy.lock"
    held = acquire(lock)
    try:
        monkeypatch.setenv("GEMMALOOP_GPU_LOCK_PATH", str(lock))
        with pytest.raises(SystemExit) as excinfo:
            campaign.main(
                [
                    "--stage",
                    "A",
                    "--plan",
                    str(tmp_path / "plan.json"),
                    "--corpus",
                    str(tmp_path / "corpus.jsonl"),
                    "--results-root",
                    str(tmp_path / "results"),
                    "--stage-a-dir",
                    str(tmp_path / "results" / "stageA"),
                    "--eval-config",
                    str(tmp_path / "eval.yaml"),
                    "--heartbeat",
                    str(tmp_path / "hb.json"),
                    "--prereg",
                    str(tmp_path / "prereg.yaml"),
                ]
            )
        assert excinfo.value.code == 75
    finally:
        held.release()


def test_stage_b_is_blocked_until_allowed(tmp_path):
    common = [
        "--plan",
        str(tmp_path / "plan.json"),
        "--corpus",
        str(tmp_path / "corpus.jsonl"),
        "--results-root",
        str(tmp_path / "results"),
        "--stage-a-dir",
        str(tmp_path / "results" / "stageA"),
        "--eval-config",
        str(tmp_path / "eval.yaml"),
        "--heartbeat",
        str(tmp_path / "hb.json"),
    ]
    with pytest.raises(SystemExit, match="BLOCKED"):
        campaign.main(["--stage", "B", *common])
    with pytest.raises(SystemExit, match="BLOCKED"):
        campaign.main(["--stage", "all", *common])
    with pytest.raises(SystemExit, match="BLOCKED"):
        campaign.main(["--stage", "B2", *common])
