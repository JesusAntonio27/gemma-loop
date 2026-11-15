"""A resumed CPU run must equal an uninterrupted run bit-for-bit (Task 14b)."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import TrainerCallback

from gemmaloop.train.checkpoint import COMPLETE_FILE
from gemmaloop.train.trainer import TokenBudgetCallback
from tests.train.test_trainer_tiny import _losses, _make_trainer


def _parameters(trainer) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }


def test_crash_and_resume_are_bit_exact(tmp_path):
    uninterrupted = _make_trainer(
        tmp_path / "full", seed=0, loops=True, max_steps=6
    )
    uninterrupted.train()
    expected_losses = _losses(uninterrupted)
    expected_params = _parameters(uninterrupted)

    first = _make_trainer(
        tmp_path / "first",
        seed=0,
        loops=True,
        max_steps=6,
        save_steps=3,
        callbacks=[TokenBudgetCallback(36)],
    )
    first.train()
    checkpoint = tmp_path / "first" / "checkpoint-3"
    assert (checkpoint / COMPLETE_FILE).is_file()
    marker = json.loads((checkpoint / COMPLETE_FILE).read_text(encoding="utf-8"))
    assert marker["data_order"] == {"epoch": 0, "row_position": 3}
    assert marker["rng_states_saved"] is True
    assert marker["has_optimizer"] is True
    assert marker["has_scheduler"] is True

    resumed = _make_trainer(
        tmp_path / "resumed", seed=0, loops=True, max_steps=6
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert _losses(resumed) == expected_losses
    actual_params = _parameters(resumed)
    assert set(actual_params) == set(expected_params)
    for name, expected in expected_params.items():
        assert torch.equal(actual_params[name], expected), name

    assert resumed.state.global_step == 6
    assert resumed.micro_step == 6
    assert resumed.tokens_seen == uninterrupted.tokens_seen
    assert resumed.optimizer.state_dict()["param_groups"] == (
        uninterrupted.optimizer.state_dict()["param_groups"]
    )


def test_newer_incomplete_checkpoint_is_ignored_on_auto_resume(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    first = _make_trainer(
        out,
        seed=0,
        loops=True,
        max_steps=6,
        save_steps=3,
        callbacks=[TokenBudgetCallback(36)],
    )
    first.set_checkpoint_context(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        run_identity_hash="identity",
    )
    first.train()
    stale = out / "checkpoint-5.tmp"
    stale.mkdir()
    (stale / "adapter_model.safetensors").write_text("partial", encoding="utf-8")

    metadata = json.loads(
        (out / "checkpoint-3" / COMPLETE_FILE).read_text(encoding="utf-8")
    )
    path = train_script.resolve_resume_checkpoint(
        "auto",
        out,
        run_config_sha256=metadata["run_config_sha256"],
        data_manifest_sha256=metadata["data_manifest_sha256"],
        git_sha=metadata["git_sha"],
        prereg_sha=metadata["prereg_sha"],
        identity_hash=metadata["run_identity_hash"],
    )
    assert path == out / "checkpoint-3"

    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            "auto",
            out,
            run_config_sha256=metadata["run_config_sha256"],
            data_manifest_sha256="different",
            git_sha=metadata["git_sha"],
            prereg_sha=metadata["prereg_sha"],
            identity_hash=metadata["run_identity_hash"],
        )
    assert excinfo.value.code == 2


def test_auto_resume_without_a_checkpoint_starts_fresh(tmp_path):
    import scripts.train as train_script

    path = train_script.resolve_resume_checkpoint(
        "auto",
        tmp_path / "empty",
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        identity_hash="identity",
    )
    assert path is None


def test_explicit_missing_checkpoint_is_exit_2(tmp_path):
    import scripts.train as train_script

    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            str(tmp_path / "checkpoint-99"),
            tmp_path,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha=None,
            identity_hash="identity",
        )
    assert excinfo.value.code == 2


def test_checkpoint_without_optimizer_state_is_refused_with_exit_2(tmp_path):
    import scripts.train as train_script
    from gemmaloop.train.checkpoint import finalize_checkpoint

    staging = tmp_path / "checkpoint-1.tmp"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    finalize_checkpoint(
        staging,
        tmp_path / "checkpoint-1",
        {
            "step": 1,
            "has_optimizer": False,
            "has_scheduler": True,
            "rng_states_saved": True,
            "loop_sampler": {"seed": 0},
            "data_order": {"epoch": 0, "row_position": 1},
            "run_config_sha256": "cfg",
            "data_manifest_sha256": "data",
            "git_sha": "abc",
            "prereg_sha": None,
            "run_identity_hash": "identity",
        },
    )
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            str(tmp_path / "checkpoint-1"),
            tmp_path,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha=None,
            identity_hash="identity",
        )
    assert excinfo.value.code == 2


class _StopAfter(TrainerCallback):
    """Stop right after an optimizer step but request a checkpoint."""

    def __init__(self, step: int) -> None:
        self.step = int(step)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if int(state.global_step) >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def _distinct_rows(count: int = 8) -> list[dict[str, list[int]]]:
    rows = []
    for index in range(count):
        length = 5 + index % 4
        ids = [2] + [10 + ((index * 7 + position * 3) % 200) for position in range(length)] + [1]
        rows.append({"input_ids": ids, "labels": list(ids)})
    return rows


def _kwargs(rows) -> dict:
    return dict(
        seed=0,
        loops=True,
        ple_mode="loop_vector",
        max_steps=5,
        rows=rows,
        train_sampling_strategy="random",
        lora_dropout=0.25,
    )


def _optimizer_state(trainer) -> tuple[dict, dict]:
    state = trainer.optimizer.state_dict()
    return state["param_groups"], {
        int(key): {name: value.detach().clone() for name, value in values.items()}
        for key, values in state["state"].items()
    }


def test_shuffled_distinct_rows_dropout_and_loop_ple_resume_exactly(tmp_path):
    rows = _distinct_rows()
    kwargs = _kwargs(rows)
    full = _make_trainer(tmp_path / "full", **kwargs)
    full.train()
    expected_params = _parameters(full)
    expected_groups, expected_optimizer = _optimizer_state(full)

    first = _make_trainer(
        tmp_path / "first", save_steps=500, callbacks=[_StopAfter(2)], **kwargs
    )
    first.train()
    checkpoint = tmp_path / "first" / "checkpoint-2"
    assert (checkpoint / COMPLETE_FILE).is_file()

    resumed = _make_trainer(tmp_path / "resumed", **kwargs)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    for name, expected in expected_params.items():
        assert torch.equal(_parameters(resumed)[name], expected), name
    groups, optimizer_state = _optimizer_state(resumed)
    assert groups == expected_groups
    assert set(optimizer_state) == set(expected_optimizer)
    for key, values in expected_optimizer.items():
        assert set(optimizer_state[key]) == set(values)
        for name, value in values.items():
            assert torch.equal(optimizer_state[key][name], value), (key, name)
    # The loop_ple parameter and its optimizer group were exercised.
    assert hasattr(resumed.loop_text_model, "loop_ple")
    assert any(
        group.get("name") == "loop_ple" for group in resumed.optimizer.param_groups
    )


@pytest.mark.parametrize("broken", ["rng", "data_position", "optimizer_state"])
def test_broken_restore_diverges(tmp_path, monkeypatch, broken):
    """A deliberately broken restore must NOT reproduce the uninterrupted run."""
    rows = _distinct_rows()
    kwargs = _kwargs(rows)
    full = _make_trainer(tmp_path / "full", **kwargs)
    full.train()
    expected = _parameters(full)
    expected_losses = _losses(full)

    first = _make_trainer(
        tmp_path / "first", save_steps=500, callbacks=[_StopAfter(2)], **kwargs
    )
    first.train()
    checkpoint = tmp_path / "first" / "checkpoint-2"

    resumed = _make_trainer(tmp_path / "resumed", **kwargs)
    if broken == "rng":
        monkeypatch.setattr(torch.random, "set_rng_state", lambda *a, **k: None)
    elif broken == "data_position":
        original = resumed._restore_resume_state

        def broken_restore(path):
            original(path)
            resumed._sampler_initial_position = (
                resumed._sampler_initial_position + 1
            ) % len(rows)

        monkeypatch.setattr(resumed, "_restore_resume_state", broken_restore)
    else:
        monkeypatch.setattr(
            resumed, "_load_optimizer_and_scheduler", lambda *a, **k: None
        )
    resumed.train(resume_from_checkpoint=str(checkpoint))

    actual = _parameters(resumed)
    parameters_differ = any(
        not torch.equal(actual[name], value) for name, value in expected.items()
    )
    losses_differ = _losses(resumed) != expected_losses
    assert parameters_differ or losses_differ, f"broken {broken} restore looked exact"


@pytest.mark.parametrize("bound", ["max_steps", "token_budget"])
def test_resume_at_the_training_bound_does_not_take_an_extra_step(tmp_path, bound):
    kwargs = dict(seed=0, loops=True, max_steps=4)
    first = _make_trainer(
        tmp_path / "first",
        save_steps=500,
        callbacks=[_StopAfter(2)],
        **kwargs,
    )
    first.train()
    checkpoint = tmp_path / "first" / "checkpoint-2"
    checkpoint_params = _parameters(first)

    if bound == "max_steps":
        resumed_kwargs = dict(kwargs, max_steps=2)
    else:
        resumed_kwargs = dict(kwargs, token_budget=1)
    resumed = _make_trainer(tmp_path / "resumed", **resumed_kwargs)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 2
    assert resumed.micro_step == 2
    # No new optimizer step was taken; the restored log history is unchanged.
    assert _losses(resumed) == _losses(first)
    for name, expected in checkpoint_params.items():
        assert torch.equal(_parameters(resumed)[name], expected), name


def test_resume_with_allow_repeat_is_refused_with_exit_2():
    import scripts.train as train_script

    with pytest.raises(SystemExit) as excinfo:
        train_script.main(
            [
                "--arm", "L", "--size", "e2b", "--data", "d",
                "--tokens", "10", "--seed", "0", "--out", "o",
                "--allow-repeat", "--resume", "auto",
            ]
        )
    assert excinfo.value.code == 2


def test_fresh_start_over_existing_checkpoints_is_refused(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    first = _make_trainer(tmp_path / "first", seed=0, loops=True, max_steps=2,
                          save_steps=1)
    first.train()
    # Move the trainer output to the run directory the CLI will see.
    import shutil

    shutil.copytree(tmp_path / "first", out)
    with pytest.raises(SystemExit) as excinfo:
        train_script.main(
            [
                "--arm", "L", "--size", "e2b", "--data", "d",
                "--tokens", "10", "--seed", "0", "--out", str(out),
            ]
        )
    assert excinfo.value.code == 2


def test_guard_at_final_budget_step_is_treated_as_complete():
    import scripts.train as train_script
    from types import SimpleNamespace

    at_budget = SimpleNamespace(
        token_budget=100,
        tokens_seen=100,
        state=SimpleNamespace(global_step=3),
    )
    at_bound = SimpleNamespace(
        token_budget=100,
        tokens_seen=80,
        state=SimpleNamespace(global_step=5),
    )
    midway = SimpleNamespace(
        token_budget=100,
        tokens_seen=80,
        state=SimpleNamespace(global_step=4),
    )
    assert train_script.guard_is_at_bound(at_budget, 50)
    assert train_script.guard_is_at_bound(at_bound, 5)
    assert not train_script.guard_is_at_bound(midway, 5)


def test_auto_with_only_invalid_checkpoints_is_exit_2(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    (out / "checkpoint-3").mkdir(parents=True)
    (out / "checkpoint-3" / "adapter_model.safetensors").write_text(
        "torn", encoding="utf-8"
    )
    (out / "run_identity.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            "auto",
            out,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha=None,
            identity_hash="identity",
        )
    assert excinfo.value.code == 2


def test_auto_only_accepts_matching_identity(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    first = _make_trainer(
        out, seed=0, loops=True, max_steps=2, save_steps=1,
    )
    first.set_checkpoint_context(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        run_identity_hash="old-identity",
    )
    first.train()
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            "auto",
            out,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha=None,
            identity_hash="new-identity",
        )
    assert excinfo.value.code == 2


def test_prereg_sha_mismatch_is_exit_2(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    first = _make_trainer(
        out, seed=0, loops=True, max_steps=2, save_steps=1,
    )
    first.set_checkpoint_context(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha="old-prereg",
        run_identity_hash="identity",
    )
    first.train()
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            "auto",
            out,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha="new-prereg",
            identity_hash="identity",
        )
    assert excinfo.value.code == 2


def test_done_marker_short_circuits_and_recovers_from_run_json(tmp_path):
    import json
    import scripts.train as train_script
    from gemmaloop.train.checkpoint import DONE_FILE

    out = tmp_path / "run"
    out.mkdir()
    assert not train_script.check_already_complete(out, "identity")
    (out / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_identity_hash": "identity",
                "run_config_sha256": "cfg",
                "data_manifest_sha256": "data",
                "git_sha": "sha",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    assert train_script.check_already_complete(out, "identity")
    assert (out / DONE_FILE).is_file()
    assert train_script.check_already_complete(out, "identity")

    (out / DONE_FILE).write_text(
        json.dumps({"status": "complete", "run_identity_hash": "other"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        train_script.check_already_complete(out, "identity")
    assert excinfo.value.code == 2


def test_identity_only_state_starts_fresh_for_the_same_identity(tmp_path):
    import json
    import scripts.train as train_script

    out = tmp_path / "run"
    out.mkdir()
    (out / "run_identity.json").write_text(
        json.dumps({"run_identity_hash": "identity"}), encoding="utf-8"
    )
    (out / "status.json").write_text(
        json.dumps({"step": 3, "tokens_seen": 30}), encoding="utf-8"
    )
    path = train_script.resolve_resume_checkpoint(
        "auto",
        out,
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        identity_hash="identity",
    )
    assert path is None


def test_identity_only_state_from_another_run_is_exit_2(tmp_path):
    import json
    import scripts.train as train_script

    out = tmp_path / "run"
    out.mkdir()
    (out / "run_identity.json").write_text(
        json.dumps({"run_identity_hash": "other"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_resume_checkpoint(
            "auto",
            out,
            run_config_sha256="cfg",
            data_manifest_sha256="data",
            git_sha="abc",
            prereg_sha=None,
            identity_hash="identity",
        )
    assert excinfo.value.code == 2


def test_allow_mismatch_works_with_auto(tmp_path):
    import scripts.train as train_script

    out = tmp_path / "run"
    first = _make_trainer(out, seed=0, loops=True, max_steps=2, save_steps=1)
    first.set_checkpoint_context(
        run_config_sha256="old-cfg",
        data_manifest_sha256="old-data",
        git_sha="old-sha",
        prereg_sha=None,
        run_identity_hash="old-identity",
    )
    first.train()
    path = train_script.resolve_resume_checkpoint(
        "auto",
        out,
        run_config_sha256="new-cfg",
        data_manifest_sha256="new-data",
        git_sha="new-sha",
        prereg_sha=None,
        identity_hash="new-identity",
        allow_mismatch=True,
    )
    assert path is not None and path.name.startswith("checkpoint-")


def test_exception_never_writes_a_mid_step_checkpoint_and_resume_is_exact(
    tmp_path, monkeypatch
):
    """N1: an OOM/CUDA error mid-step must not save the torn counters."""
    rows = _distinct_rows()
    kwargs = dict(
        seed=0,
        loops=True,
        ple_mode="loop_vector",
        max_steps=4,
        rows=rows,
        train_sampling_strategy="random",
        lora_dropout=0.25,
        gradient_accumulation_steps=2,
        save_steps=1,
    )
    endless = _make_trainer(tmp_path / "full", **kwargs)
    endless.train()
    expected_params = _parameters(endless)
    expected_tokens = endless.tokens_seen
    expected_micro = endless.micro_batches_seen

    crashed = _make_trainer(tmp_path / "crashed", **kwargs)
    original_compute_loss = crashed.compute_loss
    calls = {"count": 0}

    def oom_after_one_step(model, inputs, *args, **kwargs2):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 3:  # first micro-batch of step 2
            raise torch.OutOfMemoryError("CUDA out of memory mid-step")
        return original_compute_loss(model, inputs, *args, **kwargs2)

    monkeypatch.setattr(crashed, "compute_loss", oom_after_one_step)
    with pytest.raises(torch.OutOfMemoryError):
        crashed.train()

    published = sorted(
        path.name
        for path in (tmp_path / "crashed").glob("checkpoint-*")
        if path.is_dir() and (path / COMPLETE_FILE).is_file()
    )
    assert published == ["checkpoint-1"]

    resumed = _make_trainer(tmp_path / "resumed", **kwargs)
    resumed.train(resume_from_checkpoint=str(tmp_path / "crashed" / "checkpoint-1"))
    actual = _parameters(resumed)
    for name, value in expected_params.items():
        assert torch.equal(actual[name], value), name
    assert resumed.tokens_seen == expected_tokens
    assert resumed.micro_batches_seen == expected_micro
    assert resumed.micro_step == endless.micro_step
    assert _losses(resumed)[1:] == _losses(endless)[1:]


def test_auto_picks_the_matching_older_checkpoint_not_a_newer_foreign_one(tmp_path):
    import shutil
    import scripts.train as train_script

    out = tmp_path / "run"
    matching = _make_trainer(out, seed=0, loops=True, max_steps=1, save_steps=1)
    matching.set_checkpoint_context(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        run_identity_hash="identity",
    )
    matching.train()
    assert (out / "checkpoint-1").is_dir()

    foreign_dir = tmp_path / "foreign"
    foreign = _make_trainer(
        foreign_dir, seed=0, loops=True, max_steps=2, save_steps=1
    )
    foreign.set_checkpoint_context(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        run_identity_hash="foreign",
    )
    foreign.train()
    shutil.copytree(foreign_dir / "checkpoint-2", out / "checkpoint-2")

    path = train_script.resolve_resume_checkpoint(
        "auto",
        out,
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="abc",
        prereg_sha=None,
        identity_hash="identity",
    )
    assert path == out / "checkpoint-1"


def _dial_rows(count: int = 8) -> list[dict]:
    from tests.train.test_trainer_tiny import _pattern

    pattern = _pattern()
    rows = []
    for _ in range(count):
        length = len(pattern)
        rows.append(
            {
                "input_ids": list(pattern),
                "labels": list(pattern),
                "teacher_topk_probs": [[0.5, 0.4]] * length,
                "teacher_topk_indices": [[10, 11]] * length,
                "teacher_tail_prob": [0.1] * length,
            }
        )
    return rows


def test_dial_crash_and_resume_are_bit_exact(tmp_path):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from gemmaloop.train.teacher_collator import TeacherDataCollator

    rows = _dial_rows()
    collator = TeacherDataCollator(
        DataCollatorForLanguageModeling(pad_token_id=0)
    )
    horizon = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    kwargs = dict(
        seed=0,
        loops=True,
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
        rows=rows,
        data_collator=collator,
        dial_enabled=True,
        dial_horizon_probs=horizon,
        max_steps=6,
    )
    full = _make_trainer(tmp_path / "dial-full", **kwargs)
    full.train()
    expected_losses = _losses(full)
    expected_params = _parameters(full)

    first = _make_trainer(
        tmp_path / "dial-first",
        save_steps=500,
        callbacks=[_StopAfter(3)],
        **kwargs,
    )
    first.train()
    checkpoint = tmp_path / "dial-first" / "checkpoint-3"
    assert (checkpoint / COMPLETE_FILE).is_file()

    resumed = _make_trainer(tmp_path / "dial-resumed", **kwargs)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert _losses(resumed) == expected_losses
    actual = _parameters(resumed)
    for name, expected in expected_params.items():
        assert torch.equal(actual[name], expected), name
    assert resumed.micro_step == full.micro_step
    assert resumed.tokens_seen == full.tokens_seen
    assert resumed.last_n_loops == full.last_n_loops
