"""Teacher refusal and dataset-join tests for the Stage 1 heal loader."""

from __future__ import annotations

import json

import pytest
import torch

from gemmaloop.teacher.runtime import runtime_identity

import scripts.train as train_script
from gemmaloop.data.teacher_topk import (
    DEFAULT_MASK_TOKEN,
    ShardEntry,
    TeacherManifest,
    TeacherRow,
    write_teacher_shard,
)


def _teacher_dir(tmp_path, *, data_sha="d" * 64):
    row = TeacherRow(
        row_id=0,
        positions=torch.tensor([1], dtype=torch.int32),
        indices=torch.arange(64, dtype=torch.int32).unsqueeze(0),
        probs=torch.full((1, 64), 0.5 / 64, dtype=torch.float16),
        tail=torch.tensor([0.5], dtype=torch.float32),
    )
    path = tmp_path / "shard-00000.pt"
    digest = write_teacher_shard(path, [row])
    manifest = TeacherManifest(
        teacher={"key": "e4b", "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", "runtime": "bf16", "max_length": 8},
        runtime_identity=runtime_identity("e4b", "bf16"),
        data_manifest_sha256=data_sha,
        top_k=64,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=249788,
        rows_total=1,
        tokens_total=1,
        chat_template_sha256="template",
        tokenizer_sha256="student",
        render_template_sha256="template",
        student_vocab_sha256="student",
        teacher_vocab_sha256="teacher",
        teacher_vocab_ex_mask_sha256="student",
        lm_head_rows=262144,
        shards=(
            ShardEntry(
                index=0,
                file=path.name,
                sha256=digest,
                row_ids=(0,),
                tokens=1,
            ),
        ),
    )
    manifest.write(tmp_path / "teacher_manifest.json")
    return tmp_path


def test_dial_without_teacher_refuses_with_exit_2():
    with pytest.raises(SystemExit) as excinfo:
        train_script.ensure_teacher_available(
            needs_teacher=True,
            teacher_available=False,
            allow_no_teacher=False,
        )
    assert excinfo.value.code == 2


def test_allow_no_teacher_keeps_the_smoke_fallback():
    train_script.ensure_teacher_available(
        needs_teacher=True,
        teacher_available=False,
        allow_no_teacher=True,
    )


def test_teacher_manifest_must_match_the_data_manifest(tmp_path):
    teacher = _teacher_dir(tmp_path, data_sha="e" * 64)
    with pytest.raises(SystemExit) as excinfo:
        train_script.resolve_teacher_index("d" * 64, teacher)
    assert excinfo.value.code == 2


def test_matching_teacher_data_is_loaded(tmp_path):
    teacher = _teacher_dir(tmp_path)
    index = train_script.resolve_teacher_index("d" * 64, teacher)
    assert index is not None
    assert index.manifest.rows_total == 1
    assert index.manifest_sha256


def test_build_dataset_keeps_the_source_row_id(monkeypatch):
    def fake_labels(tokenizer, messages, max_length, tools=None):
        del tokenizer, max_length, tools
        if messages[0]["content"] == "empty":
            return {"input_ids": [], "labels": []}
        return {
            "input_ids": [1, 2, 3],
            "labels": [-100, int(messages[0]["content"] == "train"), -100],
        }

    monkeypatch.setattr(train_script, "assistant_labels", fake_labels)
    rows = [
        {"messages": [{"content": "empty"}]},
        {"messages": [{"content": "train"}]},
        {"messages": [{"content": "other"}]},
    ]
    dataset = train_script.build_dataset(None, rows, 8, include_row_id=True)
    assert list(dataset["row_id"]) == [1, 2]
    assert len(dataset) == 2


def _registered_prereg() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2] / "prereg" / "prereg.yaml")


def _registered_base() -> list[str]:
    return [
        "--size", "e2b", "--data", "unused", "--tokens", "1",
        "--seed", "0", "--out", "unused", "--prereg", _registered_prereg(),
        "--adapter-policy", "loop_only", "--lora-target-layers", "loop_block",
    ]


def test_prereg_binds_the_dial_and_heal_objectives():
    from gemmaloop.prereg import load_prereg

    prereg = load_prereg(_registered_prereg())
    base = ["--arm", "L", "--size", "e2b", "--data", "unused", "--tokens", "1",
            "--seed", "0", "--out", "unused"]
    dial = train_script.resolve_objective(train_script._parse_args(base), prereg)
    assert dial.dial is True and dial.heal is False
    assert dial.window == prereg.dial_objective.window_trainable_passes
    assert dial.beta == prereg.stage1.teacher.beta_parent_kl
    assert dial.horizon == tuple(prereg.dial_objective.horizon)

    base[1] = "O-H"
    heal = train_script.resolve_objective(train_script._parse_args(base), prereg)
    assert heal.dial is False and heal.heal is True
    assert heal.beta == prereg.stage1.teacher.beta_parent_kl

    base[1] = "O"
    control = train_script.resolve_objective(train_script._parse_args(base), prereg)
    assert control.dial is False and control.heal is False
    assert control.needs_teacher is False


def test_prereg_refuses_cli_overrides_and_the_one_hot_opt_in(tmp_path):
    for extra in (
        ["--dial-beta", "0.2"],
        ["--dial-lambda", "2.0"],
        ["--dial-window", "4"],
        ["--loop-probs", "0,0,0,1"],
        ["--allow-no-teacher"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            train_script.main(_registered_base() + ["--arm", "L"] + extra)
        assert excinfo.value.code == 2


def test_prereg_stage1_without_teacher_data_exits_2(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": "deadbeef",
                "target_tokens": 10,
                "max_length": 8192,
                "chat_template_sha256": "",
            }
        ),
        encoding="utf-8",
    )
    base = [
        "--size", "e2b", "--data", str(data), "--tokens", "1",
        "--seed", "0", "--out", str(tmp_path / "run"),
        "--prereg", _registered_prereg(),
        "--max-length", "8192",
        "--adapter-policy", "loop_only", "--lora-target-layers", "loop_block",
    ]
    for arm in ("L", "O-H"):
        with pytest.raises(SystemExit) as excinfo:
            train_script.main(base + ["--arm", arm])
        assert excinfo.value.code == 2


def test_phase_plan_passes_the_registered_objective_and_teacher(tmp_path):
    from scripts.phase1_plan import registered_objective_args

    teacher = tmp_path / "teacher"
    assert registered_objective_args(None, "L", teacher) == [
        "--dial-objective", "--teacher-data", str(teacher)
    ]
    assert registered_objective_args(None, "O-H", teacher) == [
        "--heal-teacher", "--teacher-data", str(teacher)
    ]
    assert registered_objective_args(None, "O", teacher) == []
    with pytest.raises(ValueError, match="teacher-data"):
        registered_objective_args(None, "L", None)


def test_teacher_manifest_identity_is_checked():
    from dataclasses import replace

    from gemmaloop.data.teacher_topk import (
        DEFAULT_MASK_TOKEN,
        GLITCH_TOKEN_ID,
        validate_teacher_manifest_identity,
    )

    base = TeacherManifest(
        teacher={
            "key": "e4b",
            "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            "runtime": "bf16",
        },
        data_manifest_sha256="d" * 64,
        top_k=64,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=GLITCH_TOKEN_ID,
        rows_total=0,
        tokens_total=0,
        chat_template_sha256="template",
        tokenizer_sha256="student",
        render_template_sha256="template",
        student_vocab_sha256="student",
        teacher_vocab_sha256="teacher",
        teacher_vocab_ex_mask_sha256="student",
        lm_head_rows=262144,
        runtime_identity=runtime_identity("e4b", "bf16"),
    )
    validate_teacher_manifest_identity(base)
    validate_teacher_manifest_identity(
        base,
        render_template_sha256="template",
        student_vocab_sha256="student",
        student_vocab_ex_mask_sha256="student",
        teacher_vocab_sha256="teacher",
    )
    for manifest in (
        replace(base, teacher={**base.teacher, "revision": "wrong"}),
        replace(base, teacher={**base.teacher, "runtime": "nf4"}),
        replace(base, mask_token_id=1),
        replace(base, lm_head_rows=32000),
        replace(base, max_rows=8),
    ):
        with pytest.raises(Exception, match="teacher"):
            validate_teacher_manifest_identity(manifest)
    with pytest.raises(Exception, match="chat template"):
        validate_teacher_manifest_identity(
            base, render_template_sha256="different"
        )
    with pytest.raises(Exception, match="student tokenizer"):
        validate_teacher_manifest_identity(base, student_vocab_sha256="different")
    with pytest.raises(Exception, match="masked id"):
        validate_teacher_manifest_identity(
            base, student_vocab_ex_mask_sha256="different"
        )
    with pytest.raises(Exception, match="teacher tokenizer"):
        validate_teacher_manifest_identity(base, teacher_vocab_sha256="different")


def test_allow_no_teacher_requires_smoke():
    base = ["--arm", "L", "--size", "e2b", "--data", "unused", "--tokens", "1",
            "--seed", "0", "--out", "unused", "--allow-no-teacher"]
    with pytest.raises(SystemExit) as excinfo:
        train_script.main(base)
    assert excinfo.value.code == 2


def test_registered_stage1_never_uses_the_legacy_fallback(tmp_path):
    from gemmaloop.prereg import load_prereg

    registered = load_prereg(_registered_prereg())
    base = train_script._parse_args(
        ["--arm", "L", "--size", "e2b", "--data", "unused", "--tokens", "1",
         "--seed", "0", "--out", "unused"]
    )
    assert train_script.resolve_objective(base, registered).dial is True
    # A prereg without dial/stage1 keeps the legacy CLI objective.
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "primary_n: 2\nloop_probs: [0.25, 0.40, 0.20, 0.15]\n"
        "benches:\n  lcb: {limit: 10, samples: 2, seed: 0, role: confirmatory}\n"
        "google: {}\ntraining: {max_length: 32, length_mode: filter}\n"
        "seeds: {e2b_oh: [0], e2b_l: [0]}\n",
        encoding="utf-8",
    )
    legacy_prereg = load_prereg(legacy)
    assert train_script.resolve_objective(base, legacy_prereg).dial is False


def test_run_config_and_identity_include_objective_and_teacher():
    from gemmaloop.train.frozen import run_identity_hash

    args = train_script._parse_args(
        ["--arm", "L", "--size", "e2b", "--data", "unused", "--tokens", "1",
         "--seed", "0", "--out", "unused", "--dial-objective",
         "--teacher-data", "teacher-a"]
    )
    objective = train_script.resolve_objective(args, None)
    config_a = train_script.build_run_config(
        args,
        quant="bf16",
        max_length=32,
        objective=objective,
        teacher_manifest_sha256="teacher-a-sha",
    )
    assert config_a["dial_objective"] is True
    assert config_a["dial_window"] == objective.window
    assert config_a["dial_beta"] == objective.beta
    assert config_a["dial_lambda"] == objective.lambda_hinge
    assert config_a["dial_horizon_probs"] == list(objective.horizon)
    assert config_a["teacher_manifest_sha256"] == "teacher-a-sha"

    config_b = train_script.build_run_config(
        args,
        quant="bf16",
        max_length=32,
        objective=objective,
        teacher_manifest_sha256="teacher-b-sha",
    )
    sha_a = train_script.run_config_sha256(config_a)
    sha_b = train_script.run_config_sha256(config_b)
    assert sha_a != sha_b
    identity_a = run_identity_hash(
        run_config_sha256=sha_a,
        data_manifest_sha256="data",
        git_sha="git",
        seed=0,
    )
    identity_b = run_identity_hash(
        run_config_sha256=sha_b,
        data_manifest_sha256="data",
        git_sha="git",
        seed=0,
    )
    assert identity_a != identity_b


class _VocabTokenizer:
    def __init__(self, vocab, template="template"):
        self._vocab = dict(vocab)
        self.chat_template = template

    def get_vocab(self):
        return dict(self._vocab)


def _identity_manifest(student, teacher, template="template"):
    from gemmaloop.data.teacher_topk import (
        DEFAULT_MASK_TOKEN,
        GLITCH_TOKEN_ID,
        TeacherManifest,
        tokenizer_vocab_sha256,
        vocab_ex_mask_sha256,
    )

    return TeacherManifest(
        teacher={
            "key": "12b_qat",
            "revision": "b6ed86275a6a5735884e208bfed95b445a684ca2",
            "runtime": "nf4",
        },
        data_manifest_sha256="d" * 64,
        top_k=64,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=GLITCH_TOKEN_ID,
        rows_total=0,
        tokens_total=0,
        render_template_sha256=template,
        student_vocab_sha256=tokenizer_vocab_sha256(student),
        teacher_vocab_sha256=tokenizer_vocab_sha256(teacher),
        teacher_vocab_ex_mask_sha256=vocab_ex_mask_sha256(teacher),
        lm_head_rows=262144,
        runtime_identity=runtime_identity("12b_qat", "nf4"),
    )


def test_valid_12b_style_teacher_manifest_is_accepted():
    from gemmaloop.data.teacher_topk import (
        DEFAULT_MASK_TOKEN,
        tokenizer_vocab_sha256,
        validate_teacher_manifest_identity,
        vocab_ex_mask_sha256,
    )

    student_vocab = {"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 249788, "c": 3}
    teacher_vocab = {
        "a": 0, "b": 1, "<different_placeholder>": 249788, "c": 3
    }
    student = _VocabTokenizer(student_vocab)
    teacher = _VocabTokenizer(teacher_vocab)
    manifest = _identity_manifest(student, teacher)
    validate_teacher_manifest_identity(
        manifest,
        render_template_sha256="template",
        student_vocab_sha256=tokenizer_vocab_sha256(student),
        student_vocab_ex_mask_sha256=vocab_ex_mask_sha256(student),
        teacher_vocab_sha256=tokenizer_vocab_sha256(teacher),
    )


def test_teacher_vocab_differing_outside_the_mask_is_refused():
    from gemmaloop.data.teacher_topk import (
        DEFAULT_MASK_TOKEN,
        tokenizer_vocab_sha256,
        validate_teacher_manifest_identity,
        vocab_ex_mask_sha256,
    )

    student_vocab = {"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 249788, "c": 3}
    teacher_vocab = {
        "a": 0, "b": 1, "<different_placeholder>": 249788, "d": 3
    }
    student = _VocabTokenizer(student_vocab)
    teacher = _VocabTokenizer(teacher_vocab)
    manifest = _identity_manifest(student, teacher)
    with pytest.raises(Exception, match="masked id"):
        validate_teacher_manifest_identity(
            manifest,
            student_vocab_ex_mask_sha256=vocab_ex_mask_sha256(student),
        )


def test_empty_or_missing_identity_fields_are_refused():
    from gemmaloop.data.teacher_topk import (
        DEFAULT_MASK_TOKEN,
        validate_teacher_manifest_identity,
    )
    from dataclasses import replace

    student = _VocabTokenizer({"a": 0, DEFAULT_MASK_TOKEN: 1})
    manifest = _identity_manifest(student, student)
    for field in (
        "render_template_sha256",
        "student_vocab_sha256",
        "teacher_vocab_sha256",
        "teacher_vocab_ex_mask_sha256",
    ):
        with pytest.raises(Exception, match="identity fields"):
            validate_teacher_manifest_identity(replace(manifest, **{field: ""}))
    with pytest.raises(Exception, match="chat template"):
        validate_teacher_manifest_identity(
            manifest, render_template_sha256="other-template"
        )
