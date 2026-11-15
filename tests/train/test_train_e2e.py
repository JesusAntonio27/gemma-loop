"""CPU end-to-end run of ``scripts/train.py main()`` for the Phase-1 arms.

The harness uses a tiny randomly initialised ``Gemma4ForConditionalGeneration``
(the real class TRL sees) with the real cached E2B tokenizer / chat template, so
labels, the chat-template guard, the token schedule, the loop sampler and
``run.json`` are exercised together.  The processor is skipped when the pinned
snapshot is not in the local Hugging Face cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import TrainerCallback

import scripts.train as train_script
import gemmaloop.train.trainer as trainer_module
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import apply_loops
from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.snapshot import E2B_REVISION, MODEL_REVISION, chat_template_sha256
from gemmaloop.train.budget import flops_matched_tokens
from gemmaloop.train.loop_sampler import DEFAULT_LOOP_PROBS
from tests.model.tiny import tiny_config

MODEL_ID = "google/gemma-4-E2B-it"
ROWS = 96
MAX_LENGTH = 64


def _tiny_text_config(tokenizer):
    config = tiny_config()
    vocab = len(tokenizer)
    config.vocab_size = vocab
    config.vocab_size_per_layer_input = vocab
    config.max_position_embeddings = 128
    return config


@pytest.fixture(scope="module")
def processor():
    transformers = pytest.importorskip("transformers")
    try:
        loaded = transformers.AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cached E2B processor unavailable: {exc}")
    tokenizer = getattr(loaded, "tokenizer", loaded)
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip("cached E2B tokenizer has no chat template")
    return loaded


def _tiny_conditional(tokenizer):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    config = _tiny_text_config(tokenizer)
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=config, vision_config=None, audio_config=None)
    )


def _write_rows(path: Path) -> list[dict]:
    rows = []
    for index in range(ROWS):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"What is {index} + 3?"},
                    {"role": "assistant", "content": str(index + 3)},
                ]
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return rows


def _write_prereg(path: Path) -> None:
    path.write_text(
        f"""primary_n: 2
loop_probs: [0.25, 0.40, 0.20, 0.15]
benches:
  lcb: {{limit: 10, samples: 2, seed: 0, role: confirmatory}}
google: {{}}
training: {{max_length: {MAX_LENGTH}, length_mode: filter}}
seeds: {{e2b_oh: [0], e2b_l: [0]}}
""",
        encoding="utf-8",
    )


def _rendered_tokens(tokenizer, rows) -> int:
    from gemmaloop.train.labels import assistant_labels

    return sum(
        len(assistant_labels(tokenizer, row["messages"], MAX_LENGTH)["input_ids"])
        for row in rows
    )


def _run_arm(
    tmp_path: Path,
    processor,
    arm: str,
    extra_args: list[str] | None = None,
    expect_code: int | None = None,
    expect_exit: int | None = None,
    out_name: str | None = None,
) -> tuple[dict, object]:
    tokenizer = getattr(processor, "tokenizer", processor)
    data = tmp_path / "data"
    rows = _write_rows(data / "train.jsonl")
    rendered = _rendered_tokens(tokenizer, rows)
    tiny = _tiny_conditional(tokenizer)
    text_config = tiny.config.text_config.to_dict()
    # Use a large probe: flops_matched_tokens rounds to an integer, so a
    # 1-token probe loses the ratio (1.42 -> 1).
    ratio = flops_matched_tokens(
        1_000_000,
        per_layer_flops(text_config),
        make_layer_plan(text_config),
        DEFAULT_LOOP_PROBS,
    ) / 1_000_000
    target = int(rendered * 0.8)
    base_tokens = max(1, int(target / ratio)) if arm == "O-H-ext" else target
    template_sha = chat_template_sha256(tokenizer)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": "deadbeef",
                "target_tokens": rendered,
                "max_length": MAX_LENGTH,
                "chat_template_sha256": template_sha,
            }
        ),
        encoding="utf-8",
    )
    prereg = tmp_path / "prereg.yaml"
    _write_prereg(prereg)

    def fake_load_model(spec):  # noqa: ANN001 - mirrors load_model
        model = tiny
        plan = make_layer_plan(model.config.text_config.to_dict())
        if spec.loop is not None:
            apply_loops(model.model.language_model, plan, spec.loop)
        model.eval()
        return model, processor, plan

    captured = []

    class CapturingTrainer(train_script.LoopSFTTrainer):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            captured.append(self)
            self.lr_after_steps: list[float] = []
            record = self

            class _LRRecorder(TrainerCallback):
                def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
                    record.lr_after_steps.append(
                        max(record.lr_scheduler.get_last_lr())
                    )
                    return control

            self.add_callback(_LRRecorder())

    loop_calls: list[int] = []
    real_set_n_loops = trainer_module.set_n_loops

    def counting_set_n_loops(text_model, n):  # noqa: ANN001
        loop_calls.append(int(n))
        return real_set_n_loops(text_model, n)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(train_script, "load_model", fake_load_model)
        monkey.setattr(train_script, "LoopSFTTrainer", CapturingTrainer)
        # The registered smoke combination is refused by M7; this CPU e2e
        # suite exercises the training path, so bypass the tag/dirty gate and
        # the monitor install explicitly (never via --smoke).
        monkey.setattr(train_script, "check_frozen_run", lambda *a, **k: None)
        monkey.setattr(train_script, "install_heal_monitor", lambda *a, **k: None)
        monkey.setattr(torch.cuda, "is_available", lambda: False)
        monkey.setattr(trainer_module, "set_n_loops", counting_set_n_loops)
        out = tmp_path / (out_name or f"run-{arm}")
        try:
            code = train_script.main(
                [
                "--arm",
                arm,
                "--size",
                "e2b",
                "--data",
                str(data),
                "--tokens",
                str(base_tokens),
                "--seed",
                "0",
                "--out",
                str(out),
                "--prereg",
                str(prereg),
                "--vram-cap",
                "0",
                "--no-host-performance-cores",
                "--allow-dirty",
                    *(extra_args or []),
                ]
            )
        except SystemExit as exc:
            if expect_exit is None:
                raise
            assert exc.code == expect_exit
            return {}, (captured[-1] if captured else None)
    finally:
        monkey.undo()

    if expect_code is not None:
        assert code == expect_code
        return {}, (captured[-1] if captured else None)
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert len(captured) == 1
    trainer = captured[0]
    assert count_loops(arm, loop_calls, trainer)
    return payload, trainer


def count_loops(arm: str, loop_calls: list[int], trainer) -> bool:  # noqa: ANN001
    if arm in {"L", "L-plevec"}:
        return (
            len(loop_calls) == trainer.micro_batches_seen
            and trainer.micro_batches_seen <= len(trainer.train_dataset)
        )
    return loop_calls == []


@pytest.mark.parametrize("arm", ["O-H", "L", "O-H-ext", "L-plevec"])
def test_train_main_end_to_end(tmp_path, processor, arm):
    payload, trainer = _run_arm(tmp_path, processor, arm)

    assert payload["status"] == "complete"
    assert payload["smoke"] is False
    assert payload["arm"] == arm
    assert payload["max_length"] == MAX_LENGTH
    assert payload["length_mode"] == "filter"
    assert payload["budget_tokens"] == payload["target_tokens"] > 1
    assert payload["budget_tokens"] == trainer.token_budget
    assert payload["tokens_consumed"] >= payload["budget_tokens"]
    supervised = payload["supervised_tokens_per_step"]
    assert supervised["count"] == payload["optimizer_steps"]
    assert supervised["mean"] > 0
    assert 0 < supervised["min"] <= supervised["max"]
    assert payload["micro_batches_seen"] == trainer.micro_batches_seen
    assert payload["micro_batches_seen"] <= len(trainer.train_dataset)
    assert payload["chat_template_sha256"] == payload["data_manifest_chat_template_sha256"]
    assert payload["chat_template_sha256"]
    assert payload["model_revision"] == E2B_REVISION == MODEL_REVISION
    # The run records the optimizer that actually ran (CPU here), not the
    # registered ``AdamW`` spelling, and no bitsandbytes version on CPU.
    assert payload["optimizer"] == "adamw_torch"
    assert payload["optimizer_requested"] in {"adamw", "adamw_8bit"}
    assert payload["bitsandbytes_version"] is None
    identity = json.loads(
        (tmp_path / f"run-{arm}" / "run_identity.json").read_text(encoding="utf-8")
    )
    assert identity["run_config"]["optimizer"] == "adamw_torch"
    assert identity["run_config"]["optimizer_requested"] in {"adamw", "adamw_8bit"}
    assert identity["run_config"]["bitsandbytes_version"] is None

    # A realistic budget takes more than one optimizer step, so the final LR
    # assertion is discriminating: mid-run LR was positive, the budget brings
    # it to exactly zero.
    assert trainer.state.global_step >= 2
    assert max(trainer.lr_after_steps) > 0
    final_lr = max(trainer.lr_scheduler.get_last_lr())
    assert final_lr == pytest.approx(0.0, abs=1e-12)
    assert trainer.lr_after_steps[-1] == pytest.approx(0.0, abs=1e-12)


def test_time_save_callback_is_wired(tmp_path, processor):
    """Dropping the time-save callback from train.py must fail this test."""
    _payload, _trainer = _run_arm(
        tmp_path,
        processor,
        "O-H",
        extra_args=["--save-every-minutes", "0.00001"],
    )
    out = tmp_path / "run-O-H"
    assert list(out.glob("checkpoint-*")), "time-save did not produce a checkpoint"


def test_retryable_checkpoint_publish_failure_exits_75(tmp_path, processor, monkeypatch):
    """A permanently held checkpoint directory must map to exit 75."""
    import gemmaloop.train.trainer as trainer_module
    from gemmaloop.train.checkpoint import RetryableIOError

    def always_fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RetryableIOError("checkpoint directory held open")

    monkeypatch.setattr(trainer_module, "finalize_checkpoint", always_fail)
    _run_arm(
        tmp_path,
        processor,
        "O-H",
        extra_args=["--save-every-minutes", "0.00001"],
        expect_code=75,
    )
    out = tmp_path / "run-O-H"
    assert not (out / "run.json").exists()


def test_cuda_oom_mid_run_exits_75(tmp_path, processor, monkeypatch):
    import torch

    def oom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise torch.OutOfMemoryError("CUDA out of memory at the allocator cap")

    monkeypatch.setattr(train_script.LoopSFTTrainer, "train", oom)
    _run_arm(tmp_path, processor, "O-H", expect_code=75)
    assert not (tmp_path / "run-O-H" / "run.json").exists()


def test_duplicate_launch_cannot_touch_a_live_run(tmp_path, processor, monkeypatch):
    from gemmaloop.gpulock import GpuLockBusy

    out = tmp_path / "run-O-H"
    out.mkdir()
    stale = out / "checkpoint-6.tmp"
    stale.mkdir()
    (stale / "partial").write_text("in flight", encoding="utf-8")

    def busy(_path):
        raise GpuLockBusy("per-run lock held")

    monkeypatch.setattr(train_script, "acquire", busy)
    _run_arm(tmp_path, processor, "O-H", extra_args=["--fresh"], expect_exit=75)
    assert stale.exists()
    assert (stale / "partial").read_text(encoding="utf-8") == "in flight"
    assert not (out / "attic").exists()


def test_fresh_main_archives_old_checkpoints(tmp_path, processor):
    out = tmp_path / "run-O-H"
    old = out / "checkpoint-9"
    old.mkdir(parents=True)
    (old / "partial.safetensors").write_text("old", encoding="utf-8")
    _run_arm(tmp_path, processor, "O-H", extra_args=["--fresh"])
    assert (out / "attic").is_dir()
    assert not old.exists()
    assert list((out / "attic").rglob("checkpoint-9"))


def test_rewind_main_archives_the_newer_branch(tmp_path, processor):
    import shutil
    from pathlib import Path

    _run_arm(
        tmp_path,
        processor,
        "O-H",
        extra_args=["--save-every-minutes", "0.00001"],
        out_name="base",
    )
    base = tmp_path / "base"
    checkpoints = sorted(
        (path for path in base.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )
    assert len(checkpoints) >= 2
    older, newer = checkpoints[-2], checkpoints[-1]

    rewind = tmp_path / "rewind"
    shutil.copytree(older, rewind / older.name)
    shutil.copytree(newer, rewind / newer.name)
    _run_arm(
        tmp_path,
        processor,
        "O-H",
        extra_args=["--resume", str(rewind / older.name)],
        out_name="rewind",
    )
    assert (rewind / "attic").is_dir()
    assert not (rewind / newer.name).exists()
    assert (rewind / "attic").rglob(newer.name)


def test_code_location_failure_is_exit_2(tmp_path, processor, monkeypatch):
    from gemmaloop.train.frozen import FrozenRunError

    def refuse(_repo):
        raise FrozenRunError("imported gemmaloop is outside the worktree")

    monkeypatch.setattr(train_script, "assert_code_under", refuse)
    _run_arm(tmp_path, processor, "O-H", expect_exit=2)
