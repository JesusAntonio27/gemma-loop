"""Guard trip -> exit 75 -> orchestrator resume to the same result (Task 14b).

This wires the real ``scripts/train.py main()`` callbacks (guard, status,
checkpoint writer) to the real ``scripts/orchestrate.py`` retry loop on the
tiny CPU model.  The child process is replaced by an in-process runner so the
suite never spawns a GPU job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from transformers import TrainerCallback

import gemmaloop.train.guards as guards_module
import gemmaloop.train.status as status_module
import scripts.orchestrate as orchestrate
import scripts.train as train_script
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import apply_loops
from gemmaloop.model.snapshot import MODEL_REVISION, chat_template_sha256
from tests.model.tiny import tiny_config

MODEL_ID = "google/gemma-4-E2B-it"
ROWS = 96
MAX_LENGTH = 64


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


def _write_data(tmp_path: Path, tokenizer) -> Path:
    data = tmp_path / "data"
    data.mkdir()
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
    (data / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    from gemmaloop.train.labels import assistant_labels

    rendered = sum(
        len(assistant_labels(tokenizer, row["messages"], MAX_LENGTH)["input_ids"])
        for row in rows
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": "guard-integration",
                "target_tokens": rendered,
                "max_length": MAX_LENGTH,
                "chat_template_sha256": chat_template_sha256(tokenizer),
            }
        ),
        encoding="utf-8",
    )
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(
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
    budget = max(20, int(rendered * 0.8))
    return data, prereg, budget


def _base_args(data: Path, prereg: Path, budget: int, out: Path) -> list[str]:
    return [
        "--arm", "O-H",
        "--size", "e2b",
        "--data", str(data),
        "--tokens", str(budget),
        "--seed", "0",
        "--out", str(out),
        "--prereg", str(prereg),
        "--vram-cap", "0",
        "--no-host-performance-cores",
        "--allow-dirty",
    ]


def _run(tmp_path, processor, monkeypatch):
    tokenizer = getattr(processor, "tokenizer", processor)
    data, prereg, budget = _write_data(tmp_path, tokenizer)
    out = tmp_path / "run"
    full_out = tmp_path / "full"
    args = _base_args(data, prereg, budget, out)
    full_args = _base_args(data, prereg, budget, full_out)

    def fake_load_model(spec):  # noqa: ANN001
        torch.manual_seed(0)
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration

        config = tiny_config()
        config.vocab_size = len(tokenizer)
        config.vocab_size_per_layer_input = len(tokenizer)
        config.max_position_embeddings = 128
        model = Gemma4ForConditionalGeneration(
            Gemma4Config(text_config=config, vision_config=None, audio_config=None)
        )
        plan = make_layer_plan(model.config.text_config.to_dict())
        if spec.loop is not None:
            apply_loops(model.model.language_model, plan, spec.loop)
        model.eval()
        return model, processor, plan

    trainers = []
    original_trainer = train_script.LoopSFTTrainer

    class CapturingTrainer(original_trainer):
        def __init__(self, *a, **k):  # noqa: ANN002, ANN003
            super().__init__(*a, **k)
            trainers.append(self)

    state = {"armed": True, "status_broken": True}
    real_ram = guards_module._read_ram_gib

    def fake_ram():
        return 0.5 if state["armed"] else 8.0

    real_status_write = status_module.atomic_write_json

    def maybe_broken_status(path, payload):
        if state["status_broken"]:
            raise RuntimeError("status file held open by antivirus")
        return real_status_write(path, payload)

    monkeypatch.setattr(train_script, "load_model", fake_load_model)
    monkeypatch.setattr(train_script, "LoopSFTTrainer", CapturingTrainer)
    # M7 refuses --smoke with --prereg; fake the frozen gate and the monitor
    # install explicitly instead of using the forbidden smoke combination.
    monkeypatch.setattr(train_script, "check_frozen_run", lambda *a, **k: None)
    monkeypatch.setattr(train_script, "install_heal_monitor", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(guards_module, "_read_ram_gib", fake_ram)
    monkeypatch.setattr(status_module, "atomic_write_json", maybe_broken_status)

    calls = {"count": 0}

    real_run = subprocess.run

    def fake_run(command, env=None, cwd=None, **kwargs):
        if not command or command[0] != "run-train":
            return real_run(command, env=env, cwd=cwd, **kwargs)
        calls["count"] += 1
        previous = os.environ.get("GEMMALOOP_RUN_LOCK_HELD")
        if env and env.get("GEMMALOOP_RUN_LOCK_HELD"):
            os.environ["GEMMALOOP_RUN_LOCK_HELD"] = env["GEMMALOOP_RUN_LOCK_HELD"]
        try:
            code = train_script.main(list(command[1:]))
        finally:
            if previous is None:
                os.environ.pop("GEMMALOOP_RUN_LOCK_HELD", None)
            else:
                os.environ["GEMMALOOP_RUN_LOCK_HELD"] = previous
        if calls["count"] == 1:
            assert int(code) == 75
            # The guard checkpoint is complete, but this is not a finished run.
            marker = json.loads(
                (out / "checkpoint-1" / "COMPLETE.json").read_text(encoding="utf-8")
            )
            assert marker["run_identity_hash"]
            assert not (out / "run.json").exists()
            assert not (out / "adapter_model.safetensors").exists()
            # The retry must run normally and the status file is writable again.
            state["armed"] = False
            state["status_broken"] = False
        return SimpleNamespace(returncode=int(code))

    monkeypatch.setattr(orchestrate.subprocess, "run", fake_run)
    code = orchestrate.main(
        [
            "--out", str(out),
            "--no-lock-check",
            "--backoff-initial", "0",
            "--max-retries", "5",
            "--",
            "run-train",
            *args,
        ]
    )
    assert code == 0
    assert calls["count"] == 2

    # The successful retry wrote the final artifacts and run.json.
    assert (out / "run.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()

    first, resumed = trainers[0], trainers[1]
    assert first.exit_code == 75
    assert first.exit_reason and "ram" in first.exit_reason
    # The retry really resumed (this kills a train.py that passes None).
    assert resumed._resume_global_step == 1
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["resumed_from"] and run_json["resumed_from"].endswith("checkpoint-1")

    # Uninterrupted reference with the same command.
    assert train_script.main(full_args) == 0
    train_script.release_run_lock()
    full = trainers[-1]
    for name, parameter in resumed.model.named_parameters():
        assert torch.equal(parameter.detach(), dict(full.model.named_parameters())[name].detach()), name
    assert resumed.state.global_step == full.state.global_step
    assert resumed.tokens_seen == full.tokens_seen

    # C1: relaunching the completed CLI run is a no-op and touches nothing.
    run_before = (full_out / "run.json").read_bytes()
    adapter_before = (full_out / "adapter_model.safetensors").read_bytes()
    assert train_script.main(full_args) == 0
    train_script.release_run_lock()
    assert (full_out / "run.json").read_bytes() == run_before
    assert (full_out / "adapter_model.safetensors").read_bytes() == adapter_before
    assert (full_out / "DONE.json").is_file()
    done = json.loads((full_out / "DONE.json").read_text(encoding="utf-8"))
    assert done["smoke"] is False

    # C1: the orchestrator also exits 0 without launching a completed run.
    launches_before = calls["count"]
    events_before = len(
        (full_out / orchestrate.ORCHESTRATOR_LOG).read_text(encoding="utf-8").splitlines()
    ) if (full_out / orchestrate.ORCHESTRATOR_LOG).is_file() else 0
    assert orchestrate.main(
        [
            "--out", str(full_out),
            "--no-lock-check",
            "--",
            "run-train",
            *full_args,
        ]
    ) == 0
    assert calls["count"] == launches_before
    events = (full_out / orchestrate.ORCHESTRATOR_LOG).read_text(encoding="utf-8").splitlines()
    assert len(events) == events_before + 2  # start + done, no launch


def test_guard_exit_75_then_orchestrator_resume_matches_uninterrupted(tmp_path, processor, monkeypatch):
    _run(tmp_path, processor, monkeypatch)
