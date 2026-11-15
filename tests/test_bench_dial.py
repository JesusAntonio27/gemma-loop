"""CPU checks for the T-dial-bench registered configuration."""

from __future__ import annotations

from scripts import bench_dial


def test_bench_defaults_are_the_registered_configuration():
    args = bench_dial.parse_args([])
    assert args.grad_accum == 16
    assert args.stage2_horizons == "6,8"
    assert bench_dial._int_list(args.stage2_horizons) == [6, 8]
    assert args.window == 3
    assert args.length == 8192


def test_bench_records_objective_and_grad_accum_fields():
    rows = bench_dial.synthetic_rows(8, 1, seed=0)
    assert len(rows[0]["input_ids"]) == 8
    assert len(rows[0]["teacher_topk_indices"]) == 8
    assert all(len(row["teacher_topk_indices"]) == 8 for row in rows)


def test_main_runs_dial_stage2_and_heal_and_writes_atomically(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls: list[tuple[str, object, object]] = []
    writes: list[list[dict]] = []

    monkeypatch.setattr(bench_dial, "require_gpu_lock", lambda: None)
    monkeypatch.setattr(bench_dial.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        bench_dial.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=12 * 2**30),
    )
    monkeypatch.setattr(
        bench_dial.torch.cuda,
        "mem_get_info",
        lambda: (12 * 2**30, 12 * 2**30),
    )
    monkeypatch.setattr(
        bench_dial.torch.cuda, "set_per_process_memory_fraction", lambda _value: None
    )
    monkeypatch.setattr(bench_dial, "other_gpu_python_processes", lambda: [])

    def fake_measure(block, horizon, args):
        calls.append(("dial", block, horizon))
        return {"objective": "dial", "block": block, "T": horizon, "oom": False}

    def fake_stage2(block, horizon, args):
        calls.append(("stage2", block, horizon))
        return {"objective": "stage2_rl", "block": block, "T": horizon, "oom": False}

    def fake_heal(args):
        calls.append(("heal", "", 1))
        return {"objective": "heal_oh", "block": "", "T": 1, "oom": False}

    def fake_write(path, payload):
        writes.append(list(payload))

    monkeypatch.setattr(bench_dial, "measure", fake_measure)
    monkeypatch.setattr(bench_dial, "measure_stage2", fake_stage2)
    monkeypatch.setattr(bench_dial, "measure_heal", fake_heal)
    monkeypatch.setattr(bench_dial, "write_json_atomic", fake_write)

    assert (
        bench_dial.main(
            [
                "--blocks", "1-2,3-4",
                "--horizons", "4",
                "--stage2-horizons", "6,8",
                "--out", str(tmp_path / "bench.json"),
                "--work-dir", str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert calls == [
        ("dial", "1-2", 4),
        ("stage2", "1-2", 6),
        ("stage2", "1-2", 8),
        ("dial", "3-4", 4),
        ("stage2", "3-4", 6),
        ("stage2", "3-4", 8),
        ("heal", "", 1),
    ]
    assert len(writes) == len(calls)


def _tiny_bench_setup(monkeypatch):
    from types import SimpleNamespace

    import torch
    from transformers import Gemma4ForCausalLM

    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from tests.model.tiny import tiny_config
    from tests.train.test_trainer_tiny import _tiny_tokenizer

    def fake_load_model(spec):
        torch.manual_seed(0)
        model = Gemma4ForCausalLM(tiny_config())
        plan = make_layer_plan(tiny_config().to_dict())
        if spec.loop is not None:
            apply_loops(
                model.model,
                plan,
                LoopConfig(
                    n_loops=1,
                    bridge="gch",
                    ple_mode="shared",
                    adapter_policy="loop_only",
                    lora_target_layers="loop_block",
                ),
            )
        return model, _tiny_tokenizer(), plan

    monkeypatch.setattr(bench_dial, "load_model", fake_load_model)

    from peft import LoraConfig

    def tiny_lora_config(target_layers=None, plan=None):
        del target_layers, plan
        return LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

    monkeypatch.setattr(bench_dial, "default_lora_config", tiny_lora_config)
    monkeypatch.setattr(bench_dial.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        bench_dial.torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None
    )
    monkeypatch.setattr(
        bench_dial.torch.cuda, "max_memory_reserved", lambda: 0
    )
    monkeypatch.setattr(
        bench_dial.torch.cuda, "max_memory_allocated", lambda: 0
    )
    monkeypatch.setattr(bench_dial.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench_dial.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        bench_dial.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=12 * 2**30),
    )
    return fake_load_model


def _tiny_args(tmp_path):
    return bench_dial.parse_args(
        [
            "--blocks", "1-2",
            "--horizons", "4",
            "--stage2-horizons", "6",
            "--length", "24",
            "--vocab-size", "256",
            "--opt-steps", "1",
            "--warmup-steps", "0",
            "--grad-accum", "1",
            "--out", str(tmp_path / "bench.json"),
            "--work-dir", str(tmp_path / "work"),
        ]
    )


def test_measure_functions_run_for_real_on_a_tiny_model(tmp_path, monkeypatch):
    _tiny_bench_setup(monkeypatch)
    args = _tiny_args(tmp_path)

    dial = bench_dial.measure("1-2", 4, args)
    assert dial["objective"] == "dial"
    assert dial["oom"] is False
    assert "gradient_checkpointing" in dial

    stage2 = bench_dial.measure_stage2("1-2", 6, args)
    assert stage2["objective"] == "stage2_rl"
    assert stage2["oom"] is False
    assert stage2["gradient_checkpointing"] in (True, False)

    heal = bench_dial.measure_heal(args)
    assert heal["objective"] == "heal_oh"
    assert heal["oom"] is False
    assert heal["gradient_parameters"] > 0


def test_main_runs_the_real_measures_and_writes_all_sections(tmp_path, monkeypatch):
    import json

    _tiny_bench_setup(monkeypatch)
    # main() gates on CUDA (first call True); the measures must then build and
    # run on the CPU, so every later availability probe returns False.
    state = {"calls": 0}

    def first_call_true():
        state["calls"] += 1
        return state["calls"] == 1

    monkeypatch.setattr(bench_dial.torch.cuda, "is_available", first_call_true)
    real_build = bench_dial.build_sft_config

    def cpu_build(*args, **kwargs):
        kwargs.update({"use_cpu": True, "bf16": False, "fp16": False})
        return real_build(*args, **kwargs)

    monkeypatch.setattr(bench_dial, "build_sft_config", cpu_build)
    monkeypatch.setattr(bench_dial, "require_gpu_lock", lambda: None)
    monkeypatch.setattr(bench_dial, "other_gpu_python_processes", lambda: [])
    monkeypatch.setattr(
        bench_dial.torch.cuda,
        "mem_get_info",
        lambda: (12 * 2**30, 12 * 2**30),
    )
    monkeypatch.setattr(
        bench_dial.torch.cuda, "set_per_process_memory_fraction", lambda _v: None
    )
    out = tmp_path / "bench.json"
    assert (
        bench_dial.main(
            [
                "--blocks", "1-2",
                "--horizons", "4",
                "--stage2-horizons", "6",
                "--length", "24",
                "--vocab-size", "256",
                "--opt-steps", "1",
                "--warmup-steps", "0",
                "--grad-accum", "1",
                "--out", str(out),
                "--work-dir", str(tmp_path / "work"),
            ]
        )
        == 0
    )
    records = json.loads(out.read_text(encoding="utf-8"))
    assert {record["objective"] for record in records} == {
        "dial",
        "stage2_rl",
        "heal_oh",
    }
    assert len(records) == 3
