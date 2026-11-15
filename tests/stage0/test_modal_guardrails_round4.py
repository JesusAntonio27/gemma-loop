"""Round-4 regression tests: post-liveness stats baseline and preflight writer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from test_modal_guardrails_round1 import (
    _App,
    _Call,
    _Clock,
    _FakeDict,
    _FakeModal,
    _Function,
    _run,
    _write_heartbeat,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_round4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _stats_sequence(values):
    sequence = iter(values)
    last = values[-1] if values else 0

    def read():
        try:
            return next(sequence)
        except StopIteration:
            return last

    return lambda: SimpleNamespace(
        num_running_inputs=0, num_total_runners=read()
    )


def _run_with_stats(tmp_path, stats_values):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(payload):
        def on_get(call):
            _write_heartbeat(payload, "container-1", sequence=call.gets + 1)
            if clock.now >= 500.0:
                call.error = None
                call.result = {"shard": 0, "samples": 4}
                return
            raise TimeoutError("running")

        return _Call(
            wall_clock=clock, payload=payload, advance=50.0, on_get=on_get
        )

    gpu = _Function(wall_clock=clock, call_factory=make_call)
    gpu.get_current_stats = _stats_sequence(stats_values)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            max_shards=1,
        )
    )
    return result, app


def test_stats_lag_after_first_heartbeat_does_not_stop(tmp_path):
    result, app = _run_with_stats(tmp_path, [0, 1, 1, 1, 1, 1])
    assert result == 0
    assert app.stops == 0


def test_single_zero_stats_read_mid_run_does_not_stop(tmp_path):
    result, app = _run_with_stats(tmp_path, [1, 0, 1, 1, 1, 1])
    assert result == 0
    assert app.stops == 0


def test_real_one_to_two_runner_increase_after_heartbeat_stops(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(
            wall_clock=clock, error=TimeoutError("running"), advance=50.0
        )

    gpu = _Function(wall_clock=clock, call_factory=make_call)
    gpu.get_current_stats = _stats_sequence([1, 2, 2, 2])
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            max_shards=1,
        )
    )
    assert result == 1
    assert app.stops == 1


def test_preflight_proves_the_background_writer(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(
        script,
        "_verify_jit_toolchain",
        lambda: {"cuda_home": "/usr/local/cuda", "flashinfer_cuda": "/usr/local/cuda"},
    )
    monkeypatch.setattr(
        script,
        "_import_remote_closure",
        lambda: (
            SimpleNamespace(__name__="huggingface_hub"),
            SimpleNamespace(__name__="transformers"),
            SimpleNamespace(__name__="vllm"),
        ),
    )
    heartbeat = _FakeDict()
    result = script.preflight_check(
        {
            "_heartbeat": heartbeat,
            "heartbeat_key": "preflight",
            "heartbeat_interval": 0.02,
        }
    )
    assert result["ok"] is True
    assert result["heartbeat"]["background_delivered"] >= 1
