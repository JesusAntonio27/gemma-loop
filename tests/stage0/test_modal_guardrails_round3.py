"""Round-3 regression tests: cold start, main-thread work, reservation bound."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    spec = importlib.util.spec_from_file_location("modal_stage0_round3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _last_actual(tmp_path):
    entries = json.loads(
        (tmp_path / "spend.json").read_text(encoding="utf-8")
    )["entries"]
    return [e for e in entries if e["kind"] == "actual"][-1]


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


# ---------------------------------------------------------------------------
# R2-B1: normal cold start 0 -> 1 must be healthy
# ---------------------------------------------------------------------------
def test_healthy_cold_start_zero_to_one_is_not_a_second_container(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(payload):
        def on_get(call):
            if call.gets == 2:
                _write_heartbeat(payload, "container-1", sequence=1)
            if call.gets >= 4:
                call.error = None
                call.result = {"shard": 0, "samples": 4}
                return
            raise TimeoutError("starting")

        return _Call(wall_clock=clock, payload=payload, advance=30.0, on_get=on_get)

    gpu = _Function(
        wall_clock=clock, call_factory=make_call, write_heartbeat=False
    )
    gpu.get_current_stats = _stats_sequence([0, 1, 1, 1, 1, 1])
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=600.0,
            max_shards=1,
        )
    )
    assert result == 0
    assert app.stops == 0
    assert gpu.calls == 1


def test_crash_loop_zero_one_zero_one_stops(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(wall_clock=clock, error=TimeoutError("no result"), advance=30.0)

    gpu = _Function(
        wall_clock=clock, call_factory=make_call, write_heartbeat=False
    )
    gpu.get_current_stats = _stats_sequence([0, 1, 0, 1])
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
        )
    )
    assert result == 1
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "startup_retry"


def test_preemption_restart_after_first_heartbeat_stops(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(wall_clock=clock, error=TimeoutError("no result"), advance=30.0)

    gpu = _Function(wall_clock=clock, call_factory=make_call)
    gpu.get_current_stats = _stats_sequence([1, 2])
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
        )
    )
    assert result == 1
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "startup_retry"


# ---------------------------------------------------------------------------
# R2-I1: vLLM work stays on the main thread; only the writer is a daemon thread
# ---------------------------------------------------------------------------
def test_sample_shard_work_runs_on_the_main_thread(monkeypatch):
    script = _load_script()
    seen: dict[str, bool] = {}

    def fake_impl(payload, heartbeat_state=None):
        seen["main"] = threading.current_thread() is threading.main_thread()
        return {"shard": 0, "skipped": False, "samples": 0}

    monkeypatch.setattr(script, "_sample_shard_impl", fake_impl)
    result = script.sample_shard(
        {
            "run_id": "r",
            "shard_index": 0,
            "problems": [],
            "_heartbeat": _FakeDict(),
            "heartbeat_key": "shard:0",
        }
    )
    assert result == {"shard": 0, "skipped": False, "samples": 0}
    assert seen["main"] is True


# ---------------------------------------------------------------------------
# R2-M2: reservation includes the preflight worst case
# ---------------------------------------------------------------------------
def test_reservation_includes_preflight_worst_case(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    gpu = _Function(wall_clock=clock)
    preflight = _Function(wall_clock=clock, result={"ok": True})
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 0
    manifest = json.loads(
        (tmp_path / "samples" / "launch_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    gpu_worst = script.worst_case_usd(
        "A100-80GB",
        timeout_s=script.DEFAULT_TIMEOUT_SECONDS,
        startup_timeout_s=script.DEFAULT_STARTUP_TIMEOUT_SECONDS,
        scaledown_window_s=script.DEFAULT_SCALEDOWN_SECONDS,
        cpu_cores=int(script.DEFAULT_CPU),
        memory_gib=script.DEFAULT_MEMORY_GIB,
        safety_margin=script.DEFAULT_SAFETY_MARGIN,
        image_build_usd=script.DEFAULT_IMAGE_BUILD_USD,
    )["usd"]
    preflight_worst = script._preflight_cost_usd(
        script.DEFAULT_STARTUP_TIMEOUT_SECONDS
        + script.PREFLIGHT_TIMEOUT_SECONDS
        + script.REMOTE_CALL_MARGIN_SECONDS
    )
    assert manifest["worst_case_usd"] == pytest.approx(
        gpu_worst + preflight_worst
    )
