"""Round-2 regression tests for the re-audit findings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_modal_guardrails_round1 import (
    _App,
    _Call,
    _Clock,
    _FakeDict,
    _FakeModal,
    _FakeVolume,
    _Function,
    _run,
    _write_heartbeat,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_round2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _entries(tmp_path):
    import json

    return json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]


def _last_actual(tmp_path):
    return [e for e in _entries(tmp_path) if e["kind"] == "actual"][-1]


class NotFoundError(Exception):
    pass


class _ListVolume(_FakeVolume):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def listdir(self, path, recursive=False):
        raise self.exc


class _DeadHeartbeat(_FakeDict):
    def get(self, key, default=None):
        raise RuntimeError("Dict unavailable")


class _DeadDictManager:
    def __init__(self):
        self.heartbeats: list[_DeadHeartbeat] = []

    def ephemeral(self):
        from test_modal_guardrails_round1 import _FakeDictContext

        heartbeat = _DeadHeartbeat()
        self.heartbeats.append(heartbeat)
        return _FakeDictContext(heartbeat)


class _BoomHeartbeat:
    def __setitem__(self, key, value):
        raise RuntimeError("Dict write failed")


# ---------------------------------------------------------------------------
# R-B1: NotFoundError means "no shards yet"
# ---------------------------------------------------------------------------
def test_missing_run_directory_is_not_a_refusal(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal(volume=_ListVolume(NotFoundError("no such directory")))
    app = _App()
    preflight = _Function(wall_clock=clock, result={"ok": True})
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 0
    assert gpu.calls >= 1


def test_other_listing_errors_still_abort(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal(volume=_ListVolume(RuntimeError("grpc down")))
    app = _App()
    preflight = _Function(wall_clock=clock, result={"ok": True})
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 1
    assert gpu.calls == 0
    assert preflight.calls == 0


# ---------------------------------------------------------------------------
# R-B2: heartbeat channel must be proven before the GPU request
# ---------------------------------------------------------------------------
def test_dead_heartbeat_channel_aborts_before_gpu(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    modal.dicts = _DeadDictManager()
    modal.Dict = modal.dicts
    app = _App()
    preflight = _Function(wall_clock=clock, result={"ok": True})
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 1
    assert preflight.calls == 1
    assert gpu.calls == 0
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "preflight_failure"


def test_preflight_heartbeat_container_id_must_match(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    preflight = _Function(
        wall_clock=clock,
        result={
            "ok": True,
            "heartbeat": {
                "container_id": "other-container",
                "delivered": 2,
                "background_delivered": 1,
                "failed_writes": 0,
            },
        },
    )
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 1
    assert gpu.calls == 0
    assert app.stops == 1


def test_three_consecutive_heartbeat_write_failures_raise():
    script = _load_script()
    state = script._heartbeat_begin(
        {"_heartbeat": _BoomHeartbeat(), "heartbeat_key": "k"}
    )
    assert state is not None
    script._heartbeat_beat(state)
    with pytest.raises(script.HeartbeatWriteError):
        script._heartbeat_beat(state, raise_on_failure=True)


# ---------------------------------------------------------------------------
# R-I1: stats are stop-only evidence before the first heartbeat
# ---------------------------------------------------------------------------
def test_stats_second_container_stops_before_startup_deadline(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    stats = iter(
        [
            SimpleNamespace(num_running_inputs=0, num_total_runners=0),
            SimpleNamespace(num_running_inputs=0, num_total_runners=1),
            SimpleNamespace(num_running_inputs=0, num_total_runners=0),
            SimpleNamespace(num_running_inputs=0, num_total_runners=1),
        ]
    )

    def make_call(_payload):
        return _Call(wall_clock=clock, error=TimeoutError("no result"), advance=30.0)

    gpu = _Function(
        wall_clock=clock, call_factory=make_call, write_heartbeat=False
    )
    gpu.get_current_stats = lambda: next(stats)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=600.0,
        )
    )
    assert result == 1
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "startup_retry"


# ---------------------------------------------------------------------------
# R-I3: hard deadline includes the startup allowance
# ---------------------------------------------------------------------------
def _direct_call(script, clock, *, startup_timeout_s, timeout_s, finish_at):
    heartbeat = _FakeDict()

    def make_call(payload):
        def on_get(call):
            _write_heartbeat(payload, "container-1", sequence=call.gets + 1)
            if clock.now >= finish_at:
                call.error = None
                call.result = {"ok": True}
                return
            raise TimeoutError("still running")

        return _Call(wall_clock=clock, payload=payload, advance=60.0, on_get=on_get)

    fn = _Function(wall_clock=clock, call_factory=make_call)
    return script._call_remote(
        fn,
        {"_heartbeat": heartbeat, "heartbeat_key": "h"},
        wall_clock=clock,
        startup_timeout_s=startup_timeout_s,
        timeout_s=timeout_s,
        heartbeat=heartbeat,
        heartbeat_key="h",
    ), heartbeat


def test_hard_deadline_is_startup_plus_registered_timeout():
    script = _load_script()
    clock = _Clock()
    with pytest.raises(script.RemoteCallDeadline):
        _direct_call(
            script,
            clock,
            startup_timeout_s=600.0,
            timeout_s=300.0,
            finish_at=10_000.0,
        )
    assert clock.now >= 600.0 + 300.0


def test_healthy_shard_is_not_cut_before_its_registered_timeout():
    script = _load_script()
    clock = _Clock()
    result, _ = _direct_call(
        script,
        clock,
        startup_timeout_s=600.0,
        timeout_s=500.0,
        finish_at=700.0,
    )
    assert result == {"ok": True}
    assert clock.now >= 700.0


# ---------------------------------------------------------------------------
# M-R2: Dict read outage does not cancel a healthy call
# ---------------------------------------------------------------------------
class _FlakyReadHeartbeat(_FakeDict):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def get(self, key, default=None):
        self.reads += 1
        if 2 <= self.reads <= 8:
            raise RuntimeError("Dict read outage")
        return super().get(key, default)


def test_heartbeat_read_outage_does_not_cancel_a_healthy_call():
    script = _load_script()
    clock = _Clock()
    heartbeat = _FlakyReadHeartbeat()

    def make_call(payload):
        def on_get(call):
            _write_heartbeat(payload, "container-1", sequence=call.gets + 1)
            if clock.now >= 500.0:
                call.error = None
                call.result = {"ok": True}
                return
            raise TimeoutError("still running")

        return _Call(wall_clock=clock, payload=payload, advance=50.0, on_get=on_get)

    fn = _Function(wall_clock=clock, call_factory=make_call)
    result = script._call_remote(
        fn,
        {"_heartbeat": heartbeat, "heartbeat_key": "h"},
        wall_clock=clock,
        startup_timeout_s=600.0,
        timeout_s=1800.0,
        heartbeat=heartbeat,
        heartbeat_key="h",
    )
    assert result == {"ok": True}


# ---------------------------------------------------------------------------
# M-R4: success booking includes preflight CPU time and final scaledown
# ---------------------------------------------------------------------------
def test_success_booking_includes_launch_overhead(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def preflight_call(payload):
        _write_heartbeat(payload, "container-1")
        return _Call(
            wall_clock=clock,
            result={
                "ok": True,
                "heartbeat": {
                    "container_id": "container-1",
                    "delivered": 2,
                    "background_delivered": 1,
                    "failed_writes": 0,
                },
            },
            advance=100.0,
        )

    preflight = _Function(wall_clock=clock, call_factory=preflight_call)
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 0
    actual = [e for e in _entries(tmp_path) if e["kind"] == "actual"][0]
    expected = (
        script.DEFAULT_IMAGE_BUILD_USD
        + script._preflight_cost_usd(100.0)
        + script.runtime_cost_usd(
            "A100-80GB",
            script.DEFAULT_SCALEDOWN_SECONDS,
            cpu_cores=int(script.DEFAULT_CPU),
            memory_gib=script.DEFAULT_MEMORY_GIB,
            safety_margin=script.DEFAULT_SAFETY_MARGIN,
        )["usd"]
    )
    assert actual["metadata"]["launch_overhead_usd"] == pytest.approx(expected)
