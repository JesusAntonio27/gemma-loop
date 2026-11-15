"""Round-1 guard tests: real in-container heartbeat, hard deadlines, crash loops.

These fakes model the production ``spawn()`` / ``FunctionCall.get(timeout=...)``
path.  They never import or contact Modal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"
FIXED_NOW = lambda: datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_round1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _manifest_sha(payload):
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _pool(tmp_path: Path, count: int = 2) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    rows = [
        {"problem_id": f"p:{index}", "prompt": f"prompt {index}", "kind": "code"}
        for index in range(count)
    ]
    problems = "\n".join(json.dumps(row) for row in rows) + "\n"
    (pool / "problems.jsonl").write_bytes(problems.encode("utf-8"))
    payload = {
        "stage": "stage0_pool",
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(problems.encode()).hexdigest(),
                "rows": count,
            }
        },
        "decontamination": {
            "status": "ok",
            "targets": {"livecodebench_v6": {"status": "ok"}},
        },
        "limits": {
            "eval_limit": None,
            "limit_rows": None,
            "allow_missing_eval": False,
            "allow_missing_briefs": False,
            "allow_short_sources": False,
            "allow_duplicates": False,
        },
    }
    payload["manifest_sha256"] = _manifest_sha(payload)
    (pool / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return pool


class _Clock:
    def __init__(self, now: float = 0.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class _FakeDict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class _FakeDictManager:
    def __init__(self):
        self.dicts: list[_FakeDict] = []

    def ephemeral(self):
        heartbeat = _FakeDict()
        self.dicts.append(heartbeat)
        return _FakeDictContext(heartbeat)


class _FakeDictContext:
    def __init__(self, heartbeat):
        self.heartbeat = heartbeat

    def __enter__(self):
        return self.heartbeat

    def __exit__(self, *_exc):
        return False


class _FakeVolume:
    def __init__(self, *, fail_listdir: bool = False):
        self.fail_listdir = fail_listdir
        self.files: dict[str, bytes] = {}

    def listdir(self, path, recursive=False):
        if self.fail_listdir:
            raise RuntimeError("volume listing failed")
        return []

    def commit(self):
        pass


class _FakeModal:
    def __init__(self, volume=None):
        self._volume = volume or _FakeVolume()
        self.dicts = _FakeDictManager()
        self.created: list[tuple[str, bool]] = []

        class VolumeClass:
            @classmethod
            def from_name(cls, name, create_if_missing=False):
                cls_created = self.created
                cls_created.append((name, create_if_missing))
                return self._volume

        self.Volume = VolumeClass
        self.Dict = self.dicts


def _write_heartbeat(payload, container_id, *, sequence=1, phase="entered"):
    heartbeat = payload.get("_heartbeat")
    key = payload.get("heartbeat_key")
    if heartbeat is None or key is None:
        return
    heartbeat[key] = {
        "container_id": container_id,
        "sequence": sequence,
        "phase": phase,
    }


class _Call:
    def __init__(
        self,
        *,
        wall_clock,
        result=None,
        error=None,
        advance=0.0,
        on_get=None,
        payload=None,
    ):
        self.wall_clock = wall_clock
        self.result = result
        self.error = error
        self.advance = advance
        self.on_get = on_get
        self.payload = payload
        self.gets = 0
        self.cancelled = 0
        self.terminate_containers = None

    def get(self, timeout=None):
        assert timeout is not None, "call.get() must always carry a timeout"
        assert timeout >= 0
        self.gets += 1
        if self.advance:
            self.wall_clock.advance(self.advance)
        if self.on_get is not None:
            self.on_get(self)
        if self.error is not None:
            raise self.error
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled += 1
        self.terminate_containers = terminate_containers


class _Function:
    def __init__(
        self,
        *,
        wall_clock,
        result=None,
        spawn_error=None,
        call_factory=None,
        write_heartbeat=True,
        container_id="container-1",
        stats_error=None,
    ):
        self.wall_clock = wall_clock
        self.result = result if result is not None else {"shard": 0, "samples": 4}
        self.spawn_error = spawn_error
        self.call_factory = call_factory
        self.write_heartbeat = write_heartbeat
        self.container_id = container_id
        self.stats_error = stats_error
        self.calls = 0
        self.options: dict = {}
        self.calls_made: list[_Call] = []

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def spawn(self, payload):
        self.calls += 1
        if self.write_heartbeat:
            _write_heartbeat(payload, self.container_id)
        if self.spawn_error is not None:
            raise self.spawn_error
        if self.call_factory is not None:
            call = self.call_factory(payload)
        else:
            result = self.result
            if (
                isinstance(result, dict)
                and result.get("ok") is True
                and "heartbeat" not in result
            ):
                result = {
                    **result,
                    "heartbeat": {
                        "container_id": self.container_id,
                        "delivered": 2,
                        "background_delivered": 1,
                        "failed_writes": 0,
                    },
                }
            call = _Call(
                wall_clock=self.wall_clock,
                result=result,
                payload=payload,
            )
        self.calls_made.append(call)
        return call

    def get_current_stats(self):
        if self.stats_error is not None:
            raise self.stats_error
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)


class _App:
    def __init__(self):
        self.stops = 0
        self.entered = 0

    def run(self):
        return self

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_exc):
        return False

    def stop(self):
        self.stops += 1


def _run(tmp_path, modal, fn, preflight, **overrides):
    kwargs = dict(
        mode="sample",
        pool_dir=overrides.pop("pool_dir", None) or _pool(tmp_path),
        output_dir=tmp_path / "samples",
        gpu="A100-80GB",
        gpu_id="A100-80GB",
        max_usd=12.0,
        shard_size=1,
        avg_generated_tokens=3200,
        samples_per_prompt=4,
        yes=True,
        fn=fn,
        preflight_fn=preflight,
        modal_module=modal,
        ledger_path=tmp_path / "spend.json",
        spend_lock_path=tmp_path / "spend.lock",
        init_ledger=True,
        month_now=FIXED_NOW,
    )
    kwargs.update(overrides)
    return kwargs


def _entries(tmp_path):
    return json.loads(
        (tmp_path / "spend.json").read_text(encoding="utf-8")
    )["entries"]


def _last_actual(tmp_path):
    return [entry for entry in _entries(tmp_path) if entry["kind"] == "actual"][-1]


# ---------------------------------------------------------------------------
# B1 / I1 / review-2 critical 1: heartbeat instead of runner stats
# ---------------------------------------------------------------------------
def test_startup_deadline_ignores_runner_stats_and_stops(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(
            wall_clock=clock,
            error=TimeoutError("no result"),
            advance=30.0,
        )

    fn = _Function(
        wall_clock=clock,
        call_factory=make_call,
        write_heartbeat=False,
    )
    fn.get_current_stats = lambda: SimpleNamespace(
        num_running_inputs=0, num_total_runners=1
    )
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=600.0,
        )
    )
    assert result == 1
    assert fn.calls == 1
    assert app.stops == 1
    assert fn.calls_made[0].cancelled == 1
    assert fn.calls_made[0].terminate_containers is True
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "startup_timeout"


def test_crash_loop_second_container_heartbeat_stops_immediately(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def on_get(call):
        _write_heartbeat(call.payload, "container-2", sequence=1)
        raise TimeoutError("no result")

    def make_call(payload):
        return _Call(wall_clock=clock, payload=payload, advance=1.0, on_get=on_get)

    fn = _Function(wall_clock=clock, call_factory=make_call)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=600.0,
        )
    )
    assert result == 1
    assert fn.calls == 1
    assert fn.calls_made[0].gets == 1
    assert app.stops == 1
    assert fn.calls_made[0].cancelled == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "startup_retry"


def test_heartbeat_gap_stops_a_call_that_stopped_reporting(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(
            wall_clock=clock,
            error=TimeoutError("no result"),
            advance=200.0,
        )

    fn = _Function(wall_clock=clock, call_factory=make_call)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=600.0,
            timeout_s=7200.0,
        )
    )
    assert result == 1
    assert app.stops == 1
    assert fn.calls_made[0].cancelled == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "heartbeat_gap"


def test_hard_deadline_cancels_even_while_heartbeats_continue(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(payload):
        def on_get(call):
            _write_heartbeat(payload, "container-1", sequence=call.gets + 1)
            raise TimeoutError("still running")

        return _Call(wall_clock=clock, payload=payload, advance=60.0, on_get=on_get)

    fn = _Function(wall_clock=clock, call_factory=make_call)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=900.0,
            timeout_s=300.0,
        )
    )
    assert result == 1
    assert app.stops == 1
    assert fn.calls_made[0].cancelled == 1
    assert fn.calls_made[0].terminate_containers is True
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "call_deadline"


def test_stats_failure_does_not_cancel_a_healthy_heartbeating_call(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(payload):
        def on_get(call):
            _write_heartbeat(payload, "container-1", sequence=call.gets + 1)
            if call.gets >= 5:
                call.result = {"shard": 0, "samples": 4}
                call.error = None
                return
            raise TimeoutError("still running")

        return _Call(
            wall_clock=clock,
            payload=payload,
            advance=10.0,
            on_get=on_get,
        )

    fn = _Function(
        wall_clock=clock,
        call_factory=make_call,
        stats_error=RuntimeError("stats RPC failed"),
    )
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
            startup_timeout_s=30.0,
            timeout_s=7200.0,
            max_shards=1,
        )
    )
    assert result == 0
    assert app.stops == 0
    assert fn.calls == 1


# ---------------------------------------------------------------------------
# Preflight mandatory, validated, spawn-path and no GPU on failure
# ---------------------------------------------------------------------------
def test_missing_preflight_refuses_before_modal_contact(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    fn = _Function(wall_clock=clock)
    kwargs = _run(tmp_path, modal, fn, None)
    kwargs.pop("preflight_fn")
    result = script.run_mode(**kwargs)
    assert result == 2
    assert fn.calls == 0
    assert modal.created == []


def test_preflight_spawn_failure_makes_zero_gpu_calls(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    preflight = _Function(
        wall_clock=clock,
        spawn_error=RuntimeError("missing import"),
    )
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(
        **_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock)
    )
    assert result == 1
    assert preflight.calls == 1
    assert gpu.calls == 0
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "preflight_failure"


def test_preflight_not_ok_refuses_the_gpu(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    preflight = _Function(wall_clock=clock, result={"ok": False})
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(**_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock))
    assert result == 1
    assert gpu.calls == 0
    assert app.stops == 1
    assert _last_actual(tmp_path)["metadata"]["stopped_reason"] == "preflight_failure"


# ---------------------------------------------------------------------------
# Volume listing, bounded failure booking, startup timeout ceiling
# ---------------------------------------------------------------------------
def test_failed_volume_listing_aborts_without_launching(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal(volume=_FakeVolume(fail_listdir=True))
    app = _App()
    preflight = _Function(wall_clock=clock, result={"ok": True})
    gpu = _Function(wall_clock=clock)
    result = script.run_mode(**_run(tmp_path, modal, gpu, preflight, app=app, wall_clock=clock))
    assert result == 1
    assert gpu.calls == 0
    assert preflight.calls == 0


def test_failure_books_bounded_elapsed_not_full_worst(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()

    def make_call(_payload):
        return _Call(
            wall_clock=clock,
            error=RuntimeError("boom"),
            advance=60.0,
        )

    fn = _Function(wall_clock=clock, call_factory=make_call)
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            fn,
            _Function(wall_clock=clock, result={"ok": True}),
            app=app,
            wall_clock=clock,
        )
    )
    assert result == 1
    entries = _entries(tmp_path)
    reservation = [entry for entry in entries if entry["kind"] == "reservation"][-1]
    actual = _last_actual(tmp_path)
    assert actual["usd"] < reservation["usd"]
    assert actual["metadata"]["elapsed_s"] == pytest.approx(60.0)
    assert actual["metadata"]["booking_basis"] == "elapsed_upper_bound"


def test_preflight_exit_wait_fails_closed_when_stats_never_confirm(monkeypatch):
    script = _load_script()
    clock = _Clock()

    class StatsFailure:
        def get_current_stats(self):
            raise RuntimeError("stats RPC failed")

    monkeypatch.setattr(script.time, "sleep", clock.advance)
    with pytest.raises(script.PreflightError, match="confirm"):
        script._wait_for_preflight_exit(StatsFailure(), wall_clock=clock)


def test_preflight_exit_wait_polls_until_the_container_leaves(monkeypatch):
    script = _load_script()
    clock = _Clock()
    replies = iter(
        [
            SimpleNamespace(num_running_inputs=1, num_total_runners=1),
            SimpleNamespace(num_running_inputs=0, num_total_runners=0),
        ]
    )
    sleeps: list[float] = []

    class Stats:
        def get_current_stats(self):
            return next(replies)

    monkeypatch.setattr(script.time, "sleep", lambda seconds: sleeps.append(seconds))
    script._wait_for_preflight_exit(Stats(), wall_clock=clock)
    assert sleeps == [0.5]


def test_startup_timeout_cannot_be_disabled_and_has_a_ceiling(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    fn = _Function(wall_clock=clock)
    preflight = _Function(wall_clock=clock, result={"ok": True})
    for bad in (0, 901):
        with pytest.raises(ValueError, match="startup_timeout"):
            script.run_mode(
                **_run(
                    tmp_path,
                    modal,
                    fn,
                    preflight,
                    wall_clock=clock,
                    startup_timeout_s=bad,
                )
            )


# ---------------------------------------------------------------------------
# Image resources: CPU-only preflight and no nonpreemptible GPU
# ---------------------------------------------------------------------------
def test_build_app_preflight_exits_immediately_and_gpu_has_no_nonpreemptible():
    script = _load_script()
    captured: list[dict] = []

    class FakeImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        @classmethod
        def from_registry(cls, tag, **kwargs):
            return cls()

        def apt_install(self, *args):
            return self

        def pip_install(self, *args):
            return self

        def env(self, values):
            return self

        def add_local_python_source(self, *modules):
            return self

    class FakeApp:
        def __init__(self, name):
            pass

        def function(self, **kwargs):
            captured.append(kwargs)
            return lambda fn: SimpleNamespace()

    script._build_modal_app(SimpleNamespace(App=FakeApp, Image=FakeImage))
    preflight = [config for config in captured if "gpu" not in config]
    gpu = [config for config in captured if "gpu" in config]
    assert len(preflight) == 1 and len(gpu) == 1
    assert preflight[0]["scaledown_window"] == 2
    assert preflight[0]["memory"] == (16384, 16384)
    assert "nonpreemptible" not in preflight[0]
    assert "nonpreemptible" not in gpu[0]
