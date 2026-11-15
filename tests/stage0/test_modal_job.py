"""Modal job tests: fake modules only; never contacts Modal, vLLM or HF."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"
FIXED_NOW = lambda: datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_under_test", SCRIPT)
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


def _pool(tmp_path: Path, count: int, *, manifest: bool = True, smoke: bool = False) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    rows = [
        {"problem_id": f"p:{index}", "prompt": f"prompt {index}", "kind": "code"}
        for index in range(count)
    ]
    problems = "\n".join(json.dumps(row) for row in rows) + "\n"
    (pool / "problems.jsonl").write_bytes(problems.encode("utf-8"))
    if manifest:
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
                "eval_limit": 64 if smoke else None,
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


class FakeEntry:
    def __init__(self, path: str, size: int | None = None):
        self.path = path
        self.size = size


class FakeVolume:
    def __init__(self, files: dict[str, bytes] | None = None, fail_commit: bool = False):
        self.files = files or {}
        self.fail_commit = fail_commit
        self.commits = 0

    def commit(self):
        self.commits += 1
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def listdir(self, path: str, recursive: bool = False):
        prefix = path.strip("/")
        prefix = prefix + "/" if prefix else ""
        entries = []
        for name, data in self.files.items():
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix) :]
            if not recursive and "/" in rest:
                continue
            entries.append(FakeEntry(name, len(data)))
        return entries

    def read_file(self, path: str):
        data = self.files[path.lstrip("/")]
        midpoint = max(1, len(data) // 2)
        return iter([data[:midpoint], data[midpoint:]])


class FakeVolumeManager:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, name: str, allow_missing: bool = True):
        self.deleted.append(name)


class FakeHeartbeat(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class FakeDictContext:
    def __init__(self, heartbeat):
        self.heartbeat = heartbeat

    def __enter__(self):
        return self.heartbeat

    def __exit__(self, *_exc):
        return False


class FakeDictManager:
    def __init__(self):
        self.heartbeats: list[FakeHeartbeat] = []

    def ephemeral(self):
        heartbeat = FakeHeartbeat()
        self.heartbeats.append(heartbeat)
        return FakeDictContext(heartbeat)


def _fake_modal(volume: FakeVolume | None = None):
    manager = FakeVolumeManager()
    created: list[tuple[str, bool]] = []

    class VolumeClass:
        @classmethod
        def from_name(cls, name, create_if_missing=False):
            created.append((name, create_if_missing))
            if volume is None:
                raise FileNotFoundError("volume missing")
            return volume

        objects = manager

    return (
        SimpleNamespace(Volume=VolumeClass, Dict=FakeDictManager()),
        created,
        manager,
    )


def _write_fake_heartbeat(payload, container_id="container-1", sequence=1):
    heartbeat = payload.get("_heartbeat")
    key = payload.get("heartbeat_key")
    if heartbeat is not None and key is not None:
        heartbeat[key] = {
            "container_id": container_id,
            "sequence": sequence,
            "phase": "entered",
        }


class FakeCall:
    def __init__(self, result):
        self.result = result
        self.gets = 0
        self.cancelled = 0
        self.terminate_containers = None

    def get(self, timeout=None):
        assert timeout is not None
        self.gets += 1
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled += 1
        self.terminate_containers = terminate_containers


class FakeFunction:
    def __init__(self, ledger_probe: Path | None = None, result=None):
        self.options = {}
        self.calls = 0
        self.ledger_probe = ledger_probe
        self.ledger_seen: list[str] = []
        self.result = (
            result
            if result is not None
            else {"shard": 0, "samples": 4}
        )
        self.get_error: BaseException | None = None
        self.spawn_error: BaseException | None = None
        self.container_id = "container-1"
        self.calls_made: list[FakeCall] = []

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def spawn(self, payload):
        self.calls += 1
        _write_fake_heartbeat(payload, self.container_id)
        if self.ledger_probe is not None:
            self.ledger_seen.append(self.ledger_probe.read_text(encoding="utf-8"))
        if self.spawn_error is not None:
            raise self.spawn_error
        call = FakeCall(
            {"shard": payload["shard_index"], "samples": 4}
            if self.result is None
            else self.result
        )
        if self.get_error is not None:
            def _raise(_timeout=None):
                call.gets += 1
                raise self.get_error

            call.get = _raise  # type: ignore[method-assign]
        self.calls_made.append(call)
        return call

    def remote(self, payload):
        self.calls += 1
        if self.ledger_probe is not None:
            self.ledger_seen.append(self.ledger_probe.read_text(encoding="utf-8"))
        _write_fake_heartbeat(payload, self.container_id)
        return {"shard": payload["shard_index"], "samples": 4}

    def get_current_stats(self):
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)


def FakePreflight(result=None):
    return FakeFunction(
        result=result
        or {
            "ok": True,
            "versions": {},
            "heartbeat": {
                "container_id": "container-1",
                "delivered": 2,
                "background_delivered": 1,
                "failed_writes": 0,
            },
        }
    )


def _run(tmp_path, fake, fn, **overrides):
    kwargs = dict(
        mode="sample",
        pool_dir=overrides.pop("pool_dir", None) or _pool(tmp_path, 2),
        output_dir=tmp_path / "samples",
        gpu="A100-80GB",
        gpu_id="A100-80GB",
        max_usd=12.0,
        shard_size=1,
        avg_generated_tokens=3200,
        samples_per_prompt=4,
        yes=True,
        fn=fn,
        preflight_fn=overrides.pop("preflight_fn", None) or FakePreflight(),
        modal_module=fake,
        ledger_path=tmp_path / "spend.json",
        spend_lock_path=tmp_path / "spend.lock",
        init_ledger=True,
        month_now=FIXED_NOW,
    )
    kwargs.update(overrides)
    return kwargs


def _summary(capsys):
    out = capsys.readouterr().out
    marker = '{\n  "shards_completed"'
    return json.loads(out[out.rindex(marker) :])


def test_parse_completion_with_real_gemma_channel_fixtures():
    script = _load_script()
    fixtures = ROOT / "tests" / "fixtures" / "gemma_outputs"
    parsed = script.parse_completion((fixtures / "reasoning.txt").read_text(encoding="utf-8"))
    assert parsed == {"reasoning": "plan", "content": "Hola", "thought_closed": True}
    assert script.parse_completion("Hola")["content"] == "Hola"
    assert script.parse_completion("<|channel>thought\nunfinished")["thought_closed"] is False
    assert script.parse_completion("<|channel>thought\nplan<channel|>Hola<turn|>\n")["content"] == "Hola"


def test_sampling_and_llm_arguments_include_seeds_and_no_mm():
    script = _load_script()
    params = script.sampling_params_kwargs()
    assert params["skip_special_tokens"] is False and params["max_tokens"] == 32768
    kwargs = script.llm_kwargs("/model", 40000)
    assert kwargs["max_model_len"] == 40000
    assert kwargs["seed"] == 0
    assert kwargs["trust_remote_code"] is False
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "audio": 0}


def test_build_app_declares_resources_and_pins(monkeypatch):
    script = _load_script()
    captured = {}

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
            captured["pip"] = args
            return self

        def add_local_python_source(self, *modules):
            captured["local"] = modules
            return self

        def env(self, values):
            return self

    class FakeApp:
        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            captured["function"] = kwargs
            return lambda fn: FakeFunction()

    fake = SimpleNamespace(App=FakeApp, Image=FakeImage)
    app, fn = script._build_modal_app(fake, timeout_s=600, startup_timeout_s=60, scaledown_window_s=10, cpu=4, memory_gib=16)
    assert captured["function"]["max_containers"] == 1
    assert "nonpreemptible" not in captured["function"]
    assert captured["function"]["timeout"] == 600
    assert captured["function"]["startup_timeout"] == 60
    assert captured["function"]["scaledown_window"] == 10
    assert captured["function"]["cpu"] == (4.0, 4.0)
    assert captured["function"]["memory"] == (16384, 16384)
    assert captured["local"] == ("gemmaloop",)
    assert script.MODAL_VERSION in script.DELETE_COMMAND


def test_dry_run_does_not_import_modal(tmp_path, monkeypatch):
    script = _load_script()
    monkeypatch.setattr(script, "_import_modal", lambda: (_ for _ in ()).throw(AssertionError("modal imported")))
    assert script.main(["--dry-run", "--pool-dir", str(_pool(tmp_path, 1)), "--max-usd", "12"]) == 0


def test_budget_refusal_before_contact_and_hard_ceiling(tmp_path):
    script = _load_script()
    fake, created, _ = _fake_modal(FakeVolume())
    with pytest.raises(ValueError, match="max-usd"):
        script.run_mode(**_run(tmp_path, fake, FakeFunction(), max_usd=0.00001))
    with pytest.raises(ValueError, match="max_usd"):
        script.run_mode(**_run(tmp_path, fake, FakeFunction(), max_usd=30.01))
    assert created == []


def test_yes_gate_and_lock_are_mandatory(tmp_path, monkeypatch, capsys):
    script = _load_script()
    fake, created, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    assert script.run_mode(**_run(tmp_path, fake, fn, yes=False)) == 2
    assert created == [] and fn.calls == 0
    assert not (tmp_path / "spend.json").exists()

    monkeypatch.setattr(script, "require_spend_lock", lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(75)))
    with pytest.raises(SystemExit) as excinfo:
        script.run_mode(**_run(tmp_path, fake, fn))
    assert excinfo.value.code == 75
    assert fn.calls == 0


def test_pre_volume_guard_stops_when_ledger_is_at_cap(tmp_path):
    script = _load_script()
    fake, created, _ = _fake_modal(FakeVolume())
    ledger = tmp_path / "spend.json"
    script.SpendLedger(ledger).reserve(shard=0, usd=100.0)
    result = script.run_mode(**_run(tmp_path, fake, FakeFunction(), ledger_path=ledger))
    assert result == 0
    assert created == []


def test_worst_case_guard_stops_sequentially(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    ticks = iter([0, 4000, 8000])
    assert script.run_mode(**_run(tmp_path, fake, fn, clock=lambda: next(ticks))) == 0
    summary = _summary(capsys)
    assert fn.calls == 1
    assert summary["remote_calls"] == 1
    assert summary["stopped_reason"] == "spending_cap"
    expected = script.worst_case_usd(
        "A100-80GB", timeout_s=script.DEFAULT_TIMEOUT_SECONDS,
        startup_timeout_s=script.DEFAULT_STARTUP_TIMEOUT_SECONDS,
        scaledown_window_s=script.DEFAULT_SCALEDOWN_SECONDS,
        cpu_cores=int(script.DEFAULT_CPU), memory_gib=script.DEFAULT_MEMORY_GIB,
    )["usd"] + script._preflight_cost_usd(
        script.DEFAULT_STARTUP_TIMEOUT_SECONDS
        + script.PREFLIGHT_TIMEOUT_SECONDS
        + script.REMOTE_CALL_MARGIN_SECONDS
    )
    assert summary["worst_case_usd"] == pytest.approx(expected)


def test_with_options_carries_max_containers_and_resources(tmp_path):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    script.run_mode(**_run(tmp_path, fake, fn))
    assert fn.options["max_containers"] == 1
    assert fn.options["cpu"] == (8.0, 8.0)
    assert fn.options["memory"] == (48 * 1024, 48 * 1024)


def test_reservation_is_on_disk_before_remote_and_survives_restart(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    ledger = tmp_path / "spend.json"
    first = FakeFunction(ledger_probe=ledger)
    ticks = iter([0, 4000])
    script.run_mode(**_run(tmp_path, fake, first, clock=lambda: next(ticks)))
    assert first.calls == 1
    assert '"kind": "reservation"' in first.ledger_seen[0]
    second = FakeFunction()
    script.run_mode(**_run(tmp_path, fake, second))
    summary = _summary(capsys)
    assert second.calls == 0
    assert summary["stopped_reason"] == "spending_cap"


def test_failed_shard_books_bounded_upper_bound(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    fn.get_error = RuntimeError("boom")
    assert script.run_mode(**_run(tmp_path, fake, fn)) == 1
    entries = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]
    actuals = [e for e in entries if e["kind"] == "actual"]
    reservations = [e for e in entries if e["kind"] == "reservation"]
    assert actuals and reservations
    assert actuals[0]["usd"] < reservations[0]["usd"]
    assert actuals[0]["metadata"]["booking_basis"] == "elapsed_upper_bound"


def _download_manifest(tmp_path, run_id="run1", expected=2):
    out = tmp_path / "samples"
    out.mkdir(exist_ok=True)
    (out / "launch_manifest.json").write_text(
        json.dumps(
            {
                "planned_shards": expected,
                "run_id": run_id,
                "run_dir": f"shards/{run_id}",
                "stopped_reason": None,
            }
        ),
        encoding="utf-8",
    )
    return out


def _volume_files(run_id, shards=2):
    files = {}
    for index in range(shards):
        files[f"shards/{run_id}/shard_{index:05d}.jsonl"] = f'{{"x": {index}}}\n'.encode()
        files[f"shards/{run_id}/shard_{index:05d}.jsonl.committed"] = b"committed\n"
    return files


def test_download_and_cleanup_ignore_budget_and_shard_size(tmp_path):
    script = _load_script()
    out = _download_manifest(tmp_path, expected=2)
    volume = FakeVolume(_volume_files("run1", 2))
    fake, _, manager = _fake_modal(volume)
    assert script.run_mode(mode="download", pool_dir=tmp_path / "pool", output_dir=out, gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0.0, shard_size=999, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake) == 0
    assert script.run_mode(mode="cleanup", pool_dir=tmp_path / "pool", output_dir=out, gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0.0, shard_size=999, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake) == 0
    assert manager.deleted == [script.VOLUME_NAME]


def test_partial_receipt_and_cleanup_refusal(tmp_path, capsys):
    script = _load_script()
    out = _download_manifest(tmp_path, expected=2)
    volume = FakeVolume(_volume_files("run1", 1))
    fake, _, manager = _fake_modal(volume)
    common = dict(pool_dir=tmp_path / "pool", output_dir=out, gpu="A100-80GB", gpu_id="A100-80GB", max_usd=12, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake)
    assert script.run_mode(mode="download", **common) == 0
    receipt = json.loads((out / "download_receipt.json").read_text())
    assert receipt["partial"] is True and receipt["missing"] == ["shard_00001.jsonl"]
    assert script.run_mode(mode="cleanup", **common) == 2
    assert script.DELETE_COMMAND in capsys.readouterr().err
    assert manager.deleted == []
    assert script.run_mode(mode="cleanup", **common, allow_partial=True) == 0


def test_cleanup_refuses_a_volume_shard_not_in_receipt(tmp_path, capsys):
    script = _load_script()
    out = _download_manifest(tmp_path, expected=1)
    volume = FakeVolume(_volume_files("run1", 1))
    fake, _, manager = _fake_modal(volume)
    common = dict(pool_dir=tmp_path / "pool", output_dir=out, gpu="A100-80GB", gpu_id="A100-80GB", max_usd=12, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake)
    assert script.run_mode(mode="download", **common) == 0
    volume.files["shards/run1/shard_00002.jsonl"] = b"new\n"
    volume.files["shards/run1/shard_00002.jsonl.committed"] = b"committed\n"
    assert script.run_mode(mode="cleanup", **common) == 2
    assert "not in the local receipt" in capsys.readouterr().err
    assert manager.deleted == []


def test_empty_volume_cleanup_works_without_receipt(tmp_path):
    script = _load_script()
    volume = FakeVolume({})
    fake, _, manager = _fake_modal(volume)
    assert script.run_mode(mode="cleanup", pool_dir=tmp_path, output_dir=tmp_path / "samples", gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0.0, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake) == 0
    assert manager.deleted == [script.VOLUME_NAME]


def _install_fake_sampling(monkeypatch):
    captured = {"engines": 0}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["engines"] += 1
            captured["llm"] = kwargs

        def generate(self, prompts, params):
            captured["params"] = params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="<|channel>thought\nplan<channel|>Hola", finish_reason="stop", token_ids=[1, 2, 3])]) for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: "/model"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    return captured


def _clear_engine_cache(script):
    script._LLM_CACHE.clear()


def test_sample_shard_writes_run_dir_and_caches_engine(tmp_path, monkeypatch):
    script = _load_script()
    script._LLM_CACHE.clear()
    captured = _install_fake_sampling(monkeypatch)
    volume = FakeVolume()
    base = {
        "run_id": "run1",
        "run_dir": str(tmp_path / "shards" / "run1"),
        "problems": [{"problem_id": "p:0", "prompt": "one two three"}],
        "samples_per_prompt": 4,
        "engine_max_model_len": 40000,
        "model_cache_dir": str(tmp_path / "cache"),
        "_volume": volume,
    }
    first = script.sample_shard({**base, "shard_index": 0})
    second = script.sample_shard({**base, "shard_index": 1})
    assert first["samples"] == 1 and second["samples"] == 1
    assert captured["engines"] == 1
    assert captured["params"][0].kwargs["skip_special_tokens"] is False
    assert volume.commits == 2
    record = json.loads((tmp_path / "shards" / "run1" / "shard_00000.jsonl").read_text())
    assert record["run_id"] == "run1" and record["reasoning"] == "plan"
    assert record["content"] == "Hola"


def test_sample_shard_missing_volume_removes_both_files(tmp_path, monkeypatch):
    script = _load_script()
    script._LLM_CACHE.clear()
    _install_fake_sampling(monkeypatch)
    fake, _, _ = _fake_modal(None)
    monkeypatch.setattr(script, "modal", fake)
    with pytest.raises(FileNotFoundError):
        script.sample_shard({"run_id": "r", "run_dir": str(tmp_path / "shards" / "r"), "shard_index": 0, "problems": [{"problem_id": "p", "prompt": "x"}], "samples_per_prompt": 1})
    assert not (tmp_path / "shards" / "r" / "shard_00000.jsonl").exists()
    assert not (tmp_path / "shards" / "r" / "shard_00000.jsonl.committed").exists()


def test_sample_shard_commit_failure_removes_marker(tmp_path, monkeypatch):
    script = _load_script()
    script._LLM_CACHE.clear()
    _install_fake_sampling(monkeypatch)
    with pytest.raises(RuntimeError, match="commit failed"):
        script.sample_shard({"run_id": "r", "run_dir": str(tmp_path / "shards" / "r"), "shard_index": 0, "problems": [{"problem_id": "p", "prompt": "x"}], "samples_per_prompt": 1, "_volume": FakeVolume(fail_commit=True)})
    assert not (tmp_path / "shards" / "r" / "shard_00000.jsonl").exists()
    assert not (tmp_path / "shards" / "r" / "shard_00000.jsonl.committed").exists()


def test_paid_run_refuses_smoke_pool_and_wrong_frozen_sha(tmp_path):
    script = _load_script()
    fake, created, _ = _fake_modal(FakeVolume())
    smoke_pool = _pool(tmp_path, 1, smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        script.run_mode(**_run(tmp_path, fake, FakeFunction(), pool_dir=smoke_pool))
    good_pool = _pool(tmp_path / "good", 1)
    with pytest.raises(ValueError, match="frozen"):
        script.run_mode(**_run(tmp_path, fake, FakeFunction(), pool_dir=good_pool, frozen_manifest_sha256="0" * 64))
    assert created == []


class _FailingApp:
    def run(self):
        raise RuntimeError("app startup failed")


class _RecordingApp:
    def __init__(self, *, fail_run: bool = False):
        self.fail_run = fail_run
        self.stops = 0
        self.entries = 0

    def run(self):
        if self.fail_run:
            raise RuntimeError("app startup failed")
        return self

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, *_exc):
        return False

    def stop(self):
        self.stops += 1


def test_app_startup_failure_stops_and_ledger_records_it(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    result = script.run_mode(**_run(tmp_path, fake, fn, app=_FailingApp()))
    assert result == 1
    entries = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]
    assert any(entry["kind"] == "reservation" for entry in entries)
    assert any(entry["kind"] == "actual" for entry in entries)
    assert fn.calls == 0


def test_app_startup_failure_calls_app_stop_and_books_elapsed(tmp_path):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    app = _RecordingApp(fail_run=True)
    times = iter([100.0, 108.5, 108.5])
    result = script.run_mode(
        **_run(
            tmp_path,
            fake,
            fn,
            app=app,
            wall_clock=lambda: next(times),
        )
    )
    assert result == 1
    assert app.stops == 1
    assert fn.calls == 0
    entries = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]
    actual = [entry for entry in entries if entry["kind"] == "actual"][-1]
    reservation = [entry for entry in entries if entry["kind"] == "reservation"][-1]
    assert actual["usd"] < reservation["usd"]
    assert actual["metadata"]["elapsed_s"] == pytest.approx(8.5)
    assert actual["metadata"]["booking_basis"] == "elapsed_upper_bound"


def test_first_remote_failure_stops_after_one_attempt_and_calls_app_stop(tmp_path):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    fn.get_error = RuntimeError("startup boom")
    app = _RecordingApp()
    result = script.run_mode(**_run(tmp_path, fake, fn, app=app))
    assert result == 1
    assert fn.calls == 1
    assert app.stops == 1
    entries = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]
    actual = [entry for entry in entries if entry["kind"] == "actual"][-1]
    assert actual["metadata"]["elapsed_s"] >= 0.0


def test_failing_preflight_makes_zero_gpu_calls(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    gpu = FakeFunction()
    preflight = FakeFunction()
    preflight.spawn_error = RuntimeError("no module named rapidfuzz")
    app = _RecordingApp()
    result = script.run_mode(
        **_run(tmp_path, fake, gpu, app=app, preflight_fn=preflight)
    )
    assert result == 1
    assert preflight.calls == 1
    assert gpu.calls == 0
    assert app.stops == 1
    entries = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["entries"]
    actual = [entry for entry in entries if entry["kind"] == "actual"][-1]
    assert actual["metadata"]["elapsed_s"] >= 0.0
    assert "preflight" in capsys.readouterr().err


def test_build_app_registers_a_cpu_only_preflight_function():
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
            return lambda fn: FakeFunction()

    script._build_modal_app(SimpleNamespace(App=FakeApp, Image=FakeImage))
    preflight = [config for config in captured if "gpu" not in config]
    gpu = [config for config in captured if "gpu" in config]
    assert len(preflight) == 1 and len(gpu) == 1
    config = preflight[0]
    assert config["cpu"] == (1.0, 1.0)
    assert config["memory"] == (16384, 16384)
    assert config["timeout"] == 120
    assert config["scaledown_window"] == 2
    assert config["max_containers"] == 1



def test_build_app_rejects_non_int_timeouts(monkeypatch):
    script = _load_script()

    class StrictImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        @classmethod
        def from_registry(cls, tag, **kwargs):
            return cls()

        def apt_install(self, *args): return self
        def pip_install(self, *args): return self
        def add_local_python_source(self, *args): return self
        def env(self, values): return self

    class StrictApp:
        def __init__(self, name): pass
        def function(self, **kwargs):
            for key in ("timeout", "startup_timeout", "scaledown_window"):
                assert type(kwargs[key]) is int, (key, kwargs[key])
            raise RuntimeError("strict")

    fake = SimpleNamespace(App=StrictApp, Image=StrictImage)
    with pytest.raises(RuntimeError, match="strict"):
        script._build_modal_app(fake, timeout_s=600.0, startup_timeout_s=60.0, scaledown_window_s=10.0)


def test_cleanup_takes_the_spend_lock(tmp_path, monkeypatch):
    script = _load_script()
    volume = FakeVolume({})
    fake, _, manager = _fake_modal(volume)
    monkeypatch.setattr(script, "require_spend_lock", lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(75)))
    with pytest.raises(SystemExit) as excinfo:
        script.run_mode(mode="cleanup", pool_dir=tmp_path, output_dir=tmp_path / "samples", gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake)
    assert excinfo.value.code == 75
    assert manager.deleted == []


def test_cleanup_listing_error_refuses_and_notfound_is_empty(tmp_path, capsys):
    script = _load_script()

    class BadVolume(FakeVolume):
        def __init__(self, exc): super().__init__({}); self.exc = exc
        def listdir(self, path, recursive=False): raise self.exc

    fake, _, manager = _fake_modal(BadVolume(RuntimeError("grpc")))
    assert script.run_mode(mode="cleanup", pool_dir=tmp_path, output_dir=tmp_path / "samples", gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake) == 2
    assert manager.deleted == []
    assert script.DELETE_COMMAND in capsys.readouterr().err

    class NotFoundError(Exception):
        pass
    fake2, _, manager2 = _fake_modal(BadVolume(NotFoundError("missing")))
    assert script.run_mode(mode="cleanup", pool_dir=tmp_path, output_dir=tmp_path / "samples", gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake2) == 0
    assert manager2.deleted == [script.VOLUME_NAME]


def test_missing_ledger_is_refused(tmp_path):
    script = _load_script()
    fake, created, _ = _fake_modal(FakeVolume())
    result = script.run_mode(**_run(tmp_path, fake, FakeFunction(), init_ledger=False, ledger_path=tmp_path / "absent.json"))
    assert result == 2
    assert created == []


def test_reminder_prints_on_refusal_and_success(tmp_path, capsys):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    assert script.run_mode(**_run(tmp_path, fake, FakeFunction(), yes=False)) == 2
    assert script.DELETE_COMMAND in capsys.readouterr().err
    ledger = tmp_path / "spend.json"
    script.SpendLedger(ledger).reserve(shard=0, usd=100.0)
    assert script.run_mode(**_run(tmp_path, fake, FakeFunction(), ledger_path=ledger)) == 0
    assert script.DELETE_COMMAND in capsys.readouterr().err


def test_resumed_partial_uses_original_planned_total(tmp_path):
    script = _load_script()
    out = tmp_path / "samples"
    out.mkdir()
    (out / "launch_manifest.json").write_text(json.dumps({"planned_shards": 4, "run_id": "r", "run_dir": "shards/r", "already_committed": 2, "remaining_shards": 2}), encoding="utf-8")
    volume = FakeVolume({f"shards/r/shard_{i:05d}.jsonl": b"x\n" for i in range(2)} | {f"shards/r/shard_{i:05d}.jsonl.committed": b"c\n" for i in range(2)})
    fake, _, _ = _fake_modal(volume)
    assert script.run_mode(mode="download", pool_dir=tmp_path, output_dir=out, gpu="A100-80GB", gpu_id="A100-80GB", max_usd=0, shard_size=1, avg_generated_tokens=1, samples_per_prompt=1, yes=False, modal_module=fake) == 0
    receipt = json.loads((out / "download_receipt.json").read_text())
    assert receipt["expected_shards"] == 4
    assert receipt["partial"] is True
    assert receipt["missing"] == ["shard_00002.jsonl", "shard_00003.jsonl"]


def test_run_id_namespaces_volume_path_and_depends_on_pool_sha(tmp_path):
    script = _load_script()
    assert script.compute_run_id("a", shard_size=1, samples_per_prompt=4) != script.compute_run_id("b", shard_size=1, samples_per_prompt=4)
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    script.run_mode(**_run(tmp_path, fake, fn))
    manifest = json.loads((tmp_path / "samples" / "launch_manifest.json").read_text())
    assert manifest["run_id"]
    # payload namespacing is tested by the fake volume path in sample_shard tests;
    # here assert the run manifest and run id agree.
    assert manifest["run_dir"] == f"shards/{manifest['run_id']}"


def test_sample_shard_imports_modal_inside_the_container(tmp_path, monkeypatch):
    import importlib.util

    _install_fake_sampling(monkeypatch)
    volume = FakeVolume()
    created = []

    class VolumeClass:
        @classmethod
        def from_name(cls, name, create_if_missing=False):
            created.append((name, create_if_missing))
            return volume

    fake_modal = SimpleNamespace(Volume=VolumeClass)
    spec = importlib.util.spec_from_file_location("modal_stage0_container_fresh", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    spec.loader.exec_module(module)
    assert module.modal is None
    result = module.sample_shard({"run_id": "r", "run_dir": str(tmp_path / "shards" / "r"), "shard_index": 0, "problems": [{"problem_id": "p", "prompt": "x"}], "samples_per_prompt": 1, "engine_max_model_len": 40000})
    assert result["samples"] == 1
    assert created == [(module.VOLUME_NAME, False)]
    assert volume.commits == 1
    assert (tmp_path / "shards" / "r" / "shard_00000.jsonl").exists()


def test_t4_is_rejected():
    script = _load_script()
    with pytest.raises(SystemExit):
        script._parse_args(["--gpu", "T4"])
    assert "T4" not in script.MODAL_GPU_IDS
    from gemmaloop.stage0.modal_plan import GPU_HOURLY_USD, GPU_TOKENS_PER_SEC

    assert "T4" not in GPU_HOURLY_USD and "T4" not in GPU_TOKENS_PER_SEC


def test_planned_shards_is_total_and_max_shards_is_honoured(tmp_path):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    assert script.run_mode(**_run(tmp_path, fake, fn, max_shards=1)) == 0
    manifest = json.loads((tmp_path / "samples" / "launch_manifest.json").read_text())
    assert manifest["planned_shards"] == 2
    assert manifest["remaining_shards"] == 1
    assert manifest["already_committed"] == 0
    assert fn.calls == 1


def test_with_options_timeout_is_int(tmp_path):
    script = _load_script()
    fake, _, _ = _fake_modal(FakeVolume())
    fn = FakeFunction()
    script.run_mode(**_run(tmp_path, fake, fn, timeout_s=1800.0))
    assert type(fn.options["timeout"]) is int
    assert fn.options["timeout"] == 1800


def test_non_partial_receipt_count_check(tmp_path):
    script = _load_script()
    out = tmp_path / "samples"
    out.mkdir()
    name = "shard_00000.jsonl"
    data = b"x\n"
    (out / name).write_bytes(data)
    receipt = {
        "expected_shards": 2,
        "present_shards": 1,
        "missing": [],
        "partial": False,
        "shards": [{"name": name, "path": f"shards/r/{name}", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    path = out / "download_receipt.json"
    path.write_text(json.dumps(receipt))
    ok, reason = script._verify_receipt(out, path, allow_partial=False)
    assert not ok and "expected 2" in reason


def test_paid_run_accepts_a_part1_pool_without_the_creative_briefs(tmp_path):
    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

    pool = _pool(tmp_path, 1)
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    manifest["creative_briefs"] = PART1_CREATIVE_BRIEFS
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = _load_script()
    accepted = script.validate_pool_manifest(pool)
    assert accepted["creative_briefs"] == PART1_CREATIVE_BRIEFS
    assert accepted["limits"]["allow_missing_briefs"] is False

    # The smoke flag is still refused, even with the Part-1 note.
    smoke = _pool(tmp_path / "smoke", 1, smoke=True)
    smoke_manifest = json.loads((smoke / "manifest.json").read_text(encoding="utf-8"))
    smoke_manifest["creative_briefs"] = PART1_CREATIVE_BRIEFS
    smoke_manifest.pop("manifest_sha256", None)
    smoke_manifest["manifest_sha256"] = _manifest_sha(smoke_manifest)
    (smoke / "manifest.json").write_text(json.dumps(smoke_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke"):
        script.validate_pool_manifest(smoke)
