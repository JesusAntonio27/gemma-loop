"""Stage-0-lite on Modal vLLM: fake-only tests (never contacts Modal)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaloop.prereg import default_prereg_path, load_prereg
from gemmaloop.stage0 import local
from gemmaloop.stage0 import modal_lite
from gemmaloop.stage0.evals import EVAL_SPECS
from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"
FIXED_NOW = lambda: datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "modal_stage0_lite_under_test", SCRIPT
    )
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


def _problem(problem_id: str, bucket: int, kind: str = "math") -> dict:
    return {
        "problem_id": problem_id,
        "prompt": f"prompt for {problem_id}",
        "kind": kind,
        "answer": "42",
        "tests": [],
        "source": "omr" if kind == "math" else "taco",
        "split": "train" if bucket <= 49 else "pilot_dev",
        "pilot_subsplit": None,
        "bucket": bucket,
    }


def _pool_rows() -> list[dict]:
    rows = [_problem(f"train:p{index:03d}", index % 50) for index in range(80)]
    rows += [_problem(f"p1:p{index:02d}", 59) for index in range(6)]
    rows += [_problem(f"tfit:p{index:02d}", 57) for index in range(4)]
    rows += [_problem(f"sel:p{index:02d}", 54) for index in range(4)]
    return rows


def _write_pool(tmp_path: Path, *, full_manifest: bool = False) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    rows = _pool_rows()
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (pool / "problems.jsonl").write_bytes(text.encode("utf-8"))
    manifest = {
        "stage": "stage0_pool",
        "creative_briefs": PART1_CREATIVE_BRIEFS if full_manifest else None,
        "artifacts": {
            "problems.jsonl": {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "rows": len(rows),
            }
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
    if full_manifest:
        manifest["decontamination"] = {
            "status": "ok",
            "targets": {spec.name: {"status": "ok"} for spec in EVAL_SPECS},
        }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pool


def _fake_prompt_tokens(problem) -> int:
    digest = hashlib.sha256(str(problem["problem_id"]).encode()).hexdigest()
    return 100 + int(digest[:2], 16) % 40


def _fake_evaluate_filter(record, problem):
    return local.SampleVerdict(
        kept=True,
        drop_reason=None,
        rendered_tokens=int(record.get("prompt_tokens", 0))
        + int(record.get("tokens", 0)),
        reasoning_share=0.9,
        reasoning_tokens=40,
        answer_tokens=10,
        assistant_tokens=100,
    )


def _fake_evaluate_full(record, problem):
    verdict = _fake_evaluate_filter(record, problem)
    verdict.correct = True
    verdict.grade = {"decided_by": "normaliser", "timed_out": False}
    return verdict


class FakeEntry:
    def __init__(self, path: str, size: int | None = None):
        self.path = path
        self.size = size


class FakeVolume:
    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = files or {}
        self.commits = 0

    def commit(self):
        self.commits += 1

    def listdir(self, path: str, recursive: bool = False):
        prefix = str(path).strip("/")
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
        data = self.files[str(path).lstrip("/")]
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
    volume = volume if volume is not None else FakeVolume()

    class VolumeClass:
        @classmethod
        def from_name(cls, name, create_if_missing=False):
            created.append((name, create_if_missing))
            if volume is None:
                raise FileNotFoundError("volume missing")
            return volume

        objects = manager

    return SimpleNamespace(Volume=VolumeClass, Dict=FakeDictManager()), created, manager


def _write_heartbeat(payload, container_id="container-1", sequence=1):
    heartbeat = payload.get("_heartbeat")
    key = payload.get("heartbeat_key")
    if heartbeat is not None and key is not None:
        heartbeat[key] = {
            "container_id": container_id,
            "sequence": sequence,
            "phase": "entered",
        }


class FakeCall:
    def __init__(self, result, owner=None):
        self.result = result
        self.owner = owner
        self.gets = 0
        self.cancelled = 0
        self.released = False

    def get(self, timeout=None):
        assert timeout is not None
        self.gets += 1
        if self.owner is not None and not self.released:
            self.owner.active -= 1
            self.released = True
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled += 1


def _record_for(request: dict, payload: dict, shard_index: int, cap: int) -> dict:
    completion = "<|channel>thought\n" + "t " * 40 + "<channel|>answer"
    return {
        "problem_id": str(request["problem_id"]),
        "sample_index": int(request["sample_index"]),
        "request_id": str(request["request_id"]),
        "prompt": str(request["prompt"]),
        "completion": completion,
        "reasoning": "t " * 40,
        "content": "answer",
        "finish_reason": "stop",
        "thought_closed": True,
        "tokens": 50,
        "prompt_tokens": int(request["prompt_tokens"]),
        "max_new_tokens": int(request["max_new_tokens"]),
        "engine_cap": int(cap),
        "engine": "modal-vllm",
        "engine_version": "vllm==0.30.0",
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "rev",
        "run_id": str(payload["run_id"]),
        "pool_manifest_sha256": str(payload["pool_manifest_sha256"]),
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "samples_per_prompt": 4,
        "max_model_len": 8192,
        "max_prompt_tokens": int(request["prompt_tokens"]),
        "skip_special_tokens": False,
        "gpu": str(payload.get("gpu", "")),
        "shard_index": int(shard_index),
        "seed": int(request["seed"]),
    }


class FakeLiteFunction:
    def __init__(self, volume: FakeVolume):
        self.volume = volume
        self.options: dict = {}
        self.calls = 0
        self.payloads: list[dict] = []
        self.sent_request_ids: list[str] = []
        self.sent_shards: list[int] = []
        self.active = 0
        self.max_active = 0
        self.get_error: BaseException | None = None
        self.spawn_error: BaseException | None = None
        self.tamper_sha = False
        self.skip_shards: set[int] = set()
        self.container_id = "container-1"

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def get_current_stats(self):
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)

    def spawn(self, payload):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        _write_heartbeat(payload, self.container_id)
        if self.spawn_error is not None:
            self.active -= 1
            raise self.spawn_error
        self.payloads.append(payload)
        run_dir_rel = str(payload["run_dir_rel"])
        results = []
        for shard in payload["shards"]:
            shard_index = int(shard["shard_index"])
            self.sent_shards.append(shard_index)
            requests = list(shard["requests"])
            self.sent_request_ids.extend(str(r["request_id"]) for r in requests)
            if shard_index in self.skip_shards:
                continue
            cap = max(int(r["max_new_tokens"]) for r in requests)
            rows = [
                _record_for(request, payload, shard_index, cap)
                for request in requests
            ]
            data = (
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
            ).encode("utf-8")
            name = f"{run_dir_rel}/shard_{shard_index:05d}.jsonl"
            self.volume.files[name] = data
            self.volume.files[name + ".committed"] = b"committed\n"
            sha = hashlib.sha256(data).hexdigest()
            if self.tamper_sha:
                sha = "0" * 64
            results.append(
                {
                    "shard": shard_index,
                    "skipped": False,
                    "committed": True,
                    "samples": len(rows),
                    "rows": len(rows),
                    "path": name,
                    "sha256": sha,
                }
            )
        self.volume.commit()
        call = FakeCall(
            {"shards": results, "generated_tokens": len(self.sent_request_ids) * 50},
            owner=self,
        )
        if self.get_error is not None:
            def _raise(timeout=None):
                call.gets += 1
                if not call.released:
                    self.active -= 1
                    call.released = True
                raise self.get_error

            call.get = _raise  # type: ignore[method-assign]
        return call


class FakePreflight:
    def __init__(self):
        self.calls = 0

    def spawn(self, payload):
        self.calls += 1
        _write_heartbeat(payload, "container-1")
        result = {
            "ok": True,
            "versions": {},
            "heartbeat": {
                "container_id": "container-1",
                "delivered": 2,
                "background_delivered": 1,
                "failed_writes": 0,
            },
        }
        return FakeCall(result)

    def get_current_stats(self):
        return SimpleNamespace(num_running_inputs=0, num_total_runners=0)


def _run(tmp_path: Path, fake, fn, **overrides):
    pool = overrides.pop("pool_dir", None) or _write_pool(tmp_path)
    kwargs = dict(
        mode="sample-lite",
        pool_dir=pool,
        output_dir=tmp_path / "modal_lite",
        gpu="A100-80GB",
        gpu_id="A100-80GB",
        max_usd=30.0,
        shard_size=2,
        avg_generated_tokens=3200,
        samples_per_prompt=4,
        yes=True,
        fn=fn,
        preflight_fn=FakePreflight(),
        modal_module=fake,
        month_now=FIXED_NOW,
        require_frozen_pool=False,
        prereg_path=default_prereg_path(),
        seed=0,
        lite_call_size=16,
        max_train_tokens=100,
        allow_math_fallback=True,
        prompt_tokens_for=_fake_prompt_tokens,
        evaluate_filter=_fake_evaluate_filter,
        evaluate_full=_fake_evaluate_full,
    )
    kwargs.update(overrides)
    return kwargs


def _load_make_p1lite():
    path = ROOT / "scripts" / "make_p1lite_corpus.py"
    spec = importlib.util.spec_from_file_location("make_p1lite_lite_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Plan identity
# ---------------------------------------------------------------------------
def test_plan_identity_with_the_local_sampler(tmp_path: Path):
    problems = _pool_rows()
    prereg = load_prereg(default_prereg_path())
    targets = local.targets_from_prereg(prereg)
    plan = modal_lite.build_lite_plan(
        problems,
        targets,
        shard_size=2,
        samples_per_prompt=4,
        seed=0,
        prompt_tokens_for=_fake_prompt_tokens,
        call_size=16,
        run_id="r",
    )

    expected = []
    for planned in local.build_shard_plan(
        problems, targets, shard_size=2, samples_per_prompt=4, seed=0
    ):
        requests, _skipped = local.build_requests_from_refs(
            planned.requests,
            prompt_tokens_for=_fake_prompt_tokens,
            max_length=local.REGISTERED_MAX_LENGTH,
        )
        for request in requests:
            expected.append(
                (
                    planned.index,
                    planned.target.name,
                    request.request_id,
                    request.problem["problem_id"],
                    request.prompt,
                    request.budget,
                    request.sample_index,
                )
            )

    actual = [
        (
            request.shard_index,
            request.target_name,
            request.request_id,
            request.problem_id,
            request.prompt,
            request.max_new_tokens,
            request.sample_index,
        )
        for shard in plan.shards
        for request in shard.requests
    ]
    assert actual == expected
    assert [shard.index for shard in plan.shards] == list(range(len(plan.shards)))
    assert plan.requests_by_target()["monitor"] == 64
    assert plan.requests_by_target()["pilot_p1"] == 24
    assert plan.requests_by_target()["train"] == 64

    for shard in plan.shards:
        for request in shard.requests:
            assert request.max_new_tokens == local.REGISTERED_MAX_LENGTH - request.prompt_tokens
            assert request.seed == modal_lite.lite_request_seed(0, request.request_id)
            assert request.seed == (
                int.from_bytes(
                    hashlib.sha256(f"0:{request.request_id}".encode()).digest()[:8],
                    "little",
                )
                % (2**63 - 1)
            )
    assert all(
        call.request_count <= modal_lite.MAX_TRAIN_CALL_SIZE
        for call in plan.train_calls
    )
    assert len(plan.other_calls) == 1
    assert plan.other_calls[0].request_count == plan.other_requests


# The real request whose full unsigned 64-bit seed killed the approved
# 2026-11-05 23:26 lite run in vLLM (OverflowError: "Python int too large to
# convert to C long").  Its raw seed is 9368141884359661058 > 2**63 - 1.
REAL_OVERFLOW_REQUEST_ID = "chat:1d5349e7151223614b67f11b:1"
REAL_OVERFLOW_RAW_SEED = 9_368_141_884_359_661_058


def test_lite_request_seeds_fit_signed_int64():
    problems = _pool_rows()
    targets = local.targets_from_prereg(load_prereg(default_prereg_path()))
    plan = modal_lite.build_lite_plan(
        problems,
        targets,
        shard_size=2,
        samples_per_prompt=4,
        seed=0,
        prompt_tokens_for=_fake_prompt_tokens,
        call_size=16,
        run_id="r",
    )

    seeds = [
        request.seed for shard in plan.shards for request in shard.requests
    ]
    assert seeds
    assert all(0 <= int(seed) < 2**63 - 1 for seed in seeds)
    for call in (*plan.other_calls, *plan.train_calls):
        for shard in call.as_payload():
            for request in shard["requests"]:
                assert 0 <= int(request["seed"]) < 2**63 - 1

    # The real failing request id, computed offline from the real pool.
    raw = int.from_bytes(
        hashlib.sha256(f"0:{REAL_OVERFLOW_REQUEST_ID}".encode()).digest()[:8],
        "little",
    )
    assert raw == REAL_OVERFLOW_RAW_SEED
    assert raw > 2**63 - 1
    seed = modal_lite.lite_request_seed(0, REAL_OVERFLOW_REQUEST_ID)
    assert seed == raw % (2**63 - 1)
    assert 0 <= seed < 2**63 - 1


def test_launcher_refuses_out_of_range_seeds_before_modal_contact(
    tmp_path: Path, monkeypatch
):
    script = _load_script()
    volume = FakeVolume()
    fake, created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    monkeypatch.setattr(
        modal_lite, "lite_request_seed", lambda seed, request_id: 2**63
    )
    with pytest.raises(modal_lite.LitePlanError, match="signed 64-bit"):
        script.run_mode(**_run(tmp_path, fake, fn, max_train_tokens=0))
    assert created == []
    assert fn.calls == 0
    assert not (tmp_path / "modal_lite").exists()


def test_plan_refuses_to_split_a_shard_across_calls(tmp_path: Path):
    problems = _pool_rows()
    targets = local.targets_from_prereg(load_prereg(default_prereg_path()))
    with pytest.raises(modal_lite.LitePlanError, match="call size"):
        modal_lite.build_lite_plan(
            problems,
            targets,
            shard_size=2,
            samples_per_prompt=4,
            prompt_tokens_for=_fake_prompt_tokens,
            call_size=4,  # a 2-prompt train shard has 8 requests
        )


def test_launcher_default_call_size_matches_the_planner():
    script = _load_script()
    assert script.DEFAULT_LITE_CALL_SIZE == modal_lite.DEFAULT_CALL_SIZE == 1400
    assert modal_lite.MAX_TRAIN_CALL_SIZE == 1400
    assert modal_lite.MAX_CALL_SIZE == 2000


def test_estimate_uses_the_measured_rate_and_the_cap():
    problems = _pool_rows()
    targets = local.targets_from_prereg(load_prereg(default_prereg_path()))
    plan = modal_lite.build_lite_plan(
        problems,
        targets,
        shard_size=2,
        prompt_tokens_for=_fake_prompt_tokens,
        call_size=16,
    )
    estimate = modal_lite.estimate_lite_run(plan)
    assert estimate["expected_calls"] >= 1
    assert estimate["expected_generated_tokens"] > 0
    assert estimate["usd"] == pytest.approx(
        estimate["compute_usd"] + estimate["launch_overhead_usd"]
    )
    rate = estimate["measured_rate"]
    assert rate["usd_per_million_generated_tokens"] == 0.18
    assert rate["launch_overhead_usd_per_call"] == 0.26


# ---------------------------------------------------------------------------
# Runner: early stop, non-train completeness, schema, finalize
# ---------------------------------------------------------------------------
def test_early_stop_and_non_train_completeness(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, created, manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    output = tmp_path / "modal_lite"
    result = script.run_mode(**_run(tmp_path, fake, fn, output_dir=output))
    assert result == 0

    plan = modal_lite.build_lite_plan(
        _pool_rows(),
        local.targets_from_prereg(load_prereg(default_prereg_path())),
        shard_size=2,
        prompt_tokens_for=_fake_prompt_tokens,
        call_size=16,
    )
    train_ids = {
        request.request_id
        for shard in plan.shards
        if shard.target_name == "train"
        for request in shard.requests
    }
    non_train_ids = {
        request.request_id
        for shard in plan.shards
        if shard.target_name != "train"
        for request in shard.requests
    }
    sent = set(fn.sent_request_ids)
    assert sent & train_ids  # some train requests were sent
    assert sent & train_ids != train_ids  # but not all: early stop
    assert non_train_ids <= sent  # every non-train request was sampled
    assert len(fn.sent_request_ids) <= len(non_train_ids) + 16 * len(plan.train_calls)
    # I1/B3: the single non-train call runs first.
    first_payload = fn.payloads[0]
    first_shards = {int(item["shard_index"]) for item in first_payload["shards"]}
    assert first_shards <= {
        shard.index for shard in plan.shards if shard.target_name != "train"
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "modal-vllm"
    assert manifest["shards"]["skipped"]
    assert (output / "run_marker.json").exists()
    assert (output / "monitor" / "manifest.json").exists()
    assert (output / "modal_receipt.json").exists()
    receipt = json.loads((output / "modal_receipt.json").read_text())
    assert receipt["partial"] is True
    assert receipt["run_id"] == manifest["run_id"]
    assert manager.deleted == []  # cleanup is a separate explicit command

    # I2: the heal corpus is the train prefix up to the goal, not the whole
    # last call.  Each fake record is worth 100 accepted tokens and the goal is
    # 100, so only the first train shard survives.
    first_train_shard = plan.shard_by_index(0)
    assert first_train_shard.target_name == "train"
    heal_rows = local.read_jsonl(output / "heal" / "train.jsonl")
    assert len(heal_rows) == len(first_train_shard.requests)
    assert manifest["heal"]["train_goal"] == 100
    assert manifest["heal"]["train_goal_reached"] is True
    assert manifest["heal"]["train_shards_kept"] == [0]
    assert manifest["heal"]["excluded_train_records"] > 0
    assert manifest["heal"]["excluded_train_heal_rows"] > 0


def test_records_are_local_schema_and_feed_finalize_and_p1lite(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    output = tmp_path / "modal_lite"
    pool = _write_pool(tmp_path, full_manifest=True)
    assert (
        script.run_mode(
            **_run(
                tmp_path,
                fake,
                fn,
                output_dir=output,
                pool_dir=pool,
                require_frozen_pool=True,
            )
        )
        == 0
    )

    ok, reason = local.verify_manifest(output / "manifest.json")
    assert ok, reason

    records = []
    for shard in sorted((output / "shards").glob("shard_*.jsonl")):
        rows = local.read_jsonl(shard)
        assert rows
        records.extend(rows)
    assert records
    local_keys = set(
        local.build_sample_record(
            problem={"problem_id": "p", "prompt": "x"},
            sample_index=0,
            request=None,
            output=None,
            run_id="r",
            pool_manifest_sha256="s",
            shard_index=0,
            prompt_tokens=1,
            budget=2,
            engine_cap=2,
            model_revision="rev",
        )
    )
    for record in records:
        assert local_keys <= set(record)
        assert record["engine"] == "modal-vllm"
        assert record["engine_version"] == "vllm==0.30.0"
        assert record["gpu"] == "A100-80GB"
        assert record["request_id"] == f"{record['problem_id']}:{record['sample_index']}"
        assert record["seed"] == modal_lite.lite_request_seed(
            0, record["request_id"]
        )

    p1lite = _load_make_p1lite()
    committed, run_marker, shards = p1lite.load_committed_shards(output)
    assert committed and run_marker["run_id"] and shards
    pool_info = p1lite.load_pool_manifest(pool)
    assert p1lite.validate_run_pool(run_marker, pool_info) == run_marker[
        "pool_manifest_sha256"
    ]
    problems, _info = p1lite.load_pool_index(pool)
    monitor_ids, monitor_info = p1lite.resolve_monitor_ids(
        output, pool, problems, pool_info, {}
    )
    assert monitor_info["status"] == "present"
    assert len(monitor_ids) >= 64

    # The main manifest binds the same pool hash and the monitor manifest.
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pool"]["manifest_sha256"] == run_marker["pool_manifest_sha256"]
    assert manifest["monitor"]["problem_ids"]
    assert manifest["heal"]["rows"] > 0
    assert manifest["files"]["request_ids.txt"]["rows"] == manifest["requests"]["count"]


def test_one_container_at_a_time_and_options(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    result = script.run_mode(
        **_run(
            tmp_path,
            fake,
            fn,
            max_train_tokens=0,
            lite_call_size=24,
        )
    )
    assert result == 0
    assert fn.calls > 1
    assert fn.max_active == 1
    assert fn.options["max_containers"] == 1
    assert fn.options["timeout"] == int(script.DEFAULT_TIMEOUT_SECONDS)


def test_fail_fast_on_the_first_remote_error(tmp_path: Path, capsys):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    fn.get_error = RuntimeError("boom")
    app = _RecordingApp()
    result = script.run_mode(**_run(tmp_path, fake, fn, app=app))
    assert result == 1
    assert fn.calls == 1
    assert app.stops == 1
    from gemmaloop.stage0.spend import default_ledger_path

    entries = json.loads(default_ledger_path().read_text(encoding="utf-8"))["entries"]
    actual = [entry for entry in entries if entry["kind"] == "actual"][-1]
    reservation = [entry for entry in entries if entry["kind"] == "reservation"][-1]
    assert actual["usd"] < reservation["usd"]
    assert "boom" in capsys.readouterr().err


def test_preflight_failure_makes_zero_gpu_calls(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    preflight = FakePreflight()
    preflight.spawn_error = RuntimeError("no module named rapidfuzz")

    def _spawn(payload):
        preflight.calls += 1
        raise preflight.spawn_error

    preflight.spawn = _spawn  # type: ignore[method-assign]
    result = script.run_mode(**_run(tmp_path, fake, fn, preflight_fn=preflight))
    assert result == 1
    assert fn.calls == 0
    assert preflight.calls == 1


def test_bad_remote_sha_refuses_to_finalize(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    fn.tamper_sha = True
    result = script.run_mode(**_run(tmp_path, fake, fn))
    assert result == 1
    assert not (tmp_path / "modal_lite" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_cap_refusal_before_any_modal_contact(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    with pytest.raises(ValueError, match="max-usd"):
        script.run_mode(**_run(tmp_path, fake, fn, max_usd=0.5))
    assert created == []
    assert fn.calls == 0
    assert not (tmp_path / "spend.json").exists()

    with pytest.raises(ValueError, match="max-usd"):
        script.run_mode(
            **_run(tmp_path, fake, fn, mode="plan-lite", max_usd=0.5)
        )


def test_refuses_the_local_output_dir(tmp_path: Path):
    script = _load_script()
    fake, _created, _manager = _fake_modal(FakeVolume())
    with pytest.raises(ValueError, match="artifacts/stage0/local"):
        script.run_mode(
            **_run(
                tmp_path,
                fake,
                FakeLiteFunction(FakeVolume()),
                output_dir=ROOT / "artifacts" / "stage0" / "local",
                mode="plan-lite",
            )
        )


def test_plan_dry_run_does_not_import_modal(tmp_path: Path, monkeypatch):
    script = _load_script()
    pool = _write_pool(tmp_path, full_manifest=True)
    monkeypatch.setattr(
        script,
        "_import_modal",
        lambda: (_ for _ in ()).throw(AssertionError("modal imported")),
    )
    out = tmp_path / "not-written"
    assert (
        script.main(
            [
                "--mode",
                "sample-lite",
                "--dry-run",
                "--pool-dir",
                str(pool),
                "--output-dir",
                str(out),
                "--max-usd",
                "6",
                "--shard-size",
                "2",
                "--lite-call-size",
                "16",
            ]
        )
        == 0
    )
    assert not out.exists()


def test_cleanup_uses_the_lite_receipt_and_separate_command(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    output = tmp_path / "modal_lite"
    assert script.run_mode(**_run(tmp_path, fake, fn, output_dir=output)) == 0
    assert manager.deleted == []
    receipt = output / "modal_receipt.json"
    assert receipt.exists()
    result = script.run_mode(
        mode="cleanup",
        pool_dir=tmp_path / "pool",
        output_dir=output,
        gpu="A100-80GB",
        gpu_id="A100-80GB",
        max_usd=0,
        shard_size=1,
        avg_generated_tokens=1,
        samples_per_prompt=1,
        yes=False,
        modal_module=fake,
        receipt=receipt,
        allow_partial=True,
    )
    assert result == 0
    assert manager.deleted == [script.VOLUME_NAME]


def _prepopulate_month(usd: float) -> None:
    from gemmaloop.stage0.spend import SpendLedger, default_ledger_path, month_key

    month = month_key(FIXED_NOW())
    ledger = SpendLedger(default_ledger_path(), now_fn=FIXED_NOW)
    reservation = ledger.reserve(
        shard=0,
        usd=usd,
        month=month,
        metadata={"run_id": "earlier-run"},
    )
    ledger.settle(
        reservation,
        shard=0,
        actual_usd=usd,
        month=month,
        metadata={"run_id": "earlier-run"},
    )


def test_lite_books_into_the_default_monthly_ledger_run_tagged(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    output = tmp_path / "modal_lite"
    assert script.run_mode(**_run(tmp_path, fake, fn, output_dir=output)) == 0

    from gemmaloop.stage0.spend import SpendLedger, default_ledger_path

    ledger = SpendLedger(default_ledger_path(), now_fn=FIXED_NOW)
    entries = ledger.entries()
    assert entries
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    lite_entries = [
        entry
        for entry in entries
        if (entry.get("metadata") or {}).get("run_id") == run_id
    ]
    assert {entry["kind"] for entry in lite_entries} == {"reservation", "actual"}
    assert all(
        (entry.get("metadata") or {}).get("mode") == "sample-lite"
        for entry in lite_entries
    )
    # Every lite booking is in the monthly file, not a run-specific one.
    assert not (tmp_path / "spend.json").exists()


def test_month_ceiling_refuses_before_modal_contact(tmp_path: Path):
    script = _load_script()
    _prepopulate_month(28.0)
    volume = FakeVolume()
    fake, created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    result = script.run_mode(
        **_run(tmp_path, fake, fn, max_usd=6, timeout_s=1800)
    )
    assert result == 2
    assert fn.calls == 0
    assert created == []
    assert not (tmp_path / "modal_lite" / "manifest.json").exists()


def test_run_cap_stops_before_the_next_call_and_keeps_non_train_complete(
    tmp_path: Path, capsys
):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += 1800.0
        return state["now"]

    output = tmp_path / "modal_lite"
    result = script.run_mode(
        **_run(
            tmp_path,
            fake,
            fn,
            output_dir=output,
            max_usd=6,
            timeout_s=1800,
            max_train_tokens=0,
            lite_call_size=8,
            clock=clock,
        )
    )
    assert result == 0  # the train goal is 0; non-train data is complete
    assert 2 <= fn.calls < 9
    assert "spending_cap" in capsys.readouterr().err
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "modal-vllm"
    receipt = json.loads((output / "modal_receipt.json").read_text())
    assert receipt["non_train_complete"] is True
    assert not receipt["missing_non_train"]


def test_custom_ledger_or_spend_lock_is_refused(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    assert (
        script.run_mode(
            **_run(
                tmp_path,
                fake,
                fn,
                ledger_path=tmp_path / "run_ledger.json",
                timeout_s=1800,
            )
        )
        == 2
    )
    assert fn.calls == 0
    assert (
        script.run_mode(
            **_run(
                tmp_path,
                fake,
                fn,
                spend_lock_path=tmp_path / "run.lock",
                timeout_s=1800,
            )
        )
        == 2
    )
    assert fn.calls == 0


def test_incomplete_non_train_refuses_finalize_without_a_manifest(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    plan = modal_lite.build_lite_plan(
        _pool_rows(),
        local.targets_from_prereg(load_prereg(default_prereg_path())),
        shard_size=2,
        prompt_tokens_for=_fake_prompt_tokens,
        call_size=16,
    )
    non_train_shard = next(
        shard.index for shard in plan.shards if shard.target_name != "train"
    )
    fn.skip_shards = {non_train_shard}
    output = tmp_path / "modal_lite"
    result = script.run_mode(
        **_run(tmp_path, fake, fn, output_dir=output, timeout_s=1800)
    )
    assert result == 2
    assert not (output / "manifest.json").exists()
    receipt = json.loads((output / "modal_receipt.json").read_text())
    assert receipt["non_train_complete"] is False
    assert f"shard_{non_train_shard:05d}.jsonl" in receipt["missing_non_train"]


def test_train_shortfall_is_reported_and_needs_the_explicit_flag(tmp_path: Path):
    script = _load_script()
    volume = FakeVolume()
    fake, _created, _manager = _fake_modal(volume)
    fn = FakeLiteFunction(volume)
    output = tmp_path / "modal_lite"
    result = script.run_mode(
        **_run(
            tmp_path,
            fake,
            fn,
            output_dir=output,
            timeout_s=1800,
            max_train_tokens=10_000_000,
        )
    )
    assert result == 3
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shortfalls"]["train_tokens"]["target"] == 10_000_000

    accepted = tmp_path / "accepted"
    assert (
        script.run_mode(
            **_run(
                tmp_path,
                fake,
                fn,
                output_dir=accepted,
                timeout_s=1800,
                max_train_tokens=10_000_000,
                accept_train_shortfall=True,
            )
        )
        == 0
    )


class _FakeCompletion:
    text = "<|channel>thought\nreason<channel|>answer"
    finish_reason = "stop"
    token_ids = None
    completion_tokens = 7


class _FakeOutput:
    def __init__(self):
        self.outputs = [_FakeCompletion()]


class _FakeLLM:
    def __init__(self):
        self.calls: list[tuple[list, list]] = []

    def generate(self, prompts, params):
        self.calls.append((list(prompts), list(params)))
        return [_FakeOutput() for _ in prompts]


class _FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "rendered prompt"

    def decode(self, *_args, **_kwargs):
        return "decoded"


def test_one_remote_call_uses_one_generate_over_all_shards(
    tmp_path: Path, monkeypatch
):
    script = _load_script()
    llm = _FakeLLM()
    tokenizer = _FakeTokenizer()
    volume = FakeVolume()
    fake_hub = SimpleNamespace(snapshot_download=lambda *a, **k: "model-path")
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    )
    fake_vllm = SimpleNamespace(LLM=object, SamplingParams=lambda **kw: kw)
    monkeypatch.setattr(
        script,
        "_import_remote_closure",
        lambda: (fake_hub, fake_transformers, fake_vllm),
    )
    monkeypatch.setattr(
        script,
        "_cached_engine",
        lambda *args, **kwargs: llm,
    )
    request = {
        "problem_id": "p",
        "sample_index": 0,
        "request_id": "p:0",
        "prompt": "hello",
        "prompt_tokens": 10,
        "max_new_tokens": 8182,
        "seed": 7,
    }
    payload = {
        "lite": True,
        "run_id": "r",
        "run_dir": str(tmp_path / "run"),
        "run_dir_rel": "shards/r",
        "pool_manifest_sha256": "pool",
        "gpu": "A100-80GB",
        "model_revision": "rev",
        "samples_per_prompt": 4,
        "engine_max_model_len": 131072,
        "_volume": volume,
        "shards": [
            {
                "shard_index": 3,
                "commit": True,
                "requests": [dict(request, request_id="p:0")],
            },
            {
                "shard_index": 4,
                "commit": True,
                "requests": [dict(request, request_id="p:1", sample_index=1)],
            },
        ],
    }
    result = script._sample_lite_impl(payload, None)
    assert len(llm.calls) == 1
    assert len(llm.calls[0][0]) == 2
    assert [item["shard"] for item in result["shards"]] == [3, 4]
    assert (tmp_path / "run" / "shard_00003.jsonl").exists()
    assert (tmp_path / "run" / "shard_00004.jsonl").exists()
    assert volume.commits == 2


class _RecordingApp:
    def __init__(self, *, fail_run: bool = False):
        self.fail_run = fail_run
        self.stops = 0

    def run(self):
        if self.fail_run:
            raise RuntimeError("app startup failed")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def stop(self):
        self.stops += 1
