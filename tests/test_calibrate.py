"""Discriminating tests for the calibration script (CPU-only)."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest
import torch

from gemmaloop.eval.runner import PartialProvenanceError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"
_SPEC = importlib.util.spec_from_file_location("gemmaloop_calibrate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
calibrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate)

TOTAL = 12 * 2**30


def test_summarize_steps_clean():
    result = calibrate.summarize_steps(
        512, [0.5, 0.5, 0.5], int(TOTAL * 0.50), TOTAL, 1
    )
    assert result["spill"] is False
    assert result["oom"] is False
    assert result["steps"] == 3
    assert result["tok_s"] == pytest.approx(1024.0)


def test_summarize_steps_never_averages_spilled_steps():
    # Rates: 1024, 1024, 256 tok/s.  The 256 step collapsed, so the reported
    # clean throughput must be 1024, not the overall mean of 768.
    result = calibrate.summarize_steps(
        512, [0.5, 0.5, 2.0], int(TOTAL * 0.50), TOTAL, 1
    )
    assert result["spill"] is True
    assert result["tok_s"] is None
    assert result["diagnostic_tok_s"] == pytest.approx(1024.0)
    assert result["step_tok_s"] == [1024.0, 1024.0, 256.0]


def test_summarize_steps_reserved_memory_marks_spill():
    result = calibrate.summarize_steps(
        512, [0.5, 0.5, 0.5], int(TOTAL * 0.96), TOTAL, 1
    )
    assert result["spill"] is True


def test_summarize_steps_oom_has_no_rate():
    result = calibrate.summarize_steps(
        512, [0.5], int(TOTAL * 0.50), TOTAL, 1, oom=True
    )
    assert result["oom"] is True
    assert result["spill"] is True
    assert result["tok_s"] is None


def test_choose_common_length_uses_the_worst_arm():
    probes = {
        "oh": {
            "512": {"spill": False, "oom": False, "tok_s": 1000.0},
            "1024": {"spill": False, "oom": False, "tok_s": 500.0},
            "1536": {"spill": True, "oom": False, "tok_s": 100.0},
        },
        "l": {
            "512": {"spill": False, "oom": False, "tok_s": 800.0},
            "1024": {"spill": False, "oom": False, "tok_s": 400.0},
            "1536": {"spill": False, "oom": False, "tok_s": 200.0},
        },
    }
    common, per_arm = calibrate.choose_common_length(probes, ["oh", "l"])
    assert common == 1024
    assert per_arm == {"oh": 1024, "l": 1536}


def test_choose_common_length_returns_none_for_a_failed_arm():
    probes = {
        "oh": {"512": {"spill": True, "oom": False, "tok_s": None}},
        "l": {"512": {"spill": False, "oom": False, "tok_s": 800.0}},
    }
    common, per_arm = calibrate.choose_common_length(probes, ["oh", "l"])
    assert common is None
    assert per_arm == {"oh": None, "l": 512}


def _synthetic(length: int, *, spill: bool, oom: bool = False, tok_s: float | None = 100.0):
    return {
        "peak_reserved_gib": round(length / 1024, 3),
        "tok_s": None if spill or oom else tok_s,
        "spill": spill,
        "oom": oom,
    }


def test_probe_descends_to_largest_clean_length():
    results = {
        4096: _synthetic(4096, spill=True),
        3072: _synthetic(3072, spill=True),
        2048: _synthetic(2048, spill=False, tok_s=200.0),
        1024: _synthetic(1024, spill=False, tok_s=400.0),
    }
    calls: list[int] = []

    def measure(length: int):
        calls.append(length)
        return results[length]

    chosen, probe = calibrate.probe_descending(
        [4096, 3072, 2048, 1024], {}, measure
    )
    assert chosen == 2048
    assert calls == [4096, 3072, 2048]  # stops at the first clean length
    assert probe["4096"]["peak_reserved_gib"] == 4.0
    assert probe["3072"]["tok_s"] is None
    assert probe["2048"]["tok_s"] == 200.0


def test_probe_stops_at_4096_when_it_is_clean():
    calls: list[int] = []

    def measure(length: int):
        calls.append(length)
        return _synthetic(length, spill=False, tok_s=50.0)

    chosen, probe = calibrate.probe_descending(
        [4096, 3072, 2048, 1024], {}, measure
    )
    assert chosen == 4096
    assert calls == [4096]
    assert list(probe) == ["4096"]


def test_probe_returns_none_when_every_length_spills():
    calls: list[int] = []

    def measure(length: int):
        calls.append(length)
        return _synthetic(length, spill=True)

    chosen, probe = calibrate.probe_descending(
        [4096, 3072, 2048, 1024], {}, measure
    )
    assert chosen is None
    assert calls == [4096, 3072, 2048, 1024]
    assert len(probe) == 4


def test_probe_treats_oom_like_spill_and_reuses_existing_results():
    existing = {
        "4096": _synthetic(4096, spill=True, oom=True),
        "3072": _synthetic(3072, spill=False, tok_s=300.0),
    }
    calls: list[int] = []

    def measure(length: int):
        calls.append(length)
        return _synthetic(length, spill=False)

    chosen, probe = calibrate.probe_descending(
        [4096, 3072, 2048, 1024], existing, measure
    )
    assert chosen == 3072
    assert calls == []  # both verdicts were already on disk
    assert probe is existing



def test_ensure_compatible_accepts_matching_fingerprint():
    data = {"fingerprint": {"a": 1}, "train_tok_per_s": {"e2b_oh": 1.0}}
    assert calibrate.ensure_compatible(data, {"a": 1}) is data


def test_ensure_compatible_refuses_mismatch():
    with pytest.raises(PartialProvenanceError):
        calibrate.ensure_compatible(
            {"fingerprint": {"a": 1}}, {"a": 2}
        )


def test_ensure_compatible_refuses_legacy_file():
    with pytest.raises(PartialProvenanceError):
        calibrate.ensure_compatible({"train_tok_per_s": {}}, {"a": 1})


def test_ensure_compatible_allows_empty_file():
    assert calibrate.ensure_compatible({}, {"a": 1}) == {}


def test_build_fingerprint_tracks_measurement_settings():
    first = calibrate.build_fingerprint(
        calibrate.parse_args(
            ["--sizes", "e2b", "--max-new-tokens", "16384"]
        )
    )
    second = calibrate.build_fingerprint(
        calibrate.parse_args(
            [
                "--sizes",
                "e2b",
                "--max-new-tokens",
                "32768",
                "--benches",
                "lcb,mmmupro",
            ]
        )
    )
    assert first != second
    assert first["bench"]["max_new_tokens"] == 16384
    assert second["bench"]["max_new_tokens"] == 32768
    assert first["bench"]["samples"]["lcb"] == 4
    assert first["bench"]["samples"]["aime"] == 8
    assert first["bench"]["image_soft_tokens"]["lcb"] == 280
    assert second["bench"]["image_soft_tokens"]["mmmupro"] == 1120


def test_pending_benchmark_items_skips_completed_records():
    class Item:
        def __init__(self, item_id):
            self.item_id = item_id

    items = [Item("a"), Item("b"), Item("c")]
    records = {("a", 0): {"item_id": "a"}, ("c", 0): {"item_id": "c"}}
    assert [item.item_id for item in calibrate.pending_benchmark_items(items, records)] == [
        "b"
    ]


def test_unavailable_benchmark_is_retried():
    # A skipped/unavailable benchmark has no measurement, so it stays pending;
    # the `skipped` map is not consulted (retry does not require --fresh).
    assert calibrate.pending_benchmarks(
        ["lcb", "gpqa"],
        {"lcb": {"p50": 1, "p90": 2}},
        {"lcb": 0.0},
    ) == ["gpqa"]
    assert calibrate.pending_benchmarks(
        ["lcb"], {"lcb": {"p50": 1, "p90": 2}}, {"lcb": 0.0}
    ) == []


def test_benchmark_partial_resume_and_fingerprint_refusal(tmp_path):
    output = tmp_path / "calibration.json"
    partial = calibrate.bench_partial_path(output, "lcb")
    fingerprint = {"bench": "lcb", "max_new_tokens": 32768, "items": "abc"}

    records, handle = calibrate._open_partial(partial, fingerprint, fresh=False)
    assert records == {}
    calibrate._write_records(
        handle,
        [
            {
                "item_id": "q1",
                "sample_idx": 0,
                "completion_tokens": 123,
                "finish_reason": "stop",
            }
        ],
    )
    handle.close()

    resumed, handle = calibrate._open_partial(partial, fingerprint, fresh=False)
    assert ("q1", 0) in resumed
    assert resumed[("q1", 0)]["completion_tokens"] == 123
    handle.close()

    with pytest.raises(PartialProvenanceError):
        calibrate._open_partial(
            partial,
            {**fingerprint, "max_new_tokens": 16384},
            fresh=False,
        )

    fresh_records, fresh_handle = calibrate._open_partial(
        partial,
        {**fingerprint, "max_new_tokens": 16384},
        fresh=True,
    )
    fresh_handle.close()
    assert fresh_records == {}


# ---------------------------------------------------------------------------
# Wiring tests (fake model/loader/generator; CPU only)
# ---------------------------------------------------------------------------


def _probe(tok_s: float | None, *, spill: bool = False, oom: bool = False):
    return {"spill": spill, "oom": oom, "tok_s": tok_s}


def test_registered_sample_map():
    assert calibrate.registered_sample_map(
        ["lcb", "aime", "gpqa", "mmlupro", "mmmupro", "mathvision", "spatialeval", "cvbench"]
    ) == {
        "lcb": 4,
        "aime": 8,
        "gpqa": 4,
        "mmlupro": 1,
        "mmmupro": 1,
        "mathvision": 1,
        "spatialeval": 1,
        "cvbench": 1,
    }


def test_measure_training_section_uses_the_common_nonspilling_length(
    tmp_path, monkeypatch
):
    probes = {
        "e2b_oh": {
            "512": _probe(1000.0),
            "1024": _probe(500.0),
            "1536": _probe(None, spill=True),
            "2048": _probe(None, spill=True),
        },
        "e2b_l": {
            "512": _probe(800.0),
            "1024": _probe(400.0),
            "1536": _probe(None, spill=True),
            "2048": _probe(None, spill=True),
        },
    }
    data = {
        "train_length_probe": probes,
        "phase2_max_length_n2": 1024,
        "phase2_max_length_n4": 512,
    }
    args = calibrate.parse_args(
        ["--sizes", "e2b", "--train-lengths", "512,1024,1536,2048", "--sections", "train"]
    )

    monkeypatch.setattr(
        calibrate,
        "_build_training_model",
        lambda: (object(), object(), object(), object(), 1000, object()),
    )
    monkeypatch.setattr(calibrate, "free_model", lambda: None)
    monkeypatch.setattr(calibrate, "set_n_loops", lambda *a, **k: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("_measure_length must not run when every probe is present")

    monkeypatch.setattr(calibrate, "_measure_length", forbidden)

    calibrate.measure_training_section(
        args, data, tmp_path / "calibration.json"
    )

    assert data["max_nonspill_train_seq_len"] == 1024
    assert data["max_nonspill_train_seq_len_by_arm"] == {
        "e2b_oh": 1024,
        "e2b_l": 1024,
    }
    assert data["train_measure_length"] == 1024
    assert data["train_tok_per_s"] == {"e2b_oh": 500.0, "e2b_l": 400.0}


def test_measure_training_section_reuses_phase2_probe_rows(tmp_path, monkeypatch):
    probes = {
        arm: {
            "512": _probe(1000.0),
            "1024": _probe(500.0),
            "1536": _probe(None, spill=True),
            "2048": _probe(None, spill=True),
        }
        for arm in ("e2b_oh", "e2b_l")
    }
    stored_n4 = {
        "4096": _probe(None, spill=True),
        "3072": _probe(300.0),
    }
    data = {
        "train_length_probe": probes,
        "train_tok_per_s": {"e2b_oh": 500.0, "e2b_l": 400.0},
        "max_nonspill_train_seq_len": 1024,
        "phase2_max_length_n2": 2048,
        "phase2_max_length_n4": None,
        "phase2_probe_n4": stored_n4,
    }
    args = calibrate.parse_args(
        ["--sizes", "e2b", "--train-lengths", "512,1024,1536,2048", "--sections", "train"]
    )

    monkeypatch.setattr(
        calibrate,
        "_build_training_model",
        lambda: (object(), object(), object(), object(), 1000, object()),
    )
    monkeypatch.setattr(calibrate, "free_model", lambda: None)
    monkeypatch.setattr(calibrate, "set_n_loops", lambda *a, **k: None)
    calls: list[int] = []

    def measure(length, *args, **kwargs):
        calls.append(length)
        return _probe(100.0)

    monkeypatch.setattr(calibrate, "_measure_length", measure)

    calibrate.measure_training_section(
        args, data, tmp_path / "calibration.json"
    )

    assert calls == []  # 4096 and 3072 were already on disk
    assert data["phase2_probe_n4"] == stored_n4
    assert data["phase2_max_length_n4"] == 3072
    assert data["phase2_max_length"] == 3072
    assert "phase2_max_length_error" not in data


def test_measure_benchmarks_halves_batch_and_checkpoints(tmp_path, monkeypatch):
    class Item:
        def __init__(self, item_id: str):
            self.item_id = item_id

    class FakeModule:
        def load_items(self, limit=None, seed=0):
            return [Item("q1"), Item("q2")][:limit]

        def build_messages(self, item):
            return [{"role": "user", "content": item.item_id}]

    monkeypatch.setattr(calibrate, "_benchmark_module", lambda name: FakeModule())
    monkeypatch.setattr(calibrate, "free_model", lambda: None)

    calls: list[int] = []

    def fake_gen(model, processor, conversations, sampling, seed=0, image_soft_tokens=280):
        calls.append(len(conversations))
        if len(conversations) > 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return [
            types.SimpleNamespace(completion_tokens=5, finish_reason="stop")
            for _ in conversations
        ]

    def fake_load(spec):
        return object(), object(), object()

    args = calibrate.parse_args(
        [
            "--sizes",
            "e2b",
            "--benches",
            "aime",
            "--items-per-bench",
            "2",
            "--max-new-tokens",
            "8",
            "--bench-batch",
            "8",
        ]
    )
    output = tmp_path / "calibration.json"
    data: dict = {}
    calibrate.measure_benchmarks(
        args, data, output, load_fn=fake_load, gen_fn=fake_gen
    )

    assert calls == [2, 2, 2, 1, 1]  # 8 -> 4 -> 2 -> 1, then the second item
    measured = data["tokens_per_item"]["aime"]
    assert measured["n_items"] == 2
    assert measured["p50"] == 5.0
    assert measured["p90"] == 5.0
    assert measured["basis"] == "per_completion"
    assert measured["samples"] == 8
    assert data["length_finish_rate"]["aime"] == 0.0

    partial = calibrate.bench_partial_path(output, "aime")
    lines = [line for line in partial.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3  # fingerprint header + two completed records


def test_e4b_low_ram_is_skipped_with_reason(tmp_path, monkeypatch):
    args = calibrate.parse_args(["--sizes", "e4b"])
    monkeypatch.setattr(calibrate, "available_ram_gib", lambda: 10.0)
    data: dict = {}
    calibrate.measure_generation_section(
        args, data, tmp_path / "calibration.json"
    )
    assert data["gen_tok_per_s"].get("e4b", {}) == {}
    assert data["skipped"]["e4b"]["reason"] == "skipped: low RAM"
    assert "at" in data["skipped"]["e4b"]


def _generation_record(batch: int, tok_s: float = 100.0) -> dict:
    return {
        "batch": int(batch),
        "tok_s": tok_s,
        "measured_tokens": 512,
        "measured_tokens_total": 512 * int(batch),
        "cache_new_tokens": 2048,
        "peak_reserved_gib": 5.0,
        "total_vram_gib": 12.0,
        "process_ram_gib": 6.0,
        "spill": False,
        "oom": False,
        "invalid": False,
        "memory_verified": True,
        "measurement": calibrate.GEN_MEASUREMENT_VERSION,
        "attempts": [],
    }


def test_generation_keys_follow_the_measured_batch(tmp_path, monkeypatch):
    args = calibrate.parse_args(["--sizes", "e2b,e4b"])
    monkeypatch.setattr(calibrate, "available_ram_gib", lambda: 100.0)
    monkeypatch.setattr(
        calibrate,
        "load_model",
        lambda spec: (
            types.SimpleNamespace(
                model=types.SimpleNamespace(language_model=object())
            ),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(calibrate, "set_n_loops", lambda *a, **k: None)
    monkeypatch.setattr(calibrate, "free_model", lambda: None)
    monkeypatch.setattr(calibrate, "_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        calibrate,
        "_release_generation_memory",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        calibrate,
        "_measure_generation",
        lambda model, processor, args, batch, bucket, **kwargs: _generation_record(
            batch
        ),
    )
    data: dict = {}
    calibrate.measure_generation_section(
        args, data, tmp_path / "calibration.json"
    )
    buckets = {str(length) for length in calibrate.GEN_LENGTH_BUCKETS}
    assert set(data["gen_tok_per_s"]) == {"e2b", "e4b"}
    assert set(data["gen_tok_per_s"]["e2b"]) == {
        f"n{loop}_len{bucket}"
        for loop in (1, 2, 4)
        for bucket in calibrate.GEN_LENGTH_BUCKETS
    }
    assert set(data["gen_tok_per_s"]["e4b"]) == {
        f"n{loop}_len{bucket}"
        for loop in (1, 2)
        for bucket in calibrate.GEN_LENGTH_BUCKETS
    }
    assert all(
        record["tok_s"] == 100.0
        for size_rates in data["gen_tok_per_s"].values()
        for record in size_rates.values()
    )
    assert data["gen_tok_per_s"]["e2b"]["n2_len2048"]["batch"] == 64
    assert data["gen_tok_per_s"]["e4b"]["n2_len32768"]["batch"] == 4
    assert set(buckets)


def test_stale_generation_records_are_remeasured_without_fresh(tmp_path, monkeypatch):
    args = calibrate.parse_args(
        ["--sizes", "e2b", "--gen-lengths", "2048", "--sections", "gen"]
    )
    monkeypatch.setattr(
        calibrate,
        "load_model",
        lambda spec: (
            types.SimpleNamespace(
                model=types.SimpleNamespace(language_model=object())
            ),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(calibrate, "set_n_loops", lambda *a, **k: None)
    monkeypatch.setattr(calibrate, "free_model", lambda: None)
    monkeypatch.setattr(calibrate, "_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        calibrate, "_release_generation_memory", lambda *a, **k: True
    )
    calls: list[str] = []

    def fake_measure(model, processor, args, batch, bucket, **kwargs):
        calls.append(f"{batch}")
        return _generation_record(batch, tok_s=200.0)

    monkeypatch.setattr(calibrate, "_measure_generation", fake_measure)
    # The current record must be kept; the legacy one (no measurement tag)
    # must be re-measured even though the fingerprint still matches.
    data = {
        "gen_tok_per_s": {
            "e2b": {
                "n1_len2048": _generation_record(64, tok_s=111.0),
                "n2_len2048": {"batch": 64, "tok_s": 222.0},
            }
        }
    }
    calibrate.measure_generation_section(
        args, data, tmp_path / "calibration.json"
    )
    assert data["gen_tok_per_s"]["e2b"]["n1_len2048"]["tok_s"] == 111.0
    assert data["gen_tok_per_s"]["e2b"]["n2_len2048"]["tok_s"] == 200.0
    assert (
        data["gen_tok_per_s"]["e2b"]["n2_len2048"]["measurement"]
        == calibrate.GEN_MEASUREMENT_VERSION
    )
    assert data["gen_tok_per_s"]["e2b"]["n4_len2048"]["tok_s"] == 200.0
    assert len(calls) == 2  # n2 (stale) and n4 (missing); n1 kept


def test_invalid_isolated_generation_record_is_remeasured(tmp_path, monkeypatch):
    args = calibrate.parse_args(
        ["--sizes", "e2b", "--gen-lengths", "2048", "--sections", "gen"]
    )
    monkeypatch.setattr(
        calibrate,
        "load_model",
        lambda spec: (
            types.SimpleNamespace(
                model=types.SimpleNamespace(language_model=object())
            ),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(calibrate, "set_n_loops", lambda *a, **k: None)
    monkeypatch.setattr(calibrate, "free_model", lambda: None)
    monkeypatch.setattr(calibrate, "_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        calibrate, "_release_generation_memory", lambda *a, **k: True
    )
    calls: list[str] = []

    def fake_measure(model, processor, args, batch, bucket, **kwargs):
        calls.append(f"{batch}")
        return _generation_record(batch, tok_s=200.0)

    monkeypatch.setattr(calibrate, "_measure_generation", fake_measure)
    invalid = _generation_record(64, tok_s=None)
    invalid.update(
        {
            "invalid": True,
            "spill": True,
            "attempts": [
                {"batch": 64, "tok_s": None, "spill": True, "oom": False},
                {"batch": 32, "tok_s": None, "spill": True, "oom": False},
            ],
        }
    )
    data = {
        "gen_tok_per_s": {
            "e2b": {
                # Even with the isolated tag, invalid:true must be re-measured.
                "n1_len2048": invalid,
                "n2_len2048": _generation_record(64, tok_s=111.0),
            }
        }
    }
    calibrate.measure_generation_section(
        args, data, tmp_path / "calibration.json"
    )
    assert data["gen_tok_per_s"]["e2b"]["n1_len2048"]["tok_s"] == 200.0
    assert data["gen_tok_per_s"]["e2b"]["n2_len2048"]["tok_s"] == 111.0
    assert len(calls) == 2  # n1 invalid, n4 missing; n2 kept


def test_generation_record_with_unverified_memory_is_stale():
    record = _generation_record(64, tok_s=111.0)
    assert calibrate._generation_record_is_current(record)
    record["memory_verified"] = False
    assert not calibrate._generation_record_is_current(record)


def _fake_cuda(monkeypatch, peaks, *, total=TOTAL):
    """Patch the CUDA calls used by the generation measurement."""
    monkeypatch.setattr(calibrate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibrate.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        calibrate.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(calibrate.torch.cuda, "memory_allocated", lambda: 0)
    peak_iter = iter(peaks)
    monkeypatch.setattr(
        calibrate.torch.cuda,
        "max_memory_reserved",
        lambda: next(peak_iter),
    )
    monkeypatch.setattr(calibrate, "_total_vram_bytes", lambda: total)
    monkeypatch.setattr(calibrate, "sample_resources", lambda: None)
    monkeypatch.setattr(calibrate, "process_ram_gib", lambda: 3.5)
    monkeypatch.setattr(calibrate, "log", lambda *a, **k: None)


def test_generation_warmup_is_excluded_and_1024_tokens_are_timed(monkeypatch):
    _fake_cuda(monkeypatch, [int(TOTAL * 0.50)])
    calls: list[tuple[int, int]] = []

    def fake_generate(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        max_batch=None,
        cache_new_tokens=None,
    ):
        calls.append((len(conversations), sampling.max_new_tokens))
        # Warm-up deliberately reports more tokens than it asked for; the rate
        # must only count the timed call.
        completion = 1000 if sampling.max_new_tokens <= 8 else sampling.max_new_tokens
        return [
            types.SimpleNamespace(completion_tokens=completion)
            for _ in conversations
        ]

    monkeypatch.setattr(calibrate, "generate_batch", fake_generate)
    args = calibrate.parse_args(["--sizes", "e2b", "--gen-new-tokens", "100"])
    record = calibrate._measure_generation(
        types.SimpleNamespace(generation_config=types.SimpleNamespace()),
        object(),
        args,
        4,
        2048,
        baseline_allocated=0,
    )
    # 4 rows x max(100, ceil(1024/4)=256) = 1024 timed tokens.
    assert record["measured_tokens"] == 256
    assert record["measured_tokens_total"] == 1024
    assert record["tok_s"] is not None
    assert record["attempts"][0]["tokens"] == 1024
    assert calls[0] == (4, 8)  # warm-up at the bucket's batch, excluded
    assert calls[1] == (4, 256)


def test_generation_spill_retries_at_half_batch_and_records_both(monkeypatch):
    _fake_cuda(monkeypatch, [int(TOTAL * 0.99), int(TOTAL * 0.50)])
    resets: list[int] = []
    releases: list[str] = []
    monkeypatch.setattr(
        calibrate.torch.cuda,
        "reset_peak_memory_stats",
        lambda: resets.append(1),
    )
    monkeypatch.setattr(
        calibrate,
        "_release_generation_memory",
        lambda *a, **k: releases.append("release") or True,
    )
    calls: list[tuple[int, int]] = []

    def fake_generate(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        max_batch=None,
        cache_new_tokens=None,
    ):
        calls.append((len(conversations), sampling.max_new_tokens))
        return [
            types.SimpleNamespace(completion_tokens=sampling.max_new_tokens)
            for _ in conversations
        ]

    monkeypatch.setattr(calibrate, "generate_batch", fake_generate)
    args = calibrate.parse_args(["--sizes", "e2b"])
    record = calibrate._measure_generation(
        types.SimpleNamespace(generation_config=types.SimpleNamespace()),
        object(),
        args,
        64,
        2048,
        baseline_allocated=0,
    )
    assert record["batch"] == 32
    assert record["spill"] is False
    assert record["invalid"] is False
    assert record["tok_s"] is not None
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["batch"] == 64
    assert record["attempts"][0]["spill"] is True
    assert record["attempts"][1]["batch"] == 32
    assert record["attempts"][1]["spill"] is False
    assert record["attempts"][0]["peak_reserved_gib"] > record["attempts"][1][
        "peak_reserved_gib"
    ]
    assert calls[0] == (64, 8)  # attempt 1 warm-up
    assert calls[1] == (64, 512)  # attempt 1 timed
    assert calls[2] == (32, 8)  # attempt 2 warm-up
    assert calls[3] == (32, 512)  # attempt 2 timed
    # Every timed window resets the peak counter, and every attempt releases
    # its memory before the next one starts.
    assert resets == [1, 1]
    assert releases == ["release", "release"]


def test_generation_marks_invalid_when_the_half_batch_also_spills(monkeypatch):
    _fake_cuda(monkeypatch, [int(TOTAL * 0.99), int(TOTAL * 0.98)])

    def fake_generate(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        max_batch=None,
        cache_new_tokens=None,
    ):
        return [
            types.SimpleNamespace(completion_tokens=sampling.max_new_tokens)
            for _ in conversations
        ]

    monkeypatch.setattr(calibrate, "generate_batch", fake_generate)
    args = calibrate.parse_args(["--sizes", "e2b"])
    record = calibrate._measure_generation(
        types.SimpleNamespace(generation_config=types.SimpleNamespace()),
        object(),
        args,
        64,
        32768,
        baseline_allocated=0,
    )
    assert record["invalid"] is True
    assert record["spill"] is True
    assert record["tok_s"] is None
    assert len(record["attempts"]) == 2
    assert not calibrate._generation_record_is_current(record)


def test_release_generation_memory_checks_the_model_baseline(monkeypatch):
    monkeypatch.setattr(calibrate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibrate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(calibrate, "_total_vram_bytes", lambda: TOTAL)
    monkeypatch.setattr(calibrate, "log", lambda *a, **k: None)

    monkeypatch.setattr(
        calibrate.torch.cuda,
        "memory_allocated",
        lambda: 5 * 2**30,
    )
    assert calibrate._release_generation_memory(5 * 2**30) is True

    monkeypatch.setattr(
        calibrate.torch.cuda,
        "memory_allocated",
        lambda: int(5 * 2**30 + calibrate.GEN_BASELINE_TOLERANCE_BYTES * 4),
    )
    assert calibrate._release_generation_memory(5 * 2**30) is False



def test_measure_length_oom_cleans_grads_and_tries_next_candidate(monkeypatch):
    total = 12 * 2**30
    monkeypatch.setattr(calibrate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(calibrate.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibrate, "_total_vram_bytes", lambda: total)
    monkeypatch.setattr(calibrate, "sample_resources", lambda: None)

    class Optimizer:
        def __init__(self):
            self.zero_calls = 0

        def zero_grad(self, set_to_none=True):
            self.zero_calls += 1

    optimizer = Optimizer()
    calls: list[int] = []

    def fake_step(model, opt, length, vocab_size, device):
        calls.append(length)
        if length == 4096:
            raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(calibrate, "_training_step", fake_step)

    def measure(length: int):
        return calibrate._measure_length(
            None,
            optimizer,
            length,
            1000,
            torch.device("cuda"),
            steps=2,
            n_loops=4,
        )

    chosen, probe = calibrate.probe_descending([4096, 3072], {}, measure)

    assert chosen == 3072
    assert calls[:3] == [4096, 3072, 3072]
    assert optimizer.zero_calls >= 2  # once after the OOM, once after the clean run
    assert probe["4096"]["oom"] is True
    assert probe["4096"]["tok_s"] is None
    assert probe["3072"]["oom"] is False
    assert probe["3072"]["tok_s"] is not None


def test_probe_does_not_accept_zero_throughput():
    existing = {
        "4096": {"spill": False, "oom": False, "tok_s": 0.0},
        "3072": {"spill": False, "oom": False, "tok_s": 100.0},
    }

    def forbidden(_length):
        raise AssertionError("existing results must be reused")

    chosen, _probe = calibrate.probe_descending([4096, 3072], existing, forbidden)
    assert chosen == 3072



def test_measure_length_includes_the_warmup_peak(monkeypatch):
    total = 12 * 2**30
    monkeypatch.setattr(calibrate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(calibrate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibrate, "_total_vram_bytes", lambda: total)
    monkeypatch.setattr(calibrate, "sample_resources", lambda: None)

    peaks = iter([int(total * 0.96), int(total * 0.50)])
    monkeypatch.setattr(
        calibrate.torch.cuda, "max_memory_reserved", lambda: next(peaks)
    )
    monkeypatch.setattr(calibrate, "_training_step", lambda *args, **kwargs: None)

    class Optimizer:
        def zero_grad(self, set_to_none=True):
            pass

    result = calibrate._measure_length(
        None,
        Optimizer(),
        512,
        1000,
        torch.device("cuda"),
        steps=2,
        n_loops=1,
    )
    assert result["spill"] is True
    assert result["tok_s"] is None
    assert result["peak_reserved_gib"] == pytest.approx(round(0.96 * total / 2**30, 3))


def test_bench_engine_defaults_to_padded_and_is_fingerprinted():
    default = calibrate.parse_args(["--sizes", "e2b"])
    assert default.bench_engine == "padded"
    assert calibrate.build_fingerprint(default)["bench"]["engine"] == "padded"

    refill = calibrate.parse_args(
        ["--sizes", "e2b", "--bench-engine", "refill"]
    )
    assert refill.bench_engine == "refill"
    assert calibrate.build_fingerprint(refill)["bench"]["engine"] == "refill"


def test_measure_benchmarks_refill_gets_the_whole_queue(tmp_path, monkeypatch):
    class Item:
        def __init__(self, item_id: str):
            self.item_id = item_id

    class FakeModule:
        def load_items(self, limit=None, seed=0):
            return [Item(f"q{index}") for index in range(4)]

        def build_messages(self, item):
            return [{"role": "user", "content": item.item_id}]

    monkeypatch.setattr(calibrate, "_benchmark_module", lambda name: FakeModule())
    monkeypatch.setattr(calibrate, "free_model", lambda: None)

    calls: list[tuple[list[str], int | None, list[str] | None]] = []

    def fake_gen(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
    ):
        calls.append(
            (
                [conversation[0]["content"] for conversation in conversations],
                max_batch,
                request_ids,
            )
        )
        return [
            types.SimpleNamespace(completion_tokens=5, finish_reason="stop")
            for _ in conversations
        ]

    def fake_load(spec):
        return object(), object(), object()

    args = calibrate.parse_args(
        [
            "--sizes",
            "e2b",
            "--benches",
            "aime",
            "--items-per-bench",
            "4",
            "--max-new-tokens",
            "8",
            "--bench-batch",
            "2",
            "--bench-engine",
            "refill",
        ]
    )
    output = tmp_path / "calibration.json"
    data: dict = {}
    calibrate.measure_benchmarks(
        args,
        data,
        output,
        load_fn=fake_load,
        gen_fn=fake_gen,
        engine=args.bench_engine,
    )

    assert len(calls) == 1
    contents, max_batch, request_ids = calls[0]
    assert contents == ["q0", "q1", "q2", "q3"]
    assert max_batch == 2
    assert request_ids == ["q0:0", "q1:0", "q2:0", "q3:0"]
    assert data["tokens_per_item"]["aime"]["n_items"] == 4


def test_measure_benchmarks_padded_uses_the_same_request_ids(tmp_path, monkeypatch):
    class Item:
        def __init__(self, item_id: str):
            self.item_id = item_id

    class FakeModule:
        def load_items(self, limit=None, seed=0):
            return [Item(f"q{index}") for index in range(4)]

        def build_messages(self, item):
            return [{"role": "user", "content": item.item_id}]

    monkeypatch.setattr(calibrate, "_benchmark_module", lambda name: FakeModule())
    monkeypatch.setattr(calibrate, "free_model", lambda: None)

    calls: list[tuple[list[str], list[str] | None]] = []

    def fake_gen(
        model,
        processor,
        conversations,
        sampling,
        seed=0,
        image_soft_tokens=280,
        request_ids=None,
        max_batch=None,
    ):
        calls.append(
            (
                [conversation[0]["content"] for conversation in conversations],
                request_ids,
            )
        )
        return [
            types.SimpleNamespace(completion_tokens=5, finish_reason="stop")
            for _ in conversations
        ]

    def fake_load(spec):
        return object(), object(), object()

    args = calibrate.parse_args(
        [
            "--sizes",
            "e2b",
            "--benches",
            "aime",
            "--items-per-bench",
            "4",
            "--max-new-tokens",
            "8",
            "--bench-batch",
            "2",
            "--bench-engine",
            "padded",
        ]
    )
    output = tmp_path / "calibration.json"
    data: dict = {}
    calibrate.measure_benchmarks(
        args,
        data,
        output,
        load_fn=fake_load,
        gen_fn=fake_gen,
        engine=args.bench_engine,
    )

    assert calls[0] == (["q0", "q1"], ["q0:0", "q1:0"])
    assert calls[1] == (["q2", "q3"], ["q2:0", "q3:0"])


def test_calibration_args_expose_the_allocator_cap():
    default = calibrate.parse_args(["--sizes", "e2b"])
    assert default.vram_cap == pytest.approx(0.92)
    assert default.vram_free_margin_gib == pytest.approx(0.5)
    disabled = calibrate.parse_args(["--sizes", "e2b", "--vram-cap", "0"])
    assert disabled.vram_cap == 0.0
