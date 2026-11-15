"""CPU tests for scripts/smoke_probe_memory.py.

The GPU/model parts are never touched: the synthetic row construction, the
argument surface, the PASS/FAIL rule and the report writer are exercised with a
fake tokenizer and monkeypatched CUDA measure functions.  The key safety
property is that building probe rows never opens a protected manifest.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
import torch

import scripts.smoke_probe_memory as smoke


class _FakeTokenizer:
    """Minimal Gemma 4 chat-template/tokenizer used by assistant_labels."""

    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str):
        return {
            "<|turn>": 1,
            "<turn|>": 2,
            "<|tool_response>": None,
            "<tool_response|>": None,
        }.get(token)

    def encode(self, text: str, add_special_tokens: bool = False):
        return [10 + (ord(character) % 50) for character in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
        enable_thinking: bool | None = None,
        tools=None,
    ):
        assert enable_thinking is True
        ids = [1, *self.encode("user"), *self.encode("\n")]
        ids.extend(self.encode(messages[0]["content"]))
        ids.append(2)
        ids.extend([1, *self.encode("model"), *self.encode("\n")])
        assistant = messages[-1]
        ids.extend(self.encode(assistant.get("reasoning", "")))
        ids.extend(self.encode(assistant.get("content", "")))
        ids.append(2)
        return {"input_ids": ids} if return_dict else ids


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_defaults_are_the_registered_smoke(tmp_path):
    args = smoke.parse_args(["--out", str(tmp_path / "out")])
    assert args.rows == 4
    assert args.max_length == 8192
    assert args.ns == (1, 4, 8)
    assert args.generation_max_new_tokens == 512
    assert args.vram_cap == pytest.approx(0.92)
    assert args.headroom_gib == pytest.approx(0.25)


def test_ns_parsing_is_strict_and_always_includes_eight(tmp_path):
    args = smoke.parse_args(["--out", str(tmp_path), "--ns", "4"])
    assert args.ns == (4, 8)
    args = smoke.parse_args(["--out", str(tmp_path), "--ns", "1, 8, 4"])
    assert args.ns == (1, 4, 8)
    with pytest.raises(SystemExit):
        smoke.parse_args(["--out", str(tmp_path), "--ns", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--out", str(tmp_path), "--ns", "x"])


# ---------------------------------------------------------------------------
# Synthetic rows: exact length, real template path, no protected manifests
# ---------------------------------------------------------------------------


def test_synthetic_rows_are_exactly_8192_tokens_and_touch_no_manifests(monkeypatch):
    tokenizer = _FakeTokenizer()
    opened: list[str] = []
    real_open = builtins.open
    real_path_open = Path.open
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def record_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def record_path_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    def record_read_text(self, *args, **kwargs):
        opened.append(str(self))
        return real_read_text(self, *args, **kwargs)

    def record_read_bytes(self, *args, **kwargs):
        opened.append(str(self))
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", record_open)
    monkeypatch.setattr(Path, "open", record_path_open)
    monkeypatch.setattr(Path, "read_text", record_read_text)
    monkeypatch.setattr(Path, "read_bytes", record_read_bytes)

    records = smoke.build_synthetic_probe_records(
        tokenizer, rows=4, max_length=8192, seed=7
    )

    assert len(records) == 4
    assert [len(record["input_ids"]) for record in records] == [8192] * 4
    assert [len(record["labels"]) for record in records] == [8192] * 4
    assert [len(record["attention_mask"]) for record in records] == [8192] * 4
    assert all(any(label != -100 for label in record["labels"]) for record in records)
    assert all(record["attention_mask"][0] == 1 for record in records)
    # No MONITOR/pilot/T-fit/selection/confirmatory/eval manifest was opened.
    forbidden = [
        path
        for path in opened
        if any(name in path for name in smoke._PROTECTED_NAMES)
    ]
    assert forbidden == []


def test_lower_max_length_is_padded_or_truncated_exactly(monkeypatch):
    tokenizer = _FakeTokenizer()
    records = smoke.build_synthetic_probe_records(
        tokenizer, rows=2, max_length=256, seed=0
    )
    assert [len(record["input_ids"]) for record in records] == [256, 256]
    assert all(len(record["labels"]) == 256 for record in records)


def test_probe_config_n8_is_the_registered_training_config():
    config = smoke.probe_config_for_n(8)
    assert config.eval_ns == (1, 2, 3, 4, 6, 8)
    assert config.nll_ns == (1, 2, 3, 4)
    assert config.anchor_ns == (2, 3, 4, 6, 8)
    assert config.generation_ns == ()
    assert config.generation_max_new_tokens == 512

    config = smoke.probe_config_for_n(4)
    assert config.eval_ns == (1, 2, 3, 4)
    assert config.nll_ns == (1, 2, 3, 4)
    assert config.anchor_ns == (2, 3, 4)

    config = smoke.probe_config_for_n(1)
    assert config.eval_ns == (1,)
    assert config.nll_ns == (1,)
    assert config.anchor_ns == ()

    generated = smoke.probe_config_for_n(8, generation=True)
    assert generated.generation_ns == (1, 2, 4)


# ---------------------------------------------------------------------------
# PASS/FAIL rule
# ---------------------------------------------------------------------------


def _phase(name: str, headroom: float | None, *, oom: bool = False, error=None):
    return {
        "phase": name,
        "headroom_gib": headroom,
        "oom": oom,
        "error": error,
    }


def test_decision_passes_only_with_headroom_and_no_oom():
    decision = smoke.decide_status(
        [
            _phase("training_state", 1.0),
            _phase("probe-N8", 0.25),
        ]
    )
    assert decision["status"] == "PASS"
    assert decision["pass"] is True
    assert decision["failures"] == []


def test_decision_fails_on_low_headroom_and_names_the_phase():
    decision = smoke.decide_status(
        [
            _phase("training_state", 1.0),
            _phase("probe-N8", 0.249),
        ]
    )
    assert decision["status"] == "FAIL"
    assert decision["pass"] is False
    assert any("probe-N8" in failure and "headroom" in failure for failure in decision["failures"])


def test_decision_fails_on_oom_even_without_a_cap():
    decision = smoke.decide_status(
        [
            _phase("training_state", None),
            _phase("probe-N8", None, oom=True),
        ]
    )
    assert decision["status"] == "FAIL"
    assert any("probe-N8" in failure and "OOM" in failure for failure in decision["failures"])


def test_decision_treats_other_errors_as_failures():
    decision = smoke.decide_status([_phase("probe-N8", 2.0, error="RuntimeError: boom")])
    assert decision["status"] == "FAIL"
    assert decision["failures"] == ["probe-N8: RuntimeError: boom"]


# ---------------------------------------------------------------------------
# Measurement and report writing (CUDA faked)
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, *, raise_oom: bool = False) -> None:
        self.raise_oom = raise_oom
        self.calls = 0

    def run(self, *, tokens_seen, step, train_teacher_kl=None):
        self.calls += 1
        if self.raise_oom:
            raise torch.cuda.OutOfMemoryError("simulated probe OOM")
        return {
            "invariant": {"applicable": True, "ok": True},
            "anchor_kl": {"8": 0.1},
            "teacher_kl": {},
            "calibrated_nll": {"8": 2.0},
            "probe_errors": [],
            "generation": [],
        }


def _fake_cuda(monkeypatch, peak_reserved_gib: float):
    monkeypatch.setattr(smoke, "_synchronize", lambda: None)
    monkeypatch.setattr(smoke, "_reset_peak_memory", lambda: None)
    monkeypatch.setattr(smoke, "_empty_cache", lambda: None)
    monkeypatch.setattr(
        smoke,
        "_memory_totals_gib",
        lambda: {"allocated_gib": 1.0, "reserved_gib": peak_reserved_gib - 0.5},
    )
    monkeypatch.setattr(
        smoke,
        "_peak_memory_gib",
        lambda: {
            "peak_allocated_gib": peak_reserved_gib - 0.5,
            "peak_reserved_gib": peak_reserved_gib,
        },
    )


def test_run_probe_phase_measures_and_passes(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    phase = smoke.run_probe_phase(
        name="probe-N8",
        n_loops=8,
        make_runner=lambda: _FakeRunner(),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
    )
    assert phase["status"] == "PASS"
    assert phase["peak_reserved_gib"] == pytest.approx(8.5)
    assert phase["headroom_gib"] == pytest.approx(1.5)
    assert phase["probe"]["anchor_kl"] == {"8": 0.1}
    assert phase["oom"] is False


def test_run_probe_phase_records_oom_as_fail(monkeypatch):
    _fake_cuda(monkeypatch, 10.0)
    phase = smoke.run_probe_phase(
        name="probe-N8",
        n_loops=8,
        make_runner=lambda: _FakeRunner(raise_oom=True),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
    )
    assert phase["status"] == "FAIL"
    assert phase["oom"] is True
    assert "OOM" in phase["reason"]


def test_run_probe_phase_fails_on_insufficient_headroom(monkeypatch):
    _fake_cuda(monkeypatch, 9.9)
    phase = smoke.run_probe_phase(
        name="probe-N8",
        n_loops=8,
        make_runner=lambda: _FakeRunner(),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
    )
    assert phase["status"] == "FAIL"
    assert "headroom" in phase["reason"]


def _payload():
    return {
        "kind": "probe_memory_smoke",
        "status": "PASS",
        "created_utc": "2026-11-06T00:00:00+00:00",
        "out": "out",
        "headroom_gib": 0.25,
        "config": {
            "rows": 4,
            "max_length": 8192,
            "ns": [1, 4, 8],
        },
        "allocator": {
            "cap_gib": 10.0,
            "total_gib": 12.0,
            "free_at_start_gib": 11.0,
        },
        "training_state": {
            "size": "e2b",
            "quant": "bf16",
            "arm": "L",
            "block": [25, 29],
            "bridge": "gch",
            "optimizer": "adamw_8bit",
            "optimizer_state": {"parameters_with_state": 10},
            "trainable_parameters": 100,
        },
        "phases": [
            {
                "phase": "training_state",
                "n_loops": None,
                "rows": None,
                "max_length": None,
                "peak_allocated_gib": 7.0,
                "peak_reserved_gib": 8.0,
                "headroom_gib": 2.0,
                "wall_s": None,
                "status": "PASS",
            },
            {
                "phase": "probe-N8",
                "n_loops": 8,
                "rows": 4,
                "max_length": 8192,
                "peak_allocated_gib": 7.5,
                "peak_reserved_gib": 8.5,
                "headroom_gib": 1.5,
                "wall_s": 12.0,
                "status": "PASS",
            },
        ],
        "failures": [],
        "notes": ["synthetic rows only"],
    }


def test_write_reports_writes_json_and_markdown(tmp_path):
    json_path, md_path = smoke.write_reports(tmp_path, _payload())
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Probe memory smoke: PASS" in markdown
    assert "| probe-N8 | 8 | 4 | 8192 |" in markdown
    assert "none" in markdown


def test_main_without_cuda_still_writes_a_fail_report(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = smoke.main(
        [
            "--out",
            str(tmp_path / "report"),
            "--heartbeat",
            str(tmp_path / "heartbeat.json"),
        ]
    )
    assert status == 2
    payload = json.loads(
        (tmp_path / "report" / "probe_memory.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "FAIL"
    assert any("CUDA" in failure for failure in payload["failures"])
    assert (tmp_path / "report" / "probe_memory.md").is_file()


# ---------------------------------------------------------------------------
# Probe forward micro-batch is independent of the number of probe rows
# ---------------------------------------------------------------------------


def test_probe_batch_size_defaults_to_the_training_microbatch(tmp_path):
    from gemmaloop.train.heal_monitor import MonitorConfig

    args = smoke.parse_args(["--out", str(tmp_path / "out")])
    assert args.probe_batch_size == 1
    assert args.probe_batch_size == int(MonitorConfig().probe_batch_size)

    args = smoke.parse_args(
        ["--out", str(tmp_path / "out"), "--probe-batch-size", "2"]
    )
    assert args.probe_batch_size == 2
    with pytest.raises(SystemExit):
        smoke.parse_args(["--out", str(tmp_path / "out"), "--probe-batch-size", "0"])

    payload = smoke._base_payload(
        smoke.parse_args(
            ["--out", str(tmp_path / "out"), "--probe-batch-size", "3"]
        )
    )
    assert payload["config"]["probe_batch_size"] == 3


class _SmokeProbeDataset:
    column_names = ["input_ids", "labels", "attention_mask"]

    def __init__(self) -> None:
        self.rows = [
            {
                "input_ids": [3, 4, 5, 6],
                "labels": [-100, -100, 5, 6],
                "attention_mask": [1, 1, 1, 1],
            }
            for _ in range(4)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class _SmokeEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)


def test_probe_batch_size_splits_four_rows_independently_of_row_count():
    from types import SimpleNamespace

    dataset = _SmokeProbeDataset()
    model = _SmokeEmbeddingModel()
    tokenizer = SimpleNamespace(pad_token_id=0)

    micro = smoke.build_probe_batches_for_smoke(
        dataset, model=model, tokenizer=tokenizer, rows=4, batch_size=1
    )
    assert len(micro) == 4
    assert [int(batch["input_ids"].shape[0]) for batch in micro] == [1, 1, 1, 1]

    combined = smoke.build_probe_batches_for_smoke(
        dataset, model=model, tokenizer=tokenizer, rows=4, batch_size=4
    )
    assert len(combined) == 1
    assert int(combined[0]["input_ids"].shape[0]) == 4


def test_generation_prompts_can_exceed_the_probe_forward_rows():
    """4 probe rows at micro-batch 1 with 8 generation prompts, as in training."""

    from gemmaloop.train.heal_monitor import build_prompt_tensors

    tokenizer = _FakeTokenizer()
    records = smoke.build_synthetic_probe_records(
        tokenizer, rows=8, max_length=256, seed=0
    )
    assert len(records) == 8
    prompts = build_prompt_tensors(
        records, list(range(8)), device=torch.device("cpu"), count=8
    )
    assert len(prompts) == 8
    assert all(prompt.numel() > 0 for prompt in prompts)


class _ResultRunner:
    def __init__(self, result):
        self.result = result

    def run(self, *, tokens_seen, step, train_teacher_kl=None):
        return dict(self.result)


def _probe_result(*, errors=(), generation=()):
    return {
        "invariant": {"applicable": True, "ok": True},
        "anchor_kl": {"8": 0.1},
        "teacher_kl": {},
        "calibrated_nll": {"8": 2.0},
        "probe_errors": list(errors),
        "generation": list(generation),
    }


def test_run_probe_phase_fails_on_any_probe_error(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    phase = smoke.run_probe_phase(
        name="probe-N8+generation",
        n_loops=8,
        make_runner=lambda: _ResultRunner(
            _probe_result(errors=["generation: Expected all tensors on one device"])
        ),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
        expected_generation_rows=8,
    )
    assert phase["status"] == "FAIL"
    assert "generation" in phase["reason"]
    assert phase["probe_errors"] == [
        "generation: Expected all tensors on one device"
    ]


def test_run_probe_phase_fails_when_generation_rows_are_missing(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    phase = smoke.run_probe_phase(
        name="probe-N8+generation",
        n_loops=8,
        make_runner=lambda: _ResultRunner(_probe_result()),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
        expected_generation_rows=8,
    )
    assert phase["status"] == "FAIL"
    assert "generation produced 0 rows, expected 8" in phase["reason"]


def test_run_probe_phase_accepts_the_expected_generation_rows(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    rows = [
        {"finish_reason": "eos", "answer_present": True, "thinking_closed": True}
        for _ in range(8)
    ]
    phase = smoke.run_probe_phase(
        name="probe-N8+generation",
        n_loops=8,
        make_runner=lambda: _ResultRunner(_probe_result(generation=rows)),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
        expected_generation_rows=8,
    )
    assert phase["status"] == "PASS"
    assert phase["generation_rows"] == 8


def test_decision_fails_on_a_probe_error_phase_without_oom_or_headroom_failure():
    decision = smoke.decide_status(
        [
            {
                "phase": "probe-N8+generation",
                "headroom_gib": 2.0,
                "oom": False,
                "error": None,
                "probe_errors": ["generation: boom"],
            }
        ]
    )
    assert decision["status"] == "FAIL"
    assert decision["failures"] == [
        "probe-N8+generation: generation: boom"
    ]


class _OffloadEncodingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Linear(2, 2, bias=False)
        self.embed_vision = torch.nn.Embedding(2, 2)


class _TimedGenerationRunner:
    """Minimal runner that goes through the wrapped ``_generation`` timing."""

    def __init__(self, *, move_offload: bool = False) -> None:
        self.model = _OffloadEncodingModel()
        self.move_offload = move_offload

    def _generation(self, *, tokens_seen: int, step: int):
        del tokens_seen, step
        if self.move_offload:
            self.model.vision_tower.to("meta")
        return [{"finish_reason": "eos"}]

    def run(self, *, tokens_seen: int, step: int, train_teacher_kl=None):
        del train_teacher_kl
        return {
            "invariant": {"applicable": True, "ok": True},
            "anchor_kl": {"8": 0.1},
            "teacher_kl": {},
            "calibrated_nll": {"8": 2.0},
            "probe_errors": [],
            "generation": self._generation(tokens_seen=tokens_seen, step=step),
        }


def test_run_probe_phase_records_generation_time_separately(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    phase = smoke.run_probe_phase(
        name="probe-N8+generation",
        n_loops=8,
        make_runner=lambda: _TimedGenerationRunner(),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
        expected_generation_rows=1,
    )
    assert phase["status"] == "PASS"
    assert phase["generation_wall_s"] is not None
    assert phase["generation_wall_s"] >= 0.0
    assert phase["probe_forward_wall_s"] is not None
    assert phase["probe_forward_wall_s"] >= 0.0
    assert phase["generation_allocated_before_gib"] == pytest.approx(1.0)
    assert phase["generation_allocated_after_gib"] == pytest.approx(1.0)


def test_run_probe_phase_fails_when_an_offloaded_module_is_not_restored(
    monkeypatch,
):
    _fake_cuda(monkeypatch, 8.5)
    phase = smoke.run_probe_phase(
        name="probe-N8+generation",
        n_loops=8,
        make_runner=lambda: _TimedGenerationRunner(move_offload=True),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
        expected_generation_rows=1,
    )
    assert phase["status"] == "FAIL"
    assert "not restored" in phase["reason"]


def test_run_probe_phase_fails_when_the_invariant_is_false(monkeypatch):
    _fake_cuda(monkeypatch, 8.5)
    result = _probe_result()
    result["invariant"] = {
        "applicable": True,
        "ok": False,
        "max_abs_diff": 2e-3,
        "reason": "shape mismatch",
    }
    phase = smoke.run_probe_phase(
        name="probe-N8",
        n_loops=8,
        make_runner=lambda: _ResultRunner(result),
        rows=4,
        max_length=8192,
        cap_gib=10.0,
        headroom_gib=0.25,
    )
    assert phase["status"] == "FAIL"
    assert "invariant" in phase["reason"]
