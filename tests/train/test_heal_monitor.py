"""CPU tests for heal monitoring: metrics, rules, split and probe helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from gemmaloop.train.dial_objective import TeacherTopK
from gemmaloop.train.heal_monitor import (
    HealMonitor,
    HealMonitorCallback,
    MonitorAlert,
    MonitorConfig,
    MonitorRules,
    atomic_append_jsonl,
    bridge_g_stats,
    gradient_stats,
    json_safe,
    load_reserved_ids,
    lora_b_norms,
    rolling_median,
    select_monitor_rows,
    teacher_integrity,
)
from gemmaloop.train.heal_probes import (
    chunked_kl_hidden,
    chunked_nll_hidden,
    exit1_bit_exact,
    generation_sanity,
)


# ---------------------------------------------------------------------------
# Split, JSONL and pure metric helpers
# ---------------------------------------------------------------------------


def test_monitor_split_is_deterministic_and_disjoint():
    ids = [str(index) for index in range(200)]
    reserved = {"1", "2", "3", "50", "51"}
    first = select_monitor_rows(ids, reserved=reserved, count=12, seed=7)
    second = select_monitor_rows(list(reversed(ids)), reserved=reserved, count=12, seed=7)
    assert first == second
    assert len(first) == 12
    assert set(first).isdisjoint(reserved)

    with pytest.raises(ValueError, match="only"):
        select_monitor_rows(ids[:5], reserved={"0", "1"}, count=4, seed=0)


def test_reserved_ids_change_the_monitor_selection():
    ids = [str(index) for index in range(50)]
    top = select_monitor_rows(ids, reserved=set(), count=1, seed=11)[0]
    after = select_monitor_rows(ids, reserved={top}, count=1, seed=11)[0]
    assert after != top
    # A mutant that ignores reserved would return ``top`` again.
    assert top not in select_monitor_rows(ids, reserved={top}, count=5, seed=11)


def test_load_reserved_ids_supports_common_formats(tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["a", 1]), encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps({"pilot": ["p1"], "fit": ["f1"], "problem_ids": ["x"]}),
        encoding="utf-8",
    )
    assert load_reserved_ids(str(ids)) == {"a", "1"}
    assert load_reserved_ids([mapping]) == {"p1", "f1", "x"}


def test_atomic_append_jsonl_is_strict_json_and_appends(tmp_path):
    path = tmp_path / "metrics.jsonl"
    atomic_append_jsonl(path, {"a": math.nan, "b": 1.0})
    atomic_append_jsonl(path, {"a": 2.0})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": None, "b": 1.0}
    assert json.loads(lines[1]) == {"a": 2.0}
    assert json_safe({"x": [math.inf, 1.0]}) == {"x": [None, 1.0]}


def test_rolling_median_ignores_non_finite():
    assert rolling_median([1.0, 3.0, math.nan, 2.0]) == 2.0
    assert rolling_median([]) is None


def test_teacher_integrity_matches_direct_sums():
    probs = torch.tensor([[0.2, 0.3], [0.45, 0.45]], dtype=torch.float16)
    tail = torch.tensor([0.5, 0.1], dtype=torch.float32)
    teacher = TeacherTopK(probs=probs, indices=torch.tensor([[0, 1], [2, 3]]), tail_prob=tail)
    report = teacher_integrity(teacher, 8)
    before = probs.float().sum(dim=-1) + tail
    assert report["max_abs_deviation_before"] == pytest.approx(
        float((before - 1.0).abs().max())
    )
    assert report["ok"] is True

    bad = TeacherTopK(
        probs=torch.ones(2, 2), indices=torch.tensor([[0, 1], [2, 3]]), tail_prob=torch.zeros(2)
    )
    report = teacher_integrity(bad, 8, tolerance=2e-3)
    assert report["ok"] is False
    assert report["max_abs_deviation_before"] == pytest.approx(1.0)


def test_bridge_g_and_lora_norms_match_direct_computation():
    g = torch.tensor([0.0, 1e-4, 0.6, -0.6, 0.2])
    stats = bridge_g_stats(g)
    assert stats["channels"] == 5
    assert stats["max_abs"] == pytest.approx(0.6)
    assert stats["active_fraction"] == pytest.approx(3 / 5)
    assert stats["saturation_fraction"] == pytest.approx(2 / 5)
    assert stats["all_zero"] is False
    assert bridge_g_stats(torch.zeros(4))["all_zero"] is True
    assert bridge_g_stats(None) == {"available": False}

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Module()
    model.layers[0].q_proj.lora_B = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.layers[0].q_proj.lora_B.weight.copy_(torch.arange(6).reshape(2, 3))
    norms = lora_b_norms(model)
    assert norms["layers.0.q_proj"] == pytest.approx(
        float(torch.linalg.vector_norm(torch.arange(6.0)))
    )


def test_gradient_stats_match_direct_norms_per_group():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Module()
    model.layers[0].q_proj.lora_A = torch.nn.Linear(2, 2, bias=False)
    model.loop_bridge_g = torch.nn.Parameter(torch.zeros(3))
    model.loop_ple = torch.nn.Parameter(torch.zeros(2))
    for name in ("layers.0.q_proj.lora_A.weight", "loop_bridge_g", "loop_ple"):
        parameter = dict(model.named_parameters())[name]
        parameter.requires_grad_(True)
        parameter.grad = torch.ones_like(parameter)
    stats = gradient_stats(model)
    assert stats["lora"]["grad_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(torch.ones(2, 2)))
    )
    assert stats["bridge"]["grad_norm"] == pytest.approx(math.sqrt(3))
    assert stats["loop_ple"]["grad_norm"] == pytest.approx(math.sqrt(2))


# ---------------------------------------------------------------------------
# Chunked distribution helpers against direct softmax computation
# ---------------------------------------------------------------------------


def _direct_logits(hidden: torch.Tensor, weight: torch.Tensor, softcap):
    logits = hidden.float() @ weight.float().t()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def test_chunked_kl_and_nll_match_direct_softmax():
    torch.manual_seed(0)
    hidden_p = torch.randn(3, 4)
    hidden_q = torch.randn(3, 4)
    weight = torch.randn(7, 4)
    softcap = 5.0
    logits_p = _direct_logits(hidden_p, weight, softcap)
    logits_q = _direct_logits(hidden_q, weight, softcap)
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    expected_kl = (logp.exp() * (logp - logq)).sum(dim=-1)
    actual_kl = chunked_kl_hidden(
        hidden_p, hidden_q, weight, softcap=softcap, token_chunk=1, vocab_chunk=2
    )
    assert torch.allclose(actual_kl, expected_kl, atol=1e-5)

    targets = torch.tensor([0, 4, 6])
    expected_nll = -logq[torch.arange(3), targets]
    actual_nll = chunked_nll_hidden(
        hidden_q, weight, targets, softcap=softcap, token_chunk=1, vocab_chunk=3
    )
    assert torch.allclose(actual_nll, expected_nll, atol=1e-5)


# ---------------------------------------------------------------------------
# Generation sanity
# ---------------------------------------------------------------------------


def test_generation_sanity_accepts_a_clean_answer():
    text = "<|channel>thought\nstep 1<channel|>The answer is 7<turn|>"
    result = generation_sanity(text)
    assert result["thinking_closed"] is True
    assert result["answer_present"] is True
    assert result["repetition_loop"] is False


def test_generation_sanity_flags_open_thinking_missing_answer_and_loops():
    open_channel = "<|channel>thought\nstill thinking"
    result = generation_sanity(open_channel)
    assert result["thinking_closed"] is False
    assert result["answer_present"] is False

    repeated = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * 6)
    result = generation_sanity(repeated, ngram=4, repetition_max_count=3)
    assert result["repetition_loop"] is True
    assert result["max_ngram_repeat"] >= 3


# ---------------------------------------------------------------------------
# Rule engine: synthetic series
# ---------------------------------------------------------------------------


def _step(tokens: int, **overrides):
    record = {
        "kind": "step",
        "step": tokens // 100,
        "tokens_seen": int(tokens),
        "loss": 1.0,
        "grad_norm_total": 1.0,
        "teacher_integrity": {"available": True, "ok": True},
        "nonfinite": {"loss": 0, "grads": 0, "logits": 0},
        "primary_kl_teacher": 1.0,
        "hinge_active_max": 0.0,
        "bridge_g": {
            "available": True,
            "saturation_fraction": 0.0,
            "all_zero": False,
        },
        "tok_per_s": 100.0,
        "reserved_gib": 1.0,
        "memory_cap_gib": 10.0,
    }
    record.update(overrides)
    return record


def _healthy_probe(tokens: int = 500_000):
    return {
        "kind": "probe",
        "step": tokens // 100,
        "tokens_seen": tokens,
        "invariant": {"applicable": True, "ok": True},
        "anchor_kl": {"2": 0.2, "3": 0.2, "4": 0.2, "6": 0.2, "8": 0.2},
        "teacher_kl": {"2": 1.0, "4": 1.0},
        "teacher_kl_gap": {"4": 0.01},
        "calibrated_nll": {"1": 2.0, "2": 1.9, "3": 1.8, "4": 1.7},
        "generation": [
            {
                "prompt_index": 0,
                "n_loops": 2,
                "repetition_loop": False,
                "answer_present": True,
                "thinking_closed": True,
            }
        ],
    }


def test_rules_stay_silent_on_a_healthy_series():
    rules = MonitorRules(MonitorConfig())
    alerts = []
    for index in range(60):
        alerts.extend(
            rules.observe_step(
                _step(
                    100_000 + 50_000 * index,
                    primary_kl_teacher=1.0 - 0.005 * index,
                )
            )
        )
    alerts.extend(rules.observe_probe(_healthy_probe()))
    assert alerts == []
    assert rules.abort_triggered is False


def test_nonfinite_aborts_immediately():
    rules = MonitorRules(MonitorConfig())
    alerts = rules.observe_step(_step(100, nonfinite={"loss": 1, "grads": 0, "logits": 0}))
    assert [alert.severity for alert in alerts] == ["abort"]
    assert alerts[0].rule == "nonfinite"
    assert rules.abort_triggered is True


def test_teacher_integrity_failure_aborts():
    rules = MonitorRules(MonitorConfig())
    alerts = rules.observe_step(
        _step(100, teacher_integrity={"available": True, "ok": False})
    )
    assert alerts[0].rule == "teacher_integrity"
    assert alerts[0].severity == "abort"


def test_loss_spike_needs_twenty_consecutive_steps():
    config = MonitorConfig(loss_spike_steps=20, loss_warmup_steps=5, warn_repeat_tokens=0)
    rules = MonitorRules(config)
    alerts = []
    for index in range(60):
        alerts.extend(rules.observe_step(_step(index * 100, loss=1.0)))
    for index in range(19):
        alerts.extend(rules.observe_step(_step(6000 + index * 100, loss=3.5)))
    assert not any(alert.rule == "loss_spike" for alert in alerts)
    alerts.extend(rules.observe_step(_step(8000, loss=3.5)))
    assert any(alert.rule == "loss_spike" for alert in alerts)


def test_gradient_spike_warns_on_its_rolling_median():
    config = MonitorConfig(grad_warmup_steps=3, warn_repeat_tokens=0)
    rules = MonitorRules(config)
    for index in range(5):
        rules.observe_step(_step(index * 100, grad_norm_total=1.0))
    alerts = rules.observe_step(_step(600, grad_norm_total=20.0))
    assert [alert.rule for alert in alerts] == ["grad_spike"]
    assert alerts[0].severity == "warn"


def test_teacher_stall_warns_only_after_one_million_tokens():
    config = MonitorConfig(warn_repeat_tokens=0)
    rules = MonitorRules(config)
    early: list = []
    for index in range(10):
        early.extend(
            rules.observe_step(_step(index * 100_000, primary_kl_teacher=1.0))
        )
    assert not any(alert.rule == "teacher_stall" for alert in early)
    later = rules.observe_step(_step(1_100_000, primary_kl_teacher=1.0))
    assert any(alert.rule == "teacher_stall" for alert in later)


def test_hinge_bridge_memory_and_throughput_warnings():
    config = MonitorConfig(warn_repeat_tokens=0, throughput_warmup_steps=1)
    rules = MonitorRules(config)
    alerts = []
    for index in range(20):
        alerts.extend(
            rules.observe_step(
                _step(
                    1_000_000 + index * 100,
                    hinge_active_max=0.8,
                    bridge_g={
                        "available": True,
                        "saturation_fraction": 0.2,
                        "all_zero": False,
                    },
                    reserved_gib=9.8,
                    memory_cap_gib=10.0,
                    tok_per_s=100.0,
                )
            )
        )
    rules_alerts = [alert.rule for alert in alerts]
    assert "hinge_active" in rules_alerts
    assert "bridge_saturated" in rules_alerts
    assert "memory_margin" in rules_alerts

    # Fill the throughput window, then drop below half of its median.
    for index in range(5):
        rules.observe_step(_step(2_000_000 + index * 100, tok_per_s=100.0))
    alerts = rules.observe_step(_step(3_000_000, tok_per_s=10.0))
    assert any(alert.rule == "throughput_spill" for alert in alerts)


def test_parent_kl_aborts_after_three_consecutive_probes():
    config = MonitorConfig(warn_repeat_tokens=0)
    rules = MonitorRules(config)
    alerts = []
    for index in range(3):
        probe = _healthy_probe(tokens=500_000 * (index + 1))
        probe["anchor_kl"]["4"] = 1.2
        alerts.extend(rules.observe_probe(probe))
    assert not any(alert.rule == "parent_kl_runaway" for alert in alerts[:-1])
    assert alerts[-1].rule == "parent_kl_runaway"
    assert alerts[-1].severity == "abort"


def test_overfitting_gap_and_generation_sanity_warn():
    config = MonitorConfig(warn_repeat_tokens=0)
    rules = MonitorRules(config)
    first = []
    for index, gap in enumerate((0.01, 0.02)):
        probe = _healthy_probe(tokens=500_000 * (index + 1))
        probe["teacher_kl_gap"]["4"] = gap
        first.extend(rules.observe_probe(probe))
    assert not any(alert.rule == "overfitting_gap" for alert in first)
    probe = _healthy_probe(tokens=1_500_000)
    probe["teacher_kl_gap"]["4"] = 0.03
    alerts = rules.observe_probe(probe)
    assert any(alert.rule == "overfitting_gap" for alert in alerts)

    probe = _healthy_probe(tokens=2_000_000)
    probe["generation"][0]["answer_present"] = False
    probe["generation"][0]["repetition_loop"] = True
    alerts = rules.observe_probe(probe)
    assert any(alert.rule == "generation_sanity" for alert in alerts)


def test_exit1_mismatch_aborts():
    rules = MonitorRules(MonitorConfig())
    probe = _healthy_probe()
    probe["invariant"] = {"applicable": True, "ok": False, "max_abs_diff": 0.1}
    alerts = rules.observe_probe(probe)
    assert alerts[0].rule == "exit1_bit_exact"
    assert alerts[0].severity == "abort"


# ---------------------------------------------------------------------------
# Bit-exactness comparison (perturbed exit 1)
# ---------------------------------------------------------------------------


class _FakeTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace

        self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        self._gemmaloop_loop_config = SimpleNamespace(ple_mode="shared", n_loops=1)
        self._gemmaloop_plan = object()
        self._gemmaloop_n_loops = 1
        self.perturb = False

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, exits=None):
        value = input_ids.float().unsqueeze(-1)
        if exits is not None and self.perturb:
            value = value + 1e-3
        return type("Output", (), {"last_hidden_state": value, "exit_hidden_states": {1: value}})()


class _FakeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeTextModel()

    def disable_adapter(self):
        from contextlib import nullcontext

        return nullcontext()


def test_exit1_probe_passes_and_catches_a_perturbation():
    wrapper = _FakeWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    result = exit1_bit_exact(wrapper, wrapper.model, batch)
    assert result["applicable"] is True
    assert result["ok"] is True
    assert result["max_abs_diff"] == 0.0

    wrapper.model.perturb = True
    result = exit1_bit_exact(wrapper, wrapper.model, batch)
    assert result["ok"] is False
    assert result["max_abs_diff"] == pytest.approx(1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# Monitoring off / enabled is observation-only
# ---------------------------------------------------------------------------


def test_monitor_callback_does_not_change_training(tmp_path):
    from tests.train.test_dial_trainer import _trainer

    baseline = _trainer(tmp_path / "baseline", dial_enabled=False, max_steps=3)
    baseline.train()
    baseline_losses = [
        float(entry["loss"])
        for entry in baseline.state.log_history
        if "loss" in entry
    ]

    monitored = _trainer(tmp_path / "monitored", dial_enabled=False, max_steps=3)
    monitor = HealMonitor(tmp_path / "monitored")
    callback = HealMonitorCallback(monitor, monitored)
    monitored.add_callback(callback)
    monitored.train()
    monitored_losses = [
        float(entry["loss"])
        for entry in monitored.state.log_history
        if "loss" in entry
    ]
    assert monitored_losses == baseline_losses
    assert monitor.metrics_path.is_file()
    lines = monitor.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    last = json.loads(lines[-1])
    assert last["grad_groups"]["lora"]["grad_norm"] > 0
    assert last["grad_norm_total"] is not None


def test_install_disabled_touches_nothing(tmp_path):
    from types import SimpleNamespace

    from tests.train.test_dial_trainer import _trainer

    trainer = _trainer(tmp_path / "run", dial_enabled=False, max_steps=1)
    args = SimpleNamespace(out=tmp_path / "run", seed=0, smoke=True, vram_cap=0.0)
    from gemmaloop.train.heal_monitor import install_heal_monitor

    assert (
        install_heal_monitor(
            trainer,
            args,
            dataset=trainer.train_dataset,
            tokenizer=None,
            enabled=False,
        )
        is None
    )
    callbacks = getattr(trainer.callback_handler, "callbacks", [])
    assert not any(isinstance(callback, HealMonitorCallback) for callback in callbacks)


# ---------------------------------------------------------------------------
# Probe runner against a direct full-softmax computation
# ---------------------------------------------------------------------------


def _tiny_looped_text_model():
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from tests.model.tiny import tiny_text_model

    model = tiny_text_model(seed=0)
    apply_loops(
        model,
        make_layer_plan(model.config.to_dict()),
        LoopConfig(
            n_loops=1,
            bridge="gch",
            ple_mode="shared",
            adapter_policy="loop_only",
            lora_target_layers="loop_block",
        ),
    )
    return model


def test_probe_runner_anchor_teacher_and_nll_match_direct_softmax(tmp_path):
    from gemmaloop.train.heal_probes import ProbeConfig, ProbeRunner, nll_from_hidden

    text_model = _tiny_looped_text_model()
    weight = text_model.embed_tokens.weight.detach().float().clone()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]])
    k = 4
    probs = torch.full((1, 6, k), 0.2)
    indices = torch.tensor([[[0, 1, 2, 3]] * 6])
    tail = torch.full((1, 6), 0.2)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "teacher_topk_probs": probs,
        "teacher_topk_indices": indices,
        "teacher_tail_prob": tail,
    }
    runner = ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=weight,
        softcap=None,
        batches=[batch],
        config=ProbeConfig(
            anchor_ns=(2,),
            eval_ns=(1, 2),
            nll_ns=(1, 2),
            generation_ns=(),
        ),
        text_dir=None,
    )
    result = runner.run(
        tokens_seen=500_000, step=10, train_teacher_kl={"2": 0.5}
    )
    assert result["invariant"]["ok"] is True
    assert result["teacher_kl_gap"]["2"] == pytest.approx(
        result["teacher_kl"]["2"] - 0.5
    )

    from gemmaloop.model.looped import set_n_loops

    def hidden(n: int) -> torch.Tensor:
        set_n_loops(text_model, n)
        with torch.no_grad():
            return text_model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
            ).last_hidden_state

    parent = hidden(1)
    student = hidden(2)
    mask = labels[:, 1:].reshape(-1) != -100
    logits_p = parent[:, :-1].reshape(-1, 64)[mask].float() @ weight.t()
    logits_s = student[:, :-1].reshape(-1, 64)[mask].float() @ weight.t()
    logp = torch.log_softmax(logits_p, dim=-1)
    logs = torch.log_softmax(logits_s, dim=-1)
    expected_anchor = float((logp.exp() * (logp - logs)).sum(dim=-1).mean())
    assert result["anchor_kl"]["2"] == pytest.approx(expected_anchor, abs=1e-4)

    teacher_probs = probs[:, 1:].reshape(-1, k)[mask].float()
    teacher_tail = tail[:, 1:].reshape(-1)[mask].float()
    teacher_ids = indices[:, 1:].reshape(-1, k)[mask]
    student_at_ids = torch.gather(logs, 1, teacher_ids.reshape(-1, k))
    topk_kl = (teacher_probs * (torch.log(teacher_probs) - student_at_ids)).sum(dim=-1)
    q_mass = torch.exp(student_at_ids).sum(dim=-1)
    tail_kl = teacher_tail * (torch.log(teacher_tail) - torch.log1p(-q_mass))
    expected_teacher = float((topk_kl + tail_kl).mean())
    assert result["teacher_kl"]["2"] == pytest.approx(expected_teacher, abs=1e-4)

    targets = labels[:, 1:].reshape(-1)[mask]
    chosen_t = float(result["calibrated_nll_temperature"]["2"])
    logits_at_t = logits_s / chosen_t
    expected_nll = float(
        torch.logsumexp(logits_at_t, dim=-1)
        .sub(logits_at_t.gather(1, targets.unsqueeze(1)).squeeze(1))
        .mean()
    )
    assert result["calibrated_nll"]["2"] == pytest.approx(expected_nll, abs=2e-4)
    assert result["calibrated_nll_fit"]["2"] == pytest.approx(expected_nll, abs=2e-4)
    assert chosen_t in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    assert torch.isfinite(torch.tensor(result["calibrated_nll"]["1"]))
    assert targets.numel() == 4


# ---------------------------------------------------------------------------
# Per-step record against a fake trainer state
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self):
        self.per_exit_hinge = {2: torch.tensor([0.0, 1.0]), 3: torch.tensor([0.0, 0.0])}
        self.row_token_counts = torch.tensor([2.0, 2.0])


def _fake_trainer_state():
    from types import SimpleNamespace

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Module()
    model.layers[0].q_proj.lora_B = torch.nn.Linear(2, 2, bias=False)
    model.loop_bridge_g = torch.nn.Parameter(torch.tensor([0.0, 0.6, 0.7, 0.8]))
    for name in ("layers.0.q_proj.lora_B.weight", "loop_bridge_g"):
        parameter = dict(model.named_parameters())[name]
        parameter.requires_grad_(True)
        parameter.grad = torch.ones_like(parameter)

    optimizer = SimpleNamespace(
        param_groups=[
            {"name": "lora_decay", "lr": 2e-4},
            {"name": "bridge", "lr": 3e-3},
        ]
    )
    trainer = SimpleNamespace(
        model=model,
        loop_text_model=model,
        optimizer=optimizer,
        _last_dial_metrics={
            "dial_total": 2.5,
            "dial_horizon": 4,
            "dial_exit2_loss": 2.4,
            "dial_exit2_kl_teacher": 1.0,
            "dial_exit2_kl_parent": 0.10,
            "dial_exit2_hinge": 0.0,
            "dial_exit2_hidden_change": 0.4,
            "dial_exit2_period2": 0.1,
            "dial_exit3_loss": 2.6,
            "dial_exit3_kl_teacher": 1.1,
            "dial_exit3_kl_parent": 0.11,
            "dial_exit3_hinge": 0.1,
            "dial_exit3_hidden_change": 0.3,
            "dial_exit3_period2": 0.2,
        },
        _last_heal_metrics={},
        _last_grad_norm_total=2.0,
        _last_clip_fired=True,
        clip_rate=0.5,
        _supervised_tokens_per_step=[12],
        dial_horizon_probs=(0.0, 0.0, 0.0, 0.7, 0.075, 0.075, 0.075, 0.075),
        state=SimpleNamespace(log_history=[{"loss": 2.5}]),
    )
    return trainer


def test_record_step_matches_direct_metrics(tmp_path):
    trainer = _fake_trainer_state()
    monitor = HealMonitor(tmp_path / "run")
    monitor.observe_dial_result(_FakeResult())
    record = monitor.record_step(
        trainer, step=7, tokens_seen=500_000, step_seconds=2.0, memory_cap_gib=10.0
    )
    assert record["loss"] == pytest.approx(2.5)
    assert record["horizon"] == 4
    assert record["exits"][2]["kl_teacher"] == pytest.approx(1.0)
    assert record["exits"][2]["hinge_active_fraction"] == pytest.approx(0.5)
    assert record["grad_norm_total"] == pytest.approx(2.0)
    assert record["grad_groups"]["lora"]["lr"] == pytest.approx(2e-4)
    assert record["grad_groups"]["bridge"]["lr"] == pytest.approx(3e-3)
    assert record["grad_groups"]["bridge"]["update_to_weight"] > 0
    assert record["bridge_g"]["saturation_fraction"] == pytest.approx(0.75)
    assert record["horizon_fractions"]["4"] == pytest.approx(1.0)
    assert record["registered_horizon"]["4"] == pytest.approx(0.7)
    assert record["supervised_tokens"] == 12
    assert record["clip_fired"] is True
    assert record["tok_per_s"] == pytest.approx(250_000.0)
    assert record["nonfinite"] == {"loss": 0, "grads": 0, "logits": 0}
    line = (tmp_path / "run" / "heal_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(line) == 1
    assert json.loads(line[0])["step"] == 7


def test_callback_probe_runs_at_end_without_changing_training(tmp_path):
    from tests.train.test_dial_trainer import _trainer

    baseline = _trainer(tmp_path / "baseline2", dial_enabled=False, max_steps=2)
    baseline.train()
    baseline_losses = [
        float(entry["loss"]) for entry in baseline.state.log_history if "loss" in entry
    ]

    monitored = _trainer(tmp_path / "monitored2", dial_enabled=False, max_steps=2)
    monitor = HealMonitor(tmp_path / "monitored2")

    class FakeRunner:
        def run(self, *, tokens_seen, step, train_teacher_kl=None):
            return {
                "kind": "probe",
                "step": step,
                "tokens_seen": tokens_seen,
                "invariant": {"applicable": True, "ok": True},
                "anchor_kl": {"4": 0.1},
                "teacher_kl": {"4": 1.0},
                "teacher_kl_gap": {"4": 0.01},
                "calibrated_nll": {"1": 2.0},
                "generation": [],
            }

    monitored.add_callback(
        HealMonitorCallback(monitor, monitored, probe_factory=lambda: FakeRunner())
    )
    monitored.train()
    monitored_losses = [
        float(entry["loss"])
        for entry in monitored.state.log_history
        if "loss" in entry
    ]
    assert monitored_losses == baseline_losses
    assert monitor.probes_path.is_file()
    assert len(monitor.probes_path.read_text(encoding="utf-8").splitlines()) == 1


def test_probe_generation_sanity_stores_texts(tmp_path):
    from gemmaloop.train.heal_probes import ProbeConfig, ProbeRunner

    text_model = _tiny_looped_text_model()
    weight = text_model.embed_tokens.weight.detach().float().clone()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def decode(self, ids, skip_special_tokens=False):
            return "<|channel>thought\nx<channel|>The answer is 7<turn|>"

    runner = ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=weight,
        softcap=None,
        batches=[],
        config=ProbeConfig(
            eval_ns=(1,),
            anchor_ns=(),
            nll_ns=(1,),
            generation_ns=(1, 2),
            generation_prompts=2,
            generation_max_new_tokens=4,
            generation_batch_size=2,
        ),
        tokenizer=Tokenizer(),
        prompts=[torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        text_dir=tmp_path / "texts",
        generate_fn=lambda prompt, n, cap: [7, 2],
    )
    result = runner.run(tokens_seen=500_000, step=3)
    assert len(result["generation"]) == 4
    assert all(row["thinking_closed"] for row in result["generation"])
    assert all(row["answer_present"] for row in result["generation"])
    assert all(row["text_path"] and Path(row["text_path"]).is_file() for row in result["generation"])
    assert all(row["finish_reason"] == "eos" for row in result["generation"])


# ---------------------------------------------------------------------------
# C1: install on a real [B,S,K] teacher batch
# ---------------------------------------------------------------------------


def test_install_on_real_teacher_batch_does_not_abort_and_corruption_aborts(tmp_path):
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    trainer = _trainer(
        tmp_path / "teacher-run", dial_enabled=True, teacher=True, max_steps=2
    )
    config = MonitorConfig(
        monitor_rows=4,
        probe_interval_tokens=10**12,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1, 2),
    )
    args = SimpleNamespace(out=tmp_path / "teacher-run", seed=0, smoke=True, vram_cap=0.0)
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=getattr(trainer, "data_collator", None),
        enabled=True,
        config=config,
    )
    assert callback is not None
    trainer.train()
    monitor = callback.monitor
    assert monitor.rules.abort_triggered is False
    lines = monitor.metrics_path.read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        integrity = json.loads(line)["teacher_integrity"]
        assert integrity["ok"] is True
        assert integrity["max_abs_deviation_before"] <= 2e-3
    assert not any(alert.severity == "abort" for alert in monitor.alerts)

    # A real collated [B,S,K] batch with a corrupted mass must abort.
    collator = getattr(trainer, "data_collator", None)
    rows = [trainer.train_dataset[index] for index in range(2)]
    corrupted = dict(collator([dict(row) for row in rows]))
    corrupted["teacher_topk_probs"] = corrupted["teacher_topk_probs"] * 0.5
    monitor.observe_batch(corrupted, trainer.model)
    monitor.record_step(
        trainer,
        step=int(trainer.state.global_step) + 1,
        tokens_seen=int(trainer.tokens_seen) + 1,
        step_seconds=0.1,
    )
    assert monitor.rules.abort_triggered is True
    assert any(alert.rule == "teacher_integrity" for alert in monitor.alerts)


# ---------------------------------------------------------------------------
# C3: registered-run MONITOR split is fail-closed
# ---------------------------------------------------------------------------


def _registered_args(tmp_path, **overrides):
    from types import SimpleNamespace

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    values = {
        "out": tmp_path / "run",
        "seed": 0,
        "smoke": False,
        "prereg": str(tmp_path / "prereg.md"),
        "pilot_arm": None,
        "data": data,
        "vram_cap": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_registered_splits_reads_the_pilot_reserved_ids_contract(tmp_path):
    from gemmaloop.train.heal_monitor import load_registered_splits

    data = tmp_path / "data"
    data.mkdir()
    (data / "reserved_ids.json").write_text(
        json.dumps(
            {
                "monitor": ["m1", "m2"],
                "pilot_eval": ["e1"],
                "pilot_fit": ["f1"],
                "heal_train": ["t1"],
                "confirmatory": ["c1"],
                "unknown_extra_split": ["x1"],
            }
        ),
        encoding="utf-8",
    )
    args = _registered_args(tmp_path)
    reserved, training, monitor, sources = load_registered_splits(args)
    assert reserved == {"e1", "f1", "c1", "x1"}
    assert training == {"t1"}
    assert monitor == {"m1", "m2"}
    assert sources == [str(data / "reserved_ids.json")]


def test_load_registered_splits_subtracts_monitor_from_legacy_reserved_union(tmp_path):
    from gemmaloop.train.heal_monitor import load_registered_splits

    data = tmp_path / "data"
    data.mkdir()
    (data / "reserved_ids.json").write_text(
        json.dumps(
            {
                "reserved_ids": ["m1", "e1", "t1"],
                "splits": {
                    "monitor": ["m1"],
                    "evaluation": ["e1"],
                    "heal": ["t1"],
                },
            }
        ),
        encoding="utf-8",
    )
    reserved, training, monitor, _sources = load_registered_splits(
        _registered_args(tmp_path)
    )
    assert monitor == {"m1"}
    assert training == {"t1"}
    assert reserved == {"e1"}


def test_registered_install_fails_closed_without_reserved_ids(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.train import heal_monitor as monitor_module

    class DummyDataset:
        column_names = ["row_id", "input_ids", "labels"]

        def __len__(self):
            return 1

    trainer = SimpleNamespace(
        dial_enabled=True,
        args=SimpleNamespace(output_dir=str(tmp_path / "run")),
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer,
            _registered_args(tmp_path),
            dataset=DummyDataset(),
            tokenizer=object(),
            rows=[{"messages": []}],
            enabled=True,
        )
    assert excinfo.value.code == 2


def test_registered_install_fails_closed_on_unreadable_reserved(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.train import heal_monitor as monitor_module

    class DummyDataset:
        column_names = ["row_id", "input_ids", "labels"]

        def __len__(self):
            return 1

    trainer = SimpleNamespace(
        dial_enabled=True,
        args=SimpleNamespace(output_dir=str(tmp_path / "run")),
    )
    args = _registered_args(
        tmp_path, monitor_reserved=str(tmp_path / "missing-reserved.json")
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer, args, dataset=DummyDataset(), tokenizer=object(), enabled=True
        )
    assert excinfo.value.code == 2


def test_registered_install_requires_raw_rows(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.train import heal_monitor as monitor_module

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "splits.json").write_text(json.dumps({"reserved_ids": ["r1"]}), encoding="utf-8")

    class DummyDataset:
        column_names = ["row_id", "input_ids", "labels"]

        def __len__(self):
            return 1

    trainer = SimpleNamespace(
        dial_enabled=True,
        args=SimpleNamespace(output_dir=str(tmp_path / "run")),
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer, _registered_args(tmp_path), dataset=DummyDataset(), enabled=True
        )
    assert excinfo.value.code == 2



# ---------------------------------------------------------------------------
# I1/I2/I3: pre-clip grads, no abort checkpoint, clean abort state on resume
# ---------------------------------------------------------------------------


def test_group_gradients_are_captured_before_clipping(tmp_path):
    from types import SimpleNamespace

    from gemmaloop.train.heal_monitor import _install_capture

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Module()
    model.layers[0].q_proj.lora_A = torch.nn.Linear(2, 2, bias=False)
    parameter = dict(model.named_parameters())["layers.0.q_proj.lora_A.weight"]
    parameter.requires_grad_(True)
    parameter.grad = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    def clip(model, *args, **kwargs):  # noqa: ANN001 - fake HF hook
        del args, kwargs
        for value in model.parameters():
            if value.grad is not None:
                value.grad.mul_(0.1)
        return torch.tensor(0.5)

    trainer = SimpleNamespace(
        compute_loss=lambda *args, **kwargs: None,
        _clip_grad_norm=clip,
        _dial_compute_loss=None,
    )
    monitor = HealMonitor(tmp_path / "run")
    _install_capture(trainer, monitor)
    trainer._clip_grad_norm(model)
    pending = monitor._pending_grad_groups
    assert pending["lora"]["grad_norm"] == pytest.approx(5.0)
    assert pending["lora"]["weight_norm"] > 0


def test_abort_never_saves_a_checkpoint(tmp_path):
    from types import SimpleNamespace

    monitor = HealMonitor(tmp_path / "run")
    saves: list[object] = []

    def save(model, trial):  # noqa: ANN001 - fake trainer
        saves.append((model, trial))

    trainer = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        _save_checkpoint=save,
        exit_code=None,
        exit_reason=None,
    )
    callback = HealMonitorCallback(monitor, trainer)
    control = SimpleNamespace(should_save=True, should_training_stop=False)
    with pytest.raises(SystemExit) as excinfo:
        callback._abort(control)
    assert excinfo.value.code == monitor.config.abort_exit_code
    assert saves == []
    assert control.should_save is False
    assert control.should_training_stop is True


def test_persisted_abort_state_is_cleared_on_a_new_process():
    rules = MonitorRules(MonitorConfig())
    rules.load_state_dict(
        {
            "abort_triggered": True,
            "abort_emitted": ["nonfinite"],
            "loss_spike_run": 7,
            "parent_kl_run": 3,
            "warn_last": {"nonfinite": 123},
            "losses": [1.0, 2.0],
        }
    )
    assert rules.abort_triggered is False
    assert rules._abort_emitted == set()
    assert rules._loss_spike_run == 0
    assert rules._parent_kl_run == 0
    assert rules._warn_last == {}
    assert list(rules._losses) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# I4/I5: bridge saturation ungated; micro-batches are token-weighted
# ---------------------------------------------------------------------------


def test_bridge_saturation_warns_before_one_million_tokens():
    rules = MonitorRules(MonitorConfig(warn_repeat_tokens=0))
    alerts = rules.observe_step(
        _step(
            500_000,
            bridge_g={
                "available": True,
                "saturation_fraction": 0.2,
                "all_zero": False,
            },
        )
    )
    assert [alert.rule for alert in alerts] == ["bridge_saturated"]

    zero_rules = MonitorRules(MonitorConfig(warn_repeat_tokens=0))
    zero_alerts = zero_rules.observe_step(
        _step(
            500_000,
            bridge_g={
                "available": True,
                "saturation_fraction": 0.0,
                "all_zero": True,
            },
        )
    )
    assert not any(alert.rule == "bridge_zero" for alert in zero_alerts)


def test_bridge_stats_use_configurable_thresholds():
    g = torch.tensor([0.2, 0.45, 0.55, 0.9])
    stats = bridge_g_stats(g, active_abs=0.3, saturation_abs=0.5)
    assert stats["active_fraction"] == pytest.approx(3 / 4)
    assert stats["saturation_fraction"] == pytest.approx(2 / 4)


def test_step_metrics_are_token_weighted_across_microbatches(tmp_path):
    trainer = _fake_trainer_state()
    monitor = HealMonitor(tmp_path / "run")
    trainer._last_dial_metrics = {
        "dial_total": 1.0,
        "dial_tokens": 10,
        "dial_rows": 1,
        "dial_horizon": 4,
        "dial_exit2_kl_teacher": 1.0,
        "dial_exit2_hinge": 0.0,
    }
    class _Result:
        def __init__(self, active):
            self.per_exit_hinge = {2: torch.tensor([float(active)])}
            self.row_token_counts = torch.tensor([1.0])

    monitor.observe_dial_result(_Result(0))
    monitor.observe_microbatch(trainer)
    trainer._last_dial_metrics = {
        "dial_total": 3.0,
        "dial_tokens": 30,
        "dial_rows": 1,
        "dial_horizon": 8,
        "dial_exit2_kl_teacher": 2.0,
        "dial_exit2_hinge": 0.5,
    }
    monitor.observe_dial_result(_Result(1))
    monitor.observe_microbatch(trainer)
    record = monitor.record_step(trainer, step=1, tokens_seen=40, step_seconds=1.0)
    assert record["loss"] == pytest.approx((1.0 * 10 + 3.0 * 30) / 40)
    assert record["exits"][2]["kl_teacher"] == pytest.approx(
        (1.0 * 10 + 2.0 * 30) / 40
    )
    assert record["exits"][2]["hinge_active_fraction"] == pytest.approx(0.5)
    assert record["horizon"] == 8
    assert record["horizon_histogram"] == {"4": 1, "8": 1}
    assert record["horizon_fractions"] == {"4": 0.5, "8": 0.5}


# ---------------------------------------------------------------------------
# I7: monitor NLL is calibrated on two deterministic MONITOR halves
# ---------------------------------------------------------------------------


def test_chunked_nll_temperature_matches_direct_softmax():
    from gemmaloop.train.heal_probes import chunked_nll_hidden

    torch.manual_seed(0)
    hidden = torch.randn(3, 4)
    weight = torch.randn(7, 4)
    targets = torch.tensor([0, 4, 6])
    for temperature in (0.5, 1.0, 2.5):
        logits = hidden @ weight.t() / temperature
        expected = -torch.log_softmax(logits, dim=-1)[torch.arange(3), targets]
        actual = chunked_nll_hidden(
            hidden,
            weight,
            targets,
            temperature=temperature,
            token_chunk=1,
            vocab_chunk=2,
        )
        assert torch.allclose(actual, expected, atol=1e-5)


def test_split_monitor_halves_is_deterministic_and_disjoint():
    from gemmaloop.train.heal_monitor import split_monitor_halves

    indices = list(range(10))
    keys = {index: f"row{index}" for index in indices}
    first = split_monitor_halves(indices, seed=3, keys=keys)
    second = split_monitor_halves(list(reversed(indices)), seed=3, keys=keys)
    assert first == second
    fit, score = first
    assert set(fit).isdisjoint(score)
    assert set(fit) | set(score) == set(indices)
    assert len(fit) == 5 and len(score) == 5


# ---------------------------------------------------------------------------
# I6: monitor ON vs OFF is bit-identical (params, RNG, data order)
# ---------------------------------------------------------------------------


def _snapshot_training(trainer):
    return {
        "params": [parameter.detach().clone() for parameter in trainer.model.parameters()],
        "rng": torch.get_rng_state().clone(),
        "micro_batches_seen": int(trainer.micro_batches_seen),
        "micro_step": int(trainer._micro_step),
        "tokens_seen": int(trainer.tokens_seen),
        "global_step": int(trainer.state.global_step),
    }


def test_monitor_on_vs_off_is_bit_identical_with_real_probes(tmp_path):
    import random
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    baseline = _trainer(
        tmp_path / "bit-baseline", dial_enabled=True, teacher=True, max_steps=3
    )
    monitored = _trainer(
        tmp_path / "bit-monitored", dial_enabled=True, teacher=True, max_steps=3
    )
    initial = [parameter.detach().clone() for parameter in baseline.model.parameters()]
    assert all(
        torch.equal(left, right)
        for left, right in zip(initial, monitored.model.parameters())
    )

    config = MonitorConfig(
        monitor_rows=2,
        probe_interval_tokens=1,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1, 2),
        nll_temperatures=(1.0,),
    )
    args = SimpleNamespace(
        out=tmp_path / "bit-monitored", seed=0, smoke=True, vram_cap=0.0
    )
    callback = monitor_module.install_heal_monitor(
        monitored,
        args,
        dataset=monitored.train_dataset,
        tokenizer=getattr(monitored, "processing_class", None),
        data_collator=getattr(monitored, "data_collator", None),
        enabled=True,
        config=config,
    )
    assert callback is not None

    torch.manual_seed(1234)
    random.seed(1234)
    baseline.train()
    baseline_snapshot = _snapshot_training(baseline)
    baseline_python_rng = random.getstate()

    torch.manual_seed(1234)
    random.seed(1234)
    monitored.train()
    monitored_snapshot = _snapshot_training(monitored)

    assert monitored_snapshot["micro_batches_seen"] == baseline_snapshot["micro_batches_seen"]
    assert monitored_snapshot["micro_step"] == baseline_snapshot["micro_step"]
    assert monitored_snapshot["tokens_seen"] == baseline_snapshot["tokens_seen"]
    assert monitored_snapshot["global_step"] == baseline_snapshot["global_step"]
    assert torch.equal(monitored_snapshot["rng"], baseline_snapshot["rng"])
    assert random.getstate() == baseline_python_rng
    for left, right in zip(
        baseline_snapshot["params"], monitored_snapshot["params"]
    ):
        assert torch.equal(left, right)
    probe_lines = callback.monitor.probes_path.read_text(encoding="utf-8").splitlines()
    assert len(probe_lines) >= 3


def test_probe_errors_are_warned_not_silently_ignored():
    rules = MonitorRules(MonitorConfig(warn_repeat_tokens=0))
    probe = _healthy_probe()
    probe["probe_errors"] = ["generation: boom"]
    alerts = rules.observe_probe(probe)
    assert any(alert.rule == "probe_error" for alert in alerts)


# ---------------------------------------------------------------------------
# C1 (fix round 2/3): --monitor-rows is a separate held-out Stage-0 corpus
# ---------------------------------------------------------------------------


def _write_monitor_corpus(tmp_path, problem_ids):
    path = tmp_path / "monitor" / "monitor_rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q {problem_id}"},
                {"role": "assistant", "content": "answer"},
            ],
            "problem_id": problem_id,
        }
        for problem_id in problem_ids
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_registered_install_accepts_a_disjoint_monitor_rows_file_and_runs(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from datasets import Dataset

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "monitor": ["monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"],
                "pilot_eval": ["pilot-r0"],
                "pilot_fit": ["tfit-r1"],
                "heal_train": ["train-r0", "train-r1"],
                "confirmatory": ["confirm-r2"],
            }
        ),
        encoding="utf-8",
    )
    monitor_path = _write_monitor_corpus(
        tmp_path, ["monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"]
    )
    # Tokenization is covered by the sampler tests; for the install path we
    # only need a tiny in-vocabulary dataset with the original row_ids.
    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5, 6],
                    "labels": [-100, -100, 5, 6],
                    "attention_mask": [1, 1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    trainer = _trainer(
        tmp_path / "registered-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "registered-run",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=monitor_path,
        vram_cap=0.0,
    )
    config = MonitorConfig(
        monitor_rows=2,
        probe_interval_tokens=10**12,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1, 2),
        nll_temperatures=(1.0,),
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=getattr(trainer, "data_collator", None),
        rows=[
            {"problem_id": "train-r0", "messages": []},
            {"problem_id": "train-r1", "messages": []},
        ],
        max_length=32,
        enabled=True,
        config=config,
    )
    assert callback is not None
    trainer.train()
    split = json.loads(
        (tmp_path / "registered-run" / "heal_monitor_split.json").read_text(
            encoding="utf-8"
        )
    )
    assert split["probe_source"] == "monitor_rows_file"
    assert split["monitor_problem_ids"] == [
        "monitor-m0",
        "monitor-m1",
        "monitor-m2",
        "monitor-m3",
    ]
    assert split["registered"] is True
    assert len(split["problem_ids"]) == config.monitor_rows
    assert set(split["problem_ids"]).issubset(
        {"monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"}
    )
    assert split["teacher_fit"] == "unavailable"
    assert callback.monitor.rules.abort_triggered is False


def test_registered_install_refuses_monitor_rows_overlapping_reserved(tmp_path):
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module

    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    monitor_path = _write_monitor_corpus(tmp_path, ["shared-id"])
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "monitor": ["shared-id"],
                "pilot_eval": ["shared-id"],
                "heal_train": ["train-r0"],
                "confirmatory": ["c0"],
            }
        ),
        encoding="utf-8",
    )
    trainer = _trainer(
        tmp_path / "overlap-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "overlap-run",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=monitor_path,
        vram_cap=0.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer,
            args,
            dataset=trainer.train_dataset,
            tokenizer=getattr(trainer, "processing_class", None),
            data_collator=getattr(trainer, "data_collator", None),
            rows=[{"problem_id": "train-r0", "messages": []}],
            max_length=32,
            enabled=True,
            config=MonitorConfig(monitor_rows=1),
        )
    assert excinfo.value.code == 2


def test_registered_install_refuses_missing_monitor_rows(tmp_path):
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps({"pilot_eval": ["r0"], "heal_train": ["t0"]}), encoding="utf-8"
    )
    trainer = _trainer(
        tmp_path / "missing-monitor", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "missing-monitor",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=None,
        vram_cap=0.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer,
            args,
            dataset=trainer.train_dataset,
            tokenizer=getattr(trainer, "processing_class", None),
            rows=[{"problem_id": "train-r0", "messages": []}],
            max_length=32,
            enabled=True,
            config=MonitorConfig(monitor_rows=1),
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# I1 (fix round 2): padded/masked positions in ragged teacher batches
# ---------------------------------------------------------------------------


def test_ragged_teacher_batch_ignores_padding_and_still_aborts_real_corruption(
    tmp_path,
):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from gemmaloop.train.teacher_collator import TeacherDataCollator
    from tests.train.test_trainer_tiny import _tiny_model

    k = 3
    probs = [[0.3, 0.3, 0.2] for _ in range(5)]
    indices = [[0, 1, 2] for _ in range(5)]
    tails = [0.2 for _ in range(5)]
    rows = [
        {
            "input_ids": [3, 4, 5],
            "labels": [-100, 5, 6],
            "attention_mask": [1, 1, 1],
            "teacher_topk_probs": probs[:3],
            "teacher_topk_indices": indices[:3],
            "teacher_tail_prob": tails[:3],
        },
        {
            "input_ids": [3, 4, 5, 6, 7],
            "labels": [-100, 5, 6, 7, 8],
            "attention_mask": [1, 1, 1, 1, 1],
            "teacher_topk_probs": probs,
            "teacher_topk_indices": indices,
            "teacher_tail_prob": tails,
        },
    ]
    collator = TeacherDataCollator(
        DataCollatorForLanguageModeling(pad_token_id=0)
    )
    batch = collator([dict(row) for row in rows])
    assert batch["teacher_topk_probs"].shape[:2] == (2, 5)
    assert int(batch["attention_mask"][0].sum()) == 3

    monitor = HealMonitor(tmp_path / "run")
    monitor.observe_batch(batch, _tiny_model(loops=False))
    report = monitor._step_integrity[0]
    assert report["ok"] is True
    assert report["positions"] == 10
    assert report["valid_tokens"] == 6  # 2 real rows x 3 supervised tokens? see labels
    assert report["max_abs_deviation_before"] <= 2e-3

    corrupted = {
        key: (value.clone() if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    corrupted["teacher_topk_probs"][0, 1] = corrupted["teacher_topk_probs"][0, 1] * 3
    monitor.observe_batch(corrupted, _tiny_model(loops=False))
    assert monitor._step_integrity[-1]["ok"] is False


# ---------------------------------------------------------------------------
# I2 (fix round 2): two-half calibration is end-to-end and discriminating
# ---------------------------------------------------------------------------


class _FakeProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(2))

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        value = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return type("Output", (), {"last_hidden_state": value})()


def _calibration_runner(monkeypatch, fit_batches, score_batches):
    from gemmaloop.train import heal_probes

    def fake_nll_temperatures(
        hidden,
        labels,
        weight,
        temperatures,
        *,
        softcap=None,
        token_chunk=1024,
        vocab_chunk=16384,
    ):
        del hidden, weight, softcap, token_chunk, vocab_chunk
        batch_marker = int(labels[0, 0])
        optimum = 2.0 if batch_marker == 1 else 3.0
        tokens = 10 if batch_marker == 1 else 30
        return {
            float(temperature): (float(temperature - optimum) ** 2, tokens)
            for temperature in temperatures
        }

    monkeypatch.setattr(
        heal_probes, "nll_from_hidden_temperatures", fake_nll_temperatures
    )
    return heal_probes.ProbeRunner(
        model=_FakeProbeModel(),
        text_model=_FakeProbeModel(),
        weight=torch.zeros(4, 2),
        softcap=None,
        batches=[
            {
                "input_ids": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[1, 1]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[2, 2]]),
                "labels": torch.tensor([[2, 2]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            },
        ],
        config=heal_probes.ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(1,),
            generation_ns=(),
            nll_temperatures=(1.0, 2.0, 3.0, 4.0),
        ),
        nll_fit_batches=fit_batches,
        nll_score_batches=score_batches,
    )


def test_two_half_calibration_fits_on_one_half_and_scores_on_the_other(monkeypatch):
    result = _calibration_runner(monkeypatch, [0], [1]).run(
        tokens_seen=500_000, step=1
    )
    # Fit half prefers T=2; the score half prefers T=3.  A fit==score mutant
    # chooses T=3, so this assertion is discriminating.
    assert result["calibrated_nll_temperature"]["1"] == 2.0
    assert result["calibrated_nll_fit"]["1"] == pytest.approx(0.0)
    assert result["calibrated_nll"]["1"] == pytest.approx(1.0)
    assert result["calibrated_nll_fit_tokens"]["1"] == 10
    assert result["calibrated_nll_tokens"]["1"] == 30

    mutant = _calibration_runner(monkeypatch, [0, 1], [0, 1]).run(
        tokens_seen=500_000, step=1
    )
    assert mutant["calibrated_nll_temperature"]["1"] == 3.0
    assert mutant["calibrated_nll_temperature"]["1"] != result[
        "calibrated_nll_temperature"
    ]["1"]


def test_nonfinite_microbatch_loss_is_not_silently_dropped(tmp_path):
    trainer = _fake_trainer_state()
    trainer._last_dial_metrics = {
        "dial_total": float("nan"),
        "dial_tokens": 10,
        "dial_horizon": 4,
    }
    monitor = HealMonitor(tmp_path / "run")
    monitor.observe_microbatch(trainer)
    record = monitor.record_step(
        trainer, step=1, tokens_seen=10, step_seconds=1.0
    )
    assert record["nonfinite"]["loss"] == 1
    assert monitor.rules.abort_triggered is True


def _monitor_teacher_collator():
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    base = DataCollatorForLanguageModeling(pad_token_id=0)

    def collator(rows):
        cleaned = [
            {key: value for key, value in dict(row).items() if key != "row_id"}
            for row in rows
        ]
        batch = base(cleaned)
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        k = 4
        batch.update(
            {
                "teacher_topk_probs": torch.full(
                    (batch_size, seq_len, k), 0.2, dtype=torch.float16
                ),
                "teacher_topk_indices": torch.zeros(
                    batch_size, seq_len, k, dtype=torch.int32
                ),
                "teacher_tail_prob": torch.full(
                    (batch_size, seq_len), 0.2, dtype=torch.float32
                ),
            }
        )
        return batch

    return collator


def test_monitor_teacher_collator_enables_teacher_fit_and_gap(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from datasets import Dataset

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "monitor": ["monitor-m0", "monitor-m1"],
                "pilot_eval": ["pilot-r0"],
                "pilot_fit": ["tfit-r1"],
                "heal_train": ["train-r0", "train-r1"],
                "confirmatory": ["confirm-r2"],
            }
        ),
        encoding="utf-8",
    )
    monitor_path = _write_monitor_corpus(tmp_path, ["monitor-m0", "monitor-m1"])

    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5, 6],
                    "labels": [-100, -100, 5, 6],
                    "attention_mask": [1, 1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    trainer = _trainer(
        tmp_path / "teacher-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "teacher-run",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=monitor_path,
        monitor_teacher_data=str(tmp_path / "teacher"),
        vram_cap=0.0,
    )
    config = MonitorConfig(
        monitor_rows=2,
        probe_interval_tokens=10**12,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1, 2),
        nll_temperatures=(1.0,),
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=getattr(trainer, "data_collator", None),
        monitor_data_collator=_monitor_teacher_collator(),
        rows=[
            {"problem_id": "train-r0", "messages": []},
            {"problem_id": "train-r1", "messages": []},
        ],
        max_length=32,
        enabled=True,
        config=config,
    )
    assert callback is not None
    trainer.train()
    split = json.loads(
        (tmp_path / "teacher-run" / "heal_monitor_split.json").read_text(
            encoding="utf-8"
        )
    )
    assert split["teacher_fit"] == "available"
    probe = json.loads(
        callback.monitor.probes_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert probe["teacher_fit_available"] is True
    assert math.isfinite(probe["teacher_kl"]["2"])
    assert "2" in probe["teacher_kl_gap"]
    assert math.isfinite(probe["teacher_kl_gap"]["2"])


def test_short_monitor_corpus_exits_2(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from datasets import Dataset

    import gemmaloop.train.heal_monitor as monitor_module

    monitor_path = _write_monitor_corpus(
        tmp_path, [f"monitor-{index}" for index in range(10)]
    )

    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5],
                    "labels": [-100, 4, 5],
                    "attention_mask": [1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    trainer = SimpleNamespace(
        dial_enabled=True,
        args=SimpleNamespace(output_dir=str(tmp_path / "short-run")),
    )
    args = SimpleNamespace(
        out=tmp_path / "short-run",
        seed=0,
        smoke=True,
        prereg=None,
        pilot_arm=None,
        monitor_rows=monitor_path,
        vram_cap=0.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer,
            args,
            dataset=SimpleNamespace(column_names=[], __len__=lambda self: 0),
            tokenizer=object(),
            max_length=32,
            enabled=True,
            config=MonitorConfig(monitor_rows=32),
        )
    assert excinfo.value.code == 2


def test_unreadable_monitor_corpus_exits_2(tmp_path):
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module

    trainer = SimpleNamespace(
        dial_enabled=True,
        args=SimpleNamespace(output_dir=str(tmp_path / "bad-run")),
    )
    args = SimpleNamespace(
        out=tmp_path / "bad-run",
        seed=0,
        smoke=True,
        prereg=None,
        pilot_arm=None,
        monitor_rows=tmp_path / "missing-monitor.jsonl",
        vram_cap=0.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        monitor_module.install_heal_monitor(
            trainer,
            args,
            dataset=SimpleNamespace(column_names=[], __len__=lambda self: 0),
            tokenizer=object(),
            max_length=32,
            enabled=True,
            config=MonitorConfig(monitor_rows=32),
        )
    assert excinfo.value.code == 2


def test_tokenize_monitor_rows_uses_the_real_assistant_label_path():
    from gemmaloop.train.heal_monitor import tokenize_monitor_rows

    class FakeTokenizer:
        unk_token_id = None
        added_tokens_encoder = {}

        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {
                "input_ids": [250, 0, 3, 4, 251, 250, 0, 5, 6, 251],
                "attention_mask": [1] * 10,
            }

        def convert_tokens_to_ids(self, token):
            return {"<|turn>": 250, "<turn|>": 251}.get(token)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [0] if text == "model" else [0]

    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "problem_id": "m0",
        }
    ]
    dataset = tokenize_monitor_rows(FakeTokenizer(), rows, [0], max_length=32)
    assert len(dataset) == 1
    assert dataset[0]["row_id"] == 0
    assert any(label != -100 for label in dataset[0]["labels"])
    assert dataset[0]["input_ids"] == [250, 0, 3, 4, 251, 250, 0, 5, 6, 251]


def test_train_cli_exposes_monitor_rows_and_monitor_teacher_data(tmp_path):
    import scripts.train as train_script

    args = train_script._parse_args(
        [
            "--arm", "L", "--size", "e2b", "--data", str(tmp_path / "data"),
            "--tokens", "1", "--seed", "0", "--out", str(tmp_path / "run"),
            "--monitor-rows", str(tmp_path / "monitor.jsonl"),
            "--monitor-teacher-data", str(tmp_path / "teacher"),
        ]
    )
    assert args.monitor_rows == tmp_path / "monitor.jsonl"
    assert args.monitor_teacher_data == tmp_path / "teacher"


def _write_monitor_teacher_index(
    tmp_path,
    name,
    *,
    teacher_key,
    prob_value,
    rows=4,
    positions=(2, 3),
):
    """A real on-disk teacher shard set for the fake monitor dataset."""
    from gemmaloop.data.teacher_topk import (
        ShardEntry,
        TeacherManifest,
        TeacherRow,
        write_teacher_shard,
    )

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    position_tensor = torch.tensor(list(positions), dtype=torch.int32)
    k = 2
    teacher_rows = []
    for row_id in range(rows):
        n = len(positions)
        teacher_rows.append(
            TeacherRow(
                row_id=row_id,
                positions=position_tensor,
                indices=torch.tensor([[0, 1]] * n, dtype=torch.int32),
                probs=torch.full((n, k), float(prob_value), dtype=torch.float16),
                tail=torch.full((n,), 1.0 - float(prob_value) * k, dtype=torch.float32),
            )
        )
    digest = write_teacher_shard(directory / "shard-00000.pt", teacher_rows)
    tokens = rows * len(positions)
    manifest = TeacherManifest(
        teacher={
            "key": teacher_key,
            "model_id": "google/gemma-4-E4B-it",
            "revision": "rev",
            "runtime": "bf16",
            "max_length": 16,
        },
        data_manifest_sha256="d" * 64,
        top_k=k,
        temperature=1.0,
        mask_token="",
        mask_token_id=None,
        rows_total=rows,
        tokens_total=tokens,
        shards=(ShardEntry(0, "shard-00000.pt", digest, tuple(range(rows)), tokens),),
    )
    manifest.write(directory / "teacher_manifest.json")
    return directory


def test_monitor_teacher_shards_drive_both_probe_halves(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from datasets import Dataset
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from gemmaloop.data.teacher_topk import TeacherIndex, TeacherTopKCollator

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "monitor": ["monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"],
                "pilot_eval": ["pilot-r0"],
                "pilot_fit": ["tfit-r1"],
                "heal_train": ["train-r0", "train-r1"],
                "confirmatory": ["confirm-r2"],
            }
        ),
        encoding="utf-8",
    )
    monitor_path = _write_monitor_corpus(
        tmp_path, ["monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"]
    )

    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5, 6],
                    "labels": [-100, -100, 5, 6],
                    "attention_mask": [1, 1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    monitor_index = TeacherIndex.from_directory(
        _write_monitor_teacher_index(
            tmp_path, "monitor-teacher", teacher_key="monitor-teacher", prob_value=0.4
        ),
        verify_shards=True,
    )
    training_index = TeacherIndex.from_directory(
        _write_monitor_teacher_index(
            tmp_path, "training-teacher", teacher_key="training-teacher", prob_value=0.1
        ),
        verify_shards=True,
    )
    base = DataCollatorForLanguageModeling(pad_token_id=0)
    training_collator = TeacherTopKCollator(training_index, base)

    captured: list[dict] = []
    original = monitor_module.build_probe_batches

    def spy(*args, **kwargs):
        batches = original(*args, **kwargs)
        captured.extend(batches)
        return batches

    monkeypatch.setattr(monitor_module, "build_probe_batches", spy)

    trainer = _trainer(
        tmp_path / "teacher-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "teacher-run",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=monitor_path,
        monitor_teacher_data=str(monitor_index.directory),
        vram_cap=0.0,
    )
    config = MonitorConfig(
        monitor_rows=4,
        probe_interval_tokens=10**12,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1, 2),
        nll_temperatures=(1.0,),
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=training_collator,
        monitor_data_collator=TeacherTopKCollator(monitor_index, base),
        monitor_teacher_index=monitor_index,
        rows=[
            {"problem_id": "train-r0", "messages": []},
            {"problem_id": "train-r1", "messages": []},
        ],
        max_length=32,
        enabled=True,
        config=config,
    )
    assert callback is not None
    trainer.train()

    # Both probe halves must be collated with the MONITOR teacher shards.
    assert captured
    for batch in captured:
        assert set(batch["teacher_manifest_sha256"]) == {
            monitor_index.manifest_sha256
        }
        assert set(batch["teacher_id"]) == {"monitor-teacher"}

    probe = json.loads(
        callback.monitor.probes_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert probe["teacher_fit_available"] is True
    assert math.isfinite(probe["teacher_kl"]["2"])
    assert "2" in probe["teacher_kl_gap"]
    assert math.isfinite(probe["teacher_kl_gap"]["2"])
    # 4 monitor rows x 2 supervised tokens: the fit half contributes too.
    assert probe["teacher_tokens"]["2"] == 8


def test_reserved_ids_payload_from_the_pilot_producer_installs_end_to_end(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from datasets import Dataset

    import gemmaloop.train.heal_monitor as monitor_module
    from gemmaloop.pilot.splits import reserved_ids_payload
    from tests.train.test_dial_trainer import _trainer

    data = tmp_path / "data"
    data.mkdir()
    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"problem_id": "train-r0", "messages": [{"role": "user", "content": "h"}]})
        + "\n",
        encoding="utf-8",
    )
    fit = tmp_path / "fit.jsonl"
    fit.write_text(
        json.dumps({"problem_id": "fit-r0", "input_ids": [1, 2]}) + "\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(
        json.dumps({"problem_id": "eval-r0", "input_ids": [3, 4]}) + "\n",
        encoding="utf-8",
    )
    monitor_ids = ["monitor-m0", "monitor-m1", "monitor-m2", "monitor-m3"]
    monitor_path = _write_monitor_corpus(tmp_path, monitor_ids)

    payload = reserved_ids_payload(
        {
            "heal": heal,
            "fit": fit,
            "evaluation": evaluation,
            "monitor": monitor_path,
        }
    )
    assert payload["monitor"] == monitor_ids
    assert payload["heal_train"] == ["train-r0"]
    assert payload["pilot_fit"] == ["fit-r0"]
    assert payload["pilot_eval"] == ["eval-r0"]
    assert payload["content_hashes"]["monitor"]
    assert all(value not in payload["reserved_ids"] for value in monitor_ids)
    assert payload["reserved_ids"] and "train-r0" not in payload["reserved_ids"]

    reserved_path = tmp_path / "out" / "reserved_ids.json"
    reserved_path.parent.mkdir(parents=True)
    reserved_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GEMMALOOP_HEAL_MONITOR_RESERVED", str(reserved_path))

    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5, 6],
                    "labels": [-100, -100, 5, 6],
                    "attention_mask": [1, 1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    trainer = _trainer(
        tmp_path / "registered-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "registered-run",
        seed=0,
        smoke=False,
        prereg=str(tmp_path / "PREREGISTRATION.md"),
        pilot_arm=None,
        data=data,
        monitor_rows=monitor_path,
        vram_cap=0.0,
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=getattr(trainer, "data_collator", None),
        rows=[
            {"problem_id": "train-r0", "messages": []},
            {"problem_id": "train-r1", "messages": []},
        ],
        max_length=32,
        enabled=True,
        config=MonitorConfig(
            monitor_rows=2,
            probe_interval_tokens=10**12,
            generation_prompts=0,
            generation_ns=(),
            anchor_ns=(2,),
            nll_ns=(1, 2),
            nll_temperatures=(1.0,),
        ),
    )
    assert callback is not None
    split = json.loads(
        (tmp_path / "registered-run" / "heal_monitor_split.json").read_text(
            encoding="utf-8"
        )
    )
    assert split["registered"] is True
    assert split["monitor_problem_ids"] == sorted(monitor_ids)
    assert split["reserved_file"] == str(reserved_path)


def test_monitor_sha_uses_the_selected_monitor_file_and_its_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from datasets import Dataset

    import gemmaloop.train.heal_monitor as monitor_module

    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    monitor_file = monitor_dir / "monitor.jsonl"
    monitor_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "problem_id": f"m{index}",
                    "messages": [{"role": "user", "content": str(index)}],
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    import hashlib

    digest = hashlib.sha256(monitor_file.read_bytes()).hexdigest()
    (monitor_dir / "manifest.json").write_text(
        json.dumps({"sha256": digest}), encoding="utf-8"
    )

    def fake_tokenize(tokenizer, rows, raw_indices, *, max_length, include_teacher=True):
        return Dataset.from_list(
            [
                {
                    "input_ids": [3, 4, 5],
                    "labels": [-100, 4, 5],
                    "attention_mask": [1, 1, 1],
                    "row_id": int(index),
                }
                for index in raw_indices
            ]
        )

    class EmbeddingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(8, 2)

        def get_input_embeddings(self):
            return self.embeddings

    monkeypatch.setattr(monitor_module, "tokenize_monitor_rows", fake_tokenize)
    trainer = SimpleNamespace(
        model=EmbeddingModel(),
        dial_enabled=True,
        compute_loss=lambda *args, **kwargs: None,
        add_callback=lambda callback: None,
        args=SimpleNamespace(output_dir=str(tmp_path / "sha-run")),
    )
    args = SimpleNamespace(
        out=tmp_path / "sha-run",
        seed=0,
        smoke=True,
        prereg=None,
        pilot_arm=None,
        monitor_rows=monitor_dir,
        vram_cap=0.0,
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=SimpleNamespace(column_names=[], __len__=lambda self: 0),
        tokenizer=object(),
        max_length=32,
        enabled=True,
        config=MonitorConfig(
            monitor_rows=2,
            probe_interval_tokens=10**12,
            generation_prompts=0,
            generation_ns=(),
            anchor_ns=(2,),
            nll_ns=(1,),
            nll_temperatures=(1.0,),
        ),
    )
    assert callback is not None
    split = json.loads(
        (tmp_path / "sha-run" / "heal_monitor_split.json").read_text(
            encoding="utf-8"
        )
    )
    assert split["monitor_rows_sha256"] == digest
    assert split["monitor_manifest_rows_sha256"] == digest
    assert split["monitor_manifest_sha256_matches"] is True


def test_teacher_integrity_skips_a_fully_masked_batch():
    teacher = TeacherTopK(
        probs=torch.full((1, 2, 2), 0.4, dtype=torch.float16),
        indices=torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32),
        tail_prob=torch.full((1, 2), 0.2, dtype=torch.float32),
    )
    report = teacher_integrity(
        teacher, 8, labels=torch.full((1, 2), -100, dtype=torch.long)
    )
    assert report["ok"] is None
    assert "supervised" in report["skipped"].lower()


def test_fresh_archive_moves_monitor_state_and_starts_a_new_probe_clock(tmp_path):
    from gemmaloop.train.checkpoint import archive_run_state

    run = tmp_path / "run"
    config = MonitorConfig(probe_interval_tokens=500_000)
    monitor = HealMonitor(run, config=config)
    monitor._last_probe_tokens = 3_000_000
    monitor.alerts.append(
        MonitorAlert(severity="warn", rule="old", message="old run")
    )
    monitor.save_state()
    monitor.write_alerts()
    (run / "heal_metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (run / "heal_probes.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    texts = run / "heal_probe_texts"
    texts.mkdir()
    (texts / "sample.txt").write_text("old probe", encoding="utf-8")

    destination = archive_run_state(run, reason="fresh")
    assert destination.is_dir()
    for name in (
        "heal_monitor_state.json",
        "alerts.json",
        "heal_metrics.jsonl",
        "heal_probes.jsonl",
        "heal_probe_texts",
    ):
        assert (destination / name).exists(), name
        assert not (run / name).exists(), name

    fresh = HealMonitor(run, config=config)
    assert fresh._last_probe_tokens == 0
    assert fresh.alerts == []
    assert fresh.should_probe(499_999) is False
    assert fresh.should_probe(500_000) is True
    assert fresh.due_at_end(3_000_000) is True


def test_probe_exception_writes_a_row_warns_and_aborts_after_two(tmp_path):
    from types import SimpleNamespace

    config = MonitorConfig(probe_failure_abort_consecutive=2)
    monitor = HealMonitor(tmp_path / "run", config=config)
    trainer = SimpleNamespace(
        tokens_seen=500_000,
        state=SimpleNamespace(global_step=1),
        exit_code=None,
        exit_reason=None,
    )

    def factory():
        raise torch.OutOfMemoryError("probe OOM")

    callback = HealMonitorCallback(monitor, trainer, probe_factory=factory)
    control = SimpleNamespace(should_save=True, should_training_stop=False)

    callback._run_probe(control)
    assert monitor._last_probe_tokens == 500_000
    rows = [
        json.loads(line)
        for line in monitor.probes_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows and rows[-1]["probe_failed"] is True
    assert "OutOfMemoryError" in rows[-1]["error"]
    alerts = json.loads(monitor.alerts_path.read_text(encoding="utf-8"))["alerts"]
    assert any(
        alert["rule"] == "probe_failure" and alert["severity"] == "warn"
        for alert in alerts
    )

    trainer.tokens_seen = 1_000_000
    trainer.state.global_step = 2
    with pytest.raises(SystemExit) as excinfo:
        callback._run_probe(control)
    assert excinfo.value.code == 76
    alerts = json.loads(monitor.alerts_path.read_text(encoding="utf-8"))["alerts"]
    assert any(
        alert["rule"] == "probe_failure" and alert["severity"] == "abort"
        for alert in alerts
    )


def test_monitor_identity_uses_every_id_field_the_producer_emits(tmp_path):
    from gemmaloop.pilot.splits import reserved_ids_payload
    from gemmaloop.train.heal_monitor import load_registered_splits

    monitor = tmp_path / "monitor_rows.jsonl"
    monitor.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "cluster_id": "c1",
                "messages": [{"role": "user", "content": "q"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = reserved_ids_payload({"monitor": monitor, "fit": monitor})
    assert payload["monitor"] == ["c1", "p1"]
    reserved_file = tmp_path / "reserved_ids.json"
    reserved_file.write_text(json.dumps(payload), encoding="utf-8")
    args = _registered_args(tmp_path)
    _reserved, _training, manifest_monitor, _sources = load_registered_splits(
        args, env_paths=reserved_file
    )
    assert manifest_monitor == {"c1", "p1"}


def test_teacher_cache_complete_requires_the_listed_monitor_shards(tmp_path):
    from gemmaloop.pilot.harness import teacher_cache_complete

    cache = tmp_path / "monitor-teacher"
    cache.mkdir()
    (cache / "shard-00000.pt").write_bytes(b"shard")
    (cache / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "shards": [{"file": "shard-00000.pt"}, {"file": "shard-00001.pt"}],
            }
        ),
        encoding="utf-8",
    )
    assert teacher_cache_complete(cache) is False
    (cache / "shard-00001.pt").write_bytes(b"shard")
    assert teacher_cache_complete(cache) is True


def test_failing_monitor_teacher_collator_warns_and_marks_fit_unavailable(
    tmp_path, caplog
):
    """N5: the teacher-collator fallback is never silent."""

    import logging
    from types import SimpleNamespace

    from gemmaloop.train.heal_monitor import (
        _mark_teacher_fit_unavailable,
        build_probe_batches,
    )

    class Dataset:
        column_names = ["input_ids", "labels", "attention_mask", "row_id"]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "input_ids": [3, 4, 5],
                "labels": [-100, 4, 5],
                "attention_mask": [1, 1, 1],
                "row_id": int(index),
            }

    class EmbeddingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(8, 2)

        def get_input_embeddings(self):
            return self.embeddings

    def failing_collator(rows):
        raise RuntimeError("teacher row alignment broke")

    errors = []
    with caplog.at_level(logging.WARNING):
        batches = build_probe_batches(
            Dataset(),
            [0, 1],
            model=EmbeddingModel(),
            tokenizer=SimpleNamespace(pad_token_id=0),
            data_collator=failing_collator,
            batch_size=2,
            on_collator_error=errors.append,
        )
    assert len(errors) == 1
    assert "teacher row alignment broke" in str(errors[0])
    assert any(
        "MONITOR teacher collator failed" in record.getMessage()
        for record in caplog.records
    )
    assert batches and "teacher_topk_probs" not in batches[0]

    split = tmp_path / "heal_monitor_split.json"
    split.write_text(json.dumps({"teacher_fit": "available"}), encoding="utf-8")
    _mark_teacher_fit_unavailable(tmp_path, errors[0])
    payload = json.loads(split.read_text(encoding="utf-8"))
    assert payload["teacher_fit"] == "unavailable"
    assert "teacher row alignment broke" in payload["teacher_fit_error"]


def test_generation_sanity_ignores_length_truncated_completions():
    """M1: the 512-token cap alone must not warn on every probe."""

    probe = _healthy_probe(tokens=500_000)
    row = probe["generation"][0]
    row["finish_reason"] = "length"
    row["answer_present"] = False
    row["thinking_closed"] = False
    row["repetition_loop"] = False
    alerts = MonitorRules(MonitorConfig(warn_repeat_tokens=0)).observe_probe(probe)
    assert not any(alert.rule == "generation_sanity" for alert in alerts)

    stopped = _healthy_probe(tokens=1_000_000)
    row = stopped["generation"][0]
    row["finish_reason"] = "eos"
    row["answer_present"] = False
    row["thinking_closed"] = False
    row["repetition_loop"] = False
    alerts = MonitorRules(MonitorConfig(warn_repeat_tokens=0)).observe_probe(stopped)
    assert any(alert.rule == "generation_sanity" for alert in alerts)


# ---------------------------------------------------------------------------
# Probe forward micro-batching: metrics are identical to one batch of four
# ---------------------------------------------------------------------------


class _ProbeRowsDataset:
    """Ragged monitor rows (different lengths, masks and teacher fields)."""

    column_names = [
        "input_ids",
        "labels",
        "attention_mask",
        "teacher_topk_probs",
        "teacher_topk_indices",
        "teacher_tail_prob",
    ]

    def __init__(self) -> None:
        specs = [
            (
                [3, 4, 5, 6, 7, 8, 9],
                [1, 1, 1, 1, 1, 1, 1],
                [-100, -100, 5, 6, 7, 8, 9],
            ),
            (
                [3, 4, 5, 6, 7, 8, 9, 10, 11],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
                [-100, -100, 5, 6, 7, 8, 9, -100, -100],
            ),
            (
                [3, 4, 5, 6, 7],
                [1, 1, 1, 1, 1],
                [-100, -100, 5, 6, 7],
            ),
            (
                [3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [-100, -100, 5, -100, 7, 8, 9, -100],
            ),
        ]
        k = 4
        self.rows = []
        for input_ids, attention, labels in specs:
            length = len(input_ids)
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention,
                    "labels": labels,
                    "teacher_topk_probs": [[0.2] * k for _ in range(length)],
                    "teacher_topk_indices": [[0, 1, 2, 3] for _ in range(length)],
                    "teacher_tail_prob": [0.2] * length,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _microbatch_probe_config():
    from gemmaloop.train.heal_probes import ProbeConfig

    return ProbeConfig(
        anchor_ns=(2,),
        eval_ns=(1, 2),
        nll_ns=(1, 2),
        generation_ns=(),
        nll_temperatures=(1.0, 2.0),
    )


def _probe_runner(model, batches, config):
    from gemmaloop.train.heal_probes import ProbeRunner

    return ProbeRunner(
        model=model,
        text_model=model,
        weight=model.embed_tokens.weight.detach().float().clone(),
        softcap=None,
        batches=batches,
        config=config,
    )


def _assert_probe_metrics_equal(single, micro):
    for section in (
        "anchor_kl",
        "teacher_kl",
        "teacher_tokens",
        "calibrated_nll",
        "calibrated_nll_fit",
        "calibrated_nll_fit_tokens",
        "calibrated_nll_tokens",
        "calibrated_nll_temperature",
    ):
        assert set(single[section]) == set(micro[section]), section
        for key in single[section]:
            assert micro[section][key] == pytest.approx(
                single[section][key], rel=1e-4, abs=1e-6
            ), f"{section}[{key}]"
    assert micro["invariant"]["applicable"] is True
    assert single["invariant"]["applicable"] is True
    assert micro["invariant"]["ok"] is True
    assert single["invariant"]["ok"] is True
    assert micro["invariant"]["max_abs_diff"] == pytest.approx(
        single["invariant"]["max_abs_diff"], abs=1e-6
    )


def test_build_probe_batches_defaults_to_one_row_per_forward():
    from types import SimpleNamespace

    from gemmaloop.train.heal_monitor import build_probe_batches

    model = _tiny_looped_text_model()
    dataset = _ProbeRowsDataset()
    batches = build_probe_batches(
        dataset,
        [0, 1, 2, 3],
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    assert len(batches) == 4
    assert [int(batch["input_ids"].shape[0]) for batch in batches] == [1, 1, 1, 1]
    assert [int(batch["input_ids"].shape[1]) for batch in batches] == [7, 9, 5, 8]
    assert MonitorConfig().probe_batch_size == 1
    assert MonitorConfig().to_dict()["probe_batch_size"] == 1


def test_probe_microbatch_one_matches_one_batch_of_four():
    """Token-weighted aggregation makes 1-row forwards equal to 4 rows.

    Rows have different lengths, attention/label masks and padding, so the
    single 4-row batch pads every row to the longest one while the 1-row
    micro-batches forward each row exactly.  Every reported statistic must
    still match to floating tolerance.
    """
    from types import SimpleNamespace

    from gemmaloop.train.heal_monitor import build_probe_batches

    model = _tiny_looped_text_model()
    dataset = _ProbeRowsDataset()
    config = _microbatch_probe_config()
    tokenizer = SimpleNamespace(pad_token_id=0)

    single_batches = build_probe_batches(
        dataset,
        [0, 1, 2, 3],
        model=model,
        tokenizer=tokenizer,
        batch_size=4,
    )
    micro_batches = build_probe_batches(
        dataset,
        [0, 1, 2, 3],
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
    )
    assert len(single_batches) == 1
    assert len(micro_batches) == 4
    # Fit/score halves exercised on both partitionings: rows 0-1 fit, 2-3 score.
    single_result = _probe_runner(model, single_batches, config).run(
        tokens_seen=500_000, step=1, train_teacher_kl={"2": 0.5}
    )
    micro_result = _probe_runner(model, micro_batches, config).run(
        tokens_seen=500_000, step=1, train_teacher_kl={"2": 0.5}
    )
    _assert_probe_metrics_equal(single_result, micro_result)

    # The NLL fit/score split is by rows, not by micro-batch index.
    micro_split = _probe_runner(model, micro_batches, config)
    micro_split.nll_fit_batches = [0, 1]
    micro_split.nll_score_batches = [2, 3]
    split_result = micro_split.run(tokens_seen=500_000, step=1)
    # Rows 0-1 carry 5 supervised tokens each (fit), rows 2-3 carry 3+4
    # (score): the split is by rows, not by micro-batch index, and the two
    # halves cover the same tokens as the unsplit run.
    assert split_result["calibrated_nll_fit_tokens"]["2"] == 10
    assert split_result["calibrated_nll_tokens"]["2"] == 7
    assert (
        split_result["calibrated_nll_fit_tokens"]["2"]
        + split_result["calibrated_nll_tokens"]["2"]
        == micro_result["calibrated_nll_tokens"]["2"]
    )


def test_merge_invariants_is_the_conjunction_and_max_over_rows():
    from gemmaloop.train.heal_probes import merge_invariants

    merged = merge_invariants(
        [
            {"applicable": True, "ok": True, "max_abs_diff": 0.0},
            {"applicable": True, "ok": False, "max_abs_diff": 3e-3},
        ]
    )
    assert merged == {
        "applicable": True,
        "ok": False,
        "max_abs_diff": pytest.approx(3e-3),
    }
    assert merge_invariants([]) == {"applicable": False, "ok": None}
    assert merge_invariants([{"applicable": False, "ok": None, "max_abs_diff": None}]) == {
        "applicable": False,
        "ok": None,
        "max_abs_diff": None,
    }


def test_training_probe_wiring_uses_one_row_microbatches(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import gemmaloop.train.heal_monitor as monitor_module
    from tests.train.test_dial_trainer import _trainer

    captured: list[int] = []
    original = monitor_module.build_probe_batches

    def spy(*args, **kwargs):
        captured.append(int(kwargs.get("batch_size")))
        return original(*args, **kwargs)

    monkeypatch.setattr(monitor_module, "build_probe_batches", spy)

    trainer = _trainer(
        tmp_path / "microbatch-run", dial_enabled=True, teacher=True, max_steps=1
    )
    args = SimpleNamespace(
        out=tmp_path / "microbatch-run", seed=0, smoke=True, vram_cap=0.0
    )
    config = MonitorConfig(
        monitor_rows=4,
        probe_interval_tokens=10**12,
        generation_prompts=0,
        generation_ns=(),
        anchor_ns=(2,),
        nll_ns=(1,),
        nll_temperatures=(1.0,),
    )
    callback = monitor_module.install_heal_monitor(
        trainer,
        args,
        dataset=trainer.train_dataset,
        tokenizer=getattr(trainer, "processing_class", None),
        data_collator=getattr(trainer, "data_collator", None),
        enabled=True,
        config=config,
    )
    assert callback is not None
    trainer.train()
    assert captured
    assert set(captured) == {1}
    assert callback.monitor.probes_path.is_file()


# ---------------------------------------------------------------------------
# I1: the invariant runs on the fixed first four probe rows, and a raised
# check is a probe failure (WARN x2 -> ABORT), never an immediate ok=False.
# ---------------------------------------------------------------------------


def _invariant_batches(rows: int):
    return [
        {
            "input_ids": torch.tensor([[index, index + 1, index + 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        for index in range(1, rows + 1)
    ]


def test_invariant_covers_the_first_four_rows_across_micro_batches(monkeypatch):
    import gemmaloop.train.heal_probes as heal_probes

    text_model = _tiny_looped_text_model()
    calls = []

    def fake(model, text_model, batch, *, parent=None):
        calls.append(
            (
                batch["input_ids"].tolist(),
                None if parent is None else tuple(parent.shape),
            )
        )
        return {"applicable": True, "ok": True, "max_abs_diff": 0.0}

    monkeypatch.setattr(heal_probes, "exit1_bit_exact", fake)
    runner = heal_probes.ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=text_model.embed_tokens.weight.detach().float(),
        softcap=None,
        batches=_invariant_batches(6),
        config=heal_probes.ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(),
            generation_ns=(),
            invariant_rows=4,
        ),
    )
    result = runner.run(tokens_seen=0, step=1)
    assert result["invariant"]["ok"] is True
    assert [call[0][0][0] for call in calls] == [1, 2, 3, 4]
    # The adapter-off N=1 parent forward is reused, not recomputed.
    assert all(call[1] is not None for call in calls)


def test_invariant_slices_the_fixed_rows_out_of_a_larger_micro_batch(monkeypatch):
    import gemmaloop.train.heal_probes as heal_probes

    text_model = _tiny_looped_text_model()
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
            "attention_mask": torch.ones(3, 3, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6], [5, 6, 7], [6, 7, 8]]),
            "attention_mask": torch.ones(3, 3, dtype=torch.long),
        },
    ]
    calls = []

    def fake(model, text_model, batch, *, parent=None):
        calls.append(
            (
                batch["input_ids"].tolist(),
                None if parent is None else tuple(parent.shape),
            )
        )
        return {"applicable": True, "ok": True, "max_abs_diff": 0.0}

    monkeypatch.setattr(heal_probes, "exit1_bit_exact", fake)
    runner = heal_probes.ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=text_model.embed_tokens.weight.detach().float(),
        softcap=None,
        batches=batches,
        config=heal_probes.ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(),
            generation_ns=(),
            invariant_rows=4,
        ),
    )
    runner.run(tokens_seen=0, step=1)
    assert len(calls) == 2
    assert calls[0][0] == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert calls[1][0] == [[4, 5, 6]]
    assert calls[1][1][0] == 1


@pytest.mark.parametrize("batch_size", [3, 5, 7])
def test_invariant_does_not_false_abort_for_other_micro_batch_sizes(
    monkeypatch, batch_size
):
    """Rows sliced out of a larger forward must be compared on their own batch.

    ``_forward_hidden`` is stubbed to encode the batch size in the value, which
    is the CPU stand-in for the bf16 GEMM kernel selection that makes a sliced
    parent differ bitwise from the same row forwarded alone.  The old slicing
    parent would compare e.g. 3.0 (from a 3-row forward) with 1.0 (exit 1 on the
    1-row slice) and false-abort.
    """

    import gemmaloop.train.heal_probes as heal_probes

    text_model = _tiny_looped_text_model()
    rows = 8
    batches = []
    for start in range(0, rows, batch_size):
        ids = [
            [index, index + 1, index + 2]
            for index in range(start + 1, min(start + batch_size, rows) + 1)
        ]
        batches.append(
            {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(ids), 3, dtype=torch.long),
            }
        )

    def size_dependent_forward(
        model, text_model, batch, n_loops, *, adapters_off=False, exits=None
    ):
        batch_rows = int(batch["input_ids"].shape[0])
        return torch.full((batch_rows, 2), float(batch_rows))

    monkeypatch.setattr(heal_probes, "_forward_hidden", size_dependent_forward)
    runner = heal_probes.ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=text_model.embed_tokens.weight.detach().float(),
        softcap=None,
        batches=batches,
        config=heal_probes.ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(),
            generation_ns=(),
            invariant_rows=4,
        ),
    )
    result = runner.run(tokens_seen=0, step=1)
    assert result["invariant"]["ok"] is True
    assert result["invariant"]["max_abs_diff"] == 0.0


def test_invariant_exception_is_a_probe_failure_not_an_immediate_abort(monkeypatch):
    import gemmaloop.train.heal_probes as heal_probes

    text_model = _tiny_looped_text_model()

    def boom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated invariant OOM")

    monkeypatch.setattr(heal_probes, "exit1_bit_exact", boom)
    runner = heal_probes.ProbeRunner(
        model=text_model,
        text_model=text_model,
        weight=text_model.embed_tokens.weight.detach().float(),
        softcap=None,
        batches=_invariant_batches(1),
        config=heal_probes.ProbeConfig(
            anchor_ns=(),
            eval_ns=(1,),
            nll_ns=(),
            generation_ns=(),
            invariant_rows=4,
        ),
    )
    result = runner.run(tokens_seen=0, step=1)
    assert result["invariant"]["ok"] is not False
    assert result["probe_failed"] is True
    assert any(error.startswith("invariant:") for error in result["probe_errors"])
    # The probe itself carries the message, so the registered WARN/ABORT does
    # not say "unknown probe error".
    assert "simulated invariant OOM" in result["error"]

    rules = MonitorRules(MonitorConfig(probe_failure_abort_consecutive=2))
    first = rules.observe_probe(result)
    assert any(
        alert.rule == "probe_failure" and alert.severity == "warn"
        for alert in first
    )
    assert any(
        "simulated invariant OOM" in alert.message for alert in first
    )
    second = rules.observe_probe(result)
    assert any(
        alert.rule == "probe_failure" and alert.severity == "abort"
        for alert in second
    )


# ---------------------------------------------------------------------------
# I2: every registered NLL temperature shares one logits pass and is
# bit-identical to the per-temperature implementation.
# ---------------------------------------------------------------------------


def test_multi_temperature_nll_is_bit_identical_to_per_temperature():
    from gemmaloop.train.heal_probes import (
        chunked_nll_temperatures,
        nll_from_hidden_temperatures,
    )

    torch.manual_seed(0)
    hidden = torch.randn(7, 5)
    weight = torch.randn(11, 5)
    targets = torch.tensor([0, 4, 6, 1, 10, 3, 8])
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    for token_chunk, vocab_chunk in ((1, 2), (3, 4), (1024, 16384)):
        shared = chunked_nll_temperatures(
            hidden,
            weight,
            targets,
            temperatures,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        for temperature in temperatures:
            expected = chunked_nll_hidden(
                hidden,
                weight,
                targets,
                temperature=temperature,
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
            )
            assert torch.equal(shared[temperature], expected)

    labels = torch.tensor([[-100, 0, 4, 6, 1, 10, 3, 8]])
    hidden = torch.randn(1, 8, 5)
    shared_means = nll_from_hidden_temperatures(
        hidden, labels, weight, temperatures
    )
    from gemmaloop.train.heal_probes import nll_from_hidden

    for temperature in temperatures:
        expected = nll_from_hidden(
            hidden, labels, weight, temperature=temperature
        )
        assert expected is not None
        assert shared_means[temperature][1] == expected[1]
        assert shared_means[temperature][0] == pytest.approx(
            expected[0], rel=0, abs=1e-12
        )

def test_duplicate_temperatures_are_explicitly_deduplicated():
    from gemmaloop.train.heal_probes import (
        chunked_nll_hidden,
        chunked_nll_temperatures,
    )

    torch.manual_seed(1)
    hidden = torch.randn(5, 4)
    weight = torch.randn(9, 4)
    targets = torch.tensor([0, 3, 5, 8, 1])
    single = chunked_nll_hidden(
        hidden, weight, targets, temperature=1.0, token_chunk=2, vocab_chunk=3
    )
    duplicated = chunked_nll_temperatures(
        hidden, weight, targets, (1.0, 1.0), token_chunk=2, vocab_chunk=3
    )
    assert list(duplicated) == [1.0]
    assert torch.equal(duplicated[1.0], single)


def test_probe_public_api_includes_multi_temperature_nll():
    import gemmaloop.train.heal_probes as heal_probes

    assert "nll_from_hidden_temperatures" in heal_probes.__all__
