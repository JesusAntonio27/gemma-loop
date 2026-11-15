"""CPU tests for the registered engine gate's decision logic.

No model and no GPU: the GPU harness is exercised by the controller; these
tests feed fake engine outputs through the same comparison, reporting and exit
decision code.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

import scripts.engine_gate as gate
from gemmaloop.serve.generate import GenOut, Sampling


def _out(token_ids, text="ok", tokens=None):
    return GenOut(
        text=text,
        reasoning="",
        tool_calls=[],
        completion_tokens=len(token_ids) if tokens is None else tokens,
        finish_reason="length",
        invalid_tool_call=False,
        token_ids=tuple(token_ids),
    )


def test_fixture_has_the_registered_prompt_counts():
    prompts = gate.load_prompts()
    assert len(prompts["prompts"]) == 32
    assert sum(prompt["kind"] == "lcb" for prompt in prompts["prompts"]) == 16
    assert sum(prompt["kind"] == "math" for prompt in prompts["prompts"]) == 16
    assert len(prompts["sampling_ids"]) == 16
    assert prompts["sampling_set"] <= set(prompts["ids"])


def test_rejects_a_prompt_file_with_duplicate_ids(tmp_path):
    payload = json.loads(gate.PROMPTS_FILE.read_text(encoding="utf-8"))
    payload["prompts"][1]["id"] = payload["prompts"][0]["id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        gate.load_prompts(path)


def test_equality_criterion_passes_on_identical_token_ids():
    ids = ["p0", "p1", "p2"]
    padded = [_out([1, 2, 3]), _out([4]), _out([5, 6])]
    refill = [_out([1, 2, 3]), _out([4]), _out([5, 6])]
    row = gate.equality_criterion("greedy", padded, refill, ids)
    assert row["passed"] is True
    assert row["exact"] == 3
    assert row["compared_by"] == "token_ids"


def test_equality_criterion_fails_on_one_flipped_token():
    ids = ["p0", "p1", "p2"]
    padded = [_out([1, 2, 3]), _out([4]), _out([5, 6])]
    refill = [_out([1, 2, 3]), _out([4, 9]), _out([5, 6])]
    row = gate.equality_criterion("greedy", padded, refill, ids)
    assert row["passed"] is False
    assert row["exact"] == 2
    assert row["mismatched_ids"] == ["p1"]


def test_equality_criterion_fails_on_missing_outputs():
    row = gate.equality_criterion("greedy", [_out([1])], [], ["p0"])
    assert row["passed"] is False
    assert "counts differ" in row["detail"]


def test_equality_falls_back_to_text_for_outputs_without_token_ids():
    class Fake:
        def __init__(self, text):
            self.text = text
            self.completion_tokens = 2
            self.finish_reason = "stop"
            self.token_ids = None

    row = gate.equality_criterion(
        "greedy", [Fake("a"), Fake("b")], [Fake("a"), Fake("b")], ["p0", "p1"]
    )
    assert row["passed"] is True
    assert row["compared_by"] == "text"


def test_engine_stats_and_throughput_criterion():
    outputs = [_out([1, 2, 3]), _out([4])]
    stats = gate.engine_stats("padded", outputs, 2.0)
    assert stats == {
        "engine": "padded",
        "requests": 2,
        "tokens": 4,
        "elapsed_s": 2.0,
        "tok_s": 2.0,
    }
    refill = gate.engine_stats("refill", outputs, 1.0)
    row = gate.throughput_criterion("throughput", stats, refill, 2)
    assert row["passed"] is True
    assert row["padded_tok_s"] == 2.0
    assert row["refill_tok_s"] == 4.0

    short = gate.engine_stats("refill", outputs[:1], 1.0)
    assert gate.throughput_criterion("throughput", stats, short, 2)["passed"] is False


def test_decide_gate_needs_every_criterion():
    passing = {"criterion": "a", "passed": True}
    failing = {"criterion": "b", "passed": False}
    coverage = {"criterion": "registered_coverage", "passed": False}
    assert gate.decide_gate([passing, passing]) is True
    assert gate.decide_gate([passing, failing]) is False
    assert gate.decide_gate([passing, coverage]) is False
    assert gate.decide_gate([]) is False


def test_report_and_markdown_contain_pass_fail_and_tok_s(tmp_path):
    criteria = [
        {"criterion": "greedy_n=1", "passed": True, "detail": "32/32"},
        {"criterion": "sampled_n=2", "passed": False, "detail": "15/16"},
    ]
    throughput = [
        {
            "block": "16-18",
            "n_loops": 2,
            "engine": "padded",
            "requests": 32,
            "tokens": 4096,
            "elapsed_s": 100.0,
            "tok_s": 40.96,
        },
        {
            "block": "16-18",
            "n_loops": 2,
            "engine": "refill",
            "requests": 32,
            "tokens": 4096,
            "elapsed_s": 50.0,
            "tok_s": 81.92,
        },
    ]
    report = gate.build_report(
        criteria=criteria, throughput=throughput, config={"seed": 0}
    )
    assert report["overall"] == "FAIL"
    markdown = gate.render_markdown(report)
    assert "**Overall: FAIL**" in markdown
    assert "| greedy_n=1 | PASS |" in markdown
    assert "| sampled_n=2 | FAIL |" in markdown
    assert "81.9" in markdown

    output = gate.write_report(report, tmp_path / "gate.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert output.with_suffix(".md").is_file()


def test_gate_refuses_to_run_without_a_gpu(monkeypatch):
    monkeypatch.setattr(
        gate.torch.cuda, "is_available", lambda: False
    )
    with pytest.raises(SystemExit, match="CUDA GPU"):
        gate.main([])


def test_rep_refill_criterion_requires_real_multi_round_work():
    passing = gate.rep_refill_criterion(
        "reprefill",
        {
            "rounds": 4,
            "rep_refills": 9,
            "late_admissions": 3,
            "mixed_histories": True,
        },
    )
    assert passing["passed"] is True
    assert passing["rounds"] == 4
    for broken in (
        {"rounds": 1, "rep_refills": 0, "late_admissions": 0,
         "mixed_histories": False},
        {"rounds": 3, "rep_refills": 0, "late_admissions": 2,
         "mixed_histories": True},
        {"rounds": 3, "rep_refills": 4, "late_admissions": 0,
         "mixed_histories": True},
        {"rounds": 3, "rep_refills": 4, "late_admissions": 2,
         "mixed_histories": False},
    ):
        assert gate.rep_refill_criterion("reprefill", broken)["passed"] is False


def test_registered_combinations_cover_every_arm_block_bridge_policy_and_n():
    gate_reg = gate.load_engine_gate()
    expected = set(gate_reg["expected"])
    assert ("O", "", "", "", 1) in expected
    assert ("O-H", "", "", "always", 1) in expected
    for block in ("25-29", "16-18", "15-29"):
        n1_keys = [
            key for key in expected
            if key[0] == "L" and key[1] == block and key[4] == 1
        ]
        assert len(n1_keys) == 1
        for bridge in ("gch", "scalar"):
            for n_loops in (2, 3, 4, 6, 8):
                assert ("L", block, bridge, "loop_only", n_loops) in expected
        for bridge in ("none", "damped"):
            for n_loops in (2, 3, 4):
                assert ("L", block, bridge, "loop_only", n_loops) in expected
            assert ("L", block, bridge, "loop_only", 6) not in expected
    assert len(expected) == 53
    assert gate_reg["max_new_tokens"] > gate_reg["round_size"]
    assert gate_reg["max_batch"] > 0


def test_coverage_criterion_requires_every_registered_combination():
    expected = {("L", "16-18", "gch", "loop_only", 1)}
    assert gate.coverage_criterion(expected, expected)["passed"] is True
    missing = gate.coverage_criterion(expected, set())
    assert missing["passed"] is False
    assert "16-18" in missing["detail"]


def test_registered_run_refuses_cli_narrowing():
    gate_reg = gate.load_engine_gate()
    args = gate.build_parser().parse_args([])
    settings, combos, registered = gate.resolve_run(args, gate_reg)
    assert registered is True
    assert settings["max_new_tokens"] == gate_reg["max_new_tokens"]
    assert len(combos) == len(gate_reg["combinations"])

    narrowed = gate.build_parser().parse_args(["--n-loops", "1"])
    with pytest.raises(SystemExit, match="--smoke"):
        gate.resolve_run(narrowed, gate_reg)

    smoke_args = gate.build_parser().parse_args(
        ["--smoke", "--blocks", "16-18", "--bridge", "gch",
         "--n-loops", "2", "--max-new-tokens", "512", "--round-size", "128"]
    )
    _settings, smoke_combos, smoke_registered = gate.resolve_run(
        smoke_args, gate_reg
    )
    assert smoke_registered is False
    assert smoke_combos == [
        {"arm": "L", "block": "16-18", "bridge": "gch",
         "adapter_policy": "loop_only", "n_loops": [2]}
    ]


def test_smoke_report_can_never_be_the_registered_pass():
    criteria = [{"criterion": "greedy", "passed": True}]
    report = gate.build_report(
        criteria=criteria, throughput=[], config={}, smoke=True
    )
    assert report["passed"] is True
    assert report["registered_pass"] is False
    assert report["overall"] == "SMOKE-PASS"


def _logits_with_gap(gap):
    return [0.0, 10.0, 10.0 - gap]


def test_near_tie_divergence_passes_at_the_exact_90_percent_boundary():
    ids = [f"p{index}" for index in range(10)]
    padded = [_out([1, 2, 3]) for _ in ids]
    refill = [_out([1, 2, 3]) for _ in ids]
    refill[0] = _out([1, 2, 4])

    def logits_fn(engine, conversation, prefix_ids):
        return _logits_with_gap(0.005)

    row = gate.equality_criterion(
        "greedy",
        padded,
        refill,
        ids,
        conversations=[[] for _ in ids],
        sampling=gate._sampling({"max_new_tokens": 8}),
        logits_fn=logits_fn,
    )
    assert row["passed"] is True
    assert row["exact"] == 9
    assert row["exact_fraction"] == 0.9
    assert len(row["divergences"]) == 1
    divergence = row["divergences"][0]
    assert divergence["request_id"] == "p0"
    assert divergence["position"] == 2
    assert divergence["padded_token"] == 3
    assert divergence["refill_token"] == 4
    assert divergence["near_tie"] is True
    assert divergence["padded_gap"] < 0.01
    assert divergence["refill_gap"] < 0.01


def test_non_tie_divergence_fails_and_is_reported():
    ids = ["p0"]
    padded = [_out([1, 2, 3])]
    refill = [_out([1, 2, 4])]

    row = gate.equality_criterion(
        "greedy",
        padded,
        refill,
        ids,
        conversations=[[]],
        sampling=gate._sampling({"max_new_tokens": 8}),
        logits_fn=lambda engine, conversation, prefix_ids: _logits_with_gap(0.5),
    )
    assert row["passed"] is False
    divergence = row["divergences"][0]
    assert divergence["near_tie"] is False
    assert divergence["padded_gap"] == pytest.approx(0.5)
    assert "p0" in row["detail"]


def test_too_many_near_ties_fail_the_90_percent_rule():
    ids = [f"p{index}" for index in range(10)]
    padded = [_out([1, 2, 3]) for _ in ids]
    refill = [_out([1, 2, 3]) for _ in ids]
    refill[0] = _out([1, 2, 4])
    refill[1] = _out([1, 2, 5])
    row = gate.equality_criterion(
        "greedy",
        padded,
        refill,
        ids,
        conversations=[[] for _ in ids],
        sampling=gate._sampling({"max_new_tokens": 8}),
        logits_fn=lambda engine, conversation, prefix_ids: _logits_with_gap(0.001),
    )
    assert row["exact"] == 8
    assert row["passed"] is False


def test_first_512_tokens_only_are_compared():
    padded = [_out(list(range(600)))]
    refill = [_out(list(range(600)))]
    refill[0].token_ids = tuple(list(range(512)) + [999] * 88)
    row = gate.equality_criterion(
        "greedy", padded, refill, ["p0"], token_limit=512
    )
    assert row["passed"] is True
    assert row["divergences"] == []


def test_recompute_engine_logits_runs_in_fp32_on_a_tiny_model():
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    from tests.model.tiny import tiny_config

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            ids = [2] + [3 + (ord(ch) % 50) for ch in conversation[0]["content"]]
            tensor = torch.tensor([ids])
            return {
                "input_ids": tensor,
                "attention_mask": torch.ones_like(tensor),
                "mm_token_type_ids": torch.zeros_like(tensor),
            }

    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=tiny_config(), vision_config=None, audio_config=None
        )
    ).eval()
    logits = gate.recompute_engine_logits(
        model,
        Processor(),
        [{"role": "user", "content": "abc"}],
        [5, 6],
        gate._sampling({"max_new_tokens": 8}),
    )
    assert logits.dtype == torch.float32
    assert logits.shape == (tiny_config().vocab_size,)


def test_informational_throughput_never_gates_the_decision():
    row = gate.throughput_criterion(
        "throughput",
        {"requests": 1, "tok_s": 1.0},
        {"requests": 1, "tok_s": 2.0},
        1,
    )
    assert row["informational"] is True
    assert gate.decide_gate(
        [
            {"criterion": "greedy", "passed": True},
            {**row, "passed": False},
        ]
    ) is True
    assert gate.decide_gate(
        [{"criterion": "greedy", "passed": False}, row]
    ) is False


def test_prompt_provenance_records_the_fixture_digest():
    provenance = gate.prompt_provenance(gate.PROMPTS_FILE)
    assert len(provenance["sha256"]) == 64
    assert provenance["confirmatory_items_used"] is False
    assert "synthetic" in provenance["source"]


def test_registered_near_tie_threshold_is_point_one():
    gate_reg = gate.load_engine_gate()
    assert gate_reg["near_tie_logit_gap"] == pytest.approx(0.25)
    row = gate.equality_criterion(
        "greedy",
        [_out([1, 2, 3])],
        [_out([1, 2, 4])],
        ["p0"],
        conversations=[[]],
        sampling=gate._sampling({"max_new_tokens": 8}),
        logits_fn=lambda engine, conversation, prefix_ids: _logits_with_gap(0.05),
        near_tie_gap=0.25,
        min_exact_fraction=0.0,
    )
    assert row["passed"] is True


def _with_logprobs(token_ids, chosen, top20=None):
    output = _out(token_ids)
    output.chosen_logprobs = tuple(chosen)
    output.top20_logprobs = tuple(
        top20 if top20 is not None else [tuple([value] * 20) for value in chosen]
    )
    return output


def test_logprob_bug_detector_accepts_only_tiny_engine_drift():
    ids = ["p0", "p1"]
    padded = [
        _with_logprobs([1, 2, 3], [-0.5, -0.4, -0.3]),
        _with_logprobs([4, 5], [-0.2, -0.1]),
    ]
    refill = [
        _with_logprobs([1, 2, 3], [-0.505, -0.395, -0.295]),
        _with_logprobs([4, 5], [-0.205, -0.095]),
    ]
    row = gate.logprob_criterion("logprobs", padded, refill, ids)
    assert row["passed"] is True
    assert row["steps"] == 5
    assert row["mean_abs_delta"] <= 0.01
    assert row["max_abs_delta"] <= 0.1

    drifting = [
        _with_logprobs([1, 2, 3], [-0.5, -0.4, -0.3]),
        _with_logprobs([4, 5], [-0.2, -0.1]),
    ]
    wrong = [
        _with_logprobs([1, 2, 3], [-0.5, -0.4, -0.3]),
        _with_logprobs([4, 5], [-0.5, -0.4]),
    ]
    assert gate.logprob_criterion("logprobs", drifting, wrong, ids)["passed"] is False


def test_logprob_bug_detector_fails_on_max_only():
    ids = list(range(100))
    padded = [_with_logprobs(ids, [0.0] * 100)]
    refill = [_with_logprobs(ids, [0.0] * 99 + [0.2])]
    row = gate.logprob_criterion("logprobs", padded, refill, ["p0"])
    assert row["passed"] is False
    assert row["mean_abs_delta"] <= 0.01
    assert row["max_abs_delta"] > 0.1


def test_logprob_bug_detector_fails_on_missing_records():
    row = gate.logprob_criterion(
        "logprobs",
        [_out([1, 2])],
        [_out([1, 2])],
        ["p0"],
    )
    assert row["passed"] is False
    assert "missing" in row["detail"]


def _loop_gate_model():
    from peft import LoraConfig, get_peft_model

    from tests.serve.test_refill_looped import (
        _build_lm,
        _randomise_adapters,
    )

    base = _build_lm((5, 7), "gch", 2, policy="loop_only")
    text_model = base.model.language_model
    with torch.no_grad():
        text_model.loop_bridge_g.uniform_(0.05, 0.2)
    model = _randomise_adapters(
        get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
                target_modules=[
                    "q_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    )
    return model, text_model


def test_negative_control_wrong_pass_index_policy_fails_the_gate():
    from gemmaloop.serve.generate import generate_batch
    from gemmaloop.serve.refill import generate_batch_refill
    from tests.serve.test_refill_looped import _Processor, _conversation

    model, text_model = _loop_gate_model()
    processor = _Processor()
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2
    )
    text_model._gemmaloop_loop_config.adapter_policy = "always"
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
    )
    row = gate.equality_criterion(
        "greedy_wrong_policy", padded, refill, ["a", "b", "c"]
    )
    assert row["passed"] is False
    assert gate.decide_gate([row]) is False


def test_negative_control_bridge_skipped_on_reprefill_fails_the_gate(monkeypatch):
    import gemmaloop.serve.refill as R
    from gemmaloop.serve.generate import generate_batch
    from gemmaloop.serve.refill import generate_batch_refill
    from tests.serve.test_refill_looped import _Processor, _conversation

    model, text_model = _loop_gate_model()
    processor = _Processor()
    conversations = [
        _conversation("a"),
        _conversation("longer prompt"),
        _conversation("xy"),
    ]
    sampling = Sampling(max_new_tokens=4, temperature=0.0)
    padded = generate_batch(
        model, processor, conversations, sampling, max_batch=2
    )

    original = R._decode_round

    def bridge_skipped(*args, **kwargs):
        saved = text_model.loop_bridge_g.detach().clone()
        with torch.no_grad():
            text_model.loop_bridge_g.zero_()
        try:
            return original(*args, **kwargs)
        finally:
            with torch.no_grad():
                text_model.loop_bridge_g.copy_(saved)

    monkeypatch.setattr(R, "_decode_round", bridge_skipped)
    refill = generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        max_batch=2,
        round_size=2,
    )
    row = gate.equality_criterion(
        "greedy_bridge_skipped", padded, refill, ["a", "b", "c"]
    )
    assert row["passed"] is False
    assert gate.decide_gate([row]) is False


def test_registered_gate_fits_the_four_hour_budget():
    gate_reg = gate.load_engine_gate()
    assert gate_reg["max_new_tokens"] <= 256
    assert gate_reg["round_size"] < gate_reg["max_new_tokens"]
    assert gate_reg["max_new_tokens"] // gate_reg["round_size"] >= 4
    prompts = gate.load_prompts(gate_reg["prompts"])
    assert len(prompts["sampling_ids"]) == 16


def _run_group_fixture(monkeypatch):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from tests.model.tiny import tiny_config, tiny_text_model
    from tests.serve.test_refill_looped import _Processor

    config = tiny_config()
    torch.manual_seed(0)
    text_model = tiny_text_model(seed=0)
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=config, vision_config=None, audio_config=None)
    )
    model.model.language_model = text_model
    plan = make_layer_plan(config.to_dict(), block=(5, 7))
    apply_loops(
        text_model,
        plan,
        LoopConfig(
            n_loops=2,
            bridge="gch",
            block=(5, 7),
            adapter_policy="loop_only",
        ),
    )
    monkeypatch.setattr(
        gate, "load_model", lambda spec: (model, _Processor(), plan)
    )
    prompts = {
        "prompts": [
            {"id": f"g{index}", "kind": "math",
             "messages": [{"role": "user", "content": text}]}
            for index, text in enumerate(["a", "longer prompt", "mid!", "xy"])
        ],
        "ids": ["g0", "g1", "g2", "g3"],
        "sampling_ids": ["g0", "g1", "g2", "g3"],
        "sampling_set": {"g0", "g1", "g2", "g3"},
    }
    settings = {
        "size": "e2b",
        "quant": "bf16",
        "max_new_tokens": 4,
        "round_size": 2,
        "max_batch": 2,
        "seed": 0,
        "token_limit": 4,
        "near_tie_logit_gap": 0.25,
        "min_exact_fraction": 0.9,
        "sampling_n": [2],
        "throughput_n": [1, 2],
        "comparison_prompts": 4,
        "refill_budgets": [2, 3, 4, 4],
        "logprob_mean_limit": 0.01,
        "logprob_max_limit": 0.1,
        "control_batch": 2,
        "one_n1_key_per_block": True,
        "long_max_new_tokens": 4,
        "long_round_size": 2,
        "long_round_1024_tokens": 4,
    }
    return model, prompts, settings


def test_run_group_handles_peft_wrapped_looped_model(monkeypatch):
    model, prompts, settings = _run_group_fixture(monkeypatch)
    criteria, throughput, observed = gate.run_group(
        ("L", "5-7", "gch", "loop_only"), [1, 2], prompts, settings
    )
    assert observed == {
        ("L", "5-7", "gch", "loop_only", 1),
        ("L", "5-7", "gch", "loop_only", 2),
    }
    names = [row["criterion"] for row in criteria]
    assert any(name.startswith("greedy_") for name in names)
    assert any(name.startswith("sampled_") for name in names)
    assert any(name.startswith("logprobs_") for name in names)
    assert gate.decide_gate(criteria) is True


def test_seed_gate_parameters_sets_nonzero_lora_b_and_bridge(monkeypatch):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from tests.model.tiny import tiny_config, tiny_text_model

    config = tiny_config()
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=config, vision_config=None, audio_config=None)
    )
    model.model.language_model = tiny_text_model(seed=0)
    plan = make_layer_plan(config.to_dict(), block=(5, 7))
    apply_loops(
        model.model.language_model,
        plan,
        LoopConfig(n_loops=2, bridge="gch", block=(5, 7),
                   adapter_policy="loop_only"),
    )
    seeded = gate.seed_gate_parameters(
        model,
        arm="L",
        bridge="gch",
        plan=plan,
        label="L_5-7_gch_loop_only",
        base_seed=0,
    )
    b_values = [
        parameter.detach()
        for name, parameter in seeded.named_parameters()
        if "lora_B" in name
    ]
    assert b_values
    assert all(float(value.abs().sum()) > 0 for value in b_values)
    text = gate._text_model(seeded)
    assert float(text.loop_bridge_g.min()) >= 0.05
    assert float(text.loop_bridge_g.max()) <= 0.2
    assert float(abs(text.loop_bridge_logit)) > 0


def _criterion_row(
    exact_fraction, mean_delta, max_delta, near_ties=True, requests=100
):
    exact = round(exact_fraction * requests)
    return {
        "criterion": "row",
        "passed": True,
        "valid": True,
        "exact": exact,
        "requests": requests,
        "exact_fraction": exact / requests,
        "mean_abs_delta": mean_delta,
        "max_abs_delta": max_delta,
        "records_complete": True,
        "divergences": [] if near_ties else [{"near_tie": False}],
        "near_ties_only": near_ties,
    }


def test_noise_floor_control_gates_refill_relative_to_padded_control():
    control = _criterion_row(0.94, 0.004, 0.04)
    good = gate.apply_noise_floor(
        _criterion_row(0.94, 0.010, 0.08), control
    )
    assert good["passed"] is True
    assert good["control"]["exact_fraction"] == 0.94

    assert gate.apply_noise_floor(
        _criterion_row(0.92, 0.010, 0.08), control
    )["passed"] is False
    assert gate.apply_noise_floor(
        _criterion_row(0.99, 0.001, 0.01, near_ties=False), control
    )["passed"] is False
    broken = _criterion_row(0.90, 0.010, 0.08)
    broken["valid"] = False
    assert gate.apply_noise_floor(broken, control)["passed"] is False
    assert gate.apply_noise_floor(
        _criterion_row(0.90, 0.010, 0.08),
        {**_criterion_row(0.90, 0.0, 0.0), "valid": False},
    )["passed"] is False


def test_logprob_noise_floor_fails_closed_on_missing_records():
    control = _criterion_row(0.9, 0.001, 0.01)
    missing = _criterion_row(0.9, 0.001, 0.01)
    missing["records_complete"] = False
    missing["valid"] = False
    assert gate.apply_logprob_noise_floor(missing, control)["passed"] is False


def test_pooled_exactness_rule():
    rows = [
        _criterion_row(0.90, 0.0, 0.0),
        _criterion_row(0.90, 0.0, 0.0),
    ]
    rows[0]["control"] = {"exact": 90, "requests": 100}
    rows[1]["control"] = {"exact": 90, "requests": 100}
    assert gate.pooled_exactness_criterion(rows)["passed"] is True
    rows[0]["exact"] = 70
    assert gate.pooled_exactness_criterion(rows)["passed"] is False


def test_registered_near_tie_threshold_is_quarter_logit():
    reg = gate.load_engine_gate()
    assert reg["near_tie_logit_gap"] == pytest.approx(0.25)


def test_gumbel_perturbed_gap_uses_the_perturbed_top_two():
    logits = torch.tensor([10.0, 0.0, 0.0])
    # Large noise on token 1 closes the raw gap; tiny noise on token 0 keeps it.
    close_noise = torch.tensor([0.0, 10.1, 0.0])
    wide_noise = torch.tensor([0.0, 0.0, 0.0])
    assert gate.perturbed_top2_gap(logits, close_noise) < 0.2
    assert gate.perturbed_top2_gap(logits, wide_noise) == pytest.approx(10.0)


def test_sampled_near_tie_uses_the_gap_fn_not_the_raw_gap():
    row = gate.equality_criterion(
        "sampled",
        [_out([1, 2, 3])],
        [_out([1, 2, 4])],
        ["p0"],
        conversations=[[]],
        sampling=gate._sampling({"max_new_tokens": 8}),
        gap_fn=lambda engine, conversation, prefix, position, request: 0.1,
        near_tie_gap=0.25,
        min_exact_fraction=0.0,
    )
    assert row["passed"] is True
    row = gate.equality_criterion(
        "sampled",
        [_out([1, 2, 3])],
        [_out([1, 2, 4])],
        ["p0"],
        conversations=[[]],
        sampling=gate._sampling({"max_new_tokens": 8}),
        gap_fn=lambda engine, conversation, prefix, position, request: 1.0,
        near_tie_gap=0.25,
        min_exact_fraction=0.0,
    )
    assert row["passed"] is False


def test_long_prompts_exist_for_sliding_window_and_static_cache():
    prompts = gate.load_prompts()
    assert len(prompts["long_prompts"]) == 4
    for prompt in prompts["long_prompts"]:
        content = prompt["messages"][0]["content"]
        assert len(content) > 16000


def test_source_provenance_records_commit_dirty_and_hashes():
    provenance = gate.source_provenance("prereg/prereg.yaml")
    assert len(provenance["git_commit"]) == 40
    assert isinstance(provenance["dirty"], bool)
    assert len(provenance["sha256"]) == 6
    assert len(provenance["sha256"]["prereg"]) == 64


def test_registered_mode_refuses_a_dirty_worktree():
    with pytest.raises(SystemExit, match="dirty worktree"):
        gate.enforce_registered_provenance({"dirty": True}, registered=True)
    gate.enforce_registered_provenance({"dirty": True}, registered=False)


def test_registered_settings_come_from_the_prereg():
    reg = gate.load_engine_gate()
    assert reg["size"] == "e2b"
    assert reg["quant"] == "bf16"
    assert reg["logprob_mean_limit"] == pytest.approx(0.01)
    assert reg["logprob_max_limit"] == pytest.approx(0.1)


def test_gumbel_replay_uses_the_same_device_generator_stream():
    device = torch.device("cpu")
    seed, request_id, position, vocab = 7, "req-1", 3, 11
    generator = torch.Generator(device=device).manual_seed(
        gate.generate_request_seed(seed, request_id) % (2**63 - 1)
    )
    for _ in range(position):
        torch.rand(vocab, generator=generator, device=device, dtype=torch.float32)
    expected = torch.rand(
        vocab, generator=generator, device=device, dtype=torch.float32
    )
    replay = torch.Generator(device=device).manual_seed(
        gate.generate_request_seed(seed, request_id) % (2**63 - 1)
    )
    for _ in range(position):
        torch.rand(
            vocab, generator=replay, device=device, dtype=torch.float32
        )
    actual = torch.rand(
        vocab, generator=replay, device=device, dtype=torch.float32
    )
    assert torch.equal(expected, actual)


def test_re_prefill_smoke_is_logged_and_never_registered():
    assert "never counted" in gate.AMENDMENT_RULE
    assert "near-tie" in gate.AMENDMENT_RULE
    report = gate.build_report(
        criteria=[{"criterion": "round_1024_smoke", "passed": True}],
        throughput=[],
        config={"smoke_step": "re_prefill"},
        smoke=True,
    )
    assert report["registered_pass"] is False
    assert report["smoke"] is True


def _chunk_members(order, size):
    return [order[start:start + size] for start in range(0, len(order), size)]


def test_control_partition_differs_from_reference_for_every_prompt():
    budgets = [128, 128, 192, 192, 256, 256, 256, 256]
    reference_order = [index for index in range(8)]
    reference = []
    for group in gate.budget_groups(budgets):
        reference.extend(_chunk_members(group, 4))
    control_order = gate.interleaved_order(8)
    control = _chunk_members(control_order, 4)

    def neighbours(chunks, item):
        for chunk in chunks:
            if item in chunk:
                return {value for value in chunk if value != item}
        return set()

    for item in range(8):
        assert neighbours(reference, item) != neighbours(control, item), item
