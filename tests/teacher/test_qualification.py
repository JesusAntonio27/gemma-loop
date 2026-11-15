"""CPU tests for the teacher qualification metrics and acceptance gate."""

from __future__ import annotations

import json
import math

import pytest
import torch

from gemmaloop.teacher.qualification import (
    aggregate_metrics,
    bootstrap_mean_ci,
    evaluate_acceptance,
    pair_token_metrics,
    percentile,
    token_spans,
)


def _direct_metrics(
    ref_hidden: torch.Tensor,
    q_hidden: torch.Tensor,
    weight: torch.Tensor,
    actual_ids: torch.Tensor,
    *,
    top_k: int,
    mask_token_id: int | None,
    softcap: float | None,
) -> dict[str, torch.Tensor]:
    """Direct full-softmax reference for every per-token metric."""

    def distribution(hidden):
        logits = hidden.float() @ weight.float().t()
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        if mask_token_id is not None:
            logits = logits.clone()
            logits[:, mask_token_id] = float("-inf")
        logp = torch.log_softmax(logits.double(), dim=-1)
        return logits, logp, logp.exp()

    ref_logits, ref_logp, ref_p = distribution(ref_hidden)
    q_logits, q_logp, q_p = distribution(q_hidden)
    kl = torch.where(
        ref_p > 0, ref_p * (ref_logp - q_logp), torch.zeros_like(ref_p)
    ).sum(1)
    reverse = torch.where(
        q_p > 0, q_p * (q_logp - ref_logp), torch.zeros_like(q_p)
    ).sum(1)
    h_ref = -(
        ref_p * torch.where(ref_p > 0, ref_logp, torch.zeros_like(ref_logp))
    ).sum(1)
    h_q = -(q_p * torch.where(q_p > 0, q_logp, torch.zeros_like(q_logp))).sum(1)

    ref_idx = ref_logits.topk(top_k, dim=1).indices
    q_idx = q_logits.topk(top_k, dim=1).indices
    ref_at_ref = ref_p.gather(1, ref_idx)
    q_at_ref = q_p.gather(1, ref_idx)
    ref_at_q = ref_p.gather(1, q_idx)
    q_at_q = q_p.gather(1, q_idx)

    n = ref_hidden.shape[0]
    union_kl = torch.zeros(n, dtype=torch.float64)
    ref65 = torch.zeros(n, dtype=torch.float64)
    stored65 = torch.zeros(n, dtype=torch.float64)
    stored65_ref = torch.zeros(n, dtype=torch.float64)
    tail_err_stored = torch.zeros(n, dtype=torch.float64)

    q_top_probs = q_at_q.to(torch.float16).float()
    q_tail_raw = (1.0 - q_at_q.sum(1)).clamp_min(0.0)
    total = q_top_probs.sum(1) + q_tail_raw
    q_stored = q_top_probs / total.unsqueeze(1)
    q_stored_tail = q_tail_raw / total

    for row in range(n):
        ids = torch.unique(torch.cat([ref_idx[row], q_idx[row]]))
        p_union = ref_p[row, ids]
        q_union = q_p[row, ids]
        p_values = torch.cat([p_union, (1.0 - p_union.sum()).reshape(1)])
        q_values = torch.cat([q_union, (1.0 - q_union.sum()).reshape(1)])
        union_kl[row] = p_values.dot(torch.log(p_values) - torch.log(q_values))
        p_ref_support = torch.cat(
            [ref_at_ref[row], (1.0 - ref_at_ref[row].sum()).reshape(1)]
        )
        q_ref_support = torch.cat(
            [q_at_ref[row], (1.0 - q_at_ref[row].sum()).reshape(1)]
        )
        ref65[row] = p_ref_support.dot(
            torch.log(p_ref_support) - torch.log(q_ref_support)
        )
        p_qsupport = torch.cat(
            [ref_at_q[row], (1.0 - ref_at_q[row].sum()).reshape(1)]
        )
        q_qsupport = torch.cat([q_stored[row], q_stored_tail[row].reshape(1)])
        stored65[row] = p_qsupport.dot(
            torch.log(p_qsupport) - torch.log(q_qsupport)
        )
        q_matches_ref = (q_idx[row].unsqueeze(1) == ref_idx[row].unsqueeze(0)).any(1)
        q_stored_at_ref = (q_stored[row] * q_matches_ref).sum()
        tail_err_stored[row] = abs(
            (1.0 - ref_at_ref[row].sum()) - (1.0 - q_stored_at_ref)
        )
        q_stored_ref_ids = torch.zeros_like(ref_at_ref[row])
        for position, ref_id in enumerate(ref_idx[row].tolist()):
            matches = (q_idx[row] == ref_id).nonzero()
            if matches.numel():
                q_stored_ref_ids[position] = q_stored[row, matches[0, 0]]
        q_stored_ref_support = torch.cat(
            [
                q_stored_ref_ids,
                (1.0 - q_stored_ref_ids.sum()).reshape(1),
            ]
        )
        stored65_ref[row] = torch.where(
            p_ref_support > 0,
            p_ref_support
            * (
                torch.log(p_ref_support.clamp_min(1e-30))
                - torch.log(q_stored_ref_support.clamp_min(1e-30))
            ),
            torch.zeros_like(p_ref_support),
        ).sum()

    top1_match = ref_idx[:, 0] == q_idx[:, 0]
    gap = torch.log(ref_at_ref[:, 0]) - torch.log(ref_at_ref[:, 1])
    confident = gap >= 0.5
    overlap = torch.tensor(
        [
            len(set(ref_idx[row].tolist()) & set(q_idx[row].tolist()))
            for row in range(n)
        ],
        dtype=torch.long,
    )
    tail_err_ref = torch.abs((1 - ref_at_ref.sum(1)) - (1 - q_at_ref.sum(1)))
    tail_err_own = torch.abs((1 - ref_at_ref.sum(1)) - (1 - q_at_q.sum(1)))
    entropy_q_stored = -(q_stored * torch.log(q_stored)).sum(
        1
    ) - q_stored_tail * torch.log(q_stored_tail)

    ref_actual = ref_logits.gather(1, actual_ids.reshape(-1, 1)).squeeze(1).double()
    q_actual = q_logits.gather(1, actual_ids.reshape(-1, 1)).squeeze(1).double()
    ref_lse = torch.logsumexp(ref_logits.double(), 1)
    q_lse = torch.logsumexp(q_logits.double(), 1)
    logp_ref_actual = ref_actual - ref_lse
    logp_q_actual = q_actual - q_lse
    actual_in_q = (q_idx == actual_ids.reshape(-1, 1)).any(1)
    stored_prob = (q_stored * (q_idx == actual_ids.reshape(-1, 1))).sum(1)
    logp_stored = torch.where(
        actual_in_q,
        torch.log(stored_prob),
        torch.log(q_stored_tail),
    )
    stored_order = q_stored.argsort(1, descending=True)
    stored_top1 = q_idx.gather(1, stored_order)[:, 0]
    return {
        "kl": kl,
        "reverse_kl": reverse,
        "union_kl": union_kl,
        "ref65_kl": ref65,
        "stored65_kl": stored65,
        "stored65_ref_kl": stored65_ref,
        "top1_match": top1_match,
        "top1_match_stored": ref_idx[:, 0] == stored_top1,
        "confident": confident,
        "overlap": overlap,
        "ref_mass_at_q": ref_at_q.sum(1),
        "tail_err_ref": tail_err_ref,
        "tail_err_own": tail_err_own,
        "tail_err_stored": tail_err_stored,
        "entropy_ref": h_ref,
        "entropy_q": h_q,
        "entropy_q_stored": entropy_q_stored,
        "logp_actual_ref": logp_ref_actual,
        "logp_actual_q": logp_q_actual,
        "logp_actual_q_stored": logp_stored,
    }


@pytest.fixture(scope="module")
def metric_case():
    generator = torch.Generator().manual_seed(31)
    n, hidden, vocab, top_k = 13, 7, 79, 9
    ref_hidden = torch.randn(n, hidden, generator=generator)
    q_hidden = ref_hidden + 0.15 * torch.randn(n, hidden, generator=generator)
    weight = torch.randn(vocab, hidden, generator=generator)
    actual_ids = torch.randint(0, vocab, (n,), generator=generator)
    actual_ids = actual_ids[actual_ids != 17]
    filler = torch.randint(0, vocab, (n - actual_ids.numel(),), generator=generator)
    actual_ids = torch.cat([actual_ids, filler])
    return ref_hidden, q_hidden, weight, actual_ids, top_k


def test_every_metric_matches_a_direct_full_softmax(metric_case):
    ref_hidden, q_hidden, weight, actual_ids, top_k = metric_case
    got = pair_token_metrics(
        ref_hidden,
        q_hidden,
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=17,
        softcap=30.0,
        token_chunk=4,
        vocab_chunk=11,
    )
    expected = _direct_metrics(
        ref_hidden,
        q_hidden,
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=17,
        softcap=30.0,
    )
    for name in (
        "kl",
        "reverse_kl",
        "union_kl",
        "ref65_kl",
        "stored65_kl",
        "stored65_ref_kl",
        "ref_mass_at_q",
        "tail_err_ref",
        "tail_err_own",
        "tail_err_stored",
        "entropy_ref",
        "entropy_q",
        "entropy_q_stored",
        "logp_actual_ref",
        "logp_actual_q",
        "logp_actual_q_stored",
    ):
        torch.testing.assert_close(
            got[name].double(), expected[name].double(), atol=2e-5, rtol=2e-5
        )
    for name in ("top1_match", "top1_match_stored", "confident"):
        assert torch.equal(got[name], expected[name])
    assert torch.equal(got["overlap"], expected["overlap"])
    assert bool(got["valid"].all())


def test_metrics_without_masking_or_softcap(metric_case):
    ref_hidden, q_hidden, weight, actual_ids, top_k = metric_case
    got = pair_token_metrics(
        ref_hidden,
        q_hidden,
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=None,
        softcap=None,
        token_chunk=5,
        vocab_chunk=13,
    )
    expected = _direct_metrics(
        ref_hidden,
        q_hidden,
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=None,
        softcap=None,
    )
    torch.testing.assert_close(
        got["kl"].double(), expected["kl"].double(), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        got["union_kl"].double(),
        expected["union_kl"].double(),
        atol=2e-5,
        rtol=2e-5,
    )


def test_identical_hidden_states_have_zero_kl(metric_case):
    ref_hidden, _, weight, actual_ids, top_k = metric_case
    got = pair_token_metrics(
        ref_hidden,
        ref_hidden.clone(),
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=17,
        softcap=30.0,
        token_chunk=4,
        vocab_chunk=11,
    )
    torch.testing.assert_close(
        got["kl"], torch.zeros_like(got["kl"]), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        got["union_kl"], torch.zeros_like(got["union_kl"]), atol=1e-9, rtol=0
    )
    assert bool(got["top1_match"].all())


def test_metrics_mark_the_masked_actual_token_invalid(metric_case):
    ref_hidden, q_hidden, weight, _, top_k = metric_case
    actual_ids = torch.full((ref_hidden.shape[0],), 17, dtype=torch.long)
    got = pair_token_metrics(
        ref_hidden,
        q_hidden,
        weight,
        actual_ids,
        top_k=top_k,
        mask_token_id=17,
        token_chunk=4,
        vocab_chunk=11,
    )
    assert not bool(got["valid"].any())


def test_aggregate_metrics_reports_strata_spans_and_bootstrap():
    generator = torch.Generator().manual_seed(32)
    n = 40
    metrics = {
        "kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "reverse_kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "union_kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "ref65_kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "stored65_kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "stored65_ref_kl": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "top1_match": torch.rand(n, generator=generator) > 0.1,
        "top1_match_stored": torch.rand(n, generator=generator) > 0.1,
        "confident": torch.ones(n, dtype=torch.bool),
        "overlap": torch.randint(50, 65, (n,), generator=generator),
        "ref_mass_at_q": torch.rand(n, generator=generator, dtype=torch.float64),
        "tail_err_ref": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "tail_err_own": torch.rand(n, generator=generator, dtype=torch.float64) * 0.01,
        "tail_err_stored": torch.rand(n, generator=generator, dtype=torch.float64),
        "entropy_ref": torch.rand(n, generator=generator, dtype=torch.float64),
        "entropy_q": torch.rand(n, generator=generator, dtype=torch.float64),
        "entropy_q_stored": torch.rand(n, generator=generator, dtype=torch.float64),
        "logp_actual_ref": -torch.rand(n, generator=generator, dtype=torch.float64),
        "logp_actual_q": -torch.rand(n, generator=generator, dtype=torch.float64),
        "logp_actual_q_stored": -torch.rand(
            n, generator=generator, dtype=torch.float64
        ),
        "valid": torch.ones(n, dtype=torch.bool),
    }
    row_ids = [index // 2 for index in range(n)]
    strata = ["short"] * 10 + ["common"] * 20 + ["long"] * 10
    spans = ["thinking" if index % 2 else "answer" for index in range(n)]
    result = aggregate_metrics(
        metrics,
        row_ids,
        strata=strata,
        spans=spans,
        bootstrap_samples=100,
        seed=7,
    )
    assert result["tokens"] == n
    assert set(result["by_stratum"]) == {"short", "common", "long"}
    assert set(result["span"]) == {"thinking", "answer"}
    assert result["mean_full_kl_ci_low"] <= result["mean_full_kl"]
    assert result["mean_full_kl"] <= result["mean_full_kl_ci_high"]
    assert math.isfinite(result["p95_full_kl"])


def test_bootstrap_is_token_weighted_and_deterministic():
    low_a, high_a = bootstrap_mean_ci([0.0, 1.0], [100, 1], samples=200, seed=3)
    low_b, high_b = bootstrap_mean_ci([0.0, 1.0], [100, 1], samples=200, seed=3)
    assert (low_a, high_a) == (low_b, high_b)
    assert low_a <= 0.5 <= high_a
    assert percentile(torch.tensor([0.0, 1.0, 2.0]), 0.5) == pytest.approx(1.0)


def _overall(**overrides):
    values = {
        "mean_full_kl": 0.002,
        "mean_full_kl_ci_high": 0.003,
        "p95_full_kl": 0.010,
        "top1_agreement": 0.999,
        "top1_confident": 1.0,
        "tail_abs_error_mean": 0.0001,
        "tail_abs_error_p95": 0.0005,
        "invalid_tokens": 0,
        "all_finite": True,
        "by_stratum": {},
        "span": {},
    }
    values.update(overrides)
    return values


_MEMORY = {"headroom_gib": 1.0, "steady": True}


def test_acceptance_passes_comfortably():
    result = evaluate_acceptance(_overall(), memory=_MEMORY, n_rows=20, max_rows=100)
    assert result.passed and result.decision == "pass"
    assert result.next_runtime is None


def test_acceptance_expands_once_near_a_boundary():
    result = evaluate_acceptance(
        _overall(mean_full_kl=0.009), memory=_MEMORY, n_rows=20, max_rows=100
    )
    assert result.decision == "expand" and result.passed


def test_acceptance_expands_for_a_length_specific_failure():
    result = evaluate_acceptance(
        _overall(
            by_stratum={
                "long": {
                    "mean_full_kl": 0.02,
                    "p95_full_kl": 0.01,
                    "top1_agreement": 0.99,
                }
            }
        ),
        memory=_MEMORY,
        n_rows=20,
        max_rows=100,
    )
    assert result.decision == "expand"


def test_acceptance_fails_to_the_registered_fallback_after_expansion():
    result = evaluate_acceptance(
        _overall(
            by_stratum={
                "long": {
                    "mean_full_kl": 0.02,
                    "p95_full_kl": 0.01,
                    "top1_agreement": 0.99,
                }
            }
        ),
        memory=_MEMORY,
        teacher_key="e4b",
        current_runtime="bf16",
        n_rows=100,
        max_rows=100,
    )
    assert not result.passed
    assert result.decision == "fail_next_runtime"
    assert result.next_runtime == "int8"


def test_acceptance_clear_failure_moves_on_without_expansion():
    result = evaluate_acceptance(
        _overall(mean_full_kl=0.5),
        memory=_MEMORY,
        teacher_key="12b_qat",
        current_runtime="nf4",
        n_rows=20,
        max_rows=100,
    )
    assert result.decision == "fail_next_runtime"
    assert result.next_runtime == "q4_0_gguf"


def test_acceptance_refuses_insufficient_headroom():
    result = evaluate_acceptance(
        _overall(), memory={"headroom_gib": 0.1, "steady": True}, n_rows=20
    )
    assert not result.passed and result.decision == "fail_next_runtime"


def test_token_spans_classifies_thinking_markers_and_unlabelled_positions():
    thinking_open = [90, 91]
    thinking_close = [92]
    input_ids = [1, 90, 91, 5, 6, 92, 7, -100, 8]
    labels = [-100, 90, 91, 5, 6, 92, 7, -100, 8]
    spans = token_spans(
        input_ids,
        labels,
        thought_open_ids=thinking_open,
        thought_close_ids=thinking_close,
    )
    assert spans[0] is None
    assert spans[1] == "thinking" and spans[2] == "thinking"
    assert spans[3] == "thinking" and spans[4] == "thinking"
    assert spans[5] == "thinking"
    assert spans[6] == "answer"
    assert spans[7] is None
    assert spans[8] == "answer"


def test_stratified_selection_is_deterministic_and_balanced():
    from gemmaloop.teacher.qualification import stratify_indices

    lengths = [100, 1500, 1500, 3000, 8000] * 20
    first = stratify_indices(lengths)
    second = stratify_indices(lengths)
    assert first == second
    assert len(first) == 100
    assert len({index for index, _ in first}) == 100
    labels = [label for _, label in first[:20]]
    assert labels.count("short") == 4
    assert labels.count("common") == 8
    assert labels.count("medium") == 4
    assert labels.count("long") == 4


def test_stratified_selection_handles_a_small_corpus():
    from gemmaloop.teacher.qualification import stratify_indices

    selected = stratify_indices([100, 1500, 3000], n_short=4, n_common=8, n_medium=4, n_long=4)
    assert len(selected) == 3
    assert len({index for index, _ in selected}) == 3


def test_expand_without_enough_rows_is_inconclusive():
    from gemmaloop.teacher.qualification import (
        AcceptanceResult,
        evaluate_acceptance,
        resolve_decision,
    )

    screen = evaluate_acceptance(
        _overall(mean_full_kl=0.009), memory=_MEMORY, n_rows=20, max_rows=100
    )
    assert screen.decision == "expand"
    decision = resolve_decision(
        screen, None, selected_rows=20, expand_rows=100
    )
    assert decision.passed is False
    assert decision.decision == "insufficient_rows"
    part = resolve_decision(screen, None, selected_rows=63, expand_rows=100)
    assert part.passed is False and part.decision == "insufficient_rows"


def test_expand_that_remains_near_the_boundary_is_unresolved():
    from gemmaloop.teacher.qualification import (
        AcceptanceResult,
        evaluate_acceptance,
        resolve_decision,
    )

    screen = evaluate_acceptance(
        _overall(mean_full_kl=0.009), memory=_MEMORY, n_rows=20, max_rows=100
    )
    expanded = AcceptanceResult(passed=True, decision="expand")
    decision = resolve_decision(
        screen, expanded, selected_rows=100, expand_rows=100
    )
    assert decision.passed is False and decision.decision == "unresolved"


def test_resolve_decision_accepts_a_clear_expanded_result():
    from gemmaloop.teacher.qualification import (
        evaluate_acceptance,
        resolve_decision,
    )

    screen = evaluate_acceptance(
        _overall(mean_full_kl=0.009), memory=_MEMORY, n_rows=20, max_rows=100
    )
    expanded = evaluate_acceptance(
        _overall(mean_full_kl=0.002), memory=_MEMORY, n_rows=100, max_rows=100
    )
    decision = resolve_decision(
        screen, expanded, selected_rows=100, expand_rows=100
    )
    assert decision.passed and decision.decision == "pass"


def test_extract_row_id_and_protected_manifests(tmp_path):
    from gemmaloop.teacher.qualification import (
        ProtectionError,
        assert_disjoint,
        default_protected_sources,
        extract_row_id,
        load_protected_ids,
    )

    assert extract_row_id({"id": "a"}) == "a"
    assert extract_row_id({"problem_id": "b"}) == "b"
    assert extract_row_id({"extra_info": {"index": 7}}) == "7"
    assert extract_row_id({"prompt": "no id"}) is None

    pilot = tmp_path / "pilot_verified.jsonl"
    pilot.write_text('{"id": "p1"}\n{"row_id": "p2"}\n', encoding="utf-8")
    confirm = tmp_path / "confirmatory.json"
    confirm.write_text(json.dumps({"ids": ["c1", "c2"]}), encoding="utf-8")
    missing = tmp_path / "absent.jsonl"
    sources = {
        "pilot_evaluation": (pilot, missing),
        "confirmatory": (confirm,),
        "monitor": (missing,),
    }
    report, ids = load_protected_ids(sources)
    assert report["pilot_evaluation"]["status"] == "present"
    assert report["pilot_evaluation"]["ids"] == 2
    assert report["monitor"]["status"] == "absent"
    assert ids["confirmatory"] == {"c1", "c2"}

    proof = assert_disjoint(["x", "y"], ids, protected_report=report)
    assert proof["disjoint"] is True
    assert proof["sources"]["pilot_evaluation"]["intersection"] == 0
    with pytest.raises(ProtectionError, match="overlaps protected"):
        assert_disjoint(["p1", "x"], ids, protected_report=report)
    with pytest.raises(ProtectionError, match="duplicate"):
        assert_disjoint(["x", "x"], ids, protected_report=report)

    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ProtectionError, match="unreadable"):
        load_protected_ids({"confirmatory": (broken,)})
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ProtectionError, match="no row ids"):
        load_protected_ids({"confirmatory": (empty,)})

    registered = default_protected_sources(tmp_path / "stage0")
    assert {"pilot_evaluation", "temperature_fit", "monitor", "confirmatory"} <= set(
        registered
    )
    from gemmaloop.teacher.qualification import filter_disjoint

    _, unverified = filter_disjoint(
        ["x"], {"monitor": set()}, protected_report={"monitor": {"status": "absent"}}
    )
    assert unverified["verification"] == "unverified"
    assert unverified["protected_sources_absent"] == 1


def test_dapo_raw_id_matches_the_stage0_canonical_namespace(tmp_path):
    from gemmaloop.stage0.sources import _stable_id
    from gemmaloop.teacher.qualification import (
        canonical_dapo_id,
        extract_row_ids,
        filter_disjoint,
        load_protected_ids,
    )

    raw = "6ff0b17f-7e5c-4ae9-b5e9-63ebecd2b9f7"
    canonical = canonical_dapo_id(raw)
    assert canonical == f"dapo:{_stable_id(raw)}"
    p1_row = {"id": raw, "source": "BytedTsinghua-SIA/DAPO-Math-17k"}
    stage_row = {
        "problem_id": canonical,
        "source": "dapo",
        "metadata": {"raw_id": raw},
    }
    p1_aliases = extract_row_ids(p1_row)
    stage_aliases = extract_row_ids(stage_row)
    assert raw in p1_aliases and canonical in p1_aliases
    assert canonical in stage_aliases and raw in stage_aliases

    protected_file = tmp_path / "split_pilot_dev.jsonl"
    protected_file.write_text(json.dumps(stage_row) + "\n", encoding="utf-8")
    report, protected = load_protected_ids(
        {"pilot_evaluation": (protected_file,)}
    )
    other = {
        "id": "11111111-2222-4333-8444-555555555555",
        "source": "BytedTsinghua-SIA/DAPO-Math-17k",
    }
    kept, exclusion = filter_disjoint(
        [extract_row_ids(p1_row), extract_row_ids(other)],
        protected,
        protected_report=report,
    )
    assert kept == [1]
    assert exclusion["excluded_rows"] == 1
    assert exclusion["excluded_by_source"]["pilot_evaluation"] == 1
    assert exclusion["verification"] == "overlap-excluded"
    assert exclusion["sources"]["pilot_evaluation"]["intersection"] == 1


def test_load_protected_ids_keeps_u2028_inside_json_strings(tmp_path):
    from gemmaloop.teacher.qualification import load_protected_ids

    path = tmp_path / "split_pilot_dev.jsonl"
    row = {
        "id": "p1",
        "source": "apps",
        "prompt": "first line\u2028second line inside the same string",
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report, ids = load_protected_ids({"pilot_evaluation": (path,)})
    assert ids["pilot_evaluation"] == {"p1"}
    assert report["pilot_evaluation"]["rows"] == 1
