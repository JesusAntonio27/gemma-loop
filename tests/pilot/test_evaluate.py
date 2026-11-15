"""Evaluation plumbing: per-(arm, checkpoint, N) temperature and cluster NLL."""

from __future__ import annotations

import json

import pytest

from gemmaloop.pilot.arms import ArmSpec
from gemmaloop.pilot.statistics import arm_endpoints, cluster_mean
from gemmaloop.pilot.evaluate import (
    TeacherKlAlignmentError,
    _student_log_probs,
    build_arm_summary,
    evaluate_arm,
    load_eval_rows,
    KL_SUBSAMPLE_ROWS,
    matched_teacher_rows,
    select_kl_rows,
    teacher_shard_stats,
)


def _write_rows(path, clusters, length=8):
    with path.open("w", encoding="utf-8") as handle:
        for index, cluster in enumerate(clusters):
            ids = [((index + 1) * 7 + j) % 256 for j in range(length)]
            handle.write(
                json.dumps(
                    {
                        "row_id": index,
                        "cluster_id": cluster,
                        "input_ids": ids,
                        "labels_start": 2,
                        "answer_start": 5,
                    }
                )
                + "\n"
            )


def _arm() -> ArmSpec:
    return ArmSpec(
        arm_id="L-25-29-gch-e4b",
        kind="L",
        block=(25, 29),
        bridge="gch",
        teacher="e4b",
        registered_teacher="e4b",
    )


def test_student_log_probs_match_a_full_softmax_chunked():
    import torch

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(5, 4, generator=generator)
    weight = torch.randn(11, 4, generator=generator)
    indices = torch.randint(0, 11, (5, 3), generator=generator)
    log_q, tail_q = _student_log_probs(hidden, weight, indices, None, vocab_chunk=3)
    full = torch.softmax(hidden.float() @ weight.float().t(), dim=-1)
    expected = full.gather(1, indices).log()
    assert torch.allclose(log_q, expected, atol=1e-5)
    assert torch.allclose(tail_q, 1.0 - full.gather(1, indices).sum(dim=1), atol=1e-5)


def test_teacher_shard_stats_reads_entropy_and_top1(tmp_path):
    import torch
    from gemmaloop.data.teacher_topk import (
        ShardEntry,
        TeacherManifest,
        TeacherRow,
        write_teacher_shard,
    )

    shard = tmp_path / "shard-00000.pt"
    probs = torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float16)
    indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    tail = torch.tensor([0.0, 0.0], dtype=torch.float32)
    digest = write_teacher_shard(
        shard,
        [
            TeacherRow(
                row_id=0,
                positions=torch.tensor([1, 2], dtype=torch.int32),
                indices=indices,
                probs=probs,
                tail=tail,
            )
        ],
    )
    manifest = TeacherManifest(
        teacher={"key": "e4b"},
        data_manifest_sha256="x",
        top_k=2,
        temperature=1.0,
        mask_token="",
        mask_token_id=None,
        rows_total=1,
        tokens_total=2,
        shards=(ShardEntry(0, shard.name, digest, (0,), 2),),
    )
    manifest.write(tmp_path / "teacher_manifest.json")
    stats = teacher_shard_stats(tmp_path)
    assert stats["top1_confidence"] == pytest.approx(0.625)
    assert stats["tokens"] == 2
    assert stats["entropy"] > 0
    assert teacher_shard_stats(tmp_path / "missing") == {}


def test_token_weighted_calibrated_nll_matches_direct_computation():
    # C_N = sum_i S_iN / sum_i n_i.  Problem p1 carries 9 times the tokens of p0.
    nll_by_n = {
        n: {
            "p0": {"sum_loss": 1.0 * n, "tokens": 10.0},
            "p1": {"sum_loss": 9.0 * n, "tokens": 90.0},
        }
        for n in (1, 2, 3, 4, 6, 8)
    }
    expected = {n: (10.0 * n) / 100.0 for n in (1, 2, 3, 4, 6, 8)}
    assert cluster_mean(nll_by_n[1]) == pytest.approx(expected[1])
    oh = {1: {"p0": {"sum_loss": 11.0, "tokens": 10.0}, "p1": {"sum_loss": 99.0, "tokens": 90.0}}}
    stats = arm_endpoints(nll_by_n, oh, iters=2000, seed=0)
    assert stats["nll"]["1"] == pytest.approx(expected[1])
    assert stats["delta_loop"] == pytest.approx(expected[1] - expected[2])
    assert stats["delta_arm"] == pytest.approx(1.1 - expected[2])
    assert stats["delta_bar_arm"] == pytest.approx(
        (1.1 - expected[2] + 1.1 - expected[3] + 1.1 - expected[4]) / 3
    )
    # Equal-problem averaging would give 0.1 here; token weighting also gives
    # 0.1 only because the loss scales exactly with the token count.  A second
    # fixture shows the two estimands differ.
    unequal = {
        "p0": {"sum_loss": 1.0, "tokens": 10.0},
        "p1": {"sum_loss": 1.0, "tokens": 90.0},
    }
    assert cluster_mean(unequal) == pytest.approx(0.02)
    assert cluster_mean(unequal) != pytest.approx((1.0 / 10 + 1.0 / 90) / 2)


def test_evaluate_arm_computes_teacher_kl_per_n_on_heal_rows(tmp_path):
    fit = tmp_path / "fit.jsonl"
    test = tmp_path / "test.jsonl"
    heal = tmp_path / "heal.jsonl"
    _write_rows(fit, ["a"])
    _write_rows(test, ["p0"])
    _write_rows(heal, ["h0"])
    fit_rows = load_eval_rows(fit)
    test_rows = load_eval_rows(test)
    heal_rows = load_eval_rows(heal)
    kl_calls = []

    def load_fn(checkpoint):
        return checkpoint

    def fit_fn(model, batches, n):
        return 1.0

    def score_fn(model, batches, n, temperature):
        return {"clusters": {row.cluster: 1.0 for row, _ in batches}, "tokens": 1}

    def kl_fn(model, rows, teacher_dir, n):
        kl_calls.append((model, len(rows), str(teacher_dir), n))
        return {1: 5.0, 2: 2.0, 4: 1.0, 8: 0.5}[n] * (1.0 if model == "final" else 2.0)

    result = evaluate_arm(
        _arm(),
        tmp_path / "run",
        fit_rows=fit_rows,
        test_rows=test_rows,
        n_values=(1, 2, 4, 8),
        checkpoints=("initial", "final"),
        load_fn=load_fn,
        fit_fn=fit_fn,
        score_fn=score_fn,
        teacher_dir=tmp_path / "teacher",
        heal_rows=heal_rows,
        kl_fn=kl_fn,
    )
    assert [(call[0], call[3]) for call in kl_calls] == [
        ("initial", 1),
        ("initial", 2),
        ("initial", 4),
        ("initial", 8),
        ("final", 1),
        ("final", 2),
        ("final", 4),
        ("final", 8),
    ]
    teacher_kl = result["teacher_kl"]
    assert teacher_kl["n_values"] == [1, 2, 4, 8]
    assert teacher_kl["initial"]["2"] == 4.0
    assert teacher_kl["final"]["2"] == 2.0
    assert teacher_kl["gap_closed"]["2"] == pytest.approx(0.5)
    assert "structurally the original" in teacher_kl["note"]


def test_teacher_kl_uses_the_fixed_64_row_subsample(tmp_path):
    fit = tmp_path / "fit.jsonl"
    test = tmp_path / "test.jsonl"
    heal = tmp_path / "heal.jsonl"
    _write_rows(fit, ["a"])
    _write_rows(test, ["p0"])
    _write_rows(heal, [f"h{i}" for i in range(100)], length=5)
    fit_rows = load_eval_rows(fit)
    test_rows = load_eval_rows(test)
    heal_rows = load_eval_rows(heal)
    selected = select_kl_rows(heal_rows)
    assert len(selected) == KL_SUBSAMPLE_ROWS == 64
    assert [row.row_id for row in selected] == list(range(64))
    assert select_kl_rows(list(reversed(heal_rows))) == selected
    calls = []

    def kl_fn(model, rows, teacher_dir, n):
        calls.append((model, len(rows), n))
        return 1.0

    result = evaluate_arm(
        _arm(),
        tmp_path / "run",
        fit_rows=fit_rows,
        test_rows=test_rows,
        n_values=(1, 2, 3, 4, 6, 8),
        checkpoints=("initial", "final"),
        load_fn=lambda checkpoint: checkpoint,
        fit_fn=lambda model, batches, n: 1.0,
        score_fn=lambda model, batches, n, temperature: {
            "clusters": {row.cluster: 1.0 for row, _ in batches},
            "tokens": 1,
        },
        teacher_dir=tmp_path / "teacher",
        heal_rows=heal_rows,
        kl_fn=kl_fn,
    )
    assert len(calls) == 12
    assert all(count == 64 for _model, count, _n in calls)
    assert result["teacher_kl"]["rows"] == 64
    assert result["teacher_kl"]["role"] == "descriptive"
    assert result["teacher_kl"]["forward_passes"] == 12 * 64
    assert result["teacher_kl"]["cost_seconds"] >= 0.0


def test_teacher_kl_alignment_is_by_input_ids_hash_not_row_index(tmp_path):
    import torch
    from gemmaloop.data.teacher_topk import (
        ShardEntry,
        TeacherManifest,
        TeacherRow,
        input_ids_sha256,
        write_teacher_shard,
    )

    good_ids = [11, 12, 13]
    shard = tmp_path / "shard-00000.pt"
    digest = write_teacher_shard(
        shard,
        [
            TeacherRow(
                row_id=999,
                positions=torch.tensor([1, 2], dtype=torch.int32),
                indices=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                probs=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float16),
                tail=torch.tensor([0.0, 0.0], dtype=torch.float32),
                input_ids_sha256=input_ids_sha256(good_ids),
            )
        ],
    )
    TeacherManifest(
        teacher={"key": "e4b"},
        data_manifest_sha256="x",
        top_k=2,
        temperature=1.0,
        mask_token="",
        mask_token_id=None,
        rows_total=1,
        tokens_total=2,
        shards=(ShardEntry(0, shard.name, digest, (999,), 2),),
    ).write(tmp_path / "teacher_manifest.json")

    heal = tmp_path / "heal.jsonl"
    heal.write_text(
        json.dumps({"input_ids": good_ids, "cluster_id": "h0"})
        + "\n"
        + json.dumps({"input_ids": [91, 92, 93], "cluster_id": "h1"})
        + "\n",
        encoding="utf-8",
    )
    rows = load_eval_rows(heal)
    matched = matched_teacher_rows(rows, tmp_path)
    assert len(matched) == 1
    assert matched[0][0].cluster == "h0"
    with pytest.raises(TeacherKlAlignmentError):
        matched_teacher_rows([rows[1]], tmp_path)


def test_evaluate_arm_fits_temperature_per_checkpoint_and_n(tmp_path):
    fit = tmp_path / "fit.jsonl"
    test = tmp_path / "test.jsonl"
    _write_rows(fit, ["a", "b", "c"])
    _write_rows(test, ["p0", "p1", "p2", "p3"])
    fit_rows = load_eval_rows(fit)
    test_rows = load_eval_rows(test)

    fit_calls = []
    score_calls = []

    def load_fn(checkpoint):
        return {"checkpoint": checkpoint}

    def fit_fn(model, batches, n):
        fit_calls.append((model["checkpoint"], n, len(batches)))
        return 1.0 + 0.1 * n

    def score_fn(model, batches, n, temperature):
        score_calls.append((model["checkpoint"], n, temperature, len(batches)))
        return {
            "clusters": {row.cluster: 1.0 + 0.01 * n for row, _batch in batches},
            "buckets": {"<=1k": {"calibrated_nll": 1.0 + 0.01 * n, "tokens": 4}},
            "spans": {"thinking": 1.0, "answer": 2.0},
            "top1_accuracy": 0.9,
            "tokens": 4 * len(batches),
        }

    result = evaluate_arm(
        _arm(),
        tmp_path / "run",
        fit_rows=fit_rows,
        test_rows=test_rows,
        n_values=(1, 2, 4, 8),
        checkpoints=("initial", "final"),
        load_fn=load_fn,
        fit_fn=fit_fn,
        score_fn=score_fn,
    )

    # One temperature fit per (checkpoint, N); the score uses that temperature.
    assert len(fit_calls) == 2 * 4
    assert [(checkpoint, n) for checkpoint, n, _ in fit_calls] == [
        ("initial", 1),
        ("initial", 2),
        ("initial", 4),
        ("initial", 8),
        ("final", 1),
        ("final", 2),
        ("final", 4),
        ("final", 8),
    ]
    assert [call[2] for call in score_calls] == pytest.approx(
        [1.1, 1.2, 1.4, 1.8, 1.1, 1.2, 1.4, 1.8]
    )
    metrics = result["checkpoints"]["final"]["n"]
    assert set(metrics) == {"1", "2", "4", "8"}
    assert metrics["2"]["temperature"] == pytest.approx(1.2)
    assert metrics["2"]["clusters"] == {"p0": 1.02, "p1": 1.02, "p2": 1.02, "p3": 1.02}
    assert metrics["2"]["spans"]["thinking"] == 1.0


def test_build_arm_summary_uses_block_matched_oh_and_registered_endpoints():
    arm = _arm()
    nll = {
        n: {
            f"p{i}": 1.0
            + 0.01 * i
            + (0.2 if n == 1 else 0.1 if n == 2 else 0.3 if n == 8 else 0.05)
            for i in range(4)
        }
        for n in (1, 2, 3, 4, 6, 8)
    }
    oh = {1: {f"p{i}": 1.3 + 0.01 * i for i in range(4)}}
    result = {
        "checkpoints": {
            "final": {
                "n": {str(n): {"clusters": values, "temperature": 1.0} for n, values in nll.items()}
            },
            "initial": {"n": {"1": {"clusters": nll[1], "temperature": 1.0}}},
        },
        "training": {"loss": [{"step": 1, "loss": 2.0}], "tokens": 100, "tokens_per_second": 10.0},
    }
    oh_result = {
        "checkpoints": {
            "final": {"n": {"1": {"clusters": oh[1], "temperature": 1.0}}}
        }
    }
    summary = build_arm_summary(arm, result, oh_result, iters=500, seed=0)
    assert summary["arm_id"] == arm.arm_id
    assert set(summary["nll"]) == {"1", "2", "3", "4", "6", "8"}
    assert summary["delta_loop"] == pytest.approx(0.1)
    assert summary["delta_loop_lower"] > 0
    assert summary["delta_arm"] == pytest.approx((1.3 + 0.015) - 1.115)
    assert summary["delta_bar_arm"] == pytest.approx(
        ((1.3 + 0.015) - 1.115 + (1.3 + 0.015) - 1.065 + (1.3 + 0.015) - 1.065)
        / 3
    )
    assert summary["monotonic"] is False  # N=8 is worse than N=1 here
    assert summary["training"]["tokens"] == 100


def test_eval_batch_scores_exactly_the_training_label_mask(tmp_path):
    """N4: no extra newline token after end-of-turn is scored."""

    import torch

    from gemmaloop.pilot.evaluate import _row_batch

    training_labels = [-100, -100, 11, 12, 13, -100]
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "p0",
                "input_ids": [1, 2, 11, 12, 13, 14],
                "labels": training_labels,
                "labels_start": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = load_eval_rows(path)[0]
    assert row.labels == tuple(training_labels)
    batch = _row_batch(row)
    assert batch["labels"].tolist()[0] == training_labels
    scored = (batch["labels"][0, 1:] != -100).nonzero(as_tuple=False).flatten() + 1
    expected = [
        index
        for index in range(1, len(training_labels))
        if training_labels[index] != -100
    ]
    assert scored.tolist() == expected
    assert scored.tolist() == [2, 3, 4]
    assert len(scored) == sum(1 for value in training_labels if value != -100)

    # A labels-less row still falls back to the old labels_start behaviour.
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(
            {"problem_id": "p1", "input_ids": [1, 2, 11, 12, 13, 14], "labels_start": 2}
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_batch = _row_batch(load_eval_rows(legacy)[0])
    legacy_scored = (
        (legacy_batch["labels"][0, 1:] != -100).nonzero(as_tuple=False).flatten() + 1
    )
    assert legacy_scored.tolist() == [2, 3, 4, 5]  # the extra newline token
