"""CPU tests for the qualification script's row/report plumbing."""

from __future__ import annotations

import json

import torch

from scripts.qualify_teacher import (
    QualificationRow,
    _parse_args,
    teacher_spec,
    load_rows,
    prediction_slice,
    prepare_rows,
    render_markdown,
    select_rows,
)
from gemmaloop.teacher.qualification import evaluate_acceptance


def test_load_rows_accepts_rendered_input_ids_and_messages(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"input_ids": [1, 2, 3], "labels_start": 2})
        + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        + "\n",
        encoding="utf-8",
    )
    rows = load_rows(path)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 2, 3]


def test_prepare_rows_uses_rendered_ids_and_assistant_labels():
    rows = [{"input_ids": [1, 2, 3, 4, 5], "labels_start": 2, "source": "unit"}]
    prepared = prepare_rows(
        None,
        rows,
        4,
        thought_open_ids=[90],
        thought_close_ids=[91],
    )
    assert prepared[0].input_ids == [1, 2, 3, 4]
    assert prepared[0].labels == [-100, -100, 3, 4]
    assert prepared[0].length == 4
    assert prepared[0].source == "unit"
    assert prepared[0].spans == (None, None, "answer", "answer")


def test_prediction_slice_uses_the_predecessor_state():
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    prediction, actual, positions = prediction_slice(
        hidden, [-100, 2, -100, 4, 5]
    )
    assert positions.tolist() == [1, 3, 4]
    assert actual.tolist() == [2, 4, 5]
    assert torch.equal(prediction, hidden[[0, 2, 3]])


def _row(index: int, length: int) -> QualificationRow:
    return QualificationRow(
        row_id=index,
        input_ids=list(range(length)),
        labels=[-100] + list(range(1, length)),
    )


def test_select_rows_returns_an_independent_stratified_screen():
    lengths = [100, 1500, 1500, 3000, 8000] * 24
    prepared = [_row(index, length) for index, length in enumerate(lengths)]
    screen, expansion = select_rows(prepared, screen_rows=20, expand_rows=100)
    assert len(screen) == 20 and len(expansion) == 100
    labels = [label for _, label in screen]
    assert labels.count("short") == 4
    assert labels.count("common") == 8
    assert labels.count("medium") == 4
    assert labels.count("long") == 4
    assert {row.row_id for row, _ in screen} <= {
        row.row_id for row, _ in expansion
    }
    # The screen does not depend on the expansion configuration.
    screen_only, _ = select_rows(prepared, screen_rows=20, expand_rows=20)
    assert [row.row_id for row, _ in screen_only] == [
        row.row_id for row, _ in screen
    ]


def test_screen_uses_the_largest_long_rows_and_spans_the_common_range():
    lengths = []
    lengths += [200, 250, 300, 350, 400, 450]                    # short
    lengths += list(range(1000, 2050, 50))                        # common
    lengths += [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]   # medium
    lengths += [6500, 7000, 7500, 7800, 8000, 8100, 8150, 8192]   # long
    prepared = [_row(index, length) for index, length in enumerate(lengths)]
    screen, _ = select_rows(prepared, screen_rows=20, expand_rows=100)
    by_label = {}
    for row, label in screen:
        by_label.setdefault(label, []).append(row.length)
    assert max(by_label["short"]) <= 450
    assert min(by_label["common"]) <= 1050 and max(by_label["common"]) >= 2000
    assert min(by_label["long"]) >= 8000
    assert sorted(by_label["long"], reverse=True) == by_label["long"]


def test_parse_args_requires_one_teacher_and_rejects_multi_teacher_loops():
    args = _parse_args(["--teacher", "e4b", "--out", "unused"])
    assert args.teacher == "e4b"
    assert args.screen_rows == 20 and args.expand_rows == 100
    import pytest

    with pytest.raises(SystemExit):
        _parse_args(["--out", "unused"])


def test_render_markdown_contains_the_acceptance_table_and_identity():
    overall = {
        "mean_full_kl": 0.001,
        "mean_full_kl_ci_high": 0.002,
        "p95_full_kl": 0.005,
        "top1_agreement": 0.999,
        "top1_confident": 1.0,
        "tail_abs_error_mean": 0.0001,
        "tail_abs_error_p95": 0.0002,
        "invalid_tokens": 0,
        "all_finite": True,
        "by_stratum": {},
        "span": {},
    }
    decision = evaluate_acceptance(
        overall, memory={"headroom_gib": 1.0, "steady": True}
    )
    report = {
        "teacher": {"key": "e4b", "runtime": "bf16"},
        "corpus": {
            "path": "rows.jsonl",
            "note": "dev stand-in",
            "screen_rows": 20,
            "expand_rows": 100,
        },
        "generated_at": "now",
        "decision": decision.to_dict(),
        "metrics": {"screen": overall, "expanded": None},
        "memory": {"headroom_gib": 1.0, "steady": True},
        "runtime_identity": {"runtime": "bf16", "method": "bf16"},
    }
    markdown = render_markdown(report)
    assert "# Teacher qualification: e4b / bf16" in markdown
    assert "| criterion |" in markdown
    assert "mean_full_kl" in markdown
    assert '"method": "bf16"' in markdown


def test_build_hidden_cache_accepts_a_batched_teacher_hidden_fn():
    from scripts.qualify_teacher import build_hidden_cache

    row = QualificationRow(
        row_id=7,
        input_ids=[1, 2, 3, 4, 5, 6],
        labels=[-100, -100, 3, 4, -100, 6],
        spans=(None, None, "answer", "answer", None, "answer"),
    )
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)

    def batched_hidden(input_ids):
        assert input_ids.shape[0] == 1
        return hidden.unsqueeze(0)

    cache = build_hidden_cache(
        [row], batched_hidden, device="cpu", phase="test"
    )
    stored = cache[7]
    # Prediction positions are 2, 3 and 5, so the predecessor states are 1, 2 and 4.
    assert stored["positions"].tolist() == [2, 3, 5]
    assert stored["actual"].tolist() == [3, 4, 6]
    assert torch.equal(stored["hidden"], hidden[[1, 2, 4]])
    assert stored["spans"] == ["answer", "answer", "answer"]


def test_cached_12b_config_selects_the_unified_reference_family():
    import pytest

    transformers = pytest.importorskip("transformers")
    from gemmaloop.data.teacher_topk import teacher_spec
    from gemmaloop.teacher.reference import reference_family

    pytest.importorskip("huggingface_hub")
    spec = teacher_spec("12b_qat")
    try:
        config = transformers.AutoConfig.from_pretrained(
            spec.model_id, revision=spec.revision, local_files_only=True
        )
    except (OSError, ValueError):
        pytest.skip("pinned 12B config.json is not in the local cache")
    text = getattr(config, "text_config", config)
    assert "unified" in str(text.model_type)
    assert reference_family(text).name == "gemma4_unified"
    # The qualification reads these fields before streaming; they must exist.
    assert int(text.num_hidden_layers) == 48
    assert int(text.hidden_size) == 3840
    assert int(getattr(text, "hidden_size_per_layer_input", 0)) == 0
    assert float(text.final_logit_softcapping) == 30.0
    from scripts.qualify_teacher import _text_config

    configured = _text_config(spec)
    assert "unified" in str(configured.model_type)
    assert configured._attn_implementation == "sdpa"


def test_memory_smoke_runs_on_a_fake_cpu_teacher():
    from types import SimpleNamespace

    from scripts.qualify_teacher import _memory_smoke

    weight = torch.randn(256, 8)
    hidden = torch.randn(8, 8)

    def fake_hidden(input_ids):
        return hidden.unsqueeze(0)

    loaded = SimpleNamespace(
        hidden=fake_hidden,
        weight=weight,
        softcap=None,
    )
    result = _memory_smoke(
        loaded,
        max_length=8,
        top_k=4,
        token_chunk=4,
        vocab_chunk=64,
        mask_token_id=None,
    )
    assert result["length"] == 8
    assert result["tokens"] == 7
    assert result["peak_allocated_gib"] == 0.0


def test_build_reference_cache_uses_the_streamed_unified_reference(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextModel,
    )

    import scripts.qualify_teacher as script

    config = Gemma4UnifiedTextConfig(
        num_hidden_layers=3,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        vocab_size=128,
        sliding_window=8,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = Gemma4UnifiedTextModel(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(script, "_text_config", lambda spec: config)
    rows = [
        QualificationRow(
            row_id=0,
            input_ids=[2, 5, 7, 11, 13],
            labels=[-100, 5, 7, 11, 13],
            item_id="r0",
            spans=(None, "answer", "answer", "answer", "answer"),
        )
    ]
    args = SimpleNamespace(
        device="cpu", reference_ram_limit_gib=None, reference_vram_limit_gib=None
    )
    cache, memory = script.build_reference_cache(
        script.teacher_spec("12b_qat"), rows, args=args, heartbeat=None
    )
    assert cache is not None and memory is not None
    assert memory["kind"] == "layer-streamed-bf16"
    assert memory["family"] == "gemma4_unified"
    assert cache[0]["hidden"].shape == (4, config.hidden_size)
    assert cache[0]["actual"].tolist() == [5, 7, 11, 13]


def test_e4b_int8_reference_cache_path_end_to_end(monkeypatch):
    from types import SimpleNamespace

    import scripts.precompute_teacher_topk as precompute
    import scripts.qualify_teacher as script
    from gemmaloop.teacher.qualification import pair_token_metrics

    hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)

    class FakeTeacher:
        def hidden(self, input_ids):
            assert input_ids.shape[0] == 1
            return hidden.unsqueeze(0)

    def fake_load_teacher(name, device="cuda", runtime=None):
        assert name == "e4b" and runtime == "bf16"
        return FakeTeacher()

    monkeypatch.setattr(precompute, "load_teacher", fake_load_teacher)
    row = QualificationRow(
        row_id=11,
        input_ids=[1, 2, 3, 4, 5, 6],
        labels=[-100, -100, 3, 4, -100, 6],
        item_id="r11",
        spans=(None, None, "answer", "answer", None, "answer"),
    )
    spec = script.teacher_spec("e4b", runtime="int8")
    args = SimpleNamespace(device="cpu")
    cache, memory = script.build_reference_cache(
        spec, [row], args=args, heartbeat=None
    )
    assert memory == {"kind": "resident-bf16"}
    stored = cache[11]
    assert stored["hidden"].shape == (3, 4)
    assert torch.equal(stored["hidden"].float(), hidden[[1, 2, 4]])

    weight = torch.randn(32, 4)
    q_hidden = stored["hidden"].float()
    metrics = pair_token_metrics(
        q_hidden,
        q_hidden,
        weight,
        stored["actual"],
        top_k=4,
        mask_token_id=None,
        token_chunk=2,
        vocab_chunk=7,
    )
    assert metrics["kl"].shape == (3,)
    assert bool(torch.isfinite(metrics["kl"]).all())


def _write_corpus_and_manifest(tmp_path, record):
    rows = tmp_path / "corpus.jsonl"
    rows.write_text(json.dumps({"input_ids": [1, 2, 3], "labels_start": 1}) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps(record) + "\n")
    return rows


def _pool_with_manifest(tmp_path, sha):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text(
        json.dumps({"manifest_sha256": sha}) + "\n"
    )
    return pool


def test_corpus_disjointness_manifest_requires_the_stage0_record(tmp_path):
    import pytest

    import scripts.qualify_teacher as script

    pool = _pool_with_manifest(tmp_path, "a" * 64)
    rows = _write_corpus_and_manifest(tmp_path, {"kind": "p1lite_corpus"})
    with pytest.raises(script.CorpusManifestError, match="Stage 0 disjointness"):
        script.check_corpus_disjointness(rows, pool)


def test_corpus_disjointness_manifest_refuses_a_different_pool(tmp_path):
    import pytest

    import scripts.qualify_teacher as script

    pool = _pool_with_manifest(tmp_path, "b" * 64)
    rows = _write_corpus_and_manifest(
        tmp_path,
        {
            "stage0_disjointness": {
                "filter_version": "p1lite-stage0-disjoint-v1",
                "pool_manifest_sha256": "a" * 64,
            }
        },
    )
    with pytest.raises(script.CorpusManifestError, match="different pool manifest"):
        script.check_corpus_disjointness(rows, pool)


def test_corpus_disjointness_manifest_accepts_the_matching_pool(tmp_path):
    import scripts.qualify_teacher as script

    pool = _pool_with_manifest(tmp_path, "c" * 64)
    rows = _write_corpus_and_manifest(
        tmp_path,
        {
            "stage0_disjointness": {
                "filter_version": "p1lite-stage0-disjoint-v1",
                "pool_manifest_sha256": "c" * 64,
            }
        },
    )
    record = script.check_corpus_disjointness(rows, pool)
    assert record["pool_manifest_sha256"] == "c" * 64


def test_qualify_main_exits_2_on_a_corpus_without_the_stage0_record(
    tmp_path, capsys
):
    import pytest

    import scripts.qualify_teacher as script

    pool = _pool_with_manifest(tmp_path, "d" * 64)
    rows = _write_corpus_and_manifest(tmp_path, {"kind": "p1lite_corpus"})
    with pytest.raises(SystemExit) as caught:
        script.main(
            [
                "--teacher",
                "e4b",
                "--runtime",
                "bf16",
                "--rows",
                str(rows),
                "--out",
                str(tmp_path / "out"),
                "--stage0-pool",
                str(pool),
                "--skip-gpu-lock",
            ]
        )
    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "teacher qualification refused" in stderr
    assert "Stage 0 disjointness" in stderr


def test_corpus_disjointness_accepts_the_stage0_local_corpus_kind(tmp_path):
    import scripts.qualify_teacher as script

    pool = _pool_with_manifest(tmp_path, "e" * 64)
    rows = _write_corpus_and_manifest(
        tmp_path,
        {
            "kind": "p1lite_stage0_local",
            "stage0_disjointness": {
                "filter_version": "p1lite-stage0-protected-v2",
                "pool_manifest_sha256": "e" * 64,
                "protected_sets": {},
            },
        },
    )
    record = script.check_corpus_disjointness(rows, pool)
    assert record["filter_version"] == "p1lite-stage0-protected-v2"
