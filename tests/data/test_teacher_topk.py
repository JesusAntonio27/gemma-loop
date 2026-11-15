"""CPU tests for offline teacher top-k shards and collator integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from gemmaloop.teacher.runtime import runtime_identity

from gemmaloop.data.teacher_topk import (
    DEFAULT_MASK_TOKEN,
    GLITCH_TOKEN_ID,
    ShardEntry,
    TeacherIndex,
    TeacherManifest,
    TeacherManifestMismatch,
    TeacherRow,
    TeacherTopKCollator,
    assert_teacher_tokenizer_matches,
    compute_teacher_row,
    file_sha256,
    read_teacher_shard,
    teacher_spec,
    topk_distribution_from_hidden,
    validate_teacher_alignment,
    write_teacher_shard,
)


def test_teacher_specs_are_pinned_and_unknown_names_fail_closed():
    from gemmaloop.model.snapshot import teacher_model_id, teacher_revision

    spec = teacher_spec("E4B")
    assert spec.key == "e4b"
    assert spec.revision == "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    assert spec.runtime == "bf16"
    assert teacher_spec("12b_qat").runtime == "nf4"
    assert teacher_spec("e4b", runtime="int8").runtime == "int8"
    assert teacher_revision("12b_qat") == "b6ed86275a6a5735884e208bfed95b445a684ca2"
    assert teacher_model_id("12b_qat").endswith("qat-q4_0-unquantized")
    with pytest.raises(ValueError, match="unknown teacher"):
        teacher_spec("26b")


def _reference_topk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    mask_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = hidden.float() @ weight.float().t()
    if mask_token_id is not None and 0 <= mask_token_id < logits.shape[1]:
        logits[:, mask_token_id] = float("-inf")
    log_probs = torch.log_softmax(logits, dim=-1)
    values, indices = torch.topk(log_probs, top_k, dim=1)
    probs = values.exp()
    tail = (1.0 - probs.sum(dim=1)).clamp(min=0.0)
    return indices, probs, tail


def test_chunked_topk_matches_a_full_softmax_reference():
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(11, 6, generator=generator)
    weight = torch.randn(23, 6, generator=generator)
    indices, probs, tail = topk_distribution_from_hidden(
        hidden,
        weight,
        top_k=5,
        mask_token_id=7,
        token_chunk=3,
        vocab_chunk=4,
    )
    ref_indices, ref_probs, ref_tail = _reference_topk(hidden, weight, 5, 7)
    assert indices.dtype == torch.int32
    assert probs.dtype == torch.float16
    assert tail.dtype == torch.float32
    assert torch.equal(indices, ref_indices.to(torch.int32))
    torch.testing.assert_close(probs.float(), ref_probs, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(tail, ref_tail, atol=1e-6, rtol=1e-5)


def test_glitch_token_is_masked_and_the_distribution_renormalised():
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(9, 5, generator=generator)
    weight = torch.randn(17, 5, generator=generator)
    # Make the masked token the most likely one so masking must change the top-k.
    weight[GLITCH_TOKEN_ID % 17] = hidden[0] * 100
    mask_id = GLITCH_TOKEN_ID % 17
    indices, probs, tail = topk_distribution_from_hidden(
        hidden, weight, top_k=4, mask_token_id=mask_id, token_chunk=2, vocab_chunk=3
    )
    assert not (indices == mask_id).any()
    total = probs.float().sum(dim=1) + tail
    torch.testing.assert_close(total, torch.ones(9), atol=1e-3, rtol=0)
    ref = _reference_topk(hidden, weight, 4, mask_id)
    assert torch.equal(indices, ref[0].to(torch.int32))


def test_compute_teacher_row_stores_target_positions_and_teacher_at_previous_state():
    generator = torch.Generator().manual_seed(2)
    seq, hidden_size, vocab = 8, 5, 19
    hidden = torch.randn(seq, hidden_size, generator=generator)
    weight = torch.randn(vocab, hidden_size, generator=generator)
    labels = torch.tensor([-100, -100, 4, -100, 7, 8, -100, 11])
    row = compute_teacher_row(
        hidden,
        weight,
        labels,
        row_id=3,
        top_k=4,
        mask_token_id=5,
        token_chunk=3,
        vocab_chunk=4,
    )
    assert row.row_id == 3
    assert row.token_positions() == [2, 4, 5, 7]
    assert row.tokens == 4
    from gemmaloop.train.dial_objective import TeacherTopK

    TeacherTopK(row.probs, row.indices, row.tail).validate(vocab)
    # Target token i is scored by the teacher state at i - 1.
    ref = _reference_topk(hidden[[1, 3, 4, 6]], weight, 4, 5)
    assert torch.equal(row.indices, ref[0].to(torch.int32))
    torch.testing.assert_close(row.probs.float(), ref[1], atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(row.tail, ref[2], atol=1e-6, rtol=1e-5)


def test_top_k_must_fit_after_masking():
    hidden = torch.randn(2, 3)
    weight = torch.randn(5, 3)
    with pytest.raises(ValueError, match="does not fit"):
        topk_distribution_from_hidden(hidden, weight, top_k=5, mask_token_id=2)


def _row(row_id: int, labels: list[int], *, k: int = 4, vocab: int = 23) -> TeacherRow:
    generator = torch.Generator().manual_seed(row_id + 10)
    positions = [index for index, label in enumerate(labels) if label != -100 and index > 0]
    n = len(positions)
    if n == 0:
        return TeacherRow(
            row_id=row_id,
            positions=torch.zeros(0, dtype=torch.int32),
            indices=torch.zeros((0, k), dtype=torch.int32),
            probs=torch.zeros((0, k), dtype=torch.float16),
            tail=torch.zeros(0, dtype=torch.float32),
        )
    probs = torch.rand(n, k, generator=generator)
    probs = probs / probs.sum(dim=1, keepdim=True) * 0.8
    indices = torch.stack(
        [torch.randperm(vocab, generator=generator)[:k] for _ in range(n)]
    ).to(torch.int32)
    tail = (1.0 - probs.sum(dim=1)).to(torch.float32)
    return TeacherRow(
        row_id=row_id,
        positions=torch.tensor(positions, dtype=torch.int32),
        indices=indices,
        probs=probs.to(torch.float16),
        tail=tail,
    )


def _write_index(tmp_path, rows, *, data_sha="d" * 64, top_k=4):
    shards = []
    for index, group in enumerate([rows]):
        path = tmp_path / f"shard-{index:05d}.pt"
        digest = write_teacher_shard(path, group)
        shards.append(
            ShardEntry(
                index=index,
                file=path.name,
                sha256=digest,
                row_ids=tuple(row.row_id for row in group),
                tokens=sum(row.tokens for row in group),
            )
        )
    manifest = TeacherManifest(
        teacher={
            "key": "e4b",
            "model_id": "google/gemma-4-E4B-it",
            "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            "runtime": "bf16",
            "max_length": 16,
        },
        runtime_identity=runtime_identity("e4b", "bf16"),
        data_manifest_sha256=data_sha,
        top_k=top_k,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=GLITCH_TOKEN_ID,
        rows_total=len(rows),
        tokens_total=sum(row.tokens for row in rows),
        shards=tuple(shards),
    )
    manifest.write(tmp_path / "teacher_manifest.json")
    return manifest


def test_shard_and_manifest_round_trip(tmp_path):
    rows = [_row(0, [-100, 3, -100, 5]), _row(1, [-100, 2, 4, 6])]
    manifest = _write_index(tmp_path, rows)
    loaded = read_teacher_shard(tmp_path / "shard-00000.pt")
    assert [row.row_id for row in loaded] == [0, 1]
    assert torch.equal(loaded[0].indices, rows[0].indices)
    assert file_sha256(tmp_path / "shard-00000.pt") == manifest.shards[0].sha256

    index = TeacherIndex.from_directory(
        tmp_path, data_manifest_sha256="d" * 64, expect_top_k=4
    )
    assert index.row_ids() == {0, 1}
    batch = index.batch([1, 0], seq_len=6)
    assert batch["teacher_topk_probs"].shape == (2, 6, 4)
    assert batch["teacher_topk_indices"].dtype == torch.int32
    assert batch["teacher_tail_prob"].dtype == torch.float32
    # Non-assistant filler is a valid one-bucket distribution.
    assert float(batch["teacher_tail_prob"][0, 0]) == 1.0
    assert float(batch["teacher_tail_prob"][0, 1]) < 1.0


def test_manifest_mismatch_is_refused(tmp_path):
    _write_index(tmp_path, [_row(0, [-100, 3])])
    with pytest.raises(TeacherManifestMismatch, match="different data manifest"):
        TeacherIndex.from_directory(tmp_path, data_manifest_sha256="e" * 64)
    with pytest.raises(TeacherManifestMismatch, match="top_k"):
        TeacherIndex.from_directory(
            tmp_path, data_manifest_sha256="d" * 64, expect_top_k=8
        )


def test_collator_joins_rows_and_aligns_the_teacher_with_labels():
    rows = [_row(0, [-100, 7, -100, 9]), _row(1, [-100, 4, 5, -100])]
    index = _FakeIndex(rows, top_k=4)

    def base_collator(features):
        length = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.zeros((len(features), length), dtype=torch.long)
        labels = torch.full((len(features), length), -100, dtype=torch.long)
        for number, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"])
            input_ids[number, : len(ids)] = ids
            labels[number, : len(feature["labels"])] = torch.tensor(feature["labels"])
        return {"input_ids": input_ids, "labels": labels}

    collator = TeacherTopKCollator(index, base_collator)
    batch = collator(
        [
            {"row_id": 1, "input_ids": [1, 2, 3, 4], "labels": [-100, 4, 5, -100]},
            {"row_id": 0, "input_ids": [1, 2, 3, 4], "labels": [-100, 7, -100, 9]},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert "row_id" not in batch
    # Row 1 is first in the batch; its teacher token at position 2 is stored.
    assert torch.equal(batch["teacher_topk_indices"][0, 2], rows[1].indices[1])
    assert torch.equal(batch["teacher_topk_probs"][0, 2], rows[1].probs[1])
    assert float(batch["teacher_tail_prob"][0, 2]) == pytest.approx(
        float(rows[1].tail[1])
    )
    # A position with no supervised token is filler.
    assert float(batch["teacher_tail_prob"][0, 3]) == 1.0


class _FakeIndex:
    """Small stand-in for ``TeacherIndex`` for collator-only tests."""

    def __init__(self, rows, *, top_k):
        self.rows = {row.row_id: row for row in rows}
        self.manifest = SimpleNamespace(
            top_k=top_k,
            teacher={"key": "e4b", "revision": "rev"},
        )
        self.manifest_sha256 = "fake-manifest-sha"

    def row(self, row_id):
        return self.rows[int(row_id)]

    def batch(self, row_ids, seq_len):
        return TeacherIndex.batch(self, row_ids, seq_len)


def test_collator_teacher_is_shifted_onto_the_prediction_positions():
    from gemmaloop.train.dial_objective import TeacherTopK, _teacher_for_sequence

    rows = [_row(0, [-100, 7, 9, -100])]
    index = _FakeIndex(rows, top_k=4)

    def base_collator(features):
        return {
            "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
            "labels": torch.tensor([feature["labels"] for feature in features]),
        }

    batch = TeacherTopKCollator(index, base_collator)(
        [{"row_id": 0, "input_ids": [1, 2, 3, 4], "labels": [-100, 7, 9, -100]}]
    )
    teacher = TeacherTopK(
        probs=batch["teacher_topk_probs"],
        indices=batch["teacher_topk_indices"],
        tail_prob=batch["teacher_tail_prob"],
    )
    shifted_labels, shifted = _teacher_for_sequence(
        batch["labels"], teacher, ignore_index=-100
    )
    assert shifted_labels.tolist() == [[7, 9, -100]]
    # Slots 0 and 1 now score target tokens at original positions 1 and 2.
    assert torch.equal(shifted.indices[0], rows[0].indices[0])
    assert torch.equal(shifted.indices[1], rows[0].indices[1])
    assert float(shifted.tail_prob[2]) == 1.0


def test_validate_alignment_accepts_and_rejects():
    rows = [_row(0, [-100, 3, -100, 5])]
    index = _FakeIndex(rows, top_k=4)
    validate_teacher_alignment(
        index, [{"row_id": 0, "labels": [-100, 3, -100, 5]}]
    )
    with pytest.raises(TeacherManifestMismatch, match="does not align|positions"):
        validate_teacher_alignment(
            index, [{"row_id": 0, "labels": [-100, 3, 4, 5]}]
        )


class _VocabTokenizer:
    def __init__(self, vocab):
        self._vocab = dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)


def test_vocabulary_check_accepts_identical_tokenizers_except_the_mask():
    student = _VocabTokenizer({"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 2, "c": 3})
    teacher = _VocabTokenizer({"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 2, "c": 3})
    check = assert_teacher_tokenizer_matches(student, teacher)
    assert check == {"mask_token_id": 2, "vocab_size": 4}


def test_vocabulary_check_refuses_any_other_difference():
    student = _VocabTokenizer({"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 2, "c": 3})
    teacher = _VocabTokenizer({"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 2, "d": 3})
    with pytest.raises(ValueError, match="differs"):
        assert_teacher_tokenizer_matches(student, teacher)
    teacher = _VocabTokenizer({"a": 0, "b": 1, DEFAULT_MASK_TOKEN: 2, "c": 3, "x": 4})
    with pytest.raises(ValueError, match="size"):
        assert_teacher_tokenizer_matches(student, teacher)


def test_cached_e2b_and_e4b_tokenizers_match_except_the_mask_if_available():
    transformers = pytest.importorskip("transformers")
    from gemmaloop.model.snapshot import E2B_REVISION, E4B_REVISION

    try:
        student = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it",
            revision=E2B_REVISION,
            local_files_only=True,
        )
        teacher = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-4-E4B-it",
            revision=E4B_REVISION,
            local_files_only=True,
        )
    except (OSError, ValueError):
        pytest.skip("E2B/E4B tokenizer files are not in the local cache")
    check = assert_teacher_tokenizer_matches(student, teacher)
    assert check["vocab_size"] == len(student.get_vocab())


# ---------------------------------------------------------------------------
# Precompute script: shard resume
# ---------------------------------------------------------------------------


def _load_precompute_script():
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "precompute_teacher_topk.py"
    spec = importlib.util.spec_from_file_location("precompute_teacher_topk_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeVocabTokenizer:
    def get_vocab(self):
        return {f"tok{i}": i for i in range(32)}


class _FakeTextModel:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.weight[input_ids])


def test_precompute_resumes_per_shard_and_recomputes_corrupt_shards(tmp_path):
    module = _load_precompute_script()
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(32, 8, generator=generator)
    loaded = SimpleNamespace(
        spec=teacher_spec("e4b"),
        model=None,
        tokenizer=_FakeVocabTokenizer(),
        text_model=_FakeTextModel(weight),
        weight=weight,
        softcap=None,
        hidden=lambda input_ids: weight[input_ids],
    )
    prepared = [
        module.PreparedRow(row_id=0, input_ids=[1, 2, 3, 4], labels=[-100, 2, 3, -100]),
        module.PreparedRow(row_id=1, input_ids=[2, 3, 4, 5], labels=[-100, 3, -100, 5]),
    ]
    kwargs = dict(
        out=tmp_path / "teacher",
        data_manifest_sha256="a" * 64,
        top_k=4,
        max_length=8,
        shard_rows=1,
        token_chunk=4,
        vocab_chunk=7,
        mask_token_id=None,
        expected_lm_head_rows=None,
    )
    first = module.precompute_prepared(prepared, loaded, **kwargs)
    assert first["computed"] == 2 and first["skipped"] == 0
    assert first["tokens"] == 4
    manifest = TeacherManifest.read(tmp_path / "teacher" / "teacher_manifest.json")
    assert manifest.rows_total == 2 and manifest.tokens_total == 4

    second = module.precompute_prepared(prepared, loaded, **kwargs)
    assert second["computed"] == 0 and second["skipped"] == 2

    # A corrupt shard is not skipped: its SHA no longer matches the manifest.
    (tmp_path / "teacher" / "shard-00001.pt").write_bytes(b"torn")
    third = module.precompute_prepared(prepared, loaded, **kwargs)
    assert third["computed"] == 1 and third["skipped"] == 1
    repaired = TeacherManifest.read(tmp_path / "teacher" / "teacher_manifest.json")
    assert all(
        file_sha256(tmp_path / "teacher" / entry.file) == entry.sha256
        for entry in repaired.shards
    )
    # The stale manifest entry cannot hide a corrupt file from the reader.
    assert "a" * 64 == json.loads(
        (tmp_path / "teacher" / "teacher_manifest.json").read_text()
    )["data_manifest_sha256"]


def test_tail_is_fp32_before_the_float16_rounding():
    hidden = torch.zeros(1, 4)
    hidden[0, 0] = 1.0
    weight = torch.zeros(5, 4)
    weight[0] = hidden[0] * 20.0
    indices, probs, tail = topk_distribution_from_hidden(
        hidden, weight, top_k=2, mask_token_id=None, token_chunk=1, vocab_chunk=2
    )
    _, ref_probs, ref_tail = _reference_topk(hidden, weight, 2, None)
    torch.testing.assert_close(tail, ref_tail, atol=1e-7, rtol=1e-6)
    # A confident token's true tail is ~2e-9; the fp16 sum would turn it into
    # rounding noise (~1e-4).  The stored tail must stay below 1e-6.
    assert float(tail[0]) < 1e-6


def test_precompute_crash_resumes_from_the_partial_manifest(tmp_path):
    module = _load_precompute_script()
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(32, 8, generator=generator)
    prepared = [
        module.PreparedRow(
            row_id=i,
            input_ids=[2 + i, 3, 4, 5],
            labels=[-100, 3, 4, -100],
        )
        for i in range(3)
    ]

    def loaded_with(hidden_fn):
        return SimpleNamespace(
            spec=teacher_spec("e4b"),
            model=None,
            tokenizer=_FakeVocabTokenizer(),
            text_model=None,
            weight=weight,
            softcap=None,
            hidden=hidden_fn,
        )

    calls = {"n": 0}

    def crashing_hidden(input_ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated driver reset")
        return weight[input_ids]

    kwargs = dict(
        out=tmp_path / "teacher",
        data_manifest_sha256="b" * 64,
        top_k=4,
        max_length=8,
        shard_rows=1,
        token_chunk=4,
        vocab_chunk=7,
        mask_token_id=None,
        expected_lm_head_rows=None,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        module.precompute_prepared(
            prepared, loaded_with(crashing_hidden), **kwargs
        )

    partial = TeacherManifest.read(tmp_path / "teacher" / "teacher_manifest.json")
    assert partial.complete is False
    assert [entry.index for entry in partial.shards] == [0]
    with pytest.raises(TeacherManifestMismatch, match="incomplete"):
        TeacherIndex.from_directory(
            tmp_path / "teacher", data_manifest_sha256="b" * 64
        )

    result = module.precompute_prepared(
        prepared, loaded_with(lambda ids: weight[ids]), **kwargs
    )
    assert result["skipped"] == 1 and result["computed"] == 2
    final = TeacherManifest.read(tmp_path / "teacher" / "teacher_manifest.json")
    assert final.complete is True
    assert [entry.index for entry in final.shards] == [0, 1, 2]


def test_alignment_checks_the_rendered_input_id_hash():
    from gemmaloop.data.teacher_topk import input_ids_sha256

    row = _row(0, [-100, 3, -100, 5])
    row = TeacherRow(
        row_id=row.row_id,
        positions=row.positions,
        indices=row.indices,
        probs=row.probs,
        tail=row.tail,
        input_ids_sha256=input_ids_sha256([1, 2, 3, 4]),
    )
    index = _FakeIndex([row], top_k=4)
    validate_teacher_alignment(
        index,
        [{"row_id": 0, "input_ids": [1, 2, 3, 4], "labels": [-100, 3, -100, 5]}],
    )
    with pytest.raises(TeacherManifestMismatch, match="different input ids"):
        validate_teacher_alignment(
            index,
            [{"row_id": 0, "input_ids": [9, 9, 9, 9], "labels": [-100, 3, -100, 5]}],
        )


def test_index_hashes_shards_once_and_keeps_an_lru(tmp_path, monkeypatch):
    import gemmaloop.data.teacher_topk as module

    rows = [_row(i, [-100, 3 + i, 4]) for i in range(3)]
    shards = []
    for index, row in enumerate(rows):
        path = tmp_path / f"shard-{index:05d}.pt"
        digest = write_teacher_shard(path, [row])
        shards.append(
            ShardEntry(
                index=index,
                file=path.name,
                sha256=digest,
                row_ids=(row.row_id,),
                tokens=row.tokens,
            )
        )
    TeacherManifest(
        teacher={"key": "e4b", "revision": "rev", "runtime": "bf16", "max_length": 8},
        data_manifest_sha256="d" * 64,
        top_k=4,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=GLITCH_TOKEN_ID,
        rows_total=3,
        tokens_total=sum(row.tokens for row in rows),
        shards=tuple(shards),
        chat_template_sha256="template",
        tokenizer_sha256="tokenizer",
        lm_head_rows=262144,
        runtime_identity=runtime_identity("e4b", "bf16"),
    ).write(tmp_path / "teacher_manifest.json")

    calls = {"n": 0}
    original = module._sha256_file

    def counting(path):
        calls["n"] += 1
        return original(path)

    monkeypatch.setattr(module, "_sha256_file", counting)
    index = TeacherIndex.from_directory(
        tmp_path, data_manifest_sha256="d" * 64, verify_shards=True
    )
    after_startup = calls["n"]
    # three shard files + the manifest file, all hashed once at startup
    assert after_startup == 4
    for row in rows:
        index.row(row.row_id)
    assert calls["n"] == after_startup
    index.max_cached_shards = 1
    for row in rows:
        index.row(row.row_id)
    assert len(index._shard_cache) == 1


# ---------------------------------------------------------------------------
# X3-C1: no autograd graph from the teacher head or the production precompute
# ---------------------------------------------------------------------------


def test_head_projection_saves_no_autograd_tensors():
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(7, 5, generator=generator)
    weight = torch.nn.Parameter(torch.randn(29, 5, generator=generator))
    saved: list[torch.Tensor] = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.enable_grad():
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            indices, probs, tail = topk_distribution_from_hidden(
                hidden,
                weight,
                top_k=4,
                mask_token_id=3,
                token_chunk=2,
                vocab_chunk=7,
            )

    assert saved == []
    assert weight.requires_grad is True
    for tensor in (indices, probs, tail):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_teacher_row_detaches_its_stored_tensors():
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(6, 5, generator=generator)
    weight = torch.nn.Parameter(torch.randn(23, 5, generator=generator))
    labels = torch.tensor([-100, 3, -100, 7, 8, -100])
    with torch.enable_grad():
        row = compute_teacher_row(
            hidden,
            weight,
            labels,
            row_id=9,
            top_k=4,
            mask_token_id=5,
            token_chunk=2,
            vocab_chunk=6,
        )
    for tensor in (row.positions, row.indices, row.probs, row.tail):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_production_precompute_runs_in_inference_mode_and_detaches(tmp_path):
    module = _load_precompute_script()
    generator = torch.Generator().manual_seed(13)
    # A head that still requires grad: the production pass must ignore it.
    weight = torch.nn.Parameter(torch.randn(32, 8, generator=generator))
    loaded = SimpleNamespace(
        spec=teacher_spec("e4b"),
        model=None,
        tokenizer=_FakeVocabTokenizer(),
        text_model=None,
        weight=weight,
        softcap=None,
        hidden=lambda input_ids: weight[input_ids].detach(),
    )
    prepared = [
        module.PreparedRow(row_id=0, input_ids=[1, 2, 3, 4], labels=[-100, 2, 3, -100])
    ]
    module.precompute_prepared(
        prepared,
        loaded,
        out=tmp_path / "teacher",
        data_manifest_sha256="c" * 64,
        top_k=4,
        max_length=8,
        shard_rows=1,
        token_chunk=2,
        vocab_chunk=7,
        mask_token_id=None,
        expected_lm_head_rows=None,
    )
    rows = read_teacher_shard(tmp_path / "teacher" / "shard-00000.pt")
    assert len(rows) == 1
    for tensor in (rows[0].positions, rows[0].indices, rows[0].probs, rows[0].tail):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
    # The freeze helper is an explicit invariant, not just an inference-scope
    # side effect.
    assert weight.requires_grad is False


# ---------------------------------------------------------------------------
# Registered teacher runtimes and manifest identity
# ---------------------------------------------------------------------------


def test_registered_runtimes_record_the_review_semantics():
    from gemmaloop.teacher.runtime import (
        DEFAULT_VOCAB_CHUNK,
        teacher_runtime,
    )

    identity = teacher_runtime("e4b", "bf16").identity()
    assert identity["method"] == "bf16"
    assert identity["compute_dtype"] == "bfloat16"
    assert "lm_head" in identity["skipped_modules"]
    assert identity["head_projection"] == "fp32-vocab-chunked"
    assert identity["head_accumulation"] == "fp32"
    assert identity["attention_implementation"] == "sdpa"
    assert identity["vocab_chunk"] == DEFAULT_VOCAB_CHUNK
    assert {"torch", "transformers", "bitsandbytes"} <= set(identity["software"])

    int8 = teacher_runtime("e4b", "int8").identity()
    assert int8["method"] == "bitsandbytes-int8"
    assert int8["settings"]["llm_int8_threshold"] == 6.0

    nf4 = teacher_runtime("12b_qat").identity()
    assert nf4["method"] == "bitsandbytes-nf4"
    assert nf4["settings"]["bnb_4bit_quant_type"] == "nf4"


def test_unknown_or_unimplemented_runtimes_fail_closed():
    from gemmaloop.teacher.runtime import teacher_runtime

    with pytest.raises(ValueError, match="no registered runtime"):
        teacher_runtime("e4b", "nf4")
    with pytest.raises(NotImplementedError, match="q4_0_gguf"):
        teacher_runtime("12b_qat", "q4_0_gguf")


def _runtime_manifest(key: str, runtime: str) -> TeacherManifest:
    from gemmaloop.model.snapshot import teacher_revision
    from gemmaloop.teacher.runtime import runtime_identity

    return TeacherManifest(
        teacher={
            "key": key,
            "revision": teacher_revision(key),
            "runtime": runtime,
        },
        data_manifest_sha256="d" * 64,
        top_k=64,
        temperature=1.0,
        mask_token=DEFAULT_MASK_TOKEN,
        mask_token_id=GLITCH_TOKEN_ID,
        rows_total=0,
        tokens_total=0,
        render_template_sha256="template",
        student_vocab_sha256="student",
        teacher_vocab_sha256="teacher",
        teacher_vocab_ex_mask_sha256="student",
        lm_head_rows=262144,
        runtime_identity=runtime_identity(key, runtime),
    )


def test_manifest_identity_binds_each_teacher_to_its_registered_runtime():
    from dataclasses import replace

    from gemmaloop.data.teacher_topk import validate_teacher_manifest_identity
    from gemmaloop.teacher.runtime import runtime_identity

    validate_teacher_manifest_identity(_runtime_manifest("e4b", "bf16"))
    validate_teacher_manifest_identity(_runtime_manifest("e4b", "int8"))
    validate_teacher_manifest_identity(_runtime_manifest("12b_qat", "nf4"))

    e4b = _runtime_manifest("e4b", "bf16")
    with pytest.raises(TeacherManifestMismatch, match="not registered"):
        validate_teacher_manifest_identity(
            replace(e4b, teacher={**e4b.teacher, "runtime": "nf4"})
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("method", "torchao-int4"),
        ("compute_dtype", "float16"),
        ("head_projection", "bf16-vocab-chunked"),
        ("head_accumulation", "bf16"),
        ("attention_implementation", "eager"),
        ("skipped_modules", []),
        ("token_chunk", 0),
        ("vocab_chunk", 0),
        ("software", {"torch": "0.0", "transformers": "5.22.1"}),
    ],
)
def test_manifest_identity_refuses_changed_runtime_semantics(field, value):
    from dataclasses import replace

    from gemmaloop.data.teacher_topk import validate_teacher_manifest_identity

    manifest = _runtime_manifest("12b_qat", "nf4")
    identity = dict(manifest.runtime_identity)
    identity[field] = value
    with pytest.raises(TeacherManifestMismatch):
        validate_teacher_manifest_identity(
            replace(manifest, runtime_identity=identity)
        )


def test_manifest_without_runtime_identity_is_refused():
    from dataclasses import replace

    from gemmaloop.data.teacher_topk import validate_teacher_manifest_identity

    manifest = _runtime_manifest("e4b", "bf16")
    with pytest.raises(TeacherManifestMismatch, match="runtime identity"):
        validate_teacher_manifest_identity(
            replace(manifest, runtime_identity={})
        )


def test_precompute_resume_refuses_changed_runtime_or_chunks(tmp_path):
    module = _load_precompute_script()
    generator = torch.Generator().manual_seed(21)
    weight = torch.randn(32, 8, generator=generator)
    prepared = [
        module.PreparedRow(row_id=0, input_ids=[1, 2, 3, 4], labels=[-100, 2, 3, -100])
    ]

    def loaded_for(spec):
        return SimpleNamespace(
            spec=spec,
            model=None,
            tokenizer=_FakeVocabTokenizer(),
            text_model=None,
            weight=weight,
            softcap=None,
            hidden=lambda input_ids: weight[input_ids],
        )

    kwargs = dict(
        out=tmp_path / "teacher",
        data_manifest_sha256="e" * 64,
        top_k=4,
        max_length=8,
        shard_rows=1,
        mask_token_id=None,
        expected_lm_head_rows=None,
    )
    first = module.precompute_prepared(
        prepared,
        loaded_for(teacher_spec("e4b")),
        token_chunk=4,
        vocab_chunk=7,
        **kwargs,
    )
    assert first["computed"] == 1
    # Same semantics: resume skips.
    again = module.precompute_prepared(
        prepared,
        loaded_for(teacher_spec("e4b")),
        token_chunk=4,
        vocab_chunk=7,
        **kwargs,
    )
    assert again["skipped"] == 1
    # Different registered runtime: never reuse the old shards.
    changed_runtime = module.precompute_prepared(
        prepared,
        loaded_for(teacher_spec("e4b", runtime="int8")),
        token_chunk=4,
        vocab_chunk=7,
        **kwargs,
    )
    assert changed_runtime["computed"] == 1 and changed_runtime["skipped"] == 0
    # Same runtime, different chunk sizes: still different semantics.
    changed_chunks = module.precompute_prepared(
        prepared,
        loaded_for(teacher_spec("e4b", runtime="int8")),
        token_chunk=2,
        vocab_chunk=3,
        **kwargs,
    )
    assert changed_chunks["computed"] == 1 and changed_chunks["skipped"] == 0
    manifest = TeacherManifest.read(tmp_path / "teacher" / "teacher_manifest.json")
    assert manifest.runtime_identity["token_chunk"] == 2
    assert manifest.teacher["runtime"] == "int8"


def test_loaders_freeze_all_parameters_and_offload_vision():
    module = _load_precompute_script()
    from torch import nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Linear(2, 2)
            self.embed_vision = nn.Linear(2, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self.model = Inner()

    model = Model()
    model.param.requires_grad_(True)
    module._freeze_all(model)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    module._offload_unused_vision(model)
    assert model.model.vision_tower.weight.device.type == "cpu"
    assert model.model.embed_vision.weight.device.type == "cpu"


def test_compute_teacher_row_refuses_a_nonfinite_hidden_state():
    weight = torch.randn(8, 4)
    hidden = torch.randn(3, 4)
    hidden[1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite hidden"):
        compute_teacher_row(
            hidden,
            weight,
            torch.tensor([-100, 1, 2]),
            row_id=7,
            top_k=2,
            mask_token_id=None,
        )


def test_write_teacher_shard_refuses_nonfinite_rows(tmp_path):
    row = TeacherRow(
        row_id=0,
        positions=torch.tensor([1], dtype=torch.int32),
        indices=torch.tensor([[0, 1]], dtype=torch.int32),
        probs=torch.tensor([[float("nan"), 0.5]], dtype=torch.float16),
        tail=torch.tensor([0.5], dtype=torch.float32),
    )
    path = tmp_path / "shard-00000.pt"
    with pytest.raises(ValueError, match="non-finite probs"):
        write_teacher_shard(path, [row])
    assert not path.exists()


def test_manifest_identity_refuses_a_runtime_other_than_the_frozen_one():
    from gemmaloop.data.teacher_topk import validate_teacher_manifest_identity

    validate_teacher_manifest_identity(
        _runtime_manifest("e4b", "int8"), frozen_runtime="int8"
    )
    with pytest.raises(TeacherManifestMismatch, match="frozen qualified runtime"):
        validate_teacher_manifest_identity(
            _runtime_manifest("e4b", "bf16"), frozen_runtime="int8"
        )


def test_public_api_keeps_write_json_atomic():
    import gemmaloop.data.teacher_topk as teacher_topk

    assert "write_json_atomic" in teacher_topk.__all__
    assert "NonFiniteTeacherRow" in teacher_topk.__all__


def test_precompute_maps_a_non_finite_teacher_row_to_exit_2(
    tmp_path, monkeypatch, capsys
):
    module = _load_precompute_script()
    monkeypatch.setattr(
        module,
        "read_data_manifest",
        lambda path: ({"max_length": 16}, "a" * 64),
    )
    monkeypatch.setattr(module, "load_rows", lambda path: [])
    monkeypatch.setattr(module, "cap_cuda_allocator", lambda *a, **k: None)
    monkeypatch.setattr(module, "load_student_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(module, "prepare_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        module,
        "load_teacher",
        lambda *a, **k: SimpleNamespace(spec=teacher_spec("e4b"), tokenizer=None),
    )

    def boom(*args, **kwargs):
        raise module.NonFiniteTeacherRow(
            "teacher row 7 has non-finite probs; refusing to write it"
        )

    monkeypatch.setattr(module, "precompute_prepared", boom)
    code = module.main(
        [
            "--teacher",
            "e4b",
            "--runtime",
            "int8",
            "--data",
            str(tmp_path / "heal"),
            "--out",
            str(tmp_path / "teacher"),
            "--skip-gpu-lock",
            "--skip-tokenizer-check",
        ]
    )
    assert code == 2
    assert "non-finite" in capsys.readouterr().err
