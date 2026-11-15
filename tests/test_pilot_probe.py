"""CPU smoke test for the pilot probe script."""

import json

import pytest

import scripts.pilot_probe as pilot


def _rows(path, n=4, length=20):
    with path.open("w", encoding="utf-8") as handle:
        for row in range(n):
            ids = [((row + 1) * 17 + i * 13) % 256 for i in range(length)]
            handle.write(json.dumps({"input_ids": ids, "labels_start": 5}) + "\n")


def test_tiny_cpu_probe_writes_json_and_markdown(tmp_path):
    rows = tmp_path / "rows.jsonl"
    _rows(rows)
    out_json = tmp_path / "probe.json"
    out_md = tmp_path / "probe.md"
    code = pilot.main(
        [
            "--tiny",
            "--rows",
            str(rows),
            "--blocks",
            "5-7,6-10",
            "--bridges",
            "none,gch",
            "--n-loops",
            "2",
            "--max-rows",
            "4",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["rows"] == 4
    assert len(payload["layer_scalar"]) == 15
    assert len(payload["layers"]) == 15
    variants = {
        (tuple(item["block"]), item["bridge"]): item
        for item in payload["variants"]
    }
    assert set(variants) == {
        ((5, 7), "none"),
        ((5, 7), "gch"),
        ((6, 10), "none"),
        ((6, 10), "gch"),
    }
    # Zero-init gch is exactly N=1 at N=2.
    for block in ((5, 7), (6, 10)):
        assert variants[(block, "gch")]["kl_n1_n2"] == pytest.approx(0.0, abs=1e-8)
        assert variants[(block, "gch")]["delta_nll_calibrated"] == pytest.approx(
            0.0, abs=1e-6
        )
    assert "layer_scalar" in out_md.read_text(encoding="utf-8")


def test_tiny_probe_reports_every_n_with_raw_nll_and_top1(tmp_path):
    rows = tmp_path / "rows.jsonl"
    _rows(rows)
    out_json = tmp_path / "probe.json"
    code = pilot.main(
        [
            "--tiny",
            "--rows",
            str(rows),
            "--blocks",
            "5-7",
            "--bridges",
            "none,damped",
            "--n-loops",
            "1,2,3",
            "--max-rows",
            "4",
            "--out-json",
            str(out_json),
            "--out-md",
            str(tmp_path / "probe.md"),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_loop_list"] == [1, 2, 3]
    variants = payload["variants"]
    assert len(variants) == 6
    assert {item["n_loops"] for item in variants} == {1, 2, 3}
    for item in variants:
        assert "raw_nll_n" in item and "raw_nll_n1" in item
        assert 0.0 <= item["top1_n"] <= 1.0
        if item["n_loops"] == 1:
            assert item["kl_n1_n"] == 0.0
            assert item["nll_n_calibrated"] == pytest.approx(
                item["nll_n1_calibrated"]
            )



def test_chunked_kl_handles_bf16_lm_head_weight_on_cpu():
    import torch

    from tests.model.tiny import tiny_text_model

    model = tiny_text_model().to(torch.bfloat16)
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        hidden = model(input_ids=ids, use_cache=False).last_hidden_state
    weight = model.embed_tokens.weight
    assert weight.dtype == torch.bfloat16
    value = pilot._kl_from_hidden(hidden, hidden, weight, None)
    assert value == pytest.approx(0.0, abs=1e-6)
    value = pilot._kl_from_hidden(
        hidden, hidden.mul(0.5), weight, None, vocab_chunk=7
    )
    assert value > 0



def test_kl_chunk_memory_does_not_scale_with_sequence_length():
    import torch
    from torch.profiler import ProfilerActivity, profile

    tokens, vocab, hidden = 8192, 2048, 64
    generator = torch.Generator().manual_seed(0)
    one = torch.randn(1, tokens, hidden, generator=generator)
    two = torch.randn(1, tokens, hidden, generator=generator)
    weight = torch.randn(vocab, hidden, generator=generator)

    def max_matmul_bytes(length: int) -> int:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            pilot._kl_from_hidden(
                one[:, :length],
                two[:, :length],
                weight,
                None,
                token_chunk=256,
                vocab_chunk=512,
            )
        values = [
            event.cpu_memory_usage
            for event in prof.events()
            if event.key in {"aten::mm", "aten::matmul"}
            and event.cpu_memory_usage > 0
        ]
        assert values
        return max(values)

    short = max_matmul_bytes(512)
    long = max_matmul_bytes(tokens)
    # A matmul tile is at most token_chunk x vocab_chunk; the full [T, V]
    # tensor would be tokens * vocab * 4 bytes.
    assert short <= 256 * 512 * 4
    assert long <= 256 * 512 * 4
    assert long < tokens * vocab * 4 // 4


def test_tiny_probe_records_allocator_cap(tmp_path, monkeypatch):
    cap = {"total_gib": 12.0, "free_at_start_gib": 11.0, "cap_gib": 9.0}
    monkeypatch.setattr(pilot, "cap_cuda_allocator", lambda *a, **k: cap)
    rows = tmp_path / "rows.jsonl"
    _rows(rows)
    out_json = tmp_path / "probe.json"
    assert (
        pilot.main(
            [
                "--tiny",
                "--rows",
                str(rows),
                "--blocks",
                "5-7",
                "--bridges",
                "none",
                "--n-loops",
                "1",
                "--max-rows",
                "2",
                "--out-json",
                str(out_json),
                "--out-md",
                str(tmp_path / "probe.md"),
            ]
        )
        == 0
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["vram_cap"] == cap
    assert payload["corpus_sha256"]
