"""CPU test: the layer-streamed reference equals the normal tiny-model forward."""

from __future__ import annotations

import pytest
import torch

from gemmaloop.teacher.reference import InMemoryWeightSource, StreamingTextReference
from tests.model.tiny import tiny_text_model


def test_layer_streamed_reference_matches_the_tiny_model_forward():
    model = tiny_text_model(seed=4)
    input_ids = torch.tensor(
        [
            [
                2,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
            ]
        ],
        dtype=torch.long,
    )
    with torch.inference_mode():
        expected = model(input_ids, use_cache=False).last_hidden_state
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config,
        source,
        device="cpu",
        dtype=torch.float32,
    )
    actual = reference.forward(input_ids)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_layer_streamed_reference_loads_layers_one_at_a_time(monkeypatch):
    model = tiny_text_model(seed=5)
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config, source, device="cpu", dtype=torch.float32
    )
    requested: list[int] = []
    original = reference._load_layer

    def recording(index):
        requested.append(index)
        return original(index)

    monkeypatch.setattr(reference, "_load_layer", recording)
    input_ids = torch.tensor([[2, 5, 7, 11, 13, 17, 19]], dtype=torch.long)
    reference.forward(input_ids)
    assert requested == list(range(model.config.num_hidden_layers))


def test_safetensors_weight_source_round_trips_the_tiny_model(tmp_path):
    from safetensors.torch import save_file

    from gemmaloop.teacher.reference import SafetensorsWeightSource

    model = tiny_text_model(seed=6)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    source = SafetensorsWeightSource(tmp_path)
    assert source.prefix == ""
    assert torch.equal(
        source.tensor("embed_tokens.weight"), model.embed_tokens.weight.detach()
    )
    input_ids = torch.tensor([[2, 5, 7, 11, 13, 17]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids, use_cache=False).last_hidden_state
    reference = StreamingTextReference(
        model.config, source, device="cpu", dtype=torch.float32
    )
    torch.testing.assert_close(reference.forward(input_ids), expected, atol=2e-5, rtol=2e-4)
    assert reference.memory_report["per_layer"]
    assert reference.memory_report["within_ram_limit"] is True


def test_forward_many_loads_each_layer_once_and_matches_per_row_forward(monkeypatch):
    model = tiny_text_model(seed=7)
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config, source, device="cpu", dtype=torch.float32
    )
    inputs = [
        torch.tensor([[2, 5, 7, 11, 13]], dtype=torch.long),
        torch.tensor([[2, 17, 19, 23]], dtype=torch.long),
    ]
    with torch.inference_mode():
        expected = [model(ids, use_cache=False).last_hidden_state for ids in inputs]
    requested: list[int] = []
    original = reference._load_layer

    def recording(index):
        requested.append(index)
        return original(index)

    monkeypatch.setattr(reference, "_load_layer", recording)
    actual = reference.forward_many(inputs)
    assert requested == list(range(model.config.num_hidden_layers))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)


def test_streamed_reference_records_per_layer_peak_memory():
    model = tiny_text_model(seed=8)
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config,
        source,
        device="cpu",
        dtype=torch.float32,
        ram_limit_gib=None,
    )
    inputs = [
        torch.tensor([[2, 5, 7, 11]], dtype=torch.long),
        torch.tensor([[2, 17, 19]], dtype=torch.long),
    ]
    reference.forward_many(inputs)
    report = reference.memory_report
    assert {"baseline_ram_gib", "peak_ram_gib", "peak_ram_above_baseline_gib"} <= set(
        report
    )
    assert len(report["per_layer"]) == model.config.num_hidden_layers
    assert report["per_layer"][0]["layer"] == 0
    assert report["within_ram_limit"] is True


def test_streamed_reference_asserts_the_ram_bound(monkeypatch):
    import gemmaloop.teacher.reference as module

    model = tiny_text_model(seed=9)
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config,
        source,
        device="cpu",
        dtype=torch.float32,
        ram_limit_gib=1.0,
    )
    baseline = 2.0
    values = iter([baseline] + [baseline + 0.1] * 3 + [baseline + 5.0])
    monkeypatch.setattr(module, "_rss_gib", lambda: next(values, baseline + 5.0))
    with pytest.raises(RuntimeError, match="RAM bound"):
        reference.forward_many(
            [torch.tensor([[2, 5, 7, 11, 13]], dtype=torch.long)]
        )
    report = reference.memory_report
    assert report["within_ram_limit"] is False
    assert report["peak_ram_above_baseline_gib"] > 1.0


def _tiny_unified_model(seed: int = 0):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextModel,
    )

    config = Gemma4UnifiedTextConfig(
        num_hidden_layers=15,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 3,
        num_kv_shared_layers=10,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        vocab_size=256,
        sliding_window=16,
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    model = Gemma4UnifiedTextModel(config)
    model.eval()
    return model


def test_unified_reference_matches_the_unified_tiny_forward():
    from gemmaloop.teacher.reference import reference_family

    model = _tiny_unified_model(seed=10)
    assert reference_family(model.config).name == "gemma4_unified"
    input_ids = torch.tensor(
        [[2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        expected = model(input_ids, use_cache=False).last_hidden_state
    source = InMemoryWeightSource(model.state_dict(), dtype=torch.float32)
    reference = StreamingTextReference(
        model.config,
        source,
        device="cpu",
        dtype=torch.float32,
        ram_limit_gib=None,
    )
    assert reference.family.name == "gemma4_unified"
    actual = reference.forward(input_ids)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_streamed_reference_samples_ram_before_releasing_the_layer(
    tmp_path, monkeypatch
):
    from safetensors.torch import save_file

    import gemmaloop.teacher.reference as module
    from gemmaloop.teacher.reference import SafetensorsWeightSource

    model = tiny_text_model(seed=11)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    source = SafetensorsWeightSource(tmp_path)
    reference = StreamingTextReference(
        model.config,
        source,
        device="cpu",
        dtype=torch.float32,
        ram_limit_gib=None,
    )
    marker = {"live": False}
    original_load = reference._load_layer

    def load_with_injected_allocation(index):
        layer = original_load(index)
        injected = torch.zeros(1_000_000, dtype=torch.float32)
        layer._injected_allocation = injected
        marker["live"] = True
        return layer

    monkeypatch.setattr(reference, "_load_layer", load_with_injected_allocation)
    original_cleanup = module._empty_cache_or_gc

    def cleanup(device):
        marker["live"] = False
        return original_cleanup(device)

    monkeypatch.setattr(module, "_empty_cache_or_gc", cleanup)
    baseline = 2.0
    monkeypatch.setattr(
        module,
        "_rss_gib",
        lambda: baseline + (2.0 if marker["live"] else 0.0),
    )
    reference.forward_many(
        [torch.tensor([[2, 5, 7, 11, 13, 17]], dtype=torch.long)]
    )
    report = reference.memory_report
    assert report["per_layer"]
    # If the sample were taken after del layer / cleanup, the injected
    # allocation would already be gone and this would be 0.
    assert all(
        entry["ram_above_baseline_gib"] >= 2.0
        for entry in report["per_layer"]
    )
    assert report["peak_ram_above_baseline_gib"] >= 2.0
