"""CPU tests for the non-mmap :class:`SafetensorsWeightSource`.

The real 12B checkpoint is one ~24 GB file.  ``safe_open(...).get_tensor`` per
tensor keeps a copy-on-write view of the whole file alive, so these tests use
tiny synthetic checkpoints and prove the replacement copies only each tensor's
bytes into ordinary, writable process memory.
"""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from gemmaloop.teacher.reference import SafetensorsWeightSource


def _mixed_state(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "embed_tokens.weight": torch.randn(
            6, 4, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16),
        "norm.weight": torch.randn(4, generator=generator, dtype=torch.float32),
        "half.weight": torch.randn(3, 2, generator=generator, dtype=torch.float16),
        "double.weight": torch.randn(2, 3, generator=generator, dtype=torch.float64),
        "long.weight": torch.tensor(
            [[-1, 0, 1], [2**40, -(2**40), 7]], dtype=torch.int64
        ),
        "int.weight": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int32),
        "short.weight": torch.tensor([[-3, 0, 5]], dtype=torch.int16),
        "char.weight": torch.tensor([[-128, 0, 127]], dtype=torch.int8),
        "byte.weight": torch.tensor([[0, 128, 255]], dtype=torch.uint8),
        "bool.weight": torch.tensor([[True, False, True, False]], dtype=torch.bool),
        "empty.weight": torch.empty(0, 3, dtype=torch.float32),
    }


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _assert_bit_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(_bits(actual), _bits(expected))


def _write_checkpoint(path, *, data_offsets, shape=(4,), data_length=16):
    header = json.dumps(
        {
            "x": {
                "dtype": "F32",
                "shape": list(shape),
                "data_offsets": list(data_offsets),
            }
        }
    ).encode("utf-8")
    payload = struct.pack("<Q", len(header)) + header + b"\x00" * data_length
    path.write_bytes(payload)


def test_single_file_mixed_dtypes_are_bit_identical_to_load_file(
    tmp_path, monkeypatch
):
    state = _mixed_state(seed=1)
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, checkpoint)
    expected = load_file(checkpoint)

    source = SafetensorsWeightSource(tmp_path)
    assert source.prefix == ""
    for key, want in expected.items():
        _assert_bit_identical(source.tensor(key), want)

    # Regression: the old implementation constant-folded every tensor through
    # safe_open, which maps the whole file on this Windows/torch pair.
    import safetensors

    def fail(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("safe_open must not be used by the weight source")

    monkeypatch.setattr(safetensors, "safe_open", fail)
    _assert_bit_identical(
        source.tensor("embed_tokens.weight"),
        expected["embed_tokens.weight"],
    )


def test_sharded_index_checkpoint_is_bit_identical_to_load_file(tmp_path):
    state = _mixed_state(seed=2)
    names = list(state)
    first, second = names[: len(names) // 2], names[len(names) // 2 :]
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({name: state[name] for name in first}, shard_a)
    save_file({name: state[name] for name in second}, shard_b)
    weight_map = {
        **{name: shard_a.name for name in first},
        **{name: shard_b.name for name in second},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    source = SafetensorsWeightSource(tmp_path)
    expected = {**load_file(shard_a), **load_file(shard_b)}
    for key, want in expected.items():
        _assert_bit_identical(source.tensor(key), want)


def test_returned_tensor_is_writable_owned_memory_not_a_file_mapping(tmp_path):
    state = _mixed_state(seed=3)
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, checkpoint)
    original_bytes = checkpoint.read_bytes()

    source = SafetensorsWeightSource(tmp_path)
    first = source.tensor("embed_tokens.weight")
    second = source.tensor("embed_tokens.weight")
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert first.untyped_storage().resizable() is True
    assert (
        first.untyped_storage().nbytes()
        == first.numel() * first.element_size()
    )

    # Mutating one read must not touch the other read or the checkpoint.
    first.zero_()
    assert checkpoint.read_bytes() == original_bytes
    _assert_bit_identical(second, state["embed_tokens.weight"])
    _assert_bit_identical(
        source.tensor("embed_tokens.weight"), state["embed_tokens.weight"]
    )
    _assert_bit_identical(source.tensor("embed_tokens.weight"), load_file(checkpoint)[
        "embed_tokens.weight"
    ])


def test_truncated_file_raises(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file(_mixed_state(seed=4), checkpoint)
    data = checkpoint.read_bytes()
    checkpoint.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="beyond the .* data section"):
        SafetensorsWeightSource(tmp_path)


def test_trailing_bytes_raise(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file(_mixed_state(seed=4), checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="does not match"):
        SafetensorsWeightSource(tmp_path)


def test_truncated_header_raises(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(ValueError, match="truncated safetensors header"):
        SafetensorsWeightSource(tmp_path)


def test_out_of_bounds_offsets_raise(tmp_path):
    _write_checkpoint(tmp_path / "model.safetensors", data_offsets=(16, 32))
    with pytest.raises(ValueError, match="beyond the .* data section"):
        SafetensorsWeightSource(tmp_path)


def test_offsets_that_do_not_match_dtype_and_shape_raise(tmp_path):
    _write_checkpoint(tmp_path / "model.safetensors", data_offsets=(0, 999))
    with pytest.raises(ValueError, match="data_offsets .* need"):
        SafetensorsWeightSource(tmp_path)


def test_non_contiguous_offsets_raise(tmp_path):
    _write_checkpoint(
        tmp_path / "model.safetensors", data_offsets=(4, 20), data_length=40
    )
    with pytest.raises(ValueError, match="not contiguous"):
        SafetensorsWeightSource(tmp_path)


def test_unknown_dtype_raises(tmp_path):
    header = json.dumps(
        {
            "x": {
                "dtype": "F8_E4M3",
                "shape": [4],
                "data_offsets": [0, 4],
            }
        }
    ).encode("utf-8")
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 4
    )
    with pytest.raises(ValueError, match="unsupported safetensors dtype"):
        SafetensorsWeightSource(tmp_path)


def test_missing_key_raises_key_error(tmp_path):
    save_file(_mixed_state(seed=5), tmp_path / "model.safetensors")
    source = SafetensorsWeightSource(tmp_path)
    with pytest.raises(KeyError):
        source.tensor("not_a_real_tensor")


def test_prefix_inference_is_unchanged(tmp_path):
    state = {
        "model.language_model.embed_tokens.weight": torch.randn(3, 2),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
        "model.language_model.layers.1.self_attn.q_proj.weight": torch.randn(2, 2),
    }
    save_file(state, tmp_path / "model.safetensors")
    source = SafetensorsWeightSource(tmp_path)
    assert source.prefix == "model.language_model."
    _assert_bit_identical(
        source.tensor("embed_tokens.weight"),
        state["model.language_model.embed_tokens.weight"],
    )
    _assert_bit_identical(
        source.tensor("layers.0.self_attn.q_proj.weight"),
        state["model.language_model.layers.0.self_attn.q_proj.weight"],
    )
    assert SafetensorsWeightSource(
        tmp_path, prefix="model.language_model."
    ).prefix == "model.language_model."


def test_prefix_falls_back_to_embed_tokens_when_no_layer_zero(tmp_path):
    state = {"model.embed_tokens.weight": torch.randn(2, 2)}
    save_file(state, tmp_path / "model.safetensors")
    source = SafetensorsWeightSource(tmp_path)
    assert source.prefix == "model."
