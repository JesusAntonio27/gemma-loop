import gc
import json
import re
from pathlib import Path

import pytest
import torch
from unittest import mock

from gemmaloop.model.load import (
    IMAGE_TOKEN_IDS,
    ModelSpec,
    _apply_adapters,
    _resolve_loop_config,
    language_lora_target_modules,
    load_model,
)

IMAGE_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "images" / "two_shapes.png"
IMAGE_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_position_ids",
    "mm_token_type_ids",
)


def _image_inputs(processor, question="What colors are the shapes?"):
    """Run the chat template on an image + question and return model kwargs."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(IMAGE_FIXTURE)},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return inputs


def _model_kwargs(inputs, device):
    """Keep only the tensors ``Gemma4ForConditionalGeneration.forward`` needs."""
    return {
        key: inputs[key].to(device)
        for key in IMAGE_INPUT_KEYS
        if key in inputs
    }


def test_fails_fast_when_vram_low():
    with mock.patch("gemmaloop.model.load.free_vram_gib", return_value=1.0):
        with pytest.raises(RuntimeError, match="VRAM"):
            load_model(ModelSpec("e2b", "bf16"), min_free_gib=3.0)


@pytest.mark.gpu
@pytest.mark.slow
def test_e2b_loads_with_vision_and_ple_on_cpu():
    m, processor, plan = load_model(ModelSpec("e2b", "bf16"))
    ple = [p for n, p in m.named_parameters() if n.endswith("embed_tokens_per_layer.weight")][0]
    assert ple.device.type == "cpu" and not ple.requires_grad
    # The vision tower is kept now; only the audio tower is dropped.
    assert not any("audio" in n for n, _ in m.named_parameters())
    assert any("vision_tower" in n for n, _ in m.named_parameters())
    assert processor.tokenizer is not None
    assert torch.cuda.memory_allocated() / 2**30 < 9.0


@pytest.mark.gpu
@pytest.mark.slow
def test_use_cache_false_matches_true():  # regression of transformers #45242
    m, processor, _ = load_model(ModelSpec("e2b", "bf16"))
    ids = processor.tokenizer("def add(a, b):", return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        a = m(ids, use_cache=True).logits
        b = m(ids, use_cache=False).logits
    torch.testing.assert_close(a, b)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_e2b_n1_identical_to_original():
    from gemmaloop.model.looped import LoopConfig

    base, processor, _ = load_model(ModelSpec("e2b", "bf16"))
    ids = processor.tokenizer("The capital of France is", return_tensors="pt").input_ids.to(
        base.device
    )
    with torch.no_grad():
        ref = base(ids).logits
    del base
    gc.collect()
    torch.cuda.empty_cache()
    looped, _, _ = load_model(ModelSpec("e2b", "bf16", loop=LoopConfig(n_loops=1)))
    with torch.no_grad():
        out = looped(ids).logits
    torch.testing.assert_close(ref, out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Vision amendment (Task 4b)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_vision_kept_audio_dropped():
    m, processor, _ = load_model(ModelSpec("e2b", "bf16"))

    names = [n for n, _ in m.named_parameters()]
    vision = [(n, p) for n, p in m.named_parameters() if "vision" in n]
    assert vision, "the vision tower must be kept"
    assert not any("audio" in n for n in names), "the audio tower must be dropped"
    assert all(not p.requires_grad for _, p in vision), "the vision tower must be frozen"


@pytest.mark.gpu
@pytest.mark.slow
def test_image_forward_runs():
    m, processor, _ = load_model(ModelSpec("e2b", "bf16"))
    kwargs = _model_kwargs(_image_inputs(processor), m.device)

    with torch.no_grad():
        logits = m(**kwargs).logits
    assert torch.isfinite(logits).all()

    generated = m.generate(**kwargs, max_new_tokens=16, do_sample=False)
    assert generated.shape[1] > kwargs["input_ids"].shape[1]


@pytest.mark.gpu
@pytest.mark.slow
def test_n1_identical_with_image():
    from gemmaloop.model.looped import LoopConfig

    base, processor, _ = load_model(ModelSpec("e2b", "bf16"))
    kwargs = _model_kwargs(_image_inputs(processor), base.device)
    with torch.no_grad():
        ref = base(**kwargs).logits

    del base
    gc.collect()
    torch.cuda.empty_cache()

    looped, _, _ = load_model(ModelSpec("e2b", "bf16", loop=LoopConfig(n_loops=1)))
    with torch.no_grad():
        out = looped(**kwargs).logits
    torch.testing.assert_close(ref, out, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
def test_nf4_vision_not_quantized():
    from bitsandbytes.nn import Linear4bit

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m, processor, _ = load_model(ModelSpec("e2b", "nf4"))

    vision_linears = [
        module
        for name, module in m.named_modules()
        if "vision_tower" in name and isinstance(module, torch.nn.Linear)
    ]
    assert vision_linears, "the vision tower must be present"
    assert not any(
        isinstance(module, Linear4bit)
        for name, module in m.named_modules()
        if "vision" in name
    ), "vision linears must stay bf16, not NF4"

    vision_param = next(m.model.vision_tower.parameters())
    assert vision_param.device.type == "cuda"
    assert vision_param.dtype == torch.bfloat16

    ple = dict(m.named_parameters())["model.language_model.embed_tokens_per_layer.weight"]
    assert ple.device.type == "cpu"
    assert not any("audio" in n for n, _ in m.named_parameters())

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"nf4 + vision peak allocated: {peak:.3f} GiB")
    assert peak < 4.5, f"nf4 peak allocated {peak:.3f} GiB"


def test_lora_targets_language_only():
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4VisionConfig

    from tests.model.tiny import tiny_config

    pattern = language_lora_target_modules()
    assert re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(pattern, "model.language_model.layers.0.mlp.down_proj")
    assert not re.fullmatch(pattern, "model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not re.fullmatch(pattern, "model.embed_vision.linear")

    vision_config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=64,
        max_position_embeddings=64,
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=tiny_config(), vision_config=vision_config, audio_config=None)
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=pattern),
    )

    lora_names = [name for name, _ in peft_model.named_parameters() if "lora_" in name]
    assert lora_names, "the language stack must receive LoRA adapters"
    assert not any("vision" in name for name in lora_names)


def test_image_token_ids():
    assert IMAGE_TOKEN_IDS == (255999, 258880, 258882)


# ---------------------------------------------------------------------------
# Adapter / loop-state / nf4 regressions (fix round 1)
# ---------------------------------------------------------------------------


def _save_tiny_adapter(directory, *, seed=0):
    """Create a tiny LoRA adapter with non-zero B weights and save it."""
    from peft import LoraConfig, get_peft_model
    from tests.model.tiny import tiny_text_model

    base = tiny_text_model(seed=seed)
    peft_model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"]),
    )
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.25)
    peft_model.save_pretrained(directory)
    return str(directory)


def _save_loop_adapter(
    directory,
    *,
    source_seed=0,
    adapter_seed=1,
    bridge="scalar",
    ple_mode="shared",
    n_loops=1,
    bridge_logit=0.0,
    loop_ple_value=None,
    adapter_policy=None,
    block=None,
):
    """Save a tiny LoRA adapter *and* a loop-state sidecar in ``directory``."""
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, save_loop_state
    from tests.model.tiny import tiny_config, tiny_text_model

    plan = make_layer_plan(tiny_config().to_dict())
    source = tiny_text_model(seed=source_seed)
    apply_loops(
        source,
        plan,
        LoopConfig(
            n_loops=n_loops,
            bridge=bridge,
            ple_mode=ple_mode,
            adapter_policy=adapter_policy,
            block=block,
        ),
    )
    with torch.no_grad():
        source.loop_bridge_logit.fill_(bridge_logit)
        if loop_ple_value is not None:
            source.loop_ple.fill_(loop_ple_value)
    save_loop_state(source, directory)
    _save_tiny_adapter(directory, seed=adapter_seed)
    return str(directory)


def test_bf16_adapter_chain_merges_all_but_last(tmp_path):
    from peft import PeftModel
    from tests.model.tiny import tiny_text_model

    first = _save_tiny_adapter(tmp_path / "first", seed=0)
    second = _save_tiny_adapter(tmp_path / "second", seed=1)
    model = tiny_text_model(seed=42)
    q_name = "layers.0.self_attn.q_proj.weight"
    before = model.get_parameter(q_name).detach().clone()

    result = _apply_adapters(model, model, (first, second), "bf16")

    # The first adapter is merged into the base weights ...
    base_model = result.get_base_model() if hasattr(result, "get_base_model") else result
    assert not torch.equal(base_model.get_parameter(q_name).detach(), before)
    # ... and the second one is the single live PEFT adapter.
    assert isinstance(result, PeftModel)
    assert set(result.peft_config) == {"default"}
    assert result.active_adapter == "default"


def test_adapter_does_not_override_evaluation_n(tmp_path):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import (
        LoopConfig,
        apply_loops,
        save_loop_state,
        set_n_loops,
    )
    from tests.model.tiny import tiny_config, tiny_text_model

    plan = make_layer_plan(tiny_config().to_dict())
    source = tiny_text_model(seed=0)
    apply_loops(source, plan, LoopConfig(n_loops=1))
    # Training ends on the sampler's last draw; it is diagnostic only and must
    # never become the evaluation N.
    set_n_loops(source, 4)
    with torch.no_grad():
        source.loop_bridge_logit.fill_(1.25)
    adapter_dir = tmp_path / "adapter_with_loop_state"
    save_loop_state(source, adapter_dir)
    _save_tiny_adapter(adapter_dir, seed=1)

    payload = json.loads((adapter_dir / "loop_config.json").read_text(encoding="utf-8"))
    assert payload["n_loops_last_sampled"] == 4
    assert "n_loops" not in payload

    target = tiny_text_model(seed=2)
    apply_loops(target, plan, LoopConfig(n_loops=1))
    _apply_adapters(target, target, (str(adapter_dir),), "bf16")

    assert target.loop_bridge_logit.item() == pytest.approx(1.25)
    # Loading the adapter must not resurrect the transient training N.
    assert target._gemmaloop_n_loops == 1

    ids = torch.randint(0, 256, (1, 8), generator=torch.Generator().manual_seed(0))
    outputs = {}
    for n in (1, 2, 3):
        set_n_loops(target, n)
        assert target._gemmaloop_n_loops == n
        with torch.no_grad():
            outputs[n] = target(input_ids=ids).last_hidden_state.clone()

    assert not torch.equal(outputs[1], outputs[2])
    assert not torch.equal(outputs[2], outputs[3])
    assert not torch.equal(outputs[1], outputs[3])


def test_plevec_adapter_supplies_ple_mode_on_load(tmp_path):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, save_loop_state
    from tests.model.tiny import tiny_config, tiny_text_model

    plan = make_layer_plan(tiny_config().to_dict())
    source = tiny_text_model(seed=0)
    apply_loops(
        source,
        plan,
        LoopConfig(n_loops=4, bridge="scalar", ple_mode="loop_vector"),
    )
    with torch.no_grad():
        source.loop_ple.fill_(0.3)
    adapter_dir = tmp_path / "l_plevec"
    save_loop_state(source, adapter_dir)
    _save_tiny_adapter(adapter_dir, seed=1)

    # The CLI only knows --bridge/--loops; ple_mode must come from the adapter.
    resolved = _resolve_loop_config(
        LoopConfig(n_loops=2, bridge="scalar"), (str(adapter_dir),)
    )
    assert resolved is not None
    assert resolved.n_loops == 2
    assert resolved.ple_mode == "loop_vector"
    assert resolved.bridge == "scalar"

    target = tiny_text_model(seed=2)
    apply_loops(target, plan, resolved)
    _apply_adapters(target, target, (str(adapter_dir),), "bf16")

    assert target._gemmaloop_n_loops == 2
    torch.testing.assert_close(
        target.loop_ple, torch.full_like(target.loop_ple, 0.3)
    )
    ids = torch.randint(0, 256, (1, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = target(input_ids=ids).last_hidden_state
    assert torch.isfinite(logits).all()


def test_adapter_loop_config_rejects_mismatched_bridge(tmp_path):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops, save_loop_state
    from tests.model.tiny import tiny_config, tiny_text_model

    plan = make_layer_plan(tiny_config().to_dict())
    source = tiny_text_model(seed=0)
    apply_loops(source, plan, LoopConfig(n_loops=2, bridge="damped"))
    save_loop_state(source, tmp_path)

    with pytest.raises(ValueError, match="bridge"):
        _resolve_loop_config(
            LoopConfig(n_loops=2, bridge="scalar"), (str(tmp_path),)
        )


def test_adapter_chain_metadata_uses_last_loop_state(tmp_path):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from tests.model.tiny import tiny_config, tiny_text_model

    first = _save_loop_adapter(
        tmp_path / "first",
        source_seed=0,
        adapter_seed=1,
        bridge="damped",
        bridge_logit=0.1,
    )
    last = _save_loop_adapter(
        tmp_path / "last",
        source_seed=2,
        adapter_seed=3,
        bridge="scalar",
        bridge_logit=1.25,
    )
    spec = LoopConfig(n_loops=2, bridge="scalar")

    resolved = _resolve_loop_config(spec, (first, last))
    assert resolved is not None
    assert resolved.n_loops == 2
    assert resolved.bridge == "scalar"  # the last adapter, not the first
    assert resolved.ple_mode == "shared"

    plan = make_layer_plan(tiny_config().to_dict())
    target = tiny_text_model(seed=4)
    apply_loops(target, plan, resolved)
    _apply_adapters(target, target, (first, last), "bf16")

    assert target.loop_bridge_logit.item() == pytest.approx(1.25)
    assert target._gemmaloop_n_loops == 2


def test_adapter_chain_with_plain_last_adapter_keeps_earlier_loop_state(tmp_path):
    from gemmaloop.model.layer_plan import make_layer_plan
    from gemmaloop.model.looped import LoopConfig, apply_loops
    from tests.model.tiny import tiny_config, tiny_text_model

    looped = _save_loop_adapter(
        tmp_path / "looped",
        source_seed=0,
        adapter_seed=1,
        bridge="damped",
        bridge_logit=0.1,
    )
    plain = _save_tiny_adapter(tmp_path / "plain", seed=2)
    spec = LoopConfig(n_loops=3, bridge="damped")

    resolved = _resolve_loop_config(spec, (looped, plain))
    assert resolved is not None
    assert resolved.n_loops == 3
    assert resolved.bridge == "damped"

    plan = make_layer_plan(tiny_config().to_dict())
    target = tiny_text_model(seed=4)
    apply_loops(target, plan, resolved)
    _apply_adapters(target, target, (looped, plain), "bf16")

    assert target.loop_bridge_logit.item() == pytest.approx(0.1)
    assert target._gemmaloop_n_loops == 3


def test_legacy_loop_config_n_loops_is_ignored_without_spec(tmp_path):
    adapter_dir = _save_loop_adapter(
        tmp_path / "legacy",
        bridge="scalar",
        n_loops=4,
        bridge_logit=0.3,
    )
    config_path = Path(adapter_dir) / "loop_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("n_loops_last_sampled", None)
    payload["n_loops"] = 4  # legacy pre-fix file
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = _resolve_loop_config(None, (adapter_dir,))
    assert resolved is not None
    assert resolved.n_loops == 1


def test_nf4_adapter_chain_rejected_before_loading():
    with mock.patch(
        "gemmaloop.model.load.AutoModelForImageTextToText.from_pretrained"
    ) as from_pretrained:
        with pytest.raises(ValueError, match="adapter chains require bf16"):
            load_model(ModelSpec("e2b", "nf4", adapters=("a", "b")))
    from_pretrained.assert_not_called()


@pytest.mark.gpu
@pytest.mark.slow
def test_whole_model_move_keeps_ple_on_cpu():
    m, processor, _ = load_model(ModelSpec("e2b", "bf16"))
    m.to("cuda")
    ple = dict(m.named_parameters())["model.language_model.embed_tokens_per_layer.weight"]
    assert ple.device.type == "cpu"
    ids = processor.tokenizer("The capital of France is", return_tensors="pt").input_ids.to(
        m.device
    )
    with torch.no_grad():
        logits = m(ids).logits
    assert logits.device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.slow
def test_nf4_e2b_keeps_ple_on_cpu_with_vision_on_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m, processor, _ = load_model(ModelSpec("e2b", "nf4"))

    ple = dict(m.named_parameters())["model.language_model.embed_tokens_per_layer.weight"]
    assert ple.device.type == "cpu"
    assert not any("audio" in n for n, _ in m.named_parameters())
    assert any("vision_tower" in n for n, _ in m.named_parameters())
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert peak < 4.5, f"nf4 peak allocated {peak:.3f} GiB"

    ids = processor.tokenizer("The capital of France is", return_tensors="pt").input_ids.to(
        m.device
    )
    with torch.no_grad():
        m(ids)


def test_loop_only_adapter_refuses_merge(tmp_path):
    from gemmaloop.model.looped import assert_adapter_merge_allowed
    from tests.model.tiny import tiny_text_model

    loop_only = _save_loop_adapter(
        tmp_path / "loop_only",
        adapter_policy="loop_only",
        block=(5, 7),
    )
    other = _save_tiny_adapter(tmp_path / "other", seed=2)
    with pytest.raises(ValueError, match="loop_only.*cannot be merged"):
        assert_adapter_merge_allowed(loop_only)
    with pytest.raises(ValueError, match="loop_only.*cannot be merged"):
        _apply_adapters(
            tiny_text_model(), tiny_text_model(), (loop_only, other), "bf16"
        )


def test_train_init_adapter_refuses_merge_of_loop_only(tmp_path):
    import scripts.train as train_script

    loop_only = _save_loop_adapter(
        tmp_path / "init_loop_only",
        adapter_policy="loop_only",
        block=(5, 7),
    )
    with pytest.raises(ValueError, match="loop_only.*cannot be merged"):
        train_script._merge_init_adapter(
            None, None, Path(loop_only), "e2b", "bf16"
        )


def test_resolve_loop_config_inherits_and_checks_block_and_policy(tmp_path):
    from gemmaloop.model.looped import LoopConfig

    adapter = _save_loop_adapter(
        tmp_path / "geometry",
        adapter_policy="loop_only",
        block=(6, 10),
        bridge="gch",
    )
    resolved = _resolve_loop_config(
        LoopConfig(n_loops=2, bridge="gch"), (adapter,)
    )
    assert resolved is not None
    assert resolved.block == (6, 10)
    assert resolved.adapter_policy == "loop_only"

    with pytest.raises(ValueError, match="loop block"):
        _resolve_loop_config(
            LoopConfig(n_loops=2, bridge="gch", block=(5, 6)), (adapter,)
        )

    with pytest.raises(ValueError, match="adapter_policy"):
        _resolve_loop_config(
            LoopConfig(
                n_loops=2,
                bridge="gch",
                adapter_policy="always",
            ),
            (adapter,),
        )
