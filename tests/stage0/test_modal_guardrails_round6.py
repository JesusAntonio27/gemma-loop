"""Round-6: CUDA devel image and preflight JIT toolchain checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_modal_guardrails_round1 import (
    _App,
    _Clock,
    _FakeModal,
    _Function,
    _run,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_round6", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_image_uses_pinned_cuda_devel_registry():
    script = _load_script()
    calls: list[str] = []
    tags: list[str] = []
    pip_specs: list[str] = []

    class FakeImage:
        @classmethod
        def from_registry(cls, tag, **kwargs):
            calls.append("from_registry")
            tags.append(tag)
            assert kwargs.get("add_python") == "3.12"
            return cls()

        def apt_install(self, *args):
            calls.append("apt_install")
            return self

        def pip_install(self, *args):
            calls.append("pip_install")
            pip_specs.extend(args)
            return self

        def add_local_python_source(self, *args):
            calls.append("add_local_python_source")
            return self

    class FakeApp:
        def __init__(self, name):
            pass

        def function(self, **kwargs):
            return lambda fn: SimpleNamespace()

    script._build_modal_app(SimpleNamespace(App=FakeApp, Image=FakeImage))
    assert tags == [script.CUDA_IMAGE_TAG]
    assert tags[0].startswith("nvidia/cuda:")
    assert "-devel-" in tags[0]
    assert "12.8" in tags[0]
    assert script.VLLM_VERSION in pip_specs
    assert script.HF_HUB_VERSION in pip_specs
    assert script.TRANSFORMERS_VERSION in pip_specs
    assert calls[-1] == "add_local_python_source"


def test_optional_cuda_image_digest(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(script, "CUDA_IMAGE_DIGEST", "sha256:deadbeef")
    assert script._modal_base_image_ref() == "nvidia/cuda@sha256:deadbeef"


def test_missing_nvcc_fails_the_toolchain_check(monkeypatch):
    script = _load_script()

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvcc")

    monkeypatch.setattr(script.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="nvcc"):
        script._verify_jit_toolchain()


def test_missing_nvcc_preflight_makes_zero_gpu_calls(tmp_path, monkeypatch):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    app = _App()
    gpu = _Function(wall_clock=clock)

    monkeypatch.setattr(
        script,
        "_import_remote_closure",
        lambda: (
            SimpleNamespace(__name__="huggingface_hub"),
            SimpleNamespace(__name__="transformers"),
            SimpleNamespace(__name__="vllm"),
        ),
    )

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvcc")

    monkeypatch.setattr(script.subprocess, "run", missing)

    class RealPreflight:
        def __init__(self):
            self.calls = 0

        def spawn(self, payload):
            self.calls += 1
            result = script.preflight_check(payload)
            return SimpleNamespace(
                get=lambda timeout=None: result,
                cancel=lambda **kwargs: None,
            )

        def get_current_stats(self):
            return SimpleNamespace(num_running_inputs=0, num_total_runners=0)

    preflight = RealPreflight()
    result = script.run_mode(
        **_run(
            tmp_path,
            modal,
            gpu,
            preflight,
            app=app,
            wall_clock=clock,
        )
    )
    assert result == 1
    assert preflight.calls == 1
    assert gpu.calls == 0
    assert app.stops == 1
