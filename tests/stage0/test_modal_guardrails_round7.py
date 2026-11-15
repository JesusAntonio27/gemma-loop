"""Round-7: flashinfer's JIT also needs a host C++ compiler and ninja.

Attempt 5 reached image build but the toolchain preflight only proved nvcc.
These tests keep the check offline (fake subprocess) and prove a missing
``g++`` or ``ninja`` aborts before any GPU request.
"""

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
    spec = importlib.util.spec_from_file_location("modal_stage0_round7", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _fake_tool_run(script, monkeypatch, *, missing: str) -> None:
    """Succeed for every JIT tool except ``missing``, which is not on PATH."""

    def run(argv, **kwargs):
        tool = argv[0]
        if tool == missing:
            raise FileNotFoundError(tool)
        if tool == "nvcc":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Cuda compilation tools, release 12.8, V12.8.61"
                ),
                stderr="",
            )
        if tool == "g++":
            return SimpleNamespace(
                returncode=0,
                stdout="g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
                stderr="",
            )
        if tool == "ninja":
            return SimpleNamespace(returncode=0, stdout="1.11.1.1\n", stderr="")
        raise AssertionError(f"unexpected tool invocation: {argv!r}")

    monkeypatch.setattr(script.subprocess, "run", run)


def test_image_installs_build_essential_and_pinned_ninja_before_pip():
    script = _load_script()
    calls: list[str] = []
    tags: list[str] = []
    apt_specs: list[str] = []
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
            apt_specs.extend(args)
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
    assert "build-essential" in apt_specs
    assert script.NINJA_VERSION in pip_specs
    assert script.NINJA_VERSION.startswith("ninja==")
    assert script.VLLM_VERSION in pip_specs
    assert calls.index("apt_install") < calls.index("pip_install")
    assert calls[-1] == "add_local_python_source"


@pytest.mark.parametrize(
    ("missing", "pattern"),
    [("g++", r"g\+\+"), ("ninja", "ninja")],
)
def test_missing_jit_helper_fails_the_toolchain_check(
    monkeypatch, missing, pattern
):
    script = _load_script()
    monkeypatch.delenv("CXX", raising=False)
    _fake_tool_run(script, monkeypatch, missing=missing)

    with pytest.raises(RuntimeError, match=pattern):
        script._verify_jit_toolchain()


@pytest.mark.parametrize("missing", ["g++", "ninja"])
def test_missing_jit_helper_preflight_makes_zero_gpu_calls(
    tmp_path, monkeypatch, missing
):
    script = _load_script()
    monkeypatch.delenv("CXX", raising=False)
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
    _fake_tool_run(script, monkeypatch, missing=missing)

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
