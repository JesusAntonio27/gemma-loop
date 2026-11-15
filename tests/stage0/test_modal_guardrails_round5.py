"""Round-5: reject the known-invalid GPU nonpreemptible option offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_modal_guardrails_round1 import _Clock, _FakeModal, _Function, _run

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_round5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_built_gpu_spec_never_sets_nonpreemptible():
    script = _load_script()
    captured: list[dict] = []

    class FakeImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        @classmethod
        def from_registry(cls, tag, **kwargs):
            return cls()

        def apt_install(self, *args):
            return self

        def pip_install(self, *args):
            return self

        def add_local_python_source(self, *args):
            return self

    class FakeApp:
        def __init__(self, name):
            pass

        def function(self, **kwargs):
            captured.append(kwargs)
            return lambda fn: SimpleNamespace()

    script._build_modal_app(SimpleNamespace(App=FakeApp, Image=FakeImage))
    gpu = [config for config in captured if "gpu" in config]
    preflight = [config for config in captured if "gpu" not in config]
    assert len(gpu) == 1 and len(preflight) == 1
    assert "nonpreemptible" not in gpu[0]
    assert "nonpreemptible" not in preflight[0]


def test_static_check_rejects_nonpreemptible_gpu():
    script = _load_script()
    with pytest.raises(ValueError, match="Non-preemptible"):
        script._validate_modal_function_options(
            gpu="A100-80GB",
            nonpreemptible=True,
            scaledown_window_s=10,
            timeout_s=1800,
        )


def test_static_check_matches_modal_scaledown_window_bounds():
    script = _load_script()
    for bad in (1, 3601):
        with pytest.raises(ValueError, match="scaledown_window"):
            script._validate_modal_function_options(
                gpu="A100-80GB",
                nonpreemptible=False,
                scaledown_window_s=bad,
                timeout_s=1800,
            )


def test_build_app_rejects_bad_scaledown_offline():
    script = _load_script()
    # Validation runs before m.App, so a bare object is enough and no Modal
    # contact is possible.
    with pytest.raises(ValueError, match="scaledown_window"):
        script._build_modal_app(SimpleNamespace(), scaledown_window_s=1)


def test_run_mode_rejects_bad_scaledown_before_any_contact(tmp_path):
    script = _load_script()
    clock = _Clock()
    modal = _FakeModal()
    gpu = _Function(wall_clock=clock)
    preflight = _Function(wall_clock=clock, result={"ok": True})
    with pytest.raises(ValueError, match="scaledown_window"):
        script.run_mode(
            **_run(
                tmp_path,
                modal,
                gpu,
                preflight,
                wall_clock=clock,
                scaledown_window_s=1,
            )
        )
    assert gpu.calls == 0
    assert preflight.calls == 0
    assert modal.created == []
