"""Modal rejects any image build step after ``add_local_*`` (InvalidError at launch).

Regression for the first real throughput probe (2026-11-05): ``.env(...)`` came
after ``.add_local_python_source(...)`` and the launch failed before any
container ran.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "modal_stage0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("modal_stage0_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingImage:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.local_added = False

    def _step(self, name: str):
        if self.local_added and not name.startswith("add_local"):
            raise AssertionError(f"build step {name!r} after add_local_*")
        self.calls.append(name)
        return self

    def apt_install(self, *args, **kwargs):
        return self._step("apt_install")

    def pip_install(self, *args, **kwargs):
        return self._step("pip_install")

    def env(self, *args, **kwargs):
        return self._step("env")

    def run_commands(self, *args, **kwargs):
        return self._step("run_commands")

    def add_local_python_source(self, *args, **kwargs):
        self.calls.append("add_local_python_source")
        self.local_added = True
        return self


class _FakeApp:
    def __init__(self, name):
        self.name = name

    def function(self, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def test_add_local_is_the_last_image_step():
    module = _load_script()
    calls: list[str] = []
    fake_modal = SimpleNamespace(
        App=_FakeApp,
        Image=SimpleNamespace(
            debian_slim=lambda **kw: _RecordingImage(calls),
            from_registry=lambda *a, **kw: _RecordingImage(calls),
        ),
        Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
        Secret=SimpleNamespace(from_name=lambda *a, **k: object()),
        Retries=lambda *a, **k: None,
    )
    try:
        module._build_modal_app(fake_modal)
    except AssertionError:
        raise
    except Exception:
        # Other fake gaps are irrelevant here; the image chain ran first.
        pass
    assert "add_local_python_source" in calls
    assert calls[-1] == "add_local_python_source"
