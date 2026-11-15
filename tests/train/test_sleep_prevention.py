"""Windows sleep prevention helper (Task 14b fix round 3)."""

from __future__ import annotations

from types import SimpleNamespace

from gemmaloop.train.perf import prevent_sleep


class _Kernel32:
    def __init__(self):
        self.calls: list[int] = []

    def SetThreadExecutionState(self, flags):  # noqa: N802 - Windows API name
        self.calls.append(int(flags))


def test_windows_sleep_prevention_sets_and_clears_the_flag():
    kernel32 = _Kernel32()
    ctypes_module = SimpleNamespace(windll=SimpleNamespace(kernel32=kernel32))
    guard = prevent_sleep(platform="nt", ctypes_module=ctypes_module)
    with guard:
        assert guard.active
        assert kernel32.calls == [0x80000001]
    assert kernel32.calls == [0x80000001, 0x80000000]
    assert not guard.active


def test_non_windows_is_a_no_op():
    kernel32 = _Kernel32()
    ctypes_module = SimpleNamespace(windll=SimpleNamespace(kernel32=kernel32))
    guard = prevent_sleep(platform="posix", ctypes_module=ctypes_module)
    with guard:
        assert not guard.active
    assert kernel32.calls == []


def test_api_failure_is_swallowed():
    class Boom:
        def SetThreadExecutionState(self, flags):  # noqa: N802
            raise OSError("not available")

    ctypes_module = SimpleNamespace(windll=SimpleNamespace(kernel32=Boom()))
    guard = prevent_sleep(platform="nt", ctypes_module=ctypes_module)
    with guard:
        assert not guard.active
