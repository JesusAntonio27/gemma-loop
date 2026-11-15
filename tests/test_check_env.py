"""Unit tests for ``scripts/check_env.py`` decision logic (no GPU needed)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scripts import check_env as ce

GIB = 2**30
E2B, E4B = ce.PINNED_MODEL_IDS


def make_env(**overrides) -> ce.Environment:
    values = dict(
        platform_system="linux",
        platform_machine="x86_64",
        python_version=(3, 12),
        torch_version="2.11.0+cu128",
        torch_cuda_version="12.8",
        cuda_available=False,
        gpu_name=None,
        compute_capability=None,
        driver_version=None,
        nvidia_smi_available=False,
        nvidia_smi_gpu=False,
        driver_gpu_name=None,
        vram_total_bytes=None,
        vram_free_bytes=None,
        artifacts_path=Path("/artifacts"),
        artifacts_writable=True,
        artifacts_free_bytes=100 * GIB,
        hf_hub_offline="1",
        hf_datasets_offline="1",
        model_snapshots={E2B: "/cache/hf/hub/snapshots/abc", E4B: None},
    )
    values.update(overrides)
    return ce.Environment(**values)


def by_name(
    env: ce.Environment,
    min_free_gib: float = 20.0,
    min_vram_gib: float = ce.DEFAULT_MIN_VRAM_GIB,
    require_gpu: bool = False,
) -> dict[str, ce.Check]:
    return {
        check.name: check
        for check in ce.decide(
            env,
            min_free_gib=min_free_gib,
            min_vram_gib=min_vram_gib,
            require_gpu=require_gpu,
        )
    }


def gpu_env(**overrides) -> ce.Environment:
    values = dict(
        cuda_available=True,
        gpu_name="NVIDIA L4",
        compute_capability=(8, 9),
        driver_version="570.124.06",
        nvidia_smi_available=True,
        nvidia_smi_gpu=True,
        driver_gpu_name="NVIDIA L4",
        vram_total_bytes=24 * GIB,
        vram_free_bytes=22 * GIB,
    )
    values.update(overrides)
    return make_env(**values)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_cpu_environment_passes_without_a_gpu():
    env = make_env()
    checks = ce.decide(env)
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_no_gpu_is_reported_not_failed():
    checks = by_name(make_env())
    assert checks["gpu"].ok
    assert "no GPU" in checks["gpu"].detail
    assert checks["compute capability"].detail == "n/a (no GPU)"
    assert checks["driver"].detail == "n/a (no GPU)"
    assert checks["vram"].detail == "n/a (no GPU)"


def test_ampere_or_newer_gpu_passes():
    checks = ce.decide(gpu_env())
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_hopper_and_blackwell_capabilities_pass():
    for capability in ((9, 0), (10, 0), (12, 0)):
        check = ce.check_compute_capability(capability)
        assert check.ok, (capability, check)


# ---------------------------------------------------------------------------
# GPU decision logic
# ---------------------------------------------------------------------------
def test_below_ampere_fails_for_bf16():
    checks = by_name(gpu_env(compute_capability=(7, 5), gpu_name="NVIDIA T4"))
    assert not checks["compute capability"].ok
    assert "8.0" in checks["compute capability"].detail


def test_unknown_capability_fails():
    checks = by_name(gpu_env(compute_capability=None))
    assert not checks["compute capability"].ok


def test_driver_below_r570_fails():
    checks = by_name(gpu_env(driver_version="535.104.05"))
    assert not checks["driver"].ok
    assert "R570" in checks["driver"].detail


def test_missing_driver_fails_when_a_gpu_is_visible():
    checks = by_name(gpu_env(driver_version=None))
    assert not checks["driver"].ok


def test_driver_parse_and_r570_boundary():
    assert ce.parse_driver_version("570.124.06") == (570, 124, 6)
    assert ce.parse_driver_version("617.42") == (617, 42)
    assert ce.parse_driver_version("not a version") is None
    assert ce.parse_driver_version(None) is None
    assert ce.check_driver("570.0").ok
    assert not ce.check_driver("569.99").ok


def test_vram_unknown_fails_when_a_gpu_is_visible():
    checks = by_name(gpu_env(vram_total_bytes=None))
    assert not checks["vram"].ok


def test_vram_floor_rejects_tiny_gpus():
    assert ce.check_vram(8 * GIB, 7 * GIB).ok
    assert ce.check_vram(24 * GIB, 22 * GIB).ok
    small = ce.check_vram(4 * GIB, 3 * GIB)
    assert not small.ok
    assert "4.0 GiB" in small.detail
    assert not by_name(gpu_env(vram_total_bytes=4 * GIB))["vram"].ok


def test_driver_sees_gpu_but_torch_does_not_fail_clearly():
    # This is the old false-green case: nvidia-smi works, torch is CPU-only.
    env = make_env(
        nvidia_smi_available=True,
        nvidia_smi_gpu=True,
        driver_gpu_name="NVIDIA L4",
        driver_version="570.124.06",
    )
    checks = by_name(env)
    assert not checks["gpu"].ok
    assert "NVIDIA L4" in checks["gpu"].detail
    assert "not to torch" in checks["gpu"].detail
    assert not checks["torch"].ok
    assert not checks["compute capability"].ok
    assert not checks["vram"].ok
    # The independently probed driver is still validated, not reported n/a.
    assert checks["driver"].ok


def test_driver_sees_gpu_with_old_driver_fails_even_when_torch_is_cpu():
    env = make_env(
        nvidia_smi_available=True,
        nvidia_smi_gpu=True,
        driver_gpu_name="NVIDIA L4",
        driver_version="535.104.05",
    )
    checks = by_name(env)
    assert not checks["gpu"].ok
    assert not checks["driver"].ok
    assert "R570" in checks["driver"].detail


def test_require_gpu_rejects_a_cpu_only_host():
    checks = by_name(make_env(), require_gpu=True)
    assert not checks["gpu"].ok
    assert "--require-gpu" in checks["gpu"].detail
    assert "nvidia-smi is unavailable" in checks["gpu"].detail


def test_require_gpu_accepts_a_visible_gpu():
    checks = by_name(gpu_env(), require_gpu=True)
    assert all(check.ok for check in checks.values()), [
        c for c in checks.values() if not c.ok
    ]


def test_parse_nvidia_smi_output():
    assert ce.parse_nvidia_smi_output("NVIDIA L4, 570.124.06\n") == (
        True,
        "NVIDIA L4",
        "570.124.06",
    )
    assert ce.parse_nvidia_smi_output("") == (False, None, None)
    assert ce.parse_nvidia_smi_output("NVIDIA H100 80GB HBM3, 570.86.15") == (
        True,
        "NVIDIA H100 80GB HBM3",
        "570.86.15",
    )


def test_read_nvidia_smi_with_mocked_subprocess_output():
    class Result:
        returncode = 0
        stdout = "NVIDIA L4, 570.124.06\n"

    assert ce._read_nvidia_smi(lambda *_args, **_kwargs: Result()) == (
        True,
        True,
        "NVIDIA L4",
        "570.124.06",
    )

    class Missing:
        returncode = 127
        stdout = ""

    assert ce._read_nvidia_smi(lambda *_args, **_kwargs: Missing()) == (
        False,
        False,
        None,
        None,
    )


def test_gpu_requires_cu128_torch_build():
    checks = by_name(gpu_env(torch_cuda_version="12.6"))
    assert not checks["torch"].ok
    assert "12.8" in checks["torch"].detail


def test_cpu_only_torch_build_is_allowed_without_a_gpu():
    checks = by_name(make_env(torch_version="2.11.0+cpu", torch_cuda_version=None))
    assert checks["torch"].ok
    assert "CPU-only" in checks["torch"].detail


def test_unimportable_torch_fails():
    checks = by_name(make_env(torch_version=None, torch_cuda_version=None))
    assert not checks["torch"].ok


# ---------------------------------------------------------------------------
# Platform / Python / disk / HF cache
# ---------------------------------------------------------------------------
def test_platform_validation():
    assert ce.check_platform("linux", "x86_64").ok
    assert ce.check_platform("win32", "AMD64").ok
    assert not ce.check_platform("plan9", "x86_64").ok
    assert not ce.check_platform("linux", "i686").ok


def test_python_must_be_312():
    assert ce.check_python((3, 12)).ok
    assert not ce.check_python((3, 11)).ok
    assert not ce.check_python((3, 13)).ok


def test_artifacts_disk_threshold_and_writability():
    assert ce.check_disk(Path("/artifacts"), True, 30 * GIB, 20).ok
    below = ce.check_disk(Path("/artifacts"), True, 5 * GIB, 20)
    assert not below.ok
    assert "5.0 GiB" in below.detail
    assert not ce.check_disk(Path("/artifacts"), False, 100 * GIB, 20).ok


def test_hf_offline_settings_must_agree():
    assert ce.check_offline_env("1", "1").ok
    assert ce.check_offline_env(None, None).ok
    assert not ce.check_offline_env("1", None).ok
    assert not ce.check_offline_env("0", "1").ok


def test_required_model_missing_fails_but_optional_model_missing_passes():
    missing_e2b = by_name(make_env(model_snapshots={E2B: None, E4B: None}))
    assert not missing_e2b[f"model {E2B}"].ok
    assert missing_e2b[f"model {E4B}"].ok

    missing_e4b = by_name(make_env(model_snapshots={E2B: "/cache/e2b", E4B: None}))
    assert missing_e4b[f"model {E2B}"].ok
    assert missing_e4b[f"model {E4B}"].ok


def test_find_snapshot_uses_refs_then_any_complete_snapshot(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / ("models--" + E2B.replace("/", "--"))
    first = repo / "snapshots" / "aaa"
    first.mkdir(parents=True)
    (first / "config.json").write_text("{}", encoding="utf-8")
    (first / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")

    # A snapshot without config.json or without a weight file is incomplete.
    incomplete = repo / "snapshots" / "bbb"
    incomplete.mkdir()
    config_only = repo / "snapshots" / "ddd"
    config_only.mkdir()
    (config_only / "config.json").write_text("{}", encoding="utf-8")

    assert ce.find_snapshot(hub, E2B) == first

    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("bbb\n", encoding="utf-8")
    # refs/main points at an incomplete snapshot, so fall back to the complete one.
    assert ce.find_snapshot(hub, E2B) == first

    second = repo / "snapshots" / "ccc"
    second.mkdir()
    (second / "config.json").write_text("{}", encoding="utf-8")
    (second / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert ce.find_snapshot(hub, E2B, revision="ccc") == second
    assert ce.find_snapshot(hub, E2B, revision="ddd") is None
    assert ce.find_snapshot(hub, E2B, revision="missing") is None


def test_gather_environment_is_cpu_safe_and_creates_artifacts(tmp_path):
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeVersion:
        cuda = "12.8"

    class FakeTorch:
        __version__ = "2.11.0+cu128"
        version = FakeVersion()
        cuda = FakeCuda()

    hub_repo = (
        tmp_path / "hf" / "hub" / ("models--" + E2B.replace("/", "--"))
        / "snapshots" / "abc"
    )
    hub_repo.mkdir(parents=True)
    (hub_repo / "config.json").write_text("{}", encoding="utf-8")
    (hub_repo / "model.safetensors").write_text("", encoding="utf-8")

    environ = {
        "GEMMALOOP_ARTIFACTS": str(tmp_path / "artifacts"),
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    env = ce.gather_environment(
        environ=environ,
        system="linux",
        machine="x86_64",
        python_version=(3, 12),
        torch_module=FakeTorch(),
        nvidia_smi=lambda: (False, False, None, None),
        disk_usage=lambda _path: (0, 0, 50 * GIB),
    )

    assert env.cuda_available is False
    assert env.nvidia_smi_gpu is False
    assert env.artifacts_writable
    assert (tmp_path / "artifacts").is_dir()
    assert env.model_snapshots[E2B] == str(hub_repo)
    assert env.model_snapshots[E4B] is None
    checks = ce.decide(env)
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_gather_environment_honours_pinned_revision(tmp_path):
    hub = tmp_path / "hf" / "hub"
    repo = hub / ("models--" + E2B.replace("/", "--"))
    for revision in ("aaa", "bbb"):
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_text("", encoding="utf-8")

    environ = {
        "GEMMALOOP_ARTIFACTS": str(tmp_path / "artifacts"),
        "HF_HOME": str(tmp_path / "hf"),
        "GEMMALOOP_MODEL_REVISION": "bbb",
    }
    env = ce.gather_environment(
        environ=environ,
        system="linux",
        machine="x86_64",
        python_version=(3, 12),
        torch_module=None,
        nvidia_smi=lambda: (False, False, None, None),
        disk_usage=lambda _path: (0, 0, 50 * GIB),
    )
    assert env.model_snapshots[E2B] == str(repo / "snapshots" / "bbb")


def _fake_gpu_torch():
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA L4"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (8, 9)

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return 22 * GIB, 24 * GIB

    class FakeVersion:
        cuda = "12.8"

    class FakeTorch:
        __version__ = "2.11.0+cu128"
        version = FakeVersion()
        cuda = FakeCuda()

    return FakeTorch()


def _fake_cpu_torch():
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeVersion:
        cuda = "12.8"

    class FakeTorch:
        __version__ = "2.11.0+cu128"
        version = FakeVersion()
        cuda = FakeCuda()

    return FakeTorch()


def _make_complete_snapshot(tmp_path) -> Path:
    snapshot = (
        tmp_path / "hf" / "hub" / ("models--" + E2B.replace("/", "--"))
        / "snapshots" / "abc"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("", encoding="utf-8")
    return snapshot


def test_gather_environment_gpu_path_with_mocked_smi_and_torch(tmp_path):
    _make_complete_snapshot(tmp_path)
    environ = {
        "GEMMALOOP_ARTIFACTS": str(tmp_path / "artifacts"),
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    env = ce.gather_environment(
        environ=environ,
        system="linux",
        machine="x86_64",
        python_version=(3, 12),
        torch_module=_fake_gpu_torch(),
        nvidia_smi=lambda: (True, True, "NVIDIA L4", "570.124.06"),
        disk_usage=lambda _path: (0, 0, 50 * GIB),
    )

    assert env.cuda_available
    assert env.nvidia_smi_gpu
    assert env.gpu_name == "NVIDIA L4"
    assert env.driver_version == "570.124.06"
    checks = ce.decide(env)
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_gather_environment_driver_gpu_torch_cpu_is_not_green(tmp_path):
    environ = {
        "GEMMALOOP_ARTIFACTS": str(tmp_path / "artifacts"),
        "HF_HOME": str(tmp_path / "hf"),
    }
    env = ce.gather_environment(
        environ=environ,
        system="linux",
        machine="x86_64",
        python_version=(3, 12),
        torch_module=_fake_cpu_torch(),
        nvidia_smi=lambda: (True, True, "NVIDIA L4", "535.104.05"),
        disk_usage=lambda _path: (0, 0, 50 * GIB),
    )

    assert env.cuda_available is False
    assert env.nvidia_smi_gpu is True
    checks = by_name(env)
    assert not checks["gpu"].ok
    assert not checks["torch"].ok
    assert not checks["driver"].ok
    assert not checks["compute capability"].ok
    assert not checks["vram"].ok


# ---------------------------------------------------------------------------
# CLI exit status
# ---------------------------------------------------------------------------
def test_main_returns_zero_when_every_check_passes(monkeypatch, capsys):
    env = make_env()
    monkeypatch.setattr(ce, "gather_environment", lambda **_kwargs: env)
    monkeypatch.setattr(ce, "_load_torch", lambda: object())

    assert ce.main(["--min-free-gb", "0"]) == 0
    assert "Environment check OK" in capsys.readouterr().out


def test_main_returns_one_and_explains_failures(monkeypatch, capsys):
    env = replace(make_env(), model_snapshots={E2B: None, E4B: None})
    monkeypatch.setattr(ce, "gather_environment", lambda **_kwargs: env)
    monkeypatch.setattr(ce, "_load_torch", lambda: object())

    assert ce.main(["--min-free-gb", "0"]) == 1
    captured = capsys.readouterr()
    assert "Environment check FAILED" in captured.err
    assert E2B in captured.err
