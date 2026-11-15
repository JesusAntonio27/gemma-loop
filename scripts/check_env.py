"""Environment check for Gemma-Loop container and GPU hosts.

Prints and validates:

* the host platform (the container target is Linux on x86_64);
* the Python version (the project pins 3.12.x);
* the torch build and its CUDA runtime;
* the visible GPU, its compute capability (>= 8.0 for bf16) and VRAM;
* the NVIDIA driver (R570 or newer for the CUDA 12.8 build), probed with
  ``nvidia-smi`` independently of torch so a driver/torch mismatch fails;
* free disk space at ``GEMMALOOP_ARTIFACTS``;
* the Hugging Face offline switches; and
* the presence of the pinned Gemma 4 snapshots in the Hugging Face cache.

The checker is CPU-safe: on a host without a GPU it reports ``no GPU`` and
still validates everything else.  Use ``--require-gpu`` as a cloud/container
gate so a host without a visible GPU fails.  It exits with status 1 if any check
fails.
"""

from __future__ import annotations

import argparse
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DRIVER_MAJOR = 570
REQUIRED_PYTHON = (3, 12)
TORCH_CUDA_PREFIX = "12.8"
DEFAULT_MIN_FREE_GIB = 20.0
DEFAULT_MIN_VRAM_GIB = 8.0

# Model IDs used by ``gemmaloop.model.load``.  E2B is the development model and
# must be cached before a run; E4B is reported but optional for E2B-only runs.
PINNED_MODEL_IDS = ("google/gemma-4-E2B-it", "google/gemma-4-E4B-it")
REQUIRED_MODEL_IDS = (PINNED_MODEL_IDS[0],)

OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
ARTIFACTS_ENV_VAR = "GEMMALOOP_ARTIFACTS"
DEFAULT_ARTIFACTS = Path.home() / "ML" / "gemmaloop-artifacts"


@dataclass(frozen=True)
class Check:
    """One named validation entry."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Environment:
    """Everything the decision logic needs, already gathered from the host."""

    platform_system: str
    platform_machine: str
    python_version: tuple[int, int]
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    driver_version: str | None
    nvidia_smi_available: bool
    nvidia_smi_gpu: bool
    driver_gpu_name: str | None
    vram_total_bytes: int | None
    vram_free_bytes: int | None
    artifacts_path: Path
    artifacts_writable: bool
    artifacts_free_bytes: int | None
    hf_hub_offline: str | None
    hf_datasets_offline: str | None
    model_snapshots: Mapping[str, str | None]


# ---------------------------------------------------------------------------
# Pure decision logic
# ---------------------------------------------------------------------------
def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 2**30:.1f} GiB"


def parse_driver_version(text: str | None) -> tuple[int, ...] | None:
    """Parse an ``nvidia-smi`` driver version such as ``570.124.06``."""
    if not text:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(group) for group in match.groups() if group is not None)


def parse_nvidia_smi_output(stdout: str) -> tuple[bool, str | None, str | None]:
    """Parse ``nvidia-smi --query-gpu=name,driver_version`` output.

    Returns ``(gpu_visible, gpu_name, driver_version)``.  An empty output means
    the command ran but reported no GPU (not a driver failure).
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False, None, None
    name, _, driver = lines[0].partition(",")
    parsed_name = name.strip() or None
    parsed_driver = driver.strip() or None
    return parsed_name is not None, parsed_name, parsed_driver


def check_platform(system: str, machine: str) -> Check:
    known = {"linux", "win32", "darwin"}
    supported_machines = {"x86_64", "amd64", "aarch64", "arm64"}
    if system not in known:
        return Check("platform", False, f"unsupported OS {system!r}")
    if machine.lower() not in supported_machines:
        return Check("platform", False, f"unsupported architecture {machine!r}")
    target = " (container target)" if system == "linux" and machine.lower() in {
        "x86_64",
        "amd64",
    } else ""
    return Check("platform", True, f"{system}/{machine}{target}")


def check_python(version: tuple[int, int]) -> Check:
    ok = tuple(version[:2]) == REQUIRED_PYTHON
    detail = f"{version[0]}.{version[1]} (required 3.12.x)"
    return Check("python", ok, detail)


def check_torch(
    torch_version: str | None,
    torch_cuda_version: str | None,
    cuda_available: bool,
    gpu_visible: bool = False,
) -> Check:
    if torch_version is None:
        return Check("torch", False, "torch is not importable in this environment")
    if cuda_available:
        if not (torch_cuda_version or "").startswith(TORCH_CUDA_PREFIX):
            return Check(
                "torch",
                False,
                f"torch {torch_version} has CUDA {torch_cuda_version or 'none'}; "
                f"the GPU stack requires a CUDA {TORCH_CUDA_PREFIX} build",
            )
        return Check("torch", True, f"{torch_version} (CUDA {torch_cuda_version})")
    cuda_build = torch_cuda_version or "CPU-only build"
    if gpu_visible:
        return Check(
            "torch",
            False,
            f"torch {torch_version} ({cuda_build}) cannot see the GPU visible to "
            "the driver; check the driver/CUDA version, --gpus all / the NVIDIA "
            "container toolkit, and that torch is a CUDA build",
        )
    return Check("torch", True, f"{torch_version} (CUDA {cuda_build}); no GPU detected")


def check_gpu(env: Environment, require_gpu: bool = False) -> Check:
    if env.cuda_available:
        if not env.gpu_name:
            return Check("gpu", False, "CUDA is available but the device name is unknown")
        return Check("gpu", True, env.gpu_name)
    if env.nvidia_smi_gpu:
        visible = env.driver_gpu_name or "unknown device"
        return Check(
            "gpu",
            False,
            f"GPU visible to the driver ({visible}) but not to torch; check the "
            "driver/CUDA version, --gpus all / the NVIDIA container toolkit, and "
            "that torch is not a CPU-only build",
        )
    if require_gpu:
        detail = "no GPU visible (--require-gpu)"
        if not env.nvidia_smi_available:
            detail += (
                "; nvidia-smi is unavailable (is the NVIDIA container toolkit "
                "installed and --gpus all passed?)"
            )
        return Check("gpu", False, detail)
    return Check("gpu", True, "no GPU (CPU-only mode)")


def check_compute_capability(capability: tuple[int, int] | None) -> Check:
    if capability is None:
        return Check("compute capability", False, "could not determine compute capability")
    value = tuple(capability[:2])
    if value < MIN_COMPUTE_CAPABILITY:
        return Check(
            "compute capability",
            False,
            f"{value[0]}.{value[1]} < {MIN_COMPUTE_CAPABILITY[0]}."
            f"{MIN_COMPUTE_CAPABILITY[1]}; bf16 requires Ampere or newer "
            "(T4 7.5 and Volta are unsupported)",
        )
    return Check(
        "compute capability",
        True,
        f"{value[0]}.{value[1]} (>= {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]})",
    )


def check_driver(version: str | None) -> Check:
    parsed = parse_driver_version(version)
    if parsed is None:
        return Check(
            "driver",
            False,
            "could not read the NVIDIA driver version (is nvidia-smi installed?)",
        )
    if parsed[0] < MIN_DRIVER_MAJOR:
        return Check(
            "driver",
            False,
            f"{version} < R{MIN_DRIVER_MAJOR}; the Cu128 build needs a driver "
            f"R{MIN_DRIVER_MAJOR}+",
        )
    return Check("driver", True, f"{version} (>= R{MIN_DRIVER_MAJOR})")


def check_vram(
    total_bytes: int | None,
    free_bytes: int | None,
    min_vram_gib: float = DEFAULT_MIN_VRAM_GIB,
) -> Check:
    if total_bytes is None:
        return Check("vram", False, "could not read GPU memory")
    total_gib = total_bytes / 2**30
    if total_gib + 1e-9 < min_vram_gib:
        return Check(
            "vram",
            False,
            f"GPU has {total_gib:.1f} GiB total; at least {min_vram_gib:.1f} GiB "
            "is required for the intended runs",
        )
    return Check(
        "vram",
        True,
        f"{_gib(free_bytes)} free of {_gib(total_bytes)} "
        f"(>= {min_vram_gib:.1f} GiB)",
    )


def check_disk(
    path: Path,
    writable: bool,
    free_bytes: int | None,
    min_free_gib: float,
) -> Check:
    if not writable or free_bytes is None:
        return Check("artifacts disk", False, f"{path} is not writable")
    free_gib = free_bytes / 2**30
    if free_gib + 1e-9 < min_free_gib:
        return Check(
            "artifacts disk",
            False,
            f"{free_gib:.1f} GiB free at {path}; at least {min_free_gib:.1f} GiB "
            "is required",
        )
    return Check("artifacts disk", True, f"{free_gib:.1f} GiB free at {path}")


def check_offline_env(hub: str | None, datasets: str | None) -> Check:
    hub_offline = _truthy(hub)
    datasets_offline = _truthy(datasets)
    if hub_offline == datasets_offline:
        if hub_offline:
            return Check(
                "hf offline",
                True,
                "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (cache-only)",
            )
        return Check("hf offline", True, "offline mode disabled (network allowed)")
    return Check(
        "hf offline",
        False,
        f"inconsistent settings: HF_HUB_OFFLINE={hub!r}, "
        f"HF_DATASETS_OFFLINE={datasets!r}; set both to 1 for cache-only runs",
    )


def check_model_snapshot(
    model_id: str,
    snapshot: str | None,
    required: bool,
) -> Check:
    name = f"model {model_id}"
    if snapshot:
        return Check(name, True, f"cached snapshot {snapshot}")
    if required:
        return Check(
            name,
            False,
            "not found in the Hugging Face cache; download it or mount /cache "
            "before running",
        )
    return Check(name, True, "not cached (optional for E2B-only runs)")


def decide(
    env: Environment,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    min_vram_gib: float = DEFAULT_MIN_VRAM_GIB,
    require_gpu: bool = False,
) -> list[Check]:
    """Turn probed values into pass/fail checks (no I/O; unit-testable)."""
    gpu_visible = env.cuda_available or env.nvidia_smi_gpu
    checks = [
        check_platform(env.platform_system, env.platform_machine),
        check_python(env.python_version),
        check_torch(
            env.torch_version,
            env.torch_cuda_version,
            env.cuda_available,
            gpu_visible=gpu_visible,
        ),
        check_gpu(env, require_gpu=require_gpu),
    ]
    if env.cuda_available:
        checks.extend(
            [
                check_compute_capability(env.compute_capability),
                check_driver(env.driver_version),
                check_vram(env.vram_total_bytes, env.vram_free_bytes, min_vram_gib),
            ]
        )
    elif gpu_visible:
        # nvidia-smi sees a GPU but torch does not: a driver/CUDA/toolkit
        # mismatch.  Never report these as n/a -- that was the false green.
        checks.extend(
            [
                Check(
                    "compute capability",
                    False,
                    "torch cannot see CUDA, so compute capability (>= 8.0) "
                    "cannot be verified",
                ),
                check_driver(env.driver_version),
                Check(
                    "vram",
                    False,
                    "torch cannot see CUDA, so VRAM cannot be read",
                ),
            ]
        )
    else:
        checks.extend(
            [
                Check("compute capability", True, "n/a (no GPU)"),
                Check("driver", True, "n/a (no GPU)"),
                Check("vram", True, "n/a (no GPU)"),
            ]
        )
    checks.extend(
        [
            check_disk(
                env.artifacts_path,
                env.artifacts_writable,
                env.artifacts_free_bytes,
                min_free_gib,
            ),
            check_offline_env(env.hf_hub_offline, env.hf_datasets_offline),
        ]
    )
    for model_id in PINNED_MODEL_IDS:
        checks.append(
            check_model_snapshot(
                model_id,
                env.model_snapshots.get(model_id),
                required=model_id in REQUIRED_MODEL_IDS,
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Host probing
# ---------------------------------------------------------------------------
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
_WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def snapshot_complete(path: Path) -> bool:
    """A snapshot is usable only if it has a config and at least one weight file.

    ``config.json`` alone can be written early by an interrupted download, so it
    is not enough to call the snapshot complete.
    """
    if not (path / "config.json").is_file():
        return False
    try:
        entries = list(path.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.name in _WEIGHT_INDEX_FILES:
            return True
        if entry.suffix.lower() in _WEIGHT_SUFFIXES and entry.is_file():
            return True
    return False


def find_snapshot(
    hub_dir: Path, model_id: str, revision: str | None = None
) -> Path | None:
    """Return the cached snapshot directory for ``model_id``, if complete.

    ``revision`` pins an exact snapshot; without it the ``refs/main`` pointer is
    tried first and then any complete snapshot (config plus weight files).
    """
    repository = Path(hub_dir) / ("models--" + model_id.replace("/", "--"))
    snapshots = repository / "snapshots"
    candidates: list[Path] = []
    if revision:
        candidates.append(snapshots / revision)
    else:
        ref = repository / "refs" / "main"
        if ref.is_file():
            try:
                pointed = ref.read_text(encoding="utf-8").strip()
            except OSError:
                pointed = ""
            if pointed:
                candidates.append(snapshots / pointed)
        if snapshots.is_dir():
            candidates.extend(
                sorted(path for path in snapshots.iterdir() if path.is_dir())
            )
    for candidate in candidates:
        if snapshot_complete(candidate):
            return candidate
    return None


def _read_nvidia_smi(
    runner: Callable[..., object] = subprocess.run,
) -> tuple[bool, bool, str | None, str | None]:
    """Probe the driver independently of torch.

    Returns ``(available, gpu_visible, gpu_name, driver_version)``.
    ``nvidia-smi`` is injected into the container by the NVIDIA toolkit, so it
    can report a GPU even when torch cannot initialise CUDA.
    """
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, False, None, None
    if getattr(result, "returncode", 1) != 0:
        return False, False, None, None
    found, name, driver = parse_nvidia_smi_output(
        getattr(result, "stdout", "") or ""
    )
    return True, found, name, driver


def _torch_fields(torch_module: object | None) -> dict[str, object]:
    if torch_module is None:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "gpu_name": None,
            "compute_capability": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
        }

    version = getattr(torch_module, "__version__", None)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    cuda = getattr(torch_module, "cuda", None)
    try:
        available = bool(cuda is not None and cuda.is_available())
    except Exception:  # driver/library problems must be reported, not raised
        available = False

    gpu_name = None
    capability = None
    total = None
    free = None
    if available:
        try:
            gpu_name = str(cuda.get_device_name(0))
        except Exception:
            gpu_name = None
        try:
            capability = tuple(cuda.get_device_capability(0))[:2]
        except Exception:
            capability = None
        try:
            free, total = cuda.mem_get_info()
        except Exception:
            free = None
            total = None

    return {
        "torch_version": str(version) if version else None,
        "torch_cuda_version": str(cuda_version) if cuda_version else None,
        "cuda_available": available,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "vram_total_bytes": int(total) if total is not None else None,
        "vram_free_bytes": int(free) if free is not None else None,
    }


def artifacts_path(environ: Mapping[str, str] | None = None) -> Path:
    source = os.environ if environ is None else environ
    raw = source.get(ARTIFACTS_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_ARTIFACTS


def hf_cache_root(environ: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    """Return ``(hf_home, hub_cache)`` honouring HF_HOME/HF_HUB_CACHE."""
    source = os.environ if environ is None else environ
    raw_home = source.get("HF_HOME")
    home = Path(raw_home).expanduser() if raw_home else Path.home() / ".cache" / "huggingface"
    raw_hub = source.get("HF_HUB_CACHE")
    hub = Path(raw_hub).expanduser() if raw_hub else home / "hub"
    return home, hub


def gather_environment(
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int] | None = None,
    torch_module: object | None = None,
    driver_version: str | None = None,
    nvidia_smi: (
        Callable[[], tuple[bool, bool, str | None, str | None]] | None
    ) = None,
    disk_usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage,
) -> Environment:
    """Probe the host.  Every external value can be injected for testing."""
    source = os.environ if environ is None else environ
    torch_fields = _torch_fields(torch_module)
    probe = nvidia_smi if nvidia_smi is not None else _read_nvidia_smi
    smi_available, smi_found, smi_name, smi_driver = probe()

    path = artifacts_path(source)
    writable = False
    free_bytes = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        writable = path.is_dir() and os.access(path, os.W_OK)
        if writable:
            free_bytes = int(disk_usage(path)[2])
    except OSError:
        writable = False

    _, hub = hf_cache_root(source)
    revision = source.get("GEMMALOOP_MODEL_REVISION") or None
    snapshots: dict[str, str | None] = {}
    for model_id in PINNED_MODEL_IDS:
        found = find_snapshot(hub, model_id, revision)
        snapshots[model_id] = str(found) if found else None

    if driver_version is None:
        driver_version = smi_driver

    return Environment(
        platform_system=system if system is not None else sys.platform,
        platform_machine=machine if machine is not None else platform_module.machine(),
        python_version=(
            python_version if python_version is not None else tuple(sys.version_info[:2])
        ),
        artifacts_path=path,
        artifacts_writable=writable,
        artifacts_free_bytes=free_bytes,
        hf_hub_offline=source.get("HF_HUB_OFFLINE"),
        hf_datasets_offline=source.get("HF_DATASETS_OFFLINE"),
        model_snapshots=snapshots,
        driver_version=driver_version,
        nvidia_smi_available=smi_available,
        nvidia_smi_gpu=smi_found,
        driver_gpu_name=smi_name,
        **torch_fields,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_torch() -> object | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma-Loop environment check")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
        help=f"minimum free space at GEMMALOOP_ARTIFACTS in GiB "
        f"(default: {DEFAULT_MIN_FREE_GIB:.0f})",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=DEFAULT_MIN_VRAM_GIB,
        help=f"minimum total GPU VRAM in GiB (default: {DEFAULT_MIN_VRAM_GIB:.0f})",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail when no GPU is visible (cloud/container gate)",
    )
    args = parser.parse_args(argv)

    env = gather_environment(torch_module=_load_torch())
    checks = decide(
        env,
        min_free_gib=args.min_free_gb,
        min_vram_gib=args.min_vram_gb,
        require_gpu=args.require_gpu,
    )

    print("Gemma-Loop environment check")
    for check in checks:
        status = " OK " if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    failures = [check for check in checks if not check.ok]
    if failures:
        print(
            f"\nEnvironment check FAILED ({len(failures)} failure(s))",
            file=sys.stderr,
        )
        for check in failures:
            print(f"  - {check.name}: {check.detail}", file=sys.stderr)
        return 1
    print("\nEnvironment check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
