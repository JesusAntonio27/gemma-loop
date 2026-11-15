"""Modal Stage 0 sampling job (local Python entry point, stock vLLM).

The paid entry point is a **local Python script** with the Modal client supplied
in an isolated tool environment -- never the ``modal run`` CLI, which creates an
app and builds an image before any budget check::

    # offline dry run (does not import modal)
    uv run python scripts/modal_stage0.py --dry-run --pool-dir artifacts/stage0/pool

    # approve and launch (author approval required)
    uv run --with modal==1.6.3 python scripts/modal_stage0.py \
        --mode sample --pool-dir artifacts/stage0/pool --max-usd 12 --yes

    # download / delete
    uv run --with modal==1.6.3 python scripts/modal_stage0.py \
        --mode download --output-dir artifacts/stage0/samples
    uv run --with modal==1.6.3 python scripts/modal_stage0.py \
        --mode cleanup --output-dir artifacts/stage0/samples

Money safety (binding ruling): one container at a time, no retries, resources
declared on the function, worst case = (timeout + startup_timeout +
scaledown_window) x all-in rate x1.3 + image build, process-lifetime spend lock,
persistent monthly ledger, reservation before ``with app.run()``, shards
namespaced by a run id derived from the frozen pool/sampling, and a Volume
delete reminder on every exit path.

The registered Stage 0 limits are a **$30 usage limit / $0 spend limit**.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemmaloop.model.snapshot import MODEL_REVISION  # noqa: E402
from gemmaloop.stage0.modal_plan import (  # noqa: E402
    DEFAULT_CPU_CORES,
    DEFAULT_GPU,
    DEFAULT_IMAGE_BUILD_USD,
    DEFAULT_MEMORY_GIB,
    DEFAULT_SAFETY_MARGIN,
    DEFAULT_SCALEDOWN_WINDOW_SECONDS,
    DEFAULT_SHARD_SIZE,
    DEFAULT_STARTUP_SECONDS,
    CPU_HOURLY_PER_CORE_USD,
    HARD_GLOBAL_CAP_USD,
    MEMORY_HOURLY_PER_GIB_USD,
    check_budget,
    format_plan,
    runtime_cost_usd,
    sampling_plan,
    worst_case_usd,
)
from gemmaloop.stage0.spend import (  # noqa: E402
    SpendLedger,
    month_key,
    require_spend_lock,
)

MODEL_ID = "google/gemma-4-E2B-it"
MODAL_VERSION = "1.6.3"
VLLM_VERSION = "vllm==0.30.0"
TRANSFORMERS_VERSION = "transformers==5.22.1"
HF_HUB_VERSION = "huggingface_hub==2.4.0"
NINJA_VERSION = "ninja==1.11.1.1"
# Torch is pinned to the cu128 index, so the devel image must ship CUDA 12.8
# with nvcc at /usr/local/cuda for flashinfer's runtime JIT builds.  The JIT
# also needs a host C++ compiler and ninja, which the plain CUDA devel image
# does not guarantee.
CUDA_IMAGE_TAG = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
CUDA_IMAGE_DIGEST = os.environ.get("GEMMALOOP_MODAL_CUDA_IMAGE_DIGEST", "").strip()
VOLUME_NAME = "gemmaloop-stage0"
VOLUME_MOUNT = "/data"
SHARD_DIR = "shards"  # volume-relative
CONTAINER_MODEL_CACHE = "/tmp/gemmaloop-hf-cache"
MAX_CONTAINERS = 1
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_STARTUP_TIMEOUT_SECONDS = 600
MAX_STARTUP_TIMEOUT_SECONDS = 900  # hard ceiling; cannot be disabled
DEFAULT_SCALEDOWN_SECONDS = 10
PREFLIGHT_TIMEOUT_SECONDS = 120
# vLLM + torch + transformers need more than a token-sized container.
PREFLIGHT_MEMORY_MIB = 16384
PREFLIGHT_CPU = 1.0
PREFLIGHT_SCALEDOWN_SECONDS = 2  # Modal minimum; the launcher waits for exit
PREFLIGHT_EXIT_WAIT_SECONDS = 60
REMOTE_HEARTBEAT_POLL_SECONDS = 5.0
REMOTE_HEARTBEAT_INTERVAL_SECONDS = 30.0
PREFLIGHT_HEARTBEAT_INTERVAL_SECONDS = 2.0
PREFLIGHT_BACKGROUND_BEAT_TIMEOUT_SECONDS = 10.0
REMOTE_HEARTBEAT_GAP_SECONDS = 180.0
REMOTE_CALL_MARGIN_SECONDS = 10
REMOTE_HEARTBEAT_MIN_DELIVERED = 2
FAILURE_BOOKING_OVERHEAD_SECONDS = 60
# Functions that execute in the remote image.  The AST image-completeness test
# walks their bodies, the helpers they call, and the module-level closure.
REMOTE_FUNCTIONS = ("sample_shard", "preflight_check")
DEFAULT_CPU = 8.0
DEFAULT_MEMORY_GIB = 48
MODEL_MAX_CONTEXT = 131072
RECEIPT_NAME = "download_receipt.json"
LAUNCH_MANIFEST_NAME = "launch_manifest.json"
DELETE_COMMAND = (
    f"uvx --from modal=={MODAL_VERSION} modal volume delete {VOLUME_NAME} --yes"
)
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_new_tokens": 32768,
    "enable_thinking": True,
}
LITE_ENGINE = "modal-vllm"
LITE_RECORD_MAX_LENGTH = 8192  # registered Stage 0-lite rendered limit
MODEAL_LITE_CALL_KEY = "lite_call:"
DEFAULT_LITE_CALL_SIZE = 1400  # mirrors gemmaloop.stage0.modal_lite
MODAL_GPU_IDS = {
    "B300": "B300",
    "B200": "B200",
    "H200": "H200",
    "H100": "H100",
    "RTX PRO 6000": "RTX-PRO-6000",
    "A100-80GB": "A100-80GB",
    "A100-40GB": "A100-40GB",
    "L40S": "L40S",
    "A10": "A10",
    "L4": "L4",
}

modal = None  # imported lazily; --dry-run never imports it
_LLM_CACHE: dict[tuple[str, int], Any] = {}
_THOUGHT_RE = re.compile(r"<\|channel>thought\b(.*?)<channel\|>", re.DOTALL)


def _import_modal():
    global modal
    if modal is None:
        import modal as _modal  # noqa: PLC0415 - deliberately lazy

        modal = _modal
    return modal


class RemoteStartupTimeout(RuntimeError):
    """No in-container heartbeat appeared before the startup deadline."""


class RemoteStartupRetry(RuntimeError):
    """A second container started for the same call; startup retries are banned."""


class RemoteHeartbeatGap(RuntimeError):
    """The in-container heartbeat stopped for longer than the allowed gap."""


class RemoteCallDeadline(RuntimeError):
    """The whole remote call passed its local wall-clock deadline."""


class PreflightError(RuntimeError):
    """The CPU-only preflight failed; no GPU container may be requested."""


class HeartbeatWriteError(RuntimeError):
    """The in-container heartbeat failed three times in a row."""


_HEARTBEAT_READ_ERROR = object()


def _is_poll_timeout(exc: BaseException) -> bool:
    # Modal exposes its own ``modal.exception.TimeoutError``; tests and other
    # fakes use the builtin.  ``FunctionTimeoutError`` is a real function
    # failure, not a poll timeout, so match the plain class name only.
    return isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError"


def _cancel_call(call: Any) -> None:
    cancel = getattr(call, "cancel", None)
    if not callable(cancel):
        return
    try:
        cancel(terminate_containers=True)
    except TypeError:
        try:
            cancel()
        except Exception:
            pass
    except Exception:
        pass


def _read_heartbeat(heartbeat: Any, key: str | None) -> Any:
    if heartbeat is None or key is None:
        return None
    try:
        value = heartbeat.get(key)
    except Exception:
        # Distinguish a read outage from a confirmed missing beat (M-R2).
        return _HEARTBEAT_READ_ERROR
    return value if isinstance(value, dict) else None


def _probe_stats_with_retry(target_fn: Any) -> Any:
    """Read stats twice; failures are inconclusive and never cancel by themselves."""

    stats_fn = getattr(target_fn, "get_current_stats", None)
    if not callable(stats_fn):
        return None
    for _attempt in range(2):
        try:
            return stats_fn()
        except Exception:
            continue
    return None


def _call_remote(
    target_fn: Any,
    payload: dict[str, Any],
    *,
    wall_clock: Callable[[], float],
    startup_timeout_s: float,
    timeout_s: float,
    heartbeat: Any,
    heartbeat_key: str | None,
) -> Any:
    """Call a remote function with heartbeat liveness and hard deadlines.

    The only liveness signal is a heartbeat written by the remote function
    itself (a Modal ``Dict`` key).  Modal runner/task counts are never trusted
    as liveness; they are stop-only evidence of a restart or failed container
    before the first heartbeat.  ``get()`` is always bounded; the call is
    cancelled and the app is stopped on a second container, a heartbeat gap, a
    missing first heartbeat at the startup deadline, or the hard deadline.
    """

    spawn = getattr(target_fn, "spawn", None)
    if not callable(spawn):
        raise RuntimeError(
            "remote functions must expose spawn() so the launcher can poll with "
            "a bounded call.get(timeout=...)"
        )

    started = wall_clock()
    call = spawn(payload)
    # R-I3: Modal allows startup_timeout + timeout; the local bound mirrors it.
    hard_deadline = (
        started
        + float(startup_timeout_s)
        + float(timeout_s)
        + REMOTE_CALL_MARGIN_SECONDS
    )
    startup_deadline = started + min(
        float(startup_timeout_s), float(MAX_STARTUP_TIMEOUT_SECONDS)
    )
    container_id: str | None = None
    last_heartbeat: float | None = None
    last_token: tuple[Any, Any, Any] | None = None
    stats_seen = False
    stats_previous: int | None = None
    positive_seen = False
    dropped_after_positive = False

    while True:
        now = wall_clock()
        beat = _read_heartbeat(heartbeat, heartbeat_key)
        if beat is _HEARTBEAT_READ_ERROR:
            # A channel read outage is not proof that the container died.
            # Keep the last good beat fresh and let the hard deadline bound it.
            if last_heartbeat is not None:
                last_heartbeat = now
        elif beat is not None:
            beat_container = beat.get("container_id")
            token = (
                beat_container,
                beat.get("sequence"),
                beat.get("written_at"),
            )
            if container_id is None:
                container_id = str(beat_container) if beat_container else ""
                last_heartbeat = now
                last_token = token
                # Start a fresh runner baseline after liveness is established.
                stats_previous = None
            elif str(beat_container or "") != container_id:
                _cancel_call(call)
                raise RemoteStartupRetry(
                    "a second container started for the same call "
                    f"({container_id!r} -> {beat_container!r})"
                )
            elif token != last_token:
                last_heartbeat = now
                last_token = token

        # Stats are stop-only evidence, never liveness.  A normal cold start
        # goes 0 -> 1 when the first container is scheduled; that is not a
        # restart.  A second start is either (a) an increase after the count
        # dropped back to zero for this call, or (b) an increase after the
        # first heartbeat arrived.  Modal 1.6.3 FunctionCurrentStats has only
        # backlog/num_total_runners/num_running_inputs/input_headroom, so no
        # non-existent "failed container" field is consulted.
        stats = _probe_stats_with_retry(target_fn)
        if stats is not None:
            try:
                runners = max(0, int(getattr(stats, "num_total_runners", 0) or 0))
            except (TypeError, ValueError):
                runners = 0
            if last_heartbeat is None:
                if not stats_seen:
                    stats_seen = True
                    stats_previous = runners
                    positive_seen = runners > 0
                else:
                    previous = 0 if stats_previous is None else stats_previous
                    if runners > previous and positive_seen and dropped_after_positive:
                        _cancel_call(call)
                        raise RemoteStartupRetry(
                            "Modal stats show a container restart "
                            f"({previous} -> {runners}) before any heartbeat"
                        )
                    if runners == 0 and positive_seen:
                        dropped_after_positive = True
                    elif runners > 0:
                        positive_seen = True
                    stats_previous = runners
            else:
                # After liveness, keep the highest count seen, with a floor of
                # 1: one lagging/zero stats read must not make the next 1 look
                # like a second container.
                highest = max(1, stats_previous or 0)
                if stats_previous is None:
                    stats_previous = max(1, runners)
                elif runners > highest:
                    _cancel_call(call)
                    raise RemoteStartupRetry(
                        "Modal stats show a runner increase after the first "
                        f"heartbeat ({highest} -> {runners})"
                    )
                else:
                    stats_previous = max(highest, runners)

        now = wall_clock()
        if now >= hard_deadline:
            _cancel_call(call)
            raise RemoteCallDeadline(
                "remote call exceeded its startup allowance plus "
                f"{float(timeout_s):g}s timeout and "
                f"{REMOTE_CALL_MARGIN_SECONDS:g}s margin"
            )
        if last_heartbeat is None:
            if now >= startup_deadline:
                _cancel_call(call)
                raise RemoteStartupTimeout(
                    "no in-container heartbeat within "
                    f"{float(startup_timeout_s):g}s of startup"
                )
            next_deadline = min(startup_deadline, hard_deadline)
        else:
            if now - last_heartbeat > REMOTE_HEARTBEAT_GAP_SECONDS:
                _cancel_call(call)
                raise RemoteHeartbeatGap(
                    "no in-container heartbeat for more than "
                    f"{REMOTE_HEARTBEAT_GAP_SECONDS:g}s"
                )
            next_deadline = min(
                hard_deadline, last_heartbeat + REMOTE_HEARTBEAT_GAP_SECONDS
            )

        timeout = min(
            REMOTE_HEARTBEAT_POLL_SECONDS,
            max(0.001, next_deadline - now),
        )
        try:
            return call.get(timeout=timeout)
        except BaseException as exc:
            if _is_poll_timeout(exc):
                continue
            raise


def _wait_for_preflight_exit(
    target_fn: Any, *, wall_clock: Callable[[], float]
) -> None:
    """Do not start the GPU while the preflight CPU container is still alive."""

    stats_fn = getattr(target_fn, "get_current_stats", None)
    if not callable(stats_fn):
        return
    deadline = wall_clock() + PREFLIGHT_EXIT_WAIT_SECONDS
    while True:
        try:
            stats = stats_fn()
        except Exception:
            stats = None
        if stats is not None:
            running = int(getattr(stats, "num_running_inputs", 0) or 0)
            runners = int(getattr(stats, "num_total_runners", 0) or 0)
            if running == 0 and runners == 0:
                return
        if wall_clock() >= deadline:
            raise PreflightError(
                "could not confirm the preflight container exited before the "
                "GPU launch"
            )
        time.sleep(0.5)


def _stop_app(app: Any) -> None:
    """Best-effort stop: never wait for Modal to cycle containers again."""

    if app is None:
        return
    stop = getattr(app, "stop", None)
    if not callable(stop):
        return
    try:
        stop()
    except Exception as exc:
        print(
            f"warning: app.stop() failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def parse_completion(text: str) -> dict[str, Any]:
    text = text or ""
    match = _THOUGHT_RE.search(text)
    reasoning = match.group(1).strip() if match else ""
    content = _THOUGHT_RE.sub("", text).strip()
    content = re.sub(r"(?:<turn\|>|<eos>|<end_of_turn>)+$", "", content).strip()
    has_open = "<|channel>thought" in text
    has_close = "<channel|>" in text
    thought_closed = bool(match) or (not has_open) or (has_open and has_close)
    return {
        "reasoning": reasoning,
        "content": content,
        "thought_closed": thought_closed,
    }


def sampling_params_kwargs() -> dict[str, Any]:
    return {
        "n": 4,
        "temperature": float(SAMPLING["temperature"]),
        "top_p": float(SAMPLING["top_p"]),
        "top_k": int(SAMPLING["top_k"]),
        "max_tokens": int(SAMPLING["max_new_tokens"]),
        "skip_special_tokens": False,
    }


def llm_kwargs(model_path: str, max_model_len: int) -> dict[str, Any]:
    return {
        "model": model_path,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "max_model_len": int(max_model_len),
        "tensor_parallel_size": 1,
        "seed": 0,
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
    }


def compute_run_id(
    pool_manifest_sha256: str,
    *,
    shard_size: int,
    samples_per_prompt: int,
    model_revision: str = MODEL_REVISION,
    sampling: dict[str, Any] | None = None,
) -> str:
    payload = {
        "pool_manifest_sha256": pool_manifest_sha256,
        "shard_size": int(shard_size),
        "samples_per_prompt": int(samples_per_prompt),
        "model_revision": model_revision,
        "sampling": sampling or SAMPLING,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def _reminder(output_dir: Path) -> str:
    return (
        "Volume reminder: download and verify the shards, then delete the "
        f"Volume (storage bills past the usage limit):\n"
        f"  uv run --with modal=={MODAL_VERSION} python scripts/modal_stage0.py "
        f"--mode download --output-dir {output_dir}\n"
        f"  {DELETE_COMMAND}"
    )


def _lite_reminder(output_dir: Path) -> str:
    output_dir = Path(output_dir)
    receipt = output_dir / "modal_receipt.json"
    return (
        "Volume reminder: the lite shards are already downloaded and verified "
        f"under {output_dir / 'shards'}; after checking {receipt}, delete the "
        "Volume (storage bills past the usage limit).  The guarded cleanup uses "
        "the shared default spend lock:\n"
        f"  uv run --with modal=={MODAL_VERSION} python scripts/modal_stage0.py "
        f"--mode cleanup --output-dir {output_dir} --receipt {receipt} "
        f"--allow-partial\n"
        "Manual fallback (deletes EVERY run's shards on the shared Volume, so "
        f"only use it after checking them): {DELETE_COMMAND}"
    )


# ---------------------------------------------------------------------------
# Remote work
# ---------------------------------------------------------------------------
def _cached_engine(LLM: Any, model_path: str, max_model_len: int):
    key = (model_path, int(max_model_len))
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = LLM(**llm_kwargs(model_path, max_model_len))
    return _LLM_CACHE[key]


def _container_id() -> str:
    """Stable identity for this container/process, used to detect restarts."""

    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _heartbeat_write(
    heartbeat: Any,
    key: str | None,
    *,
    container_id: str,
    sequence: int,
    phase: str,
    state: dict[str, Any] | None = None,
) -> bool:
    if heartbeat is None or key is None:
        return False
    try:
        heartbeat[key] = {
            "container_id": container_id,
            "sequence": int(sequence),
            "phase": str(phase),
            "written_at": time.time(),
        }
    except Exception as exc:
        if state is not None:
            state["consecutive_failures"] = int(state["consecutive_failures"]) + 1
            state["failed_writes"] = int(state["failed_writes"]) + 1
            if (
                state["consecutive_failures"] >= 3
                and state.get("write_error") is None
            ):
                state["write_error"] = HeartbeatWriteError(
                    "the in-container heartbeat channel failed three times in a row"
                )
                state["write_error"].__cause__ = exc
        return False
    if state is not None:
        state["consecutive_failures"] = 0
        state["delivered"] = int(state["delivered"]) + 1
    return True


def _heartbeat_begin(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Deliver the first beat and start the background daemon writer."""

    heartbeat = payload.get("_heartbeat")
    key = payload.get("heartbeat_key")
    if heartbeat is None or key is None:
        return None
    state = {
        "heartbeat": heartbeat,
        "key": key,
        "container_id": _container_id(),
        "sequence": 0,
        "phase": "entered",
        "consecutive_failures": 0,
        "delivered": 0,
        "background_delivered": 0,
        "failed_writes": 0,
        "write_error": None,
        "stop": threading.Event(),
        "thread": None,
        "lock": threading.RLock(),
        "heartbeat_interval": float(
            payload.get("heartbeat_interval", REMOTE_HEARTBEAT_INTERVAL_SECONDS)
        ),
    }
    _heartbeat_beat(state)

    def loop() -> None:
        while not state["stop"].wait(state["heartbeat_interval"]):
            _heartbeat_beat(state, background=True)
            if state.get("write_error") is not None:
                break

    thread = threading.Thread(
        target=loop, name="gemmaloop-heartbeat", daemon=True
    )
    state["thread"] = thread
    thread.start()
    return state


def _heartbeat_beat(
    state: dict[str, Any] | None,
    *,
    raise_on_failure: bool = False,
    background: bool = False,
) -> bool:
    if state is None:
        return False
    with state["lock"]:
        if raise_on_failure and state.get("write_error") is not None:
            raise state["write_error"]
        state["sequence"] = int(state["sequence"]) + 1
        delivered = _heartbeat_write(
            state["heartbeat"],
            state["key"],
            container_id=state["container_id"],
            sequence=state["sequence"],
            phase=state["phase"],
            state=state,
        )
        if delivered and background:
            state["background_delivered"] = (
                int(state["background_delivered"]) + 1
            )
        if raise_on_failure and state.get("write_error") is not None:
            raise state["write_error"]
        return delivered


def _heartbeat_phase(state: dict[str, Any] | None, phase: str) -> None:
    if state is None:
        return
    state["phase"] = str(phase)
    # Safe point in the main thread: surface a dead writer loudly.
    _heartbeat_beat(state, raise_on_failure=True)


def _heartbeat_stop(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    state["stop"].set()
    thread = state.get("thread")
    if thread is not None:
        thread.join(timeout=2.0)


def _wait_for_background_heartbeat(state: dict[str, Any] | None) -> None:
    """Prove the same daemon writer used by the GPU function actually writes."""

    if state is None:
        return
    deadline = time.monotonic() + PREFLIGHT_BACKGROUND_BEAT_TIMEOUT_SECONDS
    while True:
        with state["lock"]:
            delivered = int(state["background_delivered"])
            error = state.get("write_error")
        if delivered >= 1:
            return
        if error is not None:
            raise error
        if time.monotonic() >= deadline:
            raise HeartbeatWriteError(
                "the preflight background heartbeat writer delivered no beat"
            )
        time.sleep(0.05)


def _import_remote_closure() -> tuple[Any, Any, Any]:
    """The exact third-party import closure used by the GPU function.

    ``preflight_check`` calls this same helper, so the CPU preflight cannot
    drift from the imports ``sample_shard`` actually needs.
    """

    import huggingface_hub
    import transformers
    import vllm

    return huggingface_hub, transformers, vllm


def preflight_check(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """CPU-only preflight: import the exact remote closure, report versions.

    It is registered with CPU=1, a vLLM-sized memory request, no GPU and a
    120 s timeout.  If any import in the remote closure is missing, this call
    fails before the launcher ever asks Modal for a GPU container.
    """

    from importlib import metadata as _metadata

    heartbeat_state = _heartbeat_begin(_payload or {})
    try:
        _heartbeat_phase(heartbeat_state, "importing_remote_closure")
        modules = _import_remote_closure()
        _heartbeat_phase(heartbeat_state, "checking_jit_toolchain")
        jit = _verify_jit_toolchain()
        _heartbeat_phase(heartbeat_state, "reading_versions")
        versions: dict[str, str] = {}
        for distribution in ("vllm", "transformers", "huggingface-hub"):
            try:
                versions[distribution] = _metadata.version(distribution)
            except Exception:
                versions[distribution] = "unknown"
        _wait_for_background_heartbeat(heartbeat_state)
        _heartbeat_beat(heartbeat_state, raise_on_failure=True)
        result = {
            "ok": True,
            "versions": versions,
            "model_revision": MODEL_REVISION,
            "engine": VLLM_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "huggingface_hub": HF_HUB_VERSION,
            "imported": [module.__name__ for module in modules],
            "jit_toolchain": jit,
        }
        if heartbeat_state is not None:
            result["heartbeat"] = {
                "container_id": heartbeat_state["container_id"],
                "delivered": heartbeat_state["delivered"],
                "background_delivered": heartbeat_state["background_delivered"],
                "failed_writes": heartbeat_state["failed_writes"],
                "sequence": heartbeat_state["sequence"],
            }
        return result
    finally:
        _heartbeat_stop(heartbeat_state)


def sample_shard(payload: dict[str, Any]) -> dict[str, Any]:
    """Sample one shard on the main thread; only the heartbeat is a daemon."""

    heartbeat_state = _heartbeat_begin(payload)
    try:
        result = _sample_shard_impl(payload, heartbeat_state)
        _heartbeat_beat(heartbeat_state, raise_on_failure=True)
        return result
    finally:
        _heartbeat_stop(heartbeat_state)


def _sample_shard_impl(
    payload: dict[str, Any], heartbeat_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample one shard, write it under its run-id namespace, and commit."""

    if payload.get("lite"):
        return _sample_lite_impl(payload, heartbeat_state)

    import json as _json
    import os as _os
    from pathlib import Path as _Path

    huggingface_hub, transformers, vllm = _import_remote_closure()
    snapshot_download = huggingface_hub.snapshot_download
    AutoTokenizer = transformers.AutoTokenizer
    LLM = vllm.LLM
    SamplingParams = vllm.SamplingParams

    run_id = str(payload["run_id"])
    shard_index = int(payload["shard_index"])
    problems = payload["problems"]
    run_dir = _Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"shard_{shard_index:05d}.jsonl"
    committed_marker = output_path.with_name(output_path.name + ".committed")
    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and committed_marker.exists()
    ):
        _heartbeat_phase(heartbeat_state, "already_committed")
        return {"shard": shard_index, "skipped": True, "samples": 0, "path": str(output_path)}

    _heartbeat_phase(heartbeat_state, "downloading_model")
    cache_dir = payload.get("model_cache_dir", CONTAINER_MODEL_CACHE)
    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir)
    _heartbeat_phase(heartbeat_state, "tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": problem["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(SAMPLING["enable_thinking"]),
        )
        for problem in problems
    ]
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts
    ]
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    engine_max_model_len = int(
        payload.get("engine_max_model_len", max_prompt_tokens + int(SAMPLING["max_new_tokens"]))
    )
    _heartbeat_phase(heartbeat_state, "loading_engine")
    llm = _cached_engine(LLM, model_path, engine_max_model_len)
    base_params = sampling_params_kwargs()
    params = []
    for problem in problems:
        per_prompt = dict(base_params)
        per_prompt["n"] = int(payload.get("samples_per_prompt", 4))
        per_prompt["seed"] = int(
            hashlib.sha256(str(problem["problem_id"]).encode("utf-8")).hexdigest()[:8], 16
        )
        params.append(SamplingParams(**per_prompt))
    _heartbeat_phase(heartbeat_state, "generating")
    outputs = llm.generate(prompts, params)

    _heartbeat_phase(heartbeat_state, "writing_shard")
    count = 0
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for problem, output in zip(problems, outputs):
            for sample_index, completion in enumerate(output.outputs):
                text = completion.text or ""
                parsed = parse_completion(text)
                record = {
                    "problem_id": problem["problem_id"],
                    "sample_index": sample_index,
                    "prompt": problem["prompt"],
                    "completion": text,
                    "reasoning": parsed["reasoning"],
                    "content": parsed["content"],
                    "finish_reason": str(getattr(completion, "finish_reason", "")),
                    "thought_closed": parsed["thought_closed"],
                    "tokens": len(getattr(completion, "token_ids", []) or []),
                    "engine": VLLM_VERSION,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "run_id": run_id,
                    "pool_manifest_sha256": payload.get("pool_manifest_sha256"),
                    "temperature": SAMPLING["temperature"],
                    "top_p": SAMPLING["top_p"],
                    "top_k": SAMPLING["top_k"],
                    "max_new_tokens": SAMPLING["max_new_tokens"],
                    "samples_per_prompt": int(payload.get("samples_per_prompt", 4)),
                    "max_model_len": engine_max_model_len,
                    "max_prompt_tokens": max_prompt_tokens,
                    "skip_special_tokens": False,
                    "gpu": payload.get("gpu", ""),
                    "shard_index": shard_index,
                }
                handle.write(_json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    _os.replace(tmp_path, output_path)
    committed_marker.write_text("committed\n", encoding="utf-8")

    _heartbeat_phase(heartbeat_state, "committing_volume")
    try:
        volume = payload.get("_volume")
        if volume is None:
            volume = _import_modal().Volume.from_name(
                VOLUME_NAME, create_if_missing=False
            )
        if volume is None:
            raise RuntimeError("no mounted Volume available to commit the shard")
        volume.commit()
    except BaseException:
        for path in (output_path, committed_marker):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "shard": shard_index,
        "skipped": False,
        "samples": count,
        "path": str(output_path),
        "max_model_len": engine_max_model_len,
    }


def _sample_lite_impl(
    payload: dict[str, Any], heartbeat_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample a whole-shard Stage 0-lite call with per-request budgets."""

    import json as _json
    import os as _os
    from pathlib import Path as _Path

    huggingface_hub, transformers, vllm = _import_remote_closure()
    snapshot_download = huggingface_hub.snapshot_download
    AutoTokenizer = transformers.AutoTokenizer
    LLM = vllm.LLM
    SamplingParams = vllm.SamplingParams

    run_id = str(payload["run_id"])
    run_dir = _Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = str(payload.get("gpu", ""))
    pool_sha = str(payload.get("pool_manifest_sha256") or "")
    model_revision = str(payload.get("model_revision") or MODEL_REVISION)
    samples_per_prompt = int(payload.get("samples_per_prompt", 4))
    engine_max_model_len = int(
        payload.get("engine_max_model_len", MODEL_MAX_CONTEXT)
    )

    _heartbeat_phase(heartbeat_state, "downloading_model")
    cache_dir = payload.get("model_cache_dir", CONTAINER_MODEL_CACHE)
    model_path = snapshot_download(
        MODEL_ID, revision=model_revision, cache_dir=cache_dir
    )
    _heartbeat_phase(heartbeat_state, "tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _heartbeat_phase(heartbeat_state, "loading_engine")
    llm = _cached_engine(LLM, model_path, engine_max_model_len)

    volume = payload.get("_volume")
    if volume is None:
        volume = _import_modal().Volume.from_name(
            VOLUME_NAME, create_if_missing=False
        )
    if volume is None:
        raise RuntimeError("no mounted Volume available to commit the shard")

    shards = list(payload.get("shards", []))
    results: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for shard in shards:
        shard_index = int(shard["shard_index"])
        output_path = run_dir / f"shard_{shard_index:05d}.jsonl"
        committed_marker = output_path.with_name(output_path.name + ".committed")
        if (
            output_path.exists()
            and output_path.stat().st_size > 0
            and committed_marker.exists()
        ):
            _heartbeat_phase(heartbeat_state, "already_committed")
            results.append(
                {
                    "shard": shard_index,
                    "skipped": True,
                    "committed": True,
                    "samples": 0,
                    "rows": len(
                        [
                            line
                            for line in output_path.read_text(
                                encoding="utf-8"
                            ).splitlines()
                            if line.strip()
                        ]
                    ),
                    "path": str(output_path),
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                }
            )
            continue
        for request in list(shard.get("requests", [])):
            pending.append((shard, request))

    # I1: one generation pass for the whole call, exactly like the probe.
    # Shard boundaries are only the write/commit granularity.
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(request.get("prompt") or "")}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(SAMPLING["enable_thinking"]),
        )
        for _shard, request in pending
    ]
    params = [
        SamplingParams(
            n=1,
            temperature=float(SAMPLING["temperature"]),
            top_p=float(SAMPLING["top_p"]),
            top_k=int(SAMPLING["top_k"]),
            max_tokens=int(request.get("max_new_tokens") or 0),
            skip_special_tokens=False,
            seed=int(request.get("seed") or 0),
        )
        for _shard, request in pending
    ]
    _heartbeat_phase(heartbeat_state, "generating")
    outputs = llm.generate(prompts, params) if prompts else []
    if len(outputs) != len(pending):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(pending)} requests"
        )

    by_shard: dict[int, list[tuple[dict[str, Any], Any]]] = {}
    for (shard, request), output in zip(pending, outputs):
        by_shard.setdefault(int(shard["shard_index"]), []).append((request, output))

    generated_tokens = 0
    for shard in shards:
        shard_index = int(shard["shard_index"])
        output_path = run_dir / f"shard_{shard_index:05d}.jsonl"
        committed_marker = output_path.with_name(output_path.name + ".committed")
        if any(
            int(result["shard"]) == shard_index and result.get("skipped")
            for result in results
        ):
            continue
        requests = list(shard.get("requests", []))
        call_cap = max(
            (int(request.get("max_new_tokens") or 0) for request in requests),
            default=0,
        )
        existing: dict[str, dict[str, Any]] = {}
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("request_id"):
                    existing[str(row["request_id"])] = row
        new_records: list[dict[str, Any]] = []
        for request, output in by_shard.get(shard_index, []):
            completion = output.outputs[0]
            token_ids = getattr(completion, "token_ids", None)
            if token_ids is not None:
                raw = tokenizer.decode(
                    list(token_ids), skip_special_tokens=False
                )
            else:
                raw = str(getattr(completion, "text", "") or "")
            parsed = parse_completion(raw)
            request_id = str(request["request_id"])
            record = {
                "problem_id": str(request["problem_id"]),
                "sample_index": int(request["sample_index"]),
                "request_id": request_id,
                "prompt": str(request.get("prompt") or ""),
                "completion": raw,
                "reasoning": parsed["reasoning"],
                "content": parsed["content"],
                "finish_reason": str(getattr(completion, "finish_reason", "")),
                "thought_closed": bool(parsed["thought_closed"]),
                "tokens": int(
                    len(token_ids)
                    if token_ids is not None
                    else getattr(completion, "completion_tokens", 0)
                ),
                "prompt_tokens": int(request.get("prompt_tokens") or 0),
                "max_new_tokens": int(request.get("max_new_tokens") or 0),
                "engine_cap": int(call_cap),
                "engine": LITE_ENGINE,
                "engine_version": VLLM_VERSION,
                "model_id": MODEL_ID,
                "model_revision": model_revision,
                "run_id": run_id,
                "pool_manifest_sha256": pool_sha,
                "temperature": SAMPLING["temperature"],
                "top_p": SAMPLING["top_p"],
                "top_k": SAMPLING["top_k"],
                "samples_per_prompt": samples_per_prompt,
                "max_model_len": LITE_RECORD_MAX_LENGTH,
                "max_prompt_tokens": int(request.get("prompt_tokens") or 0),
                "skip_special_tokens": False,
                "gpu": gpu,
                "shard_index": shard_index,
                "seed": int(request.get("seed") or 0),
            }
            existing[request_id] = record
            new_records.append(record)
            generated_tokens += int(record["tokens"])

        rows = sorted(
            existing.values(),
            key=lambda row: (
                str(row.get("problem_id")),
                int(row.get("sample_index") or 0),
            ),
        )
        _heartbeat_phase(heartbeat_state, "writing_shard")
        tmp_path = output_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    _json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        _os.replace(tmp_path, output_path)
        commit = bool(shard.get("commit", True))
        if commit:
            committed_marker.write_text(
                _json.dumps(
                    {"shard": shard_index, "committed": True, "engine": LITE_ENGINE}
                )
                + "\n",
                encoding="utf-8",
            )
        _heartbeat_phase(heartbeat_state, "committing_volume")
        try:
            volume.commit()
        except BaseException:
            for path in (output_path, committed_marker):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        results.append(
            {
                "shard": shard_index,
                "skipped": False,
                "committed": commit,
                "samples": len(new_records),
                "rows": len(rows),
                "path": str(output_path),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
    _heartbeat_beat(heartbeat_state, raise_on_failure=True)
    return {"shards": results, "generated_tokens": int(generated_tokens)}


def _modal_base_image_ref() -> str:
    """Pinned CUDA devel image; an optional digest can be supplied via env."""

    if CUDA_IMAGE_DIGEST:
        repository = CUDA_IMAGE_TAG.split(":", 1)[0]
        return f"{repository}@{CUDA_IMAGE_DIGEST}"
    return CUDA_IMAGE_TAG


def _require_tool_version(
    argv: list[str], *, name: str, output_contains: str | None = None
) -> str:
    """Run ``argv`` and fail loudly when the JIT-required tool is unusable."""

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{name} is not installed or not on PATH; flashinfer cannot "
            "JIT-build its sampling kernels"
        ) from exc
    output = f"{result.stdout or ''}{result.stderr or ''}"
    if (
        result.returncode != 0
        or not output.strip()
        or (output_contains is not None and output_contains not in output.lower())
    ):
        raise RuntimeError(
            f"{' '.join(argv)} failed or was not usable: {output!r}"
        )
    return output.strip()


def _verify_jit_toolchain() -> dict[str, str]:
    """Preflight check for the toolchain flashinfer JIT needs at runtime.

    flashinfer's JIT does more than invoke nvcc: it compiles host code with a
    C++ compiler and drives the build with ninja.  All three must exist before
    the launcher asks Modal for a GPU container.
    """

    _require_tool_version(
        ["nvcc", "--version"], name="nvcc", output_contains="release"
    )

    compiler = os.environ.get("CXX") or "g++"
    _require_tool_version([compiler, "--version"], name=compiler)

    ninja_output = _require_tool_version(["ninja", "--version"], name="ninja")
    ninja_version = ninja_output.splitlines()[0]

    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if not Path(cuda_home).is_dir():
        raise RuntimeError(f"CUDA_HOME does not resolve to a directory: {cuda_home}")

    try:
        cpp_ext = importlib.import_module("flashinfer.jit.cpp_ext")
        flashinfer_cuda = str(cpp_ext.get_cuda_path())
    except Exception as exc:
        raise RuntimeError(
            "flashinfer's JIT could not locate CUDA/ nvcc"
        ) from exc
    if not flashinfer_cuda or not Path(flashinfer_cuda).is_dir():
        raise RuntimeError(
            f"flashinfer resolved an unusable CUDA path: {flashinfer_cuda!r}"
        )
    return {
        "cuda_home": cuda_home,
        "flashinfer_cuda": flashinfer_cuda,
        "cxx": compiler,
        "ninja": ninja_version,
    }


def _validate_modal_function_options(
    *,
    gpu: str | None,
    nonpreemptible: bool,
    scaledown_window_s: int,
    timeout_s: int,
) -> None:
    """Fail offline for option combinations Modal 1.6.3 rejects at launch.

    The cached client source validates the autoscaler window
    (``scaledown_window`` must be 2..3600, ``_functions.py:829-830``).  The
    real 2026-11-05 probe additionally showed the server rejects
    ``nonpreemptible`` for GPU workloads with
    ``InvalidError: Non-preemptible is not supported for GPU workloads``.
    Neither modal 1.6.3 nor this launcher may pass nonpreemptible to a GPU
    function.
    """

    if gpu is not None and nonpreemptible:
        raise ValueError(
            "Modal 1.6.3 rejects nonpreemptible=True for GPU workloads "
            "(InvalidError: Non-preemptible is not supported for GPU "
            "workloads); remove nonpreemptible from the GPU function"
        )
    if not 2 <= int(scaledown_window_s) <= 3600:
        raise ValueError(
            "scaledown_window must be between 2 and 3600 (inclusive) for "
            "Modal 1.6.3"
        )
    if int(timeout_s) <= 0:
        raise ValueError("timeout must be a positive integer")


class ModalAppBundle(tuple):
    """``(app, fn)`` tuple that also carries the CPU preflight function.

    Keeping tuple unpacking preserves the existing call sites while callers that
    need the preflight read ``bundle.preflight_fn``.
    """

    def __new__(cls, app: Any, fn: Any, preflight_fn: Any):
        return super().__new__(cls, (app, fn))

    def __init__(self, app: Any, fn: Any, preflight_fn: Any) -> None:
        self.app = app
        self.fn = fn
        self.preflight_fn = preflight_fn


def _build_modal_app(
    modal_module: Any = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    scaledown_window_s: float = DEFAULT_SCALEDOWN_SECONDS,
    cpu: float = DEFAULT_CPU,
    memory_gib: int = DEFAULT_MEMORY_GIB,
):
    m = modal_module or modal
    if m is None:
        return ModalAppBundle(None, None, None)

    gpu_name = MODAL_GPU_IDS.get(DEFAULT_GPU, DEFAULT_GPU)
    _validate_modal_function_options(
        gpu=gpu_name,
        nonpreemptible=False,
        scaledown_window_s=int(scaledown_window_s),
        timeout_s=int(timeout_s),
    )
    _validate_modal_function_options(
        gpu=None,
        nonpreemptible=False,
        scaledown_window_s=int(PREFLIGHT_SCALEDOWN_SECONDS),
        timeout_s=int(PREFLIGHT_TIMEOUT_SECONDS),
    )
    app = m.App("gemmaloop-stage0")
    image = (
        m.Image.from_registry(_modal_base_image_ref(), add_python="3.12")
        # flashinfer's JIT needs a host C++ compiler in addition to nvcc.
        .apt_install("build-essential")
        .pip_install(
            VLLM_VERSION, HF_HUB_VERSION, TRANSFORMERS_VERSION, NINJA_VERSION
        )
        # Modal requires add_local_* to be the LAST image step (a build step
        # after it raises InvalidError at launch).
        .add_local_python_source("gemmaloop")
    )
    # Register the CPU-only preflight first so a fake app that records only the
    # last `function(...)` call still sees the GPU function (legacy tests).
    preflight_fn = app.function(
        image=image,
        cpu=(PREFLIGHT_CPU, PREFLIGHT_CPU),
        memory=(PREFLIGHT_MEMORY_MIB, PREFLIGHT_MEMORY_MIB),
        timeout=int(PREFLIGHT_TIMEOUT_SECONDS),
        startup_timeout=int(startup_timeout_s),
        scaledown_window=int(PREFLIGHT_SCALEDOWN_SECONDS),
        max_containers=MAX_CONTAINERS,
    )(preflight_check)
    fn = app.function(
        image=image,
        gpu=MODAL_GPU_IDS.get(DEFAULT_GPU, DEFAULT_GPU),
        cpu=(float(cpu), float(cpu)),
        memory=(int(memory_gib) * 1024, int(memory_gib) * 1024),
        timeout=int(timeout_s),
        startup_timeout=int(startup_timeout_s),
        scaledown_window=int(scaledown_window_s),
        max_containers=MAX_CONTAINERS,
    )(sample_shard)
    return ModalAppBundle(app, fn, preflight_fn)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def _preflight_cost_usd(seconds: float) -> float:
    """All-in CPU+memory cost of the preflight container, with the safety margin."""

    hours = max(0.0, float(seconds)) / 3600.0
    raw = hours * (
        PREFLIGHT_CPU * CPU_HOURLY_PER_CORE_USD
        + (PREFLIGHT_MEMORY_MIB / 1024.0) * MEMORY_HOURLY_PER_GIB_USD
    )
    return raw * DEFAULT_SAFETY_MARGIN


def _validate_preflight_heartbeat(
    result: Any, heartbeat: Any, key: str
) -> dict[str, Any]:
    """Prove the heartbeat channel with the preflight before any GPU request."""

    info = result.get("heartbeat") if isinstance(result, dict) else None
    if not isinstance(info, dict):
        raise PreflightError("preflight did not report heartbeat delivery")
    try:
        delivered = int(info.get("delivered", 0))
        background = int(info.get("background_delivered", 0))
        failed = int(info.get("failed_writes", 0))
    except (TypeError, ValueError):
        raise PreflightError(f"invalid preflight heartbeat metadata: {info!r}")
    if failed:
        raise PreflightError(
            f"preflight heartbeat channel had {failed} failed write(s)"
        )
    if delivered < REMOTE_HEARTBEAT_MIN_DELIVERED:
        raise PreflightError(
            "preflight delivered only "
            f"{delivered} heartbeat(s); need {REMOTE_HEARTBEAT_MIN_DELIVERED}"
        )
    if background < 1:
        raise PreflightError(
            "the preflight background heartbeat writer delivered no beat; "
            "the GPU writer thread is not proven"
        )
    beat = _read_heartbeat(heartbeat, key)
    if beat is _HEARTBEAT_READ_ERROR or not isinstance(beat, dict):
        raise PreflightError(
            "the launcher could not read back the preflight heartbeat"
        )
    if beat.get("container_id") != info.get("container_id"):
        raise PreflightError(
            "preflight heartbeat container id does not match the result "
            f"({beat.get('container_id')!r} != {info.get('container_id')!r})"
        )
    return info


def _is_missing_volume_path(exc: BaseException) -> bool:
    """True when a Volume listing failure means "this path has no shards yet"."""

    return isinstance(exc, FileNotFoundError) or type(exc).__name__ in {
        "NotFoundError",
        "FileNotFoundError",
    }


def _read_pool(pool_dir: Path) -> list[dict[str, Any]]:
    path = pool_dir / "problems.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _expected_shards(pool_dir: Path, shard_size: int, explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    problems = _read_pool(pool_dir)
    if not problems:
        raise ValueError("cannot infer shards: no problems.jsonl; pass --expected-shards")
    return (len(problems) + int(shard_size) - 1) // int(shard_size)


def _plan(
    pool_dir: Path,
    gpu: str,
    *,
    shard_size: int,
    avg_generated_tokens: int,
    samples_per_prompt: int,
    cpu_cores: int = int(DEFAULT_CPU),
    memory_gib: int = DEFAULT_MEMORY_GIB,
    startup_seconds: float = DEFAULT_STARTUP_SECONDS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    problems = _read_pool(pool_dir)
    if not problems:
        print(
            f"warning: no problems.jsonl at {pool_dir}; planning assumes 5,000 prompts",
            file=sys.stderr,
        )
    prompts = len(problems) if problems else 5_000
    return sampling_plan(
        prompts=prompts,
        gpu=gpu,
        shard_size=shard_size,
        avg_generated_tokens=avg_generated_tokens,
        samples_per_prompt=samples_per_prompt,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        startup_seconds=startup_seconds,
        safety_margin=safety_margin,
    )


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_sha(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_pool_manifest(
    pool_dir: Path,
    *,
    frozen_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Refuse smoke/partial-decontamination pools before a paid run (I6)."""

    # Local-only constant; importing this dependency-free module lazily keeps
    # the remote import closure free of gemmaloop.data/rapidfuzz.
    from gemmaloop.stage0.constants import PART1_CREATIVE_BRIEFS

    manifest_path = pool_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"frozen pool manifest missing: {manifest_path}")
    problems_path = pool_dir / "problems.jsonl"
    if not problems_path.exists():
        raise ValueError(f"problems.jsonl missing: {problems_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = (manifest.get("artifacts") or {}).get("problems.jsonl") or {}
    actual_sha = hashlib.sha256(problems_path.read_bytes()).hexdigest()
    if artifact.get("sha256") != actual_sha:
        raise ValueError("problems.jsonl does not match the frozen manifest hash")
    stored = manifest.get("manifest_sha256")
    if not stored or stored != _manifest_sha(manifest):
        raise ValueError("pool manifest_sha256 is missing or does not recompute")
    if frozen_manifest_sha256 and stored != frozen_manifest_sha256:
        raise ValueError(
            f"pool manifest sha {stored} differs from the frozen {frozen_manifest_sha256}"
        )
    decontam = manifest.get("decontamination") or {}
    if decontam.get("status") == "not_built":
        raise ValueError("pool decontamination was not built")
    targets = decontam.get("targets") or {}
    bad_targets = {
        name: entry
        for name, entry in targets.items()
        if isinstance(entry, dict) and entry.get("status") not in {"ok"}
    }
    if bad_targets:
        raise ValueError(f"pool has incomplete decontamination targets: {bad_targets}")
    creative_briefs = manifest.get("creative_briefs")
    if creative_briefs not in (None, "included", PART1_CREATIVE_BRIEFS):
        raise ValueError(f"unknown creative_briefs mode: {creative_briefs!r}")
    limits = manifest.get("limits") or {}
    smoke_flags = {
        "eval_limit": limits.get("eval_limit"),
        "limit_rows": limits.get("limit_rows"),
        "allow_missing_eval": limits.get("allow_missing_eval"),
        "allow_missing_briefs": limits.get("allow_missing_briefs"),
        "allow_short_sources": limits.get("allow_short_sources"),
        "allow_duplicates": limits.get("allow_duplicates"),
    }
    if any(bool(value) for value in smoke_flags.values()):
        raise ValueError(f"refusing a smoke/partial pool: {smoke_flags}")
    manifest["_pool_manifest_sha256"] = stored
    return manifest


def _receipt_local_shard(output_dir: Path, name: str) -> Path:
    """Locate a receipt's shard in the flat (sample) or shards/ (lite) layout."""

    direct = Path(output_dir) / str(name)
    if direct.exists():
        return direct
    return Path(output_dir) / "shards" / str(name)


def _verify_receipt(output_dir: Path, receipt: Path | None, *, allow_partial: bool):
    path = receipt or (output_dir / RECEIPT_NAME)
    if not path.exists():
        return False, f"missing download receipt {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable download receipt {path}: {exc}"
    expected = int(payload.get("expected_shards") or 0)
    shards = payload.get("shards") or []
    missing = payload.get("missing") or []
    for entry in shards:
        local = _receipt_local_shard(output_dir, entry["name"])
        if not local.exists():
            return False, f"missing local shard {local}"
        if hashlib.sha256(local.read_bytes()).hexdigest() != entry.get("sha256"):
            return False, f"sha256 mismatch for {entry['name']}"
    if missing or payload.get("partial"):
        if not allow_partial:
            return False, f"partial run: {len(missing)} of {expected} shard(s) missing"
    elif expected and len(shards) != expected:
        return False, f"receipt lists {len(shards)} shards, expected {expected}"
    return True, "verified"


def _validate_run_args(*, mode, max_usd, shard_size, avg_generated_tokens, samples_per_prompt, timeout_s, startup_timeout_s, scaledown_window_s, cpu, memory_gib):
    if mode in {
        "plan",
        "dry-run",
        "dry_run",
        "sample",
        "plan-lite",
        "sample-lite",
        "plan-lite-topup",
        "dry-run-lite-topup",
        "plan_lite_topup",
        "sample-lite-topup",
    } and not (0 < float(max_usd) <= HARD_GLOBAL_CAP_USD):
        raise ValueError(f"max_usd must be in (0, {HARD_GLOBAL_CAP_USD:g}]; got {max_usd!r}")
    for name, value in (("shard_size", shard_size), ("avg_generated_tokens", avg_generated_tokens), ("samples_per_prompt", samples_per_prompt)):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(timeout_s) <= 0:
        raise ValueError("timeout must be positive")
    if not 2 <= int(scaledown_window_s) <= 3600:
        raise ValueError(
            "scaledown_window must be between 2 and 3600 (inclusive) for "
            "Modal 1.6.3"
        )
    if not 0 < float(startup_timeout_s) <= MAX_STARTUP_TIMEOUT_SECONDS:
        raise ValueError(
            "startup_timeout must be in (0, "
            f"{MAX_STARTUP_TIMEOUT_SECONDS}]; the guard cannot be disabled"
        )
    if float(cpu) <= 0 or int(memory_gib) <= 0:
        raise ValueError("cpu and memory must be positive")


# ---------------------------------------------------------------------------
# Stage 0-lite on Modal vLLM
# ---------------------------------------------------------------------------
def _load_lite_sampler_script():
    """Load ``scripts/sample_stage0_local.py`` (the registered finalize)."""

    import importlib.util

    path = REPO_ROOT / "scripts" / "sample_stage0_local.py"
    spec = importlib.util.spec_from_file_location(
        "gemmaloop_lite_finalize", path
    )
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def _lite_load_pool(
    pool_dir: Path,
    *,
    require_frozen_pool: bool,
    frozen_manifest_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from gemmaloop.stage0 import local  # noqa: PLC0415 - CPU launcher only

    if not require_frozen_pool:
        problems = _read_pool(Path(pool_dir))
        if not problems:
            raise ValueError(f"no problems.jsonl at {Path(pool_dir)}")
        return problems, {"manifest_sha256": "test", "creative_briefs": None}
    validate_pool_manifest(
        Path(pool_dir), frozen_manifest_sha256=frozen_manifest_sha256
    )
    return local.load_pool(Path(pool_dir))


def _lite_targets(
    prereg_path: Path, *, max_train_tokens: int | None = None
) -> tuple[list[local.Target], str]:
    from gemmaloop.prereg import load_prereg  # noqa: PLC0415
    from gemmaloop.stage0 import local  # noqa: PLC0415

    prereg = load_prereg(Path(prereg_path))
    targets = local.targets_from_prereg(prereg)
    if max_train_tokens is not None:
        targets = [
            local.Target(
                target.name,
                target.low,
                target.high,
                target.problems,
                token_target=(
                    int(max_train_tokens)
                    if target.name == "train"
                    else target.token_target
                ),
                verified=target.verified,
                rows_target=target.rows_target,
                samples_per_prompt=target.samples_per_prompt,
            )
            for target in targets
        ]
    return targets, local.file_sha256(Path(prereg_path))


def _refuse_local_output_dir(output_dir: Path) -> None:
    local_run = (REPO_ROOT / "artifacts" / "stage0" / "local").resolve()
    candidate = Path(output_dir).resolve()
    if candidate == local_run:
        raise ValueError(
            "refusing to write a Modal-lite run into artifacts/stage0/local; "
            "pass a new --output-dir (default artifacts/stage0/modal_lite)"
        )


def _lite_receipt_path(output_dir: Path) -> Path:
    return Path(output_dir) / "modal_receipt.json"


def _lite_download_shard(
    volume: Any,
    *,
    run_dir_rel: str,
    index: int,
    output_dir: Path,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Download one committed shard, verify its sha256, keep a local copy."""

    from gemmaloop.stage0 import local  # noqa: PLC0415
    from gemmaloop.stage0 import modal_lite  # noqa: PLC0415

    name = f"shard_{int(index):05d}.jsonl"
    chunks = volume.read_file(f"{run_dir_rel}/{name}")
    data = bytes(chunks) if isinstance(chunks, (bytes, bytearray)) else b"".join(chunks)
    if not data:
        raise RuntimeError(f"remote shard {name} is empty")
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha != str(expected_sha256):
        raise RuntimeError(
            f"sha256 mismatch for {name}: remote {expected_sha256} != downloaded {sha}"
        )
    shard, committed, _partial = local.shard_paths(Path(output_dir), int(index))
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_name(shard.name + ".download.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, shard)
    local_sha = local.file_sha256(shard)
    if local_sha != sha:
        raise RuntimeError(
            f"sha256 changed while writing {shard}: {sha} != {local_sha}"
        )
    local.write_json_atomic(
        committed,
        {
            "shard": int(index),
            "committed": True,
            "sha256": sha,
            "engine": modal_lite.ENGINE,
        },
    )
    return local.read_jsonl(shard), sha


def _lite_verify_receipt(output_dir: Path, receipt_path: Path) -> tuple[bool, str]:
    if not receipt_path.exists():
        return False, f"missing Modal receipt {receipt_path}"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    base = Path(output_dir) / "shards"
    for entry in payload.get("shards") or []:
        path = base / entry["name"]
        if not path.exists():
            return False, f"missing local shard {path}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            return False, f"sha256 mismatch for {path.name}"
    return True, "verified"


def _lite_eval_train_tokens(
    records: Iterable[dict[str, Any]],
    *,
    problems_by_id: Mapping[str, Mapping[str, Any]],
    targets: Sequence[local.Target],
    monitor_ids: set[str],
    evaluate_filter: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    train_tokens: int,
) -> int:
    from gemmaloop.stage0 import local  # noqa: PLC0415

    for record in records:
        problem = problems_by_id.get(str(record.get("problem_id")))
        if problem is None:
            continue
        target = local.target_for_problem(problem, targets, monitor_ids)
        if target is None or target.name != "train":
            continue
        verdict = evaluate_filter(record, problem)
        if verdict.kept:
            train_tokens += int(verdict.assistant_tokens)
    return train_tokens


def _lite_run_spend_usd(ledger: Any, month: str, run_id: str) -> float:
    """This run's booked spend (actuals + unsettled reservations)."""

    actual = 0.0
    settled: set[str] = set()
    outstanding = 0.0
    for entry in ledger.entries():
        if entry.get("month") != month:
            continue
        metadata = entry.get("metadata") or {}
        if metadata.get("run_id") != run_id:
            continue
        if entry.get("kind") == "actual":
            actual += float(entry.get("usd", 0.0))
            settled.add(str(entry.get("reservation_id")))
        elif entry.get("kind") == "reservation":
            outstanding += float(entry.get("usd", 0.0))
    for entry in ledger.entries():
        if entry.get("month") != month:
            continue
        metadata = entry.get("metadata") or {}
        if metadata.get("run_id") != run_id or entry.get("kind") != "reservation":
            continue
        if str(entry.get("id")) in settled:
            outstanding -= float(entry.get("usd", 0.0))
    return max(0.0, actual + outstanding)


def _lite_cap_allows(
    *,
    run_spend: float,
    month_total: float,
    reservation: float,
    run_cap: float,
    global_cap: float,
) -> bool:
    return (
        float(run_spend) + float(reservation) <= float(run_cap) + 1e-9
        and float(month_total) + float(reservation) <= float(global_cap) + 1e-9
    )


def _lite_default_paths_refusal(
    *,
    ledger_path: Path | None,
    spend_lock_path: Path | None,
) -> str | None:
    """The lite run must use the shared monthly ledger and default spend lock."""

    from gemmaloop.stage0.spend import (  # noqa: PLC0415
        default_ledger_path,
        spend_lock_path as resolve_spend_lock_path,
    )

    if ledger_path is not None:
        if Path(ledger_path).expanduser().resolve() != default_ledger_path().resolve():
            return (
                "sample-lite books every reservation and actual into the default "
                f"monthly ledger ({default_ledger_path()}); --ledger is not allowed"
            )
    if spend_lock_path is not None:
        if (
            Path(spend_lock_path).expanduser().resolve()
            != resolve_spend_lock_path(None).resolve()
        ):
            return (
                "sample-lite must take the shared default spend lock "
                f"({resolve_spend_lock_path(None)}); --spend-lock is not allowed"
            )
    return None


def _run_lite_mode(
    *,
    mode: str,
    pool_dir: Path,
    output_dir: Path,
    prereg_path: Path,
    gpu: str,
    gpu_id: str,
    max_usd: float,
    shard_size: int,
    samples_per_prompt: int,
    seed: int,
    call_size: int,
    yes: bool,
    fn: Any,
    preflight_fn: Any,
    app: Any,
    modal_module: Any,
    ledger_path: Path | None,
    spend_lock_path: Path | None,
    init_ledger: bool,
    month_now: Callable[[], datetime] | None,
    clock: Callable[[], float],
    wall_clock: Callable[[], float],
    cpu: float,
    memory_gib: int,
    startup_timeout_s: float,
    scaledown_window_s: float,
    timeout_s: float,
    image_build_usd: float,
    require_frozen_pool: bool,
    frozen_manifest_sha256: str | None,
    max_train_tokens: int | None,
    accept_train_shortfall: bool,
    allow_math_fallback: bool,
    tokenizer: Any = None,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int] | None = None,
    evaluate_filter: Callable[..., Any] | None = None,
    evaluate_full: Callable[..., Any] | None = None,
) -> int:
    from gemmaloop.stage0 import local  # noqa: PLC0415
    from gemmaloop.stage0 import modal_lite  # noqa: PLC0415

    if int(samples_per_prompt) != int(local.SAMPLES_PER_PROMPT):
        raise ValueError(
            "the Stage 0-lite plan is registered with four samples per prompt"
        )
    if not 1 <= int(call_size) <= int(modal_lite.MAX_TRAIN_CALL_SIZE):
        raise ValueError(
            "lite train call size must be in (1, "
            f"{modal_lite.MAX_TRAIN_CALL_SIZE}]; got {call_size}"
        )
    _refuse_local_output_dir(Path(output_dir))

    problems, pool_manifest = _lite_load_pool(
        Path(pool_dir),
        require_frozen_pool=require_frozen_pool,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    targets, prereg_sha256 = _lite_targets(
        Path(prereg_path), max_train_tokens=max_train_tokens
    )
    pool_sha = str(pool_manifest.get("manifest_sha256"))
    run_id = local.compute_run_id(
        pool_sha,
        shard_size=int(shard_size),
        samples_per_prompt=int(samples_per_prompt),
        model_revision=MODEL_REVISION,
    )

    needs_tokenizer = (
        prompt_tokens_for is None
        or evaluate_filter is None
        or evaluate_full is None
    )
    if needs_tokenizer and tokenizer is None:
        tokenizer = modal_lite.load_tokenizer()
    if prompt_tokens_for is None:
        prompt_tokens_for = modal_lite.make_prompt_tokens_fn(tokenizer)

    plan = modal_lite.build_lite_plan(
        problems,
        targets,
        shard_size=int(shard_size),
        samples_per_prompt=int(samples_per_prompt),
        seed=int(seed),
        prompt_tokens_for=prompt_tokens_for,
        call_size=int(call_size),
        run_id=run_id,
    )
    # Refuse before any Modal contact: vLLM stores a request seed in a signed
    # 64-bit C long, and a full unsigned 64-bit digest overflows EngineCore.
    for call in (*plan.other_calls, *plan.train_calls):
        modal_lite.validate_payload_seeds({"shards": call.as_payload()})
    estimate = modal_lite.estimate_lite_run(plan)
    check_budget(
        estimate, max_usd=max_usd, hard_cap_usd=HARD_GLOBAL_CAP_USD
    )

    cap = min(float(max_usd), HARD_GLOBAL_CAP_USD)
    now = month_now or (lambda: datetime.now(timezone.utc))
    gpu_worst = worst_case_usd(
        gpu,
        timeout_s=timeout_s,
        startup_timeout_s=startup_timeout_s,
        scaledown_window_s=scaledown_window_s,
        cpu_cores=int(cpu),
        memory_gib=memory_gib,
        safety_margin=DEFAULT_SAFETY_MARGIN,
        image_build_usd=image_build_usd,
    )["usd"]
    gpu_only_worst = worst_case_usd(
        gpu,
        timeout_s=timeout_s,
        startup_timeout_s=startup_timeout_s,
        scaledown_window_s=scaledown_window_s,
        cpu_cores=int(cpu),
        memory_gib=memory_gib,
        safety_margin=DEFAULT_SAFETY_MARGIN,
        image_build_usd=0.0,
    )["usd"]
    preflight_worst = _preflight_cost_usd(
        float(startup_timeout_s)
        + float(PREFLIGHT_TIMEOUT_SECONDS)
        + float(REMOTE_CALL_MARGIN_SECONDS)
    )
    # The image build and the CPU preflight are paid once per launch; later
    # calls reserve only the GPU/CPU/memory worst case (M2).
    first_reservation_worst = gpu_worst + preflight_worst
    later_reservation_worst = gpu_only_worst

    if mode in {"plan-lite", "dry-run-lite", "plan_lite"}:
        month_total = 0.0
        try:
            month_total = SpendLedger(None, now_fn=now).cumulative_usd(
                month_key(now()), provider="modal"
            )
        except (OSError, ValueError):
            month_total = 0.0
        reservations = [
            first_reservation_worst if index == 0 else later_reservation_worst
            for index in range(int(estimate["expected_calls"]))
        ]
        print(
            modal_lite.format_plan(
                plan,
                estimate,
                max_usd=max_usd,
                reservations_usd=reservations,
                month_total_usd=month_total,
                hard_cap_usd=HARD_GLOBAL_CAP_USD,
            )
        )
        return 0

    if not yes:
        print(
            "Refusing to launch without --yes: every Modal launch needs the "
            "author's approval.",
            file=sys.stderr,
        )
        return 2
    if fn is None or modal_module is None:
        print(
            "Modal client is not available; use uv run --with modal==1.6.3 "
            "python ...",
            file=sys.stderr,
        )
        return 2
    if preflight_fn is None:
        print(
            "refusing: sample-lite launches require the CPU preflight in the "
            "same launch; no GPU may be requested without it",
            file=sys.stderr,
        )
        return 2
    if not hasattr(modal_module, "Dict"):
        print(
            "refusing: the Modal client must provide Dict for the in-container "
            "heartbeat",
            file=sys.stderr,
        )
        return 2

    from gemmaloop.stage0.grader import MATH_VERIFY_AVAILABLE

    use_math_verify = bool(MATH_VERIFY_AVAILABLE)
    if not use_math_verify and not allow_math_fallback:
        raise SystemExit(
            "math-verify is not installed (run `uv sync`) or pass "
            "--allow-math-fallback for the registered AIME normaliser alone"
        )

    refusal = _lite_default_paths_refusal(
        ledger_path=ledger_path, spend_lock_path=spend_lock_path
    )
    if refusal is not None:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2

    # B1/B2: the shared monthly ledger and the process-lifetime default lock.
    require_spend_lock(None)
    ledger = SpendLedger(None, now_fn=now)
    month = month_key(now())
    month_total = ledger.cumulative_usd(month, provider="modal")
    run_spend = _lite_run_spend_usd(ledger, month, run_id)
    if not _lite_cap_allows(
        run_spend=run_spend,
        month_total=month_total,
        reservation=first_reservation_worst,
        run_cap=cap,
        global_cap=HARD_GLOBAL_CAP_USD,
    ):
        print(
            json.dumps(
                {
                    "mode": "sample-lite",
                    "calls_completed": 0,
                    "stopped_reason": "spending_cap",
                    "run_spent_usd": run_spend,
                    "month_total_usd": month_total,
                    "worst_case_usd": first_reservation_worst,
                    "max_usd": cap,
                    "global_cap_usd": HARD_GLOBAL_CAP_USD,
                    "ledger": str(ledger.path),
                },
                indent=2,
            )
        )
        return 2

    volume = modal_module.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    run_dir_rel = f"{SHARD_DIR}/{run_id}"
    run_dir = f"{VOLUME_MOUNT}/{run_dir_rel}"
    try:
        entries = volume.listdir(run_dir_rel, recursive=False)
    except Exception as exc:
        if _is_missing_volume_path(exc):
            entries = []
        else:
            print(
                "refusing to launch: cannot list Modal Volume "
                f"{VOLUME_NAME!r} ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            return 1
    remote_committed = {
        int(Path(str(getattr(entry, "path", ""))).name.split("_")[1].split(".")[0])
        for entry in entries
        if str(getattr(entry, "path", "")).endswith(".committed")
    }

    output_dir = Path(output_dir)
    local.ensure_run_marker(
        output_dir,
        local.build_run_marker(
            run_id=run_id,
            pool_manifest_sha256=pool_sha,
            targets=targets,
            samples_per_prompt=int(samples_per_prompt),
            shard_size=int(shard_size),
            seed=int(seed),
            prereg_sha256=prereg_sha256,
            engine=modal_lite.ENGINE,
        ),
    )

    problems_by_id = {str(problem["problem_id"]): problem for problem in problems}
    monitor_ids = local.monitor_problem_ids(problems, targets, seed=int(seed))
    train_goal = next(
        (int(target.token_target) for target in targets if target.name == "train"), 0
    )
    if evaluate_filter is None:
        evaluate_filter = _load_lite_sampler_script().make_evaluator(tokenizer)

    run_stats = local.RunStats()
    downloaded: set[int] = set()
    shard_sha: dict[int, str] = {}
    shard_rows: dict[int, int] = {}
    train_tokens = 0
    for index in sorted(remote_committed):
        rows, sha = _lite_download_shard(
            volume,
            run_dir_rel=run_dir_rel,
            index=index,
            output_dir=output_dir,
        )
        downloaded.add(index)
        shard_sha[index] = sha
        shard_rows[index] = len(rows)
        run_stats.shards_committed.append(index)
        run_stats.problems_skipped.update(plan.shard_by_index(index).skipped)
        train_tokens = _lite_eval_train_tokens(
            rows,
            problems_by_id=problems_by_id,
            targets=targets,
            monitor_ids=monitor_ids,
            evaluate_filter=evaluate_filter,
            train_tokens=train_tokens,
        )

    def pending_shards(call: modal_lite.LiteCall) -> list[dict[str, Any]]:
        shards = []
        for payload in call.as_payload():
            if int(payload["shard_index"]) in downloaded:
                continue
            shards.append(payload)
        return shards

    train_calls = [
        call for call in plan.train_calls if pending_shards(call)
    ]
    other_calls = [
        call for call in plan.other_calls if pending_shards(call)
    ]

    stopped_reason: str | None = None
    calls_completed = 0
    remote_calls = 0
    preflight_result = None
    preflight_seconds = 0.0
    launch_overhead_usd = 0.0
    launch_started = wall_clock()
    current_reservation = first_reservation_worst

    def elapsed_s() -> float:
        return max(0.0, wall_clock() - launch_started)

    def failure_booking() -> tuple[float, str]:
        try:
            elapsed = elapsed_s()
            wall = (
                elapsed
                + float(scaledown_window_s)
                + FAILURE_BOOKING_OVERHEAD_SECONDS
            )
            bounded = runtime_cost_usd(
                gpu,
                wall,
                cpu_cores=int(cpu),
                memory_gib=int(memory_gib),
                safety_margin=DEFAULT_SAFETY_MARGIN,
            )["usd"] + float(image_build_usd)
        except Exception:
            return float(current_reservation), "worst_case"
        return (
            min(float(current_reservation), float(bounded)),
            "elapsed_upper_bound",
        )

    def failure_reason(exc: BaseException) -> str:
        if isinstance(exc, PreflightError):
            return "preflight_failure"
        if isinstance(exc, RemoteStartupRetry):
            return "startup_retry"
        if isinstance(exc, RemoteStartupTimeout):
            return "startup_timeout"
        if isinstance(exc, RemoteHeartbeatGap):
            return "heartbeat_gap"
        if isinstance(exc, RemoteCallDeadline):
            return "call_deadline"
        return "error"

    def make_payload(call: modal_lite.LiteCall, index: int) -> dict[str, Any]:
        shards = pending_shards(call)
        return {
            "lite": True,
            "run_id": run_id,
            "run_dir": run_dir,
            "run_dir_rel": run_dir_rel,
            "pool_manifest_sha256": pool_sha,
            "gpu": gpu_id,
            "model_revision": MODEL_REVISION,
            "samples_per_prompt": int(samples_per_prompt),
            "engine_max_model_len": MODEL_MAX_CONTEXT,
            "model_cache_dir": CONTAINER_MODEL_CACHE,
            "call_index": int(index),
            "shards": shards,
        }

    payloads: list[tuple[str, modal_lite.LiteCall, dict[str, Any]]] = []
    # B3/I1: every non-train request is one call and comes first; train calls
    # then follow in plan order.  A cap stop can therefore only leave train
    # data short, never a non-train target.
    for call in other_calls:
        payloads.append(("non_train", call, make_payload(call, len(payloads))))
    for call in train_calls:
        payloads.append(("train", call, make_payload(call, len(payloads))))

    pending = None
    if payloads:
        pending = ledger.reserve(
            shard=payloads[0][2]["shards"][0]["shard_index"],
            usd=first_reservation_worst,
            gpu=gpu,
            timeout_s=timeout_s + startup_timeout_s + scaledown_window_s,
            month=month,
            provider="modal",
            metadata={
                "mode": "sample-lite",
                "max_usd": cap,
                "run_id": run_id,
            },
        )

    try:
        if payloads and pending is not None:
            with ExitStack() as stack:
                heartbeat = stack.enter_context(modal_module.Dict.ephemeral())
                if app is not None:
                    stack.enter_context(app.run())
                preflight_started = wall_clock()
                try:
                    preflight_result = _call_remote(
                        preflight_fn,
                        {
                            "_heartbeat": heartbeat,
                            "heartbeat_key": "preflight",
                            "heartbeat_interval": (
                                PREFLIGHT_HEARTBEAT_INTERVAL_SECONDS
                            ),
                        },
                        wall_clock=wall_clock,
                        startup_timeout_s=startup_timeout_s,
                        timeout_s=PREFLIGHT_TIMEOUT_SECONDS,
                        heartbeat=heartbeat,
                        heartbeat_key="preflight",
                    )
                    if (
                        not isinstance(preflight_result, dict)
                        or preflight_result.get("ok") is not True
                    ):
                        raise PreflightError(
                            "preflight returned an invalid result: "
                            f"{preflight_result!r}"
                        )
                    _validate_preflight_heartbeat(
                        preflight_result, heartbeat, "preflight"
                    )
                    _wait_for_preflight_exit(preflight_fn, wall_clock=wall_clock)
                    preflight_seconds = max(0.0, wall_clock() - preflight_started)
                    launch_overhead_usd = (
                        float(image_build_usd)
                        + _preflight_cost_usd(preflight_seconds)
                        + runtime_cost_usd(
                            gpu,
                            float(scaledown_window_s),
                            cpu_cores=int(cpu),
                            memory_gib=int(memory_gib),
                            safety_margin=DEFAULT_SAFETY_MARGIN,
                        )["usd"]
                    )
                except BaseException as exc:
                    if isinstance(exc, PreflightError):
                        raise
                    raise PreflightError(
                        f"preflight failed: {type(exc).__name__}: {exc}"
                    ) from exc

                target_fn = fn.with_options(
                    gpu=gpu_id,
                    volumes={VOLUME_MOUNT: volume},
                    max_containers=MAX_CONTAINERS,
                    cpu=(float(cpu), float(cpu)),
                    memory=(int(memory_gib) * 1024, int(memory_gib) * 1024),
                    timeout=int(timeout_s),
                )
                for kind, call, payload in payloads:
                    if kind == "train" and train_goal and train_tokens >= train_goal:
                        # Keep sampling every non-train target.
                        continue
                    current_reservation = (
                        first_reservation_worst
                        if remote_calls == 0
                        else later_reservation_worst
                    )
                    if pending is None:
                        run_spend = _lite_run_spend_usd(ledger, month, run_id)
                        month_total = ledger.cumulative_usd(
                            month, provider="modal"
                        )
                        if not _lite_cap_allows(
                            run_spend=run_spend,
                            month_total=month_total,
                            reservation=current_reservation,
                            run_cap=cap,
                            global_cap=HARD_GLOBAL_CAP_USD,
                        ):
                            stopped_reason = "spending_cap"
                            print(
                                json.dumps(
                                    {
                                        "mode": "sample-lite",
                                        "stopped_reason": "spending_cap",
                                        "before_call_index": payload["call_index"],
                                        "before_call_kind": kind,
                                        "run_spent_usd": run_spend,
                                        "month_total_usd": month_total,
                                        "reservation_usd": current_reservation,
                                        "max_usd": cap,
                                        "global_cap_usd": HARD_GLOBAL_CAP_USD,
                                    },
                                    indent=2,
                                ),
                                file=sys.stderr,
                            )
                            break
                        pending = ledger.reserve(
                            shard=payload["shards"][0]["shard_index"],
                            usd=current_reservation,
                            gpu=gpu,
                            timeout_s=timeout_s
                            + startup_timeout_s
                            + scaledown_window_s,
                            month=month,
                            provider="modal",
                            metadata={
                                "mode": "sample-lite",
                                "max_usd": cap,
                                "run_id": run_id,
                            },
                        )
                    reservation_id, pending = pending, None
                    payload["_heartbeat"] = heartbeat
                    payload["heartbeat_key"] = (
                        f"{MODEAL_LITE_CALL_KEY}{payload['call_index']}"
                    )
                    started = clock()
                    try:
                        result = _call_remote(
                            target_fn,
                            payload,
                            wall_clock=wall_clock,
                            startup_timeout_s=startup_timeout_s,
                            timeout_s=timeout_s,
                            heartbeat=heartbeat,
                            heartbeat_key=payload["heartbeat_key"],
                        )
                    except BaseException as exc:
                        booked, basis = failure_booking()
                        ledger.settle(
                            reservation_id,
                            shard=payload["shards"][0]["shard_index"],
                            actual_usd=booked,
                            month=month,
                            provider="modal",
                            metadata={
                                "mode": "sample-lite",
                                "run_id": run_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "elapsed_s": elapsed_s(),
                                "stopped_reason": failure_reason(exc),
                                "booking_basis": basis,
                            },
                        )
                        raise
                    remote_calls += 1
                    actual = runtime_cost_usd(
                        gpu,
                        max(0.0, clock() - started),
                        cpu_cores=int(cpu),
                        memory_gib=int(memory_gib),
                        safety_margin=DEFAULT_SAFETY_MARGIN,
                    )["usd"]
                    launch_charge = 0.0
                    if remote_calls == 1:
                        launch_charge = launch_overhead_usd
                        actual += launch_charge
                    ledger.settle(
                        reservation_id,
                        shard=payload["shards"][0]["shard_index"],
                        actual_usd=actual,
                        month=month,
                        provider="modal",
                        metadata={
                            "mode": "sample-lite",
                            "run_id": run_id,
                            "elapsed_s": elapsed_s(),
                            "booking_basis": "runtime_plus_launch_overhead",
                            "launch_overhead_usd": launch_charge,
                        },
                    )
                    run_spend = _lite_run_spend_usd(ledger, month, run_id)
                    month_total = ledger.cumulative_usd(month, provider="modal")
                    calls_completed += 1

                    for shard_result in result.get("shards", []):
                        index = int(shard_result["shard"])
                        if not shard_result.get("committed", True):
                            continue
                        if index in downloaded:
                            continue
                        rows, sha = _lite_download_shard(
                            volume,
                            run_dir_rel=run_dir_rel,
                            index=index,
                            output_dir=output_dir,
                            expected_sha256=shard_result.get("sha256"),
                        )
                        downloaded.add(index)
                        shard_sha[index] = sha
                        shard_rows[index] = len(rows)
                        run_stats.shards_committed.append(index)
                        run_stats.problems_skipped.update(
                            plan.shard_by_index(index).skipped
                        )
                        train_tokens = _lite_eval_train_tokens(
                            rows,
                            problems_by_id=problems_by_id,
                            targets=targets,
                            monitor_ids=monitor_ids,
                            evaluate_filter=evaluate_filter,
                            train_tokens=train_tokens,
                        )
                    if kind == "train" and train_goal and train_tokens >= train_goal:
                        stopped_reason = "train_goal_reached"
                if pending is not None:
                    # Every remaining call was skipped (e.g. a train-only plan
                    # already at its goal on resume); release the reservation.
                    ledger.settle(
                        pending,
                        shard=-1,
                        actual_usd=0.0,
                        month=month,
                        provider="modal",
                        metadata={
                            "mode": "sample-lite",
                            "run_id": run_id,
                            "cancelled": "no_remote_call",
                        },
                    )
                    pending = None
    except BaseException as exc:
        _stop_app(app)
        if pending is not None:
            booked, basis = failure_booking()
            ledger.settle(
                pending,
                shard=-1,
                actual_usd=booked,
                month=month,
                provider="modal",
                metadata={
                    "mode": "sample-lite",
                    "run_id": run_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": elapsed_s(),
                    "stopped_reason": failure_reason(exc),
                    "booking_basis": basis,
                },
            )
        print(f"launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            json.dumps(
                {"mode": "sample-lite", "stopped_reason": "error"}, indent=2
            )
        )
        return 1
    finally:
        print(_lite_reminder(output_dir), file=sys.stderr)

    if train_goal:
        run_stats.shards_skipped = sorted(
            {
                int(shard.index)
                for shard in plan.shards
                if shard.target_name == "train"
                and int(shard.index) not in downloaded
            }
        )
        if train_tokens >= train_goal and stopped_reason is None:
            stopped_reason = "train_goal_reached"
    else:
        run_stats.shards_skipped = sorted(
            {
                int(shard.index)
                for shard in plan.shards
                if int(shard.index) not in downloaded
            }
        )

    missing_train = [
        f"shard_{shard.index:05d}.jsonl"
        for shard in plan.shards
        if shard.target_name == "train" and shard.index not in downloaded
    ]
    missing_non_train = [
        f"shard_{shard.index:05d}.jsonl"
        for shard in plan.shards
        if shard.target_name != "train" and shard.index not in downloaded
    ]
    missing = sorted(missing_train + missing_non_train)

    # A receipt makes the local shards independently verifiable before the
    # manifests are written.  The Volume cleanup stays a separate command.
    receipt = {
        "kind": "stage0_modal_lite_shards",
        "engine": modal_lite.ENGINE,
        "volume": VOLUME_NAME,
        "run_id": run_id,
        "run_dir": run_dir_rel,
        "expected_shards": len(plan.shards),
        "present_shards": len(downloaded),
        "missing": missing,
        "missing_train": missing_train,
        "missing_non_train": missing_non_train,
        "non_train_complete": not missing_non_train,
        "partial": bool(missing),
        "verified": True,
        "shards": [
            {
                "name": f"shard_{index:05d}.jsonl",
                "path": f"{run_dir_rel}/shard_{index:05d}.jsonl",
                "sha256": shard_sha[index],
                "rows": shard_rows[index],
                "bytes": (output_dir / "shards" / f"shard_{index:05d}.jsonl").stat().st_size,
            }
            for index in sorted(downloaded)
        ],
    }
    receipt_path = _lite_receipt_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(receipt_path, receipt)
    ok, reason = _lite_verify_receipt(output_dir, receipt_path)
    if not ok:
        print(f"refusing to finalize: {reason}", file=sys.stderr)
        return 1

    # B3: non-train targets are never expected to be partial.  Do not write a
    # manifest at all until every non-train shard that has requests is present.
    required_non_train = [
        shard
        for shard in plan.shards
        if shard.target_name != "train" and shard.requests
    ]
    incomplete_non_train = [
        shard.index for shard in required_non_train if shard.index not in downloaded
    ]
    if incomplete_non_train:
        missing_by_target: dict[str, list[int]] = {}
        for shard in required_non_train:
            if shard.index in incomplete_non_train:
                missing_by_target.setdefault(shard.target_name, []).append(
                    int(shard.index)
                )
        print(
            "refusing to finalize: non-train targets are incomplete "
            f"({missing_by_target}); no manifest was written.  Relaunch the "
            "same command to sample the missing shards (committed shards are "
            "never re-sent).",
            file=sys.stderr,
        )
        print(
            json.dumps(
                {
                    "mode": "sample-lite",
                    "stopped_reason": "incomplete_non_train",
                    "run_id": run_id,
                    "missing_non_train_shards": sorted(incomplete_non_train),
                    "missing_by_target": missing_by_target,
                    "stopped_before_finalize": True,
                },
                indent=2,
            )
        )
        return 2

    script = _load_lite_sampler_script()
    if evaluate_filter is None:
        evaluate_filter = script.make_evaluator(tokenizer)
    if evaluate_full is None:
        verify_fn = script.make_verify_fn(
            verify_code=False, grader=None, use_math_verify=use_math_verify
        )

        def count(text: str) -> int:
            return len(tokenizer.encode(text or "", add_special_tokens=False))

        def evaluate_full(record, problem):
            return local.evaluate_sample(
                record,
                problem,
                tokenizer=tokenizer,
                token_counter=count,
                max_length=local.REGISTERED_MAX_LENGTH,
                min_reasoning_share=local.REGISTERED_MIN_REASONING_SHARE,
                verify_fn=verify_fn,
            )

    verifiers = {
        "math": "math-verify" if use_math_verify else "aime-normalise",
        "code": "disabled",
    }
    try:
        stats, manifest = script.finalize(
            args=SimpleNamespace(),
            output_dir=output_dir,
            problems=problems,
            targets=targets,
            pool_dir=Path(pool_dir),
            pool_manifest=pool_manifest,
            problems_path=Path(pool_dir) / "problems.jsonl",
            run_id=run_id,
            model_revision=MODEL_REVISION,
            run_stats=run_stats,
            evaluate_filter=evaluate_filter,
            evaluate_full=evaluate_full,
            verifiers=verifiers,
            hidden_tests=None,
            seed=int(seed),
            engine=modal_lite.ENGINE,
            engine_note=modal_lite.ENGINE_NOTE,
            train_goal=int(train_goal),
            train_shard_order=[
                int(shard.index)
                for shard in plan.shards
                if shard.target_name == "train"
            ],
            max_train_tokens=max_train_tokens,
        )
    except local.RunIdentityError as exc:
        print(f"refusing to resume: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "mode": "sample-lite",
                "engine": modal_lite.ENGINE,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "manifest": str(output_dir / "manifest.json"),
                "records": stats.records,
                "kept": stats.kept,
                "acceptance_rate": stats.acceptance_rate,
                "train_accepted_assistant_tokens": stats.train_tokens,
                "train_goal": train_goal,
                "shards_committed": sorted(downloaded),
                "shards_skipped": run_stats.shards_skipped,
                "calls_completed": calls_completed,
                "remote_calls": remote_calls,
                "stopped_reason": stopped_reason,
                "run_spent_usd": run_spend,
                "month_total_usd": month_total,
                "spent_cumulative_usd": month_total,
                "expected_usd": estimate["usd"],
                "expected_calls": estimate["expected_calls"],
                "max_usd": cap,
                "global_cap_usd": HARD_GLOBAL_CAP_USD,
                "heal": {
                    "rows": (manifest.get("heal") or {}).get("rows"),
                    "excluded_train_records": (
                        manifest.get("heal") or {}
                    ).get("excluded_train_records", 0),
                    "excluded_train_shards": (
                        manifest.get("heal") or {}
                    ).get("excluded_train_shards", []),
                },
                "shortfalls": manifest.get("shortfalls", {}),
            },
            indent=2,
        )
    )
    if train_goal and stats.train_tokens < train_goal:
        print(
            f"train token target not reached: {stats.train_tokens} < {train_goal} "
            "(reported as a shortfall)",
            file=sys.stderr,
        )
        if not accept_train_shortfall:
            print(
                "pass --accept-train-shortfall only if this shortfall is "
                "intended; otherwise relaunch to sample more train shards.",
                file=sys.stderr,
            )
            return 3
    return 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _topup_decision_error(
    plan_path: Path | None,
    count_path: Path | None,
    freeze_path: Path | None,
) -> str | None:
    """Return why the top-up may not launch, or ``None`` when it may.

    The extension may only launch after a completed, manifest-verified official
    count whose summary binds this exact plan file/content and the verifier
    freeze, and whose ``extension_required`` decision is true.
    """

    from gemmaloop.stage0 import shortfall  # noqa: PLC0415

    if plan_path is None:
        return "--topup-plan is required for every top-up mode"
    if count_path is None:
        return (
            "sample-lite-topup requires --topup-count (the completed official "
            "count directory) before any Modal contact or spend"
        )
    if freeze_path is None:
        return (
            "sample-lite-topup requires --topup-freeze (the verifier freeze the "
            "official count must bind) before any Modal contact or spend"
        )
    try:
        plan = shortfall.validate_topup_plan(
            json.loads(Path(plan_path).read_text(encoding="utf-8"))
        )
        shortfall.load_verified_count(
            count_path,
            topup_plan_path=plan_path,
            freeze_path=freeze_path,
            require_extension=True,
        )
        if not plan.get("requests"):
            return "the top-up plan is empty"
    except Exception as exc:  # noqa: BLE001 - every refusal is fail-closed
        return f"{type(exc).__name__}: {exc}"
    return None


def _run_lite_topup_mode(
    *,
    mode: str,
    topup_plan: Path,
    pool_dir: Path,
    topup_count: Path | None = None,
    topup_freeze: Path | None = None,
    output_dir: Path,
    gpu: str,
    gpu_id: str,
    max_usd: float,
    yes: bool,
    fn: Any,
    preflight_fn: Any,
    app: Any,
    modal_module: Any,
    ledger_path: Path | None,
    spend_lock_path: Path | None,
    init_ledger: bool,
    month_now: Callable[[], datetime] | None,
    clock: Callable[[], float],
    wall_clock: Callable[[], float],
    cpu: float,
    memory_gib: int,
    startup_timeout_s: float,
    scaledown_window_s: float,
    timeout_s: float | None,
    image_build_usd: float,
    shard_size: int,
    tokenizer: Any = None,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int] | None = None,
    problems: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Run exactly the precommitted top-up plan in one Modal call.

    Money guards are the same as ``sample-lite``: the shared default monthly
    ledger, the process-wide default spend lock, a process-lifetime cap, one
    container and no retries.  The top-up has its own run id and a $3 cap so
    the immutable lite run's namespace and cap are never mixed; the plan is
    parent-linked to that run.
    """

    from gemmaloop.stage0 import local  # noqa: PLC0415
    from gemmaloop.stage0 import modal_lite  # noqa: PLC0415
    from gemmaloop.stage0 import shortfall  # noqa: PLC0415

    plan_path = Path(topup_plan)
    if not plan_path.is_file():
        print(f"refusing: top-up plan is missing: {plan_path}", file=sys.stderr)
        return 2
    plan = shortfall.validate_topup_plan(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    plan_file_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    request_count = int(plan["request_count"])
    requested_timeout = int(plan["timeout_seconds"])
    if timeout_s is None or int(timeout_s) < requested_timeout:
        timeout_s = requested_timeout
    cap = min(float(max_usd), float(plan["run_cap_usd"]), HARD_GLOBAL_CAP_USD)
    if cap <= 0:
        print("refusing: top-up run cap is not positive", file=sys.stderr)
        return 2
    _refuse_local_output_dir(Path(output_dir))

    if mode in {"plan-lite-topup", "dry-run-lite-topup", "plan_lite_topup"}:
        print(
            json.dumps(
                {
                    "mode": "sample-lite-topup",
                    "plan": str(plan_path.resolve()),
                    "plan_sha256": plan["plan_sha256"],
                    "topup_run_id": plan["topup_run_id"],
                    "parent_run_id": plan["parent_run_id"],
                    "request_count": request_count,
                    "problem_count": plan["problem_count"],
                    "requests_by_split": plan["requests_by_split"],
                    "requests_by_kind": plan["requests_by_kind"],
                    "timeout_seconds": int(timeout_s),
                    "run_cap_usd": cap,
                    "global_cap_usd": HARD_GLOBAL_CAP_USD,
                    "call_shape": "one remote call",
                },
                indent=2,
            )
        )
        return 0

    decision_error = _topup_decision_error(topup_plan, topup_count, topup_freeze)
    if decision_error is not None:
        print(
            f"refusing: the pilot top-up may not launch: {decision_error}",
            file=sys.stderr,
        )
        return 2

    if not yes:
        print(
            "Refusing to launch without --yes: every Modal launch needs the "
            "author's approval.",
            file=sys.stderr,
        )
        return 2
    if fn is None or modal_module is None:
        print(
            "Modal client is not available; use uv run --with modal==1.6.3 "
            "python ...",
            file=sys.stderr,
        )
        return 2
    if preflight_fn is None:
        print(
            "refusing: top-up launches require the CPU preflight in the same "
            "launch",
            file=sys.stderr,
        )
        return 2
    if not hasattr(modal_module, "Dict"):
        print(
            "refusing: the Modal client must provide Dict for the in-container "
            "heartbeat",
            file=sys.stderr,
        )
        return 2

    refusal = _lite_default_paths_refusal(
        ledger_path=ledger_path, spend_lock_path=spend_lock_path
    )
    if refusal is not None:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2

    if problems is None:
        problems, pool_manifest = _lite_load_pool(
            Path(pool_dir), require_frozen_pool=True, frozen_manifest_sha256=None
        )
        expected_pool = plan.get("parent_pool_manifest_sha256")
        if expected_pool and str(pool_manifest.get("manifest_sha256")) != str(
            expected_pool
        ):
            print(
                "refusing: the pool manifest differs from the plan's parent "
                "evidence",
                file=sys.stderr,
            )
            return 2
    problems_by_id = {str(problem["problem_id"]): problem for problem in problems}
    if prompt_tokens_for is None:
        if tokenizer is None:
            tokenizer = modal_lite.load_tokenizer()
        prompt_tokens_for = modal_lite.make_prompt_tokens_fn(tokenizer)

    max_length = modal_lite.local.REGISTERED_MAX_LENGTH
    resolved: list[dict[str, Any]] = []
    for request in plan["requests"]:
        problem = problems_by_id.get(str(request["problem_id"]))
        if problem is None:
            print(
                f"refusing: top-up problem {request['problem_id']} is not in "
                "the frozen pool",
                file=sys.stderr,
            )
            return 2
        prompt_tokens = int(prompt_tokens_for(problem))
        max_new_tokens = int(max_length) - prompt_tokens
        if max_new_tokens <= 0:
            print(
                f"refusing: {request['request_id']} has a non-positive "
                "max_new_tokens",
                file=sys.stderr,
            )
            return 2
        resolved.append(
            {
                "problem_id": str(request["problem_id"]),
                "sample_index": int(request["sample_index"]),
                "request_id": str(request["request_id"]),
                "prompt": str(problem.get("prompt") or ""),
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "seed": int(request["seed"]),
            }
        )
    modal_lite.validate_payload_seeds({"shards": [{"requests": resolved}]})

    shards = [
        {
            "shard_index": index // int(shard_size),
            "commit": True,
            "requests": resolved[index : index + int(shard_size)],
        }
        for index in range(0, len(resolved), int(shard_size))
    ]

    now = month_now or (lambda: datetime.now(timezone.utc))
    require_spend_lock(None)
    ledger = SpendLedger(None, now_fn=now)
    month = month_key(now())
    if not ledger.path.exists() and not init_ledger:
        print(
            f"refusing: spend ledger {ledger.path} is missing",
            file=sys.stderr,
        )
        return 2
    run_id = str(plan["topup_run_id"])
    run_dir_rel = f"{SHARD_DIR}/{run_id}"
    run_dir = f"{VOLUME_MOUNT}/{run_dir_rel}"
    month_total = ledger.cumulative_usd(month, provider="modal")
    run_spend = _lite_run_spend_usd(ledger, month, run_id)
    gpu_worst = worst_case_usd(
        gpu,
        timeout_s=timeout_s,
        startup_timeout_s=startup_timeout_s,
        scaledown_window_s=scaledown_window_s,
        cpu_cores=int(cpu),
        memory_gib=memory_gib,
        safety_margin=DEFAULT_SAFETY_MARGIN,
        image_build_usd=image_build_usd,
    )["usd"]
    reservation = gpu_worst + _preflight_cost_usd(
        float(startup_timeout_s)
        + float(PREFLIGHT_TIMEOUT_SECONDS)
        + float(REMOTE_CALL_MARGIN_SECONDS)
    )
    if not _lite_cap_allows(
        run_spend=run_spend,
        month_total=month_total,
        reservation=reservation,
        run_cap=cap,
        global_cap=HARD_GLOBAL_CAP_USD,
    ):
        print(
            json.dumps(
                {
                    "mode": "sample-lite-topup",
                    "stopped_reason": "spending_cap",
                    "run_spent_usd": run_spend,
                    "month_total_usd": month_total,
                    "reservation_usd": reservation,
                    "max_usd": cap,
                    "global_cap_usd": HARD_GLOBAL_CAP_USD,
                },
                indent=2,
            )
        )
        return 2

    volume = modal_module.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = volume.listdir(run_dir_rel, recursive=False)
    except Exception as exc:
        if _is_missing_volume_path(exc):
            entries = []
        else:
            print(
                "refusing: cannot list Modal Volume "
                f"{VOLUME_NAME!r} ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            return 1
    remote_committed = {
        int(Path(str(getattr(entry, "path", ""))).name.split("_")[1].split(".")[0])
        for entry in entries
        if str(getattr(entry, "path", "")).endswith(".committed")
    }
    downloaded: set[int] = set()
    shard_sha: dict[int, str] = {}
    for index in sorted(remote_committed):
        _rows, sha = _lite_download_shard(
            volume,
            run_dir_rel=run_dir_rel,
            index=index,
            output_dir=output_dir,
        )
        downloaded.add(index)
        shard_sha[index] = sha
    pending_shards = [shard for shard in shards if int(shard["shard_index"]) not in downloaded]

    calls_completed = 0
    run_spend_after = run_spend
    if pending_shards:
        pending = ledger.reserve(
            shard=int(pending_shards[0]["shard_index"]),
            usd=reservation,
            gpu=gpu,
            timeout_s=timeout_s + startup_timeout_s + scaledown_window_s,
            month=month,
            provider="modal",
            metadata={
                "mode": "sample-lite-topup",
                "max_usd": cap,
                "run_id": run_id,
                "parent_run_id": plan["parent_run_id"],
                "plan_sha256": plan["plan_sha256"],
            },
        )
        preflight_seconds = 0.0
        try:
            with ExitStack() as stack:
                heartbeat = stack.enter_context(modal_module.Dict.ephemeral())
                if app is not None:
                    stack.enter_context(app.run())
                preflight_started = wall_clock()
                preflight_result = _call_remote(
                    preflight_fn,
                    {
                        "_heartbeat": heartbeat,
                        "heartbeat_key": "preflight",
                        "heartbeat_interval": PREFLIGHT_HEARTBEAT_INTERVAL_SECONDS,
                    },
                    wall_clock=wall_clock,
                    startup_timeout_s=startup_timeout_s,
                    timeout_s=PREFLIGHT_TIMEOUT_SECONDS,
                    heartbeat=heartbeat,
                    heartbeat_key="preflight",
                )
                if (
                    not isinstance(preflight_result, dict)
                    or preflight_result.get("ok") is not True
                ):
                    raise PreflightError(
                        f"preflight returned an invalid result: {preflight_result!r}"
                    )
                _validate_preflight_heartbeat(
                    preflight_result, heartbeat, "preflight"
                )
                _wait_for_preflight_exit(preflight_fn, wall_clock=wall_clock)
                preflight_seconds = max(0.0, wall_clock() - preflight_started)
                launch_overhead = (
                    float(image_build_usd)
                    + _preflight_cost_usd(preflight_seconds)
                    + runtime_cost_usd(
                        gpu,
                        float(scaledown_window_s),
                        cpu_cores=int(cpu),
                        memory_gib=int(memory_gib),
                        safety_margin=DEFAULT_SAFETY_MARGIN,
                    )["usd"]
                )
                payload = {
                    "lite": True,
                    "topup": True,
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "run_dir_rel": run_dir_rel,
                    "parent_run_id": plan["parent_run_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "pool_manifest_sha256": plan.get(
                        "parent_pool_manifest_sha256"
                    ),
                    "gpu": gpu_id,
                    "model_revision": MODEL_REVISION,
                    "samples_per_prompt": 1,
                    "engine_max_model_len": MODEL_MAX_CONTEXT,
                    "model_cache_dir": CONTAINER_MODEL_CACHE,
                    "call_index": 0,
                    "heartbeat_key": f"{MODEAL_LITE_CALL_KEY}topup",
                    "shards": pending_shards,
                }
                target_fn = fn.with_options(
                    gpu=gpu_id,
                    volumes={VOLUME_MOUNT: volume},
                    max_containers=MAX_CONTAINERS,
                    cpu=(float(cpu), float(cpu)),
                    memory=(int(memory_gib) * 1024, int(memory_gib) * 1024),
                    timeout=int(timeout_s),
                )
                started = clock()
                result = _call_remote(
                    target_fn,
                    payload,
                    wall_clock=wall_clock,
                    startup_timeout_s=startup_timeout_s,
                    timeout_s=timeout_s,
                    heartbeat=heartbeat,
                    heartbeat_key=payload["heartbeat_key"],
                )
                calls_completed = 1
                actual = runtime_cost_usd(
                    gpu,
                    max(0.0, clock() - started),
                    cpu_cores=int(cpu),
                    memory_gib=int(memory_gib),
                    safety_margin=DEFAULT_SAFETY_MARGIN,
                )["usd"] + launch_overhead
                ledger.settle(
                    pending,
                    shard=int(pending_shards[0]["shard_index"]),
                    actual_usd=actual,
                    month=month,
                    provider="modal",
                    metadata={
                        "mode": "sample-lite-topup",
                        "run_id": run_id,
                        "parent_run_id": plan["parent_run_id"],
                        "plan_sha256": plan["plan_sha256"],
                        "elapsed_s": max(0.0, wall_clock() - preflight_started),
                        "booking_basis": "runtime_plus_launch_overhead",
                    },
                )
                pending = None
                run_spend_after = _lite_run_spend_usd(ledger, month, run_id)
                for shard_result in result.get("shards", []):
                    index = int(shard_result["shard"])
                    if not shard_result.get("committed", True):
                        continue
                    if index in downloaded:
                        continue
                    _rows, sha = _lite_download_shard(
                        volume,
                        run_dir_rel=run_dir_rel,
                        index=index,
                        output_dir=output_dir,
                        expected_sha256=shard_result.get("sha256"),
                    )
                    downloaded.add(index)
                    shard_sha[index] = sha
        except BaseException as exc:
            _stop_app(app)
            if pending is not None:
                booked = min(float(reservation), float(reservation))
                ledger.settle(
                    pending,
                    shard=-1,
                    actual_usd=booked,
                    month=month,
                    provider="modal",
                    metadata={
                        "mode": "sample-lite-topup",
                        "run_id": run_id,
                        "parent_run_id": plan["parent_run_id"],
                        "plan_sha256": plan["plan_sha256"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "stopped_reason": "error",
                    },
                )
            print(f"launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    # Verify the downloaded corpus is exactly the plan: every planned request
    # once, with its planned seed, and nothing else.
    records: dict[str, dict[str, Any]] = {}
    for index in sorted(downloaded):
        path = output_dir / "shards" / f"shard_{index:05d}.jsonl"
        for row in local.read_jsonl(path):
            request_id = str(row.get("request_id") or "")
            if request_id in records:
                print(
                    f"refusing: duplicate downloaded request {request_id}",
                    file=sys.stderr,
                )
                return 2
            records[request_id] = row
    expected = {str(request["request_id"]): request for request in plan["requests"]}
    missing = sorted(set(expected) - set(records))
    extra = sorted(set(records) - set(expected))
    if missing or extra:
        print(
            json.dumps(
                {
                    "mode": "sample-lite-topup",
                    "stopped_reason": "incomplete_or_extra",
                    "missing_requests": missing,
                    "extra_requests": extra,
                    "run_id": run_id,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    for request_id, row in records.items():
        request = expected[request_id]
        if int(row.get("seed") or -1) != int(request["seed"]):
            print(f"refusing: seed drift for {request_id}", file=sys.stderr)
            return 2
        if str(row.get("problem_id")) != str(request["problem_id"]):
            print(f"refusing: problem drift for {request_id}", file=sys.stderr)
            return 2
        if int(row.get("sample_index") or -1) != int(request["sample_index"]):
            print(f"refusing: sample-index drift for {request_id}", file=sys.stderr)
            return 2

    manifest = {
        "kind": "stage0_modal_lite_topup_run",
        "version": 1,
        "engine": modal_lite.ENGINE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME_NAME,
        "run_id": run_id,
        "run_dir": run_dir_rel,
        "parent_run_id": plan["parent_run_id"],
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha,
        "request_count": request_count,
        "problem_count": plan["problem_count"],
        "call_shape": "one remote call",
        "timeout_seconds": int(timeout_s),
        "run_cap_usd": cap,
        "calls_completed": calls_completed,
        "run_spent_usd": run_spend_after,
        "month_total_usd": ledger.cumulative_usd(month, provider="modal"),
        "shards": [
            {
                "name": f"shard_{index:05d}.jsonl",
                "sha256": shard_sha[index],
            }
            for index in sorted(downloaded)
        ],
        "requests": [
            {
                "request_id": request_id,
                "problem_id": row["problem_id"],
                "sample_index": int(row["sample_index"]),
                "seed": int(row["seed"]),
                "shard_index": int(row.get("shard_index") or 0),
            }
            for request_id, row in sorted(records.items())
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(output_dir / "topup_manifest.json", manifest)
    _write_json(
        output_dir / "topup_receipt.json",
        {
            "kind": "stage0_modal_lite_topup_receipt",
            "run_id": run_id,
            "parent_run_id": plan["parent_run_id"],
            "plan_sha256": plan["plan_sha256"],
            "request_count": request_count,
            "present": len(records),
            "missing": missing,
            "partial": bool(missing),
            "verified": not bool(missing or extra),
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "mode": "sample-lite-topup",
                "run_id": run_id,
                "output_dir": str(output_dir),
                "requests": len(records),
                "calls_completed": calls_completed,
                "run_spent_usd": run_spend_after,
                "month_total_usd": manifest["month_total_usd"],
                "timeout_seconds": int(timeout_s),
                "run_cap_usd": cap,
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


def run_mode(
    *,
    mode: str,
    pool_dir: Path,
    output_dir: Path,
    gpu: str,
    gpu_id: str,
    max_usd: float,
    shard_size: int,
    avg_generated_tokens: int,
    samples_per_prompt: int,
    yes: bool,
    fn: Any = None,
    preflight_fn: Any = None,
    app: Any = None,
    modal_module: Any = None,
    expected_shards: int | None = None,
    receipt: Path | None = None,
    launch_manifest: Path | None = None,
    allow_partial: bool = False,
    ledger_path: Path | None = None,
    spend_lock_path: Path | None = None,
    init_ledger: bool = False,
    month_now: Callable[[], datetime] | None = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.monotonic,
    cpu: float = DEFAULT_CPU,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    scaledown_window_s: float = DEFAULT_SCALEDOWN_SECONDS,
    timeout_s: float | None = DEFAULT_TIMEOUT_SECONDS,
    image_build_usd: float = DEFAULT_IMAGE_BUILD_USD,
    require_frozen_pool: bool = True,
    frozen_manifest_sha256: str | None = None,
    run_id_override: str | None = None,
    max_shards: int | None = None,
    prereg_path: Path | None = None,
    seed: int = 0,
    lite_call_size: int = DEFAULT_LITE_CALL_SIZE,
    max_train_tokens: int | None = None,
    accept_train_shortfall: bool = False,
    allow_shortfall: bool = False,  # deprecated alias for accept_train_shortfall
    allow_math_fallback: bool = False,
    tokenizer: Any = None,
    prompt_tokens_for: Callable[[Mapping[str, Any]], int] | None = None,
    evaluate_filter: Callable[..., Any] | None = None,
    evaluate_full: Callable[..., Any] | None = None,
    topup_plan: Path | None = None,
    topup_count: Path | None = None,
    topup_freeze: Path | None = None,
) -> int:
    mode = str(mode or "plan")
    _validate_run_args(
        mode=mode,
        max_usd=max_usd,
        shard_size=shard_size,
        avg_generated_tokens=avg_generated_tokens,
        samples_per_prompt=samples_per_prompt,
        timeout_s=timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_SECONDS,
        startup_timeout_s=startup_timeout_s,
        scaledown_window_s=scaledown_window_s,
        cpu=cpu,
        memory_gib=memory_gib,
    )
    cap = min(float(max_usd), HARD_GLOBAL_CAP_USD)

    if mode in {
        "plan-lite-topup",
        "dry-run-lite-topup",
        "plan_lite_topup",
        "sample-lite-topup",
    }:
        if topup_plan is None:
            print(
                "refusing: --topup-plan is required for every top-up mode",
                file=sys.stderr,
            )
            return 2
        return _run_lite_topup_mode(
            mode=mode,
            topup_plan=Path(topup_plan),
            pool_dir=Path(pool_dir),
            topup_count=Path(topup_count) if topup_count is not None else None,
            topup_freeze=Path(topup_freeze) if topup_freeze is not None else None,
            output_dir=Path(output_dir),
            gpu=gpu,
            gpu_id=gpu_id,
            max_usd=max_usd,
            yes=yes,
            fn=fn,
            preflight_fn=preflight_fn,
            app=app,
            modal_module=modal_module,
            ledger_path=ledger_path,
            spend_lock_path=spend_lock_path,
            init_ledger=init_ledger,
            month_now=month_now,
            clock=clock,
            wall_clock=wall_clock,
            cpu=cpu,
            memory_gib=memory_gib,
            startup_timeout_s=startup_timeout_s,
            scaledown_window_s=scaledown_window_s,
            timeout_s=timeout_s,
            image_build_usd=image_build_usd,
            shard_size=int(shard_size),
            tokenizer=tokenizer,
            prompt_tokens_for=prompt_tokens_for,
        )

    if mode in {"plan-lite", "dry-run-lite", "plan_lite", "sample-lite"}:
        return _run_lite_mode(
            mode=mode,
            pool_dir=Path(pool_dir),
            output_dir=Path(output_dir),
            prereg_path=Path(prereg_path or _default_prereg_path()),
            gpu=gpu,
            gpu_id=gpu_id,
            max_usd=max_usd,
            shard_size=shard_size,
            samples_per_prompt=samples_per_prompt,
            seed=int(seed),
            call_size=int(lite_call_size),
            yes=yes,
            fn=fn,
            preflight_fn=preflight_fn,
            app=app,
            modal_module=modal_module,
            ledger_path=ledger_path,
            spend_lock_path=spend_lock_path,
            init_ledger=init_ledger,
            month_now=month_now,
            clock=clock,
            wall_clock=wall_clock,
            cpu=cpu,
            memory_gib=memory_gib,
            startup_timeout_s=startup_timeout_s,
            scaledown_window_s=scaledown_window_s,
            timeout_s=timeout_s,
            image_build_usd=image_build_usd,
            require_frozen_pool=require_frozen_pool,
            frozen_manifest_sha256=frozen_manifest_sha256,
            max_train_tokens=max_train_tokens,
            accept_train_shortfall=bool(
                accept_train_shortfall or allow_shortfall
            ),
            allow_math_fallback=allow_math_fallback,
            tokenizer=tokenizer,
            prompt_tokens_for=prompt_tokens_for,
            evaluate_filter=evaluate_filter,
            evaluate_full=evaluate_full,
        )

    if mode in {"plan", "dry-run", "dry_run"}:
        plan = _plan(pool_dir, gpu, shard_size=shard_size, avg_generated_tokens=avg_generated_tokens, samples_per_prompt=samples_per_prompt, cpu_cores=int(cpu), memory_gib=memory_gib)
        check_budget(plan["chosen"], max_usd=max_usd, hard_cap_usd=HARD_GLOBAL_CAP_USD)
        worst = worst_case_usd(gpu, timeout_s=timeout_s, startup_timeout_s=startup_timeout_s, scaledown_window_s=scaledown_window_s, cpu_cores=int(cpu), memory_gib=memory_gib, safety_margin=DEFAULT_SAFETY_MARGIN, image_build_usd=image_build_usd)
        print(format_plan(plan))
        print(
            f"\nchosen GPU: {gpu} | all-in plan ${plan['chosen']['usd']:.2f} | "
            f"worst case/call ${worst['usd']:.2f} | cap ${max_usd:.2f}"
        )
        return 0

    if mode == "sample":
        problems = _read_pool(pool_dir)
        plan = _plan(pool_dir, gpu, shard_size=shard_size, avg_generated_tokens=avg_generated_tokens, samples_per_prompt=samples_per_prompt, cpu_cores=int(cpu), memory_gib=memory_gib)
        check_budget(plan["chosen"], max_usd=max_usd, hard_cap_usd=HARD_GLOBAL_CAP_USD)
        if not problems:
            print(f"no problems found at {pool_dir / 'problems.jsonl'}", file=sys.stderr)
            return 2
        try:
            if not yes:
                print("Refusing to launch without --yes: every Modal launch needs the author's approval.", file=sys.stderr)
                return 2
            if fn is None or modal_module is None:
                print("Modal client is not available; use uv run --with modal==1.6.3 python ...", file=sys.stderr)
                return 2
            if preflight_fn is None:
                print(
                    "refusing: sample launches require the CPU preflight in the "
                    "same launch; no GPU may be requested without it",
                    file=sys.stderr,
                )
                return 2
            if not hasattr(modal_module, "Dict"):
                print(
                    "refusing: the Modal client must provide Dict for the "
                    "in-container heartbeat",
                    file=sys.stderr,
                )
                return 2

            pool_manifest = validate_pool_manifest(pool_dir, frozen_manifest_sha256=frozen_manifest_sha256) if require_frozen_pool else {"_pool_manifest_sha256": "test"}
            pool_sha = str(pool_manifest["_pool_manifest_sha256"])
            run_id = compute_run_id(pool_sha, shard_size=shard_size, samples_per_prompt=samples_per_prompt)
            now = month_now or (lambda: datetime.now(timezone.utc))
            require_spend_lock(spend_lock_path)
            ledger = SpendLedger(ledger_path, now_fn=now)
            if not ledger.path.exists() and not init_ledger:
                print(f"refusing: spend ledger {ledger.path} is missing; pass --init-ledger for the first run", file=sys.stderr)
                return 2
            month = month_key(now())
            cumulative = ledger.cumulative_usd(month, provider="modal")
            gpu_worst = worst_case_usd(gpu, timeout_s=timeout_s, startup_timeout_s=startup_timeout_s, scaledown_window_s=scaledown_window_s, cpu_cores=int(cpu), memory_gib=memory_gib, safety_margin=DEFAULT_SAFETY_MARGIN, image_build_usd=image_build_usd)["usd"]
            preflight_worst = _preflight_cost_usd(
                float(startup_timeout_s)
                + float(PREFLIGHT_TIMEOUT_SECONDS)
                + float(REMOTE_CALL_MARGIN_SECONDS)
            )
            # The reservation covers the whole launch: GPU + image + preflight CPU.
            worst = gpu_worst + preflight_worst
            if cumulative + worst > cap:
                print(json.dumps({"shards_completed": 0, "stopped_reason": "spending_cap", "spent_cumulative_usd": cumulative, "worst_case_usd": worst, "max_usd": cap}, indent=2))
                return 0
            volume = modal_module.Volume.from_name(VOLUME_NAME, create_if_missing=True)
            run_dir_rel = f"{SHARD_DIR}/{run_id}"
            run_dir = f"{VOLUME_MOUNT}/{run_dir_rel}"
            try:
                entries = volume.listdir(run_dir_rel, recursive=False)
            except Exception as exc:
                if _is_missing_volume_path(exc):
                    entries = []  # a new run has no directory yet
                else:
                    print(
                        "refusing to launch: cannot list Modal Volume "
                        f"{VOLUME_NAME!r} ({type(exc).__name__}: {exc}); refusing to "
                        "re-launch already-finished shards",
                        file=sys.stderr,
                    )
                    return 1
            done = {
                int(Path(str(getattr(entry, "path", ""))).name.split("_")[1].split(".")[0])
                for entry in entries
                if str(getattr(entry, "path", "")).endswith(".committed")
            }
            payloads = []
            for start_index in range(0, len(problems), shard_size):
                index = start_index // shard_size
                if index in done:
                    continue
                payloads.append({
                    "shard_index": index,
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "pool_manifest_sha256": pool_sha,
                    "problems": [{"problem_id": p["problem_id"], "prompt": p["prompt"], "kind": p.get("kind"), "source": p.get("source")} for p in problems[start_index:start_index + shard_size]],
                    "samples_per_prompt": samples_per_prompt,
                    "gpu": gpu_id,
                    "engine_max_model_len": MODEL_MAX_CONTEXT,
                    "model_cache_dir": CONTAINER_MODEL_CACHE,
                })
            if max_shards is not None:
                payloads = payloads[: int(max_shards)]
            total_shards = (len(problems) + int(shard_size) - 1) // int(shard_size)
            manifest_path = launch_manifest or (output_dir / LAUNCH_MANIFEST_NAME)
            _write_json(manifest_path, {"volume": VOLUME_NAME, "run_id": run_id, "run_dir": run_dir_rel, "pool_manifest_sha256": pool_sha, "planned_shards": total_shards, "already_committed": len(done), "remaining_shards": len(payloads), "shard_size": shard_size, "samples_per_prompt": samples_per_prompt, "gpu": gpu, "timeout_s": int(timeout_s), "startup_timeout_s": int(startup_timeout_s), "scaledown_window_s": int(scaledown_window_s), "worst_case_usd": worst, "max_usd": cap, "created": datetime.now(timezone.utc).isoformat()})
            target_fn = fn.with_options(gpu=gpu_id, volumes={VOLUME_MOUNT: volume}, max_containers=MAX_CONTAINERS, cpu=(float(cpu), float(cpu)), memory=(int(memory_gib) * 1024, int(memory_gib) * 1024), timeout=int(timeout_s))
            results = []
            stopped_reason = None
            remote_calls = 0
            pending = None
            preflight_result = None
            preflight_seconds = 0.0
            launch_overhead_usd = 0.0
            launch_started = wall_clock()

            def elapsed_s() -> float:
                return max(0.0, wall_clock() - launch_started)

            def failure_booking() -> tuple[float, str]:
                """Bounded elapsed booking; full worst case if elapsed is unknown."""

                try:
                    elapsed = elapsed_s()
                    wall = (
                        elapsed
                        + float(scaledown_window_s)
                        + FAILURE_BOOKING_OVERHEAD_SECONDS
                    )
                    bounded = runtime_cost_usd(
                        gpu,
                        wall,
                        cpu_cores=int(cpu),
                        memory_gib=memory_gib,
                        safety_margin=DEFAULT_SAFETY_MARGIN,
                    )["usd"] + float(image_build_usd)
                except Exception:
                    return float(worst), "worst_case"
                return min(float(worst), float(bounded)), "elapsed_upper_bound"

            def failure_reason(exc: BaseException) -> str:
                if isinstance(exc, PreflightError):
                    return "preflight_failure"
                if isinstance(exc, RemoteStartupRetry):
                    return "startup_retry"
                if isinstance(exc, RemoteStartupTimeout):
                    return "startup_timeout"
                if isinstance(exc, RemoteHeartbeatGap):
                    return "heartbeat_gap"
                if isinstance(exc, RemoteCallDeadline):
                    return "call_deadline"
                return "error"

            if payloads:
                if cumulative + worst > cap:
                    stopped_reason = "spending_cap"
                else:
                    # Reservation exists locally before any remote contact.
                    pending = ledger.reserve(
                        shard=payloads[0]["shard_index"],
                        usd=worst,
                        gpu=gpu,
                        timeout_s=timeout_s + startup_timeout_s + scaledown_window_s,
                        month=month,
                        provider="modal",
                        metadata={"max_usd": cap},
                    )
            try:
                if payloads and pending is not None:
                    with ExitStack() as stack:
                        heartbeat = stack.enter_context(
                            modal_module.Dict.ephemeral()
                        )
                        if app is not None:
                            stack.enter_context(app.run())
                        for payload in payloads:
                            payload["_heartbeat"] = heartbeat
                            payload["heartbeat_key"] = (
                                f"shard:{payload['shard_index']}"
                            )

                        preflight_started = wall_clock()
                        try:
                            preflight_result = _call_remote(
                                preflight_fn,
                                {
                                    "_heartbeat": heartbeat,
                                    "heartbeat_key": "preflight",
                                    "heartbeat_interval": (
                                        PREFLIGHT_HEARTBEAT_INTERVAL_SECONDS
                                    ),
                                },
                                wall_clock=wall_clock,
                                startup_timeout_s=startup_timeout_s,
                                timeout_s=PREFLIGHT_TIMEOUT_SECONDS,
                                heartbeat=heartbeat,
                                heartbeat_key="preflight",
                            )
                            if (
                                not isinstance(preflight_result, dict)
                                or preflight_result.get("ok") is not True
                            ):
                                raise PreflightError(
                                    "preflight returned an invalid result: "
                                    f"{preflight_result!r}"
                                )
                            _validate_preflight_heartbeat(
                                preflight_result, heartbeat, "preflight"
                            )
                            _wait_for_preflight_exit(
                                preflight_fn, wall_clock=wall_clock
                            )
                            preflight_seconds = max(
                                0.0, wall_clock() - preflight_started
                            )
                            launch_overhead_usd = (
                                float(image_build_usd)
                                + _preflight_cost_usd(preflight_seconds)
                                + runtime_cost_usd(
                                    gpu,
                                    float(scaledown_window_s),
                                    cpu_cores=int(cpu),
                                    memory_gib=memory_gib,
                                    safety_margin=DEFAULT_SAFETY_MARGIN,
                                )["usd"]
                            )
                        except BaseException as exc:
                            if isinstance(exc, PreflightError):
                                raise
                            raise PreflightError(
                                f"preflight failed: {type(exc).__name__}: {exc}"
                            ) from exc

                        for payload in payloads:
                            if pending is None:
                                if cumulative + worst > cap:
                                    stopped_reason = "spending_cap"
                                    break
                                pending = ledger.reserve(
                                    shard=payload["shard_index"],
                                    usd=worst,
                                    gpu=gpu,
                                    timeout_s=timeout_s + startup_timeout_s + scaledown_window_s,
                                    month=month,
                                    provider="modal",
                                    metadata={"max_usd": cap},
                                )
                            reservation_id, pending = pending, None
                            started = clock()
                            try:
                                result = _call_remote(
                                    target_fn,
                                    payload,
                                    wall_clock=wall_clock,
                                    startup_timeout_s=startup_timeout_s,
                                    timeout_s=timeout_s,
                                    heartbeat=heartbeat,
                                    heartbeat_key=payload["heartbeat_key"],
                                )
                            except BaseException as exc:
                                booked, basis = failure_booking()
                                ledger.settle(
                                    reservation_id,
                                    shard=payload["shard_index"],
                                    actual_usd=booked,
                                    month=month,
                                    provider="modal",
                                    metadata={
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "elapsed_s": elapsed_s(),
                                        "stopped_reason": failure_reason(exc),
                                        "booking_basis": basis,
                                    },
                                )
                                raise
                            remote_calls += 1
                            actual = runtime_cost_usd(
                                gpu,
                                max(0.0, clock() - started),
                                cpu_cores=int(cpu),
                                memory_gib=memory_gib,
                                safety_margin=DEFAULT_SAFETY_MARGIN,
                            )["usd"]
                            launch_charge = 0.0
                            if remote_calls == 1:
                                launch_charge = launch_overhead_usd
                                actual += launch_charge
                            ledger.settle(
                                reservation_id,
                                shard=payload["shard_index"],
                                actual_usd=actual,
                                month=month,
                                provider="modal",
                                metadata={
                                    "elapsed_s": elapsed_s(),
                                    "booking_basis": "runtime_plus_launch_overhead",
                                    "launch_overhead_usd": launch_charge,
                                },
                            )
                            cumulative = ledger.cumulative_usd(month, provider="modal")
                            results.append(result)
            except BaseException as exc:
                _stop_app(app)
                if pending is not None:
                    booked, basis = failure_booking()
                    ledger.settle(
                        pending,
                        shard=-1,
                        actual_usd=booked,
                        month=month,
                        provider="modal",
                        metadata={
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_s": elapsed_s(),
                            "stopped_reason": failure_reason(exc),
                            "booking_basis": basis,
                        },
                    )
                print(f"launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                print(json.dumps({"results": results, "stopped_reason": "error"}, indent=2))
                return 1
            manifest = _read_json(manifest_path) or {}
            manifest.update({"shards_completed": len(results), "stopped_reason": stopped_reason, "spent_cumulative_usd": cumulative if 'cumulative' in locals() else 0.0, "remote_calls": remote_calls, "preflight": preflight_result, "elapsed_s": elapsed_s()})
            _write_json(manifest_path, manifest)
            print(json.dumps({"shards_completed": len(results), "shards_planned": len(payloads), "remote_calls": remote_calls, "spent_cumulative_usd": cumulative if 'cumulative' in locals() else 0.0, "worst_case_usd": worst, "max_usd": cap, "stopped_reason": stopped_reason, "run_id": run_id, "results": results}, indent=2))
            return 0
        finally:
            print(_reminder(output_dir), file=sys.stderr)

    if mode == "download":
        if modal_module is None:
            print("Modal client is not available; use uv run --with modal==... python ...", file=sys.stderr)
            return 2
        manifest = _read_json(launch_manifest or (output_dir / LAUNCH_MANIFEST_NAME)) or {}
        run_id = manifest.get("run_id") or run_id_override
        if not run_id:
            print(f"missing launch manifest/run_id; pass --run-id. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
            return 2
        run_dir_rel = manifest.get("run_dir") or f"{SHARD_DIR}/{run_id}"
        expected = int(manifest.get("planned_shards") or expected_shards or _expected_shards(pool_dir, shard_size))
        try:
            volume = modal_module.Volume.from_name(VOLUME_NAME, create_if_missing=False)
            entries = volume.listdir(run_dir_rel, recursive=False)
        except Exception as exc:
            print(f"cannot list Modal Volume: {type(exc).__name__}: {exc}. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
            return 1
        names = {Path(str(getattr(entry, "path", ""))).name for entry in entries}
        present = sorted(int(n.split("_")[1].split(".")[0]) for n in names if re.fullmatch(r"shard_\d{5}\.jsonl", n))
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_entries = []
        for index in present:
            name = f"shard_{index:05d}.jsonl"
            if f"{name}.committed" not in names:
                print(f"shard {name} has no .committed marker. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                return 1
            chunks = volume.read_file(f"{run_dir_rel}/{name}")
            data = bytes(chunks) if isinstance(chunks, (bytes, bytearray)) else b"".join(chunks)
            if not data:
                print(f"shard {name} is empty. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                return 1
            (output_dir / name).write_bytes(data)
            shard_entries.append({"name": name, "path": f"{run_dir_rel}/{name}", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        missing = [f"shard_{i:05d}.jsonl" for i in range(expected) if f"shard_{i:05d}.jsonl" not in names]
        receipt_path = receipt or (output_dir / RECEIPT_NAME)
        _write_receipt(receipt_path, {"volume": VOLUME_NAME, "run_id": run_id, "run_dir": run_dir_rel, "expected_shards": expected, "present_shards": len(shard_entries), "missing": missing, "partial": bool(missing), "shards": shard_entries, "verified": True})
        print(f"downloaded and verified {len(shard_entries)}/{expected} shard(s) to {output_dir}{' (partial)' if missing else ''}")
        return 0

    if mode == "cleanup":
        if modal_module is None:
            print("Modal client is not available; use uv run --with modal==... python ...", file=sys.stderr)
            return 2
        from gemmaloop.stage0.spend import (  # noqa: PLC0415
            spend_lock_path as resolve_spend_lock_path,
        )

        if spend_lock_path is not None and (
            Path(spend_lock_path).expanduser().resolve()
            != resolve_spend_lock_path(None).resolve()
        ):
            print(
                "refusing: cleanup must use the shared default spend lock "
                f"({resolve_spend_lock_path(None)}); --spend-lock is not allowed",
                file=sys.stderr,
            )
            return 2
        require_spend_lock(None)
        try:
            volume = modal_module.Volume.from_name(VOLUME_NAME, create_if_missing=False)
            entries = volume.listdir(SHARD_DIR, recursive=True)
            volume_shards = {
                str(getattr(entry, "path", "")): getattr(entry, "size", None)
                for entry in entries
                if str(getattr(entry, "path", "")).endswith(".jsonl")
            }
        except Exception as exc:
            if type(exc).__name__ == "NotFoundError":
                volume_shards = {}
            else:
                print(f"refusing to delete Volume {VOLUME_NAME!r}: listing failed ({type(exc).__name__}: {exc}). Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                return 2
        if volume_shards:
            ok, reason = _verify_receipt(output_dir, receipt, allow_partial=allow_partial)
            if not ok:
                print(f"refusing to delete Volume {VOLUME_NAME!r}: {reason}. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                return 2
            receipt_payload = json.loads((receipt or (output_dir / RECEIPT_NAME)).read_text(encoding="utf-8"))
            receipt_paths = {entry["path"]: entry for entry in receipt_payload.get("shards", [])}
            for path, size in volume_shards.items():
                local = receipt_paths.get(path)
                if local is None:
                    print(f"refusing to delete Volume {VOLUME_NAME!r}: {path} is not in the local receipt. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                    return 2
                local_name = Path(path).name
                local_file = _receipt_local_shard(output_dir, local_name)
                if size is not None and local_file.stat().st_size != size:
                    print(f"refusing to delete Volume {VOLUME_NAME!r}: {path} size differs from the downloaded copy. Manual fallback: {DELETE_COMMAND}", file=sys.stderr)
                    return 2
        try:
            modal_module.Volume.objects.delete(VOLUME_NAME, allow_missing=True)
        except Exception as exc:
            print(f"volume delete failed ({type(exc).__name__}: {exc}); manual: {DELETE_COMMAND}", file=sys.stderr)
            return 1
        print(f"deleted Modal volume {VOLUME_NAME!r}")
        return 0

    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "plan",
            "sample",
            "sample-lite",
            "sample-lite-topup",
            "download",
            "cleanup",
        ),
        default="plan",
    )
    parser.add_argument("--pool-dir", type=Path, default=Path("artifacts/stage0/pool"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--gpu", default=DEFAULT_GPU, choices=sorted(MODAL_GPU_IDS))
    parser.add_argument("--max-usd", type=float, default=12.0)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--avg-generated-tokens", type=int, default=3200)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lite-call-size",
        type=int,
        default=DEFAULT_LITE_CALL_SIZE,
        help=(
            "max train requests per Modal-lite call (hard ceiling 1400); all "
            "non-train requests are always one call"
        ),
    )
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument(
        "--accept-train-shortfall",
        action="store_true",
        help="exit 0 even when the train token target is not reached",
    )
    parser.add_argument("--allow-math-fallback", action="store_true")
    parser.add_argument("--expected-shards", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--topup-plan",
        type=Path,
        default=None,
        help="precommitted pilot top-up plan (required by sample-lite-topup)",
    )
    parser.add_argument(
        "--topup-count",
        type=Path,
        default=None,
        help=(
            "completed official count directory (summary.json + "
            "count_manifest.json); sample-lite-topup refuses without it and "
            "unless extension_required is true"
        ),
    )
    parser.add_argument(
        "--topup-freeze",
        type=Path,
        default=None,
        help="verifier freeze the official count must bind (defaults to none)",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--launch-manifest", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--spend-lock", type=Path, default=None)
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--frozen-manifest-sha", default=os.environ.get("GEMMALOOP_STAGE0_FROZEN_MANIFEST_SHA"))
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument("--scaledown-window", type=int, default=DEFAULT_SCALEDOWN_SECONDS)
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU)
    parser.add_argument("--memory-gib", type=int, default=DEFAULT_MEMORY_GIB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def _default_prereg_path() -> Path:
    from gemmaloop.prereg import default_prereg_path  # noqa: PLC0415

    return Path(default_prereg_path())


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, int]:
    from gemmaloop.stage0 import local  # noqa: PLC0415

    topup = args.mode == "sample-lite-topup"
    lite = args.mode == "sample-lite"
    output_dir = args.output_dir
    if output_dir is None:
        if topup:
            output_dir = Path("artifacts/stage0/modal_lite_topup")
        else:
            output_dir = Path(
                "artifacts/stage0/modal_lite" if lite else "artifacts/stage0/samples"
            )
    shard_size = args.shard_size
    if shard_size is None:
        shard_size = local.DEFAULT_SHARD_SIZE if (lite or topup) else DEFAULT_SHARD_SIZE
    return Path(output_dir), int(shard_size)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lite = args.mode == "sample-lite"
    topup = args.mode == "sample-lite-topup"
    if args.timeout_seconds is None and not topup:
        # The measured probe generated 2,000 requests in one 1,802 s call.
        args.timeout_seconds = 1800 if lite else DEFAULT_TIMEOUT_SECONDS
    output_dir, shard_size = _resolve_paths(args)
    if topup:
        mode = "plan-lite-topup" if args.dry_run else args.mode
    else:
        mode = ("plan-lite" if lite else "plan") if args.dry_run else args.mode
    if mode == "sample-lite-topup":
        # Refuse before importing the Modal client at all.
        decision_error = _topup_decision_error(
            args.topup_plan, args.topup_count, args.topup_freeze
        )
        if decision_error is not None:
            print(
                f"refusing: the pilot top-up may not launch: {decision_error}",
                file=sys.stderr,
            )
            return 2
    if mode in {"plan", "plan-lite", "plan-lite-topup"}:
        return run_mode(
            mode=mode,
            pool_dir=args.pool_dir,
            output_dir=output_dir,
            gpu=args.gpu,
            gpu_id=MODAL_GPU_IDS.get(args.gpu, args.gpu),
            max_usd=args.max_usd,
            shard_size=shard_size,
            avg_generated_tokens=args.avg_generated_tokens,
            samples_per_prompt=args.samples_per_prompt,
            yes=args.yes,
            timeout_s=args.timeout_seconds,
            startup_timeout_s=args.startup_timeout,
            scaledown_window_s=args.scaledown_window,
            cpu=args.cpu,
            memory_gib=args.memory_gib,
            require_frozen_pool=False if not (lite or topup) else True,
            prereg_path=args.prereg,
            seed=args.seed,
            lite_call_size=args.lite_call_size,
            max_train_tokens=args.max_train_tokens,
            accept_train_shortfall=args.accept_train_shortfall,
            allow_math_fallback=args.allow_math_fallback,
            topup_plan=args.topup_plan,
            topup_count=args.topup_count,
            topup_freeze=args.topup_freeze,
        )
    m = _import_modal()
    built = _build_modal_app(m, timeout_s=args.timeout_seconds, startup_timeout_s=args.startup_timeout, scaledown_window_s=args.scaledown_window, cpu=args.cpu, memory_gib=args.memory_gib)
    app, fn = built
    preflight_fn = built.preflight_fn
    return run_mode(
        mode=mode,
        pool_dir=args.pool_dir,
        output_dir=output_dir,
        gpu=args.gpu,
        gpu_id=MODAL_GPU_IDS.get(args.gpu, args.gpu),
        max_usd=args.max_usd,
        shard_size=shard_size,
        avg_generated_tokens=args.avg_generated_tokens,
        samples_per_prompt=args.samples_per_prompt,
        yes=args.yes,
        fn=fn,
        preflight_fn=preflight_fn,
        app=app,
        modal_module=m,
        expected_shards=args.expected_shards,
        receipt=args.receipt,
        launch_manifest=args.launch_manifest,
        allow_partial=args.allow_partial,
        ledger_path=args.ledger,
        spend_lock_path=args.spend_lock,
        init_ledger=args.init_ledger,
        timeout_s=args.timeout_seconds,
        startup_timeout_s=args.startup_timeout,
        scaledown_window_s=args.scaledown_window,
        cpu=args.cpu,
        memory_gib=args.memory_gib,
        frozen_manifest_sha256=args.frozen_manifest_sha,
        run_id_override=args.run_id,
        max_shards=args.max_shards,
        prereg_path=args.prereg,
        seed=args.seed,
        lite_call_size=args.lite_call_size,
        max_train_tokens=args.max_train_tokens,
        accept_train_shortfall=args.accept_train_shortfall,
        allow_math_fallback=args.allow_math_fallback,
        topup_plan=args.topup_plan,
        topup_count=args.topup_count,
        topup_freeze=args.topup_freeze,
    )


if __name__ == "__main__":
    raise SystemExit(main())
