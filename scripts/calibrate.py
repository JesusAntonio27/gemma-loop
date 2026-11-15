"""Measure generation/training throughput and answer budgets on this GPU.

Task 6 produces ``artifacts/calibration/calibration.json``, the input of the
preregistration (Task 10).  It measures, on the local RTX 5070:

* generation throughput per length bucket (2k/4k/8k/16k/32k) at the eval's
  adaptive batch and ``--gen-new-tokens`` measured tokens for E2B at
  N = 1, 2, 4 and E4B at N = 1, 2.  Every bucket has its own excluded warm-up,
  resets the peak counter, is marked invalid and retried at half batch when it
  spills past 95 % of the card, and proves the previous bucket's memory was
  returned to the model baseline;
* LoRA training throughput per arm (O-H N=1, L N=2) for every candidate
  sequence length (512, 1024, 1536, 2048).  Each length is timed step by step;
  a length is marked as **spilled** when peak reserved VRAM exceeds 95 % of the
  card or any step collapses below half that length's median throughput.  Only
  the clean steps of the largest non-spilling length feed ``train_tok_per_s``;
* the Phase-2 maximum lengths probed by descending through
  4096, 3072, 2048 and 1024 at N=4 (worst case) and N=2 (primary N), where
  spill counts as failure exactly like an OOM.  ``phase2_max_length_n4`` and
  ``phase2_max_length_n2`` record the largest clean length for each N;
  ``phase2_max_length`` is kept as an alias of the N=4 value;
* completion tokens and ``finish_reason`` for ``--items-per-bench`` prompts of
  every requested benchmark (text benchmarks by default, vision at 1120 image
  soft tokens for MMMU-Pro/MATH-Vision and 280 for spatial benches).

Benchmark completions are checkpointed to one JSONL file per benchmark after
every generated batch, using the same fingerprint/partial-file pattern as
``gemmaloop.eval.runner``.  A crash loses at most the in-flight batch; resume
skips completed items.  The calibration JSON carries a global fingerprint and
refuses to mix settings unless ``--fresh`` is passed.

Heavy artifacts never enter the repository.  Progress is printed to stdout.

Reference: ``references/gemma-loop/04`` Know-how section 6 and
``references/gemma-loop/07`` Know-how section 3.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from gemmaloop import paths
from gemmaloop.budget import (
    DEFAULT_SAMPLES_PER_BENCH,
    SPILL_COLLAPSE_FRACTION,
    SPILL_RESERVED_FRACTION,
    spill_detected,
)
from gemmaloop.eval.runner import (
    PartialProvenanceError,
    _open_partial,
    _write_records,
    run_fingerprint,
)
from gemmaloop.model.load import (
    MODEL_IDS,
    ModelSpec,
    language_lora_target_modules,
    load_model,
)
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.train.perf import cap_cuda_allocator
from gemmaloop.serve.generate import Sampling, generate_batch, max_batch_for
from gemmaloop.serve.refill import generate_batch_refill

DEFAULT_SIZES = "e2b,e4b"
DEFAULT_BENCHES = "lcb,aime,gpqa,mmlupro"
# Task 6 calibrates one MMMU-Pro key (the adapter's Standard config) and one
# SpatialEval key (VQA).  The prereg's separate MMMU-Pro Vision and SpatialEval
# TQA variants are out of Task 6 scope and reuse these per-item cost estimates.
ALL_BENCHES = (
    "lcb",
    "aime",
    "gpqa",
    "mmlupro",
    "mmmupro",
    "mathvision",
    "spatialeval",
    "cvbench",
)
GEN_LOOPS = {"e2b": (1, 2, 4), "e4b": (1, 2)}
# Generation throughput is measured per expected completion length, at the
# adaptive batch the evaluation itself will use.  Each bucket gets its own
# warm-up (excluded), allocator reset, spill check and baseline memory check;
# records carry a measurement tag so ``--sections gen`` can re-measure them into
# an existing calibration without rotating the bench/train results.
GEN_LENGTH_BUCKETS = (2048, 4096, 8192, 16384, 32768)
GEN_MEASURE_TOKENS = 512
# The configured value is the per-sequence decode window; each bucket must still
# time at least this many tokens in aggregate (noise floor), so a small batch is
# given a longer window instead of a noisy one-token-per-row rate.
GEN_MIN_MEASURED_TOKENS = 1024
# Records written before the isolated per-bucket measurement are re-measured by
# ``--sections gen`` without rotating the whole calibration file: this tag lives
# in the record (not in the fingerprint), so bench/train measurements survive.
GEN_MEASUREMENT_VERSION = "isolated-v2"
# ``memory_allocated`` may stay a little above the post-load baseline (workspace,
# allocator bookkeeping); anything larger means a previous bucket leaked.
GEN_BASELINE_TOLERANCE_BYTES = 512 * 2**20
# Generation batch is chosen from the per-size non-spilling ceilings; 0 in
# ``--gen-batch``/``--bench-batch`` means "use max_batch_for(...)".
TRAIN_LENGTHS = (512, 1024, 1536, 2048)
BENCH_IMAGE_SOFT_TOKENS = {
    "mmmupro": 1120,
    "mathvision": 1120,
    "spatialeval": 280,
    "cvbench": 280,
}
BENCH_PARTIAL_SUFFIX = ".partial.jsonl"
CALIBRATION_TAG = "calibrate-v1"
GEN_PROMPT = (
    "Write a detailed, step-by-step explanation of how a modern computer works, "
    "covering the CPU, memory hierarchy, storage and the operating system. "
    "Use at least 800 words and do not stop early."
)


def generation_batch(
    args: argparse.Namespace, size: str = "e2b", length: int | None = None
) -> int:
    """Batch for one generation-throughput bucket (0 in --gen-batch = auto)."""
    bucket = int(length if length is not None else args.gen_lengths[-1])
    return int(args.gen_batch) or max_batch_for(bucket, size)


def generation_batches(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """Per-size, per-length adaptive batch map (recorded in the fingerprint)."""
    return {
        size: {
            str(bucket): generation_batch(args, size, bucket)
            for bucket in args.gen_lengths
        }
        for size in sorted(args.sizes)
    }


def benchmark_batch(args: argparse.Namespace) -> int:
    """Batch for the completion-length section (0 = auto)."""
    # The completion-length section always loads E2B bf16.
    return int(args.bench_batch) or max_batch_for(int(args.max_new_tokens), "e2b")


def registered_sample_map(benches: list[str]) -> dict[str, int]:
    """Registered generations per item for the requested benchmarks."""
    return {
        name: int(DEFAULT_SAMPLES_PER_BENCH.get(name, 1))
        for name in benches
    }


# ---------------------------------------------------------------------------
# Resource tracking and small utilities
# ---------------------------------------------------------------------------


# Resource high-water marks sampled during the run.  ``torch.cuda`` high-water
# statistics can exceed dedicated VRAM on Windows (WDDM spills into shared
# system memory), so the report uses ``total - free`` from ``mem_get_info``.
_PEAK_VRAM_USED_GIB = 0.0
_MIN_AVAILABLE_RAM_GIB = float("inf")
_PEAK_PROCESS_RAM_GIB = 0.0


def log(message: str) -> None:
    sample_resources()
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sample_resources() -> None:
    global _PEAK_VRAM_USED_GIB, _MIN_AVAILABLE_RAM_GIB, _PEAK_PROCESS_RAM_GIB

    _MIN_AVAILABLE_RAM_GIB = min(_MIN_AVAILABLE_RAM_GIB, available_ram_gib())
    _PEAK_PROCESS_RAM_GIB = max(_PEAK_PROCESS_RAM_GIB, process_ram_gib())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        _PEAK_VRAM_USED_GIB = max(_PEAK_VRAM_USED_GIB, (total - free) / 2**30)


def percentile(values: list[int] | list[float], fraction: float) -> float:
    """Linear-interpolated percentile (same convention as numpy)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def available_ram_gib() -> float:
    """System RAM available right now, in GiB (``inf`` if it cannot be read)."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:  # pragma: no cover - psutil is normally installed
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 2**30
        return float("inf")


def process_ram_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:  # pragma: no cover
        return 0.0


def _gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _total_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_properties(0).total_memory)


def _versions() -> dict[str, str]:
    versions = {"torch": str(torch.__version__)}
    for name in ("transformers", "peft"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - dependencies are pinned
            versions[name] = "unknown"
    return versions


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: cannot read {path} ({exc}); starting from scratch")
        return {}
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_resources()
    data.setdefault("meta", {})
    data["meta"]["schema_version"] = 2
    data["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["meta"]["gpu"] = _gpu_name()
    data["meta"]["total_vram_gib"] = round(_total_vram_bytes() / 2**30, 3)
    if math.isfinite(_MIN_AVAILABLE_RAM_GIB):
        data["meta"]["min_available_ram_gib"] = round(_MIN_AVAILABLE_RAM_GIB, 2)
    else:  # strict JSON: never emit Infinity
        data["meta"]["min_available_ram_gib"] = -1.0
    data["meta"]["peak_process_ram_gib"] = round(_PEAK_PROCESS_RAM_GIB, 2)
    if torch.cuda.is_available():
        data["meta"]["peak_vram_used_gib"] = round(_PEAK_VRAM_USED_GIB, 2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def free_model() -> None:
    """Collect garbage and return cached VRAM to the driver.

    The caller must drop its own references first (``del model, ...``): deleting
    a local inside this helper does not remove the caller's binding.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(
            f"freed model; ram={available_ram_gib():.1f} GiB, "
            f"vram_free={torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB"
        )
    else:
        log(f"freed model; ram={available_ram_gib():.1f} GiB")


def _record_skip(data: dict[str, Any], key: str, reason: str) -> None:
    data.setdefault("skipped", {})[key] = {
        "reason": reason,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log(f"SKIP {key}: {reason}")


def _clear_benchmark_caches(module: Any) -> None:
    """Drop the caller-side copies of large benchmark releases (e.g. LCB)."""
    for name in ("_download_release", "_release_question_ids"):
        function = getattr(module, name, None)
        cache_clear = getattr(function, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def _benchmark_module(name: str):
    if name not in ALL_BENCHES:
        raise ValueError(f"unknown benchmark {name!r}; expected one of {ALL_BENCHES}")
    return importlib.import_module(f"gemmaloop.eval.benchmarks.{name}")


def _text_config_vocab(model: Any) -> int:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return int(getattr(text_config, "vocab_size"))


def bench_partial_path(output: Path, bench: str) -> Path:
    """Partial JSONL file for one benchmark, next to the calibration JSON."""
    return output.with_name(f"{output.stem}.{bench}{BENCH_PARTIAL_SUFFIX}")


# ---------------------------------------------------------------------------
# Configuration fingerprint
# ---------------------------------------------------------------------------


def _fingerprint_diff(old: dict, new: dict) -> list[str]:
    lines = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            lines.append(f"  {key}: partial={old.get(key)!r} current={new.get(key)!r}")
    return lines


def build_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """Describe every measurement setting that affects the output values."""
    sizes = sorted(args.sizes)
    return {
        "format_version": 2,
        "tag": CALIBRATION_TAG,
        "sizes": sizes,
        "quant": {size: ("nf4" if size == "e4b" else "bf16") for size in sizes},
        "gen": {
            "loops": {size: list(GEN_LOOPS[size]) for size in sizes},
            "lengths": sorted(int(length) for length in args.gen_lengths),
            # Configured per-sequence floor, kept for fingerprint compatibility.
            # Each bucket record carries the effective measured window (it is
            # lengthened for small batches to time >= GEN_MIN_MEASURED_TOKENS).
            "measured_tokens": int(args.gen_new_tokens),
            "batches": generation_batches(args),
            "seed": int(args.seed),
        },
        "train": {
            "lengths": sorted(int(length) for length in args.train_lengths),
            "steps": int(args.train_steps),
            "probe_steps": int(args.probe_steps),
            "probe_candidates": sorted(int(length) for length in args.probe_candidates),
            "probe_n_loops": [2, 4],
            "seed": int(args.seed),
        },
        "bench": {
            "benches": sorted(args.benches),
            "items_per_bench": int(args.items_per_bench),
            "max_new_tokens": int(args.max_new_tokens),
            "engine": str(args.bench_engine),
            "seed": int(args.seed),
            "samples": registered_sample_map(sorted(args.benches)),
            "image_soft_tokens": {
                name: int(BENCH_IMAGE_SOFT_TOKENS.get(name, 280))
                for name in sorted(args.benches)
            },
        },
        "gpu": {"name": _gpu_name(), "total_vram_bytes": _total_vram_bytes()},
        "versions": _versions(),
    }


def ensure_compatible(
    data: dict[str, Any], fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to append to a calibration produced with different settings.

    The caller decides whether ``--fresh`` was passed; this helper only checks
    the fingerprint of an existing file.
    """
    if not data:
        return data
    old = data.get("fingerprint")
    if old is None:
        raise PartialProvenanceError(
            f"existing calibration has no fingerprint (created by an older "
            f"version of scripts/calibrate.py); re-run with --fresh to rotate it"
        )
    if old != fingerprint:
        details = "\n".join(_fingerprint_diff(old, fingerprint))
        raise PartialProvenanceError(
            "calibration file belongs to a different configuration:\n"
            f"{details}\n"
            "Refusing to mix measurements. Re-run with --fresh to rotate the "
            "old calibration and its partial files."
        )
    return data


# ---------------------------------------------------------------------------
# Length summaries
# ---------------------------------------------------------------------------


def summarize_steps(
    length: int,
    step_times_s: list[float],
    peak_reserved_bytes: int,
    total_bytes: int,
    n_loops: int,
    *,
    oom: bool = False,
) -> dict[str, Any]:
    """Summarise one length's per-step timings without averaging spilled steps.

    ``tok_s`` is the mean of the clean steps (those not below half the median).
    For an OOM the measurement is incomplete, so ``tok_s`` is ``None`` even if
    some steps finished.
    """
    rates = [
        float(length) / elapsed
        for elapsed in step_times_s
        if elapsed > 0
    ]
    median = statistics.median(rates) if rates else None
    clean = (
        [rate for rate in rates if rate >= SPILL_COLLAPSE_FRACTION * median]
        if median is not None and median > 0
        else []
    )
    diagnostic_tok_s = statistics.fmean(clean) if clean else None
    if oom:
        spilled = True
    else:
        spilled = spill_detected(rates, peak_reserved_bytes, total_bytes)
    # ``tok_s`` is only usable when the length is clean.  ``diagnostic_tok_s``
    # keeps the clean-step mean for a spilled length so the probe table still
    # records a throughput number without inviting its use as a budget rate.
    clean_tok_s = None if spilled else diagnostic_tok_s
    return {
        "n_loops": int(n_loops),
        "steps": len(rates),
        "step_tok_s": [round(rate, 3) for rate in rates],
        "tok_s": round(clean_tok_s, 3) if clean_tok_s is not None else None,
        "diagnostic_tok_s": (
            round(diagnostic_tok_s, 3) if diagnostic_tok_s is not None else None
        ),
        "peak_reserved_gib": round(peak_reserved_bytes / 2**30, 3),
        "total_vram_gib": round(total_bytes / 2**30, 3),
        "spill": bool(spilled),
        "oom": bool(oom),
    }


def probe_descending(
    candidates: list[int] | tuple[int, ...],
    existing: dict[str, dict[str, Any]],
    measure: Callable[[int], dict[str, Any]],
) -> tuple[int | None, dict[str, dict[str, Any]]]:
    """Probe candidates from largest to smallest and return the first clean one.

    ``existing`` is mutated with every probed length (so a resumed run reuses
    previous results).  ``measure(length)`` returns a :func:`summarize_steps`
    record.  A length is clean when it neither spilled nor OOMed and has a
    ``tok_s``.  The scan stops at the first clean length, which is the largest
    feasible one; every length actually probed stays recorded with its peak
    reserved memory and throughput.
    """
    ordered = sorted({int(length) for length in candidates}, reverse=True)
    if not ordered:
        raise ValueError("candidates must not be empty")
    chosen: int | None = None
    for length in ordered:
        key = str(length)
        if key not in existing:
            existing[key] = measure(length)
        result = existing[key]
        clean = (
            not result.get("spill")
            and not result.get("oom")
            and (result.get("tok_s") or 0) > 0
        )
        if clean:
            chosen = length
            break
    return chosen, existing


def choose_common_length(
    probes_by_arm: dict[str, dict[str, dict[str, Any]]], arms: list[str]
) -> tuple[int | None, dict[str, int | None]]:
    """Return the largest length that is clean for every arm.

    ``probes_by_arm[arm][str(length)]`` is a :func:`summarize_steps` record.  A
    length is usable for an arm only when it did not spill/OOM and has a clean
    ``tok_s``.
    """
    per_arm: dict[str, int | None] = {}
    for arm in arms:
        usable = [
            int(length)
            for length, result in probes_by_arm.get(arm, {}).items()
            if not result.get("spill")
            and not result.get("oom")
            and (result.get("tok_s") or 0) > 0
        ]
        per_arm[arm] = max(usable) if usable else None
    if any(value is None for value in per_arm.values()):
        return None, per_arm
    common = min(value for value in per_arm.values() if value is not None)
    return common, per_arm


# ---------------------------------------------------------------------------
# Generation throughput
# ---------------------------------------------------------------------------


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory_allocated() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def _release_generation_memory(
    baseline_allocated: int | None, *, label: str = "generation"
) -> bool:
    """Free a bucket's cache/tensors and verify the model baseline is back.

    ``gc.collect()`` drops Python references, ``empty_cache()`` returns the
    allocator's cached blocks to the driver, and ``memory_allocated`` proves that
    no bucket tensor (KV cache, PLE gather, outputs) is still live.  Returns
    ``False`` (and logs) when the allocated total stays above the baseline.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return True
    torch.cuda.empty_cache()
    if baseline_allocated is None:
        return True
    tolerance = max(
        GEN_BASELINE_TOLERANCE_BYTES, int(0.02 * _total_vram_bytes())
    )
    allocated = _memory_allocated()
    if allocated <= baseline_allocated + tolerance:
        return True
    # One more pass in case a Python frame was still holding a tensor when the
    # first collect ran (e.g. a generator's exception traceback).
    gc.collect()
    torch.cuda.empty_cache()
    allocated = _memory_allocated()
    if allocated <= baseline_allocated + tolerance:
        return True
    log(
        f"WARNING: {label} retained {allocated / 2**30:.2f} GiB allocated "
        f"(baseline {baseline_allocated / 2**30:.2f} GiB) after cleanup"
    )
    return False


def _generation_measure_tokens(batch_size: int, configured: int) -> int:
    """Per-sequence window that keeps a bucket at >= 1,024 timed tokens."""
    per_row = math.ceil(GEN_MIN_MEASURED_TOKENS / max(1, int(batch_size)))
    return max(int(configured), int(per_row))


def _measure_generation_once(
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    batch_size: int,
    length_bucket: int,
    measure_tokens: int,
) -> dict[str, Any]:
    """One warmed, isolated generation window for a bucket/attempt.

    The warm-up uses the same batch and cache size as the timed window but is
    discarded, and the allocator is emptied before the peak counter is reset, so
    the timed window never inherits the warm-up's fragmented reserved pool.
    """
    conversation = [{"role": "user", "content": GEN_PROMPT}]
    conversations = [conversation for _ in range(batch_size)]
    total_bytes = _total_vram_bytes()
    previous_min = getattr(model.generation_config, "min_new_tokens", 0)
    outputs: list[Any] = []
    peak_bytes = 0
    elapsed = 0.0
    oom = False
    try:
        model.generation_config.min_new_tokens = min(8, measure_tokens)
        warm = generate_batch(
            model,
            processor,
            conversations,
            Sampling(max_new_tokens=min(8, measure_tokens)),
            seed=args.seed,
            max_batch=batch_size,
            cache_new_tokens=length_bucket,
        )
        del warm
        _sync_cuda()
        # Isolate the timed window from the warm-up's allocator state.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.generation_config.min_new_tokens = measure_tokens
        started = time.perf_counter()
        outputs = generate_batch(
            model,
            processor,
            conversations,
            Sampling(max_new_tokens=measure_tokens),
            seed=args.seed,
            max_batch=batch_size,
            cache_new_tokens=length_bucket,
        )
        _sync_cuda()
        elapsed = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError:
        oom = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        model.generation_config.min_new_tokens = previous_min
        if torch.cuda.is_available():
            peak_bytes = int(torch.cuda.max_memory_reserved())

    tokens = sum(int(output.completion_tokens) for output in outputs)
    rate = tokens / elapsed if tokens > 0 and elapsed > 0 else None
    if rate is None and not oom:
        raise RuntimeError("generation throughput measurement produced no tokens")
    if not oom and tokens < batch_size * measure_tokens:
        log(
            f"WARNING: only {tokens} tokens generated "
            f"(expected {batch_size * measure_tokens})"
        )
    sample_resources()
    rss_gib = process_ram_gib()
    spill = bool(
        total_bytes > 0
        and peak_bytes > SPILL_RESERVED_FRACTION * total_bytes
    )
    del outputs
    return {
        "batch": int(batch_size),
        "measured_tokens": int(measure_tokens),
        "tokens": int(tokens),
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(rate, 3) if rate is not None else None,
        "peak_reserved_gib": round(peak_bytes / 2**30, 3),
        "total_vram_gib": round(total_bytes / 2**30, 3),
        "process_ram_gib": round(rss_gib, 3),
        "spill": bool(spill),
        "oom": bool(oom),
    }


def _measure_generation(
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    batch_size: int,
    length_bucket: int,
    baseline_allocated: int | None = None,
) -> dict[str, Any]:
    """Measure one length bucket, retrying at half batch when it spills.

    A bucket whose ``max_memory_reserved`` exceeds 95 % of the card is invalid
    (WDDM has already spilled into shared system RAM, which is what made the
    original rates non-monotonic).  It is retried once at half batch; both
    attempts are recorded and the retry's rate is used only when it is clean.
    """
    configured = int(args.gen_new_tokens)
    attempts: list[dict[str, Any]] = []
    current_batch = max(1, int(batch_size))
    chosen: dict[str, Any] | None = None
    while True:
        measure_tokens = _generation_measure_tokens(current_batch, configured)
        attempt = _measure_generation_once(
            model,
            processor,
            args,
            current_batch,
            length_bucket,
            measure_tokens,
        )
        attempts.append(attempt)
        attempt["memory_verified"] = bool(
            _release_generation_memory(
                baseline_allocated,
                label=f"gen n*_len{length_bucket} batch={current_batch}",
            )
        )
        clean = (
            not attempt["spill"]
            and not attempt["oom"]
            and (attempt["tok_s"] or 0) > 0
            and int(attempt["tokens"]) >= GEN_MIN_MEASURED_TOKENS
        )
        if clean:
            chosen = attempt
            break
        if current_batch <= 1 or len(attempts) >= 2:
            chosen = attempt
            break
        current_batch = max(1, current_batch // 2)
        log(
            f"gen: len{length_bucket} spilled at batch "
            f"{attempt['batch']} ({attempt['peak_reserved_gib']:.2f} GiB); "
            f"retrying at batch {current_batch}"
        )

    assert chosen is not None
    invalid = bool(
        chosen["spill"]
        or chosen["oom"]
        or not (chosen["tok_s"] or 0) > 0
        or int(chosen["tokens"]) < GEN_MIN_MEASURED_TOKENS
    )
    record = {
        "batch": int(chosen["batch"]),
        # A spilled attempt's rate is not a budget rate; keep it in ``attempts``
        # as a diagnostic but never expose it as the bucket's throughput.
        "tok_s": None if invalid else chosen["tok_s"],
        "measured_tokens": int(chosen["measured_tokens"]),
        "measured_tokens_total": int(chosen["tokens"]),
        "cache_new_tokens": int(length_bucket),
        "peak_reserved_gib": float(chosen["peak_reserved_gib"]),
        "total_vram_gib": float(chosen["total_vram_gib"]),
        "process_ram_gib": float(chosen["process_ram_gib"]),
        "spill": bool(chosen["spill"]),
        "oom": bool(chosen["oom"]),
        "invalid": invalid,
        "memory_verified": bool(chosen.get("memory_verified", True)),
        "measurement": GEN_MEASUREMENT_VERSION,
        "attempts": attempts,
    }
    return record


def _generation_record_is_current(record: Any) -> bool:
    """Whether a stored bucket was produced by the isolated measurement."""
    return (
        isinstance(record, dict)
        and record.get("measurement") == GEN_MEASUREMENT_VERSION
        and not record.get("invalid", False)
        # A bucket whose memory did not return to baseline is usable now but
        # must be re-measured on the next ``--sections gen`` run.
        and record.get("memory_verified", True) is not False
    )


def measure_generation_section(
    args: argparse.Namespace, data: dict[str, Any], output: Path
) -> None:
    rates = data.setdefault("gen_tok_per_s", {})
    for size in args.sizes:
        size_rates = rates.setdefault(size, {})
        keys = {
            (loop, bucket): f"n{loop}_len{bucket}"
            for loop in GEN_LOOPS[size]
            for bucket in args.gen_lengths
        }
        stale = [
            key
            for key in keys.values()
            if not _generation_record_is_current(size_rates.get(key))
        ]
        if not stale:
            log(f"gen[{size}]: already measured")
            continue
        replaced = sum(key in size_rates for key in stale)
        if replaced:
            log(
                f"gen[{size}]: re-measuring {replaced} bucket(s) without an "
                f"isolated-measurement record"
            )

        if size == "e4b":
            free_ram = available_ram_gib()
            if free_ram < args.min_e4b_ram_gib:
                reason = "skipped: low RAM"
                log(
                    f"gen[e4b]: {free_ram:.1f} GiB free < "
                    f"{args.min_e4b_ram_gib:.0f} GiB; {reason}"
                )
                _record_skip(data, "e4b", reason)
                save_json(output, data)
                continue
            quant = "nf4"
        else:
            quant = "bf16"

        log(f"gen[{size}]: loading {size} {quant} (N=1)")
        started = time.perf_counter()
        model = processor = plan = text_model = None
        baseline_allocated: int | None = None
        try:
            model, processor, plan = load_model(
                ModelSpec(
                    size=size,
                    quant=quant,
                    loop=LoopConfig(n_loops=1),
                )
            )
            text_model = model.model.language_model
            log(f"gen[{size}]: model ready in {time.perf_counter() - started:.1f}s")
            _release_generation_memory(None, label=f"gen[{size}] model load")
            baseline_allocated = _memory_allocated()
            log(
                f"gen[{size}]: model baseline "
                f"{baseline_allocated / 2**30:.2f} GiB allocated"
            )
            for loop in GEN_LOOPS[size]:
                set_n_loops(text_model, loop)
                for bucket in args.gen_lengths:
                    key = keys[(loop, bucket)]
                    if _generation_record_is_current(size_rates.get(key)):
                        log(f"gen[{size}]: {key} already present")
                        continue
                    batch_size = generation_batch(args, size, bucket)
                    log(
                        f"gen[{size}]: measuring {key} "
                        f"(batch={batch_size}, cache={bucket}) ..."
                    )
                    record = _measure_generation(
                        model,
                        processor,
                        args,
                        batch_size,
                        bucket,
                        baseline_allocated=baseline_allocated,
                    )
                    size_rates[key] = record
                    data.get("skipped", {}).pop(size, None)
                    if record.get("invalid"):
                        log(
                            f"gen[{size}]: {key} INVALID "
                            f"(peak={record['peak_reserved_gib']:.2f} GiB, "
                            f"spill={record['spill']}, oom={record['oom']})"
                        )
                    else:
                        log(
                            f"gen[{size}]: {key} = {record['tok_s']:.1f} tok/s "
                            f"(batch={record['batch']}, "
                            f"peak={record['peak_reserved_gib']:.2f} GiB, "
                            f"rss={record['process_ram_gib']:.2f} GiB)"
                        )
                    save_json(output, data)
        except Exception as exc:
            if size != "e4b":
                raise
            reason = f"failed: {type(exc).__name__}: {exc}"[:300]
            _record_skip(data, "e4b", reason)
            save_json(output, data)
            continue
        finally:
            # Drop the local submodule reference; otherwise it keeps the whole
            # model alive past ``del model`` and the VRAM is not returned.
            del text_model, model, processor, plan
            free_model()


# ---------------------------------------------------------------------------
# Training throughput and Phase-2 sequence length
# ---------------------------------------------------------------------------


def _random_batch(
    length: int, vocab_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        0, vocab_size, (1, length), device=device, dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _training_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    length: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    """One forward/backward/optimizer step.  Timing is done by the caller."""
    batch = _random_batch(length, vocab_size, device)
    output = model(**batch)
    loss = output.loss
    if loss is None:
        raise RuntimeError("training forward did not return a loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del batch, output, loss


def _cleanup_training_state(optimizer: torch.optim.Optimizer | None) -> None:
    """Clear partial gradients and allocator state after a failed step.

    A ``loss.backward()`` that raises OOM can leave ``.grad`` buffers referenced
    by the model parameters; ``gc.collect()`` alone cannot free those.  This is
    called on every warm-up/timed-step OOM and at the end of a measurement so the
    next candidate starts clean.
    """
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"WARNING: optimizer.zero_grad failed during cleanup: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _measure_length(
    model: Any,
    optimizer: torch.optim.Optimizer,
    length: int,
    vocab_size: int,
    device: torch.device,
    *,
    steps: int,
    n_loops: int,
) -> dict[str, Any]:
    """Time ``steps`` training steps at ``length``, excluding a warm-up step."""
    total_bytes = _total_vram_bytes()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warm-up: allocation and kernel selection differ on the first step.  Its
    # peak is kept in the memory verdict so a warm-up spill cannot be hidden by
    # cached blocks during the timed steps.
    try:
        _training_step(model, optimizer, length, vocab_size, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_reserved()
        result = summarize_steps(
            length, [], peak, total_bytes, n_loops, oom=True
        )
        log(f"train: OOM during warm-up at length {length} (N={n_loops})")
        _cleanup_training_state(optimizer)
        return result

    warmup_peak = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    oom = False
    try:
        for _ in range(steps):
            started = time.perf_counter()
            try:
                _training_step(model, optimizer, length, vocab_size, device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                oom = True
                break
            step_times.append(time.perf_counter() - started)
            sample_resources()
    finally:
        timed_peak = torch.cuda.max_memory_reserved()
        _cleanup_training_state(optimizer)

    peak = max(warmup_peak, timed_peak)
    result = summarize_steps(
        length, step_times, peak, total_bytes, n_loops, oom=oom
    )
    log(
        f"train: length {length} N={n_loops} "
        f"tok/s={result['tok_s']} spill={result['spill']} oom={result['oom']} "
        f"peak_reserved={result['peak_reserved_gib']:.2f} GiB"
    )
    return result


def _build_training_model() -> tuple[Any, Any, Any, Any, int, Any]:
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model, processor, plan = load_model(
        ModelSpec(size="e2b", quant="bf16", loop=LoopConfig(n_loops=1))
    )
    text_model = model.model.language_model
    vocab_size = _text_config_vocab(model)

    # Freeze every base parameter (PLE is already frozen by the loader and stays
    # on the CPU); LoRA is the only trainable part.  Never call
    # prepare_model_for_kbit_training: its fp32 cast would blow up the PLE.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del parameter

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=language_lora_target_modules(),
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_model.enable_input_require_grads()
    peft_model.train()

    trainable = [
        name
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("LoRA produced no trainable parameters")
    forbidden = [
        name for name in trainable if "vision_tower" in name or "embed_vision" in name
    ]
    if forbidden:
        raise RuntimeError(f"vision parameters are trainable: {forbidden[:3]}")
    log(f"train: {len(trainable)} LoRA parameter tensors are trainable")

    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            [
                parameter
                for parameter in peft_model.parameters()
                if parameter.requires_grad
            ],
            lr=1e-4,
        )
    except Exception as exc:  # pragma: no cover - depends on bnb install
        log(f"WARNING: AdamW8bit unavailable ({exc}); using torch AdamW")
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in peft_model.parameters()
                if parameter.requires_grad
            ],
            lr=1e-4,
        )

    return peft_model, processor, plan, text_model, vocab_size, optimizer


def measure_training_section(
    args: argparse.Namespace, data: dict[str, Any], output: Path
) -> None:
    probes = data.setdefault("train_length_probe", {})
    arms = [("e2b_oh", 1), ("e2b_l", 2)]
    missing = any(
        str(length) not in probes.get(arm, {})
        for arm, _loop in arms
        for length in args.train_lengths
    )
    # ``None`` means the previous run found no non-spilling Phase-2 length;
    # retry the probe instead of freezing a transient spill forever.
    wants_probe = (
        data.get("phase2_max_length_n2") is None
        or data.get("phase2_max_length_n4") is None
    )
    needs_rates = (
        "train_tok_per_s" not in data
        or "max_nonspill_train_seq_len" not in data
    )
    if not missing and not needs_rates and not wants_probe:
        log("train: already measured")
        return

    log("train: loading e2b bf16 (N=1) for LoRA timing")
    peft_model = processor = plan = text_model = optimizer = None
    try:
        (
            peft_model,
            processor,
            plan,
            text_model,
            vocab_size,
            optimizer,
        ) = _build_training_model()

        device = torch.device("cuda")
        # Lengths ascending, both arms before the next length: an OOM at the
        # largest length then cannot contaminate a smaller length of the other
        # arm through the autograd state a caught OOM leaves alive.
        for length in sorted(args.train_lengths):
            for arm, loop in arms:
                set_n_loops(text_model, loop)
                arm_probe = probes.setdefault(arm, {})
                key = str(length)
                if key in arm_probe:
                    log(f"train[{arm}]: length {key} already measured")
                    continue
                log(f"train[{arm}]: measuring length {length} (N={loop})")
                arm_probe[key] = _measure_length(
                    peft_model,
                    optimizer,
                    length,
                    vocab_size,
                    device,
                    steps=args.train_steps,
                    n_loops=loop,
                )
                save_json(output, data)

        common, per_arm = choose_common_length(probes, [arm for arm, _ in arms])
        data["max_nonspill_train_seq_len_by_arm"] = per_arm
        data["max_nonspill_train_seq_len"] = common
        if common is None:
            save_json(output, data)
            raise RuntimeError(
                "no training length is non-spilling for both O-H and L; "
                "see train_length_probe"
            )
        data["train_measure_length"] = common
        rates = data.setdefault("train_tok_per_s", {})
        for arm, _loop in arms:
            summary = probes[arm][str(common)]
            if not (summary.get("tok_s") or 0) > 0:
                raise RuntimeError(
                    f"no clean throughput for {arm} at length {common}"
                )
            rates[arm] = summary["tok_s"]
        log(
            f"train: common non-spilling length {common}; "
            f"O-H={rates['e2b_oh']:.1f} tok/s, L={rates['e2b_l']:.1f} tok/s"
        )
        save_json(output, data)

        if wants_probe:
            for n_loops in (2, 4):
                key = f"phase2_max_length_n{n_loops}"
                if data.get(key) is not None:
                    continue
                # Keep completed per-length rows.  A crash or a null descent
                # resumes from them; use --fresh to re-probe intentionally.
                set_n_loops(text_model, n_loops)
                existing = data.setdefault(f"phase2_probe_n{n_loops}", {})

                def measure(
                    length: int,
                    _n: int = n_loops,
                    _existing: dict[str, dict[str, Any]] = existing,
                ) -> dict[str, Any]:
                    log(f"train: probing Phase-2 length {length} at N={_n}")
                    summary = _measure_length(
                        peft_model,
                        optimizer,
                        length,
                        vocab_size,
                        device,
                        steps=args.probe_steps,
                        n_loops=_n,
                    )
                    _existing[str(length)] = summary
                    save_json(output, data)
                    return summary

                chosen, probe = probe_descending(
                    tuple(args.probe_candidates), existing, measure
                )
                data[f"phase2_probe_n{n_loops}"] = probe
                data[key] = chosen
                if chosen is None:
                    data[f"{key}_error"] = (
                        "none of the candidate lengths "
                        f"{tuple(args.probe_candidates)} is non-spilling "
                        f"at N={n_loops}"
                    )
                else:
                    data.pop(f"{key}_error", None)
                log(f"train: {key} = {chosen}")
                save_json(output, data)

            data["phase2_max_length"] = data.get("phase2_max_length_n4")
            data["phase2_probe"] = data.get("phase2_probe_n4", {})
            data["phase2_max_length_n_loops"] = 4
            data["phase2_max_length_is_upper_bound"] = True
            if data["phase2_max_length"] is None:
                data["phase2_max_length_error"] = data.get(
                    "phase2_max_length_n4_error",
                    "no non-spilling N=4 length",
                )
            else:
                data.pop("phase2_max_length_error", None)
            save_json(output, data)
    finally:
        # The optimizer and the submodule reference keep tensors alive; drop
        # them before emptying the cache so the next block starts clean.
        del peft_model, processor, plan, text_model, optimizer
        free_model()


# ---------------------------------------------------------------------------
# Benchmark answer lengths
# ---------------------------------------------------------------------------


def pending_benchmark_items(items: list[Any], records: dict) -> list[Any]:
    """Items without a completed ``sample_idx=0`` record."""
    return [
        item
        for item in items
        if (str(item.item_id), 0) not in records
    ]


def pending_benchmarks(
    benches: list[str],
    tokens_per_item: dict[str, Any],
    finish_rates: dict[str, Any],
) -> list[str]:
    """Benchmarks that still need a measurement.

    ``skipped`` entries are deliberately **not** consulted: an "unavailable"
    dataset is retried on the next run instead of becoming a permanent hole
    that only ``--fresh`` could clear.
    """
    return [
        name
        for name in benches
        if name not in tokens_per_item or name not in finish_rates
    ]


def _benchmark_messages(module: Any, items: list[Any]) -> list[list[dict]]:
    return [module.build_messages(item) for item in items]


def _generate_chunk(
    gen_fn: Callable,
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
    image_soft_tokens: int,
    request_ids: list[str] | None = None,
    max_batch: int | None = None,
    on_complete: Callable[[int, Any], None] | None = None,
    heartbeat: Callable[[dict], None] | None = None,
) -> list[Any]:
    kwargs: dict[str, Any] = {
        "seed": seed,
        "image_soft_tokens": image_soft_tokens,
    }
    try:
        parameters = inspect.signature(gen_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = {}
    accepts_kwargs = not parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def accepts(name: str) -> bool:
        return accepts_kwargs or name in parameters

    if request_ids is not None and accepts("request_ids"):
        kwargs["request_ids"] = request_ids
    if max_batch is not None and accepts("max_batch"):
        kwargs["max_batch"] = max_batch
    if on_complete is not None and accepts("on_complete"):
        kwargs["on_complete"] = on_complete
    if heartbeat is not None and accepts("heartbeat"):
        kwargs["heartbeat"] = heartbeat
    outputs = gen_fn(
        model,
        processor,
        conversations,
        sampling,
        **kwargs,
    )
    outputs = list(outputs)
    if len(outputs) != len(conversations):
        raise RuntimeError(
            f"generator returned {len(outputs)} outputs for "
            f"{len(conversations)} conversations"
        )
    return outputs


def measure_benchmarks(
    args: argparse.Namespace,
    data: dict[str, Any],
    output: Path,
    *,
    load_fn: Callable = load_model,
    gen_fn: Callable | None = None,
    engine: str = "padded",
) -> None:
    if gen_fn is None:
        gen_fn = generate_batch if engine == "padded" else generate_batch_refill
    tokens_per_item = data.setdefault("tokens_per_item", {})
    finish_rates = data.setdefault("length_finish_rate", {})
    pending = pending_benchmarks(args.benches, tokens_per_item, finish_rates)
    if not pending:
        log("bench: already measured")
        return

    log(
        f"bench: preparing {len(pending)} benchmark(s): "
        f"{', '.join(pending)}"
    )
    spec = ModelSpec(size="e2b", quant="bf16", loop=LoopConfig(n_loops=2))
    model = processor = plan = None
    items: list[Any] = []
    conversations: list[Any] = []
    outputs: list[Any] = []
    try:
        for name in pending:
            module = _benchmark_module(name)
            try:
                items = module.load_items(
                    limit=args.items_per_bench, seed=args.seed
                )
            except Exception as exc:
                reason = f"unavailable: {type(exc).__name__}: {exc}"[:300]
                _record_skip(data, name, reason)
                save_json(output, data)
                continue
            finally:
                _clear_benchmark_caches(module)

            if not items:
                _record_skip(data, name, "unavailable: dataset returned no items")
                save_json(output, data)
                continue

            soft_tokens = int(BENCH_IMAGE_SOFT_TOKENS.get(name, 280))
            sampling = Sampling(max_new_tokens=args.max_new_tokens)
            fingerprint = run_fingerprint(
                spec,
                name,
                items,
                samples=1,
                sampling=sampling,
                seed=args.seed,
                image_soft_tokens=soft_tokens,
                prereg_tag=CALIBRATION_TAG,
                engine=engine,
            )
            partial = bench_partial_path(output, name)
            try:
                records, handle = _open_partial(
                    partial, fingerprint, fresh=args.fresh
                )
            except PartialProvenanceError as exc:
                raise SystemExit(
                    f"{exc}\nRe-run with --fresh to rotate {partial}."
                ) from exc

            if model is None:
                log("bench: loading e2b bf16 (N=2)")
                model, processor, plan = load_fn(spec)

            try:
                remaining = pending_benchmark_items(items, records)
                batch_size = benchmark_batch(args)
                if engine == "refill":
                    # The refill engine owns the slot scheduler: hand it the
                    # whole remaining benchmark as one queue and cap the slots
                    # with the benchmark batch.
                    if remaining:
                        log(
                            f"bench[{name}]: generating {len(remaining)} "
                            f"pending as one refill queue "
                            f"(slots={batch_size}, soft_tokens={soft_tokens})"
                        )
                        request_ids = [
                            f"{item.item_id}:0" for item in remaining
                        ]
                        heartbeat_path = partial.with_suffix(
                            ".heartbeat.json"
                        )

                        def beat(status: dict) -> None:
                            payload = {
                                "bench": name,
                                "engine": engine,
                                "updated_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                **status,
                            }
                            heartbeat_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            temporary = heartbeat_path.with_name(
                                f"{heartbeat_path.name}.{os.getpid()}.tmp"
                            )
                            temporary.write_text(
                                json.dumps(payload, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            os.replace(temporary, heartbeat_path)

                        while True:
                            pending = [
                                item
                                for item in remaining
                                if (str(item.item_id), 0) not in records
                            ]
                            if not pending:
                                break
                            conversations = [
                                module.build_messages(item)
                                for item in pending
                            ]
                            request_ids = [
                                f"{item.item_id}:0" for item in pending
                            ]

                            def on_complete(
                                index: int, generated: Any
                            ) -> None:
                                item = pending[index]
                                record = {
                                    "item_id": str(item.item_id),
                                    "sample_idx": 0,
                                    "completion_tokens": int(
                                        generated.completion_tokens
                                    ),
                                    "finish_reason": generated.finish_reason,
                                }
                                key = (str(item.item_id), 0)
                                records[key] = record
                                _write_records(handle, [record])

                            try:
                                outputs = _generate_chunk(
                                    gen_fn,
                                    model,
                                    processor,
                                    conversations,
                                    sampling,
                                    args.seed,
                                    soft_tokens,
                                    request_ids=request_ids,
                                    max_batch=batch_size,
                                    on_complete=on_complete,
                                    heartbeat=beat,
                                )
                            except torch.cuda.OutOfMemoryError:
                                if batch_size > 1:
                                    batch_size //= 2
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    log(
                                        f"bench[{name}]: OOM, retrying the "
                                        f"refill queue with slots={batch_size}"
                                    )
                                    continue
                                raise
                            new_records = []
                            for item, generated in zip(pending, outputs):
                                key = (str(item.item_id), 0)
                                if key in records:
                                    continue
                                record = {
                                    "item_id": str(item.item_id),
                                    "sample_idx": 0,
                                    "completion_tokens": int(
                                        generated.completion_tokens
                                    ),
                                    "finish_reason": generated.finish_reason,
                                }
                                records[key] = record
                                new_records.append(record)
                            _write_records(handle, new_records)
                            save_json(output, data)
                            break
                else:
                    index = 0
                    while index < len(remaining):
                        chunk = remaining[index : index + batch_size]
                        conversations = _benchmark_messages(module, chunk)
                        log(
                            f"bench[{name}]: generating {index + 1}-"
                            f"{index + len(chunk)} of {len(remaining)} pending "
                            f"(batch={batch_size}, soft_tokens={soft_tokens})"
                        )
                        try:
                            outputs = _generate_chunk(
                                gen_fn,
                                model,
                                processor,
                                conversations,
                                sampling,
                                args.seed,
                                soft_tokens,
                                request_ids=[
                                    f"{item.item_id}:0" for item in chunk
                                ],
                            )
                        except torch.cuda.OutOfMemoryError:
                            if batch_size > 1:
                                batch_size //= 2
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                log(
                                    f"bench[{name}]: OOM, retrying with "
                                    f"batch={batch_size}"
                                )
                                continue
                            raise
                        new_records = []
                        for item, generated in zip(chunk, outputs):
                            record = {
                                "item_id": str(item.item_id),
                                "sample_idx": 0,
                                "completion_tokens": int(
                                    generated.completion_tokens
                                ),
                                "finish_reason": generated.finish_reason,
                            }
                            records[(str(item.item_id), 0)] = record
                            new_records.append(record)
                        _write_records(handle, new_records)
                        index += len(chunk)
                        save_json(output, data)
            finally:
                handle.close()

            token_counts: list[int] = []
            length_flags: list[bool] = []
            for item in items:
                record = records.get((str(item.item_id), 0))
                if record is None:
                    raise RuntimeError(
                        f"bench[{name}]: missing completion for {item.item_id}"
                    )
                token_counts.append(int(record["completion_tokens"]))
                length_flags.append(record["finish_reason"] == "length")
            tokens_per_item[name] = {
                "basis": "per_completion",
                "samples": registered_sample_map([name])[name],
                "image_soft_tokens": soft_tokens,
                "n_items": len(items),
                "p50": round(percentile(token_counts, 0.50), 1),
                "p90": round(percentile(token_counts, 0.90), 1),
            }
            finish_rates[name] = round(
                sum(length_flags) / len(length_flags), 4
            )
            data.get("skipped", {}).pop(name, None)
            log(
                f"bench[{name}]: p50={tokens_per_item[name]['p50']:.0f} "
                f"p90={tokens_per_item[name]['p90']:.0f} "
                f"length_rate={finish_rates[name]:.3f} "
                f"soft_tokens={soft_tokens}"
            )
            save_json(output, data)
    finally:
        del outputs, conversations, items, model, processor, plan
        free_model()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _int_csv(value: str) -> list[int]:
    return [int(part) for part in _csv(value)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure generation/training throughput and evaluation budgets."
    )
    parser.add_argument(
        "--sizes", default=DEFAULT_SIZES, help="comma-separated: e2b,e4b"
    )
    parser.add_argument(
        "--sections",
        default="gen,bench,train",
        help="comma-separated sections: gen,bench,train "
        "(train always runs last so the OOM probe cannot affect other blocks)",
    )
    parser.add_argument(
        "--benches",
        default=DEFAULT_BENCHES,
        help=f"comma-separated benchmarks (default: {DEFAULT_BENCHES}); "
        f"available: {','.join(ALL_BENCHES)}",
    )
    parser.add_argument("--items-per-bench", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument(
        "--gen-new-tokens",
        type=int,
        default=GEN_MEASURE_TOKENS,
        help="tokens decoded per sequence per length bucket (the cache is "
        "preallocated at the bucket length); a bucket always times at least "
        f"{GEN_MIN_MEASURED_TOKENS} tokens in aggregate",
    )
    parser.add_argument(
        "--gen-lengths",
        default=",".join(str(length) for length in GEN_LENGTH_BUCKETS),
        help="expected completion-length buckets for generation throughput",
    )
    parser.add_argument(
        "--gen-batch",
        type=int,
        default=0,
        help="generation-throughput batch (0 = auto from max_new_tokens)",
    )
    parser.add_argument(
        "--bench-batch",
        type=int,
        default=0,
        help="completion-length batch (0 = auto from max_new_tokens)",
    )
    parser.add_argument(
        "--bench-engine",
        default="padded",
        choices=("padded", "refill"),
        help=(
            "engine for the completion-length section: padded (default) or "
            "round-based refill (opt-in until Task 5c Part B proves the win)"
        ),
    )
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument(
        "--train-lengths",
        default=",".join(str(length) for length in TRAIN_LENGTHS),
        help="candidate Phase-1 training lengths (spill measured per length)",
    )
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument(
        "--probe-candidates", default="4096,3072,2048,1024",
        help="descending Phase-2 lengths probed at N=2 and N=4",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=0.92,
        help=(
            "cap the CUDA caching allocator at this fraction of VRAM before "
            "loading, so over-budget runs OOM instead of paging to RAM; "
            "0 disables"
        ),
    )
    parser.add_argument("--vram-free-margin-gib", type=float, default=0.5)
    parser.add_argument(
        "--min-e4b-ram-gib",
        type=float,
        default=20.0,
        help="skip E4B when available system RAM is below this value",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output JSON (default: <artifacts>/calibration/calibration.json)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="rotate an existing calibration and its per-benchmark partials",
    )
    parser.add_argument(
        "--force",
        dest="fresh",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    args.sizes = _csv(args.sizes)
    args.benches = _csv(args.benches)
    args.sections = _csv(args.sections)
    args.train_lengths = _int_csv(args.train_lengths)
    args.probe_candidates = _int_csv(args.probe_candidates)
    args.gen_lengths = _int_csv(args.gen_lengths)

    for size in args.sizes:
        if size not in GEN_LOOPS:
            parser.error(f"unknown size {size!r}; expected e2b or e4b")
    for section in args.sections:
        if section not in ("gen", "train", "bench"):
            parser.error(f"unknown section {section!r}")
    for bench in args.benches:
        if bench not in ALL_BENCHES:
            parser.error(f"unknown benchmark {bench!r}")
    if not args.sizes or not args.benches or not args.sections:
        parser.error("--sizes, --benches and --sections must not be empty")
    if not args.train_lengths or not args.probe_candidates:
        parser.error("--train-lengths and --probe-candidates must not be empty")
    if not args.gen_lengths or min(args.gen_lengths) < 1:
        parser.error("--gen-lengths must be non-empty and positive")
    if args.items_per_bench < 1:
        parser.error("--items-per-bench must be at least 1")
    if args.train_steps < 1 or args.probe_steps < 1:
        parser.error("--train-steps and --probe-steps must be at least 1")
    if min(args.train_lengths) < 1 or min(args.probe_candidates) < 1:
        parser.error("--train-lengths and --probe-candidates must be positive")
    if args.max_new_tokens < 1 or args.gen_new_tokens < 1:
        parser.error("--max-new-tokens and --gen-new-tokens must be at least 1")
    if args.gen_batch < 0 or args.bench_batch < 0:
        parser.error("--gen-batch and --bench-batch must be >= 0 (0 = auto)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.vram_cap > 0:
        cap_cuda_allocator(args.vram_cap, args.vram_free_margin_gib)
    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir() / "calibration" / "calibration.json"
    )
    fingerprint = build_fingerprint(args)
    if args.fresh and output.is_file():
        log(f"fresh: rotating existing {output}")
        data: dict[str, Any] = {}
    else:
        data = load_json(output)
        try:
            data = ensure_compatible(data, fingerprint)
        except PartialProvenanceError as exc:
            raise SystemExit(str(exc)) from exc
    data["fingerprint"] = fingerprint
    data.setdefault("meta", {})["gen_batches"] = generation_batches(args)
    data["meta"]["gen_measured_tokens"] = int(args.gen_new_tokens)
    data["meta"]["bench_batch_size"] = benchmark_batch(args)
    data["meta"]["bench_engine"] = str(args.bench_engine)

    log(f"output: {output}")
    log(f"sizes: {args.sizes}; sections: {args.sections}; benches: {args.benches}")
    log(
        f"gpu: {_gpu_name() or 'none'}; "
        f"ram_available={available_ram_gib():.1f} GiB; "
        f"total_vram={_total_vram_bytes() / 2**30:.1f} GiB"
    )
    save_json(output, data)

    # Probe Phase-2 lengths last: a caught CUDA OOM can leave the autograd
    # graph (and a couple of GiB) alive until the process exits, so we do not
    # want another model load afterwards.
    for section in ("gen", "bench", "train"):
        if section not in args.sections:
            continue
        if section == "gen":
            measure_generation_section(args, data, output)
        elif section == "bench":
            measure_benchmarks(args, data, output, engine=args.bench_engine)
        else:
            measure_training_section(args, data, output)

    save_json(output, data)
    log(f"done: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
