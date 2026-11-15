"""Modal cost model for Stage 0 sampling (no Modal import).

The estimate is **all-in** and deliberately conservative: GPU + CPU + memory at
Modal's per-second list prices, plus per-container startup, multiplied by a
1.3 safety margin.  This is the number checked against ``--max-usd`` before any
Modal contact and the basis of the runtime spend guard.

Prices are the Modal list prices recorded in the project ledger (USD per hour;
the per-second price is hourly / 3600).  Throughput is a conservative planning
estimate for stock vLLM ``vllm==0.30.0`` serving Gemma 4 E2B with thinking on;
the design requires a throughput probe before the real launch.

The registered Stage 0 cloud budget is a **$30 usage limit / $0 spend limit**.
``scripts/modal_stage0.py`` additionally refuses any plan above its
``--max-usd`` argument (default 12) before contacting Modal.
"""

from __future__ import annotations

from typing import Any

# Modal list prices, USD per GPU-hour (ledger 2026-11-04).
GPU_HOURLY_USD: dict[str, float] = {
    "B300": 7.10,
    "B200": 6.25,
    "H200": 4.54,
    "H100": 3.95,
    "RTX PRO 6000": 3.03,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 1.95,
    "A10": 1.10,
    "L4": 0.80,
}

# Planning throughput for vLLM E2B (tokens/s, generation + prefill), conservative.
GPU_TOKENS_PER_SEC: dict[str, float] = {
    "B300": 22_000,
    "B200": 20_000,
    "H200": 17_000,
    "H100": 15_000,
    "RTX PRO 6000": 12_000,
    "A100-80GB": 10_000,
    "A100-40GB": 8_000,
    "L40S": 7_000,
    "A10": 4_500,
    "L4": 3_500,
}

CPU_HOURLY_PER_CORE_USD = 0.047
MEMORY_HOURLY_PER_GIB_USD = 0.008
DEFAULT_CPU_CORES = 8
DEFAULT_MEMORY_GIB = 48
DEFAULT_STARTUP_SECONDS = 600
DEFAULT_SAFETY_MARGIN = 1.3
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 10
DEFAULT_IMAGE_BUILD_USD = 0.25

DEFAULT_GPU = "A100-80GB"
DEFAULT_PROMPTS = 2000 + 2000 + 1000
DEFAULT_SAMPLES_PER_PROMPT = 4
DEFAULT_AVG_GENERATED_TOKENS = 3_200  # ~64M generated tokens / 20,000 samples
DEFAULT_AVG_PROMPT_TOKENS = 500
DEFAULT_OVERHEAD = 0.15
DEFAULT_SHARD_SIZE = 500
HARD_GLOBAL_CAP_USD = 30.0


def per_second_price(gpu: str) -> float:
    return GPU_HOURLY_USD[gpu] / 3600.0


def runtime_cost_usd(
    gpu: str,
    wall_seconds: float,
    *,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    """Cost of one container for an observed wall-clock duration."""

    if gpu not in GPU_HOURLY_USD:
        raise KeyError(f"unknown GPU {gpu!r}; known: {sorted(GPU_HOURLY_USD)}")
    hours = max(0.0, float(wall_seconds)) / 3600.0
    gpu_usd = hours * GPU_HOURLY_USD[gpu]
    cpu_usd = hours * int(cpu_cores) * CPU_HOURLY_PER_CORE_USD
    memory_usd = hours * int(memory_gib) * MEMORY_HOURLY_PER_GIB_USD
    raw = gpu_usd + cpu_usd + memory_usd
    return {
        "gpu": gpu,
        "wall_seconds": float(wall_seconds),
        "gpu_usd": gpu_usd,
        "cpu_usd": cpu_usd,
        "memory_usd": memory_usd,
        "raw_usd": raw,
        "safety_margin": float(safety_margin),
        "usd": raw * float(safety_margin),
    }


def worst_case_usd(
    gpu: str,
    *,
    timeout_s: float,
    startup_timeout_s: float = DEFAULT_STARTUP_SECONDS,
    scaledown_window_s: float = DEFAULT_SCALEDOWN_WINDOW_SECONDS,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    image_build_usd: float = DEFAULT_IMAGE_BUILD_USD,
) -> dict[str, Any]:
    """True per-call bound: timeout + startup + idle, all-in, x1.3 + image."""

    wall = float(timeout_s) + float(startup_timeout_s) + float(scaledown_window_s)
    cost = runtime_cost_usd(
        gpu,
        wall,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        safety_margin=safety_margin,
    )
    cost["timeout_s"] = float(timeout_s)
    cost["startup_timeout_s"] = float(startup_timeout_s)
    cost["scaledown_window_s"] = float(scaledown_window_s)
    cost["image_build_usd"] = float(image_build_usd)
    cost["usd"] = cost["usd"] + float(image_build_usd)
    cost["raw_usd"] = cost["raw_usd"] + float(image_build_usd)
    return cost


def estimate_gpu(
    total_tokens: int,
    gpu: str,
    *,
    tokens_per_sec: float | None = None,
    overhead: float = DEFAULT_OVERHEAD,
    containers: int = 1,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    startup_seconds: float = DEFAULT_STARTUP_SECONDS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    """All-in estimate for ``total_tokens`` split over ``containers``."""

    if gpu not in GPU_HOURLY_USD:
        raise KeyError(f"unknown GPU {gpu!r}; known: {sorted(GPU_HOURLY_USD)}")
    rate = float(tokens_per_sec or GPU_TOKENS_PER_SEC[gpu])
    if rate <= 0:
        raise ValueError("tokens_per_sec must be positive")
    containers = max(1, int(containers))
    token_hours = int(total_tokens) / rate / 3600.0
    wall_hours = token_hours * (1.0 + float(overhead)) + (
        float(startup_seconds) * containers / 3600.0
    )
    gpu_usd = wall_hours * GPU_HOURLY_USD[gpu]
    cpu_usd = wall_hours * int(cpu_cores) * CPU_HOURLY_PER_CORE_USD
    memory_usd = wall_hours * int(memory_gib) * MEMORY_HOURLY_PER_GIB_USD
    raw = gpu_usd + cpu_usd + memory_usd
    return {
        "gpu": gpu,
        "hourly_usd": GPU_HOURLY_USD[gpu],
        "per_second_usd": per_second_price(gpu),
        "tokens_per_sec": rate,
        "total_tokens": int(total_tokens),
        "containers": containers,
        "gpu_hours": wall_hours,
        "token_hours": token_hours,
        "startup_seconds": float(startup_seconds),
        "cpu_cores": int(cpu_cores),
        "memory_gib": int(memory_gib),
        "gpu_usd": gpu_usd,
        "cpu_usd": cpu_usd,
        "memory_usd": memory_usd,
        "raw_usd": raw,
        "safety_margin": float(safety_margin),
        "usd": raw * float(safety_margin),
        "per_container_wall_hours": wall_hours / containers,
        "per_container_usd": raw * float(safety_margin) / containers,
        "overhead": float(overhead),
    }


def estimate_all_gpus(
    total_tokens: int,
    *,
    overhead: float = DEFAULT_OVERHEAD,
    gpus: list[str] | None = None,
    containers: int = 1,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    startup_seconds: float = DEFAULT_STARTUP_SECONDS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, dict[str, Any]]:
    names = gpus or list(GPU_HOURLY_USD)
    return {
        name: estimate_gpu(
            total_tokens,
            name,
            overhead=overhead,
            containers=containers,
            cpu_cores=cpu_cores,
            memory_gib=memory_gib,
            startup_seconds=startup_seconds,
            safety_margin=safety_margin,
        )
        for name in names
    }


def sampling_plan(
    *,
    prompts: int = DEFAULT_PROMPTS,
    samples_per_prompt: int = DEFAULT_SAMPLES_PER_PROMPT,
    avg_generated_tokens: int = DEFAULT_AVG_GENERATED_TOKENS,
    avg_prompt_tokens: int = DEFAULT_AVG_PROMPT_TOKENS,
    gpu: str = DEFAULT_GPU,
    overhead: float = DEFAULT_OVERHEAD,
    shard_size: int = DEFAULT_SHARD_SIZE,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_gib: int = DEFAULT_MEMORY_GIB,
    startup_seconds: float = DEFAULT_STARTUP_SECONDS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    samples = int(prompts) * int(samples_per_prompt)
    generated = samples * int(avg_generated_tokens)
    prompt_tokens = samples * int(avg_prompt_tokens)
    total = generated + prompt_tokens
    shards = (int(prompts) + int(shard_size) - 1) // int(shard_size)
    chosen = estimate_gpu(
        total,
        gpu,
        overhead=overhead,
        containers=shards,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        startup_seconds=startup_seconds,
        safety_margin=safety_margin,
    )
    return {
        "prompts": int(prompts),
        "samples_per_prompt": int(samples_per_prompt),
        "samples": samples,
        "generated_tokens": generated,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total,
        "shard_size": int(shard_size),
        "shards": shards,
        "gpu": gpu,
        "chosen": chosen,
        "all_gpus": estimate_all_gpus(
            total,
            overhead=overhead,
            containers=shards,
            cpu_cores=cpu_cores,
            memory_gib=memory_gib,
            startup_seconds=startup_seconds,
            safety_margin=safety_margin,
        ),
    }


def check_budget(
    estimate: dict[str, Any],
    *,
    max_usd: float = 12.0,
    hard_cap_usd: float = HARD_GLOBAL_CAP_USD,
) -> None:
    """Refuse plans above the launch cap or the registered $30 usage limit."""

    if not 0 < float(max_usd) <= hard_cap_usd:
        raise ValueError(f"max_usd must be in (0, {hard_cap_usd:g}]; got {max_usd!r}")
    usd = float(estimate["usd"])
    if usd > max_usd:
        raise ValueError(
            f"estimated ${usd:.2f} exceeds --max-usd ${max_usd:.2f}; refusing to launch"
        )
    if usd > hard_cap_usd:
        raise ValueError(
            f"estimated ${usd:.2f} exceeds the registered ${hard_cap_usd:.0f} usage limit"
        )


def format_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"prompts={plan['prompts']} samples/prompt={plan['samples_per_prompt']} "
        f"samples={plan['samples']}",
        f"tokens: generated={plan['generated_tokens']:,} "
        f"prompt={plan['prompt_tokens']:,} total={plan['total_tokens']:,}",
        f"shards={plan['shards']} (shard_size={plan['shard_size']}, "
        f"max_containers=1, no retries)",
        "",
        f"{'GPU':<16}{'tok/s':>10}{'GPU-h':>10}{'GPU$':>9}{'CPU$':>8}"
        f"{'MEM$':>8}{'all-in$':>10}",
    ]
    for name, entry in plan["all_gpus"].items():
        marker = "*" if name == plan["gpu"] else " "
        lines.append(
            f"{marker}{name:<15}{entry['tokens_per_sec']:>10,.0f}"
            f"{entry['gpu_hours']:>10.2f}{entry['gpu_usd']:>9.2f}"
            f"{entry['cpu_usd']:>8.2f}{entry['memory_usd']:>8.2f}"
            f"{entry['usd']:>10.2f}"
        )
    chosen = plan["chosen"]
    lines.append(
        f"\nall-in includes CPU ({chosen['cpu_cores']} cores x "
        f"${CPU_HOURLY_PER_CORE_USD}/h), memory ({chosen['memory_gib']} GiB x "
        f"${MEMORY_HOURLY_PER_GIB_USD}/GiB-h), {chosen['startup_seconds']:.0f}s "
        f"startup/container and a x{chosen['safety_margin']:.1f} margin"
    )
    return "\n".join(lines)


__all__ = [
    "CPU_HOURLY_PER_CORE_USD",
    "DEFAULT_AVG_GENERATED_TOKENS",
    "DEFAULT_AVG_PROMPT_TOKENS",
    "DEFAULT_CPU_CORES",
    "DEFAULT_GPU",
    "DEFAULT_MEMORY_GIB",
    "DEFAULT_OVERHEAD",
    "DEFAULT_PROMPTS",
    "DEFAULT_SAFETY_MARGIN",
    "DEFAULT_SAMPLES_PER_PROMPT",
    "DEFAULT_SHARD_SIZE",
    "DEFAULT_STARTUP_SECONDS",
    "GPU_HOURLY_USD",
    "GPU_TOKENS_PER_SEC",
    "HARD_GLOBAL_CAP_USD",
    "MEMORY_HOURLY_PER_GIB_USD",
    "check_budget",
    "estimate_all_gpus",
    "DEFAULT_IMAGE_BUILD_USD",
    "DEFAULT_SCALEDOWN_WINDOW_SECONDS",
    "estimate_gpu",
    "format_plan",
    "per_second_price",
    "runtime_cost_usd",
    "sampling_plan",
    "worst_case_usd",
]
