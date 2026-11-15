"""Registered refill-vs-padded equivalence gate (Task 5d).

The prereg registers the round-based refill engine as *the* evaluation engine
for O, O-H and L only if it passes this gate before ``analysis-v1``.  The gate
uses a frozen 32-prompt fixture in ``tests/fixtures/engine_gate_prompts.json``
(16 LiveCodeBench-style, 16 math), never confirmatory benchmark items.  The
registered v2 gate compares the 16-prompt subset per combination (8 LCB + 8
math) with a 256-token window and round size 64 (four decode rounds):

* greedy (temperature 0) tokens must pass the near-tie and token-exactness
  rules between the padded and refill engines for every registered
  (arm, block, bridge, policy, N) combination;
* with the registered sampling settings (temperature 1.0, top-p 0.95, top-k 64)
  the per-request sampled tokens use the same rule;
* top-20 logprobs are recorded by both engines and the chosen-token logprob
  drift is gated as a bug detector;
* throughput (tokens/s) is measured from the same greedy runs and reported
  next to the equality result.

The script needs a CUDA GPU and is meant to be launched once by the controller
(never as part of the CPU test suite)::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/engine_gate.py \
        --blocks 15-29,25-29,16-18 --out artifacts/engine_gate/engine_gate.json

It writes JSON plus Markdown and exits 0 only if every criterion passes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import yaml

from gemmaloop import paths
from gemmaloop.model.layer_plan import parse_block
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling, _model_device, generate_batch, max_batch_for
from gemmaloop.serve.refill import generate_batch_refill
from gemmaloop.train.perf import cap_cuda_allocator

PROMPTS_FILE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "engine_gate_prompts.json"
)
DEFAULT_SMOKE_BLOCKS = "16-18"
REGISTERED_TEMPERATURE = 1.0
REGISTERED_TOP_P = 0.95
REGISTERED_TOP_K = 64
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_ROUND_SIZE = 128
DEFAULT_MAX_BATCH = 4
PROMPTS_PER_FAMILY = 16
LOGPROB_MEAN_LIMIT = 0.01
LOGPROB_MAX_LIMIT = 0.1
LORA_B_STD = 0.02
BRIDGE_G_LOW = 0.05
BRIDGE_G_HIGH = 0.2


# ---------------------------------------------------------------------------
# Prompt list
# ---------------------------------------------------------------------------
def load_prompts(path: str | Path = PROMPTS_FILE) -> dict[str, Any]:
    """Load and validate the frozen gate prompt list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path}: 'prompts' must be a non-empty list")
    ids = [str(prompt["id"]) for prompt in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: prompt ids are not unique")
    counts = {"lcb": 0, "math": 0}
    for prompt in prompts:
        kind = str(prompt.get("kind", ""))
        if kind not in counts:
            raise ValueError(f"{path}: prompt {prompt['id']!r} has kind {kind!r}")
        counts[kind] += 1
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}: prompt {prompt['id']!r} has no messages")
    if counts != {"lcb": PROMPTS_PER_FAMILY, "math": PROMPTS_PER_FAMILY}:
        raise ValueError(
            f"{path}: expected {PROMPTS_PER_FAMILY} lcb + "
            f"{PROMPTS_PER_FAMILY} math prompts, got {counts}"
        )
    sampling_ids = [str(value) for value in payload.get("sampling_ids", [])]
    if len(sampling_ids) != PROMPTS_PER_FAMILY or not set(sampling_ids) <= set(ids):
        raise ValueError(
            f"{path}: 'sampling_ids' must name {PROMPTS_PER_FAMILY} listed prompts"
        )
    long_prompts = payload.get("long_prompts", [])
    if not isinstance(long_prompts, list) or len(long_prompts) != 4:
        raise ValueError(f"{path}: 'long_prompts' must contain 4 prompts")
    long_ids = [str(prompt["id"]) for prompt in long_prompts]
    if len(set(long_ids)) != len(long_ids):
        raise ValueError(f"{path}: long prompt ids are not unique")
    return {
        "prompts": prompts,
        "ids": ids,
        "sampling_ids": sampling_ids,
        "sampling_set": set(sampling_ids),
        "long_prompts": long_prompts,
        "long_ids": long_ids,
    }


def prompt_provenance(path: str | Path) -> dict[str, Any]:
    """Record that the gate prompts are synthetic and held out."""

    data = Path(path).read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "source": "synthetic gate probes (tests/fixtures), never confirmatory items",
        "confirmatory_items_used": False,
    }


# ---------------------------------------------------------------------------
# Registered coverage from prereg/prereg.yaml
# ---------------------------------------------------------------------------
def _combo_key(
    arm: str,
    block: str | None,
    bridge: str | None,
    adapter_policy: str | None,
    n_loops: int,
) -> tuple[str, str, str, str, int]:
    return (
        str(arm),
        "" if block is None else str(block),
        "" if bridge is None else str(bridge),
        "" if adapter_policy is None else str(adapter_policy),
        int(n_loops),
    )


def _normalise_combination(row: dict, where: str) -> dict:
    arm = str(row.get("arm", "")).strip()
    if arm not in ("O", "O-H", "L"):
        raise ValueError(f"{where}: unknown arm {arm!r}")
    block = row.get("block")
    bridge = row.get("bridge")
    policy = row.get("adapter_policy")
    if block is not None:
        parse_block(str(block))
    if arm == "L" and block is None:
        raise ValueError(f"{where}: L needs a block")
    if bridge is not None and str(bridge) not in (
        "scalar",
        "none",
        "damped",
        "gch",
    ):
        raise ValueError(f"{where}: unknown bridge {bridge!r}")
    if policy is not None and str(policy) not in ("always", "loop_only"):
        raise ValueError(f"{where}: unknown adapter policy {policy!r}")
    n_loops = [int(value) for value in row.get("n_loops", [])]
    if not n_loops or any(value < 1 for value in n_loops):
        raise ValueError(f"{where}: n_loops must be positive")
    return {
        "arm": arm,
        "block": None if block is None else str(block),
        "bridge": None if bridge is None else str(bridge),
        "adapter_policy": None if policy is None else str(policy),
        "n_loops": sorted(set(n_loops)),
    }


def load_engine_gate(
    path: str | Path = Path("prereg/prereg.yaml"),
) -> dict[str, Any]:
    """Read the registered gate combinations from ``prereg/prereg.yaml``."""

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gate = payload.get("engine_gate")
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: no 'engine_gate' registration")
    raw = gate.get("combinations")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: engine_gate.combinations must be non-empty")
    combinations = [
        _normalise_combination(row, f"{path}: combination {index}")
        for index, row in enumerate(raw)
    ]
    if bool(gate.get("one_n1_key_per_block", True)):
        reduced: list[dict] = []
        seen_blocks: set[str] = set()
        for combo in combinations:
            if combo["arm"] == "L" and 1 in combo["n_loops"]:
                if combo["block"] in seen_blocks:
                    combo = {
                        **combo,
                        "n_loops": [
                            value for value in combo["n_loops"] if value != 1
                        ],
                    }
                else:
                    seen_blocks.add(str(combo["block"]))
            if combo["n_loops"]:
                reduced.append(combo)
        combinations = reduced
    expected = {
        _combo_key(
            combo["arm"],
            combo["block"],
            combo["bridge"],
            combo["adapter_policy"],
            n_loops,
        )
        for combo in combinations
        for n_loops in combo["n_loops"]
    }
    settings = {
        "prompts": str(gate.get("prompts", PROMPTS_FILE)),
        "size": str(gate.get("size", "e2b")),
        "quant": str(gate.get("quant", "bf16")),
        "logprob_mean_limit": float(gate.get("logprob_mean_limit", LOGPROB_MEAN_LIMIT)),
        "logprob_max_limit": float(gate.get("logprob_max_limit", LOGPROB_MAX_LIMIT)),
        "max_new_tokens": int(gate.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        "round_size": int(gate.get("round_size", DEFAULT_ROUND_SIZE)),
        "max_batch": int(gate.get("max_batch", DEFAULT_MAX_BATCH)),
        "seed": int(gate.get("seed", 0)),
        "token_limit": int(gate.get("token_limit", 512)),
        "near_tie_logit_gap": float(gate.get("near_tie_logit_gap", 0.25)),
        "min_exact_fraction": float(gate.get("min_exact_fraction", 0.90)),
        "sampling_n": [int(value) for value in gate.get("sampling_n", [2])],
        "throughput_n": [
            int(value) for value in gate.get("throughput_n", [2])
        ],
        "comparison_prompts": int(gate.get("comparison_prompts", 8)),
        "refill_budgets": [
            int(value)
            for value in gate.get(
                "refill_budgets", [128, 128, 192, 192, 256, 256, 256, 256]
            )
        ],
        "control_batch": int(gate.get("control_batch", 4)),
        "one_n1_key_per_block": bool(
            gate.get("one_n1_key_per_block", True)
        ),
        "long_max_new_tokens": int(gate.get("long_max_new_tokens", 64)),
        "long_round_size": int(gate.get("long_round_size", 64)),
        "long_round_1024_tokens": int(
            gate.get("long_round_1024_tokens", 1200)
        ),
    }
    if settings["round_size"] >= settings["max_new_tokens"]:
        raise ValueError(
            f"{path}: engine_gate.round_size must be smaller than "
            "max_new_tokens (multi-round re-prefill)"
        )
    if not settings["sampling_n"]:
        raise ValueError(f"{path}: engine_gate.sampling_n must be non-empty")
    return {
        **settings,
        "combinations": combinations,
        "expected": expected,
    }


def coverage_criterion(
    expected: Iterable[tuple], observed: Iterable[tuple]
) -> dict:
    """PASS only when every registered (arm, block, bridge, policy, N) ran."""

    expected = set(expected)
    observed = set(observed)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    passed = bool(expected) and not missing and not extra
    detail = (
        f"{len(observed)}/{len(expected)} registered combinations covered"
        + ("" if passed else f"; missing={missing}; unexpected={extra}")
    )
    return {
        "criterion": "registered_coverage",
        "passed": bool(passed),
        "expected": len(expected),
        "observed": len(observed),
        "missing": [list(item) for item in missing],
        "unexpected": [list(item) for item in extra],
        "detail": detail,
    }


def resolve_run(
    args: argparse.Namespace, gate_reg: dict[str, Any]
) -> tuple[dict[str, Any], list[dict], bool]:
    """Return ``(settings, combinations, registered)``.

    A registered run takes every parameter from ``prereg/prereg.yaml`` and
    refuses any CLI narrowing.  ``--smoke`` may narrow the run, but the report
    is marked smoke and can never be the registered pass.
    """

    narrowing = (
        args.blocks,
        args.bridge,
        args.n_loops,
        args.sampling_n,
        args.throughput_n,
        args.adapter_policy,
        args.max_new_tokens,
        args.round_size,
        args.max_batch,
        args.seed,
    )
    if not args.smoke:
        if any(value is not None for value in narrowing):
            raise SystemExit(
                "registered engine gate uses prereg/prereg.yaml; CLI "
                "narrowing requires --smoke (and a smoke run is never the "
                "registered pass)"
            )
        settings = {
            "size": gate_reg["size"],
            "quant": gate_reg["quant"],
            "prompts": gate_reg["prompts"],
            "max_new_tokens": gate_reg["max_new_tokens"],
            "round_size": gate_reg["round_size"],
            "max_batch": gate_reg["max_batch"],
            "seed": gate_reg["seed"],
            "token_limit": gate_reg["token_limit"],
            "near_tie_logit_gap": gate_reg["near_tie_logit_gap"],
            "min_exact_fraction": gate_reg["min_exact_fraction"],
            "sampling_n": list(gate_reg["sampling_n"]),
            "throughput_n": list(gate_reg["throughput_n"]),
            "logprob_mean_limit": gate_reg["logprob_mean_limit"],
            "logprob_max_limit": gate_reg["logprob_max_limit"],
            "comparison_prompts": gate_reg["comparison_prompts"],
        "refill_budgets": list(gate_reg["refill_budgets"]),
        "control_batch": gate_reg["control_batch"],
        "one_n1_key_per_block": gate_reg["one_n1_key_per_block"],
            "refill_budgets": list(gate_reg["refill_budgets"]),
            "control_batch": gate_reg["control_batch"],
            "one_n1_key_per_block": gate_reg["one_n1_key_per_block"],
            "long_max_new_tokens": gate_reg["long_max_new_tokens"],
            "long_round_size": gate_reg["long_round_size"],
            "long_round_1024_tokens": gate_reg["long_round_1024_tokens"],
        }
        return settings, list(gate_reg["combinations"]), True

    n_loops = _ints(args.n_loops or "1,2", "n-loops")
    blocks = [
        part.strip()
        for part in (args.blocks or DEFAULT_SMOKE_BLOCKS).split(",")
        if part.strip()
    ]
    bridge = args.bridge or "gch"
    policy = args.adapter_policy or "loop_only"
    combinations = [
        {
            "arm": "L",
            "block": block,
            "bridge": bridge,
            "adapter_policy": policy,
            "n_loops": n_loops,
        }
        for block in blocks
    ]
    settings = {
        "size": args.size or gate_reg["size"],
        "quant": args.quant or gate_reg["quant"],
        "prompts": args.prompts or gate_reg["prompts"],
        "max_new_tokens": args.max_new_tokens or gate_reg["max_new_tokens"],
        "round_size": args.round_size or gate_reg["round_size"],
        "max_batch": (
            gate_reg["max_batch"] if args.max_batch is None else args.max_batch
        ),
        "seed": gate_reg["seed"] if args.seed is None else args.seed,
        "token_limit": gate_reg["token_limit"],
        "near_tie_logit_gap": gate_reg["near_tie_logit_gap"],
        "min_exact_fraction": gate_reg["min_exact_fraction"],
        "sampling_n": _ints(args.sampling_n or "2", "sampling-n"),
        "throughput_n": _ints(args.throughput_n or "2", "throughput-n"),
        "logprob_mean_limit": gate_reg["logprob_mean_limit"],
        "logprob_max_limit": gate_reg["logprob_max_limit"],
        "comparison_prompts": gate_reg["comparison_prompts"],
        "refill_budgets": list(gate_reg["refill_budgets"]),
        "long_max_new_tokens": gate_reg["long_max_new_tokens"],
        "long_round_size": gate_reg["long_round_size"],
        "long_round_1024_tokens": gate_reg["long_round_1024_tokens"],
    }
    if settings["round_size"] >= settings["max_new_tokens"]:
        raise SystemExit(
            "smoke gate needs --round-size < --max-new-tokens so it exercises "
            "multi-round re-prefill"
        )
    return settings, combinations, False


# ---------------------------------------------------------------------------
# Pure gate logic (CPU-testable with fake GenOut objects)
# ---------------------------------------------------------------------------
def _comparison_key(output: Any) -> tuple:
    """Token ids when available, otherwise the decoded text contract."""
    token_ids = getattr(output, "token_ids", None)
    if token_ids is not None:
        return ("token_ids", tuple(int(token) for token in token_ids))
    return (
        "text",
        str(getattr(output, "text", "")),
        int(getattr(output, "completion_tokens", 0)),
        str(getattr(output, "finish_reason", "")),
    )


def _top2_gap(logits: Any) -> float:
    """fp32 top-1 minus top-2 logit gap at one position."""

    if isinstance(logits, dict):
        logits = logits.get("logits", logits)
    values = torch.as_tensor(logits, dtype=torch.float32).flatten()
    if values.numel() < 2:
        return float("inf")
    top2 = torch.topk(values, 2).values
    return float(top2[0].item() - top2[1].item())


def perturbed_top2_gap(logits: Any, noise: Any) -> float:
    """Top-2 gap of ``logits + noise`` (the Gumbel-perturbed scores)."""

    values = torch.as_tensor(logits, dtype=torch.float32).flatten()
    delta = torch.as_tensor(noise, dtype=torch.float32).flatten()
    if values.numel() != delta.numel() or values.numel() < 2:
        return float("inf")
    perturbed = values + delta
    top2 = torch.topk(perturbed, 2).values
    return float(top2[0].item() - top2[1].item())


def perturbed_gap(
    logits: Any,
    *,
    request_id: str,
    position: int,
    sampling: Sampling,
    seed: int,
    device: torch.device | str = "cpu",
) -> float:
    """Replay the request's per-step Gumbel draw and return the perturbed gap."""

    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    scores = torch.as_tensor(logits, dtype=torch.float32).flatten().unsqueeze(0)
    scores = TemperatureLogitsWarper(sampling.temperature)(
        None, scores
    )
    scores = TopKLogitsWarper(sampling.top_k)(None, scores)
    scores = TopPLogitsWarper(sampling.top_p)(None, scores)
    vocab = int(scores.shape[-1])
    generator = torch.Generator(device=device).manual_seed(
        generate_request_seed(seed, request_id) % (2**63 - 1)
    )
    for _ in range(int(position)):
        torch.rand(
            vocab, generator=generator, device=device, dtype=torch.float32
        )
    uniform = torch.rand(
        vocab, generator=generator, device=device, dtype=torch.float32
    ).clamp_min(torch.finfo(torch.float32).tiny)
    noise = -torch.log(-torch.log(uniform))
    return perturbed_top2_gap(scores[0], noise)


def generate_request_seed(seed: int, request_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _first_divergence(
    padded_ids: Sequence[int],
    refill_ids: Sequence[int],
    token_limit: int,
) -> int | None:
    """First differing position within the first ``token_limit`` tokens.

    A shorter output diverges at its length (the other engine continued).
    """

    compare = min(len(padded_ids), len(refill_ids), token_limit)
    for position in range(compare):
        if int(padded_ids[position]) != int(refill_ids[position]):
            return position
    if min(len(padded_ids), len(refill_ids)) >= token_limit:
        return None
    if len(padded_ids) != len(refill_ids):
        return compare
    return None


def equality_criterion(
    name: str,
    padded: Sequence[Any],
    refill: Sequence[Any],
    request_ids: Sequence[str],
    *,
    conversations: Sequence[Any] | None = None,
    sampling: Sampling | None = None,
    logits_fn: Callable[..., Any] | None = None,
    gap_fn: Callable[..., float] | None = None,
    token_limit: int = 512,
    token_limits: Sequence[int] | None = None,
    near_tie_gap: float = 0.25,
    min_exact_fraction: float = 0.90,
) -> dict:
    """Ruling-compliant equality: near-ties allowed, non-ties fail.

    At a first divergent position the caller supplies ``logits_fn(engine,
    conversation, prefix_ids)``, which recomputes that engine's logits for the
    shared prefix in fp32.  A divergence is acceptable only when both engines'
    fp32 top-2 gap is below ``near_tie_gap``.  The criterion also requires at
    least ``min_exact_fraction`` of prompts to be token-exact for the first
    ``token_limit`` generated tokens.  Every divergence is recorded.
    """

    requests = len(request_ids)
    if len(padded) != requests or len(refill) != requests:
        return {
            "criterion": str(name),
            "passed": False,
            "requests": requests,
            "exact": 0,
            "exact_fraction": 0.0,
            "divergences": [],
            "valid": False,
            "detail": (
                f"engine output counts differ: padded={len(padded)} "
                f"refill={len(refill)} requests={requests}"
            ),
        }
    if requests == 0:
        return {
            "criterion": str(name),
            "passed": False,
            "requests": 0,
            "exact": 0,
            "exact_fraction": 0.0,
            "divergences": [],
            "valid": False,
            "detail": "no requests",
        }

    exact = 0
    divergences: list[dict] = []
    compared_by = "token_ids"
    for index in range(requests):
        padded_output = padded[index]
        refill_output = refill[index]
        padded_ids = getattr(padded_output, "token_ids", None)
        refill_ids = getattr(refill_output, "token_ids", None)
        if padded_ids is None or refill_ids is None:
            compared_by = "text"
            if _comparison_key(padded_output) == _comparison_key(
                refill_output
            ):
                exact += 1
                continue
            divergences.append(
                {
                    "request_id": str(request_ids[index]),
                    "position": None,
                    "padded_token": None,
                    "refill_token": None,
                    "padded_gap": None,
                    "refill_gap": None,
                    "near_tie": False,
                }
            )
            continue

        limit = (
            int(token_limits[index])
            if token_limits is not None
            else int(token_limit)
        )
        position = _first_divergence(padded_ids, refill_ids, limit)
        if position is None:
            exact += 1
            continue
        prefix = [int(token) for token in padded_ids[:position]]
        record = {
            "request_id": str(request_ids[index]),
            "position": int(position),
            "padded_token": (
                int(padded_ids[position])
                if position < len(padded_ids)
                else None
            ),
            "refill_token": (
                int(refill_ids[position])
                if position < len(refill_ids)
                else None
            ),
            "padded_gap": None,
            "refill_gap": None,
            "near_tie": False,
        }
        if gap_fn is not None:
            padded_gap = float(
                gap_fn(
                    "padded",
                    conversations[index] if conversations is not None else None,
                    prefix,
                    position,
                    request_ids[index],
                )
            )
            refill_gap = float(
                gap_fn(
                    "refill",
                    conversations[index] if conversations is not None else None,
                    prefix,
                    position,
                    request_ids[index],
                )
            )
            record["padded_gap"] = padded_gap
            record["refill_gap"] = refill_gap
            record["near_tie"] = bool(
                padded_gap < near_tie_gap and refill_gap < near_tie_gap
            )
        elif (
            logits_fn is not None
            and conversations is not None
            and sampling is not None
        ):
            padded_logits = logits_fn(
                "padded", conversations[index], prefix
            )
            refill_logits = logits_fn(
                "refill", conversations[index], prefix
            )
            padded_gap = _top2_gap(padded_logits)
            refill_gap = _top2_gap(refill_logits)
            record["padded_gap"] = padded_gap
            record["refill_gap"] = refill_gap
            record["near_tie"] = bool(
                padded_gap < near_tie_gap and refill_gap < near_tie_gap
            )
        divergences.append(record)

    exact_fraction = exact / requests
    near_ties_only = all(row["near_tie"] for row in divergences)
    passed = bool(
        near_ties_only and exact_fraction >= float(min_exact_fraction)
    )
    detail = (
        f"{exact}/{requests} token-exact ({compared_by}, first "
        f"{token_limit} tokens; fraction {exact_fraction:.3f} >= "
        f"{min_exact_fraction:.2f}), {len(divergences)} divergence(s), "
        f"{sum(row['near_tie'] for row in divergences)} near-tie"
    )
    if not passed:
        if not near_ties_only:
            bad = [
                f"{row['request_id']}@{row['position']}: "
                f"{row['padded_token']} vs {row['refill_token']} "
                f"(gaps {row['padded_gap']}, {row['refill_gap']})"
                for row in divergences
                if not row["near_tie"]
            ]
            detail += "; non-tie divergences: " + ", ".join(bad)
        else:
            detail += "; below the token-exactness threshold"
    return {
        "criterion": str(name),
        "passed": bool(passed),
        "requests": requests,
        "exact": exact,
        "exact_fraction": round(exact_fraction, 6),
        "compared_by": compared_by,
        "near_ties_only": near_ties_only,
        "divergences": divergences,
        "mismatched_ids": [row["request_id"] for row in divergences],
        "near_tie_gap": float(near_tie_gap),
        "min_exact_fraction": float(min_exact_fraction),
        "valid": True,
        "detail": detail,
    }


def apply_noise_floor(
    refill: dict, control: dict
) -> dict:
    """Gate refill statistics relative to the padded-vs-padded control.

    Ruling v3: ``exact_fraction >= control - 0.10``,
    ``drift_mean <= 2*control + 0.005``, ``drift_max <= 2*control + 0.05``,
    and every refill divergence must be a near-tie.  The control statistics are
    copied into the JSON.
    """

    control_valid = bool(control.get("valid", False))
    refill_valid = bool(refill.get("valid", False))
    control_exact = int(control.get("exact", 0))
    control_prompts = int(control.get("requests", 0))
    refill_exact = int(refill.get("exact", 0))
    refill_prompts = int(refill.get("requests", 0))
    near_ties = bool(refill.get("near_ties_only", False))
    exact_ok = refill_exact >= control_exact - 1
    passed = bool(
        control_valid
        and refill_valid
        and control_prompts > 0
        and refill_prompts > 0
        and near_ties
        and exact_ok
    )
    detail = (
        f"refill exact prompts {refill_exact}/{refill_prompts} vs control "
        f"{control_exact}/{control_prompts} (>= control-1); "
        f"near-ties={near_ties}; refill_valid={refill_valid}, "
        f"control_valid={control_valid}"
    )
    return {
        **refill,
        "passed": passed,
        "control": {
            "exact": control_exact,
            "requests": control_prompts,
            "exact_fraction": (
                control_exact / control_prompts if control_prompts else 0.0
            ),
            "valid": control_valid,
        },
        "noise_floor": {
            "exact_ok": exact_ok,
            "near_ties_only": near_ties,
            "valid": bool(control_valid and refill_valid),
        },
        "detail": detail,
    }


def apply_logprob_noise_floor(refill: dict, control: dict) -> dict:
    """Drift bound relative to the padded-vs-padded control."""

    control_mean = float(control.get("mean_abs_delta") or 0.0)
    control_max = float(control.get("max_abs_delta") or 0.0)
    refill_mean = float(refill.get("mean_abs_delta") or 0.0)
    refill_max = float(refill.get("max_abs_delta") or 0.0)
    steps = int(refill.get("steps", 0))
    mean_ok = refill_mean <= 2.0 * control_mean + 0.005
    max_ok = refill_max <= 2.0 * control_max + 0.05
    valid = bool(
        refill.get("records_complete", False)
        and control.get("records_complete", False)
        and steps > 0
    )
    passed = bool(valid and mean_ok and max_ok)
    return {
        **refill,
        "passed": passed,
        "control": {
            "mean_abs_delta": control_mean,
            "max_abs_delta": control_max,
        },
        "noise_floor": {
            "mean_ok": mean_ok,
            "max_ok": max_ok,
            "valid": valid,
        },
        "valid": valid,
        "detail": (
            f"drift mean {refill_mean:.6f} vs 2*{control_mean:.6f}+0.005; "
            f"max {refill_max:.6f} vs 2*{control_max:.6f}+0.05"
        ),
    }


def _run_padded_budgeted(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    request_ids: list[str],
    sampling: Sampling,
    budgets: Sequence[int],
    *,
    seed: int,
    max_batch: int,
    round_size: int,
    record_top20: bool,
) -> list[Any]:
    """Padded reference with each request stopped at its own budget."""

    outputs: list[Any] = [None] * len(conversations)
    for budget in sorted({int(value) for value in budgets}, reverse=True):
        indices = [
            index
            for index, value in enumerate(budgets)
            if int(value) == budget
        ]
        group_sampling = replace(sampling, max_new_tokens=budget)
        group, _ = run_engine(
            "padded",
            model,
            processor,
            [conversations[index] for index in indices],
            [request_ids[index] for index in indices],
            group_sampling,
            seed=seed,
            max_batch=max_batch,
            round_size=round_size,
            record_top20=record_top20,
        )
        for index, output in zip(indices, group):
            outputs[index] = output
    return outputs


def budget_groups(budgets: Sequence[int]) -> list[list[int]]:
    """Reference padded chunks: one group per budget, original order."""

    groups: list[list[int]] = []
    for budget in sorted({int(value) for value in budgets}, reverse=True):
        groups.append(
            [
                index
                for index, value in enumerate(budgets)
                if int(value) == budget
            ]
        )
    return groups


def interleaved_order(count: int) -> list[int]:
    """Control order: interleave halves so every prompt gets new neighbours."""

    half = (count + 1) // 2
    first = list(range(min(half, count)))
    second = list(range(half, count))
    order: list[int] = []
    for left, right in zip(first, second):
        order.extend([left, right])
    order.extend(first[len(second):])
    return order


def _run_padded_control(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    request_ids: list[str],
    sampling: Sampling,
    *,
    seed: int,
    control_batch: int,
    round_size: int,
    budgets: Sequence[int] | None = None,
) -> list[Any]:
    """Padded control with genuinely different chunks and padding lengths."""

    order = interleaved_order(len(conversations))
    control_ids = [request_ids[index] for index in order]
    max_budget = max(int(value) for value in budgets) if budgets else None
    control_sampling = (
        replace(sampling, max_new_tokens=max_budget)
        if max_budget is not None
        else sampling
    )
    outputs, _ = run_engine(
        "padded",
        model,
        processor,
        [conversations[index] for index in order],
        control_ids,
        control_sampling,
        seed=seed,
        max_batch=max(1, int(control_batch)),
        round_size=round_size,
        record_top20=True,
    )
    by_id = dict(zip(control_ids, outputs))
    return [by_id[request_id] for request_id in request_ids]


def _common_prefix_length(
    left: Sequence[int], right: Sequence[int], token_limit: int
) -> int:
    compare = min(len(left), len(right), int(token_limit))
    for position in range(compare):
        if int(left[position]) != int(right[position]):
            return position
    return compare


def logprob_criterion(
    name: str,
    padded: Sequence[Any],
    refill: Sequence[Any],
    request_ids: Sequence[str],
    *,
    token_limit: int = 512,
    token_limits: Sequence[int] | None = None,
    mean_limit: float = 0.01,
    max_limit: float = 0.1,
) -> dict:
    """Bug detector over the token-exact prefix of every request.

    Both engines must have recorded top-20 logprobs and the chosen-token
    logprob at every generated step.  Over the identical leading tokens, the
    chosen-token ``|delta logprob|`` must have mean <= 0.01 and max <= 0.1 per
    combination; otherwise the gate fails even if every token is identical.
    """

    deltas: list[float] = []
    per_request: list[dict] = []
    missing: list[str] = []
    steps = 0
    for index, request_id in enumerate(request_ids):
        if index >= len(padded) or index >= len(refill):
            missing.append(str(request_id))
            continue
        left = padded[index]
        right = refill[index]
        left_chosen = getattr(left, "chosen_logprobs", None)
        right_chosen = getattr(right, "chosen_logprobs", None)
        left_top20 = getattr(left, "top20_logprobs", None)
        right_top20 = getattr(right, "top20_logprobs", None)
        left_ids = getattr(left, "token_ids", None)
        right_ids = getattr(right, "token_ids", None)
        if (
            left_chosen is None
            or right_chosen is None
            or left_top20 is None
            or right_top20 is None
            or left_ids is None
            or right_ids is None
        ):
            missing.append(str(request_id))
            continue
        limit = (
            int(token_limits[index])
            if token_limits is not None
            else int(token_limit)
        )
        prefix = _common_prefix_length(left_ids, right_ids, limit)
        request_deltas = [
            abs(float(left_chosen[step]) - float(right_chosen[step]))
            for step in range(min(prefix, len(left_chosen), len(right_chosen)))
        ]
        steps += len(request_deltas)
        deltas.extend(request_deltas)
        if request_deltas:
            per_request.append(
                {
                    "request_id": str(request_id),
                    "steps": len(request_deltas),
                    "mean_abs_delta": sum(request_deltas)
                    / len(request_deltas),
                    "max_abs_delta": max(request_deltas),
                }
            )

    if missing:
        return {
            "criterion": str(name),
            "passed": False,
            "steps": steps,
            "mean_abs_delta": None,
            "max_abs_delta": None,
            "missing_logprobs": missing,
            "records_complete": False,
            "valid": False,
            "detail": (
                "missing top-20/chosen logprobs for requests: "
                + ", ".join(missing[:8])
            ),
        }
    if not deltas:
        return {
            "criterion": str(name),
            "passed": False,
            "steps": 0,
            "mean_abs_delta": None,
            "max_abs_delta": None,
            "missing_logprobs": [],
            "records_complete": False,
            "valid": False,
            "detail": "no token-exact steps to compare",
        }
    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    passed = bool(
        mean_delta <= float(mean_limit) and max_delta <= float(max_limit)
    )
    return {
        "criterion": str(name),
        "passed": passed,
        "steps": steps,
        "requests": len(per_request),
        "mean_abs_delta": mean_delta,
        "max_abs_delta": max_delta,
        "mean_limit": float(mean_limit),
        "max_limit": float(max_limit),
        "records_complete": True,
        "valid": True,
        "worst_requests": sorted(
            per_request, key=lambda row: row["max_abs_delta"], reverse=True
        )[:5],
        "detail": (
            f"{steps} token-exact steps; mean |delta logprob| "
            f"{mean_delta:.6f} <= {mean_limit}, max {max_delta:.6f} "
            f"<= {max_limit}"
        ),
    }


def engine_stats(engine: str, outputs: Sequence[Any], elapsed_s: float) -> dict:
    """Completion-token throughput for one engine run."""

    tokens = sum(int(getattr(output, "completion_tokens", 0)) for output in outputs)
    requests = len(outputs)
    return {
        "engine": str(engine),
        "requests": requests,
        "tokens": tokens,
        "elapsed_s": round(float(elapsed_s), 3),
        "tok_s": round(tokens / elapsed_s, 3) if elapsed_s > 0 else 0.0,
    }


def rep_refill_criterion(name: str, heartbeat: dict) -> dict:
    """Require non-vacuous multi-round re-prefill work.

    The registered gate must cross several decode rounds and must reuse at
    least one freed slot for a queued request.  A single-round run (round size
    >= generation length) or a run that never admits a late request cannot
    establish engine equivalence for continuing requests.
    """

    rounds = int(heartbeat.get("rounds", 0))
    rep_refills = int(heartbeat.get("rep_refills", 0))
    late_admissions = int(heartbeat.get("late_admissions", 0))
    mixed_histories = bool(heartbeat.get("mixed_histories", False))
    passed = (
        rounds >= 2
        and rep_refills > 0
        and late_admissions > 0
        and mixed_histories
    )
    return {
        "criterion": str(name),
        "passed": bool(passed),
        "rounds": rounds,
        "rep_refills": rep_refills,
        "late_admissions": late_admissions,
        "mixed_histories": mixed_histories,
        "detail": (
            f"rounds={rounds}, re-prefills={rep_refills}, "
            f"late admissions={late_admissions}, "
            f"mixed histories={mixed_histories}"
        ),
    }


def throughput_criterion(
    name: str,
    padded: dict,
    refill: dict,
    expected_requests: int,
) -> dict:
    """PASS when both engines measured the full prompt set with positive tok/s."""

    measured = (
        int(padded["requests"]) == expected_requests
        and int(refill["requests"]) == expected_requests
        and float(padded["tok_s"]) > 0
        and float(refill["tok_s"]) > 0
    )
    return {
        "criterion": str(name),
        "passed": bool(measured),
        "informational": True,
        "requests": expected_requests,
        "padded_tok_s": float(padded["tok_s"]),
        "refill_tok_s": float(refill["tok_s"]),
        "detail": (
            f"padded {padded['tok_s']:.1f} tok/s; "
            f"refill {refill['tok_s']:.1f} tok/s"
            + ("" if measured else "; incomplete or zero measurement")
        ),
    }


def pooled_exactness_criterion(
    criteria: Iterable[dict], minimum: float = 0.05
) -> dict:
    """Pooled refill token-exact fraction must be >= pooled control - 0.05."""

    refill_exact = refill_total = control_exact = control_total = 0
    for row in criteria:
        control = row.get("control") or {}
        if "exact" not in row or "requests" not in row:
            continue
        refill_exact += int(row["exact"])
        refill_total += int(row["requests"])
        control_exact += int(control.get("exact", 0))
        control_total += int(control.get("requests", 0))
    refill_fraction = refill_exact / refill_total if refill_total else 0.0
    control_fraction = control_exact / control_total if control_total else 0.0
    valid = refill_total > 0 and control_total > 0
    passed = bool(valid and refill_fraction >= control_fraction - minimum)
    return {
        "criterion": "pooled_exactness",
        "passed": passed,
        "refill_exact": refill_exact,
        "refill_prompts": refill_total,
        "control_exact": control_exact,
        "control_prompts": control_total,
        "refill_fraction": refill_fraction,
        "control_fraction": control_fraction,
        "valid": valid,
        "detail": (
            f"pooled refill {refill_fraction:.4f} >= control "
            f"{control_fraction:.4f} - {minimum}"
        ),
    }


def decide_gate(criteria: Iterable[dict]) -> bool:
    """Exit-code decision: every non-informational criterion must pass."""

    criteria = [row for row in criteria if not row.get("informational")]
    return bool(criteria) and all(bool(row.get("passed")) for row in criteria)


def build_report(
    *,
    criteria: Sequence[dict],
    throughput: Sequence[dict],
    config: dict,
    smoke: bool = False,
) -> dict:
    """Assemble the JSON payload and overall decision.

    A smoke report can pass its own criteria but ``registered_pass`` is always
    false, so it can never be mistaken for the registered gate result.
    """

    passed = decide_gate(criteria)
    if smoke:
        overall = "SMOKE-PASS" if passed else "SMOKE-FAIL"
        registered_pass = False
    else:
        overall = "PASS" if passed else "FAIL"
        registered_pass = passed
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "passed": passed,
        "registered_pass": registered_pass,
        "smoke": bool(smoke),
        "config": config,
        "criteria": list(criteria),
        "throughput": list(throughput),
    }


def render_markdown(report: dict) -> str:
    """Markdown report with PASS/FAIL per criterion and the tok/s table."""

    lines = [
        "# Engine gate (refill vs padded)",
        "",
        f"**Overall: {report['overall']}**",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Criteria",
        "",
        "| criterion | result | detail |",
        "| --- | --- | --- |",
    ]
    for row in report.get("criteria", []):
        result = (
            "INFO" if row.get("informational")
            else ("PASS" if row["passed"] else "FAIL")
        )
        lines.append(
            f"| {row['criterion']} | {result} | "
            f"{row.get('detail', '')} |"
        )
    lines.extend(
        [
            "",
            "## Throughput (registered sampling, same prompts)",
            "",
            "| block | N | engine | requests | tokens | wall (s) | tok/s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("throughput", []):
        lines.append(
            f"| {row.get('block', '')} | {row.get('n_loops', '')} | "
            f"{row['engine']} | {row['requests']} | {row['tokens']} | "
            f"{row['elapsed_s']:.1f} | {row['tok_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict, output: Path) -> Path:
    """Write JSON atomically and the matching Markdown file."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    markdown = output.with_suffix(".md")
    markdown_tmp = markdown.with_name(
        f"{markdown.name}.{time.time_ns()}.tmp"
    )
    markdown_tmp.write_text(render_markdown(report), encoding="utf-8")
    markdown_tmp.replace(markdown)
    return output


# ---------------------------------------------------------------------------
# GPU harness
# ---------------------------------------------------------------------------
def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_engine(
    engine: str,
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    request_ids: list[str],
    sampling: Sampling,
    *,
    seed: int,
    max_batch: int,
    round_size: int,
    heartbeat: Callable[[dict], None] | None = None,
    record_top20: bool = False,
    budgets: Sequence[int] | None = None,
) -> tuple[list[Any], float]:
    """Run one engine once and return ``(outputs, elapsed_seconds)``."""

    started = time.perf_counter()
    if engine == "padded":
        outputs = generate_batch(
            model,
            processor,
            conversations,
            sampling,
            seed=seed,
            max_batch=max_batch,
            request_ids=request_ids,
            record_top20=record_top20,
        )
    elif engine == "refill":
        outputs = generate_batch_refill(
            model,
            processor,
            conversations,
            sampling,
            seed=seed,
            max_batch=max_batch,
            round_size=round_size,
            request_ids=request_ids,
            heartbeat=heartbeat,
            record_top20=record_top20,
            budgets=list(budgets) if budgets is not None else None,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown engine {engine!r}")
    _synchronize()
    return list(outputs), time.perf_counter() - started


def _seed_for(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _text_model(model: Any) -> Any:
    base = model
    get_base = getattr(base, "get_base_model", None)
    if callable(get_base):
        base = get_base()
    inner = getattr(base, "model", None)
    return base if inner is None else getattr(inner, "language_model", inner)


def seed_gate_parameters(
    model: Any,
    *,
    arm: str,
    bridge: str | None,
    plan: Any,
    label: str,
    base_seed: int,
) -> Any:
    """Give gate models seeded nonzero adapters and bridge parameters.

    Zero-init bridges and zero LoRA B matrices make per-pass adapter or bridge
    bugs invisible.  Every L/O-H combination therefore gets deterministic
    nonzero weights before either engine runs.
    """

    seed = _seed_for(label, base_seed)
    if arm in ("L", "O-H"):
        from peft import LoraConfig, get_peft_model

        from gemmaloop.model.load import lora_target_modules

        target = (
            lora_target_modules("loop_block", plan)
            if arm == "L"
            else lora_target_modules("language")
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
                target_modules=target,
            ),
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    values = torch.empty(
                        parameter.shape, dtype=torch.float32
                    ).normal_(mean=0.0, std=LORA_B_STD, generator=generator)
                    parameter.copy_(
                        values.to(device=parameter.device, dtype=parameter.dtype)
                    )

    if arm == "L":
        text_model = _text_model(model)
        generator = torch.Generator().manual_seed(seed ^ 0x5A5A)
        with torch.no_grad():
            g = getattr(text_model, "loop_bridge_g", None)
            if g is not None:
                values = torch.empty(g.shape, dtype=torch.float32).uniform_(
                    BRIDGE_G_LOW, BRIDGE_G_HIGH, generator=generator
                )
                g.copy_(values.to(device=g.device, dtype=g.dtype))
            logit = getattr(text_model, "loop_bridge_logit", None)
            if logit is not None:
                w = float(
                    torch.empty(1, dtype=torch.float32).uniform_(
                        BRIDGE_G_LOW, BRIDGE_G_HIGH, generator=generator
                    )[0]
                )
                logit.fill_(math.log(w / (1.0 - w)))
    return model


ENGINE_SOURCE_FILES = (
    "scripts/engine_gate.py",
    "src/gemmaloop/serve/generate.py",
    "src/gemmaloop/serve/refill.py",
    "src/gemmaloop/model/looped.py",
    "src/gemmaloop/serve/cache.py",
    "prereg/prereg.yaml",
)


def source_provenance(prereg_path: str | Path) -> dict[str, Any]:
    """Git commit, dirty flag and sha256 of the gate/engine sources."""

    import subprocess

    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    hashes: dict[str, str] = {}
    files = list(ENGINE_SOURCE_FILES)
    files[0] = "scripts/engine_gate.py"
    files[-1] = str(Path(prereg_path).resolve())
    for relative in files[:-1]:
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    prereg = Path(prereg_path)
    if not prereg.is_absolute():
        prereg = root / prereg
    hashes["prereg"] = hashlib.sha256(prereg.read_bytes()).hexdigest()
    return {"git_commit": commit, "dirty": dirty, "sha256": hashes}


def enforce_registered_provenance(provenance: dict, *, registered: bool) -> None:
    if registered and provenance.get("dirty"):
        raise SystemExit(
            "engine_gate: refusing a registered run on a dirty worktree; "
            "commit or stash before the gate"
        )


def _sampling(settings: dict) -> Sampling:
    return Sampling(
        temperature=REGISTERED_TEMPERATURE,
        top_p=REGISTERED_TOP_P,
        top_k=REGISTERED_TOP_K,
        max_new_tokens=int(settings["max_new_tokens"]),
    )


def recompute_engine_logits(
    model: Any,
    processor: Any,
    conversation: list[dict],
    prefix_ids: Sequence[int],
    sampling: Sampling,
    engine: str = "padded",
) -> torch.Tensor:
    """Recompute one engine's next-token logits in fp32 for a shared prefix.

    The transformer forward follows the engine's own path (the caller supplies
    that engine), but the final LM head is always applied in fp32 with
    vocab-sized chunks so a bf16 head cannot hide a near-tie.  Gemma's final
    logit softcap is applied before returning, matching generation.
    """

    from gemmaloop.eval.calibrated_nll import _text_model_and_head
    from gemmaloop.serve.generate import _model_device, _render
    from gemmaloop.serve.refill import _prompt_ids

    rendered = _render(processor, conversation, sampling, None)
    prompt_ids, _mm = _prompt_ids(rendered)
    ids = [int(token) for token in prompt_ids] + [
        int(token) for token in prefix_ids
    ]
    device = _model_device(model)
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones(
            1, len(ids), dtype=torch.long, device=device
        ),
        "mm_token_type_ids": torch.zeros(
            1, len(ids), dtype=torch.long, device=device
        ),
    }
    with torch.no_grad():
        # logits_to_keep=1: the HF head would otherwise materialise
        # [T, vocab] logits (2.8 GiB at ~2.7k tokens) that are never used;
        # the fp32 head below recomputes the last position only.
        output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        hidden = hidden_states[-1][0, -1]
    else:  # pragma: no cover - defensive for wrappers without hidden states
        raise RuntimeError(
            "near-tie recompute needs output_hidden_states=True"
        )
    _text_model, weight, softcap = _text_model_and_head(model)
    head_device = weight.device
    hidden = hidden.to(device=head_device, dtype=torch.float32)
    vocab = int(weight.shape[0])
    chunks: list[torch.Tensor] = []
    for v0 in range(0, vocab, 8192):
        v1 = min(v0 + 8192, vocab)
        tile = weight[v0:v1].to(device=head_device, dtype=torch.float32)
        logits = hidden @ tile.t()
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        chunks.append(logits)
    return torch.cat(chunks)


def run_group(
    group_key: tuple[str, str | None, str | None, str | None],
    n_values: Sequence[int],
    prompts: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[list[dict], list[dict], set[tuple]]:
    """Run every criterion for one (arm, block, bridge, policy) group."""

    arm, block_text, bridge, policy = group_key
    loop = None
    if arm == "L":
        loop = LoopConfig(
            n_loops=max(int(value) for value in n_values),
            bridge=bridge or "scalar",
            block=parse_block(block_text),
            adapter_policy=policy,
        )
    model = processor = None
    criteria: list[dict] = []
    throughput: list[dict] = []
    observed: set[tuple] = set()
    try:
        model, processor, plan = load_model(
            ModelSpec(
                size=settings["size"],
                quant=settings["quant"],
                loop=loop,
            )
        )
        conversations = [prompt["messages"] for prompt in prompts["prompts"]]
        request_ids = [str(prompt["id"]) for prompt in prompts["prompts"]]
        by_id = dict(zip(request_ids, conversations))
        # The registered v2 gate compares 16 prompts per combination (8 LCB
        # + 8 math) to keep the run inside the 4 h budget.
        comparison_requests = list(
            prompts["sampling_ids"][: int(settings["comparison_prompts"])]
        )
        comparison_conversations = [
            by_id[prompt_id] for prompt_id in comparison_requests
        ]
        max_batch = int(settings["max_batch"]) or max_batch_for(
            int(settings["max_new_tokens"]), settings["size"]
        )
        label = (
            f"arm={arm}_block={block_text or 'none'}_"
            f"bridge={bridge or 'none'}_policy={policy or 'none'}"
        )
        model = seed_gate_parameters(
            model,
            arm=arm,
            bridge=bridge,
            plan=plan,
            label=label,
            base_seed=int(settings["seed"]),
        )

        for n_loops in sorted(int(value) for value in n_values):
            observed.add(
                _combo_key(arm, block_text, bridge, policy, n_loops)
            )
            if loop is not None:
                text_model = _text_model(model)
                set_n_loops(text_model, n_loops)
                if int(getattr(text_model, "_gemmaloop_n_loops", -1)) != int(
                    n_loops
                ):
                    raise RuntimeError(
                        f"gate: requested N={n_loops} but the text model runs "
                        f"N={getattr(text_model, '_gemmaloop_n_loops', None)}"
                    )
            greedy = Sampling(
                max_new_tokens=int(settings["max_new_tokens"]),
                temperature=0.0,
            )
            budgets = [
                int(value)
                for value in settings["refill_budgets"][
                    : len(comparison_requests)
                ]
            ]
            padded_started = time.perf_counter()
            padded = _run_padded_budgeted(
                model,
                processor,
                comparison_conversations,
                comparison_requests,
                greedy,
                budgets,
                seed=int(settings["seed"]),
                max_batch=max_batch,
                round_size=int(settings["round_size"]),
                record_top20=True,
            )
            padded_elapsed = time.perf_counter() - padded_started
            refill_stats: dict = {}
            refill, refill_elapsed = run_engine(
                "refill",
                model,
                processor,
                comparison_conversations,
                comparison_requests,
                greedy,
                seed=int(settings["seed"]),
                max_batch=max_batch,
                round_size=int(settings["round_size"]),
                heartbeat=refill_stats.update,
                record_top20=True,
                budgets=budgets,
            )
            control = _run_padded_control(
                model,
                processor,
                comparison_conversations,
                comparison_requests,
                greedy,
                seed=int(settings["seed"]),
                control_batch=int(settings["control_batch"]),
                round_size=int(settings["round_size"]),
                budgets=budgets,
            )
            control_eq = equality_criterion(
                f"control_greedy_{label}_n={n_loops}",
                padded,
                control,
                comparison_requests,
                token_limit=int(settings["token_limit"]),
                token_limits=budgets,
                near_tie_gap=float(settings["near_tie_logit_gap"]),
                min_exact_fraction=0.0,
            )
            control_lp = logprob_criterion(
                f"control_logprobs_{label}_n={n_loops}",
                padded,
                control,
                comparison_requests,
                token_limit=int(settings["token_limit"]),
                token_limits=budgets,
                mean_limit=float(settings["logprob_mean_limit"]),
                max_limit=float(settings["logprob_max_limit"]),
            )
            refill_eq = equality_criterion(
                f"greedy_{label}_n={n_loops}",
                padded,
                refill,
                comparison_requests,
                conversations=comparison_conversations,
                sampling=greedy,
                logits_fn=lambda engine, conversation, prefix: (
                    recompute_engine_logits(
                        model,
                        processor,
                        conversation,
                        prefix,
                        greedy,
                        engine,
                    )
                ),
                token_limit=int(settings["token_limit"]),
                token_limits=budgets,
                near_tie_gap=float(settings["near_tie_logit_gap"]),
                min_exact_fraction=0.0,
            )
            refill_lp = logprob_criterion(
                f"logprobs_{label}_n={n_loops}",
                padded,
                refill,
                comparison_requests,
                token_limit=int(settings["token_limit"]),
                token_limits=budgets,
                mean_limit=float(settings["logprob_mean_limit"]),
                max_limit=float(settings["logprob_max_limit"]),
            )
            criteria.append(apply_noise_floor(refill_eq, control_eq))
            criteria.append(apply_logprob_noise_floor(refill_lp, control_lp))
            criteria.append(
                rep_refill_criterion(
                    f"rep_refills_{label}_n={n_loops}",
                    refill_stats,
                )
            )
            if n_loops in settings["throughput_n"]:
                padded_stats = {
                    **engine_stats("padded", padded, padded_elapsed),
                    "arm": arm,
                    "block": block_text,
                    "bridge": bridge,
                    "adapter_policy": policy,
                    "n_loops": n_loops,
                }
                refill_stats_row = {
                    **engine_stats("refill", refill, refill_elapsed),
                    "arm": arm,
                    "block": block_text,
                    "bridge": bridge,
                    "adapter_policy": policy,
                    "n_loops": n_loops,
                }
                throughput.extend([padded_stats, refill_stats_row])
                criteria.append(
                    throughput_criterion(
                        f"throughput_{label}_n={n_loops}",
                        padded_stats,
                        refill_stats_row,
                        len(comparison_requests),
                    )
                )
            del padded, refill
            gc.collect()

            sampling = _sampling(settings)
            if n_loops in settings["sampling_n"]:
                sampled_padded, _ = run_engine(
                    "padded",
                    model,
                    processor,
                    comparison_conversations,
                    comparison_requests,
                    sampling,
                    seed=int(settings["seed"]),
                    max_batch=max_batch,
                    round_size=int(settings["round_size"]),
                    record_top20=True,
                )
                sampled_refill, _ = run_engine(
                    "refill",
                    model,
                    processor,
                    comparison_conversations,
                    comparison_requests,
                    sampling,
                    seed=int(settings["seed"]),
                    max_batch=max_batch,
                    round_size=int(settings["round_size"]),
                    record_top20=True,
                )
                control = _run_padded_control(
                    model,
                    processor,
                    comparison_conversations,
                    comparison_requests,
                    sampling,
                    seed=int(settings["seed"]),
                    control_batch=int(settings["control_batch"]),
                    round_size=int(settings["round_size"]),
                )
                control_eq = equality_criterion(
                    f"control_sampled_{label}_n={n_loops}",
                    sampled_padded,
                    control,
                    comparison_requests,
                    token_limit=int(settings["token_limit"]),
                    near_tie_gap=float(settings["near_tie_logit_gap"]),
                    min_exact_fraction=0.0,
                )
                control_lp = logprob_criterion(
                    f"control_logprobs_sampled_{label}_n={n_loops}",
                    sampled_padded,
                    control,
                    comparison_requests,
                    token_limit=int(settings["token_limit"]),
                    mean_limit=LOGPROB_MEAN_LIMIT,
                    max_limit=LOGPROB_MAX_LIMIT,
                )
                sampled_eq = equality_criterion(
                    f"sampled_{label}_n={n_loops}",
                    sampled_padded,
                    sampled_refill,
                    comparison_requests,
                    conversations=comparison_conversations,
                    sampling=sampling,
                    gap_fn=lambda engine, conversation, prefix, position, request: (
                        perturbed_gap(
                            recompute_engine_logits(
                                model,
                                processor,
                                conversation,
                                prefix,
                                sampling,
                                engine,
                            ),
                            request_id=request,
                            position=position,
                            sampling=sampling,
                            seed=int(settings["seed"]),
                            device=_model_device(model),
                        )
                    ),
                    token_limit=int(settings["token_limit"]),
                    near_tie_gap=float(settings["near_tie_logit_gap"]),
                    min_exact_fraction=0.0,
                )
                sampled_lp = logprob_criterion(
                    f"logprobs_sampled_{label}_n={n_loops}",
                    sampled_padded,
                    sampled_refill,
                    comparison_requests,
                    token_limit=int(settings["token_limit"]),
                    mean_limit=LOGPROB_MEAN_LIMIT,
                    max_limit=LOGPROB_MAX_LIMIT,
                )
                criteria.append(apply_noise_floor(sampled_eq, control_eq))
                criteria.append(
                    apply_logprob_noise_floor(sampled_lp, control_lp)
                )
                del sampled_padded, sampled_refill
                gc.collect()

        return criteria, throughput, observed
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_coverage_extras(
    settings: dict[str, Any],
    prompts: dict[str, Any],
    combinations: Sequence[dict],
) -> list[dict]:
    """Run the Ruling v3 long/static-cache/round-1024 coverage additions."""

    from gemmaloop.serve.generate import _prompt_token_count, _render

    long_conversations = [
        prompt["messages"] for prompt in prompts["long_prompts"]
    ]
    long_ids = [str(prompt["id"]) for prompt in prompts["long_prompts"]]
    l_combo = next(
        combo for combo in combinations if combo["arm"] == "L"
    )
    groups = [
        ("O", None, None, None),
        (
            "L",
            l_combo["block"],
            l_combo["bridge"],
            l_combo["adapter_policy"],
        ),
    ]
    criteria: list[dict] = []
    for arm, block_text, bridge, policy in groups:
        loop = None
        if arm == "L":
            loop = LoopConfig(
                n_loops=2,
                bridge=bridge or "scalar",
                block=parse_block(block_text),
                adapter_policy=policy,
            )
        model = processor = None
        try:
            model, processor, plan = load_model(
                ModelSpec(
                    size=settings["size"],
                    quant=settings["quant"],
                    loop=loop,
                )
            )
            label = (
                f"arm={arm}_block={block_text or 'none'}_"
                f"bridge={bridge or 'none'}_policy={policy or 'none'}"
            )
            model = seed_gate_parameters(
                model,
                arm=arm,
                bridge=bridge,
                plan=plan,
                label=label + "_long",
                base_seed=int(settings["seed"]),
            )
            if loop is not None:
                set_n_loops(_text_model(model), 2)
            greedy = Sampling(
                max_new_tokens=int(settings["long_max_new_tokens"]),
                temperature=0.0,
            )
            rendered = _render(processor, long_conversations[0], greedy, None)
            prompt_tokens = _prompt_token_count(rendered)
            assert prompt_tokens > 4096, prompt_tokens
            padded, _ = run_engine(
                "padded",
                model,
                processor,
                long_conversations,
                long_ids,
                greedy,
                seed=int(settings["seed"]),
                max_batch=2,
                round_size=int(settings["long_round_size"]),
                record_top20=True,
            )
            refill, _ = run_engine(
                "refill",
                model,
                processor,
                long_conversations,
                long_ids,
                greedy,
                seed=int(settings["seed"]),
                max_batch=2,
                round_size=int(settings["long_round_size"]),
                record_top20=True,
            )
            row = equality_criterion(
                f"long_context_{label}",
                padded,
                refill,
                long_ids,
                conversations=long_conversations,
                sampling=greedy,
                logits_fn=lambda engine, conversation, prefix: (
                    recompute_engine_logits(
                        model,
                        processor,
                        conversation,
                        prefix,
                        greedy,
                        engine,
                    )
                ),
                token_limit=int(settings["long_max_new_tokens"]),
                near_tie_gap=float(settings["near_tie_logit_gap"]),
                min_exact_fraction=0.0,
            )
            row["prompt_tokens"] = int(prompt_tokens)
            row["static_cache"] = True
            criteria.append(row)
            criteria.append(
                logprob_criterion(
                    f"long_logprobs_{label}",
                    padded,
                    refill,
                    long_ids,
                    token_limit=int(settings["long_max_new_tokens"]),
                    mean_limit=LOGPROB_MEAN_LIMIT,
                    max_limit=LOGPROB_MAX_LIMIT,
                )
            )

            # One group at the registered production round size, >= 1,100
            # generated tokens.
            if arm == "L":
                heavy = Sampling(
                    max_new_tokens=int(settings["long_round_1024_tokens"]),
                    temperature=0.0,
                )
                heavy_conversations = long_conversations[:2]
                heavy_ids = long_ids[:2]
                padded, _ = run_engine(
                    "padded",
                    model,
                    processor,
                    heavy_conversations,
                    heavy_ids,
                    heavy,
                    seed=int(settings["seed"]),
                    max_batch=1,
                    round_size=int(settings["long_round_size"]),
                    record_top20=False,
                )
                refill, _ = run_engine(
                    "refill",
                    model,
                    processor,
                    heavy_conversations,
                    heavy_ids,
                    heavy,
                    seed=int(settings["seed"]),
                    max_batch=1,
                    round_size=1024,
                    record_top20=False,
                )
                row = equality_criterion(
                    f"round_1024_{label}",
                    padded,
                    refill,
                    heavy_ids,
                    conversations=heavy_conversations,
                    sampling=heavy,
                    logits_fn=lambda engine, conversation, prefix: (
                        recompute_engine_logits(
                            model,
                            processor,
                            conversation,
                            prefix,
                            heavy,
                            engine,
                        )
                    ),
                    token_limit=int(settings["long_round_1024_tokens"]),
                    near_tie_gap=float(settings["near_tie_logit_gap"]),
                    min_exact_fraction=0.0,
                )
                row["round_size"] = 1024
                row["generated_tokens"] = max(
                    int(output.completion_tokens) for output in refill
                )
                row["passed"] = bool(
                    row["passed"] and row["generated_tokens"] >= 1100
                )
                criteria.append(row)
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return criteria


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registered refill-vs-padded evaluation-engine gate."
    )
    parser.add_argument("--size", default=None, choices=("e2b", "e4b"))
    parser.add_argument("--quant", default=None, choices=("bf16", "nf4"))
    parser.add_argument(
        "--prereg",
        default="prereg/prereg.yaml",
        help="registration that defines the covered combinations",
    )
    parser.add_argument(
        "--refill-smoke",
        action="store_true",
        help="run only the registered re-prefill smoke step (never counted)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "narrow the run for a quick check; the report is marked smoke and "
            "can never be the registered pass"
        ),
    )
    parser.add_argument(
        "--bridge",
        default=None,
        choices=("scalar", "none", "damped", "gch"),
        help="smoke only: bridge to gate",
    )
    parser.add_argument(
        "--adapter-policy",
        default=None,
        choices=("always", "loop_only"),
        help="smoke only: adapter policy to gate",
    )
    parser.add_argument(
        "--blocks",
        default=None,
        help="smoke only: comma-separated inclusive loop blocks",
    )
    parser.add_argument(
        "--n-loops", default=None, help="smoke only: N values to gate"
    )
    parser.add_argument(
        "--sampling-n",
        default=None,
        help="smoke only: N values for the sampled-token criterion",
    )
    parser.add_argument(
        "--throughput-n",
        default=None,
        help="smoke only: N values for the throughput measurement",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="smoke only"
    )
    parser.add_argument(
        "--round-size", type=int, default=None, help="smoke only"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="smoke only: slot/chunk cap (registered value comes from prereg)",
    )
    parser.add_argument("--seed", type=int, default=None, help="smoke only")
    parser.add_argument("--vram-cap", type=float, default=0.92)
    parser.add_argument(
        "--prompts", default=None, help="smoke only: override the fixture"
    )
    parser.add_argument("--out", default="")
    return parser


def _ints(value: str, name: str) -> list[int]:
    try:
        values = [int(part) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name} must be comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise ValueError(f"--{name} must contain positive integers")
    return values


def _group_combinations(
    combinations: Sequence[dict],
) -> dict[tuple[str, str | None, str | None, str | None], set[int]]:
    grouped: dict[
        tuple[str, str | None, str | None, str | None], set[int]
    ] = {}
    for combo in combinations:
        key = (
            str(combo["arm"]),
            combo["block"],
            combo["bridge"],
            combo["adapter_policy"],
        )
        grouped.setdefault(key, set()).update(
            int(value) for value in combo["n_loops"]
        )
    return grouped


def _expected_coverage(combinations: Sequence[dict]) -> set[tuple]:
    return {
        _combo_key(
            combo["arm"],
            combo["block"],
            combo["bridge"],
            combo["adapter_policy"],
            int(n_loops),
        )
        for combo in combinations
        for n_loops in combo["n_loops"]
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate_reg = load_engine_gate(args.prereg)
    settings, combinations, registered = resolve_run(args, gate_reg)
    if not torch.cuda.is_available():
        raise SystemExit(
            "engine_gate: a CUDA GPU is required; run it on the controller, "
            "not in the CPU test suite"
        )
    if settings["round_size"] >= settings["max_new_tokens"]:
        raise SystemExit(
            "--round-size must be smaller than --max-new-tokens so the gate "
            "exercises multi-round re-prefill"
        )
    if settings["max_new_tokens"] < 1 or settings["round_size"] < 1:
        raise SystemExit("--max-new-tokens and --round-size must be positive")
    if settings["max_batch"] < 0:
        raise SystemExit("--max-batch must be >= 0 (0 = adaptive)")

    prompts = load_prompts(settings["prompts"])
    if args.refill_smoke:
        if args.vram_cap > 0:
            cap_cuda_allocator(args.vram_cap)
        smoke_output = (
            Path(args.out).expanduser()
            if args.out
            else paths.artifacts_dir()
            / "engine_gate"
            / "re_prefill_smoke.json"
        )
        criteria = run_re_prefill_smoke(settings, prompts)
        report = build_report(
            criteria=criteria,
            throughput=[],
            config={
                "smoke_step": "re_prefill",
                "counted": False,
                "amendment_rule": AMENDMENT_RULE,
                "prereg": str(Path(args.prereg).resolve()),
            },
            smoke=True,
        )
        report["counted"] = False
        report["amendment_rule"] = AMENDMENT_RULE
        write_report(report, smoke_output)
        print(f"re-prefill smoke (never counted): {smoke_output}")
        return 1
    if args.vram_cap > 0:
        cap_cuda_allocator(args.vram_cap)

    default_name = "engine_gate_smoke.json" if not registered else "engine_gate.json"
    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir() / "engine_gate" / default_name
    )
    expected = _expected_coverage(combinations)
    provenance = source_provenance(args.prereg)
    enforce_registered_provenance(provenance, registered=registered)
    prompt_source = prompt_provenance(settings["prompts"])
    config = {
        "registered": bool(registered),
        "smoke": bool(not registered),
        "prereg": str(Path(args.prereg).resolve()),
        "size": settings["size"],
        "quant": settings["quant"],
        "max_new_tokens": settings["max_new_tokens"],
        "round_size": settings["round_size"],
        "max_batch": settings["max_batch"],
        "token_limit": settings["token_limit"],
        "near_tie_logit_gap": settings["near_tie_logit_gap"],
        "min_exact_fraction": settings["min_exact_fraction"],
        "sampling_n": settings["sampling_n"],
        "throughput_n": settings["throughput_n"],
        "seed": settings["seed"],
        "prompts_file": str(Path(settings["prompts"]).resolve()),
        "prompts_sha256": prompt_source["sha256"],
        "prompt_provenance": prompt_source,
        "provenance": provenance,
        "prompt_count": len(prompts["prompts"]),
        "sampling_prompt_count": len(prompts["sampling_ids"]),
        "registered_combination_count": len(expected),
        "seeding": {
            "lora_B_std": LORA_B_STD,
            "bridge_g_uniform": [BRIDGE_G_LOW, BRIDGE_G_HIGH],
        },
    }

    criteria: list[dict] = []
    throughput: list[dict] = []
    observed: set[tuple] = set()
    groups = _group_combinations(combinations)

    def write_current() -> dict:
        coverage = coverage_criterion(expected, observed)
        pooled = pooled_exactness_criterion(criteria)
        report = build_report(
            criteria=[coverage, pooled, *criteria],
            throughput=throughput,
            config=config,
            smoke=not registered,
        )
        write_report(report, output)
        return report

    for group_key, n_values in groups.items():
        group_criteria, group_throughput, group_observed = run_group(
            group_key, sorted(n_values), prompts, settings
        )
        criteria.extend(group_criteria)
        throughput.extend(group_throughput)
        observed.update(group_observed)
        # Write after every group so a crash keeps the completed evidence.
        write_current()

    extra_criteria = run_coverage_extras(settings, prompts, combinations)
    criteria.extend(extra_criteria)
    report = write_current()
    print(
        f"engine gate: {report['overall']} -> {output} "
        f"(markdown: {output.with_suffix('.md')})"
    )
    if registered:
        return 0 if report["registered_pass"] else 1
    # A smoke run is evidence only; never return success for it.
    return 1


AMENDMENT_RULE = (
    "Registered re-prefill smoke: logged, never counted.  After the smoke only "
    "the exactness margins may be amended, and only if every refill divergence "
    "is a near-tie and logprob drift is within the registered bounds; the "
    "near-tie threshold, drift bounds, coverage and prompts stay fixed.  Any "
    "amendment is committed before the registered run."
)


def run_re_prefill_smoke(
    settings: dict[str, Any], prompts: dict[str, Any]
) -> list[dict]:
    """Registered smoke step: O N=1, L 15-29 gch N=2, one round-1024 row."""

    criteria: list[dict] = []
    smoke_settings = {
        **settings,
        "throughput_n": [],
        "sampling_n": [2],
        "comparison_prompts": min(8, int(settings["comparison_prompts"])),
    }
    for group_key, n_values, sampling_n in (
        (("O", None, None, None), [1], []),
        (("L", "15-29", "gch", "loop_only"), [2], [2]),
    ):
        group_settings = {**smoke_settings, "sampling_n": sampling_n}
        group_criteria, _, _ = run_group(
            group_key, n_values, prompts, group_settings
        )
        criteria.extend(group_criteria)

    # One round-1024 row with the registered near-tie treatment.
    long_conversations = [
        prompt["messages"] for prompt in prompts["long_prompts"][:2]
    ]
    long_ids = [str(prompt["id"]) for prompt in prompts["long_prompts"][:2]]
    loop = LoopConfig(
        n_loops=2,
        bridge="gch",
        block=parse_block("15-29"),
        adapter_policy="loop_only",
    )
    model = processor = None
    try:
        model, processor, plan = load_model(
            ModelSpec(size=settings["size"], quant=settings["quant"], loop=loop)
        )
        model = seed_gate_parameters(
            model,
            arm="L",
            bridge="gch",
            plan=plan,
            label="smoke_L_15-29_gch_loop_only",
            base_seed=int(settings["seed"]),
        )
        set_n_loops(_text_model(model), 2)
        heavy = Sampling(
            max_new_tokens=int(settings["long_round_1024_tokens"]),
            temperature=0.0,
        )
        padded, _ = run_engine(
            "padded",
            model,
            processor,
            long_conversations,
            long_ids,
            heavy,
            seed=int(settings["seed"]),
            max_batch=1,
            round_size=int(settings["long_round_size"]),
            record_top20=False,
        )
        refill, _ = run_engine(
            "refill",
            model,
            processor,
            long_conversations,
            long_ids,
            heavy,
            seed=int(settings["seed"]),
            max_batch=1,
            round_size=1024,
            record_top20=False,
        )
        row = equality_criterion(
            "round_1024_smoke",
            padded,
            refill,
            long_ids,
            conversations=long_conversations,
            sampling=heavy,
            logits_fn=lambda engine, conversation, prefix: (
                recompute_engine_logits(
                    model, processor, conversation, prefix, heavy, engine
                )
            ),
            token_limit=int(settings["long_round_1024_tokens"]),
            near_tie_gap=float(settings["near_tie_logit_gap"]),
            min_exact_fraction=0.0,
        )
        row["smoke"] = True
        criteria.append(row)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for row in criteria:
        row["smoke"] = True
    return criteria


if __name__ == "__main__":
    sys.exit(main())
