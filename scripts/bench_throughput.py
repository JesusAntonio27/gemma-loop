"""Measure aggregate generation throughput for the Gemma-Loop serve engines.

Task 5b compares, on the local RTX 5070 (12 GiB, Windows, bf16, sdpa):

* the current left-padded ``generate_batch`` at batch sizes 8..64;
* the transformers 5.22 continuous batching engine (paged sdpa), when it is
  compatible with Gemma 4 KV sharing, PLE and the loop patch;
* the slot-refill wrapper, which keeps every slot busy by replacing finished
  sequences with queued ones (left-padded rounds).

The measurement is deliberately end-to-end: prompts are rendered with the real
chat template (about ``--prompt-tokens`` tokens), ``min_new_tokens`` forces
every sequence to produce exactly ``--gen-new-tokens`` tokens so the reported
rate is a pure decode rate, and aggregate throughput counts every generated
token divided by the wall-clock time of one engine call.

Peak VRAM is ``torch.cuda.max_memory_reserved()``; a configuration *spills*
when it exceeds 95 % of the dedicated VRAM (the Task 6 rule).  The heavy
results go to ``<artifacts>/calibration/throughput.json`` plus a markdown
table in the same directory.

Usage::

    uv run python scripts/bench_throughput.py --engine padded \
        --loops 1,2,4 --batches 8,16,32,48,64 --out <...>/throughput.json

    uv run python scripts/bench_throughput.py --engine cb --loops 1,2 \
        --batches 16,32 --append <...>/throughput.json

``--append`` merges the new measurements into an existing file so the baseline
and each candidate engine can be measured in separate processes (one model
loaded at a time, VRAM freed between runs).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from gemmaloop import paths
from gemmaloop.model.load import ModelSpec, load_model
from gemmaloop.model.looped import LoopConfig, set_n_loops
from gemmaloop.serve.generate import Sampling, generate_batch

SPILL_RESERVED_FRACTION = 0.95

PROMPT_TEXT = (
    "You are a meticulous computer-science tutor. Read the following notes "
    "and then write a single, very long, self-contained explanation.\n\n"
    "Notes: A modern computer is organised as a hierarchy of memories. The "
    "registers inside the CPU are the fastest but smallest level; below them "
    "come the L1, L2 and L3 caches, then the main DRAM, then the SSD or hard "
    "disk. The operating system virtualises memory, schedules threads, "
    "services interrupts and enforces isolation between processes. The CPU "
    "pipeline fetches, decodes and executes instructions, predicts branches, "
    "reorders memory accesses and keeps caches coherent across cores. Device "
    "drivers convert generic read and write requests into commands for the "
    "storage controller, the network card and the display. Filesystems map "
    "names to inodes or extents, journal metadata to survive crashes, and "
    "flush dirty pages lazily. Virtual memory is backed by page tables and a "
    "translation lookaside buffer; a page fault suspends the instruction and "
    "lets the kernel bring the page in from disk or from a swap file. "
) * 6


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Resource tracking
# ---------------------------------------------------------------------------


def _total_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_properties(0).total_memory)


def _peak_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 2**30


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _process_ram_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _conversation(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _render_prompt_tokens(processor: Any, prompt: str) -> int:
    rendered = processor.apply_chat_template(
        _conversation(prompt),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return int(rendered["attention_mask"].sum().item())


def _prompt_for_tokens(processor: Any, target: int) -> tuple[str, int]:
    """Crop or repeat ``PROMPT_TEXT`` until the rendered prompt is ~``target``.

    The chat template wraps the text, so the number of user tokens cannot be set
    exactly from a character count.  The loop converges within a few percent.
    """
    prompt = PROMPT_TEXT
    count = _render_prompt_tokens(processor, prompt)
    if count < target:
        repetitions = max(1, target // max(1, count))
        prompt = PROMPT_TEXT * repetitions
        count = _render_prompt_tokens(processor, prompt)
    # Crop by characters (roughly proportional to tokens) until close enough.
    for _ in range(8):
        if abs(count - target) <= max(8, target // 20):
            break
        scaled = max(1, int(len(prompt) * target / max(1, count)))
        prompt = prompt[:scaled]
        count = _render_prompt_tokens(processor, prompt)
    return prompt, count


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


def _run_padded(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
) -> list[Any]:
    return generate_batch(model, processor, conversations, sampling, seed=seed)


def _run_refill(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
) -> list[Any]:
    from gemmaloop.serve.refill import generate_batch_refill

    return generate_batch_refill(
        model,
        processor,
        conversations,
        sampling,
        seed=seed,
        max_batch=len(conversations),
        stop_on_eos=False,
    )


def _run_cb(
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
) -> list[Any]:
    raise RuntimeError(
        "transformers 5.22 continuous batching is incompatible with Gemma 4: "
        "PagedAttentionCache rejects the heterogeneous per-layer head_dim "
        "(continuous_batching/utils.py:223 via cache_allocators/full_attention.py:52), "
        "and its KV-sharing path desynchronises the paged cache from the SDPA "
        "mask (sdpa_attention.py:174). See the Task 5b report section 3a."
    )


ENGINES: dict[str, Callable[..., list[Any]]] = {
    "padded": _run_padded,
    "refill": _run_refill,
    "cb": _run_cb,
}


def _measure(
    engine: Callable[..., list[Any]],
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    sampling: Sampling,
    seed: int,
) -> dict[str, Any]:
    """Run one engine call and return tokens/s plus memory statistics."""
    _reset_peak()
    started = time.perf_counter()
    try:
        outputs = engine(model, processor, conversations, sampling, seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"tok_s": None, "spill": True, "oom": True, "peak_vram_gib": None, "tokens": 0}
    elapsed = time.perf_counter() - started
    peak_gib = _peak_vram_gib()
    total_gib = _total_vram_bytes() / 2**30
    tokens = sum(int(output.completion_tokens) for output in outputs)
    tok_s = tokens / elapsed if elapsed > 0 and tokens > 0 else None
    return {
        "tok_s": round(tok_s, 3) if tok_s is not None else None,
        "elapsed_s": round(elapsed, 3),
        "tokens": tokens,
        "peak_vram_gib": round(peak_gib, 3),
        "total_vram_gib": round(total_gib, 3),
        "spill": bool(peak_gib > SPILL_RESERVED_FRACTION * total_gib),
        "oom": False,
    }


def _warmup(
    engine: Callable[..., list[Any]],
    model: Any,
    processor: Any,
    conversations: list[list[dict]],
    seed: int,
) -> None:
    previous = getattr(model.generation_config, "min_new_tokens", 0)
    try:
        model.generation_config.min_new_tokens = 1
        engine(model, processor, conversations[:1], Sampling(max_new_tokens=4), seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        model.generation_config.min_new_tokens = previous


# ---------------------------------------------------------------------------
# Measurement driver
# ---------------------------------------------------------------------------


def _set_loops(model: Any, n_loops: int) -> None:
    text_model = model.model.language_model
    if getattr(text_model, "_gemmaloop_loop_config", None) is None:
        from gemmaloop.model.layer_plan import make_layer_plan
        from gemmaloop.model.looped import apply_loops

        plan = make_layer_plan(model.config.text_config.to_dict())
        apply_loops(text_model, plan, LoopConfig(n_loops=n_loops))
    else:
        set_n_loops(text_model, n_loops)


def _measure_matrix(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    prompt: str,
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure every (N, batch) pair, saving after each record.

    ``on_record(key, record)`` is called *before* moving on, so a process that
    is killed only loses the configuration that was in flight (these sweeps can
    run for hours).  Completed non-OOM records are skipped on resume unless
    ``--force`` is set.
    """
    sampling = Sampling(max_new_tokens=args.gen_new_tokens)
    previous_min = getattr(model.generation_config, "min_new_tokens", 0)
    engine_name = getattr(args, "label", "") or args.engine
    engine = ENGINES[args.engine]
    existing = (args._existing or {}).get(engine_name, {})
    results: dict[str, Any] = {}
    try:
        for n_loops in args.loops:
            _set_loops(model, n_loops)
            for batch in args.batches:
                key = f"n{n_loops}_b{batch}"
                previous = existing.get(key)
                if (
                    not args.force
                    and previous is not None
                    and not previous.get("oom")
                    and previous.get("tok_s") is not None
                ):
                    _log(f"{engine_name}: {key} already measured, skipping")
                    results[key] = previous
                    continue
                _log(f"{engine_name}: measuring {key} ...")
                conversations = [_conversation(prompt) for _ in range(batch)]
                try:
                    _warmup(engine, model, processor, conversations, args.seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                model.generation_config.min_new_tokens = args.gen_new_tokens
                record = _measure(
                    engine, model, processor, conversations, sampling, args.seed
                )
                record["engine"] = engine_name
                record["n_loops"] = n_loops
                record["batch"] = batch
                record["prompt_tokens"] = int(getattr(args, "_prompt_tokens", 0) or 0)
                record["gen_new_tokens"] = int(args.gen_new_tokens)
                record["cache"] = (
                    "dynamic"
                    if args.gen_new_tokens <= 4096
                    else "static"
                )
                results[key] = record
                if on_record is not None:
                    on_record(key, record)
                _log(
                    f"{engine_name}: {key} tok/s={record['tok_s']} "
                    f"peak={record['peak_vram_gib']} GiB spill={record['spill']} "
                    f"oom={record['oom']}"
                )
                if record.get("oom"):
                    # Larger batches will not fit either; stop this N.
                    break
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        model.generation_config.min_new_tokens = previous_min
    return results


def _measure_long_context(
    args: argparse.Namespace, model: Any, processor: Any, prompt: str
) -> dict[str, Any]:
    if args.long_context_tokens <= 0:
        return {}
    _set_loops(model, args.long_loops)
    _log(
        f"{args.engine}: long-context n{args.long_loops} b{args.long_batch} "
        f"gen={args.long_context_tokens} ..."
    )
    conversations = [_conversation(prompt) for _ in range(args.long_batch)]
    sampling = Sampling(max_new_tokens=args.long_context_tokens)
    previous_min = getattr(model.generation_config, "min_new_tokens", 0)
    model.generation_config.min_new_tokens = args.long_context_tokens
    try:
        record = _measure(
            ENGINES[args.engine], model, processor, conversations, sampling, args.seed
        )
    finally:
        model.generation_config.min_new_tokens = previous_min
    record.update(
        {
            "engine": args.engine,
            "n_loops": args.long_loops,
            "batch": args.long_batch,
            "gen_new_tokens": args.long_context_tokens,
        }
    )
    _log(
        f"{args.engine}: long-context tok/s={record['tok_s']} "
        f"peak={record['peak_vram_gib']} GiB spill={record['spill']} "
        f"oom={record['oom']}"
    )
    return record


# ---------------------------------------------------------------------------
# Persistence / reporting
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Generation throughput (E2B bf16, sdpa)",
        "",
        f"- GPU: {data.get('meta', {}).get('gpu')}",
        f"- prompt tokens: {data.get('meta', {}).get('prompt_tokens')}",
        "- generated tokens: per row (the sweep mixes 2k/4k/8k/16k/32k buckets)",
        f"- seed: {data.get('meta', {}).get('seed')}",
        "",
        "| engine | N | batch | gen tokens | cache | tok/s | peak VRAM (GiB) | spill | OOM |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = data.get("results", {})
    for engine in sorted(rows):
        for key in sorted(rows[engine], key=_row_order):
            record = rows[engine][key]
            lines.append(
                f"| {engine} | {record['n_loops']} | {record['batch']} | "
                f"{record.get('gen_new_tokens', '-')} | "
                f"{record.get('cache', '-')} | {_fmt(record['tok_s'])} | "
                f"{_fmt(record['peak_vram_gib'], 2)} | "
                f"{record['spill']} | {record['oom']} |"
            )
    long_context = data.get("long_context", {})
    if long_context:
        lines += [
            "",
            "## Long context",
            "",
            "| engine | N | batch | generated | cache | tok/s | peak VRAM (GiB) | spill | OOM |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for engine in sorted(long_context):
            record = long_context[engine]
            lines.append(
                f"| {engine} | {record['n_loops']} | {record['batch']} | "
                f"{record.get('gen_new_tokens')} | {record.get('cache', '-')} | "
                f"{_fmt(record['tok_s'])} | "
                f"{_fmt(record['peak_vram_gib'], 2)} | {record['spill']} | "
                f"{record['oom']} |"
            )
    chosen = data.get("chosen")
    if chosen:
        lines += ["", f"Chosen default: {chosen}"]
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _row_order(key: str) -> tuple[int, int]:
    try:
        n_part, b_part = key.split("_")
        return int(n_part[1:]), int(b_part[1:])
    except (ValueError, IndexError):
        return (99, 99)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_csv(value: str) -> list[int]:
    return [int(part) for part in _csv(value)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation throughput benchmark.")
    parser.add_argument("--engine", default="padded", choices=sorted(ENGINES))
    parser.add_argument(
        "--label",
        default="",
        help="result key (defaults to --engine); lets one engine be measured twice, e.g. with/without the static cache",
    )
    parser.add_argument("--size", default="e2b", choices=("e2b", "e4b"))
    parser.add_argument("--quant", default="", help="bf16 (default for e2b) or nf4")
    parser.add_argument("--loops", default="1,2,4")
    parser.add_argument("--batches", default="8,16,32,48,64")
    parser.add_argument("--prompt-tokens", type=int, default=500)
    parser.add_argument("--gen-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--long-context-tokens", type=int, default=16384)
    parser.add_argument("--long-loops", type=int, default=2)
    parser.add_argument("--long-batch", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--append",
        action="store_true",
        help="merge into an existing throughput.json instead of replacing it",
    )
    parser.add_argument(
        "--no-long-context",
        action="store_true",
        help="skip the long-context KV memory probe",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-measure configurations already present in the output file",
    )
    args = parser.parse_args(argv)

    args.loops = _int_csv(args.loops)
    args.batches = _int_csv(args.batches)
    if not args.loops or not args.batches:
        parser.error("--loops and --batches must not be empty")
    if min(args.loops) < 1 or min(args.batches) < 1:
        parser.error("--loops and --batches must be positive")
    if args.prompt_tokens < 1 or args.gen_new_tokens < 1:
        parser.error("--prompt-tokens and --gen-new-tokens must be positive")
    if args.size == "e4b" and not args.quant:
        args.quant = "nf4"
    if not args.quant:
        args.quant = "bf16"
    if args.no_long_context:
        args.long_context_tokens = 0
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_label = args.label or args.engine
    output = (
        Path(args.out).expanduser()
        if args.out
        else paths.artifacts_dir() / "calibration" / "throughput.json"
    )
    data = _load(output) if args.append else {}
    data.setdefault("results", {})
    meta = data.setdefault("meta", {})
    meta.update(
        {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gib": round(_total_vram_bytes() / 2**30, 3),
            "prompt_tokens": None,
            "gen_new_tokens": args.gen_new_tokens,
            "seed": args.seed,
            "torch": str(torch.__version__),
        }
    )
    import transformers

    meta["transformers"] = str(transformers.__version__)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = processor = None
    try:
        _log(
            f"loading {args.size} {args.quant} "
            f"(free vram={torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB)"
        )
        model, processor, _ = load_model(
            ModelSpec(
                size=args.size,
                quant=args.quant,
                loop=LoopConfig(n_loops=args.loops[0]),
            ),
            min_free_gib=args.min_free_gib,
        )
        prompt, prompt_tokens = _prompt_for_tokens(processor, args.prompt_tokens)
        meta["prompt_tokens"] = prompt_tokens
        _log(f"prompt tokens: {prompt_tokens}")

        args._existing = data.get("results", {})
        args._prompt_tokens = prompt_tokens
        data["results"].setdefault(model_label, {})
        data["meta"]["prompt_tokens"] = prompt_tokens
        markdown_path = output.with_suffix(".md")

        def save_record(key: str, record: dict[str, Any]) -> None:
            data["results"].setdefault(model_label, {})[key] = record
            data["meta"]["peak_process_ram_gib"] = round(_process_ram_gib(), 2)
            _save(output, data)
            markdown_path.write_text(_markdown(data), encoding="utf-8")

        # Write the metadata before the first (long) configuration so a cut
        # process still leaves a readable file behind.
        data["meta"]["peak_process_ram_gib"] = round(_process_ram_gib(), 2)
        _save(output, data)
        results = _measure_matrix(
            args, model, processor, prompt, on_record=save_record
        )
        data["results"].setdefault(model_label, {}).update(results)
        if args.long_context_tokens:
            previous_long = data.get("long_context", {}).get(model_label)
            if (
                previous_long is not None
                and not args.force
                and not previous_long.get("oom")
                and previous_long.get("tok_s") is not None
            ):
                _log(f"{args.engine}: long-context already measured, skipping")
            else:
                long_record = _measure_long_context(args, model, processor, prompt)
                data.setdefault("long_context", {})[model_label] = long_record
                data["meta"]["peak_process_ram_gib"] = round(_process_ram_gib(), 2)
                _save(output, data)
                markdown_path.write_text(_markdown(data), encoding="utf-8")
        data["meta"]["peak_process_ram_gib"] = round(_process_ram_gib(), 2)
        _save(output, data)
        markdown_path.write_text(_markdown(data), encoding="utf-8")
        _log(f"wrote {output} and {markdown_path}")
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
