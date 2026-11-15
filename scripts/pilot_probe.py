"""Pilot probe: residual-stream geometry and untrained-loop endpoints.

P1 of the methodology review ("Pilot (a)"): a forward-only sweep over candidate
loop blocks and bridges on held-out rows.  For every candidate it reports

* per-layer ``cos(out, in)`` and ``RMS(out) / RMS(in)`` (residual-stream
  geometry, including the effect of Gemma 4's non-unit ``layer_scalar``);
* untrained ``KL(N=1 || N=2)`` on the labelled/assistant tokens;
* ``Delta NLL`` after fitting one scalar temperature per arm on a separate
  calibration split (raw NLL is gamed by flattening);
* the block's share of per-pass FLOPs.

It also dumps every layer's ``layer_scalar``.  Results go to a JSON file and a
markdown table.

Two modes:

* ``--tiny``: a small fp32 ``Gemma4TextModel`` on CPU (no downloads, no GPU).
  Rows must carry ``input_ids`` (and optionally ``labels_start``).  This is the
  smoke test and CI path.
* default: the production loader (``load_model``) on the real E2B/E4B
  checkpoint.  Run this on the GPU when provisioning allows it; it is never
  invoked by the test suite.

The script is read-only with respect to training artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from gemmaloop.eval.calibrated_nll import _input_device, calibrated_nll_report
from gemmaloop.gpulock import require_gpu_lock
from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path
from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.layer_plan import make_layer_plan, parse_block
from gemmaloop.model.looped import LoopConfig, apply_loops, set_n_loops
from gemmaloop.train.perf import cap_cuda_allocator

DEFAULT_BLOCKS = "15-29,15-19,16-18,25-29"
DEFAULT_TINY_BLOCKS = "5-7,6-10"
DEFAULT_BRIDGES = "none,scalar,gch,damped"


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------
@dataclass
class Row:
    input_ids: torch.Tensor
    labels_start: int = 0


def _row_from_json(payload: dict, tokenizer) -> Row | None:
    if "input_ids" in payload:
        ids = torch.tensor(payload["input_ids"], dtype=torch.long)
        start = int(payload.get("labels_start", 0))
        return Row(ids, max(0, min(start, ids.numel())))

    if "prompt" in payload and "completion" in payload:
        if tokenizer is None:
            raise ValueError("prompt/completion rows need a tokenizer (drop --tiny)")
        prompt_ids = tokenizer(
            str(payload["prompt"]), add_special_tokens=False
        ).input_ids
        full_ids = tokenizer(
            str(payload["prompt"]) + str(payload["completion"]),
            add_special_tokens=False,
        ).input_ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("prompt is not a prefix of prompt+completion")
        return Row(torch.tensor(full_ids, dtype=torch.long), len(prompt_ids))

    if "messages" in payload:
        if tokenizer is None:
            raise ValueError("messages rows need a tokenizer (drop --tiny)")
        text = tokenizer.apply_chat_template(
            payload["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return Row(torch.tensor(ids, dtype=torch.long), 0)

    if "text" in payload:
        if tokenizer is None:
            raise ValueError("text rows need a tokenizer (drop --tiny)")
        ids = tokenizer(str(payload["text"]), add_special_tokens=False).input_ids
        return Row(torch.tensor(ids, dtype=torch.long), 0)

    return None


def load_rows(path: str | Path, tokenizer=None, max_rows: int | None = None) -> list[Row]:
    rows: list[Row] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = _row_from_json(json.loads(line), tokenizer)
            if row is None:
                raise ValueError(
                    "each JSONL row needs input_ids, prompt+completion, "
                    "messages or text"
                )
            if row.input_ids.numel() >= 2:
                rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    return rows


def split_rows(rows: Sequence[Row]) -> tuple[list[Row], list[Row]]:
    """Alternate rows into calibration/test halves (test is never empty)."""
    if len(rows) < 2:
        return list(rows), list(rows)
    calibration = list(rows[::2])
    test = list(rows[1::2])
    return calibration, test


def _batch(row: Row) -> dict[str, Any]:
    labels = row.input_ids.clone()
    labels[: row.labels_start] = -100
    return {
        "input_ids": row.input_ids.unsqueeze(0),
        "attention_mask": torch.ones(1, row.input_ids.numel(), dtype=torch.long),
        "labels": labels.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------
def _text_model(model) -> torch.nn.Module:
    base = model
    get_base = getattr(base, "get_base_model", None)
    if callable(get_base):
        base = get_base()
    inner = getattr(base, "model", None)
    text = base if inner is None else getattr(inner, "language_model", inner)
    if text is None or not hasattr(text, "layers"):
        raise ValueError("cannot locate the Gemma 4 text model")
    return text


def _lm_weight(model, text_model) -> torch.Tensor:
    base = model
    get_base = getattr(base, "get_base_model", None)
    if callable(get_base):
        base = get_base()
    get_head = getattr(base, "get_output_embeddings", None)
    if callable(get_head):
        head = get_head()
        if head is not None and getattr(head, "weight", None) is not None:
            return head.weight
    return text_model.embed_tokens.weight


def _tiny_text_model() -> torch.nn.Module:
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    config = Gemma4TextConfig(
        num_hidden_layers=15,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 3,
        num_kv_shared_layers=10,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        vocab_size=256,
        vocab_size_per_layer_input=256,
        sliding_window=16,
        use_double_wide_mlp=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Gemma4TextModel(config)
    # Deliberately non-unit layer scalars, like the real E2B checkpoint.
    generator = torch.Generator().manual_seed(0)
    for layer in model.layers:
        layer.layer_scalar.copy_(torch.rand(1, generator=generator) + 0.4)
    model.eval()
    return model


def _blocks_from_args(args) -> list[tuple[int, int]]:
    blocks_text = args.blocks or (
        DEFAULT_TINY_BLOCKS if getattr(args, "tiny", False) else DEFAULT_BLOCKS
    )
    return [parse_block(item) for item in blocks_text.split(",") if item.strip()]


def load_probe_model(args) -> tuple[torch.nn.Module, Any]:
    """Return ``(model, tokenizer_or_processor)`` for the requested mode."""
    blocks = _blocks_from_args(args)
    if args.tiny:
        model = _tiny_text_model()
        plan = make_layer_plan(model.config.to_dict(), block=blocks[0])
        apply_loops(
            model,
            plan,
            LoopConfig(
                n_loops=1, bridge="gch", block=blocks[0], adapter_policy="always"
            ),
        )
        return model, None
    from gemmaloop.model.load import ModelSpec, load_model

    spec = ModelSpec(
        size=args.size,
        quant=args.quant,
        loop=LoopConfig(
            n_loops=1, bridge="gch", block=blocks[0], adapter_policy="always"
        ),
    )
    model, processor, _ = load_model(spec)
    tokenizer = getattr(processor, "tokenizer", processor)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def layer_geometry(
    text_model: torch.nn.Module,
    rows: Sequence[Row],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, float]]:
    """Per-layer input/output cosine, RMS and RMS ratio at N=1."""
    num_layers = int(text_model.config.num_hidden_layers)
    captured: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def hook(index):
        def _capture(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captured.setdefault(index, []).append(
                (inputs[0].detach().float(), out.detach().float())
            )

        return _capture

    handles = [
        text_model.layers[i].register_forward_hook(hook(i))
        for i in range(num_layers)
    ]
    sums = [
        {"cos": 0.0, "rms_in": 0.0, "rms_out": 0.0, "count": 0}
        for _ in range(num_layers)
    ]
    was_training = bool(getattr(text_model, "training", False))
    try:
        text_model.eval()
        if getattr(text_model, "_gemmaloop_loop_config", None) is not None:
            set_n_loops(text_model, 1)
        for row_index, row in enumerate(rows):
            captured.clear()
            with torch.no_grad():
                text_model(
                    input_ids=row.input_ids.unsqueeze(0).to(_input_device(text_model)),
                    use_cache=False,
                )
            if progress is not None:
                progress(
                    {
                        "phase": "geometry",
                        "items_done": row_index + 1,
                        "items_total": len(rows),
                        "tokens": int(row.input_ids.numel()),
                    }
                )
            for index in range(num_layers):
                if index not in captured:
                    continue
                h_in, h_out = captured[index][0]
                cos = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
                rms_in = h_in.pow(2).mean(-1).sqrt().mean().item()
                rms_out = h_out.pow(2).mean(-1).sqrt().mean().item()
                entry = sums[index]
                entry["cos"] += cos
                entry["rms_in"] += rms_in
                entry["rms_out"] += rms_out
                entry["count"] += 1
    finally:
        for handle in handles:
            handle.remove()
        if getattr(text_model, "training", None) is not None:
            text_model.train(was_training)

    geometry = []
    for index, entry in enumerate(sums):
        count = max(1, entry["count"])
        rms_in = entry["rms_in"] / count
        rms_out = entry["rms_out"] / count
        geometry.append(
            {
                "layer": index,
                "cos_in_out": entry["cos"] / count,
                "rms_in": rms_in,
                "rms_out": rms_out,
                "rms_ratio": rms_out / rms_in if rms_in else 0.0,
            }
        )
    return geometry


def _hidden(
    text_model: torch.nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        return text_model(
            input_ids=input_ids.unsqueeze(0).to(_input_device(text_model)),
            use_cache=False,
        ).last_hidden_state


@torch.no_grad()
def _kl_from_hidden(
    hidden_one: torch.Tensor,
    hidden_two: torch.Tensor,
    weight: torch.Tensor,
    softcap: float | None,
    *,
    token_chunk: int = 1024,
    vocab_chunk: int = 16384,
) -> float:
    """Mean KL(p1 || p2) over positions, chunked over tokens and vocabulary."""
    h1_all = hidden_one[0].float()
    h2_all = hidden_two[0].float()
    device = weight.device
    total = 0.0
    count = 0
    for t0 in range(0, h1_all.shape[0], token_chunk):
        t1 = min(t0 + token_chunk, h1_all.shape[0])
        h1 = h1_all[t0:t1].to(device)
        h2 = h2_all[t0:t1].to(device)
        n = t1 - t0
        m1 = torch.full((n,), float("-inf"), device=device)
        m2 = torch.full((n,), float("-inf"), device=device)
        s1 = torch.zeros((n,), device=device)
        s2 = torch.zeros((n,), device=device)
        for v0 in range(0, weight.shape[0], vocab_chunk):
            # `hidden` was cast to fp32 above; cast the weight tile to the same
            # dtype explicitly (the real E2B/E4B LM head is bf16, and a mixed
            # fp32 x bf16 matmul raises on torch).
            w = weight[v0 : v0 + vocab_chunk].to(
                device=device, dtype=h1.dtype
            )
            z1 = h1 @ w.t()
            z2 = h2 @ w.t()
            if softcap is not None:
                z1 = torch.tanh(z1 / float(softcap)) * float(softcap)
                z2 = torch.tanh(z2 / float(softcap)) * float(softcap)
            nm1 = torch.maximum(m1, z1.max(dim=1).values)
            s1 = s1 * torch.exp(m1 - nm1) + torch.exp(z1 - nm1.unsqueeze(1)).sum(1)
            m1 = nm1
            nm2 = torch.maximum(m2, z2.max(dim=1).values)
            s2 = s2 * torch.exp(m2 - nm2) + torch.exp(z2 - nm2.unsqueeze(1)).sum(1)
            m2 = nm2
        logz1 = m1 + torch.log(s1)
        logz2 = m2 + torch.log(s2)
        kl = torch.zeros((n,), device=device)
        for v0 in range(0, weight.shape[0], vocab_chunk):
            # `hidden` was cast to fp32 above; cast the weight tile to the same
            # dtype explicitly (the real E2B/E4B LM head is bf16, and a mixed
            # fp32 x bf16 matmul raises on torch).
            w = weight[v0 : v0 + vocab_chunk].to(
                device=device, dtype=h1.dtype
            )
            z1 = h1 @ w.t()
            z2 = h2 @ w.t()
            if softcap is not None:
                z1 = torch.tanh(z1 / float(softcap)) * float(softcap)
                z2 = torch.tanh(z2 / float(softcap)) * float(softcap)
            logp1 = z1 - logz1.unsqueeze(1)
            logp2 = z2 - logz2.unsqueeze(1)
            kl += (logp1.exp() * (logp1 - logp2)).sum(1)
        total += float(kl.sum().item())
        count += n
    return total / max(count, 1)


def _kl_n1_n(
    text_model: torch.nn.Module,
    weight: torch.Tensor,
    rows: Sequence[Row],
    n_loops: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> float:
    """Forward-only ``KL(softmax N=1 || softmax N=n)`` on labelled tokens."""
    if n_loops <= 1:
        return 0.0
    softcap = getattr(text_model.config, "final_logit_softcapping", None)
    values = []
    for row_index, row in enumerate(rows):
        start = max(0, row.labels_start - 1)
        set_n_loops(text_model, 1)
        one = _hidden(text_model, row.input_ids)[:, start:-1]
        set_n_loops(text_model, n_loops)
        many = _hidden(text_model, row.input_ids)[:, start:-1]
        if one.shape[1] == 0:
            continue
        values.append(_kl_from_hidden(one, many, weight, softcap))
        if progress is not None:
            progress(
                {
                    "phase": f"kl N={n_loops}",
                    "items_done": row_index + 1,
                    "items_total": len(rows),
                    "tokens": 2 * int(row.input_ids.numel()),
                }
            )
    return sum(values) / len(values) if values else 0.0


def _arm_metrics(
    model: torch.nn.Module,
    calibration: Sequence[Row],
    test: Sequence[Row],
    n_loops: int,
) -> dict[str, float]:
    """Calibrated and raw NLL plus top-1, with T fitted on calibration rows."""
    report = calibrated_nll_report(
        model,
        [_batch(row) for row in calibration],
        [_batch(row) for row in test],
        n_loops,
        max_iter=30,
        tol=1e-2,
    )
    return {
        "raw_nll": float(report["raw_nll"]),
        "nll_calibrated": float(report["calibrated_nll"]),
        "top1_accuracy": float(report["top1_accuracy"]),
        "temperature": float(report["temperature"]),
    }


def probe(
    model: torch.nn.Module,
    rows: Sequence[Row],
    blocks: Sequence[tuple[int, int]],
    bridges: Sequence[str],
    n_loops: int | Sequence[int],
    g_std: float = 0.0,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    text_model = _text_model(model)
    weight = _lm_weight(model, text_model)
    calibration, test = split_rows(rows)
    config = text_model.config.to_dict()
    layer_flops = per_layer_flops(config)
    constant_flops = int(getattr(layer_flops, "constant_flops", 0))
    total_flops = constant_flops + sum(int(value) for value in layer_flops)

    if isinstance(n_loops, int):
        loop_values = [int(n_loops)]
    else:
        loop_values = [int(value) for value in n_loops]
    if not loop_values or any(value < 1 for value in loop_values):
        raise ValueError("n_loops must contain values >= 1")

    geometry = layer_geometry(text_model, rows, progress=progress)
    layer_scalars = [
        float(layer.layer_scalar.item()) for layer in text_model.layers
    ]

    # N=1 is independent of the block and bridge, so fit/report it once.
    base_tokens = sum(int(row.input_ids.numel()) for row in calibration) + sum(
        int(row.input_ids.numel()) for row in test
    )
    if progress is not None:
        progress(
            {
                "phase": "calibrated N=1",
                "items_done": 0,
                "items_total": 1,
                "tokens": base_tokens,
            }
        )
    base_metrics = _arm_metrics(model, calibration, test, 1)
    variant_total = len(blocks) * len(bridges) * len(loop_values)
    variant_index = 0

    variants = []
    for block in blocks:
        plan = make_layer_plan(config, block=block)
        text_model._gemmaloop_plan = plan
        loop_flops = sum(int(layer_flops[i]) for i in plan.loop)
        for bridge in bridges:
            cfg = text_model._gemmaloop_loop_config
            cfg.bridge = bridge
            cfg.block = block
            if bridge == "gch" and hasattr(text_model, "loop_bridge_g"):
                if g_std:
                    with torch.no_grad():
                        text_model.loop_bridge_g.normal_(0.0, g_std)
                else:
                    with torch.no_grad():
                        text_model.loop_bridge_g.zero_()

            for n in loop_values:
                if n == 1:
                    metrics = dict(base_metrics)
                    kl = 0.0
                else:
                    kl = _kl_n1_n(
                        text_model, weight, test, n, progress=progress
                    )
                    metrics = _arm_metrics(model, calibration, test, n)
                variant_index += 1
                if progress is not None:
                    progress(
                        {
                            "phase": "variant",
                            "items_done": variant_index,
                            "items_total": variant_total,
                            "tokens": 0 if n == 1 else base_tokens,
                        }
                    )
                variants.append(
                    {
                        "block": [int(block[0]), int(block[1])],
                        "bridge": bridge,
                        "n_loops": int(n),
                        "kl_n1_n2": kl,
                        "kl_n1_n": kl,
                        "nll_n1_calibrated": base_metrics["nll_calibrated"],
                        "nll_n_calibrated": metrics["nll_calibrated"],
                        "delta_nll_calibrated": (
                            metrics["nll_calibrated"]
                            - base_metrics["nll_calibrated"]
                        ),
                        "raw_nll_n1": base_metrics["raw_nll"],
                        "raw_nll_n": metrics["raw_nll"],
                        "delta_raw_nll": metrics["raw_nll"] - base_metrics["raw_nll"],
                        "top1_n1": base_metrics["top1_accuracy"],
                        "top1_n": metrics["top1_accuracy"],
                        "temperature_n1": base_metrics["temperature"],
                        "temperature_n": metrics["temperature"],
                        "flops_share_per_pass": loop_flops / total_flops,
                    }
                )
    set_n_loops(text_model, 1)
    return {
        "rows": len(rows),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "n_loops": loop_values[0] if len(loop_values) == 1 else loop_values,
        "n_loop_list": loop_values,
        "layer_scalar": layer_scalars,
        "layers": geometry,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pilot probe (P1 forward sweep)",
        "",
        f"Rows: {payload['rows']} ({payload['calibration_rows']} calibration, "
        f"{payload['test_rows']} test). N = {payload['n_loops']}.",
        "",
        "| block | bridge | N | KL(N=1||N) | calibrated NLL N=1 | "
        "calibrated NLL N | raw NLL N | top-1 N | Delta NLL | T(N=1) | T(N) | "
        "FLOPs/pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for item in payload["variants"]:
        lines.append(
            "| {block[0]}-{block[1]} | {bridge} | {n_loops} | {kl_n1_n:.4f} | "
            "{nll_n1_calibrated:.4f} | {nll_n_calibrated:.4f} | "
            "{raw_nll_n:.4f} | {top1_n:.3f} | "
            "{delta_nll_calibrated:+.4f} | {temperature_n1:.3f} | "
            "{temperature_n:.3f} | {flops_share_per_pass:.1%} |".format(**item)
        )
    lines += ["", "## layer_scalar", ""]
    lines.append(
        ", ".join(
            f"L{i}={value:.3f}"
            for i, value in enumerate(payload["layer_scalar"])
        )
    )
    lines += ["", "## Per-layer residual geometry (N=1)", ""]
    lines.append("| layer | cos(out,in) | RMS in | RMS out | RMS ratio |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for layer in payload["layers"]:
        lines.append(
            "| {layer} | {cos_in_out:+.3f} | {rms_in:.3f} | {rms_out:.3f} | "
            "{rms_ratio:.3f} |".format(**layer)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path, help="held-out JSONL")
    parser.add_argument("--out-json", type=Path, default=Path("pilot_probe.json"))
    parser.add_argument("--out-md", type=Path, default=Path("pilot_probe.md"))
    parser.add_argument("--blocks", default=None)
    parser.add_argument("--bridges", default=DEFAULT_BRIDGES)
    parser.add_argument(
        "--n-loops",
        default="2",
        help="N, or a comma-separated list (e.g. 1,2,3,4)",
    )
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--g-std", type=float, default=0.0)
    parser.add_argument("--tiny", action="store_true", help="CPU fp32 tiny model")
    parser.add_argument("--size", choices=("e2b", "e4b"), default="e2b")
    parser.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument(
        "--heartbeat",
        default=None,
        help="heartbeat JSON path (default: runs/detached/pilot.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_gpu_lock()
    vram_cap = cap_cuda_allocator(0.90, 0.5)
    heartbeat = Heartbeat(
        args.heartbeat or default_heartbeat_path("pilot"),
        run="pilot-probe",
    )
    heartbeat.start(30.0)
    heartbeat.update(extra={"vram_cap": vram_cap})
    blocks = _blocks_from_args(args)
    bridges = [item.strip() for item in args.bridges.split(",") if item.strip()]
    if not blocks or not bridges:
        raise ValueError("--blocks and --bridges need at least one entry")
    loop_values = [
        int(item.strip()) for item in str(args.n_loops).split(",") if item.strip()
    ]
    if not loop_values or any(value < 1 for value in loop_values):
        raise ValueError("--n-loops must contain values >= 1 (N=1 is the reference)")

    model, tokenizer = load_probe_model(args)
    try:
        rows = load_rows(args.rows, tokenizer, max_rows=args.max_rows)
        payload = probe(
            model,
            rows,
            blocks,
            bridges,
            loop_values,
            g_std=args.g_std,
            progress=lambda update: heartbeat.update(**update),
        )
    finally:
        del model
    heartbeat.stop()

    payload["vram_cap"] = vram_cap
    payload["corpus_sha256"] = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.out_md)
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
