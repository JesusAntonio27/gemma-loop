"""Evaluate pilot checkpoints at the registered N values.

The expensive model path is intentionally small: temperatures are fitted per
``(arm, checkpoint, N)`` on the fit split, and each test row keeps its problem
cluster so the bootstrap can resample clusters.  Tests inject ``load_fn``,
``fit_fn`` and ``score_fn``; a real run uses :mod:`gemmaloop.eval.calibrated_nll`
through the defaults below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .arms import ArmSpec
from .statistics import DEFAULT_ITERS, DEFAULT_SEED, arm_endpoints, cluster_mean

__all__ = [
    "EvalRow",
    "KL_SUBSAMPLE_ROWS",
    "TeacherKlAlignmentError",
    "build_arm_summary",
    "evaluate_arm",
    "load_eval_rows",
    "matched_teacher_rows",
    "read_training_summary",
    "select_kl_rows",
    "teacher_kl",
    "teacher_rows_by_hash",
    "teacher_shard_stats",
]

N_VALUES = (1, 2, 3, 4, 6, 8)
# The why report's teacher-KL diagnostic is descriptive and cost-capped: a
# fixed deterministic subsample, never the whole 3M-token heal corpus.
KL_SUBSAMPLE_ROWS = 64


class TeacherKlAlignmentError(ValueError):
    """The heal rows and cached teacher shards cannot be paired by identity."""


@dataclass(frozen=True)
class EvalRow:
    row_id: int
    cluster: str
    input_ids: list[int]
    labels_start: int = 0
    # The exact training label mask (``-100`` outside the supervised tokens).
    # When present, evaluation scores precisely the positions training does.
    labels: tuple[int, ...] | None = None
    answer_start: int | None = None
    thinking_start: int | None = None


def select_kl_rows(
    heal_rows: Sequence[EvalRow], limit: int = KL_SUBSAMPLE_ROWS
) -> list[EvalRow]:
    """Deterministic fixed-size KL subsample: lowest 64 row ids, in order."""

    ordered = sorted(heal_rows, key=lambda row: int(row.row_id))
    return ordered[: max(0, int(limit))]


def _labels_start(payload: Mapping[str, Any]) -> int:
    """The first supervised position: explicit offset, positions or mask."""

    if payload.get("labels_start") is not None:
        return int(payload["labels_start"])
    positions = payload.get("label_positions")
    if isinstance(positions, list) and positions:
        return int(min(int(value) for value in positions))
    labels = payload.get("labels")
    if isinstance(labels, list):
        for index, value in enumerate(labels):
            if int(value) != -100:
                return index
    return 0


def load_eval_rows(
    path: str | Path,
    max_rows: int | None = None,
    *,
    require_problem_id: bool = False,
) -> list[EvalRow]:
    """Read a held-out JSONL split; cluster by problem id when present."""

    rows: list[EvalRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"row {index} in {path} is not an object")
            input_ids = payload.get("input_ids")
            if not isinstance(input_ids, list) or len(input_ids) < 2:
                raise ValueError(
                    f"row {index} in {path} needs at least two input_ids"
                )
            problem_id = payload.get("problem_id")
            if require_problem_id and problem_id is None:
                raise ValueError(
                    f"row {index} in {path} has no problem_id; a registered "
                    "pilot evaluation split must carry problem_id (see "
                    "scripts/render_pilot_eval_rows.py)"
                )
            cluster = (
                payload.get("cluster_id")
                or problem_id
                or payload.get("row_id")
                or index
            )
            labels = payload.get("labels")
            label_mask = None
            if isinstance(labels, list):
                if len(labels) != len(input_ids):
                    raise ValueError(
                        f"row {index} in {path} has {len(labels)} labels for "
                        f"{len(input_ids)} input_ids"
                    )
                label_mask = tuple(int(value) for value in labels)
            rows.append(
                EvalRow(
                    row_id=int(payload.get("row_id", index)),
                    cluster=str(cluster),
                    input_ids=[int(value) for value in input_ids],
                    labels_start=_labels_start(payload),
                    labels=label_mask,
                    answer_start=(
                        None
                        if payload.get("answer_start") is None
                        else int(payload["answer_start"])
                    ),
                    thinking_start=(
                        None
                        if payload.get("thinking_start") is None
                        else int(payload["thinking_start"])
                    ),
                )
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    return rows


def _row_batch(row: EvalRow) -> dict[str, Any]:
    import torch

    ids = torch.tensor(row.input_ids, dtype=torch.long)
    if row.labels is not None:
        labels = torch.tensor(list(row.labels), dtype=torch.long)
        if labels.numel() != ids.numel():
            raise ValueError(
                f"eval row {row.row_id} has {labels.numel()} labels for "
                f"{ids.numel()} input_ids"
            )
    else:
        labels = ids.clone()
        labels[: max(0, min(row.labels_start, ids.numel()))] = -100
    return {
        "input_ids": ids.unsqueeze(0),
        "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
        "labels": labels.unsqueeze(0),
    }


def _flops_ratio(arm: ArmSpec, model: Any) -> float | None:
    try:
        from gemmaloop.model.flops import per_layer_flops
        from gemmaloop.model.layer_plan import make_layer_plan

        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        if text_config is None or not hasattr(text_config, "to_dict"):
            return None
        payload = text_config.to_dict()
        plan = make_layer_plan(payload, block=arm.block)
        layers = per_layer_flops(payload)
        constant = int(getattr(layers, "constant_flops", 0))
        total = constant + sum(int(value) for value in layers)
        loop = sum(int(layers[index]) for index in plan.loop)
        if arm.kind != "L":
            return 1.0
        return float((total + 3 * loop) / total) if total else None
    except Exception:
        return None


def _default_load_fn(arm: ArmSpec, run_dir: str | Path):
    def load(checkpoint: str):
        from gemmaloop.model.load import ModelSpec, load_model
        from gemmaloop.model.looped import LoopConfig
        from gemmaloop.train.checkpoint import find_latest_valid_checkpoint

        adapters: tuple[str, ...] = ()
        if checkpoint != "initial":
            directory = Path(run_dir)
            if (directory / "adapter_model.safetensors").is_file():
                adapters = (str(directory),)
            else:
                info = find_latest_valid_checkpoint(directory)
                if info is None:
                    raise FileNotFoundError(
                        f"no complete checkpoint in {directory} for {arm.arm_id}"
                    )
                adapters = (str(info.path),)
        loop = None
        if arm.kind == "L":
            loop = LoopConfig(
                n_loops=1,
                bridge=arm.bridge,
                block=arm.block,
                adapter_policy="loop_only",
                lora_target_layers="loop_block",
            )
        else:
            loop = None
        spec = ModelSpec(size="e2b", quant="bf16", adapters=adapters, loop=loop)
        model, _processor, _plan = load_model(spec)
        return model

    return load


def _default_fit_fn(model, batches, n):
    from gemmaloop.eval.calibrated_nll import fit_model_temperature

    return fit_model_temperature(model, batches, int(n))


def _default_score_fn(model, batches, n, temperature):
    import torch

    from gemmaloop.eval.calibrated_nll import (
        _collect,
        _score_collected,
        bucket_report,
    )

    clusters: dict[str, dict[str, float]] = {}
    bucket_values: dict[str, dict[str, float]] = {}
    span_values: dict[str, dict[str, float]] = {}
    top1_total = 0.0
    row_count = 0
    tokens_total = 0
    for row, batch in batches:
        collected, weight, softcap = _collect(model, [batch], int(n))
        scores = _score_collected(
            collected, weight, softcap, float(temperature), token_chunk=512, vocab_chunk=16384
        )
        if len(scores) == 0:
            continue
        nll = scores.nll
        cluster = clusters.setdefault(row.cluster, {"sum_loss": 0.0, "tokens": 0.0})
        cluster["sum_loss"] += float(nll.sum().item())
        cluster["tokens"] += float(len(scores))
        top1_total += float(scores.top1.sum().item())
        row_count += 1
        tokens_total += len(scores)
        report = bucket_report(nll, nll, scores.top1, scores.positions)
        for label, values in report.get("buckets", {}).items():
            if values:
                bucket = bucket_values.setdefault(
                    label, {"sum_loss": 0.0, "tokens": 0.0, "rows": 0.0}
                )
                bucket["sum_loss"] += float(values["calibrated_nll"]) * int(values["tokens"])
                bucket["tokens"] += float(values["tokens"])
                bucket["rows"] += 1.0
        positions = scores.positions.tolist()
        answer_start = row.answer_start
        import numpy as np

        values = np.asarray(nll.tolist(), dtype=float)
        pos = np.asarray(positions, dtype=int)
        if answer_start is None:
            thinking = values
            answer = values[:0]
            all_values = values
            span = span_values.setdefault("all", {"sum_loss": 0.0, "tokens": 0.0})
            span["sum_loss"] += float(all_values.sum())
            span["tokens"] += float(all_values.size)
        else:
            thinking = values[(pos >= row.labels_start) & (pos < answer_start)]
            answer = values[pos >= answer_start]
        if thinking.size:
            span = span_values.setdefault("thinking", {"sum_loss": 0.0, "tokens": 0.0})
            span["sum_loss"] += float(thinking.sum())
            span["tokens"] += float(thinking.size)
        if answer.size:
            span = span_values.setdefault("answer", {"sum_loss": 0.0, "tokens": 0.0})
            span["sum_loss"] += float(answer.sum())
            span["tokens"] += float(answer.size)
    return {
        "clusters": {
            key: {"sum_loss": value["sum_loss"], "tokens": value["tokens"]}
            for key, value in clusters.items()
        },
        "buckets": {
            key: {
                "calibrated_nll": (
                    value["sum_loss"] / value["tokens"] if value["tokens"] else None
                ),
                "tokens": int(value["tokens"]),
                "rows": int(value["rows"]),
            }
            for key, value in bucket_values.items()
        },
        "spans": {
            key: value["sum_loss"] / value["tokens"]
            for key, value in span_values.items()
            if value["tokens"]
        },
        "top1_accuracy": top1_total / max(tokens_total, 1),
        "tokens": tokens_total,
    }


def evaluate_arm(
    arm: ArmSpec,
    run_dir: str | Path,
    *,
    fit_rows: Sequence[EvalRow],
    test_rows: Sequence[EvalRow],
    n_values: Sequence[int] = N_VALUES,
    checkpoints: Sequence[str] = ("initial", "final"),
    load_fn: Callable[[str], Any] | None = None,
    fit_fn: Callable[[Any, Sequence[Any], int], float] | None = None,
    score_fn: Callable[[Any, Sequence[Any], int, float], Mapping] | None = None,
    teacher_dir: str | Path | None = None,
    heal_rows: Sequence["EvalRow"] | None = None,
    kl_fn: Callable[[Any, Sequence["EvalRow"], str | Path, int], float] | None = None,
) -> dict:
    """Fit and score every requested ``(checkpoint, N)`` pair."""

    if not fit_rows:
        raise ValueError("the temperature fit split is empty")
    if not test_rows:
        raise ValueError("the evaluation test split is empty")
    load_fn = load_fn or _default_load_fn(arm, run_dir)
    fit_fn = fit_fn or _default_fit_fn
    score_fn = score_fn or _default_score_fn

    fit_batches = [_row_batch(row) for row in fit_rows]
    test_batches = [(row, _row_batch(row)) for row in test_rows]

    out: dict = {"arm_id": arm.arm_id, "checkpoints": {}, "flops_ratio": None}
    for checkpoint in checkpoints:
        model = load_fn(checkpoint)
        per_n: dict[str, dict] = {}
        for n in n_values:
            temperature = float(fit_fn(model, fit_batches, int(n)))
            metrics = dict(score_fn(model, test_batches, int(n), temperature))
            metrics["temperature"] = temperature
            per_n[str(int(n))] = metrics
        out["checkpoints"][str(checkpoint)] = {"n": per_n}
        if teacher_dir is not None and heal_rows:
            import time

            kl_rows = select_kl_rows(heal_rows)
            kl_values: dict[str, float] = {}
            kl_errors: dict[str, str] = {}
            kl_n = (1,) if arm.kind == "O-H" else tuple(int(n) for n in n_values)
            started = time.perf_counter()
            for n in kl_n:
                try:
                    value = (
                        kl_fn(model, kl_rows, teacher_dir, int(n))
                        if kl_fn is not None
                        else teacher_kl(model, kl_rows, teacher_dir, n=int(n))
                    )
                    kl_values[str(int(n))] = float(value)
                except Exception as exc:  # reported, never silently swallowed
                    kl_errors[str(int(n))] = f"{type(exc).__name__}: {exc}"
            cost = time.perf_counter() - started
            out["checkpoints"][str(checkpoint)]["teacher_kl_by_n"] = kl_values
            out["checkpoints"][str(checkpoint)]["teacher_kl_cost_seconds"] = cost
            out["checkpoints"][str(checkpoint)]["teacher_kl_rows"] = len(kl_rows)
            out["checkpoints"][str(checkpoint)]["teacher_kl_role"] = "descriptive"
            out["checkpoints"][str(checkpoint)]["teacher_kl_forward_passes"] = (
                len(kl_rows) * len(kl_n)
            )
            if kl_errors:
                out["checkpoints"][str(checkpoint)]["teacher_kl_errors"] = kl_errors
        if out["flops_ratio"] is None and arm.kind == "L":
            out["flops_ratio"] = _flops_ratio(arm, model)
    initial = out["checkpoints"].get("initial", {}).get("teacher_kl_by_n") or {}
    final = out["checkpoints"].get("final", {}).get("teacher_kl_by_n") or {}
    errors = {
        **(out["checkpoints"].get("initial", {}).get("teacher_kl_errors") or {}),
        **(out["checkpoints"].get("final", {}).get("teacher_kl_errors") or {}),
    }
    out["teacher_kl"] = None
    if initial and final:
        common = sorted(set(initial) & set(final), key=int)
        out["teacher_kl"] = {
            "initial": {n: float(initial[n]) for n in common},
            "final": {n: float(final[n]) for n in common},
            "gap_closed": {
                n: (1.0 - float(final[n]) / float(initial[n]))
                if float(initial[n])
                else None
                for n in common
            },
            "n_values": [int(n) for n in common],
            "note": (
                "N=1 is structurally the original model for L arms; its gap is "
                "reported for completeness and is expected to be zero."
            ),
        }
        if errors:
            out["teacher_kl"]["errors"] = errors
        out["teacher_kl"]["rows"] = min(
            [
                int(out["checkpoints"][name].get("teacher_kl_rows", 0))
                for name in out["checkpoints"]
            ]
        )
        out["teacher_kl"]["role"] = "descriptive"
        out["teacher_kl"]["cost_seconds"] = sum(
            float(out["checkpoints"][name].get("teacher_kl_cost_seconds", 0.0))
            for name in out["checkpoints"]
        )
        out["teacher_kl"]["forward_passes"] = sum(
            int(out["checkpoints"][name].get("teacher_kl_forward_passes", 0))
            for name in out["checkpoints"]
        )
    return out


def _final_metrics(evaluation: Mapping, checkpoint: str = "final") -> Mapping:
    checkpoints = evaluation.get("checkpoints") or {}
    selected = checkpoints.get(checkpoint) or {}
    return selected.get("n") or {}


def build_arm_summary(
    arm: ArmSpec,
    evaluation: Mapping,
    oh_evaluation: Mapping | None = None,
    *,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Turn one arm's evaluation into the selection/report summary."""

    final = _final_metrics(evaluation)
    nll_by_n = {
        int(n): dict(payload.get("clusters") or {})
        for n, payload in final.items()
        if payload.get("clusters")
    }
    summary = dict(arm.to_dict())
    summary["flops_ratio"] = evaluation.get("flops_ratio")
    summary["temperatures"] = {
        str(n): payload.get("temperature") for n, payload in final.items()
    }
    summary["training"] = evaluation.get("training") or {}

    if arm.kind == "O-H":
        summary["nll"] = {str(n): _cluster_mean(values) for n, values in nll_by_n.items()}
        summary["delta_loop"] = None
        summary["delta_loop_lower"] = None
        summary["delta_arm"] = None
        summary["delta_bar_arm"] = None
        summary["monotonic"] = None
        return summary

    oh_by_n: dict[int, Mapping[str, float]] | None = None
    if oh_evaluation is not None:
        oh_final = _final_metrics(oh_evaluation)
        if oh_final:
            first = sorted(int(n) for n in oh_final)[0]
            oh_by_n = {1: dict(oh_final[str(first)].get("clusters") or {})}
    if any(n not in nll_by_n for n in N_VALUES):
        summary["nll"] = {str(n): _cluster_mean(values) for n, values in nll_by_n.items()}
        summary["delta_loop"] = None
        summary["delta_loop_lower"] = None
        summary["delta_arm"] = None
        summary["delta_bar_arm"] = None
        summary["monotonic"] = False
        return summary

    stats = arm_endpoints(nll_by_n, oh_by_n, iters=iters, seed=seed)
    summary.update(stats)
    summary["nll"] = stats["nll"]
    return summary


def _all_teacher_rows(directory: str | Path) -> list[Any]:
    from gemmaloop.data.teacher_topk import TeacherManifest, read_teacher_shard

    root = Path(directory)
    manifest = TeacherManifest.read(root / "teacher_manifest.json")
    rows: list[Any] = []
    for entry in manifest.shards:
        rows.extend(read_teacher_shard(root / entry.file))
    return rows


def teacher_rows_by_hash(directory: str | Path) -> dict[str, Any]:
    """Read cached shards and key teacher rows by their input-id hash."""

    return {
        str(row.input_ids_sha256): row
        for row in _all_teacher_rows(directory)
        if row.input_ids_sha256
    }


def matched_teacher_rows(
    heal_rows: Sequence[EvalRow], teacher_dir: str | Path
) -> list[tuple[EvalRow, Any]]:
    """Pair heal rows to shards by rendered input-id hash, never by index."""

    from gemmaloop.data.teacher_topk import input_ids_sha256

    by_hash = teacher_rows_by_hash(teacher_dir)
    matched: list[tuple[EvalRow, Any]] = []
    for row in heal_rows:
        teacher_row = by_hash.get(input_ids_sha256(row.input_ids))
        if teacher_row is not None:
            matched.append((row, teacher_row))
    if not matched:
        raise TeacherKlAlignmentError(
            f"none of {len(heal_rows)} heal rows matched a cached teacher shard "
            "by input_ids_sha256"
        )
    return matched


def teacher_shard_stats(directory: str | Path) -> dict:
    """Top-k entropy and top-1 confidence on the cached heal shards."""

    try:
        rows = _all_teacher_rows(directory)
    except Exception:
        return {}
    entropy_values: list[float] = []
    top1_values: list[float] = []
    tokens = 0
    for row in rows:
        probs = row.probs.float()
        tail = row.tail.float().clamp(min=0.0)
        token_entropy = -(probs * torch_log(probs)).sum(dim=1)
        token_entropy = token_entropy - tail * torch_log(tail)
        entropy_values.extend(float(value) for value in token_entropy.tolist())
        top1_values.extend(float(value) for value in probs.max(dim=1).values.tolist())
        tokens += int(probs.shape[0])
    if not tokens:
        return {}
    return {
        "entropy": sum(entropy_values) / len(entropy_values),
        "top1_confidence": sum(top1_values) / len(top1_values),
        "tokens": tokens,
    }


def torch_log(values):
    import torch

    return torch.log(values.clamp(min=1e-12))


def _hidden_and_targets(model, batch, n):
    from gemmaloop.eval.calibrated_nll import _collect

    collected, weight, softcap = _collect(model, [batch], int(n))
    return collected, weight, softcap


def _student_log_probs(hidden, weight, indices, softcap, *, vocab_chunk: int = 16384):
    """Student log-probs at ``indices`` plus the student tail mass."""

    import torch

    n_tokens, k = indices.shape
    device = weight.device
    h = hidden.to(device=device, dtype=torch.float32)
    running_max = torch.full((n_tokens,), float("-inf"), device=device)
    sum_exp = torch.zeros((n_tokens,), device=device)
    gathered = torch.zeros((n_tokens, k), device=device)
    vocab = int(weight.shape[0])
    for v0 in range(0, vocab, vocab_chunk):
        v1 = min(v0 + vocab_chunk, vocab)
        tile = weight[v0:v1].to(device=device, dtype=torch.float32)
        logits = h @ tile.t()
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        local = indices - v0
        inside = (local >= 0) & (local < (v1 - v0))
        local_index = local.clamp(0, (v1 - v0) - 1)
        tile_values = logits.gather(1, local_index)
        gathered = torch.where(inside, tile_values, gathered)
        tile_max = logits.max(dim=1).values
        new_max = torch.maximum(running_max, tile_max)
        sum_exp = sum_exp * torch.exp(running_max - new_max) + torch.exp(
            logits - new_max.unsqueeze(1)
        ).sum(dim=1)
        running_max = new_max
    logz = running_max + torch.log(sum_exp)
    log_q = gathered - logz.unsqueeze(1)
    q = log_q.exp()
    tail_q = (1.0 - q.sum(dim=1)).clamp(min=1e-12)
    return log_q, tail_q


def teacher_kl(
    model,
    heal_rows: Sequence[EvalRow],
    teacher_dir: str | Path,
    *,
    n: int = 1,
) -> float:
    """Mean KL(teacher top-k+tail || student at N) on the cached heal rows."""

    import torch

    pairs = matched_teacher_rows(heal_rows, teacher_dir)
    total = 0.0
    count = 0
    for row, teacher_row in pairs:
        if teacher_row.tokens == 0:
            continue
        collected, weight, softcap = _hidden_and_targets(model, _row_batch(row), n)
        hidden = collected.hidden[0] if collected.hidden else None
        if hidden is None or hidden.numel() == 0:
            continue
        positions = collected.positions[0].tolist()
        index_by_position = {int(p): i for i, p in enumerate(positions)}
        keep = []
        for j, position in enumerate(teacher_row.positions.tolist()):
            if int(position) in index_by_position:
                keep.append((index_by_position[int(position)], j))
        if not keep:
            continue
        student_rows = torch.tensor([item[0] for item in keep], dtype=torch.long)
        teacher_rows = torch.tensor([item[1] for item in keep], dtype=torch.long)
        h = hidden[student_rows]
        teacher_probs = teacher_row.probs[teacher_rows].float()
        teacher_indices = teacher_row.indices[teacher_rows].long()
        teacher_tail = teacher_row.tail[teacher_rows].float().clamp(min=1e-12)
        log_q, tail_q = _student_log_probs(
            h, weight, teacher_indices.to(weight.device), softcap
        )
        log_p = torch_log(teacher_probs.to(log_q.device))
        kl = (teacher_probs.to(log_q.device) * (log_p - log_q)).sum(dim=1)
        kl = kl + teacher_tail.to(log_q.device) * (
            torch.log(teacher_tail.to(log_q.device)) - torch.log(tail_q)
        )
        total += float(kl.sum().item())
        count += int(kl.numel())
    if count == 0:
        raise ValueError("no aligned teacher/student tokens for KL")
    return total / count


def _cluster_mean(values: Mapping[str, Any]) -> float:
    return cluster_mean(values)


def read_training_summary(run_dir: str | Path) -> dict:
    """Read loss history, tokens and cost from a finished run directory."""

    directory = Path(run_dir)
    out: dict = {"loss": [], "tokens": None, "steps": None, "wall_seconds": None}
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        out["tokens"] = metrics.get("tokens_consumed")
        out["steps"] = metrics.get("global_step")
        out["loss"] = [
            {"step": entry.get("step"), "loss": entry.get("loss")}
            for entry in metrics.get("training_metrics", [])
            if isinstance(entry, Mapping) and entry.get("loss") is not None
        ]
    run_path = directory / "run.json"
    if run_path.is_file():
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        out["tokens"] = out["tokens"] or payload.get("tokens_consumed")
        out["steps"] = out["steps"] or payload.get("global_step")
        out["wall_seconds"] = payload.get("wall_seconds")
        out["tokens_per_second"] = payload.get("tokens_per_second")
        if not out["loss"] and payload.get("training_metrics"):
            out["loss"] = [
                {"step": entry.get("step"), "loss": entry.get("loss")}
                for entry in payload["training_metrics"]
                if isinstance(entry, Mapping) and entry.get("loss") is not None
            ]
    wall_file = directory / "pilot_wall_seconds.txt"
    if out.get("wall_seconds") is None and wall_file.is_file():
        try:
            out["wall_seconds"] = float(wall_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
    if out.get("wall_seconds") and out.get("tokens"):
        out["tokens_per_second"] = float(out["tokens"]) / float(out["wall_seconds"])
    return out
