"""Local Stage-0-lite sampler for the pilot (registered padded engine).

Stage 0 normally runs on Modal with stock vLLM.  The pilot's teacher step is
the critical path, so this script produces the same registered data locally:

* O = ``google/gemma-4-E2B-it`` at the pinned revision, bf16, on the padded
  adaptive-batch HF engine (:func:`gemmaloop.serve.generate.generate_batch`);
* registered Stage 0 sampling: temperature 1.0, top_p 0.95, top_k 64, thinking
  on, four samples per prompt, per-request seeds;
* the same request identity as the Modal records:
  ``"{problem_id}:{sample_index}"`` (the eval runner's ``sha256(seed:request_id)``
  scheme; the Modal job writes the same problem/sample pair);
* ``max_new_tokens = 8192 - rendered_prompt_tokens`` per request.  A request that
  would stop before the registered 8192 rendered limit is unaffected by the cap;
  one that would run past it is dropped by the registered filter either way, so
  the accepted set is exactly the one a 32,768-token run keeps.  A generation
  call may hold requests with different budgets; its engine cap is the largest
  per-request budget in the call, never smaller than any member's budget.

Targets (all from ``prereg/prereg.yaml``):

* train bucket 0-49: healed rows until >= 3,000,000 accepted assistant tokens
  pass the registered filter (finish_reason=stop, closed thought, non-empty
  answer, rendered <= 8192, reasoning share >= 75 % for code + math);
* pilot P1 bucket 59 (50 problems x 4) and the P2 fit split 57-58 (100 problems
  x 4), plus the P2 selection split 54-56 (150 problems x 4), exporting
  verified-correct traces.  Math is verified with the declared ``math-verify``
  dependency under the registered OR rule (math-verify or the AIME/integer
  normalisation; the grade records ``decided_by``).  The script refuses to
  start if math-verify is missing unless ``--allow-math-fallback`` is passed.  Code is verified only when ``--verify-code`` is given, through the
  registered Stage 0 grader (Docker sandbox by default).

Outputs are the Stage 0 shard format (``shards/shard_XXXXX.jsonl`` with
``.committed`` markers, resumable per request) plus ``heal/train.jsonl`` for
``scripts/precompute_teacher_topk.py``, ``heal/stage1_heal.jsonl``,
``pilot/pilot_p1.jsonl``, ``pilot/pilot_tfit.jsonl``,
``pilot/pilot_selection.jsonl``, ``request_ids.txt`` and ``manifest.json``.
``python -m scripts.sample_stage0_local --verify-manifest <out>`` re-checks the
manifest and every file hash.

The pool is used when present.  Otherwise it is built with the registered
builder, which downloads the pinned Hub snapshots:

    uv run python scripts/build_stage0_pool.py \
        --output-dir artifacts/stage0/pool --part1

Launch on the GPU host (one GPU job at a time)::

    powershell -ExecutionPolicy Bypass -File scripts/launch_detached.ps1 `
      -Name stage0lite `
      -Arguments "-m scripts.sample_stage0_local --pool-dir artifacts/stage0/pool --output-dir artifacts/stage0/local"

This command is math-only (the registered default).  Code traces additionally
need ``--verify-code`` and the registered hidden-tests file
(``--hidden-tests artifacts/stage0/hidden_tests.jsonl``, default).

A dry run (no lock, no model, no download) prints the plan and the exact build
command::

    uv run --no-sync python scripts/sample_stage0_local.py --dry-run

CPU tests drive the pure helpers with a fake generator; this script is the only
part that touches the GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemmaloop.gpulock import require_gpu_lock  # noqa: E402
from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path  # noqa: E402
from gemmaloop.prereg import default_prereg_path, load_prereg  # noqa: E402
from gemmaloop.stage0 import local  # noqa: E402
from gemmaloop.stage0.postprocess import (  # noqa: E402
    MAX_LENGTH,
    MIN_REASONING_SHARE,
    extract_code,
)

BUILD_COMMAND = (
    "uv run python scripts/build_stage0_pool.py "
    "--output-dir {pool_dir} --part1"
)
MODEL_SIZE = "e2b"
MODEL_QUANT = "bf16"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=Path("artifacts/stage0/pool"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage0/local"))
    parser.add_argument("--prereg", type=Path, default=default_prereg_path())
    parser.add_argument("--shard-size", type=int, default=local.DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="registered order-key seed for the train/monitor problem partition",
    )
    parser.add_argument("--batch", type=int, default=None, help="override max_batch")
    parser.add_argument("--limit-shards", type=int, default=None, help="smoke runs")
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "restrict the run to one or more targets (train, monitor, "
            "pilot_p1, pilot_tfit, pilot_selection); repeatable. The monitor "
            "target samples the held-out monitor corpus: the 64 largest "
            "sha256(seed:problem_id) keys in the train bucket."
        ),
    )
    parser.add_argument("--no-build-pool", action="store_true")
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--allow-math-fallback", action="store_true")
    parser.add_argument("--verify-code", action="store_true")
    parser.add_argument(
        "--hidden-tests",
        type=Path,
        default=Path("artifacts/stage0/hidden_tests.jsonl"),
        help="registered Stage 0 hidden-test file (required with --verify-code)",
    )
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    parser.add_argument("--allow-unsandboxed", action="store_true")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--vram-fraction", type=float, default=0.90)
    parser.add_argument("--vram-margin-gib", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="check an existing manifest and exit",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pool / model / verification setup
# ---------------------------------------------------------------------------
def build_pool_command(pool_dir: Path) -> str:
    return BUILD_COMMAND.format(pool_dir=str(pool_dir))


def ensure_pool(
    pool_dir: Path, *, build: bool, dry_run: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    manifest_path = Path(pool_dir) / "manifest.json"
    problems_path = Path(pool_dir) / "problems.jsonl"
    if manifest_path.exists() and problems_path.exists():
        return local.load_pool(Path(pool_dir))
    if dry_run:
        return None
    if not build:
        raise RuntimeError(
            f"no Stage 0 pool at {pool_dir}; run: {build_pool_command(Path(pool_dir))}"
        )
    command = [
        sys.executable,
        "-m",
        "scripts.build_stage0_pool",
        "--output-dir",
        str(pool_dir),
        "--part1",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return local.load_pool(Path(pool_dir))


def make_prompt_tokens_fn(tokenizer: Any) -> Callable[[Mapping[str, Any]], int]:
    def prompt_tokens(problem: Mapping[str, Any]) -> int:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(problem.get("prompt") or "")}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(local.SAMPLING["enable_thinking"]),
            return_dict=True,
        )
        ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return int(len(ids))

    return prompt_tokens


def make_evaluator(
    tokenizer: Any,
    *,
    max_length: int = MAX_LENGTH,
    min_reasoning_share: float = MIN_REASONING_SHARE,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], local.SampleVerdict]:
    def count(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    def evaluate(record: Mapping[str, Any], problem: Mapping[str, Any]):
        return local.evaluate_sample(
            record,
            problem,
            tokenizer=tokenizer,
            token_counter=count,
            max_length=max_length,
            min_reasoning_share=min_reasoning_share,
        )

    return evaluate


def resolve_hidden_tests(path: Path, *, verify_code: bool) -> Path | None:
    """Return the registered hidden-tests path or refuse code verification."""

    if not verify_code:
        return None
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            "code verification needs the registered Stage 0 hidden-tests file; "
            f"{path} does not exist. Run scripts/write_stage0_hidden_tests.py "
            "or pass --hidden-tests <file>, or omit --verify-code."
        )
    return path


def make_verify_fn(
    *,
    verify_code: bool,
    grader: Any = None,
    use_math_verify: bool = True,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Any, Any]]:
    from gemmaloop.stage0.grader import grade_math

    def verify(sample: Mapping[str, Any], problem: Mapping[str, Any]):
        kind = str(problem.get("kind") or "")
        if kind == "math":
            result = grade_math(
                str(sample.get("completion") or sample.get("content") or ""),
                str(problem.get("answer") or ""),
                use_math_verify=use_math_verify,
            )
            return result.correct, result.to_dict()
        if kind == "code" and verify_code:
            code = extract_code(dict(sample))
            if not code:
                return False, {"passed": False, "error": "no fenced code block"}
            result = grader.grade_code(code, list(problem.get("tests") or []))
            return bool(result.passed), result.to_dict()
        return None, None

    return verify


def _require_math_verify(args: argparse.Namespace) -> bool:
    from gemmaloop.stage0.grader import MATH_VERIFY_AVAILABLE

    if MATH_VERIFY_AVAILABLE:
        return True
    if args.allow_math_fallback:
        return False
    raise SystemExit(
        "math-verify is not installed in this environment (it is a declared "
        "dependency: run `uv sync`) or pass --allow-math-fallback to use the "
        "registered AIME normaliser alone."
    )


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------
def collect_records(
    output_dir: Path,
    *,
    run_id: str | None = None,
    pool_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard in sorted((Path(output_dir) / "shards").glob("shard_*.jsonl")):
        rows = local.read_jsonl(shard)
        if run_id is not None and pool_manifest_sha256 is not None:
            local._check_run_identity(
                rows,
                run_id=run_id,
                pool_manifest_sha256=pool_manifest_sha256,
                source=shard,
            )
        records.extend(rows)
    return local._dedupe(records)


def finalize(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    problems: Sequence[Mapping[str, Any]],
    targets: Sequence[local.Target],
    pool_dir: Path,
    pool_manifest: Mapping[str, Any],
    problems_path: Path,
    run_id: str,
    model_revision: str,
    run_stats: local.RunStats,
    evaluate_filter: Callable,
    evaluate_full: Callable,
    verifiers: Mapping[str, Any],
    hidden_tests: Mapping[str, Any] | None = None,
    seed: int = 0,
    engine: str = local.ENGINE,
    engine_note: str | None = None,
    train_goal: int | None = None,
    train_shard_order: Sequence[int] | None = None,
    max_train_tokens: int | None = None,
) -> tuple[local.RunStats, dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems_by_id = {str(problem["problem_id"]): problem for problem in problems}
    records = collect_records(
        output_dir,
        run_id=run_id,
        pool_manifest_sha256=str(pool_manifest.get("manifest_sha256")),
    )

    files: dict[str, int] = {}
    target_by_name = {target.name: target for target in targets}

    monitor_ids = local.monitor_problem_ids(problems, targets, seed=int(seed))
    # Evaluate once: only the verified-trace buckets need the expensive
    # verifier; the train heal rows use the registered filter alone.
    verdicts: dict[str, local.SampleVerdict] = {}
    for record in records:
        problem = problems_by_id.get(str(record.get("problem_id")))
        if problem is None:
            continue
        target = local.target_for_problem(problem, targets, monitor_ids)
        verdicts[str(record["request_id"])] = (
            evaluate_full(record, problem)
            if target is not None and target.verified
            else evaluate_filter(record, problem)
        )

    grade_rows = [
        local.sample_grade_row(
            record,
            problems_by_id[str(record["problem_id"])],
            verdicts[str(record["request_id"])],
        )
        for record in records
        if str(record.get("problem_id")) in problems_by_id
    ]
    grade_rows.sort(
        key=lambda row: (str(row["problem_id"]), int(row["sample_index"]))
    )
    local.write_jsonl_atomic(output_dir / "sample_grades.jsonl", grade_rows)
    files["sample_grades.jsonl"] = len(grade_rows)
    decision_counts = Counter(
        str(row["decided_by"]) for row in grade_rows if row.get("decided_by")
    )
    timed_out = sum(1 for row in grade_rows if row.get("timed_out"))

    stats = local.RunStats(
        shards_committed=list(run_stats.shards_committed),
        shards_skipped=list(run_stats.shards_skipped),
        problems_skipped=dict(run_stats.problems_skipped),
    )
    for record in records:
        problem = problems_by_id.get(str(record.get("problem_id")))
        if problem is not None:
            local._accumulate(
                stats,
                record,
                problem,
                verdicts[str(record["request_id"])],
                targets,
                monitor_ids=monitor_ids,
            )
    for name, problem_ids in stats.sampled_problem_ids.items():
        entry = stats.per_target.setdefault(
            name, local._empty_target_stats(next(t for t in targets if t.name == name))
        )
        entry["sampled_problems"] = len(problem_ids)
    for target in targets:
        stats.per_target.setdefault(target.name, local._empty_target_stats(target))

    def kept_records(target_name: str, *, verified: bool) -> list[tuple[dict, dict, local.SampleVerdict]]:
        out = []
        for record in records:
            problem = problems_by_id.get(str(record.get("problem_id")))
            if problem is None:
                continue
            target = local.target_for_problem(problem, targets, monitor_ids)
            if target is None or target.name != target_name:
                continue
            verdict = verdicts[str(record["request_id"])]
            if not verdict.kept:
                continue
            if verified and verdict.correct is not True:
                continue
            out.append((record, dict(problem), verdict))
        return out

    # Stage 1 heal rows from the train bucket until the registered token target.
    heal_all = kept_records("train", verified=False)
    train_prefix: dict[str, Any] = {}
    if train_goal is not None and train_shard_order is not None:
        goal = int(train_goal)
        order = [int(index) for index in train_shard_order]
        by_shard: dict[int, list[tuple[dict, dict, local.SampleVerdict]]] = {}
        for item in heal_all:
            by_shard.setdefault(int(item[0].get("shard_index", -1)), []).append(item)
        records_by_shard: dict[int, list[dict]] = {}
        for record in records:
            records_by_shard.setdefault(
                int(record.get("shard_index", -1)), []
            ).append(record)
        heal = []
        kept_shards: list[int] = []
        excluded_shards: set[int] = set()
        cumulative = 0
        reached = False
        for index in order:
            items = by_shard.get(index, [])
            if reached:
                if items:
                    excluded_shards.add(index)
                continue
            heal.extend(items)
            cumulative += int(
                sum(int(item[2].assistant_tokens) for item in items)
            )
            kept_shards.append(index)
            if goal > 0 and cumulative >= goal:
                reached = True
        for index, items in by_shard.items():
            if index not in order and items:
                excluded_shards.add(index)
        excluded_heal_rows = int(
            sum(len(by_shard.get(index, [])) for index in excluded_shards)
        )
        excluded_records = int(
            sum(len(records_by_shard.get(index, [])) for index in excluded_shards)
        )
        train_prefix = {
            "train_goal": goal,
            "train_goal_reached": bool(reached),
            "train_shards_kept": kept_shards,
            "train_shards_excluded": sorted(excluded_shards),
            "excluded_train_heal_rows": excluded_heal_rows,
            "excluded_train_records": excluded_records,
            "kept_train_assistant_tokens": cumulative,
        }
    else:
        heal = heal_all
    heal_rows = [local.heal_row(record, problem, verdict) for record, problem, verdict in heal]
    heal_traces = [
        local.trace_record(record, problem, verdict) for record, problem, verdict in heal
    ]
    sha = local.write_jsonl_atomic(output_dir / "heal" / "train.jsonl", heal_rows)
    files["heal/train.jsonl"] = len(heal_rows)
    local.write_jsonl_atomic(output_dir / "heal" / "stage1_heal.jsonl", heal_traces)
    files["heal/stage1_heal.jsonl"] = len(heal_traces)
    heal_manifest = {
        "kind": "stage0_local_heal",
        "version": 1,
        "sha256": sha,
        "rows": len(heal_rows),
        "assistant_tokens": int(sum(item[2].assistant_tokens for item in heal)),
        "max_length": MAX_LENGTH,
        "length_mode": "filter",
        "engine": str(engine),
        "model_revision": model_revision,
        "source": str(output_dir / "shards"),
        **train_prefix,
    }
    local.write_json_atomic(output_dir / "heal" / "manifest.json", heal_manifest)
    files["heal/manifest.json"] = 1

    # Pilot traces: the P1 corpus and the P2 selection / fit splits stay separate.
    pilot_counts: dict[str, Any] = {}
    for name in ("pilot_p1", "pilot_tfit", "pilot_selection"):
        target = target_by_name.get(name)
        if target is None:
            continue
        selected = kept_records(name, verified=True)
        rows = [local.trace_record(record, problem, verdict) for record, problem, verdict in selected]
        local.write_jsonl_atomic(output_dir / "pilot" / f"{name}.jsonl", rows)
        files[f"pilot/{name}.jsonl"] = len(rows)
        pilot_counts[name] = {
            "rows": len(rows),
            "rows_target": int(target.rows_target),
            "problems": int(target.problems),
            "verified_correct": len(rows),
        }
    if pilot_counts:
        local.write_json_atomic(
            output_dir / "pilot" / "manifest.json",
            {
                "kind": "stage0_local_pilot",
                "version": 1,
                "engine": str(engine),
                "model_revision": model_revision,
                "splits": pilot_counts,
            },
        )
        files["pilot/manifest.json"] = 1

    # Held-out MONITOR corpus: the 64 largest train-bucket order keys, one
    # sample each; every training target takes the smallest keys and excludes
    # these problems.
    monitor_manifest: dict[str, Any] | None = None
    monitor_target = target_by_name.get("monitor")
    if monitor_target is not None:
        selected = kept_records("monitor", verified=False)
        monitor_rows = [
            local.heal_row(record, problem, verdict)
            for record, problem, verdict in selected
        ]
        monitor_sha = local.write_jsonl_atomic(
            output_dir / "monitor" / "monitor_rows.jsonl", monitor_rows
        )
        files["monitor/monitor_rows.jsonl"] = len(monitor_rows)
        problem_ids = sorted(
            {str(row["problem_id"]) for row in monitor_rows}
        )
        monitor_manifest = {
            "kind": "stage0_local_monitor",
            "version": 1,
            "buckets": [int(monitor_target.low), int(monitor_target.high)],
            "problems": int(monitor_target.problems),
            "samples_per_prompt": int(monitor_target.samples_per_prompt),
            "rows": len(monitor_rows),
            "rows_target": int(monitor_target.rows_target),
            "problem_ids": problem_ids,
            "selection_seed": int(seed),
            "partition_seed": int(seed),
            "sha256": monitor_sha,
            "max_length": MAX_LENGTH,
            "length_mode": "filter",
            "verified": False,
            "engine": str(engine),
            "model_revision": model_revision,
            "source": str(output_dir / "monitor" / "monitor_rows.jsonl"),
        }
        local.write_json_atomic(
            output_dir / "monitor" / "manifest.json", monitor_manifest
        )
        files["monitor/manifest.json"] = 1

    request_ids = sorted({str(record["request_id"]) for record in records})
    request_path = output_dir / "request_ids.txt"
    local.write_text_atomic(
        request_path,
        chr(10).join(request_ids) + (chr(10) if request_ids else ""),
    )
    files["request_ids.txt"] = len(request_ids)

    for shard in sorted((output_dir / "shards").glob("shard_*.jsonl")):
        files[f"shards/{shard.name}"] = len(local.read_jsonl(shard))
    for marker in sorted((output_dir / "shards").glob("shard_*.jsonl.committed")):
        files[f"shards/{marker.name}"] = 0
    for partial in sorted((output_dir / "shards").glob("shard_*.jsonl.partial")):
        files[f"shards/{partial.name}"] = 0
    if (output_dir / "run_marker.json").exists():
        files["run_marker.json"] = 1

    manifest = local.build_manifest(
        output_dir=output_dir,
        pool_dir=Path(pool_dir),
        pool_manifest=pool_manifest,
        problems_path=Path(problems_path),
        run_id=run_id,
        stats=stats,
        targets=targets,
        request_ids=request_ids,
        model_revision=model_revision,
        created=datetime.now(timezone.utc).isoformat(),
        build_command=build_pool_command(Path(pool_dir)),
        verifiers=verifiers,
        files=files,
        selection_seed=int(seed),
        engine=str(engine),
        engine_note=engine_note,
    )
    manifest["pilot_splits"] = pilot_counts
    manifest["monitor"] = monitor_manifest
    manifest["heal"] = {
        "rows": len(heal_rows),
        "assistant_tokens": int(sum(item[2].assistant_tokens for item in heal)),
        "manifest_sha256": local.file_sha256(output_dir / "heal" / "manifest.json"),
        **train_prefix,
    }
    if max_train_tokens is not None:
        manifest["max_train_tokens"] = int(max_train_tokens)
    shortfalls = dict(manifest.get("shortfalls") or {})
    for name, counts in pilot_counts.items():
        if counts["rows_target"] and counts["rows"] < counts["rows_target"]:
            shortfalls[name] = {
                "verified_rows": counts["rows"],
                "rows_target": counts["rows_target"],
            }
        entry = (manifest.get("targets") or {}).get(name) or {}
        if entry.get("problems") and entry.get("sampled_problems", 0) < entry["problems"]:
            shortfalls[f"{name}_problems"] = {
                "sampled": entry.get("sampled_problems", 0),
                "problems": entry["problems"],
            }
    if monitor_manifest is not None:
        if monitor_manifest["rows_target"] and monitor_manifest["rows"] < monitor_manifest[
            "rows_target"
        ]:
            shortfalls["monitor_rows"] = {
                "rows": monitor_manifest["rows"],
                "rows_target": monitor_manifest["rows_target"],
            }
        if monitor_manifest["problems"] and len(monitor_manifest["problem_ids"]) < monitor_manifest[
            "problems"
        ]:
            shortfalls["monitor_problems"] = {
                "sampled": len(monitor_manifest["problem_ids"]),
                "problems": monitor_manifest["problems"],
            }
    manifest["shortfalls"] = shortfalls
    manifest["verification"] = {
        "sample_grades": "sample_grades.jsonl",
        "decided_by": dict(sorted(decision_counts.items())),
        "timed_out": int(timed_out),
    }
    manifest["hidden_tests"] = dict(hidden_tests) if hidden_tests else None
    manifest["manifest_sha256"] = local.manifest_sha256(manifest)
    local.write_json_atomic(output_dir / "manifest.json", manifest)
    return stats, manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _print_plan(
    *,
    args: argparse.Namespace,
    targets: Sequence[local.Target],
    problems: Sequence[Mapping[str, Any]] | None,
    build_command: str,
    seed: int = 0,
) -> None:
    payload: dict[str, Any] = {
        "dry_run": True,
        "engine": local.ENGINE,
        "pool_dir": str(args.pool_dir),
        "output_dir": str(args.output_dir),
        "build_command": build_command,
        "targets": {
            target.name: {
                "buckets": list(target.buckets),
                "problems": target.problems,
                "token_target": target.token_target,
                "rows_target": target.rows_target,
                "verified": target.verified,
                "samples_per_prompt": target.samples_per_prompt,
            }
            for target in targets
        },
        "sampling": dict(local.SAMPLING),
        "max_length": MAX_LENGTH,
        "samples_per_prompt": local.SAMPLES_PER_PROMPT,
    }
    if problems is None:
        payload["pool"] = "missing: the run will build it with build_command"
    else:
        plan = local.build_shard_plan(
            problems,
            targets,
            shard_size=args.shard_size,
            samples_per_prompt=local.SAMPLES_PER_PROMPT,
            seed=int(args.seed),
        )
        payload["pool"] = f"{len(problems)} problems"
        payload["planned_shards"] = len(plan)
        counts = {}
        monitor_ids = local.monitor_problem_ids(problems, targets, seed=int(seed))
        for target in targets:
            selected = local.select_problems(
                problems,
                target,
                samples_per_prompt=local.SAMPLES_PER_PROMPT,
                seed=int(args.seed),
                monitor_ids=monitor_ids,
            )
            counts[target.name] = len(selected)
        payload["pool_problems_per_target"] = counts
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.verify_manifest is not None:
        ok, reason = local.verify_manifest(Path(args.verify_manifest))
        print(json.dumps({"ok": ok, "reason": reason, "path": str(args.verify_manifest)}))
        return 0 if ok else 1

    prereg = load_prereg(args.prereg)
    targets = local.targets_from_prereg(prereg)
    if args.max_train_tokens is not None:
        targets = [
            local.Target(
                target.name,
                target.low,
                target.high,
                target.problems,
                token_target=(
                    int(args.max_train_tokens)
                    if target.name == "train"
                    else target.token_target
                ),
                verified=target.verified,
                rows_target=target.rows_target,
                samples_per_prompt=target.samples_per_prompt,
            )
            for target in targets
        ]
    if args.target:
        wanted = {str(name) for name in args.target}
        known = {target.name for target in targets}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"unknown --target {sorted(unknown)}; known targets: {sorted(known)}"
            )
        targets = [target for target in targets if target.name in wanted]

    build_command = build_pool_command(Path(args.pool_dir))
    if args.dry_run:
        loaded = ensure_pool(Path(args.pool_dir), build=False, dry_run=True)
        problems = loaded[0] if loaded is not None else None
        _print_plan(
            args=args,
            targets=targets,
            problems=problems,
            build_command=build_command,
            seed=int(args.seed),
        )
        return 0

    require_gpu_lock()
    from gemmaloop.train.perf import cap_cuda_allocator

    vram = cap_cuda_allocator(args.vram_fraction, args.vram_margin_gib)
    heartbeat = Heartbeat(
        args.heartbeat or default_heartbeat_path("stage0lite"),
        run="stage0-local",
    )
    heartbeat.start(30.0)
    heartbeat.update(extra={"engine": local.ENGINE, "vram_cap": vram})
    try:
        # Refuse a missing hidden-tests file before any pool build/download.
        hidden_path = resolve_hidden_tests(
            args.hidden_tests, verify_code=args.verify_code
        )
        loaded = ensure_pool(Path(args.pool_dir), build=not args.no_build_pool)
        assert loaded is not None
        problems, pool_manifest = loaded
        problems_path = Path(args.pool_dir) / "problems.jsonl"
        hidden_tests = (
            local.merge_hidden_tests(problems, hidden_path)
            if hidden_path is not None
            else None
        )

        from transformers import AutoTokenizer  # noqa: PLC0415

        from gemmaloop.model.load import ModelSpec, load_model  # noqa: PLC0415
        from gemmaloop.model.snapshot import MODEL_REVISION  # noqa: PLC0415
        from gemmaloop.serve.generate import generate_batch  # noqa: PLC0415
        from gemmaloop.stage0.grader import GraderService  # noqa: PLC0415

        use_math_verify = _require_math_verify(args)
        if args.verify_code:
            if args.sandbox == "local" and not args.allow_unsandboxed:
                raise SystemExit(
                    "refusing --sandbox local without --allow-unsandboxed"
                )
            if args.sandbox == "docker" and not args.allow_unsandboxed:
                from gemmaloop.stage0.grader import DockerSandbox

                if not DockerSandbox().available():
                    raise SystemExit(
                        "code verification needs Docker; start it, pass "
                        "--allow-unsandboxed (NON-SANDBOXED local fallback) or "
                        "omit --verify-code (math-only traces)"
                    )
        heartbeat.set_phase("model_load")
        model, processor, _plan = load_model(
            ModelSpec(size=MODEL_SIZE, quant=MODEL_QUANT)
        )
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            local.MODEL_ID, revision=MODEL_REVISION
        )

        run_id = local.compute_run_id(
            str(pool_manifest.get("manifest_sha256")),
            shard_size=int(args.shard_size),
            samples_per_prompt=local.SAMPLES_PER_PROMPT,
            model_revision=MODEL_REVISION,
        )
        evaluator = make_evaluator(tokenizer)
        decode = lambda ids: tokenizer.decode(  # noqa: E731
            list(ids), skip_special_tokens=False
        )
        prompt_tokens_for = make_prompt_tokens_fn(tokenizer)

        def progress(payload: Mapping[str, Any]) -> None:
            delta = int(payload.get("generated_tokens_delta", 0))
            heartbeat.update(
                items_done=int(payload.get("records", 0)),
                items_total=sum(target.problems for target in targets)
                * local.SAMPLES_PER_PROMPT,
                tokens=max(0, delta),
                phase="sampling",
                extra={
                    "train_tokens": int(payload.get("train_tokens", 0)),
                    "train_goal": int(payload.get("train_goal", 0)),
                    "kept": int(payload.get("kept", 0)),
                    "acceptance_rate": float(
                        payload.get("kept", 0) / max(1, payload.get("samples_generated", 0))
                    ),
                    "shards_committed": payload.get("shards_committed", []),
                    "shards_skipped": payload.get("shards_skipped", []),
                    "run_id": run_id,
                },
            )

        heartbeat.set_phase("sampling")
        try:
            run_stats = local.run_sampling(
                problems=problems,
                targets=targets,
                output_dir=Path(args.output_dir),
                shard_size=int(args.shard_size),
                samples_per_prompt=local.SAMPLES_PER_PROMPT,
                max_length=MAX_LENGTH,
                prompt_tokens_for=prompt_tokens_for,
                evaluate_fn=evaluator,
                generate_fn=generate_batch,
                model=model,
                processor=processor,
                decode=decode,
                run_id=run_id,
                pool_manifest_sha256=str(pool_manifest.get("manifest_sha256")),
                model_revision=MODEL_REVISION,
                max_batch_override=args.batch,
                limit_shards=args.limit_shards,
                seed=int(args.seed),
                prereg_sha256=local.file_sha256(Path(args.prereg)),
                hidden_tests_sha256=(
                    (hidden_tests or {}).get("sha256")
                    if args.verify_code
                    else None
                ),
                progress=progress,
            )
        except local.RunIdentityError as exc:
            print(f"refusing to resume: {exc}", file=sys.stderr)
            return 2

        heartbeat.set_phase("finalize")
        grader = None
        try:
            if args.verify_code:
                if args.sandbox == "local" and not args.allow_unsandboxed:
                    raise SystemExit(
                        "refusing --sandbox local without --allow-unsandboxed"
                    )
                grader = GraderService(
                    workers=args.workers,
                    timeout=args.timeout,
                    sandbox=args.sandbox,
                    allow_fallback=args.allow_unsandboxed,
                )
            verify_fn = make_verify_fn(
                verify_code=args.verify_code,
                grader=grader,
                use_math_verify=use_math_verify,
            )

            def evaluate_full(record, problem):
                return local.evaluate_sample(
                    record,
                    problem,
                    tokenizer=tokenizer,
                    token_counter=lambda text: len(
                        tokenizer.encode(text or "", add_special_tokens=False)
                    ),
                    max_length=MAX_LENGTH,
                    min_reasoning_share=MIN_REASONING_SHARE,
                    verify_fn=verify_fn,
                )

            # The verification pass above only needs the math verifier; re-run
            # the evaluation once with verification enabled for the corpora.
            verifiers = {
                "math": "math-verify" if use_math_verify else "aime-normalise",
                "code": (
                    f"stage0-grader:{args.sandbox}+hidden-tests"
                    if args.verify_code
                    else "disabled"
                ),
            }
            try:
                stats, manifest = finalize(
                    args=args,
                    output_dir=Path(args.output_dir),
                    problems=problems,
                    targets=targets,
                    pool_dir=Path(args.pool_dir),
                    pool_manifest=pool_manifest,
                    problems_path=problems_path,
                    run_id=run_id,
                    model_revision=MODEL_REVISION,
                    run_stats=run_stats,
                    evaluate_filter=evaluator,
                    evaluate_full=evaluate_full,
                    verifiers=verifiers,
                    hidden_tests=hidden_tests,
                    seed=int(args.seed),
                )
            except local.RunIdentityError as exc:
                print(f"refusing to resume: {exc}", file=sys.stderr)
                return 2
        finally:
            if grader is not None:
                grader.close()

        heartbeat.set_phase("done")
        heartbeat.update(
            items_done=int(stats.records),
            extra={
                "train_tokens": stats.train_tokens,
                "train_goal": int(
                    next(
                        (
                            target.token_target
                            for target in targets
                            if target.name == "train"
                        ),
                        0,
                    )
                ),
                "manifest": str(Path(args.output_dir) / "manifest.json"),
            },
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "manifest": str(Path(args.output_dir) / "manifest.json"),
                    "records": stats.records,
                    "kept": stats.kept,
                    "acceptance_rate": stats.acceptance_rate,
                    "train_accepted_assistant_tokens": stats.train_tokens,
                    "shards_committed": stats.shards_committed,
                    "shards_skipped": stats.shards_skipped,
                    "shortfalls": manifest.get("shortfalls", {}),
                },
                indent=2,
            )
        )
        train_goal = next(
            (target.token_target for target in targets if target.name == "train"), 0
        )
        if train_goal and stats.train_tokens < train_goal and not args.allow_shortfall:
            print(
                f"train token target not reached: {stats.train_tokens} < {train_goal}",
                file=sys.stderr,
            )
            return 3
        return 0
    finally:
        heartbeat.stop()
        del heartbeat


if __name__ == "__main__":
    raise SystemExit(main())
