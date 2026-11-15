"""Produce and freeze the registered D1 corpus (post-tag, pre-pilot).

D1 is O's dedicated sampling of the math problems in NLL/KL buckets 60-61
(temperature fit) and 62-69 (scored test): four samples per problem, the
registered Stage 0 sampling and filter, math-verify, and the lowest-index
verified-correct trace per problem.  The registered Stage 0 sampler is the
Modal stock-vLLM job (``scripts/modal_stage0.py --mode sample``) when approved,
otherwise the local padded HF engine (``gemmaloop.stage0.local``).

The existing sampler has no bucket/kind selector, so this additive CLI:

1. ``prepare-pool`` writes a frozen, Modal-compatible pool containing only the
   D1 math problems (same decontamination block, recomputed hashes).  The bound
   Modal script then samples exactly that corpus with no modification.
2. ``sample-modal`` / ``sample-local`` run the registered sampler, reusing its
   ledger, spend lock, budget caps, request identities and filters.
3. ``verify`` applies the registered Stage 0 filter plus math-verify, selects
   the lowest-index verified-correct trace per problem, and writes the traces,
   every per-sample verifier outcome and the data-addendum JSON.
4. ``freeze`` re-checks every hash and writes the frozen addendum.

Typical post-tag sequence (from the repo root)::

    uv run python scripts/sample_d1.py prepare-pool \
        --pool-dir C:/Users/<user>/ML/gemmaloop-artifacts/stage0/pool

    # Modal (approved/available), dry run first, then the real run
    uv run --with modal==1.6.3 python scripts/sample_d1.py sample-modal \
        --pool-dir C:/Users/<user>/ML/gemmaloop-artifacts/stage0/pool \
        --output-dir artifacts/stage0/d1/modal --max-usd 12 --dry-run
    uv run --with modal==1.6.3 python scripts/sample_d1.py sample-modal ... --yes

    # Local padded engine instead (one GPU job at a time, detached)
    uv run python scripts/sample_d1.py sample-local \
        --pool-dir C:/Users/<user>/ML/gemmaloop-artifacts/stage0/pool \
        --output-dir artifacts/stage0/d1/local

    uv run --offline python scripts/sample_d1.py verify \
        --pool-dir C:/Users/<user>/ML/gemmaloop-artifacts/stage0/pool \
        --raw-dir artifacts/stage0/d1/modal \
        --output-dir artifacts/stage0/d1/corpus
    uv run --offline python scripts/sample_d1.py freeze \
        --addendum artifacts/stage0/d1/corpus/d1_data_addendum.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemmaloop.stage0 import d1, local  # noqa: E402
from gemmaloop.stage0.postprocess import MAX_LENGTH, MIN_REASONING_SHARE  # noqa: E402

DEFAULT_POOL_DIR = Path("artifacts/stage0/pool")
DEFAULT_D1_POOL_DIR = Path("artifacts/stage0/d1_pool")
DEFAULT_MODAL_OUT = Path("artifacts/stage0/d1/modal")
DEFAULT_LOCAL_OUT = Path("artifacts/stage0/d1/local")
DEFAULT_CORPUS_OUT = Path("artifacts/stage0/d1/corpus")
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_SIZE = "e2b"
MODEL_QUANT = "bf16"
MATH_VERIFY_PIN = (
    "math-verify 0.9.0 (antlr4 4.13.2); registered OR rule with the "
    "AIME/integer normalisation; decided_by recorded per sample"
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def _add_pool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--d1-pool-dir", type=Path, default=DEFAULT_D1_POOL_DIR)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-pool", help="write the D1-only frozen pool")
    _add_pool_args(prepare)

    modal = sub.add_parser("sample-modal", help="run the registered Modal sampler")
    _add_pool_args(modal)
    modal.add_argument("--output-dir", type=Path, default=DEFAULT_MODAL_OUT)
    modal.add_argument("--gpu", default="A100-80GB")
    modal.add_argument("--max-usd", type=float, default=12.0)
    modal.add_argument("--shard-size", type=int, default=25)
    modal.add_argument("--avg-generated-tokens", type=int, default=32768)
    modal.add_argument("--seed", type=int, default=0)
    modal.add_argument("--timeout-seconds", type=int, default=None)
    modal.add_argument("--ledger", type=Path, default=None)
    modal.add_argument("--spend-lock", type=Path, default=None)
    modal.add_argument("--init-ledger", action="store_true")
    modal.add_argument("--rebuild-pool", action="store_true")
    modal.add_argument("--dry-run", action="store_true")
    modal.add_argument("--yes", action="store_true")

    local_parser = sub.add_parser(
        "sample-local", help="run the registered local padded HF sampler"
    )
    _add_pool_args(local_parser)
    local_parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOCAL_OUT)
    local_parser.add_argument("--prereg", type=Path, default=None)
    local_parser.add_argument("--shard-size", type=int, default=local.DEFAULT_SHARD_SIZE)
    local_parser.add_argument("--seed", type=int, default=0)
    local_parser.add_argument("--batch", type=int, default=None)
    local_parser.add_argument("--vram-fraction", type=float, default=0.90)
    local_parser.add_argument("--vram-margin-gib", type=float, default=0.5)
    local_parser.add_argument("--heartbeat", type=Path, default=None)
    local_parser.add_argument("--dry-run", action="store_true")

    verify = sub.add_parser("verify", help="filter, verify and select D1 traces")
    verify.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    verify.add_argument("--raw-dir", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_CORPUS_OUT)
    verify.add_argument("--tokenizer", default=MODEL_ID)
    verify.add_argument("--max-length", type=int, default=MAX_LENGTH)
    verify.add_argument("--reasoning-share", type=float, default=MIN_REASONING_SHARE)
    verify.add_argument("--allow-math-fallback", action="store_true")

    freeze = sub.add_parser("freeze", help="re-check and freeze the addendum")
    freeze.add_argument("--addendum", type=Path, required=True)
    freeze.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Derived D1 pool
# ---------------------------------------------------------------------------
def prepare_pool(pool_dir: Path, d1_pool_dir: Path) -> dict[str, Any]:
    """Write the derived D1 pool and print its counts."""

    selected, manifest = d1.write_d1_pool(pool_dir, d1_pool_dir)
    partition = d1.partition_d1_problems(selected)
    payload = {
        "source_pool": str(pool_dir),
        "d1_pool": str(d1_pool_dir),
        "d1_pool_manifest_sha256": manifest["manifest_sha256"],
        "derived_from": manifest["derived_from"],
        "problem_counts": partition.counts,
        "requests": {
            "fit": len(partition.fit) * d1.D1_SAMPLES_PER_PROBLEM,
            "test": len(partition.test) * d1.D1_SAMPLES_PER_PROBLEM,
            "total": len(partition.problems) * d1.D1_SAMPLES_PER_PROBLEM,
        },
    }
    return payload


def ensure_d1_pool(
    pool_dir: Path,
    d1_pool_dir: Path,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Return a valid derived D1 pool manifest, writing it when necessary."""

    manifest_path = Path(d1_pool_dir) / "manifest.json"
    problems_path = Path(d1_pool_dir) / "problems.jsonl"
    source_manifest = json.loads(
        (Path(pool_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    if not rebuild and manifest_path.exists() and problems_path.exists():
        problems, manifest = local.load_pool(Path(d1_pool_dir))
        derived = manifest.get("derived_from") or {}
        if derived.get("manifest_sha256") != source_manifest.get("manifest_sha256"):
            raise SystemExit(
                "the derived D1 pool was built from a different source pool; "
                "pass --rebuild-pool or use a fresh --d1-pool-dir"
            )
        return manifest
    prepare_pool(Path(pool_dir), Path(d1_pool_dir))
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Modal sampler (bound script, unchanged)
# ---------------------------------------------------------------------------
def _load_modal_script():
    """Import the bound Modal Stage 0 script without executing its ``main``."""

    import importlib.util  # noqa: PLC0415

    path = REPO_ROOT / "scripts" / "modal_stage0.py"
    spec = importlib.util.spec_from_file_location("gemmaloop_d1_modal_stage0", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modal_argv(args: argparse.Namespace, manifest_sha256: str) -> list[str]:
    """The exact ``scripts/modal_stage0.py`` arguments for the D1 corpus."""

    argv = [
        "--mode",
        "plan" if args.dry_run else "sample",
        "--pool-dir",
        str(args.d1_pool_dir),
        "--output-dir",
        str(args.output_dir),
        "--gpu",
        str(args.gpu),
        "--max-usd",
        str(args.max_usd),
        "--shard-size",
        str(args.shard_size),
        "--samples-per-prompt",
        str(d1.D1_SAMPLES_PER_PROBLEM),
        "--avg-generated-tokens",
        str(args.avg_generated_tokens),
        "--seed",
        str(args.seed),
        "--frozen-manifest-sha",
        str(manifest_sha256),
    ]
    if args.timeout_seconds is not None:
        argv += ["--timeout-seconds", str(args.timeout_seconds)]
    if args.ledger is not None:
        argv += ["--ledger", str(args.ledger)]
    if args.spend_lock is not None:
        argv += ["--spend-lock", str(args.spend_lock)]
    if args.init_ledger:
        argv.append("--init-ledger")
    if args.dry_run:
        argv.append("--dry-run")
    elif args.yes:
        argv.append("--yes")
    return argv


def run_modal_sampling(
    args: argparse.Namespace,
    *,
    modal_script: Any = None,
) -> int:
    """Ensure the D1 pool and hand the run to the registered Modal script."""

    manifest = ensure_d1_pool(
        Path(args.pool_dir), Path(args.d1_pool_dir), rebuild=bool(args.rebuild_pool)
    )
    script = modal_script or _load_modal_script()
    return int(
        script.main(modal_argv(args, str(manifest["manifest_sha256"])))
    )


# ---------------------------------------------------------------------------
# Local sampler
# ---------------------------------------------------------------------------
def _prompt_tokens_fn(tokenizer: Any) -> Callable[[Mapping[str, Any]], int]:
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


def _filter_evaluator(
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


def sample_local_plan(
    pool_dir: Path,
    *,
    shard_size: int = local.DEFAULT_SHARD_SIZE,
    seed: int = 0,
) -> dict[str, Any]:
    """CPU-only description of the local run (also used by ``--dry-run``)."""

    problems, manifest = local.load_pool(Path(pool_dir))
    partition = d1.partition_d1_problems(problems)
    targets = d1.d1_targets(partition)
    plan = local.build_shard_plan(
        partition.problems,
        targets,
        shard_size=int(shard_size),
        samples_per_prompt=d1.D1_SAMPLES_PER_PROBLEM,
        seed=int(seed),
    )
    return {
        "engine": d1.D1_LOCAL_ENGINE,
        "pool_manifest_sha256": manifest.get("manifest_sha256"),
        "problem_counts": partition.counts,
        "targets": local.target_plan_payload(targets),
        "planned_shards": len(plan),
        "planned_requests": sum(len(shard.requests) for shard in plan),
        "sampling": dict(local.SAMPLING),
        "max_length": MAX_LENGTH,
    }


def run_local_sampling(args: argparse.Namespace) -> int:
    """Generate the D1 raw shards on the registered local padded engine."""

    problems, manifest = local.load_pool(Path(args.pool_dir))
    partition = d1.partition_d1_problems(problems)
    targets = d1.d1_targets(partition)
    if args.dry_run:
        print(
            json.dumps(
                sample_local_plan(
                    Path(args.pool_dir),
                    shard_size=int(args.shard_size),
                    seed=int(args.seed),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from gemmaloop.gpulock import require_gpu_lock  # noqa: PLC0415
    from gemmaloop.heartbeat import Heartbeat, default_heartbeat_path  # noqa: PLC0415

    require_gpu_lock()
    from gemmaloop.train.perf import cap_cuda_allocator  # noqa: PLC0415

    vram = cap_cuda_allocator(args.vram_fraction, args.vram_margin_gib)
    heartbeat = Heartbeat(
        args.heartbeat or default_heartbeat_path("stage0d1"),
        run="stage0-d1-local",
    )
    heartbeat.start(30.0)
    heartbeat.update(extra={"engine": d1.D1_LOCAL_ENGINE, "vram_cap": vram})
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        from gemmaloop.model.load import ModelSpec, load_model  # noqa: PLC0415
        from gemmaloop.model.snapshot import MODEL_REVISION  # noqa: PLC0415
        from gemmaloop.serve.generate import generate_batch  # noqa: PLC0415

        heartbeat.set_phase("model_load")
        model, processor, _plan = load_model(
            ModelSpec(size=MODEL_SIZE, quant=MODEL_QUANT)
        )
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        run_id = local.compute_run_id(
            str(manifest.get("manifest_sha256")),
            shard_size=int(args.shard_size),
            samples_per_prompt=d1.D1_SAMPLES_PER_PROBLEM,
            model_revision=MODEL_REVISION,
        )
        evaluator = _filter_evaluator(tokenizer)
        decode = lambda ids: tokenizer.decode(  # noqa: E731
            list(ids), skip_special_tokens=False
        )
        prereg = Path(args.prereg) if args.prereg is not None else None

        def progress(payload: Mapping[str, Any]) -> None:
            heartbeat.update(
                items_done=int(payload.get("records", 0)),
                items_total=len(partition.problems) * d1.D1_SAMPLES_PER_PROBLEM,
                tokens=max(0, int(payload.get("generated_tokens_delta", 0))),
                phase="sampling",
                extra={
                    "kept": int(payload.get("kept", 0)),
                    "samples_generated": int(payload.get("samples_generated", 0)),
                    "shards_committed": payload.get("shards_committed", []),
                    "run_id": run_id,
                },
            )

        heartbeat.set_phase("sampling")
        stats = local.run_sampling(
            problems=partition.problems,
            targets=targets,
            output_dir=Path(args.output_dir),
            shard_size=int(args.shard_size),
            samples_per_prompt=d1.D1_SAMPLES_PER_PROBLEM,
            max_length=MAX_LENGTH,
            prompt_tokens_for=_prompt_tokens_fn(tokenizer),
            evaluate_fn=evaluator,
            generate_fn=generate_batch,
            model=model,
            processor=processor,
            decode=decode,
            run_id=run_id,
            pool_manifest_sha256=str(manifest.get("manifest_sha256")),
            model_revision=MODEL_REVISION,
            max_batch_override=args.batch,
            seed=int(args.seed),
            prereg_sha256=(
                local.file_sha256(prereg) if prereg is not None and prereg.exists() else None
            ),
            progress=progress,
        )
        heartbeat.set_phase("done")
        print(
            json.dumps(
                {
                    "engine": d1.D1_LOCAL_ENGINE,
                    "output_dir": str(args.output_dir),
                    "records": stats.records,
                    "kept": stats.kept,
                    "generated_tokens": stats.generated_tokens,
                    "shards_committed": stats.shards_committed,
                    "verify_command": (
                        "uv run --offline python scripts/sample_d1.py verify "
                        f"--pool-dir {args.pool_dir} --raw-dir {args.output_dir} "
                        f"--output-dir {DEFAULT_CORPUS_OUT}"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        heartbeat.stop()
        del heartbeat


# ---------------------------------------------------------------------------
# Verification and freeze
# ---------------------------------------------------------------------------
def raw_record_paths(raw_dir: Path) -> list[Path]:
    """Raw shard paths in the flat Modal and the ``shards/`` local layouts."""

    raw_dir = Path(raw_dir)
    paths: list[Path] = []
    for pattern_dir in (raw_dir, raw_dir / "shards"):
        if pattern_dir.is_dir():
            paths.extend(sorted(pattern_dir.glob("shard_*.jsonl")))
    # A directory passed as a single file is accepted too.
    if raw_dir.is_file():
        paths.append(raw_dir)
    return paths


def read_raw_records(raw_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in raw_record_paths(raw_dir):
        records.extend(local.read_jsonl(path, tolerant=True))
    return records


def verify_corpus(
    *,
    pool_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    tokenizer: Any = None,
    tokenizer_id: str = MODEL_ID,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
    token_counter: Callable[[str], int] | None = None,
    max_length: int = MAX_LENGTH,
    min_reasoning_share: float = MIN_REASONING_SHARE,
    allow_math_fallback: bool = False,
) -> dict[str, Any]:
    """Verify the raw shards and write the frozen D1 corpus + addendum.

    ``renderer``/``token_counter`` are test and diagnostics hooks; production
    verification loads the pinned tokenizer and measures the rendered length
    with it.
    """

    from gemmaloop.model.snapshot import MODEL_REVISION  # noqa: PLC0415
    from gemmaloop.stage0.grader import MATH_VERIFY_AVAILABLE  # noqa: PLC0415

    use_math_verify = True
    if not MATH_VERIFY_AVAILABLE:
        if not allow_math_fallback:
            raise SystemExit(
                "math-verify is not installed (declared dependency); run `uv "
                "sync` or pass --allow-math-fallback for the registered "
                "AIME/integer normaliser only"
            )
        use_math_verify = False

    problems, manifest = local.load_pool(Path(pool_dir))
    partition = d1.partition_d1_problems(problems)
    records = read_raw_records(Path(raw_dir))
    if not records:
        raise SystemExit(f"no raw shard records under {raw_dir}")

    if tokenizer is None and renderer is None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision=MODEL_REVISION
        )
    if token_counter is None:
        if tokenizer is None:  # pragma: no cover - renderer carries the count
            raise ValueError("token_counter is required with a renderer")
        counter = tokenizer

        def token_counter(text: str) -> int:
            return len(counter.encode(text or "", add_special_tokens=False))

    verify_fn = d1.make_math_verify_fn(use_math_verify=use_math_verify)
    corpus = d1.build_corpus(
        records,
        partition,
        tokenizer=tokenizer,
        token_counter=token_counter,
        renderer=renderer,
        verify_fn=verify_fn,
        max_length=int(max_length),
        min_reasoning_share=float(min_reasoning_share),
    )
    addendum = d1.build_addendum(
        pool_dir=Path(pool_dir),
        pool_manifest=manifest,
        pool_problems_sha256=local.file_sha256(Path(pool_dir) / "problems.jsonl"),
        partition=partition,
        corpus=corpus,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        input_counts=d1.input_bucket_counts(problems),
        verifiers={
            "math": MATH_VERIFY_PIN if use_math_verify else "aime-normalise only (fallback)",
            "math_verify_available": bool(MATH_VERIFY_AVAILABLE),
            "code": "excluded (no hidden-test harness registered for this corpus)",
            "chat": "excluded",
        },
    )
    return d1.write_corpus(Path(output_dir), corpus, addendum)


def freeze_addendum(addendum_path: Path, output: Path | None = None) -> dict[str, Any]:
    """Verify the addendum and write the frozen copy with its final hash."""

    payload = d1.verify_frozen_addendum(Path(addendum_path))
    frozen = dict(payload)
    frozen["status"] = "frozen"
    frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
    frozen.pop("manifest_sha256", None)
    frozen["manifest_sha256"] = local.manifest_sha256(frozen)
    output_path = Path(output) if output is not None else Path(addendum_path).with_name(
        "d1_data_addendum.frozen.json"
    )
    local.write_json_atomic(output_path, frozen)
    frozen["addendum_path"] = str(output_path)
    return frozen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare-pool":
        print(json.dumps(prepare_pool(args.pool_dir, args.d1_pool_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "sample-modal":
        return run_modal_sampling(args)
    if args.command == "sample-local":
        return run_local_sampling(args)
    if args.command == "verify":
        payload = verify_corpus(
            pool_dir=Path(args.pool_dir),
            raw_dir=Path(args.raw_dir),
            output_dir=Path(args.output_dir),
            tokenizer_id=args.tokenizer,
            max_length=int(args.max_length),
            min_reasoning_share=float(args.reasoning_share),
            allow_math_fallback=bool(args.allow_math_fallback),
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "addendum": str(Path(args.output_dir) / "d1_data_addendum.json"),
                    "manifest_sha256": payload.get("manifest_sha256"),
                    "counts": payload.get("counts"),
                    "fit_trace_ids": len((payload.get("selection") or {}).get("fit_trace_ids") or []),
                    "test_trace_ids": len((payload.get("selection") or {}).get("test_trace_ids") or []),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "freeze":
        payload = freeze_addendum(Path(args.addendum), args.output)
        print(
            json.dumps(
                {
                    "frozen_addendum": payload["addendum_path"],
                    "manifest_sha256": payload["manifest_sha256"],
                    "status": payload["status"],
                    "counts": payload.get("counts"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
