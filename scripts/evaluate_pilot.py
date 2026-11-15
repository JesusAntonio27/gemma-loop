"""Evaluate one registered pilot arm as an orchestrated GPU job.

This is the evaluation child of ``scripts/run_pilot.py``.  It takes the GPU
lock itself, caps the CUDA allocator, writes one resumable JSON file, and
never tries to make a selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.gpulock import require_gpu_lock  # noqa: E402
from gemmaloop.pilot.arms import pilot_arms  # noqa: E402
from gemmaloop.pilot.evaluate import (  # noqa: E402
    KL_SUBSAMPLE_ROWS,
    evaluate_arm,
    load_eval_rows,
    read_training_summary,
)
from gemmaloop.prereg import default_prereg_path, load_prereg  # noqa: E402
from gemmaloop.train.perf import cap_cuda_allocator  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--heal-data", required=True, type=Path)
    parser.add_argument("--fit-data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--teacher-data", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def _rows_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "train.jsonl"
        if candidate.is_file():
            return candidate
    return path


def _load_heal_rows(path: Path, max_length: int):
    """Load heal rows, rendering ``messages`` with the pinned student template."""

    try:
        return load_eval_rows(_rows_path(path))
    except (OSError, ValueError):
        from scripts import precompute_teacher_topk as precompute

        from gemmaloop.pilot.evaluate import EvalRow

        raw = precompute.load_rows(path)
        tokenizer = precompute.load_student_tokenizer()
        prepared = precompute.prepare_rows(
            tokenizer, raw, int(max_length), max_rows=KL_SUBSAMPLE_ROWS
        )
        rows = []
        for item in prepared:
            first = next(
                (index for index, label in enumerate(item.labels) if label != -100),
                0,
            )
            rows.append(
                EvalRow(
                    row_id=int(item.row_id),
                    cluster=str(item.row_id),
                    input_ids=list(item.input_ids),
                    labels_start=int(first),
                )
            )
        return rows


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prereg = load_prereg(args.prereg or default_prereg_path())
    arms = pilot_arms(prereg)
    matches = [arm for arm in arms if arm.arm_id == args.arm_id]
    if not matches:
        print(f"error: unknown registered pilot arm {args.arm_id!r}", file=sys.stderr)
        return 2
    arm = matches[0]

    require_gpu_lock()
    cap = cap_cuda_allocator(0.90, 0.5)

    runtime = arm.runtime
    if args.teacher_data is not None:
        try:
            from gemmaloop.data.teacher_topk import TeacherManifest

            manifest = TeacherManifest.read(
                Path(args.teacher_data) / "teacher_manifest.json"
            )
            runtime = {
                **arm.runtime,
                "manifest": dict(manifest.teacher),
            }
        except Exception:
            pass
    run_payload: dict = {}
    try:
        loaded = json.loads(
            (Path(args.run_dir) / "run.json").read_text(encoding="utf-8")
        )
        if isinstance(loaded, dict):
            run_payload = loaded
    except (OSError, json.JSONDecodeError):
        run_payload = {}
    payload: dict = {
        "status": "failed",
        "arm_id": arm.arm_id,
        "arm": arm.to_dict(),
        "runtime": runtime,
        "pilot_arm": arm.arm_id,
        "run_identity_hash": run_payload.get("run_identity_hash"),
        "run_config_sha256": run_payload.get("run_config_sha256"),
        "data_manifest_sha256": run_payload.get("data_manifest_sha256"),
        "vram_cap": cap,
    }
    try:
        fit_rows = load_eval_rows(args.fit_data)
        test_rows = load_eval_rows(args.test_data)
        heal_error = None
        try:
            heal_rows = _load_heal_rows(
                args.heal_data, int(prereg.training.max_length)
            )
        except Exception as exc:
            heal_rows = None
            heal_error = f"{type(exc).__name__}: {exc}"
        evaluation = evaluate_arm(
            arm,
            args.run_dir,
            fit_rows=fit_rows,
            test_rows=test_rows,
            n_values=(1,) if arm.kind == "O-H" else (1, 2, 3, 4, 6, 8),
            teacher_dir=args.teacher_data,
            heal_rows=heal_rows,
        )
        evaluation["training"] = read_training_summary(args.run_dir)
        if heal_error is not None:
            evaluation["teacher_kl_error"] = heal_error
        payload.update({"status": "complete", "evaluation": evaluation})
        _write_atomic(args.out, payload)
        print(json.dumps({"status": "complete", "arm": arm.arm_id, "out": str(args.out)}))
        return 0
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        # Deterministic configuration/data errors are terminal (exit 2): the
        # orchestrator must not reload the model and retry five times.
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_atomic(args.out, payload)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # the orchestrator retries non-zero exits
        payload["error"] = str(exc)
        _write_atomic(args.out, payload)
        raise


if __name__ == "__main__":
    sys.exit(main())
