"""Build a completed run in the exact layout ``scripts/train.py`` writes.

The N1 regression came from test fixtures inventing a ``teacher_runtime`` key
in ``run_config``.  Production writes the runtime only at
``run.json["teacher"]["teacher"]["runtime"]``.  These helpers call train.py's
own metadata/config builders so the fixtures cannot drift from production.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _manifest_sha(cache: Path) -> str:
    return hashlib.sha256((cache / "teacher_manifest.json").read_bytes()).hexdigest()


def train_shaped_complete_run(
    directory: str | Path,
    *,
    arm,
    cache: str | Path,
    prereg: str | Path,
    teacher_key: str,
    runtime: str,
    monitor_rows: str | Path | None = None,
    monitor_teacher_data: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``run.json`` and ``run_identity.json`` as train.py would."""

    import scripts.train as train

    destination = Path(directory)
    cache_path = Path(cache)
    destination.mkdir(parents=True, exist_ok=True)
    teacher_index = SimpleNamespace(
        manifest=SimpleNamespace(
            teacher={"key": teacher_key, "runtime": runtime},
            data_manifest_sha256=None,
            top_k=64,
            rows_total=1,
            tokens_total=1,
            mask_token="<glitch_pruned_249788>",
        ),
        manifest_sha256=_manifest_sha(cache_path),
    )
    loop = SimpleNamespace(
        ple_mode="shared",
        bridge=arm.bridge,
        bridge_init=0.99,
        block=tuple(arm.block),
        adapter_policy="loop_only",
        lora_target_layers="loop_block",
    )
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=10),
        tokens_seen=3_000_000,
        args=SimpleNamespace(learning_rate=1e-4, optim="adamw_8bit"),
        loop_metrics=None,
    )
    metadata = train.build_run_metadata(
        arm="L" if arm.kind == "L" else "O-H",
        size="e4b",
        quant="bf16",
        data=Path("data"),
        base_tokens=3_000_000,
        target_tokens=3_000_000,
        max_length=8192,
        seed=0,
        loop=loop,
        adapters_chain=(),
        trainer=trainer,
        manifest_sha256="a" * 64,
        manifest_target_tokens=None,
        loop_block=tuple(arm.block),
    )
    metadata["teacher"] = train.build_teacher_metadata(teacher_index, cache_path)
    metadata["run_identity_hash"] = "sha-" + arm.arm_id
    metadata["run_config_sha256"] = "cfg-" + arm.arm_id
    train.write_run_json(destination, metadata)

    run_config = train.build_run_config(
        SimpleNamespace(
            arm="L" if arm.kind == "L" else "O-H",
            size="e4b",
            data=Path("data"),
            monitor_rows=monitor_rows,
            monitor_teacher_data=monitor_teacher_data,
            tokens=3_000_000,
            seed=0,
            lr=1e-4,
            bridge_lr=1e-4,
            loop_ple_lr=1e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_clip=1.0,
            warmup_ratio=0.05,
            optimizer="adamw_8bit",
            bridge=arm.bridge,
            bridge_init=0.99,
            block=f"{arm.block[0]}-{arm.block[1]}",
            adapter_policy="loop_only",
            lora_target_layers="loop_block",
            allow_repeat=False,
            init_adapter=None,
            prereg=str(prereg),
            dial_objective=True,
            dial_window=512,
            dial_beta=0.0,
            dial_lambda=0.0,
            loop_probs=None,
            heal_teacher=False,
            allow_no_teacher=False,
            teacher_data=cache_path,
            pilot_arm_id=arm.arm_id,
            smoke=False,
        ),
        quant="bf16",
        max_length=8192,
    )
    train.write_run_identity(
        destination,
        {
            "run_config": run_config,
            "run_identity_hash": "sha-" + arm.arm_id,
            "run_config_sha256": "cfg-" + arm.arm_id,
        },
    )
    return metadata


__all__ = ["train_shaped_complete_run"]
