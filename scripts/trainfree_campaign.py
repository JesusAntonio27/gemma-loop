"""Resumable training-free loop campaign (exploratory, pre-preregistration).

Two GPU stages, run one job at a time:

**Stage A — P1-lite forward sweep.**  ``scripts/pilot_probe.py`` on the verified
P1-lite corpus for blocks ``25-29``, ``16-18`` and ``15-29``, bridges ``none``
(R, pure repeat) and ``damped`` (R-damped, the existing damped Euler update), and
``N in {1,2,3,4}``.  Reports per-layer geometry, forward ``KL(N=1||N)``,
temperature-calibrated NLL (T fitted per config/N on half the traces and scored
on the other half), raw NLL and top-1.

**Stage B — accuracy.**  O (original, no loops) and R / R-damped at N=2 for
every block, on LiveCodeBench post-cut (182 items, 1 sample) and GPQA Diamond
(198 items, 1 sample), with the registered eval sampling.

**Stage B2 — extrapolation.**  N=3 and N=4 for R / R-damped on **one** block,
chosen by the descriptive rule below (it is not the prereg pilot rule):

    the block with the lowest temperature-calibrated NLL at N=2 in stage A,
    taking the minimum over the two bridges, ties broken by lower block start
    and then lower block end.

The plan file is JSON and is updated atomically after every status change.  Eval
partial files are left to :mod:`gemmaloop.eval.runner`, so an interrupted stage B
job resumes from its ``*.partial.jsonl`` instead of regenerating completed
items.  The campaign checks the GPU lock before starting, then releases it and
lets each child own the lock for its own lifetime; a crashed parent therefore
cannot leave an unlocked GPU child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

from gemmaloop.gpulock import release_held_lock, require_gpu_lock
from gemmaloop.heartbeat import Heartbeat
from gemmaloop.paths import artifacts_dir
from gemmaloop.train.perf import cap_cuda_allocator

BLOCKS: tuple[tuple[int, int], ...] = ((25, 29), (16, 18), (15, 29))
STAGE_A_BRIDGES: tuple[str, ...] = ("none", "damped")
STAGE_B_BRIDGES: tuple[tuple[str, str], ...] = (
    ("R", "none"),
    ("R-damped", "damped"),
)
STAGE_A_NS: tuple[int, ...] = (1, 2, 3, 4)
STAGE_B_N = 2
STAGE_B2_NS: tuple[int, ...] = (3, 4)
BENCHES: tuple[str, ...] = ("lcb", "gpqa")
BENCH_LIMITS = {"lcb": 182, "gpqa": 198}
EVAL_TAG = "trainfree-exploratory"
B2_RULE = (
    "block with the lowest temperature-calibrated NLL at N=2 in stage A, "
    "taking the minimum over bridges none/damped; ties by lower block start "
    "then lower block end"
)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def block_text(block: Sequence[int] | None) -> str:
    return "none" if block is None else f"{int(block[0])}-{int(block[1])}"


def block_slug(block: Sequence[int] | None) -> str:
    return "O" if block is None else f"b{int(block[0])}-{int(block[1])}"


def votes_label(bridge: str) -> str:
    return "R" if bridge == "none" else "R-damped"


def stage_a_configs(
    blocks: Sequence[tuple[int, int]] = BLOCKS,
    bridges: Sequence[str] = STAGE_A_BRIDGES,
    ns: Sequence[int] = STAGE_A_NS,
) -> list[dict[str, Any]]:
    """Every (block, bridge, N) entry, in execution order."""
    configs = []
    for n in ns:
        for block in blocks:
            for bridge in bridges:
                configs.append(
                    {
                        "id": f"A_{block_slug(block)}_{bridge}_n{n}",
                        "stage": "A",
                        "job": f"A_n{n}",
                        "block": [int(block[0]), int(block[1])],
                        "bridge": bridge,
                        "n": int(n),
                        "status": "pending",
                        "output": None,
                    }
                )
    return configs


def _eval_entry(
    stage: str,
    job: str,
    variant: str,
    block: Sequence[int] | None,
    bridge: str | None,
    n: int,
    bench: str,
) -> dict[str, Any]:
    return {
        "id": f"{stage}_{job}_{bench}",
        "stage": stage,
        "job": job,
        "variant": variant,
        "block": None if block is None else [int(block[0]), int(block[1])],
        "bridge": bridge,
        "n": int(n),
        "bench": bench,
        "status": "pending",
        "result": None,
    }


def stage_b_configs(
    blocks: Sequence[tuple[int, int]] = BLOCKS,
    bridges: Sequence[tuple[str, str]] = STAGE_B_BRIDGES,
    benches: Sequence[str] = BENCHES,
) -> list[dict[str, Any]]:
    """O plus R / R-damped at N=2 for every block, one entry per bench."""
    configs = []
    for bench in benches:
        configs.append(
            _eval_entry("B", "B_O", "O", None, None, 1, bench)
        )
    for block in blocks:
        for label, bridge in bridges:
            job = f"B_{block_slug(block)}_{label}"
            variant = f"{label}_{block_slug(block)}_n{STAGE_B_N}"
            for bench in benches:
                configs.append(
                    _eval_entry(
                        "B", job, variant, block, bridge, STAGE_B_N, bench
                    )
                )
    return configs


def stage_b2_configs(
    block: tuple[int, int],
    bridges: Sequence[tuple[str, str]] = STAGE_B_BRIDGES,
    ns: Sequence[int] = STAGE_B2_NS,
    benches: Sequence[str] = BENCHES,
) -> list[dict[str, Any]]:
    """N=3/4 for R / R-damped on the chosen block."""
    configs = []
    for n in ns:
        for label, bridge in bridges:
            job = f"B2_{block_slug(block)}_{label}_n{n}"
            variant = f"{label}_{block_slug(block)}_n{n}"
            for bench in benches:
                configs.append(
                    _eval_entry("B2", job, variant, block, bridge, n, bench)
                )
    return configs


def build_plan(paths: dict[str, str], *, stage_b2: bool = False, block=None):
    plan = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated": None,
        "paths": paths,
        "b2_rule": B2_RULE,
        "b2_block": None if block is None else [int(block[0]), int(block[1])],
        "stages": {"A": "pending", "B": "pending", "B2": "pending"},
        "configs": [*stage_a_configs(), *stage_b_configs()],
    }
    if stage_b2 and block is not None:
        plan["configs"].extend(stage_b2_configs(tuple(block)))
    return plan


def save_plan(plan: dict[str, Any], path: Path) -> None:
    """Write the plan atomically (temp + replace)."""
    plan["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_create_plan(
    path: Path, paths: dict[str, str], *, stage_b2: bool = False, block=None
) -> dict[str, Any]:
    if path.exists():
        plan = load_plan(path)
        plan.setdefault("paths", paths)
        plan.setdefault("stages", {"A": "pending", "B": "pending", "B2": "pending"})
        plan.setdefault("configs", [])
        plan.setdefault("b2_rule", B2_RULE)
        plan.setdefault("b2_block", None)
        return plan
    return build_plan(paths, stage_b2=stage_b2, block=block)


# ---------------------------------------------------------------------------
# B2 block rule
# ---------------------------------------------------------------------------
def choose_b2_block(payloads: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """Lowest calibrated NLL at N=2 (min over bridges), deterministic ties."""
    best: dict[tuple[int, int], float] = {}
    for payload in payloads:
        if isinstance(payload, list):
            variants = payload
        else:
            variants = payload.get("variants", [])
        for record in variants:
            if int(record.get("n_loops", -1)) != 2:
                continue
            if record.get("bridge") not in {"none", "damped"}:
                continue
            block = tuple(int(value) for value in record["block"])
            value = record.get("nll_n_calibrated", record.get("calibrated_nll"))
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            best[block] = min(best.get(block, math.inf), value)
    if not best:
        raise ValueError("stage A has no calibrated NLL at N=2")
    return min(best, key=lambda item: (best[item], item[0], item[1]))


def stage_a_payloads(stage_a_dir: Path) -> list[dict[str, Any]]:
    files = sorted(
        path for path in stage_a_dir.glob("n*.json") if path.is_file()
    )
    if not files:
        raise ValueError(f"no stage A result files under {stage_a_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


# ---------------------------------------------------------------------------
# Eval config derived from the registered prereg (samples -> 1)
# ---------------------------------------------------------------------------
def ensure_eval_config(prereg_path: Path, out_path: Path) -> Path:
    """Write a derived run config: registered sampling, 1 sample per bench."""
    raw = yaml.safe_load(prereg_path.read_text(encoding="utf-8")) or {}
    sampling = raw.get("sampling")
    benches = raw.get("benches")
    if not isinstance(sampling, dict) or not isinstance(benches, dict):
        raise ValueError(f"{prereg_path} needs sampling and benches mappings")
    missing = [bench for bench in BENCHES if bench not in benches]
    if missing:
        raise ValueError(f"{prereg_path} is missing benches: {missing}")

    derived_benches = {}
    for bench in BENCHES:
        source = dict(benches[bench])
        source["limit"] = BENCH_LIMITS[bench]
        source["samples"] = 1
        derived_benches[bench] = source
    derived = {
        "tag": EVAL_TAG,
        "sampling": dict(sampling),
        "benches": derived_benches,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
Runner = Callable[[Sequence[str], dict[str, str]], int]


def _default_runner(command: Sequence[str], env: dict[str, str]) -> int:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(list(command), env=env, check=False).returncode


def _job_configs(plan: dict[str, Any], stage: str, job: str) -> list[dict[str, Any]]:
    return [
        config
        for config in plan["configs"]
        if config["stage"] == stage and config["job"] == job
    ]


def jobs_for_stage(plan: dict[str, Any], stage: str) -> list[str]:
    jobs: list[str] = []
    for config in plan["configs"]:
        if config["stage"] == stage and config["job"] not in jobs:
            jobs.append(config["job"])
    return jobs


def _mark(entries: Sequence[dict[str, Any]], status: str, **fields: Any) -> None:
    for entry in entries:
        entry["status"] = status
        entry.update(fields)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _child_tokens(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("generated_tokens", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _stage_a_matches(
    path: Path,
    entries: Sequence[dict[str, Any]],
    corpus_sha256: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if corpus_sha256 is not None and payload.get("corpus_sha256") != corpus_sha256:
        return False
    found = {
        (
            tuple(int(value) for value in record["block"]),
            str(record["bridge"]),
            int(record["n_loops"]),
        )
        for record in payload.get("variants", [])
    }
    wanted = {
        (tuple(entry["block"]), str(entry["bridge"]), int(entry["n"]))
        for entry in entries
    }
    return wanted <= found


def run_stage_a(
    plan: dict[str, Any],
    *,
    stage_dir: Path,
    corpus: Path,
    runner: Runner,
    env: dict[str, str],
    heartbeat: Heartbeat | None = None,
    heartbeat_dir: Path | None = None,
    plan_path: Path | None = None,
    corpus_sha256: str | None = None,
) -> None:
    """Run stage A jobs, skipping any N whose output already matches."""
    plan["stages"]["A"] = "running"
    jobs = jobs_for_stage(plan, "A")
    for index, job in enumerate(jobs, start=1):
        entries = _job_configs(plan, "A", job)
        n = int(entries[0]["n"])
        out_json = stage_dir / f"n{n}.json"
        out_md = stage_dir / f"n{n}.md"
        child_heartbeat = (
            heartbeat_dir / f"child-{job}.json"
            if heartbeat_dir is not None
            else None
        )
        if _stage_a_matches(out_json, entries, corpus_sha256):
            _mark(entries, "done", output=str(out_json))
            if plan_path is not None:
                save_plan(plan, plan_path)
            continue
        _mark(entries, "running")
        if heartbeat is not None:
            heartbeat.update(
                step=index - 1,
                items_total=len(jobs),
                phase=f"stage A N={n}",
                extra={"job": job},
            )
        command = [
            sys.executable,
            "-m",
            "scripts.pilot_probe",
            "--rows",
            str(corpus),
            "--blocks",
            ",".join(block_text(block) for block in BLOCKS),
            "--bridges",
            ",".join(STAGE_A_BRIDGES),
            "--n-loops",
            str(n),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
        if child_heartbeat is not None:
            command += ["--heartbeat", str(child_heartbeat)]
        tokens_before = _child_tokens(child_heartbeat)
        if plan_path is not None:
            save_plan(plan, plan_path)
        code = runner(command, env)
        tokens_after = _child_tokens(child_heartbeat)
        if code == 0 and _stage_a_matches(out_json, entries, corpus_sha256):
            _mark(entries, "done", output=str(out_json))
        else:
            _mark(entries, "failed", error=f"exit {code}")
        if heartbeat is not None:
            heartbeat.update(
                step=index,
                items_total=len(jobs),
                tokens=max(0, tokens_after - tokens_before),
                phase=f"stage A N={n}",
                extra={"job": job, "last_exit": code},
            )
        if plan_path is not None:
            save_plan(plan, plan_path)
    plan["stages"]["A"] = (
        "done"
        if all(entry["status"] == "done" for entry in plan["configs"] if entry["stage"] == "A")
        else "failed"
    )
    if plan_path is not None:
        save_plan(plan, plan_path)


def eval_command(
    entry: dict[str, Any],
    *,
    eval_config: Path,
    out_dir: Path,
    size: str,
    quant: str,
    heartbeat_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.run_eval",
        "--variant",
        str(entry["variant"]),
        "--size",
        size,
        "--quant",
        quant,
        "--bench",
        ",".join(BENCHES),
        "--prereg",
        str(eval_config),
        "--out-dir",
        str(out_dir),
    ]
    block = entry.get("block")
    if block is not None:
        command += [
            "--loops",
            str(entry["n"]),
            "--bridge",
            str(entry["bridge"]),
            "--block",
            block_text(block),
        ]
    if heartbeat_path is not None:
        command += ["--heartbeat", str(heartbeat_path)]
    return command


def _result_matches(entry: dict[str, Any], run: Any) -> bool:
    if str(run.bench) != str(entry["bench"]):
        return False
    expected_n = 0 if entry["variant"] == "O" else int(entry["n"])
    if int(run.n_loops) != expected_n:
        return False
    if str(run.variant) != str(entry["variant"]):
        return False
    block = entry.get("block")
    if block is not None:
        if run.loop_block is None or tuple(run.loop_block) != tuple(block):
            return False
    bridge = entry.get("bridge")
    if bridge is not None and run.bridge != bridge:
        return False
    return True


def _scan_results(job_dir: Path, entries: Sequence[dict[str, Any]]) -> None:
    from gemmaloop.eval.results import RunResult

    found: dict[str, str] = {}
    for path in sorted(job_dir.rglob("*.json")):
        try:
            run = RunResult.load(path)
        except Exception:
            continue
        for entry in entries:
            if _result_matches(entry, run):
                found[entry["id"]] = str(path)
    for entry in entries:
        if entry["id"] in found:
            entry["status"] = "done"
            entry["result"] = found[entry["id"]]
        elif entry["status"] == "running":
            entry["status"] = "failed"


def run_eval_stage(
    plan: dict[str, Any],
    stage: str,
    *,
    results_root: Path,
    eval_config: Path,
    size: str,
    quant: str,
    runner: Runner,
    env: dict[str, str],
    heartbeat: Heartbeat | None = None,
    heartbeat_dir: Path | None = None,
    plan_path: Path | None = None,
) -> None:
    """Run a stage B/B2 job per model load, resuming from eval partials."""
    plan["stages"][stage] = "running"
    jobs = jobs_for_stage(plan, stage)
    for index, job in enumerate(jobs, start=1):
        entries = _job_configs(plan, stage, job)
        if all(
            entry["status"] == "done"
            and entry.get("result")
            and Path(entry["result"]).exists()
            for entry in entries
        ):
            if plan_path is not None:
                save_plan(plan, plan_path)
            continue
        job_dir = results_root / stage / job
        _mark(entries, "running")
        if heartbeat is not None:
            heartbeat.update(
                step=index - 1,
                items_total=len(jobs),
                phase=f"stage {stage} {job}",
                extra={"job": job},
            )
        heartbeat_path = (
            heartbeat_dir / f"child-{job}.json"
            if heartbeat_dir is not None
            else None
        )
        tokens_before = _child_tokens(heartbeat_path)
        if plan_path is not None:
            save_plan(plan, plan_path)
        command = eval_command(
            entries[0],
            eval_config=eval_config,
            out_dir=job_dir,
            size=size,
            quant=quant,
            heartbeat_path=heartbeat_path,
        )
        code = runner(command, env)
        tokens_after = _child_tokens(heartbeat_path)
        _scan_results(job_dir, entries)
        if code != 0:
            for entry in entries:
                if entry["status"] != "done":
                    entry["status"] = "failed"
                    entry["error"] = f"exit {code}"
        if heartbeat is not None:
            heartbeat.update(
                step=index,
                items_total=len(jobs),
                tokens=max(0, tokens_after - tokens_before),
                phase=f"stage {stage} {job}",
                extra={"job": job, "last_exit": code},
            )
        if plan_path is not None:
            save_plan(plan, plan_path)
    stage_entries = [entry for entry in plan["configs"] if entry["stage"] == stage]
    plan["stages"][stage] = (
        "done" if stage_entries and all(e["status"] == "done" for e in stage_entries) else "failed"
    )
    if plan_path is not None:
        save_plan(plan, plan_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def default_paths() -> dict[str, str]:
    root = artifacts_dir()
    results = root / "results" / "trainfree"
    return {
        "plan": str(root / "runs" / "trainfree" / "plan.json"),
        "corpus": str(root / "data" / "p1lite" / "corpus.jsonl"),
        "results_root": str(results),
        "stage_a": str(results / "stageA"),
        "eval_config": str(root / "runs" / "trainfree" / "eval_config.yaml"),
        "heartbeat": str(root / "runs" / "detached" / "campaign.json"),
        "heartbeat_dir": str(root / "runs" / "detached"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("A", "B", "B2", "all"), default="all"
    )
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--prereg", type=Path, default=Path("prereg/prereg.yaml"))
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--stage-a-dir", type=Path, default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--size", choices=("e2b", "e4b"), default="e2b")
    parser.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the plan (and choose B2 if stage A exists) without a GPU",
    )
    parser.add_argument(
        "--allow-stage-b",
        action="store_true",
        help=(
            "required for --stage B/B2/all: accuracy stages are blocked until "
            "the author decides E3"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = default_paths()
    plan_path = args.plan or Path(defaults["plan"])
    corpus = args.corpus or Path(defaults["corpus"])
    results_root = args.results_root or Path(defaults["results_root"])
    stage_a_dir = args.stage_a_dir or Path(defaults["stage_a"])
    eval_config = args.eval_config or Path(defaults["eval_config"])
    heartbeat_path = args.heartbeat or Path(defaults["heartbeat"])
    heartbeat_dir = heartbeat_path.parent
    paths = {
        **defaults,
        "plan": str(plan_path),
        "corpus": str(corpus),
        "results_root": str(results_root),
        "stage_a": str(stage_a_dir),
        "eval_config": str(eval_config),
        "heartbeat": str(heartbeat_path),
        "heartbeat_dir": str(heartbeat_dir),
    }

    if not args.plan_only and args.stage in ("B", "B2", "all") and not args.allow_stage_b:
        raise SystemExit(
            "stage B/B2 is BLOCKED until the author decides E3; "
            "pass --allow-stage-b only after that decision"
        )

    plan = load_or_create_plan(plan_path, paths)
    wants_b2 = args.stage in ("B2", "all")

    def choose_and_add_b2(*, required: bool) -> None:
        if plan.get("b2_block") is None:
            try:
                block = choose_b2_block(stage_a_payloads(stage_a_dir))
            except ValueError:
                if required:
                    raise
                return
            plan["b2_block"] = [int(block[0]), int(block[1])]
        if not any(config["stage"] == "B2" for config in plan["configs"]):
            plan["configs"].extend(stage_b2_configs(tuple(plan["b2_block"])))

    if args.stage == "B2":
        choose_and_add_b2(required=True)
    elif args.plan_only and wants_b2:
        choose_and_add_b2(required=False)
    save_plan(plan, plan_path)

    if args.plan_only:
        print(f"plan: {plan_path}")
        if plan.get("b2_block") is not None:
            print("b2 block: " + block_text(plan["b2_block"]))
        for stage in ("A", "B", "B2"):
            entries = [c for c in plan["configs"] if c["stage"] == stage]
            if entries:
                done = sum(entry["status"] == "done" for entry in entries)
                print(f"stage {stage}: {done}/{len(entries)} done")
        return 0

    require_gpu_lock()
    vram_cap = cap_cuda_allocator(0.90, 0.5)
    plan["vram_cap"] = vram_cap
    if corpus.exists():
        plan["corpus_sha256"] = _file_sha256(corpus)
    heartbeat = Heartbeat(heartbeat_path, run="trainfree-campaign")
    heartbeat.start(30.0)
    heartbeat.update(extra={"vram_cap": vram_cap})
    # The child owns the lock while it runs.  Releasing the parent lock here is
    # what makes a parent crash unable to leave an unlocked GPU child.
    release_held_lock()
    env = os.environ.copy()
    if args.stage in ("B", "B2", "all"):
        ensure_eval_config(Path(args.prereg), eval_config)

    if args.stage in ("A", "all"):
        run_stage_a(
            plan,
            stage_dir=stage_a_dir,
            corpus=corpus,
            runner=_default_runner,
            env=env,
            heartbeat=heartbeat,
            heartbeat_dir=heartbeat_dir,
            plan_path=plan_path,
            corpus_sha256=plan.get("corpus_sha256"),
        )
        save_plan(plan, plan_path)
    if wants_b2 and args.stage == "all":
        choose_and_add_b2(required=True)
        save_plan(plan, plan_path)
    if args.stage in ("B", "all"):
        run_eval_stage(
            plan,
            "B",
            results_root=results_root,
            eval_config=eval_config,
            size=args.size,
            quant=args.quant,
            runner=_default_runner,
            env=env,
            heartbeat=heartbeat,
            heartbeat_dir=heartbeat_dir,
            plan_path=plan_path,
        )
        save_plan(plan, plan_path)
    if wants_b2:
        run_eval_stage(
            plan,
            "B2",
            results_root=results_root,
            eval_config=eval_config,
            size=args.size,
            quant=args.quant,
            runner=_default_runner,
            env=env,
            heartbeat=heartbeat,
            heartbeat_dir=heartbeat_dir,
            plan_path=plan_path,
        )
        save_plan(plan, plan_path)

    heartbeat.stop()
    print(f"plan: {plan_path}")
    for stage in ("A", "B", "B2"):
        print(f"stage {stage}: {plan['stages'][stage]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
