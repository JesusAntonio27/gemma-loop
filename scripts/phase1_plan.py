"""Build the Phase 1 E2B orchestration plan as JSON.

``scripts/phase1_e2b.ps1`` is deliberately thin: this module reads the
pre-registration, decides which steps are already complete, and prints the exact
commands (as an argv list plus a printable form) in execution order.  Keeping
the decisions here makes the Windows PowerShell 5.1 script small and lets the
plan be unit-tested without a GPU.

The plan has four kinds of steps:

* ``data`` -- build the decontaminated mixture with
  ``scripts/prepare_heal_data.py``.  The target is the largest token budget any
  Phase 1 E2B arm needs: ``train_tokens.e2b`` or the FLOPs-matched budget of
  ``O-H-ext`` computed by :func:`gemmaloop.train.budget.flops_matched_tokens`;
* ``train`` -- one ``scripts/train.py`` run per arm/seed in the brief, using the
  pre-registered token budget and seeds;
* ``eval`` -- one ``scripts/run_eval.py`` invocation per cell of the
  pre-registered matrix, followed by a ``charts`` step (``progress_charts.py``)
  and a ``table`` step (``gemmaloop.report.three_column``);
* ``criteria`` -- ``scripts/check_criteria.py`` for E2B, captured to
  ``artifacts/report/criteria_e2b.md``.

Every step carries ``skip`` (its output already exists) and, for the chart and
table steps, the id of the evaluation step that must force a refresh.  The
PowerShell driver simulates the same triggers during ``-DryRun``.

The E2B architecture is read from ``tests/fixtures/e2b_text_config.json``, the
same config used by the model tests, so the O-H-ext target matches what
``scripts/train.py`` computes from the loaded model config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gemmaloop.model.flops import per_layer_flops
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.paths import artifacts_dir
from gemmaloop.prereg import (
    Prereg,
    criteria_benches,
    harness_benches,
    load_prereg,
)
from gemmaloop.train.budget import flops_matched_tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = REPO_ROOT / "prereg" / "prereg.yaml"
DEFAULT_PREREG_TAG = "prereg-v1"
SIZES = ("e2b", "e4b")
QUANT = "bf16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slug(value: str) -> str:
    """Lower-case filesystem-safe form of an arm name."""
    return "".join(
        character if character.isalnum() else "-" for character in value.lower()
    ).strip("-")


def _quote(value: object) -> str:
    text = str(value)
    if text and not any(character in text for character in ' \t"'):
        return text
    return '"' + text.replace('"', '\\"') + '"'


def _display(argv: Sequence[object]) -> str:
    return " ".join(_quote(argument) for argument in argv)


def _read_prereg_tag(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return str(raw.get("tag", raw.get("prereg_tag", DEFAULT_PREREG_TAG)))


def _text_config(size: str) -> dict[str, Any]:
    """Return the canonical text config used for FLOPs planning."""
    path = REPO_ROOT / "tests" / "fixtures" / f"{size}_text_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"cannot compute FLOPs-matched budgets: missing {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _step(
    *,
    step_id: str,
    kind: str,
    title: str,
    argv: Sequence[object],
    skip: bool,
    skip_detail: str = "",
    outputs: Sequence[str] = (),
    trigger: str | None = None,
    force_if_any_eval: bool = False,
    output_file: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": step_id,
        "kind": kind,
        "title": title,
        "argv": [str(argument) for argument in argv],
        "display": _display(argv),
        "skip": bool(skip),
        "skip_detail": skip_detail,
        "outputs": [str(output) for output in outputs],
    }
    if trigger is not None:
        payload["trigger"] = trigger
    if force_if_any_eval:
        payload["force_if_any_eval"] = True
    if output_file is not None:
        payload["output_file"] = str(output_file)
    payload.update(extra)
    return payload


def _manifest_info(path: Path) -> dict[str, Any] | None:
    """Read the manifest fields the data step's skip check needs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    info: dict[str, Any] = {}
    for key in ("target_tokens", "max_length"):
        try:
            info[key] = int(payload[key])
        except (KeyError, TypeError, ValueError):
            info[key] = None
    mode = payload.get("length_mode")
    info["length_mode"] = str(mode) if isinstance(mode, str) and mode else None
    return info


def _manifest_target(path: Path) -> int | None:
    info = _manifest_info(path)
    return None if info is None else info.get("target_tokens")


def _train_complete(run_dir: Path) -> tuple[bool, str]:
    """Return whether ``run.json`` is the atomically written completion marker."""

    path = run_dir / "run.json"
    if not path.is_file():
        return False, f"no {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable {path.name}: {exc}"
    if not isinstance(payload, dict):
        return False, f"{path.name} is not an object"
    if payload.get("status") != "complete":
        return False, f"{path.name} status={payload.get('status')!r}"
    return True, f"{path.name} status=complete"


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-*`` directory, if any."""

    latest: tuple[int, Path] | None = None
    for child in run_dir.glob("checkpoint-*"):
        if not child.is_dir():
            continue
        suffix = child.name.split("-", 1)[1]
        if not suffix.isdigit():
            continue
        number = int(suffix)
        if latest is None or number > latest[0]:
            latest = (number, child)
    return latest[1] if latest is not None else None


def _loop_part(n_loops: int | None) -> str:
    return f"n{n_loops}" if n_loops else "n0"


def _bridge_part(n_loops: int | None, bridge: str | None) -> str:
    return bridge if n_loops and bridge else "base"


def _result_pattern(
    *,
    size: str,
    quant: str,
    n_loops: int | None,
    bridge: str | None,
    bench: str,
    tag: str,
) -> str:
    return (
        f"{size}_{quant}_{_loop_part(n_loops)}_"
        f"{_bridge_part(n_loops, bridge)}_{bench}_{tag}_*.json"
    )


def _result_matches(
    path: Path,
    *,
    bench: str,
    tag: str,
    n_loops: int | None,
    adapters: Sequence[str],
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("bench") != bench or payload.get("prereg_tag") != tag:
        return False
    expected_loops = int(n_loops or 0)
    try:
        if int(payload.get("n_loops", -1)) != expected_loops:
            return False
    except (TypeError, ValueError):
        return False
    fingerprint = payload.get("metadata", {}).get("fingerprint", {})
    if not isinstance(fingerprint, dict):
        return False
    found = [str(item) for item in fingerprint.get("adapters", [])]
    return found == [str(adapter) for adapter in adapters]


def _eval_complete(
    results_root: Path,
    *,
    variant: str,
    size: str,
    quant: str,
    n_loops: int | None,
    bridge: str | None,
    benches: Sequence[str],
    tag: str,
    adapters: Sequence[str],
) -> tuple[bool, str]:
    variant_dir = results_root / variant
    missing: list[str] = []
    for bench in benches:
        pattern = _result_pattern(
            size=size,
            quant=quant,
            n_loops=n_loops,
            bridge=bridge,
            bench=bench,
            tag=tag,
        )
        if not any(
            _result_matches(
                path,
                bench=bench,
                tag=tag,
                n_loops=n_loops,
                adapters=adapters,
            )
            for path in variant_dir.glob(pattern)
        ):
            missing.append(bench)
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, "all results present"


def _first_seed(prereg: Prereg, key: str) -> int:
    seeds = prereg.seeds.get(key)
    if not seeds:
        raise ValueError(f"prereg.seeds[{key!r}] is missing or empty")
    return int(seeds[0])


def verify_eval_step(plan: Mapping[str, Any], step_id: str) -> dict[str, Any]:
    """Re-check one evaluation step's expected RunResult files.

    The orchestrator runs this after ``run_eval.py`` exits 0 so a benchmark
    that was skipped or failed inside the runner can never leave the matrix
    silently incomplete.
    """

    step = next(
        (candidate for candidate in plan["steps"] if candidate["id"] == step_id),
        None,
    )
    if step is None:
        raise ValueError(f"unknown evaluation step {step_id!r}")
    if step["kind"] != "eval":
        raise ValueError(f"step {step_id!r} is not an evaluation step")

    complete, detail = _eval_complete(
        Path(plan["results"]),
        variant=step["variant"],
        size=plan["size"],
        quant=QUANT,
        n_loops=step["n_loops"],
        bridge=None if step["bridge"] == "base" else step["bridge"],
        benches=step["benches"],
        tag=plan["prereg_tag"],
        adapters=step["adapters"],
    )
    return {
        "ok": complete,
        "step_id": step_id,
        "variant": step["variant"],
        "n_loops": step["n_loops"],
        "benches": list(step["benches"]),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def registered_objective_args(
    prereg, arm: str, teacher_data: str | Path | None
) -> list[str]:
    """The registered Stage 1 objective flags for one training command.

    The plan must never launch a Stage 1 arm with the legacy CE objective:
    L/L-plevec get the dial objective and O-H/O-H-ext the single-exit heal,
    both with the precomputed teacher shards.  O is the control and gets none.
    """

    if arm in {"L", "L-plevec"}:
        if teacher_data is None:
            raise ValueError(f"arm {arm} needs --teacher-data for Stage 1")
        return ["--dial-objective", "--teacher-data", str(teacher_data)]
    if arm in {"O-H", "O-H-ext"}:
        if teacher_data is None:
            raise ValueError(f"arm {arm} needs --teacher-data for Stage 1")
        return ["--heal-teacher", "--teacher-data", str(teacher_data)]
    return []


def build_plan(
    prereg_path: str | Path,
    *,
    artifacts: str | Path | None = None,
    size: str = "e2b",
    text_config: Mapping[str, object] | None = None,
    tag: str | None = None,
    loop_block: str | tuple[int, int] | None = None,
    bridge: str | None = None,
    adapter_policy: str | None = None,
    lora_target_layers: str | None = None,
    teacher_data: str | Path | None = None,
) -> dict[str, Any]:
    """Return the ordered Phase 1 plan for ``size`` as plain JSON data."""
    raise RuntimeError(
        "phase1_plan is the retired single-SFT plan; prereg-v1 uses the "
        "two-stage hybrid and the E1-E4 analysis."
    )
    size = size.lower()
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")

    prereg_source = Path(prereg_path).resolve()
    prereg = load_prereg(prereg_source)
    training_max_length = int(prereg.training.max_length)
    training_length_mode = str(prereg.training.length_mode)
    artifacts_root = (
        Path(artifacts).expanduser().resolve()
        if artifacts is not None
        else artifacts_dir().resolve()
    )
    results_root = artifacts_root / "results"
    report_root = artifacts_root / "report"
    data_root = artifacts_root / "data" / "heal"
    runs_root = artifacts_root / "runs"
    run_root = runs_root / "phase1_e2b"
    chart_root = report_root / "progress"
    table_path = report_root / f"three_column_{size}.md"
    criteria_path = report_root / f"criteria_{size}.md"
    prereg_tag = tag if tag is not None else _read_prereg_tag(prereg_source)

    try:
        base_tokens = int(prereg.train_tokens[size])
    except KeyError as exc:
        raise ValueError(f"prereg.train_tokens[{size!r}] is missing") from exc
    if base_tokens <= 0:
        raise ValueError(f"prereg.train_tokens[{size!r}] must be positive")

    config = dict(text_config) if text_config is not None else _text_config(size)
    ext_tokens = flops_matched_tokens(
        base_tokens,
        per_layer_flops(config),
        make_layer_plan(config, block=loop_block),
        tuple(prereg.loop_probs),
    )
    # E[N] >= 1, so the FLOPs-matched budget is always at least the base one;
    # the ruling requires the exact flops_matched_tokens value here.
    required_data_tokens = ext_tokens

    oh_key = f"{size}_oh"
    l_key = f"{size}_l"
    oh_seeds = [int(seed) for seed in prereg.seeds.get(oh_key, [])]
    l_seeds = [int(seed) for seed in prereg.seeds.get(l_key, [])]
    if not oh_seeds:
        raise ValueError(f"prereg.seeds[{oh_key!r}] is missing or empty")
    if not l_seeds:
        raise ValueError(f"prereg.seeds[{l_key!r}] is missing or empty")

    full_benches = [str(bench) for bench in criteria_benches(prereg)]
    harness_only = [str(bench) for bench in harness_benches(prereg)]
    lcb_bench = str(prereg.criteria.s2_bench)
    if lcb_bench not in full_benches:
        raise ValueError(
            f"criteria.s2_bench={lcb_bench!r} is not a registered benchmark"
        )
    lcb_only = [lcb_bench]

    steps: list[dict[str, Any]] = []

    # (0) Data -----------------------------------------------------------------
    manifest_path = data_root / "manifest.json"
    train_jsonl = data_root / "train.jsonl"
    existing = _manifest_info(manifest_path)
    data_skip = (
        train_jsonl.is_file()
        and existing is not None
        and existing.get("target_tokens") == required_data_tokens
        and existing.get("max_length") == training_max_length
        and existing.get("length_mode") == training_length_mode
    )
    if data_skip:
        data_detail = (
            f"manifest target_tokens={existing['target_tokens']}, "
            f"max_length={existing['max_length']}, "
            f"length_mode={existing['length_mode']}"
        )
    elif not manifest_path.is_file():
        data_detail = "no manifest"
    elif not train_jsonl.is_file():
        data_detail = "manifest without train.jsonl"
    elif existing is None:
        data_detail = "unreadable manifest"
    else:
        data_detail = (
            f"manifest target_tokens={existing.get('target_tokens')}, "
            f"max_length={existing.get('max_length')}, "
            f"length_mode={existing.get('length_mode')} != "
            f"target_tokens={required_data_tokens}, "
            f"max_length={training_max_length}, "
            f"length_mode={training_length_mode}"
        )
    data_argv = [
        "uv",
        "run",
        "python",
        "scripts/prepare_heal_data.py",
        "--target-tokens",
        str(required_data_tokens),
        "--seed",
        "0",
        "--output-dir",
        str(data_root),
        "--max-length",
        str(training_max_length),
        "--length-mode",
        training_length_mode,
    ]
    steps.append(
        _step(
            step_id="data",
            kind="data",
            title=(
                "build healing mixture "
                f"({required_data_tokens} tokens, O-H-ext FLOPs-matched)"
            ),
            argv=data_argv,
            skip=data_skip,
            skip_detail=data_detail,
            outputs=[str(train_jsonl), str(manifest_path)],
            required_tokens=required_data_tokens,
            base_tokens=base_tokens,
            ext_tokens=ext_tokens,
        )
    )

    block_arg = (
        None
        if loop_block is None
        else (
            loop_block
            if isinstance(loop_block, str)
            else f"{loop_block[0]}-{loop_block[1]}"
        )
    )
    geometry_argv: list[str] = []
    if block_arg is not None:
        geometry_argv += ["--block", block_arg]
    if bridge is not None:
        geometry_argv += ["--bridge", bridge]
    if adapter_policy is not None:
        geometry_argv += ["--adapter-policy", adapter_policy]
    if lora_target_layers is not None:
        geometry_argv += ["--lora-target-layers", lora_target_layers]
    # `add_eval` already emits --bridge; the eval flags carry the rest.
    eval_geometry_argv = [
        item
        for pair in (
            ("--block", block_arg),
            ("--adapter-policy", adapter_policy),
            ("--lora-target-layers", lora_target_layers),
        )
        for item in (() if pair[1] is None else pair)
    ]

    # (1) Training ------------------------------------------------------------
    runs: dict[tuple[str, int], Path] = {}

    def add_train(arm: str, seed: int) -> None:
        run_dir = run_root / f"{_slug(arm)}-seed{seed}"
        runs[(arm, seed)] = run_dir
        completed, detail = _train_complete(run_dir)
        argv = [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            "--arm",
            arm,
            "--size",
            size,
            "--data",
            str(data_root),
            "--tokens",
            str(base_tokens),
            "--seed",
            str(seed),
            "--out",
            str(run_dir),
            "--prereg",
            str(prereg_source),
            "--max-length",
            str(training_max_length),
        ]
        argv += geometry_argv
        argv += registered_objective_args(prereg, arm, teacher_data)
        if arm == "O-H-ext":
            argv += ["--oh-ext-budget", str(ext_tokens)]
        checkpoint = None if completed else _latest_checkpoint(run_dir)
        if checkpoint is not None:
            argv += ["--resume", str(checkpoint)]
        steps.append(
            _step(
                step_id=f"train-{_slug(arm)}-s{seed}",
                kind="train",
                title=f"train {arm} seed {seed}",
                argv=argv,
                skip=completed,
                skip_detail=detail,
                outputs=[str(run_dir / "run.json")],
                arm=arm,
                seed=seed,
                run_dir=str(run_dir),
                resume_from=(
                    str(checkpoint) if checkpoint is not None else None
                ),
            )
        )

    for seed in oh_seeds:
        add_train("O-H", seed)
    for seed in l_seeds:
        add_train("L", seed)
    ext_seed = oh_seeds[0]
    plevec_seed = l_seeds[0]
    add_train("O-H-ext", ext_seed)
    add_train("L-plevec", plevec_seed)

    # (1b) Training audit -----------------------------------------------------
    audit_path = run_root / "training_audit.json"
    audit_argv = [
        "uv",
        "run",
        "python",
        "scripts/phase1_plan.py",
        "--audit",
        "--prereg",
        str(prereg_source),
        "--size",
        size,
        "--artifacts",
        str(artifacts_root),
    ]
    audit_argv += geometry_argv
    audit_exists = audit_path.is_file()
    steps.append(
        _step(
            step_id="audit-training",
            kind="audit",
            title=(
                "audit every training run (run.json, final step, consumed "
                "tokens, identical data manifest sha)"
            ),
            argv=audit_argv,
            skip=audit_exists,
            skip_detail=(
                "training audit already exists"
                if audit_exists
                else "no training audit"
            ),
            outputs=[str(audit_path)],
            force_if_any_train=True,
            output_file=str(audit_path),
        )
    )

    # (2) Evaluation matrix ---------------------------------------------------
    def add_eval(
        *,
        label: str,
        variant: str,
        n_loops: int | None,
        bridge: str | None,
        adapter: Path | None,
        benches: Sequence[str],
        seed: int,
    ) -> None:
        adapters = [str(adapter.resolve())] if adapter is not None else []
        complete, detail = _eval_complete(
            results_root,
            variant=variant,
            size=size,
            quant=QUANT,
            n_loops=n_loops,
            bridge=bridge,
            benches=benches,
            tag=prereg_tag,
            adapters=adapters,
        )
        eval_id = f"eval-{label}"
        argv = [
            "uv",
            "run",
            "python",
            "scripts/run_eval.py",
            "--variant",
            variant,
            "--size",
            size,
            "--quant",
            QUANT,
        ]
        if n_loops is not None:
            argv += ["--loops", str(n_loops)]
        if bridge is not None:
            argv += ["--bridge", bridge]
        argv += eval_geometry_argv
        if adapter is not None:
            argv += ["--adapter", str(adapter)]
        argv += [
            "--bench",
            ",".join(benches),
            "--prereg",
            str(prereg_source),
            "--out-dir",
            str(results_root),
        ]
        outputs = [
            str(
                results_root
                / variant
                / _result_pattern(
                    size=size,
                    quant=QUANT,
                    n_loops=n_loops,
                    bridge=bridge,
                    bench=bench,
                    tag=prereg_tag,
                )
            )
            for bench in benches
        ]
        steps.append(
            _step(
                step_id=eval_id,
                kind="eval",
                title=f"evaluate {label} ({', '.join(benches)})",
                argv=argv,
                skip=complete,
                skip_detail=detail,
                outputs=outputs,
                variant=variant,
                n_loops=int(n_loops or 0),
                bridge=bridge or "base",
                adapters=adapters,
                benches=list(benches),
                seed=seed,
            )
        )

        verify_argv = [
            "uv",
            "run",
            "python",
            "scripts/phase1_plan.py",
            "--verify-eval",
            eval_id,
            "--prereg",
            str(prereg_source),
            "--size",
            size,
            "--artifacts",
            str(artifacts_root),
        ]
        steps.append(
            _step(
                step_id=f"verify-{eval_id}",
                kind="verify",
                title=f"verify expected RunResult files after {label}",
                argv=verify_argv,
                skip=True,
                skip_detail="evaluation step was skipped",
                trigger=eval_id,
            )
        )

        chart_argv = [
            "uv",
            "run",
            "python",
            "scripts/progress_charts.py",
            "--results-dir",
            str(results_root),
            "--out",
            str(chart_root),
        ]
        steps.append(
            _step(
                step_id=f"charts-{eval_id}",
                kind="charts",
                title=f"regenerate progress charts after {label}",
                argv=chart_argv,
                skip=(chart_root / "summary.md").is_file(),
                skip_detail=(
                    "charts already exist"
                    if (chart_root / "summary.md").is_file()
                    else "no chart summary"
                ),
                outputs=[str(chart_root / "summary.md")],
                trigger=eval_id,
            )
        )

        table_argv = [
            "uv",
            "run",
            "python",
            "-m",
            "gemmaloop.report.three_column",
            "--size",
            size,
            "--results-dir",
            str(results_root),
            "--prereg",
            str(prereg_source),
            "--prereg-tag",
            prereg_tag,
            "--out",
            str(table_path),
        ]
        steps.append(
            _step(
                step_id=f"table-{eval_id}",
                kind="table",
                title=f"regenerate three-column table after {label}",
                argv=table_argv,
                skip=table_path.is_file(),
                skip_detail=(
                    "table already exists"
                    if table_path.is_file()
                    else "no table"
                ),
                outputs=[str(table_path)],
                trigger=eval_id,
            )
        )

    # Harness-validity benches (lcb_google) belong to the base model only:
    # their Google figure is a property of O, not of a trained arm.
    if harness_only:
        add_eval(
            label="o-harness",
            variant="O",
            n_loops=None,
            bridge=None,
            adapter=None,
            benches=harness_only,
            seed=0,
        )

    # Full suite: O-H (every seed), L at the primary N (every seed), L at N=1.
    for seed in oh_seeds:
        add_eval(
            label=f"oh-s{seed}",
            variant="O-H",
            n_loops=None,
            bridge=None,
            adapter=runs[("O-H", seed)],
            benches=full_benches,
            seed=seed,
        )
    for seed in l_seeds:
        add_eval(
            label=f"l-n{prereg.primary_n}-s{seed}",
            variant="L",
            n_loops=int(prereg.primary_n),
            bridge=(bridge or "scalar"),
            adapter=runs[("L", seed)],
            benches=full_benches,
            seed=seed,
        )
    if int(prereg.primary_n) != 1:
        add_eval(
            label=f"l-n1-s{plevec_seed}",
            variant="L",
            n_loops=1,
            bridge=(bridge or "scalar"),
            adapter=runs[("L", plevec_seed)],
            benches=full_benches,
            seed=plevec_seed,
        )

    # LCB only: L N=3,4; R and R-damped N=2,3,4; O-H-ext and L-plevec.
    for n_loops in [int(value) for value in prereg.criteria.exploratory_ns]:
        if n_loops == int(prereg.primary_n):
            continue
        add_eval(
            label=f"l-n{n_loops}-s{plevec_seed}",
            variant="L",
            n_loops=n_loops,
            bridge=(bridge or "scalar"),
            adapter=runs[("L", plevec_seed)],
            benches=lcb_only,
            seed=plevec_seed,
        )
    loop_values = sorted(
        {int(prereg.primary_n), *[int(value) for value in prereg.criteria.exploratory_ns]}
    )
    for n_loops in loop_values:
        add_eval(
            label=f"r-n{n_loops}",
            variant="R",
            n_loops=n_loops,
            bridge="none",
            adapter=None,
            benches=lcb_only,
            seed=0,
        )
    for n_loops in loop_values:
        add_eval(
            label=f"r-damped-n{n_loops}",
            variant="R-damped",
            n_loops=n_loops,
            bridge="damped",
            adapter=None,
            benches=lcb_only,
            seed=0,
        )
    add_eval(
        label=f"oh-ext-s{ext_seed}",
        variant="O-H-ext",
        n_loops=None,
        bridge=None,
        adapter=runs[("O-H-ext", ext_seed)],
        benches=lcb_only,
        seed=ext_seed,
    )
    add_eval(
        label=f"l-plevec-n{prereg.primary_n}-s{plevec_seed}",
        variant="L-plevec",
        n_loops=int(prereg.primary_n),
        bridge=(bridge or "scalar"),
        adapter=runs[("L-plevec", plevec_seed)],
        benches=lcb_only,
        seed=plevec_seed,
    )

    # (3) Criteria ------------------------------------------------------------
    criteria_argv = [
        "uv",
        "run",
        "python",
        "scripts/check_criteria.py",
        "--prereg",
        str(prereg_source),
        "--size",
        size,
        "--results",
        str(results_root),
    ]
    criteria_exists = criteria_path.is_file()
    steps.append(
        _step(
            step_id="criteria",
            kind="criteria",
            title=f"write S1-S5 verdicts to {criteria_path.name}",
            argv=criteria_argv,
            skip=criteria_exists,
            skip_detail=(
                "criteria report already exists"
                if criteria_exists
                else "no criteria report"
            ),
            outputs=[str(criteria_path)],
            force_if_any_eval=True,
            output_file=str(criteria_path),
        )
    )

    return {
        "format_version": 1,
        "size": size,
        "prereg": str(prereg_source),
        "prereg_tag": prereg_tag,
        "artifacts": str(artifacts_root),
        "results": str(results_root),
        "report": str(report_root),
        "required_data_tokens": required_data_tokens,
        "base_tokens": base_tokens,
        "ext_tokens": ext_tokens,
        "training_max_length": training_max_length,
        "training_length_mode": training_length_mode,
        "steps": steps,
    }


def audit_training(
    prereg_path: str | Path,
    *,
    artifacts: str | Path | None = None,
    size: str = "e2b",
    text_config: Mapping[str, object] | None = None,
    loop_block: str | tuple[int, int] | None = None,
    bridge: str | None = None,
    adapter_policy: str | None = None,
    lora_target_layers: str | None = None,
) -> dict[str, Any]:
    """Validate every training run before any evaluation starts.

    Checks, for every arm/seed in the plan: ``run.json`` exists, its
    ``global_step`` equals the final ``trainer_state.json`` step, consumed
    tokens are at least the arm's target budget, and every arm records the same
    data-manifest SHA-256.  Returns a JSON-ready summary; ``ok`` is false and
    ``problems`` lists every failure (the CLI exits 1 when any exist).
    """

    plan = build_plan(
        prereg_path,
        artifacts=artifacts,
        size=size,
        text_config=text_config,
        loop_block=loop_block,
        bridge=bridge,
        adapter_policy=adapter_policy,
        lora_target_layers=lora_target_layers,
    )
    train_steps = [step for step in plan["steps"] if step["kind"] == "train"]
    problems: list[str] = []
    arms: list[dict[str, Any]] = []
    by_sha: dict[str, list[str]] = {}

    for step in train_steps:
        run_dir = Path(step["run_dir"])
        label = f"{step['arm']} seed {step['seed']}"
        record: dict[str, Any] = {
            "arm": step["arm"],
            "seed": step["seed"],
            "run_dir": str(run_dir),
        }
        arms.append(record)

        run_json = run_dir / "run.json"
        if not run_json.is_file():
            problems.append(f"{label}: missing {run_json}")
            continue
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{label}: unreadable run.json ({exc})")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{label}: run.json is not an object")
            continue
        if payload.get("status") != "complete":
            problems.append(
                f"{label}: run.json status={payload.get('status')!r}, "
                "expected 'complete'"
            )

        expected_target = (
            int(plan["ext_tokens"])
            if step["arm"] == "O-H-ext"
            else int(plan["base_tokens"])
        )
        state_path = run_dir / "trainer_state.json"
        if not state_path.is_file():
            problems.append(f"{label}: missing {state_path}")
        else:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state_step = int(state.get("global_step", 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                problems.append(f"{label}: unreadable trainer_state.json ({exc})")
                state_step = None
            if state_step is not None:
                try:
                    run_step = int(payload.get("global_step", -1))
                except (TypeError, ValueError):
                    run_step = -1
                record["global_step"] = run_step
                if state_step <= 0:
                    problems.append(
                        f"{label}: final global_step is {state_step}, expected > 0"
                    )
                if run_step != state_step:
                    problems.append(
                        f"{label}: run.json global_step={run_step} != "
                        f"trainer_state global_step={state_step}"
                    )

        try:
            target = int(payload["target_tokens"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{label}: run.json has no integer target_tokens")
            target = None
        try:
            consumed = int(payload["tokens_consumed"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{label}: run.json has no integer tokens_consumed")
            consumed = None
        record["target_tokens"] = target
        record["expected_target_tokens"] = expected_target
        record["tokens_consumed"] = consumed
        if target is not None and target != expected_target:
            problems.append(
                f"{label}: run.json target_tokens={target} != "
                f"registered budget {expected_target}"
            )
        try:
            run_max_length = int(payload["max_length"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{label}: run.json has no integer max_length")
        else:
            record["max_length"] = run_max_length
            if run_max_length != int(plan["training_max_length"]):
                problems.append(
                    f"{label}: run.json max_length={run_max_length} != "
                    f"registered {plan['training_max_length']}"
                )
        if consumed is not None and consumed < expected_target:
            problems.append(
                f"{label}: consumed {consumed} tokens < registered budget "
                f"{expected_target}"
            )

        manifest_target = payload.get("data_manifest_target_tokens")
        try:
            manifest_target_value = int(manifest_target)
        except (TypeError, ValueError):
            problems.append(
                f"{label}: run.json has no integer data_manifest_target_tokens"
            )
        else:
            record["data_manifest_target_tokens"] = manifest_target_value
            if manifest_target_value != int(plan["required_data_tokens"]):
                problems.append(
                    f"{label}: data manifest target {manifest_target_value} != "
                    f"required {plan['required_data_tokens']}"
                )

        sha = payload.get("data_manifest_sha256")
        if not isinstance(sha, str) or not sha:
            problems.append(f"{label}: run.json has no data_manifest_sha256")
        else:
            record["data_manifest_sha256"] = sha
            by_sha.setdefault(sha, []).append(label)

    if len(by_sha) > 1:
        detail = "; ".join(
            f"{sha[:12]}... <- {', '.join(labels)}"
            for sha, labels in sorted(by_sha.items())
        )
        problems.append(f"data manifest sha256 differs across arms: {detail}")

    return {
        "ok": not problems,
        "size": size,
        "manifest_sha256": next(iter(by_sha)) if len(by_sha) == 1 else None,
        "arms": arms,
        "problems": problems,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the Phase 1 E2B orchestration plan as JSON."
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=DEFAULT_PREREG,
        help="pre-registration YAML (default: prereg/prereg.yaml)",
    )
    parser.add_argument(
        "--size",
        default="e2b",
        choices=SIZES,
        help="model size (default: e2b)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="artifacts root (default: $GEMMALOOP_ARTIFACTS)",
    )
    parser.add_argument(
        "--block",
        default=None,
        help="inclusive loop block for the FLOPs-matched budget (e.g. 16-18)",
    )
    parser.add_argument(
        "--bridge",
        default=None,
        choices=("scalar", "none", "damped", "gch"),
        help="loop bridge recorded in the generated train/eval commands",
    )
    parser.add_argument(
        "--adapter-policy",
        default=None,
        choices=("always", "loop_only"),
    )
    parser.add_argument(
        "--lora-target-layers",
        default=None,
        choices=("language", "loop_block", "loop_block_coda"),
    )
    parser.add_argument(
        "--teacher-data",
        type=Path,
        default=None,
        help="precomputed teacher shards passed to every Stage 1 run",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="validate the training runs instead of printing the plan",
    )
    parser.add_argument(
        "--verify-eval",
        dest="verify_eval",
        default=None,
        metavar="STEP_ID",
        help="re-check the expected RunResult files of one evaluation step",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raise RuntimeError(
        "phase1_plan is the retired single-SFT plan; prereg-v1 uses the "
        "two-stage hybrid and the E1-E4 analysis."
    )
    args = _parse_args(argv)
    if args.audit:
        summary = audit_training(
            args.prereg,
            artifacts=args.artifacts,
            size=args.size,
            loop_block=args.block,
            bridge=args.bridge,
            adapter_policy=args.adapter_policy,
            lora_target_layers=args.lora_target_layers,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not summary["ok"]:
            for problem in summary["problems"]:
                print(f"AUDIT FAILED: {problem}", file=sys.stderr)
            return 1
        return 0
    plan = build_plan(
        args.prereg,
        artifacts=args.artifacts,
        size=args.size,
        loop_block=args.block,
        bridge=args.bridge,
        adapter_policy=args.adapter_policy,
        lora_target_layers=args.lora_target_layers,
        teacher_data=args.teacher_data,
    )
    if args.verify_eval:
        try:
            summary = verify_eval_step(plan, args.verify_eval)
        except ValueError as exc:
            print(f"EVAL VERIFY FAILED: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not summary["ok"]:
            print(
                f"EVAL VERIFY FAILED: {args.verify_eval}: {summary['detail']}",
                file=sys.stderr,
            )
            return 1
        return 0
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
