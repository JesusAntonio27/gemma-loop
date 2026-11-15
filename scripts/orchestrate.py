"""Cross-platform retry orchestrator for long training/eval commands (Task 14b).

Exit-code contract (crash safety v2):

* ``0`` -- done (also when ``DONE.json`` already exists; a completed run is
  never re-launched);
* ``2`` -- terminal refusal (identity mismatch, bad configuration);
* ``76`` -- terminal heal-monitor abort (never retried).  The alert is copied
  into ``orchestrator.jsonl`` and ``status.json``; a relaunch requires the
  explicit ``--ack-abort <alert-id>`` that is recorded in ``run.json``;
* any other non-zero -- retryable, including CUDA/driver errors, OOM, killed
  children and Windows crash codes.  Retries continue while ``tokens_seen``
  progresses; after five consecutive no-progress failures the orchestrator
  stops with 75.  The retry counter/backoff reset whenever progress happened.

A retry resumes from the newest checkpoint, but an explicit ``--resume <path>``
is preserved until the child has published a newer checkpoint.  Lock waits have
their own 24 h budget and do not consume the crash-retry budget.

Every event is appended to ``<out>/orchestrator.jsonl``.  The orchestrator does
not hold the GPU lock itself: it probes the lock, then the child owns it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gemmaloop.gpulock import GpuLockBusy, acquire, try_acquire
from gemmaloop.train.checkpoint import (
    ABORT_GUARD_FILE,
    DONE_FILE,
    RetryableIOError,
    archive_run_state,
    atomic_write_json,
    decide_abort_resume,
    find_latest_valid_checkpoint,
    has_run_state,
    list_checkpoint_dirs,
    list_complete_checkpoints,
    read_done,
)

RETRYABLE_EXIT_CODE = 75
TERMINAL_EXIT_CODE = 2
MONITOR_ABORT_EXIT_CODE = 76
DEFAULT_MAX_NO_PROGRESS = 5
DEFAULT_MAX_RETRIES = 20
DEFAULT_BACKOFF_INITIAL = 60
DEFAULT_BACKOFF_MAX = 3600.0
DEFAULT_LOCK_WAIT_HOURS = 24.0
ORCHESTRATOR_LOG = "orchestrator.jsonl"
MONITOR_ALERTS_FILE = "alerts.json"
MONITOR_STATE_FILE = "heal_monitor_state.json"

__all__ = [
    "ABORT_GUARD_FILE",
    "MONITOR_ABORT_EXIT_CODE",
    "DEFAULT_BACKOFF_INITIAL",
    "DEFAULT_BACKOFF_MAX",
    "DEFAULT_MAX_NO_PROGRESS",
    "DEFAULT_MAX_RETRIES",
    "ORCHESTRATOR_LOG",
    "RETRYABLE_EXIT_CODE",
    "backoff_seconds",
    "done_signature",
    "main",
    "run_progress",
    "with_resume",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def backoff_seconds(
    retry_index: int,
    *,
    initial: float = DEFAULT_BACKOFF_INITIAL,
    maximum: float = DEFAULT_BACKOFF_MAX,
) -> float:
    """60 s, 120 s, 300 s, then double up to ``maximum``."""
    if retry_index < 0:
        raise ValueError("retry_index must be non-negative")
    base = max(0.0, float(initial))
    if base == 0:
        return 0.0
    schedule = [base, base * 2.0, base * 5.0]
    if retry_index < len(schedule):
        return min(float(maximum), schedule[retry_index])
    delay = schedule[-1]
    for _ in range(retry_index - len(schedule) + 1):
        delay *= 2.0
        if delay >= maximum:
            return float(maximum)
    return float(min(delay, maximum))


def _resume_value(command: Sequence[str]) -> str | None:
    parts = list(command)
    for index, item in enumerate(parts):
        if item == "--resume" and index + 1 < len(parts):
            return parts[index + 1]
        if item.startswith("--resume="):
            return item.split("=", 1)[1]
    return None


def set_resume(command: Sequence[str], value: str) -> list[str]:
    """Return ``command`` with ``--resume`` replaced (or appended) by ``value``."""
    result: list[str] = []
    index = 0
    parts = list(command)
    while index < len(parts):
        item = parts[index]
        if item == "--resume":
            index += 2
            continue
        if item.startswith("--resume="):
            index += 1
            continue
        result.append(item)
        index += 1
    result.extend(["--resume", str(value)])
    return result


def with_resume(
    command: Sequence[str], resume_argument: str = "--resume auto"
) -> list[str]:
    """Compatibility helper: force the auto resume argument."""
    parts = resume_argument.split()
    value = parts[-1] if len(parts) >= 2 and parts[-2] == "--resume" else "auto"
    return set_resume(command, value)


def _checkpoint_step(path: Path) -> int | None:
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        return None
    suffix = name[len("checkpoint-"):]
    return int(suffix) if suffix.isdigit() else None


def _gpu_lock_free() -> bool:
    """Check the lock without keeping it (the child must own it itself)."""
    held = try_acquire()
    if held is None:
        return False
    held.release()
    return True


def is_done(out_dir: str | Path) -> bool:
    """True when DONE.json (or a complete run.json) marks the run finished."""
    out_dir = Path(out_dir)
    done = read_done(out_dir)
    if isinstance(done, dict) and done.get("status") == "complete":
        return True
    run_json = out_dir / "run.json"
    if run_json.is_file():
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and payload.get("status") == "complete"
    return False


def done_signature(out_dir: str | Path) -> str | None:
    """Content hash of the current ``DONE.json`` (``None`` when absent).

    A forced rerun must only keep ``--rerun-complete`` while the STALE DONE
    written before the launch is still the one on disk.  A child that writes a
    fresh DONE and then exits non-zero (for example a Windows teardown crash)
    must not be relaunched into an archive loop (R2-M1).
    """

    path = Path(out_dir) / DONE_FILE
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run_progress(out_dir: str | Path) -> tuple[int, int]:
    """Durable progress: only verified COMPLETE checkpoints count (N5)."""
    out_dir = Path(out_dir)
    tokens = 0
    step = 0
    for info in list_complete_checkpoints(out_dir, verify=False):
        try:
            value = int(info.metadata.get("tokens_seen", 0) or 0)
        except (TypeError, ValueError):
            value = 0
        tokens = max(tokens, value)
        step = max(step, int(info.step))
    return tokens, step


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return any(item == flag for item in command)


def strip_fresh_archive(command: Sequence[str]) -> list[str]:
    """Drop --fresh/--rerun-complete before a retry (N3)."""
    return [
        item
        for item in command
        if item not in ("--fresh", "--rerun-complete")
    ]


def newer_checkpoint(out_dir: str | Path, step: int) -> Path | None:
    for info in reversed(list_complete_checkpoints(out_dir, verify=False)):
        if int(info.step) > int(step):
            return info.path
    return None


def alert_id(alert: dict) -> str:
    """Stable id for one persisted monitor alert."""
    canonical = json.dumps(alert, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_abort_alert(out_dir: str | Path) -> dict:
    """Return the latest abort alert from ``alerts.json`` (or a synthetic one)."""
    path = Path(out_dir) / MONITOR_ALERTS_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    alerts = payload.get("alerts") if isinstance(payload, dict) else None
    aborts = [
        dict(alert)
        for alert in (alerts or [])
        if isinstance(alert, dict) and alert.get("severity") == "abort"
    ]
    if aborts:
        alert = aborts[-1]
    else:
        alert = {
            "severity": "abort",
            "rule": "heal_monitor_abort",
            "message": "the child exited with the heal-monitor abort code",
        }
    alert.setdefault("alert_id", alert_id(alert))
    return alert


def write_abort_guard(out_dir: str | Path, alert: dict) -> Path:
    return atomic_write_json(
        Path(out_dir) / ABORT_GUARD_FILE,
        {
            "alert_id": str(alert["alert_id"]),
            "alert": alert,
            "returncode": MONITOR_ABORT_EXIT_CODE,
            "created_utc": utc_now(),
        },
    )


def _merge_json(path: Path, fields: dict) -> dict:
    payload: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(fields)
    atomic_write_json(path, payload)
    return payload


def record_monitor_abort(out_dir: str | Path, alert: dict) -> None:
    """Persist the abort for the orchestrator and the run status."""
    out_dir = Path(out_dir)
    write_abort_guard(out_dir, alert)
    _merge_json(
        out_dir / "status.json",
        {
            "phase": "aborted-heal-monitor",
            "exit_code": MONITOR_ABORT_EXIT_CODE,
            "exit_reason": f"heal-monitor abort: {alert.get('rule')}: {alert.get('message')}",
            "heal_monitor_abort": alert,
            "updated_utc": utc_now(),
        },
    )


def record_abort_ack(out_dir: str | Path, alert: dict) -> None:
    """Record ``--ack-abort`` in run.json and clear the persisted abort flags."""
    out_dir = Path(out_dir)
    payload = _merge_json(
        out_dir / "run.json",
        {},
    )
    alert_id = str(alert["alert_id"])
    acks = list(payload.get("abort_acks") or [])
    # R3-M2: a real train.py child records the same ack while starting; the
    # orchestrator must not append a second entry for it after the child exits.
    if not any(
        isinstance(item, dict) and str(item.get("alert_id")) == alert_id
        for item in acks
    ):
        acks.append(
            {
                "alert_id": alert_id,
                "acked_utc": utc_now(),
                "source": "orchestrate --ack-abort",
            }
        )
        _merge_json(out_dir / "run.json", {"abort_acks": acks})
    guard = out_dir / ABORT_GUARD_FILE
    try:
        guard.unlink()
    except FileNotFoundError:
        pass
    state_path = out_dir / MONITOR_STATE_FILE
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if isinstance(state, dict):
            rules = state.get("rules")
            if isinstance(rules, dict):
                rules["abort_triggered"] = False
                rules["abort_emitted"] = []
                rules["warn_last"] = {}
            state["abort_ack"] = {
                "alert_id": str(alert["alert_id"]),
                "acked_utc": utc_now(),
            }
            atomic_write_json(state_path, state)


def check_abort_ack(out_dir: str | Path, requested: str | None) -> tuple[bool, dict | None]:
    """Return ``(proceed, guard)``; a mismatch is a terminal refusal."""

    decision = decide_abort_resume(out_dir, requested)
    if decision.action == "refuse":
        return False, dict(decision.guard or {})
    return True, None if decision.guard is None else dict(decision.guard)


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        payload = {"ts": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="run directory")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--max-no-progress",
        type=int,
        default=None,
        help="consecutive no-progress failures before stopping (default 5)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="deprecated alias for --max-no-progress",
    )
    parser.add_argument("--backoff-initial", type=float, default=DEFAULT_BACKOFF_INITIAL)
    parser.add_argument("--backoff-max", type=float, default=DEFAULT_BACKOFF_MAX)
    parser.add_argument(
        "--lock-wait-hours",
        type=float,
        default=DEFAULT_LOCK_WAIT_HOURS,
        help="time budget for waiting on another GPU job",
    )
    parser.add_argument(
        "--resume-arg",
        default="--resume auto",
        help="arguments appended to a retried command",
    )
    parser.add_argument(
        "--no-lock-check",
        action="store_true",
        help="skip the pre-launch GPU-lock probe (tests only)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="do not auto-add --resume auto even when the run dir has a checkpoint",
    )
    parser.add_argument(
        "--ack-abort",
        default=None,
        metavar="ALERT_ID",
        help=(
            "acknowledge a persisted heal-monitor abort (exit 76) and clear its "
            "flags; the ack is recorded in run.json"
        ),
    )
    parser.add_argument("--cwd", default=None, help="working directory for the child")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        _run_lock = acquire(args.out / ".run.lock")
    except GpuLockBusy:
        print(f"another orchestrator owns {args.out}", file=sys.stderr)
        return RETRYABLE_EXIT_CODE

    max_no_progress = int(
        args.max_no_progress
        if args.max_no_progress is not None
        else (args.max_retries if args.max_retries is not None else DEFAULT_MAX_NO_PROGRESS)
    )
    if max_no_progress <= 0:
        max_no_progress = DEFAULT_MAX_NO_PROGRESS

    log = RunLog(args.out / ORCHESTRATOR_LOG)
    log.write(
        "start",
        name=args.name or args.out.name,
        command=command,
        max_no_progress=max_no_progress,
    )
    if is_done(args.out) and not (
        _has_flag(command, "--fresh") or _has_flag(command, "--rerun-complete")
    ):
        log.write("done", reason="already complete", attempts=0)
        return 0

    abort_decision = decide_abort_resume(args.out, args.ack_abort)
    guard = None if abort_decision.guard is None else dict(abort_decision.guard)
    if abort_decision.action == "refuse":
        alert = (guard or {}).get("alert") or {}
        wanted = (guard or {}).get("alert_id")
        message = (
            f"heal monitor aborted run {args.out} (alert {wanted}): "
            f"{alert.get('rule', '?')}: {alert.get('message', '')}. "
            f"Relaunch with --ack-abort {wanted} to resume."
        )
        print(message, file=sys.stderr)
        log.write(
            "stop",
            reason="abort not acknowledged",
            returncode=TERMINAL_EXIT_CODE,
            alert_id=wanted,
            alert=alert,
        )
        return TERMINAL_EXIT_CODE
    ack_guard = guard
    if guard is not None:
        log.write(
            "ack-abort",
            alert_id=(guard.get("alert_id")),
            alert=(guard.get("alert") or {}),
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["GEMMALOOP_RUN_LOCK_HELD"] = str((args.out / ".run.lock").resolve())
    cwd = args.cwd
    attempt_command = list(command)
    # ``train.py`` now enforces the same abort rule.  The orchestrator must
    # hand the acknowledgement to the child before it starts, otherwise a
    # valid resume would be refused as soon as it reads the guard file.
    if guard is not None and not _has_flag(attempt_command, "--ack-abort"):
        attempt_command.extend(["--ack-abort", str(guard.get("alert_id"))])
    explicit_resume = _resume_value(command)
    if explicit_resume == "auto":
        explicit_resume = None

    existing = find_latest_valid_checkpoint(args.out)
    command_is_fresh = _has_flag(command, "--fresh") or _has_flag(
        command, "--rerun-complete"
    )
    force_rerun = command_is_fresh
    stale_done = done_signature(args.out) if force_rerun else None
    # A crash before the first checkpoint leaves run_identity.json/status.json
    # (and maybe monitor files) but nothing resumable.  The next train child
    # would exit 2 on the existing run state, so archive that partial directory
    # and restart from zero with no manual step.  A valid checkpoint disables
    # this path; an explicit --resume branch is respected.  An acknowledged
    # monitor abort must archive in the SAME launch (N1): otherwise the acked
    # child sees the old run state and exits 2 once before the next relaunch
    # archives it.
    if (
        existing is None
        and not is_done(args.out)
        and not command_is_fresh
        and explicit_resume is None
        and (
            has_run_state(args.out)
            # An acknowledged abort is a restart from zero in this same
            # launch, even if a partial run directory holds no run-state file
            # yet (N1).
            or abort_decision.action == "restart"
        )
    ):
        try:
            destination = archive_run_state(
                args.out, reason="partial-no-checkpoint"
            )
        except RetryableIOError:
            log.write("stop", reason="could not archive the partial run dir")
            return RETRYABLE_EXIT_CODE
        log.write("archive-partial", destination=str(destination))
    if (
        existing is not None
        and not args.fresh
        and explicit_resume is None
        and not command_is_fresh
    ):
        attempt_command = with_resume(attempt_command, str(args.resume_arg))
        log.write(
            "auto-resume",
            checkpoint=str(existing.path),
            step=existing.step,
        )

    best_progress = run_progress(args.out)
    no_progress = 0
    retry_index = 0
    lock_wait_seconds = 0.0
    lock_index = 0
    attempts = 0

    while True:
        if is_done(args.out) and not (
            _has_flag(attempt_command, "--fresh")
            or _has_flag(attempt_command, "--rerun-complete")
        ):
            log.write("done", reason="already complete", attempts=attempts)
            return 0
        if not args.no_lock_check and not _gpu_lock_free():
            log.write("blocked", reason="gpu lock is held", attempt=attempts)
            if lock_wait_seconds >= float(args.lock_wait_hours) * 3600.0:
                log.write("stop", reason="lock wait budget exhausted")
                return RETRYABLE_EXIT_CODE
            delay = backoff_seconds(
                lock_index,
                initial=float(args.backoff_initial),
                maximum=float(args.backoff_max),
            )
            log.write("wait", attempt=attempts, seconds=delay, reason="gpu lock")
            lock_wait_seconds += delay
            lock_index += 1
            if delay > 0:
                time.sleep(delay)
            continue

        log.write("launch", attempt=attempts, command=attempt_command)
        started = time.time()
        try:
            completed = subprocess.run(attempt_command, env=env, cwd=cwd)
            code = int(completed.returncode)
        except OSError as exc:
            log.write("error", attempt=attempts, error=str(exc))
            return 1
        attempts += 1
        elapsed = round(time.time() - started, 3)
        log.write("exit", attempt=attempts, returncode=code, seconds=elapsed)
        if ack_guard is not None:
            # The child actually started: only now consume the ack, so a
            # failure to spawn cannot lose it.
            record_abort_ack(
                args.out,
                dict(ack_guard.get("alert") or {"alert_id": ack_guard["alert_id"]}),
            )
            ack_guard = None

        current_done = done_signature(args.out)
        fresh_done = (
            force_rerun
            and current_done is not None
            and current_done != stale_done
        )
        if is_done(args.out) and (
            not force_rerun or code == 0 or fresh_done
        ):
            log.write(
                "done",
                reason=(
                    "fresh DONE written by the forced rerun"
                    if fresh_done
                    else "DONE.json present"
                ),
                attempts=attempts,
            )
            return 0
        if code == 0:
            log.write("done", attempts=attempts)
            return 0
        if code == TERMINAL_EXIT_CODE:
            log.write("stop", reason="terminal refusal", returncode=code)
            return code
        if code == MONITOR_ABORT_EXIT_CODE:
            alert = load_abort_alert(args.out)
            record_monitor_abort(args.out, alert)
            log.write(
                "stop",
                reason="heal-monitor abort",
                returncode=code,
                alert_id=alert.get("alert_id"),
                alert=alert,
            )
            return code

        current = run_progress(args.out)
        if current > best_progress:
            best_progress = current
            no_progress = 0
            retry_index = 0
            lock_index = 0
            log.write(
                "progress",
                tokens_seen=current[0],
                step=current[1],
                attempts=attempts,
            )
        else:
            no_progress += 1

        if no_progress >= max_no_progress:
            log.write(
                "stop",
                reason="no progress",
                returncode=code,
                attempts=attempts,
                no_progress=no_progress,
            )
            return RETRYABLE_EXIT_CODE

        delay = backoff_seconds(
            retry_index,
            initial=float(args.backoff_initial),
            maximum=float(args.backoff_max),
        )
        log.write("wait", attempt=attempts, seconds=delay, no_progress=no_progress)
        if delay > 0:
            time.sleep(delay)
        retry_index += 1

        # N2: a forced rerun whose child exits 75 before train.py archives
        # still has the old DONE.json.  Dropping the force flag now would make
        # the loop-top is_done check return 0 and the harness would record the
        # stale run as retrained.  Keep the flag until the stale DONE is gone,
        # and do not add --resume (train.py refuses that combination).
        # Keep --rerun-complete only while the STALE DONE written before this
        # launch is still present.  A fresh DONE from the failure path above
        # returned already; a rewritten DONE here must not be re-archived.
        keep_force = (
            force_rerun
            and is_done(args.out)
            and current_done == stale_done
        )
        if keep_force:
            attempt_command = list(attempt_command)
        else:
            attempt_command = strip_fresh_archive(attempt_command)
            if explicit_resume is not None:
                try:
                    step = _checkpoint_step(Path(explicit_resume))
                except (OSError, ValueError):
                    step = None
                if step is not None and newer_checkpoint(args.out, step) is None:
                    # Preserve the caller's explicit branch until it publishes
                    # a newer checkpoint of its own.
                    attempt_command = list(attempt_command)
                else:
                    attempt_command = with_resume(
                        attempt_command, str(args.resume_arg)
                    )
            else:
                attempt_command = with_resume(attempt_command, str(args.resume_arg))


if __name__ == "__main__":
    sys.exit(main())
