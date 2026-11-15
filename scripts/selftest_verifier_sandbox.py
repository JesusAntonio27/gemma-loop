"""Synthetic real-Docker validation of the shortfall verifier boundary.

This harness is intentionally separate from the official v3 pipeline.  It:

* uses an already-local, explicitly pinned image and refuses to pull one;
* uses the official ``DockerSandbox`` configuration and the registered 6 s
  outer timeout;
* never reads pilot inputs, the pool, the v3 freeze or the v3 registry;
* writes a separate *synthetic* validation receipt whose outcomes can never
  become official count results.

It exercises the real Docker boundary for a known pass, wrong answer, candidate
exception, deadline sleep, stdout/stderr spam, evidence/key/parent-process
attacks, injected startup failure, post-completion delay, missing/altered
markers, failed cleanup, retry exhaustion, ledger crashes, and a documented
recovery followed by the single permitted retry.  Marker-specific expectations are parameterised because the
marker channel is being ruled on separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gemmaloop.stage0 import grader  # noqa: E402

RECEIPT_KIND = "pilot_shortfall_synthetic_validation_receipt"
RECEIPT_VERSION = 1
OFFICIAL_TIMEOUT_S = 6.0
DEFAULT_OUT = Path("synthetic_validation_receipt.json")


class SyntheticValidationError(RuntimeError):
    """The synthetic harness cannot run or a case violated the contract."""


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def inspect_local_image(
    image: str, *, runner: Callable[..., Any] = subprocess.run
) -> dict[str, Any]:
    """Return the local image record, refusing to pull a missing image."""

    completed = runner(
        ["docker", "image", "inspect", str(image)],
        capture_output=True,
        text=True,
        timeout=30,
        env=grader.sanitized_env(),
    )
    if getattr(completed, "returncode", 1) != 0:
        raise SyntheticValidationError(
            f"image {image!r} is not present locally; this harness never pulls "
            f"({getattr(completed, 'stderr', '')!r})"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SyntheticValidationError(
            f"docker image inspect {image!r} returned non-JSON"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise SyntheticValidationError(
            f"docker image inspect {image!r} returned no image"
        )
    return payload[0]


def image_digest(entry: Mapping[str, Any]) -> str:
    return str(entry.get("Id") or "")


def verify_local_digest(
    image: str,
    *,
    expected_digest: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    entry = inspect_local_image(image, runner=runner)
    actual = image_digest(entry)
    if expected_digest is not None and actual != str(expected_digest):
        raise SyntheticValidationError(
            f"local image {image!r} resolves to {actual!r}, not the pinned "
            f"digest {expected_digest!r}"
        )
    return {
        "tag": image,
        "id": actual,
        "repo_digests": list(entry.get("RepoDigests") or []),
    }


class LocalOnlySandbox(grader.DockerSandbox):
    """DockerSandbox that verifies the local image and never pulls."""

    def ensure_image(self) -> None:  # noqa: D401 - explicit override
        inspect_local_image(self.image)
        self.verify_image_digest()


def _marker_error_case(mode: str) -> tuple[str, str | None]:
    """Build classifiers for the parameterised marker cases without Docker."""

    nonce = "a" * 64
    with tempfile.TemporaryDirectory(prefix="gemmaloop_marker_case_") as tmp:
        evidence = Path(tmp)
        path = evidence / grader.LIFECYCLE_EVIDENCE_FILE
        if mode == "missing":
            return (
                grader.classify_lifecycle_evidence(
                    markers=[],
                    marker_error="no lifecycle evidence file",
                    timed_out=False,
                    cleanup_verified=True,
                    deadline_epoch=100.0,
                    status="exited",
                ),
                None,
            )
        if mode == "altered":
            path.write_text("deadbeef {}\n", encoding="utf-8")
            markers, error = grader.read_lifecycle_markers(evidence, nonce)
            return (
                grader.classify_lifecycle_evidence(
                    markers=markers,
                    marker_error=error,
                    timed_out=False,
                    cleanup_verified=True,
                    deadline_epoch=100.0,
                    status="exited",
                ),
                error,
            )
        if mode == "out_of_order":
            markers = [
                {
                    "event": grader.LIFECYCLE_EVENT_START,
                    "epoch": 90.0,
                },
                {
                    "event": grader.LIFECYCLE_EVENT_FINISH,
                    "epoch": 80.0,
                },
            ]
            return (
                grader.classify_lifecycle_evidence(
                    markers=markers,
                    marker_error=None,
                    timed_out=False,
                    cleanup_verified=True,
                    deadline_epoch=100.0,
                    status="exited",
                ),
                None,
            )
        raise SyntheticValidationError(f"unknown marker case {mode!r}")


# Parameterised marker cases: the marker design may change after the X1
# ruling, so these expectations are data, not hard-coded control flow.
MARKER_CASES: tuple[dict[str, Any], ...] = (
    {"name": "missing_markers", "mode": "missing", "expect": "infrastructure"},
    {"name": "altered_markers", "mode": "altered", "expect": "infrastructure"},
    {
        "name": "out_of_order_markers",
        "mode": "out_of_order",
        "expect": "infrastructure",
    },
    {
        "name": "post_completion_delay",
        "mode": "post_completion",
        "expect": "infrastructure",
    },
)


def classify_marker_case(case: Mapping[str, Any]) -> tuple[str, str | None]:
    if case["mode"] == "post_completion":
        markers = [
            {"event": grader.LIFECYCLE_EVENT_START, "epoch": 90.0},
            {"event": grader.LIFECYCLE_EVENT_FINISH, "epoch": 95.0},
        ]
        return (
            grader.classify_lifecycle_evidence(
                markers=markers,
                marker_error=None,
                timed_out=True,
                cleanup_verified=True,
                deadline_epoch=100.0,
                status="exited",
            ),
            None,
        )
    return _marker_error_case(str(case["mode"]))


def _class_is_infrastructure(value: str) -> bool:
    return value in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES


# Exact record errors that a late-start race may legitimately produce.  The
# substring check used in round 9 also accepted unrelated entry errors, so the
# two documented dispositions are now matched exactly.
LATE_START_RECORD_ERRORS = frozenset(
    {
        "execution entry was not received by the host before its registered "
        "deadline",
        "expected exactly one successful execution-entry record, found 0",
    }
)


def late_receipt_deterministic_case() -> dict[str, Any]:
    """A late entry that was positively received and then rejected.

    The fixture contains the authenticated execution-entry record, so the host
    did observe the late entry; the host receipt stamp is after its deadline.
    This is deterministic and independent of Docker scheduling.
    """

    markers = [
        {
            "event": grader.LIFECYCLE_EVENT_BUDGET,
            "remaining_s": 0.1,
        },
        {
            "event": grader.LIFECYCLE_EVENT_START,
            "monotonic": 1.0,
            "child_pid": 4242,
        },
        {
            "event": grader.LIFECYCLE_EVENT_ALIVE,
            "monotonic": 2.0,
            "notice_monotonic": 1.9,
            "child_pid": 4242,
        },
        {
            "event": grader.LIFECYCLE_EVENT_FINISH,
            "monotonic": 2.1,
            "child_pid": 4242,
            "terminated": True,
            "exit_observed": True,
        },
    ]
    classification = grader.classify_lifecycle_evidence(
        markers=markers,
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        deadline_epoch=100.0,
        status="exited",
        host_completed_within_budget=False,
        host_start_received_within_budget=False,
    )
    record_error = grader.lifecycle_record_error(
        markers=markers,
        marker_error=None,
        cleanup_verified=True,
        timed_out=True,
        host_completed_within_budget=False,
        host_start_received_within_budget=False,
    )
    late_entry_received = any(
        marker.get("event") == grader.LIFECYCLE_EVENT_START
        for marker in markers
    )
    detail = {
        "late_entry_received": late_entry_received,
        "host_start_received_within_budget": False,
        "late_entry_rejected": (
            classification in grader.INFRASTRUCTURE_LIFECYCLE_CLASSES
            and record_error
            == "execution entry was not received by the host before its "
            "registered deadline"
        ),
        "record_error": record_error,
    }
    return _case(
        "late_receipt_received_after_deadline",
        ok=(
            late_entry_received
            and _class_is_infrastructure(classification)
            and detail["late_entry_rejected"]
            and classification not in grader.CANDIDATE_DEADLINE_CLASSES
        ),
        classification=classification,
        detail=detail,
    )


def container_is_gone(
    name: str, *, runner: Callable[..., Any] = subprocess.run
) -> bool:
    completed = runner(
        ["docker", "inspect", "--format", "{{.Id}}", str(name)],
        capture_output=True,
        text=True,
        timeout=15,
        env=grader.sanitized_env(),
    )
    text = (
        str(getattr(completed, "stdout", "") or "")
        + str(getattr(completed, "stderr", "") or "")
    ).lower()
    return getattr(completed, "returncode", 1) != 0 and "no such" in text


def _write_workdir(workdir: Path, code: str, *, wrapper_source: str | None = None) -> None:
    grader._write_candidate(workdir, code)
    grader._write_lifecycle_runner(workdir)
    if wrapper_source is None:
        grader._write_lifecycle_wrapper(workdir)
    else:
        (workdir / grader.LIFECYCLE_WRAPPER_NAME).write_text(
            wrapper_source, encoding="utf-8"
        )


def _run_raw(
    sandbox: grader.DockerSandbox,
    workdir: Path,
    *,
    script: str = "candidate.py",
    timeout: float = OFFICIAL_TIMEOUT_S,
    with_evidence: bool = True,
    cleanup: Callable[..., Any] | None = None,
    marker_reader: Callable[..., Any] | None = None,
    state_probe: Callable[..., Any] | None = None,
    command_transform: Callable[[list[str]], list[str]] | None = None,
    input_text: str = "",
    container_name: str | None = None,
    enforcement_spawner: Callable[..., Any] | None = None,
) -> tuple[Any, str, str | None]:
    container_name = container_name or (
        f"gemmaloop-selftest-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    )
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_ev_") as ev_tmp:
        evidence = Path(ev_tmp) if with_evidence else None
        key = os.urandom(grader.LIFECYCLE_KEY_BYTES // 2).hex() if evidence else None
        attempt = os.urandom(16).hex()
        if evidence is not None:
            try:
                os.chmod(evidence, 0o777)
            except OSError:  # pragma: no cover - platform specific
                pass
        command = sandbox.command(
            workdir,
            script,
            container_name=container_name,
            evidence_dir=evidence,
            attempt=attempt,
        )
        if command_transform is not None:
            command = command_transform(list(command))
        kwargs: dict[str, Any] = {}
        if state_probe is not None:
            kwargs["state_probe"] = state_probe
        if enforcement_spawner is not None:
            kwargs["enforcement_spawner"] = enforcement_spawner
        result = grader._run_process_docker(
            command,
            input_text=input_text,
            timeout=float(timeout),
            container_name=container_name,
            evidence_dir=evidence,
            key=key,
            attempt=attempt,
            cleanup=cleanup or grader._remove_container,
            marker_reader=marker_reader or grader.read_lifecycle_markers,
            **kwargs,
        )
    return result, container_name, key


def _case(
    name: str,
    *,
    ok: bool,
    classification: str,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "classification": classification,
        "detail": dict(detail or {}),
    }


def _result_case(name: str, result: Any, *, expect: str) -> dict[str, Any]:
    test = result.tests[0]
    classification = str(test.lifecycle_class or grader.LIFECYCLE_UNRESOLVED)
    if expect == "passed":
        ok = bool(test.passed) and classification == grader.LIFECYCLE_COMPLETED_BEFORE
    elif expect == "wrong_answer":
        ok = (
            not test.passed
            and test.error == "wrong answer"
            and classification == grader.LIFECYCLE_COMPLETED_BEFORE
        )
    elif expect == "candidate_error":
        ok = (
            not test.passed
            and test.error
            and str(test.error).startswith("docker exit code")
            and classification == grader.LIFECYCLE_COMPLETED_BEFORE
        )
    elif expect == "timeout":
        ok = (
            test.timed_out
            and classification in grader.CANDIDATE_DEADLINE_CLASSES
        )
    elif expect == "infrastructure":
        ok = (not test.passed) and _class_is_infrastructure(classification)
    else:  # pragma: no cover - defensive
        raise SyntheticValidationError(f"unknown expectation {expect!r}")
    return _case(
        name,
        ok=ok,
        classification=classification,
        detail={
            "passed": bool(test.passed),
            "error": test.error,
            "timed_out": bool(test.timed_out),
            "cleanup_verified": test.cleanup_verified,
            "lifecycle_valid": test.lifecycle_valid,
            "marker_error": test.marker_error,
            "container_name": test.container_name,
            "stdout_head": (test.stdout or "")[:300],
            "stderr_head": (test.stderr or "")[:300],
        },
    )


def _raw_case(
    name: str,
    lifecycle: Mapping[str, Any],
    *,
    expect: str,
    code_rc: int | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    classification = str(lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED)
    if expect == "infrastructure":
        ok = _class_is_infrastructure(classification)
    elif expect == "deadline":
        ok = classification in grader.CANDIDATE_DEADLINE_CLASSES
    elif expect == "cleanup_failure":
        ok = (
            lifecycle.get("cleanup_verified") is False
            and _class_is_infrastructure(classification)
        )
    else:  # pragma: no cover - defensive
        raise SyntheticValidationError(f"unknown raw expectation {expect!r}")
    return _case(
        name,
        ok=ok,
        classification=classification,
        detail={
            "class": classification,
            "marker_error": lifecycle.get("marker_error"),
            "record_error": lifecycle.get("record_error"),
            "lifecycle_valid": lifecycle.get("lifecycle_valid"),
            "cleanup_verified": lifecycle.get("cleanup_verified"),
            "container_name": lifecycle.get("container_name"),
            "container_status": lifecycle.get("container_status"),
            "late_completion": lifecycle.get("late_completion"),
            "exit_code": code_rc,
            **(dict(payload or {})),
        },
    )


_CRASH_WORKER_SOURCE = r"""import json
import sys
import time

config = json.loads(open(sys.argv[1], encoding="utf-8").read())
sys.path.insert(0, config["repo"])

from gemmaloop.stage0 import grader, shortfall

sandbox = grader.DockerSandbox(
    image=config["image"], expected_digest=config["image_id"]
)


def grade_fn(code, tests, timeout, container_name=None, attempt=None):
    # Leave a small window after the ledger's per-test start record so the
    # parent can kill this process while the next test is genuinely in flight.
    time.sleep(0.25)
    return sandbox.run_tests(
        code,
        tests,
        timeout=timeout,
        container_names=(
            [container_name] if container_name is not None else None
        ),
        attempts=[attempt] if attempt is not None else None,
    )


authorization = shortfall.V3RunAuthorization(
    registry_path=config["registry_path"],
    run_token=config["run_token"],
    stage="count",
)
entries = shortfall.grade_original_requests(
    config["freeze"],
    pool_dir=config["pool_dir"],
    lite_dir=config["lite_dir"],
    grade_fn=grade_fn,
    workers=1,
    max_infrastructure_retries=1,
    recovery_check=shortfall._docker_recovery_check,
    ledger_path=config["ledger"],
    ledger_identity=config.get("ledger_identity"),
    resume=bool(config.get("resume")),
    authorization=authorization,
)
with open(config["out"], "w", encoding="utf-8") as handle:
    json.dump(entries, handle)
"""


def _live_ledger_result(path: Path, request_id: str, test_index: int) -> bool:
    """Read framed records without truncating a concurrently written tail."""

    from gemmaloop.stage0 import shortfall

    if not path.is_file():
        return False
    data = path.read_bytes()
    for line in data.split(b"\n"):
        if not line:
            continue
        try:
            record = shortfall._decode_ledger_frame(line)
        except ValueError:
            continue
        if (
            record.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
            and str(record.get("request_id")) == request_id
            and int(record.get("test_index", -1)) == test_index
        ):
            return True
    return False


def _run_ledger_crash_case(
    sandbox: grader.DockerSandbox,
    *,
    image: str,
    image_id: str | None,
    timeout: float,
) -> dict[str, Any]:
    """Kill a real multi-test Docker grade after its first committed result."""

    from gemmaloop.stage0 import shortfall

    request_id = "crash:c0:0"
    problem_id = "crash:c0"
    tests = [
        {"input": "1", "output": "1", "type": "stdin"},
        {"input": "2", "output": "2", "type": "stdin"},
        {"input": "3", "output": "3", "type": "stdin"},
    ]
    code = "import sys\nprint(sys.stdin.read().strip())\n"
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_crash_") as tmp:
        root = Path(tmp)
        pool = root / "pool"
        lite = root / "lite"
        (pool).mkdir()
        (lite / "shards").mkdir(parents=True)
        (pool / "problems.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "source": "apps",
                    "kind": "code",
                    "prompt": "crash probe",
                    "answer": "None",
                    "tests": tests,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (lite / "shards" / "shard_00000.jsonl").write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "completion": "```python\n" + code + "```",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (lite / "sample_grades.jsonl").write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "problem_id": problem_id,
                    "kind": "code",
                    "kept": True,
                    "correct": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        freeze = {
            "manifest_sha256": "c" * 64,
            "tests": {"manifest_sha256": "d" * 64},
            "inputs": {"lite_dir": str(lite)},
            "selected_inputs": {
                "splits": {
                    "pilot_p1": {
                        "requests": [
                            {
                                "request_id": request_id,
                                "problem_id": problem_id,
                                "kind": "code",
                                "sample_index": 0,
                            }
                        ]
                    },
                    "pilot_tfit": {"requests": []},
                    "pilot_selection": {"requests": []},
                }
            },
            "verifier": {
                "timeout_s": float(timeout),
                "docker_image": {"pinned": str(image_id or image)},
            },
        }
        worker = root / "crash_worker.py"
        worker.write_text(_CRASH_WORKER_SOURCE, encoding="utf-8")
        ledger = root / "attempt_ledger.jsonl"
        registry_path = root / "v3_registry.json"
        run_token = "d" * 64
        registry = {
            "kind": shortfall.V3_REGISTRY_KIND,
            "version": 1,
            "status": shortfall.V3_STATUS_CLAIMED,
            "created_utc": shortfall._utc_now(),
            "updated_utc": shortfall._utc_now(),
            "official": True,
            "run_token": run_token,
            "repo_root": str(REPO.resolve()),
            "lite_dir": str(lite.resolve()),
            "out_root": str((root / "v3").resolve()),
            "current_stage": None,
            "last_completed_stage": None,
            "failed_stage": None,
            "stages": [],
            "resume_count": 0,
            "error": None,
            "registry_sha256": "",
        }
        registry["registry_sha256"] = shortfall._registry_hash(registry)
        registry_path.write_text(json.dumps(registry), encoding="utf-8")

        def run_worker(mode: str, out: Path) -> dict[str, Any]:
            config = {
                "repo": str(REPO),
                "image": image,
                "image_id": image_id or image,
                "pool_dir": str(pool),
                "lite_dir": str(lite),
                "ledger": str(ledger),
                "ledger_identity": {"synthetic": "crash", "timeout_s": timeout},
                "freeze": freeze,
                "resume": mode == "resume",
                "out": str(out),
                "registry_path": str(registry_path),
                "run_token": run_token,
            }
            config_path = root / f"config-{mode}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            return {"argv": [sys.executable, str(worker), str(config_path)]}

        first = run_worker("run", root / "first.json")
        process = subprocess.Popen(
            first["argv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        saw_first = False
        deadline = time.time() + 90
        while time.time() < deadline:
            if _live_ledger_result(ledger, request_id, 0):
                saw_first = True
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    text=True,
                )
            else:
                process.kill()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

        # Clean up any real container named in the ledger start records.
        for record in shortfall._read_attempt_ledger(ledger):
            if record.get("kind") == shortfall.LEDGER_ATTEMPT_START:
                name = record.get("container_name")
                if name:
                    grader._remove_container(str(name))

        second = run_worker("resume", root / "second.json")
        resumed = subprocess.run(
            second["argv"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        ok = False
        detail: dict[str, Any] = {
            "first_result_committed": saw_first,
            "resume_returncode": resumed.returncode,
            "resume_stderr": (resumed.stderr or "")[-1200:],
        }
        if saw_first and resumed.returncode == 0:
            entries = json.loads((root / "second.json").read_text(encoding="utf-8"))
            entry = next(
                item for item in entries if item["request_id"] == request_id
            )
            rows = {int(row["index"]): row for row in entry["per_test"]}
            attempts0 = rows[0].get("attempts") or []
            attempts1 = rows[1].get("attempts") or []
            attempts2 = rows[2].get("attempts") or []
            interrupted = attempts1[0] if attempts1 else {}
            cleanup_recovery = interrupted.get("cleanup_recovery") or {}
            detail.update(
                {
                    "test0_attempts": [row.get("status") for row in attempts0],
                    "test1_attempts": [row.get("status") for row in attempts1],
                    "test2_attempts": [row.get("status") for row in attempts2],
                    "test1_first_interrupted": bool(
                        interrupted.get("interrupted")
                    ),
                    "test1_interrupted_status": interrupted.get("status"),
                    "test1_interrupted_cleanup_verified": interrupted.get(
                        "cleanup_verified"
                    ),
                    "test1_cleanup_recovery_container": cleanup_recovery.get(
                        "container_name"
                    ),
                    "test1_cleanup_recovery_recovered_utc": cleanup_recovery.get(
                        "recovered_utc"
                    ),
                    "verified": entry.get("verified"),
                    "coverage_error": entry.get("coverage_error"),
                }
            )
            # The interrupted first attempt keeps its infrastructure outcome
            # and its own record stays unverified; the separate container-bound
            # cleanup-recovered record lets the conclusive retry be accepted,
            # no attempt is regraded and the retry allowance is not exceeded.
            ok = (
                len(attempts0) == 1
                and len(attempts1) == 2
                and bool(interrupted.get("interrupted"))
                and interrupted.get("status") == "infrastructure_error"
                and interrupted.get("cleanup_verified") is False
                and bool(cleanup_recovery)
                and len(attempts2) == 1
                and entry.get("verified") is True
                and not entry.get("coverage_error")
            )
        return _case(
            "ledger_crash_cleanup_recovery_real_docker",
            ok=ok,
            classification="real Docker per-test ledger crash cleanup recovery",
            detail=detail,
        )


def run_synthetic_validation(
    *,
    image: str,
    expected_digest: str | None,
    out_path: Path | str,
    runner: Callable[..., Any] = subprocess.run,
    timeout: float = OFFICIAL_TIMEOUT_S,
    skip_docker: bool = False,
) -> dict[str, Any]:
    """Run every synthetic case and write the non-official receipt."""

    out_path = Path(out_path)
    local = verify_local_digest(
        image, expected_digest=expected_digest, runner=runner
    )
    sandbox = LocalOnlySandbox(
        image=image, expected_digest=expected_digest
    )
    cases: list[dict[str, Any]] = []

    # Digest mismatch must be refused before any candidate runs.
    try:
        verify_local_digest(
            image,
            expected_digest="sha256:" + "0" * 64,
            runner=runner,
        )
    except SyntheticValidationError:
        cases.append(
            _case(
                "digest_mismatch_refused",
                ok=True,
                classification="refused",
                detail={"image": image, "expected_digest": "sha256:" + "0" * 64},
            )
        )
    else:  # pragma: no cover - defensive
        cases.append(
            _case(
                "digest_mismatch_refused",
                ok=False,
                classification="accepted",
            )
        )

    # Marker-only/classifier cases are parameterised and need no camera.
    for marker_case in MARKER_CASES:
        classification, error = classify_marker_case(marker_case)
        cases.append(
            _case(
                str(marker_case["name"]),
                ok=_class_is_infrastructure(classification),
                classification=classification,
                detail={"marker_error": error, "mode": marker_case["mode"]},
            )
        )

    # Deterministic late-receipt case: the execution entry *was* received but
    # the host receipt stamp is after its deadline.  It must be rejected as
    # infrastructure/unknown, never accepted as a conclusive result.
    cases.append(late_receipt_deterministic_case())

    if skip_docker:
        for case in cases:
            case["validation_source"] = "fixture_only"
        receipt = _write_receipt(out_path, image=image, local=local, cases=cases)
        return receipt

    def basic_case(
        name: str, code: str, test: Mapping[str, Any], expect: str
    ) -> tuple[dict[str, Any], str | None]:
        result = sandbox.run_tests(code, [dict(test)], timeout=timeout)
        row = _result_case(name, result, expect=expect)
        return row, result.tests[0].container_name

    pass_case, pass_container = basic_case(
        "known_pass",
        "print(1)\n",
        {"input": "", "output": "1", "type": "stdin"},
        "passed",
    )
    cases.append(pass_case)
    wrong_case, _ = basic_case(
        "wrong_answer",
        "print(2)\n",
        {"input": "", "output": "1", "type": "stdin"},
        "wrong_answer",
    )
    cases.append(wrong_case)
    error_case, _ = basic_case(
        "candidate_exception",
        "raise ValueError('boom')\n",
        {"input": "", "output": "", "type": "stdin"},
        "candidate_error",
    )
    cases.append(error_case)
    deadline_case, _ = basic_case(
        "deadline_sleep",
        "import time\ntime.sleep(30)\n",
        {"input": "", "output": "1", "type": "stdin"},
        "timeout",
    )
    cases.append(deadline_case)
    spam = "import sys\nsys.stdout.write('x' * 1000000 + '\\n')\nsys.stderr.write('y' * 1000000 + '\\n')\n"
    spam_case, _ = basic_case(
        "stdout_stderr_spam",
        spam,
        {"input": "", "output": "spam", "type": "stdin"},
        "wrong_answer",
    )
    cases.append(spam_case)

    attack_code = r"""
import json, os
observed = []
for path in (
    "/proc/1/environ",
    "/proc/1/cmdline",
    "/proc/1/mem",
    "/evidence/budget.json",
    "/evidence/lifecycle.jsonl",
):
    try:
        with open(path, "rb") as handle:
            observed.append(["read", path, handle.read(120).decode("utf-8", "replace")])
    except Exception as exc:
        observed.append(["read-failed", path, type(exc).__name__])
try:
    observed.append(["list-parent-fds", sorted(os.listdir("/proc/1/fd"))[:20]])
except Exception as exc:
    observed.append(["parent-fds-failed", type(exc).__name__])
for fd in range(3, 12):
    try:
        os.fstat(fd)
        observed.append(["open-fd", fd])
    except OSError:
        pass
try:
    with open("/evidence/lifecycle.jsonl", "a", encoding="utf-8") as handle:
        handle.write("deadbeef GML1 {\"event\":\"candidate_finish\"}\n")
    observed.append(["forged-append", "ok"])
except Exception as exc:
    observed.append(["forge-failed", type(exc).__name__])
try:
    os.unlink("/evidence/lifecycle.jsonl")
    observed.append(["deleted-evidence", "ok"])
except Exception as exc:
    observed.append(["delete-failed", type(exc).__name__])
print(json.dumps(observed))
"""
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_attack_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, attack_code)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, stdout, stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        key_leaked = bool(key and key in stdout)
        cases.append(
            _case(
                "evidence_key_parent_mount_attack",
                ok=(
                    _class_is_infrastructure(classification)
                    and not key_leaked
                ),
                classification=classification,
                detail={
                    "key_leaked": key_leaked,
                    "lifecycle_valid": lifecycle.get("lifecycle_valid"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                    "stdout_head": stdout[:4000],
                },
            )
        )

    # Injected startup failure: the wrapper itself never starts.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_startup_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")

        def replace_wrapper(command: list[str]) -> list[str]:
            return [
                value.replace(
                    f"/work/{grader.LIFECYCLE_WRAPPER_NAME}",
                    "/work/missing_wrapper.py",
                )
                for value in command
            ]

        raw, container, key = _run_raw(
            sandbox,
            workdir,
            command_transform=replace_wrapper,
            timeout=timeout,
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "injected_startup_failure",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Post-completion delay: candidate finishes, then the wrapper stalls past
    # the unchanged outer deadline.  This is infrastructure, not a candidate
    # deadline failure.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_post_") as tmp:
        workdir = Path(tmp)
        # Candidate completes and the finish record is written; the wrapper
        # then stalls past the unchanged outer deadline.  This is a
        # post-completion/teardown delay, not a candidate deadline failure.
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "        code = returncode if returncode is not None else 121\n",
            "        time.sleep(30)\n        code = returncode if returncode is not None else 121\n",
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "real_post_completion_delay",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Failed handoff: the bootstrap entry succeeds but exec fails.  The
    # private pipe reports handoff_failed, so this is infrastructure/unknown.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_handoff_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            '                sys.executable,\n                [sys.executable,',
            '                "/work/missing_python",\n                ["/work/missing_python",',
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "failed_exec_handoff",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={
                    "container_terminated": container_is_gone(container),
                    "record_error": lifecycle.get("record_error"),
                },
            )
        )

    # Delayed handoff: entry is written but exec is stalled past the deadline;
    # there is no successful-handoff evidence, so it stays unknown.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_handoff_delay_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "            os.execv(\n",
            "            time.sleep(30)\n            os.execv(\n",
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "delayed_handoff",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Delayed exit observation: the candidate exits before the deadline but
    # the wrapper delays the finish record across it.  A delayed finish is not
    # liveness evidence and the exit interval spans the deadline -> unknown.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_delayed_exit_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "    returncode = None\n    signal_number = None\n",
            "    time.sleep(30)\n    returncode = None\n    signal_number = None\n",
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "delayed_exit_observation",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Missing markers: suppress the evidence channel for a real container.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_missing_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")
        raw, container, key = _run_raw(
            sandbox, workdir, with_evidence=False, timeout=timeout
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "real_missing_markers",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Altered markers: tamper with the evidence file after a real candidate.
    def tampering_reader(
        evidence_dir: Path | str, key: str, attempt: str | None = None
    ) -> Any:
        path = Path(evidence_dir) / grader.LIFECYCLE_EVIDENCE_FILE
        if path.is_file():
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    'deadbeef GML1 {"event":"candidate_finish"}\n'
                )
        return grader.read_lifecycle_markers(evidence_dir, key, attempt)

    def replaying_reader(
        evidence_dir: Path | str, key: str, attempt: str | None = None
    ) -> Any:
        path = Path(evidence_dir) / grader.LIFECYCLE_EVIDENCE_FILE
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            if lines:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(lines[0])
        return grader.read_lifecycle_markers(evidence_dir, key, attempt)

    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_altered_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")
        raw, container, key = _run_raw(
            sandbox,
            workdir,
            marker_reader=tampering_reader,
            timeout=timeout,
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "real_altered_markers",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Replayed authenticated bytes (duplicate sequence) are rejected.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_replay_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")
        raw, container, key = _run_raw(
            sandbox,
            workdir,
            marker_reader=replaying_reader,
            timeout=timeout,
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "replayed_evidence",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Failed cleanup: the container is removed for hygiene, but cleanup reports
    # failure and the outcome is infrastructure.
    def failing_cleanup(container_name: str) -> bool:
        grader._remove_container(container_name)
        return False

    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_cleanup_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")
        raw, container, key = _run_raw(
            sandbox,
            workdir,
            cleanup=failing_cleanup,
            timeout=timeout,
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "failed_cleanup",
                lifecycle,
                expect="cleanup_failure",
                code_rc=code_rc,
                payload={"container_terminated": container_is_gone(container)},
            )
        )

    # Deadline-notice race: a child that completes before the registered host
    # deadline must remain a normal completion, never a conclusive timeout.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_before_deadline_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "import time\ntime.sleep(1.5)\nprint(1)\n")
        raw, container, key = _run_raw(
            sandbox, workdir, timeout=3.0
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        cases.append(
            _case(
                "deadline_exit_before_notice",
                ok=(
                    classification == grader.LIFECYCLE_COMPLETED_BEFORE
                    and code_rc == 0
                    and lifecycle.get("deadline_notice", {}).get("sent") is not True
                ),
                classification=classification,
                detail={
                    "timeout_s": 3.0,
                    "lifecycle_valid": lifecycle.get("lifecycle_valid"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                },
            )
        )

    # A child still running when the authenticated notice arrives is a
    # registered outer-deadline failure with verified termination.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_after_deadline_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "import time\ntime.sleep(5.0)\nprint(1)\n")
        raw, container, key = _run_raw(
            sandbox, workdir, timeout=3.0
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        cases.append(
            _case(
                "deadline_exit_after_notice",
                ok=(
                    classification in grader.CANDIDATE_DEADLINE_CLASSES
                    and lifecycle.get("deadline_alive") is True
                    and lifecycle.get("termination_verified") is True
                    and lifecycle.get("deadline_notice", {}).get("sent") is True
                ),
                classification=classification,
                detail={
                    "timeout_s": 3.0,
                    "lifecycle_valid": lifecycle.get("lifecycle_valid"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                    "late_completion": lifecycle.get("late_completion"),
                },
            )
        )

    # Enforcement before probing: a deliberately slow/hanging state probe must
    # not delay the forced kill while the candidate keeps running past the
    # deadline.  An independent fast watcher records when the real container
    # actually stops.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_slow_probe_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "import time\ntime.sleep(30.0)\nprint(1)\n")
        container_name = (
            f"gemmaloop-selftest-slowprobe-{os.getpid()}-"
            f"{int(time.time() * 1000) % 100000}"
        )
        watched: dict[str, Any] = {"seen": False, "stopped_at": None}

        def watch_container(name: str) -> None:
            limit = time.perf_counter() + 20.0
            while time.perf_counter() < limit:
                state, error = grader._docker_container_state(name)
                status = (state or {}).get("Status")
                if state is not None and status in {"running", "created"}:
                    watched["seen"] = True
                if watched["seen"]:
                    gone = (
                        state is None
                        and error is not None
                        and "no such" in error.lower()
                    )
                    if gone or status in {"exited", "dead"}:
                        watched["stopped_at"] = time.perf_counter()
                        return
                time.sleep(0.02)

        def slow_state_probe(name: str) -> tuple[dict[str, Any] | None, str | None]:
            # Simulate a hung daemon probe.  The enforcement path must not wait
            # for this before issuing the kill.
            time.sleep(3.0)
            return grader._docker_container_state(name)

        watcher = threading.Thread(
            target=watch_container, args=(container_name,), daemon=True
        )
        enforcement_started = time.perf_counter()
        watcher.start()
        raw, container, key = _run_raw(
            sandbox,
            workdir,
            timeout=2.0,
            state_probe=slow_state_probe,
            container_name=container_name,
        )
        watcher.join(timeout=5.0)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        stopped_at = watched.get("stopped_at")
        enforcement_bound = (
            2.0
            + float(grader.LIFECYCLE_NOTICE_GRACE_S)
            + float(grader.DEADLINE_ENFORCEMENT_EXTRA_S)
            + 0.35  # documented real-Docker scheduling/allocation allowance
        )
        cases.append(
            _case(
                "deadline_slow_probe_enforced_before_observation",
                ok=(
                    watched.get("seen") is True
                    and stopped_at is not None
                    and stopped_at - enforcement_started <= enforcement_bound
                    and container_is_gone(container)
                    and lifecycle.get("late_completion") is True
                ),
                classification=classification,
                detail={
                    "timeout_s": 2.0,
                    "enforcement_bound_s": enforcement_bound,
                    "container_stopped_offset_s": (
                        None
                        if stopped_at is None
                        else stopped_at - enforcement_started
                    ),
                    "container_seen": watched.get("seen"),
                    "late_completion": lifecycle.get("late_completion"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                    "code_rc": code_rc,
                },
            )
        )

    # Enforcement fallback with a real container: the first container-kill
    # primitive is deliberately aimed at a wrong name, so only the alternative
    # primitive can stop the real container.  Both are issued by the host
    # inside the frozen bound, and the alternative stops the container.
    with tempfile.TemporaryDirectory(
        prefix="gemmaloop_selftest_enforcement_fallback_"
    ) as tmp:
        workdir = Path(tmp)
        # Hold the wrapper away from its own deadline notice so the host's
        # enforcement path (not the wrapper's self-kill) is what runs.
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "WRAPPER_STARTED = time.monotonic()",
            "WRAPPER_STARTED = time.monotonic()\ntime.sleep(5.0)",
            1,
        )
        _write_workdir(
            workdir,
            "import time\ntime.sleep(30.0)\nprint(1)\n",
            wrapper_source=source,
        )
        container_name = (
            f"gemmaloop-selftest-fallback-{os.getpid()}-"
            f"{int(time.time() * 1000) % 100000}"
        )
        wrong_name = container_name + "-missing"
        recorded_commands: list[list[str]] = []
        recorded_lock = threading.Lock()
        watched_fallback: dict[str, Any] = {
            "seen": False,
            "stopped_at": None,
        }

        def fallback_watcher(name: str) -> None:
            limit = time.perf_counter() + 20.0
            while time.perf_counter() < limit:
                state, error = grader._docker_container_state(name)
                status = (state or {}).get("Status")
                if state is not None and status in {"running", "created"}:
                    watched_fallback["seen"] = True
                if watched_fallback["seen"]:
                    gone = (
                        state is None
                        and error is not None
                        and "no such" in error.lower()
                    )
                    if gone or status in {"exited", "dead"}:
                        watched_fallback["stopped_at"] = time.perf_counter()
                        return
                time.sleep(0.02)

        def fallback_spawner(command, **kwargs):
            recorded = [str(value) for value in command]
            if len(recorded) > 1 and recorded[1] == "rm":
                # The primary removal primitive deliberately targets a missing
                # name, forcing the concurrently issued alternative kill to
                # stop the real container.  Concurrency means command order is
                # no longer an issuance order.
                recorded = [
                    value.replace(container_name, wrong_name)
                    for value in recorded
                ]
            with recorded_lock:
                recorded_commands.append(list(recorded))
            return subprocess.Popen(recorded, **kwargs)

        watcher = threading.Thread(
            target=fallback_watcher, args=(container_name,), daemon=True
        )
        enforcement_started = time.perf_counter()
        watcher.start()
        raw, container, key = _run_raw(
            sandbox,
            workdir,
            timeout=2.0,
            container_name=container_name,
            enforcement_spawner=fallback_spawner,
        )
        watcher.join(timeout=5.0)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        stopped_at = watched_fallback.get("stopped_at")
        enforcement_bound = (
            2.0
            + float(grader.LIFECYCLE_NOTICE_GRACE_S)
            + float(grader.DEADLINE_ENFORCEMENT_EXTRA_S)
            + 0.35  # documented real-Docker scheduling/allocation allowance
        )
        primary_targeted_wrong_name = any(
            wrong_name in " ".join(command)
            and "rm" in command
            for command in recorded_commands
        )
        alternative_targeted_real_name = any(
            container_name in " ".join(command)
            and "kill" in command
            for command in recorded_commands
        )
        issuance = lifecycle.get("deadline_enforcement") or {}
        cases.append(
            _case(
                "deadline_enforcement_fallback_after_failed_first_primitive",
                ok=(
                    primary_targeted_wrong_name
                    and alternative_targeted_real_name
                    and len(recorded_commands) >= 2
                    and watched_fallback.get("seen") is True
                    and stopped_at is not None
                    and stopped_at - enforcement_started <= enforcement_bound
                    and container_is_gone(container)
                    and lifecycle.get("deadline_enforcement_issued") is True
                    and issuance.get("triggered") is True
                    and issuance.get("overrun") is False
                    and (issuance.get("client_kill") or {}).get("issued")
                    is True
                ),
                classification=classification,
                detail={
                    "timeout_s": 2.0,
                    "enforcement_bound_s": enforcement_bound,
                    "container_stopped_offset_s": (
                        None
                        if stopped_at is None
                        else stopped_at - enforcement_started
                    ),
                    "container_seen": watched_fallback.get("seen"),
                    "primary_targeted_wrong_name": primary_targeted_wrong_name,
                    "alternative_targeted_real_name": (
                        alternative_targeted_real_name
                    ),
                    "recorded_commands": recorded_commands,
                    "enforcement_issued": lifecycle.get(
                        "deadline_enforcement_issued"
                    ),
                    "enforcement_overrun": issuance.get("overrun"),
                    "enforcement_client_kill": issuance.get("client_kill"),
                    "enforcement_primitives": issuance.get("primitives"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                    "code_rc": code_rc,
                },
            )
        )

    # Post-deadline start with a delayed notice: the launcher delays the
    # runner exec past the host's registered deadline and the wrapper delays
    # reading the authenticated notice until after the start frame.  With the
    # frozen <=250 ms enforcement window the host may kill the docker run
    # before it ever observes the late start; either a late receipt or no
    # receipt is infrastructure/unknown and never a registered timeout.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_late_start_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "def main():\n    target = sys.argv[1]\n    runner = sys.argv[2]",
            "def main():\n    time.sleep(3.0)\n    target = sys.argv[1]\n    runner = sys.argv[2]",
        ).replace(
            "        if not notice_verified:\n            notice = read_deadline_notice(key, attempt)\n",
            "        if not notice_verified:\n"
            "            notice = None\n"
            "            if start_emitted:\n"
            "                notice = read_deadline_notice(key, attempt)\n",
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=3.0)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        classification = str(
            lifecycle.get("class") or grader.LIFECYCLE_UNRESOLVED
        )
        cases.append(
            _case(
                "deadline_late_start_delayed_notice",
                ok=(
                    _class_is_infrastructure(classification)
                    and lifecycle.get("start_received_within_budget") is not True
                    and str(lifecycle.get("record_error") or "")
                    in LATE_START_RECORD_ERRORS
                ),
                classification=classification,
                detail={
                    "timeout_s": 3.0,
                    "start_received_within_budget": lifecycle.get(
                        "start_received_within_budget"
                    ),
                    "start_received_offset_s": lifecycle.get(
                        "start_received_offset_s"
                    ),
                    "record_error": lifecycle.get("record_error"),
                    "lifecycle_valid": lifecycle.get("lifecycle_valid"),
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "container_name": lifecycle.get("container_name"),
                    "container_terminated": container_is_gone(container),
                },
            )
        )

    # Exec failure after the entry frame: the bootstrap execs the frozen
    # launcher (positive exec_ok), but the launcher's candidate exec fails and
    # reports handoff_failed.  This must stay infrastructure/unknown.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_exec_after_entry_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            'os.path.join("/work", target)',
            '"/work/missing_candidate.py"',
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "exec_failure_after_entry_frame",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={
                    "handoff_success": lifecycle.get("handoff_success"),
                    "record_error": lifecycle.get("record_error"),
                    "container_terminated": container_is_gone(container),
                },
            )
        )

    # Launcher death after its pre-exec frame but before the runner exec: the
    # exec_ok frame is not entry evidence, no post_exec_entry is ever written,
    # and the pipe closure is indistinguishable from a successful exec.  This
    # window must stay infrastructure/unknown.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_launcher_death_") as tmp:
        workdir = Path(tmp)
        source = grader._LIFECYCLE_WRAPPER_SOURCE.replace(
            "    try:\n        # Only a successful exec of the frozen runner",
            "    try:\n        os._exit(0)\n        # Only a successful exec of the frozen runner",
        )
        _write_workdir(workdir, "print(1)\n", wrapper_source=source)
        raw, container, key = _run_raw(sandbox, workdir, timeout=timeout)
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        cases.append(
            _raw_case(
                "launcher_death_before_runner_exec",
                lifecycle,
                expect="infrastructure",
                code_rc=code_rc,
                payload={
                    "handoff_success": lifecycle.get("handoff_success"),
                    "record_error": lifecycle.get("record_error"),
                    "container_terminated": container_is_gone(container),
                },
            )
        )

    # A real Docker cleanup failure must stay blocking terminal disposition.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_cleanup_block_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")

        def blocking_cleanup(container_name: str) -> bool:
            grader._remove_container(container_name)
            return False

        raw, _container, key = _run_raw(
            sandbox, workdir, cleanup=blocking_cleanup, timeout=timeout
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, lifecycle = raw
        shortfall_module = __import__(
            "gemmaloop.stage0.shortfall", fromlist=["shortfall"]
        )
        terminal_row = {
            "index": 0,
            "status": "infrastructure_error",
            "passed": False,
            "infrastructure_failure": True,
            "cleanup_verified": lifecycle.get("cleanup_verified"),
            "timed_out": bool(lifecycle.get("class") in grader.CANDIDATE_DEADLINE_CLASSES),
            "termination_verified": lifecycle.get("termination_verified"),
            "error": "cleanup failed",
            "attempts": [
                {"status": "infrastructure_error", "error": "cleanup failed"},
                {"status": "infrastructure_error", "error": "cleanup failed"},
            ],
        }
        shortfall_module._apply_terminal_infrastructure([terminal_row], 1)
        cases.append(
            _case(
                "cleanup_failure_stays_blocking",
                ok=(
                    lifecycle.get("cleanup_verified") is False
                    and terminal_row.get("terminal_infrastructure") is not True
                ),
                classification=str(lifecycle.get("class")),
                detail={
                    "cleanup_verified": lifecycle.get("cleanup_verified"),
                    "terminal_infrastructure": terminal_row.get(
                        "terminal_infrastructure"
                    ),
                },
            )
        )

    # Documented recovery followed by the one permitted retry.
    from gemmaloop.stage0 import shortfall

    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_retry_") as tmp:
        workdir = Path(tmp)
        _write_workdir(workdir, "print(1)\n")
        raw, _container, key = _run_raw(
            sandbox,
            workdir,
            command_transform=lambda command: [
                value.replace(
                    f"/work/{grader.LIFECYCLE_WRAPPER_NAME}",
                    "/work/missing_wrapper.py",
                )
                for value in command
            ],
            timeout=timeout,
        )
        code_rc, _stdout, _stderr, _timed_out, _duration, first_lifecycle = raw
        recovery = shortfall._docker_recovery_check(
            ["startup", "daemon"], image=image, runner=runner
        )
        retry_result = sandbox.run_tests(
            "print(1)\n",
            [{"input": "", "output": "1", "type": "stdin"}],
            timeout=timeout,
        )
        retry_test = retry_result.tests[0]
        ledger_path = Path(
            tempfile.mkdtemp(prefix="gemmaloop_selftest_ledger_")
        ) / "synthetic_attempt_ledger.jsonl"
        if ledger_path.exists():
            ledger_path.unlink()
        identity = {"synthetic": True, "image": image, "timeout_s": timeout}
        shortfall._ensure_attempt_ledger(ledger_path, identity)
        test = {"input": "", "output": "1", "type": "stdin"}
        for attempt, lifecycle, row in (
            (1, first_lifecycle, None),
            (2, retry_test.__dict__, retry_test),
        ):
            shortfall._append_attempt_ledger(
                ledger_path,
                {
                    "kind": shortfall.LEDGER_ATTEMPT_START,
                    "request_id": "synthetic:retry:0",
                    "problem_id": "synthetic:retry",
                    "test_index": 0,
                    "attempt": attempt,
                    "request_sha256": shortfall._sha256_text("print(1)\n"),
                    "test_sha256": shortfall._test_identity_sha256(test),
                    "started_utc": shortfall._utc_now(),
                },
            )
            evidence = {
                "index": 0,
                "status": (
                    "passed"
                    if row is not None and row.passed
                    else "infrastructure_error"
                ),
                "passed": bool(row is not None and row.passed),
                "infrastructure_failure": not bool(
                    row is not None and row.passed
                ),
                "lifecycle_class": lifecycle.get("class"),
                "lifecycle_valid": lifecycle.get("lifecycle_valid"),
                "cleanup_verified": lifecycle.get("cleanup_verified"),
                "error": None if row is not None and row.passed else "recovered",
                "stderr": "",
                "timed_out": False,
                "exit_code": 0 if row is not None and row.passed else None,
                "duration_s": 0.0,
            }
            shortfall._append_attempt_ledger(
                ledger_path,
                shortfall._ledger_result_record(
                    request_id="synthetic:retry:0",
                    problem_id="synthetic:retry",
                    test_index=0,
                    attempt=attempt,
                    evidence=evidence,
                    sandboxed=True,
                    recovery_evidence=recovery if attempt == 2 else None,
                ),
            )
        ledger_rows = shortfall._read_attempt_ledger(ledger_path)
        attempts = [
            row
            for row in ledger_rows
            if row.get("kind") == shortfall.LEDGER_ATTEMPT_RESULT
        ]
        retry_ok = (
            bool(recovery.get("ok"))
            and "noop_container" in (recovery.get("checks") or {})
            and len(attempts) == 2
            and attempts[0].get("status") == "infrastructure_error"
            and attempts[1].get("status") == "passed"
            and (attempts[1].get("recovery_evidence") or {}).get("ok") is True
        )
        cases.append(
            _case(
                "documented_recovery_and_single_retry",
                ok=retry_ok,
                classification=str(first_lifecycle.get("class")),
                detail={
                    "recovery_checks": sorted((recovery.get("checks") or {})),
                    "recovery_ok": bool(recovery.get("ok")),
                    "attempt_statuses": [row.get("status") for row in attempts],
                    "retry_lifecycle": retry_test.lifecycle_class,
                    "ledger": str(ledger_path),
                },
            )
        )

    # Retry exhaustion: a second infrastructure/unknown attempt becomes the
    # terminal infrastructure-unverified disposition; there is no third try.
    infra_row = {
        "index": 0,
        "status": "infrastructure_error",
        "passed": False,
        "infrastructure_failure": True,
        "cleanup_verified": True,
        "timed_out": False,
        "error": "docker daemon unavailable",
        "attempts": [
            {
                "status": "infrastructure_error",
                "error": "first unknown",
                "cleanup_verified": True,
            },
            {
                "status": "infrastructure_error",
                "error": "retry unknown",
                "cleanup_verified": True,
            },
        ],
    }
    terminal_rows = [dict(infra_row)]
    shortfall._apply_terminal_infrastructure(terminal_rows, 1)
    terminal_ok = (
        terminal_rows[0].get("terminal_infrastructure") is True
        and terminal_rows[0].get("terminal_status")
        == shortfall.TEST_STATUS_TERMINAL_INFRASTRUCTURE
        and terminal_rows[0].get("terminal_reason") == "docker daemon unavailable"
        and len(terminal_rows[0]["attempts"]) == 2
    )
    cases.append(
        _case(
            "retry_exhaustion_terminal_disposition",
            ok=terminal_ok,
            classification=shortfall.TEST_STATUS_TERMINAL_INFRASTRUCTURE,
            detail={"row": terminal_rows[0]},
        )
    )

    # Ledger boundaries: a committed attempt-start followed by a torn/crashed
    # write is an interrupted attempt that consumes the retry; the planner
    # never offers a third attempt.
    with tempfile.TemporaryDirectory(prefix="gemmaloop_selftest_ledger_") as tmp:
        ledger = Path(tmp) / "attempt_ledger.jsonl"
        shortfall._ensure_attempt_ledger(ledger, {"synthetic": "crash"})
        test = {"input": "", "output": "1", "type": "stdin"}
        key = ("synthetic:crash:0", 0)
        shortfall._append_attempt_ledger(
            ledger,
            {
                "kind": shortfall.LEDGER_ATTEMPT_START,
                "request_id": key[0],
                "problem_id": "synthetic:crash",
                "test_index": 0,
                "attempt": 1,
                "request_sha256": shortfall._sha256_text("print(1)\n"),
                "test_sha256": shortfall._test_identity_sha256(test),
                "started_utc": shortfall._utc_now(),
            },
        )
        shortfall._append_attempt_ledger(
            ledger,
            shortfall._ledger_result_record(
                request_id=key[0],
                problem_id="synthetic:crash",
                test_index=0,
                attempt=1,
                evidence={
                    "status": "infrastructure_error",
                    "passed": False,
                    "infrastructure_failure": True,
                    "error": "first unknown",
                },
            ),
        )
        shortfall._append_attempt_ledger(
            ledger,
            {
                "kind": shortfall.LEDGER_ATTEMPT_START,
                "request_id": key[0],
                "problem_id": "synthetic:crash",
                "test_index": 0,
                "attempt": 2,
                "request_sha256": shortfall._sha256_text("print(1)\n"),
                "test_sha256": shortfall._test_identity_sha256(test),
                "started_utc": shortfall._utc_now(),
            },
        )
        # Simulate a torn trailing append from a hard crash.
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write('{"kind":"attempt_res')
        records = shortfall._read_attempt_ledger(ledger)
        entries = shortfall._ledger_entries_by_test(records)[key]
        result, next_attempt, in_flight, used = shortfall._ledger_plan(entries)
        resumed = shortfall._read_attempt_ledger(ledger)
        entries_again = shortfall._ledger_entries_by_test(resumed)[key]
        _r2, next_again, _f2, _u2 = shortfall._ledger_plan(entries_again)
        ledger_ok = (
            result is None
            and in_flight is not None
            and int(in_flight.get("attempt") or 1) == 2
            and next_attempt == 3
            and used == 2
            and next_again == 3
            and all(
                int(record.get("attempt") or 1) <= 2
                for record in records
                if record.get("kind")
                in {
                    shortfall.LEDGER_ATTEMPT_START,
                    shortfall.LEDGER_ATTEMPT_RESULT,
                    shortfall.LEDGER_ATTEMPT_INTERRUPTED,
                }
            )
        )
        cases.append(
            _case(
                "ledger_crash_boundary_retry_exhaustion",
                ok=ledger_ok,
                classification="infrastructure",
                detail={
                    "next_attempt": next_attempt,
                    "used": used,
                    "in_flight_attempt": (
                        in_flight.get("attempt") if in_flight else None
                    ),
                    "torn_tail_recovered": len(resumed) == len(records),
                },
            )
        )

    # Real Docker crash midway through a multi-test grade: kill the grading
    # process after the first committed per-test result, then resume.
    cases.append(
        _run_ledger_crash_case(
            sandbox,
            image=image,
            image_id=local.get("id"),
            timeout=min(float(timeout), 4.0),
        )
    )

    # Verify every real container we created was terminated.
    containers = [
        case.get("detail", {}).get("container_name")
        for case in cases
        if case.get("detail", {}).get("container_name")
    ]
    leaked = [name for name in containers if name and not container_is_gone(name)]
    cases.append(
        _case(
            "container_termination",
            ok=not leaked,
            classification="cleanup",
            detail={"containers": containers, "leaked": leaked},
        )
    )
    if pass_container is not None:
        cases.append(
            _case(
                "pass_container_terminated",
                ok=container_is_gone(pass_container),
                classification="cleanup",
                detail={"container": pass_container},
            )
        )

    return _write_receipt(out_path, image=image, local=local, cases=cases)


def _write_receipt(
    out_path: Path,
    *,
    image: str,
    local: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_only = {
        "missing_markers",
        "altered_markers",
        "out_of_order_markers",
        "post_completion_delay",
        "late_receipt_received_after_deadline",
        "retry_exhaustion_terminal_disposition",
        "ledger_crash_boundary_retry_exhaustion",
    }
    annotated = []
    for case in cases:
        item = dict(case)
        item.setdefault(
            "validation_source",
            (
                "fixture_only"
                if item.get("name") in fixture_only
                else "real_docker"
            ),
        )
        annotated.append(item)
    receipt: dict[str, Any] = {
        "kind": RECEIPT_KIND,
        "version": RECEIPT_VERSION,
        "official": False,
        "count_result": False,
        "synthetic_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image": image,
        "image_id": local.get("id"),
        "repo_digests": list(local.get("repo_digests") or []),
        "timeout_s": OFFICIAL_TIMEOUT_S,
        "cases": annotated,
        "real_docker_cases": [
            item["name"]
            for item in annotated
            if item.get("validation_source") == "real_docker"
        ],
        "fixture_only_cases": [
            item["name"]
            for item in annotated
            if item.get("validation_source") == "fixture_only"
        ],
        "ok": all(bool(case.get("ok")) for case in cases),
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    out_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=grader.DEFAULT_DOCKER_IMAGE)
    parser.add_argument(
        "--image-id",
        default=None,
        help="require this exact local image Id (sha256:...) and never pull",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=float, default=OFFICIAL_TIMEOUT_S)
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="only run the local-image and parameterised marker cases",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = run_synthetic_validation(
        image=args.image,
        expected_digest=args.image_id,
        out_path=args.out,
        timeout=float(args.timeout),
        skip_docker=bool(args.skip_docker),
    )
    print(
        json.dumps(
            {
                "ok": receipt["ok"],
                "receipt": str(Path(args.out).resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                "cases": len(receipt["cases"]),
                "official": receipt["official"],
                "count_result": receipt["count_result"],
            },
            indent=2,
        )
    )
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
