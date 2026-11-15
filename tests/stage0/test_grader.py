"""Grader service: CPU-only, no Docker and no network required."""

from __future__ import annotations

from pathlib import Path
import tempfile

import gemmaloop.stage0.grader as grader_module
from gemmaloop.stage0.grader import (
    DockerSandbox,
    GraderService,
    grade_code,
    grade_math,
    normalize_output,
    run_local_tests,
)


def test_stdin_test_passes_and_fails_and_times_out():
    tests = [{"input": "1 2", "output": "3", "type": "stdin"}]
    good = run_local_tests("print(sum(map(int, input().split())))", tests)
    assert good.passed and good.n_tests == 1 and not good.sandboxed

    bad = run_local_tests("print(4)", tests)
    assert not bad.passed and bad.passed_tests == 0

    slow = run_local_tests("while True:\n    pass\n", tests, timeout=0.5)
    assert not slow.passed and slow.timed_out
    assert slow.tests[0].timed_out


def test_functional_test_uses_fn_name():
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        {"input": "[1, 2]", "output": "3", "type": "functional", "fn_name": "add"}
    ]
    result = run_local_tests(code, tests)
    assert result.passed


def test_output_normalisation_ignores_trailing_space():
    assert normalize_output("3  \n") == "3"
    assert normalize_output("a\nb \n\n") == "a\nb"


def test_math_grading_uses_aime_normalisation():
    assert grade_math("The answer is \\boxed{1,234}.", "1234").correct
    assert grade_math("\\boxed{\\text{70}}", "70").correct
    assert not grade_math("\\boxed{71}", "70").correct
    assert not grade_math("no box here", "70").correct
    assert grade_math("\\boxed{5}", "5").method in {"aime-normalise", "math-verify"}


def test_docker_command_has_required_isolation_flags(tmp_path: Path):
    sandbox = DockerSandbox(image="python:3.12-slim", memory_mb=256, pids=32, cpus=0.5)
    command = sandbox.command(tmp_path)
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "/tmp:rw,noexec,nosuid" in joined
    assert "--memory 256m" in joined
    assert "--pids-limit 32" in joined
    assert "--user 65534:65534" in joined
    assert f"{tmp_path.resolve()}:/work:ro" in joined
    assert command[-3:] == [
        f"/work/{grader_module.LIFECYCLE_WRAPPER_NAME}",
        "candidate.py",
        "",
    ]


def test_docker_unavailable_falls_back_and_is_flagged(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "available", lambda self: False)
    result = grade_code(
        "print(1)",
        [{"input": "", "output": "1"}],
        sandbox="docker",
        allow_fallback=True,
    )
    assert result.passed
    assert result.sandboxed is False
    assert result.sandbox == "local-fallback"
    assert "NON-SANDBOXED" in (result.error or "")

    try:
        grade_code(
            "print(1)",
            [{"input": "", "output": "1"}],
            sandbox="docker",
            allow_fallback=False,
        )
    except RuntimeError as exc:
        assert "Docker" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_persistent_service_grades_code_and_math():
    with GraderService(workers=1, timeout=2, sandbox="local", allow_fallback=True) as service:
        code = service.grade_code(
            "print(7)", [{"input": "", "output": "7", "type": "stdin"}]
        )
        assert code.passed and code.n_tests == 1
        math = service.grade_math("\\boxed{42}", "42")
        assert math.correct
        batch = service.grade_math_batch([("\\boxed{1}", "1"), ("\\boxed{2}", "3")])
        assert [item.correct for item in batch] == [True, False]


def test_docker_is_required_by_default(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "available", lambda self: False)
    try:
        grade_code("print(1)", [{"input": "", "output": "1"}])
    except RuntimeError as exc:
        assert "Docker" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected Docker to be required")


def test_local_grader_scrubs_secrets(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-do-not-leak")
    result = run_local_tests(
        "import os\nprint(os.environ.get('DEEPSEEK_API_KEY', 'clean'))",
        [{"input": "", "output": "clean", "type": "stdin"}],
    )
    assert result.passed, result.tests[0].stderr


def test_docker_per_test_timeout_is_six_seconds(monkeypatch, tmp_path):
    import gemmaloop.stage0.grader as grader_module

    sandbox = DockerSandbox()
    monkeypatch.setattr(sandbox, "ensure_image", lambda: None)
    captured = {}

    def fake_run(command, *, input_text, timeout, container_name, **kwargs):
        captured["timeout"] = timeout
        captured["container_name"] = container_name
        return (
            0,
            "1",
            "",
            False,
            0.01,
            {
                "class": grader_module.LIFECYCLE_COMPLETED_BEFORE,
                "container_name": container_name,
                "container_started_at": "2026-11-06T00:00:00+00:00",
                "container_finished_at": "2026-11-06T00:00:01+00:00",
                "events": [],
                "lifecycle_valid": True,
                "cleanup_verified": True,
                "handoff_success": True,
                "termination_verified": True,
                "deadline_alive": False,
                "exit_observed": True,
            },
        )

    monkeypatch.setattr(grader_module, "_run_process_docker", fake_run)
    result = sandbox.run_tests("print(1)", [{"input": "", "output": "1"}], timeout=6)
    assert result.passed
    # The registered outer timeout is unchanged; the candidate never receives a
    # fresh clock after container startup.
    assert captured["timeout"] == 6
    assert result.tests[0].lifecycle_class == grader_module.LIFECYCLE_COMPLETED_BEFORE
    assert result.tests[0].cleanup_verified is True


def test_late_or_running_lifecycle_evidence_never_accepts_output(
    monkeypatch, tmp_path: Path
):
    import gemmaloop.stage0.grader as grader_module

    sandbox = DockerSandbox()
    monkeypatch.setattr(sandbox, "ensure_image", lambda: None)

    def fake_run(command, *, input_text, timeout, container_name, **kwargs):
        return (
            0,
            "1",
            "",
            False,
            0.01,
            {
                "class": grader_module.LIFECYCLE_CANDIDATE_EXITED_AFTER,
                "container_name": container_name,
                "cleanup_verified": True,
                "late_completion": True,
                "lifecycle_valid": True,
                "markers": [],
            },
        )

    monkeypatch.setattr(grader_module, "_run_process_docker", fake_run)
    result = sandbox.run_tests("print(1)", [{"input": "", "output": "1"}], timeout=6)
    assert result.timed_out is True
    assert result.passed is False
    assert result.tests[0].passed is False


def test_lifecycle_classes_distinguish_startup_candidate_and_after_completion():
    import gemmaloop.stage0.grader as grader_module

    deadline = 100.0

    def markers(start, finish=None):
        result = [
            {
                "event": grader_module.LIFECYCLE_EVENT_START,
                "epoch": start,
            }
        ]
        if finish is not None:
            result.append(
                {
                    "event": grader_module.LIFECYCLE_EVENT_FINISH,
                    "epoch": finish,
                }
            )
        return result

    running = grader_module.classify_lifecycle_evidence(
        markers=markers(90.0),
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        status="running",
        deadline_epoch=deadline,
    )
    assert running == grader_module.LIFECYCLE_CANDIDATE_RUNNING

    # Container state alone (running) is not candidate execution.
    container_only = grader_module.classify_lifecycle_evidence(
        markers=[],
        marker_error="no lifecycle evidence",
        timed_out=True,
        cleanup_verified=True,
        status="running",
        deadline_epoch=deadline,
    )
    assert container_only == grader_module.LIFECYCLE_UNRESOLVED

    startup = grader_module.classify_lifecycle_evidence(
        markers=[],
        marker_error="no lifecycle evidence",
        timed_out=False,
        cleanup_verified=True,
        status="created",
        deadline_epoch=deadline,
    )
    assert startup == grader_module.LIFECYCLE_STARTUP_FAILURE

    late_start = grader_module.classify_lifecycle_evidence(
        markers=markers(120.0),
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        status="running",
        deadline_epoch=deadline,
    )
    assert late_start == grader_module.LIFECYCLE_STARTUP_FAILURE

    completed = grader_module.classify_lifecycle_evidence(
        markers=markers(90.0, 95.0),
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        status="exited",
        deadline_epoch=deadline,
    )
    assert completed == grader_module.LIFECYCLE_COMPLETED_BEFORE

    after_deadline = grader_module.classify_lifecycle_evidence(
        markers=markers(90.0, 110.0),
        marker_error=None,
        timed_out=True,
        cleanup_verified=True,
        status="exited",
        deadline_epoch=deadline,
    )
    assert after_deadline == grader_module.LIFECYCLE_CANDIDATE_EXITED_AFTER

    unresolved = grader_module.classify_lifecycle_evidence(
        markers=[],
        marker_error="daemon unreachable",
        timed_out=True,
        cleanup_verified=True,
        status=None,
        deadline_epoch=deadline,
        probe_error="daemon unreachable",
    )
    assert unresolved == grader_module.LIFECYCLE_UNRESOLVED

    dirty = grader_module.classify_lifecycle_evidence(
        markers=markers(90.0),
        marker_error=None,
        timed_out=True,
        cleanup_verified=False,
        status="running",
        deadline_epoch=deadline,
    )
    assert dirty == grader_module.LIFECYCLE_UNRESOLVED


def test_lifecycle_records_require_hmac_framing_and_fresh_attempt():
    import gemmaloop.stage0.grader as grader_module
    import hashlib
    import hmac

    with tempfile.TemporaryDirectory() as tmp:
        evidence = Path(tmp)
        key = "a" * 64
        path = evidence / grader_module.LIFECYCLE_EVIDENCE_FILE
        path.write_text("deadbeef GML1 {}\n", encoding="utf-8")
        markers, error = grader_module.read_lifecycle_markers(evidence, key)
        assert markers == []
        assert error and "HMAC" in error
        assert grader_module.classify_lifecycle_evidence(
            markers=markers,
            marker_error=error,
            timed_out=True,
            cleanup_verified=True,
            status="running",
            deadline_epoch=100.0,
        ) == grader_module.LIFECYCLE_UNRESOLVED

        # A record authenticated for another attempt may not be replayed.
        body = (
            '{"attempt":"old","event":"bootstrap_ready","monotonic":1.0,'
            '"pid":1,"seq":1,"v":1,"wall":1.0}'
        )
        mac = hmac.new(
            key.encode(), ("GML1 " + body).encode(), hashlib.sha256
        ).hexdigest()
        path.write_text(mac + " GML1 " + body + "\n", encoding="utf-8")
        markers, error = grader_module.read_lifecycle_markers(
            evidence, key, "new-attempt"
        )
        assert markers == []
        assert error and "different attempt" in error

        # A torn trailing record is rejected rather than half-trusted.
        path.write_text(mac + " GML1 " + body, encoding="utf-8")
        markers, error = grader_module.read_lifecycle_markers(evidence, key)
        assert markers == []
        assert error and "torn" in error


def _boxed(value: str) -> str:
    return chr(92) + "boxed{" + value + "}"


def test_math_verify_or_normaliser_keeps_the_1_234_case(monkeypatch):
    import gemmaloop.stage0.grader as grader_module

    # math-verify's own comparator disagrees with the registered AIME
    # normalisation on thousands separators; either verifier saying correct wins.
    monkeypatch.setattr(
        grader_module, "_math_verify_grade", lambda *args, **kwargs: (False, False)
    )
    result = grade_math(_boxed("1,234"), "1234")
    assert result.correct
    assert result.decided_by == "normaliser"
    assert result.to_dict()["decided_by"] == "normaliser"


def test_math_verify_decides_fraction_vs_decimal():
    result = grade_math(_boxed("3/4"), "0.75")
    assert result.correct
    assert result.decided_by in {"math_verify", "both"}
    assert result.method == "math-verify"


def test_math_verify_timeout_falls_back_to_the_normaliser(monkeypatch):
    import subprocess

    import gemmaloop.stage0.grader as grader_module

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 10))

    monkeypatch.setattr(grader_module.subprocess, "run", hang)
    result = grade_math(_boxed("5"), "5")
    assert result.correct
    assert result.decided_by == "timeout"
    assert result.timed_out is True


def test_decided_by_reports_both_verifiers(monkeypatch):
    import gemmaloop.stage0.grader as grader_module

    monkeypatch.setattr(
        grader_module, "_math_verify_grade", lambda *args, **kwargs: (True, False)
    )
    result = grade_math(_boxed("5"), "5")
    assert result.correct
    assert result.decided_by == "both"


def test_decided_by_reports_math_verify_only(monkeypatch):
    import gemmaloop.stage0.grader as grader_module

    monkeypatch.setattr(
        grader_module, "_math_verify_grade", lambda *args, **kwargs: (True, False)
    )
    result = grade_math(_boxed("3/4"), "0.75")
    assert result.correct
    assert result.decided_by == "math_verify"


def test_math_verify_outer_timeout_is_real():
    # A deliberately tiny outer limit expires while the grading subprocess is
    # importing math-verify; the normaliser still decides.
    result = grade_math(_boxed("5"), "5", math_verify_timeout=0.001)
    assert result.correct
    assert result.decided_by == "timeout"
    assert result.timed_out is True
