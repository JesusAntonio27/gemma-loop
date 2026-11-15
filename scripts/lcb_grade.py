"""Grade LiveCodeBench samples with the official metric implementation.

Run inside the eval-only ``.venv-lcb`` environment::

    python scripts/lcb_grade.py samples.json grades.json release_v6 [num_process] [timeout]

``samples.json`` maps ``{question_id: [code, ...]}`` for the sampled problems
only.  Every problem must have the same number of samples (the official
``codegen_metrics`` requires it).  The output is
``{"grades": {question_id: [bool, ...]}, "metadata": {question_id: [dict, ...]}}``
in the same sample order; the metadata comes from LCB's ``run_test`` and keeps
its ``error_code`` (for example ``-3`` = Time Limit Exceeded, ``-2`` = Wrong
Answer).

The default per-test ``timeout`` is 12 s.  LCB's harness default is 6 s;
reference 04 recommends 12 s to reduce false negatives from slow tests.

This deliberately does not use ``custom_evaluator.py``: that script expects one
entry per problem in the release and its zip of the full benchmark with the
shorter graded list misattributes grades when only a subset was sampled.

Windows note: ``signal.alarm``/``setitimer`` are emulated with a
``threading.Timer`` that raises LiveCodeBench's ``TimeoutException`` in the
executing thread via ``PyThreadState_SetAsyncExc``.  A candidate blocked inside
a C call that never returns to the bytecode interpreter can ignore the async
exception; it is still stopped by the global multiprocessing watchdog.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

# LiveCodeBench's test runner is POSIX-oriented: it installs a SIGALRM handler
# and calls signal.alarm()/setitimer().  Windows has none of those, so emulate
# them: alarm(t) starts a threading.Timer that injects LCB's own
# TimeoutException into the thread executing the candidate code via
# PyThreadState_SetAsyncExc; alarm(0) cancels the timer.  The outer multiprocess
# watchdog in compute_code_generation_metrics remains as a backstop.
if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import ctypes
    import threading

    _alarm_lock = threading.Lock()
    _alarm_timer = None
    _alarm_generation = 0

    def _raise_lcb_timeout(thread_id: int, generation: int) -> None:
        with _alarm_lock:
            if generation != _alarm_generation:
                return
        try:
            from lcb_runner.evaluation.testing_util import TimeoutException
        except Exception:
            TimeoutException = TimeoutError  # type: ignore[assignment,misc]
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id), ctypes.py_object(TimeoutException)
        )

    def _alarm(seconds: float) -> int:
        global _alarm_timer, _alarm_generation
        with _alarm_lock:
            _alarm_generation += 1
            generation = _alarm_generation
            if _alarm_timer is not None:
                _alarm_timer.cancel()
                _alarm_timer = None
            if seconds and seconds > 0:
                timer = threading.Timer(
                    float(seconds),
                    _raise_lcb_timeout,
                    args=(threading.get_ident(), generation),
                )
                timer.daemon = True
                _alarm_timer = timer
                timer.start()
        return 0

    def _setitimer(which: int, seconds: float, interval: float = 0.0):
        _alarm(seconds)
        return (0.0, 0.0)

    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = getattr(signal, "SIGBREAK", signal.SIGINT)
    if os.environ.get("GEMMALOOP_DISABLE_LCB_ALARM") == "1":
        # Test-only escape hatch for the negative check in the report: shows
        # that the real-grader TLE test fails with the old no-op behaviour.
        signal.alarm = lambda seconds: 0
        signal.setitimer = lambda which, seconds, interval=0.0: (0.0, 0.0)
    else:
        signal.alarm = _alarm
        signal.setitimer = _setitimer

from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: lcb_grade.py <samples.json> <grades.json> <release> "
            "[num_process] [timeout]",
            file=sys.stderr,
        )
        return 2

    samples_path = Path(sys.argv[1])
    grades_path = Path(sys.argv[2])
    release = sys.argv[3]
    num_process_evaluate = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 12

    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("samples.json must map question_id -> [code, ...]")

    question_ids = list(payload)
    if not question_ids:
        grades_path.write_text('{"grades": {}, "metadata": {}}', encoding="utf-8")
        return 0

    sample_counts = {len(payload[question_id]) for question_id in question_ids}
    if len(sample_counts) != 1:
        raise ValueError(
            f"all sampled problems need the same number of samples, got {sample_counts}"
        )
    if sample_counts == {0}:
        raise ValueError("sampled problems need at least one code completion")

    benchmark = {
        problem.question_id: problem
        for problem in load_code_generation_dataset(release)
    }
    missing = [question_id for question_id in question_ids if question_id not in benchmark]
    if missing:
        raise KeyError(f"question_id(s) not in {release}: {missing[:5]}")

    samples = [benchmark[question_id].get_evaluation_sample() for question_id in question_ids]
    generations = [list(payload[question_id]) for question_id in question_ids]

    metrics = codegen_metrics(
        samples,
        generations,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    results = metrics[1]
    raw_metadata = metrics[2]

    grades: dict[str, list[bool]] = {}
    metadata: dict[str, list[dict]] = {}
    for position, question_id in enumerate(question_ids):
        per_sample = results.get(position, [])
        if len(per_sample) != len(generations[position]):
            raise RuntimeError(
                f"grader returned {len(per_sample)} results for "
                f"{len(generations[position])} samples of {question_id}"
            )
        grades[question_id] = [
            bool(all(test_result > 0 for test_result in sample_results))
            for sample_results in per_sample
        ]
        metadata[question_id] = [json.loads(entry) for entry in raw_metadata[position]]

    grades_path.write_text(
        json.dumps({"grades": grades, "metadata": metadata}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
