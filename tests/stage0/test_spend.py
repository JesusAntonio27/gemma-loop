"""Persistent Modal spend ledger: atomic, cumulative and crash-safe."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gemmaloop.stage0.spend import SpendLedger, month_key


def _now(month: int, day: int = 5):
    return lambda: datetime(2026, month, day, 12, 0, tzinfo=timezone.utc)


def test_reservation_then_actual_replaces_it(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", now_fn=_now(11))
    assert ledger.cumulative_usd() == 0.0
    reservation = ledger.reserve(shard=0, usd=8.5, gpu="A100-80GB", timeout_s=7380)
    assert ledger.cumulative_usd() == 8.5
    ledger.settle(reservation, shard=0, actual_usd=2.25)
    assert ledger.cumulative_usd() == 2.25


def test_outstanding_reservation_counts_after_crash(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", now_fn=_now(11))
    ledger.reserve(shard=0, usd=8.5)
    # New object = new process; the reservation still counts.
    reopened = SpendLedger(tmp_path / "spend.json", now_fn=_now(11))
    assert reopened.cumulative_usd() == 8.5


def test_months_are_independent_and_file_is_atomic(tmp_path):
    path = tmp_path / "spend.json"
    november = SpendLedger(path, now_fn=_now(11))
    november.reserve(shard=0, usd=5.0)
    assert november.cumulative_usd("2026-11") == 5.0
    assert november.cumulative_usd("2026-12") == 0.0

    december = SpendLedger(path, now_fn=_now(12))
    december.reserve(shard=0, usd=3.0)
    assert december.cumulative_usd("2026-11") == 5.0
    assert december.cumulative_usd("2026-12") == 3.0
    assert not path.with_name(path.name + ".tmp").exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["entries"]) == 2
    assert all("timestamp" in entry for entry in payload["entries"])


def test_month_key_utc():
    assert month_key(datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)) == "2026-11"
    assert month_key(datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)) == "2026-12"


def test_default_ledger_and_lock_paths_are_under_artifacts_dir():
    from gemmaloop.paths import artifacts_dir
    from gemmaloop.stage0.spend import SPEND_LOCK_NAME, default_ledger_path, spend_lock_path

    assert default_ledger_path() == artifacts_dir() / "stage0" / "modal_spend_ledger.json"
    assert spend_lock_path() == artifacts_dir() / "stage0" / SPEND_LOCK_NAME


def _reserve_in_child(ledger_path: str, lock_path: str, ready, release):
    from gemmaloop.stage0.spend import SpendLedger, acquire_spend_lock

    lock = acquire_spend_lock(lock_path)
    SpendLedger(ledger_path).reserve(shard=1, usd=2.0)
    ready.set()
    release.wait(30)
    lock.release()


def _hold_spend_lock(path: str, ready, release):
    from gemmaloop.stage0.spend import acquire_spend_lock

    lock = acquire_spend_lock(path)
    ready.set()
    release.wait(30)
    lock.release()


def test_second_process_cannot_hold_the_spend_lock(tmp_path):
    import multiprocessing
    import sys
    import subprocess

    from gemmaloop.stage0.spend import acquire_spend_lock, try_acquire_spend_lock

    path = tmp_path / "spend.lock"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    process = ctx.Process(target=_hold_spend_lock, args=(str(path), ready, release))
    process.start()
    try:
        assert ready.wait(30)
        assert try_acquire_spend_lock(path) is None

        code = (
            "import sys\n"
            "sys.path.insert(0, r'{}')\n"
            "from gemmaloop.stage0.spend import require_spend_lock\n"
            "require_spend_lock(sys.argv[1])\n"
        ).format(str(Path(__file__).resolve().parents[2] / "src"))
        completed = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 75
        assert "another money-spending process" in completed.stderr
    finally:
        release.set()
        process.join(30)
        assert process.exitcode == 0

    held = acquire_spend_lock(path)
    held.release()


def test_ledger_reservations_from_two_processes_are_not_lost(tmp_path):
    import multiprocessing

    from gemmaloop.stage0.spend import SpendLedger

    path = tmp_path / "spend.json"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()

    process = ctx.Process(
        target=_reserve_in_child,
        args=(str(path), str(tmp_path / "spend.lock"), ready, release),
    )
    process.start()
    try:
        assert ready.wait(30)
        SpendLedger(path).reserve(shard=2, usd=3.0)
    finally:
        release.set()
        process.join(30)
        assert process.exitcode == 0
    entries = SpendLedger(path).load()["entries"]
    assert [entry["shard"] for entry in entries] == [1, 2]
    assert SpendLedger(path).cumulative_usd() == 5.0

def test_concurrent_appends_hold_the_ledger_lock(tmp_path):
    """Two writers must not load-modify-write from the same stale snapshot."""

    import threading
    import time

    class SlowLedger(SpendLedger):
        def _write(self, payload):
            time.sleep(0.05)
            super()._write(payload)

    path = tmp_path / "spend.json"
    first = SlowLedger(path, now_fn=_now(11))
    second = SlowLedger(path, now_fn=_now(11))
    threads = [
        threading.Thread(target=first.reserve, kwargs={"shard": 0, "usd": 0.5}),
        threading.Thread(target=second.reserve, kwargs={"shard": 1, "usd": 0.5}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert [thread.is_alive() for thread in threads] == [False, False]
    entries = SpendLedger(path).load()["entries"]
    assert len(entries) == 2
    assert SpendLedger(path).cumulative_usd() == 1.0
