"""Throughput/RAM guards and time-based saves with fake clocks (Task 14b)."""

from __future__ import annotations

from types import SimpleNamespace

from gemmaloop.train.guards import (
    GUARD_EXIT_CODE,
    RamGuard,
    ThroughputGuard,
    ThroughputGuardConfig,
    TimeSaveCallback,
    WatchdogCallback,
)


def _throughput(**overrides) -> ThroughputGuard:
    config = ThroughputGuardConfig(
        window_steps=2,
        baseline_start_step=1,
        baseline_end_step=4,
        drop_ratio=0.5,
        consecutive_windows=3,
        **overrides,
    )
    return ThroughputGuard(config)


def _control():
    return SimpleNamespace(should_save=False, should_training_stop=False)


def test_stable_throughput_never_fires():
    guard = _throughput()
    for step in range(1, 20):
        decision = guard.observe(step, 1000, 10.0)  # 100 tok/s
        assert not decision.triggered, step


def test_sustained_collapse_fires_after_three_windows():
    guard = _throughput(platform="windows")
    for step in (1, 2, 3):
        assert not guard.observe(step, 1000, 10.0).triggered
    # Baseline = 100 tok/s; 10 tok/s is far below 0.5x.
    fired = []
    for step in range(4, 12):
        decision = guard.observe(step, 100, 10.0)
        if decision.triggered:
            fired.append((step, decision.reason))
    assert fired
    step, reason = fired[0]
    assert step == 9
    assert "throughput collapse" in reason
    assert "WDDM" in reason


def test_persisted_baseline_protects_immediately_after_resume():
    guard = _throughput()
    guard.load_state_dict({"baseline_tok_s": 100.0, "low_windows": 0})
    fired = []
    for step in range(1, 40):
        decision = guard.observe(step, 400, 10.0)  # 40 tok/s vs 100 baseline
        if decision.triggered:
            fired.append((step, decision.reason))
    assert fired
    step, reason = fired[0]
    assert step <= 25
    assert "throughput collapse" in reason


def test_ram_guard_threshold():
    guard = RamGuard(2.0)
    assert not guard.observe(2.5).triggered
    decision = guard.observe(1.5)
    assert decision.triggered and "available RAM" in decision.reason


def test_time_save_uses_a_fake_clock_and_rearms_on_save():
    ticks = {"now": 0.0}
    callback = TimeSaveCallback(30.0, clock=lambda: ticks["now"])
    control = _control()

    ticks["now"] = 60.0
    callback.on_step_end(None, None, control)
    assert control.should_save is False

    ticks["now"] = 30.0 * 60.0 + 1
    callback.on_step_end(None, None, control)
    assert control.should_save is True

    control.should_save = False
    callback.on_save(None, None, control)
    callback.on_step_end(None, None, control)
    assert control.should_save is False

    ticks["now"] = 61.0 * 60.0
    callback.on_step_end(None, None, control)
    assert control.should_save is True
    assert callback.saves == 1


def test_watchdog_requests_checkpoint_and_exit_75_on_low_ram():
    trainer = SimpleNamespace(tokens_seen=0, exit_code=None, exit_reason=None)
    clock = {"now": 0.0}
    watchdog = WatchdogCallback(
        trainer,
        _throughput(),
        RamGuard(2.0),
        clock=lambda: clock["now"],
        ram_reader=lambda: 1.0,
        reserved_reader=lambda: None,
    )
    control = _control()
    trainer.tokens_seen = 100
    clock["now"] = 10.0
    watchdog.on_step_end(None, SimpleNamespace(global_step=1), control)
    assert trainer.exit_code == GUARD_EXIT_CODE
    assert control.should_save and control.should_training_stop
    assert "ram" in trainer.exit_reason


def test_watchdog_does_not_double_count_tokens_after_resume():
    trainer = SimpleNamespace(tokens_seen=1_000_000, exit_code=None, exit_reason=None)
    clock = {"now": 1_000.0}
    guard = WatchdogCallback(
        trainer,
        _throughput(),
        RamGuard(0.0),  # RAM never fires in this test
        clock=lambda: clock["now"],
        ram_reader=lambda: 8.0,
        reserved_reader=lambda: None,
    )
    control = _control()
    guard.on_train_begin(None, SimpleNamespace(global_step=5_000), control)
    for step in range(5_001, 5_030):
        trainer.tokens_seen += 100  # normal per-step delta, not the whole run
        clock["now"] += 10.0
        guard.on_step_end(None, SimpleNamespace(global_step=step), control)
        assert not control.should_training_stop
    assert trainer.exit_code is None


def test_throughput_collapse_still_fires_after_a_resume():
    trainer = SimpleNamespace(tokens_seen=1_000_000, exit_code=None, exit_reason=None)
    clock = {"now": 1_000.0}
    guard = WatchdogCallback(
        trainer,
        _throughput(),
        RamGuard(0.0),
        clock=lambda: clock["now"],
        ram_reader=lambda: 8.0,
        reserved_reader=lambda: None,
    )
    control = _control()
    guard.on_train_begin(None, SimpleNamespace(global_step=5_000), control)
    for step in range(5_001, 5_004):
        trainer.tokens_seen += 1000  # 100 tok/s
        clock["now"] += 10.0
        guard.on_step_end(None, SimpleNamespace(global_step=step), control)
    for step in range(5_004, 5_020):
        trainer.tokens_seen += 100  # 10 tok/s
        clock["now"] += 10.0
        guard.on_step_end(None, SimpleNamespace(global_step=step), control)
        if control.should_training_stop:
            break
    assert trainer.exit_code == GUARD_EXIT_CODE
    assert control.should_save
    assert "throughput" in trainer.exit_reason


def test_guard_survives_a_failing_status_update():
    class BrokenStatus:
        def update(self, **kwargs):
            raise RuntimeError("status disk full")

    trainer = SimpleNamespace(tokens_seen=0, exit_code=None, exit_reason=None)
    clock = {"now": 0.0}
    guard = WatchdogCallback(
        trainer,
        _throughput(),
        RamGuard(2.0),
        clock=lambda: clock["now"],
        ram_reader=lambda: 1.0,
        reserved_reader=lambda: None,
        status=BrokenStatus(),
    )
    control = _control()
    trainer.tokens_seen = 10
    clock["now"] = 1.0
    guard.on_step_end(None, SimpleNamespace(global_step=1), control)
    assert trainer.exit_code == GUARD_EXIT_CODE
    assert control.should_save and control.should_training_stop


def test_time_save_clock_resets_at_train_begin():
    ticks = {"now": 10_000.0}
    callback = TimeSaveCallback(30.0, clock=lambda: ticks["now"])
    control = _control()
    ticks["now"] += 10_000.0  # slow model load
    callback.on_train_begin(None, None, control)
    ticks["now"] += 60.0
    callback.on_step_end(None, None, control)
    assert control.should_save is False
    ticks["now"] += 30.0 * 60.0
    callback.on_step_end(None, None, control)
    assert control.should_save is True


def test_guard_state_dict_round_trip():
    guard = _throughput()
    guard.load_state_dict({"baseline_tok_s": 123.0, "low_windows": 2})
    assert guard.state_dict() == {"baseline_tok_s": 123.0, "low_windows": 2}
    decision = guard.observe(1, 400, 10.0)
    assert decision.metrics["baseline_tok_s"] == 123.0


def test_watchdog_restores_the_persisted_baseline_immediately():
    trainer = SimpleNamespace(
        tokens_seen=0,
        exit_code=None,
        exit_reason=None,
        _persisted_guard_state={"baseline_tok_s": 100.0, "low_windows": 0},
    )
    clock = {"now": 0.0}
    watchdog = WatchdogCallback(
        trainer,
        _throughput(),
        RamGuard(0.0),
        clock=lambda: clock["now"],
        ram_reader=lambda: 8.0,
        reserved_reader=lambda: None,
    )
    control = _control()
    watchdog.on_train_begin(None, SimpleNamespace(global_step=5_000), control)
    assert trainer._throughput_baseline == 100.0
    for step in range(5_001, 5_030):
        trainer.tokens_seen += 400  # 40 tok/s vs the persisted 100 baseline
        clock["now"] += 10.0
        watchdog.on_step_end(None, SimpleNamespace(global_step=step), control)
        if control.should_training_stop:
            break
    assert trainer.exit_code == GUARD_EXIT_CODE
    assert "throughput" in trainer.exit_reason
