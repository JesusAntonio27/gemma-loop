"""The legacy Phase 1 single-SFT planner is retired in prereg-v1."""

from __future__ import annotations

import pytest

import scripts.phase1_plan as plan_mod


def test_legacy_plan_main_refuses_to_run():
    with pytest.raises(RuntimeError, match="retired"):
        plan_mod.main(["--prereg", "prereg/prereg.yaml"])


def test_legacy_build_plan_refuses_to_run():
    with pytest.raises(RuntimeError, match="retired"):
        plan_mod.build_plan("prereg/prereg.yaml")
