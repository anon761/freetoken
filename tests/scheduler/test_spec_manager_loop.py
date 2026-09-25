# Copyright (c) 2026 FreeToken contributors
# Scheduler._overlap_enabled: DSpark's synchronous round runs on normal_loop without the env
# gate (it asserted FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1, which --mtp on no longer sets).
from __future__ import annotations

from types import SimpleNamespace

from freetoken.env import ENV
from freetoken.scheduler.dspark import DSparkManager
from freetoken.scheduler.mtp import MTPManager
from freetoken.scheduler.scheduler import _NOOP_MTP, Scheduler


def _sched(mtp):
    sched = Scheduler.__new__(Scheduler)
    sched.mtp = mtp
    return sched


def test_loop_choice_follows_the_spec_manager(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    assert Scheduler._overlap_enabled(_sched(_NOOP_MTP))
    assert Scheduler._overlap_enabled(_sched(SimpleNamespace(overlaps=MTPManager.overlaps)))
    assert not Scheduler._overlap_enabled(_sched(SimpleNamespace(overlaps=DSparkManager.overlaps)))


def test_env_forces_the_drain_safe_loop(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    assert not Scheduler._overlap_enabled(_sched(_NOOP_MTP))
