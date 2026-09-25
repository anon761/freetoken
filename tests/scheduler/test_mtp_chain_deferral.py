# Copyright (c) 2026 FreeToken contributors
# Scheduler._refresh_mtp_deferred with chaining on (the default): a server without a draft
# head (the no-op manager) must not touch it -- it has no eligible() and crashed the scheduler.
from __future__ import annotations

from types import SimpleNamespace

from freetoken.env import ENV
from freetoken.scheduler.scheduler import _NOOP_MTP, Scheduler


def test_chain_refresh_ignores_the_noop_manager(monkeypatch):
    monkeypatch.setattr(ENV.MTP_CHAIN, "value", True)
    sched = Scheduler.__new__(Scheduler)
    sched.decode_manager = SimpleNamespace(running_reqs={object()})
    Scheduler._refresh_mtp_deferred(sched, _NOOP_MTP)
    assert _NOOP_MTP._deferred == set()
