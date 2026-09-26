"""A request a speculative round advanced this iteration is not decoded again by the plain
step: its bonus token stays pending for the next round's verify (a plain step would process
it outside the round — a redundant decode, and a DSpark draft context missing it)."""
from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler


def _scheduler(reqs, advanced):
    batch = SimpleNamespace(is_decode=True, is_prefill=False, reqs=list(reqs), prompt_admissions=[])
    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 99
    s.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: None)
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: batch)
    s.mtp = SimpleNamespace(_deferred=set(), _spec_iter=set(advanced))
    s._prepare_batch = lambda b: b
    s.send_result = lambda messages: None
    return s


def test_plain_decode_skips_rows_advanced_by_the_round():
    a, b = SimpleNamespace(table_idx=0), SimpleNamespace(table_idx=1)
    out = Scheduler._schedule_next_batch(_scheduler([a, b], advanced={0}))
    assert out.reqs == [b]


def test_nothing_left_to_decode_yields_no_batch():
    a = SimpleNamespace(table_idx=0)
    assert Scheduler._schedule_next_batch(_scheduler([a], advanced={0})) is None


def test_without_a_round_the_plain_decode_is_untouched():
    a, b = SimpleNamespace(table_idx=0), SimpleNamespace(table_idx=1)
    assert Scheduler._schedule_next_batch(_scheduler([a, b], advanced=set())).reqs == [a, b]
