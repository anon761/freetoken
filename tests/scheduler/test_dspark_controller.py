"""DSpark admission controller + fault latch policy (CPU, no model).

Mirrors the properties DwarfStar's ``tests/test_ds41_dspark_adaptive`` pins:
the backoff ladder, the median's robustness, the entry wait, the reasoning gate
and its clear, an evidence reset that keeps the cooldown, the admitted-prefix
rule, and the decline accounting (three declines take a window's three slots and
arm the first backoff, with ``losing_cycles`` staying zero).
"""

from __future__ import annotations

from freetoken.scheduler.dspark_controller import (
    BACKOFF_BASE,
    Decision,
    DSparkController,
    DSparkFaultLatch,
    FaultKind,
)


def _primed(**kw) -> DSparkController:
    """A controller past its entry wait with a known serial cost of 10 ms/token."""
    c = DSparkController(**kw)
    c.start_request()
    for _ in range(20):
        c.note_serial(10.0, consumed=1)
    return c


def test_adaptive_off_proposes_everywhere():
    c = DSparkController(adaptive=False)
    assert c.should_attempt() is True
    assert c.decide([0.0, 0.0, 0.0])[0] is Decision.ATTEMPT


def test_entry_wait_blocks_until_enough_serial_tokens():
    c = DSparkController(entry_wait=16)
    c.start_request()
    for _ in range(15):
        c.note_serial(10.0, consumed=1)
        assert c.should_attempt() is False
    c.note_serial(10.0, consumed=1)
    assert c.should_attempt() is True


def test_reasoning_span_is_serial_and_leaving_clears_cooldown():
    c = _primed(reasoning_serial=True)
    c.note_cycle(100.0, consumed=4, verified=True)  # losing cycle -> cooldown
    c.enter_reasoning()
    assert c.should_attempt() is False
    c.leave_reasoning()
    assert c.cooldown == 0
    assert c.should_attempt() is True


def test_admitted_prefix_is_the_leading_confident_run():
    c = DSparkController(p_min=0.75, min_draft=3)
    assert c.admitted_prefix([0.9, 0.8, 0.76, 0.5, 0.9]) == 3
    assert c.admitted_prefix([0.5, 0.9, 0.9]) == 0
    assert c.admitted_prefix([0.9, 0.9, 0.9, 0.9, 0.9]) == 5


def test_low_confidence_declines_no_attempt():
    c = _primed()
    assert c.decide([0.9, 0.9, 0.5])[0] is Decision.DECLINE   # admitted 2 < min_draft 3
    assert c.decide([0.9, 0.9, 0.9, 0.4, 0.9])[0] is Decision.ATTEMPT


def test_serial_ms_median_ignores_a_jittery_sample():
    c = DSparkController()
    c.start_request()
    for ms in [10, 10, 10, 10, 10, 10, 10, 10, 1000]:
        c.note_serial(float(ms), consumed=1)
    assert c.serial_ms() == 10.0


def test_backoff_ladder_and_never_latch_off():
    c = _primed()
    cooldowns = []
    for _ in range(4):
        # one losing window: three verified cycles, each slower than the serial cost
        for _ in range(3):
            c.note_cycle(100.0, consumed=4, verified=True)  # net = 100 - 40 > 0 (loss)
        cooldowns.append(c.cooldown)
        c.cooldown = 0  # spend it so the next window can arm
    assert cooldowns == [BACKOFF_BASE, BACKOFF_BASE * 2, BACKOFF_BASE * 4, BACKOFF_BASE * 8]
    # never latches off: a winning window clears the ladder
    for _ in range(3):
        c.note_cycle(10.0, consumed=4, verified=True)       # net = 10 - 40 < 0 (win)
    assert c.bad_run == 0 and c.cooldown == 0


def test_decline_accounting_takes_a_window_slot_without_a_losing_cycle():
    c = _primed()
    for _ in range(3):
        c.note_cycle(12.0, consumed=1, verified=False)      # decline: net = 12 - 10 > 0
    assert c.windows == 1
    assert c.backoffs == 1
    assert c.declines == 3
    assert c.losing_cycles == 0                              # declines are not rejections
    assert c.cooldown == BACKOFF_BASE


def test_evidence_reset_drops_measurements_but_keeps_cooldown():
    c = _primed()
    c.cooldown = 64
    c.note_prefix_reset()
    assert c.serial_ms() is None
    assert c.cooldown == 64


def test_fault_latch_drained_then_undrained_dominates():
    latch = DSparkFaultLatch()
    assert latch.drafter_disabled() is False
    latch.trip(FaultKind.DRAINED, "draft")
    assert latch.drafter_disabled() is True
    assert latch.unsafe() is False
    latch.trip(FaultKind.UNDRAINED, "publish")
    assert latch.unsafe() is True
    assert latch.events == ["drained:draft", "unsafe:publish"]
