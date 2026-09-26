"""DSpark admission controller + fault latch policy (CPU, no model).

Pins the drafter pause (doubling after declines and losing cycles, cleared by a win,
never latching off), the median's robustness, the entry wait, the reasoning gate and its
clear, an evidence reset that keeps the pause, the fallback admitted-prefix rule, and the
throughput-optimal draft length with its verify cost model.
"""

from __future__ import annotations

from freetoken.scheduler.dspark_controller import (
    MAX_PAUSE,
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
    c = DSparkController()
    c.start_request()
    c.note_serial(10.0, consumed=1)
    assert c.should_attempt() is False  # default: two measured serial steps first
    c.note_serial(10.0, consumed=1)
    assert c.should_attempt() is True


def test_reasoning_span_is_serial_and_leaving_clears_cooldown():
    c = _primed(reasoning_serial=True)
    c.note_cycle(100.0, consumed=4)  # losing cycle -> pause
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


def test_pause_doubles_on_unproductive_cycles_and_a_win_clears_it():
    c = _primed()
    pauses = []
    for _ in range(6):
        c.note_decline(12.0)  # a drafted cycle nobody verified
        pauses.append(c.cooldown)
        for _ in range(c.cooldown):
            c.note_serial(10.0)  # the pause is spent in serial steps
        assert c.should_attempt() is True
    assert pauses == [1, 2, 4, 8, MAX_PAUSE, MAX_PAUSE]
    c.note_cycle(100.0, consumed=4)  # losing verify (100 ms > 4 x 10 ms) keeps doubling
    assert c.cooldown == MAX_PAUSE and c.losing_cycles == 1
    c.cooldown = 0
    c.note_cycle(10.0, consumed=4)  # win: never latches off, the pause resets
    assert c.cooldown == 0 and c.streak == 0
    assert c.declines == 6 and c.attempts == 2


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


def _ctl_with_serial(ms: float):
    from freetoken.scheduler.dspark_controller import DSparkController

    ctl = DSparkController(entry_wait=0)
    ctl.note_serial(ms)
    return ctl


def test_best_length_picks_the_throughput_optimal_prefix():
    from freetoken.scheduler.dspark_controller import DSparkController

    ctl = DSparkController()
    cost = lambda t: 76.0 + 36.0 * t  # noqa: E731 -- measured 2x3090 verify model
    # confident run then a cliff: verify exactly the confident part
    assert ctl.best_length([0.99, 0.99, 0.99, 0.2, 0.2], 73.0, cost) == 3
    # nothing confident: a serial step is faster
    assert ctl.best_length([0.5, 0.4, 0.3], 73.0, cost) == 0
    # all near-certain: the longest block
    assert ctl.best_length([1.0] * 5, 73.0, cost) == 5


def test_verify_cost_model_prior_then_fit():
    from freetoken.scheduler.dspark_controller import VerifyCostModel

    m = VerifyCostModel()
    assert m(4, 70.0) == 70.0 * 3.0  # prior: serial * (1 + 0.5 t)
    for t, v in ((2, 156.0), (3, 174.0), (4, 213.0), (6, 294.0)):
        m.note(t, v)
    assert abs(m(6, 70.0) - 289.3) < 1.0 and abs(m(2, 70.0) - 147.1) < 1.0


def test_decide_uses_the_cost_model_once_serial_is_measured():
    from freetoken.scheduler.dspark_controller import Decision, VerifyCostModel

    ctl = _ctl_with_serial(73.0)
    model = VerifyCostModel()
    assert ctl.decide([0.99, 0.99, 0.99, 0.1, 0.1], model) == (Decision.ATTEMPT, 3)
    assert ctl.decide([0.3, 0.3, 0.3, 0.3, 0.3], model) == (Decision.DECLINE, 0)
