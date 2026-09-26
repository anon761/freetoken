"""DSpark admission control + drafter fault latch.

Ported (policy only, no Metal) from DwarfStar / antirez-ds4's
``docs/DSPARK-V41.md`` section 4 — the windowed cost-feedback controller with its
confidence admission and its drafter fault latch.

Why: a speculative drafter does not always pay. DwarfStar measured break-even at
~3.76 committed tokens per verify cycle, and a *fixed* block loses outright on
some real workloads. The controller bounds that loss: its floor is serial decode.
Our DS4.1 draft accepts ~14 % (weak checkpoint), so the useful lever is refusing
to verify a cycle the drafter is not confident about — not chasing acceptance.

Two independent pieces:

* :class:`DSparkController` — per request. Each cycle it verifies the draft length
  with the best expected committed tokens per ms (calibrated confidences against a
  verify cost model and the measured serial step), or declines. Unproductive probing
  is throttled: after a decline or a losing cycle the drafter pauses for 1, 2, 4 ... 16
  serial steps (the pause resets on the first winning cycle), so prose pays a draft
  only now and then while predictable spans draft every step. It never latches off.
* :class:`DSparkFaultLatch` — per request. A recoverable ("drained") drafter
  failure before the target verify falls back to serial and skips the drafter for
  the rest of the request; an unrecoverable ("undrained") failure after the
  verify batch refuses further decoding, because serial decode from there would
  answer from a state serial decode could never reach.
"""

from __future__ import annotations

import os
import statistics
from collections import deque
from enum import Enum

DEFAULT_P_MIN = 0.75          # confidence floor of the fallback prefix rule (no cost model)
DEFAULT_MIN_DRAFT = 3         # fallback: admitted prefix below this declines the cycle
DEFAULT_ENTRY_WAIT = 2        # serial steps measured before the first draft (prices them)
SERIAL_MEDIAN_N = 9           # serial cost is the median of the last N measured steps
MAX_PAUSE = 16                # drafter pause after unproductive cycles: 1, 2, 4, 8, 16 steps
GAIN_MARGIN = 0.05            # a verify must beat serial by this much in expected tokens/ms


class Decision(str, Enum):
    ATTEMPT = "attempt"       # verify the full block
    DECLINE = "decline"       # skip the verify, take a serial step instead
    SERIAL = "serial"         # not a candidate right now (gate / entry wait / backoff / off)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class DSparkController:
    """Per-request admission policy. Not thread-safe; one instance per request."""

    def __init__(
        self,
        *,
        adaptive: bool = True,
        p_min: float = DEFAULT_P_MIN,
        min_draft: int = DEFAULT_MIN_DRAFT,
        entry_wait: int = DEFAULT_ENTRY_WAIT,
        reasoning_serial: bool = True,
        block: int = 5,
    ) -> None:
        self.adaptive = adaptive
        self.p_min = p_min
        self.min_draft = min_draft
        self.entry_wait = entry_wait
        self.reasoning_serial = reasoning_serial
        self.block = block
        self.start_request()

    @classmethod
    def from_env(cls, block: int = 5) -> "DSparkController":
        """Read the documented ``=`` overrides at request start (DwarfStar's contract)."""
        def _num(name, default, cast):
            raw = os.environ.get(name)
            try:
                return cast(raw) if raw is not None and raw != "" else default
            except ValueError:
                return default
        return cls(
            adaptive=_env_flag("FREETOKEN_DSPARK_ADAPTIVE", True),
            p_min=_num("FREETOKEN_DSPARK_P_MIN", DEFAULT_P_MIN, float),
            min_draft=_num("FREETOKEN_DSPARK_MIN_DRAFT", DEFAULT_MIN_DRAFT, int),
            entry_wait=_num("FREETOKEN_DSPARK_MIN_SERIAL_TOKENS", DEFAULT_ENTRY_WAIT, int),
            reasoning_serial=_env_flag("FREETOKEN_DSPARK_REASONING_SERIAL", True),
            block=block,
        )

    def start_request(self, reasoning: bool = False) -> None:
        """Open a fresh request ledger (DwarfStar: the ledger is request-local)."""
        self.reasoning = reasoning
        self.consumed = 0                    # consumed serial tokens this request
        self.cooldown = 0                    # serial steps left before the next draft
        self.streak = 0                      # consecutive unproductive cycles
        self._serial: deque[float] = deque(maxlen=SERIAL_MEDIAN_N)
        # counters
        self.attempts = 0
        self.declines = 0
        self.losing_cycles = 0
        self.committed = 0
        self.serial_rows = 0
        self.cycle_ms = 0.0
        self.net_ms = 0.0

    # ------------------------------------------------------------- measurement
    def note_serial(self, wall_ms: float, consumed: int = 1) -> None:
        """Record a serial step (used to price cycles and to spend the cooldown)."""
        if consumed > 0:
            self._serial.append(wall_ms / consumed)
        self.consumed += consumed
        self.serial_rows += consumed
        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - consumed)

    def serial_ms(self) -> float | None:
        """Median of the last N measured serial steps (jitter-robust); None until known."""
        return statistics.median(self._serial) if self._serial else None

    # --------------------------------------------------------------- admission
    def admitted_prefix(self, confidences) -> int:
        """Longest run of drafted positions from the anchor with confidence >= p_min.

        ``confidences`` is the per-position confidence after the Markov proposal
        (already on the host for DSpark, so scoring it costs no GPU work).
        """
        n = 0
        for c in confidences:
            if float(c) >= self.p_min:
                n += 1
            else:
                break
        return n

    def should_attempt(self) -> bool:
        if not self.adaptive:
            return True                       # propose everywhere (the fixed-block arm)
        if self.reasoning and self.reasoning_serial:
            return False                      # reasoning spans decode serially
        if self.consumed < self.entry_wait:
            return False                      # entry wait: price against a measured step
        if self.cooldown > 0:
            return False                      # still backing off
        return True

    def best_length(self, confidences, serial_ms: float, verify_ms) -> int:
        """Draft length to verify: the one maximizing the expected committed tokens per ms,
        0 when a serial step is expected to be faster.

        With calibrated per-position confidences p_i, verifying the first L drafts commits
        ``1 + sum_{j<=L} prod_{i<=j} p_i`` tokens (the anchor's bonus plus the expected
        accepted run) at ``verify_ms(L + 1)``; a serial step commits one token at
        ``serial_ms``. The drafter's own cost is already spent when this is asked."""
        best, best_rate = 0, (1.0 + GAIN_MARGIN) / serial_ms
        expected, run = 0.0, 1.0
        for length, c in enumerate(confidences, start=1):
            run *= float(c)
            expected += run
            rate = (1.0 + expected) / verify_ms(length + 1)
            if rate > best_rate:
                best, best_rate = length, rate
        return best

    def decide(self, confidences, verify_ms=None) -> tuple[Decision, int]:
        """Return ``(Decision, draft length)`` for the coming cycle.

        With ``verify_ms`` (a verify cost model, ms for ``t`` tokens) and a measured
        serial step, the length is the throughput-optimal one (:meth:`best_length`);
        otherwise the confidence-floor prefix of DwarfStar's controller."""
        if not self.adaptive:
            return Decision.ATTEMPT, self.block   # propose everywhere (fixed-block arm)
        if not self.should_attempt():
            return Decision.SERIAL, 0
        serial = self.serial_ms()
        if verify_ms is not None and serial:
            length = self.best_length(confidences, serial, lambda t: verify_ms(t, serial))
            return (Decision.ATTEMPT, length) if length > 0 else (Decision.DECLINE, 0)
        admitted = self.admitted_prefix(confidences)
        if admitted < self.min_draft:
            return Decision.DECLINE, admitted
        return Decision.ATTEMPT, admitted

    # --------------------------------------------------------------- feedback
    def _pause(self) -> None:
        """One more unproductive cycle: pause the drafter for 1, 2, 4 ... MAX_PAUSE steps."""
        self.streak += 1
        self.cooldown = min(MAX_PAUSE, 1 << (self.streak - 1))

    def note_decline(self, draft_ms: float) -> None:
        """A drafted cycle the target was not asked to verify: its drafter cost is lost."""
        self.declines += 1
        self.cycle_ms += draft_ms
        self.net_ms += draft_ms
        self._pause()

    def note_cycle(self, wall_ms: float, consumed: int) -> None:
        """A verified cycle (draft + verify ``wall_ms``) that committed ``consumed`` tokens,
        priced against as many serial steps: a win clears the pause, a loss extends it."""
        self.attempts += 1
        self.committed += consumed
        self.cycle_ms += wall_ms
        serial = self.serial_ms()
        if serial is None:
            return
        net = wall_ms - consumed * serial
        self.net_ms += net
        if net < 0.0:
            self.streak = 0
            self.cooldown = 0
        else:
            self.losing_cycles += 1
            self._pause()

    # --------------------------------------------------------------- lifecycle
    def enter_reasoning(self) -> None:
        self.reasoning = True

    def leave_reasoning(self) -> None:
        if self.reasoning:
            self.reasoning = False
            self.cooldown = 0  # evidence inside a reasoning span does not price what follows

    def note_prefix_reset(self) -> None:
        """The prefix stopped being an extension of what it was: drop the measured serial
        steps. The pause survives — it is request policy, not measured evidence."""
        self._serial.clear()

    def telemetry(self) -> dict:
        return {
            "dspark_attempts": self.attempts,
            "dspark_declines": self.declines,
            "dspark_losing_cycles": self.losing_cycles,
            "dspark_committed": self.committed,
            "dspark_serial_rows": self.serial_rows,
            "dspark_cycle_ms": round(self.cycle_ms, 3),
            "dspark_net_ms": round(self.net_ms, 3),
            "dspark_cooldown": self.cooldown,
        }


class VerifyCostModel:
    """Verify wall time as a function of its token count (anchor + drafts), shared by all
    requests (it is a property of the hardware and the model). A least-squares line over
    the recent measurements once two different lengths were seen; before that a prior
    relative to the serial step, ``serial * (1 + 0.5 * t)`` (measured on 2x3090 offload:
    a verify costs ~76 + 36 t ms against a 73 ms serial step)."""

    def __init__(self, window: int = 64) -> None:
        self._obs: deque[tuple[int, float]] = deque(maxlen=window)

    def note(self, tokens: int, wall_ms: float) -> None:
        self._obs.append((int(tokens), float(wall_ms)))

    def __call__(self, tokens: int, serial_ms: float) -> float:
        ts = [t for t, _ in self._obs]
        if len(set(ts)) < 2:
            return serial_ms * (1.0 + 0.5 * tokens)
        n = len(self._obs)
        mt = sum(ts) / n
        mv = sum(v for _, v in self._obs) / n
        var = sum((t - mt) ** 2 for t in ts)
        slope = max(0.0, sum((t - mt) * (v - mv) for t, v in self._obs) / var)
        return max(1e-3, mv + slope * (tokens - mt))


class FaultKind(str, Enum):
    DRAINED = "drained"       # before the verify batch: serial fallback + skip drafter
    UNDRAINED = "unsafe"      # after the verify batch: refuse, the state cannot be un-fed


class DSparkFaultLatch:
    """Per-request drafter fault latch.

    ``DRAINED``: the failure happened up to the verify batch, so the target's KV,
    position and checkpoint are untouched — fall through to serial, skip the
    drafter for the rest of the request. ``UNDRAINED``: the target already consumed
    rows the request cannot un-feed, so there is no correct fallback and the
    request must refuse rather than answer from a state serial decode could not
    reach."""

    def __init__(self) -> None:
        self.state: FaultKind | None = None
        self.events: list[str] = []

    def trip(self, kind: FaultKind, reason: str = "") -> FaultKind:
        # An undrained failure dominates a prior drained latch.
        if self.state is None or kind == FaultKind.UNDRAINED:
            self.state = kind
        self.events.append(f"{kind.value}:{reason}" if reason else kind.value)
        return self.state

    def drafter_disabled(self) -> bool:
        return self.state is not None

    def unsafe(self) -> bool:
        return self.state == FaultKind.UNDRAINED
