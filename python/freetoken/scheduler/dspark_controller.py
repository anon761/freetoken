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

* :class:`DSparkController` — per request. The confidence gate (admit a cycle
  only when the drafted prefix is confident enough) is the part wired into the
  scheduler today; the windowed cost feedback (price a cycle from the measured
  serial step, back off for a bounded number of tokens) is complete and unit
  tested, and activates when the caller feeds it serial step timings. It never
  latches off, matching DwarfStar.
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

DEFAULT_P_MIN = 0.75          # per-position confidence floor (DwarfStar's GLM value)
DEFAULT_MIN_DRAFT = 3         # admitted prefix below this declines the cycle
DEFAULT_ENTRY_WAIT = 16       # consumed serial tokens before the first attempt
WINDOW_ATTEMPTS = 3           # a cost window is three complete attempts
SERIAL_MEDIAN_N = 9           # serial cost is the median of the last N measured steps
BACKOFF_BASE = 16             # consumed-token cooldown ladder: 16, 32, 64, 128
BAD_RUN_CAP = 4


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
        self.cooldown = 0                    # remaining consumed-token cooldown
        self.bad_run = 0
        self._serial: deque[float] = deque(maxlen=SERIAL_MEDIAN_N)
        self._win_net = 0.0
        self._win_attempts = 0
        # counters
        self.attempts = 0
        self.declines = 0
        self.losing_cycles = 0
        self.windows = 0
        self.backoffs = 0
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

    def decide(self, confidences) -> tuple[Decision, int]:
        """Return ``(Decision, admitted_prefix)`` for the coming cycle."""
        if not self.adaptive:
            return Decision.ATTEMPT, self.block   # propose everywhere (fixed-block arm)
        if not self.should_attempt():
            return Decision.SERIAL, 0
        admitted = self.admitted_prefix(confidences)
        if admitted < self.min_draft:
            return Decision.DECLINE, admitted
        return Decision.ATTEMPT, admitted

    # --------------------------------------------------------------- feedback
    def note_cycle(self, wall_ms: float, consumed: int, verified: bool) -> None:
        """Price one cycle. ``verified`` = the target was asked to check something.

        A decline is a chosen call: it takes a window slot and its net is a
        straight loss of the drafter cost (DwarfStar: getting this asymmetry wrong
        made prose run at 0.816x serial because nothing throttled probing)."""
        self.cycle_ms += wall_ms
        serial = self.serial_ms()
        if verified:
            self.attempts += 1
            self.committed += consumed
            if serial is not None and wall_ms - consumed * serial > 0.0:
                self.losing_cycles += 1
        else:
            self.declines += 1
        if serial is None:
            return
        net = wall_ms - consumed * serial
        self.net_ms += net
        self._win_net += net
        self._win_attempts += 1
        if self._win_attempts < WINDOW_ATTEMPTS:
            return
        self.windows += 1
        if self._win_net < 0.0:
            self.bad_run = 0
            self.cooldown = 0
        else:
            self.bad_run = min(self.bad_run + 1, BAD_RUN_CAP)
            self.cooldown = BACKOFF_BASE << (self.bad_run - 1)
            self.backoffs += 1
        self._win_net = 0.0
        self._win_attempts = 0

    # --------------------------------------------------------------- lifecycle
    def enter_reasoning(self) -> None:
        self.reasoning = True

    def leave_reasoning(self) -> None:
        if self.reasoning:
            self.reasoning = False
            self.cooldown = 0  # evidence inside a reasoning span does not price what follows

    def note_prefix_reset(self) -> None:
        """The prefix stopped being an extension of what it was: drop measured
        evidence (serial window + open window). The cooldown survives — it is
        request policy, not measured evidence."""
        self._serial.clear()
        self._win_net = 0.0
        self._win_attempts = 0

    def telemetry(self) -> dict:
        return {
            "dspark_attempts": self.attempts,
            "dspark_declines": self.declines,
            "dspark_losing_cycles": self.losing_cycles,
            "dspark_windows": self.windows,
            "dspark_backoffs": self.backoffs,
            "dspark_committed": self.committed,
            "dspark_serial_rows": self.serial_rows,
            "dspark_cycle_ms": round(self.cycle_ms, 3),
            "dspark_net_ms": round(self.net_ms, 3),
            "dspark_cooldown": self.cooldown,
        }


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
