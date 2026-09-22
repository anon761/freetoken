"""Loading progress bars, consistent across the framework's weight-load paths.

tqdm-backed (the framework's existing idiom) and byte-oriented -- load time tracks bytes
moved off disk, not tensor count, so a byte bar shows a meaningful GiB/s. Bars are disabled
off rank 0 (only the primary should draw) so multi-rank logs stay clean.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from tqdm import tqdm

from freetoken.distributed import try_get_tp_info
from freetoken.utils.logger import init_logger

logger = init_logger(__name__)

_PROGRESS_SINK: Optional[Callable[[str, int, int], None]] = None


def cpu_percent() -> float:
    """System-wide CPU busy percent since the previous call, from ``/proc/stat``.

    Self-sampling: the first call (or the first after a long gap) has no baseline and
    returns -1.0 so callers can skip it instead of printing a bogus instant value.
    """
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        if not parts or parts[0] != "cpu":
            return -1.0
        vals = [float(x) for x in parts[1:9]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
        total = sum(vals)
    except (OSError, ValueError, IndexError):
        return -1.0
    state = _CPU_SAMPLE
    with state["lock"]:
        prev_busy, prev_total = state["busy"], state["total"]
        state["busy"], state["total"] = total - idle, total
    if prev_total <= 0:
        return -1.0
    dtot = total - prev_total
    if dtot <= 0:
        return -1.0
    return 100.0 * ((total - idle) - prev_busy) / dtot


_CPU_SAMPLE: dict = {"busy": 0.0, "total": 0.0, "lock": threading.Lock()}


def set_progress_sink(sink: Optional[Callable[[str, int, int], None]]) -> None:
    """Install (or clear, with None) a global progress callback invoked (throttled)
    by every ``byte_bar`` as ``sink(desc, done_bytes, total_bytes)``. The scheduler's
    rank-0 process installs one that forwards to its ack_queue; call sites are unchanged."""
    global _PROGRESS_SINK
    _PROGRESS_SINK = sink


def _on_primary() -> bool:
    info = try_get_tp_info()
    return info is None or info.is_primary()


def emit_progress(desc: str, done: int, total: int) -> None:
    """Push a one-off update to the installed sink for a phase that has no ``byte_bar`` — e.g.
    CUDA-graph capture / warmup, which moves no bytes. A ``total <= 0`` reads downstream as an
    indeterminate phase (no percentage). No-op off rank 0 or when no sink is installed."""
    sink = _PROGRESS_SINK
    if sink is not None and _on_primary():
        try:
            sink(desc, done, total)
        except Exception:  # noqa: BLE001 — progress reporting must never break load
            pass


class _SinkTqdm(tqdm):
    """A tqdm that also forwards its progress to the installed ``_PROGRESS_SINK``,
    throttled to <=1 emit / 0.5 s OR a >=1% delta (plus a guaranteed final emit).

    With ``_monitor=True`` it additionally logs a plain rate + CPU line every
    ``_monitor_interval`` seconds (rank 0 only), so `journalctl`/systemd sees the load
    speed without tqdm's carriage-return bars (which journald renders as blob data).
    """

    def __init__(self, *args, _monitor: bool = False, _monitor_interval: float = 5.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sink_last_emit = 0.0
        self._sink_last_frac = -1.0
        self._stop = threading.Event()
        self._mon_thread: threading.Thread | None = None
        if _monitor and not self.disable and self.total:
            cpu_percent()  # prime the sampler so the first logged value is meaningful
            self._mon_thread = threading.Thread(
                target=self._monitor, args=(_monitor_interval,), name="load-progress",
                daemon=True,
            )
            self._mon_thread.start()

    def _monitor(self, interval: float) -> None:
        total = int(self.total or 0)
        start_t = time.monotonic()
        last_n, last_t = self.n, start_t
        while not self._stop.wait(interval):
            now = time.monotonic()
            window = (self.n - last_n) / max(now - last_t, 1e-9)
            avg = self.n / max(now - start_t, 1e-9)
            cpu = cpu_percent()
            logger.info(
                "%s: %.2f/%.2f GiB (%.0f MiB/s avg, %.0f MiB/s now, CPU %s)",
                self.desc or "load", self.n / 2**30, total / 2**30,
                avg / 2**20, window / 2**20,
                f"{cpu:.0f}%" if cpu >= 0 else "?",
            )
            last_n, last_t = self.n, now
            if total and self.n >= total:
                return

    def close(self) -> None:
        self._stop.set()
        if self._mon_thread is not None:
            self._mon_thread.join(timeout=1.0)
        super().close()

    def update(self, n: int = 1):  # type: ignore[override]
        ret = super().update(n)
        sink = _PROGRESS_SINK
        if sink is not None:
            total = int(self.total or 0)
            done = int(self.n or 0)
            frac = (done / total) if total else 0.0
            now = time.monotonic()
            if (
                now - self._sink_last_emit >= 0.5
                or frac - self._sink_last_frac >= 0.01
                or (total and done >= total)
            ):
                self._sink_last_emit = now
                self._sink_last_frac = frac
                try:
                    sink(self.desc or "", done, total)
                except Exception:  # noqa: BLE001 — progress reporting must never break load
                    pass
        return ret


def byte_bar(total: int, desc: str, monitor: bool = False) -> tqdm:
    """A byte-scaled bar (shows e.g. ``12.8GiB [00:02, 6.1GiB/s]``); ``update(nbytes)`` it
    as each tensor/bank/shard finishes reading. Also drives the progress sink when installed.
    ``monitor=True`` adds a periodic plain log line with rate + system CPU (rank 0)."""
    return _SinkTqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
                     disable=not _on_primary(), leave=False, dynamic_ncols=True,
                     _monitor=monitor)


def count_bar(iterable, desc: str, total: int | None = None) -> tqdm:
    """A plain count bar over an iterable (use when total bytes aren't known up front)."""
    return tqdm(iterable, desc=desc, total=total, disable=not _on_primary(),
                leave=False, dynamic_ncols=True)


__all__ = ["byte_bar", "count_bar", "set_progress_sink"]
