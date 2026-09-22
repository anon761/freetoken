"""Per-GPU telemetry for the periodic status lines: VRAM, utilization, temperature, power.

Best-effort and torch-free: prefers ``pynvml`` when importable, falls back to one
``nvidia-smi`` query, and yields nothing when neither is available (CPU-only host or a
stripped container). NVML is initialized once per process and held, like ``daemon.metrics``.
Sampling is TTL-cached so a busy decode log never pays a probe per line.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from freetoken.gpu_select import is_gpu_index

_SMI_QUERY = "index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"


@dataclass(frozen=True)
class GpuSample:
    """One GPU's live telemetry; ``None`` fields are values the driver did not report."""

    index: int
    uuid: str | None
    mem_used: int
    mem_total: int
    util_pct: int | None
    temp_c: int | None
    power_w: float | None


def format_gpu_line(tp_size: int, samples: Sequence[GpuSample]) -> str:
    """The compact one-line segment: ``TP=n | GPU0 VRAM 21.4/24.0G 88% 72C 315W | ...``."""
    parts = [f"TP={tp_size}"]
    for s in samples:
        fields = [f"GPU{s.index}", f"VRAM {_gib(s.mem_used)}/{_gib(s.mem_total)}G"]
        if s.util_pct is not None:
            fields.append(f"{s.util_pct}%")
        if s.temp_c is not None:
            fields.append(f"{s.temp_c}C")
        if s.power_w is not None:
            fields.append(f"{s.power_w:.0f}W")
        parts.append(" ".join(fields))
    return " | ".join(parts)


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.1f}"


class GpuTelemetry:
    """Formats the engine's GPUs as one status segment, sampling at most once per TTL.

    ``gpu_assigned`` are the --gpu UUIDs in rank order, ``gpu`` the raw entries; when
    neither is given the engine's GPUs are the ordinals ``0..tp_size-1``. ``sampler`` is
    injectable for tests (defaults to NVML, then nvidia-smi).
    """

    def __init__(
        self,
        *,
        tp_size: int,
        gpu_assigned: "Sequence[str] | None" = None,
        gpu: Sequence[str] = (),
        ttl_s: float = 1.0,
        now: Callable[[], float] = time.monotonic,
        sampler: "Callable[[], list[GpuSample]] | None" = None,
    ) -> None:
        self.tp_size = max(1, int(tp_size))
        self.gpu_assigned = tuple(gpu_assigned) if gpu_assigned else ()
        self.gpu = tuple(gpu) if gpu else ()
        self._ttl = ttl_s
        self._now = now
        self._sampler = sampler or sample_gpus
        self._lock = threading.Lock()
        self._cache: "tuple[float, list[GpuSample]] | None" = None

    def line(self) -> str:
        """The status segment, or "" when the engine's GPUs cannot be sampled."""
        samples = self._selected(self._samples())
        if not samples:
            return ""
        return format_gpu_line(self.tp_size, samples)

    def _samples(self) -> list[GpuSample]:
        with self._lock:
            if self._cache is not None and (self._now() - self._cache[0]) < self._ttl:
                return self._cache[1]
            samples = self._sampler()
            self._cache = (self._now(), samples)
            return samples

    def _selected(self, samples: Sequence[GpuSample]) -> list[GpuSample]:
        if self.gpu_assigned:
            wanted = tuple(u.upper() for u in self.gpu_assigned)
            return [s for s in samples if s.uuid and _uuid_in(s.uuid, wanted)]
        if self.gpu:
            indices = {int(e) for e in self.gpu if is_gpu_index(e)}
            if indices:
                return [s for s in samples if s.index in indices]
            wanted = tuple(e.upper() for e in self.gpu)
            return [s for s in samples if s.uuid and _uuid_in(s.uuid, wanted)]
        return [s for s in samples if s.index < self.tp_size]


def _uuid_in(uuid: str, wanted: Sequence[str]) -> bool:
    upper = uuid.upper()
    return any(upper.startswith(w) or w.startswith(upper) for w in wanted)


def sample_gpus() -> list[GpuSample]:
    """Every physical GPU's telemetry via NVML, else nvidia-smi; [] when neither works."""
    samples = _sample_nvml()
    return samples if samples else _sample_smi()


# NVML is initialized once and held for the process's life (nvmlInit/nvmlShutdown per call
# costs seconds on a busy GPU). None = not tried, False = unavailable.
_NVML = {"ready": None}
_NVML_LOCK = threading.Lock()


def _nvml_ready():
    with _NVML_LOCK:
        if _NVML["ready"] is None:
            try:
                import pynvml  # optional; not a hard dep

                pynvml.nvmlInit()
                _NVML["ready"] = pynvml
            except Exception:  # noqa: BLE001 -- optional probe
                _NVML["ready"] = False
        return _NVML["ready"]


def _sample_nvml() -> list[GpuSample]:
    pynvml = _nvml_ready()
    if not pynvml:
        return []
    try:
        samples = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            samples.append(
                GpuSample(
                    index=i,
                    uuid=_nvml_str(pynvml.nvmlDeviceGetUUID, handle),
                    mem_used=int(mem.used),
                    mem_total=int(mem.total),
                    util_pct=_nvml_call(pynvml.nvmlDeviceGetUtilizationRates, handle, "gpu", int),
                    temp_c=_nvml_call(pynvml.nvmlDeviceGetTemperature, handle, None, int, 0),
                    power_w=_nvml_call(pynvml.nvmlDeviceGetPowerUsage, handle, None, lambda v: v / 1000.0),
                )
            )
        return samples
    except Exception:  # noqa: BLE001 -- best-effort telemetry
        return []


def _nvml_str(getter, handle) -> "str | None":
    try:
        return str(getter(handle))
    except Exception:  # noqa: BLE001
        return None


def _nvml_call(getter, handle, attr, cast, *args):
    try:
        value = getter(handle, *args)
        if attr is not None:
            value = getattr(value, attr)
        return cast(value)
    except Exception:  # noqa: BLE001 -- a GPU may not expose temp/power
        return None


def _sample_smi() -> list[GpuSample]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    samples = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            samples.append(
                GpuSample(
                    index=int(parts[0]),
                    uuid=None if parts[1] in ("", "N/A") else parts[1],
                    mem_used=int(parts[2]) * (1 << 20),  # MiB
                    mem_total=int(parts[3]) * (1 << 20),
                    util_pct=_smi_int(parts[4]),
                    temp_c=_smi_int(parts[5]),
                    power_w=_smi_float(parts[6]),
                )
            )
        except ValueError:
            continue
    return samples


def _smi_int(value: str) -> "int | None":
    return int(value) if value.isascii() and value.lstrip("-").isdecimal() else None


def _smi_float(value: str) -> "float | None":
    try:
        return float(value)
    except ValueError:
        return None
