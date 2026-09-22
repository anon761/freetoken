from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _fp8_threshold_bytes() -> int:
    """Minimum reduction size for the fp8 wire format (``FREETOKEN_TP_REDUCE_FP8``).

    The e4m3 round-trip quantizes the reduced activations (~3-bit mantissa), so
    small decode-path reductions stay bf16 — their payload is latency-bound, not
    bandwidth-bound, and fp8 would cost quality for nothing. Large prefill
    reductions are bandwidth-bound and halve their wire bytes. 0 = fp8 for every
    size."""
    raw = os.environ.get("FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", "").strip()
    return int(raw) if raw else 64 * 1024


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator
    # The wire format is fixed for the process and negotiated group-wide in
    # enable_pynccl_distributed: a per-rank mismatch (fp8's uint8 all_gather vs a
    # plain all_reduce) issues a different NCCL collective and hangs the group.
    # fp8_reduce halves the wire bytes for reductions >= fp8_min_bytes (0 = fp8
    # for every size).
    fp8_reduce: bool = False
    fp8_min_bytes: int = 64 * 1024

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.fp8_reduce
            and x.dtype == torch.bfloat16
            and x.numel() * x.element_size() >= self.fp8_min_bytes
        ):
            # Quantized-wire all_reduce: NCCL rejects fp8 REDUCTIONS before sm90
            # (NCCL: "FP8 reduction support begins with sm90"), but all_gather is
            # a plain copy and works everywhere. Quantize each rank's chunk to
            # e4m3 (1 byte/elem instead of bf16's 2), gather every rank's chunk,
            # sum in fp32 locally — half the wire bytes of a bf16 all_reduce, and
            # the reduction itself is exact. Round-trip quantization error ~e4m3
            # (~3-bit mantissa) per input element.
            from .info import get_tp_info

            size = get_tp_info().size
            flat = x.view(-1)
            # transport as raw bytes (uint8 view) — NCCL's fp8 collectives are
            # sm90-gated; a byte copy has no such restriction
            q = flat.to(torch.float8_e4m3fn).view(torch.uint8)
            gathered = self.all_gather(q)
            acc = (
                gathered.view(size, -1)
                .view(torch.float8_e4m3fn)
                .to(torch.float32)
                .sum(0)
            )
            x.copy_(acc.view(*x.shape))
            return x
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def _agree_flag(
    tp_info: DistributedInfo, group: torch.distributed.ProcessGroup, name: str
) -> bool:
    """Group-wide boolean env flag: True only when every rank set it. A mismatch
    (some set, some not) changes the NCCL collective the fp8 path issues and hangs
    the group, so it is a hard error instead of a silent per-rank choice."""
    local = 1 if _env_flag(name) else 0
    t = torch.tensor([local], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    total = int(t.item())
    if total not in (0, tp_info.size):
        raise RuntimeError(
            f"{name} differs across TP ranks ({total}/{tp_info.size} set) — the "
            "wire format must be uniform or NCCL collectives mismatch and hang. "
            "Set it on every rank (or none)."
        )
    return total == tp_info.size


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    # The wire format is a group-wide decision — agree before any reduction.
    # FREETOKEN_TP_REDUCE_FP8 must be uniform (hard error otherwise), and the
    # threshold is reduced to the group minimum so every rank takes the same
    # branch for a given size.
    fp8_reduce = _agree_flag(tp_info, tp_cpu_group, "FREETOKEN_TP_REDUCE_FP8")
    fp8_min_bytes = _fp8_threshold_bytes()
    if fp8_reduce:
        thr = torch.tensor([fp8_min_bytes], dtype=torch.int64)
        dist.all_reduce(thr, op=dist.ReduceOp.MIN, group=tp_cpu_group)
        fp8_min_bytes = int(thr.item())

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(
        PyNCCLDistributedImpl(comm, fp8_reduce=fp8_reduce, fp8_min_bytes=fp8_min_bytes)
    )


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
