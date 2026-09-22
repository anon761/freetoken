"""Checkpoint key inventory + quant-storage normalization (the checkpoint-format layer).

One source of truth for "what is actually on disk", so the per-family weight readers
stop hardcoding a single exporter's naming. Reads the safetensors headers ONCE into
``name -> TensorInfo(shard, dtype, shape)`` (cheap: headers only) and answers the
questions every reader needs:

* does a companion tensor exist, under ANY known dialect suffix
  (``.weight_scale`` per-row fp8 / nvfp4 block, ``.weight_scale_inv`` block-fp8,
  ``.weight_scale_2`` / ``.weight_global_scale`` nvfp4 global, ``.input_scale``)?
* what storage form does one weight use (bf16 / fp8-per-row / fp8-block / nvfp4),
  decided from the weight dtype + which companions exist -- not from the exporter name?
* how do we dequantize that form to bf16?

Expert/MTP layout normalization (stacked vs per-expert) builds on the same inventory.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import torch

from freetoken.utils import download_hf_weight

_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Logical role -> checkpoint suffix candidates across dialects (order = preference).
# ``weight`` itself is the bare base name; these are the companions around it.
ROLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "weight_scale": (".weight_scale",),
    "weight_global": (".weight_scale_2", ".weight_global_scale"),
    "block_scale": (".weight_scale_inv",),
    "input_scale": (".input_scale",),
}
_ANY_SCALE_SUFFIXES = tuple(
    s for suffixes in ROLE_SUFFIXES.values() for s in suffixes
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _weight_files(folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    return [f for f in files if not f.endswith("consolidated.safetensors")] or files


class CheckpointIndex:
    """Every tensor of a checkpoint by name, with its shard/dtype/shape."""

    def __init__(self, folder: str, entries: dict[str, TensorInfo]):
        self.folder = folder
        self._entries = entries

    @classmethod
    def load(cls, model_path: str) -> "CheckpointIndex":
        folder = download_hf_weight(model_path)
        entries: dict[str, TensorInfo] = {}
        for path in _weight_files(folder):
            header, _ = _read_header(path)
            shard = os.path.basename(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                entries[name] = TensorInfo(
                    name=name, shard=shard, dtype=meta["dtype"],
                    shape=tuple(meta.get("shape") or ()),
                )
        return cls(folder, entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> "frozenset[str]":
        return frozenset(self._entries)

    def info(self, name: str) -> TensorInfo:
        return self._entries[name]

    def shard(self, name: str) -> str:
        return self._entries[name].shard

    def dtype(self, name: str) -> torch.dtype:
        return _ST_DTYPE[self._entries[name].dtype]

    def shape(self, name: str) -> tuple[int, ...]:
        return self._entries[name].shape

    def companion(self, base: str, role: str) -> str | None:
        """The checkpoint key feeding ``role`` for weight ``base``, or None."""
        for suffix in ROLE_SUFFIXES[role]:
            if base + suffix in self._entries:
                return base + suffix
        return None

    def scale_companion(self, base: str) -> str | None:
        """Any scale sibling of ``base`` (per-row/nvfp4/block/global), or None."""
        for suffix in _ANY_SCALE_SUFFIXES:
            if base + suffix in self._entries:
                return base + suffix
        return None

    def is_scale(self, name: str) -> bool:
        return name.endswith(_ANY_SCALE_SUFFIXES)


class TensorReader:
    """Serves checkpoint tensors by name on ``device``, opening each shard once."""

    def __init__(self, index: CheckpointIndex, device: str = "cpu"):
        self._index = index
        self._device = device
        self._handles: dict = {}

    def get(self, name: str) -> torch.Tensor:
        import safetensors

        shard = self._index.shard(name)
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._index.folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()



class StorageKind(Enum):
    BF16 = "bf16"
    FP8_PER_ROW = "fp8_per_row"
    FP8_BLOCK = "fp8_block"
    NVFP4 = "nvfp4"


def detect_storage(index: CheckpointIndex, base: str) -> StorageKind:
    """The on-disk storage form of weight ``base``, from its dtype + companions.

    ``base`` is the weight key without any suffix (e.g. ``...q_proj`` for
    ``...q_proj.weight``). Only the ``.weight`` key is inspected for dtype; the
    companion presence decides between the fp8/nvfp4 dialects."""
    weight = base + ".weight"
    if weight not in index:
        raise KeyError(f"{weight!r} not in checkpoint")
    dtype = index.dtype(weight)
    has_global = index.companion(base, "weight_global") is not None
    has_block = index.companion(base, "block_scale") is not None
    has_per_row = index.companion(base, "weight_scale") is not None
    if dtype is torch.uint8 and (has_global or has_per_row):
        return StorageKind.NVFP4
    if dtype in FP8_DTYPES:
        if has_block:
            return StorageKind.FP8_BLOCK
        if has_per_row:
            return StorageKind.FP8_PER_ROW
    return StorageKind.BF16


def dequantize_fp8_per_row(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 weight + per-output-row (or per-element) scale -> bf16."""
    scale = scale.to(torch.float32)
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def dequantize_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """block-fp8 weight + ``[out/block, in/block]`` scale -> bf16.

    Supports a separate block size per axis (DeepSeek-V3 style is 128x128; V4.1
    uses 32x32 via the scheme, callers pass it)."""
    out, inn = weight.shape
    bn = scale_inv.shape[0]
    bk = scale_inv.shape[1]
    block_n = out // bn
    block_k = inn // bk
    w = weight.to(torch.float32).reshape(bn, block_n, bk, block_k)
    s = scale_inv.to(torch.float32).reshape(bn, 1, bk, 1)
    return (w * s).reshape(out, inn).to(torch.bfloat16)


def dequantize(
    index: CheckpointIndex, base: str, get, *, block: int = 128
) -> torch.Tensor:
    """Read + dequantize ``base``'s weight to bf16, generically across dialects.

    ``get(name)`` serves a checkpoint tensor by key. NVFP4 has no bf16 target here
    (its consumers keep it packed), so it raises rather than silently inflating."""
    kind = detect_storage(index, base)
    weight = get(base + ".weight")
    if kind is StorageKind.BF16:
        return weight
    if kind is StorageKind.FP8_PER_ROW:
        scale = get(index.companion(base, "weight_scale"))
        return dequantize_fp8_per_row(weight, scale)
    if kind is StorageKind.FP8_BLOCK:
        scale = get(index.companion(base, "block_scale"))
        return dequantize_fp8_block(weight, scale, block=block)
    raise ValueError(f"{base!r} is stored NVFP4 packed; it has no bf16 dequant path")


__all__ = [
    "CheckpointIndex",
    "FP8_DTYPES",
    "ROLE_SUFFIXES",
    "StorageKind",
    "TensorInfo",
    "TensorReader",
    "dequantize",
    "dequantize_fp8_block",
    "dequantize_fp8_per_row",
    "detect_storage",
]
