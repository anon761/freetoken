"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        return F.linear(x, w, b)


class OnlineFp8LinearKernel(LinearKernel):
    """A bf16 checkpoint weight quantized to fp8-e4m3 per output row at load and run W8A16:
    half the weight bytes for the bandwidth-bound single-token GEMMs of the MTP draft head
    (only drafts are affected; the target verifies every one). Opt-in per layer via
    ``layer.online_fp8``."""

    name = "online-fp8"
    _ROWS = 4096  # quantize in row blocks: no full fp32 copy of a large weight

    def finalize(self, layer: Any) -> None:
        w = layer.weight
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        for r in range(0, w.shape[0], self._ROWS):
            blk = w[r : r + self._ROWS].float()
            s = (blk.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            q[r : r + self._ROWS] = (blk / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            scale[r : r + self._ROWS] = s
        layer.weight = q
        layer.weight_scale = scale

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        return fp8_pertensor_linear(x, layer.weight, layer.weight_scale, layer.bias)


def mark_online_fp8(root: Any, skip: frozenset = frozenset()) -> int:
    """Flag every bf16 linear under ``root`` for load-time fp8 quantization (see
    OnlineFp8LinearKernel); ``skip`` names attributes whose subtree stays bf16 (routers).
    Must run before ``finalize_quant``. Returns the number of flagged layers."""
    from freetoken.layers.base import BaseOP

    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        if isinstance(getattr(op, "quant_method", None), UnquantizedLinearMethod) and getattr(op, "weight", None) is not None:
            op.online_fp8 = True
            count += 1
        for name, value in vars(op).items():
            if name in skip:
                continue
            if isinstance(value, BaseOP):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, BaseOP))
    return count


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)

    def finalize(self, layer: Any) -> None:
        if getattr(layer, "online_fp8", False):
            self.kernel = OnlineFp8LinearKernel()
        self.kernel.finalize(layer)
