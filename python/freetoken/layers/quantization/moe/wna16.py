"""WNA16 (AutoRound / AutoGPTQ INT4) experts: served from the offload cache through the
Triton inline-dequant kernels (packed ``qweight`` + ``qzeros`` + fp16 group ``scales``)."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import (
    BankSpec,
    ExpertView,
    gated_epilogue_reason,
    limit_or_inf,
    MoEConfig,
    MoEKernel,
    MoEMethod,
)

FP16 = torch.float16


def _fuse(pieces: dict[str, torch.Tensor], role: str) -> torch.Tensor:
    """Fuse the separate gate/up pieces of ``role`` along the OUTPUT axis.

    WNA16 banks are the native AutoGPTQ ``[K/8, N]`` per expert -- the output ``N`` is the
    LAST axis -- so the merge is a ``dim=-1`` concat (``fused_piece`` merges ``dim=1``,
    which is the packed K axis here).
    """
    suffix = role[len("gate_up"):]
    return torch.cat([pieces["gate" + suffix], pieces["up" + suffix]], dim=-1)


class TritonWna16MoEKernel(MoEKernel):
    """FreeToken's inline-dequant kernels over the native AutoGPTQ INT4 rows."""

    name = "triton"
    cpu_format = None

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=True, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        return f"triton wna16 MoE kernel: {reason}" if reason else None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        from freetoken.models.wna16_banks import wna16_tp_geometry

        # The rank's intermediate slice is group-aligned (down's K == gate/up's N).
        k_lo, k_hi = wna16_tp_geometry(cfg.intermediate, cfg.tp_size, cfg.tp_rank)
        i, h = k_hi - k_lo, cfg.hidden
        return {
            "gate_up": BankSpec((h // 8, 2 * i), torch.int32),
            "gate_up_zero": BankSpec((h // 128, 2 * i // 8), torch.int32),
            "gate_up_scale": BankSpec((h // 128, 2 * i), FP16),
            "down": BankSpec((i // 8, h), torch.int32),
            "down_zero": BankSpec((i // 128, h // 8), torch.int32),
            "down_scale": BankSpec((i // 128, h), FP16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(_fuse(pieces, "gate_up"))
        out["gate_up_zero"].copy_(_fuse(pieces, "gate_up_zero"))
        out["gate_up_scale"].copy_(_fuse(pieces, "gate_up_scale"))
        out["down"].copy_(pieces["down"])
        out["down_zero"].copy_(pieces["down_zero"])
        out["down_scale"].copy_(pieces["down_scale"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_wna16 import fused_experts_decode_wna16, fused_experts_wna16

        t = view.tensors
        banks = (t["gate_up"], t["gate_up_zero"], t["gate_up_scale"],
                 t["down"], t["down_zero"], t["down_scale"])
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            return fused_experts_wna16(
                x, *banks, topk_weights, topk_ids, view.n,
                layer.activation, layer.apply_router_weight_on_input, alpha, limit,
            )
        return fused_experts_decode_wna16(
            x, *banks, topk_weights, topk_ids,
            layer.activation, layer.apply_router_weight_on_input, alpha, limit,
        )


@register_method(QuantKind.W4A16, LayerKind.MOE)
class Wna16MoEMethod(MoEMethod):
    candidates = (TritonWna16MoEKernel,)

    def create_weights(self, layer) -> None:
        raise NotImplementedError("WNA16 experts are served from the offload cache, not resident")

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError("WNA16 experts are not resident")
