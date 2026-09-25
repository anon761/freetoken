"""Host orchestration for the inline-dequant WNA16 (INT4) fused-MoE path.

Mirrors :mod:`freetoken.moe.fused_nvfp4` (gemm1 -> act -> gemm2 -> sum-reduce) but the
grouped GEMMs read the packed-int4 expert cache (AutoGPTQ ``qweight``/``qzeros``/fp16
``scales``) directly and dequantize inside the K-loop.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import triton

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.wna16_fused_moe import (
    _decode_wna16_moe_kernel,
    _decode_wna16_splitk_reduce,
    _prefill_wna16_moe_kernel,
    _tl_dtype,
)
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size

GROUP = 128


_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KW = 8
_DECODE_TARGET_PROGRAMS = 1600


def _decode_config(routes: int, N: int, K: int) -> Dict[str, int]:
    """Tiles for the route-parallel decode kernel (M x top_k routes, one expert each).

    A decode step has few routes, so narrow single-warp programs plus a deterministic
    split-K fill the SMs; the split grows until ~_DECODE_TARGET_PROGRAMS. RTX 3090,
    Qwen3.8-Flash-Next per rank (gate_up 640x2560, down 2560x320), M=1 / M=4: gate_up
    43 -> 19 / 170 -> 56 us, down 28 -> 10 / 91 -> 29 us."""
    programs = routes * triton.cdiv(N, _DECODE_BLOCK_N)
    n_kb = triton.cdiv(K // 8, _DECODE_BLOCK_KW)
    split = max(1, min(8, n_kb, _DECODE_TARGET_PROGRAMS // max(1, programs)))
    split = 1 << (split.bit_length() - 1)
    return dict(BLOCK_SIZE_N=_DECODE_BLOCK_N, BLOCK_SIZE_KW=_DECODE_BLOCK_KW,
                num_warps=1, num_stages=2, SPLIT_K=split)


def _decode_gemm(
    a: torch.Tensor,
    qw: torch.Tensor,
    qz: torch.Tensor,
    sc: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    cfg: Dict[str, int] | None = None,
) -> None:
    M, top_k = topk_ids.shape
    N = qw.shape[2]
    K = qw.shape[1] * 8
    total_routes = M * top_k
    cfg = dict(cfg or _decode_config(total_routes, N, K))
    split = cfg.pop("SPLIT_K")
    n_kb = triton.cdiv(K // 8, cfg["BLOCK_SIZE_KW"])
    split = max(1, min(split, n_kb))
    kb_per = triton.cdiv(n_kb, split)
    part = (torch.empty((split, total_routes, N), device=a.device, dtype=torch.float32)
            if split > 1 else c)  # unused dummy when split == 1
    grid = (total_routes, triton.cdiv(N, cfg["BLOCK_SIZE_N"]), split)
    _decode_wna16_moe_kernel[grid](
        a, qw, qz, sc, c, part, topk_weights, topk_ids,
        total_routes, N, K, kb_per, GROUP,
        a.stride(0), a.stride(1),
        qw.stride(0), qw.stride(1), qw.stride(2),
        qz.stride(0), qz.stride(1), qz.stride(2),
        sc.stride(0), sc.stride(1), sc.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        SPLIT_K=split,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )
    if split > 1:
        _decode_wna16_splitk_reduce[(total_routes, triton.cdiv(N, 256))](
            part, c, topk_weights, total_routes, N,
            c.stride(0), c.stride(1), c.stride(2),
            topk_weights.stride(0), topk_weights.stride(1),
            TOP_K=top_k, SPLIT_K=split, MUL_ROUTED_WEIGHT=mul_routed_weight,
            BLOCK=256, compute_type=_tl_dtype(c.dtype), num_warps=2,
        )


def _prefill_config(M: int) -> Dict[str, int]:
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KW=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KW=32,
                GROUP_SIZE_M=8, num_warps=8, num_stages=4)


def _prefill_gemm(
    a: torch.Tensor,
    qw: torch.Tensor,
    qz: torch.Tensor,
    sc: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = qw.shape[2]
    K = qw.shape[1] * 8
    EM = sorted_ids.shape[0]
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_wna16_moe_kernel[grid](
        a, qw, qz, sc, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        N, K, EM, num_valid_tokens, GROUP,
        a.stride(0), a.stride(1),
        qw.stride(0), qw.stride(1), qw.stride(2),
        qz.stride(0), qz.stride(1), qz.stride(2),
        sc.stride(0), sc.stride(1), sc.stride(2),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def _banks(args):
    (hidden_states, gate_up_qw, gate_up_qz, gate_up_sc, down_qw, down_qz, down_sc) = args[:7]
    return hidden_states, gate_up_qw, gate_up_qz, gate_up_sc, down_qw, down_qz, down_sc


def fused_experts_decode_wna16(
    hidden_states: torch.Tensor,
    gate_up_qw: torch.Tensor,
    gate_up_qz: torch.Tensor,
    gate_up_sc: torch.Tensor,
    down_qw: torch.Tensor,
    down_qz: torch.Tensor,
    down_sc: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_qw.shape[2]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _decode_gemm(hidden_states, gate_up_qw, gate_up_qz, gate_up_sc, ic1,
                 topk_weights, topk_ids, apply_router_weight_on_input, False)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _decode_gemm(ic2, down_qw, down_qz, down_sc, ic3,
                 topk_weights, topk_ids, not apply_router_weight_on_input, True)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_wna16(
    hidden_states: torch.Tensor,
    gate_up_qw: torch.Tensor,
    gate_up_qz: torch.Tensor,
    gate_up_sc: torch.Tensor,
    down_qw: torch.Tensor,
    down_qz: torch.Tensor,
    down_sc: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_qw.shape[2]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(hidden_states, gate_up_qw, gate_up_qz, gate_up_sc, ic1,
                  tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
                  apply_router_weight_on_input, cfg)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(ic2, down_qw, down_qz, down_sc, ic3,
                  tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
                  not apply_router_weight_on_input, cfg)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = ["fused_experts_decode_wna16", "fused_experts_wna16"]
