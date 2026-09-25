"""Inline-dequant WNA16 (AutoRound / AutoGPTQ INT4) fused-MoE Triton kernels.

These read the packed-int4 expert cache (AutoGPTQ ``qweight`` + ``qzeros`` + fp16 group
``scales``) directly and dequantize inside the GEMM K-loop, so the grouped MoE never
materializes a BF16 copy of the experts.

Layout, for an expert weight ``W[N, K]`` with group size ``G`` (128):
  - ``qweight[slot, k//8, n]`` int32: 8 int4 codes per word, code j -> k = 8*w + j.
  - ``qzeros[slot, k//G, n//8]`` int32: 8 zero-points per word, zero for column n is
    nibble ``n % 8`` of word ``n // 8``.
  - ``scales[slot, k//G, n]`` fp16: per-group scale.
  - ``W[n, k] = (code - zero) * scales[k//G, n]``.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KW = 64  # int32 words per K-iter (covers 512 k-values)
_DECODE_WARPS = 4


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


@triton.jit
def _decode_wna16_moe_kernel(
    a_ptr,             # [M, K] activations
    qw_ptr,            # [S, K//8, N] int32
    qz_ptr,            # [S, K//G, N//8] int32
    sc_ptr,            # [S, K//G, N] fp16
    c_ptr,             # [M, TOP_K, N] output (SPLIT_K == 1)
    part_ptr,          # [SPLIT_K, M * TOP_K, N] fp32 partials (SPLIT_K > 1)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    total_routes,
    N,
    K,
    kb_per,            # K blocks (of BLOCK_SIZE_KW words) per split
    GROUP: tl.constexpr,
    stride_am, stride_ak,
    stride_qe, stride_qkw, stride_qn,
    stride_ze, stride_zg, stride_zn,
    stride_se, stride_sg, stride_sn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    split_id = tl.program_id(2)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    qw_slot = qw_ptr + slot * stride_qe
    qz_slot = qz_ptr + slot * stride_ze
    sc_slot = sc_ptr + slot * stride_se
    kb_end = tl.minimum((split_id + 1) * kb_per, tl.cdiv(K_WORDS, BLOCK_SIZE_KW))
    for kw_start in range(split_id * kb_per, kb_end):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS
        word = tl.load(
            qw_slot + widx[:, None] * stride_qkw + offs_n[None, :] * stride_qn,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        g = widx // (GROUP // 8)
        zword = tl.load(
            qz_slot + g[:, None] * stride_zg + (offs_n[None, :] // 8) * stride_zn,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        zero = ((zword >> (4 * (offs_n[None, :] % 8))) & 0xF) + 1  # GPTQ stores zero-point - 1
        scale = tl.load(
            sc_slot + g[:, None] * stride_sg + offs_n[None, :] * stride_sn,
            mask=w_mask[:, None] & n_mask[None, :], other=0.0,
        ).to(tl.float32)

        kbase = 8 * widx
        acc_w = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        for j in tl.static_range(8):
            code = (word >> (4 * j)) & 0xF
            a_j = tl.load(a_base + (kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
            acc_w += a_j[:, None] * (code - zero).to(tl.float32)
        accumulator += tl.sum(acc_w * scale, axis=0)

    if SPLIT_K > 1:
        # deterministic split-K: partials reduced (and weighted) by _decode_wna16_splitk_reduce
        p_ptrs = part_ptr + (split_id * total_routes + route_id) * N + offs_n
        tl.store(p_ptrs, accumulator, mask=(route_id < total_routes) & n_mask)
        return

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _decode_wna16_splitk_reduce(
    part_ptr,          # [SPLIT_K, total_routes, N] fp32
    c_ptr,             # [M, TOP_K, N]
    topk_weights_ptr,  # [M, TOP_K] fp32
    total_routes,
    N,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    TOP_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
    compute_type: tl.constexpr,
):
    route_id = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + (s * total_routes + route_id) * N + offs_n, mask=n_mask, other=0.0)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K
    if MUL_ROUTED_WEIGHT:
        acc = acc * tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=n_mask)


@triton.jit
def _prefill_wna16_moe_kernel(
    a_ptr,
    qw_ptr,
    qz_ptr,
    sc_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    GROUP: tl.constexpr,
    stride_am, stride_ak,
    stride_qe, stride_qkw, stride_qn,
    stride_ze, stride_zg, stride_zn,
    stride_se, stride_sg, stride_sn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    a_row = a_ptr + offs_token[:, None] // top_k * stride_am

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    qw_base = qw_ptr + slot * stride_qe + offs_bn[None, :] * stride_qn
    qz_base = qz_ptr + slot * stride_ze + (offs_bn[None, :] // 8) * stride_zn
    sc_base = sc_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_WORDS = K // 8
    for kw in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS
        word = tl.load(qw_base + widx[:, None] * stride_qkw, mask=w_mask[:, None], other=0)
        g = widx // (GROUP // 8)
        zword = tl.load(qz_base + g[:, None] * stride_zg, mask=w_mask[:, None], other=0)
        zero = ((zword >> (4 * (offs_bn[None, :] % 8))) & 0xF) + 1  # GPTQ stores zero-point - 1
        scale = tl.load(sc_base + g[:, None] * stride_sg, mask=w_mask[:, None], other=0.0).to(tl.float32)
        a_ptrs = a_row + (8 * widx)[None, :] * stride_ak
        for j in tl.static_range(8):
            code = (word >> (4 * j)) & 0xF
            b = (code - zero).to(tl.float32) * scale
            a_j = tl.load(a_ptrs, mask=token_mask[:, None] & w_mask[None, :], other=0.0)
            accumulator += tl.dot(a_j, b.to(a_j.dtype))
            a_ptrs += stride_ak

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


__all__ = ["_decode_wna16_moe_kernel", "_decode_wna16_splitk_reduce", "_prefill_wna16_moe_kernel"]
