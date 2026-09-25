# Copyright (c) 2026 FreeToken contributors
# moe/fused_wna16.py decode path (route-parallel INT4 GEMM, split-K) against a torch
# dequant reference of the whole routed MoE.
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

G = 128


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., R*8, C] int4 codes -> [..., R, C] int32 (code j of a word -> row 8w+j)."""
    *lead, rows, cols = codes.shape
    c = codes.view(*lead, rows // 8, 8, cols).to(torch.int64)
    word = sum(c[..., j, :] << (4 * j) for j in range(8))
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)  # two's complement


def _expert_bank(S, N, K, dev):
    codes = torch.randint(0, 16, (S, K, N), device=dev)
    zeros = torch.randint(0, 15, (S, K // G, N), device=dev)  # stored zero-point - 1
    scales = (torch.rand(S, K // G, N, device=dev) * 0.02 + 0.001).half()
    qw = _pack(codes)
    qz = _pack(zeros.transpose(1, 2)).transpose(1, 2).contiguous()  # [S, K//G, N//8]
    w = (codes - (zeros + 1).repeat_interleave(G, dim=1)).float() * scales.float().repeat_interleave(G, dim=1)
    return qw, qz, scales, w.transpose(1, 2)  # dequant [S, N, K]


@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("split", [None, 1])
def test_decode_matches_dequant_reference(M, split, monkeypatch):
    from freetoken.moe import fused_wna16

    if split == 1:  # pin the pre-split tiles to cover the SPLIT_K=1 path too
        monkeypatch.setattr(fused_wna16, "_decode_config", lambda r, n, k: dict(
            BLOCK_SIZE_N=64, BLOCK_SIZE_KW=8, num_warps=1, num_stages=2, SPLIT_K=1))
    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK = 16, 512, 256, 4
    gu_qw, gu_qz, gu_sc, gu_w = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, dn_w = _expert_bank(S, H, I, dev)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(S, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    out = fused_wna16.fused_experts_decode_wna16(
        x.clone(), gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc, tw, ids
    ).float()

    ref = torch.zeros(M, H, device=dev)
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j])
            h = gu_w[e] @ x[m].float()
            act = torch.nn.functional.silu(h[:I]) * h[I:]
            ref[m] += tw[m, j] * (dn_w[e] @ act.to(torch.bfloat16).float())
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel.item() < 2e-2, rel.item()
