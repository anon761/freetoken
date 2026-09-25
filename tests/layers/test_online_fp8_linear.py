# Copyright (c) 2026 FreeToken contributors
# OnlineFp8LinearKernel: bf16 weight -> fp8 per-row at finalize, applied W8A16.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("M", [1, 4])
def test_online_fp8_matches_bf16(M):
    from freetoken.layers.quantization.linear.unquantized import OnlineFp8LinearKernel

    torch.manual_seed(0)
    N, K = 9000, 1024  # N spans several quantization row blocks
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    layer = SimpleNamespace(weight=w.clone(), bias=None)
    kernel = OnlineFp8LinearKernel()
    kernel.finalize(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn and layer.weight_scale.shape == (N,)
    # the stored fp8 weight reproduces the bf16 one within e4m3 rounding (<= 1/16 relative)
    deq = layer.weight.float() * layer.weight_scale[:, None]
    assert ((deq - w.float()).abs() <= w.float().abs() / 16 + 1e-6).all()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    got = kernel.apply(layer, x).float()
    ref = x.float() @ deq.t()
    rel = (got - ref).abs().max() / ref.abs().max()
    assert rel.item() < 1e-2, rel.item()


def test_mark_online_fp8_skips_routers():
    from freetoken.layers.base import BaseOP
    from freetoken.layers.quantization.linear.unquantized import (
        UnquantizedLinearMethod,
        mark_online_fp8,
    )

    def linear():
        op = BaseOP.__new__(BaseOP)
        op.quant_method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
        op.weight = torch.zeros(4, 4)
        return op

    mlp = BaseOP.__new__(BaseOP)
    mlp.gate, mlp.up = linear(), linear()
    other = BaseOP.__new__(BaseOP)
    other.quant_method = object()  # a quantized method: left alone
    other.weight = torch.zeros(4, 4)
    root = BaseOP.__new__(BaseOP)
    root.layers = [mlp, other]
    root.proj = linear()

    assert mark_online_fp8(root, skip=frozenset({"gate"})) == 2
    assert mlp.up.online_fp8 and root.proj.online_fp8
    assert not getattr(mlp.gate, "online_fp8", False)
    assert not getattr(other, "online_fp8", False)
