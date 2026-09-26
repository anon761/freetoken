import torch

from freetoken.models.deepseek_v41 import hc


def test_hc_mixes_micro_batches_match_one_pass(monkeypatch):
    torch.manual_seed(0)
    R = torch.randn(37, 4, 16).to(torch.bfloat16)
    hc_fn = torch.randn(24, 64)
    whole = hc._hc_mixes_slice(R, hc_fn, 1e-6)
    monkeypatch.setattr(hc, "_MIXES_MICRO_BS", 8)
    sliced = hc.hc_mixes(R, hc_fn, 1e-6)
    assert sliced.dtype == torch.float32 and sliced.shape == (37, 24)
    # the fp32 GEMM may block a slice differently: equal up to fp32 rounding
    torch.testing.assert_close(sliced, whole, rtol=1e-5, atol=1e-5)
