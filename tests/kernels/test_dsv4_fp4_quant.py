"""DSV4.1 KV quant paths against the reference math.

Two block-scale formats are in play and they are NOT interchangeable:

* the indexer K/Q use FP4 with a power-of-two (ue8m0) block scale (``fp4_act_quant``
  default);
* the compressed KV uses FP4 with an e4m3 (non-power-of-two) block scale
  (``inference/model.py:760``, ``scale_dtype=torch.float8_e4m3fn``) and the window KV
  uses FP8 e4m3 with a ue8m0 scale over block 32 (``inference/model.py:707``).

The E4M3-scale path is the one whose absence corrupted long prompts: a power-of-two
scale where the checkpoint trained with e4m3 can be up to 2x too coarse per 16-block.
These tests pin both formats to the reference (fp32 torch).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv4.fp8_linear import (
    act_quant_fp8_inplace,
    fp4_act_quant_inplace,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP4_MAX = 6.0


def _round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Signed fp4 e2m1 grid {0,.5,1,1.5,2,3,4,6}, round-to-nearest-even on magnitudes."""
    sign = torch.where(x < 0, -1.0, 1.0)
    a = x.abs()
    r = torch.where(
        a <= 0.25, torch.zeros_like(a),
        torch.where(a < 0.75, torch.full_like(a, 0.5),
        torch.where(a <= 1.25, torch.ones_like(a),
        torch.where(a < 1.75, torch.full_like(a, 1.5),
        torch.where(a <= 2.5, torch.full_like(a, 2.0),
        torch.where(a < 3.5, torch.full_like(a, 3.0),
        torch.where(a <= 5.0, torch.full_like(a, 4.0), torch.full_like(a, 6.0))))))),
    )
    return sign * r


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1] // block, block)


def _ceil_log2_pow2(v: torch.Tensor) -> torch.Tensor:
    """Exact 2**ceil(log2(v)) via the kernel's IEEE-754 bit trick (fp32, v > 0)."""
    bits = v.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    return torch.exp2((exp + (man != 0).to(torch.int32)).float())


def _ref_fp4(x: torch.Tensor, block: int, scale_dtype: torch.dtype) -> torch.Tensor:
    """Per-block FP4 quant+dequant, both scale formats, mirroring inference/kernel.py."""
    lead, N = x.shape[:-1], x.shape[-1]
    b = _blocks(x.float(), block)
    amax = b.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * (2.0 ** -9))
        s = (amax * (1.0 / FP4_MAX)).clamp_max(448.0).to(torch.float8_e4m3fn).to(torch.float32)
    else:
        amax = amax.clamp_min(FP4_MAX * (2.0 ** -126))
        s = _ceil_log2_pow2(amax * (1.0 / FP4_MAX))
    q = _round_fp4((b / s).clamp(-FP4_MAX, FP4_MAX))
    return (q * s).reshape(*lead, N)


def _ref_fp8(x: torch.Tensor, block: int) -> torch.Tensor:
    """Per-block FP8 e4m3 quant+dequant with a ue8m0 scale (reference act_quant)."""
    lead, N = x.shape[:-1], x.shape[-1]
    b = _blocks(x.float(), block)
    amax = b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = _ceil_log2_pow2(amax * (1.0 / 448.0))
    y = (b / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * s
    return y.reshape(*lead, N)


@pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float8_e8m0fnu])
def test_fp4_block16_matches_reference(scale_dtype):
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(29, 512, device="cuda", dtype=torch.bfloat16, generator=g) * 0.7
    got = fp4_act_quant_inplace(x.clone(), 16, scale_dtype).float()
    ref = _ref_fp4(x.float(), 16, scale_dtype)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_fp4_default_stays_e8m0():
    """The indexer path must not silently switch to e4m3: the two differ on most blocks."""
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    e8m0 = fp4_act_quant_inplace(x.clone(), 32)
    e4m3 = fp4_act_quant_inplace(x.clone(), 32, torch.float8_e4m3fn)
    assert not torch.equal(e8m0, e4m3)


def test_window_fp8_block32_matches_reference():
    g = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16, generator=g) * 1.5
    got = act_quant_fp8_inplace(x.clone(), 32).float()
    ref = _ref_fp8(x.float(), 32)
    assert torch.equal(got, ref), (got - ref).abs().max().item()
