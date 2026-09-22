"""Manifold-constrained Hyper-Connections for DeepSeek-V4.1.

Same math family as deepseek_v4 (hc_mult=4, Sinkhorn-doubly-stochastic combine),
with the V4.1 *delayed pre* semantics: a sublayer collapses its input with the
PREVIOUS sublayer's pre mix, and its own pre mix is handed to the next sublayer
(vLLM reference: model.py carries ``pre_mix`` through the layer loop; the first
layer sees ``pre_mix=None`` and its input is the raw embedding). There is no
learned ``hc_head`` — the model exits by collapsing with the last FFN pre mix.

Stream layout here is ``[M, hc, hidden]`` (M = tokens), fp32 mixes / bf16
stores. Weight names live on the Block (``hc_attn_fn/base/scale``,
``hc_ffn_fn/base/scale``) to match the checkpoint exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn


def hc_mixes(
    R: torch.Tensor, hc_fn: torch.Tensor, norm_eps: float
) -> torch.Tensor:
    """Coefficient logits for one sublayer: full-stream RMS-normalized GEMM.

    R [M, hc, dim] (bf16) -> mixes [M, (2+hc)*hc] fp32.
    """
    xf = R.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    return F.linear(xf, hc_fn) * rsqrt


def hc_split(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes [M, (2+hc)*hc] -> (pre [M, hc], post [M, hc], comb [M, hc, hc]).

    pre = sigmoid(+eps); post = 2*sigmoid (post_mult 2.0); comb = Sinkhorn-
    iterated softmax. Identical to the DSV4 split — the manifold math did not
    change between V4-Flash and V4.1.
    """
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    return pre, post, comb


def hc_collapse(
    R: torch.Tensor, pre_mix: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Delayed pre collapse: x[m] = sum_h pre_mix[m, h] * R[m, h, :] -> [M, dim]."""
    M, hc, dim = R.shape
    return hc_pre_combine(R.reshape(M, hc, dim), pre_mix.reshape(M, hc), dtype)


def hc_add(
    y: torch.Tensor, R: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Residual update: R'[m, h] = post[m, h] * y[m] + sum_p comb[m, h, p] * R[m, p]."""
    M, hc, dim = R.shape
    return hc_post_combine(y.reshape(M, dim), R.reshape(M, hc, dim), post, comb)


__all__ = ["hc_add", "hc_collapse", "hc_mixes", "hc_split"]
