"""MTP verify + accepted-prefix commit for a GatedDeltaNet layer (shared by qwen3_5_moe and
qwen4_exp).

The verify runs the ``t = k+1`` tokens of each request through the SAME kernels a
step-at-a-time decode uses, so its outputs are bit-identical to plain decode:

* conv: the multi-token ``causal_conv1d_update`` over a ``[n, C, t]`` view, written in
  place into a persistent per-layer buffer (``mixed``) that later feeds the commit;
* SSM: the fused recurrent kernel with ``disable_state_update`` -- the live recurrent
  state is left untouched.

The conv update does advance the live conv slot by all ``t`` tokens, so the verify first
stashes each slot's pre-verify window next to the raw conv inputs (``conv_win`` =
``[pre-state | raw inputs]``). The commit then rewrites the conv slot from that window and
advances the recurrent state over each request's accepted prefix ``L`` with one kernel
call (``num_steps``). ``L`` and the slots are device tensors, so one commit graph per batch
size covers every accepted count.
"""

from __future__ import annotations

import torch
from freetoken.kernel.causal_conv1d import causal_conv1d_update_multi

from .gdn_kernels import gdn_commit_recurrent, gdn_verify_recurrent


class GDNVerifyBuffers:
    """Per-layer static buffers of the last verify forward (sized once for the largest
    ``n``; a CUDA graph captures their addresses, so they are never reallocated smaller)."""

    def __init__(self, n: int, t: int, conv_dim: int, km1: int, v_heads: int, dtype, device):
        self.n = n
        self.t = t
        self.km1 = km1
        self.mixed = torch.empty(n, t, conv_dim, dtype=dtype, device=device)
        self.conv_win = torch.empty(n, conv_dim, km1 + t, dtype=dtype, device=device)
        self.a = torch.empty(n, t, v_heads, dtype=dtype, device=device)
        self.b = torch.empty(n, t, v_heads, dtype=dtype, device=device)


def ensure_verify_buffers(layer, n: int, t: int, dtype, device, km1: int) -> GDNVerifyBuffers:
    buf = layer._mtp_verify
    if buf is not None and buf.n >= n and buf.t == t:
        return buf
    assert buf is None or buf.t == t, f"MTP verify width changed ({buf.t} -> {t})"
    buf = GDNVerifyBuffers(
        max(n, 1), t, layer.local_conv_dim, km1, layer.local_v_heads, dtype, device
    )
    layer._mtp_verify = buf
    return buf


def gdn_verify_forward(layer, conv_in, a, b, pool, slots, n: int, t: int, dtype) -> torch.Tensor:
    """Verify-batch core of a GDN layer: ``conv_in`` [n*t, C] raw conv inputs, ``a``/``b``
    [n*t, HV] raw gates, ``slots`` [n] int32 live GDN slots. Returns ``[n*t, HV, V]``."""
    li = pool.local_index(layer.layer_id)
    conv_state = pool.conv_states[li]
    km1 = conv_state.shape[-1]
    buf = ensure_verify_buffers(layer, n, t, dtype, conv_in.device, km1)
    raw = conv_in.reshape(n, t, -1)
    buf.conv_win[:n, :, :km1].copy_(conv_state[slots.long()])
    buf.conv_win[:n, :, km1:].copy_(raw.transpose(1, 2))
    mixed = buf.mixed[:n]
    mixed.copy_(raw)
    causal_conv1d_update_multi(mixed.transpose(1, 2), conv_state, layer._conv_weight(), slots)
    buf.a[:n].copy_(a.reshape(n, t, -1))
    buf.b[:n].copy_(b.reshape(n, t, -1))
    q, k, v = _split_qkv(layer, mixed)
    o = gdn_verify_recurrent(
        q, k, v, buf.a[:n], buf.b[:n], A_log=layer.A_log, dt_bias=layer.dt_bias,
        state_source=pool.recurrent_states[li], indices=slots,
        scale=layer.head_k_dim ** -0.5,
    )
    return o.reshape(n * t, layer.local_v_heads, layer.head_v_dim)


def gdn_verify_commit(layer, pool, slots: torch.Tensor, lens: torch.Tensor) -> None:
    """Advance the live conv + recurrent state of ``slots`` over each request's accepted
    prefix ``lens`` (device int32, 1 <= L <= t) from the last verify's buffers."""
    buf = layer._mtp_verify
    if buf is None:
        return
    n = slots.shape[0]
    li = pool.local_index(layer.layer_id)
    km1 = buf.km1
    # conv slot = the km1 raw inputs ending at the last accepted token: columns [L, L+km1)
    # of [pre-state | raw inputs].
    cols = lens.long().view(n, 1, 1) + torch.arange(km1, device=lens.device).view(1, 1, km1)
    win = buf.conv_win[:n]
    pool.conv_states[li].index_copy_(
        0, slots.long(), win.gather(2, cols.expand(n, win.shape[1], km1))
    )
    q, k, v = _split_qkv(layer, buf.mixed[:n])
    gdn_commit_recurrent(
        q, k, v, buf.a[:n], buf.b[:n], A_log=layer.A_log, dt_bias=layer.dt_bias,
        state_source=pool.recurrent_states[li], indices=slots, num_steps=lens,
        scale=layer.head_k_dim ** -0.5,
    )


def _split_qkv(layer, mixed: torch.Tensor):
    """``mixed`` [n, t, C] post-conv activations -> q/k [n, t, Hk, K], v [n, t, Hv, V] views."""
    n, t, _ = mixed.shape
    qf, kf, vf = torch.split(
        mixed, [layer.local_key_dim, layer.local_key_dim, layer.local_value_dim], dim=-1
    )
    return (
        qf.reshape(n, t, layer.local_k_heads, layer.head_k_dim),
        kf.reshape(n, t, layer.local_k_heads, layer.head_k_dim),
        vf.reshape(n, t, layer.local_v_heads, layer.head_v_dim),
    )


__all__ = ["GDNVerifyBuffers", "gdn_verify_commit", "gdn_verify_forward"]
