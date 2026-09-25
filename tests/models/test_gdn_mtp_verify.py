# Copyright (c) 2026 FreeToken contributors
# MTP GDN verify + accepted-prefix commit (models/qwen3_5_moe/gdn_verify.py): the verify
# outputs and the committed conv/recurrent state must equal step-at-a-time decode exactly.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_KM1 = 3


def _layer(hk=2, hv=4, d=64):
    key_dim, value_dim = hk * d, hv * d
    conv_dim = 2 * key_dim + value_dim
    weight = torch.randn(conv_dim, _KM1 + 1, dtype=torch.bfloat16, device="cuda")
    layer = SimpleNamespace(
        layer_id=0, local_k_heads=hk, local_v_heads=hv, head_k_dim=d, head_v_dim=d,
        local_key_dim=key_dim, local_value_dim=value_dim, local_conv_dim=conv_dim,
        A_log=torch.rand(hv, device="cuda"), dt_bias=torch.rand(hv, device="cuda"),
        _mtp_verify=None,
    )
    layer._conv_weight = lambda: weight
    return layer


def _pool(conv_states, recurrent_states):
    return SimpleNamespace(
        conv_states=conv_states, recurrent_states=recurrent_states, local_index=lambda _: 0
    )


def _decode_steps(layer, pool, conv_in, a, b, slots, steps):
    """Reference: ``steps`` single-token decode forwards per request (conv + recurrence)."""
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    n = slots.shape[0]
    cu = torch.arange(n + 1, dtype=torch.int32, device="cuda")
    outs = []
    for j in range(steps):
        mixed = causal_conv1d_decode(
            conv_in[:, j].contiguous(), pool.conv_states[0], layer._conv_weight(), slots
        )
        qf, kf, vf = torch.split(
            mixed, [layer.local_key_dim, layer.local_key_dim, layer.local_value_dim], dim=-1
        )
        o = gdn_decode_fla(
            qf.reshape(1, n, layer.local_k_heads, layer.head_k_dim),
            kf.reshape(1, n, layer.local_k_heads, layer.head_k_dim),
            vf.reshape(1, n, layer.local_v_heads, layer.head_v_dim),
            a[:, j].contiguous(), b[:, j].contiguous(), A_log=layer.A_log,
            dt_bias=layer.dt_bias, state_source=pool.recurrent_states[0], indices=slots,
            cu_seqlens=cu, scale=layer.head_k_dim ** -0.5,
        )
        outs.append(o.reshape(n, -1))
    return torch.stack(outs, 1)


@pytest.mark.parametrize("lens", [[1, 4, 2], [4, 4, 4], [3, 1, 1]])
def test_verify_and_commit_match_decode(lens):
    from freetoken.models.qwen3_5_moe.gdn_verify import gdn_verify_commit, gdn_verify_forward

    torch.manual_seed(0)
    n, t = 3, 4
    layer = _layer()
    conv0 = torch.randn(1, 8, layer.local_conv_dim, _KM1, dtype=torch.bfloat16, device="cuda")
    rec0 = torch.randn(1, 8, layer.local_v_heads, layer.head_k_dim, layer.head_v_dim, device="cuda") * 0.1
    pool = _pool(conv0.clone(), rec0.clone())
    slots = torch.tensor([5, 1, 6], dtype=torch.int32, device="cuda")
    conv_in = torch.randn(n, t, layer.local_conv_dim, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(n, t, layer.local_v_heads, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, t, layer.local_v_heads, dtype=torch.bfloat16, device="cuda")

    out = gdn_verify_forward(
        layer, conv_in.reshape(n * t, -1), a.reshape(n * t, -1), b.reshape(n * t, -1),
        pool, slots, n, t, torch.bfloat16,
    )
    ref_out = _decode_steps(layer, _pool(conv0.clone(), rec0.clone()), conv_in, a, b, slots, t)
    assert torch.equal(out.reshape(n, t, -1), ref_out)
    assert torch.equal(pool.recurrent_states, rec0), "verify must not write the live SSM state"

    gdn_verify_commit(layer, pool, slots, torch.tensor(lens, dtype=torch.int32, device="cuda"))
    for i, L in enumerate(lens):
        one = slots[i : i + 1]
        ref = _pool(conv0.clone(), rec0.clone())
        _decode_steps(layer, ref, conv_in[i : i + 1], a[i : i + 1], b[i : i + 1], one, L)
        s = one.long()
        assert torch.equal(pool.conv_states[0, s], ref.conv_states[0, s]), f"conv req{i} L={L}"
        assert torch.equal(pool.recurrent_states[0, s], ref.recurrent_states[0, s]), f"ssm req{i} L={L}"
    untouched = [j for j in range(8) if j not in (5, 1, 6)]
    assert torch.equal(pool.conv_states[0, untouched], conv0[0, untouched])
    assert torch.equal(pool.recurrent_states[0, untouched], rec0[0, untouched])
