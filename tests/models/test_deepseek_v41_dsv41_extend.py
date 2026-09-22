"""V41Compressor arbitrary-start extend: the open group seeded from the ring must
merge with the first new token so the first group closes at the RIGHT boundary.

CPU only: the paged ring and the Triton `gated_pool` are replaced by a torch reference
that mirrors the decode path's carry semantics.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Context, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.deepseek_v41.compress import V41Compressor


def _tp1():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


class _FakeLinear:
    def forward(self, x):
        return x


class _FakeBackend:
    device = torch.device("cpu")

    def compress_pool(self, layer_id, tier):
        return torch.zeros(1)


def _make(ratio: int) -> V41Compressor:
    _tp1()
    import freetoken.core as core

    core._GLOBAL_CTX = None
    set_global_ctx(Context(page_size=1))
    get_global_ctx().attn_backend = _FakeBackend()
    args = SimpleNamespace(hidden_size=4, qk_rope_head_dim=2, max_batch_size=1, norm_eps=1e-6)
    c = V41Compressor(args, ratio, head_dim=4)
    c.wkv = _FakeLinear()
    c.wgate = _FakeLinear() if ratio > 1 else None
    c.norm = SimpleNamespace(forward=lambda t: t)
    c.layer_id, c.tier = 0, "attn"
    c._kv_state = torch.zeros(1, ratio, 4)
    c._score_state = torch.full((1, ratio, 4), float("-inf"))
    # monkeypatch the paged ring + boundary writes (not under test)
    c._seed_carry_from_ring = lambda slot: None
    c._write_boundary_carries_range = lambda *a, **k: None
    c._write_through_carry = lambda slot: None
    return c


def _ref_pool(ks: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
    return (ks * ss.softmax(dim=0)).sum(0)


def _expected(seed_kv: torch.Tensor, seed_ss: torch.Tensor, xs: torch.Tensor, start_pos: int, ratio: int):
    """Torch reference: scatter each token into the open group; emit at group close."""
    ks, ss = seed_kv.clone(), seed_ss.clone()
    latents = []
    for j in range(xs.size(1)):
        pos = start_pos + j
        ks[pos % ratio] = xs[0, j]
        ss[pos % ratio] = xs[0, j]
        if (pos + 1) % ratio == 0:
            latents.append(_ref_pool(ks, ss))
    out = torch.stack(latents, dim=0).unsqueeze(0) if latents else xs.new_zeros(1, 0, 4)
    return out, ks, ss


def _run(c: V41Compressor, x: torch.Tensor, start_pos: int, seed_kv, seed_ss):
    import freetoken.models.deepseek_v41.compress as mod

    orig = mod.gated_pool
    mod.gated_pool = lambda ks, ss, dtype: _ref_pool(ks[0], ss[0]).unsqueeze(0)
    try:
        c._kv_state.copy_(seed_kv.unsqueeze(0))
        c._score_state.copy_(seed_ss.unsqueeze(0))
        capture: list = []
        slots = torch.arange(start_pos, start_pos + x.size(1))
        latent = c.compute_extend(x, start_pos, slots, tail_window_slot=0, capture=capture)
        return latent, c._kv_state[0].clone(), c._score_state[0].clone(), capture
    finally:
        mod.gated_pool = orig


def test_gated_extend_arbitrary_start_merges_open_group():
    """start_pos=5, ratio=2: the seeded row 0 (pos 4) closes with new token 5 -> group
    [4,5], then [6,7]; token 8 stays open. A naive grouping from index 0 would emit
    [5,6]/[7,8] -- wrong."""
    c = _make(2)
    xs = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 4)  # tokens 5,6,7,8
    seed_kv = torch.zeros(2, 4)
    seed_ss = torch.full((2, 4), float("-inf"))
    seed_kv[0] = 100.0
    seed_ss[0] = 100.0
    ref_lat, ref_ks, ref_ss = _expected(seed_kv, seed_ss, xs, 5, 2)
    got_lat, got_ks, got_ss, cap = _run(c, xs, 5, seed_kv, seed_ss)

    assert got_lat.shape == (1, 2, 4)
    assert torch.allclose(got_lat, ref_lat)
    assert torch.equal(got_ks, ref_ks) and torch.equal(got_ss, ref_ss)
    # capture: before token 0 + one per token
    assert len(cap) == xs.size(1) + 1
    assert all(e[0] == 0 and e[1] == "attn" for e in cap)


def test_gated_extend_aligned_matches_batched_grouping():
    """start_pos=4 (group-aligned): groups [4,5], [6,7] -- same as the batched path.

    The batched path intentionally leaves the register as seeded (the next group's
    first token overwrites it before any emit), so only the latents are compared."""
    c = _make(2)
    xs = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 4)
    seed_kv = torch.zeros(2, 4)
    seed_ss = torch.full((2, 4), float("-inf"))
    ref_lat, _ref_ks, _ref_ss = _expected(seed_kv, seed_ss, xs, 4, 2)
    got_lat, _got_ks, _got_ss, _ = _run(c, xs, 4, seed_kv, seed_ss)
    assert torch.allclose(got_lat, ref_lat)


def test_restore_carry_roundtrip():
    c = _make(2)
    c._write_through_carry = lambda slot: None
    c._kv_state.fill_(3.0)
    c._score_state.fill_(-2.0)
    snap = c.snapshot_carry()
    c._kv_state.zero_()
    c._score_state.fill_(float("-inf"))
    c.restore_carry(snap, window_slot=0)
    assert torch.equal(c._kv_state, torch.full_like(c._kv_state, 3.0))
    assert torch.equal(c._score_state, torch.full_like(c._score_state, -2.0))
