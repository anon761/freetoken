"""Engram prefill micro-batching must not change results.

The prefill gate math is per-token independent, so the micro-batched forward
(_PREFILL_MICRO_BS slices, bounding the fp32 transients) must be bit-identical
to a single-shot pass over the whole chunk. Regression for the production OOM:
one 8192-token prefill peaked ~1 GiB of fp32 temporaries inside the
(1 - memory_ratio) headroom.

Runs on CPU against a fabricated Engram (no checkpoint, no GPU): a
deterministic fake row source makes every gather path-independent of batching.
"""
from __future__ import annotations

import os
from unittest import mock

import torch

from freetoken.models.deepseek_v41 import engram as engram_module
from freetoken.models.deepseek_v41.engram import Engram, _ROW_BYTES

N_HASH_COLS = 24   # n_sizes(4) * n_heads(6)
HEAD_DIM = 256     # hash row = n_hash_cols * head_dim = 6144
DIM = 64
HC = 2
M = 2500           # > 2 micro-batches at the default 1024


def _make_engram() -> Engram:
    eng = object.__new__(Engram)
    eng.layer_id = 3
    eng.n_sizes = 4
    eng.n_hash_cols = N_HASH_COLS
    eng.head_dim = HEAD_DIM
    eng.hc_mult = HC
    eng.dim = DIM
    eng.eps = 1e-6
    eng.clamp_value = 1e-6
    eng.pad_token_id = 0
    gen = torch.Generator().manual_seed(42)
    eng._vocab = torch.randint(1, 1000, (1000,), generator=gen, dtype=torch.int64)
    eng._token_cache = torch.zeros(8, 512, dtype=torch.int64)
    eng.q_weight = torch.randn(HC, DIM, generator=gen, dtype=torch.bfloat16)
    eng.k_weight = torch.randn(HC, DIM, generator=gen, dtype=torch.bfloat16)
    torch.manual_seed(7)
    eng.wkv = torch.nn.Linear(N_HASH_COLS * HEAD_DIM, DIM * (HC + 1), bias=False).to(torch.bfloat16)

    class _Layout:
        def rows_for(self, layer_id: int, tokens: torch.Tensor) -> torch.Tensor:
            base = (tokens.sum(-1, keepdim=True) * 7 + layer_id * 13) % 50000
            return (base + torch.arange(N_HASH_COLS) * 101) % 50000

    class _Source:
        def read_rows(self, rows: torch.Tensor, buf: torch.Tensor) -> None:
            r = rows.to(torch.int64)
            # 1..126: valid positive e4m3 codes (0x7F/0xFF would be NaN)
            buf[:, :_ROW_BYTES] = ((r.unsqueeze(1) * 31 + torch.arange(_ROW_BYTES)) % 126 + 1).to(torch.uint8)
            # e8m0 exponents near 127 (=1.0), like the real PLE scales — anything
            # far above/below turns exp2 into inf/0 and the gate into NaN
            buf[:, _ROW_BYTES:] = ((r % 15 + 120).unsqueeze(1)).to(torch.uint8)

    eng._layout = _Layout()
    eng._source = _Source()
    eng._scratch = None
    return eng


def test_micro_batched_forward_matches_single_shot():
    eng = _make_engram()
    gen = torch.Generator().manual_seed(123)
    R = torch.randn(M, HC, DIM, generator=gen, dtype=torch.bfloat16)
    comp_ids = torch.randint(1, 1000, (M,), generator=gen)
    cache_rows = torch.randint(0, 8, (M,), generator=gen)
    positions = torch.randint(0, 500, (M,), generator=gen)

    out_chunked = eng.forward(R, comp_ids, cache_rows, positions)

    with mock.patch.object(engram_module, "_PREFILL_MICRO_BS", 1 << 30):
        eng._scratch = None  # force re-growth along the single-shot path
        out_single = eng.forward(R, comp_ids, cache_rows, positions)

    assert torch.allclose(out_chunked.float(), out_single.float(), rtol=0, atol=0), (
        (out_chunked.float() - out_single.float()).abs().max()
    )


def test_token_cache_persist_is_complete_across_batches():
    # every token's compressed id must land in the cache even when its slice
    # is processed by a later micro-batch (the hash reads the cache)
    eng = _make_engram()
    m = 300  # positions stay within the 512-slot cache
    gen = torch.Generator().manual_seed(5)
    R = torch.randn(m, HC, DIM, generator=gen, dtype=torch.bfloat16)
    comp_ids = torch.randint(1, 1000, (m,), generator=gen)
    cache_rows = torch.zeros(m, dtype=torch.long)
    positions = torch.arange(m, dtype=torch.long)
    eng.forward(R, comp_ids, cache_rows, positions)
    assert torch.equal(eng._token_cache[0, :m], comp_ids)


def test_no_engram_flags_short_circuit(monkeypatch):
    eng = _make_engram()
    R = torch.randn(4, HC, DIM, dtype=torch.bfloat16)
    comp = torch.ones(4, dtype=torch.long)
    rows = torch.zeros(4, dtype=torch.long)
    pos = torch.arange(4)
    real_exists = os.path.exists
    monkeypatch.setattr(
        engram_module.os.path, "exists",
        lambda p: p == "/tmp/dsv41-no-engram" or real_exists(p),
    )
    assert torch.equal(eng.forward(R, comp, rows, pos), R)
