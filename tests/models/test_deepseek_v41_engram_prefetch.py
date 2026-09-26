"""Engram.prefetch_rows (CPU): hashes each token's 2/3/4-gram lookback out of the token
cache — across the previous chunk, padded at the sequence start — and hands exactly those
row ids to the row source on the background reader."""
from concurrent.futures import ThreadPoolExecutor

import torch

from freetoken.models.deepseek_v41.engram import Engram


class _Layout:
    def rows_for(self, layer_id, tokens):
        # distinct per (token window, column): 24 columns like the real layout
        base = tokens[:, 0] * 1_000_000 + tokens[:, 1] * 10_000 + tokens[:, 2] * 100 + tokens[:, 3]
        return base.unsqueeze(-1) * 24 + torch.arange(24)


class _Source:
    def __init__(self):
        self.calls = []

    def read_rows(self, rows, out):
        self.calls.append(rows.clone())
        out.zero_()


def _engram():
    e = Engram.__new__(Engram)
    e.layer_id, e.n_sizes, e.n_hash_cols, e.pad_token_id = 1, 3, 24, 2
    e._vocab = torch.arange(100) + 1000  # compressed id = token id + 1000; pad -> 1002
    e._token_cache = torch.zeros(4, 64, dtype=torch.int64)
    e._layout, e._source = _Layout(), _Source()
    e._scratch, e._pending = None, None
    e._prefetcher = ThreadPoolExecutor(max_workers=1)
    return e


def test_prefetch_hashes_lookback_across_chunks_and_pads_the_start():
    e = _engram()
    table = 2
    # chunk 1: positions 0..4, chunk 2: positions 5..7 (lookback reaches into chunk 1)
    for ids, pos in (([11, 12, 13, 14, 15], range(0, 5)), ([16, 17, 18], range(5, 8))):
        positions = torch.tensor(list(pos))
        e.prefetch_rows(torch.tensor(ids) + 1000, torch.full((len(ids),), table), positions)
        assert e._pending[0] is positions and e._pending[1] == len(ids) * 24
        e._pending[2].result()
    first, second = e._source.calls
    pad = 1002
    # token at position 1 (id 12): window [12, 11, pad, pad]
    want = _Layout().rows_for(1, torch.tensor([[1012, 1011, pad, pad]])).view(-1)
    assert torch.equal(first.view(5, 24)[1], want)
    # token at position 5 (id 16), first of chunk 2: window [16, 15, 14, 13] from chunk 1
    want = _Layout().rows_for(1, torch.tensor([[1016, 1015, 1014, 1013]])).view(-1)
    assert torch.equal(second.view(3, 24)[0], want)
    assert e._scratch.shape[0] >= 5 * 24
