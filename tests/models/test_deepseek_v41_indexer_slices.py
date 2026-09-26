"""V41Indexer.prefill_select in query slices picks exactly what one whole-chunk pass picks,
for the candidate source layer and for a consumer masked by its candidates (CPU, torch
reference in place of the triton scoring/quant kernels)."""
import torch

from freetoken.models.deepseek_v41 import indexer as idx_mod


class _Attn:
    def __init__(self, keys):
        self.keys = keys  # [n_rows, H*D]

    def indexer_keys(self, ti, n_rows, ratio, layer_id, bsz=1):
        return self.keys[:n_rows]

    def indexer_prefill_logits(self, q, keys, weights):
        # q [1, m, H, D], keys [n_rows, D], weights [1, m, H] -> [1, m, n_rows]
        return torch.einsum("bmhd,rd->bmhr", q.float(), keys.float()).relu().mul(weights.unsqueeze(-1)).sum(2)

    def blocks_to_global(self, blocks, ratio, ti=None, rows=None):
        return blocks


class _Linear:
    def __init__(self, w):
        self.w = w

    def forward(self, x):
        return x @ self.w


class _Indexer(idx_mod.V41Indexer):
    attn = None  # plain attribute in place of the global-backend property


def _indexer(source: bool, keys, heads, dim):
    ix = _Indexer.__new__(_Indexer)
    g = torch.Generator().manual_seed(1 if source else 2)
    ix.n_heads, ix.head_dim, ix.rope_head_dim = heads, dim, 4
    ix.index_topk, ix.src_ratio = 16, 4
    ix.is_candidate_source, ix.uses_candidates = source, not source
    ix.block_size, ix.topk_blocks = 8, 3
    ix.wq_b = _Linear(torch.randn(12, heads * dim, generator=g))
    ix.weights_proj = _Linear(torch.randn(10, heads, generator=g))
    ix.scale_folded = 0.1
    ix.layer_id = 20 if source else 21
    ix._freqs_cis = torch.ones(4096, 2, dtype=torch.complex64)
    ix.attn = _Attn(keys)
    return ix


def _select(monkeypatch, micro_q, start_pos, n):
    monkeypatch.setattr(idx_mod, "_SELECT_MICRO_Q", micro_q)
    monkeypatch.setattr(idx_mod, "fp4_act_quant_inplace", lambda t, bs: t)
    monkeypatch.setattr(idx_mod, "apply_rotary_emb", lambda t, f: t)
    heads, dim = 3, 8
    torch.manual_seed(0)
    keys = torch.randn((start_pos + n) // 4, dim)
    x, qr = torch.randn(1, n, 10), torch.randn(1, n, 12)
    shared: dict = {}
    src = _indexer(True, keys, heads, dim).prefill_select(x, qr, start_pos, n, 0, 0, shared)
    use = _indexer(False, keys, heads, dim).prefill_select(x, qr, start_pos, n, 0, 0, shared)
    return src, use, shared["candidates_list"][0]


def test_sliced_select_matches_one_pass(monkeypatch):
    for start_pos, n in ((0, 300), (517, 211)):
        whole = _select(monkeypatch, 10_000, start_pos, n)
        sliced = _select(monkeypatch, 37, start_pos, n)
        for a, b in zip(whole, sliced):
            assert torch.equal(a, b)
        # the causal bound holds: no pick beyond a query's live rows
        live = (start_pos + torch.arange(1, n + 1)) // 4
        assert (whole[0][0] < live.unsqueeze(-1)).all()
