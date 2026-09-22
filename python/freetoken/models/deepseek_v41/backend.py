"""DeepSeek-V4.1 attention backend: the DSV4 sparse machinery with the V4.1
kv-source indirection.

Every cr>0 layer attends its OWN window ring plus the compressed cache of its
kv-SOURCE layer (2/8/14/20 — consumers never write it). The base DSV4 backend
resolves pools per layer_id; this subclass redirects the compressed half of
``attend`` to the source's pool. Writes need no indirection: only source layers
own a compressor, and they write their own cache. ``blocks_to_global`` is
ratio-parameterized already, so consumers translate blocks with the SOURCE's
ratio.
"""

from __future__ import annotations

from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend


class Dsv41SparseAttnBackend(DSV4SparseAttnBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        args = config.dsv4_args
        source = 0
        isource = 0
        self.kv_source: dict[int, int | None] = {}
        self.idx_source: dict[int, int] = {}
        for L in range(config.num_layers):
            if args.compress_ratios[L] == 0:
                self.kv_source[L] = None
            else:
                if L in args.kv_source_layer_ids:
                    source = L
                self.kv_source[L] = source
            if L in args.indexer_layer_ids:
                isource = L
            # the K cache an index layer READS: its kv source's (24/28/32/36
            # share layer 20's; 2/8/14/20 own theirs)
            self.idx_source[L] = self.kv_source.get(L, isource) or isource

    def compress_pool(self, layer_id: int, tier: str):
        # "attn" resolves at the kv source (consumers read the source's cache);
        # "idx" at the index source (24/28/32/36 share layer 20's K cache).
        layer_id = (self.idx_source if tier == "idx" else self.kv_source).get(layer_id, layer_id)
        return super().compress_pool(layer_id, tier)

    def compress_state_ring(self, layer_id: int, tier: str):
        layer_id = (self.idx_source if tier == "idx" else self.kv_source).get(layer_id, layer_id)
        return super().compress_state_ring(layer_id, tier)

    def compress_scratch_base(self, layer_id: int, tier: str) -> int:
        layer_id = (self.idx_source if tier == "idx" else self.kv_source).get(layer_id, layer_id)
        return super().compress_scratch_base(layer_id, tier)

    def attend(
        self, q, layer_id, topk_idxs, n_window, attn_sink, softmax_scale,
        cmp_counts=None, has_compression: bool = True,
        allow_multi_query_split: bool = False,
    ):
        from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

        pool = self.pool
        if has_compression:
            src = self.kv_source[layer_id]
            assert src is not None, f"layer {layer_id} compresses but has no kv source"
            cmp = pool.cmp_pool[src]
        else:
            # ratio-0 layers have no compressed pool; the kernel never reads it there
            cmp = pool.window_pool[layer_id]
        return sparse_attn_paged(
            q, pool.window_pool[layer_id], cmp, attn_sink,
            topk_idxs.int(), n_window, softmax_scale, cmp_counts=cmp_counts,
            allow_multi_query_split=allow_multi_query_split,
        )


__all__ = ["Dsv41SparseAttnBackend"]
