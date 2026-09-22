"""Lightning Indexer for DeepSeek-V4.1 (reference inference/model.py:488-610).

One instance per INDEX-SOURCE layer ([2, 8, 14, 20, 24, 28, 32, 36]). Formula
(verbatim from the checkpoint's own reference):

    weights = weights_proj(x) * (index_head_dim**-0.5 * n_heads**-0.5)
    logits[s, t] = sum_h relu(q[s, h] . k[t]) * weights[s, h]        # NO square

- q = wq_b(qr) (the attention's normalized low-rank query), fp8-quantized linear,
  roped at TOKEN positions with the layer's compress-regime table, fp4/32
  round-trip;
- K (only kv-source indexers own it): k_norm(wk(latent)) on the compressor's
  PRE-rope normed latent, roped at the group-START position, fp4/32 round-trip,
  paged into the idx pool at the source's compressed rows; layers 24/28/32/36
  share layer 20's K cache (idx-source indirection in the backend);
- top-512 over the kv source's compressed rows, ascending, causal (query sees
  rows t < (p+1)//ratio — its own group only when it closes the group);
- candidate filter: layer 20 publishes the top-2048 blocks-of-8 (block score =
  amax over rows, newest partial block pinned) from its UNMASKED score; layers
  24+ re-score with their own weights but restrict selection to those positions.

The selection replaces Phase 3a's static all-blocks candidates and feeds the
shared per-forward buffers (topk indices + candidate mask) consumed by the
non-index layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args


class V41Indexer(BaseOP):
    def __init__(self, config, layer_id: int, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.eps = args.norm_eps
        self.compress_ratio = args.compress_ratios[layer_id]
        self.owns_k = layer_id in args.kv_source_layer_ids
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.uses_candidates = args.candidate_source_layer_id < layer_id
        self.block_size = args.candidate_block_size
        self.topk_blocks = args.candidate_topk_blocks
        # rows this indexer scores = its kv source's compressed rows
        source = 0
        for s in args.kv_source_layer_ids:
            if s <= layer_id:
                source = s
        self.src_ratio = args.compress_ratios[source]

        self.wq_b = LinearReplicated(args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.weights_proj = LinearReplicated(args.hidden_size, self.n_heads, has_bias=False, quant_config=None, prefix=f"{prefix}.weights_proj")
        if self.owns_k:
            self.wk = LinearReplicated(args.head_dim, self.head_dim, has_bias=False, quant_config=None, prefix=f"{prefix}.wk")
            self.k_norm = RMSNorm(self.head_dim, self.eps)
        else:
            self.wk = None
            self.k_norm = None
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_folded = self.softmax_scale * self.n_heads ** -0.5
        # all index layers are ratio>0 → the compress rope regime
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, args.original_seq_len,
            args.compress_rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_pool(self, pool, freqs_cis: torch.Tensor) -> None:
        self._freqs_cis = freqs_cis

    # ------------------------------------------------------------------ K write (owners)
    def write_k_prefill(self, latent: torch.Tensor, start_pos: int, ti: int) -> None:
        """Owner only: K rows for the groups closed by this prefill call."""
        if not self.owns_k or latent is None or latent.size(1) == 0:
            return
        ratio, rd = self.compress_ratio, self.rope_head_dim
        n = latent.size(1)
        first_group = start_pos // ratio
        # 3-D norm input: flashinfer's 2-D rmsnorm_cute rejects the tiny verify rows.
        lat = self.wk.forward(latent)
        if lat.dim() == 2:
            lat = lat.unsqueeze(0)
        k = self.k_norm.forward(lat).view(1, n, self.head_dim)
        freqs = self._freqs_cis.index_select(
            0, (first_group + torch.arange(n, device=k.device)) * ratio
        )
        apply_rotary_emb(k[..., -rd:], freqs)
        fp4_act_quant_inplace(k, 32)
        rows = self.attn.compress_rows_of(ti, (first_group + torch.arange(n, device=k.device)) * ratio, ratio)
        self.attn.scatter_compressed(self.layer_id, "idx", rows, k[0])

    def write_k_decode(self, latent: torch.Tensor, should: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor) -> None:
        """Owner only: per-row K for the group this token closes (scratch otherwise)."""
        if not self.owns_k:
            return
        ratio, rd = self.compress_ratio, self.rope_head_dim
        B = latent.size(0)
        k = self.k_norm.forward(self.wk.forward(latent))
        freqs = self._freqs_cis.index_select(0, (pos + 1 - ratio).clamp_min(0))
        apply_rotary_emb_decode(k[..., -rd:], freqs)
        fp4_act_quant_inplace(k, 32)
        dst = self.attn.decode_compress_rows(rows, pos, ratio, self.layer_id, "idx", should)
        self.attn.scatter_compressed(self.layer_id, "idx", dst, k.view(B, -1))

    # ------------------------------------------------------------------ scoring
    def _candidate_mask(self, scores: torch.Tensor, compress_lens: torch.Tensor) -> torch.Tensor:
        """Layer 20: top-2048 blocks-of-8 by amax, newest partial block pinned;
        returns a bool mask over positions ([..., n_rows])."""
        bs = self.block_size
        width = scores.size(-1)
        padded = F.pad(scores, (0, -width % bs), value=float("-inf"))
        block_scores = padded.unflatten(-1, (-1, bs)).amax(dim=-1)
        num_blocks = block_scores.size(-1)
        last = (compress_lens - 1) // bs  # [...] per query
        block_scores = block_scores.masked_fill(
            torch.arange(num_blocks, device=scores.device) == last.unsqueeze(-1), float("inf")
        )
        top = block_scores.topk(min(self.topk_blocks, num_blocks), dim=-1)
        keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(bs, dim=-1)[..., :width]

    def static_prefill(self, start_pos: int, n: int, ti: int):
        """Phase-3a fallback: all valid compressed blocks as candidates."""
        import os

        device = get_global_ctx().attn_backend.device
        ratio = self.src_ratio
        if start_pos == 0:
            from freetoken.models.deepseek_v4.layers import get_compress_topk_idxs

            blocks = get_compress_topk_idxs(ratio, 1, n, 0, 0).to(device)
        else:
            n_blocks = (start_pos + n) // ratio
            blk = torch.arange(n_blocks, device=device).unsqueeze(0)
            abs_p1 = (start_pos + torch.arange(n, device=device) + 1).unsqueeze(1)
            blocks = torch.where(blk < abs_p1 // ratio, blk, -1).unsqueeze(0)
        return self.attn.blocks_to_global(blocks, ratio, ti=ti).int()

    def prefill_select(self, x, qr, start_pos: int, n: int, ti: int, seg_i: int, shared: dict):
        import os

        if os.path.exists("/tmp/dsv41-no-indexer"):
            return self.static_prefill(start_pos, n, ti)
        """Score + select for one prefill/extend segment; returns the GLOBAL row
        picks [1, n, topk] (ascending, -1 pad — already translated to the shared
        cache's physical rows, so consumers slice the flat buffer directly).
        Publishes this segment's candidate mask at shared["candidates_list"][seg_i]."""
        device = x.device
        rd = self.rope_head_dim
        end = start_pos + n
        n_rows = end // self.src_ratio
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        freqs = self._freqs_cis[start_pos:end]
        apply_rotary_emb(q[..., -rd:], freqs)
        fp4_act_quant_inplace(q, 32)
        weights = self.weights_proj.forward(x) * self.scale_folded
        keys = self.attn.indexer_keys(ti, n_rows, self.src_ratio, self.layer_id, bsz=1)
        scores = self.attn.indexer_prefill_logits(q, keys, weights)  # [1, n, n_rows]
        live = ((start_pos + torch.arange(1, n + 1, device=device)) // self.src_ratio).unsqueeze(-1)
        if "candidates_list" not in shared:
            shared["candidates_list"] = {}
        if self.is_candidate_source:
            shared["candidates_list"][seg_i] = self._candidate_mask(scores, live.squeeze(-1))
        elif self.uses_candidates:
            scores = scores.masked_fill(~shared["candidates_list"][seg_i], float("-inf"))
        scores = scores.masked_fill(torch.arange(n_rows, device=device) >= live, float("-inf"))
        topk = min(self.index_topk, n_rows)
        picks = scores.topk(topk, dim=-1)[1].sort(dim=-1).values
        picks = torch.where(picks < live, picks, -1)
        return self.attn.blocks_to_global(picks, self.src_ratio, ti=ti).int()

    def decode_select(self, x, qr, pos, latent, should, rows, shared: dict, cmp_stage_cap: int):
        """Decode step: K write (owners), score, candidate mask, top-512; returns
        (GLOBAL row picks [B, 1, ≤topk], cmp_counts [B, 1])."""
        import os

        device = x.device
        if os.path.exists("/tmp/dsv41-no-indexer"):
            B = x.size(0)
            valid = (pos + 1) // self.src_ratio
            n_cmp_stage = (cmp_stage_cap + 1) // self.src_ratio
            blk = torch.arange(n_cmp_stage, device=device)
            blocks = torch.where(blk[None, :] < valid[:, None], blk[None, :], -1).view(B, 1, n_cmp_stage)
            global_rows = self.attn.blocks_to_global(blocks, self.src_ratio, rows=rows).int().view(B, 1, -1)
            return global_rows, valid.clamp(max=n_cmp_stage).to(torch.int32).view(B, 1)
        B = x.size(0)
        rd = self.rope_head_dim
        if self.owns_k:
            self.write_k_decode(latent, should, pos, rows)
        valid = (pos + 1) // self.src_ratio  # [B]
        n_stage = (cmp_stage_cap + 1) // self.src_ratio
        q = self.wq_b.forward(qr).view(B, self.n_heads, self.head_dim)
        apply_rotary_emb_decode(q[..., -rd:], self._freqs_cis.index_select(0, pos))
        fp4_act_quant_inplace(q, 32)
        weights = self.weights_proj.forward(x) * self.scale_folded
        scores = self.attn.indexer_decode_scores(
            q, weights, valid, n_stage, self.src_ratio, self.layer_id
        ).view(B, n_stage)
        if self.is_candidate_source:
            shared["candidates"] = self._candidate_mask(scores, valid)
        elif self.uses_candidates:
            scores = scores.masked_fill(~shared["candidates"], float("-inf"))
        scores = scores.masked_fill(torch.arange(n_stage, device=device)[None, :] >= valid[:, None], float("-inf"))
        picks = scores.topk(min(self.index_topk, n_stage), dim=-1)[1].sort(dim=-1).values
        idx = torch.where(picks < valid[:, None], picks, -1)
        global_rows = self.attn.blocks_to_global(idx, self.src_ratio, rows=rows).int().view(B, 1, -1)
        counts = valid.clamp(max=self.index_topk).to(torch.int32).view(B, 1)
        return global_rows, counts


__all__ = ["V41Indexer"]
