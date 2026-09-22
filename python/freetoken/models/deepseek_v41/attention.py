"""DeepSeek-V4.1 attention: SWA-128 window + shared compressed caches (CSA2).

Driver for the DSV4 paged machinery with the V4.1 layer taxonomy:

- layers 0/1 (ratio 0): window-only attention;
- layers 2/8/14 (ratio 2) and 20 (ratio 1) own a V41Compressor and write their
  compressed cache; every other cr>0 layer READS its kv source's cache
  (backend-side indirection, see backend.py);
- Phase 3a selects compressed candidates STATICALLY (all valid blocks — the
  same mechanism V4 uses for its indexierlosen ratio-128 layers), bounded per
  query by ``cmp_counts``; the learned indexer (top-512 + candidate filter)
  replaces this in Phase 3b (the DSV4.1 plan);
- per-head ``attn_sink`` rides the sparse kernel (in-kernel sink logit).

Weight mapping and the o-projection geometry are unchanged from Phase 2
(wq_a/q_norm/wq_b, weightless per-head rms, wo_a group-BMM/wo_b, inverse rope
on the output's trailing rope dims).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.models.deepseek_v4.layers import (
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from freetoken.models.deepseek_v4.ops import apply_rotary_emb
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    RMSNorm,
)
from freetoken.utils import div_even

from .args import DeepseekV41Args
from .compress import V41Compressor
from .indexer import V41Indexer


class Attention(BaseOP):
    def __init__(self, config, layer_id: int, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.eps = args.norm_eps
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.sliding_window
        self.compress_ratio = args.compress_ratios[layer_id]
        tp = get_tp_info().size
        self.local_heads = div_even(self.n_heads, tp)
        self.local_groups = div_even(self.n_groups, tp)

        self.attn_sink = torch.empty(self.n_heads, dtype=torch.float32)
        # TP: the sparse kernel indexes sinks by LOCAL query head — resolve this
        # rank's slice at attend time (the param itself stays full for strict load)
        rank = get_tp_info().rank
        self._sink_slice = slice(rank * self.local_heads, (rank + 1) * self.local_heads)
        self.wq_a = LinearReplicated(config.hidden_size, args.q_lora_rank, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wq_a")
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = LinearColParallelMerged(args.q_lora_rank, [self.n_heads * self.head_dim], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wq_b")
        self.wkv = LinearReplicated(config.hidden_size, self.head_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wkv")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        wo_a_rows = self.local_groups * self.o_lora_rank
        wo_a_k = self.local_heads * self.head_dim // self.local_groups
        self.wo_a = torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(self.n_groups * self.o_lora_rank, config.hidden_size, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wo_b")
        self.softmax_scale = self.head_dim ** -0.5

        # kv-source resolution: the layer whose compressed cache this layer reads,
        # and that source's ratio (sizes the compressed rows / block arithmetic)
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.kv_source_layer = 0
        self.src_ratio = 0
        if self.compress_ratio:
            source = 0
            for s in args.kv_source_layer_ids:
                if s <= layer_id:
                    source = s
            self.kv_source_layer = source
            self.src_ratio = args.compress_ratios[source]
            assert self.is_kv_source or self.src_ratio > 0

        # compressor on kv-source layers only (consumers read the source's cache)
        if self.is_kv_source:
            self.compressor: V41Compressor | None = V41Compressor(
                args, self.compress_ratio, self.head_dim, quant_config=config.quant, prefix=f"{prefix}.compressor"
            )
        else:
            self.compressor = None
        # indexer on index-source layers ([2,8,14,20,24,28,32,36]); non-index
        # cr>0 layers consume the shared per-forward selection buffers
        self.is_index_source = layer_id in args.indexer_layer_ids
        if self.is_index_source:
            self.indexer: V41Indexer | None = V41Indexer(
                config, layer_id, args, quant_config=quant_config, prefix=f"{prefix}.indexer"
            )
        else:
            self.indexer = None

        # two rope regimes: ratio-0 layers plain theta, ratio>0 layers the compressed-KV yarn
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_pool(self, pool, device: torch.device, freqs_cis: torch.Tensor) -> None:
        self._freqs_cis = freqs_cis
        if self.compressor is not None:
            self.compressor.bind_paged(pool, self.layer_id, freqs_cis, device, tier="attn")
        if self.indexer is not None:
            self.indexer.bind_pool(pool, freqs_cis)

    def reset(self) -> None:
        if self.compressor is not None:
            self.compressor.reset()

    def _wo(self, o: torch.Tensor, seqlen: int) -> torch.Tensor:
        o = o.reshape(1, seqlen, self.local_groups, -1)
        wo_a = self.wo_a.view(self.local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(o)

    def _q_path(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """q [1, T, local_heads, 512] (the attend kernel's 4D contract), roped
        (weightless per-head rms after wq_b)."""
        rd = self.rope_head_dim
        T = x.numel() // x.shape[-1]
        q = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(q).view(1, T, self.local_heads, self.head_dim)
        apply_rotary_emb(q[..., -rd:], freqs)
        return q

    def _kv_latent(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """The 512-wide kv latent, roped (this row goes to the layer's window ring).

        Accepts 2D [T, dim] or 3D [1, T, dim] input; returns kv [T, 512] (2D) so
        the store/attend paths keep their flat-token convention."""
        rd = self.rope_head_dim
        kv = self.kv_norm.forward(self.wkv.forward(x))
        kv3 = kv.unsqueeze(0) if kv.dim() == 2 else kv
        apply_rotary_emb(kv3[..., -rd:], freqs)
        kv = kv.view(-1, self.head_dim)
        # Reference _window_kv (model.py:707): the window K stays fp8, quantized over the
        # whole post-rope vector (RoPE tail included) with block fp8_block_size=32.
        act_quant_fp8_inplace(kv, 32)
        return kv

    def _store_window(self, kv: torch.Tensor, segments) -> None:
        slots = torch.cat([
            self.attn.window_slots_of(ti, start_pos, start_pos + n)
            for _off, n, ti, start_pos in segments
        ])
        self.attn.store_window(kv, self.layer_id, slots)

    def forward_ragged(self, x, segments, flat_positions, shared):
        """Ragged batched prefill (V4 driver contract): x [1, T, dim]; per-request
        segments (offset, extend_len, table_idx, start_pos); the compressor carry
        runs PER REQUEST for isolation; one flat attend launch over all T queries.
        ``shared`` carries the per-forward indexer selection buffers (topk picks +
        candidate mask) from each index-source layer to its consumers."""
        win = self.window_size
        rd = self.rope_head_dim
        device = x.device
        _, T, _ = x.size()
        decode_verify = get_global_ctx().dspark_decode_verify
        if len(segments) == 1:
            freqs = self._freqs_cis[segments[0][3] : segments[0][3] + T]
        else:
            freqs = self._freqs_cis.index_select(0, flat_positions)

        qr = self.q_norm.forward(self.wq_a.forward(x))  # [1, T, q_lora]
        q = self.wq_b.forward(qr).view(1, T, self.local_heads, self.head_dim)
        apply_rotary_emb(q[..., -rd:], freqs)
        kv = self._kv_latent(x, freqs)
        self._store_window(kv, segments)

        win_parts: list[torch.Tensor] = []
        cmp_parts: list[torch.Tensor] = []
        max_c = 0
        for seg_i, (off, n, ti, start_pos) in enumerate(segments):
            slots = self.attn.window_slots_of(ti, start_pos, start_pos + n)
            if decode_verify:
                # decode shares the plain-decode reduction: ring slot j (not ascending position)
                win_global = self.attn.window_ring_cols(ti, start_pos, n)
            elif start_pos == 0:
                win_cols = get_window_topk_idxs(win, 1, n, 0).to(device)
                win_global = self.attn.win_cols_to_global(win_cols, slots)
            else:
                w_lo = max(0, start_pos - win + 1)
                ws_pool = self.attn.window_slots_of(ti, w_lo, start_pos + n)
                abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
                cand = (abs_p - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
                win_cols = torch.where(cand > abs_p, -1, cand - w_lo).unsqueeze(0)
                win_global = self.attn.win_cols_to_global(win_cols, ws_pool)
            win_parts.append(win_global)

            if self.src_ratio:
                x_seg = x[:, off : off + n]
                if self.is_kv_source:
                    if start_pos == 0:
                        self.reset()
                        latent = self.compressor.compute(x_seg, 0, slots, ti=ti)
                    else:
                        tail_ws = int(self.attn.window_slots_of(ti, start_pos - 1, start_pos).item())
                        latent = self.compressor.compute_extend(
                            x_seg, start_pos, slots, tail_window_slot=tail_ws, ti=ti,
                            capture=get_global_ctx().dspark_carry_capture,
                        )
                    if self.indexer is not None:
                        self.indexer.write_k_prefill(latent, start_pos, ti)
                    self.compressor.finalize(latent, start_pos, ti)
                if self.indexer is not None:
                    qr_seg = qr[:, off : off + n]
                    cmp_global = self.indexer.prefill_select(
                        x_seg, qr_seg, start_pos, n, ti, seg_i, shared
                    )
                else:
                    # consumer: the index source's picks for this segment's tokens
                    # (already translated to global rows by the index source)
                    cmp_global = shared["topk"][:, off : off + n]
                cmp_parts.append(cmp_global)
                max_c = max(max_c, cmp_global.shape[-1])

        topk_parts = []
        for i, (_off, _n, _ti, _start) in enumerate(segments):
            wg = win_parts[i]
            if wg.shape[-1] < win and len(segments) > 1:
                wg = F.pad(wg, (0, win - wg.shape[-1]), value=-1)
            parts = [wg]
            if self.src_ratio:
                cg = cmp_parts[i]
                if cg.shape[-1] < max_c:
                    cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
                parts.append(cg)
            topk_parts.append(torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0])
        topk_idxs = (topk_parts[0] if len(topk_parts) == 1 else torch.cat(topk_parts, dim=1)).int()
        # bs>1 pads the window half to the full window; bs==1 keeps the segment's
        # natural width (a cold prompt shorter than the window yields < win columns)
        n_window = win if len(segments) > 1 else win_parts[0].shape[-1]
        if self.indexer is not None and self.src_ratio:
            # the flat picks buffer for this index source's consumers (uniform
            # topk width — the pad above already applied per segment)
            shared["topk"] = torch.cat(
                [p if p.shape[-1] == max_c else F.pad(p, (0, max_c - p.shape[-1]), value=-1)
                 for p in cmp_parts], dim=1
            ).int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink[self._sink_slice],
            self.softmax_scale, has_compression=bool(self.src_ratio),
            allow_multi_query_split=decode_verify and len(segments) == 1,
        )  # [1, T, local_heads, 512]
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, T)

    def decode_step(self, x, pos, rows, cmp_stage_cap, wctx=None, shared=None):
        """Batched single-token attention (EAGER/graph); V4 driver contract."""
        B = x.size(0)
        win = self.window_size
        rd = self.rope_head_dim
        device = x.device
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        freqs_t = self._freqs_cis.index_select(0, pos)

        qr = self.q_norm.forward(self.wq_a.forward(x))  # [B, q_lora]
        q = self.wq_b.forward(qr).view(B, self.local_heads, self.head_dim)
        q4 = q.view(1, B, self.local_heads, self.head_dim)
        apply_rotary_emb(q4[..., -rd:], freqs_t)
        kv = self._kv_latent(x, freqs_t)
        self.attn.store_window(kv, self.layer_id, window_slots)

        cmp_counts = None
        if self.src_ratio:
            if self.is_kv_source:
                latent, should = self.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
                if self.indexer is not None:
                    self.indexer.write_k_decode(latent, should, pos, rows)
            if self.indexer is not None:
                compress_global, cmp_counts = self.indexer.decode_select(
                    x, qr, pos, latent if self.is_kv_source else None,
                    should if self.is_kv_source else None,
                    rows, shared, cmp_stage_cap,
                )
                # publish for this index source's consumers (21-23 read 20, …)
                shared["topk"] = compress_global
                shared["cmp_counts"] = cmp_counts
            else:
                compress_global = shared["topk"]  # [B, 1, topk], already global
                cmp_counts = shared["cmp_counts"]
            topk_idxs = torch.cat([window_slots_topk, compress_global], dim=-1)
        else:
            topk_idxs = window_slots_topk
        topk_idxs = topk_idxs.int()

        o = self.attn.attend(
            q4, self.layer_id, topk_idxs, win, self.attn_sink[self._sink_slice],
            self.softmax_scale, cmp_counts=cmp_counts, has_compression=bool(self.src_ratio),
        )  # [1, B, local_heads, 512]
        apply_rotary_emb(o[..., -rd:], freqs_t, True)
        return self._wo(o, B)


__all__ = ["Attention"]
