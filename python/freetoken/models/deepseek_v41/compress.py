"""DeepSeek-V4.1 KV compressors — reference math (inference/model.py:429-485).

Differences to the inherited V4 machinery:
- NO ape (V4's learned positional gate bias is absent from the checkpoint);
- ratio 1 (layer 20) is GATELESS: ``latent = norm(wkv(x))`` in bf16 — no pooling,
  no gate, no carry writes (every token closes its own group);
- ratio 2 (layers 2/8/14): fp32 softmax-gated pooling over the 2-row group (the
  reference promotes wkv/wgate to fp32); the OPEN group's partial rows ride the
  carry register/ring exactly like V4's non-overlap path;
- the prefill latent is returned PRE-rope (the indexer derives its K from the
  unrotated form — reference model.py:434 docstring); rope + fp4/16 quant +
  scatter happen in :meth:`finalize`, called by the attention after the
  indexer's K write. Decode keeps V4's fused step but returns the pre-rope
  latent together with the completion mask;
- latent quant is fp4 per-16 with E4M3 scales (reference kernel.py:160-163,
  "Training's compressed KV") — NOT V4's fp8/64;
- rope at the group-START position (freqs stride start_pos:+cutoff:ratio).

The paged addressing (cmp pool, state ring, boundary carries, decode snapshot)
keeps V4's ring layout — radix carry-by-value resume unchanged.
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.compress import gated_pool
from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args


class V41Compressor(BaseOP):
    def __init__(self, args: DeepseekV41Args, compress_ratio: int, head_dim: int, *, quant_config=None, prefix: str = ""):
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = False  # V4's CSA overlap exists only for ratio 4
        self.gated = compress_ratio > 1
        self.max_batch_size = args.max_batch_size
        self.wkv = LinearReplicated(args.hidden_size, head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        if self.gated:
            self.wgate = LinearReplicated(args.hidden_size, head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wgate")
        else:
            self.wgate = None
        self.norm = RMSNorm(head_dim, args.norm_eps)
        self.layer_id: int | None = None
        self.ring_size = compress_ratio  # non-overlap: coff=1
        self.coff = 1
        self.item_size = head_dim
        self.P: int = 128
        self._freqs_cis: torch.Tensor | None = None
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None
        self.tier = "attn"

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    @property
    def cmp_pool(self) -> torch.Tensor:
        return self.attn.compress_pool(self.layer_id, self.tier)

    @property
    def state_ring(self):
        return self.attn.compress_state_ring(self.layer_id, self.tier)

    def bind_paged(self, pool, layer_id: int, freqs_cis, device, tier: str) -> None:
        self.layer_id = layer_id
        self.tier = tier
        self.P = pool.P
        self._freqs_cis = freqs_cis
        self._device = device
        self._kv_state = torch.zeros(1, self.compress_ratio, self.head_dim, dtype=torch.float32, device=device)
        self._score_state = torch.full(
            (1, self.compress_ratio, self.head_dim), float("-inf"), dtype=torch.float32, device=device
        )

    def reset(self) -> None:
        self._kv_state.zero_()
        self._score_state.fill_(float("-inf"))

    def _write_through_carry(self, window_slot: int) -> None:
        self.attn.write_carry(
            self.layer_id, self.tier, window_slot, self.ring_size,
            torch.cat([self._kv_state[0], self._score_state[0]], dim=-1),
        )

    def _seed_carry_from_ring(self, window_slot: int) -> None:
        block = self.attn.read_carry(self.layer_id, self.tier, window_slot, self.ring_size)
        item = self.item_size
        self._kv_state[0].copy_(block[:, :item])
        self._score_state[0].copy_(block[:, item:])

    def _write_boundary_carries(self, kv, score, seqlen: int, window_slots: torch.Tensor) -> None:
        self._write_boundary_carries_range(kv, score, 0, seqlen, window_slots)

    def _write_boundary_carries_range(self, kv, score, lo: int, hi: int, window_slots) -> None:
        self.attn.write_boundary_carries(
            layer_id=self.layer_id, tier=self.tier, ratio=self.compress_ratio,
            overlap=False, ring_size=self.ring_size,
            ape=torch.zeros(1, device=kv.device), kv=kv, score=score, lo=lo, hi=hi,
            window_slots=window_slots,
        )

    def _pool(self, x: torch.Tensor):
        """Closed groups' normed PRE-rope latents + raw (kv, score) for the carries.

        gateless (ratio 1): latent = norm(wkv(x)) bf16, every token closes →
        kv/score None (no carry writes needed)."""
        if not self.gated:
            return self.norm.forward(self.wkv.forward(x)), None, None
        kv = self.wkv.forward(x.float())
        score = self.wgate.forward(x.float())
        ratio = self.compress_ratio
        seqlen = kv.size(1)
        cutoff = seqlen - seqlen % ratio
        if cutoff:
            pooled = (
                kv[:, :cutoff].unflatten(1, (-1, ratio))
                * score[:, :cutoff].unflatten(1, (-1, ratio)).softmax(dim=2)
            ).sum(dim=2)
        else:
            pooled = kv[:, :0]
        # the pooling ran fp32; the norm rides the bf16 kernel path (V4-consistent)
        return self.norm.forward(pooled.to(x.dtype)), kv, score

    def compute(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot=None, ti: int = 0):
        """Cold prefill: returns the normed PRE-rope latents of the groups closed
        by this call ([1, n_closed, head_dim]; possibly empty)."""
        assert self.cmp_pool is not None
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        latent, kv, score = self._pool(x)
        if self.gated:
            remainder = seqlen % ratio
            if remainder:  # the open group's partial rows ride the carry
                self._kv_state[0, :remainder] = kv[0, seqlen - remainder :]
                self._score_state[0, :remainder] = score[0, seqlen - remainder :]
            if seqlen:
                self._write_boundary_carries(kv, score, seqlen, window_slots)
        if seqlen % self.P != 0:
            self._write_through_carry(int(window_slots[-1].item()))
        return latent

    def snapshot_carry(self):
        """The bs=1 register carry as a detached pair, for speculative rollback."""
        return self._kv_state.clone(), self._score_state.clone()

    def restore_carry(self, snapshot, window_slot: int) -> None:
        """Restore the register carry to a snapshot and persist it to the ring block of
        ``window_slot``'s page, so the next decode/verify seeds from the accepted position."""
        kv, score = snapshot
        self._kv_state.copy_(kv)
        self._score_state.copy_(score)
        self._write_through_carry(window_slot)

    def compute_extend(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot: int, ti: int = 0,
                       capture: list | None = None):
        """Carry-aware extend for [start_pos, start_pos+seqlen).

        Two paths: ``start_pos % ratio == 0`` (radix re-prefill, large seqlen) keeps the
        batched grouping; an ARBITRARY start (a speculative verify resuming mid-group)
        advances the seeded register one token at a time so the open group closes at the
        right boundary. ``capture`` (a list) receives ``(layer_id, tier, snapshot)`` before
        token 0 and after every token, so the caller can roll the carry back on rejection.
        """
        assert self.cmp_pool is not None
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        self._seed_carry_from_ring(tail_window_slot)
        if capture is not None:
            capture.append((self.layer_id, self.tier, self.snapshot_carry()))
        if not self.gated:
            # ratio 1: every token closes its own group; arbitrary start is exact.
            latent = self._pool(x)[0]
            if capture is not None:
                for _ in range(seqlen):
                    capture.append((self.layer_id, self.tier, self.snapshot_carry()))
            if seqlen and seqlen % self.P != 0:
                self._write_through_carry(int(window_slots[-1].item()))
            return latent
        if start_pos % ratio == 0:
            # group-aligned (the radix/chunk-continuation case): the batched pooling is
            # exact and avoids seqlen per-token projections.
            latent, kv, score = self._pool(x)
            remainder = seqlen % ratio
            if remainder:
                self._kv_state[0, :remainder] = kv[0, seqlen - remainder :]
                self._score_state[0, :remainder] = score[0, seqlen - remainder :]
            if capture is not None:
                for _ in range(seqlen):
                    capture.append((self.layer_id, self.tier, self.snapshot_carry()))
            end = start_pos + seqlen
            self._write_boundary_carries_range(kv, score, start_pos, end, window_slots)
            if end % self.P != 0:
                self._write_through_carry(int(window_slots[-1].item()))
            return latent
        # arbitrary start: advance the register token-by-token (mirrors decode_step) so
        # the ring-seeded open group merges with the first new token.
        pooled_parts = []
        for j in range(seqlen):
            xj = x[:, j : j + 1]
            kv = self.wkv.forward(xj.float())
            score = self.wgate.forward(xj.float())
            self._kv_state[0, (start_pos + j) % ratio] = kv[0, 0]
            self._score_state[0, (start_pos + j) % ratio] = score[0, 0]
            if (start_pos + j + 1) % ratio == 0:
                pooled_parts.append(gated_pool(self._kv_state, self._score_state, x.dtype))
            if capture is not None:
                capture.append((self.layer_id, self.tier, self.snapshot_carry()))
        # Norm the closed groups in ONE 3-D call: flashinfer's 2-D rmsnorm_cute rejects
        # the tiny per-token rows; the 3-D path (qk_rmsnorm_cute) is what prefill uses.
        if pooled_parts:
            pooled = torch.stack(pooled_parts).reshape(1, len(pooled_parts), self.head_dim)
            latent = self.norm.forward(pooled)
        else:
            latent = x.new_zeros(1, 0, self.head_dim)
        # durable page-boundary carries + the tail page (radix-resume parity)
        kv_full, score_full = self.wkv.forward(x.float()), self.wgate.forward(x.float())
        end = start_pos + seqlen
        self._write_boundary_carries_range(kv_full, score_full, start_pos, end, window_slots)
        if end % self.P != 0:
            self._write_through_carry(int(window_slots[-1].item()))
        return latent

    def finalize(self, latent: torch.Tensor, start_pos: int, ti: int = 0) -> None:
        """Rope (group-start positions) + fp4/16 quant + scatter of the closed
        groups' latents to the paged cmp pool."""
        if latent is None or latent.size(1) == 0:
            return
        ratio, rd = self.compress_ratio, self.rope_head_dim
        n = latent.size(1)
        first_group = start_pos // ratio  # cold: 0; extend: S is 128-aligned
        freqs = self._freqs_cis.index_select(
            0, (first_group + torch.arange(n, device=latent.device)) * ratio
        )
        apply_rotary_emb(latent[..., -rd:], freqs)
        fp4_act_quant_inplace(latent, 16, torch.float8_e4m3fn)
        block_starts = (first_group + torch.arange(n, device=latent.device)) * ratio
        self.attn.scatter_compressed(
            self.layer_id, self.tier,
            self.attn.compress_rows_of(ti, block_starts, ratio), latent[0],
        )

    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor, rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched single-token compressor update; returns (normed PRE-rope
        latent [B, head_dim], completion mask [B]). Rows that close no group
        scatter to their own scratch row (graph-safe), same as the cmp write."""
        assert self.cmp_pool is not None
        B = x.size(0)
        ratio, rd = self.compress_ratio, self.rope_head_dim
        x = x.view(B, -1)
        kv = bf16_linear_fp32(x, self.wkv.weight).view(B, 1, self.head_dim)
        if self.gated:
            score = bf16_linear_fp32(x, self.wgate.weight).view(B, 1, self.head_dim)
        else:
            score = torch.zeros_like(kv)
        idx_mod = pos % ratio
        should = ((pos + 1) % ratio == 0).view(B, 1, 1)

        block = self.attn.read_carry_blocks(self.layer_id, self.tier, prev_window_slots, self.ring_size)
        ks = block[..., : self.item_size].clone()
        ss = block[..., self.item_size :].clone()
        ks.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, self.item_size), kv)
        ss.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, self.item_size), score)
        pooled = gated_pool(ks, ss, x.dtype)
        self.attn.write_carry_blocks(
            self.layer_id, self.tier, window_slots, self.ring_size, torch.cat([ks, ss], dim=-1)
        )
        latent = self.norm.forward(pooled)
        freqs_t = self._freqs_cis.index_select(0, (pos + 1 - ratio).clamp_min(0))
        # The indexer derives its K from the PRE-rope latent (reference model.py:434,
        # vLLM _produce_k), so rope/quantize only the copy that lands in the compressed
        # cache and return the unrotated latent for write_k_decode.
        roped = latent.clone()
        apply_rotary_emb_decode(roped[..., -rd:], freqs_t)
        fp4_act_quant_inplace(roped, 16, torch.float8_e4m3fn)
        cmp_dst = self.attn.decode_compress_rows(rows, pos, ratio, self.layer_id, self.tier, should.view(B))
        self.attn.scatter_compressed(self.layer_id, self.tier, cmp_dst, roped.view(B, -1))
        return latent, should.view(B)


__all__ = ["V41Compressor"]
