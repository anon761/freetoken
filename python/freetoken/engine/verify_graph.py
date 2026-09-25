"""CUDA-graph capture for the MTP verify extend (Phase 3).

The MTP verify is a *prefill-phase* extend over a shape ``n x t`` (``t = k+1`` tokens per
request). The attention backends only capture decode batches, so this runner captures the
verify itself with its own static geometry -- one graph per padded batch size (mirroring
``GraphRunner``'s decode graph list), so a round with ``n`` requests replays the smallest
captured size ``>= n`` instead of always paying the largest. Unused rows are padded onto the
dummy page/slot so they never touch a real request's KV or GDN state.

* ``input_ids`` / ``positions`` / ``out_loc`` / ``gdn_slots`` are staged per round;
* the QSA backend's prefill metadata (constant ``cu_seqlens`` / ``token_to_req``; per-round
  ``kvlen`` / ``ring_slots`` / ``block_table``) is staged via ``stage_verify_round``; the
  active row count is selected with ``set_verify_n``;
* the GDN metadata (constant ``cu_seqlens``, per-round ``cache_indices``, all-continuing
  ``has_initial_state``) points at static buffers; the GDN verify writes its commit inputs
  into static per-layer buffers, so ``commit_verify`` works after replay.

Everything inside the captured region is a device kernel; the eager Python (metadata
build, page allocation, paged gather) stays outside. Off by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.engine.graph import GraphRunner
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache


class VerifyGraphRunner:
    """Captures (once) and replays the MTP verify extend for each padded batch size."""

    def __init__(
        self,
        graph_runner: "GraphRunner",
        model: "BaseLLMModel",
        attn_backend: "BaseAttnBackend",
        moe_offload_cache: "OffloadMoeCache | None",
        bs_list: List[int],
        t: int,
        vocab_size: int,
        device: torch.device,
        padding_slot: int,
    ) -> None:
        self.gr = graph_runner
        self.model = model
        self.attn = attn_backend
        self.moe = moe_offload_cache
        self.bs_list = sorted(bs_list)
        self.n_max = max(self.bs_list)
        self.t = t
        self.total = self.n_max * t
        self.device = device
        self.padding_slot = padding_slot
        self.dummy_table_idx = graph_runner.dummy_req.table_idx
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.streams_map: Dict[int, torch.Tensor] = {}

        # Static geometry sized for the largest captured size; each graph uses a prefix.
        self.input_ids = torch.zeros(self.total, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.total, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(self.total, dtype=torch.int32, device=device)
        self.gdn_slots = torch.full((self.n_max,), padding_slot, dtype=torch.int32, device=device)
        self.has_init = torch.ones(self.n_max, dtype=torch.bool, device=device)
        self.cu_seqlens = torch.arange(self.n_max + 1, dtype=torch.int32, device=device) * t
        self.logits = torch.empty(self.total, vocab_size, dtype=torch.float32, device=device)
        # Padding rows: dummy positions and the dummy page's slots (safe no-op writes).
        self.pad_positions = torch.arange(t, dtype=torch.int32, device=device).repeat(self.n_max)
        dummy_row = get_global_ctx().page_table[self.dummy_table_idx, :t].to(torch.int32)
        self.pad_out_loc = dummy_row.repeat(self.n_max).contiguous()

        self.attn.init_capture_verify(self.n_max, t)
        # Descending: the largest capture allocates the per-layer GDN verify buffers, smaller
        # graphs then reuse them (gdn_verify.ensure_verify_buffers keeps the max).
        for bs in sorted(self.bs_list, reverse=True):
            self._capture(bs)

    # ------------------------------------------------------------------ capture

    def _fla_metadata(self, bs: int):
        from freetoken.attention.linear import FLAMetadata

        return FLAMetadata(
            cu_seqlens=self.cu_seqlens[: bs + 1],
            cache_indices=self.gdn_slots[:bs],
            has_initial_state=self.has_init[:bs],
        )

    def _capture(self, bs: int) -> None:
        dummy = self.gr.dummy_req
        total = bs * self.t
        batch = Batch(reqs=[dummy] * bs, phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = self.input_ids[:total]
        batch.positions = self.positions[:total]
        batch.out_loc = self.out_loc[:total]
        batch.linear_table_idx = self.gdn_slots[:bs]
        # Enter the GDN verify branch so its per-token input capture (the Phase-2 commit's
        # source) is part of the captured kernels; the fixed shape rides on the batch.
        batch.mtp_verify = True
        batch.mtp_verify_n = bs
        batch.mtp_verify_t = self.t
        # PLE derives seq_lens from the reqs; the dummy reqs do not carry the verify's
        # extend length, so give the fixed tuple explicitly (captured as a constant).
        batch.ple_seq_lens = (self.t,) * bs
        # Valid dummy geometry for the capture (a real capture with seq_lens=0 or duplicate
        # positions trips device asserts in the QSA index/attention kernels).
        self.positions[:total] = self.pad_positions[:total]
        self.out_loc[:total] = self.pad_out_loc[:total]
        self.attn.set_verify_n(bs)
        self.attn.stage_verify_round(
            [self.dummy_table_idx] * bs,
            [self.t] * bs,
            pad_table_idx=self.dummy_table_idx,
        )
        graph = torch.cuda.CUDAGraph()
        with self.attn.verify_scratch():
            batch.attn_metadata = self.attn.make_verify_metadata()
            batch.fla_metadata = self._fla_metadata(bs)
            self._reset_moe()
            with get_global_ctx().forward_batch(batch):
                self.model.forward_logits_all()  # eager warmup (materializes scratch)
                self._reset_moe()
                with torch.cuda.graph(
                    graph, pool=getattr(self.gr, "pool", None), stream=self.gr.stream
                ):
                    self.logits[:total] = self.model.forward_logits_all()
                self._reset_moe()
        self.graph_map[bs] = graph
        self.streams_map[bs] = getattr(batch, "mtp_streams", None)

    def _reset_moe(self) -> None:
        if self.moe is not None:
            self.moe.reset()

    # ------------------------------------------------------------------ replay

    def can_use(self, n: int) -> bool:
        return 0 < n <= self.n_max

    def _select_bs(self, n: int) -> int:
        return next(bs for bs in self.bs_list if bs >= n)

    @torch.inference_mode()
    def replay(self, batch: Batch):
        """Stage ``batch``'s geometry into the static buffers and replay the smallest
        captured size ``>= batch.size``. ``batch`` must be a prepared prefill verify batch
        (page allocation done) with ``positions``/``out_loc`` set by batch preparation and
        ``input_ids`` gathered by the caller."""
        n = batch.size
        assert self.can_use(n), f"verify n={n} outside captured range (max {self.n_max})"
        bs = self._select_bs(n)
        real = n * self.t
        total = bs * self.t
        self.input_ids[:total] = 0
        self.input_ids[:real].copy_(batch.input_ids)
        self.positions[:total] = self.pad_positions[:total]
        self.positions[:real].copy_(batch.positions)
        self.out_loc[:total] = self.pad_out_loc[:total]
        self.out_loc[:real].copy_(batch.out_loc)
        reqs = batch.padded_reqs
        self.gdn_slots[:bs].fill_(self.padding_slot)
        self.gdn_slots[:n].copy_(
            torch.tensor(
                [
                    (r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx)
                    for r in reqs
                ],
                dtype=torch.int32,
                device=self.device,
            )
        )
        self.attn.set_verify_n(bs)
        self.attn.stage_verify_round(
            [r.table_idx for r in reqs],
            [r.device_len for r in reqs],
            pad_table_idx=self.dummy_table_idx,
        )
        batch.input_ids = self.input_ids[:real]
        batch.positions = self.positions[:real]
        batch.out_loc = self.out_loc[:real]
        batch.linear_table_idx = self.gdn_slots[:bs]
        batch.ple_seq_lens = (self.t,) * bs
        with self.attn.verify_scratch():
            batch.attn_metadata = self.attn.make_verify_metadata()
            batch.fla_metadata = self._fla_metadata(bs)
            self.graph_map[bs].replay()
        streams = self.streams_map.get(bs)
        return self.logits[:real], (streams[:real] if streams is not None else None)


__all__ = ["VerifyGraphRunner"]
