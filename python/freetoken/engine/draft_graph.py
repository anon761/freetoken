"""CUDA-graph capture for the MTP draft chain (Phase 3b).

The draft chain runs ``k`` sequential single-token forwards of the MTP block (a plain
bf16 resident MoE + QSA attention), each an extend of ``t=1`` token per request. The
attention backend only captures decode batches and the ``k+1`` verify extend, so this
runner captures the *single draft step* with its own static geometry -- one graph per
padded batch size (mirroring ``GraphRunner``/``VerifyGraphRunner``), replayed ``k``
times per round with the per-step positions / out_loc / MTP-KV length staged in
between. The ``R_next``/token hand-off between steps stays outside the graph (a device
copy and an argmax), so nothing here needs to know ``k``.

Everything inside the captured region is a device kernel; the eager Python (metadata
build, page allocation, per-step staging) stays outside. Off by default
(``--mtp-draft-graph`` / ``FREETOKEN_MTP_DRAFT_GRAPH=1``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.engine.graph import GraphRunner
    from freetoken.models import BaseLLMModel


class _DraftReq:
    """Minimal request view for the synthetic draft batch: the QSA metadata reads only
    ``table_idx`` (``ring_slots``) off the reqs; positions/out_loc ride on the batch."""

    __slots__ = ("table_idx", "cached_len", "device_len")

    def __init__(self, table_idx: int, cached_len: int, device_len: int) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _DraftBatch:
    """Synthetic prefill-phase batch for one captured draft step (t=1 token/req)."""

    def __init__(
        self, reqs: Sequence[_DraftReq], positions: torch.Tensor, out_loc: torch.Tensor
    ) -> None:
        self.reqs: List[_DraftReq] = list(reqs)
        self.positions = positions
        self.out_loc = out_loc
        self.phase = "prefill"
        self.attn_metadata = None

    @property
    def padded_reqs(self) -> List[_DraftReq]:
        return self.reqs

    @property
    def is_prefill(self) -> bool:
        return True

    @property
    def is_decode(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.reqs)


class DraftGraphRunner:
    """Captures (once) and replays one MTP draft step for each padded batch size."""

    def __init__(
        self,
        graph_runner: "GraphRunner",
        model: "BaseLLMModel",
        attn_backend: "BaseAttnBackend",
        bs_list: List[int],
        width: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.gr = graph_runner
        self.model = model
        self.attn = attn_backend
        self.mtp = model.mtp
        self.embed = model.model.embed_tokens.forward
        self.lm_head = model.lm_head
        self.bs_list = sorted(bs_list)
        self.n_max = max(self.bs_list)
        self.width = width
        self.device = device
        self.dummy_table_idx = graph_runner.dummy_req.table_idx
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # Private memory pool: the draft graphs must NOT share the decode/verify pool, or a
        # replay of one overwrites the other's baked transients.
        self.pool = None

        self.R_in = torch.zeros(self.n_max, width, dtype=torch.bfloat16, device=device)
        self.token_in = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        # The MTP block's output stream is the resident bf16 draft dtype.
        self.R_out = torch.empty(self.n_max, width, dtype=torch.bfloat16, device=device)
        self.logits = torch.empty(self.n_max, vocab_size, dtype=torch.float32, device=device)
        # Padding rows: position 0 and the dummy page's slot (safe no-op writes).
        self.pad_positions = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.pad_out_loc = (
            get_global_ctx().page_table[self.dummy_table_idx, :1].to(torch.int32)
        ).repeat(self.n_max).contiguous()

        self.attn.init_capture_draft(self.n_max)
        for bs in self.bs_list:
            self._capture(bs)
        # The QSA/FLA kernels call @tensor_cache helpers whose result tensors are baked into
        # these graphs; pin the current entries so a later eager call cannot evict them.
        from freetoken.kernel.fla.utils import pin_all_tensor_caches

        pin_all_tensor_caches()
        self.max_kv = getattr(attn_backend, "graph_extend_max_kv", None)
        logger.info_rank0(f"MTP draft CUDA graphs captured: bs={self.bs_list}")

    # ------------------------------------------------------------------ capture

    def _capture(self, bs: int) -> None:
        dummy = self.gr.dummy_req
        reqs = [_DraftReq(dummy.table_idx, 0, 1) for _ in range(bs)]
        self.positions[:bs] = self.pad_positions[:bs]
        self.out_loc[:bs] = self.pad_out_loc[:bs]
        sb = _DraftBatch(reqs, self.positions[:bs], self.out_loc[:bs])
        self.attn.set_draft_n(bs)
        self.attn.stage_draft_round(
            [dummy.table_idx] * bs, [1] * bs, pad_table_idx=dummy.table_idx
        )
        graph = torch.cuda.CUDAGraph()
        with self.attn.draft_scratch():
            sb.attn_metadata = self.attn.make_draft_metadata()
            with get_global_ctx().forward_batch(sb):
                self._forward(bs, sb)  # eager warmup (materializes scratch)
                # The MTP block is the only QSA layer here, so qsa_forward's slot-0
                # ``_plan_index_writes`` guard does not fire during capture; clear the
                # warmup plan so the per-step index computation is captured (its stale
                # warmup rows would otherwise be replayed against real geometry).
                sb.attn_metadata.cmp_rows = None
                with torch.cuda.graph(graph, pool=self.pool, stream=self.gr.stream):
                    self._forward(bs, sb)
        self.graph_map[bs] = graph
        if self.pool is None:
            self.pool = graph.pool()  # reuse this graph's mempool for the remaining sizes

    def _forward(self, bs: int, sb: _DraftBatch) -> None:
        R_next, sample_hidden = self.mtp.draft_step(
            self.embed, self.R_in[:bs], self.token_in[:bs], sb
        )
        self.R_out[:bs].copy_(R_next)
        # forward_all, NOT forward: the draft batch is phase="prefill", and forward's
        # last-row gather (``x = x[attn_metadata.last_indices]``) indexes a capture-time
        # temporary that is freed once the capture returns, so replaying the graph read
        # freed memory -> device-side OOB. At t=1 every row IS the last row, so
        # forward_all returns the same logits the draft chain needs.
        self.logits[:bs] = self.lm_head.forward_all(sample_hidden)

    # ------------------------------------------------------------------ replay

    def can_use(self, n: int, max_kv: int | None = None) -> bool:
        return (
            0 < n <= self.n_max
            and (max_kv is None or self.max_kv is None or max_kv <= self.max_kv)
        )

    def _select_bs(self, n: int) -> int:
        return next(bs for bs in self.bs_list if bs >= n)

    @torch.inference_mode()
    def replay_step(
        self,
        table_idx: Sequence[int],
        kvlen: Sequence[int],
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        R_in: torch.Tensor,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage one draft step's geometry + inputs into the static buffers and replay the
        smallest captured size ``>=`` the request count. Returns ``(R_next[:n], logits[:n])``."""
        n = len(table_idx)
        assert self.can_use(n), f"draft n={n} outside captured range (max {self.n_max})"
        bs = self._select_bs(n)
        self.R_in[:bs] = 0
        self.R_in[:n].copy_(R_in)
        self.token_in[:bs] = 0
        self.token_in[:n].copy_(tokens)
        self.positions[:bs] = self.pad_positions[:bs]
        self.positions[:n].copy_(positions)
        self.out_loc[:bs] = self.pad_out_loc[:bs]
        self.out_loc[:n].copy_(out_loc)
        self.attn.set_draft_n(bs)
        self.attn.stage_draft_round(list(table_idx), list(kvlen), pad_table_idx=self.dummy_table_idx)
        self.graph_map[bs].replay()
        return self.R_out[:n], self.logits[:n]


__all__ = ["DraftGraphRunner"]
