"""Frequency-adapted draft vocabulary for the MTP draft chain.

Each draft step only needs the argmax token, but the shared lm_head reads the whole vocab
shard (~750 us/step for Qwen3.8-27B's fp8 head on an RTX 3090). The draft proposes
frequent tokens almost always, so ``DraftVocabHead`` keeps the lm_head rows of ``V`` likely
tokens and takes the argmax over those only.

The set adapts to the traffic: prompt and generated tokens are counted, the most frequent
ones fill the set and the remaining slots take the lowest token ids (byte-level BPE assigns
those to the most frequent merges). It is rebuilt periodically, and immediately when the
generated tokens keep falling outside it (miss rate above ``MISS_LIMIT``). A missed token
only costs acceptance (the target verifies every draft), never correctness. The rows live
in static buffers rewritten in place, so a captured draft graph keeps valid addresses;
every TP rank sees the same token stream and builds the same set.
"""

from __future__ import annotations

import numpy as np
import torch
from freetoken.utils import init_logger

logger = init_logger(__name__)

WARMUP_TOKENS = 4096     # counted tokens (prompt + generated) before the first set is built
REBUILD_TOKENS = 16384   # generated tokens between periodic rebuilds
MIN_WINDOW = 2048        # generated tokens before a miss rate is judged
MISS_LIMIT = 0.03        # generated tokens outside the set that trigger an early rebuild


class DraftVocabHead:
    def __init__(self, lm_head, size: int, device: torch.device) -> None:
        self.lm_head = lm_head
        self.size = size
        start, count = lm_head.vocab_range
        self.start, self.count = start, count
        weight = lm_head.weight
        self.fp8 = weight.dtype == torch.float8_e4m3fn
        # rows of THIS rank's shard; unused rows carry id -1 and are masked out
        self.w = torch.zeros(size, weight.shape[1], dtype=weight.dtype, device=device)
        self.scale = torch.ones(size, dtype=torch.float32, device=device)
        self.ids = torch.full((size,), -1, dtype=torch.int32, device=device)
        self.counts = np.zeros(lm_head.num_embeddings, dtype=np.int64)
        self.member = np.zeros(lm_head.num_embeddings, dtype=bool)
        self.active = False
        self.seen = 0
        self.window = 0
        self.misses = 0

    @staticmethod
    def supported(lm_head) -> bool:
        """fp8 per-row (W8A16) or plain bf16/fp16 heads; tied or other formats keep the full
        vocab."""
        if lm_head.tied_embedding is not None:
            return False
        w = getattr(lm_head, "weight", None)
        if w is None or w.dim() != 2:
            return False
        if w.dtype == torch.float8_e4m3fn:
            s = getattr(lm_head, "weight_scale", None)
            return s is not None and s.dim() == 1 and s.shape[0] == w.shape[0]
        return w.dtype in (torch.bfloat16, torch.float16)

    # ------------------------------------------------------------------ device side

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy draft token per row ([T] int32) over the current set (graph-capturable)."""
        if self.fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

            logits = fp8_pertensor_linear(x, self.w, self.scale)
        else:
            logits = torch.nn.functional.linear(x, self.w)
        logits = logits.float().masked_fill(self.ids < 0, float("-inf"))
        val, idx = logits.max(dim=-1)
        ids = self.ids[idx]
        if self.lm_head.tp_size == 1:
            return ids
        pairs = torch.stack([val, ids.view(torch.float32)], dim=-1)
        # as raw bytes: the pynccl transport has no fp32 all_gather, uint8 is byte-transparent
        gathered = self.lm_head._comm.all_gather(pairs.view(torch.uint8)).view(torch.float32)
        gathered = gathered.view(self.lm_head.tp_size, -1, 2)
        best = gathered[..., 0].argmax(dim=0)
        rows = torch.arange(best.shape[0], device=best.device)
        return gathered[best, rows, 1].contiguous().view(torch.int32)

    # ------------------------------------------------------------------ host side

    def observe(self, tokens: list[int], generated: bool = True) -> None:
        """Count ``tokens``; generated ones also feed the miss rate and the rebuild window."""
        if not tokens:
            return
        toks = np.asarray(tokens, dtype=np.int64)
        toks = toks[(toks >= 0) & (toks < self.counts.shape[0])]
        np.add.at(self.counts, toks, 1)
        self.seen += toks.size
        if not self.active:
            if self.seen >= WARMUP_TOKENS:
                self._rebuild()
            return
        if not generated:
            return
        self.window += toks.size
        self.misses += int((~self.member[toks]).sum())
        if self.window >= REBUILD_TOKENS or (
            self.window >= MIN_WINDOW and self.misses > MISS_LIMIT * self.window
        ):
            self._rebuild()

    @torch.inference_mode()
    def _rebuild(self) -> None:
        seen = np.flatnonzero(self.counts)
        top = seen[np.argsort(-self.counts[seen], kind="stable")][: self.size]
        self.member[:] = False
        self.member[top] = True
        if top.size < self.size:  # fill the rest with the lowest (most frequent BPE) ids
            fill = np.flatnonzero(~self.member)[: self.size - top.size]
            self.member[fill] = True
        chosen = np.flatnonzero(self.member)
        mine = chosen[(chosen >= self.start) & (chosen < self.start + self.count)]
        n = mine.size
        rows = torch.from_numpy(mine - self.start).to(self.w.device)
        self.w[:n].copy_(self.lm_head.weight[rows])
        if self.fp8:
            self.scale[:n].copy_(self.lm_head.weight_scale[rows])
        self.ids.fill_(-1)
        self.ids[:n].copy_(torch.from_numpy(mine.astype(np.int32)))
        logger.info_rank0(
            f"MTP draft vocab rebuilt: {top.size} counted + {chosen.size - top.size} by id "
            f"({self.seen} tokens counted; last window {self.misses}/{self.window} missed)"
        )
        self.counts >>= 1  # age the counts so the set follows a shifting distribution
        self.active = True
        self.window = 0
        self.misses = 0


__all__ = ["DraftVocabHead"]
