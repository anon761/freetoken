"""CUDA-graph capture for the MTP GDN commit.

``commit_mtp_verify`` rewrites every GDN layer's live conv slot and advances its recurrent
state over the accepted verify prefix: ~3 small kernels per GDN layer, i.e. a launch-bound
tail when run eagerly. The live slots and accepted lengths are device tensors, so ONE graph
per padded batch size covers every accepted count; unused rows are padded onto the pool's
padding slot.

Requires the verify graph (its largest capture allocates the per-layer verify buffers the
commit reads) and a linear_state_pool. Opt-in via ``--mtp-commit-graph`` /
``FREETOKEN_MTP_COMMIT_GRAPH=1``.
"""

from __future__ import annotations

from typing import Dict, List

import torch


class CommitGraphRunner:
    def __init__(
        self, model, pool, bs_list: List[int], t: int, device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        self.model = model
        self.pool = pool
        self.bs_list = sorted(bs_list)
        self.n_max = max(self.bs_list)
        self.t = t
        self.stream = stream
        self.slots = torch.full((self.n_max,), pool.padding_slot, dtype=torch.int32, device=device)
        self.lens = torch.full((self.n_max,), t, dtype=torch.int32, device=device)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        for bs in self.bs_list:
            self._capture(bs)

    def _run(self, bs: int) -> None:
        self.model.commit_mtp_verify(self.pool, self.slots[:bs], self.lens[:bs])

    def _capture(self, bs: int) -> None:
        # capture writes the padding slot only (slots are all padding at this point)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._run(bs)  # warmup (materialize kernels/allocations)
        torch.cuda.current_stream().wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self._run(bs)
        self.graphs[bs] = graph

    def can_use(self, n: int) -> bool:
        return 0 < n <= self.n_max

    def replay(self, slots: List[int], lens: List[int]) -> None:
        n = len(slots)
        bs = next(b for b in self.bs_list if b >= n)
        self.slots[:bs].fill_(self.pool.padding_slot)
        self.lens[:bs].fill_(self.t)
        self.slots[:n].copy_(torch.tensor(slots, dtype=torch.int32))
        self.lens[:n].copy_(torch.tensor(lens, dtype=torch.int32))
        self.graphs[bs].replay()


__all__ = ["CommitGraphRunner"]
