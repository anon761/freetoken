"""CUDA-graph capture for the MTP GDN commit (Phase 2 follow-up, n=1).

``commit_mtp_verify`` replays the accepted verify prefix into every GDN layer's live conv +
SSM state via a per-token decode loop: ~2 launches per GDN layer per accepted token, which
made the commit the round's launch-bound tail (~10 ms). Capture one graph per accepted-token
count ``a in [1, k+1]`` for the single-request case (the batch=1 deployment); the graph reads
the SAME static verify-capture buffers and writes the live GDN slot, which is staged into the
prep's slot tensor before each replay. n > 1 stays eager (the per-request accepted-count
vector would explode the graph set).

Requires the verify graph (its capture allocates the per-layer ``_mtp_capture`` buffers) and a
linear_state_pool. Opt-in via ``--mtp-commit-graph`` / ``FREETOKEN_MTP_COMMIT_GRAPH=1``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class CommitGraphRunner:
    def __init__(self, model, pool, k: int, device: torch.device, stream: torch.cuda.Stream) -> None:
        self.model = model
        self.pool = pool
        self.k = k
        self.device = device
        self.stream = stream
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._preps: Dict[int, Tuple] = {}
        for a in range(1, k + 2):
            self._capture(a)

    @staticmethod
    def _build_prep(a: int, device: torch.device):
        """n=1 prep: one sequence of ``a`` tokens, its live slot staged into ``idx_all`` (kept
        as the SAME tensor object the graph reads, so staging a new slot per round works)."""
        cu_t = torch.tensor([0, a], dtype=torch.int32, device=device)
        idx_all = torch.zeros(1, dtype=torch.int32, device=device)
        has_init = torch.ones(1, dtype=torch.bool, device=device)
        sub = torch.tensor([0], dtype=torch.long, device=device)
        cu_sub = torch.arange(2, dtype=torch.int32, device=device)
        steps = [(sub, idx_all, cu_sub) for _ in range(a)]
        return (cu_t, idx_all, has_init, steps)

    def _run(self, a: int) -> None:
        self.model.commit_mtp_verify_prep(self.pool, self._preps[a], [a])

    def _capture(self, a: int) -> None:
        prep = self._build_prep(a, self.device)
        prep[1][0] = self.pool.padding_slot  # capture writes a scratch slot, never a real req
        self._preps[a] = prep
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._run(a)  # warmup (materialize kernels/allocations)
        torch.cuda.current_stream().wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self._run(a)
        self.graphs[a] = graph

    def can_use(self, n: int, a: int) -> bool:
        return n == 1 and a in self.graphs

    def replay(self, live_slot: int, a: int) -> None:
        self._preps[a][1][0] = live_slot
        self.graphs[a].replay()


__all__ = ["CommitGraphRunner"]
