"""DSpark acceptance: greedy match + speculative-sampling residual.

The sampling path needs the GPU sampling kernels, but the greedy mapping and the residual
distribution are pure tensor math -- a regression there would silently corrupt spec output.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from freetoken.engine.sample import sample_residual
from freetoken.scheduler.dspark import DSparkManager


class _Eng:
    device = torch.device("cpu")


class _Sampling:
    def __init__(self, greedy: bool) -> None:
        self.is_greedy = greedy


class _Req:
    def __init__(self, greedy: bool = True) -> None:
        self.sampling_params = _Sampling(greedy)


def _mgr() -> DSparkManager:
    mgr = object.__new__(DSparkManager)
    mgr.engine = _Eng()
    return mgr


def _logits(preds: list[int], vocab: int = 8) -> torch.Tensor:
    """One-hot target logits: row j predicts token ``preds[j]``."""
    out = torch.zeros(len(preds), vocab)
    for j, p in enumerate(preds):
        out[j, p] = 10.0
    return out


def test_greedy_all_accepted_publishes_bonus_at_last_position():
    k = 3
    drafts = torch.tensor([[0, 2, 3, 4]])  # anchor + 3 drafts
    logits = _logits([2, 3, 4, 6])  # rows predict C+1..C+k+1
    a, bonus = DSparkManager._accept(_mgr(), [_Req()], logits, drafts, k)
    assert a == [k] and bonus == [6]


def test_greedy_first_draft_rejected_corrects_from_target():
    k = 3
    drafts = torch.tensor([[0, 9, 3, 4]])  # draft 1 = 9 does not match target row 0
    logits = _logits([2, 3, 4, 6])
    a, bonus = DSparkManager._accept(_mgr(), [_Req()], logits, drafts, k)
    assert a == [0] and bonus == [2]


def test_greedy_second_draft_rejected():
    k = 3
    drafts = torch.tensor([[0, 2, 9, 4]])
    logits = _logits([2, 3, 4, 6])
    a, bonus = DSparkManager._accept(_mgr(), [_Req()], logits, drafts, k)
    assert a == [1] and bonus == [3]


def test_residual_zeroes_rejected_token_and_renormalizes():
    probs = torch.tensor([0.5, 0.3, 0.2])
    torch.manual_seed(0)
    tok = sample_residual(probs, rejected=0)
    assert int(tok) in (1, 2)  # mass on token 0 removed
    assert int(tok) != 0


def test_residual_degenerate_falls_back_to_argmax():
    probs = torch.tensor([1.0, 0.0, 0.0])
    assert int(sample_residual(probs, rejected=0)) == 0


class _Draft:
    def __init__(self, block: int) -> None:
        self.block_size = block

    def forward_spec(self, ids, aux, start):
        # always drafts block_size tokens (anchor + block_size)
        out = torch.arange(self.block_size + 1).view(1, -1).repeat(ids.size(0), 1)
        return out, None, None


class _Ctx:
    def forward_batch(self, batch):
        return nullcontext()


class _DraftReq:
    def __init__(self, table_idx: int, cached_len: int, ids: list[int]) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.input_ids = torch.tensor(ids)


def _draft_mgr(k: int, block: int) -> DSparkManager:
    mgr = object.__new__(DSparkManager)
    mgr.engine = _Eng()
    mgr.engine.ctx = _Ctx()
    mgr.draft = _Draft(block)
    mgr.k = k
    mgr._aux = {0: torch.zeros(1, 4)}
    mgr._aux_start = {0: 7}
    return mgr


def test_draft_block_truncates_to_k_so_k_smaller_than_block_size_is_consistent():
    # forward_spec drafts block_size=5, but the verify is k+1 long: the manager must
    # keep only the first k drafts or the _ids_buf slice (size k) rejects the tensor.
    mgr = _draft_mgr(k=2, block=5)
    req = _DraftReq(table_idx=0, cached_len=7, ids=[0, 1, 2, 3, 4, 5, 6, 99])
    out = mgr._draft_block([req], [7])
    assert out.shape == (1, 3)  # anchor + k drafts
    assert out[0].tolist() == [0, 1, 2]
