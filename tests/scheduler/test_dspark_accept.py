"""DSpark acceptance: greedy match + speculative-sampling residual.

The sampling path needs the GPU sampling kernels, but the greedy mapping and the residual
distribution are pure tensor math -- a regression there would silently corrupt spec output.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

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

    def forward_spec(self, ids, aux, start, row=0):
        # always drafts block_size tokens (anchor + block_size)
        self.rows = getattr(self, "rows", []) + [row]
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


def test_draft_block_uses_the_request_table_row_as_its_window_ring():
    mgr = _draft_mgr(k=2, block=5)
    mgr._aux = {3: torch.zeros(1, 4)}
    mgr._aux_start = {3: 7}
    mgr._draft_block([_DraftReq(3, 8, list(range(9)))], [8])
    assert mgr.draft.rows == [3]


class _Attn:
    window_size = 4


class _Layer:
    attn = _Attn()


class _Layers:
    op_list = [_Layer()]


class _DecodeModel:
    def __init__(self, rows):
        self.buf = rows

    def get_dspark_decode_aux(self):
        return self.buf


def _decode_mgr(buf):
    mgr = object.__new__(DSparkManager)
    mgr.enabled = True
    mgr.engine = _Eng()
    mgr.engine.model = _DecodeModel(buf)
    mgr.draft = _Draft(5)
    mgr.draft.layers = _Layers()
    mgr._aux = {0: torch.full((2, 3), 1.0)}
    mgr._aux_start = {0: 10}  # pending context covers positions 10, 11
    return mgr


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs


def test_note_decode_appends_the_serially_decoded_position_to_the_draft_context():
    buf = torch.tensor([[5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
    mgr = _decode_mgr(buf)
    mgr.note_decode(_Batch([_DraftReq(0, 12, [])]), [12])
    assert mgr._aux_start[0] == 10 and mgr._aux[0].shape[0] == 3
    assert torch.equal(mgr._aux[0][-1], buf[0])
    # two more steps: the context is capped at the draft window (4) and slides
    mgr.note_decode(_Batch([_DraftReq(0, 13, [])]), [13])
    mgr.note_decode(_Batch([_DraftReq(0, 14, [])]), [14])
    assert mgr._aux[0].shape[0] == 4 and mgr._aux_start[0] == 11


def test_note_decode_ignores_known_positions_and_restarts_after_a_hole():
    buf = torch.tensor([[7.0, 7.0, 7.0]])
    mgr = _decode_mgr(buf)
    mgr.note_decode(_Batch([_DraftReq(0, 11, [])]), [11])  # already in the context
    assert mgr._aux[0].shape[0] == 2 and mgr._aux_start[0] == 10
    mgr.note_decode(_Batch([_DraftReq(0, 20, [])]), [20])  # 12..19 never captured
    assert mgr._aux_start[0] == 20 and torch.equal(mgr._aux[0], buf[:1])


def test_a_new_request_on_a_row_gets_a_fresh_controller_and_latch():
    mgr = object.__new__(DSparkManager)
    mgr.draft = _Draft(5)
    mgr._ctl, mgr._latch, mgr._row_owner = {}, {}, {}
    first = SimpleNamespace(table_idx=2, uid=10)
    mgr._claim_row(first)
    ctl = mgr._controller(first)
    ctl.cooldown = 16  # the first request ended in a pause
    mgr._latch[2] = object()
    mgr._claim_row(first)  # same request, next chunk: state kept
    assert mgr._controller(first) is ctl and 2 in mgr._latch
    second = SimpleNamespace(table_idx=2, uid=11)
    mgr._claim_row(second)
    assert mgr._controller(second) is not ctl and mgr._controller(second).cooldown == 0
    assert 2 not in mgr._latch
