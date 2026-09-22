"""CacheManager.rollback_last — the MTP verify rollback.

After a k+1-token speculative verify, the rejected positions' KV pages must return to
the free list while every accepted position's slot stays owned by the request. Whole-page
semantics: only pages FULLY inside the rejected region are freed (a partial page holding
accepted tokens keeps its slots; the whole page returns at commit/finish via _padded_tail).
CPU, real CacheManager, no engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _req(cm: CacheManager, table_idx: int, prompt_len: int, gen_len: int) -> Req:
    """A decoding request with [0, prompt_len) committed and gen_len verify tokens
    appended (device_len = prompt_len + gen_len), pages allocated like allocate_paged."""
    ids = list(range(100, 100 + prompt_len + gen_len))
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=0,
        uid=table_idx,
        sampling_params=SamplingParams(),
        cache_handle=cm.match_req(_pend(ids[:prompt_len])).cuda_handle,
    )
    req.device_len = prompt_len + gen_len
    cm.lock(req.cache_handle)
    # allocate the prompt in one go ...
    cm.allocate_paged([req])
    req.cached_len = prompt_len
    # ... then the verify extend [prompt_len, prompt_len + gen_len)
    cm.allocate_paged([req])
    return req


def _row(cm: CacheManager, req: Req, upto: int) -> set:
    return set(cm.page_table[req.table_idx, :upto].tolist())


def test_page_size_one_frees_exactly_the_rejected_tokens():
    cm = CacheManager(64, 1, torch.zeros(8, 64, dtype=torch.int32), "radix")
    req = _req(cm, 0, prompt_len=4, gen_len=5)
    free_before = len(cm.free_slots)

    keep_to = 4 + 2  # accept the pending token + 1 draft; reject 3
    cm.rollback_last(req, keep_to)
    req.device_len = keep_to
    req.input_ids = req._ids_buf[:keep_to]

    assert len(cm.free_slots) == free_before + 3
    # accepted slots stay owned (not on the free list) and the row still names them
    kept = _row(cm, req, keep_to)
    assert kept.isdisjoint(set(cm.free_slots.tolist()))
    assert len(kept) == keep_to


def test_whole_page_rejection_frees_the_full_page():
    cm = CacheManager(8, 4, torch.zeros(4, 32, dtype=torch.int32), "radix")
    req = _req(
        cm, 0, prompt_len=6, gen_len=6
    )  # verify extend spans pages 1 (4..7) + 2 (8..11)
    free_before = len(cm.free_slots)

    keep_to = 8  # rejected region [8, 12) = exactly page 2
    cm.rollback_last(req, keep_to)
    req.device_len = keep_to
    req.input_ids = req._ids_buf[:keep_to]

    assert len(cm.free_slots) == free_before + 1  # one page base
    kept = _row(cm, req, keep_to)
    assert kept.isdisjoint(set(cm.free_slots.tolist()))
    assert len(kept) == keep_to


def test_partial_rejected_tail_page_stays_until_the_page_returns_whole():
    cm = CacheManager(8, 4, torch.zeros(4, 32, dtype=torch.int32), "radix")
    req = _req(cm, 0, prompt_len=6, gen_len=6)
    free_before = len(cm.free_slots)

    keep_to = (
        9  # rejected [9, 12): page 2 holds accepted position 8 -> nothing freeable yet
    )
    cm.rollback_last(req, keep_to)
    req.device_len = keep_to
    req.input_ids = req._ids_buf[:keep_to]

    assert len(cm.free_slots) == free_before
    kept = _row(cm, req, keep_to)
    assert kept.isdisjoint(set(cm.free_slots.tolist()))


def test_page_aligned_rejection_frees_through_the_allocated_tail():
    cm = CacheManager(8, 4, torch.zeros(4, 32, dtype=torch.int32), "radix")
    req = _req(cm, 0, prompt_len=6, gen_len=6)
    free_before = len(cm.free_slots)

    keep_to = 8  # whole-page rejection: page 2 (incl. its padding) returns
    cm.rollback_last(req, keep_to)
    req.device_len = keep_to

    assert len(cm.free_slots) == free_before + 1  # one page base


def test_rollback_never_touches_the_tree_owned_prefix():
    cm = CacheManager(32, 1, torch.zeros(4, 32, dtype=torch.int32), "radix")
    req = _req(cm, 0, prompt_len=4, gen_len=4)
    # commit the prompt so the handle (tree-owned prefix) covers [0, 4)
    cm.cache_req(req, finished=False)
    assert req.cache_handle.cached_len == 4

    with pytest.raises(AssertionError):
        cm.rollback_last(req, 2)  # below cache_handle.cached_len
