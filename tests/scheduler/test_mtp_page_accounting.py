"""Speculative-round page accounting (MTP / DSpark rollback boundary).

The verify forward allocates pages over the whole `[C, C+k+1)` range. After publishing
the accepted drafts the round must roll back to the COMMITTED prefix (`rext_end`), not to
the new `device_len` (`keep_to = rext_end + 1`). The pending bonus token's page is
intentionally left to the NEXT step's ``allocate_paged`` (which covers
``[cached_len, device_len)``).

If the round rolls back only to ``keep_to``, the verify page/slot holding ``rext_end`` is
retained, and the next ``allocate_paged`` (``first_page = div_ceil(cached_len, page_size)``)
re-allocates that same index and orphans the retained page -> the idle
``check_integrity`` raises. With ``page_size == 1`` (the qwen4_exp FTW production config)
this happens on EVERY round where a draft was rejected (``a < k``); with ``page_size > 1``
it needs ``rext_end`` page-aligned. This reproduces the greedy crash on CPU, no engine.

CPU, real CacheManager, no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager

NUM_PAGES = 64
K = 3
C = 13  # cached_len at round entry
A = 2  # a < k: the verify allocated the pending slot at rext_end
REXT_END = C + 1 + A  # 16, page-aligned for every tested page_size
KEEP_TO = REXT_END + 1


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _req(cm: CacheManager) -> Req:
    """A decoding request at the MTP entry invariant: cached_len=C, device_len=C+1
    (the pending token at C is staged, its page not yet allocated)."""
    ids = list(range(1000, 1000 + C))
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=0,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=cm.match_req(_pend(ids)).cuda_handle,
    )
    cm.lock(req.cache_handle)
    req.device_len = C
    cm.allocate_paged([req])  # committed prompt [0, C)
    req.cached_len = C
    req.device_len = C + 1
    return req


def _owned(cm: CacheManager, req: Req, upto: int) -> set:
    return set(cm.page_table[req.table_idx, :upto].tolist())


def _round_then_next_allocate(page_size: int, rollback_to: str):
    """Drive one verify allocation, the rollback and the next plain decode allocate.

    Returns (owned_before, owned_after, leaked): the page bases the request named after
    the verify, those it names after the next allocate, and the ones that ended up
    neither named nor free (the orphan).
    """
    page_table = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, page_size, page_table, "radix")
    req = _req(cm)

    # -- verify: extend [C, C+k+1) --
    req.device_len = C + K + 1
    cm.allocate_paged([req])
    owned_before = _owned(cm, req, req.device_len)

    boundary = REXT_END if rollback_to == "rext_end" else KEEP_TO
    cm.rollback_last(req, boundary)
    req.cached_len = REXT_END
    req.device_len = KEEP_TO

    # -- next plain decode: allocate the pending token's page [cached_len, device_len) --
    req.device_len = REXT_END + 1
    cm.allocate_paged([req])
    owned_after = _owned(cm, req, req.device_len)
    free = set(cm.free_slots.tolist())
    return owned_before, owned_after, owned_before - owned_after - free


@pytest.mark.parametrize("page_size", [1, 2, 4, 8])
def test_rollback_to_committed_prefix_leaves_no_orphan_page(page_size):
    owned_before, owned_after, leaked = _round_then_next_allocate(page_size, "rext_end")
    assert leaked == set(), f"orphaned pages: {sorted(leaked)}"
    assert len(owned_after) == len(owned_before)


@pytest.mark.parametrize("page_size", [1, 2, 4, 8])
def test_rollback_to_device_len_orphans_the_pending_page(page_size):
    """Documents the bug the fix removes: keep_to retains the verify slot/page at
    rext_end and the next allocate re-allocates its index, orphaning the retained base."""
    _before, _after, leaked = _round_then_next_allocate(page_size, "keep_to")
    assert leaked, "expected the keep_to boundary to orphan a page"
