# Copyright (c) 2026 FreeToken contributors
# engine/draft_vocab.py: the frequency-adapted MTP draft vocabulary (tp=1, bf16 head).
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine import draft_vocab as dvmod
from freetoken.engine.draft_vocab import DraftVocabHead

V, K = 1000, 32


def _head():
    torch.manual_seed(0)
    return SimpleNamespace(
        vocab_range=(0, V), num_embeddings=V, tp_size=1, tied_embedding=None,
        weight=torch.randn(V, K, dtype=torch.bfloat16),
    )


def _subset_argmax(head, ids, x):
    logits = torch.nn.functional.linear(x, head.weight).float()
    sub = torch.tensor(sorted(ids))
    return sub[logits[:, sub].argmax(dim=-1)].to(torch.int32)


def test_inactive_until_warmup_then_argmax_over_the_frequent_set():
    head = _head()
    dv = DraftVocabHead(head, size=16, device=torch.device("cpu"))
    assert DraftVocabHead.supported(head)
    frequent = list(range(100, 116))
    dv.observe(frequent * (dvmod.WARMUP_TOKENS // 32), generated=False)  # prompt tokens count
    assert not dv.active
    dv.observe(frequent * (dvmod.WARMUP_TOKENS // 32 + 1))
    assert dv.active
    assert sorted(dv.ids.tolist()) == frequent
    x = torch.randn(3, K, dtype=torch.bfloat16)
    assert torch.equal(dv.argmax(x), _subset_argmax(head, frequent, x))


def test_free_slots_take_the_lowest_ids():
    head = _head()
    dv = DraftVocabHead(head, size=8, device=torch.device("cpu"))
    dv.observe([500] * dvmod.WARMUP_TOKENS)
    assert sorted(dv.ids.tolist()) == [0, 1, 2, 3, 4, 5, 6, 500]
    x = torch.randn(4, K, dtype=torch.bfloat16)
    assert torch.equal(dv.argmax(x), _subset_argmax(head, dv.ids.tolist(), x))


def test_a_shifted_stream_rebuilds_the_set():
    head = _head()
    dv = DraftVocabHead(head, size=8, device=torch.device("cpu"))
    dv.observe(list(range(8)) * (dvmod.WARMUP_TOKENS // 8))
    assert sorted(dv.ids.tolist()) == list(range(8))
    # generated tokens now come from a disjoint set: past MIN_WINDOW the misses trigger a
    # rebuild, and the aged counts let the new tokens take over
    dv.observe(list(range(500, 508)) * (dvmod.MIN_WINDOW // 8 + 50))
    assert sorted(dv.ids.tolist()) == list(range(500, 508))
    assert dv.window == 0 and dv.misses == 0
