"""``--swa-eviction-interval``: env -> flag, and 0 disables proactive SWA eviction.

This was measured with a local ``FREETOKEN_SWA_NO_EVICT`` gate; this turns the same
behavior into a first-class parameter while keeping the env as the fallback
default (and 0 as the "off" spelling).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.scheduler.cache import CacheManager, _resolve_swa_eviction_interval
from freetoken.server.args import parse_args


def test_resolver_precedence():
    assert _resolve_swa_eviction_interval(7) == 7       # explicit flag wins
    assert _resolve_swa_eviction_interval(0) == 0       # 0 disables
    assert _resolve_swa_eviction_interval(None) == 128  # default


def test_resolver_env_fallback(monkeypatch):
    monkeypatch.setenv("FREETOKEN_SWA_EVICTION_INTERVAL", "64")
    assert _resolve_swa_eviction_interval(None) == 64
    monkeypatch.setenv("FREETOKEN_SWA_EVICTION_INTERVAL", "nonsense")
    with pytest.raises(ValueError):
        _resolve_swa_eviction_interval(None)


class _Boom:
    """Any attribute access means the eviction body ran."""

    def __getattr__(self, name):
        raise AssertionError("eviction body ran")


def _cm(interval: int | None) -> CacheManager:
    pool = SimpleNamespace(swa_paged=True)
    pt = torch.zeros((4, 16), dtype=torch.int32)
    return CacheManager(
        num_pages=8, page_size=1, page_table=pt, type="radix",
        swa_pool=pool, sliding_window_size=4, swa_eviction_interval=interval,
    )


def test_zero_interval_disables_eviction():
    cm = _cm(0)
    assert cm.swa_eviction_interval == 0
    cm.maybe_free_swa_out_of_window([_Boom()], forward_iter=1)
    cm.free_swa_out_of_window_extend([_Boom()])


def test_positive_interval_reaches_the_eviction_body():
    cm = _cm(1)
    with pytest.raises(AssertionError):
        cm.maybe_free_swa_out_of_window([_Boom()], forward_iter=1)
    with pytest.raises(AssertionError):
        cm.free_swa_out_of_window_extend([_Boom()])


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(extra: list[str]):
    config = _Config({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_parse_args_carries_the_interval():
    assert _parse([]).swa_eviction_interval is None
    assert _parse(["--swa-eviction-interval", "0"]).swa_eviction_interval == 0
    assert _parse(["--swa-eviction-interval", "256"]).swa_eviction_interval == 256
