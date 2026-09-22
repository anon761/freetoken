"""FREETOKEN_TP_REDUCE_FP8: the threshold parser, the group-wide flag agreement
and the fp8 quantized-wire all_reduce branch (2-rank stand-in)."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import impl
from freetoken.distributed.impl import (
    PyNCCLDistributedImpl,
    _agree_flag,
    _env_flag,
    _fp8_threshold_bytes,
)

FLAG = "FREETOKEN_TP_REDUCE_FP8"


@pytest.fixture
def tp2(monkeypatch):
    """Present a 2-rank group to the impl without the one-shot global tp info."""
    monkeypatch.setattr(
        "freetoken.distributed.info.get_tp_info", lambda: SimpleNamespace(size=2)
    )


# --- threshold parser -------------------------------------------------------


def test_unset_defaults_to_64k(monkeypatch):
    monkeypatch.delenv("FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", raising=False)
    assert _fp8_threshold_bytes() == 64 * 1024


def test_zero_means_fp8_for_every_size(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", "0")
    assert _fp8_threshold_bytes() == 0


def test_explicit_value_is_parsed(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", "1024")
    assert _fp8_threshold_bytes() == 1024


def test_blank_value_defaults(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", "   ")
    assert _fp8_threshold_bytes() == 64 * 1024


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        (" on ", True),  # .strip() exists for padded values
        ("0", False),
        ("", False),
        ("off", False),
    ],
)
def test_env_flag_truthiness(monkeypatch, raw, expected):
    monkeypatch.setenv(FLAG, raw)
    assert _env_flag(FLAG) is expected


# --- group-wide agreement ---------------------------------------------------


def _fake_reduce(total):
    return lambda t, op=None, group=None: t.fill_(total)


def test_agree_flag_true_when_every_rank_sets_it(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(impl.dist, "all_reduce", _fake_reduce(2))
    assert _agree_flag(SimpleNamespace(size=2), None, FLAG) is True


def test_agree_flag_false_when_no_rank_sets_it(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(impl.dist, "all_reduce", _fake_reduce(0))
    assert _agree_flag(SimpleNamespace(size=2), None, FLAG) is False


def test_agree_flag_raises_on_a_partial_count(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(impl.dist, "all_reduce", _fake_reduce(1))  # 1 of 2 ranks
    with pytest.raises(RuntimeError, match="differs across TP ranks"):
        _agree_flag(SimpleNamespace(size=2), None, FLAG)


# --- the fp8 quantized-wire all_reduce --------------------------------------


class _FakeComm:
    """2-rank stand-in: all_gather writes the local chunk, then the peer's."""

    def __init__(self, peer_bytes=None, reduce_fill=None):
        self.peer = peer_bytes
        self.reduce_fill = reduce_fill

    def all_gather(self, dst, src):
        n = src.numel()
        dst[:n] = src
        dst[n : 2 * n] = self.peer

    def all_reduce(self, x, op):
        if self.reduce_fill is None:
            raise AssertionError("the bf16 path must not be taken")
        x.fill_(self.reduce_fill)


def test_fp8_all_reduce_sums_the_quantized_ranks(tp2):
    x = torch.tensor([1.0, 2.0, -3.0, 4.0], dtype=torch.bfloat16)
    peer = (
        torch.tensor([0.5, -1.0, 2.5, 8.0], dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    impl_ = PyNCCLDistributedImpl(_FakeComm(peer_bytes=peer), fp8_reduce=True, fp8_min_bytes=0)
    # all_reduce mutates x in place — build the expectation first
    expected = (
        x.to(torch.float8_e4m3fn).to(torch.float32)
        + peer.view(torch.float8_e4m3fn).to(torch.float32)
    )
    out = impl_.all_reduce(x)
    assert torch.equal(out, expected.to(torch.bfloat16))


def test_below_the_threshold_stays_on_the_bf16_path(tp2):
    x = torch.ones(4, dtype=torch.bfloat16)  # 8 bytes
    comm = _FakeComm(reduce_fill=9.0)
    impl_ = PyNCCLDistributedImpl(comm, fp8_reduce=True, fp8_min_bytes=1024)
    out = impl_.all_reduce(x)
    assert torch.equal(out, torch.full((4,), 9.0, dtype=torch.bfloat16))
