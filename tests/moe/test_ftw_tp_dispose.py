"""FTW TP>1 re-slice disposes the abandoned full banks.

_tp_slice_ftw_sources builds per-rank cudaMallocHost copies of each sliced bank
and abandons the full-bank originals. Without dispose, the originals' pinned
pages (cudaHostRegister on mmap-backed HostBanks held forever by _LIVE_BUFFERS)
stay resident + page-locked for the process lifetime — at TP=2 that is ~1.5x the
serving set pinned per rank. The dispose must only hit the SLICED names' banks;
alphas and other non-sliced sources keep theirs. CPU (FREETOKEN_SKIP_BANK_PIN),
real HostBanks, no engine.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.moe.expert_banks import _tp_slice_ftw_sources
from freetoken.moe.host_banks import HostBank, _LIVE_BUFFERS


def _restore_tp():
    import freetoken.distributed.info as info

    saved = getattr(info, "_TP_INFO", None)
    yield
    info._TP_INFO = saved


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info

    info._TP_INFO = None
    from freetoken.distributed import set_tp_info

    set_tp_info(rank, size)


def _config(I: int):
    # _tp_expert_geometry reads moe_intermediate_size (+ expert alignment);
    # I=2560 slices cleanly across TP=2 (I_loc=1280, 16-block aligned).
    return SimpleNamespace(moe_intermediate_size=I)


def _banks(config, num_experts=8, layers=2):
    """mmap-backed HostBanks shaped like the NVFP4 FTW layout (gate/up rows = 2I;
    packed/scale factor along the packed axis), with fill so slices carry real bytes."""
    H = 64
    I = config.moe_intermediate_size
    names = {
        "gate_up": (num_experts, 2 * I, H // 2),
        "gate_up_scale": (num_experts, 2 * I, H // 16),
        "gate_up_global": (num_experts, 2 * I),
        "down": (num_experts, H, I // 2),
        "down_scale": (num_experts, H, I // 16),
    }
    sources, host_banks = {}, {}
    for name, shape in names.items():
        banks = [HostBank(shape, torch.uint8) for _ in range(layers)]
        for b in banks:
            b.tensor.random_()  # fill so slices carry real bytes, not zeros
        host_banks[name] = banks
        sources[name] = [b.tensor for b in banks]
    return sources, host_banks


def test_slice_disposes_only_the_sliced_names(monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    config = _config(2560)
    sources, host_banks = _banks(config)
    live_before = len(_LIVE_BUFFERS)

    _set_tp(0, 2)
    for _ in _restore_tp():
        _tp_slice_ftw_sources(sources, config, host_banks)

    # sliced names' full banks are disposed: mmaps closed + gone from _LIVE_BUFFERS
    assert all(name not in host_banks for name in sources)
    assert len(_LIVE_BUFFERS) == live_before - 10  # 5 sliced names x 2 layers
    # the rank slices are new exact-width tensors, NOT views of the disposed banks
    g = sources["gate_up"][0]
    assert g.shape[1] == 2560  # I rows gate + I rows up
    assert g.is_contiguous() and g.data_ptr() != 0


def test_slice_at_tp1_is_a_noop(monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    config = _config(2560)
    sources, host_banks = _banks(config)
    before = {n: [b.tensor for b in bs] for n, bs in host_banks.items()}

    _set_tp(0, 1)  # rank < size: TP=1 is rank 0
    for _ in _restore_tp():
        _tp_slice_ftw_sources(sources, config, host_banks)

    assert host_banks  # untouched
    for name, views in before.items():
        assert all(v is s for v, s in zip(views, sources[name]))


def test_slice_without_host_banks_still_slices(monkeypatch):
    """Non-FTW callers (no host_banks) keep the old contract: slice in place."""
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    config = _config(2560)
    sources, _ = _banks(config)

    _set_tp(0, 2)
    for _ in _restore_tp():
        _tp_slice_ftw_sources(sources, config, None)

    assert sources["gate_up"][0].shape[1] == 2560


def test_dispose_twice_is_idempotent(monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    config = _config(2560)
    sources, host_banks = _banks(config)

    _set_tp(0, 2)
    for _ in _restore_tp():
        _tp_slice_ftw_sources(sources, config, host_banks)
    # re-dispose of an already-disposed bank must not raise
    hb = HostBank((64,), torch.uint8)
    hb.dispose()
    hb.dispose()
