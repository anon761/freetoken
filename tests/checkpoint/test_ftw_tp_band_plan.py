"""TP-band planning for FTW expert banks at TP>1.

DeepSeek-FTW checkpoints slice cleanly (all banks carry an intermediate axis);
Qwen-family checkpoints (e.g. RadixArk RVN NVFP4) additionally carry 2-D
per-expert global-scale banks (``down_global`` [E, H]) that must be read
full-width on every rank, while ``gate_up_global`` [E, 2I] HAS an intermediate
axis and must band-slice like every other gate_up bank.
"""
from __future__ import annotations

import pytest

from freetoken.checkpoint.ftw import _band_axis, _tp_band_plan


def test_down_packed_slices_the_packed_columns():
    sliced, segs, mem_axis = _tp_band_plan("down_packed", (512, 2560, 320), 0, 1280, 2560, 1280)
    assert sliced == (512, 2560, 160)
    assert segs is None  # down-family: read full entry, slice columns in memory
    assert mem_axis == 2


def test_down_global_reads_full_width():
    sliced, segs, _ = _tp_band_plan("down_global", (512, 2560), 0, 1280, 2560, 1280)
    assert sliced == (512, 2560)  # [E, H]: no packed axis, nothing to slice
    assert segs == "full"         # not None: None would hit the 3-D column slice


def test_gate_up_global_band_scales_offsets_to_bytes():
    # gate_up_global is fp16 (2 bytes): the row stride is in ELEMENTS, so the byte
    # offsets/lengths must be doubled. Without the element-size factor the offsets
    # are halved and only the first half of the experts is read (the rest stay
    # zero) -> degenerate, looping output (Qwen3.8-Flash-Next on 2x3090).
    sliced, segs, _ = _tp_band_plan("gate_up_global", (512, 1280), 0, 640, 1280, 640, elsize=2)
    assert sliced == (512, 1280)  # 2 * I_loc
    assert segs[0] == (0, 1280, 0)        # expert 0 gate: 640 elems * 2 B
    assert segs[1] == (1280, 1280, 1280)  # expert 0 up
    assert segs[2] == (2560, 1280, 2560)  # expert 1 gate: (1*1280) elems * 2 B


def test_gate_up_packed_with_trailing_cols():
    # 1-byte bank: elements == bytes, so elsize=1 leaves the offsets unchanged.
    sliced, segs, _ = _tp_band_plan("gate_up_packed", (512, 1280, 1280), 0, 640, 1280, 640)
    assert sliced == (512, 1280, 1280)
    assert segs[0] == (0, 640 * 1280, 0)


def test_nvfp4_gate_up_band_takes_both_halves():
    # [E, 2I, H/2] with I=640, rank 1 of 2 -> rows [320, 640) of the gate half AND of the
    # up half. Slicing the fused axis as one block ([640, 1280)) handed rank 1 the whole up
    # half and rank 0 the whole gate half: Qwen3.8-Flash-Next-NVFP4 decoded garbage at TP=2.
    sliced, segs, mem_axis = _tp_band_plan("gate_up_packed", (2, 1280, 16), 320, 640, 640, 320)
    assert sliced == (2, 640, 16) and mem_axis is None
    assert segs[:2] == [(320 * 16, 320 * 16, 0), ((640 + 320) * 16, 320 * 16, 320 * 16)]


def test_unknown_bank_is_rejected():
    with pytest.raises(ValueError, match="unknown bank"):
        _tp_band_plan("mystery", (512,), 0, 1, 2, 1)


def test_band_axis_overrides_only_wna16():
    # WNA16 [E, packed_in, out]: gate_up's intermediate is the last axis, down's the packed one
    assert _band_axis("gate_up_qweight", (8, 320, 1280), "w4a16") == 2
    assert _band_axis("down_qweight", (8, 80, 2560), "w4a16") == 1
    assert _band_axis("down_global", (8, 2560), "w4a16") is None
    # NVFP4 keeps the default plan: an axis-1 override on the fused gate_up sliced it as one
    # block and fed each rank the wrong half (garbage output on Flash-Next-NVFP4 at TP=2)
    assert _band_axis("gate_up_packed", (8, 1280, 1280), "nvfp4") is None
    assert _band_axis("gate_up_scale", (8, 1280, 160), None) is None
