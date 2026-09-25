# Copyright (c) 2026 FreeToken contributors
# nvfp4_linear._pick_split_k_bm16: the M <= 16 split-K choice per GPU class.
from __future__ import annotations

from freetoken.kernel.triton.nvfp4_linear import _pick_split_k_bm16

# (num_mn, num_tiles) of Qwen3.8-27B's per-rank MLP at BLOCK_N=128, BLOCK_KW=16
GATE_UP = (136, 40)
DOWN = (40, 68)


def test_smem_bound_part_fills_the_tail_wave():
    # RTX 3090: 82 SMs x 2 blocks; measured optimum split 4 for both shapes
    assert _pick_split_k_bm16(*GATE_UP, 164, 2) == 4
    assert _pick_split_k_bm16(*DOWN, 164, 2) == 4


def test_register_bound_part_keeps_the_h100_rule():
    # H100: 132 SMs x 4 blocks (wave 528): the tail-wave rule does not apply
    assert _pick_split_k_bm16(*GATE_UP, 528, 4) == _pick_split_k_bm16(*GATE_UP, 528)
    assert _pick_split_k_bm16(*DOWN, 528, 4) == _pick_split_k_bm16(*DOWN, 528)
