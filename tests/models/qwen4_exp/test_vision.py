"""Qwen4Exp vision tower: geometry helpers + a tiny end-to-end forward (CPU)."""

from __future__ import annotations

import torch

from freetoken.models.qwen4_exp.config import Qwen4VisionConfig
from freetoken.models.qwen4_exp.vision import (
    Qwen4ExpVisionModel,
    _cu_seqlens,
    _interp_indices_weights,
    _position_ids,
)


def _vc(**overrides) -> Qwen4VisionConfig:
    base = dict(
        hidden_size=32,
        intermediate_size=64,
        depth=2,
        num_heads=4,
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=48,
        num_position_embeddings=36,
    )
    base.update(overrides)
    return Qwen4VisionConfig(**base)


def test_geometry_helpers():
    grid = torch.tensor([[1, 8, 8], [2, 4, 6]])
    cu = _cu_seqlens(grid)
    # frame 0: one segment of 64; frame 1: two segments of 24
    assert cu.tolist() == [0, 64, 88, 112]

    pos = _position_ids(grid, 2)
    assert pos.shape == (112, 2)

    indices, weights = _interp_indices_weights(grid, side=6, merge=2)
    assert indices.shape == (112, 4)
    assert weights.shape == (112, 4)
    # weights over the 2x2 bilinear taps sum to 1 per patch (interior, border-clamped)
    assert torch.allclose(weights.sum(dim=1), torch.ones(112), atol=1e-5)


def test_vision_forward_shape():
    vc = _vc()
    model = Qwen4ExpVisionModel(vc)
    for tensor in model.state_dict().values():
        tensor.normal_()

    t, h, w = 1, 8, 8
    grid = torch.tensor([[t, h, w]])
    num_patches = t * h * w
    pixels = torch.randn(num_patches, vc.in_channels * vc.temporal_patch_size * vc.patch_size**2)
    out = model.forward(pixels, grid)
    assert out.shape == (num_patches // vc.spatial_merge_size**2, vc.out_hidden_size)
