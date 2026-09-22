"""Qwen4Exp online image processor: smart_resize bounds + patchify geometry (CPU)."""

from __future__ import annotations

import io

import torch


def _png(height: int, width: int) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), color=(120, 30, 200))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _processor():
    from freetoken.models.qwen4_exp.processor import Qwen4ExpImageProcessor

    return Qwen4ExpImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        min_pixels=65536,
        max_pixels=16777216,
        image_token_id=248056,
    )


def test_smart_resize_is_factor_aligned_and_bounded():
    from freetoken.models.qwen4_exp.processor import smart_resize

    h, w = smart_resize(100, 100, factor=32, min_pixels=65536, max_pixels=16777216)
    assert h % 32 == 0 and w % 32 == 0
    assert h * w >= 65536


def test_preprocess_patchify_geometry():
    processor = _processor()
    pixel_values, grid, per_image = processor.preprocess([_png(512, 512)])
    # 512 is 32-aligned and above min_pixels: 32x32 patches, merge 2 -> 256 soft tokens
    assert grid.tolist() == [[1, 32, 32]]
    assert pixel_values.shape == (1024, 3 * 2 * 16 * 16)
    assert per_image == [256]
    assert pixel_values.dtype is torch.float32


def test_preprocess_multiple_images_concatenate():
    processor = _processor()
    pixel_values, grid, per_image = processor.preprocess([_png(512, 512), _png(512, 1024)])
    assert grid.tolist() == [[1, 32, 32], [1, 32, 64]]
    assert per_image == [256, 512]
    assert pixel_values.shape[0] == 1024 + 2048
