"""Qwen3.8-Flash-Next image processor (Qwen2-VL-style), for online image input.

Decodes raw image bytes, resizes per ``smart_resize`` (patch*merge-aligned, bounded by
``min_pixels``/``max_pixels``), rescales+normalizes with the checkpoint's mean/std, and
patchifies into ``pixel_values [num_patches, C*T*P*P]`` + ``image_grid_thw [num_images, 3]``.
Mirrors ``transformers.image_processing_qwen2_vl`` (the checkpoint's
``image_processor_type``); the checkpoint ships no processor class for qwen4_exp.
"""

from __future__ import annotations

import io
import json
import math
import os
from typing import Any, List, Tuple

import torch


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """Rescale so both sides are ``factor``-aligned and the pixel count stays in
    ``[min_pixels, max_pixels]``, keeping the aspect ratio (Qwen2-VL ``smart_resize``)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen4ExpImageProcessor:
    def __init__(
        self,
        *,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        image_mean: List[float],
        image_std: List[float],
        min_pixels: int,
        max_pixels: int,
        image_token_id: int | None,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token_id = image_token_id

    @classmethod
    def from_model_path(cls, model_path: str) -> "Qwen4ExpImageProcessor | None":
        cfg = _read_json(os.path.join(model_path, "config.json"))
        if not cfg or cfg.get("model_type") != "qwen4_exp":
            return None
        vc = cfg.get("vision_config") or {}
        if not vc:
            return None
        pre = _read_json(os.path.join(model_path, "preprocessor_config.json")) or {}
        size = pre.get("size") or {}
        return cls(
            patch_size=int(vc.get("patch_size", pre.get("patch_size", 16))),
            temporal_patch_size=int(vc.get("temporal_patch_size", pre.get("temporal_patch_size", 2))),
            merge_size=int(vc.get("spatial_merge_size", pre.get("merge_size", 2))),
            image_mean=list(pre.get("image_mean", [0.5, 0.5, 0.5])),
            image_std=list(pre.get("image_std", [0.5, 0.5, 0.5])),
            min_pixels=int(size.get("shortest_edge", 56 * 56)),
            max_pixels=int(size.get("longest_edge", 28 * 28 * 1280)),
            image_token_id=cfg.get("image_token_id"),
        )

    def preprocess(self, images: List[bytes]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Raw image bytes -> ``(pixel_values, image_grid_thw, tokens_per_image)``."""
        from PIL import Image
        import numpy as np

        factor = self.patch_size * self.merge_size
        merge = self.merge_size
        patch = self.patch_size
        temporal = self.temporal_patch_size

        patches: list[torch.Tensor] = []
        grids: list[list[int]] = []
        per_image: list[int] = []
        for blob in images:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            arr = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # [3, H, W]
            height, width = arr.shape[-2:]
            resized_h, resized_w = smart_resize(
                height, width, factor=factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )
            arr = torch.nn.functional.interpolate(
                arr.unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            arr = (arr / 255.0 - self.image_mean) / self.image_std
            grid_h, grid_w = resized_h // patch, resized_w // patch
            # [gh/m, gw/m, m, m, C, P, P] -> flat [gh*gw, C*T*P*P] (channel-major, temporal next)
            p = arr.reshape(3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
            p = p.permute(1, 4, 2, 5, 0, 3, 6).reshape(grid_h * grid_w, 3, patch, patch)
            p = p.unsqueeze(2).expand(-1, -1, temporal, -1, -1).reshape(
                grid_h * grid_w, 3 * temporal * patch * patch
            )
            patches.append(p)
            grids.append([1, grid_h, grid_w])
            per_image.append((grid_h * grid_w) // (merge * merge))

        pixel_values = torch.cat(patches, dim=0) if patches else torch.empty(0)
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        return pixel_values, image_grid_thw, per_image


__all__ = ["Qwen4ExpImageProcessor", "smart_resize"]
