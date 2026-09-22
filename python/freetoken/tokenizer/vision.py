"""Online image input: pull image data URIs out of OpenAI messages, run the family's
image processor, and expand the single ``<|image_pad|>`` placeholder to the soft-token
count the vision tower will emit.

The processor runs in the tokenizer worker (where ``input_ids`` are produced); the raw
processor outputs travel to the scheduler, which owns the vision tower on the GPU and
turns them into ``mm_embeds`` (see ``scheduler._process_one_msg``).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, List

import torch


@dataclass
class VisionInputs:
    pixel_values: torch.Tensor  # [num_patches, C*T*P*P]
    image_grid_thw: torch.Tensor  # [num_images, 3]
    tokens_per_image: List[int]
    image_token_id: int


def load_online_vision(model_path: str):
    """The model's online image processor, or None for text-only / unsupported families."""
    try:
        from freetoken.models.qwen4_exp.processor import Qwen4ExpImageProcessor

        return Qwen4ExpImageProcessor.from_model_path(model_path)
    except Exception:  # noqa: BLE001 — never let a processor import break tokenization
        return None


def _iter_image_parts(messages: Any):
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                yield part


def _decode_image(part: dict) -> bytes | None:
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        try:
            return base64.b64decode(payload)
        except (ValueError, base64.binascii.Error):
            return None
    return None  # remote http(s) URLs are not fetched by the engine


def preprocess_messages(processor, messages: Any) -> VisionInputs | None:
    """Decode + preprocess every image in ``messages``; None when there are no usable images."""
    if processor is None:
        return None
    blobs = [b for part in _iter_image_parts(messages) if (b := _decode_image(part)) is not None]
    if not blobs:
        return None
    pixel_values, image_grid_thw, tokens_per_image = processor.preprocess(blobs)
    return VisionInputs(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        tokens_per_image=tokens_per_image,
        image_token_id=processor.image_token_id,
    )


def expand_image_tokens(
    input_ids: torch.Tensor, tokens_per_image: List[int], image_token_id: int
) -> torch.Tensor:
    """Replace each single image-token placeholder with its ``tokens_per_image`` copies
    (the chat template emits one ``<|image_pad|>`` per image; the tower emits N soft tokens)."""
    out: list[int] = []
    remaining = list(tokens_per_image)
    for token in input_ids.tolist():
        if token == image_token_id and remaining:
            out.extend([image_token_id] * remaining.pop(0))
        else:
            out.append(token)
    return torch.tensor(out, dtype=input_ids.dtype)


__all__ = ["VisionInputs", "expand_image_tokens", "load_online_vision", "preprocess_messages"]
