"""Online image input plumbing: message extraction, placeholder expansion, and the
N-D tensor transport the processor outputs need (CPU)."""

from __future__ import annotations

import base64
import io

import torch


def _data_uri(height: int, width: int) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def test_expand_image_tokens():
    from freetoken.tokenizer.vision import expand_image_tokens

    ids = torch.tensor([1, 248056, 2, 248056, 3], dtype=torch.int32)
    out = expand_image_tokens(ids, [2, 3], 248056)
    assert out.tolist() == [1, 248056, 248056, 2, 248056, 248056, 248056, 3]
    assert out.dtype is torch.int32


def test_preprocess_messages_extracts_data_uri():
    from freetoken.tokenizer.vision import preprocess_messages

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": _data_uri(512, 512)}},
            ],
        }
    ]
    vision = preprocess_messages(_processor(), messages)
    assert vision is not None
    assert vision.image_token_id == 248056
    assert vision.tokens_per_image == [256]
    assert vision.image_grid_thw.tolist() == [[1, 32, 32]]
    assert vision.pixel_values.shape[0] == 1024


def test_preprocess_messages_text_only_is_none():
    from freetoken.tokenizer.vision import preprocess_messages

    assert preprocess_messages(object(), [{"role": "user", "content": "hi"}]) is None


def test_nd_tensor_serialization_roundtrip():
    from freetoken.message.utils import deserialize_type, serialize_type

    for shape in [(3, 4), (2, 3, 4)]:
        tensor = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
        back = deserialize_type({}, serialize_type(tensor))
        assert back.shape == shape and torch.equal(back, tensor)

    ids = torch.arange(5, dtype=torch.int32)
    back = deserialize_type({}, serialize_type(ids))
    assert back.dtype is torch.int32 and torch.equal(back, ids)
