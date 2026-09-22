"""Dense fp8 dequant through iter_weights, generically across dialects (CPU-only).

The reader must handle a dense projection stored as fp8 with EITHER a per-row
``.weight_scale`` (llm-compressor naive) OR a 128x128 ``.weight_scale_inv``
(block-fp8), dequantizing to the bf16 the model buffers expect -- decided from the
checkpoint, not a hardcoded suffix.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.checkpoint_index import dequantize_fp8_block
from freetoken.models.qwen4_exp.weight import iter_weights

from .common import hf_config

BLOCK = 128


@pytest.fixture(autouse=True)
def _tp1():
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    yield


def _jsonable(value):
    if isinstance(value, SimpleNamespace):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _dense_qkv(tmp_path, *, block_fp8: bool) -> tuple[dict[str, torch.Tensor], dict]:
    """Layer 1 (full attention) q/k/v as fp8 + scale; returns the raw tensors + expected bf16."""
    cfg = hf_config(num_layers=2, hidden=128, head_dim=64, num_q=4, num_kv=2)
    json.dump(_jsonable(vars(cfg)), open(tmp_path / "config.json", "w"))
    qo, kv, hd, H = 4, 2, 64, 128
    shape = {"q_proj": (2 * qo * hd, H), "k_proj": (kv * hd, H), "v_proj": (kv * hd, H)}
    raw: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(0)
    for proj, (out, inn) in shape.items():
        base = f"model.language_model.layers.1.self_attn.{proj}"
        w = torch.randn(out, inn, generator=gen).to(torch.float8_e4m3fn)
        if block_fp8:
            s = (torch.rand(out // BLOCK, inn // BLOCK, generator=gen) + 0.5).to(torch.bfloat16)
            raw[base + ".weight"] = w
            raw[base + ".weight_scale_inv"] = s
            # the primitive itself is verified against an independent loop in test_checkpoint_index
            expected[base] = dequantize_fp8_block(w, s, block=BLOCK)
        else:
            s = (torch.rand(out, 1, generator=gen) + 0.5).to(torch.bfloat16)
            raw[base + ".weight"] = w
            raw[base + ".weight_scale"] = s
            expected[base] = (w.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
    return raw, expected


def _run(tmp_path, raw) -> dict[str, torch.Tensor]:
    save_file(raw, str(tmp_path / "model-00001.safetensors"))
    return dict(iter_weights(str(tmp_path), torch.device("cpu"),
                             include_moe_experts=False, include_non_moe=True))


def test_dense_per_row_weight_scale(tmp_path):
    raw, expected = _dense_qkv(tmp_path, block_fp8=False)
    got = _run(tmp_path, raw)
    qkv = got["model.layers.1.self_attn.qkv_proj.weight"]
    assert qkv.dtype is torch.bfloat16
    ref = torch.cat([expected[k] for k in (
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.v_proj",
    )], dim=0)
    assert torch.equal(qkv, ref)


def test_dense_block_weight_scale_inv(tmp_path):
    raw, expected = _dense_qkv(tmp_path, block_fp8=True)
    got = _run(tmp_path, raw)
    qkv = got["model.layers.1.self_attn.qkv_proj.weight"]
    assert qkv.dtype is torch.bfloat16
    ref = torch.cat([expected[k] for k in (
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.v_proj",
    )], dim=0)
    assert torch.equal(qkv, ref)
