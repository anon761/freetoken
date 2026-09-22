"""MTP expert normalization across exporter layouts (CPU-only).

The draft module holds a plain bf16 MoELayer, so the reader must normalize whatever the
checkpoint ships to stacked bf16 [E, 2I, H] / [E, H, I]:

* RadixArk NVFP4: already stacked bf16 under ``experts.{gate_up,down}_proj``.
* NVIDIA NVFP4: per-expert block-fp8 ``experts.<e>.{proj}_proj.weight`` + ``.weight_scale_inv``.

Both must produce the same model-side keys, and the quantized one must dequantize.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.weight import iter_mtp_experts

from .common import hf_config

BLOCK = 128
E, H, I = 4, 256, 128


@pytest.fixture(autouse=True)
def _set_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    yield


def _tp1():
    return SimpleNamespace(rank=0, size=1)


def _config(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    cfg = hf_config(
        num_layers=4, hidden=H, moe_intermediate_size=I, num_experts=E,
        shared_expert_intermediate_size=I, mtp_num_hidden_layers=1,
    )
    return parse_config(cfg)


def _block_ref(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent 128x128 block dequant (loop form, not the production reshape)."""
    out, inn = weight.shape
    ref = torch.empty(out, inn, dtype=torch.float32)
    for r in range(out // BLOCK):
        for c in range(inn // BLOCK):
            ref[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK] = (
                weight[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK].to(torch.float32)
                * scale[r, c].to(torch.float32)
            )
    return ref.to(torch.bfloat16)


def _write_per_expert(path):
    expected_gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
    expected_down = torch.empty(E, H, I, dtype=torch.bfloat16)
    tensors = {}
    gen = torch.Generator().manual_seed(0)
    for e in range(E):
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            w = torch.randn(shape, generator=gen).to(torch.float8_e4m3fn)
            s = (torch.rand(shape[0] // BLOCK, shape[1] // BLOCK, generator=gen) + 0.5).to(
                torch.bfloat16
            )
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            tensors[base + ".weight"] = w
            tensors[base + ".weight_scale_inv"] = s
            dq = _block_ref(w, s)
            if proj == "gate_proj":
                expected_gate_up[e, :I] = dq
            elif proj == "up_proj":
                expected_gate_up[e, I:] = dq
            else:
                expected_down[e] = dq
    save_file(tensors, str(path / "model-fp8-mtp.safetensors"))
    return expected_gate_up, expected_down


def test_nvidia_per_expert_block_fp8_is_dequantized_and_stacked(tmp_path, monkeypatch):
    config = _config(monkeypatch)
    expected_gate_up, expected_down = _write_per_expert(tmp_path)

    got = dict(iter_mtp_experts(str(tmp_path), config, _tp1()))
    assert set(got) == {"mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"}
    assert got["mtp.layers.0.mlp.experts.gate_up_proj"].dtype is torch.bfloat16
    assert torch.equal(got["mtp.layers.0.mlp.experts.gate_up_proj"], expected_gate_up)
    assert torch.equal(got["mtp.layers.0.mlp.experts.down_proj"], expected_down)


def test_radixark_stacked_bf16_passes_through(tmp_path, monkeypatch):
    config = _config(monkeypatch)
    gate_up = torch.randn(E, 2 * I, H, dtype=torch.bfloat16)
    down = torch.randn(E, H, I, dtype=torch.bfloat16)
    save_file(
        {
            "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
            "mtp.layers.0.mlp.experts.down_proj": down,
        },
        str(tmp_path / "model-bf16-mtp.safetensors"),
    )

    got = dict(iter_mtp_experts(str(tmp_path), config, _tp1()))
    assert torch.equal(got["mtp.layers.0.mlp.experts.gate_up_proj"], gate_up)
    assert torch.equal(got["mtp.layers.0.mlp.experts.down_proj"], down)
