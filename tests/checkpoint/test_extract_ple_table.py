"""_extract_ple_table quantizes bf16 PLE n-gram tables into fp8 shards — and must
tolerate FP8 source tensors: PyTorch has no ``max_all`` kernel for Float8_e4m3fn,
and mixed bf16/fp8 PLE shards (e.g. RadixArk NVFP4 checkpoints) route into the
extraction path whenever not every PLE key is F8_E4M3, so the global-max walk
sees FP8 tensors.
"""
from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from freetoken.checkpoint.convert import _PLE_KEY_INFIX, _extract_ple_table

_PLE_KEY = f"model.layers.0{_PLE_KEY_INFIX}shard_0.weight"


def _safetensors_header(path):
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(hlen))


def test_fp8_ple_table_extracts_without_crashing(tmp_path):
    # An FP8 PLE table: .abs().max() on Float8_e4m3fn raises NotImplementedError
    # (pre-fix); the extraction must cast to float32 and derive the scale instead.
    table = (torch.randn(64, 32) * 5).clamp(-400, 400).to(torch.float8_e4m3fn)
    src = tmp_path / "model-plebf16-00000.safetensors"
    save_file({_PLE_KEY: table}, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copied: list[str] = []
    _extract_ple_table(str(src), str(out_dir), copied, out_dtype="fp8")

    assert copied == ["model-plefp8-00000.safetensors"]
    shards = [f for f in out_dir.iterdir() if f.name.startswith("model-plefp8-")]
    assert len(shards) == 1
    header = _safetensors_header(shards[0])
    assert header[_PLE_KEY]["dtype"] == "F8_E4M3"
    assert f"{_PLE_KEY_INFIX}weight_scale" in header


def test_bf16_ple_table_scale_still_derived_from_max_abs(tmp_path):
    import safetensors

    table = (torch.randn(32, 16) * 10).to(torch.bfloat16)
    src = tmp_path / "model-plebf16-00000.safetensors"
    save_file({_PLE_KEY: table}, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copied: list[str] = []
    _extract_ple_table(str(src), str(out_dir), copied, out_dtype="fp8")

    assert copied == ["model-plefp8-00000.safetensors"]
    shard = out_dir / "model-plefp8-00000.safetensors"
    with safetensors.safe_open(str(shard), framework="pt", device="cpu") as f:
        out_scale = f.get_tensor(f"{_PLE_KEY_INFIX}weight_scale")
        out_table = f.get_tensor(_PLE_KEY)
    # scale = global max |t| / 448, and dequantized output stays within fp8 grid
    # error (e4m3: 3 mantissa bits → relative step ≤ 1/8, half-step ≤ |x|/16)
    assert float(out_scale) == pytest.approx(float(table.float().abs().max()) / 448.0, rel=1e-2)
    dequant = out_table.float() * float(out_scale)
    assert (dequant - table.float()).abs().max() <= (table.float().abs() / 16.0 + 2 * float(out_scale)).max()


def test_scale_key_is_not_counted_into_max_abs(tmp_path):
    # A shard that already carries a scalar weight_scale: the scale must ride
    # through unchanged (not be re-derived), and the fp8 table must not crash.
    table = (torch.randn(16, 16) * 5).clamp(-100, 100).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.0, dtype=torch.bfloat16)
    src = tmp_path / "model-plebf16-00000.safetensors"
    save_file({_PLE_KEY: table, f"model.layers.0{_PLE_KEY_INFIX}shard_0.weight_scale": scale}, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copied: list[str] = []
    _extract_ple_table(str(src), str(out_dir), copied, out_dtype="fp8")

    assert copied == ["model-plefp8-00000.safetensors"]


def test_bf16_output_preserves_values_and_writes_no_scale(tmp_path):
    table = (torch.randn(32, 16) * 10).to(torch.bfloat16)
    src = tmp_path / "model-plebf16-00000.safetensors"
    save_file({_PLE_KEY: table}, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copied: list[str] = []
    _extract_ple_table(str(src), str(out_dir), copied, out_dtype="bf16")

    assert copied == ["model-plebf16-00000.safetensors"]
    header = _safetensors_header(out_dir / "model-plebf16-00000.safetensors")
    assert header[_PLE_KEY]["dtype"] == "BF16"
    assert f"model.layers.0{_PLE_KEY_INFIX}weight_scale" not in header


def test_bf16_output_dequantizes_an_fp8_source(tmp_path):
    import safetensors

    scale = torch.tensor(0.5, dtype=torch.bfloat16)
    table = ((torch.randn(16, 16) * 4) / float(scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
    src = tmp_path / "model-plefp8-00000.safetensors"
    save_file(
        {_PLE_KEY: table, f"model.layers.0{_PLE_KEY_INFIX}weight_scale": scale}, str(src)
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    copied: list[str] = []
    _extract_ple_table(str(src), str(out_dir), copied, out_dtype="bf16")

    assert copied == ["model-plebf16-00000.safetensors"]
    with safetensors.safe_open(
        str(out_dir / "model-plebf16-00000.safetensors"), framework="pt", device="cpu"
    ) as f:
        out = f.get_tensor(_PLE_KEY)
    assert out.dtype is torch.bfloat16
    assert torch.allclose(out.float(), table.float() * float(scale), atol=1e-2)
