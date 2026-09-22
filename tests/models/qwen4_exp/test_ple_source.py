"""FTW PLE source resolution + verbatim detection (CPU-only).

A self-contained FTW carries the PLE n-gram table as ``model-plefp8-*`` side tables; an
older/partial FTW may not, so the loader falls back to the raw source recorded in the
FTW index instead of failing the serve.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from freetoken.checkpoint.convert import _ple_rides_verbatim
from freetoken.models.qwen4_exp.weight import ple_source_folder

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _raw_source(d: str) -> None:
    """Raw HF-like dir: the PLE table lives in a combined model-fp8-mtp-ple shard."""
    tensors = {
        f"{PREFIX}.shard_0.weight": torch.zeros(2, 4, dtype=torch.float8_e4m3fn),
        f"{PREFIX}.weight_scale": torch.tensor(0.125, dtype=torch.bfloat16),
    }
    save_file(tensors, os.path.join(d, "model-fp8-mtp-ple.safetensors"))
    json.dump(
        {"weight_map": {k: "model-fp8-mtp-ple.safetensors" for k in tensors}},
        open(os.path.join(d, "model.safetensors.index.json"), "w"),
    )


def _ftw_dir(d: str, source: str) -> None:
    json.dump(
        {"format": "freetoken_weight", "source_model_path": source, "tensors": [], "shards": []},
        open(os.path.join(d, "freetoken_weight.json"), "w"),
    )


def test_ftw_without_side_tables_falls_back_to_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _raw_source(str(raw))
    ftw = tmp_path / "ftw"
    ftw.mkdir()
    _ftw_dir(str(ftw), str(raw))
    assert ple_source_folder(str(ftw)) == str(raw)


def test_ftw_with_side_tables_uses_itself(tmp_path):
    ftw = tmp_path / "ftw"
    ftw.mkdir()
    _ftw_dir(str(ftw), "/nonexistent")
    save_file(
        {f"{PREFIX}.shard_0.weight": torch.zeros(2, 4, dtype=torch.float8_e4m3fn)},
        str(ftw / "model-plefp8-00000.safetensors"),
    )
    assert ple_source_folder(str(ftw)) == str(ftw)


def test_non_ftw_is_itself(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _raw_source(str(raw))
    assert ple_source_folder(str(raw)) == str(raw)


def test_verbatim_ignores_the_bf16_weight_scale():
    header = {
        f"{PREFIX}.shard_0.weight": {"dtype": "F8_E4M3"},
        f"{PREFIX}.weight_scale": {"dtype": "BF16"},
    }
    assert _ple_rides_verbatim(header, list(header), "fp8")


def test_bf16_table_verbatim_only_for_bf16_target():
    keys = [f"{PREFIX}.shard_0.weight"]
    header = {keys[0]: {"dtype": "BF16"}}
    assert _ple_rides_verbatim(header, keys, "bf16")
    assert _ple_rides_verbatim(header, keys, "auto")
    assert not _ple_rides_verbatim(header, keys, "fp8")


def test_fp8_table_verbatim_only_for_fp8_target():
    keys = [f"{PREFIX}.shard_0.weight"]
    header = {keys[0]: {"dtype": "F8_E4M3"}}
    assert _ple_rides_verbatim(header, keys, "fp8")
    assert _ple_rides_verbatim(header, keys, "auto")
    assert not _ple_rides_verbatim(header, keys, "bf16")


def test_source_from_safetensors_bf16_geometry(tmp_path):
    """An unquantized (bf16) table: two-byte rows, no weight_scale, scale 1.0."""
    from freetoken.models.qwen4_exp.ple_disk import source_from_safetensors

    save_file(
        {
            f"{PREFIX}.shard_0.weight": torch.zeros(2, 4, dtype=torch.bfloat16),
            f"{PREFIX}.shard_1.weight": torch.zeros(2, 4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model-plebf16-00000.safetensors"),
    )
    src = source_from_safetensors(str(tmp_path))
    assert src.src_dtype == "BF16"
    assert src.elem_bytes == 2
    assert src.row_bytes == 8 and src.row_stride == 8
    assert src.scale == 1.0
    assert src.total_rows == 4


def test_load_ple_table_bf16(tmp_path):
    """A bf16 table is served byte-for-byte in a bf16 bank (scale 1.0), not quantized."""
    from .common import toy_hf_config
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.weight import load_ple_table

    args = parse_config(toy_hf_config()).qwen4_args
    rows = 32
    values = torch.randn(
        args.split_ngram_parts, rows, args.ngram_head_dim, dtype=torch.bfloat16
    )
    save_file(
        {f"{PREFIX}.shard_{i}.weight": values[i] for i in range(args.split_ngram_parts)},
        str(tmp_path / "model-plebf16-00000.safetensors"),
    )
    table = load_ple_table(str(tmp_path), args, pin=False)
    assert table.bank.tensor.dtype is torch.bfloat16
    assert float(table.weight_scale) == 1.0
    assert torch.equal(table.bank.tensor, values.reshape(-1, args.ngram_head_dim))
