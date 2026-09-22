"""``--engram-backend``: the RAM (mmap) row source must return exactly the same
bytes as the disk source. CPU-only against a synthetic safetensors engram table.
"""

from __future__ import annotations

import json

import torch

from freetoken.models.deepseek_v41.engram import (
    _ROW_BYTES,
    _SCALE_BYTES,
    EngramRowSource,
    RamEngramRowSource,
    make_engram_row_source,
)

PREFIX = "layers.1.engram"


def _write_ckpt(tmp_path, rows: int = 64):
    import safetensors.torch

    w = torch.randint(1, 255, (rows, _ROW_BYTES), dtype=torch.uint8)
    s = torch.randint(1, 255, (rows, _SCALE_BYTES), dtype=torch.uint8)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(
        {f"{PREFIX}.embed.weight": w, f"{PREFIX}.embed.scale": s}, str(shard)
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            f"{PREFIX}.embed.weight": shard.name,
            f"{PREFIX}.embed.scale": shard.name,
        }})
    )
    return w, s


def test_ram_source_matches_disk(tmp_path):
    w, s = _write_ckpt(tmp_path)
    disk = EngramRowSource(str(tmp_path), 1, PREFIX)
    ram = RamEngramRowSource(str(tmp_path), 1, PREFIX)
    idx = torch.tensor([0, 5, 63, 12, 5], dtype=torch.int64)
    out_d = torch.zeros(idx.numel(), _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8)
    out_r = torch.zeros_like(out_d)
    disk.read_rows(idx, out_d)
    ram.read_rows(idx, out_r)
    assert torch.equal(out_d, out_r)
    assert torch.equal(out_d[:, :_ROW_BYTES], w[idx])
    assert torch.equal(out_d[:, _ROW_BYTES:], s[idx])


def test_ram_source_writes_into_a_view(tmp_path):
    """The decode path writes into a slice of the pinned buffer."""
    _write_ckpt(tmp_path)
    ram = RamEngramRowSource(str(tmp_path), 1, PREFIX)
    buf = torch.zeros(3, 264, dtype=torch.uint8)
    view = buf[1:3]
    ram.read_rows(torch.tensor([0, 1], dtype=torch.int64), view)
    assert int(buf[1, 0]) != 0 and int(buf[2, 0]) != 0
    assert torch.equal(buf[0], torch.zeros(_ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8))


def test_factory_selects_backend(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    monkeypatch.delenv("FREETOKEN_ENGRAM_BACKEND", raising=False)
    assert type(make_engram_row_source(str(tmp_path), 1, PREFIX)) is EngramRowSource
    monkeypatch.setenv("FREETOKEN_ENGRAM_BACKEND", "ram")
    assert isinstance(make_engram_row_source(str(tmp_path), 1, PREFIX), RamEngramRowSource)
    monkeypatch.setenv("FREETOKEN_ENGRAM_BACKEND", "disk")
    assert type(make_engram_row_source(str(tmp_path), 1, PREFIX)) is EngramRowSource
