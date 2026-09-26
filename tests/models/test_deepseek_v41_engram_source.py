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


def _write_ftw(tmp_path, rows: int = 96):
    """An FTW converted with --include-engram: the tables under their ``model.`` names,
    with a tiny shard limit so the value table spans several shards (like the real
    ~92 GiB tables across 8 GiB shards)."""
    from freetoken.checkpoint.ftw import FTWWriter

    w = torch.randint(1, 255, (rows, _ROW_BYTES), dtype=torch.uint8)
    s = torch.randint(1, 255, (rows, _SCALE_BYTES), dtype=torch.uint8)
    out = tmp_path / "Model-FTW"
    wr = FTWWriter(str(out), shard_limit=8192)
    wr.add_tensor("model.layers.0.attn.norm.weight", torch.zeros(4096, dtype=torch.uint8))
    wr.add_tensor(f"model.{PREFIX}.embed.scale", s)
    wr.add_tensor(f"model.{PREFIX}.embed.weight", w)
    wr.finalize({})
    return str(out), w, s


def test_ftw_tables_span_shards_and_match_both_backends(tmp_path):
    path, w, s = _write_ftw(tmp_path)
    from freetoken.checkpoint.ftw import FTWReader

    assert len(FTWReader(path).file_pieces(f"model.{PREFIX}.embed.weight")) > 1
    idx = torch.tensor([0, 31, 32, 33, 95, 64, 0], dtype=torch.int64)
    for source in (EngramRowSource(path, 1, PREFIX), RamEngramRowSource(path, 1, PREFIX)):
        out = torch.zeros(idx.numel(), _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8)
        source.read_rows(idx, out)
        assert source.rows == w.shape[0]
        assert torch.equal(out[:, :_ROW_BYTES], w[idx])
        assert torch.equal(out[:, _ROW_BYTES:], s[idx])


def test_ftw_without_tables_needs_the_raw_checkpoint(tmp_path):
    from freetoken.checkpoint.ftw import FTWWriter

    out = tmp_path / "Other-FTW"
    wr = FTWWriter(str(out), shard_limit=8192)
    wr.add_tensor("model.layers.0.attn.norm.weight", torch.zeros(16, dtype=torch.uint8))
    wr.finalize({})
    try:
        EngramRowSource(str(out), 1, PREFIX)
    except FileNotFoundError as e:
        assert "--include-engram" in str(e)
    else:
        raise AssertionError("an FTW without tables and without a raw checkpoint must fail clearly")


def test_ftw_dense_load_skips_the_tables_before_reading(tmp_path):
    from freetoken.checkpoint.ftw import iter_ftw_weights

    path, _, _ = _write_ftw(tmp_path)
    names = [n for n, _ in iter_ftw_weights(path, skip=lambda n: ".engram.embed." in n)]
    assert names == ["model.layers.0.attn.norm.weight"]
