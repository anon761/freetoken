"""FTW batched/windowed read: one worker pool for a whole window of small tensors.

The dense shard is ~1000 tiny tensors; the old path read one at a time (queue depth 1,
~0.8 GiB/s on the NVMe). ``read_entries`` feeds every chunk of a window to one pool so the
drive stays saturated. Byte-for-byte equality is what matters here -- the speedup is
measured on the GPU box, but a wrong offset/short read must fail on CPU.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from freetoken.checkpoint.ftw import ALIGN, FTWReader, _entry_windows, iter_ftw_weights


def _align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _make_ftw(tmp_path, specs):
    """specs: [(name, kind, dtype_str, shape, np.ndarray)]. Tensors are packed at
    ALIGN-aligned offsets into one shard, exactly like the real writer."""
    blob = bytearray()
    tensors = []
    for name, kind, dtype_str, shape, arr in specs:
        data = arr.tobytes()
        off = _align_up(len(blob))
        blob += b"\x00" * (off - len(blob))
        blob[off : off + len(data)] = data
        tensors.append(
            {
                "name": name,
                "kind": kind,
                "global_off": off,
                "nbytes": len(data),
                "dtype": dtype_str,
                "shape": list(shape),
            }
        )
    total = _align_up(len(blob))
    blob += b"\x00" * (total - len(blob))
    shard = "data-00000.bin"
    (tmp_path / shard).write_bytes(bytes(blob))
    index = {
        "format": "freetoken_weight",
        "shards": [{"file": shard, "global_off": 0, "nbytes": total}],
        "tensors": tensors,
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))
    return str(tmp_path), tensors


def _specs():
    rng = np.random.default_rng(0)
    out = []
    for i in range(7):
        a = rng.standard_normal((3, 17)).astype(np.float32)  # 204 B, one ALIGN chunk
        out.append((f"model.layers.{i}.w", "weight", "float32", (3, 17), a))
    big = rng.standard_normal((8192,)).astype(np.float32)  # 32 KiB, spans chunks
    out.append(("lm_head.weight", "weight", "float32", (8192,), big))
    eb = rng.standard_normal((256,)).astype(np.float32)
    out.append(("model.layers.0.experts.gate_up_proj", "experts_bank", "float32", (256,), eb))
    return out


def test_read_entries_and_iter_match_the_arrays(tmp_path):
    specs = _specs()
    path, _tensors = _make_ftw(tmp_path, specs)
    want = {name: arr for name, _k, _d, _s, arr in specs}

    # read_entries: batched, one pool, byte-for-byte
    reader = FTWReader(path)
    try:
        items = reader.read_entries(reader.entries("weight"), workers=8)
        assert [n for n, *_ in items] == [n for n, k, *_ in specs if k == "weight"]
        for name, tensor, _buf, nbytes in items:
            assert nbytes == want[name].nbytes
            assert torch.equal(tensor, torch.from_numpy(want[name])), name
    finally:
        reader.close()

    # iter_ftw_weights: windowed + double-buffered; only the weight kind is yielded
    got = {name: t for name, t in iter_ftw_weights(path, workers=8, window_bytes=ALIGN * 4)}
    assert set(got) == {n for n, k, *_ in specs if k == "weight"}
    for name, arr in want.items():
        if name in got:
            assert torch.equal(got[name], torch.from_numpy(arr)), name


def test_entry_windows_group_consecutively(tmp_path):
    specs = _specs()
    _path, tensors = _make_ftw(tmp_path, specs)
    wins = list(_entry_windows(tensors, 1024))
    assert sum(len(w) for w in wins) == len(tensors)
    assert all(sum(e["nbytes"] for e in w) <= 1024 or len(w) == 1 for w in wins)
    flat = [e["name"] for w in wins for e in w]
    assert flat == [t["name"] for t in tensors]
