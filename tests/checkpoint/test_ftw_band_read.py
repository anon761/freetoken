"""FTW band-read correctness (CPU-only).

Band segments start at per-expert row offsets that are NOT block-aligned; under
O_DIRECT that is EINVAL, so `_read_aligned_span` reads an ALIGN-rounded window
through a page-aligned scratch and copies the wanted sub-range. These tests cover
the copy math, the multi-piece placement (`_pieces` dest_off), and the buffered
path.
"""

from __future__ import annotations

import mmap
import os

import pytest

from freetoken.checkpoint.ftw import _read_aligned_span, _read_band_segments

_DATA = bytes((i * 7 + 3) % 256 for i in range(20000))


def _odirect_fd(path: str) -> int | None:
    """An O_DIRECT fd on ``path``, or None when the filesystem rejects it."""
    if not hasattr(os, "O_DIRECT"):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return None
    buf = mmap.mmap(-1, 4096)
    try:
        os.preadv(fd, [memoryview(buf)[:4096]], 0)
    except OSError:
        os.close(fd)
        return None
    finally:
        buf.close()
    return fd


class _Bank:
    def __init__(self, nbytes: int):
        self._buf = bytearray(nbytes)

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)


class _Reader:
    """Minimal FTWReader surface for the band reader (buffered backend)."""

    def __init__(self, path: str, pieces):
        self._direct = 0
        self._fadvise = False
        self._path = path
        self._pieces_fn = pieces
        self._fd_handle = os.open(path, os.O_RDONLY)

    def _fd(self, _file: str) -> int:
        return self._fd_handle

    def _pieces(self, global_off: int, nbytes: int):
        return self._pieces_fn(global_off, nbytes)


def _single_piece(global_off: int, nbytes: int):
    yield "shard", global_off, 0, nbytes


def test_read_aligned_span_copies_an_unaligned_window(tmp_path):
    p = tmp_path / "shard.bin"
    p.write_bytes(_DATA)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        dest = memoryview(bytearray(1000))
        _read_aligned_span(fd, 1234, 777, dest, 10)
        assert bytes(dest[10 : 10 + 777]) == _DATA[1234 : 1234 + 777]
    finally:
        os.close(fd)


def test_read_band_segments_places_each_segment(tmp_path):
    p = tmp_path / "shard.bin"
    p.write_bytes(_DATA)
    reader = _Reader(str(p), _single_piece)
    bank = _Bank(len(_DATA) + 4096)
    segs = [(100, 50, 200), (1000, 30, 400)]
    _read_band_segments(reader, bank, {"global_off": 0, "nbytes": len(_DATA)}, segs, chunk=4096)
    mv = bank.memoryview()
    assert bytes(mv[200:250]) == _DATA[100:150]
    assert bytes(mv[400:430]) == _DATA[1000:1030]


def test_read_band_segments_honours_piece_dest_off(tmp_path):
    # A segment split across shards: the yielded dest_off must advance the write.
    p = tmp_path / "shard.bin"
    p.write_bytes(_DATA)
    reader = _Reader(str(p), _single_piece)
    bank = _Bank(len(_DATA) + 4096)

    def two_pieces(global_off: int, nbytes: int):
        half = nbytes // 2
        yield "a", global_off, 0, half
        yield "b", global_off + half, half, nbytes - half

    reader._pieces_fn = two_pieces
    segs = [(50, 40, 300)]
    _read_band_segments(reader, bank, {"global_off": 0, "nbytes": len(_DATA)}, segs, chunk=4096)
    mv = bank.memoryview()
    assert bytes(mv[300:340]) == _DATA[50:90]


def test_read_band_segments_odirect_unaligned(tmp_path):
    # The regression: an unaligned band segment on an O_DIRECT fd used to raise
    # EINVAL (preadv). It must go through the aligned-scratch path and land right.
    p = tmp_path / "shard.bin"
    p.write_bytes(_DATA)
    fd = _odirect_fd(str(p))
    if fd is None:
        pytest.skip("O_DIRECT unsupported on this filesystem")
    reader = _Reader(str(p), _single_piece)
    os.close(reader._fd_handle)
    reader._fd_handle = fd
    reader._direct = os.O_DIRECT
    bank = _Bank(len(_DATA) + 4096)
    try:
        _read_band_segments(reader, bank, {"global_off": 0, "nbytes": len(_DATA)},
                            [(123, 50, 77)], chunk=4096)
    finally:
        os.close(fd)
    assert bytes(bank.memoryview()[77:127]) == _DATA[123:173]
