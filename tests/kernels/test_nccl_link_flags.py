"""NCCL link-flag resolution for the pynccl JIT build (CPU-only).

The CUDA toolkit ships no libnccl and the nvidia-nccl-cuNN wheels install only the
versioned soname, so a hard `-lnccl` link fails on a clean box. The resolver must
find a real copy and adapt the linker flag instead of assuming one.
"""

from __future__ import annotations

from freetoken.kernel.utils import _nccl_link_flags_from, nccl_link_flags


def test_unversioned_soname_uses_plain_link_name(tmp_path):
    (tmp_path / "libnccl.so").write_bytes(b"")
    assert _nccl_link_flags_from([str(tmp_path)]) == [
        f"-L{tmp_path}",
        "-lnccl",
        f"-Wl,-rpath,{tmp_path}",
    ]


def test_versioned_soname_links_the_exact_name(tmp_path):
    (tmp_path / "libnccl.so.2").write_bytes(b"")
    flags = _nccl_link_flags_from([str(tmp_path)])
    assert flags == [f"-L{tmp_path}", "-l:libnccl.so.2", f"-Wl,-rpath,{tmp_path}"]


def test_skips_directories_without_nccl(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (real / "libnccl.so.2").write_bytes(b"")
    flags = _nccl_link_flags_from([str(empty), str(real)])
    assert f"-L{real}" in flags


def test_prefers_the_first_hit(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "libnccl.so.2").write_bytes(b"")
    (second / "libnccl.so").write_bytes(b"")
    assert _nccl_link_flags_from([str(first), str(second)])[0] == f"-L{first}"


def test_missing_everywhere_falls_back_to_plain_name(tmp_path):
    assert _nccl_link_flags_from([str(tmp_path / "nope")]) == ["-lnccl"]


def test_public_resolver_returns_flags_or_fallback():
    flags = nccl_link_flags()
    assert flags
    assert flags == ["-lnccl"] or any(f.startswith("-L") for f in flags)
