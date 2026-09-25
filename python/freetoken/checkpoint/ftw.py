"""FreeToken Weight (FTW) checkpoint: one O_DIRECT-friendly on-disk format for a whole model.

The format is a single *logical contiguous byte region* of all tensors, sliced *physically*
into shard files of at most ``shard_limit`` bytes (default 8 GiB, for HF/filesystem
friendliness). It exists because reading the original safetensors back fast is awkward:
tensors are packed with no alignment, so an individual tensor can't be O_DIRECT-read at an
arbitrary offset; and the earlier per-bank cache prototype worked around that by giving every expert bank
its own file -- which doesn't cover dense weights and turns a model's long tail of tiny
tensors (norms, biases, router) into hundreds of tiny I/Os.

FTW fixes both:

* **Aligned.** Every tensor starts at a 4096-aligned region offset and is padded to 4096;
  shards are cut at 4096-aligned boundaries. So any tensor (or any shard-local slice of one)
  is read with offset, length (rounded up to 4096), and destination all block-aligned --
  exactly what O_DIRECT requires. A tensor larger than a shard simply spans shards; because
  both its start and the shard boundary are aligned, each piece stays aligned.
* **Unified.** It holds dense weights as ``kind="weight"`` (exactly what a model's
  ``iter_weights`` yields -- post fusion/TP-shard, fed straight to ``load_state_dict``) and
  the offload expert state as ``kind="experts_bank"`` (post backend-repack -- the per-expert
  weight banks plus, distinguished only by their reserved names, the alpha scale vectors;
  the FTW content). The converter runs the per-model loaders once; this reader is
  model-agnostic.

Layout on disk::

    <dir>/freetoken_weight.json        # index: tensors[] + shards[] + meta
    <dir>/freetoken-00000.ftw         # the byte region, sliced <= shard_limit
    <dir>/freetoken-00001.ftw
    <dir>/config.json, tokenizer*, ...# copied so the dir is a self-contained checkpoint
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
FORMAT_VERSION = 1
# The DSpark draft has its own expert count (E=128 vs the target's 384), so it cannot
# share the target's offload cache. Its banks carry a dedicated kind/namespace in the
# FTW: the same bank base names, told apart at load by kind (FTWReader.entries filters).
DSPARK_BANK_KIND = "dspark_experts_bank"
DSPARK_BANK_NUM_LAYERS = "dspark_bank_num_layers"
ALIGN = 4096  # O_DIRECT block alignment (== page size on this platform)
DEFAULT_SHARD_LIMIT = 8 << 30  # 8 GiB; must be a multiple of ALIGN
_SHARD_FMT = "freetoken-{:05d}.ftw"
_DEFAULT_CHUNK = 8 << 20
# Concurrent down-family full-entry reads during the TP band load. Each holds one
# transient whole-entry buffer, so this bounds the extra host RAM to
# ~_DOWN_READ_CONCURRENCY * entry_bytes; going wider oversubscribes the cores.
_DOWN_READ_CONCURRENCY = 8
_ALPHA_NAMES = ("gate_up_alpha", "down_alpha")
# Per-layer expert-bank entry name (converter streaming path, see checkpoint/convert.py):
# each layer of a bank is its own FTW tensor instead of one flat [num_layers*E, ...] region.
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")


def layer_bank_entry_name(bank_name: str, layer_id: int) -> str:
    """Name of one per-layer ``experts_bank`` FTW entry; :func:`load_ftw_banks` groups
    entries matching ``_LAYER_ENTRY_RE`` back into a per-layer bank list by base name."""
    return f"{bank_name}#L{layer_id:05d}"


def _pread_into(fd: int, mv: memoryview, offset: int) -> None:
    """POSIX positional read into ``mv`` at ``offset``, looping over any short preadv.

    preadv may return short (a signal, or the EOF-adjacent tail); the loop resumes
    at the running offset, which stays O_DIRECT-legal: the writer pads every tensor
    to ALIGN and cuts shards at ALIGN boundaries, so direct-IO short reads land on
    block boundaries. EOF before the buffer is filled raises ``OSError`` — a
    truncated shard must not silently load garbage weights."""
    done = 0
    total = len(mv)
    while done < total:
        n = os.preadv(fd, [mv[done:]], offset + done)
        if n == 0:
            raise OSError(
                f"unexpected EOF reading FTW: got {done}/{total} bytes at offset {offset}"
            )
        done += n


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _dtype_str(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_of(s: str) -> torch.dtype:
    return getattr(torch, s)


def _elsize(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt).element_size()


def _np_dtype(dt: torch.dtype):
    """NumPy view dtype for a bank dtype (bfloat16 / fp8 have no numpy dtype: viewed as the
    same-width unsigned int, byte-identical -- the banks are only sliced, never computed on)."""
    import numpy as np

    return {
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.bfloat16: np.uint16,
        torch.float8_e4m3fn: np.uint8,
        torch.float8_e5m2: np.uint8,
        torch.float32: np.float32,
    }[dt]


def is_ftw_checkpoint(path: str) -> bool:
    """True if ``path`` is a directory holding a FreeToken Weight (FTW) index."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


def ftw_quant_format(path: str) -> str | None:
    """The ``quant_format`` an FTW checkpoint's expert banks were packed for; None when ``path`` is not an FTW checkpoint or holds no banks."""
    if not is_ftw_checkpoint(path):
        return None
    with open(os.path.join(path, INDEX_NAME)) as f:
        return json.load(f).get("quant_format")


# ============================== writer ==============================
class FTWWriter:
    """Stream tensors into the FTW, rolling shard files at ``shard_limit``.

    Tensors are written in call order into one logical byte stream; each is padded to
    ``ALIGN`` so the next starts aligned. A tensor that doesn't fit the current shard's
    remaining room is split across shards (the split point is the shard boundary, which is
    aligned). Call :meth:`add_tensor` for each tensor, then :meth:`finalize`.
    """

    def __init__(self, out_dir: str, *, shard_limit: int = DEFAULT_SHARD_LIMIT):
        assert shard_limit % ALIGN == 0, "shard_limit must be a multiple of ALIGN"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_limit = shard_limit
        self._tensors: list[dict] = []
        self._shards: list[dict] = []
        self._global = 0  # running FTW offset (incl. padding)
        self._f = None  # current shard file handle
        self._shard_idx = -1
        self._shard_start = 0  # FTW offset where the current shard began
        self._cur = 0  # bytes written to the current shard

    def _roll(self) -> None:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
        self._shard_idx += 1
        self._shard_start = self._global
        self._cur = 0
        self._f = open(os.path.join(self.out_dir, _SHARD_FMT.format(self._shard_idx)), "wb")

    def _write_raw(self, data: memoryview) -> None:
        """Write ``data`` into the FTW byte stream, splitting across shards at the limit."""
        if self._f is None:
            self._roll()
        off = 0
        n = len(data)
        while off < n:
            if self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self._global += take

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> None:
        t = tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        # A small tensor (<= shard) never splits: roll early so it lands whole in one shard.
        if self._f is None or (nbytes <= self.shard_limit
                               and self._cur + nbytes > self.shard_limit):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"
        self._write_raw(memoryview(raw.numpy()))
        self._tensors.append({"name": name, "kind": kind, "dtype": _dtype_str(t.dtype),
                              "shape": list(t.shape), "global_off": global_off, "nbytes": nbytes})
        # pad to ALIGN so the next tensor starts aligned
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))

    def finalize(self, meta: dict) -> dict:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
            self._f = None
        index = {"format": FORMAT_TAG, "version": FORMAT_VERSION, "align": ALIGN,
                 "shard_limit": self.shard_limit, "total_bytes": self._global,
                 "tensors": self._tensors, "shards": self._shards, **meta}
        tmp = os.path.join(self.out_dir, INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, os.path.join(self.out_dir, INDEX_NAME))
        return index


def _fstype_of(path: str) -> str:
    """The filesystem type of ``path``'s mount point (Linux: /proc/self/mountinfo —
    the statvfs f_basetype field is BSD/Solaris-only and missing on glibc/Linux)."""
    real = os.path.realpath(path)
    best, best_type = "", ""
    with open("/proc/self/mountinfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or "-" not in parts:
                continue
            sep = parts.index("-")
            mount, fstype = parts[4], parts[sep + 1]
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                best, best_type = mount, fstype
    return best_type


# ============================== reader ==============================
class FTWReader:
    """Random-access reader over an FTW checkpoint.

    Maps a tensor's logical byte range to one-or-more shard-file ranges (split at shard
    boundaries) and reads each piece with chunked multi-threaded O_DIRECT directly into the
    destination buffer. Offsets/lengths are all 4096-aligned (lengths rounded up into the
    rounded-up destination), so O_DIRECT is always legal -- including the tail of a tensor
    (the rounding reads into the region's padding, which is discarded by the tensor view)."""

    def __init__(self, path: str):
        with open(os.path.join(path, INDEX_NAME)) as f:
            self.index = json.load(f)
        assert self.index.get("format") == FORMAT_TAG, f"not a {FORMAT_TAG}: {path}"
        self.dir = path
        self.shards = sorted(self.index["shards"], key=lambda s: s["global_off"])
        self.tensors = {t["name"]: t for t in self.index["tensors"]}
        self._fds: dict[str, int] = {}
        self._maps: dict[str, tuple[mmap.mmap, memoryview]] = {}
        # O_DIRECT (DMA straight from disk, bypassing the page cache) is the fast path but a
        # perf choice, not a correctness one. Some filesystems reject it at open with EINVAL
        # (tmpfs, many overlay/network mounts) and the flag is Linux-only; when it's absent
        # we fall back to mmap (below), NOT to chunked buffered preadv -- a whole-shard
        # mapping + kernel readahead copies far faster than per-chunk page-cache reads.
        # 0 here means "O_DIRECT unavailable -> use the mmap path".
        self._direct = getattr(os, "O_DIRECT", 0)
        self._fadvise = False  # ZFS: O_DIRECT silently ignored — buffered reads
        # with POSIX_FADV_DONTNEED per chunk (the cache is evicted as we go, so
        # the shard bytes never double-resident next to the pinned banks)
        self._probed = False
        self._lock = threading.Lock()  # load_ftw_banks calls read_into concurrently

    def meta(self, key: str, default=None):
        return self.index.get(key, default)

    def entries(self, *kinds: str) -> list[dict]:
        keep = set(kinds)
        return [t for t in self.index["tensors"] if not keep or t["kind"] in keep]

    def _ensure_mode(self) -> None:
        """Resolve the read backend once: keep O_DIRECT if the filesystem accepts it, else
        drop to the mmap fallback. Thread-safe -- ``_probed`` is published only after
        ``_direct`` is final, so a concurrent reader never races onto a stale direct path."""
        if self._probed:
            return
        with self._lock:
            if self._probed:
                return
            if self._direct and self.shards:
                try:
                    os.close(os.open(os.path.join(self.dir, self.shards[0]["file"]),
                                     os.O_RDONLY | self._direct))
                except OSError:
                    self._direct = 0
                    logger.warning("O_DIRECT unsupported on %s; using mmap fallback for "
                                   "FTW load", self.dir)
            if self._direct:
                if _fstype_of(self.dir) == "zfs":
                    # ZFS <= 2.2 accepts the O_DIRECT open but silently ignores the
                    # flag: the reads populate the ARC/page cache ON TOP of the
                    # pinned banks (2x the bank set = OOM at TP2). Buffered reads
                    # + fadvise DONTNEED per chunk keep the cache at zero.
                    self._fadvise = True
                    logger.info_rank0("FTW load on ZFS: O_DIRECT is ignored there — "
                                      "using buffered reads with fadvise DONTNEED")
            self._probed = True

    def _fd(self, file: str) -> int:
        fd = self._fds.get(file)
        if fd is None:
            with self._lock:  # first-open only; chunk reads reuse the cached fd lock-free
                fd = self._fds.get(file)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY | self._direct)
                    self._fds[file] = fd
        return fd

    def _map(self, file: str) -> memoryview:
        entry = self._maps.get(file)
        if entry is None:
            with self._lock:
                entry = self._maps.get(file)
                if entry is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY)
                    try:
                        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                    finally:
                        os.close(fd)  # the mapping keeps its own reference to the file
                    try:
                        m.madvise(mmap.MADV_SEQUENTIAL)  # kernel readahead for streaming
                    except (AttributeError, OSError):
                        pass
                    entry = (m, memoryview(m))
                    self._maps[file] = entry
        return entry[1]

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for m, mv in self._maps.values():
            mv.release()
            m.close()
        self._maps.clear()

    def _pieces(self, global_off: int, nbytes: int):
        """Yield (file, file_off, dest_off, length) covering [global_off, +nbytes),
        split at shard boundaries. All file_off/dest_off are ALIGN-aligned."""
        dest_off = 0
        remaining = nbytes
        pos = global_off
        for sh in self.shards:
            s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
            if pos >= s1 or remaining <= 0:
                continue
            if pos < s0:  # regions are contiguous; a gap means a corrupt index
                raise ValueError("FTW gap / misordered shards")
            take = min(remaining, s1 - pos)
            yield sh["file"], pos - s0, dest_off, take
            pos += take
            dest_off += take
            remaining -= take
        if remaining:
            raise ValueError("tensor range exceeds FTW shards")

    def _jobs_for(self, dest: memoryview, entry: dict, chunk: int) -> list:
        """Chunk jobs for one tensor into ``dest``: ``(file, file_off, dest, dest_off, len)``
        covering ``[global_off, +nbytes)`` rounded up to ALIGN (padding is in-region)."""
        jobs = []
        for file, file_off, dest_off, length in self._pieces(entry["global_off"], entry["nbytes"]):
            rlen = _align_up(length)  # round the tail up; padding is in-region, harmless
            for c in range(0, rlen, chunk):
                jobs.append((file, file_off + c, dest, dest_off + c, min(chunk, rlen - c)))
        return jobs

    def _dispatch_jobs(self, jobs: list, workers: int, chunk: int) -> None:
        """Run chunk jobs through the reader's backend. The pool is what gives the disk
        its queue depth: many chunk jobs in flight == high throughput (one job == ~0.8 GiB/s
        on the NVMe, 16 == ~6 GiB/s)."""
        if not jobs:
            return
        # Open/map each distinct shard once, single-threaded, so the pool only reuses handles.
        touch = self._fd if (self._direct or self._fadvise) else self._map
        for file in {j[0] for j in jobs}:
            touch(file)

        if self._fadvise:
            def rd(job):
                file, fo, dest, do, ln = job
                fd = self._fd(file)
                done = 0
                while done < ln:
                    want = min(chunk, ln - done)
                    n = os.preadv(fd, [dest[do + done : do + done + want]], fo + done)
                    if n <= 0:
                        raise OSError(f"shard {file}: short read at {fo + done}")
                    try:
                        os.posix_fadvise(fd, fo + done, n, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
                    done += n
        elif self._direct:
            def rd(job):
                file, fo, dest, do, ln = job
                try:
                    _pread_into(self._fd(file), dest[do:do + ln], fo)
                except OSError as e:
                    raise OSError(f"shard {file}: {e}") from e
        else:
            def rd(job):
                file, fo, dest, do, ln = job
                mv = self._map(file)
                if fo + ln > len(mv):
                    raise OSError(
                        f"unexpected EOF reading FTW: shard {file} has "
                        f"{len(mv)} bytes, need {ln} at offset {fo}"
                    )
                dest[do:do + ln] = mv[fo:fo + ln]

        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                rd(j)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, jobs))

    def read_into(self, dest: memoryview, entry: dict, *, workers: int = 8,
                  chunk: int = _DEFAULT_CHUNK) -> None:
        """Read one tensor's bytes into ``dest`` (length >= entry nbytes rounded to ALIGN)."""
        self._ensure_mode()
        self._dispatch_jobs(self._jobs_for(dest, entry, chunk), workers, chunk)

    def read_entries(self, entries: list, *, workers: int = 16,
                     chunk: int = _DEFAULT_CHUNK) -> list:
        """Read a contiguous run of entries into ONE transient buffer over the whole byte
        range, with ``workers`` CONTIGUOUS segments read in parallel.

        Per-tensor reads left the disk at queue depth 1 (~0.8 GiB/s), and handing individual
        8 MiB chunks to a pool interleaved the streams so the filesystem's readahead never
        engaged (~1.8 GiB/s). Splitting the range into ``workers`` long sequential streams is
        what saturates the NVMe (~5 GiB/s here). ``entries`` must be contiguous (sorted by
        ``global_off`` with no other kind interleaved) -- the caller's windowing guarantees it.
        Returns ``[(name, tensor_view, buffer, nbytes)]`` sharing one buffer; the caller keeps
        the buffer until every view is consumed."""
        self._ensure_mode()
        start = entries[0]["global_off"]
        end = entries[-1]["global_off"] + entries[-1]["nbytes"]
        span = _align_up(end - start)
        buf = _transient_buffer(span)
        mv = memoryview(buf)
        jobs = []
        for file, file_off, dest_off, length in self._pieces(start, span):
            rlen = _align_up(length)
            for c in range(0, rlen, chunk):
                jobs.append((file, file_off + c, mv, dest_off + c, min(chunk, rlen - c)))
        if jobs:
            # Contiguous segments: worker i reads a long sequential slice (its own readahead
            # window), not a round-robin of 8 MiB chunks.
            nseg = max(1, min(workers, len(jobs)))
            per = (len(jobs) + nseg - 1) // nseg
            segs = [jobs[i : i + per] for i in range(0, len(jobs), per)]
            if len(segs) <= 1:
                self._dispatch_jobs(segs[0], 1, chunk)
            else:
                with ThreadPoolExecutor(len(segs)) as ex:
                    list(ex.map(lambda s: self._dispatch_jobs(s, 1, chunk), segs))
        out = []
        for e in entries:
            dt = _dtype_of(e["dtype"])
            elsize = _elsize(dt)
            base = (e["global_off"] - start) // elsize
            n = e["nbytes"] // elsize
            t = torch.frombuffer(buf, dtype=dt, count=span // elsize)[base : base + n]
            # a 0-d entry (a per-tensor scale) comes back 0-d, not [1]
            out.append((e["name"], t.view(*e["shape"]) if e["shape"] else t.view(()), buf, e["nbytes"]))
        return out


def _transient_buffer(nbytes: int) -> mmap.mmap:
    return mmap.mmap(-1, _align_up(nbytes))


def _entry_windows(entries: list, window_bytes: int):
    """Group consecutive entries into windows of at most ``window_bytes`` (a single entry
    larger than the window is its own window). Tensors are packed contiguously, so a window
    is one contiguous global range."""
    win: list = []
    size = 0
    for e in entries:
        nb = e["nbytes"]
        if win and size + nb > window_bytes:
            yield win
            win, size = [], 0
        win.append(e)
        size += nb
    if win:
        yield win


def iter_ftw_weights(path: str, *, kinds=("weight",), workers: int | None = None,
                     chunk: int = _DEFAULT_CHUNK, prefetch: int = 2,
                     window_bytes: int | None = None):
    """Yield ``(name, host_tensor)`` for the requested kinds.

    The dense shard is ~1000 small tensors; reading them one at a time leaves the NVMe at
    queue depth 1 (~0.8 GiB/s). Instead, consecutive entries are grouped into contiguous
    windows and each window is read into ONE buffer by ``workers`` long sequential segments
    (per-worker readahead), which is what saturates the drive (~5 GiB/s here). A background
    thread double-buffers windows so the disk stays busy while the consumer copies the
    current window to the GPU. Peak host mem ~ (prefetch+2) windows. ``workers``/
    ``window_bytes`` default from FREETOKEN_FTW_LOAD_WORKERS / FREETOKEN_FTW_LOAD_WINDOW_MB."""
    import queue
    import threading

    from freetoken.env import ENV
    from freetoken.utils.progress import byte_bar

    if workers is None:
        workers = int(ENV.FTW_LOAD_WORKERS.value)
    if window_bytes is None:
        window_bytes = int(ENV.FTW_LOAD_WINDOW_MB.value) << 20

    reader = FTWReader(path)
    entries = reader.entries(*kinds)
    if not entries:
        reader.close()
        return
    windows = list(_entry_windows(entries, window_bytes))
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []
    cancel = threading.Event()

    def _put(items) -> bool:
        # A plain q.put would deadlock teardown: if the consumer stops with the queue
        # full (early break out of the generator, or an exception mid-load), close()
        # runs the finally below, which joins this thread while it waits for queue
        # space forever. Poll the cancel flag instead of blocking indefinitely.
        while not cancel.is_set():
            try:
                q.put(items, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        try:
            for win in windows:
                items = reader.read_entries(win, workers=workers, chunk=chunk)
                if not _put(items):
                    return
        except BaseException as ex:  # surface to consumer
            err.append(ex)
        finally:
            _put(_DONE)

    th = threading.Thread(target=_producer, name="FTW-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(e["nbytes"] for e in entries), "Loading weights (FTW)", monitor=True)
    try:
        while True:
            items = q.get()
            if items is _DONE:
                break
            for name, tensor, buf, nbytes in items:
                yield name, tensor
                bar.update(nbytes)
                del tensor, buf  # buffer reclaimable once the consumer drops the tensor
    finally:
        bar.close()
        cancel.set()
        th.join()
        reader.close()
    if err:
        raise err[0]


def _band_axis(base: str, shape: tuple[int, ...], expert_quant: str | None) -> int | None:
    """In-memory slice axis overriding ``_tp_band_plan``'s default, or None to keep it.

    Only AutoRound WNA16 banks (``[E, packed_in, out]``) need one: axis 2 for gate_up
    (out == 2I), axis 1 for down (packed == I/8). NVFP4 banks (``[E, out, in]``) keep the
    default -- gate_up is contiguous per expert and read as segments taking ``[lo, hi)``
    from BOTH halves, down slices its last axis in memory."""
    if expert_quant != "w4a16":
        return None
    if base.startswith("gate_up"):
        return 2
    if base.startswith("down") and len(shape) == 3:
        return 1
    return None


def _tp_band_plan(base: str, shape: tuple[int, ...], lo: int, hi: int, I_full: int,
                  I_loc: int, elsize: int = 1, *, axis: int | None = None):
    """TP-band plan for one per-layer FTW bank entry.

    Returns ``(sliced_shape, row_segments, mem_axis)``. ``row_segments`` is None when
    the band must be read fully and sliced in memory (the NVFP4 down-family, and every
    WNA16 bank whose intermediate axis is not the leading per-expert axis);
    ``mem_axis`` is the axis to slice in that case. Otherwise it is a list of
    ``(src_off, src_len, dst_off)`` byte segments (the contiguous NVFP4 gate_up band).

    ``elsize`` is the bank dtype's element size (multi-byte banks: int32 WNA16 qweight,
    fp16 scales): row strides are in elements and must be scaled to bytes.
    """
    if axis is not None:
        # WNA16 / output-last layout: [E, packed, out]. The intermediate axis is the last
        # for gate_up (out == 2I) and the packed first for down (packed == I/8); neither is
        # contiguous per expert across all of the other axis, so read the entry fully and
        # slice in memory.
        sliced = list(shape)
        sliced[axis] = shape[axis] * I_loc // I_full
        return tuple(sliced), None, axis
    if base.startswith("gate_up"):
        E, two_i, *rest = shape
        rest_b = (int(np.prod(rest, dtype=np.int64)) if rest else 1) * elsize
        sliced = (E, 2 * I_loc, *rest)
        segs = []
        for e in range(E):
            for off, ln in ((lo, hi - lo), (two_i // 2 + lo, hi - lo)):
                segs.append(((e * two_i + off) * rest_b, ln * rest_b,
                             (e * 2 * I_loc + (0 if off < two_i // 2 else I_loc)) * rest_b))
        return sliced, segs, None
    if base.startswith("down"):
        if len(shape) == 2:
            # [E, H] per-expert global scale (e.g. down_global): no packed-column
            # axis to band-slice, so every rank reads the entry full-width.
            return tuple(shape), "full", None
        E, h_dim, cols = shape
        sliced = (E, h_dim, cols * I_loc // I_full)
        return sliced, None, 2
    raise ValueError(f"TP band read: unknown bank {base!r}")


_BAND_SCRATCH = threading.local()


def _band_scratch(nbytes: int) -> memoryview:
    """Per-thread page-aligned scratch for O_DIRECT band reads (reused, grown on demand)."""
    buf = getattr(_BAND_SCRATCH, "buf", None)
    if buf is None or len(buf) < nbytes:
        if buf is not None:
            buf.close()
        buf = mmap.mmap(-1, _align_up(nbytes))
        _BAND_SCRATCH.buf = buf
    return memoryview(buf)


def _read_aligned_span(fd: int, file_off: int, length: int, dest: memoryview, dest_off: int) -> None:
    """O_DIRECT-read an arbitrary (possibly unaligned) ``[file_off, +length)`` span into
    ``dest[dest_off:+length]``.

    Band segments start at row offsets that are not block-aligned (e.g. a per-expert
    gate_up scale row), so a direct preadv is illegal (EINVAL). Read the ALIGN-rounded
    window into a page-aligned scratch, then copy the wanted sub-range."""
    start = file_off - (file_off % ALIGN)
    delta = file_off - start
    span = _align_up(delta + length)
    size = os.fstat(fd).st_size
    if start + span > size:  # last tensor: clamp to the (ALIGN-padded) file end
        span = ((size - start) // ALIGN) * ALIGN
        if span < delta + length:
            raise OSError(f"band read past EOF: {length} bytes at {file_off} (file {size})")
    scratch = _band_scratch(span)
    _pread_into(fd, scratch[:span], start)
    dest[dest_off : dest_off + length] = scratch[delta : delta + length]


def _read_band_segments(reader, bank, entry, segs, chunk: int) -> None:
    """Strided per-expert band read: each segment is one pread chain from the
    shards into its destination offset (the sliced bank is contiguous in band
    order; the segments cross shard boundaries via reader._pieces).

    O_DIRECT (ext4/…) needs block-aligned offsets/lengths/buffers, but band rows are
    not aligned — those pieces go through :func:`_read_aligned_span`. Buffered/mmap
    backends (ZFS) preadv straight into the bank."""
    mv = memoryview(bank.memoryview())
    fadvise = getattr(reader, "_fadvise", False)
    direct = getattr(reader, "_direct", 0)
    for src_rel, ln, dst_off in segs:
        for file, file_off, piece_off, take in reader._pieces(entry["global_off"] + src_rel, ln):
            fd = reader._fd(file)
            dst = dst_off + piece_off
            if direct:
                _read_aligned_span(fd, file_off, take, mv, dst)
            else:
                done = 0
                while done < take:
                    want = min(chunk, take - done)
                    n = os.preadv(fd, [mv[dst + done : dst + done + want]], file_off + done)
                    if n <= 0:
                        raise OSError(f"band read short at {file_off + done}: got {n}")
                    done += n
            if fadvise:
                try:
                    os.posix_fadvise(fd, file_off, take, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass


def load_ftw_banks(
    path: str, *, num_layers: int, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
    layer_residency: list[str] | None = None, model_config=None,
    kind: str = "experts_bank", num_layers_meta_key: str = "expert_bank_num_layers",
):
    """Reconstruct the offload :class:`ExpertBanks` from the FTW's ``experts_bank``
    entries, on the per-layer host bank contract (one ``[num_experts, ...]``
    HostBank per layer per bank; see ``moe.offload_cache.set_bank_sources``).

    ``kind`` selects the bank namespace: the target's banks are ``experts_bank``, the
    DSpark draft's are ``DSPARK_BANK_KIND`` (same bank base names, told apart by kind).
    ``num_layers_meta_key`` is the index metadata cross-check for that namespace
    (``DSPARK_BANK_NUM_LAYERS`` for the draft).

    ``layer_residency`` (default: all pinned) settles each layer's banks per its ``HostResidency`` label as reads complete: PINNED -> cudaHostRegister, LOCKED -> mlock (CPU-executor resident, no pin quota spent).
    The applied labels are echoed back on ``ExpertBanks.layer_residency``.

    Two on-disk row layouts, distinguished per bank name (a file never mixes them for
    the same name -- checked below):

    * **Flat region** (pre-existing files, and non-streamable formats): one entry per
      bank, ONE contiguous ``[num_layers * num_experts, ...]`` region. ``num_layers``
      isn't part of that region's shape, so the caller passes it
      (``ModelConfig.num_moe_layers`` -- FTW checkpoints carry the model's config.json);
      the ``expert_bank_num_layers`` index meta the converter records is used as a
      cross-check when present. A layer's byte range within the region generally is
      NOT 4096-aligned (only the whole region's start is guaranteed aligned) -- read it
      via its ALIGNED enclosing window ``[align_down(off), align_up(off+len))`` into a
      page-aligned scratch HostBank, and view the real per-layer tensor as a
      head-offset slice.
    * **Per-layer** (streamable-format conversion, see :mod:`freetoken.checkpoint.convert`):
      one entry per ``(bank, layer)``, name ``f"{bank_name}#L{layer_id:05d}"``. Each was
      written by its own ``add_tensor`` call, so its start is already ALIGN-aligned --
      no windowing/head-pad needed, read straight into a HostBank shaped like the entry.

    Alphas (``gate_up_alpha``/``down_alpha``) stay flat ``[num_layers*num_experts]``
    vectors, unaffected by the row split (fixed GPU residency; see
    ``cache_budget.expert_bytes_per_slot``).
    """
    from freetoken.moe.host_banks import (
        HostBank, HostResidency, PinPipeline, alloc_banks, born_pinned_default,
    )
    from freetoken.utils.progress import byte_bar

    residency = layer_residency or [HostResidency.PINNED.value] * num_layers
    assert len(residency) == num_layers, (len(residency), num_layers)

    # TP>1: read ONLY this rank's intermediate band of every bank straight from
    # the shards (the FTW stores TP-agnostic full-width banks; reading them full
    # on every rank peaks at 2x the bank set and OOMs mid-load). The banks are
    # born at their SLICED shapes and the strided per-expert segments are read
    # with per-segment preads. The post-load _tp_slice_ftw_sources pass is then
    # unnecessary (the caller checks banks.tp_sliced).
    tp_loc = None
    if model_config is not None:
        from freetoken.distributed import get_tp_info

        tp = get_tp_info()
        if tp.size > 1:
            if getattr(model_config, "expert_quant", None) == "w4a16":
                # WNA16 group scales live on K and cannot split mid-group; the rank band
                # is group-aligned (may differ by one group across ranks) and gate_up's
                # output slice uses the SAME extent so it matches down's input.
                from freetoken.models.wna16_banks import wna16_tp_geometry

                I_full = int(model_config.moe_intermediate_size)
                lo, hi = wna16_tp_geometry(I_full, tp.size, tp.rank)
                tp_loc = (I_full, hi - lo, lo, hi)
            else:
                from freetoken.models.nvfp4_banks import _tp_expert_geometry

                I_full, I_loc, lo, hi = _tp_expert_geometry(model_config)
                tp_loc = (I_full, I_loc, lo, hi)
            logger.info_rank0(f"TP band reads engaged: rank band [{lo}, {hi}) of I={I_full}")

    # PINNED layers are born-pinned (cudaHostAlloc) where that wins (see born_pinned_default); LOCKED/PAGEABLE layers stay lazy mmaps
    born = born_pinned_default()

    def _backing(layer_id: int) -> str:
        if born and residency[layer_id] == HostResidency.PINNED.value:
            return "cuda"
        return "mmap"

    reader = FTWReader(path)
    bank_entries = reader.entries(kind)
    if not bank_entries:
        reader.close()
        return None
    tp_sliced = False

    alpha_entries = [e for e in bank_entries if e["name"] in _ALPHA_NAMES]
    row_entries = [e for e in bank_entries if e["name"] not in _ALPHA_NAMES]

    meta_layers = reader.meta(num_layers_meta_key)
    if meta_layers is not None and meta_layers != num_layers:
        reader.close()
        raise RuntimeError(
            f"{path!r} was converted with {meta_layers} {kind} layers but the "
            f"model config says num_moe_layers={num_layers}; the checkpoint does not "
            "match its config"
        )

    # Alphas: unchanged, one flat HostBank per entry.
    alpha_specs = {e["name"]: (tuple(e["shape"]), _dtype_of(e["dtype"])) for e in alpha_entries}
    alpha_hb = alloc_banks(alpha_specs)

    # Split row entries into the two layouts by name.
    flat_entries: list[dict] = []
    per_layer_groups: dict[str, dict[int, dict]] = {}
    for e in row_entries:
        m = _LAYER_ENTRY_RE.match(e["name"])
        if m is None:
            flat_entries.append(e)
            continue
        per_layer_groups.setdefault(m.group("base"), {})[int(m.group("layer"))] = e

    mixed = {e["name"] for e in flat_entries} & per_layer_groups.keys()
    assert not mixed, f"FTW bank(s) mix flat and per-layer row layouts: {sorted(mixed)}"

    # Row banks: one padded-window HostBank per (name, layer_id) for the flat layout, plus
    # how to carve the real [num_experts, *row_shape] tensor out of its head; ``None`` marks
    # a per-layer entry (direct view, no carving needed).
    row_hb: dict[str, list] = {}
    row_view_args: dict[str, list] = {}
    row_jobs = []  # (name, HostBank, window_off, window_len, layer_bytes) -- flat layout
    layer_jobs = []  # (name, HostBank, entry) -- per-layer layout, direct aligned read

    for e in flat_entries:
        name = e["name"]
        total, *row_shape = e["shape"]
        assert total % num_layers == 0, (name, total, num_layers)
        num_experts = total // num_layers
        dtype = _dtype_of(e["dtype"])
        row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
        layer_bytes = num_experts * row_bytes
        assert layer_bytes * num_layers == e["nbytes"], (name, layer_bytes, num_layers, e["nbytes"])
        row_hb[name] = []
        row_view_args[name] = []
        for layer_id in range(num_layers):
            off = e["global_off"] + layer_id * layer_bytes
            win_off = (off // ALIGN) * ALIGN
            win_end = _align_up(off + layer_bytes)
            head_pad = off - win_off
            bank = HostBank((win_end - win_off,), torch.uint8, backing=_backing(layer_id))
            row_hb[name].append(bank)
            row_view_args[name].append((head_pad, layer_bytes, num_experts, tuple(row_shape), dtype))
            row_jobs.append((name, bank, win_off, win_end - win_off, layer_bytes, layer_id))

    for base, by_layer in per_layer_groups.items():
        assert sorted(by_layer) == list(range(num_layers)), (
            f"FTW bank {base!r} has per-layer entries for layers {sorted(by_layer)}, "
            f"expected exactly range({num_layers})"
        )
        row_hb[base] = []
        row_view_args[base] = []
        for layer_id in range(num_layers):
            e = by_layer[layer_id]
            assert e["global_off"] % ALIGN == 0, (base, layer_id, e["global_off"])  # writer invariant
            if tp_loc is None:
                bank = HostBank(tuple(e["shape"]), _dtype_of(e["dtype"]), backing=_backing(layer_id))
                row_hb[base].append(bank)
                row_view_args[base].append(None)
                layer_jobs.append((base, bank, e, layer_id))
                continue
            tp_sliced = True
            # TP band read: allocate the SLICED shape, read the rank's strided
            # per-expert segments straight from the shard into it
            I_full, I_loc, lo, hi = tp_loc
            shape = tuple(e["shape"])
            axis = _band_axis(base, shape, getattr(model_config, "expert_quant", None))
            sliced, segs, mem_axis = _tp_band_plan(base, shape, lo, hi, I_full, I_loc, _elsize(_dtype_of(e["dtype"])), axis=axis)
            bank = HostBank(sliced, _dtype_of(e["dtype"]), backing=_backing(layer_id))
            row_hb[base].append(bank)
            row_view_args[base].append(None)
            layer_jobs.append((base, bank, e, layer_id, segs, tp_loc, mem_axis))

    total_bytes = sum(e["nbytes"] for e in bank_entries)
    bar = byte_bar(total_bytes, f"Loading {kind} (FTW)", monitor=True)

    # Jobs are per (bank, layer) -- many small reads, so a wider pool; each bank pins
    # as its read completes, overlapping cudaHostRegister with the remaining reads.
    n_jobs = len(alpha_entries) + len(row_jobs) + len(layer_jobs)
    try:
        with PinPipeline() as pins:

            # The OUTER pool below gives the queue depth (up to 16 banks read concurrently);
            # each job reads its bank with a SINGLE serial pread chain. A nested per-call
            # worker pool (outer x inner) oversubscribed the cores and starved the disk --
            # only the down-family already did it right (workers=1 under down_sem).
            def _read_alpha(e):
                bank = alpha_hb[e["name"]]
                reader.read_into(bank.memoryview(), e, workers=1, chunk=chunk)
                pins.submit(bank)
                bar.update(e["nbytes"])

            def _read_row(job):
                _name, bank, win_off, win_len, layer_bytes, layer_id = job
                reader.read_into(bank.memoryview(), {"global_off": win_off, "nbytes": win_len},
                                 workers=1, chunk=chunk)
                pins.submit(bank, residency[layer_id])
                bar.update(layer_bytes)

            # The down-family full-entry read is band-strided per H-row, so it reads the
            # whole entry transiently and slices in memory. It is followed by a 3-D
            # NumPy copy plus the bank scatter, and a nested per-call worker pool here
            # would oversubscribe the cores (outer pool x inner pool threads) -- a single
            # serial pread chain per job, with several jobs running concurrently, keeps
            # the device saturated without the thrash.
            down_sem = threading.Semaphore(_DOWN_READ_CONCURRENCY)

            def _read_layer(job):
                if len(job) == 7:
                    base, bank, entry, layer_id, segs, tp_band, mem_axis = job
                    if segs == "full":
                        # _global banks: no intermediate axis, full-width on every rank
                        reader.read_into(bank.memoryview(),
                                         {"global_off": entry["global_off"], "nbytes": entry["nbytes"]},
                                         workers=1, chunk=chunk)
                    elif segs is None:
                        with down_sem:
                            # The intermediate axis is not the leading per-expert axis (the
                            # NVFP4 down-family, and every WNA16 bank): read the full entry
                            # transiently and slice ``mem_axis`` in memory. The transient is
                            # a page-aligned mmap (O_DIRECT rejects a bytearray base).
                            tmp = _transient_buffer(entry["nbytes"])
                            arr = None
                            try:
                                reader.read_into(memoryview(tmp),
                                                 {"global_off": entry["global_off"], "nbytes": entry["nbytes"]},
                                                 workers=1, chunk=chunk)
                                shape = tuple(entry["shape"])
                                dt = _dtype_of(entry["dtype"])
                                arr = np.frombuffer(tmp, dtype=_np_dtype(dt), count=entry["nbytes"] // _elsize(dt)).reshape(shape)
                                I_full, I_loc, lo, hi = tp_band
                                if mem_axis == 2 and base.startswith("gate_up"):
                                    # WNA16 fused gate_up [E, pk, 2I]: the rank's slice is
                                    # [lo, hi) within BOTH the gate and up halves, which sit
                                    # adjacent on the last axis.
                                    half = shape[2] // 2
                                    a0, a1 = half * lo // I_full, half * hi // I_full
                                    band = np.concatenate(
                                        [arr[:, :, a0:a1], arr[:, :, half + a0 : half + a1]], axis=2
                                    ).copy()
                                else:
                                    idx = [slice(None)] * arr.ndim
                                    idx[mem_axis] = slice(shape[mem_axis] * lo // I_full, shape[mem_axis] * hi // I_full)
                                    band = arr[tuple(idx)].copy()
                                data = band.ravel().tobytes()
                                mv = memoryview(bank._buf)
                                mv[: len(data)] = data
                                mv.release()
                            finally:
                                # np.frombuffer(tmp) exports a buffer on the mmap; closing
                                # it while the array lives raises "cannot close exported
                                # pointers exist", so drop the array (and its view base) first.
                                if arr is not None:
                                    del arr
                                tmp.close()
                    else:
                        _read_band_segments(reader, bank, entry, segs, chunk)
                    pins.submit(bank, residency[layer_id])
                    bar.update(entry["nbytes"])
                    return
                _name, bank, entry, layer_id = job
                reader.read_into(bank.memoryview(), entry, workers=1, chunk=chunk)
                pins.submit(bank, residency[layer_id])
                bar.update(entry["nbytes"])

            with ThreadPoolExecutor(min(max(workers, 16), max(n_jobs, 1))) as ex:
                futures = [ex.submit(_read_alpha, e) for e in alpha_entries]
                futures += [ex.submit(_read_row, job) for job in row_jobs]
                futures += [ex.submit(_read_layer, job) for job in layer_jobs]
                for f in futures:
                    f.result()
    finally:
        bar.close()
        reader.close()

    sources: dict[str, list] = {}
    for name, banks in row_hb.items():
        views = []
        for bank, view_args in zip(banks, row_view_args[name]):
            if view_args is None:  # per-layer entry: already shaped [num_experts, ...]
                views.append(bank.tensor)
                continue
            head_pad, layer_bytes, num_experts, row_shape, dtype = view_args
            raw = bank.tensor[head_pad:head_pad + layer_bytes].view(dtype)
            views.append(raw.view(num_experts, *row_shape) if row_shape else raw.view(num_experts))
        sources[name] = views

    from freetoken.moe.legacy_format import canonical_role, kind_kernel_for
    from freetoken.moe.expert_banks import ExpertBanks

    # the file names the banks the legacy way; the quant_format tag names the (kind, kernel) they were packed for
    sources = {canonical_role(name): views for name, views in sources.items()}
    quant_format = reader.meta("quant_format")
    kind, kernel = kind_kernel_for(quant_format) if quant_format is not None else (None, None)

    # a failed mlock leaves a LOCKED layer pageable; the log and labels report what the banks actually settled at
    applied = list(residency)
    for banks in row_hb.values():
        for layer_id, bank in enumerate(banks):
            if (applied[layer_id] == HostResidency.LOCKED.value
                    and bank.residency is not HostResidency.LOCKED):
                applied[layer_id] = HostResidency.PAGEABLE.value
    unpinned = [i for i, r in enumerate(applied) if r != HostResidency.PINNED.value]
    if unpinned:
        by_layer = [0] * num_layers
        for name, banks in row_hb.items():
            for layer_id, bank in enumerate(banks):
                by_layer[layer_id] += bank.nbytes
        locked = [i for i in unpinned if applied[i] == HostResidency.LOCKED.value]
        pageable = [i for i in unpinned if i not in set(locked)]
        pinned_b = sum(b for i, b in enumerate(by_layer) if i not in set(unpinned))
        locked_b = sum(by_layer[i] for i in locked)
        pageable_part = ""
        if pageable:
            pageable_b = sum(by_layer[i] for i in pageable)
            pageable_part = (
                f" + {pageable_b / 2**30:.2f} GiB pageable "
                f"(lock failed, {len(pageable)} CPU layers: {pageable})"
            )
        logger.info(
            f"MoE bank split residency: {pinned_b / 2**30:.2f} GiB pinned "
            f"({'born-pinned cudaHostAlloc' if born else 'cudaHostRegister'}, "
            f"{num_layers - len(unpinned)} GPU layers) + "
            f"{locked_b / 2**30:.2f} GiB OS-locked ({len(locked)} CPU layers: {locked})"
            f"{pageable_part}"
        )

    # alphas are the small per-expert scale vectors, distinguished by their reserved names
    # (not a separate kind); everything else under experts_bank is a weight source.
    alpha_kw = {n: alpha_hb[n].tensor for n in alpha_hb}
    return ExpertBanks(
        quant_format, sources, **alpha_kw,
        layer_residency=applied, kind=kind, kernel=kernel, tp_sliced=tp_sliced,
        # the HostBank backings: lets the TP>1 re-slice dispose the abandoned full
        # banks (else their pinned pages stay resident + page-locked for the process
        # lifetime — ~1.5x the serving set at TP=2)
        # canonical keys: the re-slicer matches sources' canonical roles — a legacy
        # name here would miss the dispose and keep the full banks pinned forever
        host_banks={canonical_role(name): list(banks) for name, banks in row_hb.items()},
    )


__all__ = [
    "INDEX_NAME", "FORMAT_TAG", "FORMAT_VERSION", "ALIGN", "DEFAULT_SHARD_LIMIT",
    "DSPARK_BANK_KIND", "DSPARK_BANK_NUM_LAYERS",
    "is_ftw_checkpoint", "FTWWriter", "FTWReader",
    "iter_ftw_weights", "load_ftw_banks", "layer_bank_entry_name",
]
