"""DeepSeek-V4.1 Engram n-gram memory (reference inference/engram.py + model.py:296-365).

A conditional-memory lookup: tokens are hashed into 24 disjoint prime bucket
ranges per layer (2/3/4-gram × 8 heads), rows gathered from a ~189 GiB
MXFP8 table that stays ON DISK (in-place ``pread`` from the raw checkpoint's
safetensors shards — the OS page cache holds the hot set), dequantized, passed
through a learned gate and added into the raw hc residual stream at the TOP of
the block (layers 1 and 14), before that block's attention sublayer.

Hash (verbatim semantics from the reference):
  compressed id c_p per position (tokenizer decode + NFKC/NFD/StripAccents/
  Lowercase/whitespace-fold pipeline, first-seen order, 99092 ids);
  rolling_g = XOR_{i<g} (c_{p-i} * mult[layer][i])   (int64, mult odd, per-layer
  RNG seed 10007*layer_id, bound = (int64max // vocab) // 2)
  row[h, g] = rolling_g % prime[layer][h][g] + offset[layer][h][g]
  primes drawn ascending from 15999999, a GLOBAL seen set across layers, 24
  per layer (8 heads × 3 sizes), offsets = their cumsum → the layer's table row
  count (384006168 / 384016682) is the padded total.
Lookback pads with the pad token's compressed id past the sequence start; the
compressed-id cache is keyed by page-table row (continuous-batching safe).
"""

from __future__ import annotations

import bisect
import json
import mmap
import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from freetoken.layers import BaseOP, LinearReplicated

from .args import DeepseekV41Args

_ENGRAM_VOCAB_FILE = "engram-vocab.json"
_ROW_BYTES = 256
_SCALE_BYTES = 8
# engram value-read threads: ZFS serves ~5 cold random reads in parallel, more never helped
_READ_WORKERS = 8
# Prefill micro-batch (tokens per engram forward slice): the gate math is
# per-token independent, but its fp32 transients (hash gather, h/key casts)
# scale with the prefill chunk — one whole --max-extend-tokens chunk (8192)
# peaks around a GiB of temporaries and OOMs inside the (1 - memory_ratio)
# headroom. 1024 tokens bound them to ~100 MB regardless of the chunk size.
_PREFILL_MICRO_BS = 1024


# --------------------------------------------------------------------- vocab build
def _normalize_token(normalizer, token_text: str) -> str:
    normalized = normalizer.normalize_str(token_text)
    return normalized if normalized else token_text


def build_compressed_vocab(model_path: str, vocab_size: int, compressed_size: int, pad_token_id: int) -> dict:
    """token id → compressed id (first-seen order over normalized token texts).

    Cached as ``engram-vocab.json`` next to the checkpoint (a one-time tokenizer
    pass over all ``vocab_size`` ids)."""
    cache_path = os.path.join(model_path, _ENGRAM_VOCAB_FILE)
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            doc = json.load(f)
        if doc["vocab_size"] == vocab_size and len(doc["compressed_ids"]) == vocab_size:
            return doc

    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), "\ue000"),
        normalizers.Strip(),
        normalizers.Replace("\ue000", " "),
    ])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    backend = tokenizer.backend_tokenizer
    token_map: dict[str, int] = {}
    compressed_ids = []
    for tid in range(vocab_size):
        text = backend.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize — key by its raw
            # vocab form (reference engram.py:48-51)
            key = backend.id_to_token(tid)
        else:
            normalized = _normalize_token(normalizer, text)
            key = normalized if normalized else text
        if key not in token_map:
            token_map[key] = len(token_map)
        compressed_ids.append(token_map[key])
    if len(token_map) != compressed_size:
        raise ValueError(
            f"engram compressed vocab mismatch: built {len(token_map)}, checkpoint declares {compressed_size}"
        )
    doc = {
        "vocab_size": vocab_size,
        "compressed_size": compressed_size,
        "pad_token_id": pad_token_id,
        "compressed_ids": compressed_ids,
    }
    with open(cache_path, "w") as f:
        json.dump(doc, f)
    return doc


# --------------------------------------------------------------------- layout
def _next_prime(n: int, seen: set[int]) -> int:
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        f = 3
        while f * f <= x:
            if x % f == 0:
                return False
            f += 2
        return True

    while n in seen or not is_prime(n):
        n += 1
    return n


class EngramLayout:
    """Per-layer hash constants (multipliers, primes, offsets) — the RNG and
    prime-drawing order replicate the reference exactly (a deviation silently
    rehashes every table row)."""

    def __init__(self, args: DeepseekV41Args):
        vocab = args.engram_vocab_size  # 16000000
        # every hash multiplier derives from the COMPRESSED vocab size (reference
        # engram.py: "a mismatch there would silently rehash the whole table")
        bound = max(1, (np.iinfo(np.int64).max // args.engram_compressed_vocab_size) // 2)
        seen: set[int] = set()
        self.multipliers: dict[int, torch.Tensor] = {}
        self.primes: dict[int, torch.Tensor] = {}
        self.offsets: dict[int, torch.Tensor] = {}
        for layer_id in args.engram_layer_ids:
            rng = np.random.default_rng(10007 * layer_id)
            values = rng.integers(low=0, high=bound, size=(args.engram_max_ngram_size,), dtype=np.int64)
            self.multipliers[layer_id] = torch.tensor(values * 2 + 1, dtype=torch.int64)
            primes = []
            for _size in range(args.engram_max_ngram_size - 1):
                for _head in range(args.engram_n_heads):
                    p = _next_prime(vocab - 1, seen)
                    seen.add(p)
                    primes.append(p)
            # drawing order: size-major (size 0 heads 0..7, size 1 …). The offsets
            # are the GLOBAL 24-cumsum (reference: np.cumsum([0, *sizes[:-1]]) with
            # sizes = the layer's flattened primes) — sum(primes) == the table's row
            # count exactly (384006168 / 384016682), the 24 bucket ranges are disjoint.
            self.primes[layer_id] = torch.tensor(primes, dtype=torch.int64).view(self.n_sizes_of(args), args.engram_n_heads)
            self.offsets[layer_id] = torch.cumsum(torch.tensor([0, *primes[:-1]], dtype=torch.int64), dim=0)

    @staticmethod
    def n_sizes_of(args) -> int:
        return args.engram_max_ngram_size - 1

    def rows_for(self, layer_id: int, tokens: torch.Tensor) -> torch.Tensor:
        """Compressed ids [M, 4] → 24 row indices [M, heads*sizes] (int64,
        size-major: [s0h0..s0h7, s1h0.., s2h0..])."""
        mult = self.multipliers[layer_id].to(tokens.device)
        primes = self.primes[layer_id].to(tokens.device)
        offsets = self.offsets[layer_id].to(tokens.device)
        products = tokens.unsqueeze(1) * mult.view(1, -1)  # [M, 1, 4]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, products.shape[-1]):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[i - 1])
        return torch.cat(hashes, dim=-1) + offsets.view(-1)  # [M, heads*sizes] size-major


# --------------------------------------------------------------------- disk rows
class EngramRowSource:
    """In-place ``pread`` access to one layer's engram table (values uint8 fp8-codes
    [rows, 256], scales e8m0 [rows, 8]) — in the raw checkpoint's safetensors shards or,
    for an FTW converted with ``--include-engram``, in the FTW shards (a table spans
    several of them). Each table is a list of segments ``(row_lo, row_hi, path, off)``;
    value reads run on a thread pool (os.pread releases the GIL), the OS page cache holds
    the hot rows.

    The scale table (8 B/row, ~3 GiB per layer) is read into RAM once: a random 8-byte
    pread costs as much as the 256-byte value read (on ZFS both fetch a whole record,
    ~0.3 ms cold), so resident scales halve the reads on the decode critical path."""

    # the mmap subclass gathers scales from its mappings instead
    _RESIDENT_SCALES = True

    @staticmethod
    def _load_weight_map(model_path: str) -> dict:
        path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(path):
            return {}  # an FTW dir: no HF index (the tensors live in the FTW shards)
        with open(path) as f:
            return json.load(f)["weight_map"]

    def __init__(self, model_path: str, layer_id: int, prefix: str):
        self._prefix = prefix
        keys = {"w": f"{prefix}.embed.weight", "s": f"{prefix}.embed.scale"}
        widths = {"w": _ROW_BYTES, "s": _SCALE_BYTES}
        segs = self._ftw_segments(model_path, keys, widths)
        if segs is None:
            segs = self._safetensors_segments(model_path, keys)
        self._segs: dict[str, list[tuple[int, int, str, int]]] = segs
        self._starts = {k: [lo for lo, _, _, _ in v] for k, v in segs.items()}
        self.rows = segs["w"][-1][1]
        self._fds: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=_READ_WORKERS)
        self._scales: np.ndarray | None = self._load_scales() if self._RESIDENT_SCALES else None

    def _load_scales(self) -> np.ndarray:
        """The whole scale table as one [rows, 8] uint8 array (sequential reads)."""
        rows = self._segs["s"][-1][1]
        scales = np.empty((rows, _SCALE_BYTES), dtype=np.uint8)
        flat = scales.reshape(-1)
        for lo, hi, path, off in self._segs["s"]:
            view = memoryview(flat[lo * _SCALE_BYTES : hi * _SCALE_BYTES])
            with open(path, "rb", buffering=0) as f:
                f.seek(off)
                done = 0
                while done < len(view):
                    n = f.readinto(view[done:])
                    if not n:
                        raise EOFError(f"engram {self._prefix}.embed.scale: {path} ends inside the table")
                    done += n
        return scales

    @staticmethod
    def _ftw_segments(model_path: str, keys: dict, widths: dict):
        """Segments from an FTW that carries the table (``--include-engram``), else None."""
        from freetoken.checkpoint.ftw import FTWReader, is_ftw_checkpoint

        if not is_ftw_checkpoint(model_path):
            return None
        reader = FTWReader(model_path)
        try:
            names = {k: next((n for n in (key, "model." + key) if n in reader.tensors), None) for k, key in keys.items()}
            if None in names.values():
                return None
            out = {}
            for k, name in names.items():
                segs, row = [], 0
                for path, off, n in reader.file_pieces(name):
                    if n % widths[k]:
                        raise ValueError(f"engram {name}: shard piece of {n} B splits a {widths[k]} B row")
                    segs.append((row, row + n // widths[k], path, off))
                    row += n // widths[k]
                out[k] = segs
            return out
        finally:
            reader.close()

    def _safetensors_segments(self, model_path: str, keys: dict) -> dict:
        """One segment per table from the raw checkpoint's safetensors shards; an FTW
        without the tables falls back to the raw checkpoint next to it (``<name>`` for
        ``<name>-FTW``)."""
        weight_map = self._load_weight_map(model_path)
        if keys["w"] not in weight_map:
            raw = model_path[: -len("-FTW")] if model_path.endswith("-FTW") else model_path
            raw_map = self._load_weight_map(raw) if raw != model_path else {}
            if keys["w"] not in raw_map:
                raise FileNotFoundError(
                    f"engram table {keys['w']} is neither in {model_path} nor in a raw checkpoint "
                    f"at {raw} — convert with --include-engram or keep the raw checkpoint"
                )
            model_path, weight_map = raw, raw_map
        out = {}
        for k, key in keys.items():
            path = os.path.join(model_path, weight_map[key])
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            start, end = header[key]["data_offsets"]
            width = _ROW_BYTES if k == "w" else _SCALE_BYTES
            out[k] = [(0, (end - start) // width, path, 8 + n + start)]
        return out

    def _where(self, k: str, row: int) -> tuple[str, int]:
        i = bisect.bisect_right(self._starts[k], row) - 1
        lo, _, path, off = self._segs[k][i]
        width = _ROW_BYTES if k == "w" else _SCALE_BYTES
        return path, off + (row - lo) * width

    def _fd(self, path: str) -> int:
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def read_rows(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Rows ``indices`` (int64 CPU [N]) → ``out`` uint8 [N, 264] (256 value
        bytes pread on the pool, one contiguous run of rows per worker + 8 scale
        bytes gathered from the resident table)."""
        idx = indices.tolist()
        nv = len(idx)
        o = out.numpy()
        o[:, _ROW_BYTES:] = self._scales[indices.to(torch.int64).numpy()]

        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                vpath, voff = self._where("w", idx[i])
                o[i, :_ROW_BYTES] = np.frombuffer(os.pread(self._fd(vpath), _ROW_BYTES, voff), dtype=np.uint8)

        step = max(1, -(-nv // _READ_WORKERS))
        futures = [self._pool.submit(work, lo, min(lo + step, nv)) for lo in range(0, nv, step)]
        for f in futures:
            f.result()


class RamEngramRowSource(EngramRowSource):
    _RESIDENT_SCALES = False

    """mmap variant of :class:`EngramRowSource` (``--engram-backend ram``): every segment is
    mmap'd and the OS is asked to fault it in once at bind (``MADV_WILLNEED``); reads
    gather from the mappings. Page-cache resident and evictable under pressure — no
    ~189 GiB anonymous allocation, and the read is one bulk pass instead of cold random
    preads."""

    def __init__(self, model_path: str, layer_id: int, prefix: str):
        super().__init__(model_path, layer_id, prefix)
        self._maps: dict[str, mmap.mmap] = {}
        self._arrays: dict[str, list[np.ndarray]] = {}
        for k, segs in self._segs.items():
            width = _ROW_BYTES if k == "w" else _SCALE_BYTES
            arrays = []
            for lo, hi, path, off in segs:
                if path not in self._maps:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        self._maps[path] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                    finally:
                        os.close(fd)
                mm = self._maps[path]
                size = (hi - lo) * width
                try:
                    mm.madvise(mmap.MADV_WILLNEED, off - off % mmap.PAGESIZE, size + off % mmap.PAGESIZE)
                except (AttributeError, OSError):  # pragma: no cover — platform-dependent
                    pass  # unsupported: fall back to lazy page faults on access
                arrays.append(np.frombuffer(mm, dtype=np.uint8, count=size, offset=off).reshape(-1, width))
            self._arrays[k] = arrays

    def read_rows(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        idx = indices.to(torch.int64).numpy()
        seg = np.searchsorted(np.asarray(self._starts["w"]), idx, side="right") - 1
        o = out.numpy()
        for si, (lo, _, _, _) in enumerate(self._segs["w"]):
            sel = np.nonzero(seg == si)[0]
            if sel.size:
                o[sel, :_ROW_BYTES] = self._arrays["w"][si][idx[sel] - lo]
        sseg = np.searchsorted(np.asarray(self._starts["s"]), idx, side="right") - 1
        for si, (lo, _, _, _) in enumerate(self._segs["s"]):
            sel = np.nonzero(sseg == si)[0]
            if sel.size:
                o[sel, _ROW_BYTES:] = self._arrays["s"][si][idx[sel] - lo]


def make_engram_row_source(model_path: str, layer_id: int, prefix: str) -> EngramRowSource:
    """--engram-backend: 'disk' (buffered pread, default) or 'ram' (mmap + one
    bulk read). Read from FREETOKEN_ENGRAM_BACKEND, set by the --engram-backend
    flag before the workers spawn."""
    backend = os.environ.get("FREETOKEN_ENGRAM_BACKEND", "disk").strip().lower()
    if backend == "ram":
        return RamEngramRowSource(model_path, layer_id, prefix)
    return EngramRowSource(model_path, layer_id, prefix)


# --------------------------------------------------------------------- module
class Engram(BaseOP):
    """One layer's Engram memory: hash → disk gather → dequant → gated add into
    the raw hc residual stream (top of the block, before attention)."""

    def __init__(self, config, layer_id: int, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.head_dim = args.engram_head_dim       # 256
        self.n_heads = args.engram_n_heads         # 8
        self.n_sizes = args.engram_max_ngram_size - 1  # 2/3/4-gram
        self.n_hash_cols = self.n_sizes * self.n_heads  # 24
        self.hash_dim = self.n_hash_cols * self.head_dim  # 6144
        self.hc_mult = args.hc_mult
        self.dim = args.hidden_size
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        self.pad_token_id = args.engram_pad_token_id
        self.wkv = LinearReplicated(self.hash_dim, self.dim * (self.hc_mult + 1), has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.q_weight = torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16)
        self.k_weight = torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16)
        self._model_path = args.model_path  # parse_config fills this from _name_or_path
        self._layout = None   # EngramLayout (shared across layers, set by the model)
        self._vocab = None    # compressed-id lookup [vocab_size] int64 (set at bind)
        self._source = None   # EngramRowSource (set at bind)
        self._token_cache: torch.Tensor | None = None  # [cache_rows, max_seq] int64
        self._device: torch.device | None = None
        self._scratch: torch.Tensor | None = None  # CPU staging [M, 264] uint8
        # prefill prefetch: (positions tensor it was started for, row count, Future)
        self._pending: tuple[torch.Tensor, int, Future] | None = None
        self._prefetcher = ThreadPoolExecutor(max_workers=1)
        # decode (CUDA-graph) path: the host fills these BEFORE the dispatch
        # (rows pre-read from disk); the graph only does the H2D + dequant
        self._graph_pinned: torch.Tensor | None = None   # [max_bs, 264*n_hash_cols] uint8 CPU
        self._graph_dev: torch.Tensor | None = None      # same shape, device

    def bind(self, device: torch.device, layout: EngramLayout, vocab: torch.Tensor,
             max_seq_len: int, cache_rows: int, max_extend_tokens: int,
             comp_cpu: list[int], max_graph_bs: int = 16) -> None:
        """Load-time state (host side; runs before the engine touches CUDA) plus
        the host-side decode staging. Device pieces land in :meth:`rebind` (the
        pool/cache_rows exist only after the engine built the KV pool)."""
        self._layout = layout
        self._comp_cpu = comp_cpu  # token id → compressed id, CPU list (host hash)
        self._vocab_cpu = torch.tensor(comp_cpu, dtype=torch.int64)
        self._source = make_engram_row_source(self._model_path, self.layer_id, f"layers.{self.layer_id}.engram")
        self._max_seq_len = max_seq_len
        self._max_extend_tokens = max_extend_tokens
        self._max_graph_bs = max_graph_bs
        # decode pinned staging: one 24-row block per graph row (host-filled);
        # pinned memory so the captured graph's H2D copy is legal. Allocated
        # OUTSIDE inference_mode: host_fill_decode writes it from the row
        # source's worker threads (outside the forward's inference_mode), and an
        # inference tensor rejects those in-place updates.
        from freetoken.kernel.pinned import alloc_pinned_tensor

        with torch.inference_mode(False):
            self._graph_pinned = alloc_pinned_tensor(max_graph_bs * self.n_hash_cols, _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8)

    def rebind(self, device: torch.device, pool, max_seq_len: int, cache_rows: int,
               max_extend_tokens: int) -> None:
        """Device pieces on the first forward (the pool exists then)."""
        self._vocab = self._vocab_cpu.to(device)
        self._token_cache = torch.zeros(cache_rows, max_seq_len, dtype=torch.int64, device=device)
        self._graph_dev = torch.zeros(self._max_graph_bs * self.n_hash_cols, _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8, device=device)
        # CPU staging for the disk reads: a prefill chunk × 24 rows. The bind
        # runs inside the first forward's inference_mode — the worker threads
        # update the buffer OUTSIDE that context, so it must be a normal tensor.
        with torch.inference_mode(False):
            self._scratch = torch.empty(
                max_extend_tokens * self.n_hash_cols, _ROW_BYTES + _SCALE_BYTES,
                dtype=torch.uint8, device="cpu",
            )

    def host_fill_decode(self, ids_rows: list[list[int]]) -> None:
        """Host side of the decode path (runs on the engine thread BEFORE the
        dispatch): per row, hash the 4-gram from the request's CPU token
        history, pread the 24 rows and stage them into the pinned buffer."""
        layout = self._layout
        for i, ids in enumerate(ids_rows):
            tokens = torch.tensor([self._comp_cpu[t] for t in ids], dtype=torch.int64).unsqueeze(0)
            rows = layout.rows_for(self.layer_id, tokens).view(-1)
            base = i * self.n_hash_cols
            self._source.read_rows(rows, self._graph_pinned[base:base + self.n_hash_cols])

    def decode_consume(self, R: torch.Tensor) -> torch.Tensor:
        """Device side of the decode path (graph-capturable): H2D the staged
        rows, dequant, gate, add. R [B, hc, dim]."""
        B = R.shape[0]
        if os.path.exists("/tmp/dsv41-no-engram") and not torch.cuda.is_current_stream_capturing():
            return R
        hc, dim = self.hc_mult, self.dim
        n = B * self.n_hash_cols
        staged = self._graph_dev[:n]
        staged.copy_(self._graph_pinned[:n], non_blocking=True)
        vals = staged[:, :_ROW_BYTES].contiguous().view(torch.float8_e4m3fn).float()
        scales = staged[:, _ROW_BYTES:].contiguous().view(torch.uint8).float()
        scales = torch.exp2(scales - 127.0)
        emb = (vals.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)).flatten(-2).to(R.dtype)
        emb = emb.view(B, self.n_hash_cols, self.head_dim).flatten(-2)
        kv = self.wkv.forward(emb)
        key, value = kv.split([hc * dim, dim], dim=-1)
        key = key.float().unflatten(-1, (hc, dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = R.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return (R + (gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(R.dtype))

    def prefetch_rows(self, comp_ids: torch.Tensor, cache_rows: torch.Tensor, positions: torch.Tensor) -> None:
        """PREFILL: persist this chunk's compressed ids, hash every token's 2/3/4-gram
        lookback and start the disk gather of its rows on a background thread (one
        host sync for the row ids). The model calls this for every engram layer before
        the layer loop, so a later engram layer's read overlaps the earlier layers'
        compute; :meth:`forward` awaits it."""
        if os.path.exists("/tmp/dsv41-no-engram"):
            return
        self._token_cache[cache_rows, positions] = comp_ids
        M = positions.shape[0]
        pad = self._vocab[self.pad_token_id]
        tokens = torch.full((M, self.n_sizes + 1), pad, dtype=torch.int64, device=positions.device)
        blocked = torch.zeros(M, dtype=torch.bool, device=positions.device)
        for shift in range(self.n_sizes + 1):
            src = self._token_cache[cache_rows, (positions - shift).clamp_min(0)]
            blocked = blocked | (positions < shift)
            tokens[:, shift] = torch.where(blocked, pad, src)
        rows = self._layout.rows_for(self.layer_id, tokens).view(-1).cpu()  # [M*24]
        n = rows.numel()
        if self._scratch is None or self._scratch.shape[0] < n:
            # written from the reader threads, outside the forward's inference_mode
            with torch.inference_mode(False):
                self._scratch = torch.empty(n, _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8)
        future = self._prefetcher.submit(self._source.read_rows, rows, self._scratch[:n])
        self._pending = (positions, n, future)

    def forward(self, R: torch.Tensor, comp_ids: torch.Tensor, cache_rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """PREFILL path (eager): R [M, hc, dim] raw residual stream (M = flat
        tokens); comp_ids [M] compressed ids of this step's tokens;
        cache_rows/positions [M] each token's cache slot; returns the
        engram-updated stream."""
        if os.path.exists("/tmp/dsv41-no-engram"):
            return R
        if os.path.exists("/tmp/dsv41-no-engram-gate"):
            return R  # rows are gathered (cache warm) but the gate add is skipped
        M = R.shape[0]
        hc, dim = self.hc_mult, self.dim
        # 1.-3. ids persisted, hashed and the disk gather started — normally already
        # done by prefetch_rows at the top of the forward, so the read overlapped the
        # layers before this one; only its completion is awaited here.
        pending = self._pending
        self._pending = None
        if pending is None or pending[0] is not positions or pending[1] != M * self.n_hash_cols:
            self.prefetch_rows(comp_ids, cache_rows, positions)
            pending, self._pending = self._pending, None
        pending[2].result()
        rows_all = self._scratch[: M * self.n_hash_cols]
        weight = self.q_weight.float() * self.k_weight.float()  # per-layer constant
        out = torch.empty_like(R)
        # 4. in micro-batches: every step below is per-token independent, so
        # slicing M only bounds the fp32 transients, never the result
        for s in range(0, M, _PREFILL_MICRO_BS):
            e = min(s + _PREFILL_MICRO_BS, M)
            scratch = rows_all[s * self.n_hash_cols : e * self.n_hash_cols]
            vals = scratch[:, :_ROW_BYTES].to(R.device).view(torch.float8_e4m3fn).float()
            scales = scratch[:, _ROW_BYTES:].to(R.device).view(torch.uint8).float()
            scales = torch.exp2(scales - 127.0)
            emb = (vals.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)).flatten(-2).to(R.dtype)
            emb = emb.view(e - s, self.n_hash_cols, self.head_dim).flatten(-2)  # [m, 6144]
            # 4. gate (reference model.py:328-365)
            kv = self.wkv.forward(emb)  # [m, hc*dim + dim]
            key, value = kv.split([hc * dim, dim], dim=-1)
            key = key.float().unflatten(-1, (hc, dim))
            h = R[s:e].float()
            rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
            dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
            out[s:e] = R[s:e] + (gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(R.dtype)
        return out


__all__ = ["Engram", "EngramLayout", "EngramRowSource", "RamEngramRowSource", "build_compressed_vocab", "make_engram_row_source"]
