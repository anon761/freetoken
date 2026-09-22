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

import json
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from freetoken.layers import BaseOP, LinearReplicated

from .args import DeepseekV41Args

_ENGRAM_VOCAB_FILE = "engram-vocab.json"
_ROW_BYTES = 256
_SCALE_BYTES = 8
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
    """In-place ``pread`` access to one layer's engram table in the raw
    checkpoint's safetensors shards (values uint8 fp8-codes [rows, 256],
    scales e8m0 [rows, 8]). Reads happen on a thread pool (os.pread releases
    the GIL); the OS page cache holds the hot rows."""

    @staticmethod
    def _load_weight_map(model_path: str) -> dict:
        path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(path):
            return {}  # an FTW dir: no HF index (the tensors live in the FTW shards)
        with open(path) as f:
            return json.load(f)["weight_map"]

    def __init__(self, model_path: str, layer_id: int, prefix: str):
        self._prefix = prefix
        weight_map = self._load_weight_map(model_path)
        if prefix + ".embed.weight" not in weight_map:
            # the FTW carries no engram tables (the Phase-2 reader skips them) —
            # the raw checkpoint stays on disk as the O_DIRECT source
            raw = model_path[: -len("-FTW")] if model_path.endswith("-FTW") else model_path
            if raw != model_path:
                raw_map = self._load_weight_map(raw)
                if prefix + ".embed.weight" in raw_map:
                    model_path, weight_map = raw, raw_map
        self._entries: dict[str, tuple[str, int]] = {}
        self._sizes: dict[str, int] = {}
        header_sizes: dict[str, int] = {}
        for key in (f"{prefix}.embed.weight", f"{prefix}.embed.scale"):
            shard = weight_map[key]
            path = os.path.join(model_path, shard)
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
                header_sizes[path] = 8 + n
            start, end = header[key]["data_offsets"]
            self._entries[key] = (path, header_sizes[path] + start)
            self._sizes[key] = end - start
        self.rows = self._sizes[f"{prefix}.embed.weight"] // _ROW_BYTES
        self._fds: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=8)

    def _fd(self, path: str) -> int:
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def read_rows(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Rows ``indices`` (int64 CPU [N]) → ``out`` uint8 [N, 264] (256 value
        bytes + 8 scale bytes, written by one task per index)."""
        (vpath, voff) = self._entries[f"{self._prefix}.embed.weight"]
        (spath, soff) = self._entries[f"{self._prefix}.embed.scale"]
        vf, sf = self._fd(vpath), self._fd(spath)
        idx = indices.tolist()
        nv = len(idx)

        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                r = idx[i]
                out[i, :_ROW_BYTES] = torch.frombuffer(
                    bytearray(os.pread(vf, _ROW_BYTES, voff + r * _ROW_BYTES)), dtype=torch.uint8
                )
                out[i, _ROW_BYTES:] = torch.frombuffer(
                    bytearray(os.pread(sf, _SCALE_BYTES, soff + r * _SCALE_BYTES)), dtype=torch.uint8
                )

        step = max(1, nv // 16)
        futures = [self._pool.submit(work, lo, min(lo + step, nv)) for lo in range(0, nv, step)]
        for f in futures:
            f.result()


class RamEngramRowSource(EngramRowSource):
    """mmap variant of :class:`EngramRowSource` (``--engram-backend ram``): the
    layer's weight/scale regions are mmap'd and the OS is asked to fault the
    whole table in once at bind (``MADV_WILLNEED``); reads gather from the
    mapping. Page-cache resident and evictable under pressure — no ~189 GiB
    anonymous allocation, and the read is one bulk pass instead of cold random
    preads."""

    def __init__(self, model_path: str, layer_id: int, prefix: str):
        super().__init__(model_path, layer_id, prefix)
        self._maps: dict[str, mmap.mmap] = {}
        self._arrays: dict[str, np.ndarray] = {}
        for key in (f"{prefix}.embed.weight", f"{prefix}.embed.scale"):
            path, off = self._entries[key]
            size = self._sizes[key]
            if path not in self._maps:
                fd = os.open(path, os.O_RDONLY)
                try:
                    self._maps[path] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                finally:
                    os.close(fd)
            mm = self._maps[path]
            try:
                mm.madvise(mmap.MADV_WILLNEED, off, size)  # one bulk read into the page cache
            except (AttributeError, OSError):  # pragma: no cover — platform-dependent
                pass  # unsupported: fall back to lazy page faults on access
            self._arrays[key] = np.frombuffer(mm, dtype=np.uint8, count=size, offset=off)
        self._w = self._arrays[f"{prefix}.embed.weight"].reshape(-1, _ROW_BYTES)
        self._s = self._arrays[f"{prefix}.embed.scale"].reshape(-1, _SCALE_BYTES)

    def read_rows(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        idx = indices.to(torch.int64).numpy()
        out[:, :_ROW_BYTES].numpy()[...] = self._w[idx]
        out[:, _ROW_BYTES:].numpy()[...] = self._s[idx]


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
        # 1. persist this step's compressed ids (whole chunk, once)
        self._token_cache[cache_rows, positions] = comp_ids
        weight = self.q_weight.float() * self.k_weight.float()  # per-layer constant
        out = torch.empty_like(R)
        # 2.-4. in micro-batches: every step below is per-token independent, so
        # slicing M only bounds the fp32 transients, never the result
        for s in range(0, M, _PREFILL_MICRO_BS):
            e = min(s + _PREFILL_MICRO_BS, M)
            # 2. hash: 2/3/4-gram lookback out of the cache (pad past the sequence start)
            pad = self._vocab[self.pad_token_id]
            tokens = torch.full((e - s, self.n_sizes + 1), pad, dtype=torch.int64, device=R.device)
            blocked = torch.zeros(e - s, dtype=torch.bool, device=R.device)
            for shift in range(self.n_sizes + 1):
                src = self._token_cache[cache_rows[s:e], (positions[s:e] - shift).clamp_min(0)]
                blocked = blocked | (positions[s:e] < shift)
                tokens[:, shift] = torch.where(blocked, pad, src)
            rows = self._layout.rows_for(self.layer_id, tokens).view(-1)  # [m*24]
            # 3. disk gather + dequant (CPU pread → device)
            n = rows.numel()
            if self._scratch is None or self._scratch.shape[0] < n:
                self._scratch = torch.empty(n, _ROW_BYTES + _SCALE_BYTES, dtype=torch.uint8)
            scratch = self._scratch[:n]
            self._source.read_rows(rows.cpu(), scratch)
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
