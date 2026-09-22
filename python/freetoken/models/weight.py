"""Checkpoint-loading entry points, routed per model architecture.

Each loader resolves the model spec from the checkpoint config and dispatches to the
model module (``models/<name>/weight.py``). Passing ``dummy=True`` replaces the
checkpoint read with randomly filled tensors that keep the loader's exact output
contract (shapes, dtypes, pinning), so everything downstream — repack, offload cache,
kernels — runs unchanged without weights on disk. A model whose banks differ from the
default layout opts out by defining the same-named ``dummy_*`` hook in its weight
module; otherwise the defaults below (built purely from the parsed config) apply.
"""

from __future__ import annotations

import glob
import json
import mmap
import os
import queue
import time
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, Tuple

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import cached_load_hf_config

from .register import _load_attr, get_model_spec

# safetensors header dtype strings -> torch dtypes (for the parallel reader below)
_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
}
_ODIRECT_BLK = 4096


def _read_shard_odirect_parallel(path: str, workers: int, chunk: int) -> mmap.mmap:
    """Read a whole shard into a page-aligned mmap via CHUNKED multi-threaded O_DIRECT.
    Multi-threading one fd scales even for single-shard checkpoints (measured ~7x at 8
    threads): the kernel issues the parallel preads at high queue depth. DMA bypasses the
    page cache, so there's nothing to drop afterwards."""
    size = os.path.getsize(path)
    asize = ((size + _ODIRECT_BLK - 1) // _ODIRECT_BLK) * _ODIRECT_BLK
    buf = mmap.mmap(-1, asize)
    mv = memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, asize - o)
        want = min(want, ((size - o + _ODIRECT_BLK - 1) // _ODIRECT_BLK) * _ODIRECT_BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return buf


def iter_expert_tensors_parallel(
    model_path: str,
    is_expert: Callable[[str], bool],
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_cache: bool = True,
    prefetch: int = 2,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Parallel O_DIRECT analog of a model's serial expert ``iter_weights``.

    Common scaffolding for the "parallel" load path: reads every shard that holds >=1 expert
    tensor (``is_expert(name)`` is the model's per-model predicate) with chunked
    multi-threaded O_DIRECT, parses the safetensors header, and yields ``(name, tensor)``
    in checkpoint dtype/shape for the expert tensors.

    A background reader PREFETCHES the next ``prefetch`` shards (each chunked O_DIRECT)
    while the consumer places the current one, so the disk stays busy during placement
    instead of idling between shards (the gap that made the naive sequential version slow).
    Peak host memory is ~(prefetch+1) shards + the banks the caller fills. Order is
    shard-then-header order (NOT global), so the consumer must place by ``name``.
    """
    from freetoken.utils.hf import download_hf_weight

    model_path = download_hf_weight(model_path)  # resolve hub id -> local (parity w/ serial)
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:  # single-file / no-index checkpoint: map name -> shard from each shard's header
        weight_map = {}
        for shard in sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_path, "*.safetensors"))):
            with open(os.path.join(model_path, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for nm in hdr:
                if nm != "__metadata__":
                    weight_map[nm] = shard
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if is_expert(name):
            shards.setdefault(shard, []).append(name)
    shard_list = sorted(shards)

    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []

    def _reader():
        try:
            for shard in shard_list:
                path = os.path.join(model_path, shard)
                if drop_cache:
                    try:
                        fd = os.open(path, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                    except OSError:
                        pass
                buf = _read_shard_odirect_parallel(path, workers, chunk)  # overlaps placement
                n = struct.unpack("<Q", bytes(buf[:8]))[0]
                hdr = json.loads(bytes(buf[8:8 + n]))
                q.put((buf, hdr, 8 + n, shards[shard], os.path.getsize(path)))
        except BaseException as e:  # surface reader errors to the consumer
            err.append(e)
        finally:
            q.put(_DONE)

    from freetoken.utils.progress import byte_bar

    th = threading.Thread(target=_reader, name="expert-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(os.path.getsize(os.path.join(model_path, s)) for s in shard_list),
                   "Loading experts (parallel)", monitor=True)
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            buf, hdr, base, names, shard_sz = item
            mv = memoryview(buf)
            for name in names:
                meta = hdr[name]
                b, e = meta["data_offsets"]
                t = torch.frombuffer(mv[base + b: base + e], dtype=_ST_DTYPE[meta["dtype"]])
                yield name, (t.view(*meta["shape"]) if meta["shape"] else t)
            bar.update(shard_sz)
            del mv, buf  # freed once the consumer drops the last yielded tensor of this shard
    finally:
        bar.close()
        # If the consumer exits early (exception, close), the prefetch thread stays
        # blocked on q.put with a full queue — joining it here would deadlock forever.
        # Drain until the thread ends so the consumer's exception propagates.
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.01)
        th.join()
    if err:
        raise err[0]


_SCATTERED_AVG_BYTES = 16 << 20  # avg expert tensor below this -> "scattered" -> prefer parallel


def experts_scattered(model_path: str) -> bool:
    """Slow-path strategy signal: are the experts stored as many SMALL tensors?

    If yes (per-expert / quantized layouts -> avg expert tensor a few MiB), the serial
    baseline pays per-tensor overhead on thousands of tiny reads and is slow, so the parallel
    parallel whole-shard O_DIRECT reader wins. If experts are pre-packed into a few large
    tensors, the serial read already saturates the disk and parallel only adds read amplification.
    Measured cheaply from the safetensors headers (no tensor data read). This is a best-
    effort heuristic: ANY failure (unresolvable path, no safetensors, GGUF, unreadable
    header) -> False (serial), so the real loader still runs and reports real errors."""
    try:
        from freetoken.utils.hf import download_hf_weight

        model_path = download_hf_weight(model_path)  # resolve hub ids -> local (parity w/ serial)
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_path, "*.safetensors")))
        sizes: list[int] = []
        for shard in shards:
            try:
                with open(os.path.join(model_path, shard), "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    hdr = json.loads(fh.read(n))
            except (OSError, ValueError, struct.error):  # unreadable/partial shard -> skip
                continue
            for name, meta in hdr.items():
                if name != "__metadata__" and ".experts." in name:
                    b, e = meta["data_offsets"]
                    sizes.append(e - b)
        if not sizes:
            return False
        return (sum(sizes) / len(sizes)) < _SCATTERED_AVG_BYTES
    except Exception:  # heuristic only -> default to serial; the real loader reports errors
        return False


def _spec_for_model_path(model_path: str):
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    parse_config = _load_attr(spec.module, spec.parse_config)
    return parse_config(hf_config), spec


def _model_override(spec, name: str):
    """The model module's same-named hook, if it defines one."""
    try:
        return _load_attr(spec.module, name)
    except AttributeError:
        return None


def side_table_files(model_path: str) -> list[str]:
    """Model-specific weight-ADJACENT data files that the model loads OUTSIDE the dense
    weight stream (qwen4_exp: the PLE n-gram table in ``model-plefp8-*.safetensors``).
    An FTW conversion must carry these over verbatim — they end in .safetensors, so the
    generic metadata walk skips them, and the checkpoint cannot serve without them.
    ``[]`` when the model has no side tables (or the architecture is unknown)."""
    try:
        _config, spec = _spec_for_model_path(model_path)
    except Exception:  # noqa: BLE001 — unknown arch / unreadable config: nothing to carry
        return []
    fn = _model_override(spec, "side_table_files")
    if fn is None:
        return []
    try:
        return list(fn(model_path) or [])
    except Exception:  # noqa: BLE001 — never fail a conversion over the carry-over
        return []


def load_weight(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    mtp_path: str = "",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Base stream, then — with ``--mtp file`` — the standalone draft head.

    With an external draft the base stream's own ``mtp.*`` tensors are dropped
    unconditionally (the binding is external; a base checkpoint that happens to
    ship a draft must not race the artifact)."""
    if not mtp_path:
        yield from _load_weight_inner(model_path, device, include_moe_experts=include_moe_experts)
        return
    # The base stream must not read the checkpoint's own MTP head: the artifact owns it.
    # ``include_mtp=False`` suppresses the family's embedded-head reader (the base may
    # declare ``mtp_num_hidden_layers`` yet ship no ``mtp.*`` tensors, as the hybrid
    # checkpoints do). Any base ``mtp.*`` keys are filtered below too.
    base = _load_weight_inner(
        model_path, device, include_moe_experts=include_moe_experts, include_mtp=False
    )
    for name, tensor in base:
        if name.startswith(("mtp.", "model.mtp.")):
            continue
        yield name, tensor
    from freetoken.distributed import get_tp_info

    _cfg, _spec = _spec_for_model_path(model_path)
    try:
        _load_attr(_spec.module, "tp_shard_key")
    except AttributeError as exc:
        raise ValueError(
            f"--mtp file understuetzt dieses Modell-Familien-Modul nicht: {_spec.module}"
        ) from exc
    from freetoken.models.qwen4_exp.weight import iter_external_mtp_weights

    for name, tensor in iter_external_mtp_weights(mtp_path, device, _cfg, get_tp_info()):
        yield name, tensor


def _load_weight_inner(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_mtp: bool = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    # FTW checkpoint: dense weights are stored post-iter_weights, so we replay them
    # model-agnostically instead of re-running the per-model reader. Which tensors exist is
    # decided at conversion (offload -> experts live in banks, not here); a backend mismatch
    # fails loudly in load_state_dict (strict missing/unexpected expert keys), so the reader
    # just yields the stored weight tensors regardless of the include_moe_experts flag.
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, iter_ftw_weights
    from freetoken.models.config import VISION_KEY_PREFIXES, vision_load_enabled

    if is_ftw_checkpoint(model_path):
        # The FTW dense shard stores whatever existed at conversion, including the vision
        # stack. Vision is opt-in (default OFF, see vision_load_enabled): when it is off the
        # model never builds the tower, so replaying those tensors would trip load_state_dict's
        # strict unexpected-key check. Skip them here to match the model the engine built.
        # FTW stores dense weights un-sharded (TP-agnostic); a model module may expose
        # shard_ftw_weight(name, w, model_path, tp) to apply its TP key rules at replay.
        skip_vision = not vision_load_enabled()
        _cfg, _spec = _spec_for_model_path(model_path)
        # Same strictness for the MTP draft head: the FTW carries mtp.* (converted with it
        # forced on), but THIS serve may run without FREETOKEN_ENABLE_MTP — the model has no
        # mtp module then, and the keys must not reach load_state_dict.
        # (--mtp file already dropped the FTW's own mtp.* in load_weight; the external draft
        # is yielded there, so this gate only reflects the serve's MTP config.)
        mtp_on = bool(getattr(_cfg, "mtp_enabled", False)) and include_mtp
        try:
            _shard_fn = _load_attr(_spec.module, "shard_ftw_weight")
        except AttributeError:
            _shard_fn = None
        _tp = get_tp_info()
        for name, tensor in iter_ftw_weights(model_path):
            if skip_vision and name.startswith(VISION_KEY_PREFIXES):
                continue
            if not mtp_on and name.startswith(("mtp.", "model.mtp.")):
                continue
            if _shard_fn is not None and _tp.size > 1:
                tensor = _shard_fn(name, tensor, model_path, _tp)
            yield name, tensor
        return

    _config, spec = _spec_for_model_path(model_path)
    iter_weights = _load_attr(spec.module, spec.iter_weights)
    kwargs = {"include_moe_experts": include_moe_experts, "include_non_moe": True}
    if not include_mtp:
        # Only the qwen4_exp reader understands an external MTP head, so the flag is
        # only ever needed (and passed) there; other families keep the plain contract.
        kwargs["include_mtp"] = False
    yield from iter_weights(model_path, device, **kwargs)


def load_q4_0_moe_expert_sources(
    model_path: str,
    model_config,
    *,
    dummy: bool = False,
    layer_sink=None,
) -> dict:
    """Load (or fabricate, with ``dummy=True``) packed GGUF Q4_0 expert source banks.
    ``layer_sink`` (converter) streams each completed layer's banks; ignored for dummy."""
    _config, spec = _spec_for_model_path(model_path)
    if dummy:
        builder = _model_override(spec, "dummy_q4_0_expert_sources")
        assert builder is not None, "model defines no dummy_q4_0_expert_sources"
        return builder(model_config)
    loader = _load_attr(spec.module, "load_q4_0_expert_sources")
    return loader(model_path, model_config, layer_sink=layer_sink)


def dspark_expert_method(model_path: str, model_config):
    """The model's DSpark draft expert method, or ``None`` when it ships no draft.

    Dispatches to the family's ``dspark_expert_method`` hook (deepseek_v41); the draft's
    expert count differs from the target's, so it owns a separate method/bank set."""
    _config, spec = _spec_for_model_path(model_path)
    fn = _model_override(spec, "dspark_expert_method")
    if fn is None:
        return None
    return fn(model_config)


def load_dspark_banks(model_path: str, model_config, **kwargs):
    """Load the model's DSpark draft expert banks (family ``load_dspark_banks`` hook).

    Raises when the family ships no draft: a checkpoint with ``mtp.*`` but no bank
    loader would otherwise silently serve without its draft experts."""
    _config, spec = _spec_for_model_path(model_path)
    loader = _model_override(spec, "load_dspark_banks")
    if loader is None:
        raise ValueError(f"{spec.module} ships no DSpark draft expert banks")
    return loader(model_path, model_config, **kwargs)


__all__ = [
    "load_weight",
    "load_q4_0_moe_expert_sources",
    "iter_expert_tensors_parallel",
    "dspark_expert_method",
    "load_dspark_banks",
]
