"""Convert an HF safetensors checkpoint into a FreeToken Weight (FTW) checkpoint.

Model-agnostic: it drives the *existing* per-model loaders once and stores their output, so
no per-model conversion code is needed.

* dense weights = exactly what ``load_weight(include_moe_experts=...)`` yields (post
  fusion/TP-shard) -> ``kind="weight"``; at load they feed ``model.load_state_dict``.
* offload experts = exactly what ``load_expert_banks(parallel=True)`` produces (post
  backend-repack pinned banks + alpha scale vectors) -> ``kind="experts_bank"`` (alphas are
  told apart at load by their reserved names, so they need no separate kind).

The output directory is a self-contained checkpoint (config + tokenizer copied), so you can
point ``--model`` straight at it; the load path auto-detects the FTW and reads it (FTW).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import threading


import torch
from safetensors.torch import save_file

from freetoken.utils import cached_load_hf_config, download_hf_weight

from .ftw import DEFAULT_SHARD_LIMIT, DSPARK_BANK_NUM_LAYERS, FTWWriter, layer_bank_entry_name

# Machine-readable convert progress for a supervising process (e.g. a GUI frontend parses
# these `FTCONVERT <phase> <done> <total>` stdout lines to drive its convert bar). Gated by
# FREETOKEN_CONVERT_PROGRESS=1 so plain CLI use isn't spammed; the human tqdm bars stay on
# stderr. Phases: `dense` (indeterminate, done=total=0), `experts` (byte totals), `finalize`.
_EMIT_PROGRESS = os.environ.get("FREETOKEN_CONVERT_PROGRESS") == "1"


def _progress(phase: str, done: int = 0, total: int = 0) -> None:
    if _EMIT_PROGRESS:
        print(f"FTCONVERT {phase} {done} {total}", flush=True)


def _source_fingerprint(model_path: str, model_config, *, device) -> str:
    """Identity of (checkpoint + quant + GPU capability), stored in the FTW index so
    it's clear what an FTW was built from. Cheap (stat only)."""
    h = hashlib.sha256()
    h.update(f"quant={getattr(model_config, 'expert_quant', None)}|".encode())
    h.update(f"arch={getattr(model_config, 'architectures', None)}|".encode())
    try:  # nvfp4 marlin/b12x layout depends on compute capability
        h.update(f"cc={torch.cuda.get_device_capability(device)}|".encode())
    except Exception:
        pass
    files = sorted(
        glob.glob(os.path.join(model_path, "*.safetensors")) + glob.glob(os.path.join(model_path, "*.gguf"))
    )
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:16]

# Checkpoint metadata to carry over so the FTW dir is a usable checkpoint on its own.
# (Weight shards + the safetensors index are intentionally NOT copied.)
# Everything that is NOT a weight shard is metadata we carry over verbatim. A whitelist
# misses model-specific layouts (e.g. DSV4's inference/config.json + encoding/ live in
# subdirs), so we copy every non-weight file preserving its relative path instead.
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ftw")  # .ftw: a nested FTW, not source
_SKIP_NAMES = ("model.safetensors.index.json",)  # indexes shards the FTW replaces
# The PLE n-gram table's key infix (qwen4_exp; mirrors models/qwen4_exp/weight.py).
_PLE_KEY_INFIX = ".ple.ple_embedding.ngram_embedding." 


def _ple_table_source_dtype(header: dict, ple_keys: list[str]) -> str | None:
    """The PLE table's data-tensor storage tag (``"fp8"`` / ``"bf16"``), or None when the
    table is empty or mixes dtypes. The scalar ``weight_scale`` sibling is BF16 even in fp8
    tables and must not flip the verdict — a combined shard like NVIDIA's
    ``model-fp8-mtp-ple`` would otherwise be re-encoded, pulling the whole table into RAM."""
    table_keys = [k for k in ple_keys if not k.endswith(".weight_scale")]
    if not table_keys:
        return None
    dtypes = {header[k].get("dtype") for k in table_keys}
    if dtypes == {"F8_E4M3"}:
        return "fp8"
    if dtypes <= {"BF16", "F32"}:
        return "bf16"
    return None


def _ple_rides_verbatim(header: dict, ple_keys: list[str], out_dtype: str) -> bool:
    """True when the table's stored dtype already matches the requested one (or the request
    is ``auto``), so the shard(s) can be copied verbatim instead of re-encoded through RAM."""
    src = _ple_table_source_dtype(header, ple_keys)
    return src is not None and (out_dtype == "auto" or out_dtype == src)


def _ships_mtp(model_path: str) -> bool:
    """Whether the source checkpoint stores any ``mtp.*`` tensors (a draft head)."""
    import struct

    folder = download_hf_weight(model_path)
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            return any(k.startswith("mtp.") for k in json.load(fh)["weight_map"])
    for path in glob.glob(os.path.join(folder, "*.safetensors")):
        try:
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            continue
        if any(k.startswith("mtp.") for k in header):
            return True
    return False


def _validate_mtp_artifact(main_config, header: str) -> None:
    """Refuse an external draft head whose geometry differs from the base model (a wrong
    draft silently produces garbage tokens)."""
    if not os.path.exists(header):
        raise SystemExit(f"--mtp-header not found: {header}")
    folder = header if os.path.isdir(header) else os.path.dirname(os.path.abspath(header))
    from freetoken.models.qwen4_exp.config import parse_config as _qwen4_parse
    from freetoken.models.qwen4_exp.weight import mtp_compat_reasons

    try:
        artifact = _qwen4_parse(cached_load_hf_config(folder))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--mtp-header has no readable config.json ({folder}): {exc}") from None
    reasons = mtp_compat_reasons(main_config, artifact)
    if reasons:
        raise SystemExit("--mtp-header passt nicht zum Modell: " + "; ".join(reasons))
# .freetoken_expert_cache: the legacy per-bank cache (can be tens of GB of stale .bin)
_SKIP_DIRS = (".git", ".cache", ".freetoken_expert_cache")


def _copy_metadata(model_path: str, out_dir: str) -> list[str]:
    """Copy all non-weight files (config, tokenizer, remote-code, nested model configs)
    preserving directory structure, so the FTW dir is a self-contained checkpoint."""
    if os.path.isfile(model_path):
        # A single-file source has no sibling metadata to walk: a .gguf carries its config
        # AND tokenizer in its own KV section. Emit a metadata-only copy (header + KV, no
        # weight data) the FTW dir resolves those from. We deliberately do NOT sweep the
        # file's parent dir -- an HF gguf snapshot dir can hold unrelated blobs.
        from freetoken.models.gguf.reader import (
            FTW_METADATA_GGUF,
            is_gguf_path,
            write_metadata_gguf,
        )

        if is_gguf_path(model_path):
            os.makedirs(out_dir, exist_ok=True)
            write_metadata_gguf(model_path, os.path.join(out_dir, FTW_METADATA_GGUF))
            return [FTW_METADATA_GGUF]
        return []

    out_abs = os.path.abspath(out_dir)
    copied = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and os.path.abspath(os.path.join(root, d)) != out_abs]
        for name in files:
            if name.endswith(_WEIGHT_SUFFIXES) or name in _SKIP_NAMES:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, model_path)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return copied


class _ConvertSink:
    """Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each
    completed layer's banks as their own FTW entries immediately (name
    ``f"{bank_name}#L{layer_id:05d}"``, kind ``"experts_bank"``) and releases them, so
    conversion RAM peaks at ~in-flight layers instead of the whole bank set.

    Only engaged for streamable formats -- ``ExpertBanks.streamed`` reports whether this
    actually fired; if not, the caller falls back to the materialize-and-write path
    instead. The progress bar is created lazily on the first call, so a format that never
    streams never shows one.

    ``FTWWriter`` buffers file/shard state and is not thread-safe; completion callbacks
    can fire from the loader's own reader threads, so the write+release is serialized
    under one lock (disk-bound anyway).
    """

    def __init__(self, writer: FTWWriter, desc: str = "Converting expert banks", *, names: dict[str, str] | None = None,
                 kind: str = "experts_bank", progress: bool = True) -> None:
        self._writer = writer
        self._desc = desc
        # canonical role -> the bank name the file stores
        self._names = names or {}
        # the FTW kind the entries are written under: the DSpark draft uses its own
        # namespace so the target's load_ftw_banks never sees its (differently sized) banks
        self._kind = kind
        # the machine-readable FTCONVERT expert bar is sized to the target's pool; the
        # draft's extra bytes would overshoot it, so the draft sink stays off it
        self._progress = progress
        self._bar = None
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.n_written = 0
        self.n_bytes = 0

    def __call__(self, layer_id: int, banks: dict) -> None:
        with self._lock:
            assert layer_id not in self._seen, f"layer {layer_id} streamed to the sink twice"
            self._seen.add(layer_id)
            if self._bar is None:
                from freetoken.utils.progress import byte_bar

                self._bar = byte_bar(0, self._desc)  # total unknown up front (streamed)
            nbytes = 0
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    layer_bank_entry_name(self._names.get(bank_name, bank_name), layer_id), bank.tensor, kind=self._kind
                )
                nbytes += bank.nbytes
                bank.release()
                self.n_written += 1
            self.n_bytes += nbytes
            self._bar.update(nbytes)
            # Cumulative BYTES (not the bank count): the supervisor maps this against the
            # known expert-pool size for a smooth phase-budgeted bar. Total stays 0 (unknown
            # up front while streaming); the materialize path below emits a real total.
            if self._progress:
                _progress("experts", self.n_bytes, 0)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def num_layers(self) -> int:
        return len(self._seen)



def _extract_ple_table(src: str, out_dir: str, copied: list[str], *, out_dtype: str) -> None:
    """Encode the n-gram table tensors of ``src`` into dedicated ``model-plebf16/plefp8``
    shards under ``out_dir`` (keys preserved, one file per ~6 GiB). ``out_dtype`` is
    ``"bf16"`` (store the table unquantized; an fp8 source is dequantized with its scalar
    ``weight_scale``, which is exact) or ``"fp8"`` (quantize a bf16 source with one scalar
    ``weight_scale`` derived from the global max abs). Streams one shard tensor at a time;
    a source already in the requested dtype would have taken the verbatim path instead."""
    import struct

    import safetensors
    import torch

    with open(src, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    ple_keys = [k for k in header if _PLE_KEY_INFIX in k]

    tensors: dict[str, torch.Tensor] = {}
    scale = None
    max_abs = 0.0
    with safetensors.safe_open(src, framework="pt", device="cpu") as f:
        for key in ple_keys:
            t = f.get_tensor(key)
            if key.endswith(".weight_scale"):
                scale = t.reshape(()).clone()
                continue
            if out_dtype == "fp8":
                # FP8 (Float8_e4m3fn) tensors have no max_all kernel — cast to
                # float32 first (a no-op for bf16 PLE tables like the RVN's).
                max_abs = max(max_abs, t.float().abs().max().item() if t.numel() else 0.0)
            tensors[key] = t

    if out_dtype == "fp8":
        if scale is None:
            scale = torch.tensor(max(max_abs, 1e-12) / 448.0, dtype=torch.bfloat16)
        fill_scale = float(scale)

        def encode(t: torch.Tensor) -> torch.Tensor:
            return (t.to(torch.float32) / fill_scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    else:
        if scale is not None:
            dequant = float(scale)

            def encode(t: torch.Tensor) -> torch.Tensor:
                return (t.to(torch.float32) * dequant).to(torch.bfloat16)
        else:

            def encode(t: torch.Tensor) -> torch.Tensor:
                return t.to(torch.bfloat16)

    tag = "plefp8" if out_dtype == "fp8" else "plebf16"
    shard_files: dict[str, dict[str, torch.Tensor]] = {}
    file_idx = 0
    used = 0
    for key, t in tensors.items():
        enc = encode(t)
        nbytes = enc.numel() * enc.element_size()
        fname = f"model-{tag}-{file_idx:05d}.safetensors"
        if used and used + nbytes > 6 << 30:
            file_idx += 1
            used = 0
            fname = f"model-{tag}-{file_idx:05d}.safetensors"
        shard_files.setdefault(fname, {})[key] = enc
        used += nbytes
    for fname, tensors_part in shard_files.items():
        if out_dtype == "fp8":
            tensors_part[_PLE_KEY_INFIX + "weight_scale"] = scale
        save_file(tensors_part, os.path.join(out_dir, fname))
        copied.append(fname)


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    quant_backend: str | None = None,
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
    include_engram: bool = False,
    include_dspark: bool = False,
    mtp_mode: str = "auto",
    mtp_header: str = "",
    ple_mode: str = "bf16",
    vision_mode: str = "on",
) -> dict:
    """Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict.

    ``include_engram`` / ``include_dspark`` opt the model's side payloads into the FTW:
    the ~189 GiB Engram n-gram tables (otherwise read in place from the source shards at
    serve time) and the DSpark MTP draft head (otherwise skipped). Both are OFF by
    default so the FTW stays the current dense+banks layout; the index records which
    were included under ``"includes"`` so the loader can tell an FTW apart. Including the
    engram costs ~189 GiB of extra disk.

    The FTW format is TP-agnostic and conversion runs single-process, so the resulting
    checkpoint records no TP layout and loads independently of the runtime TP setting.

    ``mtp_mode`` controls the (dense) MTP draft head, mirroring ``ft serve --mtp``:
    ``"auto"``/``"on"`` include the head embedded in the checkpoint (when its config
    declares ``mtp_num_hidden_layers``); ``"off"`` never includes it; ``"file"`` includes
    the standalone artifact ``mtp_header`` (directory or single .safetensors), replacing
    any embedded head, after validating its geometry against the base model."""
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.models.weight import load_weight
    from freetoken.moe.expert_banks import load_expert_banks
    from .ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(model_path):
        raise SystemExit(f"{model_path} is already an FTW checkpoint")
    if mtp_mode not in ("auto", "on", "off", "file"):
        raise SystemExit(f"--mtp must be auto|on|off|file, got {mtp_mode!r}")
    if mtp_mode == "file" and not mtp_header:
        raise SystemExit("--mtp file requires --mtp-header PATH")
    # Opt the requested side payloads in BEFORE any config is parsed: the per-model
    # weight readers gate mtp.*/engram.* on these at parse time (mirrors the existing
    # FREETOKEN_ENABLE_MTP gate).
    if include_dspark or mtp_mode in ("on", "file"):
        os.environ["FREETOKEN_ENABLE_MTP"] = "1"
    if include_engram:
        os.environ["FREETOKEN_EXPORT_ENGRAM"] = "1"
    if vision_mode not in ("on", "off"):
        raise SystemExit(f"--vision must be on|off, got {vision_mode!r}")
    # The vision tower is opt-in at config-parse time (FREETOKEN_LOAD_VISION), exactly
    # like --load-vision at serve: set/clear it before any config is parsed.
    if vision_mode == "on":
        os.environ["FREETOKEN_LOAD_VISION"] = "1"
    else:
        os.environ.pop("FREETOKEN_LOAD_VISION", None)
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    elif tp.size != 1:
        raise SystemExit(
            f"FTW conversion runs single-process and the format records no TP layout, "
            f"but TP is already set to size={tp.size}"
        )
    dev = torch.device(device or "cuda:0")
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)  # init CUDA context (needed by nvfp4 backend pick / pinning)

    cfg = EngineConfig(model_path=model_path, tp_info=DistributedInfo(tp.rank, tp.size),
                       dtype=dtype, moe_strategy=moe_backend, quant_backend=quant_backend)
    mc = cfg.model_config
    if mtp_mode == "file":
        _validate_mtp_artifact(mc, mtp_header)
    offload = moe_backend == "offload" and getattr(mc, "is_moe", False)
    include_moe_experts = not offload
    method = None
    if offload:
        from freetoken.engine.engine import offload_expert_method
        from freetoken.layers import set_rope_device

        set_rope_device(dev)
        method = offload_expert_method(cfg)

    # The FTW is a SELF-CONTAINED checkpoint: it carries everything the source had.
    # 1) The MTP draft head: the dense reader gates mtp.* on FREETOKEN_ENABLE_MTP at
    #    config-parse time, so force it ON before any config is parsed when the source
    #    ships a draft head — the serve-time replay skips the keys when the serving
    #    engine runs without MTP (load_weight's FTW branch).
    # 2) Side tables (qwen4_exp's ~47.7 GiB PLE n-gram table): copied verbatim below,
    #    after the metadata walk (they end in .safetensors and are loaded outside the
    #    dense stream via models.weight.side_table_files).
    _hf = cached_load_hf_config(model_path)
    _text = getattr(_hf, "text_config", None) or _hf
    _mtp_layers = int(getattr(_text, "mtp_num_hidden_layers", 0) or 0)
    if _mtp_layers > 0 and mtp_mode != "off":
        os.environ["FREETOKEN_ENABLE_MTP"] = "1"


    from freetoken.utils.progress import byte_bar, count_bar

    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    n_weight = n_bank = n_alpha = 0
    n_dspark_layers = None
    n_dspark_bank = n_dspark_alpha = 0

    # 1) dense weights (host tensors; load straight to CPU to avoid GPU pressure)
    _progress("dense", 0, 0)  # phase start; per-tensor cumulative bytes follow (total unknown)
    dense_bytes = 0
    _mtp_path = mtp_header if mtp_mode == "file" else ""
    for name, tensor in count_bar(load_weight(model_path, torch.device("cpu"),
                                              include_moe_experts=include_moe_experts,
                                              mtp_path=_mtp_path),
                                  "Converting dense weights"):
        writer.add_tensor(name, tensor, kind="weight")
        n_weight += 1
        dense_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes, 0)



    # 2) offload expert banks (post-repack) + alpha scales (slow path auto-picks parallel/serial)
    quant_format = None
    num_layers = None
    if offload:
        # every method-packed format streams each layer to its own FTW entry as it completes (via the sink); the GGUF provider reports through ExpertBanks.streamed whether it engaged the sink or materialized the whole bank set first
        from freetoken.moe.legacy_format import legacy_bank_names, legacy_format_for

        names = legacy_bank_names(legacy_format_for(method.kind, method.kernel.name)) if method is not None else {}
        sink = _ConvertSink(writer, names=names)
        banks = load_expert_banks(
            model_path, mc, method=method, device=dev, dtype=dtype, layer_sink=sink
        )
        quant_format = banks.quant_format
        if banks.streamed:
            sink.close()
            num_layers = sink.num_layers  # however many distinct layers the sink actually saw
            n_bank = sink.n_written
            assert num_layers > 0, (
                "provider reported streamed=True but the sink never fired -- the FTW "
                "would silently have no expert banks"
            )
            # Formats that fold their global scales (nvfp4 marlin/b12x) stream the weight
            # banks per layer but keep the alphas as flat [L*E] GPU vectors; write those as
            # flat reserved-name entries (same kind + names the materialize branch uses, so
            # the reader's reserved-name path reconstructs them identically).
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(banks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind="experts_bank")
                    n_alpha += 1
        else:
            # The on-disk format keeps one contiguous region per bank and the writer only
            # has whole-tensor add_tensor, so the per-layer sources reassemble into one
            # flat tensor (a per-bank host RAM spike during conversion).
            items = []
            names = legacy_bank_names(quant_format)
            for name, per_layer in banks.sources.items():
                if num_layers is None:
                    num_layers = len(per_layer)
                else:
                    assert len(per_layer) == num_layers, (name, len(per_layer), num_layers)
                items.append((names.get(name, name), torch.cat(per_layer, dim=0) if len(per_layer) > 1 else per_layer[0]))
            for an in ("gate_up_alpha", "down_alpha"):
                if getattr(banks, an, None) is not None:
                    items.append((an, getattr(banks, an)))
            total_bytes = sum(t.numel() * t.element_size() for _, t in items)
            bar = byte_bar(total_bytes, "Converting expert banks")
            done_bytes = 0
            _progress("experts", 0, total_bytes)
            for name, tensor in items:
                writer.add_tensor(name, tensor, kind="experts_bank")
                nbytes = tensor.numel() * tensor.element_size()
                bar.update(nbytes)
                done_bytes += nbytes
                _progress("experts", done_bytes, total_bytes)
                n_bank += name not in ("gate_up_alpha", "down_alpha")
                n_alpha += name in ("gate_up_alpha", "down_alpha")
            bar.close()

    # 3) Draft head, two shapes:
    #    * DSV4.1 DSpark: the draft's experts have their own count (E=128 vs 384) and ride a
    #      dedicated FTW bank kind/namespace -> explicit --include-dspark exports those banks.
    #    * Qwen3.8 (qwen4_exp): the MTP head is dense bf16 and the dense reader already yields
    #      it whenever the config declares mtp layers -- there are no separate draft banks, so
    #      --include-dspark is satisfied (tolerated) rather than an error.
    if include_dspark:
        from freetoken.models.weight import dspark_expert_method

        dspark_method = dspark_expert_method(model_path, mc)
        if dspark_method is not None:
            if not offload:
                raise SystemExit("--include-dspark requires --moe-backend offload (the draft experts are offloaded)")
            from freetoken.checkpoint.ftw import DSPARK_BANK_KIND
            from freetoken.models.weight import load_dspark_banks
            from freetoken.moe.legacy_format import legacy_bank_names, legacy_format_for

            num_stages = int(mc.dsv4_args.num_nextn_predict_layers)
            names = legacy_bank_names(legacy_format_for(dspark_method.kind, dspark_method.kernel.name))
            dsink = _ConvertSink(writer, "Converting DSpark draft banks", names=names,
                                 kind=DSPARK_BANK_KIND, progress=False)
            dbanks = load_dspark_banks(
                model_path, mc, method=dspark_method, num_stages=num_stages,
                device=dev, dtype=dtype, layer_sink=dsink,
            )
            dsink.close()
            # build_expert_banks streams through the sink for method-packed formats; the
            # draft is MXFP4, so this holds and the layer count is the draft's stage count
            assert dbanks.streamed, "DSpark draft banks must stream per layer"
            n_dspark_layers = dsink.num_layers
            n_dspark_bank = dsink.n_written
            assert n_dspark_layers == num_stages, (n_dspark_layers, num_stages)
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(dbanks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind=DSPARK_BANK_KIND)
                    n_dspark_alpha += 1
        elif not _ships_mtp(model_path):
            raise SystemExit("--include-dspark: the checkpoint ships no draft head (mtp.*)")
        # else: a dense MTP head (e.g. qwen4_exp) — already yielded by the dense reader.

    _progress("finalize")  # writing shard index + copying config/tokenizer
    copied = _copy_metadata(model_path, out_dir)
    # Model-specific side tables (qwen4_exp: the PLE n-gram table): .safetensors files
    # the model loads outside the dense stream — without them the FTW dir cannot serve.
    # ``ple_mode`` selects the stored dtype: ``bf16``/``fp8`` re-encode to that dtype
    # (extracting a table embedded in big dense shards), ``auto`` preserves the source.
    # A table already in the requested dtype rides verbatim (the loader raw-byte-copies it).
    from freetoken.models.weight import side_table_files

    import struct as _struct

    ple_stored = None
    for src in side_table_files(model_path):
        src_abs = os.path.abspath(src)
        if not os.path.isfile(src_abs):
            continue
        with open(src_abs, "rb") as fh:
            (hlen,) = _struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        ple_keys = [k for k in header if _PLE_KEY_INFIX in k]
        if not ple_keys:
            continue
        src_dtype = _ple_table_source_dtype(header, ple_keys)
        if _ple_rides_verbatim(header, ple_keys, ple_mode):
            dst = os.path.join(out_dir, os.path.basename(src_abs))
            shutil.copy2(src_abs, dst)
            copied.append(os.path.basename(src_abs))
            ple_stored = src_dtype
            continue
        stored = "fp8" if ple_mode == "fp8" else "bf16"
        _extract_ple_table(src_abs, out_dir, copied, out_dtype=stored)
        ple_stored = stored

    try:
        fingerprint = _source_fingerprint(model_path, mc, device=dev)
    except Exception:
        fingerprint = None

    index = writer.finalize({
        "source_model_path": os.path.abspath(model_path),
        "fingerprint": fingerprint,
        # quant_format records the actual on-disk bank layout (e.g. nvfp4_marlin vs
        # nvfp4_b12x): the suffix is a runtime backend pick (GPU capability / env), NOT in
        # config, and the stored bytes are physically repacked into it -- so it's kept and
        # read back at load (ftw.load_ftw_banks). dtype/moe_backend were dropped: each
        # tensor already carries its own dtype, and nothing reads a model-level backend.
        "quant_format": quant_format,
        # The reader takes num_layers from the model config (copied into this
        # checkpoint); recording it here too gives load_ftw_banks a cross-check that
        # the banks match the config they ship with. None for non-offload checkpoints.
        "expert_bank_num_layers": num_layers,
        # The DSpark draft's banks live under their own kind; their layer count (the
        # draft stage count) is the cross-check for load_dspark_banks. None when absent.
        DSPARK_BANK_NUM_LAYERS: n_dspark_layers,
        "counts": {
            "weight": n_weight,
            "experts_bank": n_bank + n_alpha,
            "dspark_experts_bank": n_dspark_bank + n_dspark_alpha,
        },
        "copied_metadata": copied,
        # Which optional side payloads this FTW carries. The loader reads these to decide
        # whether the Engram tables / DSpark draft come from the FTW or are skipped.
        "includes": {
            "engram": bool(include_engram),
            "dspark": bool(include_dspark),
            "mtp_mode": mtp_mode,
            "mtp_header": os.path.abspath(mtp_header) if mtp_mode == "file" else None,
            # Stored PLE n-gram table dtype ("fp8"/"bf16"); None when the model has no table.
            "ple": ple_stored,
            # Whether the vision tower's bf16 tensors ride the FTW dense stream (only when
            # the checkpoint has a vision_config and --vision on).
            "vision": bool(vision_mode == "on" and getattr(mc, "is_multimodal", False)),
        },
    })
    return index


__all__ = ["convert_checkpoint"]
