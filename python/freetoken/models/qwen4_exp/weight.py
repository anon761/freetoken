"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the PLE n-gram table (tens of GiB, fp8-e4m3 or bf16), its
  ``ngram_embedding.shard_<i>`` tensors concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.

Dropped: ``model.visual.*`` (served text-only). ``mtp.*`` (the speculative draft head)
is dropped unless ``FREETOKEN_ENABLE_MTP=1`` (``qwen4_args.mtp_enabled``). When enabled,
the dense ``mtp.*`` tensors yield under their raw names and the routed experts are
normalized to the resident bf16 stacked layout by :func:`iter_mtp_experts` -- whatever
the checkpoint ships (RadixArk's stacked bf16 ``experts.gate_up_proj`` or NVIDIA's
per-expert block-fp8 ``experts.<e>.{proj}_proj.weight`` + ``.weight_scale_inv``).
"""

from __future__ import annotations
from functools import lru_cache

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.wna16_banks import Wna16ExpertSourceSpec
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts: per-expert, un-fused. Matched against the RAW weight_map key in
# nvfp4_banks. The wrapper root is ``model.language_model.layers.``; a language-model-only export
# (arch ``Qwen4ExpForCausalLM``) drops the ``language_model.`` segment. The ``model.``/optional-
# wrapper anchor still excludes the MTP head's ``mtp.layers.N.mlp.experts.*``.
_EXPERT_KEY_PREFIX = (
    r"^model\.(?:language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# nvidia modelopt: weight | weight_scale | weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(_EXPERT_KEY_PREFIX + r"(?P<kind>weight|weight_scale|weight_scale_2)$"),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# llm-compressor (compressed-tensors): weight_packed | weight_scale | weight_global_scale. The
# global is the QUANT-side scale (~1e4 where modelopt stores ~1e-4), so the banks keep its
# reciprocal -- vLLM inverts it identically.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        _EXPERT_KEY_PREFIX + r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="Qwen3.8-Flash-Next NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_inv", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# "naive-quantized" modelopt checkpoints store dense projections as Float8 weights
# with a bf16 per-output-row .weight_scale sibling — dequantized at load (see iter_weights).
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str, keep_mtp: bool = False, include_vision: bool = False) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("model.visual.", "visual.")):
        if not include_vision:
            return None  # text-only serving: the tower is not built (see ModelConfig.is_multimodal)
        # The vision tower lives under ``vision_tower.`` in the model; the generic FTW
        # replay gate (VISION_KEY_PREFIXES) uses the same prefix.
        prefix = "model.visual." if raw_name.startswith("model.visual.") else "visual."
        return "vision_tower." + raw_name[len(prefix) :]
    if raw_name.startswith("mtp."):
        if not keep_mtp:
            return None
        # The routed experts (stacked OR per-expert) are normalized by
        # iter_mtp_experts, because the draft module holds a plain bf16 MoELayer and
        # the checkpoint's layout/quant varies by exporter. Everything else under
        # mtp.* (attention, shared expert, norms, fc_*) is dense and passes through.
        if ".mlp.experts." in raw_name:
            return None
        return raw_name
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            from freetoken.models.loader import cat_fused_rows

            return key, cat_fused_rows(key, rows)
    return None



def _dequant_pack_quantized(
    packed: torch.Tensor, scale: torch.Tensor, *, bits: int = 4, group_size: int = 32
) -> torch.Tensor:
    """compressed-tensors ``pack-quantized`` INT4 -> bf16.

    ``packed`` is ``[N, K/ (32//bits)]`` int32 (LSB-first, ``pack`` codes per word along
    K) and ``scale`` ``[N, K/group_size]`` bf16; symmetric, so the zero-point is the
    signed midpoint ``2**(bits-1)``. Returns ``[N, K]`` bf16."""
    pack = 32 // bits
    n = packed.shape[0]
    k = packed.shape[1] * pack
    shifts = torch.arange(pack, dtype=torch.int32) * bits
    codes = ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(n, k).to(torch.float32)
    codes = codes - float(1 << (bits - 1))
    scale = scale.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :k]
    return (codes * scale).to(torch.bfloat16)


def _read_mtp_int4_g32(reader, base: str) -> torch.Tensor:
    """One MTP expert projection in the compressed-tensors INT4 g32 layout -> ``[N, K]`` bf16.

    ``base`` is the checkpoint key stem, e.g. ``mtp.layers.0.mlp.experts.0.gate_proj``
    (the tensors are ``base.weight_packed`` / ``base.weight_scale`` / ``base.weight_shape``)."""
    shape = reader.get(base + ".weight_shape")
    packed = reader.get(base + ".weight_packed")
    scale = reader.get(base + ".weight_scale")
    w = _dequant_pack_quantized(packed, scale, bits=4, group_size=32)
    n, k = int(shape[0]), int(shape[1])
    return w[:n, :k].contiguous()


def iter_mtp_experts(model_path: str, config, tp) -> Iterator[tuple[str, torch.Tensor]]:
    """Normalize the MTP draft head's routed experts to the resident bf16 stacked layout.

    ``Qwen4ExpMTP`` holds a plain bf16 ``MoELayer``, so whatever the checkpoint ships is
    dequantized and stacked to ``gate_up_proj [E, 2I, H]`` / ``down_proj [E, H, I]``:
    RadixArk's already-stacked bf16 tensors, or NVIDIA's per-expert block-fp8
    ``experts.<e>.{proj}_proj.weight`` + ``.weight_scale_inv``. The layout AND the storage
    form are read from the checkpoint index -- never from the exporter name."""
    from freetoken.models.checkpoint_index import CheckpointIndex, TensorReader

    args = config.qwen4_args
    if not (args.mtp_enabled and args.mtp_num_layers > 0):
        return
    index = CheckpointIndex.load(model_path)
    reader = TensorReader(index)
    try:
        for layer in range(args.mtp_num_layers):
            prefix = f"mtp.layers.{layer}.mlp.experts"
            gate_up, down = _read_mtp_layer(index, reader, prefix, config)
            yield prefix + ".gate_up_proj", tp_shard_key(
                prefix + ".gate_up_proj", gate_up, config, tp
            )
            yield prefix + ".down_proj", tp_shard_key(
                prefix + ".down_proj", down, config, tp
            )
    finally:
        reader.close()


def _read_mtp_layer(index, reader, prefix: str, config) -> tuple[torch.Tensor, torch.Tensor]:
    """One MTP layer's experts, in the model's stacked bf16 ``(gate_up, down)`` form."""
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    gate_up_key = next(
        (k for k in (prefix + ".gate_up_proj", prefix + ".gate_up_proj.weight") if k in index),
        None,
    )
    down_key = next(
        (k for k in (prefix + ".down_proj", prefix + ".down_proj.weight") if k in index),
        None,
    )
    if gate_up_key is not None and down_key is not None:
        return _read_mtp_weight(index, reader, gate_up_key), _read_mtp_weight(index, reader, down_key)
    # compressed-tensors INT4 group-32 per-projection layout (weight_packed/weight_scale/weight_shape)
    if f"{prefix}.0.gate_proj.weight_packed" in index:
        gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
        down = torch.empty(E, H, I, dtype=torch.bfloat16)
        for e in range(E):
            gate_up[e, :I] = _read_mtp_int4_g32(reader, f"{prefix}.{e}.gate_proj")
            gate_up[e, I:] = _read_mtp_int4_g32(reader, f"{prefix}.{e}.up_proj")
            down[e] = _read_mtp_int4_g32(reader, f"{prefix}.{e}.down_proj")
        return gate_up, down
    # Per-expert layout: stack gate|up and down into the resident shape.
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if f"{prefix}.{e}.{proj}.weight" not in index:
                raise ValueError(f"MTP expert tensor missing: {prefix}.{e}.{proj}.weight")
    gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
    down = torch.empty(E, H, I, dtype=torch.bfloat16)
    for e in range(E):
        gate_up[e, :I] = _read_mtp_weight(index, reader, f"{prefix}.{e}.gate_proj.weight")
        gate_up[e, I:] = _read_mtp_weight(index, reader, f"{prefix}.{e}.up_proj.weight")
        down[e] = _read_mtp_weight(index, reader, f"{prefix}.{e}.down_proj.weight")
    return gate_up, down


def _read_mtp_weight(index, reader, key: str) -> torch.Tensor:
    """Read one MTP expert weight, dequantizing any block/per-row fp8 dialect to bf16."""
    from freetoken.models.checkpoint_index import dequantize

    if key.endswith(".weight"):
        return dequantize(index, key[: -len(".weight")], reader.get)
    # Stacked residents store the tensor under the bare key (no ".weight"); the
    # RadixArk draft is bf16 there, so no companion is involved.
    return reader.get(key)


def tp_shard_key(name: str, w: torch.Tensor, config, tp) -> torch.Tensor:
    """This rank's view of a full resident tensor, by state-dict key. Identity at TP=1.

    Head-structured axes (attention qkv, GDN in_proj/conv) slice per segment with
    ``div_even`` widths; row-parallel outputs (o_proj/out_proj/shared down) slice
    their input columns; the shared expert's fused gate|up slices both halves on
    the intermediate axis; embed/lm_head slice vocab rows with zero-pad (mirrors
    VocabParallelEmbedding). HC, PLE, norms and the router stay replicated."""
    if tp.size == 1:
        return w
    r = tp.rank

    def rows(w: torch.Tensor, n_full: int) -> torch.Tensor:
        # contiguous(): a view would keep the full base tensor resident (see glm5).
        loc = div_even(n_full, tp.size)
        return w[r * loc : (r + 1) * loc].contiguous()

    def cols(w: torch.Tensor, n_full: int) -> torch.Tensor:
        loc = div_even(n_full, tp.size)
        return w[:, r * loc : (r + 1) * loc].contiguous()

    def segments(fused: torch.Tensor, full_sizes: list[int]) -> torch.Tensor:
        out = []
        off = 0
        for n_full in full_sizes:
            loc = div_even(n_full, tp.size)
            out.append(fused[off + r * loc : off + (r + 1) * loc])
            off += n_full
        return torch.cat(out, dim=0)

    hd, qo, kv = config.head_dim, config.num_qo_heads, config.num_kv_heads
    g = config.linear_attention_group()
    g_k, g_v = g.num_key_heads, g.num_value_heads
    g_dk, g_dv = g.key_head_dim, g.value_head_dim
    shared_i = config.shared_expert_intermediate_size

    if name.endswith(".self_attn.qkv_proj.weight"):
        # [2*qo*hd | kv*hd | kv*hd] — slice each segment on the head axis
        return segments(w, [2 * qo * hd, kv * hd, kv * hd])
    if name.endswith(".self_attn.o_proj.weight"):
        return cols(w, qo * hd)
    if name.endswith(".linear_attn.in_proj.weight"):
        # qkv = [q | k | v] = [k*dk | k*dk | v*dv], dann z = v*dv, b = v, a = v
        return segments(w, [g_k * g_dk, g_k * g_dk, g_v * g_dv,
                            g_v * g_dv, g_v, g_v])
    if name.endswith(".linear_attn.conv1d.weight"):
        # channels follow the rank's head slice, in loader q|k|v order
        return segments(w, [g_k * g_dk, g_k * g_dk, g_v * g_dv])
    if name.endswith(".linear_attn.out_proj.weight"):
        return cols(w, g_v * g_dv)
    if name.endswith(".linear_attn.dt_bias") or name.endswith(".linear_attn.A_log"):
        # per-value-head gate params: this rank's head slice
        return rows(w, g_v)
    if name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
        # gate rows [0, I) + up rows [I, 2I): slice both halves on I
        I_loc = div_even(shared_i, tp.size)
        lo, hi = r * I_loc, (r + 1) * I_loc
        return torch.cat([w[lo:hi], w[shared_i + lo : shared_i + hi]], dim=0)
    if name.endswith(".mlp.shared_expert.down_proj.weight"):
        return cols(w, shared_i)
    if name.endswith(".mlp.experts.gate_up_proj") and name.startswith("mtp."):
        # MTP's stacked BF16 experts [E, 2I, H]: slice each expert's gate|up halves on I.
        moe_i = config.moe_intermediate_size
        i_loc = div_even(moe_i, tp.size)
        lo, hi = r * i_loc, (r + 1) * i_loc
        return torch.cat([w[:, lo:hi], w[:, moe_i + lo : moe_i + hi]], dim=1).contiguous()
    if name.endswith(".mlp.experts.down_proj") and name.startswith("mtp."):
        # [E, H, I]: slice each expert's input columns on I.
        i_loc = div_even(config.moe_intermediate_size, tp.size)
        return w[:, :, r * i_loc : (r + 1) * i_loc].contiguous()
    if name == "lm_head.weight" or name.endswith(".embed_tokens.weight"):
        # vocab-parallel rows, zero-padded on the last rank
        vocab = w.shape[0]
        n_tp = div_ceil(vocab, tp.size)
        start = r * n_tp
        out = w.new_zeros(n_tp, *w.shape[1:])
        piece = w[start : start + n_tp]
        out[: piece.shape[0]] = piece
        return out
    return w


@lru_cache(maxsize=8)
def _ftw_shard_config(model_path: str):
    """Parsed model config for FTW TP sharding, cached per checkpoint. Rebuilding it for
    every one of the ~1000 dense tensors cost ~130 ms each (~150 s for the dense shard);
    the config is immutable here so one parse per load is enough."""
    return parse_config(cached_load_hf_config(model_path))


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """TP-shard one FTW-replayed dense tensor (see tp_shard_key). The FTW format
    stores dense weights un-sharded (TP-agnostic); the loader applies the same
    key rules as the HF-path iter_weights."""
    return tp_shard_key(name, w, _ftw_shard_config(model_path), tp)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_mtp: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: every release's skip
    list (modelopt ``ignore``, fp8 ``modules_to_not_convert``) covers everything except those experts,
    so attention, GDN, HC, PLE, the shared expert and lm_head are all plain bf16 (the n-gram hash
    constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from the offload cache's expert reader.

    TP: the resident state shards across ranks (per-head attention qkv, row-parallel
    o_proj/out_proj, TP-local GDN heads/conv/gates, column/row-parallel shared expert,
    vocab-parallel embed + lm_head); HC, PLE, norms and the router stay replicated. The
    PLE n-gram table is replicated (per-token lookups are rank-invariant).
    """
    config = parse_config(cached_load_hf_config(model_path))
    tp = get_tp_info()
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}

    # Generic storage inventory: lets an fp8 dense weight find its scale companion
    # under ANY dialect suffix (.weight_scale per-row, .weight_scale_inv block) and
    # across shards. Headers only, cheap.
    from freetoken.models.checkpoint_index import (
        CheckpointIndex,
        StorageKind,
        TensorReader,
        dequantize,
        detect_storage,
    )

    index = CheckpointIndex.load(model_path)
    reader = TensorReader(index, device=str(device))
    try:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    # scale siblings are consumed WITH their fp8 weight below; _rename
                    # drops them (and the nvfp4 expert scales ride the banks loader /
                    # the PLE scale reads its shard directly).
                    name = _rename(
                        raw_name,
                        keep_mtp=config.qwen4_args.mtp_enabled,
                        include_vision=config.is_multimodal,
                    )
                    if name is None:
                        continue
                    tensor = f.get_tensor(raw_name)
                    # "naive-quantized" checkpoints store dense projections (GDN in/out_proj,
                    # QSA q/k/v/o, shared expert) as Float8 + a per-row or per-block scale.
                    # torch.cat refuses any Float8 involvement, so dequantize to bf16 BEFORE
                    # the fusions — the resident dense set is bf16 in every variant.
                    if tensor.dtype in _FP8_DTYPES:
                        stem = raw_name[: -len(".weight")] if raw_name.endswith(".weight") else raw_name
                        if detect_storage(index, stem) is StorageKind.BF16:
                            raise ValueError(
                                f"fp8 weight has no .weight_scale/.weight_scale_inv companion: {stem!r}"
                            )
                        tensor = dequantize(index, stem, reader.get)
                    fused = _try_fuse(name, tensor, fuse_buf)
                    if fused is not None:
                        if fused != ():  # () means buffered, not yet complete
                            yield fused[0], tp_shard_key(fused[0], fused[1], config, tp)
                        continue
                    yield name, tp_shard_key(name, tensor, config, tp)
    finally:
        reader.close()

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"

    # MTP draft head: normalized to the resident bf16 stacked experts regardless of the
    # checkpoint's exporter layout (stacked bf16 vs per-expert block-fp8).
    # ``include_mtp=False`` (the --mtp file base stream) skips this: the head comes from
    # the standalone artifact, and the base checkpoint does not ship it here.
    if include_mtp:
        yield from iter_mtp_experts(model_path, config, tp)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale.

    An fp8 table carries its checkpoint ``weight_scale``; a bf16 table has no scale (1.0)."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` view of the bank (fp8-e4m3 or bf16)."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"
# safetensors dtype -> (torch dtype, element bytes) for PLE table shards: fp8 or unquantized.
_PLE_SRC_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def _has_ple_shards(folder: str) -> bool:
    for path in _ple_table_files(folder):
        try:
            header, _ = _safetensors_header(path)
        except (OSError, ValueError, struct.error):
            continue
        if any(_PLE_SHARD_RE.search(k) for k in header):
            return True
    return False


def ple_source_folder(model_path: str) -> str:
    """Folder to read the PLE n-gram table from.

    An FTW is self-contained and normally carries its own ``model-plefp8-*`` side
    tables; an older/partial FTW may not. In that case fall back to the source
    checkpoint recorded in the FTW index, so the table is read from the raw shards
    instead of failing the serve."""
    folder = download_hf_weight(model_path)
    from freetoken.checkpoint.ftw import FTWReader, is_ftw_checkpoint

    if not is_ftw_checkpoint(folder) or _has_ple_shards(folder):
        return folder
    try:
        source = FTWReader(folder).meta("source_model_path")
    except (OSError, ValueError):
        source = None
    if isinstance(source, str) and source and os.path.isdir(source):
        print(
            f"freetoken: FTW {folder!r} has no PLE side tables — reading them from the "
            f"source checkpoint {source!r} (re-run `ft checkpoint` to embed them)",
            flush=True,
        )
        return download_hf_weight(source)
    return folder


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int | None = None, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*``/``model-plebf16-*`` shards in header
    (lexicographic) order, so the bank is filled shard by shard at ``shard_index *
    rows_per_shard``. An fp8 table is raw-byte-copied (O_DIRECT) into an fp8 bank; a bf16 table
    is read as tensors into a bf16 bank (no quantization). Each fp8 read is O_DIRECT: the table
    is tens of GiB and must not also sit in the page cache while the bank holds the same bytes.

    ``workers`` (default FREETOKEN_PLE_LOAD_WORKERS) is the outstanding-read depth per shard;
    the ~48 GiB table is read shard-by-shard, since concentrating workers on one file beats
    spreading them across files on this pool."""
    if workers is None:
        from freetoken.env import ENV

        workers = int(ENV.PLE_LOAD_WORKERS.value)
    folder = ple_source_folder(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    bf16_parts: dict[int, str] = {}  # shard index -> key (tensor-read path)
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] == _PLE_ST_DTYPE:
                shape = meta["shape"]
                if rows and tuple(shape) != (rows, cols):
                    raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
                rows, cols = shape
                begin, end = meta["data_offsets"]
                parts[int(match.group("shard"))] = (path, base + begin, end - begin)
            elif meta["dtype"] in ("BF16", "F32"):
                # llm-compressor re-exports ship the n-gram table UNQUANTIZED (bf16,
                # embedded in the big dense shards) — dequantized is what it is: read
                # as a tensor and quantize into the fp8 bank at load (see fill below).
                shape = meta["shape"]
                if rows and tuple(shape) != (rows, cols):
                    raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
                rows, cols = shape
                bf16_parts[int(match.group("shard"))] = (key, path)
            else:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")

    if bf16_parts and parts:
        raise ValueError(
            "PLE table mixes fp8 and bf16 shards — unsupported layout"
        )
    expected = int(qwen4_args.split_ngram_parts)
    all_parts = {**parts, **bf16_parts}
    if sorted(all_parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(all_parts)}: {sorted(all_parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")

    if bf16_parts:
        # Unquantized table: serve it bf16 byte-for-byte (an F32 source is narrowed).
        # There is no scale — the gather kernel multiplies by 1.0.
        bank = HostBank((expected * rows, cols), torch.bfloat16)
        scale = torch.tensor(1.0, dtype=torch.bfloat16)
        shard_bytes = rows * cols * 2
        bar = byte_bar(expected * shard_bytes, "Loading PLE table", monitor=True)
        try:
            for shard in range(expected):
                key, path = bf16_parts[shard]
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    t = f.get_tensor(key).to(torch.bfloat16)
                bank.tensor[shard * rows : (shard + 1) * rows] = t
                bar.update(shard_bytes)
        finally:
            bar.close()
    else:
        bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
        shard_bytes = rows * cols
        if scale is None:
            raise ValueError("fp8 PLE table has no weight_scale")
        bar = byte_bar(expected * shard_bytes, "Loading PLE table", monitor=True)
        try:
            buf = bank.memoryview()
            for shard in range(expected):
                path, offset, nbytes = parts[shard]
                assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
                read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                                dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
                bar.update(nbytes)
        finally:
            bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC


# AutoRound / AutoGPTQ W4A16 (weight-only INT4) experts: same checkpoint names, the
# AutoGPTQ qweight/qzeros/scales layout instead of modelopt NVFP4.
_WNA16_EXPERT_KEY_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>qweight|qzeros|scales)$"
)
_WNA16_SOURCE_SPEC = Wna16ExpertSourceSpec(
    key_pattern=_WNA16_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next WNA16 experts",
    group_size=128,
)


def wna16_expert_spec(model_path: str, config):
    return _WNA16_SOURCE_SPEC


# --- Standalone MTP draft-head artifact (--mtp file) -------------------------

# Geometry fields a draft head MUST share with its base model; anything else
# silently produces garbage tokens. The panel uses the same list for its
# compatibility assist, so the UI and the engine can never disagree.
_MTP_COMPAT_FIELDS = (
    "hidden_size",
    "hc_count",
    "hc_lowrank",
    "head_dim",
    "num_qo_heads",
    "num_kv_heads",
    "num_experts",
    "moe_intermediate_size",
    "vocab_size",
    "linear_num_heads",
    "linear_head_dim",
)


def mtp_compat_reasons(main_config, artifact_config) -> list[str]:
    """Human-readable list of geometry mismatches between base model and draft
    artifact (empty = compatible). Reads the parsed family configs."""
    reasons = []
    a = main_config.qwen4_args
    b = artifact_config.qwen4_args
    for field in _MTP_COMPAT_FIELDS:
        av, bv = getattr(a, field, None), getattr(b, field, None)
        if av is None or bv is None:
            continue  # field unknown on one side: not a mismatch we can prove
        if av != bv:
            reasons.append(f"{field}: Modell {av} vs. Artifact {bv}")
    return reasons


def validate_mtp_compat(main_config, artifact_config, mtp_path: str) -> None:
    reasons = mtp_compat_reasons(main_config, artifact_config)
    if reasons:
        raise ValueError(
            "MTP-Artefakt passt nicht zum Modell (" + mtp_path + "): " + "; ".join(reasons)
        )


def iter_external_mtp_weights(
    mtp_path: str, device, config, tp=None
) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Yield the draft head's ``mtp.*`` tensors from a standalone artifact.

    `mtp_path` is either the artifact directory (all ``*.safetensors`` inside) or a
    single ``.safetensors`` shard. The artifact carries the family config.json (for
    the geometry fingerprint) next to the weights — keys either ``mtp.``-prefixed (a
    full-checkpoint export) or unprefixed (a draft-only export; they are
    prefixed here). Each tensor is TP-sharded with the same key rules the
    in-checkpoint draft would go through."""
    import glob
    import os

    import safetensors.torch

    if os.path.isfile(mtp_path):
        shards = [mtp_path]
    elif os.path.isdir(mtp_path):
        shards = sorted(glob.glob(os.path.join(mtp_path, "*.safetensors")))
    else:
        raise ValueError(f"MTP-Artefakt nicht gefunden: {mtp_path}")
    if not shards:
        raise ValueError(f"MTP-Artefakt {mtp_path} enthaelt keine safetensors-Dateien")
    from freetoken.distributed import try_get_tp_info

    tp = tp or try_get_tp_info()

    # Draft-only exports store the projections UNFUSED (q/k/v separate, HC
    # down+inject separate, shared-expert gate/up separate) and the routed
    # experts per-projection (switch_mlp.*). The in-checkpoint reader applies
    # _FUSIONS at load; the external path must do the same, else
    # load_state_dict misses the fused keys (qkv_proj, *_block_inject,
    # shared_expert.gate_up_proj) and the stacked experts.
    # A full-model-shaped draft export (e.g. albucino's mtp-int4-g32) ships a stub
    # ``layers.0.*`` NEXT TO the real ``mtp.*`` block; only ``mtp.*`` is the draft, and
    # prefixing the stub would collide with the block. A draft-only export has no ``mtp.``
    # keys, so its keys are prefixed instead.
    keys_by_shard: dict[str, list[str]] = {}
    has_mtp = False
    for path in shards:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        keys_by_shard[path] = keys
        has_mtp = has_mtp or any(k.startswith("mtp.") for k in keys)

    plain: dict[str, "torch.Tensor"] = {}
    experts: dict[str, dict[str, "torch.Tensor"]] = {}
    int4_experts: dict[str, dict[int, dict[str, dict[str, "torch.Tensor"]]]] = {}
    int4_re = re.compile(
        r"^mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
        r"(weight_packed|weight_scale)$"
    )
    for path, keys in keys_by_shard.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                if has_mtp and not key.startswith("mtp."):
                    continue  # full-model-shaped artifact: non-mtp.* is a stub
                name = key if key.startswith("mtp.") else "mtp." + key
                if name.endswith(".weight_shape"):
                    continue
                match = int4_re.match(name)
                if match is not None:
                    layer, e, proj, kind = match.group(1), int(match.group(2)), match.group(3), match.group(4)
                    int4_experts.setdefault(layer, {}).setdefault(e, {}).setdefault(proj, {})[kind] = f.get_tensor(key)
                    continue
                tensor = f.get_tensor(key)
                if ".mlp.switch_mlp." in name:
                    layer = name.split(".layers.")[1].split(".", 1)[0]
                    part = name.split(".mlp.switch_mlp.")[1]
                    experts.setdefault(layer, {})[part] = tensor
                    continue
                plain[name] = tensor

    # compressed-tensors INT4 group-32 experts: dequant + stack into the fused MoELayer.
    # Geometry is read only when such experts exist, so a plain (bf16) draft artifact
    # needs no numeric fields on the config (mirrors tp_shard_key's TP=1 short-circuit).
    if int4_experts:
        E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
        for layer, by_expert in sorted(int4_experts.items()):
            gate_up = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
            down = torch.empty(E, H, I, dtype=torch.bfloat16)
            for e, projs in by_expert.items():
                def _deq(proj: str) -> "torch.Tensor":
                    p = projs[proj]
                    return _dequant_pack_quantized(p["weight_packed"], p["weight_scale"], bits=4, group_size=32)
                gate_up[e, :I] = _deq("gate_proj")[:I, :H]
                gate_up[e, I:] = _deq("up_proj")[:I, :H]
                down[e] = _deq("down_proj")[:H, :I]
            base = f"mtp.layers.{layer}.mlp.experts"
            for key, tensor in ((base + ".gate_up_proj", gate_up), (base + ".down_proj", down)):
                yield key, tp_shard_key(key, tensor, config, tp).to(device)

    # routed experts: stack the per-projection export into the fused MoELayer
    # layout (keys without ".weight", qwen3_5_moe resident-expert convention).
    for layer, parts in sorted(experts.items()):
        base = f"mtp.layers.{layer}.mlp.experts"
        gate_up = torch.cat([parts["gate_proj.weight"], parts["up_proj.weight"]], dim=1)
        for key, tensor in (
            (base + ".gate_up_proj", gate_up),
            (base + ".down_proj", parts.pop("down_proj.weight")),
        ):
            yield key, tp_shard_key(key, tensor, config, tp).to(device)

    fuse_buf: dict[str, dict[int, "torch.Tensor"]] = {}
    for name, tensor in plain.items():
        fused = _try_fuse(name, tensor, fuse_buf)
        if fused is None:
            yield name, tp_shard_key(name, tensor, config, tp).to(device)
        elif fused != ():
            yield fused[0], tp_shard_key(fused[0], fused[1], config, tp).to(device)
    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"
def side_table_files(model_path: str) -> list[str]:
    """The PLE n-gram table shards (``model-plefp8-*.safetensors``): the model loads them
    OUTSIDE the dense weight stream (:func:`load_ple_table`), so an FTW conversion must
    carry them over verbatim — the generic metadata walk skips every ``.safetensors``."""
    return _ple_table_files(download_hf_weight(model_path))


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
