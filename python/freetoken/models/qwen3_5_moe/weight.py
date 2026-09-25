from __future__ import annotations
from functools import lru_cache

import re
from typing import Iterator

import safetensors
import torch
from freetoken.layers.quantization import QuantKind
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.models.loader import (
    CT_SCALE_SUFFIXES,
    ShardReader,
    ct_bf16_fuse,
    ct_fp8_fuse,
    ct_nvfp4_fuse,
    iter_weight_files,
    nvfp4_parts_ct,
)
from freetoken.models.fp8_block_banks import (
    iter_fp8_block_expert_pieces,
    moe_dims as _moe_dims,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.utils import cached_load_hf_config, div_ceil, div_even
from tqdm import tqdm

from freetoken.models.config import detect_compressed_tensors_nvfp4 as _compressed_tensors_nvfp4

from .config import parse_config

# Expert weights are stored pre-fused per layer: experts.gate_up_proj / experts.down_proj.
_PACKED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)

# NVFP4 routed experts (nvidia modelopt checkpoint): per-expert, un-fused, under the raw
# ``model.language_model.layers.N.mlp.experts.E.{proj}`` key. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP
# head's ``mtp.layers.N.mlp.experts.*`` tensors (served text-only, dropped).
_NVFP4_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.5 NVFP4 experts",
)
# Suffixes of the per-tensor modelopt quant scales; consumed alongside their ``.weight``,
# never yielded on their own.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# Gemma-style (1+weight) RMSNorm weights. Excludes GDN gated norm (linear_attn.norm),
# which is a standard weight*x norm.
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
# shared-expert gate/up merge -> shared_expert.gate_up_proj
_SHARED_GATE = ".mlp.shared_expert.gate_proj.weight"
_SHARED_UP = ".mlp.shared_expert.up_proj.weight"

# Fused projections: concat checkpoint matrices in this exact order to match the model's
# LinearColParallelMerged split. fused_suffix -> ordered parts.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
    # Dense (non-MoE) layer MLP: merge gate|up -> gate_up_proj. Only fires on a bare
    # ``.mlp.gate_proj``; ``.mlp.shared_expert.gate_proj`` (MoE) does not end with this.
    ".mlp.gate_up_proj.weight": (
        ".mlp.gate_proj.weight", ".mlp.up_proj.weight",
    ),
}


def _dequant_fp8_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Weight-only FP8 -> bf16 (per-tensor static scale). Activations stay bf16 (W8A16),
    which is at least as precise as the checkpoint's intended W8A8."""
    return weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16)


def _dequant_nvfp4_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    """Dense NVFP4 -> bf16 (W4A16): ``fp4 * block_scale * global_scale``. ``weight`` is
    [O, IN//2] uint8, ``weight_scale`` [O, IN//16] fp8-e4m3, ``weight_scale_2`` the per-tensor
    global scalar (broadcast to per-row, matching the offload-cache dequant kernel)."""
    # The dequant kernel is GPU-only; the checkpoint-conversion path loads dense weights on
    # CPU, so run on CUDA and return on the caller's device (no-op when already on GPU).
    orig_device = weight.device
    if orig_device.type != "cuda":
        dev = torch.device("cuda")
        weight, weight_scale, weight_scale_2 = (
            weight.to(dev), weight_scale.to(dev), weight_scale_2.to(dev)
        )
    out_features = weight.shape[0]
    global_scale = weight_scale_2.reshape(1).to(torch.float16).expand(out_features).contiguous()
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(),
        weight_scale.unsqueeze(0).contiguous(),
        global_scale.unsqueeze(0),
        slots,
        dtype=torch.bfloat16,
    )[0]
    return out.to(orig_device)


def _load_maybe_quantized(f, raw_name: str, keyset: set[str]) -> torch.Tensor:
    """Load ``raw_name``; if it is a quantized ``.weight`` with sibling modelopt scales in
    the same shard, dequantize to bf16 (NVFP4 if ``weight_scale_2`` present, else FP8).
    Plain bf16 weights pass through unchanged."""
    tensor = f.get_tensor(raw_name)
    if not raw_name.endswith(".weight"):
        return tensor
    base = raw_name[: -len(".weight")]
    if base + ".weight_scale_2" in keyset:  # NVFP4 (two-level block scale)
        return _dequant_nvfp4_weight(
            tensor, f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_scale_2")
        )
    if base + ".weight_scale" in keyset:  # FP8 (per-tensor scale)
        return _dequant_fp8_weight(tensor, f.get_tensor(base + ".weight_scale"))
    return tensor


def _rename(raw_name: str, keep_mtp: bool = False) -> str | None:
    """HF key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        # The MTP draft head is served text-only under its own ``mtp.*`` keys; keep it
        # only when this serve builds the head (FREETOKEN_ENABLE_MTP).
        return raw_name if keep_mtp else None
    if raw_name.startswith(("model.visual.", "visual.")):
        return None
    # ModelOpt FP8 KV-cache static scales (full-attention layers only). FreeToken keeps the
    # KV cache in the engine's native precision (>= the checkpoint's quantized KV), so these
    # per-tensor q/k/v scales are unused -- drop them rather than fail as unexpected keys.
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    name = raw_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    return name


def _is_gemma_norm(name: str) -> bool:
    # The MTP head's own norms use the same Gemma (1 + w) convention as the main stack.
    if name in ("model.norm.weight", "mtp.norm.weight") or name.startswith("mtp.pre_fc_norm"):
        return True
    return name.endswith(_GEMMA_NORM_SUFFIXES)


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """buffer a fusion part; return merged ``(name, tensor)`` once all parts arrive,
    ``()`` while incomplete, ``None`` if not a fusion part."""
    for fused_suffix, parts in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if name.endswith(part):
                key = name[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


# Quant suffixes a fused/standalone state key can carry; stripped to find the module rule.
_QUANT_SUFFIXES = (
    ".weight_scale_inv", ".weight_scale_2", ".weight_packed", ".weight_global",
    ".weight_scale", ".input_scale", ".weight",
)


def _split_quant_suffix(name: str) -> tuple[str, str]:
    for suf in _QUANT_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def tp_shard_key(name: str, w: torch.Tensor, config, tp) -> torch.Tensor:
    """This rank's view of a full resident tensor, by state-dict key. Identity at TP=1.

    Head-structured axes (attention qkv, GDN in_proj/conv) slice per segment with
    ``div_even`` widths; row-parallel outputs (o_proj/out_proj/*down_proj) slice their
    input columns; the shared/dense expert's fused gate|up slices both halves on the
    intermediate axis; stacked bf16 experts slice per expert; embed/lm_head slice vocab
    rows with zero-pad (mirrors VocabParallelEmbedding). The packing/block factor on a
    sliced axis (NVFP4 codes //2, block scales //16, fp8-block scales //128) is derived
    from the tensor shape, so the same rule shards ``.weight``, ``.weight_packed``,
    ``.weight_scale`` and ``.weight_scale_inv`` alike. Norms, the router, ``A_log``/
    ``dt_bias`` and per-output-row globals stay replicated where they must.
    """
    if tp.size == 1:
        return w
    if w.ndim == 0:
        return w  # per-module scalar (input_scale): rank-invariant
    r = tp.rank

    def _factor(axis: int, full: int) -> int:
        got = w.shape[axis]
        if got <= 0 or full % got:
            raise ValueError(
                f"cannot TP-shard {name!r}: axis {axis} size {got} does not divide the full {full}"
            )
        return full // got

    def _check(loc: int, factor: int) -> None:
        if loc % factor:
            raise ValueError(
                f"cannot TP-shard {name!r}: local slice {loc} is not {factor}-aligned"
            )

    def _rows(t: torch.Tensor, full: int, factor: int) -> torch.Tensor:
        loc = div_even(full, tp.size)
        _check(loc, factor)
        return t[r * loc // factor : (r + 1) * loc // factor].contiguous()

    def _cols(t: torch.Tensor, full: int) -> torch.Tensor:
        if t.ndim == 1:
            return t  # per-output-row vector: replicated on a row-parallel input slice
        got = t.shape[-1]
        if got <= 0 or full % got:
            raise ValueError(
                f"cannot TP-shard {name!r}: axis -1 size {got} does not divide the full {full}"
            )
        factor = full // got
        loc = div_even(full, tp.size)
        _check(loc, factor)
        return t[..., r * loc // factor : (r + 1) * loc // factor].contiguous()

    def _segments(t: torch.Tensor, full_sizes: list[int], factor: int) -> torch.Tensor:
        out: list[torch.Tensor] = []
        off = 0
        for n in full_sizes:
            loc = div_even(n, tp.size)
            _check(loc, factor)
            _check(off, factor)
            out.append(t[off // factor + r * loc // factor : off // factor + (r + 1) * loc // factor])
            off += n
        return torch.cat(out, dim=0)

    base, _suf = _split_quant_suffix(name)
    hd, qo, kv = config.head_dim, config.num_qo_heads, config.num_kv_heads
    grp = config.linear_attention_group()
    g_k, g_v = grp.num_key_heads, grp.num_value_heads
    g_dk, g_dv = grp.key_head_dim, grp.value_head_dim

    if base.endswith(".self_attn.qkv_proj"):
        sizes = [2 * qo * hd, kv * hd, kv * hd]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".self_attn.o_proj"):
        return _cols(w, qo * hd)
    if base.endswith(".linear_attn.in_proj_qkvz"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv, g_v * g_dv]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.in_proj_ba"):
        return _segments(w, [g_v, g_v], _factor(0, 2 * g_v))
    if base.endswith(".linear_attn.in_proj"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv, g_v * g_dv, g_v, g_v]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.conv1d"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.out_proj"):
        return _cols(w, g_v * g_dv)
    if base.endswith(".linear_attn.dt_bias") or base.endswith(".linear_attn.A_log"):
        return _rows(w, g_v, 1)
    if base.endswith(".mlp.shared_expert.gate_up_proj"):
        i = config.shared_expert_intermediate_size
        return _segments(w, [i, i], _factor(0, 2 * i))
    if base.endswith(".mlp.shared_expert.down_proj"):
        i = config.shared_expert_intermediate_size
        return _cols(w, i)
    if base.endswith(".mlp.gate_up_proj"):
        i = config.intermediate_size
        return _segments(w, [i, i], _factor(0, 2 * i))
    if base.endswith(".mlp.down_proj"):
        i = config.intermediate_size
        return _cols(w, i)
    if base.endswith(".mlp.experts.gate_up_proj"):
        i, loc = config.moe_intermediate_size, div_even(config.moe_intermediate_size, tp.size)
        lo, hi = r * loc, (r + 1) * loc
        return torch.cat([w[:, lo:hi], w[:, i + lo : i + hi]], dim=1).contiguous()
    if base.endswith(".mlp.experts.down_proj"):
        i, loc = config.moe_intermediate_size, div_even(config.moe_intermediate_size, tp.size)
        return w[:, :, r * loc : (r + 1) * loc].contiguous()
    if base == "lm_head" or base.endswith(".embed_tokens"):
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
    """Parsed model config for FTW TP sharding, cached per checkpoint (one parse per load;
    rebuilding it per dense tensor dominated the FTW load at TP>1)."""
    return parse_config(cached_load_hf_config(model_path))


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """TP-shard one FTW-replayed dense tensor (see tp_shard_key). The FTW format stores
    dense weights un-sharded (TP-agnostic); the loader applies the same key rules as the
    HF-path iter_weights."""
    return tp_shard_key(name, w, _ftw_shard_config(model_path), tp)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    if _compressed_tensors_nvfp4(hf_config):
        # Dense compressed-tensors NVFP4 (e.g. Qwen3.6-27B): attn (q/k/v/o, GDN out_proj) +
        # dense MLP are W4A16 NVFP4; GDN in_proj_*, lm_head, norms bf16.
        yield from _iter_weights_compressed_tensors(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
            nvfp4=config.dense_quant == "nvfp4",
        )
        return
    if config.expert_quant == "fp8_block":
        # Dense (attn/GDN/shared-expert) weights are always block-fp8; routed experts are
        # yielded here only for the resident path (include_moe_experts=True). Under offload
        # they are excluded and loaded from expert pieces instead.
        yield from _iter_weights_fp8(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
        )
        return
    if config.attn_quant == "fp8_pertensor":
        # modelopt MIXED_PRECISION: dense attn/GDN projections kept per-tensor FP8 (fp8
        # weight + per-row scale, W8A16 kernel); NVFP4 dense (shared_expert/lm_head) kept
        # native FP4 (W4A16) when dense_quant=="nvfp4", else dequantized to bf16; routed
        # NVFP4 experts excluded (offload cache).
        yield from _iter_weights_attn_fp8(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
            dense_nvfp4=config.dense_quant == "nvfp4",
            lmhead_nvfp4=config.lm_head_quant == "nvfp4",
        )
        return
    tp_info = get_tp_info()

    def _shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tp_shard_key(name, tensor, config, tp_info)

    # Pure-NVFP4 checkpoint (bf16 attn): the dense MLP projections (shared_expert) are still
    # stored as packed FP4 -- keep them native (W4A16) when dense_quant=="nvfp4" rather than
    # dequantizing to bf16. lm_head here is bf16 (pure NVFP4 doesn't quantize it).
    dense_nvfp4 = config.dense_quant == "nvfp4"
    lmhead_nvfp4 = config.lm_head_quant == "nvfp4"
    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    nvfp4_shared_buf: dict[str, dict[str, tuple]] = {}
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                # Per-expert NVFP4 tensors go to the offload cache (expert pieces),
                # not the dense pass. bf16-base stacked experts (experts.gate_up_proj) have no
                # ``.mlp.experts.<int>.`` so they are unaffected and still hit _PACKED_EXPERT.
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue
                # Standalone modelopt scales are consumed with their .weight, never yielded.
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue

                name = _rename(raw_name, config.mtp_enabled)
                if name is None:
                    continue

                is_expert = _PACKED_EXPERT_PATTERN.match(name) is not None
                if is_expert and not include_moe_experts:
                    continue
                if not is_expert and not include_non_moe:
                    continue

                # NVFP4 dense projections kept native (W4A16) where the model expects them
                # (shared_expert); everything else dequantizes to bf16 below as before.
                if (dense_nvfp4 or lmhead_nvfp4) and name.endswith(".weight") \
                        and raw_name[: -len(".weight")] + ".weight_scale_2" in keyset:
                    emit = _dense_nvfp4_emit(
                        f, name[: -len(".weight")], raw_name[: -len(".weight")],
                        shared_nvfp4=dense_nvfp4, lmhead_nvfp4=lmhead_nvfp4,
                        shared_buf=nvfp4_shared_buf,
                    )
                    if emit is not _NOT_DENSE_NVFP4:
                        for key, tensor in emit:
                            yield key, _shard(key, tensor)
                        continue

                tensor = _load_maybe_quantized(f, raw_name, keyset)

                # merge shared-expert gate/up -> gate_up_proj
                if name.endswith(_SHARED_GATE) or name.endswith(_SHARED_UP):
                    prefix = name.rsplit(".mlp.shared_expert.", 1)[0]
                    slots = shared_buf.setdefault(prefix, {})
                    slots["gate" if name.endswith(_SHARED_GATE) else "up"] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        key = f"{prefix}.mlp.shared_expert.gate_up_proj.weight"
                        yield key, _shard(key, merged)
                    continue

                # fuse q/k/v -> qkv_proj and GDN in_proj_{qkv,z,b,a} -> in_proj
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused[0], _shard(fused[0], fused[1])
                    continue

                if _is_gemma_norm(name):
                    tensor = tensor + 1.0  # (1 + weight) baked into the stored weight

                yield name, _shard(name, tensor)

    assert not shared_buf, f"Incomplete shared-expert merges: {list(shared_buf.keys())}"
    assert not nvfp4_shared_buf, f"Incomplete NVFP4 shared-expert merges: {list(nvfp4_shared_buf.keys())}"
    assert not fuse_buf, f"Incomplete projection fusions: {list(fuse_buf.keys())}"


# ======================================================================================
# Mixed-precision modelopt checkpoint (per-tensor FP8 attn/GDN + NVFP4 experts/shared/lm_head)
# ======================================================================================
# FP8 projections fused along the output dim, each part carrying its own scalar weight_scale
# -> a per-output-row scale vector. Keys are the model buffer base (sans .weight/.weight_scale).
_PT_FP8_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
    ),
    ".linear_attn.in_proj_qkvz": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ),
}
# bf16 (unquantized) GDN b|a projections fused -> in_proj_ba (matches the fp8 split).
_PT_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".linear_attn.in_proj_ba": (".linear_attn.in_proj_b", ".linear_attn.in_proj_a"),
}


def _per_row_scale(scalar: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar -> per-output-row fp32 vector ``[rows]`` (exact broadcast)."""
    return scalar.reshape(1).to(torch.float32).expand(rows)


def _pt_fp8_fuse(base: str, weight: torch.Tensor, scalar: torch.Tensor,
                 act_scale: torch.Tensor | None, buf: dict):
    """Buffer an fp8 fusion part ``(weight, scalar, act_scale)``; once all parts arrive emit
    the concatenated ``(.weight fp8, .weight_scale per-row fp32)`` plus the shared
    ``.input_scale``. ``[]`` while incomplete, ``None`` if ``base`` is not an fp8 fusion part.

    The fused parts all read the *same* activation, so modelopt calibrates one activation
    range for all of them and their ``input_scale`` values come out bit-identical (verified on
    Qwen3.8-27B-NVFP4: q/k/v all 0.2053571492, GDN qkv/z both 0.1121651828). Taking the max is
    therefore exact here, and stays correct if a future checkpoint lets them drift."""
    for fused_suffix, parts in _PT_FP8_FUSE.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = (weight, scalar, act_scale)
                if len(slots) < len(parts):
                    return []
                del buf[key]
                ws = [slots[i][0] for i in range(len(parts))]
                ss = [_per_row_scale(slots[i][1], slots[i][0].shape[0]) for i in range(len(parts))]
                emit = [
                    (key + ".weight", torch.cat(ws, dim=0)),
                    (key + ".weight_scale", torch.cat(ss, dim=0).contiguous()),
                ]
                acts = [slots[i][2] for i in range(len(parts))]
                if all(a is not None for a in acts):
                    emit.append((key + ".input_scale", torch.stack(
                        [a.reshape(()).to(torch.float32) for a in acts]).max()))
                return emit
    return None


# Native-NVFP4 dense projections (W4A16): shared-expert gate/up merged on the output dim,
# down + lm_head standalone. Each carries weight (uint8 [O,IN//2]), block scale (fp8
# [O,IN//16]), and a per-output-row global scale (weight_scale_2 broadcast, fp16 [O]). The
# fused gate|up concatenates all three (each part keeps its own global), so it is exact.
_SHARED_GATE_BASE = ".mlp.shared_expert.gate_proj"
_SHARED_UP_BASE = ".mlp.shared_expert.up_proj"
# The MoE shared-expert MLP and the dense (non-MoE) decoder MLP have identical native-FP4
# structure (gate|up merged -> gate_up_proj + standalone down_proj); they differ only in the
# ``.mlp.shared_expert.`` vs bare ``.mlp.`` infix. ``endswith(".mlp.gate_proj")`` is False for
# ``.mlp.shared_expert.gate_proj``, and routed experts are excluded upstream (_NVFP4_EXPERT_RE).
_NVFP4_MLP_LAYOUTS = (
    (".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
     ".mlp.shared_expert.down_proj", ".mlp.shared_expert."),
    (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".mlp."),
)


def _nvfp4_parts(f, raw_base: str):
    """Load a native NVFP4 weight as ``(packed uint8 [O, IN//2], block scale fp8 [O, IN//16],
    per-output-row global fp16 [O])`` -- the dense W4A16 kernels' expected buffers."""
    w = f.get_tensor(raw_base + ".weight")            # uint8 packed FP4 (2 codes/byte)
    s = f.get_tensor(raw_base + ".weight_scale")      # fp8-e4m3 per-16 block scale
    g2 = f.get_tensor(raw_base + ".weight_scale_2")   # per-tensor global scalar
    g = g2.reshape(1).to(torch.float16).expand(w.shape[0]).contiguous()
    return w, s, g


# Sentinel: ``base`` is not a dense projection the model keeps native NVFP4 (caller dequantizes).
_NOT_DENSE_NVFP4 = object()


def _dense_nvfp4_emit(
    f, base: str, raw_base: str, *, shared_nvfp4: bool, lmhead_nvfp4: bool, shared_buf: dict
):
    """For a dense ``.weight`` whose checkpoint has a ``weight_scale_2`` (NVFP4), return the list
    of ``(key, tensor)`` to yield as native FP4 -- ``(.weight uint8, .weight_scale fp8 block,
    .weight_global fp16 per-row)`` -- when the model keeps that layer native:

    * the MoE ``shared_expert.{gate,up,down}_proj`` OR the dense (non-MoE) ``.mlp.{gate,up,down}
      _proj`` when ``shared_nvfp4`` (gate/up merged -> ``gate_up_proj``, each part keeping its own
      global scale so the fused weight is exact);
    * ``lm_head`` when ``lmhead_nvfp4``.

    Returns ``[]`` while a gate/up merge is still buffered, or ``_NOT_DENSE_NVFP4`` if the model
    does not keep this layer native (the caller dequantizes to bf16 exactly as before). Shared by
    the mixed-FP8 dense pass and the default (pure-NVFP4) dense pass."""
    is_lmhead = base == "lm_head" or base.endswith(".lm_head")
    if lmhead_nvfp4 and is_lmhead:
        w, s, g = _nvfp4_parts(f, raw_base)
        return [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
    if not shared_nvfp4:
        return _NOT_DENSE_NVFP4
    for gate_b, up_b, down_b, infix in _NVFP4_MLP_LAYOUTS:
        if base.endswith(down_b):
            w, s, g = _nvfp4_parts(f, raw_base)
            return [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
        if base.endswith(gate_b) or base.endswith(up_b):
            w, s, g = _nvfp4_parts(f, raw_base)
            prefix = base.rsplit(infix, 1)[0] + infix
            slots = shared_buf.setdefault(prefix, {})
            slots["gate" if base.endswith(gate_b) else "up"] = (w, s, g)
            if "gate" not in slots or "up" not in slots:
                return []
            gw, gs, gg = slots["gate"]
            uw, us, ug = slots["up"]
            del shared_buf[prefix]
            pre = f"{prefix}gate_up_proj"
            return [
                (pre + ".weight", torch.cat([gw, uw], dim=0)),
                (pre + ".weight_scale", torch.cat([gs, us], dim=0)),
                (pre + ".weight_global", torch.cat([gg, ug], dim=0)),
            ]
    return _NOT_DENSE_NVFP4


def _iter_weights_attn_fp8(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool,
    dense_nvfp4: bool = False, lmhead_nvfp4: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for the modelopt MIXED_PRECISION Qwen3.5 checkpoint.

    Per-tensor FP8 attn/GDN projections (``self_attn.{q,k,v,o}_proj``, ``linear_attn.
    {in_proj_qkv,in_proj_z,out_proj}``) are kept fp8-e4m3 and yielded as ``.weight`` (fp8) +
    ``.weight_scale`` (per-output-row fp32) instead of dequantized to bf16 -- this halves the
    decode weight traffic of the dense backbone. q/k/v -> ``qkv_proj``, GDN qkv|z ->
    ``in_proj_qkvz`` (fp8) and b|a -> ``in_proj_ba`` (bf16). NVFP4 dense weights
    (shared_expert, lm_head): kept native FP4 -- ``.weight`` (uint8) + ``.weight_scale``
    (fp8 block) + ``.weight_global`` (fp16 per-row) for the W4A16 kernels -- when
    ``dense_nvfp4`` else dequantized to bf16. Routed NVFP4 experts are excluded (served by
    the offload cache). Gemma (1+w) norms get +1."""
    if not include_non_moe:
        return  # experts-only call: NVFP4 experts are loaded by the offload bank provider

    tp_info = get_tp_info()
    config = parse_config(cached_load_hf_config(model_path))

    def _shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tp_shard_key(name, tensor, config, tp_info)

    fp8_buf: dict[str, dict[int, tuple]] = {}
    bf16_buf: dict[str, dict[int, torch.Tensor]] = {}
    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    nvfp4_shared_buf: dict[str, dict[str, tuple]] = {}

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading mixed-fp8 weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue  # routed experts -> offload cache
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue  # scales consumed with their .weight

                name = _rename(raw_name, config.mtp_enabled)
                if name is None:
                    continue
                if _PACKED_EXPERT_PATTERN.match(name) is not None:
                    continue  # no packed experts in this checkpoint; guard anyway

                if name.endswith(".weight"):
                    base = name[: -len(".weight")]
                    raw_base = raw_name[: -len(".weight")]
                    has_s2 = raw_base + ".weight_scale_2" in keyset
                    has_s = raw_base + ".weight_scale" in keyset
                    if has_s and not has_s2:  # per-tensor FP8 dense projection
                        w = f.get_tensor(raw_name)  # fp8-e4m3, kept verbatim
                        sc = f.get_tensor(raw_base + ".weight_scale")
                        # modelopt's calibrated activation scale: kept (not dropped with the
                        # other scale suffixes) so batched decode can run W8A8 instead of
                        # W8A16. Absent -> the layer stays on the W8A16 kernel.
                        act = (f.get_tensor(raw_base + ".input_scale")
                               if raw_base + ".input_scale" in keyset else None)
                        emit = _pt_fp8_fuse(base, w, sc, act, fp8_buf)
                        if emit is not None:
                            for key, tensor in emit:
                                yield key, _shard(key, tensor)
                            continue
                        # standalone fp8 (self_attn.o_proj, linear_attn.out_proj)
                        yield base + ".weight", _shard(base + ".weight", w)
                        scale = _per_row_scale(sc, w.shape[0]).contiguous()
                        yield base + ".weight_scale", _shard(base + ".weight_scale", scale)
                        if act is not None:
                            yield base + ".input_scale", act.reshape(()).to(torch.float32)
                        continue
                    if has_s2:  # NVFP4 dense: keep native (W4A16) where the model expects it
                        emit = _dense_nvfp4_emit(
                            f, base, raw_base, shared_nvfp4=dense_nvfp4,
                            lmhead_nvfp4=lmhead_nvfp4, shared_buf=nvfp4_shared_buf,
                        )
                        if emit is not _NOT_DENSE_NVFP4:
                            for key, tensor in emit:
                                yield key, _shard(key, tensor)
                            continue
                    # NVFP4 -> bf16 (shared_expert, lm_head; dense_nvfp4 off); plain bf16 passes through.
                    tensor = _load_maybe_quantized(f, raw_name, keyset)
                    emit = _ct_bf16_fuse(base, tensor, bf16_buf, _PT_BF16_FUSE)
                    if emit is not None:
                        for key, part in emit:
                            yield key, _shard(key, part)
                        continue
                else:
                    tensor = f.get_tensor(raw_name)

                # shared-expert gate/up -> gate_up_proj (bf16, dequantized above)
                if name.endswith(_SHARED_GATE) or name.endswith(_SHARED_UP):
                    prefix = name.rsplit(".mlp.shared_expert.", 1)[0]
                    slots = shared_buf.setdefault(prefix, {})
                    slots["gate" if name.endswith(_SHARED_GATE) else "up"] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        key = f"{prefix}.mlp.shared_expert.gate_up_proj.weight"
                        yield key, _shard(key, merged)
                    continue

                if _is_gemma_norm(name):
                    tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight

                yield name, _shard(name, tensor)

    assert not fp8_buf, f"Incomplete fp8 fusions: {list(fp8_buf.keys())}"
    assert not bf16_buf, f"Incomplete bf16 fusions: {list(bf16_buf.keys())}"
    assert not shared_buf, f"Incomplete shared-expert merges: {list(shared_buf.keys())}"
    assert not nvfp4_shared_buf, f"Incomplete NVFP4 shared-expert merges: {list(nvfp4_shared_buf.keys())}"


# ======================================================================================
# compressed-tensors NVFP4 checkpoint (dense Qwen3.x, e.g. Qwen3.6-27B)
# ======================================================================================
# NVFP4 (W4A16) targets every Linear except the per-module ``ignore`` list (lm_head, GDN
# in_proj_*, vision, mtp). Storage differs from modelopt: ``weight_packed`` (uint8 [O, IN//2])
# + ``weight_scale`` (fp8-e4m3 block [O, IN//16]) + a scalar ``weight_global_scale``. The
# stored global is the *quant-side* scale, so the dequant/native global is its reciprocal
# (``1/weight_global_scale``) -- vLLM inverts it identically. Dense MLP gate/up and attention
# q/k/v fuse on the output dim into ``gate_up_proj`` / ``qkv_proj`` (each part keeps its own
# global, so the fused FP4 weight is exact). GDN ``in_proj_{qkv,z,b,a}`` stay bf16 -> ``in_proj``.
_CT_NVFP4_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}
_CT_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".linear_attn.in_proj": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
    ),
}
# When the model splits GDN (qkv|z fp8 -> ``in_proj_qkvz``, b|a bf16 -> ``in_proj_ba``),
# the bf16 parts must fuse into the same split buffers the model builds -- mirroring the
# modelopt path's ``_PT_BF16_FUSE``. Used only when the model has a scheme for
# ``in_proj_qkvz``; a fully bf16 GDN keeps the joint ``in_proj`` fusion above.
_CT_BF16_SPLIT_FUSE: dict[str, tuple[str, ...]] = {
    ".linear_attn.in_proj_qkvz": (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z"),
    ".linear_attn.in_proj_ba": (".linear_attn.in_proj_b", ".linear_attn.in_proj_a"),
}
_GDN_IN_PROJ_PARTS = (
    ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
)
_GDN_BF16_FUSE_PARTS = frozenset(
    p for groups in (_CT_BF16_FUSE, _CT_BF16_SPLIT_FUSE) for parts in groups.values() for p in parts
)
# The MTP head's bf16 block: q/k/v -> qkv_proj, gate/up -> gate_up_proj (the same fused
# buffers the main stack uses). Kept separate from _CT_BF16_FUSE so an NVFP4 checkpoint's
# main q/k/v stay native (the MTP parts are always bf16 ``.weight``).
_MTP_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}
# FP8 (channel-wise) parts that fuse on the output dim. A compressed-tensors checkpoint can
# mix an FP8 group (attention/GDN qkv|z/out_proj/lm_head, sometimes late MLP) with NVFP4.
_CT_FP8_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".linear_attn.in_proj_qkvz": (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}
_FLOAT8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# The scale suffixes and parts/fuse machinery are shared with muse_glimmer and live
# in models/loader.py.
_CT_SCALE_SUFFIXES = CT_SCALE_SUFFIXES
_nvfp4_parts_ct = nvfp4_parts_ct
_ct_bf16_fuse = ct_bf16_fuse


def _ct_nvfp4_fuse(base: str, parts_tuple: tuple, buf: dict):
    return ct_nvfp4_fuse(base, parts_tuple, buf, _CT_NVFP4_FUSE)


def _ct_fp8_fused(base: str) -> str:
    """The fused module prefix ``base`` is a part of, else ``base`` itself."""
    for fused, parts in _CT_FP8_FUSE.items():
        for part in parts:
            if base.endswith(part):
                return base[: -len(part)] + fused
    return base


def _is_fp8_scheme(scheme) -> bool:
    return scheme is not None and getattr(scheme, "kind", None) == QuantKind.FP8_TENSOR


def _model_scheme(quant, prefix: str):
    """The model's scheme for ``prefix``, or None when it is bf16 (or mixes schemes)."""
    if quant is None:
        return None
    try:
        return quant.scheme_for(prefix)
    except Exception:  # noqa: BLE001 -- a mixed/invalid fused probe means "not fp8"
        return None


def _gdn_split(quant, base: str) -> bool:
    """Whether the model builds the split GDN buffers (``in_proj_qkvz`` + ``in_proj_ba``)
    for the projection ``base`` -- true iff it has a quantization scheme for the fused
    ``in_proj_qkvz`` (the same probe ``GDNLayer._split_in_proj`` uses)."""
    for part in _GDN_IN_PROJ_PARTS:
        if base.endswith(part):
            return _model_scheme(quant, base[: -len(part)] + ".linear_attn.in_proj_qkvz") is not None
    return False


def _iter_weights_compressed_tensors(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool,
    nvfp4: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for a compressed-tensors NVFP4 checkpoint (e.g. Qwen3.6-27B).

    Keeps the NVFP4 attention (q/k/v/o, GDN out_proj) and dense MLP (gate/up/down) native
    (W4A16) -- ``.weight`` (uint8) + ``.weight_scale`` (fp8 block) + ``.weight_global`` (fp16
    per-row) -- when ``nvfp4``; otherwise dequantizes each to bf16. q/k/v -> ``qkv_proj``, dense gate/up -> ``gate_up_proj`` (output-dim concat).
    GDN ``in_proj_{qkv,z,b,a}`` stay bf16 -> fused ``in_proj``; ``conv1d``/``A_log``/``dt_bias``/
    gated ``norm`` pass through (fp32 for A_log/dt_bias). Gemma (1+w) norms get +1. lm_head and
    embeddings are bf16. The model is dense (no routed experts), so there is no experts pass."""
    if not include_non_moe:
        return  # dense checkpoint: no routed experts to load

    tp_info = get_tp_info()
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    from freetoken.models.quant import quant_config_of

    quant = quant_config_of(hf_config)

    def _shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tp_shard_key(name, tensor, config, tp_info)

    nvfp4_buf: dict[str, dict[int, tuple]] = {}
    bf16_buf: dict[str, dict[int, torch.Tensor]] = {}
    fp8_buf: dict[str, dict[int, tuple]] = {}

    def _emit_bf16_weight(name: str, tensor: torch.Tensor):
        """Plain bf16 ``.weight``: fusion, Gemma (1+w) norms, else passthrough."""
        base = name[: -len(".weight")]
        if base.startswith("mtp."):
            groups = _MTP_BF16_FUSE
        elif _gdn_split(quant, base):
            groups = _CT_BF16_SPLIT_FUSE
        else:
            groups = _CT_BF16_FUSE
        emit = _ct_bf16_fuse(base, tensor, bf16_buf, groups)
        if emit is not None:
            for key, part in emit:
                yield key, _shard(key, part)
            return
        if _is_gemma_norm(name):
            tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight
        yield name, _shard(name, tensor)

    # Scale lookups go through the shard-map reader: a weight_packed's quant scales
    # can land in a different shard than the packed weight (the Muse-Glimmer layer-49
    # shape; nothing prevents an llm-compressor Qwen export from splitting the same way).
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading compressed-tensors weights",
            disable=not tp_info.is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if raw_name.startswith(("model.visual.", "visual.")):
                    continue
                if raw_name.endswith(_CT_SCALE_SUFFIXES):
                    continue  # consumed with weight_packed (or unused W4A4 activation scales)

                name = _rename(raw_name, config.mtp_enabled)
                if name is None:
                    continue

                if raw_name.endswith(".weight_packed"):  # NVFP4 projection
                    base = name[: -len(".weight_packed")]
                    raw_base = raw_name[: -len(".weight_packed")]
                    w, s, g = _nvfp4_parts_ct(reader, raw_base)
                    # GDN in_proj_* compute in bf16 (model contract) but some checkpoints
                    # (e.g. sakamakismile/Qwen3.6-27B-NVFP4) quantize them too: dequant to
                    # bf16 here and let the bf16 fusion assemble the model's GDN buffers
                    # (joint ``in_proj``, or the split ``in_proj_qkvz``/``in_proj_ba``).
                    if any(base.endswith(p) for p in _GDN_BF16_FUSE_PARTS):
                        bf16 = _dequant_nvfp4_weight(w, s, g[:1])
                        yield from _emit_bf16_weight(base + ".weight", bf16)
                        continue
                    if nvfp4:  # keep native (W4A16)
                        emit = _ct_nvfp4_fuse(base, (w, s, g), nvfp4_buf)
                        if emit is not None:
                            for key, part in emit:
                                yield key, _shard(key, part)
                        else:  # standalone: o_proj, linear_attn.out_proj, mlp.down_proj
                            yield base + ".weight", _shard(base + ".weight", w)
                            yield base + ".weight_scale", _shard(base + ".weight_scale", s)
                            yield base + ".weight_global", _shard(base + ".weight_global", g)
                        continue
                    # bf16 A-B: dequant FP4 -> bf16, then merge q/k/v + gate/up as bf16. ``g`` is
                    # already the dequant global (1/weight_global_scale) per row; pass one element.
                    bf16 = _dequant_nvfp4_weight(w, s, g[:1])
                    emit = _ct_bf16_fuse(base, bf16, bf16_buf, _CT_NVFP4_FUSE)
                    if emit is not None:
                        for key, part in emit:
                            yield key, _shard(key, part)
                    else:
                        yield base + ".weight", _shard(base + ".weight", bf16)
                    continue

                if name.endswith(".weight"):
                    tensor = reader.get_tensor(raw_name)
                    if tensor.dtype in _FLOAT8_DTYPES:
                        # Compressed-tensors FP8 (channel-wise): keep fp8 where the model
                        # builds an FP8 linear, else dequantize to bf16 for the bf16 fusion.
                        base = name[: -len(".weight")]
                        raw_base = raw_name[: -len(".weight")]
                        scale = reader.get_tensor(raw_base + ".weight_scale")  # [O, 1] bf16
                        sc = scale.reshape(-1).to(torch.float32).contiguous()
                        if _is_fp8_scheme(_model_scheme(quant, _ct_fp8_fused(base))):
                            emit = ct_fp8_fuse(base, (tensor, sc), fp8_buf, _CT_FP8_FUSE)
                            if emit is not None:
                                for key, part in emit:
                                    yield key, _shard(key, part)
                            else:  # standalone: o_proj, out_proj, down_proj, lm_head
                                yield base + ".weight", _shard(base + ".weight", tensor)
                                yield base + ".weight_scale", _shard(base + ".weight_scale", sc)
                            continue
                        bf16 = (tensor.to(torch.bfloat16) * scale.to(torch.bfloat16)).to(torch.bfloat16)
                        yield from _emit_bf16_weight(base + ".weight", bf16)
                        continue
                    yield from _emit_bf16_weight(name, tensor)
                    continue

                # A_log / dt_bias (kept fp32 by the model; the load downcast exempts them).
                yield name, _shard(name, reader.get_tensor(raw_name))
    finally:
        reader.close()

    assert not nvfp4_buf, f"Incomplete NVFP4 fusions: {list(nvfp4_buf.keys())}"
    assert not bf16_buf, f"Incomplete bf16 fusions: {list(bf16_buf.keys())}"
    assert not fp8_buf, f"Incomplete fp8 fusions: {list(fp8_buf.keys())}"


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel read via the common chunked multi-threaded O_DIRECT reader.
    Qwen3.5 stores experts pre-fused/pre-stacked per layer (already ``[E, ...]``), so no
    merge/stack -- just rename and yield; bank builder places by name as the serial path."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_5_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    tp_info = get_tp_info()
    config = parse_config(cached_load_hf_config(model_path))

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name, config.mtp_enabled)
        return name is not None and _PACKED_EXPERT_PATTERN.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        name = _rename(raw_name, config.mtp_enabled)
        yield name, tp_shard_key(name, tensor, config, tp_info)


# ======================================================================================
# Block-FP8 checkpoint (Qwen3.5-35B-A3B-FP8): dense weights + offload expert banks.
# ======================================================================================
# fused model buffer suffix -> ordered checkpoint part suffixes (matched without the
# trailing .weight / .weight_scale_inv). Both kinds ride the same fusion (concatenated
# along dim 0); in_proj_ba carries only .weight (b/a stay bf16, no block scale).
_FP8_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
    ),
    ".linear_attn.in_proj_qkvz": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ),
    ".linear_attn.in_proj_ba": (
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
    ),
    ".mlp.shared_expert.gate_up_proj": (
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
    ),
    # Dense (non-MoE) layer MLP: merge gate|up -> gate_up_proj for both the fp8 ``.weight``
    # and the bf16 ``.weight_scale_inv`` (fused per kind by _split_kind). Only a bare
    # ``.mlp.gate_proj`` matches; the shared_expert entry above keeps the MoE case.
    ".mlp.gate_up_proj": (
        ".mlp.gate_proj", ".mlp.up_proj",
    ),
}
_FP8_KIND_SUFFIXES = (".weight_scale_inv", ".weight")



def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(base, kind_suffix)``; ``kind_suffix`` is "" for keys without a
    weight/scale suffix (A_log, dt_bias)."""
    for suf in _FP8_KIND_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def _fp8_fuse(base: str, suf: str, tensor: torch.Tensor, buf: dict) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part keyed by (fused_full_name, kind); return the concatenated
    ``(name, tensor)`` once all parts for that kind arrive, ``()`` while incomplete,
    ``None`` if ``base`` is not a fusion part."""
    for fused_suffix, parts in _FP8_FUSIONS.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                fused_base = base[: -len(part)] + fused_suffix
                key = (fused_base + suf, suf)
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return fused_base + suf, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def _iter_weights_fp8(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the block-fp8 weights, renamed + fused to the model buffers.

    fp8 weights (e4m3) and their bf16 ``weight_scale_inv`` pass through verbatim (no dtype
    cast -- the engine's load-time cast is a no-op against the fp8/bf16 model buffers).
    q/k/v -> qkv_proj, GDN in_proj_qkv|z -> in_proj_qkvz (fp8) and in_proj_b|a -> in_proj_ba
    (bf16), shared_expert gate|up -> gate_up_proj; Gemma (1+w) norms get +1.

    Routed experts: skipped under offload (loaded from expert pieces). Under the
    resident (non-offload) path ``include_moe_experts`` is True -> per-layer stacked fp8
    experts for the Fp8ResidentMoE buffers are yielded too."""
    if include_non_moe:
        tp_info = get_tp_info()
        config = parse_config(cached_load_hf_config(model_path))
        fuse_buf: dict = {}
        for file in tqdm(iter_weight_files(model_path), desc="Loading fp8 weights",
                         disable=not tp_info.is_primary()):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = _rename(raw_name, config.mtp_enabled)
                    if name is None or ".mlp.experts." in name:
                        continue  # routed experts handled below / by the offload cache
                    tensor = f.get_tensor(raw_name)
                    base, suf = _split_kind(name)
                    fused = _fp8_fuse(base, suf, tensor, fuse_buf)
                    if fused is not None:
                        if fused != ():
                            yield fused[0], tp_shard_key(fused[0], fused[1], config, tp_info)
                        continue
                    if _is_gemma_norm(name):
                        tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight
                    yield name, tp_shard_key(name, tensor, config, tp_info)
        assert not fuse_buf, f"Incomplete fp8 fusions: {sorted(k for k, _ in fuse_buf)}"

    if include_moe_experts:
        # resident experts: stack the per-expert pieces into the per-layer tensors the resident MoE method declares (pageable host; the engine copies to GPU)
        config = parse_config(cached_load_hf_config(model_path))
        if not config.is_moe:
            return  # dense checkpoint: no routed experts to build as resident banks
        yield from _resident_fp8_experts(model_path, config)


def _resident_fp8_experts(model_path, config):
    """Stack the per-expert block-fp8 pieces into the resident per-layer banks.

    TP: ``iter_fp8_block_expert_pieces`` already returns this rank's intermediate shard,
    so the stacked tensors carry the local widths -- they must NOT be sharded again."""
    from freetoken.distributed import get_tp_info
    from freetoken.kernel.triton.fp8_block_linear import FP8
    from freetoken.utils import div_even

    B = 128
    tp = get_tp_info()
    L, E, H, I_full, dense = _moe_dims(config)
    I_loc = div_even(I_full, tp.size)
    shapes = {
        "gate_up_proj": ((E, 2 * I_loc, H), FP8),
        "gate_up_scale_inv": ((E, 2 * I_loc // B, H // B), torch.bfloat16),
        "down_proj": ((E, H, I_loc), FP8),
        "down_scale_inv": ((E, H // B, I_loc // B), torch.bfloat16),
    }
    layers: dict[int, dict[str, torch.Tensor]] = {}
    placed = [0] * L
    for li, e0, e1, piece in iter_fp8_block_expert_pieces(model_path, config, parallel=None):
        stack = layers.setdefault(li, {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in shapes.items()})
        stack["gate_up_proj"][e0:e1, :I_loc] = piece["gate"]
        stack["gate_up_proj"][e0:e1, I_loc:] = piece["up"]
        stack["gate_up_scale_inv"][e0:e1, : I_loc // B] = piece["gate_scale"]
        stack["gate_up_scale_inv"][e0:e1, I_loc // B :] = piece["up_scale"]
        stack["down_proj"][e0:e1] = piece["down"]
        stack["down_scale_inv"][e0:e1] = piece["down_scale"]
        placed[li] += e1 - e0
        if placed[li] == E:
            pre = f"model.layers.{dense + li}.mlp.experts"
            for name, tensor in layers.pop(li).items():
                yield f"{pre}.{name}", tensor
    assert not layers, f"incomplete resident fp8 experts for layers {sorted(layers)}"


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "nvfp4_expert_spec",
    "tp_shard_key",
    "shard_ftw_weight",
]
