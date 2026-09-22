"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints, all in the multimodal-wrapper layout (``model.language_model.*``):
NVFP4 exports (ModelOpt tensor kinds from LibertAIDAI, compressed-tensors kinds from
RedHatAI, selected by ``quantization_config``) and the zai-org block-fp8 release. Not
supported: text-only key layouts and resident routed experts.

TP: the resident state shards across ranks (per-head q/kv projections, row-parallel
o_proj/down_proj, column-parallel gate/up, TP-local KDA heads/gates/conv, vocab-
parallel embed + lm_head); the low-rank KDA bottlenecks f_a|g_a, the MLA latent
projections, the indexer, the router and the mHC mixers stay replicated. The bf16
resident path (NVFP4 checkpoints) is TP-sharded below; checkpoints that store
fp8 resident projections (block scales on a 128-grid) stay TP=1.

Routed experts go to the offload cache from their NVFP4 or block-fp8 pieces; every
other projection loads as stored (bf16, or fp8 codes with their block scales) with keys
renamed ``model.language_model.X`` -> ``model.X``. ``model.visual.*`` and the trailing
MTP layer are never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``  = q|k|v|b|f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``   = q|k|v depthwise conv weights concatenated on the channel axis

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.glm_moe_dsa.weight import _ShardReader
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from tqdm import tqdm

from .args import Glm5NextArgs
from .config import parse_config

# Checkpoint prefix (multimodal wrapper) -> model prefix.
_CKPT = "model.language_model"
_MODEL = "model"

# MTP-layer experts (layer == num_layers under the full checkpoint) map to None
# alongside the dense prefix; the bank loader skips them.
def _layer_to_bank(layer, config):
    return (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    )


# ModelOpt export (LibertAIDAI/GLM-5.3-Flash-NVFP4): weight | weight_scale |
# weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)

# llm-compressor export (RedHatAI/GLM-5.3-Flash-NVFP4): weight_packed |
# weight_scale | weight_global_scale (quant-side global -> reciprocal at ingest).
# ``input_global_scale`` (the calibrated W4A4 activation scale) deliberately does
# not match: our routed-expert paths are W4A16 and never quantize activations.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)


def _select_expert_source_spec(model_path: str) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC

# KDA in_proj fusion order; MUST match Glm5NextKDA._in_proj_split.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")

# zai-org fp8 release: fp8 codes + fp32 128x128 block scales per expert projection
_FP8_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)


def _proj(reader, src: str, dst: str) -> Iterator[tuple[str, torch.Tensor]]:
    """One projection as the checkpoint stores it: bf16, or fp8 codes with their block scales."""
    w = reader.get(f"{src}.weight")
    if w.dtype == torch.float8_e4m3fn:
        yield f"{dst}.weight", w
        yield f"{dst}.weight_scale_inv", reader.get(f"{src}.weight_scale_inv")
    else:
        yield f"{dst}.weight", w.to(torch.bfloat16)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    return _select_expert_source_spec(model_path)


# --- TP slicing -------------------------------------------------------------
#
# The resident state shards across ranks; the model modules build their local
# shapes from get_tp_info() (div_even), so the loader must emit exactly those
# views. All output axes that split are head-major or per-channel concatenations
# with even divisibility, so a contiguous rank slice of rows/cols is the shard.


def _tp_bf16(w: torch.Tensor, what: str) -> torch.Tensor:
    """TP shard guard: the per-rank slices assume bf16 resident weights (fp8
    block scales live on a 128-grid the slices don't respect)."""
    if w.dtype == torch.float8_e4m3fn:
        raise NotImplementedError(f"{what}: fp8 resident projections support TP=1 only")
    return w.to(torch.bfloat16)


def _col_rows(w: torch.Tensor, tp, out_full: int) -> torch.Tensor:
    """Column-parallel shard: this rank's rows of a full ``[out_full, ...]`` weight.
    ``.contiguous()`` materializes the slice — a view would keep the FULL base
    tensor's storage alive on the GPU and defeat the whole sharding."""
    loc = div_even(out_full, tp.size)
    return w[tp.rank * loc : (tp.rank + 1) * loc].contiguous()


def _row_cols(w: torch.Tensor, tp, in_full: int) -> torch.Tensor:
    """Row-parallel shard: this rank's columns of a full ``[..., in_full]`` weight."""
    loc = div_even(in_full, tp.size)
    return w[:, tp.rank * loc : (tp.rank + 1) * loc].contiguous()


def _vocab_rows(w: torch.Tensor, tp, vocab: int) -> torch.Tensor:
    """Vocab-parallel shard: ``div_ceil(vocab, tp)`` rows per rank, zero-padded on
    the last rank (VocabParallelEmbedding allocates the full padded width)."""
    n_tp = div_ceil(vocab, tp.size)
    start = tp.rank * n_tp
    out = w.new_zeros(n_tp, *w.shape[1:])
    src = w[start : start + n_tp]
    out[: src.shape[0]] = src
    return out


def _kda_in_proj_local(parts: list[torch.Tensor], tp, h: int, d: int) -> torch.Tensor:
    """KDA fused in_proj (q|k|v|b|f_a|g_a, output-axis concat) → this rank's rows:
    the head-structured q|k|v|b carry the rank's head slice, the low-rank
    bottleneck f_a|g_a stays replicated (mirrors Glm5NextKDA's mixed shard)."""
    q_w, k_w, v_w, b_w, f_a, g_a = parts
    if tp.size == 1:
        return torch.cat(parts, dim=0)
    h_loc = div_even(h, tp.size)
    p_loc = h_loc * d
    r = tp.rank

    def rows(w, n):
        return w[r * n : (r + 1) * n]

    return torch.cat(
        [rows(q_w, p_loc), rows(k_w, p_loc), rows(v_w, p_loc), rows(b_w, h_loc), f_a, g_a],
        dim=0,
    )


def _kda_conv_local(convs: list[torch.Tensor], tp, h: int, d: int) -> torch.Tensor:
    """Merged q|k|v depthwise conv channels → this rank's head slice."""
    if tp.size == 1:
        return torch.cat(convs, dim=0)
    p_loc = div_even(h, tp.size) * d
    r = tp.rank
    return torch.cat([c[r * p_loc : (r + 1) * p_loc] for c in convs], dim=0)


def _iter_kda_layer(
    reader, layer: int, args: Glm5NextArgs, tp
) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    h, d = args.linear_num_heads, args.linear_head_dim
    p = h * d
    h_loc = div_even(h, tp.size)
    p_loc = h_loc * d
    # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat). Under TP the
    # head-structured q|k|v|b carry this rank's head slice; the low-rank
    # bottleneck f_a|g_a stays replicated (kda.py's mixed-shard in_proj).
    parts = [reader.get(f"{src}.{name}.weight") for name in _KDA_IN_PROJ]
    if any(x.dtype == torch.float8_e4m3fn for x in parts):
        if tp.size > 1:
            raise NotImplementedError("fp8 KDA resident projections support TP=1 only")
        raise NotImplementedError("fp8 KDA input projections are not fused by this reader")
    yield f"{dst}.in_proj.weight", _kda_in_proj_local(
        [x.to(torch.bfloat16) for x in parts], tp, h, d
    )
    # One merged depthwise conv over the q|k|v stream (channel-axis concat);
    # channels follow the rank's head slice.
    convs = [_tp_bf16(reader.get(f"{src}.{name}_conv1d.weight"), f"{dst}.conv1d") for name in ("q", "k", "v")]
    yield f"{dst}.conv1d.weight", _kda_conv_local(convs, tp, h, d)
    if tp.size > 1:
        yield f"{dst}.f_b_proj.weight", _col_rows(
            _tp_bf16(reader.get(f"{src}.f_b_proj.weight"), f"{dst}.f_b_proj"), tp, p
        )
        yield f"{dst}.g_b_proj.weight", _col_rows(
            _tp_bf16(reader.get(f"{src}.g_b_proj.weight"), f"{dst}.g_b_proj"), tp, p
        )
        yield f"{dst}.o_proj.weight", _row_cols(
            _tp_bf16(reader.get(f"{src}.o_proj.weight"), f"{dst}.o_proj"), tp, p
        )
    else:
        for name in ("f_b_proj", "g_b_proj", "o_proj"):
            yield from _proj(reader, f"{src}.{name}", f"{dst}.{name}")
    # Gate params stay fp32 (the recurrent kernels read them as fp32); TP-local
    # head/channel slices.
    A_log = reader.get(f"{src}.A_log").to(torch.float32)
    dt_bias = reader.get(f"{src}.dt_bias").to(torch.float32)
    if tp.size > 1:
        r = tp.rank
        A_log = A_log[r * h_loc : (r + 1) * h_loc].contiguous()
        dt_bias = dt_bias[r * p_loc : (r + 1) * p_loc].contiguous()
    yield f"{dst}.A_log", A_log
    yield f"{dst}.dt_bias", dt_bias
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(
    reader, layer: int, args: Glm5NextArgs, tp
) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        if tp.size > 1 and proj in ("q_b_proj", "kv_b_proj"):
            # per-head output rows: this rank's contiguous head slice
            w = _tp_bf16(reader.get(f"{src}.{proj}.weight"), f"{dst}.{proj}")
            yield f"{dst}.{proj}.weight", _col_rows(w, tp, w.shape[0])
        elif tp.size > 1 and proj == "o_proj":
            # row-parallel: the input axis is the head-major H*v_head_dim stream
            w = _tp_bf16(reader.get(f"{src}.{proj}.weight"), f"{dst}.{proj}")
            yield f"{dst}.{proj}.weight", _row_cols(w, tp, w.shape[1])
        else:
            yield from _proj(reader, f"{src}.{proj}", f"{dst}.{proj}")
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(torch.bfloat16)
    # kpool indexer (every DSA layer owns one). Kept bf16; the APE is fp32.
    for proj in ("wq_b", "wk", "weights_proj"):
        yield f"{dst}.indexer.{proj}.weight", reader.get(
            f"{src}.indexer.{proj}.weight"
        ).to(torch.bfloat16)
    for part, dtype in (
        ("k_norm.weight", torch.bfloat16),
        ("k_norm.bias", torch.bfloat16),
        ("index_kpool_compress_gate", torch.bfloat16),
        ("index_kpool_compress_ape", torch.float32),
    ):
        yield f"{dst}.indexer.{part}", reader.get(f"{src}.indexer.{part}").to(dtype)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 routed experts only serve from the offload cache; they are loaded from their expert pieces."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    tp = get_tp_info()
    # TP: the loader emits full fused KDA/DSA tensors; TP sharding (per-head
    # q|k|v|b splits, replicated f_a|g_a, row-parallel o_proj) slices them per
    # rank below. bf16 resident weights only -- the per-projection _tp_bf16
    # guards reject fp8-stored resident projections under TP.
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = tp.is_primary()
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            if args.is_kda_layer(layer):
                yield from _iter_kda_layer(reader, layer, args, tp)
            else:
                yield from _iter_dsa_layer(reader, layer, args, tp)

            # mHC mixing tensors, fp32 on every layer (replicated).
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                if tp.size > 1:
                    for proj in ("gate_proj", "up_proj"):
                        w = _tp_bf16(reader.get(f"{src}.mlp.{proj}.weight"), f"{dst}.mlp.{proj}")
                        yield f"{dst}.mlp.{proj}.weight", _col_rows(w, tp, config.intermediate_size)
                    w = _tp_bf16(reader.get(f"{src}.mlp.down_proj.weight"), f"{dst}.mlp.down_proj")
                    yield f"{dst}.mlp.down_proj.weight", _row_cols(w, tp, config.intermediate_size)
                else:
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        yield from _proj(reader, f"{src}.mlp.{proj}", f"{dst}.mlp.{proj}")
            else:
                yield f"{dst}.mlp.gate.weight", reader.get(f"{src}.mlp.gate.weight").to(
                    torch.bfloat16
                )
                yield (
                    f"{dst}.mlp.e_score_correction_bias",
                    # fp32 like HF's router math (the module declares fp32; a bf16
                    # cast would perturb top-8 selection on fp32-bias checkpoints).
                    reader.get(f"{src}.mlp.gate.e_score_correction_bias").to(torch.float32),
                )
                shared_i = config.moe_intermediate_size * max(1, config.n_shared_experts)
                if tp.size > 1:
                    for proj in ("gate_proj", "up_proj"):
                        w = _tp_bf16(reader.get(f"{src}.mlp.shared_experts.{proj}.weight"), f"{dst}.mlp.shared_experts.{proj}")
                        yield f"{dst}.mlp.shared_experts.{proj}.weight", _col_rows(w, tp, shared_i)
                    w = _tp_bf16(reader.get(f"{src}.mlp.shared_experts.down_proj.weight"), f"{dst}.mlp.shared_experts.down_proj")
                    yield f"{dst}.mlp.shared_experts.down_proj.weight", _row_cols(w, tp, shared_i)
                else:
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        yield from _proj(reader, f"{src}.mlp.shared_experts.{proj}", f"{dst}.mlp.shared_experts.{proj}")

        yield f"{_MODEL}.embed_tokens.weight", _vocab_rows(
            reader.get(f"{_CKPT}.embed_tokens.weight").to(torch.bfloat16),
            tp, config.vocab_size,
        )
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        if tp.size > 1:
            yield "lm_head.weight", _vocab_rows(
                _tp_bf16(reader.get("lm_head.weight"), "lm_head"), tp, config.vocab_size
            )
        else:
            yield "lm_head.weight", reader.get("lm_head.weight").to(torch.bfloat16)
    finally:
        reader.close()


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their ``_scale`` companions; other kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("glm5_next fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    suffix = {"weight": "", "weight_scale_inv": "_scale"}

    def locate(raw_name: str):
        m = _FP8_EXPERT_RE.match(raw_name)
        if m is None:
            return None
        bank = _layer_to_bank(int(m["layer"]), config)
        if bank is None:
            return None
        return bank, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        folder = download_hf_weight(model_path)
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        reader = _ShardReader(folder, weight_map, torch.device("cpu"))
        try:
            layers = range(config.first_k_dense_replace, config.num_layers)
            for layer in tqdm(layers, desc="Loading GLM-5.3 fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(config.num_experts):
                    base = f"{_CKPT}.layers.{layer}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind_name in suffix:
                            name = f"{base}.{proj}_proj.{kind_name}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces", "nvfp4_expert_spec"]
