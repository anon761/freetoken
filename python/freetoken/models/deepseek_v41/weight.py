"""Weight loading for DeepSeek-V4.1-Flash.

  - :func:`iter_weights` streams the resident (non-expert) tensors keyed by the
    model's attribute paths (``model.`` + checkpoint name). ``wo_a`` dequantizes
    fp8 32x32 blocks to bf16 (reference bf16 einsum). Checkpoint tensors with no
    Phase-2 consumer (attn compressor/indexer on the kv/index-source layers,
    engram tables, DSpark draft, vision tower) are counted and skipped — their
    phases consume them later (the DSV4.1 plan).
  - :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs +
    e8m0 per-32 scales, w1/w2/w3 + scale) as per-expert pieces for the expert
    quant method's banks, TP-sliced to the rank's intermediate band.
"""

from __future__ import annotations
from functools import lru_cache

import logging
import os
import re
from types import SimpleNamespace

import torch

from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.deepseek_v4.weight import (
    _ShardReader,
    _dequant_fp8_block,
    _weight_map,
)
from freetoken.models.weight import _spec_for_model_path
from freetoken.models.nvfp4_banks import _tp_expert_geometry
from freetoken.utils import div_ceil, div_even
from freetoken.utils.hf import cached_load_hf_config

from .args import load_args

logger = logging.getLogger(__name__)

_NO_TP = SimpleNamespace(size=1, rank=0, is_primary=lambda: True)


def _tp():
    """TP info, or a size-1 shim for single-process tooling (tests, convert)."""
    try:
        return get_tp_info()
    except RuntimeError:
        return _NO_TP


def _is_primary() -> bool:
    """Rank-0 gate for load logging; single-process tooling (tests, convert) has no TP set."""
    try:
        return get_tp_info().is_primary()
    except RuntimeError:
        return True


def tp_shard_key(name: str, w: torch.Tensor, config, tp) -> torch.Tensor:
    """This rank's view of a full resident tensor, by state-dict key. Identity at TP=1.

    wq_b shards rows over heads, wo_b cols and wo_a rows over the o-groups, the
    shared expert w1/w3 rows and w2 cols over the intermediate axis — each with
    its fp8 32x32 scale sliced on the same band (scale axes are the weight
    axes //32). wq_a/wkv are replicated (lora bottlenecks), as are norms, the
    router, hc coefficients and attn_sink; embed/head slice vocab rows with
    zero-pad (mirrors VocabParallelEmbedding/ParallelLMHead)."""
    if tp.size == 1:
        return w
    r = tp.rank
    block = 32  # fp8 weight scale block (config.json weight_block_size)

    def rows(w: torch.Tensor, n_full: int) -> torch.Tensor:
        loc = div_even(n_full, tp.size)
        return w[r * loc : (r + 1) * loc].contiguous()

    def cols(w: torch.Tensor, n_full: int) -> torch.Tensor:
        loc = div_even(n_full, tp.size)
        return w[:, r * loc : (r + 1) * loc].contiguous()

    def vocab_rows(w: torch.Tensor) -> torch.Tensor:
        n_tp = div_ceil(w.shape[0], tp.size)
        out = w.new_zeros(n_tp, *w.shape[1:])
        piece = w[r * n_tp : (r + 1) * n_tp]
        out[: piece.shape[0]] = piece
        return out

    if name.endswith(".weight_scale_inv"):
        base, scale = name[: -len(".weight_scale_inv")], True
    elif name.endswith(".weight"):
        base, scale = name[: -len(".weight")], False
    else:
        base, scale = name, False
    args = config.dsv41_args
    shared_i = config.shared_expert_intermediate_size

    if base in ("model.embed", "model.head") or base.endswith(".markov_head.embed") or base.endswith(".markov_head.head"):
        # vocab-parallel tables (the target's embed/head and the DSpark markov head):
        # zero-pad the rank's vocab slice exactly like VocabParallelEmbedding/ParallelLMHead
        return vocab_rows(w)
    if base.endswith(".attn.wq_b"):
        n_full = config.num_qo_heads * config.head_dim
        return rows(w, n_full // block if scale else n_full)
    if base.endswith(".attn.wo_a"):
        return rows(w, args.o_groups * args.o_lora_rank)
    if base.endswith(".attn.wo_b"):
        n_full = args.o_groups * args.o_lora_rank
        return cols(w, n_full // block if scale else n_full)
    if base.endswith(".ffn.shared_experts.w1") or base.endswith(".ffn.shared_experts.w3"):
        return rows(w, shared_i // block if scale else shared_i)
    if base.endswith(".ffn.shared_experts.w2"):
        return cols(w, shared_i // block if scale else shared_i)
    return w


# the deepseek_v4 family shares the shard reader / fp8-block dequant helpers
# (same export pipeline, identical formats at V4.1's 32x32 block size)
_WO_A_BLOCK = 32

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)
# The DSpark draft's experts use the target's per-expert MXFP4 layout under mtp.*;
# `stage` is the draft block (0..num_nextn_predict_layers-1) and becomes the bank layer.
_DSPARK_EXPERT_RE = re.compile(
    r"^mtp\.(?P<stage>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "scale": "_scale"}


def _raw_expert_dir(model_path: str) -> str:
    """The raw checkpoint backing an FTW dir's experts.

    Expert tensors live in the RAW checkpoint (the FTW carries only dense weights +
    repacked banks), so a `-FTW` dir resolves to its sibling raw dir for the serial
    TP2 build / the converter sources. A non-FTW or in-place path is returned as is.
    """
    if os.path.isfile(os.path.join(model_path, "freetoken_weight.json")) and not os.path.isfile(
        os.path.join(model_path, "model.safetensors.index.json")
    ):
        raw = model_path[: -len("-FTW")] if model_path.endswith("-FTW") else None
        if raw and os.path.isfile(os.path.join(raw, "model.safetensors.index.json")):
            return raw
    return model_path


def _tp_slice_piece(role: str, tensor: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One expert piece -> this rank's intermediate band.

    gate/up store ``[I, H/2]`` codes (scales ``[I, H/32]`` e8m0) — the sliced
    axis is the OUTPUT rows; down stores ``[H, I/2]`` codes (scale ``[H, I/32]``)
    — the sliced axis is the packed INPUT. MXFP4 scales cover 32 values (16
    packed bytes), so code slices run //2 and scale slices //32; the band bounds
    are byte- and scale-aligned (I_loc % 16 == 0, asserted by the geometry).
    """
    if role.startswith("down_scale"):
        return tensor[..., lo // 32 : hi // 32]
    if role.startswith("down"):
        return tensor[..., lo // 2 : hi // 2]
    return tensor[lo:hi]


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed MXFP4 experts come from the offload cache, so ``include_moe_experts``
    must be False (V4.1 runs --moe-strategy offload). Tensors keep checkpoint
    dtypes (fp8 + e8m0); ``wo_a`` dequantizes to bf16 (reference einsum).
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4.1 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(cached_load_hf_config(model_path), max_batch_size=1)
    config, _spec = _spec_for_model_path(model_path)
    tp = _tp()
    weight_map = _weight_map(model_path)
    reader = _ShardReader(model_path, weight_map, device)
    skipped: dict[str, int] = {}

    def skip(prefix: str, why: str):
        n = sum(1 for k in weight_map if k.startswith(prefix))
        if n:
            skipped[why] = skipped.get(why, 0) + n

    def linear(src: str, dst: str):
        yield f"{dst}.weight", tp_shard_key(f"{dst}.weight", reader.get(f"{src}.weight"), config, tp)
        # fp8 linears declare the e8m0 block scale under the quant method's role name
        if reader.has(f"{src}.scale"):
            yield f"{dst}.weight_scale_inv", tp_shard_key(
                f"{dst}.weight_scale_inv", reader.get(f"{src}.scale"), config, tp
            )

    try:
        yield "model.embed.weight", tp_shard_key("model.embed.weight", reader.get("embed.weight"), config, tp)
        yield "model.norm.weight", reader.get("norm.weight")
        yield "model.head.weight", tp_shard_key("model.head.weight", reader.get("head.weight"), config, tp)

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            m = f"model.{a}"
            # indexer tensors on the index-source layers (wq_b fp8-quantized,
            # weights_proj/wk/k_norm bf16); non-source layers have none
            if L in args.indexer_layer_ids:
                ix = f"{a}.indexer"
                im = f"{m}.indexer"
                yield from linear(f"{ix}.wq_b", f"{im}.wq_b")
                yield f"{im}.weights_proj.weight", reader.get(f"{ix}.weights_proj.weight")
                if reader.has(f"{ix}.wk.weight"):
                    yield f"{im}.wk.weight", reader.get(f"{ix}.wk.weight")
                    yield f"{im}.k_norm.weight", reader.get(f"{ix}.k_norm.weight")
            # engram: wkv (fp8-quantized linear) + q/k gate weights through the
            # state dict; the ~189 GiB tables are read IN PLACE from the shards
            # by the Engram module (O_DIRECT pread) — never yield those
            if L in args.engram_layer_ids:
                eg = f"layers.{L}.engram"  # NOT under attn/ (inventory naming)
                em = f"model.layers.{L}.engram"
                yield from linear(f"{eg}.wkv", f"{em}.wkv")
                yield f"{em}.q_weight", reader.get(f"{eg}.q_weight")
                yield f"{em}.k_weight", reader.get(f"{eg}.k_weight")
                if os.environ.get("FREETOKEN_EXPORT_ENGRAM") == "1":
                    # Converter opt-in: carry the ~189 GiB n-gram table inside the FTW
                    # (otherwise it is read in place from the source shards at serve time).
                    for key in sorted(k for k in weight_map if k.startswith(f"{eg}.embed.")):
                        yield f"model.{key}", reader.get(key)
                else:
                    skip(f"{eg}.embed.", "engram table (O_DIRECT in-place)")
            # compressor weights exist only on the kv-source layers (2/8/14: wkv+wgate
            # at ratio 2, 20: wkv only at ratio 1 — V41Compressor declares wgate only
            # for ratio 2, so the ratio-1 layer must not yield a wgate). They are
            # stored bf16 in the checkpoint (no .scale sidecars) and serve unquantized.
            if L in args.kv_source_layer_ids:
                c = f"{a}.compressor"
                cm = f"{m}.compressor"
                yield f"{cm}.wkv.weight", reader.get(f"{c}.wkv.weight")
                if reader.has(f"{c}.wgate.weight"):
                    yield f"{cm}.wgate.weight", reader.get(f"{c}.wgate.weight")
                yield f"{cm}.norm.weight", reader.get(f"{c}.norm.weight")
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", reader.get(f"{a}.q_norm.weight")  # replicated
            yield from linear(f"{a}.wq_b", f"{m}.wq_b")
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", reader.get(f"{a}.kv_norm.weight")
            # wo_a: FP8 32x32 blocks in the checkpoint, dequantized to bf16 (reference einsum)
            yield f"{m}.wo_a", tp_shard_key(
                f"{m}.wo_a",
                _dequant_fp8_block(reader.get(f"{a}.wo_a.weight"), reader.get(f"{a}.wo_a.scale"), _WO_A_BLOCK),
                config, tp,
            )
            yield from linear(f"{a}.wo_b", f"{m}.wo_b")
            yield f"{m}.attn_sink", reader.get(f"{a}.attn_sink")

            yield f"model.layers.{L}.attn_norm.weight", reader.get(f"layers.{L}.attn_norm.weight")
            yield f"model.layers.{L}.ffn_norm.weight", reader.get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"model.{g}.weight", reader.get(f"{g}.weight")
            yield f"model.{g}.bias", reader.get(f"{g}.bias")
            # image-span routing bias: text-only serving never consults it (inert param)
            yield f"model.{g}.bias_vl", reader.get(f"{g}.bias_vl")
            for proj in ("w1", "w2", "w3"):
                src = f"layers.{L}.ffn.shared_experts.{proj}"
                yield from linear(src, f"model.{src}")

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"model.layers.{L}.{nm}", reader.get(f"layers.{L}.{nm}")

            if L % 8 == 7 and _is_primary():
                logger.info("deepseek_v41: resident weights through layer %d/%d", L + 1, args.n_layers)

        # DSpark MTP draft head (mtp.{0,1,2}.*): the resident (non-expert) tensors mirror
        # the main layers minus the compressor/indexer. Gated on FREETOKEN_ENABLE_MTP so
        # text-only / non-spec serving is unaffected; the flags in convert_checkpoint set
        # it. The draft's 3x128 fp4 expert banks are packed separately (stage 2).
        if args.mtp_enabled:
            for i in range(args.num_nextn_predict_layers):
                s, d = f"mtp.{i}", f"model.mtp.{i}"
                a, m = f"{s}.attn", f"{d}.attn"
                yield from linear(f"{a}.wq_a", f"{m}.wq_a")
                yield f"{m}.q_norm.weight", reader.get(f"{a}.q_norm.weight")
                yield from linear(f"{a}.wq_b", f"{m}.wq_b")
                yield from linear(f"{a}.wkv", f"{m}.wkv")
                yield f"{m}.kv_norm.weight", reader.get(f"{a}.kv_norm.weight")
                yield f"{m}.wo_a", tp_shard_key(
                    f"{m}.wo_a",
                    _dequant_fp8_block(reader.get(f"{a}.wo_a.weight"), reader.get(f"{a}.wo_a.scale"), _WO_A_BLOCK),
                    config, tp,
                )
                yield from linear(f"{a}.wo_b", f"{m}.wo_b")
                yield f"{m}.attn_sink", reader.get(f"{a}.attn_sink")
                yield f"{d}.attn_norm.weight", reader.get(f"{s}.attn_norm.weight")
                yield f"{d}.ffn_norm.weight", reader.get(f"{s}.ffn_norm.weight")
                gg = f"{s}.ffn.gate"
                yield f"model.{gg}.weight", reader.get(f"{gg}.weight")
                yield f"model.{gg}.bias", reader.get(f"{gg}.bias")
                yield f"model.{gg}.bias_vl", reader.get(f"{gg}.bias_vl")
                for proj in ("w1", "w2", "w3"):
                    src = f"{s}.ffn.shared_experts.{proj}"
                    yield from linear(src, f"model.{src}")
                for nm in (
                    "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                    "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
                ):
                    yield f"{d}.{nm}", reader.get(f"{s}.{nm}")
                if i == 0:
                    yield from linear(f"{s}.main_proj", f"{d}.main_proj")
                    yield f"{d}.main_norm.weight", reader.get(f"{s}.main_norm.weight")
                if i == args.num_nextn_predict_layers - 1:
                    yield f"{d}.norm.weight", reader.get(f"{s}.norm.weight")
                    yield f"{d}.markov_head.embed.weight", tp_shard_key(
                        f"{d}.markov_head.embed.weight", reader.get(f"{s}.markov_head.embed.weight"), config, tp
                    )
                    yield f"{d}.markov_head.head.weight", tp_shard_key(
                        f"{d}.markov_head.head.weight", reader.get(f"{s}.markov_head.head.weight"), config, tp
                    )
                    yield f"{d}.confidence_head.proj.weight", reader.get(f"{s}.confidence_head.proj.weight")
                    if reader.has(f"{s}.confidence_head.proj.bias"):
                        yield f"{d}.confidence_head.proj.bias", reader.get(f"{s}.confidence_head.proj.bias")
            for i in range(args.num_nextn_predict_layers):
                skip(f"mtp.{i}.embed.", "aliased to the target embed/head")
                skip(f"mtp.{i}.head.", "aliased to the target embed/head")
                skip(f"mtp.{i}.ffn.experts.", "draft expert banks (stage 2)")
        else:
            skip("mtp.", "DSpark draft (not included; use --include-dspark)")
        skip("vision.", "out of scope (vision tower)")
        skip("aligner.", "out of scope (vision aligner)")
        if _is_primary():
            for why, n in sorted(skipped.items()):
                logger.info("deepseek_v41: skipped %d tensors (%s)", n, why)
    finally:
        reader.close()


def iter_expert_pieces(
    model_path: str, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20
):
    """Routed experts, one piece per expert: ``{gate, up, down}`` e2m1 pairs and
    their e8m0 ``_scale`` companions (``w1`` / ``w3`` / ``w2``), TP-sliced to the
    rank's intermediate band. The DSpark draft's experts (``mtp.``) are skipped."""
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    # served from an FTW dir: the expert tensors live in the RAW checkpoint
    # (the FTW carries only dense weights + repacked banks, and the TP2 bank
    # path reads those directly — the raw dir backs the serial TP2 build)
    model_path = _raw_expert_dir(model_path)
    args = load_args(cached_load_hf_config(model_path), max_batch_size=1)
    L, E = args.n_layers, args.n_routed_experts
    _, _I_loc, lo, hi = _tp_expert_geometry(config)

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) >= L:
            return None
        return int(m["layer"]), int(m["expert"]), _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]]

    def _sliced(names_tensors):
        for name, tensor in names_tensors:
            hit = locate(name)
            if hit is None:
                continue
            _layer, _expert, role = hit
            yield name, _tp_slice_piece(role, tensor, lo, hi)

    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(_sliced(tensors), locate, tensors_per_expert=6)

    def _serial():
        reader = _ShardReader(model_path, _weight_map(model_path), torch.device("cpu"))
        try:
            for li in range(L):
                for e in range(E):
                    base = f"layers.{li}.ffn.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_ in ("weight", "scale"):
                            yield f"{base}.{proj}.{kind_}", reader.get(f"{base}.{proj}.{kind_}")
                if li % 8 == 7 and _is_primary():
                    logger.info("deepseek_v41: expert pieces through layer %d/%d", li + 1, L)
        finally:
            reader.close()

    return per_expert_pieces(_sliced(_serial()), locate, tensors_per_expert=6)


def iter_dspark_expert_pieces(
    model_path: str, config, kind: QuantKind, *, num_stages: int, parallel: bool = False,
    workers: int = 8, chunk: int = 8 << 20,
):
    """DSpark draft routed experts, one piece per expert.

    Same per-expert MXFP4 layout as the target (`w1/w3/w2` + e8m0 `scale`), read from
    ``mtp.{stage}.ffn.experts.{e}.*`` with ``bank_layer == stage`` (0..num_stages-1):
    the draft's dedicated offload cache is keyed by the draft stage, not by a global
    layer id. E comes from ``dspark_n_routed_experts`` (128 vs the target's 384), the
    intermediate band slices like the target's (same ``moe_inter_dim``).
    """
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    model_path = _raw_expert_dir(model_path)
    args = load_args(cached_load_hf_config(model_path), max_batch_size=1)
    E = args.dspark_n_routed_experts
    _, _I_loc, lo, hi = _tp_expert_geometry(config)

    def locate(raw_name: str):
        m = _DSPARK_EXPERT_RE.match(raw_name)
        if m is None or int(m["stage"]) >= num_stages or int(m["expert"]) >= E:
            return None
        return int(m["stage"]), int(m["expert"]), _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]]

    def _sliced(names_tensors):
        for name, tensor in names_tensors:
            hit = locate(name)
            if hit is None:
                continue
            _stage, _expert, role = hit
            yield name, _tp_slice_piece(role, tensor, lo, hi)

    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(_sliced(tensors), locate, tensors_per_expert=6)

    def _serial():
        reader = _ShardReader(model_path, _weight_map(model_path), torch.device("cpu"))
        try:
            for stage in range(num_stages):
                for e in range(E):
                    base = f"mtp.{stage}.ffn.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_ in ("weight", "scale"):
                            yield f"{base}.{proj}.{kind_}", reader.get(f"{base}.{proj}.{kind_}")
        finally:
            reader.close()

    return per_expert_pieces(_sliced(_serial()), locate, tensors_per_expert=6)


def load_dspark_banks(
    model_path: str, model_config, *, method, num_stages: int, device, dtype,
    dummy: bool = False, parallel: bool | None = None, workers: int = 8,
    chunk: int = 8 << 20, layer_sink=None, layer_residency: list[str] | None = None,
):
    """The DSpark draft's routed experts as their own :class:`ExpertBanks`.

    The draft's expert count differs from the target's (128 vs 384), so it uses a
    dedicated ``ExpertBanks``/``OffloadMoeCache``; its FTW banks live under
    ``DSPARK_BANK_KIND`` and its raw pieces under ``mtp.{stage}.ffn.experts.*``.
    ``method`` is the draft's own expert method (from :func:`dspark_expert_method`).
    """
    from freetoken.checkpoint.ftw import (
        DSPARK_BANK_KIND, DSPARK_BANK_NUM_LAYERS, is_ftw_checkpoint, load_ftw_banks,
    )
    from freetoken.moe.expert_banks import build_expert_banks

    if method.kind is not QuantKind.MXFP4:
        raise ValueError(
            f"DSpark draft experts are MXFP4 only, but the draft binds {method.kind!r}"
        )
    if dummy:
        return build_expert_banks(method, num_stages, None, device=device, dummy=True)
    if is_ftw_checkpoint(model_path):
        banks = load_ftw_banks(
            model_path, num_layers=num_stages, workers=workers, chunk=chunk,
            layer_residency=layer_residency, model_config=model_config,
            kind=DSPARK_BANK_KIND, num_layers_meta_key=DSPARK_BANK_NUM_LAYERS,
        )
        if banks is None:
            # The FTW was converted without the draft banks; reading the raw checkpoint
            # would only work where it still exists, so fail with the actual remedy.
            raise ValueError(
                f"{model_path!r} is an FTW checkpoint without DSpark draft banks "
                f"(kind {DSPARK_BANK_KIND!r}); reconvert it with --include-dspark"
            )
        return banks
    pieces = iter_dspark_expert_pieces(
        model_path, model_config, method.kind, num_stages=num_stages,
        parallel=bool(parallel), workers=workers, chunk=chunk,
    )
    return build_expert_banks(method, num_stages, pieces, device=device, layer_sink=layer_sink)


def dspark_expert_method(model_config):
    """The DSpark draft's shared offload expert method, from a meta build of the exact
    draft the engine builds (the packed banks must match the draft's kernel layout).

    ``None`` when the model ships no draft. The target and the draft each own a method
    instance (their expert counts differ); this is the draft one.
    """
    from .dspark import DSparkDraft

    args = model_config.dsv4_args
    if not getattr(args, "num_nextn_predict_layers", 0) or not getattr(args, "dspark_n_routed_experts", 0):
        return None
    with torch.device("meta"):
        draft = DSparkDraft(
            model_config, args, strategy=getattr(model_config, "moe_strategy", "offload"),
            decode_target=getattr(model_config, "decode_target", "gpu"),
            quant_config=model_config.quant,
        )
    return draft.layers.op_list[0].ffn.experts.quant_method


@lru_cache(maxsize=8)
def _ftw_shard_config(model_path: str):
    """(parsed config, spec) for FTW TP sharding, cached per checkpoint (one parse per
    load; rebuilding it per dense tensor dominated the FTW load at TP>1)."""
    return _spec_for_model_path(model_path)


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """TP-shard one FTW-replayed dense tensor (the FTW stores TP-agnostic weights;
    the loader applies the same key rules as the HF-path iter_weights)."""
    config, _spec = _ftw_shard_config(model_path)
    return tp_shard_key(name, w, config, tp)


__all__ = [
    "iter_dspark_expert_pieces", "iter_expert_pieces", "iter_weights",
    "load_dspark_banks", "dspark_expert_method", "shard_ftw_weight",
]
