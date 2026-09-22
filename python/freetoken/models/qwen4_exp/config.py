from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
)


@dataclass(frozen=True)
class Qwen4VisionConfig:
    """Qwen3.8-Flash-Next vision tower geometry (model_type qwen4_exp_vision)."""

    hidden_size: int
    intermediate_size: int
    depth: int
    num_heads: int
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _parse_vision_config(top_cfg: Any) -> Qwen4VisionConfig | None:
    """The vision tower config, or None when the checkpoint has none / vision is disabled.

    Vision is opt-in (``FREETOKEN_LOAD_VISION`` / ``--load-vision``): the tower is ~1 GiB
    of resident bf16 weights text-only serving never touches, so it is neither built nor
    loaded unless requested."""
    from freetoken.models.config import vision_load_enabled

    if not vision_load_enabled():
        return None
    vc = getattr(top_cfg, "vision_config", None)
    if vc is None:
        return None
    rope = getattr(vc, "rope_parameters", None) or {}
    depth = getattr(vc, "depth", None)
    if depth is None:
        depth = getattr(vc, "num_hidden_layers")
    num_heads = getattr(vc, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(vc, "num_attention_heads")
    return Qwen4VisionConfig(
        hidden_size=int(vc.hidden_size),
        intermediate_size=int(vc.intermediate_size),
        depth=int(depth),
        num_heads=int(num_heads),
        patch_size=int(vc.patch_size),
        temporal_patch_size=int(vc.temporal_patch_size),
        in_channels=int(getattr(vc, "in_channels", 3)),
        spatial_merge_size=int(vc.spatial_merge_size),
        out_hidden_size=int(vc.out_hidden_size),
        num_position_embeddings=int(vc.num_position_embeddings),
        rope_theta=float(rope.get("rope_theta", 10000.0)),
    )


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry beyond the generic ModelConfig fields (ModelConfig.qwen4_args)."""

    hidden_size: int
    # Hyper-connections: every layer reads/writes hc_count residual streams [T, hc_count*hidden].
    hc_count: int
    hc_lowrank: int
    # PLE n-gram embedding; layer ids are zero-based decoder layers.
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    # n-gram hash windows never cross this token (the eos id); they restart after it.
    ngram_boundary_token_id: int
    # QSA indexer scoring geometry (the slab/ratio geometry lives on the attention group).
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int
    # MTP draft head (mtp.* checkpoint keys): the layer count the checkpoint ships and
    # whether FreeToken builds it. Opt-in via FREETOKEN_ENABLE_MTP=1 until the draft
    # loop is wired; when disabled, iter_weights keeps dropping the mtp.* tensors.
    mtp_num_layers: int = 0
    mtp_enabled: bool = False

    @property
    def index_topk_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    @property
    def num_ngram_heads(self) -> int:
        # one head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram)
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        # HF Qwen4ExpTextPLELayer sets the depthwise conv dilation to ngram_size
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def ple_state_width(self) -> int:
        return self.hc_count * self.hidden_size


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"


def ple_slot_states(args: Qwen4ExpArgs) -> Tuple[SlotStateSpec, ...]:
    """Per-request PLE state riding the linear-state slots (see LinearStatePool.slot_states)."""
    if not args.ple_layer_ids:
        return ()
    return (
        # dilated-conv left context; replicated (not TP-sharded), model dtype
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(args.ple_state_width, args.ple_conv_state_len),
            layer_ids=args.ple_layer_ids,
        ),
        # last ngram_size-1 token ids, shared by every PLE layer; eos = hash boundary
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(args.ngram_size - 1,),
            dtype=torch.int32,
            fill_value=float(args.ngram_boundary_token_id),
        ),
    )


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        # HF Qwen4ExpTextConfig rewrites full_attention to qwen_sparse_attention in __post_init__.
        return [
            "full_attention" if t == "qwen_sparse_attention" else t for t in layer_types
        ]
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    # int(), not round(): HF configuration_qwen4_exp truncates head_dim * partial.
    rotary_dim = int(head_dim * partial)

    # Text-only serving with the default rope type: the mRoPE sections reduce to standard
    # partial rope, and the unhashable ``mrope_section`` list must not reach get_rope's
    # cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    # Quant roles come from the shared dialect layer (the same resolution the engine
    # uses), not a second detector: FP8 block builds quantize only the routed experts;
    # compressed-tensors NVFP4 / modelopt exports follow their ignore/target lists.
    from freetoken.models.quant import quant_config_for, role_quants
    from freetoken.models.register import get_model_spec

    _spec = get_model_spec(hf_config.architectures[0])
    _roles = role_quants(
        quant_config_for(hf_config, _spec),
        expert="model.layers.0.mlp.experts.0.gate_proj",
        attn="model.layers.0.self_attn.q_proj",
        dense="model.layers.0.mlp.shared_expert.gate_proj",
        lm_head="lm_head",
    )
    expert_quant = _roles["expert_quant"]
    attn_quant = _roles["attn_quant"]
    dense_quant = _roles["dense_quant"]
    lm_head_quant = _roles["lm_head_quant"]

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # HF stores ple_layer_ids one-indexed (validated upstream as [1, num_layers]).
    ple_layer_ids = tuple(int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or ()))
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    # MTP draft layers ride the SAME QSA pool/backend as the full-attention layers:
    # their layer ids continue after the main stack (mtp.py builds blocks with
    # layer_id = num_layers + i), so the group's id -> slab-slot map must include
    # them or both the backend (_idx_slot) and the pool (_dense) raise on layer 48.
    # The extra slab/ring depth (num_index_layers) costs VRAM only when mtp_enabled.
    _mtp_num_layers = int(getattr(text, "mtp_num_hidden_layers", 0) or 0)
    _mtp_enabled = _mtp_num_layers > 0 and os.environ.get("FREETOKEN_ENABLE_MTP", "") == "1"
    mtp_ids = tuple(
        range(text.num_hidden_layers, text.num_hidden_layers + _mtp_num_layers)
    ) if _mtp_enabled else ()
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids + mtp_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids) + len(mtp_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        # HF resolves a null output_gate_type to hidden_act; mirror that instead of
        # stringifying None.
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    num_experts = int(getattr(text, "num_experts", 0) or 0)

    # HF accepts int | list here and uses the first entry (modeling_qwen4_exp Qwen4ExpTextNGramEmbedding)
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    qwen4_args = Qwen4ExpArgs(
        hidden_size=text.hidden_size,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        make_ngram_vocab_size_divisible_by=int(text.make_ngram_vocab_size_divisible_by),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        index_n_heads=int(text.indexer_n_heads),
        index_kv_heads=int(text.indexer_kv_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
        mtp_num_layers=_mtp_num_layers,
        mtp_enabled=_mtp_enabled,
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(
            getattr(text, "shared_expert_intermediate_size", 0) or 0
        ),
        # Absent from the shipped config; HF Qwen4ExpTextConfig defaults it True and the
        # Qwen3_5MoE block renormalizes unconditionally -- keep the two in agreement.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=num_experts > 0,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=_parse_vision_config(hf_config),  # None unless --load-vision
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        qwen4_args=qwen4_args,
        mtp_enabled=_mtp_enabled,
        slot_states=ple_slot_states(qwen4_args),
    )


__all__ = ["PLE_CONV_STATE", "PLE_NGRAM_STATE", "Qwen4ExpArgs", "parse_config", "ple_slot_states"]
