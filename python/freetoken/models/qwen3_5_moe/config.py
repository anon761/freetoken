from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


def _quant_roles(hf_config: Any) -> tuple[str, str, str, str, tuple[int, int] | None]:
    """``(expert, attn, dense, lm_head, weight_block_size)`` from the shared dialect layer.

    The ``QuantConfig`` the engine uses is the single source of truth; this maps its
    per-module ``QuantKind`` onto the loader/engine role strings, preserving two
    conventions: block-fp8 quantizes *every* projection (so the dense roles report
    ``none`` -- the fp8 reader owns them), and NVFP4 routed experts imply the shared
    expert is native FP4 (``dense="nvfp4"``)."""
    from freetoken.models.quant import quant_config_for, quant_kind_str
    from freetoken.models.register import get_model_spec

    spec = get_model_spec(hf_config.architectures[0])
    qc = quant_config_for(hf_config, spec)

    def kind(prefix: str) -> str:
        try:
            return quant_kind_str(qc, prefix)
        except Exception:  # noqa: BLE001 -- a fused probe that mixes schemes is not an expert/attn role
            return "none"

    num_experts = int(getattr(getattr(hf_config, "text_config", hf_config), "num_experts", 0) or 0)
    if num_experts > 0:
        expert_kind = kind("model.layers.0.mlp.experts")
    elif getattr(qc, "block", None) is not None:
        # dense block-fp8: there are no routed experts, but the role still flags the fp8
        # dense reader (a generic "Linear" group must not leak into the expert role).
        expert_kind = "fp8_block"
    else:
        expert_kind = "none"
    attn_kinds = (
        kind("model.layers.0.self_attn.qkv_proj"),
        kind("model.layers.0.linear_attn.in_proj_qkvz"),
    )
    dense_kind = kind("model.layers.0.mlp.shared_expert.gate_up_proj")
    if dense_kind == "none":
        dense_kind = kind("model.layers.0.mlp.gate_up_proj")
    lm_kind = kind("lm_head")
    block = getattr(qc, "block", None)

    expert = {"fp8_block": "fp8_block", "nvfp4": "nvfp4", "fp8_tensor": "fp8"}.get(expert_kind, "none")
    if expert == "fp8_block":
        # block-fp8 quantizes attention/GDN/shared expert too; their reader ignores these roles.
        return expert, "none", "none", "none", block
    if "nvfp4" in attn_kinds:
        attn = "nvfp4"
    elif "fp8_tensor" in attn_kinds:
        attn = "fp8_pertensor"
    else:
        attn = "none"
    dense = "nvfp4" if expert == "nvfp4" or dense_kind == "nvfp4" else "none"
    lm_head = "nvfp4" if lm_kind == "nvfp4" else "none"
    return expert, attn, dense, lm_head, block


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
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
    rotary_dim = int(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, attn_quant, dense_quant, lm_head_quant, weight_block_size = _quant_roles(hf_config)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # MTP draft head (mtp.* checkpoint keys): its full-attention blocks take layer ids
    # AFTER the main stack, so the full group's id -> KV-slot map must include them
    # (the extra KV slab costs VRAM only when enabled).
    mtp_num_layers = int(getattr(text, "mtp_num_hidden_layers", 0) or 0)
    mtp_enabled = mtp_num_layers > 0 and os.environ.get("FREETOKEN_ENABLE_MTP", "") == "1"
    mtp_ids = (
        tuple(range(text.num_hidden_layers, text.num_hidden_layers + mtp_num_layers))
        if mtp_enabled
        else ()
    )

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids + mtp_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate="silu",
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=None,  # text-only milestone
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        mtp_enabled=mtp_enabled,
        mtp_num_layers=mtp_num_layers,
    )


__all__ = ["parse_config"]
