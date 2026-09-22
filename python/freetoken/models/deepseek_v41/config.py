"""Engine-facing config for DeepSeek-V4.1-Flash.

``parse_config`` maps the transformer fields the engine needs into
:class:`ModelConfig` and carries the full :class:`DeepseekV41Args` in
``ModelConfig.dsv41_args`` (engram / DSpark / CSA2 geometry rides along inertly
until their phases land — the DSV4.1 plan).

Phase 3 attention: the DSV4 machinery (paged window ring + per-kv-source
compressed caches, ratios 0/1/2) via ``DSV4AttentionGroupConfig`` and
``ModelConfig.dsv4_args`` — the engine's DSV4 plumbing (pool class, sizing,
_adjust_dsv4_config) reads the dsv4_args payload, so V4.1 rides it directly.
``dsv41_args`` is the same object under the family's name. The learned
indexer (top-512 + candidate filter) is Phase 3b; Phase 3a selects compressed
candidates statically (all blocks — V4's own no-indexer mechanism).
"""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    DSV4AttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    args = load_args(hf_config)

    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }
    rotary = RotaryConfig(
        head_dim=args.head_dim,
        # the module owns its rope (trailing 64 dims, two regimes); the group's
        # rotary_config stays for the pool/engine surface
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position_embeddings,
        base=args.rope_theta,
        scaling=rope_scaling,
    )
    layer_ids = tuple(range(args.n_layers))

    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # MLA-style latent (K == V)
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act=args.hidden_act,
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=rotary,
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        moe_enabled=True,
        expert_quant="mxfp4",
        first_k_dense_replace=0,
        n_shared_experts=args.n_shared_experts,
        shared_expert_intermediate_size=args.moe_inter_dim,
        routed_scaling_factor=args.route_scale,
        attn_sm_scale=args.head_dim**-0.5,
        # The engine's DSV4 plumbing (pool class, sizing, _adjust_dsv4_config) reads the
        # dsv4_args payload; dsv41_args is the same bag under the family's name.
        dsv4_args=args,
        dsv41_args=args,
        mtp_enabled=args.mtp_enabled,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv41",
                layer_ids=layer_ids,
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.sliding_window,
            ),
        ),
    )


__all__ = ["parse_config"]
