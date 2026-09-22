"""DeepSeek-V4.1-Flash family args.

All geometry lives in the HF ``config.json`` (``text_config`` sub-dict; the
checkpoint is plain HF-style, unlike V4-Flash whose authoritative args sit in
``inference/config.json``). :func:`load_args` reads them off the transformers
config object into a plain dataclass so the family never pokes hf_config
attributes again.

Phase 1 (the DSV4.1 plan): config + skeleton only. Engram / DSpark /
vision fields are parsed and carried inertly — the engine never touches them
until their phases land.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _text(hf_config: Any) -> Any:
    """The text tower config: nested ``text_config`` when present (V4.1 ships a
    multimodal wrapper), else the config itself (raw/text-only dialects)."""
    text = getattr(hf_config, "text_config", None)
    return text if text is not None else hf_config


def _get(src: Any, name: str, default: Any = None) -> Any:
    val = getattr(src, name, None)
    if val is None and isinstance(src, dict):
        val = src.get(name)
    return default if val is None else val


@dataclass
class DeepseekV41Args:
    """V4.1 geometry. Field names mirror the checkpoint's config.json keys."""

    # backbone
    vocab_size: int = 129280
    hidden_size: int = 5120
    n_layers: int = 40
    n_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    norm_eps: float = 1e-20
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    # rope (YaRN x16 over 64k -> 1M)
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    max_position_embeddings: int = 1048576
    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 2304
    score_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    # CSA2 sparse attention (Phase 3); carried inertly in Phase 1
    sliding_window: int = 128
    # layers owning an indexer (own scorer) — backend routing
    indexer_layer_ids: tuple[int, ...] = ()
    # layers owning an INDEXER K CACHE (the kv-source indexers) — pool sizing
    indexer_k_layer_ids: tuple[int, ...] = (),
    compress_ratios: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # manifold-constrained hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # Engram n-gram memory (Phase 4)
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 0
    engram_compressed_vocab_size: int = 0
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    # DSpark draft (Phase 6)
    num_nextn_predict_layers: int = 0
    # Whether the DSpark draft head is built/served (opt-in via FREETOKEN_ENABLE_MTP,
    # the same gate the weight reader uses). Set in ``load_args``; False for text-only
    # serving so the draft is never constructed.
    mtp_enabled: bool = False
    dspark_block_size: int = 5
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    dspark_markov_rank: int = 0
    dspark_noise_token_id: int = 0

    # checkpoint path (the engram reads its tables in place from the shards)
    model_path: str = ""
    # runtime knobs the engine/engine-adjacent code may override
    max_batch_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)
    # DSV4-pool plumbing alias: _adjust_dsv4_config writes the resolved runtime
    # ceiling here (the pool/cost model read it back); 0 = unset.
    _max_seq_len: int = 0

    @property
    def window_size(self) -> int:
        """The SWA window (DSV4 machinery duck type — it keys the page geometry)."""
        return self.sliding_window

    @property
    def dim(self) -> int:
        return self.hidden_size

    @property
    def rope_head_dim(self) -> int:
        return self.qk_rope_head_dim

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len if self._max_seq_len else self.max_position_embeddings

    @max_seq_len.setter
    def max_seq_len(self, value: int) -> None:
        self._max_seq_len = int(value)

    def __post_init__(self) -> None:
        for name in (
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "engram_layer_ids",
            "engram_num_embeddings",
            "dspark_target_layer_ids",
        ):
            val = getattr(self, name)
            if isinstance(val, list):
                setattr(self, name, tuple(val))


def load_args(hf_config: Any, **overrides: Any) -> DeepseekV41Args:
    """Build the args bag from the transformers config (text_config required —
    the V4.1 checkpoint always ships the multimodal wrapper)."""
    if getattr(hf_config, "text_config", None) is None:
        raise ValueError(
            "DeepSeek-V4.1 parse_config requires the checkpoint's text_config "
            "(the multimodal wrapper config); none found"
        )
    src = _text(hf_config)
    top = hf_config
    rope = _get(src, "rope_scaling", {}) or {}

    def top_get(name: str, default: Any = None) -> Any:
        val = _get(top, name)
        return default if val is None else val

    num_nextn = int(_get(src, "num_nextn_predict_layers", 0))
    # Same opt-in gate as the weight reader (weight.py gates ``mtp.*`` on the env):
    # a checkpoint that ships a draft only builds/serves it under FREETOKEN_ENABLE_MTP.
    mtp_enabled = num_nextn > 0 and os.environ.get("FREETOKEN_ENABLE_MTP", "") == "1"

    args = DeepseekV41Args(
        model_path=str(getattr(hf_config, "_name_or_path", "") or getattr(hf_config, "name_or_path", "")),
        vocab_size=int(_get(src, "vocab_size", 129280)),
        hidden_size=int(_get(src, "hidden_size", 5120)),
        n_layers=int(_get(src, "num_hidden_layers", 40)),
        n_heads=int(_get(src, "num_attention_heads", 64)),
        head_dim=int(_get(src, "head_dim", 512)),
        qk_rope_head_dim=int(_get(src, "qk_rope_head_dim", 64)),
        q_lora_rank=int(_get(src, "q_lora_rank", 1280)),
        o_lora_rank=int(_get(src, "o_lora_rank", 1024)),
        o_groups=int(_get(src, "o_groups", 8)),
        norm_eps=float(_get(src, "rms_norm_eps", 1e-20)),
        hidden_act=str(_get(src, "hidden_act", "silu")),
        swiglu_limit=float(_get(src, "swiglu_limit", 10.0)),
        rope_theta=float(_get(src, "rope_theta", 10000.0)),
        rope_factor=float(rope.get("factor", 16.0)),
        beta_fast=int(rope.get("beta_fast", 32)),
        beta_slow=int(rope.get("beta_slow", 1)),
        original_seq_len=int(
            rope.get("original_max_position_embeddings", 65536)
        ),
        max_position_embeddings=int(top_get("max_position_embeddings", 0) or _get(src, "max_position_embeddings", 1048576)),
        n_routed_experts=int(_get(src, "n_routed_experts", 384)),
        n_shared_experts=int(_get(src, "n_shared_experts", 1)),
        n_activated_experts=int(_get(src, "num_experts_per_tok", 6)),
        moe_inter_dim=int(_get(src, "moe_intermediate_size", 2304)),
        score_func=str(_get(src, "scoring_func", "sqrtsoftplus")),
        topk_method=str(_get(src, "topk_method", "noaux_tc")),
        norm_topk_prob=bool(_get(src, "norm_topk_prob", True)),
        route_scale=float(_get(src, "routed_scaling_factor", 1.5)),
        sliding_window=int(_get(src, "sliding_window", 128)),
        indexer_layer_ids=tuple(int(i) for i in (_get(src, "index_source_layer_ids", ()) or ())),
        indexer_k_layer_ids=tuple(int(i) for i in (_get(src, "kv_source_layer_ids", ()) or ())),
        compress_ratios=tuple(int(r) for r in (_get(src, "compress_ratios", ()) or ())),
        compress_rope_theta=float(_get(src, "compress_rope_theta", 160000.0)),
        kv_source_layer_ids=tuple(int(i) for i in (_get(src, "kv_source_layer_ids", ()) or ())),
        index_source_layer_ids=tuple(int(i) for i in (_get(src, "index_source_layer_ids", ()) or ())),
        index_n_heads=int(_get(src, "index_n_heads", 32)),
        index_head_dim=int(_get(src, "index_head_dim", 128)),
        index_topk=int(_get(src, "index_topk", 512)),
        candidate_source_layer_id=int(_get(src, "candidate_source_layer_id", 20)),
        candidate_topk_blocks=int(_get(src, "candidate_topk_blocks", 2048)),
        candidate_block_size=int(_get(src, "candidate_block_size", 8)),
        hc_mult=int(_get(src, "hc_mult", 4)),
        hc_sinkhorn_iters=int(_get(src, "hc_sinkhorn_iters", 20)),
        hc_eps=float(_get(src, "hc_eps", 1e-6)),
        engram_layer_ids=tuple(int(i) for i in (_get(src, "engram_layer_ids", ()) or ())),
        engram_num_embeddings=tuple(int(n) for n in (_get(src, "engram_num_embeddings", ()) or ())),
        engram_max_ngram_size=int(_get(src, "engram_max_ngram_size", 4)),
        engram_vocab_size=int(_get(src, "engram_vocab_size", 0)),
        engram_compressed_vocab_size=int(_get(src, "engram_compressed_vocab_size", 0)),
        engram_n_heads=int(_get(src, "engram_n_heads", 8)),
        engram_head_dim=int(_get(src, "engram_head_dim", 256)),
        engram_pad_token_id=int(_get(src, "engram_pad_token_id", 2)),
        num_nextn_predict_layers=num_nextn,
        mtp_enabled=mtp_enabled,
        dspark_block_size=int(_get(src, "dspark_block_size", 5)),
        dspark_target_layer_ids=tuple(int(i) for i in (_get(src, "dspark_target_layer_ids", ()) or ())),
        dspark_n_routed_experts=int(_get(src, "dspark_n_routed_experts", 128)),
        dspark_num_experts_per_tok=int(_get(src, "dspark_num_experts_per_tok", 3)),
        dspark_markov_rank=int(_get(src, "dspark_markov_rank", 0)),
        dspark_noise_token_id=int(_get(src, "dspark_noise_token_id", 0)),
        **overrides,
    )
    return args


__all__ = ["DeepseekV41Args", "load_args"]
