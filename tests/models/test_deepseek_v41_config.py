"""deepseek_v41 (DeepSeek-V4.1-Flash) config parsing: registry resolution,
attention groups, MoE/rotary fields, fp8 32x32 quant dialect.

Runs off a trimmed copy of the real checkpoint config.json via RawConfigShim
(exactly what production hits — transformers doesn't know ``deepseek_v41`` yet,
so the engine falls back to the raw-JSON shim)."""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.layers.quantization.configs.fp8 import Fp8BlockConfig
from freetoken.layers.quantization.configs.base import QuantConfig
from freetoken.models.config import DSV4AttentionGroupConfig
from freetoken.models.deepseek_v41.config import parse_config
from freetoken.models.register import get_model_spec
from freetoken.utils.hf import RawConfigShim

_NUM_LAYERS = 40


def _hf_config() -> RawConfigShim:
    # Trimmed from deepseek-ai/DeepSeek-V4.1-Flash config.json (real values).
    return RawConfigShim(
        {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "dtype": "bfloat16",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "image_token_id": 129264,
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            },
            "text_config": {
                "model_type": "deepseek_v41_text",
                "vocab_size": 129280,
                "hidden_size": 5120,
                "moe_intermediate_size": 2304,
                "num_hidden_layers": _NUM_LAYERS,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "q_lora_rank": 1280,
                "o_lora_rank": 1024,
                "o_groups": 8,
                "hidden_act": "silu",
                "swiglu_limit": 10.0,
                "rms_norm_eps": 1e-20,
                "use_cache": True,
                "tie_word_embeddings": False,
                "max_position_embeddings": 1048576,
                "rope_theta": 10000,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 16,
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "original_max_position_embeddings": 65536,
                },
                "n_routed_experts": 384,
                "n_shared_experts": 1,
                "num_experts_per_tok": 6,
                "scoring_func": "sqrtsoftplus",
                "topk_method": "noaux_tc",
                "norm_topk_prob": True,
                "routed_scaling_factor": 1.5,
                "sliding_window": 128,
                "compress_ratios": [0, 0] + [2] * 10 + [1] * 28,
                "compress_rope_theta": 160000,
                "kv_source_layer_ids": [2, 8, 14, 20],
                "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 512,
                "candidate_source_layer_id": 20,
                "candidate_topk_blocks": 2048,
                "candidate_block_size": 8,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1e-06,
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [384006168, 384016682],
                "engram_max_ngram_size": 4,
                "engram_vocab_size": 16000000,
                "engram_n_heads": 8,
                "engram_head_dim": 256,
                "engram_pad_token_id": 2,
                "engram_compressed_vocab_size": 99092,
                "num_nextn_predict_layers": 3,
                "dspark_block_size": 5,
                "dspark_noise_token_id": 128799,
                "dspark_target_layer_ids": [37, 38, 39],
                "dspark_markov_rank": 256,
                "dspark_n_routed_experts": 128,
                "dspark_num_experts_per_tok": 3,
            },
            "vision_config": {
                "model_type": "deepseek_v41_vision",
                "num_hidden_layers": 32,
                "hidden_size": 1024,
            },
        }
    )


def _config():
    return parse_config(_hf_config())


def test_registry_resolves_deepseek_v41():
    spec = get_model_spec("DeepseekV41ForCausalLM")
    assert spec.module == "freetoken.models.deepseek_v41"
    import importlib

    module = importlib.import_module(spec.module)
    assert callable(getattr(module, spec.model_cls))
    assert callable(getattr(module, spec.parse_config))
    assert callable(getattr(module, spec.iter_weights))


def test_parse_config_core_fields():
    cfg = _config()
    assert cfg.num_layers == _NUM_LAYERS
    assert cfg.hidden_size == 5120
    assert cfg.head_dim == 512
    assert cfg.num_qo_heads == 64
    assert cfg.num_kv_heads == 1
    assert cfg.vocab_size == 129280
    assert cfg.moe_enabled is True
    assert cfg.is_moe is True
    assert cfg.num_moe_layers == _NUM_LAYERS  # MoE on every layer
    assert cfg.num_experts == 384
    assert cfg.num_experts_per_tok == 6
    assert cfg.expert_quant == "mxfp4"
    assert cfg.model_type == "deepseek_v41"


def test_parse_config_rotary_and_moe():
    cfg = _config()
    assert cfg.rotary_config.max_position == 1048576
    assert cfg.rotary_config.base == 10000
    assert cfg.rotary_config.scaling["factor"] == 16
    assert cfg.n_shared_experts == 1
    assert cfg.routed_scaling_factor == 1.5
    assert cfg.first_k_dense_replace == 0


def test_attention_group_is_dsv4():
    cfg = _config()
    groups = cfg.attention_groups
    assert len(groups) == 1
    group = groups[0]
    assert isinstance(group, DSV4AttentionGroupConfig)
    assert group.layer_ids == tuple(range(_NUM_LAYERS))
    assert group.num_kv_heads == 1
    assert group.head_dim == 512
    assert group.sliding_window == 128
    for layer_id in (0, 1, 20, _NUM_LAYERS - 1):
        assert cfg.attn_type_for_layer(layer_id) == AttnType.DSV4
    # the engine's DSV4 plumbing reads dsv4_args; the family keeps the same bag
    assert cfg.dsv4_args is cfg.dsv41_args
    assert cfg.dsv4_args.window_size == 128  # pool-geometry duck type
    assert cfg.dsv4_args.max_seq_len == 1048576


def test_dsv41_args_payload():
    cfg = _config()
    args = cfg.dsv41_args
    assert args.hc_mult == 4
    assert args.hc_sinkhorn_iters == 20
    assert args.compress_ratios[0] == 0 and args.compress_ratios[-1] == 1
    assert len(args.compress_ratios) == _NUM_LAYERS
    assert args.kv_source_layer_ids == (2, 8, 14, 20)
    assert args.index_source_layer_ids == (2, 8, 14, 20, 24, 28, 32, 36)
    assert args.engram_layer_ids == (1, 14)
    assert args.engram_compressed_vocab_size == 99092
    assert args.num_nextn_predict_layers == 3
    assert args.dspark_target_layer_ids == (37, 38, 39)
    # lists became tuples in __post_init__
    assert isinstance(args.compress_ratios, tuple)


def test_fp8_32x32_block_dialect():
    # The production entry: checkpoint_quant_config -> QuantConfig.from_hf over
    # the RawConfigShim (quantization_config_of normalizes the wrapped sub-dict).
    quant = QuantConfig.from_hf(_hf_config())
    assert isinstance(quant, Fp8BlockConfig)
    assert quant.block == (32, 32)
    assert quant.e8m0 is True
    assert quant.expert_fp4 is True


def test_missing_text_config_raises():
    with pytest.raises(ValueError, match="text_config"):
        parse_config(RawConfigShim({"architectures": ["DeepseekV41ForCausalLM"]}))
