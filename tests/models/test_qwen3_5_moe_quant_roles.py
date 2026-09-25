"""qwen3_5_moe quant roles now come from the shared dialect layer (CPU-only).

These minimal configs mirror the real checkpoints' ``quantization_config`` (the mapping
was characterized against the actual config.json of the 9 shipping Qwen3.5/3.6/3.8
variants and reproduces the previous bespoke detector exactly).
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.models.qwen3_5_moe.config import parse_config

_LM = "model.language_model.layers.0"


def _text(num_experts: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=4, hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
        head_dim=64, rope_parameters={"rope_theta": 10000.0}, num_experts=num_experts,
        num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        hidden_act="silu", rms_norm_eps=1e-6, tie_word_embeddings=False, vocab_size=512,
        intermediate_size=256, max_position_embeddings=4096, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )


def _hf(quant, num_experts: int = 0):
    arch = "Qwen3_5MoeForConditionalGeneration" if num_experts else "Qwen3_5ForConditionalGeneration"
    return SimpleNamespace(architectures=[arch], text_config=_text(num_experts), quantization_config=quant)


def _roles(quant, num_experts: int = 0):
    c = parse_config(_hf(quant, num_experts))
    return c.expert_quant, c.attn_quant, c.dense_quant, c.lm_head_quant, c.weight_block_size


def test_unquantized():
    assert _roles(None, num_experts=8) == ("none", "none", "none", "none", None)


def test_official_block_fp8():
    # Qwen/Qwen3.6-35B-A3B-FP8: block-fp8 on every projection; the dense roles are owned
    # by the fp8 reader and report none.
    quant = {"quant_method": "fp8", "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]}
    assert _roles(quant, num_experts=8) == ("fp8_block", "none", "none", "none", (128, 128))


def test_modelopt_moe_nvfp4_mixed():
    # nvidia/Qwen3.6-35B-A3B-NVFP4: NVFP4 experts + shared expert, FP8 attention, NVFP4 lm_head.
    quant = {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "ignore": ["mtp*"],
        "quantized_layers": {
            f"{_LM}.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            f"{_LM}.mlp.shared_expert.gate_proj": {"quant_algo": "W4A16_NVFP4"},
            f"{_LM}.mlp.shared_expert.up_proj": {"quant_algo": "W4A16_NVFP4"},
            f"{_LM}.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            f"{_LM}.linear_attn.in_proj_z": {"quant_algo": "FP8"},
            "lm_head": {"quant_algo": "W4A16_NVFP4"},
        },
    }
    assert _roles(quant, num_experts=8) == ("nvfp4", "fp8_pertensor", "nvfp4", "nvfp4", None)


def test_modelopt_dense_nvfp4_mixed():
    # RadixArk/Qwen3.8-27B-NVFP4 (dense): NVFP4 dense MLP + lm_head, FP8 attention.
    quant = {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "ignore": ["mtp*"],
        "quantized_layers": {
            f"{_LM}.mlp.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
            f"{_LM}.mlp.up_proj": {"quant_algo": "NVFP4", "group_size": 16},
            f"{_LM}.mlp.down_proj": {"quant_algo": "NVFP4", "group_size": 16},
            f"{_LM}.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            f"{_LM}.linear_attn.in_proj_z": {"quant_algo": "FP8"},
            "lm_head": {"quant_algo": "NVFP4", "group_size": 16},
        },
    }
    assert _roles(quant, num_experts=0) == ("none", "fp8_pertensor", "nvfp4", "nvfp4", None)


def test_compressed_tensors_nvfp4_dense():
    # llm-compressor W4A16 dense export: all Linears NVFP4, lm_head ignored.
    quant = {
        "quant_method": "compressed-tensors", "ignore": ["lm_head"],
        "config_groups": {"g": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16},
        }},
    }
    assert _roles(quant, num_experts=0) == ("none", "nvfp4", "nvfp4", "none", None)


def test_gdn_split_fuse_matches_the_model_buffers():
    # A hybrid compressed-tensors checkpoint (FP8 qkv|z, bf16 b|a) makes the model build the
    # split GDN buffers (in_proj_qkvz fp8 + in_proj_ba bf16). The loader must fuse the bf16
    # b|a into in_proj_ba, not the joint in_proj, which then never completes and the engine
    # fails to load with "Incomplete bf16 fusions" (Qwen3.8-27B-NVFP4).
    import torch

    from freetoken.models.qwen3_5_moe import weight as W

    class _Quant:
        def __init__(self, fp8: bool):
            self._fp8 = fp8

        def scheme_for(self, prefix):
            return object() if self._fp8 and prefix.endswith(".linear_attn.in_proj_qkvz") else None

    split = _Quant(fp8=True)
    assert W._gdn_split(split, "model.layers.0.linear_attn.in_proj_b") is True
    assert W._gdn_split(split, "model.layers.0.linear_attn.in_proj_qkv") is True
    assert W._gdn_split(split, "model.layers.0.self_attn.q_proj") is False
    # a fully bf16 GDN keeps the joint in_proj fusion
    assert W._gdn_split(_Quant(fp8=False), "model.layers.0.linear_attn.in_proj_b") is False

    buf = {}
    assert W._ct_bf16_fuse("model.layers.0.linear_attn.in_proj_b", torch.zeros(2), buf, W._CT_BF16_SPLIT_FUSE) == []
    out = W._ct_bf16_fuse("model.layers.0.linear_attn.in_proj_a", torch.ones(2), buf, W._CT_BF16_SPLIT_FUSE)
    assert [key for key, _ in out] == ["model.layers.0.linear_attn.in_proj_ba.weight"]
    assert out[0][1].shape == (4,)
    assert not buf
