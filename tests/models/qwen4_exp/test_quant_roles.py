"""qwen4_exp quant roles from the shared dialect layer (CPU-only).

``parse_config`` must report the same ``QuantKind`` strings the engine resolves for the
three shipping NVFP4/FP8 variants, instead of a second bespoke detector:
  * official FP8 block build  -> fp8_block experts, everything else bf16
  * compressed-tensors NVFP4  -> nvfp4 experts, everything else bf16
  * modelopt NVFP4 ignore-list -> nvfp4 experts, everything else bf16
"""

from __future__ import annotations

import pytest

from freetoken.models.qwen4_exp.config import parse_config

from .common import hf_config

_DENSE = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.mlp.shared_expert.gate_proj",
    "model.language_model.layers.0.mlp.shared_expert.up_proj",
]


def _cfg(quant):
    cfg = hf_config(num_layers=4, hidden=128, head_dim=64, num_q=4, num_kv=1)
    cfg.quantization_config = quant
    return cfg


def _roles(quant):
    c = parse_config(_cfg(quant))
    return c.expert_quant, c.attn_quant, c.dense_quant, c.lm_head_quant


def test_unquantized_is_all_bf16():
    assert _roles(None) == ("none", "none", "none", "none")


def test_official_fp8_block_quantizes_only_experts():
    quant = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head", *_DENSE],
    }
    assert _roles(quant) == ("fp8_block", "none", "none", "none")


def test_compressed_tensors_nvfp4_group_targets_experts():
    quant = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*mlp\\.experts\\..*proj$"],
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16},
            }
        },
    }
    assert _roles(quant) == ("nvfp4", "none", "none", "none")


def test_modelopt_nvfp4_ignore_list_keeps_dense_bf16():
    quant = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head", *_DENSE]}
    assert _roles(quant) == ("nvfp4", "none", "none", "none")
