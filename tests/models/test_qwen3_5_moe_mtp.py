# Copyright (c) 2026 FreeToken contributors
# qwen3_5_moe MTP draft head: config wiring, loader key handling and module shapes.
# CPU-only, tiny dims.
from __future__ import annotations

import os
from types import SimpleNamespace

from freetoken.distributed import set_tp_info
from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTP
from freetoken.models.qwen3_5_moe.weight import _is_gemma_norm, _rename


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info

    info._TP_INFO = None
    set_tp_info(rank, size)


def _text(mtp_layers: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=4, hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
        head_dim=64, rope_parameters={"rope_theta": 10000.0}, num_experts=0,
        num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        hidden_act="silu", rms_norm_eps=1e-6, tie_word_embeddings=False, vocab_size=512,
        intermediate_size=256, max_position_embeddings=4096, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        mtp_num_hidden_layers=mtp_layers,
    )


def _hf(mtp_layers: int = 1):
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=_text(mtp_layers),
        quantization_config=None,
    )


def test_parse_config_wires_the_mtp_layer_ids():
    old = os.environ.get("FREETOKEN_ENABLE_MTP")
    os.environ["FREETOKEN_ENABLE_MTP"] = "1"
    try:
        config = parse_config(_hf(mtp_layers=1))
    finally:
        if old is None:
            os.environ.pop("FREETOKEN_ENABLE_MTP", None)
        else:
            os.environ["FREETOKEN_ENABLE_MTP"] = old
    assert config.mtp_enabled and config.mtp_num_layers == 1
    full = next(g for g in config.attention_groups if g.name == "full")
    # The MTP block's attention slot continues after the main stack.
    assert config.num_layers in full.layer_ids


def test_parse_config_mtp_off_without_env():
    old = os.environ.pop("FREETOKEN_ENABLE_MTP", None)
    try:
        config = parse_config(_hf(mtp_layers=1))
    finally:
        if old is not None:
            os.environ["FREETOKEN_ENABLE_MTP"] = old
    assert not config.mtp_enabled and config.mtp_num_layers == 1
    full = next(g for g in config.attention_groups if g.name == "full")
    assert config.num_layers not in full.layer_ids


def test_rename_keeps_mtp_only_when_enabled():
    assert _rename("mtp.fc.weight", keep_mtp=False) is None
    assert _rename("mtp.fc.weight", keep_mtp=True) == "mtp.fc.weight"
    assert _rename("model.language_model.layers.0.mlp.down_proj.weight") == (
        "model.layers.0.mlp.down_proj.weight"
    )


def test_mtp_norms_are_gemma_ones():
    assert _is_gemma_norm("mtp.norm.weight")
    assert _is_gemma_norm("mtp.pre_fc_norm_embedding.weight")
    assert _is_gemma_norm("mtp.pre_fc_norm_hidden.weight")
    assert _is_gemma_norm("mtp.layers.0.input_layernorm.weight")
    assert _is_gemma_norm("mtp.layers.0.self_attn.q_norm.weight")
    assert not _is_gemma_norm("mtp.fc.weight")


def test_mtp_module_shapes():
    old = os.environ.get("FREETOKEN_ENABLE_MTP")
    os.environ["FREETOKEN_ENABLE_MTP"] = "1"
    try:
        config = parse_config(_hf(mtp_layers=1))
    finally:
        if old is None:
            os.environ.pop("FREETOKEN_ENABLE_MTP", None)
        else:
            os.environ["FREETOKEN_ENABLE_MTP"] = old
    _set_tp(0, 1)
    mtp = Qwen3_5MTP(config)
    h = config.hidden_size
    assert mtp.fc.weight.shape == (h, 2 * h)
    block = mtp.layers.op_list[0]
    assert block.self_attn.qkv_proj.weight.shape == (
        2 * config.num_qo_heads * config.head_dim + 2 * config.num_kv_heads * config.head_dim, h
    )
    assert block.mlp.gate_up_proj.weight.shape == (2 * config.intermediate_size, h)
    assert block.mlp.down_proj.weight.shape == (h, config.intermediate_size)
    assert mtp.norm.weight.shape == (h,)
    _set_tp(0, 1)
