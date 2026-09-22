"""ModelOptConfig reads quant_algo/ignore from a nested "quantization" block.

Some exporters (e.g. the RadixArk RVN NVFP4 checkpoints) nest the fields one
level below the top-level quantization_config:
``{"quant_method": "modelopt", "quantization": {"quant_algo": ..., "exclude_modules": [...]}}``.
Flat ModelOpt exports keep working, and top-level keys win over nested ones.
"""
from __future__ import annotations

import pytest

from freetoken.layers.quantization import ModelOptConfig


def test_flat_config_still_works():
    q = {"quant_method": "modelopt", "quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}
    cfg = ModelOptConfig(q)
    assert cfg.algo == "NVFP4"
    assert cfg.ignore("lm_head")
    assert not cfg.ignore("model.layers.0.mlp.experts")


def test_nested_quantization_block_is_read():
    q = {
        "quant_method": "modelopt",
        "producer": "converted",
        "quantization": {
            "quant_algo": "W4A16_NVFP4",
            "exclude_modules": ["lm_head", "*ple*"],
        },
    }
    cfg = ModelOptConfig(q)
    assert cfg.algo == "W4A16_NVFP4"
    assert cfg.ignore("lm_head")
    assert cfg.ignore("model.layers.0.ple.ple_embedding.ngram_embedding.weight")
    assert not cfg.ignore("model.layers.0.mlp.experts")


def test_top_level_wins_over_nested():
    q = {"quant_method": "modelopt", "quant_algo": "FP8", "quantization": {"quant_algo": "W4A16_NVFP4"}}
    assert ModelOptConfig(q).algo == "FP8"


def test_unknown_algo_still_fails_closed():
    q = {"quant_method": "modelopt", "quantization": {"quant_algo": "NOPE"}}
    with pytest.raises(NotImplementedError):
        ModelOptConfig(q)
