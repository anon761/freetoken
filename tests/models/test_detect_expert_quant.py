# Copyright (c) 2026 FreeToken contributors
# Tests for detect_expert_quant: modelopt mixed-precision checkpoints
# (e.g. RedHatAI/GLM-5.3-Flash-NVFP4) must resolve to nvfp4.
import pytest

from freetoken.models.config import detect_expert_quant


class _FakeHFConfig:
    """Minimal stand-in for a HF config with quantization_config."""

    def __init__(self, quant: dict):
        self.quantization_config = quant


def _glm_nvfp4_config():
    """Simulates RedHatAI/GLM-5.3-Flash-NVFP4: modelopt quant, two groups
    (nvfp4 routed experts + mxfp8 MTP experts), no top-level format."""
    return _FakeHFConfig({
        "quant_method": "modelopt",
        "config_groups": {
            "group_nvfp4_routed_experts": {
                "targets": ["model.language_model.layers.3.mlp.experts"],
                "weights": {"num_bits": 4, "type": "float", "dynamic": False},
            },
            "group_mxfp8_mtp_routed_experts": {
                "targets": ["model.language_model.layers.45.mlp.experts"],
                "weights": {"num_bits": 8, "type": "float", "dynamic": False},
            },
        },
    })


class TestDetectExpertQuant:

    def test_glm_nvfp4_mixed_precision(self):
        """GLM-5.3-Flash-NVFP4 (modelopt, no top-level format) → nvfp4."""
        assert detect_expert_quant(_glm_nvfp4_config()) == "nvfp4"

    def test_qwen_nvfp4_direct(self):
        """Direct quant_algo NVFP4 → nvfp4."""
        assert detect_expert_quant(_FakeHFConfig({"quant_algo": "NVFP4"})) == "nvfp4"

    def test_unquantized(self):
        """No quantization_config → none."""
        assert detect_expert_quant(_FakeHFConfig({})) == "none"

    def test_modelopt_without_nvfp4_experts(self):
        """modelopt with no nvfp4 expert groups → falls through to algo string."""
        cfg = _FakeHFConfig({
            "quant_method": "modelopt",
            "config_groups": {
                "group_fp8": {"targets": ["model.layers.0"], "weights": {"num_bits": 8}}
            },
        })
        result = detect_expert_quant(cfg)
        assert result != "nvfp4"


class TestDialectExpertKind:
    """detect_expert_quant delegates to the QuantConfig dialect (one source of truth)."""

    def test_compressed_tensors_nvfp4_experts(self):
        cfg = _FakeHFConfig({
            "quant_method": "compressed-tensors",
            "config_groups": {
                "g": {
                    "targets": ["re:.*mlp\\.experts\\..*proj$"],
                    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16},
                }
            },
        })
        assert detect_expert_quant(cfg) == "nvfp4"

    def test_official_block_fp8_experts(self):
        cfg = _FakeHFConfig({"quant_method": "fp8", "weight_block_size": [128, 128]})
        assert detect_expert_quant(cfg) == "fp8_block"

    def test_modelopt_mixed_precision_quantized_layers_experts(self):
        # MiniMax-M3-style: no config_groups; experts live in the per-module map.
        cfg = _FakeHFConfig({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "language_model.model.layers.3.block_sparse_moe.experts.0.w1": {"quant_algo": "NVFP4"},
                "language_model.model.layers.0.mlp.gate_proj": {"quant_algo": "MXFP8"},
            },
        })
        assert detect_expert_quant(cfg) == "nvfp4"

    def test_modelopt_config_groups_scheme_for_name(self):
        from freetoken.layers.quantization import ModelOptConfig

        cfg = _glm_nvfp4_config()
        qc = ModelOptConfig(cfg.quantization_config)
        assert qc.expert_kind() == "nvfp4"
        # the experts group resolves through name matching (ancestors of the probe)
        scheme = qc.scheme_for_name("model.language_model.layers.3.mlp.experts.0.gate_proj")
        assert scheme is not None and str(scheme.kind) == "nvfp4"
