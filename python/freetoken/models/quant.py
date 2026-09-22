"""Family-aware quantization resolution: one source of truth for a checkpoint's scheme.

The runtime (``engine.config``) and the weight readers/``parse_config`` must agree on
what each module's storage is. Both now resolve it through the same ``QuantConfig``
dialect layer, so a model's ``parse_config`` reports the ``QuantKind`` strings the
engine will actually use -- instead of re-deriving them from the raw
``quantization_config`` with a second detector.
"""

from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import NameMap, QuantConfig, QuantKind


def quant_config_for(hf_config: Any, spec, *, hf_quant_config: dict | None = None) -> QuantConfig:
    """The checkpoint's dialect config under the family's attribute -> checkpoint naming."""
    return QuantConfig.from_hf(
        hf_config,
        name_map=NameMap(
            roots=spec.checkpoint_roots,
            segments=spec.checkpoint_segments,
            packed=spec.packed_modules_mapping,
        ),
        unquantized=spec.unquantized_modules,
        hf_quant_config=hf_quant_config,
    )


def quant_config_of(hf_config: Any) -> QuantConfig | None:
    """The dialect config for a checkpoint, resolving the family spec from its architectures.
    Returns None when the config declares no architecture the registry knows."""
    from freetoken.models.register import get_model_spec

    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        return None
    return quant_config_for(hf_config, get_model_spec(architectures[0]))



def quant_kind_str(qc: QuantConfig, attr_prefix: str) -> str:
    """The storage kind of one module (e.g. ``"nvfp4"`` / ``"fp8_block"`` / ``"none"``)."""
    scheme = qc.scheme_for(attr_prefix)
    return str(scheme.kind) if scheme is not None else str(QuantKind.NONE)


def role_quants(
    qc: QuantConfig, *, expert: str, attn: str, dense: str, lm_head: str
) -> dict[str, str]:
    """The four ``ModelConfig`` quant roles, resolved from one dialect config."""
    return {
        "expert_quant": quant_kind_str(qc, expert),
        "attn_quant": quant_kind_str(qc, attn),
        "dense_quant": quant_kind_str(qc, dense),
        "lm_head_quant": quant_kind_str(qc, lm_head),
    }


__all__ = ["quant_config_for", "quant_config_of", "quant_kind_str", "role_quants"]
