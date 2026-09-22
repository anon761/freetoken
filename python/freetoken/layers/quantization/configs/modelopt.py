from __future__ import annotations

from typing import Any, ClassVar

from ..names import ancestors, name_set
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme
from ..scheme import (
    fp8_block_scheme,
    fp8_tensor_scheme,
    mxfp8_scheme,
    nvfp4_scheme,
)
from .base import QuantConfig


@register_dialect
class ModelOptConfig(QuantConfig):
    """NVIDIA ModelOpt exports: one ``quant_algo`` for every Linear minus ``ignore``,
    ``MIXED_PRECISION`` with a per-module ``quantized_layers`` allow-list, or a
    ``config_groups`` map (mixed-precision exports without a top-level algo)."""

    dialect = "modelopt"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "NVFP4": nvfp4_scheme(input_scale=True),
        "NVFP4_NO_INPUT": nvfp4_scheme(input_scale=False),
        "FP8": fp8_tensor_scheme("fp32", input_scale=True),
        "FP8_PER_CHANNEL_PER_TOKEN": fp8_tensor_scheme("fp32", per_row=True),
        "FP8_PB_WO": fp8_block_scheme("fp32"),
        "MXFP8": mxfp8_scheme(),
    }

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        method = str(q.get("quant_method") or "").lower()
        return method == "modelopt" or (not method and bool(q.get("quant_algo")))

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        # Some exporters (e.g. compressed-tensors-style RVN checkpoints) nest the
        # fields one level down under "quantization": {quant_algo, exclude_modules}.
        # Top-level keys win so flat ModelOpt exports stay authoritative.
        inner = q.get("quantization") if isinstance(q.get("quantization"), dict) else {}
        self.algo = str(q.get("quant_algo") or inner.get("quant_algo") or "").upper()
        self.ignore = name_set(tuple(
            q.get("ignore") or q.get("exclude_modules") or inner.get("exclude_modules") or ()
        ))
        layers = q.get("quantized_layers") or {}
        self.quantized_layers = {k: str((v or {}).get("quant_algo") or "").upper() for k, v in layers.items()} if isinstance(layers, dict) else {}
        self.with_input_scale = bool(q.get("with_input_scale", True))
        # config_groups-based mixed precision (no top-level algo): parse the groups so
        # scheme_for_name / expert_kind can answer without a second detector.
        self.group_targets: list[tuple[tuple[str, ...], QuantScheme]] = []
        for spec in (q.get("config_groups") or {}).values():
            targets = tuple(spec.get("targets") or ())
            self.group_targets.append((targets, self._group_scheme(spec)))
        if self.algo == "MIXED_PRECISION" and not self.quantized_layers:
            raise NotImplementedError("ModelOpt MIXED_PRECISION without quantized_layers in quantization_config")
        if self.algo and not self.group_targets and self.algo != "MIXED_PRECISION":
            self._scheme_of(self.algo)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.ignore(name):
            return None
        if self.group_targets:
            for targets, scheme in self.group_targets:
                if name_set(targets)(name):
                    return scheme
            return None
        algo = self._module_algo(name)
        return None if algo is None else self._scheme_of(algo)

    def expert_kind(self) -> str | None:
        kinds = [
            scheme.kind
            for targets, scheme in self.group_targets
            if any("experts" in t for t in targets)
        ]
        if kinds:
            # a mixed export can list the MTP experts separately (e.g. mxfp8); the
            # routed experts decide, so prefer nvfp4 when it is among them.
            return str(QuantKind.NVFP4) if QuantKind.NVFP4 in kinds else str(kinds[0])
        if self.algo == "MIXED_PRECISION":
            # the per-module quantized_layers map names the experts (e.g. MiniMax-M3:
            # ``...block_sparse_moe.experts.0.w1`` -> NVFP4).
            for name, algo in self.quantized_layers.items():
                if not (name.endswith(".experts") or ".experts." in name):
                    continue
                if "fp4" in algo.lower():
                    return str(QuantKind.NVFP4)
                if "fp8" in algo.lower():
                    return str(QuantKind.FP8_TENSOR)
            return None
        if self.algo:
            return str(self._scheme_of(self.algo).kind)
        return None

    def _module_algo(self, name: str) -> str | None:
        if self.algo != "MIXED_PRECISION":
            return self.algo
        for a in ancestors(name):
            algo = self.quantized_layers.get(a)
            if algo is not None:
                return algo
        return None

    def _scheme_of(self, algo: str) -> QuantScheme:
        if algo in ("NVFP4", "W4A16_NVFP4"):
            return self.SCHEMES["NVFP4" if self.with_input_scale else "NVFP4_NO_INPUT"]
        try:
            return self.SCHEMES[algo]
        except KeyError:
            raise NotImplementedError(f"ModelOpt quant_algo {algo!r} is not supported") from None

    @classmethod
    def _group_scheme(cls, spec: dict[str, Any]) -> QuantScheme:
        weights = spec.get("weights") or {}
        bits = int(weights.get("num_bits") or 0)
        wtype = str(weights.get("type") or "float").lower()
        group = int(weights.get("group_size") or 0)
        act = spec.get("input_activations") or {}
        if bits == 4 and wtype == "float":
            return nvfp4_scheme(input_scale=bool(act))
        if bits == 8 and wtype == "float":
            return mxfp8_scheme() if group == 32 else fp8_tensor_scheme("fp32", input_scale=True)
        raise NotImplementedError(f"ModelOpt config group weights {weights} are not supported")
