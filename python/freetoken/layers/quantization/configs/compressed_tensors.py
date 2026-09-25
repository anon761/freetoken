from __future__ import annotations

from typing import Any, ClassVar

from ..names import Matcher, ct_set
from ..registry import register_dialect
from ..scheme import QuantScheme
from ..scheme import fp8_block_scheme, fp8_tensor_scheme, nvfp4_scheme
from .base import QuantConfig


@register_dialect
class CompressedTensorsConfig(QuantConfig):
    """llm-compressor exports: ordered ``config_groups`` with ``targets`` (class names, names or ``re:``) minus ``ignore``."""

    dialect = "compressed-tensors"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "NVFP4": nvfp4_scheme(input_scale=False),
        "NVFP4_LOCAL": nvfp4_scheme(input_scale=True),
        "FP8_TENSOR_STATIC": fp8_tensor_scheme("bf16", input_scale=True),
        "FP8_TENSOR_DYNAMIC": fp8_tensor_scheme("bf16"),
        "FP8_CHANNEL": fp8_tensor_scheme("bf16", per_row=True),
        "FP8_BLOCK": fp8_block_scheme("float"),
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.ignore = ct_set(tuple(q.get("ignore") or ()), class_names=False, exact=True)
        groups = q.get("config_groups") or {}
        self.groups: list[tuple[Matcher, QuantScheme | None]] = []
        self.group_targets: list[tuple[tuple[str, ...], QuantScheme | None]] = []
        for spec in groups.values():
            targets = tuple(spec.get("targets") or ())
            scheme = self._scheme_of(spec)
            self.groups.append((ct_set(targets, class_names=True), scheme))
            self.group_targets.append((targets, scheme))

    def expert_kind(self) -> str | None:
        for targets, scheme in self.group_targets:
            if scheme is not None and any("experts" in t for t in targets):
                return str(scheme.kind)
        return None

    def has_kind(self, kind: str) -> bool:
        return any(scheme is not None and str(scheme.kind) == kind for _, scheme in self.group_targets)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.ignore(name):
            return None
        for targets, scheme in self.groups:
            if targets(name):
                return scheme
        return None

    @classmethod
    def _scheme_of(cls, spec: dict[str, Any]) -> QuantScheme | None:
        # "naive-quantized" groups (llm-compressor) store fp8 weights with a bf16
        # per-channel .weight_scale; the weight reader DEQUANTIZES them to bf16 at load
        # (see models/qwen4_exp/weight.py iter_weights), so the model must build them
        # UNQUANTIZED -- mapping them to FP8_CHANNEL here would make the model expect a
        # weight_scale the loader never yields (KeyError at load_state_dict).
        if str(spec.get("format") or "").lower() == "naive-quantized":
            return None
        w = spec.get("weights") or {}
        act = spec.get("input_activations")
        bits = int(w.get("num_bits") or 0)
        wtype = str(w.get("type") or "").lower()
        strategy = str(w.get("strategy") or "").lower()
        group = int(w.get("group_size") or 0)
        if bits == 4 and wtype == "int":
            # INT4 pack-quantized (compressed-tensors): FreeToken has no resident INT4
            # quant method -- the family that ships it reads the tensors directly (e.g.
            # the standalone MTP draft head's routed experts). Report it as unquantized so
            # config parsing succeeds instead of raising.
            return None
        if bits == 4 and wtype == "float":
            if strategy == "tensor_group" and group == 16:
                return cls.SCHEMES["NVFP4_LOCAL" if act and act.get("dynamic") == "local" else "NVFP4"]
            raise NotImplementedError(f"compressed-tensors 4-bit float scheme (strategy={strategy!r}, group_size={group}) is not supported; only NVFP4 is")
        if bits == 8 and wtype == "float":
            if strategy == "tensor":
                return cls.SCHEMES["FP8_TENSOR_DYNAMIC" if act is None or act.get("dynamic") else "FP8_TENSOR_STATIC"]
            if strategy == "channel":
                return cls.SCHEMES["FP8_CHANNEL"]
            if strategy == "block":
                block = tuple(int(x) for x in (w.get("block_structure") or ()))
                if block == (128, 128):
                    return cls.SCHEMES["FP8_BLOCK"]
        raise NotImplementedError(f"compressed-tensors weight scheme {w} is not supported")
