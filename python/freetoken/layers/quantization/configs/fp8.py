from __future__ import annotations

from typing import Any

from ..names import is_routed_expert, name_set, substr_set
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme
from ..scheme import FP8_BLOCK, fp8_block_scheme, fp8_tensor_scheme, mxfp4_scheme
from .base import QuantConfig, cfg_get


@register_dialect
class Fp8BlockConfig(QuantConfig):
    """HF ``quant_method: fp8`` (DeepSeek-V3 style 128x128 block scales) plus the DeepSeek-V4 e8m0 / fp4-expert variant."""

    dialect = "fp8"

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        block = tuple(int(x) for x in (q.get("weight_block_size") or ()))
        # DeepSeek-V3-style exports use 128x128 blocks; DeepSeek-V4.1 ships 32x32
        # (its official quantization_config). The block geometry rides on the
        # config so readers (and Phase-2 scale-shape validation) can branch on it.
        if q.get("weight_per_tensor") or block not in ((FP8_BLOCK, FP8_BLOCK), (32, 32)):
            raise NotImplementedError(f"fp8 checkpoint with weight_block_size={block} per_tensor={q.get('weight_per_tensor')} is not supported; only 128x128 and 32x32 blocks are")
        self.block = block
        # transformers skips lm_head when the checkpoint gives no list
        not_convert = tuple(q.get("modules_to_not_convert") or ("lm_head",))
        self.not_convert = name_set(not_convert)
        self.not_convert_substr = substr_set(not_convert)
        self.convert_tables = name_set(tuple(q.get("modules_to_convert") or ()))
        self.e8m0 = str(q.get("scale_fmt") or "").lower() == "ue8m0"
        # ``expert_dtype`` rides top-level on some exports (V4-Flash) and inside
        # the quantization_config on others (V4.1) — read both.
        expert_dtype = cfg_get(hf_config, "expert_dtype")
        if expert_dtype is None:
            expert_dtype = q.get("expert_dtype")
        self.expert_fp4 = str(expert_dtype or "").lower() == "fp4"
        # Per-instance schemes at this checkpoint's block size (the class-level
        # SCHEMES table stays the 128x128 default for other consumers).
        self._schemes = {
            "BLOCK": fp8_block_scheme("float", self.block),
            "BLOCK_E8M0": fp8_block_scheme("e8m0", self.block),
            # HF ``modules_to_convert``: a table (Qwen3.8-Flash-Next PLE) stored e4m3 with one scalar scale
            "TABLE": fp8_tensor_scheme("float"),
            "EXPERT_MXFP4": mxfp4_scheme(),
        }

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.convert_tables(name):
            return self._schemes["TABLE"]
        if self.not_convert(name) or self.not_convert_substr(name):
            return None
        if self.expert_fp4 and is_routed_expert(name):
            return self._schemes["EXPERT_MXFP4"]
        return self._schemes["BLOCK_E8M0" if self.e8m0 else "BLOCK"]

    def expert_kind(self) -> str | None:
        # block-fp8 quantizes the routed experts too (V4-Flash's fp4 experts are MXFP4).
        return str(QuantKind.MXFP4) if self.expert_fp4 else str(QuantKind.FP8_BLOCK)
