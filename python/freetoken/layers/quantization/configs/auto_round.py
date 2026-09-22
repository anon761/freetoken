from __future__ import annotations

import re
from typing import Any, ClassVar

from ..registry import register_dialect
from ..scheme import QuantScheme, wna16_scheme
from .base import QuantConfig


@register_dialect
class AutoRoundConfig(QuantConfig):
    """Intel AutoRound W4A16 (weight-only INT4) exports.

    ``quant_method: auto-round`` with ``bits``/``group_size``/``sym`` and an
    ``extra_config`` map of regex -> ``{bits, data_type}``: every module whose
    checkpoint name matches a ``bits >= 16`` entry stays BF16, everything else under
    ``block_name_to_quantize`` is served as packed INT4 (``qweight`` / ``qzeros`` /
    fp16 ``scales``). Served through the offload cache, like the other MoE kinds.
    """

    dialect = "auto-round"

    SCHEME: ClassVar[QuantScheme] = wna16_scheme()

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.bits = int(q.get("bits") or 4)
        self.group_size = int(q.get("group_size") or 128)
        self.sym = bool(q.get("sym", True))
        self.block_prefix = str(q.get("block_name_to_quantize") or "")
        self.extra: list[tuple[re.Pattern[str], int]] = []
        for pattern, spec in (q.get("extra_config") or {}).items():
            bits = int((spec or {}).get("bits") or 16)
            self.extra.append((re.compile(pattern), bits))
        self._scheme = wna16_scheme(self.group_size)

    def _stays_bf16(self, name: str) -> bool:
        # Ordered rules: the last matching extra_config entry wins (AutoRound semantics).
        result = False
        for pattern, bits in self.extra:
            if pattern.search(name):
                result = bits >= 16
        return result

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.block_prefix and not name.startswith(self.block_prefix):
            return None
        if self._stays_bf16(name):
            return None
        return self._scheme

    def expert_kind(self) -> str | None:
        return str(self._scheme.kind)

    def has_kind(self, kind: str) -> bool:
        return kind == str(self._scheme.kind)


__all__ = ["AutoRoundConfig"]
