"""DeepSeek-V4.1-Flash family (Phase 1 skeleton — the DSV4.1 plan).

Registry exports (the names the ModelSpec resolves): DeepseekV41ForCausalLM,
parse_config, iter_weights.
"""

from .args import DeepseekV41Args, load_args
from .config import parse_config
from .model import Block, DeepseekV41ForCausalLM, Transformer
from .weight import (
    dspark_expert_method,
    iter_dspark_expert_pieces,
    iter_expert_pieces,
    iter_weights,
    load_dspark_banks,
    shard_ftw_weight,
)

__all__ = [
    "Block",
    "DeepseekV41Args",
    "DeepseekV41ForCausalLM",
    "Transformer",
    "dspark_expert_method",
    "iter_dspark_expert_pieces",
    "iter_expert_pieces",
    "iter_weights",
    "load_args",
    "load_dspark_banks",
    "shard_ftw_weight",
    "parse_config",
]
