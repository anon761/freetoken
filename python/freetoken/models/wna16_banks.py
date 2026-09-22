"""WNA16 (AutoRound / AutoGPTQ) expert reader: packed INT4 ``qweight`` + ``qzeros`` + fp16 group ``scales``.

AutoRound/llm-compressor W4A16 stores each routed expert as ``{gate,up,down}_proj`` with
three tensors in the AutoGPTQ layout, ``K`` = input features, ``N`` = output features,
``group_size`` = 128::

    qweight [K/8, N]  int32   # 8 int4 codes per word, code j -> k = 8*w + j
    qzeros  [K/128, N/8] int32  # 8 zero-points per word, group g -> k = 128*g ..
    scales  [K/128, N]  fp16

and ``W[n, k] = (code(k) - zero(k)) * scale(k // group_size, n)``.

This module turns the checkpoint tensors into per-expert pieces the WNA16 MoE kernel packs
into its host banks; like the NVFP4 reader it pre-slices to the rank's intermediate shard
so the offload path shards across TP ranks.
"""

from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]

# ``K`` is the sliced (packed) axis for ``down`` (its input), ``N`` for gate/up (their output).
_WNA16_KINDS = {"qweight": "", "qzeros": "_zero", "scales": "_scale"}


@dataclass(frozen=True)
class Wna16ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str
    group_size: int = 128


def _bank_layer(spec: Wna16ExpertSourceSpec, layer: int, config) -> int | None:
    from freetoken.moe.expert_pieces import num_moe_layers

    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    if bank_layer < 0 or bank_layer >= num_moe_layers(config):
        raise ValueError(f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} is out of range")
    return bank_layer


def wna16_tp_geometry(intermediate: int, tp_size: int, rank: int, group: int = 128) -> tuple[int, int]:
    """Group-aligned ``[k_lo, k_hi)`` of the intermediate axis.

    WNA16 group scales live on ``K`` and cannot split mid-group, so an even element split
    (e.g. 640/2 = 320) is illegal. Split on the group axis instead; the ranks may differ by
    one group (640 -> [0,256) + [256,640)). The gate/up output slice uses the SAME extent so
    the activation width equals the down proj's input, and the layer all_reduce stays correct.
    """
    if tp_size == 1:
        return 0, intermediate
    groups = intermediate // group
    return (rank * groups) // tp_size * group, ((rank + 1) * groups) // tp_size * group


def _tp_slice_piece(spec: Wna16ExpertSourceSpec, role: str, tensor: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One expert piece -> this rank's intermediate slice ``[lo, hi)``.

    ``gate``/``up`` slice their output axis ``N`` (last dim); ``down`` slices its packed
    input axis ``K`` (first dim, at the group/8 granularity of each tensor).
    """
    g = spec.group_size
    if role.startswith("down"):
        if role.endswith("_scale") or role.endswith("_zero"):
            return tensor[lo // g : hi // g]
        return tensor[lo // 8 : hi // 8]
    if role.endswith("_zero"):
        return tensor[..., lo // 8 : hi // 8]
    return tensor[..., lo:hi]


def iter_wna16_expert_pieces(
    model_path: str,
    config,
    spec: Wna16ExpertSourceSpec,
    *,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_page_cache: DropPageCache | None = None,
    primary: bool = True,
):
    """One piece per routed expert: ``gate``/``up``/``down`` ``qweight`` plus their
    ``_zero`` (qzeros) and ``_scale`` companions, straight from the safetensors shards."""
    from freetoken.models.loader import drop_page_cache as _drop
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    wanted: dict[str, tuple[int, int, str]] = {}
    for name in weight_map:
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown WNA16 expert projection {proj!r}")
        suffix = _WNA16_KINDS.get(match.group("kind"))
        if suffix is None:
            raise ValueError(f"{spec.desc}: unknown WNA16 expert tensor kind {match.group('kind')!r}")
        wanted[name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + suffix)
    from freetoken.moe.expert_pieces import num_moe_layers

    expected = num_moe_layers(config) * config.num_experts * 9
    if len(wanted) != expected:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {expected}")

    from freetoken.distributed import get_tp_info

    _tp = get_tp_info()
    lo, hi = wna16_tp_geometry(config.moe_intermediate_size, _tp.size, _tp.rank, spec.group_size)

    def _serial():
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for name, shard in weight_map.items():
            if name in wanted:
                by_shard[shard].append(name)
        for shard in tqdm(sorted(by_shard), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            drop(path)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name in by_shard[shard]:
                    yield name, _tp_slice_piece(spec, wanted[name][2], f.get_tensor(name), lo, hi)
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            yield name, _tp_slice_piece(spec, wanted[name][2], tensor, lo, hi)

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)


__all__ = ["Wna16ExpertSourceSpec", "iter_wna16_expert_pieces"]
