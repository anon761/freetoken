"""Block-fp8 routed-expert source reader (shared, family-independent).

Qwen3.5/Qwen3.8 store the routed experts as per-expert 128x128 block-fp8: fp8-e4m3
``<proj>_proj.weight`` plus a bf16 ``<proj>_proj.weight_scale_inv`` companion under
``<anchor>.layers.N.mlp.experts.E.*``. This module turns that layout into the canonical
``{gate, up, down}(_scale)`` pieces the MoE methods pack -- the counterpart of
``nvfp4_banks`` -- so no family has to own the reader (or import a sibling family's).

TP: every piece is pre-sliced to the rank's intermediate shard (gate/up output rows,
down input columns, block scales on the 128-grid), matching ``MoEConfig.local_intermediate``
and the kernels' local banks.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even, download_hf_weight
from tqdm import tqdm

DEFAULT_ANCHOR = "model.language_model"
BLOCK = 128


def _expert_re(anchor: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(anchor)}\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
    )


class _ShardReader:
    """Opens safetensors shards on demand and serves tensors by name on ``device``."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._map = weight_map
        self._device = device
        self._handles: dict = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()


def moe_dims(model_config) -> tuple[int, int, int, int, int]:
    """``(num_moe_layers, num_experts, hidden, intermediate, dense_prefix)``."""
    layers = model_config.num_moe_layers
    return (
        layers, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - layers,
    )


def _expert_reader(model_path: str, device: torch.device) -> _ShardReader:
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    return _ShardReader(folder, weight_map, device)


def _tp_geometry(config):
    """``(lo, hi)`` of this rank's intermediate shard, block-aligned."""
    tp = get_tp_info()
    i = config.moe_intermediate_size
    if tp.size == 1:
        return 0, i
    i_loc = div_even(i, tp.size)
    assert i_loc % BLOCK == 0, (
        f"TP fp8-block expert slice {i_loc} must stay {BLOCK}-aligned "
        f"(moe_intermediate_size {i}, tp {tp.size})"
    )
    return tp.rank * i_loc, (tp.rank + 1) * i_loc


def _tp_slice_piece(role: str, tensor: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One expert piece -> this rank's shard of the intermediate axis.

    gate/up slice their output rows, down its input columns; block scales follow the
    128-grid on the same axis. At TP=1 ``(lo, hi) == (0, I)`` and the slice is the full
    tensor (the factor maps it back onto the scale grid)."""
    if tensor.ndim <= 1:
        return tensor
    factor = BLOCK if role.endswith("_scale") else 1
    if role.startswith("down"):
        return tensor[..., lo // factor : hi // factor].contiguous()
    return tensor[lo // factor : hi // factor].contiguous()


def _role_of(raw_name: str) -> str:
    proj = "gate" if ".gate_proj." in raw_name else "up" if ".up_proj." in raw_name else "down"
    return proj + ("_scale" if raw_name.endswith(".weight_scale_inv") else "")


def iter_fp8_block_expert_pieces(
    model_path: str,
    config,
    *,
    anchor: str = DEFAULT_ANCHOR,
    parallel: bool | None = False,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[int, int, int, dict[str, torch.Tensor]]]:
    """One piece per routed expert: ``{gate, up, down}`` fp8 codes + ``_scale`` companions,
    each pre-sliced to this rank's intermediate shard."""
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    pattern = _expert_re(anchor)
    layers, experts, _hidden, _intermediate, dense = moe_dims(config)
    suffix = {"weight": "", "weight_scale_inv": "_scale"}
    lo, hi = _tp_geometry(config)

    def locate(raw_name: str):
        match = pattern.match(raw_name)
        if match is None:
            return None
        layer = int(match["layer"]) - dense
        if not 0 <= layer < layers:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return layer, int(match["expert"]), match["proj"] + suffix[match["kind"]]

    def _sliced(tensors):
        for name, tensor in tensors:
            yield name, _tp_slice_piece(_role_of(name), tensor, lo, hi)

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda name: pattern.match(name) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(_sliced(tensors), locate, tensors_per_expert=6)

    def _serial():
        reader = _expert_reader(model_path, torch.device("cpu"))
        try:
            for layer in tqdm(
                range(layers), desc="Loading fp8 experts (serial)", disable=not get_tp_info().is_primary()
            ):
                for expert in range(experts):
                    base = f"{anchor}.layers.{dense + layer}.mlp.experts.{expert}"
                    for proj in ("gate", "up", "down"):
                        for kind, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind}"
                            yield name, _tp_slice_piece(proj + suf, reader.get(name), lo, hi)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["DEFAULT_ANCHOR", "iter_fp8_block_expert_pieces", "moe_dims"]
