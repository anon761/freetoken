"""Qwen3.8-Flash-Next vision tower (model_type qwen4_exp_vision).

A Qwen-VL-style ViT: a 3-D patch embed, a learned square position table bilinearly
resampled per image, axial 2-D RoPE, ``depth`` pre-norm blocks (LayerNorm + qkv/proj
attention + gated MLP) and a spatial-merge patch merger that projects straight to the
text hidden size (no separate multimodal embedder). Served only when vision is enabled
(``FREETOKEN_LOAD_VISION`` / ``--load-vision``); the checkpoint keys are
``model.visual.*`` and the loader renames them to ``vision_tower.*``.

The geometry helpers mirror ``transformers.vision_utils`` (interpolation taps, 2-D
position ids, packed per-image ``cu_seqlens``); attention is computed per image segment
(the fork's scheduler has no varlen vision kernel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from .config import Qwen4VisionConfig


class _LayerNorm(BaseOP):
    """LayerNorm with weight and bias (the vision blocks/merger use it, not RMSNorm)."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class _Embedding(BaseOP):
    """Plain learned embedding (the position table)."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


class _Conv3d(BaseOP):
    """3-D patch projection (stride == kernel), weight ``[out, in, T, P, P]``."""

    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, int, int]) -> None:
        self.weight = torch.empty(out_channels, in_channels, *kernel)
        self.bias = torch.empty(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight, self.bias, stride=self.weight.shape[2:])


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Axial 2-D RoPE over the whole head dim. ``q``/``k`` ``[N, heads, head_dim]``."""
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class _VisionRotary:
    """On-the-fly axial 2-D cos/sin (no learnable params)."""

    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.head_dim = vc.head_dim
        self.theta = vc.rope_theta
        self._inv_freq: torch.Tensor | None = None

    def _inv(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            spatial_dim = self.head_dim // 2
            self._inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=device) / spatial_dim)
            )
        return self._inv_freq

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """``position_ids`` is ``[N, 2]`` (h, w); returns ``[N, head_dim]`` cos/sin."""
        inv = self._inv(position_ids.device)
        freqs = position_ids[..., None].float() * inv  # [N, 2, spatial_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # recompose: concat h|w, then duplicate the (h,w) block across the full head dim
        cos = torch.cat([torch.cat([cos[:, 0], cos[:, 1]], dim=-1)] * 2, dim=-1)
        sin = torch.cat([torch.cat([sin[:, 0], sin[:, 1]], dim=-1)] * 2, dim=-1)
        return cos.to(dtype), sin.to(dtype)


class Qwen4ExpVisionMLP(BaseOP):
    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.linear_fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        act = F.gelu(self.linear_fc1.forward(hidden_state), approximate="tanh")
        return self.linear_fc2.forward(act)


class Qwen4ExpVisionPatchEmbed(BaseOP):
    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.patch_size = vc.patch_size
        self.temporal_patch_size = vc.temporal_patch_size
        self.in_channels = vc.in_channels
        self.embed_dim = vc.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = _Conv3d(self.in_channels, self.embed_dim, kernel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj.forward(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen4ExpVisionPatchMerger(BaseOP):
    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.hidden_size = vc.hidden_size * (vc.spatial_merge_size**2)
        self.norm = _LayerNorm(vc.hidden_size, eps=1e-6)
        self.linear_fc1 = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(self.hidden_size, vc.out_hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self.hidden_size)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen4ExpVisionAttention(BaseOP):
    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.dim = vc.hidden_size
        self.num_heads = vc.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = LinearReplicated(self.dim, self.dim * 3, has_bias=True)
        self.proj = LinearReplicated(self.dim, self.dim, has_bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv.forward(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = _apply_rotary(q, k, cos, sin)
        # per-image (segment) bidirectional attention — the scheduler provides no varlen kernel
        outputs = []
        for i in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            qi = q[s:e].transpose(0, 1).unsqueeze(0)
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scaling)
            outputs.append(oi.squeeze(0).transpose(0, 1))
        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj.forward(attn_output)


class Qwen4ExpVisionBlock(BaseOP):
    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.norm1 = _LayerNorm(vc.hidden_size, eps=1e-6)
        self.norm2 = _LayerNorm(vc.hidden_size, eps=1e-6)
        self.attn = Qwen4ExpVisionAttention(vc)
        self.mlp = Qwen4ExpVisionMLP(vc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.forward(
            self.norm1.forward(hidden_states), cu_seqlens, cos, sin
        )
        return hidden_states + self.mlp.forward(self.norm2.forward(hidden_states))


def _axis_taps_weights(index: torch.Tensor, size: torch.Tensor, side: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear taps into a ``side``-length table for target ``index`` on an axis of
    length ``size`` (align_corners=True, border padding) — mirrors ``vision_utils``."""
    src = index.to(torch.float32) * (side - 1) / torch.clamp(size.to(torch.float32) - 1, min=1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    weights = (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    return taps, weights


def _interp_indices_weights(grid_thw: torch.Tensor, side: int, merge: int):
    """Per-patch gather indices/weights resampling the square position table to each
    image's grid, in spatial-merge-block order (mirrors ``get_vision_interpolation_...``)."""
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(counts.sum(), device=grid_thw.device) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_weights = _axis_taps_weights(row, heights, side)
    w_taps, w_weights = _axis_taps_weights(col, widths, side)
    n = h_taps.shape[1]
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return indices, weights


def _position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """``[N, 2]`` (h, w) ids in spatial-merge-block order, repeated over frames."""
    ids = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(h, device=grid_thw.device),
            torch.arange(w, device=grid_thw.device),
            indexing="ij",
        )
        block = (h // merge, merge, w // merge, merge)
        hpos = hpos.reshape(block).transpose(1, 2).flatten()
        wpos = wpos.reshape(block).transpose(1, 2).flatten()
        ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(ids, dim=0)


def _cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Packed per-frame attention boundaries (each frame is its own segment)."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


class Qwen4ExpVisionModel(BaseOP):
    """Pixels -> text-space soft tokens: ``[num_patches, out_hidden_size]``."""

    def __init__(self, vc: "Qwen4VisionConfig") -> None:
        self.spatial_merge_size = vc.spatial_merge_size
        self.patch_embed = Qwen4ExpVisionPatchEmbed(vc)
        self.pos_embed = _Embedding(vc.num_position_embeddings, vc.hidden_size)
        self.num_grid_per_side = int(vc.num_position_embeddings**0.5)
        self.blocks = OPList([Qwen4ExpVisionBlock(vc) for _ in range(vc.depth)])
        self.merger = Qwen4ExpVisionPatchMerger(vc)
        self._rotary = _VisionRotary(vc)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.spatial_merge_size
        interp_indices, interp_weights = _interp_indices_weights(
            grid_thw, self.num_grid_per_side, merge
        )
        position_ids = _position_ids(grid_thw, merge)
        cu_seqlens = _cu_seqlens(grid_thw)

        hidden = self.patch_embed.forward(pixel_values)
        pos_embeds = (self.pos_embed.forward(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden = hidden + pos_embeds.to(hidden.dtype)
        cos, sin = self._rotary.cos_sin(position_ids, hidden.dtype)
        for block in self.blocks.op_list:
            hidden = block.forward(hidden, cu_seqlens, cos, sin)
        return self.merger.forward(hidden)


__all__ = ["Qwen4ExpVisionModel"]
