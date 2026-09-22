from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path; remains None for the (serialized) online path.
    mm_embeds: torch.Tensor | None = None
    # Online image input: raw image-processor outputs (CPU tensors) shipped from the
    # tokenizer worker. The scheduler runs the vision tower on them and fills mm_embeds.
    pixel_values: torch.Tensor | None = None  # [num_patches, C*T*P*P]
    image_grid_thw: torch.Tensor | None = None  # [num_images, 3]


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
