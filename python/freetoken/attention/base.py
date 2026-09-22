from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from freetoken.core import Batch


class AttnType(str, Enum):
    """Attention-type taxonomy, one value per KV-pool family. A model's attention
    groups map onto these (KVCacheGroupSpec.attn_type); each backend declares the
    set it can serve (BackendInfo.supported_types)."""

    FULL = "full"  # uniform causal MHA/GQA -> MHAKVCache
    SWA = "swa"  # sliding-window hybrid (window/sinks) -> HybridSWAKVCache
    MLA = "mla"  # plain latent-KV MLA -> MLAKVCache
    DSA = "dsa"  # latent-KV MLA + DSA sparse indexer -> DSAKVCache
    DSV4 = "dsv4"  # DSV4 window+compressed sparse -> DSV4PagedKVCache
    LINEAR = "linear"  # GDN/mamba state layers -> LinearStatePool
    # GQA block-sparse (MiniMax-M3): paged GQA K/V + a per-sparse-layer index-key
    # slab; the indexer picks top-k 128-token blocks per query -> BSAKVCache
    BSA = "bsa"
    # QSA compressed-block sparse (Qwen3.8-Flash-Next): paged GQA K/V + a compressed
    # index-key slab (one row per index_ratio tokens, row = slot // index_ratio) +
    # a per-request pending ring for unclosed groups -> QSAKVCache
    QSA = "qsa"

    @property
    def backend_driven(self) -> bool:
        # LINEAR layers reach their kernels directly (fla ops + batch.fla_metadata),
        # not through an attention backend, so they never constrain backend choice.
        return self is not AttnType.LINEAR


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: torch.Tensor | None = None


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...

    def reset_capture(self) -> None:
        """Drop CUDA-graph capture scratch so ``init_capture_graph`` can re-run after a
        runtime cache rebuild. The default clears the common capture state (guarded by
        ``hasattr`` so backends that hold only a subset, e.g. dsv4, are safe). Backends
        with extra per-bs graph state (FlashInfer ``graph_wrappers``) override this."""
        if hasattr(self, "capture"):
            self.capture = None
        if hasattr(self, "capture_bs"):
            self.capture_bs = []
        if hasattr(self, "max_graph_bs"):
            self.max_graph_bs = 0


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)

    def reset_capture(self) -> None:
        # Only the decode backend is ever captured (see init_capture_graph above).
        self.decode_backend.reset_capture()

    # ----- MTP verify capture (prefill backend) ---------------------------------------------
    def init_capture_verify(self, n_max: int, t: int) -> None:
        self.prefill_backend.init_capture_verify(n_max, t)

    def set_verify_n(self, n: int) -> None:
        self.prefill_backend.set_verify_n(n)

    def make_verify_metadata(self, n: int | None = None, t: int | None = None):
        return self.prefill_backend.make_verify_metadata(n, t)

    def stage_verify_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self.prefill_backend.stage_verify_round(table_idx, kvlen, pad_table_idx)

    def verify_scratch(self):
        return self.prefill_backend.verify_scratch()

    # ----- MTP draft-chain capture (prefill backend, t=1 per step) --------------------------
    def init_capture_draft(self, n_max: int) -> None:
        self.prefill_backend.init_capture_draft(n_max)

    def set_draft_n(self, n: int) -> None:
        self.prefill_backend.set_draft_n(n)

    def make_draft_metadata(self):
        return self.prefill_backend.make_draft_metadata()

    def stage_draft_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self.prefill_backend.stage_draft_round(table_idx, kvlen, pad_table_idx)

    def draft_scratch(self):
        return self.prefill_backend.draft_scratch()
