from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.utils import div_even, init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from freetoken.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # fa2 split-KV prefill needs ``tmp_v <= qo_heads_local * padded_batch_size *
        # cta_tile_q * head_dim * 4`` bytes of scratch, where flashinfer's scheduler
        # caps ``padded_batch_size`` at ~``2 * SM / kv_heads_local`` and
        # ``cta_tile_q`` is 128 (64 at head_dim >= 256); when the cap can't be met
        # it disables split-KV and allocates no tmp_v at all. The original flat
        # 128 MiB overflowed on head_dim=256 extend-prefills (Qwen3.5/3.6 MoE) and
        # the flat 256 MiB on MiniMax-M3's 64-head dense layers (H100, 132 SMs:
        # 64 heads x ceil(2*132/4)=66 padded rows x 128 x 128 x 4 B = 264 MiB of
        # tmp_v, over the flat buffer). Derive the bound
        # from the model's TP-LOCAL geometry + this device's SM count, with slack
        # for the tmp_s/merge siblings, floored at the old 256 MiB -- geometries
        # that never exceeded the flat buffer (e.g. GLM-4.7's 96q/8kv) stay at it.
        tp_size = get_tp_info().size
        qo_local = div_even(config.num_qo_heads, tp_size)
        kv_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        sm_count = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if self.device.type == "cuda"
            else 128
        )
        cta_tile_q = 64 if config.head_dim >= 256 else 128
        padded_batch = -(-2 * sm_count // max(1, kv_local))
        tmp_v_bound = qo_local * padded_batch * cta_tile_q * config.head_dim * 4
        workspace_bytes = max(256 * 1024 * 1024, tmp_v_bound + 32 * 1024 * 1024)
        self.float_workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        # MTP verify/draft extend graphs (graph-mode paged prefill; see _init_capture_extend).
        self._verify_graph: dict = {}
        self._draft_graph: dict = {}
        self.graph_extend_max_kv: int | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        if attn_spec is not None:
            # This backend has no window/sinks/sm_scale plumbing; dropping the spec
            # silently would attend with the wrong scale or an unbounded window.
            raise ValueError("The fi backend does not support per-call AttentionSpec.")

        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def reset_capture(self) -> None:
        # Base clears the common capture scratch; additionally drop the per-bs decode graph
        # wrappers (their indptr/indices alias freed capture scratch). Preserves the
        # long-lived workspace buffers. Lets init_capture_graph re-run after a cache rebuild.
        super().reset_capture()
        self.graph_wrappers = {}
        self._verify_graph = {}
        self._draft_graph = {}
        self.graph_extend_max_kv = None

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    # ----- MTP verify/draft extend graphs (graph-mode paged prefill) ----------------------
    #
    # The MTP verify is a prefill extend of t=k+1 tokens/request, the draft chain the MTP
    # block's per-step extend of t=1. FlashInfer's paged-prefill wrapper has a CUDA-graph
    # mode (``use_cuda_graph=True`` + caller-owned indptr/indices buffers): plan ONCE, then
    # per round overwrite only the device buffers and replay. Two constraints, both found on
    # hardware: split-KV must be DISABLED (its split schedule is baked for the plan-time
    # geometry and is wrong once the staged lengths differ), and the plan faults at very
    # large kv lengths -- so the graph covers kv <= MTP_GRAPH_MAX_KV and the caller falls
    # back to the eager extend above it (``graph_extend_max_kv``). Mirrors
    # QSASparseAttnBackend's init_capture_verify/draft + stage/scratch contract.

    def _init_capture_extend(self, n_max: int, t: int) -> dict:
        if self.capture is None:
            raise RuntimeError("init_capture_graph must run before an MTP extend capture")
        width = int(self.capture.page_table.numel() // max(1, self.max_graph_bs))
        width = max(t, min(width, int(ENV.MTP_GRAPH_MAX_KV.value)))
        self.graph_extend_max_kv = width
        return {
            "n": n_max,
            "t": t,
            "width": width,
            "qo": torch.arange(n_max + 1, dtype=torch.int32, device=self.device) * t,
            "ip": torch.zeros(n_max + 1, dtype=torch.int32, device=self.device),
            "lpl": torch.ones(n_max, dtype=torch.int32, device=self.device),
            "idx": torch.zeros(n_max * width, dtype=torch.int32, device=self.device),
            "wrappers": {},
        }

    def _extend_wrapper(self, store: dict, bs: int):
        w = store["wrappers"].get(bs)
        if w is not None:
            return w
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        t, width = store["t"], store["width"]
        w = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=store["qo"][: bs + 1],
            paged_kv_indptr_buf=store["ip"][: bs + 1],
            paged_kv_indices_buf=store["idx"],
            paged_kv_last_page_len_buf=store["lpl"][:bs],
            backend="fa2",
        )
        self.last_event.synchronize()  # plan reuses a pinned staging buffer (see decode plan)
        w.plan(
            torch.arange(bs + 1, dtype=torch.int32) * t,
            torch.arange(bs + 1, dtype=torch.int32) * width,
            torch.arange(bs * width, dtype=torch.int32),
            torch.ones(bs, dtype=torch.int32),
            self.qo_head_local,
            self.kv_head_local,
            self.config.head_dim,
            1,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=self.kvcache.dtype,
            kv_data_type=self.kvcache.dtype,
            disable_split_kv=True,
        )
        store["wrappers"][bs] = w
        return w

    def _extend_metadata(self, store: dict) -> FIMetadata:
        bs, t = store["n"], store["t"]
        cpu = {"device": "cpu", "dtype": torch.int32, "pin_memory": torch.cuda.is_available()}
        return FIMetadata(
            cu_seqlens_q_cpu=torch.empty(bs + 1, **cpu),
            cu_seqlens_k_cpu=torch.empty(bs + 1, **cpu),
            cu_seqlens_q_gpu=store["qo"][: bs + 1],
            indices=store["idx"],
            last_page_len_cpu=torch.ones(bs, **cpu),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=torch.empty(bs, **cpu),
            dtype=self.kvcache.dtype,
            wrapper=self._extend_wrapper(store, bs),
            initialized=True,
        )

    def _stage_extend_round(self, store: dict, table_idx, kvlen, pad_table_idx: int) -> None:
        """Fill the active rows (first n real, tail padded at ``pad_table_idx``/kv=t) so the
        captured kernels never touch a real request's KV. Runs OUTSIDE the graph, so plain
        device copies of indptr/indices are fine. ``kvlen`` is capped by
        ``graph_extend_max_kv`` (the caller checked)."""
        bs, t = store["n"], store["t"]
        n = len(table_idx)
        rows = [int(x) for x in table_idx] + [int(pad_table_idx)] * (bs - n)
        kv = [int(x) for x in kvlen] + [t] * (bs - n)
        ip = [0]
        for k in kv:
            ip.append(ip[-1] + k)
        store["ip"][: bs + 1].copy_(torch.tensor(ip, dtype=torch.int32, device=self.device))
        store["lpl"][:bs].fill_(1)
        pt = get_global_ctx().page_table
        off = 0
        for row, k in zip(rows, kv):
            store["idx"][off : off + k].copy_(pt[row, :k])
            off += k

    def init_capture_verify(self, n_max: int, t: int) -> None:
        self._verify_graph = self._init_capture_extend(n_max, t)

    def set_verify_n(self, n: int) -> None:
        self._verify_graph["n"] = n

    def make_verify_metadata(self, n: int | None = None, t: int | None = None) -> FIMetadata:
        return self._extend_metadata(self._verify_graph)

    def stage_verify_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self._stage_extend_round(self._verify_graph, table_idx, kvlen, pad_table_idx)

    def verify_scratch(self):
        return nullcontext()

    def init_capture_draft(self, n_max: int) -> None:
        self._draft_graph = self._init_capture_extend(n_max, 1)

    def set_draft_n(self, n: int) -> None:
        self._draft_graph["n"] = n

    def make_draft_metadata(self) -> FIMetadata:
        return self._extend_metadata(self._draft_graph)

    def stage_draft_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self._stage_extend_round(self._draft_graph, table_idx, kvlen, pad_table_idx)

    def draft_scratch(self):
        return nullcontext()

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
