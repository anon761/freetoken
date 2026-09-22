from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger
from freetoken.utils.hf import optional_hf_file

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)

# Implemented DSpark speculative-verify implementations (--dspark-verify). "prefill"
# runs one prefill-phase extend; "decode" additionally aligns the attention with the
# plain-decode reduction path (window ring order + decode kernel variant).
DSPARK_VERIFY_MODES: tuple[str, ...] = ("decode", "prefill")

# --mtp modes: "auto" follows FREETOKEN_ENABLE_MTP, "on" forces the base checkpoint's
# embedded draft head, "off" forces it off, "file" loads the standalone artifact given
# via --mtp-header. MTP rounds run under overlap scheduling (the verify is the batch);
# FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 forces the drain-safe non-overlap loop.
MTP_MODES: tuple[str, ...] = ("auto", "on", "off", "file")


def mtp_artifact_dir(mtp_path: str) -> str:
    """The artifact root of a standalone MTP draft head: the path itself for a
    checkpoint directory, or its parent for a single ``.safetensors`` file (the
    family config.json lives next to the shard)."""
    if os.path.isfile(mtp_path):
        return os.path.dirname(os.path.abspath(mtp_path))
    return mtp_path


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    # Parallel expert-bank reader thread count (--expert-load-workers): the O_DIRECT
    # preadv pool behind the 'parallel' read. The packing consumer is single-threaded,
    # so past its rate extra workers only queue.
    expert_load_workers: int = 16
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # MTP-verify CPU experts (--moe-verify-cpu): keep normal decode on the GPU offload
    # path, but compute the MTP verify's routed experts on the CPU executor (RAM-resident
    # host banks, batched GEMV). The verify's ~k+1 new tokens would otherwise stream their
    # experts over PCIe (the linear-cost wall); the CPU path reads them at RAM bandwidth.
    moe_verify_cpu: bool = False
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    # --swa-eviction-interval: proactive out-of-window SWA eviction every N forwards.
    # None -> FREETOKEN_SWA_EVICTION_INTERVAL env (legacy), else 128. <= 0 disables it.
    swa_eviction_interval: int | None = None
    distributed_timeout: float = 1800.0  # TP barriers span the ~30-min serial bank build
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    # Standalone MTP draft-head checkpoint (see --mtp file). Empty = the draft head
    # comes from the base checkpoint's dense stream (env-gated, as before).
    mtp_path: str = ""
    # DSpark speculative-verify implementation (--dspark-verify). "decode" (default)
    # runs the verify through the decode-aligned attention reduction; "prefill" keeps
    # the older ascending-window extend. See scheduler/dspark.py.
    dspark_verify: str = "decode"
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    def __post_init__(self):
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        model_config = parse_config(self.hf_config)
        model_config = replace(model_config, quant=checkpoint_quant_config(self.model_path, self.hf_config, spec))
        if self.mtp_path:
            model_config = self._apply_external_mtp(model_config)
        return model_config

    def _apply_external_mtp(self, model_config: ModelConfig) -> ModelConfig:
        """--mtp file: the draft head comes from the standalone artifact.

        `mtp_path` is the artifact directory (or a single .safetensors shard whose
        family config.json sits next to it). Validates the artifact against the
        base model's geometry (a mismatched draft silently produces garbage
        tokens — fail loudly at parse time), then overrides the draft-head fields:
        mtp_num_layers from the artifact config, mtp_enabled forced on (the env
        gate only ever covered the in-checkpoint draft)."""
        from freetoken.models.qwen4_exp.weight import validate_mtp_compat

        artifact_hf = cached_load_hf_config(mtp_artifact_dir(self.mtp_path))
        spec = get_model_spec(self.hf_config.architectures[0])
        if spec.module != "freetoken.models.qwen4_exp":
            raise ValueError(
                f"--mtp file is only supported for the qwen4_exp family, not {spec.module}"
            )
        artifact_parse = _load_attr(get_model_spec(artifact_hf.architectures[0]).module, "parse_config")
        artifact_cfg = artifact_parse(artifact_hf)
        validate_mtp_compat(model_config, artifact_cfg, self.mtp_path)
        num_layers = int(getattr(artifact_cfg.qwen4_args, "mtp_num_layers", 0) or 0)
        if num_layers <= 0:
            raise ValueError(
                f"mtp artifact {self.mtp_path!r}: no mtp_num_hidden_layers in its config"
            )
        # Re-parse the family config with the draft count folded in: the draft
        # layer rides the QSA group (ids continue after the main stack) and the
        # pool depth derives from text.mtp_num_hidden_layers at PARSE time --
        # patching qwen4_args afterwards leaves the draft layer unregistered
        # (qsa_forward KeyError on the draft layer id).
        from freetoken.utils.hf import RawConfigShim

        data = self.hf_config.to_dict()
        text = data.get("text_config") or data
        text["mtp_num_hidden_layers"] = num_layers
        model_config = _load_attr(spec.module, spec.parse_config)(RawConfigShim(data))
        model_config = replace(
            model_config, quant=checkpoint_quant_config(self.model_path, self.hf_config, spec)
        )
        return replace(
            model_config,
            qwen4_args=replace(model_config.qwen4_args, mtp_num_layers=num_layers, mtp_enabled=True),
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"


def checkpoint_quant_config(model_path: str, hf_config: Any, spec: Any):
    """The checkpoint's QuantConfig under the family's naming, or None for GGUF, whose native-quant ops the shared parser does not model yet."""
    from freetoken.layers.quantization import NameMap, QuantConfig

    if spec.parse_config == "parse_gguf_config":
        return None
    # NOTE: ModelOpt exports before 0.41 keep the quantization config only in hf_quant_config.json, and the weight download fetches nothing but the safetensors shards, so this sidecar is fetched on its own.
    hf_quant_config = None
    sidecar = optional_hf_file(model_path, "hf_quant_config.json")
    if sidecar is not None:
        import json

        with open(sidecar) as f:
            hf_quant_config = json.load(f)
    return QuantConfig.from_hf(
        hf_config,
        name_map=NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping),
        unquantized=spec.unquantized_modules,
        hf_quant_config=hf_quant_config,
    )
