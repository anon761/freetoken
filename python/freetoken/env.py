from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # FTW dense-weight load: outstanding O_DIRECT reads per window (queue depth) and the
    # window size in MiB. The FTW dense shard is ~1000 small tensors; reading them one at
    # a time leaves the NVMe at queue depth 1 (~0.8 GiB/s), while 16 outstanding 8 MiB
    # reads saturate a 7 GB/s drive. See checkpoint/ftw.py:iter_ftw_weights.
    FTW_LOAD_WORKERS = EnvInt(16)
    FTW_LOAD_WINDOW_MB = EnvInt(2048)
    # MTP verify/draft CUDA-graph KV cap. FlashInfer's graph-mode paged prefill plans once
    # and must run with split-KV disabled (its split schedule is baked for the plan-time
    # geometry); planning at very large kv lengths faults, so the graph is used only up to
    # this many KV tokens per request and falls back to the eager extend above it.
    MTP_GRAPH_MAX_KV = EnvInt(8192)
    # Capture the GDN commit (Phase 2 per-token replay) as a CUDA graph for n=1;
    # the commit's ~2 launches/GDN-layer per accepted token otherwise dominate the round.
    MTP_COMMIT_GRAPH = EnvBool(False)
    # PLE table load: outstanding O_DIRECT chunk reads per shard. The ~48 GiB table is
    # read shard-by-shard (concentrating workers on one file beats spreading them across
    # shards on this ZFS pool). 8 -> ~2.8 GiB/s, 16 -> ~4.2 GiB/s on the NVMe.
    PLE_LOAD_WORKERS = EnvInt(16)
    # MTP speculative decoding (eager, non-overlap).
    # Draft chain length k: each round drafts k tokens and verifies them in one
    # k+1-token extend. Must satisfy k + 1 < the GDN chunk size (64) so verify
    # extends never cross a ×64 track boundary.
    MTP_DRAFT_TOKENS = EnvInt(3)
    # Sampled-request MTP (Phase 1 rejection sampling). OFF by default: it is correct but
    # currently a net slowdown until the speed phases land, so production (sampled by
    # default) is not gated on it.
    MTP_SAMPLED = EnvBool(False)
    # Chain MTP rounds back-to-back (no plain decode step in between): the next round
    # consumes the just-published bonus as its pending token. Off by default: measured on
    # Qwen3.8-Flash-Next the round's per-token cost is ~equal to a plain graph decode
    # token, so chaining is throughput-neutral there (the win needs a cheaper verify).
    MTP_CHAIN = EnvBool(False)
    # n-gram draft combiner: before the verify, override a request's MTP draft chain with
    # the continuation after the last match of its final NGRAM_SIZE tokens in its own
    # history (repetitions need no model). OFF by default.
    MTP_NGRAM = EnvBool(False)
    MTP_NGRAM_SIZE = EnvInt(3)
    # MTP-verify CPU experts: compute the verify's routed experts on the CPU executor
    # (RAM-resident host banks) while normal decode stays on the GPU offload path.
    MOE_VERIFY_CPU = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
