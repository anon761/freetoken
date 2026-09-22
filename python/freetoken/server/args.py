from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from freetoken.distributed import DistributedInfo
from freetoken.engine import DSPARK_VERIFY_MODES, MTP_MODES
from freetoken.scheduler import SchedulerConfig
from freetoken.utils import init_logger

logger = init_logger(__name__)


class _DeprecatedAlias(argparse.Action):
    """An old flag: warns at parse time, converts the value if asked, stores it."""

    def __init__(self, *args, new_flag: str, convert=None, **kwargs):
        self.new_flag, self.convert = new_flag, convert
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning("%s is deprecated; use %s", option_string, self.new_flag)
        setattr(namespace, self.dest, self.convert(values) if self.convert else values)


def _nvfp4_entry(value: str) -> str:
    """The --quant-backend entry an old --nvfp4-backend value stands for; auto stands for none."""
    if value == "auto":
        return ""
    return "moe.nvfp4=" + {"flashinfer": "b12x"}.get(value, value)


# Runtime knobs whose consumers read os.environ (some at import time) become flags
# here and are folded into the env before the TP workers spawn. A flag wins over the
# env; an omitted flag leaves the env untouched. ``store_true`` debug/toggle flags
# -> ENABLE, disable-only flags -> DISABLE, BooleanOptionalAction -> BOOL, valued -> VALUE.
_ENV_FLAG_ENABLE = {
    "skip_bank_pin": "FREETOKEN_SKIP_BANK_PIN",
    "dspark_debug": "FREETOKEN_DSPARK_DEBUG",
    "dspark_diff": "FREETOKEN_DSPARK_DIFF",
    "dspark_timing": "FREETOKEN_DSPARK_TIMING",
    "dspark_force_a0": "FREETOKEN_DSPARK_FORCE_A0",
    "mtp_verify_graph": "FREETOKEN_MTP_VERIFY_GRAPH",
    "mtp_draft_graph": "FREETOKEN_MTP_DRAFT_GRAPH",
    "mtp_commit_graph": "FREETOKEN_MTP_COMMIT_GRAPH",
    "mtp_sampled": "FREETOKEN_MTP_SAMPLED",
    "mtp_chain": "FREETOKEN_MTP_CHAIN",
    "mtp_ngram": "FREETOKEN_MTP_NGRAM",
}
_ENV_FLAG_DISABLE = {
    "no_hybrid_overlap": "FREETOKEN_HYBRID_OVERLAP",
    "no_fused_copy": "FREETOKEN_FUSED_COPY",
    "no_cpu_moe_flag_sync": "FREETOKEN_CPU_MOE_FLAG_SYNC",
}
_ENV_FLAG_BOOL = {
    "load_vision": "FREETOKEN_LOAD_VISION",
    "ple_io_uring": "FREETOKEN_PLE_IO_URING",
    "tp_reduce_fp8": "FREETOKEN_TP_REDUCE_FP8",
    "forward_unknown_tools": "FREETOKEN_FORWARD_UNKNOWN_TOOLS",
    "bank_cuda_alloc": "FREETOKEN_BANK_CUDA_ALLOC",
    "m3_sparse": "FREETOKEN_M3_SPARSE",
    "glm_dsa": "FREETOKEN_GLM_DSA",
    "glm5_dsa": "FREETOKEN_GLM5_DSA",
    "moe_verify_cpu": "FREETOKEN_MOE_VERIFY_CPU",
}
_ENV_FLAG_VALUE = {
    "mtp_draft_tokens": "FREETOKEN_MTP_DRAFT_TOKENS",
    "mtp_ngram_size": "FREETOKEN_MTP_NGRAM_SIZE",
    "dspark_k": "FREETOKEN_DSPARK_K",
    "mamba_ssm_dtype": "FREETOKEN_MAMBA_SSM_DTYPE",
    "pin_budget_gb": "FREETOKEN_PIN_BUDGET_GB",
    "pynccl_max_buffer_size": "FREETOKEN_PYNCCL_MAX_BUFFER_SIZE",
    "hybrid_fetch_policy": "FREETOKEN_HYBRID_FETCH",
    "tp_reduce_fp8_min_bytes": "FREETOKEN_TP_REDUCE_FP8_MIN_BYTES",
    "ple_sync": "FREETOKEN_PLE_SYNC",
    "qsa_torch_topk": "FREETOKEN_QSA_TORCH_TOPK",
    "cpu_moe_isa": "FREETOKEN_CPU_MOE_ISA",
    "api_log_dir": "FREETOKEN_API_LOG_DIR",
    "m3_inner_backend": "FREETOKEN_M3_INNER_BACKEND",
    "m3_max_layers": "FREETOKEN_M3_MAX_LAYERS",
    "glm_dsa_max_layers": "FREETOKEN_GLM_DSA_MAX_LAYERS",
    "glm5_max_layers": "FREETOKEN_GLM5_MAX_LAYERS",
    "engram_backend": "FREETOKEN_ENGRAM_BACKEND",
}


def _apply_env_overrides(kwargs: dict) -> None:
    """Fold the env-backed flag values into os.environ (before the workers spawn) and
    drop them from kwargs so they never reach the frozen ServerArgs."""
    for dest, env in _ENV_FLAG_ENABLE.items():
        if kwargs.pop(dest, None):
            os.environ[env] = "1"
    for dest, env in _ENV_FLAG_DISABLE.items():
        if kwargs.pop(dest, None):
            os.environ[env] = "0"
    for dest, env in _ENV_FLAG_BOOL.items():
        value = kwargs.pop(dest, None)
        if value is not None:
            os.environ[env] = "1" if value else "0"
    for dest, env in _ENV_FLAG_VALUE.items():
        value = kwargs.pop(dest, None)
        if value is not None:
            os.environ[env] = str(value)


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # The terminal shell is attached to this server (ft shell --model / ft serve --shell-mode).
    # The workers read it to leave the shell's foreground process group, so the ^C that cancels
    # a turn cannot also kill the engine — see server/launch.py:_detach_process_group.
    shell_mode: bool = False
    served_model_name: str | None = None
    # --mtp: draft-head selection. "auto" follows FREETOKEN_ENABLE_MTP (what the
    # deployment wrapper used to export); "on"/"off" force the base checkpoint's
    # embedded draft head on/off; "file" loads the standalone artifact in
    # mtp_header (mapped onto EngineConfig.mtp_path).
    mtp: str = "auto"
    # --mtp-header: standalone MTP draft artifact path (only meaningful with
    # --mtp file). Directory with config.json + *.safetensors, or a single shard.
    mtp_header: str = ""
    tool_call_parser: str = "llama3"
    # Reasoning parser that splits <think> reasoning from content for OpenAI
    # responses. None disables it (default for models without a reasoning protocol).
    reasoning_parser: str | None = None
    # "model": fill unspecified request sampling params from generation_config.json
    # (temperature/top_k/top_p), like sglang. "none": use framework defaults only.
    sampling_defaults: str = "model"
    # Default max output (decode) tokens for a request that omits one. None falls back to the
    # adapter's built-in default (32k).
    max_output_tokens: int | None = None
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens). Mirrors sglang's --enable-cache-report.
    enable_cache_report: bool = False
    # Comma-separated CORS allow-list for browser/webview clients (e.g. the desktop
    # app). Empty string disables CORS headers entirely; "*" allows any origin.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"
    # --gpu entries in TP-rank order, empty = not given
    gpu: tuple[str, ...] = ()
    # full UUIDs resolved from --gpu, entry i = TP rank i; None = NVML unavailable, each worker then resolves its raw entry against CUDA's own enumeration
    gpu_assigned: "tuple[str, ...] | None" = None
    # --gpu-memory-ratio overrides as parsed (gpu spec -> VRAM fraction), keyed as in --gpu
    gpu_memory_ratio: tuple[tuple[str, float], ...] = ()
    # one memory_ratio per TP rank, resolved from gpu_memory_ratio in launch_server; None = no per-GPU overrides (all ranks use memory_ratio)
    gpu_memory_ratios: "tuple[float, ...] | None" = None
    # Append the per-GPU telemetry segment to the periodic status lines (--no-gpu-stats disables)
    gpu_stats: bool = True

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/freetoken_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/freetoken_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(
    args: List[str],
    run_shell: bool = False,
    prog: str | None = None,
) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from freetoken.attention import validate_attn_backend
    from freetoken.kvcache import SUPPORTED_CACHE_MANAGER
    from freetoken.moe import MOE_STRATEGIES

    def _parse_quant_backend(value: str) -> str:
        from freetoken.layers.quantization import QuantBackend

        try:
            QuantBackend.parse(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        return value

    def _parse_moe_cache_rate(value: str) -> float:
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
        if not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1]")
        return rate

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if n < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return n

    def _lazy_gpu_arg(value: str) -> tuple[str, ...]:
        from freetoken.gpu_select import gpu_arg

        return gpu_arg(value)

    def _lazy_gpu_memory_ratio_arg(value: str) -> tuple[tuple[str, float], ...]:
        from freetoken.gpu_select import gpu_memory_ratio_arg

        return gpu_memory_ratio_arg(value)

    def _lazy_memory_ratio_arg(value: str):
        from freetoken.gpu_select import memory_ratio_arg

        return memory_ratio_arg(value)

    def _infer_tool_call_parser(model_path: str) -> str:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        # M3 first: its marker also contains the bare "minimax" substring, but the
        # namespaced tool grammar is a different protocol from M2's.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3_coder"
        if (
            "qwen3_5" in marker
            or "qwen3.5" in marker
            or ("qwen3" in marker and "coder" in marker)
        ):
            return "qwen3_coder"
        if "qwen" in marker:
            return "qwen25"
        if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
            return "deepseekv32"
        if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
            return "deepseekv32"
        if "glm" in marker:
            return "glm47"
        if "mistral" in marker:
            return "mistral"
        return "llama3"

    def _infer_reasoning_parser(model_path: str) -> str | None:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        if "deepseek" in marker and any(
            tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")
        ):
            return "deepseekv32"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3"
        if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
            return "qwen3"
        if "glm" in marker:
            return "glm"
        # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
        # not M2's always-on implicit <think>.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        return None

    parser = argparse.ArgumentParser(prog=prog, description="FreeToken Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--gpu",
        type=_lazy_gpu_arg,
        default=ServerArgs.gpu,
        help=(
            "GPU(s) to run on, comma-separated; entry i is TP rank i. Each entry is a GPU "
            "UUID (GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index"
        ),
    )

    parser.add_argument(
        "--gpu-memory-ratio",
        type=_lazy_gpu_memory_ratio_arg,
        default=ServerArgs.gpu_memory_ratio,
        metavar="GPU,RATIO;...",
        help=(
            "Per-GPU VRAM fraction overrides, e.g. '0,0.8;1,0.9'. Each entry names a GPU "
            "as in --gpu (nvidia-smi index or UUID) and a fraction in [0, 1]; a rank "
            "without an entry keeps --memory-ratio. Lets an asymmetric TP layout (a busier "
            "or larger card) use more or less of its VRAM."
        ),
    )

    parser.add_argument(
        "--gpu-stats",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.gpu_stats,
        help=(
            "Append a per-GPU telemetry segment (TP count, VRAM, utilization, temperature, "
            "power) to the periodic prefill/decode status lines. Best-effort: omitted when "
            "NVML and nvidia-smi are both unavailable."
        ),
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=ServerArgs.max_output_tokens,
        help="Default max output tokens for requests that omit one (default 32k).",
    )

    parser.add_argument(
        "--memory-ratio",
        type=_lazy_memory_ratio_arg,
        default=ServerArgs.memory_ratio,
        help=(
            "Fraction of total GPU free memory the engine may use for weights + MoE "
            "cache + KV cache combined; the remainder is reserved runtime headroom. "
            "Either a single float (applies to every TP rank) or a per-GPU list, e.g. "
            "'0:0.8,1:0.9' (shell-safe) or '0,0.8;1,0.9' (quote it: ';' is a shell "
            "operator). A list is equivalent to --gpu-memory-ratio."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )
    parser.add_argument(
        "--swa-full-tokens-ratio",
        type=float,
        default=ServerArgs.swa_full_tokens_ratio,
        help="Fraction of the full KV-token budget kept as FULL-history sliding-window KV "
        "(every layer); the rest is served by the compressed/sparse tiers. Lowering it "
        "shrinks the KV pool and frees VRAM for the MoE cache and prefill activations.",
    )

    parser.add_argument(
        "--swa-eviction-interval",
        type=int,
        default=ServerArgs.swa_eviction_interval,
        help="Proactively free each decoding request's out-of-window SWA slots every N decode "
        "forwards (default: FREETOKEN_SWA_EVICTION_INTERVAL, else 128). 0 disables the "
        "out-of-window eviction entirely (legacy FREETOKEN_SWA_NO_EVICT).",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=_positive_int,
        default=ServerArgs.decode_log_interval,
        help="Print one decode scheduler status line every N decode forwards.",
    )

    kv_capacity_group = parser.add_mutually_exclusive_group()
    kv_capacity_group.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    kv_capacity_group.add_argument(
        "--num-tokens",
        dest="num_token_override",
        type=int,
        default=ServerArgs.num_token_override,
        help=(
            "Total KV-cache capacity in tokens; must be a multiple of the resolved page "
            "size (DSV4: 128 window page, TRTLLM backend: 64). Mutually exclusive with "
            "--num-pages."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache strategy (naive | radix). For hybrid GDN models 'radix' is materialized "
        "as a GDN-aware radix (cross-request GDN-state prefix reuse); pass 'naive' to opt out.",
    )

    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=ServerArgs.enable_cache_report,
        help=(
            "Return the number of prefix-cached prompt tokens in each response's usage block "
            "(OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic "
            "usage.cache_read_input_tokens, Responses usage.input_tokens_details.cached_tokens). "
            "On /v1/messages this also makes input_tokens EXCLUDE the cached prefix, matching "
            "Anthropic billing semantics."
        ),
    )

    parser.add_argument(
        "--sampling-defaults",
        type=str,
        default=ServerArgs.sampling_defaults,
        choices=["model", "none"],
        help=(
            "Source for unspecified request sampling params. 'model' fills "
            "temperature/top_k/top_p from the checkpoint's generation_config.json "
            "(recommended for reasoning models to avoid greedy repetition loops); "
            "'none' uses framework defaults only."
        ),
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Model id returned by /v1/models. Defaults to the basename of --model.",
    )

    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "llama3",
            "qwen",
            "qwen25",
            "qwen3_coder",
            "mistral",
            "deepseekv32",
            "gemma4",
            "glm47",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gpt_oss",
            "gpt-oss",
        ],
        help="Tool-call parser format for OpenAI-compatible tool responses.",
    )

    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="auto",
        choices=[
            "auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm",
            "minimax", "minimax_m3", "muse_glimmer", "gemma4",
        ],
        help=(
            "Reasoning parser that splits chain-of-thought into reasoning_content "
            "for OpenAI responses. 'auto' selects per model family (gpt-oss Harmony, "
            "<think> for qwen3/glm/minimax, <mm:think> for minimax-m3, ATEM to=self "
            "channels for muse-glimmer, gemma thought, dsv4); 'off' disables it."
        ),
    )

    parser.add_argument(
        "--moe-strategy",
        default=ServerArgs.moe_strategy,
        choices=["auto", *MOE_STRATEGIES],
        help=(
            "How the routed experts are served. 'auto' resolves a MoE model to the offload family "
            "(offload, or hybrid when a `ft bench bw` profile recommends it); resident "
            "'fused' experts must be requested explicitly."
        ),
    )

    parser.add_argument(
        "--mtp",
        default=ServerArgs.mtp,
        choices=list(MTP_MODES),
        help=(
            "MTP/DSpark draft head. 'auto' (default) follows FREETOKEN_ENABLE_MTP; "
            "'on' builds/serves the draft head embedded in the base checkpoint; "
            "'off' forces it off even when the env gate is set; 'file' loads a "
            "standalone artifact from --mtp-header. MTP rounds overlap the scheduler "
            "(the verify is the batch); set FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 to "
            "force the drain-safe non-overlap loop."
        ),
    )

    parser.add_argument(
        "--mtp-header",
        default=ServerArgs.mtp_header,
        metavar="PATH",
        help=(
            "Standalone MTP draft-head artifact: a checkpoint directory "
            "(config.json + *.safetensors) or a single .safetensors file whose "
            "config.json sits next to it. Used with --mtp file."
        ),
    )

    parser.add_argument(
        "--mtp-path",
        dest="mtp_header",
        action=_DeprecatedAlias,
        new_flag="--mtp file --mtp-header",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="[Deprecated] Use --mtp file --mtp-header PATH.",
    )

    parser.add_argument(
        "--dspark-verify",
        default=ServerArgs.dspark_verify,
        choices=list(DSPARK_VERIFY_MODES),
        help=(
            "How a DeepSeek-V4.1 DSpark speculative verify runs. 'decode' (default) aligns the "
            "attention reduction with the plain-decode path (window ring order + decode kernel "
            "variant); 'prefill' keeps the older ascending-window extend. Acceptance is greedy "
            "matching or vLLM-style rejection sampling for sampling requests. Inert unless the "
            "served model ships the DSpark draft head (FREETOKEN_ENABLE_MTP)."
        ),
    )

    # --- env-backed runtime knobs, exposed as flags (see _ENV_FLAG_* above) ---
    parser.add_argument("--mtp-draft-tokens", type=int, default=None, dest="mtp_draft_tokens",
                        help="MTP draft chain length k (default 3); k+1 must stay inside one GDN chunk.")
    parser.add_argument("--mtp-verify-graph", action="store_true", default=None,
                        dest="mtp_verify_graph",
                        help="Replay the MTP verify extend from a captured CUDA graph (one per batch size).")
    parser.add_argument("--mtp-draft-graph", action="store_true", default=None,
                        dest="mtp_draft_graph",
                        help="Replay the MTP draft chain from a captured CUDA graph (one per batch size).")
    parser.add_argument("--mtp-commit-graph", action="store_true", default=None,
                        dest="mtp_commit_graph",
                        help="Replay the GDN commit from a captured CUDA graph (single-request; needs --mtp-verify-graph).")
    parser.add_argument("--mtp-sampled", action="store_true", default=None, dest="mtp_sampled",
                        help="Enable MTP for sampled requests via rejection sampling (default off).")
    parser.add_argument("--mtp-chain", action="store_true", default=None, dest="mtp_chain",
                        help="Chain MTP rounds back-to-back instead of alternating a plain decode step.")
    parser.add_argument("--mtp-ngram", action="store_true", default=None, dest="mtp_ngram",
                        help="Override the MTP draft chain with a self-history n-gram match when it repeats.")
    parser.add_argument("--mtp-ngram-size", type=int, default=None, dest="mtp_ngram_size",
                        help="n-gram history length for --mtp-ngram (default 3).")
    parser.add_argument("--dspark-k", type=int, default=None, dest="dspark_k",
                        help="DSpark draft chain length k (default: the draft block size).")
    parser.add_argument("--mamba-ssm-dtype", choices=["float32", "bfloat16", "float16"], default=None,
                        dest="mamba_ssm_dtype", help="GDN/SSM recurrent state dtype (default float32).")
    parser.add_argument("--pin-budget-gb", type=float, default=None, dest="pin_budget_gb",
                        help="Host pinned-memory budget in GiB (default: WSL auto budget, else none).")
    parser.add_argument("--load-vision", action=argparse.BooleanOptionalAction, default=None,
                        dest="load_vision", help="Build/load the model's vision tower (default off).")
    parser.add_argument("--pynccl-max-buffer-size", default=None, dest="pynccl_max_buffer_size",
                        help="pynccl max buffer size, e.g. 1G (default 1 GiB).")
    parser.add_argument("--hybrid-fetch-policy", choices=["recency", "lowest_id"], default=None,
                        dest="hybrid_fetch_policy", help="Hybrid MoE expert fetch order (default recency).")
    parser.add_argument("--tp-reduce-fp8", action=argparse.BooleanOptionalAction, default=None,
                        dest="tp_reduce_fp8", help="Use fp8 for the TP all-reduce (default off).")
    parser.add_argument("--tp-reduce-fp8-min-bytes", type=int, default=None, dest="tp_reduce_fp8_min_bytes",
                        help="Minimum tensor bytes for the fp8 TP reduce (default engine value).")
    parser.add_argument("--ple-sync", choices=["auto", "wait", "gate"], default=None, dest="ple_sync",
                        help="PLE disk-IO sync mode (default auto).")
    parser.add_argument("--ple-io-uring", action=argparse.BooleanOptionalAction, default=None,
                        dest="ple_io_uring", help="Use io_uring for the PLE disk reads.")
    parser.add_argument("--qsa-torch-topk", type=int, default=None, dest="qsa_torch_topk",
                        help="Use a torch top-k in the QSA sparse attention (threshold).")
    parser.add_argument("--forward-unknown-tools", action=argparse.BooleanOptionalAction, default=None,
                        dest="forward_unknown_tools",
                        help="Forward unknown tool calls instead of dropping them (default on).")
    parser.add_argument("--no-hybrid-overlap", action="store_true", default=None, dest="no_hybrid_overlap",
                        help="Disable the hybrid MoE PCIe/CPU overlap (serial path).")
    parser.add_argument("--no-fused-copy", action="store_true", default=None, dest="no_fused_copy",
                        help="Disable the legacy-fused MoE copy path.")
    parser.add_argument("--no-cpu-moe-flag-sync", action="store_true", default=None, dest="no_cpu_moe_flag_sync",
                        help="Disable the CPU-MoE flag sync.")
    parser.add_argument("--bank-cuda-alloc", action=argparse.BooleanOptionalAction, default=None,
                        dest="bank_cuda_alloc", help="Force CUDA allocation for the host expert banks.")
    parser.add_argument("--skip-bank-pin", action="store_true", default=None, dest="skip_bank_pin",
                        help="Skip host-bank pinning (CPU-only tooling; never when serving).")
    parser.add_argument("--cpu-moe-isa", default=None, dest="cpu_moe_isa",
                        help="Override the CPU-MoE ISA (default auto).")
    parser.add_argument("--dspark-debug", action="store_true", default=None, dest="dspark_debug",
                        help="Verbose DSpark draft diagnostics.")
    parser.add_argument("--dspark-diff", action="store_true", default=None, dest="dspark_diff",
                        help="Run the DSpark decode-vs-verify differential harness.")
    parser.add_argument("--dspark-timing", action="store_true", default=None, dest="dspark_timing",
                        help="Log the per-round DSpark timing breakdown.")
    parser.add_argument("--dspark-force-a0", action="store_true", default=None, dest="dspark_force_a0",
                        help="Never accept drafts (plain decode + target bonus only).")
    parser.add_argument("--api-log-dir", default=None, dest="api_log_dir",
                        help="Directory for the JSONL API request log (off by default).")
    parser.add_argument("--m3-sparse", action=argparse.BooleanOptionalAction, default=None,
                        dest="m3_sparse", help="MiniMax-M3 sparse attention (default on).")
    parser.add_argument("--m3-inner-backend", default=None, dest="m3_inner_backend",
                        help="MiniMax-M3 inner attention backend override.")
    parser.add_argument("--m3-max-layers", type=int, default=None, dest="m3_max_layers",
                        help="Cap the MiniMax-M3 layer count (smoke tests).")
    parser.add_argument("--glm-dsa", action=argparse.BooleanOptionalAction, default=None,
                        dest="glm_dsa", help="GLM-MoE-DSA sparse attention (default on).")
    parser.add_argument("--glm-dsa-max-layers", type=int, default=None, dest="glm_dsa_max_layers",
                        help="Cap the GLM-MoE-DSA layer count.")
    parser.add_argument("--glm5-dsa", action=argparse.BooleanOptionalAction, default=None,
                        dest="glm5_dsa", help="GLM5-Next DSA (default on).")
    parser.add_argument("--glm5-max-layers", type=int, default=None, dest="glm5_max_layers",
                        help="Cap the GLM5-Next layer count.")
    parser.add_argument("--engram-backend", choices=["disk", "ram"], default=None, dest="engram_backend",
                        help="DeepSeek-V4.1 Engram table access: 'disk' (buffered O_DIRECT pread) or "
                        "'ram' (mmap + one bulk read: the table is pulled into the host page cache "
                        "at load, large!).")

    parser.add_argument(
        "--moe-backend",
        dest="moe_strategy",
        action=_DeprecatedAlias,
        new_flag="--moe-strategy",
        default=argparse.SUPPRESS,
        choices=["auto", *MOE_STRATEGIES],
        help="[Deprecated] Use --moe-strategy.",
    )

    parser.add_argument(
        "--quant-backend",
        default=None,
        type=_parse_quant_backend,
        help=(
            "Kernel per quantized layer type: comma-separated layer[.kind]=name entries, e.g. "
            "'linear=marlin,moe=b12x' or 'moe.nvfp4=triton'. A layer-level entry applies to every "
            "kind whose kernel table lists the name; unlisted tables stay automatic."
        ),
    )

    parser.add_argument(
        "--ple-backend",
        default=ServerArgs.ple_backend,
        choices=["pinned", "disk"],
        help=(
            "Where a PLE n-gram table lives. 'disk' (default) reads rows straight from the "
            "checkpoint files; 'pinned' preloads the whole table into page-locked host RAM."
        ),
    )

    parser.add_argument(
        "--nvfp4-backend",
        action=_DeprecatedAlias,
        new_flag="--quant-backend moe.nvfp4=<marlin|b12x|triton>",
        convert=_nvfp4_entry,
        default=argparse.SUPPRESS,
        choices=["auto", "marlin", "flashinfer", "triton"],
        help="[Deprecated] Use --quant-backend moe.nvfp4=<marlin|b12x|triton> ('flashinfer' is b12x).",
    )

    parser.add_argument(
        "--expert-load",
        default=ServerArgs.expert_load,
        choices=["auto", "serial", "parallel"],
        help=(
            "How MoE expert banks are read into host RAM. 'auto' (default) reads scattered "
            "experts in parallel (fast) but falls back to serial when free RAM can't cover "
            "the banks + the parallel reader's extra whole-shard buffer; 'serial' forces the "
            "low-memory reclaimable read (slower); 'parallel' forces the fast read."
        ),
    )
    parser.add_argument(
        "--expert-load-workers",
        type=int,
        default=ServerArgs.expert_load_workers,
        help="Parallel expert-bank reader thread count (default 16).",
    )

    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=int,
        default=ServerArgs.moe_cache_size,
        help="The number of unified MoE expert slots on GPU.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=_parse_moe_cache_rate,
        default=ServerArgs.moe_cache_rate,
        help="The fraction of all MoE experts to keep in GPU cache.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        default=ServerArgs.moe_cache_auto,
        help=(
            "Auto-pick --moe-cache-size from free VRAM and expert size, MoE-priority "
            "(KV gets --kv-reserve-tokens as a floor). Not supported for owned-KV models."
        ),
    )

    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=ServerArgs.kv_reserve_tokens,
        help="KV-cache token floor reserved before --moe-cache-auto fills experts.",
    )

    parser.add_argument(
        "--moe-cache-policy",
        default=ServerArgs.moe_cache_policy,
        choices=["lru"],
        help="The unified MoE cache eviction policy.",
    )

    parser.add_argument(
        "--moe-cpu-threads",
        type=int,
        default=ServerArgs.moe_cpu_threads,
        help=(
            "Number of CPU worker threads for --moe-strategy cpu decode experts. "
            "0 = auto (physical cores)."
        ),
    )

    parser.add_argument(
        "--moe-cpu-layers",
        type=str,
        default=ServerArgs.moe_cpu_layers,
        help=(
            "With --moe-strategy offload/hybrid: which MoE layers compute on the "
            "CPU executor instead of the GPU offload/PCIe path (where CUDA pinning "
            "is quota-capped, e.g. WSL, their banks are OS-locked instead of pinned). Explicit id list ('3,7,11'), a count ('8' = 8 "
            "layers evenly strided), a fraction ('0.5'), or 'auto'. 'auto' is for Windows/WSL "
            "only, where CUDA pinned memory is capped: it locks just enough head+tail layers "
            "for the banks over the pin budget. Any value, 'auto' included, commits to CPU "
            "decode before the model is built, so the expert format must have a CPU executor "
            "path (bf16, nvfp4, mxfp4); do not pass it on Linux. Unset = every layer on the "
            "GPU; a boot whose banks exceed a known pin budget stops and asks for this flag."
        ),
    )

    parser.add_argument(
        "--moe-hybrid-max-fetch",
        type=int,
        default=ServerArgs.moe_hybrid_max_fetch,
        help=(
            "For --moe-strategy hybrid: max experts fetched over PCIe per (layer, decode "
            "step); the rest of that step's misses are computed on the CPU, overlapped. "
            "-1 (default) = auto: fetch the benched pcie/cpu bandwidth fraction of each "
            "step's misses (perfect overlap; needs an `ft bench bw` profile, else 1). "
            "0 = never fetch (all misses on CPU); large = behaves like plain offload."
        ),
    )

    parser.add_argument(
        "--moe-verify-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="moe_verify_cpu",
        help=(
            "Compute the MTP verify's routed experts on the CPU executor (RAM-resident "
            "host banks, batched GEMV) while normal decode stays on the GPU offload path. "
            "The verify's ~k+1 new tokens would otherwise stream their experts over PCIe. "
            "Default off."
        ),
    )

    parser.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        default=ServerArgs.moe_prefill_overlap,
        help=(
            "Disable two-buffer overlap for prefill MoE expert copies. "
            "By default, prefill overlap is enabled and requires "
            "--moe-cache-size >= 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--enable-special-token-ckpt",
        action="store_true",
        dest="special_token_ckpt",
        default=ServerArgs.special_token_ckpt,
        help=(
            "Checkpoint decode state at special tokens (currently the tool-call opener). "
            "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
            "scheduler preserves a reuse point just after it (GDN: a state snapshot "
            "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
            "a client that rewrites the echoed tool call only invalidates the call body, "
            "not the turn."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hit-d2d",
        action="store_true",
        dest="moe_prefill_hit_d2d",
        default=ServerArgs.moe_prefill_hit_d2d,
        help=(
            "During prefill prefetch, copy cache-resident experts device-side into "
            "the double buffer and stream only the misses over PCIe "
            "(cudaMemcpyBatchAsync, CUDA >= 13.0). Effective with "
            "--moe-cache-size > 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--cors-origins",
        type=str,
        default=ServerArgs.cors_origins,
        help=(
            "Comma-separated CORS allow-list for browser/webview clients "
            "(default: local Tauri/Vite dev origins). '' disables, '*' allows any."
        ),
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # --mtp: resolve the draft-head mode into the env gate the model parsers read.
    # The parsers run in the spawned TP workers, which inherit this process's env;
    # --mtp-path (deprecated) folds into --mtp file.
    mtp = kwargs.get("mtp", "auto")
    mtp_header = str(kwargs.get("mtp_header") or "").strip()
    if mtp == "file" and not mtp_header:
        parser.error("--mtp file requires --mtp-header PATH")
    if mtp in ("on", "off") and mtp_header:
        parser.error(f"--mtp-header is only valid with --mtp file (got --mtp {mtp})")
    if mtp == "auto" and mtp_header:
        mtp = "file"  # a header alone means file mode (also the --mtp-path alias)
    if mtp == "file" and not os.path.exists(mtp_header):
        parser.error(f"--mtp-header {mtp_header!r} not found")
    kwargs["mtp"] = mtp
    kwargs["mtp_path"] = mtp_header if mtp == "file" else ""
    if mtp in ("on", "file"):
        os.environ["FREETOKEN_ENABLE_MTP"] = "1"
    elif mtp == "off":
        os.environ.pop("FREETOKEN_ENABLE_MTP", None)

    # env-backed runtime knobs (--mtp-draft-tokens, --load-vision, --dspark-debug, …)
    _apply_env_overrides(kwargs)

    # reject a too-long list here with a clear reason, not as a dead rank later
    if len(kwargs["gpu"]) not in (0, kwargs["tensor_parallel_size"]):
        if kwargs["tensor_parallel_size"] == 1 and len(kwargs["gpu"]) > 1:
            parser.error("tensor parallelism is not supported yet: --gpu takes one entry")
        parser.error(
            f"--gpu has {len(kwargs['gpu'])} entries but --tensor-parallel-size is "
            f"{kwargs['tensor_parallel_size']}; give one entry per TP rank"
        )

    # --memory-ratio takes a single float or a per-GPU list; a list is the same
    # override set as --gpu-memory-ratio (combining both is ambiguous).
    if isinstance(kwargs["memory_ratio"], tuple):
        if kwargs["gpu_memory_ratio"]:
            parser.error(
                "--memory-ratio with a per-GPU list cannot be combined with "
                "--gpu-memory-ratio; give the list to one of them"
            )
        kwargs["gpu_memory_ratio"] = kwargs["memory_ratio"]
        kwargs["memory_ratio"] = ServerArgs.memory_ratio

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs["shell_mode"] = run_shell
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    # the old flag stands in for one --quant-backend entry; next to the real flag it is a usage error
    entry = kwargs.pop("nvfp4_backend", None)
    if entry is not None:
        if kwargs["quant_backend"] is not None:
            parser.error("--nvfp4-backend cannot be combined with --quant-backend; write --quant-backend moe.nvfp4=... instead")
        if entry:
            kwargs["quant_backend"] = entry

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["served_model_name"] is None:
        kwargs["served_model_name"] = (
            os.path.basename(os.path.normpath(kwargs["model_path"])) or kwargs["model_path"]
        )

    if kwargs["tool_call_parser"] == "auto":
        kwargs["tool_call_parser"] = _infer_tool_call_parser(kwargs["model_path"])

    if kwargs["reasoning_parser"] == "auto":
        kwargs["reasoning_parser"] = _infer_reasoning_parser(kwargs["model_path"])
    elif kwargs["reasoning_parser"] == "off":
        kwargs["reasoning_parser"] = None

    # Offload-family backends (offload/cpu/hybrid) need a slot cache; if the user gave no
    # sizing flag at all, default to --moe-cache-auto so a bare `ft serve <FTW MoE>` works
    # out of the box (the scheduler resolves the size from free VRAM). Explicit
    # size/rate/auto is preserved.
    from freetoken.moe import is_offload_moe_strategy

    _no_cache_flag = (
        kwargs["moe_cache_size"] == 0
        and not kwargs["moe_cache_auto"]
        and (kwargs["moe_cache_rate"] is None or kwargs["moe_cache_rate"] == 0)
    )
    if is_offload_moe_strategy(kwargs["moe_strategy"]) and _no_cache_flag:
        kwargs["moe_cache_auto"] = True

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # "auto" (or an unspecified dtype) resolves to the checkpoint's dtype. Multimodal /
    # hybrid configs (e.g. Qwen3.5-MoE) keep it under ``text_config`` and use the newer
    # ``dtype`` key rather than top-level ``torch_dtype``, so check both; default bf16.
    if (dtype_str := kwargs["dtype"]) in ("auto", None):
        from freetoken.utils import cached_load_hf_config

        cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()
        text_cfg = cfg.get("text_config") or {}
        dtype_str = (
            cfg.get("torch_dtype") or cfg.get("dtype")
            or text_cfg.get("torch_dtype") or text_cfg.get("dtype") or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
