"""CLI: convert an HF safetensors checkpoint to a FreeToken Weight (FTW) checkpoint.

    ft checkpoint --model <hf_dir> --out <ftw_dir> \
        [--dtype bfloat16] [--moe-backend offload] [--quant-backend moe.nvfp4=b12x] [--shard-gib 8] [--gpu <uuid-or-index>]

The output dir is self-contained: point the server's ``--model`` at it to load via the FTW
fast path (auto-detected).
"""

from __future__ import annotations

import argparse
import time

import torch

from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, single_gpu_arg

from .convert import convert_checkpoint

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _parse_quant_backend(value: str) -> str:
    from freetoken.layers.quantization import QuantBackend

    try:
        QuantBackend.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def main(argv: list[str] | None = None, prog: str = "freetoken.checkpoint") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--model", required=True, help="source HF safetensors checkpoint dir")
    p.add_argument("--out", required=True, help="output FTW checkpoint dir")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16",
                   help="dtype the dense weights are stored in inside the FTW (default bfloat16)")
    p.add_argument("--moe-backend", default="offload",
                   help="offload (experts -> banks) or e.g. triton (experts stay dense)")
    p.add_argument("--quant-backend", type=_parse_quant_backend, default=None,
                   help="kernel per quantized layer type, as for ft serve; the expert banks are packed for the "
                        "chosen MoE kernel and the server picks that kernel back up from the FTW")
    p.add_argument("--shard-gib", type=float, default=8.0, help="max shard size in GiB")
    p.add_argument("--mtp", choices=["auto", "on", "off", "file"], default="auto",
                   help="MTP draft head: 'auto'/'on' include the checkpoint's embedded head when its "
                        "config declares mtp layers; 'off' serves without a draft; 'file' embeds the "
                        "standalone artifact from --mtp-header (geometry-checked against the base model)")
    p.add_argument("--mtp-header", default="",
                   help="standalone MTP draft-head artifact for --mtp file: a directory (config.json + "
                        "*.safetensors) or a single .safetensors with its config.json next to it")
    p.add_argument("--include-engram", action="store_true",
                   help="also export the Engram n-gram tables into the FTW (~189 GiB extra disk); "
                        "otherwise they are read in place from the source shards at serve time")
    p.add_argument("--include-dspark", action="store_true",
                   help="also export the DSpark MTP draft head (mtp.* weights + draft expert banks) "
                        "into the FTW; otherwise the draft head is skipped at conversion")
    p.add_argument("--ple", choices=["auto", "fp8", "bf16"], default="bf16",
                   help="stored dtype of the PLE n-gram table side tables (qwen4_exp): 'bf16' "
                        "(default) keeps the table unquantized, 'fp8' quantizes it (half the disk/RAM), "
                        "'auto' preserves the source dtype. A table already in the requested dtype "
                        "rides verbatim")
    p.add_argument("--vision", choices=["on", "off"], default="on",
                   help="include the vision tower's bf16 tensors in the FTW (default on; only when "
                        "the checkpoint carries a vision_config). Serving it still requires "
                        "--load-vision; 'off' drops the tower for a smaller text-only FTW")
    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="GPU for the repack: a GPU UUID (GPU-xxxx..., as nvidia-smi -L prints) or "
                        "an nvidia-smi index (default: the first visible GPU)")
    ns = p.parse_args(argv)

    # same as ft serve --gpu: resolve, then bind by UUID at CUDA init
    try:
        assign_gpu(ns.gpu)
        device = f"cuda:{bind_assigned_gpu().index}"
    except (ValueError, RuntimeError) as e:
        p.error(str(e))

    shard_limit = int(ns.shard_gib * (1 << 30))
    shard_limit -= shard_limit % 4096  # keep aligned
    t = time.perf_counter()
    index = convert_checkpoint(
        ns.model, ns.out, dtype=_DTYPES[ns.dtype],
        moe_backend=ns.moe_backend, quant_backend=ns.quant_backend, shard_limit=shard_limit, device=device,
        include_engram=ns.include_engram, include_dspark=ns.include_dspark,
        mtp_mode=ns.mtp, mtp_header=ns.mtp_header, ple_mode=ns.ple,
        vision_mode=ns.vision,
    )
    dt = time.perf_counter() - t
    c = index["counts"]
    gib = index["total_bytes"] / (1 << 30)
    print(f"\nwrote FTW checkpoint -> {ns.out}")
    print(f"  tensors: {c['weight']} weight + {c['experts_bank']} experts_bank")
    print(f"  includes: {index.get('includes')}")
    print(f"  FTW: {gib:.2f} GiB across {len(index['shards'])} shard(s) "
          f"(<= {ns.shard_gib} GiB each)")
    print(f"  quant_format: {index['quant_format']}  fingerprint={index['fingerprint']}")
    print(f"  converted in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
