<div align="center">
  <h1>FreeToken — fork</h1>
  <p><em>An anonymous fork of <a href="https://github.com/FlashML-org/FreeToken">FlashML-org/FreeToken</a></em></p>
</div>

> **Anonymous fork.** This repository is published without any personal identity. It is a
> derivative work of [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0); the
> upstream license, citation and acknowledgments are preserved below. All additions listed here
> are the fork's own work on top of upstream `main` (`3d919e9`).

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine that runs
frontier-scale open-weight models on consumer/workstation hardware. This fork extends it
with **DeepSeek-V4.1**, **vision input**, **speculative decoding**, **tensor parallelism**
and a reworked **quantization layer**, while keeping the upstream runtime intact.

---

## Contents

- [What this fork adds](#what-this-fork-adds)
- [New model family: DeepSeek-V4.1-Flash](#new-model-family-deepseek-v41-flash)
- [Vision input (Qwen3.8-Flash-Next)](#vision-input-qwen38-flash-next)
- [Speculative decoding (MTP + DSpark)](#speculative-decoding-mtp--dspark)
- [FTW fast-weight format](#ftw-fast-weight-format)
- [Tensor parallelism & memory](#tensor-parallelism--memory)
- [Quantization](#quantization)
- [Server / CLI](#server--cli)
- [Supported models](#supported-models)
- [Install / build](#install--build)
- [Changes vs upstream (full list)](#changes-vs-upstream-full-list)
- [Citation & Acknowledgment](#citation--acknowledgment)

---

## What this fork adds

| Area | Additions on top of upstream |
|---|---|
| **New model family** | DeepSeek-V4.1-Flash (`dsv41`): mHC residual streams, CSA2 sparse attention, Lightning Indexer, Engram n-gram memory, DSpark draft |
| **Vision** | Qwen3.8-Flash-Next vision tower + online image input (`image_url`), selectable PLE table dtype (bf16/fp8) |
| **Speculative decoding** | MTP draft head (embedded or external artifact), rejection sampling, commit-prefix, round chaining, n-gram combiner, verify CUDA graph |
| **FTW fast weights** | TP-slicing, side-table carry (PLE/Engram), selectable MTP head, `--include-engram` / `--include-dspark`, naive-fp8 dequant |
| **Tensor parallelism** | TP-sharded loading for `qwen4_exp` / `qwen3_5_moe` / `glm5_next`, per-rank expert banks, optional fp8 all-reduce |
| **Quantization** | config/scheme/method dialect layer, mixed-precision expert detection, compressed-tensors fp8/nvfp4 |
| **Server / CLI** | per-GPU memory ratio, env-backed knobs as CLI flags, `--swa-eviction-interval`, GPU telemetry |
| **Stability** | FTW TP band offsets, O_DIRECT band reads, scheduler spec budget, Engram hashing, reasoning/DSML parsers |

---

## New model family: DeepSeek-V4.1-Flash

Model type `deepseek_v41`, served text-only, with the same offload-MoE runtime as the rest of
the engine:

- **mHC residual streams** and a generic attention skeleton (`Phase 1`).
- **CSA2 sparse attention** — SWA-128 plus shared compressor caches and static candidates (`Phase 3a`).
- **Lightning Indexer** — top-512 over compressed rows with a candidate filter (`Phase 3b`).
- **Engram n-gram memory** — conditional-memory lookup from a ~189 GiB MXFP8 table that stays
  on disk (or RAM via `--engram-backend {disk,ram}`), with a compressed-vocabulary hash (`Phase 4`).
- **DSpark draft** — non-causal block-attention backbone with Markov + confidence heads, an
  admission controller and a fault latch, vLLM-style rejection-sampling acceptance, and
  `--dspark-verify {decode,prefill}`.

```bash
ft serve --model-path <DeepSeek-V4.1-Flash> \
  --moe-strategy offload --engram-backend disk --dspark-verify decode
```

---

## Vision input (Qwen3.8-Flash-Next)

Qwen3.8-Flash-Next (`qwen4_exp`) gains a full **vision tower** and **online image input**:

- `Qwen4ExpVisionModel` — 3-D patch embed, bilinear position-embed resampling, axial 2-D RoPE,
  27 pre-norm blocks, spatial-merge merger; `encode_images` + image-token scatter in the decoder.
- **Online serving** of OpenAI `image_url` content parts: the tokenizer worker preprocesses the
  image (Qwen2-VL `smart_resize`/normalize/patchify) and expands `<|image_pad|>`, the scheduler
  runs the tower and merges the soft tokens. Vision is opt-in (`--load-vision`) and advertised
  via `input_modalities` in `/v1/models`.
- **Selectable PLE table dtype** — `ft checkpoint --ple {auto,fp8,bf16}` (default `bf16`), served
  as a bf16 or fp8 bank (pinned-host or disk backend).

---

## Speculative decoding (MTP + DSpark)

- **MTP draft head** for Qwen3.8-Flash-Next and Qwen3.5/3.6 MoE: eager speculative decode rounds,
  a standalone draft-head loader (`--mtp-path`), `--mtp {auto,on,off,file}` + `--mtp-header`,
  rejection sampling for sampled requests, and commit-the-accepted-prefix (no full re-extend).
- **Performance levers**: one verify CUDA graph per padded batch size plus a draft-chain CUDA
  graph, overlap scheduling (the verify is the overlapped batch), pinned FLA tensor caches,
  layer-invariant tensors hoisted out of the per-layer loop, opt-in round chaining
  (`FREETOKEN_MTP_CHAIN`) and a self-history n-gram draft combiner (`FREETOKEN_MTP_NGRAM`).
- `--moe-verify-cpu` runs the verify experts on the CPU executor.

---

## FTW fast-weight format

The self-contained FTW checkpoint (fast load) gained:

- **TP-slicing** for dense replay and expert banks; correct band offsets by element size.
- **Carries everything** from the source: PLE side tables and the MTP draft head, selectable
  embedded/external; `--include-engram` / `--include-dspark`; DSpark banks under their own kind.
- **naive-fp8 dense dequant-at-load** and PLE bf16 extraction for llm-compressor checkpoints.
- Fallback to the source checkpoint when an FTW lacks PLE side tables; fp8 tables kept verbatim.
- O_DIRECT-safe band reads, parallel down-family reads, and clear rejection of unsupported fp8-dense.

---

## Tensor parallelism & memory

- **TP-sharded resident weight loading** for `qwen4_exp`, `glm5_next` and `qwen3_5_moe`,
  NVFP4 expert banks sliced per rank.
- Optional **fp8 quantized-wire all-reduce** (`FREETOKEN_TP_REDUCE_FP8`) with a group-wide format decision.
- **Per-GPU memory ratio** (`--memory-ratio 0,0.8;1,0.9`), GPU telemetry in the status lines,
  and `--gpu` to pick the device.
- Per-layer host-bank residency (split lock-CPU / pin-GPU) under a capped pin quota.

---

## Quantization

- A single **dialect layer** (`QuantConfig` → scheme → method) resolves every module's scheme;
  `detect_expert_quant` and per-family roles derive from it.
- **Mixed-precision** detection for compressed-tensors and ModelOpt exports; a shared block-fp8
  expert reader; correct handling of `naive-quantized` dense groups (dequantized to bf16).

---

## Server / CLI

- Env-backed runtime knobs exposed as CLI flags; `--swa-eviction-interval`.
- Anthropic/OpenAI-compatible APIs, DSML/reasoning parsers that respect always-think templates.

---

## Supported models

In addition to the upstream set, this fork serves:

| Model | Notes |
|---|---|
| DeepSeek-V4.1-Flash | new family (mHC, CSA2, Lightning Indexer, Engram, DSpark) |
| Qwen3.8-Flash-Next | + vision tower, online images, selectable PLE dtype |
| Qwen3.5 / 3.6 MoE | + dense MTP draft, compressed-tensors FP8 + NVFP4 hybrid, TP |
| GLM-5.3-Flash / GLM-5.2 | quantization roles via the dialect layer |

See [`docs/models.md`](docs/models.md) for the full upstream table and checkpoint links.

---

## Install / build

Same toolchain as upstream:

```bash
git clone <this-repo-url> && cd freetoken-anon
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

See [`docs/install.md`](docs/install.md) and [`docs/quickstart.md`](docs/quickstart.md).

---

## Changes vs upstream (full list)

[`CHANGELOG.md`](CHANGELOG.md) documents every addition, change and fix relative to upstream,
grouped by area (model families, MTP, FTW, TP, quantization, server, fixes, performance) with
short commit references.

---

## Citation & Acknowledgment

FreeToken (upstream) — please cite the original paper:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from [SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).
