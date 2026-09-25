# Changelog

All notable changes to this fork are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The single source of truth for the version is
[`python/freetoken/version.py`](python/freetoken/version.py).

Short commit hashes reference the fork's history (branch `ft-ftw`).

## [Unreleased]

Everything below landed after `v0.1.2` (2026-08-19). It currently lives on the
`ft-ftw` branch; tag it as the next release to move this section.

### Added

**DeepSeek-V4.1 (new model family)**
- Phase 1 skeleton: family, config, mHC, generic attention, offload-MoE; fp8-32x32 dialect; `ds_fp4` kernel for TP>1 (`62e641a`)
- Phase 2: real weight mapping — fp8-32x32 dense, MLA attention, MXFP4 expert banks with TP slicing (`5b83624`)
- Phase 3a (CSA2): SWA-128 plus shared compressor caches, static candidates (`7635ed2`)
- Phase 3b + 4: Lightning Indexer (top-512, candidate filter) and Engram n-gram memory (`a5a5ab9`)
- Engram tables: `--engram-backend {disk,ram}` (`9b61f3b`)
- DSpark draft backbone (non-causal block attention) and heads (Markov + confidence) (`51c3c36`, `f56640b`)
- DSpark wired into the model and engine; TP-sharded Markov head (`9ac0515`, `dbb9808`)
- DSpark spec-decode verify with arbitrary-start carry, plus the manager (`bb4b07b`)
- DSpark admission controller + fault latch (ported from DwarfStar), fed by serial-step timings (`9e5cd88`, `8567c98`)
- vLLM-style rejection-sampling acceptance for DSpark verify (`d90ca7d`)
- `--dspark-verify` flag plumbing (`67c30ee`)

**MTP (multi-token prediction / speculative decoding)**
- Qwen4Exp draft head (Phase 0) + eager speculative decode rounds (`e2eb0ec`)
- Standalone draft-head loader (`--mtp-path`) (`f9ac668`)
- `--mtp {auto,on,off,file}` and `--mtp-header` for the draft head (`9ace436`)
- Dense MTP draft head wired for `qwen3_5_moe` (`e10c45f`)
- Phase 1: rejection sampling for sampled requests (`d35035e`)
- Phase 2: commit the accepted prefix instead of re-extending (`8f95f0e`)
- `--moe-verify-cpu` (MTP verify experts on the CPU executor) (`d99f252`)
- Sampled-request MTP gated behind `FREETOKEN_MTP_SAMPLED` (default off) (`8b914e0`)
- Opt-in round chaining (`FREETOKEN_MTP_CHAIN`) (`a64e66d`)
- Self-history n-gram draft combiner (`FREETOKEN_MTP_NGRAM`) (`d28592f`)
- Frequency-adapted draft vocabulary (`--mtp-draft-vocab`) and fp8 draft head at load (`--mtp-head-fp8`) (`20d8a02`)

**FTW checkpoints**
- TP-slicing for FTW checkpoints (dense replay + expert banks) (`213389c`)
- Conversion carries everything from the original — PLE side tables + MTP draft head (`17de8f6`)
- Selectable MTP head (embedded or external artifact) for FTW conversion (`6a0601c`)
- `--include-engram` / `--include-dspark` FTW converter options (`7ee84c6`)
- DSpark draft expert banks under a separate FTW kind (`b02f9ea`)
- Clear rejection of fp8-dense checkpoints (`f492797`)
- naive-fp8 dense dequant-at-load and PLE BF16 extraction (`772ffab`)
- PLE fallback to the source when the FTW has no side tables; fp8 tables kept verbatim (`0d68dcf`)
- Qwen3.8-Flash-Next PLE table written next to the FTW (`3d919e9`)
- PLE n-gram table streamed from disk (`4c0bad3`)

**Tensor parallelism**
- TP-sharded resident weight loading/construction for `qwen4_exp` (`95d6e8a`), `glm5_next` (`06fbc11`) and `qwen3_5_moe` (`27ded66`)
- NVFP4 expert banks sliced per rank (`11bbab5`)
- Optional fp8 quantized-wire all-reduce (`FREETOKEN_TP_REDUCE_FP8`) (`2cffe0e`)
- `--gpu` to choose the GPU on multi-GPU machines (`2757bb5`)

**Server / CLI**
- Per-GPU memory ratio on `--memory-ratio`, with shell-safe syntax (`5e457ee`)
- Per-GPU memory ratio and GPU telemetry in the status lines (`1afcd8f`)
- Env-backed runtime knobs exposed as CLI flags (`f95f4a3`)
- `--swa-eviction-interval` (`0a6ac02`)

**Models**
- Qwen3.8-Flash-Next support (`bd8f3d5`, docs `a05c265`)
- GLM-5.3-Flash support (`a2538a4`)

**MoE / residency**
- Per-layer host-bank residency (split lock-CPU / pin-GPU layers, auto under a capped pin quota) (`c41833b`, `eebb3f5`, `831d38a`)
- In-repo Triton router for `fused_topk` (`e05cff8`)

### Changed
- Quantization config/scheme/method layers; `detect_expert_quant` delegates to the QuantConfig dialect, per-family roles derived from it (`477c860`, `ae0488d`, `80ccf01`, `3d10cf2`)
- Shared block-fp8 expert reader; dropped the cross-family hook (`8955e1e`)
- Default `expert_load_workers` raised 8 → 16 (saturates the NVMe) (`7e71fd8`)
- Small MoE chunks fetch routed experts instead of whole layers (`aa1930f`)
- MTP round chaining is on by default (`--no-mtp-chain` to interleave plain decode steps) (`20d8a02`)
- compressed-tensors `ignore` matches module names exactly; existing FTWs of checkpoints that ignore a container with quantized children (Qwen3.8-27B-NVFP4) must be re-converted (`50b629f`)

### Fixed
- **FTW: TP band offsets scaled by bank element size** — the fp16 `gate_up_global` bank was read with element offsets treated as bytes, zeroing half the experts at TP>1 (`9b0e853`)
- FTW: O_DIRECT-safe band reads (EINVAL on ext4) and `piece dest_off` (`cb56d9d`)
- FTW: release the numpy view before closing the down-family mmap (`e7180f9`)
- FTW: 2-D down banks (`down_global`) read fully on the TP band path (`c41bcf0`)
- FTW: pin TP-sliced banks; re-slice now disposes the full banks instead of keeping pinned pages resident (`0acc6c1`, `b65d92a`)
- FTW: a truncated shard raises `OSError` instead of silently loading garbage (`bd372b6`)
- Checkpoint: `ModelOptConfig` reads `quant_algo`/`ignore` from nested quantization (`9e8dbfa`)
- Checkpoint: `--include-dspark` tolerates a dense MTP head (`qwen4_exp`) (`099219e`)
- Quant: detect nvfp4 experts behind mixed-precision compressed-tensors (`6eca2d7`) and modelopt mixed-precision (`a2dd599`)
- Weight: drop FTW `model.mtp.*` dense keys when MTP is off (`cf9c61a`)
- `qwen4_exp`: generic fp8 storage + MTP expert normalization (`c5fd6d3`); slice GDN `dt_bias`/`A_log` per rank (`d351e88`)
- `qwen3_5_moe`: fuse GDN bf16 parts into the model's split buffers (`5b4a333`; reverted in `b879b9c`)
- DeepSeek-V4.1: `tp_shard_key` base naming, RoPE `[1,T,H,d]` layout, TP shim for single-process (`3dfb2cb`)
- DeepSeek-V4.1: drop the spurious weightless q-RMS after `wq_b` (length-dependent corruption) (`551624c`)
- DeepSeek-V4.1: e4m3 compressed-KV scale, pre-RoPE decode indexer K, fp8 window KV (`13fe9fc`)
- DeepSeek-V4.1: Engram prefill in micro-batches (fp32 transients) (`1571ccc`)
- DeepSeek-V4.1: Engram hash correction (multiplier bound from compressed vocab, global cumsum offsets) (`4e37741`)
- DeepSeek-V4.1: decode Engram 4-gram newest-first to match prefill/reference (`b377426`)
- DeepSeek-V4.1: DSpark draft uses the port's 3-D `hc`/token conventions (`74d7c79`)
- DeepSeek-V4.1: truncate the DSpark draft block to `k` so `k < block_size` works (`fa0455d`)
- DeepSeek-V4.1: commit only drafts published before a mid-draft finish (SWA leak) (`5f773d9`)
- DeepSeek-V4.1: `EngramRowSource` falls back to the raw checkpoint when the served FTW lacks the Engram shards (`3952aa8`)
- Engram: allocate the decode pinned buffer outside `inference_mode` (`db4a3ab`)
- Engine: clamp the KV budget to currently-free VRAM (side-load race) (`e66a8eb`)
- Engine: size TP pools from the cross-rank MIN and tolerate side-loaded ranks (`276e421`)
- Scheduler: measure the spec output budget on committed ids, not `device_len` (`f0b5e45`)
- Scheduler: match GPU telemetry by UUID under `CUDA_VISIBLE_DEVICES` (`fc56296`)
- Scheduler: unpack `ForwardOutput` per attribute (the streams field broke the 3-tuple unpack) (`2c90cb1`)
- TP: materialize weight slices — views kept the full base tensors resident (`870d507`)
- Kernel: exact Triton top-k/top-p sampling (`03c28d2`); avoid the row-wise `_scaled_mm` stall on sm_89 (`58f4b9e`); unbreak the nightly kernel-cache wheel build (`3a20a79`)
- FLA: stop `l2norm` recompiling per token count (`f7c31e9`)
- Tokenizer: decode a multi-message batch per message (spec rounds duplicated text) (`79e9e8b`); DSV4 encoder preferred over the mini-chat template (`32c2aff`)
- Server: DSML parser accepts the V4.1 tag spelling (`b63f5eb`); reasoning parser respects always-think chat templates (`bae5c5b`)
- MoE: report the residency banks actually settle at (`831d38a`)
- HF: download the shards the safetensors index names (`a80b4d3`)
- Qwen3.8-27B-NVFP4 served its fp8 GDN projections as bf16 (compressed-tensors `ignore` matched as a subtree) (`50b629f`)
- MTP chain refresh crashed the scheduler of a server without a draft head (`20d8a02`)
- DSpark aborted at start since the overlap integration: the scheduler now picks the loop per speculative manager (`41562b1`)
- FTW TP band load: fp8 banks crashed the in-memory slice (`056e333`); NVFP4 gate_up was sliced as one block and decoded garbage at TP=2 since `b5f9b45` (`0a04407`)

### Performance
- MTP: one verify CUDA graph per padded batch size (`13d7f89`); pin FLA tensor caches after capture (`2fd9f60`); hoist the GDN commit's layer-invariant tensors out of the per-layer loop (`0e7d171`)
- MTP round overhaul: decode-exact GDN verify + one-call commit, per-round split-KV re-plan of the verify/draft attention graphs, per-rank draft argmax; Qwen3.8-27B MTP 53 -> 88 tok/s at 1k and 29 -> 88 at 14k context (`20d8a02`)
- Split-K small-M W8A16 fp8 GEMM for batched decode / MTP verify (`42cade6`); tail-wave-aware NVFP4 small-M split on smem-bound GPUs (`7e11dc6`)
- WNA16 decode MoE GEMM: narrow tiles + deterministic split-K (`ed28aee`); one TP all_reduce for routed + shared experts (`c89ea83`) -- Qwen3.8-Flash-Next W4A16 plain decode 48.8 -> 55.2 tok/s
- `--dense-fp8`: per-row fp8 (W8A16) for the bf16 dense linears at load -- Qwen3.8-Flash-Next plain decode NVFP4 57 -> 71, W4A16 53 -> 70 tok/s at 1k (`081ba3e`)
- FTW: parallelize the down-family band read; load progress + CPU monitor; expert-load worker/flag knobs (`96697a9`)
- Expert loading defaults to 16 workers (`7e71fd8`)

### Documentation
- DeepSeek-V4.1 port plan, checkpoint inventory and session logs (S1–S16)
- MTP phase plan, round profiles and client benchmark harness
- `SECURITY.md` (`4b94bdc`), `CONTRIBUTING.md` (`f0abe58`), `AGENTS.md` & `CLAUDE.md` (`9d32fa8`)
- CLI help/docs completed for the newer flags (`f25ed56`)

### CI / Build
- Nightly wheels published to a rolling `nightly` release (`7dfc37a`), plus `engine-<platform>.json` manifests (`af71ba4`)
- Issue templates with FAQ/Roadmap checks and auto-labels (`86214a9`)
- sm_80 (A100/A800) added to the default kernel-cache arches (`184a4f1`)

### Tests
- Standalone harness `scripts/run-tests.sh` (`80d2b0b`)
- Real O_DIRECT unaligned band-read regression test (ext4) (`f95043e`)
- DSpark verify-vs-decode differential harness and model-level diff diagnostic (`f98b6d1`, `e77fb90`)
- Robust GPU/NVML handling; tvm-ffi arch fallback from torch (`2e88675`)

### Reverted
- `fix(quant)`: match compressed-tensors `ignore` exactly (`2d7263b` reverts `cd90b60`)
- `fix(qwen3_5_moe)`: fuse GDN bf16 parts (`b879b9c` reverts `5b4a333`)

## [0.1.2] - 2026-08-19

Upstream release this fork starts from (base for the `[Unreleased]` section above).
