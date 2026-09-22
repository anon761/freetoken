# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |
| `--gpu-stats`, `--no-gpu-stats` | on | Append the per-GPU telemetry segment (TP count, VRAM, utilization, temperature, power) to the status lines; omitted when NVML and `nvidia-smi` are both unavailable |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

For tensor parallelism the `--gpu` list is in TP-rank order and each rank's
memory budget can be set independently. Two spellings are accepted: `gpu:ratio`
entries separated by `,` (shell-safe, no quotes) or the older `gpu,ratio` entries
separated by `;` (quote it: `;` is a shell operator).

```bash
# TP=2: card 0 may use 80% of its free VRAM, card 1 90% (no quotes needed)
ft serve --model ... --tp-size 2 --gpu 0,1 --gpu-memory-ratio 0:0.8,1:0.9

# the same list can go on --memory-ratio (a single float there still means "all ranks")
ft serve --model ... --tp-size 2 --gpu 0,1 --memory-ratio 0:0.8,1:0.9

# legacy spelling, must be quoted
ft serve --model ... --tp-size 2 --gpu 0,1 --gpu-memory-ratio '0,0.8;1,0.9'
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV). A single float applies to every rank; a per-GPU list (`0:0.8,1:0.9`) is equivalent to `--gpu-memory-ratio` |
| `--gpu-memory-ratio` | — | Per-GPU overrides of `--memory-ratio`: `0:0.8,1:0.9` (shell-safe) or `'0,0.8;1,0.9'` (quoted), GPU named as in `--gpu`; ranks without an entry keep `--memory-ratio` |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile. `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for Windows/WSL only, where CUDA pinned memory is capped; every value needs an expert format the CPU executor serves (bf16, nvfp4, mxfp4), so fp8 experts cannot use it |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |
| `--expert-load` | auto | How expert banks are read into host RAM: `auto` (parallel for scattered experts, serial when RAM is tight), `serial`, `parallel` |
| `--expert-load-workers` | 16 | Parallel expert-bank reader thread count |

### PLE — Qwen3.8-Flash-Next n-gram table

| Flag | Default | Meaning |
|---|---|---|
| `--ple-backend` | disk | Where the ~47.7 GiB PLE table lives: `disk` (O_DIRECT from the checkpoint files) or `pinned` (preload into page-locked host RAM) |
| `--ple-sync` | auto | Disk-IO sync mode for the PLE reads (`auto`/`wait`/`gate`) |
| `--ple-io-uring` | off | Use io_uring for the PLE disk reads |
| `--qsa-torch-topk` | off | Use a torch top-k in the QSA sparse attention (threshold; 0 = off) |

### Speculative decoding — MTP / DSpark

| Flag | Default | Meaning |
|---|---|---|
| `--mtp` | auto | Draft head. `auto` follows `FREETOKEN_ENABLE_MTP`; `on` serves the draft embedded in the base checkpoint; `off` forces it off; `file` loads a standalone artifact from `--mtp-header`. `on`/`file` also disable overlap scheduling (Phase 1 needs the drain-safe non-overlap loop) |
| `--mtp-header` | — | Standalone draft-head artifact: a directory (`config.json` + `*.safetensors`) or a single `.safetensors` with its `config.json` next to it. Used with `--mtp file` |
| `--mtp-path` | — | Deprecated alias for `--mtp file --mtp-header PATH` |
| `--mtp-draft-tokens` | 3 | Draft chain length k (k+1 must stay inside one GDN chunk) |
| `--dspark-verify` | decode | DeepSeek-V4.1 DSpark verify path: `decode` aligns the attention reduction with plain decode; `prefill` keeps the ascending-window extend. Inert without the DSpark draft head |
| `--dspark-k` | draft block size | DSpark draft chain length |

### DeepSeek-V4.1 — Engram & SWA

| Flag | Default | Meaning |
|---|---|---|
| `--engram-backend` | disk | Engram table access: `disk` (buffered O_DIRECT pread) or `ram` (mmap + one bulk read into the page cache; large!) |
| `--swa-full-tokens-ratio` | 0.2 | Fraction of the KV token budget held as full SWA KV (pool sizing) |
| `--swa-eviction-interval` | 128 | Free each decoding request's out-of-window SWA slots every N decode forwards; `0` disables the eviction (legacy `FREETOKEN_SWA_NO_EVICT`) |

### Family / dev toggles

| Flag | Default | Meaning |
|---|---|---|
| `--m3-sparse` | on | MiniMax-M3 block-sparse attention (`--no-m3-sparse` = dense ablation) |
| `--m3-inner-backend` | auto | MiniMax-M3 inner attention backend override |
| `--m3-max-layers` | all | Cap the MiniMax-M3 layer count (smoke tests; 0 = all) |
| `--glm-dsa` | on | GLM-MoE-DSA sparse attention |
| `--glm-dsa-max-layers` | all | Cap the GLM-MoE-DSA layer count |
| `--glm5-dsa` | on | GLM5-Next DSA |
| `--glm5-max-layers` | all | Cap the GLM5-Next layer count |

### Advanced tuning & diagnostics

| Flag | Default | Meaning |
|---|---|---|
| `--mamba-ssm-dtype` | float32 | GDN/SSM recurrent-state dtype (`float32`/`bfloat16`/`float16`) |
| `--pin-budget-gb` | host default | Host pinned-memory budget in GiB (WSL auto budget, else none) |
| `--load-vision` | off | Build/load the model's vision tower |
| `--pynccl-max-buffer-size` | 1 GiB | PyNCCL reduce buffer, e.g. `1G` |
| `--hybrid-fetch-policy` | recency | Hybrid MoE fetch order (`recency`/`lowest_id`) |
| `--tp-reduce-fp8` | off | Use fp8 for the TP all-reduce |
| `--tp-reduce-fp8-min-bytes` | engine default | Minimum tensor bytes for the fp8 TP reduce |
| `--forward-unknown-tools` | on | Forward tool calls the parser does not recognise |
| `--no-hybrid-overlap` | overlap on | Disable the hybrid MoE PCIe/CPU overlap (serial path) |
| `--no-fused-copy` | fused copy on | Use the legacy fused-copy path |
| `--no-cpu-moe-flag-sync` | sync on | Disable CPU-MoE flag synchronization |
| `--bank-cuda-alloc` | auto | Force CUDA allocation for the host expert banks |
| `--skip-bank-pin` | off | Skip bank pinning (CPU tooling only; never when serving) |
| `--cpu-moe-isa` | auto | Override the CPU-MoE ISA (e.g. `avx512`) |
| `--api-log-dir` | off | Directory for the JSONL API request log |
| `--num-tokenizer` | 0 (auto) | Number of tokenizer workers in the frontend |
| `--dummy-weight` | off | Fabricate random weights (loader/kernel smoke tests, no checkpoint) |
| `--disable-pynccl` | off | Use torch.distributed instead of PyNCCL for TP |
| `--model-source` | huggingface | Where `--model` ids are resolved (`huggingface`/`modelscope`) |
| `--enable-special-token-ckpt` | off | Allow a checkpoint whose tokenizer lacks special tokens |
| `--cors-origins` | built-in | Extra allowed CORS origins (comma-separated) |
| `--shell-mode` | off | Interactive shell mode for the serve process |
| `--dspark-debug` / `--dspark-diff` / `--dspark-timing` / `--dspark-force-a0` | off | DSpark diagnostics: verbose draft logs / decode-vs-verify harness / per-round timing / never accept drafts |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [options]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. The FTW carries
everything the source had (including the draft head), so the server replays it
without re-reading the HF shards. See the FTW caveats in
[models.md](models.md#notes).

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--out` | required | Source HF checkpoint dir / output FTW dir |
| `--dtype` | bfloat16 | Dtype the dense weights are stored in inside the FTW |
| `--moe-backend` | offload | `offload` packs the experts into offload banks; e.g. `triton` keeps them dense for resident serving |
| `--quant-backend` | auto | Kernel per quantized layer type (as for `ft serve`); the expert banks are packed for the chosen kernel and the server picks it back up |
| `--shard-gib` | 8 | Max FTW shard size in GiB |
| `--include-engram` | off | Also export the Engram n-gram tables into the FTW (~189 GiB extra disk); otherwise they are read in place at serve time |
| `--include-dspark` | off | Also export the DSpark MTP draft head (mtp.* weights + draft expert banks); otherwise the draft head is skipped |
| `--gpu` | first visible GPU | GPU for the repack: a UUID from `nvidia-smi -L` or an index |

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-strategy auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.

