"""DeepSeek-V4.1-Flash model (Phase 3 — DSV4-style driver, the DSV4.1 plan).

mHC delayed-pre residual streams (Phase-1 math, checkpoint-named weights), the
CSA2 attention driver (Phase 3a: window + shared compressed caches, static
candidate selection) and the offloaded-MXFP4 MoE, wired onto the DSV4 paged
engine exactly like DeepSeek-V4-Flash's model: ragged batched prefill with
per-request compressor carry, decode with the layer-invariant window context
and device-bounded candidate counts, CUDA-graph-safe.
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.models.deepseek_v4.ops import get_freqs_cis
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel

from contextlib import contextmanager

from .args import DeepseekV41Args
from .attention import Attention
from .dspark import DSparkDraft
from .engram import Engram
from .hc import hc_add, hc_collapse, hc_mixes, hc_split
from .moe import MoE


def _decode_ngram_ids(hist: list, t: int, p: int) -> list[int]:
    """The decode 4-gram in the reference's NEWEST-first hash order.

    ``t`` is the pending token at position ``p-1`` (``hist[p-1]``); the lookback is
    ``[p-2, p-3, p-4]``. Missing positions past the sequence start use the engram pad
    RAW id 2 (matches the prefill ``Engram.forward``'s ``blocked``/pad slots; see
    ``inference/engram.py:171``).
    """
    return [
        int(t),
        int(hist[p - 2]) if p >= 2 else 2,
        int(hist[p - 3]) if p >= 3 else 2,
        int(hist[p - 4]) if p >= 4 else 2,
    ]


class _DSparkAuxCapture:
    """Accumulates the target aux hidden the DSpark draft fuses via ``main_proj``.

    Reference (vLLM ``DeepseekV4Model.forward``): after target layer ``L`` the
    post-block HC stream is collapsed by mean over its copies; the per-layer
    ``[T, hidden]`` results are concatenated in ``dspark_target_layer_ids`` order ->
    ``[T, hidden * len(ids)]``. For v4.1 the ids are the attention INPUTS of layers
    ``L``, i.e. captured after layer ``L-1`` (see PR #56214)."""

    def __init__(self, target_layer_ids):
        self.target_layer_ids = tuple(int(i) for i in target_layer_ids)
        self._parts: list[tuple[int, torch.Tensor]] = []

    def reset(self) -> None:
        self._parts = []

    def note(self, layer_id: int, streams: torch.Tensor) -> None:
        """Capture layer ``layer_id``'s post-block HC stream if it feeds a target layer."""
        target = layer_id + 1
        if target in self.target_layer_ids:
            self._parts.append((target, streams.mean(dim=1)))

    def concat(self) -> torch.Tensor | None:
        if not self._parts:
            return None
        by_id = dict(self._parts)
        return torch.cat([by_id[t] for t in self.target_layer_ids], dim=-1)


class Block(BaseOP):
    """Decoder block: delayed-pre mHC around the attention and MoE sublayers."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        model_config,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.hidden_size
        self.attn = Attention(model_config, layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.ffn = MoE(
            layer_id,
            args,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.attn_norm = RMSNorm(args.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, self.norm_eps)
        # Engram conditional memory on layers [1, 14]: additive write into the raw
        # hc stream at the top of the block (reference model.py:1261-1267)
        if layer_id in args.engram_layer_ids:
            self.engram: Engram | None = Engram(model_config, layer_id, args, quant_config=quant_config, prefix=f"{prefix}.engram")
        else:
            self.engram = None
        hc_mult = args.hc_mult
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.hidden_size
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _split(self, R: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        mixes = hc_mixes(R, hc_fn, self.norm_eps)
        pre, post, comb = hc_split(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        return pre, post, comb

    # ----- prefill (ragged batched; x streams are [M, hc, dim], M = flat tokens) -----
    def prefill_batched(self, R, pre_mix, segments, flat_positions, shared):
        if self.engram is not None:
            rows = torch.cat([torch.full((n,), ti, dtype=torch.int64, device=R.device)
                              for _off, n, ti, _start in segments])
            R = self.engram.forward(R, shared["comp_ids"], rows, flat_positions)
        pre_a, post_a, comb_a = self._split(R, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = hc_collapse(R, pre_mix, R.dtype) if pre_mix is not None else R[:, 0]
        x = self.attn_norm.forward(x)
        x = self.attn.forward_ragged(x.unsqueeze(0), segments, flat_positions, shared)
        R = hc_add(x, R, post_a, comb_a)

        pre_f, post_f, comb_f = self._split(R, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = hc_collapse(R, pre_a, R.dtype)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x)
        R = hc_add(x, R, post_f, comb_f)
        return R, pre_f

    # ----- decode (x streams are [B, hc, dim], B = batch rows) -----
    def decode_step(self, R, pre_mix, pos, rows, cmp_stage_cap, wctx, shared):
        if self.engram is not None:
            R = self.engram.decode_consume(R)
        pre_a, post_a, comb_a = self._split(R, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = hc_collapse(R, pre_mix, R.dtype) if pre_mix is not None else R[:, 0]
        x = self.attn_norm.forward(x)
        x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx, shared)
        R = hc_add(x, R, post_a, comb_a)

        pre_f, post_f, comb_f = self._split(R, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = hc_collapse(R, pre_a, R.dtype)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x)
        R = hc_add(x, R, post_f, comb_f)
        return R, pre_f


class Transformer(BaseOP):
    def __init__(
        self,
        args: DeepseekV41Args,
        model_config,
        quant_config=None,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        prefix: str = "",
    ):
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.embed = VocabParallelEmbedding(args.vocab_size, args.hidden_size)
        self.layers = OPList(
            [
                Block(
                    i,
                    args,
                    model_config,
                    strategy=strategy,
                    decode_target=decode_target,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = RMSNorm(args.hidden_size, self.norm_eps)
        self.head = ParallelLMHead(args.vocab_size, args.hidden_size, quant_config=quant_config, prefix=f"{prefix}.head")
        # DSpark draft head (opt-in via FREETOKEN_ENABLE_MTP): mounted here (not on the
        # CausalLM) so its dense keys are ``model.mtp.{i}`` like the checkpoint ships and
        # the FTW reader replays. embed/head are aliased to the target's in bind().
        self.mtp: DSparkDraft | None = None
        if args.mtp_enabled:
            self.mtp = DSparkDraft(
                model_config,
                args,
                strategy=strategy,
                decode_target=decode_target,
                quant_config=quant_config,
            )
        self.aux_capture = (
            _DSparkAuxCapture(args.dspark_target_layer_ids) if self.mtp is not None else None
        )
        self._dspark_aux_hidden: torch.Tensor | None = None

    def bind(self, pool, device: torch.device) -> None:
        # two rope regimes (ratio-0 plain theta vs compressed yarn), one table each,
        # shared by the layers of that regime; sized to the SERVED ceiling
        # (args.max_seq_len is _adjust_dsv4_config's resolved runtime cap).
        args = self.args
        rd = args.qk_rope_head_dim
        tables: dict[tuple, torch.Tensor] = {}
        for layer in self.layers.op_list:
            key = layer.attn._freqs_params
            if key not in tables:
                tables[key] = get_freqs_cis(*key, device)
        for layer in self.layers.op_list:
            layer.attn.bind_pool(pool, device, tables[layer.attn._freqs_params])
        # Engram device pieces (vocab/cache/graph staging on the real device)
        for layer in self.layers.op_list:
            if layer.engram is not None:
                layer.engram.rebind(
                    device, pool, args.max_seq_len, pool.full_loc_map.shape[0],
                    max_extend_tokens=args.max_seq_len,
                )
        # DSpark draft: alias the target's embed/head and size its window KV to the
        # served decode batch (max_running_req + 1 dummy, set by _adjust_dsv4_config).
        if self.mtp is not None:
            self.mtp.bind(device, args.max_batch_size, self.embed, self.head)

    def prefill_batched(self, input_ids: torch.Tensor, segments, flat_positions, shared: dict, logits_all: bool = False) -> torch.Tensor:
        hc = self.hc_mult
        e = self.embed.forward(input_ids.view(-1))  # [T, dim]
        R = e.unsqueeze(1).expand(-1, hc, -1).contiguous()
        pre_mix: torch.Tensor | None = None
        capture = self.aux_capture
        if capture is not None:
            capture.reset()
        for layer in self.layers.op_list:
            R, pre_mix = layer.prefill_batched(R, pre_mix, segments, flat_positions, shared)
            if capture is not None:
                capture.note(layer.layer_id, R)
        if capture is not None:
            self._dspark_aux_hidden = capture.concat()
        # model exit: collapse with the LAST FFN pre mix (no learned hc_head on V4.1)
        x = hc_collapse(R, pre_mix, R.dtype)
        x = self.norm.forward(x)
        # all-position logits for a speculative verify (every row, not the last per request)
        return self.head.forward_all(x) if logits_all else self.head.forward(x)

    def decode(self, input_ids: torch.Tensor, pos: torch.Tensor, cmp_stage_cap: int, shared: dict) -> torch.Tensor:
        B = input_ids.size(0)
        rows = torch.arange(B, device=input_ids.device)
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        hc = self.hc_mult
        e = self.embed.forward(input_ids.view(-1))  # [B, dim]
        R = e.unsqueeze(1).expand(-1, hc, -1).contiguous()
        pre_mix: torch.Tensor | None = None
        for layer in self.layers.op_list:
            R, pre_mix = layer.decode_step(R, pre_mix, pos, rows, cmp_stage_cap, wctx, shared)
        x = hc_collapse(R, pre_mix, R.dtype)
        x = self.norm.forward(x)
        return self.head.forward(x)


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine adapter: a registered :class:`BaseLLMModel` wrapping the V4.1
    transformer. KV pools, rope tables and compressor carries bind on the first
    forward (the DSV4 driver contract)."""

    def __init__(self, config):
        self._config = config
        args: DeepseekV41Args = config.dsv4_args
        self._args = args
        self.model = Transformer(
            args,
            config,
            quant_config=config.quant,
            strategy=config.moe_strategy,
            decode_target=config.decode_target,
            prefix="model",
        )
        self._bound = False

    def _comp_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compressed engram ids for this step's tokens (bound after bind)."""
        layer = next(l for l in self.model.layers.op_list if l.engram is not None)
        return layer.engram._vocab[input_ids.view(-1).long()]

    def dspark_draft(self):
        """The DSpark draft module, or None when MTP is off (engine hook)."""
        return self.model.mtp

    def dspark_compressors(self) -> list:
        """The kv-source V41Compressors carrying per-layer state (rollback hook)."""
        return [b.attn.compressor for b in self.model.layers.op_list if b.attn.compressor is not None]

    def get_dspark_aux_hidden(self) -> torch.Tensor | None:
        """Target aux hidden of the last prefill/verify forward: ``[T, hidden * len(
        dspark_target_layer_ids)]`` — the DSpark ``main_proj`` input. None when MTP is
        off or no prefill has run yet."""
        return self.model._dspark_aux_hidden

    def load_host_tables(self, config) -> int:
        """Engram disk tables: the compressed vocab, hash layout and the per-layer
        O_DIRECT row sources + pinned decode staging. The ~189 GiB tables stay in
        the raw checkpoint's shards (O_DIRECT pread) — no VRAM, no pin budget."""
        engram_layers = [l for l in self.model.layers.op_list if l.engram is not None]
        if not engram_layers:
            return 0
        from .engram import EngramLayout, build_compressed_vocab

        args = self._args
        layout = EngramLayout(args)
        doc = build_compressed_vocab(
            args.model_path, args.vocab_size, args.engram_compressed_vocab_size,
            args.engram_pad_token_id,
        )
        comp_cpu = doc["compressed_ids"]
        max_bs = max(16, config.cuda_graph_max_bs or 0)
        cache_rows = 64  # sized properly at bind (pool exists then); placeholder here
        for layer in engram_layers:
            layer.engram.bind(
                torch.device("cpu"), layout, torch.zeros(args.vocab_size, dtype=torch.int64),
                args.max_seq_len, cache_rows, max_extend_tokens=args.max_seq_len,
                comp_cpu=comp_cpu, max_graph_bs=max_bs,
            )
        self._engram_bound = False
        return 0

    @contextmanager
    def forward_host_ctx(self, batch, use_graph: bool):
        """Engine hook around every dispatch: the decode path's engram rows are
        pre-read on the HOST thread (os.pread is illegal inside a captured graph)."""
        if batch.is_decode:
            reqs = list(batch.reqs)
            ids_rows = []
            for r, t in zip(reqs, batch.input_ids.view(-1).tolist()):
                p = r.device_len  # the sampled token's position (t = hist[p-1])
                hist = r.input_ids.tolist()
                ids_rows.append(_decode_ngram_ids(hist, t, p))
            for layer in self.model.layers.op_list:
                if layer.engram is not None:
                    layer.engram.host_fill_decode(ids_rows)
        yield

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self.model.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward (runtime pool rebuild swaps the
        pool object; the model holds no buffers — everything reads via ctx)."""
        self._bound = False

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            shared = {"comp_ids": self._comp_ids(input_ids)}
            return self.model.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long(), shared
            )
        B = batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        shared = {"comp_ids": self._comp_ids(input_ids)}
        return self.model.decode(input_ids.view(B, 1), pos, cmp_stage_cap, shared)

    def forward_logits_all(self) -> torch.Tensor:
        """Like ``forward`` but WITHOUT the per-request last-row selection: full
        ``[T, vocab]`` logits for every forwarded position (DSpark verify path)."""
        self._ensure_bound()
        batch = get_global_ctx().batch
        assert batch.is_prefill, "forward_logits_all expects a prefill-phase verify batch"
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        shared = {"comp_ids": self._comp_ids(input_ids)}
        return self.model.prefill_batched(
            input_ids.view(1, -1), md.segments, batch.positions.long(), shared, logits_all=True
        )


__all__ = ["Block", "DeepseekV41ForCausalLM", "Transformer"]
