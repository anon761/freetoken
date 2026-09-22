"""DeepSeek-V4.1 DSpark MTP draft head (Phase 6, the DSV4.1 plan).

Ported from the checkpoint's own reference (``inference/model.py`` DSpark*) and the
vLLM implementation (``models/deepseek_v4_1/nvidia/dspark.py`` +
``model_executor/models/qwen3_dspark.py``). Weights live under ``mtp.{0,1,2}.*``:
three semi-autoregressive stages, each a decoder block that reuses the target's
architecture, plus a shared Markov head and a confidence head on the last stage.

This module holds the two self-contained heads first (stage 1). The draft backbone
(non-causal block attention + its own window/context KV + offloaded draft experts) is
stage 2, the scheduler verify loop stage 3.
"""

from __future__ import annotations

import os

import torch

from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, get_freqs_cis
from freetoken.utils import div_even

from .args import DeepseekV41Args
from .hc import hc_add, hc_collapse, hc_mixes, hc_split
from .moe import MoE


class DSparkMarkovHead(BaseOP):
    """Intra-block token dependency: ``logits = base_logits + w2(w1(prev_token))``.

    ``embed`` is the token->rank embedding (``markov_head.embed``), ``head`` the
    rank->vocab projection (``markov_head.head``). Both are replicated (no TP) in
    the reference; the vocab axis is sharded here so a rank's head gathers like the
    LM head does.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        self.embed = VocabParallelEmbedding(vocab_size, markov_rank)
        self.head = ParallelLMHead(vocab_size, markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed.forward(token_ids)
        logits = self.head.forward_all(embed)  # every position, not the last row
        return logits, embed


class DSparkConfidenceHead(BaseOP):
    """Per-position confidence: ``sigmoid(proj([hidden, markov_embed]))`` (fp32 proj).

    The checkpoint stores a bias-less ``proj`` (reference ``Linear(..., bias=False)``),
    kept in fp32 for the fp32 confidence score."""

    def __init__(self, input_dim: int):
        self.proj = LinearReplicated(input_dim, 1, has_bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return torch.sigmoid(self.proj.forward(x).squeeze(-1))


__all__ = ["DSparkMarkovHead", "DSparkConfidenceHead"]


class DSparkAttention(BaseOP):
    """Non-causal draft attention (reference model.py:1032-1074).

    The draft block attends the per-layer sliding window of the *main* model's KV (fed
    from ``main_x``) plus the whole draft block (future tokens included). The two halves
    map onto the fork's two-pool sparse kernel: the window cache is ``window_pool`` (per
    row offset ``r*win``), the block KV is ``cmp_pool`` (per row offset ``r*block``)."""

    def __init__(self, config, layer_id: int, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.eps = args.norm_eps
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.sliding_window
        self.block_size = args.dspark_block_size
        self.quant_block = 32  # fp8 window/block KV scale block (reference fp8_block_size)
        tp = get_tp_info().size
        self.local_heads = div_even(self.n_heads, tp)
        self.local_groups = div_even(self.n_groups, tp)
        self.attn_sink = torch.empty(self.n_heads, dtype=torch.float32)
        rank = get_tp_info().rank
        self._sink_slice = slice(rank * self.local_heads, (rank + 1) * self.local_heads)

        self.wq_a = LinearReplicated(config.hidden_size, args.q_lora_rank, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wq_a")
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = LinearColParallelMerged(args.q_lora_rank, [self.n_heads * self.head_dim], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wq_b")
        self.wkv = LinearReplicated(config.hidden_size, self.head_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wkv")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = torch.empty(self.local_groups * self.o_lora_rank, self.local_heads * self.head_dim // self.local_groups, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(self.n_groups * self.o_lora_rank, config.hidden_size, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.wo_b")
        self.softmax_scale = self.head_dim ** -0.5
        # compress_ratio == 0 layers use the plain rope theta regime
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, 0,
            args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None
        self.window_kv_cache: torch.Tensor | None = None

    def bind(self, device: torch.device, max_batch: int) -> None:
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)
        self.window_kv_cache = torch.zeros(
            max_batch, self.window_size, self.head_dim, device=device, dtype=torch.bfloat16
        )

    def _kv(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rd = self.rope_head_dim
        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv, self.quant_block)
        return kv

    def _wo(self, o: torch.Tensor, seqlen: int) -> torch.Tensor:
        o = o.reshape(1, seqlen, self.local_groups, -1)
        wo_a = self.wo_a.view(self.local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(o)

    def _write_window(self, main_kv: torch.Tensor, start_pos: int, seqlen: int) -> None:
        """Write the last ``min(seqlen, win)`` target-aux KV of [start_pos, start_pos+seqlen)
        into the ring at their absolute ring slots."""
        win = self.window_size
        bsz = main_kv.size(0)
        keep = min(seqlen, win)
        seg = main_kv[:, seqlen - keep :]
        base = start_pos + seqlen - keep
        slots = (base + torch.arange(keep, device=main_kv.device)) % win
        self.window_kv_cache[:bsz, slots] = seg

    def seed_window(self, main_x: torch.Tensor, start_pos: int) -> None:
        """Target-aux window seed for a prompt/extend, without running the draft block."""
        seqlen = main_x.size(1)
        kv = self._kv(main_x, self._freqs_cis[start_pos : start_pos + seqlen])
        self._write_window(kv, start_pos, seqlen)

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

        rd = self.rope_head_dim
        win = self.window_size
        bsz, seqlen, _ = main_x.size()
        device = x.device

        main_kv = self._kv(main_x, self._freqs_cis[start_pos : start_pos + seqlen])
        if start_pos == 0:
            # Prefill only seeds the window KV ring (the draft is not run over the prompt).
            self._write_window(main_kv, start_pos, seqlen)
            return x

        block = x.size(1)
        freqs = self._freqs_cis[start_pos + seqlen : start_pos + seqlen + block]
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).view(bsz, block, self.local_heads, self.head_dim)
        apply_rotary_emb(q[..., -rd:], freqs)
        kv = self._kv(x, freqs)
        # Write this forward's target-aux KV for positions [start_pos, start_pos+seqlen)
        # into the ring. The reference toy loop feeds one position per forward; the spec
        # loop feeds the accepted window of the last verify, so seqlen may be > 1.
        self._write_window(main_kv, start_pos, seqlen)

        w = min(win, start_pos + seqlen)
        win_half = torch.full((bsz, block, win), -1, dtype=torch.int32, device=device)
        win_half[..., :w] = (
            torch.arange(bsz, device=device).view(bsz, 1, 1) * win
            + torch.arange(w, device=device).view(1, 1, w)
        ).to(torch.int32)
        cmp_half = (
            torch.arange(bsz, device=device).view(bsz, 1, 1) * block
            + torch.arange(block, device=device).view(1, 1, block)
        ).expand(bsz, block, block).to(torch.int32)
        topk = torch.cat([win_half, cmp_half], dim=-1)

        window_pool = self.window_kv_cache[:bsz].reshape(bsz * win, self.head_dim)
        cmp_pool = kv.reshape(bsz * block, self.head_dim)
        o = sparse_attn_paged(
            q, window_pool, cmp_pool, self.attn_sink[self._sink_slice],
            topk, win, self.softmax_scale,
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, block)


class DSparkBlock(BaseOP):
    """One DSpark stage (reference model.py:1100-1156): a decoder block whose attention
    is non-causal over the draft block, plus stage-0 input fusion and the last-stage
    Markov/confidence heads."""

    def __init__(self, config, stage_id: int, n_stages: int, args: DeepseekV41Args, *, strategy="offload", decode_target="gpu", quant_config=None, prefix=""):
        self.stage_id = stage_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.block_size = args.dspark_block_size
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.hidden_size
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)
        self.attn = DSparkAttention(config, args.n_layers + stage_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        # The draft's MoE layer id is the STAGE (0..n_stages-1), not a global layer id:
        # the draft owns a dedicated offload cache (E=128 vs the target's 384) keyed by
        # stage. The attention keeps the reference's global id (it owns its own window KV).
        self.ffn = MoE(stage_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(args.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, self.norm_eps)
        self.main_proj = None
        self.main_norm = None
        if stage_id == 0:
            self.main_proj = LinearReplicated(args.hidden_size * len(args.dspark_target_layer_ids), args.hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.main_proj")
            self.main_norm = RMSNorm(args.hidden_size, self.norm_eps)
        self.norm = None
        self.markov_head = None
        self.confidence_head = None
        if stage_id == n_stages - 1:
            self.norm = RMSNorm(args.hidden_size, self.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.hidden_size + args.dspark_markov_rank)

    def bind(self, device: torch.device, max_batch: int) -> None:
        self.attn.bind(device, max_batch)

    def seed_window(self, main_x: torch.Tensor, start_pos: int) -> None:
        self.attn.seed_window(main_x, start_pos)

    def _split(self, R, hc_fn, hc_scale, hc_base):
        return hc_split(hc_mixes(R, hc_fn, self.norm_eps), hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def forward(self, R, start_pos, pre_mix, main_x):
        if start_pos == 0:
            # Prefill only seeds this stage's window KV (reference DSparkBlock.forward).
            self.attn.seed_window(main_x, start_pos)
            return R, pre_mix
        # Port convention: R is [M, hc, dim] (M = b*block flat tokens), pre_mix [M, hc].
        pre_a, post_a, comb_a = self._split(R, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = hc_collapse(R, pre_mix, R.dtype) if pre_mix is not None else R[:, 0]
        x = self.attn_norm.forward(x)
        b = main_x.size(0)
        x = self.attn.forward(x.view(b, self.block_size, -1), start_pos, main_x)
        R = hc_add(x.reshape(-1, R.shape[-1]), R, post_a, comb_a)

        pre_f, post_f, comb_f = self._split(R, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = hc_collapse(R, pre_a, R.dtype)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x)
        R = hc_add(x, R, post_f, comb_f)
        return R, pre_f


__all__ = ["DSparkMarkovHead", "DSparkConfidenceHead", "DSparkAttention", "DSparkBlock"]


class DSparkDraft(BaseOP):
    """The 3-stage DSpark draft (reference ``Transformer.forward_spec``).

    ``embed``/``head`` are aliased to the target's embedding/LM head (set in :meth:`bind`).
    Stage 0 fuses the target aux hidden via ``main_proj``; the last stage owns the norm
    and the Markov/confidence heads."""

    def __init__(self, config, args: DeepseekV41Args, *, strategy="offload", decode_target="gpu", quant_config=None, prefix="model.mtp"):
        from dataclasses import replace

        n = args.num_nextn_predict_layers
        self.args = args
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.hc_mult = args.hc_mult
        # The engine's offload-cache walk must not reach these layers: the draft owns a
        # dedicated cache (E=128 vs the target's 384) and is wired separately.
        self.offload_excluded = True
        self.embed: VocabParallelEmbedding | None = None
        self.head: ParallelLMHead | None = None
        draft_args = replace(
            args,
            n_routed_experts=args.dspark_n_routed_experts,
            n_activated_experts=args.dspark_num_experts_per_tok,
        )
        self.layers = OPList([
            DSparkBlock(
                config, i, n, draft_args, strategy=strategy, decode_target=decode_target,
                quant_config=quant_config, prefix=f"{prefix}.{i}",
            )
            for i in range(n)
        ])

    def bind(self, device: torch.device, max_batch: int, embed, head) -> None:
        self.embed = embed
        self.head = head
        for layer in self.layers.op_list:
            layer.bind(device, max_batch)

    def state_dict(self, *, prefix: str = "", result=None) -> dict:
        """The draft's checkpoint keys are ``mtp.{i}.*`` (no ``layers`` segment): the
        checkpoint packs the draft stages contiguously after the target's, so the FTW
        reader's ``model.mtp.{i}`` names must match this module's traversal."""
        result = result if result is not None else {}
        for i, layer in enumerate(self.layers.op_list):
            layer.state_dict(prefix=f"{prefix}.{i}" if prefix else str(i), result=result)
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        for i, layer in enumerate(self.layers.op_list):
            layer.load_state_dict(
                state_dict, prefix=f"{prefix}.{i}" if prefix else str(i), _internal=True
            )
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def bind_offload_cache(self, cache) -> list:
        """Attach ``cache`` to this draft's MoE layers only.

        The draft's experts have their own count (E=128), so it must never ride the
        target's cache; binding directly (not via the recursive walk, which the
        ``offload_excluded`` marker stops here) keeps it off ``ctx.moe_offload_cache``.
        """
        layers = [block.ffn.experts for block in self.layers.op_list]
        for layer in layers:
            layer.offload_cache = cache
        return layers

    def build_offload_cache(self, banks, *, cache_size: int, device: torch.device,
                            decode_target: str = "gpu", cache_policy: str = "lru",
                            prefill_overlap: bool = False, prefill_hit_d2d: bool = False):
        """Build the draft's dedicated :class:`OffloadMoeCache` from its banks and bind it.

        The target cache is sized/populated from the target's E; the draft's differs, so
        it gets its own cache whose layout/kernel come from the draft's own expert method.
        """
        from freetoken.moe.offload_cache import OffloadMoeCache

        method = self.layers.op_list[0].ffn.experts.quant_method
        if method is not None and banks.kind is not None:
            if (banks.kind, banks.kernel) != (method.kind, method.kernel.name):
                raise ValueError(
                    f"DSpark draft banks were packed for {banks.kind} / {banks.kernel} but the "
                    f"draft binds {method.kind} / {method.kernel.name}; reconvert or select that kernel"
                )
        cache = OffloadMoeCache(
            num_layers=len(self.layers.op_list),
            num_experts=self.args.dspark_n_routed_experts,
            cache_size=cache_size,
            device=device,
            cache_policy=cache_policy,
            prefill_overlap=prefill_overlap,
            prefill_hit_d2d=prefill_hit_d2d,
            quant_format=banks.quant_format,
            decode_target=decode_target,
            layout=method.layout() if method is not None else None,
            max_slots=method.slot_limit() if method is not None else None,
        )
        cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        self.bind_offload_cache(cache)
        return cache

    def seed_window(self, main_hidden: torch.Tensor, start_pos: int) -> None:
        """Seed every stage's window ring from the target aux hidden over a prompt/extend,
        without running a draft block (the prefill hook)."""
        first = self.layers.op_list[0]
        main_x = first.main_norm.forward(first.main_proj.forward(main_hidden))
        for layer in self.layers.op_list:
            layer.seed_window(main_x, start_pos)

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """``main_hidden``: target aux hidden ``[b, seqlen, dim*len(target_layers)]`` (the
        checkpoint reference captures it per target-layer INPUT and concatenates over
        layers). ``input_ids``: the anchor token per row ``[b]`` (the target's freshly
        sampled token); the block is filled with ``noise_token_id`` after it.

        Returns the hc-expanded draft streams ``[b*block, hc, dim]`` (the port's flat-token
        convention, matching ``hc_mixes``) and ``main_x [b, seqlen, dim]``."""
        first = self.layers.op_list[0]
        main_x = first.main_norm.forward(first.main_proj.forward(main_hidden))
        bsz = input_ids.size(0)
        draft_ids = input_ids.reshape(bsz).new_full((bsz, self.block_size), self.noise_token_id)
        draft_ids[:, 0] = input_ids.reshape(bsz)
        # VocabParallelEmbedding takes 1-D indices; the block is [b, block].
        x = self.embed.forward(draft_ids.reshape(-1)).view(bsz, self.block_size, -1)
        R = x.reshape(-1, 1, x.shape[-1]).expand(-1, self.hc_mult, -1).contiguous()
        return R, main_x

    def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int = 0):
        h, main_x = self.forward_embed(main_hidden, input_ids)
        pre_mix = None
        for layer in self.layers.op_list:
            h, pre_mix = layer.forward(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.forward_head(h, pre_mix, input_ids)

    def forward_head(self, x: torch.Tensor, pre_mix, input_ids: torch.Tensor):
        block = self.block_size
        b = input_ids.size(0)
        last = self.layers.op_list[-1]
        h = hc_collapse(x, pre_mix, x.dtype) if pre_mix is not None else x[:, 0]
        hb = h.reshape(-1, h.shape[-1])
        logits = self.head.forward_all(last.norm.forward(hb)).view(b, block, -1)
        output_ids = input_ids.new_empty(b, block + 1)
        output_ids[:, 0] = input_ids.reshape(b)
        # Diagnostic: /tmp/dspark-no-markov ablates the Markov transition bias so we can
        # see whether that head is helping or corrupting the draft argmax.
        ablate_markov = os.path.exists("/tmp/dspark-no-markov")
        markov_embeds = []
        for i in range(block):
            bias, markov_embed = last.markov_head.forward(output_ids[:, i])
            if not ablate_markov:
                logits[:, i] = logits[:, i] + bias
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = logits[:, i].argmax(dim=-1)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = last.confidence_head.forward(h.view(b, block, -1), markov_embed)
        return output_ids, logits, confidence


__all__ = [
    "DSparkMarkovHead", "DSparkConfidenceHead", "DSparkAttention", "DSparkBlock", "DSparkDraft",
]
