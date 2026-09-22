"""MTP draft head (EAGLE-style NextN) for Qwen3.8-Flash-Next.

The checkpoint ships one ``mtp.layers.N`` block per ``mtp_num_hidden_layers`` plus
head-level glue: ``pre_fc_norm_{hidden,embedding}`` + ``fc_{hidden,embedding}`` (the
NextN input fusion) and a ``hyper_connection_mixer`` (the stream -> block-input mixer
the shared ``lm_head`` reads). Reference semantics (fastllm qwen4_exp.cpp, MTP draft
step; the checkpoint's ``mtp.*`` tensors are all BF16 -- the NVFP4 quantization stops
at the main model's layers):

    e   = fc_embedding(pre_fc_norm_embedding(embed_tokens(t)))            # [T, H]
    h   = fc_hidden(pre_fc_norm_hidden(R_last).view(T*hc, H))             # per-stream!
    R   = (h.view(T, hc, H) + e.unsqueeze(1)).flatten(1)                  # [T, hc*H]
    R'  = mtp_block(R, batch)                                             # attn + MoE
    out = hyper_connection_mixer.mix(R')[0]                               # [T, H] -> lm_head

``fc_hidden`` is applied per stream (the [2560, 2560] weight multiplies each of the
``hc_count`` streams; the embedding projection is broadcast-added onto every stream).
The next draft step consumes ``R'`` as its hidden-state input. All norms are the
checkpoint family's zero-centered (1 + w) convention -- including the pre_fc_norms
(fastllm Qwen4AddOne's them; the trained weights sit far from 0).

The MTP block's routed experts are BF16 and stacked+fused in the checkpoint
(``experts.gate_up_proj [E, 2I, H]`` / ``down_proj [E, H, I]``) -- the resident
resident bf16 ``MoELayer`` layout, so they load dense into VRAM (~2.5
GB/rank at TP=2) and run through ``routed_forward`` regardless of the global
moe_backend (which stays offload for the main model's NVFP4 experts).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.moe.fused import fused_topk
from freetoken.models.qwen3_5_moe.moe import _SharedExpert
from freetoken.layers.moe import MoELayer

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


class Qwen4ExpMTPMoE(BaseOP):
    """The MTP block's MLP: BF16 stacked routed experts + gated shared expert.

    Same module layout as ``Qwen4ExpMoE`` (identical state-dict keys), but the routed
    experts are resident BF16 ``MoELayer`` tensors -- not the offload cache's NVFP4
    banks -- so routing goes through ``fused_topk`` + ``routed_forward`` (which
    all-reduces the row-parallel partial) instead of the global moe_backend.
    """

    def __init__(self, config: ModelConfig) -> None:
        # The MTP head is unquantized even in NVFP4 checkpoints: hide dense_quant so
        # the shared expert builds bf16 linears (same trick as Qwen4ExpMoE/expert_quant).
        self.shared_expert = _SharedExpert(
            replace(config, dense_quant="none"),
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router + shared expert before the routed experts: the fused MoE kernel may
        # write into hidden_states in place (HF evaluates the shared expert first too).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        topk_weights, topk_ids = fused_topk(
            hidden_states,
            router_logits,
            self.experts.top_k,
            self.experts.renormalize,
        )
        routed = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return (routed + shared).view(num_tokens, hidden_dim)


class Qwen4ExpMTPBlock(BaseOP):
    """One MTP decoder layer: ``mtp.layers.N`` -- the Qwen4ExpDecoderLayer contract
    (``forward(R [T, hc*hidden], batch) -> R'``) with a full-attention (QSA) mixer,
    no PLE, and the MTP MoE. ``layer_id`` is the attention backend slot, which starts
    AFTER the main model's layers."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTP(BaseOP):
    """The full draft head, mounted as ``model.mtp`` (checkpoint keys are ``mtp.*``).

    The embedding table and lm_head are shared with the target model
    (``mtp_use_dedicated_embeddings=False``); only the glue + blocks live here.
    """

    def __init__(self, config: ModelConfig) -> None:
        # The draft head is always built bf16: the artifact tensors (checkpoint
        # dense stream, FTW replay, --mtp file artifact) ship bf16, and at ~2.5
        # GB/rank quantizing it is not worth a pack-at-load path.
        import dataclasses

        config = dataclasses.replace(config, quant=None)
        args = config.qwen4_args
        assert args.mtp_num_layers > 0, "Qwen4ExpMTP needs mtp_num_layers > 0"
        self.hc_count = args.hc_count
        # fastllm normalizes the packed stream state [T, hc*H] with ONE statistic
        # (plain RMSNorm over the full width), not the per-stream grouped form.
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, num_groups=1
        )
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps, num_groups=1
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # QSA backend slots start after the main model's layers.
        self.layers = OPList(
            [
                Qwen4ExpMTPBlock(config, config.num_layers + i)
                for i in range(args.mtp_num_layers)
            ]
        )

    def draft_step(
        self,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        R_last: torch.Tensor,
        token_ids: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One draft forward: hidden-state streams + sampled token -> next prediction.

        ``R_last`` is the residual state the previous forward ended on (``[T, hc*H]``;
        the main model's last layer output, or the previous draft's ``R_next``).
        Returns ``(R_next, sample_hidden)`` -- the caller feeds ``sample_hidden``
        through the shared lm_head to sample the draft token and passes ``R_next``
        into the next ``draft_step``.
        """
        tokens = token_ids.reshape(-1)
        e = embed_fn(tokens)
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(e))
        T, hc = R_last.shape[0], self.hc_count
        h = self.pre_fc_norm_hidden.forward(R_last)
        h = self.fc_hidden.forward(h.view(T * hc, -1)).view(T, hc, -1)
        R = (h + e.to(h.dtype).unsqueeze(1)).flatten(1)
        R_next = self.layers.op_list[0].forward(R, batch)
        sample_hidden, _ = self.hyper_connection_mixer.mix(R_next)
        return R_next, sample_hidden


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMTPBlock", "Qwen4ExpMTPMoE"]
