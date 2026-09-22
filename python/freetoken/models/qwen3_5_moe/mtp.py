"""MTP draft head (NextN) for the Qwen3.5/3.6 dense family.

The checkpoint ships one ``mtp.layers.N`` block plus head-level glue
(``pre_fc_norm_{hidden,embedding}`` + ``fc``) and a final ``mtp.norm``. The head is
always BF16 (the NVFP4/FP8 quantization stops at the main model's layers):

    e   = pre_fc_norm_embedding(embed_tokens(t))          # [T, H]
    R   = fc(cat([e, pre_fc_norm_hidden(R_last)], -1))    # [T, H]
    R'  = mtp_block(R, batch)                             # full attn + dense MLP
    out = mtp.norm(R')                                    # [T, H] -> shared lm_head

``R_next`` (the block output, pre-norm) is the next draft step's ``R_last``; ``out``
goes through the shared ``lm_head`` to sample the draft token. Mirrors the
``Qwen4ExpMTP.draft_step`` contract so the scheduler's ``MTPManager`` drives it
unchanged. All norms use the family's Gemma ``(1 + w)`` convention (baked at load).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList

from .attention import Qwen3_5Attention
from .moe import Qwen3_5DenseMLP

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


class Qwen3_5MTPBlock(BaseOP):
    """One ``mtp.layers.N`` decoder block: pre-norm full attention + dense SwiGLU MLP.

    ``layer_id`` is the attention backend slot, which starts AFTER the main stack
    (``num_layers + i``); the full group's KV map includes it when MTP is enabled."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        h = self.self_attn.forward(self.input_layernorm.forward(hidden))
        h = residual + h
        return h + self.mlp.forward(self.post_attention_layernorm.forward(h))


class Qwen3_5MTP(BaseOP):
    """The full draft head, mounted as ``model.mtp`` (checkpoint keys are ``mtp.*``).

    The embedding table and lm_head are shared with the target; only the glue + blocks
    live here."""

    def __init__(self, config: ModelConfig) -> None:
        # The draft head is always built bf16 (the checkpoint tensors are bf16).
        config = replace(config, quant=None)
        assert config.mtp_num_layers > 0, "Qwen3_5MTP needs mtp_num_layers > 0"
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.layers = OPList(
            [
                Qwen3_5MTPBlock(config, config.num_layers + i, prefix=f"mtp.layers.{i}")
                for i in range(config.mtp_num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def draft_step(
        self,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        R_last: torch.Tensor,
        token_ids: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One draft forward: hidden-state stream + sampled token -> next prediction.

        ``R_last`` is the residual state the previous forward ended on (``[T, H]``; the
        main model's last pre-norm hidden, or the previous draft's ``R_next``). Returns
        ``(R_next, sample_hidden)``: the caller feeds ``sample_hidden`` through the shared
        lm_head to sample the draft token and passes ``R_next`` into the next draft step.
        """
        del batch  # the attention backend reads the current forward batch from the ctx
        tokens = token_ids.reshape(-1)
        e = self.pre_fc_norm_embedding.forward(embed_fn(tokens))
        h = self.pre_fc_norm_hidden.forward(R_last)
        R = self.fc.forward(torch.cat([e, h], dim=-1))
        R_next = self.layers.op_list[0].forward(R)
        return R_next, self.norm.forward(R_next)


__all__ = ["Qwen3_5MTP", "Qwen3_5MTPBlock"]
