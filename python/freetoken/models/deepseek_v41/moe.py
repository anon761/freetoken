"""DeepSeek-V4.1 MoE: sqrtsoftplus/noaux_tc router, shared SwiGLU expert,
offloaded MXFP4 routed experts.

Routing mirrors deepseek_v4's Gate minus the hash branch (V4.1 has no
``n_hash_layers`` — plain top-k over bias-shifted scores; ``noaux_tc`` means the
bias steers SELECTION while the routing weights come from the unshifted scores).
The shared expert w1/w2/w3 + fused_swiglu layout matches the checkpoint's
``ffn.shared_experts.*`` keys.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel, OffloadMoELayer

from .args import DeepseekV41Args


class Gate(BaseOP):
    """MoE router: sqrtsoftplus scoring + noaux_tc bias-shifted top-k."""

    def __init__(self, args: DeepseekV41Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = torch.empty(args.n_routed_experts, args.hidden_size, dtype=torch.bfloat16)
        self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        # image-span routing bias (vision tokens compete with a separate bias);
        # INERT for text-only serving — carried so the checkpoint loads 1:1
        self.bias_vl = torch.empty(args.n_routed_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        scores = bf16_linear_fp32(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            # sqrtsoftplus (the V4.1 scoring_func)
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(BaseOP):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, *, quant_config=None, prefix: str = ""):
        self.w1 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w1")
        self.w2 = LinearRowParallel(inter_dim, dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w2")
        self.w3 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w3")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1.forward(x), self.w3.forward(x), self.swiglu_limit, x.dtype)
        return self.w2.forward(h)


class V41OffloadMoELayer(OffloadMoELayer):
    """Routed MXFP4 experts on the shared offload cache (base streaming prefill +
    slot-cache decode paths)."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.hidden_size,
            intermediate_size=args.moe_inter_dim,
            renormalize=args.norm_topk_prob,
            activation=args.hidden_act,
            limit=args.swiglu_limit,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )


class MoE(BaseOP):
    """Sparse MoE: sqrtsoftplus/noaux_tc router -> offloaded MXFP4 routed experts
    + shared expert."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config=None,
        prefix: str = "",
    ):
        self.dim = args.hidden_size
        self.gate = Gate(args)
        self.shared_experts = Expert(
            args.hidden_size,
            args.moe_inter_dim,
            args.swiglu_limit,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = V41OffloadMoELayer(
            layer_id,
            args,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate.forward(x)
        # Shared expert first: hybrid decode blocks on the CPU pool inside
        # routed_forward, so this GEMM must already be on the stream to overlap
        # the CPU overflow compute (same ordering contract as DSV4).
        shared = self.shared_experts.forward(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # .to(int32) always copies, so no clone needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        return (routed + shared).view(shape)


__all__ = ["Expert", "Gate", "MoE", "V41OffloadMoELayer"]
