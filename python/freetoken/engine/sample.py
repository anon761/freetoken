from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def probs_from_logits(logits: torch.Tensor, sampling_params) -> torch.Tensor:
    """The engine's truncated next-token distribution for ONE request's logits ``[m, V]``.

    Same temperature scaling + top-k/top-p renormalization the Sampler applies, so the
    DSpark rejection sampler accepts against exactly the distribution a plain decode would
    sample from. Returns fp32 ``[m, V]`` normalized per row."""
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    m = logits.size(0)
    dev = logits.device
    temps = torch.full((m,), max(sampling_params.temperature, 1e-6), dtype=torch.float32, device=dev)
    probs = sampling.softmax(logits.float(), temps)
    if sampling_params.top_k >= 1:
        top_k = torch.full((m,), sampling_params.top_k, dtype=torch.int32, device=dev)
        probs = sampling.top_k_renorm_probs(probs, top_k)
    if sampling_params.top_p < 1.0:
        top_p = torch.full((m,), min(max(sampling_params.top_p, 1e-6), 1.0), dtype=torch.float32, device=dev)
        probs = sampling.top_p_renorm_probs(probs, top_p)
    return probs


def sample_residual(probs: torch.Tensor, rejected: int) -> torch.Tensor:
    """Speculative-sampling correction for a point-mass (greedy) draft: sample from
    ``normalize(max(0, probs - delta_rejected))`` (returns a 0-dim int64 tensor)."""
    resid = probs.clone()
    resid[rejected] = 0.0
    total = resid.sum()
    if float(total) <= 0.0:  # degenerate: the rejected token held all the mass
        return probs.argmax()
    return torch.multinomial(resid / total, 1)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
