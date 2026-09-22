"""Differential harness: DSpark verify (prefill path) vs plain decode.

Narrows the sources of the greedy divergence between the speculative verify
(``Attention.forward_ragged`` / ``V41Compressor.compute_extend``) and a plain
decode step (``Attention.decode_step`` / ``V41Compressor.decode_step``) WITHOUT
booting the model: it isolates the three structural numeric differences the
source already hints at, so a model-level diff can target the real culprit.

Three probes, each printing bit-equality and the max |delta|:

  1. compressor linear: ``bf16_linear_fp32`` (decode, M==1 GEMV) vs
     ``F.linear(x.float(), w.float())`` (the prefill/extend path). The kernel
     docstring claims "same precision, only accumulation order differs".
  2. sparse-attn kernel variant: ``sparse_attn_paged`` on the same data with
     m==1 (decode -> flash-decoding split-k) vs m>1 (verify -> single kernel).
  3. candidate order: the same window candidate SET in decode ring-slot order
     vs ascending-position order (what ``forward_ragged`` builds).

Run on the GPU box:
    CUDA_VISIBLE_DEVICES=0 python scripts/dspark_verify_diff.py
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged, split_count


def _report(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    ref = ref.float()
    got = got.float()
    if not (torch.isfinite(ref).all() and torch.isfinite(got).all()):
        print(f"  {name:<44} NON-FINITE output (check pool/index bounds)")
        return
    same = torch.equal(ref, got)
    delta = (ref - got).abs().max().item()
    rel = (delta / ref.abs().max().clamp_min(1e-12)).item()
    print(f"  {name:<44} bitexact={same!s:<5} max|d|={delta:.3e} rel={rel:.3e}")


def probe_linear(device: torch.device) -> None:
    torch.manual_seed(0)
    K, N = 4096, 2048
    w = (torch.randn(N, K, device=device) * 0.02).to(torch.bfloat16)
    print("1. compressor linear bf16_linear_fp32 vs F.linear(x.float, w.float)")
    for M in (1, 2, 5):
        x = torch.randn(M, K, device=device)
        got = bf16_linear_fp32(x, w)
        ref = torch.nn.functional.linear(x.float(), w.float())
        _report(f"M={M}", ref, got)


def probe_kernel_variant(device: torch.device) -> None:
    """Decode with a single request selects the flash-decoding split-k variant
    (m==1); the verify always runs the single-program kernel (m>1). Compare the
    same query row under both variants. Production trims the shared-memory cost by
    storing the KV pools in bf16 -- the pools here must match or the kernel's fp32
    staging overflows the ~99 KiB sm_86 budget."""
    d, h, topk, n_window = 512, 64, 640, 128
    torch.manual_seed(1)
    q = torch.randn(1, 2, h, d, device=device, dtype=torch.bfloat16)
    win = (torch.randn(n_window, d, device=device, dtype=torch.bfloat16) * 0.1)
    cmp = (torch.randn(topk - n_window, d, device=device, dtype=torch.bfloat16) * 0.1)
    sink = torch.zeros(h, device=device)
    idx = torch.cat([
        torch.arange(n_window, device=device, dtype=torch.int32),
        torch.arange(topk - n_window, device=device, dtype=torch.int32),
    ]).view(1, 1, topk)
    idx2 = idx.expand(1, 2, topk).contiguous()
    print(f"2. sparse_attn_paged bf16 pools d={d} h={h} topk={topk}")
    print(f"  split_count(m=1)={split_count(1, 1, h, topk, device)}  split_count(m=2)={split_count(1, 2, h, topk, device)}")
    o1 = sparse_attn_paged(q[:, :1].contiguous(), win, cmp, sink, idx, n_window, d ** -0.5)
    o2 = sparse_attn_paged(q, win, cmp, sink, idx2, n_window, d ** -0.5)
    _report("m=1 (split-k) vs m=2 (single) row0", o2[0, 0], o1[0, 0])


def probe_candidate_order(device: torch.device) -> None:
    d, h, topk, n_window = 512, 64, 640, 128
    torch.manual_seed(2)
    # m=2 forces the single kernel (same variant the verify uses).
    q = torch.randn(1, 2, h, d, device=device, dtype=torch.bfloat16)
    win = (torch.randn(n_window, d, device=device, dtype=torch.bfloat16) * 0.1)
    cmp = (torch.randn(topk - n_window, d, device=device, dtype=torch.bfloat16) * 0.1)
    sink = torch.zeros(h, device=device)
    cmp_idx = torch.arange(topk - n_window, device=device, dtype=torch.int32)
    win_asc = torch.arange(n_window, device=device, dtype=torch.int32)
    # the same window slots in decode ring order: slot j holds the latest position
    # p<=pos with p%win==j; a rotation by `r` models a query mid-sequence.
    r = 37
    ring = torch.cat([torch.arange(r, n_window), torch.arange(r)]).to(torch.int32).to(device)
    idx_asc = torch.cat([win_asc, cmp_idx]).view(1, 1, -1).expand(1, 2, -1).contiguous()
    idx_ring = torch.cat([ring, cmp_idx]).view(1, 1, -1).expand(1, 2, -1).contiguous()
    o_asc = sparse_attn_paged(q, win, cmp, sink, idx_asc, n_window, d ** -0.5)
    o_ring = sparse_attn_paged(q, win, cmp, sink, idx_ring, n_window, d ** -0.5)
    print("3. sparse_attn_paged single kernel, same candidate SET, ascending vs decode ring order")
    _report("ring vs ascending", o_asc[0, 0], o_ring[0, 0])


def probe_carry_divergence(device: torch.device) -> None:
    """Feed N tokens through the ratio-2 compressor register twice: decode's
    ``bf16_linear_fp32`` projections vs the extend path's ``F.linear(x.float())``,
    pooling each closed group. Shows whether the per-token linear difference lands
    in the persisted compressed cache (the state later decodes read back)."""
    import torch.nn.functional as F

    from freetoken.kernel.triton.dsv4.compress import gated_pool

    torch.manual_seed(3)
    R, H, D, N = 2, 4096, 512, 16
    wkv = (torch.randn(D, H, device=device) * 0.02).to(torch.bfloat16)
    wg = (torch.randn(D, H, device=device) * 0.02).to(torch.bfloat16)
    x = torch.randn(N, H, device=device)

    def run(linear):
        ks = torch.zeros(1, R, D, device=device, dtype=torch.float32)
        ss = torch.zeros(1, R, D, device=device, dtype=torch.float32)
        pooled = []
        for p in range(N):
            xj = x[p : p + 1]
            kv = linear(xj, wkv).view(1, 1, D)
            sc = linear(xj, wg).view(1, 1, D)
            ks.scatter_(1, torch.tensor([[[p % R]]], device=device).expand(1, 1, D), kv)
            ss.scatter_(1, torch.tensor([[[p % R]]], device=device).expand(1, 1, D), sc)
            if (p + 1) % R == 0:
                pooled.append(gated_pool(ks, ss, torch.bfloat16))
        return torch.stack(pooled)

    dec = run(lambda a, w: bf16_linear_fp32(a, w).view(1, D))
    ext = run(lambda a, w: F.linear(a.float(), w.float()))
    print("4. compressor register carry: decode bf16_linear_fp32 vs extend F.linear")
    _report(f"pooled latents over {N} tokens", ext, dec)


def main() -> None:
    assert torch.cuda.is_available(), "requires CUDA"
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"device={torch.cuda.get_device_name(device)}")
    probe_linear(device)
    probe_kernel_variant(device)
    probe_candidate_order(device)
    probe_carry_divergence(device)


if __name__ == "__main__":
    main()
