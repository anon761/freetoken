from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import (
    BaseOP,
    GatedRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
)
from freetoken.layers.quantization import QuantConfig

from .gdn_kernels import build_commit_prep, gdn_decode_fla, gdn_prefill_chunk_fla


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, *, quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        # TP: heads shard across ranks. Construction classes get the FULL sizes
        # (they divide internally); forward and the buffers carry this rank's
        # slice. The recurrent state pool derives its local widths from the
        # linear group config, so the numbers must match.
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp = get_tp_info()
        self.local_k_heads = div_even(num_k_heads, tp.size)
        self.local_v_heads = div_even(num_v_heads, tp.size)
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.local_key_dim = self.local_k_heads * head_k_dim
        self.local_value_dim = self.local_v_heads * head_v_dim
        self.local_conv_dim = 2 * self.local_key_dim + self.local_value_dim
        self.conv_kernel_size = conv_kernel_size
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        self._in_proj_split_local = [
            self.local_conv_dim, self.local_value_dim, self.local_v_heads, self.local_v_heads,
        ]
        if self._split_in_proj:
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(
                hidden_size, self._in_proj_split, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
        # Channels follow the rank's head slice.
        self.conv1d = _DepthwiseConv1d(self.local_conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast. TP-local head slices.
        self.dt_bias = torch.empty(self.local_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(self.local_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        # out_proj follows the checkpoint quant (scheme-driven). Row-parallel under
        # TP: the input is this rank's head slice, outputs all_reduced.
        if tp.size > 1:
            self.out_proj = LinearRowParallel(
                self.value_dim, hidden_size, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.out_proj",
            )
        else:
            self.out_proj = LinearReplicated(
                self.value_dim, hidden_size, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.out_proj",
            )
        # MTP-verify capture: the raw per-token kernel inputs of the last verify forward,
        # so the accepted prefix can be committed into the live state without a re-extend
        # (see _capture_verify / commit_verify). Lazy, per-op, small.
        self._mtp_capture: dict | None = None

    # ------------------------------------------------------------- MTP verify path

    def _ensure_mtp_capture(self, n: int, t: int, dtype, device) -> dict:
        cap = self._mtp_capture
        if cap is not None and cap["n"] >= n and cap["T"] >= t:
            return cap
        n, t = max(n, 1), max(t, 1)

        def empty2d(width, dt):
            return torch.empty(n, t, width, dtype=dt, device=device)

        self._mtp_capture = {
            "n": n,
            "T": t,
            "conv_in": empty2d(self.local_conv_dim, dtype),
            "q": torch.empty(n, t, self.local_k_heads, self.head_k_dim, dtype=dtype, device=device),
            "k": torch.empty(n, t, self.local_k_heads, self.head_k_dim, dtype=dtype, device=device),
            "v": torch.empty(n, t, self.local_v_heads, self.head_v_dim, dtype=dtype, device=device),
            "a": empty2d(self.local_v_heads, dtype),
            "b": empty2d(self.local_v_heads, dtype),
        }
        return self._mtp_capture

    def _capture_verify(self, conv_in, q, k, v, a, b, batch, dtype) -> None:
        """Stash the verify forward's per-token GDN inputs for ``commit_verify``. The
        verify batches are uniform (k+1 tokens per request), so the packed tensors view
        directly as [n, T, ...]."""
        reqs = batch.padded_reqs
        n = getattr(batch, "mtp_verify_n", None) or len(reqs)
        t = getattr(batch, "mtp_verify_t", None) or reqs[0].extend_len
        cap = self._ensure_mtp_capture(n, t, dtype, q.device)
        cap["conv_in"][:n, :t].copy_(conv_in.reshape(n, t, -1))
        cap["q"][:n, :t].copy_(q.reshape(n, t, self.local_k_heads, self.head_k_dim))
        cap["k"][:n, :t].copy_(k.reshape(n, t, self.local_k_heads, self.head_k_dim))
        cap["v"][:n, :t].copy_(v.reshape(n, t, self.local_v_heads, self.head_v_dim))
        cap["a"][:n, :t].copy_(a.reshape(n, t, self.local_v_heads))
        cap["b"][:n, :t].copy_(b.reshape(n, t, self.local_v_heads))

    @torch.inference_mode()
    def commit_verify(self, pool, lens, slots, prep=None) -> None:
        """Advance this layer's live conv + recurrent state over each request's accepted
        prefix (``lens[i]`` tokens from the captured verify inputs). The SSM is advanced
        one token at a time with the fused DECODE kernel -- the exact recurrence a normal
        step-at-a-time decode would run. The caller has restored the live state to the
        pre-verify boundary. No full-model re-extend.

        ``prep`` carries the layer-invariant device tensors (cumulative lengths, slots, and
        the per-step active subsets); ``commit_mtp_verify`` builds it ONCE per round so the
        GDN layers do not each pay the host->device copies."""
        cap = self._mtp_capture
        if cap is None:
            return
        device = cap["conv_in"].device
        if prep is None:
            prep = build_commit_prep(lens, slots, device)
        cu_t, idx_all, has_init, steps = prep
        cu = [0]
        cis = []
        for i, n_i in enumerate(lens):
            cis.append(cap["conv_in"][i, :n_i])
            cu.append(cu[-1] + n_i)
        self._conv_prefill(torch.cat(cis, 0), pool, cu_t, idx_all, has_init)
        li = pool.local_index(self.layer_id)
        state = pool.recurrent_states[li]
        for j, (sub, idx_sub, cu_sub) in enumerate(steps):
            q = cap["q"][sub, j].unsqueeze(0)  # [1, Bsub, H, K]
            k = cap["k"][sub, j].unsqueeze(0)
            v = cap["v"][sub, j].unsqueeze(0)
            a = cap["a"][sub, j]
            b = cap["b"][sub, j]
            gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=state, indices=idx_sub,
                cu_seqlens=cu_sub,
                scale=self.head_k_dim ** -0.5,
            )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.local_conv_dim, self.local_value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.local_v_heads, self.local_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split_local, dim=-1)
        z = z.reshape(total, self.local_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at local_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, local_conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(
                mixed, [self.local_key_dim, self.local_key_dim, self.local_value_dim], dim=-1
            )
            q = qf.reshape(1, B, self.local_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.local_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.local_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at local_k_heads, v at local_v_heads.
            qf, kf, vf = torch.split(
                mixed, [self.local_key_dim, self.local_key_dim, self.local_value_dim], dim=-1
            )
            q = qf.reshape(1, total, self.local_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.local_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.local_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.local_v_heads)
            beta = beta.float().reshape(1, total, self.local_v_heads)
            if getattr(batch, "mtp_verify", False):
                # MTP verify: stash these exact kernel inputs so the accepted prefix can be
                # replayed into the live state without a full-model re-extend (commit_verify).
                self._capture_verify(conv_in, q[0], k[0], v[0], a, b, batch, dtype)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
