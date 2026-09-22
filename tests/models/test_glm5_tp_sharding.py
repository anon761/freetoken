# Copyright (c) 2026 FreeToken contributors
# glm5_next TP sharding: slice helpers reconstruct the full tensors, and the
# TP>1 module construction allocates exactly the shapes the loader emits.
#
# CPU-only: builds the ops with tiny dims under set_tp_info(0, 1/2) and checks
# the geometry contract between weight.py (emitted slices) and the modules
# (local allocations). The GPU end-to-end is covered by the dummy-weight smoke.
import pytest
import torch

from freetoken.distributed import DistributedInfo, set_tp_info
from freetoken.models.glm5_next.args import Glm5NextArgs
from freetoken.models.glm5_next.attention import Glm5NextAttention
from freetoken.layers.quantization import NoQuantConfig
from freetoken.models.glm5_next.kda import Glm5NextKDA
from freetoken.models.glm5_next.mlp import Glm5NextGatedMLP
from freetoken.models.glm5_next.weight import (
    _col_rows,
    _kda_conv_local,
    _kda_in_proj_local,
    _row_cols,
    _vocab_rows,
)


def _tp(rank: int, size: int) -> DistributedInfo:
    return DistributedInfo(rank, size)


def _set_tp(rank: int, size: int) -> None:
    """set_tp_info is one-shot per process; reset the module state for tests."""
    import freetoken.distributed.info as info

    info._TP_INFO = None
    set_tp_info(rank, size)


def _args() -> Glm5NextArgs:
    return Glm5NextArgs(
        hidden_size=64,
        num_heads=8,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        mla_nope=True,
        norm_eps=1e-5,
        max_position=4096,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=16,
        indexer_types=("full",),
        indexer_rope_interleave=False,
        index_kpool=2,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        linear_num_heads=8,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=1e-30,
        layer_types=("linear_attention", "deepseek_sparse_attention"),
        mlp_layer_types=("dense", "sparse"),
        mhc=False,
        mhc_num_residual_streams=2,
        hc_eps=1e-5,
        mhc_sinkhorn_iterations=2,
        mhc_tau=1.0,
        mhc_post_mult_value=1.0,
        mhc_no_norm_weight=False,
        swiglu_limit=10.0,
        rope_theta=10000.0,
    )


class _Cfg:
    def __init__(self):
        self.glm5_args = _args()
        self.attn_quant = "none"
        self.quant = NoQuantConfig()


# --- slice helpers: rank slices concatenate back to the full tensor ----------


class TestSliceHelpers:

    def test_col_rows_roundtrip(self):
        w = torch.randn(10, 4)
        full = torch.cat([_col_rows(w, _tp(r, 2), 10) for r in range(2)], dim=0)
        assert torch.equal(full, w)
        assert _col_rows(w, _tp(0, 2), 10).shape == (5, 4)

    def test_row_cols_roundtrip(self):
        w = torch.randn(4, 10)
        full = torch.cat([_row_cols(w, _tp(r, 2), 10) for r in range(2)], dim=1)
        assert torch.equal(full, w)
        assert _row_cols(w, _tp(1, 2), 10).shape == (4, 5)

    def test_vocab_rows_pads_last_rank(self):
        w = torch.randn(7, 3)
        r0, r1 = _vocab_rows(w, _tp(0, 2), 7), _vocab_rows(w, _tp(1, 2), 2)[
            :0
        ]  # placeholder, replaced below
        # n_tp = ceil(7/2) = 4: rank 0 rows 0..4, rank 1 rows 4..7 padded to 4
        r0 = _vocab_rows(w, _tp(0, 2), 7)
        r1 = _vocab_rows(w, _tp(1, 2), 7)
        assert r0.shape == (4, 3) and r1.shape == (4, 3)
        assert torch.equal(torch.cat([r0, r1], dim=0)[:7], w)
        assert torch.equal(r1[3:], torch.zeros(1, 3))

    def test_vocab_rows_exact_division(self):
        w = torch.randn(8, 3)
        full = torch.cat([_vocab_rows(w, _tp(r, 2), 8) for r in range(2)], dim=0)
        assert torch.equal(full, w)


# --- KDA fused in_proj / conv: head parts slice, bottlenecks replicate -------


def _kda_parts(hidden=64):
    p, h, d = 8 * 16, 8, 16  # proj_size, heads, head_dim
    mk = torch.randn
    return [mk(p, hidden), mk(p, hidden), mk(p, hidden), mk(h, hidden), mk(d, hidden), mk(d, hidden)]


class TestKdaFusedSlices:

    def test_in_proj_head_parts_reconstruct(self):
        parts = _kda_parts()
        fused = torch.cat(parts, dim=0)
        p, h, d = 128, 8, 16
        loc0 = _kda_in_proj_local(parts, _tp(0, 2), h, d)
        loc1 = _kda_in_proj_local(parts, _tp(1, 2), h, d)
        # local rows: 3*p_loc + h_loc + 2*d (f_a|g_a replicated)
        assert loc0.shape == (3 * 64 + 4 + 2 * 16, 64)
        # head-structured parts concatenate back to full ...
        assert torch.equal(torch.cat([loc0[:64], loc1[:64]]), parts[0])
        assert torch.equal(torch.cat([loc0[128:192], loc1[128:192]]), parts[2])
        assert torch.equal(torch.cat([loc0[192:196], loc1[192:196]]), parts[3])
        # ... the bottlenecks are replicated verbatim ...
        assert torch.equal(loc0[196:212], parts[4])
        assert torch.equal(loc1[196:212], parts[4])
        # ... and at TP=1 the fusion is unchanged.
        assert torch.equal(_kda_in_proj_local(parts, _tp(0, 1), h, d), fused)

    def test_conv_channels_reconstruct(self):
        convs = [torch.randn(128, 1, 4) for _ in range(3)]
        full = torch.cat(convs, dim=0)
        loc0 = _kda_conv_local(convs, _tp(0, 2), 8, 16)
        loc1 = _kda_conv_local(convs, _tp(1, 2), 8, 16)
        assert loc0.shape == (192, 1, 4)  # q|k|v each 64 channels
        # per-component reconstruction: each stream's rank slices concatenate
        for i, c in enumerate(convs):
            assert torch.equal(
                torch.cat([loc0[i * 64 : (i + 1) * 64], loc1[i * 64 : (i + 1) * 64]]), c
            )
        assert torch.equal(_kda_conv_local(convs, _tp(0, 1), 8, 16), full)


# --- module construction: TP=2 allocations match the loader's slice shapes ----


class TestConstructionShapes:

    def test_gated_mlp(self):
        _set_tp(0, 1)
        full = Glm5NextGatedMLP(64, 12)
        assert full.gate_proj.weight.shape == (12, 64)
        assert full.down_proj.weight.shape == (64, 12)

        _set_tp(0, 2)
        loc = Glm5NextGatedMLP(64, 12)
        assert loc.gate_proj.weight.shape == (6, 64)
        assert loc.up_proj.weight.shape == (6, 64)
        assert loc.down_proj.weight.shape == (64, 6)
        _set_tp(0, 1)

    def test_kda(self):
        cfg = _Cfg()
        _set_tp(0, 1)
        full = Glm5NextKDA(cfg, 0)
        p, h, d = 128, 8, 16
        assert full.in_proj.weight.shape == (3 * p + h + 2 * d, 64)
        assert full.conv1d.weight.shape == (3 * p, 1, 4)
        assert full.A_log.shape == (h,)
        assert full.dt_bias.shape == (p,)
        assert full.o_proj.weight.shape == (64, p)

        _set_tp(0, 2)
        loc = Glm5NextKDA(cfg, 0)
        assert loc.in_proj.weight.shape == (3 * 64 + 4 + 2 * 16, 64)
        assert loc.conv1d.weight.shape == (3 * 64, 1, 4)
        assert loc.A_log.shape == (4,)
        assert loc.dt_bias.shape == (64,)
        assert loc.f_b_proj.weight.shape == (64, 16)
        assert loc.o_proj.weight.shape == (64, 64)
        _set_tp(0, 1)

    def test_dsa_attention(self):
        cfg = _Cfg()
        _set_tp(0, 1)
        full = Glm5NextAttention(cfg, 1)
        assert full.q_b_proj.weight.shape == (8 * 16, 32)
        assert full.kv_b_proj.weight.shape == (8 * (16 + 16), 32)
        assert full.o_proj.weight.shape == (64, 8 * 16)

        _set_tp(0, 2)
        loc = Glm5NextAttention(cfg, 1)
        assert loc.q_b_proj.weight.shape == (4 * 16, 32)
        assert loc.kv_b_proj.weight.shape == (4 * (16 + 16), 32)
        assert loc.o_proj.weight.shape == (64, 4 * 16)
        assert loc.q_a_proj.weight.shape == (32, 64)  # replicated
        assert loc.indexer.wq_b.weight.shape == (4 * 8, 32)  # replicated
        _set_tp(0, 1)

# --- expert bank geometry: TP slice stays NVFP4 16-block aligned --------------


class _BankCfg:
    moe_intermediate_size = 3072  # GLM-5.3-like: divisible by 2 and by 16 per rank


class TestBankGeometry:

    def test_identity_at_tp1(self):
        from freetoken.models.nvfp4_banks import _tp_expert_geometry

        _set_tp(0, 1)
        assert _tp_expert_geometry(_BankCfg()) == (3072, 3072, 0, 3072)

    def test_rank_slice_at_tp2(self):
        from freetoken.models.nvfp4_banks import _tp_expert_geometry

        _set_tp(0, 2)
        _, I_loc, lo, hi = _tp_expert_geometry(_BankCfg())
        assert (I_loc, lo, hi) == (1536, 0, 1536)
        assert I_loc % 16 == 0  # packed bytes (/2) and scale groups (/16) split cleanly
        _set_tp(1, 2)
        _, I_loc1, lo1, hi1 = _tp_expert_geometry(_BankCfg())  # rank 1
        assert (I_loc1, lo1, hi1) == (1536, 1536, 3072)
        _set_tp(0, 1)

    def test_misaligned_slice_rejected(self):
        from freetoken.models.nvfp4_banks import _tp_expert_geometry

        class _Cfg:
            moe_intermediate_size = 24  # /2 = 12 -> not a multiple of 16

        _set_tp(0, 2)
        with pytest.raises(AssertionError, match="16-block aligned"):
            _tp_expert_geometry(_Cfg())
        _set_tp(0, 1)

    def test_loader_shapes_match_construction(self):
        """The loader's slice rules (applied to full tensors) must produce exactly
        the TP=2 module shapes -- the load_state_dict contract is strict."""
        cfg = _Cfg()
        _set_tp(0, 2)
        tp = _tp(0, 2)
        kda = Glm5NextKDA(cfg, 0)
        p, h, d = 128, 8, 16
        parts = _kda_parts()
        assert _kda_in_proj_local(parts, tp, h, d).shape == kda.in_proj.weight.shape
        convs = [torch.randn(p, 1, 4) for _ in range(3)]
        assert _kda_conv_local(convs, tp, h, d).shape == kda.conv1d.weight.shape  # (192,1,4)
        assert _col_rows(torch.randn(p, d), tp, p).shape == kda.f_b_proj.weight.shape
        assert _row_cols(torch.randn(64, p), tp, p).shape == kda.o_proj.weight.shape

        attn = Glm5NextAttention(cfg, 1)
        assert _col_rows(torch.randn(8 * 16, 32), tp, 8 * 16).shape == attn.q_b_proj.weight.shape
        assert _col_rows(torch.randn(8 * 32, 32), tp, 8 * 32).shape == attn.kv_b_proj.weight.shape
        assert _row_cols(torch.randn(64, 8 * 16), tp, 8 * 16).shape == attn.o_proj.weight.shape
        _set_tp(0, 1)
