# Copyright (c) 2026 FreeToken contributors
# qwen3_5_moe TP sharding: slice rules reconstruct the full tensors, and the
# TP>1 module construction allocates exactly the shapes the loader emits.
# CPU-only, tiny dims (see tests/models/test_qwen4_tp_sharding.py for the sibling).
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.distributed import DistributedInfo, set_tp_info
from freetoken.layers.quantization import NoQuantConfig
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen3_5_moe.weight import tp_shard_key


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info

    info._TP_INFO = None
    set_tp_info(rank, size)


def _tp(rank: int, size: int) -> DistributedInfo:
    return DistributedInfo(rank, size)


class _Group:
    def __init__(self):
        self.num_key_heads, self.num_value_heads = 8, 8
        self.key_head_dim = self.value_head_dim = 16


class _Cfg:
    def __init__(self):
        self.head_dim = 16
        self.num_qo_heads = 8
        self.num_kv_heads = 2
        self.hidden_size = 64
        self.intermediate_size = 24
        self.shared_expert_intermediate_size = 12
        self.moe_intermediate_size = 24
        self._group = _Group()

    def linear_attention_group(self):
        return self._group


# --- slice rules: rank slices concatenate back to the full tensor -------------


class TestTpShardKey:

    def _roundtrip(self, name, full, cfg):
        return [tp_shard_key(name, full, cfg, _tp(r, 2)) for r in range(2)]

    def test_qkv_segments(self):
        cfg = _Cfg()
        full = torch.randn(2 * 8 * 16 + 2 * 2 * 16, 64)  # [2q|kv|kv]
        p0, p1 = self._roundtrip("model.layers.0.self_attn.qkv_proj.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 2 * 1 * 16, 64)
        assert torch.equal(torch.cat([p0[: 8 * 16], p1[: 8 * 16]]), full[: 16 * 16])
        assert torch.equal(torch.cat([p0[8 * 16 : 9 * 16], p1[8 * 16 : 9 * 16]]), full[256:288])
        assert torch.equal(torch.cat([p0[9 * 16 :], p1[9 * 16 :]]), full[288:])
        assert torch.equal(tp_shard_key("x", full, cfg, _tp(0, 1)), full)

    def test_o_proj_cols(self):
        cfg = _Cfg()
        full = torch.randn(64, 8 * 16)
        p0, p1 = self._roundtrip("model.layers.0.self_attn.o_proj.weight", full, cfg)
        assert p0.shape == (64, 4 * 16)
        assert torch.equal(torch.cat([p0, p1], dim=1), full)

    def test_gdn_in_proj_six_segments(self):
        cfg = _Cfg()
        # qkv [q|k|v] + z + b + a
        full = torch.randn(2 * 8 * 16 + 8 * 16 + 8 * 16 + 8 + 8, 64)
        p0, p1 = self._roundtrip("model.layers.0.linear_attn.in_proj.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 4 * 16 + 4 * 16 + 4 + 4, 64)
        full_off = [0, 128, 256, 384, 512, 520]
        rank_off = [0, 64, 128, 192, 256, 260]
        widths = [128, 128, 128, 128, 8, 8]
        for i in range(6):
            half = widths[i] // 2
            got = torch.cat(
                [p0[rank_off[i] : rank_off[i] + half], p1[rank_off[i] : rank_off[i] + half]]
            )
            assert torch.equal(got, full[full_off[i] : full_off[i] + widths[i]])

    def test_gdn_in_proj_qkvz_and_ba(self):
        cfg = _Cfg()
        qkvz = torch.randn(2 * 8 * 16 + 8 * 16 + 8 * 16, 64)  # qkv 384 | z 128
        p0, p1 = self._roundtrip("model.layers.0.linear_attn.in_proj_qkvz.weight", qkvz, cfg)
        assert p0.shape == (2 * 4 * 16 + 4 * 16 + 4 * 16, 64)
        ba = torch.randn(8 + 8, 64)
        b0, b1 = self._roundtrip("model.layers.0.linear_attn.in_proj_ba.weight", ba, cfg)
        assert b0.shape == (8, 64)
        assert torch.equal(torch.cat([b0[:4], b1[:4]]), ba[:8])
        assert torch.equal(torch.cat([b0[4:], b1[4:]]), ba[8:])

    def test_gdn_conv_and_gates(self):
        cfg = _Cfg()
        full = torch.randn(2 * 8 * 16 + 8 * 16, 1, 4)
        p0, p1 = self._roundtrip("model.layers.0.linear_attn.conv1d.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 4 * 16, 1, 4)
        for i in range(3):
            assert torch.equal(
                torch.cat([p0[i * 64 : (i + 1) * 64], p1[i * 64 : (i + 1) * 64]], dim=0),
                full[i * 128 : (i + 1) * 128],
            )
        gates = torch.randn(8)
        g0, g1 = self._roundtrip("model.layers.0.linear_attn.A_log", gates, cfg)
        assert g0.shape == (4,)
        assert torch.equal(torch.cat([g0, g1]), gates)

    def test_shared_and_dense_gate_up_both_halves(self):
        cfg = _Cfg()
        shared = torch.randn(2 * 12, 64)  # gate rows [0,12), up rows [12,24)
        p0, p1 = self._roundtrip("model.layers.0.mlp.shared_expert.gate_up_proj.weight", shared, cfg)
        assert p0.shape == (12, 64)
        assert torch.equal(p0[:6], shared[:6]) and torch.equal(p0[6:], shared[12:18])
        assert torch.equal(p1[:6], shared[6:12]) and torch.equal(p1[6:], shared[18:24])
        dense = torch.randn(2 * 24, 64)
        d0, d1 = self._roundtrip("model.layers.0.mlp.gate_up_proj.weight", dense, cfg)
        assert d0.shape == (24, 64)

    def test_down_proj_cols(self):
        cfg = _Cfg()
        full = torch.randn(64, 24)
        p0, p1 = self._roundtrip("model.layers.0.mlp.down_proj.weight", full, cfg)
        assert p0.shape == (64, 12)
        assert torch.equal(torch.cat([p0, p1], dim=1), full)

    def test_stacked_bf16_experts(self):
        cfg = _Cfg()
        e = 3
        gate_up = torch.randn(e, 2 * 24, 64)
        p0, p1 = self._roundtrip("model.layers.0.mlp.experts.gate_up_proj", gate_up, cfg)
        assert p0.shape == (e, 24, 64)
        assert torch.equal(torch.cat([p0[:, :12], p1[:, :12]], dim=1), gate_up[:, :24])
        assert torch.equal(torch.cat([p0[:, 12:], p1[:, 12:]], dim=1), gate_up[:, 24:])
        down = torch.randn(e, 64, 24)
        d0, d1 = self._roundtrip("model.layers.0.mlp.experts.down_proj", down, cfg)
        assert d0.shape == (e, 64, 12)
        assert torch.equal(torch.cat([d0, d1], dim=2), down)

    def test_vocab_rows_padded(self):
        cfg = _Cfg()
        full = torch.randn(7, 64)
        r0 = tp_shard_key("lm_head.weight", full, cfg, _tp(0, 2))
        r1 = tp_shard_key("model.embed_tokens.weight", full, cfg, _tp(1, 2))
        assert r0.shape == (4, 64) and r1.shape == (4, 64)
        assert torch.equal(torch.cat([r0, r1], dim=0)[:7], full)
        assert torch.equal(r1[3:], torch.zeros(1, 64))

    def test_nvfp4_packed_row_parallel_input_axis(self):
        cfg = _Cfg()
        # o_proj input is 8*16=128 -> packed codes 64, block scales 8
        codes = torch.randint(0, 256, (64, 128 // 2), dtype=torch.uint8)
        p0, p1 = self._roundtrip("model.layers.0.self_attn.o_proj.weight", codes, cfg)
        assert p0.shape == (64, 32)
        assert torch.equal(torch.cat([p0, p1], dim=1), codes)
        scale = torch.randn(64, 128 // 16, dtype=torch.bfloat16)
        s0, s1 = self._roundtrip("model.layers.0.self_attn.o_proj.weight_scale", scale, cfg)
        assert s0.shape == (64, 4)
        assert torch.equal(torch.cat([s0, s1], dim=1), scale)

    def test_fp8_block_scale_segments(self):
        cfg = _Cfg()
        cfg.head_dim = 128
        cfg.hidden_size = 128
        # qkv full rows [2048 | 256 | 256] -> block scale [20, 1]
        scale = torch.randn(20, 1, dtype=torch.bfloat16)
        p0, p1 = self._roundtrip("model.layers.0.self_attn.qkv_proj.weight_scale_inv", scale, cfg)
        assert p0.shape == (10, 1)
        # per segment: q blocks [0:16], k [16:18], v [18:20]
        assert torch.equal(torch.cat([p0[:8], p1[:8]]), scale[:16])
        assert torch.equal(torch.cat([p0[8:9], p1[8:9]]), scale[16:18])
        assert torch.equal(torch.cat([p0[9:], p1[9:]]), scale[18:])

    def test_replicated_keys_pass_through(self):
        cfg = _Cfg()
        w = torch.randn(16, 16)
        assert torch.equal(
            tp_shard_key("model.layers.0.input_layernorm.weight", w, cfg, _tp(1, 2)), w
        )
        assert torch.equal(tp_shard_key("model.layers.0.mlp.gate.weight", w, cfg, _tp(1, 2)), w)

    def test_scalar_input_scale_passes_through(self):
        cfg = _Cfg()
        scale = torch.tensor(0.25)
        out = tp_shard_key("model.layers.0.self_attn.qkv_proj.input_scale", scale, cfg, _tp(1, 2))
        assert out is scale

    def test_row_parallel_1d_weight_global_is_replicated(self):
        # NVFP4 down_proj's per-output-row global is 1-D [hidden]; it must pass through
        # even though its length does not divide the (sliced) intermediate axis.
        cfg = _Cfg()
        full = torch.randn(64)
        out = tp_shard_key("model.layers.0.mlp.down_proj.weight_global", full, cfg, _tp(1, 2))
        assert torch.equal(out, full)


# --- module construction: TP=2 allocations match the loader's slice shapes ----


def _attn_cfg(head_dim=64, hidden=64):
    return SimpleNamespace(
        num_qo_heads=8, num_kv_heads=2, head_dim=head_dim, hidden_size=hidden,
        rms_norm_eps=1e-5, quant=NoQuantConfig(),
        rotary_config=SimpleNamespace(
            rotary_dim=head_dim, max_position=4096, base=10000.0, scaling={}
        ),
    )


def _gdn():
    return Qwen3_5GatedDeltaNet(
        hidden_size=64, num_k_heads=8, num_v_heads=8, head_k_dim=16, head_v_dim=16,
        conv_kernel_size=4, rms_norm_eps=1e-5, layer_id=0,
    )


class TestConstructionShapes:

    def test_gdn(self):
        _set_tp(0, 1)
        full = _gdn()
        assert full.in_proj.weight.shape == (2 * 8 * 16 + 8 * 16 + 8 * 16 + 8 + 8, 64)
        assert full.conv1d.weight.shape == (2 * 8 * 16 + 8 * 16, 1, 4)
        assert full.A_log.shape == (8,)
        assert full.out_proj.weight.shape == (64, 8 * 16)

        _set_tp(0, 2)
        loc = _gdn()
        assert loc.in_proj.weight.shape == (2 * 4 * 16 + 4 * 16 + 4 * 16 + 4 + 4, 64)
        assert loc.conv1d.weight.shape == (2 * 4 * 16 + 4 * 16, 1, 4)
        assert loc.A_log.shape == (4,)
        assert loc.dt_bias.shape == (4,)
        assert loc.out_proj.weight.shape == (64, 4 * 16)
        _set_tp(0, 1)

    def test_attention(self):
        _set_tp(0, 1)
        full = Qwen3_5Attention(_attn_cfg(), 0)
        assert full.qkv_proj.weight.shape == (2 * 8 * 64 + 2 * 2 * 64, 64)
        assert full.o_proj.weight.shape == (64, 8 * 64)

        _set_tp(0, 2)
        loc = Qwen3_5Attention(_attn_cfg(), 0)
        assert loc.qkv_proj.weight.shape == (2 * 4 * 64 + 2 * 1 * 64, 64)
        assert loc.o_proj.weight.shape == (64, 4 * 64)
        _set_tp(0, 1)

    def test_loader_shapes_match_construction(self):
        """The slice rules applied to full tensors must produce exactly the TP=2
        module shapes — the load_state_dict contract is strict."""
        cfg = _Cfg()
        _set_tp(0, 2)
        tp = _tp(0, 2)
        gdn = _gdn()
        assert tp_shard_key(
            "model.layers.0.linear_attn.in_proj.weight",
            torch.randn(2 * 8 * 16 + 8 * 16 + 8 * 16 + 8 + 8, 64), cfg, tp,
        ).shape == gdn.in_proj.weight.shape
        assert tp_shard_key(
            "model.layers.0.linear_attn.conv1d.weight",
            torch.randn(2 * 8 * 16 + 8 * 16, 1, 4), cfg, tp,
        ).shape == gdn.conv1d.weight.shape
        assert tp_shard_key(
            "model.layers.0.linear_attn.out_proj.weight", torch.randn(64, 8 * 16), cfg, tp,
        ).shape == gdn.out_proj.weight.shape
        attn = Qwen3_5Attention(_attn_cfg(), 0)
        cfg64 = _Cfg()
        cfg64.head_dim = 64
        assert tp_shard_key(
            "model.layers.0.self_attn.qkv_proj.weight",
            torch.randn(2 * 8 * 64 + 2 * 2 * 64, 64), cfg64, tp,
        ).shape == attn.qkv_proj.weight.shape
        assert tp_shard_key(
            "model.layers.0.self_attn.o_proj.weight", torch.randn(64, 8 * 64), cfg64, tp,
        ).shape == attn.o_proj.weight.shape
        _set_tp(0, 1)
