# Copyright (c) 2026 FreeToken contributors
# qwen4_exp TP sharding: slice rules reconstruct the full tensors, and the
# TP>1 module construction allocates exactly the shapes the loader emits.
# CPU-only, tiny dims (see tests/models/test_glm5_tp_sharding.py for the glm5
# counterpart and the load_state_dict contract rationale).
import torch

from freetoken.distributed import DistributedInfo, set_tp_info
from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
from freetoken.layers.quantization import NoQuantConfig
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
from freetoken.models.qwen4_exp.weight import tp_shard_key


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
        self.shared_expert_intermediate_size = 12
        self._group = _Group()

    def linear_attention_group(self):
        return self._group


# --- slice rules: rank slices concatenate back to the full tensor -------------


class TestTpShardKey:

    def _roundtrip(self, name, full, cfg, mk_shape=None):
        parts = [tp_shard_key(name, full, cfg, _tp(r, 2)) for r in range(2)]
        return parts

    def test_qkv_segments(self):
        cfg = _Cfg()
        full = torch.randn(2 * 8 * 16 + 2 * 2 * 16, 64)  # [2q|kv|kv]
        p0, p1 = self._roundtrip("model.layers.0.self_attn.qkv_proj.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 2 * 1 * 16, 64)
        # per-segment reconstruction: q | kv | kv
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
        # qkv [k|k|v] + z + b + a = 2*8*16 + 2*8*16 + 8 + 8 rows... conv=2*k*dk+v*dv
        full = torch.randn(2 * 8 * 16 + 8 * 16 + 8 * 16 + 8 + 8, 64)
        p0, p1 = self._roundtrip("model.layers.0.linear_attn.in_proj.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 4 * 16 + 4 * 16 + 4 + 4, 64)
        # per-segment reconstruction: q | k | v | z | b | a (64/64/64/64/4/4 rows)
        offs = [0, 64, 128, 192, 256, 260]
        for i in range(6):
            a0, a1 = offs[i] + (0 if i < 4 else 0), offs[i] + (64 if i < 4 else 4)
            lo_f = i * 128 if i < 4 else 512 + (0 if i == 4 else 8)
            hi_f = lo_f + (128 if i < 4 else 8)
            assert torch.equal(torch.cat([p0[a0:a1], p1[a0:a1]], dim=0), full[lo_f:hi_f])

    def test_gdn_conv_channels(self):
        cfg = _Cfg()
        full = torch.randn(2 * 8 * 16 + 8 * 16, 1, 4)
        p0, p1 = self._roundtrip("model.layers.0.linear_attn.conv1d.weight", full, cfg)
        assert p0.shape == (2 * 4 * 16 + 4 * 16, 1, 4)
        # per-segment reconstruction: q | k | v (64 rows per rank each)
        for i in range(3):
            assert torch.equal(
                torch.cat([p0[i * 64 : (i + 1) * 64], p1[i * 64 : (i + 1) * 64]], dim=0),
                full[i * 128 : (i + 1) * 128],
            )

    def test_shared_expert_gate_up(self):
        cfg = _Cfg()
        full = torch.randn(2 * 12, 64)  # gate rows [0,12), up rows [12,24)
        p0, p1 = self._roundtrip("model.layers.0.mlp.shared_expert.gate_up_proj.weight", full, cfg)
        assert p0.shape == (12, 64)
        # rank 0: gate[0:6]+up[12:18]; rank 1: gate[6:12]+up[18:24]
        assert torch.equal(p0[:6], full[:6]) and torch.equal(p0[6:], full[12:18])
        assert torch.equal(p1[:6], full[6:12]) and torch.equal(p1[6:], full[18:24])

    def test_vocab_rows_padded(self):
        cfg = _Cfg()
        full = torch.randn(7, 64)
        r0 = tp_shard_key("lm_head.weight", full, cfg, _tp(0, 2))
        r1 = tp_shard_key("lm_head.weight", full, cfg, _tp(1, 2))
        assert r0.shape == (4, 64) and r1.shape == (4, 64)
        assert torch.equal(torch.cat([r0, r1], dim=0)[:7], full)
        assert torch.equal(r1[3:], torch.zeros(1, 64))

    def test_replicated_keys_pass_through(self):
        cfg = _Cfg()
        w = torch.randn(16, 16)
        assert torch.equal(tp_shard_key("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", w, cfg, _tp(1, 2)), w)
        assert torch.equal(tp_shard_key("model.layers.0.ple.norm_key.weight", w, cfg, _tp(1, 2)), w)


# --- module construction: TP=2 allocations match the loader's slice shapes ----


class TestConstructionShapes:

    def test_gdn(self):
        _set_tp(0, 1)
        full = Qwen4ExpGatedDeltaNet(
            hidden_size=64, num_k_heads=8, num_v_heads=8, head_k_dim=16, head_v_dim=16,
            conv_kernel_size=4, rms_norm_eps=1e-5, layer_id=0,
        )
        assert full.in_proj.weight.shape == (2 * 8 * 16 + 8 * 16 + 8 * 16 + 8 + 8, 64)
        assert full.conv1d.weight.shape == (2 * 8 * 16 + 8 * 16, 1, 4)
        assert full.A_log.shape == (8,)
        assert full.out_proj.weight.shape == (64, 8 * 16)

        _set_tp(0, 2)
        loc = Qwen4ExpGatedDeltaNet(
            hidden_size=64, num_k_heads=8, num_v_heads=8, head_k_dim=16, head_v_dim=16,
            conv_kernel_size=4, rms_norm_eps=1e-5, layer_id=0,
        )
        assert loc.in_proj.weight.shape == (2 * 4 * 16 + 4 * 16 + 4 * 16 + 4 + 4, 64)
        assert loc.conv1d.weight.shape == (2 * 4 * 16 + 4 * 16, 1, 4)
        assert loc.A_log.shape == (4,)
        assert loc.dt_bias.shape == (4,)
        assert loc.out_proj.weight.shape == (64, 4 * 16)
        _set_tp(0, 1)

    def test_qsa_attention(self):
        from types import SimpleNamespace

        def cfg():
            return SimpleNamespace(
                num_qo_heads=8, num_kv_heads=2, head_dim=64, hidden_size=64, rms_norm_eps=1e-5,
                quant=NoQuantConfig(),
                rotary_config=SimpleNamespace(
                    rotary_dim=64, max_position=4096, base=10000.0, scaling={}
                ),
                qwen4_args=SimpleNamespace(
                    hidden_size=64, index_n_heads=4, index_kv_heads=1, index_head_dim=8
                ),
            )

        _set_tp(0, 1)
        full = Qwen4ExpAttention(cfg(), 0)
        assert full.qkv_proj.weight.shape == (2 * 8 * 64 + 2 * 2 * 64, 64)
        assert full.o_proj.weight.shape == (64, 8 * 64)

        _set_tp(0, 2)
        loc = Qwen4ExpAttention(cfg(), 0)
        assert loc.qkv_proj.weight.shape == (2 * 4 * 64 + 2 * 1 * 64, 64)
        assert loc.o_proj.weight.shape == (64, 4 * 64)
        assert loc.indexer.index_qk_proj.weight.shape == (4 * 8 + 1 * 8, 64)  # replicated
        _set_tp(0, 1)

    def test_loader_shapes_match_construction(self):
        """The slice rules applied to full tensors must produce exactly the TP=2
        module shapes — the load_state_dict contract is strict."""
        cfg = _Cfg()
        _set_tp(0, 2)
        tp = _tp(0, 2)
        gdn = Qwen4ExpGatedDeltaNet(
            hidden_size=64, num_k_heads=8, num_v_heads=8, head_k_dim=16, head_v_dim=16,
            conv_kernel_size=4, rms_norm_eps=1e-5, layer_id=0,
        )
        assert tp_shard_key("model.layers.0.linear_attn.in_proj.weight", torch.randn(2 * 8 * 64 + 8 * 64 + 8 * 64 + 8 + 8, 64), cfg, tp).shape == gdn.in_proj.weight.shape
        assert tp_shard_key("model.layers.0.linear_attn.conv1d.weight", torch.randn(2 * 8 * 64 + 8 * 64, 1, 4), cfg, tp).shape == gdn.conv1d.weight.shape
        assert tp_shard_key("model.layers.0.linear_attn.out_proj.weight", torch.randn(64, 8 * 64), cfg, tp).shape == gdn.out_proj.weight.shape

        from types import SimpleNamespace

        attn = Qwen4ExpAttention(
            SimpleNamespace(
                num_qo_heads=8, num_kv_heads=2, head_dim=64, hidden_size=64, rms_norm_eps=1e-5,
                quant=NoQuantConfig(),
                rotary_config=SimpleNamespace(rotary_dim=64, max_position=4096, base=10000.0, scaling={}),
                qwen4_args=SimpleNamespace(hidden_size=64, index_n_heads=4, index_kv_heads=1, index_head_dim=8),
            ), 0)
        cfg64 = _Cfg()
        cfg64.head_dim = 64
        assert tp_shard_key("model.layers.0.self_attn.qkv_proj.weight", torch.randn(2 * 8 * 64 + 2 * 2 * 64, 64), cfg64, tp).shape == attn.qkv_proj.weight.shape
        assert tp_shard_key("model.layers.0.self_attn.o_proj.weight", torch.randn(64, 8 * 64), cfg64, tp).shape == attn.o_proj.weight.shape
        _set_tp(0, 1)
