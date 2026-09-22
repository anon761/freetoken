"""deepseek_v41 Phase-2 weight loading: fp8 32x32 linear kernel, expert-piece
TP slicing, iter_weights name census on a synthetic mini-checkpoint."""

from __future__ import annotations

import json
import os

import pytest
import torch

from freetoken.layers.quantization import QuantKind
from freetoken.models.deepseek_v41.weight import _tp_slice_piece, iter_weights

# ---------------------------------------------------------------- slice helper


def test_tp_slice_piece_bands():
    I = 2304
    lo, hi = I // 2, I  # rank 1 of 2
    gate = torch.arange(I * 8, dtype=torch.uint8).reshape(I, 8)
    gate_scale = torch.arange(I * 2, dtype=torch.uint8).reshape(I, 2)
    down = torch.arange(8 * (I // 2), dtype=torch.uint8).reshape(8, I // 2)
    down_scale = torch.arange(8 * (I // 32), dtype=torch.uint8).reshape(8, I // 32)

    g = _tp_slice_piece("gate", gate, lo, hi)
    assert g.shape == (I // 2, 8) and torch.equal(g[0], gate[lo])
    gs = _tp_slice_piece("gate_scale", gate_scale, lo, hi)
    assert gs.shape == (I // 2, 2) and torch.equal(gs[0], gate_scale[lo])
    d = _tp_slice_piece("down", down, lo, hi)
    # packed codes slice on the byte axis: I-value band [lo, hi) -> bytes [lo//2, hi//2)
    assert d.shape == (8, (hi - lo) // 2)
    assert torch.equal(d[:, 0], down[:, lo // 2])
    ds = _tp_slice_piece("down_scale", down_scale, lo, hi)
    # per-32 scales: I band -> scale cols [lo//32, hi//32)
    assert ds.shape == (8, (hi - lo) // 32)
    assert torch.equal(ds[:, 0], down_scale[:, lo // 32])


# ---------------------------------------------------------------- fp8 32x32 linear

_FP8 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu


def _fake_fp8_linear(N: int, K: int, block: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.1).to(_FP8)
    s = torch.randint(120, 135, (N // block, K // block), generator=g).to(_E8M0)
    return w, s


def _dequant_ref(w: torch.Tensor, s: torch.Tensor, block: int) -> torch.Tensor:
    n, k = w.shape
    sc = s.view(torch.uint8).to(torch.float32)
    sc = torch.exp2(sc - 127.0).repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return (w.to(torch.float32) * sc).to(torch.bfloat16)


@pytest.mark.parametrize("block", [128, 32])
def test_block_fp8_linear_matches_dequant_reference(block: int):
    if not torch.cuda.is_available():
        pytest.skip("triton kernel")
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip, block_fp8_linear

    N, K = 256, 512
    w, s = _fake_fp8_linear(N, K, block)
    ref_w = _dequant_ref(w, s, block).cuda()
    # the kernel FP8-round-trips the activation (per-128 ue8m0) — the reference
    # must apply the same quantization to the input
    x = act_quant_fp8_roundtrip(torch.randn(7, K, dtype=torch.bfloat16, device="cuda"), 128)
    out = block_fp8_linear(x, w.cuda(), s.cuda(), None, weight_block=block)
    torch.testing.assert_close(out, x @ ref_w.T, rtol=2e-2, atol=2e-2)

    x1 = act_quant_fp8_roundtrip(torch.randn(1, K, dtype=torch.bfloat16, device="cuda"), 128)  # GEMV path
    out1 = block_fp8_linear(x1, w.cuda(), s.cuda(), None, weight_block=block)
    torch.testing.assert_close(out1, x1 @ ref_w.T, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------- iter_weights census

# tiny but 32-block-aligned geometry: H=64, q_lora=32, heads=4, head_dim=32,
# groups=4 (wo_a_k=32), o_lora=32, I=32, E=4, vocab=64
_VOCAB, _H, _Q_LORA, _HEADS, _HEAD_DIM, _GROUPS, _O_LORA, _I, _E = 64, 64, 32, 4, 32, 4, 32, 32, 4


def _mini_layer_tensors(L: int) -> dict[str, list[int]]:
    """One MoE layer's checkpoint tensors with the real V4.1 naming (tiny shapes)."""
    a = f"layers.{L}.attn"
    t: dict[str, list[int]] = {
        f"{a}.wq_a.weight": [_Q_LORA, _H],
        f"{a}.wq_a.scale": [_Q_LORA // 32, _H // 32],
        f"{a}.q_norm.weight": [_Q_LORA],
        f"{a}.wq_b.weight": [_HEADS * _HEAD_DIM, _Q_LORA],
        f"{a}.wq_b.scale": [_HEADS * _HEAD_DIM // 32, _Q_LORA // 32],
        f"{a}.wkv.weight": [_HEAD_DIM, _H],
        f"{a}.wkv.scale": [_HEAD_DIM // 32, _H // 32],
        f"{a}.kv_norm.weight": [_HEAD_DIM],
        f"{a}.wo_a.weight": [_GROUPS * _O_LORA, _HEADS * _HEAD_DIM // _GROUPS],
        f"{a}.wo_a.scale": [_GROUPS * _O_LORA // 32, _HEADS * _HEAD_DIM // _GROUPS // 32],
        f"{a}.wo_b.weight": [_H, _GROUPS * _O_LORA],
        f"{a}.wo_b.scale": [_H // 32, _GROUPS * _O_LORA // 32],
        f"{a}.attn_sink": [_HEADS],
        f"layers.{L}.attn_norm.weight": [_H],
        f"layers.{L}.ffn_norm.weight": [_H],
        f"layers.{L}.ffn.gate.weight": [_E, _H],
        f"layers.{L}.ffn.gate.bias": [_E],
        f"layers.{L}.ffn.gate.bias_vl": [_E],
    }
    for proj in ("w1", "w3"):
        t[f"layers.{L}.ffn.shared_experts.{proj}.weight"] = [_I, _H]
        t[f"layers.{L}.ffn.shared_experts.{proj}.scale"] = [_I // 32, _H // 32]
    t[f"layers.{L}.ffn.shared_experts.w2.weight"] = [_H, _I]
    t[f"layers.{L}.ffn.shared_experts.w2.scale"] = [_H // 32, _I // 32]
    for nm in ("hc_attn_fn", "hc_ffn_fn"):
        t[f"layers.{L}.{nm}"] = [24, 4 * _H]
    for nm in ("hc_attn_base", "hc_ffn_base"):
        t[f"layers.{L}.{nm}"] = [24]
    for nm in ("hc_attn_scale", "hc_ffn_scale"):
        t[f"layers.{L}.{nm}"] = [3]
    return t


def _dtype_for(name: str) -> torch.dtype:
    if name.endswith(".scale"):
        return _E8M0
    if name.endswith(".wo_a.weight"):
        return _FP8
    return torch.bfloat16


def _write_mini_checkpoint(tmp_path: str) -> str:
    """Synthetic checkpoint with the real V4.1 naming: 2 layers + skipped groups."""
    tensors: dict[str, list[int]] = {
        "embed.weight": [_VOCAB, _H],
        "norm.weight": [_H],
        "head.weight": [_VOCAB, _H],
    }
    for L in range(2):
        tensors.update(_mini_layer_tensors(L))
        tensors[f"layers.{L}.attn.compressor.norm.weight"] = [8]  # phase-3 skip
        tensors[f"layers.{L}.engram.embed.weight"] = [8, 8]  # phase-4 skip
    tensors["mtp.0.attn.wq_a.weight"] = [_Q_LORA, _H]  # phase-6 skip
    tensors["vision.blocks.0.attn.wo.weight"] = [8, 8]  # out of scope
    tensors["aligner.w1.weight"] = [8, 8]  # out of scope

    tensors_t = {name: torch.zeros(shape, dtype=_dtype_for(name)) for name, shape in tensors.items()}
    from safetensors.torch import save_file

    save_file(tensors_t, os.path.join(tmp_path, "model-00001.safetensors"))
    index = {"weight_map": {k: "model-00001.safetensors" for k in tensors}}
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    # the reader needs config.json for load_args (text_config)
    cfg = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "text_config": {
            "num_hidden_layers": 2, "hidden_size": _H, "vocab_size": _VOCAB,
            "n_routed_experts": _E, "moe_intermediate_size": _I,
            "max_position_embeddings": 1024,
        },
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp_path


def test_iter_weights_census_and_skips(tmp_path):
    path = _write_mini_checkpoint(str(tmp_path))
    names = [name for name, _t in iter_weights(path, torch.device("cpu"), include_moe_experts=False)]

    assert "model.embed.weight" in names and "model.head.weight" in names and "model.norm.weight" in names
    for L in range(2):
        for key in (
            f"model.layers.{L}.attn.wq_a.weight", f"model.layers.{L}.attn.wq_a.weight_scale_inv",
            f"model.layers.{L}.attn.q_norm.weight", f"model.layers.{L}.attn.wq_b.weight",
            f"model.layers.{L}.attn.wkv.weight", f"model.layers.{L}.attn.kv_norm.weight",
            f"model.layers.{L}.attn.wo_a", f"model.layers.{L}.attn.wo_b.weight",
            f"model.layers.{L}.attn.attn_sink", f"model.layers.{L}.attn_norm.weight",
            f"model.layers.{L}.ffn.gate.weight", f"model.layers.{L}.ffn.gate.bias",
            f"model.layers.{L}.ffn.gate.bias_vl",
            f"model.layers.{L}.ffn.shared_experts.w1.weight_scale_inv",
            f"model.layers.{L}.hc_attn_fn", f"model.layers.{L}.hc_ffn_scale",
        ):
            assert key in names, key
    # skipped groups never surface as model keys, no duplicate yields
    assert not any("compressor" in n or "engram" in n or n.startswith("model.mtp.") for n in names)
    assert len(names) == len(set(names))


def test_iter_weights_wo_a_dequantizes(tmp_path):
    path = _write_mini_checkpoint(str(tmp_path))
    out = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=False))
    assert out["model.layers.0.attn.wo_a"].dtype == torch.bfloat16


def test_iter_weights_rejects_expert_include(tmp_path):
    path = _write_mini_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="offload"):
        list(iter_weights(path, torch.device("cpu"), include_moe_experts=True))


def test_iter_expert_pieces_kind_gate(tmp_path):
    from freetoken.models.deepseek_v41.weight import iter_expert_pieces

    path = _write_mini_checkpoint(str(tmp_path))
    assert iter_expert_pieces(path, None, QuantKind.NVFP4) is None


def test_tp_shard_key_tp2_shapes():
    """All resident keys yield the rank-local shapes the strict loader asserts."""
    from types import SimpleNamespace

    from freetoken.models.deepseek_v41.weight import tp_shard_key

    tp = SimpleNamespace(size=2, rank=1)
    H, QL, HD, HEADS, G, OL, I, E = 64, 32, 32, 4, 4, 32, 64, 4
    args = SimpleNamespace(o_groups=G, o_lora_rank=OL)
    config = SimpleNamespace(
        dsv41_args=args, num_qo_heads=HEADS, head_dim=HD,
        shared_expert_intermediate_size=I,
    )
    full = torch.zeros

    def key(n, shape):
        return n, tuple(tp_shard_key(n, full(shape, dtype=torch.uint8), config, tp).shape)

    assert key("model.embed.weight", [2 * E * 16, H])[1] == (E * 16, H)
    assert key("model.head.weight", [2 * E * 16, H])[1] == (E * 16, H)
    # DSpark markov head vocab tables shard like the target's embed/head
    assert key("model.mtp.2.markov_head.embed.weight", [2 * E * 16, 8])[1] == (E * 16, 8)
    assert key("model.mtp.2.markov_head.head.weight", [2 * E * 16, 8])[1] == (E * 16, 8)
    assert key("model.layers.0.attn.wq_b.weight", [HEADS * HD, QL])[1] == (HEADS * HD // 2, QL)
    assert key("model.layers.0.attn.wq_b.weight_scale_inv", [HEADS * HD // 32, QL // 32])[1] == (HEADS * HD // 64, QL // 32)
    assert key("model.layers.0.attn.wo_a", [G * OL, HEADS * HD // G])[1] == (G // 2 * OL, HEADS * HD // G)
    assert key("model.layers.0.attn.wo_b.weight", [H, G * OL])[1] == (H, G // 2 * OL)
    assert key("model.layers.0.attn.wo_b.weight_scale_inv", [H // 32, G * OL // 32])[1] == (H // 32, G // 2 * OL // 32)
    assert key("model.layers.0.ffn.shared_experts.w1.weight", [I, H])[1] == (I // 2, H)
    assert key("model.layers.0.ffn.shared_experts.w1.weight_scale_inv", [I // 32, H // 32])[1] == (I // 64, H // 32)
    assert key("model.layers.0.ffn.shared_experts.w2.weight", [H, I])[1] == (H, I // 2)
    assert key("model.layers.0.ffn.shared_experts.w2.weight_scale_inv", [H // 32, I // 32])[1] == (H // 32, I // 64)
    # replicated
    assert key("model.layers.0.attn.wq_a.weight", [QL, H])[1] == (QL, H)
    assert key("model.layers.0.attn.wkv.weight", [HD, H])[1] == (HD, H)
    assert key("model.layers.0.ffn.gate.weight", [E, H])[1] == (E, H)
    assert key("model.layers.0.hc_attn_fn", [24, 4 * H])[1] == (24, 4 * H)
