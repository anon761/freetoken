"""DSpark draft heads vs a plain-torch reference (CPU, no GPU).

Stage 1 pins the two self-contained heads: the Markov token->rank embed + rank->vocab
head, and the fp32 confidence projection. Stage 2b adds the draft's own expert banks:
the per-stage MXFP4 piece reader, the dedicated FTW bank namespace, and the draft-owned
offload cache (E=128 vs the target's 384). All CPU.
"""
from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace

import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.deepseek_v41.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)


def _tp1() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_markov_head_projection_matches_reference():
    _tp1()
    vocab, rank = 16, 4
    mh = DSparkMarkovHead(vocab, rank)
    g = torch.Generator().manual_seed(0)
    mh.embed.weight.copy_(torch.randn(vocab, rank, generator=g))
    mh.head.weight.copy_(torch.randn(vocab, rank, generator=g))
    assert mh.embed.weight.shape == (vocab, rank)
    assert mh.head.weight.shape == (vocab, rank)

    ids = torch.tensor([1, 5, 9])
    embed = mh.embed.weight[ids]  # the token->rank lookup (shared indexing kernel)
    logits = mh.head.forward_all(embed)
    logits_ref = embed.float() @ mh.head.weight.float().t()
    assert torch.allclose(logits.float(), logits_ref, atol=1e-5)


def test_confidence_head_matches_reference():
    _tp1()
    dim, rank = 6, 3
    ch = DSparkConfidenceHead(dim + rank)
    g = torch.Generator().manual_seed(1)
    ch.proj.weight.copy_(torch.randn(1, dim + rank, generator=g))
    assert ch.proj.bias is None  # the checkpoint ships a bias-less proj

    hidden = torch.randn(2, dim, generator=g)
    markov = torch.randn(2, rank, generator=g)
    got = ch.forward(hidden, markov)

    x = torch.cat([hidden, markov], dim=-1).float()
    ref = torch.sigmoid(x @ ch.proj.weight.float().t())
    assert torch.allclose(got, ref.squeeze(-1), atol=1e-5)


# ---------------------------------------------------------------- draft expert banks
# Tiny but format-true geometry: the draft's MXFP4 experts pack to the same ds_fp4
# kernel layout as the target's (I=32 gates 32 rows of H/2=32 packed bytes, 32/32=1
# scale column); the draft E is deliberately != the fabricated target's (4 vs 4 is
# irrelevant here -- what matters is that the draft owns its own E and layer count).
_H, _I, _E, _STAGES = 64, 32, 4, 2
_MINI_LAYERS = 2
# v4.1 uses the target ids as-is: the draft fuses the attention INPUTS of these layers.
_TARGETS = [1, 2]


def _mini_text() -> dict:
    return {
        "num_hidden_layers": _MINI_LAYERS, "hidden_size": _H, "vocab_size": 64,
        "num_attention_heads": 4, "head_dim": 32, "qk_rope_head_dim": 8,
        "q_lora_rank": 32, "o_lora_rank": 32, "o_groups": 4,
        "moe_intermediate_size": _I, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "num_nextn_predict_layers": _STAGES, "dspark_block_size": 2,
        "dspark_target_layer_ids": _TARGETS, "dspark_n_routed_experts": _E,
        "dspark_num_experts_per_tok": 2, "dspark_markov_rank": 2,
        "sliding_window": 8, "max_position_embeddings": 1024,
        "compress_ratios": [0] * _MINI_LAYERS,
    }


def _write_config(d: str) -> None:
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
                   "text_config": _mini_text()}, f)


def _write_dspark_checkpoint(root) -> str:
    """A mini deepseek_v41 checkpoint shipping the DSpark draft's routed experts
    (``mtp.{stage}.ffn.experts.{e}.{w1,w3,w2}.{weight,scale}``) with the real naming."""
    d = str(root)
    os.makedirs(d, exist_ok=True)
    _write_config(d)

    tensors: dict[str, torch.Tensor] = {}
    for stage in range(_STAGES):
        for e in range(_E):
            b = f"mtp.{stage}.ffn.experts.{e}"
            for proj in ("w1", "w3"):
                tensors[f"{b}.{proj}.weight"] = torch.zeros(_I, _H // 2, dtype=torch.uint8)
                tensors[f"{b}.{proj}.scale"] = torch.full((_I, _H // 32), 120, dtype=torch.uint8)
            tensors[f"{b}.w2.weight"] = torch.zeros(_H, _I // 2, dtype=torch.uint8)
            tensors[f"{b}.w2.scale"] = torch.full((_H, _I // 32), 120, dtype=torch.uint8)
    from safetensors.torch import save_file

    save_file(tensors, os.path.join(d, "model-00001.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_map": {k: "model-00001.safetensors" for k in tensors}}, f)
    return d


def _write_full_mtp_checkpoint(root) -> str:
    """A complete bf16 dense mini checkpoint (target + ``mtp.*`` draft), so
    ``iter_weights`` names can be compared 1:1 against the built model's state_dict.

    Dense linears are bf16 (no ``.scale`` sidecars) so they match a quant=None model;
    ``wo_a`` keeps its fp8 32x32 blocks because ``iter_weights`` always dequantizes it.
    The experts (offloaded) need no tensors."""
    from tests.models.test_deepseek_v41_weight import _dtype_for, _mini_layer_tensors

    d = str(root)
    os.makedirs(d, exist_ok=True)
    _write_config(d)

    def dense(prefix: str, layer: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, shape in _mini_layer_tensors(layer).items():
            if "compressor" in name or "engram" in name:
                continue  # not built at compress_ratio 0 / no engram layers
            if name.endswith(".scale") and ".wo_a." not in name:
                continue  # dense is bf16 in this fixture
            out[name.replace(f"layers.{layer}.", prefix)] = torch.zeros(shape, dtype=_dtype_for(name))
        return out

    tensors: dict[str, torch.Tensor] = {}
    for L in range(_MINI_LAYERS):
        tensors.update(dense(f"layers.{L}.", L))
        tensors.update(dense(f"mtp.{L}.", L))
    last = _STAGES - 1
    tensors["embed.weight"] = torch.zeros(64, _H, dtype=torch.bfloat16)
    tensors["head.weight"] = torch.zeros(64, _H, dtype=torch.bfloat16)
    tensors["norm.weight"] = torch.zeros(_H, dtype=torch.bfloat16)
    tensors["mtp.0.main_proj.weight"] = torch.zeros(_H, _H * len(_TARGETS), dtype=torch.bfloat16)
    tensors["mtp.0.main_norm.weight"] = torch.zeros(_H, dtype=torch.bfloat16)
    tensors[f"mtp.{last}.norm.weight"] = torch.zeros(_H, dtype=torch.bfloat16)
    tensors[f"mtp.{last}.markov_head.embed.weight"] = torch.zeros(64, 2, dtype=torch.bfloat16)
    tensors[f"mtp.{last}.markov_head.head.weight"] = torch.zeros(64, 2, dtype=torch.bfloat16)
    tensors[f"mtp.{last}.confidence_head.proj.weight"] = torch.zeros(1, _H + 2, dtype=torch.bfloat16)
    from safetensors.torch import save_file

    save_file(tensors, os.path.join(d, "model-00001.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_map": {k: "model-00001.safetensors" for k in tensors}}, f)
    return d


def _mini_model_config(model_dir: str):
    """parse_config + the MXFP4 quant the engine attaches, so the draft method resolves."""
    from freetoken.layers.quantization import QuantConfig
    from freetoken.models.deepseek_v41.config import parse_config
    from freetoken.utils import cached_load_hf_config

    mc = parse_config(cached_load_hf_config(model_dir))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": []}})
    return dataclasses.replace(mc, quant=quant, moe_strategy="offload", decode_target="gpu")


def _draft_banks(model_dir: str, mc):
    from freetoken.models.deepseek_v41.weight import dspark_expert_method, load_dspark_banks

    method = dspark_expert_method(mc)
    banks = load_dspark_banks(
        model_dir, mc, method=method, num_stages=_STAGES, device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    return method, banks


def test_iter_dspark_expert_pieces_groups_per_stage(tmp_path):
    _tp1()
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.deepseek_v41.weight import iter_dspark_expert_pieces

    path = _write_dspark_checkpoint(tmp_path)
    config = SimpleNamespace(moe_intermediate_size=_I)  # _tp_expert_geometry reads only this
    pieces = list(iter_dspark_expert_pieces(path, config, QuantKind.MXFP4, num_stages=_STAGES))

    assert len(pieces) == _STAGES * _E
    by_stage: dict[int, list[int]] = {}
    for stage, e0, e1, roles in pieces:
        by_stage.setdefault(stage, []).append(e0)
        assert e1 == e0 + 1
        assert set(roles) == {"gate", "gate_scale", "up", "up_scale", "down", "down_scale"}
        assert roles["gate"].shape == (1, _I, _H // 2)
        assert roles["gate_scale"].shape == (1, _I, _H // 32)
        assert roles["down"].shape == (1, _H, _I // 2)
        assert roles["down_scale"].shape == (1, _H, _I // 32)
    assert {s: sorted(v) for s, v in by_stage.items()} == {s: list(range(_E)) for s in range(_STAGES)}
    # the reader is MXFP4-only (the draft has no other dialect)
    assert iter_dspark_expert_pieces(path, config, QuantKind.NVFP4, num_stages=_STAGES) is None


def test_load_dspark_banks_raw_packs_kernel_layout(tmp_path):
    _tp1()
    path = _write_dspark_checkpoint(tmp_path)
    mc = _mini_model_config(path)
    method, banks = _draft_banks(path, mc)

    assert method is not None and method.cfg.num_experts == _E
    assert method.cfg.strategy == "offload"
    assert banks.quant_format == "ds_fp4" and banks.kernel == "triton"
    assert set(banks.sources) == {"gate_up", "gate_up_scale", "down", "down_scale"}
    assert all(len(per) == _STAGES for per in banks.sources.values())
    assert all(t.shape == (_E, 2 * _I, _H // 2) for t in banks.sources["gate_up"])
    assert all(t.shape == (_E, _H, _I // 2) for t in banks.sources["down"])


def test_dspark_banks_ftw_namespace_is_separate(tmp_path):
    _tp1()
    from freetoken.checkpoint.ftw import (
        DSPARK_BANK_KIND, DSPARK_BANK_NUM_LAYERS,
        FTWWriter, layer_bank_entry_name, load_ftw_banks,
    )

    path = _write_dspark_checkpoint(tmp_path / "src")
    mc = _mini_model_config(path)
    method, banks = _draft_banks(path, mc)

    out = str(tmp_path / "ftw")
    w = FTWWriter(out)
    for role, per_layer in banks.sources.items():
        for layer_id, tensor in enumerate(per_layer):
            w.add_tensor(layer_bank_entry_name(role, layer_id), tensor, kind=DSPARK_BANK_KIND)
    # a target-kind entry (same bank base name) must stay invisible to the draft reader
    w.add_tensor(layer_bank_entry_name("gate_up", 0), torch.zeros(_E, 2 * _I, _H // 2, dtype=torch.uint8),
                 kind="experts_bank")
    w.finalize({"quant_format": "ds_fp4", "expert_bank_num_layers": 1, DSPARK_BANK_NUM_LAYERS: _STAGES})

    from freetoken.models.deepseek_v41.weight import load_dspark_banks

    back = load_dspark_banks(out, mc, method=method, num_stages=_STAGES, device=torch.device("cpu"),
                             dtype=torch.bfloat16)
    assert set(back.sources) == set(banks.sources)
    assert all(len(back.sources[role]) == _STAGES for role in banks.sources)
    assert torch.equal(back.sources["gate_up"][0], banks.sources["gate_up"][0])

    target = load_ftw_banks(out, num_layers=1)
    assert target is not None and list(target.sources) == ["gate_up"]
    assert len(target.sources["gate_up"]) == 1


def test_dspark_draft_binds_dedicated_offload_cache(tmp_path):
    _tp1()
    from freetoken.models.deepseek_v41.dspark import DSparkDraft
    from freetoken.models.deepseek_v41.weight import dspark_expert_method, load_dspark_banks

    path = _write_dspark_checkpoint(tmp_path)
    mc = _mini_model_config(path)
    method = dspark_expert_method(mc)
    banks = load_dspark_banks(path, mc, method=method, num_stages=_STAGES,
                              device=torch.device("cpu"), dtype=torch.bfloat16, dummy=True)

    draft = DSparkDraft(mc, mc.dsv4_args, strategy="offload", decode_target="gpu",
                        quant_config=mc.quant)
    for stage, layer in enumerate(draft.layers.op_list):
        assert layer.ffn.experts.layer_id == stage  # stage-keyed, ready for a 3-layer cache
        assert layer.ffn.experts.num_experts == _E

    cache = draft.build_offload_cache(banks, cache_size=2 * _E, device=torch.device("cpu"))
    assert cache.num_layers == _STAGES and cache.num_experts == _E
    assert cache.quant_format == banks.quant_format
    for layer in draft.layers.op_list:
        assert layer.ffn.experts.offload_cache is cache


def test_dspark_expert_method_none_without_draft(tmp_path):
    _tp1()
    from freetoken.models.deepseek_v41.weight import dspark_expert_method

    mc = _mini_model_config(_write_dspark_checkpoint(tmp_path))
    no_draft = dataclasses.replace(
        mc, dsv4_args=dataclasses.replace(mc.dsv4_args, num_nextn_predict_layers=0)
    )
    assert dspark_expert_method(no_draft) is None


def test_dspark_bank_hooks_dispatch_via_models_weight(tmp_path):
    _tp1()
    from freetoken.models.weight import dspark_expert_method, load_dspark_banks

    path = _write_dspark_checkpoint(tmp_path)
    mc = _mini_model_config(path)
    method = dspark_expert_method(path, mc)
    assert method is not None and method.cfg.num_experts == _E
    banks = load_dspark_banks(path, mc, method=method, num_stages=_STAGES,
                              device=torch.device("cpu"), dtype=torch.bfloat16)
    assert banks.quant_format == "ds_fp4"


# ---------------------------------------------------------------- model wiring (2c)
def _build_model(mc):
    from freetoken.models import create_model

    with torch.device("meta"):
        return create_model(mc)


def test_model_builds_dspark_and_keys_match_iter_weights(tmp_path, monkeypatch):
    """The draft is mounted so its dense keys are exactly ``model.mtp.{i}`` -- the
    checkpoint's ``mtp.{i}`` naming, NOT ``.mtp.layers.{i}``. Compares the built model's
    state_dict 1:1 against the FTW/raw reader's names."""
    _tp1()
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    from freetoken.models.deepseek_v41.weight import iter_weights

    path = _write_full_mtp_checkpoint(tmp_path)
    mc = _mini_model_config(path)
    assert mc.mtp_enabled is True
    model = _build_model(mc)

    model_keys = set(model.state_dict())
    iter_keys = {n for n, _ in iter_weights(path, torch.device("cpu"), include_moe_experts=False)}
    assert iter_keys == model_keys

    mtp = {k for k in model_keys if k.startswith("model.mtp.")}
    assert mtp and not any(".mtp.layers." in k for k in mtp)
    assert "model.mtp.0.main_proj.weight" in mtp
    assert f"model.mtp.{_STAGES - 1}.markov_head.embed.weight" in mtp
    assert f"model.mtp.{_STAGES - 1}.confidence_head.proj.weight" in mtp


def test_model_has_no_dspark_without_mtp(tmp_path, monkeypatch):
    _tp1()
    monkeypatch.delenv("FREETOKEN_ENABLE_MTP", raising=False)
    path = _write_full_mtp_checkpoint(tmp_path)
    mc = _mini_model_config(path)
    assert mc.mtp_enabled is False
    model = _build_model(mc)
    assert model.model.mtp is None
    assert not any(k.startswith("model.mtp.") for k in model.state_dict())


def test_dspark_layers_are_excluded_from_the_target_offload_walk(tmp_path, monkeypatch):
    """The target's method pick / cache attach must never see the draft (its own E)."""
    _tp1()
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    from freetoken.engine.engine import shared_offload_method
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    mc = _mini_model_config(_write_full_mtp_checkpoint(tmp_path))
    model = _build_model(mc)

    target_layers = list(iter_offload_moe_layers(model))
    assert len(target_layers) == _MINI_LAYERS

    draft = model.dspark_draft()
    assert draft is not None and draft.offload_excluded is True
    draft_method = draft.layers.op_list[0].ffn.experts.quant_method
    target_method = shared_offload_method(model)
    assert target_method is model.model.layers.op_list[0].ffn.experts.quant_method
    assert target_method is not draft_method
    assert target_method.cfg.num_experts == mc.num_experts
    assert draft_method.cfg.num_experts == mc.dsv4_args.dspark_n_routed_experts
    # the draft's own layers carry the stage id, ready for its 3-layer cache
    assert [b.ffn.experts.layer_id for b in draft.layers.op_list] == list(range(_STAGES))


def test_dspark_draft_binds_aliases_and_window_kv(tmp_path, monkeypatch):
    _tp1()
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")

    mc = _mini_model_config(_write_full_mtp_checkpoint(tmp_path))
    model = _build_model(mc)
    t = model.model
    draft = t.mtp
    draft.bind(torch.device("cpu"), 3, t.embed, t.head)

    assert draft.embed is t.embed and draft.head is t.head
    for block in draft.layers.op_list:
        assert block.attn.window_kv_cache.shape == (3, mc.dsv4_args.sliding_window, mc.dsv4_args.head_dim)
        assert block.attn._freqs_cis is not None


def test_dspark_aux_capture_uses_config_order():
    _tp1()
    from freetoken.models.deepseek_v41.model import _DSparkAuxCapture

    cap = _DSparkAuxCapture((3, 1))  # deliberately not ascending
    cap.reset()
    streams = {i: torch.randn(2, 4, 5) for i in range(4)}

    cap.note(0, streams[0])  # target 1
    cap.note(1, streams[1])  # target 2 -> not a target
    cap.note(2, streams[2])  # target 3
    out = cap.concat()

    assert out is not None and out.shape == (2, 10)
    assert torch.equal(out[:, :5], streams[2].mean(dim=1))  # target 3 first (config order)
    assert torch.equal(out[:, 5:], streams[0].mean(dim=1))

    cap.reset()
    assert cap.concat() is None


def test_ftw_replay_skips_draft_dense_without_mtp(tmp_path, monkeypatch):
    """An FTW converted with --include-dspark carries ``model.mtp.*`` dense keys; a serve
    with MTP OFF must drop them (the model has no draft module then), MTP ON keeps them."""
    _tp1()
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.models.weight import load_weight

    d = str(tmp_path / "ftw")
    os.makedirs(d)
    _write_config(d)
    w = FTWWriter(d)
    w.add_tensor("model.norm.weight", torch.zeros(_H, dtype=torch.bfloat16), kind="weight")
    w.add_tensor("model.mtp.0.attn_norm.weight", torch.zeros(_H, dtype=torch.bfloat16), kind="weight")
    w.finalize({"quant_format": None, "expert_bank_num_layers": None})

    monkeypatch.delenv("FREETOKEN_ENABLE_MTP", raising=False)
    names = [n for n, _ in load_weight(d, torch.device("cpu"))]
    assert names == ["model.norm.weight"]

    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    names = [n for n, _ in load_weight(d, torch.device("cpu"))]
    assert set(names) == {"model.norm.weight", "model.mtp.0.attn_norm.weight"}


def test_decode_ngram_is_newest_first():
    """The decode engram 4-gram must be newest-first with pad-2 past the start --
    it must match the prefill `Engram.forward` / checkpoint reference hash order."""
    from freetoken.models.deepseek_v41.model import _decode_ngram_ids

    assert _decode_ngram_ids([10, 11, 12, 13, 14], 14, 5) == [14, 13, 12, 11]
    assert _decode_ngram_ids([10], 10, 1) == [10, 2, 2, 2]
    assert _decode_ngram_ids([10, 11], 11, 2) == [11, 10, 2, 2]
    assert _decode_ngram_ids([10, 11, 12], 12, 3) == [12, 11, 10, 2]


def test_ftw_without_dspark_banks_raises(tmp_path):
    """Serving MTP against an FTW converted without --include-dspark must fail loudly."""
    _tp1()
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.models.deepseek_v41.weight import dspark_expert_method, load_dspark_banks

    src = _write_dspark_checkpoint(tmp_path / "src")
    mc = _mini_model_config(src)
    method = dspark_expert_method(mc)

    out = str(tmp_path / "ftw")
    w = FTWWriter(out)
    w.add_tensor("gate_up#L00000", torch.zeros(_E, 2 * _I, _H // 2, dtype=torch.uint8), kind="experts_bank")
    w.finalize({"quant_format": "ds_fp4", "expert_bank_num_layers": 1})

    import pytest

    with pytest.raises(ValueError, match="--include-dspark"):
        load_dspark_banks(out, mc, method=method, num_stages=_STAGES,
                          device=torch.device("cpu"), dtype=torch.bfloat16)

