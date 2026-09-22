"""MTP draft head: loader contract + TP sharding + draft-step math.

The checkpoint's ``mtp.*`` tensors (all BF16, un-fused q/k/v and shared gate/up, HC
down+inject separate, routed experts stacked fused) must fuse + shard into exactly the
shapes ``Qwen4ExpMTP`` allocates -- verified here by replaying the loader pipeline
(:func:`_try_fuse` + :func:`tp_shard_key`) over a synthetic checkpoint dict and loading
strictly (BaseOP.load_state_dict asserts shape/dtype per key, so a strict load proves the
allocation contract), plus a reference check of the ``draft_step`` input fusion
(fastllm qwen4_exp.cpp semantics: per-stream fc_hidden + broadcast fc_embedding onto
every stream).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTP
from freetoken.models.qwen4_exp.weight import _rename, _try_fuse, tp_shard_key

from .common import toy_hf_config


@pytest.fixture(autouse=True)
def _restore_tp():
    """set_tp_info is a process global -- undo this module's rank/world changes."""
    import freetoken.distributed.info as info

    saved = info._TP_INFO
    yield
    info._TP_INFO = saved


class _TP:
    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.size = size


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info

    info._TP_INFO = None
    set_tp_info(rank, size)


def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    return parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))


def _fill(op, gen: torch.Generator, scale: float = 0.05) -> None:
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, scale, generator=gen)


def _zero_hc_pad(model) -> None:
    """The merged down+inject weights carry zero pad rows; keep them zero under _fill."""
    pad = 12  # toy geometry: lowrank 16 + hc 4 -> (-(16 + 4)) % 16
    for name, tensor in model.state_dict().items():
        if name.endswith("_hyper_connection.input_mix_weight_down_block_inject.weight"):
            lowrank = tensor.shape[0] - pad - 4
            tensor[lowrank + 4 :].zero_()


def _checkpoint_dict(state: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    """Invert the loader fusions: TP=1 model state -> raw checkpoint names."""
    args = config.qwen4_args
    qo, kv, hd = config.num_qo_heads, config.num_kv_heads, config.head_dim
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        if name.endswith(".self_attn.qkv_proj.weight"):
            base = name[: -len("qkv_proj.weight")]
            a, b = 2 * qo * hd, kv * hd
            out[base + "q_proj.weight"] = t[:a]
            out[base + "k_proj.weight"] = t[a : a + b]
            out[base + "v_proj.weight"] = t[a + b :]
        elif name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
            base = name[: -len("gate_up_proj.weight")]
            i = config.shared_expert_intermediate_size
            out[base + "gate_proj.weight"] = t[:i]
            out[base + "up_proj.weight"] = t[i:]
        elif name.endswith(
            "_hyper_connection.input_mix_weight_down_block_inject.weight"
        ):
            base = name[: -len("input_mix_weight_down_block_inject.weight")]
            out[base + "input_mix_weight_down.weight"] = t[: args.hc_lowrank]
            out[base + "block_inject_weight.weight"] = t[
                args.hc_lowrank : args.hc_lowrank + args.hc_count
            ]
        else:
            out[name] = t
    # the raw checkpoint stores the stacked experts WITH ".weight" (the loader's
    # _rename strips it for the model's raw-tensor keys)
    out = {
        (
            k + ".weight"
            if k.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
            else k
        ): v
        for k, v in out.items()
    }
    return out


def _load_like_loader(model, ckpt: dict[str, torch.Tensor], config, tp, tmp_path) -> None:
    """Replay the loader: dense rename/fusion + the on-disk MTP expert normalizer."""
    from safetensors.torch import save_file

    from freetoken.models.qwen4_exp.weight import iter_mtp_experts

    # Routed experts (stacked OR per-expert) are normalized by iter_mtp_experts from an
    # on-disk checkpoint, so persist the synthetic dict (clone: the inverted slices share
    # storage, which safetensors refuses).
    save_file({k: v.contiguous().clone() for k, v in ckpt.items()}, str(tmp_path / "model.safetensors"))

    buf: dict[str, dict[int, torch.Tensor]] = {}
    state: dict[str, torch.Tensor] = {}
    for raw_name, tensor in ckpt.items():
        if ".mlp.experts." in raw_name:
            continue
        name = _rename(raw_name, keep_mtp=True)
        assert name is not None, raw_name
        fused = _try_fuse(name, tensor, buf)
        if fused is None:
            state[name] = tp_shard_key(name, tensor, config, tp)
        elif fused != ():
            state[fused[0]] = tp_shard_key(fused[0], fused[1], config, tp)
    assert not buf, f"incomplete fusions: {sorted(buf)}"
    for name, tensor in iter_mtp_experts(str(tmp_path), config, tp):
        state[name] = tensor
    model.load_state_dict(state, prefix="mtp")


# --------------------------------------------------------------------------------------
# config gate
# --------------------------------------------------------------------------------------


def test_mtp_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FREETOKEN_ENABLE_MTP", raising=False)
    config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    assert config.qwen4_args.mtp_num_layers == 1
    assert config.qwen4_args.mtp_enabled is False

    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    assert config.qwen4_args.mtp_enabled is True

    no_mtp = parse_config(toy_hf_config(num_layers=4))
    assert no_mtp.qwen4_args.mtp_num_layers == 0
    assert no_mtp.qwen4_args.mtp_enabled is False


# --------------------------------------------------------------------------------------
# TP shard rules: the stacked MTP experts reconstruct the full tensors
# --------------------------------------------------------------------------------------

_E, _I, _H = 8, 64, 128


def _full(shape) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(0))


def test_stacked_gate_up_two_halves():
    name = "mtp.layers.0.mlp.experts.gate_up_proj"
    config = parse_config(toy_hf_config(num_layers=4))
    full = _full((_E, 2 * _I, _H))
    I_loc = _I // 2
    parts = [tp_shard_key(name, full, config, _TP(r, 2)) for r in range(2)]
    assert parts[0].shape == (_E, 2 * I_loc, _H)
    rebuilt = torch.zeros_like(full)
    for r, p in enumerate(parts):
        rebuilt[:, r * I_loc : (r + 1) * I_loc] = p[:, :I_loc]
        rebuilt[:, _I + r * I_loc : _I + (r + 1) * I_loc] = p[:, I_loc:]
    assert torch.equal(rebuilt, full)


def test_stacked_down_cols():
    name = "mtp.layers.0.mlp.experts.down_proj"
    config = parse_config(toy_hf_config(num_layers=4))
    full = _full((_E, _H, _I))
    I_loc = _I // 2
    parts = [tp_shard_key(name, full, config, _TP(r, 2)) for r in range(2)]
    assert parts[0].shape == (_E, _H, I_loc)
    assert torch.equal(torch.cat(parts, dim=2), full)


def test_stacked_tp1_identity():
    config = parse_config(toy_hf_config(num_layers=4))
    full = _full((_E, 2 * _I, _H))
    assert torch.equal(
        tp_shard_key("mtp.layers.0.mlp.experts.gate_up_proj", full, config, _TP(0, 1)),
        full,
    )


# --------------------------------------------------------------------------------------
# loader contract: synthetic checkpoint -> fuse -> shard -> strict load
# --------------------------------------------------------------------------------------

# The real checkpoint ships exactly these mtp tensors (fusions pre-collapse).
_REAL_MTP_KEY_COUNT = 31


def _mtp_ckpt(monkeypatch: pytest.MonkeyPatch):
    config = _config(monkeypatch)
    _set_tp(0, 1)
    gen = torch.Generator().manual_seed(42)
    ref = Qwen4ExpMTP(config)
    _fill(ref, gen)
    _zero_hc_pad(ref)
    ckpt = _checkpoint_dict(ref.state_dict(), config)
    ckpt = {f"mtp.{k}": v for k, v in ckpt.items()}
    return config, ref, ckpt


def test_mtp_load_contract_tp1(monkeypatch: pytest.MonkeyPatch, tmp_path):
    config, ref, ckpt = _mtp_ckpt(monkeypatch)
    assert len(ckpt) == _REAL_MTP_KEY_COUNT

    fresh = Qwen4ExpMTP(config)
    _load_like_loader(fresh, ckpt, config, _TP(0, 1), tmp_path)
    for name, tensor in ref.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor), name


def test_mtp_load_contract_tp2(monkeypatch: pytest.MonkeyPatch, tmp_path):
    config, ref, ckpt = _mtp_ckpt(monkeypatch)
    _set_tp(0, 2)
    model = Qwen4ExpMTP(config)
    _load_like_loader(model, ckpt, config, _TP(0, 2), tmp_path)

    full, loc = ref.state_dict(), model.state_dict()
    assert set(full) == set(loc)
    I_loc = config.moe_intermediate_size // 2
    for name, b in full.items():
        a = loc[name]
        if a.shape == b.shape:
            assert torch.equal(a, b), name
            continue
        # Shape-divergent keys are the TP-sharded ones. The strict load above already
        # proved every local shape matches the model's allocation; the standard rules'
        # content roundtrips live in test_qwen4_tp_sharding.py. Here we reconstruct the
        # NEW rules (stacked experts) and equality-check everything replicated.
        if not (
            name.endswith("mlp.experts.gate_up_proj")
            or name.endswith("mlp.experts.down_proj")
        ):
            continue
        rule_name = f"mtp.{name}"
        if name.endswith("gate_up_proj"):
            assert a.shape == (_E, 2 * I_loc, _H)
            rebuilt = torch.zeros_like(b)
            for r, p in enumerate((a, tp_shard_key(rule_name, b, config, _TP(1, 2)))):
                rebuilt[:, r * I_loc : (r + 1) * I_loc] = p[:, :I_loc]
                rebuilt[:, _I + r * I_loc : _I + (r + 1) * I_loc] = p[:, I_loc:]
            assert torch.equal(rebuilt, b), name
        else:
            assert a.shape == (_E, _H, I_loc)
            assert torch.equal(
                torch.cat((a, tp_shard_key(rule_name, b, config, _TP(1, 2))), dim=2), b
            ), name


# --------------------------------------------------------------------------------------
# draft_step math
# --------------------------------------------------------------------------------------


def test_draft_step_fusion_math(monkeypatch: pytest.MonkeyPatch):
    config = _config(monkeypatch)
    _set_tp(0, 1)
    gen = torch.Generator().manual_seed(7)
    mtp = Qwen4ExpMTP(config)
    _fill(mtp, gen)

    hc, hidden = config.qwen4_args.hc_count, config.hidden_size
    T = 3
    R_last = torch.randn(T, hc * hidden, generator=torch.Generator().manual_seed(11))
    tokens = torch.arange(T)
    emb = torch.randn(
        config.vocab_size, hidden, generator=torch.Generator().manual_seed(3)
    )

    inputs: list[torch.Tensor] = []
    block = mtp.layers.op_list[0]
    original = block.forward

    def _stub(R, batch):
        inputs.append(R.clone())
        return R * 2.0

    block.forward = _stub
    R_next, sample_hidden = mtp.draft_step(
        lambda ids: emb[ids], R_last, tokens, batch=None
    )
    block.forward = original

    eps = config.rms_norm_eps

    def _norm(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * (1.0 + w.float())).type_as(x)

    e_proj = (
        _norm(emb[tokens], mtp.pre_fc_norm_embedding.weight) @ mtp.fc_embedding.weight.T
    )
    h_proj = (
        _norm(R_last, mtp.pre_fc_norm_hidden.weight).reshape(T * hc, hidden)
        @ mtp.fc_hidden.weight.T
    ).reshape(T, hc, hidden)
    expected_input = (h_proj + e_proj.unsqueeze(1)).flatten(1)
    assert torch.allclose(inputs[0], expected_input, atol=1e-5)

    assert torch.allclose(R_next, 2.0 * expected_input, atol=1e-5)
    mixed, _ = mtp.hyper_connection_mixer.mix(2.0 * expected_input)
    assert torch.allclose(sample_hidden, mixed, atol=1e-5)


# --------------------------------------------------------------------------------------
# QSA group/pool extension: the MTP layers ride the full-attention group's id map
# --------------------------------------------------------------------------------------


def test_mtp_group_extends_qsa_layer_ids(monkeypatch: pytest.MonkeyPatch):
    """With MTP enabled the full group's layer ids continue past the main stack so both
    the backend slot map (_idx_slot) and the pool's _dense map resolve the draft layers;
    the compressed slab + pending ring grow by mtp_num_layers rows of depth."""
    monkeypatch.delenv("FREETOKEN_ENABLE_MTP", raising=False)
    base_config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    base_ids = base_config.kv_cache_group_specs()[0].layer_ids

    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    group = config.kv_cache_group_specs()[0]
    assert group.layer_ids == base_ids + (config.num_layers,)
    assert group.num_index_layers == len(base_ids) + 1

    monkeypatch.delenv("FREETOKEN_ENABLE_MTP", raising=False)
    config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    group = config.kv_cache_group_specs()[0]
    assert config.num_layers not in group.layer_ids
    assert group.num_index_layers == len(group.layer_ids)


def test_qsa_pool_addresses_the_mtp_layer(monkeypatch: pytest.MonkeyPatch):
    """The pool's layer map resolves layer_id = num_layers (the draft block's slot)
    into its OWN K/V slab + index rows, disjoint from the main stack's slots; linear
    layers stay unmapped. (store_kv itself is a CUDA JIT kernel, so this only checks
    the addressability contract.)"""
    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    config = parse_config(toy_hf_config(num_layers=4, mtp_num_hidden_layers=1))
    spec = config.kv_cache_group_specs()[0]

    from freetoken.kvcache.qsa_pool import QSAKVCache

    pool = QSAKVCache(
        num_kv_heads=spec.num_kv_heads,
        num_layers=max(config.num_layers, max(spec.layer_ids) + 1),
        head_dim=spec.head_dim,
        num_pages=8,
        page_size=spec.index_ratio,  # one group per page keeps the toy math trivial
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=spec.index_head_dim,
        num_index_layers=spec.num_index_layers,
        index_ratio=spec.index_ratio,
        num_req_slots=4,
        layer_ids=spec.layer_ids,
    )
    draft_layer = config.num_layers  # 4
    main_full = spec.layer_ids[0]
    assert pool.k_cache(draft_layer).data_ptr() != pool.k_cache(main_full).data_ptr()
    assert pool.v_cache(draft_layer).data_ptr() != pool.v_cache(main_full).data_ptr()
    # one index slab per (main + draft) layer; linear layers have no paged KV storage
    assert pool.cmp_k_cache(spec.num_index_layers - 1).shape[0] > 0
    with pytest.raises(KeyError):
        pool.k_cache(0)  # a linear_attention layer id
