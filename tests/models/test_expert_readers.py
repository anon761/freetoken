"""Expert-source readers as shared, family-independent modules (CPU-only).

Both the NVFP4 (``nvfp4_banks``) and block-fp8 (``fp8_block_banks``) readers must turn a
synthetic per-expert checkpoint into the canonical pieces ``build_expert_banks`` packs,
regardless of which model family consumes them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.fp8_block_banks import iter_fp8_block_expert_pieces

E, H, I = 2, 256, 128


@pytest.fixture(autouse=True)
def _tp1():
    import freetoken.distributed.info as info

    saved = info._TP_INFO
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    yield
    info._TP_INFO = saved


def _config():
    return SimpleNamespace(
        num_moe_layers=1, num_experts=E, hidden_size=H,
        moe_intermediate_size=I, num_layers=1, first_k_dense_replace=0,
    )


def _write(folder, tensors):
    save_file(tensors, str(folder / "model.safetensors"))
    json.dump(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}},
        open(folder / "model.safetensors.index.json", "w"),
    )


def _fp8_tensors():
    tensors = {}
    for e in range(E):
        for proj, out, inn in (("gate", I, H), ("up", I, H), ("down", H, I)):
            base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}_proj"
            tensors[base + ".weight"] = torch.randint(
                0, 256, (out, inn), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            tensors[base + ".weight_scale_inv"] = torch.ones(
                out // 128, inn // 128, dtype=torch.bfloat16
            )
    return tensors


def test_fp8_block_reader_yields_canonical_pieces(tmp_path):
    tensors = _fp8_tensors()
    _write(tmp_path, tensors)

    pieces = list(iter_fp8_block_expert_pieces(str(tmp_path), _config(), parallel=False))
    assert [p[1:3] for p in pieces] == [(0, 1), (1, 2)]
    for layer, e0, e1, piece in pieces:
        assert layer == 0 and e1 - e0 == 1
        assert set(piece) == {
            "gate", "up", "down", "gate_scale", "up_scale", "down_scale",
        }
        e = e0
        src = tensors[f"model.language_model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        # compare raw bytes: random fp8 bit patterns include NaNs, which break torch.equal
        assert torch.equal(piece["gate"][0].view(torch.uint8), src.view(torch.uint8))
        assert piece["gate"].shape == (1, I, H)
        assert piece["gate"].dtype is torch.float8_e4m3fn
        assert piece["gate_scale"].shape == (1, I // 128, H // 128)
        assert piece["down_scale"].shape == (1, H // 128, I // 128)


def test_fp8_block_reader_slices_experts_for_tp2(tmp_path):
    import freetoken.distributed.info as info

    i = 256  # I_loc = 128 at tp=2 -> block-aligned
    tensors = {}
    for e in range(E):
        for proj, out, inn in (("gate", i, H), ("up", i, H), ("down", H, i)):
            base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}_proj"
            tensors[base + ".weight"] = torch.randint(
                0, 256, (out, inn), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            tensors[base + ".weight_scale_inv"] = torch.ones(
                out // 128, inn // 128, dtype=torch.bfloat16
            )
    _write(tmp_path, tensors)
    config = SimpleNamespace(
        num_moe_layers=1, num_experts=E, hidden_size=H,
        moe_intermediate_size=i, num_layers=1, first_k_dense_replace=0,
    )

    shards = []
    for rank in range(2):
        info._TP_INFO = None
        set_tp_info(rank, 2)
        pieces = list(iter_fp8_block_expert_pieces(str(tmp_path), config, parallel=False))
        assert [p[1:3] for p in pieces] == [(0, 1), (1, 2)]
        for _layer, _e0, _e1, piece in pieces:
            assert piece["gate"].shape == (1, i // 2, H)
            assert piece["up"].shape == (1, i // 2, H)
            assert piece["gate_scale"].shape == (1, i // 2 // 128, H // 128)
            assert piece["down"].shape == (1, H, i // 2)
            assert piece["down_scale"].shape == (1, H // 128, i // 2 // 128)
        shards.append(pieces)
    info._TP_INFO = None
    set_tp_info(0, 1)

    # rank shards concatenate back to the full expert on the sliced axis
    for e in range(E):
        for role, axis, proj, kind in (
            ("gate", 1, "gate", "weight"),
            ("gate_scale", 1, "gate", "weight_scale_inv"),
            ("up", 1, "up", "weight"),
            ("down", 2, "down", "weight"),
            ("down_scale", 2, "down", "weight_scale_inv"),
        ):
            a = shards[0][e][3][role]
            b = shards[1][e][3][role]
            full = tensors[
                f"model.language_model.layers.0.mlp.experts.{e}.{proj}_proj.{kind}"
            ]
            got = torch.cat([a, b], dim=axis).squeeze(0)
            if full.dtype == torch.float8_e4m3fn:
                assert torch.equal(got.view(torch.uint8), full.view(torch.uint8))
            else:
                assert torch.equal(got, full)


def test_generic_dispatch_uses_the_shared_fp8_reader(tmp_path):
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import iter_expert_pieces

    _write(tmp_path, _fp8_tensors())
    config = _config()
    config.architectures = ["Qwen4ExpForConditionalGeneration"]  # no family fp8 hook anymore
    pieces = list(iter_expert_pieces(str(tmp_path), config, QuantKind.FP8_BLOCK, parallel=False))
    assert [p[1:3] for p in pieces] == [(0, 1), (1, 2)]


def test_nvfp4_reader_yields_canonical_pieces(tmp_path):
    from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
    from freetoken.models.qwen4_exp.weight import _NVFP4_SOURCE_SPEC

    tensors = {}
    for e in range(E):
        for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
            tensors[base + ".weight"] = torch.zeros(out, inn // 2, dtype=torch.uint8)
            tensors[base + ".weight_scale"] = torch.ones(out, inn // 16, dtype=torch.float8_e4m3fn)
            tensors[base + ".weight_scale_2"] = torch.tensor(0.5)
    _write(tmp_path, tensors)

    pieces = list(iter_nvfp4_expert_pieces(str(tmp_path), _config(), _NVFP4_SOURCE_SPEC, parallel=False))
    assert [p[1:3] for p in pieces] == [(0, 1), (1, 2)]
    for _layer, _e0, _e1, piece in pieces:
        assert set(piece) == {
            "gate", "gate_scale", "gate_global",
            "up", "up_scale", "up_global",
            "down", "down_scale", "down_global",
        }
        assert piece["gate"].dtype is torch.uint8
        assert piece["gate_global"].dtype is torch.float16


def test_nvfp4_reader_accepts_the_text_only_root(tmp_path):
    """A language-model-only export (arch ``Qwen4ExpForCausalLM``) drops the
    ``model.language_model.`` segment: the expert reader must resolve the same canonical
    pieces under the bare ``model.`` root, while the MTP head's experts stay excluded."""
    from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
    from freetoken.models.qwen4_exp.weight import _EXPERT_KEY_RE, _NVFP4_SOURCE_SPEC

    tensors = {}
    for e in range(E):
        for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            tensors[base + ".weight"] = torch.zeros(out, inn // 2, dtype=torch.uint8)
            tensors[base + ".weight_scale"] = torch.ones(out, inn // 16, dtype=torch.float8_e4m3fn)
            tensors[base + ".weight_scale_2"] = torch.tensor(0.5)
    _write(tmp_path, tensors)

    pieces = list(iter_nvfp4_expert_pieces(str(tmp_path), _config(), _NVFP4_SOURCE_SPEC, parallel=False))
    assert [p[1:3] for p in pieces] == [(0, 1), (1, 2)]
    assert _EXPERT_KEY_RE.match("mtp.layers.0.mlp.experts.0.gate_proj.weight") is None
