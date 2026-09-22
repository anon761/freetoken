"""Standalone MTP draft-head artifacts (--mtp file).

CPU-only: the compat fingerprint (a mismatched draft silently produces garbage
tokens — the panel and the engine must agree on the same check) and the
artifact key handling (mtp.-prefixed and unprefixed exports both load).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
from freetoken.models.qwen4_exp.weight import (
    iter_external_mtp_weights,
    mtp_compat_reasons,
    validate_mtp_compat,
)


@pytest.fixture(autouse=True)
def _tp1():
    if try_get_tp_info() is None:
        set_tp_info(0, 1)


def _args(**overrides):
    fields = dict(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
        num_qo_heads=4,
        num_kv_heads=2,
        num_experts=512,
        moe_intermediate_size=640,
        vocab_size=151936,
        linear_num_heads=32,
        linear_head_dim=128,
        mtp_num_layers=1,
        mtp_enabled=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _cfg(**overrides):
    return SimpleNamespace(qwen4_args=_args(**overrides))


class TestCompatFingerprint:
    def test_identical_geometry_is_compatible(self):
        assert mtp_compat_reasons(_cfg(), _cfg()) == []

    def test_mismatched_geometry_lists_every_field(self):
        artifact = _cfg(hidden_size=1280, num_experts=256, vocab_size=32000)
        reasons = mtp_compat_reasons(_cfg(), artifact)
        assert len(reasons) == 3
        assert any("hidden_size" in r for r in reasons)
        assert any("num_experts" in r for r in reasons)
        assert any("vocab_size" in r for r in reasons)

    def test_validate_raises_with_path_and_fields(self):
        with pytest.raises(ValueError, match="mtp-artikel|hidden_size|/data/mtp"):
            validate_mtp_compat(_cfg(), _cfg(hidden_size=1280), "/data/mtp")

    def test_qwen36_draft_for_qwen38_model_is_rejected(self):
        # der Warnfall aus dem Feature-Request: Qwen3.6er Draft (kleiner) am
        # Qwen3.8er Modell — die Geometrie-Felder fangen es ab
        reasons = mtp_compat_reasons(_cfg(), _cfg(hidden_size=1024, num_experts=128))
        assert reasons  # mindestens hidden_size + num_experts knallen


class TestArtifactKeyHandling:
    def test_unprefixed_keys_get_the_mtp_prefix(self, tmp_path):
        save_file({"fc_hidden.weight": torch.randn(4, 4)}, str(tmp_path / "draft.safetensors"))
        got = dict(iter_external_mtp_weights(str(tmp_path), torch.device("cpu"), _cfg(), None))
        assert set(got) == {"mtp.fc_hidden.weight"}

    def test_prefixed_keys_stay_prefixed(self, tmp_path):
        save_file({"mtp.fc_hidden.weight": torch.randn(4, 4)}, str(tmp_path / "draft.safetensors"))
        got = dict(iter_external_mtp_weights(str(tmp_path), torch.device("cpu"), _cfg(), None))
        assert set(got) == {"mtp.fc_hidden.weight"}

    def test_single_shard_path_loads_just_that_file(self, tmp_path):
        shard = tmp_path / "draft.safetensors"
        save_file({"fc_hidden.weight": torch.randn(4, 4)}, str(shard))
        got = dict(iter_external_mtp_weights(str(shard), torch.device("cpu"), _cfg(), None))
        assert set(got) == {"mtp.fc_hidden.weight"}

    def test_missing_dir_is_a_clear_error(self):
        with pytest.raises(ValueError, match="nicht gefunden"):
            next(iter_external_mtp_weights("/nonexistent/mtp", torch.device("cpu"), _cfg(), None))

    def test_dir_without_safetensors_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="safetensors"):
            next(iter_external_mtp_weights(str(tmp_path), torch.device("cpu"), _cfg(), None))


class TestExternalBaseStream:
    """--mtp file must not run the checkpoint's OWN embedded-head reader.

    Regression: a hybrid checkpoint declares ``mtp_num_hidden_layers`` yet ships no
    ``mtp.*`` tensors (the head lives in a separate artifact). The generic
    ``load_weight`` base stream used to read that embedded head anyway and died with
    "MTP expert tensor missing" before the artifact was ever consulted.
    """

    def test_base_stream_disables_embedded_head(self, monkeypatch):
        import freetoken.models.qwen4_exp.weight as q
        import freetoken.models.weight as w

        seen: dict[str, bool] = {}

        def fake_inner(model_path, device, *, include_moe_experts=True, include_mtp=True):
            seen["include_mtp"] = include_mtp
            yield "model.layers.0.x", torch.zeros(1)
            # A checkpoint that DID ship its own head: these must still be dropped.
            yield "mtp.layers.0.mlp.experts.gate_up_proj", torch.zeros(1)

        monkeypatch.setattr(w, "_load_weight_inner", fake_inner)
        monkeypatch.setattr(
            w, "_spec_for_model_path",
            lambda p: (object(), SimpleNamespace(module="freetoken.models.qwen4_exp.weight")),
        )
        monkeypatch.setattr(
            q, "iter_external_mtp_weights",
            lambda *a, **k: iter([("mtp.fc_hidden.weight", torch.zeros(1))]),
        )

        got = dict(w.load_weight("/base", torch.device("cpu"), mtp_path="/artifact"))

        assert seen["include_mtp"] is False
        assert "model.layers.0.x" in got
        assert "mtp.layers.0.mlp.experts.gate_up_proj" not in got
        assert set(k for k in got if k.startswith("mtp.")) == {"mtp.fc_hidden.weight"}

