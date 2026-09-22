"""FTW conversion carries everything the source checkpoint has.

Two model-specific carriers, verified against a fabricated toy qwen4_exp checkpoint
(CPU, no GPU, no real conversion run):

* ``side_table_files`` — the qwen4_exp PLE n-gram table shards (``model-plefp8-*``,
  ~47.7 GiB in production) load OUTSIDE the dense stream, so the converter's metadata
  walk (which skips every ``.safetensors``) must receive them from this hook to copy
  them verbatim. Without them the FTW dir cannot serve (load_ple_table finds no files).
* MTP draft head — the converter forces ``FREETOKEN_ENABLE_MTP=1`` when the source
  ships ``mtp_num_hidden_layers``, so the dense stream includes ``mtp.*``; the FTW
  replay skips those keys when the SERVING engine runs without MTP (strict
  load_state_dict would reject them otherwise).
"""
from __future__ import annotations

import json
import os

from freetoken.models.weight import side_table_files

from tests.models.qwen4_exp.common import hf_config


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _make_qwen4_exp_dir(t, *, with_mtp: bool):
    """A toy qwen4_exp checkpoint dir (real config fields via the shared toy builder)."""
    dir_ = str(t)
    os.makedirs(dir_, exist_ok=True)
    cfg = _to_jsonable(hf_config(num_layers=4, mtp_num_hidden_layers=1 if with_mtp else 0))
    with open(os.path.join(dir_, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    shards = ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"]
    weight_map = {
        f"model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_{i}.weight": s
        for i, s in enumerate(shards)
    }
    with open(os.path.join(dir_, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map}, f)
    for s in shards:
        with open(os.path.join(dir_, s), "wb") as f:
            f.write(b"\x00" * 64)
    return dir_, shards


def test_side_table_files_lists_the_ple_shards(tmp_path):
    dir_, shards = _make_qwen4_exp_dir(tmp_path, with_mtp=True)
    got = side_table_files(dir_)
    assert [os.path.basename(p) for p in got] == shards


def test_side_table_files_without_a_model_is_empty(tmp_path):
    dir_ = str(tmp_path / "plain")
    os.makedirs(dir_)
    with open(os.path.join(dir_, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, f)
    assert side_table_files(dir_) == []


def test_side_table_files_on_a_missing_dir_is_empty(tmp_path):
    assert side_table_files(str(tmp_path / "gibt-es-nicht")) == []


def test_checkpoint_cli_forwards_side_payload_flags(monkeypatch, tmp_path):
    """--include-engram / --include-dspark reach convert_checkpoint; default off."""
    from freetoken.checkpoint import __main__ as cli

    captured: dict = {}

    def fake_convert(model, out, **kw):
        captured.clear()
        captured.update(kw)
        return {
            "counts": {"weight": 0, "experts_bank": 0},
            "total_bytes": 0,
            "shards": [],
            "quant_format": None,
            "fingerprint": None,
            "includes": {"engram": kw.get("include_engram"), "dspark": kw.get("include_dspark")},
        }

    monkeypatch.setattr(cli, "convert_checkpoint", fake_convert)
    monkeypatch.setattr(cli, "assign_gpu", lambda gpu: None)
    monkeypatch.setattr(cli, "bind_assigned_gpu", lambda: type("B", (), {"index": 0})())

    assert cli.main(["--model", "src", "--out", "dst", "--include-engram", "--include-dspark"]) == 0
    assert captured["include_engram"] is True
    assert captured["include_dspark"] is True

    assert cli.main(["--model", "src", "--out", "dst"]) == 0
    assert captured["include_engram"] is False
    assert captured["include_dspark"] is False
