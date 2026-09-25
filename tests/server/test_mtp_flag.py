"""``--mtp`` / ``--mtp-header`` plumbing.

The draft-head mode is resolved into the env gate the model parsers read
(``FREETOKEN_ENABLE_MTP``), and enabling MTP also forces the drain-safe
non-overlap loop Phase 1 requires. A bogus combination must fail at boot, not
silently serve without (or with) the draft head.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from freetoken.engine.config import mtp_artifact_dir
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"
_MTP_ENV = "FREETOKEN_ENABLE_MTP"
_OVERLAP_ENV = "FREETOKEN_DISABLE_OVERLAP_SCHEDULING"
# The Phase-1/3 runtime knobs are env-backed flags (see args._ENV_FLAG_*).
_KNOB_ENVS = (
    "FREETOKEN_MTP_VERIFY_GRAPH",
    "FREETOKEN_MTP_DRAFT_GRAPH",
    "FREETOKEN_MTP_COMMIT_GRAPH",
    "FREETOKEN_MTP_SAMPLED",
    "FREETOKEN_MTP_CHAIN",
    "FREETOKEN_MTP_NGRAM",
    "FREETOKEN_MTP_NGRAM_SIZE",
    "FREETOKEN_MTP_DRAFT_VOCAB",
    "FREETOKEN_MTP_HEAD_FP8",
)


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


@pytest.fixture(autouse=True)
def _restore_env():
    """parse_args writes os.environ directly — restore it after each test."""
    keys = (_MTP_ENV, _OVERLAP_ENV, *_KNOB_ENVS)
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _parse(extra: list[str]):
    config = _Config({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", ANON_PATH, *extra])[0]


def test_default_is_auto_and_leaves_env_untouched():
    args = _parse([])
    assert args.mtp == "auto"
    assert args.mtp_path == ""
    assert _MTP_ENV not in os.environ
    assert _OVERLAP_ENV not in os.environ


def test_on_enables_embedded_draft_and_keeps_overlap():
    args = _parse(["--mtp", "on"])
    assert args.mtp == "on"
    assert args.mtp_path == ""
    assert os.environ[_MTP_ENV] == "1"
    assert _OVERLAP_ENV not in os.environ


def test_off_forces_draft_off_even_with_env_gate():
    os.environ[_MTP_ENV] = "1"
    args = _parse(["--mtp", "off"])
    assert args.mtp == "off"
    assert _MTP_ENV not in os.environ


def test_file_mode_maps_header_to_mtp_path(tmp_path):
    header = tmp_path / "draft"
    header.mkdir()
    args = _parse(["--mtp", "file", "--mtp-header", str(header)])
    assert args.mtp == "file"
    assert args.mtp_path == str(header)
    assert os.environ[_MTP_ENV] == "1"
    assert _OVERLAP_ENV not in os.environ


def test_file_mode_accepts_a_single_safetensors_shard(tmp_path):
    shard = tmp_path / "draft.safetensors"
    shard.write_bytes(b"")
    args = _parse(["--mtp", "file", "--mtp-header", str(shard)])
    assert args.mtp == "file"
    assert args.mtp_path == str(shard)


def test_header_alone_implies_file_mode(tmp_path):
    header = tmp_path / "draft"
    header.mkdir()
    args = _parse(["--mtp-header", str(header)])
    assert args.mtp == "file"
    assert args.mtp_path == str(header)


def test_deprecated_mtp_path_folds_into_file_mode(tmp_path):
    header = tmp_path / "draft"
    header.mkdir()
    args = _parse(["--mtp-path", str(header)])
    assert args.mtp == "file"
    assert args.mtp_path == str(header)


def test_file_mode_without_header_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--mtp", "file"])


def test_on_with_header_is_rejected(tmp_path):
    header = tmp_path / "draft"
    header.mkdir()
    with pytest.raises(SystemExit):
        _parse(["--mtp", "on", "--mtp-header", str(header)])


def test_missing_header_path_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _parse(["--mtp", "file", "--mtp-header", str(tmp_path / "nope")])


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--mtp", "nonsense"])


def test_mtp_artifact_dir_resolves_a_single_file_to_its_parent(tmp_path):
    folder = tmp_path / "draft"
    folder.mkdir()
    shard = folder / "draft.safetensors"
    shard.write_bytes(b"")
    # A directory stays as-is (config.json lives in it); a single shard maps to
    # the directory that carries its family config.json.
    assert mtp_artifact_dir(str(folder)) == str(folder)
    assert mtp_artifact_dir(str(shard)) == str(folder)


class TestMtpRuntimeKnobs:
    """--mtp-verify-graph / --mtp-sampled / --mtp-chain / --mtp-ngram fold into the
    env gates the engine reads (args._ENV_FLAG_ENABLE/_ENV_FLAG_VALUE)."""

    def test_absent_flags_leave_env_untouched(self):
        _parse([])
        for env in _KNOB_ENVS:
            assert env not in os.environ

    def test_flags_set_their_env_gates(self):
        _parse([
            "--mtp-verify-graph", "--mtp-draft-graph", "--mtp-sampled", "--mtp-chain",
            "--mtp-ngram", "--mtp-ngram-size", "5",
        ])
        assert os.environ["FREETOKEN_MTP_VERIFY_GRAPH"] == "1"
        assert os.environ["FREETOKEN_MTP_DRAFT_GRAPH"] == "1"
        assert os.environ["FREETOKEN_MTP_SAMPLED"] == "1"
        assert os.environ["FREETOKEN_MTP_CHAIN"] == "1"
        assert os.environ["FREETOKEN_MTP_NGRAM"] == "1"
        assert os.environ["FREETOKEN_MTP_NGRAM_SIZE"] == "5"

    def test_default_on_knobs_can_be_switched_off(self):
        _parse(["--no-mtp-chain", "--no-mtp-head-fp8", "--mtp-draft-vocab", "0"])
        assert os.environ["FREETOKEN_MTP_CHAIN"] == "0"
        assert os.environ["FREETOKEN_MTP_HEAD_FP8"] == "0"
        assert os.environ["FREETOKEN_MTP_DRAFT_VOCAB"] == "0"
