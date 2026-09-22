"""Env-backed runtime knobs exposed as CLI flags.

Each flag folds its value into the corresponding FREETOKEN_* env (before the TP
workers spawn); an omitted flag leaves the env untouched, and a flag wins over a
pre-set env. This is the CLI surface for the knobs whose consumers read
os.environ directly.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from freetoken.server.args import (
    _ENV_FLAG_BOOL,
    _ENV_FLAG_DISABLE,
    _ENV_FLAG_ENABLE,
    _ENV_FLAG_VALUE,
    parse_args,
)

ALL_ENVS = (
    set(_ENV_FLAG_ENABLE.values())
    | set(_ENV_FLAG_DISABLE.values())
    | set(_ENV_FLAG_BOOL.values())
    | set(_ENV_FLAG_VALUE.values())
)


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in ALL_ENVS}
    for k in ALL_ENVS:
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
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_omitted_flags_leave_the_env_untouched():
    _parse([])
    for env in ALL_ENVS:
        assert env not in os.environ, env


@pytest.mark.parametrize(
    ("argv", "env", "want"),
    [
        (["--mtp-draft-tokens", "5"], "FREETOKEN_MTP_DRAFT_TOKENS", "5"),
        (["--dspark-k", "4"], "FREETOKEN_DSPARK_K", "4"),
        (["--mamba-ssm-dtype", "bfloat16"], "FREETOKEN_MAMBA_SSM_DTYPE", "bfloat16"),
        (["--pin-budget-gb", "24.5"], "FREETOKEN_PIN_BUDGET_GB", "24.5"),
        (["--load-vision"], "FREETOKEN_LOAD_VISION", "1"),
        (["--no-load-vision"], "FREETOKEN_LOAD_VISION", "0"),
        (["--pynccl-max-buffer-size", "2G"], "FREETOKEN_PYNCCL_MAX_BUFFER_SIZE", "2G"),
        (["--hybrid-fetch-policy", "lowest_id"], "FREETOKEN_HYBRID_FETCH", "lowest_id"),
        (["--tp-reduce-fp8"], "FREETOKEN_TP_REDUCE_FP8", "1"),
        (["--no-tp-reduce-fp8"], "FREETOKEN_TP_REDUCE_FP8", "0"),
        (["--tp-reduce-fp8-min-bytes", "4096"], "FREETOKEN_TP_REDUCE_FP8_MIN_BYTES", "4096"),
        (["--ple-sync", "gate"], "FREETOKEN_PLE_SYNC", "gate"),
        (["--ple-io-uring"], "FREETOKEN_PLE_IO_URING", "1"),
        (["--qsa-torch-topk", "512"], "FREETOKEN_QSA_TORCH_TOPK", "512"),
        (["--forward-unknown-tools"], "FREETOKEN_FORWARD_UNKNOWN_TOOLS", "1"),
        (["--no-forward-unknown-tools"], "FREETOKEN_FORWARD_UNKNOWN_TOOLS", "0"),
        (["--no-hybrid-overlap"], "FREETOKEN_HYBRID_OVERLAP", "0"),
        (["--no-fused-copy"], "FREETOKEN_FUSED_COPY", "0"),
        (["--no-cpu-moe-flag-sync"], "FREETOKEN_CPU_MOE_FLAG_SYNC", "0"),
        (["--bank-cuda-alloc"], "FREETOKEN_BANK_CUDA_ALLOC", "1"),
        (["--no-bank-cuda-alloc"], "FREETOKEN_BANK_CUDA_ALLOC", "0"),
        (["--skip-bank-pin"], "FREETOKEN_SKIP_BANK_PIN", "1"),
        (["--cpu-moe-isa", "avx512"], "FREETOKEN_CPU_MOE_ISA", "avx512"),
        (["--dspark-debug"], "FREETOKEN_DSPARK_DEBUG", "1"),
        (["--dspark-diff"], "FREETOKEN_DSPARK_DIFF", "1"),
        (["--dspark-timing"], "FREETOKEN_DSPARK_TIMING", "1"),
        (["--dspark-force-a0"], "FREETOKEN_DSPARK_FORCE_A0", "1"),
        (["--api-log-dir", "/tmp/reqlog"], "FREETOKEN_API_LOG_DIR", "/tmp/reqlog"),
        (["--no-m3-sparse"], "FREETOKEN_M3_SPARSE", "0"),
        (["--m3-inner-backend", "flashinfer"], "FREETOKEN_M3_INNER_BACKEND", "flashinfer"),
        (["--m3-max-layers", "4"], "FREETOKEN_M3_MAX_LAYERS", "4"),
        (["--no-glm-dsa"], "FREETOKEN_GLM_DSA", "0"),
        (["--glm-dsa-max-layers", "4"], "FREETOKEN_GLM_DSA_MAX_LAYERS", "4"),
        (["--no-glm5-dsa"], "FREETOKEN_GLM5_DSA", "0"),
        (["--glm5-max-layers", "4"], "FREETOKEN_GLM5_MAX_LAYERS", "4"),
        (["--engram-backend", "ram"], "FREETOKEN_ENGRAM_BACKEND", "ram"),
    ],
)
def test_flag_folds_into_env(argv, env, want):
    _parse(argv)
    assert os.environ[env] == want


def test_flag_wins_over_a_preset_env():
    os.environ["FREETOKEN_HYBRID_OVERLAP"] = "1"
    _parse(["--no-hybrid-overlap"])
    assert os.environ["FREETOKEN_HYBRID_OVERLAP"] == "0"


def test_unknown_value_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--mamba-ssm-dtype", "float8"])
    with pytest.raises(SystemExit):
        _parse(["--ple-sync", "nonsense"])
