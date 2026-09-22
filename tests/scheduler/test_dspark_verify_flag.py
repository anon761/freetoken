"""`--dspark-verify` plumbing: arg -> EngineConfig -> DSparkManager resolver.

The verify implementation is a server-side contract (the deployment wrapper writes
it into the engine JSON); a typo must fail at boot, not silently fall back to the
default and change spec-decode numerics behind the operator's back.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from freetoken.distributed import try_get_tp_info
from freetoken.engine import DSPARK_VERIFY_MODES, EngineConfig
from freetoken.scheduler.dspark import resolve_verify_mode
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(extra: list[str]):
    config = _Config({"architectures": ["DeepseekV4ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", ANON_PATH, *extra])[0]


def test_engine_config_defaults_to_decode():
    config = EngineConfig(
        model_path=ANON_PATH, tp_info=try_get_tp_info(), dtype=torch.bfloat16
    )
    assert config.dspark_verify == "decode"
    assert DSPARK_VERIFY_MODES[0] == "decode"


def test_parse_args_carries_the_verify_mode():
    assert _parse([]).dspark_verify == "decode"
    assert _parse(["--dspark-verify", "prefill"]).dspark_verify == "prefill"
    assert _parse(["--dspark-verify", "decode"]).dspark_verify == "decode"


def test_parse_args_rejects_an_unknown_verify_mode():
    with pytest.raises(SystemExit):
        _parse(["--dspark-verify", "nonsense"])


def test_resolve_verify_mode_defaults_and_validates():
    assert resolve_verify_mode("") == "prefill"
    assert resolve_verify_mode("prefill") == "prefill"
    with pytest.raises(ValueError):
        resolve_verify_mode("nonsense")
