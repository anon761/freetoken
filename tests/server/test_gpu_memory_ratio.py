from __future__ import annotations

import argparse

import pytest

from freetoken.gpu_select import (
    gpu_memory_ratio_arg,
    memory_ratio_arg,
    parse_gpu_memory_ratio,
    parse_memory_ratio,
    resolve_memory_ratios,
)


def test_parse_index_entries():
    assert parse_gpu_memory_ratio("0,0.8;1,0.9") == (("0", 0.8), ("1", 0.9))


def test_parse_colon_entries_are_shell_safe():
    assert parse_gpu_memory_ratio("0:0.8,1:0.9") == (("0", 0.8), ("1", 0.9))
    assert parse_gpu_memory_ratio("0:0.8;1:0.9") == (("0", 0.8), ("1", 0.9))


def test_parse_single_entry_and_whitespace():
    assert parse_gpu_memory_ratio(" 0 , 0.75 ") == (("0", 0.75),)
    assert parse_gpu_memory_ratio(" 0 : 0.75 ") == (("0", 0.75),)


def test_parse_uuid_prefix_is_canonicalized():
    assert parse_gpu_memory_ratio("gpu-9e8d7c6b,0.5") == (("GPU-9e8d7c6b", 0.5),)


@pytest.mark.parametrize(
    "value", ["0", "0,", "0,abc", "0,1.5", "0,-0.1", "x,0.5", "", "0,0.5;0,0.6", "0:abc"]
)
def test_parse_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_gpu_memory_ratio(value)


def test_gpu_memory_ratio_arg_wraps_error():
    with pytest.raises(argparse.ArgumentTypeError):
        gpu_memory_ratio_arg("0,2")


def test_memory_ratio_accepts_float_or_list():
    assert parse_memory_ratio("0.5") == 0.5
    assert parse_memory_ratio(" 0.9 ") == 0.9
    assert parse_memory_ratio("0:0.5,1:0.95") == (("0", 0.5), ("1", 0.95))
    assert parse_memory_ratio("0,0.5;1,0.95") == (("0", 0.5), ("1", 0.95))


@pytest.mark.parametrize("value", ["", "abc", "0:", "0:2"])
def test_memory_ratio_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_memory_ratio(value)
    with pytest.raises(argparse.ArgumentTypeError):
        memory_ratio_arg(value)


def test_resolve_by_index_without_nvml():
    ratios = resolve_memory_ratios((("1", 0.8),), ("0", "1"), None, 2, 0.9)
    assert ratios == (0.9, 0.8)


def test_resolve_by_uuid_prefix_against_assigned():
    assigned = ("GPU-aaaa", "GPU-bbbb")
    ratios = resolve_memory_ratios((("GPU-bbbb", 0.5),), assigned, assigned, 2, 0.9)
    assert ratios == (0.9, 0.5)


def test_resolve_no_gpu_uses_ordinals():
    ratios = resolve_memory_ratios((("1", 0.7),), (), None, 2, 0.9)
    assert ratios == (0.9, 0.7)


def test_resolve_unknown_gpu_raises():
    with pytest.raises(ValueError):
        resolve_memory_ratios((("2", 0.8),), ("0", "1"), None, 2, 0.9)


def test_resolve_two_entries_same_rank_raises():
    with pytest.raises(ValueError):
        resolve_memory_ratios((("0", 0.8), ("0", 0.7)), ("0", "1"), None, 2, 0.9)
