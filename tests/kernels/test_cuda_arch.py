"""tvm-ffi JIT arch fallback (CPU-only).

tvm-ffi cannot always detect the SM version itself (containers/headless, or a
driver/library mismatch); the runtime derives TVM_FFI_CUDA_ARCH_LIST from torch,
which queried the same GPU through the CUDA runtime.
"""

from __future__ import annotations

import os

import torch

from freetoken.kernel.utils import _ensure_cuda_arch_list


def test_sets_arch_list_from_torch_when_unset(monkeypatch):
    monkeypatch.delenv("TVM_FFI_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 6) if i == 0 else (8, 9))
    _ensure_cuda_arch_list()
    assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "8.6 8.9"


def test_existing_env_wins(monkeypatch):
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "12.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (9, 0))
    _ensure_cuda_arch_list()
    assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "12.0"


def test_no_cuda_leaves_unset(monkeypatch):
    monkeypatch.delenv("TVM_FFI_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _ensure_cuda_arch_list()
    assert "TVM_FFI_CUDA_ARCH_LIST" not in os.environ