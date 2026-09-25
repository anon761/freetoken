# Copyright (c) 2026 FreeToken contributors
# checkpoint/ftw._np_dtype: every bank dtype needs a same-width numpy view -- the TP band
# path slices NVFP4 down banks (fp8 e4m3 block scales) in numpy and raised KeyError on fp8.
from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.checkpoint.ftw import _elsize, _np_dtype


@pytest.mark.parametrize("dt", [
    torch.uint8, torch.int8, torch.int32, torch.int64, torch.float16, torch.bfloat16,
    torch.float8_e4m3fn, torch.float8_e5m2, torch.float32,
])
def test_bank_dtypes_have_a_same_width_numpy_view(dt):
    assert np.dtype(_np_dtype(dt)).itemsize == _elsize(dt)


def test_fp8_band_slice_is_byte_identical():
    t = torch.randn(2, 4, 8).to(torch.float8_e4m3fn)
    arr = np.frombuffer(t.view(torch.uint8).numpy().tobytes(), dtype=_np_dtype(t.dtype)).reshape(t.shape)
    band = torch.frombuffer(bytearray(arr[:, 1:3].copy().tobytes()), dtype=torch.float8_e4m3fn)
    assert torch.equal(band.view(torch.uint8), t[:, 1:3].contiguous().view(-1).view(torch.uint8))
