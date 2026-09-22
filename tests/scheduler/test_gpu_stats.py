from __future__ import annotations

from freetoken.gpu_stats import GpuSample, GpuTelemetry, format_gpu_line


def _sample(index, uuid=None, used=21, total=24, util=88, temp=72, power=315.0):
    return GpuSample(index, uuid, used * (1 << 30), total * (1 << 30), util, temp, power)


def test_format_line_is_compact_and_speaking():
    line = format_gpu_line(2, [_sample(0), _sample(1, util=91, temp=74, power=330.0)])
    assert line == (
        "TP=2 | GPU0 VRAM 21.0/24.0G 88% 72C 315W | GPU1 VRAM 21.0/24.0G 91% 74C 330W"
    )


def test_format_omits_fields_the_driver_did_not_report():
    line = format_gpu_line(1, [GpuSample(0, None, 1 << 30, 2 << 30, None, None, None)])
    assert line == "TP=1 | GPU0 VRAM 1.0/2.0G"


def test_telemetry_filters_by_assigned_uuid():
    samples = [_sample(0, "GPU-aaaa"), _sample(1, "GPU-bbbb")]
    tel = GpuTelemetry(tp_size=2, gpu_assigned=("GPU-bbbb",), sampler=lambda: samples)
    line = tel.line()
    assert line.startswith("TP=2 | ")
    assert "GPU1" in line and "GPU0" not in line


def test_telemetry_filters_by_raw_index():
    samples = [_sample(0), _sample(1), _sample(2)]
    tel = GpuTelemetry(tp_size=1, gpu=("1",), sampler=lambda: samples)
    line = tel.line()
    assert "GPU1" in line and "GPU0" not in line and "GPU2" not in line


def test_telemetry_without_gpu_uses_tp_ordinals():
    samples = [_sample(0), _sample(1), _sample(2)]
    tel = GpuTelemetry(tp_size=2, sampler=lambda: samples)
    line = tel.line()
    assert "GPU0" in line and "GPU1" in line and "GPU2" not in line


def test_telemetry_is_ttl_cached():
    calls = {"n": 0}

    def sampler():
        calls["n"] += 1
        return [_sample(0)]

    clock = {"t": 0.0}
    tel = GpuTelemetry(tp_size=1, ttl_s=1.0, now=lambda: clock["t"], sampler=sampler)
    tel.line()
    tel.line()
    assert calls["n"] == 1
    clock["t"] = 2.0
    tel.line()
    assert calls["n"] == 2


def test_telemetry_empty_when_no_samples():
    tel = GpuTelemetry(tp_size=1, sampler=lambda: [])
    assert tel.line() == ""
