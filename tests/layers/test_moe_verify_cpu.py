"""--moe-verify-cpu dispatch: only the MTP verify's MoE runs on the CPU executor.

``OffloadMoELayer._cpu_experts_for`` decides per batch whether routed experts use the CPU
executor (RAM banks) instead of the GPU offload/PCIe path. The real forward needs the MoE
cache and kernels, but the decision itself is pure logic, so it is unit-tested here.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.layers.moe import OffloadMoELayer


def _layer(layer_id: int = 3, *, verify_cpu: bool = False, cpu_layers=()):
    cache = SimpleNamespace(
        verify_cpu=verify_cpu,
        is_cpu_layer=lambda lid: lid in set(cpu_layers),
    )
    return SimpleNamespace(layer_id=layer_id, offload_cache=cache)


def _batch(*, verify: bool):
    return SimpleNamespace(mtp_verify=verify)


def test_verify_cpu_routes_only_the_verify_to_cpu():
    layer = _layer(verify_cpu=True)
    assert OffloadMoELayer._cpu_experts_for(layer, _batch(verify=True)) is True
    assert OffloadMoELayer._cpu_experts_for(layer, _batch(verify=False)) is False


def test_verify_cpu_off_leaves_verify_on_gpu():
    layer = _layer(verify_cpu=False)
    assert OffloadMoELayer._cpu_experts_for(layer, _batch(verify=True)) is False


def test_cpu_assigned_layer_always_uses_cpu():
    layer = _layer(layer_id=7, cpu_layers=(7,))
    assert OffloadMoELayer._cpu_experts_for(layer, _batch(verify=False)) is True
    other = _layer(layer_id=8, cpu_layers=(7,), verify_cpu=False)
    assert OffloadMoELayer._cpu_experts_for(other, _batch(verify=True)) is False