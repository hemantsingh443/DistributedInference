"""Tests for compute-aware partitioning behavior."""

from distributed_inference.common.config import ModelConfig
from distributed_inference.coordinator.partitioner import partition_model
from distributed_inference.coordinator.registry import RegisteredNode


def _node(
    node_id: str,
    *,
    vram_mb: int,
    compute_tflops: float,
    bandwidth_mbps: float,
    latency_ms: float,
) -> RegisteredNode:
    return RegisteredNode(
        node_id=node_id,
        address=f"localhost:{50000 + int(node_id[-1])}",
        vram_mb=vram_mb,
        compute_tflops=compute_tflops,
        bandwidth_mbps=bandwidth_mbps,
        effective_bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        device_type="cuda",
        device_name="test-gpu",
    )


def _model() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        dtype="float16",
        num_layers=22,
        hidden_size=2048,
        intermediate_size=5632,
        num_attention_heads=32,
        num_kv_heads=4,
    )


def test_partition_model_is_deterministic() -> None:
    nodes = [
        _node("node-1", vram_mb=4096, compute_tflops=20.0, bandwidth_mbps=2000, latency_ms=2.0),
        _node("node-2", vram_mb=2048, compute_tflops=8.0, bandwidth_mbps=1000, latency_ms=5.0),
        _node("node-3", vram_mb=2048, compute_tflops=6.0, bandwidth_mbps=800, latency_ms=8.0),
    ]
    first = partition_model(nodes, _model())
    second = partition_model(nodes, _model())

    assert [
        (a.node_id, a.start_layer, a.end_layer, a.has_embedding, a.has_lm_head)
        for a in first.assignments
    ] == [
        (a.node_id, a.start_layer, a.end_layer, a.has_embedding, a.has_lm_head)
        for a in second.assignments
    ]


def test_partition_model_respects_memory_fit_margin() -> None:
    nodes = [
        _node("node-1", vram_mb=4096, compute_tflops=20.0, bandwidth_mbps=2000, latency_ms=2.0),
        _node("node-2", vram_mb=3072, compute_tflops=8.0, bandwidth_mbps=1000, latency_ms=5.0),
    ]
    margin = 0.85
    plan = partition_model(nodes, _model(), memory_safety_margin=margin)
    by_id = {n.node_id: n for n in nodes}

    for assignment in plan.assignments:
        allowed = by_id[assignment.node_id].vram_mb * margin
        assert assignment.estimated_memory_mb <= allowed + 1e-6
