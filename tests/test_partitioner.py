"""Tests for the model partitioner."""

import pytest

from distributed_inference.common.config import ModelConfig
from distributed_inference.coordinator.partitioner import partition_model, PartitionPlan
from distributed_inference.coordinator.registry import RegisteredNode


def make_node(node_id: str, vram_mb: int) -> RegisteredNode:
    """Helper to create a mock registered node."""
    return RegisteredNode(
        node_id=node_id,
        address=f"localhost:{50051 + int(node_id[-1])}",
        vram_mb=vram_mb,
        compute_tflops=1.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test-device",
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        name="test-model",
        num_layers=22,
        hidden_size=2048,
        intermediate_size=5632,
        num_attention_heads=32,
        num_kv_heads=4,
    )


class TestPartitionModel:
    """Tests for the model partitioning algorithm."""

    def test_single_node(self, model_config):
        """Single node gets all layers."""
        nodes = [make_node("node-0", 4096)]
        plan = partition_model(nodes, model_config)

        assert len(plan.assignments) == 1
        a = plan.assignments[0]
        assert a.start_layer == 0
        assert a.end_layer == 22
        assert a.has_embedding is True
        assert a.has_lm_head is True

    def test_two_equal_nodes(self, model_config):
        """Two equal nodes split layers roughly evenly."""
        nodes = [make_node("node-0", 2048), make_node("node-1", 2048)]
        plan = partition_model(nodes, model_config)

        assert len(plan.assignments) == 2
        total_layers = sum(
            a.end_layer - a.start_layer for a in plan.assignments
        )
        assert total_layers == 22

        # First node should have embedding
        first = [a for a in plan.assignments if a.has_embedding][0]
        assert first is not None

        # Last node should have lm_head
        last = [a for a in plan.assignments if a.has_lm_head][0]
        assert last is not None

    def test_three_heterogeneous_nodes(self, model_config):
        """Three nodes with different VRAM get proportional layers."""
        nodes = [
            make_node("node-0", 512),
            make_node("node-1", 1024),
            make_node("node-2", 2048),
        ]
        plan = partition_model(nodes, model_config)

        assert len(plan.assignments) == 3
        total_layers = sum(
            a.end_layer - a.start_layer for a in plan.assignments
        )
        assert total_layers == 22

        # Each node should have at least 1 layer
        for a in plan.assignments:
            assert a.end_layer - a.start_layer >= 1

    def test_contiguous_layers(self, model_config):
        """All layer ranges should be contiguous with no gaps."""
        nodes = [
            make_node("node-0", 1024),
            make_node("node-1", 1024),
            make_node("node-2", 1024),
        ]
        plan = partition_model(nodes, model_config)

        sorted_assignments = sorted(plan.assignments, key=lambda a: a.start_layer)
        for i in range(1, len(sorted_assignments)):
            assert sorted_assignments[i].start_layer == sorted_assignments[i-1].end_layer

        assert sorted_assignments[0].start_layer == 0
        assert sorted_assignments[-1].end_layer == 22

    def test_no_nodes_raises(self, model_config):
        """Should raise ValueError with no nodes."""
        with pytest.raises(ValueError, match="No nodes available"):
            partition_model([], model_config)

    def test_ordered_nodes(self, model_config):
        """get_ordered_nodes returns nodes in pipeline order."""
        nodes = [
            make_node("node-0", 1024),
            make_node("node-1", 2048),
        ]
        plan = partition_model(nodes, model_config)
        ordered = plan.get_ordered_nodes()
        assert len(ordered) == 2

    def test_plan_summary(self, model_config):
        """Partition plan summary is human-readable."""
        nodes = [make_node("node-0", 4096)]
        plan = partition_model(nodes, model_config)
        summary = plan.summary()
        assert "test-model" in summary
        assert "22 layers" in summary
