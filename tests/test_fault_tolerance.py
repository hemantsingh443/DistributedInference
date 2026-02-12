"""Tests for the fault tolerance subsystem."""

import time

import pytest
import torch

from distributed_inference.fault_tolerance.checkpointing import (
    CheckpointManager,
    Checkpoint,
)
from distributed_inference.coordinator.registry import (
    NodeRegistry,
    NodeState,
)


class TestCheckpointManager:
    """Tests for activation checkpoint management."""

    @pytest.fixture
    def manager(self):
        return CheckpointManager(max_checkpoints_per_request=5)

    def test_save_and_retrieve(self, manager):
        """Test saving and retrieving a checkpoint."""
        hidden = torch.randn(1, 10, 2048)
        manager.save("req-1", stage_index=0, hidden_states=hidden)

        cp = manager.get("req-1", 0)
        assert cp is not None
        assert cp.request_id == "req-1"
        assert cp.stage_index == 0
        assert torch.allclose(cp.hidden_states, hidden)

    def test_get_latest(self, manager):
        """Test getting the latest checkpoint."""
        for i in range(3):
            manager.save("req-1", stage_index=i, hidden_states=torch.randn(1, 10, 2048))

        latest = manager.get_latest("req-1")
        assert latest is not None
        assert latest.stage_index == 2

    def test_max_checkpoints_pruning(self, manager):
        """Test that old checkpoints are pruned."""
        for i in range(10):
            manager.save("req-1", stage_index=i, hidden_states=torch.randn(1, 5))

        # Should have at most max_checkpoints_per_request
        assert manager.total_checkpoints <= 5

    def test_clear_request(self, manager):
        """Test clearing checkpoints for a specific request."""
        manager.save("req-1", 0, torch.randn(1, 5))
        manager.save("req-2", 0, torch.randn(1, 5))

        manager.clear_request("req-1")
        assert manager.get("req-1", 0) is None
        assert manager.get("req-2", 0) is not None

    def test_clear_all(self, manager):
        """Test clearing all checkpoints."""
        manager.save("req-1", 0, torch.randn(1, 5))
        manager.save("req-2", 0, torch.randn(1, 5))

        manager.clear_all()
        assert manager.total_checkpoints == 0

    def test_checkpoint_size(self, manager):
        """Test checkpoint size estimation."""
        hidden = torch.randn(1, 100, 2048)  # ~800KB
        manager.save("req-1", 0, hidden_states=hidden)

        cp = manager.get("req-1", 0)
        assert cp.size_mb > 0

    def test_nonexistent_request(self, manager):
        """Test retrieving from nonexistent request."""
        assert manager.get("nonexistent", 0) is None
        assert manager.get_latest("nonexistent") is None


class TestNodeRegistry:
    """Tests for the node registry."""

    @pytest.fixture
    def registry(self):
        return NodeRegistry()

    def test_register_node(self, registry):
        """Test registering a node."""
        node = registry.register("node-1", "localhost:50051", 2048)
        assert node.node_id == "node-1"
        assert registry.node_count == 1

    def test_unregister_node(self, registry):
        """Test unregistering a node."""
        registry.register("node-1", "localhost:50051", 2048)
        removed = registry.unregister("node-1")
        assert removed is not None
        assert registry.node_count == 0

    def test_get_active_nodes(self, registry):
        """Test getting active nodes."""
        registry.register("node-1", "localhost:50051", 2048)
        registry.register("node-2", "localhost:50052", 1024)
        registry.mark_dead("node-2")

        active = registry.get_active_nodes()
        assert len(active) == 1
        assert active[0].node_id == "node-1"

    def test_heartbeat_updates(self, registry):
        """Test heartbeat updates."""
        registry.register("node-1", "localhost:50051", 2048)
        time.sleep(0.1)
        registry.update_heartbeat("node-1", vram_used_mb=512)

        node = registry.get_node("node-1")
        assert node.vram_used_mb == 512
        assert node.missed_heartbeats == 0

    def test_suspect_recovery(self, registry):
        """Test node recovery from suspect state."""
        registry.register("node-1", "localhost:50051", 2048)
        registry.mark_suspect("node-1")

        node = registry.get_node("node-1")
        assert node.state == NodeState.SUSPECT

        registry.update_heartbeat("node-1")
        node = registry.get_node("node-1")
        assert node.state == NodeState.READY

    def test_set_node_assignment(self, registry):
        """Test setting layer assignments."""
        registry.register("node-1", "localhost:50051", 2048)
        registry.set_node_assignment("node-1", 0, 11, has_embedding=True)

        node = registry.get_node("node-1")
        assert node.assigned_layers == (0, 11)
        assert node.has_embedding is True

    def test_on_change_callback(self, registry):
        """Test change callbacks."""
        events = []
        registry.on_change(lambda event, node: events.append(event))

        registry.register("node-1", "localhost:50051", 2048)
        assert "register" in events

    def test_summary(self, registry):
        """Test registry summary."""
        registry.register("node-1", "localhost:50051", 2048)
        summary = registry.summary()
        assert "Nodes: 1" in summary
