"""Tests for the node executor (requires model download — marked slow)."""

import pytest
import torch

# These tests require downloading TinyLlama – skip by default
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # Skip on CI
    reason="Requires model download and significant resources"
)


class TestShardExecutor:
    """Tests for the ShardExecutor — run with pytest -k test_node_executor."""

    @pytest.fixture
    def executor(self):
        from distributed_inference.node.executor import ShardExecutor
        return ShardExecutor(device_type="cpu")

    def test_executor_init(self, executor):
        """Test executor initializes correctly."""
        assert not executor.loaded
        assert executor.start_layer == 0
        assert executor.end_layer == 0

    def test_get_layer_info_unloaded(self, executor):
        """Test layer info when no shard is loaded."""
        info = executor.get_layer_info()
        assert info["loaded"] is False
        assert info["num_layers"] == 0

    def test_forward_without_load_raises(self, executor):
        """Test forward pass without loading raises error."""
        with pytest.raises(RuntimeError, match="No shard loaded"):
            executor.forward(torch.randn(1, 10, 2048))

    def test_unload(self, executor):
        """Test unload cleans up state."""
        executor.unload()
        assert not executor.loaded
