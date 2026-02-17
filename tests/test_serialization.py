"""Tests for tensor serialization utilities."""

import pytest
import torch

from distributed_inference.common.serialization import (
    serialize_tensor,
    deserialize_tensor,
    tensor_to_proto_fields,
    tensor_size_bytes,
    estimate_layer_memory_mb,
)


class TestSerializeTensor:
    """Tests for serialize/deserialize round-trip."""

    def test_roundtrip_float32(self):
        """Test float32 tensor round-trip."""
        original = torch.randn(2, 3, 4)
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data)
        assert torch.allclose(original, recovered)

    def test_roundtrip_float16(self):
        """Test float16 tensor round-trip."""
        original = torch.randn(4, 8).half()
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data)
        assert torch.allclose(original, recovered)

    def test_roundtrip_long(self):
        """Test integer tensor round-trip."""
        original = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data)
        assert torch.equal(original, recovered)

    def test_roundtrip_compressed(self):
        """Test compressed serialization round-trip."""
        original = torch.randn(16, 32)
        data = serialize_tensor(original, compress=True)
        recovered = deserialize_tensor(data, compressed=True)
        assert torch.allclose(original, recovered)

    def test_compression_reduces_size(self):
        """Test that compression reduces data size."""
        # Use a tensor with patterns (compresses better)
        original = torch.zeros(100, 100)
        uncompressed = serialize_tensor(original, compress=False)
        compressed = serialize_tensor(original, compress=True)
        assert len(compressed) < len(uncompressed)

    def test_device_placement(self):
        """Test tensor is placed on correct device."""
        original = torch.randn(2, 3)
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data, device="cpu")
        assert recovered.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gpu_tensor_serialization(self):
        """Test GPU tensor round-trip."""
        original = torch.randn(4, 4, device="cuda")
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data, device="cuda")
        assert recovered.device.type == "cuda"
        assert torch.allclose(original, recovered)

    def test_empty_tensor(self):
        """Test empty tensor round-trip."""
        original = torch.empty(0, 3)
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data)
        assert recovered.shape == (0, 3)

    def test_scalar_tensor(self):
        """Test scalar tensor round-trip."""
        original = torch.tensor(42.0)
        data = serialize_tensor(original)
        recovered = deserialize_tensor(data)
        assert recovered.item() == 42.0


class TestTensorProtoFields:
    """Tests for tensor → proto field conversion."""

    def test_basic_conversion(self):
        """Test conversion to proto fields."""
        tensor = torch.randn(3, 4)
        fields = tensor_to_proto_fields(tensor)
        assert "data" in fields
        assert fields["shape"] == [3, 4]
        assert fields["dtype"] == "float32"

    def test_float16_dtype(self):
        """Test float16 dtype string."""
        tensor = torch.randn(2, 2).half()
        fields = tensor_to_proto_fields(tensor)
        assert fields["dtype"] == "float16"


class TestEstimateLayerMemory:
    """Tests for layer memory estimation."""

    def test_tinyllama_layer(self):
        """Test memory estimate for a TinyLlama-like layer."""
        mem = estimate_layer_memory_mb(
            hidden_size=2048,
            intermediate_size=5632,
            num_heads=32,
            dtype_bytes=2,  # FP16
        )
        # Should be roughly 60-80 MB per layer
        assert 30 < mem < 200

    def test_dtype_scaling(self):
        """Test that FP32 is double FP16."""
        mem_fp16 = estimate_layer_memory_mb(2048, 5632, 32, dtype_bytes=2)
        mem_fp32 = estimate_layer_memory_mb(2048, 5632, 32, dtype_bytes=4)
        assert abs(mem_fp32 / mem_fp16 - 2.0) < 0.01


class TestTensorSizeBytes:
    """Tests for tensor size calculation."""

    def test_size_calculation(self):
        """Test byte size calculation."""
        tensor = torch.randn(10, 20)  # 200 elements * 4 bytes = 800
        assert tensor_size_bytes(tensor) == 800

    def test_float16_size(self):
        """Test float16 size is half of float32."""
        t_f32 = torch.randn(10, 20)
        t_f16 = t_f32.half()
        assert tensor_size_bytes(t_f16) == tensor_size_bytes(t_f32) // 2
