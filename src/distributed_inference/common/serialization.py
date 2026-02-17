"""Tensor serialization utilities for gRPC transport.

Converts PyTorch tensors to/from bytes for transmission via Protobuf messages.
Supports optional compression to reduce network overhead.
"""

import io
import zlib
from typing import Optional

import torch


def serialize_tensor(tensor: torch.Tensor, compress: bool = False) -> bytes:
    """Serialize a PyTorch tensor to bytes.

    Uses torch.save into a BytesIO buffer for lossless, dtype-preserving
    serialization. Optionally applies zlib compression.

    Args:
        tensor: The tensor to serialize. Will be moved to CPU if on GPU.
        compress: Whether to apply zlib compression.

    Returns:
        Raw bytes representing the serialized tensor.
    """
    buffer = io.BytesIO()
    # Always save as CPU tensor to avoid device mismatch on deserialization
    torch.save(tensor.cpu(), buffer)
    data = buffer.getvalue()

    if compress:
        data = zlib.compress(data, level=1)  # Fast compression

    return data


def deserialize_tensor(
    data: bytes,
    device: Optional[str] = None,
    compressed: bool = False,
) -> torch.Tensor:
    """Deserialize bytes back into a PyTorch tensor.

    Args:
        data: Raw bytes from serialize_tensor.
        device: Target device ('cuda', 'cpu', or None for CPU default).
        compressed: Whether the data was zlib-compressed.

    Returns:
        Reconstructed PyTorch tensor on the specified device.
    """
    if compressed:
        data = zlib.decompress(data)

    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location="cpu", weights_only=True)

    if device and device != "cpu":
        tensor = tensor.to(device)

    return tensor


def tensor_to_proto_fields(tensor: torch.Tensor) -> dict:
    """Convert a tensor to fields suitable for a TensorData protobuf message.

    Returns:
        Dict with keys: 'data' (bytes), 'shape' (list[int]), 'dtype' (str)
    """
    return {
        "data": serialize_tensor(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Estimate the serialized size of a tensor in bytes (uncompressed)."""
    return tensor.nelement() * tensor.element_size()


def estimate_layer_memory_mb(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype_bytes: int = 2,  # FP16
) -> float:
    """Estimate memory usage of a single transformer layer in MB.

    Accounts for attention weights (Q, K, V, O) and FFN weights.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        num_heads: Number of attention heads.
        dtype_bytes: Bytes per parameter (2 for FP16, 4 for FP32).

    Returns:
        Estimated memory in megabytes.
    """
    # Attention: Q, K, V, O projections = 4 * hidden^2
    attn_params = 4 * hidden_size * hidden_size
    # FFN: gate_proj + up_proj + down_proj = 3 * hidden * intermediate
    ffn_params = 3 * hidden_size * intermediate_size
    # Layer norm: 2 * hidden (negligible but included)
    norm_params = 2 * hidden_size

    total_params = attn_params + ffn_params + norm_params
    return (total_params * dtype_bytes) / (1024 * 1024)
