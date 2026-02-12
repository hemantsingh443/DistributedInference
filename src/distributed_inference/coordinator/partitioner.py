"""Model partitioning strategy — layer-wise sharding across nodes.

Implements a greedy bin-packing algorithm to assign contiguous transformer
layers to nodes proportional to their available VRAM. Handles special
components (embedding layer for the first node, lm_head for the last).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from distributed_inference.common.config import ModelConfig
from distributed_inference.common.logging import get_logger
from distributed_inference.common.serialization import estimate_layer_memory_mb
from distributed_inference.coordinator.registry import RegisteredNode

log = get_logger(__name__)


@dataclass
class LayerAssignment:
    """Describes which layers are assigned to a specific node."""
    node_id: str
    start_layer: int
    end_layer: int  # exclusive
    has_embedding: bool = False
    has_lm_head: bool = False
    estimated_memory_mb: float = 0.0


@dataclass
class PartitionPlan:
    """Complete partition plan for distributing a model across nodes."""
    assignments: List[LayerAssignment]
    model_name: str
    total_layers: int
    total_estimated_memory_mb: float = 0.0

    def get_assignment(self, node_id: str) -> Optional[LayerAssignment]:
        """Get the assignment for a specific node."""
        for a in self.assignments:
            if a.node_id == node_id:
                return a
        return None

    def get_ordered_nodes(self) -> List[str]:
        """Get node IDs in pipeline order (by start_layer)."""
        sorted_assignments = sorted(self.assignments, key=lambda a: a.start_layer)
        return [a.node_id for a in sorted_assignments]

    def summary(self) -> str:
        """Human-readable summary of the partition plan."""
        lines = [f"Partition Plan for {self.model_name} ({self.total_layers} layers):"]
        for a in sorted(self.assignments, key=lambda x: x.start_layer):
            extras = []
            if a.has_embedding:
                extras.append("embed")
            if a.has_lm_head:
                extras.append("lm_head")
            extra_str = f" + {', '.join(extras)}" if extras else ""
            lines.append(
                f"  {a.node_id}: layers [{a.start_layer}, {a.end_layer})"
                f"{extra_str} (~{a.estimated_memory_mb:.0f}MB)"
            )
        return "\n".join(lines)


def partition_model(
    nodes: List[RegisteredNode],
    model_config: ModelConfig,
) -> PartitionPlan:
    """Create a partition plan to distribute model layers across nodes.

    Uses a VRAM-proportional greedy algorithm:
    1. Estimate memory per layer
    2. Calculate each node's share proportional to its VRAM
    3. Assign contiguous layers to each node
    4. First node gets embedding, last node gets lm_head

    Args:
        nodes: List of active nodes sorted by registration order.
        model_config: Model configuration with layer counts.

    Returns:
        PartitionPlan with layer assignments for each node.

    Raises:
        ValueError: If no nodes available or insufficient VRAM.
    """
    if not nodes:
        raise ValueError("No nodes available for partitioning")

    total_layers = model_config.num_layers
    dtype_bytes = 2 if model_config.dtype == "float16" else 4

    # Estimate memory per layer
    mem_per_layer = estimate_layer_memory_mb(
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_heads=model_config.num_attention_heads,
        dtype_bytes=dtype_bytes,
    )

    # Estimate embedding and lm_head memory
    # Embedding: vocab_size * hidden_size * dtype_bytes
    # Using TinyLlama vocab_size = 32000
    vocab_size = 32000
    embed_memory_mb = (vocab_size * model_config.hidden_size * dtype_bytes) / (1024 * 1024)
    lm_head_memory_mb = embed_memory_mb  # Same size as embedding (weight tying)

    total_model_memory = (
        total_layers * mem_per_layer + embed_memory_mb + lm_head_memory_mb
    )

    log.info(
        f"Model memory estimate: {total_model_memory:.0f}MB total, "
        f"{mem_per_layer:.1f}MB/layer, "
        f"embed={embed_memory_mb:.0f}MB, lm_head={lm_head_memory_mb:.0f}MB"
    )

    # Calculate total available VRAM
    total_vram = sum(n.vram_mb for n in nodes)
    log.info(f"Total VRAM across {len(nodes)} nodes: {total_vram}MB")

    if total_vram < total_model_memory * 0.8:  # Allow some margin
        log.warning(
            f"Total VRAM ({total_vram}MB) may be insufficient for "
            f"model ({total_model_memory:.0f}MB)"
        )

    # Sort nodes by VRAM (largest first for better packing)
    sorted_nodes = sorted(nodes, key=lambda n: n.vram_mb, reverse=True)

    # Calculate proportional layer allocation
    assignments = []
    current_layer = 0

    for i, node in enumerate(sorted_nodes):
        is_first = (i == 0)
        is_last = (i == len(sorted_nodes) - 1)

        # Calculate this node's share of layers proportional to its VRAM
        # Account for extra memory needed for embedding / lm_head
        available_vram = node.vram_mb
        if is_first:
            available_vram -= embed_memory_mb
        if is_last:
            available_vram -= lm_head_memory_mb

        if is_last:
            # Last node gets all remaining layers
            num_layers = total_layers - current_layer
        else:
            # Proportional allocation based on available VRAM
            vram_fraction = available_vram / max(total_vram, 1)
            num_layers = max(1, round(total_layers * vram_fraction))
            # Don't exceed remaining layers
            num_layers = min(num_layers, total_layers - current_layer - (len(sorted_nodes) - i - 1))
            num_layers = max(1, num_layers)

        start = current_layer
        end = current_layer + num_layers

        estimated_mem = num_layers * mem_per_layer
        if is_first:
            estimated_mem += embed_memory_mb
        if is_last:
            estimated_mem += lm_head_memory_mb

        assignment = LayerAssignment(
            node_id=node.node_id,
            start_layer=start,
            end_layer=end,
            has_embedding=is_first,
            has_lm_head=is_last,
            estimated_memory_mb=estimated_mem,
        )
        assignments.append(assignment)
        current_layer = end

    plan = PartitionPlan(
        assignments=assignments,
        model_name=model_config.name,
        total_layers=total_layers,
        total_estimated_memory_mb=total_model_memory,
    )

    log.info(f"\n{plan.summary()}")
    return plan
