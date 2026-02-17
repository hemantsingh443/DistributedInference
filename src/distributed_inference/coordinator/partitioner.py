"""Compute-aware model partitioning strategy."""

import math
from dataclasses import dataclass
from typing import List, Optional

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
    estimated_latency_score: float = 0.0
    estimated_throughput_score: float = 0.0

    def get_assignment(self, node_id: str) -> Optional[LayerAssignment]:
        """Get the assignment for a specific node."""
        for assignment in self.assignments:
            if assignment.node_id == node_id:
                return assignment
        return None

    def get_ordered_nodes(self) -> List[str]:
        """Get node IDs in pipeline order (by start_layer)."""
        sorted_assignments = sorted(self.assignments, key=lambda a: a.start_layer)
        return [a.node_id for a in sorted_assignments]

    def summary(self) -> str:
        """Human-readable summary of the partition plan."""
        lines = [f"Partition Plan for {self.model_name} ({self.total_layers} layers):"]
        for assignment in sorted(self.assignments, key=lambda item: item.start_layer):
            extras = []
            if assignment.has_embedding:
                extras.append("embed")
            if assignment.has_lm_head:
                extras.append("lm_head")
            extra_str = f" + {', '.join(extras)}" if extras else ""
            lines.append(
                f"  {assignment.node_id}: layers "
                f"[{assignment.start_layer}, {assignment.end_layer})"
                f"{extra_str} (~{assignment.estimated_memory_mb:.0f}MB)"
            )
        if self.estimated_latency_score:
            lines.append(f"  score.latency={self.estimated_latency_score:.3f}")
        if self.estimated_throughput_score:
            lines.append(f"  score.throughput={self.estimated_throughput_score:.3f}")
        return "\n".join(lines)


@dataclass
class _NodeCostProfile:
    node: RegisteredNode
    score: float
    max_layers_fit: int
    layer_budget_mb: float


def partition_model(
    nodes: List[RegisteredNode],
    model_config: ModelConfig,
    *,
    alpha_latency: float = 0.7,
    beta_throughput: float = 0.3,
    default_bandwidth_mbps: float = 1000.0,
    default_latency_ms: float = 5.0,
    memory_safety_margin: float = 0.9,
) -> PartitionPlan:
    """Create a compute-aware partition plan to distribute model layers.

    The algorithm keeps contiguous layer allocation while balancing:
    - memory fit (hard constraint),
    - compute capacity,
    - network quality (bandwidth + latency).
    """
    if not nodes:
        raise ValueError("No nodes available for partitioning")

    total_layers = model_config.num_layers
    dtype_bytes = 2 if model_config.dtype == "float16" else 4

    mem_per_layer = estimate_layer_memory_mb(
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_heads=model_config.num_attention_heads,
        dtype_bytes=dtype_bytes,
    )

    vocab_size = 32000
    embed_memory_mb = (vocab_size * model_config.hidden_size * dtype_bytes) / (1024 * 1024)
    lm_head_memory_mb = embed_memory_mb
    total_model_memory = (
        total_layers * mem_per_layer + embed_memory_mb + lm_head_memory_mb
    )

    profiles = _build_cost_profiles(
        nodes=nodes,
        alpha_latency=alpha_latency,
        beta_throughput=beta_throughput,
        default_bandwidth_mbps=default_bandwidth_mbps,
        default_latency_ms=default_latency_ms,
        mem_per_layer=mem_per_layer,
        memory_safety_margin=memory_safety_margin,
    )
    if not profiles:
        raise ValueError("No feasible nodes: every node failed minimum layer memory fit")

    # Deterministic pipeline ordering: highest score first, then node_id.
    ordered_profiles = sorted(
        profiles,
        key=lambda p: (-p.score, p.node.node_id),
    )

    num_nodes = len(ordered_profiles)
    layer_counts = [0] * num_nodes
    residual_capacity = []
    for idx, profile in enumerate(ordered_profiles):
        reserve_mb = 0.0
        if idx == 0:
            reserve_mb += embed_memory_mb
        if idx == (num_nodes - 1):
            reserve_mb += lm_head_memory_mb
        capacity_mb = profile.layer_budget_mb - reserve_mb
        max_fit = int(math.floor(capacity_mb / max(mem_per_layer, 1e-6)))
        residual_capacity.append(max(0, max_fit))
    max_fit_limits = residual_capacity.copy()
    scores = [max(profile.score, 1e-6) for profile in ordered_profiles]

    total_capacity = sum(residual_capacity)
    if total_capacity < total_layers:
        raise ValueError(
            "Insufficient feasible memory for layers: "
            f"capacity={total_capacity}, required={total_layers}. "
            "Cluster does not have enough aggregate VRAM for this model."
        )

    remaining = total_layers
    if total_layers >= num_nodes:
        for idx in range(num_nodes):
            if residual_capacity[idx] <= 0:
                raise ValueError(
                    f"Node {ordered_profiles[idx].node.node_id} "
                    "cannot host minimum one layer"
                )
            layer_counts[idx] = 1
            residual_capacity[idx] -= 1
            remaining -= 1

    # Base proportional assignment.
    score_sum = sum(scores)
    for idx in range(num_nodes):
        desired = int(math.floor((scores[idx] / score_sum) * remaining))
        assign = min(desired, residual_capacity[idx])
        layer_counts[idx] += assign
        residual_capacity[idx] -= assign

    remaining = total_layers - sum(layer_counts)
    while remaining > 0:
        best_idx = max(
            range(num_nodes),
            key=lambda i: (residual_capacity[i] > 0, scores[i], -i),
        )
        if residual_capacity[best_idx] <= 0:
            raise ValueError("No residual capacity left while assigning layers")
        layer_counts[best_idx] += 1
        residual_capacity[best_idx] -= 1
        remaining -= 1

    # Ensure first/last assignment are not empty for embedding/lm_head ownership.
    _ensure_edge_assignments_non_empty(layer_counts, max_fit_limits)

    assignments: List[LayerAssignment] = []
    current_layer = 0
    for index, (profile, num_layers) in enumerate(zip(ordered_profiles, layer_counts)):
        if num_layers <= 0:
            continue

        is_first = index == 0
        is_last = index == num_nodes - 1
        start = current_layer
        end = current_layer + num_layers

        estimated_mem = (num_layers * mem_per_layer)
        if is_first:
            estimated_mem += embed_memory_mb
        if is_last:
            estimated_mem += lm_head_memory_mb

        assignments.append(
            LayerAssignment(
                node_id=profile.node.node_id,
                start_layer=start,
                end_layer=end,
                has_embedding=is_first,
                has_lm_head=is_last,
                estimated_memory_mb=estimated_mem,
            )
        )
        current_layer = end

    if current_layer != total_layers:
        raise ValueError(
            f"Partitioning bug: assigned {current_layer}/{total_layers} layers"
        )

    plan = PartitionPlan(
        assignments=assignments,
        model_name=model_config.name,
        total_layers=total_layers,
        total_estimated_memory_mb=total_model_memory,
        estimated_latency_score=sum(1.0 / max(p.score, 1e-6) for p in ordered_profiles),
        estimated_throughput_score=sum(p.score for p in ordered_profiles),
    )
    log.info(f"\n{plan.summary()}")
    return plan


def _build_cost_profiles(
    *,
    nodes: List[RegisteredNode],
    alpha_latency: float,
    beta_throughput: float,
    default_bandwidth_mbps: float,
    default_latency_ms: float,
    mem_per_layer: float,
    memory_safety_margin: float,
) -> List[_NodeCostProfile]:
    """Create normalized cost profiles and memory ceilings for each node."""
    alpha = max(0.0, alpha_latency)
    beta = max(0.0, beta_throughput)
    denom = alpha + beta if (alpha + beta) > 0 else 1.0
    alpha /= denom
    beta /= denom

    compute_vals = [max(n.compute_tflops, 0.1) for n in nodes]
    bandwidth_vals = [
        max(
            n.effective_bandwidth_mbps or n.bandwidth_mbps or default_bandwidth_mbps,
            1.0,
        )
        for n in nodes
    ]
    latency_vals = [max(n.latency_ms or default_latency_ms, 0.1) for n in nodes]
    vram_vals = [max(n.vram_mb, 1) for n in nodes]

    cmax = max(compute_vals)
    bmax = max(bandwidth_vals)
    lmax = max(latency_vals)
    vmax = max(vram_vals)

    profiles: List[_NodeCostProfile] = []
    for idx, node in enumerate(nodes):
        compute_norm = compute_vals[idx] / cmax
        bandwidth_norm = bandwidth_vals[idx] / bmax
        latency_norm = 1.0 - (latency_vals[idx] / lmax)
        vram_norm = vram_vals[idx] / vmax

        latency_score = 0.7 * compute_norm + 0.2 * bandwidth_norm + 0.1 * latency_norm
        throughput_score = 0.5 * compute_norm + 0.3 * bandwidth_norm + 0.2 * vram_norm
        score = (alpha * latency_score) + (beta * throughput_score)
        score = max(score, 1e-6)

        layer_budget_mb = max(node.vram_mb * memory_safety_margin, 0.0)
        max_layers_fit = int(math.floor(layer_budget_mb / max(mem_per_layer, 1e-6)))
        if max_layers_fit <= 0:
            continue

        profiles.append(
            _NodeCostProfile(
                node=node,
                score=score,
                max_layers_fit=max_layers_fit,
                layer_budget_mb=layer_budget_mb,
            )
        )
    return profiles


def _ensure_edge_assignments_non_empty(
    layer_counts: List[int],
    max_fit_limits: List[int],
) -> None:
    """Ensure first and last node have at least one layer each."""
    if len(layer_counts) == 1:
        if layer_counts[0] <= 0:
            layer_counts[0] = 1
        return

    first_idx = 0
    last_idx = len(layer_counts) - 1
    if layer_counts[first_idx] <= 0:
        donor_idx = _find_donor(layer_counts, exclude={first_idx})
        layer_counts[donor_idx] -= 1
        layer_counts[first_idx] += 1
    if layer_counts[last_idx] <= 0:
        donor_idx = _find_donor(layer_counts, exclude={last_idx})
        layer_counts[donor_idx] -= 1
        layer_counts[last_idx] += 1

    # Validate we did not exceed fit bounds for edges when donating.
    for idx, layers in enumerate(layer_counts):
        if layers > max_fit_limits[idx]:
            raise ValueError(
                f"Node index {idx} exceeded memory fit constraints "
                "during edge assignment"
            )


def _find_donor(layer_counts: List[int], exclude: set[int]) -> int:
    candidates = [i for i, c in enumerate(layer_counts) if i not in exclude and c > 1]
    if not candidates:
        candidates = [i for i, c in enumerate(layer_counts) if i not in exclude and c > 0]
    if not candidates:
        raise ValueError("Unable to rebalance edge assignments: no donor node available")
    return max(candidates, key=lambda i: layer_counts[i])
