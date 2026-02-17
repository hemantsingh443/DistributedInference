"""Bandwidth-aware scheduler for distributed inference pipeline.

Determines execution order, estimates latency, and manages pipeline
scheduling of inference requests across distributed nodes.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.partitioner import PartitionPlan, LayerAssignment
from distributed_inference.coordinator.registry import NodeRegistry, RegisteredNode

log = get_logger(__name__)


@dataclass
class PipelineStage:
    """A single stage in the inference pipeline."""
    node_id: str
    address: str
    start_layer: int
    end_layer: int
    has_embedding: bool
    has_lm_head: bool
    estimated_compute_ms: float = 0.0
    estimated_transfer_ms: float = 0.0


@dataclass
class ExecutionPlan:
    """Ordered plan for executing a forward pass across the pipeline."""
    stages: List[PipelineStage]
    estimated_total_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def summary(self) -> str:
        lines = [f"Execution Plan ({self.num_stages} stages, ~{self.estimated_total_ms:.0f}ms total):"]
        for i, stage in enumerate(self.stages):
            lines.append(
                f"  Stage {i}: {stage.node_id} @ {stage.address} "
                f"layers [{stage.start_layer}, {stage.end_layer}) "
                f"compute={stage.estimated_compute_ms:.0f}ms "
                f"transfer={stage.estimated_transfer_ms:.0f}ms"
            )
        return "\n".join(lines)


class Scheduler:
    """Plans and optimizes execution of distributed inference pipelines.

    Given a partition plan and node registry, creates an ordered execution
    plan that minimizes latency by considering compute time and network
    transfer overhead.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        # Rough estimates for scheduling
        ms_per_layer: float = 15.0,  # Avg compute time per transformer layer
        transfer_overhead_ms: float = 10.0,  # Fixed overhead per hop
        mb_per_activation: float = 8.0,  # Typical activation size in MB
        bandwidth_mbps_default: float = 1000.0,  # Default bandwidth
        latency_ms_default: float = 5.0,
        reference_tflops: float = 10.0,
    ):
        self.registry = registry
        self.ms_per_layer = ms_per_layer
        self.transfer_overhead_ms = transfer_overhead_ms
        self.mb_per_activation = mb_per_activation
        self.bandwidth_mbps_default = bandwidth_mbps_default
        self.latency_ms_default = latency_ms_default
        self.reference_tflops = reference_tflops

    def create_execution_plan(
        self,
        partition_plan: PartitionPlan,
    ) -> ExecutionPlan:
        """Create an execution plan from a partition plan.

        Orders stages by layer index (pipeline order) and estimates
        latency for each stage including compute and network transfer.

        Args:
            partition_plan: The model partition plan.

        Returns:
            Ordered ExecutionPlan with timing estimates.
        """
        # Sort assignments by start_layer to get pipeline order
        sorted_assignments = sorted(
            partition_plan.assignments,
            key=lambda a: a.start_layer,
        )

        stages = []
        total_ms = 0.0

        for i, assignment in enumerate(sorted_assignments):
            node = self.registry.get_node(assignment.node_id)
            address = node.address if node else "unknown"

            # Estimate compute time
            num_layers = assignment.end_layer - assignment.start_layer
            compute_tflops = max(node.compute_tflops if node else 0.1, 0.1)
            compute_scale = self.reference_tflops / compute_tflops
            compute_ms = num_layers * self.ms_per_layer * compute_scale

            # Estimate transfer time (except for the last stage)
            transfer_ms = 0.0
            if i < len(sorted_assignments) - 1:
                bandwidth = (
                    node.effective_bandwidth_mbps
                    if node and node.effective_bandwidth_mbps > 0
                    else (node.bandwidth_mbps if node else self.bandwidth_mbps_default)
                )
                latency_ms = (
                    node.latency_ms
                    if node and node.latency_ms > 0
                    else self.latency_ms_default
                )
                transfer_ms = (
                    latency_ms +
                    self.transfer_overhead_ms +
                    (self.mb_per_activation / bandwidth) * 1000  # Convert to ms
                )

            stage = PipelineStage(
                node_id=assignment.node_id,
                address=address,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                has_embedding=assignment.has_embedding,
                has_lm_head=assignment.has_lm_head,
                estimated_compute_ms=compute_ms,
                estimated_transfer_ms=transfer_ms,
            )
            stages.append(stage)
            total_ms += compute_ms + transfer_ms

        plan = ExecutionPlan(
            stages=stages,
            estimated_total_ms=total_ms,
        )

        log.info(f"\n{plan.summary()}")
        return plan

    def estimate_throughput(
        self,
        execution_plan: ExecutionPlan,
        sequence_length: int = 128,
    ) -> dict:
        """Estimate throughput metrics for the execution plan.

        Args:
            execution_plan: The execution plan to evaluate.
            sequence_length: Average sequence length for estimation.

        Returns:
            Dict with estimated metrics.
        """
        total_compute = sum(s.estimated_compute_ms for s in execution_plan.stages)
        total_transfer = sum(s.estimated_transfer_ms for s in execution_plan.stages)
        total_time = total_compute + total_transfer

        # For autoregressive generation, each token requires a full pass
        tokens_per_second = 1000.0 / total_time if total_time > 0 else 0

        return {
            "total_latency_ms": total_time,
            "compute_ms": total_compute,
            "transfer_ms": total_transfer,
            "compute_fraction": total_compute / max(total_time, 1),
            "communication_fraction": total_transfer / max(total_time, 1),
            "estimated_tokens_per_second": tokens_per_second,
        }
