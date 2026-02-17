"""Recovery module for handling node failures during inference.

Provides re-routing and re-partitioning capabilities when nodes
drop out of the distributed inference pipeline.
"""

from typing import Optional

from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.partitioner import partition_model, PartitionPlan
from distributed_inference.coordinator.registry import NodeRegistry, NodeState
from distributed_inference.coordinator.router import ActivationRouter
from distributed_inference.common.config import ModelConfig
from distributed_inference.fault_tolerance.checkpointing import CheckpointManager

log = get_logger(__name__)


class RecoveryManager:
    """Manages recovery from node failures during inference.

    When a node dies:
    1. Identifies which layers the dead node held
    2. Re-partitions the model across remaining nodes
    3. Reloads shards on the new assignments
    4. Resumes in-flight inference from the last checkpoint
    """

    def __init__(
        self,
        registry: NodeRegistry,
        router: ActivationRouter,
        checkpoint_manager: CheckpointManager,
        model_config: ModelConfig,
    ):
        self.registry = registry
        self.router = router
        self.checkpoint_manager = checkpoint_manager
        self.model_config = model_config

    def handle_node_failure(
        self,
        dead_node_id: str,
    ) -> Optional[PartitionPlan]:
        """Handle a node failure by re-partitioning and reloading.

        Args:
            dead_node_id: ID of the node that failed.

        Returns:
            New PartitionPlan if re-partitioning succeeded, None if
            insufficient nodes remain.
        """
        log.warning(f"[bold red]Handling failure of node {dead_node_id}[/]")

        # Mark the node as dead
        self.registry.mark_dead(dead_node_id)

        # Get remaining active nodes
        active_nodes = self.registry.get_active_nodes()

        if not active_nodes:
            log.error("No active nodes remaining — cannot recover")
            return None

        log.info(f"Re-partitioning across {len(active_nodes)} remaining nodes")

        # First, unload shards from all active nodes
        for node in active_nodes:
            try:
                stub = self.router._get_stub(node.address)
                from distributed_inference.proto import inference_pb2
                stub.UnloadShard(inference_pb2.Empty(), timeout=30)
            except Exception as e:
                log.warning(f"Failed to unload shard on {node.node_id}: {e}")

        # Re-partition the model
        new_plan = partition_model(
            nodes=active_nodes,
            model_config=self.model_config,
        )

        # Reload shards on new assignments
        for assignment in new_plan.assignments:
            node = self.registry.get_node(assignment.node_id)
            if not node:
                continue

            try:
                self.router.load_shard_on_node(
                    address=node.address,
                    model_name=self.model_config.name,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    has_embedding=assignment.has_embedding,
                    has_lm_head=assignment.has_lm_head,
                    dtype=self.model_config.dtype,
                )

                self.registry.set_node_assignment(
                    node_id=assignment.node_id,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    has_embedding=assignment.has_embedding,
                    has_lm_head=assignment.has_lm_head,
                )

                log.info(
                    f"Reloaded shard on {assignment.node_id}: "
                    f"layers [{assignment.start_layer}, {assignment.end_layer})"
                )

            except Exception as e:
                log.error(
                    f"Failed to reload shard on {assignment.node_id}: {e}"
                )

        log.info("[bold green]Recovery complete — pipeline reconstructed[/]")
        return new_plan
