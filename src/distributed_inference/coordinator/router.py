"""Activation router — routes intermediate tensors between nodes.

Orchestrates the forward pass across a pipeline of distributed nodes,
handling serialization/deserialization and measuring per-hop latency.
"""

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import grpc
import torch

from distributed_inference.common.logging import get_logger
from distributed_inference.common.serialization import (
    serialize_tensor,
    deserialize_tensor,
)
from distributed_inference.coordinator.scheduler import ExecutionPlan, PipelineStage
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


@dataclass
class HopTrace:
    """Trace record for a single pipeline hop."""
    hop_index: int
    stage: PipelineStage
    output: torch.Tensor
    hop_latency_ms: float


class ActivationRouter:
    """Routes activations through the distributed inference pipeline.

    Manages gRPC connections to nodes and orchestrates the sequential
    forward pass through all pipeline stages.
    """

    def __init__(self, grpc_options=None):
        self._channels = {}  # address -> channel
        self._stubs = {}     # address -> stub
        self._grpc_options = grpc_options or [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]

    def _get_stub(self, address: str) -> inference_pb2_grpc.NodeServiceStub:
        """Get or create a gRPC stub for a node address."""
        if address not in self._stubs:
            channel = grpc.insecure_channel(address, options=self._grpc_options)
            self._channels[address] = channel
            self._stubs[address] = inference_pb2_grpc.NodeServiceStub(channel)
        return self._stubs[address]

    def route_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        execution_plan: ExecutionPlan,
        request_id: str = "",
        use_cache: bool = False,
        reset_cache: bool = False,
        cache_position: Optional[int] = None,
        is_prefill: bool = False,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Route a forward pass through all pipeline stages.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            execution_plan: Ordered execution plan.
            request_id: Unique request identifier for tracing.

        Returns:
            Tuple of (output_logits, per_hop_latency_ms).
        """
        per_hop_latency = []
        current_data: Optional[torch.Tensor] = None
        for trace in self.route_forward_stream(
            input_ids=input_ids,
            attention_mask=attention_mask,
            execution_plan=execution_plan,
            request_id=request_id,
            use_cache=use_cache,
            reset_cache=reset_cache,
            cache_position=cache_position,
            is_prefill=is_prefill,
        ):
            current_data = trace.output
            per_hop_latency.append(trace.hop_latency_ms)

        if current_data is None:
            raise RuntimeError("Execution plan has no stages")
        return current_data, per_hop_latency

    def route_forward_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        execution_plan: ExecutionPlan,
        request_id: str = "",
        use_cache: bool = False,
        reset_cache: bool = False,
        cache_position: Optional[int] = None,
        is_prefill: bool = False,
    ) -> Iterator[HopTrace]:
        """Route a forward pass and emit hop traces as they complete."""
        current_data = input_ids  # Start with input_ids

        for i, stage in enumerate(execution_plan.stages):
            hop_start = time.time()

            log.info(
                f"[cyan]Hop {i}[/]: → {stage.node_id} @ {stage.address} "
                f"layers [{stage.start_layer}, {stage.end_layer})"
            )

            stub = self._get_stub(stage.address)

            hidden_data = serialize_tensor(current_data)
            activation = inference_pb2.ActivationData(
                hidden_states=inference_pb2.TensorData(
                    data=hidden_data,
                    shape=list(current_data.shape),
                    dtype=str(current_data.dtype).replace("torch.", ""),
                ),
                request_id=request_id,
                current_layer=stage.start_layer,
                use_cache=use_cache,
                reset_cache=reset_cache,
                is_prefill=is_prefill,
            )
            if cache_position is not None:
                activation.cache_position = cache_position

            if attention_mask is not None:
                mask_bytes = serialize_tensor(attention_mask)
                activation.attention_mask.CopyFrom(
                    inference_pb2.TensorData(
                        data=mask_bytes,
                        shape=list(attention_mask.shape),
                        dtype=str(attention_mask.dtype).replace("torch.", ""),
                    )
                )

            try:
                response = stub.RunForward(activation, timeout=120)
            except grpc.RpcError as e:
                log.error(
                    f"[bold red]Hop {i} FAILED[/] at {stage.node_id}: "
                    f"{e.code()} - {e.details()}"
                )
                raise RuntimeError(
                    f"Forward pass failed at node {stage.node_id}: {e.details()}"
                ) from e

            current_data = deserialize_tensor(response.hidden_states.data)
            hop_ms = (time.time() - hop_start) * 1000

            log.info(
                f"  Hop {i} complete: output_shape={list(current_data.shape)}, "
                f"time={hop_ms:.1f}ms"
            )

            yield HopTrace(
                hop_index=i,
                stage=stage,
                output=current_data,
                hop_latency_ms=hop_ms,
            )

    def clear_request_cache_on_pipeline(
        self,
        execution_plan: ExecutionPlan,
        request_id: str,
        clear_all: bool = False,
    ) -> None:
        """Clear request cache across all pipeline stages."""
        for stage in execution_plan.stages:
            try:
                stub = self._get_stub(stage.address)
                stub.ClearRequestCache(
                    inference_pb2.CacheControl(
                        request_id=request_id,
                        clear_all=clear_all,
                    ),
                    timeout=10,
                )
            except grpc.RpcError as e:
                log.warning(
                    f"Failed to clear cache on {stage.node_id}: "
                    f"{e.code()} - {e.details()}"
                )

    def cancel_request_on_pipeline(
        self,
        execution_plan: ExecutionPlan,
        request_id: str,
    ) -> None:
        """Request cancellation for an active request across all pipeline stages."""
        for stage in execution_plan.stages:
            try:
                stub = self._get_stub(stage.address)
                if hasattr(stub, "CancelRequest"):
                    stub.CancelRequest(
                        inference_pb2.CacheControl(
                            request_id=request_id,
                            clear_all=False,
                        ),
                        timeout=10,
                    )
                else:
                    stub.ClearRequestCache(
                        inference_pb2.CacheControl(
                            request_id=request_id,
                            clear_all=False,
                        ),
                        timeout=10,
                    )
            except grpc.RpcError as e:
                log.warning(
                    f"Failed to cancel request on {stage.node_id}: "
                    f"{e.code()} - {e.details()}"
                )

    def load_shard_on_node(
        self,
        address: str,
        model_name: str,
        start_layer: int,
        end_layer: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        dtype: str = "float16",
    ) -> inference_pb2.NodeStatus:
        """Instruct a node to load specific model layers.

        Args:
            address: Node's gRPC address.
            model_name: HuggingFace model ID.
            start_layer: First layer (inclusive).
            end_layer: Last layer (exclusive).
            has_embedding: Whether to load embedding layer.
            has_lm_head: Whether to load output head.
            dtype: Weight data type.

        Returns:
            NodeStatus after loading.
        """
        stub = self._get_stub(address)

        assignment = inference_pb2.ShardAssignment(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            has_embedding=has_embedding,
            has_lm_head=has_lm_head,
            dtype=dtype,
        )

        log.info(
            f"Loading shard on {address}: layers [{start_layer}, {end_layer})"
        )

        response = stub.LoadModelShard(assignment, timeout=600)
        return response

    def check_node_health(self, address: str) -> Optional[inference_pb2.NodeStatus]:
        """Send a heartbeat to check if a node is alive.

        Returns:
            NodeStatus if reachable, None if unreachable.
        """
        try:
            stub = self._get_stub(address)
            response = stub.Heartbeat(inference_pb2.Empty(), timeout=5)
            return response.status
        except grpc.RpcError:
            return None

    def unload_shard_on_node(self, address: str) -> Optional[inference_pb2.NodeStatus]:
        """Instruct a node to unload any currently loaded shard."""
        try:
            stub = self._get_stub(address)
            return stub.UnloadShard(inference_pb2.Empty(), timeout=60)
        except grpc.RpcError as e:
            log.warning(
                f"Failed to unload shard on {address}: "
                f"{e.code()} - {e.details()}"
            )
            return None

    def close(self) -> None:
        """Close all gRPC channels."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
        self._stubs.clear()
