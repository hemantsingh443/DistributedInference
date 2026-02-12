"""gRPC server for the node agent.

Implements the NodeService defined in inference.proto. Handles:
- Loading model shards on coordinator instruction
- Running forward passes on received activations
- Responding to heartbeat health checks
"""

import time
from concurrent import futures

import grpc
import torch

from distributed_inference.common.logging import get_logger
from distributed_inference.common.serialization import (
    serialize_tensor,
    deserialize_tensor,
)
from distributed_inference.node.executor import ShardExecutor
from distributed_inference.node.resources import get_vram_usage_mb
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


class NodeServiceImpl(inference_pb2_grpc.NodeServiceServicer):
    """Implementation of the NodeService gRPC service.

    Each node runs one instance of this server, which manages a
    ShardExecutor for running forward passes on assigned model layers.
    """

    def __init__(self, node_id: str, device_type: str = "cpu"):
        self.node_id = node_id
        self.device_type = device_type
        self.executor = ShardExecutor(device_type=device_type)
        self._status = inference_pb2.NodeStatus.IDLE
        self._vram_total_mb = 0

    def LoadModelShard(self, request, context):
        """Load model shard as instructed by the coordinator."""
        log.info(
            f"[bold green]LoadModelShard[/] request: "
            f"model={request.model_name}, "
            f"layers=[{request.start_layer}, {request.end_layer}), "
            f"embed={request.has_embedding}, lm_head={request.has_lm_head}"
        )

        self._status = inference_pb2.NodeStatus.LOADING

        try:
            stats = self.executor.load_shard(
                model_name=request.model_name,
                start_layer=request.start_layer,
                end_layer=request.end_layer,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                dtype=request.dtype or "float16",
            )
            self._status = inference_pb2.NodeStatus.READY
            self._vram_total_mb = stats.get("vram_used_mb", 0)

            return self._build_status()

        except Exception as e:
            log.error(f"Failed to load shard: {e}")
            self._status = inference_pb2.NodeStatus.ERROR
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return self._build_status()

    def RunForward(self, request, context):
        """Execute forward pass on the loaded shard."""
        start_time = time.time()
        self._status = inference_pb2.NodeStatus.BUSY

        try:
            # Deserialize input tensors
            hidden_states = deserialize_tensor(
                request.hidden_states.data,
                device=self.device_type,
            )

            attention_mask = None
            if request.attention_mask.data:
                attention_mask = deserialize_tensor(
                    request.attention_mask.data,
                    device=self.device_type,
                )

            position_ids = None
            if request.position_ids.data:
                position_ids = deserialize_tensor(
                    request.position_ids.data,
                    device=self.device_type,
                )

            # Run forward pass
            output = self.executor.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Serialize output
            output_bytes = serialize_tensor(output)
            elapsed_ms = (time.time() - start_time) * 1000

            log.info(
                f"Forward pass complete: "
                f"input_shape={list(hidden_states.shape)} → "
                f"output_shape={list(output.shape)}, "
                f"time={elapsed_ms:.1f}ms"
            )

            self._status = inference_pb2.NodeStatus.READY

            return inference_pb2.ActivationData(
                hidden_states=inference_pb2.TensorData(
                    data=output_bytes,
                    shape=list(output.shape),
                    dtype=str(output.dtype).replace("torch.", ""),
                ),
                attention_mask=request.attention_mask,  # Pass through
                position_ids=request.position_ids,  # Pass through
                request_id=request.request_id,
                current_layer=self.executor.end_layer,
            )

        except Exception as e:
            log.error(f"Forward pass failed: {e}")
            self._status = inference_pb2.NodeStatus.ERROR
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.ActivationData()

    def Heartbeat(self, request, context):
        """Respond to health check with current status."""
        return inference_pb2.HeartbeatResponse(
            status=self._build_status()
        )

    def UnloadShard(self, request, context):
        """Unload the current model shard."""
        log.info("Unloading shard")
        self.executor.unload()
        self._status = inference_pb2.NodeStatus.IDLE
        return self._build_status()

    def _build_status(self) -> inference_pb2.NodeStatus:
        """Build a NodeStatus protobuf message."""
        layer_info = self.executor.get_layer_info()
        assigned = list(range(layer_info["start_layer"], layer_info["end_layer"]))

        return inference_pb2.NodeStatus(
            node_id=self.node_id,
            status=self._status,
            vram_used_mb=get_vram_usage_mb(),
            vram_total_mb=self._vram_total_mb,
            assigned_layers=assigned,
            load_percent=0.0,
            timestamp_ms=int(time.time() * 1000),
        )


def create_node_server(
    node_id: str,
    port: int,
    device_type: str = "cpu",
    max_workers: int = 4,
) -> grpc.Server:
    """Create and configure a gRPC server for the node.

    Args:
        node_id: Unique identifier for this node.
        port: Port to listen on.
        device_type: Device type for inference ("cuda" or "cpu").
        max_workers: Max thread pool workers for gRPC.

    Returns:
        Configured (but not yet started) gRPC server.
    """
    # Set large message limits for tensor transfer
    options = [
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )

    servicer = NodeServiceImpl(node_id=node_id, device_type=device_type)
    inference_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")

    log.info(f"Node server configured on port {port}")
    return server
