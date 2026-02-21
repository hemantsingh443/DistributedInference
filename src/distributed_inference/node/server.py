"""gRPC server for the node agent.

Implements the NodeService defined in inference.proto. Handles:
- Loading model shards on coordinator instruction
- Running forward passes on received activations
- Responding to heartbeat health checks
"""

import time
import threading
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

    def __init__(
        self,
        node_id: str,
        device_type: str = "cpu",
        max_cached_requests: int = 8,
        max_cache_tokens_per_request: int = 4096,
        max_concurrent_lanes: int = 4,
    ):
        self.node_id = node_id
        self.device_type = device_type
        self.max_concurrent_lanes = max(1, int(max_concurrent_lanes))
        if max_cached_requests < self.max_concurrent_lanes:
            max_cached_requests = self.max_concurrent_lanes
        self.executor = ShardExecutor(
            device_type=device_type,
            max_cached_requests=max_cached_requests,
            max_cache_tokens_per_request=max_cache_tokens_per_request,
        )
        self._status = inference_pb2.NodeStatus.IDLE
        self._vram_total_mb = 0
        self._status_lock = threading.RLock()
        self._lane_semaphore = threading.BoundedSemaphore(self.max_concurrent_lanes)
        self._active_requests = 0
        self._queue_depth = 0
        self._cancelled_requests: set[str] = set()

    def _set_status(self, status: int) -> None:
        with self._status_lock:
            self._status = status

    def _get_status(self) -> int:
        with self._status_lock:
            return self._status

    def _try_acquire_lane(self, timeout_sec: float = 30.0) -> bool:
        with self._status_lock:
            self._queue_depth += 1
        acquired = self._lane_semaphore.acquire(timeout=timeout_sec)
        with self._status_lock:
            self._queue_depth = max(self._queue_depth - 1, 0)
            if acquired:
                self._active_requests += 1
                self._status = inference_pb2.NodeStatus.BUSY
        return acquired

    def _release_lane(self) -> None:
        with self._status_lock:
            self._active_requests = max(self._active_requests - 1, 0)
            if self._active_requests == 0:
                self._status = (
                    inference_pb2.NodeStatus.READY
                    if self.executor.loaded
                    else inference_pb2.NodeStatus.IDLE
                )
        self._lane_semaphore.release()

    def LoadModelShard(self, request, context):
        """Load model shard as instructed by the coordinator."""
        log.info(
            f"[bold green]LoadModelShard[/] request: "
            f"model={request.model_name}, "
            f"layers=[{request.start_layer}, {request.end_layer}), "
            f"embed={request.has_embedding}, lm_head={request.has_lm_head}"
        )

        self._set_status(inference_pb2.NodeStatus.LOADING)

        try:
            stats = self.executor.load_shard(
                model_name=request.model_name,
                start_layer=request.start_layer,
                end_layer=request.end_layer,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                dtype=request.dtype or "float16",
            )
            self._set_status(inference_pb2.NodeStatus.READY)
            self._vram_total_mb = stats.get("vram_used_mb", 0)

            return self._build_status()

        except Exception as e:
            log.error(f"Failed to load shard: {e}")
            self._set_status(inference_pb2.NodeStatus.ERROR)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return self._build_status()

    def RunForward(self, request, context):
        """Execute forward pass on the loaded shard."""
        start_time = time.time()
        request_id = request.request_id or ""

        if request.reset_cache and request_id:
            with self._status_lock:
                self._cancelled_requests.discard(request_id)

        with self._status_lock:
            if request_id and request_id in self._cancelled_requests:
                if context is not None:
                    context.set_code(grpc.StatusCode.CANCELLED)
                    context.set_details(f"request {request_id} was cancelled")
                return inference_pb2.ActivationData()

        if not self._try_acquire_lane():
            if context is not None:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("node forward lanes are saturated")
            return inference_pb2.ActivationData()

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
                request_id=request_id,
                use_cache=request.use_cache,
                reset_cache=request.reset_cache,
                cache_position=request.cache_position,
                is_prefill=request.is_prefill,
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

            return inference_pb2.ActivationData(
                hidden_states=inference_pb2.TensorData(
                    data=output_bytes,
                    shape=list(output.shape),
                    dtype=str(output.dtype).replace("torch.", ""),
                ),
                attention_mask=request.attention_mask,  # Pass through
                position_ids=request.position_ids,  # Pass through
                request_id=request_id,
                current_layer=self.executor.end_layer,
                use_cache=request.use_cache,
                reset_cache=False,
                cache_position=request.cache_position,
                is_prefill=request.is_prefill,
            )

        except grpc.RpcError:
            raise
        except Exception as e:
            log.error(f"Forward pass failed: {e}")
            self._set_status(inference_pb2.NodeStatus.ERROR)
            if context is not None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            return inference_pb2.ActivationData()
        finally:
            self._release_lane()

    def Heartbeat(self, request, context):
        """Respond to health check with current status."""
        return inference_pb2.HeartbeatResponse(
            status=self._build_status()
        )

    def UnloadShard(self, request, context):
        """Unload the current model shard."""
        log.info("Unloading shard")
        self.executor.unload()
        self._set_status(inference_pb2.NodeStatus.IDLE)
        return self._build_status()

    def ClearRequestCache(self, request, context):
        """Clear cache state for one request or all requests."""
        if request.clear_all:
            self.executor.clear_all_cache()
            with self._status_lock:
                self._cancelled_requests.clear()
        elif request.request_id:
            self.executor.clear_request_cache(request.request_id)
        return inference_pb2.Empty()

    def CancelRequest(self, request, context):
        """Cancel an in-flight request and clear its cache."""
        if request.clear_all:
            self.executor.clear_all_cache()
            with self._status_lock:
                self._cancelled_requests.clear()
            return inference_pb2.Empty()

        if request.request_id:
            self.executor.clear_request_cache(request.request_id)
            with self._status_lock:
                self._cancelled_requests.add(request.request_id)
        return inference_pb2.Empty()

    def _build_status(self) -> inference_pb2.NodeStatus:
        """Build a NodeStatus protobuf message."""
        layer_info = self.executor.get_layer_info()
        assigned = list(range(layer_info["start_layer"], layer_info["end_layer"]))
        with self._status_lock:
            active_requests = self._active_requests
            queue_depth = self._queue_depth
            status = self._status

        return inference_pb2.NodeStatus(
            node_id=self.node_id,
            status=status,
            vram_used_mb=get_vram_usage_mb(),
            vram_total_mb=self._vram_total_mb,
            assigned_layers=assigned,
            load_percent=0.0,
            timestamp_ms=int(time.time() * 1000),
            active_requests=active_requests,
            queue_depth=queue_depth,
            estimated_free_vram_mb=max(
                self._vram_total_mb - get_vram_usage_mb(),
                0,
            ),
        )


def create_node_server(
    node_id: str,
    port: int,
    device_type: str = "cpu",
    max_cached_requests: int = 8,
    max_cache_tokens_per_request: int = 4096,
    max_concurrent_lanes: int = 4,
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

    servicer = NodeServiceImpl(
        node_id=node_id,
        device_type=device_type,
        max_cached_requests=max_cached_requests,
        max_cache_tokens_per_request=max_cache_tokens_per_request,
        max_concurrent_lanes=max_concurrent_lanes,
    )
    inference_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    # Expose servicer for local heartbeat snapshots from NodeAgent.
    setattr(server, "_di_servicer", servicer)

    server.add_insecure_port(f"[::]:{port}")

    log.info(f"Node server configured on port {port}")
    return server
