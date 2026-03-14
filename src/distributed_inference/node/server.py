"""gRPC server for the node agent.

Implements the NodeService defined in inference.proto. Handles:
- Loading model shards on coordinator instruction
- Running forward passes on received activations
- Responding to heartbeat health checks
"""

import time
import asyncio
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
        self._status_lock = asyncio.Lock()
        self._lane_semaphore = asyncio.BoundedSemaphore(self.max_concurrent_lanes)
        self._active_requests = 0
        self._queue_depth = 0
        self._cancelled_requests: set[str] = set()

    async def _set_status(self, status: int) -> None:
        async with self._status_lock:
            self._status = status

    async def _get_status(self) -> int:
        async with self._status_lock:
            return self._status

    async def _try_acquire_lane(self, timeout_sec: float = 30.0) -> bool:
        async with self._status_lock:
            self._queue_depth += 1
        
        try:
            # wait_for raises TimeoutError if it times out
            await asyncio.wait_for(self._lane_semaphore.acquire(), timeout=timeout_sec)
            acquired = True
        except asyncio.TimeoutError:
            acquired = False

        async with self._status_lock:
            self._queue_depth = max(self._queue_depth - 1, 0)
            if acquired:
                self._active_requests += 1
                self._status = inference_pb2.NodeStatus.BUSY
        return acquired

    async def _release_lane(self) -> None:
        async with self._status_lock:
            self._active_requests = max(self._active_requests - 1, 0)
            if self._active_requests == 0:
                self._status = (
                    inference_pb2.NodeStatus.READY
                    if self.executor.loaded
                    else inference_pb2.NodeStatus.IDLE
                )
        self._lane_semaphore.release()

    async def LoadModelShard(self, request, context):
        """Load model shard as instructed by the coordinator."""
        log.info(
            f"[bold green]LoadModelShard[/] request: "
            f"model={request.model_name}, "
            f"layers=[{request.start_layer}, {request.end_layer}), "
            f"embed={request.has_embedding}, lm_head={request.has_lm_head}"
        )

        await self._set_status(inference_pb2.NodeStatus.LOADING)

        try:
            stats = await asyncio.to_thread(
                self.executor.load_shard,
                model_name=request.model_name,
                start_layer=request.start_layer,
                end_layer=request.end_layer,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                dtype=request.dtype or "float16",
                cache_base_path=getattr(request, "cache_base_path", ""),
            )
            await self._set_status(inference_pb2.NodeStatus.READY)
            self._vram_total_mb = stats.get("vram_used_mb", 0)

            return await self._build_status()

        except Exception as e:
            log.error(f"Failed to load shard: {e}")
            await self._set_status(inference_pb2.NodeStatus.ERROR)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return await self._build_status()

    async def RunForward(self, request, context):
        """Execute forward pass on the loaded shard."""
        start_time = time.time()
        request_id = request.request_id or ""

        if request.reset_cache and request_id:
            async with self._status_lock:
                self._cancelled_requests.discard(request_id)

        async with self._status_lock:
            if request_id and request_id in self._cancelled_requests:
                if context is not None:
                    context.set_code(grpc.StatusCode.CANCELLED)
                    context.set_details(f"request {request_id} was cancelled")
                return inference_pb2.ActivationData()

        # Bug 9 Fix: LRU Cache Eviction Data Loss (Admission Control)
        if request.is_prefill and request.use_cache:
            def check_cache_slots():
                with self.executor._cache_lock:
                    return len(self.executor._kv_cache_by_request)
            current_cache_count = await asyncio.to_thread(check_cache_slots)
            if current_cache_count >= self.executor.max_cached_requests:
                if context is not None:
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                    context.set_details(
                        f"node KV cache slots are full ({current_cache_count}/{self.executor.max_cached_requests})"
                    )
                return inference_pb2.ActivationData()

        if not await self._try_acquire_lane():
            if context is not None:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("node forward lanes are saturated")
            return inference_pb2.ActivationData()

        # Bug 7 Fix: Thread Starvation
        async with self._status_lock:
            if request_id and request_id in self._cancelled_requests:
                await self._release_lane()
                if context is not None:
                    context.set_code(grpc.StatusCode.CANCELLED)
                    context.set_details(f"request {request_id} was cancelled while in queue")
                return inference_pb2.ActivationData()

        try:
            # Deserialize input tensors offloaded to thread pool
            def _deserialize_inputs():
                h = deserialize_tensor(request.hidden_states.data, device=self.device_type)
                a = deserialize_tensor(request.attention_mask.data, device=self.device_type) if request.attention_mask.data else None
                p = deserialize_tensor(request.position_ids.data, device=self.device_type) if request.position_ids.data else None
                return h, a, p
                
            hidden_states, attention_mask, position_ids = await asyncio.to_thread(_deserialize_inputs)

            # Run forward pass (offloaded)
            output = await asyncio.to_thread(
                self.executor.forward,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                request_id=request_id,
                use_cache=request.use_cache,
                reset_cache=request.reset_cache,
                cache_position=request.cache_position,
                is_prefill=request.is_prefill,
            )

            # Serialize output (offloaded)
            output_bytes = await asyncio.to_thread(serialize_tensor, output)
            elapsed_ms = (time.time() - start_time) * 1000

            log.info(
                f"Forward pass complete: "
                f"input_shape={list(hidden_states.shape)} → "
                f"output_shape={list(output.shape)}, "
                f"time={elapsed_ms:.1f}ms"
            )

            # Bug 10 Fix: Explicit Tensor Memory Deallocation
            output_shape = list(output.shape)
            output_dtype = str(output.dtype).replace("torch.", "")
            del hidden_states
            del attention_mask
            del position_ids
            del output

            return inference_pb2.ActivationData(
                hidden_states=inference_pb2.TensorData(
                    data=output_bytes,
                    shape=output_shape,
                    dtype=output_dtype,
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
            await self._set_status(inference_pb2.NodeStatus.ERROR)
            if context is not None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            return inference_pb2.ActivationData()
        finally:
            await self._release_lane()

    async def Heartbeat(self, request, context):
        """Respond to health check with current status."""
        return inference_pb2.HeartbeatResponse(
            status=await self._build_status()
        )

    async def UnloadShard(self, request, context):
        """Unload the current model shard."""
        log.info("Unloading shard")
        await asyncio.to_thread(self.executor.unload)
        await self._set_status(inference_pb2.NodeStatus.IDLE)
        return await self._build_status()

    async def ClearRequestCache(self, request, context):
        """Clear cache state for one request or all requests."""
        if request.clear_all:
            await asyncio.to_thread(self.executor.clear_all_cache)
            async with self._status_lock:
                self._cancelled_requests.clear()
        elif request.request_id:
            await asyncio.to_thread(self.executor.clear_request_cache, request.request_id)
        return inference_pb2.Empty()

    async def CancelRequest(self, request, context):
        """Cancel an in-flight request and clear its cache."""
        if request.clear_all:
            await asyncio.to_thread(self.executor.clear_all_cache)
            async with self._status_lock:
                self._cancelled_requests.clear()
            return inference_pb2.Empty()

        if request.request_id:
            await asyncio.to_thread(self.executor.clear_request_cache, request.request_id)
            async with self._status_lock:
                self._cancelled_requests.add(request.request_id)
        return inference_pb2.Empty()

    async def _build_status(self) -> inference_pb2.NodeStatus:
        """Build a NodeStatus protobuf message."""
        layer_info = self.executor.get_layer_info()
        assigned = list(range(layer_info["start_layer"], layer_info["end_layer"]))
        async with self._status_lock:
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
) -> grpc.aio.Server:
    """Create and configure a gRPC aio server for the node.

    Args:
        node_id: Unique identifier for this node.
        port: Port to listen on.
        device_type: Device type for inference ("cuda" or "cpu").
        max_workers: Unused by grpc.aio but kept for API stability.

    Returns:
        Configured (but not yet started) gRPC server.
    """
    # Set large message limits for tensor transfer
    options = [
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]

    server = grpc.aio.server(options=options)

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

