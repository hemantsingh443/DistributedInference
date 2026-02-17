"""Node agent — main lifecycle manager for a distributed node.

Handles the full node lifecycle:
1. Start gRPC server
2. Register with coordinator
3. Wait for shard assignments
4. Load assigned model layers
5. Serve forward-pass requests
6. Send periodic heartbeats
"""

import threading
import time
import uuid
from typing import Optional

import grpc

from distributed_inference.common.config import NodeConfig, SystemConfig
from distributed_inference.common.logging import get_logger
from distributed_inference.node.resources import detect_resources, NodeCapabilities
from distributed_inference.node.server import create_node_server
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


class NodeAgent:
    """Manages the lifecycle of a single distributed inference node.

    The agent starts a gRPC server, registers with the coordinator,
    and enters a serve loop handling inference requests.
    """

    def __init__(
        self,
        port: int,
        coordinator_address: str,
        max_vram_mb: Optional[int] = None,
        device: str = "auto",
        node_id: Optional[str] = None,
        max_cached_requests: int = 1,
        max_cache_tokens_per_request: int = 4096,
        bandwidth_mbps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        require_registration_success: bool = False,
    ):
        self.port = port
        self.coordinator_address = coordinator_address
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"

        # Detect resources
        self.capabilities = detect_resources(
            max_vram_mb=max_vram_mb,
            device=device,
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
        )

        # Create gRPC server
        self.server = create_node_server(
            node_id=self.node_id,
            port=self.port,
            device_type=self.capabilities.device_type,
            max_cached_requests=max_cached_requests,
            max_cache_tokens_per_request=max_cache_tokens_per_request,
        )

        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_interval = 5.0
        self._require_registration_success = require_registration_success
        self._last_registration_error = ""

    def start(self, block: bool = True) -> None:
        """Start the node agent.

        Starts the gRPC server, registers with the coordinator,
        and begins heartbeat reporting.

        Args:
            block: If True, blocks until the server is stopped.
        """
        log.info(
            f"[bold blue]Starting node {self.node_id}[/] "
            f"on port {self.port}"
        )
        log.info(f"Capabilities: {self.capabilities.summary()}")

        # Start gRPC server
        self.server.start()
        self._running = True
        log.info(f"gRPC server started on port {self.port}")

        # Register with coordinator
        registered = self._register_with_coordinator()
        if not registered and self._require_registration_success:
            reason = self._last_registration_error or "registration failed"
            self.stop()
            raise RuntimeError(
                f"Initial registration failed (require_registration_success): {reason}"
            )

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"{self.node_id}-heartbeat",
        )
        self._heartbeat_thread.start()

        if block:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stop the node agent gracefully."""
        log.info(f"Stopping node {self.node_id}")
        self._running = False
        self.server.stop(grace=5)
        log.info(f"Node {self.node_id} stopped")

    def _register_with_coordinator(self) -> bool:
        """Register this node with the coordinator.

        Returns:
            True if registration was successful.
        """
        log.info(f"Registering with coordinator at {self.coordinator_address}")

        try:
            options = [
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ]
            channel = grpc.insecure_channel(
                self.coordinator_address, options=options
            )
            stub = inference_pb2_grpc.CoordinatorServiceStub(channel)

            node_info = inference_pb2.NodeInfo(
                node_id=self.node_id,
                address=f"localhost:{self.port}",
                vram_mb=self.capabilities.vram_mb,
                compute_tflops=self.capabilities.compute_tflops,
                bandwidth_mbps=self.capabilities.bandwidth_mbps,
                device_type=self.capabilities.device_type,
                device_name=self.capabilities.device_name,
                sram_mb=self.capabilities.sram_mb,
                latency_ms=self.capabilities.latency_ms,
                effective_bandwidth_mbps=self.capabilities.effective_bandwidth_mbps,
            )

            response = stub.RegisterNode(node_info, timeout=10)

            if response.success:
                log.info(
                    f"[bold green]Registered successfully[/]: "
                    f"{response.message}"
                )
                self._last_registration_error = ""
                return True
            else:
                rejection = (
                    f" ({response.rejection_reason})"
                    if response.rejection_reason else ""
                )
                self._last_registration_error = f"{response.message}{rejection}"
                log.error(f"Registration failed: {self._last_registration_error}")
                return False

        except grpc.RpcError as e:
            self._last_registration_error = f"{e.code().name}: {e.details()}"
            log.warning(
                f"Could not reach coordinator: {e.code()} - {e.details()}"
            )
            log.info("Will retry registration on next heartbeat")
            return False

    def _heartbeat_loop(self) -> None:
        """Background thread sending periodic heartbeats to coordinator."""
        while self._running:
            try:
                self._send_heartbeat()
            except Exception as e:
                log.warning(f"Heartbeat failed: {e}")

            time.sleep(self._heartbeat_interval)

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat to the coordinator."""
        try:
            options = [
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ]
            channel = grpc.insecure_channel(
                self.coordinator_address, options=options
            )
            stub = inference_pb2_grpc.CoordinatorServiceStub(channel)

            from distributed_inference.node.resources import get_vram_usage_mb

            status = inference_pb2.NodeStatus(
                node_id=self.node_id,
                status=inference_pb2.NodeStatus.READY,
                vram_used_mb=get_vram_usage_mb(),
                vram_total_mb=self.capabilities.vram_mb,
                timestamp_ms=int(time.time() * 1000),
            )

            stub.ReportHealth(status, timeout=5)

        except grpc.RpcError:
            pass  # Coordinator may not be available yet
