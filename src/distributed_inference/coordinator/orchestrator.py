"""Main coordinator orchestrator — ties together all coordinator components.

Manages the full lifecycle of the coordinator:
- gRPC server for accepting node registrations and client requests
- Model partitioning and shard distribution
- Inference request handling
- Node health monitoring
"""

import threading
import time
import uuid
from concurrent import futures
from typing import Optional

import grpc
import torch
from transformers import AutoTokenizer

from distributed_inference.common.config import SystemConfig, load_config
from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.partitioner import partition_model, PartitionPlan
from distributed_inference.coordinator.registry import NodeRegistry, NodeState
from distributed_inference.coordinator.router import ActivationRouter
from distributed_inference.coordinator.scheduler import Scheduler, ExecutionPlan
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


class CoordinatorServiceImpl(inference_pb2_grpc.CoordinatorServiceServicer):
    """gRPC service implementation for the coordinator.

    Handles node registration, health reports, and inference requests.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        self.orchestrator = orchestrator

    def RegisterNode(self, request, context):
        """Handle node registration."""
        node = self.orchestrator.registry.register(
            node_id=request.node_id,
            address=request.address,
            vram_mb=request.vram_mb,
            compute_tflops=request.compute_tflops,
            bandwidth_mbps=request.bandwidth_mbps,
            device_type=request.device_type,
            device_name=request.device_name,
        )

        return inference_pb2.RegistrationAck(
            success=True,
            message=f"Node {request.node_id} registered successfully",
            assigned_node_id=request.node_id,
        )

    def ReportHealth(self, request, context):
        """Handle health reports from nodes."""
        self.orchestrator.registry.update_heartbeat(
            node_id=request.node_id,
            vram_used_mb=request.vram_used_mb,
        )
        return inference_pb2.Empty()

    def SubmitInference(self, request, context):
        """Handle an inference request from a client."""
        log.info(
            f"[bold magenta]Inference request[/]: "
            f"prompt='{request.prompt[:50]}...' "
            f"max_tokens={request.max_tokens}"
        )

        try:
            result = self.orchestrator.run_inference(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 50,
                request_id=request.request_id or uuid.uuid4().hex[:8],
            )
            return result

        except Exception as e:
            log.error(f"Inference failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferenceResponse(
                request_id=request.request_id,
                generated_text=f"ERROR: {e}",
            )


class Orchestrator:
    """Central coordinator for the distributed inference system.

    Manages node registration, model partitioning, and inference
    request routing across the distributed pipeline.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or load_config()
        self.registry = NodeRegistry()
        self.router = ActivationRouter()
        self.scheduler = Scheduler(registry=self.registry)

        self._partition_plan: Optional[PartitionPlan] = None
        self._execution_plan: Optional[ExecutionPlan] = None
        self._tokenizer = None
        self._server: Optional[grpc.Server] = None
        self._running = False
        self._model_loaded = False
        self._health_thread: Optional[threading.Thread] = None

    @property
    def is_ready(self) -> bool:
        """Whether the system is ready to serve inference requests."""
        return self._model_loaded and self._partition_plan is not None

    def start(self, block: bool = True) -> None:
        """Start the coordinator gRPC server.

        Args:
            block: If True, blocks until the server is stopped.
        """
        port = self.config.coordinator.port
        host = self.config.coordinator.host

        options = [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=options,
        )

        servicer = CoordinatorServiceImpl(self)
        inference_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            servicer, self._server
        )

        self._server.add_insecure_port(f"{host}:{port}")
        self._server.start()
        self._running = True

        log.info(f"[bold blue]Coordinator started[/] on {host}:{port}")

        # Start health monitoring thread
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="health-monitor",
        )
        self._health_thread.start()

        if block:
            try:
                self._server.wait_for_termination()
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stop the coordinator."""
        log.info("Stopping coordinator...")
        self._running = False
        if self._server:
            self._server.stop(grace=5)
        self.router.close()
        log.info("Coordinator stopped")

    def wait_for_nodes(self, min_nodes: int, timeout: float = 60.0) -> bool:
        """Wait until minimum number of nodes have registered.

        Args:
            min_nodes: Minimum number of nodes to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            True if sufficient nodes registered within timeout.
        """
        log.info(f"Waiting for {min_nodes} nodes (timeout {timeout}s)...")
        start = time.time()

        while time.time() - start < timeout:
            active = self.registry.active_count
            if active >= min_nodes:
                log.info(f"Got {active} nodes, proceeding")
                return True
            time.sleep(1.0)

        log.warning(
            f"Timeout: only {self.registry.active_count}/{min_nodes} nodes registered"
        )
        return self.registry.active_count > 0

    def setup_model(self) -> None:
        """Partition the model and load shards onto nodes.

        Must be called after nodes have registered.
        """
        active_nodes = self.registry.get_active_nodes()
        if not active_nodes:
            raise RuntimeError("No active nodes available for model setup")

        log.info(f"Setting up model with {len(active_nodes)} nodes")

        # Load tokenizer
        log.info(f"Loading tokenizer for {self.config.model.name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

        # Create partition plan
        self._partition_plan = partition_model(
            nodes=active_nodes,
            model_config=self.config.model,
        )

        # Create execution plan
        self._execution_plan = self.scheduler.create_execution_plan(
            self._partition_plan
        )

        # Load shards on each node
        for assignment in self._partition_plan.assignments:
            node = self.registry.get_node(assignment.node_id)
            if not node:
                raise RuntimeError(f"Node {assignment.node_id} not found")

            log.info(
                f"Loading shard on {node.node_id}: "
                f"layers [{assignment.start_layer}, {assignment.end_layer})"
            )

            self.router.load_shard_on_node(
                address=node.address,
                model_name=self.config.model.name,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                has_embedding=assignment.has_embedding,
                has_lm_head=assignment.has_lm_head,
                dtype=self.config.model.dtype,
            )

            self.registry.set_node_assignment(
                node_id=assignment.node_id,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                has_embedding=assignment.has_embedding,
                has_lm_head=assignment.has_lm_head,
            )

        self._model_loaded = True
        log.info("[bold green]Model setup complete — ready for inference[/]")

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: str = "",
    ) -> inference_pb2.InferenceResponse:
        """Run inference on a prompt using the distributed pipeline.

        Implements autoregressive token generation across the pipeline.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            request_id: Unique request identifier.

        Returns:
            InferenceResponse with generated text and metrics.
        """
        if not self.is_ready:
            raise RuntimeError("Model not set up. Call setup_model() first.")

        start_time = time.time()

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        all_hop_latencies = []
        generated_tokens = 0

        # Autoregressive generation loop
        for step in range(max_tokens):
            # Forward pass through the distributed pipeline
            logits, hop_latencies = self.router.route_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                execution_plan=self._execution_plan,
                request_id=f"{request_id}-step{step}",
            )

            all_hop_latencies.extend(hop_latencies)

            # Sample next token from logits
            next_token_logits = logits[:, -1, :]  # Last position

            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self._tokenizer.eos_token_id:
                break

            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, dtype=torch.long)],
                    dim=-1,
                )

            generated_tokens += 1

        # Decode full generated text
        generated_text = self._tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )

        total_time_ms = (time.time() - start_time) * 1000
        tokens_per_sec = (
            generated_tokens / (total_time_ms / 1000)
            if total_time_ms > 0 else 0
        )

        log.info(
            f"[bold green]Inference complete[/]: "
            f"{generated_tokens} tokens in {total_time_ms:.0f}ms "
            f"({tokens_per_sec:.1f} tok/s)"
        )

        return inference_pb2.InferenceResponse(
            request_id=request_id,
            generated_text=generated_text,
            tokens_generated=generated_tokens,
            total_latency_ms=total_time_ms,
            per_hop_latency_ms=all_hop_latencies,
            tokens_per_second=tokens_per_sec,
        )

    def _health_monitor_loop(self) -> None:
        """Background health monitoring of registered nodes."""
        interval = self.config.coordinator.heartbeat_interval_sec
        timeout = self.config.coordinator.heartbeat_timeout_sec
        threshold = self.config.coordinator.failure_threshold

        while self._running:
            time.sleep(interval)

            for node in self.registry.get_all_nodes():
                if node.state == NodeState.DEAD:
                    continue

                elapsed = time.time() - node.last_heartbeat
                if elapsed > timeout:
                    node.missed_heartbeats += 1
                    if node.missed_heartbeats >= threshold:
                        self.registry.mark_dead(node.node_id)
                        log.error(
                            f"Node {node.node_id} declared DEAD "
                            f"({node.missed_heartbeats} missed heartbeats)"
                        )
                    else:
                        self.registry.mark_suspect(node.node_id)
