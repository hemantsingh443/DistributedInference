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
from typing import Iterator, Optional

import grpc
import torch
from transformers import AutoTokenizer

from distributed_inference.common.config import SystemConfig, load_config
from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.admission import AdmissionController
from distributed_inference.coordinator.partitioner import partition_model, PartitionPlan
from distributed_inference.coordinator.rebalancer import RebalanceController
from distributed_inference.coordinator.registry import NodeRegistry, NodeState
from distributed_inference.coordinator.router import ActivationRouter
from distributed_inference.coordinator.scheduler import Scheduler, ExecutionPlan
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


class NodeLoadError(RuntimeError):
    """Raised when a specific node cannot accept/load an assigned shard."""

    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id


class CoordinatorServiceImpl(inference_pb2_grpc.CoordinatorServiceServicer):
    """gRPC service implementation for the coordinator.

    Handles node registration, health reports, and inference requests.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        self.orchestrator = orchestrator

    def RegisterNode(self, request, context):
        """Handle node registration."""
        decision = self.orchestrator.admission.evaluate(
            vram_mb=request.vram_mb,
            compute_tflops=request.compute_tflops,
            device_type=request.device_type,
        )

        if not decision.admitted:
            log.warning(
                f"Rejected node registration {request.node_id}: {decision.reason}"
            )
            return inference_pb2.RegistrationAck(
                success=False,
                message=f"Node {request.node_id} rejected",
                assigned_node_id=request.node_id,
                admission_status=inference_pb2.RegistrationAck.REJECTED,
                rejection_reason=decision.reason,
                rebalance_plan_version=self.orchestrator.plan_version,
            )

        node = self.orchestrator.registry.register(
            node_id=request.node_id,
            address=request.address,
            vram_mb=request.vram_mb,
            compute_tflops=request.compute_tflops,
            bandwidth_mbps=request.bandwidth_mbps,
            device_type=request.device_type,
            device_name=request.device_name,
            sram_mb=request.sram_mb,
            latency_ms=request.latency_ms,
            effective_bandwidth_mbps=request.effective_bandwidth_mbps,
            admitted=True,
            admission_reason="",
        )

        return inference_pb2.RegistrationAck(
            success=True,
            message=f"Node {request.node_id} registered successfully",
            assigned_node_id=request.node_id,
            admission_status=inference_pb2.RegistrationAck.ADMITTED,
            rejection_reason="",
            rebalance_plan_version=self.orchestrator.plan_version,
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

    def SubmitInferenceStream(self, request, context):
        """Handle a streaming inference request from a client."""
        request_id = request.request_id or uuid.uuid4().hex[:8]
        log.info(
            f"[bold magenta]Streaming inference request[/]: "
            f"request_id={request_id} "
            f"prompt='{request.prompt[:50]}...' "
            f"max_tokens={request.max_tokens}"
        )

        try:
            for event in self.orchestrator.run_inference_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 50,
                request_id=request_id,
            ):
                yield event
        except Exception as e:
            log.error(f"Streaming inference failed: {e}")
            yield inference_pb2.InferenceEvent(
                request_id=request_id,
                timestamp_ms=int(time.time() * 1000),
                error=str(e),
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
        self.scheduler = Scheduler(
            registry=self.registry,
            bandwidth_mbps_default=self.config.coordinator.default_bandwidth_mbps,
            latency_ms_default=self.config.coordinator.default_latency_ms,
        )
        self.admission = AdmissionController(self.config.coordinator)

        self._partition_plan: Optional[PartitionPlan] = None
        self._execution_plan: Optional[ExecutionPlan] = None
        self._tokenizer = None
        self._server: Optional[grpc.Server] = None
        self._running = False
        self._model_loaded = False
        self._health_thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()
        self._active_requests = 0
        self._accepting_requests = True
        self._plan_version = 0
        self._last_rebalance_reason = ""
        self._rebalance_lock = threading.RLock()
        self._rebalance_controller = RebalanceController(
            callback=self._rebalance_pipeline,
            cooldown_sec=self.config.coordinator.rebalance_cooldown_sec,
        )

        self.registry.on_change(self._on_registry_change)

    @property
    def is_ready(self) -> bool:
        """Whether the system is ready to serve inference requests."""
        return self._model_loaded and self._partition_plan is not None

    @property
    def plan_version(self) -> str:
        """Current active rebalance plan version."""
        return f"v{self._plan_version}"

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
        self._rebalance_controller.start()

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
        self._rebalance_controller.stop()
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
            active = len(self.registry.get_admitted_active_nodes())
            if active >= min_nodes:
                log.info(f"Got {active} nodes, proceeding")
                return True
            time.sleep(1.0)

        log.warning(
            f"Timeout: only {len(self.registry.get_admitted_active_nodes())}/{min_nodes} nodes registered"
        )
        return len(self.registry.get_admitted_active_nodes()) > 0

    def setup_model(self) -> None:
        """Partition the model and load shards onto nodes.

        Must be called after nodes have registered.
        """
        nodes = self.registry.get_admitted_active_nodes()
        if not nodes:
            raise RuntimeError("No admitted active nodes available for model setup")
        self._ensure_tokenizer()
        self._rebalance_controller.run_now("initial_setup")

    def _ensure_tokenizer(self) -> None:
        """Load tokenizer lazily and exactly once."""
        if self._tokenizer is not None:
            return
        log.info(f"Loading tokenizer for {self.config.model.name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

    def _on_registry_change(self, event: str, _node) -> None:
        """Schedule a rebalance for topology-changing events."""
        if event in {"register", "unregister", "dead"}:
            self._rebalance_controller.request(f"topology:{event}")

    def _rebalance_pipeline(self, reason: str) -> bool:
        """Recompute and apply partitioning with drain-then-swap cutover."""
        if not self._running and reason != "initial_setup":
            return False

        with self._rebalance_lock:
            start = time.time()
            self._ensure_tokenizer()

            old_plan = self._partition_plan
            old_execution = self._execution_plan
            old_loaded = self._model_loaded
            last_error: Optional[Exception] = None
            applied = False

            self._begin_drain()
            drained = self._wait_for_drain(
                timeout_sec=self.config.coordinator.rebalance_drain_timeout_sec
            )
            if not drained:
                log.warning(
                    "Rebalance proceeding after drain timeout with active requests still running"
                )

            try:
                for attempt in range(1, 4):
                    nodes = self._get_rebalance_candidate_nodes()
                    if not nodes:
                        break

                    try:
                        new_plan = partition_model(
                            nodes=nodes,
                            model_config=self.config.model,
                            alpha_latency=self.config.coordinator.allocation_alpha_latency,
                            beta_throughput=self.config.coordinator.allocation_beta_throughput,
                            default_bandwidth_mbps=self.config.coordinator.default_bandwidth_mbps,
                            default_latency_ms=self.config.coordinator.default_latency_ms,
                            memory_safety_margin=self.config.coordinator.memory_safety_margin,
                        )
                    except ValueError as e:
                        last_error = e
                        log.warning(
                            f"Rebalance attempt {attempt} cannot produce feasible plan: {e}"
                        )
                        break
                    new_execution = self.scheduler.create_execution_plan(new_plan)

                    try:
                        self._load_partition_plan(new_plan)
                        with self._state_lock:
                            self._partition_plan = new_plan
                            self._execution_plan = new_execution
                            self._model_loaded = True
                            self._plan_version += 1
                            self._last_rebalance_reason = reason
                        self._unload_removed_nodes(old_plan, new_plan)
                        elapsed_ms = (time.time() - start) * 1000
                        log.info(
                            f"Rebalance complete ({self.plan_version}, reason={reason}) "
                            f"in {elapsed_ms:.0f}ms"
                        )
                        applied = True
                        break
                    except NodeLoadError as e:
                        last_error = e
                        self.registry.mark_dead(e.node_id)
                        log.warning(
                            f"Rebalance attempt {attempt} failed at node {e.node_id}: {e}. "
                            "Retrying with remaining healthy nodes."
                        )
                    except Exception as e:
                        last_error = e
                        log.error(
                            f"Rebalance attempt {attempt} failed ({reason}): {e}"
                        )
                        break

                if applied:
                    return True

                if (
                    old_plan
                    and old_execution
                    and old_loaded
                    and self._is_plan_viable(old_plan)
                ):
                    with self._state_lock:
                        self._partition_plan = old_plan
                        self._execution_plan = old_execution
                        self._model_loaded = old_loaded
                    log.warning(
                        "Rebalance failed; retained last known-good execution plan"
                    )
                else:
                    with self._state_lock:
                        self._model_loaded = False
                        self._partition_plan = None
                        self._execution_plan = None
                    log.error(
                        "Rebalance failed and no viable last known-good plan is available"
                    )

                message = (
                    f"Rebalance failed ({reason})"
                    + (f": {last_error}" if last_error else "")
                )
                if reason == "initial_setup":
                    raise RuntimeError(message)
                log.error(message)
                return False
            finally:
                self._end_drain()

    def _get_rebalance_candidate_nodes(self):
        """Return admitted nodes currently reachable for shard placement."""
        candidates = []
        for node in self.registry.get_admitted_active_nodes():
            health = self.router.check_node_health(node.address)
            if health is None:
                self.registry.mark_suspect(node.node_id)
                log.warning(
                    f"Excluding unreachable node from rebalance: "
                    f"{node.node_id} @ {node.address}"
                )
                continue
            candidates.append(node)
        return candidates

    def _is_plan_viable(self, plan: PartitionPlan) -> bool:
        """Check whether a plan only references currently live/reachable nodes."""
        active_ids = {n.node_id for n in self.registry.get_admitted_active_nodes()}
        plan_ids = {assignment.node_id for assignment in plan.assignments}
        if not plan_ids.issubset(active_ids):
            return False
        for assignment in plan.assignments:
            node = self.registry.get_node(assignment.node_id)
            if not node:
                return False
            if self.router.check_node_health(node.address) is None:
                return False
        return True

    def _load_partition_plan(self, plan: PartitionPlan) -> None:
        """Load a partition plan onto nodes and update registry assignments."""
        active_assignments = {
            assignment.node_id: assignment for assignment in plan.assignments
        }

        for assignment in plan.assignments:
            node = self.registry.get_node(assignment.node_id)
            if not node:
                raise RuntimeError(f"Node {assignment.node_id} not found")

            self.registry.set_node_state(node.node_id, NodeState.LOADING)
            if self.router.check_node_health(node.address) is None:
                raise NodeLoadError(
                    node_id=node.node_id,
                    message=(
                        f"Node {node.node_id} became unreachable before shard load"
                    ),
                )
            try:
                self.router.load_shard_on_node(
                    address=node.address,
                    model_name=self.config.model.name,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    has_embedding=assignment.has_embedding,
                    has_lm_head=assignment.has_lm_head,
                    dtype=self.config.model.dtype,
                )
            except Exception as e:
                raise NodeLoadError(
                    node_id=node.node_id,
                    message=f"Shard load failed on {node.node_id}: {e}",
                ) from e
            self.registry.set_node_assignment(
                node_id=assignment.node_id,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                has_embedding=assignment.has_embedding,
                has_lm_head=assignment.has_lm_head,
            )

        for node in self.registry.get_all_nodes():
            if node.node_id not in active_assignments and node.admitted:
                node.assigned_layers = (0, 0)
                node.has_embedding = False
                node.has_lm_head = False
                if node.state not in (NodeState.DEAD, NodeState.SUSPECT):
                    node.state = NodeState.IDLE

    def _unload_removed_nodes(
        self,
        old_plan: Optional[PartitionPlan],
        new_plan: PartitionPlan,
    ) -> None:
        """Unload shards from nodes removed in a new partition plan."""
        if not old_plan:
            return
        old_nodes = {assignment.node_id for assignment in old_plan.assignments}
        new_nodes = {assignment.node_id for assignment in new_plan.assignments}
        removed = old_nodes - new_nodes
        for node_id in removed:
            node = self.registry.get_node(node_id)
            if not node:
                continue
            if self.router.check_node_health(node.address) is None:
                continue
            self.router.unload_shard_on_node(node.address)

    def _begin_drain(self) -> None:
        with self._state_lock:
            self._accepting_requests = False

    def _end_drain(self) -> None:
        with self._state_lock:
            self._accepting_requests = True

    def _wait_for_drain(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._state_lock:
                if self._active_requests == 0:
                    return True
            time.sleep(0.05)
        with self._state_lock:
            return self._active_requests == 0

    def _enter_request(self) -> None:
        with self._state_lock:
            if not self._accepting_requests:
                raise RuntimeError("Coordinator is rebalancing; retry shortly")
            self._active_requests += 1

    def _exit_request(self) -> None:
        with self._state_lock:
            self._active_requests = max(self._active_requests - 1, 0)

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: str = "",
    ) -> inference_pb2.InferenceResponse:
        """Run inference and return the final aggregated response."""
        completed = None
        stream_request_id = request_id or uuid.uuid4().hex[:8]

        for event in self.run_inference_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            request_id=stream_request_id,
        ):
            if event.WhichOneof("payload") == "completed":
                completed = event.completed
            elif event.WhichOneof("payload") == "error":
                raise RuntimeError(event.error)

        if completed is None:
            raise RuntimeError("Inference stream ended without a completion event")
        return completed

    def run_inference_stream(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: str = "",
    ) -> Iterator[inference_pb2.InferenceEvent]:
        """Run inference and yield streaming hop/token/completion events."""
        self._enter_request()
        try:
            if not self.is_ready:
                raise RuntimeError("Model not set up. Call setup_model() first.")
            with self._state_lock:
                execution_plan = self._execution_plan
            if not execution_plan:
                raise RuntimeError("No execution plan available")

            start_time = time.time()
            request_id = request_id or uuid.uuid4().hex[:8]
            use_kv_cache = bool(
                getattr(self.config.inference, "enable_kv_cache", False)
            )

            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            all_hop_latencies = []
            generated_tokens = 0

            if use_kv_cache:
                logits = None
                prefill_cache_pos = int(input_ids.shape[1] - 1)
                for trace in self.router.route_forward_stream(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    execution_plan=execution_plan,
                    request_id=request_id,
                    use_cache=True,
                    reset_cache=True,
                    cache_position=prefill_cache_pos,
                    is_prefill=True,
                ):
                    logits = trace.output
                    all_hop_latencies.append(trace.hop_latency_ms)
                    yield inference_pb2.InferenceEvent(
                        request_id=request_id,
                        timestamp_ms=int(time.time() * 1000),
                        hop=inference_pb2.HopEvent(
                            step=0,
                            hop_index=trace.hop_index,
                            node_id=trace.stage.node_id,
                            address=trace.stage.address,
                            start_layer=trace.stage.start_layer,
                            end_layer=trace.stage.end_layer,
                            hop_latency_ms=trace.hop_latency_ms,
                        ),
                    )

                if logits is None:
                    raise RuntimeError("No logits returned from prefill")

                for step in range(max_tokens):
                    if step > 0:
                        decode_input = input_ids[:, -1:]
                        decode_cache_pos = int(input_ids.shape[1] - 1)
                        logits = None
                        for trace in self.router.route_forward_stream(
                            input_ids=decode_input,
                            attention_mask=attention_mask,
                            execution_plan=execution_plan,
                            request_id=request_id,
                            use_cache=True,
                            reset_cache=False,
                            cache_position=decode_cache_pos,
                            is_prefill=False,
                        ):
                            logits = trace.output
                            all_hop_latencies.append(trace.hop_latency_ms)
                            yield inference_pb2.InferenceEvent(
                                request_id=request_id,
                                timestamp_ms=int(time.time() * 1000),
                                hop=inference_pb2.HopEvent(
                                    step=step,
                                    hop_index=trace.hop_index,
                                    node_id=trace.stage.node_id,
                                    address=trace.stage.address,
                                    start_layer=trace.stage.start_layer,
                                    end_layer=trace.stage.end_layer,
                                    hop_latency_ms=trace.hop_latency_ms,
                                ),
                            )
                        if logits is None:
                            raise RuntimeError(
                                f"No logits returned on decode step {step}"
                            )

                    next_token = self._sample_next_token(
                        logits=logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break

                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype)],
                            dim=-1,
                        )

                    generated_tokens += 1
                    token_id = int(next_token.item())
                    token_text = self._tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    accumulated_text = self._tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )

                    yield inference_pb2.InferenceEvent(
                        request_id=request_id,
                        timestamp_ms=int(time.time() * 1000),
                        token=inference_pb2.TokenEvent(
                            step=step,
                            token_id=token_id,
                            token_text=token_text,
                            accumulated_text=accumulated_text,
                        ),
                    )
            else:
                for step in range(max_tokens):
                    logits = None
                    for trace in self.router.route_forward_stream(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        execution_plan=execution_plan,
                        request_id=f"{request_id}-step{step}",
                        use_cache=False,
                        reset_cache=False,
                        cache_position=None,
                        is_prefill=False,
                    ):
                        logits = trace.output
                        all_hop_latencies.append(trace.hop_latency_ms)
                        yield inference_pb2.InferenceEvent(
                            request_id=request_id,
                            timestamp_ms=int(time.time() * 1000),
                            hop=inference_pb2.HopEvent(
                                step=step,
                                hop_index=trace.hop_index,
                                node_id=trace.stage.node_id,
                                address=trace.stage.address,
                                start_layer=trace.stage.start_layer,
                                end_layer=trace.stage.end_layer,
                                hop_latency_ms=trace.hop_latency_ms,
                            ),
                        )

                    if logits is None:
                        raise RuntimeError("No logits returned from pipeline routing")

                    next_token = self._sample_next_token(
                        logits=logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break

                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype)],
                            dim=-1,
                        )

                    generated_tokens += 1
                    token_id = int(next_token.item())
                    token_text = self._tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    accumulated_text = self._tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )

                    yield inference_pb2.InferenceEvent(
                        request_id=request_id,
                        timestamp_ms=int(time.time() * 1000),
                        token=inference_pb2.TokenEvent(
                            step=step,
                            token_id=token_id,
                            token_text=token_text,
                            accumulated_text=accumulated_text,
                        ),
                    )
        finally:
            if 'use_kv_cache' in locals() and use_kv_cache and execution_plan:
                try:
                    self.router.clear_request_cache_on_pipeline(
                        execution_plan=execution_plan,
                        request_id=request_id,
                    )
                except Exception as e:
                    log.warning(f"Cache cleanup failed for {request_id}: {e}")
            self._exit_request()

        generated_text = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)
        total_time_ms = (time.time() - start_time) * 1000
        tokens_per_sec = (
            generated_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
        )

        log.info(
            f"[bold green]Inference complete[/]: "
            f"{generated_tokens} tokens in {total_time_ms:.0f}ms "
            f"({tokens_per_sec:.1f} tok/s)"
        )

        yield inference_pb2.InferenceEvent(
            request_id=request_id,
            timestamp_ms=int(time.time() * 1000),
            completed=inference_pb2.InferenceResponse(
                request_id=request_id,
                generated_text=generated_text,
                tokens_generated=generated_tokens,
                total_latency_ms=total_time_ms,
                per_hop_latency_ms=all_hop_latencies,
                tokens_per_second=tokens_per_sec,
            ),
        )

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample one token from model logits."""
        next_token_logits = logits[:, -1, :]  # Last position.

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                k = min(top_k, next_token_logits.shape[-1])
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

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
            return torch.multinomial(probs, num_samples=1)

        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

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
