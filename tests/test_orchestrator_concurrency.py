"""Tests for coordinator request concurrency, backpressure, and cancellation."""

import threading
import time

import pytest
import torch

from distributed_inference.common.config import SystemConfig
from distributed_inference.coordinator.orchestrator import (
    Orchestrator,
    RequestCancelledError,
)
from distributed_inference.coordinator.partitioner import PartitionPlan
from distributed_inference.coordinator.router import HopTrace
from distributed_inference.coordinator.scheduler import ExecutionPlan, PipelineStage


class DummyTokenizer:
    """Tokenizer stub for deterministic inference behavior."""

    eos_token_id = 2

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = list(token_ids)

        mapping = {1: "Hello", 4: " world", 2: "</s>"}
        parts = []
        for token_id in ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            parts.append(mapping.get(token_id, ""))
        return "".join(parts)


class ControlledRouter:
    """Router stub with controllable pacing and cancellation hooks."""

    def __init__(self, stage: PipelineStage, gate: threading.Event | None = None, delay_sec: float = 0.0):
        self.stage = stage
        self.gate = gate
        self.delay_sec = delay_sec
        self.route_calls = 0
        self.cancelled_requests: list[str] = []

    def route_forward_stream(
        self,
        input_ids,
        attention_mask,
        execution_plan,
        request_id,
        use_cache=False,
        reset_cache=False,
        cache_position=None,
        is_prefill=False,
    ):
        del attention_mask, execution_plan, request_id
        del use_cache, reset_cache, cache_position, is_prefill

        self.route_calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

        seq_len = input_ids.shape[1]
        logits = torch.zeros((1, seq_len, 8), dtype=torch.float32)
        logits[0, -1, 4] = 10.0  # Always emit token 4 to keep decoding alive.
        yield HopTrace(
            hop_index=0,
            stage=self.stage,
            output=logits,
            hop_latency_ms=4.0,
        )

    def clear_request_cache_on_pipeline(self, execution_plan, request_id, clear_all=False):
        del execution_plan, request_id, clear_all

    def cancel_request_on_pipeline(self, execution_plan, request_id):
        del execution_plan
        self.cancelled_requests.append(request_id)


def _build_ready_orchestrator(config: SystemConfig, router: ControlledRouter) -> Orchestrator:
    orchestrator = Orchestrator(config=config)
    orchestrator._running = True
    orchestrator._model_loaded = True
    orchestrator._partition_plan = PartitionPlan(
        assignments=[],
        model_name="dummy",
        total_layers=22,
    )
    orchestrator._execution_plan = ExecutionPlan(stages=[router.stage], estimated_total_ms=1.0)
    orchestrator._tokenizer = DummyTokenizer()
    orchestrator.router = router
    return orchestrator


def test_backpressure_rejects_when_queue_is_full():
    config = SystemConfig()
    config.inference.enable_kv_cache = False
    config.coordinator.max_concurrent_requests_global = 1
    config.coordinator.max_queue_size = 1

    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )
    gate = threading.Event()
    router = ControlledRouter(stage=stage, gate=gate, delay_sec=0.0)
    orchestrator = _build_ready_orchestrator(config=config, router=router)

    results = {}

    def _run(name: str, request_id: str):
        try:
            results[name] = orchestrator.run_inference(
                prompt="test",
                max_tokens=1,
                temperature=0.0,
                request_id=request_id,
            )
        except Exception as e:  # pragma: no cover - assertion checks errors explicitly
            results[name] = e

    t1 = threading.Thread(target=_run, args=("a", "req-a"), daemon=True)
    t1.start()

    deadline = time.time() + 2.0
    while router.route_calls == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert router.route_calls >= 1

    t2 = threading.Thread(target=_run, args=("b", "req-b"), daemon=True)
    t2.start()

    queued_deadline = time.time() + 2.0
    while time.time() < queued_deadline:
        with orchestrator._state_lock:
            if orchestrator._queued_requests >= 1:
                break
        time.sleep(0.01)
    with orchestrator._state_lock:
        assert orchestrator._queued_requests >= 1

    with pytest.raises(RuntimeError, match="queue full"):
        orchestrator.run_inference(
            prompt="overflow",
            max_tokens=1,
            temperature=0.0,
            request_id="req-c",
        )

    gate.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert "a" in results and "b" in results
    assert not isinstance(results["a"], Exception)
    assert not isinstance(results["b"], Exception)


def test_cancel_inference_interrupts_running_stream():
    config = SystemConfig()
    config.inference.enable_kv_cache = False
    config.coordinator.max_concurrent_requests_global = 2
    config.coordinator.max_queue_size = 2

    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )
    router = ControlledRouter(stage=stage, delay_sec=0.03)
    orchestrator = _build_ready_orchestrator(config=config, router=router)

    captured = {"error": None}

    def _stream():
        try:
            _ = list(
                orchestrator.run_inference_stream(
                    prompt="cancel-me",
                    max_tokens=25,
                    temperature=0.0,
                    request_id="cancel-1",
                )
            )
        except Exception as e:  # pragma: no cover - asserted below
            captured["error"] = e

    worker = threading.Thread(target=_stream, daemon=True)
    worker.start()

    deadline = time.time() + 2.0
    while router.route_calls < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert router.route_calls >= 1

    assert orchestrator.cancel_inference("cancel-1", reason="unit-test") is True
    worker.join(timeout=2.0)

    assert isinstance(captured["error"], RequestCancelledError)
    assert "cancel-1" in router.cancelled_requests
    assert orchestrator.cancel_inference("cancel-1", reason="second-attempt") is False


def test_supports_four_concurrent_active_requests():
    config = SystemConfig()
    config.inference.enable_kv_cache = False
    config.coordinator.max_concurrent_requests_global = 4
    config.coordinator.max_queue_size = 0

    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )
    gate = threading.Event()
    router = ControlledRouter(stage=stage, gate=gate, delay_sec=0.0)
    orchestrator = _build_ready_orchestrator(config=config, router=router)

    results = {}

    def _run(name: str):
        results[name] = orchestrator.run_inference(
            prompt="test",
            max_tokens=1,
            temperature=0.0,
            request_id=f"req-{name}",
        )

    workers = []
    for idx in range(4):
        worker = threading.Thread(target=_run, args=(str(idx),), daemon=True)
        workers.append(worker)
        worker.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        with orchestrator._state_lock:
            if orchestrator._active_requests == 4:
                break
        time.sleep(0.01)
    with orchestrator._state_lock:
        assert orchestrator._active_requests == 4

    with pytest.raises(RuntimeError, match="queue full"):
        orchestrator.run_inference(
            prompt="overflow",
            max_tokens=1,
            temperature=0.0,
            request_id="req-overflow",
        )

    gate.set()
    for worker in workers:
        worker.join(timeout=2.0)

    assert len(results) == 4


def test_concurrent_scheduler_emits_event_metadata_and_user_fairness_fields():
    config = SystemConfig()
    config.inference.enable_kv_cache = False
    config.coordinator.enable_concurrent_scheduler = True
    config.coordinator.max_concurrent_requests_global = 2
    config.coordinator.max_queue_size = 4

    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )
    router = ControlledRouter(stage=stage, delay_sec=0.0)
    orchestrator = _build_ready_orchestrator(config=config, router=router)
    orchestrator.registry.register(
        node_id="node-1",
        address="localhost:50051",
        vram_mb=2048,
        compute_tflops=1.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test",
    )
    orchestrator.registry.update_heartbeat(
        node_id="node-1",
        vram_used_mb=128,
        active_requests=0,
        queue_depth=0,
        estimated_free_vram_mb=1800,
    )

    try:
        events = list(
            orchestrator.run_inference_stream(
                prompt="test",
                max_tokens=1,
                temperature=0.0,
                request_id="sched-1",
                user_id="user-alpha",
            )
        )
    finally:
        if orchestrator._concurrent_scheduler is not None:
            orchestrator._concurrent_scheduler.stop()

    assert len(events) >= 2
    first = events[0]
    assert first.scheduler_policy == "balanced"
    assert first.lane_id > 0
    assert first.queue_wait_ms >= 0
    assert first.scheduler_retries >= 0


def test_run_inference_stream_waits_for_model_readiness():
    config = SystemConfig()
    config.inference.enable_kv_cache = False
    config.coordinator.ready_wait_timeout_sec = 0.5
    config.coordinator.ready_poll_interval_ms = 10

    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )
    router = ControlledRouter(stage=stage, delay_sec=0.0)
    orchestrator = _build_ready_orchestrator(config=config, router=router)
    with orchestrator._state_lock:
        orchestrator._model_loaded = False
        orchestrator._partition_plan = None
        orchestrator._execution_plan = None

    def _mark_ready():
        time.sleep(0.05)
        with orchestrator._state_lock:
            orchestrator._partition_plan = PartitionPlan(
                assignments=[],
                model_name="dummy",
                total_layers=22,
            )
            orchestrator._execution_plan = ExecutionPlan(
                stages=[stage],
                estimated_total_ms=1.0,
            )
            orchestrator._model_loaded = True

    warmup = threading.Thread(target=_mark_ready, daemon=True)
    warmup.start()
    events = list(
        orchestrator.run_inference_stream(
            prompt="warmup",
            max_tokens=1,
            temperature=0.0,
            request_id="warmup-req",
        )
    )
    warmup.join(timeout=1.0)

    assert any(event.WhichOneof("payload") == "completed" for event in events)
