"""Unit tests for the concurrent coordinator scheduler."""

import threading
import time

import pytest

from distributed_inference.coordinator.concurrent_scheduler import ConcurrentRequestScheduler
from distributed_inference.coordinator.registry import NodeRegistry
from distributed_inference.coordinator.scheduler import ExecutionPlan, PipelineStage


def _build_execution_plan(node_id: str = "node-1") -> ExecutionPlan:
    return ExecutionPlan(
        stages=[
            PipelineStage(
                node_id=node_id,
                address="localhost:50051",
                start_layer=0,
                end_layer=22,
                has_embedding=True,
                has_lm_head=True,
            )
        ],
        estimated_total_ms=1.0,
    )


def _build_registry() -> NodeRegistry:
    registry = NodeRegistry()
    registry.register(
        node_id="node-1",
        address="localhost:50051",
        vram_mb=2048,
        compute_tflops=5.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test",
    )
    registry.update_heartbeat(
        node_id="node-1",
        vram_used_mb=100,
        active_requests=0,
        queue_depth=0,
        estimated_free_vram_mb=1900,
    )
    return registry


def test_weighted_fair_dispatch_prevents_user_starvation():
    registry = _build_registry()
    scheduler = ConcurrentRequestScheduler(
        registry=registry,
        max_concurrent_requests=1,
        max_queue_size=8,
        fairness_quantum_tokens=1,
        tail_latency_guardrail_ms=10_000,
        scheduler_tick_ms=1,
        max_dispatch_per_tick=1,
    )
    plan = _build_execution_plan()

    order: list[str] = []
    gate = threading.Event()
    gate.clear()

    first = scheduler.acquire(
        request_id="req-a0",
        user_id="user-a",
        execution_plan=plan,
        cancel_event=threading.Event(),
    )
    assert first.request_id == "req-a0"

    def _worker(req_id: str, user_id: str):
        ticket = scheduler.acquire(
            request_id=req_id,
            user_id=user_id,
            execution_plan=plan,
            cancel_event=threading.Event(),
        )
        order.append(req_id)
        gate.wait(timeout=2.0)
        scheduler.release(req_id)
        return ticket

    t2 = threading.Thread(target=_worker, args=("req-a1", "user-a"), daemon=True)
    t3 = threading.Thread(target=_worker, args=("req-a2", "user-a"), daemon=True)
    t4 = threading.Thread(target=_worker, args=("req-b0", "user-b"), daemon=True)
    t2.start()
    t3.start()
    t4.start()

    time.sleep(0.05)
    scheduler.release("req-a0")
    time.sleep(0.1)
    gate.set()

    t2.join(timeout=2.0)
    t3.join(timeout=2.0)
    t4.join(timeout=2.0)
    scheduler.stop()

    assert "req-b0" in order
    assert order.index("req-b0") < order.index("req-a2")


def test_queue_overflow_returns_retry_hint():
    registry = _build_registry()
    scheduler = ConcurrentRequestScheduler(
        registry=registry,
        max_concurrent_requests=1,
        max_queue_size=1,
        scheduler_tick_ms=1,
    )
    plan = _build_execution_plan()

    _ = scheduler.acquire(
        request_id="req-1",
        user_id="user",
        execution_plan=plan,
        cancel_event=threading.Event(),
    )

    holder_gate = threading.Event()

    def _queued_worker():
        ticket = scheduler.acquire(
            request_id="req-2",
            user_id="user",
            execution_plan=plan,
            cancel_event=threading.Event(),
        )
        holder_gate.wait(timeout=2.0)
        scheduler.release("req-2")
        return ticket

    queued_thread = threading.Thread(target=_queued_worker, daemon=True)
    queued_thread.start()
    time.sleep(0.05)

    with pytest.raises(RuntimeError, match="queue full"):
        scheduler.acquire(
            request_id="req-3",
            user_id="user",
            execution_plan=plan,
            cancel_event=threading.Event(),
        )

    scheduler.release("req-1")
    holder_gate.set()
    queued_thread.join(timeout=2.0)
    scheduler.stop()


def test_capacity_retries_then_rejects_with_retry_after():
    registry = _build_registry()
    # Saturate node lane capacity so scheduler cannot dispatch.
    registry.update_heartbeat(
        node_id="node-1",
        vram_used_mb=100,
        active_requests=1,
        queue_depth=0,
        estimated_free_vram_mb=1900,
    )
    scheduler = ConcurrentRequestScheduler(
        registry=registry,
        max_concurrent_requests=1,
        max_queue_size=4,
        max_retry_attempts=1,
        retry_backoff_ms=1,
        scheduler_tick_ms=1,
        node_max_concurrent_lanes=1,
    )
    plan = _build_execution_plan()

    with pytest.raises(RuntimeError, match="retry_after_ms"):
        scheduler.acquire(
            request_id="req-capacity",
            user_id="user",
            execution_plan=plan,
            cancel_event=threading.Event(),
        )
    scheduler.stop()
