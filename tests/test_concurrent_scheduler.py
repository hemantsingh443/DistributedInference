"""Unit tests for the concurrent coordinator scheduler."""

import asyncio

import pytest
import pytest_asyncio

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


@pytest.mark.asyncio
async def test_weighted_fair_dispatch_prevents_user_starvation():
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
    scheduler.start()

    order: list[str] = []
    gate = asyncio.Event()

    first = await scheduler.acquire(
        request_id="req-a0",
        user_id="user-a",
        execution_plan=plan,
        cancel_event=asyncio.Event(),
    )
    assert first.request_id == "req-a0"

    async def _worker(req_id: str, user_id: str):
        ticket = await scheduler.acquire(
            request_id=req_id,
            user_id=user_id,
            execution_plan=plan,
            cancel_event=asyncio.Event(),
        )
        order.append(req_id)
        try:
            await asyncio.wait_for(gate.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        await scheduler.release(req_id)
        return ticket

    t2 = asyncio.create_task(_worker("req-a1", "user-a"))
    t3 = asyncio.create_task(_worker("req-a2", "user-a"))
    t4 = asyncio.create_task(_worker("req-b0", "user-b"))

    await asyncio.sleep(0.05)
    await scheduler.release("req-a0")
    await asyncio.sleep(0.1)
    gate.set()

    await asyncio.wait_for(asyncio.gather(t2, t3, t4), timeout=2.0)
    await scheduler.stop()

    assert "req-b0" in order
    assert order.index("req-b0") < order.index("req-a2")


@pytest.mark.asyncio
async def test_queue_overflow_returns_retry_hint():
    registry = _build_registry()
    scheduler = ConcurrentRequestScheduler(
        registry=registry,
        max_concurrent_requests=1,
        max_queue_size=1,
        scheduler_tick_ms=1,
    )
    plan = _build_execution_plan()
    scheduler.start()

    _ = await scheduler.acquire(
        request_id="req-1",
        user_id="user",
        execution_plan=plan,
        cancel_event=asyncio.Event(),
    )

    holder_gate = asyncio.Event()

    async def _queued_worker():
        ticket = await scheduler.acquire(
            request_id="req-2",
            user_id="user",
            execution_plan=plan,
            cancel_event=asyncio.Event(),
        )
        try:
            await asyncio.wait_for(holder_gate.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        await scheduler.release("req-2")
        return ticket

    queued_task = asyncio.create_task(_queued_worker())
    await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="queue full"):
        await scheduler.acquire(
            request_id="req-3",
            user_id="user",
            execution_plan=plan,
            cancel_event=asyncio.Event(),
        )

    await scheduler.release("req-1")
    holder_gate.set()
    await asyncio.wait_for(queued_task, timeout=2.0)
    await scheduler.stop()


@pytest.mark.asyncio
async def test_capacity_queues_and_waits_for_lane_to_free():
    registry = _build_registry()
    # Saturate node lane capacity so scheduler cannot dispatch initially.
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
    scheduler.start()

    gate = asyncio.Event()
    acquired_ticket = []

    async def _requester():
        ticket = await scheduler.acquire(
            request_id="req-capacity",
            user_id="user",
            execution_plan=plan,
            cancel_event=asyncio.Event(),
        )
        acquired_ticket.append(ticket)
        gate.set()

    t = asyncio.create_task(_requester())

    # Wait a bit to ensure it's blocked in the queue due to node saturation
    await asyncio.sleep(0.5)
    assert not gate.is_set()

    # Free up the node lane
    registry.update_heartbeat(
        node_id="node-1",
        vram_used_mb=100,
        active_requests=0,
        queue_depth=0,
        estimated_free_vram_mb=1900,
    )

    # Now it should be able to acquire
    try:
        await asyncio.wait_for(gate.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        pass
    assert gate.is_set()
    assert acquired_ticket[0].request_id == "req-capacity"

    await scheduler.stop()
