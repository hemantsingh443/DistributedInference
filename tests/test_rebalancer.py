"""Tests for rebalance controller cooldown/coalescing behavior."""

import asyncio
import time
import pytest

from distributed_inference.coordinator.rebalancer import RebalanceController


async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_rebalancer_honors_cooldown() -> None:
    calls: list[tuple[str, float]] = []

    controller = RebalanceController(
        callback=lambda reason: calls.append((reason, time.time())),
        cooldown_sec=0.2,
    )
    controller.start()
    try:
        controller.request("first")
        assert await _wait_until(lambda: len(calls) == 1)

        controller.request("second")
        await asyncio.sleep(0.05)
        assert len(calls) == 1

        assert await _wait_until(lambda: len(calls) == 2)
        assert calls[1][1] - calls[0][1] >= 0.18
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_rebalancer_coalesces_pending_reasons() -> None:
    reasons: list[str] = []
    controller = RebalanceController(
        callback=lambda reason: reasons.append(reason),
        cooldown_sec=0.0,
    )
    controller.start()
    try:
        controller.request("join")
        controller.request("dead")
        assert await _wait_until(lambda: len(reasons) >= 1)
        assert reasons[0] == "dead"
    finally:
        await controller.stop()
