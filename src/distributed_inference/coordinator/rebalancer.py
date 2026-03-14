"""Background rebalance trigger controller."""

import asyncio
import time
from typing import Awaitable, Callable, Optional

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


class RebalanceController:
    """Coalesces topology events and invokes rebalance with cooldown."""

    def __init__(
        self,
        callback: Callable[[str], Awaitable[None]],
        cooldown_sec: float = 5.0,
    ):
        self._callback = callback
        self._cooldown_sec = max(0.0, cooldown_sec)
        self._lock = None
        self._pending_reason: Optional[str] = None
        self._last_run_ts = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start background rebalance monitor task."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="rebalance-controller")

    async def stop(self) -> None:
        """Stop background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def request(self, reason: str) -> None:
        """Queue a rebalance request."""
        # Simple assignment is safe in single-threaded asyncio loop
        self._pending_reason = reason

    async def run_now(self, reason: str) -> None:
        """Run rebalance immediately, bypassing cooldown."""
        await self._callback(reason)
        if self._lock:
            async with self._lock:
                self._last_run_ts = time.time()

    async def _loop(self) -> None:
        while True:
            if self._lock is None:
                return
            async with self._lock:
                running = self._running
                pending_reason = self._pending_reason
                cooldown_remaining = (
                    self._cooldown_sec - (time.time() - self._last_run_ts)
                )
                can_run = pending_reason is not None and cooldown_remaining <= 0
                if can_run:
                    self._pending_reason = None
            if not running:
                return
            if can_run and pending_reason:
                await self._invoke(reason=pending_reason)
            await asyncio.sleep(0.2)

    async def _invoke(self, reason: str) -> None:
        try:
            await self._callback(reason)
        except Exception as e:  # pragma: no cover - defensive.
            log.exception(f"Rebalance callback failed ({reason}): {e}")
        finally:
            if self._lock:
                async with self._lock:
                    self._last_run_ts = time.time()
