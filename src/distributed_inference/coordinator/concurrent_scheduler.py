"""Concurrent request scheduler with fairness, aging, and capacity checks."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.registry import NodeRegistry
from distributed_inference.coordinator.scheduler import ExecutionPlan

log = get_logger(__name__)


@dataclass
class SchedulerTicket:
    """Dispatch metadata returned to a request when admitted to run."""

    request_id: str
    lane_id: int
    queue_wait_ms: float
    scheduler_retries: int


@dataclass
class _PendingRequest:
    request_id: str
    user_id: str
    execution_plan: ExecutionPlan
    cancel_event: threading.Event
    queued_at: float = field(default_factory=time.time)
    next_eligible_at: float = field(default_factory=time.time)
    dispatch_retries: int = 0
    granted: bool = False
    lane_id: int = 0
    queue_wait_ms: float = 0.0
    error: str = ""


class ConcurrentRequestScheduler:
    """Global concurrent scheduler for multi-user request dispatch."""

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        max_concurrent_requests: int,
        max_queue_size: int,
        scheduler_policy: str = "balanced",
        fairness_quantum_tokens: int = 16,
        tail_latency_guardrail_ms: float = 2500.0,
        per_node_vram_safety_margin: float = 0.9,
        max_retry_attempts: int = 2,
        retry_backoff_ms: int = 25,
        scheduler_tick_ms: int = 10,
        max_dispatch_per_tick: int = 8,
        node_max_concurrent_lanes: int = 4,
        on_metrics: Optional[Callable[[int, int], None]] = None,
    ):
        self.registry = registry
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.max_queue_size = max(0, int(max_queue_size))
        self.scheduler_policy = scheduler_policy or "balanced"
        self.fairness_quantum_tokens = max(1, int(fairness_quantum_tokens))
        self.tail_latency_guardrail_ms = max(0.0, float(tail_latency_guardrail_ms))
        self.per_node_vram_safety_margin = min(
            1.0,
            max(0.0, float(per_node_vram_safety_margin)),
        )
        self.max_retry_attempts = max(0, int(max_retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_ms) / 1000.0)
        self.tick_sec = max(0.001, float(scheduler_tick_ms) / 1000.0)
        self.max_dispatch_per_tick = max(1, int(max_dispatch_per_tick))
        self.node_max_concurrent_lanes = max(1, int(node_max_concurrent_lanes))
        self._on_metrics = on_metrics

        self._accepting = True
        self._active = 0
        self._queued = 0
        self._lane_counter = 0
        self._pending_by_request: dict[str, _PendingRequest] = {}
        self._pending_by_user: dict[str, deque[_PendingRequest]] = {}
        self._user_round_robin: deque[str] = deque()

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="concurrent-request-scheduler",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop scheduler dispatch thread."""
        with self._lock:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    def set_accepting(self, accepting: bool) -> None:
        with self._lock:
            self._accepting = bool(accepting)
            self._cond.notify_all()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._queued

    def acquire(
        self,
        *,
        request_id: str,
        user_id: str,
        execution_plan: ExecutionPlan,
        cancel_event: threading.Event,
    ) -> SchedulerTicket:
        """Queue request and block until scheduler dispatches it."""
        user_bucket = (user_id or request_id or "anonymous").strip() or "anonymous"
        pending = _PendingRequest(
            request_id=request_id,
            user_id=user_bucket,
            execution_plan=execution_plan,
            cancel_event=cancel_event,
        )

        with self._lock:
            if not self._accepting:
                raise RuntimeError("Coordinator is rebalancing; retry shortly")
            if request_id in self._pending_by_request:
                raise RuntimeError(f"request_id '{request_id}' is already queued")

            if self._active >= self.max_concurrent_requests and self._queued >= self.max_queue_size:
                raise RuntimeError(
                    f"Coordinator overloaded: queue full, retry_after_ms={int(self.retry_backoff_sec * 1000)}"
                )

            self._enqueue_pending(pending)
            self._emit_metrics_locked()
            self._cond.notify_all()

            while True:
                if pending.granted:
                    return SchedulerTicket(
                        request_id=request_id,
                        lane_id=pending.lane_id,
                        queue_wait_ms=pending.queue_wait_ms,
                        scheduler_retries=pending.dispatch_retries,
                    )
                if pending.error:
                    raise RuntimeError(pending.error)
                if cancel_event.is_set():
                    self._drop_pending_locked(
                        pending,
                        error=f"request {request_id} cancelled before dispatch",
                    )
                    raise RuntimeError(
                        f"request {request_id} cancelled before dispatch"
                    )
                self._cond.wait(timeout=self.tick_sec)

    def release(self, request_id: str) -> None:
        """Release one active scheduler slot for a completed request."""
        with self._lock:
            self._active = max(self._active - 1, 0)
            self._emit_metrics_locked()
            self._cond.notify_all()
        log.debug(f"Released scheduler slot for request {request_id}")

    def _enqueue_pending(self, pending: _PendingRequest) -> None:
        self._pending_by_request[pending.request_id] = pending
        queue = self._pending_by_user.get(pending.user_id)
        if queue is None:
            queue = deque()
            self._pending_by_user[pending.user_id] = queue
            self._user_round_robin.append(pending.user_id)
        queue.append(pending)
        self._queued += 1

    def _drop_pending_locked(
        self,
        pending: _PendingRequest,
        *,
        error: str = "",
    ) -> None:
        self._remove_pending_locked(pending)
        if error:
            pending.error = error
        self._emit_metrics_locked()
        self._cond.notify_all()

    def _remove_pending_locked(self, pending: _PendingRequest) -> None:
        if pending.request_id not in self._pending_by_request:
            return
        self._pending_by_request.pop(pending.request_id, None)
        self._queued = max(self._queued - 1, 0)

        queue = self._pending_by_user.get(pending.user_id)
        if queue is not None:
            try:
                queue.remove(pending)
            except ValueError:
                pass
            if not queue:
                self._pending_by_user.pop(pending.user_id, None)
                self._user_round_robin = deque(
                    uid for uid in self._user_round_robin if uid != pending.user_id
                )

    def _dispatch_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return

                dispatched = 0
                now = time.time()
                while (
                    self._accepting
                    and self._active < self.max_concurrent_requests
                    and self._queued > 0
                    and dispatched < self.max_dispatch_per_tick
                ):
                    pending = self._select_next_pending_locked(now=now)
                    if pending is None:
                        break

                    if pending.cancel_event.is_set():
                        self._drop_pending_locked(
                            pending,
                            error=f"request {pending.request_id} cancelled before dispatch",
                        )
                        continue

                    if not self._capacity_available_locked(pending.execution_plan):
                        pending.dispatch_retries += 1
                        if pending.dispatch_retries > self.max_retry_attempts:
                            self._drop_pending_locked(
                                pending,
                                error=(
                                    "Coordinator overloaded: insufficient node capacity, "
                                    f"retry_after_ms={int(self.retry_backoff_sec * 1000)}"
                                ),
                            )
                        else:
                            pending.next_eligible_at = now + self.retry_backoff_sec
                        continue

                    self._remove_pending_locked(pending)
                    self._active += 1
                    self._lane_counter += 1
                    pending.lane_id = self._lane_counter
                    pending.queue_wait_ms = (time.time() - pending.queued_at) * 1000
                    pending.granted = True
                    self._emit_metrics_locked()
                    self._cond.notify_all()
                    dispatched += 1

                self._cond.wait(timeout=self.tick_sec if dispatched == 0 else 0.0)

    def _select_next_pending_locked(self, now: float) -> Optional[_PendingRequest]:
        if self._queued <= 0:
            return None

        if self.tail_latency_guardrail_ms > 0:
            aged_threshold = self.tail_latency_guardrail_ms / 1000.0
            aged_candidate = None
            aged_ts = None
            for pending in self._pending_by_request.values():
                if pending.next_eligible_at > now:
                    continue
                wait_sec = now - pending.queued_at
                if wait_sec < aged_threshold:
                    continue
                if aged_ts is None or pending.queued_at < aged_ts:
                    aged_ts = pending.queued_at
                    aged_candidate = pending
            if aged_candidate is not None:
                return aged_candidate

        if not self._user_round_robin:
            return None

        max_scan = max(len(self._user_round_robin), self.fairness_quantum_tokens)
        for _ in range(max_scan):
            if not self._user_round_robin:
                return None
            user_id = self._user_round_robin[0]
            self._user_round_robin.rotate(-1)

            queue = self._pending_by_user.get(user_id)
            if not queue:
                continue
            for pending in queue:
                if pending.next_eligible_at <= now:
                    return pending
        return None

    def _capacity_available_locked(self, execution_plan: ExecutionPlan) -> bool:
        for stage in execution_plan.stages:
            node = self.registry.get_node(stage.node_id)
            if node is None:
                return False

            lane_cap = self.node_max_concurrent_lanes
            if node.active_requests >= lane_cap:
                return False

            reserve_fraction = max(0.0, 1.0 - self.per_node_vram_safety_margin)
            reserve_mb = node.vram_mb * reserve_fraction
            if node.estimated_free_vram_mb > 0 and node.estimated_free_vram_mb < reserve_mb:
                return False
        return True

    def _emit_metrics_locked(self) -> None:
        # Intentionally avoid callback invocation while scheduler lock is held.
        # The orchestrator queries active/queue counters directly when needed.
        return
