"""Heartbeat-based health monitor for distributed nodes.

Runs as a background service, periodically checking node health
via gRPC heartbeats and emitting events on state changes.
"""

import threading
import time
from enum import Enum
from typing import Callable, List, Optional

from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.registry import NodeRegistry, NodeState
from distributed_inference.coordinator.router import ActivationRouter

log = get_logger(__name__)


class HealthEvent(Enum):
    """Health monitoring events."""
    NODE_HEALTHY = "node_healthy"
    NODE_SUSPECT = "node_suspect"
    NODE_DEAD = "node_dead"
    NODE_RECOVERED = "node_recovered"


class HealthMonitor:
    """Monitors node health via periodic heartbeat checks.

    Sends heartbeat RPCs to all registered nodes and tracks response
    times. Emits events when nodes become suspect or dead.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        router: ActivationRouter,
        check_interval_sec: float = 5.0,
        timeout_sec: float = 15.0,
        failure_threshold: int = 3,
    ):
        self.registry = registry
        self.router = router
        self.check_interval = check_interval_sec
        self.timeout = timeout_sec
        self.failure_threshold = failure_threshold

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

    def start(self) -> None:
        """Start the health monitor background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor",
        )
        self._thread.start()
        log.info("Health monitor started")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Health monitor stopped")

    def on_event(self, callback: Callable) -> None:
        """Register a callback for health events.

        Callback signature: callback(event: HealthEvent, node_id: str)
        """
        self._callbacks.append(callback)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            self._check_all_nodes()
            time.sleep(self.check_interval)

    def _check_all_nodes(self) -> None:
        """Check health of all registered nodes."""
        for node in self.registry.get_all_nodes():
            if node.state == NodeState.DEAD:
                continue

            status = self.router.check_node_health(node.address)

            if status is not None:
                # Node is reachable
                was_suspect = node.state == NodeState.SUSPECT
                self.registry.update_heartbeat(node.node_id)

                if was_suspect:
                    self._emit(HealthEvent.NODE_RECOVERED, node.node_id)
            else:
                # Node is unreachable
                elapsed = time.time() - node.last_heartbeat
                if elapsed > self.timeout:
                    node.missed_heartbeats += 1

                    if node.missed_heartbeats >= self.failure_threshold:
                        self.registry.mark_dead(node.node_id)
                        self._emit(HealthEvent.NODE_DEAD, node.node_id)
                    else:
                        self.registry.mark_suspect(node.node_id)
                        self._emit(HealthEvent.NODE_SUSPECT, node.node_id)

    def _emit(self, event: HealthEvent, node_id: str) -> None:
        """Emit a health event to all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event, node_id)
            except Exception as e:
                log.error(f"Health event callback error: {e}")
