"""Node registry for tracking registered nodes and their status.

Thread-safe registry that maintains node information, assigned layers,
and health status. Used by the coordinator to manage the node pool.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


class NodeState(Enum):
    """Node lifecycle states."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class RegisteredNode:
    """Information about a registered node."""
    node_id: str
    address: str
    vram_mb: int
    compute_tflops: float
    bandwidth_mbps: float
    device_type: str
    device_name: str
    sram_mb: int = 0
    latency_ms: float = 0.0
    effective_bandwidth_mbps: float = 0.0
    admitted: bool = True
    admission_reason: str = ""

    # Mutable state
    state: NodeState = NodeState.IDLE
    assigned_layers: Tuple[int, int] = (0, 0)  # (start, end)
    has_embedding: bool = False
    has_lm_head: bool = False
    last_heartbeat: float = 0.0
    missed_heartbeats: int = 0
    vram_used_mb: int = 0
    active_requests: int = 0
    queue_depth: int = 0
    estimated_free_vram_mb: int = 0
    registered_at: float = field(default_factory=time.time)


class NodeRegistry:
    """Thread-safe registry of distributed inference nodes.

    Provides methods to register/unregister nodes, update their status,
    and query for available nodes.
    """

    def __init__(self):
        self._nodes: Dict[str, RegisteredNode] = {}
        self._lock = threading.RLock()
        self._on_change_callbacks = []

    def register(
        self,
        node_id: str,
        address: str,
        vram_mb: int,
        compute_tflops: float = 0.0,
        bandwidth_mbps: float = 1000.0,
        device_type: str = "cpu",
        device_name: str = "",
        sram_mb: int = 0,
        latency_ms: float = 0.0,
        effective_bandwidth_mbps: float = 0.0,
        admitted: bool = True,
        admission_reason: str = "",
    ) -> RegisteredNode:
        """Register a new node or update an existing one.

        Args:
            node_id: Unique node identifier.
            address: Network address (host:port).
            vram_mb: Available VRAM in MB.
            compute_tflops: Compute capability in TFLOPS.
            bandwidth_mbps: Network bandwidth in Mbps.
            device_type: "cuda" or "cpu".
            device_name: Human-readable device name.

        Returns:
            The registered node info.
        """
        with self._lock:
            node = RegisteredNode(
                node_id=node_id,
                address=address,
                vram_mb=vram_mb,
                compute_tflops=compute_tflops,
                bandwidth_mbps=bandwidth_mbps,
                device_type=device_type,
                device_name=device_name,
                sram_mb=sram_mb,
                latency_ms=latency_ms,
                effective_bandwidth_mbps=effective_bandwidth_mbps,
                admitted=admitted,
                admission_reason=admission_reason,
                last_heartbeat=time.time(),
            )
            self._nodes[node_id] = node

            admission_suffix = "admitted" if admitted else f"rejected: {admission_reason}"
            log.info(
                f"[bold green]Node registered:[/] {node_id} "
                f"at {address} ({vram_mb}MB VRAM, {device_type}, {admission_suffix})"
            )
            self._notify_change("register", node)
            return node

    def unregister(self, node_id: str) -> Optional[RegisteredNode]:
        """Remove a node from the registry.

        Returns:
            The removed node info, or None if not found.
        """
        with self._lock:
            node = self._nodes.pop(node_id, None)
            if node:
                log.info(f"Node unregistered: {node_id}")
                self._notify_change("unregister", node)
            return node

    def get_node(self, node_id: str) -> Optional[RegisteredNode]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[RegisteredNode]:
        """Get all registered nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_active_nodes(self) -> List[RegisteredNode]:
        """Get all nodes that are not DEAD or SUSPECT."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.state not in (NodeState.DEAD, NodeState.SUSPECT)
            ]

    def get_ready_nodes(self) -> List[RegisteredNode]:
        """Get all nodes in READY state."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.state == NodeState.READY
            ]

    def get_admitted_active_nodes(self) -> List[RegisteredNode]:
        """Get active nodes that are admitted for hosting model shards."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.admitted and n.state not in (NodeState.DEAD, NodeState.SUSPECT)
            ]

    def update_heartbeat(
        self,
        node_id: str,
        vram_used_mb: int = 0,
        active_requests: int = 0,
        queue_depth: int = 0,
        estimated_free_vram_mb: int = 0,
    ) -> None:
        """Update the last heartbeat timestamp for a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_heartbeat = time.time()
                node.missed_heartbeats = 0
                node.vram_used_mb = vram_used_mb
                node.active_requests = active_requests
                node.queue_depth = queue_depth
                node.estimated_free_vram_mb = estimated_free_vram_mb
                if node.state in (NodeState.SUSPECT, NodeState.DEAD):
                    previous_state = node.state
                    node.state = NodeState.READY
                    log.info(
                        f"Node {node_id} recovered from {previous_state.value} state"
                    )

    def increment_missed_heartbeats(self, node_id: str) -> int:
        """Increment heartbeat miss counter and return the updated value."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return 0
            if node.state == NodeState.DEAD:
                return node.missed_heartbeats
            node.missed_heartbeats += 1
            return node.missed_heartbeats

    def mark_suspect(self, node_id: str) -> None:
        """Mark a node as suspect (missing heartbeats)."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node and node.state not in (NodeState.DEAD,):
                node.state = NodeState.SUSPECT
                log.warning(
                    f"Node {node_id} marked SUSPECT "
                    f"(missed: {node.missed_heartbeats})"
                )

    def mark_dead(self, node_id: str) -> Optional[RegisteredNode]:
        """Mark a node as dead."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.state = NodeState.DEAD
                log.error(f"[bold red]Node {node_id} marked DEAD[/]")
                self._notify_change("dead", node)
            return node

    def set_node_assignment(
        self,
        node_id: str,
        start_layer: int,
        end_layer: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
    ) -> None:
        """Record which layers are assigned to a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.assigned_layers = (start_layer, end_layer)
                node.has_embedding = has_embedding
                node.has_lm_head = has_lm_head
                node.state = NodeState.READY

    def set_node_state(self, node_id: str, state: NodeState) -> None:
        """Set the state of a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.state = state

    def on_change(self, callback) -> None:
        """Register a callback for registry changes.

        Callback signature: callback(event: str, node: RegisteredNode)
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self, event: str, node: RegisteredNode) -> None:
        """Notify registered callbacks of a change."""
        for cb in self._on_change_callbacks:
            try:
                cb(event, node)
            except Exception as e:
                log.error(f"Registry callback error: {e}")

    @property
    def node_count(self) -> int:
        """Number of registered nodes (including dead)."""
        return len(self._nodes)

    @property
    def active_count(self) -> int:
        """Number of active (non-dead) nodes."""
        return len(self.get_active_nodes())

    def summary(self) -> str:
        """Return a summary of the registry state."""
        with self._lock:
            states = {}
            for n in self._nodes.values():
                states[n.state.value] = states.get(n.state.value, 0) + 1
            parts = [f"{v} {k}" for k, v in states.items()]
            return f"Nodes: {self.node_count} ({', '.join(parts) if parts else 'none'})"
