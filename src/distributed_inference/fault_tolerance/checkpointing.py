"""Activation checkpointing for fault tolerance.

Saves intermediate activations at each pipeline hop so inference
can be resumed from the last checkpoint if a node fails.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class Checkpoint:
    """A saved intermediate activation state."""
    request_id: str
    stage_index: int
    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    timestamp: float = field(default_factory=time.time)

    @property
    def size_mb(self) -> float:
        """Estimated size of this checkpoint in MB."""
        total = self.hidden_states.nelement() * self.hidden_states.element_size()
        if self.attention_mask is not None:
            total += self.attention_mask.nelement() * self.attention_mask.element_size()
        if self.position_ids is not None:
            total += self.position_ids.nelement() * self.position_ids.element_size()
        return total / (1024 * 1024)


class CheckpointManager:
    """Manages activation checkpoints for fault-tolerant inference.

    Stores checkpoints in memory for active requests. Old checkpoints
    are automatically cleaned up.
    """

    def __init__(self, max_checkpoints_per_request: int = 10):
        self._checkpoints: Dict[str, Dict[int, Checkpoint]] = {}
        self._max_per_request = max_checkpoints_per_request

    def save(
        self,
        request_id: str,
        stage_index: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Save a checkpoint for a request at a specific pipeline stage.

        Args:
            request_id: Unique request identifier.
            stage_index: Pipeline stage index (0-based).
            hidden_states: Current hidden states tensor.
            attention_mask: Current attention mask.
            position_ids: Current position IDs.
        """
        if request_id not in self._checkpoints:
            self._checkpoints[request_id] = {}

        # Clone tensors to avoid mutation
        checkpoint = Checkpoint(
            request_id=request_id,
            stage_index=stage_index,
            hidden_states=hidden_states.clone().cpu(),
            attention_mask=attention_mask.clone().cpu() if attention_mask is not None else None,
            position_ids=position_ids.clone().cpu() if position_ids is not None else None,
        )

        self._checkpoints[request_id][stage_index] = checkpoint

        # Prune old stages
        stages = self._checkpoints[request_id]
        if len(stages) > self._max_per_request:
            oldest_key = min(stages.keys())
            del stages[oldest_key]

        log.debug(
            f"Checkpoint saved: req={request_id}, stage={stage_index}, "
            f"size={checkpoint.size_mb:.1f}MB"
        )

    def get_latest(self, request_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a request.

        Returns:
            The most recent checkpoint, or None if no checkpoints exist.
        """
        stages = self._checkpoints.get(request_id)
        if not stages:
            return None
        latest_key = max(stages.keys())
        return stages[latest_key]

    def get(self, request_id: str, stage_index: int) -> Optional[Checkpoint]:
        """Get a specific checkpoint.

        Returns:
            The checkpoint at the given stage, or None.
        """
        stages = self._checkpoints.get(request_id)
        if not stages:
            return None
        return stages.get(stage_index)

    def clear_request(self, request_id: str) -> None:
        """Remove all checkpoints for a request."""
        self._checkpoints.pop(request_id, None)

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()

    @property
    def total_checkpoints(self) -> int:
        """Total number of checkpoints across all requests."""
        return sum(len(stages) for stages in self._checkpoints.values())

    @property
    def total_size_mb(self) -> float:
        """Total size of all checkpoints in MB."""
        total = 0.0
        for stages in self._checkpoints.values():
            for cp in stages.values():
                total += cp.size_mb
        return total
