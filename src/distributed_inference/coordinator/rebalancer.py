"""Background rebalance trigger controller."""

import threading
import time
from typing import Callable, Optional

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


class RebalanceController:
    """Coalesces topology events and invokes rebalance with cooldown."""

    def __init__(
        self,
        callback: Callable[[str], None],
        cooldown_sec: float = 5.0,
    ):
        self._callback = callback
        self._cooldown_sec = max(0.0, cooldown_sec)
        self._lock = threading.RLock()
        self._pending_reason: Optional[str] = None
        self._last_run_ts = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background rebalance monitor thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="rebalance-controller",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop background thread."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def request(self, reason: str) -> None:
        """Queue a rebalance request."""
        with self._lock:
            self._pending_reason = reason

    def run_now(self, reason: str) -> None:
        """Run rebalance immediately, bypassing cooldown."""
        self._callback(reason)
        with self._lock:
            self._last_run_ts = time.time()

    def _loop(self) -> None:
        while True:
            with self._lock:
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
                self._invoke(reason=pending_reason)
            time.sleep(0.2)

    def _invoke(self, reason: str) -> None:
        try:
            self._callback(reason)
        except Exception as e:  # pragma: no cover - defensive.
            log.exception(f"Rebalance callback failed ({reason}): {e}")
        finally:
            with self._lock:
                self._last_run_ts = time.time()
