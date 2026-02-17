"""High-level distributed inference pipeline.

Wraps the orchestrator to provide a simple API for running inference
across the distributed node network.
"""

from typing import Optional

from distributed_inference.common.config import SystemConfig, load_config
from distributed_inference.common.logging import get_logger
from distributed_inference.coordinator.orchestrator import Orchestrator

log = get_logger(__name__)


class DistributedInferencePipeline:
    """High-level API for distributed inference.

    Wraps the Orchestrator and provides a simplified interface for
    setting up the distributed system and running inference.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or load_config()
        self.orchestrator = Orchestrator(config=self.config)
        self._started = False

    def setup(self, min_nodes: int = 1, wait_timeout: float = 60.0) -> bool:
        """Set up the distributed inference system.

        Starts the coordinator, waits for nodes, partitions the model,
        and loads shards onto nodes.

        Args:
            min_nodes: Minimum number of nodes needed before proceeding.
            wait_timeout: How long to wait for nodes (seconds).

        Returns:
            True if setup completed successfully.
        """
        log.info("[bold]Setting up distributed inference pipeline[/]")

        # Start coordinator (non-blocking)
        self.orchestrator.start(block=False)
        self._started = True

        # Wait for nodes
        if not self.orchestrator.wait_for_nodes(min_nodes, timeout=wait_timeout):
            log.error("Not enough nodes registered")
            return False

        # Partition and load model
        self.orchestrator.setup_model()

        log.info("[bold green]Pipeline ready for inference![/]")
        return True

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.

        Returns:
            Dict with 'text', 'tokens_generated', 'latency_ms',
            'tokens_per_second', and 'per_hop_latency_ms'.
        """
        if not self.orchestrator.is_ready:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        result = self.orchestrator.run_inference(
            prompt=prompt,
            max_tokens=max_tokens or self.config.inference.max_tokens,
            temperature=temperature if temperature is not None else self.config.inference.temperature,
            top_p=top_p if top_p is not None else self.config.inference.top_p,
            top_k=top_k if top_k is not None else self.config.inference.top_k,
        )

        return {
            "text": result.generated_text,
            "tokens_generated": result.tokens_generated,
            "latency_ms": result.total_latency_ms,
            "tokens_per_second": result.tokens_per_second,
            "per_hop_latency_ms": list(result.per_hop_latency_ms),
        }

    def shutdown(self) -> None:
        """Shut down the pipeline and coordinator."""
        if self._started:
            self.orchestrator.stop()
            self._started = False
            log.info("Pipeline shut down")
