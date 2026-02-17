"""Inference profiler for benchmarking distributed pipeline performance.

Measures and reports latency, throughput, and communication overhead
for distributed inference runs.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class HopMetrics:
    """Metrics for a single hop in the pipeline."""
    stage_index: int
    node_id: str
    latency_ms: float
    compute_ms: float = 0.0
    serialization_ms: float = 0.0
    network_ms: float = 0.0


@dataclass
class InferenceMetrics:
    """Complete metrics for one inference run."""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    per_token_latency_ms: float
    tokens_per_second: float
    hop_metrics: List[HopMetrics] = field(default_factory=list)
    total_compute_ms: float = 0.0
    total_communication_ms: float = 0.0
    compute_fraction: float = 0.0
    communication_fraction: float = 0.0

    def summary(self) -> str:
        lines = [
            f"═══ Inference Metrics: {self.request_id} ═══",
            f"  Prompt tokens:    {self.prompt_tokens}",
            f"  Generated tokens: {self.generated_tokens}",
            f"  Total latency:    {self.total_latency_ms:.1f}ms",
            f"  Per-token:        {self.per_token_latency_ms:.1f}ms",
            f"  Throughput:       {self.tokens_per_second:.2f} tok/s",
            f"  Compute:          {self.total_compute_ms:.1f}ms ({self.compute_fraction*100:.1f}%)",
            f"  Communication:    {self.total_communication_ms:.1f}ms ({self.communication_fraction*100:.1f}%)",
        ]

        if self.hop_metrics:
            lines.append("  ─── Per-Hop Breakdown ───")
            for hop in self.hop_metrics:
                lines.append(
                    f"    Stage {hop.stage_index} ({hop.node_id}): "
                    f"{hop.latency_ms:.1f}ms"
                )

        return "\n".join(lines)


class InferenceProfiler:
    """Collects and analyzes inference performance metrics.

    Wraps inference calls to measure timing at each stage,
    then generates reports comparing different configurations.
    """

    def __init__(self):
        self._runs: List[InferenceMetrics] = []

    def record_run(
        self,
        request_id: str,
        prompt_tokens: int,
        generated_tokens: int,
        total_latency_ms: float,
        per_hop_latencies: List[float],
        node_ids: Optional[List[str]] = None,
    ) -> InferenceMetrics:
        """Record metrics from a completed inference run.

        Args:
            request_id: Unique request identifier.
            prompt_tokens: Number of tokens in the prompt.
            generated_tokens: Number of tokens generated.
            total_latency_ms: Total wall-clock time in ms.
            per_hop_latencies: Latency per pipeline hop in ms.
            node_ids: Node IDs for each hop.

        Returns:
            Computed InferenceMetrics.
        """
        per_token = total_latency_ms / max(generated_tokens, 1)
        tps = generated_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0

        # Build hop metrics
        hops = []
        for i, lat in enumerate(per_hop_latencies):
            nid = node_ids[i % len(node_ids)] if node_ids else f"node-{i}"
            hops.append(HopMetrics(
                stage_index=i,
                node_id=nid,
                latency_ms=lat,
            ))

        # Estimate compute vs communication
        # In pipeline execution, total = sum of all hops
        # Within each hop: compute + serialization + network
        # For now, we approximate compute as 70% and comms as 30% per hop
        total_hop_time = sum(per_hop_latencies)
        # A rough decomposition - in real-world, we'd instrument more precisely
        estimated_compute = total_hop_time * 0.7
        estimated_comms = total_hop_time * 0.3

        metrics = InferenceMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_latency_ms=total_latency_ms,
            per_token_latency_ms=per_token,
            tokens_per_second=tps,
            hop_metrics=hops,
            total_compute_ms=estimated_compute,
            total_communication_ms=estimated_comms,
            compute_fraction=estimated_compute / max(total_latency_ms, 1),
            communication_fraction=estimated_comms / max(total_latency_ms, 1),
        )

        self._runs.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        """Get aggregate summary across all recorded runs.

        Returns:
            Dict with average/min/max metrics.
        """
        if not self._runs:
            return {"runs": 0}

        latencies = [r.total_latency_ms for r in self._runs]
        tps_values = [r.tokens_per_second for r in self._runs]

        return {
            "runs": len(self._runs),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_tokens_per_second": sum(tps_values) / len(tps_values),
            "total_tokens_generated": sum(r.generated_tokens for r in self._runs),
        }

    def export_json(self, filepath: str) -> None:
        """Export all metrics as a JSON file.

        Args:
            filepath: Path to write the JSON report.
        """
        data = {
            "summary": self.get_summary(),
            "runs": [
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                    "total_latency_ms": r.total_latency_ms,
                    "per_token_latency_ms": r.per_token_latency_ms,
                    "tokens_per_second": r.tokens_per_second,
                    "compute_fraction": r.compute_fraction,
                    "communication_fraction": r.communication_fraction,
                }
                for r in self._runs
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        log.info(f"Metrics exported to {filepath}")

    def print_comparison(
        self,
        single_node_latency_ms: Optional[float] = None,
    ) -> str:
        """Print a comparison report with optional single-node baseline.

        Args:
            single_node_latency_ms: Optional baseline latency.

        Returns:
            Formatted comparison string.
        """
        summary = self.get_summary()
        lines = [
            "╔══════════════════════════════════════╗",
            "║   Distributed Inference Benchmark    ║",
            "╠══════════════════════════════════════╣",
            f"║ Runs:              {summary['runs']:>16} ║",
            f"║ Avg Latency:   {summary.get('avg_latency_ms', 0):>12.1f}ms ║",
            f"║ Min Latency:   {summary.get('min_latency_ms', 0):>12.1f}ms ║",
            f"║ Max Latency:   {summary.get('max_latency_ms', 0):>12.1f}ms ║",
            f"║ Avg Tok/s:     {summary.get('avg_tokens_per_second', 0):>12.2f}   ║",
        ]

        if single_node_latency_ms:
            avg = summary.get('avg_latency_ms', 0)
            overhead = ((avg / single_node_latency_ms) - 1) * 100
            lines.append(f"║ Baseline:      {single_node_latency_ms:>12.1f}ms ║")
            lines.append(f"║ Overhead:      {overhead:>11.1f}%   ║")

        lines.append("╚══════════════════════════════════════╝")

        report = "\n".join(lines)
        log.info(f"\n{report}")
        return report
