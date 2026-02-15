"""Benchmark KV cache on/off across repeated distributed inference runs.

Runs two benchmark phases:
1. KV cache enabled
2. KV cache disabled

Each phase spins up coordinator + node processes, runs N inference requests,
and reports per-run and aggregate latency/throughput.

Usage:
    python scripts/benchmark_kv_cache.py
    python scripts/benchmark_kv_cache.py --num-runs 4 --max-tokens 300
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from distributed_inference.common.config import load_config
from distributed_inference.common.logging import get_logger, setup_logging
from distributed_inference.coordinator.orchestrator import Orchestrator

log = get_logger(__name__)


@dataclass
class RunMetrics:
    run_idx: int
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float


def _start_nodes(
    *,
    num_nodes: int,
    coordinator_addr: str,
    coordinator_port: int,
    device: str,
    log_level: str,
) -> list[subprocess.Popen]:
    vram_caps = {
        1: [2048],
        2: [1024, 1024],
        3: [512, 1024, 1536],
        4: [512, 512, 1024, 1024],
    }
    node_vrams = vram_caps.get(num_nodes, [1024] * num_nodes)
    node_base_port = coordinator_port + 1
    procs: list[subprocess.Popen] = []

    for i in range(num_nodes):
        port = node_base_port + i
        vram = node_vrams[i]
        node_id = f"node-{i+1}"

        node_cmd = [
            sys.executable, "-m", "distributed_inference.cli.start_node",
            "--port", str(port),
            "--coordinator", coordinator_addr,
            "--max-vram-mb", str(vram),
            "--device", device,
            "--node-id", node_id,
            "--log-level", log_level,
        ]
        proc = subprocess.Popen(
            node_cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
        )
        procs.append(proc)
        time.sleep(1.0)

    return procs


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()


def _run_phase(
    *,
    enable_kv_cache: bool,
    num_runs: int,
    num_nodes: int,
    prompt: str,
    max_tokens: int,
    device: str,
    coordinator_port: int,
    startup_timeout: float,
    log_level: str,
) -> list[RunMetrics]:
    phase_name = "KV-CACHE-ON" if enable_kv_cache else "KV-CACHE-OFF"
    coordinator_addr = f"localhost:{coordinator_port}"
    orchestrator = None
    node_procs: list[subprocess.Popen] = []
    metrics: list[RunMetrics] = []

    log.info(f"\n[bold blue]=== {phase_name} ===[/]")

    try:
        config = load_config()
        config.coordinator.port = coordinator_port
        config.inference.enable_kv_cache = enable_kv_cache

        orchestrator = Orchestrator(config=config)
        orchestrator.start(block=False)
        time.sleep(1.0)

        node_procs = _start_nodes(
            num_nodes=num_nodes,
            coordinator_addr=coordinator_addr,
            coordinator_port=coordinator_port,
            device=device,
            log_level=log_level,
        )

        if not orchestrator.wait_for_nodes(num_nodes, timeout=startup_timeout):
            raise RuntimeError("Not enough nodes registered")

        orchestrator.setup_model()

        for run_idx in range(1, num_runs + 1):
            req_id = f"{phase_name.lower()}-{uuid.uuid4().hex[:8]}"
            result = orchestrator.run_inference(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                request_id=req_id,
            )

            run = RunMetrics(
                run_idx=run_idx,
                tokens_generated=result.tokens_generated,
                latency_ms=result.total_latency_ms,
                tokens_per_second=result.tokens_per_second,
            )
            metrics.append(run)
            log.info(
                f"{phase_name} run {run_idx}/{num_runs}: "
                f"tokens={run.tokens_generated}, "
                f"latency={run.latency_ms:.1f}ms, "
                f"throughput={run.tokens_per_second:.2f} tok/s"
            )

    finally:
        if orchestrator is not None:
            try:
                orchestrator.stop()
            except Exception:
                pass
        _stop_processes(node_procs)

    return metrics


def _summarize(name: str, runs: list[RunMetrics]) -> dict:
    latencies = [r.latency_ms for r in runs]
    tps = [r.tokens_per_second for r in runs]
    tokens = [r.tokens_generated for r in runs]
    return {
        "name": name,
        "runs": len(runs),
        "avg_tokens": statistics.mean(tokens) if tokens else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "avg_tps": statistics.mean(tps) if tps else 0.0,
        "median_tps": statistics.median(tps) if tps else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed KV cache")
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of AI is",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--coordinator-port-base", type=int, default=50050)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, component="kv-bench")

    on_runs = _run_phase(
        enable_kv_cache=True,
        num_runs=args.num_runs,
        num_nodes=args.num_nodes,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        device=args.device,
        coordinator_port=args.coordinator_port_base,
        startup_timeout=args.startup_timeout,
        log_level=args.log_level,
    )
    off_runs = _run_phase(
        enable_kv_cache=False,
        num_runs=args.num_runs,
        num_nodes=args.num_nodes,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        device=args.device,
        coordinator_port=args.coordinator_port_base + 100,
        startup_timeout=args.startup_timeout,
        log_level=args.log_level,
    )

    on_summary = _summarize("KV cache ON", on_runs)
    off_summary = _summarize("KV cache OFF", off_runs)

    latency_speedup = (
        off_summary["avg_latency_ms"] / on_summary["avg_latency_ms"]
        if on_summary["avg_latency_ms"] > 0
        else 0.0
    )
    tps_speedup = (
        on_summary["avg_tps"] / off_summary["avg_tps"]
        if off_summary["avg_tps"] > 0
        else 0.0
    )

    print("\n" + "=" * 72)
    print("KV CACHE BENCHMARK SUMMARY")
    print("=" * 72)
    print(
        f"{on_summary['name']}:  "
        f"avg latency={on_summary['avg_latency_ms']:.1f}ms, "
        f"avg tok/s={on_summary['avg_tps']:.2f}"
    )
    print(
        f"{off_summary['name']}: "
        f"avg latency={off_summary['avg_latency_ms']:.1f}ms, "
        f"avg tok/s={off_summary['avg_tps']:.2f}"
    )
    print("-" * 72)
    print(f"Latency improvement (OFF/ON): {latency_speedup:.2f}x")
    print(f"Throughput improvement (ON/OFF): {tps_speedup:.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
